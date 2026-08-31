#!/usr/bin/env python3
"""Milestone 5 — 品質（ε）。目前整個專案最大的洞。

## 為什麼這是洞

M3 量完了兩個模型 × 五個 baseline 的延遲，但每一列的 `quality_score` 都是
`NOT_MEASURED`。也就是說：

> 我可以說 `cpu_lru` 的 warm TTFT 快 92%，
> **但完全不知道它有沒有把答案弄壞。**

而論文的標題就是「**以品質為約束**的 KV state 管理」，式(2) 的 ε 是硬約束。
ε 現在是空的，表 8 只有一半。

## 動作空間分成兩類，量法不同

| 類別 | 動作 | 預期 ε | 這支腳本怎麼驗 |
|---|---|---|---|
| **無損** | GPU-BF16 / CPU / SSD | **必須 = 0** | 逐字元比對輸出是否**完全相同** |
| **有損** | GPU-FP8 / INT8 / INT4 | > 0，待量 | GSM8K 正確率相對 BF16 的掉幅 |
| 無損但貴 | DROP + 重算 | ≈ 0 | 同「無損」欄 |

**真正需要量的 ε 在精度階梯上**——CPU/SSD 只是把位元組搬來搬去，
若輸出不同就是 bug 而不是「品質取捨」。這支腳本把兩件事分開驗。

## 工作負載：many-shot GSM8K

GSM8K 單題只有 ~60–100 個 token。**在短 context 下 KV 根本不會有壓力，
量不到任何東西。** 所以用 **many-shot**：

```
[K 個 GSM8K 範例 ≈ 10–20K token] ← 共用前綴，會被快取／卸載／量化
[第 i 題]                        ← 每題不同
```

這個設計有三個好處：
1. 共用前綴夠長，真的會進卸載路徑
2. 推理**依賴**那個前綴，前綴被弄壞會直接反映在正確率上
3. 這是真實的 serving 樣態（共用 system prompt / few-shot 前綴）

文獻支持：Ła´ncucki et al. 的 KV cache 壓縮研究發現
「延遲逐出能保住推理能力，立即逐出會在 GSM8K 上快速崩潰」——
正是本文 `DROP` 動作可能造成的失效模式。

## 用法

    python code/m5_quality.py --gpu 0 --mode precision   # 精度階梯的 ε
    python code/m5_quality.py --gpu 0 --mode lossless    # 驗證卸載真的無損
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_guard import GpuWatcher, host_contention, wait_until_free  # noqa: E402

BIG = Path(os.environ.get("PAPER_HKV_BIG", "/ssd7/hungwei/paper-hkv"))
VENV = BIG / "venv/vllm"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results/m5_quality"
GSM = BIG / "datasets/gsm8k"

MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"
MODEL_KEY = "llama"

# 精度階梯：論文動作空間裡「住在 GPU 上」的四階。
PRECISIONS = [
    ("bf16", "auto", "無損基準"),
    ("fp8", "fp8", "純格式轉換，無額外中繼資料"),
    ("int8", "int8_per_token_head", "per-token-head 量化"),
    ("int4", "int4_per_token_head", "per-token-head 量化，最低精度階"),
]

# ── 部分量化：品質 ↔ 容量的取捨曲線 ──────────────────────────────
#
# 精度與 CPU/SSD/DROP 的性質不同：後者換的是**時間**，精度換的是
# **容量 ↔ 品質**。所以它不該進 Oracle 的動作空間，而是外層的旋鈕：
#
#   給定量化比例 f → 容量放大 m(f)（已量到）→ 用 gpu_blocks × m(f) 跑 Oracle
#                  → 品質損失 ε(f)（本實驗要量）
#   兩者合起來畫出 (ε, T) 的 Pareto 曲線，即式 (eq:opt) 的可行前緣。
#
# vLLM 用 --kv-cache-dtype-skip-layers 支援**逐層**混合精度
# （欄位定義見 vllm/config/cache.py:114）。Llama-3.1-8B 有 32 層，
# 故 f 的粒度是 1/32。這與 KVTuner（ICML'25）的逐層混合精度做法一致。
N_LAYERS = 32

def mixed_precision_configs(dtype: str, fractions: list[float]) -> list[tuple]:
    """回傳 (名稱, kv_dtype, 額外旗標, 說明) 的清單。

    f = 被量化的層數比例。skip 清單裡的層維持 BF16。
    為使被量化的層分散於整個網路（而非集中在頭或尾），採等間距取樣——
    若集中在前段，量到的會是「淺層對量化的敏感度」而非「整體 f 的效果」。
    """
    out = []
    for f in fractions:
        n_q = round(f * N_LAYERS)
        if n_q == 0:
            out.append((f"f0.00", "auto", [], "全 BF16（基準）"))
            continue
        # 等間距選出要量化的層，其餘進 skip 清單
        q = {round(i * N_LAYERS / n_q) % N_LAYERS for i in range(n_q)}
        skip = [str(i) for i in range(N_LAYERS) if i not in q]
        extra = ["--kv-cache-dtype", dtype]
        if skip:
            extra += ["--kv-cache-dtype-skip-layers", ",".join(skip)]
        out.append((f"f{f:.2f}", dtype, extra,
                    f"{n_q}/{N_LAYERS} 層量化為 {dtype}"))
    return out

CPU_BYTES = 24 * 1024**3
FS_ROOT = BIG / "kv_fs_tier_q"

# 無損驗證：這三個設定的輸出必須**完全相同**，不同就是 bug。
LOSSLESS = [
    ("full_gpu", None, "不卸載"),
    ("cpu_lru", {"kv_connector": "OffloadingConnector", "kv_role": "kv_both",
                 "kv_connector_extra_config": {
                     "spec_name": "CPUOffloadingSpec",
                     "cpu_bytes_to_use": CPU_BYTES, "eviction_policy": "lru"}},
     "CPU 階：位元組搬移，應無損"),
    ("tier_fs", {"kv_connector": "OffloadingConnector", "kv_role": "kv_both",
                 "kv_connector_extra_config": {
                     "spec_name": "TieringOffloadingSpec",
                     "cpu_bytes_to_use": 1 * 1024**3, "eviction_policy": "lru",
                     "secondary_tiers": [{"type": "fs", "root_dir": str(FS_ROOT)}]}},
     "CPU 階縮小以強迫 cascade 到磁碟"),
]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def load_gsm8k(split: str) -> list[dict]:
    p = GSM / f"{split}.jsonl"
    if not p.exists():
        raise SystemExit(f"🔴 找不到 {p}。先下載 GSM8K。")
    return [json.loads(l) for l in p.open()]


def gold(answer: str) -> str:
    """GSM8K 的標準答案在 '#### ' 之後。"""
    return answer.split("####")[-1].strip().replace(",", "")


NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract(text: str) -> str | None:
    """取模型輸出裡的最後一個數字當作答案（GSM8K 的慣例做法）。"""
    hits = NUM.findall(text.replace("$", ""))
    if not hits:
        return None
    return hits[-1].replace(",", "").rstrip(".")


def build_prefix(train: list[dict], k: int) -> str:
    """k-shot 的共用前綴。固定取前 k 筆，保證每個設定拿到完全一樣的前綴。"""
    parts = []
    for ex in train[:k]:
        cot = ex["answer"].split("####")[0].strip()
        parts.append(f"Question: {ex['question'].strip()}\n"
                     f"Answer: {cot}\nThe answer is {gold(ex['answer'])}.\n")
    return "\n".join(parts) + "\n"


class Server:
    def __init__(self, gpu: int, max_len: int, out: Path,
                 kv_dtype: str = "auto", kv_cfg: dict | None = None,
                 extra_args: list[str] | None = None):
        self.gpu, self.max_len, self.out = gpu, max_len, out
        self.kv_dtype, self.kv_cfg = kv_dtype, kv_cfg
        self.extra_args = extra_args or []
        self.port = free_port()
        self.p: subprocess.Popen | None = None
        self.kv_tokens = None

    def __enter__(self):
        self.out.mkdir(parents=True, exist_ok=True)
        cmd = [str(VENV / "bin/vllm"), "serve", MODEL, "--port", str(self.port),
               "--max-model-len", str(self.max_len),
               "--gpu-memory-utilization", "0.90"]
        if self.extra_args:
            cmd += self.extra_args           # 已含 --kv-cache-dtype
        elif self.kv_dtype != "auto":
            cmd += ["--kv-cache-dtype", self.kv_dtype]
        if self.kv_cfg:
            cmd += ["--kv-transfer-config", json.dumps(self.kv_cfg)]
        (self.out / "cmd.txt").write_text(" ".join(shlex.quote(c) for c in cmd) + "\n")

        e = dict(os.environ)
        e["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        e["PATH"] = f"{VENV / 'bin'}:{e.get('PATH', '')}"
        e.setdefault("HF_HOME", str(BIG / "hf-cache/huggingface"))
        for k, v in (("XDG_CACHE_HOME", "xdg-cache"), ("TRITON_CACHE_DIR", "triton-cache"),
                     ("VLLM_CACHE_ROOT", "vllm-cache"),
                     ("FLASHINFER_WORKSPACE_BASE", "flashinfer-cache")):
            e.setdefault(k, str(BIG / v))

        self._log = (self.out / "server.log").open("w")
        t0 = time.time()
        self.p = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT,
                                  env=e, start_new_session=True)
        while time.time() - t0 < 900:
            if self.p.poll() is not None:
                raise RuntimeError(f"server died rc={self.p.returncode}; "
                                   f"see {self.out / 'server.log'}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2)
                m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens",
                              (self.out / "server.log").read_text(errors="replace"), re.I)
                self.kv_tokens = int(m.group(1).replace(",", "")) if m else None
                return self
            except RuntimeError:
                raise
            except Exception:  # noqa: BLE001
                time.sleep(2)
        raise TimeoutError(f"not ready; see {self.out / 'server.log'}")

    def __exit__(self, *exc):
        if self.p and self.p.poll() is None:
            try:
                os.killpg(os.getpgid(self.p.pid), signal.SIGTERM)
                self.p.wait(timeout=90)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(os.getpgid(self.p.pid), signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    pass
        self._log.close()
        time.sleep(6)

    def ask(self, prompt: str, max_tokens: int = 400) -> str:
        body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
                           "temperature": 0.0, "seed": 12345,
                           "stop": ["\nQuestion:"]}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/completions", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.load(r)["choices"][0]["text"]


def run_config(name: str, kv_dtype: str, kv_cfg: dict | None, desc: str,
               gpu: int, prefix: str, tests: list[dict], max_len: int,
               root: Path, run_id: str, mode: str,
               extra_args: list[str] | None = None) -> list[dict]:
    rows = []
    out = root / name
    print(f"\n[m5] === {name} （{desc}）===", flush=True)
    try:
        with Server(gpu, max_len, out, kv_dtype=kv_dtype, kv_cfg=kv_cfg,
                    extra_args=extra_args) as s:
            print(f"[m5]   server up, GPU KV = {s.kv_tokens:,} tokens" if s.kv_tokens
                  else "[m5]   server up")
            ok = 0
            for i, ex in enumerate(tests):
                prompt = prefix + f"Question: {ex['question'].strip()}\nAnswer:"
                t0 = time.perf_counter()
                text = s.ask(prompt)
                dt = (time.perf_counter() - t0) * 1000
                pred, g = extract(text), gold(ex["answer"])
                correct = pred is not None and pred == g
                ok += correct
                rows.append({
                    "run_id": run_id, "ts": datetime.now().astimezone().isoformat(),
                    "mode": mode, "config": name, "kv_dtype": kv_dtype,
                    "model_key": MODEL_KEY, "gpu": gpu, "idx": i,
                    "gold": g, "pred": pred or "", "correct": correct,
                    "latency_ms": round(dt, 1),
                    # 輸出的雜湊：無損驗證靠這一欄逐一比對，不靠正確率
                    "out_sha1": hashlib.sha1(text.encode()).hexdigest()[:16],
                    "out_len": len(text),
                    "gpu_kv_cache_tokens": s.kv_tokens,
                    "n_shot": prefix.count("Question:"),
                    "desc": desc,
                    **{k: v for k, v in host_contention(exclude_gpu=gpu).items()
                       if k in ("level", "foreign_gpu_count", "foreign_max_util")},
                    "log": str(out / "server.log"),
                })
                if (i + 1) % 20 == 0:
                    print(f"[m5]   {i + 1}/{len(tests)}  正確率 {100 * ok / (i + 1):.1f}%",
                          flush=True)
            print(f"[m5]   → {name}: {ok}/{len(tests)} = {100 * ok / len(tests):.2f}%")
    except Exception as e:  # noqa: BLE001
        print(f"[m5]   🔴 {type(e).__name__}: {e}")
        (out / "error.txt").write_text(f"{type(e).__name__}: {e}\n")
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    if path.exists() and path.stat().st_size:
        existing = next(csv.reader(path.open(newline="")), [])
        if existing != fields:
            raise SystemExit(f"🔴 {path} schema 不合，拒絕寫入（檔案 {len(existing)} 欄 / "
                             f"資料 {len(fields)} 欄）")
        with path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerows(rows)
    else:
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print(f"\n[m5] wrote {len(rows)} rows -> {path}")


def summarise(rows: list[dict], mode: str) -> None:
    if not rows:
        print("[m5] 沒有資料")
        return
    cfgs = []
    for r in rows:
        if r["config"] not in cfgs:
            cfgs.append(r["config"])
    base = cfgs[0]
    acc = {c: sum(r["correct"] for r in rows if r["config"] == c) for c in cfgs}
    n = {c: sum(1 for r in rows if r["config"] == c) for c in cfgs}

    if mode == "mixed":
        print(f"\n{'=' * 66}\n品質 ↔ 容量的取捨曲線（GSM8K many-shot）\n{'=' * 66}")
        print(f"{'量化比例 f':>12}{'正確率':>10}{'ε (掉幅)':>12}{'容量放大':>10}{'n':>6}")
        b = 100 * acc[base] / max(1, n[base])
        COMP = {"fp8": 2.00, "int8_per_token_head": 1.94, "int4_per_token_head": 3.77}
        for c in cfgs:
            a2 = 100 * acc[c] / max(1, n[c])
            f = float(c.lstrip("f"))
            # 容量放大 = 1 / (量化層佔的比例/壓縮率 + 未量化層佔的比例)
            comp = next((v for k, v in COMP.items() if k in
                         (rows[0].get("kv_dtype") or "")), 3.77)
            m = 1.0 / (f / comp + (1 - f)) if f > 0 else 1.0
            print(f"{c:>12}{a2:>9.2f}%"
                  f"{('—' if c == base else f'{b - a2:+.2f} pt'):>12}"
                  f"{m:>9.2f}×{n[c]:>6}")
        print("\n下一步：用 gpu_blocks × 容量放大 跑 Oracle，得到 T(f)，"
              "再畫 (ε, T) 的 Pareto 前緣。")
    elif mode == "precision":
        print(f"\n{'=' * 62}\n精度階梯的 ε（GSM8K many-shot，相對 BF16）\n{'=' * 62}")
        print(f"{'設定':10s}{'正確率':>10s}{'n':>6s}{'ε (掉幅)':>12s}")
        b = 100 * acc[base] / max(1, n[base])
        for c in cfgs:
            a = 100 * acc[c] / max(1, n[c])
            print(f"{c:10s}{a:>9.2f}%{n[c]:>6d}"
                  f"{('—' if c == base else f'{b - a:+.2f} pt'):>12s}")
    else:
        print(f"\n{'=' * 62}\n無損驗證：輸出是否**逐字元相同**\n{'=' * 62}")
        by = {c: {r["idx"]: r["out_sha1"] for r in rows if r["config"] == c} for c in cfgs}
        print(f"{'設定':10s}{'正確率':>10s}{'與基準相同':>14s}{'判定':>10s}")
        for c in cfgs:
            a = 100 * acc[c] / max(1, n[c])
            same = sum(1 for i, h in by[c].items() if by[base].get(i) == h)
            tot = len(by[c])
            v = "—" if c == base else ("✅ 無損" if same == tot else "🔴 有差異")
            print(f"{c:10s}{a:>9.2f}%{f'{same}/{tot}':>14s}{v:>10s}")
        print("\nCPU/SSD 只是把位元組搬來搬去。**輸出若不同就是 bug，不是品質取捨。**")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--mode", default="precision",
                    choices=["precision", "lossless", "mixed"],
                    help="mixed = 掃量化比例 f，畫 ε(f) 曲線")
    ap.add_argument("--mixed-dtype", default="int4_per_token_head",
                    help="mixed 模式要量化成哪種 dtype")
    ap.add_argument("--fractions", type=float, nargs="*",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0],
                    help="被量化的層數比例")
    ap.add_argument("--n-shot", type=int, default=64,
                    help="共用前綴的範例數。要夠長才會進卸載路徑")
    ap.add_argument("--n-test", type=int, default=120)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    train, test = load_gsm8k("train"), load_gsm8k("test")
    prefix = build_prefix(train, a.n_shot)

    from transformers import AutoTokenizer
    os.environ.setdefault("HF_HOME", str(BIG / "hf-cache/huggingface"))
    tok = AutoTokenizer.from_pretrained(MODEL)
    plen = len(tok(prefix, add_special_tokens=False)["input_ids"])
    qmax = max(len(tok(f"Question: {e['question'].strip()}\nAnswer:",
                       add_special_tokens=False)["input_ids"]) for e in test[:a.n_test])
    max_len = plen + qmax + 400 + 256
    print(f"[m5] {a.n_shot}-shot 共用前綴 = {plen:,} tokens；"
          f"最長題目 {qmax} tokens；max_model_len = {max_len:,}")
    print(f"[m5] 測 {a.n_test} 題，mode={a.mode}")

    h = host_contention(exclude_gpu=a.gpu)
    print(f"[m5] 整機爭用：{h['level']}（外來 process {h['foreign_procs']} 個）")
    print("[m5] ℹ️  品質是正確率，不是時間——**整機爭用不影響正確率**，"
          "只影響 latency_ms 欄。")

    ok, got = wait_until_free(a.gpu, need_mib=22 * 1024, timeout_s=900)
    if not ok:
        print(f"[m5] 🔴 GPU {a.gpu} 只有 {got} MiB 可用，不開跑。")
        return 5

    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-m5-{a.mode}"
    root = BIG / "runs" / run_id
    tests = test[: a.n_test]
    if a.mode == "precision":
        todo = PRECISIONS
    elif a.mode == "mixed":
        todo = mixed_precision_configs(a.mixed_dtype, a.fractions)
    else:
        todo = [(n, "auto", c, d) for n, c, d in LOSSLESS]

    rows: list[dict] = []
    with GpuWatcher(gpu=a.gpu, out_path=str(OUT / f"gpu_guard_{a.mode}.json")) as g:
        if not g.started_clean:
            print(f"[m5] 🔴 GPU {a.gpu} 開跑前就不乾淨：{g.intruders}")
            return 2
        for item in todo:
            if a.mode == "precision":
                name, dtype, desc = item
                rows += run_config(name, dtype, None, desc, a.gpu, prefix, tests,
                                   max_len, root, run_id, a.mode)
            elif a.mode == "mixed":
                name, dtype, extra, desc = item
                rows += run_config(name, dtype, None, desc, a.gpu, prefix, tests,
                                   max_len, root, run_id, a.mode, extra_args=extra)
            else:
                name, dtype, cfg, desc = item
                rows += run_config(name, dtype, cfg, desc, a.gpu, prefix, tests,
                                   max_len, root, run_id, a.mode)

    path = Path(a.csv) if a.csv else OUT / f"gsm8k_{a.mode}.csv"
    write_rows(path, rows)
    summarise(rows, a.mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
