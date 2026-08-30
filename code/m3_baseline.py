#!/usr/bin/env python3
"""Milestone 3 — Tier 0 baselines。

## 工作負載為什麼長這樣（這是最容易做錯的地方）

KV 卸載**只在有重用時才有價值**。單發一個長請求，卸載連接器什麼也不會做——
它把 block 存到 CPU，然後沒有人再來要。這樣量出來的 `lru` 與 `arc` 會完全一樣，
而 `EXPERIMENT_PLAN.md` §4 的 M3 驗收正好把「lru 與 arc 數字完全相同」列為
**「卸載沒真的發生」的警訊**。

所以工作負載是**兩輪、共享前綴**：

    第 1 輪（cold）: 依序送 N 個各自不同的長前綴 P_1..P_N
                     N × L 刻意大於 GPU KV 容量 -> 前面的必然被逐出
    第 2 輪（warm）: 用同樣順序再送一次同樣的 P_1..P_N
                     P_1 在 GPU 裡已經沒了。它能不能從 CPU/SSD 階拿回來？

**cold TTFT 與 warm TTFT 的差，就是那一階卸載的價值。**
* `full_gpu`：沒有第二階，warm 只能重算 -> Δ ≈ 0（這是對照組）
* `cpu_lru` / `cpu_arc`：能拿回來 -> warm TTFT 明顯低於 cold
* `tier_fs`：多一層磁碟，容量更大但更慢

這同時也是論文 §2.6「重算的價值不在單次成本」的最小可觀測形式。

## 壓力軸

`EXPERIMENT_PLAN.md` §4：3090 上用「單請求 × context 遞增」，不是並行度。
context 上限由 M1 實測的懸崖決定（`results/m1_capacity/capacity.csv`），不是猜的。

## 共用機器的污染防護

這台機器有二十幾個使用者。每次 run 全程用 `gpu_guard.GpuWatcher` 監看，
只要出現外來 process 就把該 run 標成 `CONTAMINATED` 並且**不寫進結果 CSV**。
見 `code/gpu_guard.py` 的說明。

## 用法

    python code/m3_baseline.py --list
    python code/m3_baseline.py --baseline cpu_lru --model llama --gpu 2
    python code/m3_baseline.py --all --model llama          # 自動排到空閒的卡上
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_guard import GpuWatcher, idle_gpus  # noqa: E402

BIG = Path(os.environ.get("PAPER_HKV_BIG", "/ssd7/hungwei/paper-hkv"))
VENV = BIG / "venv/vllm"
REPO = Path(__file__).resolve().parent.parent

MODELS = {
    "llama": {
        "path": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "kv_kib_per_token": 128.0,
        "note": "對照模型（官方 meta-llama 為 gated，用無門檻鏡像）",
    },
    "qwen": {
        "path": str(BIG / "models/Qwen2.5-7B-Instruct-1M-noDCA"),
        "kv_kib_per_token": 56.0,
        "note": "主力模型，已移除 DCA（見 code/make_nodca_model.py）",
    },
}

# CPU 階大小。vLLM 把它配置成 /dev/shm 上的 mmap 檔，而 /dev/shm 只有 220 GB
# 且是**全機共用**的。四個 baseline 平行跑 = 4 份，所以不能開太大。
# 24 GiB 的依據：最大工作集是 4 個前綴 × 32768 token × 128 KiB = 16 GiB，
# 留 1.5 倍餘裕，確保量到的是「能不能取回」而不是「CPU 階也在 thrash」。
CPU_BYTES = 24 * 1024**3
FS_ROOT = BIG / "kv_fs_tier"

BASELINES: dict[str, dict] = {
    "full_gpu": {
        "kv": None,
        "desc": "無卸載。品質上界，也是『沒有第二階時 warm 只能重算』的對照",
    },
    "cpu_lru": {
        "kv": {"kv_connector": "OffloadingConnector", "kv_role": "kv_both",
               "kv_connector_extra_config": {
                   "spec_name": "CPUOffloadingSpec",
                   "cpu_bytes_to_use": CPU_BYTES, "eviction_policy": "lru"}},
        "desc": "vLLM 內建 LRU。EXPERIMENT_PLAN 稱之為『最誠實的對手』",
    },
    "cpu_arc": {
        "kv": {"kv_connector": "OffloadingConnector", "kv_role": "kv_both",
               "kv_connector_extra_config": {
                   "spec_name": "CPUOffloadingSpec",
                   "cpu_bytes_to_use": CPU_BYTES, "eviction_policy": "arc"}},
        "desc": "vLLM 內建 ARC。堵住『只贏最笨的』這個審稿意見",
    },
    "tier_fs": {
        "kv": {"kv_connector": "OffloadingConnector", "kv_role": "kv_both",
               "kv_connector_extra_config": {
                   "spec_name": "TieringOffloadingSpec",
                   "cpu_bytes_to_use": CPU_BYTES, "eviction_policy": "lru",
                   "secondary_tiers": [{"type": "fs", "root_dir": str(FS_ROOT)}]}},
        "desc": "CPU + 磁碟兩階。對應論文動作空間的 SSD 那一階",
    },
    "lmcache": {
        "kv": {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"},
        # ⚠️ 跑在**獨立的 venv**：lmcache 會把 prometheus-client 從 0.26.0 降到
        #    0.24.1，而 vLLM 的 /metrics 正是我們量 baseline 用的端點。
        #    已經量完的四個 baseline 不能因為裝第五個而失效。
        "venv": BIG / "venv/lmcache",
        "env": {
            "LMCACHE_CHUNK_SIZE": "256",
            "LMCACHE_LOCAL_CPU": "True",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": str(CPU_BYTES // 1024**3),
        },
        # ⚠️ 必須記錄的不對等：lmcache 0.5.4 的 wheel 沒有 lmcache.c_ops /
        #    lmcache.cuda_ops 編譯擴充（啟動時 log 明白寫著
        #    "compiled extension not found; CudaDeviceOps stays on the torch
        #    baseline for all ops"）。它的搬運路徑因此走 torch baseline，
        #    比有編譯擴充時慢。**拿它的數字跟 vLLM 原生卸載比時要註明這一點**，
        #    否則就是拿一個被handicap的對手來凸顯自己。
        "caveat": "no compiled c_ops/cuda_ops; transfer path on torch baseline",
        "desc": "LMCache（EXPERIMENT_PLAN Tier 0 #5）。獨立 venv，且無編譯擴充",
    },
}

# 每個 context 長度送幾個不同前綴。N × ctx 要大於 GPU KV 容量才會逼出逐出。
N_PREFIXES = 4
GEN_TOKENS = 32          # 每個請求產生的 token 數，用來算 TPOT
CTX_LADDER = [4096, 8192, 16384, 32768]


def shm_free_bytes() -> int:
    st = os.statvfs("/dev/shm")
    return st.f_bavail * st.f_frsize


def check_shm(need: int) -> str | None:
    """/dev/shm 夠不夠放 CPU 階。不夠就回傳訊息（呼叫端負責中止）。

    2026-08-30 踩過：killed 的 vLLM server 會把 32 GiB 的 mmap 檔留在 /dev/shm，
    累積幾輪之後 220 GB 全滿，接著每個帶卸載的 baseline 都在啟動時死掉，
    而錯誤訊息完全不指向真因。先檢查，並提示怎麼清。
    """
    free = shm_free_bytes()
    if free >= need:
        return None
    return (f"/dev/shm 只剩 {free / 1024**3:.1f} GB，這個 baseline 需要 "
            f"{need / 1024**3:.1f} GB。先跑 `python code/shm_gc.py --apply` "
            f"清掉自己洩漏的 mmap 檔。")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(url: str, payload: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def stream_ttft(url: str, payload: dict, timeout: float = 600.0) -> dict:
    """串流送一個請求，量 TTFT 與總時間。回傳實測值，不做任何推估。"""
    payload = {**payload, "stream": True}
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    n_chunks = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                d = json.loads(body)
            except json.JSONDecodeError:
                continue
            txt = (d.get("choices") or [{}])[0].get("text", "")
            if txt:
                n_chunks += 1
                if ttft is None:
                    ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    return {
        "ttft_ms": round(ttft * 1000, 3) if ttft is not None else None,
        "total_ms": round(total * 1000, 3),
        "n_chunks": n_chunks,
        # TPOT 只在有兩個以上 token 時才有定義，否則寫 None 不要硬算
        "tpot_ms": (round((total - ttft) * 1000 / (n_chunks - 1), 3)
                    if ttft is not None and n_chunks > 1 else None),
    }


def scrape_metrics(port: int) -> dict:
    """抓 vLLM /metrics 中與本研究相關的計數器。抓不到就是 None，不填估值。"""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return {}
    want = ("prefix_cache_queries", "prefix_cache_hits", "gpu_prefix_cache",
            "offload", "kv_offload", "cpu_cache")
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if not any(w in line for w in want):
            continue
        try:
            name, val = line.rsplit(" ", 1)
            out[name.split("{")[0]] = out.get(name.split("{")[0], 0.0) + float(val)
        except ValueError:
            continue
    return out


def make_prefix(tokenizer, n_tokens: int, seed: int) -> tuple[str, int]:
    """造一個**恰好** n_tokens 長、彼此不重疊的前綴。回傳 (text, 實際 token 數)。

    用固定 seed 的隨機 token id 而不是重複同一段文字——重複的文字會讓
    vLLM 的 prefix cache 在不同「前綴」之間意外命中，把 cold/warm 的對比洗掉。

    ⚠️ decode(ids) 之後再 encode 回來**不會**得到同樣的長度（BPE 會把相鄰的
    片段合併或拆開）。2026-08-30 第一版沒處理這件事，結果 ctx=32768 的 prompt
    實際超過 max_model_len，整批 run 以 `HTTP 400 Bad Request` 收場。
    所以這裡實際 encode 一次，多退少補，直到長度正確為止。
    """
    rng = random.Random(seed)
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    lo, hi = 1000, max(1001, vocab_size - 100)   # 避開特殊 token 區間

    ids = [rng.randrange(lo, hi) for _ in range(n_tokens)]
    text = tokenizer.decode(ids, skip_special_tokens=True)
    for _ in range(12):
        got = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(got) == n_tokens:
            return text, n_tokens
        if len(got) > n_tokens:
            # 太長就從 token 層面截斷，再 decode 回去
            text = tokenizer.decode(got[:n_tokens], skip_special_tokens=True)
        else:
            text = text + " " + tokenizer.decode(
                [rng.randrange(lo, hi) for _ in range(n_tokens - len(got))],
                skip_special_tokens=True)
    # 收斂不了就回報實際長度，不要假裝是 n_tokens
    return text, len(tokenizer(text, add_special_tokens=False)["input_ids"])


class Server:
    def __init__(self, model: str, max_len: int, gpu: int, kv: dict | None, out: Path,
                 venv: Path | None = None, extra_env: dict | None = None):
        self.model, self.max_len, self.gpu, self.kv, self.out = model, max_len, gpu, kv, out
        self.venv = Path(venv) if venv else VENV
        self.extra_env = extra_env or {}
        self.port = free_port()
        self.p: subprocess.Popen | None = None

    def __enter__(self) -> Server:
        self.out.mkdir(parents=True, exist_ok=True)
        cmd = [str(self.venv / "bin/vllm"), "serve", self.model,
               "--port", str(self.port),
               "--max-model-len", str(self.max_len),
               "--gpu-memory-utilization", "0.90"]
        if self.kv:
            cmd += ["--kv-transfer-config", json.dumps(self.kv)]
        (self.out / "cmd.txt").write_text(" ".join(shlex.quote(c) for c in cmd) + "\n")

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        env["PATH"] = f"{self.venv / 'bin'}:{env.get('PATH', '')}"
        env.setdefault("HF_HOME", str(BIG / "hf-cache/huggingface"))
        env.update(self.extra_env)
        for k, v in (("XDG_CACHE_HOME", "xdg-cache"), ("TRITON_CACHE_DIR", "triton-cache"),
                     ("VLLM_CACHE_ROOT", "vllm-cache"),
                     ("FLASHINFER_WORKSPACE_BASE", "flashinfer-cache")):
            env.setdefault(k, str(BIG / v))

        self._log = (self.out / "server.log").open("w")
        self.p = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT,
                                  env=env, start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < 900:
            if self.p.poll() is not None:
                raise RuntimeError(
                    f"server died rc={self.p.returncode}; see {self.out / 'server.log'}")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2)
                self.startup_s = round(time.time() - t0, 1)
                return self
            except Exception:  # noqa: BLE001
                time.sleep(2)
        raise TimeoutError(f"server not ready in 900s; see {self.out / 'server.log'}")

    def __exit__(self, *exc) -> None:
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

    def kv_cache_tokens(self) -> int | None:
        import re
        m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens",
                      (self.out / "server.log").read_text(errors="replace"), re.I)
        return int(m.group(1).replace(",", "")) if m else None


def run_one(baseline: str, model_key: str, gpu: int, ctxs: list[int],
            csv_path: Path) -> int:
    cfg = BASELINES[baseline]
    mdl = MODELS[model_key]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}-m3-{model_key}-{baseline}"
    root = BIG / "runs" / run_id
    root.mkdir(parents=True, exist_ok=True)
    print(f"[m3] run_id={run_id} gpu={gpu} baseline={baseline} model={mdl['path']}")

    if baseline == "tier_fs":
        FS_ROOT.mkdir(parents=True, exist_ok=True)

    # 帶 CPU 階的 baseline 開跑前先確認 /dev/shm 放得下
    if cfg["kv"] and "cpu_bytes_to_use" in cfg["kv"].get("kv_connector_extra_config", {}):
        msg = check_shm(cfg["kv"]["kv_connector_extra_config"]["cpu_bytes_to_use"])
        if msg:
            print(f"[m3] 🔴 {msg}")
            return 4

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mdl["path"])

    # 餘裕：GEN_TOKENS + BOS/EOS + tokenizer 邊界誤差。寧可多給也不要撞 400。
    max_len = max(ctxs) + GEN_TOKENS + 1024
    rows: list[dict] = []

    with GpuWatcher(gpu=gpu, out_path=str(root / "gpu_guard.json")) as guard:
        if not guard.started_clean:
            print(f"[m3] 🔴 GPU {gpu} 開跑前就不乾淨，放棄。intruders={guard.intruders}")
            return 2
        try:
            with Server(mdl["path"], max_len, gpu, cfg["kv"], root,
                        venv=cfg.get("venv"), extra_env=cfg.get("env")) as srv:
                kvtok = srv.kv_cache_tokens()
                print(f"[m3]   server up in {srv.startup_s}s  "
                      f"GPU KV cache = {kvtok:,} tokens" if kvtok else "[m3]   server up")
                url = f"http://127.0.0.1:{srv.port}/v1/completions"

                for ctx in ctxs:
                    built = [make_prefix(tok, ctx, seed=1000 * ctx + i)
                             for i in range(N_PREFIXES)]
                    prefixes = [b[0] for b in built]
                    actual_toks = [b[1] for b in built]
                    for rnd in ("cold", "warm"):
                        for i, pref in enumerate(prefixes):
                            r = stream_ttft(url, {
                                "model": mdl["path"], "prompt": pref,
                                "max_tokens": GEN_TOKENS, "temperature": 0.0,
                                "seed": 12345,
                            })
                            rows.append({
                                "run_id": run_id,
                                "ts": datetime.now().astimezone().isoformat(),
                                "baseline": baseline, "model_key": model_key,
                                "model": mdl["path"], "gpu": gpu,
                                "ctx": ctx, "actual_prompt_tokens": actual_toks[i],
                                "round": rnd, "prefix_idx": i,
                                "ttft_ms": r["ttft_ms"], "tpot_ms": r["tpot_ms"],
                                "total_ms": r["total_ms"], "gen_tokens": r["n_chunks"],
                                "gpu_kv_cache_tokens": kvtok,
                                "n_prefixes": N_PREFIXES,
                                "caveat": cfg.get("caveat", ""),
                                "quality_score": "NOT_MEASURED",
                                "quality_metric": "NOT_MEASURED",
                                "contaminated": guard.contaminated,
                                "guard_verdict": guard.verdict(),
                                "log": str(root / "server.log"),
                            })
                            print(f"[m3]   ctx={ctx:>6} {rnd:<4} #{i} "
                                  f"ttft={r['ttft_ms']}ms tpot={r['tpot_ms']}ms")
                    (root / f"metrics_ctx{ctx}.json").write_text(
                        json.dumps(scrape_metrics(srv.port), indent=2))
        except Exception as e:  # noqa: BLE001
            # 部分失敗不該讓已經量到的資料消失 —— 它們一樣是實測值。
            print(f"[m3] 🔴 FAILED after {len(rows)} rows: {type(e).__name__}: {e}")
            (root / "error.txt").write_text(f"{type(e).__name__}: {e}\n")
            if rows and not guard.contaminated:
                write_csv(csv_path, rows)
                summarise(rows)
            return 1

    # 禁令 1：被污染的數字不寫進結果。
    if guard.contaminated:
        print(f"[m3] 🔴 {guard.verdict()} — 有人插隊，這次不算數，必須重量。")
        print(json.dumps(guard.report(), indent=2, ensure_ascii=False))
        (root / "CONTAMINATED").write_text(json.dumps(guard.report(), indent=2))
        return 3

    write_csv(csv_path, rows)
    summarise(rows)
    return 0


def summarise(rows: list[dict]) -> None:
    from statistics import median
    print(f"\n[m3] === {rows[0]['baseline']} / {rows[0]['model_key']} ===")
    print(f"{'ctx':>8} {'cold TTFT':>12} {'warm TTFT':>12} {'Δ':>10}  {'Δ%':>7}")
    for ctx in sorted({r["ctx"] for r in rows}):
        c = [r["ttft_ms"] for r in rows if r["ctx"] == ctx and r["round"] == "cold" and r["ttft_ms"]]
        w = [r["ttft_ms"] for r in rows if r["ctx"] == ctx and r["round"] == "warm" and r["ttft_ms"]]
        if not c or not w:
            continue
        mc, mw = median(c), median(w)
        print(f"{ctx:>8} {mc:>11.1f}ms {mw:>11.1f}ms {mc - mw:>9.1f}ms "
              f"{100 * (mc - mw) / mc:>6.1f}%")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"[m3] appended {len(rows)} rows -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", choices=list(BASELINES))
    ap.add_argument("--model", default="llama", choices=list(MODELS))
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--all", action="store_true", help="所有 baseline 平行排到空閒的卡")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--ctx", type=int, nargs="*", default=CTX_LADDER)
    ap.add_argument("--csv", default=str(REPO / "results/m3_baseline/baseline.csv"))
    a = ap.parse_args()

    if a.list:
        for k, v in BASELINES.items():
            print(f"  {k:12s} {v['desc']}")
        print()
        for k, v in MODELS.items():
            print(f"  {k:12s} {v['path']}  ({v['note']})")
        return 0

    if a.all:
        free = idle_gpus()
        todo = list(BASELINES)
        if len(free) < len(todo):
            print(f"[m3] 只有 {len(free)} 張空閒卡，要跑 {len(todo)} 個 baseline。"
                  f"空閒: {free}。請減少 baseline 或等卡空出來。")
            return 2
        procs = []
        for b, g in zip(todo, free):
            log = BIG / "logs" / f"m3_{a.model}_{b}.log"
            cmd = [sys.executable, "-u", __file__, "--baseline", b,
                   "--model", a.model, "--gpu", str(g), "--csv", a.csv,
                   "--ctx", *map(str, a.ctx)]
            f = log.open("w")
            procs.append((b, g, subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)))
            print(f"[m3] gpu{g} -> {b}  (log {log})")
        rc = 0
        for b, g, p in procs:
            r = p.wait()
            print(f"[m3] {b} on gpu{g} exited {r}")
            rc = rc or r
        return rc

    if not a.baseline or a.gpu is None:
        ap.print_help()
        return 2
    return run_one(a.baseline, a.model, a.gpu, a.ctx, Path(a.csv))


if __name__ == "__main__":
    sys.exit(main())
