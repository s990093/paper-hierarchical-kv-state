#!/usr/bin/env python3
"""Milestone 2 — 成本常數的 2×N 矩陣。

`EXPERIMENT_PLAN.md` §3 的核心要求：

> 🔴 **成本常數是 2×5 矩陣，不是 1×5 向量**
> 兩類成本的性質完全不同，必須分開量：
> * **平時成本**（閒置時持續付出）—— 這個狀態佔多少位元組
> * **被需要時的成本**（一次性）—— 把它變回可用要花多少時間
>
> **若壓成一維，你會得到「重算最貴所以永不重算」的錯誤結論。**

## 論文的六階動作空間，在這台機器上各自怎麼量

| 動作 | 平時成本怎麼量 | 被需要時的成本怎麼量 |
|---|---|---|
| GPU BF16 | `--kv-cache-dtype auto` 的 KV pool 容量 | ≈0（prefix cache 直接命中） |
| GPU FP8 | `--kv-cache-dtype fp8` | 同一工作負載的 warm TTFT 差 |
| GPU INT8 | `--kv-cache-dtype int8_per_token_head` | 同上 |
| GPU INT4 | `--kv-cache-dtype int4_per_token_head` | 同上 |
| CPU | `OffloadingConnector` + `CPUOffloadingSpec` | warm TTFT（含 PCIe） |
| SSD | `TieringOffloadingSpec` + `fs` 次階 | warm TTFT（含 NVMe + PCIe） |
| **DROP** | **0**（這是它唯一的價值） | **cold TTFT，且隨位置成長** |

`EXPERIMENT_PLAN.md` 原本斷言 sm_86 量不到 FP8。**M1 已證明那是錯的**
（見 RUNLOG 發現 1）。這裡進一步發現 vLLM 還提供 `int4_per_token_head` /
`int8_per_token_head`，所以**六階裡有四階是同一個旗標的不同取值**，
在單一平台上就能量到完整的精度階梯。

## 三個子量測

* **A. 容量（平時成本）** —— 每個 KV dtype 的 `GPU KV cache size`。
  ⚠️ 這個值**run-to-run 會變**（實測在 41,648 / 48,128 兩個值之間跳，同設定同卡）。
  所以每個設定跑 `--repeats` 次，報**中位數與全距**，不報單一值。
* **B. 取回成本（被需要時的成本）** —— 兩輪共享前綴，warm TTFT 減掉
  「block 還在 GPU 裡」時的 warm TTFT，除以 token 數。
* **C. 重算成本 vs 位置** —— `C_recompute(position)`。送一個前綴 P 已被快取、
  後面接 C 個新 token 的請求，TTFT 就是「在位置 P 重算 C 個 token」的成本。
  掃 P 得到位置依賴性。**這是 `EXPERIMENT_PLAN.md` §3 說「不是常數」的那一項。**

## 用法

    python code/m2_cost_model.py --gpu 0 --stage capacity
    python code/m2_cost_model.py --gpu 0 --stage recompute
    python code/m2_cost_model.py --gpu 0 --stage all
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
import statistics
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
OUT = REPO / "results/m2_harness"

# 模型剖面。成本常數只在「同一個剖面內」可通約——
# 用 A 模型的預算配 B 模型的搬運成本是無意義的（見 m4_oracle.MODEL_PROFILES）。
MODEL_CHOICES = {
    "llama": ("NousResearch/Meta-Llama-3.1-8B-Instruct", 128.0),
    "llama-awq": ("hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", 128.0),
    "qwen-awq": (str(BIG / "models/Qwen2.5-7B-Instruct-1M-AWQ-noDCA"), 128.0),
}
MODEL = MODEL_CHOICES["llama"][0]
MODEL_KEY = "llama"
KV_KIB_PER_TOKEN_BF16 = 128.0
# 輸出檔名後綴。🔴 檔名必須帶模型，否則量 qwen-awq 會直接蓋掉 llama 的常數，
# 而兩者不可通約——覆蓋等於靜默地製造混用。
CSV_SUFFIX = ""


def out_csv(stem: str) -> Path:
    """llama 維持既有檔名（向後相容），其他模型一律帶 model_key。"""
    tag = "" if MODEL_KEY == "llama" else f"_{MODEL_KEY}"
    return OUT / f"{stem}{tag}{CSV_SUFFIX}.csv"

# 論文動作空間裡「住在 GPU 上」的四階，全部是同一個旗標的不同取值。
KV_DTYPES = [
    ("bf16", "auto", "GPU BF16 — 論文動作空間的最高精度階"),
    ("fp8", "fp8", "GPU FP8 — 計畫書原本說 sm_86 量不到，M1 已推翻"),
    ("int8", "int8_per_token_head", "GPU INT8"),
    ("int4", "int4_per_token_head", "GPU INT4 — 論文動作空間的最低精度階"),
]

CPU_BYTES = 24 * 1024**3

# 量 SSD 階時用的 CPU 階大小。必須 << 工作集，否則東西不會 cascade 下去。
SSD_TEST_CPU_BYTES = 1 * 1024**3

# 磁碟階的位置。⚠️ 論文的動作空間寫的是「SSD：NVMe I/O + PCIe」，
# 而 /ssd1..8 全部是 Samsung 870 QVO（SATA QLC，消費級最慢的一檔，
# 實測寫入 ~380 MB/s）。這台機器唯一可寫的 NVMe 是 /（Crucial P3）。
# 用 PAPER_HKV_FS_TIER 環境變數切換，兩個都量，並在結果裡標明裝置。
FS_ROOT = Path(os.environ.get("PAPER_HKV_FS_TIER", str(BIG / "kv_fs_tier")))

TIERS = [
    ("gpu_resident", None, "block 沒被逐出，warm 直接命中 GPU prefix cache（≈0 的基準）"),
    ("cpu", {"kv_connector": "OffloadingConnector", "kv_role": "kv_both",
             "kv_connector_extra_config": {
                 "spec_name": "CPUOffloadingSpec",
                 "cpu_bytes_to_use": CPU_BYTES, "eviction_policy": "lru"}},
     "CPU 階：PCIe 搬回來"),
    # 🔴 CPU 階刻意縮到 1 GiB，遠小於工作集（4 × 16384 tok × 128 KiB = 8 GiB）。
    #
    # 為什麼：2026-08-30 第一次量的時候 CPU 階開 24 GiB，工作集整個塞得進去，
    # 於是東西**根本沒 cascade 到磁碟**。vLLM 的 metrics 說得很清楚：
    #     kv_offload_tiering_chunk_queries:('0:primary',) = 2048   ← CPU 階
    #     kv_offload_tiering_chunk_queries:('1:fs',)      =    2   ← 磁碟階
    # 磁碟階只被查 2 次。那次量到的「SSD 550.9 ms」其實是 CPU 階，
    # 難怪它跟 cpu 只差 1.2%。**那不是「SSD 跟 CPU 一樣快」，是根本沒量到 SSD。**
    #
    # 縮小 CPU 階之後，block 才會真的落到磁碟上，warm 取回才會真的讀磁碟。
    ("ssd", {"kv_connector": "OffloadingConnector", "kv_role": "kv_both",
             "kv_connector_extra_config": {
                 "spec_name": "TieringOffloadingSpec",
                 "cpu_bytes_to_use": SSD_TEST_CPU_BYTES, "eviction_policy": "lru",
                 "secondary_tiers": [{"type": "fs", "root_dir": str(FS_ROOT)}]}},
     "CPU 階刻意縮小 → 強迫 cascade 到磁碟，量的才是真的磁碟階"),
    ("drop", None, "沒有第二階，warm 只能整段重算"),
    # 🔴 論文的動作空間有六階，先前的 M2 只量了四階
    #    （gpu_resident / cpu / ssd / drop）。GPU-FP8 與 GPU-INT4 是
    #    「住在 GPU 但精度降低」的狀態：容量變大，但每次讀取要反量化。
    #    這兩階的成本從來沒被量過，Oracle 因此無法把它們納入決策。
    #
    #    量法與 gpu_resident 完全相同（工作集塞得進 GPU、不逐出），
    #    只換 --kv-cache-dtype。warm TTFT 減掉 bf16 的 gpu_resident，
    #    差額就是**反量化的每 block 成本**。
    #    ⚠️ sm_86 沒有原生 FP8 tensor core，但 FP8/INT4 **儲存**可用
    #       （M1 已實測），反量化在 attention kernel 內做。
    ("gpu_fp8", None, "GPU FP8 儲存：容量 2×，讀取要反量化", "fp8"),
    ("gpu_int4", None, "GPU INT4 儲存：容量 4×，讀取要反量化",
     "int4_per_token_head"),
]


def _hc(gpu: int) -> dict:
    """整機爭用狀態，扁平化成 CSV 欄位。

    GPU 的 SM 是各卡獨占的，但 **PCIe / host RAM / /dev/shm 是全機共用**。
    別人在其他卡上跑不會出現在本卡的 compute-apps 裡，卻會拖慢搬運量測。
    所以每一列都要帶著這個，否則事後無法判斷該次數字可不可信。
    """
    h = host_contention(exclude_gpu=gpu)
    return {
        "host_contention": h["level"],          # QUIET / LIGHT / HEAVY
        "foreign_gpu_count": h["foreign_gpu_count"],
        "foreign_max_util": h["foreign_max_util"],
    }


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def env_for(extra: dict | None = None) -> dict:
    e = dict(os.environ)
    e["PATH"] = f"{VENV / 'bin'}:{e.get('PATH', '')}"
    e.setdefault("HF_HOME", str(BIG / "hf-cache/huggingface"))
    for k, v in (("XDG_CACHE_HOME", "xdg-cache"), ("TRITON_CACHE_DIR", "triton-cache"),
                 ("VLLM_CACHE_ROOT", "vllm-cache"),
                 ("FLASHINFER_WORKSPACE_BASE", "flashinfer-cache")):
        e.setdefault(k, str(BIG / v))
    e.update(extra or {})
    return e


class Server:
    """起一個 vLLM server，離開時確實收乾淨（含 /dev/shm 的 mmap）。"""

    def __init__(self, gpu: int, max_len: int, out: Path,
                 kv_dtype: str = "auto", kv_cfg: dict | None = None):
        self.gpu, self.max_len, self.out = gpu, max_len, out
        self.kv_dtype, self.kv_cfg = kv_dtype, kv_cfg
        self.port = free_port()
        self.p: subprocess.Popen | None = None
        self.kv_tokens: int | None = None
        self.kv_gib: float | None = None
        self.startup_s: float | None = None

    def __enter__(self) -> "Server":
        self.out.mkdir(parents=True, exist_ok=True)
        cmd = [str(VENV / "bin/vllm"), "serve", MODEL,
               "--port", str(self.port), "--max-model-len", str(self.max_len),
               "--gpu-memory-utilization", "0.90"]
        if self.kv_dtype != "auto":
            cmd += ["--kv-cache-dtype", self.kv_dtype]
        if self.kv_cfg:
            cmd += ["--kv-transfer-config", json.dumps(self.kv_cfg)]
        (self.out / "cmd.txt").write_text(" ".join(shlex.quote(c) for c in cmd) + "\n")

        e = env_for({"CUDA_VISIBLE_DEVICES": str(self.gpu)})
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
                self.startup_s = round(time.time() - t0, 1)
                self._scrape()
                return self
            except RuntimeError:
                raise
            except Exception:  # noqa: BLE001
                time.sleep(2)
        raise TimeoutError(f"not ready in 900s; see {self.out / 'server.log'}")

    def _scrape(self) -> None:
        import re
        t = (self.out / "server.log").read_text(errors="replace")
        m = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", t, re.I)
        self.kv_tokens = int(m.group(1).replace(",", "")) if m else None
        m = re.search(r"Available KV cache memory:\s*([\d.]+)\s*GiB", t, re.I)
        self.kv_gib = float(m.group(1)) if m else None
        # cascade 統計：磁碟階到底被查了幾次。若接近 0，代表沒量到磁碟。
        self.fs_queries = self.cpu_queries = None
        for pat, attr in ((r"chunk_queries:\('1:fs',\)=(\d+)", "fs_queries"),
                          (r"chunk_queries:\('0:primary',\)=(\d+)", "cpu_queries")):
            hits = re.findall(pat, t)
            if hits:
                setattr(self, attr, int(hits[-1]))
        self.o_direct = "falling back to buffered" not in t

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
        time.sleep(6)   # 讓 driver 把記憶體收回去

    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1/completions"


def stream_ttft(url: str, prompt: str, max_tokens: int = 8,
                timeout: float = 900.0) -> dict:
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0.0, "seed": 12345, "stream": True}).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, n = None, 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            b = line[5:].strip()
            if b == "[DONE]":
                break
            try:
                d = json.loads(b)
            except json.JSONDecodeError:
                continue
            if (d.get("choices") or [{}])[0].get("text", ""):
                n += 1
                if ttft is None:
                    ttft = time.perf_counter() - t0
    return {"ttft_ms": round(ttft * 1000, 3) if ttft else None,
            "total_ms": round((time.perf_counter() - t0) * 1000, 3), "n": n}


_TOK = None


def tok():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        os.environ.setdefault("HF_HOME", str(BIG / "hf-cache/huggingface"))
        _TOK = AutoTokenizer.from_pretrained(MODEL)
    return _TOK


def make_text(n_tokens: int, seed: int) -> str:
    """恰好 n_tokens 個 token 的文字。見 m3_baseline.make_prefix 的說明。"""
    t = tok()
    rng = random.Random(seed)
    vocab = getattr(t, "vocab_size", None) or len(t)
    lo, hi = 1000, max(1001, vocab - 100)
    text = t.decode([rng.randrange(lo, hi) for _ in range(n_tokens)],
                    skip_special_tokens=True)
    for _ in range(12):
        got = t(text, add_special_tokens=False)["input_ids"]
        if len(got) == n_tokens:
            return text
        if len(got) > n_tokens:
            text = t.decode(got[:n_tokens], skip_special_tokens=True)
        else:
            text += " " + t.decode(
                [rng.randrange(lo, hi) for _ in range(n_tokens - len(got))],
                skip_special_tokens=True)
    return text


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    if path.exists() and path.stat().st_size:
        existing = next(csv.reader(path.open(newline="")), [])
        if existing != fields:
            raise SystemExit(f"🔴 {path} 的 schema 與這批資料不合，拒絕寫入。\n"
                             f"   檔案 {len(existing)} 欄 / 資料 {len(fields)} 欄")
        with path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerows(rows)
    else:
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print(f"[m2] wrote {len(rows)} rows -> {path}")


# ───────────────────────── A. 容量（平時成本） ─────────────────────────

def stage_capacity(gpu: int, repeats: int, max_len: int) -> int:
    """每個 KV dtype 的 GPU KV pool 容量，重複 `repeats` 次。

    ⚠️ 這一項一定要重複。實測同設定同卡的 KV pool 會在兩個值之間跳
    （llama BF16：41,648 / 48,128，差 15.6%），單次量測會給出假的精確度。
    """
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-m2-capacity"
    root = BIG / "runs" / run_id
    rows = []
    for name, dtype, desc in KV_DTYPES:
        for i in range(repeats):
            out = root / f"{name}-r{i}"
            print(f"[m2] capacity {name:5s} rep {i + 1}/{repeats} ...", flush=True)
            try:
                with Server(gpu, max_len, out, kv_dtype=dtype) as s:
                    ok, kvt, kvg = True, s.kv_tokens, s.kv_gib
                    err = ""
            except Exception as e:  # noqa: BLE001
                ok, kvt, kvg, err = False, None, None, f"{type(e).__name__}: {e}"[:300]
                print(f"        🔴 {err}")
            rows.append({
                "run_id": run_id, "ts": datetime.now().astimezone().isoformat(),
                "model_key": MODEL_KEY, "kv_dtype_name": name,
                "kv_cache_dtype_flag": dtype, "rep": i, "gpu": gpu,
                "max_model_len": max_len, "server_ok": ok,
                "kv_cache_tokens": kvt if kvt else "NOT_MEASURED",
                "kv_cache_gib": kvg if kvg else "NOT_MEASURED",
                "error": err, "desc": desc,
                **_hc(gpu),
                "log": str(out / "server.log"),
            })
            if ok:
                print(f"        kv_tokens={kvt:,} kv_gib={kvg}")
    write_rows(out_csv("capacity_by_dtype"), rows)

    print(f"\n[m2] === 平時成本（容量越大 = 每 token 佔越少位元組）===")
    print(f"{'dtype':8s}{'n':>3s}{'median tok':>12s}{'min':>10s}{'max':>10s}"
          f"{'全距%':>8s}{'相對 BF16':>10s}")
    med = {}
    for name, _, _ in KV_DTYPES:
        v = [int(r["kv_cache_tokens"]) for r in rows
             if r["kv_dtype_name"] == name and r["server_ok"]]
        if not v:
            print(f"{name:8s}{0:>3d}{'NOT_MEASURED':>12s}")
            continue
        med[name] = statistics.median(v)
        rng = 100 * (max(v) - min(v)) / statistics.median(v)
        rel = med[name] / med["bf16"] if "bf16" in med else float("nan")
        print(f"{name:8s}{len(v):>3d}{med[name]:>12,.0f}{min(v):>10,}{max(v):>10,}"
              f"{rng:>7.1f}%{rel:>9.2f}×")
    return 0


# ─────────────── B. 取回成本（被需要時的成本） ───────────────

def stage_retrieval(gpu: int, ctx: int, n_prefixes: int, max_len: int,
                    only_tiers: set[str] | None = None) -> int:
    """每一階的「把 block 變回可用」要多久。

    gpu_resident 那一列刻意讓工作集塞得進 GPU（不逐出），
    它的 warm TTFT 就是「≈0 成本」的基準；其餘各階減掉它即為該階的搬運成本。
    """
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-m2-retrieval"
    root = BIG / "runs" / run_id
    rows = []
    for entry in TIERS:
        name, kv, desc = entry[0], entry[1], entry[2]
        kv_dtype = entry[3] if len(entry) > 3 else "auto"
        if only_tiers and name not in only_tiers:
            continue
        # 常駐 GPU 的各階：只送 1 個前綴，保證塞得下 → warm 是純 prefix-cache 命中
        n = 1 if name.startswith("gpu_") else n_prefixes
        out = root / name
        print(f"[m2] retrieval {name:13s} n_prefixes={n} ctx={ctx} ...", flush=True)
        try:
            with Server(gpu, max_len, out, kv_dtype=kv_dtype, kv_cfg=kv) as s:
                url = s.url()
                texts = [make_text(ctx, seed=7000 + ctx * 10 + i) for i in range(n)]
                for rnd in ("cold", "warm"):
                    for i, t in enumerate(texts):
                        r = stream_ttft(url, t)
                        rows.append({
                            "run_id": run_id,
                            "ts": datetime.now().astimezone().isoformat(),
                            "model_key": MODEL_KEY, "tier": name, "gpu": gpu,
                            "ctx": ctx, "n_prefixes": n, "round": rnd,
                            "kv_dtype": kv_dtype,
                            "prefix_idx": i, "ttft_ms": r["ttft_ms"],
                            "total_ms": r["total_ms"],
                            "gpu_kv_cache_tokens": s.kv_tokens,
                            "fs_tier_queries": s.fs_queries,
                            "cpu_tier_queries": s.cpu_queries,
                            "o_direct": s.o_direct,
                            "fs_root": str(FS_ROOT),
                            "desc": desc, **_hc(gpu),
                            "log": str(out / "server.log"),
                        })
                        print(f"        {rnd:<4} #{i} ttft={r['ttft_ms']}ms")
        except Exception as e:  # noqa: BLE001
            print(f"        🔴 {type(e).__name__}: {e}")
    write_rows(out_csv("retrieval_cost"), rows)

    base = [r["ttft_ms"] for r in rows
            if r["tier"] == "gpu_resident" and r["round"] == "warm" and r["ttft_ms"]]
    b = statistics.median(base) if base else None
    print(f"\n[m2] === 被需要時的成本（ctx={ctx}）===")
    print(f"GPU-resident warm TTFT = {b:.1f} ms（≈0 成本的基準）\n" if b else
          "🔴 沒有基準值\n")
    print(f"{'tier':14s}{'warm ms':>10s}{'減基準':>10s}{'µs/token':>11s}")
    for name, *_rest in TIERS:
        v = [r["ttft_ms"] for r in rows
             if r["tier"] == name and r["round"] == "warm" and r["ttft_ms"]]
        if not v or b is None:
            print(f"{name:14s}{'NOT_MEASURED':>10s}")
            continue
        m = statistics.median(v)
        print(f"{name:14s}{m:>10.1f}{m - b:>10.1f}{1000 * (m - b) / ctx:>11.2f}")
    return 0


# ─────────────── C. 重算成本 vs 位置 ───────────────

def stage_recompute(gpu: int, max_len: int, chunk: int, positions: list[int]) -> int:
    """C_recompute(position)：在位置 P 重算 `chunk` 個 token 要多久。

    `EXPERIMENT_PLAN.md` §3：
    > `C_recompute` **不是常數**，隨 block 的絕對位置成長（attention 二次項）。
    > 要量成 `C_recompute(position)`，不是單一純量。

    做法：先送一個長度 P 的前綴讓它進 prefix cache，再送「同樣的 P + 新的 chunk」。
    第二次的 TTFT 就是「前 P 個 token 命中、後 chunk 個 token 現算」的成本。
    P 越大，這 chunk 個 token 的 attention 要讀越多前序 KV → 成本應隨 P 成長。
    """
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-m2-recompute"
    out = BIG / "runs" / run_id
    rows = []
    print(f"[m2] recompute chunk={chunk} positions={positions}")
    with Server(gpu, max_len, out) as s:
        url = s.url()
        for P in positions:
            pref = make_text(P, seed=9000 + P) if P else ""
            if P:
                stream_ttft(url, pref)          # 先把前綴灌進 prefix cache
                stream_ttft(url, pref)          # 再一次確保命中（第一次可能還在寫入）
            for rep in range(3):
                suffix = make_text(chunk, seed=91000 + P * 10 + rep)
                r = stream_ttft(url, pref + suffix)
                rows.append({
                    "run_id": run_id, "ts": datetime.now().astimezone().isoformat(),
                    "model_key": MODEL_KEY, "gpu": gpu,
                    "cached_prefix_tokens": P, "recomputed_tokens": chunk,
                    "rep": rep, "ttft_ms": r["ttft_ms"], "total_ms": r["total_ms"],
                    "gpu_kv_cache_tokens": s.kv_tokens, **_hc(gpu),
                    "log": str(out / "server.log"),
                })
                print(f"        P={P:>6} rep{rep} ttft={r['ttft_ms']}ms")
    write_rows(out_csv("recompute_position"), rows)

    print(f"\n[m2] === C_recompute(position)，每次重算 {chunk} 個 token ===")
    print(f"{'cached prefix':>14s}{'median ms':>11s}{'vs P=0':>9s}{'µs/token':>10s}")
    med0 = None
    for P in positions:
        v = [r["ttft_ms"] for r in rows
             if r["cached_prefix_tokens"] == P and r["ttft_ms"]]
        if not v:
            continue
        m = statistics.median(v)
        if med0 is None:
            med0 = m
        print(f"{P:>14,}{m:>11.1f}{m / med0:>8.2f}×{1000 * m / chunk:>10.2f}")
    print("\n若這一欄隨 P 明顯成長，就證實了『重算成本不是常數』。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--stage", default="all",
                    choices=["all", "capacity", "retrieval", "recompute"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--n-prefixes", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--positions", type=int, nargs="*",
                    default=[0, 4096, 8192, 16384, 24576])
    ap.add_argument("--model", default="llama", choices=list(MODEL_CHOICES),
                    help="要量哪個模型的成本常數。🔴 成本常數只在同一個剖面內"
                         "可通約，換模型就要整組重量")
    ap.add_argument("--tiers", nargs="*", default=None,
                    help="只量指定的階（如 gpu_fp8 gpu_int4）。預設全部")
    ap.add_argument("--csv-suffix", default="",
                    help="輸出檔名後綴，避免覆蓋既有結果（如 _quiet）")
    a = ap.parse_args()

    global MODEL, MODEL_KEY, KV_KIB_PER_TOKEN_BF16, CSV_SUFFIX
    MODEL, KV_KIB_PER_TOKEN_BF16 = MODEL_CHOICES[a.model]
    MODEL_KEY, CSV_SUFFIX = a.model, a.csv_suffix
    print(f"[m2] 模型剖面 {a.model} → {MODEL}")
    print(f"[m2] 輸出檔：{out_csv('retrieval_cost')}")

    ok, got = wait_until_free(a.gpu, need_mib=22 * 1024, timeout_s=600)
    if not ok:
        print(f"[m2] 🔴 GPU {a.gpu} 只有 {got} MiB 可用，不開跑。")
        return 5

    h = host_contention(exclude_gpu=a.gpu)
    print(f"[m2] 整機爭用：{h['level']}  外來 process {h['foreign_procs']} 個 "
          f"在 GPU {h['foreign_gpus']}，最高使用率 {h['foreign_max_util']}%")
    if h["level"] == "HEAVY" and a.stage in ("all", "retrieval"):
        print("[m2] ⚠️  retrieval 量的是 PCIe 搬運成本，而 PCIe 是全機共用的。")
        print("[m2]    實測：整機忙碌時卸載的 warm TTFT 會被灌水 26–52%。")
        print("[m2]    這批 retrieval 數字會標成 host_contention=HEAVY，")
        print("[m2]    機器安靜時要重量一次再比對。")

    rc = 0
    with GpuWatcher(gpu=a.gpu, out_path=str(OUT / "gpu_guard_m2.json")) as g:
        if not g.started_clean:
            print(f"[m2] 🔴 GPU {a.gpu} 開跑前就不乾淨：{g.intruders}")
            return 2
        if a.stage in ("all", "capacity"):
            rc |= stage_capacity(a.gpu, a.repeats, max_len=8192)
        if a.stage in ("all", "retrieval"):
            rc |= stage_retrieval(a.gpu, a.ctx, a.n_prefixes,
                                  max_len=a.ctx + 1024,
                                  only_tiers=set(a.tiers) if a.tiers else None)
        if a.stage in ("all", "recompute"):
            rc |= stage_recompute(a.gpu, max_len=max(a.positions) + a.chunk + 1024,
                                  chunk=a.chunk, positions=a.positions)
    if g.contaminated:
        print(f"[m2] 🔴 {g.verdict()} — 有人插隊，這批數字作廢，必須重量。")
        return 3
    return rc


if __name__ == "__main__":
    sys.exit(main())
