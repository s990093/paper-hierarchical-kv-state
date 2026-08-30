#!/usr/bin/env python3
"""Milestone 1 — 容量懸崖實測。

EXPERIMENT_PLAN.md §2：論文 §2.5 宣稱「3090 上 64K 可置入、128K 超出」，
**那是算出來的**。這支腳本量真的懸崖在哪。

## 為什麼不做樸素的二分搜尋

樸素做法是對 max_model_len 二分，每次啟一個 server 看會不會 OOM ——
7B 模型每次啟動約 60–120 秒，七次迭代就是 15 分鐘，而且**量到的是「啟動成功與否」
這個離散訊號**，資訊量很低。

vLLM 啟動時會直接把答案印在 log 裡：

    GPU KV cache size: 123,456 tokens

這一個數字就是**這張卡在這個設定下能裝的 KV token 上限**，也就是懸崖本身。
所以流程是：

  1. **量測**：用一個保證裝得下的 max_model_len 起 server，讀出 `GPU KV cache size`
  2. **驗證**：用量到的 N 起 server（應該成功）、再用 N×OVERSHOOT 起（應該失敗）
     ← 沒有這一步就只是抄 log，不算量測

一個設定 3 次啟動，不是 7 次，而且產出的是連續量而非二元訊號。

## 用法

    python code/m1_capacity.py --gpu 0 --config qwen-bf16
    python code/m1_capacity.py --list

輸出 results/m1_capacity/capacity.csv（append），每列都帶 run_id。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

BIG = Path(os.environ.get("PAPER_HKV_BIG", "/ssd7/hungwei/paper-hkv"))
VENV = BIG / "venv/vllm"
REPO = Path(__file__).resolve().parent.parent

# 探測用的起始長度：必須小到「任何設定都裝得下」，否則讀不到 KV cache size。
PROBE_LEN = 8192
# 驗證上界時超出多少才算「確定爆掉」。1.15 給碎片化留餘裕，避免把邊界雜訊當成懸崖。
OVERSHOOT = 1.15

CONFIGS: dict[str, dict] = {
    "qwen-bf16": {
        # 用 no-DCA 變體：vLLM 0.28.0 V1 載不動啟用 DCA 的原版。
        # 見 code/make_nodca_model.py 的完整說明。
        "model": str(BIG / "models/Qwen2.5-7B-Instruct-1M-noDCA"),
        "weight_dtype": "BF16",
        "kv_dtype": "auto",
        "extra": [],
        "note": "主力模型的 BF16 基準（敏感度分析用）",
    },
    "qwen-bf16-kvfp8": {
        "model": str(BIG / "models/Qwen2.5-7B-Instruct-1M-noDCA"),
        "weight_dtype": "BF16",
        "kv_dtype": "fp8",
        "extra": ["--kv-cache-dtype", "fp8"],
        "note": "sm_86 無原生 FP8。此設定用來『量出 vLLM 實際接不接受』，不是假設。",
    },
    "llama-bf16": {
        "model": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "weight_dtype": "BF16",
        "kv_dtype": "auto",
        "extra": [],
        "note": "對照模型，κ 與 Qwen 差 2 倍",
    },
    "llama-bf16-kvfp8": {
        "model": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "weight_dtype": "BF16",
        "kv_dtype": "fp8",
        "extra": ["--kv-cache-dtype", "fp8"],
        "note": "同上，驗證 Ampere 的 KV dtype 支援度",
    },
}

# vLLM 把可用 KV 容量印成這一行；版本間措辭會變，所以多留幾個 pattern。
KV_PATTERNS = [
    re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens", re.I),
    re.compile(r"KV cache size:\s*([\d,]+)\s*tokens", re.I),
    re.compile(r"# GPU blocks:\s*([\d,]+)", re.I),
]
CONC_PAT = re.compile(r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x", re.I)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def launch(model: str, max_len: int, gpu: int, extra: list[str], out: Path,
           timeout: int = 900) -> dict:
    """啟一個 vLLM server，等它 ready 或死掉。回傳量到的東西，不做任何推估。"""
    out.mkdir(parents=True, exist_ok=True)
    port = free_port()
    cmd = [str(VENV / "bin/vllm"), "serve", model,
           "--port", str(port),
           "--max-model-len", str(max_len),
           "--gpu-memory-utilization", "0.90",
           *extra]
    (out / "cmd.txt").write_text(" ".join(shlex.quote(c) for c in cmd) + "\n")

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PATH"] = f"{VENV / 'bin'}:{env.get('PATH', '')}"
    env.setdefault("HF_HOME", str(BIG / "hf-cache/huggingface"))
    for k, v in (("XDG_CACHE_HOME", "xdg-cache"), ("TRITON_CACHE_DIR", "triton-cache"),
                 ("VLLM_CACHE_ROOT", "vllm-cache"), ("FLASHINFER_WORKSPACE_BASE", "flashinfer-cache")):
        env.setdefault(k, str(BIG / v))

    log = (out / "server.log").open("w")
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                         start_new_session=True)

    ready, died = False, False
    while time.time() - t0 < timeout:
        if p.poll() is not None:
            died = True
            break
        try:
            import urllib.request
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            ready = True
            break
        except Exception:  # noqa: BLE001
            time.sleep(2)

    if not died:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            p.wait(timeout=60)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
    log.close()

    text = (out / "server.log").read_text(errors="replace")
    kv_tokens = None
    for pat in KV_PATTERNS:
        m = pat.search(text)
        if m and "blocks" not in pat.pattern:
            kv_tokens = int(m.group(1).replace(",", ""))
            break
    conc = CONC_PAT.search(text)

    # 失敗時把錯誤原因抓出來 —— 禁令 2：不准跳過失敗。
    err = None
    if not ready:
        for line in text.splitlines():
            if any(k in line for k in ("ValueError", "RuntimeError", "torch.OutOfMemoryError",
                                       "CUDA out of memory", "Error", "is larger than the maximum")):
                err = line.strip()[:400]
                break

    res = {
        "ready": ready,
        "exit_code": p.returncode,
        "elapsed_s": round(time.time() - t0, 1),
        "kv_cache_tokens": kv_tokens,
        "max_concurrency_tokens": int(conc.group(1).replace(",", "")) if conc else None,
        "max_concurrency_x": float(conc.group(2)) if conc else None,
        "error_line": err,
        "log": str(out / "server.log"),
    }
    (out / "result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="CONFIGS 的鍵")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--csv", default=str(REPO / "results/m1_capacity/capacity.csv"))
    args = ap.parse_args()

    if args.list or not args.config:
        for k, v in CONFIGS.items():
            print(f"  {k:22s} {v['model']:48s} kv={v['kv_dtype']:6s} {v['note']}")
        return 0

    cfg = CONFIGS[args.config]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}-m1-{args.config}"
    root = BIG / "runs" / run_id
    print(f"[m1] run_id={run_id} gpu={args.gpu} model={cfg['model']} kv={cfg['kv_dtype']}")

    rows = []

    def record(phase: str, max_len: int, r: dict, verdict: str) -> None:
        rows.append({
            "run_id": run_id, "ts": datetime.now().astimezone().isoformat(),
            "config": args.config, "model": cfg["model"],
            "weight_dtype": cfg["weight_dtype"], "kv_dtype": cfg["kv_dtype"],
            "phase": phase, "max_model_len": max_len,
            "server_ready": r["ready"], "verdict": verdict,
            "kv_cache_tokens": r["kv_cache_tokens"],
            "kv_gib": (round(r["kv_cache_tokens"] * kv_bytes_per_tok / 2**30, 3)
                       if r["kv_cache_tokens"] and kv_bytes_per_tok else None),
            "elapsed_s": r["elapsed_s"], "gpu": args.gpu,
            "error_line": r["error_line"] or "",
            "log": r["log"], "note": cfg["note"],
        })

    # 從已驗證的 config 讀 KV/token（results/m1_capacity/model_configs.json）
    kv_bytes_per_tok = None
    mc = REPO / "results/m1_capacity/model_configs.json"
    # 本地 no-DCA 目錄的 KV/token 與上游 repo 相同（只改了 config 的 DCA 欄位）
    alias = {str(BIG / "models/Qwen2.5-7B-Instruct-1M-noDCA"): "Qwen/Qwen2.5-7B-Instruct-1M"}
    want = alias.get(cfg["model"], cfg["model"])
    if mc.exists():
        for v in json.loads(mc.read_text()).values():
            if v.get("repo") == want:
                kv_bytes_per_tok = v.get("kv_bytes_per_token")
    if cfg["kv_dtype"] == "fp8" and kv_bytes_per_tok:
        kv_bytes_per_tok //= 2  # FP8 是 1 byte/elem，BF16 是 2

    # ---- 1. 量測 ----
    print(f"[m1] phase=measure  max_model_len={PROBE_LEN}")
    r = launch(cfg["model"], PROBE_LEN, args.gpu, cfg["extra"], root / "measure")
    print(f"     ready={r['ready']} kv_cache_tokens={r['kv_cache_tokens']} "
          f"({r['elapsed_s']}s) err={r['error_line']}")
    record("measure", PROBE_LEN, r, "OK" if r["ready"] else "FAIL")

    cliff = r["kv_cache_tokens"]
    if not r["ready"] or not cliff:
        print("[m1] 量測階段失敗 —— 停下來，不要往下猜。")
        write_csv(args.csv, rows)
        return 1

    # ---- 2. 驗證下界：懸崖本身應該起得來 ----
    at = cliff
    print(f"[m1] phase=verify_at  max_model_len={at}")
    r_at = launch(cfg["model"], at, args.gpu, cfg["extra"], root / "verify_at")
    print(f"     ready={r_at['ready']} ({r_at['elapsed_s']}s) err={r_at['error_line']}")
    record("verify_at", at, r_at, "OK" if r_at["ready"] else "UNEXPECTED_FAIL")

    # ---- 3. 驗證上界：超過懸崖應該失敗 ----
    over = int(cliff * OVERSHOOT)
    print(f"[m1] phase=verify_over  max_model_len={over}")
    r_ov = launch(cfg["model"], over, args.gpu, cfg["extra"], root / "verify_over")
    print(f"     ready={r_ov['ready']} ({r_ov['elapsed_s']}s) err={r_ov['error_line']}")
    record("verify_over", over, r_ov,
           "UNEXPECTED_OK" if r_ov["ready"] else "OK_FAILED_AS_EXPECTED")

    write_csv(args.csv, rows)

    print(f"\n[m1] === {args.config} ===")
    print(f"  懸崖（實測 KV 容量）: {cliff:,} tokens")
    if kv_bytes_per_tok:
        print(f"  ≈ {cliff * kv_bytes_per_tok / 2**30:.2f} GiB KV")
    print(f"  在懸崖啟動          : {'OK' if r_at['ready'] else '🔴 失敗（與預期不符）'}")
    print(f"  超出 {OVERSHOOT}× 啟動   : {'🔴 竟然成功（與預期不符）' if r_ov['ready'] else 'OK 如預期失敗'}")
    return 0


def write_csv(path: str, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    if not rows:
        return
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"[m1] appended {len(rows)} rows -> {p}")


if __name__ == "__main__":
    sys.exit(main())
