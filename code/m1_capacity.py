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

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parent.parent
# 平台由環境決定：平台 B（MI300X @ MLSteam）大檔在 NFS data、venv 在 workspace
PLATFORM = os.environ.get("TIARA_PLATFORM") or ("B" if Path("/opt/rocm").exists() else "A")
if PLATFORM == "B":
    BIG = Path(os.environ.get("TIARA_DATA", "/mlsteam/data/tiara"))
    VENV = Path(os.environ.get("TIARA_VENV", "/mlsteam/workspace/venv/tiara-v028"))
else:
    BIG = Path(os.environ.get("PAPER_HKV_BIG", "/ssd7/hungwei/paper-hkv"))
    VENV = BIG / "venv/vllm"

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
    # ══ 主力設定：AWQ-INT4 權重 ══════════════════════════════════════
    # EXPERIMENT_PLAN §2 的表格寫得很清楚：
    #   Qwen2.5-7B-Instruct-1M | AWQ-INT4 | ~315K | **主力設定**
    #   Qwen2.5-7B-Instruct-1M | BF16     | 較早  | 敏感度分析
    # 先前一路跑的是 BF16（敏感度那列），因為當時沒有 AWQ 權重。
    # 後果：BF16 權重吃掉 24GB 裡的 15GB，只剩 5.9GB 給 KV → 容量只有 48K
    #      → **整個實驗被鎖在短 context，量不到論文真正關心的 128K+ 區間**。
    # AWQ-INT4 權重只要 ~4.7GB → 剩 ~15GB 給 KV → 容量 122K–565K。
    "llama-awq": {
        "model": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        "weight_dtype": "AWQ-INT4",
        "kv_dtype": "auto",
        "extra": [],
        "note": "對照組主力。推估容量 ~122K，模型上限 131,072",
    },
    "llama-awq-kvfp8": {
        "model": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        "weight_dtype": "AWQ-INT4",
        "kv_dtype": "fp8",
        "extra": ["--kv-cache-dtype", "fp8"],
        "note": "推估容量 ~244K > 模型上限 131,072 → 可跑滿 128K",
    },
    "qwen-awq": {
        # ⚠️ 只有社群 AWQ（graelo，592 下載），無官方版。已知限制。
        # noDCA 變體由 code/make_nodca_model.py 產生（vLLM 0.28 跑不了 DCA）
        "model": str(BIG / "models/Qwen2.5-7B-Instruct-1M-AWQ-noDCA"),
        "weight_dtype": "AWQ-INT4",
        "kv_dtype": "auto",
        "extra": [],
        "note": "主力設定。推估容量 ~282K，模型上限 262,144（移除 DCA 後）",
    },
    "qwen-awq-kvfp8": {
        "model": str(BIG / "models/Qwen2.5-7B-Instruct-1M-AWQ-noDCA"),
        "weight_dtype": "AWQ-INT4",
        "kv_dtype": "fp8",
        "extra": ["--kv-cache-dtype", "fp8"],
        "note": "推估容量 ~565K > 模型上限 → 記憶體不再是瓶頸",
    },
    # ══ 512K 探測：刻意超出模型定址上限 ═══════════════════════════
    # 記憶體算得出來放得下（AWQ 權重 + FP8-KV 可容 565K），
    # 卡住的是模型只能定址 262,144。用 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 繞過。
    #
    # 🔴 **這個設定只能用於「容量」與「時間」，絕對不能用於品質。**
    #    vLLM 原始碼的警告寫得很清楚：超出 derived_max_model_len 的位置，
    #    RoPE 會產生 NaN。所以輸出內容是垃圾，但：
    #      * 容量  = KV pool 配得出來嗎 → 純記憶體問題，與 RoPE 無關 ✅
    #      * 時間  = prefill 的計算量只跟 token 數有關，與內容無關 ✅
    #      * 品質  = 🔴 完全無效，不要量、不要報
    #    每一列都會標 extrapolated=True，分析時必須排除品質欄位。
    "qwen-awq-512k": {
        "model": str(BIG / "models/Qwen2.5-7B-Instruct-1M-AWQ-noDCA"),
        "weight_dtype": "AWQ-INT4",
        "kv_dtype": "fp8",
        "extra": ["--kv-cache-dtype", "fp8"],
        "env": {"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"},
        "probe_len": 524288,
        "note": "🔴 外推到 512K：只驗容量與時間，品質無效（RoPE 超界會 NaN）",
    },
    # ── MLA：論文動作空間的第三個 κ 點 ────────────────────────────────
    # Llama(GQA) 128 KiB/tok、Qwen(GQA) 56 KiB/tok、DeepSeek-V2-Lite(MLA) 30.4 KiB/tok
    #   = 27 層 × (kv_lora_rank 512 + qk_rope_head_dim 64) × 2 bytes
    # 同一張卡上 4.2× 的 κ 跨度，而且 MLA 是 2026 年長上下文的主流架構
    # （DeepSeek-V3 70 KB/tok vs Llama-3.1-405B 516 KB/tok，93% 壓縮）。
    # BF16 權重 31 GB 放不下 24 GB，所以只有 AWQ 版本可用。
    # 預期：AWQ 權重 ~7.8 GB → 剩 ~13.4 GiB 給 KV → 約 462K tokens
    #   → **單請求容量壓力消失，瓶頸完全轉到 PCIe**，正是論文演算法的目標情境。
    "mla-awq": {
        "model": "TechxGenus/DeepSeek-V2-Lite-Chat-AWQ",
        "weight_dtype": "AWQ-INT4",
        "kv_dtype": "auto",
        "extra": ["--trust-remote-code"],
        "note": "MLA 架構，KV/token 只有 GQA 的 1/4.2",
    },
    "mla-awq-kvfp8": {
        "model": "TechxGenus/DeepSeek-V2-Lite-Chat-AWQ",
        "weight_dtype": "AWQ-INT4",
        "kv_dtype": "fp8",
        "extra": ["--trust-remote-code", "--kv-cache-dtype", "fp8"],
        "note": "MLA + FP8 KV，容量的理論上限",
    },
}

# ══ 平台 B（單張 MI300X 192 GB）：BF16 權重，選模見 results/model_selection/ ══════
# 2026-09-15 使用者決定：dense 與 MoE，小模型與 30B 級都做；hybrid 暫不做。
# vLLM v0.28.0（與平台 A 同版）：KV dtype 有 auto / fp8 / int8_per_token_head / int4_per_token_head。
CONFIGS_B: dict[str, dict] = {
    "b-llama8b-bf16": {"model": "unsloth/Llama-3.1-8B-Instruct", "weight_dtype": "BF16", "kv_dtype": "auto",
                       "extra": [], "category": "dense",
                       "note": "與平台 A llama-bf16 同架構（官方 repo gated，用 config 逐欄相同的公開鏡像）"},
    "b-llama8b-bf16-kvfp8": {"model": "unsloth/Llama-3.1-8B-Instruct", "weight_dtype": "BF16", "kv_dtype": "fp8",
                             "extra": ["--kv-cache-dtype", "fp8"], "category": "dense", "note": "FP8 KV 容量倍數"},
    "b-ultralong8b-1m-bf16": {"model": "nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct", "weight_dtype": "BF16",
                              "kv_dtype": "auto", "extra": ["--dtype", "bfloat16"], "category": "dense",
                              "note": "原生 1M 的 dense 錨點。repo 以 F32 發布，載入時轉 BF16"},
    "b-qwen7b-1m-bf16": {"model": str(BIG / "models/Qwen2.5-7B-Instruct-1M-noDCA"), "weight_dtype": "BF16",
                         "kv_dtype": "auto", "extra": [], "category": "dense",
                         "note": "與平台 A 主力同模型。0.19.1 無 DCA 後端 → noDCA 變體，上限 262,144"},
    "b-qwen3-30b-a3b-bf16": {"model": "Qwen/Qwen3-30B-A3B-Instruct-2507", "weight_dtype": "BF16", "kv_dtype": "auto",
                             "extra": [], "category": "moe", "note": "MoE 主選。原生 262,144"},
    "b-qwen3-30b-a3b-bf16-kvfp8": {"model": "Qwen/Qwen3-30B-A3B-Instruct-2507", "weight_dtype": "BF16",
                                   "kv_dtype": "fp8", "extra": ["--kv-cache-dtype", "fp8"], "category": "moe",
                                   "note": "MoE + FP8 KV"},
    "b-seedoss36b-bf16": {"model": "ByteDance-Seed/Seed-OSS-36B-Instruct", "weight_dtype": "BF16", "kv_dtype": "auto",
                          "extra": [], "category": "dense",
                          "note": "dense 30B 級主選。估算懸崖 ~399K < 原生 524,288 → 純記憶體限制"},
    "b-seedoss36b-bf16-kvfp8": {"model": "ByteDance-Seed/Seed-OSS-36B-Instruct", "weight_dtype": "BF16",
                                "kv_dtype": "fp8", "extra": ["--kv-cache-dtype", "fp8"], "category": "dense",
                                "note": "dense 30B + FP8 KV"},
    # ── 13–15B 級（2026-09-16 使用者要求加入）────────────────────────────
    # Qwen2.5-14B 是這一級在論文裡最常見的（490 篇中 29 篇，5.9%，排第 9；
    # results/model_selection/paper_model_frequency_min3.csv），而 -1M 變體是
    # 這一級**上下文最長**的（1,010,000）。DCA 同 7B 版要移除（vLLM v1 無 DCA 後端）。
    "b-qwen14b-1m-bf16": {"model": str(BIG / "models/Qwen2.5-14B-Instruct-1M-noDCA"), "weight_dtype": "BF16",
                          "kv_dtype": "auto", "extra": [], "category": "dense",
                          "note": "13–15B 級主選。48 層 × 8 KV head × 128 = 192 KiB/token。noDCA 後上限 262,144"},
    # 不同家族的 12B 對照：原生 131,072（不需 YaRN，符合禁令），40 層 × 8 × 128 = 160 KiB/token
    "b-mistral-nemo12b-bf16": {"model": "mistralai/Mistral-Nemo-Instruct-2407", "weight_dtype": "BF16",
                               "kv_dtype": "auto", "extra": [], "category": "dense",
                               "note": "13–15B 級的第二個家族（Mistral/NVIDIA 合作）。原生 131,072，無 rope_scaling"},
}
for _k in [k for k in CONFIGS_B if k.endswith("-bf16")]:
    for _tag, _dt in (("kvfp8", "fp8"), ("kvint8", "int8_per_token_head"), ("kvint4", "int4_per_token_head")):
        if f"{_k}-{_tag}" in CONFIGS_B:
            continue
        _c = dict(CONFIGS_B[_k]); _c["kv_dtype"] = _dt
        _c["extra"] = CONFIGS_B[_k]["extra"] + ["--kv-cache-dtype", _dt]; _c["note"] = f"{_dt} 容量倍數（含 scale 中繼資料）"
        CONFIGS_B[f"{_k}-{_tag}"] = _c
if PLATFORM == "B":
    CONFIGS = CONFIGS_B


def model_max_positions(model: str) -> int | None:
    """模型可定址長度（config 的 max_position_embeddings）。本地路徑或 HF 快取皆可。"""
    cand = [Path(model) / "config.json"]
    hub = Path(os.environ.get("HF_HOME", str(BIG / "hf-cache"))) / "hub" / ("models--" + model.replace("/", "--")) / "snapshots"
    if hub.exists():
        cand += sorted(hub.glob("*/config.json"))
    for c in cand:
        if c.exists():
            cfg = json.loads(c.read_text())
            tc = cfg.get("text_config") or {}
            return tc.get("max_position_embeddings") or cfg.get("max_position_embeddings")
    return None


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


WATCHER = None   # main() 設定；launch() 關 server 時暫停它


def launch(model: str, max_len: int, gpu: int, extra: list[str], out: Path,
           timeout: int = 900, extra_env: dict | None = None) -> dict:
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
    env.update(extra_env or {})
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if PLATFORM == "B":
        env["HIP_VISIBLE_DEVICES"] = str(gpu)
        env["PATH"] = f"{VENV / 'bin'}:/opt/rocm/bin:{env.get('PATH', '')}"
        env.setdefault("HF_HOME", str(BIG / "hf-cache"))
    else:
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

    from contextlib import nullcontext
    from gpu_guard import wait_until_released
    with (WATCHER.pause(f"teardown {out.name}") if WATCHER else nullcontext()):
        if not died:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                p.wait(timeout=60)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    pass
        released, left = wait_until_released(gpu, timeout_s=300)
        if not released:
            print(f"[m1] 🔴 server 關閉 300 秒後 GPU 仍有 {left} MiB 被佔用")
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
                                       "CUDA out of memory", "HIP out of memory", "Error",
                                       "is larger than the maximum", "exceeds")):
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
    ap.add_argument("--csv", default=str(REPO / ("results/m1_capacity/capacity_mi300x.csv" if PLATFORM == "B"
                                                  else "results/m1_capacity/capacity.csv")))
    args = ap.parse_args()

    if args.list or not args.config:
        for k, v in CONFIGS.items():
            print(f"  {k:22s} {v['model']:48s} kv={v['kv_dtype']:6s} {v['note']}")
        return 0

    cfg = CONFIGS[args.config]
    from gpu_guard import GpuWatcher, foreign_on, wait_until_released
    wait_until_released(args.gpu, timeout_s=120)     # 上一個設定的 server 可能還在收記憶體
    pre = foreign_on(args.gpu)
    if pre:
        print(f"[m1] 🔴 GPU {args.gpu} 開跑前不乾淨：{pre} —— 不開始（CLAUDE.md §3）")
        return 5
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}-m1-{args.config}"
    root = Path(os.environ.get("TIARA_RUNS", str(BIG / "runs"))) / run_id
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
            "extrapolated": bool(cfg.get("env", {}).get("VLLM_ALLOW_LONG_MAX_MODEL_LEN")),
            "log": r["log"], "note": cfg["note"],
            "platform": PLATFORM, "category": cfg.get("category", ""), "vllm_version": vllm_version,
            "model_max_positions": mmax, "guard_verdict": "PENDING",
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

    mmax = model_max_positions(cfg["model"])
    vllm_version = subprocess.run([str(VENV / "bin/python"), "-c", "import vllm; print(vllm.__version__)"],
                                  capture_output=True, text=True, cwd="/tmp").stdout.strip().splitlines()[-1:] or ["UNKNOWN"]
    vllm_version = vllm_version[0]
    print(f"[m1] venv={VENV} vllm={vllm_version}")
    global WATCHER
    watcher = GpuWatcher(gpu=args.gpu, out_path=str(root / "gpu_guard.json"))
    watcher.__enter__()
    WATCHER = watcher

    # ---- 1. 量測 ----
    print(f"[m1] phase=measure  max_model_len={cfg.get('probe_len', PROBE_LEN)}")
    probe = cfg.get("probe_len", PROBE_LEN)
    r = launch(cfg["model"], probe, args.gpu, cfg["extra"], root / "measure",
               extra_env=cfg.get("env"))
    print(f"     ready={r['ready']} kv_cache_tokens={r['kv_cache_tokens']} "
          f"({r['elapsed_s']}s) err={r['error_line']}")
    record("measure", probe, r, "OK" if r["ready"] else "FAIL")

    cliff = r["kv_cache_tokens"]
    if not r["ready"] or not cliff:
        print("[m1] 量測階段失敗 —— 停下來，不要往下猜。")
        watcher.__exit__(None, None, None)
        for row in rows:
            row["guard_verdict"] = watcher.verdict()
            row["limited_by"] = ""
        write_csv(args.csv, rows)
        return 1

    # ---- 2. 驗證下界：懸崖本身應該起得來 ----
    # 懸崖超過模型可定址長度時，max_model_len 會被 vLLM 以「超出模型上限」拒絕，
    # 那不是記憶體訊號 → 夾到模型上限，並記 limited_by=model。
    limited_by = "memory"
    at = cliff
    if mmax and cliff > mmax:
        at, limited_by = mmax, "model"
    print(f"[m1] phase=verify_at  max_model_len={at} (limited_by={limited_by})")
    r_at = launch(cfg["model"], at, args.gpu, cfg["extra"], root / "verify_at",
                  extra_env=cfg.get("env"))
    print(f"     ready={r_at['ready']} kv={r_at['kv_cache_tokens']} ({r_at['elapsed_s']}s) err={r_at['error_line']}")
    record("verify_at", at, r_at, "OK" if r_at["ready"] else "UNEXPECTED_FAIL")

    # ---- 3. 驗證上界：超過懸崖應該失敗（只有記憶體是瓶頸時才有意義） ----
    r_ov = None
    if limited_by == "memory":
        # 用 verify_at 那次量到的 KV 容量當懸崖（KV pool 會隨 max_model_len 變動，見平台 A 發現 6）
        cliff2 = r_at["kv_cache_tokens"] or cliff
        over = int(cliff2 * OVERSHOOT)
        if mmax and over > mmax:
            print(f"[m1] phase=verify_over SKIPPED：{over} > 模型上限 {mmax}")
        else:
            print(f"[m1] phase=verify_over  max_model_len={over}")
            r_ov = launch(cfg["model"], over, args.gpu, cfg["extra"], root / "verify_over",
                          extra_env=cfg.get("env"))
            print(f"     ready={r_ov['ready']} ({r_ov['elapsed_s']}s) err={r_ov['error_line']}")
            record("verify_over", over, r_ov,
                   "UNEXPECTED_OK" if r_ov["ready"] else "OK_FAILED_AS_EXPECTED")
    else:
        print("[m1] phase=verify_over SKIPPED：懸崖受模型上限限制，不是記憶體")

    watcher.__exit__(None, None, None)
    for row in rows:
        row["guard_verdict"] = watcher.verdict()
        row["limited_by"] = limited_by
    if watcher.contaminated:
        (root / "CONTAMINATED").write_text(json.dumps(watcher.report(), indent=2))
        print(f"[m1] 🔴 {watcher.verdict()}：結果不寫進 results/（CLAUDE.md §3），run 目錄保留")
        return 3
    write_csv(args.csv, rows)

    print(f"\n[m1] === {args.config} ===")
    print(f"  懸崖（實測 KV 容量，max_model_len={probe}）: {cliff:,} tokens")
    if r_at["kv_cache_tokens"]:
        print(f"  懸崖（max_model_len={at}）: {r_at['kv_cache_tokens']:,} tokens")
    print(f"  模型可定址上限: {mmax}  → limited_by={limited_by}")
    print(f"  在懸崖啟動: {'OK' if r_at['ready'] else '🔴 失敗（與預期不符）'}")
    if r_ov is not None:
        print(f"  超出 {OVERSHOOT}× 啟動: {'🔴 竟然成功（與預期不符）' if r_ov['ready'] else 'OK 如預期失敗'}")
    return 0


def write_csv(path: str, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    if not rows:
        return
    fields = list(rows[0])
    if not new:
        head = next(csv.reader(p.open(newline="")), [])
        if head != fields:
            raise SystemExit(f"🔴 {p} 的欄位與這批資料不同，拒絕 append（會整排錯位）。舊檔先移走。")
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"[m1] appended {len(rows)} rows -> {p}")


if __name__ == "__main__":
    sys.exit(main())
