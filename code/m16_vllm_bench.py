#!/usr/bin/env python3
"""m16_vllm_bench.py — 第三階段 V0／F1／F2／I3：對 vLLM 0.28 的 OpenAI 相容 server 送工作負載、量 TTFT／TPOT。

子命令
  serve       只「組」`vllm serve` 指令與環境變數，寫進 run 目錄（serve_cmd.sh、serve_config.json、serve_cleanup.sh）。
              **不啟動**；啟動交給呼叫者（之後的 runner）：`setsid bash <RD>/serve_cmd.sh &`。
  wait-ready  等 /health 與 /v1/models 都回 200（--pid 給了就在 server 先死掉時立刻失敗）。
  run         讀 m16_workload.py 的 JSONL，送 /v1/completions（stream、temperature 0、ignore_eos），量每個請求。
              open-loop（prefix_overlap、mooncake）照 t_arrive；agentic 照「上一輪結束＋暫停」；
              shared_prefix_burst 照 seed → evict → burst 三段。
  v0check     V0 正確性：同一個 prompt 的「不命中／GPU 命中／CPU 載入（／fs 載入）」輸出逐 token 比對。
  count-log   數 server log 裡 OffloadingConnector 的延後訊息（F2；要 serve --connector-debug）。

client 端（以上全部）只用標準庫＋aiohttp，**不 import torch／vllm**：AMD 的 gpu_guard 把「本容器內開 /dev/kfd 的行程」
當入侵者，client 一碰 GPU runtime，正在跑的計時量測就要重量。GpuWatcher（gpu_guard.py）只在 run／v0check 的
--guard on 時才 import；它呼叫 amd-smi，不開 kfd 的行程也照樣可以用。設 M16_FORBID_GPU_IMPORTS=1 會在載入時裝上
import 阻擋器（m16_workload.install_gpu_import_blocker），測試一律這樣跑。

量測定義（client 端，time.perf_counter）
  t_send   送出 POST 的前一刻（JSON body 在 sleep 之前就序列化好，不算進 TTFT）
  TTFT     t_send → 第一個帶 choices 的 SSE chunk。vLLM 0.28 只在有新 token 時送 chunk
           （entrypoints/openai/completion/serving.py:374-381：delta 空就 continue），所以第一個 chunk＝第一個 token。
  TPOT     (最後一個 token chunk − 第一個 chunk) ÷ (n_out − 1)；n_out＝server 回的 usage.completion_tokens
  ITL_chunk 同一段時間 ÷ (chunk 數 − 1)（一個 chunk 可能含多個 token）
  E2E      t_send → 收到 [DONE]
  cached_tokens  server 回的 usage.prompt_tokens_details.cached_tokens（＝本機 prefix cache＋connector 載入，
           v1/metrics/stats.py:284）；要 serve 時加 --enable-prompt-tokens-details（serve 子命令一律加）。沒有就 NOT_AVAILABLE。
  goodput  TTFT ≤ SLO 的請求數 ÷ makespan（--slo-ttft 可給多個；--slo-tpot 給了就兩個都要滿足）。
  百分位   線性內插（同 numpy 預設 'linear'）。

/metrics（開跑前、結束後各抓一次；三段式另外在每段前後抓）：名稱以 vLLM 0.28 原始碼為準
  prefix cache    vllm:prefix_cache_queries／hits、vllm:external_prefix_cache_queries／hits（v1/metrics/loggers.py:584-628）
  prompt token    vllm:prompt_tokens、vllm:prompt_tokens_cached、vllm:prompt_tokens_by_source{source=local_compute|
                  local_cache_hit|external_kv_transfer}（loggers.py:670-702、stats.py:316-320）
  offload         vllm:kv_offload_load_bytes／load_time／store_bytes／store_time／allocation_failure、
                  vllm:kv_offload_lookup_async_delay_seconds（histogram）（kv_connector/v1/offloading/metrics.py:22-38）、
                  舊名 vllm:kv_offload_total_bytes{transfer_type=CPU_to_GPU|GPU_to_CPU}（同檔 :132-134）
  tiering         vllm:kv_offload_tiering_read_bytes／write_bytes／chunk_queries／chunk_hits{tier=…}（v1/kv_offload/tiering/base.py:37-56）
  prometheus_client 會在 counter 名稱後面加 _total（exposition.py：mtype=='counter' → name+'_total'）。
  🔴 規則 7：抓不到（HTTP 失敗、家族沒註冊）＝NOT_AVAILABLE，**不填 0**。家族有註冊（有 # TYPE 行）但沒有樣本＝
     「從沒被觀測過」→ 記 0 並標 REGISTERED_NO_SAMPLE（prometheus_client 的 labeled 子項是第一次觀測才建立）。

warm-up 預取（I3；只用在 agentic 工作負載）
  第 t 輪（t ≥ 1）的實際送出時間 t_act＝上一輪結束（client 收到 [DONE]）＋pause_t（工作負載的真值）。
  預測：pause_pred＝pause_t ×（1＋e），e ~ Uniform(−noise, ＋noise)（--warmup-noise；亂數由 --warmup-seed／session／turn 決定）。
  提前量：--warmup-lead "Ns"＝N 秒；"Nx"＝N × R(warm_prefix_len)，R 由 --restore-est 給（fixed:<s> 或 linear:<a>:<b_s_per_token>）。
  warm-up 送出時間 t_warm＝上一輪結束＋pause_pred−lead；早於上一輪結束就夾到上一輪結束（warm_clamped=1）。
  warm-up 請求＝本輪 prompt 的前 warm_prefix_len 個 token（上一輪 prompt＋上一輪輸出的替代 token，不含本輪新的工具輸出）、
  max_tokens＝1。實際請求照原定 t_act 送，**不等 warm-up**。
  warm_status（每個 warm-up 一個）：
    hidden   warm-up 在實際請求送出前完成，且實際請求的 cached_tokens ≥ 門檻
             （門檻＝floor(min(warm_prefix_len, prompt_len−1) / block)×block，--block-size 預設 16）
    late     實際請求送出時 warm-up 還沒完成（預測太晚，或 noise 讓實際送出提早）                → 算白搬
    evicted  warm-up 在送出前完成，但實際請求的 cached_tokens 低於門檻（搬上來又被擠掉）           → 算白搬
    unknown  沒有 cached_tokens（server 沒開 prompt token details）而且不是 late
    failed   warm-up 請求本身失敗
  白搬比例＝(late＋evicted) ÷ (總數−unknown−failed)，分子分母都寫進 summary。
  注意：hidden 只代表「送出時 KV 已在 GPU」，不代表 warm-up 真的搬了東西（可能本來就還在 GPU）；warm-up 自己的
  cached_tokens 有記。warm-up 多搬的位元組只能從 /metrics 的 kv_offload_load_bytes 差值（跟不開 warm-up 的格子比）得到。

三段式（F2，shared_prefix_burst）
  seed 依序送 → 等 --phase-gap-s → evict（--evict fill：依序送填充請求；reset：POST /reset_prefix_cache 只清 GPU；
  none：跳過）→ 等 --phase-gap-s → burst 全部同一時刻送。主指標只算 burst 段；每段前後各抓一次 /metrics。
  --server-log 給了 server 的 log 檔，就記下這次 run 在 log 裡的位元組區間，事後數延後訊息（count-log 同一套）：
    "Delaying request %s since some of its chunks are already being loaded"  offloading/scheduler.py:855-856（0.19.1 是 blocks，:136-137）
    "Delaying request %s since it still has in-flight transfers"              scheduler.py:964
    "Offloading manager delayed request %s as backend requested"             scheduler.py:830
    "Request %s hit %s offloaded tokens after %s GPU hit tokens"             scheduler.py:860-865
  這些都是 logger.debug：serve 要加 --connector-debug（VLLM_LOGGING_CONFIG_PATH 只把
  vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler 調到 DEBUG）。沒開就標 DEBUG_OFF，計數不能用。

污染（CLAUDE.md §3）
  --guard on（預設）：需要 TIARA_GPU（實體卡號，和 amd-smi 一致）。--guard-own-root 預設＝本行程的父行程（job 腳本）：
  vLLM server 必須是它的子孫，否則 server 自己會被 gpu_guard 算成同容器入侵者。
  contaminated → run 目錄寫 CONTAMINATED、summary 標記、**不寫進 --out-dir（results/）**、結束碼 4（時間欄作廢，要重量）。
  --guard off：只給 CPU 測試；每列 timing_valid=0、summary guard=DISABLED。
  M16_GUARD_STUB=contaminated：測試用替身，一律判污染（只能判髒、不能判乾淨，所以不會被拿來洗白）。

結束碼：0＝完成；2＝設定錯誤／server 不對；3＝有請求失敗（照規則 2，全部記錄後停）；4＝contaminated。
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import glob
import hashlib
import json
import math
import os
import random
import re
import shlex
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m16_workload as W  # noqa: E402  （純 Python；M16_FORBID_GPU_IMPORTS=1 時順便裝上阻擋器）

VENV = os.environ.get("M7_VLLM_VENV", "/mlsteam/workspace/venv/tiara-v028")
EXTRA_PATH = os.environ.get("M7_EXTRA_PATH", "/opt/rocm/bin")
USE_ROCR_FLAG = os.environ.get("M16_USE_ROCR_FLAG", "/mlsteam/data/tiara/USE_ROCR")   # 第二階段 44d_xgpu_check2 過了才會出現
MODELS = {   # 同 code/m10_vllm_ttft.py
    "llama": "/mlsteam/data/tiara/hf-cache/hub/models--unsloth--Llama-3.1-8B-Instruct/snapshots/*",
    "qwen": "/mlsteam/data/tiara/models/Qwen2.5-7B-Instruct-1M-noDCA",   # vLLM 0.28 ROCm 不支援 DCA（RUNLOG_MI300X 868-869）
}
GIB = 2 ** 30
NA = "NOT_AVAILABLE"
OFFLOAD_SCHED_LOGGER = "vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler"

CSV_COLS = ["run_id", "ts", "cfg_label", "cfg_model", "cfg_apc", "cfg_offload", "cfg_gpu_blocks", "cfg_kv_cache_gib",
            "cfg_gpu_mem_util", "cfg_rocm_patch", "cfg_attn_backend", "cfg_max_model_len", "workload_kind",
            "workload_sha", "warmup_lead", "warmup_noise", "req_id", "kind", "phase", "session", "turn", "group",
            "is_warmup", "prompt_len", "prompt_tokens_server", "shared_len", "warm_prefix_len", "max_tokens",
            "t_sched_s", "t_send_s", "send_lag_s", "ttft_s", "tpot_s", "itl_chunk_mean_s", "e2e_s", "t_end_s",
            "n_out", "n_chunks", "cached_tokens", "ok", "error", "warm_of", "warm_status", "warm_lead_s",
            "warm_pred_err", "warm_clamped", "warm_slack_s", "guard_verdict", "timing_valid"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_dir(explicit: str | None = None) -> tuple[str, str]:
    if explicit:
        rd = os.path.abspath(explicit)
        rid = os.environ.get("RUN_ID") or os.path.basename(rd.rstrip("/"))
    else:
        rid = os.environ.get("RUN_ID") or ("adhoc-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-m16")
        rd = os.path.join(os.environ.get("TIARA_RUNS", "/mlsteam/data/tiara/runs"), rid)
    os.makedirs(rd, exist_ok=True)
    return rd, rid


def die(msg: str, code: int = 2):
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


# ═════════════════════════════════ serve ═════════════════════════════════
def resolve_model(name: str) -> str:
    pat = os.environ.get(f"M7_{name.upper()}_GLOB") or MODELS[name]
    hits = sorted(glob.glob(pat))
    if not hits:
        die(f"找不到模型 {name}：{pat}")
    return hits[-1]


def parse_offload(spec: str, engine_id: str, block_size: int | None, store_threshold: int | None,
                  eviction: str) -> tuple[dict | None, dict]:
    """回傳 (kv_transfer_config 或 None, 說明)。寫法照 code/m9_f1_run.py:331-347、code/m8_vllm_drop.py:477-491。"""
    info: dict = {"spec": spec}
    if spec == "none":
        return None, info
    parts = spec.split(":")
    if parts[0] == "cpu" and len(parts) == 2:
        cpu_gib = float(parts[1])
        extra = {"spec_name": "CPUOffloadingSpec", "cpu_bytes_to_use": int(cpu_gib * GIB), "eviction_policy": eviction}
        info.update(kind="cpu", cpu_gib=cpu_gib)
    elif parts[0] == "tier" and len(parts) == 4:
        cpu_gib, fs_path, fs_gib = float(parts[1]), os.path.abspath(parts[2]), float(parts[3])
        extra = {"spec_name": "TieringOffloadingSpec", "cpu_bytes_to_use": int(cpu_gib * GIB),
                 "eviction_policy": eviction, "secondary_tiers": [{"type": "fs", "root_dir": fs_path}]}
        probe = fs_path
        while not os.path.exists(probe):
            probe = os.path.dirname(probe)
        st = os.statvfs(probe)
        free_gib = st.f_bavail * st.f_frsize / GIB
        info.update(kind="tier", cpu_gib=cpu_gib, fs_path=fs_path, fs_gib_requested=fs_gib, fs_free_gib_at_compose=round(free_gib, 1),
                    fs_capacity_enforced=False,
                    fs_capacity_note="vLLM 0.28 的 FileSystemTierManager.__init__ 沒有容量參數"
                                     "（v1/kv_offload/tiering/fs/manager.py:109-118）：fsGiB 只記錄＋開跑前檢查剩餘空間")
        if free_gib < fs_gib:
            die(f"fs 層 {fs_path} 只剩 {free_gib:.1f} GiB < 要求的 {fs_gib} GiB（vLLM 不會限制用量，可能塞爆共用磁碟）")
    else:
        die(f"--offload 格式：none｜cpu:<GiB>｜tier:<cpuGiB>:<fs_path>:<fsGiB>（收到 {spec}）")
    if block_size:
        extra["block_size"] = block_size
    if store_threshold is not None:
        extra["store_threshold"] = store_threshold
    cfg = {"kv_connector": "OffloadingConnector", "kv_role": "kv_both", "engine_id": engine_id,
           "kv_connector_extra_config": extra}
    info["shm_path"] = f"/dev/shm/vllm_offload_{engine_id}.mmap"   # v1/kv_offload/cpu/shared_offload_region.py:94
    return cfg, info


def connector_debug_logging_config() -> dict:
    """VLLM_LOGGING_CONFIG_PATH 用的 dictConfig：其他 vllm logger 照預設 INFO，只有 offloading scheduler 開 DEBUG。
    格式照 vllm/logger.py:22-25、41-71（DEFAULT_LOGGING_CONFIG）。"""
    return {"version": 1, "disable_existing_loggers": False,
            "formatters": {"vllm": {"class": "vllm.logging_utils.NewLineFormatter", "datefmt": "%m-%d %H:%M:%S",
                                    "format": "%(levelname)s %(asctime)s [%(fileinfo)s:%(lineno)d] %(message)s"}},
            "handlers": {"vllm": {"class": "logging.StreamHandler", "formatter": "vllm", "level": "DEBUG",
                                  "stream": "ext://sys.stdout"}},
            "loggers": {"vllm": {"handlers": ["vllm"], "level": "INFO", "propagate": False},
                        OFFLOAD_SCHED_LOGGER: {"level": "DEBUG"}}}


def build_serve(a, rd: str, rid: str) -> dict:
    if a.gpu_blocks is not None and a.kv_cache_gib is not None:
        die("--gpu-blocks 與 --kv-cache-gib 只能給一個")
    if a.apc == "off" and a.offload != "none":
        die("--apc off 搭配卸載沒有測過（F1 矩陣也不需要）；要做請先在 V0 單獨驗證")
    gpu = a.gpu if a.gpu is not None else os.environ.get("TIARA_GPU")
    if gpu is None or not str(gpu).isdigit():
        die("要指定 GPU：--gpu N 或環境變數 TIARA_GPU（實體卡號，和 amd-smi 一致）")
    model_path = resolve_model(a.model)
    engine_id = f"m16-{rid}"
    ktc, off_info = parse_offload(a.offload, engine_id, a.offload_block_size, a.store_threshold, a.eviction_policy)
    args = [model_path, "--host", "127.0.0.1", "--port", str(a.port), "--served-model-name", "m",
            "--max-model-len", str(a.max_model_len), "--seed", "0", "--enable-prompt-tokens-details",
            "--enable-prefix-caching" if a.apc == "on" else "--no-enable-prefix-caching"]
    if a.gpu_blocks is not None:
        args += ["--num-gpu-blocks-override", str(a.gpu_blocks)]
    if a.kv_cache_gib is not None:
        args += ["--kv-cache-memory-bytes", str(int(a.kv_cache_gib * GIB))]
    if a.gpu_mem_util is not None:
        args += ["--gpu-memory-utilization", str(a.gpu_mem_util)]
    if a.block_size is not None:
        args += ["--block-size", str(a.block_size)]
    if ktc is not None:
        args += ["--kv-transfer-config", json.dumps(ktc, separators=(",", ":"))]
    attn = a.attention_backend
    if a.rocm_attn_patch:
        if ktc is None:
            print("WARNING: --rocm-attn-patch 沒有 KV connector 時沒有作用（ROCm 本來就選 ROCM_ATTN）", file=sys.stderr)
        if attn not in (None, "ROCM_ATTN"):
            die("--rocm-attn-patch 一定要 ROCM_ATTN")
        attn = "ROCM_ATTN"
    if attn:
        args += ["--attention-backend", attn]
    if a.extra_args:
        args += shlex.split(a.extra_args)
    if a.rocm_attn_patch:
        cmd = [f"{VENV}/bin/python", os.path.join(HERE, "m16_serve_patched.py"), "serve"] + args
    else:
        cmd = [f"{VENV}/bin/vllm", "serve"] + args
    # 同 code/m7_env.sh：有 USE_ROCR 旗子時只設 ROCR（實體卡號）並 unset HIP／CUDA，否則 HIP／CUDA（RUNLOG_MI300X 15.16）
    use_rocr = os.path.exists(USE_ROCR_FLAG)
    dev_env = ({"ROCR_VISIBLE_DEVICES": str(gpu)} if use_rocr
               else {"HIP_VISIBLE_DEVICES": str(gpu), "CUDA_VISIBLE_DEVICES": str(gpu)})
    env = {"VLLM_SERVER_DEV_MODE": "1",            # /reset_prefix_cache 在 dev 路由（entrypoints/serve/dev/cache/api_router.py）
           **dev_env,
           # 平台 B 的值（code/m7_env.sh）。不沿用目前 shell 的 HF_HOME：這個 repo 的 .claude/settings.json 帶的是 3090 的 /ssd7 路徑
           "HF_HOME": os.environ.get("M16_HF_HOME", "/mlsteam/data/tiara/hf-cache"),
           "PYTHONUNBUFFERED": "1"}
    log_cfg_path = None
    if a.connector_debug:
        log_cfg_path = os.path.join(rd, "vllm_logging.json")
        json.dump(connector_debug_logging_config(), open(log_cfg_path, "w"), indent=1)
        env["VLLM_LOGGING_CONFIG_PATH"] = log_cfg_path
    log_path = os.path.join(rd, "vllm_server.log")
    cfg = dict(run_id=rid, created=now_iso(), model=a.model, model_path=model_path, apc=a.apc, offload=a.offload,
               offload_info=off_info, kv_transfer_config=ktc, gpu=int(gpu), gpu_blocks=a.gpu_blocks,
               kv_cache_gib=a.kv_cache_gib, gpu_mem_util=a.gpu_mem_util, block_size=a.block_size,
               max_model_len=a.max_model_len, rocm_attn_patch=bool(a.rocm_attn_patch), attention_backend=attn,
               connector_debug=bool(a.connector_debug), logging_config=log_cfg_path, extra_args=a.extra_args,
               port=a.port, url=f"http://127.0.0.1:{a.port}", engine_id=engine_id, server_log=log_path,
               venv=VENV, cmd=cmd, env=env, use_rocr=use_rocr, label=a.label)
    with open(os.path.join(rd, "serve_config.json"), "w") as f:
        json.dump(cfg, f, indent=1, ensure_ascii=False)
    lines = ["#!/usr/bin/env bash",
             f"# 由 m16_vllm_bench.py serve 產生（{now_iso()}，run_id={rid}）。只啟動 server：",
             "#   setsid bash serve_cmd.sh &   （exec 之後這個 PID 就是 server；kill -TERM -- -$PID 收掉整組）",
             "set -u",
             f"cd {shlex.quote(rd)}   # ROCm 當掉時 gpucore.* 掉在 cwd（m8_vllm_drop.py 的作法）"]
    lines.append("unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES" if use_rocr else "unset ROCR_VISIBLE_DEVICES")
    for k, v in env.items():
        lines.append(f"export {k}={shlex.quote(v)}")
    lines.append(f"export PATH={shlex.quote(VENV + '/bin')}:{shlex.quote(EXTRA_PATH) + ':' if EXTRA_PATH else ''}\"$PATH\"")
    if off_info.get("kind") == "tier":
        lines.append(f"mkdir -p {shlex.quote(off_info['fs_path'])}")
    lines.append("exec " + " ".join(shlex.quote(c) for c in cmd) + f" >> {shlex.quote(log_path)} 2>&1")
    sp = os.path.join(rd, "serve_cmd.sh")
    open(sp, "w").write("\n".join(lines) + "\n")
    clean = ["#!/usr/bin/env bash", "# server 停掉之後再跑：只刪這個 run 自己的東西", "set -u"]
    if "shm_path" in off_info:
        clean.append(f"rm -f {shlex.quote(off_info['shm_path'])}   # /dev/shm 殘留（m7_c6_vllm.py 結尾、RUNLOG_MI300X 發現 7、16）")
    if off_info.get("kind") == "tier":
        clean.append(f"du -sh {shlex.quote(off_info['fs_path'])} || true   # fs 層用量；要不要刪由 runner 決定")
    open(os.path.join(rd, "serve_cleanup.sh"), "w").write("\n".join(clean) + "\n")
    return cfg


def cmd_serve(a) -> int:
    rd, rid = run_dir(a.run_dir)
    cfg = build_serve(a, rd, rid)
    print(f"serve_cmd: {os.path.join(rd, 'serve_cmd.sh')}")
    print("command:", " ".join(shlex.quote(c) for c in cfg["cmd"]))
    print("（沒有啟動。啟動：setsid bash serve_cmd.sh & ；等：m16_vllm_bench.py wait-ready --url", cfg["url"], "）")
    return 0


# ═════════════════════════════════ HTTP 小工具（同步） ═════════════════════════════════
def http_get(url: str, timeout: float = 10.0) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read() if hasattr(e, "read") else b""


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:   # zombie 也算死了（code/m10 的 VRAM 帳本 e5b5676 踩過）
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def wait_ready(url: str, timeout: float = 900.0, pid: int | None = None, poll: float = 2.0) -> dict:
    t0 = time.time()
    last = ""
    while True:
        if pid is not None and not pid_alive(pid):
            raise RuntimeError(f"server 行程 {pid} 已經結束（看 vllm_server.log）")
        try:
            st, _ = http_get(url.rstrip("/") + "/health", timeout=5)
            if st == 200:
                st2, body = http_get(url.rstrip("/") + "/v1/models", timeout=5)
                if st2 == 200:
                    return json.loads(body)
                last = f"/v1/models HTTP {st2}"
            else:
                last = f"/health HTTP {st}"
        except Exception as e:  # noqa: BLE001  連不上就是還沒好
            last = f"{type(e).__name__}: {e}"
        if time.time() - t0 > timeout:
            raise TimeoutError(f"{timeout:.0f} s 內沒有 ready（最後：{last}）")
        time.sleep(poll)


def cmd_wait_ready(a) -> int:
    try:
        models = wait_ready(a.url, a.timeout, a.pid)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(json.dumps(models))
    return 0


# ═════════════════════════════════ /metrics ═════════════════════════════════
_SAMPLE_RE = re.compile(r'^([^\s{]+)(?:\{(.*)\})?\s+(\S+)(?:\s+\S+)?$')
_LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')
AGG_DROP_LABELS = {"engine", "model_name"}
METRIC_PREFIXES = ("vllm:prefix_cache_", "vllm:external_prefix_cache_", "vllm:prompt_tokens", "vllm:generation_tokens",
                   "vllm:num_preemptions", "vllm:request_success", "vllm:kv_offload_")
NAMED_SERIES = {   # 名稱見檔頭；counter 名稱後面的 _total 是 prometheus_client 加的
    "prefix_cache_queries": "vllm:prefix_cache_queries_total",
    "prefix_cache_hits": "vllm:prefix_cache_hits_total",
    "external_prefix_cache_queries": "vllm:external_prefix_cache_queries_total",
    "external_prefix_cache_hits": "vllm:external_prefix_cache_hits_total",
    "prompt_tokens": "vllm:prompt_tokens_total",
    "prompt_tokens_cached": "vllm:prompt_tokens_cached_total",
    "prompt_tokens_local_compute": 'vllm:prompt_tokens_by_source_total{source="local_compute"}',
    "prompt_tokens_local_cache_hit": 'vllm:prompt_tokens_by_source_total{source="local_cache_hit"}',
    "prompt_tokens_external_kv_transfer": 'vllm:prompt_tokens_by_source_total{source="external_kv_transfer"}',
    "generation_tokens": "vllm:generation_tokens_total",
    "num_preemptions": "vllm:num_preemptions_total",
    "offload_load_bytes": "vllm:kv_offload_load_bytes_total",
    "offload_load_time_s": "vllm:kv_offload_load_time_total",
    "offload_store_bytes": "vllm:kv_offload_store_bytes_total",
    "offload_store_time_s": "vllm:kv_offload_store_time_total",
    "offload_allocation_failure": "vllm:kv_offload_allocation_failure_total",
    "offload_lookup_async_delay_sum_s": "vllm:kv_offload_lookup_async_delay_seconds_sum",
    "offload_lookup_async_delay_count": "vllm:kv_offload_lookup_async_delay_seconds_count",
    "offload_legacy_bytes_cpu_to_gpu": 'vllm:kv_offload_total_bytes_total{transfer_type="CPU_to_GPU"}',
    "offload_legacy_bytes_gpu_to_cpu": 'vllm:kv_offload_total_bytes_total{transfer_type="GPU_to_CPU"}',
    "tiering_read_bytes": "vllm:kv_offload_tiering_read_bytes_total",
    "tiering_write_bytes": "vllm:kv_offload_tiering_write_bytes_total",
    "tiering_chunk_queries": "vllm:kv_offload_tiering_chunk_queries_total",
    "tiering_chunk_hits": "vllm:kv_offload_tiering_chunk_hits_total",
}


def parse_prom(text: str) -> dict:
    """{'types': {family: type}, 'series': {(name, frozenset(labels-除了 engine/model_name)): 加總值}}"""
    types: dict = {}
    series: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            p = line.split()
            if len(p) >= 4 and p[1] == "TYPE":
                types[p[2]] = p[3]
            continue
        m = _SAMPLE_RE.match(line)
        if not m:
            continue
        name, lab, val = m.group(1), m.group(2) or "", m.group(3)
        if not name.startswith(METRIC_PREFIXES) or name.endswith(("_bucket", "_created")):
            continue
        try:
            v = float(val)
        except ValueError:
            continue
        labels = frozenset((k, vv) for k, vv in _LABEL_RE.findall(lab) if k not in AGG_DROP_LABELS)
        series[(name, labels)] = series.get((name, labels), 0.0) + v
    return {"types": types, "series": series}


def _family_registered(types: dict, name: str) -> bool:
    if name in types:
        return True
    for suf in ("_sum", "_count"):
        if name.endswith(suf) and types.get(name[: -len(suf)]) in ("histogram", "summary"):
            return True
    return False


def _lookup(snap: dict | None, key: str) -> tuple[float | None, str]:
    if snap is None:
        return None, "SCRAPE_FAILED"
    m = re.match(r'^([^{]+)(?:\{(.*)\})?$', key)
    name, sel = m.group(1), dict(_LABEL_RE.findall(m.group(2) or ""))
    if not _family_registered(snap["types"], name):
        return None, "NOT_REGISTERED"
    tot, hit = 0.0, False
    for (n, labels), v in snap["series"].items():
        if n == name and all((k, vv) in labels for k, vv in sel.items()):
            tot += v
            hit = True
    if not hit:
        return 0.0, "REGISTERED_NO_SAMPLE"
    return tot, "OK"


def metrics_delta(before: dict | None, after: dict | None) -> dict:
    out: dict = {}
    for k, key in NAMED_SERIES.items():
        b, sb = _lookup(before, key)
        a_, sa = _lookup(after, key)
        if a_ is None or b is None:
            out[k] = {"delta": NA, "status": f"before={sb};after={sa}", "series": key}
        elif a_ < b:
            out[k] = {"delta": NA, "status": "COUNTER_DECREASED(server 重啟？)", "series": key, "before": b, "after": a_}
        else:
            out[k] = {"delta": a_ - b, "status": f"before={sb};after={sa}", "series": key, "before": b, "after": a_}

    def ratio(n, d):
        x, y = out[n]["delta"], out[d]["delta"]
        if x == NA or y == NA:
            return NA
        return "NO_QUERIES" if y == 0 else x / y
    der = {"local_prefix_hit_rate": ratio("prefix_cache_hits", "prefix_cache_queries"),
           "external_prefix_hit_rate": ratio("external_prefix_cache_hits", "external_prefix_cache_queries"),
           "cached_prompt_token_frac": ratio("prompt_tokens_cached", "prompt_tokens")}
    lb, lt = out["offload_load_bytes"]["delta"], out["offload_load_time_s"]["delta"]
    der["offload_load_gib_per_s_vllm"] = NA if NA in (lb, lt) else ("NO_LOADS" if lt == 0 else lb / lt / GIB)
    return {"series": out, "derived": der}


async def scrape(session, base: str) -> tuple[dict | None, str, str]:
    try:
        async with session.get(base + "/metrics") as r:
            txt = await r.text()
            if r.status != 200:
                return None, f"HTTP {r.status}", txt[:200]
            return parse_prom(txt), "OK", txt
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}", ""


# ═════════════════════════════════ server log（F2） ═════════════════════════════════
LOG_PATTERNS = {   # 字串以 vLLM 0.28 原始碼為準（offloading/scheduler.py 行號見檔頭）
    "delay_chunks_being_loaded": re.compile(r"Delaying request (\S+) since some of its (?:chunks|blocks) are already being loaded"),
    "delay_inflight_transfers": re.compile(r"Delaying request (\S+) since it still has in-flight transfers"),
    "delay_backend_requested": re.compile(r"Offloading manager delayed request (\S+) as backend requested"),
    "hit_offloaded_tokens": re.compile(r"Request (\S+) hit (\d+) offloaded tokens after (\d+) GPU hit tokens"),
}


def count_log(path: str, start: int = 0, end: int | None = None) -> dict:
    if not path or not os.path.exists(path):
        return {"status": NA, "reason": f"log 不存在：{path}"}
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(None if end is None else max(0, end - start))
    text = data.decode(errors="replace")
    out: dict = {"status": "OK", "path": path, "byte_range": [start, start + len(data)]}
    for k, rx in LOG_PATTERNS.items():
        ids = [m.group(1) for m in rx.finditer(text)]
        out[k] = {"lines": len(ids), "distinct_requests": len(set(ids))}
    out["debug_lines_from_offloading_scheduler"] = sum(
        1 for ln in text.splitlines() if "DEBUG" in ln and "scheduler.py" in ln)
    return out


def cmd_count_log(a) -> int:
    print(json.dumps(count_log(a.log, a.start, a.end), ensure_ascii=False, indent=1))
    return 0


# ═════════════════════════════════ guard ═════════════════════════════════
class _StubWatcher:
    """M16_GUARD_STUB=contaminated：測試用，一律判污染。"""

    contaminated = True

    def __init__(self, rd):
        self.rd = rd

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        json.dump(self.report(), open(os.path.join(self.rd, "gpu_guard.json"), "w"), indent=1)

    def verdict(self):
        return "CONTAMINATED_STUB"

    def report(self):
        return {"impl": "STUB", "contaminated": True, "verdict": self.verdict()}


def make_guard(mode: str, own_root: int | None, rd: str):
    stub = os.environ.get("M16_GUARD_STUB")
    if stub:
        if stub != "contaminated":
            die("M16_GUARD_STUB 只接受 contaminated（替身只能判髒）")
        return _StubWatcher(rd), "STUB"
    if mode == "off":
        return None, "DISABLED"
    gpu = os.environ.get("TIARA_GPU")
    if gpu is None or not gpu.isdigit():
        die("--guard on 需要 TIARA_GPU（實體卡號）。查不到就不能量時間（規則 7）；CPU 測試才用 --guard off")
    from gpu_guard import GpuWatcher   # 只有真的量時才 import（它只呼叫 amd-smi／nvidia-smi，不開 kfd）
    root = own_root if own_root is not None else os.getppid()
    return GpuWatcher(gpu=int(gpu), own_root=root, out_path=os.path.join(rd, "gpu_guard.json")), f"GpuWatcher(gpu={gpu},own_root={root})"


class _NullCtx:
    contaminated = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ═════════════════════════════════ 請求 ═════════════════════════════════
def pct(xs: list[float], q: float):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return "NO_DATA"
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def stats(xs: list) -> dict:
    v = [x for x in xs if isinstance(x, (int, float))]
    if not v:
        return {"n": 0, "p50": "NO_DATA", "p95": "NO_DATA", "p99": "NO_DATA", "mean": "NO_DATA", "max": "NO_DATA"}
    return {"n": len(v), "p50": pct(v, 50), "p95": pct(v, 95), "p99": pct(v, 99), "mean": sum(v) / len(v), "max": max(v)}


class Client:
    def __init__(self, base: str, model: str, rid: str, timeout_s: float, return_token_ids: bool = False,
                 logprobs: int | None = None):
        import aiohttp
        self.aiohttp = aiohttp
        self.base, self.model, self.rid, self.timeout_s = base.rstrip("/"), model, rid, timeout_s
        self.return_token_ids, self.logprobs = return_token_ids, logprobs
        self.session = None
        self.t0 = None

    async def __aenter__(self):
        a = self.aiohttp
        self.session = a.ClientSession(connector=a.TCPConnector(limit=0),
                                       timeout=a.ClientTimeout(total=None, sock_connect=30, sock_read=self.timeout_s))
        return self

    async def __aexit__(self, *exc):
        await self.session.close()

    def body(self, r: dict, max_tokens: int | None = None, prompt=None, rid_suffix: str = "") -> bytes:
        b = {"model": self.model, "prompt": prompt if prompt is not None else r["prompt_token_ids"],
             "max_tokens": int(max_tokens if max_tokens is not None else r["max_tokens"]), "temperature": 0.0,
             "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True},
             "request_id": f"{self.rid}-{r['req_id']}{rid_suffix}"}
        if self.return_token_ids:
            b["return_token_ids"] = True
        if self.logprobs is not None:
            b["logprobs"] = self.logprobs
            b["return_tokens_as_token_ids"] = True
        return json.dumps(b, separators=(",", ":")).encode()

    async def send(self, data: bytes) -> dict:
        """送一個 streaming completion。回傳時間都是相對 self.t0 的秒數。"""
        res = {"ok": 0, "error": "", "ttft_s": None, "tpot_s": None, "itl_chunk_mean_s": None, "e2e_s": None,
               "n_out": None, "n_chunks": 0, "cached_tokens": NA, "prompt_tokens_server": NA, "token_ids": [],
               "top_logprobs": []}
        t_send = time.perf_counter()
        res["t_send_s"] = t_send - self.t0
        first = last = None
        usage = None
        done = False
        try:
            async with self.session.post(self.base + "/v1/completions", data=data,
                                         headers={"Content-Type": "application/json"}) as resp:
                if resp.status != 200:
                    txt = await resp.text()
                    res["error"] = f"HTTP {resp.status}: {txt[:300]}"
                else:
                    buf = b""
                    async for chunk in resp.content.iter_any():
                        now = time.perf_counter()
                        buf += chunk
                        while not done:
                            i = buf.find(b"\n")
                            if i < 0:
                                break
                            line, buf = buf[:i].strip(), buf[i + 1:]
                            if not line.startswith(b"data:"):
                                continue
                            payload = line[5:].strip()
                            if payload == b"[DONE]":
                                done = True
                                break
                            obj = json.loads(payload)
                            if "error" in obj and not obj.get("choices"):
                                res["error"] = f"server error: {str(obj['error'])[:300]}"
                            ch = obj.get("choices") or []
                            if ch:
                                res["n_chunks"] += 1
                                if first is None:
                                    first = now
                                last = now
                                c = ch[0]
                                if c.get("token_ids"):
                                    res["token_ids"].extend(c["token_ids"])
                                lp = c.get("logprobs")
                                if lp and lp.get("top_logprobs"):
                                    res["top_logprobs"].extend(lp["top_logprobs"])
                            if obj.get("usage"):
                                usage = obj["usage"]
                        if done:
                            break
                    if not done and not res["error"]:
                        res["error"] = "stream 在 [DONE] 之前結束"
        except Exception as e:  # noqa: BLE001
            res["error"] = f"{type(e).__name__}: {e}"
        t_end = time.perf_counter()
        res["t_end_s"] = t_end - self.t0
        res["e2e_s"] = t_end - t_send
        if first is not None:
            res["ttft_s"] = first - t_send
        if usage:
            res["n_out"] = usage.get("completion_tokens")
            res["prompt_tokens_server"] = usage.get("prompt_tokens", NA)
            ptd = usage.get("prompt_tokens_details") or {}
            if ptd.get("cached_tokens") is not None:
                res["cached_tokens"] = ptd["cached_tokens"]
        if res["n_out"] is None and res["token_ids"]:
            res["n_out"] = len(res["token_ids"])
        if res["n_out"] and res["n_out"] > 1 and first is not None:
            res["tpot_s"] = (last - first) / (res["n_out"] - 1)
        if res["n_chunks"] > 1:
            res["itl_chunk_mean_s"] = (last - first) / (res["n_chunks"] - 1)
        if not res["error"] and first is not None:
            res["ok"] = 1
        elif not res["error"]:
            res["error"] = "沒有收到任何 token"
        return res

    async def reset(self, external: bool, tries: int = 40, gap: float = 0.5) -> dict:
        """POST /reset_prefix_cache（dev 路由，要 VLLM_SERVER_DEV_MODE=1）。有 block 被占用時回 success=false → 重試。"""
        url = f"{self.base}/reset_prefix_cache?reset_external={'true' if external else 'false'}"
        last = ""
        for k in range(tries):
            try:
                async with self.session.post(url) as r:
                    txt = await r.text()
                    if r.status == 404:
                        return {"ok": False, "tries": k + 1, "error": "404：server 沒開 VLLM_SERVER_DEV_MODE=1"}
                    if r.status == 200 and json.loads(txt).get("success"):
                        return {"ok": True, "tries": k + 1}
                    last = f"HTTP {r.status} {txt[:100]}"
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
            await asyncio.sleep(gap)
        return {"ok": False, "tries": tries, "error": last}

    async def models(self) -> dict:
        async with self.session.get(self.base + "/v1/models") as r:
            if r.status != 200:
                raise RuntimeError(f"/v1/models HTTP {r.status}")
            return json.loads(await r.text())


async def sleep_until(t_abs: float) -> None:
    d = t_abs - time.perf_counter()
    if d > 0:
        await asyncio.sleep(d)


def parse_lead(spec: str | None):
    if spec is None:
        return None
    s = spec.strip()
    if s.endswith("x"):
        return ("x", float(s[:-1]))
    if s.endswith("s"):
        return ("s", float(s[:-1]))
    return ("s", float(s))


def parse_restore(spec: str | None):
    if spec is None:
        return None
    k, _, rest = spec.partition(":")
    if k == "fixed":
        v = float(rest)
        return lambda n: v
    if k == "linear":
        a_, b_ = (float(x) for x in rest.split(":"))
        return lambda n: a_ + b_ * n
    raise ValueError(f"--restore-est 只接受 fixed:<s>、linear:<a>:<b>（收到 {spec}）")


# ═════════════════════════════════ run ═════════════════════════════════
def load_server_config(path: str | None) -> dict:
    if not path:
        return {}
    return json.load(open(path))


def cfg_cols(sc: dict, label: str | None) -> dict:
    g = lambda k: sc.get(k, "NOT_GIVEN") if sc else "NOT_GIVEN"   # noqa: E731
    return dict(cfg_label=label or sc.get("label") or "", cfg_model=g("model"), cfg_apc=g("apc"),
                cfg_offload=g("offload"), cfg_gpu_blocks=g("gpu_blocks"), cfg_kv_cache_gib=g("kv_cache_gib"),
                cfg_gpu_mem_util=g("gpu_mem_util"), cfg_rocm_patch=g("rocm_attn_patch"),
                cfg_attn_backend=g("attention_backend"), cfg_max_model_len=g("max_model_len"))


def plan_dry(reqs: list[dict], a) -> list[dict]:
    """--dry-run：不連 server，只排時間（服務時間假設＝--dry-service-s）。"""
    kind = reqs[0]["kind"]
    rows = []
    lead = parse_lead(a.warmup_lead)
    rest = parse_restore(a.restore_est)
    if lead and kind != "agentic":
        die("--warmup-lead 只用在 agentic 工作負載")
    if lead and lead[0] == "x" and rest is None:
        die("--warmup-lead 用倍數（Nx）時要給 --restore-est")
    if kind == "agentic":
        by: dict = {}
        for r in reqs:
            by.setdefault(r["session"], []).append(r)
        for s, turns in sorted(by.items()):
            prev_end = None
            for tr in sorted(turns, key=lambda x: x["turn"]):
                t = tr["t_ready"] if tr["turn"] == 0 else prev_end + tr["t_ready"]
                if lead and tr["turn"] >= 1 and tr["warm_prefix_len"] > 0:
                    e = warm_err(a, s, tr["turn"])
                    lv = lead[1] if lead[0] == "s" else lead[1] * rest(tr["warm_prefix_len"])
                    tw = max(prev_end, prev_end + tr["t_ready"] * (1 + e) - lv)
                    rows.append(dict(req_id=tr["req_id"] + "-warm", is_warmup=1, t_sched_s=tw,
                                     prompt_len=tr["warm_prefix_len"], warm_lead_s=lv, warm_pred_err=e))
                rows.append(dict(req_id=tr["req_id"], is_warmup=0, t_sched_s=t, prompt_len=len(tr["prompt_token_ids"])))
                prev_end = t + a.dry_service_s
    elif kind == "shared_prefix_burst":
        t = 0.0
        for ph in ("seed", "evict", "burst"):
            for r in reqs:
                if r.get("phase") == ph:
                    rows.append(dict(req_id=r["req_id"], is_warmup=0, t_sched_s=t, prompt_len=len(r["prompt_token_ids"]),
                                     phase=ph))
                    if ph != "burst":
                        t += a.dry_service_s
            t += a.phase_gap_s
    else:
        for r in reqs:
            rows.append(dict(req_id=r["req_id"], is_warmup=0, t_sched_s=r["t_arrive"] * a.time_scale,
                             prompt_len=len(r["prompt_token_ids"])))
    return rows


def warm_err(a, session, turn) -> float:
    if not a.warmup_noise:
        return 0.0
    return random.Random(f"{a.warmup_seed}|{session}|{turn}").uniform(-a.warmup_noise, a.warmup_noise)


def validate_workload(reqs: list[dict], max_model_len) -> str:
    if not reqs:
        return "工作負載是空的"
    kinds = {r["kind"] for r in reqs}
    if len(kinds) != 1:
        return f"一個檔案只能有一種 kind：{kinds}"
    ids = [r["req_id"] for r in reqs]
    if len(set(ids)) != len(ids):
        return "req_id 重複"
    if isinstance(max_model_len, int):
        bad = [r["req_id"] for r in reqs if len(r["prompt_token_ids"]) + int(r["max_tokens"]) > max_model_len]
        if bad:
            return f"{len(bad)} 個請求超過 max_model_len={max_model_len}（例：{bad[:3]}）"
    return ""


def make_row(base: dict, r: dict, res: dict, is_warm: bool, t_sched: float, **extra) -> dict:
    row = dict(base)
    row.update(req_id=r["req_id"] + ("-warm" if is_warm else ""), kind=r["kind"], phase=r.get("phase", ""),
               session="" if r.get("session") is None else r["session"],
               turn="" if r.get("turn") is None else r["turn"], group="" if r.get("group") is None else r["group"],
               is_warmup=int(is_warm), shared_len=r.get("shared_len", ""), max_tokens=extra.pop("max_tokens", r["max_tokens"]),
               warm_prefix_len=r.get("warm_prefix_len", ""), t_sched_s=t_sched)
    for k in ("t_send_s", "ttft_s", "tpot_s", "itl_chunk_mean_s", "e2e_s", "t_end_s", "n_out", "n_chunks",
              "cached_tokens", "ok", "error", "prompt_tokens_server"):
        row[k] = res.get(k)
    row["prompt_len"] = extra.pop("prompt_len", len(r["prompt_token_ids"]))
    row["send_lag_s"] = (res["t_send_s"] - t_sched) if res.get("t_send_s") is not None else None
    row.update(extra)
    return row


async def run_async(a, reqs: list[dict], wmeta: dict, sc: dict, rd: str, rid: str, guard_desc: str) -> dict:
    base_url = a.url or sc.get("url")
    if not base_url:
        die("沒有 --url，--server-config 裡也沒有 url")
    kind = reqs[0]["kind"]
    lead = parse_lead(a.warmup_lead)
    rest = parse_restore(a.restore_est)
    if lead and kind != "agentic":
        die("--warmup-lead 只用在 agentic 工作負載")
    if lead and lead[0] == "x" and rest is None:
        die("--warmup-lead 用倍數（Nx）時要給 --restore-est")
    info: dict = {"phases": [], "resets": []}
    rows: list[dict] = []
    base_cols = dict(run_id=rid, **cfg_cols(sc, a.label), workload_kind=kind, workload_sha=wmeta.get("sha256", NA),
                     warmup_lead=a.warmup_lead or "", warmup_noise=a.warmup_noise if lead else "")
    async with Client(base_url, a.model_name, rid, a.timeout_s) as cl:
        mods = await cl.models()
        ids = [m.get("id") for m in mods.get("data", [])]
        if a.model_name not in ids:
            die(f"server 的模型是 {ids}，不是 {a.model_name}")
        srv_mml = next((m.get("max_model_len") for m in mods["data"] if m.get("id") == a.model_name), None)
        info["server_models"] = mods
        if sc.get("max_model_len") is not None and srv_mml is not None and int(srv_mml) != int(sc["max_model_len"]):
            die(f"server 回報 max_model_len={srv_mml}，serve_config 是 {sc['max_model_len']} → 連到別的 server？")
        msg = validate_workload(reqs, int(srv_mml) if srv_mml is not None else None)
        if msg:
            die(msg)
        cl.t0 = time.perf_counter()
        # server 暖機（Triton JIT），不計；和工作負載無關的 token
        for i in range(a.server_warmup):
            L = 512 if i == 0 else min(8192, max(len(r["prompt_token_ids"]) for r in reqs))
            rr = {"req_id": f"srvwarm{i}", "prompt_token_ids": W._toks(random.Random(f"srvwarm|{i}"), L, 1000, 120000),
                  "max_tokens": 2}
            res = await cl.send(cl.body(rr))
            info.setdefault("server_warmup", []).append({k: res[k] for k in ("ok", "error", "ttft_s", "e2e_s")})
            if not res["ok"]:
                die(f"server 暖機失敗：{res['error']}", 3)
        has_offload = sc.get("offload", "none") not in ("none", "NOT_GIVEN") if sc else a.reset_external
        if a.reset_before == "on":
            rr = await cl.reset(external=bool(has_offload or a.reset_external))
            info["resets"].append(dict(when="before", **rr))
            if not rr["ok"]:
                die(f"開跑前 /reset_prefix_cache 失敗：{rr}（--reset-before off 可略過，但格子就不是乾淨開始）", 3)
        await asyncio.sleep(a.settle_s)
        log_start = os.path.getsize(a.server_log) if a.server_log and os.path.exists(a.server_log) else None
        n_ol = len(reqs) if a.max_requests is None else min(a.max_requests, len(reqs))
        # 大 prompt 的 JSON 序列化（64×32K token 約零點幾秒）放在計時起點之前，不讓它變成 send_lag
        pre_bodies = {r["req_id"]: cl.body(r) for r in (reqs[:n_ol] if kind in ("prefix_overlap", "mooncake") else
                                                         [x for x in reqs if x.get("phase") == "burst"])}
        m_before, st_b, raw_b = await scrape(cl.session, cl.base)
        open(os.path.join(rd, "metrics_before.txt"), "w").write(raw_b)
        cl.t0 = time.perf_counter()
        info["t0_wall"] = now_iso()

        async def one(r, t_sched_abs, is_warm=False, data=None, **extra):
            if data is None:
                data = cl.body(r)
            await sleep_until(t_sched_abs)
            res = await cl.send(data)
            row = make_row(base_cols, r, res, is_warm, t_sched_abs - cl.t0, **extra)
            rows.append(row)
            return row, res

        if kind in ("prefix_overlap", "mooncake"):
            n = n_ol
            sem = asyncio.Semaphore(a.max_inflight) if a.max_inflight else None
            bodies = [pre_bodies[r["req_id"]] for r in reqs[:n]]

            async def ol(r, d):
                ts = cl.t0 + r["t_arrive"] * a.time_scale
                if sem:
                    await sleep_until(ts)
                    async with sem:
                        return await one(r, ts, data=d)
                return await one(r, ts, data=d)
            await asyncio.gather(*(ol(r, d) for r, d in zip(reqs[:n], bodies)))
        elif kind == "agentic":
            by: dict = {}
            for r in reqs:
                by.setdefault(r["session"], []).append(r)
            warm_rows: dict = {}

            async def sess(s, turns):
                prev_end = None
                failed = False
                wtasks = []
                for tr in sorted(turns, key=lambda x: x["turn"]):
                    if failed:
                        rows.append(make_row(base_cols, tr, {"ok": 0, "error": "SKIPPED_PREV_TURN_FAILED"}, False, ""))
                        continue
                    target = cl.t0 + tr["t_ready"] if tr["turn"] == 0 else prev_end + tr["t_ready"]
                    wtask = None
                    if lead and tr["turn"] >= 1 and tr["warm_prefix_len"] > 0:
                        e = warm_err(a, s, tr["turn"])
                        lv = lead[1] if lead[0] == "s" else lead[1] * rest(tr["warm_prefix_len"])
                        tw = prev_end + tr["t_ready"] * (1 + e) - lv
                        clamped = int(tw < prev_end)
                        tw = max(tw, prev_end)
                        wl = tr["warm_prefix_len"]
                        wd = cl.body(tr, max_tokens=1, prompt=tr["prompt_token_ids"][:wl], rid_suffix="-warm")
                        wtask = asyncio.create_task(one(tr, tw, True, wd, max_tokens=1, prompt_len=wl,
                                                        warm_of=tr["req_id"], warm_lead_s=lv, warm_pred_err=e,
                                                        warm_clamped=clamped))
                        wtasks.append(wtask)
                    body = cl.body(tr)
                    row, res = await one(tr, target, False, body)
                    if wtask is not None:
                        warm_rows[tr["req_id"]] = (wtask, row)
                    if res["ok"]:
                        prev_end = cl.t0 + res["t_end_s"]
                    else:
                        failed = True
                if wtasks:
                    await asyncio.gather(*wtasks)
            await asyncio.gather(*(sess(s, t) for s, t in sorted(by.items())))
            bs = a.block_size
            for rq, (wt, arow) in warm_rows.items():
                wrow, wres = wt.result()
                arow["warm_of"] = wrow["req_id"]
                plen = arow["prompt_len"]
                thr = (min(int(wrow["prompt_len"]), plen - 1) // bs) * bs
                if not wres["ok"]:
                    st = "failed"
                elif wres["t_end_s"] > arow["t_send_s"]:
                    st = "late"
                elif not isinstance(arow["cached_tokens"], int):
                    st = "unknown"
                elif arow["cached_tokens"] >= thr:
                    st = "hidden"
                else:
                    st = "evicted"
                slack = (arow["t_send_s"] - wres["t_end_s"]) if wres["ok"] else None
                for rr_ in (wrow, arow):
                    rr_["warm_status"] = st
                    rr_["warm_slack_s"] = slack
                arow["warm_lead_s"], arow["warm_pred_err"], arow["warm_clamped"] = (
                    wrow["warm_lead_s"], wrow["warm_pred_err"], wrow["warm_clamped"])
        elif kind == "shared_prefix_burst":
            evict = a.evict or ("fill" if any(r.get("phase") == "evict" for r in reqs) else "none")
            info["evict_mode"] = evict
            phases = [x for x in a.phases.split(",") if x]
            if not phases or any(x not in ("seed", "evict", "burst") for x in phases) or phases != sorted(
                    phases, key=["seed", "evict", "burst"].index):
                die(f"--phases 只能是 seed,evict,burst 的依序子集（收到 {a.phases}）")
            info["phases_run"] = phases
            for ph in phases:
                pr = [r for r in reqs if r.get("phase") == ph]
                pm_b, _, _ = await scrape(cl.session, cl.base)
                ph_info = {"phase": ph, "n": len(pr), "t_start_s": time.perf_counter() - cl.t0}
                if ph == "evict" and evict == "reset":
                    ph_info["reset"] = await cl.reset(external=False)
                    info["resets"].append(dict(when="evict", **ph_info["reset"]))
                    if not ph_info["reset"]["ok"]:
                        die(f"evict 段 /reset_prefix_cache 失敗：{ph_info['reset']}", 3)
                    ph_info["n_sent"] = 0
                elif ph == "evict" and evict == "none":
                    ph_info["n_sent"] = 0
                elif ph == "burst":
                    bodies = [pre_bodies[r["req_id"]] for r in pr]
                    ts = time.perf_counter() + 0.05
                    await asyncio.gather(*(one(r, ts, data=d) for r, d in zip(pr, bodies)))
                    ph_info["n_sent"] = len(pr)
                else:
                    for r in pr:
                        await one(r, time.perf_counter())
                    ph_info["n_sent"] = len(pr)
                ph_info["t_end_s"] = time.perf_counter() - cl.t0
                await asyncio.sleep(a.phase_gap_s)
                pm_a, _, _ = await scrape(cl.session, cl.base)
                ph_info["metrics"] = metrics_delta(pm_b, pm_a)
                info["phases"].append(ph_info)
        else:
            die(f"不認得的工作負載 kind：{kind}")
        info["t_last_s"] = time.perf_counter() - cl.t0
        await asyncio.sleep(a.settle_s)
        m_after, st_a, raw_a = await scrape(cl.session, cl.base)
        open(os.path.join(rd, "metrics_after.txt"), "w").write(raw_a)
        log_end = os.path.getsize(a.server_log) if a.server_log and os.path.exists(a.server_log) else None
    info["metrics_scrape"] = {"before": st_b, "after": st_a}
    info["metrics"] = metrics_delta(m_before, m_after)
    if a.server_log:
        if log_start is None:
            info["connector_log"] = {"status": NA, "reason": f"開跑時 log 不存在：{a.server_log}"}
        else:
            cl_ = count_log(a.server_log, log_start, log_end)
            if not sc.get("connector_debug"):
                cl_["status"] = "DEBUG_OFF（serve 沒加 --connector-debug：延後訊息是 DEBUG，計數不能用）"
            info["connector_log"] = cl_
    else:
        info["connector_log"] = {"status": NA, "reason": "沒有給 --server-log"}
    info["rows"] = rows
    return info


def summarize(rows: list[dict], info: dict, a, kind: str) -> dict:
    meas = [r for r in rows if not r["is_warmup"] and (kind != "shared_prefix_burst" or r["phase"] == "burst")]
    ok = [r for r in meas if r["ok"] == 1]
    s: dict = {"n_measured": len(meas), "n_ok": len(ok), "n_failed": len(meas) - len(ok),
               "n_failed_all_rows": sum(1 for r in rows if r["ok"] != 1),
               "ttft_s": stats([r["ttft_s"] for r in ok]), "tpot_s": stats([r["tpot_s"] for r in ok]),
               "itl_chunk_mean_s": stats([r["itl_chunk_mean_s"] for r in ok]), "e2e_s": stats([r["e2e_s"] for r in ok]),
               "send_lag_s": stats([r["send_lag_s"] for r in ok])}
    s["n_out_mismatch"] = sum(1 for r in ok if r["n_out"] != r["max_tokens"])
    s["prompt_len_mismatch"] = sum(1 for r in ok if isinstance(r["prompt_tokens_server"], int)
                                   and r["prompt_tokens_server"] != r["prompt_len"])
    if ok:
        t_a = min(r["t_send_s"] for r in ok)
        t_b = max(r["t_end_s"] for r in ok)
        mk = t_b - t_a
        n_out = sum(r["n_out"] or 0 for r in ok)
        s["throughput"] = {"makespan_s": mk, "out_tok_per_s": n_out / mk if mk > 0 else NA,
                           "req_per_s": len(ok) / mk if mk > 0 else NA, "n_out_total": n_out}
        gp = []
        for slo in a.slo_ttft or []:
            meet = [r for r in ok if r["ttft_s"] <= slo and (a.slo_tpot is None or r["tpot_s"] is None
                                                              or r["tpot_s"] <= a.slo_tpot)]
            gp.append({"slo_ttft_s": slo, "slo_tpot_s": a.slo_tpot, "n_meet": len(meet),
                       "frac_of_measured": len(meet) / len(meas) if meas else NA,
                       "goodput_req_per_s": len(meet) / mk if mk > 0 else NA})
        s["goodput"] = gp
    else:
        s["throughput"] = NA
        s["goodput"] = []
    ct = [r for r in ok if isinstance(r["cached_tokens"], int)]
    s["client_cached"] = ({"n": len(ct), "sum_cached": sum(r["cached_tokens"] for r in ct),
                           "sum_prompt": sum(r["prompt_len"] for r in ct),
                           "frac": sum(r["cached_tokens"] for r in ct) / max(1, sum(r["prompt_len"] for r in ct))}
                          if ct and len(ct) == len(ok) else
                          {"status": NA, "reason": "server 沒回 cached_tokens（serve 要 --enable-prompt-tokens-details）"
                           if not ct else f"只有 {len(ct)}/{len(ok)} 個請求有 cached_tokens"})
    if kind == "agentic":
        per_turn: dict = {}
        for r in ok:
            per_turn.setdefault(r["turn"], []).append(r["ttft_s"])
        s["ttft_by_turn"] = {str(t): stats(v) for t, v in sorted(per_turn.items())}
        warm = [r for r in rows if r["is_warmup"]]
        if warm:
            cnt: dict = {}
            for r in warm:
                cnt[r.get("warm_status", "missing")] = cnt.get(r.get("warm_status", "missing"), 0) + 1
            den = len(warm) - cnt.get("unknown", 0) - cnt.get("failed", 0)
            num = cnt.get("late", 0) + cnt.get("evicted", 0)
            s["warmup"] = {"n": len(warm), "status_counts": cnt, "wasted_numerator": num, "wasted_denominator": den,
                           "wasted_frac": num / den if den > 0 else NA,
                           "warm_ttft_s": stats([r["ttft_s"] for r in warm if r["ok"] == 1]),
                           "warm_e2e_s": stats([r["e2e_s"] for r in warm if r["ok"] == 1]),
                           "slack_s": stats([r.get("warm_slack_s") for r in warm])}
    if kind == "shared_prefix_burst" and ok:
        s["burst"] = {"n": len(ok), "makespan_s": s["throughput"]["makespan_s"],
                      "ttft_max_s": max(r["ttft_s"] for r in ok)}
    return s


def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in CSV_COLS})


def cmd_run(a) -> int:
    rd, rid = run_dir(a.run_dir)
    reqs, wmeta = W.read_jsonl(a.workload)
    sc = load_server_config(a.server_config)
    msg = validate_workload(reqs, sc.get("max_model_len") if sc else None)
    if msg:
        die(msg)
    kind = reqs[0]["kind"]
    if a.dry_run:
        plan = plan_dry(reqs, a)
        p = os.path.join(rd, "dryrun_schedule.csv")
        cols = ["req_id", "is_warmup", "phase", "t_sched_s", "prompt_len", "warm_lead_s", "warm_pred_err"]
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["run_id"] + cols, extrasaction="ignore")
            w.writeheader()
            for r in plan:
                w.writerow(dict(run_id=rid, **{k: r.get(k, "") for k in cols}))
        tmax = max(r["t_sched_s"] for r in plan)
        print(json.dumps({"dry_run": True, "kind": kind, "n_requests": len(reqs), "n_planned_sends": len(plan),
                          "n_warmups": sum(r["is_warmup"] for r in plan), "last_send_s": tmax,
                          "max_prompt_len": max(len(r["prompt_token_ids"]) for r in reqs),
                          "workload_sha": wmeta.get("sha256"), "schedule_csv": p}, ensure_ascii=False))
        return 0
    guard, guard_desc = make_guard(a.guard, a.guard_own_root, rd)
    t_start = now_iso()
    with (guard if guard is not None else _NullCtx()) as g:
        info = asyncio.run(run_async(a, reqs, wmeta, sc, rd, rid, guard_desc))
    contaminated = bool(getattr(guard, "contaminated", False)) if guard is not None else False
    verdict = guard.verdict() if guard is not None else "DISABLED"
    timing_valid = int(guard is not None and guard_desc != "STUB" and not contaminated)
    rows = info.pop("rows")
    ts = now_iso()
    for r in rows:
        r["ts"] = ts
        r["guard_verdict"] = verdict
        r["timing_valid"] = timing_valid
    summ = {"run_id": rid, "ts_start": t_start, "ts_end": ts, "workload": {"path": os.path.abspath(a.workload),
            "kind": kind, "sha256": wmeta.get("sha256", NA), "n": len(reqs), "meta": {k: v for k, v in wmeta.items()
                                                                                       if k != "args"}},
            "server_config": sc or "NOT_GIVEN", "args": vars(a),
            "guard": {"impl": guard_desc, "verdict": verdict, "contaminated": contaminated,
                      "report": guard.report() if guard is not None else None},
            "timing_valid": bool(timing_valid),
            "timing_note": ("OK" if timing_valid else
                            "TIMING_VOID：" + ("contaminated（CLAUDE.md §3：時間欄作廢、要重量）" if contaminated
                                              else "沒有 gpu_guard（--guard off／替身），不得當成乾淨計時")),
            **summarize(rows, info, a, kind), "server": info}
    write_csv(os.path.join(rd, "requests.csv"), rows)
    json.dump(summ, open(os.path.join(rd, "summary.json"), "w"), indent=1, ensure_ascii=False, default=str)
    if contaminated:
        open(os.path.join(rd, "CONTAMINATED"), "w").write(json.dumps(summ["guard"], default=str) + "\n")
        print(f"CONTAMINATED：{verdict}。結果只留在 {rd}，沒有寫進 --out-dir；時間要重量。", file=sys.stderr)
        return 4
    if guard is None:
        print("WARNING: --guard off：timing_valid=0（只給 CPU 測試用）", file=sys.stderr)
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        write_csv(os.path.join(a.out_dir, f"requests_{rid}.csv"), rows)
        json.dump(summ, open(os.path.join(a.out_dir, f"summary_{rid}.json"), "w"), indent=1, ensure_ascii=False,
                  default=str)
    print(json.dumps({"run_id": rid, "n_ok": summ["n_ok"], "n_failed": summ["n_failed"],
                      "ttft_p50": summ["ttft_s"]["p50"], "ttft_p95": summ["ttft_s"]["p95"],
                      "tpot_p50": summ["tpot_s"]["p50"], "timing_valid": summ["timing_valid"],
                      "summary": os.path.join(rd, "summary.json")}, ensure_ascii=False, default=str))
    return 3 if summ["n_failed_all_rows"] else 0


# ═════════════════════════════════ v0check ═════════════════════════════════
def first_diverge(a_: list, b_: list) -> int | None:
    for i, (x, y) in enumerate(zip(a_, b_)):
        if x != y:
            return i
    return None if len(a_) == len(b_) else min(len(a_), len(b_))


def margin_at(top: list, k: int | None):
    if k is None or k >= len(top) or not top[k]:
        return NA
    v = sorted(top[k].values(), reverse=True)
    return v[0] - v[1] if len(v) >= 2 else NA


async def v0_async(a, sc: dict, rd: str, rid: str) -> dict:
    base_url = a.url or sc.get("url")
    if not base_url:
        die("沒有 --url，--server-config 裡也沒有 url")
    mode = a.mode
    if mode == "auto":
        off = (sc.get("offload_info") or {}).get("kind")
        mode = {"cpu": "cpu", "tier": "tier"}.get(off, "apc" if sc.get("apc") == "on" else "none")
    out: dict = {"mode": mode, "prompts": []}
    rows: list[dict] = []
    bs = a.block_size
    async with Client(base_url, a.model_name, rid, a.timeout_s, return_token_ids=True, logprobs=2) as cl:
        await cl.models()
        cl.t0 = time.perf_counter()
        for i in range(a.server_warmup):
            rr = {"req_id": f"srvwarm{i}", "prompt_token_ids": W._toks(random.Random(f"v0warm|{i}"), 512, 1000, 120000),
                  "max_tokens": 2}
            await cl.send(cl.body(rr))
        texts = None
        if a.text_file:
            raw = open(a.text_file, errors="replace").read()
            texts = [raw[k * a.text_chars:(k + 1) * a.text_chars] for k in range(a.n_prompts)]
            if any(len(t) < a.text_chars for t in texts):
                die(f"{a.text_file} 不夠長：要 {a.n_prompts}×{a.text_chars} 個字元")
        idx = 0
        out["note_text_mode"] = "文字 prompt：忽略 --lengths，長度由 --text-chars 決定（token 數看 server 回的 prompt_tokens）" if texts else ""
        for L in ([None] if texts else a.lengths):
            for j in range(a.n_prompts):
                idx += 1
                if texts:
                    prompt, src = texts[j], f"text:{os.path.basename(a.text_file)}[{j}]"
                else:
                    prompt = W._toks(random.Random(f"v0|{a.seed}|{L}|{j}"), L, W.VOCAB_LO_DEFAULT, W.VOCAB_HI_DEFAULT)
                    src = f"random_tokens(seed={a.seed},L={L},j={j})"
                r = {"req_id": f"v0-{idx:03d}", "prompt_token_ids": prompt, "max_tokens": a.max_tokens}
                phases = [("miss", "all")]
                if mode in ("apc", "cpu", "tier"):
                    phases.append(("gpu_hit", None))
                if mode in ("cpu", "tier"):
                    phases.append(("cpu_hit", "local"))
                if mode == "tier":
                    phases.append(("fs_hit", "external"))
                if mode != "tier":
                    phases.append(("miss2", "all"))
                ref = None
                pinfo = {"prompt_idx": idx, "src": src, "phases": {}}
                for ph, reset in phases:
                    if reset is not None:
                        if ph != "miss":
                            await asyncio.sleep(a.settle_s)   # 讓背景的 GPU→CPU（→fs）寫完
                        if reset == "all":
                            rr_ = await cl.reset(external=mode in ("cpu", "tier"))
                        elif reset == "local":
                            rr_ = await cl.reset(external=False)
                        else:
                            rr_ = await cl.reset(external=True)
                        if not rr_["ok"]:
                            die(f"/reset_prefix_cache 失敗：{rr_}", 3)
                    mb, _, _ = await scrape(cl.session, cl.base)
                    res = await cl.send(cl.body(r, rid_suffix=f"-{ph}"))
                    ma, _, _ = await scrape(cl.session, cl.base)
                    md = metrics_delta(mb, ma)["series"]
                    toks = res["token_ids"]
                    if ph == "miss":
                        ref = res
                    k = first_diverge(ref["token_ids"], toks) if ref is not None else None
                    plen = res["prompt_tokens_server"] if isinstance(res["prompt_tokens_server"], int) else (
                        len(prompt) if isinstance(prompt, list) else NA)
                    row = dict(run_id=rid, ts=now_iso(), **cfg_cols(sc, a.label), v0_mode=mode, prompt_idx=idx,
                               prompt_src=src, prompt_len=plen, phase=ph,
                               reset=reset or "", cached_tokens=res["cached_tokens"], ttft_s=res["ttft_s"],
                               e2e_s=res["e2e_s"], n_out=res["n_out"], ok=res["ok"], error=res["error"],
                               out_sha1=hashlib.sha1(json.dumps(toks).encode()).hexdigest()[:16],
                               match_vs_miss=int(k is None) if ph != "miss" else "",
                               first_diverge="" if k is None else k,
                               miss_margin_at_diverge="" if k is None else margin_at(ref["top_logprobs"], k),
                               load_bytes_delta=md["offload_load_bytes"]["delta"],
                               external_hits_delta=md["external_prefix_cache_hits"]["delta"],
                               tier_read_bytes_delta=md["tiering_read_bytes"]["delta"])
                    rows.append(row)
                    pinfo["phases"][ph] = row
                    if not res["ok"]:
                        die(f"v0check {ph} 請求失敗：{res['error']}", 3)
                out["prompts"].append(pinfo)
    # 判定
    checks: dict = {}

    def add(name, val):
        checks.setdefault(name, []).append(val)
    for p in out["prompts"]:
        ph = p["phases"]
        L = ph["miss"]["prompt_len"]
        need = (L - 2 * bs) if isinstance(L, int) else None
        mc = ph["miss"]["cached_tokens"]
        add("miss_cold", NA if not isinstance(mc, int) else ("PASS" if mc < bs else "FAIL"))
        for name in ("gpu_hit", "cpu_hit", "fs_hit"):
            if name not in ph:
                continue
            row = ph[name]
            ct = row["cached_tokens"]
            add(f"{name}_cached", NA if not isinstance(ct, int) or need is None else ("PASS" if ct >= need else "FAIL"))
            add(f"{name}_match", "PASS" if row["match_vs_miss"] == 1 else "FAIL")
        if "cpu_hit" in ph:
            lb = ph["cpu_hit"]["load_bytes_delta"]
            add("cpu_hit_loaded_bytes", NA if lb == NA else ("PASS" if lb > 0 else "FAIL"))
        if "fs_hit" in ph:
            tb = ph["fs_hit"]["tier_read_bytes_delta"]
            add("fs_hit_tier_read_bytes", NA if tb == NA else ("PASS" if tb > 0 else "FAIL"))
        if "miss2" in ph:
            add("determinism_miss_vs_miss2", "PASS" if ph["miss2"]["match_vs_miss"] == 1 else "FAIL")
    verdict = {}
    for k, vs in checks.items():
        verdict[k] = "FAIL" if "FAIL" in vs else (NA if NA in vs else "PASS")
    allv = list(verdict.values())
    out["checks"] = checks
    out["verdict"] = verdict
    out["overall"] = "FAIL" if "FAIL" in allv else ("INCOMPLETE" if NA in allv else "PASS")
    out["rows"] = rows
    return out


V0_COLS = ["run_id", "ts", "cfg_label", "cfg_model", "cfg_apc", "cfg_offload", "cfg_gpu_blocks", "cfg_kv_cache_gib",
           "cfg_gpu_mem_util", "cfg_rocm_patch", "cfg_attn_backend", "cfg_max_model_len", "v0_mode", "prompt_idx",
           "prompt_src", "prompt_len", "phase", "reset", "cached_tokens", "ttft_s", "e2e_s", "n_out", "ok", "error",
           "out_sha1", "match_vs_miss", "first_diverge", "miss_margin_at_diverge", "load_bytes_delta",
           "external_hits_delta", "tier_read_bytes_delta", "guard_verdict", "timing_valid"]


def cmd_v0check(a) -> int:
    rd, rid = run_dir(a.run_dir)
    sc = load_server_config(a.server_config)
    guard, guard_desc = make_guard(a.guard, a.guard_own_root, rd)
    with (guard if guard is not None else _NullCtx()):
        out = asyncio.run(v0_async(a, sc, rd, rid))
    contaminated = bool(getattr(guard, "contaminated", False)) if guard is not None else False
    verdict = guard.verdict() if guard is not None else "DISABLED"
    tv = int(guard is not None and guard_desc != "STUB" and not contaminated)
    rows = out.pop("rows")
    for r in rows:
        r["guard_verdict"], r["timing_valid"] = verdict, tv
    out.update(run_id=rid, ts=now_iso(), server_config=sc or "NOT_GIVEN", args=vars(a),
               guard={"impl": guard_desc, "verdict": verdict, "contaminated": contaminated},
               timing_note="正確性判定不受污染影響（輸出由 temperature 0 決定）；ttft_s 只在 timing_valid=1 時可用")

    def dump(dirp):
        with open(os.path.join(dirp, f"v0check_{rid}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=V0_COLS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in V0_COLS})
        json.dump(out, open(os.path.join(dirp, f"v0check_{rid}.json"), "w"), indent=1, ensure_ascii=False, default=str)
    dump(rd)
    if contaminated:   # 正確性結果照 CLAUDE.md §3 品質量測的規則保留，只是時間欄作廢
        open(os.path.join(rd, "CONTAMINATED"), "w").write(verdict + "\n")
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        dump(a.out_dir)
    print(json.dumps({"run_id": rid, "mode": out["mode"], "overall": out["overall"], "verdict": out["verdict"]},
                     ensure_ascii=False))
    return 0 if out["overall"] == "PASS" else 3


# ═════════════════════════════════ CLI ═════════════════════════════════
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="m16 vLLM serving bench（client 端不 import torch／vllm）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="只組 vllm serve 指令，不啟動")
    p.add_argument("--model", choices=list(MODELS), required=True)
    p.add_argument("--apc", choices=["on", "off"], required=True)
    p.add_argument("--offload", default="none", help="none｜cpu:<GiB>｜tier:<cpuGiB>:<fs_path>:<fsGiB>")
    p.add_argument("--gpu-blocks", type=int, default=None, help="--num-gpu-blocks-override")
    p.add_argument("--kv-cache-gib", type=float, default=None, help="--kv-cache-memory-bytes")
    p.add_argument("--gpu-mem-util", type=float, default=None)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--max-model-len", type=int, required=True)
    p.add_argument("--rocm-attn-patch", action="store_true", help="F1 的修法：ROCM_ATTN＋K/V 拆開註冊（m16_serve_patched.py）")
    p.add_argument("--attention-backend", default=None, help="例如 TRITON_ATTN（讓有無卸載用同一個 kernel）")
    p.add_argument("--offload-block-size", type=int, default=None, help="kv_connector_extra_config.block_size（token）")
    p.add_argument("--store-threshold", type=int, default=None)
    p.add_argument("--eviction-policy", default="lru")
    p.add_argument("--connector-debug", action="store_true", help="offloading scheduler 的 logger 調到 DEBUG（F2）")
    p.add_argument("--extra-args", default="", help="原樣附加給 vllm serve（例：'--max-num-seqs 64'）")
    p.add_argument("--port", type=int, default=8016)
    p.add_argument("--gpu", default=None, help="實體卡號；沒給就用 TIARA_GPU")
    p.add_argument("--label", default=None)
    p.add_argument("--run-dir", default=None)

    p = sub.add_parser("wait-ready")
    p.add_argument("--url", required=True)
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--pid", type=int, default=None)

    def client_common(p):
        p.add_argument("--url", default=None)
        p.add_argument("--server-config", default=None, help="serve 寫出的 serve_config.json")
        p.add_argument("--model-name", default="m")
        p.add_argument("--label", default=None)
        p.add_argument("--out-dir", default=None, help="results/m16_stage3_mi300x/<實驗>/（污染時不寫）")
        p.add_argument("--run-dir", default=None)
        p.add_argument("--timeout-s", type=float, default=1800)
        p.add_argument("--server-warmup", type=int, default=2, help="開跑前的暖機請求數（不計）")
        p.add_argument("--settle-s", type=float, default=3.0, help="reset／結束後等背景搬移的秒數")
        p.add_argument("--block-size", type=int, default=16)
        p.add_argument("--guard", choices=["on", "off"], default="on")
        p.add_argument("--guard-own-root", type=int, default=None)

    p = sub.add_parser("run")
    p.add_argument("--workload", required=True)
    client_common(p)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--dry-service-s", type=float, default=0.0)
    p.add_argument("--time-scale", type=float, default=1.0, help="open-loop 的 t_arrive 乘上這個")
    p.add_argument("--max-requests", type=int, default=None)
    p.add_argument("--max-inflight", type=int, default=None)
    p.add_argument("--reset-before", choices=["on", "off"], default="on")
    p.add_argument("--reset-external", action="store_true", help="沒有 server-config 時也清 connector")
    p.add_argument("--slo-ttft", type=float, nargs="*", default=[])
    p.add_argument("--slo-tpot", type=float, default=None)
    p.add_argument("--warmup-lead", default=None, help="'2s'＝2 秒；'1x'＝1×還原時間估計（要 --restore-est）")
    p.add_argument("--warmup-noise", type=float, default=0.0)
    p.add_argument("--warmup-seed", type=int, default=0)
    p.add_argument("--restore-est", default=None, help="fixed:<s>｜linear:<a>:<b_s_per_token>")
    p.add_argument("--evict", choices=["fill", "reset", "none"], default=None, help="三段式的 evict 段")
    p.add_argument("--phases", default="seed,evict,burst", help="三段式要跑哪幾段（例：burst＝冷啟動，沒有 seed）")
    p.add_argument("--phase-gap-s", type=float, default=2.0)
    p.add_argument("--server-log", default=None, help="vllm_server.log：數 connector 的延後訊息")

    p = sub.add_parser("v0check")
    client_common(p)
    p.add_argument("--mode", choices=["auto", "none", "apc", "cpu", "tier"], default="auto")
    p.add_argument("--lengths", type=int, nargs="+", default=[8192])
    p.add_argument("--n-prompts", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--text-file", default=None, help="用真文字（server 自己 tokenize）避開亂數 token 的近似平手")
    p.add_argument("--text-chars", type=int, default=30000)

    p = sub.add_parser("count-log")
    p.add_argument("--log", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)

    a = ap.parse_args(argv)
    return {"serve": cmd_serve, "wait-ready": cmd_wait_ready, "run": cmd_run, "v0check": cmd_v0check,
            "count-log": cmd_count_log}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
