"""x8_capacity.py — X8 的容量算術、長 context 還原時間〔外插〕、session 層級模型的重複計算倍率

輸出（results/p3_explore/x8/）：
  x8_capacity.csv    每種長度 × 每層：一個 session 幾 GiB、放得下幾個（BF16／INT4）〔算術〕
  x8_restore_len.csv 每種長度 × 層 × GPU 狀態：Cake 還原時間〔模擬〕（block ≥80 的 f 是直線外插〔外插〕）
  x8_dedup.csv       session 層級（每個 session 存自己整份 context）vs block 層級（共用前綴只存一次）的位元組倍率
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x8_sim as X  # noqa: E402

REPO = X.REPO
OUT = X.PF.explore_out("x8")
LLAMA = glob.glob(os.path.join(os.environ.get("TIARA_HF_HUB", "/ssd8/hungwei/paper-hkv/hf-cache/hub"), "models--unsloth--Llama-3.1-8B-Instruct/snapshots/*/"))[0]
GiB = 1 << 30


def w(path, rows):
    keys = ["run_id", "ts"] + list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        r0 = os.environ.get("RUN_ID", "adhoc")
        t0 = datetime.now().astimezone().isoformat(timespec="seconds")
        for r in rows:
            wr.writerow(dict(run_id=r0, ts=t0, **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}))


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = json.load(open(os.path.join(LLAMA, "config.json")))
    hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    kv_tok = 2 * cfg["num_hidden_layers"] * cfg["num_key_value_heads"] * hd * 2   # K,V × BF16
    # 規則 6：用 config 驗算 main.tex §2.1 的「64 MiB／512-token block」
    assert kv_tok * 512 == 64 * (1 << 20), f"KV/token={kv_tok}，和 64 MiB/block 對不上"
    wbytes = sum(os.stat(p).st_size for p in glob.glob(os.path.join(LLAMA, "*.safetensors")))
    w_gib = wbytes / GiB
    print(f"Llama-3.1-8B: KV {kv_tok} B/token; safetensors {w_gib:.2f} GiB ({LLAMA})")
    HBM = 192.0          # MI300X（main.tex §3.1）
    pool = HBM - w_gib   # 上界：沒扣 activation 與 gpu_memory_utilization（vLLM 實際 KV 池＝NOT_MEASURED）
    int4_ratio = 1.25 / 4.0   # main.tex §6.3：32K Llama session 4 GiB BF16 → 1.25 GiB INT4（KIVI，含 scale）
    rows = []
    for L in (8192, 32768, 65536, 131072):
        s_bf = L * kv_tok / GiB
        for tier, cap in [("HBM_minus_weights_upper_bound", pool), ("pin_budget_16", 16.0), ("pin_budget_64", 64.0),
                          ("cpu_128", 128.0), ("cpu_256", 256.0), ("cpu_512", 512.0), ("cpu_1024", 1024.0)]:
            rows.append(dict(context_tok=L, tier=tier, tier_GiB=cap, session_GiB_bf16=s_bf,
                             session_GiB_int4=s_bf * int4_ratio, n_sessions_bf16=math.floor(cap / s_bf),
                             n_sessions_int4=math.floor(cap / (s_bf * int4_ratio)),
                             note=("weights from safetensors file sizes; activations/vLLM utilization not subtracted"
                                   if tier.startswith("HBM") else "")))
    w(os.path.join(OUT, "x8_capacity.csv"), rows)
    # 長度 × 層的還原時間
    f, ell, cm = X.load_costs()
    T = X.cost_tables(f, ell)
    rr = []
    for L in (8192, 16384, 32768, 65536, 131072):
        nb = L // 512
        for g in X.GPUS:
            for tier in ("cpu", "local", "nfs", "D"):
                t = T[(g, tier)][nb]
                meet = X.cake(nb, ell[tier], T[(g, "f")])[1] if tier != "D" else nb
                rr.append(dict(context_tok=L, n_blocks=nb, gpu=g, tier=("drop_recompute" if tier == "D" else tier),
                               restore_s=float(t), meet_block=int(meet), extrapolated=nb > cm["f_measured_blocks"],
                               f_source=cm["f_run"][0]))
    w(os.path.join(OUT, "x8_restore_len.csv"), rr)
    # 重複計算倍率：session 層級（每個 session 的最大物件）vs 不同 block 數
    dd = []
    for name in ("conversation", "toolagent"):
        rows_t = [json.loads(l) for l in open(os.path.join(X_TR_MOON, f"{name}_trace.jsonl"))]
        distinct = len({h for r in rows_t for h in r["hash_ids"]})
        z = np.load(os.path.join(TRACES, f"mooncake_{name}.npz"), allow_pickle=True)
        mx = defaultdict(int)
        for s, o in zip(z["sid"], z["obj_tok"]):
            mx[int(s)] = max(mx[int(s)], int(math.ceil(o / 512)))
        dd.append(dict(workload=f"mooncake_{name}", distinct_blocks_512=distinct, session_level_blocks=sum(mx.values()),
                       inflation=sum(mx.values()) / distinct))
    rows_b = [json.loads(l) for l in open(os.path.join(X_TR_BAIL, "qwen_coder_blksz_16.jsonl"))]
    distinct16 = len({h for r in rows_b for h in r["hash_ids"]})
    z = np.load(os.path.join(TRACES, "bailian_coder.npz"), allow_pickle=True)
    mx = defaultdict(int)
    for s, o in zip(z["sid"], z["obj_tok"]):
        mx[int(s)] = max(mx[int(s)], int(o))
    dd.append(dict(workload="bailian_coder", distinct_blocks_512=distinct16 * 16 / 512,
                   session_level_blocks=sum(mx.values()) / 512, inflation=(sum(mx.values()) / 512) / (distinct16 * 16 / 512)))
    w(os.path.join(OUT, "x8_dedup.csv"), dd)
    for r in dd:
        print(r)


X_TR_MOON = os.path.join(os.environ.get("TIARA_DATASETS", "/ssd8/hungwei/paper-hkv/datasets"), "mooncake/traces")
X_TR_BAIL = os.path.join(os.environ.get("TIARA_DATASETS", "/ssd8/hungwei/paper-hkv/datasets"), "qwen-bailian-usagetraces-anon")
TRACES = sys.argv[1] if len(sys.argv) > 1 else ""

if __name__ == "__main__":
    main()
