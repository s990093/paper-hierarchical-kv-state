#!/usr/bin/env python3
"""從 vLLM server.log 抽出**連接器層**的 KV 搬運量與搬運時間。

## 為什麼需要這個（RUNLOG 發現 11）

M2 的 CPU／SSD 階成本是用 TTFT 減 gpu_resident 基準算的。對 Qwen3-30B-A3B
這個相減得到 **−49 ms**（CPU 階 warm 比基準還快），成本被 clamp 成 0，
Oracle 於是以為「搬到 CPU 免費」，模擬器驗證因此爆掉（sim ratio 60 vs 實測 10.8）。

vLLM 自己會印 `vllm:kv_offload_load_bytes` 與 `vllm:kv_offload_load_time`，
那是**連接器層的直接量測**，與 TTFT 無關，不受計算重疊影響。
用它換算每個 block 的搬運成本，補上相減法失效的那一格。

交叉驗證（兩種獨立量法必須吻合，否則不能用）：
    qwen7b  TTFT 法 0.376 ms/block  ↔  指標法 0.398 ms/block（差 6%）
    seedoss TTFT 法 0.902 ms/block  ↔  指標法 0.929 ms/block（差 3%）

用法：
    python code/m2_connector_metrics.py --runs /mlsteam/data/tiara/runs
"""
from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results/m2_harness_mi300x/connector_transfer_mi300x.csv"

# 每個模型的 BF16 KV 大小（KiB/token），與 m2_cost_model / m4_oracle 一致
KV_KIB = {"b-llama8b": 128.0, "b-ultralong8b-1m": 128.0, "b-qwen7b-1m": 56.0,
          "b-qwen3-30b-a3b": 96.0, "b-seedoss36b": 256.0}
BLOCK = 16

# 🔴 m2_cost_model 的 run 目錄不帶模型名（20260915-124109-m2-retrieval/cpu_r0），
#    所以模型一律從 server.log 裡的 model='...' 判定；比對前先轉小寫。
MODEL_PAT = [("ultralong", "b-ultralong8b-1m"),          # 也含 llama，必須排在前面
             ("qwen3-30b-a3b", "b-qwen3-30b-a3b"),
             ("qwen2.5-7b", "b-qwen7b-1m"),
             ("seed-oss-36b", "b-seedoss36b"),
             ("llama-3.1-8b", "b-llama8b")]


def model_of(path: str) -> str | None:
    low = path.lower()
    for pat, key in MODEL_PAT:
        if pat in low:
            return key
    return None


def scrape(log: Path) -> dict:
    t = log.read_text(errors="replace")
    out: dict[str, float] = {}
    for k in ("kv_offload_load_bytes", "kv_offload_load_time",
              "kv_offload_store_bytes", "kv_offload_store_time"):
        hits = re.findall(rf"vllm:{k}=([0-9.e+-]+)", t)
        if hits:
            out[k] = float(hits[-1])
    m = re.search(r"model='([^']+)'", t)
    out["model_path"] = m.group(1) if m else ""
    return out


RUN_ID = ""


def main() -> int:
    global RUN_ID
    RUN_ID = f"{datetime.now():%Y%m%d-%H%M%S}-m2-connector"
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/mlsteam/data/tiara/runs")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    rows = []
    logs = sorted(Path(a.runs).glob("*/*/server.log")) + sorted(Path(a.runs).glob("*/server.log"))
    for log in logs:
        # 只看 CPU 主階：m2 的 cpu_r*，以及 m3 的 cpu_lru / cpu_arc run
        name = log.parent.name
        if not (name.startswith("cpu") or "-cpu_lru" in name or "-cpu_arc" in name):
            continue
        d = scrape(log)
        if not d.get("kv_offload_load_time") or not d.get("kv_offload_load_bytes"):
            continue
        mk = model_of(d.get("model_path", "")) or model_of(str(log))
        if not mk:
            continue
        lb, lt = d["kv_offload_load_bytes"], d["kv_offload_load_time"]
        sb, stt = d.get("kv_offload_store_bytes", 0.0), d.get("kv_offload_store_time", 0.0)
        block_bytes = KV_KIB[mk] * 1024 * BLOCK
        rows.append({
            "run_id": RUN_ID,
            "model_key": mk,
            "stage": "m3_baseline" if "-m3-" in str(log) else "m2_retrieval",
            "run_dir": str(log.parent),
            "load_bytes": int(lb), "load_time_s": lt,
            "load_gb_per_s": round(lb / 1e9 / lt, 3),
            "load_ms_per_block": round(block_bytes / (lb / lt) * 1000, 5),
            "store_bytes": int(sb), "store_time_s": stt,
            "store_gb_per_s": round(sb / 1e9 / stt, 3) if stt else "",
            "kv_kib_per_token": KV_KIB[mk],
            "block_bytes": int(block_bytes),
            "ts": datetime.now().astimezone().isoformat(),
            "log": str(log),
        })
    if not rows:
        print("🔴 沒有抓到任何 cpu_* 的連接器指標")
        return 1
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"寫入 {len(rows)} 列 -> {a.out}\n")
    print(f"{'model':18s}{'n':>3s}{'GB/s 中位':>11s}{'ms/block 中位':>14s}")
    import statistics as st
    for mk in dict.fromkeys(r["model_key"] for r in rows):
        v = [r for r in rows if r["model_key"] == mk]
        print(f"{mk:18s}{len(v):>3d}{st.median(x['load_gb_per_s'] for x in v):>11.2f}"
              f"{st.median(x['load_ms_per_block'] for x in v):>14.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
