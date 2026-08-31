#!/usr/bin/env python3
"""M4 附屬實驗：**Oracle 的優勢集中在多長的請求上？**

## 為什麼需要這個

論文的目標區間是 128K–512K，但公開 trace 的中位數只有 6,352 token。
先前想用「把 block 粒度放大」來合成長上下文，但那要假設
「重用結構與請求長度無關」——沒有資料可以驗證的假設。

其實不必假設。真實 trace 裡**有**長請求，只是佔比小：

    toolagent     ≥32K：726 筆（3.1%）　≥64K：211 筆（0.9%）　最長 126,195
    conversation  ≥32K：829 筆（6.9%）　≥64K：254 筆（2.1%）　最長 126,195

所以直接問：**把每個請求的成本按長度分箱，Oracle 相對最佳 baseline
的節省是集中在長請求還是短請求？** 全部用真實資料，零假設。

如果節省集中在長請求，那論文「長上下文才需要這套」的主張就有了直接證據，
而且可以外推的方向是明確的（越長越有價值）。
如果反過來，那主張要修正。

用法：
  python code/m4_by_length.py --trace toolagent conversation
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
from pathlib import Path

from m4_invariants import check_results, preflight
from m4_oracle import (BLOCK, SIM_VERSION, MODEL_PROFILES, OUT, Sim, load_cost_model,
                       mooncake_trace, profile)

POLICIES = {
    "full_gpu": ("lru", False, False),
    "cpu_lru": ("lru", True, False),
    "cpu_arc": ("arc", True, False),
    "tier_fs": ("lru", True, True),
}
# 分箱邊界（token）。最後一箱是 128K 以上（實際上是空的，保留供將來對照）
BINS = [0, 4096, 8192, 16384, 32768, 65536, 131072, 10**9]


def label(lo: int, hi: int) -> str:
    def f(x: int) -> str:
        return "∞" if x >= 10**9 else (f"{x // 1024}K" if x >= 1024 else str(x))
    return f"{f(lo)}–{f(hi)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="nvme", choices=["sata", "nvme"])
    ap.add_argument("--model", default="llama-bf16", choices=list(MODEL_PROFILES))
    ap.add_argument("--trace", nargs="*", default=["toolagent", "conversation"])
    ap.add_argument("--ssd-gib", type=float, default=512.0,
                    help="SSD 階容量。預設 512 GiB —— 一個實體上放得下的值")
    ap.add_argument("--cpu-gib", type=float, default=24.0)
    ap.add_argument("--oracle-dest", default="best",
                    choices=["best", "cost-aware", "cascade"])
    ap.add_argument("--out", default=str(OUT / "by_length.csv"))
    a = ap.parse_args()

    prof = profile(a.model)
    cm = load_cost_model(a.device, require_model_key=prof["cost_model_key"])
    gpu_blocks = prof["gpu_kv_tokens"] // BLOCK
    bpb = prof["kv_bytes_per_token"] * BLOCK
    cpu_blocks = int(a.cpu_gib * 1024**3) // bpb
    ssd_blocks = int(a.ssd_gib * 1024**3) // bpb
    sem = {"prefix_semantics": True, "prefetch": True, "per_request": True}
    print(f"[剖面] {a.model}：GPU {gpu_blocks:,}、CPU {cpu_blocks:,}、"
          f"SSD {ssd_blocks:,} blocks")

    rows: list[dict] = []
    for tname in a.trace:
        trace = mooncake_trace(tname)
        preflight(cm, trace, tname, gpu_blocks, cpu_blocks, ssd_blocks, bpb)
        sim = Sim(cm, gpu_blocks, cpu_blocks, ssd_blocks=ssd_blocks)
        res = {k: sim.run_online(trace, *v, **sem) for k, v in POLICIES.items()}
        res["oracle"] = sim.run_oracle(trace, True, True, dest=a.oracle_dest,
                                       **sem)
        best = min((k for k in res if k != "oracle"),
                   key=lambda k: res[k]["total_ms"])
        check_results(res, trace, best)
        bl = res[best]["per_request_ms"]
        ol = res["oracle"]["per_request_ms"]
        lens = [len(r) * BLOCK for r in trace]

        print(f"\n{'=' * 96}\ntrace「{tname}」，最佳 baseline = {best}")
        print(f"{'請求長度':>12s}{'筆數':>8s}{'佔全部時間':>11s}"
              f"{'baseline ms':>14s}{'oracle ms':>13s}{'節省':>9s}"
              f"{'佔總節省':>10s}")
        tot_save = sum(bl) - sum(ol)
        tot_ms = sum(bl)
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            idx = [i for i, x in enumerate(lens) if lo <= x < hi]
            if not idx:
                continue
            b = sum(bl[i] for i in idx)
            o = sum(ol[i] for i in idx)
            save = b - o
            print(f"{label(lo, hi):>12s}{len(idx):>8,}{100 * b / tot_ms:>10.1f}%"
                  f"{b:>14,.0f}{o:>13,.0f}"
                  f"{100 * save / b if b else 0:>8.2f}%"
                  f"{100 * save / tot_save if tot_save else 0:>9.1f}%")
            rows.append({
                "ts": datetime.now().astimezone().isoformat(),
                    "sim_version": SIM_VERSION,
                "trace": tname, "bin": label(lo, hi),
                "bin_lo_tokens": lo, "bin_hi_tokens": hi,
                "requests": len(idx),
                "share_of_total_time_pct": round(100 * b / tot_ms, 3),
                "best_baseline": best,
                "baseline_ms": round(b, 2), "oracle_ms": round(o, 2),
                "saving_pct_within_bin": round(100 * save / b, 3) if b else "",
                "share_of_total_saving_pct":
                    round(100 * save / tot_save, 2) if tot_save else "",
                "model_profile": a.model, "ssd_gib": a.ssd_gib,
                "cpu_gib": a.cpu_gib, "oracle_dest": a.oracle_dest,
                "device": a.device,
            })
        print(f"{'全部':>12s}{len(lens):>8,}{100.0:>10.1f}%"
              f"{tot_ms:>14,.0f}{sum(ol):>13,.0f}"
              f"{100 * tot_save / tot_ms:>8.2f}%{100.0:>9.1f}%")
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
