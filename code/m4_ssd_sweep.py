#!/usr/bin/env python3
"""M4 附屬實驗：**SSD 階要多大，階層才成立？**

## 為什麼需要這個

先前所有模擬把 SSD 階設成 `ssd_blocks = 10**9`（實質無限）。
用正確的 512-token 粒度解碼 Mooncake 之後才發現這是**物理上不可能**的：

    toolagent    5,457,182 blocks × 16 token × 128 KiB/token = **10.4 TiB**
    conversation 5,674,025 blocks × 16 token × 128 KiB/token = **10.8 TiB**

而這台機器最大的單一磁碟是 7.3 TB（`/ssd7`，且目前只剩 132 GB 可用）。
無限 SSD 等於白送給 `tier_fs` 一個 10 TB 的快取——而 `tier_fs` 正是
修正粒度後勝出的 baseline，所以 **headroom 被系統性低估**。

## 做法

把 SSD 容量從 0（沒有磁碟階）掃到無限，其餘設定不動。
兩個實體參考點會標在輸出裡：
  * `/ssd7` 目前可用空間
  * `/ssd7` 的裝置總容量（7.3 TB，但是與二十幾個使用者共用）

這條曲線同時回答兩件事：
  1. 論文主張的「SSD 階」在多大的預算下才真的有貢獻
  2. 先前 `tier_fs` 的優勢有多少是來自不切實際的容量假設

用法：
  python code/m4_ssd_sweep.py --trace toolagent conversation
"""
from __future__ import annotations
import argparse
import csv
import math
import shutil
from datetime import datetime
from pathlib import Path

from m4_oracle import (BLOCK, MODEL_PROFILES, OUT, Sim, load_cost_model,
                       mooncake_trace, profile)

POLICIES = {
    "full_gpu": ("lru", False, False),
    "cpu_lru": ("lru", True, False),
    "cpu_arc": ("arc", True, False),
    "tier_fs": ("lru", True, True),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="sata", choices=["sata", "nvme"])
    ap.add_argument("--model", default="llama-bf16", choices=list(MODEL_PROFILES))
    ap.add_argument("--trace", nargs="*", default=["toolagent", "conversation"])
    ap.add_argument("--ssd-gib", type=float, nargs="*",
                    default=[0, 32, 128, 512, 2048, 8192, -1],
                    help="SSD 階容量（GiB）。-1 = 無限（先前的預設，物理上不可能）")
    ap.add_argument("--cpu-gib", type=float, default=24.0)
    ap.add_argument("--lookup", choices=["prefix", "per-block"], default="prefix")
    ap.add_argument("--prefetch", action="store_true", default=True)
    ap.add_argument("--no-prefetch", dest="prefetch", action="store_false")
    ap.add_argument("--fs-root", default="/ssd7",
                    help="用來報告實體可用空間的掛載點")
    ap.add_argument("--out", default=str(OUT / "ssd_sweep.csv"))
    a = ap.parse_args()

    prof = profile(a.model)
    cm = load_cost_model(a.device, require_model_key=prof["cost_model_key"])
    gpu_blocks = prof["gpu_kv_tokens"] // BLOCK
    bytes_per_block = prof["kv_bytes_per_token"] * BLOCK
    cpu_blocks = int(a.cpu_gib * 1024**3) // bytes_per_block
    sem = {"prefix_semantics": a.lookup == "prefix", "prefetch": a.prefetch}

    du = shutil.disk_usage(a.fs_root)
    print(f"[剖面] {a.model}：GPU {prof['gpu_kv_tokens']:,} token = {gpu_blocks:,} "
          f"blocks；每 block {bytes_per_block / 1024**2:.1f} MiB")
    print(f"[實體] {a.fs_root}：裝置 {du.total / 1024**4:.1f} TiB、"
          f"目前可用 {du.free / 1024**3:.0f} GiB "
          f"（= {du.free // bytes_per_block:,} blocks）")

    rows: list[dict] = []
    for tname in a.trace:
        trace = mooncake_trace(tname)
        uniq = len({b for r in trace for b in r})
        need_tib = uniq * bytes_per_block / 1024**4
        print(f"\n{'=' * 104}\ntrace「{tname}」：{len(trace):,} 請求、"
              f"{sum(len(r) for r in trace):,} 次存取、{uniq:,} 不重複 block")
        print(f"  整個工作集若全部放磁碟需要 **{need_tib:.1f} TiB**"
              f"（裝置只有 {du.total / 1024**4:.1f} TiB）")
        print(f"{'SSD 容量':>12s}{'blocks':>12s}{'覆蓋工作集':>12s}"
              f"{'tier_fs ms':>14s}{'oracle ms':>13s}{'best':>10s}"
              f"{'headroom':>10s}{'判定':>9s}")
        for g in a.ssd_gib:
            ssd_blocks = 10**9 if g < 0 else int(g * 1024**3) // bytes_per_block
            use_ssd_possible = ssd_blocks > 0
            sim = Sim(cm, gpu_blocks, cpu_blocks, ssd_blocks=ssd_blocks)
            res = {}
            for k, (pol, uc, us) in POLICIES.items():
                if us and not use_ssd_possible:
                    continue          # 沒有磁碟階時 tier_fs 不存在
                res[k] = sim.run_online(trace, pol, uc, us, **sem)
            res["oracle"] = sim.run_oracle(trace, True, use_ssd_possible, **sem)
            best = min((k for k in res if k != "oracle"),
                       key=lambda k: res[k]["total_ms"])
            head = 100 * (res[best]["total_ms"] - res["oracle"]["total_ms"]) \
                / res[best]["total_ms"]
            verdict = ("GO" if head > 15 else
                       "MARGINAL" if head >= 5 else "NO_GO")
            cover = min(1.0, ssd_blocks / uniq)
            label = "無限" if g < 0 else f"{g:,.0f} GiB"
            fs_ms = res.get("tier_fs", {}).get("total_ms", math.nan)
            print(f"{label:>12s}{ssd_blocks:>12,}{100 * cover:>11.1f}%"
                  f"{fs_ms:>14,.0f}{res['oracle']['total_ms']:>13,.0f}"
                  f"{best:>10s}{head:>9.2f}%{verdict:>9s}")
            for pol, v in res.items():
                e = v.get("evict", {})
                rows.append({
                    "ts": datetime.now().astimezone().isoformat(),
                    "trace": tname, "ssd_gib": g if g >= 0 else "unlimited",
                    "ssd_blocks": ssd_blocks,
                    "ssd_covers_working_set_pct": round(100 * cover, 2),
                    "working_set_tib": round(need_tib, 2),
                    "device_total_tib": round(du.total / 1024**4, 2),
                    "device_free_gib": round(du.free / 1024**3, 1),
                    "policy": pol, "total_ms": round(v["total_ms"], 2),
                    "gpu_hits": v["hits"]["gpu"], "cpu_hits": v["hits"]["cpu"],
                    "ssd_hits": v["hits"]["ssd"], "recompute": v["hits"]["drop"],
                    "evict_free": e.get("free", ""),
                    "evict_to_cpu": e.get("to_cpu", ""),
                    "evict_to_ssd": e.get("to_ssd", ""),
                    "best_baseline": best,
                    "oracle_headroom_pct": round(head, 3) if pol == "oracle" else "",
                    "verdict": verdict if pol == "oracle" else "",
                    "model_profile": a.model,
                    "gpu_budget_tokens": prof["gpu_kv_tokens"],
                    "cpu_budget_gib": a.cpu_gib, "unique_blocks": uniq,
                    "lookup": a.lookup, "prefetch": int(a.prefetch),
                    "device": a.device,
                    "cost_model": str(OUT / "cost_model.json"),
                })
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {p}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
