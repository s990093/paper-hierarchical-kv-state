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
                       mooncake_trace, profile, trace_duration_s)

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
    ap.add_argument("--device-write-mibps", type=float, default=181.0,
                    help="磁碟的**持續**寫入頻寬（MiB/s），用於可行性判定。"
                         "預設 181 = /ssd7（Samsung 870 QVO, SATA QLC）實測值。"
                         "⚠️ 1 GiB 的短測會落在 QLC 的 SLC 快取裡量到 492；"
                         "16 GiB 的長測才是持續值 181。KV 階是持續寫入，"
                         "所以要用後者。NVMe（Crucial P3）實測為 2,512。"
                         "見 results/m2_harness/disk_bw*.csv")
    ap.add_argument("--fs-root", default="/ssd7",
                    help="用來報告實體可用空間的掛載點")
    ap.add_argument("--oracle-dest", default="cost-aware",
                    choices=["cost-aware", "cascade"],
                    help="Oracle 逐出後的目的地選擇。cost-aware=比較各去處在"
                         "『下次使用的位置』上的實際成本（放 SSD 5.536 ms 對上"
                         "重算 4.008+0.00021×位置，交叉點 7,278 token）；"
                         "cascade=無條件往下推（舊行為，會系統性低估 Oracle）")
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
        dur = trace_duration_s(tname)
        uniq = len({b for r in trace for b in r})
        need_tib = uniq * bytes_per_block / 1024**4
        print(f"\n{'=' * 104}\ntrace「{tname}」：{len(trace):,} 請求、"
              f"{sum(len(r) for r in trace):,} 次存取、{uniq:,} 不重複 block")
        print(f"  整個工作集若全部放磁碟需要 **{need_tib:.1f} TiB**"
              f"（裝置只有 {du.total / 1024**4:.1f} TiB）")
        print(f"  trace 時長 {dur / 60:.1f} 分鐘；每個 block 寫一次 = "
              f"{bytes_per_block / 1024**2:.0f} MiB")
        print(f"{'SSD 容量':>11s}{'覆蓋':>7s}{'best':>9s}{'headroom':>10s}"
              f"{'判定':>9s}{'best 寫 SSD':>13s}{'需要頻寬':>12s}"
              f"{'可行?':>7s}{'oracle 寫 SSD':>14s}{'需要頻寬':>12s}")
        for g in a.ssd_gib:
            ssd_blocks = 10**9 if g < 0 else int(g * 1024**3) // bytes_per_block
            use_ssd_possible = ssd_blocks > 0
            sim = Sim(cm, gpu_blocks, cpu_blocks, ssd_blocks=ssd_blocks)
            res = {}
            for k, (pol, uc, us) in POLICIES.items():
                if us and not use_ssd_possible:
                    continue          # 沒有磁碟階時 tier_fs 不存在
                res[k] = sim.run_online(trace, pol, uc, us, **sem)
            res["oracle"] = sim.run_oracle(trace, True, use_ssd_possible,
                                           dest=a.oracle_dest, **sem)
            best = min((k for k in res if k != "oracle"),
                       key=lambda k: res[k]["total_ms"])
            head = 100 * (res[best]["total_ms"] - res["oracle"]["total_ms"]) \
                / res[best]["total_ms"]
            verdict = ("GO" if head > 15 else
                       "MARGINAL" if head >= 5 else "NO_GO")
            cover = min(1.0, ssd_blocks / uniq)
            label = "無限" if g < 0 else f"{g:,.0f} GiB"
            fs_ms = res.get("tier_fs", {}).get("total_ms", math.nan)
            # 🔴 可行性：把「寫了幾個 block」換算成需要的持續寫入頻寬，
            #    與裝置實測能力比較。模擬的成本模型沒有向寫入收費，
            #    所以一個策略可能在模擬裡很快、在真機上根本寫不下去。
            #    /ssd7（Samsung 870 QVO, SATA QLC）**持續**寫入實測 181 MiB/s
            #    （短測 492，但那是 SLC 快取；KV 階是持續寫入）。
            #    /（Crucial P3, NVMe）實測 2,512 MiB/s —— 差 13.9 倍。
            #    同一個策略在一顆碟上可行、在另一顆上不可行，
            #    這就是論文 κ 主張的實證。
            DEV_MBPS = a.device_write_mibps
            def bw(w):
                return w * bytes_per_block / 1024**2 / dur if dur else float("nan")
            wb = res[best].get("writes", {}).get("ssd", 0)
            wo = res["oracle"].get("writes", {}).get("ssd", 0)
            feas = "✅" if bw(wb) <= DEV_MBPS else "🔴"
            print(f"{label:>11s}{100 * cover:>6.1f}%{best:>9s}{head:>9.2f}%"
                  f"{verdict:>9s}{wb:>13,}{bw(wb):>10,.0f}MB/s{feas:>6s}"
                  f"{wo:>14,}{bw(wo):>10,.0f}MB/s")
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
                    "ssd_writes": v.get("writes", {}).get("ssd", ""),
                    "cpu_writes": v.get("writes", {}).get("cpu", ""),
                    "trace_duration_s": round(dur, 1) if dur else "",
                    "ssd_write_mibps": round(
                        v.get("writes", {}).get("ssd", 0) * bytes_per_block
                        / 1024**2 / dur, 1) if dur else "",
                    "device_write_mibps_sustained": a.device_write_mibps,
                    "best_baseline": best,
                    "oracle_headroom_pct": round(head, 3) if pol == "oracle" else "",
                    "verdict": verdict if pol == "oracle" else "",
                    "model_profile": a.model,
                    "gpu_budget_tokens": prof["gpu_kv_tokens"],
                    "cpu_budget_gib": a.cpu_gib, "unique_blocks": uniq,
                    "lookup": a.lookup, "prefetch": int(a.prefetch),
                    "oracle_dest": a.oracle_dest,
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
