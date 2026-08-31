#!/usr/bin/env python3
"""M4 附屬實驗：**把真實 trace 拉到 128K / 512K 的長上下文區間**。

## 為什麼需要這個

論文的目標區間是 128K–512K，但公開 trace 沒有這個東西：
Mooncake conversation 的中位數只有 6,906 token、toolagent 6,346 token，
**沒有任何一個請求 ≥ 128K**。

先前用「縮小 GPU 預算」來製造壓力，但那**只等價於壓力軸，不等價於成本軸**：
重算成本隨 block 的絕對位置線性成長（式 eq:recompute-position），
把預算縮小 20 倍不會讓任何一個 block 的位置變遠，
所以「重算 vs 搬運」的相對價格完全沒動。而那正是論文主張的核心。

## 做法：block 粒度放大

把 trace 裡的每個 block b 展開成 S 個連續 block `b*S … b*S+S-1`。

* 重用結構**完全保留**（誰跟誰共用前綴、共用多長，一個位元都沒改）
* 每個請求的長度變成 S 倍 → 中位數 6,906 → 6,906×S
* 每個 block 的**絕對位置**也變成 S 倍 → 重算成本用的是真正的長上下文位置

⚠️ **這裡做了一個假設並且必須標明**：
   「重用結構與請求長度無關」。
   真實的 128K 工作負載可能有不同的共用形態（例如更長的共用文件前綴、
   更少的短查詢）。沒有公開資料可以驗證這一點，所以本節的結果
   標為 `assumption=reuse_structure_invariant_to_length`，
   **不可與直接量測的數字並列呈現**。

⚠️ 第二個要標明的外插：`recompute_slope_per_token` 是在位置 0–24,576 之間
   擬合的，用到 512K 是 21 倍外插。理論上 prefill 的每個 block 要對
   前面所有 block 做注意力，成本本來就隨位置線性成長，所以線性外插有依據；
   但這仍是外插不是量測，欄位 `extrapolated_position` 會記下最大位置。

用法：
  python code/m4_longctx.py --targets 8192 32768 131072 524288
"""
from __future__ import annotations
import argparse
import csv
import statistics
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


def scale_trace(trace: list[list[int]], s: int) -> list[list[int]]:
    """把每個 block 展開成 S 個連續 block。重用結構不變，長度與位置 ×S。"""
    if s == 1:
        return trace
    return [[b * s + k for b in req for k in range(s)] for req in trace]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="sata", choices=["sata", "nvme"])
    ap.add_argument("--model", default="llama-bf16", choices=list(MODEL_PROFILES))
    ap.add_argument("--trace", nargs="*", default=["toolagent", "conversation"])
    ap.add_argument("--targets", type=int, nargs="*",
                    default=[8192, 32768, 131072, 524288],
                    help="目標中位數請求長度（token）。S 由此推得")
    ap.add_argument("--cpu-gib", type=float, default=24.0)
    ap.add_argument("--lookup", choices=["prefix", "per-block"], default="prefix")
    ap.add_argument("--prefetch", action="store_true", default=True)
    ap.add_argument("--no-prefetch", dest="prefetch", action="store_false")
    ap.add_argument("--max-accesses", type=int, default=1_500_000,
                    help="放大後的 block 存取總數上限。S 越大取越少請求，"
                         "讓每個 S 的模擬工作量相當、執行時間可控")
    ap.add_argument("--out", default=str(OUT / "longctx.csv"))
    a = ap.parse_args()

    prof = profile(a.model)
    cm = load_cost_model(a.device, require_model_key=prof["cost_model_key"])
    gpu_blocks = prof["gpu_kv_tokens"] // BLOCK
    cpu_blocks = int(a.cpu_gib * 1024**3) // (prof["kv_bytes_per_token"] * BLOCK)
    sem = {"prefix_semantics": a.lookup == "prefix", "prefetch": a.prefetch}
    print(f"[剖面] {a.model}：GPU {prof['gpu_kv_tokens']:,} token = "
          f"{gpu_blocks:,} blocks；CPU {cpu_blocks:,} blocks")
    print(f"[語意] lookup={a.lookup} prefetch={a.prefetch}")
    print(f"[成本] 重算 = {cm.recompute_base:.3f} + "
          f"{cm.recompute_slope_per_token:.6f}×位置 ms/block；"
          f"CPU {cm.cpu:.3f}、SSD {cm.ssd:.3f} ms/block")
    xover = (cm.ssd - cm.recompute_base) / cm.recompute_slope_per_token
    print(f"[成本] SSD 與重算的交叉點在位置 {xover:,.0f} token："
          f"更前面重算較便宜，更後面搬運較便宜")

    rows: list[dict] = []
    for tname in a.trace:
        full = mooncake_trace(tname)
        # 🔴 S 必須用**整條 trace** 的中位數推，不能用前綴的。
        #    toolagent 前 4,000 個請求的中位數只有 208 token，
        #    整條卻是 6,346 token——用前綴推 S 會讓「目標 128K」這個標籤失真。
        med_full = statistics.median(len(r) for r in full) * BLOCK
        acc_full = sum(len(r) for r in full)
        print(f"\n{'=' * 100}\ntrace「{tname}」：整條 {len(full):,} 請求、"
              f"中位數長度 {med_full:,.0f} token（S 由此推得）")
        print(f"{'目標長度':>10s}{'S':>5s}{'請求數':>8s}{'實際中位數':>12s}"
              f"{'最長請求':>11s}{'壓力':>9s}{'重算佔比':>10s}{'best':>10s}"
              f"{'headroom':>10s}{'判定':>9s}")
        for target in a.targets:
            s = max(1, round(target / med_full))
            # 取前綴（保留時間順序與區域性），長度依 S 反比縮放，
            # 讓每個 S 的模擬存取總數相當
            n_req = len(full)
            if acc_full * s > a.max_accesses:
                n_req = max(200, int(len(full) * a.max_accesses / (acc_full * s)))
            tr = scale_trace(full[:n_req], s)
            med = statistics.median(len(r) for r in tr) * BLOCK
            longest = max(len(r) for r in tr) * BLOCK
            uniq = len({b for r in tr for b in r})
            sim = Sim(cm, gpu_blocks, cpu_blocks, ssd_blocks=10**9)
            res = {k: sim.run_online(tr, *v, **sem) for k, v in POLICIES.items()}
            res["oracle"] = sim.run_oracle(tr, True, True, **sem)
            best = min((k for k in res if k != "oracle"),
                       key=lambda k: res[k]["total_ms"])
            head = 100 * (res[best]["total_ms"] - res["oracle"]["total_ms"]) \
                / res[best]["total_ms"]
            # baseline 的時間有多少花在重算？這決定「省重算」的空間有多大
            bh = res[best]["hits"]
            verdict = ("GO" if head > 15 else
                       "MARGINAL" if head >= 5 else "NO_GO")
            rec_share = 100 * bh["drop"] / sum(bh.values())
            print(f"{target:>10,}{s:>5d}{len(tr):>8,}{med:>12,.0f}{longest:>11,.0f}"
                  f"{uniq / gpu_blocks:>8.0f}×{rec_share:>9.1f}%"
                  f"{best:>10s}{head:>9.2f}%{verdict:>9s}")
            for pol, v in res.items():
                e = v.get("evict", {})
                rows.append({
                    "ts": datetime.now().astimezone().isoformat(),
                    "trace": tname, "target_median_tokens": target,
                    "scale_factor": s, "actual_median_tokens": round(med),
                    "longest_request_tokens": round(longest),
                    "requests": len(tr), "unique_blocks": uniq,
                    "pressure_x": round(uniq / gpu_blocks, 2),
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
                    "cpu_budget_gib": a.cpu_gib,
                    "lookup": a.lookup, "prefetch": int(a.prefetch),
                    "device": a.device,
                    # 🔴 兩個必須跟著數字一起走的標記
                    "assumption": "reuse_structure_invariant_to_length",
                    "extrapolated_position_tokens": round(longest),
                    "cost_fit_max_position_tokens": 24576,
                    "cost_model": str(OUT / "cost_model.json"),
                })
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {p}  ({len(rows)} rows)")
    print("\n⚠️ 這批數字帶兩個標記，引用時必須一起講：")
    print("   assumption = reuse_structure_invariant_to_length（重用結構假設不隨長度改變）")
    print("   成本模型的位置擬合上限是 24,576 token，更長的位置是線性外插")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
