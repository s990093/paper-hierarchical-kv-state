#!/usr/bin/env python3
"""M4 附屬實驗：**GPU 預算多小，多階層才開始有價值？**

為什麼要問這個。2026-08-31 的診斷發現：在目前的設定下，
Oracle 的逐出有 **100% 是「免費」的**——被逐出的 block 之後再也用不到，
丟掉成本為 0。也就是說 Bélády 永遠找得到不用付錢的犧牲者，
**CPU 與 SSD 階對最佳解的貢獻剛好是 0**。
既然如此，先前量到的 headroom（4–7%）全部來自「GPU 階內要逐出誰」，
**不能拿來支持論文的六階動作空間**。

成因也量到了：逐出當下 GPU 裡有 36–91% 是已經死掉的 block
（真實 trace 有 45–63% 的 block 一生只被存取一次）。
GPU 預算相對於**活著的**工作集太大了。

所以真正該掃的自變數是 **GPU 預算**：預算縮到多小，
死 block 的存量才會被吃光、Oracle 才被迫下放到 CPU？
那個轉折點就是「階層式管理開始有意義」的門檻，
也就是論文該主張的適用範圍。

輸出兩條曲線：
  1. 免費逐出佔比 vs 預算  —— 機制（為什麼）
  2. headroom vs 預算      —— 後果（值不值得做）

用法：
  python code/m4_budget_sweep.py --trace toolagent conversation
"""
from __future__ import annotations
import argparse
import csv
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
    ap.add_argument("--trace", nargs="*",
                    default=["toolagent", "conversation"],
                    choices=["conversation", "toolagent", "mooncake"])
    ap.add_argument("--model", default="llama-bf16", choices=list(MODEL_PROFILES),
                    help="模型剖面，鎖住『預算 + KV/token + 成本模型』三者")
    ap.add_argument("--budgets", type=int, nargs="*", default=None,
                    help="GPU KV 預算（token）。預設由剖面的實測容量逐半下探")
    ap.add_argument("--cpu-gib", type=float, default=24.0)
    ap.add_argument("--lookup", choices=["prefix", "per-block"], default="prefix")
    ap.add_argument("--prefetch", action="store_true", default=True)
    ap.add_argument("--no-prefetch", dest="prefetch", action="store_false")
    ap.add_argument("--oracle-dest", default="cost-aware",
                    choices=["cost-aware", "cascade"],
                    help="Oracle 逐出後的目的地選擇。cost-aware=比較各去處在"
                         "『下次使用的位置』上的實際成本（放 SSD 5.536 ms 對上"
                         "重算 4.008+0.00021×位置，交叉點 7,278 token）；"
                         "cascade=無條件往下推（舊行為，會系統性低估 Oracle）")
    ap.add_argument("--out", default=str(OUT / "budget_sweep.csv"))
    a = ap.parse_args()

    prof = profile(a.model)
    full = prof["gpu_kv_tokens"]
    budgets = a.budgets or [full] + [full // (2 ** k) for k in range(1, 9)]
    cm = load_cost_model(a.device, require_model_key=prof["cost_model_key"])
    sem = {"prefix_semantics": a.lookup == "prefix", "prefetch": a.prefetch}
    cpu_blocks = int(a.cpu_gib * 1024**3) // (prof["kv_bytes_per_token"] * BLOCK)
    print(f"[剖面] {a.model}：實測 GPU 預算 {full:,} token；"
          f"KV {prof['kv_bytes_per_token']//1024} KiB/token")
    print(f"[語意] lookup={a.lookup} prefetch={a.prefetch}；"
          f"CPU 階 = {a.cpu_gib} GiB = {cpu_blocks:,} blocks")

    rows: list[dict] = []
    for tname in a.trace:
        trace = mooncake_trace(tname)
        uniq = len({b for r in trace for b in r})
        acc = sum(len(r) for r in trace)
        print(f"\n{'=' * 96}\ntrace「{tname}」：{len(trace):,} 請求、"
              f"{acc:,} 次存取、{uniq:,} 不重複 block")
        print(f"{'GPU 預算 (token)':>17s}{'blocks':>9s}{'壓力':>8s}"
              f"{'免費逐出':>10s}{'→CPU':>9s}{'oracle CPU 命中':>16s}"
              f"{'best baseline':>15s}{'headroom':>10s}")
        for bt in budgets:
            gb = bt // BLOCK
            sim = Sim(cm, gb, cpu_blocks, ssd_blocks=10**9)
            res = {k: sim.run_online(trace, *v, **sem) for k, v in POLICIES.items()}
            res["oracle"] = sim.run_oracle(trace, True, True,
                                           dest=a.oracle_dest, **sem)
            best = min((k for k in res if k != "oracle"),
                       key=lambda k: res[k]["total_ms"])
            head = 100 * (res[best]["total_ms"] - res["oracle"]["total_ms"]) \
                / res[best]["total_ms"]
            ev = res["oracle"]["evict"]
            nev = sum(ev.values())
            free_pct = 100 * ev["free"] / nev if nev else float("nan")
            print(f"{bt:>17,}{gb:>9,}{uniq / gb:>7.1f}×"
                  f"{free_pct:>9.1f}%{ev['to_cpu'] + ev['swap_cpu']:>9,}"
                  f"{res['oracle']['hits']['cpu']:>16,}"
                  f"{best:>15s}{head:>9.2f}%")
            for pol, v in res.items():
                e = v.get("evict", {})
                rows.append({
                    "ts": datetime.now().astimezone().isoformat(),
                    "trace": tname, "gpu_budget_tokens": bt, "gpu_blocks": gb,
                    "unique_blocks": uniq, "accesses": acc,
                    "requests": len(trace),
                    "pressure_x": round(uniq / gb, 3),
                    "policy": pol, "total_ms": round(v["total_ms"], 2),
                    "gpu_hits": v["hits"]["gpu"], "cpu_hits": v["hits"]["cpu"],
                    "ssd_hits": v["hits"]["ssd"], "recompute": v["hits"]["drop"],
                    "evict_free": e.get("free", ""),
                    "evict_to_cpu": e.get("to_cpu", ""),
                    "evict_to_ssd": e.get("to_ssd", ""),
                    "evict_swap_cpu": e.get("swap_cpu", ""),
                    "evict_lost": e.get("lost", ""),
                    "evict_free_pct": round(free_pct, 3) if pol == "oracle" else "",
                    "best_baseline": best,
                    "oracle_headroom_pct": round(head, 3) if pol == "oracle" else "",
                    "verdict": ("GO" if head > 15 else
                                "MARGINAL_ASK_HUMAN" if head >= 5
                                else "NO_GO") if pol == "oracle" else "",
                    "model_profile": a.model,
                    "lookup": a.lookup, "prefetch": int(a.prefetch),
                    "oracle_dest": a.oracle_dest,
                    "cpu_budget_gib": a.cpu_gib, "device": a.device,
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
