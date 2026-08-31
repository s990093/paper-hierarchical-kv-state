#!/usr/bin/env python3
"""M4 附屬實驗：模擬器的**系統語意假設**各自把 headroom 推動多少。

動機：headroom 是 go/no-go 的唯一依據，而它不是實測，是模擬。
所以「模擬器假設了什麼」就跟「量到什麼」一樣需要被檢驗與公開。

兩個假設各自的方向是可以事先論證的：

  A. lookup 語意
     per-block（舊）：每個 block 獨立命中 → **低估** baseline 的重算量
     prefix（新，符合 vLLM）：缺口之後全部重算
     → 改成 prefix 之後 baseline 變慢，Oracle 也變慢（Oracle 的逐出
       沒有做前綴感知，同樣被罰），淨效果需要實測。

  B. 預取
     off（舊）：CPU/SSD 取回在存取當下才付錢 → **高估**卸載策略的成本
     on（新）：取回可與前一個請求的計算重疊
     → 只有用到 CPU/SSD 的策略會變快；full_gpu 完全不受影響。

兩者都**同時**套用在 Oracle 與所有 baseline 上——只給 Oracle 就是作弊。

用法：
  python code/m4_semantics_ablation.py --trace toolagent conversation
  python code/m4_semantics_ablation.py --pressure 0.5 1 2 4 8
"""
from __future__ import annotations
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from m4_oracle import (BLOCK, SIM_VERSION, MODEL_PROFILES, OUT, Sim, load_cost_model,
                       mooncake_trace, profile, zipf_trace)

POLICIES = {
    "full_gpu": ("lru", False, False),
    "cpu_lru": ("lru", True, False),
    "cpu_arc": ("arc", True, False),
    "tier_fs": ("lru", True, True),
}
SEMANTICS = [
    ("per-block/no-prefetch", False, False),   # 修正前的模型
    ("prefix/no-prefetch", True, False),       # 只修 lookup
    ("per-block/prefetch", False, True),       # 只修預取
    ("prefix/prefetch", True, True),           # 兩個都修（最接近 vLLM）
]


def one(sim: Sim, trace, prefix: bool, prefetch: bool,
        dest: str = "cost-aware") -> dict:
    res = {k: sim.run_online(trace, *a, prefix_semantics=prefix,
                             prefetch=prefetch)
           for k, a in POLICIES.items()}
    res["oracle"] = sim.run_oracle(trace, True, True, dest=dest,
                                   prefix_semantics=prefix, prefetch=prefetch)
    best = min((k for k in res if k != "oracle"), key=lambda k: res[k]["total_ms"])
    head = 100 * (res[best]["total_ms"] - res["oracle"]["total_ms"]) \
        / res[best]["total_ms"]
    if head < -1e-9:
        raise SystemExit(
            f"🔴 headroom = {head:.2f}% < 0：Oracle 輸給了 baseline {best}。"
            f"Oracle 是上界，這在定義上不可能，代表模擬器有錯（最可能是"
            f"目的地規則的邊際成本估計不對）。停止，不要把這個數字寫進任何地方。")
    return {"res": res, "best": best, "head": head}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="sata", choices=["sata", "nvme"])
    ap.add_argument("--trace", nargs="*", default=[],
                    choices=["conversation", "toolagent", "mooncake"])
    ap.add_argument("--pressure", type=float, nargs="*", default=[])
    ap.add_argument("--model", default="llama-bf16", choices=list(MODEL_PROFILES),
                    help="模型剖面，鎖住『預算 + KV/token + 成本模型』三者")
    ap.add_argument("--cpu-gib", type=float, default=24.0)
    ap.add_argument("--doc-tokens", type=int, default=4096)
    ap.add_argument("--requests", type=int, default=0,
                    help="0 = 自動取 10×文件數")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--oracle-dest", default="best",
                    choices=["best", "cost-aware", "cascade"],
                    help="Oracle 逐出後的目的地選擇。cost-aware=比較各去處在"
                         "『下次使用的位置』上的實際成本（放 SSD 5.536 ms 對上"
                         "重算 4.008+0.00021×位置，交叉點 7,278 token）；"
                         "cascade=無條件往下推（舊行為，會系統性低估 Oracle）")
    ap.add_argument("--out", default=str(OUT / "semantics_ablation.csv"))
    a = ap.parse_args()

    prof = profile(a.model)
    cm = load_cost_model(a.device, require_model_key=prof["cost_model_key"])
    gpu_blocks = prof["gpu_kv_tokens"] // BLOCK
    cpu_blocks = int(a.cpu_gib * 1024**3) // (prof["kv_bytes_per_token"] * BLOCK)
    print(f"[剖面] {a.model}：GPU {prof['gpu_kv_tokens']:,} token "
          f"= {gpu_blocks:,} blocks；KV {prof['kv_bytes_per_token']//1024} KiB/token")
    doc_blocks = a.doc_tokens // BLOCK

    cases: list[tuple[str, list[list[int]]]] = []
    for t in a.trace:
        cases.append((f"trace:{t}", mooncake_trace(t)))
    for r in a.pressure:
        # 🔴 請求數必須隨文件數放大，否則抽樣碰不到那麼多文件，
        #    「名目 8×」實際只有 2.8×。標籤一律以實際值標示。
        n_docs = max(2, round(r * gpu_blocks / doc_blocks))
        n_req = a.requests or max(400, 10 * n_docs)
        tr = zipf_trace(n_docs, doc_blocks, n_req, 0.9, a.seed)
        real = len({b for q in tr for b in q}) / gpu_blocks
        cases.append((f"pressure:{real:.1f}x(nom {r:g}x)", tr))

    rows: list[dict] = []
    for label, trace in cases:
        uniq = len({b for r_ in trace for b in r_})
        print(f"\n{'=' * 78}\n{label}：{len(trace):,} 請求、"
              f"{sum(len(r_) for r_ in trace):,} 次存取、"
              f"{uniq:,} 不重複 block（工作集 = {uniq / gpu_blocks:.1f}× 預算）")
        print(f"{'語意':24s}{'best baseline':>16s}{'baseline ms':>14s}"
              f"{'oracle ms':>13s}{'headroom':>10s}")
        base_head = None
        for name, prefix, prefetch in SEMANTICS:
            sim = Sim(cm, gpu_blocks, cpu_blocks, ssd_blocks=10**9)
            o = one(sim, trace, prefix, prefetch, a.oracle_dest)
            if base_head is None:
                base_head = o["head"]
            delta = o["head"] - base_head
            print(f"{name:24s}{o['best']:>16s}"
                  f"{o['res'][o['best']]['total_ms']:>14,.0f}"
                  f"{o['res']['oracle']['total_ms']:>13,.0f}"
                  f"{o['head']:>9.2f}%"
                  + (f"  ({delta:+.2f})" if delta else "  (基準)"))
            for pol, v in o["res"].items():
                rows.append({
                    "ts": datetime.now().astimezone().isoformat(),
                    "sim_version": SIM_VERSION,
                    "workload": label, "semantics": name,
                    "lookup": "prefix" if prefix else "per-block",
                    "oracle_dest": a.oracle_dest,
                    "prefetch": int(prefetch),
                    "policy": pol, "total_ms": round(v["total_ms"], 2),
                    "gpu_hits": v["hits"]["gpu"], "cpu_hits": v["hits"]["cpu"],
                    "ssd_hits": v["hits"]["ssd"], "recompute": v["hits"]["drop"],
                    "best_baseline": o["best"],
                    "oracle_headroom_pct": round(o["head"], 3)
                    if pol == "oracle" else "",
                    "verdict": ("GO" if o["head"] > 15 else
                                "MARGINAL_ASK_HUMAN" if o["head"] >= 5
                                else "NO_GO") if pol == "oracle" else "",
                    "model_profile": a.model,
                    "gpu_budget_tokens": prof["gpu_kv_tokens"],
                    "cpu_budget_gib": a.cpu_gib,
                    "unique_blocks": uniq, "requests": len(trace),
                    "device": a.device, "seed": a.seed,
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
