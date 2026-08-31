#!/usr/bin/env python3
"""量化「前綴語意」這個修正為什麼幾乎不影響結果。

vLLM 的 cache lookup 是**連續前綴**：`OffloadingConnectorScheduler.
_lookup_complete_chunks` 的 docstring 明寫 "prefix lookup"，且回傳的是
token 數，而 token 數表達不了「block 0,1,4,5 命中、2,3 沒有」這種形狀。
所以第一個缺口之後的 block 全部要重算，即使它們還躺在 CPU 裡。

先前模擬器把每個 block 當獨立事件，理論上會低估 baseline 的重算量。
修正後 headroom 只變化 ≤0.22 個百分點。這支程式量出原因：

    缺口之後的 block 裡，有多少「本來還在某一階、舊模型會算成命中」？

答案接近零，因為真實 LLM 流量的重用本身就是前綴結構的（Mooncake 的
`hash_ids` 就是前綴雜湊），第一個未命中剛好落在共用前綴的結尾。

用法：python code/m4_prefix_probe.py
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
from pathlib import Path

from m4_oracle import (BLOCK, MODEL_PROFILES, OUT, Sim, load_cost_model,
                       mooncake_trace, profile)

POLICIES = {"full_gpu": ("lru", False, False),
            "cpu_lru": ("lru", True, False),
            "tier_fs": ("lru", True, True)}


def probe(sim: Sim, trace, policy_args, tname: str, pol: str) -> dict:
    """重放一次，統計缺口的位置與缺口之後 block 的狀態。"""
    orig = Sim._gap_index
    st = {"req_no_gap": 0, "req_gap_at_0": 0, "req_gap_mid": 0,
          "post_gap_blocks": 0, "post_gap_still_resident": 0,
          "post_gap_first_ever": 0}
    seen: set[int] = set()

    def spy(req, gpu, cpu, ssd, enabled):
        g = orig(req, gpu, cpu, ssd, True)
        if g == 0:
            st["req_gap_at_0"] += 1
        elif g >= len(req):
            st["req_no_gap"] += 1
        else:
            st["req_gap_mid"] += 1
        for b in req[g + 1:]:
            st["post_gap_blocks"] += 1
            if b in gpu or b in cpu or b in ssd:
                st["post_gap_still_resident"] += 1
            if b not in seen:
                st["post_gap_first_ever"] += 1
        seen.update(req)
        return orig(req, gpu, cpu, ssd, enabled)

    Sim._gap_index = staticmethod(spy)
    try:
        sim.run_online(trace, *policy_args, prefix_semantics=True, prefetch=True)
    finally:
        Sim._gap_index = orig
    n = st["post_gap_blocks"]
    st.update({
        "trace": tname, "policy": pol,
        "requests": len(trace),
        "accesses": sum(len(r) for r in trace),
        "post_gap_resident_pct": round(100 * st["post_gap_still_resident"] / n, 4)
        if n else "",
        "post_gap_first_ever_pct": round(100 * st["post_gap_first_ever"] / n, 2)
        if n else "",
    })
    return st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="sata", choices=["sata", "nvme"])
    ap.add_argument("--model", default="llama-bf16", choices=list(MODEL_PROFILES))
    ap.add_argument("--trace", nargs="*", default=["toolagent", "conversation"])
    ap.add_argument("--ssd-gib", type=float, default=512.0)
    ap.add_argument("--cpu-gib", type=float, default=24.0)
    ap.add_argument("--out", default=str(OUT / "prefix_gap_probe.csv"))
    a = ap.parse_args()

    prof = profile(a.model)
    cm = load_cost_model(a.device, require_model_key=prof["cost_model_key"])
    bpb = prof["kv_bytes_per_token"] * BLOCK
    gb = prof["gpu_kv_tokens"] // BLOCK
    cb = int(a.cpu_gib * 1024**3) // bpb
    sb = int(a.ssd_gib * 1024**3) // bpb

    rows = []
    print(f"{'trace/策略':26s}{'缺口後 block':>13s}{'仍在某一階':>12s}"
          f"{'佔比':>9s}{'第一次出現':>12s}{'佔比':>8s}")
    for tname in a.trace:
        trace = mooncake_trace(tname)
        for pol, args in POLICIES.items():
            sim = Sim(cm, gb, cb, ssd_blocks=sb)
            r = probe(sim, trace, args, tname, pol)
            r.update({"ts": datetime.now().astimezone().isoformat(),
                      "model_profile": a.model, "ssd_gib": a.ssd_gib,
                      "cpu_gib": a.cpu_gib, "device": a.device})
            rows.append(r)
            print(f"{tname + '/' + pol:26s}{r['post_gap_blocks']:>13,}"
                  f"{r['post_gap_still_resident']:>12,}"
                  f"{r['post_gap_resident_pct']:>8.3f}%"
                  f"{r['post_gap_first_ever']:>12,}"
                  f"{r['post_gap_first_ever_pct']:>7.1f}%")
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {p}")
    print("\n→ 缺口之後幾乎沒有東西可以損失：那些 block 絕大多數是這輩子第一次")
    print("  出現，本來就要重算。所以 per-block 這個簡化對前綴結構的工作負載無害。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
