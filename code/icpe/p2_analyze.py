"""p2_analyze.py — ICPE P2（docs/h200/icpe/P2_g5b_cpu_busy_reseed.md §3）：G5b 在 CPU 層、GPU 忙，換 2 個新 seed 還選錯嗎？

  python code/icpe/p2_analyze.py --dir results/icpe_h200/p2

每格（模型 × 格式 × 配置）、每個新 seed（310、311）：r＝G5b（mode=cake）中位數 ÷ min(claim 中位數, loadall 中位數)。
兩個 seed 都 r > 1.05 → 穩；一個 → 不穩；都不是 → 沒重現。原本選錯的 10 格裡 ≥8 穩 → 穩定的平台差異；≤2 → 偶然；其他 → 部分穩定。
→ p2_cells.csv、p2_verdict.json（run_id、ts）。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime

import pandas as pd

ORIG = {  # results/m10_restore2_h200/report_13.json 的 q2（job 21109；G5b ÷ 最好）
    ("llama", "fp8", "E0"): 1.305, ("llama", "fp8", "E1"): 2.038, ("llama", "fp8", "R25"): 1.006,
    ("llama", "int4", "E0"): 1.009, ("llama", "int4", "E1"): 3.110, ("llama", "int4", "R25"): 1.988,
    ("qwen", "fp8", "E0"): 3.447, ("qwen", "fp8", "E1"): 3.507, ("qwen", "fp8", "R25"): 3.437,
    ("qwen", "int4", "E0"): 3.424, ("qwen", "int4", "E1"): 3.480, ("qwen", "int4", "R25"): 3.429}
SEEDS = (310, 311)


def run_id():
    return os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc"


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    rows, problems = [], []
    for (m, tag, cfg), r0 in ORIG.items():
        rs, fr, med = [], [], {}
        for L in SEEDS:
            pth = os.path.join(a.dir, f"{m}_{tag}_cpu_g5b_L{L}", "restores.csv")
            if not os.path.exists(pth):
                problems.append(f"缺 {pth}")
                continue
            d = pd.read_csv(pth)
            d = d[(d.config == cfg) & (d.bw_name == "cpu") & (d.gpu_state == "busy") & (d.launch == L)]
            if (d.contaminated_gpu.astype(str) == "True").any() or pd.to_numeric(d.kv_bad, errors="coerce").fillna(0).sum():
                problems.append(f"{m} {tag} {cfg} L{L}：有污染或 kv_bad")
            g = d[d["mode"] == "cake"].t_restore.median()
            c = d[d["mode"] == "claim"].t_restore.median()
            l = d[d["mode"] == "loadall"].t_restore.median()
            r = g / min(c, l)
            rs.append(r)
            fr.append(float((d[d["mode"] == "cake"].meet_rule_used == "none").mean()))
            med[L] = dict(g5b=g, cake=c, loadall=l)
        n_wrong = sum(x > 1.05 for x in rs)
        status = "穩" if n_wrong == 2 else ("不穩" if n_wrong == 1 else "沒重現")
        rows.append(dict(run_id=run_id(), ts=ts(), model=m, fmt=tag, config=cfg, orig_ratio=r0, orig_wrong=r0 > 1.05,
                         r_310=rs[0] if len(rs) > 0 else None, r_311=rs[1] if len(rs) > 1 else None,
                         frac_loadall_310=fr[0] if fr else None, frac_loadall_311=fr[1] if len(fr) > 1 else None,
                         cake_310=med.get(310, {}).get("cake"), loadall_310=med.get(310, {}).get("loadall"),
                         cake_311=med.get(311, {}).get("cake"), loadall_311=med.get(311, {}).get("loadall"),
                         status=status if len(rs) == 2 else "INCOMPLETE"))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(a.dir, "p2_cells.csv"), index=False)
    w = T[T.orig_wrong]
    n_st = int((w.status == "穩").sum())
    verdict = "穩定的平台差異" if n_st >= 8 else ("偶然" if n_st <= 2 else "部分穩定")
    out = dict(run_id=run_id(), ts=ts(), n_orig_wrong=int(len(w)), n_stable=n_st, verdict=verdict if not problems else f"{verdict}（有問題，見 problems）",
               controls=T[~T.orig_wrong][["model", "fmt", "config", "r_310", "r_311"]].to_dict("records"), problems=problems,
               rule="P2 §3：兩個新 seed 都 r>1.05＝穩；原本選錯的 10 格 ≥8 穩＝穩定的平台差異；≤2＝偶然；其他＝部分穩定")
    json.dump(out, open(os.path.join(a.dir, "p2_verdict.json"), "w"), ensure_ascii=False, indent=1, default=str)
    pd.set_option("display.width", 200)
    print(T.drop(columns=["run_id", "ts"]).round(3).to_string(index=False))
    print(json.dumps({k: out[k] for k in ("n_orig_wrong", "n_stable", "verdict", "problems")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
