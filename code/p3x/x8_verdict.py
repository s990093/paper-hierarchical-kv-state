"""x8_verdict.py — 照 X8_agents_longctx.md §2.3 的字面判定（＋§2.5 事後補充的各版本），並產生摘要表

輸入（results/p3_explore/x8/）：
  x8_headroom.csv        W0–W4（Mooncake 兩種重建 R-carry／R-strict），三種長度，H_s ∈ {0,1800,3600}
  x8_headroom_fi.csv     W5 FreeInference（fi、fi_clip），原始長度，H_s ∈ {0,1800,3600}
  x8_headroom_dedup.csv  共用前綴只存一次的敏感度（只報告，H＝0）
輸出：x8_verdict.json、x8_summary.csv

版本：
  A  原登記（W0–W4；Mooncake 每格取 R-carry 與 R-strict 中 headroom 較小的那個；H＝0）  ← 總結以這個為準
  B  A ＋ FreeInference（fi；登記之後才加）
  A/B@H  只評估離觀察窗結束 ≥H 秒的決定（H＝1800、3600；事後加的穩健性檢查）
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
import p3_platform as PF  # noqa: E402  （P3_CONST：沒設＝MI300X；h200＝results/*_h200）
OUT = PF.explore_out("x8")
KEYS = ["length", "G_GiB", "C_GiB", "slow", "gpu", "H_s"]


def _p(name):
    """結果檔超過 1 MB 的會 gzip（CLAUDE.md §2）；兩種都接受。"""
    a = os.path.join(OUT, name)
    return a if os.path.exists(a) else a + ".gz"


def conservative_mooncake(h):
    """Mooncake：同一格兩種重建規則取 headroom 較小的那一列（CI 跟著那一列）；session 數取兩者較小。"""
    out = [h[~h.workload.str.startswith("mc_")]]
    for base in ("mc_conv", "mc_tool"):
        a = h[h.workload == base].set_index(KEYS)
        b = h[h.workload == base + "_s"].set_index(KEYS)
        j = a.join(b, lsuffix="_c", rsuffix="_s", how="inner")
        pick_s = j.headroom_s < j.headroom_c
        rows = pd.DataFrame(index=j.index)
        for col in ("headroom", "ci_lo", "ci_hi", "abs_gap_ms", "hr_restore_only", "share_of_headroom_from_whether"):
            rows[col] = np.where(pick_s, j[col + "_s"], j[col + "_c"])
        rows["n_sess"] = np.minimum(j.n_sess_c, j.n_sess_s)
        rows["rule_used"] = np.where(pick_s, "strict", "carry")
        rows["workload"] = base + "(min of rules)"
        out.append(rows.reset_index())
    return pd.concat(out, ignore_index=True)


def judge(df, eligible, real):
    """§2.3：有看頭＝≥1,000 session 的工作負載、某 (G,慢層,GPU) 在 ≥2 種 CPU 大小 headroom≥15% 且 CI 下界≥10%；
    死路＝所有真實工作負載所有格子 <5%；其他＝可能。"""
    o = df[df.length == "orig"]
    sess = o.groupby("workload").n_sess.min().to_dict()
    el = o[o.workload.isin(eligible) & o.workload.map(lambda w: sess.get(w, 0) >= 1000)]
    ok = (el.headroom >= 0.15) & (el.ci_lo >= 0.10)
    g = el.assign(ok=ok).groupby(["workload", "G_GiB", "slow", "gpu"]).agg(
        n_ok=("ok", "sum"), hr_max=("headroom", "max"), hr_min=("headroom", "min")).reset_index()
    good = g[g.n_ok >= 2]
    rr = o[o.workload.isin(real)]
    dead = bool(len(rr)) and bool((rr.headroom.fillna(0) < 0.05).all())
    v = "有看頭" if len(good) else ("死路" if dead else "可能")
    return dict(verdict=v, sessions=sess, n_qualifying_groups=int(len(good)),
                qualifying_groups=good.to_dict("records"),
                qualifying_groups_local_ssd=good[good.slow == "local"].to_dict("records"),
                max_headroom_by_workload=rr.groupby("workload").headroom.max().round(4).to_dict(),
                frac_cells_ge15_by_workload=rr.assign(x=rr.headroom >= 0.15).groupby("workload").x.mean().round(3).to_dict(),
                frac_cells_lt5_by_workload=rr.assign(x=rr.headroom < 0.05).groupby("workload").x.mean().round(3).to_dict())


def main():
    h = pd.read_csv(_p("x8_headroom.csv"))
    fi_p = _p("x8_headroom_fi.csv")
    hf = pd.read_csv(fi_p) if os.path.exists(fi_p) else None
    runs = sorted(set(h.run_id) | (set(hf.run_id) if hf is not None else set()))
    hc = conservative_mooncake(h)
    A_el = ["mc_conv(min of rules)", "mc_tool(min of rules)", "bl_coder"]
    A_real = A_el + ["cw_orig", "cw_stack"]
    out = dict(rule="X8_agents_longctx.md §2.3（2026-10-17T09:43:22Z）＋§2.5（10:06Z）", source_run_ids=runs,
               ts=datetime.now().astimezone().isoformat(timespec="seconds"))
    for H in (0, 1800, 3600):
        dA = hc[hc.H_s == H]
        out[f"A_H{H}"] = judge(dA, A_el, A_real)
        # 每種重建規則單獨判（只報告）
        for rule, names in (("carry", ["mc_conv", "mc_tool"]), ("strict", ["mc_conv_s", "mc_tool_s"])):
            d1 = h[(h.H_s == H)]
            out[f"A_H{H}_rule_{rule}"] = judge(d1, names + ["bl_coder"], names + ["bl_coder", "cw_orig", "cw_stack"])["verdict"]
        if hf is not None:
            dB = pd.concat([dA, hf[(hf.H_s == H) & (hf.workload == "fi")]], ignore_index=True)
            out[f"B_H{H}"] = judge(dB, A_el + ["fi"], A_real + ["fi"])
            out[f"fi_only_H{H}"] = judge(hf[(hf.H_s == H) & (hf.workload == "fi")], ["fi"], ["fi"])
            out[f"fi_clip_only_H{H}"] = judge(hf[(hf.H_s == H) & (hf.workload == "fi_clip")], ["fi_clip"], ["fi_clip"])
    out["final_registered_verdict_A_H0"] = out["A_H0"]["verdict"]
    out["most_conservative_A"] = min((out[f"A_H{H}"]["verdict"] for H in (0, 1800, 3600)),
                                     key=lambda v: {"死路": 0, "可能": 1, "有看頭": 2}[v])
    # 事後敏感度：H-Turn2 也算進最好的固定規則（點估計，沒有 CI）
    o = hc[(hc.H_s == 0)]
    t = h[(h.H_s == 0) & (h.length == "orig")]
    out["posthoc_HTurn2"] = dict(
        note="H-Turn2 是看過結果才加的規則，只報告",
        frac_cells_Turn2_beats_LRU=float((t.Turn2_gain_vs_fixed > 0).mean()),
        Turn2_gain_vs_LRU_by_workload_median=t.groupby("workload").Turn2_gain_vs_fixed.median().round(4).to_dict(),
        hr_vs_Turn2_by_workload_max=t.groupby("workload").hr_vs_Turn2.max().round(4).to_dict())
    viol = pd.concat([h, hf]) if hf is not None else h
    vv = viol[viol.oracle_violations.fillna("") != ""]
    out["oracle_violation_cells"] = int(len(vv))
    out["oracle_violation_examples"] = vv.head(10)[["workload", "length", "G_GiB", "C_GiB", "slow", "gpu", "H_s",
                                                    "oracle_violations"]].to_dict("records")
    dp = _p("x8_headroom_dedup.csv")
    if os.path.exists(dp):
        hd = pd.read_csv(dp)
        hd["H_s"] = 0
        od = hd[hd.length == "orig"]
        out["dedup_sensitivity"] = dict(note="共用前綴只存一次、永遠在 GPU（只報告，R-carry 重建）",
                                        source_run_ids=sorted(hd.run_id.unique()),
                                        verdict_if_used=judge(od, list(od.workload.unique()), list(od.workload.unique()))["verdict"],
                                        max_headroom_by_workload=od.groupby("workload").headroom.max().round(4).to_dict(),
                                        local_idle_G16=od[(od.slow == "local") & (od.gpu == "idle") & (od.G_GiB == 16)][
                                            ["workload", "C_GiB", "headroom", "ci_lo"]].round(4).to_dict("records"))
    json.dump(out, open(os.path.join(OUT, "x8_verdict.json"), "w"), ensure_ascii=False, indent=1, default=str)
    allh = pd.concat([h, hf], ignore_index=True) if hf is not None else h
    s = allh.pivot_table(index=["workload", "length", "H_s", "slow", "gpu", "G_GiB"], columns="C_GiB",
                         values="headroom").round(4)
    s.columns = [f"hr_C{c}" for c in s.columns]
    s = s.reset_index()
    s.insert(0, "ts", out["ts"])
    s.insert(0, "run_id", "|".join(runs))
    s.to_csv(os.path.join(OUT, "x8_summary.csv"), index=False)
    for k in [k for k in out if k.startswith(("A_H", "B_H", "fi_"))]:
        v = out[k]
        print(k, v if isinstance(v, str) else (v["verdict"], v["n_qualifying_groups"], v["sessions"]))
    print("final", out["final_registered_verdict_A_H0"], "most conservative A", out["most_conservative_A"])


if __name__ == "__main__":
    main()
