"""p12_compare.py — ICPE P1-2 §4（docs/h200/icpe/P1-2_cpu_symmetric_reruns.md）：每一項 H200 常數的判定字／數字和 3090 分支（MI300X 常數）並排，標一致／不一致。

  python code/icpe/p12_compare.py            → results/icpe_h200/p12/p12_compare.{csv,json}

兩邊都讀同一種檔、用同一個函式取判定字（X1 的 verdict.csv、X5 的 verdict json、X4 的 verdicts.csv、X5 的 le32k）。
3090 分支沒有檔的（A1：X1 截尾的判定只寫在 X1 §4.7 的表），照文件原文列，`ref_src` 寫出處。只讀檔，不重算任何模擬。
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
H200 = os.path.join(REPO, "results", "icpe_h200")
REF = os.path.join(REPO, "results", "p3_explore")          # 3090 分支（MI300X 常數）
OUT = os.path.join(H200, "p12")


def run_id():
    return os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc"


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def x1_word(path, rule):
    """X1 的整體判定字（同 p01_tools._x1_word，但 FreeInference 的 rule 是 session_id）＋熱組數、各成分最大 headroom（Cake）。"""
    v = pd.read_csv(path)
    v = v[v.rule == rule]
    w = v.verdict_rule.unique()
    assert len(w) == 1, (path, w)
    return dict(word=str(w[0]), hot_groups=int(v.n_groups_hot.sum()),
                hot_by_comp={r.component: int(r.n_groups_hot) for r in v.itertuples()},
                max_cake={r.component: round(float(r.max_headroom_cake), 4) for r in v.itertuples()},
                src=str(v.run_id.iloc[0]))


def x5_word(path):
    d = json.load(open(path))
    return dict(word=d["verdict_registered"], main_ge15=d["n_main_cells_ge15"], main_max=round(d["main_headroom_max"], 4),
                reg_max=round(d["reg_headroom_max"], 4), src=d["run_id"])


def x4_word(path, metric):
    v = pd.read_csv(path)
    v = v[v.metric == metric].set_index("group")
    loose, strict = v.loc["rule=loose"], v.loc["rule=strict"]
    order = {"死路": 0, "可能": 1, "有看頭": 2}
    cons = min((loose.verdict, strict.verdict), key=order.get)   # X4 §4.5：兩種規則照較保守的
    return dict(word=cons, loose=loose.verdict, strict=strict.verdict, max_loose=round(float(loose.max_hr_med), 4),
                max_strict=round(float(strict.max_hr_med), 4), n_eligible=int(v.loc["ALL"].n_eligible), src=str(v.run_id.iloc[0]))


def le32k(path):
    d = pd.read_csv(path)
    out = {}
    for rule, g in d.groupby("rule"):
        gg = g[g.headroom_all >= 0.15]
        out[rule] = dict(cells_ge15_all=len(gg), still_ge15_le32k=int((gg.headroom_le32k >= 0.15).sum()),
                         le32k_min=round(float(gg.headroom_le32k.min()), 4) if len(gg) else None,
                         le32k_max=round(float(gg.headroom_le32k.max()), 4) if len(gg) else None,
                         any_main_le32k_ge15=int((g.headroom_le32k >= 0.15).sum()))
    out["word_like"] = "有看頭" if any(v["any_main_le32k_ge15"] for v in out.values() if isinstance(v, dict)) else "沒有 ≥15% 的格"
    return dict(word=out["word_like"], detail={k: v for k, v in out.items() if k != "word_like"}, src=str(d.run_id.iloc[0]))


def main():
    rows = []

    def add(item, desc, h, r, ref_src=None):
        rows.append(dict(run_id=run_id(), ts=ts(), item=item, desc=desc, h200_word=h["word"], ref_word=r["word"],
                         same="一致" if h["word"] == r["word"] else "不一致",
                         h200=json.dumps({k: v for k, v in h.items() if k != "word"}, ensure_ascii=False),
                         ref=json.dumps({k: v for k, v in r.items() if k != "word"}, ensure_ascii=False),
                         ref_src=ref_src or r.get("src")))

    sv0 = os.path.join(H200, "p01", "sim_V0", "x1")
    # A1：3090 只有文件（X1 §4.7 的表）
    add("A1-H1800", "X1 截尾 H=1,800 s（rule=min）", x1_word(os.path.join(sv0, "x1h1800_verdict.csv"), "min"),
        dict(word="有看頭", hot_groups=21, note="O-DROP 只剩 1 組"), "docs/phase3_20261017/explore/X1_oracle_decomposition.md §4.7")
    add("A1-H3600", "X1 截尾 H=3,600 s（rule=min）", x1_word(os.path.join(sv0, "x1h3600_verdict.csv"), "min"),
        dict(word="有看頭", hot_groups=20, note="只有 O-ORD、O-PF"), "docs/phase3_20261017/explore/X1_oracle_decomposition.md §4.7")
    for pre, desc in (("x1fi", "X1 FreeInference（6 週，Cake）"), ("x1fiuser", "X1 FreeInference，只算使用者發起")):
        add(f"A2-{pre}", desc, x1_word(os.path.join(OUT, "x1", f"{pre}_verdict.csv"), "session_id"),
            x1_word(os.path.join(REF, "x1", f"{pre}_verdict.csv"), "session_id"))
    for sfx, item, desc in (("_strict", "B1", "X5 嚴格重建規則"), ("_fi", "B2", "X5 FreeInference"),
                            ("_cens1800", "B3-sub1800", "X5 設限 子樹 H1800"), ("_strict_cens1800", "B3-str1800", "X5 設限 嚴格 H1800"),
                            ("_fi_cens1800", "B3-fi1800", "X5 設限 FI H1800"), ("_fi_cens3600", "B3-fi3600", "X5 設限 FI H3600"),
                            ("_loadonly", "B4", "X5 只載入還原")):
        add(item, desc, x5_word(os.path.join(OUT, "x5", f"x5_verdict{sfx}.json")), x5_word(os.path.join(REF, "x5", f"x5_verdict{sfx}.json")))
    add("B5", "X5 只算 ≤32K 的回來（主設定 ≥15% 的格還剩幾格）", le32k(os.path.join(OUT, "x5", "x5_headroom_le32k.csv")),
        le32k(os.path.join(REF, "x5", "x5_headroom_le32k.csv")))
    for metric in ("p99", "mean", "p90"):
        add(f"C1-{metric}", f"X4 f×0.5 {metric}（寬鬆／嚴格取較保守；p99 是登記判定，其餘只報告）",
            x4_word(os.path.join(OUT, "x4", "x4_f05_verdict_verdicts.csv"), metric),
            x4_word(os.path.join(REF, "x4", "x4_f05_verdict_verdicts.csv"), metric))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "p12_compare.csv"), index=False)
    out = dict(run_id=run_id(), ts=ts(), n_items=len(d), n_same=int((d.same == "一致").sum()),
               differ=d[d.same == "不一致"].item.tolist(), rule="P1-2 §4：H200 判定字與 3090 分支（MI300X 常數）的並排；不是 go/no-go")
    json.dump(out, open(os.path.join(OUT, "p12_compare.json"), "w"), ensure_ascii=False, indent=1)
    for r in rows:
        print(f"{r['item']:<12} H200 {r['h200_word']:<8} 3090 {r['ref_word']:<8} {r['same']}  {r['desc']}")
        print(f"             H200 {r['h200']}\n             3090 {r['ref']}")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
