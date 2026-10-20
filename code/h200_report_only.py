"""h200_report_only.py — 55／60／88（只報告）在 H200 的分析（13_h200_repro §15；封存規則同 REPORT.md 附錄 D，節點換成 H200 的）。

重用 MI300X 凍結的程式：code/report_stage2/figs_report_only.py 的 analyse()（FP8／INT4 端點的 T1、LongBench；Llama INT4 混合配置的 T1），
統計是 code/m10_analyze.py 的 cond_rows／eps_check／paired／cluster_boot。這裡只換目錄，並加 H200 主線才有的兩項：
  Qwen FP8 混合配置的 T1（FP8 Step 2 的 Qwen，H200 執行過）、Qwen fp8_head 的逐 block 敏感度地圖。
每一項標「主線」（H200 執行過的節點）或「對照」（照 MI300X 的路，H200 沒執行；§10）。只用 CPU。

  python code/h200_report_only.py   → results/m10_precision_h200/analysis/report_only/{report_only.json, tables_report_only.md}
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "report_stage2"))
sys.path.insert(0, HERE)
import common as CM  # noqa: E402  （擋掉 torch／GPU）
import figs_report_only as F  # noqa: E402
import m10_analyze as A  # noqa: E402

PREC = "m10_precision_h200"
F.P = (PREC,)
OUT = os.path.join(CM.REPO, "results", PREC, "analysis", "report_only")
MAIN, CMP = "主線", "對照"


def mix_t1(model, sealed, strip):
    """混合配置（F25／M25／B25／R25）的 T1：對 E0（四種讀法），以及讀法 B 時對 R25（同 figs_report_only 的 int4mix 那段）。"""
    base = F._rows(F._shards(model, "T1"))
    ref = {k: r for k, r in A.cond_rows(base, "E0", "A").items() if r["task"] == "T1"}
    rm = F._rows(F._sealed(sealed), strip_fmt=strip)
    out, vs_r = {}, {}
    for cfg in ("F25", "M25", "B25", "R25"):
        out[cfg] = {c: A.eps_check(ref, x, main_tasks=["T1"]) for c in A.CONDS if (x := A.cond_rows(rm, cfg, c))}
    for cfg in ("F25", "M25", "B25"):
        vs_r[cfg] = {}
        for cond in ("B-idle", "B-busy"):
            d, n, *_ = A.paired(A.cond_rows(rm, "R25", cond), A.cond_rows(rm, cfg, cond), tasks=["T1"])
            p, lo, hi = A.cluster_boot(d)
            vs_r[cfg][cond] = {"diff": round(p, 4), "ci95": [round(lo, 4), round(hi, 4)], "n_rows": n}
    return out, vs_r


def sensmap(model, fmt):
    """逐 block 敏感度地圖（同 figs_report_only.f31 的數字部分，不畫圖）。"""
    d = CM.rp(PREC, "quality_sealed")
    rows = F.read_csv(os.path.join(d, f"sensmap_{model}_{fmt}_s0.csv")) + F.read_csv(os.path.join(d, f"sensmap_{model}_{fmt}_s1.csv"))
    by_block, dl_block = defaultdict(dict), defaultdict(dict)
    hit, miss = defaultdict(list), defaultdict(list)
    for r in rows:
        b, iid, kl = int(r["block"]), (r["item_id"], r["turn"]), float(r["kl_vs_e0"])
        by_block[b][iid] = [kl]
        dl_block[b][iid] = [float(r["gold_logprob"]) - float(r["gold_logprob_e0"])]
        needles = json.loads(r["needle_blocks"]) if r["needle_blocks"] not in ("", "[]") else []
        (hit if b in needles else miss)[r["item_id"]].append(kl)
    blocks = sorted(by_block)
    kl = {b: A.cluster_boot(by_block[b], n_boot=2000) for b in blocks}
    dl = {b: A.cluster_boot(dl_block[b], n_boot=2000) for b in blocks}
    return {"run_ids": sorted({r["run_id"] for r in rows}), "n_items": len({r["item_id"] for r in rows}), "n_rows": len(rows),
            "per_block_mean_kl": {b: round(kl[b][0], 6) for b in blocks},
            "per_block_kl_ci": {b: [round(kl[b][1], 6), round(kl[b][2], 6)] for b in blocks},
            "per_block_mean_dlogp": {b: round(dl[b][0], 5) for b in blocks},
            "answer_block_kl": list(A.cluster_boot(hit, n_boot=2000)), "other_block_kl": list(A.cluster_boot(miss, n_boot=2000)),
            "answer_rows": sum(len(v) for v in hit.values()), "other_rows": sum(len(v) for v in miss.values())}


def _pp(e):
    return "—" if e is None else f"{100 * e['diff']:+.1f} [{100 * e['ci95'][0]:+.1f}, {100 * e['ci95'][1]:+.1f}]"


def main():
    os.makedirs(OUT, exist_ok=True)
    res = F.analyse()                                      # MI300X 的同一套：FP8／INT4 端點、Llama INT4 混合
    J = {"rule": "13_h200_repro §15（REPORT.md 附錄 D 的規則，節點換成 H200 的）", "items": {}}
    for (m, tier, kind), v in res.items():
        tag = MAIN if tier in ("fp8", "e0acc") else CMP
        J["items"][f"{m}|{tier}|{kind}"] = {"label": tag, "result": v}
    q_out, q_vr = mix_t1("qwen", "fp8mix_qwen_T1", None)
    J["items"]["qwen|fp8mix|T1"] = {"label": MAIN, "result": q_out}
    J["items"]["qwen|fp8mix_vsR25|T1"] = {"label": MAIN, "result": q_vr}
    J["items"]["qwen|sensmap|fp8_head"] = {"label": MAIN, "result": sensmap("qwen", "fp8_head")}
    J["items"]["llama|sensmap|int4_kivi"] = {"label": CMP, "result": sensmap("llama", "int4_kivi")}
    J["not_analysed"] = ["sensmap_llama_fp8_token", "sensmap_qwen_int4_kivi", "fp8mix_llama_*", "int4mix_qwen_*"]
    json.dump(json.loads(json.dumps(J, default=float)), open(os.path.join(OUT, "report_only.json"), "w"), indent=1, ensure_ascii=False)

    it = J["items"]
    md = ["# 55／60／88 在 H200（只報告；13_h200_repro §15）", "",
          "對 E0 的差（百分點，95% CI；正確率或 LongBench 分數）。「主線」＝H200 執行過的節點；「對照」＝照 MI300X 的路（H200 沒執行，§10）。",
          f"E0：T1 Llama {100 * it['llama|e0acc|T1']['result']['acc']:.1f}%、Qwen {100 * it['qwen|e0acc|T1']['result']['acc']:.1f}%；"
          f"LongBench Llama {100 * it['llama|e0acc|LB']['result']['acc']:.1f}、Qwen {100 * it['qwen|e0acc|LB']['result']['acc']:.1f}。", "",
          "## T1（多鍵 NIAH）與 LongBench-6：端點", "",
          "| 模型 | 階 | 配置 | 標記 | T1 讀法 A | T1 B-idle | T1 B-busy | T1 B-i2b | LongBench 讀法 A |",
          "|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for m in ("llama", "qwen"):
        for tier in ("fp8", "int4"):
            for cfg in ("E1", "E1s", "E1sr"):
                t1 = it[f"{m}|{tier}|T1"]["result"][cfg]
                lb = it[f"{m}|{tier}|LB"]["result"][cfg]
                md.append(f"| {m} | {tier} | {cfg} | {it[f'{m}|{tier}|T1']['label']} | " + " | ".join(_pp(t1.get(c)) for c in A.CONDS)
                          + f" | {_pp(lb.get('A'))} |")
    md += ["", "## T1：混合配置（b32）", "",
           "| 模型・格式 | 標記 | 配置 | 對 E0 讀法 A | 對 E0 B-idle | 對 E0 B-busy | 對 R25 B-idle | 對 R25 B-busy |",
           "|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for key, vkey, lab in (("qwen|fp8mix|T1", "qwen|fp8mix_vsR25|T1", "Qwen fp8_head"), ("llama|int4mix|T1", "llama|int4mix_vsR25|T1", "Llama int4_kivi")):
        for cfg in ("F25", "M25", "B25", "R25"):
            e = it[key]["result"][cfg]
            vr = it[vkey]["result"].get(cfg, {})
            md.append(f"| {lab} | {it[key]['label']} | {cfg} | {_pp(e.get('A'))} | {_pp(e.get('B-idle'))} | {_pp(e.get('B-busy'))} | "
                      f"{_pp(vr.get('B-idle'))} | {_pp(vr.get('B-busy'))} |")
    md += ["", "## 逐 block 敏感度地圖（讀法 A；單一個 block 存成低精度、其他 BF16；平均 KL(答案‖E0)）", "",
           "| 模型・格式 | 標記 | 題數 | 列數 | 含答案的 block | 其他 block | KL 最大的 5 個 block |", "|:--|:--|--:|--:|:--|:--|:--|"]
    for key, lab in (("qwen|sensmap|fp8_head", "Qwen fp8_head"), ("llama|sensmap|int4_kivi", "Llama int4_kivi")):
        s = it[key]["result"]
        top = sorted(s["per_block_mean_kl"].items(), key=lambda kv: -kv[1])[:5]
        f = lambda t: f"{t[0]:.4g} [{t[1]:.4g}, {t[2]:.4g}]"  # noqa: E731
        md.append(f"| {lab} | {it[key]['label']} | {s['n_items']} | {s['n_rows']} | {f(s['answer_block_kl'])} | {f(s['other_block_kl'])} | "
                  + "、".join(f"{b}（{v:.3g}）" for b, v in top) + " |")
    md += ["", "跑了、沒分析（封存規則）：" + "、".join(J["not_analysed"]), ""]
    open(os.path.join(OUT, "tables_report_only.md"), "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
