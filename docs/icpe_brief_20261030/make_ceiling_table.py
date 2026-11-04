#!/usr/bin/env python3
"""Data table for the planned 'ceiling vs CPU capacity' chart (no chart yet): FIG1_DATA.md + figs/ceiling_by_capacity.csv.

    bash -c 'source code/env_h200.sh >/dev/null 2>&1; python docs/icpe_brief_20261030/make_ceiling_table.py'

Rule (same as the paper's Fig. 9): oracle headroom over the best same-action rival, Cake restore, V0 constants; each cell = max over
GPU idle/busy. Mooncake: results/icpe_h200/t1/p01/sim_V0/x1/x1_headroom_components.csv, session rule = min (the smaller of the two
reconstructions, as in the paper), CPU 128-1024 GiB only. FreeInference: figs/fig9_fi_capacity.sources.json (16-64 GiB from T5, 128-1024 from T1).
Nothing is measured here; every value is read from those files and the anchors are asserted against EVIDENCE.md.
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PAPER = HERE.parent / "icpe_paper_20261030"
EVID = (PAPER / "EVIDENCE.md").read_text(encoding="utf-8").splitlines()
FI_SIDECAR = json.load(open(PAPER / "figs" / "fig9_fi_capacity.sources.json"))["plotted_values"]


def chk(prefix, *literals):
    """The EVIDENCE line that starts with `prefix` must contain every literal."""
    hits = [l for l in EVID if l.startswith(prefix)]
    assert hits, f"no EVIDENCE line starts with {prefix!r}"
    for lit in literals:
        assert any(lit in l for l in hits), f"{lit!r} not on EVIDENCE line {prefix!r}"

SRC = REPO / "results/icpe_h200/t1/p01/sim_V0/x1/x1_headroom_components.csv"
CAPS = ["16", "32", "64", "128", "256", "512", "1024"]
TIERS = [("b32", "b32"), ("ssd", "SSD 級"), ("nfs", "NFS 級")]
COMP = [("O-PF", "預取（回來前先搬好）"), ("O-ORD", "淘汰順序（CPU 滿了先擠掉誰）")]
TRACES = [("freeinference", "FreeInference"), ("conversation", "Mooncake conversation"), ("toolagent", "Mooncake toolagent")]


def load():
    rows = [r for r in csv.DictReader(open(SRC, encoding="utf-8")) if r["rule"] == "min" and r["restore"] == "cake"]
    assert len(rows) == 192
    val, sat = {}, {}
    for tr in ("conversation", "toolagent"):
        for comp, _ in COMP:
            for tier, _ in TIERS:
                for cap in CAPS[3:]:
                    cs = [r for r in rows if r["trace"] == tr and r["component"] == comp and r["slow"] == tier and r["cpu_GiB"] == cap]
                    assert len(cs) == 2, (tr, comp, tier, cap)
                    val[(tr, comp, tier, cap)] = max(float(r["headroom"]) for r in cs) * 100
                    sat[(tr, comp, tier, cap)] = any(r["write_saturated"] == "True" for r in cs)
    for comp, _ in COMP:
        for tier, _ in TIERS:
            for cap in CAPS:
                val[("freeinference", comp, tier, cap)] = FI_SIDECAR[comp][tier][cap] * 100
                sat[("freeinference", comp, tier, cap)] = False
    return val, sat


def by_gpu():
    """Mooncake, SSD class, Cake, rule=min: headroom % split by GPU idle/busy (the figure itself shows the larger of the two)."""
    rows = [r for r in csv.DictReader(open(SRC, encoding="utf-8")) if r["rule"] == "min" and r["restore"] == "cake" and r["slow"] == "ssd"]
    out = {}
    for r in rows:
        if r["component"] in ("O-PF", "O-ORD"):
            out[(r["trace"], r["component"], r["cpu_GiB"], r["gpu"])] = float(r["headroom"]) * 100
    return out


def cake_vs_load():
    """Mooncake, SSD class, rule=min, larger of GPU idle/busy: ceiling with Cake vs with full load (no Cake)."""
    rows = [r for r in csv.DictReader(open(SRC, encoding="utf-8")) if r["rule"] == "min" and r["slow"] == "ssd"]
    out = {}
    for tr in ("conversation", "toolagent"):
        for comp in ("O-PF", "O-ORD"):
            for cap in CAPS[3:]:
                v = {}
                for rs in ("cake", "load"):
                    cs = [r for r in rows if r["trace"] == tr and r["component"] == comp and r["cpu_GiB"] == cap and r["restore"] == rs]
                    assert len(cs) == 2, (tr, comp, cap, rs)
                    assert not any(r["write_saturated"] == "True" for r in cs), "SSD-class cell saturated"
                    v[rs] = max(float(r["headroom"]) for r in cs) * 100
                out[(tr, comp, cap)] = v
    return out


def fmt(v, s):
    if v is None:
        return "—"
    return (f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}") + ("S" if s else "")


def main():
    val, sat = load()
    # anchors against the paper's own rows
    chk("| B1 |", "50.5", "46.5", "27.6", "31.0")
    chk("| B3-T5 |", "4.89", "11.81", "5.92")
    chk("| B3-X1 |", "0.30%")
    chk("| B8 |", "55.5", "41.6")
    assert round(val[("conversation", "O-PF", "ssd", "128")], 1) == 50.5 and round(val[("conversation", "O-ORD", "ssd", "256")], 1) == 31.0
    assert round(val[("toolagent", "O-ORD", "nfs", "256")], 1) == 55.5 and round(val[("toolagent", "O-PF", "nfs", "512")], 1) == 41.6
    assert round(val[("freeinference", "O-PF", "b32", "32")], 2) == 4.89 and round(val[("freeinference", "O-ORD", "nfs", "32")], 2) == 11.81
    assert round(max(val[("freeinference", c, t, k)] for c in ("O-PF", "O-ORD") for t, _ in TIERS for k in CAPS[3:]), 2) == 0.30

    out = ["# 圖 2 的資料表（預測上限；圖是 figs/brief_d_prediction_ceiling.png，只畫 SSD 級那一列；圖 1 是寫入時套用的對照）", "",
           "**每個格子**：完美預知（oracle）比「同一個動作下最好的不預測做法」讓回來請求的 TTFT 縮短的 %，Cake 還原，模擬；每格取 GPU 閒／忙的較大者（和論文 Fig. 9 同一規則）。",
           "**S**＝這格寫入飽和（慢層寫不及），論文主結果不計入。**—**＝沒跑（Mooncake 只跑 CPU 128–1,024 GiB；FreeInference 16–64 GiB 來自 T5、128–1,024 來自 T1）。",
           "Mooncake 的對話是用兩種規則重建的，這裡取較小的（`rule=min`，同論文）。",
           "來源：Mooncake `results/icpe_h200/t1/p01/sim_V0/x1/x1_headroom_components.csv`；FreeInference `docs/icpe_paper_20261030/figs/fig9_fi_capacity.sources.json`。", ""]
    csv_rows = [["trace", "component", "slow", "cpu_GiB", "headroom_pct", "write_saturated"]]
    for comp, cname in COMP:
        out += [f"## {cname}（{comp}）", "", "| 紀錄 | 慢層 | " + " | ".join(f"{c}" for c in CAPS) + " |", "|:--|:--|" + "--:|" * len(CAPS)]
        for tr, trn in TRACES:
            for tier, tn in TIERS:
                cells = []
                for cap in CAPS:
                    k = (tr, comp, tier, cap)
                    cells.append(fmt(val.get(k), sat.get(k, False)))
                    if k in val:
                        csv_rows.append([tr, comp, tier, cap, f"{val[k]:.4f}", sat[k]])
                out.append(f"| {trn} | {tn} | " + " | ".join(cells) + " |")
        out.append("")
    out += ["（欄＝CPU 層容量 GiB）", "", "## GPU 閒與忙（Mooncake、SSD 級；圖 2 取兩者較大的）", "",
            "| 紀錄 | 上限 | GPU | " + " | ".join(CAPS[3:]) + " |", "|:--|:--|:--|" + "--:|" * 4]
    g = by_gpu()
    for tr, trn in TRACES[1:]:
        for comp, cname in COMP:
            for gpu, gn in (("idle", "閒"), ("busy", "忙")):
                out.append(f"| {trn} | {cname.split('（')[0]} | {gn} | " + " | ".join(f"{g[(tr, comp, c, gpu)]:.1f}" for c in CAPS[3:]) + " |")
    out.append("")
    assert round(g[("conversation", "O-PF", "128", "busy")], 1) == 38.4 and round(g[("toolagent", "O-PF", "128", "busy")], 1) == 20.8
    cl = cake_vs_load()
    assert round(cl[("conversation", "O-PF", "128")]["cake"], 1) == 50.5 and round(cl[("conversation", "O-PF", "128")]["load"], 1) == 59.2
    gaps = [v["load"] - v["cake"] for v in cl.values()]
    assert round(min(gaps), 1) == 1.9 and round(max(gaps), 1) == 13.8 and min(gaps) > 0
    out += ["## 拿掉 Cake 會怎樣（Mooncake、SSD 級、取 GPU 閒／忙較大者）", "",
            "**Cake**＝邊重算前段、邊載入後段；**整段載入**＝沒有 Cake，整段從慢層讀回。其他條件相同（`restore=cake` 對 `restore=load`），模擬。",
            f"整段載入的上限比 Cake 高 {min(gaps):.1f}–{max(gaps):.1f} 個百分點，排序與「隨 CPU 容量下降」的趨勢不變。NFS 級的 Mooncake 格全是寫入飽和，不列。", "",
            "| 紀錄 | 上限 | 還原 | " + " | ".join(CAPS[3:]) + " |", "|:--|:--|:--|" + "--:|" * 4]
    for tr, trn in TRACES[1:]:
        for comp, cname in COMP:
            for rs, rn in (("cake", "Cake"), ("load", "整段載入")):
                out.append(f"| {trn} | {cname.split('（')[0]} | {rn} | " + " | ".join(f"{cl[(tr, comp, c)][rs]:.1f}" for c in CAPS[3:]) + " |")
    out.append("")
    (HERE / "FIG1_DATA.md").write_text("\n".join(out), encoding="utf-8")
    with open(HERE / "figs" / "ceiling_by_capacity.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(csv_rows)
    print("\n".join(out))


if __name__ == "__main__":
    main()
