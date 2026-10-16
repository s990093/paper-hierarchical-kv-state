"""m10_iowall.py — 12 號 A：IO 牆、容量與溢出模型（docs/precision_20261013/12_tier_plan.md §2；探索，只報告；不用 GPU）

A1 容量表〔算術〕：每段對話的位元組（32K／64K／128K × BF16／FP8／INT4 KIVI），各主機記憶體放得下幾段。
   每個 block 的位元組取自主線速度格子 restores.csv 的 bytes_stored（E0、E1 的 load_only 列 ÷ 64），不是從欄位名推。
A2 牆的曲線：還原時間對頻寬。
   - 實測點：主線 FP8／INT4 速度格子的 cake 中位數（b16／b32／b48 的頻寬＝bw.json；CPU 層畫在實測 pinned H2D 頻寬）；
     12 §3 的 stage 44 跑完後，加上 6.98、0.331、1、3、12、20 GiB/s 的 S-bf16／S-int4（--tier2-dir）。
   - 推算線（只在 ≤ 7 GiB/s，12 §2.2）：m10_sweep_sim.simulate；每個 block 在頻寬 B 的載入時間
     l_i(B)＝max(位元組 ÷ B, h_i)＋d_i，h_i＝CPU 層只載的時間扣掉反量化、d_i＝反量化（restores.csv 的 t_dequant ÷ 低精度 block 數）。
   - 驗證：在 b16／b32／b48 推算 E0、E1 的 cake，和實測比；過門檻（會合點 ±1 內 ≥ 90%、誤差中位數 ≤ 5%）才畫推算線、算 k(B)。
   - k(B)：E1 在 B 的時間＝E0 在 k·B 的時間（兩邊都用推算；k·B 超過 7 GiB/s 就寫「範圍外」）。
   - 損益兩平點 B×：只用實測點，相鄰兩個實測頻寬之間 E0−E1 變號時，在 log 頻寬上線性內插。
A3 溢出模型：主機記憶體 M、N 段 32K 對話；CPU 放得下 C 段，其餘在慢層（SSD 級 6.98／NFS 級 0.331 GiB/s）。
   平均還原時間＝p·T_CPU＋(1−p)·T_慢層，p＝min(1, C/N)。三種存法：全 BF16、全 INT4、分層（CPU BF16、慢層 INT4）。
   T_CPU 用實測；T_慢層 先用推算（標〔推算〕），stage 44 有實測就換成實測。
   牆＝平均還原時間 > 2 × T_CPU(E0) 的最小 N。
用法：python code/m10_iowall.py [--tier2-dir results/m10_tier_mi300x] [--out-dir results/m10_tier_mi300x]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.dirname(HERE)
R = os.environ.get("M10_PRECISION_DIR") or os.path.join(REPO, "results", "m10_precision_mi300x")   # H200：M10_PRECISION_DIR（沒設時不變；2026-10-17）
RUNS = os.environ.get("M10_RUNS_DIR") or "/mlsteam/data/tiara/runs"   # H200：$TIARA_RUNS（同上）
GiB = 1 << 30
GRIDS = {   # 主線速度格子（08 附錄 A）；Qwen FP8 用重量那次（判定用的）
    ("llama", "fp8"): ("20261015-161226-m10-grid-fp8-llama", "speed_llama/summary_fp8.csv", "speed_llama/restores.csv"),
    ("llama", "int4"): ("20261016-034902-m10-grid-int4-llama", "speed_llama/summary_int4.csv", "speed_llama/restores.csv"),
    ("qwen", "fp8"): ("20261015-194150-m10-regrid-fp8-qwen", "speed_qwen_regrid/summary_fp8.csv",
                      "speed_qwen_regrid/restores.csv.gz"),
    ("qwen", "int4"): ("20261016-052143-m10-grid-int4-qwen", "speed_qwen/summary_int4.csv", "speed_qwen/restores.csv"),
}
LENGTHS = {"32K": 64, "64K": 128, "128K": 252}          # 10 號計畫的 block 數
MEMS = [256, 512, 1024, 2048]                           # GiB
SLOW = {"ssd": None, "nfs": None}                        # 從 tier_params.json 讀（m10_tier2.slow_tiers）
SIM_MAX = 7.0                                            # 推算只用在 ≤ 7 GiB/s（12 §2.2）
GATE = {"b_within1_frac": 0.90, "median_abs_rel_err": 0.05}


def _open(p):
    if not os.path.exists(p) and os.path.exists(p + ".gz"):
        p = p + ".gz"
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p)


def load_grid(model, tier):
    rid, summ, rest = GRIDS[(model, tier)]
    fmt = json.load(open(os.path.join(R, "fmt_select", "fmt_choice.json")))["choice"][model][tier]
    S = [r for r in csv.DictReader(_open(os.path.join(R, summ))) if r["length"] == "32768"]
    rows = [r for r in csv.DictReader(_open(os.path.join(R, rest))) if r["run_id"] == rid and r["length"] == "32768"]
    import m10_sweep_sim as sim
    parts = sim.load_parts(os.path.join(RUNS, rid, "blocks.csv"))
    return dict(run_id=rid, fmt=fmt, summary=S, rows=rows, parts=parts)


def med(xs):
    return statistics.median(xs) if xs else None


def block_bytes(g, n=64):
    """每個 block 的位元組：load_only 列的 bytes_stored ÷ n（E0＝BF16、E1＝低精度）。"""
    out = {}
    for cfg, key in (("E0", "bf16"), ("E1", "lp")):
        v = {int(r["bytes_stored"]) for r in g["rows"] if r["config"] == cfg and r["mode"] == "load_only"}
        assert len(v) == 1, (cfg, v)
        out[key] = v.pop() / n
    return out


def dequant_per_block(g, state):
    """CPU 層 E1 只載：t_dequant ÷ 低精度 block 數（中位數）。"""
    v = [float(r["t_dequant"]) / int(r["fp8_read"]) for r in g["rows"]
         if r["config"] == "E1" and r["mode"] == "load_only" and r["bw_name"] == "cpu" and r["gpu_state"] == state
         and int(r["fp8_read"]) > 0]
    return med(v)


def load_model(g, state, lp: bool, B_GiBps: float, bb):
    """每個 block 在頻寬 B 的載入時間（12 §2.2）。"""
    key = ("cpu", state, g["fmt"] if lp else "bf16")
    lcpu = g["parts"]["l"][key]
    d = dequant_per_block(g, state) if lp else 0.0
    nbytes = bb["lp" if lp else "bf16"]
    return [max(nbytes / (B_GiBps * GiB), li - d) + d for li in lcpu]


def sim_cake(g, state, lp: bool, B, bb):
    import m10_sweep_sim as sim
    f = g["parts"]["f"][state]
    b, t, _, _ = sim.simulate(f, load_model(g, state, lp, B, bb))
    return b, t


def measured(g, bw, state, cfg):
    for r in g["summary"]:
        if r["bw_name"] == bw and r["gpu_state"] == state and r["config"] == cfg and r["mode"] == "cake":
            return float(r["median_t_restore"]), [int(x) for x in r["b_values"].split()]
    return None, None


def validate(g, bwj, bb):
    cells, errs, inb = [], [], []
    for bw in ("b16", "b32", "b48"):
        B = float(bwj["targets"][bw]["GiBps"])
        for st in ("idle", "busy"):
            for cfg, lp in (("E0", False), ("E1", True)):
                mt, mb = measured(g, bw, st, cfg)
                if mt is None:
                    continue
                b, t = sim_cake(g, st, lp, B, bb)
                e = (t - mt) / mt
                ok = min(mb) - 1 <= b <= max(mb) + 1
                errs.append(abs(e))
                inb.append(ok)
                cells.append(dict(bw=bw, GiBps=B, gpu_state=st, config=cfg, b_pred=b, b_meas=mb, t_pred=round(t, 4),
                                  t_meas=mt, rel_err=round(e, 4), b_ok=ok))
    res = dict(n_cells=len(cells), b_within1_frac=round(sum(inb) / len(inb), 4),
               median_abs_rel_err=round(statistics.median(errs), 4), max_abs_rel_err=round(max(errs), 4), cells=cells)
    res["pass"] = res["b_within1_frac"] >= GATE["b_within1_frac"] and res["median_abs_rel_err"] <= GATE["median_abs_rel_err"]
    return res


def tier2_points(tier2_dir, model):
    """stage 44 的實測：{(tier, state, 'E0'|'E1'): (GiB/s, 中位數)}；E0＝S-bf16、E1＝S-int4。沒有就回 {}。"""
    p = os.path.join(tier2_dir or "", f"summary_{model}.csv")
    if not tier2_dir or not os.path.exists(p):
        return {}
    out = {}
    for r in csv.DictReader(open(p)):
        if r["config"] in ("S-bf16", "S-int4") and r["tier"] != "cpu":
            out[(r["tier"], r["gpu_state"], "E0" if r["config"] == "S-bf16" else "E1")] = (
                float(r["slow_GiBps"]), float(r["median_t_restore"]))
    return out


def breakeven(points):
    """points：[(GiB/s, t_E0, t_E1)]，只用實測。回傳 E0−E1 由正（低精度快）轉負的頻寬（log 內插）；沒有變號回 None。"""
    pts = sorted(points)
    for (b1, e01, e11), (b2, e02, e12) in zip(pts, pts[1:]):
        d1, d2 = e01 - e11, e02 - e12
        if d1 > 0 >= d2:
            x = math.log(b1) + (math.log(b2) - math.log(b1)) * d1 / (d1 - d2)
            return dict(GiBps=round(math.exp(x), 3), between=[b1, b2])
    return None


def k_equiv(g, st, B, bb):
    """E1 在 B 的時間＝E0 在 k·B 的時間（推算；k·B 要 ≤ SIM_MAX）。"""
    t1 = sim_cake(g, st, True, B, bb)[1]
    if sim_cake(g, st, False, SIM_MAX, bb)[1] > t1:
        return None, round(t1, 4)
    lo, hi = B, SIM_MAX
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if sim_cake(g, st, False, mid, bb)[1] > t1:
            lo = mid
        else:
            hi = mid
    return round(hi / B, 3), round(t1, 4)


def wall(T_c0, T_c, T_s, C):
    """平均還原時間 > 2·T_c0 的最小 N（12 §2.3）。"""
    thr = 2 * T_c0
    if T_c > thr:
        return 0.0
    if T_s <= thr:
        return math.inf
    one_minus_p = (thr - T_c) / (T_s - T_c)
    return C / (1 - one_minus_p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier2-dir", default=os.path.join(REPO, "results", "m10_tier_mi300x"))
    ap.add_argument("--out-dir", default=os.path.join(REPO, "results", "m10_tier_mi300x"))
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()
    import m10_tier2
    slow = m10_tier2.slow_tiers()
    os.makedirs(a.out_dir, exist_ok=True)
    md = ["# IO 牆、容量與溢出模型（12 §2；探索，只報告）", ""]
    allres = {}
    for model in ("llama", "qwen"):
        bwj = json.load(open(os.path.join(R, f"speed_{model}", "bw.json")))
        G = {t: load_grid(model, t) for t in ("fp8", "int4")}
        bb = {t: block_bytes(G[t]) for t in G}
        assert abs(bb["fp8"]["bf16"] - bwj["chunk_bytes"]) < 1 and abs(bb["int4"]["bf16"] - bwj["chunk_bytes"]) < 1
        h2d = float(bwj["real_pinned_h2d_GiBps"])
        res = dict(model=model, runs={t: G[t]["run_id"] for t in G}, fmt={t: G[t]["fmt"] for t in G},
                   bytes_per_block={"bf16": bb["int4"]["bf16"], "fp8": bb["fp8"]["lp"], "int4": bb["int4"]["lp"]},
                   real_pinned_h2d_GiBps=h2d, slow_GiBps=slow)
        # ---- A1
        cap = {}
        for L, nb in LENGTHS.items():
            for f in ("bf16", "fp8", "int4"):
                size = nb * res["bytes_per_block"][f]
                cap[f"{L}|{f}"] = dict(GiB=round(size / GiB, 3), **{f"n_in_{m}GiB": int(m * GiB // size) for m in MEMS})
        res["A1"] = cap
        # ---- A2
        res["A2"] = {}
        t2 = tier2_points(a.tier2_dir, model)
        for t in ("fp8", "int4"):
            g = G[t]
            v = validate(g, bwj, bb[t])
            ent = dict(validation=v, curve={}, measured={}, k={}, breakeven={})
            for st in ("idle", "busy"):
                meas = []
                for bw in ("b48", "b32", "b16", "cpu"):
                    B = h2d if bw == "cpu" else float(bwj["targets"][bw]["GiBps"])
                    e0, _ = measured(g, bw, st, "E0")
                    e1, _ = measured(g, bw, st, "E1")
                    meas.append((B, e0, e1, bw))
                if t == "int4":                         # stage 44 的實測點（只有 INT4 KIVI）
                    for tn in ("nfs", "w1", "w3", "ssd", "w12", "w20"):
                        if (tn, st, "E0") in t2 and (tn, st, "E1") in t2:
                            meas.append((t2[(tn, st, "E0")][0], t2[(tn, st, "E0")][1], t2[(tn, st, "E1")][1], tn))
                ent["measured"][st] = [dict(GiBps=round(B, 4), tier=n, E0=e0, E1=e1) for B, e0, e1, n in sorted(meas)]
                ent["breakeven"][st] = breakeven([(B, e0, e1) for B, e0, e1, _ in meas if e0 and e1])
                if v["pass"]:
                    xs = [0.2 * (SIM_MAX / 0.2) ** (i / 40) for i in range(41)]
                    ent["curve"][st] = dict(GiBps=[round(x, 4) for x in xs],
                                            E0=[round(sim_cake(g, st, False, x, bb[t])[1], 4) for x in xs],
                                            E1=[round(sim_cake(g, st, True, x, bb[t])[1], 4) for x in xs])
                    ent["k"][st] = {str(B): dict(zip(("k", "t_E1"), k_equiv(g, st, B, bb[t])))
                                    for B in (slow["nfs"], 1.0, 3.0, slow["ssd"])}
            res["A2"][t] = ent
        # ---- A3（32K；INT4 KIVI；T_慢層 先推算，有 stage 44 的實測就換）
        g = G["int4"]
        res["A3"] = {}
        for st in ("idle", "busy"):
            Tc0, _ = measured(g, "cpu", st, "E0")
            Tc1, _ = measured(g, "cpu", st, "E1")
            for tn in ("ssd", "nfs"):
                src = {}
                Ts = {}
                for cfg, lp in (("E0", False), ("E1", True)):
                    if (tn, st, cfg) in t2:
                        Ts[cfg], src[cfg] = t2[(tn, st, cfg)][1], "實測（stage 44）"
                    elif slow[tn] <= SIM_MAX and res["A2"]["int4"]["validation"]["pass"]:
                        Ts[cfg], src[cfg] = sim_cake(g, st, lp, slow[tn], bb["int4"])[1], "〔推算〕"
                    else:
                        Ts[cfg], src[cfg] = None, "NOT_MEASURED"
                pol = {}
                for name, cpu_fmt, Tc, Tsx in (("全 BF16", "bf16", Tc0, Ts["E0"]), ("全 INT4", "int4", Tc1, Ts["E1"]),
                                               ("分層", "bf16", Tc0, Ts["E1"])):
                    size = 64 * res["bytes_per_block"][cpu_fmt]
                    pol[name] = dict(T_cpu=Tc, T_slow=Tsx, C_per_TiB=1024 * GiB / size,
                                     N_wall_per_TiB=None if Tsx is None else wall(Tc0, Tc, Tsx, 1024 * GiB / size))
                res["A3"][f"{tn}|{st}"] = dict(T_cpu_E0=Tc0, T_cpu_E1=Tc1, T_slow=Ts, T_slow_source=src, policies=pol)
        allres[model] = res
        md += report_md(res)
    json.dump(allres, open(os.path.join(a.out_dir, "iowall.json"), "w"), indent=1, ensure_ascii=False,
              default=lambda x: None if isinstance(x, float) and math.isinf(x) else x)
    open(os.path.join(a.out_dir, "tables_iowall.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    if not a.no_fig:
        figs(allres, os.path.join(a.out_dir, "figs"))


def _fmt_n(x):
    if x is None:
        return "NOT_MEASURED"
    if math.isinf(x):
        return "沒有牆"
    return f"{x:,.0f}"


def report_md(res):
    m = res["model"]
    out = [f"## {m}", "", f"run：{'、'.join(res['runs'].values())}；每個 block：BF16 {res['bytes_per_block']['bf16'] / 2**20:.2f} MiB、"
           f"FP8（{res['fmt']['fp8']}）{res['bytes_per_block']['fp8'] / 2**20:.2f} MiB、INT4 KIVI {res['bytes_per_block']['int4'] / 2**20:.2f} MiB", "",
           "### A1 容量〔算術〕：每段對話多大、主機記憶體放得下幾段", "",
           "| 長度 | 格式 | 每段 (GiB) | 256 GiB | 512 GiB | 1 TiB | 2 TiB |", "|:--|:--|--:|--:|--:|--:|--:|"]
    for k, v in res["A1"].items():
        L, f = k.split("|")
        out.append(f"| {L} | {f} | {v['GiB']:.2f} | {v['n_in_256GiB']} | {v['n_in_512GiB']} | {v['n_in_1024GiB']} | {v['n_in_2048GiB']} |")
    out += ["", "### A2 還原時間對頻寬（32K）", ""]
    for t, e in res["A2"].items():
        v = e["validation"]
        out.append(f"- **{t}（{res['fmt'][t]}）推算驗證**：{v['n_cells']} 格，會合點 ±1 內 {100 * v['b_within1_frac']:.0f}%、"
                   f"誤差中位數 {100 * v['median_abs_rel_err']:.1f}%、最大 {100 * v['max_abs_rel_err']:.1f}% → "
                   f"{'**過**（畫推算線）' if v['pass'] else '**沒過**（只畫實測點）'}")
        for st in ("idle", "busy"):
            be = e["breakeven"].get(st)
            pts = "；".join(f"{p['tier']} {p['GiBps']:g}: E0 {p['E0']:.3f}／E1 {p['E1']:.3f}" for p in e["measured"][st] if p["E0"])
            out.append(f"  - GPU {'閒' if st == 'idle' else '忙'}：實測（s）{pts}")
            out.append(f"    - 損益兩平點（只用實測）：{'%.2f GiB/s（在 %g 和 %g 之間）' % (be['GiBps'], *be['between']) if be else '量到的範圍內沒有變號'}")
            if st in e["k"]:
                ks = "；".join(f"{float(B):g} GiB/s → {('k＝%.2f' % x['k']) if x['k'] else '範圍外（> 7 GiB/s）'}" for B, x in e["k"][st].items())
                out.append(f"    - 等效頻寬倍數〔推算〕：{ks}")
    out += ["", "### A3 溢出模型（32K、INT4 KIVI；每 1 TiB 主機記憶體；牆＝平均還原時間 > 2 × 全部在 CPU 的 E0）", "",
            "| 慢層 | GPU | T_CPU E0／E1 (s) | T_慢層 BF16／INT4 (s) | T_慢層 ÷ T_CPU(E0)：BF16／INT4 | 來源 | 全 BF16：放得下／牆 | 全 INT4：放得下／牆 | 分層：放得下／牆 |",
            "|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for k, v in res["A3"].items():
        tn, st = k.split("|")
        ts = v["T_slow"]
        p = v["policies"]
        out.append(f"| {tn}（{res['slow_GiBps'][tn]:.3g} GiB/s） | {'閒' if st == 'idle' else '忙'} | {v['T_cpu_E0']:.3f}／{v['T_cpu_E1']:.3f} | "
                   f"{ts['E0'] if ts['E0'] is None else round(ts['E0'], 3)}／{ts['E1'] if ts['E1'] is None else round(ts['E1'], 3)} | "
                   f"{'—' if ts['E0'] is None else round(ts['E0'] / v['T_cpu_E0'], 2)}／{'—' if ts['E1'] is None else round(ts['E1'] / v['T_cpu_E0'], 2)} | "
                   f"{v['T_slow_source']['E0']}／{v['T_slow_source']['E1']} | "
                   + " | ".join(f"{p[n]['C_per_TiB']:.0f}／{_fmt_n(p[n]['N_wall_per_TiB'])}" for n in ("全 BF16", "全 INT4", "分層")) + " |")
    out.append("")
    return out


def figs(allres, out_dir):
    import m10_figs as F
    plt = F.plt
    col = {"E0": F.INK2, "fp8": F.SLOTS[0], "int4": F.SLOTS[1]}
    for model, res in allres.items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), squeeze=False)
        for j, st in enumerate(("idle", "busy")):
            ax = axes[0][j]
            for t in ("fp8", "int4"):
                e = res["A2"][t]
                if st in e["curve"]:
                    c = e["curve"][st]
                    if t == "int4":
                        ax.plot(c["GiBps"], c["E0"], "--", color=col["E0"], lw=1.5, label="全 BF16（推算）")
                    ax.plot(c["GiBps"], c["E1"], "--", color=col[t], lw=1.5, label=f"全 {t.upper()}（推算）")
                pts = [p for p in e["measured"][st] if p["E0"]]
                if t == "int4":
                    ax.plot([p["GiBps"] for p in pts], [p["E0"] for p in pts], "o", color=col["E0"], ms=6, label="全 BF16（實測）")
                ax.plot([p["GiBps"] for p in pts], [p["E1"] for p in pts], "o", color=col[t], ms=6, label=f"全 {t.upper()}（實測）")
            for name, B in (("NFS 級", res["slow_GiBps"]["nfs"]), ("SSD 級", res["slow_GiBps"]["ssd"]),
                            ("CPU（真實 H2D）", res["real_pinned_h2d_GiBps"])):
                ax.axvline(B, color=F.BASE, lw=1)
                ax.text(B, 0.97, name, rotation=90, va="top", ha="right", fontsize=7,
                        color=F.MUTED, transform=ax.get_xaxis_transform())
            ax.set_xscale("log")
            ax.set_yscale("log")
            from matplotlib.ticker import FuncFormatter
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                axis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
            ax.set_xlabel("慢層頻寬（GiB/s，log）")
            ax.set_ylabel("還原時間中位數（s，log）")
            ax.set_title(f"GPU {'閒' if st == 'idle' else '忙'}")
            if j == 0:
                ax.legend(fontsize=7)
        fig.suptitle(f"圖 A2　{model}：還原時間對頻寬（32K；實心＝實測，虛線＝推算，只畫 ≤ 7 GiB/s）", color=F.INK, fontsize=10)
        fig.subplots_adjust(bottom=0.18, top=0.88, wspace=0.25)
        F.source_note(fig, "12_tier_plan §2；主線速度格子 " + "、".join(res["runs"].values()))
        F.save(fig, os.path.join(out_dir, f"A2_{model}_wall.png"))
        # A3：平均還原時間對 N（每 1 TiB）
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.0), squeeze=False)
        pc = {"全 BF16": F.INK2, "全 INT4": F.SLOTS[1], "分層": F.SLOTS[2]}
        for i, st in enumerate(("idle", "busy")):
            for j, tn in enumerate(("ssd", "nfs")):
                ax = axes[i][j]
                v = res["A3"][f"{tn}|{st}"]
                Cmax = max(p["C_per_TiB"] for p in v["policies"].values())
                xs = [Cmax * 4 * k / 200 for k in range(1, 201)]
                for name, p in v["policies"].items():
                    if p["T_slow"] is None:
                        continue
                    ys = [min(1, p["C_per_TiB"] / x) * p["T_cpu"] + (1 - min(1, p["C_per_TiB"] / x)) * p["T_slow"] for x in xs]
                    ax.plot(xs, ys, color=pc[name], lw=2, label=name)
                ax.axhline(2 * v["T_cpu_E0"], color=F.SLOTS[7], lw=1, ls=":", label="牆（2 × 全部在 CPU 的 E0）")
                ax.set_title(f"慢層 {tn}（{res['slow_GiBps'][tn]:.3g} GiB/s）・GPU {'閒' if st == 'idle' else '忙'}"
                             f"（慢層：{v['T_slow_source']['E0']}）")
                ax.set_xlabel("32K 對話數（每 1 TiB 主機記憶體）")
                ax.set_ylabel("平均還原時間（s）")
                if i == 0 and j == 0:
                    ax.legend(fontsize=7)
        fig.suptitle(f"圖 A3　{model}：對話變多、CPU 放不下時的平均還原時間（均勻回來；12 §2.4 的假設）", color=F.INK, fontsize=10)
        fig.subplots_adjust(hspace=0.5, wspace=0.25, bottom=0.1, top=0.9)
        F.source_note(fig, "12_tier_plan §2.3；T_CPU 實測、T_慢層 見標題")
        F.save(fig, os.path.join(out_dir, f"A3_{model}_spill.png"))


if __name__ == "__main__":
    main()
