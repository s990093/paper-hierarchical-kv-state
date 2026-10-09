"""m7_report_fill.py — 把 summary_*.csv / verdict.json 的數字填進 07_report.md 的 <<...>> 佔位（不手打數字）"""
import json, os, re, sys
import pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); R = f"{H}/../results/m7_write_policy_mi300x"
D = f"{H}/../docs/phase1_20261008/07_report.md"
TPL = f"{H}/../docs/phase1_20261008/07_report.template.md"
V = json.load(open(f"{R}/verdict.json"))
s = open(TPL).read()

def ms(x): return f"{x*1e3:,.0f}"

# C4 rows
c4 = pd.read_csv(f"{R}/summary_c4.csv")
rows = "\n".join(f"| {r.slot} | {'同一個 process 連跑' if r.proc_mode=='same' else '每次重開 process'} | {r.n} | {ms(r.median_s)} ms | {ms(r.ci_lo)}–{ms(r.ci_hi)} ms | {r.delta_pct:.1f}% |" for r in c4.itertuples())
s = s.replace("<<C4_ROWS>>", rows).replace("<<DELTA>>", f"{V['calib']['C4']['delta_pct']:.1f}")
# C7 fixed
p = f"{R}/calib_c7_self_fixed.csv"
if os.path.exists(p):
    c = pd.read_csv(p).groupby("set_GiBps").err_pct.median()
    s = s.replace("<<C7_FIXED>>", "／".join(f"{v:+.1f}%" for v in c.values) + f"（設定 {'／'.join(f'{k:g}' for k in c.index)} GiB/s）→ " + ("**通過**" if c.abs().max() <= 5 else "**仍未通過**"))
else:
    s = s.replace("<<C7_FIXED>>", "NOT_MEASURED（排在執行鏈裡，尚未跑完）")
# B tables
t = pd.read_csv(f"{R}/summary_b.csv")
v = pd.read_csv(f"{R}/summary_b_verdict.csv")
main = t[(t.io_model == "share") & (t.cpu_tier == "harness") & (t.ssd_dev == "nfs") & (t.gap_s == 0) & (t.ssd_frac == 1) & (t.wl_seed == 0)]
order = ["R0", "S0", "S1", "S2b", "S3", "S4", "S4+", "S4+P", "S5", "S5s"]
lines = ["| 策略 | " + " | ".join(f"{int(c*100)}%：中位數〔95% CI〕／平均／P90（s）" for c in sorted(main.cpu_frac.unique())) + " |",
         "|:--|" + "--:|" * main.cpu_frac.nunique()]
for st in order:
    cells = []
    for c in sorted(main.cpu_frac.unique()):
        r = main[(main.strategy == st) & (main.cpu_frac == c)]
        cells.append("—" if not len(r) else f"{r['median'].iloc[0]:.2f}〔{r.ci_lo.iloc[0]:.2f}–{r.ci_hi.iloc[0]:.2f}〕／{r['mean'].iloc[0]:.2f}／{r.p90.iloc[0]:.2f}（{int(r.reps.iloc[0])} 次）")
    lines.append(f"| {st} | " + " | ".join(cells) + " |")
s = s.replace("<<B_MAIN_TABLE>>", "\n".join(lines))
def cfgname(r):
    parts = []
    parts.append({"nfs": "SSD 層＝NFS 參數", "local": "SSD 層＝本地 SSD 參數"}[r.ssd_dev])
    if r.io_model == "fifo": parts.append("干擾＝FIFO")
    if r.cpu_tier != "harness": parts.append("CPU 層＝vLLM 實測速度（3.69 GiB/s）")
    if r.gap_s: parts.append(f"間隔 {r.gap_s:g} s")
    if r.ssd_frac < 1: parts.append(f"SSD 容量 {r.ssd_frac:.0%}")
    parts.append(f"seed {r.wl_seed}")
    return "，".join(parts)
vl = ["| 設定 | CPU 容量 | 最好的簡單策略 | 它的中位數 | S5 中位數 | S5 相對最好簡單策略 | S5 vs S4+ | 成對平均 | 重複 | 判讀 |", "|:--|--:|:--|--:|--:|--:|--:|--:|--:|:--|"]
for r in v.sort_values(["wl_seed", "ssd_dev", "io_model", "cpu_tier", "gap_s", "ssd_frac", "cpu_frac"]).itertuples():
    vl.append(f"| {cfgname(r)} | {r.cpu_frac:.0%} | {r.best_simple} | {r.best_median:.3f} s | {r.s5_median:.3f} s | {r.gain_pct:+.1f}% | {r.s5_vs_s4plus_pct:+.1f}% | {r.paired_mean_gain_pct:+.1f}% | {r.n_reps} | {r.verdict}{'（執行中，部分資料）' if r.n_reps < 3 else ''} |")
s = s.replace("<<B_VERDICT_TABLE>>", "\n".join(vl))
# 敏感度摘要（自動）
s = s.replace("<<S5_S4P_RANGE>>", f"{v.s5_vs_s4plus_pct.min():+.1f}%～{v.s5_vs_s4plus_pct.max():+.1f}%")
s = s.replace("<<B_SENS_SUMMARY>>", f"在全部 {len(v)} 個（設定 × 容量）組合裡，S5 相對最好簡單策略的中位數差距是 {v.gain_pct.min():+.1f}%～{v.gain_pct.max():+.1f}%，"
              + ("**沒有一個 >5%**" if (v.gain_pct <= 5).all() else f"**有 {(v.gain_pct > 5).sum()} 個 >5%**")
              + f"；S5 和 S4+ 的差距在 {v.s5_vs_s4plus_pct.min():+.1f}%～{v.s5_vs_s4plus_pct.max():+.1f}%。")
q = v[(v.cpu_frac == 0.25) & (v.io_model == "share") & (v.n_reps >= 3)]
lru = q[q.best_simple.isin(["S1", "S4"])]
oth = q[~q.best_simple.isin(["S1", "S4"])]
txt = (f"25% 容量、share 干擾、跑完的 {len(q)} 個設定裡，有 {len(lru)} 個最好的簡單策略是 LRU 類（S1 或 S4），"
       f"S5（＝S4+）比它慢 {-lru.gain_pct.max():.0f}–{-lru.gain_pct.min():.0f}%（中位數）")
if len(oth):
    txt += "；其餘 " + "、".join(f"{cfgname(r)}（最好的是 {r.best_simple}，S5 差 {r.gain_pct:+.1f}%）" for r in oth.itertuples())
fq = v[(v.cpu_frac == 0.25) & (v.io_model == "fifo")]
if len(fq):
    txt += f"；FIFO 干擾下 S5 慢 {-fq.gain_pct.min():.0f}%（但平均幾乎相同，見 §8.4）"
s = s.replace("<<LRU_SUMMARY>>", txt + "。")
extra = []
for r in v[(v.wl_seed == 2) | (v.gap_s > 0) | (v.ssd_frac < 1)].sort_values(["wl_seed", "gap_s", "ssd_frac", "cpu_frac"]).itertuples():
    extra.append(f"| {cfgname(r)}（{r.cpu_frac:.0%}） | 補充設定 | S5 {r.s5_median:.3f} s vs 最好簡單 {r.best_simple} {r.best_median:.3f} s（{r.gain_pct:+.1f}%）；S5 vs S4+ {r.s5_vs_s4plus_pct:+.1f}%{'（執行中，部分資料，' + str(r.n_reps) + ' 次）' if r.n_reps < 3 else ''} |")
pending = [n for n, ok in (("seed 2", (v.wl_seed == 2).any()), ("間隔 2 s", (v.gap_s > 0).any()), ("SSD 容量 50%", (v.ssd_frac < 1).any())) if not ok]
if pending:
    extra.append(f"| {'、'.join(pending)} | 補充設定 | NOT_MEASURED（排在執行鏈最後，報告產生時還沒跑完） |")
s = s.replace("<<B_EXTRA_ROWS>>", "\n".join(extra))
# run table
runs = sorted(d for d in os.listdir("/mlsteam/data/tiara/runs") if d.startswith("20261008") and "-m7-" in d)
rt = ["| run_id | exit | 用途 |", "|:--|:--|:--|"]
for d in runs:
    ec = open(f"/mlsteam/data/tiara/runs/{d}/exit_code").read().strip() if os.path.exists(f"/mlsteam/data/tiara/runs/{d}/exit_code") else "（執行中）"
    cmd = open(f"/mlsteam/data/tiara/runs/{d}/cmd.sh").read().strip().split(" python ")[-1][:110] if os.path.exists(f"/mlsteam/data/tiara/runs/{d}/cmd.sh") else ""
    rt.append(f"| `{d}` | {ec} | `{cmd}` |")
s = s.replace("<<RUN_TABLE>>", "\n".join(rt))
open(D, "w").write(s)
left = re.findall(r"<<[A-Z0-9_]+>>", s)
print("remaining placeholders:", left)
