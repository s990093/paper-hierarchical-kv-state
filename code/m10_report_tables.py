"""m10_report_tables.py — 把判定程式的 JSON／CSV 轉成報告用的 Markdown 表（不重算任何判準，只排版）。

  python code/m10_report_tables.py k0       --json <k0_verdict.json>                       # K0 每格
  python code/m10_report_tables.py speed    --summary <summary.csv> [--length 32768]       # 每格中位數
  python code/m10_report_tables.py quality  --json <quality_*.json> [--configs …]          # ε 判定
  python code/m10_report_tables.py e0       --json <quality_*.json>                        # E0 各格答對率
  python code/m10_report_tables.py lb       --table <quality_*_table.csv>                  # LongBench（只報告）
  python code/m10_report_tables.py decide   --json <decide_*.json>
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

ORDER = ["E0", "E1", "E1s", "E1sr", "F25", "M25", "B25", "R25"]
ST = {"idle": "閒", "busy": "忙"}
CONDN = {"A": "A（全載入）", "B-idle": "B・閒", "B-busy": "B・忙", "B-i2b": "B・閒→忙"}


def pct(x, k=1):
    return "—" if x is None else f"{100 * x:+.{k}f}"


def k0(a):
    d = json.load(open(a.json))
    print(f"δ*＝{100 * d['delta_star']:.2f}%（{d['delta_source']}）；格式 {d['fmt']}；run {d['run_id']}\n")
    print("| 頻寬 | GPU | E0 中位數 (s) | E1 中位數 (s) | E1 快多少 | 配對差 95% CI (s) | E1 贏？ | 會合點 E0／E1 |")
    print("|:--|:--|--:|--:|--:|:--|:--|:--|")
    for c in d["cells"]:
        lo, hi = c["paired_diff_ci95_s"]
        print(f"| {c['bw_name']} | {ST[c['gpu_state']]} | {c['median_E0_s']:.3f} | {c['median_E1_s']:.3f} | "
              f"{100 * c['gain_E1_over_E0']:.1f}% | [{lo:.3f}, {hi:.3f}] | {'是' if c['E1_wins'] else '否'} | "
              f"{','.join(map(str, c['b_E0']))}／{','.join(map(str, c['b_E1']))} |")
    print(f"\n**判定**：{d['verdict']}")


def speed(a):
    rows = [r for r in csv.DictReader(open(a.summary)) if int(r["length"]) == a.length]
    cell = defaultdict(dict)
    for r in rows:
        k = r["config"] if r["mode"] == "cake" else ("只算" if r["mode"] == "compute_only" else f"只載({r['config']})")
        cell[(r["bw_name"], r["gpu_state"])][k] = (float(r["median_t_restore"]), r["b_values"])
    cols = [c for c in ORDER if any(c in v for v in cell.values())] + \
        [c for c in ("只算", "只載(E0)", "只載(E1)") if any(c in v for v in cell.values())]
    print("| 頻寬 | GPU | " + " | ".join(cols) + " |")
    print("|:--|:--|" + "--:|" * len(cols))
    for bw in ("b16", "b32", "b48", "cpu"):
        for st in ("idle", "busy"):
            v = cell.get((bw, st))
            if not v:
                continue
            print(f"| {bw} | {ST[st]} | " + " | ".join(f"{v[c][0]:.3f}" if c in v else "—" for c in cols) + " |")
    print("\n（還原時間中位數，秒；32K）會合點：")
    for bw in ("b16", "b32", "b48", "cpu"):
        for st in ("idle", "busy"):
            v = cell.get((bw, st))
            if v:
                print(f"- {bw}・{ST[st]}：" + "；".join(f"{c} {v[c][1]}" for c in ORDER if c in v))


def quality(a):
    d = json.load(open(a.json))
    cfgs = a.configs or list(d["configs"])
    print(f"模型 {d['model']}（{d['tag']}）；題數 {d['n_items']}；檢定力 {'夠' if d['power_ok'] else '不夠' if d['power_ok'] is False else '—'}\n")
    print("| 配置 | 情境 | 列數 | 對 E0 的差（百分點） | 95% CI | 主判準 | 輔助 | ε | 實際讀到低精度（block，平均） | 翻轉 E0對→錯／錯→對 | 和 E0 輸出相同 | 平均 KL |")
    print("|:--|:--|--:|--:|:--|:--|:--|:--|--:|:--|--:|--:|")
    for c in cfgs:
        for cond, e in d["configs"].get(c, {}).items():
            fl = e["flips"]
            same = "—" if e["same_as_e0"] is None else f"{100 * e['same_as_e0']:.1f}%"
            kl = "—" if e["mean_kl_vs_e0"] is None else f"{e['mean_kl_vs_e0']:.4f}"
            print(f"| {c} | {CONDN[cond]} | {e['n_rows']} | {pct(e['diff'])} | [{pct(e['ci95'][0])}, {pct(e['ci95'][1])}] | "
                  f"{'過' if e['main_pass'] else '不過'} | {'過' if e['aux_pass'] else '不過'} | **{'過' if e['eps_pass'] else '不過'}** | "
                  f"{e['mean_lp_read']} | {fl['e0_right_x_wrong']}／{fl['e0_wrong_x_right']} | {same} | {kl} |")
    bad = [(c, cond, x) for c in cfgs for cond, e in d["configs"].get(c, {}).items() for x in e["aux"] if not x["pass"]]
    if bad:
        print("\n輔助判準沒過的（任務族、輪，點估計差）：" + "；".join(f"{c} {CONDN[cond]} {x['task']} 第 {x['turn']} 輪 {pct(x['diff'])}"
                                              for c, cond, x in bad))


def e0(a):
    d = json.load(open(a.json))
    print("| 任務 | 輪 | 帶 | 題數 | E0 正確率 | 在 60–95%？ |")
    print("|:--|--:|:--|--:|--:|:--|")
    for r in d["e0_accuracy"]:
        if r["task"].startswith("LB-"):
            continue
        print(f"| {r['task']} | {r['turn']} | {r['band']} | {r['n']} | {100 * r['acc']:.1f}% | {'是' if r['in_60_95'] else '否'} |")


def lb(a):
    rows = [r for r in csv.DictReader(open(a.table)) if r["task"].startswith("LB-")]
    by = defaultdict(dict)
    for r in rows:
        by[r["task"]][r["config"]] = r
    cfgs = sorted({r["config"] for r in rows}, key=lambda c: ORDER.index(c) if c in ORDER else 99)
    print("| 資料集 | 題數 | E0 | " + " | ".join(f"{c} 差（CI）" for c in cfgs) + " |")
    print("|:--|--:|--:|" + ":--|" * len(cfgs))
    for t, v in sorted(by.items()):
        any_ = next(iter(v.values()))
        print(f"| {t[3:]} | {any_['n']} | {100 * float(any_['acc_e0']):.1f} | " + " | ".join(
            f"{100 * float(v[c]['diff']):+.1f} [{100 * float(v[c]['ci_lo']):+.1f}, {100 * float(v[c]['ci_hi']):+.1f}]" if c in v else "—"
            for c in cfgs) + " |")


def decide(a):
    d = json.load(open(a.json))
    for m in ("llama", "qwen"):
        v = d[m]
        print(f"- **{m}**：K1＝{v['K1']}、K1s＝{v['K1s']}、模型過＝{v['model_pass']}（逐配置：{v['per_config']}；檢定力 {v['power_ok']}）")
    print(f"- **下一步**：{d['next']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n, args in (("k0", ["--json"]), ("quality", ["--json"]), ("e0", ["--json"]), ("lb", ["--table"]),
                    ("decide", ["--json"]), ("speed", ["--summary"])):
        p = sub.add_parser(n)
        for x in args:
            p.add_argument(x, required=True)
        if n == "quality":
            p.add_argument("--configs", nargs="*")
        if n == "speed":
            p.add_argument("--length", type=int, default=32768)
    a = ap.parse_args()
    {"k0": k0, "speed": speed, "quality": quality, "e0": e0, "lb": lb, "decide": decide}[a.cmd](a)


if __name__ == "__main__":
    main()
