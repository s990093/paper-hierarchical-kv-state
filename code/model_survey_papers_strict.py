#!/usr/bin/env python3
"""model_survey_papers.py 的補強：只數「全文提到 ≥N 次」的模型，當作『實際評測用』的較嚴代理。
related work 順帶提一次的不算。讀同一批已存的全文，不重抓。"""
import argparse, csv, os, re, sys, collections
sys.path.insert(0, os.path.dirname(__file__))
from model_survey_papers import MODELS

ap = argparse.ArgumentParser()
ap.add_argument("--papers-dir", required=True)
ap.add_argument("--min-mentions", type=int, default=3)
args = ap.parse_args()
rows = list(csv.DictReader(open(os.path.join(args.papers_dir, "mentions.csv"))))
usable = [r for r in rows if int(r["text_chars"]) >= 5000]
res = [(n, re.compile(p, re.I)) for n, p in MODELS]
cnt_any, cnt_strict = collections.Counter(), collections.Counter()
c25, c26 = collections.Counter(), collections.Counter()
n25 = sum(1 for r in usable if r["published"] < "2026-01-01"); n26 = len(usable) - n25
for r in usable:
    txt = open(os.path.join(args.papers_dir, "text", r["arxiv_id"] + ".txt")).read()
    for n, rx in res:
        k = len(rx.findall(txt))
        if k >= 1: cnt_any[n] += 1
        if k >= args.min_mentions:
            cnt_strict[n] += 1
            (c25 if r["published"] < "2026-01-01" else c26)[n] += 1
out = os.path.join(args.papers_dir, f"model_frequency_min{args.min_mentions}.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", f"papers_ge{args.min_mentions}", "share", "share_2025", "share_2026", "papers_any", "share_any", "n_usable", "n_2025", "n_2026"])
    for n, c in cnt_strict.most_common():
        w.writerow([n, c, f"{c/len(usable):.3f}", f"{c25[n]/n25:.3f}", f"{c26[n]/n26:.3f}", cnt_any[n], f"{cnt_any[n]/len(usable):.3f}", len(usable), n25, n26])
print(f"usable={len(usable)} (2025={n25}, 2026={n26}) → {out}")
for n, c in cnt_strict.most_common(60):
    print(f"  {n:<26} ≥{args.min_mentions}次 {c:>4} {c/len(usable):6.1%}  (2025 {c25[n]/n25:5.1%} → 2026 {c26[n]/n26:5.1%})   任一次 {cnt_any[n]/len(usable):6.1%}")
