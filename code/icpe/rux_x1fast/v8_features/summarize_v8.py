"""icpe-rux v8：smoke_v8.py 的結果彙總（每份紀錄 × 每類檢查的 PASS／FAIL 數；每列有 run_id、ts、host）。
用法：python -I summarize_v8.py <out.csv> <run 目錄> ...（讀 <run>/o/smoke_v8.jsonl）"""
import collections, csv, json, os, sys
out, dirs = sys.argv[1], sys.argv[2:]
rows = []
for d in dirs:
    p = os.path.join(d, "o", "smoke_v8.jsonl")
    R = [json.loads(x) for x in open(p) if x.strip()]
    ctx = open(os.path.join(d, "context.txt")).read().splitlines()
    c = collections.defaultdict(lambda: [0, 0])
    for r in R:
        kind = r["test"].split(" ")[0]
        c[kind][0 if r["ok"] else 1] += 1
    for kind, (np_, nf) in sorted(c.items()):
        rows.append(dict(run_id=os.path.basename(d), ts=ctx[0].strip(), host=ctx[1].strip(), trace=R[0]["trace"], check=kind, n_pass=np_, n_fail=nf))
    rows.append(dict(run_id=os.path.basename(d), ts=ctx[0].strip(), host=ctx[1].strip(), trace=R[0]["trace"], check="ALL",
                     n_pass=sum(v[0] for v in c.values()), n_fail=sum(v[1] for v in c.values())))
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["run_id", "ts", "host", "trace", "check", "n_pass", "n_fail"])
    w.writeheader()
    w.writerows(rows)
for r in rows:
    if r["check"] == "ALL":
        print(r["trace"], r["n_pass"], r["n_fail"])
print(f"{len(rows)} rows -> {out}")
