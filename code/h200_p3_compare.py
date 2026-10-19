"""h200_p3_compare.py — 平台 C（H200）重跑的第三階段輸出 vs 3090 的同一份檔，逐值對拍（2026-10-19）。

  python code/h200_p3_compare.py --ref results/p3_data --new results/p3_data_h200 \
      --files summary.csv crosscheck.csv service_constants.json synth_fit_fi_week01.json --out results/p3_data_h200/compare_3090.json

只用於「不依硬體」的輸出（資料層、統計）：同一份輸入、同一支程式，數字應該逐位元相同。
不比的欄位：run_id、ts、x_run_id 以外的執行資訊（run_id、ts、path、code、cache、host）；路徑字串裡的資料根目錄統一換成 <DS>。
有差異就照列（每個差異：檔、列鍵、欄、3090 值、H200 值、相對差），**不改判、不吞**；verdict＝IDENTICAL／DIFF／MISSING。
只用標準函式庫。
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime

VOLATILE = {"run_id", "ts", "path", "code", "cache", "host", "downloaded_by", "download_date"}
ROOTS = ("/ssd8/hungwei/paper-hkv/datasets", os.environ.get("TIARA_DATASETS", "\0"))


def norm(v):
    if isinstance(v, str):
        for r in ROOTS:
            if r and r != "\0":
                v = v.replace(r, "<DS>")
    return v


def rel(a, b):
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return None
    if fa == fb:
        return 0.0
    return abs(fa - fb) / max(abs(fa), abs(fb), 1e-300)


def cmp_csv(pr, pn):
    ra = list(csv.DictReader(open(pr, encoding="utf-8")))
    rb = list(csv.DictReader(open(pn, encoding="utf-8")))
    cols = [c for c in (ra[0].keys() if ra else []) if c not in VOLATILE]
    colsb = [c for c in (rb[0].keys() if rb else []) if c not in VOLATILE]
    diffs = []
    if cols != colsb:
        diffs.append(dict(kind="columns", ref=cols, new=colsb))
    key = [c for c in ("source", "variant", "metric", "tier") if c in cols]

    def k(r):
        return tuple(norm(r.get(c, "")) for c in key)
    da, db = {}, {}
    for r in ra:
        da.setdefault(k(r), []).append(r)
    for r in rb:
        db.setdefault(k(r), []).append(r)
    for kk in sorted(set(da) | set(db)):
        xa, xb = da.get(kk, []), db.get(kk, [])
        if len(xa) != len(xb):
            diffs.append(dict(kind="row_count", key=kk, ref=len(xa), new=len(xb)))
            continue
        for a, b in zip(xa, xb):
            for c in cols:
                va, vb = norm(a.get(c, "")), norm(b.get(c, ""))
                if va != vb:
                    diffs.append(dict(kind="value", key=kk, col=c, ref=va, new=vb, rel=rel(va, vb)))
    return dict(n_ref=len(ra), n_new=len(rb), key=key, n_diff=len(diffs), diffs=diffs[:200])


def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for kk in sorted(set(a) | set(b)):
            if kk in VOLATILE:
                continue
            if kk not in a or kk not in b:
                out.append(dict(kind="key", path=path + [kk], ref=kk in a, new=kk in b))
                continue
            walk(a[kk], b[kk], path + [kk], out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(dict(kind="len", path=path, ref=len(a), new=len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, path + [i], out)
    else:
        va, vb = norm(a), norm(b)
        if va != vb:
            out.append(dict(kind="value", path=path, ref=va, new=vb, rel=rel(va, vb)))


def cmp_json(pr, pn):
    out = []
    walk(json.load(open(pr, encoding="utf-8")), json.load(open(pn, encoding="utf-8")), [], out)
    return dict(n_diff=len(out), diffs=out[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep = dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"),
               ref=a.ref, new=a.new, ignored_fields=sorted(VOLATILE), files={})
    worst = "IDENTICAL"
    for f in a.files:
        pr, pn = os.path.join(a.ref, f), os.path.join(a.new, f)
        if not (os.path.exists(pr) and os.path.exists(pn)):
            rep["files"][f] = dict(verdict="MISSING", ref_exists=os.path.exists(pr), new_exists=os.path.exists(pn))
            worst = "MISSING"
            continue
        r = cmp_csv(pr, pn) if f.endswith(".csv") else cmp_json(pr, pn)
        r["verdict"] = "IDENTICAL" if r["n_diff"] == 0 else "DIFF"
        if r["verdict"] == "DIFF" and worst == "IDENTICAL":
            worst = "DIFF"
        rep["files"][f] = r
        mx = max([d["rel"] for d in r["diffs"] if d.get("rel") is not None] or [0.0])
        print(f"[compare] {f}: {r['verdict']}（差異 {r['n_diff']} 處，最大相對差 {mx:.3g}）")
    rep["verdict"] = worst
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rep, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print(f"[compare] verdict={worst} → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
