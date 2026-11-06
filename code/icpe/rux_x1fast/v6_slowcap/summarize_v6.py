"""icpe-rux v6／v7：對拍與速度的摘要 CSV（每列有 run_id、ts、host）。
用法：python -I summarize_v6.py <out.csv> <run 目錄> [<run 目錄> ...]
  validate.sh 的 run：每個子集 → CSV／NPZ 的列數、格數、陣列數、不同數；Rust 與 Python 的 job 數、sweep 牆鐘（12 worker）、最大 RSS。
  G1／xc_sa 的 run：next_b_g0 的格數與不同數。
  speed_v6.py 的 run：每個設定的中位數（單執行緒），Python ÷ Rust，指紋是否相同。"""
import csv, glob, json, os, re, statistics, sys

out, dirs = sys.argv[1], sys.argv[2:]
W = []


def ctx(d):
    L = open(os.path.join(d, "context.txt"), encoding="utf-8").read().splitlines()
    return L[0].strip(), L[1].strip()


def put(d, trace, subset, metric, value, unit, kind="measured", note="", threads=""):
    ts, host = ctx(d)
    W.append(dict(run_id=os.path.basename(d), ts=ts, host=host, trace=trace, subset=subset, metric=metric, value=value, unit=unit,
                  kind=kind, threads=threads, note=note))


for d in dirs:
    so = open(os.path.join(d, "stdout.log"), encoding="utf-8", errors="replace").read()
    name = os.path.basename(d)
    if re.search(r"-v[67]-(val|chk|w16)-", name):
        trace = re.search(r"validate\.sh (\S+)", open(os.path.join(d, "cmd.sh")).read()).group(1)
        for sub in sorted(os.listdir(os.path.join(d, "v"))):
            blk = so.split(f"== compare {sub} ")[1].split("== ")[0] if f"== compare {sub} " in so else ""
            m = re.search(r"rows ref=(\d+) new=(\d+) only_ref=(\d+) only_new=(\d+)\ncols_compared=(\d+) cells compared=(\d+) differing=(\d+)", blk)
            if m:
                put(d, trace, sub, "csv_rows", int(m.group(1)), "rows")
                put(d, trace, sub, "csv_cells_compared", int(m.group(6)), "cells")
                put(d, trace, sub, "csv_cells_differing", int(m.group(7)), "cells", note=f"only_ref {m.group(3)} only_new {m.group(4)}")
            m = re.search(r"files ref=(\d+) new=(\d+) arrays_compared=(\d+) differing=(\d+)", blk)
            if m:
                put(d, trace, sub, "npz_files", int(m.group(1)), "files")
                put(d, trace, sub, "npz_arrays_compared", int(m.group(3)), "arrays")
                put(d, trace, sub, "npz_arrays_differing", int(m.group(4)), "arrays")
            for eng in ("rust", "py"):
                lg = os.path.join(d, "v", sub, eng, "sweep.log")
                er = os.path.join(d, "v", sub, eng, "sweep.err")
                if not os.path.exists(lg):
                    continue
                t = open(lg, encoding="utf-8", errors="replace").read()
                reused = eng == "py" and f"== {sub} py：沿用" in so
                j = re.search(r"^jobs=(\d+) workers=(\d+)", t, re.M)
                s = re.search(r"^sweep done ([\d.]+)", t, re.M)
                note = "沿用前一次的 Python 參考（硬連結）" if reused else ""
                if j:
                    put(d, trace, sub, f"{eng}_jobs", int(j.group(1)), "jobs", note=note, threads=j.group(2))
                if s:
                    put(d, trace, sub, f"{eng}_sweep_wall_s", round(float(s.group(1)), 2), "s", note=note, threads=j.group(2) if j else "")
                if os.path.exists(er):
                    r = re.search(r"Maximum resident set size \(kbytes\): (\d+)", open(er).read())
                    if r:
                        put(d, trace, sub, f"{eng}_max_rss_MiB", round(int(r.group(1)) / 1024, 1), "MiB", note=note)
            a = [x for x in W if x["run_id"] == name and x["subset"] == sub]
            rw = [x for x in a if x["metric"] == "rust_sweep_wall_s"]
            pw = [x for x in a if x["metric"] == "py_sweep_wall_s"]
            if rw and pw:
                put(d, trace, sub, "py_over_rust_sweep_wall", round(pw[0]["value"] / rw[0]["value"], 1), "x", kind="推算",
                    note=f"同一批 job、Rust {rw[0]['threads']} worker／Python {pw[0]['threads']} worker、不同時段" + ("（Python 是沿用的參考）" if pw[0]["note"] else ""), threads=rw[0]["threads"])
    elif re.search(r"-v[67]-off-", name):
        for m in re.finditer(r"rows ref=(\d+) new=(\d+) cols_compared=(\d+)\ncells compared=(\d+) differing=(\d+)\n(G0_RESULT \w+)", so):
            put(d, "g1" if "g1" in name else "xc_sa", "feature_off", "g0_cells_compared", int(m.group(4)), "cells")
            put(d, "g1" if "g1" in name else "xc_sa", "feature_off", "g0_cells_differing", int(m.group(5)), "cells", note=m.group(6))
    for sp in glob.glob(os.path.join(d, "**", "speed.jsonl"), recursive=True):
        rows = [json.loads(x) for x in open(sp) if x.strip()]
        by = {}
        for r in rows:
            by.setdefault((r["trace"], r["config"], r["engine"]), []).append(r)
        for (tr, cfg, eng), rs in sorted(by.items()):
            xs = [r["cpu_s"] for r in rs]
            dg = sorted({r["digest"] for r in rs})
            put(d, tr, cfg, f"{eng}_cpu_s_median", round(statistics.median(xs), 5), "s", threads=1,
                note=f"n {len(xs)} min {min(xs)} max {max(xs)} load {min(r['load_1m'] for r in rs)}–{max(r['load_1m'] for r in rs)} "
                     f"digests {'same' if len(dg) == 1 else 'DIFFER'} {dg[0]} sc(clean,drop) {rs[0]['sc_counts']}")
        for (tr, cfg, eng), rs in sorted(by.items()):
            if eng != "py":
                continue
            for rt in ("rust_v6", "rust_v7"):
                rr = by.get((tr, cfg, rt))
                if rr:
                    pm = statistics.median([r["cpu_s"] for r in rs])
                    rm = statistics.median([r["cpu_s"] for r in rr])
                    same = {r["digest"] for r in rs} == {r["digest"] for r in rr}
                    put(d, tr, cfg, f"py_over_{rt}", round(pm / rm, 1), "x", kind="推算", threads=1, note=f"py {pm:.4f} / {rt} {rm:.5f}")
                    put(d, tr, cfg, f"digest_py_eq_{rt}", int(same), "bool", threads=1)
        for (tr, cfg, eng), rs in sorted(by.items()):
            if eng != "rust_v6" or (tr, cfg, "rust_v7") not in by:
                continue
            r7 = by[(tr, cfg, "rust_v7")]
            m6 = statistics.median([r["cpu_s"] for r in rs]); m7 = statistics.median([r["cpu_s"] for r in r7])
            put(d, tr, cfg, "rust_v6_over_v7", round(m6 / m7, 2), "x", kind="推算", threads=1, note=f"v6 {m6:.5f} / v7 {m7:.5f}")
            put(d, tr, cfg, "digest_v6_eq_v7", int({r["digest"] for r in rs} == {r["digest"] for r in r7}), "bool", threads=1)

cols = ["run_id", "ts", "host", "trace", "subset", "metric", "value", "unit", "kind", "threads", "note"]
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for x in W:
        w.writerow(x)
print(f"{len(W)} rows -> {out}")
