"""icpe-rux 第 1 段：把 prof_sc.py 的 rows.jsonl 彙總成摘要 CSV（每列都有 run_id、ts、host）。
用法：python -I summarize.py <out.csv> <rows.jsonl> [<rows.jsonl> ...]
列的種類（metric）：
  py_cpu_s／rust_cpu_s：同一設定各次的中位數（另記最小、最大、次數、load 範圍、各次 digest 是否相同）
  ratio_sc_over_nocap：Python 有上限 ÷ 沒有上限（同樣的時間設定數；GDSF 對 nocap1）〔算術〕
  ratio_py_over_rust：Python ÷ Rust v5（只有沒有上限的設定）〔算術〕
  digest_py_eq_rust：Python 與 Rust v5 的逐請求陣列指紋是否相同（沒有上限的設定）
  counter_*：引擎計數（sc_cap、n_sc_clean、n_sc_drop、n_demote、n_drop）
  ops_*：cProfile 那一次的慢層結構操作次數（select 次數、select 時慢層 session 數的平均與最大）
  cprof_*：cProfile 自身時間占比前幾名"""
import csv, json, statistics, sys

out, paths = sys.argv[1], sys.argv[2:]
rows, fo_rows, cnt_rows = [], [], []
for p in paths:
    for ln in open(p):
        if not ln.strip().startswith("{"):
            continue
        r = json.loads(ln)
        if "rstar" in r:
            fo_rows.append(r)          # first_overflow.py
        elif "jobs_per_variant" in r:
            cnt_rows.append(r)         # count_jobs.py
        else:
            rows.append(r)

W = []


def put(r0, metric, config, value, unit, kind, **extra):
    W.append(dict(run_id=r0["run_id"], ts=r0["ts"], host=r0["host"], trace=r0["trace"], cpu_GiB=r0["cpu_GiB"], policy=r0["policy"],
                  gpu_policy=r0["gpu_policy"], config=config, metric=metric, value=value, unit=unit, kind=kind,
                  threads=r0.get("threads", 1), n=extra.pop("n", ""), load_1m_min=extra.pop("lmin", ""), load_1m_max=extra.pop("lmax", ""),
                  note=extra.pop("note", "")))


groups = {}
for r in rows:
    if r["kind"] in ("py_time", "rust_time"):
        groups.setdefault((r["run_id"], r["trace"], r["cpu_GiB"], r["policy"], r["gpu_policy"], r["engine"], r["config"]), []).append(r)
    if r["kind"] == "prep":
        for k in ("n_req", "n_sess", "B", "n_blocks_full", "chunks"):
            put(r, f"trace_{k}", "-", r[k], "count", "measured")
        put(r, "py_prep_s", "-", r["prep_s"], "s", "measured", lmin=r["load_1m"], lmax=r["load_1m"])

med = {}
for (rid, tr, cg, pol, gp, eng, cfg), rs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
    xs = [x["cpu_s"] for x in rs]
    ls = [x["load_1m_before"] for x in rs] + [x["load_1m_after"] for x in rs]
    dg = sorted({x["digest"] for x in rs})
    m = statistics.median(xs)
    med[(rid, tr, cg, pol, gp, eng, cfg)] = (m, dg, rs[0])
    put(rs[0], f"{'py' if eng == 'py' else 'rust'}_cpu_s", cfg, round(m, 5), "s", "measured", n=len(xs), lmin=min(ls), lmax=max(ls),
        note=f"min {min(xs)} max {max(xs)}; digests {'same' if len(dg) == 1 else 'DIFFER ' + ','.join(dg)} ({dg[0]}); n_views {rs[0]['n_views']}")
    if eng == "py":
        for k in ("sc_cap", "n_sc_clean", "n_sc_drop", "n_demote", "n_drop", "n_gevict"):
            if rs[0].get(k) is not None:
                put(rs[0], f"counter_{k}", cfg, rs[0][k], "count", "measured")

for (rid, tr, cg, pol, gp, eng, cfg), (m, dg, r0) in med.items():
    if eng != "py":
        continue
    if cfg.startswith("sc:"):
        base = "nocap1" if cfg == "sc:GDSF" else "nocap6"
        b = med.get((rid, tr, cg, pol, gp, "py", base))
        if b:
            put(r0, "ratio_sc_over_nocap", cfg, round(m / b[0], 3), "x", "推算", note=f"py {cfg} {m:.4f} / py {base} {b[0]:.4f}")
    else:
        rr = med.get((rid, tr, cg, pol, gp, "rust_v5", cfg))
        if rr:
            put(r0, "ratio_py_over_rust", cfg, round(m / rr[0], 2), "x", "推算", note=f"py {m:.4f} / rust_v5 {rr[0]:.5f}")
            put(r0, "digest_py_eq_rust", cfg, int(dg == rr[1]), "bool", "measured", note=f"py {dg} rust {rr[1]}")

for r in rows:
    if r["kind"] != "cprofile":
        continue
    o = r.get("ops") or {}
    if "select" in o:
        put(r, "ops_slow_select", r["config"], o["select"], "count", "measured")
        put(r, "ops_slow_add", r["config"], o.get("add", 0), "count", "measured")
        put(r, "ops_slow_remove", r["config"], o.get("remove", 0), "count", "measured")
        if o["select"]:
            put(r, "ops_slow_select_mem_mean", r["config"], round(o["select_mem_sum"] / o["select"], 1), "sessions", "measured")
            put(r, "ops_slow_select_mem_max", r["config"], o["select_mem_max"], "sessions", "measured")
    put(r, "cprof_total_s", r["config"], r["tottime_sum"], "s", "measured", lmin=r["load_1m_before"], lmax=r["load_1m_before"],
        note="cProfile 開著（比不開慢）")
    for i, t in enumerate(r["top20_tottime"][:8]):
        put(r, f"cprof_top{i + 1}", r["config"], t["tot_share"], "share", "measured",
            note=f"{t['fn']} ncalls {t['ncalls']} tottime {t['tottime']} cumtime {t['cumtime']}")

for r in fo_rows:
    r0 = dict(r, cpu_GiB=r["cpu_GiB"])
    for k in ("sc_cap", "first_clean_r", "first_drop_r", "rstar", "rstar_frac", "n_sc_clean", "n_sc_drop"):
        put(r0, f"first_overflow_{k}", "sc:LRU", r[k], "frac" if k == "rstar_frac" else "count", "measured",
            note="r*＝第一次真的清副本或丟 session 的請求序號；1 個時間設定（ssd idle）")
for r in cnt_rows:
    r0 = dict(r, trace="(每個變體)", cpu_GiB="9 個", policy=f"CPU {r['n_cpu']}", gpu_policy=f"GPU {r['n_gpu']}")
    put(r0, "jobs_per_variant", r["set"], r["jobs_per_variant"], "jobs", "measured", note="x1_sim.jobs_for 直接數")
    put(r0, "rows_per_variant", r["set"], r["rows_per_variant"], "rows", "measured")
    put(r0, "nocap_jobs_per_variant", r["set"], r["nocap_jobs_per_variant"], "jobs", "measured", note="同集合、沒有上限（v5 支援）")
    for k, v in r["jobs_by_slow_policy"].items():
        put(r0, "jobs_per_variant_by_slow_policy", f"{r['set']}:{k}", v, "jobs", "measured")

cols = ["run_id", "ts", "host", "trace", "cpu_GiB", "policy", "gpu_policy", "config", "metric", "value", "unit", "kind", "threads", "n",
        "load_1m_min", "load_1m_max", "note"]
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    for x in W:
        w.writerow(x)
print(f"{len(W)} rows -> {out}")
