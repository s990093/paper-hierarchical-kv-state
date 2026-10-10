"""m9_f1_analyze.py — 把 F1 各 run 的輸出合併到 results/m9_followup/，套用事先寫好的判準。

用法：python code/m9_f1_analyze.py <out_dir> <run_dir>...
判準：docs/research_20261010_followup/F1_offload_slowdown.md §2（寫於 2026-10-10T07:41:00Z）。
  主指標：chat C=1 → 回來請求 TTFT 中位數；doc C=1 → 首輪 TTFT 中位數；doc C=16 → 回來請求 TTFT 中位數。
  某格重現＝3 個 seed 裡 ≥2 個 cpu50/off−1 ≥ 10%；某格是假象＝≥2 個 < 10%。
  整體重現＝≥2 格重現；整體是假象＝≥2 格是假象。
GpuWatcher 規則（沿用 D2 的 guard_status）：全部外來者都是 pid −1、且第一次出現在最後一格寫完之後 → 已知誤報，接受。
"""
from __future__ import annotations

import collections
import csv
import json
import os
import statistics
import sys
from datetime import datetime, timezone

MAIN = {("chat", 1): "ret", ("doc", 1): "first", ("doc", 16): "ret"}
SEEDS = (1, 2, 3)


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def guard_status(rd):
    gp = os.path.join(rd, "gpu_guard.json")
    if not os.path.exists(gp):
        return False, "no_gpu_guard_json"
    g = json.load(open(gp))
    if not g.get("contaminated"):
        return True, "clean"
    if not g.get("started_clean", False):
        return False, "dirty_at_start"
    # 「最後一格」＝這個 run 最後寫出的結果列（正式格、profile 或 kvcheck，取最晚的）
    rows = []
    for name in ("f1_cells.csv", "f1_profile.csv", "f1_kvcheck.csv", "d2_cells.csv"):
        pth = os.path.join(rd, name)
        if os.path.exists(pth):
            rows += list(csv.DictReader(open(pth)))
    if not rows:
        return False, "contaminated_no_cells"
    last_ts = max(datetime.fromisoformat(r["ts"]) for r in rows)
    intr = g.get("intruders", [])
    if intr and all(i.get("pid") == -1 for i in intr) and \
            min(datetime.fromisoformat(i["first_seen"]) for i in intr) >= last_ts:
        return True, "pid-1_after_last_cell(known_false_positive)"
    return False, "contaminated:" + json.dumps(intr)[:300]


def med(xs):
    return statistics.median(xs) if xs else None


def write(path, rows):
    if not rows:
        print("EMPTY", path)
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("WROTE", path, len(rows))


def main():
    out_dir, run_dirs = sys.argv[1], sys.argv[2:]
    os.makedirs(out_dir, exist_ok=True)
    cells, reqs, steps, outs, prof, guard, kvc, tier = [], [], [], [], [], [], [], []
    for rd in run_dirs:
        rid = os.path.basename(rd.rstrip("/"))
        ok, note = guard_status(rd)
        ec = open(os.path.join(rd, "exit_code")).read().strip() if os.path.exists(os.path.join(rd, "exit_code")) else "NA"
        guard.append(dict(run_id=rid, ts=now_iso(), guard_ok=ok, guard_note=note, exit_code=ec))
        if not ok:
            continue
        for name, lst in (("f1_cells.csv", cells), ("f1_requests.csv", reqs), ("f1_steps.csv", steps),
                          ("f1_outputs.csv", outs), ("f1_profile.csv", prof), ("f1_kvcheck.csv", kvc),
                          ("d2_cells.csv", tier)):
            p = os.path.join(rd, name)
            if os.path.exists(p):
                lst += list(csv.DictReader(open(p)))
    write(os.path.join(out_dir, "f1_guard.csv"), guard)
    write(os.path.join(out_dir, "f1_cells.csv"), cells)
    if kvc:
        write(os.path.join(out_dir, "f1_kvcheck.csv"), kvc)
    if tier:
        write(os.path.join(out_dir, "f1_tier50_cells.csv"), tier)
        trs = []
        for rd in run_dirs:
            pth = os.path.join(rd, "d2_requests.csv")
            if os.path.exists(pth) and any(t["run_id"] == os.path.basename(rd.rstrip("/")) for t in tier):
                rr = [r for r in csv.DictReader(open(pth)) if int(r["turn"]) > 0]
                tt = sorted(float(r["ttft_s"]) for r in rr if r["ttft_s"])
                prev = sum(int(r["prev_ctx_len"]) for r in rr)
                rec_prev = sum(max(0, int(r["prev_ctx_len"]) - int(r["gpu_hit"]) - int(r["cpu_hit"])) for r in rr)
                trs.append(dict(run_id=rr[0]["run_id"], ts=now_iso(), n_ret=len(rr),
                                ret_ttft_med=round(statistics.median(tt), 4), ret_ttft_p90=round(tt[int(0.9 * (len(tt) - 1))], 4),
                                ret_ttft_max=round(tt[-1], 4), ret_defer_s_max=round(max(float(r["defer_s"]) for r in rr), 4),
                                n_ret_ttft_gt_10s=sum(t > 10 for t in tt),
                                ret_prev_recomputed_frac=round(rec_prev / prev, 5) if prev else ""))
        if trs:
            write(os.path.join(out_dir, "f1_tier50_summary.csv"), trs)
    if prof:
        write(os.path.join(out_dir, "f1_profile.csv"), prof)
        ps = []
        for r in prof:
            if r["category_totals_json"]:
                tot = json.loads(r["category_totals_json"])
                ps.append(dict(run_id=r["run_id"], ts=now_iso(), cfg=r["cfg"], backend=r["backend"],
                               wall_s=r["wall_s"], **{f"{k}_s": round(v / 1e6, 4) for k, v in sorted(tot.items())},
                               top_kernel=r["kernel"][:80]))
        write(os.path.join(out_dir, "f1_profile_summary.csv"), ps)

    # ---- 每 (cfg, seed, workload, conc) 的主指標 ----
    by = collections.defaultdict(list)
    for r in reqs:
        by[(r["cfg"], int(r["seed"]), r["workload"], int(r["conc"]))].append(r)
    metric, runof = {}, {}
    rowsm = []
    for (cfg, seed, wl, conc), rr in sorted(by.items()):
        first = [float(x["ttft_s"]) for x in rr if int(x["turn"]) == 0 and x["ttft_s"]]
        ret = [float(x["ttft_s"]) for x in rr if int(x["turn"]) > 0 and x["ttft_s"]]
        prev = sum(int(x["prev_ctx_len"]) for x in rr if int(x["turn"]) > 0)
        g = sum(int(x["gpu_hit"]) for x in rr if int(x["turn"]) > 0)
        cp = sum(int(x["cpu_hit"]) for x in rr if int(x["turn"]) > 0)
        kind = MAIN.get((wl, conc))
        m = med(first) if kind == "first" else med(ret) if kind == "ret" else None
        metric[(cfg, seed, wl, conc)] = m
        runof[(cfg, seed, wl, conc)] = rr[0]["run_id"]
        # 每個 8192-token 的整塊 prefill 步花多久（只看「單一請求、排滿 8192」的步）
        st = [float(s["dt_s"]) for s in steps if s["cfg"] == cfg and s["workload"] == wl and int(s["conc"]) == conc
              and s["run_id"] == rr[0]["run_id"] and int(s["sched_tokens"]) == 8192 and int(s["n_reqs"]) == 1]
        rowsm.append(dict(run_id=rr[0]["run_id"], ts=now_iso(), cfg=cfg, seed=seed, workload=wl, conc=conc,
                          backend=rr[0].get("backend", ""), n_first=len(first), n_ret=len(ret),
                          first_ttft_med=round(med(first), 4) if first else "",
                          ret_ttft_med=round(med(ret), 4) if ret else "",
                          main_metric=kind or "", main_value=round(m, 4) if m is not None else "",
                          ret_gpu_frac=round(g / prev, 4) if prev else "", ret_cpu_frac=round(cp / prev, 4) if prev else "",
                          ret_recomp_frac=round((prev - g - cp) / prev, 4) if prev else "",
                          full_chunk_steps=len(st), full_chunk_step_med_s=round(med(st), 5) if st else ""))
    write(os.path.join(out_dir, "f1_metrics.csv"), rowsm)

    # ---- 重現判定（事先寫好的）----
    rep_rows, cell_verdict = [], {}
    for (wl, conc), kind in MAIN.items():
        n_rep = n_art = 0
        for s in SEEDS:
            o, c = metric.get(("off", s, wl, conc)), metric.get(("cpu50", s, wl, conc))
            sl = (c / o - 1) if (o and c) else None
            if sl is not None:
                n_rep += sl >= 0.10
                n_art += sl < 0.10
            rep_rows.append(dict(run_id=f'{runof.get(("off", s, wl, conc), "NA")};{runof.get(("cpu50", s, wl, conc), "NA")}',
                                 ts=now_iso(), workload=wl, conc=conc, metric=kind, seed=s,
                                 off_s=round(o, 4) if o else "NOT_MEASURED", cpu50_s=round(c, 4) if c else "NOT_MEASURED",
                                 slowdown=round(sl, 4) if sl is not None else "NOT_MEASURED"))
        v = "reproduces" if n_rep >= 2 else "artifact" if n_art >= 2 else "undecided"
        cell_verdict[(wl, conc)] = v
        rep_rows.append(dict(run_id="", ts=now_iso(), workload=wl, conc=conc, metric=kind, seed="verdict",
                             off_s="", cpu50_s="", slowdown=f"{v} (n>=10%: {n_rep}, n<10%: {n_art})"))
    nv = collections.Counter(cell_verdict.values())
    overall = "reproduces" if nv["reproduces"] >= 2 else "artifact" if nv["artifact"] >= 2 else "undecided"
    rep_rows.append(dict(run_id="", ts=now_iso(), workload="ALL", conc="", metric="", seed="overall", off_s="",
                         cpu50_s="", slowdown=overall))
    write(os.path.join(out_dir, "f1_repro.csv"), rep_rows)
    print("OVERALL", overall, dict(cell_verdict))

    # ---- 拆解：每一格、每個 seed，各 cfg 相對 off 多了多少；backend 解釋多少 ----
    cfgs = sorted({k[0] for k in metric})
    dec = []
    for (wl, conc), kind in MAIN.items():
        for s in sorted({k[1] for k in metric}):
            o = metric.get(("off", s, wl, conc))
            c = metric.get(("cpu50", s, wl, conc))
            if not o:
                continue
            row = dict(run_id=";".join(runof[k] for k in runof if k[1] == s and k[2] == wl and k[3] == conc),
                       ts=now_iso(), workload=wl, conc=conc, metric=kind, seed=s)
            for cf in cfgs:
                v = metric.get((cf, s, wl, conc))
                row[f"{cf}_s"] = round(v, 4) if v else ""
            exc = (c - o) if c else None
            row["excess_cpu50_minus_off_s"] = round(exc, 4) if exc is not None else ""
            t = metric.get(("off_triton", s, wl, conc))
            row["backend_part_s"] = round(t - o, 4) if t else ""
            row["backend_explained_frac"] = round((t - o) / exc, 4) if (t and exc) else ""
            f = metric.get(("cpu50_rocmfix", s, wl, conc))
            row["rocmfix_minus_off_s"] = round(f - o, 4) if f else ""
            row["rocmfix_removed_frac"] = round((c - f) / exc, 4) if (f and exc) else ""
            ns = metric.get(("cpu50_nostore", s, wl, conc))
            row["store_part_s"] = round(c - ns, 4) if (ns and c) else ""
            row["store_part_frac"] = round((c - ns) / exc, 4) if (ns and c and exc) else ""
            row["connector_triton_nostore_minus_off_s"] = round(ns - o, 4) if ns else ""
            row["connector_triton_nostore_frac"] = round((ns - o) / exc, 4) if (ns and exc) else ""
            fc = metric.get(("cpu50_rocmfix_bs256_coal", s, wl, conc))
            row["rocmfix_coal_minus_off_s"] = round(fc - o, 4) if fc else ""
            row["rocmfix_coal_removed_frac"] = round((c - fc) / exc, 4) if (fc and exc) else ""
            fns = metric.get(("cpu50_rocmfix_nostore", s, wl, conc))
            row["rocmfix_store_part_s"] = round(f - fns, 4) if (f and fns) else ""
            dec.append(row)
        # 事先寫好的：excess 用 3 個 seed 的中位數（每個 cfg 先取各 seed 的中位數）
        mv = {}
        for cf in cfgs:
            xs = [metric[(cf, s, wl, conc)] for s in SEEDS if metric.get((cf, s, wl, conc))]
            mv[cf] = (statistics.median(xs), len(xs)) if xs else (None, 0)
        o, c, t, f = mv.get("off", (None, 0))[0], mv.get("cpu50", (None, 0))[0], mv.get("off_triton", (None, 0))[0], \
            mv.get("cpu50_rocmfix", (None, 0))[0]
        if o and c:
            exc = c - o
            row = dict(run_id=";".join(sorted({runof[k] for k in runof if k[2] == wl and k[3] == conc and k[1] in SEEDS})),
                       ts=now_iso(), workload=wl, conc=conc, metric=kind, seed="median_over_seeds")
            for cf in cfgs:
                row[f"{cf}_s"] = round(mv[cf][0], 4) if mv[cf][0] else ""
                row[f"{cf}_nseeds"] = mv[cf][1]
            row["excess_cpu50_minus_off_s"] = round(exc, 4)
            row["backend_part_s"] = round(t - o, 4) if t else ""
            row["backend_explained_frac"] = round((t - o) / exc, 4) if t else ""
            row["rocmfix_minus_off_s"] = round(f - o, 4) if f else ""
            row["rocmfix_removed_frac"] = round((c - f) / exc, 4) if f else ""
            dec.append(row)
    write(os.path.join(out_dir, "f1_decomp.csv"), dec)

    # ---- 回來的請求，依「在 cpu50 run 裡有沒有 CPU 命中」分兩組，跨 cfg 比同一批請求 ----
    rmap = collections.defaultdict(dict)
    for r in reqs:
        rmap[(r["cfg"], int(r["seed"]), r["workload"], int(r["conc"]))][(r["sess"], r["turn"])] = r
    split = []
    for (cf, sd, wl, conc), d in sorted(rmap.items()):
        ref = rmap.get(("cpu50", sd, wl, conc))
        if not ref:
            continue
        for grp, sel in (("cpu_hit_in_cpu50", lambda r: int(r["cpu_hit"]) > 0),
                         ("no_cpu_hit_in_cpu50", lambda r: int(r["cpu_hit"]) == 0)):
            ks = [k for k, r in ref.items() if int(r["turn"]) > 0 and sel(r)]
            v = [float(d[k]["ttft_s"]) for k in ks if k in d and d[k]["ttft_s"]]
            if not v:
                continue
            split.append(dict(run_id=next(iter(d.values()))["run_id"], ts=now_iso(), cfg=cf, seed=sd, workload=wl,
                              conc=conc, group=grp, n=len(v), ttft_med=round(statistics.median(v), 4),
                              cpu_hit_tok_med_this_cfg=statistics.median(int(d[k]["cpu_hit"]) for k in ks if k in d),
                              prev_ctx_med=statistics.median(int(d[k]["prev_ctx_len"]) for k in ks if k in d)))
    if split:
        write(os.path.join(out_dir, "f1_ret_split.csv"), split)

    # ---- 輸出 token 比對（修法有沒有弄壞 KV）：同 seed、同格、同 req_id ----
    cpuhit = {}
    for r in reqs:
        rid_ = f'c{int(r["conc"])}s{int(r["sess"]):02d}t{int(r["turn"])}'
        cpuhit[(r["cfg"], int(r["seed"]), r["workload"], int(r["conc"]), rid_)] = int(r["cpu_hit"] or 0)
    tok = {}
    for r in outs:
        tok[(r["cfg"], int(r["seed"]), r["workload"], int(r["conc"]), r["req_id"])] = (r["tokens"], r["run_id"])
    pairs = [("off", "off_triton"), ("off_triton", "cpu50"), ("off", "cpu50"), ("off", "cpu50_rocmfix"),
             ("cpu50", "cpu50_rocmfix"), ("off", "cpu50_rocmnaive"), ("off_triton", "cpu50_nostore"),
             ("off", "cpu50_rocmfix_nostore"), ("off", "cpu50_rocmfix_bs256_coal"),
             ("cpu50_rocmfix", "cpu50_rocmfix_bs256_coal")]
    mrows = []
    for a, b in pairs:
        grp = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0])   # n, exact, first exact, ret exact, n cpu-hit, cpu-hit exact
        rids = collections.defaultdict(set)
        for (cf, s, wl, conc, rid), (t, ra) in tok.items():
            if cf != a:
                continue
            t2 = tok.get((b, s, wl, conc, rid))
            if t2 is None:
                continue
            t2, rb = t2
            rids[(s, wl, conc)].update((ra, rb))
            g = grp[(s, wl, conc)]
            same = t == t2
            g[0] += 1
            g[1] += same
            if rid.endswith("t0"):
                g[2] += same
            else:
                g[3] += same
            if cpuhit.get((b, s, wl, conc, rid), 0) > 0 or cpuhit.get((a, s, wl, conc, rid), 0) > 0:
                g[4] += 1
                g[5] += same
        for (s, wl, conc), (n, ex, ex0, exr, nh, exh) in sorted(grp.items()):
            mrows.append(dict(run_id=";".join(sorted(rids[(s, wl, conc)])), ts=now_iso(), cfg_a=a, cfg_b=b, seed=s, workload=wl, conc=conc, n_req=n,
                              exact_match=ex, exact_match_first_turn=ex0, exact_match_returning=exr,
                              match_frac=round(ex / n, 4) if n else "",
                              n_req_with_cpu_hit=nh, exact_match_cpu_hit=exh))
    if mrows:
        write(os.path.join(out_dir, "f1_output_match.csv"), mrows)


if __name__ == "__main__":
    main()
