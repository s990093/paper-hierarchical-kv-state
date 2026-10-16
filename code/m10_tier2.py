"""m10_tier2.py — 12 號 B：兩層混合還原（docs/precision_20261013/12_tier_plan.md §3；探索，只報告）

一段 32K 的對話（64 個 block），一部分 block 在 CPU 層（pinned、真實 H2D、不限速，存 BF16），
一部分被降到慢層（同一份 pinned 記憶體＋限速器，頻寬 6.98 或 0.331 GiB/s＝第一階段本地 SSD／NFS 的實測讀頻寬），
慢層存 BF16 或 INT4（int4_kivi）。還原用主線同一個還原器（m7_restore_harness.Restorer，Cake；每個 block 有自己的層 loc[i]）。
主線的 m10_speed.py 一行都不動：這裡 import 它的 Bench（模型、合成 token、寫入時的 BF16／INT4 主機副本、kv_bad）、
Busy（背景負載）、_watcher（gpu_guard）、bootstrap_ci、欄位名。

配置（12 §3.1）：
  兩層：T<old|new|rnd><25|50|75>-<bf16|int4>
        old＝block 0..k−1（最舊的先降），new＝最後 k 個（尾端先淘汰），rnd＝從 64 個抽 k 個（seed＝launch×1000＋rep）
  參考：S-bf16、S-int4（全部在慢層）；E0、E1（全部在 CPU 層，BF16／INT4）
  牆的中間點：w1、w3、w12、w20 這幾個限速層的 S-bf16、S-int4（給 12 §2 的 A2）
層：cpu（只跑 E0、E1）；ssd、nfs（兩層 18 個＋S-bf16、S-int4）；w1／w3／w12／w20（S-bf16、S-int4）

子命令：
  grid     開 --launches 個子 process（每個＝一次「重開」），每個子 process：capture → 暖機 → GPU 狀態 × 重複 × 層 × 配置（同一次重複裡隨機打散）
  smoke    真模型冒煙：1 launch × 1 rep、只跑 ssd、閒，全部配置都檢查 KV（stage 44 開頭）
  summary  restores.csv → summary.csv、compare.csv、tables.md、圖（不用 GPU）
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import random
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

GiB = 1 << 30
TIER_PARAMS = os.path.join(HERE, "..", "results", "m7_write_policy_mi300x", "tier_params.json")
WALL_GIBPS = {"w1": 1.0, "w3": 3.0, "w12": 12.0, "w20": 20.0}
ORDERS = ("old", "new", "rnd")
FRACS = (25, 50, 75)
PRECS = ("bf16", "int4")
TWO_TIER = [f"T{o}{f}-{p}" for o in ORDERS for f in FRACS for p in PRECS]
SLOW_REFS = ["S-bf16", "S-int4"]
CPU_REFS = ["E0", "E1"]
EXTRA_FIELDS = ["tier_slow", "slow_GiBps", "order", "frac", "prec", "n_slow", "n_load_cpu", "n_load_ssd",
                "t_load_cpu", "t_load_ssd"]


def slow_tiers() -> dict:
    """ssd／nfs 的讀頻寬（GiB/s）取自第一階段的 tier_params.json（12 §2.1）；w*＝牆的中間點。"""
    tp = json.load(open(TIER_PARAMS))
    d = {"ssd": float(tp["local"]["read_GiBps"]), "nfs": float(tp["nfs"]["read_GiBps"])}
    d.update(WALL_GIBPS)
    return d


def configs_for(tier: str) -> list:
    if tier == "cpu":
        return list(CPU_REFS)
    if tier in ("ssd", "nfs"):
        return TWO_TIER + SLOW_REFS
    return list(SLOW_REFS)


def parse_cfg(cfg: str):
    """→ (order, frac, prec)；參考點 order＝'all'（S-*：全部在慢層）或 'none'（E0／E1：全部在 CPU）。"""
    if cfg in ("E0", "E1"):
        return "none", 0, "bf16" if cfg == "E0" else "int4"
    if cfg.startswith("S-"):
        return "all", 100, cfg[2:]
    head, prec = cfg[1:].split("-")
    order, frac = head[:3], int(head[3:])
    assert order in ORDERS and frac in FRACS and prec in PRECS, cfg
    return order, frac, prec


def demoted(order: str, frac: int, n: int, seed: int) -> list:
    """每個 block 在不在慢層。"""
    if order == "none":
        return [False] * n
    if order == "all":
        return [True] * n
    k = n * frac // 100
    if order == "old":
        s = set(range(k))
    elif order == "new":
        s = set(range(n - k, n))
    else:
        s = set(random.Random(seed).sample(range(n), k))
    return [i in s for i in range(n)]


def layout2(cfg: str, n: int, seed: int):
    """→ (slow_mask, lp_mask)：哪些 block 在慢層；哪些 block 存 INT4。E1＝全部在 CPU 但存 INT4。"""
    order, frac, prec = parse_cfg(cfg)
    slow = demoted(order, frac, n, seed)
    if order == "none":
        lp = [prec == "int4"] * n
    else:
        lp = [s and prec == "int4" for s in slow]
    return slow, lp


# ====================================================================== GPU 端（子 process）
def restore2(bench, slow_mask, lp_mask, slow_Bps, n):
    """一次兩層的 Cake 還原（每次新建兩個 Tier，時鐘歸零；同 m10_speed.Bench.restore）。"""
    from m7_restore_harness import Tier
    torch, km, rs = bench.torch, bench.km, bench.rs
    cpu = Tier("cpu_real", read_Bps=0, write_Bps=0)
    slow = Tier("ssd_slow", read_Bps=slow_Bps, write_Bps=slow_Bps) if slow_Bps else None
    host = [bench.lp[i] if lp_mask[i] else bench.bf[i] for i in range(n)]
    loc = [slow if slow_mask[i] else cpu for i in range(n)]
    km.kv[:, :, :, :n * bench.CHUNK].zero_()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    rec = rs.restore(bench.ids, n, host, loc, "cake")
    t_new, _ = rs.new_turn(bench.ids[n * bench.CHUNK:n * bench.CHUNK + bench.new_tokens], n * bench.CHUNK)
    rec["ttft"] = time.perf_counter() - t0
    rec["t_new_turn"] = t_new
    return rec


def worker(spec: dict):
    import torch  # noqa: F401
    from m10_speed import BLOCK_FIELDS, RESTORE_FIELDS, Bench, Busy, _watcher, run_id, ts
    rng = random.Random(f"m10-tier2|{spec['launch']}")
    n = spec["length"] // 512
    bench = Bench(spec["model"], n, "int4_kivi", spec["new_tokens"])
    rid = run_id()
    os.makedirs(spec["out_dir"], exist_ok=True)
    rf = open(os.path.join(spec["out_dir"], "restores.csv"), "a", newline="")
    rw = csv.DictWriter(rf, fieldnames=RESTORE_FIELDS + EXTRA_FIELDS, extrasaction="ignore")
    if rf.tell() == 0:
        rw.writeheader()
    bf = open(spec["blocks_out"], "a", newline="")
    bw_ = csv.DictWriter(bf, fieldnames=BLOCK_FIELDS + ["tier"])
    if bf.tell() == 0:
        bw_.writeheader()
    watch, level = _watcher(os.path.join(spec["out_dir"], f"gpu_guard_{rid}_L{spec['launch']}.json"))
    busy_pid = set()

    def contaminated():
        if watch is None:
            return "NOT_CHECKED"
        bad = {p: v for p, v in watch.intruders.items() if p not in busy_pid}
        return bool(bad) or watch.started_clean is False

    bw = spec["tiers"]
    bench.capture(n)
    for cfg in ("E1", "S-int4"):                       # 暖機（不記）：CPU 層、慢層各一次
        sm, lm = layout2(cfg, n, 0)
        restore2(bench, sm, lm, bw.get("ssd", 0) * GiB if cfg.startswith("S") else 0, n)
    verified = set()
    for st in spec["gpu_states"]:
        with Busy(st["duty"], watch) as busy:
            if busy.p:
                busy_pid.add(busy.p.pid)
            for rep in range(spec["reps"]):
                seed = spec["launch"] * 1000 + rep
                jobs = [(t, c) for t in spec["tier_names"] for c in configs_for(t)]
                rng.shuffle(jobs)                      # 同一次重複裡隨機打散（主線 §4.2 的做法）
                for k, (tname, cfg) in enumerate(jobs):
                    order, frac, prec = parse_cfg(cfg)
                    sm, lm = layout2(cfg, n, seed)
                    Bps = 0 if tname == "cpu" else bw[tname] * GiB
                    rec = restore2(bench, sm, lm, Bps, n)
                    key = (cfg, tname, st["name"])
                    kvb = ""
                    if spec["verify"] and (key not in verified or spec.get("verify_all")):
                        kvb = bench.kv_bad(rec, n)
                        verified.add(key)
                    srcs = [s for _, s, _, _ in rec["blocks"]]
                    stored = sum(bench.lp[i].nbytes if lm[i] else bench.km.chunk_bytes for i in range(n))
                    row = dict(
                        run_id=rid, ts=ts(), model=spec["model"], length=spec["length"], n_blocks=n, fmt="int4_kivi",
                        config=cfg, mode="cake", placement_seed=seed if order == "rnd" else "",
                        bw_name=tname, bw_GiBps=bw.get(tname, "real"), gpu_state=st["name"], duty=st["duty"],
                        launch=spec["launch"], rep=rep, order_idx=k, b=rec["meet"], n_recompute=rec["n_recompute"],
                        n_load_bf16=rec["n_load_bf16"], n_load_fp8=rec["n_load_fp8"], n_load_int8=rec["n_load_int8"],
                        n_load_int4=rec["n_load_int4"], fp8_stored=sum(lm),
                        fp8_read=sum(s not in ("recompute", "load_bf16") for s in srcs),
                        bytes_stored=stored, bytes_read=rec["bytes_read"], t_encode="NOT_MEASURED",
                        t_d2h="NOT_MEASURED", t_write="NOT_MEASURED",
                        t_read=round(rec["t_load_cpu"] + rec["t_load_ssd"], 6), t_h2d=round(rec.get("t_h2d", 0.0), 6),
                        t_dequant=round(rec["t_dequant"], 6), t_recompute=round(rec["t_recompute"], 6),
                        t_wait=round(rec.get("t_wait", 0.0), 6), waiting_line=rec.get("waiting_line", ""),
                        t_restore=round(rec["t_restore"], 6), t_new_turn=round(rec["t_new_turn"], 6),
                        ttft=round(rec["ttft"], 6), copy_bound=rec["n_copy_bound"], kv_bad=kvb,
                        contaminated_gpu=contaminated(), host_level=level,
                        tier_slow="" if tname == "cpu" else tname, slow_GiBps="" if tname == "cpu" else bw[tname],
                        order=order, frac=frac, prec=prec, n_slow=sum(sm), n_load_cpu=rec["n_load_cpu"],
                        n_load_ssd=rec["n_load_ssd"], t_load_cpu=round(rec["t_load_cpu"], 6),
                        t_load_ssd=round(rec["t_load_ssd"], 6))
                    rw.writerow(row)
                    for i, s, t_s, t_e in rec["blocks"]:
                        bw_.writerow(dict(run_id=rid, launch=spec["launch"], rep=rep, length=spec["length"],
                                          bw_name=tname, gpu_state=st["name"], config=cfg, mode="cake", block=i,
                                          stored_dtype="int4" if lm[i] else "bf16", source=s,
                                          t_start=round(t_s, 6), t_end=round(t_e, 6),
                                          tier="slow" if sm[i] else "cpu"))
                rf.flush()
                bf.flush()
                print(f"[L{spec['launch']} {st['name']} rep {rep}] ok ({len(jobs)} 次還原)", flush=True)
        if busy.p:
            busy_pid.discard(busy.p.pid)
    rf.close()
    bf.close()
    if watch is not None:
        watch.__exit__(None, None, None)


# ====================================================================== 外層
def load_states(names, busy_levels, busy_key):
    lv = json.load(open(busy_levels))
    return [{"name": s, "duty": 0.0 if s == "idle" else float(lv[busy_key if s == "busy" else s])} for s in names]


def run_launches(a, tier_names, states, reps, launches, verify_all=False):
    from m10_speed import run_id
    os.makedirs(a.out_dir, exist_ok=True)
    spec = dict(model=a.model, length=a.length, new_tokens=a.new_tokens, tiers=slow_tiers(), tier_names=tier_names,
                gpu_states=load_states(states, a.busy_levels, a.busy_key), reps=reps, out_dir=a.out_dir,
                blocks_out=a.blocks_out or os.path.join(a.out_dir, "blocks.csv"), verify=True, verify_all=verify_all)
    json.dump({**spec, "argv": sys.argv, "run_id": run_id()}, open(os.path.join(a.out_dir, f"spec_{a.cmd}.json"), "w"),
              indent=1, ensure_ascii=False)
    for launch in range(launches):
        s = dict(spec, launch=launch)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "_worker", json.dumps(s)])
        if r.returncode != 0:
            raise SystemExit(f"launch {launch} 失敗（exit {r.returncode}）；照規則 2 停下來")


def cmd_grid(a):
    run_launches(a, ["cpu", "ssd", "nfs"] + list(WALL_GIBPS), a.gpu_states, a.reps, a.launches)


def cmd_smoke(a):
    """真模型冒煙：1 launch × 1 rep、cpu＋ssd、閒；每次還原都檢查 KV。有任何 kv_bad > 0 → exit 1。"""
    run_launches(a, ["cpu", "ssd"], ["idle"], 1, 1, verify_all=True)
    rows = list(csv.DictReader(open(os.path.join(a.out_dir, "restores.csv"))))
    bad = [r for r in rows if r["kv_bad"] not in ("", "0")]
    cfgs = sorted({r["config"] for r in rows})
    print(f"smoke：{len(rows)} 次還原、{len(cfgs)} 種配置、kv_bad>0 的 {len(bad)} 次")
    for r in rows:
        print(f"  {r['bw_name']:4s} {r['config']:13s} b={r['b']:>3s} 慢層 {r['n_slow']:>2s} 塊、從慢層載 {r['n_load_ssd']:>2s}、"
              f"從 CPU 載 {r['n_load_cpu']:>2s}、INT4 讀到 {r['fp8_read']:>2s}  t={float(r['t_restore']):.3f}s kv_bad={r['kv_bad']}")
    want = len(configs_for("cpu")) + len(configs_for("ssd"))
    if bad or len(cfgs) != want:
        raise SystemExit(f"smoke 失敗：kv_bad {len(bad)} 次、配置 {len(cfgs)}／{want}")


# ---------------------------------------------------------------------- summary（CPU）
def _open(p):
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p)


def _rows(path, only_run=None):
    rows = []
    for r in csv.DictReader(_open(path)):
        if only_run and r["run_id"] not in only_run:
            continue
        r["t_restore"] = float(r["t_restore"])
        rows.append(r)
    return rows


def compare(cell_rows, x, y, dstar):
    """X 對 Y（同一個 launch、rep 的配對）；正＝X 比較快。照主線 k0_verdict 的「贏」：(med_Y−med_X)/med_Y > δ*、CI 不含 0。"""
    from m10_speed import bootstrap_ci
    tx = {(r["launch"], r["rep"]): r["t_restore"] for r in cell_rows if r["config"] == x}
    ty = {(r["launch"], r["rep"]): r["t_restore"] for r in cell_rows if r["config"] == y}
    keys = sorted(set(tx) & set(ty))
    if len(keys) < 2:
        return None
    mx, my = statistics.median(tx[k] for k in keys), statistics.median(ty[k] for k in keys)
    diffs = [ty[k] - tx[k] for k in keys]
    lo, hi = bootstrap_ci(diffs)
    gain = (my - mx) / my
    return dict(x=x, y=y, n_pairs=len(keys), med_x=round(mx, 5), med_y=round(my, 5), gain_x_over_y=round(gain, 4),
                ci95_s=[round(lo, 5), round(hi, 5)], x_faster=bool(gain > dstar and lo > 0),
                y_faster=bool(-gain > dstar and hi < 0))


def cmd_summary(a):
    rows = _rows(a.restores, set(a.only_run) if a.only_run else None)
    model = rows[0]["model"]
    dstar = json.load(open(a.noise))["delta_star"]
    bad_cont = [r for r in rows if r["contaminated_gpu"] not in ("False",)]
    kvbad = [r for r in rows if r["kv_bad"] not in ("", "0")]
    clean = [r for r in rows if r["contaminated_gpu"] == "False" or a.include_contaminated]
    if not clean:
        raise SystemExit(f"沒有乾淨的列（{len(rows)} 列全部被 gpu_guard 標污染或沒檢查）：照 CLAUDE.md §3 時間不能用，要重量")
    cells = {}
    for r in clean:
        cells.setdefault((r["bw_name"], r["gpu_state"]), []).append(r)
    cpu = {st: v for (t, st), v in cells.items() if t == "cpu"}
    summ, comps = [], []
    for (t, st), v in sorted(cells.items()):
        for cfg in sorted({r["config"] for r in v}, key=lambda c: (configs_for(t).index(c) if c in configs_for(t) else 99)):
            x = [r for r in v if r["config"] == cfg]
            med = lambda k: statistics.median(float(r[k]) for r in x)  # noqa: E731
            o, f, p = parse_cfg(cfg)
            summ.append(dict(model=model, tier=t, slow_GiBps=x[0]["slow_GiBps"], gpu_state=st, config=cfg, order=o,
                             frac=f, prec=p, n=len(x), median_t_restore=round(med("t_restore"), 5),
                             median_ttft=round(med("ttft"), 5), median_b=med("b"), median_n_load_ssd=med("n_load_ssd"),
                             median_n_load_cpu=med("n_load_cpu"), median_int4_read=med("fp8_read"),
                             median_bytes_read=int(med("bytes_read")), median_t_dequant=round(med("t_dequant"), 5),
                             b_values=" ".join(str(b) for b in sorted({int(r["b"]) for r in x}))))
        if t not in ("ssd", "nfs"):
            continue
        for o in ORDERS:                                   # B1：INT4 對 BF16
            for f in FRACS:
                c = compare(v, f"T{o}{f}-int4", f"T{o}{f}-bf16", dstar)
                if c:
                    comps.append(dict(model=model, tier=t, gpu_state=st, test="B1", **c))
        for p in PRECS:                                    # B2：舊／新 對 隨機
            for f in FRACS:
                for o in ("old", "new"):
                    c = compare(v, f"T{o}{f}-{p}", f"Trnd{f}-{p}", dstar)
                    if c:
                        comps.append(dict(model=model, tier=t, gpu_state=st, test="B2", **c))
        e0 = {(r["launch"], r["rep"]): r for r in cpu.get(st, []) if r["config"] == "E0"}
        for cfg in TWO_TIER + SLOW_REFS:                   # B3：對全部在 CPU 的 E0（同一次重複）
            merged = [dict(r, config="E0") for r in e0.values()] + [r for r in v if r["config"] == cfg]
            c = compare(merged, cfg, "E0", dstar)
            if c:
                comps.append(dict(model=model, tier=t, gpu_state=st, test="B3", **c))
    os.makedirs(a.out_dir, exist_ok=True)
    tag = f"_{model}"
    with open(os.path.join(a.out_dir, f"summary{tag}.csv"), "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    json.dump(dict(model=model, delta_star=dstar, noise=a.noise, restores=a.restores,
                   run_ids=sorted({r["run_id"] for r in rows}), n_rows=len(rows), n_contaminated=len(bad_cont),
                   n_kv_bad=len(kvbad), rule="快＝(med_Y−med_X)/med_Y > δ*、配對差 bootstrap 95% CI 不含 0（同主線 k0_verdict）",
                   comparisons=comps), open(os.path.join(a.out_dir, f"compare{tag}.json"), "w"), indent=1, ensure_ascii=False)
    md = [f"## {model}：兩層混合還原（12 §3；32K；δ*＝{dstar:.4f}）", ""]
    if a.include_contaminated:
        md += ["**測試用：包含被標污染的列（--include-contaminated），數字不能用**", ""]
    md += [
          f"run：{'、'.join(sorted({r['run_id'] for r in rows}))}；{len(rows)} 次還原，污染 {len(bad_cont)}、KV 不對 {len(kvbad)}", ""]
    for t in ("ssd", "nfs"):
        for st in ("idle", "busy"):
            sel = [s for s in summ if s["tier"] == t and s["gpu_state"] == st]
            if not sel:
                continue
            e0c = [s for s in summ if s["tier"] == "cpu" and s["gpu_state"] == st and s["config"] == "E0"]
            ref = f"；全部在 CPU 的 E0＝{e0c[0]['median_t_restore']:.3f} s" if e0c else ""
            md += [f"### 慢層 {t}（{sel[0]['slow_GiBps']} GiB/s）・GPU {'閒' if st == 'idle' else '忙'}{ref}", "",
                   "| 配置 | 還原時間中位數 (s) | 會合點 b | 從慢層載 | 從 CPU 載 | INT4 讀到 | 反量化 (s) |",
                   "|:--|--:|:--|--:|--:|--:|--:|"]
            for s in sel:
                md.append(f"| {s['config']} | {s['median_t_restore']:.3f} | {s['b_values']} | {s['median_n_load_ssd']:g} | "
                          f"{s['median_n_load_cpu']:g} | {s['median_int4_read']:g} | {s['median_t_dequant']:.4f} |")
            md.append("")
    md += ["### 事先寫的比較（12 §3.4）", "", "| 檢定 | 慢層 | GPU | X | Y | X 對 Y 快多少 | 配對差 95% CI (s) | X 快 | Y 快 |",
           "|:--|:--|:--|:--|:--|--:|:--|:--|:--|"]
    for c in comps:
        md.append(f"| {c['test']} | {c['tier']} | {c['gpu_state']} | {c['x']} | {c['y']} | {100 * c['gain_x_over_y']:+.1f}% | "
                  f"[{c['ci95_s'][0]:+.4f}, {c['ci95_s'][1]:+.4f}] | {'是' if c['x_faster'] else ''} | {'是' if c['y_faster'] else ''} |")
    open(os.path.join(a.out_dir, f"tables{tag}.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    if a.fig:
        fig_tier2(summ, model, os.path.join(a.out_dir, "figs", f"B_{model}_tier2.png"))


def fig_tier2(summ, model, out):
    import m10_figs as F
    plt = F.plt
    tiers = [t for t in ("ssd", "nfs") if any(s["tier"] == t for s in summ)]
    states = [s for s in ("idle", "busy") if any(x["gpu_state"] == s for x in summ)]
    fig, axes = plt.subplots(len(states), len(tiers), figsize=(5.2 * len(tiers), 3.6 * len(states)), squeeze=False)
    col = {"old": F.SLOTS[0], "new": F.SLOTS[1], "rnd": F.SLOTS[2]}
    lab = {"old": "先降最舊", "new": "先降最新", "rnd": "隨機降"}
    for i, st in enumerate(states):
        for j, t in enumerate(tiers):
            ax = axes[i][j]
            sel = {s["config"]: s for s in summ if s["tier"] == t and s["gpu_state"] == st}
            for o in ORDERS:
                for p, ls in (("bf16", "--"), ("int4", "-")):
                    xs = [0] + list(FRACS) + [100]
                    e0 = [s for s in summ if s["tier"] == "cpu" and s["gpu_state"] == st and s["config"] == "E0"]
                    y0 = e0[0]["median_t_restore"] if e0 else None
                    ys = [y0] + [sel.get(f"T{o}{f}-{p}", {}).get("median_t_restore") for f in FRACS] + \
                         [sel.get(f"S-{p}", {}).get("median_t_restore")]
                    pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
                    ax.plot([x for x, _ in pts], [y for _, y in pts], ls, color=col[o], lw=2, marker="o", ms=4,
                            label=f"{lab[o]}・慢層 {'INT4' if p == 'int4' else 'BF16'}")
            v = next(iter(sel.values()), None)
            ax.set_title(f"慢層 {t}（{v['slow_GiBps'] if v else ''} GiB/s）・GPU {'閒' if st == 'idle' else '忙'}")
            ax.set_xlabel("降到慢層的 block 比例（%；0＝全部在 CPU 的 E0，100＝全部在慢層）")
            ax.set_ylabel("還原時間中位數（s）")
            ax.set_xticks([0, 25, 50, 75, 100])
            if i == 0 and j == 0:
                ax.legend(fontsize=7, ncol=2)
    fig.suptitle(f"圖 B　{model}：兩層混合還原（32K；CPU 層存 BF16；實線＝慢層存 INT4、虛線＝慢層存 BF16）",
                 color=F.INK, fontsize=10)
    F.source_note(fig, "12_tier_plan §3；results/m10_tier_mi300x")
    F.save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("_worker")
    w.add_argument("spec")
    for name in ("grid", "smoke"):
        p = sub.add_parser(name)
        p.add_argument("--model", choices=["llama", "qwen"], required=True)
        p.add_argument("--out-dir", required=True)
        p.add_argument("--length", type=int, default=32768)
        p.add_argument("--new-tokens", type=int, default=256)
        p.add_argument("--gpu-states", nargs="+", default=["idle", "busy"])
        p.add_argument("--busy-levels", default=os.environ.get("M10_BUSY_LEVELS") or os.path.join(HERE, "..", "results", "m7_write_policy_mi300x",   # （H200：環境變數，沒設時不變；2026-10-17）
                                                             "busy_levels.json"))
        p.add_argument("--busy-key", default="busy2")
        p.add_argument("--launches", type=int, default=2)
        p.add_argument("--reps", type=int, default=5)
        p.add_argument("--blocks-out")
    s = sub.add_parser("summary")
    s.add_argument("--restores", required=True)
    s.add_argument("--noise", required=True, help="主線 speed_<m>/noise.json（δ*）")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--only-run", nargs="*")
    s.add_argument("--fig", action="store_true")
    s.add_argument("--include-contaminated", action="store_true", help="只給測試用（假模型和其他工作同時跑）；正式報告不能用")
    a = ap.parse_args()
    if a.cmd == "_worker":
        worker(json.loads(a.spec))
        return
    {"grid": cmd_grid, "smoke": cmd_smoke, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
