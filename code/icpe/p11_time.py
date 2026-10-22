"""p11_time.py — ICPE P1-1 (T)（docs/h200/icpe/P1-1_xkv_int4_h200.md §3）：H200 上量「全部載入」的還原時間與 xKV 的壓縮時間（GPU，量時間）。

  python code/icpe/p11_time.py loadpath --model llama --out results/icpe_h200/p11/time [--reps 5] [--smoke]
  python code/icpe/p11_time.py wcost    --model llama --out results/icpe_h200/p11/time [--reps 3]

loadpath：m16_xtime.gpu_loadpath（MI300X X3 validate 的同一個函式：限速器＋真實 H2D、從最後一塊往前、最後一次拆＋重建），
  段＝int4_kivi 與這個模型 P1-1 的 xKV＋INT4（每題 SVD 版搬基底、:cal 版不搬），層＝b32（H200 這個模型 bw.json 的 b32）、
  nfs（tier_params 的 0.331084 GiB/s）、cpu（真實 pinned，只報告），duty 0（閒）與 1.0（H200 的 busy2）。→ loadpath.csv
wcost：每題 SVD（m16_xtime.build_wcost：逆 RoPE、svd_lowrank、取係數、INT4 量化）與 :cal 的投影（Z·U＋INT4 量化），32K 區域，
  閒／忙。→ wcost.csv
H200 移植（只在這支裡設；沒設時 m16_xtime 照舊）：M16_XKV_EXTRA（新段）、M10_SWEEP_DIR／M10_PRECISION_DIR（K*、格式選擇）、
  busy 等級與格式選擇的檔改讀 H200 的。每列 run_id、ts、contaminated。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
REPO = os.path.dirname(CODE)
sys.path.insert(0, CODE)
RANKS = {"qwen": [(34, 51)], "llama": [(29, 44), (19, 29)]}
os.environ["M16_XKV_EXTRA"] = ",".join(f"k{a}v{b}_int4{c}" for m in RANKS for a, b in RANKS[m] for c in ("", "_cal"))
os.environ.setdefault("M10_SWEEP_DIR", os.path.join(REPO, "results", "m10_sweep_h200"))
os.environ.setdefault("M10_PRECISION_DIR", os.path.join(REPO, "results", "m10_precision_h200"))
import m16_xtime as XT  # noqa: E402

XT.DEFAULT_PATHS["fmt_choice"] = os.path.join(REPO, "results", "m10_precision_h200", "fmt_select", "fmt_choice.json")
XT.BUSY_LEVELS = os.path.join(REPO, "results", "m10_restore2_h200", "busy_levels_q4.json")
TIER_PARAMS = os.environ.get("M7_TIER_PARAMS") or os.path.join(REPO, "results", "m7_write_policy_mi300x", "tier_params.json")
DUTIES = (0.0, 1.0)
LP_FIELDS = ["run_id", "ts", "model", "tier", "GiBps", "duty", "seg", "cal", "dec", "meas_s", "n_reps", "q1_s", "q3_s",
             "bytes_total", "bytes_lb_s", "contaminated", "note"]
WC_FIELDS = ["run_id", "ts", "model", "duty", "pair", "op", "reps", "t_med_s", "t_q1_s", "t_q3_s", "t_min_s", "t_max_s", "T",
             "contaminated", "note"]


def run_id():
    return os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc"


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def segs_of(m):
    return ["int4_kivi"] + [f"xkv_k{a}v{b}_int4{c}" for a, b in RANKS[m] for c in ("", "_cal")]


def tiers_of(m):
    bw = json.load(open(os.path.join(REPO, "results", "m10_precision_h200", f"speed_{m}", "bw.json")))
    nfs = float(json.load(open(TIER_PARAMS))["nfs"]["read_GiBps"])
    return [("b32", float(bw["targets"]["b32"]["GiBps"])), ("nfs", nfs), ("cpu", None)]


def seg_bytes(m, seg):
    L, H, D = XT.SHAPES[m]
    bf_blk = L * 2 * H * XT.CHUNK * D * 2
    if seg == "int4_kivi":
        return int(XT.N_CTX * bf_blk / 3.2)
    a, b, f = XT.XKV2[seg]
    cb, basis = XT.xkv_bytes(m, a, b, f, XT.CHUNK)
    return XT.N_CTX * cb + (0 if seg in XT.XKV2_CAL else basis)


def writer(path, fields):
    new = not os.path.exists(path)
    fh = open(path, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    if new:
        w.writeheader()
    return fh, w


def cmd_loadpath(a):
    os.makedirs(a.out, exist_ok=True)
    items = [dict(model=a.model, tier=t, GiBps=g, duty=d, seg=s, dec="oneshot", cal=s in XT.XKV2_CAL)
             for d in DUTIES for t, g in tiers_of(a.model) for s in segs_of(a.model)]
    res = XT.gpu_loadpath(a.model, items, a.reps, a.out, smoke=a.smoke)
    fh, w = writer(os.path.join(a.out, "loadpath_smoke.csv" if a.smoke else "loadpath.csv"), LP_FIELDS)
    for r in res:
        nb = seg_bytes(a.model, r["seg"])
        q1, q3 = r["q"] if isinstance(r.get("q"), (list, tuple)) else (None, None)
        w.writerow(dict(r, run_id=run_id(), ts=ts(), q1_s=q1, q3_s=q3, bytes_total=nb,
                        bytes_lb_s=(nb / (r["GiBps"] * XT.GiB)) if r["GiBps"] else None))
        print(json.dumps({k: r[k] for k in ("tier", "duty", "seg", "meas_s", "contaminated")}, ensure_ascii=False, default=str), flush=True)
    fh.close()
    if any(r.get("contaminated") is True for r in res):
        print("[p11_time] 有污染的格（不進判定）", flush=True)
        sys.exit(3)


def cmd_wcost(a):
    import torch
    import m10_codec_x as X
    from m10_speed import Busy, _watcher
    theta, _ = XT.check_config(a.model)
    L, H, D = XT.SHAPES[a.model]
    kv = torch.zeros((L, 2, H, XT.N_CTX * XT.CHUNK + XT.CHUNK, D), dtype=torch.bfloat16, device="cuda:0")
    ctx = XT.GCtx(torch, a.model, kv, XT.N_CTX, theta, "triton")
    pairs = RANKS[a.model]
    ops, info, state = XT.build_wcost(ctx, pairs, {p: {"kivi4"} for p in pairs})
    # :cal 的投影：Z（同 build_wcost 的 Z）× 事先算好的基底 U[:, :r]（隨機正交；時間和值無關）＋係數 INT4 量化
    W, T, Lg = XT.XKV_W, ctx.T, L // XT.XKV_W
    src = ctx.randn(L, 2, H, T, D, scale=2.0)
    Ucal = {hi: [torch.linalg.qr(torch.randn(W * H * D, 64 * W * H, device="cuda:0"))[0] for _ in range(Lg)] for hi in (0, 1)}

    def cal_proj(hi, rh):
        r = rh * W * H
        As = []
        for w_ in range(Lg):
            Z = torch.cat([src[l, hi].float().permute(1, 0, 2).reshape(T, H * D) for l in range(w_ * W, (w_ + 1) * W)], dim=1)
            As.append(Z @ Ucal[hi][w_][:, :r])
        A4 = torch.stack(As).unsqueeze(1)
        return X.enc_half(A4.contiguous(), "kivi4", "k")
    for rk, rv in pairs:
        key = f"k{rk}v{rv}"
        ops.append((key, "cal_proj_k_int4", lambda rk=rk: cal_proj(0, rk)))
        ops.append((key, "cal_proj_v_int4", lambda rv=rv: cal_proj(1, rv)))
    msgs = XT.wcost_check(ctx, state, pairs, {p: {"kivi4"} for p in pairs}) if state else []
    rid = run_id()
    os.makedirs(a.out, exist_ok=True)
    watch, level = _watcher(os.path.join(a.out, f"gpu_guard_{rid}_wcost_{a.model}.json"))
    busy_pids, rows = set(), []
    rng = random.Random(f"p11|{a.model}")
    for i, d in enumerate(DUTIES):
        with Busy(d, watch) as b:
            if b.p:
                busy_pids.add(b.p.pid)
            res = XT._time_slow(torch, ops, a.reps, rng, warm=(i == 0))
            bad = {p for p in (watch.intruders if watch else {}) if p not in busy_pids}
            cflag = (bool(bad) or watch.started_clean is False) if watch else "NOT_CHECKED"
        for (k, o), v in res.items():
            q1, q3 = XT.quart(v)
            rows.append(dict(run_id=rid, ts=ts(), model=a.model, duty=d, pair=k, op=o, reps=len(v), t_med_s=XT.med(v), t_q1_s=q1,
                             t_q3_s=q3, t_min_s=min(v), t_max_s=max(v), T=T, contaminated=cflag))
        print(f"[wcost] {a.model} duty {d} 完成", flush=True)
    if watch:
        watch.__exit__(None, None, None)
    msgs += XT.wcost_check(ctx, state, pairs, {p: {"kivi4"} for p in pairs})
    if msgs:
        raise SystemExit(f"wcost 的輸出核對沒過：{msgs}（規則 2：停）")
    fh, w = writer(os.path.join(a.out, "wcost.csv"), WC_FIELDS)
    for r in rows:
        w.writerow(r)
    fh.close()
    if any(r["contaminated"] is True for r in rows):
        sys.exit(3)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    lp = sp.add_parser("loadpath")
    lp.add_argument("--model", choices=list(RANKS), required=True)
    lp.add_argument("--out", required=True)
    lp.add_argument("--reps", type=int, default=5)
    lp.add_argument("--smoke", action="store_true")
    wc = sp.add_parser("wcost")
    wc.add_argument("--model", choices=list(RANKS), required=True)
    wc.add_argument("--out", required=True)
    wc.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    {"loadpath": cmd_loadpath, "wcost": cmd_wcost}[a.cmd](a)


if __name__ == "__main__":
    main()
