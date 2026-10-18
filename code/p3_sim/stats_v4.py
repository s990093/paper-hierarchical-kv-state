"""stats_v4.py — 04 v4 要用的資料統計（06 §15）。**只算資料本身的統計，不算任何策略之間的比較（不算「省多少」）。**

  gaps      回來的間隔：每個有父請求的請求，子的到達 − 父的到達（秒）；n、p50／p90／p95／p99、max、>1,800 s、>3,600 s 的比例、切片長度。
  sessions  每個切片的 session 數、≥2 個請求的 session 數（以 session 為單位 bootstrap 夠不夠）。
  kgpu      E6 的 GPU 數 k：session 依 sha1(session_id) mod k 分到 k 張 GPU，每張各跑一次 block 引擎（C0＋E2＋MI300X 常數），
            GPU 閒、GPU 忙兩種，記每張 GPU 的 fifo1 忙碌比例：(i) 整個切片、(ii) 每 1 小時窗口的最大值。列出四種定義下 ≤0.8 的最小 k（不選）。

用法（記錄殼）：
  TIARA_NO_GPU=1 bash code/run3090.sh p3sim-gaps python -I code/p3_sim/stats_v4.py gaps
  TIARA_NO_GPU=1 bash code/run3090.sh p3sim-kgpu python -I code/p3_sim/stats_v4.py kgpu --procs 8 --cap 32
輸出：$RUN_DIR/*.csv（每列有 run_id、ts），另抄一份到 results/p3_sim/（--results）。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import sys
import time
from datetime import datetime
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "p3_data"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RUN_DIR = os.environ.get("RUN_DIR") or "/ssd8/hungwei/paper-hkv/opencode/scratch/stats_adhoc"
RESULTS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "results", "p3_sim")
FI_STAGE_A = ["week01", "week05", "week09", "week13"]
FI_ALL = [f"week{i:02d}" for i in range(1, 17)]
# 地圖用的切片（04 §5、§10）：整份資料（沒有切窗）
SLICES = [("bailian", "coder"), ("bailian", "traceA"), ("wildchat", "1m"), ("mooncake", "conv-rule-strict")] + \
         [("freeinference", w) for w in FI_ALL] + [("synth", "default-seed0"), ("synth", "small-seed1")]


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def rid():
    return os.environ.get("RUN_ID", "adhoc")


def write(name, rows, results=False):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    paths = [os.path.join(RUN_DIR, name)] + ([os.path.join(RESULTS, name)] if results else [])
    for p in paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["run_id", "ts"] + keys)
            w.writeheader()
            for r in rows:
                w.writerow(dict(run_id=rid(), ts=now(), **r))


# ================================================================== gaps／sessions
def _load(source, variant):
    import schema
    df = schema.read_unified(source, variant, ["session_id", "turn_idx", "parent_turn_idx", "t_start_ms", "obs_end_ms"])
    return df


def _gaps(df):
    """子 t_start − 父 t_start（秒）；父＝同 session、turn_idx＝parent_turn_idx 的那一列（規則 6：父一定找得到，找不到就丟錯）。"""
    key = df.set_index(["session_id", "turn_idx"])["t_start_ms"]
    assert key.index.is_unique, "session_id＋turn_idx 不唯一"
    c = df[df["parent_turn_idx"].notna()]
    pt = key.reindex(pd.MultiIndex.from_arrays([c["session_id"].values, c["parent_turn_idx"].astype(np.int64).values])).values
    if np.isnan(pt.astype(float)).any():
        raise AssertionError(f"有 {int(np.isnan(pt.astype(float)).sum())} 個父請求找不到")
    return (c["t_start_ms"].values.astype(np.int64) - pt.astype(np.int64)) / 1000.0


def _window_s(df):
    return (int(df["obs_end_ms"].max()) - int(df["t_start_ms"].min())) / 1000.0


def gap_row(name, g, win_s, n_req):
    g = np.asarray(g, float)
    r = dict(slice=name, n_req=n_req, window_s=win_s, n_gaps=int(len(g)), n_negative=int((g < 0).sum()))
    if len(g):
        q = np.percentile(g, [50, 90, 95, 99])
        r.update(p50_s=float(q[0]), p90_s=float(q[1]), p95_s=float(q[2]), p99_s=float(q[3]), max_s=float(g.max()),
                 frac_gt_1800s=float((g > 1800).mean()), frac_gt_3600s=float((g > 3600).mean()),
                 H_v4_s=max(1800.0, float(q[3])))          # 04 v4 §5 的截尾 H＝max(1,800 s, 回來間隔 p99)〔算術〕
    else:   # 0/0：沒有回來的請求 → 分位數與比例寫 NaN
        r.update(p50_s=float("nan"), p90_s=float("nan"), p95_s=float("nan"), p99_s=float("nan"), max_s=float("nan"),
                 frac_gt_1800s=float("nan"), frac_gt_3600s=float("nan"))
    return r


def sess_row(name, df):
    n = df.groupby("session_id").size().values
    p = n / n.sum()
    return dict(slice=name, n_req=int(n.sum()), n_sessions=int(len(n)), n_sessions_ge2=int((n >= 2).sum()),
                frac_req_in_ge2_sessions=float(n[n >= 2].sum() / n.sum()), max_req_per_session=int(n.max()),
                top1pct_sessions_req_share=float(np.sort(n)[::-1][:max(1, len(n) // 100)].sum() / n.sum()),
                effective_n_sessions_by_req=float(1.0 / np.sum(p * p)))


def cmd_gaps(a):
    gr, sr = [], []
    fi_g, fi_win, fi_n, fi_df = {}, {}, {}, []
    for src, var in SLICES:
        t0 = time.time()
        df = _load(src, var)
        g = _gaps(df)
        name = f"{src}/{var}"
        if src == "freeinference":
            fi_g[var], fi_win[var], fi_n[var] = g, _window_s(df), len(df)
            fi_df.append(df[["session_id"]].assign(week=var))
        gr.append(gap_row(name, g, _window_s(df), len(df)))
        sr.append(sess_row(name, df))
        print(f"{name}: n={len(df)} ({time.time() - t0:.0f}s)", flush=True)
    for tag, weeks in (("freeinference/weeks01+05+09+13", FI_STAGE_A), ("freeinference/all16", FI_ALL)):
        g = np.concatenate([fi_g[w] for w in weeks])
        gr.append(gap_row(tag, g, float(sum(fi_win[w] for w in weeks)), int(sum(fi_n[w] for w in weeks))))
        d = pd.concat([x for x in fi_df if x["week"].iloc[0] in weeks], ignore_index=True)
        sr.append(sess_row(tag, d))
        # session_id 跨週會不會重複（重複的話 pooled 的 session 數會少算；照實記）
        sr[-1]["n_session_ids_in_more_than_one_week"] = int((d.drop_duplicates().groupby("session_id").size() > 1).sum())
    write("return_gaps.csv", gr, a.results)
    write("session_counts.csv", sr, a.results)
    for r in gr:
        print(r, flush=True)
    for r in sr:
        print(r, flush=True)


# ================================================================== kgpu
def _sub_P(P, idx):
    """P 的子集（idx 遞增）：父請求一定在同一個 session → 同一張 GPU。"""
    inv = {int(r): i for i, r in enumerate(idx)}
    par = []
    for r in idx:
        p = P["parentl"][r]
        if p >= 0:
            assert p in inv, "父請求不在同一張 GPU（session 被拆開了）"
            par.append(inv[p])
        else:
            par.append(-1)
    return dict(full=[P["full"][r] for r in idx], nall=[P["nall"][r] for r in idx], tl=[P["tl"][r] for r in idx],
                kindc=[P["kindc"][r] for r in idx], parentl=par, sessl=[P["sessl"][r] for r in idx],
                obs_end=[P["obs_end"][r] for r in idx], win=P["win"], meta=P["meta"], R=P["R"])


def _busy_stats(tgo, tend, lo, hi):
    """每張 GPU：忙碌區間 [t_go, T)（T 含 capture）；整個切片＝和 [lo, hi) 的重疊 ÷ (hi − lo)；
    每 1 小時窗口＝完整的小時 [lo+3600i, lo+3600(i+1)) 的重疊 ÷ 3600（切片不到 1 小時 → 用整個切片當一個窗口）。"""
    L = hi - lo
    a = np.clip(np.asarray(tgo, float), lo, hi)
    b = np.clip(np.asarray(tend, float), lo, hi)
    whole = float(np.sum(b - a)) / L if L > 0 else float("nan")
    nh = int(L // 3600)
    if nh == 0:
        return whole, whole, 0
    edges = lo + 3600.0 * np.arange(nh + 1)
    # 每個區間對每個小時的重疊：用累積忙碌時間 B(t)＝Σ clamp(t − a, 0, b − a)，在整點上求值
    order_a, order_b = np.sort(a), np.sort(b)
    ca, cb = np.concatenate([[0.0], np.cumsum(order_a)]), np.concatenate([[0.0], np.cumsum(order_b)])

    def B(t):
        ka = np.searchsorted(order_a, t, side="right")
        kb = np.searchsorted(order_b, t, side="right")
        return (ka * t - ca[ka]) - (kb * t - cb[kb])
    Bt = np.array([B(t) for t in edges])
    per_h = np.diff(Bt) / 3600.0
    return whole, float(per_h.max()), nh


def kgpu_job(args):
    src, var, cap, kmax, thr, fmt = args
    import sim
    import trace as TR
    from engine_block import BView, BlockEngine, reqs_from_P
    from formats import load_costs, load_format_table
    from policies import make_policy
    t0 = time.time()
    P = TR.prepare_cached(src, var)
    tp = time.time() - t0
    lo, hi = P["win"]
    h = np.array([int(hashlib.sha1(str(s).encode("utf-8")).hexdigest(), 16) for s in P["sess_key"]], dtype=object)
    sl = np.asarray(P["sessl"])
    mf = load_format_table("llama", "mi300x", fmt)
    costs = load_costs("mi300x", "llama", "mi300x_local")
    oc = sim.outer_cfg("E2")
    rows = []
    defs = [(gpu, d) for gpu in ("idle", "busy") for d in ("whole", "hourmax")]

    def run_groups(greq, ng):
        """每一組（＝一張 GPU）各跑一次 block 引擎；回傳 {gpu: [(組, 請求數, 整個切片, 小時最大, 完整小時數)]}。"""
        per = {"idle": [], "busy": []}
        for g in range(ng):
            idx = np.nonzero(greq == g)[0]
            if len(idx) == 0:
                for gpu in per:
                    per[gpu].append((g, 0, 0.0, 0.0, 0))
                continue
            Pg = _sub_P(P, idx)
            reqs = reqs_from_P(Pg)
            nmax = max(max(len(q["h"]) for q in reqs), max(Pg["nall"])) + 1
            for gpu in ("idle", "busy"):
                v = BView(costs, mf, gpu, nmax, d2h_Bps=costs.cpu["w"] * (1 << 30), label=f"mi300x|gpu={gpu}|fifo1")
                BlockEngine(reqs, oc, make_policy("C0", mf), int(cap * (1 << 30)), v, mf, qp=False).run_events()
                wh, mh, nh = _busy_stats(v.tgo, v.tend, lo, hi)
                per[gpu].append((g, len(idx), wh, mh, nh))
        return per

    def put(k, per, brief=False):
        out = {}
        for gpu in ("idle", "busy"):
            wmax = max(x[2] for x in per[gpu])
            hmax = max(x[3] for x in per[gpu])
            out[(gpu, "whole")], out[(gpu, "hourmax")] = wmax, hmax
            rows.append(dict(slice=f"{src}/{var}", cap_GiB=cap, policy="C0", outer="E2", constants="mi300x", ssd="mi300x_local",
                             format_table_sha256=mf.src["sha256"], unified_output_sha256=P["meta"].get("unified_output_sha256"),
                             gpu=gpu, k=k, n_req=len(sl), window_s=hi - lo, n_full_hours=per[gpu][0][4],
                             whole_busy_max_over_gpus=wmax, whole_busy_mean_over_gpus=float(np.mean([x[2] for x in per[gpu]])),
                             hour_busy_max_over_gpus_and_hours=hmax,
                             per_gpu_whole="" if brief else ";".join(f"{x[2]:.4f}" for x in per[gpu]),
                             per_gpu_hourmax="" if brief else ";".join(f"{x[3]:.4f}" for x in per[gpu]),
                             per_gpu_nreq="" if brief else ";".join(str(x[1]) for x in per[gpu]), prepare_s=tp, elapsed_s=time.time() - t0))
        return out

    # k＝1 先跑；四種定義都過就停。沒過的話先算「每個 session 單獨一張 GPU」（k→∞，hash 分配能到的極限）：
    # session 不能拆，所以一個 session 單獨跑都 > thr 的定義，任何 k 都過不了 → 標 unreachable，不再為它往上加 k。
    # （單獨一張 GPU 時 CPU 池只有它自己用、沒有別人排在前面，所以是近似的下界；照實標「近似」。）
    first = put(1, run_groups(np.zeros(len(sl), np.int64), 1))
    need = {d for d in defs if first[d] > thr}
    reach = set(need)
    if need:
        iso = put("isolated", run_groups(sl.astype(np.int64), int(P["R"]["n_sess"])), brief=True)
        reach = {d for d in need if iso[d] <= thr}
    for k in range(2, kmax + 1):
        if not reach:
            break
        gsess = np.array([int(x % k) for x in h])
        cur = put(k, run_groups(gsess[sl], k))
        reach = {d for d in reach if cur[d] > thr}
    return rows


def cmd_kgpu(a):
    sl = [s for s in SLICES if (not a.only or f"{s[0]}/{s[1]}" in a.only)]
    jobs = [(src, var, cap, a.kmax, a.thr, a.fmt) for cap in a.cap for (src, var) in sl]
    jobs.sort(key=lambda j: 0 if j[0] == "freeinference" else 1)       # 慢的先排
    rows = []
    t0 = time.time()
    with Pool(min(a.procs, len(jobs))) as pool:
        for r in pool.imap_unordered(kgpu_job, jobs):
            rows += r
            print(f"[{time.time() - t0:.0f}s] {r[0]['slice']} cap={r[0]['cap_GiB']} k≤{max(x['k'] for x in r if x['k'] != 'isolated')}", flush=True)
            write(f"kgpu_busy{a.tag}.csv", rows, False)
    rows.sort(key=lambda x: (x["slice"], x["cap_GiB"], x["gpu"], 10 ** 9 if x["k"] == "isolated" else x["k"]))
    write(f"kgpu_busy{a.tag}.csv", rows, a.results)
    # 四種定義下 ≤ thr 的最小 k（不選；v4 決定）
    summ = []
    df = pd.DataFrame(rows)
    for (s, cap), g in df.groupby(["slice", "cap_GiB"]):
        gk = g[g.k != "isolated"].copy()
        gk["k"] = gk.k.astype(int)
        gi = g[g.k == "isolated"]
        r = dict(slice=s, cap_GiB=cap, thr=a.thr, kmax_tried=int(gk.k.max()))
        for gpu in ("idle", "busy"):
            gg = gk[gk.gpu == gpu].sort_values("k")
            ii = gi[gi.gpu == gpu]
            for d, col in (("whole", "whole_busy_max_over_gpus"), ("hourmax", "hour_busy_max_over_gpus_and_hours")):
                ok = gg[gg[col] <= a.thr]
                if len(ok):
                    r[f"k_min_{gpu}_{d}"] = int(ok.k.iloc[0])
                elif len(ii) and float(ii[col].iloc[0]) > a.thr:
                    r[f"k_min_{gpu}_{d}"] = "unreachable(isolated>thr)"
                else:
                    r[f"k_min_{gpu}_{d}"] = f">{int(gg.k.max())}"
                r[f"isolated_{gpu}_{d}"] = float(ii[col].iloc[0]) if len(ii) else float("nan")
        summ.append(r)
    write(f"kgpu_kmin{a.tag}.csv", summ, a.results)
    for r in summ:
        print(r, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gaps", "kgpu"])
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--cap", type=float, nargs="*", default=[32.0])
    ap.add_argument("--kmax", type=int, default=16)
    ap.add_argument("--thr", type=float, default=0.8)
    ap.add_argument("--fmt", default="official")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--results", action="store_true", help="另抄一份到 results/p3_sim/")
    ap.add_argument("--tag", default="", help="kgpu 輸出檔名的後綴（例如 _mem63_170_435）")
    a = ap.parse_args()
    os.makedirs(RUN_DIR, exist_ok=True)
    {"gaps": cmd_gaps, "kgpu": cmd_kgpu}[a.cmd](a)


if __name__ == "__main__":
    main()
