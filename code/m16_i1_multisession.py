#!/usr/bin/env python3
"""m16_i1_multisession.py — 第三階段 I1 的 GPU 實測：(a) 計算與搬運互相干擾多少；(b) N 個 session 同時還原，實測 vs 「互不干擾」的預測。

計畫：docs/stage3_20261019/experiment_matrix.md 的 I1；事先預測：code/m16_i1_predict.py（run 20261019-134046-m16-i1pred）。
**要 GPU**，只在第三階段窗口跑（runner 會設 HIP_VISIBLE_DEVICES／TIARA_GPU）。

  python code/m16_i1_multisession.py interf --model llama --seconds 6 --gpu-states idle busy2 --out <dir>
  python code/m16_i1_multisession.py grid   --model llama --N 1 4 8 --workloads homog_cpu homog_b32 mix_cpu_b32 \
         --modes cake g5b joint --orders rr fcfs srpt --reps 3 --out <dir>

## (a) interf：前作（CacheFlow 附錄 A.1、InferCept p.6、FSL-26-02 p.22）假設計算與搬運互不干擾
  solo_c   只有計算線：一直重算第 0..63 塊（Llama 32K 的 f(i)）
  solo_l   只有載入線：一直把 64 MiB 的 BF16 chunk 從 pinned 記憶體 H2D 到 GPU（真實 CPU 層，不限速）
  both     兩條同時跑（不同 stream、不同 thread）
  → k_rec＝both 的每塊重算時間 ÷ solo_c（同一塊位置配對）；k_load＝both 的每次 H2D ÷ solo_l。GPU 閒、busy2 各一組。
  I2（併在這裡）：solo_d＝只有 decode（32K context 之後一次 1 個 token，batch＝1），both_d＝decode＋載入線同時跑
  → k_dec＝both_d 的每步時間 ÷ solo_d（背景還原／預取會不會拖慢正在生成的請求）；同時記載入線被 decode 拖慢多少。
  限制：KVModel 只有單一序列，decode 是 batch＝1（真實 serving 的 batch decode 沒量）。
## (b) grid：用 m16_mcake.Sched 在 GPU 上真的跑（一條計算線＝一條 compute stream；每層一條載入線＝各自的 I/O stream）
  每格：每個 session 的還原完成時間、TTFT（＋新一輪 256 token）；第一次逐位元比對每個 session 的 KV。
  同一個 process 裡先量 solo 的 f(i) 與每塊載入時間，再用 m16_mcake.simulate 算「互不干擾」的預測，寫在同一列（pred_*），
  之後分析只看「實測 ÷ 預測」。
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import os
import statistics
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m16_mcake import Sched, Sess, apply_mode, simulate, stats  # noqa: E402

GiB = 1 << 30
NEW_TOKENS = 256


def now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat()


class Env:
    """一個 process 一份：模型、合成 token、prefill 後的參考 KV、BF16 pinned 主機副本（照 m10_speed.Bench）。"""

    def __init__(self, model: str, n_max: int):
        import torch
        from m10_speed import Bench
        self.torch = torch
        self.b = Bench(model, n_max, "bf16", NEW_TOKENS)       # 建 KVModel＋Restorer（Restorer 不用，只是 Bench 會建一個）
        self.b.capture(n_max)
        self.km, self.ids, self.n_max = self.b.km, self.b.ids, n_max
        self.CHUNK = self.b.CHUNK
        self.host = self.b.bf                                  # list of pinned [L,2,H,512,D]
        self.cstream = torch.cuda.Stream()
        self.clones: list = []

    def clones_for(self, N: int):
        """N 個 session 各自的 KV 緩衝（共用權重）。forward_span 只用 self.kv／hf／_masks／max_len，淺拷貝後換 kv 即可。"""
        torch = self.torch
        while len(self.clones) < N:
            c = copy.copy(self.km)
            c.kv = torch.empty_like(self.km.kv)
            c._masks = {}
            c.key_keep = None
            self.clones.append(c)
        return self.clones[:N]

    def recompute(self, km, i: int):
        with self.torch.cuda.stream(self.cstream):
            km.forward_span(self.ids[i * self.CHUNK:(i + 1) * self.CHUNK], i * self.CHUNK)
        self.cstream.synchronize()

    def new_turn(self, km, n: int):
        with self.torch.cuda.stream(self.cstream):
            km.forward_span(self.ids[n * self.CHUNK:n * self.CHUNK + NEW_TOKENS], n * self.CHUNK, want_logits=True)
        self.cstream.synchronize()


def sleep_until(t_abs: float):
    while True:
        d = t_abs - time.perf_counter()
        if d <= 0:
            return
        time.sleep(d if d > 0.002 else 0.0002)


# ---------------------------------------------------------------- (a) 干擾
def cmd_interf(a):
    E = Env(a.model, 64)
    torch = E.torch
    from m10_speed import Busy, _watcher
    run_id = os.environ.get("RUN_ID", "NOT_SET")
    busy_lv = json.load(open(os.path.join(os.path.dirname(HERE), "results/m7_write_policy_mi300x/busy_levels.json")))
    km = E.clones_for(2)[0]
    km.kv.copy_(E.km.kv)                                       # 重算第 i 塊需要 0..i−1：用參考 KV 當前綴
    dst = E.clones_for(2)[1]
    iost = torch.cuda.Stream()
    rows = []
    w, level = _watcher(os.path.join(a.out, f"gpu_guard_{run_id}_interf.json"))

    def c_loop(stop, out):
        i = 0
        while not stop.is_set():
            t0 = time.perf_counter()
            E.recompute(km, i)
            out.append((i, time.perf_counter() - t0))
            i = (i + 1) % 64

    def d_loop(stop, out):
        pos = 64 * E.CHUNK                                    # 32K context 之後同一個位置重複 decode（KV 寫同一格，不累積）
        tok = E.ids[pos:pos + 1]
        while not stop.is_set():
            t0 = time.perf_counter()
            with torch.cuda.stream(E.cstream):
                km.forward_span(tok, pos, want_logits=True)
            E.cstream.synchronize()
            out.append((0, time.perf_counter() - t0))

    def l_loop(stop, out):
        j = 0
        while not stop.is_set():
            t0 = time.perf_counter()
            with torch.cuda.stream(iost):
                dst.chunk_view(j).copy_(E.host[j], non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(iost)
            ev.synchronize()
            out.append((j, time.perf_counter() - t0))
            j = (j + 1) % 64

    for st in a.gpu_states:
        duty = 0.0 if st == "idle" else float(busy_lv[st])
        with Busy(duty, watch=w):
            for case in ("solo_c", "solo_l", "both", "solo_d", "both_d", "solo_c2", "solo_l2", "solo_d2"):  # 前後各一次 solo，看漂移
                oc, ol = [], []
                stop = threading.Event()
                th = []
                if case in ("solo_c", "both", "solo_c2"):
                    th.append(threading.Thread(target=c_loop, args=(stop, oc)))
                if case in ("solo_d", "both_d", "solo_d2"):
                    th.append(threading.Thread(target=d_loop, args=(stop, oc)))
                if case in ("solo_l", "both", "solo_l2", "both_d"):
                    th.append(threading.Thread(target=l_loop, args=(stop, ol)))
                for t in th:
                    t.start()
                time.sleep(a.seconds)
                stop.set()
                for t in th:
                    t.join()
                for kind, arr in (("dec" if case.endswith("_d") or case == "solo_d2" else "rec", oc), ("h2d", ol)):
                    for idx, d in arr[1:]:                    # 丟第一個（暖機）
                        rows.append(dict(run_id=run_id, ts=now_iso(), gpu_state=st, case=case, kind=kind, chunk=idx,
                                         seconds=round(d, 7)))
                print(st, case, "rec", len(oc), "h2d", len(ol), flush=True)
    contaminated = bool(w and w.contaminated)
    if w:
        w.__exit__(None, None, None)
    path = os.path.join(a.out, f"interf_{a.model}.csv")
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) + ["contaminated", "host_level"])
        wr.writeheader()
        for r in rows:
            wr.writerow(dict(r, contaminated=contaminated, host_level=level))
    # 摘要：同一位置配對的中位數比
    summ = {}
    for st in a.gpu_states:
        for kind, solo, both in (("rec", "solo_c", "both"), ("h2d", "solo_l", "both"), ("dec", "solo_d", "both_d"),
                                 ("h2d", "solo_l", "both_d")):
            def med_by(case):
                d = {}
                for r in rows:
                    if r["gpu_state"] == st and r["case"] == case and r["kind"] == kind:
                        d.setdefault(r["chunk"], []).append(r["seconds"])
                return {k: statistics.median(v) for k, v in d.items()}
            s1, s2, b = med_by(solo), med_by(solo + "2"), med_by(both)
            ks = sorted(set(s1) & set(b))
            ratios = [b[k] / s1[k] for k in ks]
            drift = [s2[k] / s1[k] for k in sorted(set(s1) & set(s2))]
            summ[f"{st}|{kind}|{both}"] = dict(k_median=round(statistics.median(ratios), 4) if ratios else None,
                                        k_q1q3=[round(x, 4) for x in statistics.quantiles(ratios, n=4)[::2]] if len(ratios) > 3 else None,
                                        solo_drift_median=round(statistics.median(drift), 4) if drift else None,
                                        n_pos=len(ks))
    json.dump(dict(run_id=run_id, contaminated=contaminated, host_level=level, seconds=a.seconds, summary=summ),
              open(path.replace(".csv", ".json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
    if contaminated:
        open(os.path.join(a.out, "CONTAMINATED"), "w").write("gpu_guard\n")


# ---------------------------------------------------------------- (b) 多 session
WORKLOADS = {
    "homog_cpu": lambda N: [("cpu", 64)] * N,
    "homog_b32": lambda N: [("b32", 64)] * N,
    "mix_cpu_b32": lambda N: [("cpu" if k % 2 == 0 else "b32", 64) for k in range(N)],
    "mix_len": lambda N: [("cpu" if k % 2 == 0 else "b32", (16, 32, 64)[k % 3]) for k in range(N)],
}


def run_cell(E, tiers_bps: dict, spec: list, mode: str, order: str, f_est, tl_est, check: bool):
    """真的在 GPU 上跑一格；回傳每個 session 的結果與事件 log。"""
    torch = E.torch
    from m7_restore_harness import Tier
    N = len(spec)
    clones = E.clones_for(N)
    for c in clones:
        c.kv.zero_()
    torch.cuda.synchronize()
    S = apply_mode([Sess(n=n, tier=t) for t, n in spec], mode, f_est, tl_est)
    sc = Sched(S, "fcfs" if order == "split" else order, new_turn=True)
    tiers = sorted({t for t, _ in spec})
    tobj = {t: Tier(f"cpu_real" if t == "cpu" else f"ssd_{t}", read_Bps=tiers_bps[t], write_Bps=tiers_bps[t]) for t in tiers}
    iost = {t: torch.cuda.Stream() for t in tiers}
    log, err = [], []
    t0 = time.perf_counter()

    def comp():
        try:
            while True:
                r = sc.next_compute(time.perf_counter() - t0)
                if r is None:
                    return
                if r[0] == "wait":
                    if r[1] is None:
                        if sc.all_done():
                            return
                        time.sleep(0.0002)
                    else:
                        sleep_until(t0 + r[1])
                    continue
                ts = time.perf_counter() - t0
                if r[0] == "new":
                    k = r[1]
                    E.new_turn(clones[k], S[k].n)
                    te = time.perf_counter() - t0
                    log.append(("new", k, -1, ts, te))
                    sc.finish(k, "new", te)
                else:
                    _, k, i = r
                    E.recompute(clones[k], i)
                    te = time.perf_counter() - t0
                    log.append(("rec", k, i, ts, te))
                    sc.finish(k, "rec", te)
        except Exception as ex:                                # 執行緒的例外要帶回主執行緒（規則 2：不吞錯）
            err.append(repr(ex))

    def load(tier):
        try:
            tb = tobj[tier]
            while True:
                r = sc.next_load(tier, time.perf_counter() - t0)
                if r is None:
                    return
                if r[0] == "wait":
                    sleep_until(t0 + r[1])
                    continue
                _, k, i = r
                ts = time.perf_counter() - t0
                done = tb.reserve_read(E.km.chunk_bytes)
                with torch.cuda.stream(iost[tier]):
                    clones[k].chunk_view(i).copy_(E.host[i], non_blocking=True)
                    ev = torch.cuda.Event()
                    ev.record(iost[tier])
                sleep_until(done)
                ev.synchronize()
                te = time.perf_counter() - t0
                log.append(("load", k, i, ts, te))
                sc.finish(k, "load", te)
        except Exception as ex:
            err.append(repr(ex))

    th = [threading.Thread(target=load, args=(t,)) for t in tiers]
    for t in th:
        t.start()
    comp()
    for t in th:
        t.join()
    torch.cuda.synchronize()
    if err:
        raise RuntimeError("; ".join(err))
    bad = None
    if check:
        bad = 0
        for k, s in enumerate(S):
            n = s.n * E.CHUNK
            bad += int(not torch.equal(clones[k].kv[:, :, :, :n], E.b.ref[:, :, :, :n]))
    return S, log, bad


def cmd_grid(a):
    E = Env(a.model, 64)
    torch = E.torch
    from m10_speed import Busy, _watcher
    run_id = os.environ.get("RUN_ID", "NOT_SET")
    bw = json.load(open(os.path.join(os.path.dirname(HERE), f"results/m10_precision_mi300x/speed_{a.model}/bw.json")))
    busy_lv = json.load(open(os.path.join(os.path.dirname(HERE), "results/m7_write_policy_mi300x/busy_levels.json")))
    tiers_bps = {"cpu": 0, **{k: v["GiBps"] * GiB for k, v in bw["targets"].items()}}
    w, level = _watcher(os.path.join(a.out, f"gpu_guard_{run_id}_grid.json"))
    rows, events = [], []
    for st in a.gpu_states:
        duty = 0.0 if st == "idle" else float(busy_lv[st])
        with Busy(duty, watch=w):
            # 這個 process、這個 GPU 狀態下的 solo 常數（預測用）：f(i) 與每塊載入時間
            km0 = E.clones_for(1)[0]
            km0.kv.copy_(E.km.kv)
            f_meas = []
            for i in range(64):
                ts = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    E.recompute(km0, i)
                    ts.append(time.perf_counter() - t0)
                f_meas.append(statistics.median(ts))
            S1, log1, _ = run_cell(E, tiers_bps, [("cpu", 64)], "cake", "rr", lambda i: f_meas[i],
                                   {"cpu": 0.0013, **{k: bw["chunk_bytes"] / v for k, v in tiers_bps.items() if v}}, False)
            l_cpu = statistics.median(e[4] - e[3] for e in log1 if e[0] == "load")
            t_new = statistics.median(e[4] - e[3] for e in log1 if e[0] == "new")
            tl_est = {"cpu": l_cpu, **{k: bw["chunk_bytes"] / v for k, v in tiers_bps.items() if v}}
            f_est = lambda i: f_meas[i]                        # noqa: E731
            print(st, "solo 常數：f0", round(f_meas[0], 4), "f63", round(f_meas[63], 4), "l_cpu", round(l_cpu, 5),
                  "t_new", round(t_new, 4), flush=True)
            for wl in a.workloads:
                for N in a.N:
                    spec = WORKLOADS[wl](N)
                    for mode in a.modes:
                        for order in a.orders:
                            if N == 1 and (mode, order) != ("cake", "rr"):
                                continue                      # N＝1 時所有順序相同；mode 另外由 G5b 決定
                            pred = simulate([Sess(n=n, tier=t) for t, n in spec] if mode == "cake" else
                                            apply_mode([Sess(n=n, tier=t) for t, n in spec], mode, f_est, tl_est),
                                            order, f_est, tl_est, t_new=t_new)
                            pst = stats([s["ttft"] for s in pred["sessions"]])
                            for rep in range(a.reps):
                                S, log, bad = run_cell(E, tiers_bps, spec, mode, order, f_est, tl_est, check=(rep == 0))
                                tt = stats([s.t_ttft - s.arrive for s in S])
                                rows.append(dict(
                                    run_id=run_id, ts=now_iso(), model=a.model, gpu_state=st, workload=wl, N=N, mode=mode,
                                    order=order, rep=rep, ttft_mean=round(tt["mean"], 5), ttft_p50=round(tt["p50"], 5),
                                    ttft_p95=round(tt["p95"], 5), makespan=round(max(s.t_ttft for s in S), 5),
                                    n_rec=sum(s.n_rec for s in S), n_load=sum(s.n_load for s in S),
                                    assign="".join(s.assign or "-" for s in S), kv_bad=bad,
                                    pred_ttft_mean=round(pst["mean"], 5), pred_ttft_p95=round(pst["p95"], 5),
                                    pred_makespan=round(pred["makespan"], 5),
                                    ratio_mean=round(tt["mean"] / pst["mean"], 4), ratio_p95=round(tt["p95"] / pst["p95"], 4)))
                                for e in log:
                                    events.append(dict(run_id=run_id, gpu_state=st, workload=wl, N=N, mode=mode, order=order,
                                                       rep=rep, kind=e[0], sess=e[1], chunk=e[2], t0=round(e[3], 6),
                                                       t1=round(e[4], 6)))
                                print({k: rows[-1][k] for k in ("gpu_state", "workload", "N", "mode", "order", "rep",
                                                                "ttft_mean", "pred_ttft_mean", "ratio_mean", "kv_bad")}, flush=True)
                                if bad:
                                    raise RuntimeError(f"KV 不相同：{wl} N={N} {mode} {order}（{bad} 個 session）")
    contaminated = bool(w and w.contaminated)
    if w:
        w.__exit__(None, None, None)
    for r in rows:
        r["contaminated"], r["host_level"] = contaminated, level
    for name, rr in (("grid", rows), ("events", events)):
        with open(os.path.join(a.out, f"{name}_{a.model}.csv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rr[0].keys()))
            wr.writeheader()
            wr.writerows(rr)
    if contaminated:
        open(os.path.join(a.out, "CONTAMINATED"), "w").write("gpu_guard\n")
    print("寫好", a.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("interf")
    p.add_argument("--model", default="llama")
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--gpu-states", nargs="+", default=["idle", "busy2"])
    p.add_argument("--out", required=True)
    p = sub.add_parser("grid")
    p.add_argument("--model", default="llama")
    p.add_argument("--N", nargs="+", type=int, default=[1, 4, 8])
    p.add_argument("--workloads", nargs="+", default=["homog_cpu", "homog_b32", "mix_cpu_b32"])
    p.add_argument("--modes", nargs="+", default=["cake", "g5b", "joint"])
    p.add_argument("--orders", nargs="+", default=["rr", "fcfs", "srpt"])
    p.add_argument("--gpu-states", nargs="+", default=["idle"])
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"interf": cmd_interf, "grid": cmd_grid}[a.cmd](a)


if __name__ == "__main__":
    main()
