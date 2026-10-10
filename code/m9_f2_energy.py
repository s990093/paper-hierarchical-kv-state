"""m9_f2_energy.py — F2：能耗 κ（D8-01 κ_E）的量測

子命令：
  probe    (a) 計數器解析度／更新週期＋用功率讀值驗證；(b) P_idle、P_dma、P_wait；(c) P_b 與每 chunk 重算能量
  sweep    (d) 32K（64 chunk）還原的 GPU 能量：只重算 R、只載入 L、動態 Cake C、固定切點 S(m)、Cake-E
  sampler  背景取樣（另一個 process）：能量累加值、功率、GPU 使用率，寫 CSV，直到被 SIGTERM

只 import harness（m7_model、m7_restore_harness），不改它。固定切點 S(m) 用 Restorer._load/_recompute 組出來。
輸出：$TIARA_RUNS/$RUN_ID/ 下的 CSV（每列 run_id、ts）。repo 的摘要 CSV 由 m9_f2_analyze.py 產生。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RUNS = os.environ.get("TIARA_RUNS", "/mlsteam/data/tiara/runs")
RID = os.environ.get("RUN_ID") or (datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc")
RD = os.path.join(RUNS, RID)
os.makedirs(RD, exist_ok=True)
GiB = 1 << 30
MiB = 1 << 20


def ts():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


class Out:
    def __init__(self, path, fields):
        new = not os.path.exists(path)
        self.f = open(path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=["run_id", "ts"] + fields, extrasaction="ignore")
        if new:
            self.w.writeheader()

    def row(self, **kw):
        kw.update(run_id=RID, ts=ts())
        for k, v in list(kw.items()):
            if isinstance(v, float):
                kw[k] = round(v, 6)
        self.w.writerow(kw)
        self.f.flush()
        print(json.dumps(kw), flush=True)


# ------------------------------------------------------------------ amdsmi
class Smi:
    def __init__(self):
        import amdsmi
        self.a = amdsmi
        amdsmi.amdsmi_init(amdsmi.AmdSmiInitFlags.INIT_AMD_GPUS)
        hs = amdsmi.amdsmi_get_processor_handles()
        if len(hs) != 1:
            # 規則 7：不確定是哪一張就停，不要猜
            raise RuntimeError(f"expected exactly 1 GPU handle, got {len(hs)}")
        self.h = hs[0]
        self.bdf = amdsmi.amdsmi_get_gpu_device_bdf(self.h)

    def energy(self):
        """回傳 (host monotonic s, 能量 J, 原始累加值, 解析度 µJ, 計數器時間戳)。"""
        t = time.monotonic()
        d = self.a.amdsmi_get_energy_count(self.h)
        return t, d["energy_accumulator"] * d["counter_resolution"] / 1e6, d["energy_accumulator"], \
            d["counter_resolution"], d["timestamp"]

    def power(self):
        return self.a.amdsmi_get_power_info(self.h)

    def activity(self):
        return self.a.amdsmi_get_gpu_activity(self.h)

    def metrics(self):
        return self.a.amdsmi_get_gpu_metrics_info(self.h)


def _num(x):
    try:
        return float(x)
    except Exception:
        return None


# ------------------------------------------------------------------ sampler（另一個 process）
def sampler(a):
    smi = Smi()
    f = open(a.out, "w", newline="")
    w = csv.writer(f)
    w.writerow(["run_id", "ts", "t_mono", "E_J", "acc_raw", "cur_W", "avg_W", "gfx_act", "umc_act", "mm_act"])
    stop = {"x": False}

    def _h(*_):
        stop["x"] = True
    signal.signal(signal.SIGTERM, _h)
    signal.signal(signal.SIGINT, _h)
    dt = 1.0 / a.hz
    nxt = time.monotonic()
    while not stop["x"]:
        try:
            t, E, raw, _, _ = smi.energy()
            p = smi.power()
            act = smi.activity()
            w.writerow([RID, ts(), f"{t:.6f}", f"{E:.6f}", raw, p.get("current_socket_power"),
                        p.get("average_socket_power"), act.get("gfx_activity"), act.get("umc_activity"),
                        act.get("mm_activity")])
        except Exception as e:   # 規則 3：記錯誤，不吞
            w.writerow([RID, ts(), f"{time.monotonic():.6f}", "ERR", repr(e), "", "", "", "", ""])
        f.flush()
        nxt += dt
        d = nxt - time.monotonic()
        if d > 0:
            time.sleep(d)
        else:
            nxt = time.monotonic()
    f.close()


class SamplerProc:
    def __init__(self, path, hz):
        self.path, self.hz = path, hz

    def __enter__(self):
        self.p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "sampler", "--out", self.path,
                                   "--hz", str(self.hz)])
        time.sleep(1.0)
        return self

    def __exit__(self, *e):
        self.p.send_signal(signal.SIGTERM)
        self.p.wait(30)
        return False


# ------------------------------------------------------------------ 視窗量測
class Win:
    """量一段：前後讀計數器（先同步 GPU）。"""

    def __init__(self, smi, sync=True):
        self.smi, self.sync = smi, sync

    def __enter__(self):
        if self.sync:
            import torch
            torch.cuda.synchronize()
        self.t0, self.E0, *_ = self.smi.energy()
        return self

    def __exit__(self, *e):
        if self.sync:
            import torch
            torch.cuda.synchronize()
        self.t1, self.E1, *_ = self.smi.energy()
        self.secs = self.t1 - self.t0
        self.E = self.E1 - self.E0
        self.P = self.E / self.secs if self.secs > 0 else float("nan")
        return False


class PowerPoll:
    """同一個 process 裡的功率輪詢執行緒（只在 probe 用；sweep 用另一個 process，避免 GIL 干擾還原）。"""

    def __init__(self, smi, period=0.02):
        self.smi, self.period = smi, period
        self.samples = []

    def __enter__(self):
        self.stop = threading.Event()
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()
        return self

    def _loop(self):
        while not self.stop.is_set():
            t = time.monotonic()
            try:
                p = self.smi.power()
                act = self.smi.activity()
                self.samples.append((t, _num(p.get("current_socket_power")), _num(p.get("average_socket_power")),
                                     _num(act.get("gfx_activity"))))
            except Exception as e:
                self.samples.append((t, None, None, None))
                print("POWERPOLL_ERR", repr(e), flush=True)
            self.stop.wait(self.period)

    def __exit__(self, *e):
        self.stop.set()
        self.th.join()
        return False

    def summary(self, t0, t1):
        s = [x for x in self.samples if t0 <= x[0] <= t1]
        cur = [x[1] for x in s if x[1] is not None]
        avg = [x[2] for x in s if x[2] is not None]
        act = [x[3] for x in s if x[3] is not None]
        m = lambda v: statistics.fmean(v) if v else None  # noqa: E731
        return dict(n_samples=len(s), cur_W_mean=m(cur), avg_W_mean=m(avg), gfx_act_mean=m(act),
                    gfx_act_max=max(act) if act else None)


# ------------------------------------------------------------------ probe
def probe(a):
    smi = Smi()
    o = Out(os.path.join(RD, "f2_probe.csv"),
            ["phase", "rep", "secs", "E_J", "P_W", "n_samples", "cur_W_mean", "avg_W_mean", "gfx_act_mean",
             "gfx_act_max", "work", "note"])
    meta = {"bdf": smi.bdf}
    try:
        meta["gpu_metrics_snapshot"] = {k: str(v) for k, v in smi.metrics().items()}
    except Exception as e:
        meta["gpu_metrics_err"] = repr(e)
    meta["power_info"] = {k: str(v) for k, v in smi.power().items()}

    # (a1) 計數器：緊密迴圈讀 a.tight_s 秒，記下每次變化
    raw = []
    t_end = time.monotonic() + a.tight_s
    lat = []
    last = None
    while time.monotonic() < t_end:
        c0 = time.monotonic()
        t, E, acc, res, stamp = smi.energy()
        lat.append(time.monotonic() - c0)
        if last is None or acc != last[2]:
            raw.append((t, E, acc, res, stamp))
            last = (t, E, acc, res, stamp)
    with open(os.path.join(RD, "f2_counter_raw.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "ts", "t_mono", "E_J", "acc_raw", "res_uJ", "stamp"])
        for r in raw:
            w.writerow([RID, ts(), f"{r[0]:.6f}", f"{r[1]:.6f}", r[2], r[3], r[4]])
    dts = [raw[i + 1][0] - raw[i][0] for i in range(len(raw) - 1)]
    dacc = [raw[i + 1][2] - raw[i][2] for i in range(len(raw) - 1)]
    dstamp = [raw[i + 1][4] - raw[i][4] for i in range(len(raw) - 1)]
    meta["counter"] = dict(
        n_changes=len(raw), tight_s=a.tight_s, n_reads=len(lat),
        read_latency_ms_median=statistics.median(lat) * 1e3, read_latency_ms_max=max(lat) * 1e3,
        res_uJ=raw[0][3] if raw else None,
        update_period_ms_median=statistics.median(dts) * 1e3 if dts else None,
        update_period_ms_p10=sorted(dts)[len(dts) // 10] * 1e3 if dts else None,
        update_period_ms_p90=sorted(dts)[9 * len(dts) // 10] * 1e3 if dts else None,
        update_period_ms_max=max(dts) * 1e3 if dts else None,
        min_increment_raw=min(dacc) if dacc else None,
        stamp_delta_median=statistics.median(dstamp) if dstamp else None,
        stamp_delta_min=min(dstamp) if dstamp else None,
    )
    print("COUNTER", json.dumps(meta["counter"]), flush=True)
    json.dump(meta, open(os.path.join(RD, "f2_probe_meta.json"), "w"), indent=1)

    with PowerPoll(smi, a.poll) as pp:
        # (b0) 還沒碰 torch：裸閒置（驗證計數器 vs 功率讀值）
        for rep in range(a.reps):
            with Win(smi, sync=False) as w:
                time.sleep(a.idle_short_s)
            o.row(phase="idle_bare", rep=rep, secs=w.secs, E_J=w.E, P_W=w.P, **pp.summary(w.t0, w.t1))

        import torch
        from m7_model import CHUNK, KVModel
        from m7_restore_harness import Restorer, Tier
        n = 64
        km = KVModel(max_len=n * CHUNK + 512)
        g = torch.Generator().manual_seed(21)
        ids = torch.randint(1000, min(120000, km.cfg.vocab_size), (n * CHUNK + 512,), generator=g).to(km.device)
        km.prefill_chunked(ids[:n * CHUNK])
        host = [km.host_chunk() for _ in range(n)]
        for i in range(n):
            host[i].copy_(km.chunk_view(i), non_blocking=True)
        torch.cuda.synchronize()
        rs = Restorer(km)
        time.sleep(5.0)

        # (b1) 模型已載入、GPU 沒事做
        for rep in range(a.reps):
            with Win(smi) as w:
                time.sleep(a.idle_s)
            o.row(phase="idle_model", rep=rep, secs=w.secs, E_J=w.E, P_W=w.P, **pp.summary(w.t0, w.t1))

        # (b2) 連續不限速 pinned H2D（64 MiB → KV 緩衝的 strided 視圖），和 Restorer._load 同路徑但不等限速
        s = torch.cuda.Stream()
        for rep in range(a.reps):
            k = 0
            with Win(smi) as w:
                t_end = time.monotonic() + a.dma_s
                while time.monotonic() < t_end:
                    i = k % n
                    with torch.cuda.stream(s):
                        km.chunk_view(i).copy_(host[i], non_blocking=True)
                        ev = torch.cuda.Event()
                        ev.record(s)
                    ev.synchronize()
                    k += 1
            o.row(phase="dma_h2d", rep=rep, secs=w.secs, E_J=w.E, P_W=w.P, work=k,
                  note=f"GiBps={k * km.chunk_bytes / w.secs / GiB:.2f}", **pp.summary(w.t0, w.t1))

        # (b3) 等限速載入：NFS 速度只載入 64 chunk（harness 的 load_only）
        P = json.load(open(os.path.join(HERE, "..", "results", "m7_write_policy_mi300x", "tier_params.json")))
        for rep in range(a.reps):
            t = Tier("ssd_nfs", read_Bps=P["nfs"]["read_GiBps"] * GiB, write_Bps=P["nfs"]["write_GiBps"] * GiB)
            km.kv[:, :, :, :n * CHUNK].zero_()
            torch.cuda.synchronize()
            time.sleep(a.gap)
            with Win(smi) as w:
                rec = rs.restore(ids, n, host, [t] * n, "load_only")
            o.row(phase="wait_nfs_load_only", rep=rep, secs=w.secs, E_J=w.E, P_W=w.P, work=n,
                  note=f"t_restore={rec['t_restore']:.4f}", **pp.summary(w.t0, w.t1))

        # (c1) 連續重算：整段 64 chunk（harness 的 compute_only），P_b
        for rep in range(a.reps):
            km.kv[:, :, :, :n * CHUNK].zero_()
            torch.cuda.synchronize()
            time.sleep(a.gap)
            with Win(smi) as w:
                rec = rs.restore(ids, n, host, [None] * n, "compute_only")
            o.row(phase="busy_compute_only_64", rep=rep, secs=w.secs, E_J=w.E, P_W=w.P, work=n,
                  note=f"t_restore={rec['t_restore']:.4f}", **pp.summary(w.t0, w.t1))
        # 讓 KV 回到完整狀態（後面逐 chunk 重算需要前綴）
        km.prefill_chunked(ids[:n * CHUNK])
        torch.cuda.synchronize()

        # (c2) 每個位置的每 chunk 重算能量：同一個 chunk 背靠背 K 次
        for rep in range(a.reps):
            for i in a.positions:
                time.sleep(a.gap)
                with Win(smi) as w:
                    for _ in range(a.k_rec):
                        km.forward_span(ids[i * CHUNK:(i + 1) * CHUNK], i * CHUNK)
                o.row(phase="e_rec", rep=rep, secs=w.secs, E_J=w.E, P_W=w.P, work=a.k_rec,
                      note=f"chunk={i};ms_per_chunk={w.secs / a.k_rec * 1e3:.3f};J_per_chunk={w.E / a.k_rec:.4f}",
                      **pp.summary(w.t0, w.t1))
        # (b4) 結尾再量一次閒置（漂移檢查）
        time.sleep(5.0)
        with Win(smi) as w:
            time.sleep(a.idle_s)
        o.row(phase="idle_model_end", rep=0, secs=w.secs, E_J=w.E, P_W=w.P, **pp.summary(w.t0, w.t1))
    # 存原始功率樣本
    with open(os.path.join(RD, "f2_probe_power_samples.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["run_id", "ts", "t_mono", "cur_W", "avg_W", "gfx_act"])
        for x in pp.samples:
            wr.writerow([RID, ts(), f"{x[0]:.6f}", x[1], x[2], x[3]])


# ------------------------------------------------------------------ sweep
def sweep(a):
    import torch
    from m7_model import CHUNK, KVModel
    from m7_restore_harness import Restorer, Tier

    smi = Smi()
    n = 64
    km = KVModel(max_len=n * CHUNK + 512)
    g = torch.Generator().manual_seed(a.seed)
    ids = torch.randint(1000, min(120000, km.cfg.vocab_size), (n * CHUNK + 512,), generator=g).to(km.device)
    km.prefill_chunked(ids[:n * CHUNK])
    host = [km.host_chunk() for _ in range(n)]
    for i in range(n):
        host[i].copy_(km.chunk_view(i), non_blocking=True)
    torch.cuda.synchronize()
    rs = Restorer(km)
    tiers = []
    for spec in a.tiers:
        name, gib = spec.split("=")
        tiers.append((name, float(gib)))
    cake_e = dict((x.split("=")[0], int(x.split("=")[1])) for x in a.cake_e)

    o = Out(os.path.join(RD, "f2_sweep.csv"),
            ["rep", "tier", "read_GiBps", "mode", "m", "secs", "E_J", "P_W", "t_restore", "t_rline", "t_lline",
             "n_recompute", "n_load", "meet", "kv_bad", "gap_s", "order_idx"])

    def verify():
        bad = 0
        for i in range(n):
            if not torch.equal(km.chunk_view(i).cpu(), host[i]):
                bad += 1
        return bad

    def split_restore(tier, m):
        """固定切點：前 m 個重算（GPU 線），後 n-m 個從後往前載入（I/O 線），兩線同時。"""
        t0 = time.perf_counter()
        tt = {}

        def lline():
            s0 = time.perf_counter()
            for q in range(n - 1, m - 1, -1):
                rs._load(q, host[q], tier)
            tt["l"] = time.perf_counter() - s0
        th = None
        if m < n:
            th = threading.Thread(target=lline)
            th.start()
        s0 = time.perf_counter()
        for i in range(m):
            rs._recompute(ids, i)
        tt["r"] = time.perf_counter() - s0
        if th is not None:
            th.join()
        torch.cuda.current_stream().wait_stream(rs.iostream)
        torch.cuda.current_stream().wait_stream(rs.cstream)
        torch.cuda.synchronize()
        return dict(t_restore=time.perf_counter() - t0, t_rline=tt.get("r", 0.0), t_lline=tt.get("l", 0.0),
                    n_recompute=m, n_load=n - m, meet=m)

    def one(rep, tname, gib, mode, m, order_idx):
        km.kv[:, :, :, :n * CHUNK].zero_()
        torch.cuda.synchronize()
        time.sleep(a.gap)
        tier = Tier(tname, read_Bps=gib * GiB, write_Bps=gib * GiB, c_s=0.0) if gib else None
        with Win(smi) as w:
            if mode == "R":
                rec = rs.restore(ids, n, host, [None] * n, "compute_only")
                info = dict(t_restore=rec["t_restore"], t_rline=rec["t_recompute"], t_lline=0.0,
                            n_recompute=n, n_load=0, meet=n)
            elif mode == "L":
                rec = rs.restore(ids, n, host, [tier] * n, "load_only")
                info = dict(t_restore=rec["t_restore"], t_rline=rec["t_recompute"], t_lline=rec["t_load"],
                            n_recompute=rec["n_recompute"], n_load=rec["n_load_cpu"] + rec["n_load_ssd"],
                            meet=rec["meet"])
            elif mode == "C":
                rec = rs.restore(ids, n, host, [tier] * n, "cake")
                info = dict(t_restore=rec["t_restore"], t_rline=rec["t_recompute"], t_lline=rec["t_load"],
                            n_recompute=rec["n_recompute"], n_load=rec["n_load_cpu"] + rec["n_load_ssd"],
                            meet=rec["meet"])
            else:   # S / CE / SMEET：固定切點
                info = split_restore(tier, m)
        bad = verify() if (rep == 0 and a.verify) else ""
        o.row(rep=rep, tier=tname, read_GiBps=gib or 0, mode=mode, m=info["meet"], secs=w.secs, E_J=w.E, P_W=w.P,
              kv_bad=bad, gap_s=a.gap, order_idx=order_idx, **info)
        return info

    # 暖機（不記）：每層各一次 Cake，加一次只重算
    for tname, gib in tiers:
        one(-1, tname, gib, "C", None, -1)
    one(-1, "-", 0, "R", None, -1)

    with SamplerProc(os.path.join(RD, "f2_sweep_power_trace.csv"), a.hz):
        for rep in range(a.reps):
            plan = [("-", 0, "R", None)]
            for tname, gib in tiers:
                plan.append((tname, gib, "L", None))
                plan.append((tname, gib, "C", None))
                for m in a.m_grid:
                    if 0 < m < n:
                        plan.append((tname, gib, "S", m))
                if tname in cake_e:
                    plan.append((tname, gib, "CE", cake_e[tname]))
                plan.append((tname, gib, "SMEET", None))
            # 只重算每輪跑 r_per_rep 次（殺掉條件 a 需要 ≥10 次），平均插在序列裡
            for j in range(1, a.r_per_rep):
                plan.insert(j * len(plan) // a.r_per_rep, ("-", 0, "R", None))
            # 每輪轉動順序，避免固定先後造成的系統偏差
            k = (rep * 7) % len(plan)
            plan = plan[k:] + plan[:k]
            meet = {}
            # SMEET 需要同一輪 Cake 的會合點：先跑完非 SMEET，再補
            for idx, (tname, gib, mode, m) in enumerate(plan):
                if mode == "SMEET":
                    continue
                info = one(rep, tname, gib, mode, m, idx)
                if mode == "C":
                    meet[tname] = info["meet"]
            for idx, (tname, gib, mode, m) in enumerate(plan):
                if mode == "SMEET":
                    one(rep, tname, gib, "SMEET", meet[tname], idx)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["probe", "sweep", "sampler"])
    p.add_argument("--out")
    p.add_argument("--hz", type=float, default=20.0)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--tight-s", type=float, default=5.0)
    p.add_argument("--poll", type=float, default=0.02)
    p.add_argument("--idle-s", type=float, default=30.0)
    p.add_argument("--idle-short-s", type=float, default=10.0)
    p.add_argument("--dma-s", type=float, default=10.0)
    p.add_argument("--gap", type=float, default=0.5)
    p.add_argument("--positions", type=int, nargs="*", default=[0, 15, 31, 47, 63])
    p.add_argument("--k-rec", type=int, default=20)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--tiers", nargs="*", default=["cpu35=35.4159", "cpu11=11.6", "cpu369=3.69",
                                                    "local=6.98115234375", "nfs=0.331083984375"])
    p.add_argument("--m-grid", type=int, nargs="*", default=[4, 8, 16, 24, 32, 40, 48, 56])
    p.add_argument("--cake-e", nargs="*", default=[])
    p.add_argument("--verify", type=int, default=1)
    p.add_argument("--r-per-rep", type=int, default=2)
    a = p.parse_args()
    {"probe": probe, "sweep": sweep, "sampler": sampler}[a.cmd](a)


if __name__ == "__main__":
    main()
