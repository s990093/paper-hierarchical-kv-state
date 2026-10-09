"""m7_write_policy.py — 第一階段實驗驅動（phase1/05：C3–C7、A0、A1、A2、B）

子命令：
  busy-calib  C5：不同 duty 的背景負載下，量 chunk 重算變慢多少
  noise       C4：S0、L=16K、GPU 閒、限速器（local）重複還原
  a0          A0：只算／只載／Cake × L × 讀取頻寬
  a1          A1：單一 session，S0/S2a/S3/S5/CPU-all × L × GPU 狀態 × SSD 參數 × gap
  anchor      C7(2)：限速器 vs 真實裝置（CPU 真實 H2D、本地 SSD、NFS 檔案）
  a2          A2：真實裝置上，還原時背景有別的 session 在寫
  b           B：8 session × 4 輪，容量有限，8 個策略
所有 CSV 都有 run_id、ts，並附 contaminated_gpu、contaminated_disk 欄。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m7_model import CHUNK, KVModel  # noqa: E402
from m7_restore_harness import GiB, Restorer, Tier, sleep_until, write_boundary  # noqa: E402

MiB = 1 << 20
# 每個 chunk 的位元組數；預設 Llama-3.1-8B 的 64 MiB。換模型時用 M7_CHUNK_BYTES 設（第 1 輪破解計劃）
CHUNK_BYTES = int(os.environ.get("M7_CHUNK_BYTES", 64 * MiB))
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "m7_write_policy_mi300x")


def run_id():
    return os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc"


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


class Out:
    def __init__(self, path, fields):
        new = not os.path.exists(path)
        self.f = open(path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=["run_id", "ts"] + fields, extrasaction="ignore")
        if new:
            self.w.writeheader()
        self.rid = run_id()

    def row(self, **kw):
        kw.update(run_id=self.rid, ts=ts())
        for k, v in list(kw.items()):
            if isinstance(v, float):
                kw[k] = round(v, 6)
        self.w.writerow(kw)
        self.f.flush()


def load_params():
    return json.load(open(os.path.join(RES, "tier_params.json")))


def load_f():
    """C1 的 f(i)（秒），取每個 chunk 的中位數；GPU 閒。"""
    import pandas as pd
    d = pd.read_csv(os.environ.get("M7_F_CSV") or os.path.join(RES, "calib_c1.csv"))
    d = d[(d.item == "f_chunk") & (d.gpu_state == "idle")]
    m = d.groupby("chunk_idx").ms.median()
    return [float(m[i]) / 1e3 for i in sorted(m.index)]


IO_MODEL = os.environ.get("M7_IO_MODEL", "fifo")   # fifo | share（share 的 k 來自 tier_params.json 的 k_read）


CPU_GIBPS_OVERRIDE = os.environ.get("M7_CPU_GIBPS")   # 敏感度：CPU 層改用 vLLM OffloadingConnector 實測速度（C6）


def mk_tier(name, p):
    if name == "cpu" and CPU_GIBPS_OVERRIDE:
        p = dict(p, read_GiBps=float(CPU_GIBPS_OVERRIDE), write_GiBps=float(CPU_GIBPS_OVERRIDE))
    return Tier(name, read_Bps=p["read_GiBps"] * GiB, write_Bps=p["write_GiBps"] * GiB, c_s=p.get("c_ms", 0) / 1e3,
                mode=IO_MODEL, k_read=p.get("k_read", 1.0) if IO_MODEL == "share" else 1.0)


# ------------------------------------------------------------------ 爭用檢查
def _sda():
    for line in open("/proc/diskstats"):
        p = line.split()
        if p[2] == "sda":
            return int(p[5]) * 512, int(p[9]) * 512
    raise RuntimeError("sda not found in /proc/diskstats")  # 規則 7：查不到不等於沒有


_WATCH = None   # 整個 run 一個 GpuWatcher（amd-smi 輪詢太慢，不能每個 rep 開一個）
_ALLOW = set()  # 自己的背景負載（C5）pid


def start_watch(out_path):
    global _WATCH
    from gpu_guard import GpuWatcher
    _WATCH = GpuWatcher(gpu=0, own_root=os.getpid(), out_path=out_path)
    _WATCH.__enter__()


def stop_watch():
    if _WATCH is not None:
        _WATCH.__exit__(None, None, None)


class Guard:
    """GPU：讀全域 GpuWatcher 到目前為止有沒有外來 pid（自己的 busy 背景負載列白名單）。
    磁碟：sda 前後差（container 看到的是整台主機的 sda，含鄰居）。"""

    def __init__(self, allow_pids=()):
        self.allow = set(allow_pids) | _ALLOW

    def __enter__(self):
        self.d0 = _sda()
        return self

    def __exit__(self, *a):
        self.d1 = _sda()
        return False

    @property
    def gpu_contaminated(self):
        if _WATCH is None:
            return "NOT_CHECKED"
        bad = {p: v for p, v in _WATCH.intruders.items() if p not in self.allow}
        return bool(bad) or _WATCH.started_clean is False

    def disk_foreign_mib(self, own_bytes):
        tot = (self.d1[0] - self.d0[0]) + (self.d1[1] - self.d0[1])
        return (tot - own_bytes) / MiB


# ------------------------------------------------------------------ GPU 忙
class Busy:
    def __init__(self, duty):
        self.duty = duty
        self.p = None

    def __enter__(self):
        if self.duty > 0:
            self.p = subprocess.Popen([sys.executable, os.path.join(HERE, "m7_busy.py"), "--duty", str(self.duty)],
                                      stdout=subprocess.PIPE, text=True)
            line = self.p.stdout.readline()
            if not line.startswith("READY"):
                raise RuntimeError(f"busy did not start: {line}")
            time.sleep(1.0)
            _ALLOW.add(self.p.pid)
        return self

    def __exit__(self, *a):
        if self.p:
            self.p.terminate()
            self.p.wait(30)
        return False

    @property
    def pid(self):
        return self.p.pid if self.p else None


def gpu_states():
    p = os.path.join(RES, "busy_levels.json")
    if os.path.exists(p):
        return json.load(open(p))
    return {"idle": 0.0}


# ------------------------------------------------------------------ 共用：一個 session 的參考 KV
class Session:
    def __init__(self, km: KVModel, n_chunks: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.ids = torch.randint(1000, min(120000, km.cfg.vocab_size), (n_chunks * CHUNK + 512,), generator=g).to(km.device)
        self.n = n_chunks


class HostPool:
    def __init__(self, km, n):
        self.free = [km.host_chunk() for _ in range(n)]

    def get(self):
        return self.free.pop()

    def put(self, t):
        self.free.append(t)


def capture(km, idxs, pool):
    out = {}
    for i in idxs:
        h = pool.get()
        h.copy_(km.chunk_view(i), non_blocking=True)
        out[i] = h
    torch.cuda.synchronize()
    return out


def verify(km, host, loc, n):
    """還原後：每個有存的 chunk，GPU 上的 KV 是否與存的那份逐位元組相同。"""
    bad = 0
    for i in range(n):
        if host.get(i) is not None:
            if not torch.equal(km.chunk_view(i).cpu(), host[i]):
                bad += 1
    return bad


# ================================================================== C5
def busy_calib(a):
    o = Out(os.path.join(RES, "calib_c5.csv"), ["duty", "rep", "chunk_idx", "ms", "slowdown", "busy_pid"])
    km = KVModel(max_len=34 * CHUNK)
    s = Session(km, 33, 7)
    km.prefill_chunked(s.ids[:32 * CHUNK])

    def meas():
        r = []
        for _ in range(10):
            torch.cuda.synchronize(); t = time.perf_counter()
            km.forward_span(s.ids[32 * CHUNK:33 * CHUNK], 32 * CHUNK)
            torch.cuda.synchronize(); r.append(time.perf_counter() - t)
        r.sort()
        return r[len(r) // 2]
    base = meas()
    for duty in [0.0] + a.duties:
        with Busy(duty) as b:
            for rep in range(3):
                m = meas()
                o.row(duty=duty, rep=rep, chunk_idx=32, ms=m * 1e3, slowdown=m / base, busy_pid=b.pid)
                print(duty, rep, m * 1e3, m / base, flush=True)


# ================================================================== C4 / A0 / A1 共用
def single_setup(maxL_chunks, pool_n=None):
    km = KVModel(max_len=maxL_chunks * CHUNK + 512)
    pool = HostPool(km, pool_n or maxL_chunks)
    return km, Restorer(km), pool


def restore_once(km, rs, s, n, host, loc, mode, new_tokens=256):
    km.kv[:, :, :, :n * CHUNK].zero_()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    rec = rs.restore(s.ids, n, [host.get(i) for i in range(n)], loc, mode)
    t_new, tok = rs.new_turn(s.ids[n * CHUNK:n * CHUNK + new_tokens], n * CHUNK)
    rec["t_new"] = t_new
    rec["ttft"] = time.perf_counter() - t0
    rec["first_tok"] = tok
    return rec


REC_FIELDS = ["mode", "n_chunks", "meet", "n_recompute", "n_load_cpu", "n_load_ssd", "t_recompute", "t_load",
              "t_load_cpu", "t_load_ssd", "t_wait", "waiting_line", "t_restore", "t_new", "ttft", "first_tok"]


def noise(a):
    P = load_params()
    o = Out(os.path.join(RES, "c4_noise.csv"), ["slot", "proc_mode", "rep", "L", "contaminated_gpu"] + REC_FIELDS)
    n = 32
    km, rs, pool = single_setup(n)
    s = Session(km, n, 11)
    km.prefill_chunked(s.ids[:n * CHUNK])
    host = capture(km, range(n), pool)
    ssd = mk_tier("ssd_local", P["local"])
    loc = [ssd] * n
    restore_once(km, rs, s, n, host, loc, "cake")  # 暖機
    for rep in range(a.reps):
        ssd.reset_clock()
        with Guard() as g:
            rec = restore_once(km, rs, s, n, host, loc, "cake")
        o.row(slot=a.slot, proc_mode=a.proc_mode, rep=rep, L=n * CHUNK, contaminated_gpu=g.gpu_contaminated, **rec)
        print(rep, rec["ttft"], rec["meet"], flush=True)


def gbps_to_tier(gbps):
    Bps = gbps * 1e9 / 8
    return Tier(f"ssd_{gbps}Gbps", read_Bps=Bps, write_Bps=Bps, c_s=0.0)


def a0(a):
    P = load_params()
    o = Out(os.path.join(RES, "a0.csv"), ["bw_name", "read_GiBps", "rep", "L", "contaminated_gpu", "kv_bad"] + REC_FIELDS)
    maxn = max(a.L) // CHUNK
    km, rs, pool = single_setup(maxn)
    s = Session(km, maxn, 21)
    km.prefill_chunked(s.ids[:maxn * CHUNK])
    host_all = capture(km, range(maxn), pool)
    tiers = [(f"{g}Gbps", gbps_to_tier(g)) for g in a.gbps] + \
            [("local", mk_tier("ssd_local", P["local"])), ("nfs", mk_tier("ssd_nfs", P["nfs"]))]
    restore_once(km, rs, s, 8, host_all, [tiers[0][1]] * 8, "cake")
    for L in a.L:
        n = L // CHUNK
        for rep in range(a.reps):
            with Guard() as g:
                rec = restore_once(km, rs, s, n, host_all, [None] * n, "compute_only")
            o.row(bw_name="-", read_GiBps=0, rep=rep, L=L, contaminated_gpu=g.gpu_contaminated, **rec)
        for name, t in tiers:
            for mode in ("load_only", "cake"):
                for rep in range(a.reps):
                    t.reset_clock()
                    with Guard() as g:
                        rec = restore_once(km, rs, s, n, host_all, [t] * n, mode)
                    bad = verify(km, host_all, None, n) if rep == 0 else ""
                    o.row(bw_name=name, read_GiBps=t.read_Bps / GiB, rep=rep, L=L,
                          contaminated_gpu=g.gpu_contaminated, kv_bad=bad, **rec)
                    print(L, name, mode, rep, round(rec["ttft"], 4), rec["meet"], bad, flush=True)


# ------------------------------------------------------------------ A1
def a1_layout(strategy, n, f, P, tiers):
    """回傳 (存在哪一層的 list, 每層寫入的 chunk 數 dict, b)。"""
    cpu, ssd = tiers["cpu"], tiers["ssd"]
    ell_ssd = CHUNK_BYTES / ssd.read_Bps + ssd.c_s
    ell_cpu = CHUNK_BYTES / cpu.read_Bps + cpu.c_s
    if strategy == "S0":
        return [ssd] * n, {"ssd": n}, 0
    if strategy == "CPUall":
        return [cpu] * n, {"cpu": n}, 0
    if strategy in ("S2a", "S3"):
        b = write_boundary(n, f, ell_ssd)
        loc = [None] * b + [ssd] * (n - b)
        return loc, {"ssd": n if strategy == "S2a" else n - b}, b
    if strategy == "S5":
        b = write_boundary(n, f, ell_cpu)
        return [ssd] * b + [cpu] * (n - b), {"ssd": b, "cpu": n - b}, b
    raise ValueError(strategy)


def a1(a):
    P = load_params()
    f = load_f()
    o = Out(os.path.join(RES, f"a1_{IO_MODEL}.csv"), ["strategy", "gpu_state", "busy_duty", "ssd_dev", "gap", "rep", "L", "b",
                                          "w_chunks_ssd", "w_chunks_cpu", "t_write_ssd_s", "contaminated_gpu",
                                          "kv_bad"] + REC_FIELDS)
    maxn = max(a.L) // CHUNK
    km, rs, pool = single_setup(maxn)
    s = Session(km, maxn, 31)
    km.prefill_chunked(s.ids[:maxn * CHUNK])
    host_all = capture(km, range(maxn), pool)
    states = gpu_states()
    for gs in a.gpu_states:
        duty = states[gs]
        with Busy(duty) as bz:
            for dev in a.ssd_devs:
                tiers = {"cpu": mk_tier("cpu", P["cpu"]), "ssd": mk_tier(f"ssd_{dev}", P[dev])}
                for L in a.L:
                    n = L // CHUNK
                    for strat in a.strategies:
                        loc, w, b = a1_layout(strat, n, f, P, tiers)
                        host = {i: host_all[i] for i in range(n) if loc[i] is not None}
                        for gap in a.gaps:
                            for rep in range(a.reps):
                                for t in tiers.values():
                                    t.reset_clock()
                                t_ws = 0.0
                                if gap == "0":   # 寫完立刻回來：寫入佔用裝置時間
                                    now = time.perf_counter()
                                    for k, cnt in w.items():
                                        for _ in range(cnt):
                                            tiers[k].reserve_write(CHUNK_BYTES, now)
                                    t_ws = tiers["ssd"].backlog(now)
                                else:
                                    t_ws = w.get("ssd", 0) * CHUNK_BYTES / tiers["ssd"].write_Bps
                                with Guard(allow_pids=[bz.pid] if bz.pid else []) as g:
                                    rec = restore_once(km, rs, s, n, host, loc, "cake")
                                bad = verify(km, host, loc, n) if (rep == 0 and gap == a.gaps[0]) else ""
                                o.row(strategy=strat, gpu_state=gs, busy_duty=duty, ssd_dev=dev, gap=gap, rep=rep,
                                      L=L, b=b, w_chunks_ssd=w.get("ssd", 0), w_chunks_cpu=w.get("cpu", 0),
                                      t_write_ssd_s=t_ws, contaminated_gpu=g.gpu_contaminated, kv_bad=bad, **rec)
                            print(gs, dev, L, strat, gap, round(rec["ttft"], 4), rec["meet"], flush=True)


# ------------------------------------------------------------------ 真實裝置層（C7 錨點、A2）
class FileTier(Tier):
    """真的把 chunk 寫成檔案、用 O_DIRECT 讀回 pinned 緩衝再 H2D。時間全是真實的。"""

    def __init__(self, name, root):
        super().__init__(name, read_Bps=0, write_Bps=0)
        self.root = root
        os.makedirs(root, exist_ok=True)

    def path(self, key):
        return f"{self.root}/{key}.bin"

    def write_file(self, key, host):
        mv = memoryview(host.view(torch.uint8).numpy())
        fd = os.open(self.path(key), os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
        try:
            os.write(fd, mv)
        finally:
            os.close(fd)

    def read_file(self, key, host):
        mv = memoryview(host.view(torch.uint8).numpy())
        fd = os.open(self.path(key), os.O_RDONLY | os.O_DIRECT)
        try:
            n = os.readv(fd, [mv])
        finally:
            os.close(fd)
        assert n == host.numel() * 2, n


class RealRestorer(Restorer):
    """載入線改走真實裝置：FileTier → 讀檔到 staging pinned → H2D；CPU 真實層 → 直接 H2D。"""

    def __init__(self, km, staging):
        super().__init__(km)
        self.staging = staging
        self.keys = {}

    def _load(self, i, host, tier):
        if isinstance(tier, FileTier):
            st = self.staging
            tier.read_file(self.keys[i], st)
            tier.bytes_read += self.km.chunk_bytes
            src = st
        else:
            src = host
        with torch.cuda.stream(self.iostream):
            self.km.chunk_view(i).copy_(src, non_blocking=True)
        self.iostream.synchronize()
        return time.perf_counter()


def anchor(a):
    """C7(2)：同一個 L，限速器（實測參數）vs 真實裝置，用 load_only 與 cake 各跑。"""
    P = load_params()
    o = Out(os.path.join(RES, "c7_anchor.csv"), ["device", "io", "rep", "L", "read_bytes_delta", "disk_foreign_mib",
                                                 "contaminated_gpu", "contaminated_disk", "kv_bad"] + REC_FIELDS)
    maxn = max(a.L) // CHUNK
    km, rs, pool = single_setup(maxn)
    staging = km.host_chunk()
    rr = RealRestorer(km, staging)
    s = Session(km, maxn, 41)
    km.prefill_chunked(s.ids[:maxn * CHUNK])
    host_all = capture(km, range(maxn), pool)
    roots = {"local": "/var/tmp/m7anchor", "nfs": "/mlsteam/data/tiara/runs/_m7anchor"}
    for dev in ("cpu", "local", "nfs"):
        sim = mk_tier(f"{'cpu' if dev == 'cpu' else 'ssd_' + dev}", P[dev])
        if dev == "cpu":
            real = Tier("cpu_real", read_Bps=0, write_Bps=0)
        else:
            real = FileTier(f"ssd_{dev}_real", roots[dev])
            for i in range(maxn):
                real.write_file(f"c{i}", host_all[i])
                rr.keys[i] = f"c{i}"
            os.sync()
        for L in a.L:
            n = L // CHUNK
            for mode in ("load_only", "cake"):
                for io, tier, R in (("sim", sim, rs), ("real", real, rr)):
                    for rep in range(a.reps):
                        tier.reset_clock()
                        io0 = int([x for x in open("/proc/self/io") if x.startswith("read_bytes")][0].split()[1])
                        with Guard() as g:
                            km.kv[:, :, :, :n * CHUNK].zero_(); torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            rec = R.restore(s.ids, n, [host_all[i] for i in range(n)], [tier] * n, mode)
                            t_new, tok = R.new_turn(s.ids[n * CHUNK:n * CHUNK + 256], n * CHUNK)
                            rec.update(t_new=t_new, ttft=time.perf_counter() - t0, first_tok=tok)
                        io1 = int([x for x in open("/proc/self/io") if x.startswith("read_bytes")][0].split()[1])
                        nload = rec["n_load_cpu"] + rec["n_load_ssd"]
                        own = nload * km.chunk_bytes if (io == "real" and dev == "local") else 0
                        foreign = g.disk_foreign_mib(own)
                        bad = verify(km, host_all, None, n) if rep == 0 else ""
                        o.row(device=dev, io=io, rep=rep, L=L, read_bytes_delta=io1 - io0, disk_foreign_mib=foreign,
                              contaminated_gpu=g.gpu_contaminated, contaminated_disk=foreign > 64, kv_bad=bad, **rec)
                        print(dev, L, mode, io, rep, round(rec["ttft"], 4), rec["meet"], io1 - io0, flush=True)
        if dev != "cpu":
            for i in range(maxn):
                os.remove(real.path(f"c{i}"))


def a2(a):
    """真實裝置：session X（L=16K）用 Cake 從真實 SSD/NFS 還原；同時背景執行緒寫另一個 session 的 KV。"""
    P = load_params()
    f = load_f()
    o = Out(os.path.join(RES, f"a2_{IO_MODEL}.csv"), ["device", "io", "bg_write", "bg_chunks", "rep", "L", "bg_write_s",
                                          "disk_foreign_mib", "contaminated_gpu", "contaminated_disk"] + REC_FIELDS)
    nX = a.L // CHUNK
    nW = 64   # 背景 session 的長度：32K
    km, rs, pool = single_setup(max(nX, nW), pool_n=nX + 1)
    rr = RealRestorer(km, km.host_chunk())
    s = Session(km, nX, 51)
    km.prefill_chunked(s.ids[:nX * CHUNK])
    hostX = capture(km, range(nX), pool)
    wbuf = pool.get()   # 背景寫入的內容（大小正確即可）
    roots = {"local": "/var/tmp/m7a2", "nfs": "/mlsteam/data/tiara/runs/_m7a2"}
    for dev in a.devs:
        real = FileTier(f"ssd_{dev}_real", roots[dev])
        for i in range(nX):
            real.write_file(f"x{i}", hostX[i]); rr.keys[i] = f"x{i}"
        os.sync()
        sim = mk_tier(f"ssd_{dev}", P[dev])
        ell = CHUNK_BYTES / sim.read_Bps
        b = write_boundary(nW, f, ell)
        bg = {"none": 0, "S3_partial": nW - b, "S0_full": nW}
        for bgname, cnt in bg.items():
            for io in ("real", "sim"):
                for rep in range(a.reps):
                    sim.reset_clock()
                    tier = real if io == "real" else sim
                    R = rr if io == "real" else rs
                    bg_t = {}

                    def writer():
                        t = time.perf_counter()
                        for k in range(cnt):
                            real.write_file(f"w{k}", wbuf)
                        bg_t["s"] = time.perf_counter() - t
                    th = None
                    with Guard() as g:
                        if cnt:
                            if io == "real":
                                th = threading.Thread(target=writer); th.start(); time.sleep(0.05)
                            else:
                                now = time.perf_counter()
                                for _ in range(cnt):
                                    sim.reserve_write(CHUNK_BYTES, now)
                                bg_t["s"] = sim.backlog(now)
                        km.kv[:, :, :, :nX * CHUNK].zero_(); torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        rec = R.restore(s.ids, nX, [hostX[i] for i in range(nX)], [tier] * nX, "cake")
                        t_new, tok = R.new_turn(s.ids[nX * CHUNK:nX * CHUNK + 256], nX * CHUNK)
                        rec.update(t_new=t_new, ttft=time.perf_counter() - t0, first_tok=tok)
                        if th:
                            th.join()
                    nload = rec["n_load_ssd"]
                    own = 0
                    if dev == "local" and io == "real":
                        own = nload * km.chunk_bytes + cnt * km.chunk_bytes
                    foreign = g.disk_foreign_mib(own) if dev == "local" else 0.0
                    o.row(device=dev, io=io, bg_write=bgname, bg_chunks=cnt, rep=rep, L=a.L,
                          bg_write_s=bg_t.get("s", 0.0), disk_foreign_mib=foreign,
                          contaminated_gpu=g.gpu_contaminated, contaminated_disk=foreign > 256, **rec)
                    print(dev, bgname, io, rep, round(rec["ttft"], 4), rec["meet"], round(bg_t.get("s", 0), 3), flush=True)
                    for k in range(cnt if io == "real" else 0):
                        try:
                            os.remove(real.path(f"w{k}"))
                        except FileNotFoundError:
                            pass
        for i in range(nX):
            os.remove(real.path(f"x{i}"))


# ================================================================== B
# S4+P：照 Pensieve 原文的保留值 V=Cost/T（related_papers_update.md 建議，事後新增）
# S5s：分界 b 以 SSD 層算（預期「從 SSD 還原時會被重算」的前段直接寫 SSD），事後新增的探索組，不進判定
STRATS_B = ["R0", "S0", "S1", "S2b", "S3", "S4", "S4+", "S4+P", "S5", "S5s"]
# 08_ablation_plan.md §4（2026-10-09 開跑前列出）：S4L／S5L（LRU session 內從前段搬）、S5P（S5 寫入＋Pensieve 淘汰）、
# S5c（全寫 CPU，另把 i<b_ssd 預先複製到 SSD；淘汰同 S4+，有副本就直接丟）
STRATS_B2 = ["S4L", "S5L", "S5P", "S5c"]
# 08 §8 修正（2026-10-09，看過模擬掃描之後才加，只會讓結論更保守）：S4B＝全寫 CPU，滿了才「延後」套用同一個分界 b：
# 先搬任何 session 裡 i<b_cpu(目前歷史長度) 的 chunk（最久沒用的 session 先），沒有了才照 S4L。S5L 的延後版對照組
STRATS_B2 += ["S4B"]
# 11_round1_plan.md（2026-10-09 晚，開跑前寫死）：S4W＝背景版。寫入同 S4B（全寫 CPU），淘汰順序同 S4B，
# 但每輪結束就先搬到 CPU 至少空出 25%，不等滿。hold 下它是「延後但不卡請求路徑」的對照組
STRATS_B2 += ["S4W"]
# 11 追加 2（2026-10-09 18:05）：S4C＝預先清理版。寫入、淘汰同 S4+；每輪結束後確保淘汰順序最前面的 25%×容量 個 chunk 有 SSD 副本（複製、不移走）
STRATS_B2 += ["S4C"]
# 每輪新增的 chunk 數：chat＝每輪 8K；doc＝第 1 輪一次寫 32K，之後每輪問 512 token
SCHED = {"chat": [16, 16, 16, 16], "doc": [64, 1, 1, 1]}


def make_workload(seed, n_sess=8, rounds=4, p_stop=0.25):
    """回傳事件序列 [(session, round)]。每個 session 依序走；每輪之後有 p_stop 機率不再回來。"""
    rng = random.Random(seed)
    plan = {}
    for s in range(n_sess):
        r = 1
        while r < rounds and rng.random() >= p_stop:
            r += 1
        plan[s] = r
    pending = {s: 1 for s in range(n_sess)}
    ev = []
    while pending:
        s = rng.choice(sorted(pending))
        ev.append((s, pending[s]))
        pending[s] += 1
        if pending[s] > plan[s]:
            del pending[s]
    return ev, plan


class BState:
    """每個 chunk 的狀態：key=(s,i) → {'host':tensor,'cpu':bool,'ssd':bool}。"""

    def __init__(self, strategy, cpu_cap, ssd_cap, tiers, f, pool, P, release="free"):
        self.st = strategy
        # free＝搬走時 CPU 空間立刻釋放（第一階段 D9）；hold＝寫往 SSD 的 chunk 在寫完前占一格 CPU（08 §3）
        self.release = release
        self.inflight = []
        self.hist = {}        # session → 目前歷史長度（chunk），S4B 用
        self.cpu_cap, self.ssd_cap = cpu_cap, ssd_cap
        self.t = tiers
        self.f = f
        self.pool = pool
        self.c = {}
        self.last = {}        # session → 最後使用時間（事件序號）
        self.idle_since = {}  # session → 上次服務結束的事件序號（Pensieve 的 T）
        self.cpu_used = 0
        self.ssd_used = 0
        self.w_bytes = {"cpu": 0, "ssd": 0}
        self.n_demote = 0
        self.ell = {k: CHUNK_BYTES / v.read_Bps for k, v in tiers.items()}

    # --- 動作 ---
    def _put(self, key, tier, now):
        c = self.c[key]
        if not c[tier]:
            c[tier] = True
            if tier == "cpu":
                self.cpu_used += 1
            else:
                self.ssd_used += 1
            done = self.t[tier].reserve_write(CHUNK_BYTES, now)
            self.w_bytes[tier] += CHUNK_BYTES
            if tier == "ssd":
                c["ssd_done"] = done
                if not c["cpu"]:
                    self.inflight.append(done)   # 直接寫 SSD：經 CPU 暫存區，寫完前占一格

    def _drop(self, key, tier):
        c = self.c[key]
        if c[tier]:
            c[tier] = False
            if tier == "cpu":
                self.cpu_used -= 1
                if c["ssd"]:
                    self.inflight.append(c.get("ssd_done", 0.0))   # SSD 那份還沒寫完 → 這格要等
            else:
                self.ssd_used -= 1
            if not c["cpu"] and not c["ssd"]:
                self.pool.put(c["host"])
                del self.c[key]

    def _demote(self, key, now):
        if not self.c[key]["ssd"]:      # S5c 預先複製過的 chunk：直接丟 CPU 那份，不算搬移
            self._put(key, "ssd", now)
            self.n_demote += 1
        self._drop(key, "cpu")

    # --- 選逐出對象 ---
    def _lru_session(self, tier):
        cand = {k[0] for k, c in self.c.items() if c[tier]}
        return min(cand, key=lambda s: self.last[s])

    def _cheapest(self, tier):
        cand = [k for k, c in self.c.items() if c[tier]]
        return min(cand, key=lambda k: (k[1], self.last[k[0]]))

    def _lru_prefix(self, tier):
        """S4L／S5L：最久沒用的 session 裡，位置最前面的 chunk。"""
        ls = self._lru_session(tier)
        return min(k for k, c in self.c.items() if k[0] == ls and c[tier])

    def _lazy_b(self, tier):
        """S4B：先找 i < b_cpu(該 session 目前長度) 的 chunk（最久沒用的 session 先、位置前的先）；沒有就照 S4L。"""
        bs = {s: write_boundary(n, self.f, self.ell["cpu"]) for s, n in self.hist.items()}
        cand = [k for k, c in self.c.items() if c[tier] and k[1] < bs[k[0]]]
        if cand:
            return min(cand, key=lambda k: (self.last[k[0]], k[1]))
        return self._lru_prefix(tier)

    def gate(self, t):
        """hold：下一個請求要等「CPU 已用＋寫往 SSD 中的 chunk ≤ 容量」才能開始。回傳可以開始的時間。"""
        if self.release != "hold":
            self.inflight.clear()
            return t
        pend = sorted(d for d in self.inflight if d > t)
        self.inflight = pend
        over = self.cpu_used + len(pend) - self.cpu_cap
        return t if over <= 0 else pend[over - 1]

    def _pensieve(self, tier, ev_idx):
        """Pensieve：保留值 V = Cost(s,l)/T，逐出 V 最小的；Cost 為重算該 chunk 的時間，T 為閒置時間。"""
        cand = [k for k, c in self.c.items() if c[tier]]

        def V(k):
            T = max(1, ev_idx - self.idle_since.get(k[0], ev_idx) + 1)
            return self.f[k[1]] / T
        return min(cand, key=lambda k: (V(k), k[1]))

    # --- 服務一輪之後 ---
    def after_round(self, s, n_old, n_new, captured, now, ev_idx):
        self.last[s] = ev_idx
        self.idle_since[s] = ev_idx
        self.hist[s] = n_new
        S = self.st
        b_ssd = write_boundary(n_new, self.f, self.ell["ssd"])
        b_cpu = write_boundary(n_new, self.f, self.ell["cpu"])
        for i in range(n_old, n_new):
            key = (s, i)
            if S == "S3" and i < b_ssd:
                self.pool.put(captured[i])
                continue
            self.c[key] = {"host": captured[i], "cpu": False, "ssd": False}
            if S in ("R0", "S1", "S2b"):
                self._put(key, "cpu", now); self._put(key, "ssd", now)
            elif S in ("S0", "S3"):
                self._put(key, "ssd", now)
            elif S in ("S4", "S4+", "S4+P", "S4L", "S4B", "S4W", "S4C"):
                self._put(key, "cpu", now)
            elif S in ("S5", "S5L", "S5P"):
                self._put(key, "cpu" if i >= b_cpu else "ssd", now)
            elif S == "S5c":
                self._put(key, "cpu", now)
                if i < b_ssd:
                    self._put(key, "ssd", now)
            elif S == "S5s":
                self._put(key, "cpu" if i >= b_ssd else "ssd", now)
            else:
                raise ValueError(S)
        # CPU 滿
        while self.cpu_used > self.cpu_cap:
            if S in ("R0", "S1"):
                ls = self._lru_session("cpu")
                for k in [k for k, c in self.c.items() if k[0] == ls and c["cpu"]]:
                    self._drop(k, "cpu")
            elif S == "S2b":
                self._drop(self._cheapest("cpu"), "cpu")
            elif S == "S4":
                ls = self._lru_session("cpu")
                for k in sorted(k for k, c in self.c.items() if k[0] == ls and c["cpu"]):
                    self._demote(k, now)
            elif S in ("S4+", "S5", "S5s", "S5c", "S4C"):
                self._demote(self._cheapest("cpu"), now)
            elif S in ("S4+P", "S5P"):
                self._demote(self._pensieve("cpu", ev_idx), now)
            elif S in ("S4L", "S5L"):
                self._demote(self._lru_prefix("cpu"), now)
            elif S in ("S4B", "S4W"):
                self._demote(self._lazy_b("cpu"), now)
            else:
                raise RuntimeError(f"{S} has no CPU tier but cpu_used={self.cpu_used}")
        # S4C（11 追加 2 的預先清理版）：淘汰順序（同 _cheapest）最前面的 25%×容量 個 chunk，沒有 SSD 副本的就複製一份
        if S == "S4C":
            k = int(0.25 * self.cpu_cap)
            order = sorted((kk for kk, c in self.c.items() if c["cpu"]), key=lambda kk: (kk[1], self.last[kk[0]]))
            for kk in order[:k]:
                if not self.c[kk]["ssd"]:
                    self._put(kk, "ssd", now)
        # S4W（11 第 1 輪新增的背景版）：每輪結束後就先搬，讓 CPU 至少空出 25%，不等到滿
        if S == "S4W":
            lo = self.cpu_cap - int(0.25 * self.cpu_cap)
            while self.cpu_used > lo:
                self._demote(self._lazy_b("cpu"), now)
        # SSD 滿
        while self.ssd_cap is not None and self.ssd_used > self.ssd_cap:
            if S in ("R0", "S1", "S0", "S3", "S4"):
                ls = self._lru_session("ssd")
                for k in [k for k, c in self.c.items() if k[0] == ls and c["ssd"]]:
                    self._drop(k, "ssd")
            elif S in ("S4+P", "S5P"):
                self._drop(self._pensieve("ssd", ev_idx), "ssd")
            else:
                self._drop(self._cheapest("ssd"), "ssd")

    def layout(self, s, n):
        host, loc = {}, []
        for i in range(n):
            c = self.c.get((s, i))
            if c is None:
                loc.append(None)
            else:
                host[i] = c["host"]
                loc.append(self.t["cpu"] if c["cpu"] else self.t["ssd"])
        return host, loc


def b_exp(a):
    P = load_params()
    f = load_f()
    sched = SCHED[a.workload]
    # 08 消融（--b2）寫到新檔，多記 cpu_gibps／release／workload／t_gate；第一階段的 b_*.csv 欄位不動
    if a.b2:
        path = os.environ.get("M7_B2_OUT") or os.path.join(RES, f"b2_{IO_MODEL}.csv")   # smoke test 用 M7_B2_OUT 另寫
        extra = ["cpu_gibps", "release", "workload", "t_gate"]
    else:
        assert a.release == "free" and a.workload == "chat", "release/workload 只能和 --b2 一起用"
        path = os.path.join(RES, f"b_{IO_MODEL}" + (f"_cpu{CPU_GIBPS_OVERRIDE}" if CPU_GIBPS_OVERRIDE else "") + ".csv")
        extra = []
    o = Out(path,
            ["strategy", "cpu_frac", "cpu_cap_chunks", "ssd_frac", "ssd_dev", "gap_s", "wl_seed", "rep", "ev",
             "session", "round", "hist_chunks", "n_missing", "cpu_used", "ssd_used", "w_bytes_cpu", "w_bytes_ssd",
             "n_demote", "ssd_backlog_s", "io_model", "contaminated_gpu", "kv_bad"] + extra + REC_FIELDS)
    n_sess, rounds = 8, len(sched)
    per_sess = sum(sched)
    total = n_sess * per_sess
    km = KVModel(max_len=per_sess * CHUNK + 512)
    rs = Restorer(km)
    pool = HostPool(km, total + 16)
    sess = [Session(km, per_sess, 1000 + s) for s in range(n_sess)]
    ev, plan = make_workload(a.wl_seed, n_sess, rounds)
    print("workload", a.workload, sched, ev, plan, flush=True)
    json.dump({"seed": a.wl_seed, "events": ev, "rounds_per_session": plan},
              open(os.path.join(RES, f"b_workload_seed{a.wl_seed}.json"), "w"))
    for rep in range(a.rep_start, a.rep_start + a.reps):
        for cf in a.cpu_fracs:
            for strat in a.strategies:
                cap = int(cf * total)
                tiers = {"cpu": mk_tier("cpu", P["cpu"]), "ssd": mk_tier(f"ssd_{a.ssd_dev}", P[a.ssd_dev])}
                st = BState(strat, cap, None if a.ssd_frac >= 1 else int(a.ssd_frac * total), tiers, f, pool, P,
                            release=a.release)
                for e, (s, r) in enumerate(ev):
                    n_old = sum(sched[:r - 1])
                    n_new = n_old + sched[r - 1]
                    host, loc = st.layout(s, n_old)
                    missing = sum(1 for x in loc if x is None)
                    backlog = tiers["ssd"].backlog(time.perf_counter())
                    km.kv[:, :, :, :n_new * CHUNK].zero_()
                    torch.cuda.synchronize()
                    with Guard() as g:
                        t0 = time.perf_counter()          # 請求到達
                        t_go = st.gate(t0)                # hold：等前一輪卸載完（free 時 = t0）
                        sleep_until(t_go)
                        if n_old:
                            mode = "load_only" if strat == "R0" else "cake"
                            rec = rs.restore(sess[s].ids, n_old, [host.get(i) for i in range(n_old)], loc, mode)
                        else:
                            rec = {"mode": "none", "n_chunks": 0}
                        t_new, tok = rs.new_turn(sess[s].ids[n_old * CHUNK:n_new * CHUNK], n_old * CHUNK)
                        rec.update(t_new=t_new, ttft=time.perf_counter() - t0, first_tok=tok)
                    bad = ""
                    if a.verify and rep == a.rep_start and n_old:
                        bad = verify(km, host, loc, n_old)
                    captured = capture(km, range(n_old, n_new), pool)
                    st.after_round(s, n_old, n_new, captured, time.perf_counter(), e)
                    o.row(strategy=strat, cpu_frac=cf, cpu_cap_chunks=cap, ssd_frac=a.ssd_frac, ssd_dev=a.ssd_dev,
                          gap_s=a.gap, wl_seed=a.wl_seed, rep=rep, ev=e, session=s, round=r, hist_chunks=n_old,
                          n_missing=missing, cpu_used=st.cpu_used, ssd_used=st.ssd_used,
                          w_bytes_cpu=st.w_bytes["cpu"], w_bytes_ssd=st.w_bytes["ssd"], n_demote=st.n_demote,
                          ssd_backlog_s=backlog, io_model=IO_MODEL, contaminated_gpu=g.gpu_contaminated, kv_bad=bad,
                          cpu_gibps=CPU_GIBPS_OVERRIDE or P["cpu"]["read_GiBps"], release=a.release,
                          workload=a.workload, t_gate=t_go - t0, **rec)
                    if a.gap > 0:
                        time.sleep(a.gap)
                print(rep, cf, strat, "done", flush=True)
                for k in list(st.c):
                    pool.put(st.c[k]["host"])
                st.c.clear()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    p.add_argument("--reps", type=int, default=6)
    p.add_argument("--L", type=int, nargs="*", default=[4096, 8192, 16384, 32768])
    p.add_argument("--gbps", type=float, nargs="*", default=[7, 25, 32, 56, 100])
    p.add_argument("--duties", type=float, nargs="*", default=[0.2, 0.4, 0.5, 0.6, 0.8])
    p.add_argument("--slot", default="t1")
    p.add_argument("--proc-mode", default="same")
    p.add_argument("--strategies", nargs="*", default=None)
    p.add_argument("--gpu-states", nargs="*", default=["idle", "busy1", "busy2"])
    p.add_argument("--ssd-devs", nargs="*", default=["local", "nfs"])
    p.add_argument("--gaps", nargs="*", default=["inf", "0"])
    p.add_argument("--devs", nargs="*", default=["local", "nfs"])
    p.add_argument("--cpu-fracs", type=float, nargs="*", default=[0.25, 0.5, 1.0])
    p.add_argument("--ssd-frac", type=float, default=1.0)
    p.add_argument("--ssd-dev", default="nfs")
    p.add_argument("--gap", type=float, default=0.0)
    p.add_argument("--wl-seed", type=int, default=0)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--rep-start", type=int, default=0)
    p.add_argument("--b2", action="store_true", help="08 消融：寫到 b2_*.csv")
    p.add_argument("--release", default="free", choices=["free", "hold"])
    p.add_argument("--workload", default="chat", choices=list(SCHED))
    a = p.parse_args()
    if a.strategies is None:   # 2026-10-08：B 曾誤用 A1 的預設清單而崩潰（run m7-b-nfs-gap0），改成依子命令給預設
        a.strategies = STRATS_B if a.cmd == "b" else ["S0", "S2a", "S3", "S5", "CPUall"]
    if a.cmd == "a2":
        a.L = a.L[0] if isinstance(a.L, list) else a.L
    rd = os.path.join(os.environ.get("TIARA_RUNS", "/mlsteam/data/tiara/runs"), run_id())
    os.makedirs(rd, exist_ok=True)
    start_watch(os.path.join(rd, "gpu_guard.json"))
    try:
        {"busy-calib": busy_calib, "noise": noise, "a0": a0, "a1": a1, "anchor": anchor, "a2": a2,
         "b": b_exp}[a.cmd](a)
    finally:
        stop_watch()
        if _WATCH is not None and any(p not in _ALLOW for p in _WATCH.intruders):
            open(os.path.join(rd, "CONTAMINATED"), "w").write(json.dumps(_WATCH.intruders, default=str))


if __name__ == "__main__":
    main()
