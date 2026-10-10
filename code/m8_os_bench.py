#!/usr/bin/env python3
"""D7（OS 角度）微量測：page cache、dirty writeback、讀取方向、SSD 掉崖、NFS 去重。

不用 GPU。每個子命令都要經 m7run 執行（讀環境變數 RUN_ID）。

  env   記錄 vm.dirty_*、cgroup memory、裝置資訊（唯讀）
  pc    (a)(b) 寫 N 個 64 MiB chunk（每 chunk 一個檔），buffered／fsync／odirect；
        邊寫邊記 /proc/meminfo、cgroup memory.stat、自己檔案的 page cache（cachestat）
  rd    (c) 讀取方向 × IO 大小 × O_DIRECT／buffered（冷）× 有無背景寫入
  seqw  (e) O_DIRECT 連續順序寫，每秒吞吐量
  dedup (f) NFS O_DIRECT：每 4 KiB 都不同 vs 4 KiB 重複
  bufmt 多執行緒 buffered 寫，寫完立刻刪（不落盤），量進 page cache 的速度
  analyze 從 results 的原始 CSV 算彙整表（讀取方向、干擾倍數、寫入延遲）
  warm  只讀：已 100% 在 page cache 的既有檔，量 buffered 熱讀（＝「SSD 層」其實從 DRAM 讀）

裝置路徑沿用第一階段校準（code/m7_calib.py:5）：
  local = /var/tmp/d7_<RUN_ID>         （overlay on sda／dm-1）
  nfs   = $TIARA_RUNS/<RUN_ID>/io       （NetApp nfs4）
所有資料檔在子命令結束時刪除（try/finally）。
"""
import argparse
import csv
import ctypes
import ctypes.util
import datetime as dt
import mmap
import os
import random
import shutil
import statistics
import sys
import threading
import time

import numpy as np

MiB = 1 << 20
GiB = 1 << 30
CHUNK = 64 * MiB
PAGE = os.sysconf("SC_PAGE_SIZE")
RUN_ID = os.environ.get("RUN_ID", "")
RUNS = os.environ.get("TIARA_RUNS", "/mlsteam/data/tiara/runs")
REPO = "/mlsteam/workspace/paper-hierarchical-kv-state"
RES = f"{REPO}/results/m8_directions"

libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
libc.syscall.restype = ctypes.c_long
SYS_CACHESTAT = 451  # x86_64, Linux >= 6.5


class _CSRange(ctypes.Structure):
    _fields_ = [("off", ctypes.c_uint64), ("len", ctypes.c_uint64)]


class _CS(ctypes.Structure):
    _fields_ = [("nr_cache", ctypes.c_uint64), ("nr_dirty", ctypes.c_uint64),
                ("nr_writeback", ctypes.c_uint64), ("nr_evicted", ctypes.c_uint64),
                ("nr_recently_evicted", ctypes.c_uint64)]


def cachestat(fd):
    """(cached, dirty, writeback) 的 page 數；失敗丟例外（不回傳 0，CLAUDE.md §1 規則 7）。"""
    r = _CSRange(0, 0)
    c = _CS()
    rc = libc.syscall(SYS_CACHESTAT, ctypes.c_uint(fd), ctypes.byref(r), ctypes.byref(c), ctypes.c_uint(0))
    if rc != 0:
        e = ctypes.get_errno()
        raise OSError(e, f"cachestat failed: {os.strerror(e)}")
    return c.nr_cache, c.nr_dirty, c.nr_writeback


def mincore_pages(path):
    """用 mmap+mincore 數一個檔有幾頁在 page cache（cachestat 的交叉驗證）。"""
    size = os.path.getsize(path)
    if size == 0:
        return 0
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if addr in (None, ctypes.c_void_p(-1).value):
            e = ctypes.get_errno()
            raise OSError(e, f"mmap failed: {os.strerror(e)}")
        try:
            n = (size + PAGE - 1) // PAGE
            vec = ctypes.create_string_buffer(n)
            if libc.mincore(addr, size, vec) != 0:
                e = ctypes.get_errno()
                raise OSError(e, f"mincore failed: {os.strerror(e)}")
            return int(np.frombuffer(vec.raw, dtype=np.uint8).__and__(1).sum())
        finally:
            libc.munmap(addr, size)
    finally:
        os.close(fd)


def now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class Out:
    """CSV 同時寫到 run 目錄與 results/m8_directions/（append）；每列都有 run_id、ts。"""

    def __init__(self, name, fields, to_results=True):
        self.fields = ["run_id", "ts"] + fields
        self.paths = [f"{run_dir()}/{name}"]
        if to_results and not os.environ.get("D7_SMOKE"):  # 煙霧測試只寫 run 目錄
            self.paths.append(f"{RES}/{name}")
        for p in self.paths:
            new = not os.path.exists(p)
            with open(p, "a", newline="") as f:
                if new:
                    csv.writer(f).writerow(self.fields)
        self.lock = threading.Lock()

    def row(self, **kw):
        kw = {"run_id": RUN_ID, "ts": now_iso(), **kw}
        with self.lock:
            for p in self.paths:
                with open(p, "a", newline="") as f:
                    csv.DictWriter(f, self.fields).writerow(kw)


def run_dir():
    if not RUN_ID:
        sys.exit("RUN_ID 沒設：請用 m7run 執行（CLAUDE.md §4.1）")
    d = f"{RUNS}/{RUN_ID}"
    if not os.path.isdir(d):
        # m7run 與 runsh 各自算一次時間戳，跨秒時目錄名會差 1 秒；照實記錄，不猜
        print(f"[warn] run dir {d} 不存在（m7run/runsh 時間戳跨秒？），建立之", file=sys.stderr)
        os.makedirs(d, exist_ok=True)
    return d


def dev_root(dev):
    if dev == "local":
        return f"/var/tmp/d7_{RUN_ID}"
    if dev == "nfs":
        return f"{run_dir()}/io"
    raise ValueError(dev)


# ------------------------------------------------------------------ 系統計數器（唯讀）
def meminfo():
    m = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":")
        m[k] = int(v.split()[0]) // 1024  # MiB
    return m


def cg_memstat():
    m = {}
    for line in open("/sys/fs/cgroup/memory.stat"):
        k, v = line.split()
        m[k] = int(v)
    return m


def vmstat():
    m = {}
    for line in open("/proc/vmstat"):
        k, v = line.split()
        m[k] = int(v)
    return m


def io_stat(dev="252:1"):
    for line in open("/sys/fs/cgroup/io.stat"):
        p = line.split()
        if p[0] == dev:
            return {kv.split("=")[0]: int(kv.split("=")[1]) for kv in p[1:]}
    raise RuntimeError(f"io.stat 沒有 {dev}")


def diskstats(name="sda"):
    for line in open("/proc/diskstats"):
        p = line.split()
        if p[2] == name:
            # sectors read, sectors written, io_ticks(ms)
            return int(p[5]), int(p[9]), int(p[12])
    raise RuntimeError(f"/proc/diskstats 沒有 {name}")


def nfs_bytes(mnt="/mlsteam/data/tiara"):
    """/proc/self/mountstats 的 bytes: 行（normal/direct read/write、server read/write）。"""
    lines = open("/proc/self/mountstats").read().splitlines()
    on = False
    for line in lines:
        if line.startswith("device "):
            on = f" mounted on {mnt} " in line
        elif on and line.strip().startswith("bytes:"):
            v = [int(x) for x in line.split()[1:]]
            keys = ["normal_read", "normal_write", "direct_read", "direct_write", "server_read", "server_write",
                    "read_pages", "write_pages"]
            return dict(zip(keys, v))
    raise RuntimeError(f"mountstats 找不到 {mnt}")


# ------------------------------------------------------------------ 資料
def aligned(n):
    return mmap.mmap(-1, n)  # page 對齊，可給 O_DIRECT


def fill_random(buf, seed=0):
    rng = np.random.default_rng(seed)
    a = np.frombuffer(buf, dtype=np.uint64)
    a[:] = rng.integers(0, 2**63, size=a.size, dtype=np.uint64)


def stamp(buf, chunk_id):
    """讓每個 4 KiB 區塊都不同（避免儲存端去重），其餘位元組保持隨機。"""
    a = np.frombuffer(buf, dtype=np.uint64)
    n4k = len(buf) // 4096
    a[::512] = (np.uint64(chunk_id) << np.uint64(24)) + np.arange(n4k, dtype=np.uint64)


# ------------------------------------------------------------------ 取樣器
class Sampler(threading.Thread):
    def __init__(self, out, period, tags, fds_ref, paths_ref, use_mincore):
        super().__init__(daemon=True)
        self.out, self.period, self.tags, self.fds_ref = out, period, tags, fds_ref
        # overlayfs 上 cachestat 永遠回 0（看的是 overlay inode 的 mapping，不是底層檔；
        # 見 run 20261010-054238-d7-env 的 probe），所以 local 改用 mmap+mincore，只拿得到「在不在 cache」
        self.paths_ref, self.use_mincore = paths_ref, use_mincore
        self.phase = "base"
        self.t0 = time.perf_counter()
        self.stop_ev = threading.Event()
        self.last = {}
        self.hist = []
        self.err = None

    def own(self):
        if self.use_mincore:
            c = sum(mincore_pages(p) for p in list(self.paths_ref))
            return c * PAGE / MiB, None, None
        c = d = w = 0
        for fd in list(self.fds_ref):
            a, b, e = cachestat(fd)
            c += a; d += b; w += e
        return c * PAGE / MiB, d * PAGE / MiB, w * PAGE / MiB

    def sample(self):
        oc, od, ow = self.own()
        mi, cg, vm = meminfo(), cg_memstat(), vmstat()
        r = dict(**self.tags, phase=self.phase, t_s=round(time.perf_counter() - self.t0, 3),
                 own_cached_mib=round(oc, 1), own_dirty_mib="NA" if od is None else round(od, 1),
                 own_wb_mib="NA" if ow is None else round(ow, 1),
                 mi_cached_mib=mi["Cached"], mi_dirty_mib=mi["Dirty"], mi_writeback_mib=mi["Writeback"],
                 mi_memavail_mib=mi["MemAvailable"], mi_memfree_mib=mi["MemFree"],
                 cg_file_mib=cg["file"] // MiB, cg_file_dirty_mib=cg["file_dirty"] // MiB,
                 cg_file_wb_mib=cg["file_writeback"] // MiB, cg_anon_mib=cg["anon"] // MiB,
                 cg_current_mib=int(open("/sys/fs/cgroup/memory.current").read()) // MiB,
                 vm_nr_dirty=vm["nr_dirty"], vm_nr_writeback=vm["nr_writeback"],
                 vm_nr_dirty_threshold=vm["nr_dirty_threshold"],
                 vm_nr_dirty_bg_threshold=vm["nr_dirty_background_threshold"],
                 cg_pgscan_direct=cg["pgscan_direct"], cg_pgsteal_direct=cg["pgsteal_direct"])
        self.last = r
        self.hist.append(r)
        self.out.row(**r)
        return r

    def run(self):
        try:
            while not self.stop_ev.is_set():
                self.sample()
                self.stop_ev.wait(self.period)
        except Exception as e:  # 記下來，主執行緒會檢查並失敗
            self.err = repr(e)

    def stop(self):
        self.stop_ev.set()
        self.join()
        if self.err:
            raise RuntimeError(f"sampler 失敗：{self.err}")


# ------------------------------------------------------------------ env
def cmd_env(a):
    o = Out("d7_env.csv", ["key", "value"])
    for f in ["dirty_ratio", "dirty_background_ratio", "dirty_bytes", "dirty_background_bytes",
              "dirty_expire_centisecs", "dirty_writeback_centisecs", "vfs_cache_pressure", "swappiness",
              "min_free_kbytes", "watermark_scale_factor"]:
        o.row(key=f"vm.{f}", value=open(f"/proc/sys/vm/{f}").read().strip())
    for f in ["memory.max", "memory.high", "memory.current"]:
        o.row(key=f"cgroup.{f}", value=open(f"/sys/fs/cgroup/{f}").read().strip())
    ev = open("/sys/fs/cgroup/memory.events").read().split()
    o.row(key="cgroup.memory.events", value=" ".join(ev))
    cg = cg_memstat()
    for k in ["file", "file_dirty", "file_writeback", "anon", "pgscan_direct", "pgsteal_direct",
              "workingset_refault_file"]:
        o.row(key=f"cgroup.memory.stat.{k}", value=cg[k])
    vm = vmstat()
    for k in ["nr_dirty_threshold", "nr_dirty_background_threshold"]:
        o.row(key=f"vmstat.{k}", value=vm[k])
    mi = meminfo()
    for k in ["MemTotal", "MemAvailable", "Cached", "Dirty", "Writeback"]:
        o.row(key=f"meminfo.{k}_mib", value=mi[k])
    for k, p in [("sda.model", "/sys/block/sda/device/model"), ("sda.vendor", "/sys/block/sda/device/vendor"),
                 ("sda.rev", "/sys/block/sda/device/rev"), ("sda.rotational", "/sys/block/sda/queue/rotational"),
                 ("sda.write_cache", "/sys/block/sda/queue/write_cache"),
                 ("sda.read_ahead_kb", "/sys/block/sda/queue/read_ahead_kb"),
                 ("dm-1.name", "/sys/block/dm-1/dm/name"), ("nfs_bdi.read_ahead_kb", "/sys/class/bdi/0:354/read_ahead_kb"),
                 ("kernel", "/proc/sys/kernel/osrelease")]:
        try:
            o.row(key=k, value=open(p).read().strip())
        except OSError as e:
            o.row(key=k, value=f"ERROR {e!r}")
    for line in open("/proc/mounts"):
        if " /mlsteam/data/tiara " in line:
            o.row(key="nfs.mount", value=line.strip())
    # cachestat 與 mincore 互相驗證（overlay 與 NFS 各一個 64 MiB buffered 檔）
    buf = aligned(CHUNK)
    fill_random(buf, 1)
    for dev in ("local", "nfs"):
        root = dev_root(dev)
        os.makedirs(root, exist_ok=True)
        p = f"{root}/probe.bin"
        try:
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            os.write(fd, buf); os.fsync(fd); os.close(fd)
            fd = os.open(p, os.O_RDONLY)
            c0 = cachestat(fd)
            m0 = mincore_pages(p)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            c1 = cachestat(fd)
            m1 = mincore_pages(p)
            os.close(fd)
            o.row(key=f"probe.{dev}.after_write_fsync",
                  value=f"cachestat_cached_pages={c0[0]} dirty={c0[1]} wb={c0[2]} mincore_pages={m0} file_pages={CHUNK // PAGE}")
            o.row(key=f"probe.{dev}.after_dontneed",
                  value=f"cachestat_cached_pages={c1[0]} dirty={c1[1]} wb={c1[2]} mincore_pages={m1}")
        finally:
            if os.path.exists(p):
                os.remove(p)
        if dev == "local":
            shutil.rmtree(root, ignore_errors=True)


# ------------------------------------------------------------------ (a)(b) page cache
def cmd_pc(a):
    tag_extra = a.tag or ""
    ow = Out("d7_pc_writes.csv", ["dev", "mode", "tag", "idx", "elapsed_s", "t_open_ms", "t_write_ms",
                                  "t_fsync_ms", "t_close_ms", "t_total_ms", "own_dirty_mib", "mi_dirty_mib",
                                  "cg_file_dirty_mib", "cg_file_wb_mib"],
             to_results=a.write_rows_to_results)
    ots = Out("d7_pc_timeseries.csv", ["dev", "mode", "tag", "phase", "t_s", "own_cached_mib", "own_dirty_mib",
                                       "own_wb_mib", "mi_cached_mib", "mi_dirty_mib", "mi_writeback_mib",
                                       "mi_memavail_mib", "mi_memfree_mib", "cg_file_mib", "cg_file_dirty_mib",
                                       "cg_file_wb_mib", "cg_anon_mib", "cg_current_mib", "vm_nr_dirty",
                                       "vm_nr_writeback", "vm_nr_dirty_threshold", "vm_nr_dirty_bg_threshold",
                                       "cg_pgscan_direct", "cg_pgsteal_direct"],
              to_results=a.write_rows_to_results)
    osum = Out("d7_pc_summary.csv", ["dev", "mode", "tag", "n_chunks", "gib", "write_s", "write_GiBps",
                                     "lat_p50_ms", "lat_p90_ms", "lat_p99_ms", "lat_max_ms", "close_p50_ms",
                                     "close_max_ms", "own_cached_end_write_mib", "own_dirty_peak_mib",
                                     "own_cached_peak_mib", "own_cached_post30_mib", "own_cached_post_end_mib",
                                     "t_dirty_zero_after_write_s", "drop_s", "own_cached_after_drop_mib",
                                     "dev_write_mib_delta", "nfs_normal_write_mib", "nfs_direct_write_mib",
                                     "nfs_server_write_mib", "mi_dirty_peak_mib", "cg_file_dirty_peak_mib",
                                     "note"])
    n = int(a.gib * GiB // CHUNK)
    buf = aligned(CHUNK)
    fill_random(buf, 7)
    for mode in a.modes:
        root = f"{dev_root(a.dev)}/{mode}{tag_extra}"
        os.makedirs(root, exist_ok=True)
        stat_fds = []
        paths = []
        smp = Sampler(ots, a.sample_s, dict(dev=a.dev, mode=mode, tag=a.tag), stat_fds, paths,
                      use_mincore=(a.dev == "local"))
        try:
            io0 = io_stat() if a.dev == "local" else None
            nb0 = nfs_bytes() if a.dev == "nfs" else None
            smp.start()
            time.sleep(a.base_s)
            smp.phase = "write"
            lats, closes = [], []
            t0 = time.perf_counter()
            for i in range(n):
                stamp(buf, i)
                p = f"{root}/c{i:05d}.bin"
                paths.append(p)
                flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
                if mode == "odirect":
                    flags |= os.O_DIRECT
                ta = time.perf_counter()
                fd = os.open(p, flags, 0o644)
                tb = time.perf_counter()
                w = os.write(fd, buf)
                if w != CHUNK:
                    raise OSError(f"short write {w} != {CHUNK} at {p}")
                tc = time.perf_counter()
                if mode == "fsync":
                    os.fsync(fd)
                td = time.perf_counter()
                os.close(fd)
                te = time.perf_counter()
                sfd = os.open(p, os.O_RDONLY)
                stat_fds.append(sfd)
                od = "NA" if a.dev == "local" else round(cachestat(sfd)[1] * PAGE / MiB, 1)
                cgm = cg_memstat()
                lats.append((te - ta) * 1e3)
                closes.append((te - td) * 1e3)
                if a.write_rows:
                    ow.row(dev=a.dev, mode=mode, tag=a.tag, idx=i, elapsed_s=round(te - t0, 4),
                           t_open_ms=round((tb - ta) * 1e3, 3), t_write_ms=round((tc - tb) * 1e3, 3),
                           t_fsync_ms=round((td - tc) * 1e3, 3), t_close_ms=round((te - td) * 1e3, 3),
                           t_total_ms=round((te - ta) * 1e3, 3), own_dirty_mib=od,
                           mi_dirty_mib=meminfo()["Dirty"], cg_file_dirty_mib=cgm["file_dirty"] // MiB,
                           cg_file_wb_mib=cgm["file_writeback"] // MiB)
            write_s = time.perf_counter() - t0
            t_write_end = time.perf_counter()
            end_write = smp.sample()
            smp.phase = "post"
            t_post0 = end_write["t_s"]
            post30 = None
            while True:
                el = time.perf_counter() - t_write_end
                if post30 is None and el >= 30:
                    post30 = smp.sample()["own_cached_mib"]
                if el >= a.post_s:
                    break
                time.sleep(a.sample_s)
            post_end = smp.sample()
            # 延後版：fdatasync + DONTNEED，量多久能把 DRAM 還回來
            smp.phase = "drop"
            td0 = time.perf_counter()
            for sfd in stat_fds:
                os.fdatasync(sfd)
                os.posix_fadvise(sfd, 0, 0, os.POSIX_FADV_DONTNEED)
            drop_s = time.perf_counter() - td0
            after = smp.sample()
            smp.phase = "after"
            time.sleep(1.0)
            smp.stop()
            # dirty 指標：NFS 用自己檔案的 cachestat；local（overlay）只能用整個 cgroup 的 file_dirty+file_writeback
            def dm(r):
                if r["own_dirty_mib"] != "NA":
                    return r["own_dirty_mib"] + r["own_wb_mib"]
                return r["cg_file_dirty_mib"] + r["cg_file_wb_mib"]
            H = smp.hist
            base_d = max([dm(r) for r in H if r["phase"] == "base"] or [0])
            dirty_peak = max(dm(r) for r in H)
            cached_peak = max(r["own_cached_mib"] for r in H)
            mi_dirty_peak = max(r["mi_dirty_mib"] for r in H)
            cg_dirty_peak = max(r["cg_file_dirty_mib"] for r in H)
            tol = 0 if a.dev == "nfs" else base_d + 64
            t_dirty_zero = next((round(r["t_s"] - t_post0, 2) for r in H
                                 if r["phase"] == "post" and r["t_s"] >= t_post0 and dm(r) <= tol), None)
            # 寫入 dm-1／NFS 的實際位元組（含 writeback）
            if a.dev == "local":
                io1 = io_stat()
                dev_mib = round((io1["wbytes"] - io0["wbytes"]) / MiB, 1)
                nfs_n = nfs_d = nfs_s = ""
            else:
                nb1 = nfs_bytes()
                dev_mib = ""
                nfs_n = round((nb1["normal_write"] - nb0["normal_write"]) / MiB, 1)
                nfs_d = round((nb1["direct_write"] - nb0["direct_write"]) / MiB, 1)
                nfs_s = round((nb1["server_write"] - nb0["server_write"]) / MiB, 1)
            q = statistics.quantiles(lats, n=100, method="inclusive") if len(lats) >= 2 else [lats[0]] * 99
            osum.row(dev=a.dev, mode=mode, tag=a.tag, n_chunks=n, gib=a.gib, write_s=round(write_s, 3),
                     write_GiBps=round(n * CHUNK / GiB / write_s, 3),
                     lat_p50_ms=round(statistics.median(lats), 2), lat_p90_ms=round(q[89], 2),
                     lat_p99_ms=round(q[98], 2), lat_max_ms=round(max(lats), 2),
                     close_p50_ms=round(statistics.median(closes), 2), close_max_ms=round(max(closes), 2),
                     own_cached_end_write_mib=end_write["own_cached_mib"], own_dirty_peak_mib=dirty_peak,
                     own_cached_peak_mib=cached_peak, own_cached_post30_mib=post30,
                     own_cached_post_end_mib=post_end["own_cached_mib"],
                     t_dirty_zero_after_write_s=t_dirty_zero if t_dirty_zero is not None else f">{a.post_s}",
                     drop_s=round(drop_s, 3), own_cached_after_drop_mib=after["own_cached_mib"],
                     dev_write_mib_delta=dev_mib, nfs_normal_write_mib=nfs_n, nfs_direct_write_mib=nfs_d,
                     nfs_server_write_mib=nfs_s, mi_dirty_peak_mib=mi_dirty_peak, cg_file_dirty_peak_mib=cg_dirty_peak,
                     note=("dirty=own cachestat" if a.dev == "nfs" else
                           f"dirty=cgroup file_dirty+writeback (base {base_d} MiB, tol {tol}); own_cached=mincore") +
                          "; dev_write_mib_delta=cgroup io.stat dm-1 wbytes（整個 container）")
            print(f"[pc] {a.dev} {mode}: {n} chunks {write_s:.2f}s p50={statistics.median(lats):.1f}ms "
                  f"max={max(lats):.1f}ms cached_end={end_write['own_cached_mib']} dirty_peak={dirty_peak} "
                  f"post30={post30} dirty0={t_dirty_zero} drop={drop_s:.2f}s after={after['own_cached_mib']}",
                  flush=True)
        finally:
            if smp.is_alive():
                smp.stop_ev.set(); smp.join()
            for sfd in stat_fds:
                try:
                    os.close(sfd)
                except OSError:
                    pass
            shutil.rmtree(root, ignore_errors=True)
    if a.dev == "local":
        shutil.rmtree(dev_root("local"), ignore_errors=True)


# ------------------------------------------------------------------ (c) 讀取方向
class BgWriter(threading.Thread):
    """背景 O_DIRECT 寫 64 MiB chunk，輪流覆寫 k 個檔（不增加占用空間）。"""

    def __init__(self, root, k=8):
        super().__init__(daemon=True)
        self.root, self.k = root, k
        self.buf = aligned(CHUNK)
        fill_random(self.buf, 99)
        self.go = threading.Event()
        self.stop_ev = threading.Event()
        self.bytes = 0
        self.busy_s = 0.0
        self.err = None
        os.makedirs(root, exist_ok=True)

    def run(self):
        i = 0
        try:
            while not self.stop_ev.is_set():
                if not self.go.wait(0.05):
                    continue
                stamp(self.buf, 10_000 + i)
                t = time.perf_counter()
                fd = os.open(f"{self.root}/w{i % self.k}.bin", os.O_WRONLY | os.O_CREAT | os.O_DIRECT, 0o644)
                os.write(fd, self.buf)
                os.close(fd)
                self.busy_s += time.perf_counter() - t
                self.bytes += CHUNK
                i += 1
        except Exception as e:
            self.err = repr(e)

    def stop(self):
        self.stop_ev.set(); self.join()
        if self.err:
            raise RuntimeError(f"背景寫入失敗：{self.err}")


def make_file(path, nbytes):
    buf = aligned(CHUNK)
    fill_random(buf, 3)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
    try:
        for i in range(nbytes // CHUNK):
            stamp(buf, 50_000 + i)
            if os.pwrite(fd, buf, i * CHUNK) != CHUNK:
                raise OSError("short write")
        os.fsync(fd)
    finally:
        os.close(fd)


def evict(path, dev):
    """fdatasync + DONTNEED，回傳之後還在 page cache 的 MiB（local=overlay 用 mincore，NFS 用 cachestat）。"""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fdatasync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        if dev == "local":
            return mincore_pages(path) * PAGE / MiB
        return cachestat(fd)[0] * PAGE / MiB
    finally:
        os.close(fd)


def read_pass(paths, io, direction, mode, region):
    """paths：一個大檔（layout=bigfile）或多個 chunk 檔（layout=files）。回傳 (bytes, secs, per-IO ms list)。"""
    flags = os.O_RDONLY | (os.O_DIRECT if mode == "odirect" else 0)
    buf = aligned(io)
    reqs = []
    if len(paths) == 1:
        offs = list(range(0, region, io))
        if direction == "bwd":
            offs.reverse()
        reqs = [(paths[0], o) for o in offs]
    else:
        ps = list(paths)
        if direction == "bwd":
            ps.reverse()
        for p in ps:
            offs = list(range(0, CHUNK, io))
            if direction == "bwd":
                offs.reverse()
            reqs += [(p, o) for o in offs]
    fds = {p: os.open(p, flags) for p in set(paths)}
    lat = []
    try:
        t0 = time.perf_counter()
        for p, o in reqs:
            ta = time.perf_counter()
            r = os.preadv(fds[p], [buf], o)
            if r != io:
                raise OSError(f"short read {r} != {io} at {p}+{o}")
            lat.append((time.perf_counter() - ta) * 1e3)
        secs = time.perf_counter() - t0
    finally:
        for fd in fds.values():
            os.close(fd)
        buf.close()
    return len(reqs) * io, secs, lat


def cmd_rd(a):
    o = Out("d7_read.csv", ["dev", "layout", "mode", "io_mib", "direction", "rep", "concurrent_write", "region_mib",
                            "bytes", "secs", "MiBps", "io_p50_ms", "io_p99_ms", "own_cached_before_mib",
                            "writer_MiBps", "order_idx"])
    root = dev_root(a.dev)
    os.makedirs(root, exist_ok=True)
    big = f"{root}/big.bin"
    file_bytes = int(a.file_gib * GiB)
    writer = None
    try:
        t = time.perf_counter()
        make_file(big, file_bytes)
        print(f"[rd] {a.dev}: wrote {a.file_gib} GiB in {time.perf_counter() - t:.1f}s", flush=True)
        chunk_paths = []
        if a.files_n:
            buf = aligned(CHUNK)
            fill_random(buf, 5)
            for i in range(a.files_n):
                p = f"{root}/f{i:04d}.bin"
                fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
                stamp(buf, 70_000 + i)
                os.write(fd, buf); os.close(fd)
                chunk_paths.append(p)
        cells = []
        for rep in range(a.reps):
            for io in a.sizes:
                for mode in a.modes:
                    for d in ("fwd", "bwd"):
                        cells.append(("bigfile", rep, io, mode, d, 0))
            if a.files_n:
                for mode in a.modes:
                    for d in ("fwd", "bwd"):
                        cells.append(("files", rep, 64, mode, d, 0))
            for io in a.conc_sizes:
                for d in ("fwd", "bwd"):
                    cells.append(("bigfile", rep, io, "odirect", d, 1))
        rng = random.Random(a.seed)
        # 每個 rep 內部隨機排序（避免時間漂移偏向某個方向），rep 之間照順序
        ordered = []
        for rep in range(a.reps):
            c = [x for x in cells if x[1] == rep]
            rng.shuffle(c)
            ordered += c
        if any(x[5] for x in ordered):
            writer = BgWriter(f"{root}/bgw")
            writer.start()
        for k, (layout, rep, io, mode, d, conc) in enumerate(ordered):
            region = min(file_bytes, int(a.conc_region_gib * GiB)) if conc else file_bytes
            paths = [big] if layout == "bigfile" else chunk_paths
            if mode == "buffered":
                before = round(sum(evict(p, a.dev) for p in paths), 1)
            else:
                before = ""
            wb0 = wt0 = 0
            if conc:
                wb0 = writer.bytes; wt0 = time.perf_counter()
                writer.go.set()
                time.sleep(0.3)  # 讓寫入先進入穩態
            nbytes, secs, lat = read_pass(paths, io * MiB, d, mode, region)
            wmibps = ""
            if conc:
                writer.go.clear()
                wmibps = round((writer.bytes - wb0) / MiB / (time.perf_counter() - wt0), 1)
                time.sleep(0.5)
            q = statistics.quantiles(lat, n=100, method="inclusive") if len(lat) >= 2 else [lat[0]] * 99
            o.row(dev=a.dev, layout=layout, mode=mode, io_mib=io, direction=d, rep=rep, concurrent_write=conc,
                  region_mib=(region if layout == "bigfile" else len(paths) * CHUNK) // MiB, bytes=nbytes,
                  secs=round(secs, 4), MiBps=round(nbytes / MiB / secs, 1), io_p50_ms=round(statistics.median(lat), 3),
                  io_p99_ms=round(q[98], 3), own_cached_before_mib=before, writer_MiBps=wmibps, order_idx=k)
            print(f"[rd] {a.dev} {layout} {mode} io={io} {d} rep={rep} conc={conc}: {nbytes / MiB / secs:.0f} MiB/s",
                  flush=True)
            if mode == "buffered":
                for p in paths:
                    evict(p, a.dev)
    finally:
        if writer is not None and writer.is_alive():
            writer.stop()
        shutil.rmtree(root, ignore_errors=True)


# ------------------------------------------------------------------ (e) 連續寫
def cmd_seqw(a):
    o = Out("d7_seqw.csv", ["dev", "sec", "gib_written", "MiBps", "n_writes", "lat_max_ms", "sda_util_pct",
                            "sda_write_mib"])
    oc = Out("d7_seqw_precheck.csv", ["dev", "window_s", "sda_util_pct", "sda_read_mib", "sda_write_mib"])
    r0, w0, t0d = diskstats(); time.sleep(a.precheck_s); r1, w1, t1d = diskstats()
    util = (t1d - t0d) / (a.precheck_s * 1000) * 100
    oc.row(dev="local", window_s=a.precheck_s, sda_util_pct=round(util, 2), sda_read_mib=round((r1 - r0) * 512 / MiB, 1),
           sda_write_mib=round((w1 - w0) * 512 / MiB, 1))
    print(f"[seqw] precheck sda util={util:.2f}%", flush=True)
    if util > 10:
        print("[seqw] sda 被別人用 >10%：照判準跳過", flush=True)
        return
    root = dev_root("local")
    os.makedirs(root, exist_ok=True)
    p = f"{root}/seq.bin"
    buf = aligned(CHUNK)
    fill_random(buf, 11)
    n = int(a.gib * GiB // CHUNK)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
    try:
        t0 = time.perf_counter()
        t_last = t0
        sec, nw, lmax, b_sec = 0, 0, 0.0, 0
        ds_prev = diskstats()
        for i in range(n):
            stamp(buf, 90_000 + i)
            ta = time.perf_counter()
            if os.pwrite(fd, buf, i * CHUNK) != CHUNK:
                raise OSError("short write")
            tb = time.perf_counter()
            nw += 1; b_sec += CHUNK; lmax = max(lmax, (tb - ta) * 1e3)
            if tb - t0 >= sec + 1 or i == n - 1:
                ds = diskstats()
                span = tb - t_last
                t_last = tb
                o.row(dev="local", sec=sec, gib_written=round((i + 1) * CHUNK / GiB, 3),
                      MiBps=round(b_sec / MiB / span, 1), n_writes=nw, lat_max_ms=round(lmax, 2),
                      sda_util_pct=round((ds[2] - ds_prev[2]) / (span * 1000) * 100, 1),
                      sda_write_mib=round((ds[1] - ds_prev[1]) * 512 / MiB, 1))
                ds_prev = ds
                sec = int(tb - t0); nw = 0; lmax = 0.0; b_sec = 0
        os.fsync(fd)
        print(f"[seqw] {a.gib} GiB in {time.perf_counter() - t0:.1f}s", flush=True)
    finally:
        os.close(fd)
        shutil.rmtree(root, ignore_errors=True)


# ------------------------------------------------------------------ (f) NFS 去重／壓縮
def cmd_dedup(a):
    o = Out("d7_dedup.csv", ["dev", "pattern", "rep", "gib", "secs", "MiBps", "nfs_direct_write_mib"])
    root = dev_root(a.dev)
    os.makedirs(root, exist_ok=True)
    n = int(a.gib * GiB // CHUNK)
    buf = aligned(CHUNK)
    try:
        for rep in range(a.reps):
            for pat in (["unique", "repeat4k"] if rep % 2 == 0 else ["repeat4k", "unique"]):
                if pat == "repeat4k":
                    blk = np.random.default_rng(rep).integers(0, 256, 4096, dtype=np.uint8).tobytes()
                    buf[:] = blk * (CHUNK // 4096)  # 和 code/m7_calib.py:166,189 同樣的寫法
                else:
                    fill_random(buf, 1000 + rep)
                nb0 = nfs_bytes() if a.dev == "nfs" else None
                t0 = time.perf_counter()
                for i in range(n):
                    if pat == "unique":
                        stamp(buf, rep * 100_000 + i)
                    fd = os.open(f"{root}/{pat}_{i:04d}.bin", os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
                    os.write(fd, buf); os.close(fd)
                secs = time.perf_counter() - t0
                dw = round((nfs_bytes()["direct_write"] - nb0["direct_write"]) / MiB, 1) if nb0 else ""
                o.row(dev=a.dev, pattern=pat, rep=rep, gib=a.gib, secs=round(secs, 3),
                      MiBps=round(n * CHUNK / MiB / secs, 1), nfs_direct_write_mib=dw)
                print(f"[dedup] {a.dev} {pat} rep={rep}: {n * CHUNK / MiB / secs:.0f} MiB/s", flush=True)
                for i in range(n):
                    os.remove(f"{root}/{pat}_{i:04d}.bin")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ------------------------------------------------------------------ 多執行緒 buffered（只進 page cache，不落盤）
def cmd_bufmt(a):
    """T 條執行緒 buffered 寫 64 MiB chunk 檔，寫完立刻刪掉（在 dirty_expire 30 s 之前），
    所以資料只進 page cache、不會被寫到磁碟。用 cgroup io.stat 的 dm-1 wbytes 確認幾乎沒有落盤。
    目的：量「寫進 page cache」本身的速度（多條執行緒時會不會超過磁碟的 O_DIRECT 速度）。"""
    o = Out("d7_bufmt.csv", ["dev", "threads", "gib", "rep", "secs", "GiBps", "lat_p50_ms", "lat_p99_ms",
                             "lat_max_ms", "cg_dirty_peak_mib", "dm1_wbytes_delta_mib", "delete_s",
                             "first_write_to_delete_s"])
    root = dev_root("local")
    for rep in range(a.reps):
        for T in a.threads:
            d = f"{root}/t{T}_r{rep}"
            os.makedirs(d, exist_ok=True)
            n = int(a.gib * GiB // CHUNK)
            lats = []
            lk = threading.Lock()
            peak = [0]

            def worker(tid):
                b = aligned(CHUNK)
                fill_random(b, 500 + tid)
                for i in range(tid, n, T):
                    stamp(b, 200_000 + i)
                    ta = time.perf_counter()
                    fd = os.open(f"{d}/c{i:05d}.bin", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
                    if os.write(fd, b) != CHUNK:
                        raise OSError("short write")
                    os.close(fd)
                    with lk:
                        lats.append((time.perf_counter() - ta) * 1e3)
                b.close()

            io0 = io_stat()
            ths = [threading.Thread(target=worker, args=(t,)) for t in range(T)]
            t0 = time.perf_counter()
            for t in ths:
                t.start()
            while any(t.is_alive() for t in ths):
                cg = cg_memstat()
                peak[0] = max(peak[0], (cg["file_dirty"] + cg["file_writeback"]) // MiB)
                time.sleep(0.05)
            for t in ths:
                t.join()
            secs = time.perf_counter() - t0
            if len(lats) != n:
                raise RuntimeError(f"只完成 {len(lats)}/{n} 個 write（某條執行緒失敗）")
            cg = cg_memstat()
            peak[0] = max(peak[0], (cg["file_dirty"] + cg["file_writeback"]) // MiB)
            td = time.perf_counter()
            shutil.rmtree(d)
            delete_s = time.perf_counter() - td
            time.sleep(1.0)
            io1 = io_stat()
            q = statistics.quantiles(lats, n=100, method="inclusive")
            o.row(dev="local", threads=T, gib=a.gib, rep=rep, secs=round(secs, 3),
                  GiBps=round(n * CHUNK / GiB / secs, 3), lat_p50_ms=round(statistics.median(lats), 2),
                  lat_p99_ms=round(q[98], 2), lat_max_ms=round(max(lats), 2), cg_dirty_peak_mib=peak[0],
                  dm1_wbytes_delta_mib=round((io1["wbytes"] - io0["wbytes"]) / MiB, 1),
                  delete_s=round(delete_s, 3), first_write_to_delete_s=round(td + delete_s - t0, 2))
            print(f"[bufmt] T={T} rep={rep}: {n * CHUNK / GiB / secs:.2f} GiB/s p50={statistics.median(lats):.1f}ms "
                  f"dm1_w={(io1['wbytes'] - io0['wbytes']) / MiB:.0f}MiB", flush=True)
    shutil.rmtree(root, ignore_errors=True)


# ------------------------------------------------------------------ 熱讀（只讀，不寫任何東西）
def cmd_warm(a):
    """讀一個「已經 100% 在 page cache」的既有檔（NFS 上的模型權重），量 buffered 熱讀吞吐量。
    只讀、不寫、不 DONTNEED（別人的檔不能丟），前後用 cachestat 確認仍是 100%。"""
    o = Out("d7_warm.csv", ["path", "size_mib", "io_mib", "rep", "cached_frac_before", "cached_frac_after",
                            "secs", "MiBps", "io_p50_ms"])
    size = os.path.getsize(a.path)
    for rep in range(a.reps):
        for io in a.sizes:
            fd = os.open(a.path, os.O_RDONLY)
            c0 = cachestat(fd)[0] * PAGE / size
            buf = aligned(io * MiB)
            n = size // (io * MiB)
            lat = []
            t0 = time.perf_counter()
            for i in range(n):
                ta = time.perf_counter()
                os.preadv(fd, [buf], i * io * MiB)
                lat.append((time.perf_counter() - ta) * 1e3)
            secs = time.perf_counter() - t0
            c1 = cachestat(fd)[0] * PAGE / size
            os.close(fd); buf.close()
            o.row(path=a.path, size_mib=size // MiB, io_mib=io, rep=rep, cached_frac_before=round(c0, 4),
                  cached_frac_after=round(c1, 4), secs=round(secs, 4), MiBps=round(n * io / secs, 1),
                  io_p50_ms=round(statistics.median(lat), 3))
            print(f"[warm] io={io} rep={rep} cached {c0:.3f}->{c1:.3f}: {n * io / secs:.0f} MiB/s", flush=True)


# ------------------------------------------------------------------ 彙整（只讀 results 裡的原始 CSV）
def cmd_analyze(a):
    """從原始 CSV 算：讀取方向表、干擾倍數、寫入延遲（inclusive 分位數；d7_pc_summary 的 p99 用了
    exclusive 法，會比 max 還大，以這裡為準）。每列的 run_id 指回原始 run。"""
    from collections import defaultdict

    def load(name):
        return list(csv.DictReader(open(f"{RES}/{name}")))

    def q(v, k):
        return statistics.quantiles(v, n=100, method="inclusive")[k - 1] if len(v) > 1 else v[0]

    # 1) 讀取方向
    o = Out("d7_read_summary.csv", ["src_run_id", "dev", "layout", "mode", "io_mib", "concurrent_write", "n",
                                    "fwd_med", "fwd_min", "fwd_max", "bwd_med", "bwd_min", "bwd_max",
                                    "bwd_vs_fwd_pct", "ranges_separate"])
    g = defaultdict(list)
    for r in load("d7_read.csv"):
        g[(r["run_id"], r["dev"], r["layout"], r["mode"], int(r["io_mib"]), r["concurrent_write"],
           r["direction"])].append(float(r["MiBps"]))
    for k in sorted({k[:6] for k in g}):
        f, b = g[k + ("fwd",)], g[k + ("bwd",)]
        mf, mb = statistics.median(f), statistics.median(b)
        o.row(src_run_id=k[0], dev=k[1], layout=k[2], mode=k[3], io_mib=k[4], concurrent_write=k[5], n=len(f),
              fwd_med=round(mf, 1), fwd_min=min(f), fwd_max=max(f), bwd_med=round(mb, 1), bwd_min=min(b),
              bwd_max=max(b), bwd_vs_fwd_pct=round((mb - mf) / mf * 100, 1),
              ranges_separate=int(min(f) > max(b) or min(b) > max(f)))
    # 2) 背景寫入讓讀取慢幾倍（O_DIRECT、bigfile、同方向）
    o2 = Out("d7_interference.csv", ["src_run_id", "dev", "io_mib", "direction", "alone_med_MiBps",
                                     "with_write_med_MiBps", "slowdown_x", "writer_med_MiBps"])
    wr = defaultdict(list)
    for r in load("d7_read.csv"):
        if r["writer_MiBps"]:
            wr[(r["run_id"], r["dev"], int(r["io_mib"]), r["direction"])].append(float(r["writer_MiBps"]))
    for k in sorted(wr):
        rid, dev, io, d = k
        alone = g[(rid, dev, "bigfile", "odirect", io, "0", d)]
        withw = g[(rid, dev, "bigfile", "odirect", io, "1", d)]
        o2.row(src_run_id=rid, dev=dev, io_mib=io, direction=d, alone_med_MiBps=statistics.median(alone),
               with_write_med_MiBps=statistics.median(withw),
               slowdown_x=round(statistics.median(alone) / statistics.median(withw), 2),
               writer_med_MiBps=statistics.median(wr[k]))
    # 3) 寫入延遲
    o3 = Out("d7_pc_latency.csv", ["src_run_id", "dev", "mode", "n", "p50_ms", "p90_ms", "p99_ms", "max_ms",
                                   "p99_over_p50", "write_p50_ms", "fsync_p50_ms", "close_p50_ms", "n_over_1s",
                                   "q1_p50_ms", "q2_p50_ms", "q3_p50_ms", "q4_p50_ms"])
    w = defaultdict(list)
    for r in load("d7_pc_writes.csv"):
        w[(r["run_id"], r["dev"], r["mode"])].append(r)
    for k in sorted(w):
        rs = w[k]
        t = [float(x["t_total_ms"]) for x in rs]
        n4 = len(t) // 4
        quart = [round(statistics.median(t[i * n4:(i + 1) * n4]), 2) for i in range(4)]
        o3.row(src_run_id=k[0], dev=k[1], mode=k[2], n=len(t), p50_ms=round(statistics.median(t), 2),
               p90_ms=round(q(t, 90), 2), p99_ms=round(q(t, 99), 2), max_ms=round(max(t), 2),
               p99_over_p50=round(q(t, 99) / statistics.median(t), 2),
               write_p50_ms=round(statistics.median([float(x["t_write_ms"]) for x in rs]), 2),
               fsync_p50_ms=round(statistics.median([float(x["t_fsync_ms"]) for x in rs]), 2),
               close_p50_ms=round(statistics.median([float(x["t_close_ms"]) for x in rs]), 2),
               n_over_1s=sum(v > 1000 for v in t), q1_p50_ms=quart[0], q2_p50_ms=quart[1], q3_p50_ms=quart[2],
               q4_p50_ms=quart[3])
    print("[analyze] wrote d7_read_summary.csv, d7_interference.csv, d7_pc_latency.csv", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("env")
    p = sp.add_parser("pc")
    p.add_argument("--dev", required=True, choices=["local", "nfs"])
    p.add_argument("--modes", nargs="+", default=["odirect", "buffered", "fsync"])
    p.add_argument("--gib", type=float, default=8)
    p.add_argument("--base-s", type=float, default=3)
    p.add_argument("--post-s", type=float, default=45)
    p.add_argument("--sample-s", type=float, default=0.2)
    p.add_argument("--tag", default="")
    p.add_argument("--write-rows", type=int, default=1)
    p.add_argument("--write-rows-to-results", type=int, default=1)
    p = sp.add_parser("rd")
    p.add_argument("--dev", required=True, choices=["local", "nfs"])
    p.add_argument("--file-gib", type=float, default=8)
    p.add_argument("--sizes", nargs="+", type=int, default=[2, 8, 32, 64, 256])
    p.add_argument("--modes", nargs="+", default=["odirect", "buffered"])
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--files-n", type=int, default=32)
    p.add_argument("--conc-sizes", nargs="*", type=int, default=[2, 64])
    p.add_argument("--conc-region-gib", type=float, default=2)
    p.add_argument("--seed", type=int, default=0)
    p = sp.add_parser("seqw")
    p.add_argument("--gib", type=float, default=32)
    p.add_argument("--precheck-s", type=float, default=10)
    p = sp.add_parser("dedup")
    p.add_argument("--dev", default="nfs", choices=["local", "nfs"])
    p.add_argument("--gib", type=float, default=2)
    p.add_argument("--reps", type=int, default=2)
    sp.add_parser("analyze")
    p = sp.add_parser("bufmt")
    p.add_argument("--threads", nargs="+", type=int, default=[1, 4, 16])
    p.add_argument("--gib", type=float, default=4)
    p.add_argument("--reps", type=int, default=2)
    p = sp.add_parser("warm")
    p.add_argument("--path", required=True)
    p.add_argument("--sizes", nargs="+", type=int, default=[2, 64])
    p.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    print(f"[d7] RUN_ID={RUN_ID} cmd={a.cmd} args={vars(a)}", flush=True)
    {"env": cmd_env, "pc": cmd_pc, "rd": cmd_rd, "seqw": cmd_seqw, "dedup": cmd_dedup,
     "warm": cmd_warm, "bufmt": cmd_bufmt, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
