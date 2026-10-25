#!/usr/bin/env python3
"""gfio.py — 第三階段 G-fio（mechanisms/06 §2）：hgx-cn01 的 /tmp（md0）與 /work 的循序／隨機讀；`mount`、`df -T`。
只是把算術用的下層常數換成 H200 實測（arith 的 NVMe 是 MI300X 的 6.981 GiB/s）。這台沒有 fio：用 O_DIRECT 的 os.preadv 自己量
（繞過 page cache；每次讀之前另外用 posix_fadvise DONTNEED 再保險一次）。量時間：獨佔（runner 檢查）。

  run --dir D [--dir D2] --gib G --out CSV     每個目錄寫一個 G GiB 的測試檔（O_DIRECT 寫）→ 循序讀（1／4／64 MiB）、隨機讀（128 KiB、
                                               2 MiB、64 MiB），1／4／8 個執行緒 → 刪掉測試檔
  --self-test                                  CPU：對齊的緩衝、偏移產生
"""
from __future__ import annotations

import argparse
import csv
import json
import mmap
import os
import random
import subprocess
import sys
import threading
import time
from datetime import datetime

import numa_probe as NP                          # 只記錄：job 的 cpuset 與這個 job 的卡在哪個 NUMA 節點（gfio 不用 pinned 記憶體）

FIELDS = ["run_id", "ts", "dir", "fs", "pattern", "io_bytes", "threads", "total_bytes", "t_s", "gibps", "iops", "file_gib",
          "n_rep", "sigma_rel", "delta_star", "gibps_reps"]
NM = "NOT_MEASURED"


def noise(vals: list) -> tuple:
    """05 §2.1：(sigma_rel, delta_star)；少於 6 次 → NOT_MEASURED。"""
    import statistics
    v = [float(x) for x in vals]
    if len(v) < 6 or statistics.median(v) == 0:
        return NM, NM
    sr = statistics.stdev(v) / statistics.median(v)
    return round(sr, 6), round(max(0.05, 2 * sr), 6)
MiB = 1 << 20


def run_id():
    return os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc"


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def aligned(n: int):
    return mmap.mmap(-1, n)                      # 頁對齊（O_DIRECT 要求）


def write_file(path: str, gib: float):
    n = int(gib * (1 << 30)) // (64 * MiB) * (64 * MiB)
    buf = aligned(64 * MiB)
    buf.write(os.urandom(64 * MiB))
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o600)
    try:
        off = 0
        while off < n:
            os.pwritev(fd, [buf], off)
            off += 64 * MiB
        os.fsync(fd)
    finally:
        os.close(fd)
    return n


def offsets(size: int, io: int, pattern: str, total: int, seed: int) -> list:
    k = total // io
    if pattern == "seq":
        return [(i * io) % (size - io + 1) // io * io for i in range(k)]
    r = random.Random(seed)
    return [r.randrange(0, size // io) * io for _ in range(k)]


def read_test(path: str, size: int, io: int, pattern: str, threads: int, total: int) -> float:
    try:
        fd0 = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd0, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd0)
    except OSError:
        pass
    offs = offsets(size, io, pattern, total, seed=io * 31 + threads)
    parts = [offs[i::threads] for i in range(threads)]
    errs = []

    def worker(lst):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        buf = aligned(io)
        try:
            for o in lst:
                if os.preadv(fd, [buf], o) != io:
                    errs.append(f"short read at {o}")
        finally:
            os.close(fd)
    th = [threading.Thread(target=worker, args=(p,)) for p in parts]
    t0 = time.perf_counter()
    for t in th:
        t.start()
    for t in th:
        t.join()
    dt = time.perf_counter() - t0
    if errs:
        raise RuntimeError(errs[:3])
    return dt


def fs_info(d: str) -> dict:
    def sh(c):
        return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
    return dict(dir=d, df_T=sh(f"df -T {d} | tail -1"), findmnt=sh(f"findmnt -no SOURCE,FSTYPE,OPTIONS -T {d}"))


def cmd_run(a):
    rid, rows, info = run_id(), [], []
    numa = NP.probe()
    print("[gfio] NUMA：", json.dumps(dict(cpus=numa["job"].get("affinity"), cpuset_nodes=numa["job"].get("cpuset_nodes"),
                                           gpus=numa["gpus_visible"]), ensure_ascii=False), flush=True)
    gibs = a.gib if len(a.gib) == len(a.dir) else None
    if gibs is None:
        raise SystemExit(f"--dir {len(a.dir)} 個、--gib {len(a.gib)} 個：要一對一")
    tots = a.total_gib if a.total_gib and len(a.total_gib) == len(a.dir) else [4.0] * len(a.dir)
    for d, gib, tot_gib in zip(a.dir, gibs, tots):
        os.makedirs(d, exist_ok=True)
        fi = fs_info(d)
        info.append(fi)
        path = os.path.join(d, f"gfio_{rid}.bin")
        try:
            size = write_file(path, gib)
            for pattern, ios in (("seq", (1 * MiB, 4 * MiB, 64 * MiB)), ("rand", (128 * 1024, 2 * MiB, 64 * MiB))):
                for io in ios:
                    for thr in (1, 4, 8):
                        total = min(size, max(io * thr * 4, int(tot_gib * (1 << 30)) // io * io))
                        dts = [read_test(path, size, io, pattern, thr, total) for _ in range(a.reps)]
                        g = [total / dt / (1 << 30) for dt in dts]
                        import statistics
                        sr, ds = noise(g)
                        dt = statistics.median(dts)
                        rows.append(dict(run_id=rid, ts=ts(), dir=d, fs=fi["findmnt"], pattern=pattern, io_bytes=io, threads=thr,
                                         total_bytes=total, t_s=round(dt, 4), gibps=round(statistics.median(g), 3),
                                         iops=round(total / io / dt, 1), file_gib=round(size / (1 << 30), 2), n_rep=len(g),
                                         sigma_rel=sr, delta_star=ds, gibps_reps=";".join(f"{x:.3f}" for x in g)))
                        print(json.dumps(rows[-1]), flush=True)
        finally:
            if os.path.exists(path):
                os.remove(path)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    json.dump(dict(run_id=rid, ts=ts(), fs=info, numa=numa), open(os.path.splitext(a.out)[0] + "_fs.json", "w"), indent=1, ensure_ascii=False)


def self_test() -> bool:
    b = aligned(4096)
    ok = (len(b) == 4096)
    o = offsets(64 * MiB, 2 * MiB, "seq", 8 * MiB, 0)
    ok &= o == [0, 2 * MiB, 4 * MiB, 6 * MiB]
    r = offsets(64 * MiB, 2 * MiB, "rand", 8 * MiB, 0)
    ok &= len(r) == 4 and all(x % (2 * MiB) == 0 and x <= 62 * MiB for x in r)
    print("self-test", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("run")
    s.add_argument("--dir", action="append", required=True)
    s.add_argument("--gib", type=float, action="append", required=True, help="每個 --dir 一個（同樣順序）")
    s.add_argument("--total-gib", type=float, action="append", help="每個 --dir 一個：每次讀多少（預設 4 GiB）")
    s.add_argument("--reps", type=int, default=6, help="每格讀幾次（05 §2.1：σ 要 ≥6 次）")
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    cmd_run(a)


if __name__ == "__main__":
    main()
