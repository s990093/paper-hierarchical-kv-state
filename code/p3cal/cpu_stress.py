"""cpu_stress.py — 主機 CPU／記憶體頻寬的壓力：N 個執行緒各自不停做大塊 numpy 記憶體複製（第三階段 3090 校準，2026-10-18）。

為什麼：第三階段地圖（04_prereg_map v2 §5）的「CPU 忙」敏感度要一個係數：主機忙的時候，CPU→GPU 的搬運慢多少。
共用機上別人的 CPU 負載看得到（loadavg）但控制不了，所以自己造一個「已知大小」的負載。

做法：
  - 一個 process、N 個 threading.Thread；每個執行緒自己配 src、dst（各 --mib MiB，預設 256 MiB，遠大於 L3 的 35.75 MB），
    先寫過一遍（first-touch：頁面配在執行緒當時所在的 NUMA 節點；--cpus 限定 CPU 時就在那個節點）。
  - 迴圈 np.copyto(dst, src)。numpy 的連續陣列 copy 會放開 GIL（raw_array_assign_array 的 NPY_BEGIN_THREADS_THRESHOLDED），
    所以 N 個執行緒真的同時在搬；實際做到的總複製速度每秒記一筆（--report），讓呼叫者確認負載真的有放大。
  - 「複製速度」＝每秒複製完成的位元組（一次 copy B 位元組 ⇒ 記 B）。記憶體控制器上的流量約 2–3 倍（讀 src＋寫 dst
    ＋寫入前的讀取（write-allocate）），這裡不換算、只照實記。
  - 啟動完成印 READY（呼叫者等這行）；SIGTERM／SIGINT 結束，印 DONE {json}。

  python code/p3cal/cpu_stress.py --threads 16 --cpus 0-23,48-71 --report $RUN_DIR/stress_16.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import threading
import time
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "1")       # 只要我們自己的 N 個執行緒，不要 numpy 底下的 BLAS 執行緒
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np  # noqa: E402

MiB = 1 << 20
GiB = 1 << 30


def parse_cpus(s: str | None):
    if not s:
        return None
    out = set()
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--mib", type=int, default=256, help="每個執行緒的 src、dst 各多大（MiB）")
    ap.add_argument("--cpus", default=None, help="CPU 清單（例 0-23,48-71）；不給＝不限定")
    ap.add_argument("--report", default=None, help="每秒一列的 CSV")
    ap.add_argument("--duration", type=float, default=0, help=">0：跑這麼多秒就自己結束（自測用）")
    a = ap.parse_args()
    cpus = parse_cpus(a.cpus)
    if cpus:
        os.sched_setaffinity(0, cpus)                 # 執行緒會繼承（Linux 上 sched_setaffinity(0) 只設呼叫的執行緒；
        # 所以每個 worker 也自己設一次，見下面）
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    n = a.mib * MiB
    counts = [0] * a.threads
    ready = threading.Barrier(a.threads + 1)
    errs = []

    def worker(k):
        try:
            if cpus:
                os.sched_setaffinity(0, cpus)
            src = np.empty(n, dtype=np.uint8)
            dst = np.empty(n, dtype=np.uint8)
            src.fill(k & 0xFF)                        # first-touch
            dst.fill(0)
        except Exception as e:  # noqa: BLE001
            errs.append(repr(e))
            ready.abort()
            return
        ready.wait()
        while not stop.is_set():
            np.copyto(dst, src)
            counts[k] += 1

    ths = [threading.Thread(target=worker, args=(k,), daemon=True) for k in range(a.threads)]
    for t in ths:
        t.start()
    try:
        ready.wait()
    except threading.BrokenBarrierError:
        print("FAILED " + json.dumps({"errors": errs}), flush=True)
        return 2
    t0 = time.monotonic()
    print(f"READY threads={a.threads} mib={a.mib} cpus={a.cpus or 'ALL'} pid={os.getpid()}", flush=True)
    fh = open(a.report, "w", newline="") if a.report else None
    w = None
    if fh:
        w = csv.DictWriter(fh, fieldnames=["ts", "t_s", "threads", "copies_total", "copy_GiBps_1s", "loadavg1",
                                           "procs_running"])
        w.writeheader()
    last_c, last_t = 0, t0
    rates = []
    while not stop.is_set():
        stop.wait(1.0)
        now = time.monotonic()
        c = sum(counts)
        r = (c - last_c) * n / (now - last_t) / GiB
        rates.append(r)
        last_c, last_t = c, now
        if w:
            la = open("/proc/loadavg").read().split()
            w.writerow({"ts": datetime.now().astimezone().isoformat(timespec="milliseconds"), "t_s": round(now - t0, 3),
                        "threads": a.threads, "copies_total": c, "copy_GiBps_1s": round(r, 3), "loadavg1": la[0],
                        "procs_running": la[3].split("/")[0]})
            fh.flush()
        if a.duration and now - t0 >= a.duration:
            stop.set()
    for t in ths:
        t.join(timeout=10)
    dur = time.monotonic() - t0
    tot = sum(counts) * n
    if fh:
        fh.close()
    print("DONE " + json.dumps({"threads": a.threads, "mib": a.mib, "cpus": a.cpus or "ALL", "seconds": round(dur, 3),
                                "copied_GiB": round(tot / GiB, 3), "mean_copy_GiBps": round(tot / dur / GiB, 3),
                                "median_1s_GiBps": round(sorted(rates)[len(rates) // 2], 3) if rates else None}),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
