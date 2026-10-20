"""diskstats_sampler.py — 每 dt 秒記一筆 /proc/diskstats 的某顆碟（第三階段 3090 校準，2026-10-18）。

為什麼：disk_bw.py 只給整段平均，而且看不到同一顆碟上別人的 I/O。
  - 逐秒的裝置寫入速度看得出 QLC 的 SLC 快取什麼時候用完（懸崖）。
  - 整段裝置總量 − 我們自己寫／讀的量 ＝ 別人（＋ext4 journal/metadata）的量，照實記。
O_DIRECT 的寫入不經 page cache，所以 diskstats 的速度就是裝置的速度。

  python code/p3cal/diskstats_sampler.py --disk sdg --dt 1 --out $RUN_DIR/diskstats.csv &
  （SIGTERM 結束；每列 flush）
"""
from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from datetime import datetime

# /proc/diskstats 欄位（kernel Documentation/admin-guide/iostats.rst），p[0..2]=major minor name
F = {"rd_ios": 3, "rd_merges": 4, "rd_sectors": 5, "rd_ticks": 6, "wr_ios": 7, "wr_merges": 8,
     "wr_sectors": 9, "wr_ticks": 10, "in_flight": 11, "io_ticks": 12, "time_in_queue": 13}


def read(disk):
    for line in open("/proc/diskstats"):
        p = line.split()
        if p[2] == disk:
            return {k: int(p[i]) for k, i in F.items()}
    raise RuntimeError(f"/proc/diskstats 沒有 {disk}")   # 查不到不是 0（CLAUDE.md 規則 7）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disk", required=True)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    stop = {"v": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("v", True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("v", True))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["ts", "t_mono", "disk", *F.keys(), "loadavg1", "procs_running"])
        w.writeheader()
        while not stop["v"]:
            r = read(a.disk)
            la = open("/proc/loadavg").read().split()
            w.writerow({"ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                        "t_mono": round(time.monotonic(), 3), "disk": a.disk, **r,
                        "loadavg1": la[0], "procs_running": la[3].split("/")[0]})
            fh.flush()
            time.sleep(a.dt)
        r = read(a.disk)
        w.writerow({"ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                    "t_mono": round(time.monotonic(), 3), "disk": a.disk, **r,
                    "loadavg1": open("/proc/loadavg").read().split()[0], "procs_running": ""})
    return 0


if __name__ == "__main__":
    sys.exit(main())
