#!/usr/bin/env python3
"""`make nvtop` 的合併畫面：所有 RUNNING 的 job 的卡畫在同一個畫面（像 nvtop 的多卡）。

每個 slurm job 的 cgroup 只放行自己那張卡、計算節點又不能 ssh，所以單一 nvtop 只看得到一張。
這裡改成每個 job 開一個常駐的 `srun --overlap nvidia-smi --query-gpu ... -lms`（只讀 NVML，
不在 GPU 上開 process），讀回來合併畫。每 30 秒重掃 squeue，新開跑的 job 自動加進來。

  python3 code/h200_gputop.py [JOBID ...] [--interval-ms 1000]
  q 離開。要看行程表或 nvtop 原本的畫面：make nvtop JOB=<id>
"""
import argparse
import curses
import getpass
import os
import subprocess
import threading
import time
from collections import deque

SLURM_BIN = os.environ.get("SLURM_BIN", "/cm/local/apps/slurm/current/bin")
FIELDS = ["name", "pci.bus_id", "utilization.gpu", "utilization.memory", "memory.used",
          "memory.total", "power.draw", "power.limit", "temperature.gpu", "clocks.sm"]
HIST = 400
SPARK = " ▁▂▃▄▅▆▇█"


def running_jobs():
    out = subprocess.run([f"{SLURM_BIN}/squeue", "-h", "-u", getpass.getuser(), "-t", "R",
                          "-o", "%i|%j|%N"], capture_output=True, text=True).stdout
    return {l.split("|")[0]: l.split("|") for l in out.split("\n") if l}


def num(s):
    try:
        return float(s)
    except ValueError:  # [N/A]、[Not Supported]
        return None


class Feed:
    """一個 job 一條常駐 srun；最新一列與歷史放在這裡，由背景執行緒更新。"""

    def __init__(self, job, name, node, interval_ms):
        self.job, self.name, self.node = job, name, node
        self.cur, self.err, self.alive = None, "", True
        self.util, self.mem = deque(maxlen=HIST), deque(maxlen=HIST)
        cmd = [f"{SLURM_BIN}/srun", f"--jobid={job}", "--overlap", "-n1", "--quiet",
               "nvidia-smi", f"--query-gpu={','.join(FIELDS)}",
               "--format=csv,noheader,nounits", f"-lms={interval_ms}"]
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, bufsize=1)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            v = [x.strip() for x in line.split(",")]
            if len(v) != len(FIELDS):
                continue
            d = dict(zip(FIELDS, v))
            self.cur = d
            self.util.append(num(d["utilization.gpu"]) or 0)
            mu, mt = num(d["memory.used"]), num(d["memory.total"])
            self.mem.append(100 * mu / mt if mu is not None and mt else 0)
        self.alive = False
        self.err = (self.p.stderr.read() or "").strip().splitlines()[-1:] or ["srun 結束"]
        self.err = self.err[0]

    def stop(self):
        if self.p.poll() is None:
            self.p.terminate()


def color_for(pct):
    return 3 if pct >= 90 else 2 if pct >= 50 else 1


def bar(win, y, x, w, pct, label, text):
    """`GPU [||||||   ] 97%` 樣式；w 為整條寬度。"""
    if w < 12:
        return
    win.addstr(y, x, f"{label} [")
    inner = w - len(label) - 3
    fill = int(round(inner * min(max(pct, 0), 100) / 100))
    win.addstr(y, x + len(label) + 2, "|" * fill, curses.color_pair(color_for(pct)) | curses.A_BOLD)
    t = text[:inner]
    win.addstr(y, x + len(label) + 2 + inner - len(t), t, curses.A_BOLD)
    win.addstr(y, x + w - 1, "]")


def spark(win, y, x, w, hist, pair):
    if w <= 0:
        return
    vals = list(hist)[-w:]
    s = "".join(SPARK[min(8, int(round(v / 100 * 8)))] for v in vals)
    win.addstr(y, x + w - len(s), s, curses.color_pair(pair))


def draw(scr, feeds, interval_ms):
    scr.erase()
    H, W = scr.getmaxyx()
    W -= 1  # 最後一欄寫入會觸發 curses 錯誤
    hdr = f" gputop · {len(feeds)} 張卡（每 job 一張）· {interval_ms} ms · q 離開 · 行程表：make nvtop JOB=<id>"
    scr.addstr(0, 0, hdr[:W].ljust(W), curses.A_REVERSE)
    y = 2
    barw = min(60, max(24, W // 2 - 2))
    for f in feeds:
        if y + 3 >= H:
            scr.addstr(H - 1, 0, f" …還有卡畫不下（終端機太矮）"[:W], curses.A_DIM)
            break
        d = f.cur
        title = f"{f.job} {f.name}"
        if d is None:
            msg = "等第一筆資料…" if f.alive else f"讀不到：{f.err}"
            scr.addstr(y, 1, f"{title} · {f.node}  {msg}"[:W - 1], curses.color_pair(4) | curses.A_BOLD)
            y += 4
            continue
        info = (f"{d['name']}  {d['pci.bus_id'][-7:]}  {d['temperature.gpu']}°C  "
                f"{num(d['power.draw']) or 0:.0f}/{num(d['power.limit']) or 0:.0f} W  {d['clocks.sm']} MHz")
        state = "" if f.alive else f"  [已停止：{f.err}]"
        scr.addstr(y, 1, f"{title} · {f.node}"[:W - 1], curses.color_pair(4) | curses.A_BOLD)
        rest = W - 1 - (len(title) + len(f.node) + 6)
        if rest > 10:
            scr.addstr(y, W - min(rest, len(info + state)), (info + state)[-rest:])
        u = num(d["utilization.gpu"]) or 0
        mu, mt = num(d["memory.used"]) or 0, num(d["memory.total"]) or 1
        mp = 100 * mu / mt
        bar(scr, y + 1, 2, barw, u, "GPU", f"{u:.0f}%")
        bar(scr, y + 2, 2, barw, mp, "MEM", f"{mu / 1024:.1f}/{mt / 1024:.0f}G {mp:.0f}%")
        sw = W - barw - 5
        spark(scr, y + 1, barw + 4, sw, f.util, color_for(u))
        spark(scr, y + 2, barw + 4, sw, f.mem, color_for(mp))
        y += 4
    scr.refresh()


def main(scr, a):
    curses.curs_set(0)
    curses.use_default_colors()
    for i, c in enumerate([curses.COLOR_GREEN, curses.COLOR_YELLOW, curses.COLOR_RED, curses.COLOR_CYAN], 1):
        curses.init_pair(i, c, -1)
    scr.timeout(250)
    feeds, last_scan = {}, 0
    try:
        while True:
            if not a.jobs and time.time() - last_scan > 30:
                run = running_jobs()
                for j in [j for j in feeds if j not in run and not feeds[j].alive]:
                    del feeds[j]  # job 已結束
                for j, (_, name, node) in run.items():
                    if j not in feeds or not feeds[j].alive:
                        feeds[j] = Feed(j, name, node, a.interval_ms)
                last_scan = time.time()
            elif a.jobs and not feeds:
                info = running_jobs()
                for j in a.jobs:
                    _, name, node = info.get(j, (j, "?", "?"))
                    feeds[j] = Feed(j, name, node, a.interval_ms)
            draw(scr, [feeds[j] for j in sorted(feeds, key=int)], a.interval_ms)
            if scr.getch() in (ord("q"), ord("Q"), 27):
                break
    finally:
        for f in feeds.values():
            f.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="*")
    ap.add_argument("--interval-ms", type=int, default=1000)
    a = ap.parse_args()
    if not a.jobs and not running_jobs():
        raise SystemExit("沒有正在跑的 job（make status 看）")
    curses.wrapper(main, a)
