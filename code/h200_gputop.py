#!/usr/bin/env python3
"""`make nvtop` 的合併畫面：所有 RUNNING 的 job 的卡畫在同一個畫面（像 nvtop 的多卡）。

每個 slurm job 的 cgroup 只放行自己那張卡、計算節點又不能 ssh，所以單一 nvtop 只看得到一張。
這裡改成每個 job 開一個常駐的 `srun --overlap nvidia-smi --query-gpu ... -lms`（只讀 NVML，
不在 GPU 上開 process），讀回來合併畫。每 30 秒重掃 squeue，新開跑的 job 自動加進來。

畫面仿 btop／gpur：一張卡一個圓角框，左邊漸層條（⅛ 格精度），右邊 braille 歷史圖（一欄兩個取樣）。

  python3 code/h200_gputop.py [JOBID ...] [--interval-ms 1000]
  q 離開；m 切換歷史圖（GPU 使用率 ↔ 顯存）。要看行程表：make nvtop JOB=<id>
"""
import argparse
import ctypes
import curses
import getpass
import locale
import os
import signal
import subprocess
import threading
import time
import unicodedata
from collections import deque

SLURM_BIN = os.environ.get("SLURM_BIN", "/cm/local/apps/slurm/current/bin")
FIELDS = ["name", "pci.bus_id", "utilization.gpu", "utilization.memory", "memory.used",
          "memory.total", "power.draw", "power.limit", "temperature.gpu", "clocks.sm"]
HIST = 600
EIGHTHS = "▏▎▍▌▋▊▉"
# braille 點位：左欄由下往上 7,3,2,1、右欄 8,6,5,4
BR_L, BR_R = [0x40, 0x04, 0x02, 0x01], [0x80, 0x20, 0x10, 0x08]
# 綠 → 黃 → 紅（xterm-256）；終端機只有 8 色時退回綠／黃／紅
GRAD256 = [42, 78, 114, 150, 186, 222, 216, 210, 204, 198]
C_DIM, C_TITLE, C_ACCENT = 1, 2, 3  # 色對編號；漸層從 10 開始


def wid(s):
    """終端機顯示寬度（CJK 兩格）。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def running_jobs():
    out = subprocess.run([f"{SLURM_BIN}/squeue", "-h", "-u", getpass.getuser(), "-t", "R",
                          "-o", "%i|%j|%N"], capture_output=True, text=True).stdout
    return {l.split("|")[0]: l.split("|") for l in out.split("\n") if l}


def num(s):
    try:
        return float(s)
    except ValueError:  # [N/A]、[Not Supported]
        return None


def _die_with_parent():
    """srun 在 python 死掉（關終端機、kill）時跟著收到 SIGTERM，不留下孤兒 step。"""
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


class Feed:
    """一個 job 一條常駐 srun；最新一列與歷史放在這裡，由背景執行緒更新。"""

    def __init__(self, job, name, node, interval_ms):
        self.job, self.name, self.node = job, name, node
        self.cur, self.err, self.alive = None, "", True
        self.util, self.mem = deque(maxlen=HIST), deque(maxlen=HIST)
        cmd = [f"{SLURM_BIN}/srun", f"--jobid={job}", "--overlap", "-n1", "--quiet", "--input=none",
               "nvidia-smi", f"--query-gpu={','.join(FIELDS)}",
               "--format=csv,noheader,nounits", f"-lms={interval_ms}"]
        self.p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, bufsize=1, preexec_fn=_die_with_parent,
                                  start_new_session=True)  # 不掛在終端機上：不會被 SIGTTOU 停住，Ctrl-C 也由這裡統一收
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
        self.err = ((self.p.stderr.read() or "").strip().splitlines()[-1:] or ["srun 結束"])[0]

    def stop(self):
        if self.p.poll() is None:
            self.p.terminate()  # srun 收到 SIGTERM 會取消 step


class Painter:
    def __init__(self, scr):
        self.scr = scr
        curses.use_default_colors()
        many = curses.COLORS >= 256
        curses.init_pair(C_DIM, 244 if many else curses.COLOR_WHITE, -1)
        curses.init_pair(C_TITLE, 117 if many else curses.COLOR_CYAN, -1)
        curses.init_pair(C_ACCENT, 183 if many else curses.COLOR_MAGENTA, -1)
        grad = GRAD256 if many else [curses.COLOR_GREEN] * 4 + [curses.COLOR_YELLOW] * 3 + [curses.COLOR_RED] * 3
        for i, c in enumerate(grad):
            curses.init_pair(10 + i, c, -1)
        self.ngrad = len(grad)
        self.dim = curses.color_pair(C_DIM) | (0 if many else curses.A_DIM)

    def grad(self, frac):
        return curses.color_pair(10 + min(self.ngrad - 1, max(0, int(frac * self.ngrad))))

    def put(self, y, x, s, attr=0):
        H, W = self.scr.getmaxyx()
        if 0 <= y < H and x < W - 1:
            try:
                self.scr.addstr(y, x, s[:W - 1 - x], attr)
            except curses.error:
                pass

    def meter(self, y, x, w, pct):
        """⅛ 格精度的漸層條，空的部分畫暗色軌道。"""
        f = min(max(pct, 0), 100) / 100
        eighths = int(round(f * w * 8))
        full, rem = divmod(eighths, 8)
        for i in range(w):
            if i < full:
                self.put(y, x + i, "█", self.grad(i / max(w - 1, 1)))
            elif i == full and rem:
                self.put(y, x + i, EIGHTHS[rem - 1], self.grad(i / max(w - 1, 1)))
            else:
                self.put(y, x + i, "─", self.dim)

    def graph(self, y, x, w, rows, hist):
        """braille 面積圖：一欄兩個取樣、一列 4 點高；顏色依高度漸層（越高越紅）。"""
        vals = list(hist)[-2 * w:]
        vals = [0.0] * (2 * w - len(vals)) + vals
        dots = rows * 4
        hts = [int(round(v / 100 * dots)) for v in vals]
        for r in range(rows):  # r=0 是最上面一列
            base = (rows - 1 - r) * 4
            line = []
            for c in range(w):
                code = 0
                for k in range(4):
                    if hts[2 * c] > base + k:
                        code |= BR_L[k]
                    if hts[2 * c + 1] > base + k:
                        code |= BR_R[k]
                line.append(chr(0x2800 + code) if code else " ")
            attr = self.grad((rows - 1 - r + 0.5) / rows)
            self.put(y + r, x, "".join(line), attr)

    def box(self, y, x, w, h, left, right, foot):
        d = self.dim
        self.put(y, x, "╭" + "─" * (w - 2) + "╮", d)
        for r in range(1, h - 1):
            self.put(y + r, x, "│", d)
            self.put(y + r, x + w - 1, "│", d)
        self.put(y + h - 1, x, "╰" + "─" * (w - 2) + "╯", d)
        pos = x + 2
        for text, attr in left:  # 左上標題：[(字, 屬性), ...]
            self.put(y, pos, text, attr)
            pos += wid(text)
        rlen = sum(wid(t) for t, _ in right)
        pos = x + w - 2 - rlen
        for text, attr in right:
            self.put(y, pos, text, attr)
            pos += wid(text)
        if foot:
            self.put(y + h - 1, x + w - 2 - wid(foot), foot, d)


def card(p, y, x, w, f, mode):
    """一張卡 5 列高：框 + 3 列內容。"""
    d = f.cur
    left = [(" ", 0), (f.job, curses.color_pair(C_ACCENT) | curses.A_BOLD), (" ", 0),
            (f.name, curses.color_pair(C_TITLE) | curses.A_BOLD), (" ", 0)]
    if d is None:
        p.box(y, x, w, 5, left, [(f" {f.node} ", p.dim)], "")
        msg = "等第一筆資料…" if f.alive else f"讀不到：{f.err}"
        p.put(y + 2, x + 3, msg, p.dim)
        return
    gname = d["name"].replace("NVIDIA ", "")
    right = [(f" {gname} ", curses.A_BOLD), ("· ", p.dim), (d["pci.bus_id"][-7:], 0),
             (" · ", p.dim), (f"{f.node} ", 0)]
    foot = (f" {'GPU 使用率' if mode == 'util' else '顯存'} · {len(f.util)}s " if f.alive
            else f" 已停止：{f.err} ")
    p.box(y, x, w, 5, left, right, foot)

    u = num(d["utilization.gpu"]) or 0
    mu, mt = num(d["memory.used"]) or 0, num(d["memory.total"]) or 1
    pw, pl = num(d["power.draw"]) or 0, num(d["power.limit"]) or 1
    temp = d["temperature.gpu"]
    inner = w - 4
    left_w = min(56, max(34, inner * 11 // 20))
    mw = left_w - 4 - 12  # 「GPU 」+ 條 + 右邊 12 格文字
    rows = [("GPU", u, f"{u:5.0f}%"),
            ("MEM", 100 * mu / mt, f"{mu / 1024:5.1f}/{mt / 1024:.0f}G"),
            ("PWR", 100 * pw / pl, f"{pw:4.0f}W {temp:>3}°C")]
    for i, (lab, pct, txt) in enumerate(rows):
        yy, xx = y + 1 + i, x + 2
        p.put(yy, xx, lab, p.dim)
        p.meter(yy, xx + 4, mw, pct)
        p.put(yy, xx + 5 + mw, txt.rjust(11), curses.A_BOLD if lab != "PWR" else 0)
    gx, gw = x + 2 + left_w + 2, inner - left_w - 2
    if gw >= 8:
        p.put(y + 1, gx - 1, "│", p.dim)
        p.put(y + 2, gx - 1, "│", p.dim)
        p.put(y + 3, gx - 1, "│", p.dim)
        p.graph(y + 1, gx + 1, gw - 1, 3, f.util if mode == "util" else f.mem)


def draw(p, feeds, interval_ms, mode):
    scr = p.scr
    scr.erase()
    H, W = scr.getmaxyx()
    nodes = sorted({f.node for f in feeds})
    p.put(0, 1, "gputop", curses.color_pair(C_ACCENT) | curses.A_BOLD)
    p.put(0, 8, f"{len(feeds)} GPU · {','.join(nodes)} · {interval_ms / 1000:g}s", p.dim)
    keys = "q 離開  m 切換圖  行程表：make nvtop JOB=<id>"
    p.put(0, W - 2 - wid(keys), keys, p.dim)
    y = 1
    w = min(W - 2, 140)
    for f in feeds:
        if y + 5 > H:
            p.put(H - 1, 2, "…還有卡畫不下（終端機太矮）", p.dim)
            break
        card(p, y, 1, w, f, mode)
        y += 5
    scr.refresh()


def have_tty():
    try:
        os.close(os.open("/dev/tty", os.O_RDWR))
        return True
    except OSError:
        return False


def main(scr, a):
    curses.curs_set(0)
    p = Painter(scr)
    scr.timeout(250)
    feeds, last_scan, mode = {}, 0, "util"
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
            draw(p, [feeds[j] for j in sorted(feeds, key=int)], a.interval_ms, mode)
            k = scr.getch()
            if not have_tty():  # 終端機斷了（關視窗、ssh 斷線）：getch 會一直立刻回 -1，不能空轉重開 srun
                break
            if k in (ord("q"), ord("Q"), 27):
                break
            if k in (ord("m"), ord("M")):
                mode = "mem" if mode == "util" else "util"
    finally:
        for f in feeds.values():
            f.stop()
        deadline = time.time() + 3
        for f in feeds.values():
            try:
                f.p.wait(max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                f.p.kill()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="*")
    ap.add_argument("--interval-ms", type=int, default=1000)
    a = ap.parse_args()
    if not a.jobs and not running_jobs():
        raise SystemExit("沒有正在跑的 job（make status 看）")
    locale.setlocale(locale.LC_ALL, "")
    for sig in (signal.SIGHUP, signal.SIGTERM):  # 讓 finally 有機會收掉 srun
        signal.signal(sig, lambda *_: (_ for _ in ()).throw(SystemExit(0)))
    curses.wrapper(main, a)
