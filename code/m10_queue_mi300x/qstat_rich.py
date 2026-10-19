#!/usr/bin/env python3
"""MI300X GPU 佇列的狀態板（rich 版；10/19）。純文字版是 qstat.sh。

用法：python qstat_rich.py [--compact]
  make status        完整版（runner、卡、顯存帳、跑中的 job、排隊、最近完成、NEEDS_HUMAN）
  ssh 登入時          --compact（zshrc.tiara 呼叫）
"""
from __future__ import annotations

import datetime as D
import glob
import json
import os
import re
import subprocess
import sys

from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

J = "/tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs"
TW = D.timezone(D.timedelta(hours=8))
UTC = D.timezone.utc
ROLE_GPU = {"main": 1, "helper": 0}
QDIR = {"main": "queue", "helper": "queue_g0"}
RLOG = {"main": "runner.log", "helper": "runner_g0.log"}
console = Console()


def now():
    return D.datetime.now(UTC)


def dur(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m"


def tw(t: D.datetime) -> str:
    return t.astimezone(TW).strftime("%m/%d %H:%M")


# ---------------------------------------------------------------- 資料
def runner_pid(role: str):
    out = subprocess.run(["pgrep", "-f", "scratchpad/jobs/runner.sh"], capture_output=True, text=True).stdout.split()
    for p in out:
        try:
            env = open(f"/proc/{p}/environ", "rb").read().split(b"\0")
        except OSError:
            continue
        if f"ROLE={role}".encode() in env:
            return int(p)
    return None


def runner_events():
    ev = []
    for role, f in RLOG.items():
        try:
            for l in open(os.path.join(J, f)):
                m = re.search(r"\[runner(?::(\w+))?[^\]]*\] (START|DONE|FAILED|PREEMPTED) (\S+)(.*?)(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)", l)
                if m:
                    ev.append((m.group(1) or role, m.group(2), m.group(3), m.group(4),
                               D.datetime.fromisoformat(m.group(5).replace("Z", "+00:00"))))
        except FileNotFoundError:
            pass
    return ev


def gpus():
    bdf = {}
    try:
        for blk in subprocess.run(["amd-smi", "list"], capture_output=True, text=True).stdout.split("GPU:")[1:]:
            g = int(blk.split()[0])
            m = re.search(r"BDF:\s*(\S+)", blk)
            bdf[g] = m.group(1) if m else "?"
        d = json.loads(subprocess.run(["amd-smi", "metric", "--mem-usage", "--usage", "--json"], capture_output=True,
                                      text=True).stdout)
        d = d if isinstance(d, list) else d["gpu_data"]
    except Exception:
        return []
    out = []
    for x in d:
        m, u = x["mem_usage"], x.get("usage", {})
        a = u.get("gfx_activity")
        a = a.get("value") if isinstance(a, dict) else a
        tot, free = m["total_vram"]["value"] / 1024, m["free_vram"]["value"] / 1024
        out.append(dict(gpu=x["gpu"], bdf=bdf.get(x["gpu"], "?"), used=tot - free, total=tot, util=a))
    return out


def ledger():
    try:
        d = json.load(open(os.path.join(J, "vram_reserved.json")))
    except Exception:
        return {}
    return {k: v for k, v in d.items() if os.path.exists(f"/proc/{v['pid']}")}


def queue(role: str):
    out = []
    for f in sorted(glob.glob(os.path.join(J, QDIR[role], "*.sh"))):
        txt = open(f, errors="replace").read(400)
        m = re.search(r"^# GPU-NEED: *(one|both)", txt, re.M)
        need = "one" if role == "helper" else (m.group(1) if m else "both")
        out.append((os.path.basename(f)[:-3], need))
    return out


def running_stage(role: str):
    fs = glob.glob(os.path.join(J, QDIR[role], "*.running"))
    return os.path.basename(fs[0])[:-8] if fs else None


def pool_jobs(role: str, stage: str):
    log = os.path.join(J, QDIR[role], f"{stage}.log")
    if not os.path.exists(log):
        return None
    st, dn, fl = {}, set(), set()
    for l in open(log, errors="replace"):
        m = re.match(r"\[pool\] start (\S+)(?: gpu(\d))?", l)
        if m:
            st[m.group(1)] = int(m.group(2)) if m.group(2) else ROLE_GPU[role]
        m = re.match(r"\[pool\] (done|FAILED) (\S+)", l)
        if m:
            (dn if m.group(1) == "done" else fl).add(m.group(2))
    if not st and not dn:
        return dict(total=None, done=0, failed=[], running=[])
    total = None
    for jf in glob.glob(os.path.join(J, "jobs_*.json")):
        try:
            names = [x["name"] for x in json.load(open(jf))]
        except Exception:
            continue
        if names and all(n in names for n in st):
            total = len(names) if total is None or len(names) < total else total
    cmds = {}
    for jf in glob.glob(os.path.join(J, "jobs_*.json")):
        try:
            for x in json.load(open(jf)):
                cmds.setdefault(x["name"], x["cmd"])
        except Exception:
            pass
    run = []
    for n, g in st.items():
        if n in dn or n in fl:
            continue
        prog, rate, started, eta, frac = "載入中", "", None, "", None
        tot = shard_items(cmds.get(n, ""))
        p = os.path.join(J, f"pool_{n}.log")
        if os.path.exists(p):
            dirs = [x.split("dir=", 1)[1].strip() for x in open(p, errors="replace") if "[runsh] dir=" in x]
            done_items, last, sec = 0, "", None
            for dd in dirs:                         # 續跑過的 job 有好幾個 run 目錄：題數加總
                so = os.path.join(dd, "stdout.log")
                if os.path.exists(so):
                    ls = [x.strip() for x in open(so, errors="replace") if re.match(r"\[\d+\]", x)]
                    done_items += len(ls)
                    if ls:
                        last = ls[-1]
            if dirs:
                try:
                    started = D.datetime.strptime(os.path.basename(dirs[-1])[:15], "%Y%m%d-%H%M%S").replace(tzinfo=UTC)
                except ValueError:
                    pass
            if last:
                r = re.search(r"([\d.]+)s/題", last)
                sec = float(r.group(1)) if r else None
                rate = f"{sec:.0f} 秒/題" if sec else ""
                prog = f"{done_items}/{tot} 題" if tot else f"{done_items} 題"
                if tot:
                    frac = min(1.0, done_items / tot)
                    if sec:
                        eta = dur(max(0, tot - done_items) * sec)
        run.append(dict(name=n, gpu=g, prog=prog, rate=rate, eta=eta, frac=frac,
                        age=dur((now() - started).total_seconds()) if started else ""))
    return dict(total=total, done=len(dn), failed=sorted(fl), running=run)


def shard_items(cmd: str):
    """job 指令裡的 --tasks／--n／--shard → 這個 shard 的題數（plan_items：每個任務 n 題，第 k 題給 k mod 份數＝i 的 shard）。"""
    m = re.search(r"--n (\d+)", cmd)
    if not m:
        return None
    n = int(m.group(1))
    t = re.search(r"--tasks ((?:T\d\s*)+)", cmd)
    nt = len(t.group(1).split()) if t else 4
    sh = re.search(r"--shard (\d+)/(\d+)", cmd)
    i, k = (int(sh.group(1)), int(sh.group(2))) if sh else (0, 1)
    return len([x for x in range(n * nt) if x % k == i])


def needs_human():
    try:
        return [l.rstrip() for l in open(os.path.join(J, "NEEDS_HUMAN")) if l.strip()]
    except FileNotFoundError:
        return []


# ---------------------------------------------------------------- 畫
def header():
    t = now()
    txt = Text.assemble(("Tiara · MI300X ×2 ", "bold magenta"), ("  台灣 ", "dim"), (tw(t), "bold"),
                        ("   UTC ", "dim"), (t.strftime("%m/%d %H:%M"), ""))
    b = "/mlsteam/data/tiara/jobs_backup/auto/BACKUP_TIME"
    if os.path.exists(b):
        txt.append(f"   佇列備份 {open(b).read().strip()[11:16]}Z", style="dim")
    return txt


def runners_table(ev):
    t = Table(box=box.SIMPLE_HEAVY, expand=True, show_edge=False, pad_edge=False)
    for c, kw in (("runner", {}), ("卡", {}), ("狀態", {}), ("正在跑", {"ratio": 3}), ("類型", {}), ("已跑", {"justify": "right"})):
        t.add_column(c, **kw)
    for role in ("main", "helper"):
        pid = runner_pid(role)
        last = [e for e in ev if e[0] == role]
        cur = running_stage(role)
        st = Text("● 活著", style="bold green") if pid else Text("● 沒在跑", style="bold red")
        typ, age = "", ""
        if cur:
            s = [e for e in last if e[1] == "START" and e[2] == cur]
            if s:
                need = re.search(r"need=(\w+)", s[-1][3])
                typ = Text("量時間・兩張獨佔", style="bold yellow") if need and need.group(1) == "both" else Text("品質", style="cyan")
                age = dur((now() - s[-1][4]).total_seconds())
        t.add_row(Text(role, style="bold"), f"{ROLE_GPU[role]}", st, Text(cur or "（閒置）", style="bold" if cur else "dim"), typ, age)
    want = os.path.exists(os.path.join(J, "WANT_BOTH"))
    return Panel(t, title="Runner", title_align="left", border_style="yellow" if want else "blue",
                 subtitle=Text("⏸ 量時間中：helper 讓出卡 0", style="yellow") if want else None, subtitle_align="right")


def gpu_panel():
    led = ledger()
    rows = []
    for g in gpus():
        res = sum(v["need"] for v in led.values() if v["gpu"] == g["gpu"])
        role = "main" if g["gpu"] == 1 else "helper"
        frac = g["used"] / g["total"] if g["total"] else 0
        col = "red" if frac > 0.92 else "yellow" if frac > 0.75 else "green"
        line = Table.grid(padding=(0, 1), expand=True)
        line.add_column(width=22)
        line.add_column(ratio=1)
        line.add_column(width=34, justify="right")
        line.add_row(Text.assemble((f"卡 {g['gpu']} ", "bold"), (g["bdf"][5:], "dim"), (f"  {role}", "dim")),
                     ProgressBar(total=g["total"], completed=g["used"], complete_style=col, finished_style=col),
                     Text.assemble((f"{g['used']:.0f}/{g['total']:.0f} GB", col), (f"  使用率 {g['util']}%", ""),
                                   (f"  帳 {res:.0f} GB", "dim")))
        rows.append(line)
    return Panel(Group(*rows) if rows else Text("amd-smi 查不到", style="red"), title="GPU", title_align="left",
                 border_style="blue")


def jobs_panel():
    parts = []
    for role in ("main", "helper"):
        cur = running_stage(role)
        if not cur:
            continue
        pj = pool_jobs(role, cur)
        head = Text.assemble((f"{cur}", "bold"), (f"  ({role})", "dim"))
        if pj is None or (pj["total"] is None and not pj["running"] and pj["done"] == 0):
            parts.append(Group(head, Text("  還沒開任何 job（可能在等顯存：看 GPU 的「帳」）", style="dim")))
            continue
        tot = pj["total"] or (pj["done"] + len(pj["running"]) + len(pj["failed"]))
        bar = Table.grid(padding=(0, 1))
        bar.add_column(width=30)
        bar.add_column(width=36)
        bar.add_row(Text(f"  job {pj['done']}/{tot} 完成，{len(pj['running'])} 在跑", style=""),
                    ProgressBar(total=max(tot, 1), completed=pj["done"], complete_style="cyan", finished_style="green"))
        t = Table(box=box.MINIMAL, show_header=True, header_style="dim", pad_edge=False, expand=True)
        t.add_column("卡", width=3)
        t.add_column("job", ratio=2)
        t.add_column("進度", ratio=2)
        t.add_column("", width=14)
        t.add_column("速度", justify="right")
        t.add_column("還要", justify="right")
        t.add_column("已跑", justify="right")
        for r in pj["running"]:
            pb = ProgressBar(total=1.0, completed=r["frac"], width=14, complete_style="cyan") if r["frac"] is not None else ""
            t.add_row(str(r["gpu"]), r["name"], r["prog"], pb, r["rate"], Text(r["eta"], style="bold"), r["age"])
        items = [head, bar, t]
        for f in pj["failed"]:
            items.append(Text(f"  ✗ 失敗：{f}（{J}/pool_{f}.log）", style="bold red"))
        parts.append(Group(*items))
    if not parts:
        parts = [Text("沒有跑中的 pool stage", style="dim")]
    return Panel(Group(*parts), title="跑中的 job", title_align="left", border_style="cyan")


def queue_panel():
    cols = []
    for role in ("main", "helper"):
        t = Table(box=box.SIMPLE, title=f"{role}（卡 {ROLE_GPU[role]}）", title_style="bold", show_header=False,
                  pad_edge=False, expand=True)
        t.add_column("stage")
        t.add_column("類型", justify="right")
        q = queue(role)
        for name, need in q[:14]:
            t.add_row(name, Text("量時間", style="yellow") if need == "both" else Text("品質/CPU", style="cyan"))
        if len(q) > 14:
            t.add_row(Text(f"… 還有 {len(q) - 14} 個", style="dim"), "")
        if not q:
            t.add_row(Text("（空）", style="dim"), "")
        cols.append(t)
    return Panel(Columns(cols, expand=True, equal=True), title="排隊", title_align="left", border_style="blue")


def done_panel(ev, k=8):
    st, out = {}, []
    for role, kind, n, _rest, t in ev:
        if kind == "START":
            st[(role, n)] = t
        elif (role, n) in st:
            out.append((t, role, kind, n, (t - st[(role, n)]).total_seconds()))
    out.sort()
    tb = Table(box=box.SIMPLE, show_header=False, pad_edge=False, expand=True)
    for c in ("台灣時間", "結果", "stage", "花了"):
        tb.add_column(c, justify="right" if c == "花了" else "left")
    for t, role, kind, n, s in out[-k:]:
        sty = {"DONE": "green", "FAILED": "bold red", "PREEMPTED": "yellow"}[kind]
        tb.add_row(tw(t), Text(kind, style=sty), Text(n, style="" if role == "main" else "dim"), dur(s))
    return Panel(tb, title="最近完成", title_align="left", border_style="blue")


def nh_panel():
    nh = needs_human()
    if not nh:
        return Panel(Text("沒有待處理的 ✓", style="green"), title="NEEDS_HUMAN", title_align="left", border_style="green")
    return Panel(Group(*[Text(l[:150], style="red") for l in nh[-6:]]), title=f"NEEDS_HUMAN（{len(nh)}）",
                 title_align="left", border_style="red")


def main():
    if not os.path.isdir(os.path.join(J, "queue")):
        console.print(Panel(Text("找不到 GPU 佇列（容器重啟後還沒還原？）\n看 /mlsteam/workspace/RESUME_20261019_2GPU.md", style="yellow"),
                            title="Tiara MI300X"))
        return
    ev = runner_events()
    compact = "--compact" in sys.argv
    console.print(header())
    console.print(runners_table(ev))
    console.print(gpu_panel())
    console.print(jobs_panel())
    if not compact:
        console.print(queue_panel())
        console.print(done_panel(ev))
    console.print(nh_panel())
    if compact:
        console.print(Text("make status（完整）・make q・make jobs・make watch・make tail", style="dim"))


if __name__ == "__main__":
    main()
