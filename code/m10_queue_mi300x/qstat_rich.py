#!/usr/bin/env python3
"""MI300X GPU 佇列的狀態板（rich；10/19）。純文字備用版是 qstat.sh（make status-plain）。

用法（repo 根目錄的 make 會呼叫這支）：
  qstat_rich.py [status]            完整狀態板                      make status
  qstat_rich.py compact             精簡版（ssh 登入時）              —
  qstat_rich.py q|gpu|jobs|nh|done  各看一部分                      make q／gpu／jobs／nh／done
  qstat_rich.py tail [N] [JOB]      跑中 stage 的 log（或某個 job）   make tail N=40 JOB=ds1-llama-s0
  qstat_rich.py watch [秒]          全螢幕自動刷新（Ctrl-C 離開）     make watch
"""
from __future__ import annotations

import datetime as D
import glob
import json
import os
import re
import subprocess
import sys
import time

from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

J = "/tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs"
BACKUP = "/mlsteam/data/tiara/jobs_backup/auto/BACKUP_TIME"
TW = D.timezone(D.timedelta(hours=8))
UTC = D.timezone.utc
ROLES = ("main", "helper")
ROLE_GPU = {"main": 1, "helper": 0}
QDIR = {"main": "queue", "helper": "queue_g0"}
RLOG = {"main": "runner.log", "helper": "runner_g0.log"}
C_MAIN, C_HELP, C_TIME, C_QUAL = "magenta", "cyan", "yellow", "cyan"
console = Console(highlight=False)


# ================================================================ 小工具
def now():
    return D.datetime.now(UTC)


def dur(sec) -> str:
    if sec is None:
        return ""
    sec = int(max(0, sec))
    if sec >= 86400:
        return f"{sec // 86400}d{sec % 86400 // 3600}h"
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m"


def tw(t: D.datetime) -> str:
    return t.astimezone(TW).strftime("%m/%d %H:%M")


def role_style(role):
    return C_MAIN if role == "main" else C_HELP


def badge(need):
    return Text(" 量時間 ", style=f"bold black on {C_TIME}") if need == "both" else Text(" 品質 ", style=f"black on {C_QUAL}")


def bar(frac, width=None, color="cyan"):
    frac = 0.0 if frac is None else max(0.0, min(1.0, frac))
    return ProgressBar(total=1.0, completed=frac, width=width, complete_style=color, finished_style="green")


# ================================================================ 資料
def run_bg(cmd):
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)


class Snap:
    """一次刷新的所有資料（amd-smi 兩個指令平行跑）。"""

    def __init__(self):
        p_list, p_met = run_bg(["amd-smi", "list"]), run_bg(["amd-smi", "metric", "--mem-usage", "--usage", "--json"])
        self.t = now()
        self.ev = self._events()
        self.pids = self._pids()
        self.queues = {r: self._queue(r) for r in ROLES}
        self.running = {r: self._running(r) for r in ROLES}
        self.cmds = self._cmds()
        self.jobs = {r: self._pool(r, self.running[r]) for r in ROLES if self.running[r]}
        self.nh = self._nh()
        self.ledger = self._ledger()
        self.want_both = os.path.exists(os.path.join(J, "WANT_BOTH"))
        self.backup = open(BACKUP).read().strip() if os.path.exists(BACKUP) else ""
        self.gpus = self._gpus(p_list.communicate()[0], p_met.communicate()[0])

    # -- runner
    def _events(self):
        ev = []
        for role, f in RLOG.items():
            try:
                for l in open(os.path.join(J, f), errors="replace"):
                    m = re.search(r"\[runner(?::(\w+))?[^\]]*\] (START|DONE|FAILED|PREEMPTED) (\S+)(.*?)(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)", l)
                    if m:
                        ev.append((m.group(1) or role, m.group(2), m.group(3), m.group(4),
                                   D.datetime.fromisoformat(m.group(5).replace("Z", "+00:00"))))
            except FileNotFoundError:
                pass
        return ev

    def _pids(self):
        out = {}
        ps = subprocess.run(["pgrep", "-f", "scratchpad/jobs/runner.sh"], capture_output=True, text=True).stdout.split()
        for p in ps:
            try:
                env = open(f"/proc/{p}/environ", "rb").read().split(b"\0")
            except OSError:
                continue
            for r in ROLES:
                if f"ROLE={r}".encode() in env:
                    out[r] = int(p)
        return out

    def _queue(self, role):
        out = []
        for f in sorted(glob.glob(os.path.join(J, QDIR[role], "*.sh"))):
            txt = open(f, errors="replace").read(600)
            m = re.search(r"^# GPU-NEED: *(one|both)", txt, re.M)
            need = "one" if role == "helper" else (m.group(1) if m else "both")
            out.append((os.path.basename(f)[:-3], need))
        return out

    def _running(self, role):
        fs = glob.glob(os.path.join(J, QDIR[role], "*.running"))
        if not fs:
            return None
        name = os.path.basename(fs[0])[:-8]
        st = [e for e in self.ev if e[0] == role and e[1] == "START" and e[2] == name]
        need = re.search(r"need=(\w+)", st[-1][3]).group(1) if st and "need=" in st[-1][3] else ("one" if role == "helper" else "both")
        return dict(name=name, need=need, since=st[-1][4] if st else None)

    def _cmds(self):
        cmds, files = {}, {}
        for jf in glob.glob(os.path.join(J, "jobs_*.json")):
            try:
                xs = json.load(open(jf))
            except Exception:
                continue
            files[jf] = [x["name"] for x in xs]
            for x in xs:
                cmds.setdefault(x["name"], x["cmd"])
        self.jobfiles = files
        return cmds

    def _stage_jobs(self, role, name):
        """stage 腳本裡 `pool <jobs.json>` 用的 job 清單（還沒開任何 job 時也知道有哪些）。"""
        f = os.path.join(J, QDIR[role], f"{name}.running")
        try:
            txt = open(f, errors="replace").read()
        except OSError:
            return None
        names = []
        for m in re.finditer(r"^\s*pool\s+(\S+\.json)", txt, re.M):
            jf = m.group(1).replace("$JOBS", J).replace("${JOBS}", J)
            jf = jf if os.path.isabs(jf) else os.path.join(J, jf)
            try:
                names += [x["name"] for x in json.load(open(jf))]
            except Exception:
                pass
        return names if re.search(r"^\s*pool\s", txt, re.M) else None

    def _pool(self, role, stage):
        log = os.path.join(J, QDIR[role], f"{stage['name']}.log")
        st, dn, fl, pre = {}, set(), set(), False
        planned = self._stage_jobs(role, stage["name"])
        if os.path.exists(log):
            for l in open(log, errors="replace"):
                m = re.match(r"\[pool\] start (\S+)(?: gpu(\d))?", l)
                if m:
                    st[m.group(1)] = int(m.group(2)) if m.group(2) else ROLE_GPU[role]
                m = re.match(r"\[pool\] (done|FAILED) (\S+)", l)
                if m:
                    (dn if m.group(1) == "done" else fl).add(m.group(2))
                pre |= "PREEMPTED" in l
        if planned is None and not st:
            return dict(pool=False, all=[], done=set(), failed=set(), running=[], pending=[], items_done=0, items_total=0)
        alln = planned or None
        if not alln:
            for jf, names in self.jobfiles.items():       # 這個 stage 的 jobs 檔＝包含所有已開的 job、最小的那個
                if all(n in names for n in st) and (alln is None or len(names) < len(alln)):
                    alln = names
        alln = alln or list(st)
        run = []
        for n, g in st.items():
            if n in dn or n in fl:
                continue
            run.append(self._job(n, g))
        pend = [n for n in alln if n not in st]
        it_tot = sum((shard_items(self.cmds.get(n, "")) or 0) for n in alln)
        it_done = sum((shard_items(self.cmds.get(n, "")) or 0) for n in dn) + sum(r["done_items"] for r in run)
        return dict(pool=True, all=alln, done=dn, failed=fl, running=run, pending=pend, items_done=it_done,
                    items_total=it_tot, preempted=pre)

    def _job(self, n, g):
        tot = shard_items(self.cmds.get(n, ""))
        p = os.path.join(J, f"pool_{n}.log")
        done_items, last, started = 0, "", None
        if os.path.exists(p):
            dirs = [x.split("dir=", 1)[1].strip() for x in open(p, errors="replace") if "[runsh] dir=" in x]
            for dd in dirs:                                # 續跑過的 job 有好幾個 run 目錄：題數加總
                so = os.path.join(dd, "stdout.log")
                if os.path.exists(so):
                    ls = [x for x in open(so, errors="replace") if re.match(r"\[\d+\]", x)]
                    done_items += len(ls)
                    if ls:
                        last = ls[-1]
            if dirs:
                try:
                    started = D.datetime.strptime(os.path.basename(dirs[-1])[:15], "%Y%m%d-%H%M%S").replace(tzinfo=UTC)
                except ValueError:
                    pass
        r = re.search(r"([\d.]+)s/題", last)
        sec = float(r.group(1)) if r else None
        eta = max(0, tot - done_items) * sec if (tot and sec) else None
        return dict(name=n, gpu=g, done_items=done_items, total=tot, sec=sec, eta=eta,
                    age=(self.t - started).total_seconds() if started else None)

    def _nh(self):
        try:
            return [l.rstrip() for l in open(os.path.join(J, "NEEDS_HUMAN"), errors="replace") if l.strip()]
        except FileNotFoundError:
            return []

    def _ledger(self):
        try:
            d = json.load(open(os.path.join(J, "vram_reserved.json")))
        except Exception:
            return {}
        def alive(pid):                                  # 殭屍（Z）不算：容器的 PID 1 不收殭屍
            try:
                return open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
            except (OSError, IndexError):
                return False
        return {k: v for k, v in d.items() if alive(v["pid"])}

    def _gpus(self, lst, met):
        bdf = {}
        for blk in (lst or "").split("GPU:")[1:]:
            try:
                g = int(blk.split()[0])
            except (IndexError, ValueError):
                continue
            m = re.search(r"BDF:\s*(\S+)", blk)
            bdf[g] = m.group(1) if m else "?"
        try:
            d = json.loads(met)
            d = d if isinstance(d, list) else d["gpu_data"]
        except Exception:
            return []
        out = []
        for x in d:
            m, u = x["mem_usage"], x.get("usage", {})
            a = u.get("gfx_activity")
            a = a.get("value") if isinstance(a, dict) else a
            tot, free = m["total_vram"]["value"] / 1024, m["free_vram"]["value"] / 1024
            res = sum(v["need"] for v in self.ledger.values() if v["gpu"] == x["gpu"])
            out.append(dict(gpu=x["gpu"], bdf=bdf.get(x["gpu"], "?"), used=tot - free, total=tot, util=a, res=res))
        return out


def shard_items(cmd: str):
    """job 指令的 --tasks／--n／--shard → 這個 shard 的題數（plan_items：每個任務 n 題，第 k 題給 k mod 份數＝i）。"""
    m = re.search(r"--n (\d+)", cmd or "")
    if not m:
        return None
    n = int(m.group(1))
    t = re.search(r"--tasks ((?:T\d\s*)+)", cmd)
    nt = len(t.group(1).split()) if t else 4
    sh = re.search(r"--shard (\d+)/(\d+)", cmd)
    i, k = (int(sh.group(1)), int(sh.group(2))) if sh else (0, 1)
    return len([x for x in range(n * nt) if x % k == i])


# ================================================================ 畫
def header(s: Snap):
    alive = sum(1 for r in ROLES if r in s.pids)
    chips = [Text.assemble(("Tiara", "bold magenta"), (" · MI300X ×2", "bold"))]
    chips.append(Text(f" runner {alive}/2 ", style=f"bold black on {'green' if alive == 2 else 'red'}"))
    for g in s.gpus:
        col = "red" if g["used"] / g["total"] > 0.92 else "yellow" if g["used"] / g["total"] > 0.75 else "green"
        chips.append(Text.assemble((f" 卡{g['gpu']} ", f"bold black on {role_style('main' if g['gpu'] == 1 else 'helper')}"),
                                   (f" {g['util']}% {g['used']:.0f}G ", f"black on {col}")))
    chips.append(Text(f" 問題 {len(s.nh)} ", style=f"bold black on {'red' if s.nh else 'green'}"))
    if s.want_both:
        chips.append(Text(" ⏸ 量時間中 ", style=f"bold black on {C_TIME}"))
    nxt = next((n for n, need in s.queues["main"] if need == "both"), None)
    if nxt:
        chips.append(Text.assemble(("下一個量時間 ", "dim"), (nxt, C_TIME)))
    sub = Text.assemble(("台灣 ", "dim"), (tw(s.t), "bold"), ("  ·  UTC ", "dim"), (s.t.strftime("%m/%d %H:%M"), ""),
                        (f"  ·  佇列備份 {s.backup[11:16]}Z" if s.backup else "", "dim"))
    return Panel(Group(Columns(chips, padding=(0, 1)), sub), box=box.ROUNDED, border_style="magenta", padding=(0, 1))


def runners_panel(s: Snap):
    t = Table(box=box.SIMPLE_HEAD, expand=True, show_edge=False, pad_edge=False, header_style="dim")
    for c, kw in (("runner", {}), ("卡", {}), ("狀態", {}), ("正在跑", {"ratio": 3}), ("", {}), ("已跑", {"justify": "right"}),
                  ("下一個", {"ratio": 2})):
        t.add_column(c, **kw)
    for role in ROLES:
        cur, pid = s.running[role], s.pids.get(role)
        st = Text("● 活著", style="bold green") if pid else Text("● 沒在跑", style="bold red")
        nxt = s.queues[role][0][0] if s.queues[role] else "—"
        t.add_row(Text(role, style=f"bold {role_style(role)}"), str(ROLE_GPU[role]), st,
                  Text(cur["name"], style="bold") if cur else Text("（閒置）", style="dim"),
                  badge(cur["need"]) if cur else "", dur((s.t - cur["since"]).total_seconds()) if cur and cur["since"] else "",
                  Text(nxt, style="dim"))
    sub = Text("⏸ main 在量時間：helper 讓出卡 0", style=C_TIME) if s.want_both else None
    return Panel(t, title="[bold]Runner", title_align="left", border_style=C_TIME if s.want_both else "blue",
                 subtitle=sub, subtitle_align="right")


def gpu_panel(s: Snap):
    if not s.gpus:
        return Panel(Text("amd-smi 查不到", style="red"), title="[bold]GPU", title_align="left", border_style="red")
    t = Table.grid(padding=(0, 1), expand=True)
    for c in ({"width": 6}, {"width": 9}, {"width": 7}, {"ratio": 1}, {"width": 13, "justify": "right"}, {"ratio": 1},
              {"width": 9, "justify": "right"}, {"width": 12, "justify": "right"}):
        t.add_column(**c)
    for g in s.gpus:
        role = "main" if g["gpu"] == 1 else "helper"
        f = g["used"] / g["total"]
        col = "red" if f > 0.92 else "yellow" if f > 0.75 else "green"
        uc = "green" if (g["util"] or 0) >= 80 else "yellow" if (g["util"] or 0) >= 30 else "red"
        t.add_row(Text(f"卡 {g['gpu']}", style=f"bold {role_style(role)}"), Text(g["bdf"][5:], style="dim"),
                  Text(role, style=role_style(role)), bar(f, color=col), Text(f"{g['used']:.0f}/{g['total']:.0f} GB", style=col),
                  bar((g["util"] or 0) / 100, color=uc), Text(f"使用 {g['util']}%", style=uc),
                  Text(f"帳 {g['res']:.0f} GB", style="dim"))
    return Panel(t, title="[bold]GPU", title_align="left", border_style="blue",
                 subtitle=Text("帳＝pool 開 job 前登記的峰值；同一張卡加總放得下才開", style="dim"), subtitle_align="right")


def jobs_panel(s: Snap, compact=False):
    parts = []
    for role in ROLES:
        cur = s.running[role]
        if not cur:
            continue
        pj = s.jobs.get(role)
        head = Text.assemble((cur["name"], f"bold {role_style(role)}"), (f"  {role}  ", "dim"))
        head.append_text(badge(cur["need"]))
        if not pj or not pj["pool"]:
            parts.append(Group(head, Text("  不是 job pool 的 stage（make tail 看 log）", style="dim")))
            continue
        if not pj["all"]:
            parts.append(Group(head, Text("  還沒開任何 job（可能在等顯存：看 GPU 的「帳」）", style="yellow")))
            continue
        if not pj["running"] and not pj["done"] and pj["pending"]:
            parts.append(Group(head, Text.assemble(("  ", ""), (f"{len(pj['pending'])} 個 job 排著、還沒開", "yellow"),
                                                   (f"（共 {pj['items_total']} 題）" if pj["items_total"] else "", "dim"),
                                                   ("：在等顯存，看 GPU 的「帳」", "dim"))))
            continue
        nall, ndone = len(pj["all"]), len(pj["done"])
        f = pj["items_done"] / pj["items_total"] if pj["items_total"] else ndone / max(1, nall)
        etas = [r["eta"] for r in pj["running"] if r["eta"] is not None]
        summ = Table.grid(padding=(0, 1), expand=True)
        summ.add_column(width=34)
        summ.add_column(ratio=1)
        summ.add_column(width=26, justify="right")
        summ.add_row(Text.assemble(("  job ", "dim"), (f"{ndone}/{nall}", "bold"), (" 完成  ", "dim"),
                                   (f"{len(pj['running'])}", "bold cyan"), (" 在跑  ", "dim"),
                                   (f"{len(pj['pending'])}", "bold"), (" 排著", "dim")),
                     bar(f, color="cyan"),
                     Text.assemble((f"{pj['items_done']}/{pj['items_total']} 題 " if pj["items_total"] else "", ""),
                                   (f"{100 * f:.0f}%", "bold")))
        items = [head, summ]
        if not compact or len(pj["running"]) <= 8:
            t = Table(box=box.SIMPLE, show_header=True, header_style="dim", pad_edge=False, expand=True, show_edge=False)
            for c, kw in (("卡", {"width": 3}), ("job", {"ratio": 3}), ("進度", {"ratio": 3}), ("", {"width": 9, "justify": "right"}),
                          ("秒/題", {"justify": "right"}), ("還要", {"justify": "right"}), ("已跑", {"justify": "right"})):
                t.add_column(c, **kw)
            for r in sorted(pj["running"], key=lambda r: (r["eta"] is None, -(r["eta"] or 0))):
                fr = r["done_items"] / r["total"] if r["total"] else None
                t.add_row(Text(str(r["gpu"]), style=role_style("main" if r["gpu"] == 1 else "helper")), r["name"],
                          bar(fr, color="cyan") if fr is not None else Text("載入中", style="dim"),
                          Text(f"{r['done_items']}/{r['total']}" if r["total"] else f"{r['done_items']}"),
                          f"{r['sec']:.0f}" if r["sec"] else "", Text(dur(r["eta"]), style="bold"), Text(dur(r["age"]), style="dim"))
            items.append(t)
        if etas and pj["pending"]:
            items.append(Text(f"  在跑的最慢還要 {dur(max(etas))}；另外還有 {len(pj['pending'])} 個 job 排著", style="dim"))
        elif etas:
            items.append(Text(f"  這個 stage 大約還要 {dur(max(etas))}", style="dim"))
        for x in sorted(pj["failed"]):
            items.append(Text(f"  ✗ 失敗：{x}　（make tail JOB={x}）", style="bold red"))
        parts.append(Group(*items))
    if not parts:
        parts = [Text("沒有跑中的 stage", style="dim")]
    out = []
    for i, p in enumerate(parts):
        if i:
            out.append(Rule(style="dim"))
        out.append(p)
    return Panel(Group(*out), title="[bold]跑中的 job", title_align="left", border_style="cyan")


def queue_panel(s: Snap):
    cols = []
    for role in ROLES:
        t = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, expand=True, header_style=f"bold {role_style(role)}")
        t.add_column(f"#", width=3, justify="right", style="dim")
        t.add_column(f"{role}（卡 {ROLE_GPU[role]}）", ratio=1)
        t.add_column("", justify="right")
        q = s.queues[role]
        cur = s.running[role]
        if cur:
            t.add_row("▶", Text(cur["name"], style="bold"), badge(cur["need"]))
        for i, (name, need) in enumerate(q, 1):
            t.add_row(str(i), name, badge(need))
        if not q and not cur:
            t.add_row("", Text("（空）", style="dim"), "")
        cols.append(t)
    g = Table.grid(expand=True, padding=(0, 2))
    g.add_column(ratio=1)
    g.add_column(ratio=1)
    g.add_row(*cols)
    return Panel(g, title="[bold]佇列", title_align="left", border_style="blue",
                 subtitle=Text("量時間＝兩張卡獨佔（helper 會讓出）；品質＝一張卡", style="dim"), subtitle_align="right")


def notes():
    """jobs/stage_notes.json：stage → 說明（失敗或被中斷的 stage 已經怎麼處理）。狀態板在旁邊註明，免得看到 ✗ 要猜。"""
    try:
        return json.load(open(os.path.join(J, "stage_notes.json")))
    except Exception:
        return {}


def done_panel(s: Snap, k=10):
    st, out = {}, []
    nt = notes()
    for role, kind, n, _r, t in s.ev:
        if kind == "START":
            st[(role, n)] = t
        elif (role, n) in st:
            out.append((t, role, kind, n, (t - st[(role, n)]).total_seconds()))
    out.sort()
    tb = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, expand=True, header_style="dim")
    for c, kw in (("台灣時間", {}), ("", {}), ("runner", {}), ("stage", {"ratio": 2}), ("花了", {"justify": "right"}),
                  ("說明", {"ratio": 3})):
        tb.add_column(c, **kw)
    for t, role, kind, n, sec in out[-k:]:
        sty = {"DONE": ("✓", "green"), "FAILED": ("✗", "bold red"), "PREEMPTED": ("⏸", C_TIME)}[kind]
        note = nt.get(n, "")
        if note and kind == "FAILED" and note.startswith("已處理"):
            sty = ("✗→✓", "yellow")
        tb.add_row(tw(t), Text(sty[0], style=sty[1]), Text(role, style=role_style(role)), n, dur(sec), Text(note, style="dim"))
    return Panel(tb, title="[bold]最近完成", title_align="left", border_style="blue")


def nh_panel(s: Snap):
    if not s.nh:
        return Panel(Text("沒有待處理的 ✓", style="green"), title="[bold]NEEDS_HUMAN", title_align="left", border_style="green")
    return Panel(Group(*[Text("• " + l[:170], style="red") for l in s.nh[-8:]]), title=f"[bold]NEEDS_HUMAN（{len(s.nh)}）",
                 title_align="left", border_style="red")


def footer():
    return Text("make status · watch · q · gpu · jobs · nh · done · tail [N=40] [JOB=…] · status-plain", style="dim")


def view(kind: str):
    s = Snap()
    body = {
        "status": [header(s), runners_panel(s), gpu_panel(s), jobs_panel(s), queue_panel(s), done_panel(s, 8), nh_panel(s), footer()],
        "compact": [header(s), runners_panel(s), jobs_panel(s, compact=True), nh_panel(s), footer()],
        "q": [header(s), queue_panel(s)],
        "gpu": [header(s), gpu_panel(s)],
        "jobs": [header(s), jobs_panel(s)],
        "nh": [nh_panel(s)],
        "done": [done_panel(s, 25)],
    }[kind]
    return Group(*body)


def tail(n: int, job: str | None):
    pats = [(r"Traceback|Error|FAILED|失敗|OutOfMemory|exit=[1-9]", "bold red"), (r"\[pool\] start", "cyan"),
            (r"\[pool\] done|exit=0|DONE", "green"), (r"PREEMPTED|WANT_BOTH|NEEDS_HUMAN", C_TIME), (r"^\[\d+\]", "")]

    def show(title, path, n, style):
        if not path or not os.path.exists(path):
            return
        lines = open(path, errors="replace").read().splitlines()[-n:]
        out = []
        for l in lines:
            t = Text(l[:220])
            for p, sty in pats:
                if sty and re.search(p, l):
                    t.stylize(sty)
                    break
            out.append(t)
        console.print(Panel(Group(*out) if out else Text("（空）", style="dim"), title=f"[bold]{title}", title_align="left",
                            border_style=style, subtitle=Text(path.replace(J, "jobs").replace("/mlsteam/data/tiara/runs", "runs"), style="dim"),
                            subtitle_align="right"))
    if job:
        p = os.path.join(J, f"pool_{job}.log")
        show(f"pool_{job}.log", p, n, "cyan")
        if os.path.exists(p):
            dirs = [x.split("dir=", 1)[1].strip() for x in open(p, errors="replace") if "[runsh] dir=" in x]
            if dirs:
                show(f"{os.path.basename(dirs[-1])} / stdout", os.path.join(dirs[-1], "stdout.log"), n, "cyan")
                show(f"{os.path.basename(dirs[-1])} / stderr", os.path.join(dirs[-1], "stderr.log"), min(n, 15), "red")
        return
    for role in ROLES:
        fs = glob.glob(os.path.join(J, QDIR[role], "*.running"))
        if fs:
            show(f"{os.path.basename(fs[0])[:-8]}（{role}）", fs[0][:-8] + ".log", n, role_style(role))
    show("runner（main）", os.path.join(J, RLOG["main"]), 4, "dim")


def main():
    if not os.path.isdir(os.path.join(J, "queue")):
        console.print(Panel(Text("找不到 GPU 佇列（容器重啟後還沒還原？）\n看 /mlsteam/workspace/RESUME_20261019_2GPU.md", style="yellow"),
                            title="Tiara MI300X", border_style="yellow"))
        return
    a = [x for x in sys.argv[1:] if x]
    kind = a[0] if a else "status"
    kind = {"--compact": "compact", "queue": "q"}.get(kind, kind)
    if kind == "tail":
        n = int(a[1]) if len(a) > 1 and a[1].isdigit() else 30
        job = next((x for x in a[1:] if not x.isdigit()), None)
        tail(n, job)
        return
    if kind == "watch":
        sec = int(a[1]) if len(a) > 1 and a[1].isdigit() else 30
        try:
            with Live(view("status"), console=console, screen=True, auto_refresh=False) as live:
                while True:
                    time.sleep(sec)
                    live.update(view("status"), refresh=True)
        except KeyboardInterrupt:
            pass
        return
    console.print(view(kind))


if __name__ == "__main__":
    main()
