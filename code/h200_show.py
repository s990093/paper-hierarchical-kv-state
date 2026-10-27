#!/usr/bin/env python3
"""`make log`／`make runs`／`make status`／論文建置摘要的整理版輸出（rich）。

  python3 code/h200_show.py log    [--job ID] [--raw] [-n 12]  slurm-*.out 整理成步驟表＋最新 run 的末段
  python3 code/h200_show.py runs   [-n 15]                     最近的 run 目錄：開始、耗時、exit、污染
  python3 code/h200_show.py status                             自己的 job（squeue）＋今天跑過的（sacct）
  python3 code/h200_show.py latex  [main]                      xelatex .log 的警告摘要
  python3 code/h200_show.py help | usage                       make help／make usage
  <輸出> | python3 code/h200_show.py gpu|env|dryrun|testsubmit  把 nvidia-smi／env_h200.sh／--dry-run／sbatch --test-only 的輸出排版

slurm-*.out 的格式來自 r_h200*.sh 的 `=== <ts> <步驟>` 與 tiara.sbatch 的 `[tiara] RUN_ID=／exit=`。
只讀檔與 squeue／sacct，不改任何東西。
"""
import argparse
import datetime as dt
import getpass
import glob
import os
import re
import shutil
import subprocess

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

RUNS = os.environ.get("RUNS") or os.path.expanduser("~/tiara-runs")
SLURM_BIN = "/cm/local/apps/slurm/current/bin"
con = Console(highlight=False)

STATE_STYLE = {"COMPLETED": "green", "RUNNING": "bold yellow", "PENDING": "cyan",
               "FAILED": "bold red", "CANCELLED": "red", "TIMEOUT": "red", "OUT_OF_MEMORY": "red"}


def short_paths(ln):
    """repo 的絕對路徑（含 /raid 前綴的那個）縮成相對路徑。"""
    repo = os.getcwd()
    for pre in ("/raid" + repo + "/", repo + "/"):
        ln = ln.replace(pre, "")
    ln = ln.replace("/raid" + repo, ".").replace(repo, ".")
    ln = re.sub(r"hf-cache/hub/models--([^/\s]+)/snapshots/[0-9a-f]{40}", r"hf-cache/‹\1›", ln)
    home = os.path.expanduser("~")
    return ln.replace("/raid" + home, "~").replace(home, "~") if home != "/" else ln


def fmt_dur(sec):
    if sec is None:
        return "—"
    sec = int(sec)
    h, m = divmod(sec // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{sec % 60:02d}s" if m < 10 else f"{m}m"


def run_start(run_id):
    m = re.match(r"(\d{8}-\d{6})", run_id)
    return dt.datetime.strptime(m.group(1), "%Y%m%d-%H%M%S") if m else None


def run_info(d):
    """(exit 字串或 None, 結束時間或 None, 是否污染)"""
    ef = os.path.join(d, "exit_code")
    rc, end = None, None
    if os.path.exists(ef):
        rc = open(ef).read().strip() or None
        end = dt.datetime.fromtimestamp(os.path.getmtime(ef))
    return rc, end, os.path.exists(os.path.join(d, "CONTAMINATED"))


def exit_text(rc, running=True):
    if rc is None:
        return Text("… 跑中", "yellow") if running else Text("? 沒寫", "yellow")
    return Text("✓ 0", "green") if rc == "0" else Text(f"✗ {rc}", "bold red")


def state_text(state):
    return Text(state, STATE_STYLE.get(state.split()[0] if state else "", "white"))


def tail_lines(path, n):
    """末 n 行；tqdm 的 \\r 進度只留最後一格，空行丟掉。"""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 200_000))
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    lines = [ln.rstrip().split("\r")[-1].rstrip() for ln in text.splitlines()]
    return [ln for ln in lines if ln][-n:]


def slurm(*args):
    exe = shutil.which(args[0]) or os.path.join(SLURM_BIN, args[0])
    try:
        return subprocess.run([exe, *args[1:]], capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def job_state(jobid):
    out = (slurm("squeue", "-h", "-j", jobid, "-o", "%T") or "").strip()
    if out:
        return out
    out = (slurm("sacct", "-X", "-n", "-P", "-j", jobid, "-o", "state") or "").strip()
    return out.splitlines()[0] if out else "?"


def parse_slurm(path):
    head, steps, tail_msgs, skips = [], [], [], []
    cur = None
    for raw in open(path, encoding="utf-8", errors="replace"):
        ln = short_paths(raw.rstrip("\n").split("\r")[-1].rstrip())
        m = re.match(r"=== (\S+) (\S+)$", ln)
        if m:
            cur = {"ts": m.group(1), "name": m.group(2), "runs": [], "exit": None, "other": []}
            steps.append(cur)
            continue
        if ln.startswith("=== "):  # `=== <ts> R2-C6 失敗（exit 1）→ ...` 這類收尾訊息
            tail_msgs.append(ln.split(" ", 2)[-1])
            cur = None
            continue
        m = re.match(r"\[tiara\] RUN_ID=(\S+)", ln)
        if m and cur is not None:
            cur["runs"].append(m.group(1))
            continue
        m = re.match(r"\[tiara\] exit=(\S+)", ln)
        if m and cur is not None:
            cur["exit"] = m.group(1)
            continue
        if ln.startswith("[tiara] dir=") or not ln.strip():
            continue
        if ln.startswith("[skip] "):
            skips.append(ln[7:])
            continue
        (cur["other"] if cur is not None else head).append(ln)
    return head, steps, tail_msgs, skips


def collapse(steps):
    """連續同名的步驟（例如 R2-C4-t1-restart ×10）併成一列。"""
    out = []
    for s in steps:
        p = out[-1] if out else None
        if p and p["name"] == s["name"] and p["exit"] == "0" and s["exit"] == "0" and not s["other"]:
            p["n"] += 1
            p["runs"] += s["runs"]
        else:
            out.append(dict(s, n=1, runs=list(s["runs"])))
    return out


def cmd_log(a):
    if a.job:
        files = glob.glob(f"slurm-*-{a.job}.out")
    else:
        files = sorted(glob.glob("slurm-*.out"), key=os.path.getmtime, reverse=True)[:1]
    if not files:
        con.print("還沒有 slurm-*.out（job 還沒開始）" if not a.job else f"找不到 slurm-*-{a.job}.out")
        return
    path = files[0]
    if a.raw:
        for ln in tail_lines(path, 40) or []:
            print(ln)
        return
    jobid = re.search(r"-(\d+)\.out$", path)
    state = job_state(jobid.group(1)) if jobid else "?"
    mtime = dt.datetime.fromtimestamp(os.path.getmtime(path))
    head, steps, tail_msgs, skips = parse_slurm(path)

    skips += [s["name"] for s in steps if not s["runs"] and s["exit"] is None and not s["other"]]
    real = collapse([s for s in steps if s["runs"] or s["exit"] is not None or s["other"]])
    running = state in ("RUNNING", "PENDING", "?")

    t = Table(box=box.SIMPLE_HEAD, header_style="bold dim", pad_edge=False, expand=False)
    t.add_column("開始", style="dim")
    t.add_column("步驟", style="bold")
    t.add_column("耗時", justify="right")
    t.add_column("exit", no_wrap=True)
    t.add_column("run_id", style="dim", overflow="ellipsis", no_wrap=True)
    fails, notes = [], []
    for i, s in enumerate(real):
        last = i == len(real) - 1
        t0 = dt.datetime.fromisoformat(s["ts"]).replace(tzinfo=None)
        rc = s["exit"]
        if not last:
            t1 = dt.datetime.fromisoformat(real[i + 1]["ts"]).replace(tzinfo=None)
        elif rc is not None and s["runs"]:
            t1 = run_info(os.path.join(RUNS, s["runs"][-1]))[1] or mtime
        elif rc is None and running:
            t1 = dt.datetime.now()
        else:
            t1 = mtime
        dur = fmt_dur((t1 - t0).total_seconds()) + ("+" if rc is None and running and last else "")
        rid = (f"{s['runs'][0]} … {s['runs'][-1][16:]}" if len(s["runs"]) > 1
               else s["runs"][0] if s["runs"] else "—")
        name = Text(s["name"]) + (Text(f" ×{s['n']}", "cyan") if s["n"] > 1 else Text())
        failed = rc not in (None, "0")
        t.add_row(f"{t0:%H:%M}", name, dur, exit_text(rc, running and last), rid,
                  style="red" if failed else None)
        # 失敗的步驟另開紅框貼它的輸出（tiara.sbatch 會貼 stderr 末 40 行）；成功的只留末 3 行當備註
        if failed:
            fails.append((s["name"], rc, s["other"][-12:]))
        else:
            notes += [(s["name"], ln) for ln in s["other"][-3:]]

    parts = []
    hdr = [ln for ln in head if not ln.startswith("[env_h200] 清掉 3090")]  # 每次都一樣的長行
    if hdr:
        parts.append(Text("\n".join(hdr), "dim", overflow="ellipsis", no_wrap=True))
    if real:
        parts.append(t)
    for name, ln in notes:
        parts.append(Text.assemble((f"{name}  ", "bold dim"), (ln, "dim"), overflow="ellipsis", no_wrap=True))
    for name, rc, lines in fails:
        parts.append(Panel(Text("\n".join(lines) or "（沒有輸出）", "red", overflow="ellipsis", no_wrap=True),
                           title=Text(f"✗ {name}  exit {rc}", "bold red"), title_align="left",
                           border_style="red"))
    if skips:
        parts.append(Text(f"跳過 {len(skips)} 步：" + "、".join(skips), "dim",
                          overflow="ellipsis", no_wrap=True))
    for m in tail_msgs:
        parts.append(Text(m, "bold red" if "失敗" in m else "bold green"))

    title = Text.assemble((path, "bold"), "  ", state_text(state))
    sub = Text(f"最後寫入 {mtime:%m/%d %H:%M}", "dim")
    con.print(Panel(Group(*parts), title=title, subtitle=sub, title_align="left",
                    subtitle_align="right", border_style=STATE_STYLE.get(state, "white").split()[-1]))

    # 這個 job 最後一步的 run 目錄
    ids = [r for s in real for r in s["runs"]]
    d = os.path.join(RUNS, ids[-1]) if ids else None
    if not d or not os.path.isdir(d):
        return
    rc, end, contam = run_info(d)
    t0 = run_start(os.path.basename(d))
    el = ((end or dt.datetime.now()) - t0).total_seconds() if t0 else None
    title = Text.assemble(("最新 run  ", "bold"), (os.path.basename(d), "bold cyan"), "  ",
                          exit_text(rc), "  ", (fmt_dur(el), "dim"),
                          ("  CONTAMINATED", "bold red") if contam else "")
    parts = []
    for name, n in (("stdout", a.n), ("stderr", max(4, a.n // 2))):
        lines = tail_lines(os.path.join(d, f"{name}.log"), n)
        if lines is None:
            continue
        parts.append(Text(f"── {name} 末 {len(lines)} 行" if lines else f"── {name}（空）", "bold dim"))
        if lines:
            parts.append(Text("\n".join(map(short_paths, lines)), "red" if name == "stderr" and rc not in (None, "0") else "",
                              overflow="ellipsis", no_wrap=True))
    con.print(Panel(Group(*parts), title=title, title_align="left", border_style="dim"))


def cmd_runs(a):
    dirs = sorted(glob.glob(os.path.join(RUNS, "*/")), reverse=True)[:a.n]
    if not dirs:
        con.print(f"{RUNS} 底下沒有 run 目錄")
        return
    t = Table(box=box.SIMPLE_HEAD, header_style="bold dim", title=f"最近 {len(dirs)} 個 run",
              title_justify="left", caption=RUNS, caption_justify="left", caption_style="dim")
    t.add_column("日期", style="dim")
    t.add_column("開始", style="dim")
    t.add_column("名稱", overflow="ellipsis", no_wrap=True, max_width=52)
    t.add_column("耗時", justify="right")
    t.add_column("exit", no_wrap=True)
    t.add_column("", no_wrap=True)
    last_day = None
    for d in dirs:
        d = d.rstrip("/")
        base = os.path.basename(d)
        rc, end, contam = run_info(d)
        t0 = run_start(base)
        el = ((end or dt.datetime.now()) - t0).total_seconds() if t0 and (end or rc is None) else None
        day = f"{t0:%m/%d}" if t0 else "?"
        t.add_row(day if day != last_day else "", f"{t0:%H:%M}" if t0 else "?",
                  base[16:] if t0 else base, fmt_dur(el) + ("+" if rc is None else ""),
                  exit_text(rc), Text("CONTAMINATED", "bold red") if contam else "",
                  style="bold" if rc is None else None)
        last_day = day
    con.print(t)


def cmd_status(a):
    user = os.environ.get("USER") or getpass.getuser()
    q = slurm("squeue", "-h", "-u", user, "-o", "%i|%j|%T|%Q|%r|%M|%l|%S")
    s = slurm("sacct", "-u", user, "-X", "-n", "-P", "-S", "today",
              "-o", "jobid,jobname,state,start,end,elapsed,exitcode")
    if q is None or s is None:
        con.print("[red]squeue／sacct 跑不起來（不在 head node？）[/red]")
        raise SystemExit(1)

    t = Table(box=box.ROUNDED, header_style="bold", title="排隊中／正在跑", title_justify="left")
    for col, j in (("job", "right"), ("名稱", "left"), ("狀態", "left"), ("prio", "right"),
                   ("原因", "left"), ("已跑", "right"), ("上限", "right"), ("開始", "left")):
        t.add_column(col, justify=j, no_wrap=True)
    for ln in q.strip().splitlines():
        jid, name, st, prio, reason, used, lim, start = ln.split("|")
        start = "—" if start in ("N/A", "Unknown") else start.replace("T", " ")[5:16]
        t.add_row(jid, Text(name, "bold"), state_text(st), prio,
                  Text(reason, "dim" if reason == "None" else "yellow"), used, lim, start)
    con.print(t if t.row_count else Text("沒有排隊中或正在跑的 job", "dim"))

    t = Table(box=box.SIMPLE_HEAD, header_style="bold dim", title="今天的 job（sacct）",
              title_justify="left")
    for col, j in (("job", "right"), ("名稱", "left"), ("狀態", "left"), ("開始", "left"),
                   ("結束", "left"), ("耗時", "right"), ("exit", "left")):
        t.add_column(col, justify=j, no_wrap=True)
    for ln in s.strip().splitlines():
        jid, name, st, start, end, el, ec = ln.split("|")
        hm = lambda x: x[11:16] if "T" in x else "—"  # noqa: E731
        t.add_row(jid, name, state_text(st), hm(start), hm(end), el,
                  Text(ec, "green" if ec == "0:0" else "red"))
    if t.row_count:
        con.print(t)


def cmd_latex(a):
    log = f"{a.main}.log"
    try:
        text = open(log, encoding="utf-8", errors="replace").read()
    except OSError:
        con.print(f"[red]找不到 {log}[/red]")
        raise SystemExit(1)
    font = [m.strip() for m in re.findall(r"Font Warning: *(.*)", text)]
    boxes = [f"{m.group(1)}  {m.group(2)}" for m in re.finditer(
        r"^Overfull \\hbox \(([0-9.]+pt) too wide\) (?:in paragraph )?((?:at lines? [0-9-]+)|(?:detected at line \d+)|.*)",
        text, re.M)]
    undef = sorted(set(re.findall(r"((?:Citation|Reference) [`'][^']*' (?:on page \d+ )?undefined)", text)))
    out = re.findall(rf"Output written on {re.escape(a.main)}\.pdf \(([^)]*)\)", text)

    t = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    t.add_column(style="bold")
    t.add_column(justify="right")
    t.add_column(overflow="fold")
    for label, items in (("font warnings", font), ("overfull box", boxes), ("undefined", undef)):
        n = len(items)
        shown = "\n".join(items[:a.n]) + (f"\n… 還有 {n - a.n} 筆（見 {log}）" if n > a.n else "")
        t.add_row(label, Text(str(n), "green" if n == 0 else "bold yellow"), Text(shown, "dim"))
    ok = bool(out)
    sub = Text(f"✓ {a.main}.pdf（{out[-1]}）", "green") if ok else Text(f"✗ 沒有產生 {a.main}.pdf", "bold red")
    clean = ok and not (font or boxes or undef)
    con.print(Panel(t, title="build summary", title_align="left", subtitle=sub, subtitle_align="left",
                    border_style="green" if clean else "yellow" if ok else "red"))


# ---------------------------------------------------------------- 其他 make 指令（讀 stdin 或自己跑 Slurm）

def cmd_help(a):
    e = os.environ.get
    J, R, L, Q, F = (e("JOBNAME", "r-h200-all"), e("RESTNAME", "r-h200-rest"), e("LAF2NAME", "r-h200-la-f2"),
                     e("QWNAME", "r-h200-qwen3"), e("F2NAME", "r-h200-f2"))
    groups = [
        ("看狀態", "cyan", [
            ("q", "精簡版：lab 要等幾個人、每個 job 預測開始／結束、全部跑完要多久"),
            ("queue", "排第幾、前面是誰、什麼時候開始（code/h200_queue.py）"),
            ("queue-watch", "同上，每 5 分鐘重畫（Ctrl-C 離開）"),
            ("status", "自己的 job（squeue）＋今天跑過的（sacct）"),
            ("log [JOB=<id>] [RAW=1]", "slurm-*.out 整理成步驟表（耗時、exit、失敗的錯誤）＋最新 run 末段"),
            ("runs [N=15]", "最近的 run 目錄：開始、耗時、exit、CONTAMINATED"),
            ("gpu [JOB=<id>]", "進正在跑的 job 看一次 GPU（使用率、記憶體、功率、process）"),
            ("nvtop [JOB=<id>|PANES=1]", "所有正在跑的 job 的卡畫在同一個畫面（q 離開）；JOB= 原生 nvtop；PANES=1 tmux 分格"),
            ("usage", "lab 內每個人的用量與 fairshare（協調用）"),
        ]),
        ("送 job", "yellow", [
            ("test-submit", "只問 Slurm 預估什麼時候開始，不送"),
            ("submit", f"送 R0＋R1–R3（{J}）；已有同名 job 或有未 commit 的改動就不送"),
            ("resume FROM=<步驟>", "R1–R3 中途停了（R0 已過）：從那一步接著跑（步驟名：bash code/r_h200.sh --list）"),
            ("submit-rest", f"第一階段其餘的實驗（{R}）：接在 {J} 後面（afterok），它失敗就自動撤回"),
            ("submit-r1x", f"LongAlpaca＋F2 能耗（{L}，和 {R} 平行）與 Qwen3-30B-A3B（{Q}，等 {R} 跑完）"),
            ("submit-f2", f"F2 能耗（K=21，只做 Llama）：probe → gate → cakee → sweep → final（{F}，10 小時）"),
            ("probe", "只送 R0 硬體報告（1 小時）"),
            ("cancel JOB=<id>", "撤回"),
        ]),
        ("檢查（不用 GPU）", "green", [
            ("dry-run", "印出 R1–R3 每一步的指令"),
            ("dry-run-rest", "印出 r_h200_rest.sh 每一步的指令"),
            ("dry-run-r1x", "印出 r_h200_r1x.sh（--part la、--part qwen3）每一步的指令"),
            ("env-check", "印出 env_h200.sh 的所有路徑、標出 MISSING"),
            ("setup / setup-check", "下載模型與資料並檢查 sha256／只檢查"),
        ]),
        ("論文", "magenta", [
            ("make", "編譯 main.pdf（V=1 看 xelatex 原始輸出）"),
            ("check", "列出 TODO 與佔位符"),
            ("clean", "清掉中間檔（slurm-*.out 不動）"),
        ]),
    ]
    t = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    t.add_column(no_wrap=True)
    t.add_column()
    for gi, (title, color, rows) in enumerate(groups):
        t.add_row(Text(title, f"bold {color}"), "", end_section=False)
        for cmd, desc in rows:
            t.add_row(Text("  make " if cmd != "make" else "  ", "dim") + Text(cmd, color), desc)
        if gi < len(groups) - 1:
            t.add_row("", "")
    con.print(Panel(t, title="[bold]H200[/bold]（repo 根目錄、head node 上用）", title_align="left",
                    subtitle="docs/H200_SLURM.md", subtitle_align="right", border_style="cyan"))


def read_csv_blocks(text):
    """nvidia-smi --format=csv 的輸出，多段用 `---` 隔開 → [[dict, ...], ...]"""
    import csv
    blocks = []
    for chunk in text.split("\n---\n"):
        rows = list(csv.reader([ln for ln in chunk.strip().splitlines() if ln.strip()], skipinitialspace=True))
        if rows:
            blocks.append([dict(zip(rows[0], r)) for r in rows[1:]])
    return blocks


def bar(frac, width=20, style="green"):
    frac = max(0.0, min(1.0, frac))
    n = round(frac * width)
    col = "red" if frac > 0.9 else "yellow" if frac > 0.6 else style
    return Text("█" * n, col) + Text("░" * (width - n), "dim")


def num(x):
    m = re.search(r"[-0-9.]+", x or "")
    return float(m.group()) if m else 0.0


def cmd_gpu(a):
    import sys
    text = sys.stdin.read()
    blocks = read_csv_blocks(text)
    if not blocks:
        con.print(Text(text.strip() or "沒有輸出", "red"))
        raise SystemExit(1)
    gpus = blocks[0]
    for g in gpus:
        k = lambda p: next((v for kk, v in g.items() if kk.startswith(p)), "")  # noqa: E731
        util, mu, mt = num(k("utilization.gpu")), num(k("memory.used")), num(k("memory.total"))
        g2 = Table.grid(padding=(0, 2))
        g2.add_column(style="dim", no_wrap=True)
        g2.add_column(no_wrap=True)
        g2.add_column(no_wrap=True)
        g2.add_row("使用率", bar(util / 100), f"{util:.0f} %")
        g2.add_row("記憶體", bar(mu / mt if mt else 0), f"{mu / 1024:.1f} / {mt / 1024:.0f} GiB")
        g2.add_row("功率", Text(k("power.draw")), "")
        g2.add_row("溫度／SM 時脈", Text(f"{k('temperature.gpu')} °C／{k('clocks.sm') or k('clocks.current.sm')}"), "")
        con.print(Panel(g2, title=Text(f"GPU {k('index')}  {k('name')}", "bold"), title_align="left",
                        subtitle=Text(f"job {a.job}", "dim") if a.job else None, subtitle_align="right",
                        border_style="green", expand=False))
    procs = blocks[1] if len(blocks) > 1 else []
    t = Table(box=box.SIMPLE_HEAD, header_style="bold dim", title="process", title_justify="left")
    for col in ("pid", "process", "記憶體"):
        t.add_column(col, no_wrap=True, justify="right" if col != "process" else "left")
    for pr in procs:
        v = list(pr.values())
        t.add_row(*(v + ["", "", ""])[:3])
    con.print(t if procs else Text("（GPU 上沒有 process）", "dim"))


def slurm_ok(*args):
    """像 slurm()，但指令失敗（非 0 結束、逾時、找不到）回 None——「查不到」不能被讀成「沒有人在用」。"""
    exe = shutil.which(args[0]) or os.path.join(SLURM_BIN, args[0])
    try:
        r = subprocess.run([exe, *args[1:]], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def gpu_count(gres):
    m = re.search(r"gpu(?::[A-Za-z0-9_]+)?:(\d+)", gres or "")
    return int(m.group(1)) if m else 0


def usage_cluster(a, user):
    """啟動前總覽：H200 八張卡誰在用、各 lab 用量、/work 配額、我們自己的 job。"""
    out = slurm_ok("squeue", "-h", "-p", "h200q", "-o", "%T|%a|%u|%b|%i")
    if out is None:
        con.print(Panel(Text("squeue 查詢失敗——不能當作「卡是空的」", "bold red"), title="h200q 佔用",
                        title_align="left", border_style="red"))
    else:
        run, pend = {}, {}
        for ln in out.splitlines():
            st, acct, u, gres, _ = (ln.split("|") + [""] * 5)[:5]
            d = run if st == "RUNNING" else pend if st == "PENDING" else None
            if d is not None:
                d.setdefault(acct, {}).setdefault(u, 0)
                d[acct][u] += gpu_count(gres)
        used = sum(sum(v.values()) for v in run.values())
        mine = sum(run.get(a.account, {}).values())
        t = Table(box=box.SIMPLE_HEAD, header_style="bold dim", title="h200q 佔用（RUNNING＝正在用卡；PENDING＝排隊中的卡數）",
                  title_justify="left", title_style="bold")
        for col, j in (("lab", "left"), ("user", "left"), ("跑中卡", "right"), ("排隊卡", "right")):
            t.add_column(col, justify=j, no_wrap=True)
        for acct in sorted(set(run) | set(pend), key=lambda x: -sum(run.get(x, {}).values())):
            for u in sorted(set(run.get(acct, {})) | set(pend.get(acct, {}))):
                t.add_row(acct, u, str(run.get(acct, {}).get(u, 0)) or "0", str(pend.get(acct, {}).get(u, 0)),
                          style="bold magenta" if u == user else None)
        con.print(t)
        cap = 4
        con.print(Text("  全機 8 張：", "dim") + bar(used / 8, 16) + Text(f" {used}/8 在用，{8 - used} 空", "dim"))
        con.print(Text(f"  {a.account}：", "dim") + bar(mine / cap, 16) + Text(
            f" {mine}/{cap}（lab 上限）" + ("，已滿，新 job 會排隊" if mine >= cap else ""), "dim"))
    mine_q = slurm_ok("squeue", "-h", "-u", user, "-o", "%i|%j|%T|%M|%b|%R")
    if mine_q is None:
        con.print(Text("  我的 job：squeue 查詢失敗", "bold red"))
    else:
        rows = [r.split("|") for r in mine_q.splitlines() if r]
        t = Table(box=box.SIMPLE_HEAD, header_style="bold dim", title=f"{user} 的 job（{len(rows)} 個）",
                  title_justify="left", title_style="bold")
        for col in ("id", "名稱", "狀態", "已跑", "卡", "節點／原因"):
            t.add_column(col, no_wrap=True)
        for r in rows:
            t.add_row(r[0], r[1], Text(r[2], STATE_STYLE.get(r[2], "")), r[3], str(gpu_count(r[4])), r[5])
        con.print(t if rows else Text("  沒有 job 在跑或排隊", "dim"))
    sh = slurm_ok("sshare", "-a", "-P", "-n", "-o", "account,user,rawusage,effectvusage,fairshare")
    if sh is not None:
        accts = [r.split("|") for r in sh.splitlines() if r.count("|") >= 4 and not r.split("|")[1].strip()]
        accts = [r for r in accts if r[0].strip() not in ("root", "admin") and num(r[2]) > 0]
        accts.sort(key=lambda r: -num(r[2]))
        top = num(accts[0][2]) if accts else 1
        t = Table(box=box.SIMPLE_HEAD, header_style="bold dim", title="各 lab 累計用量（RawUsage，有衰減）",
                  title_justify="left", title_style="bold")
        t.add_column("lab", no_wrap=True)
        t.add_column("", no_wrap=True)
        t.add_column("RawUsage", justify="right")
        t.add_column("占全機", justify="right")
        for r in accts:
            t.add_row(r[0].strip(), bar(num(r[2]) / top, 20, "cyan"), f"{int(num(r[2])):,}",
                      f"{num(r[3]) * 100:.1f}%",
                      style="bold magenta" if r[0].strip() == a.account else None)
        con.print(t)
    # 硬碟：/work 配額是 group 共用，滿了所有寫入都會失敗（Errno 122）
    t = Table(box=box.SIMPLE_HEAD, header_style="bold dim", title="硬碟", title_justify="left", title_style="bold")
    for col in ("位置", "已用", "上限", "", "備註"):
        t.add_column(col, no_wrap=col != "備註", overflow="fold")
    for who, args in (("group", ["-g", a.group]), ("user", ["-u", user])):
        try:
            q = subprocess.run(["quota", "-s", *args], capture_output=True, text=True, timeout=15)
            f = q.stdout.strip().splitlines()[-1].split()
            used, lim = f[1], f[3]
            u_g, l_g = (num(used) / (1024 if used.endswith("M") else 1), num(lim) / (1024 if lim.endswith("M") else 1))
            t.add_row(f"/work 配額 {who}", used, lim, bar(u_g / l_g, 16),
                      Text("接近上限，先別送批次" if u_g / l_g > 0.95 else "", "bold red"))
        except (OSError, subprocess.SubprocessError, IndexError, ZeroDivisionError):
            t.add_row(f"/work 配額 {who}", Text("查詢失敗", "bold red"), "", "", "")
    for path, note in (("/work", "整個檔案系統（配額才是實際上限）"),
                       (os.environ.get("TMPDIR") or "/tmp", "head node 的 /tmp——滿了工具會 ENOSPC")):
        try:
            d = shutil.disk_usage(path)
            t.add_row(path, f"{d.used / 2**30:,.0f}G", f"{d.total / 2**30:,.0f}G", bar(d.used / d.total, 16),
                      Text(note, "dim"))
        except OSError:
            t.add_row(path, Text("查詢失敗", "bold red"), "", "", "")
    con.print(t)


def cmd_usage(a):
    user = os.environ.get("USER") or getpass.getuser()
    usage_cluster(a, user)
    out = slurm("sshare", "-A", a.account, "-a", "-P", "-n",
                "-o", "account,user,rawshares,normshares,rawusage,effectvusage,fairshare")
    if out is None:
        con.print("[red]sshare 跑不起來[/red]")
        raise SystemExit(1)
    rows = [ln.split("|") for ln in out.strip().splitlines()]
    tot = next((r for r in rows if not r[1]), None)
    users = sorted((r for r in rows if r[1]), key=lambda r: -num(r[4]))
    top = max((num(r[4]) for r in users), default=0) or 1
    t = Table(box=box.SIMPLE_HEAD, header_style="bold dim", title=f"{a.account} 的用量與 fairshare",
              title_justify="left", title_style="bold")
    t.add_column("user", no_wrap=True)
    t.add_column("用量（RawUsage）", no_wrap=True)
    t.add_column("", justify="right", no_wrap=True)
    t.add_column("占 lab", justify="right")
    t.add_column("FairShare", justify="right")
    for r in users:
        fs = num(r[6])
        t.add_row(r[1], bar(num(r[4]) / top, 24, "cyan"), f"{int(num(r[4])):,}", f"{num(r[5]) * 100:.1f}%",
                  Text(r[6], "green" if fs >= 0.2 else "yellow" if fs >= 0.15 else "red"),
                  style="bold magenta" if r[1] == user else None)
    con.print(t)
    if tot:
        con.print(Text(f"  lab 合計 RawUsage {int(num(tot[4])):,}，占全叢集 {num(tot[5]) * 100:.1f}%"
                       "（FairShare 越低，排隊優先序越低）", "dim"))


def cmd_env(a):
    import sys
    t = Table(box=box.SIMPLE_HEAD, header_style="bold dim", pad_edge=False)
    t.add_column("變數", style="bold", no_wrap=True)
    t.add_column("路徑", overflow="fold")
    t.add_column("", no_wrap=True)
    notes, bad = [], 0
    for ln in sys.stdin.read().splitlines():
        m = re.match(r"\s+(\S+)\s+(.*?)\s+(ok|MISSING.*)$", ln) or re.match(r"\s+(\S+)\s+(''|\S*)()$", ln)
        if m:
            ok = m.group(3) in ("ok", "")
            bad += not ok
            st = Text("✓ ok", "green") if m.group(3) == "ok" else Text("—", "dim") if ok else \
                Text("✗ " + m.group(3), "bold red")
            t.add_row(m.group(1), Text(short_paths(m.group(2)), "" if ok else "red"), st)
        elif ln.startswith("[env_h200] 清掉 3090"):
            notes.append(Text("[env_h200] 清掉 3090 的設定（這台沒有 /ssd7）", "dim"))
        elif ln.strip():
            notes.append(Text(short_paths(ln), "dim", overflow="ellipsis", no_wrap=True))
    con.print(Panel(Group(*notes, t), title="env_h200.sh", title_align="left",
                    subtitle=Text(f"{bad} 個 MISSING", "bold red") if bad else Text("全部 ok", "green"),
                    subtitle_align="right", border_style="red" if bad else "green"))


def cmd_dryrun(a):
    import sys
    t = Table(box=box.SIMPLE_HEAD, header_style="bold dim", pad_edge=False, title=a.title,
              title_justify="left", title_style="bold")
    t.add_column("#", justify="right", style="dim")
    t.add_column("步驟", style="bold cyan", no_wrap=True)
    t.add_column("指令", overflow="fold")
    t.add_column("", style="yellow")
    n = 0
    for ln in sys.stdin.read().splitlines():
        ln = short_paths(ln)
        m = re.match(r"\[([^\]]+)\] (.*)$", ln)
        if m and not m.group(1).startswith(("env_h200", "r_h200", "skip")):
            cmd, note = m.group(2), ""
            mm = re.search(r"\s*[（(]([^（(]*)[)）]\s*$", cmd)
            if mm:
                cmd, note = cmd[:mm.start()], mm.group(1)
            n += 1
            t.add_row(str(n), m.group(1), Text(cmd.strip()), note)
        elif ln.startswith("[env_h200] 清掉 3090"):
            continue
        elif ln.strip():
            t.add_row("", "", Text(ln, "dim"), "")
    con.print(t)


def cmd_testsubmit(a):
    import sys
    text = sys.stdin.read().strip()
    m = re.search(r"Job (\d+) to start at (\S+) using (\d+) processors on nodes (\S+) in partition (\S+)", text)
    if not m:
        con.print(Panel(Text(text or "（沒有輸出）", "red"), title="sbatch --test-only", border_style="red"))
        raise SystemExit(1)
    t0 = dt.datetime.fromisoformat(m.group(2))
    wait = (t0 - dt.datetime.now()).total_seconds()
    g = Table.grid(padding=(0, 2))
    g.add_column(style="dim")
    g.add_column()
    g.add_row("預估開始", Text.assemble((f"{t0:%m/%d %H:%M}", "bold"),
                                     (f"（{fmt_dur(wait)} 後）" if wait > 0 else "（馬上）", "dim")))
    g.add_row("節點／partition", f"{m.group(4)}／{m.group(5)}，{m.group(3)} 核")
    con.print(Panel(g, title=Text(f"sbatch --test-only {a.script}", "bold"), title_align="left",
                    subtitle=Text("只問、沒送；Slurm 的預估常常偏悲觀", "dim"), subtitle_align="right",
                    border_style="cyan", expand=False))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("log")
    pl.add_argument("--job")
    pl.add_argument("--raw", action="store_true", help="不整理，印 slurm-*.out 原始末 40 行")
    pl.add_argument("-n", type=int, default=12, help="最新 run 的 stdout 印幾行（stderr 一半）")
    pr = sub.add_parser("runs")
    pr.add_argument("-n", type=int, default=15)
    sub.add_parser("status")
    px = sub.add_parser("latex")
    px.add_argument("main", nargs="?", default="main")
    px.add_argument("-n", type=int, default=5, help="每種警告列幾筆")
    sub.add_parser("help")
    sub.add_parser("gpu").add_argument("--job")
    sp = sub.add_parser("usage")
    sp.add_argument("--account", default="chihchunghsu")
    sp.add_argument("--group", default="ChihChungHsu")
    sub.add_parser("env")
    sub.add_parser("dryrun").add_argument("--title", default="")
    sub.add_parser("testsubmit").add_argument("--script", default="")
    a = p.parse_args()
    {"log": cmd_log, "runs": cmd_runs, "status": cmd_status, "latex": cmd_latex, "help": cmd_help,
     "gpu": cmd_gpu, "usage": cmd_usage, "env": cmd_env, "dryrun": cmd_dryrun,
     "testsubmit": cmd_testsubmit}[a.cmd](a)


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)   # `| head` 時安靜結束
    main()
