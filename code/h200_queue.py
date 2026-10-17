#!/usr/bin/env python3
"""h200_queue.py — 我的 job 在 h200q 排第幾、前面是誰、大概還要等多久（2026-10-15）

    python3 code/h200_queue.py                 # 自己（$USER）的 job
    python3 code/h200_queue.py --job 20417     # 指定 job
    python3 code/h200_queue.py --no-history    # 不查 sacct（快一點；少了「照過去執行時間」那個模擬）
    python3 code/h200_queue.py --watch 300     # 每 5 分鐘重畫
    python3 code/h200_queue.py --brief         # 精簡版（make q）：lab 要等幾個人、我每個 job 的預測開始／結束（含等前置的）
    python3 code/h200_queue.py --gpus 2 --time 2-00:00:00   # 沒有 job 時：「假如現在送」用的卡數與時間上限

只讀 Slurm（squeue、sprio、scontrol、sacctmgr、sacct），不送、不改任何 job。用系統的 python3 就能跑（標準庫＋rich；系統的 python3 已經有 rich）。

三個「什麼時候開始」：
  1. Slurm 的預估（squeue --start）：排程器自己算的，常常很悲觀，也常常是 N/A。
  2. 模擬・跑滿上限：假設正在跑的、排在前面的 job 都用到自己的時間上限。照優先序一個一個放，
     遵守總卡數與每個 account 的上限（lab 4 張），能插空就插空（跟 backfill 一樣）。
  3. 模擬・照過去：同上，但執行時間改用「同一人、同一個時間上限」過去 30 天在 h200q 的中位數；
     正在跑的只看過去跑得比它現在還久的那些（條件中位數）。樣本 < 3 個就用上限。參考值。
  ⚠️ 2、3 是推算，不是保證：之後有人送出優先序更高的 job，就會排到前面；優先序也會隨時間變（Age、FairShare）。
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

PART = "h200q"
SLURM_BIN = "/cm/local/apps/slurm/current/bin"
# 不是「現在就在搶卡」的 pending 原因（hold、等前置 job、等開始時間）
NOT_COMPETING = ("Dependency", "DependencyNeverSatisfied", "JobHeldUser", "JobHeldAdmin", "BeginTime",
                 "PartitionDown", "PartitionInactive", "InvalidAccount", "InvalidQOS")
FAR = timedelta(days=365)
STATE_STYLE = {"RUNNING": "green", "PENDING": "yellow", "COMPLETING": "cyan", "SUSPENDED": "red"}
con = Console(highlight=False)


def sh(*cmd: str) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        sys.exit(f"找不到指令 {cmd[0]}（Slurm 不在 PATH？）")
    if r.returncode != 0:
        sys.exit(f"指令失敗：{' '.join(cmd)}\n{r.stderr.strip()}")   # 查不到就停，不當成「沒有」（CLAUDE.md 規則 7）
    return r.stdout


def ptime(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def plimit(s: str) -> timedelta:
    """Slurm 的時間格式：D-HH:MM:SS、HH:MM:SS、MM:SS、MM、UNLIMITED。"""
    if s in ("UNLIMITED", "INVALID", "NOT_SET", "N/A", ""):
        return FAR
    d = 0
    if "-" in s:
        d, s = s.split("-", 1)
        d = int(d)
    p = [int(x) for x in s.split(":")]
    if len(p) == 1:
        h, m, sec = (p[0], 0, 0) if d else (0, p[0], 0)      # D-HH 或 MM
    elif len(p) == 2:
        h, m, sec = (0, p[0], p[1]) if d == 0 else (p[0], p[1], 0)
    else:
        h, m, sec = p
    return timedelta(days=d, hours=h, minutes=m, seconds=sec)


def fmt_td(td: timedelta) -> str:
    s = int(td.total_seconds())
    if s < 0:
        return "-" + fmt_td(-td)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m = s // 60
    return f"{d}天{h}時{m:02d}分" if d else f"{h}時{m:02d}分"


def fmt_dt(t: datetime | None) -> str:
    return t.strftime("%m/%d %H:%M") if t else "N/A"


def gpus_of(tres: str) -> int:
    m = re.search(r"gpu(?::[^:,]+)?[:=](\d+)", tres)
    return int(m.group(1)) if m else 0


def load_jobs():
    out = sh("squeue", "-p", PART, "-h", "-r", "-o", "%i|%u|%a|%T|%Q|%r|%b|%l|%S|%V|%e|%M|%E|%j")
    jobs = []
    for line in out.splitlines():
        f = line.split("|", 13)
        if len(f) < 14:
            continue
        jid, user, acct, state, prio, reason, tres, lim, start, submit, end, used, dep, name = f
        jobs.append(dict(id=jid, user=user, acct=acct.lower(), state=state, prio=int(prio) if prio.isdigit() else 0,
                         reason=reason, gpus=gpus_of(tres), limit=plimit(lim), start=ptime(start),
                         submit=ptime(submit), end=ptime(end), name=name, deps=parse_deps(dep)))
    return jobs


def parse_deps(s: str) -> list[tuple[str, str, str]]:
    """squeue %E：'afterok:20704(unfulfilled),after:20705(unfulfilled)' → [(type, job id, 狀態)]。
    OR（?）當成 AND 處理（偏晚）；after:<id>+<分鐘> 的延遲不算。"""
    if s in ("", "(null)"):
        return []
    out = []
    for part in re.split(r"[,?]", s):
        m = re.match(r"(\w+):([\d_*]+)(?:\+\d+)?(?:\(([^)]*)\))?", part)
        if m:
            out.append((m.group(1), m.group(2), m.group(3) or ""))
    return out


def total_gpus() -> int:
    n = 0
    for line in sh("sinfo", "-p", PART, "-h", "-N", "-o", "%N %G").splitlines():
        n += sum(int(x) for x in re.findall(r"gpu(?::[^:(]+)?:(\d+)", line))
    if n == 0:
        sys.exit("查不到 h200q 的 GPU 數（sinfo %G）")
    return n


def acct_limit(acct: str) -> int | None:
    """account 的 GPU 上限：association 的 GrpTRES，沒有就用 QOS 的 MaxTRESPA。查不到回 None（模擬時不設限並註明）。"""
    out = sh("sacctmgr", "-n", "-P", "show", "assoc", f"account={acct}", "format=user,grptres")
    for line in out.splitlines():
        u, g = (line.split("|") + [""])[:2]
        if u == "" and "gpu" in g:
            return gpus_of(g)
    out = sh("sacctmgr", "-n", "-P", "show", "qos", f"{PART}_1g", "format=maxtrespa")
    return gpus_of(out) or None


def default_acct(user: str) -> str:
    out = sh("sacctmgr", "-n", "-P", "show", "user", user, "format=defaultaccount").strip()
    if not out:
        sys.exit(f"查不到 {user} 的 default account（sacctmgr show user）")
    return out.splitlines()[0].lower()


MIN_N = 3   # 同一人、同一個時間上限的歷史 job 少於這個數 → 不猜，用上限


def history(days: int = 30) -> dict[tuple[str, int], list[int]]:
    """(user, 時間上限的分鐘數) → 過去 days 天在 h200q 跑完的 job 的實際秒數。
    只拿同樣上限的比：同一個人短的測試 job 和 2 天的正式 job 混在一起，中位數會小得離譜。"""
    out = sh("sacct", "-a", "-r", PART, "-S", f"now-{days}days", "-E", "now", "-X", "-n", "-P",
             "-o", "user,elapsedraw,timelimitraw,state")
    per: dict[tuple[str, int], list[int]] = {}
    for line in out.splitlines():
        f = line.split("|")
        if len(f) < 4 or not f[1].isdigit() or int(f[1]) <= 0 or not f[2].isdigit():
            continue
        if f[3].split()[0] in ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL"):
            per.setdefault((f[0], int(f[2])), []).append(int(f[1]))
    return per


def typical(hist, user: str, limit: timedelta, ran: timedelta = timedelta(0)) -> timedelta | None:
    """同一人、同一上限的過去 job 裡，跑得比 ran 久的那些的中位數（正在跑的 job：已經跑了 ran）。樣本不夠回 None。"""
    v = [s for s in hist.get((user, int(limit.total_seconds() // 60)), []) if s > ran.total_seconds()]
    return timedelta(seconds=statistics.median(v)) if len(v) >= MIN_N else None


def simulate(now, running, pending, total, limits, dur_of, end_of):
    """照優先序放 pending job；每個 job 找最早一個「整段時間裡總卡數、account 上限都夠」的開始時間（含插空）。"""
    iv = [(r["start"] or now, max(end_of(r), now), r["gpus"], r["acct"]) for r in running]

    def ok(t0, t1, g, a):
        pts = [t0] + [s for s, _, _, _ in iv if t0 < s < t1]
        for p in pts:
            used = sum(gg for s, e, gg, _ in iv if s <= p < e)
            if used + g > total:
                return False
            lim = limits.get(a)
            if lim is not None and sum(gg for s, e, gg, aa in iv if aa == a and s <= p < e) + g > lim:
                return False
        return True

    res = {}
    for j in pending:
        d = dur_of(j)
        if j["gpus"] > total or (limits.get(j["acct"]) is not None and j["gpus"] > limits[j["acct"]]):
            res[j["id"]] = None
            continue
        cands = sorted({now} | {e for _, e, _, _ in iv if e > now})
        t = next((c for c in cands if ok(c, c + d, j["gpus"], j["acct"])), None)
        res[j["id"]] = t
        if t is not None:
            iv.append((t, t + d, j["gpus"], j["acct"]))
    return res


def simulate_dag(now, running, pending, all_ids, total, limits, dur_of, end_of):
    """事件驅動版：pending 裡可以有等前置的 job（after → 前置開始後、afterok／afterany → 前置結束後；afterok 假設會成功）。
    每個時間點把「前置已滿足」的 job 依優先序放，總卡數與 account 上限都夠就開始（job 都是 1–2 張卡，近似 backfill）。
    前置是 hold、永遠不會滿足、或 failed 的 → 放不進去（不在回傳的 start 裡）。回傳 (start, end)，含正在跑的。"""
    start = {r["id"]: r["start"] or now for r in running}
    end = {r["id"]: max(end_of(r), now) for r in running}
    sim_ids = set(start) | {j["id"] for j in pending}

    def targets(d):
        b = d.removesuffix("_*")
        return [i for i in all_ids if i == d or (("_" not in d or d.endswith("_*")) and i.split("_")[0] == b)]

    def ready_at(j):   # 前置都已排進模擬 → 最早可開始的時間；還不知道 → None
        t = now
        for typ, d, st in j["deps"]:
            if "failed" in st:
                return None
            for i in targets(d):     # 不在佇列 = 已經結束（squeue 只列沒滿足的）
                if i not in sim_ids:
                    return None
                src = start if typ == "after" else end
                if i not in src:
                    return None
                t = max(t, src[i])
        return t

    act = [(end[r["id"]], r["gpus"], r["acct"]) for r in running]
    waiting = sorted(pending, key=lambda j: (-j["prio"], j["submit"] or now, j["id"]))
    t = now
    while waiting:
        act = [x for x in act if x[0] > t]
        progress = True
        while progress:          # 同一刻開始的 job 會讓 after:<它> 的 job 也能在這一刻開始，所以掃到沒有新的為止
            progress = False
            for j in list(waiting):
                r = ready_at(j)
                if r is None or r > t:
                    continue
                lim = limits.get(j["acct"])
                if sum(g for _, g, _ in act) + j["gpus"] > total or \
                        (lim is not None and sum(g for _, g, a in act if a == j["acct"]) + j["gpus"] > lim):
                    continue
                start[j["id"]], end[j["id"]] = t, t + dur_of(j)
                act.append((end[j["id"]], j["gpus"], j["acct"]))
                waiting.remove(j)
                progress = True
                break            # 從優先序最高的重新看
        nxt = [e for e, _, _ in act if e > t] + [r for j in waiting if (r := ready_at(j)) is not None and r > t]
        if not nxt:
            break
        t = min(nxt)
    return start, end


def render_brief(a):
    """make q：lab 要等幾個人、我的 job 預測什麼時候開始／結束、全部跑完要多久。"""
    now = datetime.now().replace(microsecond=0)
    jobs = load_jobs()
    total = total_gpus()
    acct = default_acct(a.user)
    limits = {ac: acct_limit(ac) for ac in {j["acct"] for j in jobs} | {acct}}
    hist = {} if a.no_history else history()
    running = [j for j in jobs if j["state"] in ("RUNNING", "COMPLETING", "SUSPENDED")]
    pend_all = [j for j in jobs if j["state"] == "PENDING"]
    # 會被模擬的 pending：在搶卡的＋等前置的（前置是 hold／failed 的，simulate_dag 自己會放不進去）
    simmable = [j for j in pend_all if j["reason"] not in NOT_COMPETING or j["reason"] == "Dependency"]
    held = [j for j in pend_all if j["acct"] == acct and j["reason"] in ("JobHeldUser", "JobHeldAdmin")]
    all_ids = [j["id"] for j in jobs]

    def lim_end(r):
        return r["end"] or ((r["start"] or now) + r["limit"])

    def hist_end(r):
        m = typical(hist, r["user"], r["limit"], now - r["start"] if r["start"] else timedelta(0))
        return (r["start"] or now) + min(m, r["limit"]) if m else lim_end(r)

    def hist_dur(j):
        m = typical(hist, j["user"], j["limit"])
        return min(m, j["limit"]) if m else j["limit"]

    s_lim, e_lim = simulate_dag(now, running, simmable, all_ids, total, limits, lambda j: j["limit"], lim_end)
    s_his, e_his = simulate_dag(now, running, simmable, all_ids, total, limits, hist_dur, hist_end) \
        if hist else (s_lim, e_lim)

    def count(js):
        by = {}
        for j in js:
            by[j["user"]] = by.get(j["user"], 0) + 1
        return "、".join(f"{'你' if u == a.user else u} {n}" for u, n in sorted(by.items(), key=lambda x: -x[1]))

    def rel(t):
        if not t:
            return Text("—", "dim")
        d = fmt_td(t - now)
        return Text.assemble((fmt_dt(t), "bold"), (f" +{d[:d.index('時') + 1] if '天' in d else d}", "dim"))

    lab_run = [r for r in running if r["acct"] == acct]
    lab_comp = [j for j in pend_all if j["acct"] == acct and j["reason"] not in NOT_COMPETING]
    used = sum(r["gpus"] for r in running)
    lab_used = sum(r["gpus"] for r in lab_run)
    lim = limits.get(acct)
    con.print(Rule(Text(f"{PART}  {now:%m/%d %H:%M}", "bold"), style="cyan"))
    con.print(Text.assemble("  全部 GPU ", (f"{used}/{total}", "bold"), "｜lab ", (acct, "cyan"), " ",
                            (f"{lab_used}/{lim if lim is not None else '?'}", "bold yellow"), " 張"))
    con.print(Text.assemble("  lab 在跑 ", (str(len(lab_run)), "bold"), f"（{count(lab_run) or '無'}）｜在搶卡 ",
                            (str(len(lab_comp)), "bold"), f"（{count(lab_comp) or '無'}）｜hold ",
                            (str(len(held)), "bold"), (f"（{count(held)}；放行後會一起搶 lab 的 {lim} 張）" if held else "", "dim")))

    mine = [j for j in jobs if j["user"] == a.user]
    if not mine:
        con.print(Text(f"\n（{a.user} 在 {PART} 沒有 job；make queue 看「假如現在送」）", "dim"))
        return
    t = Table(box=box.SIMPLE_HEAD, header_style="bold dim", pad_edge=False, title="你的 job（照過去 30 天推算）",
              title_justify="left", title_style="bold")
    t.add_column("job", no_wrap=True)
    t.add_column("名稱", no_wrap=True, overflow="ellipsis", max_width=15)
    t.add_column("前面(lab)", justify="right", no_wrap=True)
    t.add_column("預測開始", no_wrap=True)
    t.add_column("預測結束", no_wrap=True)
    lab_ids = [j for j in running + simmable if j["acct"] == acct]
    for j in sorted(mine, key=lambda j: (s_his.get(j["id"]) or now + FAR, j["id"])):
        st = s_his.get(j["id"])
        if j["state"] != "PENDING":
            ahead = Text("跑中", "green")
        elif st is None:
            ahead = Text(f"{j['reason']}，模擬不到", "red")
        else:
            # 要等的 lab job：在我之前開始、到現在還沒結束（會佔 lab 的卡）的；同一刻開始的只算優先序比我高的
            aj = [x for x in lab_ids if x["id"] != j["id"] and s_his.get(x["id"]) is not None
                  and e_his[x["id"]] > now and (s_his[x["id"]] < st or (s_his[x["id"]] == st and x["prio"] > j["prio"]))]
            others = {x["user"] for x in aj if x["user"] != a.user}
            ahead = Text(f"{len(aj)} 個／{len(others)} 人")
        t.add_row(j["id"], j["name"], ahead,
                  rel(st) if j["state"] == "PENDING" else Text(fmt_dt(j["start"]), "dim"), fmt_dt(e_his.get(j["id"])))
    con.print(t)

    pend = [j for j in mine if j["state"] == "PENDING"]
    sim_pend = [j for j in pend if s_his.get(j["id"])]

    def last(d, js):
        v = [d[j["id"]] for j in js if d.get(j["id"])]
        return max(v) if v else None

    rows = []
    if sim_pend:
        nxt = min(s_his[j["id"]] for j in sim_pend)
        rows.append(("下一個開始", rel(nxt), Text(fmt_dt(min(s_lim[j["id"]] for j in sim_pend if s_lim.get(j["id"]))), "dim")))
        rows.append(("全部開始", rel(last(s_his, sim_pend)), Text(fmt_dt(last(s_lim, sim_pend)), "dim")))
    rows.append(("全部跑完", rel(last(e_his, mine)), Text(fmt_dt(last(e_lim, mine)), "dim")))
    g = Table.grid(padding=(0, 2))
    g.add_column(style="bold")
    g.add_column()
    g.add_column()
    g.add_row(Text("合計", "dim"), Text("照過去 30 天", "dim"), Text("跑滿上限", "dim"))
    for r in rows:
        g.add_row(*r)
    if len(sim_pend) < len(pend):
        g.add_row("模擬不到", Text(f"{len(pend) - len(sim_pend)} 個（前置是 hold／failed，或放不進去）", "red"), "")
    con.print(g)
    con.print(Text("⚠ 推算：前置 job 假設會成功；之後有優先序更高的 job（含 lab 的 hold 放行）就會排到前面。"
                   "「前面(lab)」＝從現在到它開始，佔過 lab 卡的 job 數（含你自己的）／其他人數。詳細：make queue", "dim italic"))


def render(a):
    now = datetime.now().replace(microsecond=0)
    jobs = load_jobs()
    total = total_gpus()
    running = [j for j in jobs if j["state"] in ("RUNNING", "COMPLETING", "SUSPENDED")]
    pend_all = [j for j in jobs if j["state"] == "PENDING"]
    competing = sorted([j for j in pend_all if j["reason"] not in NOT_COMPETING],
                       key=lambda j: (-j["prio"], j["submit"] or now, j["id"]))
    waiting_dep = [j for j in pend_all if j["reason"] in NOT_COMPETING]
    accts = {j["acct"] for j in jobs}
    limits = {ac: acct_limit(ac) for ac in accts}
    hist = {} if a.no_history else history()

    mine = [j for j in jobs if (j["id"] == a.job or j["id"].split("_")[0] == a.job)] if a.job \
        else [j for j in jobs if j["user"] == a.user]
    used = sum(j["gpus"] for j in running)

    cfg = sh("scontrol", "show", "config")
    w_age = re.search(r"PriorityWeightAge\s*=\s*(\d+)", cfg)
    max_age = re.search(r"PriorityMaxAge\s*=\s*(\d+)-", cfg)
    age_per_day = int(w_age.group(1)) / int(max_age.group(1)) if w_age and max_age and int(max_age.group(1)) else None

    con.print(Rule(Text(f"{PART} 佇列  {now:%Y-%m-%d %H:%M}", "bold"), style="cyan"))
    bar = Text("█" * used, "bold yellow") + Text("░" * (total - used), "green")
    con.print(Text.assemble("  GPU ", bar, f"  使用中 ", (str(used), "bold yellow"), f"／{total}，空 ",
                            (str(total - used), "bold green"), "｜正在跑 ", (str(len(running)), "bold"),
                            "｜排隊 ", (str(len(pend_all)), "bold"),
                            (f"（在搶卡 {len(competing)}、hold 或等前置 {len(waiting_dep)}）", "dim")))

    # 模擬
    def lim_end(r):
        return r["end"] or ((r["start"] or now) + r["limit"])

    def ran_of(r):
        return now - r["start"] if r["start"] else timedelta(0)

    def hist_end(r):
        m = typical(hist, r["user"], r["limit"], ran_of(r))
        return (r["start"] or now) + min(m, r["limit"]) if m else lim_end(r)

    def hist_dur(j):
        m = typical(hist, j["user"], j["limit"])
        return min(m, j["limit"]) if m else j["limit"]

    sim_lim = simulate(now, running, competing, total, limits, lambda j: j["limit"], lim_end)
    sim_hist = simulate(now, running, competing, total, limits, hist_dur, hist_end) if hist else {}

    def when(t):
        return Text.assemble((fmt_dt(t), "bold"), (f"（{fmt_td(t - now)} 後）", "dim")) if t else Text("N/A", "dim")

    def ahead_text(ahead):
        return (f"前面 {len(ahead)} 個 job（{len({c['user'] for c in ahead})} 個人）、"
                f"共要 {sum(c['gpus'] for c in ahead)} 張卡")

    if not mine:
        con.print(Text(f"\n（{a.job or a.user} 在 {PART} 沒有 job）", "dim"))
    if not mine and not a.job:
        # 新 job 的優先序要送出後才知道（FairShare 會變、Age 從 0 起算），所以只給兩端：排最後、排最前
        acct = default_acct(a.user)
        if acct not in limits:
            limits[acct] = acct_limit(acct)
        hyp = dict(id="(假如現在送)", user=a.user, acct=acct, gpus=a.gpus, limit=plimit(a.time))
        worst = simulate(now, running, competing + [hyp], total, limits, lambda j: j["limit"], lim_end)[hyp["id"]]
        best = simulate(now, running, [hyp] + competing, total, limits, lambda j: j["limit"], lim_end)[hyp["id"]]
        g = Table.grid(padding=(0, 2))
        g.add_column(style="dim", no_wrap=True)
        g.add_column()
        g.add_row("假設", f"{a.gpus} 張卡、上限 {fmt_td(hyp['limit'])}、account {acct}（--gpus／--time 可改）")
        g.add_row("排在前面", Text.assemble(("最多 " + str(len(competing)), "bold magenta"),
                                         f" 個：{ahead_text(competing)}" if competing else " 個：現在沒有人在搶卡"))
        lim_ac = limits.get(acct)
        run_ac = sum(r["gpus"] for r in running if r["acct"] == acct)
        g.add_row("account", f"{acct} 正在用 {run_ac} 張，上限 {lim_ac if lim_ac is not None else '查不到'}")
        if competing:
            g.add_row("模擬・排最後", when(worst) if worst else Text("放不進去", "red"))
            g.add_row("模擬・排最前", when(best) if best else Text("放不進去", "red"))
        else:
            g.add_row("模擬", when(worst) if worst else Text("放不進去", "red"))
        con.print(Panel(g, title=Text(" 假如現在送 ", "bold"), title_align="left", border_style="magenta",
                        padding=(0, 1)))
    for j in mine:
        who = "你的 job" if j["user"] == a.user else f"{j['user']} 的 job"
        g = Table.grid(padding=(0, 2))
        g.add_column(style="dim", no_wrap=True)
        g.add_column()
        style = STATE_STYLE.get(j["state"], "white")
        title = Text.assemble((f" {who} {j['id']} ", "bold"), (j["name"], "bold cyan"), "  ",
                              (j["state"], style), " ")
        if j["state"] != "PENDING":
            g.add_row("開始", Text.assemble((fmt_dt(j["start"]), "bold"),
                                          f"，已跑 {fmt_td(now - j['start']) if j['start'] else 'N/A'}"))
            g.add_row("最晚結束", Text(fmt_dt(j["end"]), "bold"))
            g.add_row("GPU／上限", f"{j['gpus']} 張／{fmt_td(j['limit'])}")
            con.print(Panel(g, title=title, title_align="left", border_style=style, padding=(0, 1)))
            continue
        g.add_row("原因", Text(j["reason"], "yellow"))
        g.add_row("GPU／上限", f"{j['gpus']} 張／{fmt_td(j['limit'])}")
        g.add_row("送出", f"{fmt_dt(j['submit'])}，已等 {fmt_td(now - j['submit']) if j['submit'] else 'N/A'}")
        pr = sh("sprio", "-h", "-j", j["id"].split("_")[0], "-o", "%Y|%A|%F|%J|%Q|%N").strip().split("|")
        if len(pr) >= 6:
            grow = f"；Age 每天約 +{age_per_day:.1f}（最多 +{w_age.group(1)}）" if age_per_day else ""
            g.add_row("優先序", Text.assemble((pr[0], "bold"),
                      (f" ＝ FairShare {pr[2]} ＋ QOS {pr[4]} ＋ Age {pr[1]} ＋ JobSize {pr[3]}"
                       f"{' ＋ Nice ' + pr[5] if pr[5] not in ('0', '') else ''}{grow}", "dim")))
        if j["reason"] in NOT_COMPETING:
            g.add_row("", Text(f"現在不在搶卡（{j['reason']}），放行或前置 job 完成後才會排", "italic"))
        else:
            rank = next(i for i, c in enumerate(competing, 1) if c["id"] == j["id"])
            ahead = competing[:rank - 1]
            g.add_row("排名", Text.assemble(("★ 第 " + str(rank) + " 位", "bold magenta"), f"：{ahead_text(ahead)}"))
            lim_ac = limits.get(j["acct"])
            run_ac = sum(r["gpus"] for r in running if r["acct"] == j["acct"])
            g.add_row("account", f"{j['acct']} 正在用 {run_ac} 張，上限 {lim_ac if lim_ac is not None else '查不到'}")
            g.add_row("Slurm 預估", when(j["start"]))
            t = sim_lim.get(j["id"])
            g.add_row("模擬・跑滿上限", when(t) if t else Text("放不進去", "red"))
            if sim_hist:
                g.add_row("模擬・照過去 30 天", when(sim_hist.get(j["id"])) + Text("  ← 參考值", "dim"))
        con.print(Panel(g, title=title, title_align="left", border_style=style, padding=(0, 1)))

    def hist_s(j, ran=timedelta(0)):
        m = typical(hist, j["user"], j["limit"], ran)
        return fmt_td(m) if m else "用上限"

    my_ids = {j["id"] for j in mine}
    me_style = "bold magenta"
    t = Table(box=box.SIMPLE_HEAD, header_style="bold dim", title="在搶卡的 pending（依優先序；★＝你）",
              title_justify="left", title_style="bold")
    for col, jj in (("", "left"), ("#", "right"), ("job", "left"), ("user", "left"), ("account", "left"),
                    ("prio", "right"), ("GPU", "right"), ("上限", "right"), ("已等", "right"),
                    ("過去同上限", "right"), ("模擬開始：上限", "left"), ("過去", "left"), ("原因", "left")):
        t.add_column(col, justify=jj, no_wrap=True)
    for i, c in enumerate(competing, 1):
        me = c["id"] in my_ids
        t.add_row("★" if me else "", str(i), c["id"], c["user"], c["acct"], str(c["prio"]), str(c["gpus"]),
                  fmt_td(c["limit"]), fmt_td(now - c["submit"]) if c["submit"] else "N/A", hist_s(c),
                  fmt_dt(sim_lim.get(c["id"])), fmt_dt(sim_hist.get(c["id"])) if sim_hist else "—",
                  Text(c["reason"], "yellow"), style=me_style if me else None)
    con.print(t if competing else Text("在搶卡的 pending：沒有", "dim"))

    t = Table(box=box.SIMPLE_HEAD, header_style="bold dim", title="正在跑（依結束時間）",
              title_justify="left", title_style="bold")
    for col, jj in (("job", "left"), ("user", "left"), ("account", "left"), ("GPU", "right"), ("開始", "right"),
                    ("已跑", "right"), ("最晚結束", "right"), ("照過去結束", "right")):
        t.add_column(col, justify=jj, no_wrap=True)
    for r in sorted(running, key=lim_end):
        t.add_row(r["id"], r["user"], r["acct"], str(r["gpus"]), fmt_dt(r["start"]),
                  fmt_td(now - r["start"]) if r["start"] else "N/A", fmt_dt(lim_end(r)),
                  fmt_dt(hist_end(r)) if hist else "—", style=me_style if r["user"] == a.user else None)
    con.print(t)

    if waiting_dep:
        # squeue -r 把 array 展開成一個 task 一列，所以「個數」是 task 數；「送出」是 sbatch 的次數
        by = {}
        for w in waiting_dep:
            k = (w["user"], w["acct"], w["reason"])
            by.setdefault(k, []).append(w)
        my_acct = default_acct(a.user) if a.user else ""
        t = Table(box=box.SIMPLE_HEAD, header_style="bold dim", title="現在不搶卡（hold／等前置 job；放行後依優先序加入）",
                  title_justify="left", title_style="bold")
        for col, jj in (("user", "left"), ("account", "left"), ("task", "right"), ("送出", "right"),
                        ("最早送出", "right"), ("原因", "left"), ("", "left")):
            t.add_column(col, justify=jj, no_wrap=True)
        for (u, ac, rs), ws in sorted(by.items(), key=lambda x: -len(x[1])):
            first = min((w["submit"] for w in ws if w["submit"]), default=None)
            note = []
            if rs == "DependencyNeverSatisfied":
                note.append(("永遠不會跑（前置失敗）", "red"))
            if ac == my_acct and u != a.user:
                note.append(("同 account，放行後共用上限", "yellow"))
            t.add_row(u, ac, str(len(ws)), str(len({w["id"].split("_")[0] for w in ws})),
                      f"{(now - first).days} 天前" if first else "N/A", Text(rs, "dim"),
                      Text("；").join(Text(x, st) for x, st in note) if note else "",
                      style=me_style if u == a.user else None)
        con.print(t)
    con.print(Text("⚠ 模擬是推算：之後有優先序更高的 job 送進來就會排到前面；跑滿上限那個通常偏晚、照過去那個可能偏早。",
                   "dim italic"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--job", help="job id（預設：自己所有在 h200q 的 job）")
    ap.add_argument("--user", default=os.environ.get("USER", ""))
    ap.add_argument("--no-history", action="store_true", help="不查 sacct")
    ap.add_argument("--gpus", type=int, default=1, help="沒有 job 時，「假如現在送」的卡數（預設 1）")
    ap.add_argument("--time", default="1-00:00:00", help="沒有 job 時，「假如現在送」的時間上限（預設 1 天）")
    ap.add_argument("--watch", type=int, metavar="秒", help="每隔幾秒重畫一次（Ctrl-C 離開）")
    ap.add_argument("--brief", action="store_true", help="精簡版（make q）：lab 要等幾個人、預測開始／結束、全部跑完要多久")
    a = ap.parse_args()
    draw = render_brief if a.brief else render
    if not shutil.which("squeue") and os.path.isdir(SLURM_BIN):
        os.environ["PATH"] = SLURM_BIN + ":" + os.environ.get("PATH", "")
        os.environ.setdefault("SLURM_CONF", "/cm/shared/apps/slurm/etc/slurm/slurm.conf")
    if not a.watch:
        draw(a)
        return
    try:
        while True:
            with con.capture() as cap:   # 先畫好再一次換掉，畫面不會閃
                draw(a)
            con.clear()
            con.print(cap.get(), end="")
            con.print(Text(f"每 {a.watch} 秒更新；Ctrl-C 離開", "dim"))
            time.sleep(a.watch)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)   # `| head` 時安靜結束
    main()
