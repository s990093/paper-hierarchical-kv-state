#!/usr/bin/env python3
"""h200_queue.py — 我的 job 在 h200q 排第幾、前面是誰、大概還要等多久（2026-10-15）

    python3 code/h200_queue.py                 # 自己（$USER）的 job
    python3 code/h200_queue.py --job 20417     # 指定 job
    python3 code/h200_queue.py --no-history    # 不查 sacct（快一點；少了「照過去執行時間」那個模擬）
    watch -n 300 python3 code/h200_queue.py    # 每 5 分鐘更新

只讀 Slurm（squeue、sprio、scontrol、sacctmgr、sacct），不送、不改任何 job。用系統的 python3 就能跑（只用標準庫）。

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
from datetime import datetime, timedelta

PART = "h200q"
SLURM_BIN = "/cm/local/apps/slurm/current/bin"
# 不是「現在就在搶卡」的 pending 原因（hold、等前置 job、等開始時間）
NOT_COMPETING = ("Dependency", "DependencyNeverSatisfied", "JobHeldUser", "JobHeldAdmin", "BeginTime",
                 "PartitionDown", "PartitionInactive", "InvalidAccount", "InvalidQOS")
FAR = timedelta(days=365)


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
    out = sh("squeue", "-p", PART, "-h", "-r", "-o", "%i|%u|%a|%T|%Q|%r|%b|%l|%S|%V|%e|%j|%M")
    jobs = []
    for line in out.splitlines():
        f = line.split("|")
        if len(f) < 13:
            continue
        jid, user, acct, state, prio, reason, tres, lim, start, submit, end, name, used = f[:13]
        jobs.append(dict(id=jid, user=user, acct=acct.lower(), state=state, prio=int(prio) if prio.isdigit() else 0,
                         reason=reason, gpus=gpus_of(tres), limit=plimit(lim), start=ptime(start),
                         submit=ptime(submit), end=ptime(end), name=name))
    return jobs


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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--job", help="job id（預設：自己所有在 h200q 的 job）")
    ap.add_argument("--user", default=os.environ.get("USER", ""))
    ap.add_argument("--no-history", action="store_true", help="不查 sacct")
    a = ap.parse_args()
    if not shutil.which("squeue") and os.path.isdir(SLURM_BIN):
        os.environ["PATH"] = SLURM_BIN + ":" + os.environ.get("PATH", "")
        os.environ.setdefault("SLURM_CONF", "/cm/shared/apps/slurm/etc/slurm/slurm.conf")

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

    print(f"=== {PART} 佇列 @ {now:%Y-%m-%d %H:%M} ===")
    print(f"GPU：共 {total} 張，使用中 {used}，空 {total - used}｜正在跑 {len(running)} 個 job｜"
          f"排隊 {len(pend_all)} 個（現在在搶卡 {len(competing)}、hold 或等前置 job {len(waiting_dep)}）")

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

    if not mine:
        print(f"\n（{a.job or a.user} 在 {PART} 沒有 job）")
    for j in mine:
        who = "你的 job" if j["user"] == a.user else f"{j['user']} 的 job"
        print(f"\n▶ {who} {j['id']}（{j['name']}）：{j['state']}　原因 {j['reason']}　GPU {j['gpus']}　上限 {fmt_td(j['limit'])}")
        if j["state"] != "PENDING":
            print(f"  開始 {fmt_dt(j['start'])}，已跑 {fmt_td(now - j['start']) if j['start'] else 'N/A'}，最晚 {fmt_dt(j['end'])} 結束")
            continue
        print(f"  送出 {fmt_dt(j['submit'])}，已等 {fmt_td(now - j['submit']) if j['submit'] else 'N/A'}")
        pr = sh("sprio", "-h", "-j", j["id"].split("_")[0], "-o", "%Y|%A|%F|%J|%Q|%N").strip().split("|")
        if len(pr) >= 6:
            grow = f"；Age 每天約 +{age_per_day:.1f}（最多 +{w_age.group(1)}）" if age_per_day else ""
            print(f"  優先序 {pr[0]}＝FairShare {pr[2]}＋QOS {pr[4]}＋Age {pr[1]}＋JobSize {pr[3]}"
                  f"{'＋Nice ' + pr[5] if pr[5] not in ('0', '') else ''}{grow}")
        if j["reason"] in NOT_COMPETING:
            print(f"  這個 job 現在不在搶卡（{j['reason']}），放行或前置 job 完成後才會排")
            continue
        rank = next(i for i, c in enumerate(competing, 1) if c["id"] == j["id"])
        ahead = competing[:rank - 1]
        print(f"  ★ 排第 {rank} 位：前面 {len(ahead)} 個 job、共要 {sum(c['gpus'] for c in ahead)} 張卡")
        lim_ac = limits.get(j["acct"])
        run_ac = sum(r["gpus"] for r in running if r["acct"] == j["acct"])
        print(f"  你的 account（{j['acct']}）正在用 {run_ac} 張，上限 {lim_ac if lim_ac is not None else '查不到'}")
        print(f"  什麼時候開始：")
        print(f"    Slurm 預估　　　　　 {fmt_dt(j['start'])}{'（' + fmt_td(j['start'] - now) + ' 後）' if j['start'] else ''}")
        t = sim_lim.get(j["id"])
        print(f"    模擬・大家跑滿上限　 {fmt_dt(t)}{'（' + fmt_td(t - now) + ' 後）' if t else '（放不進去）'}")
        if sim_hist:
            t = sim_hist.get(j["id"])
            print(f"    模擬・照過去 30 天　 {fmt_dt(t)}{'（' + fmt_td(t - now) + ' 後）' if t else ''}　← 參考值")

    def hist_s(j, ran=timedelta(0)):
        m = typical(hist, j["user"], j["limit"], ran)
        return fmt_td(m) if m else "用上限"

    my_ids = {j["id"] for j in mine}
    print(f"\n── 在搶卡的 pending（依優先序；★＝你）──")
    print(f"{'':2}{'#':>3} {'job':<14}{'user':<12}{'account':<15}{'prio':>5} {'GPU':>3} {'上限':>10} {'已等':>10} {'過去同上限':>10}  {'模擬開始(上限/過去)':<20} 原因")
    for i, c in enumerate(competing, 1):
        mark = "★" if c["id"] in my_ids else ""
        wait = fmt_td(now - c["submit"]) if c["submit"] else "N/A"
        s1 = fmt_dt(sim_lim.get(c["id"]))
        s2 = fmt_dt(sim_hist.get(c["id"])) if sim_hist else "—"
        print(f"{mark:2}{i:>3} {c['id']:<14}{c['user']:<12}{c['acct']:<15}{c['prio']:>5} {c['gpus']:>3} "
              f"{fmt_td(c['limit']):>10} {wait:>10} {hist_s(c):>10}  {s1}／{s2:<12} {c['reason']}")

    print(f"\n── 正在跑（依結束時間）──")
    print(f"  {'job':<14}{'user':<12}{'account':<15}{'GPU':>3} {'開始':>12} {'已跑':>10} {'最晚結束':>12} {'照過去結束':>12}")
    for r in sorted(running, key=lim_end):
        ran = fmt_td(now - r["start"]) if r["start"] else "N/A"
        print(f"  {r['id']:<14}{r['user']:<12}{r['acct']:<15}{r['gpus']:>3} {fmt_dt(r['start']):>12} {ran:>10} "
              f"{fmt_dt(lim_end(r)):>12} {fmt_dt(hist_end(r)) if hist else '—':>12}")

    if waiting_dep:
        by = {}
        for w in waiting_dep:
            k = (w["user"], w["acct"], w["reason"])
            by[k] = by.get(k, 0) + 1
        print(f"\n── 現在不搶卡（hold／等前置 job；放行後會依優先序加入）──")
        for (u, ac, rs), n in sorted(by.items(), key=lambda x: -x[1]):
            print(f"  {u:<12}{ac:<15}{n:>3} 個  {rs}")
    print("\n⚠️ 模擬是推算：之後有優先序更高的 job 送進來就會排到前面；跑滿上限那個通常偏晚、照過去那個可能偏早。")


if __name__ == "__main__":
    main()
