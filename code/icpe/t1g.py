"""t1g.py — ICPE T1G（docs/h200/icpe/T1G_gpu_rerun.md）：CPU 層換成 H200 常數之後，重量 GPU 實測的 run。

給 runner（code/r_h200_t1g.sh）用的兩個排程小工具（量時間不和我們自己的 job 同時跑；memory h200-no-self-cotenancy-for-timing）：
  wait-clear --job J [--node N] [--max-min M]   等到節點上沒有我們自己的別的 job 在跑（每 60 s 查一次 squeue）；逾時 exit 2
  overlap --job J --start T0 --end T1 --step S --short SH --log CSV
                                                這一步的時間窗 [T0, T1] 裡，節點上有沒有我們自己的別的 job（sacct）；
                                                有 → 寫進 CSV、exit 1（照重疊重跑）；沒有 → 也寫一列、exit 0；查不到 → exit 2（規則 7）
分析（analyze-*）在資料收齊之後、打開任何輸出之前另外寫（登記 §6）。
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta

SLURM_BIN = "/cm/local/apps/slurm/current/bin"


def _now():
    return datetime.now().astimezone()


def _run(cmd):
    env = dict(os.environ, PATH=SLURM_BIN + ":" + os.environ.get("PATH", ""))
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} 失敗（exit {p.returncode}）：{p.stderr.strip()}")
    return p.stdout


def _node(a):
    return a.node or os.environ.get("SLURMD_NODENAME") or socket.gethostname().split(".")[0]


def cmd_wait_clear(a):
    node, t_end = _node(a), time.time() + a.max_min * 60
    n = 0
    while True:
        try:
            out = _run(["squeue", "-h", "-w", node, "-u", os.environ["USER"], "-t", "R", "-o", "%i %j"])
        except Exception as e:                      # 查不到 ≠ 沒有（規則 7）
            print(f"[t1g] squeue 查不到：{e} → 停", file=sys.stderr)
            return 2
        others = [ln for ln in out.split("\n") if ln.strip() and ln.split()[0] != str(a.job)]
        if not others:
            print(f"[t1g] {_now().isoformat(timespec='seconds')} 節點 {node} 上沒有我們自己的別的 job（等了 {n} 次）", flush=True)
            return 0
        if time.time() > t_end:
            print(f"[t1g] 等了 {a.max_min} 分鐘仍有我們自己的 job：{others} → 停", file=sys.stderr)
            return 2
        if n % 10 == 0:
            print(f"[t1g] {_now().isoformat(timespec='seconds')} 等我們自己的 job 結束：{others}", flush=True)
        n += 1
        time.sleep(60)


def _parse(s):
    if s in ("", "Unknown", "None"):
        return None
    return datetime.fromisoformat(s).astimezone()


def cmd_overlap(a):
    node = _node(a)
    t0, t1 = datetime.fromisoformat(a.start).astimezone(), datetime.fromisoformat(a.end).astimezone()
    since = (t0 - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%S")
    try:
        out = _run(["sacct", "-u", os.environ["USER"], "-X", "-P", "-n", "-S", since,
                    "-o", "JobID,JobName,NodeList,Start,End,State"])
    except Exception as e:
        print(f"[t1g] sacct 查不到：{e} → 停", file=sys.stderr)
        return 2
    hits = []
    for ln in out.split("\n"):
        f = ln.split("|")
        if len(f) < 6 or f[0] == str(a.job) or node not in f[2]:
            continue
        s, e = _parse(f[3]), _parse(f[4])
        if s is None:
            continue
        e = e or _now()
        if s < t1 and e > t0:
            hits.append(f"{f[0]}:{f[1]}:{f[3]}..{f[4]}")
    runs = sorted(glob.glob(os.path.join(os.environ.get("TIARA_RUNS", "."), f"*-{a.short}")))
    rid = os.path.basename(runs[-1]) if runs else "NOT_FOUND"
    new = not os.path.exists(a.log)
    os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
    with open(a.log, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["run_id", "ts", "job", "node", "step", "t_start", "t_end", "overlap", "overlapping_jobs"])
        w.writerow([rid, _now().isoformat(timespec="seconds"), a.job, node, a.step, a.start, a.end, int(bool(hits)), ";".join(hits)])
    print(f"[t1g] {a.step} run {rid} [{a.start} → {a.end}] 重疊：{hits or '無'}", flush=True)
    return 1 if hits else 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("wait-clear"); s.add_argument("--job", required=True); s.add_argument("--node"); s.add_argument("--max-min", type=float, default=360)
    s = sp.add_parser("overlap"); s.add_argument("--job", required=True); s.add_argument("--node")
    s.add_argument("--start", required=True); s.add_argument("--end", required=True); s.add_argument("--step", required=True)
    s.add_argument("--short", required=True); s.add_argument("--log", required=True)
    a = ap.parse_args()
    sys.exit({"wait-clear": cmd_wait_clear, "overlap": cmd_overlap}[a.cmd](a))


if __name__ == "__main__":
    main()
