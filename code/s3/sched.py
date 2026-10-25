#!/usr/bin/env python3
"""sched.py — stage3 的 job 和 ICPE 量時間的 job 之間的讓位規則（使用者 2026-10-25「排滿：stage3 品質先填空檔」，memory stage3-gpu-scheduling）。

規則：stage3 的分片送出時，下一個要開跑的 ICPE 量時間 job 會加上 afterany:<分片>（送出端做），所以分片先跑時 ICPE 會等它。
但如果 ICPE 的那個 job「除了在等 stage3 之外什麼都不缺」，新開跑的分片就不該再佔用那個空檔（不然 ICPE 會被一直往後推）：
分片開跑時呼叫 `sched.py yield-target`，印出應該讓位的 ICPE job id（沒有就什麼都不印）；runner 看到就把自己重送一份
（--dependency=afterany:<那個 ICPE job>）然後 exit 0。

  yield-target            印出「只在等 s3-* job」的 ICPE 量時間 job（沒 hold 的）
  next-icpe               印出目前所有沒 hold、還在排隊的 ICPE 量時間 job（送出新分片時要加 afterany 的對象）
  add-dep --to J --job K  把 afterany:K 加到 J 原本的依賴後面（保留原本的）
  --self-test             解析依賴字串
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys

ICPE_TIMING = re.compile(r"^icpe-(p01s1|p01ab|p01brr|p02|p2|p11t)$")
S3_TIMING = re.compile(r"^s3-T-")                       # stage-3 量時間的 job 一律用這個前綴命名
SLURM = "/cm/local/apps/slurm/current/bin"


def sh(args: list) -> str:
    return subprocess.run([f"{SLURM}/{args[0]}"] + args[1:], capture_output=True, text=True, check=True).stdout


def my_jobs() -> dict:
    """job id → (name, state, reason)。"""
    out = {}
    for line in sh(["squeue", "-h", "--me", "-o", "%i|%j|%T|%r"]).splitlines():
        i, n, st, rs = line.split("|", 3)
        out[i] = (n, st, rs)
    return out


def parse_dep(dep: str) -> list:
    """'afterany:1(unfulfilled),afterany:2(unfulfilled)' 或 'afterany:1:2' → [(kind, id, state)]。"""
    out = []
    if not dep or dep == "(null)":
        return out
    for part in dep.split(","):
        m = re.match(r"(\w+):(.+)", part)
        if not m:
            continue
        kind, rest = m.groups()
        for tok in rest.split(":"):
            mm = re.match(r"(\d+)(?:_\*)?(?:\((\w+)\))?", tok)
            if mm:
                out.append((kind, mm.group(1), mm.group(2) or "unfulfilled"))
    return out


def dep_of(job: str) -> str:
    m = re.search(r"Dependency=(\S+)", sh(["scontrol", "show", "job", job]))
    return m.group(1) if m else ""


def next_icpe(jobs: dict) -> list:
    return [i for i, (n, st, rs) in jobs.items() if ICPE_TIMING.match(n) and st == "PENDING" and "Held" not in rs]


HELD_GRACE_MIN = 60     # ICPE session 10/25 的要求：前一個 ICPE 量時間的 job 結束後這段時間內，hold 中的下一個也算「準備好了」


def last_icpe_timing_end() -> float | None:
    """最近一個已結束的 ICPE 量時間 job 的結束時間（epoch 秒）；查不到 → None。"""
    import time as _t
    out = sh(["sacct", "--me", "-X", "-n", "-P", "-S", "now-2days", "-o", "JobName,State,End"])
    ends = []
    for line in out.splitlines():
        n, st, end = (line.split("|") + ["", "", ""])[:3]
        if ICPE_TIMING.match(n) and end and end not in ("Unknown", "None") and not st.startswith(("PENDING", "RUNNING")):
            try:
                ends.append(_t.mktime(_t.strptime(end, "%Y-%m-%dT%H:%M:%S")))
            except ValueError:
                pass
    return max(ends) if ends else None


def held_icpe(jobs: dict) -> list:
    return [i for i, (n, st, rs) in jobs.items() if ICPE_TIMING.match(n) and st == "PENDING" and "Held" in rs]


def yield_target(jobs: dict) -> list:
    out = []
    # hold 中的 ICPE 量時間 job：沒有其他沒 hold 的 ICPE 量時間 job 在排隊或在跑，而且上一個剛結束（≤ HELD_GRACE_MIN 分鐘）
    # → ICPE session 正在看上一個的結果、馬上要放行它（10/25 的要求）→ 也讓位
    active = [i for i, (n, st, rs) in jobs.items() if ICPE_TIMING.match(n) and ("Held" not in rs)]
    held = held_icpe(jobs)
    if held and not active:
        import time as _t
        last = last_icpe_timing_end()
        if last is not None and _t.time() - last <= HELD_GRACE_MIN * 60:
            out.append(min(held, key=int))
    for i in next_icpe(jobs):
        unf = [d for _, d, s in parse_dep(dep_of(i)) if s == "unfulfilled"]
        if unf and all(jobs.get(d, ("",))[0].startswith("s3-") for d in unf):
            out.append(i)
        elif not unf:                     # 沒有在等任何人（等資源）→ 也讓位：我們一開跑它就會被我們擋住
            out.append(i)
    return out


def s3_timing_target(jobs: dict, me: str) -> list:
    """品質（不量時間）的 stage-3 job 讓位給「只剩在等 s3-* 的 job（或什麼都不等）」的 stage-3 量時間 job（和 ICPE 同一個規則）。"""
    out = []
    for i, (n, st, rs) in jobs.items():
        if i == me or not S3_TIMING.match(n) or st != "PENDING" or "Held" in rs:
            continue
        unf = [d for _, d, s in parse_dep(dep_of(i)) if s == "unfulfilled"]
        # 只有「等的全是 stage-3 的品質 job」（或什麼都不等）才算準備好了；在等鏈上前一個 s3-T-*、或在等 ICPE 的 → 還沒輪到它
        # （10/25 21:5x 的 bug：鏈上 22272–22281 被當成準備好，所有品質 job 都讓位到整條鏈之後）
        def quality(d):
            n = jobs.get(d, ("",))[0]
            return n.startswith("s3-") and not S3_TIMING.match(n)
        if not unf or all(quality(d) for d in unf):
            out.append(i)
    return out


def running_mine_on_node(jobs: dict, me: str, node: str) -> list:
    out = sh(["squeue", "-h", "--me", "-w", node, "-t", "R", "-o", "%i"]).split()
    return [i for i in out if i != me]


def add_dep(to: str, job: str):
    deps = [f"{k}:{d}" for k, d, s in parse_dep(dep_of(to)) if s == "unfulfilled"]
    deps.append(f"afterany:{job}")
    sh(["scontrol", "update", f"JobId={to}", f"Dependency={','.join(deps)}"])
    print(dep_of(to))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("yield-target")
    s = sub.add_parser("s3-timing-target")
    s.add_argument("--me", required=True)
    s = sub.add_parser("running-mine")
    s.add_argument("--me", required=True)
    s.add_argument("--node", required=True)
    sub.add_parser("next-icpe")
    s = sub.add_parser("add-dep")
    s.add_argument("--to", required=True)
    s.add_argument("--job", required=True)
    a = ap.parse_args()
    if a.self_test:
        ok = parse_dep("afterany:21907(unfulfilled),afterany:22241(unfulfilled)") == [("afterany", "21907", "unfulfilled"),
                                                                                       ("afterany", "22241", "unfulfilled")]
        ok &= parse_dep("afterany:1:2") == [("afterany", "1", "unfulfilled"), ("afterany", "2", "unfulfilled")]
        ok &= parse_dep("(null)") == []
        ok &= parse_dep("afterok:5(failed)") == [("afterok", "5", "failed")]
        print("self-test", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    if a.cmd == "yield-target":
        print(" ".join(yield_target(my_jobs())))
    elif a.cmd == "s3-timing-target":
        print(" ".join(s3_timing_target(my_jobs(), a.me)))
    elif a.cmd == "running-mine":
        print(" ".join(running_mine_on_node(my_jobs(), a.me, a.node)))
    elif a.cmd == "next-icpe":
        print(" ".join(next_icpe(my_jobs())))
    elif a.cmd == "add-dep":
        # 目標已經「準備好」（只剩在等 s3-* 或什麼都不等）→ 不加：新的 s3 job 開跑時反正會讓位給它，加了只會讓它等到
        # 我們的 job 拿到卡、讓位之後才能排程（10/26 00:3x：21655 p02 因此多等 22356／22357）
        if a.to in yield_target(my_jobs()):
            print(f"[sched] {a.to} 已經準備好（只剩在等 s3 的 job 或什麼都不等）→ 不加 afterany:{a.job}（新 job 會讓位）")
        else:
            add_dep(a.to, a.job)


if __name__ == "__main__":
    main()
