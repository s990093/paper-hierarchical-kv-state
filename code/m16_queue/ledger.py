#!/usr/bin/env python3
"""第三階段（m16）用第二階段那本共用顯存帳（jobs/vram_reserved.json，fcntl 鎖）。格式與規則照 code/m10_queue_mi300x/pool.py：
{name: {"gpu": g, "need": GB, "pid": pid, "ts": t}}；行程不在（或殭屍）的項目自動拿掉；同一張卡 已登記＋need＋8 GB ≤ 總量。

  python ledger.py reserve <name> <gpu> <need_gb> <pid>   → exit 0 登記成功；exit 3 放不下
  python ledger.py release <name>
  python ledger.py card_clear <gpu>                       → exit 0：帳上沒有這張卡的登記、而且 amd-smi 這張卡已用 < 1.5 GB（XGPU 條件的 (b)）
  python ledger.py show
不 import torch（不開 /dev/kfd）；總量與空餘用 amd-smi 查，查不到就丟例外（CLAUDE.md 規則 7：查不到不等於沒有）。
"""
import fcntl
import json
import os
import subprocess
import sys
import time

S2 = os.environ.get("S2_JOBS", "/tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs")
LEDGER = os.path.join(S2, "vram_reserved.json")
MARGIN = 8


def alive(pid: int) -> bool:
    try:
        st = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return st not in ("Z", "X")


def mem(g: int):
    out = subprocess.run(["amd-smi", "metric", "--mem-usage", "--json"], capture_output=True, text=True, timeout=60)
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(f"amd-smi 查不到顯存（rc={out.returncode}）：{out.stderr[:300]}")
    d = json.loads(out.stdout)
    d = d["gpu_data"] if isinstance(d, dict) else d
    d = next(x for x in d if int(x.get("gpu", -1)) == g)
    return d["mem_usage"]["free_vram"]["value"] / 1024, d["mem_usage"]["total_vram"]["value"] / 1024


def _ledger(fn):
    if not os.path.isdir(S2):
        raise RuntimeError(f"第二階段的 jobs 目錄不在：{S2}（容器重啟後還沒還原？）")
    with open(LEDGER + ".lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            d = json.load(open(LEDGER))
        except (FileNotFoundError, ValueError):
            d = {}
        d = {k: v for k, v in d.items() if alive(v["pid"])}
        out = fn(d)
        json.dump(d, open(LEDGER + ".tmp", "w"))
        os.replace(LEDGER + ".tmp", LEDGER)
        return out


def main():
    cmd = sys.argv[1]
    if cmd == "reserve":
        name, g, need, pid = sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
        free, total = mem(g)

        def f(d):
            used = sum(v["need"] for v in d.values() if v["gpu"] == g)
            if used + need + MARGIN > total or free < need + MARGIN:
                return False
            d[name] = {"gpu": g, "need": need, "pid": pid, "ts": time.time(), "owner": "stage3"}
            return True
        ok = _ledger(f)
        print(json.dumps(dict(ok=ok, gpu=g, need=need, free=round(free, 1), total=round(total, 1))))
        sys.exit(0 if ok else 3)
    elif cmd == "card_clear":
        g = int(sys.argv[2])
        fake = os.environ.get("S3_FAKE_USED_GB")              # 只給 CPU 測試用
        if fake is not None:
            used = float(fake)
        else:
            free, total = mem(g)
            used = total - free
        regs = _ledger(lambda d: [k for k, v in d.items() if v["gpu"] == g])
        ok = (not regs) and used < 1.5
        print(json.dumps(dict(ok=ok, gpu=g, used_gb=round(used, 2), registered=regs)))
        sys.exit(0 if ok else 4)
    elif cmd == "release":
        _ledger(lambda d: d.pop(sys.argv[2], None))
    elif cmd == "show":
        print(json.dumps(_ledger(lambda d: dict(d)), indent=1))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
