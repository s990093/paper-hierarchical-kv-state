"""在 GpuWatcher 底下跑一個指令（子行程算自己人），每張卡一份 guard 報告。

10/20：70（vLLM TTFT）跑的時候兩張卡獨佔（WANT_BOTH＋兩把鎖），但 m10_vllm_ttft.py 沒開 gpu_guard，
乾淨與否沒有紀錄（CLAUDE.md §3：guard 是必用的）。這支包在外面補上，不改被包的程式。

用法：python code/m10_guarded_run.py --gpus 0 1 --out <dir> -- <指令…>
結束碼：指令本身失敗 → 指令的結束碼；指令成功但有一張卡被污染 → 3（並在 $TIARA_RUNS/$RUN_ID 留 CONTAMINATED）。
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_guard import GpuWatcher  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", nargs="+", type=int, required=True)
    ap.add_argument("--out", required=True, help="guard 報告的目錄（每張卡一個 gpu_guard_<g>.json）")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("沒有指令")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = os.environ.get("RUN_ID", "run")
    ws = [GpuWatcher(gpu=g, own_root=os.getpid(), out_path=str(out / f"gpu_guard_{tag}_gpu{g}.json")) for g in a.gpus]
    for w in ws:
        w.__enter__()
    try:
        rc = subprocess.call(cmd)
    finally:
        for w in ws:
            w.__exit__(None, None, None)
    rep = {str(w.gpu): w.verdict() for w in ws}
    print(f"[guarded] {json.dumps(rep)}", flush=True)
    if rc != 0:
        return rc
    if any(w.contaminated for w in ws):
        run = os.environ.get("TIARA_RUNS"), os.environ.get("RUN_ID")
        if all(run):
            Path(run[0], run[1], "CONTAMINATED").write_text(json.dumps(rep) + "\n")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
