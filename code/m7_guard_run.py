"""m7_guard_run.py — 用 GpuWatcher 包住任一個子指令（第 1 輪破解計劃的校準用；m7_calib.py 本身沒有包）。

用法：python code/m7_guard_run.py <指令...>
輸出：$TIARA_RUNS/$RUN_ID/gpu_guard.json；子指令的 exit code 原樣傳回；污染時另外印 CONTAMINATED 並以 3 結束。
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from gpu_guard import GpuWatcher  # noqa: E402

rd = os.path.join(os.environ.get("TIARA_RUNS", "/mlsteam/data/tiara/runs"), os.environ.get("RUN_ID", "no-run-id"))
os.makedirs(rd, exist_ok=True)
with GpuWatcher(gpu=0, own_root=os.getpid(), out_path=os.path.join(rd, "gpu_guard.json")) as w:
    rc = subprocess.run(sys.argv[1:]).returncode
if w.contaminated:
    print("CONTAMINATED", flush=True)
    open(os.path.join(rd, "CONTAMINATED"), "w").close()
    sys.exit(3)
sys.exit(rc)
