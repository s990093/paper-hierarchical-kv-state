"""m7_guard_run.py — 用 GpuWatcher 包住任一個子指令（第 1 輪破解計劃的校準用；m7_calib.py 本身沒有包）。

用法：python code/m7_guard_run.py <指令...>
輸出：$TIARA_RUNS/$RUN_ID/gpu_guard.json；子指令的 exit code 原樣傳回；污染時另外印 CONTAMINATED 並以 3 結束。
"""
import contextlib
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from gpu_guard import GpuWatcher, smi_index  # noqa: E402

rd = os.path.join(os.environ.get("TIARA_RUNS", "/mlsteam/data/tiara/runs"), os.environ.get("RUN_ID", "no-run-id"))
os.makedirs(rd, exist_ok=True)
# 看得到幾張卡就監看幾張（2026-10-17：run3090.sh 的 TIARA_TWO_GPUS=1 會給兩張）。一張卡時和原本完全相同。
n_vis = len([x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if x.strip()]) or 1
with contextlib.ExitStack() as stack:
    ws = [stack.enter_context(GpuWatcher(gpu=smi_index(i), own_root=os.getpid(),   # cuda:i 在 smi 裡的編號（Slurm）
                                         out_path=os.path.join(rd, "gpu_guard.json" if i == 0 else f"gpu_guard_cuda{i}.json")))
          for i in range(n_vis)]
    rc = subprocess.run(sys.argv[1:]).returncode
if any(w.contaminated for w in ws):
    print("CONTAMINATED", flush=True)
    open(os.path.join(rd, "CONTAMINATED"), "w").close()
    sys.exit(3)
sys.exit(rc)
