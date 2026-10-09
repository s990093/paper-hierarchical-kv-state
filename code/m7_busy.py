"""m7_busy.py — 模擬「GPU 忙」（phase1/05 C5）：同一張卡上的背景 matmul，用 duty cycle 調強度。

  python code/m7_busy.py --duty 0.5 --period-ms 20 --n 8192
每個週期先連續做 matmul 約 duty×period，再睡 (1-duty)×period。小記憶體（3 個 n×n BF16）。
收到 SIGTERM 就結束。啟動完成時在 stdout 印 READY。
"""
import argparse
import signal
import sys
import time

import torch

p = argparse.ArgumentParser()
p.add_argument("--duty", type=float, required=True)
p.add_argument("--period-ms", type=float, default=20.0)
p.add_argument("--n", type=int, default=8192)
a = p.parse_args()
stop = False


def _h(*_):
    global stop
    stop = True


signal.signal(signal.SIGTERM, _h)
x = torch.randn(a.n, a.n, dtype=torch.bfloat16, device="cuda")
y = torch.randn(a.n, a.n, dtype=torch.bfloat16, device="cuda")
z = torch.empty_like(x)
for _ in range(5):
    torch.matmul(x, y, out=z)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    torch.matmul(x, y, out=z)
torch.cuda.synchronize()
mm = (time.perf_counter() - t0) / 20
print(f"READY matmul_ms={mm*1e3:.3f}", flush=True)
on = a.duty * a.period_ms / 1e3
off = (1 - a.duty) * a.period_ms / 1e3
while not stop:
    t = time.perf_counter()
    while time.perf_counter() - t < on and not stop:
        torch.matmul(x, y, out=z)
        torch.cuda.synchronize()
    if off > 0:
        time.sleep(off)
sys.exit(0)
