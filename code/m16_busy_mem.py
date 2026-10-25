"""m16_busy_mem.py — 「GPU 忙」的記憶體頻寬型版本（10/22 文獻查新 L7b 的效度風險：m7_busy.py 是純 matmul＝算力型，
真實服務的背景主要是 decode＝記憶體頻寬型，翻轉方向可能不同）。介面和輸出照 m7_busy.py，給 m10_speed.Busy 用
（呼叫端設 M10_BUSY_SCRIPT 指到這支）。

  python code/m16_busy_mem.py --duty 0.5 --period-ms 20 --gib 2
每個週期先連續做大陣列複製（y.copy_(x)：讀 gib＋寫 gib）約 duty×period，再睡 (1-duty)×period。佔 2×gib 的顯存。
收到 SIGTERM 就結束。啟動完成時在 stdout 印 READY；結束時印 DONE n=<複製次數> s=<秒>（和 m7_busy 同格式，Busy 照樣讀）。
"""
import argparse
import signal
import sys
import time

import torch

p = argparse.ArgumentParser()
p.add_argument("--duty", type=float, required=True)
p.add_argument("--period-ms", type=float, default=20.0)
p.add_argument("--gib", type=float, default=2.0)
a = p.parse_args()
stop = False


def _h(*_):
    global stop
    stop = True


signal.signal(signal.SIGTERM, _h)
n = int(a.gib * 2 ** 30 / 2)
x = torch.randn(n, dtype=torch.bfloat16, device="cuda")
y = torch.empty_like(x)
for _ in range(5):
    y.copy_(x)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    y.copy_(x)
torch.cuda.synchronize()
cp = (time.perf_counter() - t0) / 20
print(f"READY copy_ms={cp*1e3:.3f} GBps={2 * n * 2 / cp / 1e9:.1f}", flush=True)
on = a.duty * a.period_ms / 1e3
off = (1 - a.duty) * a.period_ms / 1e3
count, t_start = 0, time.perf_counter()
while not stop:
    t = time.perf_counter()
    while time.perf_counter() - t < on and not stop:
        y.copy_(x)
        torch.cuda.synchronize()
        count += 1
    if off > 0:
        time.sleep(off)
print(f"DONE n={count} s={time.perf_counter() - t_start:.3f}", flush=True)
sys.exit(0)
