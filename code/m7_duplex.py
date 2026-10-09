"""m7_duplex.py — C0 補充：H2D 與 D2H 同時進行時，H2D 掉多少（CPU 層的讀寫干擾係數）。"""
import csv, os, sys, threading, time
from datetime import datetime
import torch
MiB = 1 << 20; GiB = 1 << 30
out = sys.argv[1]
n = 64 * MiB
g1 = torch.empty(n * 16, dtype=torch.uint8, device="cuda"); g2 = torch.empty_like(g1)
h1 = torch.empty(n * 16, dtype=torch.uint8, pin_memory=True); h2 = torch.empty_like(h1)
s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
def h2d():
    with torch.cuda.stream(s1):
        for _ in range(4): g1.copy_(h1, non_blocking=True)
    s1.synchronize()
def d2h():
    with torch.cuda.stream(s2):
        for _ in range(6): h2.copy_(g2, non_blocking=True)
    s2.synchronize()
new = not os.path.exists(out); f = open(out, "a", newline=""); w = csv.writer(f)
if new: w.writerow(["run_id", "ts", "case", "rep", "h2d_GiBps"])
h2d()
for rep in range(5):
    for case in ("alone", "with_d2h"):
        th = None
        if case == "with_d2h":
            th = threading.Thread(target=d2h); th.start(); time.sleep(0.005)
        t = time.perf_counter(); h2d(); dt = time.perf_counter() - t
        if th: th.join()
        v = 4 * 16 * n / dt / GiB
        w.writerow([os.environ.get("RUN_ID"), datetime.now().astimezone().isoformat(timespec="seconds"), case, rep, round(v, 3)])
        print(case, rep, round(v, 2), flush=True)
f.close()
