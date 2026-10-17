"""x7_pcie_flag.py — X7：每一筆還原期間，同一顆 PCIe switch 上其他卡的 PCIe 流量（nvidia-smi dmon -s t），標出「鄰居有流量」的列。

為什麼：這台 3090 的 GPU 0–3、4–6 各在一顆 PLX PEX8796 後面，共用一條 Gen3 x16 上行；gpu_guard 只看自己那張卡，
看不到鄰居卡的 H2D／D2H 搶上行（CLAUDE.md §3 host_contention 的說明）。§2.2「污染」那條：結論只用鄰居流量可忽略的量測。

  python code/p3x/x7_pcie_flag.py <run_dir> [--thr-mbps 100]
→ 在 run_dir 寫 restores_pcie.csv（原欄位＋gpu、sibling_max_mbps、sibling_busy）。
時間對齊：一筆還原的區間＝[ts − ceil(ttft) − 1 s, ts + 1 s]（ts 是寫入那列的時間，秒；dmon 每 1 s 一筆、HH:MM:SS）。
"""
from __future__ import annotations

import argparse
import math
import os
from datetime import datetime, timedelta

import pandas as pd

SWITCH = {0: (0, 1, 2, 3), 1: (0, 1, 2, 3), 2: (0, 1, 2, 3), 3: (0, 1, 2, 3), 4: (4, 5, 6), 5: (4, 5, 6), 6: (4, 5, 6)}


def run_gpu(rd):
    for ln in open(os.path.join(rd, "context.txt")):
        if ln.startswith("short="):
            return int(ln.split("cuda_visible='")[1].split("'")[0])
    raise SystemExit("context.txt 沒有 cuda_visible")


def load_dmon(rd, day):
    rows = []
    for ln in open(os.path.join(rd, "dmon_pcie.txt")):
        if ln.startswith("#") or not ln.strip():
            continue
        p = ln.split()
        if len(p) < 4 or not p[1].isdigit():
            continue
        try:
            t = datetime.combine(day, datetime.strptime(p[0], "%H:%M:%S").time())
            rows.append((t, int(p[1]), int(p[2]) if p[2].isdigit() else 0, int(p[3]) if p[3].isdigit() else 0))
        except ValueError:
            continue
    return pd.DataFrame(rows, columns=["t", "gpu", "rx", "tx"])


def flag(rd, thr=100.0):
    g = run_gpu(rd)
    r = pd.read_csv(os.path.join(rd, "restores.csv"))
    ts = pd.to_datetime(r.ts).dt.tz_localize(None)
    dm = load_dmon(rd, ts.iloc[0].date())
    sib = dm[dm.gpu.isin([x for x in SWITCH[g] if x != g])].copy()
    sib["mbps"] = sib.rx + sib.tx
    mx = []
    for t1, dur in zip(ts, r.ttft):
        lo, hi = t1 - timedelta(seconds=math.ceil(dur) + 1), t1 + timedelta(seconds=1)
        w = sib[(sib.t >= lo) & (sib.t <= hi)]
        mx.append(int(w.mbps.max()) if len(w) else -1)       # -1＝這段時間沒有 dmon 取樣（不當成乾淨）
    r["gpu"] = g
    r["sibling_max_mbps"] = mx
    r["sibling_busy"] = [(m < 0) or (m >= thr) for m in mx]
    r.to_csv(os.path.join(rd, "restores_pcie.csv"), index=False)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--thr-mbps", type=float, default=100.0)
    a = ap.parse_args()
    x = flag(a.run_dir, a.thr_mbps)
    print(x.groupby(["tier", "gpu_state"]).sibling_busy.agg(["sum", "size"]).to_string())
