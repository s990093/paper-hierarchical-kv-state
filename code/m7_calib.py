"""m7_calib.py — 第一階段的校準（phase1/05 §3 C0、C1、C2、C7 自檢）

  python code/m7_calib.py c0 --out results/m7_write_policy_mi300x/calib_c0.csv
  python code/m7_calib.py c1 --out ... --max-chunks 80 --reps 3
  python code/m7_calib.py c2 --out ... --dev local=/var/tmp/m7io nfs=/mlsteam/data/tiara/runs/_m7io
  python code/m7_calib.py c7 --out ... --params results/m7_write_policy_mi300x/tier_params.json

每列都有 run_id、ts。run_id 取自環境變數 RUN_ID（runsh 設定的目錄名），沒有就用時間。
"""
from __future__ import annotations

import argparse
import csv
import json
import mmap
import os
import subprocess
import sys
import time
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MiB = 1 << 20
GiB = 1 << 30


def run_id():
    return os.environ.get("RUN_ID") or os.path.basename(os.environ.get("RUN_DIR", "")) or \
        datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc"


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


class Out:
    def __init__(self, path, fields):
        new = not os.path.exists(path)
        self.f = open(path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=["run_id", "ts"] + fields, extrasaction="ignore")
        if new:
            self.w.writeheader()
        self.rid = run_id()

    def row(self, **kw):
        kw.update(run_id=self.rid, ts=ts())
        self.w.writerow(kw)
        self.f.flush()
        print(json.dumps(kw), flush=True)


# ---------------------------------------------------------------- C0
def c0(a):
    o = Out(a.out, ["item", "kind", "size_mib", "rep", "value", "unit", "note"])
    # PCIe 與分割資訊
    for cmd in (["amd-smi", "static", "--bus"], ["rocm-smi", "--showbus", "--showpcie" ]):
        try:
            s = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
        except Exception as e:  # noqa
            s = f"ERR {e}"
        o.row(item="cmd:" + " ".join(cmd), kind="info", value=s.replace("\n", " | ")[:2000])
    dev = torch.device("cuda:0")
    for size_mib in (64, 1024):
        n = size_mib * MiB
        g = torch.empty(n, dtype=torch.uint8, device=dev)
        for pinned in (True, False):
            h = torch.empty(n, dtype=torch.uint8, pin_memory=pinned)
            h.fill_(1)
            for rep in range(a.reps):
                for direction in ("H2D", "D2H"):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    if direction == "H2D":
                        g.copy_(h, non_blocking=pinned)
                    else:
                        h.copy_(g, non_blocking=pinned)
                    torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                    o.row(item=f"{direction}_{'pinned' if pinned else 'pageable'}", kind="bw",
                          size_mib=size_mib, rep=rep, value=round(n / dt / GiB, 3), unit="GiB/s")
    # 非連續目的地（真實還原用的路徑：主機連續 64 MiB → GPU KV 緩衝的 strided 視圖）
    from m7_model import CHUNK
    L, H, D = 32, 8, 128
    kv = torch.empty((L, 2, H, 4096, D), dtype=torch.bfloat16, device=dev)
    h = torch.empty((L, 2, H, CHUNK, D), dtype=torch.bfloat16, pin_memory=True)
    for rep in range(a.reps * 3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        kv[:, :, :, CHUNK:2 * CHUNK].copy_(h, non_blocking=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        o.row(item="H2D_pinned_chunk_into_kv_view", kind="bw", size_mib=64, rep=rep,
              value=round(64 * MiB / dt / GiB, 3), unit="GiB/s", note=f"{dt*1e3:.3f}ms")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        h.copy_(kv[:, :, :, CHUNK:2 * CHUNK], non_blocking=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        o.row(item="D2H_pinned_chunk_from_kv_view", kind="bw", size_mib=64, rep=rep,
              value=round(64 * MiB / dt / GiB, 3), unit="GiB/s", note=f"{dt*1e3:.3f}ms")


# ---------------------------------------------------------------- C1
def c1(a):
    from m7_model import CHUNK, KVModel
    o = Out(a.out, ["item", "chunk_idx", "pos_end", "rep", "ms", "gpu_state"])
    km = KVModel(max_len=a.max_chunks * CHUNK + 512)
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(1000, 120000, (a.max_chunks * CHUNK + 512,), generator=g).to(km.device)
    # 暖機
    for i in range(4):
        km.forward_span(ids[i * CHUNK:(i + 1) * CHUNK], i * CHUNK)
    torch.cuda.synchronize()
    for rep in range(a.reps):
        for i in range(a.max_chunks):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            km.forward_span(ids[i * CHUNK:(i + 1) * CHUNK], i * CHUNK)
            torch.cuda.synchronize()
            o.row(item="f_chunk", chunk_idx=i, pos_end=(i + 1) * CHUNK, rep=rep,
                  ms=round((time.perf_counter() - t0) * 1e3, 3), gpu_state=a.gpu_state)
        # 新一輪 256 token 在不同長度上的成本
        for Lc in (8, 16, 32, 64):
            if Lc > a.max_chunks:
                continue
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            km.forward_span(ids[Lc * CHUNK:Lc * CHUNK + 256], Lc * CHUNK, want_logits=True)
            torch.cuda.synchronize()
            o.row(item="t_new256", chunk_idx=Lc, pos_end=Lc * CHUNK + 256, rep=rep,
                  ms=round((time.perf_counter() - t0) * 1e3, 3), gpu_state=a.gpu_state)


# ---------------------------------------------------------------- C2（真實裝置）
def _aligned(n):
    return mmap.mmap(-1, n)


def _proc_io():
    d = {}
    for line in open("/proc/self/io"):
        k, v = line.split(":")
        d[k.strip()] = int(v)
    return d


def _diskstats():
    tot = {}
    for line in open("/proc/diskstats"):
        p = line.split()
        if p[2] in ("sda",):
            tot[p[2]] = (int(p[5]), int(p[9]))  # sectors read, sectors written
    return tot


def c2(a):
    o = Out(a.out, ["device", "op", "size_mib", "n", "rep", "secs", "MiBps", "read_bytes_delta",
                    "sda_read_mib_delta", "sda_write_mib_delta", "note"])
    devs = dict(x.split("=", 1) for x in a.dev)
    for name, root in devs.items():
        os.makedirs(root, exist_ok=True)
        for rep in range(a.reps):
            # 1) 不同大小的單次讀寫 → 固定開銷迴歸
            for size_mib in (1, 4, 16, 64):
                n = size_mib * MiB
                buf = _aligned(n)
                buf.write(os.urandom(4096) * (n // 4096))
                path = f"{root}/sz{size_mib}_{rep}.bin"
                for k in range(a.n_small):
                    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
                    t0 = time.perf_counter(); os.write(fd, buf); os.fsync(fd)
                    dt = time.perf_counter() - t0; os.close(fd)
                    o.row(device=name, op="write1", size_mib=size_mib, n=1, rep=rep, secs=round(dt, 6),
                          MiBps=round(size_mib / dt, 1))
                    io0 = _proc_io()
                    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
                    t0 = time.perf_counter(); os.readv(fd, [buf]); dt = time.perf_counter() - t0
                    os.close(fd)
                    io1 = _proc_io()
                    o.row(device=name, op="read1", size_mib=size_mib, n=1, rep=rep, secs=round(dt, 6),
                          MiBps=round(size_mib / dt, 1), read_bytes_delta=io1["read_bytes"] - io0["read_bytes"])
                os.remove(path)
            # 2) 持續寫入：nchunks × 64 MiB（KV chunk 大小），每個 chunk 一個檔
            n = 64 * MiB
            buf = _aligned(n)
            buf.write(os.urandom(4096) * (n // 4096))
            ds0 = _diskstats(); t0 = time.perf_counter()
            for k in range(a.n_sustained):
                fd = os.open(f"{root}/c{k}.bin", os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
                os.write(fd, buf); os.close(fd)
            os.sync()
            dt = time.perf_counter() - t0; ds1 = _diskstats()
            d = {k: ((ds1[k][0] - ds0[k][0]) * 512 / MiB, (ds1[k][1] - ds0[k][1]) * 512 / MiB) for k in ds0}
            o.row(device=name, op="write_sustained", size_mib=64, n=a.n_sustained, rep=rep, secs=round(dt, 4),
                  MiBps=round(64 * a.n_sustained / dt, 1),
                  sda_read_mib_delta=round(d.get("sda", (0, 0))[0], 1),
                  sda_write_mib_delta=round(d.get("sda", (0, 0))[1], 1))
            io0 = _proc_io(); ds0 = _diskstats(); t0 = time.perf_counter()
            for k in range(a.n_sustained):
                fd = os.open(f"{root}/c{k}.bin", os.O_RDONLY | os.O_DIRECT)
                os.readv(fd, [buf]); os.close(fd)
            dt = time.perf_counter() - t0; io1 = _proc_io(); ds1 = _diskstats()
            d = {k: ((ds1[k][0] - ds0[k][0]) * 512 / MiB, (ds1[k][1] - ds0[k][1]) * 512 / MiB) for k in ds0}
            o.row(device=name, op="read_sustained", size_mib=64, n=a.n_sustained, rep=rep, secs=round(dt, 4),
                  MiBps=round(64 * a.n_sustained / dt, 1), read_bytes_delta=io1["read_bytes"] - io0["read_bytes"],
                  sda_read_mib_delta=round(d.get("sda", (0, 0))[0], 1),
                  sda_write_mib_delta=round(d.get("sda", (0, 0))[1], 1))
            for k in range(a.n_sustained):
                os.remove(f"{root}/c{k}.bin")


# ---------------------------------------------------------------- C7 自檢
def c7(a):
    from m7_model import CHUNK
    from m7_restore_harness import Tier, sleep_until  # noqa
    o = Out(a.out, ["set_GiBps", "c_ms", "n", "rep", "secs", "eff_GiBps", "err_pct"])
    dev = torch.device("cuda:0")
    L, H, D = 32, 8, 128
    kv = torch.empty((L, 2, H, 64 * CHUNK, D), dtype=torch.bfloat16, device=dev)
    hosts = [torch.empty((L, 2, H, CHUNK, D), dtype=torch.bfloat16, pin_memory=True) for _ in range(32)]
    s = torch.cuda.Stream()
    nbytes = 64 * MiB
    for X in a.set_gibps:
        for rep in range(a.reps):
            t = Tier("x", read_Bps=X * GiB, write_Bps=X * GiB, c_s=0.0)
            t0 = time.perf_counter()
            for i, h in enumerate(hosts):
                done = t.reserve_read(nbytes)
                with torch.cuda.stream(s):
                    kv[:, :, :, i * CHUNK:(i + 1) * CHUNK].copy_(h, non_blocking=True)
                    ev = torch.cuda.Event(); ev.record(s)
                sleep_until(done)
                ev.synchronize()
            dt = time.perf_counter() - t0
            eff = len(hosts) * nbytes / dt / GiB
            o.row(set_GiBps=X, c_ms=0, n=len(hosts), rep=rep, secs=round(dt, 5), eff_GiBps=round(eff, 3),
                  err_pct=round((eff - X) / X * 100, 2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    p.add_argument("--out", required=True)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--max-chunks", type=int, default=80)
    p.add_argument("--gpu-state", default="idle")
    p.add_argument("--dev", nargs="*", default=[])
    p.add_argument("--n-small", type=int, default=3)
    p.add_argument("--n-sustained", type=int, default=64)
    p.add_argument("--set-gibps", type=float, nargs="*", default=[0.25, 1, 5, 20, 40])
    a = p.parse_args()
    {"c0": c0, "c1": c1, "c2": c2, "c7": c7}[a.cmd](a)


if __name__ == "__main__":
    main()
