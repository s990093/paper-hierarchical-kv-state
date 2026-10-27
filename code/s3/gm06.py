#!/usr/bin/env python3
"""gm06.py — 第三階段 G-M06＋M07（mechanisms/03 M06、M07）：借鄰卡的 PCIe 多路還原、NUMA 放置。4 卡 job、量時間、獨佔。
判準：docs/stage3_design_20261024/runs/G_M06_multipath.md（登記，不改）。速度判準照 05 §2.1（每點 6 次）。

  run --out-dir D    拓撲 → 1／2／4 卡同時 H2D → NVLink D2D → 16 GiB 經 1／2／4 條路徑送進 GPU 0 → NUMA 本地／遠端的 H2D
  report             判定
  --self-test        CPU：切片、路徑分配、管線的位元組帳
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import glob
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime

import numa_probe as NP                          # 只記錄：每份來源 pinned 實際落在哪個節點（numa_local／remote 有沒有真的綁到）
import guard as GD                               # 量時間的 run 的 GPU 監看（CLAUDE.md §3）；4 張卡都看

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "results", "s3_h200", "gm06")
GiB = 1 << 30
REPS = 6
CHUNK_MIB = 256
TOTAL_GIB = 16
NM = "NOT_MEASURED"


def run_id():
    return os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc"


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def noise(vals):
    v = [float(x) for x in vals]
    if len(v) < 6 or statistics.median(v) == 0:
        return NM, NM
    sr = statistics.stdev(v) / statistics.median(v)
    return round(sr, 6), round(max(0.05, 2 * sr), 6)


def split_bytes(total: int, k: int, chunk: int) -> list:
    """total 位元組分給 k 條路徑（以 chunk 為單位盡量平均）→ 每條路徑的 [(offset, size), ...]。"""
    n = (total + chunk - 1) // chunk
    out = [[] for _ in range(k)]
    for c in range(n):
        off = c * chunk
        out[c % k].append((off, min(chunk, total - off)))
    return out


# ============================================================================ 拓撲
def topo() -> dict:
    import torch

    def sh(c):
        try:
            return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as ex:  # noqa: BLE001
            return f"QUERY_FAILED {ex!r}"
    out = dict(topo_m=sh("nvidia-smi topo -m"), lscpu=sh("lscpu | grep -E 'NUMA|Socket|Model name'"),
               cpuset=sh("cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null || cat /sys/fs/cgroup/cpuset/cpuset.cpus 2>/dev/null"),
               mems=sh("cat /sys/fs/cgroup/cpuset.mems.effective 2>/dev/null || cat /sys/fs/cgroup/cpuset/cpuset.mems 2>/dev/null"),
               gpus=[])
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        bus = f"{getattr(p, 'pci_domain_id', 0):04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"     # Slurm 下 nvidia-smi 的編號≠CUDA 的
        node = sh(f"cat /sys/bus/pci/devices/{bus}/numa_node")
        out["gpus"].append(dict(index=i, name=p.name, bus=bus, numa_node=node,
                                peer={j: torch.cuda.can_device_access_peer(i, j) for j in range(torch.cuda.device_count()) if j != i}))
    return out


# ============================================================================ NUMA：在指定節點配 pinned 記憶體（set_mempolicy）
MPOL_BIND, MPOL_DEFAULT = 2, 0


def set_mempolicy(node: int | None):
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if node is None:
        r = libc.syscall(238, MPOL_DEFAULT, None, 0)
    else:
        mask = ctypes.c_ulong(1 << node)
        r = libc.syscall(238, MPOL_BIND, ctypes.byref(mask), 64)        # 238＝set_mempolicy（x86_64）
    if r != 0:
        raise OSError(ctypes.get_errno(), "set_mempolicy 失敗")


def pinned(nbytes: int, node: int | None = None):
    import torch
    if node is not None:
        set_mempolicy(node)
    try:
        t = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        t.view(-1)[::4096] = 1                                         # 每頁碰一下
    finally:
        if node is not None:
            set_mempolicy(None)
    return t


# ============================================================================ 量
def timed_parallel(fns: list) -> tuple:
    """每個 fn 在自己的執行緒裡跑（同時開始）；回傳 (總時間, 各自時間)。"""
    bar = threading.Barrier(len(fns) + 1)
    res = [None] * len(fns)

    def run(i, f):
        bar.wait()
        t0 = time.perf_counter()
        f()
        res[i] = time.perf_counter() - t0
    th = [threading.Thread(target=run, args=(i, f)) for i, f in enumerate(fns)]
    for t in th:
        t.start()
    bar.wait()
    t0 = time.perf_counter()
    for t in th:
        t.join()
    return time.perf_counter() - t0, res


def h2d_fn(dev: int, src, dst):
    import torch
    s = torch.cuda.Stream(device=dev)

    def f():
        with torch.cuda.stream(s):
            dst.copy_(src, non_blocking=True)
        s.synchronize()
    return f


def relay_fn(relay: int, target: int, src, dst, parts: list):
    """鄰卡 relay：每片先 H2D 到 relay 的暫存區（兩塊輪流），再 D2D（NVLink）寫進 target 的 dst[off:off+size]。relay＝target 時直接 H2D。"""
    import torch
    if relay == target:
        s = torch.cuda.Stream(device=target)

        def f():
            with torch.cuda.stream(s):
                for off, size in parts:
                    dst[off:off + size].copy_(src[off:off + size], non_blocking=True)
            s.synchronize()
        return f
    sa, sb = torch.cuda.Stream(device=relay), torch.cuda.Stream(device=relay)
    stage = [torch.empty(CHUNK_MIB << 20, dtype=torch.uint8, device=f"cuda:{relay}") for _ in range(2)]
    done = [torch.cuda.Event() for _ in range(2)]

    def f():
        for j, (off, size) in enumerate(parts):
            k = j % 2
            with torch.cuda.stream(sa):
                if j >= 2:
                    sa.wait_event(done[k])
                stage[k][:size].copy_(src[off:off + size], non_blocking=True)
                e = torch.cuda.Event()
                e.record(sa)
            with torch.cuda.stream(sb):
                sb.wait_event(e)
                dst[off:off + size].copy_(stage[k][:size], non_blocking=True)
                done[k].record(sb)
        sa.synchronize()
        sb.synchronize()
    return f


def cmd_run(a):
    import torch
    n = torch.cuda.device_count()
    rid = run_id()
    os.makedirs(a.out_dir, exist_ok=True)
    tp = topo()
    json.dump(tp, open(os.path.join(a.out_dir, f"topo_{rid}.json"), "w"), indent=1)
    print(tp["topo_m"], flush=True)
    rows = []

    def rec(test, k, times, nbytes, **kw):
        g = [nbytes / t / GiB for t in times]
        sr, ds = noise(times)
        rows.append(dict(run_id=rid, ts=ts(), test=test, k=k, n_rep=len(times), t_med_s=statistics.median(times),
                         gibps_med=statistics.median(g), sigma_rel=sr, delta_star=ds, reps_s=";".join(f"{t:.5f}" for t in times), **kw))
        print(json.dumps(rows[-1]), flush=True)
    out_csv = os.path.join(a.out_dir, f"gm06_{rid}.csv")
    gw = GD.Timing(out_csv, gpus=range(n))
    per = 4 * GiB
    srcs = [pinned(per) for _ in range(n)]
    dsts = [torch.empty(per, dtype=torch.uint8, device=f"cuda:{i}") for i in range(n)]
    numa = {"srcs": NP.probe(0, {f"src_gpu{i}": s for i, s in enumerate(srcs)})}
    smi = numa["srcs"]["gpus_visible"]

    def bn(p, name):
        """probe 裡一組緩衝的「節點:抽樣頁數」；查不到 → NOT_AVAILABLE＋原因。"""
        b = p["buffers"][name]
        if "status" in b:
            return b["status"]
        s = ";".join(f"{k}:{v}" for k, v in b["nodes"].items())
        return s + (f";unresolved:{sum(b['errors'].values())}" if b["errors"] else "") if s else f"{NP.NA}: {b['errors']}"
    # 1. 1／2／4 張卡同時 H2D（每張卡 4 GiB）
    for k in (1, 2, 4):
        if k > n:
            continue
        tot, each = [], []
        for rep in range(REPS + 1):
            t, e = timed_parallel([h2d_fn(i, srcs[i], dsts[i]) for i in range(k)])
            if rep:
                tot.append(t)
                each.append(e)
        rec("h2d_simul", k, tot, k * per, per_gpu_gibps=";".join(f"{per / statistics.median([x[i] for x in each]) / GiB:.2f}" for i in range(k)),
            src_numa=" | ".join(f"g{i}={bn(numa['srcs'], f'src_gpu{i}')}" for i in range(k)))
    # 2. NVLink D2D：GPU j → GPU 0（各自、同時）
    for j in range(1, n):
        times = []
        for rep in range(REPS + 1):
            torch.cuda.synchronize(j)
            t0 = time.perf_counter()
            dsts[0].copy_(dsts[j])
            torch.cuda.synchronize(0)
            if rep:
                times.append(time.perf_counter() - t0)
        rec("d2d_to0", 1, times, per, src_gpu=j)
    # 3. 16 GiB 經 1／2／4 條路徑送進 GPU 0
    total = TOTAL_GIB * GiB
    big = pinned(total)
    dst0 = torch.empty(total, dtype=torch.uint8, device="cuda:0")
    numa["big"] = NP.probe(0, {"big_16g": big}, smi=smi)
    for k in (1, 2, 4):
        if k > n:
            continue
        parts = split_bytes(total, k, CHUNK_MIB << 20)
        times = []
        for rep in range(REPS + 1):
            fns = [relay_fn(r, 0, big, dst0, parts[r]) for r in range(k)]
            t, _ = timed_parallel(fns)
            if rep:
                times.append(t)
        rec("multipath_16g", k, times, total, src_numa=bn(numa["big"], "big_16g"))
        if k == 1:
            ok = True
        else:
            ok = bool(torch.equal(dst0[:1 << 20].cpu(), big[:1 << 20]) and torch.equal(dst0[-(1 << 20):].cpu(), big[-(1 << 20):]))
        rows[-1]["data_ok"] = int(ok)
    # 4. NUMA：GPU 0 從本地／遠端節點的 pinned 記憶體 H2D（M07）
    node0 = tp["gpus"][0]["numa_node"]
    try:
        nodes = sorted(int(d[4:]) for d in os.listdir("/sys/devices/system/node") if d.startswith("node") and d[4:].isdigit())
    except OSError:
        nodes = []
    if node0 not in ("", "-1") and len(nodes) >= 2:
        for where, node in (("local", int(node0)), ("remote", [x for x in nodes if x != int(node0)][0])):
            try:
                src = pinned(per, node)
            except OSError as ex:
                rows.append(dict(run_id=rid, ts=ts(), test=f"numa_{where}", k=1, n_rep=0, t_med_s=NM, gibps_med=NM, sigma_rel=NM,
                                 delta_star=NM, reps_s="", note=f"set_mempolicy 失敗：{ex}（cgroup 不允許那個節點？）"))
                continue
            numa[f"numa_{where}"] = NP.probe(0, {f"numa_{where}": src}, smi=smi)      # 綁的節點有沒有真的生效
            times = []
            for rep in range(REPS + 1):
                t, _ = timed_parallel([h2d_fn(0, src, dsts[0])])
                if rep:
                    times.append(t)
            rec(f"numa_{where}", 1, times, per, numa_node=node, src_numa=bn(numa[f"numa_{where}"], f"numa_{where}"))
            del src
    else:
        rows.append(dict(run_id=rid, ts=ts(), test="numa", k=1, n_rep=0, t_med_s=NM, gibps_med=NM, sigma_rel=NM, delta_star=NM, reps_s="",
                         note=f"GPU 0 的 NUMA 節點 {node0!r}、節點數 {len(nodes)}：量不了"))
    dest = gw.finish()                                            # 污染 → run 目錄，不進 results
    for r in rows:
        r["own_gpu_intruders"] = gw.status()
    keys = sorted({k for r in rows for k in r})
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("[gm06] NUMA 存在", NP.save(dest, rid, **numa), flush=True)


def cmd_report(a):
    import numpy as np
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, "gm06_*.csv"))):
        rows += list(csv.DictReader(open(p)))
    rows, bad = GD.drop_contaminated(rows)
    mp = {int(r["k"]): [float(x) for x in r["reps_s"].split(";")] for r in rows if r["test"] == "multipath_16g" and r["reps_s"]}
    res = dict(run_id=run_id(), ts=ts(), excluded_contaminated_runs=bad)
    if 1 in mp and 2 in mp:
        rng = np.random.default_rng(0)
        a1, a2 = np.asarray(mp[1]), np.asarray(mp[2])
        sp = np.median(a1[rng.integers(0, a1.size, (10000, a1.size))], 1) / np.median(a2[rng.integers(0, a2.size, (10000, a2.size))], 1)
        med = float(np.median(a1) / np.median(a2))
        lo = float(np.percentile(sp, 2.5))
        _, ds1 = noise(mp[1])
        noisy = isinstance(ds1, float) and ds1 > 0.05
        v = "NOISY" if noisy else ("PASS" if med >= 1.6 and lo >= 1.6 else "INCONCLUSIVE" if med >= 1.6 else "STOP")
        res.update(speedup_k2=med, speedup_k2_ci_lo=lo, verdict=v,
                   speedup_k4=(float(np.median(a1) / np.median(mp[4])) if 4 in mp else None))
    else:
        res["verdict"] = "NOT_MEASURED"
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "gm06_report.json"), "w"), indent=1)
    print(json.dumps(res))


def self_test() -> bool:
    p = split_bytes(10, 3, 3)
    ok = p == [[(0, 3), (9, 1)], [(3, 3)], [(6, 3)]]
    ok &= sum(s for path in split_bytes(16 * GiB, 4, CHUNK_MIB << 20) for _, s in path) == 16 * GiB
    print("self-test", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("run")
    s.add_argument("--out-dir", default=OUT)
    sub.add_parser("report")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    {"run": cmd_run, "report": cmd_report}[a.cmd](a)
