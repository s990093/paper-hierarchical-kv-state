#!/usr/bin/env python3
"""numa_probe.py — 量時間的 run 記下 NUMA 位置（只記錄，不改任何判準；使用者 2026-10-25）。
記三件事：GPU 在哪個 NUMA 節點、job 的 cpuset（以及它涵蓋哪些節點）、主機緩衝（pinned 或一般）的頁實際在哪個節點。
H200 節點是 2× EPYC 9554：pinned 記憶體和卡不在同一邊時 H2D 會變慢（H200_SLURM.md；慢多少 NOT_MEASURED），
而每個 job 拿到的卡可能不同 → 每個量時間的 run 都要記。
查不到就照實寫 NOT_AVAILABLE（附原因），不當成「同一邊」（CLAUDE.md 規則 7）。

  job [--out J]   不用 torch：這個 job 看得到的卡（nvidia-smi 的 PCI bus id → sysfs numa_node）、cpuset、mems、每個節點的 CPU
  --self-test     解析與判定邏輯（任何平台）；Linux 上另配一塊一般記憶體、用 libcuda 配一塊 pinned，move_pages 要查得到節點

程式內：
  probe(torch_dev, {名稱: 緩衝}) → dict（直接寫 JSON）；緩衝＝tensor、有 .parts 的 Enc、或它們的 list
  row_fields(p) → 每列 CSV 加的四欄：gpu_numa、cpuset_numa、buf_numa、numa_local
頁的位置用 move_pages(2)（nodes＝NULL：只查、不搬）每組抽 SAMPLES 頁；另記包住這些頁的 VMA 在 /proc/self/numa_maps 的整段頁數。
"""
from __future__ import annotations

import argparse
import errno
import glob
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime

NA = "NOT_AVAILABLE"
SAMPLES = 64
ROW_FIELDS = ["gpu_numa", "cpuset_numa", "buf_numa", "numa_local"]
_NR_MOVE_PAGES = {"x86_64": 279, "aarch64": 239}


def ts() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_list(s: str) -> list:
    """「0-3,8」→ [0, 1, 2, 3, 8]（cpulist／Mems_allowed_list 的格式）。"""
    out = []
    for part in s.strip().split(","):
        if not part:
            continue
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def fmt_list(xs) -> str:
    xs = sorted(set(xs))
    out, i = [], 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(str(xs[i]) if i == j else f"{xs[i]}-{xs[j]}")
        i = j + 1
    return ",".join(out)


def node_cpus() -> dict:
    """{節點: [CPU]}（sysfs）。"""
    out = {}
    for p in sorted(glob.glob("/sys/devices/system/node/node[0-9]*/cpulist")):
        out[int(os.path.basename(os.path.dirname(p))[4:])] = parse_list(open(p).read())
    if not out:
        raise FileNotFoundError("/sys/devices/system/node/node*/cpulist")
    return out


def job_info() -> dict:
    out = dict(env={k: os.environ.get(k, "") for k in ("SLURM_JOB_ID", "SLURMD_NODENAME", "SLURM_CPUS_ON_NODE", "SLURM_JOB_GPUS",
                                                        "CUDA_VISIBLE_DEVICES")})
    try:
        st = dict(l.split(":", 1) for l in open("/proc/self/status").read().splitlines() if ":" in l)
        out["cpus_allowed"] = st.get("Cpus_allowed_list", NA).strip()
        out["mems_allowed"] = st.get("Mems_allowed_list", NA).strip()
    except OSError as ex:
        out["cpus_allowed"] = out["mems_allowed"] = f"{NA}: {ex!r}"
    try:
        aff = sorted(os.sched_getaffinity(0))
        out["affinity"] = fmt_list(aff)
        nc = node_cpus()
        out["node_cpulist"] = {str(n): fmt_list(c) for n, c in nc.items()}
        out["cpuset_nodes"] = [n for n, c in sorted(nc.items()) if set(c) & set(aff)]
    except (AttributeError, OSError, ValueError) as ex:          # macOS 沒有 sched_getaffinity／sysfs
        out.setdefault("affinity", f"{NA}: {ex!r}")
        out["cpuset_nodes"] = f"{NA}: {ex!r}"
    # 本地節點沒空間時，kernel 會把頁配到另一個節點（zone_reclaim_mode＝0 時不先回收本地的 page cache）：記下來才看得出原因
    # （lab-3090 10/25：CPU 綁節點 1、節點 1 只剩 0.47 GiB → pinned 57／64 頁落在節點 0）
    try:
        mem = {}
        for p in sorted(glob.glob("/sys/devices/system/node/node[0-9]*/meminfo")):
            kv = {f[2].rstrip(":"): int(f[3]) for f in (l.split() for l in open(p)) if len(f) >= 4 and f[3].isdigit()}
            mem[os.path.basename(os.path.dirname(p))[4:]] = {k: round(kv[k] / 2**20, 2) for k in ("MemTotal", "MemFree", "FilePages")
                                                              if k in kv}
        out["node_mem_gib"] = mem or f"{NA}: 沒有 /sys/devices/system/node/node*/meminfo"
    except (OSError, ValueError) as ex:
        out["node_mem_gib"] = f"{NA}: {ex!r}"
    for k, p in (("zone_reclaim_mode", "/proc/sys/vm/zone_reclaim_mode"), ("numa_balancing", "/proc/sys/kernel/numa_balancing")):
        try:
            out[k] = open(p).read().strip()
        except OSError as ex:
            out[k] = f"{NA}: {ex!r}"
    return out


def sysfs_bus(bus_id: str) -> str:
    """nvidia-smi 的「00000000:1B:00.0」→ sysfs 的「0000:1b:00.0」。"""
    dom, rest = bus_id.strip().split(":", 1)
    return f"{int(dom, 16):04x}:{rest.lower()}"


def pci_numa(bus: str):
    """sysfs numa_node（-1＝平台沒報，照實記，不當成 0）；讀不到 → NOT_AVAILABLE 字串。"""
    try:
        return int(open(f"/sys/bus/pci/devices/{bus}/numa_node").read())
    except (OSError, ValueError) as ex:
        return f"{NA}: {ex!r}"


def smi_gpus():
    """這個 job 看得到的卡：[{index, uuid, bus, numa_node}]；nvidia-smi 失敗 → NOT_AVAILABLE 字串。"""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=60)
        if r.returncode != 0:
            return f"{NA}: nvidia-smi exit {r.returncode}: {r.stderr.strip()[:200]}"
        out = []
        for line in r.stdout.strip().splitlines():
            idx, uuid, bus_id = (x.strip() for x in line.split(","))
            bus = sysfs_bus(bus_id)
            out.append(dict(index=int(idx), uuid=uuid, bus=bus, numa_node=pci_numa(bus)))
        return out
    except (OSError, subprocess.SubprocessError, ValueError) as ex:
        return f"{NA}: {ex!r}"


def _strip_gpu(u: str) -> str:
    u = u.strip().lower()
    return u[4:] if u.startswith("gpu-") else u


def torch_gpu(dev: int, smi=None) -> dict:
    """torch 的 device dev 是哪一張實體卡（PCI 位址優先、其次 uuid 對 nvidia-smi）、在哪個 NUMA 節點。"""
    import torch
    pr = torch.cuda.get_device_properties(dev)
    out = dict(torch_index=dev, name=pr.name, bus=None, uuid=_strip_gpu(str(getattr(pr, "uuid", "") or "")) or None)
    if all(hasattr(pr, k) for k in ("pci_domain_id", "pci_bus_id", "pci_device_id")):
        out["bus"] = f"{pr.pci_domain_id:04x}:{pr.pci_bus_id:02x}:{pr.pci_device_id:02x}.0"
    smi = smi_gpus() if smi is None else smi
    if isinstance(smi, list):
        hit = [g for g in smi if (out["bus"] and g["bus"] == out["bus"]) or (out["uuid"] and _strip_gpu(g["uuid"]) == out["uuid"])]
        if len(hit) == 1:
            out.update(smi_index=hit[0]["index"], bus=hit[0]["bus"])
        elif not out["bus"]:
            out["match_error"] = f"{len(hit)} 張卡對得上 uuid {out['uuid']}"
    out["numa_node"] = pci_numa(out["bus"]) if out["bus"] else f"{NA}: 沒有 PCI 位址（{out.get('match_error', smi)}）"
    return out


def spans(x) -> list:
    """緩衝 → [(ptr, nbytes)]：torch tensor、有 .parts 的 Enc（m10_codec）、(ptr, nbytes)、或它們的 list。"""
    if isinstance(x, tuple) and len(x) == 2 and all(isinstance(v, int) for v in x):
        return [x]
    if isinstance(x, (list, tuple)):
        return [s for y in x for s in spans(y)]
    if hasattr(x, "parts") and isinstance(x.parts, dict):
        return spans(list(x.parts.values()))
    if hasattr(x, "data_ptr"):
        return [(x.data_ptr(), x.numel() * x.element_size())] if x.numel() else []
    raise TypeError(type(x))


def _move_pages(addrs: list):
    """move_pages(2) 只查：每頁的節點或 -errno；整個呼叫失敗 → NOT_AVAILABLE 字串。"""
    import ctypes
    nr = _NR_MOVE_PAGES.get(platform.machine())
    if not sys.platform.startswith("linux") or nr is None:
        return f"{NA}: move_pages 只在 Linux x86_64／aarch64（{sys.platform} {platform.machine()}）"
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    pages = (ctypes.c_void_p * len(addrs))(*addrs)
    st = (ctypes.c_int * len(addrs))()
    rc = libc.syscall(ctypes.c_long(nr), ctypes.c_int(0), ctypes.c_ulong(len(addrs)), pages, None, st, ctypes.c_int(0))
    if rc < 0:
        e = ctypes.get_errno()
        return f"{NA}: move_pages rc={rc} errno={errno.errorcode.get(e, e)}"
    return list(st)


def _vma_numa(starts: set) -> dict:
    """包住抽樣頁的 VMA（/proc/self/maps）在 /proc/self/numa_maps 的「N<節點>=頁數」加總（整段，不只抽樣的頁）。"""
    vmas = []
    for line in open("/proc/self/maps"):
        a, b = (int(v, 16) for v in line.split()[0].split("-"))
        vmas.append((a, b))
    want = {a for a, b in vmas for p in starts if a <= p < b}
    tot, seen = Counter(), 0
    for line in open("/proc/self/numa_maps"):
        f = line.split()
        if int(f[0], 16) in want:
            seen += 1
            for kv in f[1:]:
                if kv[0] == "N" and "=" in kv and kv[1:kv.index("=")].isdigit():
                    tot[int(kv[1:kv.index("=")])] += int(kv.split("=")[1])
    return dict(vmas=len(want), vmas_in_numa_maps=seen, pages={str(n): c for n, c in sorted(tot.items())})


def buffer_nodes(x, n: int = SAMPLES) -> dict:
    """一組緩衝：均勻抽 n 頁（每段至少 1 頁）查節點 → {pages_sampled, nodes, errors, bytes, vma}。"""
    sp = spans(x)
    if not sp:
        return dict(status=f"{NA}: 沒有緩衝")
    pg = os.sysconf("SC_PAGE_SIZE")
    per = max(1, n // len(sp))
    addrs = []
    for ptr, nb in sp:
        first = ptr // pg * pg
        npg = max(1, (ptr + nb - first + pg - 1) // pg)
        addrs += sorted({first + (i * npg // min(per, npg)) * pg for i in range(min(per, npg))})
    st = _move_pages(addrs)
    out = dict(spans=len(sp), bytes=sum(nb for _, nb in sp), pages_sampled=len(addrs))
    if isinstance(st, str):
        out["status"] = st
    else:
        out["nodes"] = {str(k): v for k, v in sorted(Counter(s for s in st if s >= 0).items())}
        out["errors"] = {errno.errorcode.get(-s, str(s)): c for s, c in Counter(s for s in st if s < 0).items()}
    try:
        out["vma"] = _vma_numa(set(addrs))
    except (OSError, ValueError) as ex:
        out["vma"] = f"{NA}: {ex!r}"
    return out


def summarize(p: dict) -> dict:
    """四欄。numa_local：GPU、job 的 CPU、抽樣到的緩衝頁全在同一個節點 → yes；看得到任何一個不在 GPU 那一邊 → no；
    其餘（GPU 節點未知或 -1、cpuset 節點未知、有頁查不到）→ NOT_AVAILABLE＋原因。"""
    g = p.get("gpu", {}).get("numa_node", f"{NA}: 沒有指定 GPU")
    cn = p["job"].get("cpuset_nodes", NA)
    pages, bad = Counter(), 0
    for b in p.get("buffers", {}).values():
        for k, c in b.get("nodes", {}).items():
            pages[int(k)] += c
        bad += sum(b.get("errors", {}).values()) + (b["pages_sampled"] if "status" in b and "pages_sampled" in b else 0)
    g_ok = isinstance(g, int) and g >= 0
    cn_ok = isinstance(cn, list) and len(cn) > 0
    if g_ok and ((pages and set(pages) != {g}) or (cn_ok and set(cn) != {g})):
        local = "no"
    else:
        why = ([] if g_ok else [f"gpu numa_node={g}"]) + ([] if cn_ok else ["cpuset 節點未知"]) + \
              ([] if pages else ["沒有查到任何緩衝頁"]) + ([f"{bad} 頁查不到"] if bad else [])
        local = f"{NA}: " + "; ".join(why) if why else "yes"
    return dict(gpu_numa=g if g_ok else (g if isinstance(g, int) else NA), cpuset_numa=",".join(map(str, cn)) if cn_ok else NA,
                buf_numa=";".join(f"{k}:{v}" for k, v in sorted(pages.items())) or NA, numa_local=local)


def probe(torch_dev=None, buffers=None, smi=None) -> dict:
    """smi：前一次 probe 的 gpus_visible（同一個 job 裡卡不會變；省一次 nvidia-smi）。"""
    t0 = time.perf_counter()
    smi = smi_gpus() if smi is None else smi
    p = dict(ts=ts(), host=platform.node(), kernel=platform.release(), job=job_info(), gpus_visible=smi)
    if torch_dev is not None:
        try:
            p["gpu"] = torch_gpu(torch_dev, smi)
        except Exception as ex:          # noqa: BLE001
            p["gpu"] = dict(numa_node=f"{NA}: {ex!r}")
    p["buffers"] = {k: buffer_nodes(v) for k, v in (buffers or {}).items()}
    p["summary"] = summarize(p)
    p["probe_s"] = round(time.perf_counter() - t0, 4)
    return p


def row_fields(p: dict) -> dict:
    return {k: p["summary"][k] for k in ROW_FIELDS}


def save(out_csv: str, rid: str, **probes):
    """CSV 旁邊的 <名稱>_numa.json：{run_id: {時點: probe}}（CSV 用附加模式時，舊的 run 保留）。"""
    path = os.path.splitext(out_csv)[0] + "_numa.json"
    d = json.load(open(path)) if os.path.exists(path) else {}
    d[rid] = probes
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json.dump(d, open(path, "w"), indent=1, ensure_ascii=False)
    return path


def self_test() -> bool:
    ok = True

    def chk(name, c):
        nonlocal ok
        print(("PASS " if c else "FAIL ") + name)
        ok &= bool(c)
    chk("parse_list／fmt_list 互逆", parse_list("0-3,8,10-11") == [0, 1, 2, 3, 8, 10, 11] and fmt_list([11, 0, 1, 2, 3, 8, 10]) == "0-3,8,10-11")
    chk("sysfs_bus", sysfs_bus("00000000:1B:00.0") == "0000:1b:00.0")
    base = dict(job=dict(cpuset_nodes=[1]), gpu=dict(numa_node=1))
    chk("全在節點 1 → yes", summarize(dict(base, buffers={"a": dict(nodes={"1": 64}, errors={}, pages_sampled=64)}))["numa_local"] == "yes")
    chk("有頁在節點 0 → no", summarize(dict(base, buffers={"a": dict(nodes={"0": 3, "1": 61}, errors={}, pages_sampled=64)}))["numa_local"] == "no")
    chk("cpuset 在節點 0 → no", summarize(dict(base, job=dict(cpuset_nodes=[0]), buffers={"a": dict(nodes={"1": 64}, errors={})}))["numa_local"] == "no")
    chk("GPU 節點 -1 → NOT_AVAILABLE", summarize(dict(base, gpu=dict(numa_node=-1), buffers={"a": dict(nodes={"1": 64}, errors={})}))
        ["numa_local"].startswith(NA))
    chk("有頁查不到 → NOT_AVAILABLE（不當成 yes）",
        summarize(dict(base, buffers={"a": dict(nodes={"1": 60}, errors={"EFAULT": 4}, pages_sampled=64)}))["numa_local"].startswith(NA))
    chk("move_pages 整個失敗 → NOT_AVAILABLE",
        summarize(dict(base, buffers={"a": dict(status=f"{NA}: x", pages_sampled=64)}))["numa_local"].startswith(NA))
    chk("沒有緩衝 → NOT_AVAILABLE", summarize(dict(base, buffers={}))["numa_local"].startswith(NA))
    if sys.platform.startswith("linux"):
        import ctypes
        buf = ctypes.create_string_buffer(8 << 20)
        ctypes.memset(buf, 1, 8 << 20)
        r = buffer_nodes([(ctypes.addressof(buf), 8 << 20)])
        print(" 一般記憶體：", json.dumps(r, ensure_ascii=False))
        chk("一般記憶體：每頁都查得到節點", sum(r.get("nodes", {}).values()) == r["pages_sampled"])
        try:
            cu = ctypes.CDLL("libcuda.so.1")
        except OSError as ex:
            cu = None
            print(f" 沒有 libcuda（{ex!r}）→ 跳過 pinned")
        if cu is not None:
            dev, ctx, p = ctypes.c_int(), ctypes.c_void_p(), ctypes.c_void_p()
            rc = [cu.cuInit(0), cu.cuDeviceGet(ctypes.byref(dev), 0), cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev),
                  cu.cuMemHostAlloc(ctypes.byref(p), ctypes.c_size_t(8 << 20), ctypes.c_uint(0))]
            if any(rc):
                print(f" libcuda 回傳 {rc} → 跳過 pinned（照實記）")
            else:
                ctypes.memset(p, 1, 8 << 20)
                r = buffer_nodes([(p.value, 8 << 20)])
                print(" pinned（cuMemHostAlloc）：", json.dumps(r, ensure_ascii=False))
                chk("pinned：每頁都查得到節點（查不到時 probe 會寫 NOT_AVAILABLE，這裡只是確認這台的 driver 查得到）",
                    sum(r.get("nodes", {}).values()) == r["pages_sampled"])
        print(" job：", json.dumps(job_info(), ensure_ascii=False))
        print(" gpus：", json.dumps(smi_gpus(), ensure_ascii=False))
    else:
        print(f" {sys.platform}：", _move_pages([0]))
    print("self-test", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("job")
    s.add_argument("--out", default="")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    p = probe()
    print(json.dumps(p, ensure_ascii=False, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(p, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
