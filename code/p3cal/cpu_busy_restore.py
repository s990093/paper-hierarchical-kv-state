"""cpu_busy_restore.py — 主機 CPU 忙的時候，CPU→GPU 搬運慢多少（第三階段 3090 校準，2026-10-18）。

一個 GPU process 載入模型一次，然後依 --schedule 逐段（phase）開關背景壓力（code/p3cal/cpu_stress.py：N 個執行緒做 numpy 大塊複製），
每一段量兩件事：
  (a) 搬運頻寬：m7_calib.py c0 的量法逐字照抄（64 MiB／1 GiB × pinned／pageable × H2D／D2H，torch.cuda.synchronize 牆鐘；
      加 c0 的「Llama 形狀 64 MiB block → KV 緩衝 strided 視圖」H2D／D2H），另加 m10_speed calib-f 的「本模型 block → km.chunk_view」H2D。
  (b) 真實 CPU 層的還原：m7_restore_harness.Restorer（不改它），config E0（全 BF16）、tier＝cpu_real（不限速、真的 H2D），
      mode cake／load_only（--reps 次）＋compute_only（--reps-compute 次），每段內順序洗牌；寫入（prefill＋pinned 副本）和
      code/p3x/x7_restore.py 相同。每段每個 mode 的第一次做 KV 逐位元檢查（kv_bad）。
每段另記：/proc/loadavg（段頭、段尾）、/proc/stat 算出的兩個 NUMA 節點 CPU 使用率、壓力程式自己每秒的複製速度
（取段內中位數）、host_contention、壓力程式的 READY／DONE。

--schedule 例："0:same 8:same 16:same 32:same 48:same 48:same 32:same 16:same 8:same 0:same 48:all 48:other"
  N:place，place＝same（壓力綁在這張 GPU 所在的 NUMA 節點的 CPU；量測 process 也在那裡，呼叫者用 taskset）
               ／other（另一個 NUMA 節點）／all（不綁）。N＝0 不開壓力。
輸出（--out-dir）：h2d.csv、restores.csv、phases.csv、stress_<k>.csv、meta_<run_id>.json。每列都有 run_id、ts、phase、N、place。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)

import torch  # noqa: E402

MiB = 1 << 20
GiB = 1 << 30
PH = ["run_id", "ts", "phase", "N", "place", "stress_cpus"]
RFIELDS = PH + ["model", "length", "n_blocks", "config", "mode", "tier", "rep", "order_idx", "b", "n_recompute",
                "n_load_cpu", "bytes_read", "t_restore", "t_recompute", "t_load", "t_h2d", "t_dequant", "t_wait",
                "waiting_line", "t_new_turn", "ttft", "copy_bound", "kv_bad", "contaminated_gpu"]
HFIELDS = PH + ["item", "size_mib", "rep", "GiBps", "ms"]
PFIELDS = PH + ["t_start", "t_end", "secs", "loadavg_start", "loadavg_end", "procs_running_start", "procs_running_end",
                "cpu_util_node0", "cpu_util_node1", "cpu_util_all", "stress_ready", "stress_copy_GiBps_median",
                "stress_copy_GiBps_min", "stress_copy_GiBps_max", "stress_n_samples", "stress_done",
                "host_level", "host_foreign_procs", "host_foreign_gpus", "self_affinity"]


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def cpulist(node):
    return open(f"/sys/devices/system/node/node{node}/cpulist").read().strip()


def expand(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        else:
            out.append(int(part))
    return out


def gpu_numa_node(phys_idx):
    bus = subprocess.run(["nvidia-smi", "-i", str(phys_idx), "--query-gpu=pci.bus_id", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=30, check=True).stdout.strip()
    # nvidia-smi：00000000:1A:00.0 → sysfs：0000:1a:00.0
    dom, rest = bus.split(":", 1)
    path = f"/sys/bus/pci/devices/{dom[-4:].lower()}:{rest.lower()}/numa_node"
    v = int(open(path).read().strip())
    if v < 0:
        raise RuntimeError(f"{path} = {v}：查不到 GPU 的 NUMA 節點（規則 7：不猜）")
    return v, bus


def stat_snapshot():
    d = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3] != " ":
            p = line.split()
            v = list(map(int, p[1:]))
            idle = v[3] + v[4]                       # idle + iowait
            d[int(p[0][3:])] = (sum(v[:8]), idle)    # user..steal（不含 guest，guest 已算在 user 裡）
    return d


def util(s0, s1, cpus):
    tot = sum(s1[c][0] - s0[c][0] for c in cpus)
    idle = sum(s1[c][1] - s0[c][1] for c in cpus)
    return round(1 - idle / tot, 4) if tot > 0 else ""


class Stress:
    def __init__(self, n, cpus, mib, report):
        self.n, self.cpus, self.mib, self.report = n, cpus, mib, report
        self.p, self.ready, self.done = None, "", ""

    def __enter__(self):
        if self.n > 0:
            cmd = [sys.executable, os.path.join(HERE, "cpu_stress.py"), "--threads", str(self.n), "--mib", str(self.mib),
                   "--report", self.report]
            if self.cpus:
                cmd += ["--cpus", self.cpus]
            # 壓力程式自己的 affinity 由 --cpus 設；呼叫者（我們）被 taskset 綁在 GPU 的節點，子 process 會繼承，
            # 所以 place＝all／other 時要先把子 process 的 affinity 放開，交給 cpu_stress 自己設
            self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True,
                                      preexec_fn=lambda: os.sched_setaffinity(0, range(os.cpu_count())))
            line = self.p.stdout.readline()
            if not line.startswith("READY"):
                self.p.kill()
                raise RuntimeError(f"cpu_stress 沒有啟動：{line!r}")
            self.ready = line.strip()
        return self

    def __exit__(self, *a):
        if self.p:
            self.p.terminate()
            out, _ = self.p.communicate(timeout=120)
            self.done = " ".join(x for x in out.splitlines() if x.startswith("DONE"))
            if self.p.returncode != 0:
                raise RuntimeError(f"cpu_stress rc={self.p.returncode}: {out[-2000:]}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["llama", "qwen"], default="qwen")
    ap.add_argument("--length", type=int, default=32768)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--reps", type=int, default=5, help="每段 cake、load_only 各幾次")
    ap.add_argument("--reps-compute", type=int, default=1, help="每段 compute_only 幾次")
    ap.add_argument("--reps-c0", type=int, default=5, help="c0 每種組合幾次（c0 預設 5）")
    ap.add_argument("--reps-block", type=int, default=20, help="本模型 block H2D 幾次（calib-f 預設 20）")
    ap.add_argument("--stress-mib", type=int, default=256)
    ap.add_argument("--warmup-s", type=float, default=3.0)
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.environ["M7_MODEL_GLOB"] = os.environ[f"M7_{a.model.upper()}_GLOB"]
    from m7_model import CHUNK, KVModel
    from m7_restore_harness import Restorer, Tier
    from gpu_guard import GpuWatcher, host_contention, smi_index

    rid = os.environ.get("RUN_ID", "NO_RUN_ID")
    os.makedirs(a.out_dir, exist_ok=True)
    phys = smi_index(0)
    node, bus = gpu_numa_node(phys)
    nodes = sorted(int(x[4:]) for x in os.listdir("/sys/devices/system/node") if x.startswith("node") and x[4:].isdigit())
    other = [x for x in nodes if x != node]
    place_cpus = {"same": cpulist(node), "other": cpulist(other[0]) if other else None, "all": None}
    node_cpus = {x: expand(cpulist(x)) for x in nodes}
    self_aff = sorted(os.sched_getaffinity(0))
    sched = []
    for s in a.schedule.split():
        nn, pl = s.split(":")
        if pl not in place_cpus:
            raise ValueError(s)
        if pl == "other" and not other:
            raise RuntimeError("只有一個 NUMA 節點，沒有 other")
        sched.append((int(nn), pl))

    n = a.length // CHUNK
    km = KVModel(max_len=n * CHUNK + max(512, a.new_tokens))
    rs = Restorer(km)
    rs.time_h2d = True
    g = torch.Generator().manual_seed(a.seed)
    ids = torch.randint(1000, min(120000, km.cfg.vocab_size), (n * CHUNK + a.new_tokens,), generator=g).to(km.device)
    km.prefill_chunked(ids[:n * CHUNK])
    torch.cuda.synchronize()
    bf = []
    for i in range(n):
        h = km.host_chunk()
        h.copy_(km.chunk_view(i))
        bf.append(h)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # c0 的緩衝：一次配好，每段共用（配置成本不進量測）
    dev = km.device
    c0buf = {}
    for size_mib in (64, 1024):
        nb = size_mib * MiB
        gd = torch.empty(nb, dtype=torch.uint8, device=dev)
        for pinned in (True, False):
            hh = torch.empty(nb, dtype=torch.uint8, pin_memory=pinned)
            hh.fill_(1)
            c0buf[(size_mib, pinned)] = (gd, hh)
    LL, HH, DD = 32, 8, 128                                  # c0 的 Llama 形狀
    kvL = torch.empty((LL, 2, HH, 4096, DD), dtype=torch.bfloat16, device=dev)
    hL = torch.empty((LL, 2, HH, CHUNK, DD), dtype=torch.bfloat16, pin_memory=True)
    hM = km.host_chunk()
    hM.copy_(km.chunk_view(0))
    torch.cuda.synchronize()

    def restore(mode):
        t = Tier("cpu_real", read_Bps=0, write_Bps=0)
        km.kv[:, :, :, :n * CHUNK].zero_()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rec = rs.restore(ids, n, bf, [t] * n, mode)
        t_new, _ = rs.new_turn(ids[n * CHUNK:n * CHUNK + a.new_tokens], n * CHUNK)
        rec["ttft"] = time.perf_counter() - t0
        rec["t_new_turn"] = t_new
        return rec

    def kv_bad(rec):
        bad = 0
        for i, src, _, _ in rec["blocks"]:
            bad += int(not torch.equal(km.chunk_view(i), bf[i].to(dev)))
        return bad

    try:
        watch = GpuWatcher(gpu=phys, own_root=os.getpid(), out_path=os.path.join(a.out_dir, f"gpu_guard_inner_{rid}.json"))
        watch.__enter__()
    except Exception as ex:  # noqa: BLE001  查不到不等於乾淨（規則 7）
        print("GpuWatcher unavailable:", repr(ex), flush=True)
        watch = None

    def contaminated():
        return "NOT_CHECKED" if watch is None else (bool(watch.intruders) or watch.started_clean is False)

    files = {}
    writers = {}
    for k, fl in (("h2d", HFIELDS), ("restores", RFIELDS), ("phases", PFIELDS)):
        f = open(os.path.join(a.out_dir, f"{k}.csv"), "a", newline="")
        w = csv.DictWriter(f, fieldnames=fl, extrasaction="ignore")
        if f.tell() == 0:
            w.writeheader()
        files[k], writers[k] = f, w

    # 暖機（不記）：每個 mode 一次＋c0 每種一次
    for m in ("cake", "load_only", "compute_only"):
        restore(m)
    for (size_mib, pinned), (gd, hh) in c0buf.items():
        gd.copy_(hh, non_blocking=pinned)
        hh.copy_(gd, non_blocking=pinned)
    torch.cuda.synchronize()

    rng = random.Random(f"p3cal-cpubusy|{rid}|{a.seed}")
    meta_ph = []
    for k, (N, place) in enumerate(sched):
        cpus = place_cpus[place]
        base = dict(run_id=rid, phase=k, N=N, place=place, stress_cpus=cpus or "ALL")
        rep_path = os.path.join(a.out_dir, f"stress_{k}.csv")
        try:
            hc = host_contention(exclude_gpu=phys)
        except Exception as ex:  # noqa: BLE001
            hc = {"level": f"UNKNOWN({type(ex).__name__})"}
        with Stress(N, cpus, a.stress_mib, rep_path) as st:
            time.sleep(a.warmup_s if N > 0 else 0.5)
            la0 = open("/proc/loadavg").read().split()
            s0 = stat_snapshot()
            e0 = time.time()
            # (a) c0：照 m7_calib.c0 的迴圈順序
            for size_mib in (64, 1024):
                nb = size_mib * MiB
                for pinned in (True, False):
                    gd, hh = c0buf[(size_mib, pinned)]
                    for rep in range(a.reps_c0):
                        for direction in ("H2D", "D2H"):
                            torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            if direction == "H2D":
                                gd.copy_(hh, non_blocking=pinned)
                            else:
                                hh.copy_(gd, non_blocking=pinned)
                            torch.cuda.synchronize()
                            dt = time.perf_counter() - t0
                            writers["h2d"].writerow(dict(base, ts=ts(), item=f"{direction}_{'pinned' if pinned else 'pageable'}",
                                                         size_mib=size_mib, rep=rep, GiBps=round(nb / dt / GiB, 3),
                                                         ms=round(dt * 1e3, 3)))
            for rep in range(a.reps_c0 * 3):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                kvL[:, :, :, CHUNK:2 * CHUNK].copy_(hL, non_blocking=True)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                writers["h2d"].writerow(dict(base, ts=ts(), item="H2D_pinned_chunk_into_kv_view", size_mib=64, rep=rep,
                                             GiBps=round(64 * MiB / dt / GiB, 3), ms=round(dt * 1e3, 3)))
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                hL.copy_(kvL[:, :, :, CHUNK:2 * CHUNK], non_blocking=True)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                writers["h2d"].writerow(dict(base, ts=ts(), item="D2H_pinned_chunk_from_kv_view", size_mib=64, rep=rep,
                                             GiBps=round(64 * MiB / dt / GiB, 3), ms=round(dt * 1e3, 3)))
            for rep in range(a.reps_block):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                km.chunk_view(1).copy_(hM, non_blocking=True)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                writers["h2d"].writerow(dict(base, ts=ts(), item=f"H2D_pinned_{a.model}_block_into_kv_view",
                                             size_mib=round(km.chunk_bytes / MiB, 3), rep=rep,
                                             GiBps=round(km.chunk_bytes / dt / GiB, 3), ms=round(dt * 1e3, 3)))
            files["h2d"].flush()
            # (b) 還原
            todo = [("cake", r) for r in range(a.reps)] + [("load_only", r) for r in range(a.reps)] + \
                   [("compute_only", r) for r in range(a.reps_compute)]
            rng.shuffle(todo)
            checked = set()
            for oi, (m, rep) in enumerate(todo):
                rec = restore(m)
                kb = ""
                if m not in checked:
                    kb = kv_bad(rec)
                    checked.add(m)
                writers["restores"].writerow(dict(
                    base, ts=ts(), model=a.model, length=a.length, n_blocks=n, config="E0", mode=m, tier="cpu",
                    rep=rep, order_idx=oi, b=rec["meet"], n_recompute=rec["n_recompute"], n_load_cpu=rec["n_load_cpu"],
                    bytes_read=rec["bytes_read"], t_restore=round(rec["t_restore"], 6),
                    t_recompute=round(rec["t_recompute"], 6), t_load=round(rec.get("t_load", 0.0), 6),
                    t_h2d=round(rec["t_h2d"], 6), t_dequant=round(rec["t_dequant"], 6),
                    t_wait=round(rec.get("t_wait", 0.0), 6) if m == "cake" else "",
                    waiting_line=rec.get("waiting_line", ""), t_new_turn=round(rec["t_new_turn"], 6),
                    ttft=round(rec["ttft"], 6), copy_bound=rec["n_copy_bound"], kv_bad=kb,
                    contaminated_gpu=contaminated()))
            files["restores"].flush()
            e1 = time.time()
            s1 = stat_snapshot()
            la1 = open("/proc/loadavg").read().split()
        # 壓力程式段內的每秒複製速度
        sr = []
        if N > 0 and os.path.exists(rep_path):
            for row in csv.DictReader(open(rep_path)):
                t = datetime.fromisoformat(row["ts"]).timestamp()
                if e0 <= t <= e1:
                    sr.append(float(row["copy_GiBps_1s"]))
        writers["phases"].writerow(dict(
            base, ts=ts(), t_start=datetime.fromtimestamp(e0).astimezone().isoformat(timespec="seconds"),
            t_end=datetime.fromtimestamp(e1).astimezone().isoformat(timespec="seconds"), secs=round(e1 - e0, 2),
            loadavg_start=la0[0], loadavg_end=la1[0], procs_running_start=la0[3].split("/")[0],
            procs_running_end=la1[3].split("/")[0],
            cpu_util_node0=util(s0, s1, node_cpus[nodes[0]]),
            cpu_util_node1=util(s0, s1, node_cpus[nodes[1]]) if len(nodes) > 1 else "",
            cpu_util_all=util(s0, s1, sorted(s0)), stress_ready=st.ready,
            stress_copy_GiBps_median=round(statistics.median(sr), 3) if sr else "",
            stress_copy_GiBps_min=min(sr) if sr else "", stress_copy_GiBps_max=max(sr) if sr else "",
            stress_n_samples=len(sr), stress_done=st.done, host_level=hc.get("level"),
            host_foreign_procs=hc.get("foreign_procs", ""), host_foreign_gpus=" ".join(map(str, hc.get("foreign_gpus", []))),
            self_affinity=f"{self_aff[0]}..{self_aff[-1]} ({len(self_aff)} cpus)"))
        files["phases"].flush()
        meta_ph.append({"phase": k, "N": N, "place": place, "ready": st.ready, "done": st.done})
        print(f"[phase {k}] N={N} place={place} {round(e1 - e0, 1)}s stress_med={sr and statistics.median(sr)} "
              f"load={la0[0]}->{la1[0]} contaminated={contaminated()}", flush=True)
        time.sleep(2.0)                                      # 壓力結束後讓記憶體控制器回到原狀

    for f in files.values():
        f.close()
    if watch is not None:
        watch.__exit__(None, None, None)
    json.dump({"run_id": rid, "argv": sys.argv, "gpu_phys": phys, "gpu_bus": bus, "gpu_numa_node": node,
               "place_cpus": place_cpus, "self_affinity": self_aff, "phases": meta_ph,
               "contaminated_inner": None if watch is None else watch.contaminated,
               "inner_verdict": None if watch is None else watch.verdict(),
               "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / GiB, 3)},
              open(os.path.join(a.out_dir, f"meta_{rid}.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
