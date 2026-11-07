#!/usr/bin/env python3
"""next_a_corun.py — W6a 與 W6b 在同一個 Slurm job 的兩張卡上同時跑（PREREG 偏離 20；T2 的 CAL-OK：
docs/h200/icpe/T2_parallel_calibration.md，兩個縮小版 GPU 重播 job、卡在不同 NUMA 節點，可以同時跑）。

  plan --out-dir D [--fake JSON]    看這個 job 的兩張卡：CUDA_VISIBLE_DEVICES → nvidia-smi 的 PCI → sysfs 的 numa_node；
                                    CPU 依 NUMA 分給兩邊。兩張卡在不同 NUMA 節點 → 印出 shell 變數（GA GB CA CB …）、寫
                                    D/corun_plan_<job>.json，exit 0。不是兩張卡、同一個 NUMA、查不到 NUMA → exit 2（不平行、不跑）。
  classify --out-dir D              只報告：每個 run 的時間窗（w6a／w6b 的 overlap.csv）和另一邊 runner 的活動時間窗
                                    （D/corun.csv）比，覆蓋率 ≥ 0.9 → par、0 → solo、其他 → partial（同 T2 §3）。→ D/corun_runs.csv
  selftest                          plan 的判斷在合成輸入上的自檢

查不到 ≠ 沒有（CLAUDE.md 規則 7）：任何一項查不到都當成「不能平行」，exit 2。
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from datetime import datetime

COVER = 0.9


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sysfs_bus(bus):
    """nvidia-smi 的 00000000:83:00.0 → sysfs 的 0000:83:00.0。"""
    dom, rest = bus.strip().split(":", 1)
    return f"{int(dom, 16):04x}:{rest.lower()}"


def _probe():
    """真機：讀 CUDA_VISIBLE_DEVICES、nvidia-smi、sysfs、這個 process 能用的 CPU。"""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,pci.bus_id,uuid", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise RuntimeError(f"nvidia-smi 失敗（exit {out.returncode}）：{out.stderr.strip()}")
    smi = [[x.strip() for x in ln.split(",")] for ln in out.stdout.strip().split("\n") if ln.strip()]
    numa = {}
    for r in smi:
        p = f"/sys/bus/pci/devices/{_sysfs_bus(r[1])}/numa_node"
        numa[r[1]] = int(open(p).read().strip())
    cpus = sorted(os.sched_getaffinity(0))
    cpu_node = {}
    for c in cpus:
        nodes = glob.glob(f"/sys/devices/system/cpu/cpu{c}/node[0-9]*")
        if len(nodes) != 1:
            raise RuntimeError(f"cpu{c} 的 NUMA 節點查不到：{nodes}")
        cpu_node[c] = int(os.path.basename(nodes[0])[4:])
    return dict(cvd=cvd, smi=smi, numa=numa, cpus=cpus, cpu_node=cpu_node)


def make_plan(inp):
    """回傳 (ok, plan dict)。ok＝False 時 plan['reason'] 說為什麼不能平行。"""
    cvd, smi, numa, cpus, cpu_node = inp["cvd"], inp["smi"], inp["numa"], list(inp["cpus"]), {int(k): v for k, v in inp["cpu_node"].items()}
    plan = dict(ts=_now(), cvd=cvd, smi=smi, cpus=cpus)
    if not cvd:
        return False, dict(plan, reason="沒有 CUDA_VISIBLE_DEVICES")
    ents = [e.strip() for e in cvd.split(",") if e.strip()]
    if len(ents) != 2:
        return False, dict(plan, reason=f"要剛好兩張卡，CUDA_VISIBLE_DEVICES={cvd!r}")
    gpus = []
    for e in ents:
        if e.startswith(("GPU-", "MIG-")):
            hit = [r for r in smi if r[2].startswith(e)]
        else:
            hit = [r for r in smi if r[0] == e]
        if len(hit) != 1:
            return False, dict(plan, reason=f"CUDA_VISIBLE_DEVICES 的 {e!r} 對不到 nvidia-smi 的卡：{smi}")
        bus = hit[0][1]
        n = numa.get(bus)
        if n is None or n < 0:
            return False, dict(plan, reason=f"{bus} 的 NUMA 節點查不到（{n}）")
        gpus.append(dict(cvd=e, smi_index=hit[0][0], bus=bus, uuid=hit[0][2], numa=n))
    if gpus[0]["numa"] == gpus[1]["numa"]:
        return False, dict(plan, gpus=gpus, reason=f"兩張卡在同一個 NUMA 節點（{gpus[0]['numa']}）：T2 沒有量過 → 不平行")
    half = len(cpus) // 2
    loc = [[c for c in cpus if cpu_node.get(c) == g["numa"]] for g in gpus]
    if half >= 1 and all(len(x) >= half for x in loc):
        sets, how = [x[:half] for x in loc], "local"           # 各給自己那張卡的 NUMA 節點上的 half 個 CPU
    else:
        sets, how = [cpus[:half], cpus[half:2 * half]], "halves"   # Slurm 給的 CPU 沒有兩邊各一半：照編號對半分（照實記）
    if any(len(s) == 0 for s in sets):
        return False, dict(plan, gpus=gpus, reason=f"CPU 分不出兩份：{cpus}")
    for g, s in zip(gpus, sets):
        g["cpus"] = s
        g["cpu_nodes"] = sorted({cpu_node.get(c, -1) for c in s})
        g["numa_local"] = all(cpu_node.get(c) == g["numa"] for c in s)
    return True, dict(plan, gpus=gpus, cpu_split=how)


def _cpulist(s):
    return ",".join(str(c) for c in s)


def cmd_plan(a):
    try:
        inp = json.load(open(a.fake)) if a.fake else _probe()
    except Exception as e:                                   # 查不到 ≠ 沒有（規則 7）
        print(f"[corun] 查不到卡／NUMA／CPU：{e} → 不平行", file=sys.stderr)
        return 2
    ok, plan = make_plan(inp)
    job = os.environ.get("SLURM_JOB_ID", "nojob")
    plan.update(job=job, node=os.environ.get("SLURMD_NODENAME", ""), slurm_job_gpus=os.environ.get("SLURM_JOB_GPUS", ""), ok=ok)
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        json.dump(plan, open(os.path.join(a.out_dir, f"corun_plan_{job}.json"), "w"), ensure_ascii=False, indent=1)
    if not ok:
        print(f"[corun] 不平行：{plan['reason']}", file=sys.stderr)
        return 2
    ga, gb = plan["gpus"]
    print(f"GA={ga['cvd']} GB={gb['cvd']} CA={_cpulist(ga['cpus'])} CB={_cpulist(gb['cpus'])} "
          f"NA={ga['numa']} NB={gb['numa']} BA={ga['bus']} BB={gb['bus']} SPLIT={plan['cpu_split']}")
    return 0


def _windows(path):
    if not os.path.exists(path):
        return []
    return list(csv.DictReader(open(path)))


def cmd_classify(a):
    import pandas as pd
    D = a.out_dir
    cr = pd.read_csv(os.path.join(D, "corun.csv"), dtype=str)
    rows = []
    for sub, other in (("w6a", "w6b"), ("w6b", "w6a")):
        o = cr[(cr.sub == other)]
        act = [(pd.Timestamp(r.t_start), pd.Timestamp(r.t_end)) for r in o.itertuples() if isinstance(r.t_end, str) and r.t_end]
        for w in _windows(os.path.join(a.res_dir, sub, "overlap.csv")):
            t0, t1 = pd.Timestamp(w["t_start"]), pd.Timestamp(w["t_end"])
            if w.get("job") not in set(cr.job):
                continue
            dur = (t1 - t0).total_seconds()
            cov = sum(max(0.0, (min(t1, e) - max(t0, s)).total_seconds()) for s, e in act) / dur if dur > 0 else float("nan")
            cond = "par" if cov >= COVER else ("solo" if cov == 0 else "partial")
            rows.append(dict(run_id=w["run_id"], sub=sub, step=w["step"], t_start=w["t_start"], t_end=w["t_end"], cover=round(cov, 4), cond=cond))
    R = pd.DataFrame(rows)
    R.insert(1, "ts", _now())
    R.to_csv(os.path.join(D, "corun_runs.csv"), index=False)
    print(R.groupby(["sub", "cond"]).size().to_string() if len(R) else "沒有 run")
    return 0


def cmd_selftest(_a):
    smi = [["0", "00000000:31:00.0", "GPU-a"], ["1", "00000000:E5:00.0", "GPU-b"]]
    numa = {"00000000:31:00.0": 0, "00000000:E5:00.0": 1}
    node = {c: (0 if c < 64 else 1) for c in range(128)}
    cases = [
        ("不同 NUMA、CPU 兩邊各 16", dict(cvd="0,1", smi=smi, numa=numa, cpus=list(range(16)) + list(range(64, 80)), cpu_node=node), True, "local"),
        ("不同 NUMA、CPU 全在 0", dict(cvd="0,1", smi=smi, numa=numa, cpus=list(range(32)), cpu_node=node), True, "halves"),
        ("同一個 NUMA", dict(cvd="0,1", smi=smi, numa={"00000000:31:00.0": 0, "00000000:E5:00.0": 0}, cpus=list(range(32)), cpu_node=node), False, None),
        ("NUMA 查不到（-1）", dict(cvd="0,1", smi=smi, numa={"00000000:31:00.0": -1, "00000000:E5:00.0": 1}, cpus=list(range(32)), cpu_node=node), False, None),
        ("只有一張卡", dict(cvd="0", smi=smi[:1], numa=numa, cpus=list(range(16)), cpu_node=node), False, None),
        ("卡對不到", dict(cvd="0,3", smi=smi, numa=numa, cpus=list(range(32)), cpu_node=node), False, None),
        ("UUID 形式", dict(cvd="GPU-b,GPU-a", smi=smi, numa=numa, cpus=list(range(16)) + list(range(64, 80)), cpu_node=node), True, "local"),
    ]
    bad = 0
    for name, inp, want_ok, want_split in cases:
        ok, p = make_plan(inp)
        good = ok == want_ok and (not ok or p["cpu_split"] == want_split)
        if ok and good:
            ga, gb = p["gpus"]
            good = ga["numa"] != gb["numa"] and not (set(ga["cpus"]) & set(gb["cpus"])) and len(ga["cpus"]) == len(gb["cpus"]) == 16
            if want_split == "local":
                good = good and ga["numa_local"] and gb["numa_local"]
        print(("PASS" if good else "FAIL"), name, "→", (p.get("cpu_split") if ok else p.get("reason")))
        bad += not good
    print("selftest", "PASS" if bad == 0 else f"FAIL（{bad}）")
    return 0 if bad == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("plan"); s.add_argument("--out-dir"); s.add_argument("--fake")
    s = sp.add_parser("classify"); s.add_argument("--out-dir", required=True); s.add_argument("--res-dir", required=True)
    sp.add_parser("selftest")
    a = ap.parse_args()
    sys.exit({"plan": cmd_plan, "classify": cmd_classify, "selftest": cmd_selftest}[a.cmd](a))


if __name__ == "__main__":
    main()
