#!/usr/bin/env python3
"""next_a_corun.py — W6 在同一個 Slurm job 的幾張卡上同時跑。偏離 20：2 張（W6a ∥ W6b，T2 的 CAL-OK：
docs/h200/icpe/T2_parallel_calibration.md，兩個縮小版 GPU 重播 job、卡在不同 NUMA 節點）。偏離 21：4 張（兩個 NUMA 節點各 2 張；
同一個 NUMA 兩張、4 個同時跑都超出 T2 量過的範圍，使用者決定照跑、只報告）。

  plan --n 2|4 --out-dir D [--fake JSON]
                                    看這個 job 的卡（n＝4 時要兩個 NUMA 節點各 2 張）：CUDA_VISIBLE_DEVICES → nvidia-smi 的 PCI → sysfs 的 numa_node；
                                    CPU 依 NUMA 分給兩邊。兩張卡在不同 NUMA 節點 → 印出 shell 變數（GA GB CA CB …）、寫
                                    D/corun_plan_<job>.json，exit 0。不是兩張卡、同一個 NUMA、查不到 NUMA → exit 2（不平行、不跑）。
  classify --out-dir D [--res-dir R]
                                    只報告：每個 run 的時間窗（corun.csv 的 ovl 欄指的 overlap 檔）對同一個 job 其他 lane 的活動
                                    時間窗算覆蓋率；≥ 0.9 的 lane 數＝k → n_concurrent＝1＋k；有 0＜c＜0.9 → partial（同 T2 §3）→ D/corun_runs.csv
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


def make_plan(inp, n=2):
    """回傳 (ok, plan dict)。n 張卡要剛好分在兩個 NUMA 節點、每個節點 n/2 張（n＝2：偏離 20；n＝4：偏離 21）。
    ok＝False 時 plan['reason'] 說為什麼不能平行。卡照 (NUMA, PCI) 排序，第 i 張給第 i 條 lane。"""
    cvd, smi, numa, cpus, cpu_node = inp["cvd"], inp["smi"], inp["numa"], list(inp["cpus"]), {int(k): v for k, v in inp["cpu_node"].items()}
    plan = dict(ts=_now(), n=n, cvd=cvd, smi=smi, cpus=cpus)
    if n not in (2, 4):
        return False, dict(plan, reason=f"n 只能是 2 或 4：{n}")
    if not cvd:
        return False, dict(plan, reason="沒有 CUDA_VISIBLE_DEVICES")
    ents = [e.strip() for e in cvd.split(",") if e.strip()]
    if len(ents) != n:
        return False, dict(plan, reason=f"要剛好 {n} 張卡，CUDA_VISIBLE_DEVICES={cvd!r}")
    gpus = []
    for e in ents:
        if e.startswith(("GPU-", "MIG-")):
            hit = [r for r in smi if r[2].startswith(e)]
        else:
            hit = [r for r in smi if r[0] == e]
        if len(hit) != 1:
            return False, dict(plan, reason=f"CUDA_VISIBLE_DEVICES 的 {e!r} 對不到 nvidia-smi 的卡：{smi}")
        bus = hit[0][1]
        nn = numa.get(bus)
        if nn is None or nn < 0:
            return False, dict(plan, reason=f"{bus} 的 NUMA 節點查不到（{nn}）")
        gpus.append(dict(cvd=e, smi_index=hit[0][0], bus=bus, uuid=hit[0][2], numa=nn))
    gpus.sort(key=lambda g: (g["numa"], g["bus"].lower()))
    cnt = {}
    for g in gpus:
        cnt[g["numa"]] = cnt.get(g["numa"], 0) + 1
    if len(cnt) != 2 or any(v != n // 2 for v in cnt.values()):
        return False, dict(plan, gpus=gpus, reason=f"卡的 NUMA 分布 {cnt} 不是兩個節點各 {n // 2} 張 → 不平行")
    per = len(cpus) // n
    if per < 1:
        return False, dict(plan, gpus=gpus, reason=f"CPU 不夠分：{cpus}")
    loc = {k: [c for c in cpus if cpu_node.get(c) == k] for k in cnt}
    if all(len(loc[k]) >= per * cnt[k] for k in cnt):
        how, used = "local", {k: 0 for k in cnt}            # 各給自己那張卡的 NUMA 節點上的 per 個 CPU
        for g in gpus:
            k = g["numa"]
            g["cpus"] = loc[k][used[k] * per:(used[k] + 1) * per]
            used[k] += 1
    else:
        how = "halves" if n == 2 else "chunks"               # Slurm 給的 CPU 不夠每邊各自的份：照編號均分（照實記）
        for i, g in enumerate(gpus):
            g["cpus"] = cpus[i * per:(i + 1) * per]
    for g in gpus:
        g["cpu_nodes"] = sorted({cpu_node.get(c, -1) for c in g["cpus"]})
        g["numa_local"] = all(cpu_node.get(c) == g["numa"] for c in g["cpus"])
    return True, dict(plan, gpus=gpus, cpu_split=how)


def _cpulist(s):
    return ",".join(str(c) for c in s)


def cmd_plan(a):
    try:
        inp = json.load(open(a.fake)) if a.fake else _probe()
    except Exception as e:                                   # 查不到 ≠ 沒有（規則 7）
        print(f"[corun] 查不到卡／NUMA／CPU：{e} → 不平行", file=sys.stderr)
        return 2
    ok, plan = make_plan(inp, a.n)
    job = os.environ.get("SLURM_JOB_ID", "nojob")
    plan.update(job=job, node=os.environ.get("SLURMD_NODENAME", ""), slurm_job_gpus=os.environ.get("SLURM_JOB_GPUS", ""), ok=ok)
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        json.dump(plan, open(os.path.join(a.out_dir, f"corun_plan_{job}.json"), "w"), ensure_ascii=False, indent=1)
    if not ok:
        print(f"[corun] 不平行：{plan['reason']}", file=sys.stderr)
        return 2
    out = [f"SPLIT={plan['cpu_split']}"]
    for i, g in enumerate(plan["gpus"]):
        out.append(f"G{i}={g['cvd']} B{i}={g['bus']} N{i}={g['numa']} C{i}={_cpulist(g['cpus'])}")
    if a.n == 2:                                             # 偏離 20 的 next_a_w6ab.sh 用的名字
        ga, gb = plan["gpus"]
        out.append(f"GA={ga['cvd']} GB={gb['cvd']} CA={_cpulist(ga['cpus'])} CB={_cpulist(gb['cpus'])} "
                   f"NA={ga['numa']} NB={gb['numa']} BA={ga['bus']} BB={gb['bus']}")
    print(" ".join(out))
    return 0


def _windows(path):
    if not os.path.exists(path):
        return []
    return list(csv.DictReader(open(path)))


def cmd_classify(a):
    """只報告：每個 run 的時間窗對同一個 job 其他 lane 的活動時間窗算覆蓋率；c ≥ 0.9 的 lane 數＝k，
    同時跑的數目＝1＋k；有 0＜c＜0.9 的 lane → partial（同 T2 §3）。"""
    import pandas as pd
    D = a.out_dir
    cr = pd.read_csv(os.path.join(D, "corun.csv"), dtype=str)
    rows = []
    for job, lanes in cr.groupby("job"):
        act = {r.sub: (pd.Timestamp(r.t_start), pd.Timestamp(r.t_end)) for r in lanes.itertuples()}
        for r in lanes.itertuples():
            for w in _windows(os.path.join(a.res_dir, r.ovl) if a.res_dir else r.ovl):
                if w.get("job") != job:
                    continue
                t0, t1 = pd.Timestamp(w["t_start"]), pd.Timestamp(w["t_end"])
                dur = (t1 - t0).total_seconds()
                covs = {o: (max(0.0, (min(t1, e) - max(t0, s0)).total_seconds()) / dur if dur > 0 else float("nan"))
                        for o, (s0, e) in act.items() if o != r.sub}
                k = sum(c >= COVER for c in covs.values())
                partial = any(0 < c < COVER for c in covs.values())
                rows.append(dict(run_id=w["run_id"], job=job, lane=r.sub, step=w["step"], t_start=w["t_start"], t_end=w["t_end"],
                                 n_concurrent=1 + k, partial=int(partial), covers=json.dumps({o: round(c, 4) for o, c in covs.items()})))
    R = pd.DataFrame(rows)
    R.insert(1, "ts", _now())
    R.to_csv(os.path.join(D, "corun_runs.csv"), index=False)
    print(R.groupby(["lane", "n_concurrent", "partial"]).size().to_string() if len(R) else "沒有 run")
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
    smi4 = [["0", "00000000:31:00.0", "GPU-a"], ["1", "00000000:4B:00.0", "GPU-c"], ["2", "00000000:CA:00.0", "GPU-d"], ["3", "00000000:E5:00.0", "GPU-b"]]
    numa4 = {"00000000:31:00.0": 0, "00000000:4B:00.0": 0, "00000000:CA:00.0": 1, "00000000:E5:00.0": 1}
    c32 = list(range(32)) + list(range(64, 96))
    cases = [(nm, inp, ok, sp, 2) for nm, inp, ok, sp in cases] + [
        ("4 張：兩個 NUMA 各 2、CPU 兩邊各 32", dict(cvd="0,1,2,3", smi=smi4, numa=numa4, cpus=c32, cpu_node=node), True, "local", 4),
        ("4 張：CPU 全在 NUMA 0", dict(cvd="0,1,2,3", smi=smi4, numa=numa4, cpus=list(range(64)), cpu_node=node), True, "chunks", 4),
        ("4 張：3＋1", dict(cvd="0,1,2,3", smi=smi4, numa=dict(numa4, **{"00000000:CA:00.0": 0}), cpus=c32, cpu_node=node), False, None, 4),
        ("4 張：全在 NUMA 0", dict(cvd="0,1,2,3", smi=smi4, numa={k: 0 for k in numa4}, cpus=c32, cpu_node=node), False, None, 4),
        ("要 4 張只有 2 張", dict(cvd="0,1", smi=smi, numa=numa, cpus=c32, cpu_node=node), False, None, 4),
    ]
    bad = 0
    for name, inp, want_ok, want_split, n in cases:
        ok, p = make_plan(inp, n)
        good = ok == want_ok and (not ok or p["cpu_split"] == want_split)
        if ok and good:
            gs = p["gpus"]
            allc = [c for g in gs for c in g["cpus"]]
            good = (len(allc) == len(set(allc)) == 16 * n and all(len(g["cpus"]) == 16 for g in gs)
                    and len({g["numa"] for g in gs}) == 2 and [g["numa"] for g in gs] == sorted(g["numa"] for g in gs))
            if want_split == "local":
                good = good and all(g["numa_local"] for g in gs)
        print(("PASS" if good else "FAIL"), f"n={n}", name, "→", (p.get("cpu_split") if ok else p.get("reason")))
        bad += not good
    print("selftest", "PASS" if bad == 0 else f"FAIL（{bad}）")
    return 0 if bad == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("plan"); s.add_argument("--out-dir"); s.add_argument("--fake"); s.add_argument("--n", type=int, default=2)
    s = sp.add_parser("classify"); s.add_argument("--out-dir", required=True); s.add_argument("--res-dir", help="corun.csv 的 ovl 是相對路徑時的根（repo）")
    sp.add_parser("selftest")
    a = ap.parse_args()
    sys.exit({"plan": cmd_plan, "classify": cmd_classify, "selftest": cmd_selftest}[a.cmd](a))


if __name__ == "__main__":
    main()
