"""next_sim_dev14_list.py — icpe-sim：PREREG 偏離 14（＋SIM_CHANGES 23）的重跑清單，逐 block 的 job 數，給 icpe-data 分機器。

  python -I code/icpe/next_sim_dev14_list.py --snap <icpe-next-data 的快照目錄> --done <next_b_queue_done.jsonl> \
         --out <csv> --run-id <RUN_ID> --data-sha <icpe-next-data 的 commit>

- 計畫＝快照裡的 code/icpe/next_b_queue.py 的 blocks()（含 P4 的格子清單與 *_ttl7 的 cells.csv，都從快照讀）；
  job 的展開用本 worktree 的 code/p3x/x1_sim.py（jobs_for／gpu_specs／noise_specs；先 import，next_b_queue 會用到同一份）。
- 每個 job＝(trace, rule, cpu, policy, views, gpu spec, noise spec, slow spec)，和 sweep 開跑時印的 jobs= 同一套展開。
- 要重跑的 job（旗標全開時結果可能變的；其他 job 已驗證逐位元不變，見 STATUS_SIM）：
  (i)  --pf-expire：CPU 規則有 opf 預取的（POLICIES[p][3]＝"opf"：O-PF、O-ALL、O-ALL-H1800、O-ALL-H3600）；
  (a)  --noise-overdue resample：雜訊模型 P、λ＞0 的 job（全部；它們都用到完美預知，見 jobs_for 的 uses_oracle）；
  (23) --pf-gpu-recheck：只改 O-ALL＋預取到 GPU（已包含在 (i) 裡）。
- CacheWise 的 block 列出但標 excluded（PREREG 偏離 15：不跑新的 CacheWise 工作）。
- 只寫彙總（每個 block 一列），不讀紀錄、不跑模擬。"""
from __future__ import annotations

import argparse
import csv
import datetime
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "code", "p3x"))
sys.path.insert(0, os.path.join(REPO, "code"))
import x1_sim as XS  # noqa: E402  先 import 本 worktree 的；next_b_queue.import_xs() 會拿到同一個模組

FLAGS = "--pf-expire --pf-gpu-recheck"
FLAGS_P = "--pf-expire --pf-gpu-recheck --noise-overdue resample"


def load_queue(snap):
    p = os.path.join(snap, "code", "icpe", "next_b_queue.py")
    spec = importlib.util.spec_from_file_location("next_b_queue_snap", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(p)))) == os.path.abspath(snap)
    return m


def expand(Q, tr, args):
    """同 next_b_queue.cmd_count。"""
    d = Q.parse_sweep(args)
    for nm in d["slows"]:
        if nm not in XS.SLOWS and nm not in XS.EXTRA_SLOWS:
            XS.EXTRA_SLOWS.append(nm)
    gs = XS.gpu_specs([float(x) for x in d["gpu-gib"]], d["gpu-write"], d["gpu-policies"], d["pf-target"])
    ns = XS.noise_specs(d["noise-models"], [float(x) for x in d["noise-lambdas"]], d["noise-draws"],
                        [int(x) for x in d["noise-seeds"]]) if d.get("noise-models") else None
    J = XS.jobs_for(tr["traces"], tr["rules"], [float(x) for x in d["cpu"]], d["policies"], d["modes"],
                    slows=d["slows"], gspecs=gs, nspecs=ns)
    if tr.get("jobs"):
        keep = {(t, r, float(c), p, g, s, gp) for t, r, c, p, g, s, gp in tr["jobs"]}
        J = [j for j in J if any((j[0], j[1], float(j[2]), j[3], j[5]["pol"] if j[5] else "none", v[0], v[1]) in keep
                                 for v in j[4])]
    return d, J


def affected(j):
    p, nz = j[3], j[6]
    why = []
    if XS.POLICIES[p][3] == "opf":
        why.append("i")
    if nz is not None and nz["model"] == "P" and nz["lam"] > 0:
        why.append("a")
    return why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", required=True)
    ap.add_argument("--done", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--data-sha", required=True)
    a = ap.parse_args()
    Q = load_queue(a.snap)
    Q.check_p1(XS)
    done = {}
    for ln in open(a.done):
        r = json.loads(ln)
        done[(r["plan"], r["block"])] = r
    ts = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    rows = []
    for plan, bl in Q.blocks().items():
        for name, tr, args in bl:
            d, J = expand(Q, tr, args)
            A = [j for j in J if affected(j)]
            pols = sorted({j[3] for j in A}, key=d["policies"].index)
            cw = any("cachewise" in t for t in tr["traces"])
            if cw:
                action = "excluded_dev15"
            elif plan == "xcheck":
                action = "reference_only"      # 跨機器對拍（旗標關），不是正式結果；旗標開的對拍另見 xref_dev14
            elif not A:
                action = "no_rerun"
            else:
                action = "rerun"
            nzP = any(j[6] is not None and j[6]["model"] == "P" for j in A)
            dn = done.get((plan, name))
            rows.append(dict(
                run_id=a.run_id, ts=ts, host="acvlab", data_sha=a.data_sha, x1_sim_sha256=XS_SHA,
                plan=plan, block=name, traces=" ".join(tr["traces"]), rules=" ".join(tr["rules"]),
                jobs_block=len(J), jobs_rerun=len(A),
                jobs_rerun_i=sum("i" in affected(j) for j in A), jobs_rerun_a=sum("a" in affected(j) for j in A),
                rerun_policies=" ".join(pols) if action == "rerun" else "",
                pf_targets=" ".join(sorted({(j[5] or {}).get("pf", "cpu") for j in A})) if action == "rerun" else "",
                action=action, flags=(FLAGS_P if nzP else FLAGS) if action == "rerun" else "",
                acvlab_done_run_id=dn["run_id"] if dn else "", acvlab_done_elapsed_s=dn["elapsed_s"] if dn else ""))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    agg = {}
    for r in rows:
        k = (r["plan"], r["action"])
        x = agg.setdefault(k, [0, 0, 0])
        x[0] += 1
        x[1] += r["jobs_block"]
        x[2] += r["jobs_rerun"]
    for (plan, act), (nb, jb, jr) in agg.items():
        print(f"{plan:16s} {act:15s} blocks={nb:4d} jobs_block={jb:8d} jobs_rerun={jr:8d}")
    print("TOTAL rerun jobs", sum(r["jobs_rerun"] for r in rows if r["action"] == "rerun"))
    return 0


def _sha(p):
    import hashlib
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


XS_SHA = _sha(XS.__file__)

if __name__ == "__main__":
    sys.exit(main())
