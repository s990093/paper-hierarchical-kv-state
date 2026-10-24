"""h200_mps.py — 13 號 Q4 的 P4（留運算單元給還原）在 H200 的替代版：CUDA MPS（13_h200_repro §14；只在 H200，只報告）。

MI300X 的 P4a／P4b 用 ROCm 的 HSA_CU_MASK 把背景 matmul 限在 7/8、3/4 的 CU；CUDA 沒有這個，改用 MPS：
背景 process 設 CUDA_MPS_ACTIVE_THREAD_PERCENTAGE（同樣 7/8、3/4 的 SM，取整數百分比），還原的 process 不限。
MPS 本身就改變兩個 process 共用 GPU 的方式（空間分享，不是時間片輪流），所以另量一個「MPS 底下、不限」的 P0mps 當基準；
三欄（P0mps、P4a_mps、P4b_mps）只和彼此比，不和 MI300X 的 P4a／P4b、也不和 H200 沒開 MPS 的 P0–P5 混成同一欄。

  python code/h200_mps.py check --out J     # GPU，MPS daemon 要先開：背景單獨跑的 matmul 時間（不限／p4a／p4b）→ 限制有沒有生效
  python code/h200_mps.py report            # CPU：results/m10_restore2_h200/q4mps/ 的表（並列沒開 MPS 的 P0、P3 當參考）
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.dirname(HERE)
R2 = os.environ.get("M10_RESTORE2_DIR") or os.path.join(REPO, "results", "m10_restore2_h200")
METHODS = ("P0mps", "P4a_mps", "P4b_mps")
REFS = ("P0", "P3")                                   # 43b 沒開 MPS 的格子（同一台、同一個 duty）


def pcts(n_sm: int) -> dict:
    """和 m10_restore2 cumask 的遮罩同一個比例（7/8、3/4 個運算單元），換成百分比（MPS 只收整數）。"""
    return {"p4a": round(100 * (n_sm * 7 // 8) / n_sm), "p4b": round(100 * (n_sm * 3 // 4) / n_sm)}


def _ready_ms(pct, duty, reps=3):
    out = []
    for _ in range(reps):
        env = dict(os.environ)
        env.pop("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", None)
        if pct is not None:
            env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(pct)
        p = subprocess.Popen([sys.executable, os.path.join(HERE, "m7_busy.py"), "--duty", str(duty)],
                             stdout=subprocess.PIPE, text=True, env=env)
        line = p.stdout.readline()
        p.terminate()
        p.wait(30)
        if not line.startswith("READY"):
            raise SystemExit(f"m7_busy 沒有啟動：{line!r}")
        out.append(float(line.split("matmul_ms=")[1].split()[0]))
    return statistics.median(out), out


def cmd_check(a):
    servers = subprocess.run(["bash", "-c", "echo get_server_list | nvidia-cuda-mps-control"],
                             capture_output=True, text=True, timeout=60)
    import torch
    n_sm = torch.cuda.get_device_properties(0).multi_processor_count
    del torch
    pc = pcts(n_sm)
    res = {"n_sm": n_sm, "pct": pc, "duty": a.duty, "matmul_ms": {},
           "mps_pipe": os.environ.get("CUDA_MPS_PIPE_DIRECTORY", ""),
           "rule": "限制生效＝背景單獨跑的 matmul 時間 ≥ 不限的 ×1.05（p4a）／×1.10（p4b）（同 m10_restore2 cumask）"}
    for k, pct in (("none", None), ("p4a", pc["p4a"]), ("p4b", pc["p4b"])):
        res["matmul_ms"][k] = _ready_ms(pct, a.duty)
    after = subprocess.run(["bash", "-c", "echo get_server_list | nvidia-cuda-mps-control"],
                           capture_output=True, text=True, timeout=60)
    res["mps_server_list"] = {"before": servers.stdout.split(), "after": after.stdout.split(), "rc": after.returncode}
    base = res["matmul_ms"]["none"][0]
    res["effective"] = {"p4a": res["matmul_ms"]["p4a"][0] >= 1.05 * base, "p4b": res["matmul_ms"]["p4b"][0] >= 1.10 * base}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))
    if not res["mps_server_list"]["after"]:
        raise SystemExit("MPS server 沒有起來（get_server_list 是空的）→ P4-MPS 不跑")
    if not all(res["effective"].values()):
        raise SystemExit("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE 沒有生效 → P4-MPS 不跑")


def cmd_report(a):
    import m10_restore2 as RS
    md = ["# P4 的 MPS 版（H200 才有；13_h200_repro §14；只報告）", "",
          "真實 CPU 層、32K；還原時間中位數（s）。P0mps／P4a_mps／P4b_mps 都在 CUDA MPS 底下量（背景 duty 同 H200 主線 busy2）；",
          "P4a_mps、P4b_mps＝背景只能用 7/8、3/4 的 SM（CUDA_MPS_ACTIVE_THREAD_PERCENTAGE）。背景吞吐＝÷ 同格 P0mps。",
          "P0、P3＝43b 沒開 MPS（時間片輪流）的格子，只當參考，不和 MPS 的三欄比判定。", ""]
    out = {}
    mj = os.path.join(R2, "q4mps", "mps.json")
    if os.path.exists(mj):
        j = json.load(open(mj))
        md += [f"MPS 限制檢查（`q4mps/mps.json`）：SM {j['n_sm']}、百分比 {j['pct']}、背景單獨 matmul ms "
               f"{ {k: v[0] for k, v in j['matmul_ms'].items()} } → 生效 {j['effective']}", ""]
    for m in RS.MODELS:
        for tag in RS.TAGS:
            md += [f"## {m}・{RS.fmt_of(m, tag)}", "",
                   "| 方法 | 閒 E0／E1 cake | 閒 E0／E1 loadall | 忙 E0／E1 cake | 忙 E0／E1 loadall | 背景吞吐 | 可用 | run |",
                   "|:--|:--|:--|:--|:--|--:|:--|:--|"]
            r0 = RS._bg_rate(os.path.join(R2, "q4mps", f"{m}_{tag}_P0mps"))
            for meth in METHODS + REFS:
                d = os.path.join(R2, "q4mps" if meth in METHODS else "q4", f"{m}_{tag}_{meth}")
                rows = RS.latest_rows(d)
                if not rows:
                    md.append(f"| {meth} | NOT_MEASURED |  |  |  |  |  |  |")
                    continue
                c = {(mo, st, cf): RS.med(rows, mode=mo, gpu_state=st, config=cf) for mo in ("cake", "loadall")
                     for st in ("idle", "busy") for cf in ("E0", "E1")}
                rr = RS._bg_rate(d)
                ratio = rr / r0 if (rr and r0 and meth in METHODS) else None
                ok = RS.usable(d, rows)
                out[f"{m}|{tag}|{meth}"] = dict(med={"|".join(k): v for k, v in c.items()}, bg_ratio_vs_P0mps=ratio,
                                                usable=ok, run_id=rows[0]["run_id"])
                f = RS._f
                md.append(f"| {meth}{'（參考，沒開 MPS）' if meth in REFS else ''} | {f(c['cake', 'idle', 'E0'])}／{f(c['cake', 'idle', 'E1'])} | "
                          f"{f(c['loadall', 'idle', 'E0'])}／{f(c['loadall', 'idle', 'E1'])} | {f(c['cake', 'busy', 'E0'])}／"
                          f"{f(c['cake', 'busy', 'E1'])} | {f(c['loadall', 'busy', 'E0'])}／{f(c['loadall', 'busy', 'E1'])} | "
                          f"{f(ratio, 2) if ratio else '—'} | {'是' if ok else '**否**'} | `{rows[0]['run_id']}` |")
            md.append("")
    os.makedirs(os.path.join(R2, "q4mps"), exist_ok=True)
    json.dump(out, open(os.path.join(R2, "q4mps", "report_q4mps.json"), "w"), indent=1, ensure_ascii=False)
    open(os.path.join(R2, "q4mps", "tables_q4mps.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--out", required=True)
    c.add_argument("--duty", type=float, default=1.0)
    sub.add_parser("report")
    a = ap.parse_args()
    {"check": cmd_check, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
