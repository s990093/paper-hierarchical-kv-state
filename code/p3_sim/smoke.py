"""smoke.py — 小樣本冒煙（06_simulator.md §7）：每份資料一小段 × 每個外層做法 × 每個壓縮做法，確認跑得動、記帳加得起來。

**不是地圖**：一個 CPU 容量、小窗、placeholder 格式表（品質帳沒有意義）。輸出只寫 $RUN_DIR（不進 results/）。
不算、不印任何「省多少」。
用法：TIARA_NO_GPU=1 bash code/run3090.sh p3sim-smoke python code/p3_sim/smoke.py --procs 6
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from multiprocessing import Pool

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sim  # noqa: E402
import trace as TR  # noqa: E402
from formats import NotMeasured  # noqa: E402

RUN_DIR = os.environ.get("RUN_DIR", "/tmp/p3sim_smoke_adhoc")
DATA = {   # (source, variant, 起點秒, 長度秒)：每份一小段
    "freeinference": ("freeinference", "week01", 0, None),
    "bailian_coder": ("bailian", "coder", 0, 1800),
    "bailian_traceA": ("bailian", "traceA", 0, 1800),
    "wildchat": ("wildchat", "1m", 200 * 86400, 6 * 3600),
    "mooncake_strict": ("mooncake", "conv-rule-strict", 0, 1800),
    "synth": ("synth", "small-seed1", 0, 4 * 3600),
}
OUTERS = [("E1", None), ("E2", None), ("E3", None), ("E4", None), ("E5", 300), ("E5", 1800), ("E5", 3600), ("E6", None)]
POLS = ["C0", "C1", "C2", "C3", "C4", "O"]
SPECS = [("mi300x", "idle", "idle", 1, "cake", "mi300x_local"), ("mi300x", "busy", "idle", 4, "cake", "mi300x_local"),
         ("mi300x", "idle", "idle", 1, "load", "mi300x_local"), ("3090", "idle", "idle", 1, "cake", "mi300x_local")]


def job(args):
    name, cap, H = args
    src, var, a, d = DATA[name]
    out = dict(cells=[], quality=[], ids=[], errors=[])
    t0 = time.time()
    try:
        P = TR.prepare(src, var, a, d)
    except Exception as e:  # noqa: BLE001
        out["errors"].append(dict(data=name, stage="prepare", error=repr(e), tb=traceback.format_exc()))
        return out
    meta = P["meta"]
    for code, T in OUTERS:
        for pol in POLS:
            try:
                if code == "E6":
                    # E6 只在 serve=fifo1 有佇列；同一個 block 引擎另跑 E2 當基準。一張 GPU 撐不住的就照錄 GPU 忙碌比例。
                    for c2 in ("E2", "E6"):
                        r, q, ids, eng, vs = sim.run_cell_fifo1(P, c2, pol, cap, "llama", SPECS[0], H=H)
                        for x in r:
                            x["outer"] = c2 + "@fifo1"
                        out["cells"] += r
                        out["quality"] += [dict(data=name, **x) for x in q]
                        out["ids"] += [dict(data=name, **x) for x in ids]
                else:
                    r, q, ids, eng, vs = sim.run_cell(P, code, pol, cap, "llama", SPECS, T=T, H=H, check=500)
                    out["cells"] += r
                    out["quality"] += [dict(data=name, **x) for x in q]
                    out["ids"] += [dict(data=name, **x) for x in ids]
            except NotMeasured as e:
                out["errors"].append(dict(data=name, stage=f"{code}/{pol}", error="NOT_MEASURED: " + str(e), tb=""))
            except Exception as e:  # noqa: BLE001
                out["errors"].append(dict(data=name, stage=f"{code}/{pol}", error=repr(e), tb=traceback.format_exc()))
    for x in out["cells"]:
        x["data"] = name
        x["data_window"] = f"{a}+{d}"
    out["meta"] = dict(data=name, **{k: v for k, v in meta.items() if k != "checks"}, checks=str(meta["checks"]), job_s=time.time() - t0)
    return out


def extra_axes(H):
    """模型軸（Qwen、70B）、疊加 ×2、session 抽樣：各一格，只確認跑得動。"""
    out = dict(cells=[], quality=[], ids=[], errors=[])
    P = TR.prepare("bailian", "coder", 0, 1800)
    for model, specs in (("qwen", [("mi300x", "idle", "idle", 1, "cake", "mi300x_local"), ("3090", "busy", "idle", 1, "cake", "mi300x_local")]),
                         ("llama70b", [("mi300x", "idle", "idle", 1, "load", "mi300x_local")])):
        for pol in ("C0", "C1", "C2", "O"):
            try:
                r, q, ids, eng, vs = sim.run_cell(P, "E2", pol, 64, model, specs, H=H)
                for x in r:
                    x["data"] = "bailian_coder"
                out["cells"] += r
                out["ids"] += [dict(data="bailian_coder", **x) for x in ids]
            except NotMeasured as e:
                out["errors"].append(dict(data="bailian_coder", stage=f"{model}/{pol}", error="NOT_MEASURED: " + str(e), tb=""))
    for k in (2, 4):
        Pk = TR.prepare("bailian", "coder", 0, 1800, overlay=k, seed=k)
        for pol in ("C1", "O"):
            r, q, ids, eng, vs = sim.run_cell(Pk, "E2", pol, 64, "llama", SPECS[:1], H=H)
            for x in r:
                x["data"] = f"bailian_coder_x{k}(semi-synthetic)"
            out["cells"] += r
            out["ids"] += [dict(data=f"bailian_coder_x{k}", **x) for x in ids]
    Ps = TR.prepare("mooncake", "conv-rule-strict", 0, 1800, session_frac=0.25, session_seed=0)
    for c2 in ("E2", "E6"):
        r, q, ids, eng, vs = sim.run_cell_fifo1(Ps, c2, "C1", 64, "llama", SPECS[0], H=H)
        for x in r:
            x["data"] = "mooncake_strict_p0.25"
            x["outer"] = c2 + "@fifo1"
        out["cells"] += r
        out["ids"] += [dict(data="mooncake_strict_p0.25", **x) for x in ids]
    return out


def write(name, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(RUN_DIR, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run_id"] + keys)
        w.writeheader()
        for r in rows:
            w.writerow(dict(run_id=os.environ.get("RUN_ID", "adhoc"), **r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--cap", type=float, default=64)
    ap.add_argument("--H", type=float, default=1800)
    ap.add_argument("--data", nargs="*", default=list(DATA))
    a = ap.parse_args()
    os.makedirs(RUN_DIR, exist_ok=True)
    t0 = time.time()
    allc, allq, alli, alle, metas = [], [], [], [], []
    with Pool(min(a.procs, 8)) as pool:
        for o in pool.imap_unordered(job, [(n, a.cap, a.H) for n in a.data]):
            allc += o["cells"]
            allq += o["quality"]
            alli += o["ids"]
            alle += o["errors"]
            if "meta" in o:
                metas.append(o["meta"])
                print(f"[{time.time()-t0:.0f}s] {o['meta']['data']}: n_req={o['meta']['n_req']} n_sess={o['meta']['n_sess']} "
                      f"cells={len(o['cells'])} errors={len(o['errors'])} job={o['meta']['job_s']:.0f}s", flush=True)
    o = extra_axes(a.H)
    allc += o["cells"]
    alli += o["ids"]
    alle += o["errors"]
    write("smoke_cells.csv", allc)
    write("smoke_quality.csv", allq)
    write("smoke_identities.csv", alli)
    write("smoke_errors.csv", alle)
    write("smoke_data.csv", metas)
    nbad = sum(1 for x in alli if not x["ok"])
    print(f"cells={len(allc)} identities={len(alli)} identity_failures={nbad} errors={len(alle)} total={time.time()-t0:.0f}s", flush=True)
    for e in alle:
        print("ERROR", e["data"], e["stage"], e["error"], flush=True)
    sys.exit(1 if nbad else 0)


if __name__ == "__main__":
    main()
