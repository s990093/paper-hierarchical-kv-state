"""x3_restore.py — X3「需要的提前量」：一個 session 從慢層還原要多久、預取 k 個尾端 block 之後剩多久。

模型（docs/phase3_20261017/explore/X3_short_horizon_prefetch.md §2.1）：
  - 還原＝code/m7_sim.py 的 sim_restore（Cake 兩條線：重算從 block 0 往後、載入從 n−1 往前；不改它，只 import）。
  - f(i)＝第二階段 MI300X 實測（results/m10_precision_mi300x/speed_<model>/bw.json 的 f_ms_median_by_chunk，64 個 block、GPU 閒）；
    i≥64 用 block 16–63 的直線外插〔外插〕。GPU 忙＝f×2.07（results/m7_write_policy_mi300x/busy_levels.json，main.tex §3.4）。
  - CPU 層：tier_params.json 的 35.4159 GiB/s（main.tex §2.2 的 1.76 ms/block）。慢層：限速層，每個 block 的時間＝位元組÷頻寬。
  - INT4（int4_kivi）：每個 block 的位元組數與 GPU 反量化時間取自 results/m10_sweep_mi300x/parts/parts_<model>.json（實測）；
    載入一個 INT4 block＝INT4 位元組÷頻寬＋反量化（主模型 'm7'）。敏感度 'plus_h2d'：再加一次 位元組÷CPU 頻寬 的 H2D。
    兩者都用 m7_restore_harness.Tier 表示：read_Bps 縮放成「CB÷read_Bps＝實際位元組÷頻寬」，c_s＝每個 block 的固定額外時間。
  - 預取：從慢層、以該層讀頻寬、從最後一個 block 往前搬進 CPU（存原精度）。k 個在 CPU、n−k 個在慢層時跑一次 sim_restore＝T(n,k)。

子命令：
  validate  用第二階段實測的 E0（全 BF16）／E1（全 INT4）Cake 還原中位數驗證這個模型（summary_int4.csv），輸出 x3_restore_validate.csv
  tables    每個 (模型, 層, 精度, GPU) 算 T(n,k)，n＝1..NMAX、k＝0..n，存 npz（run 目錄）＋ needed-lead 摘要 CSV
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from multiprocessing import Pool

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
REPO = os.path.dirname(CODE)
sys.path.insert(0, CODE)

import numpy as np  # noqa: E402
from m7_restore_harness import GiB, Tier  # noqa: E402
from m7_sim import CB, sim_restore  # noqa: E402

MiB = 1 << 20
import p3_platform as PF  # noqa: E402  （P3_CONST：沒設＝MI300X；h200＝results/*_h200，限速層兩台都用 MI300X 的）
TIER_PARAMS = PF.TIER_PARAMS
BUSY_LEVELS = PF.rdir("m7_write_policy_mi300x", "busy_levels.json")
SPEED = PF.rdir("m10_precision_mi300x", "speed_{m}")
PARTS = PF.rdir("m10_sweep_mi300x", "parts", "parts_{m}.json")
OUT = PF.explore_out("x3")
MODELS = ("llama", "qwen")
PRECS = ("bf16", "int4")
GPUS = ("idle", "busy")
NMAX = 256          # 128K token；trace 裡更長的 context 截在這裡（hide 會記錄截掉幾個）
BAND = (0.33, 4.0)  # README X3：Cake 帶內 0.33–4 GiB/s（main.tex §4.1）


def ts():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def tiers_gibps() -> dict:
    """慢層清單：名稱 → 讀頻寬（GiB/s）。b* 值是各模型在 32K 校準的限速值（bw.json），這裡當成一般的層，兩個模型都跑。"""
    tp = json.load(open(TIER_PARAMS))
    d = {"nfs": tp["nfs"]["read_GiBps"], "local": tp["local"]["read_GiBps"]}
    for m in MODELS:
        bw = json.load(open(os.path.join(SPEED.format(m=m), "bw.json")))
        for k, v in bw["targets"].items():
            d[f"{m}_{k}"] = v["GiBps"]
    for g in (0.5, 1.0, 2.0, 4.0):
        d[f"bw{g:g}"] = g
    return dict(sorted(d.items(), key=lambda kv: kv[1]))


def in_band(g: float) -> bool:
    return BAND[0] <= g <= BAND[1]


def consts(model: str) -> dict:
    bw = json.load(open(os.path.join(SPEED.format(m=model), "bw.json")))
    parts = json.load(open(PARTS.format(m=model)))
    tp = json.load(open(TIER_PARAMS))
    busy = json.load(open(BUSY_LEVELS))["_measured_slowdown"]["busy2"]
    f = [x / 1e3 for x in bw["f_ms_median_by_chunk"]]
    assert len(f) == bw["n_blocks"] == 64, (model, len(f))
    bb = parts["chunk_bytes_bf16"]
    assert bb == bw["chunk_bytes"], (bb, bw["chunk_bytes"])   # 規則 6：兩個來源的 block 位元組數要一致
    mp = parts["median_per_block"]
    i4 = mp["int4_kivi|idle"]
    assert abs(i4["bytes"] / bb - 5.0 / 16.0) < 1e-6, i4      # INT4 KIVI＝每值 5 bit（含 scale）→ 位元組＝BF16 的 5/16
    # 外插：block 16–63 的直線
    xs = np.arange(16, 64)
    a, b = np.polyfit(xs, np.array(f[16:64]), 1)
    fext = f + [float(a * i + b) for i in range(64, NMAX + 1)]
    return dict(model=model, f_idle=fext, f_slope_ms=a * 1e3, f_icpt_ms=b * 1e3, busy=busy,
                bytes={"bf16": bb, "int4": float(i4["bytes"])},
                dq={("int4", "idle"): mp["int4_kivi|idle"]["t_dequant_torch_ms"] / 1e3,
                    ("int4", "busy"): mp["int4_kivi|busy"]["t_dequant_torch_ms"] / 1e3,
                    ("bf16", "idle"): 0.0, ("bf16", "busy"): 0.0},
                cpu_Bps=tp["cpu"]["read_GiBps"] * GiB, f_run=bw["run_id"], parts_run=parts["run_id"],
                n_meas=64)


def mk(name: str, gibps: float, nbytes: float, extra_s: float) -> Tier:
    """CB÷read_Bps＝nbytes÷頻寬；c_s＝每個 block 的額外時間（反量化／H2D）。io 模型 fifo（單一請求，沒有寫入）。"""
    return Tier(name, read_Bps=gibps * GiB * CB / nbytes, write_Bps=gibps * GiB, c_s=extra_s, mode="fifo")


def make_tiers(C: dict, gibps: float, prec: str, gpu: str, variant: str = "m7"):
    nb = C["bytes"][prec]
    dq = C["dq"][(prec, gpu)]
    h2d = nb / C["cpu_Bps"] if variant == "plus_h2d" else 0.0
    slow = mk("slow", gibps, nb, dq + h2d)
    cpu = Tier("cpu", read_Bps=C["cpu_Bps"] * CB / nb, write_Bps=C["cpu_Bps"], c_s=dq, mode="fifo")
    return slow, cpu


def f_of(C: dict, gpu: str):
    s = C["busy"] if gpu == "busy" else 1.0
    return [x * s for x in C["f_idle"]]


def restore(n: int, k: int, f, slow: Tier, cpu: Tier):
    """k 個尾端 block 在 CPU、前 n−k 個在慢層 → (還原時間 s, 會合點＝重算的 block 數, 從 CPU 載入數, 從慢層載入數)。"""
    slow.reset_clock()
    cpu.reset_clock()
    loc = [slow] * (n - k) + [cpu] * k
    t, meet, nc, ns = sim_restore(n, loc, f, 0.0, "cake")
    return t, meet, nc, ns


# ---------------------------------------------------------------- validate
def cmd_validate(a):
    rows = []
    for m in MODELS:
        C = consts(m)
        bw = json.load(open(os.path.join(SPEED.format(m=m), "bw.json")))
        path = os.path.join(SPEED.format(m=m), "summary_int4.csv")
        for r in csv.DictReader(open(path)):
            if r["mode"] != "cake" or r["config"] not in ("E0", "E1"):
                continue
            n = int(r["length"]) // 512
            prec = "bf16" if r["config"] == "E0" else "int4"
            gpu = r["gpu_state"]
            for variant in ("m7", "plus_h2d"):
                if r["bw_name"] == "cpu":
                    slow, cpu = make_tiers(C, 1.0, prec, gpu, variant)
                    t, meet, _, _ = restore(n, n, f_of(C, gpu), slow, cpu)
                    g = C["cpu_Bps"] / GiB
                else:
                    g = bw["targets"][r["bw_name"]]["GiBps"]
                    slow, cpu = make_tiers(C, g, prec, gpu, variant)
                    t, meet, _, _ = restore(n, 0, f_of(C, gpu), slow, cpu)
                meas = float(r["median_t_restore"])
                bvals = [int(x) for x in r["b_values"].split()]
                rows.append(dict(model=m, length=int(r["length"]), n_blocks=n, tier=r["bw_name"], GiBps=round(g, 4),
                                 gpu=gpu, prec=prec, variant=variant, t_pred_s=round(t, 5), t_meas_s=meas,
                                 rel_err=round((t - meas) / meas, 4), b_pred=meet, b_meas=r["b_values"],
                                 b_within1=int(min(bvals) - 1 <= meet <= max(bvals) + 1),
                                 meas_run_id=r["run_ids"], f_run=C["f_run"], parts_run=C["parts_run"]))
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "x3_restore_validate.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, ["run_id", "ts"] + list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(dict(run_id=a.run_id, ts=ts(), **r))
    # 摘要
    summ = []
    for variant in ("m7", "plus_h2d"):
        for scope in ("slow", "cpu"):
            x = [r for r in rows if r["variant"] == variant and ((r["tier"] == "cpu") == (scope == "cpu"))]
            e = np.abs([r["rel_err"] for r in x])
            summ.append(dict(variant=variant, scope=scope, n_cells=len(x), median_abs_rel_err=round(float(np.median(e)), 4),
                             max_abs_rel_err=round(float(np.max(e)), 4),
                             b_within1_frac=round(float(np.mean([r["b_within1"] for r in x])), 4)))
    print(json.dumps(summ, indent=1))
    p2 = os.path.join(OUT, "x3_restore_validate_summary.csv")
    with open(p2, "w", newline="") as fh:
        w = csv.DictWriter(fh, ["run_id", "ts"] + list(summ[0].keys()))
        w.writeheader()
        for r in summ:
            w.writerow(dict(run_id=a.run_id, ts=ts(), **r))
    print("wrote", p, p2)


# ---------------------------------------------------------------- tables
def _table(spec):
    m, tier, g, prec, gpu, variant = spec
    C = consts(m)
    f = f_of(C, gpu)
    slow, cpu = make_tiers(C, g, prec, gpu, variant)
    T = np.full((NMAX + 1, NMAX + 1), np.nan)
    meet0 = np.zeros(NMAX + 1, int)
    for n in range(1, NMAX + 1):
        for k in range(0, n + 1):
            t, meet, _, _ = restore(n, k, f, slow, cpu)
            T[n, k] = t
            if k == 0:
                meet0[n] = meet
    return spec, T, meet0, C["bytes"][prec]


def all_specs(variants=("m7",)):
    tg = tiers_gibps()
    return [(m, t, g, p, gp, v) for m in MODELS for t, g in tg.items() for p in PRECS for gp in GPUS for v in variants]


def table_path(run_dir, spec):
    m, tier, g, prec, gpu, variant = spec
    return os.path.join(run_dir, "tables", f"T_{m}_{tier}_{prec}_{gpu}_{variant}.npz")


def cmd_tables(a):
    specs = all_specs()
    os.makedirs(os.path.join(a.run_dir, "tables"), exist_ok=True)
    rows = []
    with Pool(a.procs) as pool:
        for spec, T, meet0, nb in pool.imap_unordered(_table, specs):
            m, tier, g, prec, gpu, variant = spec
            np.savez_compressed(table_path(a.run_dir, spec), T=T, meet0=meet0, block_bytes=nb, GiBps=g)
            for n in (1, 2, 4, 8, 16, 32, 64, 128, 256):
                t_none, t_full, b = T[n, 0], T[n, n], int(meet0[n])
                rows.append(dict(model=m, tier=tier, GiBps=round(g, 4), in_band=int(in_band(g)), prec=prec, gpu=gpu,
                                 variant=variant, n_blocks=n, tokens=n * 512, extrapolated_f=int(n > 64),
                                 t_none_s=round(t_none, 5), t_full_s=round(t_full, 5),
                                 slow_restore_s=round(t_none - t_full, 5), meet_none=b,
                                 lead_full_s=round(n * nb / (g * GiB), 5), lead_tail_s=round((n - b) * nb / (g * GiB), 5),
                                 tail_blocks=n - b, tail_GiB=round((n - b) * nb / GiB, 4), full_GiB=round(n * nb / GiB, 4),
                                 t_tail_s=round(T[n, n - b], 5),
                                 h_tail=round((t_none - T[n, n - b]) / (t_none - t_full), 4) if t_none > t_full else 1.0))
            print("done", spec, flush=True)
    rows.sort(key=lambda r: (r["model"], r["GiBps"], r["prec"], r["gpu"], r["n_blocks"]))
    p = os.path.join(OUT, "x3_needed_lead.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, ["run_id", "ts"] + list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(dict(run_id=a.run_id, ts=ts(), **r))
    json.dump(dict(run_id=a.run_id, specs=[list(s) for s in specs], nmax=NMAX, tiers=tiers_gibps(),
                   consts={m: {k: (v if k not in ("f_idle", "dq", "bytes") else None) for k, v in consts(m).items()}
                           for m in MODELS}),
              open(os.path.join(a.run_dir, "tables", "index.json"), "w"), indent=1, default=str)
    print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "tables", "consts"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "validate":
        cmd_validate(a)
    elif a.cmd == "tables":
        cmd_tables(a)
    else:
        for m in MODELS:
            C = consts(m)
            print(m, {k: v for k, v in C.items() if k != "f_idle"}, "f[0]", C["f_idle"][0], "f[63]", C["f_idle"][63],
                  "f[255]", C["f_idle"][255])
        print(tiers_gibps())


if __name__ == "__main__":
    main()
