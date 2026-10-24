"""p3_fvar.py — ICPE P0-1 (c)（docs/h200/icpe/P0-1_recompute_speed.md §3.3）：3b 的 X1、X5 只換重算線的常數重跑。

  python code/icpe/p3_fvar.py --variant V1 --calib results/icpe_h200/p01/calib --out-root results/icpe_h200/p01/sim_V1 x1 sweep --workers 14
  python code/icpe/p3_fvar.py ... x1 analyze --sweep x1_sweep.csv --traces conversation toolagent [--dead-h 3600 --out-prefix x1h3600]
  python code/icpe/p3_fvar.py ... x5 --out-dir <out-root>/x5 --raw-dir <raw> --procs 14        （＝x5_run.py 的參數）
  python code/icpe/p3_fvar.py ... x5a --out-dir <out-root>/x5 --raw-dir <raw>                   （＝x5_analyze.py 的參數）
  python code/icpe/p3_fvar.py --variant V0 ... x5s --out-dir ... --sets ...                      （＝x5_short_only.py，P1-2 B5）
  python code/icpe/p3_fvar.py --variant V0 ... x4 sweep|analyze ...                              （＝x4_sim.py，只改輸出目錄；P1-2 C1）

版本：
  V0  不換（回歸：結果應和 results/p3_explore_h200 逐值相同，run_id／ts 除外）
  V1  f(i)＝S1 勝者重量的（X1：<calib>/calib_c1.csv；X5：<calib>/speed_<m>/bw.json）；GPU 忙倍數＝<calib>/calib_c5.csv 在 duty 1.0 的 slowdown 中位數
  V2  f(i) × (vLLM 預設 32K TTFT ÷ harness 只算 32K)：Llama 1.2106/2.12、Qwen 1.0590/1.865（RUNLOG_H200 job 21409）；忙倍數不變
只換 f 與忙倍數；X5 的 CPU 層校正比值（實測÷推算）先用原本的 f 算好再換（§3.3：校正比值不動）。不改 x1_sim.py／x5_*.py：
這支在同一個 process 裡改它們的模組屬性，之後才建 Pool（fork，子行程繼承）。輸出寫到 --out-root（不碰 results/p3_explore_h200）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
REPO = os.path.dirname(CODE)
sys.path.insert(0, os.path.join(CODE, "p3x"))
sys.path.insert(0, CODE)

import numpy as np  # noqa: E402

V2_RATIO = {"llama": 1.2106 / 2.12, "qwen": 1.0590 / 1.865}


def c1_f(path):
    """同 m8_trace_oracle.load_f：f_chunk、GPU 閒、每塊中位數（ms）；回傳 (f_ms, slope, icpt)。"""
    import pandas as pd
    d = pd.read_csv(path)
    d = d[(d.item == "f_chunk") & (d.gpu_state == "idle")]
    m = d.groupby("chunk_idx").ms.median().sort_index()
    x, y = m.index.values.astype(float), m.values.astype(float)
    slope, icpt = np.polyfit(x, y, 1)
    return [float(v) for v in y], float(slope), float(icpt), sorted(d.run_id.unique())


def c5_busy(path, duty=1.0):
    import pandas as pd
    d = pd.read_csv(path)
    q = d[np.isclose(d.duty.astype(float), duty)]
    assert len(q), f"{path} 沒有 duty={duty} 的列"
    return float(q.slowdown.median()), sorted(q.run_id.unique())


def patch_x1(variant, calib, out_root, note):
    import x1_sim as XS
    XS.OUTDIR = os.path.join(out_root, "x1")
    os.makedirs(XS.OUTDIR, exist_ok=True)
    orig = XS.constants

    def constants():
        K = orig()
        if variant == "V1":
            f, s, i, runs = c1_f(os.path.join(calib, "calib_c1.csv"))
            b, bruns = c5_busy(os.path.join(calib, "calib_c5.csv"))
            K.update(f_ms=f, f_slope=s, f_icpt=i, f_runs=runs, busy_factor=b)
            note.update(x1_f_runs=runs, x1_busy_runs=bruns)
        elif variant == "V2":
            r = V2_RATIO["llama"]
            K.update(f_ms=[x * r for x in K["f_ms"]], f_slope=K["f_slope"] * r, f_icpt=K["f_icpt"] * r)
        note.update(x1_f0_ms=K["f_ms"][0], x1_f63_ms=K["f_ms"][63], x1_f_slope=K["f_slope"], x1_busy=K["busy_factor"])
        return K
    XS.constants = constants
    return XS


def patch_x5(variant, calib, note):
    import x5_costs as XC
    orig_init = XC.Costs.__init__

    def init(self, model, variant_c="reg"):
        orig_init(self, model, variant_c)
        if variant == "V0":
            return
        # 校正比值先用原本的 f 算好（之後換 f 不重算）
        for st in ("BF16", "INT4"):
            for g in XC.GPUS:
                for dq in XC.DQS:
                    try:
                        self._ratio(st, 64, g, dq)
                    except AssertionError:
                        pass
        if variant_c == "excess":
            for st in ("BF16", "INT4", "SLOW"):
                for g in XC.GPUS:
                    for sl in XC.SLOW:
                        for dq in XC.DQS:
                            self._excess(st, 64, g, sl, dq)
        if variant == "V1":
            b, _ = c5_busy(os.path.join(calib, "calib_c5.csv"))
            self.busy_factor = b
            pbw = os.path.join(calib, f"speed_{model}", "bw.json")
            if os.path.exists(pbw):
                bw = json.load(open(pbw))
                f = np.array(bw["f_ms_median_by_chunk"], dtype=float) / 1000.0
                note.setdefault("x5_bw_runs", {})[model] = bw["run_id"]
            else:                  # 這個模型在 S1 停損（最快的仍是 sdpa）→ V1 的 f＝原本的 f（§2.3）
                f = self.f_idle[:64].copy()
                note.setdefault("x5_bw_runs", {})[model] = "NO_V1_CALIB（S1 停損）→ 用 V0 的 f"
        else:
            f = self.f_idle[:64] * V2_RATIO[model]
        x = np.arange(len(f))
        slope, icpt = np.polyfit(x, f, 1)
        self.f_slope, self.f_icpt = float(slope), float(icpt)
        xi = np.arange(XC.NMAX + 1)
        self.f_idle = np.where(xi < len(f), np.r_[f, np.zeros(XC.NMAX + 1 - len(f))], icpt + slope * xi)
        self.f = dict(idle=self.f_idle, busy=self.f_idle * self.busy_factor)
        XC.Costs.restore.cache_clear()
        note.setdefault("x5_consts", {})[f"{model}_{variant_c}"] = dict(f0_ms=1000 * self.f_idle[0], f63_ms=1000 * self.f_idle[63],
                                                                      busy=self.busy_factor)
    XC.Costs.__init__ = init
    return XC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["V0", "V1", "V2"], required=True)
    ap.add_argument("--calib", default=os.path.join(REPO, "results", "icpe_h200", "p01", "calib"))
    ap.add_argument("--out-root", required=True)
    ap.add_argument("tool", choices=["x1", "x5", "x5a", "x5s", "x4"])
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    os.makedirs(a.out_root, exist_ok=True)
    note = {"variant": a.variant, "run_id": os.environ.get("RUN_ID", "adhoc"), "tool": a.tool, "argv": a.rest}
    if a.tool == "x1":
        XS = patch_x1(a.variant, a.calib, a.out_root, note)
        sys.argv = ["x1_sim.py"] + a.rest
        XS.main()
    elif a.tool == "x5":
        patch_x5(a.variant, a.calib, note)
        import x5_run
        sys.argv = ["x5_run.py"] + a.rest
        x5_run.main()
    elif a.tool == "x5a":
        import x5_analyze
        sys.argv = ["x5_analyze.py"] + a.rest
        x5_analyze.main()
    elif a.tool == "x5s":                         # P1-2 B5：只算 ≤32K（只讀 raw，不用常數）
        import x5_short_only
        sys.argv = ["x5_short_only.py"] + a.rest
        x5_short_only.main()
    else:                                        # P1-2 C1：X4（f×0.5 用它自己的 --fscale；V1／V2 不支援）
        assert a.variant == "V0", "x4 只支援 V0（P1-2 的 f×0.5 用 x4_sim 自己的 --fscale）"
        import x4_sim
        x4_sim.OUT = os.path.join(a.out_root, "x4")
        os.makedirs(x4_sim.OUT, exist_ok=True)
        sys.argv = ["x4_sim.py"] + a.rest
        x4_sim.main()
    with open(os.path.join(a.out_root, "fvar_runs.jsonl"), "a") as fh:
        fh.write(json.dumps(note, ensure_ascii=False, default=str) + "\n")


if __name__ == "__main__":
    main()
