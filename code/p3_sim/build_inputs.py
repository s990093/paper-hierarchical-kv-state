"""build_inputs.py — 從 repo 裡的實測檔產生模擬器的輸入常數（只讀，不量任何東西）。

產出（都寫在 code/p3_sim/config/，每個檔裡都記 run_id 與來源檔／來源 run）：
  costs_mi300x.json   MI300X 的 f(i)、GPU 忙倍數、CPU 層、慢層（tier_params／bw.json）
  costs_3090.json     3090 的 f(i)、GPU 忙倍數、CPU 層（X7，results/p3_explore/x7/constants_3090.json）
  format_table.PLACEHOLDER.json
                      **明確標 placeholder** 的格式表：大小與解壓時間是照 X1／X5 的規則從第二階段 restores 抽的
                      （附 run_id），品質一律 unverified（bf16 是參考＝pass）。正式的格式表由另一個 agent 做
                      （results/p3_formats/format_table.json）；測試不准依賴這份 placeholder 的數值。

規則（照抄既有程式，不改它們）：
  * MI300X Llama 的 f(i)：x1_sim.constants()（m8_trace_oracle.load_f：calib_c1.csv 的 C1 中位數，i ≥ 80 線性外插）。
  * MI300X Qwen 的 f(i)：results/m10_precision_mi300x/speed_qwen/bw.json 的 f_ms_median_by_chunk（x5_costs 的規則），64 個以上線性外插。
  * GPU 忙（MI300X）：busy_levels.json 的 _measured_slowdown.busy2（2.07；X1／X8 用的）。Qwen 另記 x5 規則的值（只報告）。
  * 3090：constants_3090.json（X7）；Llama 的 CPU 層＝ell_cpu_ms（真實 H2D），寫＝c0 的 D2H 中位數，k＝duplex。
  * 解壓（每 block）：X1 的規則＝E1、load_only、bw_name=cpu、fp8_read>0 的 median(t_dequant / fp8_read)，依 GPU 狀態。
  * 每 block 位元組：E1 load_only 的 median(bytes_stored / n_blocks)（X5 的規則）。
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
REPO = os.path.dirname(CODE)
CFG = os.path.join(HERE, "config")
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "p3x"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

R10 = os.path.join(REPO, "results", "m10_precision_mi300x")
RES7 = os.path.join(REPO, "results", "m7_write_policy_mi300x")
NOT = "NOT_MEASURED"
GRID = {   # (model, 格式) → MI300X restores 的 run（X1／X5 用的同一批）
    ("llama", "int4"): "20261016-034902-m10-grid-int4-llama",
    ("llama", "8bit"): "20261015-161226-m10-grid-fp8-llama",
    ("qwen", "int4"): "20261016-052143-m10-grid-int4-qwen",
    ("qwen", "8bit"): "20261015-193516-m10-grid-2b-int8_token-qwen",   # Qwen 的 8-bit＝INT8（FP8 掉 32.8 分，04 §1）
}
CONCRETE = {("llama", "8bit"): "fp8_token", ("qwen", "8bit"): "int8_token", ("llama", "int4"): "int4_kivi",
            ("qwen", "int4"): "int4_kivi", ("llama", "int2"): "int2_kivi", ("qwen", "int2"): "int2_kivi"}


def rid():
    return os.environ.get("RUN_ID", "adhoc")


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def extrap(f_ms):
    x = np.arange(len(f_ms), dtype=float)
    slope, icpt = np.polyfit(x, np.array(f_ms, float), 1)
    return float(slope), float(icpt)


def costs_mi300x():
    import x1_sim
    K = x1_sim.constants()
    tp = json.load(open(os.path.join(RES7, "tier_params.json")))
    busy = json.load(open(os.path.join(RES7, "busy_levels.json")))
    bwq = json.load(open(os.path.join(R10, "speed_qwen", "bw.json")))
    bwl = json.load(open(os.path.join(R10, "speed_llama", "bw.json")))
    fq = [float(x) for x in bwq["f_ms_median_by_chunk"]]
    sq, iq = extrap(fq)
    # Qwen 的 GPU 忙倍數（x5_costs 的規則：同一個 INT4 grid 的 compute_only 32K 忙÷閒），只報告
    d = pd.read_csv(os.path.join(R10, "speed_qwen", "restores.csv.gz"))
    g = d[(d.run_id == GRID[("qwen", "int4")]) & (d.length == 32768) & (d["mode"] == "compute_only") & (d.bw_name == "cpu")]
    co = g.groupby("gpu_state").t_restore.median()
    cpu = dict(read_GiBps=K["cpu"]["r"], write_GiBps=K["cpu"]["w"], k_read=K["cpu"]["k"],
               src="results/m7_write_policy_mi300x/tier_params.json cpu（runs " + ",".join(tp["source_runs"]) + "）")
    return dict(
        platform="mi300x", run_id=rid(), ts=now(),
        note="〔MI300X 常數〕全部是 repo 裡第一、二階段的實測檔，見各欄 src。",
        models=dict(
            llama=dict(f_ms=K["f_ms"], f_slope_ms=K["f_slope"], f_icpt_ms=K["f_icpt"],
                       f_src="results/m7_write_policy_mi300x/calib_c1.csv（m8_trace_oracle.load_f；runs " + ",".join(K["f_runs"]) + "），i≥80 線性外插",
                       busy_factor=K["busy_factor"], busy_src="results/m7_write_policy_mi300x/busy_levels.json _measured_slowdown.busy2（runs " + ",".join(busy["_source_runs"]) + "）",
                       cpu=cpu),
            qwen=dict(f_ms=fq, f_slope_ms=sq, f_icpt_ms=iq,
                      f_src=f"results/m10_precision_mi300x/speed_qwen/bw.json f_ms_median_by_chunk（run {bwq['run_id']}），i≥{len(fq)} 線性外插",
                      busy_factor=float(busy["_measured_slowdown"]["busy2"]),
                      busy_src="busy_levels.json busy2（Llama 量的；Qwen 自己的 x5 規則值見 busy_factor_x5_rule，只報告）",
                      busy_factor_x5_rule=float(co["busy"] / co["idle"]),
                      cpu=cpu),
            llama70b=dict(f_ms=NOT, busy_factor=NOT, cpu=cpu,
                          f_src="NOT_MEASURED：70B 級沒有 f(i)。只算頻寬（04 §5），還原一律 load-only、TTFT 不算"),
        ),
        ssd_profiles=dict(
            mi300x_local=dict(read_GiBps=tp["local"]["read_GiBps"], write_GiBps=tp["local"]["write_GiBps"], k_read=tp["local"]["k_read"],
                              src="tier_params.json local（04 §5 主地圖：讀 6.98、寫 1.91 GiB/s）"),
            nfs=dict(read_GiBps=tp["nfs"]["read_GiBps"], write_GiBps=tp["nfs"]["write_GiBps"], k_read=tp["nfs"]["k_read"],
                     src="tier_params.json nfs"),
            b32=dict(read_GiBps=float(bwl["targets"]["b32"]["GiBps"]), write_GiBps=float(bwl["targets"]["b32"]["GiBps"]), k_read=1.0,
                     src=f"speed_llama/bw.json b32（run {bwl['run_id']}）；寫＝讀、k=1 是 X1 的假設"),
            ssd8_3090=dict(read_GiBps=NOT, write_GiBps=NOT, k_read=NOT, src="NOT_MEASURED：3090 /ssd8 要新量（04 §7-3）"),
        ),
        cpu_busy_h2d_factor=NOT,
        cpu_busy_note="CPU 忙時 H2D 變慢的倍數：由 3090 量（04 §5）；沒量到就 NOT_MEASURED，模擬器拒跑 cpu=busy",
    )


def costs_3090():
    c = json.load(open(os.path.join(REPO, "results", "p3_explore", "x7", "constants_3090.json")))
    tp = json.load(open(os.path.join(RES7, "tier_params.json")))
    out = dict(platform="3090", run_id=rid(), ts=now(),
               note="〔3090 常數〕results/p3_explore/x7/constants_3090.json（X7，summarize run " + c["summarize_run_id"] + "）",
               models={}, ssd_profiles={}, cpu_busy_h2d_factor=NOT,
               cpu_busy_note="CPU 忙時 H2D 變慢的倍數：3090 還沒量（04 §7-3）→ NOT_MEASURED")
    wr = c["cpu_tier_d2h_llama_chunk_GiBps_c0"]["median_of_run_medians"]
    kd = c["cpu_tier_k_read_duplex"]["per_run"]
    k = float(np.median(list(kd.values())))
    for m in ("llama", "qwen"):
        mm = c["models"][m]
        f = [float(x) for x in mm["f_ms_idle_32k"]]
        s, i = extrap(f)
        r = mm["chunk_bytes"] / (mm["ell_cpu_ms"] / 1e3) / (1 << 30)
        out["models"][m] = dict(
            f_ms=f, f_slope_ms=s, f_icpt_ms=i, f_src=f"x7 f_ms_idle_32k（run {mm['f_src_run']}），i≥{len(f)} 線性外插",
            busy_factor=float(mm["busy_0.7_slowdown"]["median_all_chunks"]), busy_src=f"x7 busy_0.7_slowdown（run {mm['busy_0.7_slowdown']['src_run']}）",
            cpu=dict(read_GiBps=float(r), write_GiBps=float(wr), k_read=k,
                     src=f"讀＝chunk_bytes÷ell_cpu_ms（x7，run {mm['cpu_tier_h2d_src_run']}）；寫＝c0 D2H 中位數（Llama chunk，runs {','.join(c['cpu_tier_d2h_llama_chunk_GiBps_c0']['per_run'])}）；k＝duplex 中位數"))
    out["models"]["llama70b"] = dict(f_ms=NOT, busy_factor=NOT, cpu=out["models"]["llama"]["cpu"],
                                     f_src="NOT_MEASURED（只算頻寬）")
    out["ssd_profiles"] = dict(
        mi300x_local=dict(read_GiBps=tp["local"]["read_GiBps"], write_GiBps=tp["local"]["write_GiBps"], k_read=tp["local"]["k_read"],
                          src="MI300X 的本地 SSD（04 §5 主地圖的 SSD 頻寬，平台常數換成 3090 時 SSD 仍用這個）"),
        ssd8_3090=dict(read_GiBps=NOT, write_GiBps=NOT, k_read=NOT, src="NOT_MEASURED：3090 /ssd8 要新量（04 §7-3）"))
    return out


def dq_rule(model, fmt):
    """X1 的解壓規則（每 block 秒）與 X5 的每 block 位元組。沒有符合的列 → NOT_MEASURED。"""
    run = GRID.get((model, fmt))
    if run is None:
        return NOT, NOT, None
    d = pd.read_csv(os.path.join(R10, f"speed_{model}", "restores.csv.gz"))
    g = d[(d.run_id == run) & (d.config == "E1") & (d["mode"] == "load_only") & (d.bw_name == "cpu") & (d.fp8_read > 0)]
    if len(g) == 0:
        return NOT, NOT, run
    dq = (g.t_dequant / g.fp8_read).groupby(g.gpu_state).median()
    nbytes = float((g.bytes_stored / g.n_blocks).median())
    return {s: float(dq[s]) for s in ("idle", "busy")}, nbytes, run


def placeholder_formats():
    c3 = json.load(open(os.path.join(REPO, "results", "p3_explore", "x7", "constants_3090.json")))
    models = {}
    for m, bpt in (("llama", 131072), ("qwen", 57344), ("llama70b", 327680)):
        blk = bpt * 512
        F = {"bf16": dict(concrete="bf16", bytes_per_block=blk, bytes_src="kv_bytes_per_token×512（架構）",
                          dequant_s_per_block={"mi300x": {"idle": 0.0, "busy": 0.0}, "3090": {"idle": 0.0, "busy": 0.0}},
                          quality="reference")}
        for fmt in ("8bit", "int4", "int2"):
            conc = CONCRETE.get((m, fmt), NOT)
            if m == "llama70b":
                F[fmt] = dict(concrete=conc, bytes_per_block=NOT, dequant_s_per_block={}, bytes_src="NOT_MEASURED（70B 沒有第二階段實測）")
                continue
            dq, nb, run = dq_rule(m, fmt)
            d3 = {}
            key = {"8bit": "fp8_head" if m == "qwen" else "fp8_token", "int4": "int4_kivi"}.get(fmt)
            mm = c3["models"][m].get("dequant_ms_per_block_in_restore_cpu")
            if isinstance(mm, dict) and key and f"{key}|cake|idle" in mm and not (m == "qwen" and fmt == "8bit"):
                d3 = {"idle": mm[f"{key}|cake|idle"] / 1e3, "busy": mm[f"{key}|cake|busy"] / 1e3}
            F[fmt] = dict(concrete=conc, bytes_per_block=nb,
                          bytes_src=(f"median(bytes_stored/n_blocks)，E1 load_only cpu（run {run}）" if run and nb != NOT else "NOT_MEASURED"),
                          dequant_s_per_block={"mi300x": dq, "3090": (d3 or NOT)},
                          dequant_src=(f"MI300X：X1 規則 median(t_dequant/fp8_read)（run {run}）；3090：x7 dequant_ms_per_block_in_restore_cpu cake（run "
                                       + ",".join(c3["models"][m].get("restore_grid_runs", [])) + "）" if run else "NOT_MEASURED"))
        models[m] = dict(kv_bytes_per_token_bf16=bpt, block_tokens=512, formats=F)
    bins = [0, 8192, 16384, 32768, 65536, 131072, 262144, 1 << 40]
    q = {}
    for m in models:
        q[m] = {"bf16": ["pass"] * (len(bins) - 1)}
        for fmt in ("8bit", "int4", "int2"):
            q[m][fmt] = ["unverified"] * (len(bins) - 1)
    return dict(
        PLACEHOLDER=True,
        WARNING="PLACEHOLDER — 不是正式格式表。品質一律 unverified（bf16＝參考）。正式表：results/p3_formats/format_table.json（另一個 agent）。"
                "測試不准依賴這份的數值；冒煙用它只是確認程式跑得動。",
        run_id=rid(), ts=now(),
        quality_rule="品質過關＝配對答對率差的 95% CI 下界 > −2 分（04 §3）；INT2 在 stage 43 之前只准給「不會再被讀」的 session（never_read_only）",
        never_read_only=["int2"],
        len_bins_tokens=bins,
        quality=q,
        models=models,
    )


def main():
    os.makedirs(CFG, exist_ok=True)
    for name, obj in (("costs_mi300x.json", costs_mi300x()), ("costs_3090.json", costs_3090()),
                      ("format_table.PLACEHOLDER.json", placeholder_formats())):
        p = os.path.join(CFG, name)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=1)
        print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
