"""make_constants_v2.py — results/p3_calib_3090/constants_3090_v2.json（第三階段 3090 校準，2026-10-18）。

不覆蓋 X7 的 results/p3_explore/x7/constants_3090.json：
  - inherited_from_x7：X7 的值原樣複製（附 X7 的 summarize_run_id 與各值自己的 src_run），這次沒有重量。
  - new_in_v2：這次（p3cal）量到的；每個值附 run_id。
  - superseded_notes：X7 的值和這次量到的不一樣、要注意的地方（不改 X7 的檔，只在這裡寫）。
只讀 results/p3_calib_3090/ 下 summarize_*.py 寫好的 CSV／JSON，和 X7 的 constants_3090.json。

  python code/p3cal/make_constants_v2.py --dir results/p3_calib_3090 [--c0 <run_id>]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
X7C = os.path.join(REPO, "results", "p3_explore", "x7", "constants_3090.json")
SRID = os.environ.get("RUN_ID", "NO_RUN_ID")
NM = "NOT_MEASURED"


def rows(p):
    return list(csv.DictReader(open(p)))


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--c0", default=None, help="c0c 的 run_id（有量到才給）")
    a = ap.parse_args()
    D = lambda n: os.path.join(a.dir, n)  # noqa: E731
    x7 = json.load(open(X7C))
    out = {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"), "summarize_run_id": SRID,
        "platform": x7["platform"],
        "note": "〔3090 常數 v2〕＝X7 的 constants_3090.json（inherited_from_x7，沒有重量）＋p3cal 的新量測（new_in_v2）。"
                "X7 的檔沒有改。每個值附 run_id；沒量到的寫 NOT_MEASURED。",
        "x7_source": {"path": "results/p3_explore/x7/constants_3090.json", "x7_summarize_run_id": x7["summarize_run_id"]},
    }
    # ---------------- 沿用 X7
    inh = {"how_to_use_x7": x7["how_to_use"], "models": {}}
    for m, v in x7["models"].items():
        inh["models"][m] = {k: v[k] for k in v if k in (
            "chunk_bytes", "n_blocks_32k", "f_ms_idle_32k", "f_sum_s_32k", "f_src_run", "cpu_tier_h2d_GiBps_chunk_view",
            "cpu_tier_h2d_src_run", "ell_cpu_ms", "bstar_GiBps", "bstar_src_run", "busy_0.7_slowdown", "longest_fit")}
    for k in ("cpu_tier_h2d_llama_chunk_GiBps_c0", "cpu_tier_d2h_llama_chunk_GiBps_c0", "cpu_tier_k_read_duplex",
              "dequant_bench_ms_per_block", "dequant_bench_src_run"):
        inh[k] = x7[k]
    inh["qwen_restore_cpu_tier_x7_L0"] = {k: x7["models"]["qwen"][k] for k in (
        "cpu_tier_meet_E0_32k", "t_new_turn_256_s_32k", "dequant_ms_per_block_in_restore_cpu", "restore_grid_runs")}
    out["inherited_from_x7"] = inh

    new = {}
    # ---------------- /ssd8
    s = json.load(open(D("ssd8_bw_summary.json")))
    new["ssd8_disk"] = {
        "device": "/dev/sdg（Samsung SSD 870 QVO 8TB，SATA，ext4，/ssd8；共用碟，78% 滿）",
        "method": "code/disk_bw.py --size-mib 16384 --repeats 3（O_DIRECT 循序寫＋fsync、再循序讀，4 MiB 一次）；逐秒 /proc/diskstats",
        "write_MiBps_median": s["write_MiBps_median"], "read_MiBps_median": s["read_MiBps_median"],
        "write_GiBps_median": s["write_GiBps_median"], "read_GiBps_median": s["read_GiBps_median"],
        "write_MiBps_each": s["write_MiBps"], "read_MiBps_each": s["read_MiBps"],
        "foreign_write_MiB_during_our_write": s["foreign_write_MiB_during_write"],
        "slc_cache_phase_seen": False,
        "slc_note": "逐秒寫入速度從第 1 秒起就是 135–188 MiB/s（ssd8_bw_persec.csv），16 GiB × 3 都沒有看到比較快的開頭；"
                    "這次量到的是 SLC 之外的持續寫入。SLC 快取當時的大小：NOT_MEASURED（看不到韌體狀態）。",
        "run_id": s["run_id"]}
    # ---------------- CPU 忙
    sd = rows(D("cpubusy_slowdown.csv"))
    cs = json.load(open(D("cpubusy_summary.json")))
    want = {"H2D_pinned_1024MiB": "bw", "D2H_pinned_1024MiB": "bw", "H2D_pageable_1024MiB": "bw",
            "D2H_pageable_1024MiB": "bw", "H2D_pinned_chunk_into_kv_view_64MiB": "bw",
            "H2D_pinned_qwen_block_into_kv_view_28.0MiB": "bw", "load_only:t_restore": "time", "cake:t_restore": "time",
            "load_only:ttft": "time", "cake:ttft": "time", "compute_only:t_restore": "time", "load_only:t_h2d": "time",
            "cake:b": "count"}
    tab = {}
    for r in sd:
        if r["metric"] not in want:
            continue
        key = f"N={r['N']}:{r['place']}"
        tab.setdefault(r["metric"], {})[key] = {
            "median": f(r["median"]), "unit": r["unit"], "slowdown_x": f(r["slowdown_x"]) if r["slowdown_x"] else None,
            "n": int(r["n"]), "median_by_phase": r["median_by_phase"]}
    new["cpu_busy"] = {
        "definition": "N 個執行緒的 numpy 大塊複製（code/p3cal/cpu_stress.py，每執行緒 src、dst 各 256 MiB）。"
                      "place＝same：壓力綁 GPU 所在 NUMA 節點的 CPU（量測 process 也綁在那裡）；other：綁另一個節點；all：不綁。"
                      "頻寬的 slowdown_x＝N＝0 中位數÷該格中位數；時間的 slowdown_x＝該格中位數÷N＝0 中位數。"
                      "N＝0 的基準是開頭、結尾兩段合併；same 的每個 N 是往上、往下兩段合併（median_by_phase 有各段）。",
        "model": "qwen 32K（64 block），BF16、真實 CPU 層（不限速）、GPU 閒", "gpu": cs["gpu_phys"], "gpu_numa_node": cs["gpu_numa_node"],
        "slowdown": tab,
        "phases": cs["phases"],
        "caveats": [
            "壓力程式的記憶體沒有綁節點（沒有 numactl，沒用 libnuma）。事後（02:33–02:35）查：GPU 6 的 node1 MemFree 只剩 0.5–1.3 GB（其餘是 page cache），"
            "zone_reclaim_mode=0，綁 node1 CPU 的壓力有一大半配到 node0（run 20261018-023443-p3cal-stress-numa，啟動後 5 秒時已配的 anon 頁："
            "N=8 node1 佔 44.5%、N=48 佔 23.2%）。cpubusy run 當時（00:50–01:02）的分佈：NOT_MEASURED。"
            "量測 process 的 pinned 緩衝在哪個節點：NOT_MEASURED。",
            "所以 same 的倍數不隨 N 單調（pinned H2D 1 GiB：N=8 1.35×、16 1.42×、32 1.06×、48 1.04×），而 other（壓力綁 node0）反而最慢（1.76×）。"
            "這是這台機器、這個時段的狀態，不是一般規律。",
            "N＝0 的兩段（開頭、結尾）幾乎一樣（pinned H2D 1 GiB 7.347／7.345 GiB/s），但同一個 N 的往上、往下兩段差到 20–25%（median_by_phase）。",
            "compute_only 在 same 的 N≥32 慢 3.1×、在 other 只慢 1.006×：這是 GPU 發 kernel 的 Python 執行緒搶不到 CPU（同節點 48 個邏輯 CPU 被壓力佔滿），"
            "不是記憶體頻寬。Cake 的重算線也受這個影響。",
            "每格 n＝10（same，兩段各 5）或 5（all、other）；只有一個 launch。"],
        "run_id": cs["run_id"]}
    # ---------------- C0（不含 duplex）：cpubusy run 的兩段 N＝0（主機安靜：執行中行程 2–6 個），X7 之後的另一個時段
    c0 = {}
    for r in rows(D("cpubusy_h2d_raw.csv")):
        if r["N"] == "0":
            c0.setdefault(f"{r['item']}_{r['size_mib']}MiB", []).append(float(r["GiBps"]))
    new["c0_gpu6_from_cpubusy_N0"] = {
        "values_GiBps_median": {k: round(sorted(v)[len(v) // 2] if len(v) % 2 else (sorted(v)[len(v) // 2 - 1] + sorted(v)[len(v) // 2]) / 2, 4)
                                for k, v in sorted(c0.items())},
        "n": {k: len(v) for k, v in sorted(c0.items())},
        "note": "GPU 6（node1，taskset node1）。和 m7_calib c0 同一個迴圈；duplex 沒量。X7 同一張卡（20261017-175001-p3x7-c0-g6，主機 loadavg 約 45）："
                "pinned H2D 1 GiB 6.567、D2H 6.048、pageable H2D 2.738 GiB/s。",
        "run_id": cs["run_id"]}
    # ---------------- 還原格子
    S = rows(D("restore_3090_v2_summary.csv"))
    M = rows(D("meet_check_v2.csv"))
    L = rows(D("lowprec_cpu_v2.csv"))
    for m in ("llama", "qwen"):
        g = "grid_llama" if m == "llama" else "grid_qwen"
        cpu = [r for r in S if r["model"] == m and r["grid"] == g and r["tier"] == "cpu"]
        meet = {r["gpu_state"]: f(r["b_median"]) for r in cpu if r["config"] == "E0" and r["mode"] == "cake"}
        tnew = {}
        for st in ("idle", "busy"):
            v = sorted(f(r["t_new_turn_median_s"]) for r in S if r["model"] == m and r["grid"] == g and r["gpu_state"] == st
                       and r["length"] == "32768")
            tnew[st] = v[len(v) // 2] if v else NM
        rt = {}
        for impl, gg in (("torch(m10_codec)", g), ("fused_triton", "grid_fused")):
            for r in S:
                if r["model"] == m and r["grid"] == gg and r["tier"] == "cpu" and r["dequant_impl"] == impl:
                    rt[f"{impl}|{r['fmt']}|{r['mode']}|{r['gpu_state']}"] = {
                        "t_restore_median_s": f(r["t_restore_median_s"]), "ttft_median_s": f(r["ttft_median_s"]),
                        "b_median": f(r["b_median"]), "dequant_ms_per_lp_block": f(r["dequant_ms_per_lp_block"]) if r["dequant_ms_per_lp_block"] else None,
                        "n": int(r["n"]), "run_id": r["run_id"]}
        new.setdefault("models", {})[m] = {
            "cpu_tier_meet_E0_32k_measured": meet,
            "cpu_tier_meet_E0_32k_x7_model_prediction": {r["gpu_state"]: f(r["b_model"]) for r in M
                                                         if r["model"] == m and r["grid"] == g and r["tier"] == "cpu"},
            "meet_model_check_within_1": f"{sum(r['within_1'] == 'True' for r in M if r['model'] == m and r['grid'] == g)}/"
                                         f"{sum(1 for r in M if r['model'] == m and r['grid'] == g)}",
            "t_new_turn_256_s_32k": tnew,
            "cpu_tier_restore_32k": rt,
            "restore_grid_runs": sorted({x for r in cpu for x in r["run_id"].split()}),
            "fused_grid_runs": sorted({x for r in S if r["model"] == m and r["grid"] == "grid_fused" for x in r["run_id"].split()})}
    new["models"]["llama"]["cpu_tier_64k"] = (
        "NOT_MEASURED：run 20261018-013157-p3cal-grid-llama64k-idle 在暖機的第一次 Cake 還原時 OOM（GPU 6 乾淨、只有我們的 process）："
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 MiB. GPU 0 has a total capacity of 23.68 GiB of which 58.94 MiB is free. "
        "… this process has 23.60 GiB memory in use. Of the allocated memory 23.10 GiB is allocated by PyTorch（m7_restore_harness._load 的 copy_）。"
        "busy 版沒跑（同一設定再加背景 m7_busy）。")
    new["lowprec_cpu_tier"] = {"rows": [{k: f(v) for k, v in r.items() if k not in ("summarize_run_id",)} for r in L],
                               "src": "lowprec_cpu_v2.csv"}
    new["c0_third_session"] = ({"run_id": a.c0, "src": "h2d_c0c_3090.csv"} if a.c0 else
                               "NOT_MEASURED：等不到乾淨的卡（見 06b_3090_calib.md §6）")
    out["new_in_v2"] = new
    # ---------------- 和 X7 不一樣、要注意的
    q7 = x7["models"]["qwen"]
    out["superseded_notes"] = [
        "Qwen CPU 層的還原「時間」：X7 的 L0（GPU 1，同 switch 鄰居 GPU 0、2 有別人的 PCIe 流量，最高 1.7 GB/s）BF16 只載入閒 0.369 s；"
        "這次的 L1b（GPU 6，鄰居 4、5 的流量 ≤7 MB/s）0.248 s（和 64×ℓ＝0.253 s 一致）。X7 已寫「時間欄可能偏高」；要用時間就用 v2 的值。"
        "會合點 b 兩次相同（閒 3、忙 2）。",
        "Llama CPU 層會合點：X7 只有模型預測（閒 4、忙 3），v2 實測兩個 launch 都是閒 4、忙 3。",
        "Llama 的新一輪 256 token：X7 只有 ladder 的 1 次（閒 0.143 s、忙 NOT_MEASURED），v2 有 32K 還原格子的中位數（閒、忙）。",
        f"X7 的 Qwen dequant_ms_per_block_in_restore_cpu（{q7['restore_grid_runs']}）：v2 的 L1b 主線值相近（見 lowprec_cpu_tier）。"]
    json.dump(out, open(D("constants_3090_v2.json"), "w"), indent=1, ensure_ascii=False)
    print("wrote", D("constants_3090_v2.json"))


if __name__ == "__main__":
    main()
