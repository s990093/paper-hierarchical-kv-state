#!/usr/bin/env python3
"""F4 稽核：κ 的兩件事（只讀既有資料，不碰 GPU）。

A. 3090 的 κ_cpu＝8.9 能不能用？
   來源檔在 git 52803bc^:results/m2_harness/retrieval_cost_{sata,nvme}.csv（每列 HEAVY）。
   同一個量（Llama-3.1-8B、ctx 16,384、NVMe）在 20260831-211745 的交錯重量裡，
   最後一輪剛好碰到整機 QUIET → 一個「同一支程式、同一天、只差整機爭用」的自然對照。
   算法和 main.tex 表 kappa 一樣：(warm TTFT 中位數 − gpu_resident warm 中位數) / ctx。

B. main.tex 的 κ（MI300X 1.5）和 harness 第一個 chunk 的 f/ℓ（≈16）為什麼差 10 倍？
   把差距拆成「傳輸路徑」「重算引擎」「位置」三個倍數，每個都用實測值。

輸出：results/audit_20261010/kappa_3090_contention.csv、kappa_reconcile.csv
用法：source code/m7_env.sh; HIP_VISIBLE_DEVICES= m7run f4-kappa python code/m9_f4_kappa.py
"""
from __future__ import annotations

import csv
import io
import os
import statistics as st
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("F4_OUT", REPO / "results/audit_20261010"))
RID = os.environ.get("RUN_ID", "NO_RUN_ID")
NOW = datetime.now().astimezone().isoformat(timespec="seconds")
COMMIT = "52803bc^"


def git_csv(path: str) -> list[dict]:
    txt = subprocess.run(["git", "-C", str(REPO), "show", f"{COMMIT}:{path}"],
                         capture_output=True, text=True, check=True).stdout
    return list(csv.DictReader(io.StringIO(txt)))


def kappa(rows: list[dict], label: str, src: str, note: str = "") -> dict:
    by = defaultdict(list)
    for r in rows:
        if r["ttft_ms"]:
            by[(r["tier"], r["round"])].append(float(r["ttft_ms"]))
    base = by[("gpu_resident", "warm")]
    cpu, ssd, drop = by[("cpu", "warm")], by[("ssd", "warm")], by[("drop", "warm")]
    ctx = int(rows[0]["ctx"])
    out = {"audit_run_id": RID, "ts": NOW, "variant": label, "source": src,
           "source_run_id": ";".join(sorted({r["run_id"] for r in rows})),
           "host_contention_levels": ";".join(sorted({r.get("host_contention", "") for r in rows})),
           "n_base": len(base), "n_cpu": len(cpu), "n_ssd": len(ssd), "n_drop": len(drop), "ctx": ctx}
    if not (base and cpu and drop):
        out["note"] = "樣本不足 → NOT_MEASURED；" + note
        return out
    b = st.median(base)
    us = lambda v: 1000 * (st.median(v) - b) / ctx  # noqa: E731
    out.update({"base_ms": round(b, 1), "cpu_warm_med_ms": round(st.median(cpu), 1),
                "drop_warm_med_ms": round(st.median(drop), 1),
                "recompute_us_tok": round(us(drop), 2), "cpu_us_tok": round(us(cpu), 2),
                "ssd_us_tok": round(us(ssd), 2) if ssd else "",
                "kappa_cpu": round(us(drop) / us(cpu), 3),
                "kappa_ssd": round(us(drop) / us(ssd), 3) if ssd else "",
                "cpu_warm_min_max_ms": f"{min(cpu):.1f}–{max(cpu):.1f}",
                "kappa_cpu_if_cpu_min": round((st.median(drop) - b) / (min(cpu) - b), 3),
                "kappa_cpu_if_cpu_max": round((st.median(drop) - b) / (max(cpu) - b), 3),
                "note": note})
    return out


def part_a() -> list[dict]:
    rows = []
    sata = git_csv("results/m2_harness/retrieval_cost_sata.csv")
    nvme = git_csv("results/m2_harness/retrieval_cost_nvme.csv")
    inter = git_csv("results/m2_harness/retrieval_cost_nvme_interleaved.csv")
    rows.append(kappa(sata, "3090_SATA_paper", "git 52803bc^:results/m2_harness/retrieval_cost_sata.csv",
                      "main.tex 表 kappa 第 1 列的來源；26 列全 HEAVY（foreign_gpu_count=6）"))
    rows.append(kappa(nvme, "3090_NVMe_paper", "git 52803bc^:results/m2_harness/retrieval_cost_nvme.csv",
                      "main.tex 表 kappa 第 2 列的來源；26 列全 HEAVY"))
    rows.append(kappa(inter, "3090_NVMe_interleaved_all",
                      "git 52803bc^:results/m2_harness/retrieval_cost_nvme_interleaved.csv",
                      "同一個量的交錯重量（3 輪），HEAVY 66 列＋LIGHT 1＋QUIET 11 混在一起"))
    heavy = [r for r in inter if r["host_contention"] == "HEAVY"]
    quiet = [r for r in inter if r["host_contention"] == "QUIET"]
    rows.append(kappa(heavy, "3090_NVMe_interleaved_HEAVY_only",
                      "同上，只取 host_contention=HEAVY", "8 個 cpu warm、11 個 drop warm"))
    rows.append(kappa(quiet, "3090_NVMe_interleaved_QUIET_only",
                      "同上，只取 host_contention=QUIET",
                      "QUIET 只有最後一輪：base 1、cpu warm 4、drop warm 1（樣本很少）"))
    # QUIET 的 cpu＋全部 drop／base（drop 幾乎不受整機爭用影響，見 drop 欄）
    mix = [r for r in inter if (r["tier"] == "cpu" and r["host_contention"] == "QUIET")
           or r["tier"] in ("drop", "gpu_resident")]
    rows.append(kappa(mix, "3090_NVMe_interleaved_cpuQUIET_dropAll",
                      "同上；cpu 只取 QUIET，drop 與 base 全部（各 12／3 列）",
                      "敏感度：drop 在 HEAVY 與 QUIET 下幾乎相同，所以借用全部 drop 增加樣本"))
    return rows


def part_b() -> list[dict]:
    """把 harness 第一個 chunk 的 f/ℓ（≈16）一步步換算成 main.tex 的 1.53。"""
    out = []
    # 1) harness：f(0) 取 C1 的中位數；ℓ 取 tier_params.json 的 CPU ℓ_chunk
    c1 = list(csv.DictReader(open(REPO / "results/m7_write_policy_mi300x/calib_c1.csv")))
    f = defaultdict(list)
    for r in c1:
        if r["item"] == "f_chunk":
            f[int(r["chunk_idx"])].append(float(r["ms"]))
    f0 = st.median(f[0])
    favg32 = st.mean(st.median(f[i]) for i in range(32))      # 16,384 token = 32 個 chunk
    import json
    tp = json.load(open(REPO / "results/m7_write_policy_mi300x/tier_params.json"))
    ell = tp["cpu"]["ell_chunk_ms"]
    # 2) vLLM 的 per-chunk f（m7_vllm_f.py，f_chunk＝扣掉固定開銷）
    vf = defaultdict(list)
    for r in csv.DictReader(open(REPO / "results/m7_explore_mi300x/vllmf2_llama31_8b.csv")):
        if r["item"] == "f_chunk":
            vf[int(r["chunk_idx"])].append(float(r["ms"]))
    vf0 = st.median(vf[0])
    vfavg32 = st.mean(st.median(vf[i]) for i in range(32))
    # 3) main.tex 的量：m2 retrieval（vLLM 0.28，整段 16,384 warm TTFT 減基準）
    rec_us, cpu_us = 46.742, 30.525      # 由 results/audit_20261010/cost_constants_mi300x_rebuilt.csv 重算
    chunk = 512
    steps = [
        ("H0 harness 第一個 chunk：f(0)/ℓ_cpu", f0 / ell,
         f"f(0)={f0:.2f} ms（C1，HF transformers SDPA）÷ ℓ={ell:.3f} ms（35.4 GiB/s×0.9 derate，64 MiB）",
         "20261008-130316-m7-c1;20261008-130257-m7-c0"),
        ("H1 改用 16K 前綴的平均 f（位置效應）", favg32 / ell,
         f"前 32 個 chunk 的 f 平均 {favg32:.2f} ms ÷ ℓ", "20261008-130316-m7-c1"),
        ("H2 改用 vLLM 的 per-chunk f（扣固定開銷）", vfavg32 / ell,
         f"vLLM f_chunk 前 32 個平均 {vfavg32:.2f} ms（f(0)={vf0:.2f}）÷ ℓ",
         "20261009-180348-r1-vllmf2-llama31-8b"),
        ("H3 改用 vLLM 整段 prefill 的平均重算（含每步固定開銷）", rec_us * chunk / 1000 / ell,
         f"{rec_us} µs/tok×512＝{rec_us * chunk / 1000:.2f} ms ÷ ℓ（仍是 harness 的 35.4 GiB/s）",
         "20260915-124058-m2-b-llama8b-retrieval"),
        ("H4 再把傳輸換成 vLLM OffloadingConnector 實測（= main.tex 的 κ_cpu）",
         rec_us / cpu_us,
         f"{rec_us}／{cpu_us} µs/tok；CPU 路徑有效頻寬 ≈{128 * 1024 / cpu_us / 1e3:.2f} GB/s",
         "20260915-124058-m2-b-llama8b-retrieval"),
    ]
    for name, val, how, rid in steps:
        out.append({"audit_run_id": RID, "ts": NOW, "step": name, "value": round(val, 3),
                    "how": how, "source_run_id": rid})
    out.append({"audit_run_id": RID, "ts": NOW, "step": "倍數：傳輸路徑（vLLM CPU 路徑 ÷ harness ℓ）",
                "value": round(cpu_us * chunk / 1000 / ell, 3),
                "how": "同樣 512 token 的搬運時間比", "source_run_id": "20260915-124058;20261008-130257-m7-c0"})
    out.append({"audit_run_id": RID, "ts": NOW, "step": "倍數：重算引擎（HF f(0) ÷ vLLM 整段平均）",
                "value": round(f0 / (rec_us * chunk / 1000), 3),
                "how": "HF 第一個 chunk 對 vLLM 16K 平均", "source_run_id": "20261008-130316-m7-c1;20260915-124058"})
    return out


def write(path: Path, rows: list[dict]) -> None:
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[f4] wrote {len(rows)} rows -> {path}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    a = part_a()
    b = part_b()
    write(OUT / "kappa_3090_contention.csv", a)
    write(OUT / "kappa_reconcile.csv", b)
    for r in a:
        print(f"{r['variant']:42s} κ_cpu={r.get('kappa_cpu', 'NA')}  recompute={r.get('recompute_us_tok')}"
              f"  cpu={r.get('cpu_us_tok')}  levels={r['host_contention_levels']}")
    for r in b:
        print(f"{r['step']:60s} {r['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
