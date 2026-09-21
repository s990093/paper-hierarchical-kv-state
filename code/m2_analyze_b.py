#!/usr/bin/env python3
"""M2 分析（平台 B）：把 retrieval / recompute 的原始列整理成成本常數。

只做算術，不生資料。每一列都指回原始 CSV 與 run_id。

輸出：
  results/m2_harness_mi300x/cost_constants_mi300x.csv   每個模型每一階的成本
  results/m2_harness_mi300x/recompute_fit_mi300x.csv    C_recompute(P) = C0 + a·P 的最小平方擬合
  stdout                                                κ 與 P* 的表
"""
from __future__ import annotations

import csv
import statistics as st
from datetime import datetime
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "results/m2_harness_mi300x"

# BF16 KV 的 KiB/token（由各模型 config 算；M1 已用實測容量交叉驗證）
KV_KIB = {"b-llama8b": 128.0, "b-qwen7b-1m": 56.0, "b-qwen3-30b-a3b": 96.0,
          "b-seedoss36b": 256.0, "b-ultralong8b-1m": 128.0}
# 用哪一份 retrieval 檔。_v2 是修正 CPU 主階大小後重量的（見 RUNLOG 發現 8）。
RETRIEVAL = {
    "b-llama8b": "retrieval_cost_b-llama8b.csv",
    "b-qwen7b-1m": "retrieval_cost_b-qwen7b-1m_v2.csv",
    "b-qwen3-30b-a3b": "retrieval_cost_b-qwen3-30b-a3b_v2.csv",
    "b-seedoss36b": "retrieval_cost_b-seedoss36b_v2.csv",
}
TIERS = ["gpu_resident", "gpu_fp8", "gpu_int4", "cpu", "ssd", "drop"]


def med(v: list[float]) -> float | None:
    return st.median(v) if v else None


def fit_linear(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """最小平方 y = C0 + a·x，回傳 (C0, a, R²)。"""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx
    c0 = my - a * mx
    ss_res = sum((y - (c0 + a * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return c0, a, 1 - ss_res / ss_tot if ss_tot else float("nan")


def load_retrieval() -> dict:
    """每個模型每一階的 warm/cold 中位數（ms），以及驗證欄位。"""
    out = {}
    for model, fname in RETRIEVAL.items():
        rows = [r for r in csv.DictReader((OUT / fname).open())]
        if not rows:
            continue
        ctx = int(rows[0]["ctx"])
        by = defaultdict(list)
        for r in rows:
            if r["ttft_ms"]:
                by[(r["tier"], r["round"])].append(float(r["ttft_ms"]))
        # ssd 必須每列 warm 都完整讀回一個前綴，否則不採用（RUNLOG 發現 8）
        ssd_warm = [r for r in rows if r["tier"] == "ssd" and r["round"] == "warm"]
        need = 0.95 * ctx * KV_KIB[model] * 1024
        ssd_ok = bool(ssd_warm) and all(int(r["req_read_bytes"] or 0) >= need for r in ssd_warm)
        out[model] = {
            "ctx": ctx, "file": fname, "run_id": rows[0]["run_id"],
            "prefix_kv_bytes": ctx * KV_KIB[model] * 1024,
            "ssd_full_read": ssd_ok,
            "warm": {t: med(by[(t, "warm")]) for t in TIERS},
            "cold": {t: med(by[(t, "cold")]) for t in TIERS},
            "n": {t: len(by[(t, "warm")]) for t in TIERS},
        }
    return out


def load_recompute() -> dict:
    out = {}
    for f in sorted(OUT.glob("recompute_position_b-*.csv")):
        rows = list(csv.DictReader(f.open()))
        model = rows[0]["model_key"]
        by = defaultdict(list)
        for r in rows:
            if r["ttft_ms"]:
                by[int(r["cached_prefix_tokens"])].append(float(r["ttft_ms"]))
        out[model] = {"file": f.name, "run_id": rows[0]["run_id"],
                      "chunk": int(rows[0]["recomputed_tokens"]),
                      "med": {P: st.median(v) for P, v in sorted(by.items())},
                      # 🔴 P ≥ 163,840 時同一位置的 3 次會在兩個值之間跳（比值約 1.6，見 RUNLOG 發現 10）。
                      #    中位數會隨機落在其中一個模式上，擬合殘差因此很大。
                      #    min 取的是「乾淨那一條路徑」，兩種都算，並列報告。
                      "min": {P: min(v) for P, v in sorted(by.items())},
                      "reps": {P: v for P, v in sorted(by.items())}}
    return out


def main() -> int:
    ret, rec = load_retrieval(), load_recompute()

    # ── 1. 每一階的成本常數 ────────────────────────────────────────────
    rows = []
    print("\n=== 取回成本（warm TTFT 減 gpu_resident 基準）===")
    hdr = f"{'model':18s}{'ctx':>9s}  {'tier':14s}{'warm ms':>10s}{'減基準 ms':>11s}{'µs/token':>10s}{'GB/s':>8s}{'n':>3s}"
    print(hdr)
    for model, d in ret.items():
        base = d["warm"]["gpu_resident"]
        for t in TIERS:
            w = d["warm"][t]
            if w is None or base is None:
                print(f"{model:18s}{d['ctx']:>9,}  {t:14s}{'NOT_MEASURED':>10s}")
                continue
            delta = w - base
            gbs = (d["prefix_kv_bytes"] / 1e9) / (delta / 1000) if delta > 0 else float("nan")
            note = ""
            if t == "ssd" and not d["ssd_full_read"]:
                note = "（未完整讀回，不採用）"
            print(f"{model:18s}{d['ctx']:>9,}  {t:14s}{w:>10.1f}{delta:>11.1f}"
                  f"{1000 * delta / d['ctx']:>10.2f}{gbs:>8.2f}{d['n'][t]:>3d} {note}")
            rows.append({
                "ts": datetime.now().astimezone().isoformat(),
                "model_key": model, "ctx": d["ctx"], "tier": t,
                "warm_ttft_ms_median": round(w, 1),
                "delta_vs_gpu_resident_ms": round(delta, 1),
                "us_per_token": round(1000 * delta / d["ctx"], 3),
                "effective_gb_per_s": round(gbs, 3) if delta > 0 else "",
                "prefix_kv_bytes": int(d["prefix_kv_bytes"]),
                "cold_ttft_ms_median": round(d["cold"][t], 1) if d["cold"][t] else "",
                "n_warm": d["n"][t],
                "ssd_full_prefix_read": d["ssd_full_read"] if t == "ssd" else "",
                "source_csv": d["file"], "run_id": d["run_id"],
            })
    with (OUT / "cost_constants_mi300x.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[m2] -> {OUT / 'cost_constants_mi300x.csv'}")

    # ── 2. C_recompute(P) 線性擬合 ─────────────────────────────────────
    print("\n=== C_recompute(P) = C0 + a·P（每次重算 chunk 個 token）===")
    print(f"{'model':18s}{'範圍':>20s}{'C0 ms':>8s}{'a µs/千token':>14s}{'R²':>8s}{'最大偏離%':>10s}")
    fits = {}
    frows = []
    for model, d in rec.items():
      for stat in ("med", "min"):
        pts = sorted(d[stat].items())
        # 分兩段：≤114,688（三次重複差 < 4%）與全範圍
        for label, sel in ((f"{stat} P<=114688", [p for p in pts if p[0] <= 114688]),
                           (f"{stat} all", pts)):
            if len(sel) < 3:
                continue
            xs = [float(p) for p, _ in sel]
            ys = [v for _, v in sel]
            c0, a, r2 = fit_linear(xs, ys)
            dev = max(abs(y - (c0 + a * x)) / y for x, y in zip(xs, ys)) * 100
            print(f"{model:18s}{label:>20s}{c0:>8.1f}{1000 * a:>14.2f}{r2:>8.4f}{dev:>10.1f}")
            frows.append({"ts": datetime.now().astimezone().isoformat(),
                          "model_key": model, "stat": stat, "range": label, "n_positions": len(sel),
                          "C0_ms": round(c0, 2), "a_ms_per_token": round(a, 8),
                          "a_us_per_1k_token": round(1000 * a, 3), "r2": round(r2, 5),
                          "max_abs_dev_pct": round(dev, 2), "chunk_tokens": d["chunk"],
                          "source_csv": d["file"], "run_id": d["run_id"]})
            if label == "min all":      # P* 用乾淨路徑（min）的全範圍擬合
                fits[model] = (c0, a, d["chunk"])
    with (OUT / "recompute_fit_mi300x.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(frows[0]))
        w.writeheader()
        w.writerows(frows)
    print(f"[m2] -> {OUT / 'recompute_fit_mi300x.csv'}")

    # ── 3. κ 與 P* ────────────────────────────────────────────────────
    # κ = 重算成本 / 傳輸成本（同一批 token、同一個 ctx）。
    # 論文主張 κ 跨硬體變動達 32 倍 → 這裡給平台 B 的實測值。
    print("\n=== κ = 重算 / 傳輸（在量測用的 ctx 上，以 warm 減基準計）===")
    print(f"{'model':18s}{'ctx':>8s}{'κ_cpu':>9s}{'κ_ssd':>9s}")
    for model, d in ret.items():
        base, drop = d["warm"]["gpu_resident"], d["warm"]["drop"]
        if base is None or drop is None:
            continue
        rc = drop - base
        k = {}
        for t in ("cpu", "ssd"):
            w = d["warm"][t]
            ok = (t != "ssd") or d["ssd_full_read"]
            k[t] = (rc / (w - base)) if (w and ok and w - base > 0) else None
        cpu_s = f"{k['cpu']:.2f}x" if k["cpu"] else "N/A"
        ssd_s = f"{k['ssd']:.2f}x" if k["ssd"] else "N/A"
        print(f"{model:18s}{d['ctx']:>8,}{cpu_s:>9s}{ssd_s:>9s}")

    # P*：SSD 取回一個 chunk 的成本 == 在位置 P 重算同一個 chunk 的成本
    #     SSD 每 token 成本 × chunk  ==  C0 + a·P
    print("\n=== P*：SSD 與 DROP 的交叉點（chunk = 2,048 token）===")
    print(f"{'model':18s}{'SSD ms/chunk':>14s}{'C0 ms':>8s}{'a µs/千':>10s}{'P* token':>12s}{'說明':>6s}")
    for model, d in ret.items():
        if model not in fits or not d["ssd_full_read"]:
            print(f"{model:18s}{'NOT_MEASURED':>14s}")
            continue
        c0, a, chunk = fits[model]
        base, ssd = d["warm"]["gpu_resident"], d["warm"]["ssd"]
        ssd_chunk = (ssd - base) / d["ctx"] * chunk
        pstar = (ssd_chunk - c0) / a
        note = "" if pstar > 0 else "（SSD 一直比重算便宜）"
        print(f"{model:18s}{ssd_chunk:>14.1f}{c0:>8.1f}{1000 * a:>10.2f}{pstar:>12,.0f} {note}")
    print("\nP* 的意思：前綴位置小於 P* 時重算比較便宜，大於 P* 時從 SSD 取回比較便宜。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
