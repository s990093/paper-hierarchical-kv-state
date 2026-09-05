#!/usr/bin/env python3
"""M5 第二階段的判定材料。**全部從 CSV 讀，不手打任何數字。**

與 `m4_verdict.py` 同一個角色：把散在幾個 CSV 裡的結果排成一張可以直接讀的表，
讀不到的印 `NOT_MEASURED`。

    python code/m5_summary.py
"""
from __future__ import annotations
import csv
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "results/m5_predictor"


def rows(name: str) -> list[dict]:
    p = OUT / name
    return list(csv.DictReader(p.open())) if p.exists() else []


def f(x, d=2):
    if x is None:
        return "NOT_MEASURED"
    if x == "":
        return "—"            # 不適用（平凡基線沒有 ECE/AUC），不是「沒量」
    try:
        return f"{float(x):,.{d}f}"
    except (TypeError, ValueError):
        return "NOT_MEASURED"


def h(title: str) -> None:
    print(f"\n## {title}")


def main() -> int:
    print("=" * 78)
    print(" Milestone 5 第二階段 — 未來效用預測器（實作 + 訓練）")
    print(" 全部數字自 results/m5_predictor/*.csv 讀出。")
    print("=" * 78)

    h("A. 訓練資料：標籤的兩個機制各貢獻多少")
    print(f"   {'trace':<14}{'W(存取)':>10}{'樣本':>12}{'正樣本率':>10}"
          f"{'機制(a)':>12}{'機制(b)':>12}{'尾巴丟棄':>10}")
    for r in rows("samples.csv"):
        print(f"   {r['trace']:<14}{int(r['window_accesses']):>10,}"
              f"{int(r['n_samples']):>12,}{100 * float(r['positive_rate']):>9.1f}%"
              f"{int(r['n_labeled_by_next_access']):>12,}"
              f"{int(r['n_labeled_by_window']):>12,}"
              f"{int(r['n_dropped_tail']):>10,}")
    print("   ※ 機制 (b)（滑動視窗到期）是負樣本的唯一來源；沒有它訓練集全是正樣本。")

    h("B. 預測器品質（測試段＝時序切分的後 30%）")
    met = rows("predictor_metrics.csv")
    print(f"   {'trace':<13}{'損失':<13}{'門檻':<8}{'葉':>4}{'ECE':>8}{'AUC':>8}"
          f"{'錯誤率':>9}{'成本(ms)':>14}{'μs/決策':>11}{'丟棄率':>8}")
    for r in met:
        print(f"   {r['trace']:<13}{r['loss']:<13}{r['threshold_rule']:<8}"
              f"{r.get('num_leaves', ''):>4}{f(r['ece'], 4):>8}{f(r['auc'], 4):>8}"
              f"{f(r['err_rate'], 4):>9}{f(r['cost_ms']):>14}"
              f"{f(r['cost_us_per_decision'], 3):>11}{f(r['drop_rate'], 3):>8}")
    print("   ※ 成本 = 錯掉多少毫秒（FN 付重算、FP 付一個 CPU 槽位），不是錯了幾次。")

    h("C. (B) 區塊：門檻移動 vs 加權訓練，各自值多少")
    for tr in sorted({r["trace"] for r in met}):
        for nl in sorted({r.get("num_leaves", "") for r in met
                          if r["trace"] == tr}, key=lambda x: -int(x or 0)):
            g = {(r["loss"], r["threshold_rule"]): r for r in met
                 if r["trace"] == tr and r.get("num_leaves", "") == nl}
            s05 = g.get(("sym_l2", "0.5"))
            sps = g.get(("sym_l2", "p_star"))
            cps = g.get(("cost_l2", "p_star"))
            if not (s05 and sps and cps):
                continue
            a, b, c = (float(x["cost_ms"]) for x in (s05, sps, cps))
            print(f"   {tr}（葉 {nl}）：對稱@0.5 {a:,.0f} ms"
                  f" → 對稱@p* {b:,.0f} ms（門檻移動省 {100 * (a - b) / a:.1f}%）"
                  f" → 加權@p* {c:,.0f} ms（加權再省 {100 * (b - c) / b:+.1f}%）")
    print("   ※ 論文 §5.3 的可證偽預測：加權訓練的增益應隨 κ 增大而擴大。")

    h("D. κ 隨 block 位置變動時，加權訓練的增益如何變")
    pb = rows("cost_by_position.csv")
    for tr in sorted({r["trace"] for r in pb}):
        print(f"   {tr}")
        print(f"      {'位置':<14}{'κ':>7}{'p*':>8}{'正樣本率':>9}"
              f"{'對稱@p*(ms)':>13}{'加權@p*(ms)':>13}{'加權省':>8}")
        for lo in sorted({int(r["pos_lo"]) for r in pb if r["trace"] == tr}):
            s = next((r for r in pb if r["trace"] == tr and int(r["pos_lo"]) == lo
                      and r["loss"] == "sym_l2" and r["threshold_rule"] == "p_star"), None)
            c = next((r for r in pb if r["trace"] == tr and int(r["pos_lo"]) == lo
                      and r["loss"] == "cost_l2" and r["threshold_rule"] == "p_star"), None)
            if not (s and c):
                continue
            sv, cv = float(s["cost_ms"]), float(c["cost_ms"])
            hi = int(s["pos_hi"])
            lab = f"{lo // 1024}K–{hi // 1024}K" if hi < 10 ** 9 else f"{lo // 1024}K+"
            print(f"      {lab:<14}{f(s['kappa_median'], 1):>7}"
                  f"{f(s['p_star_median'], 4):>8}"
                  f"{100 * float(s['positive_rate']):>8.1f}%{sv:>13,.0f}{cv:>13,.0f}"
                  f"{(100 * (sv - cv) / sv if sv else 0):>7.1f}%")

    h("E. 線上策略拿到 oracle headroom 的多少（同一段 trace、同樣的記帳）")
    ps = rows("policy_sim.csv")
    def kf(r):
        return (r["trace"], r["segment"], r["decode"], int(r["window_accesses"]),
                r.get("drop_cost_rule", "tail"), float(r["ssd_gib"]))

    for key in sorted({kf(r) for r in ps}):
        g = [r for r in ps if kf(r) == key]
        print(f"   {key[0]}　segment={key[1]}　decode={'含' if key[2] == '1' else '不含'}"
              f"　W={key[3]:,}　丟棄計價={key[4]}　SSD={key[5]:,.0f} GiB")
        print(f"      最佳 baseline = {g[0]['best_baseline']}"
              f"　oracle headroom = {f(g[0]['oracle_headroom_pct'], 2)}%")
        print(f"      {'策略':<22}{'總時間(ms)':>16}{'vs baseline':>13}"
              f"{'拿到 headroom':>14}{'SSD 寫入':>12}{'可行':>6}")
        for r in sorted(g, key=lambda r: float(r["total_ms"])):
            print(f"      {r['policy']:<22}{float(r['total_ms']):>16,.0f}"
                  f"{f(r['vs_best_baseline_pct'], 2):>12}%"
                  f"{f(r['headroom_captured_pct'], 1):>13}%"
                  f"{f(r['ssd_write_mibps'], 0):>10} MiB/s"
                  f"{('✅' if r['write_feasible'] == '1' else '🔴'):>5}")

    h("F. 熱路徑延遲（預測器的成本要從它的收益裡扣掉）")
    for r in met:
        if r.get("lat_median_us"):
            print(f"   {r['trace']:<13}{r['loss']:<9}"
                  f"對 {r['lat_k']} 個候選推論一次 {f(r['lat_median_us'], 1)} μs"
                  f"（每候選 {f(r['lat_per_candidate_us'], 3)} μs，"
                  f"單執行緒，量測時 loadavg {r.get('lat_loadavg_1m', '?')}）")
            break
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
