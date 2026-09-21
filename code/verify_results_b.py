#!/usr/bin/env python3
"""平台 B 的結果自檢。跑完實驗一定要跑這支，再宣稱「做完了」。

## 為什麼需要

這個專案這幾天踩過的坑，共通點都是「跑完了、rc=0、但結果其實有問題」：

* `oracle.csv` 固定檔名 → 7 個模型互相覆蓋，只剩最後一次的 20 列（規則 3 失守）
* `/dev/shm` 被殘留檔塞滿 → CPU 階起不來，某些 tier 變成 NOT_MEASURED
* Mistral-Nemo 對隨機 token 立刻吐 EOS → 11/27 個請求的 TTFT 是空的
* 合成工作負載在 B 上不構成壓力 → oracle headroom 0.00%，看起來像「策略沒用」
* 平台 A 的常數（936 GB/s、qwen-awq 剖面、sata/nvme）留在腳本裡

這些都不會讓程式崩潰，只會讓數字安靜地錯。所以自檢必須檢查**內容**，不只看 rc。

## 檢查項目

A. 檔案層級：results 下每個 CSV 都要有 run_id 與 ts 欄（規則 3）
B. 覆蓋率：7 個模型 × 11 項實驗，列出缺哪幾格
C. 完整性：關鍵欄位不得為空（TTFT、容量、分數）
D. 污染：guard 必須 CLEAN、逐列 foreign_gpu_count = 0
E. 退化訊號：工作集沒壓力（headroom 0.00%）、所有 baseline 數字相同、
   磁碟階 fs_dir_bytes = 0 這類「看起來正常但其實什麼都沒發生」
F. run 目錄：exit_code 非 0 的 run 要列出來

用法：
    python code/verify_results_b.py            # 人看的報告
    python code/verify_results_b.py --strict   # 有任何 FAIL 就回傳 1（給佇列當最後一步）
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
R = REPO / "results"
RUNS = Path("/mlsteam/data/tiara/runs")
MODELS = ["b-llama8b", "b-ultralong8b-1m", "b-qwen7b-1m", "b-qwen3-30b-a3b",
          "b-seedoss36b", "b-qwen14b-1m", "b-mistral-nemo12b"]

OK, WARN, FAIL = "✅", "⚠️ ", "🔴"
issues: list[tuple[str, str]] = []


def say(level: str, msg: str) -> None:
    print(f"{level} {msg}")
    if level in (WARN, FAIL):
        issues.append((level, msg))


def rows(p) -> list[dict]:
    p = Path(p)
    if not p.exists():
        return []
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def check_traceability() -> None:
    print("\n── A. 可追溯性：每個 CSV 都要有 run_id 與 ts ──")
    bad = []
    for f in sorted(glob.glob(str(R / "*_mi300x*/*.csv"))) + sorted(glob.glob(str(R / "m1_capacity/*mi300x*.csv"))):
        rs = rows(f)
        if not rs:
            bad.append((f, "空檔"))
            continue
        miss = [c for c in ("run_id", "ts") if c not in rs[0]]
        if miss:
            bad.append((f, f"缺欄位 {miss}"))
    if bad:
        for f, why in bad:
            say(FAIL, f"{Path(f).name}：{why}")
    else:
        say(OK, f"掃描的 CSV 全部都有 run_id 與 ts")


def check_coverage() -> dict:
    print("\n── B. 覆蓋率：7 個模型 × 11 項實驗 ──")
    have = defaultdict(set)
    for r in rows(R / "m1_capacity/capacity_mi300x.csv"):
        for m in MODELS:
            if r["config"].startswith(m + "-bf16"):
                have["M1"].add(m)
    for f in glob.glob(str(R / "m2_harness_mi300x/retrieval_cost_b-*.csv")):
        for r in rows(f):
            have["M2取回"].add(r.get("model_key", ""))
    for f in glob.glob(str(R / "m2_harness_mi300x/recompute_position_b-*.csv")):
        for r in rows(f):
            have["M2重算"].add(r.get("model_key", ""))
    for r in rows(R / "m3_baseline_mi300x/baseline_mi300x.csv"):
        have["M3"].add(r.get("model_key", ""))
    for f in glob.glob(str(R / "m4_oracle_mi300x/oracle_*.csv")):
        for r in rows(f):
            have["M4"].add(r.get("model_profile", ""))
    for key, pat in (("無損", "lossless_b-*.csv"), ("撈針", "needle_b*.csv"),
                     ("GSM8K", "gsm8k_precision_*.csv"),
                     ("LongBench", "longbench_precision_*.csv"),
                     ("RULER", "ruler_precision_*.csv")):
        for f in glob.glob(str(R / f"m5_quality_mi300x/{pat}")):
            for r in rows(f):
                have[key].add(r.get("model_key", ""))
    for f in glob.glob(str(R / "m5_attention_mi300x/alignment_*.json")):
        have["注意力"].add(Path(f).stem[len("alignment_"):])
    cols = ["M1", "M2取回", "M2重算", "M3", "M4", "無損", "撈針", "GSM8K",
            "LongBench", "RULER", "注意力"]
    miss = [(m, c) for m in MODELS for c in cols if m not in have[c]]
    done = len(MODELS) * len(cols) - len(miss)
    print(f"   完成 {done}/{len(MODELS) * len(cols)} 格")
    if miss:
        by_model = defaultdict(list)
        for m, c in miss:
            by_model[m].append(c)
        for m, cs in by_model.items():
            say(WARN, f"{m} 缺：{'、'.join(cs)}")
    else:
        say(OK, "覆蓋矩陣全滿")
    return have


def check_completeness() -> None:
    print("\n── C. 完整性：關鍵欄位不得為空 ──")
    # M2 重算：空的 ttft 代表模型沒生成（Mistral-Nemo 踩過）
    for f in sorted(glob.glob(str(R / "m2_harness_mi300x/recompute_position_b-*.csv"))):
        rs = rows(f)
        empty = [r for r in rs if not r.get("ttft_ms")]
        if empty:
            pos = sorted({int(r["cached_prefix_tokens"]) for r in empty})
            lvl = FAIL if len(empty) > len(rs) * 0.2 else WARN
            say(lvl, f"{Path(f).name}：{len(empty)}/{len(rs)} 列沒有 TTFT（位置 {pos[:6]}）")
    # M3：每個 (模型, ctx, baseline) 都該有 cold 與 warm
    g = defaultdict(set)
    for r in rows(R / "m3_baseline_mi300x/baseline_mi300x.csv"):
        g[(r["model_key"], r["ctx"], r["baseline"])].add(r["round"])
    bad = [k for k, v in g.items() if v != {"cold", "warm"}]
    if bad:
        say(WARN, f"M3 有 {len(bad)} 組缺 cold 或 warm：{bad[:3]}")
    else:
        say(OK, f"M3 的 {len(g)} 組設定都有 cold 與 warm")
    # M5 品質：每個模型每個精度的樣本數要一致
    for pat in ("needle_b*.csv", "gsm8k_precision_*.csv"):
        for f in sorted(glob.glob(str(R / f"m5_quality_mi300x/{pat}"))):
            c = defaultdict(int)
            for r in rows(f):
                c[r.get("config", "")] += 1
            if c and len(set(c.values())) > 1:
                say(WARN, f"{Path(f).name}：各精度樣本數不一致 {dict(c)}")


EXPLAINED = REPO / "results/contamination_explained.json"


def _explained() -> dict:
    """已調查並解釋過的污染紀錄。

    🔴 這不是「忽略清單」。每一筆都必須寫明：怎麼判定的、證據是什麼、影響哪些欄位。
       沒有 evidence 欄位的條目一律不接受，仍然算 FAIL。
    """
    if not EXPLAINED.exists():
        return {}
    d = json.loads(EXPLAINED.read_text())
    return {k: v for k, v in d.items() if isinstance(v, dict) and v.get("evidence")}


def check_contamination() -> None:
    print("\n── D. 污染：guard 必須 CLEAN ──")
    exp = _explained()
    bad = []
    for f in glob.glob(str(R / "*_mi300x*/gpu_guard*.json")):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:                                    # noqa: BLE001
            continue
        if d.get("contaminated"):
            n = Path(f).name
            if n in exp:
                say(WARN, f"{n} 標為污染，但已調查：{exp[n]['cause']}"
                          f"（影響：{exp[n].get('impact', '未寫')}）")
            else:
                bad.append((n, d.get("verdict", "")))
    if bad:
        for n, v in bad:
            say(FAIL, f"{n} 被標成污染且**未經調查**：{v}")
    elif not exp:
        say(OK, "所有 guard 檔都是 CLEAN")
    else:
        say(OK, "沒有未經調查的污染")
    n_bad = 0
    for f in glob.glob(str(R / "*_mi300x*/*.csv")):
        for r in rows(f):
            v = r.get("foreign_gpu_count", "0")
            if v not in ("", "0"):
                n_bad += 1
    if n_bad:
        say(FAIL, f"有 {n_bad} 列的 foreign_gpu_count 不是 0")
    else:
        say(OK, "逐列的 foreign_gpu_count 全部為 0")


def check_degenerate() -> None:
    print("\n── E. 退化訊號：看起來正常但其實什麼都沒發生 ──")
    # E1 M3：同一 ctx 下四個 baseline 的 warm 幾乎相同 → 沒發生逐出
    g = defaultdict(dict)
    for r in rows(R / "m3_baseline_mi300x/baseline_mi300x.csv"):
        if r["round"] == "warm" and r["ttft_ms"]:
            g[(r["model_key"], int(r["ctx"]))].setdefault(r["baseline"], []).append(float(r["ttft_ms"]))
    flat = []
    for (m, ctx), d in g.items():
        if len(d) < 4:
            continue
        med = {k: st.median(v) for k, v in d.items()}
        lo, hi = min(med.values()), max(med.values())
        if hi < lo * 1.05:
            flat.append((m, ctx, round(lo)))
    if flat:
        say(WARN, f"M3 有 {len(flat)} 組的四個 baseline 差 <5%（工作集可能塞得下）：{flat[:4]}")
    else:
        say(OK, "M3 每組設定的 baseline 之間都有可分辨的差異")
    # E2 tier_fs 必須真的寫到磁碟
    zero = [(r["model_key"], r["ctx"]) for r in rows(R / "m3_baseline_mi300x/baseline_mi300x.csv")
            if r.get("baseline") == "tier_fs" and r.get("fs_dir_bytes") in ("0", "")]
    if zero:
        say(WARN, f"tier_fs 有 {len(zero)} 列的 fs_dir_bytes 是 0（磁碟階沒被用到）")
    else:
        say(OK, "tier_fs 每一列都有實際寫入磁碟")
    # E3 M4：headroom 0.00% 代表工作負載對這張卡沒壓力
    z = []
    for f in glob.glob(str(R / "m4_oracle_mi300x/oracle_*.csv")):
        for r in rows(f):
            if r.get("policy") == "oracle":
                try:
                    if abs(float(r["oracle_headroom_pct"])) < 1e-9:
                        z.append((r.get("model_profile", ""), r.get("workload", "")))
                except (KeyError, ValueError):
                    pass
    if z:
        say(WARN, f"M4 有 {len(z)} 個工作負載的 headroom 恰為 0.00%（多半是塞得下、無決策可做）：{z[:4]}")
    # E4 M2：SSD 階的 warm 必須完整讀回一個前綴
    kv_kib = {"b-llama8b": 128.0, "b-ultralong8b-1m": 128.0, "b-qwen7b-1m": 56.0,
              "b-qwen3-30b-a3b": 96.0, "b-seedoss36b": 256.0,
              "b-qwen14b-1m": 192.0, "b-mistral-nemo12b": 160.0}
    for f in sorted(glob.glob(str(R / "m2_harness_mi300x/retrieval_cost_b-*.csv"))):
        rs = [r for r in rows(f) if r.get("tier") == "ssd" and r.get("round") == "warm"]
        if not rs:
            continue
        m, ctx = rs[0]["model_key"], int(rs[0]["ctx"])
        need = 0.95 * ctx * kv_kib.get(m, 128.0) * 1024
        short = [r for r in rs if int(r.get("req_read_bytes") or 0) < need]
        if short:
            say(WARN, f"{Path(f).name}：SSD 階有 {len(short)}/{len(rs)} 列讀回不足一個前綴"
                      f"（這種列不可當 SSD 成本用）")


def check_runs() -> None:
    print("\n── F. run 目錄：exit_code 非 0 ──")
    bad = []
    for p in sorted(RUNS.glob("*/exit_code")):
        try:
            code = p.read_text().strip()
        except OSError:
            continue
        if code not in ("0", ""):
            bad.append((p.parent.name, code))
    recent = bad[-12:]
    if bad:
        say(WARN, f"{len(bad)} 個 run 的 exit_code 非 0（多數已修正後重跑，最近幾個）：")
        for n, c in recent:
            print(f"      {n} → {c}")
    else:
        say(OK, "所有 run 的 exit_code 都是 0")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="有 FAIL 就回傳 1")
    a = ap.parse_args()
    print("=" * 74)
    print("平台 B 結果自檢")
    print("=" * 74)
    check_traceability()
    check_coverage()
    check_completeness()
    check_contamination()
    check_degenerate()
    check_runs()
    n_fail = sum(1 for lv, _ in issues if lv == FAIL)
    n_warn = sum(1 for lv, _ in issues if lv == WARN)
    print("\n" + "=" * 74)
    print(f"結論：{n_fail} 個 FAIL、{n_warn} 個 WARN")
    if n_fail:
        print("🔴 有 FAIL —— 這些結果不可引用，必須先處理。")
    elif n_warn:
        print("⚠️  沒有 FAIL，但有 WARN —— 多半是已知且已記錄的限制，引用前確認 RUNLOG 有寫。")
    else:
        print("✅ 全部通過。")
    return 1 if (a.strict and n_fail) else 0


if __name__ == "__main__":
    raise SystemExit(main())
