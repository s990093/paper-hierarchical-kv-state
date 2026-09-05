#!/usr/bin/env python3
"""模擬器的不變量檢查。**每一支掃描腳本開跑前都會呼叫。**

## 為什麼存在

2026-08-31 一天之內在同一支模擬器裡找到六個錯誤：

| # | 錯誤 | 被什麼抓到 |
|---|---|---|
| 1 | 模型混用（A 的預算 × B 的成本） | 人工比對兩支腳本的預設值 |
| 2 | Mooncake 粒度差 32 倍 | **同一個量被兩條路徑算出來且對不上** |
| 3 | SSD 階設成無限（工作集要 10.4 TiB，碟只有 7.3 TB） | **物理上不可能** |
| 4 | 寫入完全免費（需要 4,666 MiB/s，碟只有 181） | **物理上不可能** |
| 5 | 成本感知規則不是前綴感知的 | **邏輯上不可能**（headroom 為負） |
| 6 | `_smi()` 查詢失敗回傳空清單 = 「整機乾淨」 | 想到要移植 AMD 才發現 |

**沒有一個是讀程式碼看出來的。** 全部是靠「這個數字跟另一個數字對不上」
或「這在物理／邏輯上不可能」抓到的。

所以正確的防線不是更小心地讀 code，而是**把這些交叉檢查變成自動的**。
這支模組就是那條防線。

用法：
    from m4_invariants import preflight, check_results
    preflight(cm, trace, gpu_blocks, cpu_blocks, ssd_blocks, tname)
    ...跑模擬...
    check_results(res, trace, best, head)
"""
from __future__ import annotations
import json
import math
from pathlib import Path

BLOCK = 16
MOONCAKE_BLOCK = 512
TRACES = Path("/ssd7/hungwei/paper-hkv/datasets/traces")


class InvariantViolation(SystemExit):
    """不變量被違反。這一定是程式錯，不是實驗結果——中止，不要寫進 results/。"""


def _fail(msg: str) -> None:
    raise InvariantViolation(f"🔴 不變量違反\n{msg}\n"
                             f"這是程式錯誤，不是實驗結果。"
                             f"不要把任何相關數字寫進 results/ 或論文。")


def check_trace_units(name: str, trace: list[list[int]]) -> dict:
    """用資料自身驗算 trace 的 token 單位（CLAUDE.md 禁令 6）。

    模擬器內部的長度必須對得上 trace 檔裡 input_length 欄位的統計。
    對不上就代表粒度解錯了——而粒度解錯會讓每個 block 的**絕對位置**跟著錯，
    重算成本是位置的線性函數，所以整個成本模型會系統性偏移。
    """
    p = TRACES / f"{name}_trace.jsonl"
    if not p.exists():
        return {"checked": False, "reason": f"{p} 不存在"}
    from statistics import median
    src = [json.loads(l)["input_length"] for l in p.open()]
    med_src = median(src)
    med_sim = median(len(r) for r in trace) * BLOCK
    if not 0.95 * med_src <= med_sim <= 1.05 * med_src:
        _fail(f"trace「{name}」的長度單位對不上：\n"
              f"  檔案 input_length 的中位數 = {med_src:,.0f} token\n"
              f"  模擬器算出來的中位數     = {med_sim:,.0f} token\n"
              f"  比值 {med_sim / med_src:.2f}×"
              f"（若接近 {MOONCAKE_BLOCK // BLOCK} 或 {BLOCK / MOONCAKE_BLOCK:.4f}，"
              f"就是 hash_id 的 512-token 粒度沒有展開）")
    return {"checked": True, "median_src": med_src, "median_sim": med_sim}


def check_cost_extrapolation(cm, trace: list[list[int]]) -> dict:
    """重算成本的位置係數是擬合出來的；工作負載若超出擬合範圍就是外插。"""
    fit = cm.source.get("position_fit_max_tokens", 0)
    used = max(len(r) for r in trace) * BLOCK
    n_over = sum(1 for r in trace for i in range(len(r)) if i * BLOCK > fit)
    n_tot = sum(len(r) for r in trace)
    out = {"fit_max": fit, "max_used": used,
           "extrapolation_x": round(used / fit, 2) if fit else None,
           "pct_accesses_extrapolated": round(100 * n_over / n_tot, 1)}
    if fit and used > fit:
        print(f"  ⚠️ 重算成本外插：擬合上限 {fit:,} token、工作負載最大位置 "
              f"{used:,} token（{used / fit:.1f}×），"
              f"{out['pct_accesses_extrapolated']}% 的存取落在外插區")
    return out


def check_capacity_physical(ssd_blocks: int, uniq: int, bytes_per_block: int,
                            fs_root: str = "/ssd7") -> dict:
    """宣告的階層容量在這台機器上放得下嗎。"""
    import shutil
    need_tib = uniq * bytes_per_block / 1024**4
    try:
        du = shutil.disk_usage(fs_root)
    except OSError:
        return {"checked": False}
    have_tib = du.total / 1024**4
    ssd_tib = ssd_blocks * bytes_per_block / 1024**4
    out = {"working_set_tib": round(need_tib, 2),
           "device_total_tib": round(have_tib, 2),
           "declared_ssd_tib": round(ssd_tib, 2),
           "ssd_exceeds_device": ssd_tib > have_tib}
    if ssd_tib > have_tib:
        print(f"  ⚠️ 宣告的 SSD 階 {ssd_tib:,.1f} TiB > {fs_root} 的 "
              f"{have_tib:.1f} TiB。這是**不可實作**的設定，"
              f"只能當作上界參考，不可作為主結果。")
    return out


def check_write_feasibility(writes_ssd: int, bytes_per_block: int,
                            duration_s: float | None,
                            device_mibps: float, tag: str = "") -> dict:
    """把「寫了幾個 block」換算成需要的持續寫入頻寬，與實測裝置能力比較。

    成本模型只向「讀回來」收費，寫下去是免費的。所以一個策略可能在模擬裡
    很快、在真機上根本寫不下去。這個檢查不會中止程式——它產生一個
    必須跟著數字一起呈現的可行性標籤。
    """
    if not duration_s:
        return {"checked": False}
    mibps = writes_ssd * bytes_per_block / 1024**2 / duration_s
    feasible = mibps <= device_mibps
    if not feasible:
        print(f"  🔴 {tag} 需要 {mibps:,.0f} MiB/s 的持續 SSD 寫入，"
              f"裝置只有 {device_mibps:,.0f} MiB/s（超出 {mibps / device_mibps:.1f}×）"
              f" → 這個策略在真機上跑不起來")
    return {"ssd_write_mibps": round(mibps, 1),
            "device_write_mibps": device_mibps, "feasible": feasible}


def check_oracle_dominates(res: dict, best: str) -> None:
    """Oracle 是上界：它不可以輸給任何 baseline。"""
    o = res["oracle"]["total_ms"]
    for k, v in res.items():
        if k == "oracle":
            continue
        if v["total_ms"] < o - 1e-6:
            _fail(f"Oracle（{o:,.2f} ms）輸給 baseline {k}（{v['total_ms']:,.2f} ms）。\n"
                  f"  Oracle 知道全部的未來，不可能比線上策略差。\n"
                  f"  最可能的原因：目的地規則用的邊際成本估計不對"
                  f"（例如沒有把前綴語意的『尾巴要重算』算進去）。")


def check_recompute_floor(res: dict, trace: list[list[int]]) -> None:
    """每個不重複 block 至少要算一次——沒有任何策略能低於這個下限。"""
    floor = len({b for r in trace for b in r})
    for k, v in res.items():
        if v["hits"]["drop"] < floor:
            _fail(f"策略 {k} 的重算次數 {v['hits']['drop']:,} < 強制未命中下限 "
                  f"{floor:,}。每個 block 至少要被算出來一次，這在物理上不可能。")


def check_hits_conserved(res: dict, trace: list[list[int]]) -> None:
    """命中數的總和必須等於存取總數——不多不少。"""
    n = sum(len(r) for r in trace)
    for k, v in res.items():
        s = sum(v["hits"].values())
        if s != n:
            _fail(f"策略 {k} 的命中總數 {s:,} != 存取總數 {n:,}"
                  f"（差 {s - n:+,}）。有存取被漏記或重複記了。")


def check_capacity_monotone(points: list[tuple[float, float]],
                            tag: str = "") -> dict:
    """**多給一階容量，oracle 不可以變慢。**

    這是「最佳策略」的定義直接推出來的：多一個可用資源，最差也可以不用它。
    所以 oracle 的 `total_ms` 必須隨任一階的容量**單調不增**。

    🔴 2026-09-05 這條檢查抓到的事：qwen-awq/NVMe 下，把 SSD 階由 0 開到 512 GiB，
       `run_oracle` 反而**慢 3.3%**。原因是 `cost-aware` 的目的地規則拿「整條尾巴」
       當丟棄的邊際成本，而那條尾巴是該請求所有缺塊共用的——重複計價使門檻過低，
       於是把大量 block 寫上 SSD，再用 10.245 ms 讀回，而重算只要 3.5–4 ms。
       修法是新增 `dest="marginal"`（逐 block 計價）並讓 `dest="best"` 取三者較佳。

    不中止：違反代表**oracle 的構造不夠緊**（headroom 是下界），
    不代表這一輪的量測是錯的。但它必須被看見並記錄。
    """
    out = {"checked": True, "violations": []}
    pts = sorted(points)
    for (c0, t0), (c1, t1) in zip(pts, pts[1:]):
        if t1 > t0 + 1e-6:
            out["violations"].append(
                {"from": c0, "to": c1, "ms_from": t0, "ms_to": t1,
                 "worse_pct": round(100 * (t1 - t0) / t0, 3)})
    if out["violations"]:
        v = out["violations"]
        print(f"  🔴 容量單調性被違反{(' ' + tag) if tag else ''}："
              f"{len(v)} 處。oracle 多拿到容量卻變慢——")
        for x in v:
            print(f"     {x['from']} -> {x['to']}：{x['ms_from']:,.0f} -> "
                  f"{x['ms_to']:,.0f} ms（慢 {x['worse_pct']:.2f}%）")
        print("     這代表 oracle 的構造不是最優的，headroom 是**下界**。"
              "見 PAPER_DELTAS.md B7。")
    else:
        print(f"  ✅ 容量單調性通過{(' ' + tag) if tag else ''}"
              f"（{len(pts)} 個點）")
    return out


def preflight(cm, trace, tname, gpu_blocks, cpu_blocks, ssd_blocks,
              bytes_per_block, fs_root="/ssd7") -> dict:
    """跑模擬之前的全部檢查。回傳的字典應該原封不動寫進結果 CSV。"""
    print("[不變量] 開跑前檢查……")
    uniq = len({b for r in trace for b in r})
    out = {
        "units": check_trace_units(tname, trace) if tname else {"checked": False},
        "extrapolation": check_cost_extrapolation(cm, trace),
        "capacity": check_capacity_physical(ssd_blocks, uniq, bytes_per_block,
                                            fs_root),
    }
    if uniq <= gpu_blocks:
        print(f"  ⚠️ 工作集 {uniq:,} blocks ≤ GPU 預算 {gpu_blocks:,} blocks："
              f"整個塞得下、不會逐出，所有策略必然相同。這個點沒有鑑別力。")
    print("[不變量] 開跑前檢查通過。")
    return out


def check_results(res: dict, trace: list[list[int]], best: str) -> None:
    """跑完之後的全部檢查。任何一項失敗都代表模擬器有錯。"""
    check_hits_conserved(res, trace)
    check_recompute_floor(res, trace)
    check_oracle_dominates(res, best)


if __name__ == "__main__":
    print(__doc__)
