# 共享 × 隱藏 × 排程：run 文件（登記）

> 回到核心頁：[../README.md](../README.md)
> 規矩照第三階段（`docs/stage3_design_20261024/runs/README.md`、skill `stage3-h200`）：每個閘一個 md，**最上面是事先登記的判準與量法，commit＋`date -u` 之後不改**；之後的修改一律寫成文末的「偏離 N」。冒煙只驗程式、不進 results、不看分數。沒量到寫 `NOT_MEASURED`。

## 1. 先做決定性的小實驗，過了才放大

| 候選 | 決定性實驗（先跑） | 多小 | 當場就能停的條件 | 過了才做的放大 |
|:--|:--|:--|:--|:--|
| A：多人共用的縮圖還原 | S0-2 只跑 **L128**（Llama-3.1-8B、128K） | 96 題、約 1 GPU 小時〔估計〕 | L128 兩種粒度的 u(2) 中位數都 > 0.85 → STOP-S，其他格不跑 | S0-2 其餘 3 格 → A0 GO 之後 S0-3（K≤8、3 個模型、真實資料） |
| B：跟隨者的等／載／算 | F0 的 **12 格切片**（H200、Llama；3 個 tier × 2 個長度 × 2 種 GPU 負載；N＝16） | CPU 幾分鐘 | 12 格的最佳固定規則全部相同，**而且** P6 在 12 格都贏不到 5% → STOP-F，全網格不跑 | F0 全網格（2 平台 × 2 模型 × 324 格）→ F1（H200 真引擎） |
| 輔助 | T0 trace 統計 | CPU 幾分鐘 | 單位檢查不過（重現不了 Strata 的量級）→ 停下查單位 | — |

**為什麼這樣排**：決定性實驗挑的是「最可能讓它死」的那一格。A 的關鍵是聯集大不大，用最普通的 Llama 128K 先看；B 的關鍵是「固定規則會不會翻轉」，切片涵蓋最會翻轉的兩個軸（tier 與 GPU 負載）。

## 2. 狀態

| 閘 | 文件 | 程式 | 結果 | 狀態 |
|:--|:--|:--|:--|:--|
| T0 | [T0_arrivals.md](T0_arrivals.md) | `code/s3/kvs_t0.py` | `results/s3_h200/kvs_t0/` | 已登記（2026-10-26），程式 CPU 自測 |
| S0（S0-2） | [S0_union.md](S0_union.md) | `code/s3/kvs_s0.py`；runner part `kvs-s0` | `results/s3_h200/kvs_s0/` | 已登記（2026-10-26），程式 CPU 自測；GPU 冒煙未跑 |
| F0 | [F0_follower_sim.md](F0_follower_sim.md) | `code/s3/kvs_f0.py` | `results/s3_h200/kvs_f0/` | 已登記（2026-10-26），程式 CPU 自測 |
| R0（復現 Strata、KVFlow） | [R0_reproduce.md](R0_reproduce.md) | — | — | **計畫，未登記**：先查可行性 |

## 3. 共同規則

- 平台：H200（hgx-cn01）與 MI300X（只有既有常數，租約 10/22 已結束）。3090 不做多租戶結論（CLAUDE.md §8）。
- Mooncake trace 的 `hash_ids`＝512 token；載入時用 `input_length` 驗算（CLAUDE.md §1 規則 6，沿用 `code/m8_trace_oracle.py load_trace`）。
- 所有 CSV 都有 `run_id`、`ts`；大輸出放 run 目錄（`$TIARA_RUNS/<run_id>/`），results 只放摘要。
- 這三個閘都**不量時間**（T0、F0 是 CPU；S0-2 是品質型），所以不套用 05 §2.1 的速度雜訊規則。F0 的結論是〔模擬〕，要 F1 在真引擎上驗證才算數。
- 文獻與這幾份登記都是 self-review，不是 cross-model review（CLAUDE.md §5）。
