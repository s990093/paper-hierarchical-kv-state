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
| T0 | [T0_arrivals.md](T0_arrivals.md) | `code/s3/kvs_t0.py` | `results/s3_h200/kvs_t0/` | **完成**（10/26，run `20261026-140345-s3-kvs-t0`）：單位檢查 PASS（toolagent 0.389） |
| S0（S0-2） | [S0_union.md](S0_union.md) | `code/s3/kvs_s0.py`；runner part `kvs-s0` | `results/s3_h200/kvs_s0/` | L128 已送出（job 22455，10/26，排隊中）；其他 3 格等使用者決定 |
| F0 | [F0_follower_sim.md](F0_follower_sim.md) | `code/s3/kvs_f0.py` | `results/s3_h200/kvs_f0/` | 尺準 **PASS**；決定性實驗 **CONTINUE**〔模擬〕（10/26）；全網格等使用者決定 |
| F0P（F0＋兩個優先級：優先權反轉） | [F0P_priority.md](F0P_priority.md) | `code/s3/kvs_f0p.py`（import `kvs_f0`，不改它） | `results/s3_h200/kvs_f0p/` | **已登記**（10/26）；偏離 1：改在 **H200 head node** 跑（使用者 10/26「不要在 mac 跑」），Mac 上的決定性 run 作廢、全網格中止；**等 H200 跑** |
| R0（復現 Strata、KVFlow） | [R0_reproduce.md](R0_reproduce.md) | — | — | **計畫，未登記**：先查可行性 |

## 3. 共同規則

- 平台：H200（hgx-cn01）與 MI300X（只有既有常數，租約 10/22 已結束）。3090 不做多租戶結論（CLAUDE.md §8）。
- Mooncake trace 的 `hash_ids`＝512 token；載入時用 `input_length` 驗算（CLAUDE.md §1 規則 6，沿用 `code/m8_trace_oracle.py load_trace`）。
- 所有 CSV 都有 `run_id`、`ts`；大輸出放 run 目錄（`$TIARA_RUNS/<run_id>/`），results 只放摘要。
- 這三個閘都**不量時間**（T0、F0 是 CPU；S0-2 是品質型），所以不套用 05 §2.1 的速度雜訊規則。F0 的結論是〔模擬〕，要 F1 在真引擎上驗證才算數。
- 文獻與這幾份登記都是 self-review，不是 cross-model review（CLAUDE.md §5）。

## 4. 下一步（怎麼跑）

**使用者 2026-10-26：三個閘全部在 H200 跑**（F0 雖然是純 Python、任何 CPU 都能跑，也在 H200 head node 跑，紀錄一律進 `results/RUNLOG_H200_S3.md` 的 `S3-KVS-*` 節）。

| 閘 | 在哪跑 | 指令 | 時間 |
|:--|:--|:--|:--|
| F0 | H200 head node（純 Python，不用 GPU、不用 conda） | `python code/s3/kvs_f0.py cal` → `run --slice decisive` → （沒 STOP 才）`run --slice full --jobs 16` → `report` | 決定性幾分鐘；全網格單核約 40 分鐘〔估計〕 |
| T0 | H200 head node（trace 在 `$TIARA_DATASETS/traces`） | `source code/env_h200.sh; python code/s3/kvs_t0.py run` | 幾分鐘 |
| S0-2 | H200 GPU（品質型 job） | `sbatch --job-name=s3-kvs-s0-L128-0 --time=4:00:00 code/r_h200_s3.sh --part kvs-s0 --cell L128 --shard 0/1`；送出時照 skill §4 加進 ICPE 的依賴 | 約 1 GPU 小時（L128）〔估計〕 |
