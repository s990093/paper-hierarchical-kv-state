# P1-2　便宜的 CPU 對稱重跑：3b 的事後版本在 H200 常數下還站得住嗎？

> ICPE 補量 P1-2（[README](README.md)）。只用 CPU（包在 hgx-cn01 的 1-GPU job 裡，不和量時間的 job 同時跑）。**§1–§5 是事先登記**，結果寫在 §6 以後。
> 任務原文：「X1 截尾（H＝1,800／3,600 s）與 FreeInference、X5 事後版本、X4 的 f×0.5；X4『排隊占比』的拆解（先寫判準再算）」。任務說這幾項「H200 報告 §8 已列」，**我在兩個 checkout（`h200`、`h200-p3`）都找不到那份報告的 §8**，所以照任務文字列的四項做；若那份 §8 還有別的項目，要使用者給檔名再補。
> **登記寫於 2026-10-22T04:14:28Z（`date -u`），repo HEAD `b0f0fba`**；寫的時候沒有任何這個實驗的資料（P1-2 的 D 讀的 H200 X4 掃描早就跑完了，但「排隊占比」與「平均值拆解」這兩個數我還沒算、也沒看）。
> 標記：〔模擬〕〔資料〕〔判讀〕同 repo 慣例。全部是模擬；H200 常數＝`P3_CONST=h200`（第一、二階段 H200 實測；限速層與位元組照 MI300X，03 §R2）。

---

## 1. 為什麼要做

3b（`docs/phase3_20261017/explore/`）在 H200 上只重跑了**原登記**的主結果（`results/p3_explore_h200/`，RUNLOG_H200_P3 P3H-1d）。論文的說法「預知回來時間只在窄區域有用」靠的有好幾個是**事後版本**（截尾、FreeInference、只載入、f×0.5、排隊占尾端），那些只在 3090 分支用 MI300X 常數跑過。這裡用 H200 常數把它們對稱地重跑一次，看判定字和方向是否相同。

## 2. 要跑的（照 3090 分支原本的指令，只換常數與輸出目錄）

輸出一律在 `results/icpe_h200/p12/<x>/`（不覆寫 `results/p3_explore_h200/`）。X1／X5／X4 經 `code/icpe/p3_fvar.py --variant V0`（只改輸出目錄，常數＝H200 原樣；P0-1 已驗證 V0 和 git 逐值相同）。

| # | 項目 | 指令（3090 的出處） | 3090（MI300X 常數）的判定 |
|:--|:--|:--|:--|
| A1 | X1 截尾 H＝1,800／3,600 s（Mooncake 兩份、R1／R2 取小） | **和 P0-1 (c) 的 V0 是同一個計算**：`x1_sim.py analyze --dead-h 1800／3600`，輸出在 `results/icpe_h200/p01/sim_V0/x1/x1h{1800,3600}_*`；這裡只引用、不重跑 | 兩個都「有看頭」（H1800：21 組；H3600：20 組，只剩 O-ORD、O-PF）（X1 §4.7） |
| A2 | X1 的 FreeInference（前 6 週，2026-05-17 到 06-27，只跑 Cake） | `x1_sim.py sweep --traces fi:<6 週> --rules session_id --modes cake --out x1_sweep_fi.csv`；`analyze --sweep x1_sweep_fi.csv --traces freeinference --out-prefix x1fi`（另一個 `--metric mean_ret_user --out-prefix x1fiuser`）（X1 §3） | 死路（最大 0.22%；只算使用者發起 1.06%）（X1 §4.6） |
| B1 | X5 嚴格重建規則（reg） | `x5_run.py --rule strict --variants reg`；`x5_analyze.py --sfx _strict` | X5 §4.3 |
| B2 | X5 FreeInference（reg） | `x5_run.py --fi`；`x5_analyze.py --sfx _fi` | 全部 <1%（X5 §4.7） |
| B3 | X5 設限穩健性：子樹 H1800、嚴格 H1800、FI H1800、FI H3600（主設定） | `x5_run.py --censor-H <H> --main-only [--rule strict｜--fi]`；`x5_analyze.py --sfx _<…>_cens<H>` | X5 §4.8 |
| B4 | X5 只載入還原（主設定） | `x5_run.py --variants loadonly --main-only`；`x5_analyze.py --variant loadonly --sfx _loadonly` | X5 §4.9 |
| B5 | X5 只算 ≤ 32K 的回來請求 | `x5_short_only.py --sets "subtree::<主格子 raw>" "strict:_strict:<B1 raw>"` | X5 §4.5 |
| C1 | X4 f×0.5（p＝0.3／0.4／0.5，寬鬆＋嚴格） | `x4_sim.py sweep --rules loose strict --fscale 0.5 --p 0.3 0.4 0.5 --out x4_sweep_f05.csv`；`analyze --inp <…> --out-seed x4_f05_verdict_seed.csv --out-cfg x4_f05_verdict_cfg.csv` | p99：寬鬆「可能」（1 格 5.34%）、嚴格「死路」→ 照較保守＝死路（X4 §4.5） |
| D | X4「排隊占比」的拆解 | 新的分析（§3），讀 H200 已跑的 X4 主掃描（run `20261021-012002-p3h-x4-sweep` 的 `x4_sweep.csv`、`results/p3_explore_h200/x4/x4_verdict_cfg.csv`），不重跑模擬 | 3090：NP 的排隊占 p99 中位 0.995、最小 0.939（X4 §4.4） |

- FreeInference 的資料在 hgx-cn01 本地 `/tmp/hungweilai-p3/datasets`（P3 移植時下載、sha256 核對過）；開跑前照 `r_h200_p3.sh` 的 `need_data`＋`h200_p3_verify_data.py` 再核一次，不對就停。
- 每一項的判定規則照它在 3090 分支原本的寫法（各 X 文件），不改。

## 3. D：「排隊占比」的拆解（判準寫在算之前）

H200 的 X4 原登記判定：p99 仍是「死路」，但**平均、p90 從「可能」變「有看頭」**（RUNLOG_H200_P3：可進判定的格 137 → 262，因為 H200 的 prefill 快、使用率較低）。要問兩件事：

- **D1（尾端是不是排隊）**：每個可進判定的格（`eligible`＝True）算 share＝`NP_qwait_p99_med ÷ NP_p99_med`。
  - **判準**：share 的中位數 ≥ 0.90 → 「H200 上 X4 的 p99 尾端仍幾乎全是排隊」成立；< 0.90 → 不成立。另報最小值與分布。
- **D2（平均值的空間從哪裡來）**：在可進判定、而且 `hr_mean_med ≥ 5%` 的格（H200 讓平均變「有看頭」的那些格）裡，對每個 seed 拆「最好對手（該格平均的最好對手）− OP」的平均 TTFT 差：Δ＝Δ排隊（`qwait_mean`）＋Δ還原（`restore_mean`）＋Δ其餘（新一輪 prefill 等＝Δ − 前兩項）；每格取 seed 中位數。
  - **判準**：這些格裡 ≥ 50% 的格「Δ還原 ÷ Δ ≥ 0.5」→ 「平均值的空間主要來自還原變快（預取本身）」；否則 → 「主要來自排隊變短（還原變快後，後面的人少等）」。另報 p90 的同樣拆解（只用 p99 以外沒有逐項的 p90，所以 p90 只報 `hr_restore_p99_med` 等現成欄，不判）。
- 只報告：QP（不用預測的佇列預取）對 NP 的平均與 p99 改善，和 OP 並排（「不用預測就拿到多少」）。

## 4. 總判定

P1-2 不是 go/no-go。每一項寫「H200 的判定字／數字」與「3090 分支（MI300X 常數）的判定字」並排，標 **一致／不一致**；不一致的照列、寫可能原因（〔判讀〕）。D 照 §3 的兩條判準寫成立／不成立。

## 5. 停損、失敗、時間

- 任何指令失敗：完整錯誤記下、停下來回報。資料核對不過就停（不重新下載以外的方式湊）。
- 跑在 P0 的 job 之後（`afterany`），不和量時間的 job 重疊。
- 時間〔估計〕：A2 約 30 分（3090 上 FI sweep 7 個 worker 的時間 NOT_MEASURED，這裡 14 個）；B 約 30–40 分；C1 約 20 分；D 幾分鐘（head node 就能算，只讀 CSV）。

---

## 6. 結果

（還沒有。）

## 7. 偏離登記

（還沒有。）
