# ICPE 2027：論文主張 → 證據對照表（H200 補量）

> **狀態：骨架（2026-10-22 開檔）**。11/2 前填完；每一列一個主張、一個數字、平台、run_id、實測或模擬。
> 「待」＝ICPE 補量的實驗還沒跑完（括號是哪一個）；「待補 run_id」＝數字已知、run_id 還沒從 RUNLOG 抄過來核對。
> 規則：數字只抄原始檔或 RUNLOG 裡有 run_id 的；平台分開寫，不混成一個數；模擬一律標〔模擬〕。self-review，不是 cross-model review。
> 實驗與判準：[`icpe/README.md`](icpe/README.md)；執行紀錄：[`results/RUNLOG_H200_ICPE.md`](../../results/RUNLOG_H200_ICPE.md)。

## A. 寫入時預知＝0（第一、二階段）

| # | 主張 | 數字 | 平台 | run_id／來源 | 實測／模擬 | 狀態 |
|:--|:--|:--|:--|:--|:--|:--|
| A1 | 寫入時依位置決定（S5）沒有贏過延後版（S4+），主設定 | S5 對 S4+：+0.26%（rep 0–5） | MI300X | 待補 run_id（RUNLOG_MI300X M7） | 實測 | 待補 |
| A1' | 同上 | 合併組 −0.12%（δ\* 5%）；cpu 0.25／0.5／1.0 都在 ±δ\* | H200 | `20261016-063929-m7h-b-nfs-gap0`＋`20261016-172445-m7h-b-nfs-gap0-r36-rr`（`results/m7_write_policy_h200_merged/`） | 實測 | 已有 |
| A1'' | 同上，重算換成 H200 最快的 attention | 待 | H200 | 待（P0-1 (a)） | 實測 | 待 |
| A2 | 08 消融：寫入時決定同時贏最好的不看位置與 S4B 各 ≥5% | 0/16 | MI300X、H200 | 待補 run_id；H200 合併組 `results/m7_write_policy_h200_merged/b2_verdict.csv` | 實測 | 待補 |
| A3 | 第 1 輪模擬存活設定上 GPU 確認 | MI300X 0/5；H200：MI300X 那格 0/5、H200 自己 3 格 0/3 | 兩台 | H200 job 20660（`results/m7_explore_h200/r1_gpu_verdict_{A,B,C}.csv`） | 實測 | 待補 MI300X |
| A4 | 第二階段：依位置決定精度沒有幫助（K2） | MI300X：Llama INT4 Step 2 F／M／B 對 R25 在 ±0.33 內；H200：Qwen FP8 Step 2 → K2（F25 好是因為沒被讀到） | 兩台 | 待補 run_id（`docs/precision_20261013/report_stage2/REPORT.md`、`docs/h200/STAGE2_H200_VS_MI300X.md`） | 實測 | 待補 |
| A5 | 第二階段：每層最佳還原方式（CPU 層全部載入＋一次拆完；慢層 Cake）在重算變快時不翻轉 | 待 | H200 | 待（P0-1 (b)） | 實測 | 待 |

## B. 預知回來時間只在窄區域有用（第三階段 3b）

| # | 主張 | 數字 | 平台（常數） | run_id／來源 | 實測／模擬 | 狀態 |
|:--|:--|:--|:--|:--|:--|:--|
| B1 | X1 headroom 最大的設定：conversation、本地 SSD、CPU 128／256 GiB、GPU 閒、Cake | O-PF 48.9／44.6%、O-ORD 25.4／27.0%、O-DROP 27.0／32.8%（對最好的對稱對手，R1／R2 取小） | H200 常數 | `20261021-010140-p3h-x1-sweep`→`…-011301-p3h-x1-analyze`（`results/p3_explore_h200/x1/x1_headroom_components.csv`） | 〔模擬〕 | 已有 |
| B1' | 同上（MI300X 常數） | O-PF 44.5／40.2%、O-ORD 22.5／24.1%、O-DROP 22.8／28.2% | MI300X 常數 | `20261017-180517-p3x1-sweep`→`…-182005-p3x1-analyze`（3090 分支） | 〔模擬〕 | 已有 |
| B2 | 截尾後只剩預取與降級順序（「不回來就丟」大半是 trace 太短） | 待（H200 常數） | H200 常數 | 待（P0-1 (c) V0、P1-2 A1） | 〔模擬〕 | 待 |
| B3 | 換成真實 agent trace（FreeInference）空間消失 | MI300X 常數：最大 0.22%（死路）；H200：待 | 兩種常數 | 待（P1-2 A2） | 〔模擬〕 | 待 |
| B4 | 併發時 p99 尾端是排隊，不是預測 | H200 常數：排隊占 NP p99 中位 0.983（262 格）；平均值變「有看頭」的 63 格裡 43 格主要來自還原變快 | H200 常數 | `20261022-121515-icpe-p12-queue`（`results/icpe_h200/p12/x4/`） | 〔模擬〕 | 已有 |
| B5 | X5：值錢的是「知道誰不回來」，只在 CPU ≤256 GiB、慢層慢時 | H200 常數：有看頭，48 格裡 11 格 ≥15%，最大 37.4% | H200 常數 | `20261021-012558-p3h-x5-analyze`（`results/p3_explore_h200/x5/x5_verdict.json`） | 〔模擬〕 | 已有；事後版本待（P1-2 B） |
| B6 | X1 的模擬在 GPU 上站得住（實測÷模擬、排序、headroom 正負號） | 待 | H200 | 待（P0-2） | 實測 | 待 |
| B7 | 重算跟 vLLM 一樣快時，X1／X5 判定字不變 | 待 | H200 常數 | 待（P0-1 (c) V1、V2） | 〔模擬〕 | 待 |

## C. 預取用打字提示就夠

| # | 主張 | 數字 | 平台（常數） | run_id／來源 | 實測／模擬 | 狀態 |
|:--|:--|:--|:--|:--|:--|:--|
| C1 | 打字提示的提前量夠藏住聊天的慢層還原 | X3 判定「有看頭」（兩種常數相同） | H200 常數 | `results/p3_explore_h200/x3/x3_verdict.json`（待補 run_id） | 〔模擬〕 | 待補 |
| C2 | 打字提示預取（HINT-PF）拿到 oracle 預取的大部分 | 縮小設定（k＝8，CPU 16／32 GiB）：LRU 0.305／0.294 s、HINT-PF 0.244／0.244、O-PF 0.235／0.235 | H200 常數 | `20261022-120549-icpe-p02-sim-orig`（`results/icpe_h200/p02/sim_requests.csv.gz`） | 〔模擬〕 | 已有；全尺寸待（P0-2 fullsim），GPU 待（P0-2） |

## D. 不用預知就有用的：把位元組變小

| # | 主張 | 數字 | 平台 | run_id／來源 | 實測／模擬 | 狀態 |
|:--|:--|:--|:--|:--|:--|:--|
| D1 | INT4 容量 3.2 倍 | 每 block 20,971,520 vs 67,108,864 B | 兩台（算術） | `results/m10_tier_mi300x/iowall.json` | 算術 | 已有 |
| D2 | INT4 在慢層讓還原變快 | H200 21.5–54.5%；MI300X 22–57% | 兩台 | 待補 run_id（`STAGE2_H200_VS_MI300X.md` §3） | 實測 | 待補 |
| D3 | 低秩＋INT4 約 8 倍：Qwen 接近無損 | MI300X X4b：`xl:k34:v51+q:int4_kivi` 8.01 倍，對 E0 −0.7 [−3.5, +2.0]（MLPerf 99% 過） | MI300X | 待補 run_id（`results/m16_stage3_mi300x/x_stack/x4/x4_report.json`） | 實測 | 已有；H200 待（P1-1 Q） |
| D4 | 低秩＋INT4 在慢層比 INT4 快 | MI300X validate：SSD 級 2.5–3.4 倍；CPU 層 GPU 忙反而慢 | MI300X | `20261022-011846-m16-x3-validate2` | 實測 | 已有；H200 待（P1-1 T） |

## E. 效度威脅與跨平台

| # | 主張 | 數字 | 平台 | run_id／來源 | 實測／模擬 | 狀態 |
|:--|:--|:--|:--|:--|:--|:--|
| E1 | harness 的重算比 vLLM 慢 | 32K 只算 ÷ vLLM 預設：Llama 1.75、Qwen 1.76（MI300X 1.67／1.72） | H200 | job 21409（`20261021-025538-m10h-vllmg1-llama-default` 等 4 個 run） | 實測 | 已有 |
| E2 | 用 H200 最快的 attention 之後的 harness ÷ vLLM | 待 | H200 | 待（P0-1 S1） | 實測 | 待 |
| E3 | G5b 在 CPU 層、GPU 忙時選錯是穩定的平台差異 | H200 原值：12 格裡 10 格 G5b ÷ 最好 1.3–3.5 倍（整體 11.5%；MI300X 0.4%） | H200 | job 21109（`results/m10_restore2_h200/report_13.json`）；新 seed 待（P2） | 實測 | 待 |
