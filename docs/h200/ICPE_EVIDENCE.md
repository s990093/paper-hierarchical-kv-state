# ICPE 2027：論文主張 → 證據對照表（H200 補量）

> **狀態：填寫中（2026-10-22 開檔；10/25 補上 MI300X 與已有 H200 列的 run_id）**。11/2 前填完；每一列一個主張、一個數字、平台、run_id、實測或模擬。
> 「待」＝ICPE 補量的實驗還沒跑完（括號是哪一個）；「待補 run_id」＝數字已知、run_id 還沒從 RUNLOG 抄過來核對。
> 規則：數字只抄原始檔或 RUNLOG 裡有 run_id 的；平台分開寫，不混成一個數；模擬一律標〔模擬〕。self-review，不是 cross-model review。
> 實驗與判準：[`icpe/README.md`](icpe/README.md)；執行紀錄：[`results/RUNLOG_H200_ICPE.md`](../../results/RUNLOG_H200_ICPE.md)。

## A. 寫入時預知＝0（第一、二階段）

| # | 主張 | 數字 | 平台 | run_id／來源 | 實測／模擬 | 狀態 |
|:--|:--|:--|:--|:--|:--|:--|
| A1 | 寫入時依位置決定（S5）沒有贏過延後版（S4+），主設定 | S5 對 S4+（回來請求 TTFT 中位數，正＝S5 快）cpu 0.25／0.5／1.0 ＝ +0.07%／+0.26%／−0.30%（rep 6／6／3），δ\* 6.17% | MI300X | `20261008-142647-m7-b-nfs-gap0`（rep 0–2）＋`20261008-161514-m7-b-nfs-gap0-r36`（rep 3–5）→ `results/m7_write_policy_mi300x/summary_b_verdict.csv`（main） | 實測 | 已有 |
| A1' | 同上 | 合併組 −0.12%（δ\* 5%）；cpu 0.25／0.5／1.0 都在 ±δ\* | H200 | `20261016-063929-m7h-b-nfs-gap0`＋`20261016-172445-m7h-b-nfs-gap0-r36-rr`（`results/m7_write_policy_h200_merged/`） | 實測 | 已有 |
| A1'' | 同上，重算換成 H200 最快的 attention（fa3） | +0.07%／−0.47%／+0.03%（rep 6，δ\* 5%）→ 三格都在 ±δ\* → 成立 | H200 | 資料 `20261022-210344-icpe-p01a-b-nfs-gap0`（job 21654）→ 分析 `20261022-234955-icpe-p01a-analyze`（`results/icpe_h200/p01/p01a_{table.csv,verdict.json}`） | 實測 | 已有 |
| A2 | 08 消融：寫入時決定同時贏最好的不看位置與 S4B 各 ≥5% | GPU 列 0/16（兩台）；MI300X 的模擬列 1/16（doc、cpu 0.5、S5L），GPU 沒確認 | MI300X、H200 | MI300X `20261009-070326-m7-b2-g1-doc-hold-cpu3.69-s0`、`…-090235-…-s1`、`…-102010-m7-b2-g2-chat-hold-s0`（`results/m7_write_policy_mi300x/b2_verdict.csv`）；H200 `20261016-132601-m7h-b2-g1-doc-hold-cpu3.69-s1`、`…-142019-m7h-b2-g2-chat-hold-s0`、`…-202603-m7h-b2-g1-doc-hold-cpu3.69-s0-rr`（`results/m7_write_policy_h200_merged/b2_verdict.csv`） | 實測 | 已有 |
| A3 | 第 1 輪模擬存活設定上 GPU 確認 | MI300X 0/5 seed（判準 ≥3/5、四個對手各 ≥5%；Llama、doc、free、本地 SSD、CPU 50%）；H200：MI300X 那格 0/5、H200 自己 3 格 0/3 | 兩台 | MI300X `20261009-183252-r1-gpu-llama-doc-free-local-s0`、`…-192035-…-s1`、`…-195129-…-s2`、`…-202956-…-s3`、`…-210351-…-s4`（`results/m7_explore_mi300x/r1_gpu_verdict.csv`）；H200 job 20660（`results/m7_explore_h200/r1_gpu_verdict_{A,B,C}.csv`） | 實測 | 已有 |
| A4 | 第二階段：依位置決定精度沒有幫助（K2） | MI300X：Llama INT4 Step 2 F／M／B 對 R25 在 ±0.33 個百分點內 → K2；H200：Qwen FP8 Step 2 → K2（F25 品質好是因為沒被讀到，且比 R25 慢 8–10% > δ\*） | 兩台 | MI300X `20261016-102921-m10-i4m-llama-T24-s0`～`…-103536-…-s5`、`…-115740-m10-i4m-llama-T5`（`results/m10_precision_mi300x/analysis/int4/step2_llama_int4.json`）；H200 `20261018-074632-m10h-f8m-qwen-T24-s0`～`…-085357-…-s5`、`…-092427-m10h-f8m-qwen-T5`（`results/m10_precision_h200/analysis/fp8/step2_qwen_seedA.json`） | 實測 | 已有 |
| A5 | 第二階段：每層最佳還原方式（CPU 層全部載入＋一次拆完；慢層 Cake）在重算變快時不翻轉 | **暫定**：24 格 0 翻轉；1 格弱變化（Llama CPU 層閒 E0：sdpa 平手 → fa3 全部載入較快）；慢層每格仍是 Cake（fa3 的 Cake 快 4–19%） | H200 | job 21654 → `20261022-235022-icpe-p01b-analyze-provisional`（`results/icpe_h200/p01/p01b_verdict.json`）；Qwen fa3 L501 兩個 run 重疊 → 重跑 job 22233（p01brr） | 實測 | 暫定（待 p01brr） |

## B. 預知回來時間只在窄區域有用（第三階段 3b）

| # | 主張 | 數字 | 平台（常數） | run_id／來源 | 實測／模擬 | 狀態 |
|:--|:--|:--|:--|:--|:--|:--|
| B1 | X1 headroom 最大的設定：conversation、本地 SSD、CPU 128／256 GiB、GPU 閒、Cake | O-PF 48.9／44.6%、O-ORD 25.4／27.0%、O-DROP 27.0／32.8%（對最好的對稱對手，R1／R2 取小） | H200 常數 | `20261021-010140-p3h-x1-sweep`→`…-011301-p3h-x1-analyze`（`results/p3_explore_h200/x1/x1_headroom_components.csv`） | 〔模擬〕 | 已有 |
| B1' | 同上（MI300X 常數） | O-PF 44.5／40.2%、O-ORD 22.5／24.1%、O-DROP 22.8／28.2% | MI300X 常數 | `20261017-180517-p3x1-sweep`→`…-182005-p3x1-analyze`（3090 分支） | 〔模擬〕 | 已有 |
| B2 | 截尾後只剩預取與降級順序（「不回來就丟」大半是 trace 太短） | H200 常數、V0、rule＝min、384 格：不截尾 O-DROP 最大 45.9%（48 格 ≥15%）→ 截尾 H＝3,600 s 0.0%（0 格）；O-PF 91.8%、O-ORD 99.9% 不變。判定字 H1800／H3600 都「有看頭」（18／18 組，H3600 只有 O-ORD、O-PF），和 3090 分支（21／20 組）一致。只報告：H＝1,800 的 O-DROP 最大 22.6% 但 0 個熱組 | H200 常數 | `20261025-194554-icpe-p01c-x1-h3600-V0`、`…-194549-…-x1-h1800-V0`；並排 `20261025-215527-icpe-p12-compare`（`results/icpe_h200/p12/p12_compare.csv`） | 〔模擬〕 | 已有 |
| B3 | 換成真實 agent trace（FreeInference）空間消失 | X1：H200 最大 0.28%（只算使用者發起 1.48%）、MI300X 常數 0.22%（1.06%）→ 兩邊死路；X5：H200 最大 2.44%、MI300X 常數 0.64% → 死路（設限 H1800／3600 同） | 兩種常數 | H200 `20261025-203058-icpe-p12-x1fi-an`、`…-203104-…-x1fiuser-an`、`…-203614-icpe-p12-x5-fi-an`；MI300X 常數 `20261017-213544-p3x1-analyze-fi`、`…-213548-…-fiuser`、`20261017-220031-p3x5-analyze-fi` | 〔模擬〕 | 已有 |
| B4 | 併發時 p99 尾端是排隊，不是預測 | H200 常數：排隊占 NP p99 中位 0.983（262 格）；平均值變「有看頭」的 63 格裡 43 格主要來自還原變快 | H200 常數 | `20261022-121515-icpe-p12-queue`（`results/icpe_h200/p12/x4/`） | 〔模擬〕 | 已有 |
| B5 | X5：值錢的是「知道誰不回來」，只在 CPU ≤256 GiB、慢層慢時 | H200 常數：有看頭，48 格裡 11 格 ≥15%，最大 37.4%。事後版本（H200）：嚴格規則 20 格／48.6%、設限子樹 H1800 11 格／37.1%、只載入 14 格／59.6%、只算 ≤32K 子樹 11→9 格、嚴格 20→15 格 → 判定字全部和 3090 分支一致 | H200 常數 | `20261021-012558-p3h-x5-analyze`（`results/p3_explore_h200/x5/x5_verdict.json`）；事後 `20261025-203333-icpe-p12-x5-strict-an`、`…-205114-…-cens1800-an`、`…-210002-…-loadonly-an`、`…-210316-icpe-p12-x5-le32k`；並排 `20261025-215527-icpe-p12-compare` | 〔模擬〕 | 已有 |
| B6 | X1 的模擬在 GPU 上站得住（實測÷模擬、排序、headroom 正負號） | 待 | H200 | 待（P0-2） | 實測 | 待 |
| B7 | 重算跟 vLLM 一樣快時，X1／X5 判定字不變 | X1 原登記、X1 截尾 H＝3,600、X5 登記：V0／V1（fa3 實測）／V2（vLLM 級，模擬）全部「有看頭」；V0 和 git 逐值相同。X5 ≥15% 的格數 11／16／17（重算變快空間變大） | H200 常數 | `20261025-202422-icpe-p01c-analyze`（`results/icpe_h200/p01/p01c_verdict.json`；資料 job 21656） | 〔模擬〕 | 已有 |

## C. 預取用打字提示就夠

| # | 主張 | 數字 | 平台（常數） | run_id／來源 | 實測／模擬 | 狀態 |
|:--|:--|:--|:--|:--|:--|:--|
| C1 | 打字提示的提前量夠藏住聊天的慢層還原 | X3 判定「有看頭」：原登記 36/36 格、加 FreeInference 37/54 格；最好的格（bailian traceA、打字提示、NFS 級）時間加權藏住 98.7% | H200 常數 | `20261021-011752-p3h-x3-hide`（`results/p3_explore_h200/x3/x3_verdict.json`） | 〔模擬〕 | 已有 |
| C2 | 打字提示預取（HINT-PF）拿到 oracle 預取的大部分 | 全尺寸 24 格（conversation、R1、Cake、SSD／b32／NFS × 閒／忙 × 128–1,024 GiB）：HINT-PF 對 LRU 5.6–47.8%、O-PF 對 LRU 5.5–52.4%；HINT-PF÷O-PF 的改善比例中位 SSD 0.939、b32 0.949、NFS 1.117；6 格 >1（O-PF 騰空間照 LRU，不是上界〔推論〕）。縮小設定（k＝8）：LRU 0.305／0.294 s、HINT-PF 0.244／0.244、O-PF 0.235／0.235 | H200 常數 | 全尺寸 `20261025-202314-icpe-p02-fullsim`（`results/icpe_h200/p02/fullsim_hint.csv`）；縮小 `20261022-120549-icpe-p02-sim-orig` | 〔模擬〕 | 已有；GPU 待（P0-2） |

## D. 不用預知就有用的：把位元組變小

| # | 主張 | 數字 | 平台 | run_id／來源 | 實測／模擬 | 狀態 |
|:--|:--|:--|:--|:--|:--|:--|
| D1 | INT4 容量 3.2 倍 | 每 block 20,971,520 vs 67,108,864 B | 兩台（算術） | `results/m10_tier_mi300x/iowall.json` | 算術 | 已有 |
| D2 | INT4 在慢層讓還原變快 | INT4 對 BF16（E1 對 E0，限速層 b16／b32／b48，閒＋忙，每台 12 格）：H200 Llama 37.4–49.1%、Qwen 36.7–54.5%；MI300X Llama 36.1–55.2%、Qwen 38.1–56.9%；污染列 0 | 兩台 | H200 `20261017-094511-m10h-grid-int4-llama`、`20261017-103540-m10h-grid-int4-qwen`（`results/m10_precision_h200/speed_*/k0_verdict_int4.json`）；MI300X `20261016-034902-m10-grid-int4-llama`、`20261016-052143-m10-grid-int4-qwen`（`results/m10_precision_mi300x/speed_*/k0_verdict_int4.json`） | 實測 | 已有（舊列的 21.5–54.5%／22–57% 含 FP8，這裡只算 INT4） |
| D3 | 低秩＋INT4 約 8 倍：Qwen 接近無損 | MI300X X4b（32K，192 題，seed 65000）：Qwen `xl:k34:v51+q:int4_kivi` 8.01 倍 −0.7 [−3.5, +2.0]（MLPerf 99% 過、ε 沒過）；Llama `k29v44` 7.98 倍 −3.3 [−6.5, −0.4]、`k19v29` 12.14 倍 −3.6 [−6.9, −0.5]。**H200**（192 題，seed 67000 新題）：Qwen 主配置 `xl:k34:v51:cal+q:int4_kivi` 9.64 倍 −0.6 [−2.9, +1.8]（MLPerf 99% 過，邊際 0.08 個百分點）、每題 SVD 8.01 倍 −0.4 [−2.7, +1.8]；Llama `:cal` 11.22 倍 −0.6 [−3.3, +2.0]、17.07 倍 −0.7 [−3.4, +2.1]；Llama 每題 SVD 7.98 倍 +0.1、12.14 倍 +0.4（ε 過；和 MI300X CI 相容但判定字不同） | 兩台 | MI300X `20261022-155523-m16-x4-report`（main）；H200 `20261025-203913-icpe-p11-gen-qwen`、`20261025-210417-icpe-p11-gen-llama` → `20261025-215237-icpe-p11-report-q`（`results/icpe_h200/p11/p11q_report.json`） | 實測 | 已有 |
| D4 | 低秩＋INT4 在慢層比 INT4 快 | MI300X validate：SSD 級 2.5–3.4 倍；CPU 層 GPU 忙反而慢 | MI300X | `20261022-011846-m16-x3-validate2` | 實測 | 已有；H200 待（P1-1 T） |

## E. 效度威脅與跨平台

| # | 主張 | 數字 | 平台 | run_id／來源 | 實測／模擬 | 狀態 |
|:--|:--|:--|:--|:--|:--|:--|
| E1 | harness 的重算比 vLLM 慢 | 32K 只算 ÷ vLLM 預設：Llama 1.75、Qwen 1.76（MI300X 1.67／1.72） | H200 | job 21409（`20261021-025538-m10h-vllmg1-llama-default` 等 4 個 run） | 實測 | 已有 |
| E2 | 用 H200 最快的 attention 之後的 harness ÷ vLLM | 32K：Llama 1.574／1.2106＝1.30、Qwen 1.393／1.0590＝1.32（sdpa 時 1.75／1.76）；fa3 Σ64 f 是 sdpa 的 0.751／0.755 | H200 | job 21653：`20261022-205738-icpe-p01-attn-llama`、`…-205836-icpe-p01-attn-qwen`、`…-205953-icpe-p01-calibf-llama`、`…-210325-icpe-p01-calibf-qwen`（`results/icpe_h200/p01/attn/choice.json`、`calib/speed_*`）；vLLM 分母同 E1 | 實測 | 已有 |
| E3 | G5b 在 CPU 層、GPU 忙時選錯是穩定的平台差異 | H200 原值：12 格裡 10 格 G5b ÷ 最好 1.3–3.5 倍（整體 11.5%；MI300X 0.4%） | H200 | job 21109（`results/m10_restore2_h200/report_13.json`）；新 seed 待（P2） | 實測 | 待 |
