# ICPE 2027 的 H200 補量（2026-10-23 → 11-02）

> 任務：使用者 2026-10-22 的「ICPE 2027 論文的 H200 補量」。截止：摘要 11/9、全文 11/16（AoE）；**所有數字 11/2 前收齊**。
> 論文只回答一個問題：「KV 快取管理裡，先預知未來值多少？」這裡的每個實驗都只是替那個答案補證據或找反例。
> 平台：H200（hgx-cn01，Slurm），分支 `h200`。最後的總表：[`../ICPE_EVIDENCE.md`](../ICPE_EVIDENCE.md)（11/2 寫）。

## 規則（照舊，每個實驗都適用）

1. 每個實驗先寫事先登記（判準、停損、格子），**commit 之後才跑**；登記的時間用 `date -u`。
2. 結果放 `results/icpe_h200/<實驗>/`，每列有 `run_id`、`ts`；原始 log 在 `$TIARA_RUNS/<run_id>/`。
3. 量時間的 job 不和我們自己的其他 job 同時跑（`gpu_guard`＋開跑前檢查 `squeue`；memory h200-no-self-cotenancy-for-timing）。
4. 不改 `main.tex`。每個實驗一份 md（登記＋結果，同一個檔，結果寫在登記後面，登記那幾節不改）。
5. 不跑第三階段 3b 的完整地圖 v4（約 5,600 CPU 小時，來不及）。
6. 沒量到就寫 `NOT_MEASURED`；判定照字面，不美化；翻轉的格子逐格列出。
7. push 只用 `code/h200_gitpush.sh`（H200 版的 `gitpush.sh`：fetch → merge → push `origin/h200`，不用 stash／rebase）。

## 用語

- **3a**＝MI300X 的第三階段（`m16_*`：壓縮疊加 X1–X4、C1a／C1b、I1）。**3b**＝預測這條線（`p3x/`：探索 X1–X9、地圖）。兩邊都有 X1、X4，這裡的「X1」「X5」都指 3b（`docs/phase3_20261017/explore/`），「X4b」指 3a（`results/m16_stage3_mi300x/x_stack/x4/`）。
- **harness**＝我們自己的還原程式（`code/m7_model.py`＋`m7_restore_harness.py`）；**f(i)**＝重算第 i 個 512-token block 的時間。

## 實驗清單

| 代號 | 內容 | 登記＋結果 | 狀態 |
|:--|:--|:--|:--|
| P0-1 | 重算速度的效度威脅（harness ÷ vLLM ≈ 1.75）：最快的 attention 重量 f(i)，重跑第一階段延後測試、第二階段還原格子、3b X1／X5 | [P0-1_recompute_speed.md](P0-1_recompute_speed.md) | 登記 |
| P0-2 | 3b「猜誰會回來」的 GPU 驗證（X1 headroom 最大的設定，縮小比例重播） | [P0-2_x1_gpu_validation.md](P0-2_x1_gpu_validation.md) | 登記 |
| P1-1 | 壓縮疊加（低秩＋INT4）在 H200 重現：品質＋慢層端到端還原 | [P1-1_xkv_int4_h200.md](P1-1_xkv_int4_h200.md) | 登記 |
| P1-2 | 便宜的 CPU 對稱重跑（X1 截尾、FreeInference、X5 事後版、X4 f×0.5、X4 排隊占比拆解） | [P1-2_cpu_symmetric_reruns.md](P1-2_cpu_symmetric_reruns.md) | 登記 |
| P2 | CPU 層、GPU 忙時 G5b 選錯 11.5% 那幾格，2 個新 seed | [P2_g5b_cpu_busy_reseed.md](P2_g5b_cpu_busy_reseed.md) | 登記 |
| T1 | ICPE 改成只用 H200（使用者 10/28）：CPU 層換成 H200 自己量的常數，重跑 B、C 節讀 CPU 層常數的模擬＋E2E-C（SSD、NFS 不換） | [T1_h200_cpu_tier.md](T1_h200_cpu_tier.md) | 結果：B、C 節判定字全部不變；E2E-C 的 T1 版在跑（10/28 16:09 起） |
| T1G | T1 §4 的限制：CPU 層在 GPU 上被限速成 35.4 GiB/s 的 GPU 實測 run，用 H200 的常數重量（第一批：A1''、A1'、A2、B6／C2；A3、A4 另登記） | [T1G_gpu_rerun.md](T1G_gpu_rerun.md) | 已登記（10/28） |
| T1G-2 | A3：第 1 輪模擬在 T1 常數下重選存活設定（CPU），存活的上 GPU 確認（同原本的做法） | [T1G2_round1_rerun.md](T1G2_round1_rerun.md) | 已登記（10/28） |
| （最低優先） | 下一篇的「讀取時依問題決定每塊清楚度」前向篩選 | 有空檔的 GPU 才做；不進這篇 | — |

執行紀錄：[`results/RUNLOG_H200_ICPE.md`](../../../results/RUNLOG_H200_ICPE.md)。
