# 第三階段實驗紀錄（逐實驗）

> 規則：每個數字都附 run_id 與輸出檔；〔模擬〕＝用實測常數算的，不是量測；NOT_MEASURED＝還沒量。原始 log：`/mlsteam/data/tiara/runs/<run_id>/`（cmd.sh、context.txt、stdout、stderr、exit_code）。
> 時間一律 UTC（台灣＝＋8）。

---

## X0　環境快照（10/19 13:01–13:34Z）

- 產出：`environment_snapshot.md`。只用唯讀指令，沒開 GPU 行程。
- 失敗與異常：無。`unshare -m`（想用 mount namespace 隔離 /dev/kfd）→ `Operation not permitted`；改成「計時期間不 import torch」的規則。

## X1　第二階段十條觀察的核實＋工具箱（10/19 13:0x–13:2xZ，agent，self-review）

- 產出：`stage2_factcheck_and_toolbox.md`。
- 結論：十條裡第 2–10 條出自**第一階段**（M7）；202／2720 ms 是含新一輪的 TTFT、「NFS」是限速模擬；0.33／6.98 GiB/s 是限速參數。詳見 `experiment_matrix.md` §1。
- 順帶發現第二階段的分析 bug（`m10_longctx_analyze.py` 沒篩長度）→ 已轉告第二階段 session，修好並重跑（commit 79cbdc7，RUNLOG_MI300X 15.11）。

## X2　文獻地圖（10/19 13:0x–13:5xZ，4 個 agent，self-review）

- 產出：`lit/L0`–`L3`、彙整 `research_map.md`。L4（全文查新：Cascade、CacheSmith、CacheFlow）跑中。

---

## S1　無損壓縮（程式就緒，資料等 GPU）

- `code/m16_s1_lossless.py --self-test`：ALL PASS（25 項；5 個變體 × zstd／zlib 的速度測試都逐位元相同）。過程抓到一個自己的 bug：zstd context 被兩個 thread 共用 → "Data corruption detected"，改成每個 thread 一份。
- `code/m16_s1_kvdump.py`：還沒跑（要 GPU；第三階段窗口的第一個 job）。
- 關鍵數字：NOT_MEASURED。

## I1　多 session 同時還原：事先預測〔模擬〕（10/19 13:40Z）

- **run_id**：`20261019-134046-m16-i1pred`（之前兩次 `-133930-`、`-134035-` 是同一支程式的草稿版與一次 bug（變數遮蔽，exit 1），以這次為準）。
- **指令**：`m7run m16-i1pred nice -n 19 taskset -c 0-3 python -I code/m16_i1_predict.py --model llama --out results/m16_stage3_mi300x/i1_concurrent_restore/prediction`
- **產出**：`results/m16_stage3_mi300x/i1_concurrent_restore/prediction/prediction_llama.{csv,json}`。
- **模型**：`code/m16_mcake.py`（多 session 的 Cake 排程；一條計算線、每層一條載入線；自測 ALL PASS）。常數全部讀檔：f(i)、chunk、pinned H2D 49.858 GiB/s、b16／b32／b48（`speed_llama/bw.json`，run `20261015-154558-m10-calibf-llama`）；新一輪 0.0755 s（閒）／0.235 s（忙）、單 session 實測還原（`speed_llama/summary_fp8.csv`，run `20261015-161226-m10-grid-fp8-llama`）；GPU 忙時重算 ×2.07（`busy_levels.json`）。「載入在計算忙時 +50%」只是敏感度參數（08 §2.1 的算術，不是本實驗的量測）。
- **尺準**（N＝1，模擬 vs 第二階段實測，GPU 閒）：cpu 0.0889 vs 0.0954 s（×0.932）、b16 0.5688 vs 0.5782（×0.984）、b32 1.3635 vs 1.3802（×0.988）、b48 2.3832 vs 2.4215（×0.984）。
- **關鍵數字〔模擬〕**（N＝4、8；四種工作負載 × 閒／忙 × 有無干擾）：
  - **A：從「各自 Cake＋輪流」到「各請求各自最佳」（第二階段的 G5b＋先到先服務／最短優先）**：P95 少 4–42%、平均少 10–58%。來源是兩個已知的東西：G5b（CPU 層全部載入比 Cake 快，第二階段 11 問題三）與排隊順序（處理器共享 vs 先到先服務的教科書結果）。
  - **B：從「各請求各自最佳」到「同時考慮所有 session」（joint，每個 session 指定全部重算或全部載入、讓兩條線總量平衡）**：**每一格都是負的**（−14% 到 −60%）。在這個模型下，聯合決定沒有好處。
- **判讀**：I1 原本的新意假說（「聯合決定比各自 Cake 好」）在「互不干擾」的模型下**不成立**；能拿到的好處都來自已知的規則與排隊順序。剩下唯一的問題是「真實的干擾會不會讓模型錯」→ GPU 實測降級成 1 小時的驗證（N＝4、8，cpu／b32，閒；看實測和預測差多少）。如果實測和預測差 <10%，I1 這個方向收掉（負面結果：第二階段的單請求規則＋排隊順序就夠）。
- **失敗與異常**：無（第二次執行的 exit 1 是自己的變數遮蔽 bug，已修，見上）。
- **限制**：joint 只試了「每個 session 全算或全載」這一種；更細的聯合（例如每個 session 的會合點一起調）沒試。
