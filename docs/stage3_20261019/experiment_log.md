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

## I3s　agent 工具暫停時 KV 去留的模擬〔模擬〕（10/19 14:1xZ，agent 寫、我核對）

- **run_id**：`20261019-141254-m16-pausesim`（exit 0，11,232 次模擬）。第一次 `20261019-141044-m16-pausesim` exit 1：帳目斷言抓到「下一輪截斷歷史時，舊副本記成整段長度」的 bug，已修、加回歸測試（`FAILED_NOTE.txt`）。
- **程式**：`code/m16_pause_sim.py`（自測 74 項 ALL PASS；只用 CPU，結尾斷言沒開 /dev/kfd）。報告：`i3_pause_sim.md`。結果：`results/m16_stage3_mi300x/i3_pause_sim/`。
- **常數**：MI300X（Llama、Qwen）全部讀檔、附 run_id（`costs_sources.csv`）；延伸處已標（Qwen 新一輪 <32K 用 32K、GPU 忙的倍數 <32K 用 32K、Qwen 的 D2H 沿用 Llama）。3090 缺 D2H、忙的倍數、原始 H2D，而且舊量測是 HEAVY 污染 → 只進靜態表（NOT_MEASURED）。
- **判定（照 L3 缺口 1 事先寫的證偽條件）**：事前凍結的 κ 感知四選一，主掃描 **0／96 格比最好的現成規則好 ≥5%**（中位 −2.8%，最好 +1.3%）→ **缺口不成立，照 CLAUDE.md 規則 4 停**。
- **探索性（不進判定）**：
  - 最好的現成規則多半是 vLLM 0.28 原生的 CPU＋SSD 串寫；勝負取決於「有沒有快的下層接住 CPU 放不下的 KV」。
  - 下層從本地 SSD 換成 NFS：31／32 組最好的規則換人；把本地 SSD 上的贏家搬到 NFS，P95 中位 +60%（最多 +482%）——「哪個規則最好」隨硬體翻轉，但 κ 感知本身打不贏該硬體上最好的規則。
  - 事後加的 `kappa_tier` 只在 CPU 層縮成 0.25×HBM 時明顯贏（中位 +49%），一部分來自 vLLM 沒有的寫入路徑；事後候選，不能下判定。
  - 真實 HBM（192 GB）下沒有容量壓力：C＝32 時容量是參考工作集的 4.2 倍（Llama）／9.5 倍（Qwen）。
  - 最大的改善空間是「知道 session 已經結束」（NFS 設定下中位 +26%），比 κ 大。
- **限制**：這是模擬（常數是實測的，排程與佇列是模型）；GPU 驗證點沒有跑。
- **對 agent 方向的結論**：使用者的「場景改 agent 感覺太急」在這個模擬裡得到支持——κ 感知的去留決策沒有空間。剩下兩個可能的改題（「依實測 κ 自動選規則」「應用層告知 session 結束」）要使用者決定要不要考慮。

## I3r　I3 模擬換基準重讀〔模擬〕（10/20 04:02Z）

- **為什麼**：使用者質疑探索「太多否定」；我（10/20）說過「I3 只跟神諭比，換成固定作法結論會翻」。這次照實重算檢查那句話。
- **run_id**：`20261020-040244-m16-i3reread`（同一支程式的前一次 `-040223-` 少了 vllm_tier 基準，以這次為準）。只讀 `20261019-141254-m16-pausesim` 的 `cells.csv`／`regret.csv`，不重跑模擬。
- **指令**：`m7run m16-i3reread nice -n 19 taskset -c 0-3 python3 -I code/m16_i3_reread.py`
- **產出**：`results/m16_stage3_mi300x/i3_pause_sim/reread_baselines.{csv,json}`。
- **關鍵數字〔模擬〕**（回來那一輪 TTFT P95；gain＝1 − 事前凍結的 κ ÷ 基準，>0＝κ 較好；中位數）：

| 設定組 | 對神諭（每格最好） | 對 vLLM 原生卸載（vllm_tier，固定） | 對不卸載的 vLLM 預設 |
|:--|--:|--:|--:|
| main（本地 SSD、CPU 大，96 格） | −2.8% | −1.7% | +9.8% |
| GPU 外加忙（busy_exo） | −48.9% | −48.9% | +67.7% |
| Cake 還原 | −65.5% | −65.5% | +68.8% |
| 下層換 NFS（ssd_nfs） | −5.3% | **+27.3%** | +69.7% |
| CPU 層小（cpu_small） | +35.6% | **+35.6%** | +50.9% |
| CPU 層小＋NFS | +7.1% | **+38.2%** | +14.1% |

- **搬錯硬體**（regret.csv）：本地 SSD→NFS 方向，κ 比「原封搬過去」好 31／32 組（regret 中位 5.3% vs 60.3%）；**反方向 NFS→本地 SSD 只好 3／32**（κ 57.3% vs 25.7%）；GPU 閒↔忙兩個方向都是 0／32（原封搬過去幾乎沒損失）。
- **判讀**：
  - 我 10/20 說的「換基準結論就翻」**大部分不成立**：在本地 SSD、CPU 層夠大的設定（main、忙、Cake、sigma2），連可部署的固定規則（vLLM 原生卸載）κ 都打不贏，原判定（停）在這些設定成立。
  - **只有在不尋常的硬體**（下層是 NFS、或 CPU 層小）κ 才明顯贏 vLLM 原生卸載（中位 +27～38%）；但這些設定裡現成的 Tokencake 也贏 vLLM 原生卸載，κ 對神諭仍差 5%（ssd_nfs）。也就是：價值在「硬體不尋常時別用 vLLM 預設」，不在 κ 本身。
  - 「κ 能避免選錯」只在 SSD→NFS 一個方向成立，不能一般化。
  - 對「不卸載」的預設，任何卸載規則都大贏，這個基準太弱，不能拿來支持 κ。
- **限制**：同 I3s（模擬；成本常數是第二階段 harness 路徑，vLLM 實際 CPU 層約 4.1 GiB/s 沒代入）。
