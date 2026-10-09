# RUNLOG — 平台 B（AMD MI300X @ MLSteam）

> 格式照 CLAUDE.md §4.2。原始 log 在 `/mlsteam/data/tiara/runs/<run_id>/`。沒量到就寫 `NOT_MEASURED`。
> **舊紀錄（M1–M6，2026-09-15 → 10-01）見 commit `9deda4f` 的 `results/RUNLOG_MI300X.md`**（main 在 `52803bc` 推倒重來時刪掉了）。

---

## Milestone 7 — 第一階段：寫入時依位置放置 vs 簡單策略（BF16，Cake 還原）

**設計**：`docs/phase1_20261008/`（03 策略、05 實驗、05 §6 判準開跑前寫死）
**報告**：`docs/phase1_20261008/07_report.md`
**狀態**：DONE（判定：Q4 < 5%，簡單策略追平；A0 有 1／28 格例外，見報告 §0 總表）
**執行時間**：2026-10-08 12:59 → 見報告附錄 A 的最後一個 run_id
**程式**：`code/m7_model.py`、`code/m7_restore_harness.py`、`code/m7_write_policy.py`、`code/m7_calib.py`、
`code/m7_params.py`、`code/m7_busy.py`、`code/m7_duplex.py`、`code/m7_c6_vllm.py`、`code/m7_analyze.py`、
`code/test_m7_correctness.py`；執行鏈 `code/m7_run_phase1*.sh`
**產出檔**：`results/m7_write_policy_mi300x/`（每個 CSV 每列都有 run_id、ts）；圖在 `docs/phase1_20261008/report/figs/`
**關鍵數字**：全部在報告裡，並由 `code/m7_analyze.py` 從 CSV 自動產生（`verdict.json`、`summary_*.csv`）

### 失敗與異常（完整）

1. **B 主設定第一次執行崩潰**（run `20261008-141014-m7-b-nfs-gap0`）：
   `ValueError: S2a`（`m7_write_policy.py` line 663, in after_round）。
   原因：執行鏈沒有給 `--strategies`，B 用到 A1 的預設清單（S0/S2a/S3/S5/CPUall），而 B 不支援 S2a、CPUall。
   這是**我的設定錯誤，不是實驗結果**。處置：改成依子命令給預設（B＝R0…S5s），部分輸出移到
   `<run_dir>/b_share_ABORTED_partial.csv`，不進結果；用同樣的參數重跑（run `20261008-142647-m7-b-nfs-gap0`）。
2. **C7 自檢第一次未過**（run `20261008-132940-m7-c7self`）：設 40 GiB/s 時實得慢 6.5%（判準 5%）。
   原因：連續讀取之間的軟體開銷（sleep 喚醒、event 同步）被當成裝置閒置。修正：距上一筆完成 <0.5 ms 視為背靠背。
   A0 用修正前的版本（只用到 ≤11.6 GiB/s，誤差 ≤1.5%）；A1、B 用修正後的版本；修正後重做自檢（run `20261008-183255-m7-c7self-fixed`）：最大誤差 −0.3%，通過。
3. **FIFO 限速器與真實裝置不符**（A2，run `20261008-131619-m7-a2`）：真實本地 SSD 在背景寫入時，還原者 TTFT 0.24→0.52 s；
   FIFO 限速器 0.28→2.15 s。改用 share 模型（讀寫各自排隊、寫入進行中讀取變慢 k 倍，k 由同一份 A2 校準），
   驗證 run `20261008-132038-m7-a2share`。FIFO 保留為敏感度（B-fifo）。
4. **C5 的 GpuWatcher 標記 CONTAMINATED**（run `20261008-130834-m7-c5`）：單一取樣 pid −1、20 GB，
   出現在自己的背景負載 process 結束的瞬間，和舊紀錄發現 6（amd-smi 在 process 結束後仍回報數秒）同型。
   C5 量的是自己背景負載造成的變慢倍數，判斷為假污染，數字保留，在此註明。
5. **C6 結束後 /dev/shm 殘留 48 GiB 的 `vllm_offload_*.mmap`**（同舊紀錄發現 7、16）：確認 vLLM 已結束後手動刪除。
   `m7_c6_vllm.py` 沒有自動清理，下次要補。
6. 我自己的兩次 `pkill -f` 誤殺了自己的 shell（指令列含同一字串），只影響 smoke test 與等待中的排程腳本，不影響任何結果檔。

7. **GPU 短暫變慢**（A1，run `20261008-134526-m7-a1`）：GPU 閒的 480 列中 68 列的 t_new 比最小值高 >15%（最多 2.5 倍、持續數秒），GpuWatcher 沒有看到外來 pid。原因未明。B 的分析另外報排除這類事件的版本（B 主設定裡 0 個事件被標記）。
8. **執行鏈重排**：B 的敏感度原本要跑 9 個策略 × 3 容量，估計會跑到隔天，改成 25%／50% × 6 個策略（S1、S4、S4+、S4+P、S5、S5s）（`code/m7_run_phase1f.sh`）。判定用的主設定沒有縮減。

### 與設計（05）的差異
見報告 §2.3（D1–D10）。

### 此次 audit
報告裡的判讀是 **self-review，非 cross-model**（CLAUDE.md §5：沒有 Codex MCP；開 zero-context reviewer 需使用者要求）。

---

## Milestone 7b — 08 消融：「寫入時決定」何時不會被 Cake 抵消

**設計**：`docs/phase1_20261008/08_ablation_plan.md`（判準開跑前寫死；§8 是開跑後的修正，加了對照組 S4B）
**報告**：`docs/phase1_20261008/09_ablation_report.md`
**狀態**：DONE（判定：GPU 16 組比較 0 組通過 08 §6＋§8 → 寫入時的位置決定被 Cake 抵消；唯一沒排除的例外是重算便宜（MHA），NOT_MEASURED）
**執行時間**：2026-10-09 06:49 → 11:14 UTC
**目前問題總表**：`docs/phase1_20261008/QUESTIONS.md`
**程式**：`code/m7_sim.py`（虛擬時鐘，和 GPU harness 共用 BState 與限速器）、`code/m7_ablate_analyze.py`、`code/m7_run_ablate.sh`；
`code/m7_write_policy.py` 新增 `--b2 --release free|hold --workload chat|doc` 與策略 S4L、S5L、S5P、S5c、S4B（舊行為不變：預設 free＋chat，輸出仍寫原檔）
**run_id**：`20261009-064908-m7-sim-validate`、`20261009-064926-m7-sim-sweep`（第一次，沒有 S4B，已被 sweep2 覆蓋）、`20261009-065112-m7-sim-sweep2`、
`20261009-*-m7-sim-probe-*`、`20261009-065431-m7-b2-smoke`（輸出另寫 runs 目錄，不進 results）、
`20261009-070326-m7-b2-g1-doc-hold-cpu3.69-s0`、`20261009-090235-m7-b2-g1-doc-hold-cpu3.69-s1`、`20261009-102010-m7-b2-g2-chat-hold-s0`（三個都 exit 0、GpuWatcher CLEAN、kv_bad 0 列）
**產出檔**：`sim_validate.*`、`sim_sweep*.csv`、`sim_probe.csv`、`b2_share.csv`（GPU 原始列）、`b2_summary.csv`、`b2_verdict.csv`（`code/m7_b2_analyze.py`）
**關鍵數字**：見報告 §0、§4；模擬器在新設定 43／44 格誤差 ≤10%

### 失敗與異常
1. **下載 LongAlpaca-7B 第一次失敗**（run `20261009-070357-m7-dl-longalpaca7b`，exit 2）：`Argument expected for the -c option`。
   原因：我把多行的 `python -c` 塞進 `bash -c` 字串，引號被拆開。**是我的指令錯誤，不是下載問題**。改成腳本 `code/m7_dl_model.py` 重跑（run `20261009-070408-m7-dl-longalpaca7b`）。
2. 第一次模擬掃描（`m7-sim-sweep`）的 `sim_sweep.csv` 被第二次（加了 S4B）覆蓋；第一次的輸出只留在 run 目錄的 stdout。
3. `b_share.csv`（2.1 MB）超過 git 的 1 MB 原則：進 git 的是 `b_share.csv.gz`，原檔留在本機（`.gitignore`）。
4. **我中途改估的完成時間錯了**：原本估 3.5–4 小時，實際 4 小時 11 分。中途我看 seed 0 跑了 2 小時，就改估到 12:30 UTC，但 seed 1 只有 20 個事件（seed 0 有 32 個），1 小時 17 分就跑完，實際 11:14 結束。不影響結果。
5. **分析腳本第一版漏掉 seed 1**：`m7_b2_analyze.py` 用全部資料的最大事件數判斷「rep 是否完整」，但每個 seed 的事件數不同（make_workload 每輪後 25% 機率不再回來：seed 0 有 32 個、seed 1 有 20 個），seed 1 全被當成不完整而略過。改成逐設定算後重跑；報告用的是修正後的輸出。

---

## Milestone 7c — 破解計劃第 1 輪（10、11、12）

**設計**：`docs/phase1_20261008/10_breakthrough_plan.md`、`11_round1_plan.md`（判準開跑前寫死；三段追加都寫在對應的 run 之前）
**報告**：`docs/phase1_20261008/12_round1_report.md`；總報告 `FULL_REPORT.md`
**狀態**：DONE（判定：沒有任何假設證明寫入時決定有效；H1 殺掉；H0 顯示真實系統預設「放不下就丟」；模擬存活者 Llama 上 GPU 0／5 通過；反方審查後，第 2 輪建議改成先建「併發＋寫入積壓＋放不下就丟」的模型）
**執行時間**：2026-10-09 16:45 → 21:29 UTC（GPU 確認 18:32 → 21:28）
**程式**：`code/m7_round1_analyze.py`（h1、kappa、sim）、`code/m7_round1_sim.py`、`code/m7_guard_run.py`、`code/m7_run_round1.sh`；
`m7_model.py`／`m7_write_policy.py`／`m7_sim.py` 加環境變數 `M7_MODEL_GLOB`、`M7_EXPERTS_IMPL`、`M7_CHUNK_BYTES`、`M7_F_CSV`（不設時行為不變）；
`m7_model.py` 支援 Qwen3 的 q_norm／k_norm；新策略 S4W（背景版）；`test_m7_correctness.py` 加 `M7_TEST_L`
**subagent**：Lit-A、Lit-B、Lit-C（文獻）、H0（讀原始碼）；反方審查（zero-context，使用者 10/9 同意）

**run_id**：
- 正確性：`20261009-165051-r1-ok-longalpaca7b`（崩潰）、`20261009-175543-r1-ok-longalpaca7b-fix`、`20261009-165546-r1-ok-qwen3-30b-a3b`（假污染）
- f(i)：`20261009-171842-r1-c1-llama31-8b`、`20261009-171938-r1-c1-longalpaca7b`（卡住，手動 kill）、`20261009-175625-r1-c1-longalpaca7b-fix`、
  `20261009-172430-r1-c1-qwen3-30b-a3b-grouped`、`20261009-173811-r1-c1-qwen3-30b-a3b-eager`
- vLLM：`20261009-175737-r1-vllmf-llama31-8b`、`20261009-180018-r1-vllmf-qwen3-30b-a3b`（相減法，沒過驗證）；
  `20261009-180348-r1-vllmf2-llama31-8b`、`20261009-180549-r1-vllmf2-longalpaca7b`、`20261009-180757-r1-vllmf2-qwen3-30b-a3b`（引擎時間戳，探索性）
- 模擬：`20261009-165340-r1-sim-llama31-8b`（沒有 S4C，已被覆蓋）、`20261009-180002-r1-sim2-llama31-8b`、`20261009-180002-r1-sim2-longalpaca7b`
  （另有 `20261009-175750-r1-sim-longalpaca7b`：沒有 S4C 的第一版，已被 sim2 覆蓋）
- 文獻與原始碼：`20261009-164921-h0-src`（LMCache、SGLang、Dynamo 的 clone，只讀）
- GPU 確認（11 追加 4；`code/m7_run_round1e.sh`，log `/mlsteam/data/tiara/runs/m7_round1e_chain_1832.log`）：`20261009-183252-r1-gpu-llama-doc-free-local-s0`、`20261009-192035-…-s1`、`20261009-195129-…-s2`、`20261009-202956-…-s3`、`20261009-210351-…-s4`。
  5 個都是 exit 0、gpu_guard CLEAN、kv_bad 0。指令：`python m7_write_policy.py b --b2 --workload doc --release free --ssd-dev local --wl-seed <s> --strategies S1 S2b S4 S4+ S4+P S4L S4B S4W S4C S5L --cpu-fracs 0.5 --reps 3 --verify`（`M7_IO_MODEL=share M7_CPU_GIBPS=11.6`）
**產出檔**：`results/m7_explore_mi300x/`（h1_*、kappa_screen、r1_sim_*、calib_c1_*、correct_*、vllmf*、r1_gpu_*）；`docs/research_20261009_explore/`

**關鍵數字（GPU 確認）**：Llama、CPU 11.6、free、doc、本地 SSD、50%。S5L 回來請求 TTFT 中位數 vs 最好的不看位置／S4B／S4W／S4C：
seed 0 −3.1%／+1.2%／+11.8%／+4.5%；seed 1 +0.5%／+0.5%／+9.0%／+9.5%；seed 2 −0.6%／−0.7%／+10.5%／+3.3%；seed 3 −3.4%／−2.4%／+16.8%／+8.4%；seed 4 +0.1%／+0.2%／+8.0%／+30.9%。
**0／5 個 seed 通過**（判準 ≥3／5，四個對手各 ≥5%）。模擬誤差 50 格裡 47 格 ≤10%（中位 3.1%）；S5L 5／5 個 seed 被模擬低估（−1.8% 到 −8.8%），S4B 平均 +1.1%。來源：`r1_gpu_summary.csv`、`r1_gpu_verdict.csv`。

### 失敗與異常
1. **LongAlpaca-7B 正確性測試崩潰、C1 卡住**：
   - 正確性測試（run `20261009-165051-r1-ok-longalpaca7b`，exit 250）：`HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware exception. code: 0x1016`，kernel grid=[2097152, 2, 1]。
   - C1（run `20261009-171938-r1-c1-longalpaca7b`）：GPU 100%，4 分半沒有寫出任何一列；我手動 kill（exit 241）。
   - **原因**：測試與校準用 `torch.randint(1000, 120000)` 產生隨機 token，但 LongAlpaca-7B（Llama-2）的詞表只有 32001，embedding 查表越界。Llama-3.1-8B（128256）和 Qwen3（151936）的詞表都大於 120000，所以第一階段沒遇到。
     我一開始猜是注意力權重 2^31 個造成 int32 溢位，**猜錯了**，已更正。
   - **處置**：`m7_calib.py`、`test_m7_correctness.py`、`m7_write_policy.py` 的上限改成 `min(120000, vocab_size)`。對 Llama-3.1-8B 和 Qwen3 產生的 token 完全相同（上限沒變），第一階段的結果不受影響。LongAlpaca 重跑。
2. **Qwen3-30B-A3B 正確性測試被標 CONTAMINATED**（run `20261009-165546-r1-ok-qwen3-30b-a3b`，exit 3）：GpuWatcher 只有一筆外來樣本，pid −1、70 GB，時間 17:18:40，正好是自己的 process 結束的那一刻。
   和第一階段 C5 的假污染同型（amd-smi 在 process 結束後仍回報幾秒）。70 GB 也和 Qwen3 自己的權重加 KV 相符。這是正確性測試，不是計時，數字保留。
3. **vLLM 相減法沒過事先寫的驗證**（Llama：vLLM／harness 比值中位 0.275，判準 0.85–1.15）。照判準，Qwen3 不採用；改用引擎時間戳，只當觀察（11 追加 3）。
4. **分析腳本把自己的輸出當輸入**：`m7_round1_analyze.py sim` 用 glob `r1_sim_*.csv`，抓到自己產生的 `r1_sim_verdict.csv`，出現 `KeyError: 'wl_seed'`。已排除這兩個輸出檔後重跑。
5. **11 的追加段落時間寫錯**：一開始寫的時間比實際晚（例如寫 18:05，但對應的 run 是 18:00:02 開始的）。已改成「寫在 run XXX 之前」。內容沒有變。
6. **GPU core dump 寫進 repo**：崩潰和被 kill 的兩個 process 在 `code/` 留下 `gpucore.3082060`（18 GB）和 `gpucore.3094121`（22 GB）。已移到對應的 run 目錄，`.gitignore` 加了 `gpucore.*`。可以刪除。
7. **subagent 在 repo 根目錄留下 `nfs5.html`**（nfs(5) man page，17:06 下載）。不在它被允許寫入的範圍；已移到 scratchpad，沒有進 git。
8. `r1_sim_longalpaca7b.csv` 1.0 MB，超過 1 MB 原則：進 git 的是 `.gz`。
9. LongAlpaca 的 vLLM 量測只量到 62 個 chunk（31,744 token），因為它的 rope 線性 ×8 上限是 32,768。我一開始設 64 個，加上 1,024 會超過上限，在啟動前改掉了（重啟等待中的執行鏈；舊的那個在 sleep 中被 kill，變成 zombie，沒有跑任何東西）。
10. **模擬存活者在 GPU 上消失**（不是程式錯誤，是方法上的發現）：模擬器對 S5L 的相對偏差（約 5–6 個百分點）和模擬裡的優勢（7.6%）差不多大。模擬用 5% 門檻挑存活者不可靠；之後模擬門檻要 ≥10%，或每個存活者都上 GPU。
