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

## Milestone 7d — 8 個方向同時探索（D1–D8）

**設計**：`docs/research_20261010_directions/README.md`（2026-10-10T05:29:54Z，開 agent 之前）；各方向的判準寫在各自文件的 §2
**報告**：`docs/research_20261010_directions/SUMMARY.md`（主 session 的 self-review，不是 zero-context 審查）
**狀態**：DONE（判定：寫入時決定從 8 個角度都沒有不可取代的地方；還活著的線索見 SUMMARY §3）
**執行時間**：2026-10-10 05:30 → 07:20 UTC
**subagent**：8 個 general-purpose agent，每個方向一個。只有 D2 用 GPU（約 80 分鐘）
**程式**：`code/m8_conc_sim.py`（D1，含修好 keep-head bug 的 `restore_v2`）、`m8_vllm_drop.py`（D2）、`m8_kappa_map.py`（D3）、`m8_trace_oracle.py`（D4）、`m8_arch_arith.py`（D5）、`m8_precision_arith.py`、`m8_d6_quality_inventory.py`（D6）、`m8_os_bench.py`（D7）。沒有改任何既有檔案
**產出檔**：`results/m8_directions/d{1..7}_*.csv`，每列都有 run_id、ts；文獻卡在 `docs/research_20261010_directions/cards/`（86 張）
**run_id**：完整清單在各方向文件的 §3。主要的有：
- D1：`20261010-054631-d1-sweep`、`-054724-d1-analyze`、`-055311-d1-analyze-admitfirst`
- D2：`20261010-054520-m8-d2-off-doc` 到 `-065733-m8-d2-cpu50bs512-doc`（9 個），彙整 `-070054-m8-d2-collect`
- D3：`20261010-054706-d3-kappa-map`
- D4：`20261010-054225-d4-char`、`-054330-d4-sim`、`-055356-d4-gap`、`-060621-d4-gap-prereg`
- D5：`20261010-055012-d5-arith`
- D6：`20261010-055349-d6-arith`
- D7：`20261010-054238-d7-env` 到 `-060721-d7-analyze`（11 個）

**關鍵數字**：
- D2：0／133,504 個 block 被跳過〔實測〕。開 CPU 卸載的 TTFT：chat C＝1 回來請求 0.88→1.63 s；doc C＝16 43.7→67.0 s〔實測，單一 seed〕。
- D1：模擬持續排隊 ρ 最大 0.99〔模擬〕。
- D4：75–78% 的 block 寫了就沒人再讀〔trace〕。

### 失敗與異常
1. **D1 的判定在事後改了**：事先判準的結果是「有看頭」，5／192 格通過。agent 修了對手的兩個程式錯誤後是 0／192，之後又加了兩個新對手。兩個結果都列在 D1 §0。
2. **D4 的判定取決於對手集合**：文字列 7 個，程式放 25 個。「7 個」那次是看過「25 個」的結果之後才算的。
3. **D2 的 gpu_guard**：9 個 run 裡 7 個被標 CONTAMINATED，都是已知的假警報型態（自己的 process 結束後出現 pid −1）。接受規則是看到標記之後才寫的，見 `d2_guard.csv`。時間數字只有一個 seed，要重跑才能當定論；被跳過 block 數這類計數不受影響。
4. **D7 的寫入量超過預算**：約 161 GiB，上限 150。資料檔都已刪除。
5. **D5 的子 agent 把原始碼 clone 到 repo 根目錄**，約 15 秒後移走，根目錄已確認乾淨。
6. **`runsh` 的參數不能含空白、括號或引號**：它用 `"$*"` 寫 cmd.sh，D4（gap2／gap3）、D6（054930）、D8 都踩到。完整錯誤記在各文件 §7。要修 `/mlsteam/workspace/bin/runsh`。
7. **資料完整性問題**（SUMMARY §4）：
   - `main.tex` 的 3090 κ＝8.9，來源資料標 `host_contention=HEAVY`。
   - MI300X 的品質 CSV 在 git 和本機都不見。
   - Llama 129K 時 FP8 和 BF16 的輸出逐字相同，FP8 疑似沒生效。
   - 都沒改 main.tex。

## Milestone 7e — 後續 F1–F6

**設計**：`docs/research_20261010_followup/README.md`（07:38:27Z；F6 追加 07:57:08Z，都在開 agent 之前）
**報告**：`docs/research_20261010_followup/SUMMARY.md`（self-review）
**狀態**：DONE
**執行時間**：2026-10-10 07:38 → 11:30 UTC
**GPU**：F1 約 168 分、F2 約 40 分、F3 約 11 分。三個 agent 用 `flock /mlsteam/data/tiara/gpu.lock` 輪流
**基礎設施**：`/mlsteam/workspace/bin/runsh` 修好引號（`%q`）和 RUN_ID 不一致的問題；測試 run `20261010-073741-infra-runsh-test`；備份 `runsh.bak-20261010`
**程式**：`code/m9_f1_*`、`m9_f2_*`、`m9_f3_*`、`m9_f4_*`、`m9_f6_analyze.py`（都是新檔）
**產出檔**：`results/m9_followup/f{1,2,3,6}_*.csv`、`results/audit_20261010/`、`docs/audit_20261010/`、`docs/paper_negative_20261010/`
**run_id**：完整清單在各文件。主要的有：
- F1：`20261010-075828-f1-s1-off` 到 `20261010-112119-f1-s1-rocmfixcoal`（25 個），分析 `20261010-112856-f1-analyze-final`
- F2：`20261010-074627-f2-probe`、`-075158-f2-sweep`、`-082923-f2-sweep-la7b`，計算 `-083136-f2-final`
- F3：`20261010-074630-f3-dequant`、`-074830-f3-fp8`
- F4：`20261010-075351-f4-rebuild`、`-075352-f4-kappa`、`-080228-f4-platA`、`-074654-f4-platB`、`-075307-f4-platBc`、`-080301-f4-claims`
- F6：`20261010-080457-f6-sweep-main`、`-080603-f6-sweep-l8`、`-080637-f6-sweep-l8q16`、`-081031-f6-summary`

**關鍵數字**：
- F1：attention 1.367→3.277 s（ROCM_ATTN→TRITON_ATTN，33K），doc C＝1 第一輪 2.24→4.40 s（3／3 seed），修法消掉 45–74%〔實測〕。
- F2：GPU 726.4 W（重算）vs 168.9 W（等待）〔實測〕。
- F3：INT8 反量化 0.031 ms／chunk；FP8 容量 2.000×〔實測〕。
- F6：ρ<1 時 0／216〔模擬〕。

### 失敗與異常
1. **C6 的對照用的是 vLLM 的慢路徑**（F1 發現）：harness 的重算比預設 vLLM 慢約 1.7–2 倍。07 §7 的「計算端相當」不成立，已在 07 和 FULL_REPORT 加上更正。Cake 頻寬帶的絕對數字要重量。真實速度（約 f×0.5–0.6）正好落在 09 唯一還沒排除的例外，要用快的 attention 重量 f(i) 後，用完整的對手集合重跑。我一開始在 07／FULL_REPORT 寫「結構性結論仍成立」，是錯的，F5 指出後改正。
2. **D2 的 TTFT 比較混了 attention kernel 的差**，doc C＝1「快 2 倍」是兩群中位數在跳的假象（F1）。
3. **gpu_guard**：F1 有 19 個 run 標成已知假警報；「最後一格」的定義是事後擴充的。
4. **F2 只有一個 seed**；主機耗電量不到。
5. **main.tex 稽核**（F4）：241 個數字中 59 個不一致、17 個找不到來源、70 個有疑慮。沒改 main.tex。
6. **我寫的 SUMMARY 第一版漏了 chat C＝4**（卸載有幫助的格）。F5 指出後改正。

## Milestone 7f — F7：重算快 2 倍時結論還成立嗎〔模擬〕

**判準**：`docs/research_20261010_followup/README.md`「追加 F7」（2026-10-12T08:48:50Z，跑之前寫的）
**狀態**：DONE（判定：free 下 f×0.5、×0.6 都是 0 個存活 → 結論在模擬層級成立）
**run_id**：`20261012-084917-f7-fscale-sim`、`20261012-084936-f7-judge`
**指令**：`m7run f7-fscale-sim python code/m9_f7_fscale_sim.py run --scales 1.0 0.5 0.6`；`m7run f7-judge python code/m9_f7_fscale_sim.py judge`（`HIP_VISIBLE_DEVICES=""`，沒用 GPU）
**產出檔**：`results/m9_followup/f7_sim_f{1.0,0.5,0.6}.csv`、`f7_verdict.csv`、`f7_configs.csv`；`docs/research_20261010_followup/F7_fscale.md`
**驗證**：f×1.0 的 4,800 列和第 1 輪的 `r1_sim_llama31_8b.csv` 逐列相同（最大差 0.0）
**關鍵數字**：
- free：f×0.5、×0.6 都是 0／32 個設定存活（≥5% 和 ≥10% 都是），最大領先 4.9%／3.2%。
- hold：≥10% 存活 1 個（f×0.6，CPU 11.6、doc、NFS、50%），3 個 seed 贏 22–30%，2 個 seed 輸寫穿 20–27%。

**失敗與異常**：無
**限制**：只有模擬；f 是整體縮放（真實的比值 8K 時 2.0 倍、32K 時 1.7 倍）

**更正（2026-10-12）**：07 §5 寫「寫入一提交就回來：48 格裡 6 格 S3 顯著較快（CI 不重疊且 >δ）」有兩處錯。分母應是 24；6 格只滿足「CI 不重疊」，其中只有 3 格 >δ（6.2%）。重算自 `results/m7_write_policy_mi300x/summary_a1_pairs.csv`，已改 07 與給老師的報告。結論（「大致沒有」）不變。

## Milestone 10-pilot — 精度這條線的彩排（MI300X；不進任何圖或判定）

**狀態**：PASS（03_plan_h200 §7「等帳號」那一列的程式＋在真 GPU 上抓 bug）
**執行時間**：2026-10-15 02:19 → 02:50 UTC（GPU run）；程式 02:05 起
**為什麼在 MI300X**：H200 約一週後才有。這裡的數字只用來確認程式對不對，**不和 H200 的數字放在同一張圖或判定**（03 §1）。**沒有跑任何題目、沒看任何低精度的答對率**（FP8 只測「有沒有壞」：位元比對、NaN），事先登記還沒凍結，沒有被看過的結果影響。
**產出檔**：`results/m10_pilot_mi300x/`（JSON；`regress_xproc/` 另有判準檔 `CRITERION.md` 與 `summary.txt`）

**程式**
- 新：`code/m10_codec.py`（FP8 三種 scale 粒度、K 去偏移 FP8、INT8、INT4 兩種分組）、`code/test_m10_restore.py`、`code/m10_tasks.py`（T1–T5、三個位置帶、兩輪、HotpotQA）、`code/env_h200.sh`、`code/tiara.sbatch`、`code/h200_probe.sh`、`code/gpu_smoke.py`、`code/test_gpu_guard_slurm.py`
- 改：`code/m7_restore_harness.py`（host 可以是低精度的 Enc：限速器照實際位元組計時、資料到齊後才反量化、逐 block 來源與時間、bytes_read、t_dequant、copy_bound；BF16 路徑不變）；`code/m7_model.py`（Qwen2 的 head_dim）；`code/gpu_guard.py`（Slurm 下看不到整個節點 → `UNOBSERVABLE_HOST`；`smi_index`）；`m7_write_policy.py`、`m7_calib.py`、`m7_c6_vllm.py`、`m7_guard_run.py`、`m7_env.sh`、`m7_run_phase1*.sh`（路徑改讀環境變數，**不設時和原本相同**）

**run_id 與指令**（全部 `flock gpu.lock` ＋ `runsh`；venv `tiara`，torch 2.10.0+rocm7.0）
| run_id | 指令 | 結果 |
|:--|:--|:--|
| `20261015-021958-m10-codec-selftest` | `python code/m10_codec.py --self-test --out …/codec_selftest.json` | 全過 |
| `20261015-022041-m10-corr-llama-8k` | `python code/test_m7_correctness.py …/correctness_llama_8k.json` | 全過 |
| `20261015-022121-m10-corr-qwen-8k`、`-022228-…-qwen-32k` | 同上，`M7_MODEL_GLOB=<Qwen>`（32K 加 `M7_TEST_L=32768`） | **失敗**：`AttributeError: 'Qwen2Config' object has no attribute 'head_dim'` |
| `20261015-022155`、`-022255`、`-022325`（`m10-mixed-qwen-{fp8_head,fp8k0_head,int8_token}`） | `python code/test_m10_restore.py mixed --fmt <f> …` | **失敗**：同上 |
| `20261015-022346-m10-corr-qwen-8k`、`-022529-…-qwen-32k` | 修好後重跑 | 全過 |
| `20261015-022340`（int4_kivi，修好之後才開始）、`-022452`（Llama fp8_head）、`-022722`、`-022857`、`-023435`、`-023550`（Qwen 4 種格式） | `test_m10_restore.py mixed --fmt <f>` | 全過 |
| `20261015-022635-m10-regress-llama` | `test_m10_restore.py regress --old <HEAD f6595f0 的 m7_restore_harness.py>` | **失敗**：測試腳本自己的 importlib 錯（沒先放進 sys.modules） |
| `20261015-022822-m10-regress-llama` | 同上，修好腳本後 | **1／6 格沒過**（見異常 4） |
| `20261015-023459`、`-023657`、`-023922`（`m10-regdiag-{old_new,new_new2,old_old2}`） | `regress --impls <順序> --cells local:cake,nfs:cake` | 診斷（事後加的） |
| `20261015-024116` → `-024849`（`m10-regx-{new,old}{1,2,3}`） | `regress --impls new` 或 `old`，每個 process 一個，交替，`--reps 5` | 6／6 格過（判準 02:41:16Z 先寫） |

**關鍵數字**（只寫實測；MI300X、合成 token）
- codec：10 種格式在 GPU 與 CPU 上編出來的位元組、解碼結果全部逐位元相同；沒有 NaN／Inf，沒有值被截在 ±448。Qwen 一個 fp8_head block＝14,680,960 B＝14 MiB＋896 B（和 03 §1 的算術相同）。
- 正確性：Llama 8K 每一項都和 10/8 的 `correctness.json` 相同。Qwen 8K／32K：HF 跑兩次逐位元相同、重算兩次 KV 逐位元相同、主機往返逐位元相同；對 HF 的 logit 最大差 0.109／0.102，argmax 相同，top-5 重疊 5／4；K 對 HF 的最大差（第 0、中間、最後一層）8K：0.5、0.25、2.0；32K：0.5、0.656、4.0（Qwen 的 K 值大，這是 bf16 的 1–2 個最小刻度；相對誤差要等 D0 的 K 分布）。
- 低精度載入：Llama fp8_head、Qwen 4 種格式 × E0／E1／F25／B25 × 只算／只載／Cake，每個 block 都逐位元等於預期（重算＝原本 KV、載 BF16＝原本 KV、載低精度＝decode(encode(原本 KV))），bytes_read、來源計數都對。反量化（未融合的 torch 運算）每 block：Qwen fp8 約 0.27 ms、Llama fp8 約 0.38 ms（E1 只載 64 個 block 的 t_dequant ÷ 64）。
- 全 BF16 回歸（每個 process 一個還原器）：6 格新／舊時間中位數比 0.9975–1.0002，會合點集合相同，KV 全部逐位元相同。

**觀察（彩排、合成 token、一個限速值，不進判定，只證明量測做得出來）**：Qwen、限速約 0.55 GiB/s、Cake：E0 會合在 36；F25 一個 FP8 block 都沒被讀到（來源和 E0 完全相同）；B25 讀到 16 個 FP8；E1 會合點前移到 27。Llama 同樣形狀（E0 34、F25 讀到 0 個、B25 讀到 16 個、E1 25）。

**失敗與異常**
1. **Qwen2 的 config 沒有 `head_dim`**，`KVModel` 一載入就失敗。改成照 HF Qwen2Attention 的預設（hidden_size ÷ head 數）。正是要在 H200 之前抓的那一類錯。
2. **GPU 上「除以 Python 純量」會改成乘倒數**，和 CPU 的真除法差最後一位 → scale 不同 → FP8 位元組不同（第一次自測 8 種格式 GPU≠CPU）。codec 改成除以同裝置的 0 維 tensor（`_div_c`），之後全部相同。H200 上第一天要重跑 `--self-test` 確認。
3. **這兩版 torch（2.10、2.12，ROCm）的 e4m3fn 轉型，超出範圍會變 NaN、不會截在 448**（P04 §3 的記錄是 torch 2.9 NaN、2.13 飽和）。codec 一律自己先截，不依賴 torch。
4. **同一個 process 比新舊還原器，local:cake 那格沒過（新/舊 0.8625，會合 7 對 8）**。診斷：差距跟著「哪個 Restorer 先建立」走，同一份程式互比也重現（先建立的 0.2247 s、會合 7；後建立的約 0.260 s、會合 8），和程式改動無關。後建立的每次載入多約 0.5 ms（9.46 對 8.94 ms），推測是 stream 對到的硬體佇列不同〔推論，沒查 HIP〕。**規則：正式腳本一個 process 只建一個 Restorer**。第一階段的腳本本來就是一個 process 一個，結果不受影響。乾淨比法（每 process 一個）是看到失敗之後才加的，判準在跑之前寫進 `regress_xproc/CRITERION.md`。
5. HotpotQA 官方網址（curtis.ml.cmu.edu）連不上（4 次各約 134 s 逾時），改用 HF `hotpotqa/hotpot_qa` distractor validation（同一份 7,405 題；sha256 `c20b638c…`，記在 `/mlsteam/data/tiara/datasets/hotpotqa/README.txt`）。可用 6,155 題。

**還沒決定、要人決定的**
- T5（cwe）的第 2 輪：目前只有單輪。
- 生成長度：03 §1 寫每題最多 64 token，但上游是 niah 128、vt 30、qa 32、cwe 120；`m10_quality.py` 寫之前要定。
- T5 的 1-shot 範例約 1,000 token，超出 block 0（清單從 block 1／2 開始）；T4 預設不放本題自己的干擾段落（`n_hard_distractors` 可在 D1 調難度時加）。
- §R2／§R3 在 H200 上依序跑的 job 腳本還沒寫（`m7_run_phase1.sh` 含 H200 不做的 C7-anchor，不能直接用）；`busy_levels.json` 要在 H200 上由 C5 重產生。
- `m7_calib.py c2` 的磁碟統計仍寫死 `sda`、查不到記 0（違反規則 7）；C2 在 H200 不跑，沒修，要跑之前必須先修。
- H200_SLURM §4.3 的 env 範例、§5.2 的範本路徑已過時（`env_h200.sh`、`tiara.sbatch` 為準）。

**與計劃的差異**：低精度 block 的反量化排在限速器的 done（資料到齊）之後，在 I/O stream 上執行、和重算搶 GPU（03 §5.4 的表）；BF16 的載入行為不變。

### Milestone 10-pilot（續）— 第 3 步 K／V 分布、第 4 步只用 BF16 調難度、一鍵驗證腳本

**狀態**：PASS（第 3、4 步）；難度參數是**候選**，H200 上用 E0 再確認一次才凍結
**執行時間**：2026-10-15 03:01 → 04:38 UTC
**還是只有 E0（全 BF16）的答對率**；低精度只算重建誤差，沒有任何低精度的答對率。

**程式**：新 `code/m10_kv_stats.py`、`code/m10_quality.py`（目前只有 `tune`＝E0、`summary`）、`code/m10_e0_t4_breakdown.py`、`code/m10_verify.sh`＋`m10_verify_summary.py`（一鍵驗證，給 H200 第一天或 3090）；改 `m10_codec.py`（`rel_err_centered`）、`m10_tasks.py`（`qa_f1`；T4 參數 `max_answer_words`，預設 0＝不變）、`test_m10_restore.py`（`mixed --n`，預設 64＝不變）、`env_h200.sh`（`M10_HOTPOT`）

**run_id**
- K／V 分布：`20261015-030147-m10-kvstats-qwen`、`-030248-…-llama`（第一版）；`20261015-030452-m10-kvstats2-qwen`、`-030742-m10-kvstats2-llama`（加「扣平均後」誤差，`results/m10_pilot_mi300x/kv_stats/` 是這兩個的輸出）
- E0 小測：`20261015-030416-m10-e0-smoke-qwen`（15 題，輸出在 scratch，沒進 results）
- 調難度第 1 輪：`20261015-030854`／`-030856`／`-030858`／`-030900`（`m10-e0p1-{def,hard}-{qwen,llama}`）
- 第 2 輪：`20261015-034403` → `-034417`（`m10-e0p2-*`，8 個）
- 第 3 輪（新 seed 2000 起）：`20261015-040149` → `-040201`（`m10-e0p3-*`，7 個）
- T4 拆解：`20261015-043805-m10-e0-t4-breakdown` → `e0_tune/t4_breakdown.csv`
- 指令：`python code/m10_kv_stats.py --model <m> --out results/m10_pilot_mi300x/kv_stats --raw $TIARA_RUNS/$RUN_ID/raw`；`python code/m10_quality.py tune --model <m> --tasks … --n … [--seed0 2000] --params '<JSON>' --out results/m10_pilot_mi300x/e0_tune/<檔>.csv`（同一個 job 裡多個 process 同時跑；品質 run，時間欄不用）

**關鍵數字（第 3 步，K／V 分布；5 個 32K prompt；模型本身的性質＋codec 重建誤差）**
- Qwen：max|K|＝422（＜448，不會溢位；外部量測約 423）、max|V|＝82；14,336 個 channel 裡 645 個「|平均|／標準差 > 3」、219 個 > 10；最大的一個在第 27 層 head 1，平均 415、標準差 1.7。Llama：max|K|＝34.5、max|V|＝7.6；最大 |平均| 24.9。
- K 的重建誤差（相對 L2 ／ 扣掉平均後）：Qwen fp8_head 2.5%／**18.0%**、fp8_channel 1.7%／12.4%、fp8k0_head 0.4%／**2.7%**、int8_token 1.2%／8.5%、int4_token 11%／84%、int4_kivi 0.8%／6.0%。Llama fp8_head 2.6%／4.2%、fp8k0_head 1.7%／2.7%、int8_token 1.2%／1.9%、int4_token 21%／34%、int4_kivi 3.7%／5.8%。V 兩個模型的 FP8 都約 2.5–3%。
- 讀法：相對 L2 看不出 Qwen 的問題（2.5% 對 Llama 2.6%）；扣掉平均後 Qwen 的 FP8 是 Llama 的 4.3 倍，減掉偏移（K0）後一樣。和 9 月的舊實測方向一致（Qwen FP8 撈針壞、INT8 大致沒壞、INT4 全壞）。**這是重建誤差，不是答對率**；03 §1 的 D1 用重建誤差挑 scale 粒度時，應該用扣平均後的版本〔判讀，計劃文字待改〕。

**關鍵數字（第 4 步，E0 調難度）**：候選參數與理由在 `e0_tune/CANDIDATE_PARAMS.json`。第 3 輪（新 seed、每格 20 題）：
- Llama：T1 6／6 格 100%；T2（100 個干擾）0.70–0.84；T3（6 跳＋2 條干擾鏈）0.59–0.78；T4（答案 ≤2 字）0.70–0.85；T5（freq_cw 40）0.82、（45）0.90。
- Qwen：T1 0.90–1.00（4／6 格 100%）；T2（150 個干擾）0.79–0.95；T3（4 跳＋2 條干擾鏈）0.58–0.90；T4（答案 ≤2 字）0.60–0.75；T5（預設）0.69。
- T4 拆解：沒有篩選時，答案 1–2 個字的題目 Llama 0.73／Qwen 0.64–0.70，3 個字以上的 0.15–0.37（string_match_part）；用 LongBench 的 qa_f1 時 3 字以上升到 0.44–0.60。長答案多半是比對太嚴（"Steve Coogan" 對 "Stephen John Coogan"），也有真的答錯（McLaren 對 Team Penske）。
- 生成長度：第 3 輪只有 Qwen T5 的兩種截法不同（截 64：0.67；截 120：0.69），其餘相同。第 1 輪 Qwen T5 有 13–37% 的題目寫到 120 個 token 的上限。

**一鍵驗證**：`M10_GPU_LOCK=/mlsteam/data/tiara/gpu.lock bash code/m10_verify.sh --quick` 在 MI300X 上 10 項過、0 項失敗（輸出在 scratch）。完整模式沒有整支跑過（第 7 項的各個元件分開驗過）。NVIDIA 路徑全部還沒在 NVIDIA 卡上跑過。

**失敗與異常**
1. `m10-e0-smoke-qwen` 第一次：我的 job 腳本少定義一個變數（`S: unbound variable`），還沒進 Python 就失敗；修好重跑。
2. `20261015-040151-m10-e0p3-L-b`、`-040159-…-Q-b` exit 1：120 題都跑完、逐題 CSV 完整，最後寫摘要時當掉（T3 和 T4 在同一個 run，摘要欄位只取第一列 → `ValueError: dict contains fields not in fieldnames: 'f1_up'…`）。修好 `write_summary`（欄位取聯集），用 `m10_quality.py summary <csv>` 從逐題 CSV 重算摘要，沒有重跑。
3. T4 的改動（`max_answer_words`＝2）是看了 E0 的結果才決定的。只用 BF16，不偏向任何低精度配置（03 §3.2 允許），但要在凍結的事先登記裡寫明。

**還沒決定、要人決定的（新增）**
- T1：兩個模型都碰天花板。照 03 §3.2 不進主判準？還是另做更難的變體？
- T4 的主計分：照計劃用 string_match_part（篩選答案 ≤2 字之後），還是改用 LongBench 的 qa_f1？
- 03 §1 D1「用重建誤差挑 scale 粒度」：改成扣平均後的誤差。

### Milestone 10-pilot（續 2）— 「都做掉」：低精度作答、速度實驗、H200 復現 job、LongBench、T1 加難、數字對照論文

**狀態**：PASS（程式全部寫好、在 MI300X 上結構檢查全過）。**還是沒有任何低精度的答對率**：`m10_quality.py run` 試跑用 `--blind`，低精度列的回答、分數、log-prob、KL 都寫成 BLIND；selfcheck 只比位元。
**執行時間**：2026-10-15 05:20 → 07:54 UTC
**使用者 10/15 的決定**（寫進 03 開頭「10/15 改動」）：T1 另做更難的版本；T4 維持 RULER 比對＋答案 ≤2 字；生成長度 T5 120、其他 64；T5 單輪。

**程式**
- 新：`code/m10_speed.py`（calib-f／noise／k0／grid／summary＋`meets.json`）、`code/r_h200.sh`＋`code/m7_repro_compare.py`（H200 上 §R1–R3 一個 sbatch job：dry-run 34 步；MI300X 對 MI300X 自我比較判「重現」）、`code/m10_longbench.py`（6 任務 × 50 題、官方截斷、自測 9,006 項過）、`code/m10_ruler_cwe_check.py`
- 改：`code/m10_quality.py`（`run`：8 個配置＋@格式、讀法 A／B 重放、兩輪〔第 2 輪 (a)(b)〕、gold log-prob、對 E0 的 KL、逐題可續跑、`--blind`；`selfcheck`）、`code/m7_model.py`（`n_logits`；**model_path 改成載入時才讀 M7_MODEL_GLOB**，見異常 1）、`code/m7_restore_harness.py`（`time_h2d`，預設關＝路徑不變）、`code/m10_tasks.py`（T1：`n_queries`、`n_similar`／`similar_prefix`，預設不變）、`code/env_h200.sh`（不再 mkdir）、`docs/H200_SLURM.md`
- 文件：`docs/precision_20261013/cards/V2_numbers_vs_papers_20261015.md`（我們的數字 vs 論文，26 列）

**run_id**
- 速度彩排（Qwen，縮小規模）：`20261015-053634-m10-spd-calibf-qwen`、`-053705-m10-spd-noise-qwen`、`-053838-m10-spd-k0-qwen`、`-061051-m10-spd-grid-qwen`；H2D 欄位：`-074427-m10-spd-h2d-qwen`（輸出在 scratch）
- T1 加難：`20261015-053838` → `-053852`（第 1 輪）、`-061237` → `-061251`（第 2 輪），`e0_tune/t1_*`
- 上游 RULER CWE：`20261015-054607`（失敗：RULER clone 沒拉 Git LFS，`english_words.json` 是指標檔）、`-061335-m10-ruler-cwe-llama`、`-065849-m10-ruler-cwe-hfgen`
- LongBench E0 小測：`20261015-053944-m10-lb-e0-smoke-llama`（scratch）；接進 m10_quality：`-074530-m10-lb-blind-llama`（scratch）
- 結構檢查：Llama `20261015-060045-m10-qcheck-llama-fp8`；Qwen `-074622-m10-qcheck-qwen-fp8`、`-075010-m10-qcheck-qwen-int4`（修好之後）；失敗的見異常 1
- 遮蔽試跑：`20261015-063545`、`-073041-m10-qblind-qwen`（scratch）
- 還原器加 H2D 計時後的 BF16 回歸：`20261015-073718` → `-074144`（`m10-regx2-*`，判準 07:22:31Z 先寫）；低精度載入 `-074316-m10-mixed2-qwen-fp8`

**關鍵數字**
- 結構檢查：Llama fp8 50／50、Qwen fp8 50／50、Qwen int4_kivi 20／20 過。「真的跑 Cake」和「重放同一個會合點」：KV 與第 1 輪問題的 logits 逐位元相同（最大差 0.0），會合點 b＝15–36；沒讀到低精度時和 E0 逐位元相同。
- BF16 回歸（加 `time_h2d` 之後，每 process 一個）：6／6 格過，新／舊 0.9950–1.0007，會合點相同，KV 全部逐位元相同。
- 速度彩排（Qwen、MI300X、縮小規模，不進判定）：f(i) 23.6 → 72.5 ms；三個限速 b16＝2.77、b32＝0.76、b48＝0.22 GiB/s，E0 會合在 13／31／47（閒）、8／21／38（忙）；k0：b32 時 E1 比 E0 快 32%（閒）／38%（忙），CPU 層（不限速）E1 反而慢 6–10%；E1 的 bytes_read＝E0 的 0.50003 倍；kv_bad 70 格全 0。
- T1 加難：相似鍵（連只差最後 1 個字元）兩個模型仍 0.70–1.00；一次問多個鍵會掉分，但主要是生成長度不夠（Qwen 逐鍵重寫）。64 token 下調不到 60–95% → T1 參數維持預設、不進主判準（CANDIDATE_PARAMS 的 not_in_range.T1）。
- 數字對照論文（V2）：26 列，✅ 16、⚠️ 4、❓ 5、❌ 1。一致的例子：Qwen max|K| 422 對第三方量測 423（同一批 channel）；Llama INT8 每 token 的 K／V 誤差 1.18%／0.79% 對 KVTuner Table 9 的 1.23%／0.79%；FP8 2.5% 對 e4m3 理論值 2.65%；T1–T4 的 BF16 正確率和 RULER、Double-P 一致。
- ❌ 那一列（T5 CWE）：上游 RULER 原版題目在我們的環境，Llama 只有 1.0%（加 BOS）／6.0%（不加），低於 RULER 論文推得出的下限（CWE ≥ 8.2）；我們自己的 T5 題目 47.7%。HF 原生 generate 和我們的還原器 10／10 逐字相同，所以不是還原器的問題；差在 prompt 包裝（推論，沒做消融），也可能是 RULER 版本或框架。H200 上用 vLLM 跑同一份 jsonl 分清楚。不擋精度實驗（T5 是和 E0 配對比較）。

**失敗與異常**
1. **靜默載錯模型（已修）**：`m10_quality.py selfcheck` 先 `import m7_restore_harness`（它會 import m7_model），之後 Answerer 才設 `M7_MODEL_GLOB`；m7_model 在 import 時就把 glob 讀成常數 → **Qwen 的 tokenizer 配上 Llama 的權重**，Qwen 的 token id（最大 151,645）超出 Llama 的詞表（128,256）→ GPU 上 `vectorized_gather_kernel` 越界 → `HSA_STATUS_ERROR_EXCEPTION`，abort。失敗的 run：`20261015-060043`、`-060047`、`-062655`、`-063120`、`-072215`、`-072625`（`m10-qcheck-qwen-*`），以及我的定位腳本。`HIP_LAUNCH_BLOCKING=1` 時沒當掉（越界讀到剛好有配置的記憶體）——也就是說同樣的錯誤可能靜默發生。修法：`model_path()` 在載入時才讀環境變數；Answerer 載入後檢查模型和 tokenizer 是同一個 snapshot、tokenizer 不大於詞表，不一致就丟錯。**檢查過其他程式**：m10_speed、m10_kv_stats、m10_quality 的 tune／run 都是先設 glob 才 import；測試腳本由命令列帶入；Llama 的 run 用的就是預設——今天提交的結果都不受影響（kv_stats 的 summary 也記了實際載入的模型路徑）。
2. 每次 abort 在 cwd 留下約 21 GB 的 `gpucore.*`（兩個落在 repo 根目錄，已刪；之後的 job 改在 scratch 執行）。
3. `m10_quality.py run --blind` 第一次：遮蔽迴圈的變數 `k` 蓋掉外層的題號 → 印進度時 `TypeError`（`-060049`）；已改名。
4. 第一次 4 個 process 同時跑時兩個 Qwen 檢查先當掉，我一開始以為是同時用 GPU 造成的，單獨重跑仍當掉才確定是異常 1。
5. T1 第 2 輪第一次：新腳本少了執行權限，什麼都沒跑就失敗，補上重跑。

**還沒做（要 H200 或要人決定）**
- 03 的事先登記凍結（D1，H200 上 E0 確認難度＋δ*＋scale 粒度之後）；凍結前 `m10_quality.py run` 只准 `--blind`。
- busy_levels：`m7_repro_compare.py busy-levels` 從 H200 的 C5 結果產生（`r_h200.sh` 的 R2-busy-levels 步驟）。
- T5 CWE 的 ❌：H200 上用 vLLM 跑上游 jsonl。

## Milestone 10 — 第二階段（精度）主實驗，MI300X（進行中）

**狀態**：進行中（GPU 佇列照順序跑；這一節每完成一段就補）
**為什麼在 MI300X**：使用者 10/15 晚改動——主實驗在 MI300X 跑完，H200 只做復現（03 開頭「10/15 晚改動」）。
**事先登記**：[07_freeze_mi300x](../docs/precision_20261013/07_freeze_mi300x.md)，凍結 2026-10-15T16:13:06Z（commit 2f4fcfa）；判定程式 `code/m10_analyze.py`（commit 0e79da7 起，selftest 過；d974dfc 補 K3、圖表）。
**產出檔**：`results/m10_precision_mi300x/`（`speed_<m>/`、`fmt_select/`、`analysis/<fp8|int4>/`、`figs/`；之後 `quality/`、`quality_sealed/`）
**佇列**：`jobs/runner.sh` 依名稱順序跑 `jobs/queue/NN_*.sh`，每段拿 `gpu.lock`；品質 run 用 `jobs/pool.py` 同時開多個 process（看 amd-smi 的剩餘 VRAM 才開下一個）。腳本副本：`code/m10_queue_mi300x/`。
**環境**：venv `tiara`（torch 2.10.0+rocm7.0、HIP 7.0.51831、transformers 5.17.0）；每個 run 走 `m7run`（`$TIARA_RUNS/<run_id>/` 有 cmd.sh、context、stdout／stderr、exit_code）。

### 10.1 凍結前（速度校正、雜訊、格式小測）

| run_id | 指令 | 結果 |
|:--|:--|:--|
| `20261015-154558-m10-calibf-llama`、`-154647-m10-calibf-qwen` | `m10_speed.py calib-f --model <m> --out-dir speed_<m>` | f(i) 總和 3.631／3.077 s；pinned H2D 49.9／48.6 GiB/s；限速 b16／b32／b48＝5.2746／1.4668／0.4196（Llama）、2.7712／0.7635／0.2175 GiB/s（Qwen） |
| `20261015-154722-m10-noise-llama`、`-155626-m10-noise-qwen` | `m10_speed.py noise …`（E0，4 頻寬 × 閒／忙，3 process × 10 次） | δ＝2.97% → δ\*＝5%（Llama）；δ＝9.43% → δ\*＝9.43%（Qwen） |
| `20261015-160426`、`-160541`、`-160656`、`-160811`（`m10-fmtsel-{llama,qwen}-s{0,1}`） | `m10_quality.py run --seed0 4000 --n 5 --tasks T2 T3 T4 T5 --modes A --configs E0 E1@<5 種格式> --shard s/2`；`m10_fmt_select.py` | Llama fp8_token、int4_kivi；Qwen fp8_head、int4_kivi（KL 見 07 §3） |
| `20261015-160602-m10-smoke-bpath`、`-160602-m10-smoke-sensmap` | 讀法 B 路徑、敏感度地圖的結構試跑（`--blind`，輸出在 scratch） | 結構正確；沒有看任何低精度答對率 |

### 10.2 FP8 速度格子（K0）——Llama

- **run_id**：`20261015-161226-m10-grid-fp8-llama`（16:12 → 17:50 UTC）、`20261015-175049-m10-k0v-fp8-llama`
- **指令**：`python code/m10_speed.py grid --model llama --fmt fp8_token --base-fmt fp8_token --out-dir results/m10_precision_mi300x/speed_llama --bw-json …/bw.json --blocks-out $TIARA_RUNS/$RUN_ID/blocks.csv`；`m10_speed.py k0-verdict --model llama --from-run <grid run_id> --delta-json …/noise.json`；`m10_analyze.py speed --model llama --summary …/summary_fp8.csv --noise …/noise.json --out-dir analysis/fp8`
- **GPU 監控**：3 個 process 都 CLEAN（`gpu_guard_…_L{0,1,2}.json`：開跑前乾淨、沒有外來 PID、沒有暫停）。
- **產出**：`speed_llama/summary_fp8.csv`、`restores.csv.gz`（逐次還原，2,640 列；未壓縮版 1.1 MB 不進 git）、`meets.json`、`k0_verdict_fp8.json`、`analysis/fp8/speed_llama.json`；圖 `figs/speed_llama_fp8.png`、`figs/sources_llama_fp8_b32.png`。

**K0（32K，還原時間中位數，秒；δ\*＝5%）**

| 頻寬 | GPU | E0 | E1（fp8_token） | E1 快多少 | 配對差 95% CI (s) | E1 贏？ | 會合點 E0／E1 |
|:--|:--|--:|--:|--:|:--|:--|:--|
| b16 | 閒 | 0.578 | 0.378 | 34.6% | [0.196, 0.208] | 是 | 14–16／9–11 |
| b16 | 忙 | 0.666 | 0.501 | 24.7% | [0.163, 0.174] | 是 | 8–9／6 |
| b32 | 閒 | 1.380 | 0.950 | 31.1% | [0.416, 0.442] | 是 | 31–32／22–24 |
| b32 | 忙 | 1.933 | 1.214 | 37.2% | [0.707, 0.722] | 是 | 20–21／14 |
| b48 | 閒 | 2.421 | 1.873 | 22.7% | [0.523, 0.572] | 是 | 47–48／39–40 |
| b48 | 忙 | 3.951 | 2.833 | 28.3% | [1.103, 1.123] | 是 | 38／28–29 |
| cpu | 閒 | 0.095 | 0.072 | 24.8% | [0.019, 0.024] | 是 | 2–3／2 |
| cpu | 忙 | 0.165 | 0.171 | −3.5% | [−0.007, −0.005] | 否 | 2／2 |

- **判定**：K0 不成立（7／8 格 E1 贏）——Llama 上低精度對還原時間有賺頭，往下判品質。
- **速度不輸**（每格 ≤ 最快的混合配置 ×1.05）：E1、E1s、E1sr 全部 8 格都過（`analysis/fp8/speed_llama.json`）。
- 只有 cpu・忙 那一格 E1 比 E0 慢 3.5%：不限速時搬運不是瓶頸，反量化（在 I/O stream 上、和忙碌負載搶 GPU）多出來的時間沒被省下的位元組補回來。和彩排（Qwen、CPU 層 E1 慢 6–10%）方向相同。
- **讀到多少低精度（圖 B，b32）**：F25 的 16 個低精度 block 在閒、忙都 **一個都沒讀到**（全部落在重算段）；B25 16 個全讀到；M25 閒時讀到 10 個、忙時 16 個；R25 閒 9 個、忙 11.5 個（中位數）。這是「前段量化沒掉品質可能只是沒被讀到」的直接證據（03 §3.3）。

### 10.3 FP8 速度格子（K0）——Qwen（L0 監控報污染 → 已整批重量，判定用 10.6）

- **run_id**：`20261015-175050-m10-grid-fp8-qwen`（17:50 → 19:13 UTC）、`20261015-191337-m10-k0v-fp8-qwen`
- **指令**：同 10.2，`--model qwen --fmt fp8_head --base-fmt fp8_head --out-dir results/m10_precision_mi300x/speed_qwen`
- **GPU 監控**：L1、L2 CLEAN；**L0 `CONTAMINATED_DURING_RUN`**。
  - 內容：`intruders=[{pid: -1, first_seen: 2026-10-15T18:16:32.578Z, peak_mib: 21198, n: 1}]`，`n_dirty_samples=1`，`paused_intervals=[]`。
  - pid −1＝AMD 的數量判斷：amd-smi 佔 VRAM 的行程數 − 容器內開 kfd 的行程數 > 0。peak_mib 是卡上全部行程的總和，不是外來行程自己的。
- **判讀（很可能是誤報；但照規則重量）**：
  - 時間點：L0 最後一列「32K・忙」是 18:16:31，第一列「8K・閒」是 18:16:33。中間就是 `Busy.__exit__` 關掉 `m7_busy` 子行程的那一秒。
  - 機制：和 2026-09-15 vLLM 關閉時同一個（`gpu_guard.GpuWatcher.pause` 的說明）——kfd 已關、amd-smi 還列著它的 VRAM 數秒。`m10_speed` 的 Busy 收尾當時沒用 pause。輪詢間隔是 3 s，所以只抓到 1 筆。
  - 逐列旗標：32K 的 1,200 列全部是 `contaminated_gpu=False`；之後 320 列 8K／16K 是 True，`summary` 已排除。
  - L0 的 32K 中位數和 L1、L2 一致：E0、E1 × 16 格，L0 對三個 launch 中位數的差在 −1.8%～+0.6%。
  - 把上一次乾淨取樣之後的 4 列（18:16:29 以後，全是 cpu・忙）拿掉，所有格子的中位數最多變 0.10%。
  - **處理**：CLAUDE.md §3「contaminated 的 run 必須重量」→ 排 stage 22 `m10-regrid-fp8-qwen` 整批重量（輸出 `speed_qwen_regrid/`），Qwen 的 K0／速度不輸以重量那次為準。讀法 B 的會合點仍用這次沒被標污染的列（07 第 54 列）。
- **K0（這次、沒被標污染的列；δ\*＝9.43%）**：限速層 6 格 E1 全贏（快 22.5–39.2%）；**cpu 兩格 E1 都輸**（閒 0.052 → 0.055，−5.5%，CI [−0.004, −0.003]；忙 0.066 → 0.074，−13.2%，CI [−0.009, −0.006]）→ K0 不成立。
- **速度不輸**（`analysis/fp8/speed_qwen.json`）：E1、E1s、E1sr 在 cpu・忙沒過（÷ 最快混合配置＝1.135、1.140、1.103；門檻 1.0943），其他 7 格都過。
  - 照凍結的 `model_verdict`（ε 過而且速度不輸），Qwen 的 K1／K1s 不管品質都不會成立。**（重量後 E1s 過了，這句作廢，見 10.6。）**
  - 和彩排一致：Qwen 的 CPU 層 E1 慢 6–10%。
- **cpu 層分項**（中位數 ms，`restores.csv` 的 t_h2d／t_dequant）：
  - Qwen 閒：E0 H2D 36.6；E1 H2D 17.7＋反量化 10.6 → 還原 52.0 → 54.9。
  - Qwen 忙：H2D 50.5 → 19.7，反量化 26.6 → 還原 65.8 → 74.5。
  - Qwen 每 block 的位元組是 Llama 的 0.4375 倍（只載 E0：1,879,048,192 vs 4,294,967,296 bytes）。
- **讀到多少低精度（b32）**：和 Llama 相同。
  - F25 閒、忙都 0／16，B25 16／16，M25 10／16（閒）、16／16（忙），R25 9 個（閒）、11.5 個（忙）。
  - cpu 層 F25 15–16／16。
- **讀法 B 會合點**（`speed_qwen/meets.json`，b32 閒／忙）：E0 32／21、E1／E1s／E1sr 23／14、F25 32／21、M25 30／18、B25 28／18、R25 30／19。
- **產出**：`speed_qwen/summary_fp8.csv`、`restores.csv.gz`、`meets.json`、`k0_verdict_fp8.json`、`spec_grid_fp8.json`、`analysis/fp8/speed_qwen.json`、`analysis/fp8/tables_speed_qwen.md`、`figs/speed_qwen_fp8.png`、`figs/sources_qwen_fp8_{b16,b32,b48,cpu}.png`。

### 10.4 執行中的修正（19:2x UTC；不動判準）

- **`code/m10_speed.py`**：
  - `Busy(duty, watch)` 收尾時，在 `M10_BUSY_TEARDOWN_PAUSE=1` 下：
    1. 暫停 gpu_guard 取樣；
    2. terminate／wait 背景負載；
    3. 等 amd-smi 連 2 次不再多出 pid −1 條目（最多 30 s）再恢復。
  - 暫停區間寫進 gpu_guard JSON。真的外來行程如果那時還在，恢復後下一筆照樣抓到。
  - 預設關閉，所以正在跑的 stage 21 行為不變。stage 22 起由 `jobs/lib.sh` 打開。
  - 用假的 watcher 和假的 `foreign_on` 測過：開啟時輪詢到連 2 次乾淨才結束，關閉時不輪詢。
- **佇列腳本**：
  - 30／35／55／60／85／86／88 開頭的 `git log … | grep -q .` 在 `pipefail` 下回 141（SIGPIPE），會被當成「07 沒凍結」而整段失敗。改成 `[ -n "$(git log -1 …)" ]`，每段的出題腳本都乾跑過。
  - `pool.py` 加 `pool_skip.txt`（名稱前綴；判定流程走不到的封存工作開跑前略過，log 記 skipped）。
- **佇列順序**：21 → 22（Qwen 重量）→ 23（Qwen E1-INT8 速度重量，10.5）→ 30 → 35（新：FP8 混合配置 F25／M25／B25／R25，兩個模型，Qwen 先，封存）→ 55（只留 FP8 的 T1／LB，加 FP8 混合的 T1）→ 60 → 70 → 75 → 80（INT4 格子，原 40）→ 85／86（原 50／51）→ 88（從 55 拆出的 INT4 T1／LB，要用 80 的會合點）。
  - 理由：Qwen 的 K1／K1s 被速度不輸擋住 → 「兩個模型都過 → INT4」幾乎走不到，Step 2 的機率變高。（重量後這個理由不成立，見 10.6；順序暫不動，判定出來再調。）
- **spec 檔**：`spec_grid.json` 每次 grid 都會覆寫。FP8 那次另存 `spec_grid_fp8.json`（Llama 從 git 取回）；之後的段落自動另存。

### 10.5 Step 2b 的速度（stage 21，19:13 → 19:41 UTC）

- **run_id**：`20261015-191338-m10-grid-2b-fp8k0_token-llama`、`-192112-m10-grid-2b-int8_token-llama`、`-192845-m10-grid-2b-fp8k0_head-qwen`、`-193516-m10-grid-2b-int8_token-qwen`
- **指令**：`m10_speed.py grid --model <m> --fmt <F> --base-fmt <G> --configs E1s --refs --lengths --out-dir speed_<m> …`（32K、只有 E1s、4 頻寬 × 閒／忙、3 process × 10 次）；`summary.csv` → `summary_2b_<F>.csv`
- **GPU 監控**：11／12 個 process CLEAN。**`-193516-…-int8_token-qwen` 的 L1 `CONTAMINATED_DURING_RUN`**：
  - 內容：1 筆 pid −1，19:39:35.79。
  - 時間點：L1 最後一列（忙）19:39:35 之後、L2 第一列 19:40:26 之前，也就是 `m7_busy` 收尾那一刻。和 10.3 同一個機制；stage 21 刻意沒開修正。
  - 240 列的 `contaminated_gpu` 全是 False。
  - 處理：照 CLAUDE.md §3 排 stage 23 重量（輸出 `speed_qwen_regrid/summary_2b_int8_token.csv`），`analyze_2b.sh` 改吃重量那次。
- **meets.json**：合併進 `E1s@fp8k0_token`／`E1s@int8_token`（Llama，b32 閒／忙＝24／14）、`E1s@fp8k0_head`／`E1s@int8_token`（Qwen，23／14）。給 stage 30 讀法 B 用。
- **速度不輸預覽**（`m10_analyze.py speed --x-summary`，輸出在 scratch；對 FP8 格子的混合配置）：
  - E1-INT8 兩個模型 8 格全過。CPU・忙：Llama 1.019、Qwen 1.070。
  - E1-K0 兩個模型都在 CPU・忙沒過（Llama 1.094 > 1.05；Qwen 1.178 > 1.094），其他 7 格全過。

### 10.6 Qwen 速度重量（stage 22、23，19:41 → 21:11 UTC）——判定用這次

- **run_id**：`20261015-194150-m10-regrid-fp8-qwen`（19:41:50 → 21:04:55）、`20261015-210455-m10-k0v-regrid-fp8-qwen`、`20261015-210456-m10-regrid-2b-int8_token-qwen`（21:04:56 → 21:11:44）。三個 exit_code 都是 0。
- **指令**：同 10.3／10.5，`--out-dir results/m10_precision_mi300x/speed_qwen_regrid`，`M10_BUSY_TEARDOWN_PAUSE=1`。
- **GPU 監控**：6 個 process（L0–L2 × 2 個 run）**全部 CLEAN**，`n_dirty_samples=0`。
  - 每個 JSON 都有一段 `paused_intervals`，原因「m7_busy teardown」，長 4.8–7.7 s（例：L0 20:07:32.23 → 20:07:37.18）。
  - `-194150` 的 L2 那段暫停裡丟掉 1 筆取樣（`discarded_samples_overlapping_pause=1`）＝收尾殘影；其他 5 個是 0。
  - 結論：10.3 的污染就是收尾殘影，amd-smi 在背景程式結束後還列出它的 VRAM 約 5 s。
  - 逐列：3,600＋240 列的 `contaminated_gpu` 全是 False；`host_level` 全是 `UNOBSERVABLE_HOST`（AMD 的已知限制）。
- **K0**（`speed_qwen_regrid/k0_verdict_fp8.json`）：限速層 6 格 E1 全贏（快 22.0–39.2%）；cpu 閒 0.052 → 0.055（−5.1%，CI [−0.003, −0.001]）、忙 0.066 → 0.074（−12.6%，CI [−0.009, −0.007]）→ **K0 不成立**，同第一次。
- **和第一次比**（32K、cake、各配置 × 各格的中位數）：差在 −3.8%～+3.4%；唯一例外 E1sr「b16・忙」0.340 → 0.398（+17%）。這一格 E1sr 的 b 只會是 5 或 6（0.34 s／0.40 s 兩群），b＝6 的次數從 12／30 變 17／30，中位數跳群。
- **cpu 層分項**（中位數 ms）：Qwen 閒 E0 H2D 36.5；E1 H2D 17.7＋反量化 10.7 → 還原 52.2 → 54.9。忙：H2D 50.7 → 19.8，反量化 26.1 → 還原 65.8 → 74.1。和第一次差 ≤ 0.5 ms。
- **速度不輸**（`analysis/fp8_regrid/speed_qwen.json`）：cpu・忙 ÷ 最快混合配置＝E1 1.116 ✗、**E1s 1.090 ✓**、E1sr 1.100 ✗（門檻 1.0943）；其他 7 格全過。
  - **和第一次唯一不同的判定**：E1s 從 1.140（✗）變 1.090（✓）。最快混合配置 0.0656 → 0.0664 s，E1s 0.0748 → 0.0724 s。
  - E1s 和 E1 做的事幾乎相同（63 vs 64 個 FP8 block），兩次之間誰快也互換 → 這一格的差異就是雜訊。照規則以重量為準，不改判。
  - 影響：Qwen 的 K1s 可以靠 E1s 成立 → 「兩個模型都過 → INT4」又走得到。是否把速度判準限制在限速層，留給老師。
- **讀法 B 會合點**（`speed_qwen_regrid/meets.json`，b32 閒／忙）：和 10.3 只差 E1sr 閒 23 → 24；E1s@int8_token 23／14 相同。stage 30 照 07 第 54 列仍用 `speed_qwen/meets.json`。
- **讀到多少低精度**：每格和第一次差 0–1 個 block。
- **Step 2b 速度（Qwen，對重量的混合配置）**：E1-INT8（重量）cpu・忙 1.073 ✓、其他 7 格全過；E1-K0（`-192845`，監控 CLEAN，沿用）cpu・忙 1.164 ✗。結論同 10.5。
- **產出**：`speed_qwen_regrid/summary_fp8.csv`、`summary_2b_int8_token.csv`、`restores.csv.gz`、`meets.json`、`k0_verdict_fp8.json`、`spec_grid_fp8.json`、`spec_grid_2b_int8_token.json`、6 個 gpu_guard JSON；`analysis/fp8_regrid/speed_qwen.json`、`tables_speed_qwen.md`；`figs/speed_qwen_fp8_regrid.png`、`figs/sources_qwen_fp8_regrid_{b16,b32,b48,cpu}.png`。
- **失敗與異常**：無。
