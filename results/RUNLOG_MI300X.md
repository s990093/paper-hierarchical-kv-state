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

