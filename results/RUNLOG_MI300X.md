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
