# STATUS_TRX（icpe-trx，TRX40）

> 只由 `icpe-trx` 寫。格式照 STATUS_DATA：日期時間／做完什麼（commit、run_id）／卡住什麼（完整錯誤）／需要別的 session 做什麼／需要使用者決定什麼。最新的一節在最上面。
> 工作＝STATUS_DATA 2026-11-05T09:00+08:00 節「給 `icpe-trx`（TRX40）的步驟」0–7：SemiAnalysis 的 `sa_main` → `sa_main_ttl7` → `sens_rep --only 'sa_*'`。不解讀上限、不下判定。

## 2026-11-05T09:14+08:00 — 指紋（不含 meta_json）13／13 相同；G0 PASS；xc_sa IDENTICAL；**`sa_main` 開跑（40 worker）**

**做完什麼**
- **merge** `origin/icpe-next-data` @ ec74563 → `828e1d3`（`--no-ff`）。模擬程式 `code/p3x`、`code/icpe/p3_fvar.py`、`code/p3_platform.py` 和 f4f5668 逐位元相同（`git diff` 空）；改到的只有 `next_b_queue.py`（digest 跳過 `meta_json`）、`next_sim_analyze.py`（`--rival-only-gpu`）、`next_a_*` 與文件。
- **指紋**：run `20261105-091102-icpe-trx-npz-digest-nometa`（程式 828e1d3 的 `next_b_queue.py digest`）對 `npz_digest.csv` 的 `content_sha256_no_meta`（acvlab run `20261105-090851-icpe-data-npz-digest-nometa`）：**13／13 SAME**（`DIGEST_RESULT IDENTICAL`）。和本檔 09:07 節的表也相同。icpe-data 的訊息說它比過也是 13／13；這裡是我自己重算、重比的結果。
- **4. G0 PASS**：run `20261105-090000-icpe-trx-g0-f4f5668`（程式 f4f5668、新功能全關、16 worker、nice 10）：
  - 指令：`P3_CONST=h200 P3_TIER=h200cpu M7_TIER_PARAMS=results/icpe_h200/t1/tier_params_h200cpu.json M7_RES=results/m7_write_policy_h200 X1_TRACE_DIR=$ROOT/datasets/mooncake/traces RUN_DIR=<run>/req OMP_NUM_THREADS=1 python -I code/icpe/p3_fvar.py --variant V1 --calib results/icpe_h200/p01/calib --out-root <run>/out x1 sweep --workers 16`（acvlab 的 G0 cmd.sh 在 acvlab 上、這裡看不到；照 STATUS_SIM G1 的指令＋queue 的環境變數組的）。
  - `next_b_g0.py results/icpe_h200/t1/p01/sim_V1/x1/x1_sweep.csv.gz <run>/out/x1/x1_sweep.csv`：**4,416 列 × 29 欄＝128,064 格，0 個不同**（`G0_RESULT IDENTICAL`）。
  - sweep 1,328 個 job 687 s（16 worker）；最大 RSS（單一行程）385 MiB；整組 PSS 最多 2.96 GiB。
  - G0 跑在 merge 前的 f4f5668；merge 沒有改模擬程式（上面），所以沒有重做。
- **5. xc_sa IDENTICAL**：run `20261105-091126-icpe-trx-xcheck-xc_sa`（程式 828e1d3、16 worker、28 job、sweep 40.6 s），比對 run `20261105-091229-icpe-trx-xcheck-compare-xc_sa`：`next_b_g0.py results/icpe_next/b/xcheck/xc_sa/x1_sweep.csv.xz <run>/out/x1/x1_sweep.csv` → **32 列 × 40 欄＝1,280 格，0 個不同**。沒有 collect。
  - 記憶體（09:12:03 一次取樣，`/proc/<pid>/smaps_rollup`）：16 個 worker 的 PSS 149–205 MiB、parent 174 MiB、queue 11 MiB，合計 3.2 GiB。這是 CPU 64 GiB 的小格子；`sa_main` 有到 1,024 GiB 的容量，worker 可能更大，開跑後再量。
- **6. `sa_main` 開跑**：09:12:41，`python -I code/icpe/next_b_queue.py run --plans sa_main --workers 40`（程式 828e1d3；setsid；log `$ROOT/runs/queue-sa_main-w40-20261105-091241.log`；記憶體監看 `…091241.mem.log`，每 30 s，MemAvailable < 6 GiB 就停整組）。開跑時 MemAvailable 55 GiB、swap 245 MiB（基準）、load 6.1。第一個 block `sa_N16_s1`。

**卡住什麼**：無。

**需要別的 session 做什麼**：無。

**需要使用者決定什麼**：無。

## 2026-11-05T09:07+08:00 — 轉換完成；**npz 內容指紋 13／13 和 acvlab 不同 → 停**（原因：指紋含 `meta_json`）；G0 跑中；**等 icpe-data 的新指紋**

**做完什麼**
- **3. 轉換**：run `20261105-085858-icpe-trx-convert-semianalysis`（exit 0）。p3x＝`git archive 6b17191 code/p3x` 解到 `$ROOT/code-archive/p3x-6b17191`。
  - `TIARA_DATASETS=$ROOT/datasets python -I code/icpe/next_b_convert.py semianalysis --N 16 32 64 72 --seeds 1 2 3 --out-dir $ROOT/datasets/x1_nb --p3x <archive>/code/p3x`：3 分 50 秒，最大 RSS 2.03 GiB（`/usr/bin/time -v`）。
  - 截斷版 `semianalysis --N 16 --seeds 1 --trunc-128k ...`：48 秒，最大 RSS 1.66 GiB。
  - 規則 6：`len(hash_ids)×64=in` 98827／98827。
- **指紋比對**：run `20261105-090355-icpe-trx-npz-digest`（exit 1）：`next_b_queue.py digest <13 個名字>` 和 `results/icpe_next/b/traces/npz_digest.csv` **13 個全部不同**（`DIGEST_RESULT DIFFER n=13 bad=13`）。照步驟 3「不同就停」：**SA 的 xcheck／正式模擬都沒開始。**

**原因（診斷，沒有改任何程式）**
- npz 裡除了 8 個資料陣列（`full_key`、`full_off`、`kind`、`nall`、`parent`、`sess`、`t`、`user_trigger`）還有一個 **`meta_json`** 字串陣列，內容含 `run_id`、`ts`、`host`、`argv`（含 `--out-dir`、`--p3x` 的絕對路徑）、`source.file`（絕對路徑）。
- `cmd_digest` 把 `meta_json` 也算進去 → 換一台機器、甚至同一台重轉一次，指紋必然不同。所以 `npz_digest.csv` 的值在別台**照定義就不可能相同**。
- 診斷 run `20261105-090538-icpe-trx-digest-diag`（腳本在該 run 目錄的 `digest_diag.py`，不在 repo）：
  - (a) 同一套算法、**跳過 `meta_json`** 的指紋 `content_sha256_no_meta`（下表）——acvlab 沒有這個值可比。
  - (b) `meta_json` 解出來的統計欄，和 git 裡 acvlab 的 `results/icpe_next/b/traces/trace_summary.csv` 同名列逐欄比（n_conv、active_median／p10、n_req、n_sess、n_cont、full_chunks_total、n_keys、span_h、inp_tokens_median／p90、frac_inp_gt_128k、trunc_frac_req_cut、time_fix_within_session、rule6）：**13／13 全部相同**（`SUMMARY_FIELDS_RESULT ALL_SAME`）。這是彙總量相同，不是逐位元相同。
  - 先前兩個診斷 run `…090500…`（腳本 bug：`meta_json` 是 0 維，`ABORTED`）、`…090513…`（欄位對應錯：n_keys 在 `meta.check`，`SUPERSEDED`）作廢。

| name | content_sha256_no_meta（TRX40） |
|:--|:--|
| semianalysis_N16_s1 | 3d6192cae774cdb66036527301ded814692b0b457d1f8c926907a9cb4c962b04 |
| semianalysis_N16_s1_t128k | f51da2d56ca961a47110ea13061ae2ee28b01879ad42b18956b25a461e81608f |
| semianalysis_N16_s2 | 460bb9572953b039ba3d589a97afa80549fd6ec309aee8fc62f09f93890b52fd |
| semianalysis_N16_s3 | 8ab19616850d09c155f58b9d53838dab92ebf0b4321cd9133d05c7068120eefe |
| semianalysis_N32_s1 | 517b8bbe9ce428ba7bb76eb76aba54ddf9404bb6c5162fb2f6680efa743a8b7a |
| semianalysis_N32_s2 | 0fb05f6661031aa0b8b8b93bee3a4a4484c9d02875d66f4598904ff05656ff4c |
| semianalysis_N32_s3 | 6ceac31f0c802142e7ba0f8ac9fe6d8f9ba8a6699bc0604743b3258773481584 |
| semianalysis_N64_s1 | f844e1a3b954bfccbff9814c25d4cddb8da50e69030aae5675f48fddc3b0c038 |
| semianalysis_N64_s2 | f5fd5d2534b099cf57e7f63261992c3c4444df73da48bde2378ace2aa18e54b4 |
| semianalysis_N64_s3 | 2074092728cf504f15fcf7337bff9114ada9008b11bec727b91ac7a8fa2d09b6 |
| semianalysis_N72_s1 | ef0f0a9fbd1832f7025886376353475a1320328ba2bc7235144352867d5553a1 |
| semianalysis_N72_s2 | d12b3468afc7c667bea791332e1f7a879bf9e748cd1733cce8ce4ef6049872e6 |
| semianalysis_N72_s3 | af8752b4bf015d9b2227be6d51c6d71594d7cdf36edc8bf829958de1ff190686 |

**跑中**：G0 run `20261105-090000-icpe-trx-g0-f4f5668`（Mooncake，16 worker；和 SA 的紀錄無關，所以沒停）。

**卡住什麼**：SA 紀錄的跨機器相等性沒有逐位元的證據（上面的原因）。

**需要 icpe-data 做什麼**
- `cmd_digest` 跳過 `meta_json`（或只留 meta 裡和機器無關的欄），在 acvlab 對同 13 個名字重算，寫進 `npz_digest.csv`（或另一個欄）。我會和上表逐名比。上表的算法＝`cmd_digest` 原樣、只在 `k == "meta_json"` 時 `continue`。
- 同一個問題也影響 h200 的 Bailian／FI 指紋比對。

**使用者決定（11/5 09:08，在 icpe-trx）**
- **等 icpe-data 的新指紋**：acvlab 用「不含 `meta_json`」的算法重算、寫進 git，13 個都和上表相同，才跑 `xc_sa` → `sa_main`。在那之前 SA 什麼都不跑（xcheck 也不跑）。
- **`sa_main` 直接 40 worker**（使用者：「CPU：48 執行緒 改吃40?」→ 選「sa_main 直接 40」）：第一個 block 就開 40，不先用 16 量。保險＝記憶體監看（MemAvailable < 6 GiB 就停整組；被停的 block 作廢、降 worker 重跑）。swap 以 245 MiB 為基準，增加就記錄並回報。

## 2026-11-05T09:00+08:00 — repo、venv、下載完成；轉換＋G0 跑中

**機器**：`acvlab-TRX40-AORUS-MASTER`，Threadripper 3960X 48 執行緒，62 GiB RAM，swap 8 GiB（開跑前別人已用約 245 MiB，以這個為基準，不准再增加）。Titan RTX 不碰。

**做完什麼**
- **0. Repo**：用本機已有的 clone（`/home/hungwei/paper-hierarchical-kv-state`），分支 `icpe-next-trx` 從 `origin/icpe-next-data` @ `f4f5668` 開。
  - `$ROOT=/ssd4/hungwei/icpe-next`（/ssd4 剩 958G、45%），底下 `venv/`、`uv-python/`、`datasets/`、`runs/`、`hf-home/`、`tools/`、`code-archive/`。程式在 repo 裡，和資料分開。
- **1. venv**：`$ROOT/venv`，Python **3.12.15**、numpy **2.3.5**、pandas **3.0.5**、pyarrow **25.0.1**、python-dateutil **4.0.0**、tzdata **2026.6**（`uv pip freeze` 只有這 5 個）。
  - 這台沒有 `uv`、`curl`、`pip`。另外用 micromamba（conda-forge）建工具環境 `$ROOT/tools/env`（uv 0.13.7、curl 8.23.0、huggingface_hub 的 `hf` 2.4.0），只用來裝 Python 與下載，和模擬用的 venv 分開。
  - Python：`uv python install 3.12.15 --install-dir $ROOT/uv-python --no-bin`（cpython-3.12.15-linux-x86_64-gnu）。
  - repo 的 `.claude/settings.json` 把 `HF_HOME`、`UV_*` 指到 `/ssd7`（這台沒有），還設了 `HF_HUB_ENABLE_HF_TRANSFER=1`；每條指令都明確蓋掉／unset。
- **2. 下載**：run `20261105-085632-icpe-trx-dl-semianalysis-mooncake`（exit 0）。`TIARA_SKIP_VENV=1 TIARA_HOME=$ROOT TIARA_DATASETS=$ROOT/datasets HF_HOME=$ROOT/hf-home TIARA_PY=$ROOT/venv/bin/python TIARA_HF_CLI=$ROOT/tools/env/bin/hf bash code/setup_3090_data.sh semianalysis mooncake`。
  - SemiAnalysis：revision 23f152f6，`hf_verify semianalysisai/cc-traces-weka-062126 sha=23f152f6f0f9399a85901b89a6458def0ef16729 files=3 bad=0`；`traces.jsonl` sha256 `29b6a19e…`。
  - Mooncake：commit d750cdda，6 個檔 git-blob OK；conversation sha256 `b8cbb061…`、toolagent `48a2db1a…`（和指定的相同）；單位驗算 4 份都是 rate=1.000000。

**卡住什麼**：無。

**需要別的 session 做什麼**：無。

**需要使用者決定什麼**：無。
