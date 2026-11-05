# STATUS_TRX（icpe-trx，TRX40）

> 只由 `icpe-trx` 寫。格式照 STATUS_DATA：日期時間／做完什麼（commit、run_id）／卡住什麼（完整錯誤）／需要別的 session 做什麼／需要使用者決定什麼。最新的一節在最上面。
> 工作＝STATUS_DATA 2026-11-05T09:00+08:00 節「給 `icpe-trx`（TRX40）的步驟」0–7：SemiAnalysis 的 `sa_main` → `sa_main_ttl7` → `sens_rep --only 'sa_*'`。不解讀上限、不下判定。

## 2026-11-05T09:07+08:00 — 轉換完成；**npz 內容指紋 13／13 和 acvlab 不同 → 停**（原因：指紋含 `meta_json`）；G0 跑中

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

**需要使用者決定什麼**：要不要在 acvlab 的新指紋出來之前先往下跑（見對話）。

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
