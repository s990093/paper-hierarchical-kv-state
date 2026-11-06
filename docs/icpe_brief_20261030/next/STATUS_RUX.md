# STATUS_RUX（icpe-rux，RUX3DPRO）

本檔由 icpe-rux 寫，最新的一節在最上面。格式照 `STATUS_TRX.md`。
**本次只做「環境＋對拍」：沒有開跑任何正式模擬、沒有動佇列、沒有改別人的檔。**

---

## 2026-11-07T00:15+08:00 — 環境建好；**對拍全部通過**：npz 指紋 13／13 SAME、G0（py／rust）、xc_sa（py／rust）、雜訊 B1／B2 全部 0 個不同

**機器（實測，開跑前記錄）**
- `nproc`＝**16**；Ryzen 7 7800X3D（8 核 16 執行緒、1 socket）。
- `free -g`：total **22**、used 2、available 20；swap **6 GiB**（開跑前已用 0 MiB，以此為基準）。
- `df -h /`＝`/dev/sdd` **1007 G**，可用 947 G；`$HOME=/root` 就在 `/dev/sdd` 上（`stat -c %d`＝2096，與 `/` 同）。
- `/etc/wsl.conf`：只有 `[boot] systemd=true`。Windows 側 `C:\Users\asdf9\` **沒有 `.wslconfig`**（使用者未提供內容）。
- `MemTotal` 24,028,820 kB＝22.9 GiB；`MemAvailable` 開跑前 21,584,640 kB。
- 沒有 NVIDIA 顯卡（`nvidia-smi` 不存在，只有 WSL 的 `/dev/dxg`）。

**0. 目錄**
- `$ROOT=/root/icpe-next`，在 `/dev/sdd` 上。底下 `venv/ uv-python/ datasets/ runs/ hf-home/ tools/`（另有 `x1fast-work/`）。
- 程式和資料分開：repo 在 `$HOME/paper-hierarchical-kv-state`（照 TRX40 的配置）。

**1. Repo**
- clone `https://github.com/s990093/paper-hierarchical-kv-state`（public），從 **`origin/icpe-next-data` @ `8d70cbca`** 開分支 **`icpe-next-rux`**（`git checkout -B icpe-next-rux origin/icpe-next-data`）。
- repo 的 `.claude/settings.json` 把 `HF_HOME`、`UV_*` 指到 `/ssd7`、並設 `HF_HUB_ENABLE_HF_TRANSFER=1`；每條指令都明確覆蓋或 unset（`tools/env-rux.sh`）。

**2. venv（`$ROOT/venv`）**
- Python **3.12.15**（`cpython-3.12.15-linux-x86_64-gnu`，`uv python install --install-dir $ROOT/uv-python --no-bin`）。
- `uv pip freeze` **正好 5 個**：`numpy==2.3.5`、`pandas==3.0.5`、`pyarrow==25.0.1`、`python-dateutil==4.0.0`、`tzdata==2026.6`。
- 只用來建 Python／下載的 `uv 0.13.7`、`hf 2.4.0`（huggingface_hub）放在 **另一個** venv `$ROOT/tools/hf-venv`，不污染模擬用的 venv。

**3. Rust（`$ROOT/tools`）**
- rustup 裝到 `RUSTUP_HOME=$ROOT/tools/rustup`、`CARGO_HOME=$ROOT/tools/cargo-home`，`--default-toolchain 1.99.1`：
  - `rustc 1.99.1 (58cf8f9c3 2026-10-13)`、`cargo 1.99.1 (5f94df478 2026-08-27)`——和 acvlab／TRX40 相同。
- 環境：`X1_CARGO=$ROOT/tools/cargo-home/bin/cargo CARGO_HOME=$ROOT/tools/cargo-home RUSTUP_HOME=$ROOT/tools/rustup X1_RUST_WORK=$ROOT/x1fast-work`。
- 編譯照 `code/p3x/x1_rust.py`（`--profile fast --offline --locked`，不加 `target-cpu`）：試編 run `20261106-233220-icpe-rux-buildtest` exit 0、**3.79 s**。
- `code/p3x/x1fast` 的 `[dependencies]` 是空的，所以 `--offline` 不需要 registry 快取。

**4. 資料**（run `20261106-233235-icpe-rux-dl`，exit 0；`setup_3090_data.sh mooncake semianalysis bailian tracelab`）
- 環境照 TRX40：`TIARA_SKIP_VENV=1 TIARA_HOME=$ROOT TIARA_DATASETS=$ROOT/datasets HF_HOME=$ROOT/hf-home TIARA_PY=$ROOT/venv/bin/python TIARA_HF_CLI=$ROOT/tools/hf-venv/bin/hf`。
- **CacheWise 沒有下載、沒有出現在這台**（`ls $ROOT/datasets` 只有 mooncake／semianalysis_cc_traces／qwen-bailian-usagetraces-anon／tracelab）。

| 資料 | 要求的值 | 本機實測 | 結果 |
|:--|:--|:--|:--|
| Mooncake conversation | `b8cbb061…` | `b8cbb061a85206d729d91cdc2981f43c9e0d99209dce588d3af5f7934408b9df` | SAME |
| Mooncake toolagent | `48a2db1a…` | `48a2db1a13d3bc05e6330140c64f604ba366df20d3c9e128b5c35a01c1fa5f71` | SAME |
| Mooncake commit | `d750cdda` | 6 個檔 git-blob OK（GitHub tree API） | SAME |
| SemiAnalysis revision | `23f152f6` | `23f152f6f0f9399a85901b89a6458def0ef16729` | SAME |
| SemiAnalysis `traces.jsonl` | `29b6a19e…` | `29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276` | SAME |
| TraceLab release | v0.0.2 / `11ce51ec…` | `11ce51ec0a25e3d1d95b025bca2f7d1647e47571eb7cc968acd5fc64d4b4fb65` | SAME（完整值相符） |
| Bailian commit | `5f7439c5` | `5f7439c51ec248a0c585f7d90a41a6f57773b912`；4 個 jsonl 的 sha256＝LFS pointer oid | SAME |

- Mooncake 單位驗算（規則 6，block 512）：conversation 12031／12031、synthetic 3993／3993、toolagent 23608／23608、mooncake 23608／23608，**rate 全部 1.000000**。
- 下載後的單位驗算與 `m8_trace_oracle.load_trace` 斷言全部通過（`[mooncake] OK`、`bailian OK`、`tracelab OK`）。

**轉換**（run `20261106-233531-icpe-rux-convert-sa`，exit 0）
- `next_b_convert.py semianalysis --N 16 32 64 72 --seeds 1 2 3 --out-dir $ROOT/datasets/x1_nb --p3x $REPO/code/p3x`，再跑 `--N 16 --seeds 1 --trunc-128k`。共 13 個 npz。
- 規則 6：`len(hash_ids)×64=in` **98827／98827 = 1.0000**。每個變體 `n_req=98827 n_sess=2090 chunks=42212088`；截斷版 `n_req=40799 n_sess=2079 chunks=4785716`。
- 用 **HEAD 的 `code/p3x`**（`x1_nb.py` sha256 `8251e258…`，和 STATUS_DATA 11/6 09:35 宣告的 acvlab 值 `8251e25` 相同）。TRX40 當時用的是 `git archive 6b17191`；兩版 `x1_nb.py` 的差異只在 `check()` 向量化，且 `write_nb` 是 `st, _ = check(...)`——稠密 id 被丟棄，**輸出陣列不受影響**；下面的指紋也證實了這點。

**npz 內容指紋（不含 `meta_json`）**
- run `20261106-233907-icpe-rux-npz-digest-nometa2`（算）、`20261106-233920-icpe-rux-compare-npz-digest`（比）。
- 比對基準：`results/icpe_next/b/traces/npz_digest.csv`（acvlab，run `20261105-090851-icpe-data-npz-digest-nometa`）。
- **`DIGEST_RESULT IDENTICAL n=13 bad=0`**——13／13 SAME。

**5. 對拍（全部 0 個不同）**

| 項目 | run_id | 結果 | sweep | 牆鐘 | CPU(user) | 最大 RSS | swap |
|:--|:--|:--|--:|--:|--:|--:|:--|
| (a) G0 Python | `20261106-233928-icpe-rux-g0-py` | 4,416 列 × 29 欄＝128,064 格，**0 不同** | 922.9 s | 924.5 s | 10,788.3 s | 386 MiB | 0 |
| (b) G0 Rust | `20261106-235852-icpe-rux-g0-rust` | 同上，**0 不同** | 40.3 s | 46.1 s | 429.0 s | 308 MiB | 0 |
| (c1) xc_sa Python | `20261107-000317-icpe-rux-xcheck-xc_sa-py` | 32 列 × 40 欄＝1,280 格，**0 不同** | 46.4 s | 54.5 s | 448.2 s | 2,111 MiB | 0 |
| (c2) xc_sa Rust | `20261107-000429-icpe-rux-xcheck-xc_sa-rust` | 同上，**0 不同** | 2.07 s | 11.7 s | 19.9 s | 2,855 MiB | 0 |
| (d1) 雜訊 B1 | `20261107-000520-icpe-rux-p4-noise-subset2/sa_p4_b1_semianalysis_N16_s1_P_B1` | 1,394 列 × 29 欄＝40,426 格，**0 不同** | 19.4 s | 20.3 s | 226.6 s | 1,029 MiB | 0 |
| (d2) 雜訊 B2 | `20261107-000520-icpe-rux-p4-noise-subset2/sa_p4_b2_semianalysis_N16_s1_P_B2` | 1,360 列 × 29 欄＝39,440 格，**0 不同** | 17.2 s | 17.8 s | 203.8 s | 1,029 MiB | 0 |

- 全部 `--workers 12`（這台 22 GiB，照指示 ≤12）；`nice -n 10`。全部 `Swaps: 0`。
- 記憶體監看（15 s 一次，`MemAvailable < 3 GiB` 就停整組）：G0-Python 最低 **19.93 GiB**、G0-Rust 最低 **19.96 GiB**、xc_sa-Python 最低 **18.08 GiB**；xc_sa-Rust 與兩個雜訊 run 沒開監看 → 那幾項的 MemAvailable 與 load 峰值 **NOT_MEASURED**（`/usr/bin/time -v` 的 `Swaps: 0` 有量到）。load 峰值：G0-Python 12.24、G0-Rust 6.41、xc_sa-Python 5.79，其餘 NOT_MEASURED。
- **(a)(b) 比對**：`next_b_g0.py results/icpe_h200/t1/p01/sim_V1/x1/x1_sweep.csv.gz <run>/out/x1/x1_sweep.csv`（run `20261106-235845-icpe-rux-compare-g0-py`、`20261107-000309-icpe-rux-compare-g0-rust`）。
- **(c) 比對**：`next_b_g0.py results/icpe_next/b/xcheck/xc_sa/x1_sweep.csv.xz <run>/out/x1/x1_sweep.csv`（run `20261107-000429-icpe-rux-compare-xc_sa-py`、`20261107-000452-icpe-rux-compare-xc_sa-rust`）。xc_sa 的參數照 queue 的 `xcheck` 定義：`--cpu 64 --policies LRU GDSF O-BEL O-PF --slows ssd --modes cake --gpu-gib 111.0625 --gpu-write wb --gpu-policies LRU BEL TTLA:10 TTLB:10 --pf-target cpu`。**沒有 collect。**
- **(d) 做法**：`sa_p4_b1`（B1）與 `sa_p4_b2`（B2）各挑一個 block－`semianalysis_N16_s1_P_B1`、`semianalysis_N16_s1_P_B2`。每個 block 共 287 個 job，用 `--jobs-from` 取 **16 個**（等距抽）。參考檔取自 `origin/icpe-next-trx:results/icpe_next/b/<plan>/<block>/x1_sweep.csv.xz`。`--engine rust`，`py_fallback_jobs=0 {}`（B2 沒有退回 Python）。
  - ⚠ **`next_b_g0.py` 不能直接用在這兩個 block**：兩邊的 CSV 都沒有 `slow_cap_frac`／`slow_policy`，所以比對器退回 7 欄鍵，而雜訊 block 的 7 欄鍵**不唯一**（B1：1,394 列只有 33 個相異 7 欄鍵）。改用 run 目錄裡的 `cmp_noise.py`：鍵＝7 欄＋`gpu_GiB gpu_write gpu_policy pf_target noise_model noise_lambda noise_draw noise_seed`（15 欄），要求鍵唯一且每一個非鍵欄逐格字串相同。**這是這台新寫的比對腳本，不在 repo 裡。**
  - `--jobs-from` 是 **job 粒度**（一個 job 的多個 view 都會跑），所以產出列數會比 jobs.csv 的 7 元組多：16 job → B1 1,394 列、B2 1,360 列。

**速度（只代表這台的相對速度，不能拿來下任何時間的結論）**
- G0：Python 922.9 s（12 worker）→ Rust 40.3 s，**22.9×**（同一台、同 12 worker）。
- xc_sa：Python 46.4 s → Rust 2.07 s，**22.4×**。
- G0 的 worker-秒數：Python 922.9×12＝11,075、TRX40 是 687×16＝10,992（16 worker）——**每核速度與 TRX40 幾乎相同**〔推算，同程式、不同 worker 數〕。

**卡住什麼**：無。

**需要別的 session 做什麼**：無。

**需要使用者決定什麼**
- **push**：`icpe-next-rux` 目前只有本機 commit，**還沒 push**（照規則 push 前先問）。
- 之後要不要把正式的 block 分給這台跑（本節沒有跑任何正式模擬）。

**已知落差／限制**
- 這台是 **WSL2**：Windows 休眠、自動更新重開機、WSL 閒置被關都會打斷 job。今天 13:38:50 就正常關機過一次、23:00:28 才再開（本 session 前半段連不上就是這個原因）。**開跑前需要使用者去關掉 Windows 的睡眠與自動重開機**，我沒有改任何 Windows 設定。
- 記憶體 22 GiB，比 acvlab（1 TiB）與 TRX40（62 GiB）小很多；本次對拍最高單一行程 RSS 2,855 MiB（xc_sa Rust），G0 只有 386 MiB。正式 block 的 worker 數還沒量過。
- 沒有 GPU，所有 GPU 相關的實機實驗與 vLLM 都不能在這台跑。
