# STATUS_RUX（icpe-rux，RUX3DPRO）

本檔由 icpe-rux 寫，最新的一節在最上面。格式照 `STATUS_TRX.md`。
**到目前為止：沒有開跑任何正式模擬、沒有動佇列、沒有改別人的檔。**

---

## 2026-11-07T04:30+08:00 — 第 4 段（v8：第 2–5 項的新功能，只有 Rust）：APC、W5 完成並接進 sweep；W3 引擎完成、缺資料；排隊**要使用者決定 Q3-a**

報告：`next/rux/V8_FEATURES.md`；語意：`next/rux/SEMANTICS_2to5.md`；程式：`code/icpe/rux_x1fast/v8_features/`（最終版 `main.rs` sha256 `a4ded3ca…`）。

**驗證（沒有 Python 參考，照使用者同意的做法）**
- **新功能關閉時**：v8＝Python。3 份紀錄 × 4 組子集，CSV 177,072 格、npz 11,424 個陣列，0 個不同（`…-v8f-reg-mc／bl／sa`）。G1、xc_sa 0 個不同（`…-v8f-off-g1`、`…-v8f-off-xc_sa`）。
- **四項新功能**：最終版 `smoke_v8.py` 在 3 份紀錄上共 2,088 項，**0 個 FAIL**（`…-v8f-smoke-mc／sa／bl`）。檢查方式：
  - 退化檢查：新功能設成等於沒開時＝v7。
  - 精確重算：排隊。
  - 不變量：W5 的位元組。
  - 稽核：APC 每 200 個請求從頭重算 GPU 引用數。
  - 兩份實作互比：BELP 的捷徑＝每塊全掃。
- **端到端**：
  - sweep 管線重現版本 B 的 toolagent 上限表（306 列逐值相同）。
  - APC 一次逐出整個 session 時＝LRU／BEL。
  - W5 走新路徑但參數等於沒開時＝BF16：17 × 16 個規則，254,592 格相同。
  - BELP 捷徑：832 格相同、快 2.3 倍。

**探索（Mooncake toolagent R1／R2；不是正式結果）**
- APC 加進 ② 和 ④（另外跑了 SemiAnalysis N16 s1、Bailian coder；三份都重現版本 B 的表，逐值相同）：
  - Mooncake toolagent、Bailian：上限最多變 0.03 個百分點。
  - **SemiAnalysis（長 session）明顯**：APC 在每格都是 ② 的最佳，② 最多快 28%；O-GPU（④ 加 BELP）上限中位數 8.5% → 4.6%，111 GiB、nfs 從 18–21% 降到 6–9%，≥15% 的格 6 → 4。
  - **④ 一定要有 BELP**：沒有的話，SemiAnalysis 的 O-GPU 最大上限會被壓到 25%（有 BELP 是 44%）。
  - 原本的 `next_sim_analyze.py` 會把 BELP 當成不預測。所以另寫 `v8_features/analyze_v8.py`（原檔不改）：BELP 只進 ④ 的 O-GPU。已檢查和原本的分析、獨立算法都 0 個不同。
  - 建議 PR:95 的重算至少包含長 session 的紀錄（SemiAnalysis、TraceLab、CacheWise）。
- W5（CPU 111／128／256 GiB；另外跑了 SemiAnalysis、Bailian：多半照 NP:53 下降，SemiAnalysis 的 O-GPU ≥15% 的格 4 → 0、Bailian 8 → 2；下面的上升主要是 Mooncake toolagent 的現象）：
  - 上限有升有降：O-GPU −6 到 −8 個百分點；O-DROP、O-ORD 上升，nfs 的格最多 +39 個百分點。
  - **違反 NP:53 事先登記的預期**（「下降或不變；若上升，要查原因」）。
  - 查了上升超過 5 個百分點的 45 個 (格, 元件)：O-DROP、O-ORD、O-PF 的 ④ 用 INT4 多出來的 CPU 容量，慢層載入減 79–99.5%，② 只減 13–26%；大多在 nfs。

**還沒做／要決定**
- **Q3-a（使用者）**：排隊的 ④ 知不知道排隊造成的新服務順序。
  - open-loop（`q=1`）在 SemiAnalysis 上平均要等 2,000–19,000 秒；照紀錄順序的 closed-loop（`q=2`）多等 30–300 倍。兩個都不能用。
  - 正確的工作守恆 closed-loop 要先決定這件事。
- **W3（icpe-data 在 acvlab 上跑）**：工具鏈都好了，指令在 `v8_features/README.md`。流程是 make_cue → check → sweep CUEP／CUEO → analyze_w3，③ 只要約 450 個 job，不用改 nb 檔。這台記憶體不夠跑 TraceLab，所以只在 SemiAnalysis 上用代用 cue 跑通管線。TraceLab 原始紀錄上，「輸出有工具呼叫」和「下一輪由工具結果觸發」一致 96.9%。
- **登記（h200／icpe-data）**：SEMANTICS_2to5.md 是 PREREG 的草稿。
- **審查（icpe-sim 或另一個 session）**：審語意與程式的對應，決定要不要併進 `code/p3x/x1fast`。

**規則遵守**
- 每個 run 都用 `rux_run.sh`。
- 舊執行檔上的兩個測試 run（`20261107-033505-…-v8-smoke-all-sa／bl`）被最終版取代，是我停掉的，所以沒有 exit_code。停掉前 SemiAnalysis 476 PASS、Bailian 593 PASS，0 FAIL。
- 沒有跑正式 block、沒有動佇列、沒有改別人的檔。

---

## 2026-11-07T03:45+08:00 — 第 2 段（v6：慢層上限的 Rust 版）與第 3 段（v7：加速）完成，**和 Python 逐位元相同**；第 4 段（第 2–5 項的新功能，只有 Rust）進行中

**使用者的指示（原話）**
- 「ok push 跟 後段都你做不需要跟其他session 做」
- 「好就這樣」：第 2 段之後接著做低風險的加速。
- 「也就是 python 不寫完整移植 rust」：之後的新功能只寫 Rust。
- 「第 2 到 5 細節 都可以你決定 符合實驗命題且有規範 其他 paper 合理這樣 我要睡了 都交給妳」
- 「這太電腦改用更高執行緒加速 也可」：Rust 驗證改用 16 個 worker。

**第 2 段：v6（`code/icpe/rux_x1fast/v6_slowcap/`；報告 `next/rux/V6_SLOWCAP.md`、`V6_rows.csv`）**
- 慢層容量上限照 `x1_sim.py` 一字不改移植。`code/p3x/*` 沒動。
- **功能打開**：3 份紀錄（Mooncake conversation R1、Bailian coder、SemiAnalysis N16 s1）× 4 組子集（main、wt、pfgpu、noise B1／B2）× 7 種慢層規則。對 Python：CSV 177,072 格、npz 11,424 個陣列，**0 個不同**。
- **功能關閉**：G1 128,064 格、xc_sa 1,280 格，0 個不同。
- **第一版有錯，已改**：`make_room` 尾端少了一次 `_slow_enforce`（py:1580–1581）。Mooncake 主子集有 225 格不同，照規則停下、改好、全部重跑（run `20261107-004914-…-v6-val-mc`，§3c）。
- **速度**：
  - 主子集 12 worker，Python ÷ Rust＝26.3×（Mooncake）、51.1×（Bailian）、44.2×（SemiAnalysis）。
  - 單執行緒同一個 job 20–66×。
  - 正式格子每種紀錄在這台約 0.5–2.3 小時〔推算〕。

**第 3 段：v7（`v7_fast/`；報告 `V7_FAST.md`、`V7_rows.csv`）**
- 做了 ANALYSIS §3d 的候選 2（LRU2 有序集合）、3（PENS 成本快取）、5（快速路徑＋補記 `clean_q`）。
- 驗證模式與一般模式各對拍 3 份紀錄 × 4 組子集，重用 v6 那次的 Python 參考，**0 個不同**。G1、xc_sa 也 0 個不同。
- 單執行緒 v6 ÷ v7：
  - Bailian 慢層 LRU2 12.6×、PENS 2.2×；
  - Mooncake LRU2 3.1×；
  - SemiAnalysis 各規則 1.16–1.26×。
- 16 worker 主子集：Bailian 54.5 → 37.1 s，SemiAnalysis 71.8 → 60.2 s。
- 候選 1（慢層 BEL lazy heap）、4（7 種規則共用前段）**沒做**。

**第 4 段（進行中）：第 2–5 項（`v8_features/`；語意 `next/rux/SEMANTICS_2to5.md`）**
- 語意由 icpe-rux 決定（使用者授權），依據 PR、NP、SS 與外部系統。
  - APC：session LRU，從尾端一塊一塊逐出。wb 下第一次逐出時整個 session 寫進 CPU；④ 也有 BELP。
  - 排隊：FCFS-1，只跑固定並行的紀錄。
  - W3：CUEP（釘住）／CUEO（立刻逐出）。
  - W5：CPU 層 INT4＋M3-g 的一次拆完殘差。
- **要登記**：PR:57、95 要求跑之前登記。PREREG 由 h200／icpe-data 寫，`SEMANTICS_2to5.md` 是登記內容的草稿。
- **W3 的資料**：nb 檔現在只存「使用者觸發」，「none」會被當成工具觸發。要 icpe-data 在 nb 檔多存一欄「工具觸發」。
- 狀態與驗證結果見下一節（做完再寫）。

**規則遵守**
- 每個 run 都用 `rux_run.sh`（cmd.sh、context.txt、exit_code、mem.log）。v6／v7 共 21 個 run，MemAvailable 最低 10.4 GiB。
- 沒有跑正式 block、沒有動佇列、沒有改別人的檔。CacheWise 不在這台。

---

## 2026-11-07T00:40+08:00 — 第 1 段（分析）完成：未支援的 7 項模擬的形態、時間、Rust 缺口、加速候選；**停下來等使用者決定**

完整報告：`docs/icpe_brief_20261030/next/rux/ANALYSIS.md`；摘要 CSV：同目錄 `ANALYSIS_rows.csv`（每列有 run_id、ts、host、threads、load）。

**做完什麼**
- 前提：重新抽查第一份 prompt 的對拍，全部 0 個不同；雜訊 B1／B2 的比對上一段沒存檔，重跑一次（run `20261107-001138-icpe-rux-recheck-noise`）。上一段有 7 個 run 目錄缺 `exit_code`（比對 run 都有），之後的 run 一律用 `code/icpe/rux_x1fast/stage1/rux_run.sh`（寫 exit_code、每 5 s 記 MemAvailable 與 load、< 3 GiB 自動停）。
- pull：paper repo 的遠端各分支沒有新 commit；kvsweep fast-forward 到 `6d3c5f9`。
- Bailian coder 在這台轉成 npz，指紋和 acvlab 相同（`192949471c…`）。
- 慢層上限（項目 1）在 Mooncake、Bailian、SemiAnalysis 各一個代表 job 上量 Python（有／沒有上限、7 種慢層規則）與 Rust v5（沒有上限），加 cProfile 與操作次數；用 `jobs_for` 數 job；量第一次超過上限的位置 r\*。
- 其他項目（APC、排隊、W3、W5）量了最接近的 v5 job 形狀。

**主要發現**
- Python ÷ Rust v5（沒有上限、逐請求指紋全部相同）：這台 17–50 倍（Mooncake、Bailian 約 20 倍，SemiAnalysis 約 44–50 倍）。
- 慢層上限讓 Python 變慢的原因有兩種：
  - SemiAnalysis：7 種規則都約 1.9×，因為開上限時關掉 `_fix` 快速路徑（`_fix` 呼叫 ×3.52）。
  - Mooncake／Bailian：LRU2、PENS、BEL 每次丟 session 都掃全部慢層 session → 7.8–125×（Bailian BEL 一個 job 76 s，1.29 億次 `next_req`）。
- Rust 照抄版〔推算〕：比 Python 快 20 倍以上，但 Mooncake／Bailian 的 LRU2、PENS、BEL 仍會是 v5 一般 job 的 30–100 倍 → 之後值得做 lazy heap 等加速（逐位元相同）。
- 第一次超過上限在 Mooncake／Bailian 約第 45–57% 個請求、SemiAnalysis 約 90%：在那之前 7 種慢層規則的狀態相同，可共用。
- ARC 與 S3FIFO 在 3 份紀錄的代表 job 輸出逐位元相同（SemiAnalysis 上 BEL 和 LRU 也相同）；只看了 1 個 job，報告裡有說明。
- Rust 版最大的風險：`clean_q` 的順序取決於每一次 `_fix` 呼叫，v5 的三個快速路徑（rs:1198、1302、1366）開上限時必須關掉。

**卡住什麼**：無；照指示停下來等決定。

**需要使用者決定什麼**（細節見 ANALYSIS.md §9）
- D0：項目 2–5 的 Python 參考實作由誰寫。
- D1-1：「被丟的對話回來時全部重算」照 Python（別人還持有的共用前綴照常還原）是否就是 PREREG 的意思。
- D1-2：「先丟乾淨副本」「×7 配對」要不要請 h200 寫進 PREREG。
- D1-3：第 2 段只做照抄版（建議），還是接著做低風險的加速。
- D1-4：慢層上限要跑的格子（不卡第 2 段）。
- D1-5：第 2 段驗證也涵蓋 wt 與預取進 GPU（我的預設：要）。
- D1-6：TraceLab 要不要在這台量（記憶體 22 GiB）。
- 項目 2–5 的語意選項。

**需要別的 session 做什麼**：無（第 2 段做完才通知 icpe-sim）。

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
- **更正（00:10 發現）**：這台**有** NVIDIA GPU。先前只看 PATH 上的 `nvidia-smi`（不存在）就判定沒有 GPU，是錯的。
  - Windows 主機：`NVIDIA GeForce RTX 3080`（DriverVersion 32.0.16.2114）＋ `AMD Radeon(TM) Graphics`（內顯）。
  - WSL 內 GPU 直通**可用**：`/usr/lib/wsl/lib/nvidia-smi` 執行成功 → `RTX 3080`、`10240 MiB`、`compute_cap 8.6`、driver `620.70.03`、CUDA UMD `13.5`、`343 W`。
  - 原本 PATH 上沒有 `nvidia-smi`；已加 `/usr/local/bin/nvidia-smi` 連結與 `PATH` 的 `/usr/lib/wsl/lib`。
  - **尚未用 PyTorch／CUDA 實測算例 → CUDA 可用性 NOT_MEASURED**（目前只到驅動層）。
  - 限制：10 GiB 顯存（acvlab 的 3090 是 24 GiB），且開機後 Windows 桌面已用掉約 1.8 GiB。sm_86（Ampere）與 3090 同一世代。
  - 對本次對拍的影響：**無**（G0／xc_sa／雜訊全部是 CPU 路徑）。
