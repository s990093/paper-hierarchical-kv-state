# xref_dev14：偏離 14＋SIM_CHANGES 23 三個旗標都開的跨機器參考（icpe-sim，acvlab）

- **用途**：別台機器（TRX40、icpe-data 合併後的 G0）在**開旗標**跑正式 block 之前，先用同一份程式重跑這 6 個 sweep，逐列和這裡比。不同的列必須是 0。
- **程式**：`icpe-next-sim` commit `63b8e32`（之後到本目錄的 commit 只改文件與 `code/icpe/next_sim_*`，引擎檔不變）。
  - `x1_sim.py` sha256 `8b53588b85c3…`、`x1_rust.py` `0d89c4ba33c5…`、`x1_nb.py` `8251e2587464…`。
  - Rust src_sha256 `0bfacb59700380ba7850fcd8a7f76d94f3bdcf3eebba87fb40f408afab60054a`。
  - code_sha：`3180504e8d57+rs0bfacb59`（Rust）、`3180504e8d57`（Python）。
- **run**：`20261107-155727-icpe-sim-xref-dev14`（acvlab；Python 5 個 worker、Rust 1 個；nice 10）。
  - Python 與 Rust 6 個 sweep 都逐列相同（`xref_check.csv`：1,184 列，不同的列 0；Rust `py_fallback_jobs=0`）。
  - 這裡的 CSV＝Rust 的輸出，在 `ts` 後面多一欄 `host`。
- **環境**：`env.sh`（原樣；`cd` 到凍結的 worktree，`P3_CONST=h200 P3_TIER=h200cpu`，`--variant V1 --calib results/icpe_h200/p01/calib`）。
- **6 個 sweep**（`$SW` 等變數見 `env.sh`；`ON="--pf-expire --noise-overdue resample --pf-gpu-recheck"`）：

| 檔 | 指令 | job | 列 |
|:--|:--|--:|--:|
| `xref_sa_qb.csv` | `$SW … x1 sweep $TSA $XQB $ON` | 178 | 208 |
| `xref_sa_qt.csv` | `$SW … x1 sweep $TSA $XQT $ON` | 108 | 128 |
| `xref_mc_qb.csv` | `$SW … x1 sweep $TMC $XQB $ON` | 178 | 208 |
| `xref_mc_qt.csv` | `$SW … x1 sweep $TMC $XQT $ON` | 108 | 128 |
| `xref_sa_n.csv` | `$SW … x1 sweep $TSA $XN $ON` | 232 | 256 |
| `xref_mc_n.csv` | `$SW … x1 sweep $TMC $XN $ON` | 232 | 256 |

  - `TSA="--traces nb:semianalysis_N16_s1 --rules nb"`、`TMC="--traces conversation --rules R1"`。
  - `XQB`：8 個規則（LRU、LRU+RPF、O-BEL、O-BELS、O-DROP、O-NOW、O-PF、O-ALL）× CPU 8／64 × b32 × GPU 0／111.0625 wb × LRU／BEL／TTLA:10／TTLB:30 × 預取到 CPU／GPU。
  - `XQT`：同上，但 GPU 111.0625 wt、只預取到 CPU。
  - `XN`：6 個規則 × CPU 8／64 × GPU 111.0625 wb × BEL／TTLB:30 × 預取到 CPU／GPU × P／M × λ 0.3 × B1／B2 × seed 0。
- **不在網格裡的組合**（已知問題，正式 block 都不用）：
  - wt × 預取到 GPU：旗標關也會 assert。使用者 11/7 選 A，不修。
  - 預取到 GPU 卻沒開 `--pf-gpu-recheck`：有雜訊時旗標關也會 assert。規則：預取到 GPU 一律開。
  - 見 STATUS_SIM 11/7 15:56、15:59、16:35。
- **比法**：`python -I code/icpe/next_sim_dev14_check.py <這裡的 csv> <你的 x1_sweep.csv> <out.csv> <run_id> <tag> all`。
  - 鍵：trace／rule／cpu／policy／slow／gpu／restore／gpu_GiB／gpu_write／gpu_policy／pf_target／雜訊 4 欄。
  - 比兩邊共有、去掉 run_id／ts／host／code_sha／elapsed_s／三個旗標欄的每一欄，逐字串相同。
  - 印 `DEV14 <tag> PASS` 才算過。G1 的比法（`next_sim_g1.py`）也可以。
- 檔案 sha256（加 host 欄之後）：見 commit 訊息與 STATUS_SIM 同一節。
