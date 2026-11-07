# v9_qev：二階（事件驅動）排隊（PREREG 偏離 19）

設計與驗證見 `docs/icpe_brief_20261030/next/rux/QUEUE2_DESIGN.md`。

| 檔 | 用途 |
|:--|:--|
| `make_v9.py` | 從 `../v8_dev14/src/main.rs` 產生 `src/main.rs`（v8_dev14＋`qev=1`＋`qpost=2`）。v8、v8_dev14 一行不改。改 v9 一律改這支再重產，不直接改 `src/main.rs` |
| `src/main.rs`、`Cargo.toml`、`Cargo.lock` | v9 引擎（Cargo 檔和 v8 相同） |
| `sweep_v9.py` | `x1 sweep --engine rust` 用 v9：`RUX_QEV=1`（二階）、`RUX_QPOST2=1`（一階、到達不早於上一個完成） |
| `smoke_v9.py` | 驗證 T1–T6（退化＝v8、Python 精確重算、稽核、決定性、`qpost=2`） |
| `explore_queue2.py` | 一個變體的二階分析：結構檢查、上限表（`analyze_v8`）、負載分級、和版本 C／一階比 |
| `run_queue2_sweep.sh` | 一個變體的模擬（開 pfx=1、pgr=1）：A 段一階 `qpost=2`、B 段二階 `qev` |
| `run_queue2_analyze.sh` | 一個變體的分析與收集：和修正版版本 C（icpe-data 第二段）比 → `results/icpe_next/queue1m/`、`results/icpe_next/queue2/` |
| `run_queue2_chain.sh` | 12 個變體的模擬，一個變體一個 run；已完成的跳過、沒跑完的用新 run 整個重跑，檢查 code_sha（11/7 19:57 WSL 重啟後從 scratchpad 移進來） |
| `run_queue2_chain_analyze.sh` | 12 個（或指定的）變體的分析：從 `origin/icpe-next-trx` 的 `results/icpe_next/apc_d14/` 取修正版版本 C，逐個跑 `run_queue2_analyze.sh` |

重產與編譯：
```
python -I code/icpe/rux_x1fast/v9_qev/make_v9.py
# sweep 第一次用到時會自己編（x1_rust_v6.binary()，build-<src sha 前 16 碼>）
```
