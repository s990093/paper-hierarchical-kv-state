# x1fast v7：v6＋慢層上限的加速（icpe-rux，RUX3DPRO）

> 2026-11-07。**不改語意**：輸出和 v6、和 Python（`x1_sim.py`）逐位元相同。結果見 `docs/icpe_brief_20261030/next/rux/V7_FAST.md`。
> 沒有併進 `code/p3x/x1fast`（icpe-sim 的檔）。

| 檔 | 是什麼 |
|:--|:--|
| `src/main.rs` | v7 引擎。由 `patch_v7.py` 從 v6（`../v6_slowcap/src/main.rs`，sha256 `2eb5a040…`）產生 |
| `patch_v7.py` | 三個加速：慢層 LRU2 用有序集合（候選 2）、慢層 PENS 的成本快取（候選 3）、開上限也走快速路徑＋補記 `clean_q`（候選 5）。每處精確取代 |
| `Cargo.toml`、`Cargo.lock` | 和 v5／v6 相同 |
| `validate.sh` | v6 的 `validate.sh` 多兩個環境變數：`WORKERS`（預設 16）、`RUX_X1FAST_SRC`（預設本目錄）。配 `REUSE_PY=<v6 的輸出>` 只重跑 Rust |
| `speed_v7.py` | 單執行緒：同一個 job，Python、v6、v7 輪流跑，比時間與逐請求指紋 |

Python 端沿用 `../v6_slowcap/` 的 `x1_rust_v6.py`（`RUX_X1FAST_SRC` 指定要編哪一版）、`sweep_v6.py`、`cmp_v6.py`、`env_sim.sh`。

驗證模式：`X1FAST_SC_CHECK=k`。開了之後：
- 每 k 個請求稽核一次；
- 每次選擇都另外全掃比對；
- PENS 的成本快取每次重算並比位元；
- 快速路徑每次跳過 `fix` 時，斷言那一步確實什麼都不改。

## 怎麼重現

```bash
source /root/icpe-next/tools/env-rux.sh
python -I code/icpe/rux_x1fast/v7_fast/patch_v7.py code/icpe/rux_x1fast/v6_slowcap/src/main.rs code/icpe/rux_x1fast/v7_fast/src/main.rs
/root/icpe-next/tools/rux_run.sh v7-chk-mc 'X1FAST_SC_CHECK=500 REUSE_PY=<v6 val 的 v 目錄> bash $REPO/code/icpe/rux_x1fast/v7_fast/validate.sh conversation R1 $RUN/v'
```
