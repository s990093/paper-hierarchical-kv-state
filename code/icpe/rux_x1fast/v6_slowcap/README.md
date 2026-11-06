# x1fast v6：v5＋慢層容量上限（icpe-rux，RUX3DPRO）

> 2026-11-07。這是 icpe-rux 自己目錄裡的版本，**沒有併進 `code/p3x/x1fast`**（那是 icpe-sim 的檔）。
> 語意照 `code/p3x/x1_sim.py`（blob `8d0563ed`）的慢層容量上限，一字不改（PREREG §2 第 5 項；使用者 11/4 的兩個決定見 STATUS_SIM 23:49）。

## 檔案

| 檔 | 是什麼 |
|:--|:--|
| `src/main.rs` | v6 引擎。由 `patch_v6.py` 從 v5（`code/p3x/x1fast/src/main.rs`，sha256 `3642370a…`）產生，可重現 |
| `patch_v6.py` | v5 → v6 的全部改動（精確字串取代，每處都要剛好出現一次） |
| `Cargo.toml`、`Cargo.lock` | 和 v5 相同（沒有相依套件；`--profile fast`，不加 target-cpu） |
| `x1_rust_v6.py` | `code/p3x/x1_rust.py` 的複本：SRC＝本目錄；`unsupported()` 放行慢層上限；`write_job` 多一欄 `sc=<frac>,<規則>,<上限塊數>`；`read_outputs` 讀 `eng_sc.i64` |
| `sweep_v6.py` | 用 v6 跑 `x1 sweep --engine rust`：把 `x1_rust_v6` 掛成 `x1_rust`，`x1_sim._rust_sweep` 只改一處（`write_job` 多傳 `sc=j[7]`），其他照 `p3_fvar.py` |
| `smoke_v6.py` | 冒煙測試：同一個 job，Python Engine 對 Rust v6（批次模式），比逐請求陣列、I/O 線、引擎計數的 sha256 |
| `cmp_v6.py` | 正式對拍：CSV 逐格（鍵含 `slow_cap_frac`、`slow_policy`、雜訊欄）、npz 逐陣列（`allow_pickle=False`，NaN 視為相同） |
| `validate.sh` | 一份紀錄的正式對拍：主子集、wt、預取進 GPU、雜訊 B1／B2，各跑 Rust 與 Python 再比 |
| `env_sim.sh` | sweep 的環境（P3_CONST 等，照 STATUS_RUX 對拍時的設定） |

## 改了什麼（對照 x1_sim.py）

| x1_sim.py | v6 |
|:--|:--|
| `SlowOrder`（py:521–695）：7 種規則的 add／remove／select | `SlowCap` 結構＋`sc_add`／`SlowCap::remove`／`sc_select`。`mem`＝`OList`（OrderedDict 的插入順序）；LRU2／PENS／BEL 照 `mem` 的順序全掃，只在嚴格更小／更大時換（＝Python `min`／`max` 取第一個）；BEL 用 `next_req(x, 1)`，雜訊 B2 時每個成員抽一次亂數，順序和 Python 相同 |
| `_sc_init`（py:947–965） | `new_eng` 建 `SlowCap`；上限塊數由 Python 算（`int(floor(frac × n_blocks_full))`）傳進來；慢層 GDSF 的成本表用 views[0]（`max(f_i − (i4 ÷ slow_r ＋ d), 0)`，max 照 Python：0 > x 才取 0） |
| `_fix_sc`（py:973–981） | `fix` 外包一層（v5 的本體改名 `fix_inner`）：記 n_slow、變乾淨的塊放進 `clean_q` |
| `_demote_sc`（py:983–986） | `demote` 尾端 `sc_add` |
| `_enforce_sc`／`_slow_enforce`（py:988–1007） | `enforce` 尾端 `slow_enforce`；丟 session 用 `drop_`，回傳的寫入量不算（同 Python） |
| `make_room` 尾端（py:1580–1581） | `make_room` 尾端 `slow_enforce`（**第一版漏了這處**，見下面「驗證」） |
| `drop`（py:1471–1472）、`step`（py:1638–1639） | `drop_` 開頭、`step` 在 `cur` 設好之後：離開慢層（被丟／回來＝命中） |
| `_fix_fast=False`（py:961） | 開上限時三個回合結束的快速路徑都關掉（`step`、`turn_end_wb`、`turn_end_wt`）：不跳過互相抵銷的前綴、每塊都 `fix`；wb 的順序照 Python 的慢路徑（全部遞減 → `g_release(old)` → 全部 `fix`） |
| `audit`（py:1997–2000） | 驗證模式 `X1FAST_SC_CHECK=k`：每 k 個請求稽核一次，另外斷言 LRU2／PENS／BEL 選擇時沒有同分 |
| `_job_rows`（py:2207–2211） | `write_out` 另寫 `eng_sc.i64`（上限、清副本次數、丟 session 次數）；`eng.i64` 照舊 7 個值 |

沒有上限（jobs.txt 沒有 `sc=`）時，每條路徑和 v5 相同。

## 驗證

（結果寫在 `docs/icpe_brief_20261030/next/rux/V6_SLOWCAP.md`；run 目錄在 `/root/icpe-next/runs/`。加速版見 `../v7_fast/`。`x1_rust_v6.py` 後來多了 `RUX_X1FAST_SRC`（指定編哪一版；沒設＝本目錄，行為不變）。）

- 功能關閉：G1（4,416 列 × 29 欄，對 H200 參考）、xc_sa（32 列 × 40 欄），用最終版執行檔重跑。
- 功能打開：Mooncake conversation R1、Bailian coder、SemiAnalysis N16 s1，各 4 組子集（`validate.sh`）。
- 第一版漏了 `make_room` 尾端的 `_slow_enforce`：Mooncake 主子集 420 組裡有 22 組（全是 LRU+RPF × GPU BEL／TTLA:10）不同，run `20261107-004914-icpe-rux-v6-val-mc`。補上之後重跑全部相同（run `20261107-010508-icpe-rux-v6-val-mc2`）。

## 怎麼重現

```bash
source /root/icpe-next/tools/env-rux.sh
python -I code/icpe/rux_x1fast/v6_slowcap/patch_v6.py code/p3x/x1fast/src/main.rs code/icpe/rux_x1fast/v6_slowcap/src/main.rs
/root/icpe-next/tools/rux_run.sh v6-val-mc 'bash $REPO/code/icpe/rux_x1fast/v6_slowcap/validate.sh conversation R1 $RUN/v'
```
