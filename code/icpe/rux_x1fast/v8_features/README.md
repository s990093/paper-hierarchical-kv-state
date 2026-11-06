# x1fast v8：v7＋第 2–5 項的新功能（icpe-rux，RUX3DPRO；只有 Rust）

> 2026-11-07。語意：`docs/icpe_brief_20261030/next/rux/SEMANTICS_2to5.md`（icpe-rux 依使用者授權決定；**正式跑之前要登記進 PREREG**）。
> 結果：`docs/icpe_brief_20261030/next/rux/V8_FEATURES.md`。沒有併進 `code/p3x/x1fast`。
> **沒有 Python 參考**（使用者：「python 不寫完整移植 rust」）。驗證改用退化檢查、精確重算、不變量稽核，見下面。

## 功能與 jobs.txt 的寫法

| 功能 | jobs.txt | 另外的檔 | 限制 |
|:--|:--|:--|:--|
| APC（項目 2） | GPU 規則欄寫 `apc`（session LRU、尾端一塊一塊逐出）或 `belp`（同樣逐塊，選 session 照 Belady） | — | 要開 GPU 層；不支援預取進 GPU |
| 排隊（項目 3） | 多一欄 `q=1`（open-loop FCFS-1）或 `q=2`（closed-loop、照紀錄順序服務） | — | 一個 job 1 個時間設定。**兩個都不適合固定並行的紀錄**（SEMANTICS §2a）：只當紀錄，不要拿來出結果 |
| W3 線索（項目 4） | GPU 規則欄寫 `cuep`（有工具呼叫 → 釘到下一個請求）或 `cueo`（有工具呼叫 → 回合結束立刻逐出）；多一欄 `cue=1` | `<job 目錄>/cue.u8`：每個請求 1 個 byte | 要開 GPU 層；不支援預取進 GPU |
| W5（項目 5） | 多一欄 `w5=<CPU 每塊位元組>`（INT4＝i4） | `<job 目錄>/w5.f64`：每個時間設定 [dc1, dc2]＝讀 CPU 的 BF16 位置／INT4 預取位置的塊之後加的殘差 | 不支援預取進 GPU |

沒有這些欄位時，每條路徑和 v7 相同（G1、xc_sa、3 份紀錄 × 4 組子集對 Python 逐位元相同，見 V8_FEATURES.md）。

**輸出**
- 另外多寫：`qwait.f64`、`qstart.f64`、`qsvc.f64`（排隊）、`eng_v8.i64`（APC 逐出的塊數）。
- `eng.i64` 照舊 7 個值。
- 排隊時 `v0_ttft` 已經含等待。

## 檔案

| 檔 | 是什麼 |
|:--|:--|
| `patch_v8.py` | v7 → v8 的全部改動（A 排隊、B W3 線索、C W5、D APC），精確取代 |
| `src/main.rs` | 由 `patch_v8.py` 產生 |
| `smoke_v8.py` | 驗證：`--tests queue cue w5 apc`（見下） |
| `sweep_v8.py` | 用 v8 跑 `x1 sweep --engine rust`：多認 GPU 規則 `APC`、`BELP`（Python 的 Engine 遇到這兩個就停，不能退回 Python）；`RUX_W5=1` → 每個 job 都是 W5，CSV 多一欄 `cpu_fmt`（輸出放另一個目錄）。排隊、W3 還沒接進 sweep |
| `explore_w5.py` | 探索：W5 的上限表（BF16、強制 INT4、每個組合挑較快的格式） |
| `analyze_v8.py` | **正式分析用**：`next_sim_analyze.py` 加上「BELP 是 oracle、只進 O-GPU 的 ④」（原檔一行不改，只在執行時換 `analyze()` 的兩處）；APC 照原本的規則＝② 和 ④ 都有 |
| `check_analyze_v8.py` | 檢查 analyze_v8：O-GPU 以外＝原本的分析（拿掉 BELP 列）；O-GPU＝獨立算法 |
| `w5_mechanism.py` | W5 上限上升的格：④／② 的載入次數變化 |
| `cmp_pol.py` | 同一個 x1_sweep.csv 裡兩個 GPU 規則逐格比（APC 一次逐出整個 session 時要＝LRU） |
| `explore_apc.py` | 探索：APC／BELP 加進 GPU 規則後的上限表（重現版本 B → 加 APC → O-GPU 加 BELP） |
| `w3_cue_stats.py` | TraceLab 原始紀錄上，「輸出有工具呼叫」和「下一輪由工具結果觸發」的一致率 |

## 驗證（`smoke_v8.py`；每個 trace 一個 run）

- **退化檢查**：新功能的參數設成「等於沒開」時，v8 和 v7 的原始輸出逐位元相同（每個時間設定的逐請求陣列、I/O 線狀態、引擎計數）。
  - W3：cue 全 0 時 CUEP、CUEO＝GPU LRU；cue 全 1 時 CUEP＝TTLB:∞。
  - 排隊：到達時刻改成 r × 10⁶ 秒時，q=1＝沒有排隊，等待全為 0。
  - W5：`w5=BF`、dc1＝0、dc2＝d 時＝v7（走 W5 的程式碼路徑）。
  - APC／BELP：`X1FAST_APC_WHOLE=1`（一次逐出整個 session）時＝v7 的 GPU LRU／BEL，含 wb 的「先寫進 CPU」路徑；GPU 很大時＝v7。
- **精確重算**（排隊）：開始＝max(到達, 前一個開始＋服務)、TTFT＝服務＋等待，逐請求位元相同。
- **不變量**（W5）：CPU 讀的位元組＝CPU 命中塊數 × i4；寫入是 i4 的整數倍。
- **稽核**（APC，`X1FAST_V8_CHECK=k`）：每 k 個請求從頭重算 ref_gpu、ref_gc（乾淨副本的引用）、gpu_used，並斷言 S_GPU 的 session 整個在 GPU 上、GPU 不超過預算。
- **端到端**（sweep）：`X1FAST_APC_WHOLE=1` 下 sweep 出來的 APC／BELP 列＝LRU／BEL 列逐格相同（`cmp_pol.py`）。
- **兩份實作互比**（BELP）：預設的捷徑和 `X1FAST_BELP_SLOW=1`（每逐出一塊都全掃）逐位元相同；驗證模式下每次選擇也另外全掃比對。
  - 捷徑：逐出的塊沒有別人持有時，其他 session 的鍵都不變。被選的 session 的鍵沒有變小，就仍是最大，不用重掃。只在沒有雜訊時用。
- **抓不到的**：語意本身理解錯。所以語意先寫成文件，建議另一個 session 審。

## 怎麼重現

```bash
source /root/icpe-next/tools/env-rux.sh
python -I code/icpe/rux_x1fast/v8_features/patch_v8.py code/icpe/rux_x1fast/v7_fast/src/main.rs code/icpe/rux_x1fast/v8_features/src/main.rs
/root/icpe-next/tools/rux_run.sh v8-smoke-mc 'source $REPO/code/icpe/rux_x1fast/v6_slowcap/env_sim.sh && cd $REPO && $ROOT/venv/bin/python -I code/icpe/rux_x1fast/v8_features/smoke_v8.py --trace conversation --rule R1 --tests queue cue w5 apc --out $RUN/o'
```
