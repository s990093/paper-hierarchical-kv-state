# x1fast v6：慢層容量上限的 Rust 版（icpe-rux 第 2 段，RUX3DPRO）

> 2026-11-07。程式在 `code/icpe/rux_x1fast/v6_slowcap/`（說明見那裡的 README）。**沒有併進 `code/p3x/x1fast`**（那是 icpe-sim 的檔；要不要併、由誰審，見 §6）。
> 數字全部來自這台的 run 目錄（`/root/icpe-next/runs/`），逐列在 `V6_rows.csv`（每列有 run_id、ts、host）。〔推算〕＝由實測換算，〔推論〕＝由程式推出、沒有直接量。

## 0. 問題／要證明什麼

- **問題**：慢層容量上限（PREREG §2 第 5 項；使用者 11/4 的兩個決定）只有 Python 實作。現有 Rust（v5）遇到就整個 job 退回 Python，Python 比 Rust 慢 17–125 倍（第 1 段 ANALYSIS §3b），正式格子跑不完。
- **要證明**：
  1. v6 的輸出和 Python（`x1_sim.py`）**逐位元相同**：功能打開時 3 份紀錄 × 4 組子集，CSV 逐格、逐請求 npz 逐陣列；功能關閉時和 v5／H200 參考相同。
  2. 快多少：單執行緒與多執行緒，記負載。

## 1. 結論

- **逐位元相同**：功能打開 3 份紀錄 × 4 組子集，CSV 177,072 格、npz 11,424 個陣列，**0 個不同**；功能關閉 G1 128,064 格、xc_sa 1,280 格，0 個不同。
- **第一版有一個錯，對拍抓到、已改**：`make_room` 尾端少了一次 `_slow_enforce`（py:1580–1581）。只在「預取擠 CPU」的 job（LRU+RPF × GPU BEL／TTLA:10）出現。改了之後全部相同（§3c）。
- **速度**：主子集（1,104 個 job）12 worker 的 sweep 牆鐘，Python ÷ Rust v6＝Mooncake 26.3×、Bailian 51.1×、SemiAnalysis 44.2×；單執行緒同一個 job 20–66×（§4）。

## 2. 改了什麼

見 `code/icpe/rux_x1fast/v6_slowcap/README.md` 的對照表（x1_sim.py 每個掛鉤 → v6 的對應處）。要點：
- `SlowCap`＝`SlowOrder`＋`_sc_*`；7 種慢層規則照 Python 的運算順序；LRU2／PENS／BEL 照 `mem` 的插入順序全掃、只在嚴格更小／更大時換（＝Python `min`／`max` 取第一個）。
- 開上限時照 Python（`_fix_fast=False`）關掉三個回合結束的快速路徑：每塊都 `fix`，`clean_q` 的內容與順序才和 Python 相同。
- 沒有上限時每條路徑和 v5 相同。
- 驗證模式 `X1FAST_SC_CHECK=k`：每 k 個請求做 py:1997–2000 的稽核，並斷言 LRU2／PENS／BEL 選擇時沒有同分。

## 3. 驗證

### 3a. 功能關閉（新程式碼不影響舊路徑）

| 對拍 | 參考 | 格數 | 不同 | run |
|:--|:--|--:|--:|:--|
| G1（4,416 列 × 29 欄） | H200 的 `results/icpe_h200/t1/p01/sim_V1/x1/x1_sweep.csv.gz` | 128,064 | 0 | `20261107-010742-icpe-rux-v6-off-g1-final` |
| xc_sa（32 列 × 40 欄） | `results/icpe_next/b/xcheck/xc_sa/x1_sweep.csv.xz` | 1,280 | 0 | `20261107-010822-icpe-rux-v6-off-xc_sa-final` |

（最終版執行檔，`code_sha …+rs5dfbfbb4`。）

### 3b. 功能打開（Python 對 Rust v6）

子集（`validate.sh`；每組都是慢層 7 種規則、上限 frac 0.5）：
- **main**：CPU 8、111.0625 GiB × CPU 規則 LRU、LRU2、PENS、GDSF、ARC、S3FIFO、LRU+RPF、LRU+2HA、O-BEL、O-NOW × 慢層 ssd、nfs × GPU wb、規則 LRU／BEL／TTLA:10 → 1,104 個 job、1,680 列。
- **wt**：CPU 111.0625 × LRU、O-BEL × GPU wt LRU → 20 個 job。
- **pfgpu**：CPU 111.0625 × LRU+RPF、O-PF × 預取進 GPU → 56 個 job。
- **noise**：W4 雜訊模型 P／M、λ 0.3、B1／B2、seed 0 × CPU LRU、O-BEL × GPU LRU／BEL × 慢層 BEL、LRU → 28 個 job。

比對：CSV 逐格（鍵＝trace、rule、cpu_GiB、policy、slow、gpu、restore、gpu_GiB、gpu_write、gpu_policy、pf_target、雜訊欄、slow_cap_frac、slow_policy；不比 run_id、ts、elapsed_s、code_sha、host）；逐請求 npz 逐陣列（`allow_pickle=False`，NaN 視為相同）。

| 紀錄 | 子集 | CSV 列 | 比的格數 | 不同 | npz 陣列 | 不同 | run |
|:--|:--|--:|--:|--:|--:|--:|:--|
| Mooncake conversation R1 | main | 1,680 | 48,720 | 0 | 3,360 | 0 | `20261107-010508-icpe-rux-v6-val-mc2` |
| 〃 | wt | 56 | 1,624 | 0 | 112 | 0 | 〃 |
| 〃 | pfgpu | 56 | 1,624 | 0 | 112 | 0 | 〃 |
| 〃 | noise | 112 | 3,248 | 0 | 224 | 0 | 〃 |
| Bailian coder | main | 1,680 | 53,760 | 0 | 3,360 | 0 | `20261107-010827-icpe-rux-v6-val-bl` |
| 〃 | wt | 56 | 1,792 | 0 | 112 | 0 | 〃 |
| 〃 | pfgpu | 56 | 1,792 | 0 | 112 | 0 | 〃 |
| 〃 | noise | 112 | 3,584 | 0 | 224 | 0 | 〃 |
| SemiAnalysis N16 s1 | main | 1,680 | 53,760 | 0 | 3,360 | 0 | `20261107-020916-icpe-rux-v6-val-sa` |
| 〃 | wt | 56 | 1,792 | 0 | 112 | 0 | 〃 |
| 〃 | pfgpu | 56 | 1,792 | 0 | 112 | 0 | 〃 |
| 〃 | noise | 112 | 3,584 | 0 | 224 | 0 | 〃 |
| **合計** | | | **177,072** | **0** | **11,424** | **0** | |

- Mooncake 的 CSV 比 29 欄、nb 紀錄 32 欄（nb 多 3 個「使用者觸發」的欄）。
- Mooncake 的 Python 參考是第一次（`…-v6-val-mc`）跑的，第二次沿用（硬連結；Python 不受 Rust 修改影響）。
- 另外 v6 用 16 worker 再跑一次 3 份紀錄的主子集，也全部相同（`…-v6-w16-mc／bl／sa`）。

### 3c. 第一版的錯

- run `20261107-004914-icpe-rux-v6-val-mc`（Mooncake 主子集）：`cells compared=48720 differing=225`，`CSV_RESULT DIFFERENT`，exit 1（照規則停下）。
- 不同的格全部是 CPU LRU+RPF × GPU BEL／TTLA:10 的組合。例：`(conversation, R1, 8, LRU+RPF, nfs, idle, …, BEL, cpu, 0.5, S3FIFO)` 的 `slow_write_GiB` Python 3589.53125、Rust 3589.667969。
- 原因：Python 的 `make_room`（預取時擠 CPU）尾端有 `if self._sc: self._slow_enforce()`（py:1580–1581），v6 第一版只掛在 `enforce` 尾端。
- 改法：`make_room` 尾端也呼叫 `slow_enforce`（`patch_v6.py`）。
- 改後：預取冒煙測試 84／84 相同（run `20261107-005820-icpe-rux-v6-smoke-rpf`）；Mooncake 4 組子集全部相同（run `20261107-010508-icpe-rux-v6-val-mc2`）；Bailian、SemiAnalysis 用最終版跑。
- 這個錯是「和 Python 對拍」才抓到的：驗證模式的稽核（上限、成員）在漏掉時仍成立，因為下一次 `enforce` 會補做。

## 4. 速度

**多執行緒**（主子集 1,104 個 job、1,680 列；`validate.sh` 的 sweep 牆鐘；Rust 和 Python 不同時跑；量的時候 load 見各 run 的 stdout）

| 紀錄 | Rust v6 12 worker | Rust v6 16 worker | Python 12 worker | Python ÷ Rust（12） |
|:--|--:|--:|--:|--:|
| Mooncake | 17.6 s | 17.0 s | 462.6 s | 26.3× |
| Bailian | 55.8 s | 54.5 s | 2,853.8 s | 51.1× |
| SemiAnalysis | 73.8 s | 71.8 s | 3,263.2 s | 44.2× |

- 12 → 16 worker 幾乎沒變快：牆鐘被最慢的幾個 job（慢層 BEL、PENS）拖住〔推論〕。
- 其他三組子集（wt、pfgpu、noise）Python ÷ Rust＝19–54×（`V6_rows.csv`）。

**單執行緒**（同一個 job：CPU LRU × GPU-LRU wb、CPU 111.0625 GiB、6 個時間設定（GDSF 1 個）；Python＝process_time、Rust＝引擎時間；三份紀錄各一個行程同時跑、交錯重複；Mooncake Python 3 次／Rust 5 次，Bailian、SemiAnalysis Python 1 次／Rust 3 次；中位數；run `20261107-032339-icpe-rux-speed-v67`；load 2.7–13（同時有 v8 的測試在跑））

| 設定 | Mooncake Python | v6 | Python÷v6 | Bailian Python | v6 | Python÷v6 | SemiAnalysis Python | v6 | Python÷v6 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 沒上限（6） | 0.293 | 0.0117 | 25.1× | 0.734 | 0.0242 | 30.4× | 13.45 | 0.300 | 44.9× |
| sc:LRU | 0.390 | 0.0139 | 28.1× | 0.942 | 0.0309 | 30.5× | 25.65 | 0.535 | 47.9× |
| sc:ARC | 0.397 | 0.0141 | 28.1× | 0.865 | 0.0333 | 26.0× | 24.81 | 0.563 | 44.1× |
| sc:S3FIFO | 0.406 | 0.0138 | 29.5× | 0.899 | 0.0315 | 28.5× | 24.70 | 0.561 | 44.0× |
| sc:GDSF（1） | 0.316 | 0.0130 | 24.2× | 0.691 | 0.0267 | 25.9× | 18.91 | 0.407 | 46.4× |
| sc:LRU2 | 2.248 | 0.0466 | 48.3× | 16.40 | 0.440 | 37.3× | 25.12 | 0.546 | 46.0× |
| sc:PENS | 9.950 | 0.218 | 45.7× | 64.11 | 1.469 | 43.6× | 25.67 | 0.586 | 43.8× |
| sc:BEL | 11.88 | 0.449 | 26.4× | 80.01 | 1.221 | 65.5× | 24.45 | 0.554 | 44.2× |

- 每個設定 Python、v6（和 v7）的逐請求指紋都相同。
- 開上限後 v6 比沒上限的 v5 路徑慢：LRU 類 1.2–1.8×；Bailian 的 LRU2／PENS／BEL 18–60×（全掃）。加速見 `V7_FAST.md`。

## 5. 換算到正式量

〔推算〕每種紀錄的完整格子（129,924 個 job）在這台 12 worker：Mooncake 約 33 分、Bailian 約 1.8 小時、SemiAnalysis 約 2.3 小時（用主子集的 Rust 每 job 執行緒秒 0.185／0.600／0.776 × job 數 ÷ 12）。子集的規則組成和正式格子不同，只是量級。v7 再快 1–1.5 倍（`V7_FAST.md`）。

## 6. 還沒做／要決定的

- 併進 `code/p3x/x1fast`：icpe-sim 的檔，要它審查、包裝，並在 STATUS_SIM 寫 G1／xcheck PASS 的 commit（照偏離 6 的流程）。
- 正式要跑哪些格子（ANALYSIS §9 D1-4）未定；不跑正式 block。

## 7. run 目錄

| run | 內容 | 結果 |
|:--|:--|:--|
| `20261107-004613-icpe-rux-v6-smoke-mc` | 冒煙：Python Engine 對 v6，7 種慢層規則 | 7／7 相同 |
| `20261107-004743-…-v6-off-g1`、`…-004831-…-v6-off-xc_sa` | 功能關閉（第一版執行檔） | 0 個不同 |
| `20261107-004914-icpe-rux-v6-val-mc` | Mooncake 主子集（第一版） | **225 格不同 → 停下**（§3c） |
| `20261107-005820-icpe-rux-v6-smoke-rpf` | 改了 make_room 之後，預取的冒煙 | 84／84 相同 |
| `20261107-010508-icpe-rux-v6-val-mc2` | Mooncake 4 組子集（最終版） | 0 個不同 |
| `20261107-010742-…-v6-off-g1-final`、`20261107-010822-…-v6-off-xc_sa-final` | 功能關閉（最終版） | 0 個不同 |
| `20261107-010827-icpe-rux-v6-val-bl` | Bailian 4 組子集 | 0 個不同 |
| `20261107-020916-icpe-rux-v6-val-sa` | SemiAnalysis 4 組子集 | 0 個不同 |
| `20261107-031958／032018／032123-…-v6-w16-*` | 3 份紀錄主子集、16 worker | 0 個不同 |
| `20261107-032339-icpe-rux-speed-v67` | 單執行緒速度（Python、v6、v7） | 指紋全部相同 |

每個 run 目錄有 cmd.sh、context.txt、stdout／stderr、exit_code、mem.log（v6、v7 全部 21 個 run 的 MemAvailable 最低 10.4 GiB（10,913,556 kB），沒有觸發 3 GiB 的停止線）。
