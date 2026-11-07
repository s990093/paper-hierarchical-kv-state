# W7 合成參數地圖：設計（icpe-rux，RUX3DPRO；PREREG 偏離 12 的依據）

**問題／要證明什麼**
- NEXT_PLAN W7：在哪些工作負載條件下，預測（④ 對 ② 的上限）≥ 15%。
- 只回答「什麼條件下預測有空間」，不取代真實紀錄。
- 合成紀錄的結果和真實紀錄的結果分開報（EXT:88、PR:56）。
- 只給 CCGrid 版，ICPE 論文不用（使用者 11/6「ICPE 之後」）。

**使用者的決定**（AskUserQuestion，11/7 約 09:35；選項原文在 PREREG 偏離 12）
- 維度：9 維全部（kvsweep `docs/stage1/A_shapes.md` §8 6a-2 的 T2–T10）。
- 抽樣：LHS n＝10d × 3 seed，也就是 90 點 × 3＝270 份。
- 判定：逐點判，最多加密一輪。
- 其餘細節交給 icpe-rux，push 前給使用者看。
- 之後使用者補充：「需要用 rust 寫」。所以生成器是 Rust；Python 只做包檔（既有的 `x1_nb.write_nb`）、獨立檢查與彙總。
- 看過偏離 12 草稿後（11/7 10:05 之後）：
  - 「push 並通知」；
  - GPU 選「GPU 固定 111.0625 GiB」（另一個選項是隨 WS_ref 縮放）。

---

## 1. 程式（`code/icpe/rux_x1fast/w7/`）

| 檔 | 做什麼 |
|:--|:--|
| `src/main.rs`（`w7gen`，Rust，無外部 crate，`cargo --offline --locked`） | `design`：LHS 設計表；`gen`：一份合成紀錄 → 小端原始陣列＋meta（含 WS_ref） |
| `w7_pack.py` | 獨立檢查（見 §4），再用 `code/p3x/x1_nb.write_nb`（icpe-sim 的介面，不改）包成 nb 的 npz |
| `w7_ws.py` | 活工作集 WS_ref 的第二份實作（Python）；也用來量真實紀錄 |
| `w7_stats.py` | 實現值 vs 設定值（只報告） |
| `w7_map.py` | 彙總成地圖（§3） |
| `run_w7.sh` | 正式執行：產生＋檢查 → 270 個 sweep → 地圖 → collect |

引擎不改：v8 `main.rs` 48e4efce（和偏離 10、11 同一版），經 `sweep_v8.py` 的 `nb:` 入口。

## 2. 合成模型

### 2a. 9 個維度（每個點的 u∈[0,1)^9 由 LHS 給）

| # | 維度 | 範圍與映射 | 每個 session／每輪怎麼用 | 出處 |
|:--|:--|:--|:--|:--|
| T2 | `turns_mean` | 2–200，對數均勻 | 輪數 K ~ Geometric(1/turns_mean)，K ≥ 1 | EXT:78 |
| T3 | `gap_family` | exp／lognormal／lomax，各占 1/3 | 間隔＝分佈族的單位平均樣本 × 該類的平均。lognormal 的 σ＝1.5684（FI week01 工具間隔的擬合）；lomax 的 α＝1.5（平均有限、變異無限） | EXT:77、DD:115 |
| T4 | `gap_mean_s` | 1–600 s，對數均勻 | 工具與人混合後的平均間隔；人的平均＝工具的 51.33 倍（FI week01 中位數比 174,983.5 ms ÷ 3,409 ms） | EXT:77、DR:239 |
| T5 | `tool_share` | 0–1，均勻 | 回來的請求由工具觸發的機率，其餘是人觸發；新對話的第一個請求是人 | DD:114 |
| T6 | `init_ctx_tokens` | 512–65,536，對數均勻 | 新對話自己的起始內容（不含 system prompt），每個 session 取 log-normal，中位數＝這個值、σ＝0.5 | EXT:79 |
| T7 | `inc_tokens` | 16–8,192，對數均勻 | 每輪增量（模型輸出＋工具結果或使用者訊息），log-normal，中位數＝這個值、σ＝1.0，≥ 1 | EXT:79 |
| T8 | `sys_tokens` | 0–32,767，均勻 | 所有新對話共用的 system prompt，它的完整塊共用同一組鍵 | EXT:80 |
| T9 | `fork_frac` | 0–0.5，均勻 | 開新 session 時，以這個機率從當下活著的 session 均勻挑一個 fork（kind 2）：共用那個 session 最新請求的完整塊，再加一次增量 | EXT:80 |
| T10 | `N` | 8–128，對數均勻後取整 | 固定並行：N 個槽，一個 session 結束就在那個時刻開下一個（同 SemiAnalysis 的到達合成） | EXT:81、PR:34 |

### 2b. 固定的部分

**規模與上限**
- 規模（T1）：每份開到總請求數 ≥ 20,000 就不再開新的 session；已開的全部保留。20,000 落在 1–5 萬的範圍，和 Bailian 同級。
- 上下文上限：1,048,576 token（2,048 塊）。下一輪會超過時，這個 session 就結束。
  - 根據：SemiAnalysis N16 s1 實測最大 990,208 token，53% 的請求超過 128K。主結果沒有截在 128K，128K 只當敏感度。
  - 原本草稿用 128K，會把長上下文點的輪數截到原本的 1/10 以下，所以在讀任何模擬結果之前就改了。

**時刻**
- t_k＝t_{k−1}＋svc_{k−1}＋gap_k。
- svc＝(A＋B·out＋C·input)/1000 秒，A、B、C 用 FreeInference 的 `service_constants.json`。svc 只用來排時刻，TTFT 由引擎自己算。
- 輸出 token 取 log-normal：中位數 137，σ＝(ln 813 − ln 43)/(2·1.2816)＝1.147（FI 的 p10／p50／p90）。

**鍵**
- 512-token 一塊，只存完整的塊；nall＝ceil(input/512)。
- system prompt 的塊用 0..sys_full−1，所有新對話共用。
- 下一輪＝上一輪的完整塊，再接新鍵。這是鏈式前綴，和 `x1_nb.check` 的規則相同。

**其他**
- user_trigger：新對話的第一個請求、人觸發的請求是 true；fork 的第一個請求、工具觸發的請求是 false。
- 亂數：xoshiro256**，用 splitmix64 播種。
  - LHS 的種子是 20261107。每份紀錄的種子是 (鹽, 20261107, 點, seed)，seed＝1、2、3。
  - 每個 session、每一輪抽亂數的順序是固定的。
  - 同一個點、同一個 seed 重產，輸出逐位元相同（已測）。
- `design.csv` 的 sha256＝`ab39ad31c11972ecb6e236fb1399f64c1051bd4b81482f7e2752c20584720fe8`。`run_w7.sh` 會重產並比對，不一樣就停。

### 2c. 引擎網格
- 版本 C 全套：CPU 17 × GPU 18 規則 × ssd／b32／nfs × 閒／忙 × Cake。GPU 111.0625 GiB、wb、預取到 CPU。每份 14,184 個 job。
- **CPU 容量**：9 個水準，WS_ref × 2^k，k＝−6..2，取整到 1/16 GiB（一塊），最少一塊。
- **WS_ref**（活工作集）的定義：
  - session 在它第一個到最後一個請求之間算「活的」；
  - 時刻 τ 的工作集＝所有活 session 在 τ 之前（含）最新請求的完整塊的聯集，共用前綴只算一次；
  - WS_ref＝這個工作集在整份紀錄上的時間加權平均 × 64 MiB。
- **真實紀錄的對照**（run `20261107-092554-icpe-rux-w7-ws-real-calib`，`w7_ws.py`）：

| 紀錄 | WS_ref (GiB) | 主結果 8–1,024 GiB 換成 WS_ref 的比例 |
|:--|--:|:--|
| Bailian coder | 360.6 | 0.022–2.84 |
| SemiAnalysis N16 s1 | 438.7 | 0.018–2.33 |
| SemiAnalysis N72 s1 | 870.4 | 0.009–1.18 |

- 所以 2^−6（0.016）到 2^2（4）涵蓋真實紀錄在主結果裡的範圍。
- **設計診斷**（只看輸入，沒有任何模擬結果）：
  - 270 份的 WS_ref：中位數 82.9 GiB，p10 12.0，p90 875，最大 2,257 GiB。
  - 57.8% 小於 GPU（111 GiB），18.1% 超過 4 倍 GPU。
  - GPU 固定時，WS_ref 小於 GPU 的點大多用不到 CPU 層，上限應該接近 0。這本身就是地圖的一部分（壓力不夠時預測沒有空間）。
  - WS_ref／GPU 當衍生變數一起報。
  - 真實紀錄的 WS_ref／GPU 是 3.2–7.8。

## 3. 判定（`w7_map.py`）
- **單位**：(點, 水準 k, 慢層, 閒／忙, 元件)，只用 cake。
- **上限**：3 個 seed 的中位數，另報最小、最大。中位數 ≥ 0.15 就算在區域內。3 個 seed 不齊的單位不判。
- **元件**：用 `analyze_v8`（APC 算不預測，BELP 只進 O-GPU 的 ④）。
  - 判定用：O-ORD、O-DROP、O-NOW、O-PF、O-GPU（偏離 7）。
  - O-ALL(report) 只報告。
- **報告**：
  - 每個元件在區域內的比例、各水準的比例、seed 之間範圍的中位數。
  - 9 維＋WS_ref／GPU 和點層級上限的 Spearman 等級相關，只報告，不檢定。
  - 寫入飽和照算，另報格數（偏離 7 第 3 點）。
- **加密一輪**：某個判定元件的比例落在 [5%, 95%] 才可以做，而且要先再登記範圍與點數。最多一輪。
- **寫法限制**：結論只能寫「在這個合成模型裡…」，不和主結果混成一個數字。

## 4. 驗證（沒有參考實作）

| 檢查 | 做法 | 結果（11/7 開發時） |
|:--|:--|:--|
| 格式、鏈式前綴 | `x1_nb.check`（`write_nb` 寫之前跑） | 9 份全部通過 |
| 時刻遞推 | t[i]＝(t[parent]＋svc[parent])＋gap[i]，逐位元 | 通過 |
| 固定並行 | 任何時刻占用的槽 ≤ N、最大＝N | 通過 |
| 鍵的來源 | 系統 prompt、上一輪延伸、fork 來源是當下活著的 session、新鍵不重複 | 通過 |
| WS_ref 雙實作 | Rust 和 Python 的 WS_ref、WS_max 逐位元相同；另有 4 個請求的手算例子 | 通過 |
| 決定性 | 同點同 seed 重產逐位元相同、不同 seed 不同；design.csv 重產相同 | 通過 |
| LHS | 每一維 90 個分層各恰好一個點；分佈族各 30 點 | 通過 |
| 端到端 | `run_w7.sh` 煙霧測試（點 0 × 3 個 seed）＋中斷後接續 | 通過（修正 1 個 bug，見 §5） |
| 實現值 | 工具比例、各類平均間隔、單位中位數 vs 理論（exp 0.693、lognormal 0.292、lomax 0.294）、fork 比例（`w7_stats.py`，7 份） | 都在抽樣誤差內，lomax 的平均因為重尾差得較多 |

## 5. 試跑（揭露）
- 登記前跑了 4 份合成紀錄的 sweep，只為了量時間、確認引擎能跑。上限表有算出來（在 run 目錄），**icpe-rux 沒有讀**。
  - run `20261107-093048-icpe-rux-w7-pilot-timing`：w7_p000_s1、w7_p051_s1。這兩份用的是 128K 上限的舊版；p051 跑到一半就停了。
  - run `20261107-093834-icpe-rux-w7-pilot-timing2`：w7_p051_s2（最重的一份，2,227 萬塊）、w7_p070_s1（中位數大小）。
- `run_w7.sh` 的端到端煙霧測試：點 0 × 3 個 seed，收集到暫存目錄。
  - run `20261107-100727-icpe-rux-w7-smoke-runner`：抓到 `w7_map.py` 的欄名重複（analyze_v8 的表本身已有 run_id），腳本在那一步停下。
  - 修正後用接續模式跑 run `20261107-101311-icpe-rux-w7-smoke-runner-resume`：3 份沿用、地圖 324 個單位全齊、MANIFEST 對到引擎 48e4efce。
  - 地圖數字沒讀。
- 範圍、1M 上限、GPU 固定、CPU 水準都在讀任何上限數字之前定下。
- 例外照實記：
  - 11/7 09:4x 看 w7_p051_s2 的進度時，用 `tail` 讀到 sweep log 的 5 行進度。每行有一個 job 的 `mean_ret[0]`，都是 445.6875 GiB、O-ALL 的組合。
  - 上限表、②、④ 都沒有讀。之後只讀進度計數。

## 6. 規模與時間
- 270 個 sweep × 14,184 個 job＝383 萬個 job。
- 時間見偏離 12。用試跑估計，RUX3DPRO 用 16 個 worker。
