# x1fast v8：第 2–5 項的新功能（只有 Rust；icpe-rux 第 4 段，RUX3DPRO）

> 2026-11-07。程式：`code/icpe/rux_x1fast/v8_features/`（README 有用法）。語意：`SEMANTICS_2to5.md`（icpe-rux 依使用者授權決定，**正式跑之前要登記進 PREREG**）。
> **沒有 Python 參考**（使用者：「python 不寫完整移植 rust」）。所以「對」的意思是：退化檢查、精確重算、不變量稽核、兩份實作互比都過；**語意本身理解錯是這些抓不到的**，建議另一個 session 審 SEMANTICS 與程式的對應。
> 探索的數字（§4）**不是正式結果**：只有一個變體、3 個 CPU 容量，沒有登記。

## 0. 問題／要證明什麼

- **問題**：第 2–5 項（block 粒度 APC、排隊、W3 線索 (a)、W5）現有的 Rust 和 Python 都沒有。使用者要直接寫 Rust，語意交給 icpe-rux 決定。
- **要證明**：
  1. 新功能關閉時，v8 和 v7 逐位元相同，所以和 Python 也相同。
  2. 新功能照語意文件運作：退化檢查、精確重算、稽核都要過。
  3. 這些功能對上限有沒有影響（探索）。

## 1. 結論

- **新功能關閉時和 Python 逐位元相同**：v8 最終版（`main.rs` sha256 `a4ded3ca…`）對 v6 那次的 Python 參考，3 份紀錄 × 4 組子集，CSV 177,072 格、npz 11,424 個陣列，0 個不同；G1、xc_sa 0 個不同。
- **四項新功能的驗證**全部通過：最終版在 3 份紀錄上共 2,088 項，0 個 FAIL（§3）。
- **端到端**：
  - sweep 管線重現版本 B 的 toolagent 上限表：306 列逐值相同。
  - APC 一次逐出整個 session 時＝LRU／BEL。
  - W5 走新路徑但參數等於沒開時＝BF16：254,592 格相同。
  - BELP 的捷徑＝每塊全掃。
- **探索**（Mooncake toolagent，不是正式結果）：
  - **APC 幾乎不影響上限**，最多 0.03 個百分點。BELP 讓 O-GPU 的 ④ 略強。
  - **W5 讓上限有升有降**：O-GPU −6 到 −8 個百分點；O-DROP、O-ORD 上升，nfs 的格最多升 39 個百分點。這違反 NP:53 事先登記的預期（「下降或不變」），要查原因。看了一格，和「oracle 更能用多出來的 CPU 容量」一致。
- **排隊沒做完**：open-loop 和照紀錄順序的 closed-loop 都不適合固定並行的紀錄。正確的工作守恆 closed-loop **要先請使用者決定 Q3-a**（SEMANTICS §2a）。
- **W3**：引擎做好了，**資料還沒有**。nb 檔要多一欄「這一輪的輸出有工具呼叫」；TraceLab 原始紀錄上它和「下一輪由工具結果觸發」一致 96.9%。

## 2. 做了什麼

| 項目 | 狀態 | 用法 |
|:--|:--|:--|
| 2 APC | 完成，接進 sweep（`--gpu-policies APC BELP`） | GPU 規則 APC（session LRU、尾端一塊一塊逐出）、BELP（同樣逐塊、選 session 照 Belady，給 ④ 的 O-GPU 用） |
| 3 排隊 | **只做了一半**：open-loop（`q=1`）與「照紀錄順序的 closed-loop」（`q=2`）都做了、也驗了，但兩個都**不適合**固定並行的紀錄（§3c）。正確的模型（工作守恆的 closed-loop）要先決定 oracle 知道什麼（Q3-a） | jobs.txt `q=1`／`q=2`；沒接進 sweep |
| 4 W3 線索 | 引擎完成（GPU 規則 CUEP、CUEO）；**資料還沒有**：nb 檔要多一欄「這一輪的輸出有工具呼叫」（icpe-data） | jobs.txt `cue=1`＋`cue.u8`；沒接進 sweep |
| 5 W5 | 完成，接進 sweep（`RUX_W5=1`） | CPU 層 INT4＋一次拆完的殘差 |

## 3. 驗證

**新功能關閉時＝Python**
- v8 最終版（`a4ded3ca…`）對 v6 那次的 Python 參考，3 份紀錄 × 4 組子集：CSV 177,072 格、npz 11,424 個陣列，0 個不同。runs：`20261107-035250-icpe-rux-v8f-reg-mc`、`…-035436-…-v8f-reg-bl`、`…-035817-…-v8f-reg-sa`。
- G1 128,064 格、xc_sa 1,280 格，0 個不同：`…-040213-…-v8f-off-g1`、`…-040439-…-v8f-off-xc_sa`。

**新功能**（`smoke_v8.py --tests queue queue2 cue w5 apc`）
- 每份紀錄 696 項，3 份紀錄共 **2,088 項，0 個 FAIL**。
- runs：`20261107-035250-icpe-rux-v8f-smoke-mc／sa／bl`。逐類數字在 `V8_rows.csv`。
- 每一類在每份紀錄的項數都相同：

| 檢查 | 每份紀錄的項數 | 做法 |
|:--|--:|:--|
| `w5_degen_eq_v7` | 110 | W5 路徑、w5＝BF、dc1＝0、dc2＝d → 和 v7 逐位元相同（CPU 11 規則 × GPU 4 種 × 慢層上限開／關 × CPU 8／111 GiB） |
| `w5_real_bytes` | 110 | 真的 W5：CPU 讀的位元組＝CPU 命中塊數 × i4；寫入是 i4 的整數倍 |
| `apc_whole_eq_v7_LRU`、`belp_whole_eq_v7_BEL` | 42＋42 | 一次逐出整個 session ＝ v7 GPU LRU／BEL（CPU 7 規則 × wb／wt × 慢層上限 × CPU 2 種） |
| `apc_partial_audit`、`belp_partial_audit` | 42＋42 | 逐塊逐出；每 200 個請求從頭重算 ref_gpu、ref_gc、gpu_used；斷言 S_GPU 完整、不超過預算；BELP 每次選擇另外全掃比對 |
| `belp_fast_eq_slow` | 42 | 兩份實作互比：捷徑＝每塊全掃 |
| `apc_bigGPU_eq_v7`、`belp_bigGPU_eq_v7` | 42＋42 | GPU 夠大、不會逐出 → ＝v7 |
| `belp_*_noiseB1` | 2＋2 | 有 W4 雜訊 B1 時 BELP 一次整個＝BEL；逐塊稽核 |
| `apc_w5_audit`、`apc_queue_audit` | 2＋2 | APC 和 W5、排隊一起開，稽核通過 |
| `cue0_cuep／cueo_eq_v7_lru` | 16＋16 | 線索全 0 → ＝GPU LRU |
| `cue1_cuep_eq_v7_ttlb_inf` | 16 | 線索全 1 → CUEP＝TTLB:∞ |
| `cue_rand_*_runs` | 16＋16 | 合成線索（Bernoulli 0.5）能跑完 |
| `queue_spread_*`、`queue2_spread_eq_noqueue` | 20＋20＋16 | 到達改成 r × 10⁶ 秒 → q=1／q=2＝沒有排隊，等待全 0 |
| `queue_real_exact`、`queue2_real_exact` | 20＋16 | 原本的到達：開始時刻、等待、TTFT 逐請求精確重算（位元相同） |
| `noq_v8_eq_v7` | 2 | 沒有新欄位的 job：v8＝v7 |

**端到端（sweep）**
- 管線重現版本 B：§4a，306 列逐值相同。
- APC 一次逐出整個 session：run `20261107-033743-icpe-rux-v8-sweep-whole-mc`，wb、wt 各 1,040 格，APC＝LRU、BELP＝BEL，0 個不同；`py_fallback_jobs=0`。
- W5 的退化：§4b，254,592 格。
- BELP 捷徑：run `20261107-041355-icpe-rux-v8-belp-speed-ta`。和每塊全掃比，CSV 832 格、npz 64 個陣列，0 個不同。14 個 job 合計 571 → 246 執行緒秒（快 2.3 倍）；平均仍要 17.5 s，約是 BEL 的 10 倍。

**另外**
- W3 的資料：TraceLab 原始紀錄（run `20261107-033551-icpe-rux-w3-cue-stats-tracelab`），「輸出有工具呼叫」和「下一輪由工具結果觸發」一致 96.9%（SEMANTICS 4-2）。
- 舊執行檔上的測試（`20261107-033505-…-v8-smoke-all-sa／bl`）是我停掉的，被最終版取代，所以沒有 exit_code。停掉前 476＋593 PASS、0 FAIL。

### 3c. 排隊的發現

見 `SEMANTICS_2to5.md` §2a：
- **open-loop**：SemiAnalysis N16 在 nfs、GPU 忙時平均等 2,000–19,000 秒，沒有意義。
- **照紀錄順序的 closed-loop**：比工作守恆的版本多等 30–300 倍。
- **正確的模型**：工作守恆的 closed-loop。用同樣的服務時間估，ssd、GPU 閒時平均等 1.5–1.8 s，服務時間本身是 1.4–2.2 s，所以排隊不能忽略。

## 4. 探索（不是正式結果）

### 4a. APC／BELP（Mooncake toolagent R1／R2；版本 B 上限最高的工具類變體）

**怎麼跑**
- `sweep_v8.py`：CPU 8、111.0625、1,024 GiB × CPU 17 個規則（POL17）× 慢層 ssd／b32／nfs × GPU wb 111.0625 GiB × GPU 18 個規則（LRU、BEL、**APC、BELP**、TTLA／TTLB 各 7 個）。
- 9,456 個 job、16 worker，牆鐘 22 分鐘（run `20261107-033906-icpe-rux-v8-explore-apc-ta`，執行檔 `rsdb455a36`）。
- 分析（run `20261107-040117-icpe-rux-v8-explore-apc-analyze`，`explore_apc.py`）：
  - 同版本 B 的算法（`next_sim_analyze.py`，沒有 `--rival-only-gpu`）；
  - APC 是不預測的 GPU 規則，② 和 ④ 都有；
  - O-GPU 的 ④ 另外加 BELP，自己算。

**重現檢查（先證明管線對）**
- 拿掉 APC、BELP 的列重算，版本 B 表上這些格的 306 列（`mc_p1_headroom_components.csv.xz`，origin/icpe-next-main 最後改在 `baa508d0`）全部對上。
- T_oracle、T_rival、上限、最佳對手、oracle 的組合，**逐值相同、0 個不同**。

**結果**（逐列：`V8_explore_apc_rows.csv`、`V8_explore_ogpu_belp_rows.csv`）
- **APC 幾乎沒有影響**：
  - 306 列裡，② 最多快 **0.035%**（R1、111.0625 GiB、nfs、閒：0.243572 → 0.243487 s），④ 最多快 0.004%，上限最多變 **0.027 個百分點**。
  - 上限 ≥15% 的格數每個元件都不變。
  - APC 只在 nfs 的格子成為最佳：② 9 列、④ 5 列。
- **O-GPU 的 ④ 加 BELP**：上限略升。R1 中位數 25.06% → 25.25%、最大 40.44% → 40.91%；R2 中位數 23.85% → 24.46%、最大 36.66% → 37.43%。BELP 比 BEL 好的格：R1 12／18、R2 13／18。
- **解讀**〔推論〕：
  - 這個變體在 111 GiB 的 GPU 上，逐塊保留前段帶來的額外命中很少。
  - 所以「② 不含 block 粒度 APC」這個缺口在 Mooncake toolagent 上不影響結論。
  - **其他變體 NOT_MEASURED**（TraceLab 的 session 長很多，可能不同）。
- **速度**：
  - APC 每個 job 0.18 s（LRU 0.17 s），幾乎不多花。
  - BELP 在這次的執行檔（每逐出一塊都全掃）要 78.8 s，是 BEL（1.76 s）的 45 倍。
  - v8 最終版加了「不改結果的捷徑」（§3），速度見 §3。

### 4b. W5（同一個變體，CPU 111.0625、128、256 GiB）

**怎麼跑**
- 同一組 CPU 17 × GPU 16 個規則（LRU、BEL、TTLA／TTLB × 7），各 9,012 個 job、16 worker：
  - BF16（現在的 CPU 層）：run `20261107-040115-icpe-rux-v8-explore-w5-base`，231 s；
  - W5（`RUX_W5=1`）：run `20261107-040512-icpe-rux-v8-explore-w5-int4`，207 s。
- 分析：`explore_w5.py`，run `20261107-040843-icpe-rux-v8-explore-w5-analyze`。三種情況都用版本 B 的算法：
  - base＝BF16；
  - forced＝全部 W5；
  - allowed＝每個組合事後挑兩種格式裡較快的（SEMANTICS 5-2）。
- **端到端退化檢查**（run `20261107-040948-icpe-rux-v8-explore-w5-degen`）：走 W5 的路徑、參數等於沒開（`RUX_W5=degen`），和 BF16 的 sweep 比。全部 17 × 16 個規則：CSV 254,592 格、npz 19,584 個陣列，**0 個不同**。這補上 `smoke_v8.py` 沒涵蓋的 O-DROP、O-BELS、O-ALL、FREQ2Q、LRU+NONE、LRU+2HW。

**結果**（逐列：`V8_explore_w5_rows.csv`）
- **W5 幾乎全面更快**：9,792 個組合裡 9,788 個 W5 較快，所以 forced 和 allowed 的上限相同。
- **② 和 ④ 都變快**（中位數）：② 快 18–25%；④ 視元件快 5–24%（O-ALL 5%、O-GPU 10%、O-PF 16%、O-ORD 22%、O-DROP 22%、O-NOW 24%）。
- **上限有升有降**（每格變化的中位數，百分點）：

  | 元件 | R1 | R2 | 上限 ≥15% 的格數（R1：base → W5） |
  |:--|--:|--:|:--|
  | O-GPU | −8.1 | −6.3 | 18 → 8 |
  | O-ALL(report) | −8.8 | −7.8 | 18 → 18 |
  | O-PF | −5.1 | −3.3 | 18 → 14 |
  | O-NOW | 0.0 | 0.0 | 0 → 0 |
  | O-DROP | +0.1 | +2.7 | 18 → 16 |
  | O-ORD | +2.2 | +6.0 | 17 → 14 |

- **NP:53 事先登記的預期是「上限下降或不變；若上升，要查原因」**。O-ORD 的中位數上升，O-DROP 在 nfs 的格最多升 **30–39 個百分點**。例：R2、111.0625 GiB、nfs、閒，13.4% → 52.5%：④ 0.424 → 0.198 s，② 0.489 → 0.417 s。
  - 退化檢查證明程式路徑是對的，所以這是模型的效應，不是接錯〔推論〕。
  - 可能的原因〔推論，沒有驗證〕：INT4 讓 CPU 層多放 3.2 倍。O-DROP 丟掉不會再回來的 session 後，能把要回來的都留在 CPU、避開 nfs；不預測的規則留著那些不會回來的 session，仍要從 nfs 讀。
  - **查了一格**（R2、111.0625 GiB、nfs、閒；同一份 sweep 的引擎計數，沒有新的模擬），回來請求的次數：

    | 這格的最佳組合 | 格式 | CPU 載入 | 慢層載入 | 重算 | 丟掉 session |
    |:--|:--|--:|--:|--:|--:|
    | ④ O-DROP × TTLA:100 | BF16 | 10,660 | 5,110 | 34,387 | 12,629 |
    | ④ O-DROP × TTLB:300 | W5 | 24,926 | 1,361 | 19,520 | 12,629 |
    | ② LRU+2HA × TTLA:300 | BF16 | 6,507 | 1,874 | 46,821 | 21,230 |
    | ② LRU+2HW × TTLA:100 | W5 | 13,539 | 2,380 | 39,501 | 19,997 |

  - 和推測一致：④ 用多出來的容量，把慢層載入減 73%、重算減 43%；② 的最佳做法本來就丟掉大部分 session，重算只少 16%。只看了一格。
- 只有一個變體、3 個 CPU 容量；**不是正式結果**，正式範圍要登記（PR:57）。

## 5. 還沒做、要決定的

- **使用者決定**：
  - **Q3-a**：排隊的 ④ 知不知道排隊造成的新服務順序（SEMANTICS §2a）。決定後才寫工作守恆的 closed-loop 引擎。
  - **登記**：SEMANTICS_2to5.md 是 PREREG 的草稿，要由 h200／icpe-data 寫進去，並列出已看過的版本 A、B 結果，以及這份探索。
- **要別人做**：
  - **icpe-data**：nb 檔多存一欄「這一輪的輸出有工具呼叫」（TraceLab：`tools` 非空；CacheWise：`trigger`；FreeInference：`step_trigger`），W3 才能跑真資料。
  - **icpe-sim**：審 v6／v7／v8，決定要不要併進 `code/p3x/x1fast`；把 W3、排隊接進 sweep。
  - **另一個 session**：審 SEMANTICS 與 v8 程式的對應。
- **還可以做的**：
  - W5 上限上升的原因多看幾格；
  - BELP 再加速（現在是 BEL 的 10 倍時間）；
  - 慢層上限的加速候選 1、4。

## 6. run 目錄（`/root/icpe-next/runs/`）

| run | 內容 | 結果 |
|:--|:--|:--|
| `20261107-031517-icpe-rux-v8-smoke-qc-mc`、`…-031722-…-v8-smoke-w5-mc`、`…-032149-…-v8-smoke-apc-mc` | 開發中的 Mooncake 測試（排隊＋線索、W5、APC） | 142、220、260 PASS，0 FAIL |
| `20261107-033505-…-v8-smoke-all-sa／bl` | 舊執行檔上的測試，**被我停掉** | 停掉前 476、593 PASS，0 FAIL |
| `20261107-034306-…-v8-smoke-q2-sa`（exit 1，測試程式的錯）、`…-034507-…-v8-smoke-q2-sa` | closed-loop 排隊的測試 | 第二次 94 PASS |
| `20261107-033729-…-v8-sweep-whole-mc`（exit 1，忘了建 req 目錄）、`…-033743-…-v8-sweep-whole-mc` | sweep 端到端 | 第二次 0 個不同 |
| `20261107-033551-icpe-rux-w3-cue-stats-tracelab` | TraceLab 線索統計 | §3 |
| `20261107-033906-…-v8-explore-apc-ta`、`…-040117-…-v8-explore-apc-analyze` | APC 探索 | §4a |
| `20261107-040115-…-w5-base`、`…-040512-…-w5-int4`、`…-040843-…-w5-analyze`、`…-040948-…-w5-degen` | W5 探索 | §4b |
| `20261107-041355-…-v8-belp-speed-ta` | BELP 捷徑 | §3 |
| `20261107-035250-…-v8f-reg-mc`、`…-035436-…-v8f-reg-bl`、`…-035817-…-v8f-reg-sa`、`…-040213-…-v8f-off-g1`、`…-040439-…-v8f-off-xc_sa` | 最終版：新功能關閉 | 0 個不同 |
| `20261107-035250-…-v8f-smoke-mc／sa／bl` | 最終版：新功能 | 2,088 PASS，0 FAIL |

所有 run 的 MemAvailable 都沒有低於 3 GiB。
