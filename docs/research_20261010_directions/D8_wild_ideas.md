# D8：激進方向／換問題（wildcard）

**agent**：D8（8 個方向探索 agent 之一；不碰 GPU）
**開始**：2026-10-10T05:33:35Z（`date -u`）
**判準寫下的時間**：2026-10-10T06:06:40Z（`date -u`）。
- 誠實註記：這時文獻搜尋**已經做完**，但每個方向的分數和前 3 名**還沒寫**。所以判準不是「完全看不到資料」寫的。
**產出**：本檔＋`cards/D8_*.md`（46 張）；新下載的 PDF 在 `/mlsteam/data/tiara/papers/d8_20261010/`（不進 git）。
**本檔沒有任何新的 GPU 量測。** 用到的數字只有三種：
- 既有 CSV（標 run_id）；
- 論文原文（標〔原文 p.X〕或〔摘要〕）；
- 我自己用公式算的（標〔算術〕）。

**標記**：〔原文 p.X〕PDF 實體頁；〔摘要〕只讀了 arXiv 摘要頁；〔文件〕系統文件或部落格；〔程式碼 檔:行〕；〔算術〕；〔判讀〕我的推論；〔未查證〕；`NOT_MEASURED`。

---

## 0. 一頁看懂

**問題**：第一階段到第 1 輪都顯示，「寫入時依位置決定放哪層」會被延後版追平（一次一個請求的前提下）。D1–D7 在補那個前提的洞。D8 的工作是**換問題**：在 2025–2026 年 KV 分層的前線，找出 (i) 我們的資產有優勢、(ii) 一張 MI300X 上 2–3 週內有機會拿到**站得住的正面結果**的方向。

**交出來的**：15 個方向（§3 總表、§4 細節），每個都有 ≤1 天的殺法；前 3 名有開跑前寫死的預先登記草稿（§6）。

**前 3 名**（理由見 §5）：

| # | 方向 | 一句話 |
|:--|:--|:--|
| 1 | **D8-01 κ_E：能耗帳** | 「重算 chunk i 省不省能量」有一條門檻：κ(i) < (P_i+P_s)/(P_b−P_i)。拿已量的 f(i) 和各層頻寬代入：只要 GPU 忙／閒功率比 P_b/P_i 落在 1.6–3.2（`NOT_MEASURED`），CPU 層最省能量的是「只載入」、NFS 層是 Cake——和時間最佳的答案（每層都 Cake）不同。只缺兩個功耗值，1 天可殺 |
| 2 | **D8-02 κ_$：KV 版五分鐘規則** | 把「丟掉、要用時重算」當成一層來算錢。已量的 f(i)：32K 尾端 chunk 的重算成本是開頭的 2.93 倍，所以「一個 byte 值得存多久」隨位置變；在 0.33 GiB/s 的層，Cake 只需要存最後約 20% 的 bytes〔Lit-C §1.3 的 b/n＝0.797〕。前作 MatKV 的「十天規則」沒有分層、沒有部分重算 |
| 3 | **D8-03 遠端 KV 的尾延遲保險** | 真實遠端層有固定延遲下限 F 和長尾。〔算術〕F 讓 Cake 在打平點的好處從 2 倍降到 2−F/C，但尾巴上 Cake 有上界。要比的是延後版「晚一點才開始重算」能不能用更少 GPU 拿到一樣的 p99。第一步只用 fio 量 NFS，不用 GPU |

**其他 12 個**：方向 4（κ 隨同機負載變）、7（RL 過期 KV）、11（閒時重算）、15（跨請求 Cake）列為「可能」；方向 5、6、10、12、13 是「可能但有更強的前作或做不到」；方向 8、9、14 預期是死路（§3 的判定欄）。

**查證**：39 篇論文（34 篇是專案先前沒收的）＋7 份系統文件／部落格，每篇一張卡（`cards/D8_*.md`，共 46 張）。

**你必須知道的五件事**：
1. **這個 container 讀不到主機能耗（RAPL）**：`/sys/class/powercap/` 下有 `intel-rapl:*` 項目，但沒有可讀的 `energy_uj`。GPU 的能量計數器有：`amd-smi metric -E`，以及 Python `amdsmi_get_energy_count`（回傳累加值、解析度、時間戳）〔程式碼 `/mlsteam/workspace/venv/tiara/lib/python3.12/site-packages/amdsmi/amdsmi_interface.py:4269–4286`〕。所以方向 1 只能量 GPU 那一塊，整機只能用參數掃。
2. **vLLM 已經會「搶佔後從 CPU 載回」**：0.19.1 的 OffloadingConnector 每算完一個 block 就寫進 CPU〔程式碼 `/mlsteam/workspace/src/vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:194–267`〕，搶佔時先等寫完〔同檔 :269–287；`worker.py:299–308`〕；排程器搶佔時把 `num_computed_tokens` 歸零〔`v1/core/sched/scheduler.py:956–976`〕，回來時靠 connector 查到的前綴載入。方向 6 的新穎性因此很低。
3. **RL 的「過期 KV」大概不用刷新**：一篇二手部落格說 Magistral 與 Nemotron 3 Super 試過「權重同步後重算 KV」，沒有好處〔文件，`cards/D8_Blog_AsyncRLSolved.md`；原始報告未查證〕。方向 7 多半會死，而且它是有損的，要老師同意。
4. **`runsh` 會吃掉引號**：它用 `printf '%s\n' "$*"` 寫 `cmd.sh`〔`/mlsteam/workspace/bin/runsh`〕，所以 `m7run x bash -c 'a; b'` 會被拆壞。§6 的第一個指令都是單一程式、不用引號。
5. **D3 已經判死「事先算誰會贏」**，而且發現：用校準頻寬預測真實裝置（CPU／本地 SSD／NFS）上 Cake 的加速，誤差中位 25%，原因是真實路徑的有效頻寬 ≠ 校準頻寬〔`D3_kappa_map.md` §0 的 R1 列〕。這反而支持方向 3：靠校準做「抓或算二選一」（llm-d 的做法）在打平點附近會選錯，Cake 不需要校準。

---

## 1. 問題是什麼、要證明什麼

**問題**：
- KV 分層研究在 2025–2026 年搬到了新的戰場：agent 的工具呼叫、RL rollout、遠端 KV 池、能耗與成本。
- 我們手上有別人少有的東西：bit-exact 的 Cake harness（GPU 從前面重算、I/O 從後面載入）、真的 pinned CPU 層加上限速的 SSD／NFS 層、誤差 1–3% 的虛擬時鐘模擬器、兩個平台量過的 κ、MI300X 的計數器。
- 問題是：哪些新方向能讓這些資產變成**正面、站得住**的結果？

**要證明的（本檔層級）**：
- 對 15 個方向，每個都給出「≤1 天就能殺掉它」的測試，並誠實說出預期結果。
- 對前 3 名，把主張、對照組、指標、停損在開跑前寫死。

**不是要證明的**：不是要救「寫入時依位置決定」（S5）；本檔也不寫論文。

**白話名詞**：
- **κ（kappa）**：本論文的定義，重算一段 KV 的成本 ÷ 把它搬回來的成本。κ 大＝重算貴，該載入；κ 小＝重算便宜。下面的 κ(i) 是第 i 個 chunk（512 token）的值。（Lit-C 用的是倒數 ℓ/f，注意不要混。）
- **Cake**：還原時 GPU 從前面重算、I/O 從後面載入，兩邊在中間會合。
- **層（tier）**：KV 可以放的地方：GPU 記憶體、CPU 記憶體、本地 SSD、NFS、遠端 KV 池；還有「不存，要用時重算」。
- **TTL**：存活時間，過了就丟。
- **搶佔（preemption）**：GPU 記憶體不夠時，把正在跑的請求先踢出去，之後再接回來。
- **p99**：100 次裡第 99 慢的那次；「尾延遲」。
- **下限（floor）F**：遠端讀取不管多小都要付的固定延遲（RPC、註冊記憶體、找 metadata）。
- **hedge（避險）**：主路徑太慢時，晚一點再開一條備援路徑，用先完成的。

---

## 2. 判準（06:06:40Z 寫下，之後不改）

### 2.1 每個方向要交代的事

一行主張與「為什麼是現在」；最近前作（打開過、附 URL）；我們的資產多給了什麼；延後版或更簡單的雙胞胎；≤1 天的最便宜殺法；到 GPU 確認的大概成本；三個分數。

### 2.2 三個分數（1–5）

| 分數 | 5 分 | 3 分 | 1 分 |
|:--|:--|:--|:--|
| **新穎**（N） | 查不到有人做過這個組合 | 有相鄰的前作，我們多一個明確維度 | 已有人做過，或只是換個場景重跑 |
| **可行**（F） | 現有 harness／模擬器／CSV 就能做，1 週內 | 要寫新程式或下載新模型，2–3 週 | 一張 MI300X 做不到（要多卡、多節點、或大量訓練） |
| **回報**（P） | 若成立，會改變本論文的主張或給出新的一章 | 若成立，是一節或 workshop 等級的結果 | 若成立，也只是小註腳 |

「可行」也包含「正面結果有沒有可能出現」。我有證據預期會被殺掉的方向，F 或 P 會扣分，並寫出證據。

### 2.3 選前 3 名的規則

1. 先排除需要老師同意的有損方向（10 §3G、CLAUDE.md），除非沒有別的選擇。
2. 依 N×F×P 排序。
3. 同分或接近時，優先：(a) 有 ≤1 天的殺法；(b) 用到別人沒有的資產（MI300X 計數器、bit-exact 的 Cake harness、兩平台 κ）；(c) 不和 D1–D7 重疊。
4. 任何「提早決定」型的想法，預先登記裡一定要放延後版雙胞胎（計劃 10 §2）。

### 2.4 「驗證過的論文」怎麼算

- 只算我**自己打開過**的：arXiv 摘要頁、PDF、或會議／書目頁。
- 系統文件與部落格另外算，不算論文。
- 只在別人參考文獻裡看到書目的（例如 Gray & Putzolu 1987），標〔二手書目〕，不算。

---

## 3. 總表：15 個方向

- 分數依 §2.2；「判定」是**預期**（還沒跑任何測試），用 README 的三級：有看頭／可能／死路。
- 「重疊」欄：和哪個 agent 的範圍有交集，跑之前要協調。

| # | 方向 | 一行主張 | N | F | P | N×F×P | 預期判定 | 重疊 |
|:--|:--|:--|--:|--:|--:|--:|:--|:--|
| 01 | κ_E 能耗帳 | 能耗最省的還原切點 ≠ 時間最省的切點；而且隨「算不算整機」翻轉 | 4 | 4 | 3 | 48 | **有看頭** | — |
| 02 | κ_$ 五分鐘規則 | 把重算當一層、讓 Cake 決定哪些 bytes 不必存，「值得存多久」隨位置、層、平台變幾倍 | 3 | 5 | 3 | 45 | **有看頭** | D4（trace） |
| 03 | 遠端 KV 尾延遲保險 | 有下限與長尾的遠端層上，Cake 的 p99 好處 ≫ 中位數好處；延後版 hedge 可能用更少 GPU 拿到大部分 | 3 | 5 | 3 | 45 | **有看頭** | D7（NFS） |
| 15 | 跨請求 Cake | 多個還原同時進行時，把重算給 I/O 最慢的那個 | 3 | 3 | 3 | 27 | 可能 | D1 |
| 04 | κ 隨同機負載變 | 旁邊跑 decode 與跑 prefill 時，重算的邊際成本不同 | 3 | 4 | 2 | 24 | 可能 | D3 |
| 07 | RL 過期 KV 部分刷新 | 換權重後只刷新部分位置／層 | 4 | 3 | 2 | 24 | 可能（有反證；有損） | — |
| 11 | 閒時重算當預取 | 不存前段，GPU 有空時先重算好 | 3 | 4 | 2 | 24 | 可能 | D1、D4 |
| 05 | 多卡主機 DRAM 爭用 | 8 卡同時卸載讓 CPU 層掉進 Cake 甜蜜區 | 3 | 2 | 3 | 18 | 可能（多半被殺） | D7 |
| 06 | RL rollout 搶佔 | 搶佔改成從 CPU 載回 | 2 | 3 | 3 | 18 | 可能（新穎性低） | D2 |
| 10 | 推測式還原 | 還原期間小模型先草擬，KV 到齊後驗證 | 3 | 3 | 2 | 18 | 可能（多半被殺） | — |
| 13 | SLO 導向淘汰＋Cake 成本 | Tail-Optimized LRU 的成本換成 Cake 的還原時間 | 2 | 4 | 2 | 16 | 可能 | D4 |
| 12 | 跨模型／LoRA KV | 位置 × 層二維 Cake | 2 | 3 | 2 | 12 | 可能（有損） | — |
| 08 | 工具呼叫 TTL＋Cake | 用 Cake 的還原時間設 TTL | 2 | 4 | 1 | 8 | 死路（預期） | — |
| 09 | tree search 分支 KV | 分支尾段用 Cake 還原 | 2 | 3 | 1 | 6 | 死路（預期） | — |
| 14 | decode attention 搬 CPU | NEO 式，在 MI300X 上 | 1 | 3 | 1 | 3 | 死路（預期） | — |

---

## 4. 各方向細節

共同的已量數字（之後直接引用，不再重複來源）：
- f(i)：Llama-3.1-8B，512 token／chunk，GPU 閒置，3 次中位數〔`results/m7_write_policy_mi300x/calib_c1.csv`，run `20261008-130316-m7-c1`〕：f(0)＝28.88 ms、f(31)＝56.12 ms、f(63)＝84.68 ms（32K）；0–63 加總 3.63 s。
- 各層頻寬〔`tier_params.json`，runs `20261008-130257-m7-c0`、`20261008-130438-m7-c2`〕：CPU 35.4 GiB/s、本地 SSD 6.98、NFS 0.331；vLLM 實測 CPU 路徑 3.69〔FULL_REPORT §1〕。一個 chunk＝64 MiB。
- Cake 加速（32K，對兩者較快的）：0.33／0.81／2.9／3.7／6.5／11.6 GiB/s → 1.38／1.93／1.50／1.39／1.24／1.05 倍〔FULL_REPORT §2〕。
- 由上面算出的 κ(i)＝f(i)/ℓ〔算術〕：

| 層 | ℓ（ms／chunk） | κ(0) | κ(63) | 只載入 32K（s） |
|:--|--:|--:|--:|--:|
| CPU 35.4 GiB/s | 1.76 | 16.4 | 48.0 | 0.11 |
| 本地 SSD 6.98 | 8.95 | 3.23 | 9.46 | 0.57 |
| vLLM CPU 路徑 3.69 | 16.94 | 1.71 | 5.00 | 1.08 |
| NFS 0.331 | 188.8 | 0.153 | 0.449 | 12.08 |

### D8-01 κ_E：重算與載入的「焦耳比」和「時間比」不同

- **一行主張**：在 GPU 能量的帳上，「重算 chunk i 划不划算」有一條門檻；代入已量的 f(i) 與頻寬，能耗最省的切點在 CPU 層和 Cake 的時間最佳切點不同，而且隨「算不算整機靜態功耗」翻轉。
- **為什麼是現在**：
  - MatKV（ICDE'26）用「SSD 7 W」對「GPU 峰值 350 W × 0.5 s」，宣稱載入比重算省 1,200 倍能量〔MatKV 原文 p.3、p.6〕。175 J 看起來就是 350 W × 0.5 s 算出來的，不是量的〔判讀〕；而且沒算 GPU 等載入時也在耗電。
  - 卸載服務時 GPU 只用 22–28% TDP〔Bottlenecks 原文 p.8〕：等待不是零功耗，也不是滿載。
  - 只看 GPU 計數器會漏掉整機能耗的 41–45%〔WhereEnergyGo 摘要〕：「邊界」選哪裡會改變結論。
- **算術（門檻）**：
  - 記號：P_b＝GPU 重算時的功率；P_i＝GPU 等 I/O 時的功率；P_s＝分攤到這張卡的整機其他功率。
  - 在 Cake 會合點之前，多重算一個 chunk i：多花 (P_b−P_i)·f(i)，少等一個 ℓ、省 (P_i+P_s)·ℓ。
  - 所以「重算 chunk i 省能量」⇔ **κ(i) < (P_i+P_s)/(P_b−P_i)**〔算術；忽略 DMA 本身的功率差，量的時候要驗證〕。
  - P_s＝0（只算 GPU）時，要「重算才省能量」，P_b/P_i 必須小於下表的值〔算術，＝1＋1/κ(i)〕：

| 層 | chunk 0 值得重算需要 P_b/P_i < | chunk 63 需要 < |
|:--|--:|--:|
| CPU 35.4 GiB/s | 1.06 | 1.02 |
| 本地 SSD 6.98 | 1.31 | 1.11 |
| vLLM CPU 路徑 3.69 | 1.59 | 1.20 |
| NFS 0.331 | 7.54 | 3.23 |

  - **預測**（P_b/P_i 是 `NOT_MEASURED`）：只要 MI300X 的 P_b/P_i 落在 1.6–3.2，GPU 能量最省的是「CPU 與本地 SSD 層只載入、NFS 層用 Cake」；而時間最省的在每一層都是 Cake。兩者在 CPU 層不同。
  - 整機邊界：P_s 越大門檻越鬆，Cake 越省能量。
- **最近前作**：MatKV（[卡](cards/D8_MatKV.md)）、Bottlenecks（[卡](cards/D8_Bottlenecks_energy.md)）、WhereEnergyGo（[卡](cards/D8_WhereEnergyGo.md)）、P/D 能耗重探（[卡](cards/D8_DisaggEnergy.md)，量了不同 KV 傳輸路徑的能耗）、Attention to Detail（[卡](cards/D8_AttentionToDetail.md)，prefix caching 的能耗效應，在 vLLM 設定層級）。都沒有「重算 vs 載入的能耗比」或「混合還原的能耗最佳切點」〔查不到≠沒有〕。
- **我們多了什麼**：
  - harness 能把重算和載入分開、逐 chunk 計時；
  - GPU 能量累加器（`amd-smi metric -E`；Python `amdsmi_get_energy_count`）；
  - A0 的加速數字；
  - 3090 沒有分軌能耗計數器（CLAUDE.md §3），所以這件事只有 MI300X 能做。
- **限制**：主機 RAPL 讀不到（§0 第 1 點）→ P_s 只能掃參數，標〔假設〕。
- **雙胞胎**：這不是「提早決定」。簡單對照：Cake（時間最佳）、只載入、只重算、「最快就最省」（race-to-idle）。
- **最便宜的殺法（≤1 天）**：
  1. 量能量累加器的解析度與更新週期（太粗就 `NOT_MEASURABLE`，停）。
  2. 量 P_i：harness 只載入、限速到 NFS 速度（GPU 只在等）。
  3. 量 P_b：harness 連續重算 chunk（C1 迴圈）。
  4. 代入上表。殺：P_b/P_i ≤ 1.2（等待時幾乎滿載功耗 → 能量跟著時間走），或每一層的能耗最佳切點都和 Cake 差 <1 個 chunk。
- **到 GPU 確認**：2–3 天 GPU。Cake-E（只重算滿足門檻的 chunk）在 harness 裡是改會合規則的小改動。
- **選做：功耗上限軸**：`amd-smi set -o ppt0 <W>` 可以設上限〔amd-smi help〕；container 有沒有權限 `NOT_MEASURED`。上限變低時 f 變大、P_b 變小，門檻會移動。
- **分數**：N4 F4 P3。

### D8-02 κ_$：KV 版的「五分鐘規則」，把重算當成一層

- **一行主張**：把「不存、要用時重算」當成一層，並讓 Cake 決定哪些 bytes 不必存，KV「值得存多久」會隨位置、層、平台差好幾倍；照這條規則設 TTL，在同樣 p90 TTFT 下比固定 TTL 便宜。
- **為什麼是現在**：
  - 五分鐘規則被重新拿出來：ISCA'26 版說 GPU 主機上 DRAM↔flash 的兩平時間縮到秒級，KV tensor 更短〔FiveMinuteRule40 原文 p.1、p.6、p.10〕；MatKV 的「十天規則」〔原文 p.3–4〕。
  - 兩篇都沒有「部分重算」。ISCA'26 版只比 DRAM 和 flash；MatKV 只比 SSD 和整段重算。
  - Gray & Putzolu 1987 原本就有「10 byte rule：用記憶體換 CPU 時間」〔二手書目：FiveMinuteRule40 參考文獻 [19]；原文未讀〕。KV 正是「用記憶體換 GPU 時間」。
- **算術**：
  - chunk i 在層 X 存 τ 秒，租金是 r_X·64 MiB·τ；不存，回來時多花 g·f(i)（g＝每 GPU 秒的價格）。
  - 所以兩平時間 **τ*_i,X ＝ g·f(i) ／ (r_X·64 MiB)**〔算術；還沒加 TTFT 的 SLO 懲罰〕。
  - f(i) 隨位置變大：32K 時 f(63)/f(0)＝84.68/28.88＝**2.93**〔算術〕→ 尾端 chunk 值得存的時間是開頭的 2.93 倍。
  - Cake 會合點之前的 chunk，回來時本來就會被重算。Llama 的 b/n：0.33 GiB/s 時 0.797、3.69 時 0.312、35.4 時 0.047〔Lit-C §1.3，用 calib_c1 算〕。所以在 NFS 這種層，存前面約 80% 的 bytes 幾乎買不到 TTFT。
  - 跨平台：τ* 正比於 g·f(i)。本機只有 MI300X 的逐 chunk f(i)；3090 只有 main.tex 摘要的 κ_cpu＝8.9（16K）。逐 chunk 的 3090 曲線在這台機器上沒找到（未查證）。
- **最近前作**：MatKV（[卡](cards/D8_MatKV.md)）、FiveMinuteRule40（[卡](cards/D8_FiveMinuteRule40.md)）、Can I Buy Your KV Cache（[卡](cards/D8_BuyKV.md)，用錢看存 vs 算，沒分層、沒部分重算）、Continuum（[卡](cards/D8_Continuum.md)，TTL 依重算／載入成本與排隊）。
- **我們多了什麼**：逐 chunk 實測的 f(i)、各層實測頻寬、Cake 的 b、兩平台 κ（部分）、Mooncake trace（D4 已驗證 `hash_ids` 是 512-token block：`len(hash_ids)==ceil(input_length/512)` 三個 trace 都 100%〔D4 §2〕）。
- **雙胞胎**（這是「寫入／閒置時決定 TTL」的想法）：
  - 固定 TTL（5 分鐘、1 小時）；
  - 固定容量 LRU（延後到淘汰時才決定）；
  - 背景掃描器：每 10 秒丟掉超過 τ* 的 chunk（背景版）；
  - 全存；MatKV 式十天規則（整段、只有 SSD vs 重算）。
  - 〔判讀〕依位置的 TTL 和「淘汰時先丟開頭」在錢上幾乎等價（租金按秒算，延後幾秒只多付幾秒）。所以 D8-02 的貢獻是**規則和數字本身**，不是「寫入時」。
- **最便宜的殺法（≤1 天，不用 GPU）**：寫出 τ* 表（層 × 位置 × 平台 × 價格 ±3 倍），再用 Mooncake conversation／toolagent trace 的「同一 block 兩次出現的間隔」，算每種 TTL 政策的「每千次回來請求的成本」與 p90 TTFT。殺：(a) Cake 讓 τ* 變化 <1.3 倍、而且跨平台差 <2 倍（＝我們多加的維度都沒差，MatKV 的十天規則已經夠用）；或 (b) τ* 政策在同 p90 TTFT 下比最好的固定 TTL 省 <5%。
- **到 GPU 確認**：以模擬為主；GPU 只抽查 3 個政策 × 8 個 session 的 TTFT（約 1 天，可和 D8-01 同一批）。
- **分數**：N3 F5 P3。

### D8-03 遠端 KV 的尾延遲保險：Cake、延後版 hedge、校準二選一

- **一行主張**：在有固定下限 F 和長尾的遠端層上，Cake 的 p99 好處遠大於中位數好處；而「晚一點才開始重算」（hedge，延後版）可能用少很多的 GPU 拿到大部分的 p99 好處。
- **為什麼是現在**：
  - llm-d（2026-08）在 pod 之間用 NIXL／RDMA 抓 KV，「抓或算」二選一；GLM-5.2 的抓取有約 1.2–1.3 s 下限，約 8K token 打平；文中警告沒校準時短 prefix 的抓取可能比重算慢〔llm-d 部落格〕。
  - KVCodec（SIGCOMM'26）：遠端重用只有在抓得比算快時才划算；為了省錢，現代服務多跑在「幾十 Gbps 或更低」的網路上，租用儲存伺服器在 AWS 上受限於 19 Gbps；Mooncake 的逐層抓取沒有處理網路抖動的機制〔KVCodec 原文 p.3、p.5〕。19 Gbps ≈ 2.2 GiB/s，正好落在 Cake 的甜蜜區 0.3–4 GiB/s〔算術〕。
  - Cake 自己對波動只給一張軌跡圖，沒有 p99 之類的分布統計〔Cake 原文 p.8〕。
  - D3：用校準頻寬預測真實裝置上的 Cake 加速，誤差中位 25%（真實路徑的有效頻寬 ≠ 校準頻寬）〔`D3_kappa_map.md` §0 的 R1 列〕→ 靠校準的二選一在打平點附近容易選錯〔判讀〕。
- **算術**（理想化：重算時間線性、不計 chunk 粒度）：
  - 整段重算要 C；抓取要 F＋L。
  - 在打平點（F＋L＝C），Cake 完成時間 T＝C²/(2C−F)，比二選一快 **(2−F/C)** 倍〔算術〕。
  - F＝0 時 2 倍；F＝C/2 時 1.5 倍；F＝C 時 1 倍。
  - 所以**中位數**上，下限會吃掉 Cake 的好處；**尾巴**上，抓取卡住時 Cake 最慢也只要 C。
- **最近前作**：Cake §5.7（[卡](cards/D8_Cake_補充.md)，定性）、CacheGen（[卡](cards/D8_CacheGen.md)，逐 chunk 改傳文字讓 LLM 重算〔原文 p.2、p.6〕）、llm-d（[卡](cards/D8_llmd_P2P.md)，校準二選一）、KVCodec（[卡](cards/D8_KVCodec.md)，無損壓縮＋pipeline，和 Cake 正交）、The Tail at Scale（[卡](cards/D8_TailAtScale.md)，hedged request；書目查證，內容二手）。
- **我們多了什麼**：
  - 真的 NFS 掛載（`/mlsteam/data/tiara`，nfs4）可以量真實的長尾，不用假設分布；
  - harness 的 `Tier` 已有每次 I/O 的固定開銷 `c_s`〔`code/m7_restore_harness.py` 的 `Tier`〕，加「每請求下限」和「從實測分布抽延遲」是小改動；
  - 模擬器誤差 1–3%（但對策略間的相對差不夠準，FULL_REPORT §7，所以要上 GPU）。
- **雙胞胎**（Cake 在 t＝0 就開始重算，是「提早決定」；延後版是 hedge）：
  - 依中位數校準的二選一（llm-d 式）；
  - hedge：先只抓，超過期限 τ 才從前面開始重算（τ＝校準抓取時間的 p50／p75／p90）；
  - 逐 chunk 靜態分配（CacheGen 式，無損版）；
  - Cake。
- **最便宜的殺法（≤1 天，不用 GPU）**：fio 量 NFS 上 64 MiB 讀取的延遲分布（30 分鐘）；把分布丟進模擬器跑四個政策。殺：p99/p50 <1.5，而且在下限掃描裡 Cake 對二選一的 p99 好處 <10%。
- **到 GPU 確認**：harness 加兩個小功能，每格 ≥200 次還原 × 3 seed，約 3–4 天 GPU。
- **分數**：N3 F5 P3。

### D8-04 κ 隨同機負載變

- **一行主張**：同一個重算 chunk，旁邊跑 decode（吃記憶體頻寬）時的邊際成本，和旁邊跑 prefill（吃算力）時不同。所以 κ 不只隨硬體、模型、長度變，也隨「同一張卡上在跑什麼」變。
- **為什麼是現在**：chunked prefill 與 prefill–decode 混合 batch 已是預設（Sarathi-Serve〔摘要〕）；POD-Attention 讓 prefill 與 decode 的 attention 在同一個 SM 並行，attention 最多快 59%〔摘要〕。Cake §5.8 只示範一個例子〔Cake 原文 p.8〕。
- **我們多了什麼**：A1 已量過**算力型**干擾：m7_busy 的 matmul 背景讓重算慢 1.48／2.07 倍〔`busy_levels.json`，runs `20261008-130834-m7-c5`、`20261008-130956-m7-c5b`〕。**記憶體型**（decode）干擾還沒量。
- **雙胞胎**：Cake 的會合點會依實際速度自動移動，不需要知道 κ。所以這個方向的價值是**量測**（給 D3 的 κ 地圖補一個軸），不是新政策。
- **最便宜的殺法（≤1 天）**：C1 校準在兩種背景下各跑一次：(a) m7_busy（算力型）；(b) 記憶體型背景（vLLM 只做 decode 的大 batch，或 device 內大塊 memcpy 迴圈）。殺：兩種背景造成的變慢倍數相差 <20%。
- **到 GPU 確認**：約 1 週。先和 D3 協調。
- **分數**：N3 F4 P2。

### D8-05 多卡同時卸載：主機 DRAM 爭用讓「快的層」變慢

- **一行主張**：8 卡機器同時卸載／還原時，每張卡分到的 CPU 層頻寬可能掉到 Cake 的甜蜜區（約 0.3–4 GiB/s），讓第一階段「CPU 層太快、Cake 沒用」的結論在多卡機器上不成立。
- **為什麼是現在**：生產機器多是 8 卡；KV 卸載時 99% 延遲在傳輸〔Bottlenecks 原文 p.1–2〕；SwiftCache 用 NVLink 借別張卡的記憶體來避開 PCIe〔摘要〕。我沒找到專門量「多卡同時卸載時主機 DRAM 爭用」的論文〔查不到≠沒有〕。
- **我們多了什麼**：只有 1 張 MI300X，**只能模擬 DRAM 端的爭用**（在 GPU 所在的 NUMA 節點跑 CPU 記憶體頻寬負載），量不到 PCIe switch 的爭用。主機是 2 顆 EPYC 9684X、2 個 NUMA 節點〔`lscpu`〕。
- **雙胞胎**：不是提早決定。
- **最便宜的殺法（≤半天）**：C0 校準（pinned H2D）在不同強度的 CPU 記憶體負載下各跑一次。殺：最強負載下 H2D 仍 ≥20 GiB/s（A0：≥11.6 GiB/s 時 Cake 只剩約 5%）。
- **預期**〔判讀〕：單顆 Genoa-X 的 DRAM 頻寬遠大於一張卡的 PCIe，很可能被殺；真正的 8 卡爭用在這台機器上量不到。
- **到 GPU 確認**：真的 8 卡做不到；模擬版約 1 週。和 D7（E4 NUMA）協調。
- **分數**：N3 F2 P3。

### D8-06 RL rollout 的搶佔：從 CPU 載回，而不是重算

- **一行主張**：GRPO 類 rollout（每個 prompt n 個長回答、和訓練共用 GPU 所以 KV 空間只有一半）裡，vLLM 的「搶佔就重算」吃掉可觀的 GPU 時間；開 CPU 卸載讓被搶佔的請求從 CPU 載回，rollout 會變快。
- **為什麼是現在**：Seer：「搶佔特別貴，因為要重新 prefill」〔原文 p.1〕；verl 把 rollout KV 卸到 Mooncake store〔文件〕；vLLM-Ascend 有「搶佔時卸到 CPU、回來時載回」的 connector，沒有效能數字〔文件〕；TideRL 把「重複的 prefill 重算」列為純浪費〔摘要〕。
- **我們多了什麼**：MI300X＋vLLM ROCm；D2 正在建的「數 vLLM 卸載／丟棄」工具可以重用。
- 〔判讀〕**Cake 在這裡不對**：rollout 看吞吐，GPU 算力是稀缺資源，Cake 重算的那一半是成本，不是免費。最好的應該是「只載入」，而那是現成功能（§0 第 2 點）。新穎性只剩「第一次在 RL rollout 上量出來」。
- **子情境：共置 RL 的睡／醒**：訓練階段要 HBM，rollout 的 KV 被卸到 CPU 或丟掉（ART：level-1 卸到 CPU、level-2 丟掉〔HF 部落格，[卡](cards/D8_Blog_HF_AsyncRL.md)〕），醒來時載回。〔判讀〕載回時間相對一個訓練 step 很可能 <5%，預期被殺。
- **雙胞胎**：調小 `max_num_seqs` 不讓它搶佔（vLLM 文件建議的做法〔[卡](cards/D8_vLLM_Optimization_Preemption.md)〕）；重算搶佔（V1 預設〔同卡〕）；只載入（OffloadingConnector）；CacheOPT 的「swap 和重算選快的」〔摘要〕；RollPacker 用排程避開長尾〔摘要〕。
- **最便宜的殺法（≤1 天）**：vLLM 0.28 離線產生，64 prompt × n＝16，`max_tokens` 8K–16K（`ignore_eos` 控制長度），`gpu_memory_utilization`＝0.5，`max_num_seqs` ∈ {64, 256, 1024}；記錄搶佔次數與「因搶佔重算的 token 數」。殺：因搶佔重算的 token <5% 總處理 token，或開 OffloadingConnector 對總時間 <3%。
- **到 GPU 確認**：1–2 週（含 3 seed）。和 D2 協調。
- **分數**：N2 F3 P3。

### D8-07 RL 換權重後的「過期 KV」：部分刷新

- **一行主張**：非同步 RL 換權重時，KV 有三種現成做法：全部重算（AReaL〔原文 p.4〕）、全部沿用（vLLM `clear_cache=False`〔文件〕；PipelineRL 的中途換權重〔摘要〕）、全清（verl〔文件〕）。中間的「只刷新部分位置或層」可能用少量重算拿回大部分新鮮度。
- **為什麼是現在**：非同步 RL 是 2025–2026 年主流；Elastic-Cache 在擴散 LLM 上做了「何時、何處刷新」〔摘要〕；RaReCache、DroidSpeak 做跨模型的選擇性重算〔摘要〕。RL 權重版本之間的部分刷新我沒找到〔查不到≠沒有〕。
- **主要反證**：部落格說 Magistral、Nemotron 3 Super 試過「權重同步後重算 KV」，沒有好處〔[卡](cards/D8_Blog_AsyncRLSolved.md)，二手〕。如果全部沿用就夠，部分刷新沒有市場。
- **我們多了什麼**：harness 能逐 chunk 算 KV、逐位元比對；可以用 θ＋ε·(θ_RL−θ_base) 做出很小的權重更新（需要同架構的 base 與 RL 版權重；本機沒有，要下載，可用性未查證）。
- **雙胞胎**：全沿用、全重算。
- **最便宜的殺法（≤1 天，但要老師先同意，因為是有損）**：32K context，ε ∈ {1e-3, 1e-2}，比較「過期 KV 接著生成」和「新鮮 KV 接著生成」的逐 token KL 與 log-ratio。殺：過期 KV 的 |log ratio| 平均 <0.01，或低於 bf16 不確定性造成的雜訊。
- **到 GPU 確認**：proxy 指標約 1 週；真正的訓練層級結論，一張卡做不到。
- **分數**：N4 F3 P2。

### D8-08 Agent 工具呼叫期間的 TTL＋Cake

- **一行主張**：把 Continuum 的 TTL 公式裡的「重新載入成本」換成 Cake 的還原時間，TTL 會變短、HBM 早點釋放。
- **最近前作**：Continuum（TTL 依重新載入／重算成本＋排隊延遲〔原文 p.1–2〕）；Ask the Tool（工具回報進度，工具呼叫後 p90 TTFT −20.7%〔摘要〕）；InferCept（中斷造成的重算佔 forward 時間 37–40%〔摘要〕）；TraceLab（資料〔摘要〕）；MORI（Lit-A 已有卡 `A_MORI.md`）。
- **殺掉它的證據**：Continuum 原文說，就算瞬間載回，回來的請求還是要排隊等 GPU 記憶體〔原文 p.5〕→ Cake 只改善不是瓶頸的那一項。
- **雙胞胎**：Continuum 的 TTL（用只載入的成本）。
- **最便宜的殺法（≤半天，不用 GPU）**：TraceLab 的工具時間分布＋A0 的還原時間代入 Continuum 式的 TTL。殺：>80% 的工具呼叫 TTL 變化 <20%。資料欄位單位先照規則 6 驗證。
- **到 GPU 確認**：若沒死，要在 vLLM 裡實作 TTL＋Cake 還原，約 2 週。
- **分數**：N2 F4 P1。預期死路。

### D8-09 測試時計算：tree search／best-of-n 的分支 KV

- **一行主張**：分支暫停時卸掉分支尾段的 KV，回來時用 Cake 還原。
- **殺掉它的算術**：分支尾段在深位置，重算比開頭貴（32K 處 84.68 ms vs 28.88 ms），而且只有尾段要還原 → κ 大，幾乎都該載入，Cake 沒有用武之地〔算術〕。前作走別條路：ETS 用剪枝提高 KV 共享〔摘要〕；FastTTS（ASPLOS'26）用 prefix-aware 排程〔摘要〕。
- **雙胞胎**：分支尾段只載入（不用 Cake）；或 ETS 式直接剪掉分支。
- **最便宜的殺法（1 小時，不用 GPU）**：用 calib_c1 算「從位置 p 開始、長 k 的尾段」在 3.69 GiB/s 下的 b/k。殺：b/k <10%。
- **到 GPU 確認**：若沒死，harness 支援「從中間位置開始還原」約 1 週。
- **分數**：N2 F3 P1。預期死路。

### D8-10 推測式還原

- **一行主張**：KV 還原期間，小草稿模型先用自己（便宜重算）的 KV 草擬 k 個 token，KV 到齊後一次驗證；無損。
- **前作**：Cake 說和推測解碼正交〔原文 p.9〕；TriForce（COLM'24，推測解碼＋檢索式稀疏 KV〔摘要〕）；Speculative Pre-Positioning（閒時先往下解碼〔摘要〕）。
- **疑點**〔判讀〕：同一張卡上，草稿模型會和 Cake 的重算線搶算力；省下的是 k 個 decode step，相對於還原時間可能很小。本機沒有 0.5B–1B 草稿模型（Lit-C 表：小模型不在本機）。
- **雙胞胎**：不草擬，等 Cake 還原完再 decode。
- **最便宜的殺法（≤1 天）**：先算上限 (k × 每步 decode 時間) ÷ 還原時間；每步 decode 時間 `NOT_MEASURED`，要量。殺：k ≤ 8 時上限 <5%。
- **到 GPU 確認**：要下載草稿模型、寫驗證流程，約 1–2 週。
- **分數**：N3 F3 P2。

### D8-11 閒時重算：把重算當成預取

- **一行主張**：前段不存（或只存尾段），GPU 有空、而且預期 session 快回來時，先把前段重算好放 CPU；用閒置算力換儲存空間。
- **前作**：CachedAttention 的 job-queue 預取（I/O 預取；Lit-A 卡 `A_CachedAttention_補充.md` p.7）；MORI 利用工具呼叫空檔（Lit-A 卡）；Speculative Pre-Positioning（閒時先算〔摘要〕）。
- 這是「延後」型想法（寫入時少存、之後再補），所以要和「寫入時就存」「SSD＋I/O 預取」比。
- **最便宜的殺法（≤1 天，不用 GPU）**：Mooncake trace＋簡單負載假設，算「回來之前有足夠 GPU 空檔重算前段」的比例。殺：<10%。在 D1 的併發模型下可能更低。
- **到 GPU 確認**：harness 加「閒時重算」約 1–2 週。
- **分數**：N3 F4 P2。

### D8-12 跨模型／LoRA 變體共用 KV

- **一行主張**：同一 base 的不同微調版共用 KV：載入 base 的後段，重算前段與少數層（位置 × 層的二維 Cake）。
- **前作**：DroidSpeak 已把「逐層重算」和「載入」pipeline 起來〔摘要〕；aLoRA 從模型設計讓 adapter 直接吃 base 的 KV〔摘要〕；RaReCache（2026-10-08）跨大小模型的選擇性重算〔摘要〕。
- **限制**：有損，要老師同意；品質評估 2–3 週。
- **雙胞胎**：DroidSpeak 式只切層；aLoRA 式從模型設計解決（無損）。
- **最便宜的殺法（≤1 天，不用 GPU）**：用 f(i) 與 ℓ 算「位置 × 層」切分比 DroidSpeak 式「只切層」多省多少 TTFT。殺：<10%。
- **到 GPU 確認**：含品質評估 2–3 週（有損，要老師同意）。
- **分數**：N2 F3 P2。

### D8-13 SLO／尾延遲導向的 prefix 淘汰＋Cake 還原成本

- **一行主張**：把 Tail-Optimized LRU 裡「條目被淘汰後回來要多等多久」改用 Cake 的還原時間（依層、依位置）來算，淘汰得更準、p90 更低。
- **前作**：Tail-Optimized LRU（P90 TTFT −27.5%、200 ms SLO 違規 −38.9%，WildChat〔摘要〕）。
- **我們只多一點**：還原成本改用 Cake 算（依層、依位置）。和 D4 的 trace oracle 重疊。
- **雙胞胎**：T-LRU 原版（本身就是淘汰時才決定）。
- **最便宜的殺法（≤1 天，不用 GPU）**：在 D4 的框架裡比 T-LRU（只載入成本）與 T-LRU（Cake 成本）。殺：p90 差 <5%。
- **到 GPU 確認**：模擬為主，GPU 抽查約 1 天。
- **分數**：N2 F4 P2。

### D8-14 Decode 時把 attention 搬到 CPU

- **一行主張**：在 MI300X 上把 decode 的部分 attention 和 KV 搬到主機 CPU，加大 GPU 的 batch（NEO 式）。
- **前作**：NEO：吞吐 T4 最多 7.5 倍、A10G 26%、H100 14%〔摘要〕——GPU 記憶體越大，好處越小。
- **殺掉它的算術**：MI300X 192 GB，扣掉 Llama-8B 約 16 GB 權重，以 128 KiB／token 算約可放 1.3M token 的 KV〔算術〕。記憶體很可能不是 decode batch 的瓶頸。主機 EPYC 9684X 的大 L3 是有趣的資產，但用不上〔判讀〕。
- **雙胞胎**：不搬（HBM 本來就夠大）。
- **最便宜的殺法（≤半天）**：用上面的算術加上 vLLM 啟動時印出的 KV 容量，看典型負載的 decode batch 是不是被記憶體卡住；不是就殺。
- **到 GPU 確認**：若沒死，要移植 NEO 式 CPU attention，2–3 週以上。
- **分數**：N1 F3 P1。預期死路。

### D8-15 跨請求 Cake

- **一行主張**：多個請求同時還原、而且來自不同層時，把 GPU 重算優先給 I/O 最慢的那個，平均 TTFT 比每個請求各做各的 Cake 好。
- **前作**：Strata（OSDI'26）的排程器已經「讓載入配上足夠的計算」，卡住時在空泡插入 decode〔原文 p.5、p.7〕——但它的計算是別人的新 token，不是重算自己的快取（[卡](cards/D8_Strata_排程補充.md)）；Cake §5.8 只有一個例子。
- **雙胞胎**〔判讀〕：每個請求各自 Cake＋排隊中的請求先開始 I/O——不用聯合排程就可能拿到大部分好處。
- **最便宜的殺法（≤1 天，模擬）**：兩個同時還原（CPU 3.69 vs NFS 0.331），比雙胞胎與聯合排程。殺：<5%。
- **到 GPU 確認**：harness 要支援兩個同時還原，約 1 週。和 D1 的併發模擬器重疊。
- **分數**：N3 F3 P3。

---

## 5. 前 3 名與理由

**依 §2.3 的規則**：沒有一個前 3 名是有損的；N×F×P 分別是 48、45、45，第 4 名（D8-15）是 27，差距明顯。

| 排名 | 方向 | 為什麼選它 | 為什麼可能失敗 |
|:--|:--|:--|:--|
| 1 | D8-01 κ_E | 只有 MI300X 量得到（3090 沒有分軌計數器）；門檻公式已經寫好，只缺兩個功耗值，1 天就知道生死；直接修正一篇 ICDE'26 前作的能耗算法；把本論文「κ 會變」的主張推到「目標函數不同，κ 也不同」 | P_b/P_i 可能太大（任何層都只載入最省）或太小（能量跟著時間走）；只能量 GPU，整機只能掃參數 |
| 2 | D8-02 κ_$ | 不用 GPU 就能做完主體；用的是已量的 f(i)、頻寬、b/n 和已驗證單位的 trace；五分鐘規則是審稿人熟悉的語言；和 D8-01 共用量測 | 結果高度依賴價格（輸入，不是量測）；如果 Cake 只讓 τ* 變一點點，就只是重講 MatKV |
| 3 | D8-03 遠端尾延遲 | 用到真實的 NFS 長尾（不是假設分布）；harness 只要小改；預先登記裡就放了延後版（hedge），符合本專案的延後測試；和 D3「用校準頻寬預測真實裝置，誤差中位 25%」的發現互相呼應 | Cake ≥ 二選一幾乎是定義上成立，審稿人可能說「顯然」；真正的新東西是「延後版夠不夠」和「下限吃掉多少」，這兩個可能都很平淡 |

**為什麼不是其他的**：
- **D8-15 跨請求 Cake**（第 4）：和 D1 重疊大；Strata 已做「載入配計算」；雙胞胎很可能追平。
- **D8-07 RL 過期 KV**：最新穎，但有損（要老師同意），而且有二手證據說全部沿用就夠。
- **D8-06 RL 搶佔**：回報不錯，但「從 CPU 載回」在 vLLM 已是現成功能，Cake 在吞吐目標下不對。
- **D8-04 κ 隨負載變**：Cake 自己就會適應，價值只剩量測；和 D3 重疊。

**三者合起來的故事**〔判讀〕：「κ 不只隨硬體、模型、長度變，也隨**目標**（時間、能量、錢）和**層的延遲分布**（不只是頻寬）變。」這可以接在 FULL_REPORT §6 的選項 C（測量型論文）後面。

---

## 6. 前 3 名的預先登記草稿

> 這些是**草稿**。真正開跑前，負責的 agent 要把它抄進自己的計劃檔、寫上 `date -u`，之後不改。

### 6.1 D8-01 κ_E

| 項目 | 內容 |
|:--|:--|
| **主張** | H-E1：先量出 MI300X 的 P_b、P_i。若 P_b/P_i ≥ 1.6，則在 CPU 35.4 與 3.69 GiB/s 兩層，GPU 能量最省的還原是「只載入」，Cake 多花 ≥10% GPU 能量。H-E2：若 P_b/P_i < 3.2，則在 NFS 0.331 GiB/s，Cake 的 GPU 能量 ≤ 只載入。H-E3：Cake-E（只重算滿足 κ(i) < (P_i+P_s)/(P_b−P_i) 的 chunk）至少在一層、P_s＝0 時，比 Cake 省 ≥10% GPU 還原能量；TTFT 增加多少照實報 |
| **對照組** | Cake（時間最佳）、只載入、只重算、Cake-E |
| **設定** | Llama-3.1-8B（主）、LongAlpaca-7B（MHA，b 大）；32K；層：CPU 35.4、CPU 3.69（限速）、本地 SSD 6.98、NFS 0.331（限速）；每格 ≥20 次還原（次數依計數器解析度調整，先量），3 seed；`GpuWatcher` 包住，contaminated 不進 results |
| **指標** | 每次還原的 GPU 能量（J，累加器差值；扣與不扣閒置基線兩種都報）；TTFT；P_i、P_b、P_dma（載入時）；整機：P_s ∈ {0, 0.5, 1, 2}×P_i 的參數掃描，標〔假設〕 |
| **殺掉條件** | (a) 計數器太粗：20 次重複的變異係數 >10% → `NOT_MEASURABLE`，停；(b) P_b/P_i ≤ 1.2；(c) 每一層 Cake-E 和 Cake 的能量差都 <5% |
| **判定** | 沒被殺，且 H-E1 或 H-E2 成立 → 有看頭；只有門檻式成立、但差 <10% → 可能；被殺 → 死路 |
| **第一個指令** | 見下（只量閒置功耗與計數器的更新週期，2 分鐘） |

```bash
source code/m7_env.sh
m7run d8-e0-idle amd-smi metric -g 0 -E -p -w 1 -W 120 --json
```
- 如果 `-w` 和 `--json` 不能並用，改成 `--csv`（未試，不碰 GPU 的規則下我沒跑）。
- 下一步：寫 `code/m8_d8_energy.py`，在 harness 的 `restore()` 前後讀 `amdsmi_get_energy_count`，不改 `m7_*.py`。

### 6.2 D8-02 κ_$

| 項目 | 內容 |
|:--|:--|
| **主張** | H-F1：在 Mooncake conversation 與 toolagent 兩個 trace 上，「依位置與層的 τ* 丟棄」政策在 p90 TTFT 不比最好的固定 TTL 差 >2% 的條件下，每千次回來請求的成本少 ≥15%；在價格 ±3 倍的掃描中至少 2/3 的價格點成立。H-F2：和 MatKV 式（整段、只有 SSD vs 重算）比，NFS 類層的有效 τ* 差 ≥2 倍 |
| **對照組** | 固定 TTL 5 分鐘、1 小時；固定容量 LRU（延後版）；背景掃描版（每 10 s 丟掉超過 τ* 的 chunk）；全存；MatKV 式十天規則 |
| **指標** | 每千次回來請求的成本（租金＋GPU 秒 × 價格）；p90 TTFT（用 Cake 還原模型）；SSD 寫入量 |
| **輸入** | f(i)：calib_c1；頻寬：tier_params.json；b/n：Lit-C §1.3；價格：從公開頁面抄進 `docs/research_20261010_directions/d8_prices.md`（附 URL 與抄錄日期），沒有就 `NOT_MEASURED`，**不准估** |
| **殺掉條件** | (a) H-F1 省 <5%；或 (b) 和 MatKV 式比 τ* 差 <1.3 倍 |
| **判定** | H-F1 與 H-F2 都成立 → 有看頭；只有一個 → 可能；被殺 → 死路 |
| **第一個指令** | 腳本要先寫（約 150 行）；第一段就是規則 6 的斷言 `len(hash_ids)==ceil(input_length/512)`，並用 trace 自身驗證 timestamp 的單位 |

```bash
source code/m7_env.sh
m7run d8-f0-reuse python code/m8_d8_fivemin.py --step reuse --trace /mlsteam/data/tiara/datasets/traces/conversation_trace.jsonl --out results/m8_directions/d8_f0_reuse.csv
```

### 6.3 D8-03 遠端尾延遲

| 項目 | 內容 |
|:--|:--|
| **主張** | H-R1：真實 NFS（不限速）上 64 MiB 讀取的 p99/p50 ≥ 1.5。H-R2：Llama-3.1-8B、16K 與 32K，Cake 的 p99 TTFT 比依中位數校準的二選一少 ≥25%。H-R3（延後測試）：hedge（τ＝p75）拿到 Cake p99 改善的 ≥80%，而重算花的 GPU 秒 ≤ Cake 的 50%——若成立，結論寫成「延後版夠用」，Cake-at-0 只在 H-R3 不成立時才算贏。H-R4：加每請求下限 F 時，Cake 對二選一的中位數加速符合 2−F/C（±10%） |
| **對照組** | 二選一（中位數校準）、hedge（τ＝p50／p75／p90）、逐 chunk 靜態分配、Cake |
| **設定** | 真實 NFS，以及限速 NFS（每 chunk 延遲從實測分布抽）；F ∈ {0, 50, 200, 1000} ms；每格 ≥200 次還原 × 3 seed；`GpuWatcher`；NFS 量測時段和 D7 錯開 |
| **指標** | p50、p99 TTFT；每次還原的重算 GPU 秒；（可選）D8-01 的 GPU 能量 |
| **殺掉條件** | H-R1 不成立，**而且**所有 F 下 Cake 對二選一的 p99 好處都 <10% |
| **判定** | H-R2 成立 → 有看頭（H-R3 決定主角是 Cake 還是 hedge）；只有 H-R4 成立 → 可能（只是驗證算術）；被殺 → 死路 |
| **第一個指令** | 不用 GPU；fio 會先在 NFS 上寫一個 4 GiB 檔，跑完刪掉 |

```bash
source code/m7_env.sh
m7run d8-r0-nfslat fio --name=nfslat --directory=/mlsteam/data/tiara/runs/_m7io --rw=read --bs=64m --size=4g --direct=1 --ioengine=psync --time_based --runtime=1800 --percentile_list=50:90:99:99.9 --unlink=1 --output-format=json
```
- 注意〔判讀〕：同一個 4 GiB 檔重複讀，NFS 伺服器端的快取可能讓尾巴看起來比較短。第二次量要把檔案加大，或換不同時段。

---

## 7. 刻意沒列的（屬於其他 agent）

| 題目 | 交給 |
|:--|:--|
| 併發、寫入排隊（H3） | D1 |
| 真實 vLLM 丟資料／卡住的頻率 | D2 |
| κ 地圖、事先算誰會贏 | D3 |
| 一寫多讀、會不會回來的 oracle | D4 |
| 長影片 KV、VLM、SSM 混合模型的 Cake、滑動視窗、P/D | D5 |
| 精度當成一層（FP8／INT4） | D6 |
| page cache、O_DIRECT、反向讀、GDS | D7 |

---

## 8. 查證過的文獻

- 39 篇論文；其中 Cake、Strata、Bottlenecks、Sarathi-Serve、CacheGen 這 5 篇專案之前就收過（03_paper_map），其餘 34 篇是新的。另有 7 份系統文件／部落格。
- 「venue」欄寫「未查證」的，只確認了 arXiv 編號與標題、作者。

| 卡片 | 論文 | URL | venue／年（怎麼查的） | 讀了哪裡 |
|:--|:--|:--|:--|:--|
| [MatKV](cards/D8_MatKV.md) | MatKV: Trading Compute for Flash Storage in LLM Inference | https://arxiv.org/abs/2512.22195 | ICDE 2026（arXiv comment） | PDF p.1–4、6 |
| [FiveMinuteRule40](cards/D8_FiveMinuteRule40.md) | Five-Minute Rule 40 Years Later | https://arxiv.org/abs/2511.03944 | ISCA 2026（arXiv journal-ref） | PDF p.1–3、6、10、14 |
| [Continuum](cards/D8_Continuum.md) | Continuum: … KV Cache Time-to-Live | https://arxiv.org/abs/2511.02230 | 未查證 | PDF p.1–5 |
| [AskTheTool](cards/D8_AskTheTool.md) | Ask the Tool, Don't Guess | https://arxiv.org/abs/2609.18849 | 未查證 | 摘要 |
| [TraceLab](cards/D8_TraceLab.md) | TraceLab: Characterizing Coding Agent Workloads | https://arxiv.org/abs/2606.30560 | 未查證 | 摘要 |
| [Seer](cards/D8_Seer.md) | Seer: Online Context Learning for Fast Synchronous LLM RL | https://arxiv.org/abs/2511.14617 | 未查證 | PDF p.1–2 |
| [AReaL](cards/D8_AReaL.md) | AReaL: A Large-Scale Asynchronous RL System | https://arxiv.org/abs/2505.24298 | 未查證（搜尋結果有 NeurIPS 2025 頁，未開） | PDF p.2–4、24 |
| [PipelineRL](cards/D8_PipelineRL.md) | PipelineRL | https://arxiv.org/abs/2509.19128 | 未查證（搜尋結果有 TMLR 2026 頁，未開） | 摘要 |
| [RollPacker](cards/D8_RollPacker.md) | RollPacker | https://arxiv.org/abs/2509.21009 | 未查證 | 摘要 |
| [MISAT](cards/D8_MISAT.md) | Scheduling Mixed RL Rollouts Beyond Prefix Locality | https://arxiv.org/abs/2608.11152 | 未查證 | 摘要 |
| [CacheGen](cards/D8_CacheGen.md) | CacheGen | https://arxiv.org/abs/2310.07240 | SIGCOMM 2024（arXiv comment） | PDF p.2、6 |
| [KVCodec](cards/D8_KVCodec.md) | Efficient Remote KV Cache Reuse with GPU-native Video Codec | https://arxiv.org/abs/2602.09725 | SIGCOMM 2026（arXiv comment） | PDF p.1–5 |
| [TailOptLRU](cards/D8_TailOptLRU.md) | Tail-Optimized Caching for LLM Inference | https://arxiv.org/abs/2510.15152 | 未查證 | 摘要 |
| [BuyKV](cards/D8_BuyKV.md) | Can I Buy Your KV Cache? | https://arxiv.org/abs/2606.13361 | 未查證 | 摘要 |
| [Bottlenecks_energy](cards/D8_Bottlenecks_energy.md) | Understanding Bottlenecks for Efficiently Serving LLM Inference with KV Offloading | https://arxiv.org/abs/2601.19910 | 未查證（本機檔名寫 MLSys26） | 本機 PDF p.1、2、8 |
| [WhereEnergyGo](cards/D8_WhereEnergyGo.md) | Where Does the Energy Go? | https://arxiv.org/abs/2609.29707 | 未查證 | 摘要 |
| [DisaggEnergy](cards/D8_DisaggEnergy.md) | Revisiting Disaggregated LLM Serving for Performance and Energy | https://arxiv.org/abs/2601.08833 | 未查證 | 摘要 |
| [AttentionToDetail](cards/D8_AttentionToDetail.md) | Attention to Detail: … Across vLLM Configurations | https://arxiv.org/abs/2607.09172 | 未查證 | 摘要 |
| [MultiRequestEnergy](cards/D8_MultiRequestEnergy.md) | Performance-Energy Trade-offs … Multi-Request Workflows | https://arxiv.org/abs/2604.09611 | 未查證 | 摘要 |
| [DroidSpeak](cards/D8_DroidSpeak.md) | DroidSpeak | https://arxiv.org/abs/2411.02820 | 未查證 | 摘要 |
| [RaReCache](cards/D8_RaReCache.md) | RaReCache | https://arxiv.org/abs/2610.11358 | 未查證 | 摘要 |
| [aLoRA](cards/D8_aLoRA.md) | Activated LoRA | https://arxiv.org/abs/2504.12397 | 未查證 | 摘要 |
| [ETS](cards/D8_ETS.md) | ETS: Efficient Tree Search | https://arxiv.org/abs/2502.13575 | 未查證 | 摘要 |
| [FastTTS](cards/D8_FastTTS.md) | FastTTS | https://arxiv.org/abs/2509.00195 | ASPLOS 2026（arXiv comment） | 摘要 |
| [NEO](cards/D8_NEO.md) | NEO: CPU Offloading for Online LLM Inference | https://arxiv.org/abs/2411.01142 | 未查證 | 摘要 |
| [TriForce](cards/D8_TriForce.md) | TriForce | https://arxiv.org/abs/2404.11912 | COLM 2024（arXiv comment） | 摘要 |
| [Llumnix](cards/D8_Llumnix.md) | Llumnix | https://arxiv.org/abs/2406.03243 | OSDI 2024（arXiv comment） | 摘要 |
| [CacheOPT](cards/D8_CacheOPT.md) | Mitigating KV Cache Competition (CacheOPT) | https://arxiv.org/abs/2503.13773 | 未查證 | 摘要 |
| [POD_Attention](cards/D8_POD_Attention.md) | POD-Attention | https://arxiv.org/abs/2410.18038 | ASPLOS 2025（arXiv comment） | 摘要 |
| [SarathiServe](cards/D8_SarathiServe.md) | Sarathi-Serve | https://arxiv.org/abs/2403.02310 | 未查證（03_paper_map 記 OSDI'24） | 摘要 |
| [InferCept](cards/D8_InferCept.md) | InferCept | https://arxiv.org/abs/2402.01869 | 未查證 | 摘要 |
| [TideRL](cards/D8_TideRL.md) | TideRL | https://arxiv.org/abs/2608.10402 | 未查證 | 摘要 |
| [RLBoost](cards/D8_RLBoost.md) | RLBoost | https://arxiv.org/abs/2510.19225 | 未查證 | 摘要 |
| [ElasticCache](cards/D8_ElasticCache.md) | Attention Is All You Need for KV Cache in Diffusion LLMs | https://arxiv.org/abs/2510.14973 | 未查證 | 摘要 |
| [SwiftCache](cards/D8_SwiftCache.md) | SwiftCache | https://arxiv.org/abs/2606.16135 | 未查證 | 摘要 |
| [SpecPrePositioning](cards/D8_SpecPrePositioning.md) | Speculative Pre-Positioning | https://arxiv.org/abs/2606.29565 | 未查證 | 摘要 |
| [Cake_補充](cards/D8_Cake_補充.md) | Compute Or Load KV Cache? Why Not Both? | https://arxiv.org/abs/2410.03065 | ICML 2025（Lit-A 已查證） | 本機 PMLR PDF p.1–9 |
| [Strata_排程補充](cards/D8_Strata_排程補充.md) | Strata | https://arxiv.org/abs/2508.18572 | OSDI 2026（PDF 頁尾） | 本機 PDF p.2–8 |
| [TailAtScale](cards/D8_TailAtScale.md) | The Tail at Scale | https://research.google/pubs/pub40801/ | CACM 56 (2013) pp.74–80（書目頁） | 只有書目頁 |

**系統文件／部落格（不算論文）**：

| 卡片 | 來源 | URL |
|:--|:--|:--|
| [llmd_P2P](cards/D8_llmd_P2P.md) | llm-d 部落格（2026-08-15） | https://llm-d.ai/blog/p2p-kv-cache-sharing-llm-d |
| [vLLMAscend](cards/D8_vLLMAscend_RecomputeCPUOffload.md) | vLLM-Ascend 文件 v0.24.0rc | https://docs.vllm.ai/projects/ascend/en/v0.24.0rc/user_guide/feature_guide/recompute_cpu_offload.html |
| [verl](cards/D8_verl_RolloutKVOffload.md) | verl 文件（2026-05-27） | https://verl.readthedocs.io/en/latest/perf/rollout_kv_offload.html |
| [vLLM_AsyncRL](cards/D8_vLLM_AsyncRL.md) | vLLM 文件 | https://docs.vllm.ai/en/stable/training/async_rl/ |
| [Blog_AsyncRLSolved](cards/D8_Blog_AsyncRLSolved.md) | Luke J. Huang 部落格（2026-05-31，二手） | https://luk-huang.github.io/personal-website/blog/is-frontier-asynchronous-rl-solved.html |
| [Blog_HF_AsyncRL](cards/D8_Blog_HF_AsyncRL.md) | Hugging Face 部落格（2026-03-10） | https://huggingface.co/blog/async-rl-training-landscape |
| [vLLM_Optimization](cards/D8_vLLM_Optimization_Preemption.md) | vLLM 文件（搶佔一節） | https://docs.vllm.ai/en/stable/configuration/optimization.html |

**只有二手書目、沒讀原文（不算）**：Gray & Putzolu, "The 5 minute rule for trading memory for disc accesses and the 10 byte rule for trading memory for CPU time", SIGMOD 1987, pp. 395–398（書目取自 FiveMinuteRule40 參考文獻 [19]；ACM 頁 403、DBLP 被擋）。

**看過但沒用的**：搜尋摘要把 arXiv 2511.14510 說成「CLO（CPU 端瓶頸）」，打開後是 LiteCache（head 級的 top-k KV 快取），和本檔無關，沒用。另一份搜尋摘要把「Magistral、Nemotron 試過重算 KV」歸給 HF 部落格，打開後不在那篇（在 Luke J. Huang 的部落格）。

---

## 9. 限制與誠實聲明

- **沒有任何新的 GPU 數字**。§4 的預測（例如方向 1 的「CPU 層只載入最省」）都是條件式的，前提 P_b/P_i 是 `NOT_MEASURED`。
- **判準寫在文獻搜尋之後**（§2 開頭的註記）。分數是我一個人打的，沒有 zero-context 審查（CLAUDE.md §5：開審查 subagent 要使用者同意）。
- **很多 venue 沒查證**：arXiv API 一度回 429（可能是多個 agent 同時在查〔判讀〕），改用 arxiv.org 的摘要頁與網頁搜尋；Semantic Scholar 沒用；DBLP 被 Anubis 擋；ACM 頁回 403。
- **28 篇只讀了摘要、1 篇只開了書目頁**（§8「讀了哪裡」欄）；讀了 PDF 相關頁的有 10 篇。前 3 名的最近前作（MatKV、FiveMinuteRule40、Continuum、CacheGen、KVCodec、Cake）都讀了 PDF 的相關頁。
- **二手證據**：方向 7 的主要反證（Magistral、Nemotron 3 Super）來自部落格，原始報告沒讀。
- **「查不到」不等於「沒有」**（CLAUDE.md §1-7）：方向 1、5、7 的「沒找到前作」都只是有限搜尋的結果。
- **算術的理想化**：方向 3 的 2−F/C 假設重算時間和長度成正比；實際 f(i) 隨位置變大（28.88 → 84.68 ms），真實值要在 harness 上量。方向 1 的門檻忽略了 DMA 本身的功耗差。
- **沒有改任何既有檔案**，也沒有 commit。新下載的 10 個 PDF 在 `/mlsteam/data/tiara/papers/d8_20261010/`（不進 git）。
