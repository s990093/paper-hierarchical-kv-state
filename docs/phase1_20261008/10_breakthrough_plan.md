# 10 破解計劃：在什麼條件下「寫入時決定」才有看頭（給 agent 執行）

> **一句話**：第一階段和 08 消融都顯示，在 GQA 模型上，「寫入時依位置放」會被三種更簡單的做法追平：Cake 在讀取時自己調整、等滿了才套用同一條規則（S4B）、寫穿。
> 這份計劃讓 agent 有系統地往外找：換模型架構、模態、硬體、工作負載、OS 機制、指標、動作。要找的是**寫入當下才有、之後就沒有**的東西。

**日期**：2026-10-09
**狀態**：計劃，**還沒執行**。執行前先給老師看 §0 與 §9 的優先順序。
**上游**：[07_report.md](07_report.md)（第一階段）、[09_ablation_report.md](09_ablation_report.md)（消融）、[QUESTIONS.md](QUESTIONS.md)
**執行者**：agent，可以平行開幾個。§8 有可以直接貼上的指令。

---

## 0. 一頁看懂：問題是什麼、要證明什麼

**研究問題**（[CORE_QUESTION](../CORE_QUESTION_20261008.md)）：KV 寫入時就依「回來時會怎麼被用」決定放哪一層，能不能比「等空間不夠才搬」讓使用者等得更少？什麼條件下能？

**目前的答案**：在測過的設定上（MI300X、Llama-3.1-8B GQA、CPU＋SSD 兩層），**不能**。寫入時決定能做到的事，三種更簡單的做法都做得到：

| 更簡單的做法 | 為什麼追得上 |
|:--|:--|
| Cake 在讀取時決定會合點 | 資料在哪一層，它就自己調整重算多少 |
| 等滿了才套用同一條規則（S4B） | 結果和寫入時先放好一樣 |
| 寫穿（S1、S2b） | 在「搬移卡在請求路徑上」時，它最快 |

**要證明的只有一件事**：

> 存在一個**真實會發生的**設定，在這個設定下，寫入時決定比所有「延後版」（§2）都快 ≥5%，在 GPU 上實測，而且換 ≥3 個 seed 都成立。

其他事情（Cake 有效、空間要緊、CPU 載入要慢）都已經證明了，或者只是前提。

**結果有兩種，都要接受**：
- **找到了**：照原本的方向寫論文。貢獻是「這個條件下，寫入時決定是必要的」，並附上為什麼延後做不到。
- **限時內找不到**：照實寫成「讀取端會自適應時，寫入時放置不需要」，附上條件邊界。這是有價值的負面結果，就像 CLAUDE.md 對 NO-GO 的態度。另一個選擇是換問題。要走哪條，由老師決定。

**和原本計劃（`06` §6）的關係**：`06` §6 開跑前寫死：S5 >15% 才進第二階段（FP8、INT4、品質 ε）；<5% 是「照實寫成何時不值得做，再決定調整哪裡」。第一階段和 08 都是 <5%，所以**照計劃不會自動進第二階段**。這份計劃就是「決定調整哪裡」的那一步；第二階段的精度選擇列為其中一個候選（H14），一樣要過 §2 的延後測試。

**怎麼找**：只往「**寫入當下才有、之後就沒有**」的方向找（§2 的 N1–N4）。不然會一再找到「位置有用，但延後也一樣有用」的條件，08 就是這樣。

---

## 0.1 給 agent：開始之前

**必守規則**（CLAUDE.md §1，加上本專案的做法）：
1. **不准編造數字**。沒量到就寫 `NOT_MEASURED`。來源要標註：文獻標〔原文 p.X〕，自己用公式算的標〔算術〕，推論標〔推論〕。
2. **每個數字都要追溯到 run_id 與輸出檔**。原始輸出放 `/mlsteam/data/tiara/runs/<run_id>/`，不要放 `/tmp`、`/root` 或 repo。用 `code/m7_env.sh` 的 `m7run` 包住指令。
3. **失敗要停下來記錄**：把完整錯誤訊息寫進 `results/RUNLOG_MI300X.md`，不要換個方法硬幹到有輸出。
4. **跑之前先把設計與判準寫死**（像 `08_ablation_plan.md`）。跑完要照實列出所有試過的組合，包括沒用的。
5. 不改 `main.tex`。不 commit 秘密（`SECRETS.local.md`、`code/mi300x/*.local`）。進 git 的檔案要 <1 MB。
6. **外部 trace 或資料集的單位，要用資料自身交叉驗證**（CLAUDE.md §1 規則 6，Mooncake 的教訓）。
7. **文獻要打開原文核對**作者、年份、venue、頁碼。本文件列的論文有些是憑記憶寫的，都標了〔待查〕，不可以直接引用。
8. 每次量測用 `GpuWatcher` 包住；`contaminated` 的 run 不進 `results/`。

**必讀**（照順序；已經讀過的論文不要重查）：
1. `docs/CORE_QUESTION_20261008.md`
2. `docs/phase1_20261008/QUESTIONS.md`
3. `docs/phase1_20261008/09_ablation_report.md`（§0、§4、§5）
4. `docs/phase1_20261008/07_report.md` §0
5. `docs/os_mapping_20261008/03_paper_map.md`：已讀過的 61 篇，**不要重查**
6. `docs/write_path_20261008/02_sota_write_techniques.md`、`03_bottlenecks_and_gaps.md`：17 個系統怎麼寫
7. `docs/research_20261007_vlm/V01_video_kv_rekv_mukv.md`：影片 KV（ReKV、MuKV）
8. `docs/DIRECTION_CAKE_EXTENSION.md`、`docs/WRITE_PROBLEM.md`

---

## 1. 目前已知（不要重做）

### 1.1 Cake 雙向還原（A0，已驗證）

| 條件 | 結果 |
|:--|:--|
| 頻寬約 0.3–4 GiB/s（32K） | 比「只載」「只算」兩者較快的再快 1.4–1.9 倍 |
| 頻寬 ≥7 GiB/s | 好處 <20% |
| 頻寬 ≥11.6 GiB/s | 約 5%；短 context 甚至變慢（4K × 100 Gbps 慢 5.6%） |
| context 長度 | 越長越有效 |

### 1.2 S5 的四個條件：08 已經一起測過了

之前寫「四個條件都要成立，但沒有一起驗證過」。**08 的 G1 就是四個一起**：

| 條件 | G1 的設定 |
|:--|:--|
| 1. 快的層 ≲4 GiB/s | CPU 3.69 GiB/s（vLLM 實測的 CPU 載入速度） |
| 2. 快的層空間緊 | 容量 25%、50% |
| 3. 滿了再搬卡在請求路徑上 | hold（harness 的模型，不是 vLLM 實測） |
| 4. 淘汰時考慮閒置時間 | S5L 與 S4L：最久沒用的 session 先搬 |

GPU 結果（50% 容量，`b2_summary.csv`）：

| 比較對象 | seed 0 | seed 1 |
|:--|:--|:--|
| 淘汰規則相同的 S4L | S5L 快 31.3% | — |
| 不看位置的策略中最好的 | S5L 快 18.9%（對 S2b） | S5L 慢 6.8%（對 S1） |
| **S4B（延後版）** | **S5L 只快 0.7%** | **平手** |

**結論：四個條件必要，但不夠。** 缺的第五個條件是「**這個決定延後做不到**」。另外有兩個新發現：
- 只有 **doc 型**的工作負載（一次寫一大段）才有機會贏。chat 每次新寫的都是尾巴，S5 沒有東西可決定。
- 條件 3 成立時，**寫穿**（S1、S2b）通常最好。

### 1.3 已經追平或打敗 S5 的簡單做法，以及還沒測的更強對照組

| 簡單做法 | 證據 |
|:--|:--|
| Cake 讀取時調整 | 第一階段整體結論 |
| S4+、S4+P（滿了才搬，加上成本感知或閒置時間） | 第一階段 free 設定：S5 ≈ S4+（差 ≤3%）；S4+P ≈ LRU |
| S4B（滿了才套用 b 規則） | 08：和 S5L 平手 |
| 寫穿 S1、限速寫穿 S2b | 08：chat＋hold 時，S1 比 S5 快一倍以上 |

**還沒測、但一定要加的更強對照組**（任何新想法都要贏過它們）：
- **背景提早搬（水位線）**：CPU 用到 x% 就在背景先搬到 SSD，不等滿。這是 Linux kswapd 高低水位的做法，TPP〔待查〕也用。在 hold 下，它可能和寫穿一樣好，又不用寫兩份。
- **回來前預取**：知道 session 快回來時，先從 SSD 搬到 CPU。CachedAttention／AttentionStore〔待查〕有這個做法。
- **頻率感知**：看每段被讀過幾次。一寫多讀時，LRU 分不出哪段常被讀。

---

## 2. 核心判準：「延後測試」

每個新的寫入時想法 X，都要配四個對照組：

| 對照組 | 意思 | 08 的例子 |
|:--|:--|:--|
| 延後版 | 同一條規則，等滿了（或搬移時）才套用 | S4B |
| 寫穿版 | 全部寫到每一層，不做決定 | S1、S2b |
| 背景版 | 同一條規則，在背景或空閒時提早套用 | （還沒做，要新加） |
| 讀取時版 | 讀取時才決定（Cake 會合點、預取） | Cake 本身 |

**判準**：X 要在 GPU 實測、≥3 個 seed 上，**都**贏過四個對照組各 ≥5%，才算「寫入時決定有效」；≥15% 算強。只贏過其中一部分的，照實寫成「某某規則有效」，不能說是「寫入時」有效。

**所以要找的是「寫入時有、之後就沒有」的東西**，只有四種：

| 代號 | 寫入時才有的東西 | 例子 | 延後版能不能補 |
|:--|:--|:--|:--|
| **N1 資訊** | 寫完就消失的資訊 | SSM 狀態在 prefill 過程中被覆蓋；投影前的 hidden state；prefill 時的注意力分數 | **結構上不能** |
| **N2 資源** | 延後要多付一次稀缺資源 | 延後搬＝多一次 CPU 讀加一次 SSD 寫；寫穿＝寫兩份。稀缺的可能是 SSD 寫入額度、DRAM 頻寬、PCIe | 看硬體 |
| **N3 時機** | 延後需要空閒時間，但沒有 | 持續高負載、突發到達、多個請求同時進來 | 看負載 |
| **N4 形式** | 寫成什麼樣子，決定讀取時能做什麼 | 格式（KV、hidden state、embedding）、布局（連續、反序、逐層）、SSM 檢查點的位置 | **結構上不能** |

N1、N4 是結構性的，延後版根本做不到，所以最有看頭。N2、N3 要看硬體與負載，得量了才知道。

---

## 3. 搜尋地圖：七個軸

### A. 模型架構

關鍵量是 **κ＝ℓ／f**：ℓ 是從快的層載入一個 chunk 的時間，f 是重算它的時間。κ 越接近 1，Cake 重算的前段 b 就越大，位置也就越重要。

下表的 KV 大小是從本機 `config.json` 算的〔算術〕：2 × 層數 × KV head 數 × head_dim × 2 bytes。**跑之前要用 harness 實際的 tensor 大小驗算一次**（規則 6）。

| 模型 | 架構 | KV／token | 本機已有 | 為什麼有意思 |
|:--|:--|:--|:--|:--|
| Llama-3.1-8B（現在用的） | GQA，8／32 個 KV head | 128 KiB | ✓ | 基準 |
| **LongAlpaca-7B** | **MHA**，32／32；rope 線性 ×8 到 32K | 512 KiB | ✓ | Cake 原文的主要模型；KV 是 4 倍，重算相對便宜，b 會變大 |
| **Qwen3-30B-A3B** | **MoE**，128 個專家選 8；GQA 4／32 | 96 KiB | ✓ | 每個 token 只算約 3B 參數，重算便宜，效果像 MHA |
| Qwen2.5-7B-1M | GQA 4／28 | 56 KiB | ✓ | KV 小，b 會變小（反方向的對照） |
| Qwen2.5-14B-1M | GQA 8／40 | 192 KiB | ✓ | 中型 |
| Mistral-Nemo-12B | GQA 8／32 | 160 KiB | ✓ | 中型 |
| Nemotron-8B-UltraLong-1M | GQA（Llama） | 128 KiB | ✓ | 128K–1M 長 context |
| Seed-OSS-36B | GQA 8／80 | 256 KiB | ✓ | 大模型，重算貴 |
| Qwen3-VL-8B | VLM | NOT_MEASURED | ✗（資料夾是空的，要重下） | 見軸 B |
| MLA（DeepSeek-V2-Lite 等） | 存潛在向量 | 用 config 算 | ✗ | KV 很小，b≈0；用來確認反方向 |
| MQA（Falcon-7B 等） | 只有 1 個 KV head | 用 config 算 | ✗ | 同上 |
| 滑動視窗混合（Gemma 2／3、gpt-oss） | 局部層加全域層 | 用 config 算 | ✗ | 局部層的舊 KV 用不到，每層可以做不同的決定（N2） |
| SSM／線性注意力混合（Jamba、Zamba2、Falcon-H1、Nemotron-H、Qwen3-Next） | 大部分層沒有 KV，只有固定大小的狀態 | 用 config 算 | ✗ | **N1**：Cake 要從後面載入，需要會合點的 SSM 狀態，而只有寫入時存下的檢查點才有 |
| 小模型（Llama-3.2-1B／3B、Qwen2.5-0.5B–3B） | GQA | 用 config 算 | ✗ | 重算便宜 |

**格式選擇的算術**〔算術〕：存 hidden state（HCache 的做法）每層每 token 是 hidden_size × 2 bytes；存 KV 是 2 × KV head 數 × head_dim × 2 bytes。

| 模型 | hidden state | KV | 結論 |
|:--|:--|:--|:--|
| Llama-3.1-8B | 8 KiB | 4 KiB | hidden state 是 KV 的 2 倍 |
| LongAlpaca-7B | 8 KiB | 16 KiB | hidden state 只要一半 |
| Qwen3-30B-A3B | 4 KiB | 2 KiB | hidden state 是 KV 的 2 倍 |

所以「格式」這個決定只對 MHA 這類模型有意義（H6）。

### B. 模態
- **文字 LLM**：現在做的。
- **圖片 VLM**：重算圖片 token 要先跑 vision encoder，再跑 LLM prefill，所以成本不只看位置。可以存 vision embedding（每 token 一個 hidden_size 向量），讀取時再補 LLM 那段，這是格式選擇（N4）。圖片常在前段，也常被重複使用。
- **影片**：一次寫入幾萬個 token，之後問很多短問題，正好是 doc 型負載，也是模擬裡唯一贏過的類型。ReKV 和 MuKV 的評測卡已經有了。
- **音訊**：優先順序最低。

### C. 硬體與層
- **GPU 速度**：GPU 越快，f 越小，b 越大。模擬可以用 f_scale 近似，但**只有 GPU 實測才算數**。
- **快的層頻寬**：
  - 單一 stream 用 pinned memory 是 35.4 GiB/s，vLLM 實測只有 3.69 GiB/s。
  - 一台機器 8 張 GPU 同時卸載時，每張卡分到的頻寬可能更低（NUMA、DRAM 頻寬、PCIe root complex）。
- **CXL 中間層**：形成三層。只能模擬。
- **慢的層**：本地 NVMe、NFS、遠端 KV store（Mooncake store、LMCache remote、3FS 等，經過網路）、object store。
- **P／D 分離**：prefill 節點把 KV 傳給 decode 節點，這次傳輸本身就是一次「寫入」。

### D. 工作負載
- **chat**（每輪新增 8K）和 **doc**（一次 32K，之後只問短問題）：現在有的。
- **共享前綴、一寫多讀**（RAG、system prompt）：**SCBench 已下載**（`/mlsteam/data/tiara/datasets/scbench`、`hf-cache/hub/datasets--microsoft--SCBench`）。
- **agent 與工具呼叫**：對話中間插入一大段工具輸出。
- **分支**：tree search、best-of-n，多條路徑共享前綴。
- **影片 QA**。
- **到達過程**：現在是一個接一個、gap＝0、沒有併發。要加 Poisson、突發、併發。
- **回來間隔**：從真實 trace 取（`/mlsteam/data/tiara/datasets/traces`、LMSYS-Chat-1M、BurstGPT 等）。**欄位單位要先驗證**。
- **規模**：現在只有 8 個 session，要試更多。

### E. OS 機制（之前沒考慮到的）

| # | 機制 | 和寫入時決定的關係 | 怎麼量 |
|:--|:--|:--|:--|
| E1 | **page cache 與 O_DIRECT** | 寫到 SSD 的資料經過 page cache 時，其實還占著 DRAM（雙重快取），等於偷走 CPU 層的空間。寫入時直接 O_DIRECT 可以避免 | 寫入前後看 `/proc/meminfo` 的 Cached、Dirty |
| E2 | **dirty writeback 的節奏** | `dirty_ratio` 這類參數決定「寫完」實際發生在什麼時候，也就決定 hold 會卡多久 | 用 buffered 和 O_DIRECT 各寫 64 MiB chunk，看延遲分布 |
| E3 | **pinned memory** | 分配很慢，而且有上限（RLIMIT_MEMLOCK）。延後搬需要額外的 pinned 暫存區 | 量分配時間對大小 |
| E4 | **NUMA、多 GPU 爭用** | 條件 1（快的層慢）在多 GPU 機器上可能自然成立 | `numactl` 綁不同節點跑 C0；併發 N 個 stream |
| E5 | **PCIe 讀寫同時** | 搬移（D2H）和載入（H2D）互相拖慢 | **先看已有的** `calib_c0_duplex.csv` |
| E6 | **SSD 內部** | SLC cache 用完後寫入掉崖、GC 停頓、寫入壽命（DWPD）。寫穿和延後搬寫得比較多，會更早掉崖（N2） | fio 持續寫到掉崖（A2 已經量過讀寫干擾） |
| E7 | **NFS 的 close-to-open 語意** | 寫入可能要到 close 才真的送出 | fio 加 strace |
| E8 | **記憶體壓力** | 節點和 7 個鄰居共用，可用的 CPU 空間會變。寫入時決定是在不確定下做的，延後版看得到最新狀況。這一條預期對延後版有利，當作反證 | 記錄 `/proc/meminfo` 的時間序列 |
| E9 | **GPU 直接寫 SSD**（GDS；ROCm 的對應〔待查〕） | 寫入時直接寫 SSD，不占 CPU 暫存區，hold 的定義會改變 | 先查 ROCm 有沒有支援 |
| E10 | **經典 OS 理論** | 見 §5.6 的 Lit-B：多層快取的 exclusive 與 DEMOTE、promotion 比 demotion 好、flash 快取為了壽命而做的 admission、tiered memory 的配置與搬移、以 lineage 重算代替複製 | 文獻 |

### F. 指標
- 現在只看回來請求的 TTFT 中位數。
- 還要看：p99、高負載下的吞吐量與 goodput、**SSD 寫入量**（壽命、成本）、DRAM 用量、每個請求的成本、能耗（MI300X 的功耗計數器〔待查〕）。
- 第一階段 Q1 說「少寫不會縮短等待」，但**沒有評估寫入量本身的成本**。S5 有可能在「TTFT 一樣、少寫 X%」上有 Pareto 優勢，這是 H1。

### G. 動作
- 放哪：現在做的。
- 存不存：admission。
- 存什麼格式：KV、hidden state、embedding，或只存 token。
- 怎麼排布局。
- 切多細：chunk 大小、逐層。
- SSM 檢查點放哪些位置。
- **超出範圍**：有損壓縮或量化（Tiara 的 GPU-FP8／INT4）。第一階段不做，要老師同意才做。

---

## 4. 假設清單：每個都先找最便宜的方法殺掉它

優先順序：**第 1 輪**最便宜、結果最有決定性；**第 2 輪**要寫程式；**第 3 輪**是激進方向，大多只能模擬或查文獻。

| # | 假設 | 為什麼延後做不到 | 最便宜的測法 | 停損（成立就停） | 成本 | 輪次 |
|:--|:--|:--|:--|:--|:--|:--|
| **H0** | 真實系統裡，搬移會擋住請求（hold 是真的） | 前提條件，決定 N3 有沒有意義 | 讀 vLLM OffloadingConnector、LMCache、SGLang HiCache 的原始碼：CPU 滿了、GPU 要釋放時怎麼辦。再用 vLLM 在 MI300X 上把 CPU offload 填滿，看新請求會不會卡住 | 不會卡住 → hold 那組結果只剩理論意義 | 半天 | 1 |
| **H1** | **寫入量的 Pareto**：TTFT 一樣，S5 寫得比較少 | N2 | **只重分析現有的 CSV**（`b_share.csv.gz`、`b2_share.csv` 的 `w_bytes_ssd`、`n_demote`），不用 GPU | TTFT 差 ≤5% 時，S5 比最快的簡單策略少寫不到 20% | 1 小時 | 1 |
| **H2** | **重算便宜的模型**（MHA、低 active MoE、小模型）讓 b 變大 | 本身不是，S4B 也能做；但這是 09 唯一還沒排除的例外 | 每個模型做 C1 校準 f(i)（約 10–20 分鐘 GPU）→ κ 篩選表 → 模擬 → 模擬通過所有對照組才上 GPU | 快的層的 b／n <30% | 1–2 天 | 1 |
| **H3** | **沒有空閒時間**：併發、突發到達 | N3 | 擴充模擬器：多個 stream 同時共用各層和 GPU，加到達過程，**加背景水位線對照組** | 模擬裡背景版追平 | 1 天（模擬） | 1 |
| H4 | **一寫多讀**：共享前綴、RAG、SCBench | N1：寫入當下就知道它會被很多人讀，頻率感知要先看到幾次讀取才知道 | 分析 SCBench 的結構（多少請求共用多少 context、重用距離）→ 模擬 | 頻率感知對照組追平 | 1 天 | 2 |
| H5 | **VLM／影片**：一次寫一大段，加上視覺 token 的重算成本不只看位置 | N4：可以存 vision embedding | 重下 Qwen3-VL-8B；校準「vision encoder＋prefill」每個 chunk 的成本 | b／n <30%，或 embedding 沒有比較省 | 2 天 | 2 |
| H6 | **寫入時選格式**（KV 或 hidden state），讀取時三選一：重算、從 hidden state 投影、載入 KV | N1＋N4 | 只對 MHA 做（看 §3A 的算術）。先查 HCache〔待查〕和後續論文做到哪了；驗證投影回來的 KV 位元完全一致（`kv_bad`） | 新穎性不夠，或 b 沒有變化 | 3 天 | 2 |
| H7 | **混合 SSM 模型的檢查點**：寫入時決定把哪些位置的 SSM 狀態存下來 | **N1，結構上延後做不到** | 先查文獻（Marconi〔待查〕）和可行性（ROCm 上的 mamba kernel；transformers 的純 torch 路徑）。可行才實作 | 已經有人做了，或在 ROCm 上跑不起來 | 查 1 天，做 1 週 | 2 |
| H8 | **預測 session 會不會回來**，不會回來就根本不寫（admission） | N2 | 用真實 trace 的回來分布算 **oracle 上限** | 完美預測的上限 <5% | 1 天 | 3 |
| H9 | **滑動視窗層**：局部層的舊 KV 不存 | N2 | 用 config 和算術估省下多少，再模擬 | 省下的不到 20% | 1 天 | 3 |
| H10 | **寫入布局**：為 Cake「從後面讀」把資料排成連續或反序 | N4 | 用 fio 在 NFS 和本地 SSD 量 IO 大小 2 MiB（每層每 chunk）到 256 MiB 的吞吐量 | ≥2 MiB 時差距 <10% | 半天 | 3 |
| H11 | **P／D 分離**：prefill 節點決定哪些 chunk 傳給 decode 節點、哪些讓對方重算 | N1＋N3：prefill 節點傳完就丟掉 | 文獻加模擬（沒有兩個節點） | 已經有人做了，或模擬 <5% | 2 天 | 3 |
| H12 | **三層（加 CXL）**：延後搬要搬兩次 | N2 | 只能模擬 | 模擬 <5% | 1 天 | 3 |
| H14 | **寫入時選精度**（第二階段的內容：依位置挑哪些 chunk 用 FP8／INT4，總位元組數相同） | N2：延後版要先存 BF16 再降，多寫一次；但「搬移時才降精度」是很自然的延後版，一定要比 | 先算：同預算下 Cake 的會合點會移多少（FP8 讓 ℓ 減半）；再看 `06` §6.1 的組別 | 搬移時才降精度追平，或品質 ε 不可接受 | 1 週（含品質量測） | **要老師同意（有損）** |
| H13 | **page cache 偷 DRAM**：延後搬到「SSD」的資料其實還占著 DRAM | N2 | 量 E1、E2；看 vLLM、LMCache 寫磁碟時是不是用 O_DIRECT | 都用 O_DIRECT，或差 <5% | 半天 | 2 |

**激進方向**（H6、H7、H11 以外，也可以考慮換問題本身）：
- 把問題從「放哪」換成「**存不存＋存什麼格式**」。
- **測量型論文**：「讀取端會自適應時，寫入時放置什麼時候還重要？」把所有軸的邊界畫出來。
- 有損格式：超出第一階段範圍，要老師同意。

---

## 5. 文獻搜尋：怎麼教 agent 查

### 5.1 來源（照順序）
1. **repo 裡已有的卡片**：§0.1 的必讀清單。已讀過的不要重查。
2. **Semantic Scholar API**：
   - 搜尋：`https://api.semanticscholar.org/graph/v1/paper/search?query=<q>&fields=title,year,venue,externalIds,citationCount`
   - 往前追：`/paper/<id>/references`；往後追：`/paper/<id>/citations`
3. **arXiv**：cs.DC、cs.OS、cs.LG、cs.AR，2023–2026。
4. **DBLP 的會議清單**：OSDI、SOSP、NSDI、ATC、EuroSys、FAST、ASPLOS、MLSys、SIGCOMM、ISCA、MICRO、HPCA、SC、SoCC。「重算還是儲存」這個題目也查 VLDB 和 SIGMOD。
5. **系統原始碼與文件**：vLLM（OffloadingConnector、KV connector）、SGLang HiCache（寫入策略選項；我記得有 write_through、write_through_selective、write_back，〔待查〕）、LMCache、NVIDIA Dynamo KVBM、Mooncake store。**程式碼也是證據**：寫入策略常常只寫在程式碼裡，論文沒提。

### 5.2 關鍵字庫（英文）

**Lit-A：LLM KV 系統**
- `KV cache admission policy`、`selective KV cache offloading`、`KV cache write policy write-through write-back`
- `compute or load KV cache`、`KV cache recomputation versus loading`、`hidden state cache LLM restoration`
- `prefix caching hybrid Mamba attention models`、`state checkpointing SSM inference cache`
- `KV cache sliding window hybrid memory management`、`heterogeneous KV cache layers`
- `multimodal KV cache reuse vision tokens`、`video LLM KV cache offloading`、`vision embedding cache`
- `disaggregated prefill decode KV transfer selective recompute`
- `KV cache SSD endurance write amplification`、`KV cache storage layout`
- `position independent context caching`、`agentic workload KV cache reuse`、`multi-turn conversation KV cache tiered storage`
- **找所有引用 Cake 的論文**（Semantic Scholar 的 citations）

**Lit-B：OS 與儲存**
- `exclusive caching multi-level DEMOTE`、`promotion versus demotion multi-level cache`
- `flash cache admission endurance`、`SSD cache admission policy write reduction`
- `tiered memory page placement allocation time versus migration`、`proactive demotion watermark tiered memory`
- `write-allocate no-write-allocate`、`write-through versus write-back storage cache`
- `data lifetime placement SSD multi-stream`、`temperature-aware placement at write time hybrid storage`
- `lineage recomputation instead of replication`、`recompute versus store tradeoff checkpointing`

**Lit-C：模型架構與模態**
- 每個架構族：官方 `config.json`、原始論文、KV 大小公式、prefill FLOPs
- ROCm 支援：vLLM ROCm 的 supported models，以及 mamba／causal-conv1d 有沒有 ROCm 版
- `hybrid model inference memory`、`MLA KV cache size`、`MoE prefill cost active parameters`

### 5.3 滾雪球
1. 每篇高度相關的論文，往前和往後各追一層。
2. 最多追 2 輪；如果某一輪新找到的相關論文少於 2 篇，就停。

### 5.4 怎樣算「相關」
符合任一條就算相關：
- 在寫入或 admission 時，決定放哪、存不存、存什麼格式；
- 比較了延後和提早（寫穿、背景、延後）的做法；
- 跨架構或跨硬體量了重算和載入的成本。

### 5.5 每篇論文的卡片（存到 `docs/research_20261009_explore/cards/<代號>.md`）

```markdown
# <代號> <論文名>
- 出處：作者、年份、venue、連結；讀到第幾頁
- 寫入時做了什麼決定：存不存／放哪／什麼格式／什麼布局／何時寫
- 用什麼資訊做決定？這個資訊寫完之後還在不在？（N1）
- 有沒有和延後版、寫穿版、背景版比較？差多少？〔原文 p.X〕
- 硬體：GPU、各層頻寬、容量
- 模型架構、模態
- 和本研究的關係：支持哪個 H？威脅哪個 H（提供了更簡單的做法）？
- 證據等級：〔原文 p.X〕／〔判讀〕
```

### 5.6 要查證的候選清單（**全部〔待查〕**：名稱和 venue 是憑記憶寫的，必須打開原文核對）

- **LLM KV**：Cake（arXiv 2410.03065）、HCache、CachedAttention／AttentionStore、Pensieve、Mooncake、LMCache、CacheGen、CacheBlend、IMPRESS、Marconi、Jenga、EPIC、SGLang HiCache、Dynamo KVBM，以及所有引用 Cake 的論文。
- **OS 與儲存**：
  - Wong & Wilkes "My cache or yours?"（DEMOTE，ATC'02）
  - Gill "On multi-level exclusive caching"（promotion 比 demotion 好，FAST'08）
  - Karma（FAST'07）、ULC
  - Flashield（NSDI'19）、Kangaroo（SOSP'21）、CacheLib（OSDI'20）、FairyWREN（OSDI'24）
  - TPP（ASPLOS'23）、HeMem（SOSP'21）、Memtis（SOSP'23）、Colloid（SOSP'24）
  - Tachyon（SoCC'14，以 lineage 重算代替複製）
  - Checkmate（MLSys'20，activation 要重算還是儲存）

---

## 6. 篩選計算（用很少的 GPU）

### 6.1 κ 篩選表（H2）
1. **chunk 大小**：從 `config.json` 算，再用 harness 實際的 tensor 大小驗算（規則 6）。
2. **f(i)**：做 C1 校準，512 token 一個 chunk，量到 32K（或模型上限）。
3. **算 b**：頻寬取 {0.33, 1, 3.69, 11.6, 35.4} GiB/s，用 `write_boundary` 算 b 和 b／n。
4. **輸出**：`results/m7_explore_mi300x/kappa_screen.csv`，要有 run_id 和 ts 欄。
5. **通過條件**：在快的層（3.69 或 35.4 GiB/s）時 b／n ≥30%，才進模擬。

**要先改的程式**：
- `code/m7_model.py` 的 `MODEL_GLOB` 寫死成 Llama-3.1-8B，要加環境變數覆寫。
- `BState` 的 `64*MiB`、`m7_sim.py` 的 `CB` 寫死成固定 chunk 大小，要改成參數。
- 非 Llama 架構（Qwen3 MoE、VLM）要走 HF transformers 的路徑。

### 6.2 重分析舊資料（H1）
算每個設定的「TTFT 對 SSD 寫入量」，畫 Pareto 前緣。不用 GPU。

### 6.3 OS 微量測（H0、H10、H13，以及 E1–E7）
每項都是獨立的 run_id。fio、page cache、pinned memory 分配、NUMA，各自一個 run。

---

## 7. 模擬與 GPU 確認（每一關都要過）

1. **模擬器的任何擴充**（併發、新模型、寫入額度、背景對照組）都要先和至少一組 GPU 實測比對，誤差 ±10% 以內（像 08）才能用。
2. **掃描前先把設計寫死**在 `11_explore_ablation_plan.md`：要掃的軸、對照組、判準。
3. **GPU**：每輪最多 4 組設定，每組 3 次、≥3 個 seed。全部照實報告。
4. **選擇偏誤**：從模擬挑出來的最好一組，GPU 上一定要換 seed 再驗（08 的 seed 1 就是例子）。

---

## 8. agent 分工與可以直接貼上的指令

| 角色 | 做什麼 | 碰不碰 GPU | 什麼時候開 |
|:--|:--|:--|:--|
| Lit-A | LLM KV 系統 | 不碰 | 第 1 輪，三個 Lit 平行 |
| Lit-B | OS 與儲存 | 不碰 | 第 1 輪 |
| Lit-C | 模型架構與模態 | 不碰 | 第 1 輪 |
| Screen | H0、H1、κ 表、OS 微量測 | 少量 | 第 1 輪 |
| Sim | 擴充模擬器、掃描 | 不碰（驗證時才用） | Screen 之後 |
| Red-team | 對每個還活著的假設，寫出「更簡單的做法也追得上」的最強論證 | 不碰 | **要使用者明確同意才開**（CLAUDE.md §5：zero-context reviewer） |

**Lit-A／B／C 的指令**（把 `<X>` 換成 A、B 或 C）：
```
你是本專案的文獻 agent Lit-<X>。repo：/mlsteam/workspace/paper-hierarchical-kv-state。
1. 先讀 docs/phase1_20261008/10_breakthrough_plan.md 的 §0、§0.1、§2、§3，以及 §0.1 列出的必讀檔。
   docs/os_mapping_20261008/03_paper_map.md 裡已經讀過的論文不要重查。
2. 用 §5.1 的來源與 §5.2 的「Lit-<X>」關鍵字搜尋，照 §5.3 滾雪球，用 §5.4 判斷相關性。
3. 每篇相關論文照 §5.5 寫一張卡，存到 docs/research_20261009_explore/cards/。
   必須打開原文，寫出頁碼；查不到原文的標「未讀原文」。
4. 最後寫 docs/research_20261009_explore/lit_<X>_summary.md：
   - 對 §4 的每個 H：有哪些論文支持、哪些威脅（提供更簡單的做法）、哪些已經做過
   - 新發現、但 §4 沒列的方向
   - §5.6 每一篇的查證結果（作者、年份、venue 是否正確）
規則：不准編造。原文沒寫的就寫「原文未提」。不要改 main.tex，不要 commit。
```

**Screen 的指令**：
```
你是本專案的篩選 agent。repo：/mlsteam/workspace/paper-hierarchical-kv-state。平台：AMD MI300X。
先讀 CLAUDE.md §1、§4，以及 docs/phase1_20261008/10_breakthrough_plan.md 的 §0–§4、§6。
依序做：
(1) H1：重分析 results/m7_write_policy_mi300x/ 的 b_share.csv.gz 和 b2_share.csv，畫 TTFT 對 SSD 寫入量的 Pareto。
(2) H0：讀 vLLM OffloadingConnector、LMCache、SGLang HiCache 的原始碼，回答「CPU 滿了、GPU 要釋放 KV 時，新請求會不會等」，附檔名和行號；再設計一個最小實驗。
    實驗設計先寫進文件，先不要跑。
(3) κ 表：照 §6.1。要先改程式（MODEL_GLOB 環境變數、chunk 大小參數化），每個改動都要附測試；
    第一個模型做 LongAlpaca-7B。
每個 run 用 m7run 包住，輸出到 /mlsteam/data/tiara/runs/<run_id>/；結果 CSV 要有 run_id、ts 欄。
失敗就記到 results/RUNLOG_MI300X.md 然後停下。
沒量到寫 NOT_MEASURED。量測要用 GpuWatcher 包住，contaminated 的結果不進 results/。
```

**Sim 的指令**：
```
你是本專案的模擬 agent。先讀 docs/phase1_20261008/08_ablation_plan.md、09_ablation_report.md、10_breakthrough_plan.md §2、§7，
以及 code/m7_sim.py、code/m7_write_policy.py。
任務：替 m7_sim.py 加上 (a) 背景水位線對照組，(b) 回來前預取對照組，(c) 多個 stream 併發與到達過程，(d) 新模型的 f(i) 和 chunk 大小。
每個擴充都要先和一組 GPU 實測比對（誤差 ±10% 內），沒過就不准用它下結論。
掃描前先寫 docs/phase1_20261008/11_explore_ablation_plan.md，把設計和判準寫死。
判準照 §2：要同時贏過四個對照組各 ≥5%，≥3 個 seed 都成立。
報告要列出所有試過的組合。模擬結果都要標〔模擬〕，只能用來挑設定。
```

---

## 9. 優先順序、時間與停止條件

| 輪次 | 內容 | 時間 | 過關條件 |
|:--|:--|:--|:--|
| 1 | H0、H1、H2（κ 表）、H3（模擬），Lit-A／B／C 平行 | 2 天 | 至少一個 H 的上限 ≥5%，而且找得到「延後做不到」的理由 |
| 2 | 第 1 輪活下來的，加上 H4–H7、H13 | 3–4 天 | 模擬裡贏過四個對照組，≥3 個 seed 都成立 |
| 3 | 上 GPU 確認；有時間再做 H8–H12 | 2–3 天 | §2 的判準 |

**停止條件**：
- 第 2 輪結束時，模擬裡沒有任何假設贏過四個對照組 → 停。寫「負面結果與條件邊界」報告，找老師決定換問題，或改寫成測量型論文。
- 任一輪發現前提錯了（例如 H0 顯示真實系統不會卡住）→ 停下來更新本文件，再繼續。

## 10. 產出

| 類型 | 位置 |
|:--|:--|
| 文獻 | `docs/research_20261009_explore/`（卡片、`lit_A/B/C_summary.md`） |
| 數據 | `results/m7_explore_mi300x/`（`kappa_screen.csv` 等，每列都有 run_id） |
| 設計與報告 | `docs/phase1_20261008/11_explore_ablation_plan.md`、`12_explore_report.md` |
| 記錄 | `results/RUNLOG_MI300X.md` 新增一節 |
