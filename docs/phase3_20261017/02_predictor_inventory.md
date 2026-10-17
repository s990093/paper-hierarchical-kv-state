# 預測器盤點：用什麼訊號、什麼 loss、哪一種最準（2026-10-17，第一版）

> **為什麼寫這份**：使用者 10/17 問「預測器與 loss 要參考 paper 的題目和消融，看哪樣可以最高：model 中間狀態？query？freq（2-hit／3-hit）？」，並說「可以盤點再決定」。
> **來源**：只用已經讀過、有卡片的整理：`research_20261006_eval/cards/E05_eviction_policy.md`、`E06_learned_eviction.md`（10/06–07，子 agent 讀原文、複核過）、`research/LIT_AND_TRACES.md`（10/17 文獻 agent）、`research_20261010_directions/D4_workload.md` §5、`PAPER_DELTAS.md` B5–B13（我們 9 月的 GBDT）。標記照原卡片：〔原文 p.X〕、〔摘要〕、〔未查證〕、〔判讀〕。
> **缺口**：§4 列的幾件事還沒查證，另開一個 agent 去補（結果寫在 `research/FEATURES_SIGNALS.md`）。**本文沒有任何第三階段的實驗結果。**

---

## 0. 一句話

**沒有一篇論文在同一個任務上，正面比過「中繼資料 vs 文字內容 vs 模型中間狀態」哪一種最能預測使用者會不會回來、多久回來。** 各篇只在自己的任務上做局部消融，數字不能互相比較。所以「哪樣最高」要自己量；而且有一個切法可以把它接回論文主線：

| 訊號 | 什麼時候拿得到 | 和論文主線的關係 |
|:--|:--|:--|
| 已閒置多久（LRU 用的） | 越晚越多，**只有延後才有** | 延後的決定天生多這一項 |
| 中繼資料：輪數、長度、工具名、請求類型、時段 | 寫入時就有，之後一直在 | 延後也拿得到 |
| 使用者歷史：過去的間隔、回來次數、頻率 | 越晚越多 | 延後也拿得到 |
| 文字內容的嵌入（LPC） | 文字一直在，隨時可以重算（小模型、便宜） | 延後也拿得到 |
| **服務模型的中間狀態**（hidden state、attention） | **只在那一輪前向的當下**；之後要嘛存下來，要嘛重做整段 prefill（很貴） | **唯一「寫入後就消失」的訊號**：論文 §6.2 的第二個缺口 |
| 工具執行進度（Ask-the-Tool） | 閒置期間才有 | 延後才有 |
| 下一輪的 query | **只有回來時（讀取時）才有** | 只能用在讀取時（例如決定還原哪些 block），不能拿來決定降級 |
| 回來時 GPU 多忙 | 讀取時才有 | Cake 已經在讀取時處理 |

所以「模型中間狀態有沒有用」不只是調特徵，它直接回答論文留下的開放問題：**有沒有一種只在寫入當下才有的資訊，值得在寫入時就量下來？** 要注意的是，就算它有用，做法也是「寫入時算一個分數存起來，降級時才用」，決定本身仍然可以延後〔判讀〕。

---

## 1. 訊號家族盤點（每種訊號誰用過、量到什麼）

### 1.1 不用學的：近期、頻率、規則

| 訊號 | 誰用 | 量到什麼 | 出處 |
|:--|:--|:--|:--|
| 近期（閒置時間） | LRU（vLLM、LMCache、Mooncake、SGLang HiCache 的預設） | Mooncake：LRU 比 LFU、LengthAware 好；原文說「不可能準確預測未來使用量」 | D4 §5〔原文 p.7–8，抽查〕 |
| **頻率：2-hit／3-hit 准入** | SGLang `write_through_selective`（hit ≥2 才寫 host）、Dynamo KVBM（≥2 才下放 SSD）、Marconi（第 2 次出現才收、第 3 次才命中） | When Fancy Eviction Fails：拿到第 2 次命中後，之後再被重用的機率從 69.4% 升到 88.7%，「頻率適合當准入、不適合排序」 | D4 §5〔原文 p.8，抽查〕；SGLang 程式碼〔子 agent 讀〕 |
| 頻率：LRU-K、LFU、TinyLFU、S3-FIFO、ARC | 經典快取 | D4 在 Mooncake 上當延後版對手；CPU ≥25% 工作集時 LRU 就在 oracle 的 2% 內 | D4 §5〔模擬〕 |
| 上一輪回答長度 | Bidaw（FAST'26） | 用來給下次重用距離的下界；**換成 Poisson 時間戳後就沒效** | D4 §5〔原文 p.12，抽查〕 |
| 請求類型×輪數的重用分布 | KVCache-in-the-wild WA（ATC'25，Alibaba） | 背景擬合每類的指數分布；比最好的基準只多 1.5–3.9% 命中率；沒和 Belady 比 | D4 §5〔原文 p.12，抽查〕；LIT B.3〔原文 p.11〕 |
| 工具名稱的歷史時長 | Continuum（經驗 CDF→TTL）、CacheWise（工具名＋TF-IDF 叢集→經驗存活曲線） | CacheWise 用真實工具時長的版本（CacheWise\*）表現相近 → 工具回來時間「猜得到」 | LIT B.3〔原文 p.5–8〕 |
| 已經過的時間當剩餘時長估計 | InferCept | 達到 oracle 時長的 93% | LIT B.3〔原文 p.7〕 |
| 執行中工具的進度 | Ask-the-Tool | 呼叫**之前**的估計排不出長呼叫的順序（負載一變，Kendall τ 只剩 −0.04／−0.07）；所以改讀進度 | LIT B.3〔原文 p.4 Fig.3c〕 |

### 1.2 學出來的：中繼資料特徵（GBDT／小 MLP）

| 誰 | 特徵 | 模型／loss | 量到什麼 | 出處 |
|:--|:--|:--|:--|:--|
| LRB（NSDI'20，CDN） | 32 個存取間隔（deltas）、10 個指數衰減計數器（EDC）、物件大小與類型 | LightGBM，log 下次時間，L2 | 有特徵累加、模型種類、8 種 loss、訓練集大小的消融（**具體數字本文沒抄，待補**） | E05〔原文 proc. p.533–535〕 |
| HALP（NSDI'23，YouTube） | 32 個間隔、10 個 EDC、次數、平均間隔、距上次存取、end-of-chunk | 1 層 MLP，兩兩比「誰先被存取」，BCE | 神經元超過 8 就沒有幫助；候選從 2 增到 4，byte miss 60.4%→59.3% | E05〔原文 proc. p.1158–1159〕 |
| LARU（2509.20979，LLM） | 10 個間隔、10 個 EDC、session 輪數、prompt 長度 | LightGBM，線上重訓 | P99 TTFT −6.0～−28.3%；沒有特徵消融 | E06〔原文 p.7〕 |
| **我們 9 月的 M5**（Mooncake toolagent，block 級） | static（位置、請求長度、位置差、請求間隔）／deltas／EDC | LightGBM | **只用 static 最好**：AUC 0.9998、成本加權錯誤 579.84 ms；全部 829.00 ms；只有 deltas 753.68；只有 EDC 762.64 → LRB 的頻率特徵在 KV 上沒貢獻 | PAPER_DELTAS B12〔實測〕 |
| Copilot 2026（arXiv 2608.00101） | 原文的特徵清單**待補** | LightGBM 12 個分位數 → S(t)；過了 t₀ 用 S(t)/S(t₀) 更新 | >60 s 閒置的 AUC 0.73，「只看上一次閒置」的規則 0.58；沒有端到端策略評估 | LIT B.3〔原文 p.17〕 |
| LAVA（VM 壽命，跨領域） | 以「已經活多久」為條件 | GBDT | C-index 0.84，tree-based Cox 0.78，線性 Cox 0.52 | LIT §0(4)、cards/P3_LAVA |

### 1.3 學出來的：文字內容

| 誰 | 輸入 | 模型 | 量到什麼 | 出處 |
|:--|:--|:--|:--|:--|
| **LPC**（NeurIPS'25） | 新 prompt＋前 4 輪 user prompt（512 token 預算），凍結的 multilingual-e5-small（118M）嵌入＋已發生輪數 | 3 層 MLP，加權 BCE：會不會有下一輪；乘上時間衰減 | MCC 0.28–0.36（Table 3 最大 0.3863）；**只補回 LRU→oracle 差距的 22–66%**；加入模型回應沒變好；有「只用輪數」的 LPC_turns 消融（**數字待補**） | E06〔原文 p.4–5、p.20–23〕 |

### 1.4 學出來的：服務模型的中間狀態

| 誰 | 輸入 | 預測什麼 | 量到什麼 | 出處 |
|:--|:--|:--|:--|:--|
| SAECache（2605.18825） | p.7–8 說是「服務中 LLM 最後一層、最後一個輸入 token 的 hidden state」；p.19–20 卻列手工特徵（open_ended_score、response_length），**前後矛盾** | 首輪會不會變多輪（3 層 MLP，約 1M 參數） | 整套的逐步消融在合成負載上：token 類型權重 +23%、log-normal 時序 +14%、位置衰減 +12%、佇列權重 +39%；**hidden state 這一塊單獨值多少沒有報** | E06〔原文 p.6–8、p.15–16、p.19–20〕 |
| PBKV（2605.06472） | prefix＋prefill embedding | 接下來 K 步會呼叫哪些 agent（CE） | 比 LRU 快 1.85×；激進預取在預測雜訊 ≥30% 時比不預取還差 | LIT B.3〔原文 p.1–6、p.18–19〕 |
| TierKV（lit_A） | prefill 的 hidden state | — | 原文說「prefill 在行動端是免費的預測器；伺服器上要另外跑預測器」 | `research_20261009_explore/lit_A_summary.md`〔原文 p.3〕 |
| 用 LLM 中間層預測輸出長度（例如 TRAIL）、learning-to-rank 排程等 | — | 輸出長度（不是回來時間） | **本 repo 還沒有卡片，待補** | — |

### 1.5 query 與注意力（不同的問題：哪些 token 重要）

| 誰 | 輸入 | 預測什麼 | 和第三階段的關係 |
|:--|:--|:--|:--|
| KVP（ICML'26） | (k, v, pos)，**不看 query** | 請求內哪些 token 可以逐出 | 有損、每個模型各訓一次；目標是 token 重要度，不是回來時間 |
| ForesightKV（ICML'26） | k、v＋注意力統計 | 同上（RL） | 同上 |
| LookaheadKV（ICLR'26） | 用 LoRA 產生的「預看 query」對 prompt key 的注意力 | 同上 | 同上 |
| TRIM-KV（ICLR'26） | 每層的 hidden x_t | 同上 | 同上 |

- 出處：E06〔原文 KVP p.5、FKV p.15、LKV p.5、TRIM p.5〕。四篇都是**單請求、有損**，沒有到達過程（E06 共同模式 2）。
- 對我們的意義〔判讀〕：**下一輪的 query 只有使用者回來時才知道**，所以 query 類訊號不能拿來決定「降級誰」。它能用的地方是讀取時：回來時已經看到新問題，可以決定哪些 block 要精確還原、哪些可以用低精度或略過。那是另一條有損的研究線，和第二階段的精度問題比較近，不是回來時間的預測器。

---

## 2. loss 盤點（細節在 `research/LIT_AND_TRACES.md` §0(2)、B.1–B.3）

| 類別 | 代表 | 「很晚回來」vs「不回來」 | 我們 9 月踩到的 |
|:--|:--|:--|:--|
| 回歸＋補值 | LRB（視窗外一律 2×視窗，刻意同分）、3L-Cache（刻意拉開）、GL-Cache | LRB 同分；3L-Cache 拉開 | B5：同分讓排序壞掉（AUC 0.9995、Spearman 0.233） |
| 成對／排序 | HALP（兩兩 BCE）、Parrot（可微 NDCG＋DAgger，on-policy 比 off-policy 平均多 9.8%） | HALP：一個先回來就有標籤 | — |
| 模仿 OPT | Glider／Hawkeye、Parrot、LFO | 標籤＝OPT 的決策 | — |
| 存活／分布 | LHD、LAVA、Copilot（分位數→S(t)）、CacheWise、Continuum、SAECache（log-normal） | 天然處理右設限 | — |
| 分類 | LPC（加權 BCE）、PBKV（CE） | 二元，不管多久 | B6：成本加權訓練輸給「普通訓練＋事後移門檻」 |
| 決策導向（SPO+） | 查不到用在快取或 KV 逐出的例子 | — | — |

---

## 3. 初步決定（等 §4 的缺口補完再定案）

1. **「哪樣最高」用一個特徵家族的消融來回答**，loss、模型族、決策介面都固定，一次只加一個家族：
   - F0 只看閒置時間（＝LRU）
   - F1 ＋中繼資料（輪數、長度、工具名、請求類型、時段）
   - F2 ＋使用者歷史（過去間隔、回來次數；FreeInference 的 `user_id`、WildChat 的 `hashed_ip`）
   - F3 ＋文字嵌入（凍結 e5-small，LPC 式）
   - F4 ＋服務模型中間狀態（Llama／Qwen 某幾層的 hidden state，線性 probe 或小 MLP；掃層）
   - query 不放進來（回來前拿不到）。
2. **資料限制**：F3、F4 要有文字，公開資料裡只有 **WildChat**（有文字、真實時間戳、hashed_ip，但對話短）。FreeInference 沒有文字，只能做 F0–F2。所以「中間狀態有沒有用」的結論只會在 WildChat 這類聊天資料上成立，要寫清楚範圍。
3. **頻率（2-hit／3-hit）**：當作准入規則放進對手（LRU＋2-hit 已在對手清單），也把「過去回來幾次」放進 F2；不另做頻率預測器。9 月 B12 已經顯示 LRB 式頻率特徵在 block 級沒貢獻；session 級還沒量過。
4. **loss**：照計畫書 §6 的清單，加上文獻 agent 建議的「以已閒置時間為條件的分布」（LAVA／Copilot 式）。主指標仍是決策 regret，不是 AUC。
5. **3090 多卡的真正用途在 F4**：要把 WildChat 的對話餵過 Llama／Qwen 抽 hidden state（只做推論、每張卡跑一份，不需要 tensor parallel），之後再訓小 probe。這是第三階段最吃 GPU 的部分。
6. **順序不變**：F1–F4 的**訓練**都要等 P3-3（oracle）判定之後（CLAUDE.md §8）。抽 hidden state 只是準備資料，但很吃共用的 GPU，所以等 §4 的查證顯示值得做、而且 oracle 判 GO 再開始。

---

## 4. 還沒查證、另開 agent 補的

| # | 缺什麼 | 為什麼要 |
|:--|:--|:--|
| G1 | 用 LLM 中間層 hidden state 預測系統層目標（輸出長度、會不會續聊、使用者滿意度）的論文：哪一層最好、和文字嵌入／中繼資料比差多少 | 決定 F4 要不要做、掃哪幾層 |
| G2 | LPC 的 LPC_turns（只用輪數）對完整版的數字；Copilot 預測器用了哪些特徵 | 量「內容比輪數多出多少」的前例 |
| G3 | LRB 特徵累加消融的數字；頻率准入（2-hit／3-hit、TinyLFU、LRU-K）在 LLM KV trace 上的比較 | 頻率有沒有用 |
| G4 | 對話系統／社群網路裡的「使用者會不會回到對話」預測（例如 conversation re-entry），用了什麼特徵、內容與使用者歷史各值多少 | 同一個問題在別的領域的答案 |
| G5 | 多輪情境下 query-aware vs query-agnostic 的 KV 選取（SCBench 等），以及「回來時依新 query 決定還原哪些 block」的前作 | query 能不能用在讀取時 |

---

## 5. 補查結果（10/17 晚，`research/FEATURES_SIGNALS.md`）與調整後的決定

**補查到的（各附頁碼見原檔；主 session 抽查了 LPC Table 3，PDF p.21 的數字一致）**：
- **沒有論文同時比過三種訊號。** 只有兩兩的比較：
  - 文字 vs 中間狀態，目標是**輸出長度**：服務模型自己的 hidden state 比外部文字模型好。OUTLETS 的 MAE 低 5.5–55%，EGTP 低 28–48%，TRAIL 低 5–19%。但 EGTP 的端到端表裡，MAE 較低的不一定決策較好。
  - 中繼資料 vs 文字，目標是**會不會續聊／回來**：結果依資料集而定。
    - LPC：只用輪數，MCC 在 LMSys 0.351、ShareGPT −0.001、Arena 0.078；完整版是 0.386、0.282、0.369。
    - Backstrom（WSDM'13）：文字接近隨機，結構與時間特徵勝出。
    - Zeng（ACL'19）：內容最有用，使用者的回來**頻率**沒用。
- **哪一層最好**：深度的中間 1/3–2/3 最好，第 0 層最差。TRAIL（Llama3-8B）是第 10–15 層；EGTP（Qwen2.5-0.5B）是第 12 層；所有層串接比任何單一層都好。
- **查不到任何論文用服務模型的 hidden state 預測回來時間或續聊**。SAECache 那段前後矛盾仍然無法確認：只有 v1、沒有程式碼。
- **時間型訊號**：凡是有時間特徵的研究，「已閒置多久、最近的間隔」都是最強或第二強的特徵。Copilot 的特徵重要度前三是平均閒置、輪數、上一次閒置〔讀圖〕。
- **query 在讀取時**：前作做了一半。IMPRESS（FAST'25）重用前綴時，依新 query 只從慢層載入重要的 token；EpiCache 依 query 從 CPU 抓一段。查不到有人在 session 回來時**同時**決定「還原哪些 block、用什麼精度、哪些重算」。這條線是有損的，接第二階段，不屬於回來時間預測器。

**調整後的消融設計**（取代 §3 第 1 點；仍然要等 P3-3 的 oracle 判定才訓練）：
1. F2 拆成兩個家族：F2t 時間型態（過去的間隔、平均閒置），F2c 次數／頻率（回來幾次、請求數）。理由是 X2 量到 Bailian 上頻率有用、時間型訊號沒有；Copilot 卻是平均閒置排第一。
2. 除了逐步累加，也做「F0＋單一家族」和「拿掉一個家族」。這樣就是第一個在同一任務上正面比較「中繼資料 vs 文字 vs 中間狀態」的結果。
3. F4 先做小規模 probe：一個模型，取約 1/4、1/3、1/2、2/3 深度與最後一層，再加全部層串接；平均 pooling 和最後一個 token 各一組。要 F0＋F3 明顯贏過 F0＋F2，才擴大。
4. F3 與 F4 要公平：用同一個 probe、同樣的 pooling；除了凍結的 e5-small，另加一個微調過的小文字模型。
5. 加一個 LPC_turns 式的下界：依輪數分箱、取多數類。
6. 至少 3 個 seed 並報 CI；主指標仍是決策 regret。
7. 頻率只當准入規則（2-hit，對手裡已有），不當排序特徵。理由是 LLM prefix cache 上用頻率排序（LFU、W-TinyLFU）會崩（Fancy-eviction p.4），X2 的 2Q 式規則則已經在對手集合裡。
8. **事前預期〔判讀〕**：F4 不會比 F3 好。理由是回來時間沒有前例，LPC 加入模型回應也沒有變好。但正因為沒有前例，F4 對論文 §6.2「寫入後就消失的資訊」是一個乾淨的測試；量出來「沒用」也是結果。
