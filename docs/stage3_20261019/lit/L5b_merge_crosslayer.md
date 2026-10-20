# L5b　KV 的 token 合併與跨層共用：文獻地圖＋可在 MI300X 上 2.5 天內做的最小實驗

> **日期**：2026-10-20（UTC）。**執行者**：文獻 agent（沒有開任何 GPU 行程、沒有 import torch／vllm；只寫這一個檔）。
> **範圍**：(a) token 合併（相鄰或相似的 token 合成一個 KV）；(b) 跨層共用（不同層的同一個 token 共用 KV，或共用低秩基底）。eviction（丟 token）已在 `L1`，量化、無損壓縮、卸載已在 `L2`，本檔只引用、不重寫。單層低秩（Palu、Eigen Attention、ShadowKV 的低秩 K 等）不在本檔範圍，只在需要對照時提一句。
> **查證方式**：
> - 每個 arXiv 編號都在 10/20 用 arXiv API（`export.arxiv.org/api/query?id_list=…`）抓標題、第一作者、arXiv comment（venue）、摘要。一個我記錯的編號（LoCoCo 先猜 2406.04636，實際是 2406.05317）就是這樣抓到的。
> - 全文：抓 `arxiv.org/html/<id>`，用標準函式庫轉成純文字後讀（CPU 工作都用 `nice -n 19 taskset -c 0-3`）。CaM 沒有 arXiv 版，抓 PMLR 的 PDF，用 tiara venv 的 pypdf 抽文字（不碰 torch）。
> - 程式碼：GitHub API 的 `license`、`languages`、`pushed_at`，加上 README；只讀、不裝。
> - 全文與 PDF 存在 `/mlsteam/data/tiara/papers/stage3_L5/`（`<arXiv id>.html`／`.txt`、`CaM_icml24_zhang24n.pdf`）。同一個目錄裡的 palu、kvtc、loki、lorc 等 PDF 不是本 agent 下載的。
> **證據等級**：〔全文〕＝讀了正文＋相關附錄；〔部分〕＝摘要＋從全文抓實驗設定、主要數字、限制的段落；〔摘要〕＝只讀摘要或會議頁；〔README〕／〔程式碼〕＝讀 repo；〔判讀〕＝我根據原文做的解讀；〔推論〕＝原文沒寫、我推出來的；〔算術〕＝我用原文或 repo 的公式算的。查不到寫 NOT_FOUND。
> **數字規則**：所有原文數字都是原作者在**自己的**模型、硬體、長度、預算下量的，**不能跨列比較或排名**。repo 自己的數字一律附出處。標〔預測〕的是事前預測，不是結果。
> **審查性質**：self-review，不是 cross-model（CLAUDE.md §5）。

---

## 0. 一頁結論

1. **在我們這兩個 GQA 模型、長 context 下，「相鄰層的 KV 很像」這個 MiniCache 的前提，第三方實測幾乎都說不成立**。
   - MiniCache 自己的觀察是在 LLaMA-3-70B（GQA）的短任務上量的；它的 LongBench 實驗用 Llama-2（MHA）與 Mistral-7B，沒有 ≥32K〔全文〕。
   - xKV 在 **Llama-3.1-8B-Instruct** 上量到：相鄰層逐 token 的 cosine（value cache）「出奇地低」。把 MiniCache 照官方設定放進 RULER 64K，只壓 1.3 倍，平均就從 91.89 掉到 45.04。LongBench 上，Llama-3.1-8B 掉 12.57%，**Qwen2.5-7B-Instruct-1M 掉 26.91%**〔全文〕。
   - XQuant（2508.10395）在 Llama-3.1-8B 上看到：pre-RoPE K 與 V 相鄰層差很多，只有層的輸入 X 很像〔部分〕。
   - 這些都不是我們自己量的，而且 xKV 是 MiniCache 的競爭者，MiniCache 是它重新實作的〔判讀〕。**所以 §4 的現象量測值得做，而且不用 GPU**：
     - 資料：S1 的 KV 匯出（`queue_q/10_s1_kvdump.sh`，已排）就夠。
     - 工具：repo 已有 `code/m16_kvstruct.py`（K1，未 commit），只要補 CKA、跨層串接秩、中心化 cosine、量化碼一致率。
2. **跨層的冗餘在「子空間」，不在「逐 token 向量」**：xKV 用 CKA 顯示，相鄰層的主奇異向量高度對齊。把 4 層串起來做 SVD：
   - Llama-3.1-8B RULER 64K：8.03 倍壓縮時 88.50，比 KIVI-2（7.1 倍）的 86.87 高。
   - Qwen2.5 1M 系列：xKV-4 是 90.19，KIVI-2 是 90.45，打平。
   - 再加 4-bit RTN：25.6 倍，平均 88.85 → 87.64〔全文〕。
   - 它是本檔唯一在我們兩個模型、≥32K 上，和量化、丟 token 都比過的跨層方法。需要做 SVD（線上，不用訓練、不用校準資料）。
3. **token 合併在長 context、GQA 上的證據很薄，而且邊際效益小**：
   - 多數合併方法只在 ≤16K（LongBench 截斷）或 MHA 上測。
   - 第三方 GRKV（EMNLP 2026）在不給問題的設定下看到：把「丟掉的 token 合進留下的」加在 eviction 上，Llama-3.1-8B 的 LongBench 只多 0.6 分、RULER 16K 只多 0.7–1.7 分；CaM、D2O、AsymKV 常常反而變差〔部分〕。
   - SelKV 在 Llama-3.1-8B 上：盲目合併（39.01）**比只丟不合（39.69）還差**，要加相似度閘門與 attention 補償，才到 39.81〔部分〕。
   - **Qwen2.5-7B-1M 上的 token 合併：NOT_FOUND。**
4. **最便宜、最乾淨的同位元組測試，可以直接塞進已寫好的 S2**（`code/m16_s2_equalbytes.py`，只動載入區 block 32–63）：
   - 合併被寫成新的「格式」：encode→decode 後，用既有的 `load_fmt`／`place` 放回去，不改 attention kernel。
   - 關鍵是 AsymKV 證明的等價式：把一組 g 個相鄰 token 的 key 都換成同一個合併後的 key、value 保持原樣，attention 輸出**完全等於**「只存 1 個 key＋value 總和＋計數」的壓縮 cache〔全文式 (8)–(10)；推論：所以可在 cache 裡原地模擬〕。
   - 固定 g＝2／4／8 剛好是 S2 的 1/2、1/4、1/8 預算。
   - 跨層則用「深半段兩兩合併（MiniCache 式）＋INT4」對「INT4＋丟 25% block」與「K4V2」，三者都約 0.24–0.25 倍位元組。
   - 估計約 0.6 GPU 小時（32K）＋約 1 GPU 小時（128K 切片）（§8）。
5. **對分層的意義**：
   - 合併與跨層共用**只在載入區有意義**（Cake 的前段反正會被精確重算，和 L1 G1 同理）。
   - 它們都**保留 block 結構**：每個 512-token block 等比例變小。這一點和逐 head 丟 token 不同（L1 §3.2），所以可以直接換成「少搬的位元組」。
   - 還原成本：token 合併只要在 GPU 上「展開」（重複或加 log 計數的 bias）；MiniCache 是逐元素縮放；xKV 每層要做一次 GEMM（A·B）再補回 RoPE。
   - 以上還原成本在 GPU 忙時會變多少，**沒有任何前人量過**。我們第二階段已量到 FP8 反量化在 GPU 忙時慢 2.4–3.8 倍（research_map 第 3 點）。
6. **「合併比丟掉在多輪更保住資訊」目前只有間接證據**：
   - xKV 的多輪 NIAH（Llama-3.1-8B，64K，8 倍）：SnapKV、PyramidKV 第 1 輪後陡降，xKV 各輪穩定。原文只有圖與文字，沒有逐輪數字〔全文〕。
   - D2O 在 MT-bench 上只有 4 個樣本的質性比較〔部分〕。
   - **token 合併在多輪上的定量比較：NOT_FOUND。**
   - 理由面〔判讀〕：固定分組的合併與跨層共用本來就和問題無關，不會有 L1 記錄的「依第 1 輪問題挑、第 2 輪就壞」（SCBench、KVzip、RocketKV-MT、EpiCache；L1 §1、§4 G4）。但這只代表「不會因為挑錯而壞」，不代表「同位元組下比量化好」。
7. **事前門檻（§7 寫死）**：
   - 第二階段已知 INT4 KIVI 在 32K–128K 幾乎不傷（RUNLOG_MI300X 15.9）。Llama 32K 連整份 INT2 KIVI 也只掉約 1 個百分點（RUNLOG_MI300X 11.2h）。
   - 所以合併只有在 **≤0.19 倍位元組（INT2 以下）且 64K–128K** 才可能有空間。
   - 若在任何預算下，合併與跨層都沒有落在量化前緣（FP8→INT4→K4V2→INT2）**之上**超過 ε＝2 個百分點（配對 CI 支持），就照 CLAUDE.md 規則 4 的精神收掉，寫成負面結果：「在本設定下，合併與跨層共用被量化支配」。
8. **要訓練或改架構的**（CLA、YOCO、LCKV、MLKV、A Systematic Study、SwiftKV、FusedKV、Stochastic KV Routing、LISA、UniAttn、LoCoCo 的卷積頭）本次都不可行（§3.3）。可以引用的觀察：
   - 「2 倍以內的層共用，多數組態不太掉」（A Systematic Study，1.1B 從頭訓練）。
   - FusedKV：上層的 V 主要來自底層，K 來自底層＋中層（自己訓練的模型）。

---

## 1. 和既有文獻檔的關係（引用、不重寫）

| 主題 | 已在哪 | 本檔只補 |
|:--|:--|:--|
| eviction、query-aware／agnostic、多輪會壞（SCBench、KVzip、RocketKV-MT、EpiCache、Online compaction、FlowKV） | `L1` §1、§3.3、§4 G4 | 合併與跨層在多輪的證據（§6.4） |
| 「邏輯上少看 ≠ 實體上少搬」（kvpress 假 key 遮罩、vToken、Tangram） | `L1` §3.2 | 合併與跨層是否保留 block 結構（§6.1） |
| 有損量化、無損壓縮、卸載、預取、GPU 忙時的干擾 | `L2` §0–§3；research_map 第 3、6 點 | 合併與跨層的還原成本放在同一個框架（§6.2） |
| ShadowKV（低秩 K＋V 卸載）、InfiniGen（SVD 扭轉） | `L1` C5、`L2` D1–D2 | xKV 的 xK-SR 就是「把 ShadowKV 的單層 SVD 換成跨層 SVD」（§3.2 xKV 卡） |
| HCache（從層輸入重算 KV） | `L2` §3 | XQuant-CL（快取層輸入 X 的跨層差值，再重算 KV）（§3.2） |

---

## 2. 總表

### 2.1 token 合併（同一層內）

「≥32K」欄只算**品質**實驗，不算速度。

| # | 方法（venue） | arXiv | 合併什麼／怎麼合 | 訓練 | GQA 實測 | ≥32K 品質 | 程式碼（授權） | 深度 |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| M1 | **KVMerger**（venue 未查證） | 2407.08454 | 相鄰、key cosine > 0.75 的連續 token 成一組；以累積 attention 最高者為樞紐，Gaussian kernel 加權 | 否 | Mistral-7B-Instruct（v1.0） | 否（約 4K） | NOT_FOUND | 〔全文〕 |
| M2 | CaM（ICML 2024） | 無（PMLR v235） | 要被丟的 **value** 依 attention 比例抽樣合進留下的；key 照丟 | 否 | 否（LLaMA-1、OPT、GPT-NeoX） | 否 | zyxxmu/cam（無授權） | 〔部分〕 |
| M3 | D2O（ICLR 2025） | 2406.13035 | 被丟的 token 若和留下的最近鄰 key cosine 超過 EMA 門檻，就加權合進去；逐層動態預算 | 否 | Llama-3-8B、Mistral-7B-v0.2 | **是**：NIAH 50K／100K（Llama-2-7B-80k，MHA） | AIoT-MLSys-Lab/D2O（Apache-2.0） | 〔部分〕 |
| M4 | KeepKV（AAAI 2026） | 2504.09936 | 合併時記錄票數（合併歷史）、調整權重，使單步 attention 輸出不變 | 否 | Llama-3-8B、Mistral-7B | NOT_FOUND | NOT_FOUND | 〔部分〕 |
| M5 | AsymKV（NeurIPS 2025） | 2506.05410 | 相鄰 key 換成同一個（Fisher 加權），value 存總和＋計數（**無損**） | 否 | Llama-3.1-8B、Mistral-v0.3、Qwen2-7B | LongBench v2（8K–2M；cache 上限 2048） | the-scale-lab/AsymKV（Apache-2.0） | 〔部分〕 |
| M6 | GRKV（EMNLP 2026） | 2605.31105 | 用 ridge regression 把被丟 token 的資訊分到留下的 token，最小化 attention 輸出差 | 否 | Llama-3.1-8B、Mistral-v0.3、Qwen3-14B | RULER 16K（品質）；16K–96K（只量效率） | pjunjie/GRKV（Apache-2.0） | 〔部分〕 |
| M7 | SelKV（venue 未查證） | 2607.16213 | value cosine 軟閘門決定合或丟＋解碼時加 attention 比例的 logit bias | 否 | Llama-3.1-8B、Gemma-2-9B | 否（GQA 模型只到 3.5K；LongChat 31.5K） | NOT_FOUND | 〔部分〕 |
| M8 | EMS（venue 未查證） | 2412.08521 | 逐 head「先丟、再合」；Global-Local 分數 | 否 | Llama-3-8B、Mistral-7B-v0.2 | NIAH ≤32K（Mistral） | NOT_FOUND | 〔部分〕 |
| M9 | ZSMerge（venue 未查證） | 2503.10714 | head 級預算＋殘差合併＋補償的 attention 分數 | 否 | Mistral-7B；延伸到 Llama-3.1-8B | InfiniteBench（100K+）；54K 吞吐 | SusCom-Lab/ZSMerge（MIT） | 〔部分〕 |
| M10 | SpindleKV（ACL 2025） | 2507.06517 | 淺層用 codebook（相似向量合成一個）、深層用 attention eviction | 否 | LLaMA-3-8B、Mistral-7B | NOT_FOUND | tyxqc/SpindleKV（MIT） | 〔部分〕 |
| M11 | LoCoCo（ICML 2024） | 2406.05317 | 固定大小 cache，用 1-D 卷積核算混合權重，把新 token 融進 cache 槽 | **要**（卷積頭用少量資料訓練） | 否（Llama-2） | 微調到 32K | VITA-Group/LoCoCo（無授權） | 〔部分〕 |

### 2.2 跨層共用（不用訓練，或只要少量校準）

| # | 方法（venue） | arXiv | 共用什麼 | 校準 | GQA 實測（我們的模型？） | ≥32K 品質 | 程式碼（授權） | 深度 |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| X1 | **MiniCache**（NeurIPS 2024） | 2405.14366 | 深半段相鄰兩層、同位置 token：SLERP 合方向＋各自保留範數＋保留 γ 比例最不像的 token | 否 | LLaMA-3-8B／70B、Mistral、Mixtral（**沒有** Llama-3.1／Qwen2.5） | 否 | **沒有程式碼**（repo 只有專案頁）；xKV repo 有重做的版本 | 〔全文〕 |
| X2 | **KVSharer**（venue 未查證；arXiv comment：「Under Review by ICLR2025」） | 2410.18517 | 整層 KV 直接拿別層的來用；**挑最不像的層對** | 30 句×64 token（Wikipedia），搜尋約 1 分鐘 | InternLM2（GQA〔判讀〕）、Mistral-7B-v0.3 | 否（PPL 用 2048） | yangyifei729/KVSharer（無授權） | 〔全文〕 |
| X3 | **xKV**（ICML 2026） | 2503.18893 | 每 4 層把 KV 串起來做 SVD：共用 token 基底 A、各層自己的 B；K 用 pre-RoPE | 否（prefill 時線上 SVD） | **Llama-3.1-8B-Instruct、Qwen2.5-7B-Instruct-1M**、Qwen3-4B | **是**：RULER 64K、多輪 NIAH 64K | abdelfattah-lab/xKV（Apache-2.0） | 〔全文〕 |
| X4 | CommonKV（venue 未查證） | 2508.16134 | 相鄰層 **W_k／W_v** 用 SVD 共用 → latent KV 可合併；依 cosine 分配預算 | 2048×1024 token（Wikitext-2，Fisher） | Llama-3.1-8B、Mistral-v0.2 | RULER（長度：§3.2） | rommel2021/CommonKV（無授權） | 〔部分〕 |
| X5 | XQuant（EMNLP 2025） | 2510.11236 | 相鄰兩層共用 KIVI-2 的**量化碼**（value），各層保留自己的 scale／zero | 不用資料（data-free calibration） | Mistral-7B-v0.3／Instruct-v0.2 | Mistral 截到 30,000 token | brinenick511/XQuant（MIT） | 〔部分〕 |
| X6 | XQuant／XQuant-CL（venue 未查證） | 2508.10395 | 不存 KV，存層輸入 X（或 X 的跨層差值），用時重算 KV | 離線 SVD W_k／W_v（GQA 才需要） | Llama-3.1-8B、Mistral-7B-v0.3 | NOT_FOUND | NOT_FOUND | 〔部分〕 |
| X7 | ChunkKV 的逐層 index 重用（NeurIPS 2025） | 2502.00299 | 不共用 KV，共用「留哪些 token」的 index | 否 | LLaMA-3-8B、Qwen2-7B、Mistral-7B | LongBench、NIAH（>10K） | NVIDIA/kvpress | 〔部分〕 |
| X8 | DepthWeave-KV、FreqDepthKV（2026，venue 未查證） | 2607.06523、2607.06519 | 相鄰層共用低秩或低頻成分＋殘差 | DepthWeave 的 gate／router 要訓練 | **原文沒寫基底模型與硬體** | 聲稱 64K、32K | NOT_FOUND | 〔部分〕；**可信度低，不列為依據** |

---

## 3. 卡片

> 每張卡固定七欄：1 機制；2 訓練與校準；3 GQA 與實測模型；4 原文壓縮比與答對率、≥32K；5 還原成本、和量化或丟 token 的疊加或比較；6 卸載或搬運；7 程式碼、授權、ROCm。

### 3.1 token 合併

#### M1　KVMerger：Model Tells You Where to Merge〔全文〕
- **連結**：arXiv 2407.08454（Wang, Zheng 等，2024-07）。arXiv comment 沒寫 venue，搜尋也查不到 → venue 未查證。
- **1 機制**：
  - **觀察**：同一序列裡 key 在 token 層級很像，而且有局部性，越靠近對角線越像，部分相鄰 token >0.9。value 沒有這個性質。
  - 原文用 RoPE 的引理解釋 key 為什麼有局部相似（Lemma 3.1–3.2）。
  - **選組**：受限的階層聚類，只合**連續**的 token；從最後一個 token 往前，相鄰 cosine > ε（實驗 0.75）就併進同一組。累積 attention 前 k 名（含 sink、heavy hitter）和最近的 token 不參與合併。
  - **合法**：每組以累積 attention 最大者為樞紐，其他 key 依 Gaussian kernel exp(−‖k_p−k_i‖²/2σ²) 加權合進樞紐。value 用同樣權重，再乘上組的大小。
  - **持續性觀察**：Llama2-7B-chat、LongBench 200 題，在固定門檻下各層的合併比例跨樣本、跨任務都很一致。第一、二層與最後幾層合得少（Fig. 3b）。
- **2 訓練**：不用。門檻 0.75、σ 取組內平均（約 5）。
- **3 GQA**：Mistral-7B-Instruct-v1.0（每個 KV head 配 4 個 query head）。GQA 下用 4 個 query head 平均出一張 attention map，來挑樞紐。
- **4 數字**（LongBench 9 個子集平均；**context 約 4K**，Llama2 原生 4K；NIAH 只到 1000–4500 token；A100 40/80GB）：

  | 模型 | Full | 50%：H2O／CaM／KVMerger | 35%：H2O／CaM／KVMerger |
  |:--|--:|:--|:--|
  | Llama2-7B-chat | 35.22 | 34.00／33.67／35.02 | 33.49／33.36／34.50 |
  | Llama2-13B-chat | 33.07 | 32.40／32.16／33.01 | 31.57／31.60／32.27 |
  | Mistral-7B-Instruct（GQA） | 38.44 | 29.92／32.72／35.79 | **13.48**／28.27／35.04 |

  另外，原文的「平均合併」初版比 H2O 差（Fig. 3c），作者說合併會引入雜訊，所以才改用 Gaussian 權重。≥32K：**否**。
- **5 還原與疊加**：合併後就是一般 cache，不用還原。要用 attention 分數挑樞紐，所以需要 eager attention〔判讀〕。沒有和量化疊加或比較（原文只說「互補」）。
- **6 卸載**：無。
- **7 程式碼**：論文內沒有 repo 連結；GitHub 搜尋「KVMerger」沒有結果 → NOT_FOUND。要自己實作。

#### M2　CaM：Cache Merging〔部分〕
- **連結**：ICML 2024（PMLR v235，pp. 58840–58850）；`github.com/zyxxmu/cam`。
- **1 機制**：
  - 論點：只要丟就一定擾動輸出，壓得越多擾動越大。
  - 做法：把要被丟的 **value** 合進留下的 value，合併比例依「被丟位置 attention 的顯著程度」做 Bernoulli 抽樣。key 照樣丟。
  - 定位是外掛：可加在 StreamingLLM、H2O、Scissorhands 上。
- **2 訓練**：不用。
- **3 GQA**：否。LLaMA-7B／30B／65B、OPT-13B、GPT-NeoX-20B 都是 MHA。
- **4 數字**：20% 預算、短任務（OpenBookQA、PiQA、XSUM、CNN/DM 等）。例：LLaMA-7B＋H2O 在第一欄從 22.8 → 27.6（加 CaM）。≥32K：否。
  - 第三方：KVMerger 在 LongBench 上看到 CaM 不如 H2O，推測是因為 key 照樣被丟（M1 表）。GRKV 也看到 CaM 常讓平均變低（M6）。
- **5／6**：沒有疊加量化；沒有卸載。
- **7**：Python 23.7 萬位元組，**沒有授權檔**；ROCm 未驗證。

#### M3　D2O：Dynamic Discriminative Operations〔部分〕
- **連結**：arXiv 2406.13035（ICLR 2025）；`github.com/AIoT-MLSys-Lab/D2O`（Apache-2.0）。
- **1 機制**：
  - 層級：依淺層 attention 密、深層稀疏，用 inverse-variance softmax 動態分配每層預算。
  - token 級：被丟的 token 找留下 token 中最近鄰的 key。cosine 超過 EMA 門檻（β＝0.7）才加權合進去，相似度與權重 K、V 共用。
- **2 訓練**：不用。
- **3 GQA**：Llama-3-8B、Mistral-7B-Instruct-v0.2（GQA）；另有 Llama-2-7B／13B、Falcon-7B、Code-Llama-34B。
- **4 數字**：
  - LongBench 預算 ρ＝0.2。
  - **≥32K**：NIAH，**Llama-2-7B-80k（MHA）**：

    | 預算 | 長度 | Full | H2O | SnapKV | CaM | D2O |
    |:--|:--|--:|--:|--:|--:|--:|
    | 4096 | 50K | 97.88 | 79.84 | 83.55 | 82.66 | 91.27 |
    | 4096 | 100K | 94.46 | 69.81 | 76.22 | 78.22 | 87.74 |
    | 8192 | 50K | 97.88 | — | 86.63 | — | 94.48 |
    | 8192 | 100K | 94.46 | — | 80.42 | — | 91.88 |

  - 吞吐：Llama-3-8B、A100-80G、不卸載，2.34–3.04 倍。
  - 第三方 GRKV：在「留連續片段」的 eviction 上，D2O 常讓 RULER 變差，原因是過度合併（M6）。
- **5**：沒有和量化比。多輪只有 MT-bench 4 個樣本的質性對照：H2O、StreamingLLM 在第 2 個樣本後開始答非所問，D2O 沒有（附錄 A.4）〔部分〕。
- **6**：無。
- **7**：Python＋13.5K 位元組 C++；ROCm 未驗證。

#### M4　KeepKV〔部分〕
- **連結**：arXiv 2504.09936；**AAAI 2026**（Proceedings Vol. 40 No. 39，2026-03-14，已用 ojs.aaai.org 確認）。
- **1 機制**：
  - 「Electoral Votes」記錄每個合併槽代表幾個原 token。
  - 「ZIP-Merging」調整權重，使**當步**的 attention 輸出和合併前完全相同（單步無損）。
  - 多步時用歷史 attention 預測未來 attention，並給出擾動上界。
  - 候選用相似度挑（門檻 0.8），預算分配沿用 PyramidInfer。
- **2**：不用訓練。
- **3**：Llama-2-7B／13B、Llama-3-8B、Mistral-7B。
- **4**：LongBench（20% 壓縮比）；10% 預算時仍維持品質（摘要）；吞吐約 2 倍（Llama-2-7B、A100）。≥32K：NOT_FOUND。
- **5**：自承限制：需要 attention 分數，**和 FlashAttention 不相容**；合併有額外計算，某些情況不如單純丟（附錄 C.1）。
- **6 卸載**：原文提出「週期性無損」：**完整 KV 存在外部，定期把壓縮後的表示載入記憶體**〔部分〕。這是本次搜尋到唯一把合併和「另一層存完整 KV」放在一起的設計。但方向和 Tiara 相反：它是 HBM 放壓縮版、外部放完整版。
- **7**：論文內沒有 repo 連結 → NOT_FOUND。

#### M5　AsymKV：Homogeneous Keys, Heterogeneous Values〔部分〕
- **連結**：arXiv 2506.05410（NeurIPS 2025）；`github.com/the-scale-lab/AsymKV`（Apache-2.0）。
- **1 機制**：
  - **觀察**：相鄰 key 拿到相近的 attention（局部同質），相鄰 value 分布不同（異質）。
  - 做法：相鄰 key 換成同一個 k\*（用 Fisher 資訊近似 Hessian 推出的最佳合法）；value 存**總和** v_m＋v_{m+1}，另存一個計數 c。
  - attention 分母改成 Σ exp(q·k_j)·c_j（Locally Merged Attention），和「兩個槽都放 k\*、各自保留原 value」**數學上完全相等**（式 (8)–(10)）〔全文式〕。
  - 〔推論〕所以 value 的合併是無損的，所有損失都集中在「把 key 換成 k\*」。在 cache 裡原地模擬時，只要把整組的 key 都寫成 k\*、value 不動即可（§5 的實作就是用這個）。
- **2**：不用訓練。
- **3**：Llama2-7B-chat、**Llama3.1-8B-Instruct**、Mistral-7B-Instruct-v0.3、Qwen2-7B-Instruct（後三個是 GQA）。
- **4**：
  - LongBench：Llama3.1-8B 平均 43.95，H2O 38.89。設定：壓縮後 context 上限 2048 token、chunk 512；A100 80GB；單次執行，沒有誤差棒（NeurIPS checklist）。
  - LongBench v2：8K–2M。≥32K：有，但是在「壓到 2048」的極端預算下。
- **5 第三方**：GRKV 說 AsymKV 在「留連續片段」的 eviction 上常讓 LongBench 變差，而且在 RULER 16K 的延遲增加約 15 倍，所以沒放進 RULER 表〔部分〕。
- **6**：無。
- **7**：Python＋notebook；ROCm 未驗證。

#### M6　GRKV：Global Regression for KV Cache（第三方的合併評測）〔部分〕
- **連結**：arXiv 2605.31105（Accepted by EMNLP 2026 Main）；`github.com/pjunjie/GRKV`（Apache-2.0，2026-10-04 還有更新）。
- **1 機制**：在 eviction 之後，用 ridge regression 更新留下的 K、V，最小化「壓縮 cache vs 完整 cache」的 attention 輸出差。用 prompt 尾端 32 個 query 當未來 query 的代理（surrogate window），Woodbury 恆等式降維。
- **2**：不用訓練；每個 prompt 一步交替更新。
- **3**：Llama-3.1-8B-Instruct、Mistral-7B-Instruct-v0.3、Qwen3-14B。
- **4**（10% 預算；**壓縮時不給最後的問題**＝query-agnostic，比較接近多輪）：
  - LongBench：Llama-3.1-8B，SnapKV 33.96 → 34.58、CriticalKV 36.00 → 36.58；Mistral，SnapKV 33.12 → 33.75。
  - RULER 16K：Llama，SnapKV 27.44 → 29.09、CriticalKV 40.83 → 41.51。
  - 其他合併法（CaM、D2O、AsymKV）「常常變差」，尤其是檢索任務，原因是過度合併。
  - 效率：A6000，16K–96K。
- **對本題最重要的一點**〔判讀〕：在 Llama-3.1-8B、query-agnostic 下，「合併比丟」的邊際好處是 **+0.6（LongBench）到 +1.7（RULER 16K）分**，而且是這個方向寫得最仔細的方法。
- **5／6**：沒有和量化比；沒有卸載。
- **7**：Python，用 FlashAttention-2（我們的 venv 沒有 flash-attn，L1 §0 第 4 點）；ROCm 未驗證。

#### M7　SelKV〔部分〕
- **連結**：arXiv 2607.16213（2026；venue 未查證）。論文內沒有自己的 repo 連結 → NOT_FOUND。
- **1 機制**：
  - 依 value cosine 的軟閘門決定逐 token「全合、部分合、或丟」。
  - 解碼時依 prefill 的 attention 統計加 logit bias（α·log R_i，α＝0.5），修正「一個合併 token 只拿到一份 softmax 質量」的 attention sag。
  - GQA：同一個 KV head 的 query 分數先平均，再各自選 token。
- **2**：不用訓練。
- **3**：Llama-3.1-8B（GQA 4:1）、Gemma-2-9B（GQA）、LongChat-7B-32k（MHA）。
- **4**：
  - LongBench，25% 保留：Llama-3.1 為 39.81（比 full 低 0.73）。
  - **消融**：只丟 39.69、盲目合併 **39.01**、加閘門 39.57、再加補償 39.81。作者自己的解釋是 GQA 4:1 下 value 常常對不上。
  - GQA 模型的 context 只有 **3.5K**。10% 預算時 SnapKV 反而最好（「剩下能合的對象太少」）。
  - 效率：H100，100K token 時解碼快 3.3 倍。
- **5／6**：說和 KIVI 互補，但沒比；沒有卸載。
- **7**：NOT_FOUND。

#### M8–M10　EMS、ZSMerge、SpindleKV〔部分〕
- **EMS**（arXiv 2412.08521；venue 未查證；程式碼 NOT_FOUND）：
  - 機制：逐 head 先丟後合；「redundancy score」高於 τ（0.55）就合、低於就丟。
  - 模型與硬體：Llama-2-7B-Chat、Llama-3-8B-Instruct、LongChat-7B-32k、Mistral-7B-v0.2；單張 A100 40GB。
  - 數字：預算 256 時 LongBench 平均多 1.28 分以上；Mistral NIAH（最長 32K）用 <2% 預算維持 95%。
- **ZSMerge**（arXiv 2503.10714；venue 未查證；SusCom-Lab/ZSMerge，MIT）：
  - 機制：head 級預算＋殘差合併＋補償的 attention 分數。
  - 數字：LLaMA2-7B 20:1（保留 5%）品質相近；54K context 吞吐約 3 倍；也跑了 InfiniteBench（100K+）。
  - 環境：A800；另外延伸到 Llama-3.1-8B。
- **SpindleKV**（arXiv 2507.06517，ACL 2025 main；tyxqc/SpindleKV，MIT）：
  - 深層用 attention eviction，淺層用相似度學出的 codebook 取代（把很像的向量合成一個碼字）。
  - 明講處理了 attention 型 eviction 在 GQA 下的困境。
  - 模型：LLaMA2-7b、LLaMA3-8b、Mistral-7b；LongBench。
  - 摘要沒有單一數字 → NOT_EXTRACTED；≥32K：NOT_FOUND。
  - 〔判讀〕它的觀察「淺層 KV 的組成很像、深層靠 eviction」和 MiniCache 的「中後段相鄰層很像」方向相反，但兩者量的是不同的東西：層內 token 間 vs 跨層同一 token。

#### M11　LoCoCo〔部分〕（要訓練：本次不可行）
- arXiv 2406.05317（ICML 2024；VITA-Group/LoCoCo，無授權）。
- 機制：固定大小 cache，新 token 以 1-D 卷積核算出的權重融進 cache 槽。卷積參數要用少量資料「end-to-end 訓練」（原文 §3）。
- 數字：推論時把 3482 token 壓進 128 槽；微調時用 512 槽把 Llama-2 從 4K 延伸到 32K。
- **要訓練（雖然輕量）→ 本次不可行。**

### 3.2 跨層共用（不用訓練）

#### X1　MiniCache：KV Cache Compression in Depth Dimension〔全文〕
- **連結**：arXiv 2405.14366（v2，2024-09-07）。**NeurIPS 2024**：proceedings.neurips.cc 的 hash 頁標題相符，10/20 抓取確認。
- **1 機制**：
  - 從 S＝L/2 層開始，相鄰兩層（l−1, l）**同一位置 token** 的 K（V 同理）合成一個方向向量：
    - e＝SLERP(x^{l−1}/‖x^{l−1}‖, x^l/‖x^l‖; t)，t＝0.6，偏向後一層。
    - 每層各存自己的範數，還原時 x̂^l＝e·‖x^l‖/‖e‖。
  - **token 保留**：角距離 d＝Ω/π 的「最不像」的那些 token 對不合，原樣存兩份。γ＝0.05。
  - ⚠️ 原文式 (4) 字面是保留 d_i < d_min＋(d_max−d_min)·γ，也就是距離**最小**的那些。這和正文「保留最不像的」相反〔全文〕。若要重做，照正文語意實作，並在結果裡註明這個不一致。
  - 只能兩兩合併（作者自承 SLERP 只能合兩個向量）。
  - 附錄 C 的替代做法：F_max＝平均方向×兩者較大的範數，比 SLERP 便宜但較不準。
- **2**：不用訓練、不用校準。
- **3 GQA**：
  - 觀察（Fig. 1a，相鄰層角距離）在 **LLaMA-3-70B**（GQA）上量，任務是 COQA、GSM8K、TruthfulQA 的短 prompt。token 級的觀察（Fig. 2b）在 LLaMA-2-7B（MHA）上量。
  - 合併實驗有 LLaMA-3-8B／70B、Phi-3-Mini、Mixtral-8x7B（GSM8K、COQA、TruthfulQA）；LongBench 用 Llama-2-7B／13B-Chat、Mistral-7B（-Instruct）。
  - **沒有 Llama-3.1、沒有 Qwen2.5、沒有 ≥32K。**
- **4 數字**：
  - 記憶體公式：深半段兩兩合併，4brh(s+n) → 3brh(s+n)，即 KV **只省 25%**；加上範數與 5% 保留，約 (3.1h+2)br(s+n)。
  - **LongBench（MiniCache 疊在 KIVI-4 上，5.02 倍）vs KIVI-2（3.95 倍）**，7 個子集平均：

    | 模型 | Baseline | KIVI-2（3.95 倍） | MiniCache＋KIVI-4（5.02 倍） |
    |:--|--:|--:|--:|
    | Llama-2-7B-Chat | 36.41 | 31.51 | 35.44 |
    | Llama-2-13B-Chat | 32.71 | 32.42 | 32.61 |
    | Mistral-7B | 37.33 | 33.43 | 35.75 |
    | Mistral-7B-Instruct | 48.32 | 43.74 | 46.99 |

    〔判讀〕這是本次找到**唯一**一個「跨層合併＋4-bit」對「更低位元的量化」、而且合併方用更少位元組的直接比較，結論是前者好。但它是短 context、主要是 MHA，而且第三方在長 context 的 GQA 模型上重做時結論相反（見下方 xKV）。
  - **LLaMA-3-8B GSM8K**（表 E；x＝合併的層數）：0 層 0.488；16 層（深半段）0.477（平均法 0.343）；32 層 0.397（平均法 0.025）。
    - 0 層時 MiniCache（0.488）和平均法（0.467）的值不同，原文沒解釋。
    - 〔判讀〕在 GQA 的 Llama-3-8B、短 context 上，深半段合併只掉約 1 個百分點；但**單純平均就掉約 14 個百分點**，所以範數保留與 SLERP 是關鍵。
  - γ 消融（LLaMA-2-7B）：γ＝0 時 COQA 0.603、GSM8K 0.108；γ＝0.05 時 0.647、0.152；γ＝1 時 0.643、0.159。
  - 吞吐：ShareGPT（平均輸入 161、輸出 338 token）、LLaMA-2-7B、單張 A100-80GB、batch 128。記憶體少 41%，吞吐約 5 倍（對 FP16），比 KIVI-2 高 1.29 倍。
  - **第三方（xKV，Llama-3.1-8B-Instruct／Qwen2.5-7B-1M）**：
    - RULER 64K：只壓 1.3 倍就從 91.89 → 45.04（Qwen 1M 系列：93.36 → 13.78）。
    - LongBench：Llama 掉 12.57%、Qwen2.5-7B-1M 掉 26.91%。
    - CommonKV 也說 MiniCache「受固有的不相似所苦」〔部分〕。
- **5 還原**：
  - 每個 token：方向向量乘範數（逐元素），保留的 token 放回原位。成本與 FP8 反量化同級的記憶體操作〔推論〕。
  - 原文說和量化、稀疏正交：有疊 KIVI-4，附錄 A 有和 H2O 比。
- **6**：無卸載。
- **7**：
  - 官方 `github.com/AkideLiu/MiniCache`（BSD-2-Clause）只有專案頁（JavaScript／HTML／CSS），README 只有一行標題 →〔README〕**沒有程式碼**。
  - xKV repo 有重做的版本（`--layer_merge_impl slerp`），但那個環境要 flash-attn 2.7.4＋MInference（CUDA）。
  - 核心運算是十幾行 torch 逐元素運算，可以在我們的 KVModel 上重寫，ROCm 沒有障礙〔推論〕。

#### X2　KVSharer：Layer-Wise Dissimilar KV Cache Sharing〔全文〕
- **連結**：arXiv 2410.18517（v1，2024-10-24；comment「Under Review by ICLR2025」，之後的 venue 查不到 → **venue 未查證**）；`github.com/yangyifei729/KVSharer`（Python，**無授權**，README 寫「under construction」）。
- **1 機制**：
  - 在校準集上，算每層 KV（對樣本平均、攤平、K 與 V 再各自平均）兩兩之間的**歐氏距離**，由大到小排序。
  - 依序嘗試「讓靠近輸出的那層直接用靠近輸入那層的 KV」。若最後一層 hidden state 和原模型的 cosine < 門檻 T（0.5），就放棄這一對。直到共用的層數達到目標 C。
  - 推論時（prefill 與生成）照表複製 KV。被共用的層仍然算 Q、MLP，只是不存自己的 KV。
  - **反直覺的發現**：照「最像」排序來共用，PPL 常是照「最不像」的 2 倍以上（Fig. 6）。隨機共用：Llama2-13B 的 PPL 9.11 → 51.41（表 4）。
- **2 校準**：英文 Wikipedia 隨機 30 句、每句 64 token；搜尋約 1 分鐘以內（4×A100）。換 BookCorpus 幾乎不變（表 3）。
- **3 GQA**：Mistral-7B-Instruct-v0.3（GQA）、InternLM2-7B／20B（GQA〔判讀〕）；Llama2-7B／13B（MHA）。**沒有 Llama-3.x、Qwen2.5。**
- **4 數字**（OpenCompass 短任務；PPL 用 2048 token）：
  - 共用 25%：Llama2-7B 平均 45.58（佔 full 的 97.9%）、Llama2-13B 89.4%、InternLM2-7B 97.0%、**Mistral-7B 88.4%**。
  - 共用 12.5%：Mistral 96.0%。共用 37.5%：Mistral 78.8%。
  - 記憶體與速度：Llama2-13B、1024＋4096 token，記憶體降到 64%，生成快 1.53 倍；prefill 不變。
  - **≥32K：否。**
- **5**：和 H2O、PyramidInfer 疊加（各再壓 20%），PPL 只小幅上升；沒有和量化比。還原成本為零（就是指標重用）。
- **6**：無。
- **7**：依賴 torch 2.1.2、transformers ≥4.38，monkeypatch `LlamaForCausalLM`。和我們 venv 的 transformers 5.17 相容性 NOT_VERIFIED。但做法本身只是「某層 KV 換成另一層的」，在 KVModel 上重寫是幾行〔推論〕。
- 〔判讀〕KVSharer 的「相似度」是**層平均後的整體向量**，和 MiniCache 的逐 token 角距離是不同的量。它的反直覺結果最多說明「這種粗相似度不是好的共用指標」，不能拿來否定或支持逐 token 的相似。

#### X3　xKV：Cross-Layer SVD〔全文〕
- **連結**：arXiv 2503.18893（v2；arXiv comment 與 README 都寫 **ICML 2026**）；`github.com/abdelfattah-lab/xKV`（Apache-2.0，2026-07-07 釋出效率測試）。
- **1 機制**：
  - **觀察**：
    - Llama-3.1-8B-Instruct、RULER multi-value NIAH：相鄰層**逐 token cosine 低**（Fig. 2a，value cache）。
    - 但線性 CKA 高（Fig. 2b）。CKA＝中心化 Gram 矩陣的對齊；高 CKA 代表兩層的主左奇異向量對齊（附錄 A 證明）。
    - 把 W 層水平串接後，保留 95% 特徵值能量所需的秩比例，隨 W 增加而下降（Fig. 2c）。
    - 附錄 F：Llama-3.2-1B、Qwen3-4B、GPT-OSS-120B 也有同樣現象。
  - **CLF**：每 W 層（預設 4）把 pre-RoPE K（或 V）串成 [L, W·d]，做秩 r 的 SVD → 共用 token 基底 A＝UΣ（L×r）＋各層 B_ℓ（r×d）。K、V 的秩比 1:1.5。用 `torch.svd_lowrank`，另有自寫的隨機 SVD kernel（快 4 倍）。
  - **SR（Selective Reconstruction）**：解碼時用 ShadowKV 的 chunk landmark（post-RoPE K 的 chunk 平均），每個 KV head 選 top-k chunk，只重建那些列。
- **2**：不用訓練、不用校準資料，prefill 時線上做 SVD。
  - 開銷：Llama-3.1-8B、W＝4，64K 時 0.57 s（佔 prefill 5.72%），128K 時 0.80 s（2.50%）。
  - ⚠️ 正文寫「A6000、64K 佔 3.17%」，表 3 標題寫「A100」、數字 5.72%，原文自己不一致〔全文〕。
- **3 GQA**：**Llama-3.1-8B-Instruct（8 個 KV head）、Qwen2.5-7B-Instruct-1M（4 個 KV head）**、Qwen3-4B（8 個）；附錄還有 DeepSeek-Coder-V2-Lite（MLA）。**和我們的兩個模型完全相同。**
- **4 數字**（RULER，**64K**；壓縮比假設 64K context；新生成的 token 不壓）：

  | 方法 | 壓縮 | Llama-3.1-8B 平均 | Qwen2.5 1M 系列平均（表頭寫 14B） |
  |:--|--:|--:|--:|
  | Full | 1 | 91.89 | 93.36 |
  | KIVI-2（block 128，2.25 bit） | 7.1（Qwen 6.4） | 86.87 | 90.45 |
  | SnapKV | 8.0（Qwen 6.0） | 89.68 | 91.66 |
  | PyramidKV | 8.0（Qwen 6.0） | 89.94 | 91.17 |
  | 單層 SVD | 8.4（Qwen 6.35） | 45.71 | 71.79 |
  | MiniCache | 1.3 | 45.04 | 13.78 |
  | xKV（W＝4） | 8.03（Qwen 6.21） | **88.50** | 90.19 |

  - ⚠️ 表 1 的第二個模型標 Qwen2.5-**14B**-Instruct-1M，正文卻寫 Qwen2.5-**7B**-Instruct-1M、xKV 89.22%（差 full 2.6%）。原文自己不一致，引用時要註明〔全文〕。
  - 視窗消融（Llama，同壓縮比）：W＝1 45.71、W＝2 75.15、W＝4 88.50、W＝8 ≈ W＝4。
  - 只壓 K 或只壓 V：K 比 V 好壓；QA 子任務可以壓到 16 倍；VT、N-MQ 超過 8 倍就開始掉（D.3）。
  - **疊加量化**（D.5，Llama-3.1-8B RULER）：xKV 8 倍 88.85 → ＋4-bit RTN **25.6 倍 87.64** → ＋3-bit 32 倍 84.64。
  - LongBench：MiniCache 在 Llama 掉 12.57%、Qwen2.5-7B-1M 掉 26.91%。xKV 8.03 倍：Llama 42.27、Qwen 40.37，和 SnapKV、PyramidKV 接近。
  - **多輪**（Fig. 4，Llama-3.1-8B，64K，全部 8 倍）：SnapKV、PyramidKV 第 1 輪後陡降，xKV 各輪穩定。只有圖，正文沒有逐輪數字。
- **5 還原**：
  - 密集重建 X̂＝A·B_ℓ，每層 L·r·d FLOPs，K 還要再套 RoPE。原文說密集重建讓 attention 變 compute-bound，在 122K 時 kernel 只剩 0.4 倍速。
  - SR 只重建選中的列。xKV-SR（5.35 倍）在 A100 上端到端吞吐最多 4.23 倍。
  - 〔推論〕Tiara 的用法不同：還原時重建**一次**，之後在 HBM 放完整 BF16。所以 decode 時的 compute-bound 問題不存在，換成「還原時多一次 GEMM」。
- **6 卸載**：
  - xK-SR：K 用跨層 SVD 留 GPU，V 卸到 CPU（沿用 ShadowKV），重建和 V 的 PCIe 傳輸重疊。1.63 倍 KV 壓縮（考慮卸載後 GPU 記憶體省 8.9 倍）時 89.70，ShadowKV 87.17。
  - **沒有 GPU 忙時的量測**（和 L2 G2 同一個盲點）。
- **7**：Python＋C++／CUDA kernel。README 要 flash-attn 2.7.4、MInference（打 KIVI 的 Triton patch）。
  - 〔判讀〕整套直接在 ROCm 上跑：期限內不可行（我們的 venv 沒有 flash-attn，L1 §0）。
  - CLF 本身只用 `torch.svd_lowrank`（matmul＋QR），在 KVModel 上重寫約 40 行，ROCm 能跑〔推論，NOT_VERIFIED〕。

#### X4　CommonKV〔部分〕
- arXiv 2508.16134（venue 未查證）；rommel2021/CommonKV（無授權）。
- 機制：
  - 依「相鄰層 hidden state 很像」，用 SVD 讓相鄰層的 **W_k／W_v 參數共用**，得到容易合併的 latent KV。
  - 依 latent cache 的 cosine 動態分配每組的壓縮預算。
- **GQA 的難處（原文明講）**：GQA 的 d_kv＜d_hidden，跨層 SVD 後單層的 latent 反而要比原 cache 稍大才能保住品質。所以先把 K、V 串起來共用、再降秩，最後在 GQA 上做到每 4 層合併、0.5 壓縮比。
- 校準：Wikitext-2 2048 筆 × 1024 token，算 Fisher 資訊。
- 模型：Llama-3.1-8B-Instruct、Mistral-v0.2-7B；LongBench、RULER。
- 數字：說和量化、eviction 正交，三者合起來 98% 壓縮「沒有明顯掉分」（摘要）；8K 下和 xKV 比，線上 SVD 的延遲不可忽略。
- 〔判讀〕它要**改權重**（latent 投影），等於換掉 KV 的定義，不能直接插進我們「cache 原地替換」的 harness → 期限內不做。

#### X5　XQuant（EMNLP 2025）：跨層共用量化碼〔部分〕
- arXiv 2510.11236；brinenick511/XQuant（MIT，有 CUDA kernel）。
- **觀察**：Mistral-7B-Instruct-v0.2、LongBench 隨機樣本、KIVI-2 量化後：相鄰層同位置的 2-bit 碼，>80% 差值是 0 或 1，差 3 的 <5%。
- 機制：相鄰兩層（G＝2；從第 8 層起）共用同一份 value 的量化碼，各層保留自己的 scale、zero-point；再加 data-free 的 scale 校正。
- 數字：Mistral 的 LongBench 平均 41.98，等效 **1.38 bit**，「超過 KIVI-2bit」。
- 環境：RTX 3090；Mistral 截到 30,000 token、Llama-2 截到 8192。
- 〔判讀〕這是「跨層＋量化」最便宜的形式（不用 SVD、不用 SLERP），而且可以在 S1 的 dump 上用 CPU 先量「相鄰層 INT2 碼的一致率」（§4 的 P6）。

#### X6　XQuant／XQuant-CL（2508.10395）：快取 X、跨層差值、重算 KV〔部分〕
- venue 未查證（arXiv comment 只寫 24 pages）；程式碼 NOT_FOUND。
- 機制：
  - 不存 K、V，存層輸入 X（post-norm）的低位元量化，用時再乘 W_k、W_v 重算。
  - XQuant-CL 改存相鄰層 X 的**差值** ΔX，並維持一個累加器。
  - GQA 下 X（Llama-3.1-8B：4096 維）比 KV（2×1024）大，所以要離線對 W_k、W_v 做 SVD，把 X 降到潛在空間。
- **觀察（Fig. 3，Llama-3.1-8B，Wikitext-2 2K）**：相鄰層的 X 分布非常像，**pre-RoPE K 與 V 則明顯不同**。
- 數字：XQuant-CL 最多省 10 倍記憶體，PPL 只差 0.01；12.5 倍時差 0.1（摘要）。
- 和 Tiara 的關係〔判讀〕：這是「DROP＋重算」軸上的中間點，存 X、只重算投影，和 HCache（L2）同類。它也提供了「Llama-3.1-8B 上 K／V 跨層不像」的另一份第三方證據。

#### X7　ChunkKV 的逐層 index 重用〔部分〕
- arXiv 2502.00299（NeurIPS 2025）；程式碼在 NVIDIA/kvpress。
- 觀察：相鄰層「保留下來的 token index」的 Jaccard 相似度：

  | 模型 | H2O | SnapKV | ChunkKV |
  |:--|--:|--:|--:|
  | LLaMA-3-8B | 25.31% | 27.95% | 57.74% |
  | Qwen2-7B | 14.91% | 16.50% | 44.26% |
  | Mistral-7B | 15.15% | 15.78% | 52.16% |

- 做法：每 N 層重用同一組 index，壓縮時間少 20%，效能只掉 0.5%。
- 〔判讀〕這是「共用**選擇**而不是共用**數值**」。對 Tiara 的意義：如果要做選擇性還原（research_map 缺口 #2），逐層各自打分可以改成幾層共用一組 block 分數。但 token 級的 Jaccard 只有 15–28%（H2O、SnapKV），共用到 512-token block 粒度後會不會變高，要用 C1a 的分數另外看。

#### X8　DepthWeave-KV、FreqDepthKV（2026）〔部分〕→ 不列為依據
- arXiv 2607.06523、2607.06519（同一位第一作者 Anna Córdoba、同日提交）。
- 聲稱：64K／32K、8.3 倍／3.9 倍壓縮，品質接近 full。
- **但全文 grep 找不到任何基底模型名稱或 GPU 型號**；DepthWeave 正文說「不重訓基底」，但 gate、router 要訓練；沒有程式碼連結。
- 〔判讀〕可信度低，本檔不拿它們的數字當任何判斷的依據。

### 3.3 要訓練或改架構：本次不可行（列出可以引用的觀察）

| 方法（venue） | arXiv | 要做什麼 | 可引用的觀察 | 深度 |
|:--|:--|:--|:--|:--|
| CLA（venue 未查證） | 2405.12981 | 相鄰層共用 KV head，從頭訓練 1B／3B | 在 MQA 之上再省 2 倍、準確度幾乎不變 | 摘要 |
| YOCO（NeurIPS 2024） | 2405.05254 | decoder-decoder 架構，只快取一次 | prefill 可提前結束 | 摘要 |
| LCKV（ACL 2024） | 2405.10637 | 只算、只存少數層（頂層）的 KV；要訓練 | 吞吐最多 26 倍（原文條件） | 摘要 |
| MLKV（venue 未查證） | 2406.09297 | 多層共用 KV head；uptrain Pythia-160M | 比 MQA 再省到 6 倍 | 摘要 |
| A Systematic Study of Cross-Layer KV Sharing（NAACL 2025） | 2410.14442 | 統一框架；1.1B 從頭訓練（SlimPajama 100B token）；作者自承只做 1.1B | **KV 減 2 倍時多數組態不太掉**；要再減，「所有層的 Q 配上層的 KV」較好，但訓練與 prefill 更貴 | 部分 |
| SwiftKV（venue 未查證） | 2410.03960 | 後段層的 KV 由較早一層的輸出算；**蒸餾**（Llama-3.1-8B：680M token、8×H100 約 3 小時）；AcrossKV 跨層共用也要微調 | 後段層的輸入很像；prefill 計算少 25–50% | 部分 |
| FusedKV（ICLR 2026） | 2512.03870 | 上層 KV＝底層與中層 KV 的可學習融合；從頭預訓練 332M–4B | **上層的 V 主要來自底層，K 來自底層＋中層**；融合直接在 post-RoPE K 上做 | 部分 |
| Stochastic KV Routing（venue 未查證） | 2604.22782 | 訓練或微調時隨機讓層用前面層的 KV（R-CLA） | 作者說 MiniCache 這類事後方法只能共用到有限程度，因為模型是用逐層 KV 訓練的 | 部分 |
| LISA（TACL） | 2408.01890 | 用小 FFN 對齊相鄰層的 head＋低秩補差；輕量訓練 | 多數層的 attention 型態很像；**淺層對小偏差很敏感**；不先重排 head 直接共用無效 | 摘要 |
| UniAttn（venue 未查證） | 2502.00439 | 後訓練時統一跨 block 的 softmax | — | 摘要 |
| LoCoCo（ICML 2024） | 2406.05317 | 卷積頭要訓練 | 見 M11 | 部分 |

其他只列不寫卡：
- CHAI（2403.08058）：runtime 合併相關的 head，不用微調。KV 記憶體少 21.4%，只測 MHA。
- SepLLM（2412.12094，ICML 2025）：把一段壓進分隔符，有不用訓練的版本。
- Palu（2407.21118）：單層低秩投影，xKV 的前作。

---

## 4. 現象：在真實 KV dump 上便宜量得到的東西

**資料來源**：S1 的匯出 job `stage3/jobs/queue_q/10_s1_kvdump.sh`（已排在第三階段 GPU 窗口第一個）。
- 存 Llama／Qwen × T4（HotpotQA 段落，自然文字）／T2（RULER 重複句子）× 64 block（32K）全層，外加 128K（252 block）的 5 層。
- 格式：`kv_bf16.u16`，每個 block 是 [L', 2, H, 512, D]。K 是 KVModel 快取裡的 **post-RoPE** K（`m7_model.py` 先 `apply_rope` 再存）。
- **量測本身 0 GPU 小時**，只用 numpy。T2 與 T4 要分開報，S1 自己也這樣規定：重複文字可能特別像。

### 4.1 原文用的指標 → 我們照著量

| # | 量什麼 | 原文指標（出處） | 我們怎麼量（照原文＋必要的補充） | 需要 pre-RoPE K？ |
|:--|:--|:--|:--|:--|
| P1 | 相鄰層、同 token 的相似度 | MiniCache：角距離 d＝Ω/π＝arccos(cos)/π，逐層對平均（Fig. 1a）、逐 token（Fig. 2b）；xKV：逐 token cosine 對 token 平均（Fig. 2a，只畫 V） | 每個層對 (l, l+1)、K 和 V 分開：(i) 每個（token, KV head）的 128 維 cosine；(ii) 每個 token 把所有 KV head 串起來的 cosine（MiniCache 的 h 維）。報中位數、p10、p90，以及 d 的分布。另報**中心化版**：先減去每（層, head）對 token 的平均向量（K 有固定的大通道，可能讓未中心化的 cosine 偏高〔推論〕） | **不需要**：同一 token 在兩層套的是同一個 RoPE 旋轉（正交），cosine 不變〔推論〕 |
| P2 | 跨層子空間對齊 | xKV：線性 CKA（Fig. 2b） | 每個層對、K 和 V 分開，在載入區（block 32–63，16K token）取樣 4K token，用 d×d 的寫法：‖YᵀX‖_F² /(‖XᵀX‖_F‖YᵀY‖_F)，先中心化 | K 要（xKV 用 pre-RoPE K） |
| P3 | 跨層串接的秩 | xKV：保留 95% 特徵值能量所需的秩比例 vs W＝1, 2, 4（Fig. 2c） | 同上的 token；W＝1, 2, 4（W＝8 的 8192² 特徵分解較貴，可省略） | K 要 |
| P4 | 相鄰 token 的相似度 | KVMerger：key 的 token 間 cosine 圖、相鄰 token cosine（Fig. 2–3a）；門檻 0.75 下的逐層合併比例（Fig. 3b）；AsymKV：key 局部同質、value 異質 | 每層、每 KV head：cos(k_t, k_{t+1})、cos(v_t, v_{t+1})，post-RoPE（快取裡存的，KVMerger 合的就是這個）與 pre-RoPE 都算；> 0.75 的比例；照 KVMerger 演算法 1 算的逐層「可合併比例」，跨題目、T2 vs T4 看是否持續 | 兩種都算（KVMerger 認為局部相似和 RoPE 有關） |
| P5 | 共用選擇 | ChunkKV：相鄰層保留 index 的 Jaccard | 需要 attention 分數，dump 裡沒有 → 不在這裡做；交給 C1a 的分數輸出〔判讀〕 | — |
| P6 | 量化碼跨層一致 | XQuant（EMNLP'25）：相鄰層 KIVI-2 碼的差值分布（0／1／2／3） | 用 BF16 在 CPU 上照 `m10_codec_x` 的 int2_kivi 規則重算碼（或用 S1 已存的 int4_kivi 位元組），算相鄰層同位置碼相同、差 1 的比例 | 否（照快取的 post-RoPE） |

**pre-RoPE K 怎麼來**〔推論〕：
- 用位置 t 的反旋轉 R(−t) 作用在 post-RoPE K 上。除了 BF16 捨入，這是精確的。
- 要用和 KVModel 相同的 inv_freq：Llama-3.1 有 `rope_scaling: llama3`；Qwen2.5-7B-1M 有自己的 rope_theta。要從模型的 `config.json` 照 HF 的公式重算，**不能**直接用預設的 10000。
- 單位驗算（CLAUDE.md 規則 6 的精神）：反旋轉後再正旋轉，必須逐位回到原值（容許 BF16 誤差）；也要看 pre-RoPE K 的頻譜集中度是否明顯高於 post-RoPE（ShadowKV、Palu 的已知性質）。

**repo 裡已經有的（10/20 寫這段時看到，是別的 session 寫的、尚未 commit，本 agent 沒有改它）**：`code/m16_kvstruct.py`（K1）。
- 已涵蓋：
  - 本表的 P1 未中心化版（逐 head 與整層串接兩種）；
  - P4 的 K_post／K_pre／V 相鄰 token cosine 與門檻比例；
  - 單層低秩；
  - RoPE 逆旋轉，含探針核對與資料自身核對。`m16_s1_kvdump.py` 也在 10/20 加了 `rope_probe` 等 meta。
- **還沒有的**：
  - P1 的中心化版；
  - P2 跨層 CKA；
  - P3「W 層串接」的聯合秩（xKV 的 Fig. 2c；K1 只有單層與單層內 head 串接）；
  - P4 照 KVMerger 演算法 1 的「可合併比例」；
  - P6 INT2 碼的跨層一致率。
- 這幾項加在 K1 上即可，不必另寫一支（§8 的 E-L5a）。

### 4.2 回答「中後段相鄰層很像」在 GQA 上原文有沒有驗證

| 來源 | 模型（GQA？） | context | 結論 | 深度 |
|:--|:--|:--|:--|:--|
| MiniCache Fig. 1a | LLaMA-3-70B（GQA） | 短（COQA、GSM8K、TruthfulQA） | 淺層不像、中後段很像（角距離） | 全文 |
| MiniCache 表 E | LLaMA-3-8B（GQA） | 短（GSM8K） | 深半段合併只掉約 1 個百分點；單純平均掉約 14 個百分點 | 全文 |
| xKV Fig. 2a、表 1、表 6 | **Llama-3.1-8B-Instruct**、**Qwen2.5-7B-1M** | RULER 64K、LongBench | 逐 token cosine 低；MiniCache 1.3 倍就崩 | 全文 |
| XQuant Fig. 3 | Llama-3.1-8B | 2K | pre-RoPE K、V 跨層明顯不同；X 很像 | 部分 |
| CommonKV | Llama-3.1-8B、Mistral | LongBench、RULER | MiniCache 受「固有的不相似」所苦 | 部分 |
| XQuant（EMNLP'25） | Mistral-7B-v0.2（GQA） | LongBench | 量化**碼**相鄰層 >80% 只差 0 或 1 | 部分 |

〔判讀〕
- 原文（MiniCache）**有**在 GQA 模型上驗證，但只在短 context。
- 長 context 與我們這兩個模型上，第三方一致說「逐 token 不夠像」。
- 「量化後的碼很像」（XQuant）和「逐 token 浮點向量不像」（xKV）並不矛盾：粗量化會把小差異抹平。
- P1 與 P6 一起量，可以直接回答這兩件事在我們的 dump 上哪個成立。

### 4.3 CPU 成本〔估計〕
- Llama 32K 全層 BF16 約 4 GiB，Qwen 約 1.75 GiB〔算術：32×2×8×32768×128×2 B；28×2×4×…〕。
- P1、P4、P6 是逐元素運算。P2、P3 是 4K×1024（Llama）的矩陣乘與 4096² 的對稱特徵分解。
- 合計約 30–60 分鐘（4 核、nice 19）。在第二階段量時間時也可以跑，但要照 experiment_matrix §6：≤4 核、nice 19，NFS 大量讀取避開 44e3。

---

## 5. 同位元組品質測試：哪一種、最簡單怎麼做

### 5.1 選哪幾種（1–2 GPU 小時、不用訓練）

| 候選 | 能不能在 1–2 GPU 小時內、和 S2 同一條路徑比 | 理由 |
|:--|:--|:--|
| **TM-g：固定 g 個相鄰 token 合成 1 個**（AsymKV 等價式；key＝post-RoPE 平均） | **能，最簡單** | 每個 block 內原地改 K 就好；g＝2／4／8 剛好是 1/2、1/4、1/8；保留 block 結構；和問題無關（多輪不會因挑錯而壞） |
| **XL-slerp：MiniCache 式跨層**（深半段兩兩合併、t＝0.6、保留 5%） | **能** | 每個 block 內跨層改 K、V；BF16 時約 0.78 倍，疊 INT4 約 0.24 倍，可以對 K4V2（0.25）與「INT4＋丟 25%」（0.234） |
| XL-copy：KVSharer 式「直接用前一層」（不搜尋，固定深半段） | 能（當下界） | 最便宜的跨層；KVSharer 的搜尋要 30 句校準，可省略 |
| **xKV-lite：W＝4 跨層 SVD** | 能，但程式較多（約 40 行；要反旋轉 RoPE 再轉回） | 在我們兩個模型上，唯一有 64K 證據贏或平 KIVI-2 的；1/8 預算是決勝點 |
| KVMerger／D2O／KeepKV／GRKV／SelKV | 期限內不建議 | 都要 attention 分數或 eager attention、變長分組（每個 head 不同 → 位元組不能直接換成搬運量，L1 §3.2）；而且第三方說邊際好處小（M6、M7） |
| CommonKV、SwiftKV、CLA 等 | 不行 | 要改權重或訓練 |

### 5.2 最簡單的實作：當成 S2 的新「格式」〔推論；只寫設計，沒有改任何程式〕

S2（`m16_s2_equalbytes.py`）的量化組已經是「對每個載入區 block 做 encode→decode，用 `Item.load_fmt`／`place` 放回 KV 緩衝」（`m10_sweep.py:128–150`）。合併只要提供「decode 後的 block」與「實際位元組」：

1. **TM-g**（每個 block [L, 2, H, 512, D]）：
   - 對每個（層, KV head），把 512 個 token 切成 512/g 組。K\*＝組內 post-RoPE K 的平均，**整組的 K 都寫成 K\***，V 不動。
   - 依 AsymKV 式 (8)–(10)：這和「存 1 個 key＋value 總和＋計數 g」的壓縮 cache，attention 輸出完全相同。所以不用改 attention、不用加 bias。
   - 位元組＝BF16 的 1/g；固定 g 不需要額外中繼資料。
   - 可選第二版：pre-RoPE 平均後，轉到組中心位置。P4 若顯示 pre-RoPE 相鄰 token 明顯比 post-RoPE 像，才加。
2. **XL-slerp**（同一個 block）：
   - 深半段（Llama 第 16–31 層 → 8 對；Qwen 第 14–27 層 → 7 對），對每個（token, KV head）的 128 維向量做 MiniCache 式 (2)–(3)：
     - 方向 e＝SLERP(t＝0.6)，兩層各保留自己的範數。
     - 角距離最大的 5% token 不合，原樣保留（照正文語意，見 X1 的 ⚠️）。
   - 位元組〔算術〕：
     - 每對 (D＋2＋γ·2D)/(2D) ≈ 0.5＋0.008＋0.05 ≈ 0.558，整個 block ≈ 0.5×1＋0.5×0.558 ≈ **0.78**。
     - 疊 INT4 KIVI（方向向量乘平均範數後量化，範數另存 BF16）≈ 0.3125×0.78 ≈ **0.24**。
3. **XL-copy**：深半段奇數層的 K、V＝前一個偶數層的。位元組 0.75（兩個模型的層數都是 4 的倍數，深半段剛好一半被省）。
4. **xKV-lite**（要整個載入區一起做，不是逐 block）：
   - 對 block 32–63 串起來的 16K token、每 4 層一窗：K 先反旋轉成 pre-RoPE，V 直接用。`torch.svd_lowrank` 到秩 r_K:r_V＝1:1.5，重建、K 再轉回 post-RoPE，切回 32 個 block 放進 `dec`。
   - 秩依位元組預算反推：
     - 每窗每 token 存 r_K＋r_V 個值，另加每層 B（r×d，攤到 16K token 上）。
     - 例：Llama d＝1024、W＝4，1/8 預算 ≈ r_K＋r_V ≈ 960（xKV 的 8 倍設定是 (384, 576)〔推論，由附錄 D.4 的線性縮放反推〕）。
     - 程式要照實算 nbytes 再報，不能用名目值。
5. **INT4＋丟 25%**（對照 XL-slerp＋INT4）：S2 現有的「刪除組」遮罩＋量化組的 decode 疊在一起。丟 RND0 與 DS1 前 2 名寫入時方法各一組。

**度量與判定**照 S2 不改：
- 對 E0 的 gold_lp 差、KL、first_ok；以題目為單位的配對 bootstrap（10,000 次）。
- 合併組走 sdpa 路徑（和量化同一條）；和刪除組配對時另跑 mask 路徑。
- 位元組一律照實算（`ratio_region`），不用名目值。

**同位元組對照表**〔算術：依 `m10_codec.py:246–253`、`m10_codec_x.py:321–324` 的 nbytes 公式，含 scale／zero〕：

| 名目預算 | 量化組（實際比） | 刪除組 | 合併組（實際比） |
|:--|:--|:--|:--|
| 1/2 | FP8 token／INT8 token（約 0.516） | 丟 50% | TM-2（0.5） |
| 約 0.25 | INT4 KIVI（0.3125）、K4V2（0.25） | 丟 75%；INT4＋丟 25%（約 0.234） | TM-4（0.25）、XL-slerp＋INT4（約 0.24）、xKV-lite 1/4 |
| 1/8 | INT2 KIVI（0.1875） | 丟 87.5% | TM-8（0.125）、TM-2＋INT4（約 0.156）、xKV-lite 1/8 |

### 5.3 成本〔估計〕
- 32K：
  - 每題 prefill 約 8 s（Llama）、7 s（Qwen）；每次 teacher forcing 約 0.11–0.13 s（c1a_s2_howto §5，引 09 海選的 stdout）。
  - 新加約 10 個配置，每題多約 1.3 s；xKV 的 SVD 每題約 1–3 s（xKV 表 3：A100 上 64K、W＝4 是 0.57 s；MI300X NOT_MEASURED）。
  - 若**併進 S2 的 shard**（共用 prefill）：96 題×2 個模型多約 10–15 分鐘。若單獨跑：約 35–40 分鐘。
- 128K：
  - 第二階段 252-block 的 screen 每題 Llama 69.6 s、Qwen 81.6 s（23 列，其中 7 列有生成；RUNLOG_MI300X 第 782 行）。峰值 63.1／35.4 GiB。
  - 只做 teacher forcing 的 10 個配置，每題 <70 s；每個模型 24 題 ≈ 0.5 小時，兩個模型約 1 小時。

---

## 6. 跟分層的關係

### 6.1 搬運量：保不保留 block 結構

| 做法 | 每個 512-token block 的位元組 | 佈局 | 能不能直接換成「少搬」 |
|:--|:--|:--|:--|
| TM-g（固定） | ×1/g（所有層、head 一樣） | 規則：每個 block 變成 512/g 列 | **能** |
| KVMerger、D2O、KeepKV、EMS、SelKV（相似度或分數決定） | 每個 head 不同 | 參差（ragged），和逐 head eviction 同樣的問題（L1 §3.2：kvpress 遮罩不省、Tangram 碎片） | 要另外打包、存 offset 才行〔推論〕 |
| XL-slerp、XL-copy | 深半段減半 | 規則：block 的層維度變小 | **能** |
| xKV | 每個 block 存 A 的對應列（r 個值／token）；每窗一個 B（r×d），整個請求共用 | 規則；B 要先到 | **能**，但 B 是「先載的字典」（像 codebook）〔推論〕 |
| XQuant 碼共用 | 深層 value 碼減半 | 規則 | 能 |

### 6.2 還原成本（Tiara 的用法：下層存壓縮版，還原時一次展開成 BF16 放 HBM）

〔推論〕第三階段 I3s 模擬已看到「真實 HBM 192 GB 下沒有容量壓力」（experiment_log I3s）。所以合併對 Tiara 的價值只在**下層的存放與搬運**，不在 HBM 容量。這也代表文獻在意的「解碼時 kernel 支不支援合併後的 cache」對我們不重要：還原時展開即可。

| 做法 | 還原時要做的 | 計算型態 | GPU 忙時〔推論，NOT_MEASURED〕 |
|:--|:--|:--|:--|
| TM-g | 每組重複 g 次（或保留合併形式、attention 加 log g 的 bias） | 記憶體密集，和反量化同級 | 可能和 FP8 反量化一樣變慢（第二階段：忙時慢 2.4–3.8 倍，research_map 第 3 點） |
| XL-slerp | 方向×範數、保留的 token 放回 | 逐元素 | 同上 |
| XL-copy | 指標重用，或複製一份 | 幾乎為零 | 幾乎不受影響 |
| xKV | 每層 A·B_ℓ（L·r·d FLOPs），K 再套 RoPE | **GEMM，計算密集** | 和還原時的重算（Cake 的計算線）搶 CU，可能比反量化更受影響 |
| XQuant 碼共用 | 和 INT2 反量化一樣，只是 scale 不同 | 同反量化 | 同反量化 |

〔算術〕xKV 重建的規模：Llama 載入區 16K token、r≈1000、d＝1024，K 加 V 每層約 2×16384×1000×1024 ≈ 34 GFLOP（乘加算 2 則約 67 GFLOP）。
- 和它省下的搬運比：BF16 每層 K＋V 是 64 MiB，1/8 剩 8 MiB；在 49.9 GiB/s 的 pinned H2D 上約省 1.1 ms／層。
- GEMM 在閒置的 MI300X 上應遠小於 1 ms，但實際數字 NOT_MEASURED。
- 在 NFS 級（實測讀 377–447 MiB/s）上，省下的時間大很多。

### 6.3 和 Cake、重算、量化的關係
- **只動載入區**：Cake 的前段會被精確重算，在前段合併既不省時間也不影響品質（和 L1 G1 對丟 token 的論證完全相同）。所以 §5 只動 block 32–63。
- **和 DROP＋重算的差別**：合併是有損的，而且**不能被精確重算回來**（除非從 token 重新 prefill）。Tiara 的 DROP＋重算是無損的。合併是「量化的另一種」，不是「丟的另一種」〔判讀〕。
- **和量化相乘**：MiniCache＋KIVI-4、xKV＋RTN-4（25.6 倍）、XQuant、CommonKV＋量化＋eviction（98%）都顯示可以疊。所以在 Tiara 的動作空間裡，合併與跨層若有用，應該是「INT4 之下的那幾階」〔判讀〕。
- **NFS 級的意義**〔推論〕：research_map 第 6 點記錄 NFS 級還原比 CPU 層慢 28–34 倍，INT4 只拉到 18–19 倍。要讓 NFS 級接近 CPU 層，載入區需要「數十倍」的位元組縮減。文獻裡唯一到這個量級、品質還被量過的是 xKV＋RTN-4 的 25.6 倍（Llama-3.1-8B RULER 64K，88.85 → 87.64）。這是 xKV-lite 值得放進 §5 的主要理由。Cake 會重疊計算與載入，還原時間不會和位元組成正比，所以這只是方向，不是預測值。

### 6.4 多輪：合併是否比丟掉保住資訊（和 L1 的證據連起來）
- **L1 已經有的**：寫入時依問題挑的刪除，多輪會壞（SCBench、KVzip、RocketKV-MT、EpiCache；L1 §0 第 2 點、§4 G4）；agent 下延後壓縮可以救回（2608.00902）。
- **合併、跨層這邊找到的**：
  - xKV 多輪 NIAH（Llama-3.1-8B、64K、8 倍）：SnapKV、PyramidKV 第 1 輪後陡降，xKV 各輪穩定。只有圖〔全文〕。
  - D2O MT-bench：4 個樣本的質性比較〔部分〕。
  - GRKV 用 query-agnostic 的協定：合併在 eviction 之上只多 0.6–1.7 分〔部分〕。
  - **token 合併的多輪定量比較：NOT_FOUND。**
- 〔判讀〕
  - 固定分組的 TM、XL、xKV 和問題無關，不會出現「第 1 輪挑對、第 2 輪挑錯」，所以它們的第 2 輪減第 1 輪應該約 0。這是設計上的性質，不需要實驗證明。
  - 要實驗回答的是：**同位元組下，第 2 輪的答對率，合併有沒有比「query-agnostic 的丟」與「量化」好**。
  - 第二階段已知量化本身也和問題無關，而且 INT4 幾乎不傷。所以真正的對手是量化，不是 QRY 丟。
  - 若 §5 在第 1 輪就被量化支配，第 2 輪不太可能翻轉〔推論〕，那第 2 輪的實驗可以省掉（§8 的 E-L5c 是條件式的）。

---

## 7. 缺口與停止條件

### 7.1 前人沒做、我們 2.5 天內能做的

| # | 缺口 | 前人最接近的 | 我們能做的 | GPU |
|:--|:--|:--|:--|:--|
| G-L5-1 | 我們兩個模型、32K／128K、自然文字 vs 合成文字下的跨層與相鄰 token 相似度（P1–P4、P6） | xKV（Llama-3.1-8B 的 CKA 與 cosine；Qwen2.5-7B-1M 只有品質、沒有 CKA 圖）；MiniCache（短 context） | E-L5a，在 S1 的 dump 上 | 0 |
| G-L5-2 | **Cake 載入區、同位元組**：token 合併／跨層 vs 量化階梯 vs block 丟棄，32K 與 128K | MiniCache 表 1（短、MHA、合併方用更少位元組時贏 KIVI-2）；xKV 表 1（64K，但位元組不對齊、不是載入區） | E-L5b，接在 S2 | 約 0.6＋1 |
| G-L5-3 | 合併當成「搬運用編碼」：還原時展開的成本，GPU 閒 vs 忙 | KeepKV（方向相反：完整版在外部）；xKV 的 xK-SR（V 卸載，沒有量忙時） | 只有 E-L5b 成立才做（微測，約 0.3 GPU 小時） | 條件式 |
| G-L5-4 | 第 2 輪：同位元組下合併 vs query-agnostic 丟 vs 量化 | 沒有定量的 | E-L5c，用 C1a 的 DS2 兩輪題 | 條件式約 0.5 |

### 7.2 停止條件（事前寫死；ε＝2 個百分點的 first_ok，和 S2、第二階段相同）
- **S-a（跨層「逐 token 合併」這一支）**：只要下列兩者之一成立，就不再做 MiniCache、KVSharer 式的直接共用，寫成負面結果，和 xKV 的第三方報告一致：
  - E-L5a 在 T4（自然文字）32K 上，兩個模型深半段的 P1 中心化 cosine 中位數都 < 0.5（K 或 V 任一）；
  - 或 E-L5b 的 XL-slerp（BF16，0.78）對 E0 掉超過 ε（配對 CI 上界 < −ε）。
- **S-b（token 合併）**：同時滿足下列兩條，就判「token 合併被量化支配」，停：
  - 每個預算下，TM-g 都沒有比位元組 ≥ 它的最好量化組好（配對差 CI 上界 ≤ 0）；
  - TM-8（0.125）也沒有比 INT2 KIVI（0.1875，位元組更多）在 CI 下「不差超過 ε」。
- **S-c（跨層＋INT4）**：XL-slerp＋INT4（約 0.24）沒有比 K4V2（0.25）**也**沒有比 INT4＋丟 25%（約 0.234）好（CI）→ 跨層被「降 V 位元」或「丟 block」支配，停。
- **S-d（xKV-lite）**：在 128K 上，xKV-lite 1/8（0.125）不如 INT2 KIVI（0.1875）超過 ε → 跨層 SVD 沒有把前緣往外推，停。只有它在 0.125 時 ≥ INT2 −ε，才做 G-L5-3 的還原成本微測。
- **S-e（多輪）**：第 2 輪同位元組下，合併對最好的 query-agnostic 丟或量化的差，CI 包含 0 或 < ε → 「合併比丟保住資訊」在本設定不成立。
- **全域**：若 S-b、S-c、S-d 都觸發，第三階段不把合併或跨層放進 Tiara 的動作空間。寫成「在 Llama-3.1-8B／Qwen2.5-7B-1M、32K–128K、Cake 載入區，合併與跨層共用被量化支配」，照 CLAUDE.md 規則 4 的精神當成有價值的負面結果。

### 7.3 事前預測〔預測；不是結果〕
- **32K 很可能飽和**：Llama 32K 連整份 INT2 KIVI 也只掉約 1 個百分點（RUNLOG_MI300X 11.2h）。所以 32K 上所有 ≥0.19 的組很可能都和 E0 分不開，**決勝在 128K、1/8 預算**（15.9：INT2 這類太狠的格式在部分格子隨長度變差）。
- TM-4、TM-8 在 T2（檢索）上很可能輸 INT4、INT2：needle 的 key 被平均掉。SelKV 也說預算很低時「能合的對象太少」。
- XL-slerp 在 Qwen 上很可能失敗（xKV：Qwen2.5-7B-1M 的 LongBench 掉 26.91%）。
- xKV-lite 在 Llama 的 1/8 也許能平或贏 INT2（xKV 表 1：88.50 vs KIVI-2 86.87）；在 Qwen 很可能打平（90.19 vs 90.45）。

---

## 8. 建議的最小實驗

### E-L5a　現象量測（0 GPU；CPU 約 30–60 分鐘）
- **輸入**：S1 的 dump（`10_s1_kvdump.sh`；llama／qwen × T4／T2 × 64 block；另有 252 block 的 5 層）。
- **量**：§4.1 的 P1（含中心化）、P2、P3（W＝1, 2, 4）、P4（post／pre-RoPE、0.75 門檻下的可合併比例）、P6。
- **工具**：直接用 `code/m16_kvstruct.py analyze`（K1，已有 P1 未中心化、P4、RoPE 核對）。在它上面加 P1 中心化、P2 CKA、P3 跨層串接秩、KVMerger 可合併比例、P6，產出放在 K1 的 `--out` 旁邊；不要另寫一支重複的〔判讀〕。
- **產出**：CSV（每列有 run_id、ts，CLAUDE.md §4.3）＋圖；位置跟 K1 的決定走。
- **規則**：
  - K1 已經在開頭與結尾斷言沒有 import torch／vllm／transformers。
  - `OMP_NUM_THREADS=4 nice -n 19 taskset -c 0-3`。
  - pre-RoPE 一律要先過 K1 的 `ropecheck`；沒過就停（規則 2）。
- **判定**：S-a 的第一條。結果另外決定 E-L5b 要不要放 XL-slerp：
  - 若中心化 cosine 中位數 <0.5，XL-slerp 只留一組當負對照。
  - 若 CKA 也低，xKV-lite 改成可選。

### E-L5b　同位元組品質（約 0.6 GPU 小時〔32K〕＋約 1 GPU 小時〔128K 切片〕）
- **做法**：§5.2。
  - 32K：併進 S2 的 shard（同題、同 seed 60000 起，T2–T5 各 24 題），加 TM-2／4／8、XL-copy、XL-slerp、XL-slerp＋INT4、INT4＋丟 25%（RND0、DS1 第 1 名）、TM-2＋INT4、xKV-lite 1/4、1/8，共約 11 組。
  - 128K：每個模型 24 題，只放決勝的 1/8 組（INT2 KIVI、K4V2、TM-8、TM-2＋INT4、xKV-lite 1/8、丟 87.5%）＋E0。
- **判定**：S-b、S-c、S-d。S2 的 report 已有配對 bootstrap 與「照實等位元組」，只要加組別。
- **GPU 規則**：品質實驗。照 CLAUDE.md §3 的品質規則，污染時保留分數、時間欄作廢；排在第二階段佇列之後，用第三階段 runner。

### E-L5c（條件式）　第 2 輪（約 0.5 GPU 小時）
- **前提**：只有 E-L5b 至少有一組沒被量化支配時才做。
- 用 C1a 的 DS2 兩輪題（seed 70000），只對第 2 輪問題做 teacher forcing。比較：該組 vs 同位元組的量化 vs SNAP／EXPATT 丟 vs 第 1 輪 QRY 丟。
- **判定**：S-e。

**總計**：
- 必做：E-L5a 0 GPU＋E-L5b 約 1.6 GPU 小時〔估計〕。
- 條件式：E-L5c 0.5＋G-L5-3 0.3。
- E-L5b 若只做 32K、或併進 S2 共用 prefill，可降到約 0.3 GPU 小時；但 32K 很可能飽和（§7.3），這樣省下的可能是一個沒有資訊的實驗。

---

## 9. 參考清單

> arXiv 編號、標題、第一作者、arXiv comment：10/20 由 arXiv API 讀到。venue 只寫 arXiv comment、會議頁或出版頁有寫的；其他標「venue 未查證」。深度同檔頭定義。

### token 合併

| 編號 | 第一作者 | 標題 | venue | 深度 | 連結 |
|:--|:--|:--|:--|:--|:--|
| 2407.08454 | Wang, Zheng | Model Tells You Where to Merge: Adaptive KV Cache Merging for LLMs on Long-Context Tasks（KVMerger） | 未查證 | 全文 | https://arxiv.org/abs/2407.08454 |
| PMLR v235 | Zhang, Yuxin | CaM: Cache Merging for Memory-efficient LLMs Inference | ICML 2024 | 部分 | https://proceedings.mlr.press/v235/zhang24n.html ；https://github.com/zyxxmu/cam |
| 2406.13035 | Wan, Zhongwei | D2O: Dynamic Discriminative Operations for Efficient Long-Context Inference of LLMs | ICLR 2025 | 部分 | https://arxiv.org/abs/2406.13035 ；https://github.com/AIoT-MLSys-Lab/D2O |
| 2504.09936 | Tian, Yuxuan | KeepKV: Achieving Periodic Lossless KV Cache Compression for Efficient LLM Inference | AAAI 2026 | 部分 | https://arxiv.org/abs/2504.09936 ；https://ojs.aaai.org/index.php/AAAI/article/view/40611 |
| 2506.05410 | Cui, Wanyun | Homogeneous Keys, Heterogeneous Values（AsymKV） | NeurIPS 2025 | 部分 | https://arxiv.org/abs/2506.05410 ；https://github.com/the-scale-lab/AsymKV |
| 2605.31105 | Peng, Junjie | GRKV: Global Regression for Training-Free KV Cache Compression in Long-Context LLMs | EMNLP 2026 Main | 部分 | https://arxiv.org/abs/2605.31105 ；https://github.com/pjunjie/GRKV |
| 2607.16213 | Bouyahiaoui, Soumia | SelKV: Selective KV Cache Merging with Per-Token Merge-or-Drop and Attention Compensation | 未查證 | 部分 | https://arxiv.org/abs/2607.16213 |
| 2412.08521 | Li, Yingxin | EMS: Adaptive Evict-then-Merge Strategy for Head-wise KV Cache Compression | 未查證 | 部分 | https://arxiv.org/abs/2412.08521 |
| 2503.10714 | Liu, Xin | ZSMerge: Zero-Shot KV Cache Compression for Memory-Efficient Long-Context LLMs | 未查證 | 部分 | https://arxiv.org/abs/2503.10714 ；https://github.com/SusCom-Lab/ZSMerge |
| 2507.06517 | Tang, Zicong | SpindleKV: A Novel KV Cache Reduction Method Balancing Both Shallow and Deep Layers | ACL 2025 | 部分 | https://arxiv.org/abs/2507.06517 ；https://github.com/tyxqc/SpindleKV |
| 2406.05317 | Cai, Ruisi | LoCoCo: Dropping In Convolutions for Long Context Compression | ICML 2024 | 部分 | https://arxiv.org/abs/2406.05317 ；https://github.com/VITA-Group/LoCoCo |

### 跨層共用（不用訓練）

| 編號 | 第一作者 | 標題 | venue | 深度 | 連結 |
|:--|:--|:--|:--|:--|:--|
| 2405.14366 | Liu, Akide | MiniCache: KV Cache Compression in Depth Dimension for LLMs | NeurIPS 2024 | 全文 | https://arxiv.org/abs/2405.14366 ；https://github.com/AkideLiu/MiniCache（只有專案頁） |
| 2410.18517 | Yang, Yifei | KVSharer: Efficient Inference via Layer-Wise Dissimilar KV Cache Sharing | 未查證（comment：Under Review by ICLR2025） | 全文 | https://arxiv.org/abs/2410.18517 ；https://github.com/yangyifei729/KVSharer |
| 2503.18893 | Chang, Chi-Chih | xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction | ICML 2026 | 全文 | https://arxiv.org/abs/2503.18893 ；https://github.com/abdelfattah-lab/xKV |
| 2508.16134 | Wang, Yixuan | CommonKV: Compressing KV Cache with Cross-layer Parameter Sharing | 未查證 | 部分 | https://arxiv.org/abs/2508.16134 ；https://github.com/rommel2021/CommonKV |
| 2510.11236 | Yang, Haoqi | XQuant: Achieving Ultra-Low Bit KV Cache Quantization with Cross-Layer Compression | EMNLP 2025 | 部分 | https://arxiv.org/abs/2510.11236 ；https://github.com/brinenick511/XQuant |
| 2508.10395 | Tomar, Aditya | XQuant: Breaking the Memory Wall for LLM Inference with KV Cache Rematerialization | 未查證 | 部分 | https://arxiv.org/abs/2508.10395 |
| 2502.00299 | Liu, Xiang | ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference | NeurIPS 2025 | 部分 | https://arxiv.org/abs/2502.00299 |
| 2607.06523 | Córdoba, Anna | DepthWeave-KV | 未查證 | 部分（可信度低） | https://arxiv.org/abs/2607.06523 |
| 2607.06519 | Córdoba, Anna | FreqDepthKV | 未查證 | 部分（可信度低） | https://arxiv.org/abs/2607.06519 |

### 要訓練或改架構（本次不可行）

| 編號 | 第一作者 | 標題 | venue | 深度 | 連結 |
|:--|:--|:--|:--|:--|:--|
| 2405.12981 | Brandon, William | Reducing Transformer Key-Value Cache Size with Cross-Layer Attention（CLA） | 未查證 | 摘要 | https://arxiv.org/abs/2405.12981 |
| 2405.05254 | Sun, Yutao | You Only Cache Once（YOCO） | NeurIPS 2024 | 摘要 | https://arxiv.org/abs/2405.05254 |
| 2405.10637 | Wu, Haoyi | Layer-Condensed KV Cache（LCKV） | ACL 2024 | 摘要 | https://arxiv.org/abs/2405.10637 ；https://github.com/whyNLP/LCKV |
| 2406.09297 | Zuhri, Zayd M. K. | MLKV: Multi-Layer Key-Value Heads | 未查證 | 摘要 | https://arxiv.org/abs/2406.09297 |
| 2410.14442 | Wu, You | A Systematic Study of Cross-Layer KV Sharing for Efficient LLM Inference | NAACL 2025 | 部分 | https://arxiv.org/abs/2410.14442 |
| 2410.03960 | Qiao, Aurick | SwiftKV: Fast Prefill-Optimized Inference with Knowledge-Preserving Model Transformation | 未查證 | 部分 | https://arxiv.org/abs/2410.03960 ；https://github.com/snowflakedb/ArcticTraining |
| 2512.03870 | Lin, Hongzhan | Reconstructing KV Caches with Cross-layer Fusion for Enhanced Transformers（FusedKV） | ICLR 2026 | 部分 | https://arxiv.org/abs/2512.03870 ；https://github.com/LivingFutureLab/FusedKV |
| 2604.22782 | Filippova, Anastasiia | Stochastic KV Routing: Enabling Adaptive Depth-Wise Cache Sharing | 未查證 | 部分 | https://arxiv.org/abs/2604.22782 |
| 2408.01890 | Mu, Yongyu | Cross-layer Attention Sharing for Pre-trained LLMs（LISA） | TACL | 摘要 | https://arxiv.org/abs/2408.01890 |
| 2502.00439 | Xiong, Yizhe | UniAttn: Reducing Inference Costs via Softmax Unification for Post-Training LLMs | 未查證 | 摘要 | https://arxiv.org/abs/2502.00439 |
| 2609.28006 | Xian, Xinglang | Shared Global KV with Layer-Specific Local History | 未查證 | 摘要 | https://arxiv.org/abs/2609.28006 |

### 只列（相關）

| 編號 | 標題 | venue | 深度 | 連結 |
|:--|:--|:--|:--|:--|
| 2403.08058 | CHAI: Clustered Head Attention for Efficient LLM Inference | 未查證 | 摘要 | https://arxiv.org/abs/2403.08058 |
| 2412.12094 | SepLLM | ICML 2025 | 摘要 | https://arxiv.org/abs/2412.12094 |
| 2407.21118 | Palu: Compressing KV-Cache with Low-Rank Projection | 未查證 | 摘要 | https://arxiv.org/abs/2407.21118 |
| 2505.15347 | FlowKV: Multi-Turn Isolated KV Cache Management | NeurIPS 2025 Workshop | 摘要（已在 L1「只列」） | https://arxiv.org/abs/2505.15347 |

### repo 內的依據

- `docs/stage3_20261019/research_map.md`（第 3、6 點）、`lit/L1_eviction_selective_loading.md`（§1、§3.2、§4 G1、G4）、`lit/L2_compression_offload_prefetch.md`。
- `docs/stage3_20261019/experiment_matrix.md`（S1、S2、§6）、`c1a_s2_howto.md`（§5 時間）、`experiment_log.md`（I3s）。
- `results/RUNLOG_MI300X.md`：11.2h（INT2 KIVI 只掉約 1 個百分點）、15.9（128K 好格式 ±2 個百分點）、第 782 行（252-block 每題時間與峰值記憶體）。
- `code/m16_s2_equalbytes.py`、`code/m10_sweep.py:108–150`、`code/m10_codec.py:246–253`、`code/m10_codec_x.py:321–324`、`code/m7_model.py`（post-RoPE 快取）、`code/m16_s1_kvdump.py`。
- `/mlsteam/data/tiara/stage3/jobs/queue_q/10_s1_kvdump.sh`（S1 dump 的 job，已排）。
- `code/m16_kvstruct.py`（K1：低秩、跨層 cosine、相鄰 token cosine、RoPE 核對；10/20 別的 session 寫的、未 commit；E-L5a 在它上面擴充）。
