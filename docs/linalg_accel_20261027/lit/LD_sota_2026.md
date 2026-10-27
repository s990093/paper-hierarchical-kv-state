# LD　2026 年底的 SOTA 前沿：還原、生產系統、混合架構、注意力的結構加速

- **日期**：2026-10-27（所有網頁、程式碼、arXiv 的存取日期都是這天）
- **執行者**：Claude（文獻 subagent D）。只搜尋與閱讀；沒有在這台 Mac 跑任何實驗或模擬；只寫這一個檔
- **查法**：arXiv API（`export.arxiv.org/api/query`，依提交日期排序，約 12 組查詢）、Semantic Scholar 的 Cake 引用清單、WebSearch 約 10 次、WebFetch 3 次（OSDI'26 議程、MiniMax 部落格）、`gh api` 讀 vLLM／SGLang／LMCache／TensorRT-LLM／Dynamo 原始碼與 PR、HF `config.json`。PDF 一律 `pdftotext -layout`，頁碼是 PDF 頁
- **審查方式**：**self-review，不是 cross-model。** 沒有獨立 verifier 抽查
- **分工**：A＝SSM／線性注意力理論與 softmax→linear 轉換；B＝混合模型的狀態快取；C＝softmax 注意力內部的結構（HCache GEMM、RoPE、LSE 合併、稀疏 prefill 估計、Hadamard、MLA 吸收）。本檔只補「2026 最新」與「誰是現在的 SOTA」，重疊處只給一行並指向該 agent
- **與 repo 既有整理的關係**：先 grep 過 `docs/stage3_design_20261024/03_competitors.md`、`lit/L8a–L8e`、`mechanisms/07_lit_quickcheck.md`、`docs/SOTA_ANALYSIS_20261006.md`、`docs/stage3_20261019/lit/L5a/L7a/L7d`。下面標「新」的是這些檔都沒出現過的 arXiv id；標「已有」的只補新事實

---

## 0. 一頁結論（白話）

**一句話**：2026 年沒有人推翻 Cake 的「前段重算、後段搬」；各種線性代數招數（隱藏狀態＋矩陣乘、跨層低秩、最小平方壓縮、迴歸補層）都**有人做了**，但都是「單獨做」——**沒找到任何一篇把它們和 Cake 式雙向還原、128K 以上、GPU 忙碌情況放在一起**。產業的模型則明顯分成兩派：線性混合（Qwen、Kimi、NVIDIA）和稀疏 softmax（DeepSeek、GLM、MiniMax M3），沒有人退回純 full attention 當旗艦。

逐軸（證據等級在括號裡）：

1. **暫停後回來的 KV 還原**：單請求的「算＋搬」雙向會合仍以 **Cake（ICML 2025）** 為基準；2026 的延伸是 **CacheFlow**（2604.25080，token×層×GPU 三維平行）、**Cascade**（2608.06557，延遲預算切分）——這兩篇 repo 已有。2026-10 新出現、和「線性代數＋中斷後還原」直接相關的有三篇：**OrbitWeaver**（2610.29183，衛星 VLM 斷電後用隱藏狀態＋KV 的區塊式檢查點恢復，以 HCache 為對照）〔部分〕、**RISK-KV**（2610.29544，用相鄰層 KV 做嶺迴歸補回省略的層，用在請求遷移與中斷恢復，但只在 ≤1.5B 模型、512 token）〔部分〕、**BellowsKV**（2610.35023，相鄰層隱藏狀態壓成一個 latent、每層用固定投影讀回，再量化；只在 GPU、只到 32K）〔部分〕。純「搬」那一側的新 SOTA 是 **KVStream**（2610.31827，多層儲存聚合頻寬）與 **Range**（2610.21241，把每物件成本拿掉）〔摘要〕。
2. **生產系統（2026-10）**：**Hadamard 旋轉的低位元 KV 已經在 vLLM v0.32.0 出貨**（TurboQuant、`int4_per_token_head` 用 RHT、UltraQuant FP4 的 K 在寫入時做 Hadamard）〔程式碼〕，但都是 GPU 上的格式。**SSM 狀態卸載已出貨**：vLLM OffloadingConnector 認得 `MambaSpec`、SGLang HiCache 有 `MambaPoolHost`〔程式碼〕。**隱藏狀態快取只有 LMCache 的 `HiddenStateStore`，用途是 vLLM-Omni 的 thinker→talker，不是拿來重建 KV**〔程式碼〕。**卸載路徑上的低秩 KV：沒找到**；壓縮只有 TensorRT-LLM 的 NVFP4 冷頁（repo L8c 已有）與 SGLang 一個無損 LZ4／ANS 的草稿 PR（#40551）〔程式碼〕。
3. **線性／混合 vs Transformer（長上下文）**：2026 年三家大廠的旗艦都是 3:1 線性混合——**Kimi K3**（69 KDA＋24 MLA，2026-07／08）〔全文（相關節）〕、**Qwen3.5／3.6／3.8-Next**（GDN:full = 3:1，3.8 再把 full 換成稀疏 QSA，2026-08）〔全文（相關節）〕、**Nemotron 3 Super**（Mamba＋attention，2026-04）〔部分〕。另一派是稀疏 softmax：**MiniMax M3**（MSA，2026-06）明說選稀疏是為了沿用既有軟硬體〔部分〕；**MiniMax M2（2025-10）確實退回 full attention**，官方部落格的理由是評測抓不到混合模型在多跳推理的缺陷、線性注意力基礎設施不成熟、數值精度敏感、prefix caching 與低精度狀態儲存還沒解決〔部分，原文已核〕。公開的「同訓練、同規模、≥64K」混合 vs full 對照幾乎沒有：Qwen3.8 的 Table 1 只有短上下文、Nemotron／Kimi 的長上下文比較對象本身也是混合或 SWA 模型〔判讀〕。
4. **注意力本身的結構加速（2026）**：要訓練的路線（稀疏注意力 QSA、MSA）在 128K–1M 報告與 full 打平或更好、prefill 快 7.6–14.2 倍〔部分〕；**不訓練、直接用在 Llama／Qwen 的結構招數，≥64K 的品質證據很少**：RACE（2610.28385，RoPE 成對座標選擇）在 256K 只量了速度，品質只到 LongBench〔部分〕；Attention Matching（2602.16284）用最小平方／NNLS 封閉解在 60K 做離線壓縮〔部分〕；線性化（OPAL 2609.31947、HyLo 2604.24715）要蒸餾訓練〔摘要〕。

**搶先（scoop）風險**〔判讀〕：沒有一篇同時具備「暫停後還原」＋「線性代數結構」＋「Cake 式雙向」＋「128K 以上 GQA」。最接近的是 OrbitWeaver（中斷恢復＋隱藏狀態＋層×chunk 排程，但在 Jetson、≤8B VLM、不比 Cake）與 RISK-KV（中斷恢復＋跨層迴歸，但玩具規模）。BellowsKV 讓 repo `results_analysis.md` 第 12、95、142 行「跨層低秩＋量化」那條儲存格式線更沒有新意（它用隱藏狀態解釋了同一件事）。

---

## 1. SOTA 總表

| 軸 | 現在的 SOTA（誰） | 主張 | 設定 | 日期 | 證據 | 我們能怎麼用〔判讀〕 |
|:--|:--|:--|:--|:--|:--|:--|
| 1a 單請求雙向還原 | **Cake**（ICML 2025） | 前段重算、後段從慢層搬，兩邊同時跑到會合 | dense GQA、長 prefix | 2024-10 arXiv | repo SOTA_ANALYSIS §2.1〔全文〕 | 仍是要打的基準 |
| 1b 多層／多 GPU 平行還原 | **CacheFlow**（2604.25080 v2） | token、層、GPU 三維平行；DP 規劃 | 多 session | 2026-04 | repo §2.2〔全文〕 | 已有 |
| 1c 延遲預算切分 | **Cascade**（2608.06557） | 預算決定搬多少、其餘重算 | Vidur 模擬為主 | 2026-08 | repo 03 §2.1〔全文〕 | 已有 |
| 1d 從多層儲存「搬」 | **KVStream**（2610.31827，新） | 每層儲存自行領下一塊，達到聚合頻寬上界的 5% 內 | LMCache、最多 128 GPU | 2026-10-22 | 〔摘要〕 | 搬的那一側變快 → Cake 會合點往「多搬」移 |
| 1e 單機 SSD 還原路徑 | **Range**（2610.21241，repo 已有 id）；SARIA（2610.31169，repo 已有 id） | 成本只隨位元組而非物件數；DRAM／SSD 協調重載 | vLLM＋LMCache | 2026-10 | 〔摘要〕 | 同上 |
| 1f 中斷恢復＋隱藏狀態 | **OrbitWeaver**（2610.29183，新） | 隱藏狀態＋KV 的區塊式檢查點，依層×chunk 相依排序恢復 | Jetson Orin、Qwen3-VL ≤8B | 2026-10-21 | 〔部分〕 | 最接近「暫停後還原＋線性代數」的系統；邊緣、短 |
| 1g 中斷恢復＋跨層迴歸 | **RISK-KV**（2610.29544，repo 只有摘要） | 省略某些層的 KV，用兩側層的 KV 嶺迴歸預測該層輸入再乘 W_K／W_V | ≤1.5B、512 token | 2026-10-21 | 〔部分〕 | 概念已被占；規模空白 |
| 2 生產系統 | vLLM v0.32.0、SGLang v0.5.22、LMCache main、TRT-LLM、Dynamo v1.6.0 | 見 §2.5 | — | 2026-10 | 〔程式碼〕 | 低秩卸載、HCache 式還原：都沒出貨 |
| 3 模型架構 | 線性混合：Kimi K3、Qwen3.6／3.8-Next、Nemotron 3；稀疏：DeepSeek V3.2／V4、GLM-5.x、MiniMax M3 | 兩派並存；Qwen3.8 兩者合用 | 1M 上下文 | 2026 | 〔全文（相關節）〕〔程式碼〕 | attention 層的 KV 問題還在，但每 token 小 4–16 倍（repo L8c §3） |
| 4a 結構加速（要訓練） | **QSA**（Qwen3.8-Next）、**MSA**（MiniMax M3） | 稀疏 softmax；與 full 打平；1M 時 prefill 快 7.6×（QSA kernel）／14.2×（MSA） | 109B–125B 模型 | 2026-06／08 | 〔部分〕 | 不適用於現成 Llama／Qwen2.5 |
| 4b 結構加速（不訓練） | ≥64K 有品質證據者：**沒找到**；最接近 Attention Matching（60K，離線） | 最小平方／NNLS 封閉解壓縮 | Qwen3-4B、Llama-3.1-8B、Gemma-3-12B | 2026-02 | 〔部分〕 | 可當「縮圖」的候選製法 |

---

## 2. 卡片：2026 的新論文

### 2.1 軸 1：還原與重用

### OrbitWeaver：Resuming Vision-Language Inference across Onboard Interruptions（venue 未列；arXiv 2610.29183）
- **讀了什麼**：〔部分〕v1 PDF 17 頁，讀了摘要、p.2–3 動機與設計概述、p.10–11 baseline 與主結果。https://arxiv.org/abs/2610.29183 ，查證 2026-10-27
- **一句話**：衛星上的 VLM 被斷電打斷後，從部分完成的狀態最快恢復
- **線性代數的關鍵一步**：沒有新的矩陣結構；用的是 HCache 式「存隱藏狀態、恢復時用投影算回 KV」，加上把 prefill 看成「層 × token chunk」的二維相依格，挑區塊（block-wise）做檢查點，恢復時解出最佳執行順序（p.2–3；摘要：規劃器 100 ms 內解完）
- **成立的前提**：不訓練、不改模型；對象是 Qwen3-VL 2B／4B／8B 與 MiniCPM-V 4.5（p.3），硬體是 Jetson Orin NX 16 GB 與 AGX Orin 64 GB（p.2）
- **量到的數字**：相對 baseline 端到端最多快 4.07 倍、SSD 寫入最多少 3.56 倍（摘要）；baseline 是 no-checkpoint、SONIC-layer、SONIC-chunk、HCache（HCache 沒開源，作者自己重做，p.10）；對 SONIC-layer／SONIC-chunk／HCache 的最大加速 4.07×／2.89×／2.67×（p.10–11）
- **跟 Tiara 的關係**〔判讀〕：「暫停後回來」換成「斷電後回來」，形狀相同；它證明「隱藏狀態＋KV 混合檢查點、依層×chunk 排程」在邊緣已有人做。它**沒有**比 Cake、沒有 128K、沒有資料中心 GPU 的爭用。可當我們相關工作裡「HCache 家族的最新延伸」
- **與既有整理的關係**：repo 沒有這個 id（grep 2610.29183 為 0）

### RISK-KV：Quality-Aware Selective KV Cache Reconstruction for Distributed LLM Inference（venue 未列；arXiv 2610.29544）
- **讀了什麼**：〔部分〕v1 PDF 10 頁，讀了 p.1–2（摘要、引言、與 HCache 的差別、模型清單）。https://arxiv.org/abs/2610.29544 ，查證 2026-10-27
- **一句話**：請求遷移或中斷恢復時，少存幾層的 KV，回來時從相鄰層算回來
- **線性代數的關鍵一步**：用目標層兩側「確切保留」的 KV 當錨，跨層**嶺迴歸**預測該層的注意力輸入，再把原模型的 W_K、W_V 摺進預測器以減少儲存（摘要；p.2）；用實際重建造成的 NLL 變化當風險，搜尋要省哪幾層
- **成立的前提**：不改預訓練權重，但要為每個模型校準迴歸器；只測了 Qwen2.5-0.5B／1.5B-Instruct 與 TinyLlama-1.1B（p.2）
- **量到的數字**：省 18.2%–33.3% 的歷史 KV、困惑度變化 <0.5%；512 token 歷史下比整段重播快 1.51–3.31 倍（摘要）
- **跟 Tiara 的關係**〔判讀〕：「依層選擇哪些不存、回來時用線性映射補」這個點子已經有人寫了，但規模離 128K、7B 很遠；而且它把恢復成本只和「整段重播」比，沒有和「搬」或 Cake 比
- **與既有整理的關係**：repo `docs/stage3_20261019/lit/L7d_recheck_lowrank_index.md` 第 436 行只列了 id，深度〔摘要〕。本卡升到〔部分〕

### BellowsKV：One Latent for Many Layers, Expanded on Demand（venue 未列；arXiv 2610.35023）
- **讀了什麼**：〔部分〕v1 PDF 36 頁，讀了摘要、§3.1–3.3（p.5–6）、§4 評測設定（p.6）、Table 1（p.7）、解碼速度一句（p.9）。https://arxiv.org/abs/2610.35023 ，查證 2026-10-27（v1 提交 2026-10-26）
- **一句話**：n 個相鄰層只存一份 latent，每層用固定投影讀回自己的 K、V
- **線性代數的關鍵一步**：把 n 層的輸入隱藏狀態疊成 h_B,t，c_t = W↓ᵀ h_B,t、k̂ = W↑⁽ⁱ⁾ c_t（Eq. 1，p.5）；W↓、W↑ 由離線二階矩 G = E[HᵀH] 的**降秩迴歸**封閉解得到（Eq. 2–3，p.5），G 的非對角塊（層與層輸入的共變）就是它比逐層低秩多出來的東西；latent 維度固定 R = p、改用 INT8／INT4 量化，另加每序列的低秩「whisper」殘差修正（p.6）
- **成立的前提**：不訓練，但要離線校準；Llama-3.1-8B、Mistral-7B、Qwen2.5-7B（摘要）
- **量到的數字**：Llama-3.1-8B、4K–32K 輸入（Protocol B，p.6）：n=4 INT4 在 13.84 倍時平均保留 98.0%，xKV 在 16.00 倍時 83.5%（Table 1，p.7）；摘要：3.6 倍約 99%、13–14 倍 97–98%，最強跨層對手 16 倍最多 84%
- **跟 Tiara 的關係**〔判讀〕：它的兩個觀察——相鄰層 KV 很不像但輸入隱藏狀態很像、同記憶體下「降維」比「量化」傷得多——和 repo `results_analysis.md` 第 89–99、142 行（K1／K1b、低秩＋INT4）一致。所以「跨層低秩＋量化當慢層格式」連量測價值都變薄。它**只在 GPU 上、只到 32K、沒有卸載與還原**；若當慢層格式，回來時多一次「latent→K」的矩陣乘（論文說 decode 時藏在讀取下，H100 32K 解碼 0.995 倍 dense，p.9）
- **與既有整理的關係**：repo 沒有這個 id

### EncBank：Cache the Encoder Within — Compact, Reusable Memory across LLM Queries（venue 未列；arXiv 2610.10058）
- **讀了什麼**：〔部分〕v1 PDF 17 頁，讀了摘要、§1 末、§2（p.2）。https://arxiv.org/abs/2610.10058 ，查證 2026-10-27
- **一句話**：文件只跑前 j 層，存那一層的殘差（4-bit），查詢時由調過的上層讀
- **線性代數的關鍵一步**：不是矩陣分解，而是「在第 j 層切開」：存隱藏狀態、上層 (L−j) 重算（類似 KVPR／HCache 的部分重算起點），儲存精度 H16／H8／H4
- **成立的前提**：要為每個 backbone 訓練上層的 LoRA reader（43.6M–72.7M 參數，p.2）；Qwen3 系列，含 full attention 與混合架構（摘要）
- **量到的數字**：Qwen3-8B，j 從 6 到 12，prefill 830.3→664.4 ms、RULER-B 98.29→96.07；j=18 時 499.5 ms 但 55.41（p.2）；4-bit 儲存各基準總分與原精度差在 1 分內（摘要）
- **跟 Tiara 的關係**〔判讀〕：證明「存某一層隱藏狀態、回來只重算上半」可以換到約 20% prefill 時間，但要訓練 reader。暫停還原若不准訓練，這條只能當對照
- **與既有整理的關係**：repo 沒有這個 id

### Attention Matching：Fast KV Compaction via Attention Matching（venue 未列；arXiv 2602.16284）
- **讀了什麼**：〔部分〕v2 PDF 29 頁，讀了摘要、§3 方法（p.3–4）、§4 資料設定（p.6）、Table 3（p.8）。https://arxiv.org/abs/2602.16284 ，查證 2026-10-27
- **一句話**：在幾秒內把一段長 context 的 KV 壓成小很多、仍能回答未知問題的 KV
- **線性代數的關鍵一步**：對參考 query 要求壓縮後的注意力輸出與注意力質量（mass）都和原本一致；分解成子問題：選 key 子集（OMP 貪婪）、每個 key 的偏差 β 由**非負最小平方**解（β = log w）、V 由**最小平方**封閉解（p.3–4，Algorithm 1）
- **成立的前提**：不訓練模型；每段 context 要離線算一次；Qwen3-4B、Llama-3.1-8B、Gemma-3-12B（p.5 圖）
- **量到的數字**：最多 50 倍、數秒內完成（摘要）；LongHealth 每段約 60K token（p.6），Table 3 是 Gemma-3-12B 在 60K 上、單張 H200 的時間拆解（p.8）
- **跟 Tiara 的關係**〔判讀〕：這是「縮圖」最有原則的製法——它直接最小化注意力輸出誤差，而不是 KV 本身的誤差。但它的目的是**取代**原 KV；我們的縮圖是**挑塊**用。拿它的輸出當「挑哪些原版塊要搬回」的索引，這輪沒找到（查詢見 §3）
- **後續**（都是〔摘要〕）：KV-Surgeon（2610.25933）在 50 倍時比 Attention Matching 多最多 4 分、成本相近，4B–30B；Latent Compaction（2610.20582）用 test-time training 擬合注意力輸出與 log-normalizer，RULER 32 倍時 96.7% 對 KVZip+ 40.0%，但只到 32K、Qwen3-1.7B 為主（p.2 Fig.1，〔部分〕）；Cartridges++（2609.35621）指出壓縮後的 KV 對「文件以外的問題」會變差
- **與既有整理的關係**：repo 沒有 2602.16284；KV-Surgeon 在 `docs/stage3_20261019/lit/L5d_stacking.md`、`L7a` 有提到

### 只讀到摘要的軸 1 新論文（〔摘要〕，不填細節）

| 論文 | 出處 | 摘要說了什麼 | 和我們的關係〔判讀〕 | repo |
|:--|:--|:--|:--|:--|
| KVStream | arXiv 2610.31827，2026-10-22 | 每步批次讀、寫延到 forward 之後、每塊複製到每一層儲存、哪層有空就領下一塊；在頻寬上界 5% 內；比 LMCache 吞吐多 1.49 倍、突發時平均 TTFT 最多少 6.1 倍；最多 128 GPU | 「搬」更快；不重算、不壓縮 | 新 |
| Range | arXiv 2610.21241，2026-10-15 | 同樣 6 GiB，切成 192 個物件 2.1 s、6,144 個 3.0 s；Range 1.79–1.80 s，LMCache 2.12→3.01 s | 還原時間的物件數懸崖；我們的 512-token 塊粒度要注意 | 已有 id（L7b、kv_sharing L1） |
| SARIA | arXiv 2610.31169，2026-10-22 | 依重用歷史選擇性寫 SSD、SSD 命中的塊回升到 DRAM、協調 DRAM 與 SSD 讀；SSD 寫少 57.1–58.9%、TTFT 比只用 DRAM 少 42.3–79.7%；含混合注意力模型 | 寫入時決策；與我們的寫入側一致 | 已有 id（kv_sharing L1） |
| Bounded-State Restoration | arXiv 2608.17826，2026-08-18 | 還原時本地暫存的「工作集」與外部狀態大小脫鉤；DeepSeek-V4-Flash、TP=2、兩台 DGX Spark，外部狀態 1.956→31.277 GiB/rank 時暫存固定 500.75 MiB；512K 還原 TTFT 43.1→17.6 s | 還原的記憶體面，不是時間面 | 已有 id（NOVELTY_CHECK_20261001） |
| SparPR | arXiv 2610.18140，2026-10-13 | 筆電 RTX 3070 Ti；依當前 query 挑塊、估每塊 copy 與重算成本、局部重算；1k／2k／4k prompt 延遲少 31／52／63% | 「依 query 挑塊＋算或搬」在消費級、短上下文已有雛形 | 新 |
| SARP | arXiv 2610.26849，2026-10-19 | 當機後 KV 索引的快速恢復；比記憶體索引恢復快 8.0–77.5 倍 | 索引，不是 KV 本身 | 新 |
| ATTUNER | arXiv 2609.36722，2026-09-29 | 位置無關快取的品質損失來自注意力分數而不是 KV；在 query 投影加低秩 adapter（<0.05% 參數）讀凍結的 KV；Qwen3-4B／8B，最多 3.73 倍 | 「改 query 而不改 KV」；需訓練 | 新 |
| LatentRemedy | arXiv 2610.18014，2026-10-13 | 用 MLP 把目標 prefix 的最後隱藏狀態映成幾個 latent token 放在重用的 KV 後面；需蒸餾訓練 | 非前綴重用，不是暫停還原 | 新 |
| InfoFlow KV | arXiv 2603.05353；ICML 2026（arXiv Comments） | 選擇性重算當成資訊流問題，用 query 的注意力範數挑 token | RAG 片段重用 | 新 |
| KV Packet | arXiv 2604.13226；NeurIPS 2026（arXiv Comments） | 文件 KV 不重算，外面包可訓練的 soft-token adapter | 同上 | 新 |

---

### 2.2 軸 3：架構的方向

### Kimi K3：Open Frontier Intelligence（技術報告；GitHub `MoonshotAI/Kimi-K3` commit `3cb39dfd`，2026-08-06）
- **讀了什麼**：〔全文（相關節）〕PDF 47 頁，讀了 §2.1（p.4）、架構表（p.11）、§5.4.1（p.22–24）、Table 2（p.27）。https://github.com/MoonshotAI/Kimi-K3/blob/3cb39dfd32e51c3328e2e4b4af21341247d06c43/k3_tech_report.pdf ，查證 2026-10-27
- **一句話**：2.8T 參數旗艦用 KDA（線性）:Gated MLA = 3:1 的混合
- **線性代數的關鍵一步**：KDA 是 delta rule 加 channel-wise forget gate（p.4–5）；繼承 Kimi Linear 的 DPLR 變體（agent A 範圍）
- **事實**：每個 block 3 層 KDA＋1 層 Gated MLA，共 93 層、69 KDA＋24 MLA（p.4、p.11）；HF `config.json` 的 `model_type` 是 `kimi_linear`、93 層、`max_position_embeddings` 1,048,576〔程式碼〕
- **服務端（和我們最相關）**：KDA 狀態與 MLA KV 放在同一個分頁池；因為狀態大、只能在稀疏的邊界存檢查點，原本 block 大小被迫 1024–6144 token，於是把雜湊粒度拆成 512 token、檢查點只存在 MLA 雜湊端點，**對話輪次邊界的檢查點保留給跨請求重用**（p.22–23，Fig.12）；speculative decoding 時不存每個 draft 位置的狀態，而是存「投影後的輸入」、在晶片上重放（和 ReplaySSM 同時期提出，p.24）——這是「存小的輸入、算回大的狀態」，和 HCache 同一種取捨
- **長上下文數字**：AA-LCR 74.7，表中最高（GLM-5.2 71.3；Table 2，p.27）；注意 AA-LCR 分數是引自 Artificial Analysis（2026-07-23）
- **跟 Tiara 的關係**〔判讀〕：旗艦級模型的暫停後還原＝「24 層 MLA 的 KV（可 Cake）＋69 層的固定狀態（只能整份搬或從檢查點重放）」。agent B 會處理狀態那一半
- **與既有整理的關係**：repo 沒有 Kimi K3；Kimi Linear（2510.26692）在 L8c 只讀摘要

### Qwen3.8-Next：On the Design of Qwen3.8-Next Architecture（arXiv 2608.30320，2026-08-31，36 位作者）
- **讀了什麼**：〔全文（相關節）〕v1 PDF 28 頁，讀了 §2.1.1（p.4–5）、Table 1（p.5）、§2.1.2 動機、Table 2–3（p.8）。https://arxiv.org/abs/2608.30320 ，查證 2026-10-27
- **一句話**：Qwen 的下一代：GDN 線性層＋每 4 層 1 層 full，持續預訓練時 full 換成稀疏 QSA
- **事實**：每 4 層 1 層 full attention，其餘 GDN；原文說週期性的 full 層對長上下文特別重要（p.4）；full 層保留 RoPE，因為 NoPE 版本在後訓練後更常停不下來（p.4）
- **混合 vs full 的直接比較**：25B-A3B、預訓練只到 32K，九個短上下文基準平均 Full 49.87、SWA 混合 51.15、GDN 混合 53.81（Table 1，p.5）——**沒有 ≥64K 的混合 vs 純 Transformer 對照**
- **長上下文**：同一個混合模型裡 full 層 vs QSA：RULER 512K–1M 90.08→93.00、MRCR 1M 20.71→26.44（Table 3，p.8）；QSA 在 1M 時 kernel 級 prefill 快 7.6 倍、decode 快 4.9 倍（p.2）
- **另查**：HF config：Qwen3.6-27B 64 層＝48 linear＋16 full、Qwen3.6-35B-A3B 40 層＝30＋10，`max_position_embeddings` 262,144〔程式碼〕
- **跟 Tiara 的關係**〔判讀〕：Qwen 的路線是「線性＋稀疏」疊在一起。我們的主力 Qwen2.5 在 Qwen 自己的產品線裡已經是上兩代的純 GQA
- **與既有整理的關係**：repo 沒有 2608.30320；Qwen3.5 的 config 在 L8c §3

### Nemotron 3 Super（arXiv 2604.12374，2026-04-14；NVIDIA）
- **讀了什麼**：〔部分〕v1 PDF 51 頁，只讀了摘要、base 結果表（p.17）、post-trained 結果表（p.31）。https://arxiv.org/abs/2604.12374 ，查證 2026-10-27
- **事實**：120B（active 12B）Mamba-Attention 混合 MoE，最長 1M；摘要宣稱吞吐是 GPT-OSS-120B 的 2.2 倍、Qwen3.5-122B 的 7.5 倍
- **長上下文**：post-trained RULER 1M：Nemotron 3 Super 91.64、Qwen3.5-122B-A10B 91.33、GPT-OSS-120B 22.30；AA-LCR 58.31／66.90／51.00（p.31）。base 模型 RULER 64K 92.26 → 1M 71.00（p.17）
- **跟 Tiara 的關係**〔判讀〕：這張表的三個對手**沒有一個是純 full attention**（Qwen3.5 是 GDN 混合、GPT-OSS 是 SWA／full 交替），所以它不能回答「混合 vs full」
- **與既有整理的關係**：repo 有 Nemotron-Nano-9B-v2、Nemotron-H 的 config（L8c），沒有 Nemotron 3 Super。Nemotron 3 白皮書（2512.20856）只讀摘要：Nano／Super／Ultra 全系列是 MoE 混合 Mamba-Transformer

### MiniMax：M2 退回 full、M3 改走稀疏
- **M2 部落格**：〔部分〕"Why Did MiniMax M2 End Up as a Full Attention Model?"，HF 社群文章，2025-10-30。https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model ，原頁文字已用 curl 取回核對，查證 2026-10-27。原文理由（改寫）：(1) 真正的瓶頸是評測——小規模混合模型在排行榜上打平，放大後在**複雜多跳推理**明顯不足；(2) 線性與稀疏注意力的基礎設施遠不如 full 成熟，很多線性架構連訓練都是 memory-bound；(3) 線性注意力對數值精度更敏感；(4) 推論端還要解決**低精度狀態儲存、prefix caching**、以 linear 為 backbone 的 speculative decoding；理論上幾千 token 就該划算，但只是理論。附錄：試過把 checkpoint 改成 SWA 混合，context 越長越差
- **M3／MSA**：〔部分〕arXiv 2606.13392（v1 2026-06-11，v2；Lai 等），讀了摘要、§1（p.2）、Table 3（p.12）。MSA 是建在 GQA 上的區塊稀疏 softmax，原文說選稀疏 softmax 是為了最大程度沿用既有軟硬體（p.2）；109B 模型上 per-token 注意力計算在 1M 時少 28.4 倍，H800 prefill 快 14.2 倍、decode 7.6 倍（摘要）；128K 持續預訓練後 RULER-128K Overall Full 72.00 vs MSA 72.12、HELMET-128K 46.53 vs 45.93（Table 3，p.12）。HF `MiniMaxAI/MiniMax-M3` config：60 層、`sparse_topk_blocks` 16、`sparse_block_size` 128、`max_position_embeddings` 1,048,576〔程式碼〕
- **結論**〔判讀〕：「MiniMax 退回 full attention」只對 M2（2025-10）成立；2026 的 M3 是稀疏 softmax，**沒有回到線性**，也沒有留在 full
- **與既有整理的關係**：repo L8c 第 133 行由 config 推得「M2 退回 full」〔官方文件〕；L8d 第 98 行有 M3 的稀疏設定。本節補上官方理由與 MSA 原文

---

### 2.3 軸 4：注意力本身的結構加速（只列最新）

### RACE：RoPE-Aligned Attention Compression Based on Query-Key Activation Energies（arXiv 2610.28385，2026-10-20；IBM 等 5 人）
- **讀了什麼**：〔部分〕v1 PDF 28 頁，讀了摘要、p.7–8 評測與效率、p.26 附錄說明。https://arxiv.org/abs/2610.28385 ，查證 2026-10-27
- **一句話**：不重建、直接在較小的 head 維度做注意力，且和 RoPE 相容
- **線性代數的關鍵一步**：Q、K 共用一個降維投影；限制成「座標選擇」後證明 RoPE 相容若且唯若整對旋轉座標一起保留；依 Q、K 聯合的激活能量挑旋轉對，等於最小化 QKᵀ 近似誤差的一個可分上界（摘要）；結果摺進權重，不需特製 kernel
- **成立的前提**：用校準集、不訓練；Qwen2.5-7B、Llama-3.2-3B、Granite-4.1-3B（p.7）
- **量到的數字**：50% KV 保留、256K 時，總 FLOPs 少 47.8%、吞吐 1.79 倍、TTFT 少 43.2%（摘要；單張 H100）。**品質只量到 LongBench 與短任務**（p.7），256K 那段在附錄講的是記憶體與速度（p.26）——**≥64K 的品質：論文沒量**
- **跟 Tiara 的關係**〔判讀〕：會讓重算（prefill）本身變快，Cake 的會合點往「多算」移；但改變了模型（需整體採用）
- **與既有整理的關係**：repo 沒有這個 id；RoPE 結構屬 agent C，這裡只記它是 2026-10 的最新

### 只讀到摘要的軸 4 新論文（〔摘要〕）

| 論文 | 出處 | 摘要說了什麼 | ≥64K 品質 | repo |
|:--|:--|:--|:--|:--|
| OPAL：On-Policy Attention Linearization | arXiv 2609.31947，2026-09-25 | 把 Qwen3-4B、MiMo-7B 蒸餾成線性混合，學生自己取樣長軌跡、老師給密集監督；NIAH 100%、常識 87–94%、數學 83–93%，3B token | 摘要沒寫長度 | 新（agent A 範圍） |
| HyLo：Long-Context Aware Upcycling | arXiv 2604.24715，2026-04-27 | 把 Llama／Qwen 改成 MLA＋Mamba2／GDN 混合；KV 少 90% 以上；1.7B 只用 10B token 在 RULER-64K 贏 JetNemotron | 只有 1B／3B 級 | 新（agent A 範圍） |
| NoteBank：History Replays Itself | arXiv 2610.15715，2026-10-11 | 線性模型外掛稀疏情節記憶、需要時重放原文；1M 時 GDN 在六個 RULER 檢索任務 1.7%→70.3%；儲存約 MHA KV 的 1/2,600 | 有（1M），但對象是線性模型 | 新 |
| Mechanics of Long-Context Hybrid Models Part 1.1 | arXiv 2610.10114，2026-10-07 | 線性混合較受益於長上下文持續預訓練、SWA 混合較會外推；提出 Sliding-Window Linear Attention，64K NIAH-SK1 100% | 有（64K NIAH） | 新 |
| Sketch&Walk | arXiv 2602.07397，2026-02-07 | Hadamard sketch 估注意力分數、跨層 walk 累積、選 top-k 區塊；20% 密度近無損、最多 6 倍 | 摘要沒寫 | 新（agent C 範圍） |
| SMat-Attention | arXiv 2609.36062，2026-09-28 | 用 VC 維度控制的結構化遮罩連接 softmax 與線性注意力；小規模實驗 | 沒有 | 新 |
| Kronecker Product Attention | arXiv 2610.12657，2026-10-08 | K、V 表示成 Kronecker 積之和，推廣 TPA；KV 最多少 10 倍；要從頭訓練 | 摘要沒寫 | 新 |

---

## 2.5 軸 2：生產系統（程式碼證據）

**一句話**：Hadamard 低位元 KV 與 SSM 狀態卸載已經出貨；HCache 式隱藏狀態還原、卸載路徑上的低秩 KV 都沒有。

| 系統（版本／commit） | Hadamard／旋轉的低位元 KV | SSM 狀態卸載 | 隱藏狀態快取 | 卸載路徑上的壓縮／低秩 | 證據 |
|:--|:--|:--|:--|:--|:--|
| **vLLM v0.32.0**（2026-10-18，`5df63a72`） | 有：`turboquant_attn.py` 建 Sylvester Hadamard（`_build_hadamard`，約第 105–116 行），所有層共用一個旋轉（`_ensure_on_device` docstring，約第 410–416 行）；`int4_per_token_head.py` 第 5 行寫寫入前做一次 RHT，第 955–997 行是 RHT 與 Hadamard 矩陣快取；`ultraquant/format.py` 第 16 行：K 在寫入時 Hadamard 旋轉、V 不旋轉 | 有：OffloadingConnector `offloading/scheduler.py` 第 147–149 行對 `MambaSpec` 回傳「只依賴單一狀態」；`canonical_mapping.py` 第 398 行處理 MambaSpec 頁大小；`cache.py` 第 189 行 `mamba_cache_mode`（`align`＝在 block 邊界存最後狀態）、約第 200–213 行 ReplaySSM 選項 | 沒找到 | `vllm/v1/kv_offload/` 目錄 tree 中沒有任何檔名含 compress／quant／lowrank／svd（以 `git/trees` 列出、0 筆）；open PR #63586（2026-10-25）是讓卸載支援 compressed-MLA | 〔程式碼〕 |
| **SGLang v0.5.22**（2026-10-19） | 只出現在 DSA／DeepSeek-V4 indexer 相關檔（code search `hadamard` 9 筆，全在 `dsa/`、`dsv4/`、`deepseek_v4_backend.py`） | 有：`mem_cache/pool_host/mamba.py` 第 75 行 `MambaPoolHost(HostKVCache)`，HiCache 的 host 池（最後 commit `26a1e511`，2026-10-25） | 沒找到 | open PR #40551（2026-09-21）：HiCache L2 與 P/D 共用的**無損** LZ4／ANS（nvCOMP）壓縮，明寫不量化 KV、不宣稱加速 | 〔程式碼〕 |
| **LMCache** main（`6993f46`，2026-10-27） | 沒找到 | 沒查（agent B） | **有，但用途不同**：`lmcache/v1/config.py` 第 635 行 `enable_hidden_state_cache`，說明是為 vLLM-Omni 多階段管線（thinker→talker）在 KV 旁邊存隱藏狀態；設計文件 `docs/design/v1/hidden_state_store.md` 說下游需要的中間激活「無法只從 KV 重建」 | code search `low_rank` 0 筆；`svd` 2 筆（`rust/raw_block/src/lib.rs`、一個 SnapKV 範例 notebook，內容未查證）；CacheGen 可選（repo L8c） | 〔程式碼〕 |
| **TensorRT-LLM**（v1.4.0rc1，2026-10-22） | `cpp/tensorrt_llm/batch_manager` 搜 `hadamard` 0 筆 | GDN／SSM／DSA 附屬 buffer 在冷頁壓縮時無損保留（repo L8c R11） | 沒查 | NVFP4 冷頁（只在 host／disk、Blackwell；repo L8c） | 〔程式碼〕〔文件〕 |
| **NVIDIA Dynamo** v1.6.0（2026-10-22） | 沒查 | `lib/llm/src/block_manager` 搜 `mamba` 0 筆 → 〔未查證〕 | 沒查 | KVBM G1–G4 不壓縮（repo L8c R29） | 〔程式碼〕〔文件〕 |
| **Mooncake** v0.3.14-pre（2026-10-26） | 沒查 | 沒查 | 沒查 | repo L8c：設計文件沒提壓縮 | — |
| **Kimi K3 生產服務**（技術報告 §5.4） | — | KDA 狀態與 MLA KV 共用分頁池，輪次邊界檢查點保留（p.22–23） | speculative 時存投影後輸入（p.24） | — | 〔全文（相關節）〕 |

〔判讀〕對 Tiara：(1)「Hadamard 旋轉＋INT4 放 GPU」是 vLLM 現成選項，不能當貢獻，但可當基準；(2) 卸載路徑上沒有任何引擎做「低秩」或「隱藏狀態重建 KV」，這兩個在生產端仍是空白，但學界已有 HCache（2025）與 OrbitWeaver、RISK-KV、BellowsKV（2026）；(3) 混合模型的狀態卸載已經是 vLLM／SGLang 的主線功能。

---

## 3. 哪些已被做掉、哪些還空著（判讀）

**一句話**：每一招單獨都有人做了；「組合」和「規模」還空著。

### 3.1 已被做掉（有原文）

| 招數 | 誰做了 | 範圍 |
|:--|:--|:--|
| 存隱藏狀態、還原時乘 W_K／W_V 算回 KV | HCache（EuroSys 2025，repo 已有）；OrbitWeaver（2026-10，中斷恢復，邊緣） | 單請求、批次 1 或邊緣 |
| 跨層把 KV 壓成共用 latent 再量化 | xKV、JoLT、KVTC（repo 已有）；**BellowsKV**（2026-10，從隱藏狀態出發、降秩迴歸封閉解） | GPU 上、≤32K |
| 少存幾層、用相鄰層迴歸補回（中斷恢復） | **RISK-KV**（2026-10） | ≤1.5B、512 token |
| 用最小平方／NNLS 做出能回答未知問題的小 KV | **Attention Matching**（2026-02）、KV-Surgeon、Latent Compaction | 離線、單文件、≤60K |
| Hadamard 旋轉的低位元 KV | vLLM v0.32.0 出貨 | GPU 常駐格式 |
| 混合模型的狀態檢查點與輪次邊界保留 | Kimi K3 生產、vLLM `align`、SGLang `MambaPoolHost` | 見 agent B |
| 依 query 挑塊＋估算 copy／重算成本 | SparPR（筆電、≤8K）；repo 已有 ReadKV、Janus 等 | 短上下文或原生稀疏模型 |

### 3.2 這輪沒找到（＝空白，不等於不存在；CLAUDE.md 規則 7）

1. **線性代數還原＋Cake 式雙向會合**：沒有任何一篇把 HCache 式 GEMM 還原、跨層 latent、或迴歸補層，當成 Cake「搬」那一側的第三種選項，並在 128K–256K 的 GQA 模型上量會合點。查詢：arXiv `abs:"KV cache" AND (abs:"hidden state" OR abs:"hidden states" OR abs:activation) AND (abs:restore OR abs:offload OR abs:reconstruct OR abs:recompute)`；Semantic Scholar 的 Cake 引用清單（索引到 40 筆，2026 年 17 筆）
2. **壓縮後的小 KV 當「挑塊索引」**：Attention Matching／KV-Surgeon 的產物用來決定回來時搬哪些原版塊——沒找到。查詢：arXiv `(abs:compaction OR abs:cartridge OR ti:"attention matching") AND abs:"KV cache"`
3. **GPU 爭用下的格式選擇**（repo 主線 B）：這輪新讀的 2026-10 論文（KVStream、Range、SARIA、OrbitWeaver）都沒有把「GPU 忙碌」放進還原決策；OrbitWeaver 看的是電力與斷電，不是 SM 爭用
4. **混合 vs 純 full attention 的 ≥64K 同條件對照**：2026 公開報告裡沒找到（Qwen3.8 只有短上下文；Nemotron 3 Super 與 Kimi K3 的對手都不是同訓練的純 full）

### 3.3 對研究方向的含意〔判讀〕

- **產業方向**：2026 的旗艦不是線性混合（Kimi K3、Qwen3.5–3.8、Nemotron 3），就是稀疏 softmax（DeepSeek V3.2／V4、GLM-5.x、MiniMax M3），Qwen3.8 兩者都用。**沒有人以 full attention 當 2026 旗艦架構**（就這輪讀到的範圍；Meta、Google、OpenAI 的 2026 模型這輪沒查）。所以 Tiara 的對象要寫成「attention 層的 KV」：混合模型每 4 層還有 1 層、稀疏模型仍存完整 KV（DeepSeek V4 prefix 命中時整段還原，repo L8d）
- **Llama-3.1-8B／Qwen2.5 當主模型**：在 2026 年是「純 GQA 的代表」，審稿人會問混合模型；repo 05 §X5 已登記用 Qwen3.5／Nemotron-Nano 擴充，這輪的證據支持這樣做
- **線性代數招數的定位**：最有可能的新意是「還原方式的選單多一項（隱藏狀態 GEMM／跨層 latent），由當下資源決定」，而不是招數本身

---

## 4. 沒查到、沒查證的

- **2026 系統 venue 的完整議程**：只用 WebFetch 讀了 OSDI'26 議程（三段讀完，摘要級）：KV 相關有 Strata、ECHO、"No Buffer, No Bottleneck"（都已在 repo）。**ATC'26、EuroSys'26、NSDI'26、FAST'26、ASPLOS'26、MLSys'26、SOSP'26 這輪沒逐一翻議程**，依賴 repo 既有整理（Bidaw FAST'26、Strata OSDI'26 等）
- **Kimi K3 的 KDA 是否與 Kimi Linear 相同**：報告 p.4–5 有改動（下界衰減、全秩輸出 gate），細節屬 agent A
- **Nemotron 3 Ultra**、**Jamba 2026 版**、**Meta／Google／OpenAI 的 2026 架構**：沒查
- **OPAL、HyLo、NoteBank、KVStream、Range、SARIA、SparPR 等表內論文**：只讀摘要，細節不得引用
- **LMCache `svd` 兩筆搜尋結果**的內容沒打開
- **Dynamo 是否支援混合模型卸載**：code search 0 筆，〔未查證〕
- WebFetch 對 MiniMax 部落格的整理有改寫風險；本檔用到的句子都已用 curl 取回原頁文字核對過

---

## References（存取日期一律 2026-10-27）

| # | 來源 | URL | 狀態 |
|:--|:--|:--|:--|
| D1 | OrbitWeaver, arXiv 2610.29183 v1 | https://arxiv.org/abs/2610.29183 | 〔部分〕 |
| D2 | RISK-KV, arXiv 2610.29544 v1 | https://arxiv.org/abs/2610.29544 | 〔部分〕 |
| D3 | BellowsKV, arXiv 2610.35023 v1 | https://arxiv.org/abs/2610.35023 | 〔部分〕 |
| D4 | EncBank, arXiv 2610.10058 v1 | https://arxiv.org/abs/2610.10058 | 〔部分〕 |
| D5 | Fast KV Compaction via Attention Matching, arXiv 2602.16284 v2 | https://arxiv.org/abs/2602.16284 | 〔部分〕 |
| D6 | KV-Surgeon, arXiv 2610.25933 | https://arxiv.org/abs/2610.25933 | 〔摘要〕 |
| D7 | Latent Compaction with TTT, arXiv 2610.20582 | https://arxiv.org/abs/2610.20582 | 〔部分〕（摘要＋p.2、p.5） |
| D8 | Cartridges++, arXiv 2609.35621 | https://arxiv.org/abs/2609.35621 | 〔摘要〕 |
| D9 | KVStream, arXiv 2610.31827 | https://arxiv.org/abs/2610.31827 | 〔摘要〕 |
| D10 | Range, arXiv 2610.21241 | https://arxiv.org/abs/2610.21241 | 〔摘要〕 |
| D11 | SARIA, arXiv 2610.31169 | https://arxiv.org/abs/2610.31169 | 〔摘要〕 |
| D12 | Bounded-State Restoration, arXiv 2608.17826 | https://arxiv.org/abs/2608.17826 | 〔摘要〕 |
| D13 | SparPR, arXiv 2610.18140 | https://arxiv.org/abs/2610.18140 | 〔摘要〕 |
| D14 | SARP, arXiv 2610.26849 | https://arxiv.org/abs/2610.26849 | 〔摘要〕 |
| D15 | ATTUNER, arXiv 2609.36722 | https://arxiv.org/abs/2609.36722 | 〔摘要〕 |
| D16 | LatentRemedy, arXiv 2610.18014 | https://arxiv.org/abs/2610.18014 | 〔摘要〕 |
| D17 | InfoFlow KV, arXiv 2603.05353（ICML 2026 依 Comments） | https://arxiv.org/abs/2603.05353 | 〔摘要〕 |
| D18 | KV Packet, arXiv 2604.13226（NeurIPS 2026 依 Comments） | https://arxiv.org/abs/2604.13226 | 〔摘要〕 |
| D19 | Kimi K3 技術報告（GitHub commit 3cb39dfd） | https://github.com/MoonshotAI/Kimi-K3/blob/3cb39dfd32e51c3328e2e4b4af21341247d06c43/k3_tech_report.pdf | 〔全文（相關節）〕 |
| D20 | Kimi-K3 config.json | https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json | 〔程式碼〕 |
| D21 | Qwen3.8-Next 架構論文, arXiv 2608.30320 | https://arxiv.org/abs/2608.30320 | 〔全文（相關節）〕 |
| D22 | Qwen3.6-27B、Qwen3.6-35B-A3B config.json | https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/config.json ；https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json | 〔程式碼〕 |
| D23 | Nemotron 3 Super, arXiv 2604.12374 | https://arxiv.org/abs/2604.12374 | 〔部分〕 |
| D24 | NVIDIA Nemotron 3 白皮書, arXiv 2512.20856 | https://arxiv.org/abs/2512.20856 | 〔摘要〕 |
| D25 | MiniMax, "Why Did MiniMax M2 End Up as a Full Attention Model?"（2025-10-30） | https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model （同文：https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model ） | 〔部分〕原頁文字核對 |
| D26 | MiniMax Sparse Attention, arXiv 2606.13392 v2 | https://arxiv.org/abs/2606.13392 | 〔部分〕 |
| D27 | MiniMax-M3 config.json | https://huggingface.co/MiniMaxAI/MiniMax-M3/raw/main/config.json | 〔程式碼〕 |
| D28 | Kimi Linear, arXiv 2510.26692 | https://arxiv.org/abs/2510.26692 | 〔摘要〕 |
| D29 | RACE, arXiv 2610.28385 | https://arxiv.org/abs/2610.28385 | 〔部分〕 |
| D30 | OPAL, arXiv 2609.31947 | https://arxiv.org/abs/2609.31947 | 〔摘要〕 |
| D31 | HyLo, arXiv 2604.24715 | https://arxiv.org/abs/2604.24715 | 〔摘要〕 |
| D32 | NoteBank, arXiv 2610.15715 | https://arxiv.org/abs/2610.15715 | 〔摘要〕 |
| D33 | Mechanics of Long-Context Hybrid Models 1.1, arXiv 2610.10114 | https://arxiv.org/abs/2610.10114 | 〔摘要〕 |
| D34 | Sketch&Walk, arXiv 2602.07397 | https://arxiv.org/abs/2602.07397 | 〔摘要〕 |
| D35 | SMat-Attention, arXiv 2609.36062 | https://arxiv.org/abs/2609.36062 | 〔摘要〕 |
| D36 | Kronecker Product Attention, arXiv 2610.12657 | https://arxiv.org/abs/2610.12657 | 〔摘要〕 |
| D37 | vLLM v0.32.0（commit 5df63a7203cd）：`vllm/v1/attention/backends/turboquant_attn.py`、`vllm/v1/attention/ops/int4_per_token_head.py`、`vllm/v1/attention/ops/ultraquant/format.py`、`vllm/config/cache.py`、`vllm/distributed/kv_transfer/kv_connector/v1/offloading/{scheduler,config,canonical_mapping}.py` | https://github.com/vllm-project/vllm/tree/v0.32.0 | 〔程式碼〕 |
| D38 | vLLM PR #63586（open，2026-10-25） | https://github.com/vllm-project/vllm/pull/63586 | 〔程式碼〕只讀標題 |
| D39 | SGLang `python/sglang/srt/mem_cache/pool_host/mamba.py`（commit 26a1e511） | https://github.com/sgl-project/sglang/blob/26a1e5110d4a3b185bdb9050145050615b3f805f/python/sglang/srt/mem_cache/pool_host/mamba.py | 〔程式碼〕 |
| D40 | SGLang PR #40551（open，2026-09-21） | https://github.com/sgl-project/sglang/pull/40551 | 〔程式碼〕讀了描述 |
| D41 | LMCache `lmcache/v1/config.py`、`docs/design/v1/hidden_state_store.md`（main 6993f46） | https://github.com/LMCache/LMCache/blob/6993f46c91f182def4e73b7882177e8881671301/lmcache/v1/config.py | 〔程式碼〕 |
| D42 | GitHub releases：vLLM v0.32.0、SGLang v0.5.22、TensorRT-LLM v1.4.0rc1、Dynamo v1.6.0、Mooncake v0.3.14-pre | https://github.com/vllm-project/vllm/releases ；https://github.com/sgl-project/sglang/releases ；https://github.com/NVIDIA/TensorRT-LLM/releases ；https://github.com/ai-dynamo/dynamo/releases ；https://github.com/kvcache-ai/Mooncake/releases | 〔程式碼〕只讀版本與日期 |
| D43 | OSDI'26 technical sessions | https://www.usenix.org/conference/osdi26/technical-sessions | 〔摘要〕（WebFetch 整理） |
| D44 | Semantic Scholar：Cake（arXiv 2410.03065）的引用清單 | https://api.semanticscholar.org/graph/v1/paper/arXiv:2410.03065/citations | 〔摘要〕 |
| — | repo 既有：`docs/stage3_design_20261024/03_competitors.md`、`lit/L8c_practice_and_trends.md`、`lit/L8d_native_sparse_index_restore.md`、`docs/SOTA_ANALYSIS_20261006.md`、`docs/stage3_20261019/results_analysis.md`、`docs/stage3_20261019/lit/L7d_recheck_lowrank_index.md` | （本 repo） | 引用，未重抄 |
