# P06｜「KV 存在 GPU 外、用某個精度、之後還原」的系統論文怎麼設計實驗

> **一句話**：這一派的慣例是「全精度＋均勻量化＋只重算／只載入」當對手、用真實資料集的標準指標量品質、用品質對延遲（或大小）的圖呈現、掃頻寬與長度。幾乎沒有人報重複次數或信賴區間，也**沒有人做依位置的精度實驗**。我們的計劃在統計、位元組、時間拆解上比慣例嚴；但缺真實 benchmark、缺已發表方法（KIVI 原版）當對手、只有一個長度與一個主模型、沒有併發下的端到端量測。

---

## 0. 這張卡的證據等級（先講清楚）

- **本次（2026-10-14）重讀原文**：CacheGen（arXiv v6，§7 全節＋附錄 C–E）、Cake（ICML 正式版 PMLR PDF，§5.1–5.3）、LMCache 技術報告（arXiv 2510.09665v2，§8）。PDF 放在 `/private/tmp/claude-501/p06_sys_eval_dl_20261014/`，只用 `pdftotext` 轉文字讀，沒有執行任何下載內容。頁碼一律是 PDF 頁。
- **由抽取工作讀原文、我 grep 核對關鍵句**：KVServe、PatchKV、HA-RAG、SparKV、EvicPress、AdaptCache、FlexGen、QServe、Rethinking、Yuan'24。抽取工作逐節讀評估章節；我對每篇 3–10 句關鍵原文用 grep 確認頁碼相符，其餘細節（尤其抽取工作自己算的數字）沒有逐一覆核。
- 沒有任何論文只靠既有卡。

---

## 1. 總表

縮寫：FP＝全精度；UQ＝均勻量化；RC＝只重算；LD＝只載入。「讀到的步驟」指被壓縮的 KV 第一次被誰讀。

| 論文 | 場景 | 主要自變數 | 對手 | 品質（資料／指標／讀到的步驟） | 速度 | 位元組 | 呈現 | 硬體／模型／重複 | 依位置的實驗 |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| CacheGen | 遠端儲存 → 網路串流；可退回送文字重算 〔原文 §5.3 / p.6–7〕 | 頻寬 0.4–400 Gbps、長度 0.1K–15K、併發數、SLO 〔原文 §7.3 / p.10〕 | FP（存儲成本圖）、UQ 3/4/8-bit、RC（vLLM 文字）、H2O、LLMLingua 〔原文 §7.1 / p.8–9〕 | 4 資料集 662 context；acc／F1／PPL；新問題的 prefill 讀解碼後 KV 〔原文 §7.1 / p.8；§6 / p.7〕；單輪 | TTFT＝載入＋新問題 prefill；拆解含解碼 〔原文 §7.1 / p.8；Fig. 14a / p.11〕；頻寬怎麼施加：原文沒寫 | 實際位元流大小 〔原文 §7.1 / p.8〕 | 品質對 TTFT 散點、品質對大小 〔原文 Fig. 8 / p.9；Fig. 9 / p.10〕 | 4×A40／3 模型／只有 Fig. 13 寫「20 條 trace 平均」 | 沒有（只有熵依 token 位置分組的壓縮分析） |
| Cake（量化實驗） | 儲存層 → GPU 載入，與重算雙向並行 〔原文 App. A / p.12〕 | 位元 16/8/3、頻寬 7–56 Gbps、GPU 佔比、長度 4K–16K 〔原文 §5.1 / p.5–6；Table 7 / p.8〕 | RC（vLLM 0.6.2）、LD（LMCache 0.1.4） 〔原文 §5.1 / p.5〕 | **沒有任何品質數字** 〔P01 卡 B.11〕 | 只有 TTFT；I/O 是延遲注入模擬 〔原文 §5.1 / p.5〕 | 名目（由表推得約 8.2／3.1 bit）〔P01 卡 A.6，計算〕 | 加速比表 | 2×A100 TP／量化只用 LongAlpaca-13B／原文沒寫 | 沒有 |
| KVServe | P→D 網路傳輸，或從遠端 KV pool 取回；**自己沒有重算路徑** 〔原文 §3.1 / p.4；§6.2 / p.9〕 | 頻寬 5–100 Gbps 多組 〔原文 Fig. 12–14 / p.10–11〕；3 個模型；6 個任務 〔原文 §7.1 / p.9–10〕；長度、併發：原文沒寫 | 不壓縮 BF16、CacheGen、KIVI（2-bit）、DuoAttention 〔原文 p.10〕 | lm-eval-harness 嵌進線上推論；相對準確率，門檻 97% 〔原文 §7.1 / p.9；Table 1 / p.11〕；題數：原文沒寫 | JCT（PD）、TTFT（prefix caching）；拆成 Prefill／Comp／Comm／Decomp／Decode 〔原文 Fig. 15 / p.11〕；**真實網路**＋traffic shaping 限速 〔原文 §7.2 / p.11〕 | bytes 之比；KIVI 有算 metadata 〔原文 p.4；p.12〕 | 延遲對準確率散點、3D Pareto；門檻下比壓縮比 〔原文 Fig. 5 / p.4；Fig. 10 / p.7；Table 1 / p.11〕 | 4×A100、H100、4090／5090 等／3 模型／原文沒寫 | 沒有（MixHQ 依 head）〔原文 p.6〕 |
| PatchKV | CPU DRAM → GPU；近編輯點重算、其餘載入 〔原文 §IV / p.4–5；§V-A / p.6〕 | 精度 FP16／K8V8／K8V4；髒窗、edit 長度、block 大小 〔原文 p.5–9〕；3 個模型 〔原文 §V-A1 / p.6〕 | 只重算（Prefix Cache）、只載入（Direct KV Reuse，FP16）、CacheBlend；消融才比均勻 INT8／INT4 〔原文 §V-A3 / p.6；Fig. 9 / p.8〕 | HotpotQA-E、2WikiMQA-E、TriviaQA-E 的 F1 〔原文 p.6–7〕；題數：原文沒寫；resume 後讀 | resume TTFT，含 fetch＋反量化＋重算；**編碼不算** 〔原文 p.6；p.9〕；真實 H2D（pageable 20.4–21.0、pinned 23.2–23.3 GiB/s）〔原文 p.6〕 | 實測 MiB 〔原文 §V-E / p.9〕 | F1 對 TTFT 散點 〔原文 Fig. 7 / p.7〕；固定重算集只換精度 〔原文 Fig. 9 / p.8〕 | 8×A800／3 模型／**paired bootstrap 區間**（Table II、Fig. 8）、微基準跑 5 次 〔原文 p.8；p.9〕 | 有「離編輯點的距離」實驗；精度和位置的關係沒測 〔原文 Fig. 3 / p.3〕 |
| HA-RAG | 預算好的 RAG KV chunk（512 token），依熱度放 GPU／pinned／pageable／disk，載入時解壓；沒有重算 〔原文 p.5–6〕 | 四種 8-bit 格式與比例、放置門檻、test set 256–4096 〔原文 p.7–8〕；只有 LLaMA-2-7B 〔原文 p.6〕 | 只有 TurboRAG（BF16）＋自家消融 〔原文 p.6；p.9〕 | TriviaQA 1024 題；ROUGE-1 F1，**參考答案是 TurboRAG 的輸出** 〔原文 p.7；p.9〕；另有 RMSE | TTFT（輸入到第一個 token）〔原文 p.6–7〕；解壓時間以秒報、對象單位不明 〔原文 Table II / p.7〕 | 只報壓縮比約 2.0 〔原文 Table II / p.7〕 | 速度、品質分開畫；逐題排序曲線 | 1×A100 40GB／1 模型／原文沒寫 | 沒有（依熱度） |
| SparKV | 雲端 KV 經真實 Wi-Fi 串流＋本地稀疏重算，兩路並行 〔原文 §III / p.3；§VI-A / p.7〕 | 精度不掃；真實 Wi-Fi＋干擾；併發 1–4；長度 10–38K；5 個模型（4-bit 權重）；9 個資料集 〔原文 p.7–10〕 | CacheGen（只載）、Strong Hybrid（＝Cake＋同樣量化）、Local Prefill（只算）〔原文 §VI-A / p.8〕 | LongBench／VideoMME 官方指標，每集 22–200 題 〔原文 Table III / p.7〕；KV 經 past_key_values 給第一次問題 prefill 〔原文 §V / p.7〕 | TTFT（max_new_tokens＝1）〔原文 p.7〕；傳輸、解碼＋搬移佔比 〔原文 p.9〕 | 實際壓縮大小當排程輸入 〔原文 p.5〕；總量與 metadata：原文沒寫 | TTFT CDF 與品質 CDF 分開 〔原文 Fig. 9、11 / p.8–9〕 | RTX 5080 筆電、Jetson AGX／5 模型／主結果原文沒寫 | 有，但只量**計算時間**對位置；結論是位置是差勁代理 〔原文 Fig. 3、5 / p.3–4〕 |
| EvicPress | GPU／DRAM／SSD／遠端磁碟；從某層載入 〔原文 §4.1 / p.5；§5 / p.7〕 | 壓縮法（3 種丟 token）、α、QPS；5 個模型 〔原文 p.8–9〕；**沒有量化** 〔原文 §8 / p.12〕 | Prefill（vLLM 0.11.2）、只逐出 LRU、各壓縮法＋LRU、IMPRESS 〔原文 §6.1 / p.8〕 | LongBench 12 子集 555 個 context；每個 context 用 GPT-5 生 100 題（50 profile／50 測）〔原文 Table 2 / p.7；p.8〕；**品質＝和未壓縮答案的 embedding 相似度** 〔原文 p.9〕 | TTFT（含取 KV＋新問題 prefill）、ITL、同 TTFT 的 QPS 〔原文 p.9〕；真實 H100＋SSD 〔原文 p.8〕 | 名目壓縮比 〔原文 p.4〕 | 品質對 TTFT 曲線；**同品質比 TTFT、同 TTFT 比品質都報** 〔原文 Fig. 7 / p.9〕 | 1×H100／5 模型／原文沒寫 | 沒有（粒度是整個 context）〔原文 §4.2 / p.6〕 |
| AdaptCache | DRAM＋SSD，對每個 entry 選逐出／全精度／壓縮 〔原文 p.1–2〕 | 延遲與品質的權重；Poisson 到達、多個 rate 〔原文 p.2〕；1 個模型 | 不壓縮、KIVI＋LRU、StreamingLLM＋LRU、Prefill 〔原文 p.2〕 | LongBench 6 子集 1,100 個 context；**品質＝和原始 prefill 答案比** F1／ROUGE-L／CodeBLEU 〔原文 p.1–2〕 | TTFT；解壓有量但沒給數字 〔原文 p.2〕；磁碟 1 GB/s（實測或限速：原文沒寫） | 原文沒寫 | 品質對 TTFT 圖；同品質比速度、同速度比品質 〔原文 Fig. 2 / p.2〕 | 1×A100／1 模型／原文沒寫 | 沒有 |
| FlexGen | KV 依比例放 GPU／CPU／disk；權重與 KV 一起壓成 4-bit 〔原文 §4 / p.5–6；§5 / p.7〕；沒有重算 | 只有一組：4-bit、group 64 〔原文 §5 / p.7〕；長度 512／1024＋32 〔原文 §6 / p.7〕 | 速度：DeepSpeed、Accelerate、Petals、FlexGen 不壓縮 〔原文 §6 / p.7〕；品質：FP16 〔原文 Table 5 / p.9〕 | Lambada acc、WikiText PPL；速度 run 用 dummy 權重，品質另跑 〔原文 §6 / p.7；Table 5 / p.9〕；讀到的步驟：原文沒寫 | 端到端吞吐，真 SSD 〔原文 §6 / p.7〕；反量化沒單獨拆 | 名目 4-bit | 吞吐表＋品質表分開 | T4、3090 〔原文 Table 1 / p.7；Table 12 / p.19〕／OPT／原文沒寫 | 沒有 |
| QServe（對照） | KV4 常駐 GPU，寫入 page 時每 head 動態量化 〔原文 §5.1 / p.8〕 | KV 只有 INT4 一種；序列長 128–1536（kernel）〔原文 Table 1 / p.10〕 | 品質：FP16、SmoothQuant、AWQ、QuaRot、Atom 等；kernel：TRT-LLM KV8 〔原文 p.10–12〕 | WikiText2 PPL、zero-shot、LongBench 10 子任務 〔原文 Table 2–3 / p.11–12；Table 5 / p.13〕；品質走 HF 實作 〔原文 p.10〕 | 同記憶體下最大吞吐（實測）〔原文 §6.3 / p.12〕；KV4 單獨一步 1.47× 〔原文 Fig. 16 / p.13–14〕 | 名目 4-bit | 表＋正規化長條 | A100、L40S／10 個以上模型／原文沒寫 | 沒有 |
| LMCache 報告 | CPU／遠端儲存／PD | 模型、QPS、長度、頻寬 32/64/128 Gbps 〔原文 §8.1 / p.10–11；Fig. 15 / p.14〕 | vLLM 0.10.2、vLLM CPU offload 0.11.0、兩個商用服務 〔原文 §8.1 / p.11〕 | **不評估壓縮**：壓縮 API「沒有涵蓋在上述應用」〔原文 §7 / p.10〕 | TTFT、ITL 對 QPS 〔原文 Fig. 8 / p.11〕 | 不適用 | 延遲對 QPS 曲線 | 8×H100／6 模型／原文沒寫 | 沒有 |

方法論論文（不是系統，但講「應該怎麼做」），見第 3 節：Rethinking（MLSys'25）、Yuan'24（EMNLP Findings）。

---

## 2. 逐篇

### 2.1 CacheGen（SIGCOMM'24，arXiv 2310.07240v6）

機制見 [P02](P02_CacheGen_mechanism.md)。這裡只補評估設計。

- **場景**：KV 編碼後存在儲存伺服器，經網路串流；chunk 可改送文字由 GPU 重算 〔原文 §5.3 / p.6–7〕。程式介面是 `store_kv`／`get_kv`／`generate_with_kv`，後者把 KV 經 `past_key_values` 傳給 generate 〔原文 §6 / p.7〕。
- **自變數**：
  - 頻寬：0.4–15 與 15–400 Gbps，長度固定 16K 〔原文 §7.3 / p.10 / Fig. 11〕。
  - 併發請求數（代表「每個請求分到的 GPU 週期」）〔原文 §7.3 / p.10 / Fig. 12〕。附錄把 GPU 可用度定義成 1/n，n＝併發數，和頻寬一起畫熱圖 〔原文 App. D / p.19 / Fig. 19〕。
  - 長度：0.1K–15K，頻寬固定 3 Gbps 〔原文 §7.3 / p.10 / Fig. 12〕。
  - SLO 0.5／1 s；每個 chunk 的頻寬從 0.1–10 Gbps 隨機抽 〔原文 §7.4 / p.10–11 / Fig. 13〕。
  - 模型：長 context 微調版 Mistral-7B、Llama-34B、Llama-70B 〔原文 §7.1 / p.8〕。
- **對手**：
  - 「Default quantization」：每層同位元，3／4／8-bit 〔原文 §7.1 / p.8〕。
  - 「Text context」：vLLM 從文字重算（RC）〔原文 §7.1 / p.9〕。
  - Context compression：LLMLingua、理想化 H2O（離線就用 query）〔原文 §7.2 / p.9〕。
  - 未壓縮 KV 只出現在儲存成本圖 〔原文 Fig. 14d / p.11〕。
  - 沒有「只載入未壓縮 KV」的 TTFT 對照〔判讀，依 Fig. 8 / p.9 圖例只有 Quantization、Text、CacheGen〕。
- **品質**：
  - LongChat 200、TriviaQA 200、NarrativeQA 200、WikiText 62，共 662 個 context 〔原文 Table 2 / p.8〕。每個 context 一個問題〔判讀，原文沒寫題數〕。
  - 指標：LongChat accuracy（答案含正確主題）、F1、PPL 〔原文 §7.1 / p.8〕。
  - 讀到的步驟：TTFT 包含「載入 KV＋新問題的 prefill」〔原文 §7.1 / p.8〕，所以解碼後的 KV 由新問題的 prefill 讀〔判讀〕。單輪。
  - 設計編碼器用的資料是評估資料的子集 〔原文 §7.1 / p.8〕。
  - 另有 MTurk 使用者研究 270 個評分 〔原文 §7.5 / p.11〕。
- **速度**：
  - TTFT＝query 到達到第一個 token 〔原文 §7.1 / p.8〕。
  - TTFT 拆解有 Comp.、Trans.、Decode 三段 〔原文 Fig. 14a / p.11〕。編碼是離線的，約 200 ms 〔原文 §7.5 / p.11 / Fig. 14c〕。
  - **頻寬怎麼施加：原文沒寫**（全文搜 throttle、emulate、simulate、tc，方法段沒有命中）。
- **位元組**：「KV cache 壓縮後的大小」〔原文 §7.1 / p.8〕，是實際位元流。機率表是否算進去：原文沒寫（見 E07 卡）。
- **呈現**：
  - 品質對 TTFT 散點，每個子圖標「Better」方向 〔原文 Fig. 8 / p.9〕。
  - 品質對大小 〔原文 Fig. 9 / p.10〕。
  - 文字主張是「**同品質下**，大小小 3.5–4.3×」〔原文 §7.2 / p.9〕。也就是同品質比大小／速度。
  - 「無損」門檻：accuracy 掉 ≤2%、F1 ＜0.1%、PPL ＜0.1 〔原文 §7.2 / p.9〕。並稱 8-bit「幾乎無損」〔原文 §7.2 / p.9〕。
- **重複**：只有 Fig. 13「每點是 20 條頻寬 trace 的平均」〔原文 §7.4 / p.11〕。其他圖原文沒寫。沒有誤差棒。
- **依位置**：只有「依 token 位置分組的熵下降最少」這種壓縮分析 〔原文 §5.1.3 / p.5〕，以及 anchor token 用 8-bit 〔原文 §5.2 / p.6〕。**沒有品質對位置的實驗**。

### 2.2 Cake（ICML 2025，PMLR 267）

機制見 [P01](P01_Cake_quant_mechanism.md)。

- **場景**：儲存層 → GPU；前段重算、後段載入，會合點自然形成 〔原文 App. A / p.12〕。
- **自變數**：
  - 位元：16／8／3（3-bit 是 KVQuant）〔原文 §5.1 / p.6；Table 7 / p.8〕。
  - 頻寬：主表 7／25／32／56／100 Gbps，對到 GCP egress、SSD、RoCE 〔原文 Table 2 / p.6〕。量化表只用 7–56 Gbps 四點〔P01 卡 A.7〕。
  - GPU 佔比：用 vLLM token budget 定義，例：512 裡給 Cake 256＝50% 〔原文 §5.1 / p.6〕。
  - 長度：4K–16K 每 2K 一點，合成 prompt，理由是「只有 token 長度影響效能」〔原文 §5.1 / p.5〕。
  - 模型：LongAlpaca-7B／13B（MHA）、Llama-3.1-8B／70B（70B 用 FP8 權重）〔原文 §5.1 / p.5 / Table 1〕。量化實驗只用 LongAlpaca-13B、16K 〔P01 卡 A.7〕。
- **對手**：只算（vLLM 0.6.2 chunked prefill，budget 512）與只載（LMCache 0.1.4）〔原文 §5.1 / p.5〕。量化實驗裡，只載基線也載量化後的 KV〔P01 卡 A.3，計算〕。**沒有和其他壓縮法比**。
- **品質**：**沒有**。資料集名稱（LongChat、TriviaQA、NarrativeQA）只用來決定長度範圍 〔原文 §5.1 / p.5〕。
- **速度**：TTFT 〔原文 §5.1 / p.5〕。I/O 是依 chunk 大小與頻寬算延遲、到時才放行的模擬 〔原文 §5.1 / p.5〕。反量化成本原文沒寫〔P01 卡 A.6〕。
- **位元組**：原文沒寫。由表推得約 8.2／3.1 bit／元素〔P01 卡 A.6，計算〕。
- **呈現**：「對只載＼對只算」的加速比表 〔原文 Table 3 / p.6；Table 7 / p.8〕。平均加速排除了對極弱基線的格 〔原文 §5.2 / p.6〕。
- **重複**：原文沒寫（搜 trial、variance、deviation、median 沒有命中）。
- **依位置**：沒有。

### 2.3 KVServe（arXiv 2605.13734，頁首 SIGCOMM'26）〔本次由抽取工作讀 §7 全節；關鍵 8 處我 grep 核對頁碼相符〕

- 場景：PD 分離的網路傳輸，或 prefix caching 從遠端 KV pool 取回 〔原文 §3.1 / p.4；§7.2 / p.11〕。pool 實際放哪一層：原文沒寫。
- 自己沒有重算路徑；可行集為空時退回預設保守設定 〔原文 §6.2 / p.9〕。「最多快 32.8×」的分母是重算 〔原文 p.11〕。
- 自變數：
  - 策略空間＝transform × 混精度 quantizer × codec 〔原文 §5.1 / p.6 / Fig. 7〕。各 profile 實際幾 bit：原文沒寫。
  - 頻寬：Fig. 13 用 5–100 Gbps，Fig. 14 用 5–15 Gbps 〔原文 p.10–11〕。
  - 模型：Qwen2.5-7B-Instruct、Llama-3.1-8B-Instruct、Qwen2.5-32B-Instruct 〔原文 §7.1 / p.9〕。
  - 任務：GSM8K、HumanEval、Multi-News、Qasper 用來 profile；2WikiMQA、HotpotQA 當沒看過的 〔原文 p.9–10〕。
  - context 長度、併發、GPU 負載：沒有掃。
- 對手：不壓縮 BF16、CacheGen、KIVI（固定非對稱 2-bit）、DuoAttention；CacheGen 與 KIVI 是把核心演算法改寫成它的 pipeline 模組 〔原文 p.10〕。
- 品質：
  - 把 lm-eval-harness 嵌進線上推論量 〔原文 §7.1 / p.9〕。
  - 報「相對 Default 的百分比」，門檻 97% 〔原文 Table 1 / p.11；Fig. 12 / p.10〕。
  - 各任務的指標與題數：原文沒寫。
  - 讀到的步驟〔判讀〕：prefill 端壓、decode 端解 〔原文 Fig. 6 / p.5〕，所以 PD 情境下是 decode 讀有損 KV。
  - 抽取工作用 Table 1 重算：KVServe 在 HotpotQA 單項約 95.8%，低於 97%，只有平均過門檻〔判讀，未獨立覆核〕。
- 速度：JCT 與 TTFT；Fig. 15 拆成 Prefill、Comp、Comm、Decomp、Decode 〔原文 p.11〕。頻寬用 sender 端 traffic shaping 和 RoCE 網卡限速，是**真實網路** 〔原文 §7.2 / p.11〕。
- 位元組：壓縮比用 bytes 定義 〔原文 §3.1 / p.4〕；KIVI 的 group metadata 把 2-bit 的理論 8× 壓到約 5.33× 〔原文 p.12〕。KVServe 自己的 metadata：原文沒寫。
- 呈現：延遲對準確率散點、3D Pareto；在 97% 門檻下比壓縮比；不達門檻的對手打 × 〔原文 Fig. 5 / p.4；Fig. 10 / p.7；Table 1 / p.11〕。
- 重複、誤差棒：原文沒寫。
- 依位置：沒有。MixHQ 依 head 分精度 〔原文 p.6〕。
- 既有卡 D6_KVServe 要修：原文斷行成 "re-computation"，所以 grep "recomput" 找不到；p.9 的 lm-eval、三個模型、traffic shaping、KIVI metadata 卡上都漏了〔抽取工作的回報〕。

### 2.4 PatchKV（arXiv 2609.26219）〔本次由抽取工作讀評估節；關鍵 10 處我 grep 核對頁碼相符〕

- 場景：只有 CPU DRAM 一層 〔原文 §V-A2 / p.6〕。resume 時重算新 span 與「髒窗＋attention 挑出的遠處 block」，其餘從 CPU 載入、融合反量化 〔原文 §IV / p.4–5〕。
- 精度：FP16／K8V8／K8V4 逐 block 混合，比例由校準門檻決定、沒有直接掃 〔原文 p.5–6〕。
- 對手：Prefix Cache（只重算）、Direct KV Reuse（FP16 全載入）、CacheBlend 〔原文 §V-A3 / p.6〕。均勻 INT8／INT4 只在消融 〔原文 Fig. 9 / p.8〕。KIVI、CacheGen 只在相關工作。
- 品質：HotpotQA-E、2WikiMQA-E、TriviaQA-E 的官方 F1 〔原文 p.6–7〕。題數、context 長度、edit 怎麼造：原文沒寫。
- 速度：resume TTFT 從 KV 恢復開始算，含 fetch＋反量化＋重算＋第一個 token 〔原文 p.6〕。量化與儲存（336.3 ms）在 offload 時做，**不算進 TTFT** 〔原文 p.9〕。真實 H2D 〔原文 p.6〕。
- 位元組：實測 MiB（Qwen3-32B、8K：FP16 2,007.8、INT8 1,023.4、INT4 523.4、混合 799.4）〔原文 p.9〕。
- 呈現：F1 對 TTFT 散點 〔原文 Fig. 7 / p.7〕；固定重算集、只換精度的 ΔF1 對 TTFT 〔原文 Fig. 9 / p.8〕。這個消融只在 Llama-3.3-70B 上做〔抽取工作的回報〕。
- 統計：Table II 與 Fig. 8 有 **paired bootstrap 區間** 〔原文 p.8〕；微基準跑 5 次 〔原文 p.9〕。是本卡唯一報配對區間的系統論文。
- 依位置：K／V 漂移對「離編輯點距離」0–1000 token 〔原文 Fig. 3 / p.3〕；同預算下比 near-edit、random、top-attn 等選法 〔原文 Table I / p.4〕。**精度依位置沒測**。
- 既有卡 D6_PatchKV 要修：頻寬把 pageable 和 pinned 混寫成 20.4–23.3；「比 INT8 快 5.6%」是含重算的 TTFT，只看還原段是 13.9%〔抽取工作的回報〕。

### 2.5 HA-RAG（HPCA 2026）〔本次由抽取工作讀全文；關鍵 6 處我 grep 核對頁碼相符〕

- 場景：RAG 文件的 KV chunk（512 token）預先算好，依熱度放 GPU、CPU pinned、CPU pageable、disk 〔原文 p.5–6〕。只有載入。
- 自變數：四種 8-bit 格式（INT8、FP8 E4M3、FP8 E5M2、GSE-8）與各占比例；放置門檻；test set 256–4096 〔原文 p.3；p.7–8〕。頻寬、併發：原文沒寫。
- 模型：只有 LLaMA-2-7B；語料 MS MARCO 1,000 篇，問題來自 TriviaQA 〔原文 p.6〕。
- 對手：只有 TurboRAG（BF16）〔原文 p.6〕。
- 品質：1024 題的 ROUGE-1 F1，**參考是 TurboRAG 的輸出**，不是標準答案 〔原文 p.7；p.9〕。約 60% 的題和 TurboRAG 完全相同 〔原文 p.9〕。另報 RMSE。
- 速度：TTFT。Table II 的解壓時間以秒報，對象是「大小 1024 的資料集」，單位與裝置不明 〔原文 Table II / p.7〕。
- 硬體：1×A100 40GB、PCIe 4.0 〔原文 Table I / p.6〕。重複：原文沒寫。
- 依位置：沒有。
- 既有卡 D6_HARAG 要修：硬體其實有寫（Table I / p.6）；品質不只 RMSE，還有 1024 題 ROUGE-1〔抽取工作的回報，我已 grep 核對〕。

### 2.6 SparKV（IEEE IoT J.，arXiv 2604.21231）〔本次由抽取工作讀全文；關鍵 4 處我 grep 核對頁碼相符〕

- 場景：KV 事先算好放雲端，經真實 Wi-Fi 6 串流；其餘 chunk 本地用稀疏 attention 重算，兩路並行 〔原文 §III / p.3；§VI-A / p.7〕。
- 精度**不掃**：固定用 layer-wise 非均勻量化＋Huffman，位元數原文沒寫 〔原文 §V / p.7〕。
- 自變數：真實鏈路（雲到邊平均 0.64 Gbps）與干擾裝置、併發 1–4、長度 10–38K、5 個模型、9 個資料集 〔原文 p.7–10〕。
- 對手：CacheGen、Strong Hybrid、Local Prefill 〔原文 §VI-A / p.8〕。Strong Hybrid 就是 Cake：早 chunk 本地算、晚 chunk 串流，串流段用一樣的量化＋Huffman 〔原文 p.8〕。
- 品質：LongBench／VideoMME 官方指標，每集 22–200 題 〔原文 Table III / p.7〕。品質參考是 Local Prefill，而它本身是稀疏 attention〔抽取工作的回報〕。
- 速度：TTFT 設 max_new_tokens＝1 〔原文 p.7〕。排程離線做，不在 TTFT 內〔判讀〕。
- 呈現：TTFT CDF 與品質 CDF 分開畫，沒有品質對延遲的取捨曲線 〔原文 Fig. 9、11 / p.8–9〕。
- 依位置：Fig. 3、Fig. 5 量各位置 chunk 的計算與串流時間 〔原文 p.3–4〕；結論是「位置是 chunk 成本的差勁代理」〔原文 §III-C / p.4〕。**沒有品質對位置**。

### 2.7 EvicPress、AdaptCache（UChicago）〔本次由抽取工作讀全文；關鍵 6 處我 grep 核對頁碼相符〕

- EvicPress：
  - 1×H100、80 GB DRAM、800 GB SSD 〔原文 p.8〕；5 個模型 〔原文 p.8〕。
  - 資料：LongBench 12 子集、555 個 context；每個 context 用 GPT-5 生 100 題，50 題 profile、50 題測 〔原文 Table 2 / p.7；p.8〕。
  - **品質不對標準答案**：用 MiniLM-L6-v2 算「壓縮 KV 的答案」和「未壓縮 KV 的答案」的 embedding 相似度 〔原文 p.9〕。
  - TTFT 含取 KV 與新問題的 prefill 〔原文 p.9〕。所以壓縮 KV 由新問題的 prefill 讀〔判讀〕。
  - 呈現：品質對 TTFT 曲線，**同品質比 TTFT、同 TTFT 比品質兩種都報** 〔原文 Fig. 7 / p.9〕。
  - **評估沒有量化**，量化列為未來工作 〔原文 §8 / p.12〕。重複：原文沒寫。
- AdaptCache（3 頁短文）：
  - 1×A100、100 GB DRAM、400 GB SSD、磁碟讀 1 GB/s 〔原文 p.2〕；只有 Llama-3.1-8B。
  - 資料：LongBench 6 子集、1,100 個 context；Poisson 到達 〔原文 p.2〕。
  - 品質和「原始 prefill 的答案」比 〔原文 p.1 註 1〕。profile 時每個資料集只抽 10 個 entry 〔原文 p.2〕。
  - 呈現：品質對 TTFT 圖，同品質比速度、同速度比品質 〔原文 Fig. 2 / p.2〕。
  - 既有卡把 KIVI 寫成 AdaptCache 的選項之一，原文裡 KIVI 只是基線〔抽取工作的回報〕。

### 2.8 FlexGen（ICML 2023，arXiv 2303.06865）〔本次由抽取工作讀評估節；4 句關鍵原文已由我 grep 核對〕

- 場景：KV 依比例分放 GPU、CPU、disk 〔原文 §4 / p.5–6〕。壓縮是為了讓權重和 KV 都放進 CPU、避開 disk 〔原文 §6.1 / p.8〕。沒有重算路徑（原文沒寫）。
- 自變數：壓縮只有一組設定，4-bit、group 64、權重與 KV 一起壓 〔原文 §5 / p.7〕。3-bit 試過「保不住準確率」，沒給數據 〔原文 §6.2 / p.9〕。長度 512／1024＋32 〔原文 §6 / p.7〕。
- 品質：OPT-30B／175B 的 Lambada acc 與 WikiText PPL（175B：0.758→0.756；10.82→10.94）〔原文 Table 5 / p.9〕。
- **速度和品質不是同一次 run**：吞吐用 dummy 權重 〔原文 §6 / p.7〕。量化 KV 在 prefill 還是 decode 被讀：原文沒寫。
- 速度：端到端吞吐，真 SSD＋mmap 〔原文 §6 / p.7〕。反量化沒有單獨拆；KV 壓縮和權重壓縮的速度沒有分開〔判讀〕。小模型開壓縮反而變慢（6.7B 在 3090 上 233.8→120.2 tok/s）〔原文 Table 12 / p.19〕。
- 硬體：GCP T4＋1.5 TB NVMe 〔原文 Table 1 / p.7〕；另有 3090 機器 〔原文 Table 12 / p.19〕。重複次數：原文沒寫。
- 位置：沒有。
- 原文內部不一致：SSD 頻寬正文寫讀約 2 GB/s，Table 24 寫 1.6 GB/s 〔原文 p.7；Table 24 / p.23〕。

### 2.9 QServe（MLSys 2025，arXiv 2405.04532）〔本次由抽取工作讀評估節；4 句關鍵原文已由我 grep 核對〕

- KV4 常駐 GPU，每 head 動態量化，scale／zero 存在 page 裡 〔原文 §5.1 / p.8〕。不是 offload，當對照。
- 天真 KV4 在 L40S 快、在 A100 上比 KV8 慢 1.2× 〔原文 §5.3 / p.10〕。Table 1 的數字是 0.86–0.90×，圖說寫 1.1–1.2×，正文寫 1.2×，三處說法略有出入〔抽取工作的計算，未獨立覆核〕。
- 品質：WikiText2 PPL、五個 zero-shot 任務、**LongBench 10 個子任務**（Llama-3.1-8B-Instruct）〔原文 Table 2–3 / p.11–12；Table 5 / p.13〕。品質走 HF／PyTorch 實作，吞吐走自己的 CUDA 系統 〔原文 p.10–11〕。
- 速度：同記憶體預算下的最大吞吐，實測 〔原文 §6.3 / p.12〕。有一個只加 KV4 的消融：L40S、Llama-2-7B，PPL 5.66→5.80，吞吐 1.47× 〔原文 Fig. 16 / p.13–14〕。
- 重複次數：原文沒寫。
- 既有卡 D6_QServe 要修：「0.86–0.90× 慢」應寫成「速度是 KV8 的 0.86–0.90 倍」；Table 1 可由 p.14 的 0.48 ms 對到 A100、batch 64；卡片漏了 Fig. 16 與 LongBench〔抽取工作的回報〕。

### 2.10 LMCache 技術報告（arXiv 2510.09665v2；MLSys 2026 頁面列為 oral〔文件 mlsys.org/virtual/2026/oral/3646〕）

- **評估不涵蓋壓縮**：controller 有 compress／decompress API，但原文說這些 API 沒有涵蓋在評估的應用裡 〔原文 §7 / p.10〕。
- 評估本身：6 個模型（Llama-3.1-8B／70B、Sao10K-L3-8B、Qwen2.5-Coder-32B、Qwen2.5-72B、Qwen3-Coder-480B-FP8）；資料＝模擬多輪 QA、LongBench、vLLM random；8×H100；指標 TTFT、ITL；基線 vLLM 0.10.2、vLLM CPU offload 0.11.0、兩個商用服務 〔原文 §8.1 / p.10–11〕。
- 多輪 QA 每個 query 約 10K token（Llama-8B 用 20K），輸出最多 100 token，使用者依 QPS 加入 〔原文 §8.2 / p.11〕。
- 長度敏感度：32 Gbps 時，載入要在輸入 ＞256K 才比 prefill 快 〔原文 §8.7 / p.14 / Fig. 15〕。
- 重複次數：原文沒寫〔本次只讀 §8〕。

---

## 3. 方法論論文怎麼說〔E08 卡的內容；本次抽取工作重讀全文，核對 R1–R10、Y1–Y8 頁碼無誤，並補了下面兩點〕

- **補 1**：兩篇都說量化 KV **不參與 prefill，只影響 decode** 〔原文 Rethinking §2.3 / p.3；Yuan §2.2 / p.3〕。〔判讀〕所以它們的量化品質數字不涵蓋「新問題的 prefill 讀量化 KV」這個我們的情境。
- **補 2**：Rethinking 的 Table 1 調查 41 個方法，35 個只在 HF transformers 上量；報 decode 吞吐的 22 個〔原文 Table 1 / p.4；抽取工作的計數〕。表裡沒有輸出長度、逐樣本、TP 這幾欄。
- **補 3**：兩篇都沒有信賴區間、標準差或顯著性檢定；也都沒評估「量化 KV 卸載後還原」、多輪、依位置給精度〔抽取工作的回報〕。兩篇都把全精度 recent window 當成麻煩（paging 變複雜、壓縮比隨長度變）〔原文 Rethinking p.4；Yuan p.5–6〕。


**Rethinking KV Cache Compression（MLSys'25，arXiv 2503.24000v1）**
- 吞吐要在有 PagedAttention／FlashAttention 的服務框架量，不要在 HF transformers 量（Missing Piece 1）〔原文 p.5，引自 E08 R1〕。
- 有損壓縮會改變輸出長度；固定輸出長度量吞吐是錯的（Missing Piece 2）〔原文 p.5，引自 E08 R3〕。
- 平均分數會藏住逐樣本的失敗，要看 negative samples（Missing Piece 3）〔原文 p.6，引自 E08 R4〕。
- 加速比依 batch、長度、TP 而定，有負加速的區域（Observation 2）〔原文 p.8，引自 E08 R8〕。KIVI-4 的 decode 吞吐只有 FP16 的 0.88–0.98×（TP 1／2／4）〔原文 Table 3 / p.8，引自 E08〕。
- 吞吐取 3 次平均 〔原文 p.16，引自 E08〕。
- 它自己用 LongBench 量品質 〔原文 p.9–10，引自 E08〕。

**Yuan'24（Findings of EMNLP 2024，arXiv 2407.01527v2）**
- 不同流派只能用「壓縮比」對齊；KIVI 有全精度 residual，壓縮比要指定參考長度 〔原文 Table 2 註 / p.5–6，引自 E08 Y1〕。
- 在 prefill 內壓縮和在 prefill 後壓縮差很多 〔原文 p.7–8，引自 E08 Y2〕。
- 第一個輸出 token 是「免費的」：7-digit passkey 讓 H2O 看起來 100%，64-digit 掉到 35% 〔原文 p.9，引自 E08 Y3〕。
- 效率難以 apple-to-apple 比較，所以它不量系統指標 〔原文 p.9，引自 E08〕。
- 任務覆蓋：LongBench 15 任務＋needle 〔原文 p.4–5，引自 E08〕。

兩篇都沒有評估「offload 後還原的量化 KV」、多輪、依位置的壓縮（見補 3）。

---

## 4. 慣例

### 4.1 標準做法（≥3 篇這樣做）

| 慣例 | 哪些論文 | 證據 |
|:--|:--|:--|
| 全精度當品質上限、均勻量化當主要對手 | CacheGen（3/4/8-bit）、Cake（16/8/3）、PatchKV（FP16／INT8／INT4）、Yuan'24（KIVI-2/4、FlexGen-4） | 〔原文 CacheGen §7.1 / p.8〕〔原文 Cake Table 7 / p.8〕〔原文 PatchKV p.8–9，引自既有卡〕〔原文 Yuan p.6，引自 E08〕 |
| 「只重算」當對手 | CacheGen（Text）、Cake（compute-only）、EvicPress、AdaptCache（Prefill）、SparKV（Local Prefill） | 各篇對手格 |
| 拿已發表的壓縮法當對手（KIVI、CacheGen、H2O） | CacheGen（H2O、LLMLingua）、KVServe（KIVI、CacheGen）、AdaptCache（KIVI）、SparKV（CacheGen） | 各篇對手格 |
| 品質用真實資料集（多半是 LongBench 系） | CacheGen（4 集）、KVServe（6 任務）、PatchKV（HotpotQA-E 等）、SparKV（9 集）、EvicPress（12 子集）、AdaptCache（6 子集）、QServe（LongBench 10 子任務）、Yuan、Rethinking | 各篇品質格 |
| 品質對「全精度的輸出」比，而不是對標準答案 | KVServe（相對 Default）、EvicPress（embedding 相似度）、AdaptCache（對原 prefill 答案）、HA-RAG（ROUGE 對 TurboRAG 輸出） | 〔原文 KVServe Table 1 / p.11〕〔原文 EvicPress p.9〕〔原文 AdaptCache p.1〕〔原文 HA-RAG p.7〕 |
| 掃頻寬 | CacheGen、Cake、KVServe、LMCache、SparKV（真實干擾） | 各篇自變數格 |
| 掃 context 長度 | CacheGen（0.1K–15K）、Cake（4K–16K）、LMCache（到數百 K）、SparKV（10–38K） | 〔原文 CacheGen Fig. 12 / p.10〕〔原文 Cake Table 5 / p.7〕〔原文 LMCache Fig. 15 / p.14〕〔原文 SparKV Fig. 15 / p.10〕 |
| GPU 負載或併發用「併發數、QPS、排程佔比」表示 | CacheGen（併發數 1/n）、Cake（token budget 佔比）、LMCache（QPS）、EvicPress（QPS）、AdaptCache（Poisson）、SparKV（1–4 請求） | 各篇自變數格 |
| 用多個模型（≥3） | CacheGen 3、Cake 4、KVServe 3、PatchKV 3、SparKV 5、EvicPress 5、LMCache 6 | 各篇硬體格 |
| 用資料中心 GPU | 多數；例外是 SparKV（筆電、Jetson）與 FlexGen（T4、3090） | 同上 |
| 品質對延遲（或大小）的取捨圖 | CacheGen Fig. 8–9、PatchKV Fig. 7、KVServe Fig. 5／10、EvicPress Fig. 7、AdaptCache Fig. 2 | 各篇呈現格 |
| 時間拆解含解碼／反量化 | CacheGen Fig. 14a、KVServe Fig. 15、PatchKV、SparKV（佔比） | 各篇速度格 |

### 4.2 少見（1–2 篇）

- **同時報「同品質比速度」與「同速度比品質」**：EvicPress、AdaptCache；CacheGen 主要是同品質比大小。
- **實測位元組**：CacheGen（位元流）、PatchKV（MiB）、KVServe（bytes 之比，KIVI 含 metadata）。Cake、FlexGen、QServe 只有名目位元。
- **真實儲存或網路**：其實有 5 篇，已接近慣例——PatchKV（真 H2D）、KVServe（真網路＋traffic shaping 限速）、SparKV（真 Wi-Fi）、FlexGen（真 SSD）、EvicPress（真 SSD）。Cake 是延遲注入；CacheGen、AdaptCache 原文沒寫。
- **重複與統計**：PatchKV 有 paired bootstrap 區間（Table II、Fig. 8）〔原文 p.8〕；CacheGen 一張圖寫 20 條 trace 平均；Rethinking 3 次平均。其他原文沒寫。
- **多輪**：LMCache 有模擬多輪 QA，但只量速度、不量壓縮。CacheGen 的 LongChat 是「對話歷史當 context」，仍是單一問題。
- **把重算和量化載入綁在一起**：Cake（只量速度）、SparKV（Strong Hybrid 基線）、PatchKV（重算集＋混合精度載入）。

### 4.3 沒人做

- **依 token／chunk 位置決定精度，並量品質**。沒有任何一篇。最接近的是 PatchKV 的「離編輯點距離」決定重算集，以及 SparKV 反過來主張位置不是好代理。
- **把「不敏感」和「沒被讀到」分開**（記錄每個 block 是重算、載入高精度、還是載入低精度）。沒有人報。
- **事先登記、雜訊底線、全面的配對區間**。只有 PatchKV 在部分表報 paired bootstrap 區間 〔原文 p.8〕；沒有一篇事先登記判準，也沒有一篇量雜訊底線〔判讀〕。
- **Cake 式並行還原下的品質**，而且重算段是精確的 BF16。Cake 本身沒量品質；SparKV 有量 Strong Hybrid 的品質，但它的本地重算是稀疏 attention，不是精確重算〔原文 p.8，抽取工作的回報〕。

---

## 5. 對照我們的計劃（[03_plan_h200.md](../03_plan_h200.md) §1–§5）

全部判定都是〔判讀〕。

| # | 我們的做法 | 慣例 | 判定 | 建議 |
|:--|:--|:--|:--|:--|
| 1 | 品質用 RULER 風格合成（T1–T3、T5）＋自建 HotpotQA 兩跳（T4）〔03 §3.1〕 | 真實資料集＋標準指標：CacheGen 4 個資料集；KVServe 用 LongBench 系的 HotpotQA、2WikiMQA、Qasper、Multi-News；PatchKV 用 HotpotQA-E 等；QServe 用 LongBench 10 子任務；Yuan、Rethinking 用 LongBench | **缺（部分）** | 審稿人會問「真實任務呢」。加一組 LongBench 子集（例：HotpotQA、2WikiMQA、Qasper、NarrativeQA，32K 內），只跑 E0／E1／E1s 端點當外部效度，不進位置判定。T4 用的是 HotpotQA 真文字，要在報告裡講清楚它是「重組」不是官方設定 |
| 2 | 對手：E0、E1（均勻 FP8）、E1s、R25；INT4 階才用 KIVI 式分組，而且**不留全精度窗口**〔03 §1、§2.1〕 | 會拿已發表方法當對手（KVServe、AdaptCache 比 KIVI；SparKV、KVServe 比 CacheGen） | **缺** | KIVI 的「最近 R 個 token 全精度」本身就是**已發表的位置規則**。審稿人一定會問 F25（壓前段、保後段）和它差在哪。建議加一個「KIVI 原版」配置：同存量位元組下，最後 R 個 token BF16、其餘低精度。至少在「均勻不夠」的那一階加 |
| 3 | 速度的 6 個配置 × 頻寬 × GPU 狀態；沒有明列只算、只載的時間〔03 §4.2〕 | Cake 每格都報「對只載＼對只算」兩個比值；CacheGen 有 Text 基線；PatchKV 有 Prefix Cache（只算）與 Direct KV Reuse（只載）〔原文 §V-A3 / p.6〕 | **缺（小）** | harness 已有 `compute_only`、`load_only`。每格多跑這兩個（各精度），照 Cake 格式報兩個比值。成本低 |
| 4 | 只有 32K 一個長度〔03 §1〕 | CacheGen、Cake、LMCache 都掃長度 | **缺** | 速度格子加 8K、16K（只量時間可用合成 token，Cake 的依據〔原文 §5.1 / p.5〕）。品質留在 32K |
| 5 | 端點兩個模型、主比較一個模型〔03 §1.1〕 | 多數 ≥3 個模型 | **缺（審稿會問）** | 已事先登記選模型規則，這點好。建議：如果「穩定差距」成立，在第二個模型上重跑 F25／B25／R25 的方向檢查（讀法 A 即可），否則結論只能寫「在一個模型上」 |
| 6 | 儲存層＝pinned 記憶體＋限速器；CPU 層是真實 H2D〔03 §1、§5.1〕 | Cake 是延遲注入；PatchKV 真 H2D、KVServe 真網路限速、SparKV 真 Wi-Fi、EvicPress 真 SSD | **符合 Cake，弱於多數** | 已有真實 CPU 錨點、也已改成「限速層 X GiB/s」的寫法，這符合誠實報告。建議之後補一個真 SSD 錨點（03 §5.1 已列為之後要另外登記） |
| 7 | GPU 忙＝背景 matmul，讓重算慢約 2 倍〔03 §1〕 | Cake 用 token budget 佔比；CacheGen 用併發數 1/n；LMCache 用 QPS | **偏離（已寫明）** | 可以接受，因為已在計劃寫明和 Cake 不同。建議另報一個「等效 GPU 佔比」，讓讀者能對到 Cake 的 12.5%／50%／87.5% |
| 8 | 沒有併發、沒有到達過程、沒有吞吐〔03 §4〕 | CacheGen、SparKV 有併發；LMCache、EvicPress、AdaptCache、Rethinking 有 QPS／Poisson；Cake 只有一個混合負載例子 | **缺（看投稿場合）** | 6 頁 workshop 可以把範圍寫成「單請求還原延遲」。投正式系統會議，審稿人會要求至少一張「併發下的 TTFT」 |
| 9 | 時間拆成編碼、D2H、寫入、讀取、H2D、反量化、等待、重算、新一輪〔03 §5.4〕 | CacheGen 有三段拆解；PatchKV 有 fetch＋反量化；Cake 沒有 | **更嚴** | 保留。QServe 顯示反量化成本依硬體變號〔原文 p.10，引自既有卡〕，3090 上單獨量是對的 |
| 10 | 限速器依實際位元組計時，FP8 block 含 scale〔03 §1〕 | CacheGen、PatchKV 用實測大小；Cake 只有名目位元 | **更嚴** | 保留。報告裡寫明 metadata 佔多少（Yuan 指出 KIVI residual 讓壓縮比難對齊〔引自 E08 Y1〕） |
| 11 | 配對 bootstrap CI、3 次重開 × 10 次、δ*、新題目重現、事先登記〔03 §2.4、§5.3〕 | 只有 PatchKV 在部分表報 paired bootstrap 區間 〔原文 p.8〕；其他多數沒寫重複 | **更嚴** | 保留。PatchKV 證明這一派的審稿人接受配對 bootstrap，引它當先例 |
| 12 | 量化 KV 在還原後被新問題讀到；還有第 2 輪〔03 §3.3–3.4〕 | CacheGen、EvicPress、SparKV 同樣是「新問題的 prefill 讀載入的 KV」；Rethinking、Yuan 的量化只影響 decode；多輪沒人做 | **符合＋更嚴** | 保留。和 vLLM 原生 FP8「第一次 prefill 就讀量化 KV」的差別，報告要寫（計劃已有） |
| 13 | 答案位置帶 × FP8 位置帶交叉；逐 block 記錄來源〔03 §2.2、§3.1〕 | 沒人做 | **新（無慣例可依）** | 這是貢獻所在。要預期 SparKV 式的反對：「位置是差勁代理」。03 的 S25（風險分數）就是在回答這點 |
| 14 | 主判準＝對標準答案的正確率，和 E0 配對比非劣性；log-prob、KL 只報告〔03 §2.4〕 | 標準指標（acc、F1、PPL）；不少系統論文只和全精度的輸出比（KVServe、EvicPress、AdaptCache、HA-RAG）；Rethinking 要看逐樣本失敗 | **符合，而且比「對全精度輸出比」嚴** | 加一張「逐題翻轉表」（E0 對、X 錯／E0 錯、X 對 的題數），對應 Rethinking 的 negative samples |
| 15 | 圖 A：每格品質對還原時間，畫 ε 線〔03 §8〕 | CacheGen Fig. 8 就是品質對 TTFT | **符合** | 保留。另外加一張「品質對存量位元組」，對應 CacheGen Fig. 9 |
| 16 | F25／B25／R25 存量位元組相同，比品質；速度用「不輸」〔03 §2.1、§2.4〕 | CacheGen「同品質比大小」；Yuan「同壓縮比比品質」 | **符合** | 保留 |
| 17 | 時間由自寫 harness（HF 模組＋自寫 attention）量〔03 §5.2〕 | Cake 用 vLLM＋LMCache；Rethinking 明說不要在 HF 量吞吐 | **風險（已有對策）** | §5.2 的「每 chunk ≤ vLLM(512) 的 1.2 倍」接受條件要真的達成並報。達不到時，時間結論只能當相對比較 |
| 18 | 精度只走兩階（FP8 → INT4），有條件才往下〔03 §2.5〕 | CacheGen 掃 3/4/8；Cake 16/8/3；Yuan 2/4 | **少於慣例（有理由）** | 停損設計合理。但審稿人可能想看一條「品質對位元」的曲線。若兩階都停，可在 E0／E1 上補一個 2-bit 端點當參考，不進判定 |
| 19 | 單張 RTX 3090 〔03 §1〕 | 資料中心 GPU | **偏離** | 報告寫明範圍。κ 的跨硬體主張本來就要靠第二平台，這週的數字不和 MI300X 混用（計劃已寫） |
| 20 | 輸出 greedy、最多 64 token；只量 TTFT 與還原時間〔03 §1、§5.4〕 | Rethinking：有損壓縮會改變輸出長度，端到端要連長度算 | **符合（TTFT 不受影響）** | 每題記輸出長度，報告附分布。不量端到端就明寫 |

### 5.1 最該補的五件事（依審稿風險排序）〔判讀〕

1. **已發表方法對手**：KIVI 原版（含最近窗口）就是位置規則，必須在同存量下比（第 2 列）。
2. **真實 benchmark**：LongBench 子集跑端點（第 1 列）。
3. **長度與模型只有一個**：速度加 8K／16K；主結論在第二個模型上做方向檢查（第 4、5 列）。
4. **只算／只載參考線**：每格都報，照 Cake 格式（第 3 列）。
5. **服務層面的真實性**：harness 對 vLLM 的差距要達標並報；正式會議需要一張併發下的 TTFT（第 8、17 列）。

---

## 6. 沒查到的

- 抽取工作自己算的數字（例：KVServe HotpotQA 95.8%、QServe 慢 1.11–1.16×、Rethinking Table 1 的 35／41）沒有獨立覆核。
- 「原文沒寫」最多的格：題數（KVServe、PatchKV、AdaptCache）、重複次數（幾乎全部）、metadata 是否算進大小（SparKV、EvicPress、AdaptCache、KVServe 自家）、頻寬是真實還是模擬（CacheGen、AdaptCache、LMCache）。
- 既有卡要修的地方（D6_KVServe、D6_PatchKV、D6_HARAG、D6_SparKV、D6_EvicPress、D6_AdaptCache、D6_QServe、E08、EVAL_FOUNDATIONS §8 第 9 列）已寫在第 2 節各篇的最後一點與抽取回報裡；**本次沒有改那些檔案**。
- FlexGen、QServe 的格子來自抽取工作（交件前收到），我只 grep 核對了 8 句關鍵原文，其餘數字沒有逐一覆核。
- 計劃檔 03_plan_h200.md 在我讀完之後被改過。第 5 節的對照依據的是改之前的版本，要再對一次。
- CacheGen：頻寬怎麼施加（原文沒寫）；每個 context 的題數（原文沒寫）；機率表是否算進大小（原文沒寫）。
- Cake：每格重複次數（原文沒寫）；量化實驗的 8-bit 實作（原文沒寫）。
- LMCache：重複次數（只讀 §8，原文沒寫）。
- 「沒人做依位置的精度實驗」這個結論只涵蓋本卡的 10 篇＋既有卡讀過的範圍（P01 附錄、P03、P05），**不是新穎性結論**。
- EVAL_FOUNDATIONS §8 第 9 列把 Yuan 整篇列為「長 context 其實 ≤8K」太寬：只有 Llama-3 的 LongBench 截到 7,500；Mistral／LongChat 到 31,500〔抽取工作的回報，原文 Table 3 / p.15〕。
