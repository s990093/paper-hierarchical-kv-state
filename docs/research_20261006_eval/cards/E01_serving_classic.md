# E01　經典 LLM serving 論文的評測方法＋指標字典

> 抽取者：子 agent E01（2026-10-06）。狀態：**已獨立複核（V01，2026-10-07）**，修正處標〔複核修正〕／〔複核補充〕，明細見檔尾「複核紀錄」。
> 規則依 `../README.md`：每格附出處與證據等級；〔判讀〕與原文理由分開寫。

## 範圍

這組論文定義了 sys-level LLM 實驗的「標準語言」：負載怎麼合成、到達過程、掃什麼、指標怎麼定義、主圖長什麼樣子。
本檔逐篇填評測卡，再整理成指標字典，最後判讀哪些慣例可以搬到「16K–512K 長 context、單請求為主的 KV 分層」研究。

**頁碼慣例**：一律是 PDF 頁（arXiv PDF 的第 n 頁）。Orca 另外附 OSDI 論文集頁碼（PDF 頁＋519）。

## 讀了哪些來源

| # | 論文 | 版本 | URL | 讀法 |
|:--|:--|:--|:--|:--|
| 1 | Orca（OSDI'22） | USENIX 論文集 PDF（19 頁，pp. 521–538） | https://www.usenix.org/system/files/osdi22-yu.pdf | 全文 |
| 2 | vLLM／PagedAttention（SOSP'23） | arXiv v1，2023-09-12，comment 寫「SOSP 2023」；16 頁 | https://arxiv.org/abs/2309.06180 | 全文（未與 ACM 版逐字對照） |
| 3 | SGLang／RadixAttention（NeurIPS'24） | arXiv v2，2024-06-06；20 頁含附錄 | https://arxiv.org/abs/2312.07104 | 全文（未與 NeurIPS camera-ready 對照） |
| 4 | Sarathi-Serve（OSDI'24） | arXiv v3，2024-06-17；含 Artifact Appendix | https://arxiv.org/abs/2403.02310 | 全文 |
| 5 | DistServe（OSDI'24） | arXiv v3，2024-06-06，comment 寫「OSDI 2024」；18 頁 | https://arxiv.org/abs/2401.09670 | 全文 |
| 6 | Splitwise（ISCA'24） | arXiv v2，2024-05-20；15 頁 IEEE 雙欄 | https://arxiv.org/abs/2311.18677 | 全文（逐欄抽字；圖中數值只採正文文字） |
| 7 | Etalon（v1 名為 Metron） | arXiv v2，2024-08-30；12 頁 | https://arxiv.org/abs/2407.07000 （v1 標題頁 https://arxiv.org/abs/2407.07000v1 顯示 "Metron: …"） | 全文 |

輔助查證：HF 模型 config（判定注意力類型），`https://huggingface.co/<model>/resolve/main/config.json`，查證 2026-10-06：facebook/opt-13b、NousResearch/Llama-2-7b-hf、NousResearch/Llama-2-70b-hf、mistralai/Mixtral-8x7B-v0.1、NousResearch/Meta-Llama-3-8B、NousResearch/Meta-Llama-3-70B、bigscience/bloom。

所有 PDF 與轉出的文字放在 scratchpad `E01/`，沒有寫進 repo。

## 本組的共同模式

1. **負載＝真實資料集的「長度分布」＋合成內容＋Poisson 到達。** vLLM 與 Sarathi-Serve 明說只取資料集的 input／output 長度，再用 Poisson 產生到達時間；〔複核修正〕DistServe 只寫「從資料集取樣請求」並用 Poisson 產生到達時間（§6.1 p9），**沒有說是否只用長度**（LongBench 註 4 說把 input「capped」到 2048，反而暗示送的是真實內容），原卡把它併入「只取長度」沒有原文依據；Orca 連長度都用均勻分布合成；Splitwise 用 production trace 的長度，叢集評估仍用 Poisson。幾篇明說內容不重要，並強制輸出長度（Orca 讓模型永不輸出 EOS；Splitwise 強制生成指定數量的 token）。例外只有 SGLang（用帶內容的程式型 benchmark，沒有到達過程）和 Etalon（每小時打一次公開 API）。**七篇的主結果都不是用真實到達時間重播出來的。**〔原文〕出自 Orca、vLLM、Sarathi-Serve、DistServe、Splitwise、SGLang、Etalon 卡。
2. **主張的形式都是「在某個延遲條件下，系統可撐的負載是對手的幾倍」**，主圖是延遲指標對負載的曲線，倍數取在對手已經飽和的膝點附近〔判讀；複核修正：「膝點附近」是讀圖推論，不是原文陳述；SGLang 與 Etalon 的主張也不是這個形式〕。條件從「相近的 normalized latency」（Orca、vLLM，沒有明確門檻）逐步變成「明確 SLO＋percentile 或 attainment」（Sarathi-Serve 的 capacity、DistServe 的 goodput、Splitwise 的九個 SLO、Etalon 的 fluidity capacity）。〔原文〕出自 Orca、vLLM、Sarathi-Serve、DistServe、Splitwise、Etalon 卡。
3. **同名但不同義的指標很多。** normalized latency：Orca 取中位數，vLLM 取平均；SGLang 圖上的 "Normalized" 則是相對某個參考系統正規化的長條，和前兩者無關。TTFT：DistServe 引言說是「prefill 階段的持續時間」，Sarathi-Serve 與 Etalon 明定從請求到達開始算、包含排程延遲。TBT：Sarathi-Serve 對每個 token 各算一次，Splitwise 的表寫的是「平均」。連以批評指標為主旨的 Etalon，也把 vLLM 的 normalized latency 誤述成中位數。七篇論文都沒有用 ITL 這個名詞。〔原文〕出自 Orca、vLLM、SGLang、Sarathi-Serve、DistServe、Splitwise、Etalon 卡。
4. **SLO 門檻都是作者自己訂的，沒有業界標準。** 比較好的做法是相對化：Sarathi-Serve 取「無干擾 decode iteration 時間」的 5 倍與 25 倍；Splitwise 用「相對 DGX-A100 無競爭時的 slowdown」；Etalon 讓第一個 token 的期限隨 prompt 長度擬合。DistServe 憑經驗設門檻，但另外用 SLO Scale 掃描，檢查結論對門檻是否穩健。〔原文〕出自 Sarathi-Serve、Splitwise、Etalon、DistServe 卡。
5. **長度上限落在 2K–16K（只有 Etalon 的動機圖到 32K），其中三篇被 2048 卡死；跨請求 KV 重用多半被刻意關掉。** 〔複核修正：原寫「長度都在 8K 以內」不成立——Sarathi-Serve 的 arxiv_summarization 只剔除總長 >16,384 的請求，prompt P90 為 12,985（Table 2 p10）；Etalon 開源評估用 rope-scaling 支援 >8192 的 prompt（§5.2 p8），Fig 1 到 32K（p4）、Fig 5c 軸到 32,768（p8）。〕Orca 與 OPT 的上限是 2048；vLLM 的 chatbot 實驗把 prompt 截到 1024；DistServe 把 LongBench 截到 2048；Sarathi-Serve 剔除總長超過 8K／16K 的請求；Splitwise 的中位數約 1–1.5K、Fig 5a 掃到 8,192。vLLM 的 chatbot 實驗不保留輪與輪之間的 KV；Splitwise 說「在特性化（§III）中」不重用 KV〔複核修正：原文限定 "For this characterization"（p3），叢集評估沒有重述〕。只有 SGLang 把 KV 重用（hit rate）當成主角。〔原文〕出自各卡「長度」與「重用結構」兩格。
6. **七篇都沒有報誤差棒或信賴區間；除了 Etalon 的公開 API 每小時量一次、共 24 次（p7）以外，也都沒有報重複次數。**〔複核修正：原寫「都沒有報重複次數」與 Etalon 卡自己的「重複與統計」格矛盾〕 大規模的結論靠模擬器，並自報模擬器的驗證誤差：DistServe 的 SLO attainment 誤差小於 2%；Splitwise 的效能模型 MAPE 小於 3%。〔原文〕出自全部卡的「重複與統計」一格。

---

## 評測卡

### Orca：A Distributed Serving System for Transformer-Based Generative Models（OSDI 2022；無 arXiv）

- **讀了什麼**：〔全文〕USENIX OSDI'22 論文集 PDF，https://www.usenix.org/system/files/osdi22-yu.pdf ，查證 2026-10-06
- **一句話**：以 iteration-level scheduling 加 selective batching，讓請求可以逐 iteration 進出 batch。
- **評測要證明的主張**：在 GPT-3 175B 上，同一延遲水準下，吞吐比 FasterTransformer 高 36.9 倍（abstract，PDF p2）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | GPT-3 架構的 13B、101B、175B、341B。13B 與 175B 取自 GPT-3 論文，101B 與 341B 是作者自己改層數與 hidden size 得到的（層數 40／80／96／120；hidden 5120／10240／12288／15360）。最大序列長度 2048；參數與中間 activation 為 fp16。**作者沒有實際的 checkpoint**，權重怎麼產生沒有說明。注意力類型沒有寫〔未查證〕。 | 〔原文〕§6 PDF p11（p530）；Table 1 PDF p10（p529） |
| 硬體 | Azure ND96asr A100 v4 VM，每台 8 張 A100-40GB，以 NVLink 互連；最多用 4 台。每台有 8 張 200 Gbps HDR InfiniBand，VM 間共 1.6 Tb/s。平行化：13B 單卡；101B 1 個 inter × 8 個 intra；175B 2×8，共 16 GPU；341B 4×8，共 32 GPU。DRAM 與 SSD 沒有寫。 | 〔原文〕§6 PDF p11；Table 1 PDF p10；Fig 10 caption PDF p12 |
| 軟體與版本 | Orca 本體是 13K 行 C++，建在 CUDA 上；控制面用 gRPC，資料面用 NCCL；LayerNorm、Attention、GeLU 有融合 kernel。FasterTransformer 的版本沒有寫〔未查證〕。 | 〔原文〕§5 PDF p10 |
| 資料／負載 | **沒有公開 trace，全部合成。** ① Engine microbenchmark：同一個 batch 內每個請求的 input 都是 32 或 128、輸出都是 32；不跑 scheduler，反覆把同一個 batch 送進 engine，模擬 request-level 排程。② 端到端：input ~ U(32, 512)，max_gen_tokens ~ U(1, 128)；**強制不輸出 EOS**，每個請求都生成到 max_gen_tokens。③ 同質 trace：(input, gen) = (32, 32) 與 (256, 256)。 | 〔原文〕§6 PDF p11；§6.1 PDF p11；§6.2 PDF p12；Fig 11 PDF p13 |
| 長度 | 單位 token。input 32–512，output 1–128，序列上限 2048。 | 〔原文〕PDF p11–12 |
| 到達與併發 | Poisson，靠改變到達率來調負載。Orca 的 max batch size 取 {1, 8, 16, 32}。FasterTransformer 的 (max_bs, mbs) 試過所有組合，圖上只畫表現最好的 (1, 1) 與 (8, 8)。KV 依每個請求的 max_tokens 預先保留 slot。 | 〔原文〕PDF p11；PDF p12–13；§4.2 Algorithm 1 PDF p9–10 |
| 重用結構 | 無。 | 〔原文〕 |
| 掃描的自變數 | 到達率；max batch size；microbatch size（只有 FasterTransformer 有）；模型大小。microbenchmark 另掃 batch size 1–32 與 input 32／128。 | 〔原文〕§6.1–6.2 PDF p11–13 |
| 對手 | FasterTransformer，搭配作者自己寫的 scheduler：每次從佇列取最多 max_bs 個請求動態組 batch，類似 Triton 與 TF Serving 的做法。不比 Megatron-LM 與 DeepSpeed，理由是它們主要為訓練設計。 | 〔原文〕§6 PDF p11；§6.2 PDF p12 |
| 系統指標 | microbenchmark：處理整個 batch 的時間（ms）。端到端：**median normalized latency**，也就是每個請求的端到端延遲除以它生成的 token 數，再取**中位數**，單位 ms/token；對應的 x 軸是 throughput（req/s）。同質 trace 不做正規化。沒有尾端 percentile，沒有 SLO。 | 〔原文〕Fig 10 caption PDF p12；§6.2 PDF p12；Fig 11 caption PDF p13 |
| 品質指標 | 無。權重與文字都不是真實的。 | 〔原文〕PDF p11 |
| 主要結果 | 175B：要達到 median normalized latency 190 ms（orca(128) 正規化執行時間的兩倍），FasterTransformer 只有 0.185 req/s，Orca 有 6.81 req/s，即 36.9 倍。101B：FasterTransformer 的峰值吞吐 0.49 req/s；低負載時兩者差不多。175B 的 engine microbenchmark 在關掉 pipelining 時，Orca 最多快 47%。 | 〔原文〕§6.2 PDF p13（p532）；§6.1 PDF p12 |
| 主圖形狀 | Fig 10：x 軸標為 "Throughput (req/s)"，y 軸是 median normalized latency（ms/token，log 刻度，軸刻度 10²、10³），每個系統、每種 max_bs 各一條曲線；主張是「同樣延遲下的吞吐比」。〔複核修正：原文軸名只寫 Throughput，沒有說是「達成的」吞吐；下一句整句屬〔判讀〕〕x 軸應是**達成的吞吐**，不是送入的請求率；vLLM 用的是送入的請求率〔判讀〕。 | 〔原文〕Fig 10 PDF p12 |
| 消融／敏感度／開銷 | Engine microbenchmark（Fig 9）；改 max batch size；同質 trace（Fig 11）。 | 〔原文〕PDF p11–13 |
| 重複與統計 | 未說明。 | 〔原文〕全文未提 |
| 程式碼／資料 | 論文中沒有程式碼連結。vLLM 論文說 Orca 沒有公開〔二手：vLLM PDF p10〕。 | 〔原文〕；〔二手〕 |
| 設計理由（原文） | ① 沒有公開的生成式 LM 請求 trace，所以自己合成（PDF p11）。② 沒有 checkpoint 也沒有真實文字，猜不到何時會出 EOS，所以強制生成到 max_gen_tokens（PDF p11）。③ 每個請求的處理時間大致與生成 token 數成正比，所以用生成 token 數正規化延遲（PDF p12）。④ microbenchmark 不跑 scheduler，以便單獨評估 engine（PDF p11）。⑤ 36.9 倍取在 orca(128) 正規化執行時間的兩倍這個延遲水準（PDF p13）。 | 〔原文〕 |
| 設計理由〔判讀〕 | normalized latency 把 prefill 成本攤到每個輸出 token 上。input 只有 32–512 時這樣做還算合理；到了長 context，prefill 主導總延遲，這個指標就會失真（見 Etalon 卡）。「強制輸出長度」讓 decode 的工作量可以控制，這是後來「只看長度」評測法的源頭。 | 〔判讀〕 |
| 原文沒講清楚的地方 | 權重怎麼產生；每個負載點跑多少請求、trace 多長；FasterTransformer 版本；normalized latency 的端到端延遲是否包含排隊（既然叫端到端，推定包含〔判讀〕）；Poisson 的 seed。 | 〔原文〕 |
| 與既有整理不一致 | 既有檔都沒有收 Orca（在 workloads_eval、PAPERS_BY_LEVEL、intro.txt、sota.txt 中搜尋 "Orca"，0 筆）。 | 〔計算〕grep，2026-10-06 |
| 對本研究的意義〔判讀〕 | 「同延遲下的吞吐倍數」這種主張是從這篇開始的。長 context、單請求的 PoC 不應該用 normalized latency。可以借用「強制輸出長度」來固定 decode 的工作量，讓 TTFT 的差異只來自還原與重算。 | 〔判讀〕 |

### vLLM：Efficient Memory Management for Large Language Model Serving with PagedAttention（SOSP 2023；arXiv 2309.06180）

- **讀了什麼**：〔全文〕arXiv v1（2023-09-12，comment 寫「SOSP 2023」），https://arxiv.org/abs/2309.06180 ，查證 2026-10-06
- **一句話**：仿照 OS 的分頁，用 PagedAttention 管理 KV，消除碎片並支援共享。
- **評測要證明的主張**：同樣延遲下，吞吐比 FasterTransformer 與 Orca 高 2–4 倍；序列越長、模型越大、解碼演算法越複雜，改善越明顯（abstract p1）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | OPT-13B／66B／175B，以及 LLaMA-13B（只用在 shared prefix 實驗）。OPT 是 MHA：HF config 為 40 個 attention head、沒有 KV head 欄位、max_position_embeddings 2048。原文沒寫權重與 KV 的 dtype；Table 1 中 13B 的參數佔 26 GB，可推得是 2 bytes/參數，即 FP16〔計算：13B×2B≈26GB〕。 | 〔原文〕§6.1 p10、Table 1 p9；〔文件〕HF facebook/opt-13b config |
| 硬體 | GCP A2 instance。13B 用 1 張 A100-40GB；66B 用 4 張 A100（共 160 GB）；175B 用 8 張 A100-80GB。KV 可用記憶體分別是 12／21／264 GB，最多可放 15.7K／9.7K／60.1K 個 token 的 KV。swap 到 CPU RAM；RAM 大小與 SSD 沒有寫。 | 〔原文〕Table 1 p9；§6.1 p10；§4.5 p8 |
| 軟體與版本 | vLLM 本體 8.5K 行 Python 加 2K 行 C++／CUDA；FastAPI 前端，延伸 OpenAI API；NCCL；模型實作基於 PyTorch 與 Transformers。 | 〔原文〕§5 p9 |
| 資料／負載 | ShareGPT（引用 sharegpt.com）與 Alpaca（Stanford Alpaca repo）。**先 tokenize，再只取 input／output 長度來合成請求。** ShareGPT 的 input 平均 161.31、output 平均 337.99 token；Alpaca 為 19.31／58.45。ShareGPT 的 input 平均長 8.4 倍、output 長 5.8 倍，變異也較大。**原文完全沒有描述過濾、截斷、取樣筆數，也沒寫 ShareGPT 用哪個版本。** 常被說成「vLLM 論文的 ShareGPT 過濾規則」的東西其實在 benchmark 程式碼裡，不在論文中（見 E02）。其他實驗的負載：parallel sampling 與 beam search 用 Alpaca＋OPT-13B；shared prefix 用 WMT16 英翻德＋LLaMA-13B，前綴為 1-shot（80 token）或 5-shot（341 token）；chatbot 用 ShareGPT 合成對話歷史與提問，prompt 截到最後 1024 token，最多生成 1024 token。 | 〔原文〕§6.1 p10；Fig 11 p9；§6.3 p11；§6.4 p11–12、Fig 16；§6.5 p12 |
| 長度 | 單位 token。平均長度見上一格。模型上限 2048（Orca(Max) 就是預留到 2048）。chatbot 的 prompt ≤1024。 | 〔原文〕p9–p10、p12 |
| 到達與併發 | 資料集沒有時間戳，所以用 Poisson 產生到達時間，掃不同請求率。trace 長 1 小時；OPT-175B 因成本只跑 15 分鐘。FasterTransformer 的最大 batch B 依 GPU 記憶體盡量設大。 | 〔原文〕§6.1 p10 |
| 重用結構 | basic sampling 沒有重用。parallel sampling 與 beam search 在同一個請求內共享 prompt 的 KV。shared prefix 用服務端預先保留的固定前綴（§4.4）。chatbot 實驗**不保留輪與輪之間的 KV**。 | 〔原文〕§6.3 p11；§4.4 p8；§6.5 p12 |
| 掃描的自變數 | 請求率；模型與 GPU 數；資料集；parallel size 2／4／6 與 beam width 2／4／6；block size；swap 或重算。 | 〔原文〕Fig 12、14 p10–11；§7.2–7.3 p12–13 |
| 對手 | ① FasterTransformer，搭配作者寫的動態批次 scheduler（每次取最早到的 B 個請求）。② **Orca 由作者自行重做**（因為 Orca 沒有公開），假設用 buddy allocation，分三個版本：Oracle 知道真實輸出長度，是做不到的上界；Pow2 最多多預留 2 倍；Max 一律預留到 2048。 | 〔原文〕§6.1 p10 |
| 系統指標 | **normalized latency＝每個請求的端到端延遲除以它的輸出長度，再取平均（mean）**，原文註明沿用 Orca（但 Orca 取的是中位數，見 Orca 卡）。其他：Fig 13 為平均同時 batch 的請求數；Fig 15 的記憶體節省率＝因共享省下的 block 數 ÷ 不共享時的總 block 數；Fig 2 為 KV 記憶體中實際存放 token 狀態的比例。全文沒有 TTFT、TPOT、percentile，也沒有 SLO。 | 〔原文〕§6.1 p10；p11；p2；〔計算〕全文搜尋 TTFT／TPOT／P99 皆 0 筆 |
| 品質指標 | 無。 | 〔原文〕 |
| 主要結果 | ShareGPT 上，在相近延遲下可撐的請求率：是 Orca(Oracle) 的 1.7–2.7 倍、Orca(Max) 的 2.7–8 倍、FasterTransformer 的最多 22 倍。OPT-13B 在 ShareGPT 2 req/s 時，同時 batch 的請求數是 Orca(Oracle) 的 2.2 倍、Orca(Max) 的 4.3 倍。Alpaca 上 beam width 6 時，對 Orca(Oracle) 的優勢從 basic sampling 的 1.3 倍升到 2.3 倍。shared prefix：1-shot 1.67 倍、5-shot 3.58 倍。chatbot：2 倍。既有系統的 KV 記憶體只有 20.4–38.2% 真的存了 token 狀態。 | 〔原文〕§6.2 p11；Fig 13 p10–11；§6.3 p11；§6.4 p12；§6.5 p12；§1 p2 |
| 主圖形狀 | Fig 12、14、16、17：x 軸是請求率（req/s），y 軸是 normalized latency（s/token）。曲線先緩慢上升，請求率超過容量後突然爆炸；原文的解釋是佇列無限增長。比較的是「爆炸之前能撐的請求率」。 | 〔原文〕p10–12；解釋見 §6.2 p11 |
| 消融／敏感度／開銷 | attention kernel 比 FasterTransformer 慢 20–26%。block size：ShareGPT 上 16–128 最好；Alpaca 上 16 與 32 好，更大就變差；預設 16。swap 或重算：block 小時 swap 開銷很大；重算的開銷不超過 swap 的 20%；block 在 16–64 時兩者端到端相近。OPT-175B＋Alpaca 的優勢縮小，因為 KV 空間充足，系統變成算力瓶頸而不是記憶體瓶頸。 | 〔原文〕§7.1–7.3 p12–13；§6.2 p11 |
| 重複與統計 | 未說明。 | 〔原文〕 |
| 程式碼／資料 | https://github.com/vllm-project/vllm | 〔原文〕abstract p1 |
| 設計理由（原文） | ① ShareGPT 與 Alpaca 含有真實 LLM 服務的輸入與輸出。② 沒有時間戳，所以用 Poisson。③ normalized latency 沿用 Orca；高吞吐的系統應該在高請求率下仍維持低 normalized latency。④ 175B 因成本只跑 15 分鐘。⑤ Orca 沒有公開，所以重做三個版本，涵蓋不同的預留策略。⑥ chatbot 不保留 KV，因為輪與輪之間會佔住其他請求的空間。⑦ 13B 與 66B 是 leaderboard 上常見的大小，175B 是 GPT-3 的大小。 | 〔原文〕§6.1 p10；§6.5 p12 |
| 設計理由〔判讀〕 | 這篇要證明的是「記憶體碎片限制 batch 大小」，所以只需要長度。只取長度會丟掉內容與前綴身分，因此**無法評估 KV 重用**。ShareGPT 的平均 input 只有 161 token，結論停在短 prompt 區間。mean 會被少數極長延遲拉高，所以 vLLM 與 Orca 的 normalized latency 不能直接比。 | 〔判讀〕 |
| 原文沒講清楚的地方 | ShareGPT 的版本與清理方式；是否強制輸出長度（忽略 EOS）；每個點跑多少請求；端到端延遲的起點是否包含排隊；Poisson 的 seed；「可撐的請求率」怎麼從曲線上讀出來（沒有給門檻）。 | 〔原文〕 |
| 與既有整理不一致 | ① intro.txt 表 8 把 vLLM [40]（即這篇 SOSP 論文）列為「頁面替換：LRU、ARC；Belady 上界」的代表。論文裡的逐出其實是**以序列為單位的 all-or-nothing**：在 FCFS 下最晚到的請求先被搶佔，再用 swap 到 CPU 或重算來恢復（§4.5 p8）。論文中沒有 LRU 或 ARC。LRU 見於 SGLang §3（p4）與後來 vLLM 的程式碼（本卡未查證程式碼）。建議改引用，或註明「論文版是序列級搶佔」。〔複核：✅ 指控成立，但要收窄——vLLM 論文 §4.5 確實把這段寫成經典的逐出問題（"Which blocks should it evict?"，p8），所以把 [40] 列在「空間不夠時的逐出」這一欄是可以的；不成立的只是把 LRU／ARC 與 [40] 配在同一列。〕② workloads_eval §6.3 C.2 的「vLLM 原生 `lru`、`arc`」指的是現行程式碼的 offloading policy，不是 SOSP 論文〔判讀〕。〔複核：② 不是錯誤指控，只是澄清；workloads_eval L531 本身沒有引用 SOSP 論文，✅〕 | 〔原文〕vLLM §4.5 p8；intro.txt 表 8（intro.txt L397–419） |
| 對本研究的意義〔判讀〕 | 「指標對請求率作圖，看膝點」是 sys 論文的標準圖，但它預設有很多請求並行。16K–512K 的單請求 PoC 應該改成「指標對 context 長度作圖」。三個版本的 Orca（Oracle／Pow2／Max）是「上界＋實務做法＋最差」這種對手設計的好範本，可以對應到 Tiara 的 Oracle、heuristic 與 naive。§7.3 的 swap 或重算取捨隨 block size 變化，是 κ 這個概念的早期版本。§6.2 的 175B＋Alpaca 例子顯示：記憶體不吃緊時，記憶體管理的收益**明顯縮小**〔複核修正：原寫「就消失」；原文是 vLLM 對 Orca(Oracle)／(Pow2) 的優勢 "less pronounced"，§6.2 p11〕，這與 workloads_eval 的斷言「工作集必須大於 GPU＋CPU 容量」一致。 | 〔判讀〕 |

### SGLang：Efficient Execution of Structured Language Model Programs（NeurIPS 2024；arXiv 2312.07104）

- **讀了什麼**：〔全文〕arXiv v2（2024-06-06），https://arxiv.org/abs/2312.07104 ，查證 2026-10-06
- **一句話**：前端語言加上 runtime（RadixAttention、compressed FSM、API speculative execution），加速 LM 程式。
- **評測要證明的主張**：在多種 LM 程式上，吞吐最多高 6.4 倍、延遲最多降 3.7 倍（abstract p1；§6.2 p7）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | Llama-2 7B–70B、Mixtral-8x7B、LLaVA-v1.5-7B（圖）、LLaVA-NeXT-34B（影片），以及 GPT-3.5（API）。開放權重的模型用 float16。注意力類型依 HF config：Llama-2-7B 為 32 個 query head／32 個 KV head，即 MHA；Llama-2-70B 為 64／8，Mixtral 為 32／8，即 GQA。 | 〔原文〕§6.1 p7；〔文件〕HF config |
| 硬體 | AWS EC2 G5（A10G 24GB）。7B 單卡，較大的模型多卡 TP；部分實驗用 A100 80GB。逐圖細節：Fig 5／6 為 Llama-7B 單張 A10G；Fig 7 為 Mixtral 8 張 A10G TP；Fig 12 為 Llama-70B 4 張 A100 TP；LLaVA-7B 單張 A10G；LLaVA-NeXT-34B 單張 A100。 | 〔原文〕§6.1 p7；App C p18 |
| 軟體與版本 | PyTorch，加上 FlashInfer 與 Triton 的 kernel。對手：Guidance v0.1.8（llama.cpp 後端）、vLLM v0.2.5（預設 API server）、LMQL v0.7.3（HF Transformers 後端）。**不開任何會改變計算結果的優化**。 | 〔原文〕§6、§6.1 p7 |
| 資料／負載 | 5-shot MMLU（只 decode 1 token）；20-shot HellaSwag（用 select 選機率最高的選項）；ReAct agent 與 generative agents（從原論文抽出 trace 重播）；Tree-of-thought（GSM-8K）；Skeleton-of-thought（生成 tips）；LLM judge（branch-solve-merge）；JSON decoding（regex schema）；多輪聊天（4 輪，每輪 input 隨機取 256–512 token；short 版輸出 4–8 token，long 版 256–512）；DSPy RAG（官方範例）。多模態：llava-bench-in-the-wild、ActivityNet。開銷測試：ShareGPT 100 個請求。各 benchmark 的筆數未說明。 | 〔原文〕§6.1 p7；p8；§6.3 p9 |
| 長度 | 只有多輪聊天給了長度（每輪 256–512 token）；其他沒有給長度分布。 | 〔原文〕p7 |
| 到達與併發 | **沒有到達過程。** 吞吐：一次送出「夠大批」的程式實例，量最大吞吐。延遲：一次只跑一個程式、不 batch，取多個實例的平均。 | 〔原文〕§6.1 Metrics p7 |
| 重用結構 | **這是本篇的核心**：few-shot 範例共用；HellaSwag 兩層共享（範例與題目前綴）；agent 的模板與先前的呼叫；多輪歷史；RAG 的共用 context；同一張圖被問多個問題。 | 〔原文〕§6.2 p8 |
| 掃描的自變數 | benchmark × 模型 × 系統。消融實驗在 runtime 部分停用已匹配的 token，藉此**把 cache hit rate 當自變數來掃**。 | 〔原文〕Fig 8a/b p9 |
| 對手 | Guidance、vLLM、LMQL；多模態用作者原始的 HF 實作。vLLM 刻意用舊版，因為新版已經部分整合了 RadixAttention。 | 〔原文〕p7 註腳；p8 |
| 系統指標 | 吞吐＝programs per second（p/s）；延遲＝單一程式的平均延遲。Fig 5–7 的 "Normalized" 是**相對某個參考系統正規化的長條**（caption 只寫 Higher／Lower is better），參考系統是誰原文沒說明；**這和 Orca／vLLM 的 per-token normalized latency 不同**。cache hit rate＝已快取的 prompt token 數 ÷ prompt token 數。Fig 8 另有 first token latency、total latency、batch size、throughput（tokens/s）。 | 〔原文〕p7；Fig 5–7 caption p7–8；§3 p5；Fig 8 p9 |
| 品質指標 | 不開會改變結果的優化，讓所有系統算出相同結果（p7）；全文沒有品質量測。〔複核修正：「因為……所以不量品質」的因果是推論，原文只說前半句〔判讀〕〕API speculative execution 只說準確度高，input token 成本約降為三分之一。 | 〔原文〕p7；p9 |
| 主要結果 | 吞吐最多 6.4 倍、延遲最多降 3.7 倍。hit rate 介於 50–99%；cache-aware 排程平均達到最佳 hit rate 的 96%。Chatbot Arena 部署一個月：LLaVA-Next-34B 命中 52.4%、Vicuna-33B 命中 74.1%，後者的 first-token latency 平均降 1.7 倍。RadixAttention 的開銷：ShareGPT 100 個請求共 74.3 s，其中只有 0.2 s 用在維護樹（不到 0.3%）。 | 〔原文〕p7–8；Fig 13 p19；§6.3 p9 |
| 主圖形狀 | 每個 benchmark 一組正規化的吞吐或延遲長條（Fig 5–7）。Fig 8a/b：x 軸是 cache hit rate（%），y 軸是延遲、吞吐或 batch size。Fig 8c：元件消融的長條。 | 〔原文〕p8–9 |
| 消融／敏感度／開銷 | 拿掉快取、不用樹（改用表格式快取）、改 FCFS 排程、改隨機排程、關掉前端平行、關掉前端 hint（Fig 8c）。compressed FSM 讓吞吐高 1.6 倍；若每個請求都重做 FSM 前處理，吞吐低 2.4 倍。 | 〔原文〕§6.3 p9 |
| 重複與統計 | 未說明。只說每根長條要跑幾分鐘到一小時。 | 〔原文〕App C p18 |
| 程式碼／資料 | https://github.com/sgl-project/sglang | 〔原文〕abstract p1 |
| 設計理由（原文） | ① 不開會改變結果的優化，讓所有系統算出相同的結果（p7）。② vLLM 用舊版，因為新版已部分整合 RadixAttention（p7）。③ 用 hit rate 作為中介變數來解釋效能（p9）。④ cache-aware 排程（最長共享前綴優先，等同 DFS）在快取 ≥ 最大請求長度時離線最優（Thm 3.1，p6）。 | 〔原文〕 |
| 設計理由〔判讀〕 | SGLang 評的是「程式結構」，不是「服務負載」：沒有到達過程，吞吐是飽和批次下量的，延遲是無競爭的單一程式。它的數字與 vLLM 那種掃請求率的評測不能比。它的 hit rate 是 token 加權的，與 workloads_eval 的 H1（Marconi 定義）同型；而且 page size＝1 token（§3 p4〔複核修正：原寫 p5〕），不受 block 粒度影響。 | 〔判讀〕 |
| 原文沒講清楚的地方 | 每個 benchmark 跑幾個實例；正規化的基準是誰；hit rate 是否包含 decode 產生的 token（定義只提 prompt token）；LRU 逐出時的容量設定。〔複核補充：容量有部分答案——§3 p4 說不預先配置固定大小的 cache pool，快取與執行中的請求共用同一個記憶體池，等待的請求夠多時會把快取全部逐出以換取更大的 batch；所以快取容量是動態的，實驗時實際可用多少仍未說明〕 | 〔原文〕 |
| 與既有整理不一致 | intro.txt 表 8 把 SGLang 列在 LRU 頁面替換之下，與 §3（LRU，先逐出葉節點）一致。PAPERS_BY_LEVEL L361 說「只認從第一個 token 起完全相同的前綴」，與 radix tree 的前綴匹配一致〔判讀〕。沒有發現錯誤。〔複核：✅ 兩處都與原文一致〕 | 〔原文〕§3 p4（LRU、先逐出葉節點）〔複核修正：原寫 p5〕；PAPERS_BY_LEVEL L361 |
| 對本研究的意義〔判讀〕 | 七篇中唯一直接量 KV 重用的經典論文，它的 hit rate 定義可以沿用。「部分停用已匹配的 token 來掃 hit rate」是把命中率當自變數的好方法，可以借來掃「各層命中比例」。缺點是負載太短，最多幾千 token。 | 〔判讀〕 |

### Sarathi-Serve：Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve（OSDI 2024；arXiv 2403.02310）

- **讀了什麼**：〔全文〕arXiv v3（2024-06-17，含 Artifact Appendix），https://arxiv.org/abs/2403.02310 ，查證 2026-10-06
- **一句話**：用 chunked-prefills 加 stall-free batching，在不拖慢 decode 的前提下塞進 prefill。
- **評測要證明的主張**：在尾端 TBT 的 SLO 下提高 capacity：相對 vLLM，Mistral-7B（1 張 A100）2.6 倍、Yi-34B（2 張 A100）最多 3.7 倍；Falcon-180B 搭配 pipeline parallelism 最多 5.6 倍（abstract p1）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | Mistral-7B（GQA，含 sliding window）、Yi-34B（GQA）、LLaMA2-70B（GQA）、Falcon-180B（GQA）。dtype 沒有寫〔未查證〕。 | 〔原文〕Table 1 p10 |
| 硬體 | Azure NC96ads v4：每台 4 張 A100-80GB，兩兩以 NVLink 相連，節點間 100 Gbps 乙太網路。LLaMA2-70B 用一台 8 張 A40-48GB 的機器。平行化：Mistral 單卡；Yi-34B 2-way TP；LLaMA2-70B 與 Falcon-180B 為 TP4-PP2，Falcon 跨兩個節點。 | 〔原文〕Table 1、§5 p10 |
| 軟體與版本 | 以 vLLM 的 fork 為基礎；paged chunked prefill 支援 FlashAttention v2 與 FlashInfer，評測一律用 FlashAttention；NCCL；自建 telemetry。CUDA 12.1。token budget 由 Vidur 模擬器決定。 | 〔原文〕§4.4 p9–10；App A p18；§4.3 p9 |
| 資料／負載 | openchat_sharegpt4（多輪對話，每一輪當成獨立請求）與 arxiv_summarization。**用資料集的長度特性產生 trace。** 過濾：總長超過 8192（sharegpt4）與 16384（arxiv）的請求剔除。長度（Table 2）：sharegpt4 的 prompt 中位數 1730、P90 5696、std 2088，output 415／834／101；arxiv 的 prompt 7059／12985／3638，output 208／371／265。 | 〔原文〕§5 Workloads p10；Table 2 p10 |
| 長度 | 單位 token。總長上限為 8K（sharegpt4）與 16K（arxiv）。 | 〔原文〕p10 |
| 到達與併發 | Poisson。Fig 1 與 Table 4 用 128 個請求。 | 〔原文〕p10；Fig 1 p1；Table 4 p13 |
| 重用結構 | 無：每一輪都當成獨立請求，原文沒有提 prefix caching〔判讀〕。 | 〔原文〕p10 |
| 掃描的自變數 | 負載（QPS），用來找 capacity；SLO 分 strict 與 relaxed；Fig 12 掃 5 個 P99 TBT SLO 值，vLLM 的 max batch 取 32／64／128，Sarathi 的 token budget 取 512 與 2048；Falcon 比較 TP 與 PP；chunk size 512／1024／2048 對不同 prefill 長度（Fig 14）。 | 〔原文〕§5.1–5.4 p10–13 |
| 對手 | vLLM 與 Orca，理由是兩者代表當時的 SOTA。兩者都是 FCFS 的 iteration-level batching、會積極接納 prefill；差別在 batch 組成：Orca 可混 prefill 與 decode，vLLM 只允許全 prefill 或全 decode（§3.2）。對手怎麼實作（是否在自家 fork 裡以排程策略實作）原文沒有明說；§4.4 說擴充了 vLLM 程式碼以支援多種排程策略，推測是如此〔判讀〕。 | 〔原文〕p10；§3.2 p6；§4.4 p9–10 |
| 系統指標 | **TTFT**＝從請求到達系統開始，到產生第一個輸出 token 的延遲；負載高時，排程延遲會推高 TTFT。**TBT**＝同一個請求相鄰兩個輸出 token 的間隔。**Capacity**＝在滿足特定延遲目標的前提下，系統能承受的最大負載（QPS）。報表上 TTFT 取中位數；TBT 取 P99，因為**每個 decode token 都產生一個 TBT 值**（等於以 token 加權）。SLO：P99 TBT ≤ 參考 decode iteration 時間的 5 倍（strict）或 25 倍（relaxed）；參考情境是 prefill 長 4K、batch 32、沒有 prefill 干擾。絕對值（Table 3，strict／relaxed）：Mistral 0.1／0.5 s、Yi 0.2／1 s、LLaMA2-70B 1／5 s、Falcon 1／5 s。負載是否可持續：要求 **median scheduling delay ≤ 2 s**。**TTFT 不在 capacity 的 SLO 裡**，只透過排程延遲的上限間接受到約束〔判讀〕。 | 〔原文〕§2.4 p4；§5 Metrics p10；§5.1 p10；Table 3 p10 |
| 品質指標 | 無。 | 〔原文〕 |
| 主要結果 | strict SLO 下，Yi-34B（sharegpt4）的 capacity 是 Orca 的 4.0 倍、vLLM 的 3.7 倍；LLaMA2-70B 是 Orca 的 6.3 倍、vLLM 的 4.3 倍。Mistral-7B 在 100 ms 的 SLO 下是 vLLM 的 3.5 倍；Yi-34B 在 1 s 的 SLO 下是 1.65 倍。Falcon-180B 在 strict SLO 下是 vLLM TP-only 的 4.3 倍、vLLM hybrid 的 3.6 倍；relaxed 下是 1.48 倍。chunk 為 512 時 prefill 開銷最多約 25%，2048 時幾乎沒有。Table 4（Yi-34B、budget 1024、128 個請求）：兩項技術合併時，P50 TTFT／P99 TBT 為 0.76／0.14 s（sharegpt4）與 3.90／0.17 s（arxiv）。 | 〔原文〕§5.1 p11；§5.2 p12；§5.3 p12；§5.4 p13；Table 4 p13 |
| 主圖形狀 | Fig 10／11：每個模型×資料集一組 capacity（QPS）長條，分 SLO-S 與 SLO-R。Fig 12：x 軸是 P99 TBT 的 SLO，y 軸是最大 capacity。Fig 1b：x 軸是 QPS，y 軸是 P99 TBT。Fig 14：開銷長條。 | 〔原文〕p1；p11–13 |
| 消融／敏感度／開銷 | chunked-prefill 的開銷（Fig 14）；只用 hybrid batching、只用 chunked prefill、兩者合併（Table 4）；tile quantization：chunk 取 257 比取 256 的 prefill 時間多 32%；TP 與 PP 的比較（Fig 13）；直接混 prefill 與 decode 會讓 TBT 最多變成 decode-only 的 28.3 倍（Fig 9）。 | 〔原文〕p8–9；p12–13 |
| 重複與統計 | 未說明。 | 〔原文〕 |
| 程式碼／資料 | https://github.com/microsoft/sarathi-serve ；OSDI artifact 在 `osdi-sarathi-serve` branch；trace 放在 repo 的 `/data`。 | 〔原文〕p10；App A p18 |
| 設計理由（原文） | ① 每個請求只有一個 TTFT，所以看中位數；每個 token 都有一個 TBT，所以看 P99（p10）。② SLO 定為 decode iteration 時間的倍數，仿照 Patel et al.（即 Splitwise），以涵蓋各模型×硬體組合本身的效能差異（p10）。③ strict 代表聊天這類互動應用；relaxed 代表只要求整段輸出在可預期的時間內完成（p10）。④ 用 2 s 的排程延遲上限確保負載可持續（p10）。⑤ sharegpt4 代表多輪、長度變異大；arxiv 代表長 prompt，類似 M365 Copilot（p10）。⑥ token budget 依 SLO 調整：strict 用 512，relaxed 用 2048（LLaMA2-70B relaxed 用 1536）（p11）。 | 〔原文〕 |
| 設計理由〔判讀〕 | 只把 P99 TBT 當 SLO，是因為這篇要證明的是「不拖慢 decode」。如果主張的是 TTFT（例如 KV 還原），照搬這個設計會漏掉主要效果。P99 TBT 以 token 加權，會被輸出長的請求主導。 | 〔判讀〕 |
| 原文沒講清楚的地方 | capacity 怎麼搜尋（二分法？步長多少？）；每個負載點的請求數與時長；Orca 對手怎麼實作；Poisson 的 seed；過濾後剩多少請求。另外有一處數字對不上：Fig 12 caption 說 3.5 倍是 Yi-34B，正文卻說是 Mistral-7B 在 100 ms 下（p12）。〔複核修正：原卡另列「abstract 說 Falcon 最多 5.6 倍，但正文 §5.3 寫的是 4.3 倍與 3.6 倍」不算矛盾——Fig 11a（p11，openchat_sharegpt4）Falcon-180B 的 SLO-R 長條上標 "5.62x"，SLO-S 標 "4.69x"；依長條高度（Orca 約 0.13、Sarathi-Serve 約 0.73）這組倍數是對 Orca〔判讀：讀圖〕。§5.3 的 4.3×／3.6× 則是對 vLLM 的 TP-only／hybrid（Fig 13b）。兩者對手不同，abstract 沒寫對手才看起來像矛盾。同理，abstract 的 Mistral-7B 2.6× 在 Fig 10 的長條標籤（2.78x、2.15x）中找不到，對應關係未查證。〕同一份 arxiv_summarization，Sarathi 的 output 中位數是 208，Etalon（同一群作者）寫 228。 | 〔原文〕p1、p12；Etalon p3 |
| 與既有整理不一致 | 既有檔都沒有收 Sarathi-Serve（搜尋結果 0 筆）。 | 〔計算〕grep |
| 對本研究的意義〔判讀〕 | 兩件事可以直接搬：「SLO＝k 倍的無干擾基準」，以及「capacity 要附可持續條件（排程延遲上限）」。長 prefill 會造成好幾秒的 generation stall（Fig 1a）。Tiara 的還原（重算＋載入）如果與其他請求的 decode 同時跑，也會造成 stall，所以應該報 P99 TBT。PoC 要註明有沒有開 chunked prefill（現行 vLLM 的預設值本卡未查證）。 | 〔判讀〕 |

### DistServe：Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving（OSDI 2024；arXiv 2401.09670）

- **讀了什麼**：〔全文〕arXiv v3（2024-06-06，comment 寫「OSDI 2024」），https://arxiv.org/abs/2401.09670 ，查證 2026-10-06
- **一句話**：把 prefill 與 decode 放到不同 GPU，並分別搜尋兩者的平行化與配置，以最大化每張 GPU 的 goodput。
- **評測要證明的主張**：在超過 90% 的請求滿足延遲限制的前提下，比 SOTA 多服務 7.4 倍的請求，或撐住緊 12.6 倍的 SLO（abstract p1）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | OPT-13B／66B／175B，FP16。**刻意選 MHA 的 OPT**，以加重 KV 傳輸的壓力。 | 〔原文〕§6.1 p9 |
| 硬體 | 4 個節點共 32 張 GPU；每個節點 8 張 A100-SXM-80GB，以 NVLink 互連；跨節點只有 25 Gbps，所以主實驗用 low node-affinity 的配置演算法。 | 〔原文〕§6.1 p9 |
| 軟體與版本 | 配置演算法、RESTful 前端與 orchestration 層共 6.5K 行 Python；執行引擎 8.1K 行 C++／CUDA；GPU worker 以 Ray actor 實作；跨節點用 NCCL，節點內用非同步 cudaMemcpy；整合了 continuous batching、FlashAttention、PagedAttention。 | 〔原文〕§5 p8–9 |
| 資料／負載 | 三種應用（Table 1）：聊天用 ShareGPT（OPT-13B／66B／175B）；程式補全用 HumanEval（164 題，OPT-66B）；摘要用 LongBench 的摘要任務（OPT-66B）。原文寫「從資料集取樣請求」，取樣數沒寫。平均長度（Fig 7）：ShareGPT input 755.5／output 200.3；HumanEval 171.3／98.2；LongBench 1738.3／90.7。**LongBench 的 input 被截斷**，因為 OPT 的絕對位置編碼最多只支援 2048。 | 〔原文〕Table 1、Fig 7 p9；註腳 4 p9 |
| 長度 | 單位 token，上限 2048（受 OPT 限制）。 | 〔原文〕p9 |
| 到達與併發 | 資料集都沒有時間戳，所以用 Poisson，掃不同請求率。 | 〔原文〕p9 |
| 重用結構 | 無。 | 〔原文〕 |
| 掃描的自變數 | 每張 GPU 的請求率（固定 SLO）；**SLO Scale**（固定請求率，把 TTFT 與 TPOT 的門檻同時按比例縮放）；attainment 目標正文用 90%，附錄另跑 99%。 | 〔原文〕§6.2 p10–11；App C p18 |
| 對手 | vLLM：intra-op 平行度依原論文設為 1／4／8。DeepSpeed-MII：有 chunked prefill，intra-op 與 vLLM 相同；OPT-175B 跑不了，因為它的 kernel 要求 vocab_size／intra_op 是 8 的倍數，而改成 4 會 OOM。消融另有 vLLM++（列舉所有平行化取最佳）與 DistServe-High（模擬）。版本沒寫〔未查證〕。 | 〔原文〕p10；§6.4 p12 |
| 系統指標 | **TTFT**：引言稱之為 prefill 階段的持續時間。**TPOT**：每個請求平均產生一個 token 所需的時間，**不含第一個 token**；總延遲＝TTFT＋TPOT×decode 階段生成的 token 數。**SLO attainment**：滿足 SLO 的請求所佔比例。**per-GPU goodput**：在 attainment 目標（例如 90%）下，每張 GPU 能服務的最大請求率。SLO 門檻（Table 1，TTFT／TPOT）：聊天 13B 0.25 s／0.1 s、66B 2.5／0.15、175B 4.0／0.2；程式補全 0.125／0.2；摘要 15／0.15。一個請求要 TTFT 與 TPOT **兩者都達標**才算數；原文沒有給公式，但用「vLLM 大量違反 TPOT 而拉低整體 attainment」來解釋結果〔判讀〕。〔複核補充：abstract p1 把 goodput 寫成 "within both TTFT and TPOT constraints" 下的最大請求率，支持「兩項同時達標」的推定；但 §4.1 p7 在只談 prefill instance 時又把 attainment 寫成「滿足 TTFT 要求的請求比例」，所以單一 phase 的配置搜尋時只看一項〕TTFT 雖然在引言被說成 prefill 時間，但 §2 註 3 說延遲包含執行與排隊；M/D/1 式 (1) 的平均 TTFT 有排隊項；Fig 10 也拆出 prefill queuing。所以**實際使用的 TTFT 包含排隊**〔判讀〕。 | 〔原文〕p1–2、註腳 1 p2；p3 註 3；式 (1) p5；Table 1 p9；p10–11 |
| 品質指標 | 無。 | 〔原文〕 |
| 主要結果 | 聊天：請求率是 vLLM 的 2.0–4.6 倍、DeepSpeed-MII 的 1.6–7.4 倍；SLO 可緊 1.8–3.2 倍（相對 vLLM）與 1.7–1.8 倍（相對 MII）。程式補全：5.7 倍與 1.4 倍（vLLM）、1.6 倍與 1.4 倍（MII）。摘要：4.3 倍與 12.6 倍（vLLM）、1.8 倍與 2.6 倍（MII）。OPT-175B 的 KV 傳輸不到總延遲的 0.1%，超過 95% 的請求傳輸時間少於 30 ms。模擬器與實機的 SLO attainment 誤差小於 2%。99% 目標下：對 vLLM 請求率 3–8 倍、SLO 1.24–6.67 倍。動機實驗（Fig 1；13B、input 512／output 64、單張 A100）：混合部署的 goodput 約 1.6 rps；只跑 prefill 5.6 rps，只跑 decode 10 rps；兩張 prefill 加一張 decode 可達每張 GPU 3.3 rps，即 2.1 倍。 | 〔原文〕§6.2 p10–11；§6.3 p11；Table 2 p12；App C p18；p1–2 |
| 主圖形狀 | Fig 8／9 上排：x 軸是每張 GPU 的請求率（req/s），y 軸是 SLO attainment（%），在 90% 處畫垂直線讀出 goodput。下排：x 軸是 SLO Scale（由大到小），y 軸是 SLO attainment。Fig 10：延遲拆解，以及 KV 傳輸時間的 CDF。Fig 1：P90 TTFT 與 P90 TPOT 對請求率。 | 〔原文〕p1、p10–11 |
| 消融／敏感度／開銷 | vLLM++ 與 vLLM 一樣好（預設的 intra-op=4 已是最佳）；DistServe-Low 對 DistServe-High（模擬）；演算法執行時間最長 1.3 分鐘；模擬器準確度（Table 2）。 | 〔原文〕§6.4–6.5 p12；p7 |
| 重複與統計 | 未說明。 | 〔原文〕 |
| 程式碼／資料 | https://github.com/LLMServe/DistServe | 〔原文〕註腳 2 p2 |
| 設計理由（原文） | ① 仿照 vLLM 選 OPT，而且 MHA 讓 KV 傳輸的壓力夠大（p9）。② 據作者所知沒有現成的 SLO 設定，所以依各應用的服務目標憑經驗設定（p9）。③ 聊天的 TPOT 0.1 s 比人類閱讀速度快；程式補全兩項都要緊；摘要的 TTFT 可以鬆、TPOT 要緊（p9）。④ 沒有時間戳，所以用 Poisson（p9）。⑤ 在實機上量 SLO attainment 太耗時，所以用模擬器搜尋配置（p7）。⑥ 消融用模擬，因為 vLLM 不支援 inter-op 平行，測試床的跨節點頻寬也不夠（p12）。⑦ TPOT 只要比閱讀速度（每分鐘 250 字）快就夠了（p2）。 | 〔原文〕 |
| 設計理由〔判讀〕 | goodput 把整個延遲分布壓成一個請求率。要能比較，必須同時寫出 attainment 目標、SLO 門檻，以及是否以每張 GPU 正規化。SLO Scale 曲線是檢查結論是否只是門檻選得剛好的好工具。 | 〔判讀〕 |
| 原文沒講清楚的地方 | 每個點跑多少請求；LongBench 怎麼截斷；TTFT 的量測起點（沒有明文定義）；attainment 是否要求兩項同時達標（只能推定）；vLLM 與 MII 的版本。 | 〔原文〕 |
| 與既有整理不一致 | workloads_eval §6.3 B.2 寫「明訂 SLO（TTFT 與 TPOT 門檻）下的 goodput」，沒有寫 attainment 目標（DistServe 用 90% 與 99%），也沒寫是否以每張 GPU 正規化，屬於簡化。B.1 把「TPOT／ITL」並列成同一項：DistServe 的 TPOT 是每請求平均、不含第一個 token；ITL 在七篇論文中都沒有出現，是工具的用語（見 E02）。〔複核：B.2 ✅ 指控成立（L517 只寫「明訂 SLO（TTFT 與 TPOT 門檻）下的 goodput」，確實沒寫 attainment 目標與是否以每張 GPU 正規化；屬簡化，不是事實錯誤）。B.1 ✅ 指控成立（L516 寫作「TPOT／ITL」，把兩個母體不同的指標寫成一項；屬含糊，建議拆開）。〕 | 〔原文〕DistServe p1–2；workloads_eval L516–517 |
| 對本研究的意義〔判讀〕 | goodput、SLO attainment、SLO Scale 這三件可以直接用在多請求實驗。但在單卡、單請求的 PoC 中，per-GPU goodput 沒有意義；在 16K–512K，SLO 門檻必須隨長度改變（見 Etalon 卡）。§7 說長 context 下「KV 傳輸 ÷ prefill」的比例會下降（p13），這可以當作 κ 隨情境變動的旁證，但原文只是論述，沒有量測。 | 〔判讀〕；〔原文〕§7 p13 |

### Splitwise：Efficient Generative LLM Inference Using Phase Splitting（ISCA 2024；arXiv 2311.18677）

- **讀了什麼**：〔全文〕arXiv v2（2024-05-20），https://arxiv.org/abs/2311.18677 ，查證 2026-10-06。IEEE 雙欄，逐欄抽字閱讀；圖中數值只採正文文字。
- **一句話**：把 prompt 階段與 token 階段放到不同（可以是異質的）機器上，並用模擬器設計成本、功耗或吞吐最佳的叢集。
- **評測要證明的主張**：吞吐高 1.4 倍、成本低 20%；或在相同功耗與成本下吞吐高 2.35 倍（abstract p1）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | Llama2-70B（Table III：80 層、hidden 8192、#Heads 32）與 BLOOM-176B（70 層、hidden 14336、112 heads）。HF config 顯示 Llama-2-70B 是 64 個 query head／8 個 KV head（GQA），BLOOM 是 112 heads、70 層（MHA）。**Table III 寫 Llama2-70B 有 32 個 head，與 config 不符。** dtype 沒有寫。 | 〔原文〕Table III p3；〔文件〕HF config |
| 硬體 | Azure 上的 DGX-A100 與 DGX-H100 VM 各兩台；以 InfiniBand 互連，H100 為 400 Gbps，A100 為 200 Gbps。Table I：A100／H100 的規格與每台成本（$17.6/hr 與 $38/hr）。8 張 GPU 做 TP。叢集層級（數十台）的結果全部來自模擬。 | 〔原文〕§V-A p8；§VI-A p9；Table I p1；§II-E p3 |
| 軟體與版本 | 在 vLLM 上實作 KV 傳輸（vllm-project/vllm PR #2809），用 MSCCL++ 的 zero-copy 單向 put。因為原生 vLLM 只有會搶佔 token 階段的 continuous batching，作者另外實作了 mixed continuous batching。模擬器為 SplitwiseSim（https://github.com/Mutinifni/splitwise-sim）。 | 〔原文〕§V-A p8；§V-B p8；參考文獻 [1]、[20] p13 |
| 資料／負載 | Azure 兩個 LLM 推論服務（coding 與 conversation）的 production trace，日期 2023-11-11。特性化分析用的 trace 長 20 分鐘，含到達時間、input 長度、output 長度。**只公開了一部分**（AzurePublicDataset）。因為隱私看不到內容，作者送出長度符合的 prompt，並**強制生成指定數量的 token**。長度：coding 的 prompt 中位數 1500、output 中位數 13；conversation 為 1020 與 129，output 近似雙峰。 | 〔原文〕§III p3；§III-A p3 |
| 長度 | 單位 token。見上一格。Fig 5a 的 TTFT 掃到 prompt 長約 8K（依圖軸）。 | 〔原文〕p3–4 |
| 到達與併發 | 特性化：把 trace 縮成每秒 2 個請求，在單機上跑；怎麼縮放沒有說明。叢集評估：用 production 的長度分布，**調整 Poisson 的到達率**來改變負載。 | 〔原文〕§III-B p4；§V-B p9 |
| 重用結構 | **刻意不在請求之間重用 KV**，以模擬有安全保證的雲端服務。〔複核修正：原文句首是 "For this characterization"（§III p3），範圍限於特性化；§V／§VI 的叢集評估沒有重述，但也沒有描述任何 KV 重用〕 | 〔原文〕p3 |
| 掃描的自變數 | 請求率（RPS）；叢集設計（Splitwise-AA／HH／HA／HHcap、Baseline-A100／H100）；prompt 與 token 機器的台數（Fig 12 的搜尋空間）；最佳化目標（iso-power、iso-cost、iso-throughput）；互換負載與模型（§VI-D）；prompt 長度（量 KV 傳輸）。 | 〔原文〕§IV-D p7；§VI p9–11 |
| 對手 | Baseline-A100 與 Baseline-H100，同樣用 mixed continuous batching。 | 〔原文〕§V-B p9 |
| 系統指標 | Table II：E2E＝使用者看到的整個查詢時間；TTFT＝使用者多快看到第一個回應；TBT＝**平均**的 token 串流延遲；Throughput＝每秒請求數。SLO 是 TTFT、TBT、E2E 各取 P50／P90／P99，共 **9 個**，都以「相對 DGX-A100 無競爭時的 slowdown」表示（Table VI：TTFT 2×／3×／6×；TBT 1.25×／1.5×／5×；E2E 1.25×／1.5×／5×）。**9 個都要滿足**。prompt 階段的吞吐＝每秒處理的 prompt token 數。另外報成本、provisioned 功耗、機器數。 | 〔原文〕Table II p2；Table VI 與 SLOs 段 p9；§III-D p4；p8 |
| 品質指標 | 不影響準確度，因為 KV 傳輸是無損的。 | 〔原文〕§IV-E p8 |
| 主要結果 | KV 傳輸開銷不到 prompt 計算時間的 7%；逐層傳輸後，沒被重疊掉的時間在 A100 約 8 ms、H100 約 5 ms。coding trace 端到端：E2E 只多 0.8%（序列傳輸為 3%）；第二個 token 多 16.5%（序列傳輸為 64%）。iso-power（conversation）：Splitwise-AA 的吞吐是 Baseline-A100 的 2.15 倍；Splitwise-HA 為 1.18 倍，成本低 10%。iso-cost：Splitwise-AA 比 Baseline-H100 多 1.4 倍吞吐，但多 25% 功耗、2 倍空間。iso-throughput 的功耗最佳化：HHcap 少 25% 功耗。批次工作：每美元 0.89 RPS（AA）與 0.75 RPS（HH）。〔複核補充：原文說 Baseline-A100 與 Splitwise-AA 並列 0.89 RPS/$、Splitwise-HH 與 Baseline-H100 並列 0.75 RPS/$；高負載下 Splitwise 退化成同台數的 Baseline，**這一項沒有優勢**〕 | 〔原文〕§VI-A p9；§VI-B p10；§VI-C p10–11；§VI-E p11 |
| 主圖形狀 | Fig 16／20：x 軸是請求率（req/s），y 軸是各 percentile 的 TTFT、TBT、E2E（Fig 20 為正規化的 p90），紅色虛線是 SLO。Fig 18／19：各設計的吞吐、成本、功耗、機器數，以正規化長條呈現。Fig 12：prompt 機器數×token 機器數的二維搜尋空間。 | 〔原文〕p7；p10–11 |
| 消融／敏感度／開銷 | KV 傳輸延遲對 prompt 長度（Fig 14–15）；負載與模型錯配（§VI-D）；batch token 數的分布（Fig 17，70 與 130 RPS）。 | 〔原文〕p9–11 |
| 重複與統計 | 未說明。模擬器的驗證：分段線性效能模型的 MAPE 小於 3%（80:20 切分訓練與測試）；另用超過 50K 個 iteration 的 production 負載做端到端驗證。 | 〔原文〕§V-B p9 |
| 程式碼／資料 | vLLM PR #2809；SplitwiseSim；AzurePublicDataset 上的部分 trace。 | 〔原文〕p3；p8；p13 |
| 設計理由（原文） | ① 因隱私看不到內容，所以只用長度並強制輸出；prompt 的文字不影響所量的效能指標，它們只取決於 input 與 output 的長度（p3）。② 不重用 KV，以模擬有安全保證的雲端服務（p3）。③ SLO 以 A100 無競爭為參考的 slowdown 表示；TTFT 的 SLO 較鬆，因為它對 E2E 的影響較小（p9）。④ 用模擬器在大規模下探索叢集設計（p8）。 | 〔原文〕 |
| 設計理由〔判讀〕 | 「SLO＝相對無競爭的 slowdown」把硬體差異正規化掉，正是跨平台比較所需要的（A100 對 H100；或 Tiara 的 3090 對 MI300X）。但要求 9 個 SLO 全部滿足非常嚴格，capacity 由最緊的那一個決定，所以結論對門檻很敏感。 | 〔判讀〕 |
| 原文沒講清楚的地方 | TBT 的 percentile 是對每個 token 算，還是對每個請求的平均值算（Table II 寫的是「平均」）；slowdown 的分母是每個請求自己的無競爭延遲，還是整體統計；怎麼把 trace 縮成每秒 2 個請求；abstract 的 2.35 倍與正文的 2.15 倍對應哪個對照組（可能基準不同）；Llama2-70B 的 head 數。 | 〔原文〕 |
| 與既有整理不一致 | ~~workloads_eval L240 與 L244 說「Azure 2023 那份就是 Splitwise 的資料」。但原文說只公開了一部分，而且論文特性化用的 trace 長 20 分鐘、日期 2023-11-11（p3）；workloads_eval 自己算出的檔內時間是 2023-11-16 18:15–19:14（約 1 小時）。兩者日期與長度都對不上，所以公開檔很可能不是論文用的那段 trace。引用時應寫成「Splitwise 作者公開的 Azure 2023 trace（不是論文用的原 trace）」〔判讀〕。~~ 〔複核修正：❌ **指控不成立，workloads_eval 的寫法不需要改**。① 原文 p3 說 "We have released a subset of our traces"，並以參考文獻 [4] 指向的正是 `AzureLLMInferenceDataset2023.md`（p13）。② 該資料集 README（raw.githubusercontent.com，查證 2026-10-07，repo HEAD 215becd）明寫這份資料就是 Splitwise 論文描述與分析的資料、是論文用的 sample，並附 notebook 用公開檔重現論文的特性化圖。③ 原卡指出的不一致是真的：README 與論文寫 November 11th、20 分鐘，而兩個 CSV 的時間戳實際是 2023-11-16 18:15:46–19:14:19（code 18:17:03 起），約 1 小時（V01 下載重算）。但這只能說明「日期與時長的描述和檔案對不上，原因未說明」，不能推出「不是論文用的 trace」。建議寫法：「Splitwise 作者公開的 sample（README 稱即論文所用）；檔內時間戳為 2023-11-16、約 1 小時，與論文所寫 11-11、20 分鐘不一致」。〕 | 〔原文〕p3、參考文獻 [4] p13；〔文件〕AzureLLMInferenceDataset2023.md；〔計算〕CSV 時間戳；workloads_eval L240、L244 |
| 對本研究的意義〔判讀〕 | 這是七篇中唯一有 production trace 長度與時間的，但它沒有內容（因隱私看不到）、特性化時刻意不重用，對 KV 分層沒有直接用處。它驗證模擬器的方式（MAPE＋端到端 iteration）可以當作 Tiara M4 模擬器的驗證範本。slowdown 式的 SLO 可以用在跨平台的比較。 | 〔判讀〕 |

### Etalon（原名 Metron）：Holistic Performance Evaluation Framework for LLM Inference Systems（arXiv 2407.07000）

- **讀了什麼**：〔全文〕arXiv v2（2024-08-30），https://arxiv.org/abs/2407.07000 ，查證 2026-10-06。v1 標題為 "Metron: …"（https://arxiv.org/abs/2407.07000v1）。原文沒標發表場所〔未查證〕。
- **一句話**：指出 TTFT、TBT、TPOT、normalized latency 各自的盲點，提出以 token 期限為基礎的 fluidity-index。
- **評測要證明的主張**：現行指標不足，甚至會誤導；fluidity-index 與 fluid token generation rate 更貼近使用者的實際體驗（abstract p1）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | 公開 API：Llama3-70B、Mixtral-8x7B。〔複核修正：原寫「Fig 5 另有 Llama3-8B」。渲染後的 Fig 5（p8）每個子圖只有 Mixtral-8x7B 與 Llama3-70B 兩格，caption 也只列這兩個；Llama3-8B 的字樣只存在 PDF 文字層（被裁掉的面板），不能算原文內容〕開源系統：Llama3-8B，單張 H100，用 rope-scaling 支援超過 8192 的 prompt。動機圖：Yi-34B，兩張 H100。注意力類型依 HF config：Llama3-8B 為 32／8，Llama3-70B 為 64／8，Mixtral 為 32／8，皆為 GQA。 | 〔原文〕§5.1 p7；§5.2 p8；Fig 1–2 p4；〔文件〕HF config |
| 硬體 | 開源評估用 H100；公開 API 的硬體不知道。 | 〔原文〕p8 |
| 軟體與版本 | fork 自 LLMPerf，支援 OpenAI 相容 API 與 vLLM。Sarathi-Serve 以「vLLM 打開 chunked-prefill」來代表。 | 〔原文〕§4.2 p7；§5.2 p8 |
| 資料／負載 | 公開 API：自訂負載，prefill 長 256–8K，最多生成 256 token。開源：從 Arxiv-Summarization 隨機取樣；Fig 6 另有 ShareGPT。D_p 的擬合：用 vLLM 單獨跑 10 個請求（數量可調），量 prefill 時間對長度的關係並擬合曲線。原文引用 LMSys-Chat-1M 的 prefill 長度中位數 417、P90 1418。 | 〔原文〕§5.1 p7；§5.2 p8；Fig 6 p9；§4.1 p6–7 |
| 長度 | 公開 API 的 prefill 為 256–8K；Fig 1 的 prompt 到 32K（Yi-34B）；Fig 5c 的軸到 32768。 | 〔原文〕p4；p7–8 |
| 到達與併發 | 公開 API：每小時跑一次，連續 24 小時；到達過程沒有說明。開源：capacity 搜尋（掃 QPS），分布沒有說明。 | 〔原文〕p7–8 |
| 重用結構 | 無。 | 〔原文〕 |
| 掃描的自變數 | 目標 TBT（D_d）；QPS（找 capacity）；prompt 長度。 | 〔原文〕§5 p7–9 |
| 對手 | 公開 API：Anyscale、Groq、Fireworks。開源：vLLM 與 Sarathi-Serve。 | 〔原文〕§5 p7–8 |
| 系統指標 | 現行指標的定義（§2.2）：**TTFT**＝從請求到達到第一個輸出 token，包含排程延遲（到達到開始處理 prompt）與 prompt 處理時間。**TBT**＝decode 階段每個後續 token 的延遲。**TPOT**＝一個請求的總 decode 時間 ÷ decode token 數。**Normalized latency**＝請求的總執行時間（含排程、prompt、decode）÷ decode token 數。**Capacity**＝滿足 SLO 時的最大 QPS。新指標（§4.1）：第 i 個 token 的期限為 D_i＝D_p＋i·D_d；token 提早到時把剩下的時間累積成 slack；遲到則記為 miss，一次 stall 依 D_d 可記成多次 miss，並把之後的期限從實際生成時間重新起算（Algorithm 1）。**fluidity-index**＝達成的期限數 ÷ 總期限數，每個請求一個值。D_p＝依 prompt 長度預測的 prefill 時間＋scheduling slack；D_d 分三檔：25、50、100 ms。**fluid token generation rate**＝能讓「99% 的請求 fluidity-index ≥ 0.9」成立的最小 D_d 的倒數。 | 〔原文〕§2.2 p2–3；§4.1 p5–7；Algorithm 1 p6；§5.1 p7 |
| 品質指標 | 無。 | 〔原文〕 |
| 主要結果 | Fig 2（arxiv_summarization、1.5 QPS、Yi-34B 兩張 H100）：vLLM 約 60% 的請求排程延遲超過 25 s，Sarathi-Serve 最多 15 s，但兩者的 normalized latency 只差幾百 ms。Groq 依 TPOT 估算為 600 tok/s，依 tail TBT 估算則低 4 倍。Llama3-8B（H100）：SLO 設為「99% 的請求 deadline miss 少於 10%、D_d＝25 ms」時，兩個系統的 fluidity capacity 都是 0.6 QPS；若用 tail TBT 看，Sarathi 的 token 吞吐比 vLLM 低 2 倍。 | 〔原文〕§3.1 p3–4；§5.1 p8；§5.2 p8–9 |
| 主圖形狀 | Fig 3b 與 Fig 5a：用三種指標推算出的吞吐長條。Fig 5d：x 軸是目標 TBT，y 軸是 fluidity-index（P99）。Fig 6a：三種 SLO 定義下的 capacity 長條。Fig 6b：deadline miss rate 的 CDF。 | 〔原文〕p4；p8–9 |
| 消融／敏感度／開銷 | 沒有正式的消融；這是一篇方法學論文。 | 〔原文〕 |
| 重複與統計 | 公開 API 跑 24 次（每小時一次）；其餘未說明。 | 〔原文〕p7 |
| 程式碼／資料 | github.com/project-etalon/etalon | 〔原文〕abstract p1 |
| 設計理由（原文） | ① 仿照即時系統中週期性任務的期限評估，以及影音串流的緩衝（p2、p5）。② D_p 隨 prompt 長度改變，因為 prefill 時間隨長度（二次方）增長，固定的 TTFT SLO 不實際（p3、p6）。③ 現行指標各有問題：normalized latency 會把排程延遲攤平；TPOT 會把 stall 攤平；tail TBT 過度懲罰；TBT 處理不了 speculative decoding 一次產生多個 token 的情況（p3–4）。④ 公開 API 每小時測一次，以涵蓋一天內的流量變化（p7）。 | 〔原文〕 |
| 設計理由〔判讀〕 | 對長 context 最有價值的是「D_p 是 prompt 長度的函數」：在 16K–512K，固定的 TTFT 門檻沒有意義。slack 的設計表示「TTFT 很短、但 decode 偶爾卡一下」可以被吸收。缺點是 D_p 以某個系統（vLLM）的 prefill 曲線當目標；對靠快取避免 prefill 的系統（如 Tiara），等於以全部重算為基準，只要命中就一定達標，所以需要另訂基準。〔複核補充：原文 §4.1 p7 只把 vLLM 當「建議」的基準，並明說這條 prefill 曲線應在**被評估的系統上**重新量；§6 p9 也承認黑箱系統的 D_p 難定。上面的缺點在照原文建議做時仍成立（在快取系統上「量 prefill 曲線」本身就要先決定是否命中），但不是原文忽略了這件事〕 | 〔判讀〕 |
| 原文沒講清楚的地方 | scheduling slack 的數值（只說憑經驗決定，p6、p9）；擬合函數的形式；capacity 怎麼搜尋；每個點跑多少請求；Algorithm 1 以相鄰 token 間隔為輸入，與 §4.1 文字所寫的絕對期限之間如何對應；「1000 個請求」只出現在例子裡（p4）。同一份 arxiv_summarization，output 中位數這裡寫 228，Sarathi-Serve Table 2 寫 208。 | 〔原文〕 |
| 與既有整理不一致 | 既有檔都沒有收 Etalon（搜尋結果 0 筆）。另外，Etalon 自己說 Orca 與 vLLM 都用 median normalized latency（§2.2 p3），但 vLLM 論文明寫取的是平均（vLLM p10），所以 **Etalon 對 vLLM 的轉述不精確**。〔複核：✅ Etalon p3 "Median Normalised Latency has been used in [30, 25]"，[30]＝Orca、[25]＝vLLM（參考文獻 p11–12）；vLLM p10 寫 "the mean of every request's end-to-end latency divided by its output length"〕 | 〔原文〕Etalon p3 對照 vLLM p10 |
| 對本研究的意義〔判讀〕 | 這篇是指標字典的仲裁者。PoC 應該把 TTFT 拆成「排隊＋還原（重算或載入）」分別報，並用隨長度變化的 TTFT 目標來評估。 | 〔判讀〕 |

---

## 指標字典

本節只寫**論文**裡的定義；benchmark 工具（vLLM bench、SGLang bench_serving、GenAI-Perf…）的定義由 E02 讀程式碼負責。「何時用」與「常見誤用」兩欄是〔判讀〕，其中以原文批評為依據的部分另外標出處。

| 指標 | 各論文的原文定義（出處） | 彼此差異 | 何時該用〔判讀〕 | 常見誤用〔判讀〕 | 工具定義 |
|:--|:--|:--|:--|:--|:--|
| **TTFT** | Sarathi-Serve：從請求到達系統到產生第一個輸出 token；負載高時排程延遲會推高它（§2.4 p4）〔原文〕。Etalon：從請求到達到第一個 token，明確包含排程延遲與 prompt 處理時間（§2.2 p2）〔原文〕。DistServe：引言稱為「prefill 階段的持續時間」（p1），但 M/D/1 式 (1) 與 Fig 10 都含排隊（p5、p11）〔原文〕。Splitwise：使用者多快看到第一個回應，沒有給起點（Table II p2）〔原文〕。SGLang：用 "first token latency"，沒有定義（p8–9）〔原文〕。Orca、vLLM：不用這個指標〔原文〕。 | **是否包含排隊**：Sarathi-Serve 與 Etalon 明說包含；DistServe 的文字與用法不一致；Splitwise 沒說。Sarathi-Serve 報**中位數**；DistServe 在動機圖用 P90，SLO 用 attainment；Splitwise 用 P50／P90／P99。 | 主張與還原或 prefill 有關時（例如 KV 分層）一定要報，並**拆成排隊與處理兩段**。長度跨度大時，要配上「隨長度變化的目標」（Etalon D_p）。 | ① 不同長度的請求共用一個固定的 TTFT SLO（Etalon p3 批評）。② 把 TTFT 除以 prompt 長度，等於連排程延遲一起正規化，會不公平地懲罰短 prompt（Etalon p3）。③ 比較時沒說是否含排隊。 | 見 E02 |
| **TPOT** | DistServe：每個請求平均產生一個 token 的時間，**不含第一個 token**；總延遲＝TTFT＋TPOT×decode 生成的 token 數（p1–2，註 1）〔原文〕。Etalon：一個請求的總 decode 時間 ÷ decode token 數（§2.2 p2）〔原文〕。Sarathi-Serve、Splitwise、Orca、vLLM、SGLang 都不用（全文搜尋 TPOT 為 0 筆）〔計算〕。 | 一個請求一個值，是**請求內的平均**，會把 stall 攤平。DistServe 與 Etalon 的定義本質相同。 | 想表達「這個請求整體的生成速度」，或用 per-request attainment（DistServe）時使用。 | ① 拿 TPOT 當成每個 token 的延遲：一次 10 s 的 stall 在長輸出中只會讓 TPOT 微幅上升（Etalon §3.1 p3–4，Fig 3a）。② 用 1／mean TPOT 宣稱吞吐：Groq 依 TPOT 算是 600 tok/s，依 tail TBT 算低 4 倍（Etalon p8）。③ 把 TPOT 與 ITL 寫成同一項（workloads_eval L516 寫作「TPOT／ITL」）〔複核修正：原寫「TPOT 與 ITL／TBT」，但 L516 沒有提到 TBT〕。 | 見 E02 |
| **TBT** | Sarathi-Serve：同一個請求相鄰兩個輸出 token 的間隔；**每個 decode token 都產生一個值**，報 P99（§2.4 p4；§5 p10）〔原文〕。Etalon：decode 階段每個後續 token 的延遲（§2.2 p2）〔原文〕。Splitwise：**平均**的 token 串流延遲（Table II p2），SLO 取 P50／P90／P99（Table VI p9）〔原文〕。 | Sarathi-Serve 與 Etalon 是**每個 token 一個值**，所以分布以 token 加權；Splitwise 寫的是「平均」，它的 percentile 是對 token 還是對請求取，原文沒寫。 | 有 prefill 或還原與 decode 搶同一張 GPU 時（chunked prefill、背景載入 KV）用來看 stall；報 P99 或整個 CDF。 | ① 只看 P99 TBT，會過度懲罰偶爾的 stall，也看不出 stall 發生在請求的哪個位置（Etalon p4–5，Fig 3c）。② speculative decoding 一次產生多個 token 時，TBT 會出現 0 與 3T 這種失真值（Etalon p4）。③ token 加權的 P99 會被輸出長的請求主導〔判讀〕。 | 見 E02 |
| **ITL** | **七篇論文都沒有使用 ITL 這個名詞**（全文搜尋為 0 筆）〔計算〕。Etalon 只在描述現象時用到 inter-token jitter（p1），演算法輸入叫 inter_token_times（Algorithm 1 p6）〔原文〕。 | ITL 屬於工具的用語，與 TBT 的關係（是否逐 chunk、是否含第一個 token）見 E02。 | — | 在論文中寫 ITL 時要附上工具的定義與版本，否則讀者會以為是 TBT 或 TPOT〔判讀〕。 | **見 E02** |
| **E2E latency** | Splitwise：使用者看到的整個查詢時間（Table II p2）〔原文〕。DistServe：TTFT＋TPOT×生成 token 數（註 1 p2）〔原文〕。Orca、vLLM：每個請求的端到端延遲，作為 normalized latency 的分子（Orca p12；vLLM p10）〔原文〕。Etalon：總執行時間＝排程＋prompt＋decode（p3）〔原文〕。SGLang：程式的總延遲（p7、Fig 8）〔原文〕。 | 本質相同，但起點（到達或開始執行）不一定寫明。SGLang 是「整個程式」，不是單一請求。 | 批次型或 agent 型的任務，以完成時間為準時使用。 | 輸出長度不固定時直接比較 E2E：差異其實來自生成長度，不是系統（Orca p12 正因如此才正規化）。 | 見 E02 |
| **Normalized latency** | Orca：每個請求的 E2E ÷ 生成的 token 數，取**中位數**，單位 ms/token（Fig 10 p12）〔原文〕。vLLM：每個請求的 E2E ÷ 輸出長度，取**平均（mean）**，並說沿用 Orca（p10）〔原文〕。Etalon：總執行時間 ÷ decode token 數；並說 Orca 與 vLLM 都用中位數（p3）——**對 vLLM 的部分不正確**〔原文對照〕。SGLang：Fig 5–7 的 "Normalized latency／throughput" 是**相對某參考系統正規化的長條**，與上面無關（p7–8）〔原文〕。 | mean 與 median 不能混著比；分母是「生成 token 數」或「decode token 數」，差在第一個 token 算不算；SGLang 的同名指標意思完全不同。 | 只在 input 短、輸出長度變異大、主張是吞吐時，作為「整體吞吐」的代理。**長 context 不應使用。** | ① 把排程延遲攤平：vLLM 與 Sarathi 的排程延遲差了 10 s 以上，normalized latency 只差幾百 ms（Etalon Fig 2，p3–4）。② 把 stall 攤平（Etalon p3）。③ 長 prompt、短輸出時，數值由 prefill 主導，讀起來像 decode 很慢〔判讀〕。 | 見 E02 |
| **Throughput：req/s** | Orca：Fig 10 的 x 軸，軸名 "Throughput (req/s)"（p12）〔原文〕；是否為「達成的」吞吐原文沒寫〔判讀；複核修正〕。Splitwise：每秒請求數（Table II p2），叢集最大吞吐（p9）〔原文〕。DistServe：每張 GPU 的請求率（Fig 8 p10）〔原文〕。Sarathi-Serve 與 Etalon：capacity 的單位是 QPS（p4；p3）〔原文〕。vLLM：x 軸是**送入的請求率**（Fig 12 p10）〔原文〕。SGLang：programs per second（p7）〔原文〕。 | 「達成的」與「送入的」不同：系統未飽和時兩者相等，飽和後送入的請求率只會讓佇列變長〔判讀〕。有些論文另外以每張 GPU 正規化（DistServe）。 | 多請求的服務實驗。 | ① 拿請求長度分布不同的負載比 req/s。② 報 req/s 卻不附延遲條件（請求率可以無限送）。 | 見 E02 |
| **Throughput：output tok/s** | vLLM：Fig 1 的「Throughput (token/s)」，只是示意圖，沒有定義（p1）〔原文〕。DistServe：描述既有系統最大化的是「所有使用者與請求每秒生成的 token 數」（p2）〔原文〕。SGLang：Fig 8 的 tokens/s（p9）〔原文〕。Splitwise：token 階段的吞吐，以 tokens/s 表示（§III-D p4）〔原文〕。Etalon：用 1／mean TPOT、1／P99 TBT、fluid rate 三種方式推算 tok/s（p7）〔原文〕。 | 從哪個延遲指標推算，結果可差到 4 倍（Etalon p8）。 | 純 decode 能力，或離線批次。 | 用 1／TPOT 宣稱「使用者可得的速度」（Etalon p8）。 | 見 E02 |
| **Throughput：total tok/s** | **七篇論文都沒有定義「input＋output 的總 tok/s」**。最接近的是 Splitwise 的 prompt 階段吞吐，即每秒處理的 prompt token 數（§III-D p4）〔原文〕。 | — | 不建議作為主指標；prefill 與 decode 的 token 成本差了好幾個數量級〔判讀〕。 | prompt 很長時，total tok/s 由 prefill 主導，會誤以為系統很快〔判讀〕。 | **見 E02** |
| **Goodput** | DistServe：per-GPU goodput＝在 SLO attainment 目標（例如 90%）下，每張 GPU 能服務的最大請求率（p2；結論 p13）〔原文〕。其他六篇沒有定義；Sarathi-Serve 與 Etalon 只在參考文獻的標題中出現（搜尋確認）〔計算〕。 | 一定要附三個參數：SLO 門檻、attainment 目標、是否以每張 GPU 正規化。工具的「goodput」可能是另一種定義（見 E02）。 | 多請求、有明確 SLO 時，作為**單一總結數字**。 | ① 只寫 goodput，沒寫 attainment 目標與門檻（workloads_eval §6.3 B.2 的寫法）。② 把 goodput 當成「符合 SLO 的 token 吞吐」〔判讀：DistServe 的單位是 req/s/GPU〕。 | 見 E02 |
| **SLO attainment** | DistServe：滿足 SLO 的請求所佔比例（p1）；用來決定 goodput（p2）；主圖為 attainment 對請求率與 SLO Scale（p10–11）〔原文〕。一個請求要同時滿足 TTFT 與 TPOT 才算〔判讀，見 DistServe 卡〕。Etalon 的「99% 的請求 fluidity-index ≥ 0.9」是同一種形式（p2、p7）〔原文〕。 | attainment 是**每個請求聯合判定**再算比例；Splitwise 則是對**每個指標各自取 percentile** 再全部要求達標（p9）。兩者不同：前者要求同一個請求兩項都好，後者各項的尾端可以落在不同請求上〔判讀〕。 | 多個 SLO 要一起滿足時。 | 把「P90 TTFT ≤ X 且 P90 TPOT ≤ Y」當成「90% attainment」〔判讀〕。 | 見 E02 |
| **Capacity（最大可撐請求率）** | Sarathi-Serve：在滿足特定延遲目標下，系統能承受的最大負載（QPS）（§2.4 p4）；實際條件是 P99 TBT 不超過 SLO，加上 median scheduling delay ≤ 2 s（p10）〔原文〕。Etalon：同樣的定義（引用 Vidur 與 Sarathi-Serve），條件改為 fluidity SLO（p3、p7）〔原文〕。Splitwise：叢集在 9 個 SLO 都滿足時的最大吞吐（p9）〔原文〕。vLLM：非正式的說法，「維持相近延遲時可撐的請求率」（p11）〔原文〕。 | 條件不同：Sarathi-Serve 只看 TBT，加上排程延遲上限；Splitwise 9 個 SLO 全部要滿足；Etalon 看 fluidity；vLLM 沒有門檻。搜尋方法都沒有寫〔原文〕。 | 主張「同樣品質下能服務更多」時使用；一定要附**可持續條件**。 | ① 用有限長度的 trace 量超載點：延遲數字取決於 trace 長度，而不是系統〔判讀〕。② 沒寫 SLO 就比 capacity。 | 見 E02 |
| **P50／P90／P99** | Orca：中位數（p12）；vLLM：平均（p10）；Sarathi-Serve：TTFT 取 P50、TBT 取 P99（p10）；DistServe：動機圖用 P90，SLO 用 90% 與 99% 的 attainment（p1、p10、p18）；Splitwise：TTFT、TBT、E2E 都取 P50／P90／P99（p9）；Etalon：P99 TBT，以及 99% 的請求（p7）；SGLang：平均（p7）〔原文〕。 | **母體不同**：對請求取（TTFT、TPOT、E2E、fluidity）或對 token 取（Sarathi-Serve 的 TBT）。Sarathi-Serve 的理由是：TTFT 每個請求一個值，TBT 每個 token 一個值（p10）〔原文〕。 | 尾端要與使用者體驗掛鉤時用 P99；母體要寫清楚。 | ① 不寫是對請求還是對 token 取 percentile。② 請求數少時報 P99（例如 128 個請求的 P99 只由一兩個請求決定）〔判讀〕。 | 見 E02 |
| **fluidity-index**（與 fluid token generation rate） | Etalon：每個 token 有期限 D_i＝D_p＋i·D_d，提早到的時間累積成 slack，遲到時重設之後的期限；fluidity-index＝達成的期限 ÷ 總期限，每個請求一個值（§4.1 p5–6，Algorithm 1）。fluid token generation rate＝讓「99% 的請求 fluidity ≥ 0.9」成立的最小 D_d 的倒數（p7）〔原文〕。 | 唯一同時考慮 stall 的大小、頻率與發生位置的指標；但依賴 D_p 曲線與 slack 這兩個由作者決定的參數（p6、p9）。 | 串流互動、而且 decode 期間會被插入 prefill 或還原工作時。 | ① D_p 取自另一個系統的 prefill 曲線，卻用在有快取的系統上〔判讀，見 Etalon 卡〕。② 不報 slack 值〔原文 p9 自承〕。 | 見 E02 |
| （補充）**Scheduling delay** | Sarathi-Serve：capacity 的可持續條件是中位數 ≤ 2 s（p10）〔原文〕。Etalon：從到達到開始處理 prompt 的時間，是 TTFT 的一部分（p2）；Fig 2b 畫出它的 CDF（p4）〔原文〕。DistServe：拆成 prefill queuing 與 decoding queuing（Fig 10 p11）〔原文〕。 | — | KV 分層會把「等 KV 載入」也算進去，應該分開報〔判讀〕。 | 把它埋在 TTFT 或 normalized latency 裡（Etalon p3）。 | 見 E02 |
| （補充）**Cache hit rate** | SGLang：已快取的 prompt token 數 ÷ prompt token 數（§3 p5）〔原文〕；以 token 加權，page＝1 token（p4〔複核修正：原寫 p5〕）〔原文〕。其餘六篇不量。 | 與 workloads_eval 的 H1（Marconi）同型；但不是「連續前綴到第一個缺口為止」的 block 版本〔判讀〕。 | KV 重用研究的主要中介變數（SGLang Fig 8 就是用它解釋效能）。 | 沒寫是以 token、block 還是請求加權〔判讀〕。 | 見 E02 |

---

## 本組對 PoC 設計的建議〔判讀〕

背景：本研究是 16K–512K 長 context、以單請求為主的 KV 分層。寫入時決定每段 KV 的狀態（GPU BF16／FP8／INT4、CPU、SSD 或不存）；讀取時沿用 Cake 的「重算＋載入」在中間會合。

### 可以照用的

1. **「SLO＝k 倍的無干擾基準」**（Sarathi-Serve 5×／25×、Splitwise 的 slowdown）。對 KV 分層，基準可以設為「KV 全部在 GPU 上的 TTFT」（理想值）或「全部重算的 TTFT」（最差值），再報結果落在兩者之間的哪個位置。這樣跨平台（3090 對 MI300X）也能比較。
2. **TTFT 包含排隊，並拆開報**：排隊、還原（載入）、重算、剩下的 prefill（參考 DistServe Fig 10 的五段拆解、Etalon 對排程延遲的批評）。
3. **TTFT 目標隨長度改變**（Etalon 的 D_p）。在 16K–512K，固定的 TTFT 門檻沒有意義。但基準不能用「全部重算」的曲線，否則只要命中就一定達標（見 Etalon 卡）。
4. **對手用「上界＋實務做法＋最差」來夾**（vLLM 的 Orca Oracle／Pow2／Max）。這對應 Tiara 的 Oracle（M4）、簡單 heuristic、全部重算或 LRU。
5. **模擬器要自報驗證誤差**（DistServe 的 SLO attainment 誤差小於 2%；Splitwise 的 MAPE 小於 3%，加上 80:20 切分與端到端 iteration 驗證）。M4 的 trace 驅動模擬照這個格式報。
6. **把命中率當自變數來掃**（SGLang 在 runtime 部分停用已匹配的 token）。可以用來掃「各層命中比例 × 長度」，畫出 TTFT 的反應曲面。
7. **用 SLO Scale 檢驗門檻是否穩健**（DistServe）。任何以門檻定義的結論，都附上門檻縮放曲線。

### 要改的

1. **負載不能「只取長度」。** 七篇中有四篇明說只用長度或連長度都合成（vLLM、Sarathi-Serve、Splitwise、Orca）〔複核修正：原寫「五篇，含 DistServe」；DistServe 只說從資料集取樣請求，沒說只用長度，見「共同模式」第 1 點〕，而且多半刻意關掉跨請求重用（vLLM chatbot、Splitwise 的特性化）。KV 分層的收益取決於「同一段內容的重用距離與時間」，所以需要有內容身分（hash）的 trace，以及兩次重用之間的真實時間間隔。Splitwise 說「文字不影響效能指標」，前提是沒有快取；一旦有前綴快取，這個前提就不成立。
2. **Poisson 只能當敏感度分析。** 經典論文用 Poisson，是因為資料集沒有時間戳（vLLM p10、DistServe p9），不是因為 Poisson 比較好。這與 intro.txt 表 16 的立場一致。
3. **主圖的 x 軸從請求率改成 context 長度。** 以單請求為主時，請求率曲線與 capacity 都沒有意義。可以改報「長度上的 capacity」：在 TTFT 不超過理想值 k 倍的條件下，能撐住的最長 context（呼應 M1 的容量懸崖）。如果有多請求實驗（例如一個長請求加上多個短請求，或多個 session 輪流回來），再補上 Sarathi 或 DistServe 式的 capacity 與 goodput。
4. **不要用 normalized latency。** 長 prompt、短輸出時，它由 prefill 主導，而且會把排程延遲攤平（Etalon p3–4）。
5. **TBT 只在「還原或重算與其他請求的 decode 共用 GPU」時才是主指標。** 純單請求時，decode 不受影響；但如果 Tiara 在背景載入或預取別人的 KV，就要報 P99 TBT 與 fluidity，並寫明是否開了 chunked prefill。
6. **長度區間要重新驗證經典結論。** 七篇的長度上限在 2K–16K（Etalon 的動機圖到 32K），其中 Orca、vLLM、DistServe 被 2048 卡死〔複核修正：原寫「七篇都在 8K 以內」，Sarathi-Serve 的 arxiv 上限 16,384、Etalon 用 rope-scaling 超過 8,192，見「共同模式」第 5 點〕。Sarathi 的長 prefill stall（Fig 1a）、Etalon 的 prefill 長度效應（Fig 1）、DistServe §7 的「長 context 下傳輸÷prefill 比例下降」，在 16K–512K 都需要重新量；尤其 DistServe 那一條是推論，不是量測。
7. **重複與信賴區間要自己補。** 七篇都沒有報，所以不能用「經典論文都沒做」當理由。這與 intro.txt 表 16「每點至少 3 次，附信賴區間」一致。

### 指標名詞的約定（建議寫進論文的 methodology）

* TTFT：從請求到達到第一個輸出 token，包含排隊；另外拆出還原、重算與 prefill。
* TPOT：每個請求、不含第一個 token 的平均（DistServe 定義）。
* TBT：每個 token 一個值，報 P50 與 P99，母體寫成「token」。
* 不使用「ITL」；必須用時，附上工具名稱與版本（見 E02）。
* goodput：寫成「在 {TTFT ≤ a·基準, TPOT ≤ b} 且 attainment ≥ 90% 下的最大請求率」，並註明是否以每張 GPU 正規化。

---

## 未查證清單

1. 各論文的會議版（vLLM SOSP、SGLang NeurIPS、Sarathi-Serve OSDI、DistServe OSDI、Splitwise ISCA）與所讀的 arXiv 版，沒有逐字對照。
2. vLLM 論文所用的 ShareGPT 版本、清理與過濾方式、是否強制輸出長度：論文沒寫。程式碼中的過濾規則由 E02 負責。
3. Orca 的權重如何產生、FasterTransformer 版本、每個負載點的請求數。
4. Sarathi-Serve 的 capacity 搜尋方法（二分法或步長）、Orca 對手的實作方式、Fig 12 的 3.5× 是 Yi-34B 還是 Mistral-7B、abstract 的 Mistral-7B 2.6× 對應哪一根長條。〔複核修正：5.6× 已在 Fig 11a 找到（Falcon-180B、SLO-R、5.62x，讀圖判定對手是 Orca），不再列為不一致〕Artifact README 沒有讀。
5. DistServe 的每點請求數、LongBench 的截斷方式、vLLM 與 DeepSpeed-MII 的版本；SLO attainment 是否確實是「兩項同時達標」（推定）。
6. Splitwise 的 TBT percentile 母體、slowdown 的分母、「縮成每秒 2 個請求」的方法、abstract 的 2.35× 與正文的 2.15× 如何對應、Table III 中 Llama2-70B 的 head 數；公開的 Azure 2023 trace 的檔內時間戳（2023-11-16、約 1 小時）為何與論文和 README 寫的 November 11th、20 分鐘不一致。〔複核修正：原寫「公開 trace 與論文所用 trace 的關係」；資料集 README 已明說它就是論文用的 sample，關係不再是未查證，只剩時間戳不一致的原因未知〕
7. Etalon 的發表場所、scheduling slack 的數值、D_p 擬合函數的形式、capacity 搜尋步驟。
8. SGLang 的 "Normalized" 長條以哪個系統為基準；每個 benchmark 跑幾個實例。
9. GPT-3 架構（Orca）的注意力類型；Sarathi-Serve 與 Splitwise 的權重 dtype。
10. 圖中的數值（曲線、長條高度）都沒有數位化；本卡只採正文寫出的數字。

---

## 複核紀錄

- **複核者**：V01（獨立子 agent，未參與抽取，沒有讀抽取者的推理或筆記）
- **日期**：2026-10-07
- **原文來源**：抽取者下載的 7 份 PDF（scratchpad `E01/`），V01 自行用 `pdftotext -layout`／`-raw` 重新轉文字到 scratchpad `V01/`，沒有用抽取者轉好的文字檔。先確認版本：每份 PDF 第 1 頁的 arXiv 戳記分別為 2309.06180v1（2023-09-12）、2312.07104v2（2024-06-06）、2403.02310v3（2024-06-17）、2401.09670v3（2024-06-06）、2311.18677v2（2024-05-20）、2407.07000v2（2024-08-30），與卡上所寫一致；Orca 為 USENIX 論文集 PDF，PDF p2＝p521，確認「PDF 頁＋519」的換算。arXiv v1 標題 "Metron: …" 另以 arxiv.org/abs/2407.07000v1 確認。HF config 7 個重新抓取確認（attention head／KV head 數全部吻合）。
- **額外下載**（scratchpad `V01/`）：Azure `AzureLLMInferenceDataset2023.md` 與兩個 CSV（repo HEAD 215becd），用來檢查 Splitwise 卡的指控。
- **看圖**：只渲染了需要的 4 頁（Sarathi p11 的 Fig 10–11、DistServe p10 的 Fig 8、Splitwise p4 的 Fig 5、Etalon p8 的 Fig 5）。

### 檢查格數與判定

| 範圍 | 檢查格數 | ✅ | ❌ | ⚠️ |
|:--|--:|--:|--:|--:|
| Orca（3 個表頭欄＋21 列） | 24 | 23 | 0 | 1 |
| vLLM | 24 | 23 | 1 | 0 |
| SGLang | 24 | 20 | 2 | 2 |
| Sarathi-Serve | 24 | 23 | 1 | 0 |
| DistServe | 24 | 24 | 0 | 0 |
| Splitwise | 24 | 20 | 1 | 3 |
| Etalon | 24 | 22 | 1 | 1 |
| 本組的共同模式（6 點） | 6 | 2 | 2 | 2 |
| 指標字典（16 列） | 16 | 13 | 1 | 2 |
| 讀了哪些來源（7 列） | 7 | 7 | 0 | 0 |
| PoC 建議中的事實陳述（可以照用 7＋要改 7） | 14 | 12 | 2 | 0 |
| **合計** | **211** | **189** | **11** | **11** |

〔判讀〕格只檢查有沒有把推論寫成事實；判讀本身的對錯不評。

### 對使用者文件與既有整理的指控（逐條）

| # | 指控（出處格） | 判定 | 依據 |
|:--|:--|:--|:--|
| 1 | intro.txt 表 8 把 vLLM [40] 列在「LRU、ARC」那一列，但 SOSP 論文沒有 LRU／ARC，只有序列級 all-or-nothing（vLLM 卡） | ✅ 成立，但收窄 | vLLM §4.5 p8 確有 all-or-nothing、FCFS、最晚到的先搶佔；全文 0 筆 LRU／ARC。不過 §4.5 自己把這段寫成經典的逐出問題，所以把 [40] 放在「空間不夠時的逐出」這一欄沒有錯，錯的只是與 LRU／ARC 配在同一列 |
| 2 | workloads_eval §6.3 C.2 的「vLLM 原生 lru、arc」指現行程式碼（vLLM 卡） | ✅（澄清，不是錯誤指控） | workloads_eval L531 本身沒有引用 SOSP 論文 |
| 3 | intro.txt 表 8 的 SGLang、PAPERS_BY_LEVEL L361 沒有錯（SGLang 卡） | ✅ | SGLang §3 p4：LRU、先逐出葉節點；L361 的前綴語意與 radix tree 一致 |
| 4 | workloads_eval §6.3 B.2 的 goodput 沒寫 attainment 目標、是否以每張 GPU 正規化（DistServe 卡；指標字典 Goodput 列） | ✅ 成立（屬簡化） | L517 原文；DistServe p2 的 per-GPU goodput 定義、p10 的 90%、App C 的 99% |
| 5 | workloads_eval §6.3 B.1 把「TPOT／ITL」寫成一項（DistServe 卡） | ✅ 成立（屬含糊） | L516 原文；七篇 ITL 皆 0 筆 |
| 6 | 指標字典 TPOT 列「把 TPOT 與 ITL／TBT 當成同一件事（L516）」 | ⚠️ 部分成立，措辭已改 | L516 只有「TPOT／ITL」，沒有 TBT |
| 7 | **workloads_eval L240、L244「Azure 2023 那份就是 Splitwise 的資料」是錯的，公開檔很可能不是論文用的 trace（Splitwise 卡）** | **❌ 不成立** | Splitwise p3 說 "released a subset of our traces"，參考文獻 [4] 正是這份 md；資料集 README 明說這就是論文所用的 sample，並用它重現論文的特性化圖。日期與時長不一致是真的（CSV 為 2023-11-16 18:15:46–19:14:19；論文與 README 寫 November 11th、20 分鐘），但推不出「不是論文用的 trace」。**workloads_eval 的寫法不需要改**；頂多補一句「時間戳與文字描述不一致」 |
| 8 | Orca、Sarathi-Serve、Etalon、DistServe 都沒有被既有檔收錄 | ✅ | V01 對 workloads_eval、PAPERS_BY_LEVEL、intro.txt、sota.txt 做不分大小寫搜尋，四個名稱（含 Metron）皆 0 筆 |
| 9 | 「Poisson 只當敏感度分析」「每點至少 3 次附 CI」與 intro.txt 表 16 一致（PoC 建議） | ✅ | intro.txt 表 16，L820 與 L830 |
| 10 | Etalon 把 vLLM 的 normalized latency 誤述為中位數（Etalon 卡；對象是 Etalon，不是使用者文件） | ✅ | Etalon p3 引用 [30]＝Orca、[25]＝vLLM；vLLM p10 寫 mean |

### 逐條修改（原內容 → 新內容；出處）

1. **共同模式 1**：「vLLM、Sarathi-Serve、DistServe 都只取資料集的 input／output 長度」→ 只有 vLLM 與 Sarathi-Serve 明說；DistServe 只寫「從資料集取樣請求」（§6.1 p9），LongBench 被 capped 到 2048（註 4 p9）反而暗示送真實內容。❌
2. **共同模式 2**：「倍數取在對手已經飽和的膝點附近」原標〔原文〕→ 改標〔判讀〕，並註明 SGLang、Etalon 的主張不是這個形式。⚠️
3. **共同模式 5**：「長度都在 8K 以內」→「上限落在 2K–16K（Etalon 動機圖到 32K）」；依據 Sarathi Table 2 p10（arxiv prompt P90 12,985、上限 16,384）、Etalon §5.2 p8（rope-scaling >8192）、Fig 1 p4、Fig 5c p8。Splitwise「不重用 KV」補上原文限定 "For this characterization"（p3）。❌
4. **共同模式 6**：「七篇都沒有報重複次數」→ 排除 Etalon 公開 API 每小時一次共 24 次（p7）。⚠️
5. **Orca 主圖形狀**：「x 軸是達成的 throughput」原標〔原文〕→ 軸名只寫 "Throughput (req/s)"（Fig 10 p12），「達成的」改標〔判讀〕；補上 y 軸刻度 10²、10³ 證實 log 刻度。⚠️
6. **vLLM 對本研究的意義**：「記憶體管理的收益就消失」→「明顯縮小」；原文 "less pronounced"（§6.2 p11）。❌
7. **vLLM 與既有整理不一致**：保留指控，加註收窄理由（§4.5 p8 "Which blocks should it evict?"）；② 加註只是澄清（workloads_eval L531）。✅（加註）
8. **SGLang 品質指標**：「因為不開會改變結果的優化，所以不量品質」→ 前半是原文（p7），因果改標〔判讀〕。⚠️
9. **SGLang 設計理由〔判讀〕**：page size＝1 token 的出處 §3 p5 → §3 p4。❌（頁碼）
10. **SGLang 原文沒講清楚的地方**：「LRU 逐出時的容量設定」→ 補上 §3 p4：不預先配置固定 cache pool，快取與執行中的請求共用記憶體池。⚠️
11. **SGLang 與既有整理不一致**：出處 §3 p5 → §3 p4。❌（頁碼）
12. **Sarathi-Serve 原文沒講清楚的地方**：刪掉「abstract 說 Falcon 最多 5.6 倍，但正文寫 4.3／3.6 倍」這個不一致 → Fig 11a（p11）Falcon-180B SLO-R 標 "5.62x"，讀圖判定對手是 Orca；§5.3 的 4.3×／3.6× 是對 vLLM。保留 Fig 12 caption 與正文的 3.5× 不一致（p12，已確認）。另記：abstract 的 Mistral 2.6× 在 Fig 10 標籤找不到。❌
13. **DistServe 系統指標**：補上 abstract p1 "within both TTFT and TPOT constraints"（支持兩項同時達標），以及 §4.1 p7 在 prefill instance 搜尋時 attainment 只看 TTFT。✅（補充）
14. **DistServe 與既有整理不一致**：兩項指控加註 ✅。
15. **Splitwise 重用結構**：補上原文限定 "For this characterization"（§III p3）。⚠️
16. **Splitwise 主要結果**：0.89／0.75 RPS/$ 補上「Baseline-A100 與 Splitwise-AA 並列、Splitwise-HH 與 Baseline-H100 並列，Splitwise 在這一項沒有優勢」（§VI-E p11）。⚠️
17. **Splitwise 與既有整理不一致**：原指控整段加刪除線，改為 ❌ 不成立及理由（見上表第 7 條）。❌
18. **Splitwise 對本研究的意義〔判讀〕**：「刻意丟掉了內容與重用」→「沒有內容（因隱私看不到）、特性化時刻意不重用」；內容不是刻意丟掉，是看不到（p3）。⚠️
19. **Etalon 模型**：刪掉「Fig 5 另有 Llama3-8B」→ 渲染後的 Fig 5（p8）只有 Mixtral-8x7B 與 Llama3-70B，caption 也只列這兩個；Llama3-8B 只在 PDF 文字層。❌
20. **Etalon 設計理由〔判讀〕**：補上原文 §4.1 p7 只把 vLLM 當「建議」的基準，並要求在被評估的系統上重量 prefill 曲線；§6 p9 自承黑箱系統的 D_p 難定。⚠️
21. **Etalon 與既有整理不一致**：加註 ✅ 與參考文獻頁碼。
22. **指標字典 TPOT 列 ③**：「TPOT 與 ITL／TBT」→「TPOT 與 ITL（L516 寫作 TPOT／ITL）」。⚠️
23. **指標字典 Throughput req/s 列**：Orca「達成的 req/s」→ 軸名原文、「達成的」改〔判讀〕。⚠️
24. **指標字典 Cache hit rate 列**：page＝1 token 的出處 p5 → p4。❌（頁碼）
25. **PoC 要改的 1**：「五篇只用長度（含 DistServe）」→「四篇（vLLM、Sarathi-Serve、Splitwise、Orca）」。❌
26. **PoC 要改的 6**：「七篇都在 8K 以內」→「2K–16K，Etalon 動機圖到 32K；三篇被 2048 卡死」。❌
27. **未查證清單 4**：移除 5.6× 的不一致，改列 Mistral 2.6× 的對應。
28. **未查證清單 6**：「公開 trace 與論文 trace 的關係」→「時間戳與文字描述為何不一致」。
29. 檔頭狀態列改為「已獨立複核」。

### 確認無誤、值得一提的項目（抽樣，非全部）

- 所有主要倍數都在原文找到對應：Orca 36.9×（0.185 對 6.81 req/s，p13）、vLLM 1.7–2.7×／2.7–8×／22×（p11）、SGLang 6.4×／3.7×（p1、p7）、Sarathi 4.0×／3.7×／6.3×／4.3×（p11）、DistServe 7.4×／12.6× 與各應用倍數（p10–11）、Splitwise 2.15×／1.18×／1.4×（p10）、Etalon 600 tok/s 與 4×（p8）。
- SLO 門檻：Sarathi Table 3、DistServe Table 1、Splitwise Table VI、Etalon D_d 25／50／100 ms，逐值對過。
- 長度：vLLM 161.31／337.99、19.31／58.45；Sarathi Table 2 全部 12 個數；DistServe Fig 7 六個平均值；Splitwise 1500／13、1020／129。
- 七篇 ITL 0 筆、TPOT 只出現在 DistServe 與 Etalon、goodput 在 Sarathi 與 Etalon 只出現在參考文獻：V01 重新搜尋確認。

### 沒有檢查的部分（如實記錄）

- 圖中曲線與長條的數值：只渲染 4 頁，其餘圖只核對 caption 與正文；「依圖軸」類敘述只核了 Splitwise Fig 5a（到 8,192）、Etalon Fig 5c（到 32,768）、DistServe Fig 8（SLO Scale 由大到小）。
- 會議版（SOSP／NeurIPS／OSDI／ISCA）與 arXiv 版沒有逐字對照，與抽取者相同。
- 「指標名詞的約定」5 條是純判讀，沒有檢查。
- 「未查證清單」只更新了 4、6 兩條，其餘沒有替抽取者補查。
- workloads_eval 自算的 Azure 筆數（8,819／19,366）與中位數不在本卡範圍，沒有核。
- 卡中所有「見 E02」的工具定義沒有檢查。
- 〔判讀〕格只檢查事實與推論有沒有混寫，判讀的合理性沒有評。
