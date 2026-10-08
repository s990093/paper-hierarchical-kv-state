# E07 壓縮與量化：評測卡（逐出／保留、量化、傳輸壓縮與非前綴重用）

> 抽取者：E07（子 agent），抽取日期 2026-10-06。狀態：**已複核（V07，2026-10-07）**，修正見各格〔複核修正〕〔複核補充〕與檔尾「複核紀錄」。
> 頁碼一律為 **PDF 頁**（`pdftotext -layout` 以換頁符切頁）；§、表、圖編號照原文。
> 「判讀」只代表抽取者的推論，不是原文主張。

## 範圍

KV 壓縮／量化這一派（演算法派）的評測世界：預算怎麼定義、用哪些 benchmark、系統指標怎麼量、為什麼這樣設計。九篇：

- 逐出／保留：StreamingLLM、H2O、SnapKV、PyramidKV
- 量化：KIVI、KVTuner、KVQuant
- 傳輸壓縮與非前綴重用（UChicago，LMCache 同團隊）：CacheGen、CacheBlend

## 來源清單（全部讀 arXiv PDF 全文；查證 2026-10-06）

| 簡稱 | 讀的版本 | URL | 版本說明 | 程式碼（只查 GitHub API 最新 commit；有讀 README 者另註） |
|:--|:--|:--|:--|:--|
| StreamingLLM | arXiv **v4**（2024-04-07） | https://arxiv.org/abs/2309.17453v4 | 頁首標 ICLR 2024 camera-ready；附錄 A–I | mit-han-lab/streaming-llm @2e5042606d（未讀碼） |
| H2O | arXiv **v3**（2023-12-18） | https://arxiv.org/abs/2306.14048v3 | 頁首標 NeurIPS 2023；正文＋附錄 A–C 全讀，**附錄 D（理論證明，p32–48）只看目錄** | FMInference/H2O @ac75c2a8a9（未讀碼） |
| SnapKV | arXiv **v2**（2024-06-17） | https://arxiv.org/abs/2404.14469v2 | 頁首仍標 "Preprint. Under review."；**NeurIPS'24 camera-ready 未讀** | FasterDecoding/SnapKV @e216ddc84c（讀 README） |
| PyramidKV | arXiv **v4**（2025-05-15） | https://arxiv.org/abs/2406.02069v4 | 頁首仍標 "Preprint. Under review."；附錄 A–R | Zefan-Cai/PyramidKV（KVCache-Factory）@68cd9551a6（讀 README） |
| KIVI | arXiv **v2**（2024-07-25） | https://arxiv.org/abs/2402.02750v2 | ICML 2024（PMLR 235）版面；附錄 A–D | jy-yuan/KIVI @876b4d2d08（讀 README） |
| KVTuner | arXiv **v5**（2025-11-20） | https://arxiv.org/abs/2502.04420v5 | ICML 2025（PMLR 267）版面；附錄 A–F（附錄的熱圖／曲線只讀標題與說明文字） | cmd2001/KVTuner @96dd05eb2f（未讀碼） |
| KVQuant | arXiv **v6**（2025-05-28） | https://arxiv.org/abs/2401.18079v6 | 頁首標 NeurIPS 2024；附錄 A–S | SqueezeAILab/KVQuant @57a238357f（未讀碼） |
| CacheGen | arXiv **v6**（2024-07-19） | https://arxiv.org/abs/2310.07240v6 | SIGCOMM'24 版面；原文註明附錄未經同行審查（p17） | UChi-JCL/CacheGen @6bed34ca9d（未讀碼） |
| CacheBlend | arXiv **v3**（2025-04-03） | https://arxiv.org/abs/2405.16444v3 | EuroSys'25 版面 | 程式碼在 LMCache/LMCache（原文 p1） |

另外核對：`docs/research_20260924/workloads_eval.md`（KIVI、KVTuner 在 §1.2 第 90–91 行；CacheBlend 在 §1.1 第 65 行）、`docs/PAPERS_BY_LEVEL.md`（KIVI、KVTuner 第 115–127 行；CacheBlend 第 363–369 行）、使用者文件 `intro.txt`（表 5、表 6）與 `sota.txt`（表 S6）。

---

## 本組的共同模式（跨論文比較）

**P1. 「預算」的單位彼此不可比，而且多數不算 metadata。** 〔原文〕各篇出處見下表；〔判讀〕跨篇直接比「壓縮比」沒有意義。

| 方法 | 預算的單位 | 分母／基準 | 有沒有算 metadata（scale、zero、索引、residual） |
|:--|:--|:--|:--|
| StreamingLLM | 絕對 token 數「x 個 sink + y 個最近」（如 4+1020） | 無（固定 cache） | 不適用 |
| H2O | prompt 長度的 4／10／20／60%（App A p20），heavy hitter 與 recent 各半（p7） | prompt 長度 | 不適用；未討論累積分數的儲存 |
| SnapKV | 每 head 保留的 prompt token 數 1024／2048／4096，含觀察視窗（p9） | 輸入長度（1024／約 13K → 92%，p9） | 不適用 |
| PyramidKV | **平均每層** token 數 64–2048，層間等差分配（p5–6） | 序列長度（Table 2 p9 以 8192 為分母）；與摘要的 12%／0.7% 對不上（見卡片） | 不適用 |
| KIVI | 名目 2／4 bit；group 32；最近 R=128 token 保持 FP16（p6） | 16-bit | **沒有**：原文只說 residual 的記憶體可忽略（p5）；KVQuant 以含 metadata 的假設估 KIVI-2 約 3.05–3.17 bit（KVQuant p8–9）〔二手〕 |
| KVTuner | 「等效位元」f_m(P)=ΣP/(2L)，每層 (K,V) 精度對（p6 §5.1） | 16-bit | **沒有**：公式只算 K/V 位元平均〔原文 p6；判讀〕 |
| KVQuant | 平均位元數（nuq4／3／2 ＋ 0.1–1% 稀疏離群值） | fp16，以 128K 序列估算（p22 App M） | **有**：NUQ／NF 的 zero 與 offset 各 16-bit（整數量化則假設低精度 offset＋16-bit scale）；稀疏矩陣的 per-token 索引 32-bit、元素值與 per-element 索引 16-bit（CSR 為 32-bit 列＋16-bit 欄與值；CSC 為 32-bit 欄＋16-bit 列與值）（p22）〔複核修正：原寫「32-bit 列＋16-bit 欄與值」只對 CSR 成立，Key 用的是 CSC，見 p22 App M、p26 App R〕 |
| CacheGen | 編碼後位元流大小（MB） | 8-bit 量化等基線的 MB（p2 Table 1） | 以實際位元流計；每模型離線機率表是否計入未說明〔未查證〕。〔複核補充〕原文對機率分布的來源前後不一：§5.2（p6）說每個 LLM 離線 profile 一次、所有 KV 共用；§6（p8）說由「對應 context」的量化符號頻率統計而得。若是後者，分布表是逐 context 的 metadata |
| CacheBlend | 每層重算的 token 比例 r%（預設 15%） | 全重算 | 不適用（KV 不壓縮） |

**P2. Query-aware 是逐出派的預設；量化派與傳輸派是 query-agnostic。**〔原文〕SnapKV／PyramidKV 用 prompt 末端的觀察視窗（含問題）投票（SnapKV p4–5；PyramidKV p5）；H2O 累積 prompt（含問題）與生成 token 的注意力（p6、p20）；StreamingLLM 只看位置；KIVI／KVTuner／KVQuant 不丟 token；CacheGen 明說不需要知道 query，所以能離線壓縮（p12 §8）。關鍵證據：**SnapKV 自己的 §4.2.1 與 Fig. 4（p5–6）顯示同一份文件換一個指令，被選中的重要位置重疊率下降**。〔判讀〕對「寫入時決策」（下一輪問題尚未出現）而言，query-aware 的選擇不能假設對下一輪有效；只有 query-agnostic 的方法（量化、CacheGen 式編碼）能在寫入時安全使用。

**P3. 品質評測＝單請求、跟 full KV 比、多半 ≤32K，「近乎無損」沒有共同門檻。**〔原文〕LongBench 都被截斷：StreamingLLM 3,500（p17）、KIVI 4,096／Mistral 8,192（p6）、PyramidKV 依模型 8K／32K（p26）、KVQuant 31,500（p8）。超過 32K 的只有 NIAH／passkey 類（SnapKV 到 380K，p7）、PPL 串流（StreamingLLM 到 4M，p7）、或沒有 full 基線的表（PyramidKV 128K，p25）。〔複核修正：還有兩個例外——SnapKV 的 Command-R 在 Cohere 內部 RAG citation benchmark 上跑 20K–40K，並報告相對 baseline 的差（F1 −1.2%，SnapKV p10 Table 3）；StreamingLLM 的 StreamEval（逐行檢索問答）到約 120K（StreamingLLM p8 Fig. 9）。前者是內部資料無法重現，後者屬檢索類〕「無損」門檻：KIVI 約 2% 準確度（p6–8）、KVQuant <0.1 PPL（p1）、CacheGen ≤2% 準確度／<0.1% F1／<0.1 PPL（p9）、CacheBlend ≤0.02 F1／Rouge-L（p12）、KVTuner 以紅／橘色標示退化程度但沒給數值門檻（p8 Table 5 說明）。

**P4. 系統指標多是「省記憶體 → 更大 batch → 吞吐」或 per-token decode 微基準，九篇都沒有真實到達過程。**〔原文〕KIVI：ShareGPT 合成（平均輸入 161／輸出 338），batch 加到 OOM（p7–8）；H2O：FlexGen、合成等長 prompt、T4 上開 CPU offload（p9）；SnapKV、StreamingLLM：HF 實作的 ms/token（SnapKV p8、StreamingLLM p9）；KVQuant：batch 1 的 matvec kernel μs（p10）；KVTuner：固定 BS 64／16／8、輸入 128–1024（p9）。只有 CacheGen（含網路載入的 TTFT、併發請求數，p8–10）與 CacheBlend（TTFT、吞吐 vs 平均請求率，p11）像 serving，但到達分布都未說明。〔判讀〕原因：這一派的主張是「每個請求的品質—記憶體取捨」，KV 的生命期就是一個請求，到達過程不改變準確度；系統收益靠「容量換 batch」間接論證。

**P5. 量化誤差有沒有進入 prefill，各篇不同，直接影響 ε 的意義。**〔原文〕KIVI：prefill 時往下一層傳的是精確 K/V，只有存下來的 KV 被量化（p5 §3.3）；KVQuant：prefill 的注意力用 fp16 K/V 計算後才壓縮（p22 App M）；KVTuner：**刻意**在 prefill 也用反量化 KV，以放大誤差累積（p7、p23）；CacheGen／CacheBlend：載入的（壓縮或預算的）KV 被新 prompt 的 prefill 使用（CacheGen p3 Fig. 2、CacheBlend p5 Fig. 4）。〔判讀〕使用者的情境是「KV 以低精度存著，下一輪新 token 的 prefill 要讀它」，對應後兩類；KIVI／KVQuant 的品質數字會低估這個情境的 ε。

**P6. 與 PagedAttention／FlashAttention 的相容性很少被量。**〔原文〕逐出與量化七篇中，PyramidKV 是唯一描述 vLLM 實作的〔複核修正：原寫「唯一描述 vLLM 實作的」與同段 CacheBlend 實作在 vLLM 上（CacheBlend p9）矛盾，限定為逐出＋量化七篇〕：在標準 paged attention 下，逐層不同預算只能按「壓縮率最低的層」省記憶體，其餘只增加碎片，需要逐層 block table（p36 App R）；KVTuner 以「token 級混合精度難以整合 FlashAttention 與 vLLM」作為選層級粒度的理由（p2），宣稱可套用到 vLLM 但只在 HF 上實作（p17）；CacheBlend 實作在 vLLM 上（p9）；KIVI／KVQuant 自寫 kernel 但在 HF／自建 pipeline 上（KIVI p6；KVQuant p9）；SnapKV／PyramidKV 的 FlashAttention v2 路徑只在 README 出現〔文件〕。

---

## 逐篇評測卡

### StreamingLLM：Efficient Streaming Language Models with Attention Sinks（ICLR'24；arXiv 2309.17453）

- **讀了什麼**：〔全文〕arXiv v4，https://arxiv.org/abs/2309.17453v4 ，查證 2026-10-06（含附錄 A–I）
- **一句話**：只留開頭幾個 attention sink＋最近視窗，讓模型無限長串流生成。
- **評測要證明的主張**：window attention 一逐出開頭 token 就崩潰；保留 4 個 sink＋rolling cache、不微調就能在 4M token 上維持穩定 PPL；比 sliding window w/ re-computation 每 token 快到 22.2×；預訓練時加一個專用 sink token 有助串流（p1、p6–9）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | Llama-2-7B/13B/70B、MPT-7B/30B、Falcon-7B/40B、Pythia-2.8B/6.9B/12B；Llama-2-7B/13B/70B-Chat；LongChat-7b-v1.5-32k、Llama-2-7B-32K-Instruct；自行預訓練 160M 模型。位置編碼：Llama-2／Falcon／Pythia 用 RoPE，MPT 用 ALiBi。注意力類型與 KV dtype 原文未標 | 〔原文〕p6 §4、§4.1；p7 §4.2；p8 §4.3 |
| 硬體 | 效率實驗：單張 NVIDIA A6000；預訓練：8×A6000；品質實驗硬體未說明 | 〔原文〕p9 §4.5；p7 §4.2 |
| 軟體與版本 | 兩種方法都用 HF Transformers 實作；RoPE 模型存 **pre-RoPE key**，每個 decode 步重新套位置；位置用 cache 內位置而非原文位置。版本未說明 | 〔原文〕p5 §3.2；p9 §4.5 |
| 資料／負載 | PG19 測試集串接（100 本書）；PG19 第一本書（65K token）；400K token 的串接 PG19；ARC-Easy/Challenge 全部 QA 串成一條流；自建 StreamEval（每 10 行問一次，答案在 20 行前；100 samples，每個 100 個 query，每行 23 token）；LongBench 6 子集（NarrativeQA、Qasper、HotpotQA、2WikiMQA、GovReport、MultiNews） | 〔原文〕p2 Fig. 1；p5 Table 1–2；p6 §4.1；p8 §4.3；p16 App C；p17 App D |
| 長度 | PPL 曲線 20K（Fig. 3）到 4M token（Fig. 5）；StreamEval 到約 120K token；LongBench 預設截到 3,500（頭尾各 1,750）。單位 token | 〔原文〕p4；p7；p8；p17 |
| 到達與併發 | 無到達過程；單一串流；效率實驗的 batch 未說明 | 〔原文〕p9；batch〔未查證〕 |
| 重用結構 | 無跨請求重用；多輪 QA 以「串接成一條流」模擬 | 〔原文〕p8 §4.3 |
| 掃描的自變數 | sink 數 0／1／2／4／8；cache 大小（4+252…4+4092）；模型族與尺寸；效率掃 cache 256–4096；StreamEval 掃問答距離與 cache 4+2044…4+16380 | 〔原文〕p5 Table 2；p9 Table 6、Fig. 10；p16 Table 7 |
| 對手 | dense attention、window attention、sliding window w/ re-computation（品質上視為 oracle）；LongBench 對「截斷 1750+1750」 | 〔原文〕p6 §4–4.1；p17 Table 8 |
| 系統指標 | 每 token decode 延遲（ms）與記憶體（GB）對 cache 大小；無 percentile、無 SLO | 〔原文〕p9 Fig. 10 |
| 品質指標 | PG19 PPL；ARC 完全匹配；StreamEval 準確率；LongBench 官方指標；比 dense／逐樣本 one-shot | 〔原文〕p5–8；p17 |
| 主要結果 | Llama-2-13B、PG19 第一本：window（0+1024）PPL 5158.07，4+1020 為 5.40；每 token 最高 22.2× 加速；LongBench 上 4+3496 輸給截斷（NarrativeQA 11.6 vs 18.7），改成 1750+1750 才回到 18.2；StreamEval 問答距離超過 cache 後準確率歸零 | 〔原文〕p5 Table 1；p9 Fig. 10；p17 Table 8；p16 Table 7 |
| 消融／敏感度／開銷 | sink 數、cache 大小、預訓練 sink token（含 2 個 sink）、與 recompute 的記憶體比較 | 〔原文〕p5；p9；p6 Table 3；p21 App I |
| 重複與統計 | StreamEval 平均 100 samples；其餘未說明。Fig. 12 的誤差棒是 head 之間的標準差，不是重複實驗 | 〔原文〕p8 Fig. 9；p16；p18 Fig. 12 |
| 程式碼／資料 | 公開：https://github.com/mit-han-lab/streaming-llm | 〔原文〕p1；〔文件〕GitHub API 2026-10-06 |
| 設計理由（原文） | cache 取預訓練視窗的一半是為了圖好看；4 個 sink 足夠、1–2 個不夠；cache 內位置是功能關鍵；只比 recompute 是因為它是唯一品質可接受的基線；附錄自承不適合長文件 QA 與摘要 | 〔原文〕p6 §4.1；p5、p9 §4.4；p5 §3.2；p9 §4.5；p15 App A |
| 設計理由〔判讀〕 | 評測目標是「無限串流下的穩定」而非「長文件理解」，所以主評測用 PPL 與近距離問答；LongBench 放附錄且輸給截斷 | 〔判讀〕 |
| 原文沒講清楚的地方 | 效率實驗的 batch、prompt 與生成長度；品質實驗硬體；transformers 版本；LongBench 1750+1750 的設定下長 prompt 的 prefill 怎麼處理 | 〔原文〕查無 |
| 與既有整理不一致 | workloads_eval 沒有獨立列（只在他篇對手中出現，§1.3 統計 7 次）；使用者文件未直接描述，無可對照 | 〔原文〕workloads_eval 第 125 行 |
| 對本研究的意義〔判讀〕 | (1) sink 效應：除最底兩層外，第一個 token 的注意力常超過一半（App F p18）→ 若前段「不存／降精度」，開頭幾個 token 仍應全精度保留（與 KVQuant 保留第一個 token fp16 一致，KVQuant p6 §3.5）。(2) 位置重排需要 pre-RoPE key，與 vLLM 存 post-RoPE KV 的佈局不同。(3) 不可逆地丟掉中段，不適合放進使用者的動作空間，只適合當「最便宜的有損基線」 | 〔判讀〕 |

**壓縮設定**

| 項目 | 內容 | 出處 |
|:--|:--|:--|
| 預算的定義 | 絕對 token 數 x+y（sink＋最近），如 4+1020、4+2044；原文沒有逐層或逐 head 的差別分配 | 〔原文〕p5 Table 2；p6 §4.1 |
| 是否 query-aware | 否，純位置規則 | 〔原文〕p5 §3.2 |
| 作用階段 | 生成／串流過程中持續滾動逐出 | 〔原文〕p5 Fig. 4 |
| 品質怎麼量 | PG19 PPL、ARC 串流完全匹配、StreamEval、LongBench 6 子集；比 dense、recompute、截斷 | 〔原文〕p5–8、p17 |
| 系統指標與相容性 | HF 實作、單 A6000、per-token 延遲與記憶體；batch 未說明；未討論 FlashAttention／PagedAttention | 〔原文〕p9；相容性〔原文〕查無 |
| 壓縮比是否計 metadata | 不適用（不量化，也沒有索引成本討論） | 〔原文〕 |

---

### H2O：Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models（NeurIPS'23；arXiv 2306.14048）

- **讀了什麼**：〔部分〕arXiv v3，https://arxiv.org/abs/2306.14048v3 ，查證 2026-10-06。正文與附錄 A–C 全讀；附錄 D（理論證明）只看目錄。
- **一句話**：以累積注意力保留 heavy hitter＋最近 token 的固定預算逐出策略。
- **評測要證明的主張**：20% 預算下品質與 full KV 相當；在 FlexGen 上吞吐比 DeepSpeed ZeRO-Inference、HF Accelerate、FlexGen 高 29×、29×、3×，同 batch 延遲最多降 1.9×（p1、p8–9）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | OPT、LLaMA、GPT-NeoX-20B；圖中有 LLaMA-7B/13B/30B、OPT-30B/66B；正文宣稱尺寸「6.7B 到 175B」但未見 175B 的表；吞吐用 OPT-6.7B/30B〔複核修正：A100 的 Table 5 另有 OPT-13B（5000+5000、batch 4），p9〕。注意力類型與 dtype 原文未標 | 〔原文〕p7 §5.1、Fig. 4；p9；175B 結果〔未查證〕 |
| 硬體 | 準確度：1×A100 80GB；吞吐：NVIDIA T4 16GB 與 A100 80GB；T4 上放不下就開 CPU offloading | 〔原文〕p3、p7；p9 |
| 軟體與版本 | 實作在 FlexGen 上（OPT 白箱，預先配置 KV 記憶體、最近 K 個用 circular queue）；評測框架 lm-eval-harness 與 HELM；版本未說明 | 〔原文〕p9；p20 App A |
| 資料／負載 | HELM：XSUM、CNN/DailyMail 各 1000 筆（zero-shot）；lm-eval：COPA、MathQA、OpenBookQA、PiQA、RTE、Winogrande（預設 5-shot）；稀疏性觀察用 Wiki-Text-103；吞吐用等長 padding 的合成 prompt 與 XSUM；PG-19 第一本做 4M 串流；10 篇文件的多文件 QA（比 StreamingLLM）。正文說 AlpacaEval 與 MT-bench 的細節在附錄，但 v3 附錄找不到這兩項結果 | 〔原文〕p20 App A；p4；p9；p10 Fig. 5；p26 App C.4；p7 |
| 長度 | 0/1-shot 只有 100–300 token；5-shot 長度未說明；吞吐 512+32／512+512／512+1024（T4），7000+1024、5000+5000、2048+2048（A100），「最長到 10K」 | 〔原文〕p25 App C.3；p8–9 Table 3、5 |
| 到達與併發 | 無到達過程；吞吐用固定 batch（T4 表中括號為有效 batch，1–728；A100 為 1／4／24／64） | 〔原文〕p8–9 |
| 重用結構 | 無 | 〔原文〕 |
| 掃描的自變數 | KV 預算 4%–100%（附錄 A 列 4／10／20／60%）；shots 0／1／5／10；序列長度 | 〔原文〕p8；p20；p26 Table 10 |
| 對手 | Full；Local（只留最近）；Sparse Transformer（strided、fixed）；少 shot 的 Full；Top-K；SpAtten；StreamingLLM；系統：DeepSpeed ZeRO-Inference、HF Accelerate、FlexGen；FlexGen 的 4-bit 量化〔複核補充：量化不是對手而是「結合」實驗——Table 6 的 Quant-4bit 是品質對照；Table 7 的 H2O-c 是 H2O＋4-bit **權重**壓縮，用來放大 batch，p24 App C.2〕 | 〔原文〕p7–8；p26–27 App C.5、C.9；p25 C.4；p9；p24 |
| 系統指標 | 生成吞吐＝生成 token 數／（prompt 時間＋decode 時間），端到端、含建構 H2 cache 的時間；固定 batch 的延遲（秒）；無 percentile／SLO | 〔原文〕p9 |
| 品質指標 | 各任務 accuracy／ROUGE；與 Full 比；Self-BLEU（多樣性） | 〔原文〕p7–8；p23 |
| 主要結果 | OPT-30B、20% 預算：COPA 84.00 vs Full 85.00，PiQA 78.45 vs 78.51；T4、OPT-30B、512+512：H2O 18.83 tok/s（batch 416）vs FlexGen 8.5（80）vs Accelerate 0.6；A100、OPT-6.7B、2048+2048、batch 24：延遲 53.5 s vs FlexGen 99.5 s | 〔原文〕p8 Table 2–3；p9 Table 5 |
| 消融／敏感度／開銷 | H2 與 recent 分開保留；shots；與量化結合；無限長輸入；Top-K、SpAtten；MLP 中的 heavy hitter | 〔原文〕p10 §5.3；p24–27 |
| 重複與統計 | 未說明 | 〔原文〕查無 |
| 程式碼／資料 | 公開：https://github.com/FMInference/H2O | 〔原文〕p1；〔文件〕GitHub API |
| 設計理由（原文） | 預算平均分給 H2 與 recent；用累積而非平均注意力（平均版試過較差）；吞吐沿用 FlexGen 論文的設定；OPT 只訓練到 2K 仍跑到 10K 以展示潛力 | 〔原文〕p7；p21–22 App B.2；p9 |
| 設計理由〔判讀〕 | 系統收益建立在「省 KV → 更大 batch／免 offload」；選 T4 這種會被迫 offload 的小卡最能放大倍數（29× 的對象是 offload 到 CPU 的系統） | 〔判讀〕 |
| 原文沒講清楚的地方 | 準確度實驗的 prompt 長度與 batch；預算百分比在 decode 增長後如何計；AlpacaEval／MT-bench 結果；每 head 獨立逐出時的記憶體佈局與 padding | 〔原文〕查無 |
| 與既有整理不一致 | workloads_eval 無獨立列；CacheGen 用的是「理想化 H2O」（離線就用 prompt 的 query，CacheGen p9）——引用 H2O 當對手的數字時要注意版本；使用者文件未直接描述 | 〔原文〕CacheGen p9 |
| 對本研究的意義〔判讀〕 | 預算以「prompt 長度百分比」定義，與 SnapKV／PyramidKV 的絕對 token 數不可比；它的系統倍數在「T4＋offload」條件下被放大；可當 GPU 內有損逐出的品質下界對照，但不可逆、不跨請求 | 〔判讀〕 |

**壓縮設定**

| 項目 | 內容 | 出處 |
|:--|:--|:--|
| 預算的定義 | prompt 長度的 4／10／20／60%；H2:recent＝1:1；各層、各 head 獨立選擇但預算相同；吞吐實驗固定 20% | 〔原文〕p20 App A；p7；p27 App C.9 |
| 是否 query-aware | 間接：累積的是 prompt（含問題）與已生成 token 給的注意力 | 〔原文〕p5–6 Alg. 1；〔判讀〕問題在 prompt 末端時會影響選擇 |
| 作用階段 | prefill 結束時若 prompt ≥2K 先選 K 個 heavy hitter＋K 個 recent；decode 每步每 head 逐出一個〔複核修正（釐清）：此處「2K」是 2×K（K 為 heavy hitter 參數），不是 2,000 token，p20 App A〕 | 〔原文〕p20 App A |
| 品質怎麼量 | lm-eval 5-shot、HELM zero-shot 1000 筆，與 Full 比 | 〔原文〕p20 |
| 系統指標與相容性 | FlexGen 端到端吞吐；T4／A100；batch 由記憶體決定。FlashAttention／PagedAttention 未討論 | 〔原文〕p9；〔判讀〕需要每步的注意力分數，與不物化注意力矩陣的 kernel 不直接相容 |
| 壓縮比是否計 metadata | 不適用；未計入索引與累積分數的儲存 | 〔原文〕查無 |

---

### SnapKV：LLM Knows What You are Looking for Before Generation（NeurIPS'24；arXiv 2404.14469）

- **讀了什麼**：〔全文〕arXiv v2，https://arxiv.org/abs/2404.14469v2 ，查證 2026-10-06（含附錄 A–B）。v2 頁首仍標 "Preprint. Under review."，NeurIPS'24 camera-ready 未讀。README @e216ddc84c。
- **一句話**：用 prompt 末端觀察視窗的注意力投票，prefill 後一次把 prompt KV 壓到固定大小。
- **評測要證明的主張**：每個 head 在生成時關注的 prompt 特徵穩定，且可以由 prompt 末端預先找出；16K 輸入時生成快 3.6×、記憶體效率 8.2×；LongBench 16 個資料集與 baseline 相當；單張 A100-80GB 可處理 380K（NIAH）（p1）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | LWM-Text-Chat-1M（7B）、LongChat-7b-v1.5-32k、Mistral-7B-Instruct-v0.2、Mixtral-8x7B-Instruct-v0.1、Command-R（35B）；觀察實驗用 Mistral-7B-Instruct-v0.2。注意力類型與 dtype 未標 | 〔原文〕p9 §5.3–5.4；p5 §4.2 |
| 硬體 | NIAH 與 decode 延遲：單張 A100-80GB；LongBench、Command-R 實驗硬體未說明 | 〔原文〕p7 Fig. 6；p8 Fig. 7 |
| 軟體與版本 | HuggingFace 實作小改；README：測試於 transformers==4.37.0、flash-attn==2.4.0 | 〔原文〕p1、p7；〔文件〕README @e216ddc84c |
| 資料／負載 | Ultrachat 篩 response >512 且 prompt >3K，共 3050 筆（觀察）；QMSum／Openreview／SPACE（hit rate 分析）；LongBench 16 集；NIAH；LongEval-Lines（pooling 消融）；Command-R：NIAH（每個長度×深度的 context 排列 8 次）、Cohere 內部 RAG citation benchmark（每 prompt 100 篇，20K–40K）、bioasq（30／100／200 篇，各 3 次）、修改版 HotpotQA 的 end-to-end RAG（檢索 200 → rerank 成 100，平均約 16K）；Medusa 用 QASPER 子集 | 〔原文〕p3 Fig. 2；p5–6；p9；p7；p8；p10–11；p12 |
| 長度 | LongBench 四模型平均約 13K；NIAH 1K–380K；decode 延遲軸 4K–262K；Command-R 到 128K；RAG 8K–40K | 〔原文〕p9；p7；p8；p10–11 |
| 到達與併發 | 無到達；decode 實驗 batch 1／2／4／8 | 〔原文〕p8 Fig. 7 |
| 重用結構 | 無（單請求） | 〔原文〕 |
| 掃描的自變數 | prompt KV 預算 1024／2048／4096；batch 與輸入長度；pooling 有無；文件數與 ground-truth 位置 | 〔原文〕p9；p8；p11 Table 4 |
| 對手 | All KV；H2O（prompt capacity 設 4096）；Medusa | 〔原文〕p9；p12 |
| 系統指標 | decode 延遲 ms/token；OOM 前可處理的最長輸入；附錄的 prompting vs generation 延遲分解；無吞吐、percentile、SLO | 〔原文〕p8；p16 App A |
| 品質指標 | LongBench 官方指標；NIAH 分數；RAG F1 與 accuracy 以「相對 baseline 的 % 差」呈現 | 〔原文〕p9–11 |
| 主要結果 | 16K、batch 2：baseline decode >100 ms/token，SnapKV <40 ms（約 3.6×）；batch 2 時 baseline 16K 以上 OOM，SnapKV 到 131K（約 8.2×）；Mistral-7B 預算 1024 在 16 個資料集中 11 個優於 H2O 4096；Command-R：NIAH −0.5%、RAG citation F1 −1.2%、end-to-end −2.1% | 〔原文〕p7–8；p9；p10 Table 2–3 |
| 消融／敏感度／開銷 | pooling；指令位置；不同指令的重疊；與 Medusa 結合 | 〔原文〕p8 Fig. 8；p6 Fig. 4–5；p12 |
| 重複與統計 | NIAH 8 次排列；bioasq 每組 3 次；其餘未說明；無信賴區間 | 〔原文〕p10–11 |
| 程式碼／資料 | 公開：https://github.com/FasterDecoding/SnapKV | 〔文件〕 |
| 設計理由（原文） | 觀察視窗放在 prompt 末端，因為它與生成時的注意力重疊率最高；pooling 是為了保住被選 token 周圍的完整性（如電話號碼）；H2O 設 4096 以求公平 | 〔原文〕p3–4 Fig. 2–3；p6 §4.3；p9 |
| 設計理由〔判讀〕 | 只壓 prompt KV（不壓 decode KV），因為「長輸入短輸出」場景 prompt 是記憶體主體；評測因此集中在 LongBench／NIAH 這類長輸入短答案 | 〔判讀〕 |
| 原文沒講清楚的地方 | baseline 在 33K 就 OOM 的原因（注意力實作？）；LongBench 截斷長度；各實驗超參數為何不同（window 16／32／64、kernel 5／7／13）；計算觀察視窗注意力的額外 prefill 成本 | 〔原文〕p7、p9、p10 |
| 與既有整理不一致 | workloads_eval 未收獨立列；使用者文件只在 KVP 列提到「優於 SnapKV 等」，無可對照的設定描述 | 〔原文〕intro.txt 表 6 |
| 對本研究的意義〔判讀〕 | 最關鍵的是 §4.2.1 與 Fig. 4（p5–6）：同一份文件換指令，選出的重要位置不同 → 為某個問題壓縮的 KV 不能假設對下一輪問題仍好。使用者的寫入時決策發生在下一輪問題之前，不能直接借用這類 query-aware 選擇；若把它當對照，要用「換一個問題再問」的測法 | 〔判讀〕 |

**壓縮設定**

| 項目 | 內容 | 出處 |
|:--|:--|:--|
| 預算的定義 | 每 head 保留固定 token 數（max_capacity_prompt＝1024／2048／4096，含觀察視窗）；§4 公式另有比例定義 k＝⌊p×L_prefix⌋，但實驗用絕對數；層間相同；decode 新增的 KV 不壓 | 〔原文〕p4 §4；p9；層間相同〔判讀〕（pseudo code 無層參數，p5） |
| 是否 query-aware | 是：觀察視窗＝prompt 最後 16／32／64 個 token（LongBench 中通常含問題）；問題放開頭或結尾 hit rate 都高 | 〔原文〕p7、p9、p10；p6 Fig. 5 |
| 作用階段 | prefill 後一次（pseudo code 斷言在 prompt phase） | 〔原文〕p5 Listing 1 |
| 品質怎麼量 | LongBench 16 集 vs All KV；NIAH；RAG（內部 benchmark） | 〔原文〕p9–11 |
| 系統指標與相容性 | HF；A100；batch 1–8；decode 延遲；README 要求 flash-attn 2.4.0；PagedAttention 未討論 | 〔原文〕p8；〔文件〕README |
| 壓縮比是否計 metadata | 不適用；壓縮比＝預算／輸入長度（1024 對約 13K → 92%） | 〔原文〕p9 |

---

### PyramidKV：Dynamic KV Cache Compression based on Pyramidal Information Funneling（arXiv 2406.02069）

- **讀了什麼**：〔全文〕arXiv v4，https://arxiv.org/abs/2406.02069v4 ，查證 2026-10-06（附錄 A–R）。v4 頁首仍標 "Preprint. Under review."。README（KVCache-Factory）@68cd9551a6。
- **一句話**：層間預算呈金字塔（低層多、高層少），層內沿用 SnapKV 式選擇。
- **評測要證明的主張**：注意力隨層數由分散到集中；LongBench 上只留 12% KV 可與 full 相當，只留 0.7% 時優於其他方法（TREC 最多 +20.5）；Llama-3-70B 只留 128 個即可在 NIAH 拿 100.0（p1）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | LLaMa-3-8B-Instruct、LLaMa-3-70B-Instruct、Mistral-7B-Instruct（版本未標；附錄稱 32k）；附錄另有 Llama-3-8B-Instruct-Gradient-1048k（128K）；Mixtral-8x7B 只做注意力圖。注意力類型與 dtype 未標 | 〔原文〕p6 §5.1.1；p25 App O；p14 Fig. 6；p26 App P |
| 硬體 | NVIDIA A100（數量未說明） | 〔原文〕p17 App F |
| 軟體與版本 | 以 torch.gather 逐出；附錄描述 vLLM 實作；README：FlashAttention v2 與 SDPA 路徑；版本未說明 | 〔原文〕p16 App E；p36 App R；〔文件〕README |
| 資料／負載 | LongBench 16 個英文資料集（正文寫 17 個，表中 16 個）；NIAH（haystack 設定照 Wu et al.）；每個資料集用同一 prompt；greedy decoding | 〔原文〕p17 Table 3；p2、p6；p8；p6 |
| 長度 | LongBench 各集平均 1,235–18,409（原文稱 token）；Llama-3 為 8K、Mistral 為 32K；附錄 128K；記憶體表用 8192 | 〔原文〕p6、p17；p26；p25；p9。平均長度的單位〔判讀〕LongBench 官方英文集是 word 數，交 E09 確認 |
| 到達與併發 | 無；記憶體表 batch 1；vLLM 吞吐圖的到達方式未說明 | 〔原文〕p9；p37 Fig. 19 |
| 重用結構 | 無（vLLM 實驗明說無共享前綴） | 〔原文〕p36 App R |
| 掃描的自變數 | 平均每層預算 64／96／128／256／2048；β、α；層間分配策略（線性／幾何／指數／entropy／Gini） | 〔原文〕p7 Fig. 3、Table 1；p19 Table 4；App I |
| 對手 | FullKV、SnapKV、H2O、StreamingLLM（每層固定同預算，PyramidKV 的平均預算對齊）；附錄 PyramidInfer、MInference | 〔原文〕p6 §5.1；App J–K |
| 系統指標 | 記憶體（M）與壓縮比；[prompt, gen] 組合的總推論時間（秒）；vLLM tok/s | 〔原文〕p9 Table 2；p21–22 Table 9–10；p37 Fig. 19 |
| 品質指標 | LongBench 官方指標（F1／Rouge-L／Acc／Edit Sim）；NIAH Acc.；attention recall rate | 〔原文〕p6；p8；App N |
| 主要結果 | Llama-3-8B 預算 64：平均 34.76，SnapKV 33.05、H2O 33.89、StreamingLLM 30.43、Full 41.46；預算 2048：41.49 vs Full 41.46；記憶體 512／1024／2048 → 428M／856M／1712M，Full 6848M；推論時間與對手相近（4096+4096：138.87 s vs SnapKV 138.57 s） | 〔原文〕p7 Table 1；p9 Table 2；p22 Table 10 |
| 消融／敏感度／開銷 | 分配策略、α、β、與 MInference 結合、對 PyramidInfer、attention recall、128K | 〔原文〕App I–O |
| 重複與統計 | 未說明 | 〔原文〕查無 |
| 程式碼／資料 | 公開：https://github.com/Zefan-Cai/PyramidKV | 〔原文〕p1；〔文件〕 |
| 設計理由（原文） | 等差數列：對齊觀察到的模式、實驗最好、計算便宜；不重排 RoPE 位置以免位置序列不單調；α 取 8／16 較好、β 不敏感；設 β＝20、α＝8 | 〔原文〕p16 App E；p18 App H；p19 App I；p6 §5.1 |
| 設計理由〔判讀〕 | 為了與 SnapKV／H2O 公平比，只控制「平均每層 token 數」相等：記憶體總量相同，但 runtime 結構不同（paged 實作的碎片問題直到 App R 才承認） | 〔判讀〕 |
| 原文沒講清楚的地方 | 「12%」與「0.7%」的分母：Table 2 以 8192 為分母時 2048＝25.0%、1024＝12.5%（p9），與摘要「12%（size 2048）」不一致〔複核修正：摘要（p1）與 p7 只說「12%」，沒有寫對應 size 2048；把 12% 對到 2048 是抽取者的推定。〔計算〕以 Table 1 列出的 16 集平均長度（算術平均 118,815／16≈7,426）為分母：64≈0.86%（p8 正文寫「約 0.8%」，摘要寫 0.7%）、2048≈27.6%；以 Table 2 的 8192 為分母：1024＝12.5%。哪個分母產生「12%」與「0.7%」原文未說明〕；Table 2 的 Full 6,848M，而 Llama-3-8B（32 層、8 個 KV head、head dim 128）BF16 KV 在 8192 token 的算術值是 1,024 MiB，對不上；Table 14（128K）沒有 FullKV 列，且 H2O 與 StreamingLLM 有 11 欄數值完全相同〔複核修正：16 個資料集欄中有 **14 欄**完全相同，只有 GovReport（24.13 vs 19.21）與 TriviaQA（81.45 vs 78.21）不同，p25 Table 14〕；LongBench 截斷方式 | 〔原文〕p1、p7、p8、p9、p25；〔計算〕2×32×8×128×2 B×8192＝1,073,741,824 B，config 取自 huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct（2026-10-06；複核者 2026-10-07 重抓 config 確認 32 層／8 KV heads） |
| 與既有整理不一致 | workloads_eval 未收獨立列；使用者文件未描述 | 〔原文〕 |
| 對本研究的意義〔判讀〕 | (1) App R：標準 PagedAttention 下逐層不同預算只會按壓縮率最低的層省記憶體，其餘只增加碎片 → 使用者若要在 vLLM 裡做「逐層／逐 chunk 不同精度或保留」，需要逐層（或逐精度）的 block table；(2) 預算單位是「平均每層 token」；(3) 128K 結果沒有 full 基線，不能當長 context 品質證據 | 〔判讀〕 |

**壓縮設定**

| 項目 | 內容 | 出處 |
|:--|:--|:--|
| 預算的定義 | 總預算＝平均每層 token×層數；最上層 k＝k_total／(β·m)，最底層 2k_total／m − k_top，中間等差；另外每層保留最後 α＝8 個 token；README：`--max_capacity_prompts` 是每層目標預算，PyramidKV 在層間重分配 | 〔原文〕p5 §4.2.1；p6；〔文件〕README |
| 是否 query-aware | 是：用最後 α 個「instruction token」的注意力（SnapKV 式＋pooling） | 〔原文〕p5 §4.2.2 |
| 作用階段 | prefill 後一次；被丟的 KV 之後不再使用 | 〔原文〕p5 |
| 品質怎麼量 | LongBench 16 集 vs FullKV；NIAH | 〔原文〕p6–8 |
| 系統指標與相容性 | A100；batch 1 記憶體；FlashAttention v2／SDPA（README）；PagedAttention 要改成逐層 block table | 〔原文〕p9、p36；〔文件〕 |
| 壓縮比是否計 metadata | 不適用；壓縮比＝預算／序列長度 | 〔原文〕p9 Table 2 |

---

### KIVI：A Tuning-Free Asymmetric 2bit Quantization for KV Cache（ICML'24；arXiv 2402.02750）

- **讀了什麼**：〔全文〕arXiv v2，https://arxiv.org/abs/2402.02750v2 ，查證 2026-10-06（附錄 A–D）。README @876b4d2d08。
- **一句話**：K 逐通道、V 逐 token 的非對稱 2-bit 量化，最近 R 個 token 保持 FP16。
- **評測要證明的主張**：2-bit 時 per-channel K＋per-token V 最準；免調參的 2-bit KIVI 品質幾乎不變，峰值記憶體少 2.6×、batch 大 4×、吞吐 2.35–3.47×（p1、p8）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | Llama-2-7B/13B（及 -Chat）、Falcon-7B、Mistral-7B；附錄 Llama-3-8B-Instruct、Mistral-7B-Instruct-v0.2、LongChat-7B-v1.5-32K。**原文內部不一致**：§4.1 說 Llama 與 Mistral 是 MHA、Falcon 是 MQA；附錄表 8、9 說 Llama-3-8B-Instruct 與 Mistral-7B-Instruct-v0.2 用 GQA | 〔原文〕p6 §4.1；p9 Table 4；p14 Table 8–9 |
| 硬體 | 效率：單張 A100 80GB；準確度實驗硬體未說明 | 〔原文〕p8 |
| 軟體與版本 | HF Transformers 上實作；CUDA 融合反量化＋矩陣乘（Q_MatMul）、Triton 量化 kernel；LM-Eval；README：2024-04-04 加入 prefill flash-attention（beta），2025-01-18 才支援 GQA | 〔原文〕p6；〔文件〕README |
| 資料／負載 | LM-Eval：CoQA（EM）、TruthfulQA（BLEU）、GSM8K（EM），預設參數；LongBench 8 子集（Qasper、QMSum、MultiNews、TREC、TriviaQA、SAMSum、LCC、RepoBench-P），附錄 16 子集〔複核修正：附錄 Table 8–10 是 **15** 個子集（NarrativeQA、Qasper、MultiFieldQA、HotpotQA、MuSiQue、2WikiMQA、GovReport、QMSum、MultiNews、LCC、RepoBench-P、TriviaQA、SAMSum、TREC、PR；沒有 PassageCount），p14–15〕；NIAH（7 位數 passkey＋Paul Graham essays）；效率：依 vLLM 做法合成 ShareGPT（平均輸入 161、輸出 338） | 〔原文〕p6；p14–15；p13 App B；p7–8 |
| 長度 | LongBench 最長 4096（Mistral 8192）；NIAH 20K words ≈ 27K／30K token；吞吐輸入 161／輸出 338 | 〔原文〕p6；p7 Fig. 4；p7–8 |
| 到達與併發 | 無到達；吞吐實驗「batch 加到 OOM」 | 〔原文〕p8 |
| 重用結構 | 無 | 〔原文〕 |
| 掃描的自變數 | group size 32／64／128；residual 32／64／96／128；2／4 bit | 〔原文〕p9 Table 5 |
| 對手 | 16-bit；4-bit per-token 假量化；四種 2-bit 假量化配置（假量化＝量化後在注意力層反量化，所有 token 都量化）；效率對 FP16 | 〔原文〕p3；p8 Table 3 |
| 系統指標 | 峰值記憶體、吞吐（tokens/s）對 batch；無延遲 percentile | 〔原文〕p8 Fig. 5 |
| 品質指標 | 任務指標，與 16-bit 比 | 〔原文〕p8–9 |
| 主要結果 | Llama-2-7B：CoQA 63.88 → KIVI-2 63.05，GSM8K 13.50 → 12.74；同樣 2-bit 配置但沒有 residual 的假量化 GSM8K 只有 5.76；LongBench Llama2-7B 平均 44.52 → 44.27；Falcon-7B（MQA）2-bit 掉較多，需 4-bit；吞吐 2.35–3.47×〔複核補充〔判讀，讀 Fig. 5 圖〕：FP16 最高約 760 tok/s，KIVI R128 最高約 1,780（≈2.35×）、R32 最高約 2,630（≈3.47×）；也就是 3.47× 來自非預設的 R32〕 | 〔原文〕p8 Table 3；p9 Table 4；p6–7；p8 Fig. 5 |
| 消融／敏感度／開銷 | group size、residual 長度；R32 全表 | 〔原文〕p9 Table 5；p13–14 Table 6–7 |
| 重複與統計 | 未說明 | 〔原文〕查無 |
| 程式碼／資料 | 公開：https://github.com/jy-yuan/KIVI | 〔原文〕p1 |
| 設計理由（原文） | group 32 沿用 FlexGen；R＝128；假量化的比較是為了「所有 token 都量化」的公平；MMLU 類只解碼一步，不適合評 KV 壓縮；residual 視窗對 GSM8K 這類難題很關鍵 | 〔原文〕p6；p8 Table 3 說明；p6 註 1；p6 |
| 設計理由〔判讀〕 | 吞吐用短 ShareGPT 是沿用 vLLM 的評法；收益完全來自「省記憶體 → 大 batch」，不是長 context 下的延遲 | 〔判讀〕 |
| 原文沒講清楚的地方 | 吞吐實驗的排程方式；R32 與 R128 的實際記憶體差；Llama-3-8B（8K）在 NIAH 跑到 27K 的設定；「2bit」是否計 scale／zero | 〔原文〕查無 |
| 與既有整理不一致 | (a) intro.txt 表 6「KIVI／KVTuner … 吞吐 2.35–3.47×」：這個數字只屬於 KIVI，條件是 ShareGPT 平均輸入 161／輸出 338、Llama-2-7B、A100、batch 加到 OOM，不是長 context，也不是 KVTuner 的結果（KVTuner 的是比 KIVI-KV8 高 16.79–21.25%，見下張卡）。(b) intro.txt 表 5 把 KIVI 列為「BF16／FP8／INT4」的代表：KIVI 只評 INT2／INT4（加 FP16 residual），沒有 FP8。(c) workloads_eval 把 Mistral-7B 標 GQA：符合模型事實，但原文 §4.1 寫 MHA，是原文自身不一致，不是 workloads_eval 錯。(d) PAPERS_BY_LEVEL 的成果數字與原文一致，但缺「短序列、batch 加到 OOM」的條件；「2-bit」在 KVQuant 的估算下實際約 3.05（32K）–3.17（12.2K）bit。〔複核判定〕(a) ✅ 成立：2.35–3.47× 出自 KIVI p1、p8（Fig. 5，Llama-2-7B、單張 A100-80GB、ShareGPT 合成 161／338、batch 加到 OOM）；KVTuner 全文的吞吐數字只有 Table 8 的 +9.22%～+21.25%（p9）。另 intro 同列「KV 量化到 2–4 bit」對 KVTuner 也不精確：它的精度對是 {2,4,8}²，配置的等效位元 3.25–5.96（KVTuner p6、p8 Table 5）。(b) ✅ 事實成立：KIVI 全文無 FP8（只有 16／4／2 bit，p8 Table 3）；但 intro 表 5 是把 KIVI／KVTuner 列為 L1「存多細」的代表工作、決策對象寫 BF16／FP8／INT4，不是逐格標成「FP8 代表」——較精確的說法是「表 5 L1 列出的精度與代表工作實際評測的精度（INT2／4／8）不一致」。(c) ✅ 成立（KIVI p6 §4.1 寫 MHA、p14 Table 9 寫 GQA；Mistral-7B-v0.1 的 HF config 為 32 個 query head／8 個 KV head，複核者 2026-10-07 查）。(d) ✅ 成立 | 〔原文〕p7–8、p6；KVQuant p8 Table 2、p9 Table 3〔二手〕 |
| 對本研究的意義〔判讀〕 | (1) 最近一段保持高精度很重要 → 使用者「後段留 GPU」時，最末端應為 BF16；(2) K 與 V 應分開設定精度與量化軸；(3) KIVI 的品質數字是 prefill 用精確 KV 量的，不涵蓋「下一輪 prefill 讀低精度 KV」 | 〔判讀〕 |

**壓縮設定**

| 項目 | 內容 | 出處 |
|:--|:--|:--|
| 預算的定義 | 名目 2／4 bit；group size 32；residual R＝128（另有 R32）全精度；各層各 head 相同；K 每 G 個 token 一組做 per-channel，V 做 per-token | 〔原文〕p5–6 |
| 是否 query-aware | 否 | 〔原文〕 |
| 作用階段 | prefill 後量化（residual 除外）；decode 時 residual 滿 R 才量化併入；prefill 時往下一層傳精確 K/V | 〔原文〕p5；p12 Algorithm 1 |
| 品質怎麼量 | LM-Eval 3 任務、LongBench 8／16 子集〔複核修正：8／15 子集，p14–15 Table 8–10〕、NIAH，與 16-bit 比 | 〔原文〕p6–9、p14–15 |
| 系統指標與相容性 | 真 kernel（CUDA／Triton）；A100；batch 加到 OOM；FlashAttention 只在 prefill（README beta）；PagedAttention 未支援（原文稱系統層方法正交） | 〔原文〕p6、p8、p9；〔文件〕 |
| 壓縮比是否計 metadata | 原文未把 scale／zero 計入「2bit」；KVQuant 以含 metadata 與 residual 的假設估 KIVI-2-gs32-r128＝3.05 bit（32K）／3.17（12.2K） | 〔原文〕p5；〔二手〕KVQuant p8–9；〔計算〕group 32、fp16 scale＋zero → 每元素多 32／32＝1 bit，2＋1＋residual（128×(16−3)／32768≈0.05）≈3.05，與 KVQuant 的估值相符〔複核：重算成立。同一公式在 12.2K 得 3＋128×13／12,200≈3.14，不是 KVQuant Table 3 的 3.17；KVQuant 沒寫出它對 KIVI 的計法，3.17 無法用此公式重現〕 |

---

### KVTuner：Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization for Efficient and Nearly Lossless LLM Inference（ICML'25；arXiv 2502.04420）

- **讀了什麼**：〔全文〕arXiv v5，https://arxiv.org/abs/2502.04420v5 ，查證 2026-10-06（正文＋附錄 A–F；附錄的逐層誤差圖只讀標題與說明）。
- **一句話**：離線用多目標最佳化搜尋逐層的 K/V 精度對（如 K8V4、K4V2）。
- **評測要證明的主張**：層對 KV 量化的敏感度是模型固有性質、與輸入無關；K 比 V 重要；數學推理上 Llama-3.1-8B 3.25-bit、Qwen2.5-7B 4.0-bit 近乎無損；吞吐比 KIVI-KV8 高 21.25%（p1）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | Llama-3.1-8B-Instruct、Mistral-7B-Instruct-v0.3、Qwen2.5-3B/7B/14B/32B-Instruct、Qwen2.5-3B-Instruct-AWQ、Qwen2.5-Math-7B-Instruct；Llama2-7B/13B-chat 只出現在 PPL 表與範例。注意力類型未標 | 〔原文〕p17 App C；p4 Table 2；p3 Table 1 |
| 硬體 | 全文（含附錄）未說明 GPU 型號與數量 | 〔原文〕查無 |
| 軟體與版本 | HF transformers v4.46.2 的 KIVI-HQQ 實作；整合進 lm-evaluation-harness；Optuna＋MOEA/D；DBSCAN（eps＝0.05、min_samples＝2）；吞吐用 KIVI 的 GPU kernel | 〔原文〕p5；p17；p18 App D.1.2〔複核修正：App D.1.2 在 p19〕；p8–9 |
| 資料／負載 | 校準：GSM8K 前 200 筆 4-shot；逐層敏感度：GSM8K zero-shot 前 20 個 prompt、multiturn-softage；評測：CEVAL、MMLU、TriviaQA、RACE、TruthfulQA、GSM8K 0／4／8／16-shot、fewshot_as_multiturn、GPQA；LongBench 20 集（哪 20 集未列）；wikitext word-PPL | 〔原文〕p17；p16、p27〔複核修正：multiturn-softage 在 p28 App F〕；p20；p8–9 Table 7；p4 Table 2 |
| 長度 | GSM8K few-shot（長度未給）；LongBench 截斷未說明；**吞吐只到輸入 128／512／1024** | 〔原文〕p9 Table 8 |
| 到達與併發 | 無；吞吐固定 BS 64／16／8 | 〔原文〕p9 Table 8 |
| 重用結構 | 無（few-shot-as-multiturn 是單請求內的多輪格式） | 〔原文〕p20〔複核修正：fewshot_as_multiturn 的說明在 p23 App E.1〕 |
| 掃描的自變數 | 等效位元數（Pareto 前緣）、量化模式（KIVI vs per-token-asym）、shots | 〔原文〕p7–8 |
| 對手 | BF16；全層統一 KV8／KV4／KV2（per-token-asym）；KIVI-8／4／2（residual 32、group 32）；KIVI-K8V4 | 〔原文〕p17；p8–9 |
| 系統指標 | 吞吐＝每秒生成 token 數，端到端、含量化／反量化開銷；最大吞吐與對應 batch | 〔原文〕p8–9 |
| 品質指標 | lm-eval accuracy；LongBench 平均（數值 0.79 量級，平均方法未說明）；word-PPL | 〔原文〕p8–9；p4 |
| 主要結果 | Qwen2.5-7B per-token〔複核修正：Table 2 說明寫的是 HF transformers 的 **KIVI-HQQ** 實作（p4），App E.1 說 KIVI 模式即 HQQ quantizer、residual 與 group 皆 32（p23），不是 per-token-asym〕：wikitext PPL KV8 9.56、K8V4 9.39、KV4 235.03；Qwen2.5-7B GSM8K 平均：BF16 0.7755、KIVI-4 0.1043、per-token C4.00 0.7559、**KIVI 模式 C3.92 0.6160**；Llama-3.1-8B KIVI 模式 C3.25 0.7925 vs BF16 0.8038；LongBench Qwen2.5-7B：BF16 0.7956、per-token KV4 0.6343、C4.0 0.7960；吞吐 C3.25 比 KV8：BS64／輸入 128 +21.25%，BS8／輸入 1024 +16.79% | 〔原文〕p4 Table 2；p8 Table 5；p9 Table 7–8 |
| 消融／敏感度／開銷 | 不做搜尋空間剪枝的搜尋；量化模式；層群分析 | 〔原文〕p9 Fig. 6；p20 Fig. 10；p9 §6.5 |
| 重複與統計 | 未說明；Qwen2.5-7B KIVI 模式 200 次搜尋「結果異常」，改跑 500 次 | 〔原文〕p20 App D.3 |
| 程式碼／資料 | 公開：https://github.com/cmd2001/KVTuner（含搜出的配置） | 〔原文〕p1 |
| 設計理由（原文） | 選層級粗粒度而非 token 級：token 級難整合 FlashAttention 與 vLLM，線上控制流不適合靜態圖；校準用數學推理，並在 prefill 也用反量化 KV 來放大誤差累積；soft constraint 4／6-bit、200 次迭代 | 〔原文〕p2；p7；p23；p17 |
| 設計理由〔判讀〕 | 用 GSM8K 當校準與主評測，是刻意挑「一個 token 翻轉就答錯」的任務放大差異；代價是「近乎無損」綁在單一任務族 | 〔判讀〕 |
| 原文沒講清楚的地方 | 硬體；LongBench 20 集清單、截斷、分數平均方式；吞吐實驗的輸出長度；等效位元是否含 residual FP16 與 scale／zero；摘要的 Qwen「4.0-bit 近乎無損」對應的是 per-token C4.00，KIVI 模式的 C3.92 在 GSM8K 掉約 0.16〔複核補充：原文內部不一致——p7 §6.1 正文說 Fig. 5b（per-token-asym、Qwen2.5-7B）可用等效 **3.92-bit** 達到 KV8 準確度，但 Table 5、Table 7 把 C3.92 列在 KIVI 模式、per-token 只列 C5.00／C4.00；p8 正文又說 KIVI 模式對「三個模型」分別是 3.92／3.17／5.96-bit，沒指明是哪三個；Table 5 的 KIVI 配置是 Llama C4.91／C3.25、Qwen-3B C3.44／C3.17、Qwen-7B C5.96／C3.92，無法一對一對上〕 | 〔原文〕p6、p7、p8–9 |
| 與既有整理不一致 | (a) intro.txt 表 6 把 KIVI 的 2.35–3.47× 放在「KIVI／KVTuner」同一列：KVTuner 自己的吞吐是比 KIVI-KV8 高 16.79–21.25%，輸入 128–1024。(b) intro.txt 表 5 L1「BF16／FP8／INT4」：KVTuner 只有 INT8／4／2，沒有 FP8 實驗；它在引言說 INT8／FP8 在多數應用無損（p2），但文中沒有 FP8 數據。(c) PAPERS_BY_LEVEL「Qwen2.5-7B 要 4.0 bit」符合摘要，但應加註：4.0-bit 只對 per-token-asym 成立，KIVI 模式 3.92-bit 掉約 16 點；且 Qwen2.5-14B／32B 對低位元較穩，「Qwen2.5 怕量化」只對 3B／7B 成立。(d) workloads_eval 長度欄只寫「LongBench 20 集」，應補上吞吐實驗只到輸入 1024。〔複核判定〕(a) ✅ 成立（KVTuner p9 Table 8）。(b) ✅ 事實成立（全文 FP8 只出現在 p2 引言一句），措辭見 KIVI 卡的複核註。(c) ⚠️ 部分修正：「KIVI 模式 3.92-bit 掉約 16 點」只在 GSM8K（Table 5：0.6160 vs BF16 0.7755）；在 LongBench（Table 7）KIVI C3.92 為 0.7903 vs BF16 0.7956，接近無損；「怕量化」的小模型原文寫的是 Qwen2.5-{3B, 7B, Math-7B} 與 3B-AWQ（p23 App E.1；p5 稱 7B 與 Math-7B 連 int4 key 都敏感）。(d) ✅ 成立 | 〔原文〕p9 Table 8；p2；p8 Table 5；p9 Table 7；p5；p23 App E.1 |
| 對本研究的意義〔判讀〕 | (1) 使用者的模型是 Qwen2.5-7B-Instruct-1M（CLAUDE.md），與 KVTuner 最敏感的 Qwen2.5-7B-Instruct 同家族：4-bit key 災難、K8V4 幾乎無損 → 使用者的「INT4」層應拆成 K/V 不同精度，至少同時測 KV4 與 K8V4；1M no-DCA 變體是否一樣敏感必須自己量。(2) KVTuner「prefill 也反量化」的設定最接近「下一輪 prefill 讀已降精度 KV」的情境。(3) 層級精度放進 vLLM 需要逐層不同 dtype 的 KV 佈局 | 〔判讀〕；1M 變體〔未查證〕 |

**壓縮設定**

| 項目 | 內容 | 出處 |
|:--|:--|:--|
| 預算的定義 | 等效位元 f_m(P)＝ΣP／(2L)；每層 (Pk, Pv)∈{2,4,8}²；約束 soft 4／6-bit；層內所有 head 同精度；KIVI 模式另有 residual 32 FP16、group 32 | 〔原文〕p6 §5.1；p17 |
| 是否 query-aware | 否（離線校準；主張敏感度與輸入無關） | 〔原文〕p5–6 |
| 作用階段 | 線上 prefill 與 decode 都量化（評測刻意含 prefill）；精度配置離線決定 | 〔原文〕p7；p23 |
| 品質怎麼量 | GSM8K、GPQA、LongBench 等，與 BF16 比 | 〔原文〕p8–9 |
| 系統指標與相容性 | KIVI GPU kernel；BS 64／16／8；Llama-3.1-8B；硬體未說明；宣稱可套用 vLLM 等框架，但未實測 | 〔原文〕p8–9；p17 |
| 壓縮比是否計 metadata | 不計：f_m 只平均 K/V 位元 | 〔原文〕p6；〔判讀〕 |

---

### KVQuant：Towards 10 Million Context Length LLM Inference with KV Cache Quantization（NeurIPS'24；arXiv 2401.18079）

- **讀了什麼**：〔全文〕arXiv v6，https://arxiv.org/abs/2401.18079v6 ，查證 2026-10-06（附錄 A–S）。
- **一句話**：pre-RoPE per-channel K、敏感度加權的非均勻資料型別、逐向量稠密＋稀疏離群值，達到 3-bit 以下。
- **評測要證明的主張**：3-bit 在 Wikitext-2 與 C4 上 PPL 退化 <0.1；LLaMA-7B 單張 A100-80GB 可到 1M context、8 GPU 可到 10M；kernel 比 fp16 matvec 快約 1.7×（p1）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | LLaMA-7B/13B/30B/65B、Llama-2-7B/13B/70B、Llama-3-8B/70B、Mistral-7B；長 context：LLaMA-2-7B-32K、Llama-2-70B-32K（LongLoRA）；基線 fp16 | 〔原文〕p7 §4.1；p8 §4.2 |
| 硬體 | kernel 延遲：A6000；topk 執行時間：A6000＋Xeon Gold 6126；Fisher 計算：8×A100-80GB；k-means：Xeon Gold 6442Y；品質實驗硬體未說明 | 〔原文〕p10 Table 6；p21 Table 15–16 |
| 軟體與版本 | 自寫 CUDA kernel（4-bit 查表＋CSC／CSR 稀疏）；端到端生成 pipeline；基線用 post-RoPE | 〔原文〕p7；p25–26 App R；p9；p22 |
| 資料／負載 | Wikitext-2、C4 PPL（teacher forcing）；passkey（50 samples）；LongBench（最長輸入 31,500，平均 12.2K token）；RULER（32K）；校準 16 筆 2K 的 Wikitext-2 train | 〔原文〕p7；p8 Table 2；p8–9 Table 3；p9 Table 4；p21 App M |
| 長度 | PPL 用模型最大 context（2K／4K／8K）；長 context 到 32K；KV 大小以 128K 估算；1M／10M 是記憶體估算（沒有在該長度跑品質）；kernel 在 l＝2K／4K／16K | 〔原文〕p22 App M；p8–9；p7 Table 1；p15 Table 8；p10 |
| 到達與併發 | 無；kernel batch 1 | 〔原文〕p9 §4.4 |
| 重用結構 | 無 | 〔原文〕 |
| 掃描的自變數 | 位元 4／3／2；離群值 0.1–1%；模型 | 〔原文〕p7；App N |
| 對手 | fp16、intX、nfX（無分組）、ATOM、FlexGen（有分組）；KIVI-2-gs32-r128（passkey、LongBench、RULER）；KIVI 的 LLaMA 程式碼不支援 GQA，所以 70B 未比 | 〔原文〕p7；p8 Table 2 說明 |
| 系統指標 | kernel 延遲 μs（1000 次平均）；KV 大小 GB（估算） | 〔原文〕p26 App R；p7 Table 1 |
| 品質指標 | PPL；passkey 成功率；LongBench、RULER 分數 | 〔原文〕p7–9 |
| 主要結果 | LLaMA-7B Wikitext-2：fp16 5.68、KVQuant-3bit-1% 5.75、int3 10.87；RULER（LLaMA-2-7B-32K）平均：fp16 56.40、KIVI-2 39.78（3.05 bit）、KVQuant-3bit-1% 53.65（3.33 bit）；passkey 上 KIVI 0.68–0.76，KVQuant 0.98–1；Key nuq4-1% 在 l＝16K 為 126.3 μs vs fp16 219.4 μs | 〔原文〕p7 Table 1；p9 Table 4；p8 Table 2；p10 Table 6 |
| 消融／敏感度／開銷 | per-channel、pre-RoPE、NUQ、稠密＋稀疏、sink-aware、離線 vs 線上校準、校準資料穩健性 | 〔原文〕App G–Q |
| 重複與統計 | kernel 1000 次平均；passkey 50 samples；其餘未說明 | 〔原文〕p26；p8 |
| 程式碼／資料 | 公開：https://github.com/SqueezeAILab/KVQuant | 〔原文〕p1 |
| 設計理由（原文） | K 的 scale 離線校準，免得每加一個 token 就要重算所有 channel 的 scale；V 線上 per-token，topk 可移到 CPU 平行；第一個 token 保持 fp16（attention sink）；基線用 post-RoPE，因為 per-token 時它較準，是更強的基線 | 〔原文〕p6 §3.6；p20 Fig. 5；p21；p6 §3.5；p24 App P〔複核修正：App P 在 p25〕 |
| 設計理由〔判讀〕 | 九篇中唯一把 scale／zero／稀疏索引算進平均位元數的；但 1M／10M 只是記憶體算術 | 〔判讀〕 |
| 原文沒講清楚的地方 | 端到端生成的吞吐或延遲數字（只有 kernel）；prefill 壓縮成本（自承只量 decode，p27）；品質實驗硬體 | 〔原文〕p27 App S |
| 與既有整理不一致 | workloads_eval 未收獨立列（只出現在 QEvict 的對手中）；使用者文件未描述 | 〔原文〕workloads_eval 第 82 行 |
| 對本研究的意義〔判讀〕 | (1) 位元數要算 metadata，App M 的假設可直接沿用；(2) KIVI 的「2-bit」在長 context 下有效約 3 bit；(3) passkey 不夠：KIVI 與 KVQuant 在 passkey 差距不大，在 RULER 差很多（KIVI 39.78 vs fp16 56.40）→ ε 要用 RULER 這類多任務長 context，不能只用 NIAH〔複核修正：前提錯誤。KVQuant Table 2（p8）中 KIVI-2 的 passkey 只有 0.68–0.76，KVQuant nuq3／nuq4-1% 為 0.98–1、fp16 為 1，差距很大。較站得住的說法是：KIVI 自己的 NIAH（KIVI p7 Fig. 4，Llama-3-8B／Mistral-7B-v0.2、到約 27K–30K token）原文稱 2-bit 仍維持檢索能力，換成 KVQuant 的 passkey（LLaMA-2-7B-32K）與 RULER 就退化明顯——同一方法在不同長 context 測法、不同模型上結論可以相反，所以 ε 不能只靠單一 NIAH 類測試〕 | 〔判讀〕；passkey 數字〔原文〕KVQuant p8 Table 2 |

**壓縮設定**

| 項目 | 內容 | 出處 |
|:--|:--|:--|
| 預算的定義 | 名目 nuq4／3／2＋0.1–1% 稀疏離群值；每層一個非均勻資料型別，再逐 channel／token 縮放；第一個 token fp16 | 〔原文〕p5–6 |
| 是否 query-aware | 否 | 〔原文〕 |
| 作用階段 | prefill 時用 fp16 K/V 算注意力，之後才壓縮；decode 逐 token 壓縮（V 線上；K 用離線 scale） | 〔原文〕p22 App M；p6 §3.6 |
| 品質怎麼量 | PPL、passkey、LongBench（31.5K）、RULER（32K），與 fp16 比，並比 KIVI | 〔原文〕p7–9 |
| 系統指標與相容性 | 自寫 kernel；A6000；batch 1；只量 matvec；未討論 PagedAttention／FlashAttention | 〔原文〕p10、p26–27 |
| 壓縮比是否計 metadata | 計入：整數量化假設低精度 offset＋16-bit scale；NF／NUQ 假設 zero 與 offset 各 16-bit；稀疏矩陣 32-bit 列索引、16-bit 欄與值〔複核修正：是 per-token 索引 32-bit、元素值與 per-element 索引 16-bit；CSR（Value 用）為 32-bit 列，CSC（Key 用）為 32-bit 欄，p22 App M、p26 App R〕 | 〔原文〕p22 App M |

---

### CacheGen：KV Cache Compression and Streaming for Fast Large Language Model Serving（SIGCOMM'24；arXiv 2310.07240）

- **讀了什麼**：〔全文〕arXiv v6，https://arxiv.org/abs/2310.07240v6 ，查證 2026-10-06（附錄 A–E；原文註明附錄未經同行審查，p17）。
- **一句話**：把 KV 編成位元流（delta＋分層量化＋算術編碼），依頻寬逐 chunk 調整壓縮級別或改送文字重算。
- **評測要證明的主張**：KV 比量化基線小 3.5–4.3×，抓取與處理 context 的總延遲降 3.2–3.7×，品質影響可忽略（p1–2）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | 微調成長 context（到 32K）的 Mistral-7B、Llama-34B、Llama-70B（Fig. 8 標成 Llama-33B）；洞察實驗用 Llama-7B／13B；附錄 LongChat-7b-16k、Llama-7B／3B。注意力類型未標 | 〔原文〕p8 §7.1；p9 Fig. 8；p4；p17 |
| 硬體 | 一台 4×NVIDIA A40 server、384GB 記憶體、2×Xeon Gold 6130；頻寬如何施加（真實網路或限速）未說明 | 〔原文〕p9；施加方式〔未查證〕 |
| 軟體與版本 | 約 2K 行 Python＋1K 行 CUDA，PyTorch 2.0、CUDA 12.0；HF transformers 上的 calculate_kv／generate_with_kv 介面；整合 LangChain；文字基線用 vLLM（xFormers） | 〔原文〕p7；p9 |
| 資料／負載 | LongChat 200（中位 9.4K）、TriviaQA 200（9.3K）、NarrativeQA 200（14K）、WikiText 62（5.9K），共 662 個 context、1.4K–16K；設計編碼器用的資料是評測資料的子集 | 〔原文〕p2；p8 Table 2 |
| 長度 | 1.4K–16K；長度掃描 0.1K–15K；頻寬掃描固定 16K | 〔原文〕p8；p10 |
| 到達與併發 | 掃描併發請求數（到達過程未說明）；T 秒內到達的請求一起批次串流 | 〔原文〕p10 Fig. 12；p7 |
| 重用結構 | context 跨請求重用：預先編碼後存在儲存伺服器 | 〔原文〕p2；p7–8 |
| 掃描的自變數 | 頻寬 0.4–15 與 15–400 Gbps；併發請求數；context 長度；SLO 0.5／1 s；編碼級別 | 〔原文〕p10 Fig. 11–13 |
| 對手 | 預設量化（每層同 3／4／8-bit）；文字 context（vLLM prefill）；H2O（理想化：離線就用 prompt 的 query）；LLMLingua；附錄：較小模型、Scissorhands*（理想化）、Gisting | 〔原文〕p9；p17 App B |
| 系統指標 | KV 大小（MB）；TTFT＝query 到達到第一個 token，含 KV 載入與新問題的 prefill；SLO 違反率；無 percentile | 〔原文〕p8；p10 |
| 品質指標 | LongChat accuracy（答案是否包含正確主題）、TriviaQA／NarrativeQA F1、WikiText PPL；MTurk 使用者研究（270 個評分） | 〔原文〕p8；p11 |
| 主要結果 | Mistral-7B LongChat：8-bit 622 MB／acc 1.00，CacheGen 176 MB／0.98；3 Gbps 下 TTFT 比文字快 3.1–4.7×、比量化快 3.2–3.7×，比 8-bit 快 1.67–1.81×；SLO 1 s 時違反率 81% → 8%；離線編碼延遲約 200 ms | 〔原文〕p2 Table 1；p9；p10；p11 |
| 消融／敏感度／開銷 | 編碼器三元件逐步加入；頻寬、併發、長度；附錄熱圖；儲存成本 | 〔原文〕p11 Fig. 15；p10；p19 App D–E |
| 重複與統計 | Fig. 13 平均 20 條頻寬 trace；其餘未說明 | 〔原文〕p10–11 |
| 程式碼／資料 | 公開：https://github.com/UChi-JCL/CacheGen | 〔原文〕p1 |
| 設計理由（原文） | chunk 取 1.5K token，在「能及時反應頻寬變化」與「改送文字時 GPU 批次效率」之間取捨；每 10 個 token 共用一個 anchor 以便平行；anchor 用 8-bit；早層較敏感所以早層量化 bin 較小；只在 A40 上測，承認高階 GPU＋低頻寬時可能不如送文字 | 〔原文〕p7；p6；p5–6；p12 |
| 設計理由〔判讀〕 | 以「頻寬」為主自變數是網路論文的評測傳統；品質只在 ≤16K 的單輪問答驗證 | 〔判讀〕 |
| 原文沒講清楚的地方 | 頻寬如何施加；併發實驗的到達分布；每模型的離線機率表是否算進 KV 大小；各編碼級別的實際位元數；Llama-34B 與 Fig. 8 的 Llama-33B 是否同一模型 | 〔原文〕p8–10 |
| 與既有整理不一致 | workloads_eval 只在 KVServe 列以對手身分出現，無數值主張可對照；使用者文件未直接描述 | 〔原文〕workloads_eval 第 92 行 |
| 對本研究的意義〔判讀〕 | (1) 「每個 chunk 選壓縮級別或改送文字重算」就是使用者動作空間的傳輸版，但它在讀取時、依頻寬決定；(2) 它的品質評法（解壓後的 KV 被新問題的 prefill 使用）與使用者情境一致，可沿用；(3) query-agnostic 是它能離線壓縮的前提 | 〔判讀〕 |

**壓縮設定**

| 項目 | 內容 | 出處 |
|:--|:--|:--|
| 預算的定義 | 編碼級別（量化 bin 大小；預設三個層群 0.5／1／1.5）＋算術編碼；結果以位元流大小（MB）呈現；每 chunk 可用不同級別 | 〔原文〕p19 App C.2；p7 |
| 是否 query-aware | 否（明說不需要知道 query） | 〔原文〕p12 §8 |
| 作用階段 | 離線編碼（prefill 後）；傳輸時串流解碼，並與傳輸 pipeline | 〔原文〕p7 |
| 品質怎麼量 | 4 個資料集的標準指標，與 8-bit 量化等比較 | 〔原文〕p8–9 |
| 系統指標與相容性 | GPU 解碼 kernel；4×A40；TTFT；併發請求；CacheGen 本身在 HF 上，文字基線在 vLLM | 〔原文〕p7；p9 |
| 壓縮比是否計 metadata | 以實際位元流大小計（含 anchor）；每模型機率表是否計入未說明〔複核補充：§5.2 說每個 LLM 離線 profile 一套 channel×layer 分布（p6），§6 卻說分布由對應 context 的符號頻率統計（p8），兩處不一致；後者代表分布表是逐 context 的 metadata〕 | 〔判讀〕；機率表〔未查證〕；分布來源〔原文〕p6、p8 |

---

### CacheBlend：Fast Large Language Model Serving for RAG with Cached Knowledge Fusion（EuroSys'25；arXiv 2405.16444）

- **讀了什麼**：〔全文〕arXiv v3，https://arxiv.org/abs/2405.16444v3 ，查證 2026-10-06。
- **一句話**：把多段預先算好的 KV 拼接，每層只重算一小部分高偏差 token 以補 cross-attention，並與載入 pipeline。
- **評測要證明的主張**：TTFT 降 2.2–3.3×、吞吐升 2.8–5×、品質不損；比 full KV reuse 的 F1 高 0.1–0.2、Rouge-L 高 0.03–0.25（p1–2）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | Mistral-7B、Yi-34B、Llama-70B；Yi-34B 與 Llama-70B 用 8-bit 模型量化。注意力類型與 Llama-70B 的版本未標 | 〔原文〕p10 §7.1 |
| 硬體 | Runpod，128 GB RAM、2×A40、1TB NVMe（實測 4.8 GB/s）；Mistral-7B 與 Yi-34B 用 1 張 GPU，Llama-70B 用 2 張 | 〔原文〕p10 |
| 軟體與版本 | vLLM 上約 3K 行 Python，PyTorch 2.0；KV 在磁碟用 torch.load／torch.save；儲存滿了用 LRU 逐出；只用單一層級的儲存 | 〔原文〕p9；p8 |
| 資料／負載 | 2WikiMQA 200、Musique 150、SAMSum 200、MultiNews 60；context 用 Langchain 切 512-token chunk（SAMSum 用原本 200–400 token 的 chunk）；extended 版：每集取 1500 個 query，用 GPT-4 各生 3 個相似 query，每集共 6000；依 L2 距離取 top-6 chunk、隨機順序；略過前 1K 個 query（暖機）；2WikiMQA／Musique 的 prompt 加上「5 字內作答」 | 〔原文〕p10–11；p10 註 7 |
| 長度 | 6 個 512-token chunk ≈ 3K token＋query；chunk 數 3–12、chunk 長 300–900；top-6 是 Llama-70B 輸入上限放得下的最大數 | 〔原文〕p11–12；p11 Fig. 15；p11 註 8 |
| 到達與併發 | 掃描平均請求率（Fig. 14 橫軸），分布未說明 | 〔原文〕p11 Fig. 14 |
| 重用結構 | RAG chunk 的非前綴重用 | 〔原文〕p1–2 |
| 掃描的自變數 | 請求率；重算比例 5–18%；chunk 數／長度；batch size 2–10；儲存裝置（CPU RAM、4 Gbps 慢磁碟） | 〔原文〕p11–12 Fig. 14–17 |
| 對手 | Full KV recompute；Prefix caching（SGLang 技術，KV 放 RAM＋SSD，**假設載入零延遲**）；Full KV reuse（PromptCache）；MapReduce、MapRerank（LangChain） | 〔原文〕p11 |
| 系統指標 | 平均 TTFT；吞吐對請求率；無 percentile | 〔原文〕p10–12 |
| 品質指標 | F1（2WikiMQA、Musique）、Rouge-L（SAMSum、MultiNews） | 〔原文〕p11 |
| 主要結果 | 對 full recompute 與 prefix caching：TTFT 降 2.2–3.3×，F1／Rouge-L 掉 ≤0.02；同 TTFT 下吞吐比 full recompute 高至 5×、比 prefix caching 高 3.3×；重算 5–18% 時品質掉 ≤0.002（Yi-34B）；Llama-7B、4K context：15% 重算每層 3 ms，NVMe 載入一層 16 ms | 〔原文〕p12；p10；p12 Fig. 16；p8 |
| 消融／敏感度／開銷 | 重算比例、chunk 數／長度、batch、儲存裝置 | 〔原文〕§7.3 p12 |
| 重複與統計 | 未說明 | 〔原文〕查無 |
| 程式碼／資料 | 公開：https://github.com/LMCache/LMCache | 〔原文〕p1 |
| 設計理由（原文） | 預設重算 15%（由 Fig. 16 的經驗）作為品質下限；用逐層漸進篩選挑 HKVD token，因為相鄰層的 HKVD 高度相關；prefix caching 給零載入延遲是刻意偏袒對手 | 〔原文〕p8；p6–7 Fig. 8；p11 |
| 設計理由〔判讀〕 | 用 GPT-4 擴增相似 query，是在沒有真實 RAG trace 下製造 chunk 重用；重用率因此由構造決定 | 〔判讀〕 |
| 原文沒講清楚的地方 | 到達過程；GPT-4 生成 query 的 prompt；實際重用率；Llama-70B 的版本；「2.2–3.3×」的比較對象在引言寫 prefix caching（p2），在摘要與 §7 寫 full KV recompute（p1、p10、p12）〔複核修正：摘要（p1）、Fig. 12 說明與 §7 takeaways（p10）寫 full KV recompute；引言（p2）寫 prefix caching；§7.2 正文（p12）寫「與 full KV recompute **和** prefix caching 相比」。另 §4（p5）把兩者並列為「full KV recompute (i.e., full prefill or prefix caching)」。Llama-70B 的引用 [2] 指向 arXiv 2302.13971（LLaMA 第一代，該代沒有 70B），版本無法由原文確定〕 | 〔原文〕p1、p2、p5、p10、p12、p13 參考文獻 [2] |
| 與既有整理不一致 | (a) intro.txt 表 6「TTFT 2.2–3.3×」數字正確，但基準在原文中寫法不一（引言 vs §7），而且 prefix caching 基準被設成載入零延遲，引用時要標明基準。(b) PAPERS_BY_LEVEL「品質不掉」過度簡化：原文是 F1／Rouge-L 掉 ≤0.02（p12），takeaways 寫 0.01–0.03（p10）；「EuroSys'25 Best Paper」原文未載。(c) workloads_eval「皆 GQA」原文未標注意力類型，屬外部知識，應標〔未查證〕；「2×A40」應補「7B／34B 只用 1 張」。(d) sota.txt 表 S6「CacheBlend 程式碼：有」與原文一致。〔複核判定〕(a) ✅ 成立（數字對，基準在原文中不一；prefix caching 基準「假設 RAM／SSD 到 GPU 無載入延遲」見 p11）。(b-1) ⚠️ 部分成立：≤0.02（p12）與 0.01–0.03（p10）都對；但 PAPERS_BY_LEVEL 的「品質不掉」與原文摘要措辭 "without compromising generation quality"（p1）一致，且同一條目的短板已寫「選擇性重算是有損的」，不算錯，只是少了數字。(b-2) ❌ 指控不成立：「Best Paper」雖不在論文內，但 EuroSys 2025 官網 Awards 頁（https://2025.eurosys.org/awards.html ，2026-10-07 查）列 CacheBlend 為 Best Paper（Spring submissions），PAPERS_BY_LEVEL 的寫法正確。(c) ✅ 成立（原文確實未標；另：Mistral-7B 與 Yi-34B 的 HF config 皆為 8 個 KV head〔文件〕huggingface.co/mistralai/Mistral-7B-v0.1、01-ai/Yi-34B config.json，2026-10-07；Llama-70B 版本不明，見上一格，所以「皆 GQA」三者中只有兩者可由外部資料確認）。(d) ✅ | 〔原文〕p1、p2、p10–12；Best Paper〔文件〕EuroSys'25 官網 |
| 對本研究的意義〔判讀〕 | 它是「非前綴重用要部分重算」的代表，重算比例本身就是一個品質－成本旋鈕；它示範了「每層載入時間 vs 每層重算時間」的 pipeline 評測方法（p8），與使用者的 κ 思路相同；但 KV 本身不壓縮（只有模型權重 8-bit） | 〔判讀〕 |

**壓縮設定**

| 項目 | 內容 | 出處 |
|:--|:--|:--|
| 預算的定義 | 每層重算比例 r%（預設 15%，掃 5–18%）；由載入延遲估計器與品質下限 r* 取最大值；KV 本身不壓縮 | 〔原文〕p8；p12 |
| 是否 query-aware | 在請求時決定：依這次請求的 chunk 組合與順序計算 KV 偏差 | 〔原文〕p6–7；〔判讀〕不依問題內容挑 token |
| 作用階段 | 請求 prefill 時逐層選擇性重算，與下一層 KV 的載入 pipeline | 〔原文〕p8–9 |
| 品質怎麼量 | 4 個資料集的 F1／Rouge-L，與 full recompute 比 | 〔原文〕p11–12 |
| 系統指標與相容性 | 整合進 vLLM；2×A40；NVMe 4.8 GB/s；TTFT、吞吐對請求率 | 〔原文〕p9–11 |
| 壓縮比是否計 metadata | 不適用（KV 不壓縮）；hash table 每百萬 chunk 16MB | 〔原文〕p9 |

---

## 本組對 PoC 設計的建議〔判讀〕

以下全部是抽取者的判讀，重點是**使用者的精度分層 BF16／FP8／INT4 的品質 ε 要怎麼量**。

1. **ε 的定義要固定、要成對比較。** 九篇的「近乎無損」門檻各不相同（P3）。建議：同一批 prompt、greedy decoding，與 BF16 full KV 逐題成對比較；報兩個量——任務分數差（含 bootstrap 信賴區間），以及與 BF16 輸出一致的比例（答案完全相同／第一個分歧 token 的位置）。九篇幾乎都沒有重複或信賴區間；greedy 是確定性的，所以統計要做在「樣本」上，不是重跑。

2. **一定要量「讀取方的 prefill 會看到低精度 KV」的情境。** 使用者的情境是：第 t 輪的 KV 以 FP8／INT4 存著，第 t+1 輪新 token 的 prefill 要讀反量化後的 KV，再 decode。KIVI 與 KVQuant 的品質數字是 prefill 用精確 KV 量的（KIVI p5、KVQuant p22），會低估這個情境；KVTuner 的「prefill 也反量化」（p7、p23）與 CacheGen／CacheBlend 的「載入後 prefill」比較接近。PoC 應該至少有兩輪：先 prefill context 並把 KV 降精度，再送問題。

3. **使用者要的是「依位置」的精度，沒有人評過。** KIVI 只有「最近 R 個 token 全精度」、KVQuant 只有「第一個 token 全精度」、KVTuner 與 CacheGen 是逐層。所以 ε 要做成矩陣 ε(位置帶, 精度, context 長度)：其他 chunk 保持 BF16，只把某一段降成 FP8 或 INT4，掃前／中／後段與比例。另外，開頭幾個 sink token 一律保持全精度（StreamingLLM App F p18；KVQuant §3.5 p6）。

4. **K 與 V 分開，量化軸照 KIVI／KVQuant。** KVTuner 顯示 Qwen2.5-7B-Instruct 的 4-bit key 是災難（per-token KV4 PPL 235.03），K8V4 卻幾乎無損（p4 Table 2）〔複核修正：Table 2 是 KIVI-HQQ 實作（p4 表說明），不是 per-token；KIVI 模式在 GSM8K／一般任務上同樣是 KV4 平均 0.2319、K8V4 0.6619 vs BF16 0.6653（p27 Table 13）〕。使用者的模型是 Qwen2.5-7B-Instruct-1M，所以「INT4」層至少要同時測 KV4 與 K8V4；K 用 per-channel、V 用 per-token。1M no-DCA 變體是否一樣敏感，只能自己量。

5. **用部署時真的會走的路徑，不用假量化。** KIVI 的同一個 2-bit 配置，假量化（沒有 residual）GSM8K 5.76，真實 KIVI 12.74（p8 Table 3）——實作細節會改變 ε。FP8：九篇都沒有 FP8 KV 的實驗（KVTuner 只在引言有一句主張，p2），所以 FP8 的 ε 必須用 vLLM 的 fp8 KV 路徑自己量，並記下 scale 怎麼算。

6. **benchmark 不能只有 NIAH。** KVQuant 顯示 KIVI 在 passkey 上差距不大，在 RULER 上掉了 16.6 分（p8–9）〔複核修正：passkey 上 KIVI-2 也明顯掉（0.68–0.76 vs fp16 1，KVQuant p8 Table 2）；「差距不大」的是 KIVI 自己的 NIAH（KIVI p7）。RULER 56.40−39.78＝16.62 重算成立〕。建議用 RULER（多子任務、可到 128K）加 LongBench 中較長的子集，在 16K／32K／64K／128K 掃，必要時往 512K 延伸。九篇超過 32K 的品質證據只有 NIAH 類（SnapKV 380K）或沒有 full 基線的表（PyramidKV 128K）。

7. **壓縮比按實際位元組算。** 照 KVQuant App M（p22）把 scale、zero、residual、索引都算進去。〔計算〕group 32、fp16 scale＋zero 時每元素多 1 bit：INT4-g32 實際約 5 bit、INT2-g32 約 3 bit；FP8 若是 per-tensor scale 則可忽略。使用者的成本常數（每 chunk 的位元組數）要用這個口徑。

8. **寫入時不能依賴 query-aware 的選擇。** SnapKV §4.2.1／Fig. 4（p5–6）顯示換一個指令，重要位置就變了。如果 PoC 要放 SnapKV／PyramidKV 當對照，必須用「同一 context、換問題再問」的測法（SCBench 式），否則會高估它們在多輪下的品質。

9. **把「讀 INT4／FP8 的成本」當成要量的成本，而不是零。** 九篇的反量化成本只在 decode matvec 量過（KIVI、KVQuant），沒有人量「prefill 讀低精度 KV」的成本〔複核修正：KVQuant 只量 matvec kernel（p10 Table 6）；KIVI 與 KVTuner 量的是含量化／反量化的端到端吞吐（KIVI p8 Fig. 5；KVTuner p8–9），沒有單獨拆出反量化；CacheGen 在 TTFT 分解中量了位元流解碼（decompression）開銷（p11 Fig. 14a），那是「載入後給 prefill 用」之前的解碼成本，但不是 prefill 直接讀低精度 KV 的成本〕；PyramidKV App R 提醒，在 paged KV 裡混用不同大小或精度會造成碎片。使用者 κ 模型中 GPU-FP8／GPU-INT4 的讀取成本需要實測，且要在 vLLM 的 paged 佈局下量。

10. **「全部降精度＋LRU」是必要的強基線。** 使用者的 intro 已經列為簡單強基線（intro.txt 表 18），這組論文支持這個判斷：若模型對 K8V4 或 FP8 不敏感（Llama-3.1-8B、Qwen2.5-14B／32B），全部降精度可能就吃掉大部分空間壓力。

---

## 未查證清單

1. SnapKV 的 NeurIPS'24 camera-ready 與 arXiv v2 之間是否有實驗差異。
2. PyramidKV 的正式發表場所（arXiv v4 仍標 under review）。
3. ~~CacheBlend「EuroSys'25 Best Paper」（原文未載，需查 EuroSys 官網）。~~〔複核已查證：EuroSys 2025 官網 Awards 頁列為 Best Paper（Spring submissions），https://2025.eurosys.org/awards.html ，2026-10-07〕
4. KVTuner 的硬體（全文未載，可能需讀程式碼或問作者）。
5. CacheGen 頻寬如何施加（真實網路或 tc 限速）、機率表是否計入 KV 大小。
6. 各篇模型的注意力類型：原文多半未標，只有 KIVI 附錄與 StreamingLLM 的位置編碼有寫；workloads_eval 填的 GQA／MHA 屬外部知識。
7. LongBench 平均長度的單位（PyramidKV 稱 token，官方英文集可能是 word 數）——交 E09。
8. H2O 的 AlpacaEval／MT-bench 結果、OPT-175B 結果（正文提到但 v3 未見）。
9. H2O 附錄 D（理論證明）未逐行讀。
10. 所有程式碼層級的細節（每 head 預算是否一致、FlashAttention 路徑如何取得注意力分數等）：只讀了 SnapKV、PyramidKV、KIVI 的 README，沒有讀原始碼。
11. KVTuner 的 LongBench 20 集清單、截斷長度、分數尺度。
12. Qwen2.5-7B-Instruct-1M（含 no-DCA 變體）是否與 Qwen2.5-7B-Instruct 一樣對 4-bit key 敏感。
13. StreamingLLM 效率實驗的 batch 與 prompt 長度。
14. PyramidKV Table 2 記憶體的量測方式（與 KV 算術值差約 6.7×）。
15. CacheGen 的 Llama-34B 與 Fig. 8 的 Llama-33B 是否同一模型。

---

## 複核紀錄

- **複核者**：V07（獨立子 agent，未參與抽取、未讀抽取者筆記或推理）
- **日期**：2026-10-07
- **原文來源**：使用抽取者下載的 9 份 PDF（`scratchpad/E07/*.pdf`），複核者自行重跑 `pdftotext -layout` 到 `scratchpad/V07/`，逐份確認 arXiv 編號與版本（2306.14048v3、2309.17453v4、2310.07240v6、2401.18079v6、2402.02750v2、2404.14469v2、2405.16444v3、2406.02069v4、2502.04420v5）與卡片「來源清單」一致；頁碼皆為 PDF 頁。圖中數值只對 KIVI p8（Fig. 5）、KVTuner p4（Table 2）兩頁另外轉成圖檔目視確認。README 重新從 GitHub 下載比對（SnapKV、KIVI、KVCache-Factory）；8 個 repo 中 7 個的最新 commit SHA 與卡片相符（PyramidKV repo 已更名，API 未回傳，未確認）。外部查證：EuroSys 2025 官網 Awards 頁、HF config（Llama-3-8B-Instruct、Mistral-7B-v0.1、Yi-34B）。
- **標記方式**：❌＝與原文不符（含頁碼錯、數量錯、標籤錯、被推翻的指控）；⚠️＝措辭過度、需要限定，或原文找不到；〔複核補充〕＝抽取者漏掉的重要事實，不計入錯誤。下表 ✅ 欄計「格」，❌／⚠️／補充欄計「條」；同一格可能有兩條，所以三者相加不一定等於檢查格數。

### 逐卡統計（每卡 29 格＝3 條開頭＋20 格評測表＋6 格壓縮設定）

| 卡 | 檢查格數 | ✅ | ❌ | ⚠️ | 補充 |
|:--|:--|:--|:--|:--|:--|
| StreamingLLM | 29 | 29 | 0 | 0 | 0 |
| H2O | 29 | 27 | 0 | 2 | 1 |
| SnapKV | 29 | 29 | 0 | 0 | 0 |
| PyramidKV | 29 | 28 | 1 | 1（與 ❌ 同一格） | 0 |
| KIVI | 29 | 27 | 2 | 0 | 2 |
| KVTuner | 29 | 24 | 4 | 1 | 1 |
| KVQuant | 29 | 26 | 2 | 1 | 0 |
| CacheGen | 29 | 29 | 0 | 0 | 1 |
| CacheBlend | 29 | 27 | 1 | 2（其一與 ❌ 同格） | 1 |
| 共同模式 P1–P6（P1 表 9 列＋5 段） | 14 | 11 | 1 | 2 | 1 |
| PoC 建議 1–10（只核其中的事實陳述） | 10 | 7 | 2 | 1 | 0 |
| **合計** | **285** | **264** | **13** | **10** | **7** |

### 逐條修改（原內容 → 新內容＋出處）

**❌（13）**
1. PyramidKV「原文沒講清楚」：Table 14 中 H2O 與 StreamingLLM「11 欄」相同 → **14 欄**相同（只有 GovReport、TriviaQA 不同）。PyramidKV p25 Table 14。
2. KIVI「資料／負載」：附錄「16 子集」→ **15 子集**（無 PassageCount）。KIVI p14–15 Table 8–10。
3. KIVI 壓縮設定「品質怎麼量」：「LongBench 8／16」→「8／15」。同上。
4. KVTuner「主要結果」：Table 2 的 Qwen2.5-7B PPL 標為「per-token」→ 是 **KIVI-HQQ** 實作。KVTuner p4 Table 2 說明、p23 App E.1。
5. KVTuner「軟體與版本」：DBSCAN 出處 p18 → **p19**（App D.1.2）。
6. KVTuner「資料／負載」：multiturn-softage 出處 p27 → **p28**（App F）。
7. KVTuner「重用結構」：fewshot_as_multiturn 出處 p20 → **p23**（App E.1）。
8. KVQuant「設計理由（原文）」：App P 出處 p24 → **p25**。
9. KVQuant「對本研究的意義」(3)：「KIVI 與 KVQuant 在 passkey 差距不大」→ 前提錯誤，KIVI-2 passkey 0.68–0.76，KVQuant 0.98–1。KVQuant p8 Table 2。改寫為「KIVI 自己的 NIAH 原文稱維持，換成 KVQuant 的 passkey 與 RULER 就明顯退化」。
10. CacheBlend「與既有整理不一致」(b)：「EuroSys'25 Best Paper 原文未載」暗示 PAPERS_BY_LEVEL 有誤 → **指控不成立**，EuroSys 2025 官網 Awards 頁列 CacheBlend 為 Best Paper（Spring submissions），https://2025.eurosys.org/awards.html 。未查證清單第 3 條同步劃掉。
11. 共同模式 P6：「PyramidKV 是唯一描述 vLLM 實作的」與同段「CacheBlend 實作在 vLLM 上」矛盾 → 限定為「逐出＋量化七篇中唯一」。CacheBlend p9。
12. 建議 4：「per-token KV4 PPL 235.03」→ KIVI-HQQ；並補 Table 13（p27）KIVI 模式 KV4 0.2319、K8V4 0.6619 vs BF16 0.6653。
13. 建議 6：「KIVI 在 passkey 上差距不大」→ 同第 9 條更正。

**⚠️（10）**
1. H2O「模型」：吞吐模型漏了 OPT-13B（A100 Table 5，5000+5000、batch 4）。H2O p9。
2. H2O 壓縮設定「作用階段」：「prompt ≥2K」易誤讀為 2,000 token → 註明是 2×K。H2O p20 App A。
3. PyramidKV「原文沒講清楚」：「摘要『12%（size 2048）』」→ 摘要與 p7 只寫 12%，沒有對到 2048；補〔計算〕：16 集平均長度≈7,426 時 64≈0.86%、2048≈27.6%；8192 為分母時 1024＝12.5%。PyramidKV p1、p7 Table 1、p8、p9 Table 2。
4. KVTuner「與既有整理不一致」(c)：「KIVI 模式 3.92-bit 掉約 16 點」限定為 GSM8K（Table 5）；LongBench 上 C3.92 為 0.7903 vs 0.7956（Table 7）；敏感小模型應含 Math-7B。KVTuner p8–9、p5、p23。
5. KVQuant 壓縮設定 metadata：「稀疏 32-bit 列＋16-bit 欄與值」只對 CSR 成立 → per-token 索引 32-bit（CSC 為欄）。KVQuant p22 App M、p26 App R。
6. CacheBlend「原文沒講清楚」：「§7 寫 full KV recompute」→ §7.2 正文（p12）寫「full KV recompute 與 prefix caching」，Fig. 12 與 takeaways（p10）才只寫 full KV recompute；並補 §4（p5）把 prefix caching 視為 full recompute 的一種、Llama-70B 引用指向 LLaMA-1（p13 ref [2]）。
7. CacheBlend (b)「品質不掉過度簡化」→ 部分成立：措辭與原文摘要一致（p1），PAPERS_BY_LEVEL 短板已寫有損。
8. 共同模式 P1 KVQuant 列：同 ⚠️5。
9. 共同模式 P3：「超過 32K 的只有 NIAH／passkey、PPL、無基線表」→ 補 SnapKV Command-R RAG citation 20K–40K（p10 Table 3）、StreamingLLM StreamEval 約 120K（p8）。
10. 建議 9：「反量化成本只在 decode matvec 量過（KIVI、KVQuant）」→ KIVI、KVTuner 量的是端到端吞吐；CacheGen 量了位元流解碼開銷（p11 Fig. 14a）。

**〔複核補充〕（7）**：H2O 的 4-bit 量化是結合實驗、不是對手（p24）；KIVI 2.35×／3.47× 分別對應 R128／R32（Fig. 5 讀圖，判讀）；KIVI 3.17 bit 無法用 3.05 的公式重現（12.2K 得 3.14）；KVTuner 3.92-bit 屬哪個模式，原文 p7 正文與 Table 5／7 不一致；CacheGen 機率分布來源 §5.2（p6，逐模型）與 §6（p8，逐 context）不一致（P1 與 CacheGen 卡各一處）；CacheBlend 的 Llama-70B 引用指向 LLaMA-1。

### 對使用者文件的指控：逐條判定

| 指控 | 判定 | 依據 |
|:--|:--|:--|
| intro 表 6 把 2.35–3.47× 放在 KIVI／KVTuner 同一列，數字只屬 KIVI | ✅ 成立 | KIVI p1、p8；KVTuner 只有 Table 8 的 +9.22%～+21.25%（p9）。另補：同列「2–4 bit」對 KVTuner 不精確（{2,4,8}²） |
| intro 表 5 把 KIVI／KVTuner 列為 BF16／FP8／INT4 的代表，但兩篇都沒測 FP8 | ✅ 事實成立（措辭需修） | KIVI 全文無 FP8；KVTuner 只有 p2 一句。表 5 是把兩篇列為 L1 代表，不是逐格指為 FP8 代表 |
| CacheBlend 2.2–3.3× 的比較對象原文不一 | ✅ 成立（細節修正） | p1、p10：full recompute；p2：prefix caching；p12：兩者並列 |
| CacheBlend 的 prefix caching 基準假設載入零延遲 | ✅ 成立 | p11 |
| CacheBlend 品質掉 ≤0.02，PAPERS_BY_LEVEL「品質不掉」過度簡化 | ⚠️ 部分成立 | 數字正確（p12；p10 為 0.01–0.03），但「品質不掉」與原文摘要措辭一致（p1） |
| CacheBlend「EuroSys'25 Best Paper」查不到 | ❌ 不成立 | EuroSys 2025 官網 Awards 頁 |
| workloads_eval 把 CacheBlend 標成「皆 GQA」 | ✅ 成立（原文未標） | p10 未標注意力類型；外部 config 可確認 Mistral-7B、Yi-34B 為 8 KV heads，Llama-70B 版本不明 |
| workloads_eval CacheBlend「2×A40」應補 7B／34B 只用 1 張 | ✅ 成立 | p10 |
| workloads_eval KIVI 把 Mistral-7B 標 GQA（原文 §4.1 寫 MHA，是原文自身不一致） | ✅ 成立 | KIVI p6、p14 Table 9 |
| PAPERS_BY_LEVEL KIVI 成果缺條件；「2-bit」實際約 3 bit | ✅ 成立 | KIVI p7–8；KVQuant p8–9 |
| PAPERS_BY_LEVEL KVTuner「Qwen2.5-7B 要 4.0 bit」應加註模式 | ⚠️ 部分修正 | 見 ⚠️4 |
| workloads_eval KVTuner 長度欄應補吞吐只到輸入 1024 | ✅ 成立 | KVTuner p9 Table 8 |

### 時間所限未檢查的部分（如實記錄）

- 各篇圖表中只畫在圖上的數值，除 KIVI Fig. 5、KVTuner Table 2 外沒有轉成圖檔逐點讀（例如 CacheBlend Fig. 12–17、CacheGen Fig. 8–13 的座標值、PyramidKV Fig. 3）；這些格以正文文字為準。
- H2O 附錄 D（理論證明，p32–49）、PyramidKV 附錄 A–D 與 G、KVQuant 附錄 G–K、SnapKV 附錄 B、KVTuner 附錄 F 的逐層熱圖，只讀了與卡片相關的段落。
- 「本組對 PoC 設計的建議」與各〔判讀〕格只核對其中的事實陳述，沒有評論判讀本身是否合理。
- 「未查證清單」除第 3 條（Best Paper）與第 6 條的 CacheBlend 部分外，沒有再追查（如 KVTuner 硬體、LongBench 20 集清單、SnapKV camera-ready 差異、CacheGen 頻寬施加方式）。
- 沒有讀任何原始碼（與抽取者相同，只讀 README）。
