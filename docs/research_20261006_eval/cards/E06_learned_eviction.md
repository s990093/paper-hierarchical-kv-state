# E06 學習式逐出與預測：評測卡

- **範圍**：學習式快取／逐出，分兩組。
  - 跨請求的學習式快取（L2）：LARU（LCR）、LPC、SAECache（semantic-aware eviction）。
  - 請求內的學習式 token 逐出（L1 附近）：KVP、ForesightKV、LookaheadKV、TRIM-KV。
- **目的**：整理這一派「怎麼訓練、label 從哪來、怎麼切資料、怎麼量開銷、跟上界怎麼比」，供輕量預測器（GBDT 類）的 PoC 評測設計使用。
- **抽取者**：E06（子 agent），2026-10-06。狀態：抽取完成；V06 已複核（2026-10-07，見檔尾）。
- **頁碼規則**：一律是 PDF 頁（pdftotext 的換頁符號），本組七篇的 PDF 頁與印刷頁碼相同。
- **出處代號**：表格「出處」欄用下列代號＋頁碼／章節／表圖。

## 來源清單

| 代號 | 論文 | 讀的版本 | URL | 讀的範圍 |
|:--|:--|:--|:--|:--|
| [LARU] | Chen et al., Toward Robust and Efficient ML-Based GPU Caching for Modern Inference | arXiv v2（2026-04-24）；另比對 v1（2025-09-25）摘要，標題數字相同 | https://arxiv.org/pdf/2509.20979v2 | 全文＋附錄 A–F（19 頁） |
| [LPC] | Yang, Li, Li, Lloyd, Learned Prefix Caching for Efficient LLM Inference（NeurIPS 2025） | NeurIPS 2025 proceedings 版（含附錄 A–B 與 checklist，23 頁）。OpenReview PDF 需通過驗證頁，未取得；未找到 arXiv 版 | https://proceedings.neurips.cc/paper_files/paper/2025/file/414f642a1ea9350006669774cba9bcd4-Paper-Conference.pdf | 全文＋附錄 |
| [LPC-code] | 同上的官方程式碼 | commit 3e750fa（2025-10-10） | https://github.com/yangdsh/LPC | README、`vllm/vllm/core/learn_conversation.py`、`vllm_cache_bench/get_predictor_accuracy.py`、`vllm_cache_bench/run_nips.py` |
| [SAE] | Fang et al., Not All Tokens Are Worth Caching: Learning Semantic-Aware Eviction for LLM Prefix Caches | arXiv v1（2026-05-12） | https://arxiv.org/pdf/2605.18825v1 | 全文＋附錄 A–E＋checklist（31 頁） |
| [KVP] | Moschella, Manduchi, Sener, Learning to Evict from Key-Value Cache（ICML 2026） | arXiv v2（2026-06-26；PDF 內標 2026-06-29） | https://arxiv.org/pdf/2602.10238v2 | 全文＋附錄 A.1–A.7（30 頁） |
| [FKV] | Dong et al., ForesightKV（ICML 2026，PMLR 306） | arXiv v2（2026-06-01） | https://arxiv.org/pdf/2602.03203v2 | 全文＋附錄 A–E（20 頁） |
| [LKV] | Ahn et al., LookaheadKV（ICLR 2026） | arXiv v1（2026-03-11，ICLR 版面） | https://arxiv.org/pdf/2603.10899v1 | 全文＋附錄 A–H（25 頁） |
| [TRIM] | Bui et al., Cache What Lasts: Token Retention for Memory-Bounded KV Cache in LLMs（ICLR 2026） | arXiv v2（2026-03-01，ICLR 版面） | https://arxiv.org/pdf/2512.03324v2 | 全文＋附錄 A–C（27 頁） |
| [intro] | 使用者寄給老師的研究介紹文字版 | — | `<SCRATCH>/intro.txt` | §4.1 表6、§4.2 表7、§5.3 表9、§5.4 表10、§5.5、§8.4 表12–13、§8.5 表14、§9 |
| [sota] | 使用者附檔文字版 | — | `<SCRATCH>/sota.txt` | §3.1 表S1、§3.2 表S2、§5 表S5、§6 表S6 |
| [WE] | `docs/research_20260924/workloads_eval.md` | repo 現況 | — | §1.1 第 69 行、§1.2 第 83–86 行、第 501 行 |

程式碼 repo 的存在與最新 commit 以 GitHub API 查證（2026-10-06）：apple/ml-learning-to-evict（已轉到 apple-aiml-research，d33c2dc，2026-09-11）、RUCAIBox/ForesightKV（fdb541f，2026-05-19）、SamsungLabs/LookaheadKV（27a69fd，MIT）、ngocbh/trimkv（49d2a29，Apache-2.0）。這四個只查了存在，**沒有讀程式碼**。

## 本組的共同模式（L2 學習式快取 vs L1 學習式 token 逐出）

1. **量的東西完全不同。** L2 三篇量命中率與 TTFT（LARU 用 P99、SAECache 用平均；〔複核修正〕LPC 的 TTFT 原文沒寫統計量，p7 只定義為 client 端 TTFT，p9 Fig.7 是微基準的 miss vs hit 比較），快取本身無損，不量品質〔原文〕[LARU] p7、[LPC] p7、p9、[SAE] p8。L1 四篇量任務品質對預算的曲線（accuracy、pass@1、perplexity、LongBench 平均），系統面只報 TTFT overhead、FLOPs 或 decode 吞吐〔原文〕[KVP] p8、p17；[FKV] p6–7；[LKV] p8；[TRIM] p16。兩組沒有共同指標。
2. **長度與到達過程剛好互補，都沒覆蓋「長 context × 跨請求」。** L2 都是短 context：LPC 三個資料集平均輸入 36–113 token〔原文〕[LPC] p7 Table 1；LARU 只報 Aibrix-Synthetic 平均 prompt 2,761 token〔原文〕[LARU] p7；SAECache 主結果沒給長度〔原文〕[SAE] p8。L1 評到 32K–128K（KVP RULER-128K、LookaheadKV 附錄 RULER 128K、TRIM-KV LongMemEval_S 約 123K），但都是單請求、沒有到達過程〔原文〕[KVP] p9、[LKV] p18、[TRIM] p18。
3. **label 的來源決定了是否綁 LLM。** L2 的 label 來自 trace 本身的未來存取（下次請求時間、對話是否延續、命中回饋），不需要跑 LLM〔原文〕[LARU] p5–7、[LPC] p3–4、[SAE] p12–13。L1 的 label 來自目標 LLM 的未來注意力（KVP、ForesightKV、LookaheadKV）或對原模型輸出分布的蒸餾（TRIM-KV），需要目標 LLM 的 forward（甚至 backward），所以每個模型各訓一次〔原文〕[KVP] p5–6、[FKV] p16、[LKV] p5–6、[TRIM] p6。〔複核補充〕要不要穿過 LLM 做 backward 各篇不同：KVP 只在收 trace 時跑一次 forward，訓練迴圈內不跑 LLM（[KVP] p6 §3.2）；ForesightKV 的 RL 階段要用目標 LLM 在逐出後的 forward 算 loss，但明說 LLM 不做 backward、不更新參數（[FKV] p2、p4）；LookaheadKV 的 loss 要反傳到各層的 LoRA（[LKV] p5 §3.2）、TRIM-KV 的 gate 嵌在每個 attention block 內（[TRIM] p5–6），兩者都要穿過凍結的 LLM 做 backward。例外：SAECache 的「首輪是否變多輪」分類器讀的是服務中 LLM 的 hidden state，等於也綁 LLM〔原文〕[SAE] p7–8。
4. **沒有一篇做「依時間切」的離線評估。** LARU 是線上 prequential（滑動窗、每 10^3 筆重訓）〔原文〕[LARU] p7；LPC 用資料集的一半對話訓練（資料沒有時間戳）〔原文〕[LPC] p4、p7；SAECache 沒說分類器怎麼切〔原文〕[SAE] p19–20。L1 的慣例是「訓練資料集 ≠ 評測 benchmark」再加長度外推〔原文〕[KVP] p8–9、[FKV] p8、[LKV] p9、[TRIM] p20。〔複核修正〕KVP 是部分例外：主結果（Fig.2）是在 RULER-4k、OASST2-4k 的同資料集 test 切分上評（in-distribution），跨資料集只在零樣本泛化實驗（[KVP] p6–8）。
5. **跟上界比的方式各自為政。** LARU 只在動機圖畫 OPT〔原文〕[LARU] p2 Fig.1；LPC 自訂「完美知道是否延續」的 Oracle，不是 Belady〔原文〕[LPC] p23 B.4；KVP 以「依真實未來注意力排序」的成本做 reward 正規化〔原文〕[KVP] p5；ForesightKV 比的是 Golden Eviction 的 LM loss，不是任務分數〔原文〕[FKV] p5 Table 2；SAECache、LookaheadKV、TRIM-KV 沒有上界〔原文〕。沒有一篇在主結果中報「同一成本模型下的 OPT／Belady」。
6. **只有 LARU 系統性地測「預測錯時會怎樣」。** LARU 用弱預測器（訓練窗縮到 0.1%）與合成雜訊（以機率 p 把預測換成最壞值）測，並有 1-consistent、O(k)-robust 的保證〔原文〕[LARU] p4–5、p8–9。其他篇測的是分布外泛化：LPC 的 unified vs specialized〔原文〕[LPC] p22；SAECache 的固定參數錯配〔原文〕[SAE] p5–6；L1 的跨任務、跨長度、跨溫度〔原文〕。也沒有一篇報預測機率的校準（calibration）〔原文：七篇全文中未見〕。

---

## 評測卡

### LARU（LCR）：Toward Robust and Efficient ML-Based GPU Caching for Modern Inference（preprint；arXiv 2509.20979）

- **讀了什麼**：〔全文〕arXiv v2（2026-04-24），https://arxiv.org/pdf/2509.20979v2 ，查證 2026-10-06。v1（2025-09-25）摘要裡的「P99 TTFT −28.3%、吞吐 +24.2%」兩版相同。
- **一句話**：在 LRU 上接 GBDT 預測，偵測到預測錯就逐步縮回 LRU 的 GPU 快取。
- **評測要證明的主張**：LARU 預測準時接近最佳（1-consistent），預測爛時接近 LRU（O(k)-robust），時間與空間開銷和 LRU 同級；接到 SGLang（LLM）與 HugeCTR SLS（DLRM）後，尾延遲和吞吐都改善，預測爛時也不崩。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | 〔原文〕LLM：Qwen2.5-32B、DeepSeek-R1-671B。〔未查證〕注意力類型、權重與 KV dtype 原文沒寫 | [LARU] p7 §5.1 |
| 硬體 | 〔原文〕Qwen2.5-32B：一台 4×RTX 4090（24 GB）；DeepSeek-R1-671B：一台 8×H800，原文寫每張「140 GB HBM」〔判讀：與 H800 常見規格不合，可能是筆誤，未查證〕。DLRM：2×A10、512 GB DDR4，GPU 約 600 GB/s、DRAM 隨機讀約 60 GB/s | [LARU] p7 §5.1；p17 App E.1 |
| 軟體與版本 | 〔原文〕SGLang 0.4.9.post2：加 LightGBM 與線上訓練框架，把 RadixTree 的 LRU 換成 LARU；逐出候選限於 leaf node，以符合前綴語意。DLRM 用自建 SLS benchmark（HugeCTR SlabHash 索引＋Redis 當 DRAM 後端） | [LARU] p6–7 §5.1；p18 App F；p17 |
| 資料／負載 | 〔原文〕Aibrix-Synthetic（Aibrix benchmark 生成，500 對話、1,817 請求）；Online-QA（Company A 內部多輪日誌，2,000 對話、7,268 請求）；Qwen-Bailian Trace A（超過 40,000 個多輪請求）**只出現在動機圖 Fig.1，主結果沒有用**。DLRM：AD-CTR-User（內部）、QB-video（Tenrec 子集） | [LARU] p7 §5.1；p2 Fig.1；p17 E.1 |
| 長度 | 〔原文〕Aibrix-Synthetic 平均 prompt 2,761 token；Online-QA 的長度沒給 | [LARU] p7 |
| 到達與併發 | 〔原文〕以併發數控制：Aibrix 2／5；Online-QA 10／20。〔未查證〕是否用原始時間戳、是否閉迴路，原文沒說 | [LARU] p7 Fig.3；p8 Fig.4 |
| 重用結構 | 〔原文〕多輪對話的前綴，以 RadixTree 節點為逐出單位 | [LARU] p7 |
| 掃描的自變數 | 〔原文〕cache ratio（Aibrix 5／10／15%，約 2.7×10^4–8.6×10^4 token；Online-QA 2／4／8%，約 15.2×10^4–46.9×10^4 token）、併發、雜訊機率 0–1、訓練窗大小（10^5 vs 10^2） | [LARU] p7–9 |
| 對手 | 〔原文〕SGLang 原生 LRU；FPB（盲從預測）；HF（仿 HALP，先用 LRU 篩出 4 個候選）。為了公平，FPB、HF、LARU 都用同一個 LightGBM | [LARU] p7 §5.1 |
| 系統指標 | 〔原文〕主指標 P99 TTFT；KV cache hit rate。DLRM：平均 SLS 延遲、吞吐 | [LARU] p7 §5.2；p17 E.2 |
| 品質指標 | 〔判讀〕無：快取無損，不涉及品質 | — |
| 主要結果 | 〔原文〕Aibrix＋Qwen2.5-32B：P99 TTFT −6.0%～−10.7%（cache ratio 5–15%）。Online-QA＋DeepSeek-R1-671B：−13.5%～−28.3%（2–8%）。DLRM：SLS 延遲最多 −19.5%（AD-CTR-User）、−14.2%（QB-video），吞吐 +24.2%／+16.6% 〔複核補充〕Fig.3 圖內共 6 個標註：−4.6%、−11.4%、−6.5%、−6.7%、−6.0%、−10.7%。依標註位置判讀，內文的 6.0–10.7% 只對應 5 併發；2 併發是 −4.6%、−11.4%、−6.5%（[LARU] p7 Fig.3，對應關係為〔判讀〕） | [LARU] p7；p17 E.2 |
| 消融／敏感度／開銷 | 〔原文〕(1) 弱預測器：訓練窗從 10^5 縮到 10^2；(2) 合成雜訊：以機率 p 把預測換成「真實下次請求時間的負值」，p=0.1–1.0；(3) LightGBM 線上訓練數萬筆 <100 ms；(4) 批次預測 10^4 個候選 <10 ms；(5) DLRM batch 512 預測 <1 ms；(6) DLRM 的 LARU 額外 hash table <0.2% 快取大小 | [LARU] p6 §4.1；p7；p8–9 Fig.5–6；p17 |
| 重複與統計 | 〔原文〕未說明重複次數，也沒有誤差棒 | — |
| 程式碼／資料 | 〔原文〕作者說補充材料含全部程式碼與非機密 trace。〔未查證〕公開 repo 沒找到 | [LARU] p18 App F |
| 設計理由（原文） | 〔原文〕用 LightGBM：快取預測器要微秒級推論、記憶體小、快速適應、不搶 GPU；深度模型是毫秒級且吃 HBM，RL 適應太慢（p6）。LLM 用同步模式：每 GPU 併發 <100，TTFT 數百毫秒到數秒，預測延遲可忽略（p7）。主指標選 P99 TTFT：SLA 看尾延遲（p7）。不用 BLINDORACLE&LRU：要維護兩套索引，在 RadixTree 上太貴（p2、p4） | [LARU] p2、p4、p6–7 |
| 設計理由〔判讀〕 | 〔判讀〕「phase 內被預測踢掉的項目又被請求」不需要知道真實未來就能當誤判訊號，所以能線上偵測；候選集大小 λk 連續調整，介於 FPB（全信）與 HF（固定 4 個）之間 | — |
| 原文沒講清楚的地方 | 〔判讀〕(1) 之後再也沒被存取的節點，label（下次請求時間）怎麼處理（截尾）；(2) Online-QA 的到達方式與長度；(3) LLM 部分 cache ratio 的分母；(4) 4090 上 32B 怎麼部署（TP？）；(5) Qwen-Bailian 為何沒進主結果；(6) 重複次數。〔複核補充〕(7) 前後矛盾：p3 貢獻列表寫吞吐 +24.2% 是在 QB-Video，p9 與 p17 E.2 則寫 AD-CTR-User 延遲 −19.5%、QB-video −14.2%，「分別」對應吞吐 +24.2%、+16.6%；1/(1−0.195)≈1.242 支持 24.2% 屬於 AD-CTR-User〔計算〕 | [LARU] p3、p9、p17 |
| 與既有整理不一致 | 〔原文〕intro 表6、§5.4 表10、§5.5，以及 sota 表S1、表S2、表S5 大致一致。細節修正：(1) sota 表S1 寫 LARU「預測什麼：逐出優先順序」→ 原文預測的是**下次請求時間（reuse interval）**，逐出時在 LRU 前 l 個候選中挑預測最遠的（[LARU] p5 Alg.1 第 16–18 行、p7）。〔複核修正〕這一條不算錯：原文 p5 §4 自己也說 LARU 向 predictor 查詢「eviction priorities」；只是模型輸出的量是下次請求時間，再當逐出優先順序用。建議措辭補成「下次請求時間（當逐出優先順序用）」，不必列為錯誤。(2)「偵測到預測錯就退回 LRU」→ 原文是每偵測到一次就做一次 LRU 逐出、λ 減半，新 phase 重設 λ=1；是**逐步縮小候選集**，不是整個切換（p4–5）。(3) intro §8.4「GBDT 微秒級推論」只有批次攤提的依據：10^4 候選 <10 ms，即每個候選 <1 µs〔計算〕；單次呼叫延遲原文沒報。〔複核補充〕原文 p6 §4.1 把「microsecond-level inference」列為快取預測器的需求，並說 LightGBM 推論快，但沒有單次呼叫的量測。(4) intro 表7 LARU「最長評測 —」可補：唯一有給的長度是 Aibrix 平均 prompt 2,761 token。(5) intro 表6 的 28.3% 是 Online-QA＋DeepSeek-R1 的最佳值，Aibrix 最多 10.7%。WE 沒有收錄 LARU | [intro] 行 305–306、359、467、494、722；[sota] 行 218–223、290、295、419 |
| 對本研究的意義〔判讀〕 | 〔判讀〕這是「學習式＋有保證的保底」的直接對手。可以沿用：同一個預測器比 FPB／HF／LARU、雜訊模型、弱預測器三種壓力測試。不可比：單層、每次 miss 成本相同（p3 §2 把成本定義為 miss 次數）、沒有長 context。要小心：在閉迴路重播裡，「下次請求時間」包含伺服器回應時間，是部分綁硬體的 label | [LARU] p3 §2 |

**學習設定**

| 項目 | 內容 | 出處 |
|:--|:--|:--|
| 預測目標與 label | 〔原文〕每個 RadixTree 節點的下次請求時間（reuse interval）；從滑動歷史窗中「同一節點再被存取的時間」取得 label | [LARU] p5 Alg.1、§4（Fig.2 說明）；p7 |
| 特徵 | 〔原文〕最近 10 個請求間隔（delta）、10 個指數衰減計數器（EDC，取自 LRB）、session 輪數、prompt 長度。DLRM 版只用 4 個：timestamp、key、請求次數、上次間隔 | [LARU] p7；p17 |
| 模型種類與大小 | 〔原文〕LightGBM（GBDT），在 CPU 上執行。〔未查證〕樹數與深度沒寫 | [LARU] p6 §4.1 |
| 訓練資料 | 〔原文〕線上產生：每次存取一個節點產生一筆樣本；滑動窗 10^5 筆；每新增 10^3 筆重訓一次 | [LARU] p7 |
| 切分 | 〔判讀〕沒有離線切分，等於 prequential（一律用過去預測未來）；不存在 train/test 洩漏問題，但也沒有離線準確度數字 | [LARU] p7 |
| 訓練成本 | 〔原文〕數萬筆樣本 <100 ms（CPU） | [LARU] p6 |
| 推論開銷怎麼量 | 〔原文〕同步模式下，10^4 候選批次預測 <10 ms；沒有實測 TTFT 拆解，以「TTFT 數百毫秒到數秒」論證可忽略 | [LARU] p7 |
| 換模型／負載／硬體 | 〔原文〕靠持續線上重訓追負載；兩個 LLM 用同一套特徵。〔原文：未見〕沒測跨 trace、跨硬體的遷移 | [LARU] p6–7 |
| fallback／保證 | 〔原文〕1-consistent、O(k)-robust（Theorem 1）；時間 O(n log k)、空間 O(k) | [LARU] p4–5 |
| 與 Oracle／Belady | 〔原文〕OPT 只出現在動機圖：LRU 與 OPT 命中率差距，LLM 最多 28.7%、DLRM 15.6–23.6%。評測主結果沒有 OPT 線 | [LARU] p2 Fig.1 |

---

### LPC：Learned Prefix Caching for Efficient LLM Inference（NeurIPS 2025；未找到 arXiv 版）

- **讀了什麼**：〔全文〕NeurIPS 2025 proceedings PDF（23 頁，含附錄 A–B 與 checklist），https://proceedings.neurips.cc/paper_files/paper/2025/file/414f642a1ea9350006669774cba9bcd4-Paper-Conference.pdf ，查證 2026-10-06。OpenReview（https://openreview.net/forum?id=Vj48eXaQDM）的 PDF 被驗證頁擋住，沒取得。另讀〔程式碼〕https://github.com/yangdsh/LPC commit 3e750fa（2025-10-10）。
- **一句話**：用對話文字預測「這段對話會不會繼續」，取代 LRU 的時間戳排序。
- **評測要證明的主張**：學到的延續機率加上時間衰減，命中率高於 LRU，同命中率可省 18–47% 快取；預測器的記憶體與計算開銷很小。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | 〔原文〕Qwen3-32B-FP8（32B 推理模型），單卡。預測器另用 multilingual-e5-small（118M 參數，float16 權重 240 MB）。〔未查證〕注意力類型原文沒寫 | [LPC] p4 §3.2；p6–7 §4.1 |
| 硬體 | 〔原文〕1×H100 80 GB HBM3；扣掉模型與 PyTorch 後，KV＋prefix cache 共 40 GB；8 CPU、64 GB CPU 記憶體 | [LPC] p6–7 |
| 軟體與版本 | 〔原文〕在 vLLM main branch（2025-03-10）上實作；預測器是外部元件，逐出策略寫進 vLLM 的 prefix cache；其餘 vLLM 參數用預設值。〔程式碼〕README 以 vLLM 0.7.3 預編譯 wheel 安裝 | [LPC] p6 §4.1；[LPC-code] README |
| 資料／負載 | 〔原文〕LMSys（1×10^6 對話；平均輸入 63、輸出 179 token；follow-up 35%）、ShareGPT（9.5×10^4；113／305；55%）、Chatbot-Arena（3.3×10^4；36／155；8%）。吞吐微基準是合成的：1,000-token context＋50-token 新 prompt | [LPC] p7 Table 1；p9 §4.5 |
| 長度 | 〔原文〕三個資料集平均輸入 36–113 token；微基準 1,000-token context | [LPC] p7；p9 |
| 到達與併發 | 〔原文〕兩階段：會話內間隔取指數分布（平均 λ_chat，預設 100 s）；會話開始時間錯開，使同時活躍的會話 ≤ N_conv=200；執行時，下一請求＝上一回應的實際完成時間＋取樣的間隔；每次跑 20 分鐘，避開尾端冷卻。微基準假設 prefill 請求率 <10 req/s | [LPC] p7 §4.2；p8 §4.5 |
| 重用結構 | 〔原文〕多輪對話前綴；多個對話共用的 block（例如系統提示）取各對話機率的最大值 | [LPC] p6 §3.5 |
| 掃描的自變數 | 〔原文〕總 token 容量 60K–160K（約 15.6–40 GB）；λ_chat；預測器記憶體 0×／1×／2×／4× | [LPC] p7 §4.3；p8 §4.4；p20 A.2 |
| 對手 | 〔原文〕只有 vLLM 的 LRU（作者在限制中自承） | [LPC] p7；p10 §6 |
| 系統指標 | 〔原文〕hit ratio＝所有請求的平均 prefix 命中率；TTFT（client 端）；throughput＝每秒 prefill 的 sequence 數；DCGM 的 SM Active、SM Occupancy、Tensor Pipe Active | [LPC] p7；p20 Table 2 |
| 品質指標 | 〔原文〕無（快取無損）；預測器另用 MCC 與 macro F1 | [LPC] p21 A.3 |
| 主要結果 | 〔原文〕同命中率下省 18–47% 快取；命中率相對 LRU 提升 13–38%（LMSys）、14–98%（ShareGPT）、15–30%（Arena）；固定 40 GB、改 λ_chat 時提升 5–18%；微基準命中率 35%→42%，對應 prefill 吞吐 10.01→11.11 req/s（+11.1%）；context 命中讓 TTFT 最多 −73% | [LPC] p7–9 |
| 消融／敏感度／開銷 | 〔原文〕只用輪數的 LPC_turns；加入模型回應；prompt window 5 vs 10；unified vs specialized 預測器；預測器記憶體敏感度；與 Oracle 比；GPU 計算開銷（DCGM） | [LPC] p20–23 |
| 重複與統計 | 〔原文〕每個點至少跑 5 次，誤差棒為 min–max；任一設定的 max−min <0.013，相對變異 <4.6% | [LPC] p23 B.5 |
| 程式碼／資料 | 〔程式碼〕公開；GitHub API 顯示無 license 檔 | [LPC] p2；[LPC-code] |
| 設計理由（原文） | 〔原文〕只用 user prompt、不用模型回應：回應的新資訊少，消融也沒變好（p4、p21–22）。N=4：資料中 94–99% 的對話不超過 5 輪（p22）。凍結 embedding、只訓 MLP：全量微調太貴，只訓 MLP 才能每天重訓（p5）。自訂兩階段到達：公開資料沒有時間戳，Poisson 會讓請求早於前一回應完成（p7）。扣 1 GB 快取：對 LRU 公平（p7）。decay scale 取平均輪間隔（約 100 s）的倒數（p6） | [LPC] p4–7、p21–22 |
| 設計理由〔判讀〕 | 〔判讀〕把預測頻率壓到「每個使用者請求一次」，讓 118M 的嵌入模型也能上線；時間衰減其實是用時間把錯的高機率慢慢沖掉，扮演一點保底角色，但沒有保證 | — |
| 原文沒講清楚的地方 | 〔原文〕論文只說訓練用「資料集一半的對話、與線上評測嚴格隔離」（p4）。〔程式碼〕`load_conv_data` 在 N<0 時取資料檔**尾端** N 個對話當訓練集，`prepare_dataloaders` 再對「逐輪樣本」做隨機分層切分（`train_test_split(..., random_state=42)`），同一對話的不同輪可能同時落在訓練與驗證（只影響 checkpoint 選擇）；〔複核修正〕`get_predictor_accuracy.py`（main 區塊 N=20000）不是「前 20,000 個對話」：`load_conv_data` 的計數器 `processed_count` 每處理一輪 user→assistant 就加 1，所以是從資料開頭讀到約 20,000 個輪樣本為止，再以 `test_size=0.5` 對逐輪樣本隨機切，前半挑 MCC 最佳門檻、後半報 MCC／F1（L118–170、L551–580、L643–675）。同理，N<0 時 `data[-N:]` 取尾端 N 個對話，但同樣在累計 N 個輪樣本時停。另：`learn_conversation.py` main 區塊的 `lr=5e-5`，與論文的 5×10^-4 不同（L490 vs [LPC] p5）。〔未查證〕論文數字是否就是這組設定。其他：MLP 訓練硬體；decay scale 是否看過測試資料才調；預測吞吐的量測條件；Fig.9 圖說寫 22–66%，內文最大 64%（p23）；〔複核修正〕74% 出現在引言（p1「as shown in Section 4.5」），不是摘要；內文 73%（p9）；MCC 內文寫 0.28–0.36，Table 3 最大 0.3863（p21）。〔複核補充〕「省 18–47% 快取」只出現在摘要、p2 貢獻與 Fig.4 圖說；p8 內文給的是 LMSys 最多 18%、ShareGPT 最多 30%、Chatbot-Arena 85K vs 150K（約 43%），結論 p10 寫「up to 43%」，47% 在內文找不到對應 | [LPC] p1、p4、p8–10、p21、p23；[LPC-code] `learn_conversation.py` L69–165、L450–492、`get_predictor_accuracy.py` L118–170、L551–675（複核者 2026-10-07 以 GitHub API 確認 main 仍是 3e750fa，且與抽取者的檔案逐位元相同） |
| 與既有整理不一致 | (1) intro 表9 說明與 sota 表S1 寫「LPC 要訓練一個 118M 參數的文字嵌入模型（依 SAECache 的描述）」→ **錯**。原文：118M 的 e5-small 凍結不訓，只訓 3 層、每層 128 的 MLP，訓練「通常 <10 分鐘」〔原文〕[LPC] p4–5。錯誤源頭是 SAECache 的轉述〔原文〕[SAE] p11。(2) intro 表9 把 LPC 歸「深度學習：訓練成本高（多張資料中心級 GPU）；重訓原因：更換 LLM」→ 與原文不符。訓練 <10 分鐘；原文建議每天重訓，理由是使用者行為漂移〔原文〕p5；輸入只有文字與輪數，不讀 LLM 的 K/V，換 LLM 照理不必重訓〔判讀〕。原文只說原理與模型無關，實驗只用一個模型族〔原文〕p10。(3) sota 表S1 LPC「保底：未查證」→ 原文**沒有**退回機制或保證，只有機率衰減〔原文〕p5–6；`research_20260924/novelty_sota.md` 第 353 行「是否有 fallback：未查證」同樣可更正。(4) intro 摘要第 4 點「深度學習派確實訓練重、而且綁定 LLM」對 LPC 不成立。(5) sota 表S6「LPC 改的是 vLLM 0.7.3」與 README 一致（論文寫 2025-03-10 main branch）。(6) intro 表6「快取 ↓18–47%」、表14「用對話內容預測會不會延續」已核對一致。〔複核補充〕18–47% 與 LPC 摘要一致，但 LPC 內文 p8 的具體值最大約 43%、結論寫「up to 43%」，引用時宜註明出自摘要。WE 沒有收錄 LPC | [intro] 行 31–34、308–309、431–444、447–448、746；[sota] 行 245–249、446 |
| 對本研究的意義〔判讀〕 | 〔判讀〕值得沿用：「扣掉預測器自己佔的記憶體再比」、至少 5 次重複、自訂 Oracle 量「縮小多少差距」、unified vs specialized 測跨負載。不可比：輸入只有幾十到一百多 token、只比 LRU、到達過程是合成的。要小心：label「會不會延續」與硬體無關，但 decay 的時間尺度（約 100 s）是從同一批資料調出來的 | [LPC] p6、p7、p22–23 |

**學習設定**

| 項目 | 內容 | 出處 |
|:--|:--|:--|
| 預測目標與 label | 〔原文〕「這段對話在本輪之後還會不會有下一輪」的二元 label，直接由資料集的對話結構得到。〔程式碼〕assistant 回覆之後的下一則訊息（`messages[i+2]`）是不是 user 就是 label（〔複核修正〕原寫「下一則訊息」，精確是 assistant 回覆之後那一則，`learn_conversation.py` L111） | [LPC] p3 §3.1、p4；[LPC-code] `learn_conversation.py` `load_conv_data` |
| 特徵 | 〔原文〕新 prompt 加前 4 輪 user prompt：512 token 預算平均分給 5 輪，每輪取頭尾各半，送進 e5-small 得 384 維嵌入；另加已發生的輪數 | [LPC] p4 |
| 模型種類與大小 | 〔原文〕凍結的 e5-small（118M）加 3 層 MLP（每層 128），輸出 0–1 機率；BCE（少數類加權）、Adam 5×10^-4、20 個 epoch 內收斂、取驗證 loss 最低者 | [LPC] p4–5 |
| 訓練資料 | 〔原文〕每個資料集各訓一個模型，用該資料集一半的對話 | [LPC] p4 §3.3 |
| 切分 | 〔原文〕依對話劃分（資料沒有時間戳，無法依時間切）；程式碼細節見上表 | [LPC] p4；[LPC-code] |
| 訓練成本 | 〔原文〕MLP 訓練 <10 分鐘；第一個 epoch 後快取 embedding，省 90% 時間。〔未查證〕硬體沒寫 | [LPC] p5 |
| 推論開銷怎麼量 | 〔原文〕每個使用者請求只預測一次；GPU 記憶體 1 GB（CUDA context 560 MB＋權重 240 MB＋執行期 200 MB），而且從 LPC 的快取扣掉；最多 1,000 次預測／秒／GPU；把預測器加到 LRU 上，SM Active +0.72%；預測器記憶體從 0× 到 1×，命中率降 0.6–2.4%（40 GB）、2.3–8.1%（16 GB） | [LPC] p6；p9 §4.6；p20 A.1–A.2 |
| 換模型／負載／硬體 | 〔原文〕建議每天離線重訓；unified 預測器在 ShareGPT 上 MCC 從 0.2777 掉到 0.1271。〔原文：未見〕沒測換 LLM、換硬體 | [LPC] p5；p22 Table 6 |
| fallback／保證 | 〔原文〕無；以指數衰減逐步壓低高估的機率 | [LPC] p5–6 §3.4.2 |
| 與 Oracle／Belady | 〔原文〕自訂 Oracle＝完美知道對話會不會延續（不是 Belady）。LPC 把 LRU 到 Oracle 的命中率差距縮小 22–30%（LMSys）、28–64%（ShareGPT）、25–28%（Arena） | [LPC] p23 B.4 |

---

### SAECache：Not All Tokens Are Worth Caching: Learning Semantic-Aware Eviction for LLM Prefix Caches（preprint；arXiv 2605.18825）

- **讀了什麼**：〔全文〕arXiv v1（2026-05-12），https://arxiv.org/pdf/2605.18825v1 ，31 頁含附錄 A–E 與 NeurIPS checklist，查證 2026-10-06。
- **一句話**：依 token 類型（系統提示、CoT…）與 session 結構把 block 分到不同佇列，參數線上學的前綴快取逐出。
- **評測要證明的主張**：不同 token 類型的重用率差很多；多佇列加線上學參數，命中率高於 LRU 與 LPC；固定參數在負載錯配時會差到 2.7×。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | 〔原文〕整合驗證用 Qwen2.5-1.5B-Instruct。主結果是 trace-driven 模擬〔原文：checklist Q8〕，〔未查證〕模擬時對應的模型沒寫。附錄 Fig.18 拿「LLAMA-8B 上的 SAECache」比「Qwen3-32B 上的 LPC」 | [SAE] p8；p28；p21 |
| 硬體 | 〔原文〕1×A40（整合驗證） | [SAE] p8；p28 |
| 軟體與版本 | 〔原文〕vLLM v0.8.5（V0 engine）；新增 Cache Evictor 類別取代 LRU Evictor，不改 scheduler、executor | [SAE] p8 |
| 資料／負載 | 〔原文〕主結果：ShareGPT（74% 多輪）、LMSys（33%）、Chatbot-Arena（12%）。消融用 4 個合成負載：tool_use（282 請求、85% 共享前綴）、multi_turn_dominant、balanced、single_turn_dominant（Table 2）。Qwen-Bailian（9,012 session、19,957 個間隔）與 CC-Bench-V1.1（370 條軌跡、15,755 個間隔）只用來擬合輪間間隔。〔未查證〕Table 1 重用率特徵化用的是哪份資料，沒明說 | [SAE] p8；p15 App D；p16–17 App E；p5 |
| 長度 | 〔原文〕主結果沒給長度；附錄 Fig.18 只提「1000-token context」 | [SAE] p21 |
| 到達與併發 | 〔原文〕固定注入間隔 {0.02, 0.03, 0.05, 0.08} s，作者說四個都在 GPU 飽和、高負載區 | [SAE] p8 §4.1 |
| 重用結構 | 〔原文〕多輪 session（chat／agent 分開）；單輪的模板重用（系統提示、工具描述） | [SAE] p3–4 |
| 掃描的自變數 | 〔原文〕注入間隔 × 3 個資料集；消融逐步加入學習元件；token 權重更新式的 (α, β) 共 30 組 | [SAE] p8；p14–16 |
| 對手 | 〔原文〕LRU、LPC。〔未查證〕LPC 是否用官方程式碼重跑沒說 | [SAE] p8 |
| 系統指標 | 〔原文〕prefix cache hit ratio；mean TTFT，以 prefill_tokens = prompt_length×(1−hit_ratio) 連結。〔未查證〕TTFT 是實測還是由此式算出 | [SAE] p8 |
| 品質指標 | 〔判讀〕無 | — |
| 主要結果 | 〔原文〕12 格命中率全部最高，比該格最強對手多 4.8–5.9 個百分點；LMSys mean TTFT 比 LRU −4–8%、比 LPC 最多 −16%；ShareGPT 在注入間隔 ≤0.03 s 時與最佳對手持平，其餘在 5% 內；Chatbot-Arena 比最佳對手**慢 12–34%**。〔判讀〕摘要的「TTFT 1.4–2.7×」在主結果中找不到對應 | [SAE] p8–9；p1 |
| 消融／敏感度／開銷 | 〔原文〕合成負載上的逐步消融：token 類型權重 +23%、log-normal 時序 +14%、位置衰減 +12%、佇列權重 +39%，合計比固定參數 +88%。固定參數錯配：TTFT 4.014→7.127 s；學習版 0.934–1.886 s。(α, β) 30 組 TTFT 都是 3.564 s。預測器開銷：1,137 vs 129（LPC e5-small）次預測／秒；1M vs 118M 參數；約 4 MB vs 472 MB | [SAE] p15–16；p5 Fig.4；p14；p9 |
| 重複與統計 | 〔原文〕沒有誤差棒、沒有多種子（checklist Q7 答 No） | [SAE] p27 |
| 程式碼／資料 | 〔原文〕「接受後釋出」。〔未查證〕目前沒找到 | [SAE] p26 Q5 |
| 設計理由（原文） | 〔原文〕系統提示重用率 92.3%、CoT 2.2%，差 42×（p5 Table 1）；多輪重用 99.28%（ShareGPT 型）、99.96%（AgentBank 型）發生在 session 內（p4）；輪間間隔服從 log-normal（R²≥0.987、K-S≤0.060，p4、p17）；固定參數在錯配時很脆弱，所以全部參數都線上學（p6）；session predictor 重用已算好的 hidden state，不需要外部模型（p8） | [SAE] p4–8、p17 |
| 設計理由〔判讀〕 | 〔判讀〕把 block 依「內容類型」分佇列，等於把寫入當下就知道的資訊（系統提示、CoT、工具輸出）拿來當先驗；這正是使用者「寫入時決策」的訊號來源之一 | — |
| 原文沒講清楚的地方 | 〔原文〕(1) 主實驗的模型、硬體、是不是真的執行；(2) TTFT 怎麼得到；(3) session predictor 的訓練資料與切分；(4) 輸入前後矛盾：p7–8 說用最後一層 hidden state，p20 Fig.15 的特徵重要度卻列 open_ended_score、response_length 等手工特徵；(5) 重用率差距三個數字不一致：摘要 756×、p2 寫 44×、p5 寫 42×；(6) 1.4–2.7× 的來源 | [SAE] p1、p2、p5、p7–8、p19–20 |
| 與既有整理不一致 | (1) intro 表9 把 SAECache 歸「GBDT／輕量模型：預測下次被存取的時間、在 CPU 執行、在 CPU 上線上重訓」→ 不精確。SAECache 沒有 GBDT；它用 log-normal 存活機率（隨經過時間遞減）、token 類型權重、佇列權重的線上統計更新，另有一個**離線訓練**的 3 層 MLP（約 1M 參數），讀**服務中 LLM 最後一層的 hidden state**，判斷首輪會不會變多輪〔原文〕p6–8、p19。這個 MLP 綁定 LLM 的 hidden 維度〔判讀〕。〔複核補充〕「離線」是推論：原文只說在真實對話資料上訓練 50 個 epoch（p19），沒用 offline 一詞；原文全文沒有 GBDT／LightGBM，也沒說預測器在 CPU 上執行（全文檢索無 CPU 執行的描述）；且 p19–20 把同一個分類器描述成看 <prompt, response> 與手工特徵，與 p7–8 的 hidden state 說法矛盾。(2) intro 表9 說明「SAECache 不需要定期重訓，參數線上更新」是 SAECache 自己的說法〔原文〕p11，對統計參數成立；但 MLP 訓了 50 個 epoch〔原文〕p19，換 LLM 要不要重訓原文沒討論。(3) WE 第 501 行「SAECache 對 Bailian 擬合 μ≈4.1、σ≈1.0（p17）」→ 那是**線上估計的收斂值**；離線 MLE 參考值是 μ=4.82、σ=1.25（[SAE] p17 Table 3）；p6 還有一組固定參數 μ=4.15、σ=0.97。(4) SAECache 對 LPC 的轉述有兩處錯：說 LPC「訓練 118M 嵌入模型」（實為凍結）；說 LPC decay scale「e.g., 0.01s」（LPC 原文 scale=10^-2，對應約 100 s 的平均輪間隔）〔原文〕[SAE] p4、p11 vs [LPC] p5–6。(5) WE 第 69 行其他欄位已核對一致。〔複核修正〕WE 第 69 行寫 Qwen2.5-1.5B「GQA」，SAECache 全文沒提注意力類型（全文檢索無 GQA），這一格應標〔未查證〕，不算「一致」 | [intro] 行 431–447；[sota] 行 225–228；[WE] 行 69、501 |
| 對本研究的意義〔判讀〕 | 〔判讀〕可以沿用：輪間間隔的 log-normal 模型與 Bailian、CC-Bench 參數（「多久之後回來」的先驗）；「固定參數 vs 線上學」的錯配實驗。要小心：主結果是模擬、沒重複、TTFT 可能是公式算的；Chatbot-Arena 變慢說明預測器與簿記開銷在低重用負載下會反噬；Fig.18 跨模型比較不可採信 | [SAE] p16–17、p8–9、p21 |

**學習設定**

| 項目 | 內容 | 出處 |
|:--|:--|:--|
| 預測目標與 label | 〔原文〕(a) 每個多輪佇列的輪間間隔分布（log-normal μ、σ），用來算 block 的存活機率；label＝多輪 block 命中時距上次存取的時間。(b) token 類型權重、位置衰減冪次 γ、佇列權重 α；回饋來源是逐出後又被請求的比例（miss-after-eviction）、各類命中率、各佇列每單位容量的命中 token 數。(c) 首輪請求會不會變成多輪（二元），label 來自對話資料 | [SAE] p6–8；p12–14；p19 |
| 特徵 | 〔原文〕(c) 最後一層、最後一個輸入 token 的 hidden state（p7）；但 p20 列的是手工特徵，前後矛盾 | [SAE] p7–8；p20 |
| 模型種類與大小 | 〔原文〕(a)(b) 線上統計加 EMA；(c) 3 層 MLP，hidden 256／64，約 1M 參數 | [SAE] p8 |
| 訓練資料 | 〔原文〕(c)「真實對話資料」，訓練 50 個 epoch，訓練準確率 73%→82% | [SAE] p19 |
| 切分 | 〔原文〕混淆矩陣共 376 筆，準確率 77.1%、F1 0.803。〔未查證〕切分方式沒說 | [SAE] p20 |
| 訓練成本 | 〔原文〕沒報；checklist 說主要成本是 trace replay，不是訓練 | [SAE] p28 |
| 推論開銷怎麼量 | 〔原文〕每次逐出常數時間（候選挑選除外，可用每佇列一個 heap）；MLP 1,137 次／秒、約 4 MB；Chatbot-Arena 上開銷大於命中收益 | [SAE] p7（〔複核修正〕常數時間的說法在 p7 §3.1 末段，原寫 p6）；p9 |
| 換模型／負載／硬體 | 〔原文〕參數線上適應負載；固定參數錯配差 2.7×。〔原文：未見〕沒測換 LLM、換硬體 | [SAE] p5–6 |
| fallback／保證 | 〔原文〕沒有保證；evict-first 佇列先清；權重有上下限（佇列 0.1–3.0、token 0.1–5.0） | [SAE] p6；p22–23 Alg.1–3 |
| 與 Oracle／Belady | 〔原文：未見〕無 | — |

---

### KVP：Learning to Evict from Key-Value Cache（ICML 2026；arXiv 2602.10238）

- **讀了什麼**：〔全文〕arXiv v2（2026-06-26），https://arxiv.org/pdf/2602.10238v2 ，30 頁含附錄 A.1–A.7，查證 2026-10-06。程式碼 repo 存在（見來源清單），沒讀。
- **一句話**：每個 KV head 一個小 RL agent，只看 K、V、位置就排出 token 的保留順序。
- **評測要證明的主張**：學到的排序在所有預算下都比啟發式好，能泛化到 32 倍長度與沒看過的任務；訓練完全離線，不需要額外推論。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | 〔原文〕主：Qwen2.5-7B-Chat，GQA，28 層 × 4 個 KV head＝112 個 agent；附錄：Phi-4 14B。計時用 bfloat16／TF32 | [KVP] p6 §4；p16 A.2；p18 Fig.7；p24 Fig.13 |
| 硬體 | 〔原文〕收 trace：7 個節點、每個 8 張 GPU（型號沒寫）；訓練：一個節點 8×H100；計時微基準：B200。〔未查證〕準確度評測用的硬體沒寫 | [KVP] p16–17 A.3；p18 |
| 軟體與版本 | 〔原文〕短 benchmark 用 FlexAttention 自訂 mask 模擬逐出；長 benchmark（RULER-128K、LongBench passage retrieval）實際壓實 K/V tensor；下游任務用 EleutherAI lm-eval-harness | [KVP] p16 A.2；p6 |
| 資料／負載 | 〔原文〕訓練＋評測：RULER-4k（約 4,500 token；約 6,000 個訓練樣本）、OASST2-4k（訓練切分 4,649 筆）。泛化：RULER-128K（13 個子任務）、LongBench GovReport、Passage Retrieval（EN／ZH）、BoolQ、ARC-Challenge、MMLU、HellaSwag | [KVP] p6；p16 |
| 長度 | 〔原文〕訓練約 4K；評測到 128K | [KVP] p6；p8–9 |
| 到達與併發 | 〔原文〕無（單請求） | — |
| 重用結構 | 〔原文〕無；BoolQ 與 GovReport 的 prefill 不含問題，模擬「同一段文字之後才收到問題」 | [KVP] p8 |
| 掃描的自變數 | 〔原文〕絕對 KV cache 大小（保留的 token 數），刻意不用壓縮比 | [KVP] p8 |
| 對手 | 〔原文〕用注意力的 TOVA、SnapKV；不用注意力的 Random、StreamingLLM、LagKV、KeyDiff、K-Norm；附錄 JudgeQ（沒有官方碼，自己重做、用同樣資料訓練）。所有對手都改成輸出完整排序 | [KVP] p7；p17 A.4 |
| 系統指標 | 〔原文〕單一 head 的逐出決策時間（ms）；每 token FLOPs | [KVP] p17–18 |
| 品質指標 | 〔原文〕RULER 官方 accuracy、OASST2 perplexity、各下游 accuracy／Rouge-L／retrieval score；每預算成本 −R_b | [KVP] p8–10 |
| 主要結果 | 〔原文〕RULER-4k 多數預算 accuracy 最高；OASST2 PPL 除最小預算外都低於只用 K/V 的方法；RULER-128K 每個預算都領先；10K context 時單次壓縮 0.71 ms，全模型 prefill 404 ms（差 570×） | [KVP] p7–9；p18 Fig.7 |
| 消融／敏感度／開銷 | 〔原文〕reward 有效性（per-budget cost）；RL vs 5 種監督式 learning-to-rank；各 head 排序差異（每位置 rank 變異平均 294.8）；FLOPs 估算 | [KVP] p10–11；p17–18 |
| 重複與統計 | 〔原文〕計時取 30 次平均；準確度沒有誤差棒、沒說重複 | [KVP] p18 |
| 程式碼／資料 | 〔原文〕公開 github.com/apple/ml-learning-to-evict | [KVP] p1 |
| 設計理由（原文） | 〔原文〕在唯一性與巢狀性下，「所有預算的最佳逐出」等於一個排序（Prop.1），所以學一個打分函數就好（p4）。只用 K/V/位置：和 FlashAttention 相容，可在事後套用（p2、p5）。per-head：各 head 的注意力模式不同（p3、p11）。用 RL 不用監督：注意力分布重尾，監督 loss 不是被 sink 主導，就是只看排名而在小預算失準（p11、p17）。報絕對 cache 大小：實務上是固定記憶體（p8） | [KVP] p2–5、p8、p11、p17 |
| 設計理由〔判讀〕 | 〔判讀〕reward 只用 trace 中的未來注意力，不需要在訓練迴圈內跑 LLM，所以訓練便宜；代價是收 trace 很貴（1.2 TB） | — |
| 原文沒講清楚的地方 | 〔原文〕chunked prefill 的 chunk 大小；RULER-4k 的 train／test 怎麼切；準確度實驗硬體；Fig.7 圖說前半寫量的是「single KV head」，同一段圖說後半又寫「single-layer compression」（〔複核修正〕兩者都在圖說內，原寫「內文」；p18）；沒有端到端系統延遲（作者自承，p11）；〔複核補充〕「8×H100 不到 30 分鐘」沒說是 112 個 agent 全部還是單一 agent（p17 只寫 The agent training completes…） | [KVP] p7、p11、p18 |
| 與既有整理不一致 | 〔原文〕intro 表9 說明、§5.4 表10 與 sota 表S1、S2 的「收 trace 用 7 台 × 8 GPU（不到 2 小時、約 1.2 TB），訓練用 8 張 H100（不到 30 分鐘）」「每個 (層, KV head) 一個約 65 萬參數的 agent，Qwen2.5-7B 為 112 個」**已核對一致**（[KVP] p16–17）。補充：(a) 56 張 GPU 的型號原文沒寫，「資料中心級」是推論；(b) 這組數字是 OASST2 訓練切分（4,649 筆）的例子；(c) 換成 GPU 小時：收 trace <112，訓練 <4 H100 小時〔計算〕。〔複核補充〕「<4 H100 小時」假設 30 分鐘涵蓋全部 112 個 agent；原文 p17 只寫「completes in less than 30 minutes on a single node of 8 NVIDIA H100 GPUs」（主詞是 The agent training），沒說是全部還是單一 agent，若是後者總量會更高。原文另說收 trace 的成本「約等於在該資料集上做一次標準推論」，並自稱 low computational footprint（p16–17）。WE 第 83 行已核對一致；v2 另含 Phi-4、LongBench passage retrieval／GovReport、MMLU、HellaSwag、JudgeQ | [intro] 行 447、461–465；[sota] 行 251–260、285–288；[WE] 行 83 |
| 對本研究的意義〔判讀〕 | 〔判讀〕KVP 的 per-head 排序可以拿來當「段內哪些 token 該留在快的層」的訊號（作者自己也這樣提，p3），但只解單請求、有損，跟使用者的跨請求放置不在同一層。可以沿用：「所有預算一起評」的 AUC 指標與「依真實未來排序」的正規化，很適合當寫入時排序品質的離線指標 | [KVP] p3、p5 |

**學習設定**

| 項目 | 內容 | 出處 |
|:--|:--|:--|
| 預測目標與 label | 〔原文〕每個 token 的保留排序。label 是離線 trace 中未來 f 個 token 對它的注意力總和（GQA 取組內最大），用來算「所有預算下被逐出的未來注意力」的 AUC，再以最佳排序 σ* 的成本正規化 | [KVP] p5 §3.1.1 |
| 特徵 | 〔原文〕(k_i, v_i, pos_i)；不看 query、注意力分數或未來資訊 | [KVP] p5 |
| 模型種類與大小 | 〔原文〕每個 KV head 一個 2 層 MLP（256 hidden），約 650K 參數、checkpoint 2.6 MB，合計 72.8M；Plackett-Luce 策略、Gumbel-Sort 取樣、REINFORCE 加 RLOO baseline | [KVP] p5–6；p16–17 |
| 訓練資料 | 〔原文〕基底 LLM 對訓練語料跑一次，存每個序列的 Q/K/V（不存注意力矩陣） | [KVP] p6 §3.2；p16 A.3 |
| 切分 | 〔原文〕RULER-4k 與 OASST2 各有 train／test；另以 KVP_R（RULER 訓）、KVP_S（OASST2 訓）做跨資料集的零樣本測試 | [KVP] p8–9 |
| 訓練成本 | 〔原文〕收 trace：56 張 GPU 不到 2 小時、約 1.2 TB（每個 agent 只載入約 11 GB）；訓練：8×H100 不到 30 分鐘；每個 agent 4,000 步、AdamW 5×10^-5 | [KVP] p16–17 |
| 推論開銷怎麼量 | 〔原文〕prefill 從 14.00 增加到 14.15 GFLOPs／token（+1%）；只在 prefill 後壓一次，decode 零額外 FLOPs；實測見主要結果 | [KVP] p17 |
| 換模型／負載／硬體 | 〔原文〕每個模型各自訓練（Phi-4 有自己一組 agent）；有測跨任務、跨長度的零樣本。〔複核修正〕「Phi-4 有自己一組 agent」原文沒有明寫：Fig.13 圖說只說把 KVP 框架用在 Phi-4 14B 上、標為「KVP (Phi-4)」；因為 agent 的輸入是該模型的 K/V，必須另訓是〔判讀〕。〔判讀〕品質與硬體無關，換硬體只影響開銷 | [KVP] p24 Fig.13；p8–9 |
| fallback／保證 | 〔原文〕沒有保證；永遠保留最前 4 個與最後 16 個 token | [KVP] p16 A.2 |
| 與 Oracle／Belady | 〔原文〕有：reward 以「依真實未來注意力排序」的成本正規化，Fig.6 報每預算的相對成本；但下游 accuracy 沒有「依真實未來逐出」的對照 | [KVP] p5；p10 |

---

### ForesightKV：Optimizing KV Cache Eviction for Reasoning Models by Learning Long-Term Contribution（ICML 2026；arXiv 2602.03203）

- **讀了什麼**：〔全文〕arXiv v2（2026-06-01），https://arxiv.org/pdf/2602.03203v2 ，20 頁含附錄 A–E，查證 2026-10-06。程式碼 repo 存在，沒讀。
- **一句話**：長推理生成中，用小 MLP 預測每個 KV 的長期貢獻；先監督學習，再用 GRPO 強化學習。
- **評測要證明的主張**：用一半的 budget 就能贏 SnapKV／H2O／R-KV；2K、4K 預算下分別保留原模型 92%、99% 的表現；吞吐大幅提升。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | 〔原文〕主：Qwen3-1.7B、Qwen3-4B、DeepSeek-R1-Distill-Qwen-7B；附錄：MiniCPM-4.1-8B、Qwen3-14B；Qwen3-32B 只做 loss 分析。KV 以 BF16 計（Qwen3-4B 在 32K 單例 4.5 GB） | [FKV] p6；p18–19；p1 |
| 硬體 | 〔原文〕吞吐：1×A800；訓練：H800 | [FKV] p7；p16 |
| 軟體與版本 | 〔未查證〕推論引擎沒寫 | — |
| 資料／負載 | 〔原文〕訓練：Qwen3-4B 在 STILL 題目上生成的推理軌跡，只取答對且長於 4,096 token 的。評測：AIME2024、AIME2025。泛化：GPQA、LiveCodeBench-v3、LongBench（non-thinking、1K budget、prefill 後只壓一次）；另以 SCP-116K 訓練、GPQA 測 | [FKV] p16 E.1；p6；p8；p17–18 |
| 長度 | 〔原文〕生成 8K／16K／32K（吞吐表）；訓練資料最長 32K（另試 8K） | [FKV] p7 Table 3；p18 E.5 |
| 到達與併發 | 〔原文〕無到達過程；吞吐表用最大併發 batch | [FKV] p7 |
| 重用結構 | 〔原文〕無 | — |
| 掃描的自變數 | 〔原文〕budget 1K／2K／4K（附錄到 8K）；eviction length L=256 | [FKV] p6 |
| 對手 | 〔原文〕SnapKV、H2O、R-KV（依 R-KV 設定，每 L 步壓一次）；附錄 G-KV（數字取自原論文）、DuoAttention、RPC | [FKV] p6；p16 E.2；p19 E.8 |
| 系統指標 | 〔原文〕最大併發 batch、throughput（單位沒標）、端到端時間 | [FKV] p7；p19 E.9 |
| 品質指標 | 〔原文〕pass@1（溫度 0.6、top-k 20、top-p 0.95、每個 benchmark 獨立跑 32 次取平均）；LM loss ratio；注意力輸出 cosine 相似度 | [FKV] p6；p5；p15 |
| 主要結果 | 〔原文〕Qwen3-4B AIME24：1K 預算 54.5，高於 R-KV 2K 預算的 44.8；生成 32K 時 1K 預算吞吐 9.79×（A800）；逐出計算佔總時間 2.7%（R-KV 8.1%） | [FKV] p6；p7 Table 3；p19 E.9 |
| 消融／敏感度／開銷 | 〔原文〕〔複核修正〕5 種 reward（−L_all、−L_low、−L_high、−L_low,large、−L_ours），另加一欄只做監督、不做 RL 的對照（Table 4；原寫「6 種 reward」）；輸入（只用注意力特徵 vs 注意力＋KV）；取樣（Top-K、multinomial、兩者合用）；低熵門檻；訓練長度 8K vs 32K；Golden Eviction vs 啟發式的 loss | [FKV] p7–8；p17–18 |
| 重複與統計 | 〔原文〕32 次取平均；沒有誤差棒 | [FKV] p6 |
| 程式碼／資料 | 〔原文〕公開 github.com/RUCAIBox/ForesightKV | [FKV] p1 |
| 設計理由（原文） | 〔原文〕推理資料的注意力有語意區塊、會轉移，規則難抓（p2）；被逐出後低熵 token 的 loss 暴增（數學 +147%，p3 Table 1），所以 reward 盯低熵 token；Top-K multinomial 兼顧穩定與探索（p4）；監督學習無法處理逐出造成的分布偏移，所以加 RL（p5） | [FKV] p2–5 |
| 設計理由〔判讀〕 | 〔判讀〕Golden Eviction 只用原模型完整軌跡的注意力，便宜但有「訓練時沒逐出、推論時有逐出」的偏移，RL 階段是在補這個洞 | — |
| 原文沒講清楚的地方 | 〔原文〕STILL 軌跡筆數；scoring model 參數量（只寫中間層 16、輸入 6g+2D 維）；吞吐的單位與引擎。〔判讀〕STILL 與 AIME 是否重疊（污染）沒討論 | [FKV] p6；p15–16 |
| 與既有整理不一致 | 〔原文〕sota 表S1「先監督學習、再 GRPO 強化學習」已核對一致。可補訓練成本：R1-Distill-Qwen-7B（32K）監督階段 8 小時 × 2 張 H800、RL 階段 6 小時 × 8 張 H800，合計約 64 H800 小時〔計算〕（[FKV] p16 E.1）。intro／sota「深度學習派訓練很重」對 ForesightKV 有數字支持。〔複核補充〕這個數字只對 R1-Distill-Qwen-7B（32K）這一組；「很重」是相對說法，原文只報 wall-clock，沒有自評輕重。WE 第 84 行一致；補充：A800 只用於吞吐，訓練用 H800；v2 附錄另有 MiniCPM、Qwen3-14B、GPQA、LiveCodeBench、LongBench | [sota] 行 262–265；[WE] 行 84 |
| 對本研究的意義〔判讀〕 | 〔判讀〕「用最差影響的 token（低熵、loss 暴增）當 reward」的想法，可以對應到使用者「依位置的重算成本」：損失要以實際後果加權，不是以平均準確度加權。不可比：單請求、有損、純品質 | — |

**學習設定**

| 項目 | 內容 | 出處 |
|:--|:--|:--|
| 預測目標與 label | 〔原文〕每個 KV 的重要度分數。監督 label 來自 Golden Eviction：在原模型完整軌跡上算注意力，沿 query 方向每 L 個 token 分一塊、跨 head 組取平均，「未來各塊的最大值」當 future score；以 pairwise ranking loss 訓練。RL 階段 reward＝在熵最低 80% 且 loss 增加超過 η=1.5 的 token 上，loss 增量平方和取負（〔複核補充〕p6 文字寫「average square」，式 (8) 寫成加總，原文自身不一致） | [FKV] p5–6；p16 |
| 特徵 | 〔原文〕k、v（各 D 維）加 6g 維注意力特徵（最近 8／16／32／L 個 token 窗、累積和、每塊衰減 0.9 的累積和） | [FKV] p15 B.1 |
| 模型種類與大小 | 〔原文〕每個 attention group 一個 2 層 MLP，中間層 16 | [FKV] p6；p15 |
| 訓練資料 | 〔原文〕Qwen3-4B 生成的 STILL 推理軌跡（答對、>4,096 token）；每個 LLM 在同一份資料上各自獨立訓練 | [FKV] p16 |
| 切分 | 〔原文〕訓練（STILL）與評測（AIME／GPQA／LiveCodeBench／LongBench）是不同資料集 | [FKV] p6；p8 |
| 訓練成本 | 〔原文〕監督 1,000 步、batch 8；RL 200 步、batch 32、每個樣本 8 條軌跡；R1-Distill-Qwen-7B（32K）：8h×2 H800＋6h×8 H800。〔複核補充〕訓練成本只報了 R1-Distill-Qwen-7B（32K）這一組，Qwen3-1.7B／4B 沒報；全程 LLM 凍結，只更新 scoring model，LLM 不做 backward（p2、p4） | [FKV] p16 E.1；p2；p4 |
| 推論開銷怎麼量 | 〔原文〕Qwen3-4B、2K 預算、生成 32K、batch 64：總共 5,813 s，逐出 157 s（2.7%） | [FKV] p19 E.9 |
| 換模型／負載／硬體 | 〔原文〕每個 LLM 一組 scoring model；測了跨領域與跨長度（〔複核修正〕原寫「8K 訓、32K 測」不精確：Table 12 是「訓練資料截到最長 8K」vs「最長 32K」，兩者都在同一 AIME2024、1K 預算設定下測，分別得 52.1、51.7；測試的生成長度原文沒寫） | [FKV] p16；p18 E.5 Table 12 |
| fallback／保證 | 〔原文〕沒有保證；只在預測分數最低的 2L 個候選中取樣，最近 L 個永遠保留 | [FKV] p3–4 |
| 與 Oracle／Belady | 〔原文〕Golden Eviction 當參考：Qwen3-4B 上 LM loss ratio 1.017–1.072，H2O／SnapKV／R-KV 為 1.09–1.48（Table 2），14B／32B 趨勢相同（Table 15）；但沒用 Golden Eviction 跑 AIME 準確度 | [FKV] p5；p19 |

---

### LookaheadKV：Fast and Accurate KV Cache Eviction by Glimpsing into the Future without Generation（ICLR 2026；arXiv 2603.10899）

- **讀了什麼**：〔全文〕arXiv v1（2026-03-11，ICLR 2026 版面），https://arxiv.org/pdf/2603.10899v1 ，25 頁含附錄 A–H，查證 2026-10-06。程式碼 repo 存在，沒讀。
- **一句話**：加 32 個可學的 lookahead token，加上只對它們生效的 LoRA，在 prefill 時預測回應會看哪些 prompt token。
- **評測要證明的主張**：比 draft-based 方法（LAQ、SpecKV）準又快；32K 時逐出開銷 <2.16%，比 LAQ 低 14.5×。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | 〔原文〕LLaMA3.2-1B／3B、LLaMA3.1-8B、Qwen3-1.7B／4B／8B（皆 Instruct）；理論估算假設權重與 activation 都是 half precision | [LKV] p6；p16 App B |
| 硬體 | 〔原文〕理論估算模擬單張 H100 80 GB、batch 1。〔判讀〕作者說理論設定是為了對齊實測，推測實測也是 H100，但原文沒明寫。〔原文〕訓練硬體與時間沒報；作者說算力有限，沒做更大的模型 | [LKV] p16；p10 |
| 軟體與版本 | 〔原文〕實作在 KVCache-Factory 上；SpecKV 用官方碼；LAQ 沒有官方碼，自己重做 | [LKV] p8；p25 App F–G |
| 資料／負載 | 〔原文〕訓練：ChatQA2 long-SFT 50K、Tulu-3 20K、The Stack 7K、MetaMath／HellaSwag／ARC few-shot 各 3K（共 9K）。指令資料去掉最後一個 assistant 回應、預訓練文字隨機截斷，再由目標模型 greedy 生成 ≤512 token 當 Y。評測：LongBench 16 個英文任務、RULER 13 個子任務（4K–32K；附錄 64K／128K 每任務隨機 50 例）、LongProc HTML→TSV（12K→0.5K、23K→2K）、MT-Bench（Qwen3-235B-A22B 當評審） | [LKV] p6–7；p18；p25 |
| 長度 | 〔原文〕訓練輸入 ≤16K、回應 ≤512；評測 4K–32K，附錄到 128K | [LKV] p6；p18 |
| 到達與併發 | 〔原文〕無 | — |
| 重用結構 | 〔原文〕MT-Bench 是多輪 | [LKV] p8 |
| 掃描的自變數 | 〔原文〕cache budget 64–2048；context 長度；溫度；lookahead 大小 4–128 × LoRA 位置 | [LKV] p7；p9 |
| 對手 | 〔原文〕SnapKV、PyramidKV、StreamingLLM、LAQ、SpecKV（8B 級；draft 模型分別是 Llama3.2-1B、Qwen3-1.7B）。觀察窗 32、maxpool kernel 7；LAQ、SpecKV 的 draft 生成上限設為 32 token，與 lookahead 數相同 | [LKV] p7；p25 App F |
| 系統指標 | 〔原文〕TTFT 與 TTFT overhead：理論值（依 Davies et al. 的 FLOPs／記憶體流量模型，flops 效率 0.7、記憶體效率 0.9）與實測值 | [LKV] p8；p16 |
| 品質指標 | 〔原文〕LongBench 平均、RULER 平均、LongProc F1、MT-Bench 分數 | [LKV] p7–8 |
| 主要結果 | 〔原文〕所有模型與預算下 LongBench 平均最高；Llama3.1-8B、32K、budget 128：TTFT overhead 理論 1.74 ms、實測 38 ms，LAQ 554 ms、SpecKV 503 ms；RULER 128K：54.83，SnapKV 30.56、LAQ 50.67、FullKV 73.72 | [LKV] p6–8；p18 Table 6；p24 Table 15 |
| 消融／敏感度／開銷 | 〔原文〕lookahead 大小 × LoRA 位置；溫度 0.2／0.8；訓練長度 2K／4K／8K；用資料集原回應代替模型生成；加 SnapKV 的後綴窗 | [LKV] p9；p17–18 |
| 重複與統計 | 〔原文〕沒說重複；沒有誤差棒 | — |
| 程式碼／資料 | 〔原文〕公開 github.com/SamsungLabs/LookaheadKV | [LKV] p1 |
| 設計理由（原文） | 〔原文〕用模型回應估重要度比用 prompt 後綴準，但 draft 生成很慢；改用學到的 token 代替 draft 回應（p2–4）。LoRA 只作用在 lookahead token，原模型輸出不變、模組可開關（p5）。對 L1 正規化分數算 KL，等同 ListNet（p5）。lookahead=32 是增益飽和點（p9） | [LKV] p2–5；p9 |
| 設計理由〔判讀〕 | 〔判讀〕本質是把「讀回應才知道的未來」蒸餾成 prefill 時就能算的訊號；和使用者「寫入時就預測讀取時需求」同一種思路，只是對象是 token 重要度而不是重用時間 | — |
| 原文沒講清楚的地方 | 〔原文〕訓練 GPU 與時數；實測 TTFT 的硬體；重複次數；附錄說實測 overhead 大於理論值可能來自量測雜訊與實作效率（p24） | [LKV] p24 |
| 與既有整理不一致 | 〔原文〕sota 表S1「訓練 lookahead token 與 LoRA」一致。可補：可訓參數 5.4M–21.5M（<0.5%）、7,600 iters、batch 32、約 86K 樣本（[LKV] p6 Table 1、p25 Table 16）。訓練成本原文沒報，intro 表9／§5.4 若把它也算進「多張資料中心級 GPU」只是推論。WE 第 85 行一致；硬體欄可從「未查證」改成「理論估算模擬 H100；訓練硬體未報」 | [intro] 行 431–447；[sota] 行 266–269；[WE] 行 85 |
| 對本研究的意義〔判讀〕 | 〔判讀〕可以沿用：同時報「理論開銷」與「實測開銷」，並明說兩者差異來源。不可比：單請求、只在 prefill 逐出一次、有損 | [LKV] p8、p16、p24 |

**學習設定**

| 項目 | 內容 | 出處 |
|:--|:--|:--|
| 預測目標與 label | 〔原文〕每層每 head 的 prompt token 重要度分布。label（GT importance）＝目標模型實際回應 Y 對 prompt X 的平均 cross-attention | [LKV] p3；p5 |
| 特徵 | 〔原文〕lookahead token 經 LoRA 後的 query 對 prompt key 的注意力 | [LKV] p5 |
| 模型種類與大小 | 〔原文〕32 個可學 embedding 加所有線性層的 LoRA（r=8、α=32）；5.4M–21.5M 參數 | [LKV] p6 Table 1 |
| 訓練資料 | 〔原文〕約 86K 樣本（見上）；KL 散度，跨層跨 head 平均；Adam、batch 32、7,600 iters；LR 1×10^-3（Llama）／2×10^-4（Qwen），LR 有在 4 個值中搜尋 | [LKV] p5–6；p25 Table 16 |
| 切分 | 〔原文〕訓練資料與評測 benchmark 不同；訓練 ≤16K，評到 32K／128K | [LKV] p6；p9；p18 |
| 訓練成本 | 〔原文〕沒報 | — |
| 推論開銷怎麼量 | 〔原文〕只在 prefill；32K 時 <2.16%；decode 無額外開銷 | [LKV] p3；p4 |
| 換模型／負載／硬體 | 〔原文〕每個模型各自訓練；用資料集原回應代替模型生成，低預算時略差（App D） | [LKV] p6；p17 |
| fallback／保證 | 〔原文〕沒有；但模組只作用於 lookahead token，可以整組關掉 | [LKV] p5 |
| 與 Oracle／Belady | 〔原文：未見〕沒有用 GT 重要度逐出當上界；只在 Table 8 比較不同溫度下 GT 分數的相似度（recall@512 91–95%） | [LKV] p18–19 |

---

### TRIM-KV：Cache What Lasts: Token Retention for Memory-Bounded KV Cache in LLMs（ICLR 2026；arXiv 2512.03324）

- **讀了什麼**：〔全文〕arXiv v2（2026-03-01，ICLR 2026 版面），https://arxiv.org/pdf/2512.03324v2 ，27 頁含附錄 A–C，查證 2026-10-06。程式碼 repo 存在，沒讀。
- **一句話**：token 生成時由小 gate 給一個隨時間指數衰減的保留分數，超出預算就丟最低者。
- **評測要證明的主張**：比注意力啟發式與可學檢索（SeerAttn-R）都好，低預算尤其明顯；有時甚至勝過 full cache。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | 〔原文〕Qwen3-1.7B／4B／8B／14B、DeepSeek-R1-Distill-Qwen-7B、DeepSeek-R1-Distill-Llama-8B；長 context：Qwen3-4B-Instruct；chunked prefill：Phi3-mini-128K | [TRIM] p7；p10；p18–19 |
| 硬體 | 〔原文〕訓練 4×H100 80 GB；吞吐 1×H200 | [TRIM] p8；p16 Table 6 |
| 軟體與版本 | 〔原文〕FlexAttention 加自訂 Triton kernel（capacity loss）；Hugging Face Trainer | [TRIM] p6；p17 |
| 資料／負載 | 〔原文〕訓練：OpenR1-Math-220K（564M token）；長 context：SynthLong-32K、BookSum、Buddhi（32K–128K，10,000 步）；chunked prefill：LongAlpaca。評測：AIME24（64 個樣本平均 pass@1）、GSM8K、MATH-500（8 個樣本）、LongProc（greedy）、LongMemEval_S（最長約 123K）、SCBench、LongBench、LongBench-V2（只取 <128K） | [TRIM] p7；p17–19 |
| 長度 | 〔原文〕數學訓練最長 16,384；長 context 訓練最長 128K；吞吐測 16K／32K context、生成 1,024 | [TRIM] p17–18；p16 |
| 到達與併發 | 〔原文〕無到達過程；吞吐 batch 4／8 | [TRIM] p16 |
| 重用結構 | 〔原文〕LongMemEval 依 SCBench 的多輪多 session 協定：每個 query 前把累積對話壓成固定大小、可重用的 KV | [TRIM] p18 |
| 掃描的自變數 | 〔原文〕KV budget；訓練時的 M；gate 架構；訓練資料 | [TRIM] p8–10；p20–21 |
| 對手 | 〔原文〕SeerAttn-R（可學檢索，把 KV 卸到 host）、R-KV、SnapKV、H2O、StreamingLLM；附錄 KeyDiff、LocRet（部分數字取自 LocRet 原文） | [TRIM] p7；p17；p19 |
| 系統指標 | 〔原文〕decode 吞吐（tok/s）、decode 時間 | [TRIM] p16 Table 6 |
| 品質指標 | 〔原文〕pass@1、LongProc F1／accuracy、LongMemEval 準確率（由 Qwen3-4B-Instruct 判定）、SCBench、LongBench | [TRIM] p7；p18 |
| 主要結果 | 〔原文〕同預算下比 R-KV／SnapKV 相對 +198.4%，比 SeerAttn-R +58.9% pass@1；LongMemEval 32K 預算 44.8，Full KV 49.4、兩個啟發式約 27.6–27.8；32K context、batch 4 吞吐 130.48 tok/s，FullKV 68.44（H200） | [TRIM] p8；p10；p16 |
| 消融／敏感度／開銷 | 〔原文〕loss 組合（拿掉 L_cap 掉到 42.9）、訓練資料（數學 vs 一般）、gate 架構（線性 vs MLP、初始 bias）、M | [TRIM] p10 Table 5；p20–21 |
| 重複與統計 | 〔原文〕AIME 取 64 個樣本、GSM8K／MATH 取 8 個樣本平均；沒有誤差棒 | [TRIM] p7 |
| 程式碼／資料 | 〔原文〕公開 github.com/ngocbh/trimkv | [TRIM] p1 |
| 設計理由（原文） | 〔原文〕注意力是短視的代理，逐出是長期決策，應依 token 生成當下的內在重要度（p5）；用指數衰減取代 sigmoid，避免梯度消失、也不需要知道序列長度（p4）；所有 gate 端到端一起訓，避免逐層貪婪（p5）；初始 bias 設大（18），讓訓練初期幾乎不遺忘（p8、p21） | [TRIM] p4–5；p8；p21 |
| 設計理由〔判讀〕 | 〔判讀〕分數只在 token 產生時算一次，之後只做次方衰減，開銷是 O(M)；這是「寫入時打分、之後不重算」的範例，與使用者的寫入時決策同構 | [TRIM] p16 A.2 |
| 原文沒講清楚的地方 | 〔原文〕數學訓練的步數與時數；LongMemEval 由同一家族的 Qwen3-4B-Instruct 判定；SCBench 的子集與長度；以 M=256 訓練、推論時用其他預算的對應方式。前後不一致：LongMemEval 32K 預算的兩個啟發式，Table 3（p10）寫 StreamingLLM 27.6、SnapKV 27.8，Table 8（p19）寫 27.8、27.6 | [TRIM] p10；p18–19 |
| 與既有整理不一致 | 〔原文〕sota 表S1「凍結 LLM、只訓練 retention gate；4 張 H100，最長訓練到 128K」一致（[TRIM] p6、p8、p18）；intro §5.4「TRIM-KV 用 4 張 H100」一致，但原文沒有訓練時數，「訓練成本高」只能說到「需要 4 張 H100」。intro 參考文獻 [36]、sota [24] 列為 arXiv 2025，原文 v2 已標 ICLR 2026。WE 第 86 行一致；可補 Phi3-mini-128K／LocRet、吞吐硬體 H200。〔複核修正〕WE 第 86 行模型欄的「GQA」TRIM-KV 全文沒寫（全文檢索無 GQA），應標〔未查證〕 | [intro] 行 447、462、1074；[sota] 行 270–274、507；[WE] 行 86 |
| 對本研究的意義〔判讀〕 | 〔判讀〕可以沿用：「寫入時給分、之後只衰減」的資料結構，與 LongMemEval／SCBench 的多 session 協定（最接近跨請求情境的 L1 評測）。不可比：有損，品質依模型與資料 | [TRIM] p18 |

**學習設定**

| 項目 | 內容 | 出處 |
|:--|:--|:--|
| 預測目標與 label | 〔原文〕每個 token、每個 KV head 的保留分數 β∈[0,1]，有效分數為 β^(t−i)。沒有顯式 label：以 retention-gated attention 模擬逐出，目標＝對原模型的 forward KL＋next-token loss＋容量 hinge loss（λ_cap=1；M=256、1024 或 512，依實驗） | [TRIM] p4–6；p7；p18–19 |
| 特徵 | 〔原文〕token 在該層的 hidden x_t | [TRIM] p5 |
| 模型種類與大小 | 〔原文〕每層一個 MLP gate：d→512→h（h＝KV head 數），sigmoid 輸出 | [TRIM] p8 |
| 訓練資料 | 〔原文〕只更新 gate；LR 2×10^-4、weight decay 0.01、每 GPU batch 1、gradient accumulation 4 | [TRIM] p17 |
| 切分 | 〔原文〕訓練資料與評測 benchmark 不同；另做「一般資料訓練、數學評測」的交叉測試 | [TRIM] p20 |
| 訓練成本 | 〔原文〕4×H100；時數沒報 | [TRIM] p8；p17 |
| 推論開銷怎麼量 | 〔原文〕每個 token 多存 1 個 scalar（約 1/d_h）；逐出 O(M)；gate 可與 QKV 投影融合 | [TRIM] p16 A.2 |
| 換模型／負載／硬體 | 〔原文〕每個模型各訓一組 gate；建議 M 對應部署時的預算 | [TRIM] p21 |
| fallback／保證 | 〔原文：未見〕無 | — |
| 與 Oracle／Belady | 〔原文：未見〕沒有 oracle；只與 Full KV 比（有時超越） | [TRIM] p8 |

---

## 使用者文件核對彙整

逐條對照 intro §5.3–§5.4、摘要第 4 點，以及 sota §3.1–§3.2。逐篇細節見各卡「與既有整理不一致」。

| 使用者文件的說法 | 位置 | 判定 | 依據 | 複核（V06，2026-10-07） |
|:--|:--|:--|:--|:--|
| LPC 要訓練 118M 參數的文字嵌入模型 | [intro] 行 447–448；[sota] 行 245–248 | ❌ 錯 | e5-small 凍結，只訓 3 層 MLP，<10 分鐘〔原文〕[LPC] p4–5；錯誤來自 SAECache 的轉述 [SAE] p11 | ✅ 指控成立。LPC p4–5 明說只更新 MLP、e5-small 凍結；SAECache p11 確實寫 LPC「trains a 118M-parameter text embedding model」。使用者文件已註明是轉述，錯在 SAECache 這個源頭 |
| LPC 屬「深度學習：訓練成本高、重訓原因是更換 LLM」 | [intro] 表9 行 431–444 | ❌ 不符 | 訓練 <10 分鐘；建議每天重訓以追行為漂移；不讀 LLM 內部狀態〔原文〕[LPC] p5、p10 | ✅ 指控成立（限「訓練成本」「重訓原因」兩欄）。LPC 的訓練硬體原文沒寫，「多張資料中心級 GPU」無依據；重訓理由是使用者行為漂移（p5）。但表9「執行位置 GPU」「輸入特徵 文字內容」兩欄對 LPC 成立（predictor 佔 1 GB GPU 記憶體，p6；輸入是 user prompt，p4） |
| 深度學習派確實訓練重、而且綁定 LLM | [intro] 行 32–33、表10 行 461–465；[sota] 行 285–288 | ⚠️ 只對 L1 四篇成立 | KVP、ForesightKV、LookaheadKV、TRIM-KV 都依各自 LLM 訓練〔原文〕；有報成本的：KVP 收 trace <112 GPU 小時＋訓練 <4 H100 小時、ForesightKV 約 64 H800 小時〔計算〕；TRIM-KV 只報 4×H100、LookaheadKV 沒報。LPC 不成立 | ✅ 指控成立（LPC 是反例）。〔複核補充〕即使只看 L1 四篇，「綁定 LLM」成立，「訓練重」只能說到「要多張資料中心級 GPU，加上目標 LLM 的 forward（LookaheadKV、TRIM-KV 還要穿過 LLM 做 backward）」：KVP 作者自稱 low computational footprint，且「30 分鐘」沒說是否涵蓋全部 112 個 agent（[KVP] p16–17）；ForesightKV 的 64 H800 小時只對 R1-Distill-Qwen-7B 32K 一組（[FKV] p16）；TRIM-KV、LookaheadKV 沒有時數 |
| KVP：56 張 GPU 收約 1.2 TB trace，8 張 H100 訓練；每 (層, KV head) 一個約 65 萬參數 agent，Qwen2.5-7B 112 個 | [intro] 行 447、461、464；[sota] 行 251–260 | ✅ 一致 | [KVP] p16–17；56 張 GPU 的型號原文沒寫 | ✅ 一致屬實（p16–17 逐字核對）。sota 表S2「KVP 要多張資料中心級 GPU」靠 8×H100 訓練這一段成立；收 trace 的 56 張型號未寫 |
| TRIM-KV：4 張 H100、凍結 LLM、最長訓到 128K | [intro] 行 447、462；[sota] 行 270–274 | ✅ 一致 | [TRIM] p6、p8、p18；訓練時數沒報 | ✅ 一致屬實（[TRIM] p6、p8、p17–18） |
| SAECache 屬 GBDT／輕量、CPU、預測下次存取時間；不需要定期重訓 | [intro] 表9 行 431–446 | ⚠️ 不精確 | 沒有 GBDT；統計參數線上更新，但另有讀 LLM hidden state 的離線訓練 MLP〔原文〕[SAE] p6–8、p19 | ✅ 指控成立。全文沒有 GBDT／LightGBM，也沒寫在 CPU 執行；首輪分類器讀服務中 LLM 的 hidden state（p7–8），但 p19–20 的特徵描述與此矛盾。「預測下次存取時間」可大致對應 log-normal 存活機率（p6–7），這一點不算錯；「不需要定期重訓」是 SAECache 自述（p11），對統計參數成立，對 MLP 原文沒討論 |
| LCR／LARU 預測「逐出優先順序」 | [sota] 行 218 | ⚠️ 措辭可補充（〔複核修正〕原判「不精確」過重） | 預測的是下次請求時間〔原文〕[LARU] p5、p7 | ❌ 指控不成立（改為措辭建議）。原文 p5 §4 自己也說 LARU 向 predictor 取得「eviction priorities」；只是模型實際輸出的是下次請求時間（p5 Alg.1 第 16 行、p7）。補一句「下次請求時間，當逐出優先順序用」即可，不算錯誤 |
| LARU 偵測到預測錯就退回 LRU，1-consistent、O(k)-robust | [intro] 行 494；[sota] 行 218–223 | ✅ 一致（細節：逐步縮小候選集，不是整個切換） | [LARU] p4–5 | ✅ 一致屬實。原文 p5 也說「safe fallback to near LRU performance」；細節是每偵測一次做一次 LRU 逐出並把 λ 減半（Alg.1 第 10–12 行） |
| 多數系統盲從預測（FPB），預測錯時比 LRU 差 | [intro] 行 448–449、473 | ✅ 一致 | [LARU] p2、p8、p15–16 Table 1 | ✅ 一致屬實（p15 寫 FPB「can even become worse than classical heuristics」；p8 Fig.5–6） |
| GBDT 派在 CPU 上線上訓練，LARU 線上訓練 LightGBM | [intro] 行 467；[sota] 行 290 | ✅ 一致 | 每 10^3 筆重訓、窗口 10^5、<100 ms〔原文〕[LARU] p6–7；LRB 不在本組範圍 | ✅ 一致屬實（p6 §4.1、p7） |
| 換 server 就要重訓：視預測對象 | [intro] 行 470；[sota] 行 292 | ⚠️ 本組沒有任何一篇測過跨硬體 | 七篇都沒做換硬體實驗〔原文：未見〕。〔判讀〕在閉迴路重播裡，「下次請求時間」的 label 含伺服器回應時間（LPC 的到達模型就是「上一回應完成時間＋think time」，[LPC] p7），所以「只學負載」的模型若預測 wall-clock 回來時間，label 本身會隨硬體漂移 | ✅「七篇都沒做換硬體實驗」屬實（各篇只在各自的一兩種硬體上跑）。這一條不推翻使用者「視預測對象」的結論：L1 四篇的 label 是未來注意力或對原模型的 KL／loss，不含時間量，換硬體不必重訓〔判讀〕，反而支持使用者的區分。〔複核補充〕label 隨硬體漂移的判讀另有一處原文支撐：SAECache 對 Bailian 的輪間間隔定義為「子請求時間戳 − 父請求時間戳」（[SAE] p16），含父請求的服務時間〔判讀〕 |
| 預測派大多只比預測準不準，很少報預測錯的時候 | [intro] 行 453–454 | ✅ 大致成立 | LARU 是例外〔原文〕[LARU] p8–9 | ✅ 大致成立。〔複核補充〕SAECache 的固定參數錯配（p5–6）、LPC 的 unified vs specialized（p22）是分布外測試，不是「預測錯時決策會怎樣」；這類測試只有 LARU 做（p8–9） |
| 預測派預測的是「會不會再用」，沒人預測「回來時 GPU 多忙」 | [intro] 行 372 | ⚠️ 前半不完整 | LARU 預測下次請求時間、SAECache 擬合輪間間隔分布，都是在預測「何時回來」〔原文〕[LARU] p7、[SAE] p6；後半成立：本組七篇沒有預測讀取時 GPU 負載〔原文：未見〕。intro 表13「多久之後回來」只引 Bidaw，可補 LARU 與 SAECache | ✅ 指控成立（前半）。LARU Alg.1 第 16 行預測 reuse interval（p5）；SAECache 以 log-normal 建模輪間間隔、算存活機率（p6–7）。後半「沒人預測回來時 GPU 多忙」在本組七篇屬實 |
| sota 表S1 LPC「保底：未查證」 | [sota] 行 245 | 可更正為「無」 | [LPC] p5–6 | ✅ 指控成立。LPC 只有機率的指數衰減（p5–6 §3.4.2），沒有退回機制或保證 |
| sota 表S6 LPC 改的是 vLLM 0.7.3 | [sota] 行 446 | ✅ 一致 | README（論文寫 2025-03-10 main branch） | ✅ 一致屬實（README 的 VLLM_PRECOMPILED_WHEEL 是 vllm-0.7.3；複核者 2026-10-07 重讀） |
| GBDT「微秒級推論」 | [intro] 行 722 | ⚠️ 只有批次攤提的依據 | 10^4 候選 <10 ms，即每候選 <1 µs〔計算〕[LARU] p7；單次延遲沒報 | ✅ 指控成立（部分）。原文 p6 §4.1 把「microsecond-level inference」列為需求，並說 LightGBM 推論快，但只量了批次（p7） |
| TRIM-KV 參考文獻列為 arXiv 2025 | [intro] 行 1074；[sota] 行 507 | ⚠️ 可更新 | v2 已標 ICLR 2026〔原文〕[TRIM] p1 | ✅ 指控成立（屬「可更新」，不是錯：v1 為 2025-12，列 arXiv 2025 當時無誤；v2 首頁標 ICLR 2026） |

---

## 本組對 PoC 設計的建議〔判讀〕

前提：使用者要做的是 GBDT 類輕量預測器，在寫入時預測三個量：會不會再用、多久之後回來、回來時 GPU 多忙；硬體成本用量測常數。以下全部是〔判讀〕，引用的原文只當依據。

1. **資料切分：依時間切，加上依 session 分組，不要隨機切逐輪樣本。**
   - 用有真實時間戳的 trace（Bailian、Mooncake、TraceLab 一類）；ShareGPT／LMSys 沒有時間戳，只能像 LPC 那樣切對話〔原文〕[LPC] p4、p7。
   - 主評估用 rolling-origin：前段時間訓練，後段時間測試，中間留緩衝，避開跨邊界的 session。
   - 同一個 session 的所有樣本只能在一邊。LPC 程式碼對「逐輪樣本」隨機切驗證集，同一對話會跨兩邊（雖然只影響 checkpoint 選擇）〔程式碼〕[LPC-code]，這正是審稿人會抓的洩漏型態。
   - 另做跨 trace 測試（在 A 訓、在 B 測），對應 LPC 的 unified vs specialized〔原文〕[LPC] p22。
2. **防洩漏的四個檢查點。**
   - (a) 特徵只能用寫入當下（prefill 完成時）已知的資訊；不能用 session 總輪數、最終長度、未來的工具呼叫。LPC 用「已發生輪數」是合法的〔原文〕[LPC] p4。
   - (b) label 有截尾：trace 結尾前的請求看不到未來，要嘛排除最後一段，要嘛當作 censored；LARU 沒說它怎麼處理〔原文：未見〕。
   - (c) 超參數（例如 LPC 的 decay scale 取「平均輪間隔」）只能從訓練段估〔原文〕[LPC] p6。
   - (d) label 的時間單位：用 think time（上一回應完成到下一請求）或開迴路時間戳，不要用閉迴路重播的 wall-clock 間隔。後者含伺服器回應時間，會讓「只學負載」的模型偷偷學到硬體〔原文〕[LPC] p7 的到達模型。
3. **與 Oracle 比要分三層，報「縮小了多少差距」。**
   - 依序比較：規則（LRU 或 Cake 式規則）、規則＋學到的預測、規則＋完美預測（把真實 label 餵給同一個策略，LPC 式 Oracle）、同一多層成本模型下的離線最佳（專案現有的 M4 Oracle）。
   - 報 (學到的 − 規則)／(離線最佳 − 規則) 與 (完美預測 − 規則)／(離線最佳 − 規則)。前者是「預測派貢獻」，後者分出「預測誤差」與「策略本身不夠好」〔原文：LPC 報縮小 LRU→Oracle 差距的比例，[LPC] p23〕。本組沒有一篇在主結果報同一成本模型下的 OPT，這一格是可以佔的。
4. **對手要含「學習式＋保底」那一格。** 用同一個預測器跑 FPB、HF（LRU 預篩 4 個）、LARU 式 λ 縮放，以及使用者自己的保底；這是 LARU 的公平做法〔原文〕[LARU] p7。只比 LRU 會被打（LPC 自承這是限制〔原文〕[LPC] p10）。
5. **壓力測試照 LARU 與 SAECache 的做法。**
   - 合成雜訊：以機率 p 把預測換成最壞值，p 從 0 掃到 1〔原文〕[LARU] p8。
   - 弱預測器：訓練窗縮到 0.1%〔原文〕[LARU] p8。
   - 負載錯配：〔複核修正〕SAECache 的做法是「用 chat 主導的 trace 擬合參數，套到單輪主導的負載」（原寫「chat 訓、agent 測」）〔原文〕[SAE] p5–6、p15；chat 訓、agent 測是可以加做的延伸〔判讀〕。
   - 跨硬體：在 A 機器訓練，B 機器只換量測常數（這就是 H3，本組七篇都沒做過）。
6. **開銷要量到長 context 的決策數，而且要扣資源。**
   - 量每個決策在 CPU 上的 p50／p99 延遲（單次與批次分開），在 500K、每 512 token 一個 chunk（約 1,000 個決策，intro 的算術）的情況下報總和。
   - 在卸載 I/O 執行緒同時佔 CPU 的情況下量。
   - 報每次重訓的 CPU 時間與特徵存放的記憶體（LARU 的 O(k)）。
   - 若預測器佔 GPU，就從快取預算扣掉再比〔原文〕[LPC] p7；端到端 TTFT 要含預測器（KVP 自承只報 policy 層級的時間〔原文〕[KVP] p11）。
7. **報校準，不只報準確度。** 使用者的設計是「成本變了只移動門檻」，這依賴機率校準；本組七篇沒有一篇報 calibration〔原文：未見〕，LPC 只報 MCC／F1〔原文〕[LPC] p21。至少報 reliability diagram、Brier score，以及換成本常數後門檻移動造成的決策品質變化。
8. **統計：** 每點至少 5 次、報 min–max 或信賴區間〔原文：LPC 的做法，[LPC] p23〕；切分用的種子也要報。
9. **「回來時 GPU 多忙」要用硬體中立的單位預測。** 預測佇列中的 token 數、活躍請求數，再用量測常數換成時間；直接預測「讀取時延遲」會把硬體學進模型，違反 H3。本組沒有前例〔原文：未見〕。

---

## 未查證清單

| 項目 | 原因 |
|:--|:--|
| LARU 的公開程式碼 | 原文說在補充材料，搜尋沒找到公開 repo |
| LARU：Online-QA 的長度與到達方式、label 截尾處理、4090 上 32B 的部署方式、重複次數、DeepSeek-R1 那台「H800 140 GB」是否筆誤 | 原文沒寫 |
| LPC 的 OpenReview 版是否與 proceedings 版相同 | OpenReview 驗證頁擋住；讀的是 proceedings 版 |
| LPC 論文數字是否就是 repo 中 `load_conv_data`／`prepare_dataloaders` 的設定（尾端 N 個對話訓練、逐輪隨機切驗證） | 只讀了腳本，沒重跑；腳本 main 區塊的參數（N、test_size、lr）可能不是論文設定 |
| LPC MLP 的訓練硬體；預測吞吐「1,000 次／秒」的量測條件（SAECache 量到的 LPC 只有 129 次／秒，條件也沒寫） | 兩篇都沒寫 |
| SAECache 主實驗的模型、硬體、是否真實執行、TTFT 怎麼得到；session predictor 的輸入（hidden state 或手工特徵）、訓練資料與切分；1.4–2.7× 的來源；LPC 是否用官方碼重跑；程式碼 | 原文沒寫或前後矛盾 |
| KVP：收 trace 的 GPU 型號、準確度評測硬體、chunked prefill 的 chunk 大小、RULER-4k 的切分 | 原文沒寫 |
| ForesightKV：STILL 軌跡筆數、scoring model 參數量、推論引擎、吞吐單位、STILL 與 AIME 是否重疊 | 原文沒寫 |
| LookaheadKV：訓練硬體與時數、實測 TTFT 的硬體 | 原文沒寫（實測硬體只能推測是 H100） |
| TRIM-KV：訓練時數與步數（數學）、SCBench 子集與長度 | 原文沒寫 |
| 各篇圖表上的數值 | 只讀內文與表格；只畫在圖上、內文沒寫的數值沒抄 |
| 各模型的注意力類型（GQA／MLA）與 dtype | 多數原文沒寫；依規則不從記憶補 |
| KVP、ForesightKV、LookaheadKV、TRIM-KV 的程式碼內容 | 只查了 repo 存在與最新 commit，沒讀碼 |
| LRB、HALP、Bidaw、MTDS、CacheFlow、KVPR 在使用者文件中的描述 | 不在本組範圍（見 E03／E05） |

---

## 抽取紀錄

- 抽取者：E06，2026-10-06。七篇都讀全文（LPC 讀 NeurIPS proceedings 版＋官方程式碼的三個檔案）。
- 複核：V06，2026-10-07，見下方「## 複核紀錄」。

---

## 複核紀錄

- **複核者**：V06（子 agent，沒有參與抽取，沒讀抽取者的推理或筆記）。日期 2026-10-07。
- **用的原文**：抽取者下載到 `scratchpad/E06/` 的七份 PDF。複核者先確認每份 PDF 首頁的 arXiv 編號與版本（LARU 2509.20979v2、SAECache 2605.18825v1、KVP 2602.10238v2、ForesightKV 2602.03203v2、LookaheadKV 2603.10899v1、TRIM-KV 2512.03324v2；LPC 為 NeurIPS 2025 proceedings 版，首頁頁尾標 NeurIPS 2025），再自己用 `pdftotext -layout` 轉成帶換頁符號的文字，放在 `scratchpad/V06/`，頁碼依此。2026-10-07 查 arXiv abs 頁，六篇讀的都是最新版本；LARU v1（2025-09-25）摘要確實也是 24.2%／28.3%。LPC 程式碼由複核者重新從 GitHub 下載：main 仍是 3e750fa，三個檔案與抽取者的副本逐位元相同；repo 根目錄沒有 license 檔，README 的 wheel 是 vllm-0.7.3。四個 L1 repo 的存在、最新 commit 與 license 也用 GitHub API 重查過，與來源清單一致。
- **做法**：逐格回到原文。優先順序是數字，其次設定、定義、「設計理由（原文）」；〔判讀〕格只檢查有沒有被寫成事實。

### 每張卡檢查的格數與結果

每張卡共 33 格：抬頭 3 格（讀了什麼、一句話、主張）、主表 20 格、學習設定 10 格。

| 卡 | 檢查格數 | ✅ | ❌（已改） | ⚠️（改標〔判讀〕或加註） | 含〔複核補充〕的格（與前三欄重疊計） |
|:--|:--|:--|:--|:--|:--|
| LARU | 33 | 32 | 1 | 0 | 3 |
| LPC | 33 | 31 | 2 | 0 | 2 |
| SAECache | 33 | 31 | 2 | 0 | 1 |
| KVP | 33 | 30 | 1 | 2 | 2 |
| ForesightKV | 33 | 31 | 2 | 0 | 3 |
| LookaheadKV | 33 | 33 | 0 | 0 | 0 |
| TRIM-KV | 33 | 32 | 1 | 0 | 0 |
| 共同模式（6 條） | 6 | 4 | 1 | 1 | 1 |
| 使用者文件核對彙整（17 列） | 17 | 16 | 1 | 0 | 已加「複核」欄 |
| PoC 建議中的事實引用 | 約 15 處 | 約 14 | 1 | 0 | — |
| 來源清單（10 列＋repo 一行） | 11 | 11 | 0 | 0 | — |
| **合計** | **約 280** | — | **12** | **3** | — |

〔複核補充〕指的是抽取者沒寫錯，但原文本身前後不一致，或漏了對評測設定有影響的事實，由複核者補上；同一格可能同時有 ❌ 與補充。LARU 與彙整表的「逐出優先順序」是同一條指控，兩邊各計一次。

### 逐條修改（原內容 → 新內容＋出處）

**❌ 錯誤，已改正**

1. 使用者文件核對彙整「LCR／LARU 預測『逐出優先順序』」，以及 LARU 卡「與既有整理不一致」(1)：原判 ⚠️「不精確」，等於指控 sota 表S1 寫錯 → 改為「措辭可補充，不算錯」。依據：[LARU] p5 §4 原文自己說 LARU 向 predictor 取得「eviction priorities」；模型輸出的量是下次請求時間（p5 Alg.1 第 16 行、p7）。**這是本卡唯一一條站不住的指控。**
2. LPC「原文沒講清楚」：「`get_predictor_accuracy.py` 在前 20,000 個對話上評估」→ 計數器 `processed_count` 每處理一輪就加 1，所以是讀到約 20,000 個**輪樣本**為止，再以 `test_size=0.5` 對逐輪樣本隨機切，前半挑 MCC 最佳門檻、後半報分數（`get_predictor_accuracy.py` L118–170、L551–580、L643–675）。同格另補 `learn_conversation.py` main 區塊 `lr=5e-5`，論文寫 5×10^-4（L490 vs [LPC] p5）。
3. LPC「原文沒講清楚」：「摘要寫 74%」→ 74% 在**引言** p1，摘要沒有這個數字（[LPC] p1）。
4. LPC 學習設定「預測目標與 label」：「下一則訊息是不是 user」→ 「assistant 回覆之後那一則（`messages[i+2]`）是不是 user」（`learn_conversation.py` L111）。
5. SAECache 學習設定「推論開銷怎麼量」：出處 p6 → p7（常數時間的說法在 [SAE] p7 §3.1 末段）。
6. SAECache「與既有整理不一致」(5)：「WE 第 69 行其他欄位已核對一致」→ WE 第 69 行寫 Qwen2.5-1.5B「GQA」，SAECache 全文沒寫注意力類型（全文檢索無 GQA），應標〔未查證〕。
7. TRIM-KV「與既有整理不一致」：「WE 第 86 行一致」→ WE 第 86 行的「GQA」TRIM-KV 全文沒寫，應標〔未查證〕。
8. KVP「原文沒講清楚」：「Fig.7 圖說寫 single KV head，內文寫 single-layer compression」→ 兩種說法都在 Fig.7 圖說裡（[KVP] p18）。
9. ForesightKV「消融」：「6 種 reward」→ 5 種 reward（−L_all、−L_low、−L_high、−L_low,large、−L_ours），另加一欄只做監督、不做 RL 的對照（[FKV] p8 Table 4）。
10. ForesightKV 學習設定「換模型」：「8K 訓、32K 測：52.1 vs 51.7」→ 訓練資料截到最長 8K vs 最長 32K，兩者都在同一 AIME2024、1K 預算設定下測（52.1 vs 51.7）；測試的生成長度原文沒寫（[FKV] p18 E.5 Table 12）。
11. 共同模式 #1：「LPC 與 SAECache 用平均（TTFT）」→ SAECache 是 mean TTFT（[SAE] p8）；LPC 的 TTFT 原文沒寫統計量（[LPC] p7 定義、p9 微基準）。
12. PoC 建議 5：「負載錯配：chat 訓、agent 測〔原文〕[SAE] p5–6」→ SAECache 的做法是用 chat 主導的 trace 擬合參數，套到單輪主導的負載（[SAE] p5–6、p15）；「chat 訓、agent 測」改標〔判讀〕的延伸。

**⚠️ 原文沒有明寫，改標〔判讀〕或加註**

13. KVP 學習設定「換模型」：「Phi-4 有自己一組 agent」原文沒有明寫。Fig.13 只說把框架用在 Phi-4 14B 上，標為「KVP (Phi-4)」。因為 agent 的輸入是該模型的 K/V，必須另訓是〔判讀〕（[KVP] p24）。
14. KVP「與既有整理不一致」(c)：「訓練 <4 H100 小時〔計算〕」假設 30 分鐘涵蓋全部 112 個 agent，但原文只寫「completes in less than 30 minutes on a single node of 8 NVIDIA H100 GPUs」（主詞是 The agent training），沒說是全部還是單一 agent。已加註；也補上作者自稱 low computational footprint、收 trace 約等於一次標準推論（[KVP] p16–17）。
15. 共同模式 #4：「L1 慣例是訓練資料集 ≠ 評測 benchmark」對 KVP 只部分成立。KVP 主結果是同資料集的 test 切分，跨資料集只在零樣本實驗（[KVP] p6–8）。

**〔複核補充〕（抽取者沒寫錯，補原文中與評測設定有關的事實）**

- LARU 主要結果：Fig.3 圖內 6 個標註是 −4.6%、−11.4%、−6.5%、−6.7%、−6.0%、−10.7%。依標註位置判讀，內文的 6.0–10.7% 只對應 5 併發（[LARU] p7）。
- LARU 原文沒講清楚：p3 寫吞吐 +24.2% 在 QB-Video，p9／p17 是 AD-CTR-User；1/(1−0.195)≈1.242 支持後者〔計算〕。
- LARU 與既有整理 (3)：p6 把「microsecond-level inference」列為需求，但沒有單次呼叫的量測。
- LPC：「省 18–47% 快取」只在摘要、p2、Fig.4 圖說；內文 p8 最大約 43%，結論寫「up to 43%」。
- SAECache 與既有整理 (1)：全文沒有 GBDT，也沒寫在 CPU 執行；「離線」是推論；p19–20 的特徵描述與 p7–8 矛盾。
- ForesightKV：訓練成本只報 R1-Distill-Qwen-7B（32K）一組；LLM 全程凍結、不做 backward（[FKV] p2、p4、p16）。式 (8) 是加總，p6 文字卻寫 average。
- 共同模式 #3：各篇是否要穿過 LLM 做 backward。KVP、ForesightKV 不用；LookaheadKV、TRIM-KV 要（[KVP] p6、[FKV] p2/p4、[LKV] p5、[TRIM] p5–6）。
- 使用者文件核對彙整：新增「複核（V06）」欄，逐列標 ✅ 指控成立／❌ 指控不成立，並附依據。

### 對使用者文件的指控：複核結論

- **確認（✅ 指控成立）**：
  - LPC「要訓練 118M 嵌入模型」是錯的。錯在 SAECache p11 的轉述，LPC p4–5 明說 e5-small 凍結。
  - 把 LPC 歸入「訓練成本高、因換 LLM 而重訓」不成立：訓練 <10 分鐘，每天重訓是為了追使用者行為漂移。
  - 「深度學習派訓練重且綁定 LLM」對 LPC 不成立；對 L1 四篇，「綁定 LLM」成立，「訓練重」只能說到「要多張資料中心級 GPU」。有時數的只有 KVP（作者自稱輕量）與 ForesightKV（單一設定約 64 H800 小時）。
  - SAECache 不是 GBDT，也沒說在 CPU 上跑。
  - 「預測派只預測會不會再用」前半不完整：LARU 與 SAECache 都在預測何時回來。
  - LPC 沒有保底。
  - 「微秒級推論」只有批次攤提的依據。
  - TRIM-KV 參考文獻可更新為 ICLR 2026。
- **推翻（❌ 指控不成立）**：sota 表S1 說 LARU 預測「逐出優先順序」不算錯，原文 p5 自己就這樣稱呼 predictor 的輸出。
- **「換一台 server 就要重訓」**：本組七篇都沒做換硬體實驗，這一點屬實。抽取者的判讀沒有推翻使用者「視預測對象」的結論，反而支持它：L1 四篇的 label 不含時間量。抽取者提醒「若預測 wall-clock 回來時間，label 含伺服器回應時間」，這有 LPC p7 與 SAECache p16 的原文支撐。

### 因時間沒有檢查的部分（如實記錄）

- 沒有重新上網搜尋 LARU 是否有公開 repo，也沒有嘗試 LPC 的 OpenReview 版（沿用抽取者的〔未查證〕）。
- 只核對內文與表格的數字；只畫在圖上的數值沒有逐點核對。例外是 LARU Fig.3 的文字標註。
- 「本組對 PoC 設計的建議〔判讀〕」只核對其中引用原文的事實，判讀內容本身沒有評論。
- 「未查證清單」只確認各項在原文中確實找不到，沒有另外找二手來源補齊。
- KVP、ForesightKV、LookaheadKV、TRIM-KV 的程式碼沒有讀，與抽取者相同。LPC 的 `run_nips.py` 沒有細讀，卡片也沒有引用它的內容。
