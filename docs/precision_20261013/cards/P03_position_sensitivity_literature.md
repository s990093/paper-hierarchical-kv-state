# P03｜文獻到底支持哪一種「位置敏感」？（StreamingLLM、MuKV、Lost／Found in the Middle、KV 量化的位置規則）

> 一句話：文獻只支持兩條很窄的位置規則：「最前面極少數 token（1–32 個 sink）在低位元時要留高精度」和「最近約 128 個 token 留高精度」。兩條都只在 INT2–INT4 量過，沒有人在 FP8 量過。「整個前半段比較重要」沒有任何一篇量過。「越前面 attention 越高」有一部分是因果遮罩造成的計數假象（ZipCache 正面處理過，MuKV 沒處理）。我們要的「前／後／隨機 25% 區塊」對照，我找不到有人做過。

- **日期**：2026-10-13
- **緣由**：老師 10/13 提醒：StreamingLLM 的 sink 只是最前面少數 token，不能推成整個前半段都重要；MuKV 也用內容訊號修正 attention 的位置偏差；要先把壓前段還是後段寫成清楚的規則，再去驗。
- **讀法**：頁碼一律是 PDF 頁（`pdftotext -layout` 依換頁符號切）。每篇都標了讀的版本。數字都從原文表格或內文抄，沒有從圖上抄（唯一例外是 SKVQ 圖 6，已標〔判讀〕）。
- **標記**：〔原文 §／p.／Table／Fig〕＝原文寫的；〔程式碼 repo file:line〕＝程式碼寫的；〔判讀〕＝我的推論；〔未查證〕＝沒讀到原文。
- **複核**：本卡是 self-check（逐項回到 PDF 文字 grep 核對），**不是**獨立 reviewer 的複核。

---

## 總表

| 論文 | venue | 什麼對位置敏感 | 幾個 token／哪一段 | 量到的效果 | 證據 |
|:--|:--|:--|:--|:--|:--|
| StreamingLLM | ICLR 2024 | **驅逐**（不是量化）時，最前面幾個 token 的 KV 在不在 | 預設 4 個；Falcon-7B、Pythia-12B 留 1 個就夠，Llama-2-7B 要 4 個 | Llama-2-13B：只留最近 1024 個 PPL 5158.07；加回前 4 個 5.40；前 4 個換成 "\n" 5.60 | 〔原文 §3.1 / p.5 / Table 1、2〕 |
| MuKV | CVPR 2026 | 視訊 token **剪枝**時，最後一層 attention 偏向**區塊內**較早的 token | 區塊內順序（每塊 49／196／784 個視覺 token，各自單獨編碼）；不是長上下文的位置 | 沒有隔離「位置偏差修正」的消融。只有 Att 55.9/45.1、Freq 56.6/45.3、兩者融合 57.3/45.6 | 〔原文 p.7 Table 3、p.8〕〔程式碼 MuKV model/mukv_dualsignal_compression.py:343、:474〕 |
| Lost in the Middle | TACL 2024 | **答案所在文件**在 prompt 裡的位置 → 模型用不用得到 | 10／20／30 份文件中的第幾份；U 形 | GPT-3.5-Turbo 最多掉 20% 以上；最差時低於不給文件的 56.1% | 〔原文 §2.3 / p.5 / Fig.5〕；全文沒有量化 |
| Found in the Middle | Findings of ACL 2024 | attention 有 U 形位置偏差，和內容相關性無關 | 文件層級（K＝10、20） | K＝20 Recall@3：未校正 0.2052 → 校正 0.6832；答案在中間時 RAG +6–15 點 | 〔原文 p.6 Table 3、p.7 Fig.5〕 |
| KVQuant | NeurIPS 2024 | 第一個 token 的量化誤差 | 1 個 token 留 fp16 | nuq2：Llama-2-13B PPL 23.34→9.59（fp16 4.57）；nuq4：4.62→4.60 | 〔原文 §3.5 / p.6；App. K / p.20 / Table 13〕 |
| KIVI | ICML 2024 | 最近 R 個 token 留全精度 | R＝128（Key 平均只有 R/2） | Llama-2-13B GSM8K：全部量化成 2-bit 12.21；KIVI-2 20.77；16-bit 22.67 | 〔原文 p.6；p.8 Table 3〕 |
| SKVQ | COLM 2024 | 最近 w 個 + 最前 5 個 sink 留全精度 | w＝128；sink＝5 | LongBench 平均：RTN 35.55 → 加窗口 45.73（+10.18）…再加 sink 48.14（+0.15）；FP16 48.66 | 〔原文 §4.3 / p.8 / Table 3〕 |
| IntactKV | Findings of ACL 2024 | pivot token（[BOS] 和開頭附近的分隔符） | [BOS] 1 個，或整段 system prompt（Vicuna-v1.5-7B 34 個） | 主要是**權重**量化。KV 量化只在 INT3/4 權重量化下一起量，沒有單獨的數字 | 〔原文 p.2 Fig.1、p.4 §3.1、p.8–9 §4.3〕 |
| ZipCache | NeurIPS 2024 | **累積** attention 分數有結構性前偏 → 用每欄非零個數正規化 | 不看位置；高精度比例手動指定 | GSM8K CoT LLaMA3-8B：FP16 55.88、KIVI 47.99、ZipCache 53.75 | 〔原文 §4.2 / p.6 / Eq.8；p.8 Table 3〕 |
| QEvict | arXiv 2608.05326 v1（venue 原文沒寫） | sink + 最近區留全精度 | sink 5；最近區正文寫 32，附錄寫 LongBench 128、RULER 64、GSM8K 預算的 25% | 沒有「拿掉保護」的消融 | 〔原文 p.7、p.11、p.16〕 |
| KVSink | COLM 2025 | sink 不一定在最前面 | 預測的 5 個 sink vs 保留前 N 個（N＝0–20） | LLaMA2-70B per-token static INT4：保留前 5 個 PPL 59.5；KVSink-5 5.8（不保護 169.0） | 〔原文 p.2 Fig.1；§5.2 / p.9 / Fig.9〕 |
| Kitty | arXiv 2511.18643 v1（venue 原文沒寫） | 前 32 個 token 留全精度 | 32 | 2-bit KIVI，Qwen3-8B MATH-Algebra 47.29→74.92；平均 +8.28（Qwen3-8B）、+5.33（LLaMA3-8B） | 〔原文 §3.1 / p.4 / Table 2〕 |
| InnerQ | arXiv 2602.23200 v2 | 128 個高精度 token 裡 sink 和 recent 怎麼分 | 32 sink + 96 recent vs 0 + 128 | 短任務平均 +1.7；LongBench 只 +0.2，原文說序列越長 sink 的好處越小 | 〔原文 p.9；§6.1 / p.13〕 |
| SemKV | arXiv 2608.28911 v1 | 不看位置；比較「隨機」與「內容指標」選出的 25% 高精度 token | 25% token 拿高的那一級 | Llama-3.1-8B：兩級都在懸崖之上時，8 種指標沒有差。Mistral：隨機選 Δ＝−0.0213（p<10⁻³） | 〔原文 p.2、p.14、p.24〕 |
| Cocktail | DATE 2025 | 依 query 和 chunk 的相似度分 FP16／INT4／INT2 | chunk 層級 | 沒有位置對照組 | 〔原文 §III / p.3〕 |

---

## 1. StreamingLLM

### 出處
- Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis. *Efficient Streaming Language Models with Attention Sinks.* ICLR 2024〔原文 p.1 頁首〕。
- 讀的版本：arXiv 2309.17453v4。讀了 §1、§3、§4 開頭、App. A、App. D（p.2–7、p.15、p.17）。

### 機制
- **現象**：大部分 attention 落在最前面的 token。和這些 token 跟任務有沒有關係無關〔原文 §1 / p.2〕。
- **層別**：Llama-2-7B 第 0、1 層是「看附近」的模式。第 2 層以上，所有層所有 head 都大量看最前面的 token〔原文 p.3 / Fig.2；§3.1 / p.4〕。
- **為什麼**：softmax 的總和必須是 1。query 沒有好的對象時，也得把 attention 放在某處。最前面的 token 對後面所有 token 都看得到，所以最容易被訓練成「倒 attention 的地方」〔原文 §1 / p.2；§3.1 / p.4–5〕。
- **證據是位置，不是語意**：把前 4 個 token 換成換行符號 "\n"，模型仍然大量看它們。把它們加回 cache，PPL 恢復到接近原本〔原文 §3.1 / p.4〕。Llama-2-13B、PG19 第一本書（65K token）：0+1024（純窗口）5158.07；4+1020 5.40；4 個 "\n"+1020 5.60〔原文 p.5 / Table 1〕。
- **要留幾個**：留 1、2 個不夠完全恢復；4 個通常夠，再多效益遞減〔原文 p.5 / Table 2 標題〕。
  - Llama-2-7B（cache 4096）：0 個 3359.95；1 個 11.88；2 個 10.51；4 個 9.59；8 個 9.54〔原文 p.5 / Table 2〕。
  - Falcon-7B：0 個 17.90；1、2、4、8 個都是 12.12〔原文 p.5 / Table 2〕。
  - 實驗預設用 4 個 sink〔原文 §4 / p.6〕。
  - 作者的解釋：預訓練時沒有固定的開頭 token，所以模型用好幾個開頭 token 當 sink〔原文 §3.1 / p.5〕。
- **預訓練加 sink token**：160M 模型從頭訓練（PG19 第一筆）〔原文 §3.3 / p.6 / Table 3〕。
  - Vanilla：0+1024 27.87；1+1023 18.49；4+1020 18.05。
  - 加一個可學的 sink token：0+1024 1235；1+1023 18.01；4+1020 18.02。也就是只要留那一個 sink 就夠。
  - Zero Sink（softmax-off-by-one）有改善但仍要其他開頭 token。
  - 加 sink token 不影響收斂，也不影響 7 個 zero-shot benchmark〔原文 p.7 / Table 4〕。

### 量到什麼
- 全部是**驅逐**（只留 sink + 最近窗口），**沒有任何量化實驗**〔原文 §3–4〕。

### 原文沒寫
- 原文沒寫「前段（sink 以外）的 KV 比較重要」。
- 唯一碰到大段前文的是 App. D（p.17 / Table 8）：LongBench 上，4+3496 比「前 1750 + 後 1750 截斷」差（例如 NarrativeQA 11.6 vs 18.7）；改成 1750+1750 後回到截斷的水準。作者的歸因是**丟了開頭的提示資訊**（"loss of crucial initial input prompt information"），不是 sink〔原文 App. D / p.17〕。
- 作者自己說 StreamingLLM 不擴大 context，不適合長文件 QA 和摘要〔原文 App. A / p.15〕。

### 和我們規則的關係〔判讀〕
- 能支持的只有：「最前面 1–4 個 token」在**驅逐**時不能丟。推到量化要靠 KVQuant、IntactKV、KVSink、Kitty 等（見下）。
- 在 512-token 區塊下，這 1–4 個 token 都在區塊 0。所以「前 25% BF16」一定保護到 sink；「後 25% BF16」一定沒保護到。**不控制區塊 0，前／後的比較就混了 sink 效應。**
- App. D 的「前 1750 有用」是資訊內容（任務指令、文件開頭），而且是驅逐。不能拿來支持「前半段 KV 的量化誤差影響較大」。

---

## 2. MuKV

### 出處
- Junbin Xiao, Jiajun Chen, Tianxiang Sun, Xun Yang, Angela Yao（USTC、NUS）。*MuKV: Multi-Grained KV Cache Compression for Long Streaming Video Question-Answering.* CVPR 2026〔原文 CVF 版 p.1〕；arXiv 2605.22269v1。
- 讀的版本：CVF Open Access 版（PDF p.1–8，印刷頁 11381–11388）全文；arXiv v1 對過超參數與 §3.3 文字（相同）。補充材料（arXiv p.12–15）grep 過，沒有 position／bias 的討論。
- 程式碼：github.com/IMBALDY/MuKV，commit `2127b87`（和 V01 卡同一個 commit）。

### 機制
- **在做什麼**：每段影片存成 patch／frame／segment 三種粒度的 KV。存之前依分數**剪掉**大部分 token（不是降精度）〔原文 §3.3 / p.4〕。
- **attention 訊號**：最後一層 attention，對所有 head 與所有 query 加總：I_att＝(1/(H·P)) Σ_h Σ_i A_{h,i}〔原文 §3.3 / p.4 / Eq.1〕。理由是中間層的分數「看不出明顯分布」〔原文 p.4〕。
- **頻率訊號**：對 key 向量做 FFT，再對各維度的幅度取平均〔原文 §3.3 / p.4 / Eq.2–3〕。
  - 設計理由（原文）：attention 反映對預訓練任務的語意重要性，可能 overfit、對新任務泛化差；頻率反映「內容變化」，與任務無關。靜態、重複的內容頻率低〔原文 §3.3 / p.4〕。
  - **注意**：§3.3 的設計理由**沒有提到位置偏差**〔原文 p.4〕。
- **融合**：I＝α·Î_att＋(1−α)·Î_fft，Î 是 min-max 正規化到 [0,1]〔原文 p.4 / Eq.4〕。α＝{0.5, 0.7, 0.8}，保留比例 ρ＝{0.1, 0.1, 0.8}〔原文 §4.1 / p.5〕。
- **位置偏差的說法**：只出現在消融段。圖 5 顯示「較早到來的 token 通常拿到較高 attention，較晚的較低，不論樣本」；頻率分布比較依樣本而定，因此「提供了校正 self-attention 位置偏差的有效方法」〔原文 p.7 / Fig.5；p.8 "Effect of Frequency"〕。
  - 圖 5 的橫軸是 token index 0–約 50，標題寫 Layer 23〔原文 p.7 / Fig.5〕。
  - 〔判讀〕50 左右對得上 patch 粒度（49 個 token）。Layer 23 應是 0.5B 模型的最後一層〔未查證：Qwen2-0.5B 的層數〕。

### 程式碼查到的〔程式碼〕
- 每個區塊（49／196／784 個視覺 token）**單獨**跑一次 forward，past 只有 system prompt〔程式碼 MuKV model/mukv_dualsignal_compression.py:474–479〕。
- attention 分數是 `attn.mean(batch).mean(head).sum(query)`，也就是**欄加總，沒有除以每欄的有效 query 數**〔程式碼 MuKV model/mukv_dualsignal_compression.py:343〕；再切掉 system prompt 的部分〔同檔:351〕。
- 頻率：執行腳本設 `FFT_METHOD="fft"`〔程式碼 MuKV scripts/sh/run_mukv_rvs_ego.sh:65〕。實作是沿 **token 軸**做 rFFT，只留上半部頻率（高頻），反轉回 token 域，再對維度取平均〔程式碼 MuKV model/mukv_dualsignal_compression.py:232–262〕。Python 參數預設是 `diff`（相鄰 token 的 key 差的範數）〔程式碼 MuKV scripts/run_mukv_rvs_ego.py:231；model/mukv_dualsignal_compression.py:228〕。
- 權重：fft 權重＝1−α〔程式碼 MuKV model/mukv_rerank.py:423–426〕；融合在〔同 repo model/mukv_dualsignal_compression.py:375–391〕。

### 量到什麼
- 壓縮訊號（MuKV，保留 59K）：只用 attention 55.9／45.1；只用頻率 56.6／45.3；兩者融合 DCP(67%) 57.3／45.6（RVS-Ego／RVS-Movie）〔原文 p.7 / Table 3〕。
- 不壓縮的 MuKV（177K）53.7／44.3；ReKV 隨機丟一半 46.7／34.7〔原文 p.7 / Table 3〕。
- 高頻 vs 低頻（ReKV 上剪 50%）：留高頻 55.1／43.9；留低頻 51.4／41.4〔原文 p.7 / Table 6〕。

### 原文沒寫
- 沒有任何實驗把「位置偏差」單獨拿出來：沒有「attention 加位置校正」或「依位置剪枝」的對照組。位置偏差的說法只靠圖 5 的定性觀察。
- 沒說位置偏差是指區塊內的順序；也沒說 Eq.1 的加總本身會偏向前面。
- 沒有任何**量化**（降精度）實驗。

### 和我們規則的關係〔判讀〕
- MuKV 的「位置」是**一個獨立編碼的視覺區塊內**的順序（對 frame 來說約等於影像 patch 的掃描順序）。不是長文字 context 的前後。不能拿來支持或反對「前 25% 區塊」。
- Eq.1 是因果遮罩下的欄加總：第 j 個 token 只被 j 之後的 query 看到，越前面的 token 累積的項越多。ZipCache 正好指出這是**分數的計數假象**（見 §9）。所以圖 5 的「越前面越高」至少有一部分不是模型真的偏好前面。
- 它真正支持的是老師的方法論：**用 attention 分數判斷重要性時，先排除位置造成的假象**。這和我們要做的「位置 → 精度」規則是兩件事。

---

## 3. Lost in the Middle

### 出處
- Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang. *Lost in the Middle: How Language Models Use Long Contexts.* TACL 12:157–173（2024），DOI 10.1162/tacl_a_00638〔ACL Anthology 2024.tacl-1.9〕。
- 讀的版本：arXiv 2307.03172v3（頁碼與 TACL 版不同）。讀了 p.1–2、§2.3（p.5）、§3（p.6–7）、§4.1–4.2（p.7–8）。

### 機制（它在測什麼）
- 改變「含答案那份文件」在 prompt 中的位置，看模型答不答得對〔原文 p.1 / Fig.1〕。
- 模型：GPT-3.5-Turbo、Claude-1.3、MPT-30B-Instruct、LongChat-13B(16K)〔原文 p.2〕。
- 問題放在文件**之後**〔原文 §4.2 / p.7〕。
- **模型與 KV 都沒有壓縮**。全文沒有量化實驗。

### 量到什麼
- U 形：答案在最前或最後時最好（primacy／recency），在中間時明顯變差〔原文 p.1 / Fig.1；§2.3 / p.5 / Fig.5〕。
- GPT-3.5-Turbo 多文件 QA 最多掉 20% 以上；20、30 份文件的最差情況低於不給文件（closed-book 56.1%）〔原文 §2.3 / p.5〕。
- 合成 key-value 檢索任務也有 U 形〔原文 §3 / p.7 / Fig.7〕。
- encoder-decoder 模型（Flan-UL2）在訓練長度內很穩：最好與最差只差 1.9%；超過訓練長度後中間開始變差〔原文 §4.1 / p.7〕。
- 把問題放在文件前後（query-aware contextualization）讓 KV 檢索任務近乎完美，但多文件 QA 的趨勢幾乎不變〔原文 p.2；§4.2 / p.7–8〕。

### 原文沒寫
- 沒有任何 KV 精度、量化、壓縮的實驗。
- 沒說「中間的 KV 誤差比較不影響輸出」。

### 和我們規則的關係〔判讀〕
- 它量的是「**資訊放在哪裡，模型用得到**」，不是「**KV 哪裡有誤差，模型受影響**」。兩者不能互推。
- 就算硬要類比，它也是 **U 形**（前、後都好），不是「前段比較重要」。
- 一個可以拿來預登記的弱假說：「答案在中間的題目，本來就比較難；如果中間區塊被量化，看到的退化可能和模型本身的位置偏差混在一起」。所以測試集要讓答案位置分布均勻，並在 BF16 下先量每個位置的基線。

---

## 4. Found in the Middle

### 出處
- Cheng-Yu Hsieh, Yung-Sung Chuang, Chun-Liang Li, Zifeng Wang, Long T. Le, Abhishek Kumar, James Glass, Alexander Ratner, Chen-Yu Lee, Ranjay Krishna, Tomas Pfister. *Found in the Middle: Calibrating Positional Attention Bias Improves Long Context Utilization.* Findings of ACL 2024〔arXiv 頁面 Comments: "ACL Findings 2024"〕。
- 讀的版本：arXiv 2406.16008v2。讀了 p.1–7、p.9、App. B（p.13）。

### 機制
- **觀察**：Vicuna-7b-v1.5-16k，NaturalQuestion，20 份文件。答錯時，模型的回答和**第一份文件**最像，即使打亂文件順序也一樣〔原文 §2 / p.3 / Fig.3；p.4〕。
- 把 attention 依文件平均（所有 token、所有層、所有 head）：開頭和結尾的文件拿到較多 attention，不論內容〔原文 p.3 / Fig.4〕。
- 答錯的例子中，最常被用到的文件落在「attention 較高的一半」的有 526 例（74%），落在較低一半的有 186 例（26%）〔原文 p.4 / Table 1〕。
- **模型**：Attn(x_doc, k)＝rel(x_doc)＋bias(k)＋ε（線性假設）〔原文 §3.1 / p.5 / Eq.2〕。
  - 驗證：「位置偏差單調」與「相關性單調」兩個條件在 100 個例子上成立的比例是 83% 與 72%〔原文 p.5 / Table 2〕。
- **校正**：在同一位置 k 放一份固定的 dummy 文件，用 Attn(x_doc,k)−Attn(x_dum,k) 估計 rel(x_doc)〔原文 p.5 / Eq.3–4〕。
- **介入**：把每份文件得到的 attention 重新分配成 Softmax(rel/t)，總量不變〔原文 §4.1 / p.6 / Eq.5–6〕。只改最後 16 層（共 32 層）；改前面的層會讓生成不穩〔原文 App. B / p.13〕。
- 問題在文件前後各放一次〔原文 p.3 註 1〕。

### 量到什麼
- 用校正後的 attention 排序文件（答案在中間）：Recall@3，K＝10：未校正 0.3638、校正 0.7427；K＝20：0.2052、0.6832〔原文 p.6 / Table 3〕。
- 用校正後的 attention 生成：24 個設定中 22 個比原本好；答案在中間時提升 6–15 點〔原文 p.7 / Fig.5〕。

### 原文沒寫
- 沒有任何 KV 量化或壓縮實驗。
- 成本：每個位置要多跑 O(K) 次 forward 才能校正〔原文 §6 / p.9〕。
- 作者自己提醒：位置偏差在某些任務可能是**有益**的〔原文 §6 / p.9〕。

### 和我們規則的關係〔判讀〕
- 支持老師的警告：**attention 高不等於重要**。開頭與結尾的高 attention 有一部分和內容無關。
- 也說明模型其實**會看**中間的相關文件，只是被開頭、結尾分心〔原文 p.1 / Fig.1(c)〕。所以「中間沒人看，所以可以壓」沒有根據。
- 它的校正需要問題（query）。我們是寫入時決定精度，那時還沒有問題。

---

## 5. KVQuant

### 出處
- Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W. Mahoney, Yakun Sophia Shao, Kurt Keutzer, Amir Gholami. NeurIPS 2024〔原文 p.1 頁尾〕。
- 讀的版本：arXiv 2401.18079v6。讀了 §3.5（p.6）、App. K（p.19–20）。

### 機制
- 「Attention Sink-Aware Quantization」：**只把第一個 token** 留在 fp16〔原文 §3.5 / p.6〕。
- 校正 nuqX 資料型態與 Key 的 scale 時，也排除第一個 token〔原文 §3.5 / p.6〕。
- 原文說同樣的觀察也出現在同期的 IntactKV（參考文獻 [23]）〔原文 p.6；p.12 參考文獻〕。

### 量到什麼（Wikitext-2 PPL；X＝不保護，O＝保護第一個 token）〔原文 App. K / p.20 / Table 13〕
- nuq2：Llama-2-7B 11.20→7.03；Llama-2-13B 23.34→9.59（fp16 4.57）；Llama-3-8B 16.63→7.04（fp16 5.54）。
- nuq3：Llama-2-13B 4.82→4.71；Llama-3-8B 6.10→5.84。
- nuq4：Llama-2-13B 4.62→4.60；Llama-3-8B 5.64→5.60。
- 原文結論：所有位元數都有改善；位元越低、沒有 sparse outlier 時越明顯〔原文 p.19–20〕。

### 原文沒寫
- 沒有試「保護前 2、4、8… 個」。
- 沒有下游任務（只有 PPL）的 sink 消融。
- 沒有 fp8。

### 和我們規則的關係〔判讀〕
- 規則很窄：**1 個 token**，而且效果強烈依位元數。4-bit 時只剩 0.02–0.04 PPL。
- FP8 比 nuq4 精度更高還是更低，要看 scale 粒度，不能直接比。但依這張表的趨勢，FP8 下的 sink 效應很可能很小。這是預測，要量。

---

## 6. KIVI

### 出處
- Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen (Henry) Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, Xia Hu. ICML 2024（PMLR 235）〔原文 p.1 頁尾〕。
- 讀的版本：arXiv 2402.02750v2。讀了 §3.1、§3.3、§4.1–4.2.3（p.3、p.5–9）。

### 機制
- 最近的 R 個 token 留全精度。Key 的 residual 滿 R 個才一起量化；Value 用佇列，最舊的一個被擠出時才量化〔原文 §3.3 / p.5〕。
- 所以全精度窗口 Key 平均是 R/2，Value 是 R〔原文 p.6〕。
- 實驗設定 R＝128，group size G＝32〔原文 §4.1 / p.6〕。
- **prefill 時傳給下一層的是精確的 KV**，只有存起來的是量化版〔原文 §3.3 / p.5〕。

### 量到什麼
- Llama-2-13B GSM8K：16-bit 22.67；「全部 token 都量化」的 fake 2-bit（K per-channel、V per-token）12.21；KIVI-2 20.77〔原文 p.8 / Table 3〕。
- Llama-2-7B GSM8K：16-bit 13.50；fake 2-bit 5.76；KIVI-2 12.74〔原文 p.8 / Table 3〕。
- 原文把差距歸因於全精度滑動窗口，並說它對 GSM8K 這種難的生成任務「至關重要」〔原文 §4.2.1 / p.6〕。
- R＝32／64／96／128：GSM8K 20.62／19.86／20.55／20.77。原文說沒有一致趨勢，但「要有夠長的窗口」〔原文 §4.2.3 / p.7；p.9 / Table 5〕。

### 原文沒寫
- Table 5 沒有 R＝0 這一列。「窗口很重要」的證據是 fake quant 對 KIVI。
- 〔判讀〕這兩者不只差窗口：fake quant 的 per-channel 是對全部 token 補零分組（p.3），KIVI 是每 G 個 token 一組。所以差距不能全算在窗口上。
- 沒有「保護前面幾個 token」的實驗。

### 和我們規則的關係〔判讀〕
- KIVI 的「最近」主要是**decode 時新生成的 token**，加上 prompt 的最後一小段。
- 我們的情境是「存起來的前綴」被新問題讀取。新問題本身永遠是新算的 BF16，不會被存起來量化。所以 KIVI 支持的是「問題與生成的 token 要精確」，對「前綴尾端」只是類比。
- 另外，KIVI 的 prefill 用精確 KV。我們的新問題 prefill 會讀到量化過的前綴。誤差路徑不同。

---

## 7. SKVQ

### 出處
- Haojie Duanmu, Zhihang Yuan, Xiuhong Li, Jiangfei Duan, Xingcheng Zhang, Dahua Lin. *SKVQ: Sliding-window Key and Value Cache Quantization for Large Language Models.* COLM 2024〔原文 p.1 頁首〕。
- 讀的版本：arXiv 2405.06219v3。讀了 §3.2（p.5–6）、§4.2–4.3（p.7–9）。

### 機制
- 最近 w 個 token 留全精度。prefill 先用全精度 KV 算完 attention，再量化，只保留最後 w 個全精度〔原文 §3.2 / p.5–6〕。
- 「filter rules」：最前面幾個 sink token 也留高精度〔原文 §3.2 / p.6〕。
- 作者試過保留 heavy hitter，但「改善不明顯」，加上 FlashAttention 拿不到分數，所以沒用〔原文 §3.2 / p.6〕。
- 主實驗：w＝128，sink＝5〔原文 p.8 NIAH 段；p.8 Table 3 段〕。

### 量到什麼
- 逐步加入各技術，LongBench 平均（KV 2-bit、group 32）：FP16 48.66；RTN 35.55；+窗口 128 → 45.73（+10.18）；+clipping 46.44；+channel reorder 47.99；+sink（5 個）48.14（+0.15）；+FP8 存 scale 48.04〔原文 §4.3 / p.8 / Table 3〕。
  - 〔判讀〕表沒寫模型。FP16 48.66 和 Table 1 的 Mistral-7B-Instruct 相同，應是這個模型。
- 窗口大小（Mistral-7B-Instruct-v0.2、Llama2-7b-chat，2-bit、group 128）：原文說平均分隨窗口變大而上升〔原文 p.9 / Fig.6〕。
  - 〔判讀，圖上目測〕Mistral：w0 約 44.0、w128 約 46.0。Llama2：w0 約 34.75、w32 約 35.9、w64 約 35.6、w128 約 36.0。Llama2 不是單調。
- NIAH（Llama2-7b-80k）：KIVI 244.5；SKVQ（2-bit K、1.5-bit V，窗口 128、sink 5）272.2〔原文 p.8〕。不是單獨量 sink 或窗口。

### 原文沒寫
- 圖 6 沒說是否同時開 sink。
- sink 的效果只在「已經加了窗口、clipping、reorder 之後」量，是邊際值。沒有單獨只加 sink 的數字。

### 和我們規則的關係〔判讀〕
- 在 2-bit 下，**最近窗口**的效果（+10.18）遠大於 **sink**（+0.15，而且是最後才加）。
- 它的窗口在 LongBench 上涵蓋的是 prompt 最後 128 個 token，通常就是問題。這又是「問題要精確」，對我們的前綴尾端只是類比。

---

## 8. IntactKV

### 出處
- Ruikang Liu, Haoli Bai, Haokun Lin, Yuening Li, Han Gao, Zhengzhuo Xu, Lu Hou, Jun Yao, Chun Yuan. *IntactKV: Improving Large Language Model Quantization by Keeping Pivot Tokens Intact.* Findings of ACL 2024〔arXiv 頁面 Comments: "Accepted by ACL 2024 findings"〕。
- 讀的版本：arXiv 2403.01241v2。讀了 §2.2、§3.1、§4.3、§6、App. C.2–C.3（p.2–4、p.8–9、p.12）。

### 機制
- 「pivot token」：輸出 activation 有超大 outlier（大於 1e3）的 token，出現在 [BOS] 和一些開頭附近、沒資訊的分隔符（例如 "." 或 ","）〔原文 §2.2 / p.3〕。
  - 例子：LLaMA-30B 第 28 個 token「'」，LLaMA-2-7B 第 13 個 token「.」〔原文 p.2 / Fig.1〕。
  - attention 集中在這些 pivot token，尤其是 [BOS]〔原文 p.3〕。
  - 附錄說 pivot token 只出現在序列的最開頭，[BOS] 永遠是 pivot〔原文 App. C.2 / p.12〕。
- 做法：用**全精度模型**算出 pivot token 的 KV，離線存起來，量化模型拿來當前綴〔原文 §3.1 / p.3〕。
  - 預訓練模型：只用 [BOS]（IntactKV[B]）。
  - SFT 模型：用整段 system prompt（IntactKV[P]）；Vicuna-v1.5-7B 的 system prompt 是 34 個 token〔原文 p.4〕。

### 量到什麼
- 主要是**權重量化**（以及權重＋activation）。例如 AWQ 3-bit Vicuna-v1.5-13B 的 MT-Bench 5.17→5.34〔原文 p.8 / Table 5〕。
- KV 量化的延伸：IntactKV 留 FP16，其餘 KV 量化。結果和 INT3／INT4 **權重**量化綁在一起畫（Fig.4，MMLU，KV 3／4／8／16 bit）〔原文 §4.3 / p.8–9〕。
  - INT4 權重下，平均 +0.27%；INT8 KV 幾乎不掉分〔原文 p.9〕。

### 原文沒寫
- 沒有「只量化 KV、權重全精度」時保護 pivot token 的數字。
- 作者自己說長 context 對量化模型的影響「仍未知」〔原文 §6 / p.9〕。

### 和我們規則的關係〔判讀〕
- 支持「sink 不只 [BOS]，還有開頭附近的分隔符」。但它的例子都在前 30 個 token 內，都落在我們的區塊 0。
- 它保護的是「由全精度模型算出的 KV」，主要對抗權重量化的誤差。和我們「權重全精度、只有 KV 是 FP8」的設定不同，效果大小不能直接搬。

---

## 9. ZipCache

### 出處
- Yefei He, Luoming Zhang, Weijia Wu, Jing Liu, Hong Zhou, Bohan Zhuang. NeurIPS 2024（poster）〔neurips.cc/virtual/2024/poster/96563〕。arXiv 只有 v1，頁首仍寫「Preprint. Under review.」〔原文 p.1〕。
- 讀的版本：arXiv 2405.14256v1。讀了 §1（p.2）、§4.2–4.3（p.6–7）、§5.2（p.8–9）。

### 機制
- **累積 attention 分數的前偏是結構性的**〔原文 §4.2 / p.6〕：
  - 下三角遮罩：長度 l 的序列，第一個 token 累積 l 個正值，最後一個只累積 1 個。
  - softmax：前面幾列參與 softmax 的數較少，值偏大。
  - 所以「最後一個 token 的累積分數永遠小於第一個」。
- **修正**：p̃_i＝Σ_k A_{k,i} ／ nnz(A_{:,i})，也就是除以該欄非零個數〔原文 p.6 / Eq.8〕。
- 例子：一個 GSM8K CoT 樣本（LLaMA3-8B），正規化後的顯著 token 在 **prompt 結尾的問題**；累積分數卻把它們判成不重要〔原文 p.6 / Fig.3〕。〔判讀〕這是**單一樣本**，不是統計。
- 為了能用 FlashAttention，只對 10% 的 probe token 算完整分數：5% 最近＋5% 隨機〔原文 §4.3 / p.7〕。
- 原文批評「保留最近 token 為全精度」：最近的 token 不一定最重要〔原文 p.2、p.6、p.8〕。

### 量到什麼
- probe 選法（LLaMA3-8B GSM8K，40% 顯著 token 4-bit、60% 2-bit）：全部 token 52.54；隨機 47.46；特殊 token 46.78；最近 51.10；隨機＋最近 52.08〔原文 p.7 / Table 2〕。
- GSM8K CoT（平均輸入 840 token）〔原文 p.8 / Table 3〕：
  - LLaMA3-8B：FP16 55.88；KIVI（16/2，最近 15.2% 全精度）47.99；MiKV（累積分數）35.48；ZipCache（4/2，70% 高精度）53.75。
  - Mistral-7B：FP16 41.62；KIVI 39.04；ZipCache（4/2，60% 用 4-bit）41.24。
- Line Retrieval：比 KIVI 好，原文歸因於「顯著 token 可能在 context 任何位置」〔原文 §5.2.2 / p.9〕。

### 原文沒寫
- 沒有純位置（前段／後段）的精度對照。
- 正規化分數需要整個 prompt 的 query；沒說寫入時（還沒有問題）怎麼辦。

### 和我們規則的關係〔判讀〕
- 這是回答老師疑慮最直接的一篇：「越前面 attention 越高」**有一部分是計數方式造成的**。任何用累積 attention 推出「前段重要」的論證都要先除掉這個假象。
- 它的顯著 token 是**問題**，在我們的情境裡問題永遠是新算的 BF16。剩下的「前綴裡哪些 token 重要」它沒有給位置規律。

---

## 10. QEvict

### 出處
- Ayushman Garg, Akshita Gupta, Shaswata Bhattacharya, Abhishek Gupta, Sandeep Kumar, Manoj Kumar（IIT Roorkee、IIT Delhi）。arXiv 2608.05326v1（2026-08-05）。arXiv 頁面沒有 venue〔arXiv 頁面〕。
- 讀的版本：v1。讀了 p.1、p.4、p.7、App. B–C（p.11–12）、App. F–G（p.16）、p.20。

### 機制
- 三層：全精度、可回收的 INT2、逐出。依累積 attention 分數升降級〔原文 p.1、p.6–7〕。
- 保護區：開頭 sink 和最近區域固定全精度〔原文 p.4〕。
- 數量：
  - 正文：「除非另外說明，QEvict 用 Ω＝8、5 個 sink、32 個 recent token、INT2，跨 LongBench、RULER、GSM8K」〔原文 p.7〕。
  - 附錄：LongBench「存 128 個 recent token」〔原文 App. C.1 / p.11〕；RULER「protected recent region 64 個 token」〔原文 p.12→p.16〕；GSM8K「保留持久預算的 25% 給 recent」〔原文 p.16〕。
  - **正文與附錄互相矛盾**。

### 量到什麼
- 消融只改 Ω、量化層比例 q、量化位元〔原文 p.7；App. G / p.16〕。**沒有拿掉 sink 或 recent 保護的消融。**
- 機制分析：保護的 local 區只占參考 attention 質量的 1.9%〔原文 p.20〕。

### 原文沒寫
- sink／recent 保護本身的品質貢獻。

### 和我們規則的關係〔判讀〕
- 它的形狀是「兩頭高、中間依分數」。但兩頭的保護是設計前提，沒被驗證。只能當「別人這樣做」，不能當「這樣做有效」的證據。

---

## 11. 問題 6：有沒有人直接量「只量化前／後／中間 X%」？

### 結論
- **沒找到。**我讀到的所有位置規則都是「保護一小撮固定 token」（1–128 個），不是「把 context 切成 25% 的區段輪流量化」。〔判讀：沒找到不代表沒有〕

### 搜尋紀錄（2026-10-13）
- 用過的查詢：
  - "position-dependent KV cache quantization sensitivity which tokens quantized first last middle of context"
  - "KV cache quantization older tokens fewer bits recency-based precision decay"
  - "KV cache compression ... needle position depth quantization degrades middle"
  - "\"KV cache\" quantization \"first half\" \"second half\" tokens ablation"
  - "\"mixed-precision\" KV cache \"token position\" OR \"positional\" bit allocation early tokens later tokens"
  - "KVSink attention sinks KV cache quantization preserve first N tokens"
  - "chunk-adaptive mixed-precision KV cache quantization query relevance"
- 搜尋結果的摘要是搜尋工具生成的，下面列的每一條我都回到 PDF 核對過；只出現在搜尋摘要、沒讀原文的，標〔未查證〕。

### 最接近的（都讀過原文）
1. **KVSink**（Zunhai Su, Kehong Yuan；COLM 2025；arXiv 2508.04257v1）
   - sink 可以不在最前面：LLaMA2-7B 的 MMLU prompt 在 token 0 和 **token 14** 都有 sink〔原文 p.2 / Fig.1〕。
   - 比較「保留前 N 個（PFN）」和「保留預測出的 N 個 sink」，N＝0、5、10、15、20〔原文 §5.1 / p.8〕。
   - LLaMA2-70B per-token static INT4：KVSink-5 比不保護少 163.2 PPL，只比 FP16 高 2.5；PFN 是 59.5〔原文 §5.2 / p.9〕。
   - 套在 KVQuant 2-bit（KVQuant 原本只排除第一個 token）：Llama2-13B Wikitext-2 16.26→5.10〔原文 p.9 / Table 2〕。
   - 〔判讀〕KVSink 量到的 KVQuant 數字（16.26）和 KVQuant 自己的 Table 13（9.59）差很多，設定不同，跨論文不可比。
2. **RotateKV**（Zunhai Su 等；arXiv 2501.16383v2；venue arXiv 頁面沒寫）：LLaMA-2-7B 的一個 WikiText-2 例子，massive activation 與 sink 出現在 token 0 和 **token 110**〔原文 p.5 / Fig.5〕。
3. **Kitty**（arXiv 2511.18643v1）：KIVI 2-bit 加「前 32 個 token 全精度」〔原文 §3.1 / p.3–4〕。
   - Qwen3-8B：MATH-Algebra 47.29→74.92；AIME24 57.00→67.67；平均 +8.28〔原文 p.4 / Table 2〕。
   - LLaMA3-8B：平均 +5.33〔原文 p.4〕。
   - 最大生成長度 4096〔原文 p.9 / Table 3 註〕。任務是數學、程式、GPQA，prompt 應該不長〔判讀〕，所以前 32 個 token 占 prompt 的比例不小。
4. **InnerQ**（McGill；arXiv 2602.23200v2）：固定 128 個高精度 token，KIVI（0 sink＋128 recent）vs KIVISink（32＋96）。
   - GSM8K、Minerva、HumanEval、MBPP 平均 +1.7〔原文 p.9 / Table 1 後文〕。
   - LongBench 只 +0.2；有些情況 KIVI 反而較好；原文說「序列越長，sink 窗口的好處越小」〔原文 p.9〕。
   - 窗口分配的效果「高度依模型與任務而定」〔原文 §6.1 / p.13〕。
5. **SemKV**（ETRI；arXiv 2608.28911v1）：不看位置，用內容指標挑 **25%** token 拿較高一級精度（2.585 bit），其餘 2.322 bit〔原文 p.14〕。
   - Llama-3.1-8B：兩級都在「品質懸崖」之上時，8 種指標彼此無差（「indicator choice does not change the outcome」）〔原文 p.2〕。
   - Mistral-7B-Instruct-v0.3：**隨機**挑 25% 會顯著變差，Δ＝−0.0213（p<10⁻³，三個 seed 一致），集中在檢索類任務；內容指標則安全〔原文 p.24；p.25 / Fig.16〕。
   - 評測：LongBench n＝900、三個 seed〔原文 p.1〕；報的是對 full KV 的配對差〔原文 p.25 / Table 5 標題〕。
6. **Cocktail**（DATE 2025；arXiv 2503.23294v1）：用 RAG 的 encoder 算 query 與 chunk 的相似度，高於 T_high 的 chunk FP16、低於 T_low 的 INT2、中間 INT4〔原文 §III / p.3〕。沒有位置或隨機對照組。
7. **ReadKV**（Amazon、UC Berkeley；arXiv 2610.11245v1）：明說「KV 存下時還不知道未來的 query，但每個 query 需要的精度位置不同」，並給出一個例子證明依 query 讀取嚴格優於任何不看 query 的讀法〔原文 p.1 摘要〕。
8. **Protection Is (Nearly) All You Need**（獨立研究者；arXiv 2605.18053v1；原文自稱 "Preprint—not peer reviewed"）〔原文 p.1〕：
   - 這是**驅逐**，不是量化。
   - 不保護 prompt 邊界時，7 種驅逐策略都崩（F1≤0.064）；頭尾各保留 10% cache 可恢復 69–90%〔原文 p.1 摘要〕。
   - Qwen2.5-3B（N＝30）：位置 0 的 sink 占前段 attention 質量約 75%；其他邊界 token 只有均勻期望的約 0.41 倍，所以 attention 分數會留住 sink 卻丟掉指令分隔符、問題邊界〔原文 p.1–2〕。
9. **PM-KVQ**（ICLR 2026；arXiv 2505.18610v2）：沿用「第一個 token 存 INT16＋最近 token 滑動窗口」〔原文 p.3〕。它的 "block" 是 transformer 層，不是 token 區段。

### 只在搜尋摘要看到、沒讀原文〔未查證〕
- MixKVQ（ACL 2026，query-aware 的 key channel 精度）、MoQAE（ACL 2025）、WKVQuant、MiKV、QAQ、「Don't Waste Bits!」（arXiv 2604.04722）、RDKV（arXiv 2605.08317）。
- When Attention Sink Emerges in Language Models（arXiv 2410.10781）、Massive Activations in LLMs（arXiv 2402.17762）：sink 機制的分析，IntactKV／KVSink 有引用。

---

## 文獻真正支持的規則

### 支持的（可以預先登記成假說）

| 編號 | 規則 | 支持的論文 | 範圍限制 |
|:--|:--|:--|:--|
| S1 | **最前面極少數 token**（1–32 個）在低位元時要高精度 | StreamingLLM（驅逐，4 個）〔p.5–6〕；KVQuant（1 個）〔p.20〕；SKVQ／QEvict（5 個）〔p.8／p.7〕；Kitty（32 個）〔p.4〕；InnerQ（32 個）〔p.9〕 | 效果都在 INT2–INT4。KVQuant 4-bit 時只剩 0.02–0.04 PPL。**FP8 沒人量過**。InnerQ 說長序列時好處變小 |
| S1b | 開頭附近的**分隔符**也可能是 sink | IntactKV（LLaMA-2-7B 第 13 個）〔p.2〕；KVSink（第 14 個）〔p.2〕；RotateKV（第 110 個）〔p.5〕 | 我讀到的例子都在前 110 個 token 內。KVSink 說 sink 會出現在「開頭以外」，但我讀的段落沒看到長文後段的例子 |
| S2 | **最近約 128 個 token**（含新生成的）在 2-bit 時要高精度 | KIVI〔p.8〕；SKVQ（+10.18）〔p.8〕 | 這些方法的「最近」主要是問題與生成的 token；prefill 用精確 KV。對「存起來的前綴尾端」只是類比〔判讀〕 |
| S3 | **和 query 相關的內容**要高精度 | ZipCache〔p.6–8〕；Cocktail〔p.3〕；Found in the Middle〔p.6–7〕；SemKV（Mistral 上隨機不行）〔p.24〕 | 都需要 query。寫入時沒有 query（ReadKV〔p.1〕）。只能當 oracle 上界〔判讀〕 |

### 不支持的

| 編號 | 說法 | 為什麼不支持 |
|:--|:--|:--|
| N1 | 「整個前半段（或前 25%）比較重要」 | 沒有任何一篇在**量化**下量過。StreamingLLM 的 sink 消融只試到 8 個初始 token〔p.5 / Table 2〕。它唯一碰到大段前文的是 App. D 的 1750+1750，那是驅逐，作者把差距歸因於前文的**資訊內容**〔p.17〕 |
| N2 | 「attention 越前面越高，所以越前面越重要」 | 累積 attention 的前偏有一部分是因果遮罩的計數假象〔ZipCache p.6〕。MuKV 的觀察用的正是沒有正規化的欄加總〔程式碼 :343〕，而且是區塊內順序。Found in the Middle 的 U 形偏差和相關性無關〔p.3–5〕 |
| N3 | 「中間沒人看，所以中間可以壓」 | Lost in the Middle 量的是資訊位置，不是 KV 誤差〔全文無量化〕。Found in the Middle 顯示模型其實會看中間的相關文件〔p.1 / Fig.1(c)〕 |
| N4 | 「低位元的位置規則可以直接搬到 FP8」 | 所有位置效應都在 INT2–INT4 量到。KVQuant 的趨勢是位元越高效應越小〔p.20〕。SemKV 顯示當兩級精度都夠高時，在 Llama-3.1-8B 上連隨機選都一樣，但 Mistral 不行〔p.2、p.24〕〔判讀：模型相依〕 |

### 建議寫成的規則〔判讀，給實驗設計參考，不是文獻結論〕

1. **控制 sink**：所有實驗臂的**區塊 0（token 0–511）一律 BF16**。否則「前 25% BF16」自動贏在 S1，前後比較就混了 sink 效應。
2. 另外加一臂「**只有區塊 0 BF16、其餘全 FP8**」，單獨量 S1 在 FP8 下有多大。
3. 在區塊 0 以外，比較「前 25%／後 25%／隨機 25% 的區塊 BF16，其餘 FP8」。隨機臂要固定多個 seed。
4. 預先寫下的預測（從文獻推，等著被推翻）：
   - P0：全 FP8 和全 BF16 沒有可偵測的差 → 依老師 10/13 的停損，停在這裡。
   - P1：若有差，「只有區塊 0 BF16」補回大部分 → S1 成立；前／後／隨機之間應無顯著差。
   - P2：若控制區塊 0 後仍有差，「後 25% BF16」＞「前 25%」≈「隨機」→ 符合 S2 的類比。
   - P3：若「前 25%」＞「後 25%」（已控制區塊 0）→ 文獻**沒有**預測到，是新結果，要先排除答案位置分布的混淆（見 Lost in the Middle）。
5. 統計：效果預期很小，要用配對差與多 seed（SemKV 用 n＝900、三個 seed、配對差）〔原文 SemKV p.1、p.25〕。測試集的答案位置要均勻，並先量 BF16 下每個位置的基線。
6. 可選的診斷：記錄每個 prompt 的 massive activation 位置，確認有沒有 sink 落在區塊 0 以外（KVSink、RotateKV 的方法）。

---

## 與舊卡不一致之處

### V01（`docs/research_20261007_vlm/V01_video_kv_rekv_mukv.md`，MuKV 段）
1. **頁碼**：「設計理由（原文）……頻率可以校正注意力的位置偏差……〔原文〕p.2 §1」——p.2 §1 只說「頻率與 attention 都是有效指標」，**沒有提位置偏差**。位置偏差只在 p.8「Effect of Frequency」段（圖在 p.7）〔原文 p.2、p.7–8〕。
2. **歸類**：原文的頻率**設計理由**是「attention 可能 overfit、泛化差；頻率反映與任務無關的內容變化」〔原文 §3.3 / p.4〕。「校正位置偏差」是消融段的**事後解讀**，不是設計理由。
3. **缺的脈絡**：位置是指**單獨編碼的視覺區塊內**的順序（49／196／784 token），不是長上下文的位置〔程式碼 MuKV model/mukv_dualsignal_compression.py:474–479〕。
4. **缺的機制**：attention 分數是因果遮罩下沒正規化的欄加總〔程式碼 同檔:343〕。越前面的 token 結構上就累積較多（ZipCache Eq.8 處理的就是這個）〔判讀〕。
5. **頻率實作**：論文 Eq.2–3 寫「FFT 後對維度取幅度平均」；腳本實際用的是沿 token 軸的高通濾波再反轉回 token 域〔程式碼 同檔:232–262；scripts/sh/run_mukv_rvs_ego.sh:65〕；Python 預設則是相鄰 token 差分〔程式碼 scripts/run_mukv_rvs_ego.py:231〕。
6. **證據強度**：沒有隔離「位置偏差修正」的消融。只有圖 5 的定性觀察，加上 Table 3 的 Att／Freq／DCP（55.9／56.6／57.3）〔原文 p.7–8〕。

### D6_ZipCache
7. **「60% 用 2-bit」寫反了**：Table 3 的 Saliency Ratio 是**高精度**的比例（KIVI 那列 15.2%＝128/840 全精度可對照）。Mistral-7B ZipCache 是 60% 用 4-bit、40% 用 2-bit；內文也說「additionally quantizes 40% tokens to 2-bit」〔原文 p.8 / Table 3 與內文〕。壓縮率 4.98× 也對得上〔判讀〕。（同卡 Table 2 的「40% 4-bit／60% 2-bit」是對的〔原文 p.7〕。）
8. **venue**：舊卡寫「正式 venue 未查證」→ NeurIPS 2024 poster〔neurips.cc/virtual/2024/poster/96563〕。
9. 「salient token 集中在 prompt 結尾」來自**單一** GSM8K 樣本〔原文 p.6 / Fig.3〕，舊卡沒標。

### D6_QEvict
10. 舊卡寫「最近區域（預設 32 個 token）」。正文確實這樣寫〔原文 p.7〕，但附錄 LongBench 用 128、RULER 用 64、GSM8K 用預算的 25%〔原文 p.11、p.16〕。原文自相矛盾，舊卡沒註明。
11. 舊卡「論文沒有單獨做拿掉保護的消融（我讀的段落裡沒看到）」→ 已確認：消融只改 Ω、q、位元〔原文 p.7；App. G / p.16〕。

### D6_KIVI
12. 數字全部核對正確（Table 3：22.67／12.21／20.77；Table 5：20.62／19.86／20.55／20.77）〔原文 p.8–9〕。
13. 補充：Table 5 沒有 R＝0；「窗口重要」的證據是 fake quant 對 KIVI，兩者分組方式也不同〔原文 p.3、p.8〕〔判讀〕。KIVI 的 prefill 用精確 KV〔原文 p.5〕，和我們「新問題讀量化前綴」的誤差路徑不同〔判讀〕。

### D6_KVQuant
14. 數字全部核對正確（Table 13：Llama-2-13B nuq2 23.34→9.59；Llama-2-7B 11.20→7.03；nuq4 4.62→4.60）〔原文 p.20〕。§3.5 在 p.6。
15. 補充：只保護 1 個 token；沒有試 N＞1。

### D6_MiniKV
16. 這次**沒有重讀**（它的位置效應是留／丟，不是位元數，和本卡問題無關）。內容未複查。

---

## 本卡用到的檔案（不進 git）
- PDF 與文字：`/private/tmp/claude-501/p03_pos_lit_1791897577/`、`/private/tmp/claude-501/p03_txt_1791897577/`。
- MuKV 程式碼：`/private/tmp/claude-501/p03_code_1791897863/MuKV-2127b87/`（codeload tarball，commit `2127b87`，沒有執行任何檔案）。
