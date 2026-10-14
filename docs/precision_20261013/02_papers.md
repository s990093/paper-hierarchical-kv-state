# 02　前作在系統層怎麼做：Cake、CacheGen、位置敏感度、最接近的做法、FP8 校準

> 回到 [README](README.md)。這一頁是摘要；每個事實的完整出處（版本、頁碼、表號、程式碼行號）在 [cards/](cards/)。
> **標記**：〔原文 §／p.／Table〕〔程式碼 repo@commit file:line〕〔計算〕〔判讀〕〔未查證〕。頁碼是 PDF 頁。
> **查證狀態**：卡片是 2026-10-13 重讀原文（和程式碼）寫的。之後另有一輪獨立複核，結果見 §7。

**一句話**：Cake 和 CacheGen 都已經把「量化」和「重算／載入」接在一起，但 **Cake 只量速度、沒量品質；CacheGen 是讀取時逐 chunk 選、而且要多存幾份**。文獻支持的「位置規則」只有兩條很窄的（最前面 1–32 個 token、最近約 128 個 token），而且都只在 2–4 bit 量過。**在我們查到的文獻裡**，沒有人在 FP8 量過這兩條，也沒有人量過「整個前段比較重要」。

---

## 1. Cake：把量化當成資料、雙向會合

卡片：[cards/P01_Cake_quant_mechanism.md](cards/P01_Cake_quant_mechanism.md)（ICML 2025 正式版與 arXiv v2 全文；本機 `papers/cake2025.pdf` 是 arXiv v2）

### 1.1 還原機制（系統層）

| 面向 | Cake 怎麼做 | 出處 |
|:--|:--|:--|
| 方向 | 計算從 chunk 0 往後，I/O 從最後一個 chunk 往前 | 〔原文 ICML App. A / p.12〕 |
| **會合點怎麼決定** | **不事先規劃**。計算執行緒每算完一個 chunk，就檢查下一個 chunk 是否已載入；已載入就停。接到 vLLM 時，每個 engine step 後用 `is_resident` 檢查。**沒有估計器**，估計機制列為未來工作 | 〔原文 ICML App. A–B.2 / p.12–13；§5.4–5.5 / p.7〕 |
| 粒度 | 計算端＝vLLM token budget，預設 512；**I/O 端 128 token**。但 App. B.2 又寫「依排定的 token budget 切 chunk、把 key 反序推進佇列」，兩處說法有張力，原文沒說怎麼對齊 | 〔原文 ICML §5.1 / p.5；App. B.2 / p.13〕 |
| GPU 忙 | 用「token budget 佔比」模擬（例：512 裡給 Cake 256＝50%）；另有排程優先序 | 〔原文 ICML §5.1 / p.6；§4 / p.5〕 |
| 頻寬 | **延遲注入**，沒有真實儲存裝置。頻寬點對到雲端 egress、SSD 讀取等規格 | 〔原文 ICML §5.1 / p.5；Table 2 / p.6〕 |
| 實作 | 擴充 LMCache 並接 vLLM，約 1,000 行；基線 vLLM v0.6.2、LMCache v0.1.4 | 〔原文 ICML App. B.1 / p.12–13；§5.1 / p.5〕 |
| 下一輪 | 評估時「事先把所有請求的 KV 算好存好」，只有單次還原；寫回與下一輪原文沒寫 | 〔原文 ICML §5.1 / p.5〕 |
| 程式碼 | **沒有公開**。一作的 vLLM fork 相對上游 0 個 commit；二作 LMCache fork 只有存讀原始位元組的 serde | 〔程式碼，見卡片 E 段〕 |

### 1.2 量化實驗到底做了什麼

- **位置**：arXiv v2 是 §5.5 / Table 6 / p.7；**ICML 正式版是 §5.6 / Table 7 / p.8**（ICML 多了 chunk 大小的 §5.3，之後的節號、表號都加一）〔原文〕。
- **用了什麼**：「最常見的 8-bit 量化」和「KVQuant 的 3-bit」。3-bit 明確歸給 KVQuant；8-bit 用哪個實作，句子有歧義〔原文 ICML §5.1 / p.6〕。
- **設定**：LongAlpaca-13B（MHA）、16K、budget 512、2×A100（TP）、GPU 100%、7／25／32／56 Gbps〔原文 ICML Table 7 / p.8〕。16-bit 欄和主表 Table 3 的同一列逐格相同，是沿用，不是另外跑〔原文 ICML Table 3 / p.6〕。
- **品質**：三個版本全文**都沒有任何品質數字**〔原文〕。
- **前段重算是什麼精度、何時量化、在哪反量化**：原文都沒寫〔原文〕。〔判讀〕前段由 vLLM 從 token 現場算，預設 BF16，所以應是全精度。
- **從表推出來的兩件事**〔計算，只用表上的兩位小數〕：
  1. 「只載入」基線也是載量化後的 KV：16→8 bit 時它的時間縮 1.88–1.94 倍，16→3 bit 縮 4.21–5.02 倍。
  2. 表上看不到反量化成本：把 I/O 時間對 1／頻寬做直線擬合，和頻寬無關的常數項在 8-bit、3-bit 並沒有比 16-bit 大。
- **ICML §6 的主張**：Cake 把 KV 當資料，「agnostic to its specific representation or compression scheme」，能和量化無縫整合〔原文 ICML §6 / p.9〕。證據只有 TTFT。

### 1.3 對我們的意思〔判讀〕

1. **「Cake＋均勻量化」的品質是空白**。老師說的「Cake 已測過搭配量化」只成立在延遲面。所以 E1（全 FP8）在讀法 B 下量到的品質，本身就是 Cake 沒報的東西。
2. **「把 KV 當資料」也代表 Cake 不會依位置、或依「會被重算還是載入」去選精度**。位置相依的精度是 Cake 沒碰的軸，但它也沒說這條軸有用。
3. **sink 的保護是有條件的**：Cake 從 chunk 0 開始算，所以前幾個 token 通常是 BF16；但只有在計算端至少算完第一個 chunk 時才成立。GPU 很忙時，I/O 可能一路載到最前面。
4. **我們偏離 Cake 的地方要寫明**：I/O 粒度 512（Cake 128）、GPU 忙用背景 matmul（Cake 用 budget 佔比）。

---

## 2. CacheGen：讀取時逐 chunk 選等級，或改送文字重算

卡片：[cards/P02_CacheGen_mechanism.md](cards/P02_CacheGen_mechanism.md)（arXiv v6 正文與附錄全讀；論文 artifact `UChi-JCL/CacheGen@6bed34c` 與現行 `LMCache@240506e` 逐行讀過）

### 2.1 機制（論文寫的）

| 階段 | 做什麼 | 出處 |
|:--|:--|:--|
| 寫入（離線） | 先 prefill 整段，沿 token 維度切 chunk（預設 1.5K token），**每個 chunk 編成好幾個等級**，各自可獨立解碼 | 〔原文 §5.3 / p.7〕 |
| 等級內部 | 10 個 token 一組，組內第一個 anchor 用 8-bit，其餘存對 anchor 的差值；層分前、中、後三組，淺層量得比較細；再做算術編碼 | 〔原文 §5.2 / p.5–6；App. C.2 / p.19〕 |
| 讀取（逐 chunk） | 用**上一個 chunk 量到的吞吐量**估頻寬；挑「壓縮損失最小、又在 SLO 內」的設定。**送文字讓 GPU 重算被列為損失最小的選項**，來得及就優先 | 〔原文 §5.3 / p.7；Algorithm 1 / App. C.1 / p.19〕 |
| 重算 | 文字 chunk 要用「前一個已收到並解碼的 chunk 的 KV」來算，所以是**依序**的 | 〔原文 §5.3 / p.7〕 |
| 重疊 | 只寫了「傳 chunk i 的同時解碼 chunk i−1」；重算能不能和傳輸並行，原文沒寫 | 〔原文 §6 / p.8〕 |
| 位置 | **沒有位置規則**；哪個 chunk 被重算，看迴圈走到它時的頻寬與剩餘時間（例子裡被重算的是中間的 chunk）。唯一和位置有關的是 anchor 8-bit，以及「依 token 位置分組，熵下降遠小於依 channel、layer 分組」（這是壓縮觀點，不是品質） | 〔原文 §5.3 / p.6–7 / Fig. 7；§5.1.3 / p.5〕 |
| 敏感度 | 量的是**層**：淺層加誤差掉最多 | 〔原文 §5.1.2 / p.5 / Fig. 4〕 |
| 品質 | 有損壓縮掉幅：accuracy ≤2%、F1 <0.1%、perplexity <0.1；原文兩次稱 8-bit「幾乎無損」 | 〔原文 §7.2 / p.9；§1 / p.2〕 |
| 代價 | 多存幾份：8.5K token、Llama-13B 的全部版本約 5 GB | 〔原文 App. E / p.19〕 |

### 2.2 程式碼和論文不一樣的地方

- 論文 artifact〔程式碼 UChi-JCL/CacheGen@6bed34c `src/utils.py:294–314、357–376`〕：
  - 先試編碼等級，三級都塞不進才送文字（**和論文順序相反**）；
  - 頻寬直接取軌跡上該 chunk 的值，不是上一個 chunk 的實測；
  - 重算成本是固定 0.2 秒，**不隨位置變**（`run_adaptation.py:77` 設定，`src/utils.py:311` 使用）；
  - 「重算」的 chunk 直接拿原始全精度 KV 代替 → 品質偏樂觀〔判讀〕；
  - 編碼器沒有 anchor／差值。
- 現行 LMCache〔程式碼 LMCache@240506e〕：**只剩一組固定 codec**，沒有多等級、沒有依頻寬切換、沒有送文字重算；新的 MP mode 明寫 CacheGen「尚未實作」。

### 2.3 對我們的意思〔判讀〕

1. CacheGen 是「**讀取時才選精度**」的代表：寫入時多存，讀取時挑。它天生避開「存成 FP8 就回不去」的問題，代價是存量。
2. **我們這週的配置都是「只存一份」**，所以 CacheGen 式的「每個 block 存 BF16＋FP8、讀取時挑」（存量 1.5 倍）不放進這週的比較（存量不同）。但它是位置策略活下來之後**必須面對的上界**（[03 §0](03_plan_h200.md#0-這週只回答一個問題)）。
3. 在 vLLM 的前綴語意下，重算只能從最前面連續做。所以「CacheGen 式逐 chunk 選＋重算」在我們的設定裡會收斂成「前段重算（像 Cake）＋後段逐 block 選 BF16 或 FP8」。它和「Cake＋位置精度」的差別只剩：**精度在讀取時選（要多存）還是存的時候定（只存一份）**。這正是 [04 §2](04_story.md#2-寫入時決定還要不要) 那條軸。
4. 不要拿現行 LMCache 的 CacheGen 當「逐 chunk 適應」的基線，它已經沒有那段邏輯。

---

## 3. 位置敏感度：文獻真正支持什麼

卡片：[cards/P03_position_sensitivity_literature.md](cards/P03_position_sensitivity_literature.md)（16 篇；本卡是作者自己回 PDF 核對，不是獨立複核，見 §7）

### 3.1 老師點名的兩篇

**StreamingLLM**（ICLR 2024）
- 大量 attention（原文「a surprisingly large amount」）落在最前面幾個 token，和這些 token 跟任務有沒有關係無關。原因是 softmax 總和必須是 1，最前面的 token 對後面所有 token 都看得到，最容易被訓練成「倒 attention 的地方」〔原文 §1 / p.2；§3.1 / p.4–5〕。
- **證據是位置，不是語意**：Llama-2-13B 只留最近 1,024 個 token，PPL 5158.07；加回前 4 個 5.40；前 4 個換成換行符號 5.60〔原文 p.5 / Table 1〕。
- **要留幾個**：預設 4 個；Falcon-7B 留 1 個就夠，Llama-2-7B 要 4 個〔原文 p.5 / Table 2；§4 / p.6〕。
- **全文都是「驅逐」，沒有量化實驗**〔原文 §3–4〕。唯一碰到大段前文的是附錄 D 的「前 1750＋後 1750」，那也是驅逐，作者歸因於丟了開頭的**提示資訊**，不是 sink〔原文 App. D / p.17〕。

**MuKV**（CVPR 2026）
- 做的是**剪枝**（存之前剪掉大部分視覺 token），不是降精度〔原文 §3.3 / p.4〕。
- 它的「位置偏差」是**單獨編碼的視覺區塊內**（49／196／784 個 token）的順序，不是長上下文的前後〔程式碼 IMBALDY/MuKV@2127b87 `model/mukv_dualsignal_compression.py:474–479`〕。
- attention 分數是因果遮罩下沒正規化的欄加總〔程式碼 同檔 :343〕，越前面的 token 結構上就累積越多〔判讀〕。
- 「頻率訊號修正位置偏差」是消融段的事後解讀（p.8），設計理由是「attention 可能 overfit」（p.4）；**沒有隔離這個修正的消融**，只有 Att／Freq／兩者 55.9／56.6／57.3〔原文 p.4、p.7 Table 3、p.8〕。

### 3.2 其他有位置規則的 KV 量化

| 論文 | 保護什麼 | 量到的效果 | 出處 |
|:--|:--|:--|:--|
| KVQuant（NeurIPS 2024） | 第 1 個 token 留 FP16 | 2-bit：Llama-2-13B PPL 23.34→9.59；**4-bit：只剩 4.62→4.60** | 〔原文 App. K / p.20 / Table 13〕 |
| KIVI（ICML 2024） | 最近 128 個 token 全精度 | Llama-2-13B GSM8K：全部 2-bit 12.21；KIVI-2 20.77；16-bit 22.67 | 〔原文 p.8 Table 3〕 |
| SKVQ（COLM 2024） | 最近 128＋最前 5 個 | LongBench 平均：加窗口 +10.18，再加 sink 只 +0.15 | 〔原文 §4.3 / p.8 / Table 3〕 |
| Kitty（arXiv 2511.18643） | 前 32 個 | 2-bit，Qwen3-8B MATH-Algebra 47.29→74.92 | 〔原文 §3.1 / p.4 / Table 2〕 |
| InnerQ（arXiv 2602.23200） | 32 sink＋96 recent vs 0＋128 | 短任務 +1.7；LongBench 只 +0.2，序列越長 sink 的好處越小 | 〔原文 p.9；§6.1 / p.13〕 |
| KVSink（COLM 2025） | sink 不一定在最前面（例：第 14 個 token） | INT4：保留前 5 個 PPL 59.5；預測的 5 個 sink 5.8 | 〔原文 p.2 Fig. 1；§5.2 / p.9〕 |
| ZipCache（NeurIPS 2024） | 不看位置；累積 attention 除以每欄非零個數，去掉因果遮罩的前偏 | GSM8K CoT LLaMA3-8B：FP16 55.88、KIVI 47.99、ZipCache 53.75 | 〔原文 §4.2 / p.6 / Eq. 8；p.8 Table 3〕 |
| SemKV（arXiv 2608.28911） | 不看位置；用 8 種內部指標選 25% token 拿較高一級（2.585 bit，其餘 2.322 bit） | Llama-3.1-8B：懸崖之上 8 種指標彼此沒差（這個點**沒報隨機**）；Mistral：8 種指標都安全，**隨機選顯著變差** Δ＝−0.0213（p<10⁻³） | 〔原文 p.2、p.14、p.17 Table 3、p.24〕 |
| Lost in the Middle（TACL 2024） | **答案所在文件**在 prompt 裡的位置 → U 形 | 全文沒有量化 | 〔原文 §2.3 / p.5〕 |
| Found in the Middle（Findings ACL 2024） | attention 的 U 形位置偏差，校正需要 query | K＝20 Recall@3：0.2052→0.6832 | 〔原文 p.6 Table 3〕 |

### 3.3 支持什麼、不支持什麼

| | 說法 | 判定 |
|:--|:--|:--|
| ✅ S1 | **最前面極少數 token（1–32 個）在低位元時要高精度** | 多篇支持，但效果都在 INT2–INT4，位元越高越小；**我們查到的文獻裡沒有 FP8 的量測** |
| ✅ S2 | **最近約 128 個 token 在 2-bit 時要高精度** | KIVI、SKVQ。但它們的「最近」是問題和新生成的 token，prefill 用精確 KV；對「存起來的前綴尾端」只是類比〔判讀〕 |
| ✅ S3 | **和內容相關的 token 要高精度** | ZipCache、Found in the Middle 需要 query，寫入時只能當上界〔判讀〕。**SemKV 不一樣**：8 種指標裡 7 種只用 token 自己與之前的狀態，寫入時就能算〔原文 SemKV p.7〕；它在 Mistral 上明顯比隨機好，是「寫入時可用的內容訊號」的證據 |
| ❌ N1 | 「整個前半段（或前 25%）比較重要」 | **沒有任何一篇在量化下量過** |
| ❌ N2 | 「attention 越前面越高，所以越前面越重要」 | 累積 attention 的前偏有一部分是因果遮罩的計數假象（ZipCache）；MuKV 用的正是沒正規化的分數 |
| ❌ N3 | 「中間沒人看，所以中間可以壓」 | Lost in the Middle 量的是資訊位置，不是 KV 誤差 |
| ❌ N4 | 「低位元的位置規則可以直接搬到 FP8」 | 所有位置效應都在 INT2–INT4 量到；KVQuant 的趨勢是位元越高效應越小 |

### 3.4 寫進計劃的規則

1. **混合配置一律把 block 0 留成 BF16**（sink 都落在 block 0 裡）；另加 E1s（全 FP8、只有 block 0 BF16）單獨量 sink（[03 §2.1](03_plan_h200.md#21-六個配置)）。
2. **答案位置和 FP8 位置交叉**：資訊放前、中、後，FP8 放前、後、隨機，才能分開「位置敏感」和「剛好壓到答案」（[03 §3.1](03_plan_h200.md#31-任務不要只用容易滿分的單題撈針)）。
3. **預測先寫死**（[03 §2.3](03_plan_h200.md#23-先寫清楚的預測)）：文獻預測 E1s 在 FP8 下幫助不大（N4），前段和後段的差別主要來自答案在哪裡（N1、N3）。如果控制 sink 之後前段仍然比較敏感，那是文獻沒預測到的新結果。

---

## 4. 最接近的做法（新穎性威脅）

卡片：[cards/P05_closest_work_mixed_precision.md](cards/P05_closest_work_mixed_precision.md)（多數 B 級：讀了全文，但經工具摘要或 grep 定位；只讀摘要的標 C）

### 4.1 三類前作

| 類別 | 代表 | 精度由什麼決定 | 何時決定 | 和重算／載入 | 出處 |
|:--|:--|:--|:--|:--|:--|
| **精度跟著重算／載入的切分走** | **PatchKV**（arXiv 2609.26219） | attention 質量 × 低 bit 重建誤差 → 每個 block 的風險 | 卸載時凍結 | 卸載時**所有** block 都依凍結的標籤量化存起來（FP16／K8V8／K8V4）；恢復時重算集不讀存的那份，**只有載入集讀回並反量化** | 〔原文 p.5 Eq. 12、p.6〕 |
| | **SparKV**（IEEE IoT-J；arXiv 2604.21231） | 逐層非均勻量化，**不隨位置變** | 串流前 | 把「Cake＋載入段用同樣量化」當 baseline（Strong Hybrid）；說位置是 chunk **成本**的差勁代理：在 sparse attention 下，同位置 chunk 的計算時間可差 3–5 倍（邊緣裝置）。講的是成本，不是品質敏感度 | 〔原文 p.4、p.8〕 |
| | **CacheGen**（SIGCOMM 2024） | 頻寬＋SLO | 讀取時逐 chunk；寫入時存多份 | 可改送文字重算，依序 | 見 §2 |
| **已存 KV 逐 chunk 選格式＋多層放置** | **HA-RAG**（HPCA 2026；arXiv 2510.20878） | 存取熱度：最熱 10% INT8、次 10% E4M3、再 10% E5M2、其餘 GSE-8 | 離線 | 4 層放置，**沒有重算**；有逐格式的品質（Fig. 10–11），但混精度整體只和未壓縮的 TurboRAG 比，沒和均勻 INT8 並排 | 〔原文 p.7–9，B 級〕 |
| | LMCache fp8 serde | 依存放層 | 寫到該層時 | 沒有重算決策 | 〔文件〕 |
| **GPU 內的逐 chunk／逐 token 混精度** | Cocktail（DATE 2025） | query 和 32-token chunk 的相似度 | 每個請求、推論前（要先知道 query） | 沒有 | 〔原文 §III / p.3〕 |
| | MoQAE（ACL 2025） | 學出來的 router；**第一個 chunk 固定 FP16** | prefill | 沒有 | 〔原文，B 級〕 |
| | PM-KVQ（arXiv 2505.18610） | 記憶體壓力＋逐層敏感度；**最前與最近 128 token 保持 16-bit** | 快滿時才逐步降 | 沒有 | 〔原文，B 級〕 |
| | DiffKV（SOSP 2025） | 每 token 收到的平均 attention；最近 64 個高精度 | prompt 階段一次，decode 中降級 | 沒有 | 〔原文，B 級〕 |
| | KVTuner（ICML 2025） | 逐層敏感度 | 離線 | 沒有 | 〔原文，B 級〕 |
| | SemKV（arXiv 2608.28911） | 8 種內部指標，刻意不用位置 | prefill 後 | 沒有 | 〔原文 p.2、p.14、p.24〕 |

### 4.2 威脅排序與還剩的空白〔判讀〕

| 排名 | 前作 | 威脅到哪一句 | 還剩什麼 |
|:--|:--|:--|:--|
| 1 | PatchKV | 「精度跟重算／載入切分綁在一起」已經有了 | 切分依編輯位置、不依會合點 b；精度看 attention 風險、不看位置；原文沒說重算和傳輸是否兩頭並行；收益只比均勻 INT8 快 5.6% |
| 2 | SparKV | 「Cake＋載入段量化」已經是別人的 baseline；而且說位置是**成本**的差勁代理 | 精度不隨位置變；邊緣裝置；沒量品質對位置 |
| 3 | CacheGen | 「逐 chunk 精度＋可退回重算」已經有了 | 讀取時選、要多存；重算依序；精度不看位置 |
| 4 | HA-RAG | 「已存 KV 逐 chunk 選精度＋多層放置」已經有了 | 依熱度；沒有重算；混精度沒和均勻 INT8 並排比 |
| 5 | PM-KVQ | 「先高精度、壓力來了才降」＋「最前／最近保持高精度」 | GPU 內，沒有存放層、沒有重算 |
| 6 | Cake | 「量化讓會合點往前移」不新 | **Cake＋量化的品質是空白** |

**我們找到的空白**（四條都「沒找到」，不等於「沒有」）：
1. 精度依「block 和會合點 b 的相對位置」決定。
2. 在 Cake 式還原下量**品質**，並把「被重算蓋掉所以沒讀到」和「讀到了但不敏感」分開。
3. 同樣位元組、同樣 codec、同樣 attention 下，前／後／隨機的對照。
4. 多輪重用時逐輪報量化 KV 的品質：SCBench（ICLR 2025）只測了 KIVI 2-bit 而且只報平均；SemKV 對多輪掉分的歸因是推測，沒有消融〔原文，P05 §3〕。

**要誠實面對的反方證據**：
- SparKV 說位置是 chunk **成本**的差勁代理（sparse attention 下），和精度敏感度無關。SemKV 在 Llama 懸崖之上只顯示「8 種指標彼此沒差」；在 Mistral 上**隨機明顯比指標差**。所以文獻沒有說「隨機 25% 會很強」。R25 仍是主要對照，因為它是「同位元組、不用任何資訊」的基準；位置策略連它都贏不了，就沒有貢獻。
- 反過來，SemKV 顯示「寫入時就算得出來的內容訊號」在某些模型上明顯勝過隨機。這是故事 C（敏感度才是關鍵）的支持證據，也代表位置規則面對的強對手（S25）是有根據的。
- PatchKV 的混精度只比均勻 INT8 快 5.6%，我們的收益量級可能也一樣小。
- Qwen2.5 的全 FP8 若是因為 K 的固定偏移而失敗（§5），位置怎麼放都救不了，那時該換的是 codec。

---

## 5. vLLM 的 FP8：怎樣才算「正確校準」

卡片：[cards/P04_vllm_fp8_calibration.md](cards/P04_vllm_fp8_calibration.md)（vLLM `v0.28.0`＝commit `2cf0a69`、FlashInfer `v0.6.16.post3`、PyTorch 2.13.0／2.9.0 原始碼）

### 5.1 vLLM 0.28.0 實際怎麼做

| 問題 | 答案 | 出處 |
|:--|:--|:--|
| scale 從哪來 | 對 `fp8`／`fp8_e4m3`（per-tensor）只有兩種：checkpoint 的 per-tensor scale，沒有就 1.0；純 BF16 或 AWQ 權重都是 1.0。另有 `*_per_token_head`（fp8／int8／int4）在寫入時逐（token, head）動態算 scale，但只有 TRITON_ATTN 支援，而它在 SM89 以下拒收所有 fp8*，**所以 3090 上只有 INT8 動態版能用** | 〔程式碼 vllm@v0.28.0 `attention.py:124-145、:580-592`；`kv_cache.py:82-111`；`auto_awq.py`；`triton_attn.py:540-549`〕 |
| `--calculate-kv-scales` | **已移除**（PR #49389，2026-08-03 合併）。移除前是「第一次 forward 算一次、整個 tensor 一個值；Q、K 是 max÷200，**V 是 max÷100**」 | 〔文件 PR #49389；程式碼 vllm@0cf49a5 `attention.py:584-594`、`envs.py:1328-1333`〕 |
| 3090 上誰收 FP8 KV | **只有 FlashInfer**。FlashAttention 在 sm_86 退回 FA2、不收 FP8；TRITON_ATTN 在 SM89 以下拒收所有 fp8* | 〔程式碼 vllm@v0.28.0 `fa_utils.py:223-243`；`flashinfer.py:408-417`；`triton_attn.py:540-549`〕 |
| 同一條 attention 路徑嗎 | **不是**：BF16 預設 FA2，FP8 走 FlashInfer。要比要兩邊都指定 `--attention-backend FLASHINFER` | 〔判讀，依上列程式碼〕 |
| FlashInfer 在 sm_86 怎麼算 FP8 | Q 保持 BF16；prefill 把 FP8 的 K／V 先轉 BF16 再算，decode 轉 FP32 | 〔程式碼 flashinfer@v0.6.16.post3 `prefill.cuh:1098-1124`、`decode.cuh:72-79`〕 |
| 超過 ±448 | 截在 448（`__NV_SATFINITE`），不會變 NaN | 〔程式碼 vllm@v0.28.0 `csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh:490-503`〕 |
| 依位置或逐 block 的混精度 | **沒有**；只有逐層跳過量化（`--kv-cache-dtype-skip-layers`） | 〔程式碼 `cache.py:114-116`；判讀〕 |
| 一個坑 | checkpoint 帶 q_scale 時，sm_86 上 Q 沒被量化，FlashInfer 卻照樣把 q_scale 乘進 softmax 縮放 → 要用不帶 q_scale 的 KV-only checkpoint | 〔程式碼 `flashinfer.py:952-961、:2154`；判讀，未實測〕 |

### 5.2 Qwen2.5 的 FP8 為什麼會壞〔判讀〕

- **多半不是溢位**：第三方量到 Qwen2.5-7B 的 max|K| 約 419–423，小於 448〔文件 第三方 fork issue #28，GPTQ 權重，未重現〕；我們在 MI300X 上「不可能溢位」的逐 token 逐 head 動態 FP8 也只拿 0/20（但走的是另一個 kernel）〔實測 20260916-065920〕。
- **比較像是 K 的固定偏移 × 3 位尾數**：K 在少數 channel 有很大、幾乎固定的偏移（k_proj bias 在最低頻的 RoPE channel 幾乎不被轉動）。FP8 在 400 附近的格子寬 32，捨入誤差蓋過 q·k 裡有用的小差異。同一個 fork 量到：只改 scale 的各種做法，第 27 層 attention 的最大相對誤差仍有 0.71–0.94；**先減掉每（head, channel）平均、再對殘差用每（head, channel）scale，降到 0.031**（只減平均、scale 1.0 是 0.070）〔文件 fork issue #28、PR #30〕。
- **對計劃的影響**：
  1. 「正確校準的全 FP8」在 Qwen 上很可能還是過不了，這不是校準做錯，是格式的問題。
  2. 所以 D0 先量 K／V 的實際分布（30 分鐘），確認這個機制在 BF16 權重的 Qwen2.5-7B-1M 上成不成立。
  3. E1 不過時，Step 2b 先試「K 去偏移的 FP8」和 INT8。任一個過 → 停損 K3（[03 §2.4](03_plan_h200.md#24-判準開跑前寫死)）。

### 5.3 我們的端點為什麼不用 vLLM 原生 FP8

- 老師要求「codec 和 attention 計算方式相同」。vLLM 原生 FP8 在 3090 上會換 kernel（FA2 → FlashInfer），而且 per-tensor scale 只能來自 checkpoint，Qwen 沒有現成的。
- 所以 E1 用我們自己的 codec：每個（block, 層, K/V, KV head）用 amax÷448 當 scale，反量化成 BF16 後走和 E0 相同的 attention。這在溢位和次正規兩端都比 per-tensor 靜態 scale 好，對停損是保守的方向。
- 原生 FP8 只當選做的參考點（[03 §9](03_plan_h200.md#9-風險與對策)）。

---

## 6. 對這週實驗設計的影響（彙整）

| 從哪裡來 | 改了計劃的什麼 | 計劃的位置 |
|:--|:--|:--|
| Cake 只量速度、沒量品質 | E1 在讀法 B 下的品質本身就是 Cake 沒報的東西；E1 定位成「Cake＋均勻量化」 | [03 §2.1](03_plan_h200.md#21-六個配置) |
| Cake 的會合點自然形成、沒有估計器 | harness 照同樣方式會合；b 從速度實驗量出來 | [03 §1](03_plan_h200.md#1-固定的條件開跑前寫死) |
| Cake 的 I/O 粒度 128、GPU 忙用 budget 佔比 | 我們用 512 與背景 matmul，偏離處寫明 | [03 §1](03_plan_h200.md#1-固定的條件開跑前寫死) |
| CacheGen 讀取時選、存多份 | 「每個 block 存兩份、讀取時挑」不放進這週（存量不同），但記為位置策略活下來後的上界 | [03 §0](03_plan_h200.md#0-這週只回答一個問題) |
| sink 只在最前面 1–32 個 token，我們查到的文獻裡沒有 FP8 的量測 | 混合配置一律把 block 0 留 BF16；加 E1s 單獨量 sink；加停損 K1s | [03 §2.1、§2.4](03_plan_h200.md#21-六個配置) |
| 「整個前段比較重要」沒有證據；Lost in the Middle 講的是資訊位置 | 答案位置 × FP8 位置交叉設計 | [03 §3.1](03_plan_h200.md#31-任務不要只用容易滿分的單題撈針) |
| SemKV（Mistral 上隨機變差、寫入時可算的內容指標勝過隨機）；SparKV（只談成本） | R25 是「不用任何資訊」的基準，位置策略至少要贏它；文獻沒有預測它強或弱。強規則 S25 有文獻根據 | [03 §2.3](03_plan_h200.md#23-先寫清楚的預測) |
| Qwen 的 FP8 失敗多半是 K 偏移 × 3 位尾數 | D0 量 K／V 分布；Step 2b 換 codec 診斷（E1-K0、E1-INT8）；停損 K3 | [03 §1、§2.1、§2.4](03_plan_h200.md#1-固定的條件開跑前寫死) |
| vLLM 原生 FP8 在 3090 換 kernel、scale 只來自 checkpoint | 端點用自己的 codec；原生 FP8 只當選做參考 | [03 §9](03_plan_h200.md#9-風險與對策) |
| SCBench 用標準答案接後續輪 | 第 2 輪的歷史固定成標準答案 | [03 §3.4](03_plan_h200.md#34-第-2-輪重用) |

---

## 7. 舊卡要更正的地方與複核

### 7.1 舊卡的更正（卡片作者重讀原文時發現；**還沒有改舊卡**，等複核後一起改）

| 舊卡 | 要改的 | 依據 |
|:--|:--|:--|
| D6_Cake | 量化表在 ICML 版是 **Table 7**（§5.6, p.8），arXiv v2 才是 Table 6（§5.5, p.7）；8-bit 是否為 KVQuant 有歧義；「sink 永遠是 BF16」要加「計算端至少算完第一個 chunk」；開銷文字在 ICML §5.9 / p.8 | P01 修正清單 1–6 |
| E03 | 二作的 LMCache fork 已確認不是 Cake 的實作；v1 的量化實驗是 8-bit＋CacheGen、14K、單張 A100 | P01 修正清單 7–8 |
| D6_CacheGen／D5_CacheGen | 算術編碼的分布原文 p.6 與 p.8 互相矛盾；公開程式碼沒有 anchor／差值；Algorithm 1 看的是「剩下所有 chunk」，而且送文字重算是優先選項；artifact 的順序和論文相反 | P02 修正清單 1–5 |
| D8_CacheGen | 「逐 chunk 靜態選擇」應為**動態**（依上一個 chunk 的吞吐量） | P02 修正清單 6 |
| V01（MuKV） | 「位置偏差」只出現在 p.8 的消融討論，不是 p.2 §1 的設計理由；指的是視覺區塊內的順序；分數是沒正規化的欄加總；論文的頻率公式和腳本實作不同 | P03 修正清單 1–6 |
| D6_ZipCache | 「60% 用 2-bit」寫反了，應是 60% 用 4-bit；venue 是 NeurIPS 2024 | P03 修正清單 7–8 |
| D6_QEvict | 最近區的長度原文自相矛盾（正文 32，附錄 128／64／25%） | P03 修正清單 10 |

### 7.2 複核狀態

- P01–P05 是卡片作者自己回到原文、程式碼核對（self-check）。
- **獨立複核**（2026-10-13，沒看過前文的 agent，只對原文、原始碼、官方頁面核對；結果在 [cards/V_verification_20261013.md](cards/V_verification_20261013.md)）：本頁、01 §3、03 §1 的算術、04 §5 的日期，共 **85 項：✅ 71、⚠️ 12、❌ 2、❓ 0**。
  - 兩個 ❌ 都是 SemKV：(1) 它**不支持**「隨機 25% 是強對照」，Mistral 上隨機明顯變差；(2) 它的指標大多**不需要 query**，寫入時就算得出來。已改正（§3.2、§3.3、§4.2、§6）。
  - ⚠️ 已改正：Cake I/O 粒度的兩處說法、CacheGen 0.2 秒的出處、StreamingLLM 的措辭、PatchKV（全部量化存起來，重算集只是不讀回）、SparKV（講的是成本不是品質）、HA-RAG（HPCA 2026，有逐格式品質）、vLLM scale 來源（另有 per-token-head 動態版，3090 上只有 INT8 能用）、舊版 V 的除數（÷100）、fork 的 0.031 要兩步一起做、MI300X Qwen 的長度。
  - 這是同一個模型家族的 agent，**不是 cross-model 審查**（CLAUDE.md §5）。
  - **沒查到的**（照實列）：「FP8 沒人量過」是對整個文獻的否定句，只能寫成「我們查到的文獻裡沒有」；PatchKV 是否兩頭並行；MoQAE／PM-KVQ／DiffKV／KVTuner 只核短語；§7.1 舊卡更正表裡 QEvict 與 MuKV 公式兩項沒核；MuKV 的 CVF 版頁碼；第三方 fork 的數字沒有重現。
