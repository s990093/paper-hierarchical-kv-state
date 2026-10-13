# P05｜最接近的前作：逐 chunk／依位置的混精度 KV，以及它和還原、重算、多輪重用的關係

> **2026-10-13 獨立複核更正**（見 [V_verification_20261013.md](V_verification_20261013.md)）：(1) SemKV **不支持**「隨機 25% 是強 baseline」：Llama 那一點只比了 8 種指標、沒報隨機；Mistral 上隨機顯著變差（Δ＝−0.0213），8 種指標都安全。(2) PatchKV 卸載時**所有** block 都量化存起來，重算集只是恢復時不讀回，不是「不存」。(3) SparKV 的「位置是差勁代理」講的是 sparse attention 下的計算**成本**，不是品質。(4) HA-RAG 是 HPCA 2026，有逐格式的品質（Fig. 10–11）。下文保留原句，以這段為準。

> 一句話：「逐 chunk 混精度」本身不新（Cocktail、MoQAE、HA-RAG、PatchKV 都做了），「精度跟著重算／載入切分走」也有人碰過（PatchKV 只量化要載入的那段；SparKV 拿「Cake＋載入段量化」當 baseline；CacheGen 每個 chunk 可退回重算）。但我找不到任何一篇**依 block 和會合點 b 的相對位置**決定精度，也找不到任何一篇在 Cake 式雙向還原下量品質、把「沒被讀到」和「不敏感」分開。多輪重用的品質也幾乎沒人逐輪報：SCBench 只報平均，SemKV 只推測原因、沒有做消融。

- **日期**：2026-10-13。
- **和 P03 的分工**：位置規則的文獻（StreamingLLM、KVQuant、KIVI、SKVQ、IntactKV、ZipCache、QEvict、KVSink、Kitty、InnerQ）在 [P03](P03_position_sensitivity_literature.md)。本卡只看三件事：(1) 混精度的**粒度與決定時機**；(2) 是否和**卸載／還原／重算**接在一起；(3) **多輪重用**與**資訊深度**的評估。
- **證據等級**：A＝自己逐頁讀 PDF；B＝讀了全文，但經 WebFetch 摘要或 grep 定位段落（數字取自工具回報的表格，我沒逐格對 PDF）；C＝只讀摘要或搜尋結果。引用既有卡片時沿用該卡的等級。
- **標記**：〔原文 §／p.／Table〕、〔文件 URL〕、〔程式碼 repo file:line〕、〔判讀〕、〔未查證〕。
- **複核**：self-check，不是獨立 reviewer。venue 用 OpenAlex DOI、PMLR／ICLR 頁面或論文頁首確認，每列都寫了來源。

---

## 總表

| 論文 | venue／ID | 精度由什麼決定 | 何時決定 | 粒度 | 和重算／載入 | 對均勻量化的增益 | 等級 |
|:--|:--|:--|:--|:--|:--|:--|:--|
| **PatchKV** | arXiv 2609.26219（venue 未查證） | 存下的 attention 質量 × 低 bit 重建殘差 → 風險 | 卸載 checkpoint 時凍結；resume 前完成 | block | **只有載入集 F 量化**；重算集 R 依「離編輯點的距離」挑 | 比均勻 INT8 快 5.6%、F1 差 ≤0.05 | B（D6_PatchKV 卡） |
| **SparKV** | IEEE IoT-J；arXiv 2604.21231 | 層級非均勻量化＋Huffman，**不隨位置變** | 串流前（編碼） | (chunk, 層, head) 排程；精度固定 | 每 chunk 選串流或本地算；把「Cake＋載入段量化」當 baseline（Strong Hybrid） | Strong Hybrid 仍比 SparKV 慢約 1.3× | B（D6_SparKV 卡） |
| **CacheGen** | SIGCOMM 2024 | 頻寬＋SLO | **讀取時**逐 chunk 挑；寫入時存多個等級 | chunk（預設 1.5K token） | 每 chunk 可改送文字讓 GPU 重算；**依序**，不是兩頭並行 | 見 P02 | A（D6 卡、P02） |
| **Cake** | ICML 2025 | 不決定；只試均勻 8-bit／3-bit | — | — | 雙向並行（前段算、後段載） | 只報加速，**沒有品質** | A（P01） |
| **HA-RAG** | arXiv 2510.20878（頁首：HPCA 2026 投稿） | **存取熱度**（hot→INT8、次熱→E4M3→E5M2、其餘 GSE-8） | 離線 | 512-token RAG chunk | 4 層放置（GPU／pinned／pageable／disk）；**沒有重算** | TTFT 對 TurboRAG（BF16）平均 2.10×；**沒和均勻 8-bit 比** | B |
| **LMCache fp8 serde** | 官方文件 | **依存放層**（每個 L2 adapter 各選） | 寫到該層時 | adapter（層） | 無重算決策 | 文件只說「minor accuracy loss」 | 〔文件〕 |
| **Cocktail** | DATE 2025 | query 與 chunk 的 Contriever 相似度 | 每個請求、正式推論前（要先知道 query） | 32-token chunk；FP16／INT4／INT2 | 只在 GPU 內 | LongBench 平均比 FP16 只低 0.055；4 個模型都比 KVQuant 高 0.30–0.47 | B |
| **MoQAE** | ACL 2025 | 學出來的 router（要微調 router）；**第一個 chunk 固定 FP16** | prefill，逐 chunk | 32-token chunk | 只在 GPU 內 | MultiNews：FP16 3.51、MoQAE 3.47、KIVI-2b 0.97 | B |
| **PM-KVQ** | arXiv 2505.18610 | 記憶體壓力（先 16-bit，快滿才逐步降）＋逐層敏感度；**最前與最近 128 token 保持 16-bit** | 降階在 decode 中、記憶體快滿時 | 層 × 時間 | 只在 GPU 內 | Qwen-7B 2-bit AIME24：KIVI 32.08 → 40.00 | B |
| **DiffKV** | SOSP 2025 | 每 token 收到的平均 attention；**最近 64 個 token 一律高精度** | prompt 階段一次；decode 中逐 token 降級或剪掉 | token；K8V4／K4V2／剪 | 只在 GPU 內 | Llama3-8B GSM8K：DiffKV 75.7（19.3% 記憶體）vs INT4 72.6（25%） | B |
| **KVTuner** | ICML 2025（PMLR 267） | 逐層敏感度（離線搜尋） | 離線 | 層（K、V 分開） | 無 | Qwen2.5-7B：KV4 GSM8K 0.0078 → KVTuner-C4.0 0.7559 | B |
| **SemKV** | arXiv 2608.28911 | 8 種模型內部指標（**刻意不用位置**） | prompt：prefill 後；生成：每 64 token 一塊 | token；相鄰兩級 bit | 只在 GPU 內（fake quant） | 6.0× 壓縮、偵測不到品質損失（n=900） | B |
| **Minima-KV** | arXiv 2608.23834 | **頁的新舊與是否為 anchor**：最近／受保護頁 FP8，較舊頁 TQ3 | 頁變舊時 | 頁 | 摘要沒提 | LongBench v2：16K／32K／64K 掉 0.80／0.60／0.40 點 | C |
| **TierKV** | arXiv 2609.21172 | 從 prefill hidden state 預測未來需求 | prefill 後、decode 前 | token；exact／low-rank／存 flash | 有卸載到 flash | 摘要只說「minor accuracy loss」 | C |
| **EvicPress／AdaptCache** | arXiv 2512.14946／2509.00105 | 每個 context 的 utility（頻率×品質−延遲） | 層滿時降級（延後型） | context | 有多層，沒有重算並行 | 見 D6 卡 | A（D6 卡） |
| **KVServe** | arXiv 2605.13734（頁首 SIGCOMM'26） | 工作負載、頻寬、SLO | 搬移開始時，整個請求固定 | 請求（內部可分層／head） | 沒有重算 | Qwen2.5-7B 相對準確率 100.35%、壓縮 8.28× | B（D6 卡） |

只看摘要、威脅低、不放進總表的：MiKV（arXiv 2402.18096，被驅逐的 token 改存低精度）〔文件 https://arxiv.org/abs/2402.18096〕；MixKVQ（arXiv 2512.19206，依 query 選 K 的高精度 channel）〔文件 https://arxiv.org/abs/2512.19206〕；HqeKV（Findings of ACL 2026，量化與驅逐的混合動作）〔文件 https://preview.aclanthology.org/ingest-acl/2026.findings-acl.201/，未開頁，只看搜尋摘要〕；UltraQuant（arXiv 2606.20474，多輪 agent 前綴重用下的**均勻** 4-bit，對照 vLLM FP8）〔未查證：只看到搜尋結果，沒開論文〕；SwiftCache（arXiv 2606.16135，名字有「heterogeneous」但全部 FP16，和精度無關）〔原文 §5，B〕。

---

## 1. 和「精度 × 重算／載入」直接相關的

### 1.1 PatchKV（最接近）

- **精度跟著切分走**：先把 block 分成重算集 R 和載入集 F，只有 F 進入量化傳輸；重算的部分不需要精度〔原文 p.5，轉引 D6_PatchKV 卡〕。
- R 依「離編輯點的距離」挑（KV 漂移隨距離衰減），加上 attention 挑的遠處 block〔原文 p.3、p.5，轉引〕。F 內每個 block 選 FP16、K8/V8、K8/V4〔原文 p.5，轉引〕。
- F 內的精度看 attention 風險，**不看位置**〔原文 p.5，轉引〕。
- 有非同步 H2D，但「兩組都放進目標 KV row 之後才開始 decode」，沒有 Cake 式兩頭會合〔原文 p.5–6，轉引〕。
- 增益：固定重算集、只改精度時，比 FP16 快 15.0%（F1 −0.35），比 INT8 快 5.6%（F1 差 ≤0.05）〔原文 p.8，轉引〕。
- 場景是「編輯後恢復」，不是一般前綴還原〔判讀〕。

### 1.2 SparKV（把「Cake＋載入段量化」當 baseline）

- Strong Hybrid baseline：早的 chunk 本地算、晚的 chunk 串流，串流段用和 SparKV 一樣的量化＋Huffman〔原文 p.8，轉引 D6_SparKV 卡〕。
- 原文明確說「位置是 chunk 成本的差勁代理」，因為同位置的 chunk 計算時間可差 3–5×〔原文 p.4，轉引〕。
- 精度不隨 chunk 是串流還是本地算而變〔原文 p.5，轉引〕。
- 平台是筆電 GPU 和 Jetson，模型 4-bit 權重〔原文 p.7，轉引〕。伺服器上 chunk 計算時間隨位置線性增加（Cake Fig. 4）〔原文 Cake p.4，轉引 D6_Cake 卡〕，所以兩邊的結論不衝突〔判讀〕。

### 1.3 CacheGen 與 Cake

- 細節分別在 [P02](P02_CacheGen_mechanism.md)、[P01](P01_Cake_quant_mechanism.md)。這裡只記和本卡相關的兩點：
- CacheGen：寫入時存多個等級，讀取時逐 chunk 依量到的吞吐量挑；來得及就送文字讓 GPU 重算；重算依序進行〔原文，見 P02〕。
- Cake：均勻 8-bit／3-bit 接在雙向還原上；32 Gbps 時相對只算的加速是 16-bit／8-bit／3-bit＝1.59／2.12／3.73×；**沒有任何品質數字**〔原文，見 P01、D6_Cake 卡〕。

### 1.4 HA-RAG（逐 chunk 精度＋多層放置）

- 存的是知識庫文件的預算 KV chunk，每 chunk 512 token〔原文 §II-A、§III-A〕。
- 精度依**存取熱度**：最熱的用 INT8，再來 E4M3、E5M2，其餘用自訂的 8-bit 共享指數格式 GSE-8；門檻 τ1＝τ2＝τ3＝10%〔原文 §II-B3、§III-B1、Alg. 1〕。
- 壓縮離線做；放置在執行時用 LRU，分 GPU／pinned／pageable／disk 四層〔原文 §II-A、§II-C、Alg. 2〕。
- **沒有重算**，Cake 只出現在相關工作〔原文 §IV〕。
- 結果：TTFT 對 TurboRAG（BF16）平均 2.10×、最高 10.49×〔原文 摘要、§III-C〕。準確率只報 ROUGE-1 對 TurboRAG 的分布〔原文 §III-C、Fig. 13〕。**沒有和「全部 INT8」比**〔原文，工具回報；判讀〕。
- 原文自己前後不一致：GSE-8 的位元分配 §II-B2 寫 1+3+4，§III-B1 寫 1+4+3〔原文〕。
- 模型只有 LLaMA-2-7B，A100 40GB〔原文 §III〕。

### 1.5 LMCache 的 fp8 serde

- 每個 L2 adapter（存放層）各自選 serde；「one `--l2-adapter` may use fp8 while another stores raw bytes」〔文件 https://docs.lmcache.ai/mp/serde.html〕。
- 文件建議 fp8 用在 disk／remote，說「halves byte volume vs. bf16 with minor accuracy loss」〔文件 同上〕。scale 怎麼算、能不能逐 chunk 選，文件都沒寫〔文件 同上〕。
- 這是「**依層**決定精度」的工程做法，不是依位置〔判讀〕。

### 1.6 只做無損切分的（對照用）

- HCache（EuroSys 2025）：按**層**切重算與載入，全部無損〔原文 p.7、p.13，轉引 D6_HCache 卡〕。
- KVPR（Findings of ACL 2025）：前段從激活重算、後段載 KV，FP16；另有均勻 4-bit 的相容實驗〔原文 p.4–5、p.8，轉引 D6_KVPR 卡〕。

---

## 2. 逐 chunk 混精度，但只在 GPU 內

### 2.1 Cocktail（DATE 2025）

- venue：DOI 10.23919/date64628.2025.10992912〔文件 OpenAlex〕。arXiv HTML 頁是匿名版，沒寫 venue〔原文 arXiv 2503.23294v1〕。
- 精度：query 和每個 chunk 的 Contriever embedding 算餘弦相似度；高於 T_high → FP16，低於 T_low → INT2，其餘 INT4；α＝0.6、β＝0.1〔原文 §III-B，Eq. 1–3〕。
- chunk 32 token；chunk 256 時 Rouge 掉到 16.35（32 時 21.21）〔原文 Table III〕。
- 何時：正式推論前，要先知道 query〔原文 §III〕。
- 量化前依精度把 chunk 重排，原文說 attention 對排列不變〔原文 §III-C，Eq. 4–5〕。原文沒討論位置編碼〔原文〕。
- 增益（LongBench 平均，Table II）：Llama2-7B FP16 44.55、KVQuant 44.25、Cocktail 44.56；Mistral-7B 48.80／48.27／48.74〔原文 Table II〕。TPOT 少 32–52%（只有文字，圖沒讀到）〔原文 §IV-B〕。
- 沒有卸載、重算、前綴重用〔原文〕。

### 2.2 MoQAE（ACL 2025）

- 同一群作者（華中科大＋平安科技）〔原文 p.1〕。DOI 10.18653/v1/2025.acl-long.531〔文件 OpenAlex〕。
- 把 FP16／INT4／INT2 當成 MoE 的「專家」，router 逐 chunk 選；只微調 router〔原文 §3.2–3.3〕。
- **位置規則**：觀察到開頭 token 的 attention 權重較高（前兩層除外），所以第一個 chunk 不進 router，固定 FP16（routing freezing）〔原文 §3.4、Fig. 3〕。
- chunk 32、λ＝0.5、H20 GPU〔原文 §4.1〕。
- LongBench（Table 2）：MultiNews FP16 3.51、KIVI-2b 0.97、MiKV 0.85、MoQAE 3.47〔原文 Table 2〕。Table 2 用哪個模型，我抽到的文字沒寫〔未查證〕。

### 2.3 PM-KVQ（延後降階＋位置規則）

- 「漸進」不是看 token 年齡：每塊先存 16-bit，記憶體快滿才一路降到目標 bit（16→8→4→2）〔原文 §1、§3.1〕。
- 降階用「等效右移」，等於把 2b-bit 反量化再量成 b-bit；捨棄的低位元回不來〔原文 §3.1，Eq. 3；判讀：不可逆〕。
- **位置規則**：最前與最近 128 個 token 保持 16-bit〔原文 §4.1.2〕。
- 逐層位元由一階 Taylor 敏感度＋整數規劃決定；**Qwen 模型的第一層在前 15 層裡最敏感，LLaMA 沒有這現象**〔原文 §3.2、§4.4.2〕。
- 校準：RedPajama arXiv 子集 512 筆 × 2,048 token，用位置內插 s＝4 對應 8,192 長度〔原文 §3.3、§4.1.1、Table 5〕。
- 增益：Qwen-7B 2-bit AIME24：KIVI 32.08 → 40.00；70B：51.88 → 64.79〔原文 Table 2、§4.2〕。
- 沒有卸載或重算〔原文〕。和我們「延後寫入」那條軸同構：先高精度、壓力來了才降〔判讀〕。

### 2.4 DiffKV（SOSP 2025）

- venue：DOI 10.1145/3731569.3764810〔文件 OpenAlex〕。
- 每個 token 的重要性＝後面 token 給它的平均 attention；高於 αh/i → K8V4，介於 → K4V2，否則剪掉〔原文 §4〕。
- **最近 64 個 token 一律高精度**，沒有 sink 的特別處理〔原文 §4〕。
- prompt 階段一次決定；decode 中每步處理一個離開最近視窗的 token，並可把舊 token 降級或剪掉〔原文 §4、§5.3〕。
- **Qwen2.5-7B 對 4-bit K 特別敏感**：K4V8 時準確率「drops to nearly zero」，所以對它關掉低精度〔原文 §7.2、Fig. 8〕。同一節又寫 αl＝0.04，原文前後不一致〔原文 §7.2〕。
- 沒有卸載或重算〔原文〕。

### 2.5 KVTuner（ICML 2025）

- venue：PMLR 267:36451–36485〔文件 https://proceedings.mlr.press/v267/li25dd.html〕。
- 逐層的 K／V 位元配對，離線多目標搜尋，執行時沒有額外開銷〔原文 §5〕。
- Qwen2.5-7B-Instruct：wikitext PPL KV8 9.56、K4V8 220.83、KV4 235.03〔原文 §4.3 Table 2〕。GSM8K：KV8 0.7712、KV4 0.0078、KVTuner-C4.0 0.7559〔原文 Table 5〕。
- 沒測 FP8；引言說 INT8／FP8 KV 在多數實際應用「lossless」〔原文 §1〕。
- 沒有逐 token 位置分析〔原文〕。

### 2.6 SemKV（品質懸崖之上，指標沒差）

- 精度用 8 種模型內部指標排序，相鄰兩級 bit；原文說「Rather than relying on token position or recency heuristics」〔原文 §3.2〕。
- Llama-3.1-8B-Instruct 在 2.322 bit 還和 FP16 一樣，2.0 bit 崩掉〔原文 摘要〕。懸崖之上 8 種指標統計上沒差〔原文 摘要、§4.3〕。
- prompt 在 prefill 後一次決定；生成的 token 每滿 64 個一塊再決定，最近的不完整塊保持 FP16〔原文 §3.4、§3.7、Alg. 1–2〕。
- 品質實驗是 fake quantization（量化再反量化，用 FP16 算）〔原文 §4.1〕。

### 2.7 Minima-KV、TierKV（只讀摘要）

- Minima-KV：最近與受保護的 anchor 頁用 FP8，較舊的非 anchor 頁轉成 TQ3；各格式各自算部分 attention 再合併〔文件 https://arxiv.org/abs/2608.23834〕。這是「依新舊、逐頁」的混精度 paged attention〔判讀〕。anchor 怎麼選，摘要沒寫〔未查證〕。Qwen3.6-27B、RTX PRO 6000〔文件 同上〕。
- TierKV：用 prefill hidden state 預測未來需求，把 token 分到 exact／low-rank／存 flash 三層〔文件 https://arxiv.org/abs/2609.21172〕。這是「寫入時預測、決定表示法與存放位置」〔判讀〕。是否有量化等級、是否重算，摘要沒寫〔未查證〕。

---

## 3. 多輪重用（第二輪以後）的品質

| 論文 | 量化方法 | 多輪怎麼做 | 量到什麼 | 等級 |
|:--|:--|:--|:--|:--|
| SCBench（ICLR 2025） | 只有 KIVI（2-bit、group 32、residual 32） | 多輪模式：同一 session 快取 context；**後續輪用標準答案當上下文**，不是模型自己的答案。多請求模式：跨請求共用 KV | 只報平均。KIVI（τ＝1/8）：Llama-3.1-8B 多輪 32.0（基準 48.7）、多請求 24.8（37.2）；Qwen2.5-72B 39.1（52.9）、35.8（45.8） | B |
| SemKV | 自己的分數 bit 量化 | MT-Eval，每輪對累積的對話做同樣處理 | 只報整段對話平均。均勻 2.0 bit Δ＝−0.082；2.322 bit −0.021；SemKV 2.41 bit −0.008～−0.009 | B |
| xKV（arXiv 2503.18893） | KIVI-2（baseline） | 多輪撈針 64K、壓縮 8× | 逐輪圖只有 SnapKV、PyramidKV、xKV，**沒有 KIVI**；KIVI-2 只在 RULER 64K 報平均：Qwen2.5-1M 90.45（全精度 93.36） | B |
| UltraQuant | 均勻 4-bit | 生產 agent trace 重播 | 只看到吞吐量與 TTFT；品質未見 | C／〔未查證〕 |

- **SCBench 的細節**：
  - venue：ICLR 2025〔文件 https://proceedings.iclr.cc/paper_files/paper/2025/file/a540b17fb2295c736d5afd6c507acf66-Paper-Conference.pdf，經搜尋結果確認〕。
  - KIVI 是唯一的量化方法，歸在「KV cache compression」階段〔原文 §2、Table 1、App. D.1〕。
  - 正文**沒有 KIVI 的逐輪數字**；附錄 Table 10–13 也沒有 KIVI 列〔原文，HTML 全文由工具查過〕。逐輪的圖（Fig. 3a）有沒有 KIVI，文字無法確認〔未查證：圖沒讀〕。
  - 和量化有關的結論只有一句：KIVI 這類「O(n) 記憶體＋稀疏解碼」的方法，在高壓縮率下維持得比較好〔原文 §1 finding 3〕。
  - 「KV 壓縮方法在共用情境普遍表現差，只在第一輪有一點好處」這句講的是壓縮方法整體〔原文 §4〕，是否包含 KIVI 原文沒分開〔判讀〕。
  - 它用標準答案接後續輪，所以多輪的掉分**不是**因為前一輪答錯被帶下去〔判讀：由 §3.2 的設計推〕。
- **SemKV 的歸因只是推測**：它把多輪的差距歸給「對話文字本身的誤差累積」，不是每步重用 cache；但限制段自己說這是從任務分布推的，「rather than directly ablated」〔原文 §4.6、§5〕。
- **評估協定要注意**：有些量化評測在 prefill 時用全精度，量化只影響 decode〔原文 Yuan et al., Findings of EMNLP 2024, §2.2〕。vLLM 原生 FP8 則是 prefill 也從量化後的 cache 讀（見 P04 §2.1）。我們的 harness 只量化「存下、之後載入」的 block，第一輪不受影響〔判讀〕。三種協定量到的不是同一件事，比較時要寫明。

---

## 4. 資訊放在不同深度時的量化評估

- **Yuan et al.（Findings of EMNLP 2024，DOI 10.18653/v1/2024.findings-emnlp.266）**：
  - 撈針：10 種長度（512–20,480 字）× 10 個深度，每格 3 次〔原文 App. B.2〕。
  - KIVI-2／KIVI-4／FlexGen-4bit 在 Llama-3-8B-Instruct 全部 100.0；Mistral-7B-Instruct-v0.2 是 99.0／99.0／98.3〔原文 Table 2〕。**沒有逐深度的量化結果**〔原文，工具回報〕。
  - 64 位數的長 passkey：KIVI-2 從 100% 掉到 91.0%（H2O 掉到 35.0%）〔原文 App. E〕。
  - 量化的 KV 不用在 prefill，只影響 decode〔原文 §2.2〕。
- **SCBench**：Retr.KV 的目標「evenly distributed across the input」，但沒做深度分析〔原文 §3.1〕。
- **xKV**：RULER 64K 的平均，沒有依深度拆〔原文 Table 1〕。
- **KVDiagnosis（arXiv 2608.09412）**：摘要沒提量化，也沒提深度〔文件 https://arxiv.org/abs/2608.09412，C〕。
- **我們自己**：3090 有逐深度撈針（5 深度 × 4 題），每格只有 4 題，不能當訊號〔實測 20260831-181930-m5-needle，見 D6_precision.md §4.5〕。
- **結論**：在這次搜尋範圍內，沒找到「答案深度 × 量化位置」交叉的實驗，也沒找到用多跳／聚合任務（RULER variable tracking、multi-key NIAH）做這個交叉的〔判讀：搜尋範圍見本卡各節；P03 也得到同樣結論〕。

---

## 5. 新穎性威脅排序〔判讀〕

這週要主張的東西（暫定）：在 Cake 式雙向還原下，**同樣位元組、同樣 codec、同樣 attention 路徑**，25% FP8 block 放前段／後段／隨機，品質與延遲有何不同；並逐 block 記錄「重算／載入 BF16／載入 FP8」，把「沒被讀到」和「不敏感」分開。

| 排名 | 前作 | 威脅到哪一句 | 還剩什麼空白 |
|:--|:--|:--|:--|
| 1 | **PatchKV** | 「精度跟重算／載入切分綁在一起」已經有了：重算的不量化，載入的逐 block 混精度 | 切分依編輯位置、不依會合點 b；精度看 attention 風險、不看位置；沒有兩頭並行；收益只有 5.6%（對均勻 INT8） |
| 2 | **SparKV** | 「Cake＋載入段量化」已經是別人的 baseline；而且它明說位置是差勁的代理 | 精度不隨位置變；平台是邊緣裝置；沒量品質對位置 |
| 3 | **CacheGen** | 「逐 chunk 精度＋可退回重算」已經有了 | 讀取時選、要多存幾份；重算是依序的；精度不看位置 |
| 4 | **HA-RAG** | 「已存 KV 逐 chunk 選精度＋多層放置」已經有了 | 依熱度、離線；沒有重算；沒和均勻 8-bit 比品質 |
| 5 | **PM-KVQ** | 「先高精度、壓力來了才降」＋「最前／最近 128 token 保持高精度」 | GPU 內；沒有存放層、沒有重算 |
| 6 | **Minima-KV** | 「依新舊、逐頁的 FP8／低 bit 混合 paged attention」 | 只讀到摘要；GPU 內；沒看到重算 |
| 7 | **Cake** | 「量化讓 b 往前移」不新 | **Cake＋量化的品質是空白**（P01） |
| 8 | Cocktail／MoQAE | 「逐 chunk 混精度」不新；MoQAE 的「第一個 chunk 固定 FP16」就是一條位置規則 | GPU 內；依 query 或學出來的 router |
| 9 | DiffKV／KVTuner／SemKV／ZipCache／QEvict | 「依重要性或逐層選精度」不新 | 都在 GPU 內；SemKV 顯示懸崖之上連隨機都差不多 |
| 10 | LMCache serde、EvicPress、AdaptCache、KVServe | 「依層、依 context、依請求選精度」不新 | 粒度太粗；沒有位置 |

**還站得住的空白**〔判讀〕：
1. 精度依「block 和會合點 b 的相對位置」決定——沒找到。
2. 在 Cake 式還原下量**品質**，並把「重算掉了所以沒讀到」和「讀到了但不敏感」分開——沒找到。
3. 同樣位元組下的前／後／隨機對照——沒找到（P03 同結論）。
4. 多輪重用時逐輪報量化 KV 的品質——沒找到；SCBench、SemKV 都只報平均。

**要誠實面對的反方證據**〔判讀〕：
- SparKV 說位置是差勁的代理；SemKV 說懸崖之上隨機和各種指標差不多。所以「隨機 25%」很可能是個強 baseline，預測要先寫死。
- PatchKV 的混精度只比均勻 INT8 快 5.6%。我們的收益量級可能也一樣小。
- 若 Qwen2.5 的全 FP8 因為 K 偏移而失敗（P04 §4.1），位置怎麼放都救不了，那時該換的是 codec，不是位置。

---

## 6. 沒查到的

- PatchKV、HA-RAG 的 venue〔未查證〕。
- Minima-KV、TierKV、HqeKV、MixKVQ、UltraQuant 的全文〔未查證〕。
- SCBench 逐輪圖裡 KIVI 的曲線〔未查證〕。
- Cocktail 的 TPOT 圖、MoQAE Table 2 的模型名〔未查證〕。
- Semantic Scholar 的 Cake 引用清單有 39 篇，我只對標題做了篩選；HA-RAG 是從這份清單找到的〔文件 Semantic Scholar API，2026-10-13〕。清單外的引用沒查。
