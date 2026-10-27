# LA_ssm_structure：Mamba 類模型為什麼快，這些線性代數技巧能不能搬到 softmax attention 的 KV 還原上

> 2026-10-27，文獻 agent litA。規則見同資料夾 `RULES.md`。
> 問題（使用者原話大意）：「Mamba 用對角化之類的線性代數技巧變快，我們的 KV cache 還原／重算能不能也用？」
> 範圍：只做理論面與文獻證據；RoPE（區塊對角旋轉）與 pre-RoPE K 低秩由另一個 agent 負責，本檔只在 §3.4 簡短指路。
> 暫存（PDF、pdftotext 文字檔、搜尋腳本）：`<SCRATCH>/litA/pdf/`、`<SCRATCH>/litA/pg.py`。本檔沒有在 Mac 上跑任何實驗；只有〔算術〕。

**證據等級用法（本檔）**：〔部分〕＝下載 arXiv PDF、用 `pdftotext -layout` 讀了卡片列出的頁與節，數字都對照 PDF 文字抄；沒有一篇是從頭到尾逐頁讀完，所以本檔沒有標〔全文〕。〔二手〕＝數字出自另一篇論文的表。頁碼一律是 **PDF 頁**（不是印刷頁碼，兩者不同時會註明）。

---

## §0 一頁結論（白話）

1. **Mamba 的「快」不是事後對一個現成矩陣做對角化，而是模型在設計時就規定好「狀態轉移矩陣 A 必須是對角的（Mamba-1）或純量乘單位矩陣（Mamba-2）」，再從頭訓練。** 有了這個結構，每個 token 只要更新一個固定大小的狀態，總成本跟長度成線性；S4 要用「正規＋低秩、Woodbury、Cauchy kernel」那一大套，是因為它的 A（HiPPO 矩陣）直接對角化在數值上會爆（S4 §3.1，PDF p.5）。結構是**訓練出來的前提**，不是可以套在任意模型上的加速法。
2. **Mamba-1 的關鍵是「讓參數跟輸入有關」（selective）**——這一步其實**打破**了 S4 那套「對角化 → 卷積 → FFT」的捷徑（Mamba §3.2，PDF p.5），所以它改回遞迴、用平行 scan＋把狀態只放在 SRAM＋反向時重算，靠的是 GPU 記憶體階層的工程，而不是更多線性代數（§3.3.2，PDF p.7）。
3. **Mamba-2（SSD）講得最清楚：SSM＝半可分（semiseparable）矩陣乘法**，下三角裡任何子塊的秩都 ≤ N（Def. 3.1，PDF p.7）。於是可以切 chunk：chunk 內用矩陣乘法精確算，chunk 之間只傳一個 N×P 的狀態（§6，PDF p.18–22）。**原文自己說 SSD「不涵蓋標準 softmax attention」，也不涵蓋任何沒有有限維特徵映射的 attention**（§10.3，PDF p.38）；而且「能有效率地自迴歸」的遮罩 attention 必然是半可分的（Thm 5.2，PDF p.17）。
4. **softmax 的 exp 就是卡關點**：exp(q·k) 對應的特徵映射是無限維的，所以「精確的 softmax 無法線性化」（Katharopoulos et al. §3.2.1，PDF p.3）。fine-grained complexity 給了一條很硬的界線：Q、K 的元素若被 o(√log n) 限住，就有 n^{1+o(1)} 的近似演算法；若到 Θ(√log n)，在 SETH 假設下**不可能**真正次二次（Alman & Song Thm 1.3/1.4，PDF p.3）。〔算術＋判讀〕要讓一個 needle 在 n＝128K 個 token 中拿到一半以上的注意力，logit 差至少要 ln(n−1)≈11.8——這正好落在「快速演算法不成立」的區域。**也就是說：能被便宜近似的 attention，恰好是不需要精準檢索的 attention。**
5. **固定大小狀態的模型在「檢索／複製」上有下界**：任何固定狀態的模型要複製長度 L 的字串，狀態必須隨 L 線性長（Jelassi et al. Thm 2.7、Cor 2.8，PDF p.4）；任何遞迴模型要解 MQAR 需要 Ω(N) bits 的狀態（Based Thm 3.1，PDF p.4）。〔算術〕Llama-3.1-8B 若每層每個 KV head 用一個 128×128 的線性注意力狀態，整個模型是 8 MiB（BF16），只等於 64 個 token 的 KV；128K 的 KV 是它的 2,048 倍——資訊量上不可能無損。
6. **不訓練直接換，證據都是掉很多**：Performer 把預訓練 Transformer 的權重直接搬過去，起始準確率只有 0.07，要再微調才回來（Performer Fig. 5，PDF p.7）；HyperAttention 把 chatglm2-6b-32k 的 28 層全部換掉，LongBench 的 synthetic 任務從 84.00 掉到 8.33、single-doc QA 從 80.63 掉到 27.07（Table 1，PDF p.10）。唯一「免訓練又幾乎不掉」的 LightTransfer，**根本沒有線性化**：它只是把一半「本來就只看開頭和最近 token」的層改成 streaming attention，仍是 softmax（Llama-3-8B-Instruct LongBench 平均 40.8 → 38.7，Table 2，PDF p.8）。
7. **輕度轉換（幾千萬到幾百億 token 的再訓練）在短 context 上接近原模型，但長 context 檢索是弱點，而且沒有人在 Llama-3.1-8B／Qwen2.5 的 128K 上證明過**：LoLCATs 只用 40M token，但用 Alpaca 線性化的 Llama-3-8B 在 8K passkey 上是 0%（Table 21，PDF p.28）；STILL 的表上 LoLCATs、Liger 在 RULER-4K 平均只有 7.2、2.1（Table 3，PDF p.9）；Mamba-in-the-Llama 保留 50% attention 層才在 NIAH 做到 8B 完美到 16K（PDF p.11）。真正在長 context 贏過 full attention 的（Kimi Linear）是**從頭預訓練 1.4T token**，而且還保留 1/4 的 full attention 層（PDF p.1–2）。
8. **對 Tiara 的意涵（判讀）**：「對角化／線性化 softmax attention 來加速還原或重算」這條路**不建議走**：(a) 我們的 Llama／Qwen 沒有那個結構；(b) 近似 attention 重算出來的 KV 是有損的，而且就算 attention 無限快，128K 的 prefill 理論上最多只快 3.1 倍（Amdahl，〔算術〕§5.3）。真正搬得過來的是**SSD／STILL 的「結構」：少數重要的部分精確算、其餘用壓縮表示**——這跟我們的「縮圖還原」是同一個形狀，而且 STILL 的資料剛好說明「檢索能力只有在被選中的 token 保留精確 softmax 時才回來」。另外，hybrid 模型（GDN／KDA／Mamba2 層）讓「還原」對象多了一個固定大小的狀態快照，這是 repo 既有整理已經在追的方向（見 §5.4）。

---

## §1 SSM 這條線：結構從哪來、為什麼快

### S4：Efficiently Modeling Long Sequences with Structured State Spaces（arXiv 2111.00396v3；venue 〔未查證〕，使用者稱 ICLR 2022）
- **讀了什麼**：〔部分〕v3（2022-08-05），https://arxiv.org/abs/2111.00396 ，讀 PDF p.1–7（§2.3–§3.4、Table 1），查證 2026-10-27
- **一句話**：把連續時間 SSM 算得又快又穩，用來做長序列建模。
- **線性代數的關鍵一步**：
  - SSM 可以換基底：(A,B,C) ~ (V⁻¹AV, V⁻¹B, CV)（Lemma 3.1，PDF p.5）。若 A 是對角，卷積核 K 就是 Vandermonde 乘法（PDF p.5）。
  - **但 HiPPO 矩陣直接對角化，元素會隨狀態大小 N 指數變大，數值上不可行**（§3.1，PDF p.5）。所以只能用條件數好的（unitary）V 去轉，而能被 unitary 對角化的正好是 normal 矩陣（§3.2，PDF p.5）。
  - 解法：HiPPO＝normal＋低秩（NPLR），A = VΛV* − PQᵀ，秩 r＝1 或 2（Thm 1，式 (6)，PDF p.6）→ 轉成對角＋低秩（DPLR）。卷積核改在頻域算：低秩項用 Woodbury 恆等式消掉，剩下對角部分＝Cauchy kernel，最後 iFFT（Algorithm 1，PDF p.6）。
  - 複雜度：遞迴一步 O(N)（Thm 2）；卷積核 Õ(N+L) 時間、O(N+L) 空間（Thm 3，PDF p.6）。
- **成立的前提**：模型本身就是**線性非時變（LTI）**的 SSM，而且 A 被規定成 HiPPO／DPLR 形式、從頭訓練（§3.4：用 HiPPO 初始化，5N 個可訓練參數，PDF p.7）。對 Llama／Qwen 的 softmax attention **不適用**：attention 不是固定的線性運算子，沒有一個 A 可以拿來對角化〔判讀，依據見 §3〕。
- **量到的數字**：比 LSSL 快最多 30 倍、記憶體少 400 倍（PDF p.2）。Table 1 列出各模型的複雜度（PDF p.7）。
- **跟 Tiara 的關係**：無直接關係。唯一可借的觀念是「直接對角化在數值上可能不可行，要找條件數好的分解」——之後若有人提議對 KV 做特徵分解，這是要先檢查的地方〔判讀〕。
- **與既有整理的關係**：repo 沒有整理過 S4（`grep -rli "DPLR\|S4 " docs` 無相關結果，`S4D` 只出現在 `docs/report-3090-20260905.html` 的 base64 圖片字串裡，非真提及）。

### S4D：On the Parameterization and Initialization of Diagonal State Space Models（arXiv 2206.11893v2；venue 〔未查證〕）
- **讀了什麼**：〔部分〕v2（2022-08-05），https://arxiv.org/abs/2206.11893 ，讀 PDF p.1–8（§3、§4、Listing 1），查證 2026-10-27
- **一句話**：證明「拿掉 S4 的低秩項、只留對角」也行，並把對角 SSM 簡化到幾行程式。
- **線性代數的關鍵一步**：
  - 對角 A 時，卷積核 K = (B∘C)·V_L(A)，V_L 是 Vandermonde 矩陣（式 (7)，PDF p.5）；理論 Õ(N+L)，GPU 上直接 O(NL) 加總但不展開矩陣（PDF p.5）。
  - 實數 SSM 對角化到複數時特徵值成共軛對，所以只存一半、取兩倍實部（PDF p.6）。
  - 可對角化矩陣在 ℂ^{N×N} 中稠密且測度為滿（Prop. 2，PDF p.7）→「幾乎所有 SSM 都等價於對角 SSM」。
  - 但原文明說：**這只是表達力，不保證訓練後效果好**；隨機初始化的稠密實矩陣或對角複矩陣都表現差；也不處理數值表示問題（PDF p.8）。Thm 3：HiPPO-LegS 的對角部分在 N→∞ 時重現原本的基函數（PDF p.8），這是這個特殊矩陣的性質，一般 DPLR 不成立。
- **成立的前提**：模型要用對角 A **訓練**，且初始化很關鍵（§4 標題：「which diagonal state matrices A are actually effective?」，PDF p.7）。
- **量到的數字**：LRA 平均 85%（摘要，PDF p.1）。
- **跟 Tiara 的關係**：無。觀念重點：「可以被對角化」≠「對角化後好用」，後者要靠訓練與初始化〔判讀〕。
- **與既有整理的關係**：repo 無。

### DSS：Diagonal State Spaces are as Effective as Structured State Spaces（arXiv 2203.14343v3；venue 〔未查證〕）
- **讀了什麼**：〔部分〕v3（2022-05-18），https://arxiv.org/abs/2203.14343 ，讀 PDF p.3–4（Prop. 1、Algorithm 1），查證 2026-10-27
- **一句話**：第一個發現純對角 SSM 可以跟 S4 一樣好。
- **線性代數的關鍵一步**：Prop. 1（PDF p.3）：對可對角化的 A，卷積核可以寫成 w·Λ⁻¹·row-softmax(P)，P_{ik}=λ_i·k·Δ——不再有矩陣冪次，只剩結構化的矩陣–向量乘。注意這裡的 **softmax 是沿時間軸對 exp(λkΔ) 做正規化以防溢位**（§A.2，PDF p.4），跟 attention 的 softmax 無關。
- **成立的前提**：從頭訓練的 LTI SSM；DSS-EXP 版要求特徵值實部為負，否則長輸入訓練不穩（PDF p.3–4）。
- **量到的數字**：本次未抄（不影響結論）。
- **跟 Tiara 的關係**：無。提這篇只為避免誤會：「SSM 裡也有 softmax」不代表 softmax attention 是 SSM。
- **與既有整理的關係**：repo 無。

### Mamba：Linear-Time Sequence Modeling with Selective State Spaces（arXiv 2312.00752；venue 〔未查證〕）
- **讀了什麼**：〔部分〕arXiv 下載版（PDF 無版本戳，下載 2026-10-27），https://arxiv.org/abs/2312.00752 ，讀 PDF p.1–7（§2、§3.1–3.4）、p.26（§4.5）、p.45（附錄 D），查證 2026-10-27
- **一句話**：讓 SSM 的參數依輸入變化，補上 LTI 模型做不到的「選擇性記憶」。
- **線性代數的關鍵一步**：
  - **對角**：A 用對角結構（沿用 S4D／DSS），A∈ℝ^{N×N}、B、C 都只要 N 個數（§2「Structure and Dimensions」，PDF p.4）。每個 channel 是 N 個獨立的一維遞迴，所以一步更新是 O(N)、總成本 O(BLDN)（PDF p.7）。
  - **打破 LTI**：Δ、B、C 變成輸入的函數，模型從時不變變成時變，「失去與卷積的等價性」（§3.2，PDF p.5）。所以 S4／S4D 的 FFT 卷積路徑不能用了。
  - **用什麼補**：三個「經典技巧」——kernel fusion、平行 scan、重算（§3.3.2，PDF p.7）：(B,L,D,N) 大小的展開狀態**只在 SRAM 裡**存在，HBM 只讀寫 (B,L,D) 大小的輸入輸出；遞迴雖然是時變的，仍是結合律可平行的 scan（Blelloch）；反向傳播不存中間狀態、重讀輸入時重算。IO 因此少 O(N) 倍（附錄 D，PDF p.45）。
- **成立的前提**：模型從頭用這個結構訓練（Mamba-3B 等在 Pile 上預訓練）。對角 A 的作用是讓狀態更新是逐元素的、能放進 SRAM 平行 scan；這不是對任意模型可用的加速〔判讀〕。
- **量到的數字**：scan 比 PyTorch 標準實作快 20–40 倍、在 2K 以上比 FlashAttention-2 快；推論吞吐比同大小 Transformer 高 4–5 倍，原文寫明原因是「沒有 KV cache 所以 batch 可以大很多」（§4.5，PDF p.26）；scan 在 32K 時比 attention 快到 7 倍（附錄 D，PDF p.45）。
- **跟 Tiara 的關係**：Mamba 的「重算」是**訓練時反向傳播**的重算（不存中間狀態），跟我們的「KV 還原時重算」是不同層次的東西，名字一樣而已〔判讀〕。真正相關的是：Mamba 的 decode 吞吐優勢來自「不存 KV」——這正是我們的 dense GQA 模型沒有的性質。
- **與既有整理的關係**：`docs/research_20261009_explore/lit_C_architectures.md` L37（scan 的 FLOPs 粗估）、L169–173（狀態大小）、L282–283（vLLM 的 selective_scan／SSD kernel）已整理 Mamba 系在 vLLM 上的狀態與 kernel；本卡只補「為什麼結構讓它快」。

### Mamba-2／SSD：Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality（arXiv 2405.21060；venue 〔未查證〕，使用者稱 ICML 2024）
- **讀了什麼**：〔部分〕arXiv 下載版（PDF 無版本戳，下載 2026-10-27），https://arxiv.org/abs/2405.21060 ，讀 PDF p.2、4–10（§2–§3）、12–13（§4.1.3）、16–22（§5–§6、Listing 1）、33（§9.2.3）、36（§9.4.3）、38（§10.3），查證 2026-10-27
- **一句話**：證明「某一類 SSM」與「某一類遮罩線性注意力」是同一個矩陣的兩種算法，並據此寫出主要靠矩陣乘法的 kernel。
- **線性代數的關鍵一步**（這是本檔最重要的一張卡）：
  1. **SSM＝半可分矩陣**：把 SSM 寫成 y = Mx，M_{ji} = C_jᵀ A_j⋯A_{i+1} B_i（式 (3)，PDF p.7）。Def. 3.1：下三角矩陣 M 是 N-semiseparable，若下三角部分（含對角）的**每個子矩陣秩都 ≤ N**（PDF p.7）。Thm 3.5：狀態大小 N 的 SSM 等於乘一個 N-SS 矩陣（PDF p.9）；Thm 3.7：任何狀態大小 N 的 SSM 可在 O(TN) 時間算完（PDF p.10）。
  2. **A 再簡化成「純量×單位矩陣」**：Mamba-1 是對角，Mamba-2 讓每個 A_t 只是一個純量；head 維度 P 從 1 放大到 64 或 128（§2.4，PDF p.5）。Cor. 5.1：1-半可分遮罩的注意力（1-SS SMA）正是「對角 A 且對角元素全相同」的 SSM（PDF p.17）。
  3. **chunk 分解**：把 T×T 的 M 切成 (T/Q)×(T/Q) 個 Q×Q 塊；**非對角塊依定義是低秩的**（PDF p.18）。對角塊（chunk 內）用二次的「attention 形式」以矩陣乘法算；非對角塊拆成右因子（B 項）、中間因子（chunk 間的純量 SSM 遞迴，長度只有 T/Q）、左因子（C 項）（§6.1–6.2，PDF p.19–22）。Thm 6.1：O(TN²) 訓練 FLOPs、O(TN) 推論 FLOPs、O(N²) 推論記憶體，且工作量以矩陣乘法為主（PDF p.17–18）。chunk 間的 scan 成本比 Mamba 的完整 scan 小 Q 倍，「可忽略」（PDF p.22）。
- **原文對 softmax attention 怎麼說**（使用者特別要的，逐條附節號）：
  - §2.2（PDF p.6）：SSD 相對於標準 softmax attention 有兩個差別——**softmax 被拿掉**、attention 矩陣逐元素乘上一個遮罩 L。
  - §4.1.3 Kernel Attention（PDF p.12）：softmax 可拆成「取 exp」與「沿 S 軸正規化」兩步；exp 項可視為核轉換，存在一個（**無限維的**）特徵映射 φ 使 exp(QKᵀ)=φ(Q)φ(K)ᵀ。之後整篇都把 Q、K 當成「已經過有限特徵映射」的版本，**也就是在這一步就離開了 softmax**。
  - Thm 5.2（PDF p.17）：任何「有界階數自迴歸」的結構化遮罩注意力，其遮罩 L **必須是半可分矩陣**——原文的話是「efficient autoregressive attention is general semiseparable SMA」。
  - §10.3（PDF p.38，印刷頁 35）：SSD 不用 softmax，「這就是 attention 二次複雜度的來源」；拿掉 softmax 才能走線性注意力的線性路徑；半可分遮罩「可視為取代了 softmax」。並明寫 **SSD 不涵蓋標準 softmax attention，也不涵蓋任何沒有有限特徵映射 ψ 的 attention 核矩陣轉換**；SSD 的優點是用 N 壓縮歷史，對比 attention 快取整段歷史（「T ≫ N」）。
  - §9.4.3（PDF p.36）：他們試了多種「近似 softmax 的 exp 核」的 kernel feature map，都沒有比簡單的好；原文自己解釋：那些線性注意力是為了近似「沒有 1-SS 遮罩」的 softmax attention 而設計的，所以負結果「或許不意外」。
- **成立的前提**：A 的結構（純量×I）是模型定義的一部分、從頭訓練。對 softmax attention：**原文明確說不適用**（§10.3）。
- **量到的數字**：SSD 實作比 Mamba 的 selective scan 快 2–8 倍；跟 FlashAttention-2 在 2K 交叉，16K 時快 6 倍（摘要，PDF p.2）。混合模型：約 10% 的層是 attention 最好；他們的假設是 SSM 層做一般序列映射、attention 層當「檢索機制」，免得把整段 context 壓進狀態（§9.2.3，PDF p.33；350M、48 層、Pile 7B token）。
- **跟 Tiara 的關係**〔判讀〕：
  - SSD 的 chunk 分解「chunk 內精確、chunk 間只傳壓縮狀態」在**形狀上**跟縮圖還原（重要 block 拿原版、其他用縮圖）很像；但 SSD 的「chunk 間低秩」是模型被訓練成半可分才成立的**精確等式**，softmax attention 的非對角塊沒有這個保證（§10.3）。所以 Tiara 能借的是「挑出要精確的部分」這個策略形狀，不是 SSD 的數學。
  - §9.2.3 的「attention 層＝檢索機制」也間接說明：檢索這件事本來就是 SSM 壓縮狀態做不好的，所以才要留 attention 層。
- **與既有整理的關係**：`lit_C_architectures.md` L283 已記 vLLM 的 SSD Triton kernel 檔名（`ssd_*.py`）；L37 有 Mamba2 scan 的 FLOPs 粗估。本卡補的是理論結構與原文對 softmax 的陳述，repo 先前沒有（`grep -rli semiseparable docs` 無結果）。

---

## §2 線性注意力與 delta rule：結構從哪來

### LinAttn：Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention（arXiv 2006.16236v3；venue 〔未查證〕，使用者稱 ICML 2020）
- **讀了什麼**：〔部分〕v3（2020-08-31），https://arxiv.org/abs/2006.16236 ，讀 PDF p.1–4（§3.1–§3.2），查證 2026-10-27
- **一句話**：把 attention 的相似度換成核函數，利用矩陣乘法結合律讓自迴歸 Transformer 變成 RNN。
- **線性代數的關鍵一步**：sim(q,k)=φ(q)ᵀφ(k)，再用結合律把 (φ(Q)φ(K)ᵀ)V 改成 φ(Q)(φ(K)ᵀV)（式 (4)–(6)，PDF p.3）；Σφ(k_j)v_jᵀ 只算一次、每個 query 重用，所以時間與記憶體是 O(N)。狀態是 C×M（特徵維度 × value 維度），不隨長度長。φ 用 elu(x)+1（式 (7)，PDF p.4）。
- **為什麼 softmax 不能這樣做**（原文）：**指數核對應的特徵函數是無限維的，所以精確 softmax attention 的線性化不可行**；多項式核有有限維的精確特徵映射（§3.2.1，PDF p.3）。
- **成立的前提**：用 φ 取代 softmax **重新訓練**。對預訓練的 softmax 模型直接換 φ，等於換了一個不同的函數〔判讀〕。
- **量到的數字**：長序列自迴歸預測最多快 4000 倍（摘要，PDF p.1）。
- **跟 Tiara 的關係**：解釋「為什麼不能把 Llama 的 KV 壓成一個 d×d 狀態」——那需要 exp 有有限維分解，而它沒有（〔判讀〕，依據即上段）。
- **與既有整理的關係**：repo 中「linear attention」多出現在 hybrid 模型的服務整理（如 `docs/research_20261009_explore/cards/C_KVBuffer.md`、`C_TailReplay.md`、`docs/kv_sharing_hiding_20261026/lit/L1_strata_and_engines.md`），沒有整理這篇原始論文的推導。

### DeltaNet 平行化：Parallelizing Linear Transformers with the Delta Rule over Sequence Length（NeurIPS 2024，PDF 首頁有寫；arXiv 2406.06484v6）
- **讀了什麼**：〔部分〕v6（2025-01-15），https://arxiv.org/abs/2406.06484 ，讀 PDF p.1–6（§2–§3.2、Fig. 1），查證 2026-10-27
- **一句話**：把「delta rule」線性 Transformer 的訓練改成可沿序列平行、主要用矩陣乘法。
- **線性代數的關鍵一步**：
  - DeltaNet 的狀態更新 S_t = S_{t−1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ，原文指出這是對前一狀態施加一個**廣義 Householder 變換**（單位矩陣＋秩一）（§3.1，PDF p.4）。
  - Householder 乘積可用 **WY 表示法**緊湊表示，不必在每一步展開 d×d 狀態（PDF p.2、p.4）→ 可套用線性注意力既有的 chunkwise 平行形式。
  - chunk 內原本是遞迴的式 (7) 不能用 tensor core，所以再用 **UT 變換**：T = (I + tril(diag(β)KKᵀ, −1))⁻¹ diag(β)，W = TK、U = TV（式 (10)–(11)，PDF p.5）；下三角反矩陣用前向代換解。
  - 完全平行形式要對整段長度求反矩陣，成本是長度的三次方，所以不用（PDF p.6）。
- **成立的前提**：模型是 DeltaNet 架構、從頭訓練（1.3B、100B token，PDF p.1）。
- **量到的數字**：chunkwise 相對遞迴形式的加速隨長度與 head 維度增加（Fig. 1，PDF p.5；d=2048，batch×長度＝16384）——本次沒抄具體倍數。原文也寫線性 Transformer 在需要 in-context 檢索的任務上仍落後 Transformer（摘要，PDF p.1）。
- **跟 Tiara 的關係**：無直接關係。觀念：這類「快」都靠「狀態轉移是單位＋低秩／對角」這種被設計出來的結構〔判讀〕。
- **與既有整理的關係**：repo 多處提 DeltaNet／GDN（`docs/PAPERS_BY_LEVEL.md` L71、`docs/DIRECTIONS_20260927.md` L220、`docs/research_20261006_eval/cards/E03_restore_recompute.md` L130），都是**服務端狀態快取**的角度；本卡只補演算法結構。

### GDN＋KDA：Gated Delta Networks（ICLR 2025；arXiv 2412.06464v3）與 Kimi Linear（arXiv 2510.26692v2，ID 已核對）
- **讀了什麼**：GDN〔部分〕v3（2025-03-06），https://arxiv.org/abs/2412.06464 ，讀 PDF p.1–3；Kimi Linear〔部分〕v2（2025-11-01），https://arxiv.org/abs/2510.26692 ，讀 PDF p.1–6、8–10，查證 2026-10-27
- **一句話**：GDN 把 Mamba-2 的「整體衰減」與 delta rule 的「定點改寫」合在一起；KDA 再把衰減改成逐 channel。
- **線性代數的關鍵一步**：
  - Mamba-2 的更新 S_t = α_t S_{t−1} + v_t k_tᵀ（α 是純量）；GDN 的 gated delta rule：S_t = α_t (I − β_t k_t k_tᵀ) S_{t−1} + β_t k_t v_tᵀ（Kimi PDF p.3 的寫法；GDN PDF p.1–3）。
  - KDA：S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t−1} + β_t k_t v_tᵀ（PDF p.4）——轉移矩陣是**對角＋低秩（DPLR）的特化版**；原文說它比一般 DPLR 的 chunkwise 演算法省很多計算（摘要與 §3，PDF p.1、p.4–5）。
- **成立的前提**：全新架構、從頭預訓練。Kimi Linear 是 KDA 與 MLA 以 **3:1** 交錯的 hybrid（PDF p.2），MLA 層用 NoPE（PDF p.6）。公平比較的訓練量是 **1.4T token**（PDF p.1–2）；48B 總參數、3B 啟用（摘要）。
- **量到的數字**（只抄原文）：KV cache 最多少 75%、1M context 下 decode 吞吐最多 6 倍（摘要，PDF p.1）；Fig. 1：RULER 128K 得分 84.3、3.98× 加速；1M 時 TPOT 1.84 ms vs MLA 11.48 ms（6.3×）（PDF p.1）。
- **跟 Tiara 的關係**〔判讀〕：75% 的 KV 縮減**來自 3/4 的層沒有 KV**（repo 已用 config 算出 1−7/27＝74%），不是對既有 KV 做任何線性代數。要得到這個好處必須換模型、重新預訓練——不在我們「Llama／Qwen 不改模型」的範圍內。
- **與既有整理的關係**：`docs/stage3_20261019/lit/L6_mainstream_practice.md` L212–215（Kimi Linear〔摘要＋config〕、74% 算術）、L62；`docs/stage3_design_20261024/lit/L8c_practice_and_trends.md` L34、L127、L256。本卡把 L6 的〔摘要〕升級為〔部分〕：Fig. 1 的 RULER-128K 與 TPOT 數字、1.4T token、NoPE 是本次新抄的；與既有整理無衝突。

---

## §3 能不能搬到 softmax attention

> 白話結論：**不能免費搬**。理論上，softmax 的 exp 沒有有限維分解；複雜度理論說「需要精準檢索的 attention」不可能被真正次二次地近似；實驗上，免訓練替換都大掉，輕度再訓練在長 context 檢索上也撐不住。

### 3.1 理論界線

#### Alman & Song：Fast Attention Requires Bounded Entries（arXiv 2302.13214v2；venue 〔未查證〕）
- **讀了什麼**：〔部分〕v2（2023-05-09），https://arxiv.org/abs/2302.13214 ，讀 PDF p.1–4（摘要、Def. 1.1–1.2、Thm 1.3–1.4、技術概述），查證 2026-10-27
- **一句話**：用 fine-grained complexity 刻出「attention 何時能近乎線性時間近似」。
- **線性代數的關鍵一步**：
  - 問題定義：Q,K,V ∈ [−B,B]^{n×d}，A = exp(QKᵀ/d)（**注意：除以 d，不是 √d**），要在 1/poly(n) 的加性誤差內輸出 D⁻¹AV（Def. 1.1–1.2，PDF p.2–3）。
  - 上界（Thm 1.4，PDF p.3）：d = O(log n)、B = o(√log n) 時有 n^{1+o(1)} 演算法。方法是 **polynomial method**：若 M 低秩且 f 可被低次多項式逼近，則 f(M) 接近一個可快速算出分解的低秩矩陣；用 exp 的多項式逼近得到 A 的低秩近似（PDF p.4）。
  - 下界（Thm 1.3，PDF p.3）：d = C log n、B = C_b √log n 時，在 SETH 假設下沒有 O(n^{2−q}) 演算法。
  - 原文的解讀：這解釋了為什麼量化、低次多項式近似在實務上有效——**當元素小的時候**（PDF p.3–4）。
- **成立的前提**：純理論；最壞情況與特定參數區間（d = Θ(log n)）。對真實模型的 d=128 是否落在這個區間，原文沒有討論〔判讀〕。
- **〔算術＋判讀〕跟檢索的關係**：Alman–Song 的 logit 是 QKᵀ/d，每個元素絕對值 ≤ d·B²/d = B²。快速區間 B = o(√log n) ⇔ logit = o(log n)。另一方面，若一個 needle 的 logit 比其他 n−1 個高 Δ，它拿到的權重是 e^Δ/(e^Δ + n − 1)；要 ≥ 1/2 需要 Δ ≥ ln(n−1)；n = 131,072 時 ln(131,071) ≈ 11.78，n = 262,144 時 ≈ 12.48。所以「在長 context 裡把注意力集中到單一 token」需要 Θ(log n) 等級的 logit，**正好就是下界所在的區間**。白話：可以被便宜近似的 attention，就是那種平均分散、不需要精準挑中某個 token 的 attention。這是我的推論，不是原文的結論。
- **跟 Tiara 的關係**：回答「能不能免費把 softmax 線性化／對角化」——一般情況下不能，而 Tiara 最在意的「長 context 中找到少數關鍵 block」正好是難的那一類〔判讀〕。
- **與既有整理的關係**：repo 無（`Alman` 只在 results CSV 中誤命中）。

#### Repeat After Me（Jelassi et al.）：Transformers are Better than State Space Models at Copying（arXiv 2402.01032v2；venue 〔未查證〕，使用者稱 ICML 2024）
- **讀了什麼**：〔部分〕v2（2024-06-03），https://arxiv.org/abs/2402.01032 ，讀 PDF p.1–4，查證 2026-10-27
- **一句話**：證明並驗證固定大小狀態的模型在複製／從 context 取資料上先天受限。
- **關鍵結果**：
  - 定義 GSSM：任何狀態空間有限、狀態大小不隨長度變的序列模型，涵蓋 S4、Mamba、線性注意力、RNN（PDF p.1–2）。
  - Thm 2.3：深度 2、維度 O(n log D) 的 Transformer 可以複製指數長度的字串（PDF p.4）。
  - Thm 2.7：任何 GSSM 在長度 L 的均勻複製分布上誤差 > 1 − |S|/D^L；Cor. 2.8：狀態 bits < L log D − 1 時誤差 > 1/2（PDF p.4）。也就是**狀態必須隨長度線性成長**。
  - 實驗：預訓練模型 1-shot 做「電話簿查詢」，Pythia（Transformer）在各規模都大幅勝過 Mamba（Fig. 1(c) 說明，PDF p.2）。
- **跟 Tiara 的關係**：把 KV 換成固定大小狀態＝放棄精確檢索，這不是工程問題而是資訊量問題〔判讀〕。
- **與既有整理的關係**：repo 無。

#### Zoology（arXiv 2312.04927v1）與 Based（arXiv 2402.18668v2）
- **讀了什麼**：Zoology〔摘要〕v1，https://arxiv.org/abs/2312.04927 （只讀 PDF p.1 摘要）；Based〔部分〕v2（2025-03-07），https://arxiv.org/abs/2402.18668 ，讀 PDF p.1–5，查證 2026-10-27
- **一句話**：量化「狀態大小 vs 召回能力」的取捨，並給下界。
- **關鍵結果**：
  - Zoology（摘要）：gated-convolution 模型與 attention 的 Pile 困惑度差距中，82% 由 in-context 召回能力解釋；70M 的 attention 模型在聯想召回上勝過 1.4B 的 gated-convolution 模型；提出 MQAR 任務。
  - Based Thm 3.1（PDF p.4）：任何因果的遞迴模型要解 MQAR，狀態需要 **Ω(N) bits**；原文說這表示 Fig. 2 觀察到的取捨「是根本的，不是架構的怪癖」。Based 用 2nd-order Taylor 近似 exp 當線性注意力特徵映射，再加滑動視窗（PDF p.2、p.5）。摘要：attention 在真實召回密集任務上比 Mamba 高 32.2 分（Table 1），Based 比次二次模型高 10.36 分（PDF p.1）。
- **跟 Tiara 的關係**：同上——召回需要的記憶體與 context 長度成正比；KV cache 就是這個「Ω(N)」〔判讀〕。
- **與既有整理的關係**：repo 無。

### 3.2 免訓練的近似 softmax

#### Performer：Rethinking Attention with Performers（ICLR 2021，PDF 頁首有寫；arXiv 2009.14794v4）
- **讀了什麼**：〔部分〕v4（2022-11-19），https://arxiv.org/abs/2009.14794 ，讀 PDF p.1–7、9，查證 2026-10-27
- **一句話**：用正的正交隨機特徵（FAVOR+）無偏地估計 softmax 核，讓 attention 線性化。
- **線性代數的關鍵一步**：Lemma 1（PDF p.4）：SM(x,y) = E_ω[exp(ωᵀx − ‖x‖²/2)·exp(ωᵀy − ‖y‖²/2)]，ω～N(0,I)——用隨機的 **有限維**特徵 **估計**無限維的 exp 核；隨機向量用 Gram-Schmidt 正交化以降變異（PDF p.5）。Thm 4（PDF p.6）：所需隨機投影數 m 與序列長度 L 無關，但與 q/k 範數上界 R 有關（透過 h* = exp(l²/2)、l = R·d^{−1/4}；pdftotext 擷出的公式排版不全，此處只記依賴關係）。
- **成立的前提**：原文自己寫：即使 attention 近似很緊，小誤差也會經多層放大，所以需要「少量微調」才能與預訓練模型相容（§4.3，PDF p.7）。**注意原文前後不一致**：§4.3 說要微調，結論 PDF p.9 卻說「即使不 up-train，FAVOR+ 仍可用於快速推論且不損準確率」。以實驗圖為準：
- **量到的數字**：把預訓練 Transformer 權重直接搬到 Performer，**起始準確率 0.07**，之後用原本訓練步數的一小部分恢復；PG-19 上三角函數特徵不穩，正特徵需重抽樣才追得上（Fig. 5 說明，PDF p.7）。
- **跟 Tiara 的關係**：直接證據顯示「免訓練換掉 softmax」在語言模型上不可用〔判讀，依據 Fig. 5〕。
- **與既有整理的關係**：repo 中 `Performer` 的命中（`docs/research_20260924/novelty_sota.md` 等）都是「performer」字串的其他用法，與本論文無關。

#### HyperAttention：Long-context Attention in Near-Linear Time（arXiv 2310.05869v3；venue 〔未查證〕，使用者稱 ICLR 2024）
- **讀了什麼**：〔部分〕v3（2023-12-01），https://arxiv.org/abs/2310.05869 ，讀 PDF p.1–3、8–10（§3.1、§4.1–4.2、Table 1），查證 2026-10-27
- **一句話**：用 LSH 找大元素＋均勻抽樣近似其餘，在「兩個細粒度參數小」時做到近線性時間。
- **線性代數的關鍵一步**：把難度刻畫成兩個參數——正規化後 attention 矩陣的最大欄範數、移除大元素後未正規化矩陣的列範數比（摘要，PDF p.1）。sortLSH 把相近的 Q、K 列排在一起，得到區塊對角近似；其餘部分靠抽樣 V 的列（PDF p.9）。因果遮罩用遞迴切三塊（Fig. 2、Algorithm 4，PDF p.8）。原文說靈感來自 Alman–Song 的困難實例（PDF p.3）。
- **成立的前提**：免訓練（monkey patch），但假設欄範數分布相對均勻（PDF p.4）。
- **量到的數字**（使用者要的「換層後掉多少」，逐字抄 Table 1，PDF p.10；chatglm2-6b-32k、LongBench 中長度 > 32,768 的子集截到 32,768、bucket b=256、抽樣 m=256；從**最後一層往前**換）：

  | 換掉的層數 | single-qa | multi-qa | summarization | few-shot | synthetic | code |
  |---|---|---|---|---|---|---|
  | 0（精確） | 80.63 | 68.14 | 53.12 | 186.58 | 84.00 | 99.57 |
  | 7 | 72.34 | 65.37 | 52.54 | 182.12 | 82.00 | 102.06 |
  | 14 | 71.32 | 62.97 | 52.13 | 169.02 | 80.00 | 92.12 |
  | 21 | 44.75 | 48.91 | 50.90 | 150.44 | 31.00 | 82.58 |
  | 28 | 27.07 | 22.65 | 46.28 | 65.74 | 8.33 | 73.68 |

  - 文字（PDF p.9–10）：換 20 層困惑度約 +1，attention 部分約快 50%；全換困惑度升到 12、快約 2.3 倍；換一半（14 層）時多數任務掉不超過 13%，summarization 幾乎不變，原文認為摘要與程式碼補全比問答「更耐近似」。摘要：32K 時推論快 50%、困惑度 5.6 → 6.3。
  - 單層：131K 時無遮罩最多快 54 倍、因果遮罩 5.4 倍（PDF p.10，d=64、12 heads）。
- **跟 Tiara 的關係**〔判讀〕：這是「拿近似 attention 去重算 KV」最接近的證據——perplexity 就是 prefill。即使只換最後 7 層（最不影響其他層 KV 的位置），single-qa 掉 10%（80.63→72.34）；需要檢索的 synthetic 在全換時崩到 8.33。對以「無損還原」為前提的 Tiara，這是一個新的有損動作，且損失集中在我們最在乎的檢索類任務。
- **與既有整理的關係**：repo 無。

#### LightTransfer：Your Long-Context LLM is Secretly a Hybrid Model with Effortless Adaptation（TMLR 09/2025，PDF 頁首有寫；arXiv 2410.13846v3）
- **讀了什麼**：〔部分〕v3（2026-05-18），https://arxiv.org/abs/2410.13846 ，讀 PDF p.1–4、6、8–9，查證 2026-10-27
- **一句話**：找出「懶層」（只看開頭與最近 token 的層），把它們改成 streaming attention，免訓練得到 hybrid。
- **關鍵一步**：不是線性化，而是利用**預訓練模型裡已存在的結構**（部分層的注意力集中在 sink 與最近 token）；在 prefill 階段即時判斷懶層（LightTransfer-Test，PDF p.3）。全部層都換成 streaming 會嚴重退化（PDF p.2）。
- **量到的數字**（Table 2，PDF p.8，LongBench 平均；一半層被判為懶層）：LLaMA-3-8B-Instruct：Standard 40.8、全層 Streaming 36.1、LightTransfer 38.7；Mistral-7B-Instruct：47.6／37.4／47.4；LLaMA-3-70B-Instruct：44.8／40.0／43.6。原文總結「平均掉 1.5%」（PDF p.8–9）。NIAH（Mistral-7B-Instruct，RULER 的單／多 key，4K–32K）：32K 單 key 96.7% vs 96.6%、多 key 78.2% vs 78.9%（PDF p.9）。吞吐最多 2.17 倍（摘要）。
- **跟 Tiara 的關係**〔判讀〕：唯一「免訓練又幾乎不掉」的例子，**而它保留了 softmax、只是丟掉部分層的遠端 KV**——本質是逐層 KV 淘汰，不是對角化。對 Tiara 的啟示是：可利用的結構是「某些層／head 不需要遠端 KV」，這跟縮圖還原「依層挑 block」屬同一類訊號。
- **與既有整理的關係**：repo 無。

### 3.3 輕度轉換（需要再訓練）及其長 context 表現

#### Hedgehog：The Hedgehog & the Porcupine: Expressive Linear Attentions with Softmax Mimicry（arXiv 2402.04347v1；venue 〔未查證〕，使用者稱 ICLR 2024）
- **讀了什麼**：〔部分〕v1（2024-02-06），https://arxiv.org/abs/2402.04347 ，讀 PDF p.1–2，查證 2026-10-27
- **一句話**：找出既有線性注意力缺的兩個 softmax 性質，用可訓練 MLP 特徵映射去模仿。
- **關鍵一步**：原文認為線性注意力缺 (1) 低熵的「尖銳」權重、(2) 對內積的單調性；簡單特徵映射（如 Taylor 展開）有這些性質但在線性注意力中算起來沒效率，所以改用可訓練 MLP（摘要，PDF p.1–2）。
- **量到的數字**（摘要）：從頭訓練與 finetuned-conversion 可回復 99% 以上；把預訓練 GPT-2 轉成線性版在 WikiText-103 得 16.7 困惑度；Llama-2 7B 加 LoRA 後 ROUGE-1 比原模型高 28.1（摘要用語，未讀該實驗細節）。
- **跟 Tiara 的關係**：「尖銳」正是檢索需要、也是線性映射最難學的性質，跟 §3.1 的算術一致〔判讀〕。
- **與既有整理的關係**：repo 無。

#### LoLCATs：On Low-Rank Linearizing of Large Language Models（arXiv 2410.10254v3；venue 〔未查證〕，使用者稱 ICLR 2025）
- **讀了什麼**：〔部分〕v3（2025-03-05），https://arxiv.org/abs/2410.10254 ，讀 PDF p.1–9、11–12、21、26–29（含附錄 B.5），查證 2026-10-27
- **一句話**：兩步把 LLM 線性化：先逐層讓線性注意力模仿 softmax 輸出（attention transfer），再用 LoRA 微調。
- **關鍵一步**：特徵映射 φ_q、φ_k 為每 head 可訓練（Llama-3-8B，d'=64，約 16.8M 參數，PDF p.5）；為了比對得更像，每層用**大小 64 的 softmax 滑動視窗＋線性注意力**的混合（PDF p.7）。
- **成立的前提**：需要再訓練：約 **40M token**（PDF p.3、p.6），LoRA r=8，單張 40GB GPU 可做（PDF p.3、p.6）。
- **量到的數字**：
  - Llama-3-8B 零樣本 LM Eval 73.1 vs 原模型 74.2；FlashAttention-2 對照下 3 倍吞吐、64 倍 batch（PDF p.3）。5-shot MMLU 仍落後，原文「LoL SAD」段：最多落後 42.4 分（指先前的低秩線性化設定，PDF p.6）。
  - **長 context**：原文 Limitations 寫明「低秩線性化如何應用到長 context」留作未來工作（PDF p.12）。
  - **檢索**（Table 21，PDF p.28；8192-token passkey，依位置分 10 段）：原 Llama-3-8B 92–100；**用 Alpaca 線性化的 LoLCATs Llama-3-8B 全部 0.00**；改用 passkey 樣本當線性化資料則全部 100.00。原文也承認 MMLU 的差距部分來自線性注意力不擅長 in-context 檢索（附錄 B.5.2，PDF p.26）。
  - 〔二手：STILL Table 3，PDF p.9〕在 STILL 的設定下（Llama-3.1-8B，視窗 128–512，4K），LoLCATs 的 RULER 平均 7.2。
- **跟 Tiara 的關係**〔判讀〕：「用 passkey 資料訓練就能 100%」只說明它能學會**那一種**檢索格式，不代表一般長 context 檢索；且沒有 128K 的結果。不能作為「Llama 可以線性化後還原」的依據。
- **與既有整理的關係**：repo 無。

#### Liger：Linearizing Large Language Models to Gated Recurrent Structures（arXiv 2503.01496v2；venue 〔未查證〕）
- **讀了什麼**：〔部分〕v2（2025-05-07），https://arxiv.org/abs/2503.01496 ，讀 PDF p.1–2、6–7，查證 2026-10-27
- **一句話**：不加新參數，重用 key 投影權重做 gate，把 LLM 轉成 gated 線性遞迴，再加層內 hybrid（Liger Attention）。
- **成立的前提**：再訓練約 0.02B token（100,000 筆 Alpaca，序列補到 1024，PDF p.6）。
- **量到的數字**：恢復 Llama-3.2-1B 約 93% 表現，用預訓練 token 的 0.02%（PDF p.1）。本次在全文中搜尋 needle／NIAH／haystack **沒找到**任何長 context 檢索評測（查詢字串：`needle|NIAH|haystack`）。〔二手：STILL Table 2–3，PDF p.9〕Liger-GLA 在 S-NIAH（Llama-3.1-8B）1K 只有 1.0、2K／4K 為 0.0；RULER-4K 平均 2.1。
- **跟 Tiara 的關係**：同 LoLCATs：短 context 指標好看，檢索崩〔判讀〕。
- **與既有整理的關係**：repo 無。

#### SUPRA：Linearizing Large Language Models（Mercat et al.，arXiv 2405.06640v1）
- **讀了什麼**：〔部分〕v1（2024-05-10），https://arxiv.org/abs/2405.06640 ，讀 PDF p.1–2、4–7，查證 2026-10-27
- **一句話**：把 Mistral／Llama2 uptrain 成 RetNet 式線性模型。
- **量到的數字**：Table 2（PDF p.6，Qasper 2-shot，依 context 截斷長度 2K/4K/8K/16K）：Mistral-7B 21.53／25.50／33.61／6.88；Mistral-SUPRA 19.44／17.13／17.11／17.22。原文自己認為 SUPRA 在長長度「保留」的分數來自衰減讓有效 context 變短（PDF p.7），並結論：即使從強的長 context Transformer uptrain，線性模型與 Transformer 仍有顯著差距（PDF p.7）。訓練 token：〔二手：LoLCATs 表，PDF p.9〕Mistral-7B SUPRA 為 100B。
- **跟 Tiara 的關係**：再一次說明「長 context 是線性化的弱點」〔判讀〕。
- **與既有整理的關係**：repo 無。

#### Mamba-in-the-Llama：Distilling and Accelerating Hybrid Models（arXiv 2408.15237v4；venue 〔未查證〕，使用者稱 NeurIPS 2024）
- **讀了什麼**：〔部分〕v4（2025-06-27），https://arxiv.org/abs/2408.15237 ，讀 PDF p.1–2、4、7–8、11，查證 2026-10-27
- **一句話**：重用 attention 的 Q/K/V 投影權重初始化 Mamba 的 C/B/X，蒸餾出 hybrid 模型，並加上 Mamba 的 speculative decoding。
- **關鍵一步**：Q、K、V 的線性投影直接拿來初始化 C、B、X（Fig. 1，PDF p.4）——利用的是 SSD 證明的「1-SS 遮罩線性注意力 ≡ 純量 A 的 SSM」這個對應〔判讀，對應 SSD Cor. 5.1〕。
- **成立的前提**：蒸餾約 **20B token**（PDF p.2）；每個 hybrid 在 8×A100-80G 上不到 5 天（PDF p.8）；保留 50%／25%／12.5%／0% 的 attention 層（PDF p.8）。
- **量到的數字**：NIAH（Fig. 4，PDF p.11）：蒸餾長度只有 2K；3B 的 **50% attention** hybrid 到 10K 完美；8B 的 50% hybrid 到 16K 完美，Mamba-Llama3.1-8B 到 38K 仍不錯。原文 NIAH 只報 50% attention 的版本。〔二手：STILL Table 5，PDF p.10，Llama-3.2 1B 系〕MambaInLlama 的 RULER：4K 38.75、8K 21.55、16K 3.88。
- **跟 Tiara 的關係**〔判讀〕：要保留一半 attention 層才有可用的 NIAH，KV 最多省一半；且沒有 128K 的證據。
- **與既有整理的關係**：repo 無。

#### Llamba（arXiv 2502.14458v2）與 Lizard（arXiv 2507.09025v4）——簡卡
- **讀了什麼**：Llamba〔部分〕v2，https://arxiv.org/abs/2502.14458 ，讀 PDF p.1、3、5–6；Lizard〔部分，只讀引言〕v4，https://arxiv.org/abs/2507.09025 ，讀 PDF p.1–4，查證 2026-10-27
- **重點**：Llamba-3.1-8B 用 MOHAWK 蒸餾成純 Mamba-2，共 12B token（PDF p.1、p.6）；本次在 Llamba 全文搜尋 needle／NIAH／RULER **沒找到**長 context 檢索評測（查詢字串：`needle|NIAH|RULER`）。Lizard 引言（PDF p.2）說：5-shot MMLU 上 LoLCATs 落後老師 13.8 分、Liger 落後 21.9 分；並指出保留 RoPE 的線性化模型受限於訓練時看過的長度。Lizard 自己的長 context 結果本次未讀。
- **與既有整理的關係**：repo 無。

#### STILL：Selecting Tokens for Intra-Layer Hybrid Attention to Linearize LLMs（arXiv 2602.02180v2；預印本）
- **讀了什麼**：〔部分〕v2（2026-10-05），https://arxiv.org/abs/2602.02180 ，讀 PDF p.1–4、8–11，查證 2026-10-27
- **一句話**：線性化時不只留滑動視窗，而是用局部可算的「self-saliency」挑出少量重要 token 走精確 softmax，其餘走線性注意力。
- **關鍵一步**：層內 hybrid：輸出＝選中 token 的 softmax 分支＋其餘 token 的線性注意力分支；用 chunk 為單位挑 token 以利平行（PDF p.3–4）。
- **成立的前提**：沿用 LoLCATs 的兩階段訓練，Alpaca 20M token、長度 1024（PDF p.8）。
- **量到的數字**（Llama-3.1-8B 為底，總快取 token 數與 baseline 視窗相同）：
  - S-NIAH（Table 2，PDF p.9）0.5K/1K/2K/4K：SWA 100／25.2／14.6／5.8；LoLCATs 100／65.6／24.6／8.8；Liger-GLA 73.4／1.0／0.0／0.0；STILL 100／99.6／89.0／86.2。
  - RULER-4K 平均（Table 3，PDF p.9）：Mamba2 44.7、LoLCATs 7.2、Liger-GLA 2.1、STILL 47.9。
  - 層間＋層內 hybrid（Table 5，PDF p.10；Llama-3.2 1B，保留 25% full attention 層、DCLM 上再訓練 350M token）：RULER 4K/8K/16K：LoLCATs 33.99／17.93／7.59；STILL 53.24／35.66／27.20。
  - 注意：表上沒有原始 full-attention 模型的 RULER 分數可對照（本次沒在表中找到；查詢 `full attention`）。
- **跟 Tiara 的關係**〔判讀，本檔最直接的連結〕：STILL 的結果說明「**檢索能力只在被挑中的 token 保留精確 softmax 時才回來**」，線性（壓縮）分支只負責其餘的大略資訊。這和縮圖還原的分工（縮圖負責挑、被挑中的 block 拿原版精確算）是同一種結構；而 STILL 的上限仍很低（16K RULER 27.2），也提醒我們：挑選的品質決定一切。
- **與既有整理的關係**：repo 無。L7d（`docs/stage3_20261019/lit/L7d_recheck_lowrank_index.md`）整理的是「用低秩 K 當選擇性還原的索引」，跟 STILL 的「用 self-saliency 挑 token」是兩種不同的挑選訊號，可對照。

### 3.4 softmax 模型裡**確實存在**、可利用的結構（簡述並指路）
- **RoPE**：每 2 維一組的旋轉＝區塊對角的正交矩陣；在複數形式下就是對角矩陣 diag(e^{iθ_j m})——形式上跟 S4D 的「複數對角 A、模長 1、無衰減」相同〔判讀〕。RoPE 前的 K 低秩、RoPE 後秩變高：已整理在 `docs/stage3_20261019/lit/L5a_lowrank.md` L21–29、L110–127；本輪由另一個 agent 處理，本檔不重複。
- **層／head 的稀疏性**（LightTransfer 的懶層、HyperAttention 的「大元素少」假設）：這是免訓練就能用、而且已被證實存在的結構，但它是「挑選」而不是「分解」〔判讀〕。

---

## §4 證據總表（給趕時間的人）

| 路線 | 需要訓練？ | 長 context 檢索證據 | 出處 |
|---|---|---|---|
| 換基底／對角化（S4、S4D、DSS、Mamba） | 從頭訓練，結構是模型定義 | 無；Mamba 類在複製／查表上輸 Transformer | S4 §3；Jelassi Fig. 1(c) |
| SSD 半可分 | 從頭訓練 | 原文：不涵蓋 softmax；混合模型留約 10% attention 當檢索 | SSD §10.3、§9.2.3 |
| 精確線性化 softmax | 不可能（exp 無限維） | — | Katharopoulos §3.2.1 |
| 次二次近似（一般情況） | — | Q、K 元素達 Θ(√log n)（即 logit 達 Θ(log n)）時 SETH 下不可能〔複核 2026-10-27：原寫「logit 達 Θ(√log n)」，改為「Q、K 元素達 Θ(√log n)（logit 達 Θ(log n)）」；Alman–Song Thm 1.3 的參數是 B＝‖Q‖∞,‖K‖∞＝C_b√log n，logit＝QKᵀ/d ≤ B²，PDF p.3〕 | Alman–Song Thm 1.3 |
| Performer 直接換 | 免訓練 → 0.07 準確率 | — | Performer Fig. 5 |
| HyperAttention 直接換 | 免訓練 | 全換：synthetic 84.00 → 8.33（32K） | HyperAttention Table 1 |
| LightTransfer（仍是 softmax） | 免訓練 | Llama-3-8B LongBench 40.8 → 38.7；NIAH 32K 持平 | LightTransfer Table 2、Fig. 4 |
| LoLCATs / Liger | 40M / 20M token | 8K passkey 0%（Alpaca）；RULER-4K 7.2 / 2.1 | LoLCATs Table 21；STILL Table 3 |
| Mamba-in-the-Llama | 20B token，留 50% attention | 8B 到 16K 完美 | MambaInLlama Fig. 4 |
| Kimi Linear（KDA:MLA＝3:1） | 從頭 1.4T token | RULER-128K 84.3 | Kimi Fig. 1 |

---

## §5 對 Tiara 的意涵（判讀）

> 白話：線性代數加速是「模型被訓練成有那個結構」才有的紅利；我們的 Llama／Qwen 沒有，硬套就是有損近似，而且省不了多少。能借的是「少數精確、其餘壓縮」的分工形狀，以及 hybrid 模型帶來的新還原對象。

### 5.1 「對角化／線性化 softmax 來加速還原」為什麼不建議
- 還原的兩條路——「搬」與「重算」——都不會因為對角化變快：
  - **搬**：KV 是資料，不是線性運算子；能壓的是它的冗餘（低秩、量化），那是 L5a／另一 agent 的範圍，不是 SSM 的技巧。
  - **重算**：重算＝對前綴做 prefill。prefill 的 attention 是 softmax，§3.1–3.2 的證據說：精確地做不到次二次；近似地做，KV 會帶誤差（HyperAttention 換最後 7 層 single-qa 就掉 10%）。對 Tiara「CPU／SSD 兩階無損」的定義（CLAUDE.md §0），這等於引入一個新的有損動作，必須跟縮圖一樣量品質。
- 從預訓練 Llama／Qwen 做輕度轉換，目前沒有任何一篇在 128K 上證明檢索可保留（§3.3），而且轉換後就是另一個模型，不再是「同一個模型的 KV 管理」問題。

### 5.2 就算 attention 變免費，重算最多快多少（〔算術〕）
- Llama-3.1-8B（32 層、32 個 Q head × 128 ＝ 4096；config 見 `docs/stage3_20261019/lit/L6_mainstream_practice.md` L270）。prefill 每 token：
  - 線性層 ≈ 2 × 參數量 ≈ 2 × 8.03e9 ≈ 16.1 GFLOP（參數量取公開值，本次未重新抓 config）。
  - attention（QKᵀ 與 AV）≈ 4 × 4096 × ctx × 32 層；因果遮罩下平均 ctx ＝ n/2。
  - n＝131,072：attention ≈ 4 × 4096 × 65,536 × 32 ≈ 34.4 GFLOP／token，占 34.4／(34.4＋16.1) ≈ 68%；attention 降到 0 時上限加速 ≈ 50.5／16.1 ≈ **3.1 倍**。
  - n＝262,144：attention ≈ 68.7 GFLOP／token，占 81%；上限 ≈ **5.3 倍**。
- 這只是 FLOPs 的 Amdahl 上限，忽略 FlashAttention 實際效率與記憶體頻寬；而 HyperAttention 實測的端到端數字（32K 推論快 50%）遠低於此。結論：即使接受有損，重算端能拿到的加速也有限〔判讀〕。

### 5.3 真正能借的東西
1. **「少數精確、其餘壓縮」的結構**：SSD 的 chunk 內精確／chunk 間低秩、STILL 的「挑中 token 走 softmax、其餘走線性」、以及我們的縮圖還原，是同一個形狀。STILL 的資料（LoLCATs 視窗式 8.8 → STILL 挑選式 86.2，S-NIAH 4K）可以當作「挑選訊號比壓縮方式重要」的外部旁證，用在論文的動機段〔判讀〕。
2. **固定狀態不能取代 KV 的定量論據**（Jelassi Thm 2.7、Based Thm 3.1、§0 第 5 點的 2,048 倍算術）：可以拿來回答審稿人「為什麼不直接把 KV 壓成 RNN 狀態」。
3. **Alman–Song 的界線**：可用來說明「能被便宜近似的 head 是注意力分散的 head」——這提供一個可檢驗的假說：縮圖（有損）對「平坦」的 head 應該較安全，對「尖銳」的 head 較危險。是否成立需要量，目前 NOT_MEASURED。

### 5.4 hybrid 模型改變了還原的對象（指路，不重抄）
- GDN／KDA／Mamba-2 層沒有逐 token 的 KV，只有固定大小的狀態快照；還原這些層＝搬一個狀態或從檢查點重放。repo 已整理：`docs/DIRECTIONS_20260927.md` L220–239（狀態檢查點當新快取物件、vLLM 的 `MambaSpec` 處理）、`docs/research_20261009_explore/lit_C_architectures.md` L169–195（狀態大小、`mamba_cache_mode`）、`cards/C_LinearKV.md`（Mamba-2 與 GDN 的近似行為不同）、`cards/C_TailReplay.md`。
- 本卡新增的理論補充：SSD 的 chunk 演算法本身就是「每個 chunk 邊界算出一個狀態」（Listing 1 第 3 步，PDF p.21），所以「每 chunk 存一份狀態、還原時從最近的檢查點重放」在演算法上是自然的；成本是 chunk 間 scan（原文說可忽略）加上一個 chunk 的 intra-chunk 計算〔判讀〕。

### 5.5 沒查證／沒做的
- 沒有找到任何把 Llama-3.1-8B 或 Qwen2.5 線性化後在 **128K** 上做 RULER／NIAH 的論文（查詢：「linearizing pretrained LLM long context needle-in-a-haystack RULER evaluation linear attention conversion 2025」、「distill Llama-3.1-8B into hybrid Mamba Gated DeltaNet RULER 128K retrieval degradation」）。查不到不等於沒有。
- Zoology 只讀了摘要；Lizard 只讀了引言；Hedgehog 的 Llama-2 實驗細節未讀。
- S4、S4D、DSS、Mamba、SSD、LinAttn、Hedgehog、LoLCATs、Liger、Mamba-in-the-Llama、Repeat After Me、HyperAttention、Alman–Song 的 venue 本次沒在 PDF 上核到，標〔未查證〕。
- Performer Thm 4 的公式因 pdftotext 排版不全，只記依賴關係。

---

## 參考文獻（IEEE 風格，含 URL；查證 2026-10-27）

[1] A. Gu, K. Goel, C. Ré, "Efficiently Modeling Long Sequences with Structured State Spaces," arXiv:2111.00396v3, 2022. https://arxiv.org/abs/2111.00396
[2] A. Gu, A. Gupta, K. Goel, C. Ré, "On the Parameterization and Initialization of Diagonal State Space Models," arXiv:2206.11893v2, 2022. https://arxiv.org/abs/2206.11893
[3] A. Gupta, A. Gu, J. Berant, "Diagonal State Spaces are as Effective as Structured State Spaces," arXiv:2203.14343v3, 2022. https://arxiv.org/abs/2203.14343
[4] A. Gu, T. Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," arXiv:2312.00752. https://arxiv.org/abs/2312.00752
[5] T. Dao, A. Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality," arXiv:2405.21060. https://arxiv.org/abs/2405.21060
[6] A. Katharopoulos, A. Vyas, N. Pappas, F. Fleuret, "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention," arXiv:2006.16236v3, 2020. https://arxiv.org/abs/2006.16236
[7] S. Yang, B. Wang, Y. Zhang, Y. Shen, Y. Kim, "Parallelizing Linear Transformers with the Delta Rule over Sequence Length," NeurIPS 2024, arXiv:2406.06484v6. https://arxiv.org/abs/2406.06484
[8] S. Yang, J. Kautz, A. Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta Rule," ICLR 2025, arXiv:2412.06464v3. https://arxiv.org/abs/2412.06464
[9] Kimi Team, "Kimi Linear: An Expressive, Efficient Attention Architecture," arXiv:2510.26692v2, 2025. https://arxiv.org/abs/2510.26692
[10] J. Alman, Z. Song, "Fast Attention Requires Bounded Entries," arXiv:2302.13214v2, 2023. https://arxiv.org/abs/2302.13214
[11] S. Jelassi, D. Brandfonbrener, S. M. Kakade, E. Malach, "Repeat After Me: Transformers are Better than State Space Models at Copying," arXiv:2402.01032v2, 2024. https://arxiv.org/abs/2402.01032
[12] S. Arora et al., "Zoology: Measuring and Improving Recall in Efficient Language Models," arXiv:2312.04927v1, 2023. https://arxiv.org/abs/2312.04927
[13] S. Arora et al., "Simple linear attention language models balance the recall-throughput tradeoff," arXiv:2402.18668v2. https://arxiv.org/abs/2402.18668
[14] K. Choromanski et al., "Rethinking Attention with Performers," ICLR 2021, arXiv:2009.14794v4. https://arxiv.org/abs/2009.14794
[15] I. Han, R. Jayaram, A. Karbasi, V. Mirrokni, D. P. Woodruff, A. Zandieh, "HyperAttention: Long-context Attention in Near-Linear Time," arXiv:2310.05869v3, 2023. https://arxiv.org/abs/2310.05869
[16] X. Zhang et al., "LightTransfer: Your Long-Context LLM is Secretly a Hybrid Model with Effortless Adaptation," TMLR 09/2025, arXiv:2410.13846v3. https://arxiv.org/abs/2410.13846
[17] M. Zhang, K. Bhatia, H. Kumbong, C. Ré, "The Hedgehog & the Porcupine: Expressive Linear Attentions with Softmax Mimicry," arXiv:2402.04347v1, 2024. https://arxiv.org/abs/2402.04347
[18] M. Zhang et al., "LoLCATs: On Low-Rank Linearizing of Large Language Models," arXiv:2410.10254v3, 2025. https://arxiv.org/abs/2410.10254
[19] D. Lan, W. Sun, J. Hu, J. Du, Y. Cheng, "Liger: Linearizing Large Language Models to Gated Recurrent Structures," arXiv:2503.01496v2, 2025. https://arxiv.org/abs/2503.01496
[20] J. Mercat et al., "Linearizing Large Language Models," arXiv:2405.06640v1, 2024. https://arxiv.org/abs/2405.06640
[21] J. Wang, D. Paliotta, A. May, A. M. Rush, T. Dao, "The Mamba in the Llama: Distilling and Accelerating Hybrid Models," arXiv:2408.15237v4, 2025. https://arxiv.org/abs/2408.15237
[22] A. Bick et al., "Llamba: Scaling Distilled Recurrent Models for Efficient Language Processing," arXiv:2502.14458v2, 2025. https://arxiv.org/abs/2502.14458
[23] C. V. Nguyen et al., "Lizard: An Efficient Linearization Framework for Large Language Models," arXiv:2507.09025v4. https://arxiv.org/abs/2507.09025
[24] W. Meng et al., "STILL: Selecting Tokens for Intra-Layer Hybrid Attention to Linearize LLMs," arXiv:2602.02180v2, 2026. https://arxiv.org/abs/2602.02180
