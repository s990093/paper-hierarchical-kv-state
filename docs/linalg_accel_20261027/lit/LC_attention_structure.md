# LC　softmax attention 本身的線性代數結構：哪些能讓「還原」的計算變便宜或重組

> **日期**：2026-10-27。**執行者**：文獻 agent C（只讀、只上網；這台 Mac 只做 PDF 轉文字與〔算術〕，沒有跑任何實驗或模擬）。只寫這一個檔。
> **範圍**：Llama-3.1-8B、Qwen2.5-7B／14B（GQA、RoPE）上，六種結構：投影的線性（§1）、RoPE 的對角性（§2）、softmax 正規化的結合律（§3）、注意力圖樣的便宜估計（§4）、量化前的正交旋轉（§5）、權重吸收（§6）。
> **不重做**：位元組那一面（低秩／SVD、跨層 xKV＋INT4 疊加、RoPE 前低秩 K 當索引）已在 `docs/stage3_20261019/lit/L5a_lowrank.md` §0、`L7d_recheck_lowrank_index.md` §1、`docs/stage3_20261019/results_analysis.md` 摘要節。本檔只在需要時引用。
> **原文**：PDF 都在 `<SCRATCH>/litC/pdf/<arXiv id>.pdf`，用 `pdftotext -layout` 轉文字，頁碼＝PDF 頁。HGCA 圖 6 另轉 PNG 看過。
> **標記**：〔全文〕〔部分〕〔摘要〕〔程式碼〕〔文件〕〔二手〕〔算術〕〔判讀〕〔未查證〕，定義照 `RULES.md`。本檔是 self-review，不是 cross-model review。

---

## 0. 一頁結論（白話）

**一句話**：六種結構裡，能讓「暫停後回來」的還原少算或少搬的，前人幾乎都在 **decode（一次一個 token）** 或 **MHA 模型** 上用過了。換到我們的設定（GQA、新問題一次幾十個 token 的 prefill、128K–256K），大多數不是不划算，就是早就被別人做掉。真正沒人碰過的只有兩小塊，都是〔判讀〕，沒有任何數字支持。

1. **投影是線性的 → 存 hidden state、回來時用一個 GEMM 算回 KV（HCache）。在我們三個模型上一定輸。**
   - GQA 的 KV 比 hidden state 小。每層每 token 的 hidden／KV 比〔算術〕：Llama-3.1-8B **2.0**、Qwen2.5-7B **3.5**、Qwen2.5-14B **2.5**。
   - 也就是說：要搬的位元組變成 2–3.5 倍，還要多做一次 GEMM。
   - 這招只在 g＝n_q／n_kv < 2（實際上就是 MHA）時才贏。HCache 原文自己也只支援 MHA（p13）〔全文〕。
   - 把 hidden state 投影到 W_K 的奇異向量上再量化（XQuant 的 GQA 版）〔全文〕：位元組和 KV 一樣，代數上等於「在另一組基底下存 RoPE 前的 K」。所以它屬於位元組那一面，L5a 已經涵蓋這一類。
2. **RoPE 在複數形式下是對角的 → 換位置只要逐元素乘一下。但「暫停後回來」位置不變，這招用不到。**
   - 只有兩種情況需要它：
     - 存 RoPE 前的低秩 K（A0 的粗底已經這樣做）。
     - 跨 session 共用 KV（KVS 那條線）。
   - 重新旋轉的成本：每篇論文都說「可忽略」，但沒有一篇把它拆出來量（CacheBlend p6、KVLink p3、SparseX p9）〔全文〕。
   - 我們的〔算術〕：Llama 128K 全部 K 讀寫一次約 4 ms，和反量化融在同一個 kernel 裡就趨近 0。
3. **softmax 正規化有結合律（(o, lse) 可以合併）→「CPU 算一部分注意力、只傳結果回來」。這招已經被做滿了，而且全部只用在 decode。**
   - 用過的系統：FastDecode、NEO、MagicPIG、RetroInfer、HGCA、ScoutAttention、HybridGen。InstInfer 把同樣的想法放進 SSD。
   - prefill 的注意力，NEO（p4）、InstInfer（p5）、MagicPIG（p25）都**明寫留在 GPU**〔全文〕。
   - 唯一量了交叉點的是 HGCA 圖 6（p5）〔部分〕：A6000、PCIe 4.0、問題長 1 時 CPU 贏；問題長 32 時 GPU（含 PCIe 搬運）和 CPU 打平。
   - 我們 R2 的〔算術〕和這個交叉點**一致**：交叉點 q\* ≈ CPU FLOPS ÷（PCIe 頻寬 × GQA 組大小 g）。
     - HGCA 的硬體、g＝1：算出 38，實測約 32。
     - 我們：Qwen-7B 7.9、Qwen-14B 11.1（arith §7 寫 7、11）。
   - 結論：**R2 的否決站得住**，而且 GQA 讓交叉點再縮小 g 倍。
4. **用便宜的線性代數估注意力圖樣，讓重算的 prefill 變稀疏：估計器很多，都是給第一次 prefill 用的。**
   - 估計器：MInference 用最後 64 個 q；XAttention 用反對角線；FlexPrefill 用池化後的 Q·Kᵀ；SampleAttention 用抽樣。
   - SparseX〔全文〕不是稀疏重算整段前綴：它挑「哪些 token」要修正，而且全部在 GPU 記憶體裡（p14 明寫不是多層快取）。repo 的 M12 對它的描述（摘要級）和全文一致。
   - 唯一沒查到的小點〔判讀〕：還原時要重算的是**同一段** context，第一次 prefill 時其實「看過」真實的注意力圖樣。可以在寫入時把區塊級的注意力圖存下來（Llama 128K 約 32 MB〔算術〕），回來時直接當稀疏遮罩用，不必再估。本次搜尋沒找到這樣做的（見 §4.3）。
5. **量化前先做正交旋轉（Hadamard）：對 A0 的「條件 2」大概幫不上。**
   - 已知事實〔全文〕：
     - KVLinC 表 1（p7）：2 bit 下，QuaRot 式「K、V 都旋轉＋逐 token 量化」在 Qwen2.5-7B 上困惑度 3380（完全壞掉）。
     - 最好的組合是「K 不旋轉、逐 channel 量化；V 先 Hadamard、再逐 token 量化」（圖 3，p4）。只換這個組合、不訓練，Llama-3.1-8B 的 GSM8K 從 34.1 升到 36.9（表 3，p8）。
     - 4 bit 時各種方法都接近 BF16（RotateKV 表 4，p6）。
   - A0 的設定：粗底的係數已經在校準好的 SVD 基底裡，而且每個主成分各自有 scale（`code/m16_xstack.py:23–26`）；精修是 INT4-KIVI，我們自己量過不掉分。
   - 再疊一個 Hadamard 會把成分的變異數攪勻，和「每個成分各自給 scale」打架（repo 已經寫過：`L5d_stacking.md:270`）。
   - 條件 2 差的 0.05 pp 是 288 題時 CI 的寬度問題（`A0_progressive_restore.md:190`），不是誤差的偏差。
   - 〔判讀〕旋轉頂多微調點估計，不足以當成「把 A0 推過門檻」的手段。真正的檢驗是 A0-C 的 864 題。
6. **權重吸收（MLA 式）：碰到 RoPE 就吸收不了。**
   - 能吸收的只有不帶 RoPE 的部分：
     - Palu：V 的部分可以吸收進 W_O；K 要先重建、再套 RoPE（L5a §0 第 1、2 點）。
     - TransMLA 用 RoRoPE 把位置資訊集中到一個 head，其他維度丟掉 RoPE 才能吸收。完全不訓練時，**Llama-2** 壓 68.75% 掉 1.65%；壓 93% 要 6B token 的微調才救得回來（p2–3）〔全文〕。
     - MHA2MLA 要用 0.6–1% 的預訓練資料微調（摘要）。
   - 我們不訓練，所以只剩 Palu 那一類，repo 已經涵蓋。

**真正沒查到前作的（兩條，都只是〔判讀〕）**：
- (a) §4 的「把寫入時看到的注意力圖存起來，給重算當遮罩」。
- (b) §3 的「同一層裡，粗底那一份的部分注意力 (o, lse) 先算，精修的 block 到了再合併」。這只是同層內的重疊，上限很小，不建議單獨立題。

**沒能查證的**：HGCA 圖 6 用哪個模型、KV 軸的單位；HybridGen 有沒有 prefill；Grouped Value Attention（2609.13285，要訓練）沒讀；XQuant 沒量 GEMM 的實際延遲。見 §7。

---

## 1. 投影的線性：K＝norm(X)·W_K、V＝norm(X)·W_V → 從 hidden state 用一個 GEMM 還原 KV

**白話**：hidden state 存一份，回來時乘一次矩陣就能得到 K、V，完全不用做注意力。但 GQA 模型的 KV 本來就比 hidden state 小，所以存 hidden state 反而搬得更多。

**數學**
- 第 ℓ 層：K_ℓ＝RoPE(r_t·(x_ℓ⊙γ)·W_K (＋b_K))，V_ℓ＝r_t·(x_ℓ⊙γ)·W_V (＋b_V)。r_t＝1／rms(x_ℓ) 是每個 token 一個純量；Qwen2.5 的 Q、K、V 有 bias（KVLinC p6 也這樣描述）〔全文〕。所以 KV 是 x_ℓ 的仿射函數。
- 每層每 token：hidden 有 d 個值，KV 有 2·n_kv·d_h 個值。d＝n_q·d_h，所以 hidden／KV＝n_q／(2·n_kv)＝**g／2**。只有 g < 2 時 hidden 比較小〔算術〕。
- **依賴關係**：x_ℓ 是第 ℓ−1 層的輸出，要算它，就得在第 0…ℓ−1 層把全部 N 個 token 的注意力和 FFN 都算一遍（＝重算）。HCache 之所以能逐層獨立還原，是因為**每一層的 x_ℓ 都存了**。如果只存某幾層的 x（檢查點），中間各層就退化成「從檢查點往後做 prefill」，就是 HCache 的 L_O 層、Cake 前段的那種重算。

### 1.1 〔算術〕我們三個模型的位元組比與 GEMM 成本

config 來源：Qwen 兩個是 HF `Qwen/Qwen2.5-{7B,14B}-Instruct-1M/config.json`（2026-10-27 取得，存 `<SCRATCH>/litC/`）。Llama 的官方 repo 要登入，改用 `unsloth/Meta-Llama-3.1-8B-Instruct` 的鏡像〔文件〕，數值和 repo 既有整理一致（`docs/PAPER_GUIDE.md:617`、`docs/RELATED_WORK_WEAKNESSES.md:134`、`arith.md` §1）。

| 模型 | d | n_q／n_kv（g） | 層 | KV 每層每 token（值） | hidden（值） | hidden÷KV | N | 整段 KV（BF16） | 整段 hidden | 從 hidden 算回 KV 的 GEMM |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Llama-3.1-8B | 4096 | 32／8（4） | 32 | 2048 | 4096 | **2.00** | 128K | 16.0 GiB | 32.0 GiB | 70.4 TFLOP |
| Qwen2.5-7B | 3584 | 28／4（7） | 28 | 1024 | 3584 | **3.50** | 256K | 14.0 GiB | 49.0 GiB | 53.9 TFLOP |
| Qwen2.5-14B | 5120 | 40／8（5） | 48 | 2048 | 5120 | **2.50** | 256K | 48.0 GiB | 120.0 GiB | 263.9 TFLOP |
| （對照）Llama-2-7B，MHA | 4096 | 32／32（1） | 32 | 8192 | 4096 | 0.50 | — | — | — | — |

公式：
- 整段大小＝值數 × 2 B × 層 × N。
- GEMM＝2·N·d·(2·n_kv·d_h)·層。

套 repo 的假設（PCIe 57.6 GB/s，由 arith §7 的 14 GiB／0.261 s 反推；GEMM 用 788.1 TFLOPS 的 50%，arith §6），一趟還原的時間〔算術〕：

| 模型 | 搬 KV | 搬 hidden | 搬 hidden 之外還要的 GEMM |
|:--|--:|--:|--:|
| Llama 128K | 0.30 s | 0.60 s | ＋0.18 s |
| Qwen-7B 256K | 0.26 s | 0.91 s | ＋0.14 s |
| Qwen-14B 256K | 0.89 s | 2.24 s | ＋0.67 s |

- 要讓 HCache 在 GQA 上贏，hidden 就得先壓到 2／g 倍以下。但同樣的線性壓縮直接套在 KV 上也做得到（見 1.3 的判讀）。
- **什麼時候贏、什麼時候輸**：只在 g < 2（MHA）時，hidden 比 KV 小，HCache 才贏。在 GQA 上，位元組和計算兩邊都輸，是被嚴格支配的選項。這個結論和連結速度無關：GPU 再閒、連結再慢，hidden 都是比較大的那一份。

### 1.2 卡片

#### HCache：Fast State Restoration in LLM Serving with HCache（EuroSys 2025；arXiv 2410.05004）
- **讀了什麼**：〔全文〕v1。repo 已有完整卡（`docs/research_20261006_eval/cards/E03_restore_recompute.md:191–222`），這裡只補核對過的幾點，查證 2026-10-27。
- **一句話**：每層存 hidden state，還原時用 GEMM 投影回 KV，再和 token 重算、KV 搬運依層分工。
- **線性代數的關鍵一步**：K、V 對 hidden 是線性的；從 hidden 算 KV 是 4·N·D² 的 GEMM，沒有 N² 的注意力項（p5）〔全文〕。
- **成立的前提**：原文 §7（p13）：MQA／GQA 要先把 hidden 投影到低秩表示再存，這「需要改模型結構，超出本文範圍」；目前只支援 MHA（Llama2、Gemma、Phi2、Qwen1.5）〔全文〕。
- **量到的數字**（原文 p2、p10）：還原速度比 KV offload 快 1.33–2.66×；TTFT 對 KV offload 1.27–1.93×、對重算 2.21–5.73×。模型是 Llama2-7B／13B、OPT-30B（MHA）〔全文，摘自 repo 卡並核對 p2 的 1.33–2.66×〕。
- **跟 Tiara 的關係**〔判讀〕：在我們三個模型上，hidden 是 KV 的 2–3.5 倍（§1.1），不能當還原的格式。可以借用的是「依層決定存什麼」的排程想法，repo 已經整理過（`03_competitors.md:65`）。
- **與既有整理的關係**：`PAPER_GUIDE.md:617`、`RELATED_WORK_WEAKNESSES.md:134` 的 3.5×／2× 和本檔一致；本檔補上 Qwen-14B 的 2.5×。

#### XQuant：Breaking the Memory Wall for LLM Inference with KV Cache Rematerialization（venue 未查證；arXiv 2508.10395）
- **讀了什麼**：〔全文〕v1（2025-08-14），§3、§4、表 1、2、4，<https://arxiv.org/pdf/2508.10395>，查證 2026-10-27。
- **一句話**：快取量化後的層輸入 X，用的時候再算回 KV。跨層版只存相鄰層 X 的差。
- **線性代數的關鍵一步**：
  - (i) **GQA 版**（§3.3.1，p8）：離線對 W_K 做 SVD，W_K＝U_k Σ_k B_kᵀ；快取 X·U_k（d／g 維），用的時候乘上融合好的 Σ_k B_kᵀ（(d／g)×(d/g) 的方陣）。
  - (ii) **XQuant-CL**（§3.2–3.3，p7–9）：靠殘差流讓相鄰層的 X 很像，存 ΔX̂_i＝Q(X_i − X̂_{i−1})，並維持一個累加器 X̂_i＝X_0＋Σ ΔX̂_j。
  - GQA 下的 CL：累加器是 l×d 的**完整寬度**，delta 先降到 U_kv 的 latent 再存（p9）。
- **成立的前提**：不用訓練（SVD 離線做）。適用 Llama-3.1-8B（GQA）與 Mistral-7B〔全文〕。
- **量到的數字**（只抄原文；KIVI\* 是作者的設定，group 128）：
  - 表 1（p11），Llama-3.1-8B 的 WikiText-2 PPL，基線 6.24：
    - KV 0.27：KIVI\*-4bit 6.31、XQuant-4bit 6.28。
    - KV 0.14：KIVI\*-2bit 9.95、XQuant-2bit 7.74。
  - 表 2（p12），Llama-3.1-8B-Instruct 的 LongBench 平均，全 KV 48.5：
    - KV 0.27：KIVI\*-4bit 48.4、XQuant-4bit 48.6。
    - KV 0.14：KIVI\*-2bit 39.9、XQuant-2bit 44.6。
  - 表 4（p13），Llama-3.1-8B PPL，KV 0.15：KVQuant-2bit-1% 7.45、XQuant-2bit 7.38、XQuant-CL-2bit 6.60。
  - 延遲：**只有 roofline 推導（p10，假設 H100），沒有量測**；限制節（p14）自己承認重新物化會增加計算、可能拉高延遲。
- **跟 Tiara 的關係**〔判讀〕：
  - GQA 版快取的 X·U_k＝K_preRoPE·B_k Σ_k⁻¹（W_K 的列滿秩時成立）〔算術〕，是 RoPE 前 K 的可逆線性變換。所以它是「換一組基底存 KV」，和 L5a 的 (c) 類（座標轉換後量化）同一家族，位元組和 KV 一樣。
  - XQuant-CL 的結構對**逐層還原**是天然的：GPU 上只留一個 l×d 的累加器（Llama 128K 是 1 GiB〔算術〕：131072×4096×2 B），每層只搬一份 latent delta。但第 i 層的 X̂_i 依賴前面所有 delta（依序累加），所以「只搬挑中的 block、跳層」不成立。
  - CL 在 GQA 上的好處只有 PPL 的證據，沒有長 context 的生成答對率。
- **與既有整理的關係**：repo grep「XQuant」0 筆（2026-10-27）。新卡。

#### 其他追蹤
- **CacheFlow**（arXiv 2604.25080）：repo 已有（`E03_restore_recompute.md`）。本次搜尋摘要提到它認為 HCache 式的作法在混合（recurrent）模型上會失去儲存優勢〔二手，搜尋摘要；沒重讀原文〕。
- **Grouped Value Attention**（2609.13285）：要訓練，沒讀〔未查證〕。
- 「低秩或量化的 hidden state 取代 KV」除了 XQuant 之外，本次搜尋沒找到（查詢字串：`store hidden states instead of KV cache restore GEMM GQA low-rank quantized activation cache 2025`、`"hidden cache" OR "activation cache" LLM serving restore KV from hidden states GQA`）。

### 1.3 〔判讀〕對 Tiara 還原的意義
- **搬 vs 重算 vs 縮圖**：hidden state 不是一個新的有利格式。它對 KV 的關係是「多了 g／2 倍的冗餘」，任何對 hidden 的線性壓縮 P 都會誘導出 KV 上一個維度不更大的壓縮（KV＝r_t·x·diag(γ)·W，壓縮後的 x≈cPᵀ，代入得 KV≈r_t·c·(Pᵀ diag(γ) W)）〔算術〕。所以「存 hidden」不可能比「直接存同維度的 KV 係數」省位元組，只多了 GEMM。
- 唯一的不同點是殘差流的跨層相似（XQuant-CL）。但 repo 已量過相鄰層 K 的逐 token cosine ≈ 0（`05_rejected.md` R6）。K 不像，X 很像，兩件事不矛盾（K 經過不同的 W_K）。「存 X 的跨層 delta」在我們的 128K–256K 生成答對率上 NOT_MEASURED，而且和 A0 的 xKV 粗底屬於同一類位元組手段，不在本檔範圍。

---

## 2. RoPE 是 2×2 區塊對角（複數形式是對角）→ RoPE 前儲存、逐元素平移、換位置重用

**白話**：RoPE 等於每兩個維度轉一個角度。要把一段 KV 搬到別的位置，只要每個 K 再轉一下就好，V 不用動。但「暫停後回來」位置沒變，所以還原用不到這一招。

**數學**
- 把 K 每兩維看成一個複數：k̃_j(m)＝k_j·e^{i·m·θ_j}。換到位置 m＋Δ：k̃_j(m＋Δ)＝k̃_j(m)·e^{i·Δ·θ_j}，逐元素乘（SparseX 式 (3.x)，p9；Block-Attention 式 (1)(2)，p3–4）〔全文〕。
- q·k 只和相對位移有關，V 不帶相位（SparseX p9）〔全文〕。
- 推論〔全文，TransMLA p2〕：同一頻率、跨 head 的旋轉 U，只要實部和虛部用同一個 U，就和 RoPE 可交換（RoRoPE 的不變性條件）。

### 2.1 卡片（只列和還原有關的事實）

#### CacheBlend：Fast LLM Serving for RAG with Cached Knowledge Fusion（EuroSys 2025；arXiv 2405.16444）
- **讀了什麼**：〔部分〕§4 註腳 3（p6），查證 2026-10-27。repo 已多處引用（`ADVISOR_REPORT_20260930.md:194` 等）。
- **RoPE 的處理**：預先算好的 chunk，K 要乘一個旋轉矩陣改成正確的位置；原文說這步「negligible overhead」，因為只做一次（p6）。**沒有拆開量**〔全文〕。
- **跟 Tiara 的關係**〔判讀〕：還原時位置不變，用不到。

#### Block-Attention for Efficient Prefilling（ICLR 2025；arXiv 2409.15355）
- **讀了什麼**：〔部分〕§2.2–2.3（p3–4）。
- **RoPE 的處理**：三步。先逆向旋轉回位置 0，再轉到新位置（式 1、2）。原文註腳說不考慮 KV cache 本身的成本（p3）。**要微調模型**才能適應 block 分段（摘要、p1）〔全文〕。
- **跟 Tiara**：不適用（要訓練、場景是 RAG）。

#### KVLink（NeurIPS 2025；arXiv 2502.16002）
- **讀了什麼**：〔部分〕§2.2（p3）。
- **RoPE 的處理**：存 W_{k,v}·x（不含 RoPE），拼接後才套全域的 RoPE；原文說「只引入可忽略的時間」，沒給數字（p3）。另外有可訓練的連結 token（摘要、p2）〔全文〕。

#### EPIC：Efficient Position-Independent Caching（venue 未查證；arXiv 2410.15332 v3）
- **讀了什麼**：〔部分〕§1–4.1（p1–4）。
- **事實**：每個 chunk 從位置 0 開始「編譯」；LegoLink 只重算每個 chunk 開頭的 k ≤ 32 個 token，讓它們知道自己不在開頭（p2、p4）。全文 grep「rope／rotary」0 筆，本次沒找到它對 RoPE 平移的描述〔全文 grep〕。

#### SparseX（arXiv 2606.01751 v2，2026-06-07；venue 未查證）
- **RoPE 的處理**：Delta-RoPE，成本 O(|S|·d_k)，「一次讀、轉、寫」；實作成一個融合的「複製＋對齊」kernel（p9）。**沒有單獨的時間數字**〔全文〕。其餘見 §4。

### 2.2 〔算術〕重新旋轉要多少
- Llama-3.1-8B 128K，全部的 K：131072 × 32 層 × 8 × 128 × 2 B＝8.59 GB。
- 讀一次、寫一次＝17.2 GB；以 H200 實測裝置內複製 4173.9 GB/s（arith §8）算，約 **4.1 ms**。
- 如果和反量化或粗底重建融在同一個 kernel（本來就要讀寫），增量是每個元素幾個 FLOP，可以忽略。
- 對照：同一段 KV 過 PCIe 是 0.30 s（§1.1）。

### 2.3 repo 現況與〔判讀〕
- **已在 repo 裡**：
  - A0 的粗底校準時「K 先逆 RoPE」（`A0_progressive_restore.md:44`），重建後再套 RoPE。這是 RoPE 結構在 Tiara 裡唯一實際用到的地方，L5a §0 第 2 點已經說明理由（RoPE 前 K 的秩最低）。
  - 跨請求共用時「K 存未旋轉形式、在 kernel 內才套 RoPE」的前作（MEPIC、MiniPIC）在 `docs/DIRECTIONS_20260927.md:326`。
- **〔判讀〕對還原的意義**：
  - 搬或重算都不受影響（位置不變）。
  - 縮圖：已經在用（RoPE 前的低秩）。
  - TransMLA 的不變性條件提供另一條路：在 RoPE 之後、跨 KV head、同頻率內做 PCA，不必逆 RoPE。但在 GQA 下每層只有 4–8 個 KV head，這個自由度很小。沒人在還原的情境量過，不建議投入。

---

## 3. softmax 正規化的結合律：(o, lse) 的合併

**白話**：注意力可以拆開算。把 KV 分成幾堆，各算一份「部分結果＋一個正規化數」，最後合起來就和一起算完全一樣。所以可以讓 CPU 在 KV 所在的地方算一部分，只把很小的結果傳回 GPU。這招和 Mamba 的平行 scan、FlashAttention 的 online softmax 是同一個代數結構（monoid）。

**數學**（FlashInfer §2.2，p3）〔全文〕
- [O(I∪J), LSE(I∪J)]＝[O(I), LSE(I)] ⊕ [O(J), LSE(J)]，其中
  O＝(e^{LSE(I)}O(I)＋e^{LSE(J)}O(J))／(e^{LSE(I)}＋e^{LSE(J)})，LSE＝log(e^{LSE(I)}＋e^{LSE(J)})。
- ⊕ 有結合律、交換律 → 任意順序合併。
- 傳回的量：每個 query token、每個 q head 傳 d_h＋1 個數；搬 KV 則是每個 context token 每層 2·n_kv·d_h 個數。

### 3.1 卡片

#### FlashInfer（venue 未查證；arXiv 2501.01005 v2）
- **讀了什麼**：〔部分〕§2.2–3.1（p3–4）。
- **事實**：把 attention state（O、LSE）定為注意力運算的標準輸出，⊕ 當成標準的歸約運算子。原文說 Flash-Decoding 和 Ring-Attention 用的就是這個性質（p3）〔全文〕。Flash-Decoding 的 split-K 是 decode 時沿 KV 長度切開、最後合併，這裡只採 FlashInfer 的轉述〔二手〕。「composable formats」用同一個 ⊕ 處理共享前綴（p4）。
- **跟 Tiara**：這是所有「拆開算再合併」的基礎，本身不是還原手段。

#### FastDecode（venue 未查證；arXiv 2403.11421 v1）
- **讀了什麼**：〔部分〕§1（p2）。
- **事實**：把模型切成 R-Part（含 KV cache，放 CPU）和 S-Part（GPU）；「在 CPU 上靠近 KV 計算」，只傳比 KV 小好幾個數量級的 activation（p2）。場景是 decode 的吞吐量〔全文〕。

#### NEO（venue 未查證；arXiv 2411.01142 v1）
- **讀了什麼**：〔部分〕§2.3–3.1（p3–4）。
- **事實**：
  - 把部分請求的 **decode** 注意力（計算和 KV）卸到 CPU。
  - prefill 批次「不卸載，直接在 GPU 跑」，因為 prefill 需要大量矩陣乘法（p4）〔全文〕。
  - 引用的 CPU 數字：約 200 GB/s、1.2 TFLOPS，對比 GPU 的 125 TFLOPS（p3）。

#### InstInfer（venue 未查證；arXiv 2409.04992 v1）
- **讀了什麼**：〔部分〕§III（p5）。
- **事實**：把 decode 的注意力卸到 computational storage（FPGA CSD）。用 A6000 與 Zynq7045 的 roofline 分析後結論是：**prefill 的注意力也應該留在 GPU**（p5）〔全文〕。

#### MagicPIG（venue 未查證；arXiv 2410.16179 v4）
- **讀了什麼**：〔部分〕§4（p9）、附錄 G（p25）。
- **事實**：
  - CPU 上做 LSH 抽樣注意力，sink 與 local token 留在 GPU；用 recursive attention（＝FlashInfer 的 ⊕）合併 CPU 與 GPU 的輸出（p9）。
  - 限制節明寫「還沒在 prefill 階段實作」（p25）〔全文〕。

#### RetroInfer（PVLDB 19(5) 2026，依 repo `L8a` 的書目；arXiv 2505.02922）
- **讀了什麼**：〔部分〕§4.4（p8）、§4（p9）、相容性段（p10）。repo 已有詳細卡（`docs/stage3_design_20261024/lit/L8a_progressive_query_aware_restore.md:240`）。
- **補的事實**：
  - 三個區域的注意力輸出合併（p9）。
  - prefill 時 Q、K、V 在 GPU 算，KV 非同步卸到 CPU，GPU 同時做分段分群；只有分群在 prefill 的關鍵路徑上（< 5%，p8）〔全文〕。
  - → prefill 的注意力在 GPU。

#### HGCA：Hybrid GPU-CPU Attention for Long Context LLM Inference（venue 未查證；arXiv 2507.03153 v1）
- **讀了什麼**：〔部分〕§1–3.1、§5 平台與 §5.1（p1–6、p10），圖 6 轉 PNG 看過，<https://arxiv.org/pdf/2507.03153>，查證 2026-10-27。
- **一句話**：近期的 KV 在 GPU 做完整注意力，較舊的 KV 在 CPU 做逐 head 的稀疏注意力，兩邊用 LSE 合併。
- **線性代數的關鍵一步**：兩邊各自產生部分輸出加正規化統計量，在 GPU 上合併。原文說傳回的量比整份 KV「小好幾個數量級」（p6）〔全文〕。
- **把問題分三種**（p2）：prefill（query：KV＝1:1，計算型）、append、decode（頻寬型）。原文主要處理 append 與 decode〔全文〕。
- **量到的交叉點**（圖 6，p5）：
  - 平台：Xeon Gold 6430（每 socket 32 核）＋A6000＋PCIe 4.0（單向 32 GB/s，p10）。
  - query 長 1：GPU（含從 host 搬 KV）比 CPU 慢。
  - query 長 32、batch 1：GPU 和 CPU 打平。
  - 所有設定下，PCIe 搬運都主導 GPU 那一側的時間〔全文＋看圖〕。
  - **沒寫的**：圖 6 用哪個模型（§5.1 的微基準是三個 OPT，原文寫「hidden dimension size of 128」，p10）；KV 軸「100–1000」的單位（每個時間只有零點幾毫秒，看起來是單層、短 KV）〔未查證〕。
  - 引言的 CPU 數字：FP16 1.229 TFLOPS 對 A6000 38.7 TFLOPS；記憶體頻寬 500 對 768 GB/s（p2）。
- **跟 Tiara 的關係**：見 3.3。
- **與既有整理的關係**：repo grep「HGCA」0 筆。新卡。

#### ScoutAttention（arXiv 2603.27138 v1）、HybridGen（arXiv 2604.18529 v1）
- ScoutAttention〔部分〕：
  - CPU 提前一層算區塊稀疏注意力，在 GPU 上用 FlashAttention 的方式合併（p4）。
  - 「只評估 decode 實例，prefill 另外處理」（p6）〔全文〕。
- HybridGen〔摘要＋部分〕：CPU 算部分 attention logit，再和 GPU 的結果串接；目標是 CXL 分層記憶體。圖 3 區分 prefill 與 generation，但我沒確認它在 prefill 時是否用 CPU〔未查證〕。

### 3.2 R2 的否決和文獻一致嗎？〔算術〕
- R2（`05_rejected.md:9`，arith §7 第 85–92 行）：
  - 在 CPU 上對 CPU 層的 KV 算新問題的注意力。
  - 假設 16 核、3.2 TFLOPS、整份 KV 過 PCIe（約 57.6 GB/s）。
  - 結果：問題超過 7（Qwen-7B 256K）或 11（Qwen-14B 256K）個 token，CPU 就輸。
- **通式**：
  - 每個 context token、每層：搬運 2·n_kv·d_h·2 B；每個 query token 要 4·n_q·d_h FLOP。
  - FLOP／byte／query＝n_q／n_kv＝**g**。
  - 交叉點 q\*＝C_CPU／(BW_PCIe·g)。
- 代入：

| 情況 | C_CPU | BW | g | q\* |
|:--|--:|--:|--:|--:|
| HGCA 的硬體（MHA 當 g＝1） | 1.229 TF | 32 GB/s | 1 | **38**（原文量到約 32） |
| R2：Qwen-7B 256K | 3.2 TF | 57.6 GB/s | 7 | **7.9**（arith 寫 7） |
| R2：Qwen-14B 256K | 3.2 TF | 57.5 GB/s | 5 | **11.1**（arith 寫 11） |
| 同法：Llama-3.1-8B | 3.2 TF | 57.6 GB/s | 4 | 13.9 |

- **結論**〔判讀〕：
  - R2 的算術形式和 HGCA 量到的交叉點吻合（MHA、PCIe 4.0 下約 32–38）。
  - GQA 讓每 byte 的計算量多 g 倍，交叉點跟著縮小 g 倍。
  - 回來那一輪的新問題是幾十個 token 的 prefill，正好落在 CPU 輸的那一側。
  - 所有讀到的系統（NEO p4、InstInfer p5、MagicPIG p25、RetroInfer p8、ScoutAttention p6）都把 prefill 的注意力留在 GPU。
  - **R2 的否決和文獻一致**，而且 3.2 TFLOPS 是偏樂觀的假設：C 越小，交叉點越小，否決越穩。
- **文獻沒涵蓋的一點**：回答期間的 decode 是 q＝1，CPU 在單層、短 KV 時可以贏（HGCA）。但每一步 decode 都得把 CPU 上的整份 KV 讀一次，這時受 DRAM 頻寬限制。所以「整輪留在 CPU」要和 decode 步數一起算。NOT_MEASURED，本檔不推。

### 3.3 〔判讀〕對 Tiara 還原的意義
- **搬**：⊕ 讓「先到的 block 先算」成為可能。可以在一層裡把粗底（常駐 GPU）中**沒被挑中的 block** 的部分注意力 (o, lse) 先算好，挑中的精修 block 到了再算它們的部分結果、合併。這是「同層內」的重疊；跨層的流水 repo 已經有（M02 逐層挑塊，`02_bytes_quality.md`）。上限是每層「粗底那一份的注意力時間」，相對於一層的精修傳輸很小。
  - 本次搜尋沒找到在還原情境明確這樣做的（查詢字串：`overlap KV cache loading with attention computation chunk-wise partial attention merge as blocks arrive prefill offloaded KV restore`；只找到逐層預載，例如 CachedAttention、Mooncake，〔摘要〕）。不建議單獨立題。
- **重算**：⊕ 不改變重算的成本。
- **縮圖**：A0 的「挑中的 block 整塊取代粗底」，在數學上就是對（未挑中的粗底 ∪ 挑中的精修）做一次 ⊕。所以不論先算哪一份，結果都一樣，可以放心重排計算順序。

---

## 4. 注意力圖樣的便宜估計：讓（重新）prefill 變稀疏

**白話**：先花一點計算猜出「哪些 query block 需要哪些 key block」，只算那些。猜的方法都是線性代數的近似：抽幾個 q、對 Q、K 做平均池化、或只沿反對角線取樣。

**數學**
- 真正的圖樣 A＝softmax(QKᵀ/√d)。各家估計 Â 的方式：
  - MInference：Q[−64:]·Kᵀ。
  - FlexPrefill：ā＝softmax(avgpool(Q̂)·avgpool(K)ᵀ/√d)，這是 QKᵀ 在區塊平均子空間上的秩約化。
  - XAttention：每個 block 內沿步長 S 的反對角線求和。
- 依 Â 挑出累積機率 ≥ γ 的 block，用 block-sparse kernel 計算。

### 4.1 卡片

#### MInference 1.0（NeurIPS 2024；arXiv 2407.02490 v2）
- **讀了什麼**：〔部分〕§1–4（p1–6）。
- **估計**：Vertical-Slash head 用最後 last_q 個 query 估垂直與斜線的索引（p2）；實驗設定 last_q＝64、block 64（p6）〔全文〕。
- **數字**：LLaMA-3-8B、1M token、單張 A100，prefill 最多快 10×（p1、p2）〔全文〕。
- **前提**：不用訓練；只用在第一次 prefill。

#### XAttention（venue 未查證；arXiv 2503.16428 v1）
- **讀了什麼**：〔部分〕§2–4（p1–8）。
- **估計**：block 內反對角線（步長 S）求和當重要性分數，再依閾值挑 block（p1–2）。消融比較了 S＝4、16、64（p6）〔全文〕。
- **數字**：256K 時 prefill 注意力最多快 13.5×（密度 7.32%）與 9.8×（6.89%）（p6）。另有 Llama-3.1-8B-Instruct 的 RULER／LongBench 比較（表 1、2）〔全文〕。

#### FlexPrefill（ICLR 2025；arXiv 2502.20766）
- **讀了什麼**：〔部分〕§3（p4–5）。
- **估計**：取最後 block_size 個 query 當代表，ā 用池化後的 Q·Kᵀ，â 用逐點 softmax 後再池化；兩者 JS 散度的平方根決定這個 head 用「query 感知」還是固定的垂直斜線圖樣（p4）。實驗含 Meta-Llama-3.1-8B〔全文〕。

#### SampleAttention（venue 未查證；arXiv 2406.15486 v3）
- **讀了什麼**：〔摘要＋部分〕摘要與 §1（p1–2）。
- **估計**：兩階段、由 query 引導的 KV 篩選（抽樣 query 找重要的列），加上固定的局部窗口。
- **數字**：TTFT 最多少 5.29×（p2）；模型 ChatGLM、YI、InternLM〔部分〕。

#### SparseX：Segment-Level KV Cache Sharing for Interleaved LLM Serving（venue 未查證；arXiv 2606.01751 v2）
- **讀了什麼**：〔全文〕§1–5（p1–20），<https://arxiv.org/pdf/2606.01751>，查證 2026-10-27。
- **一句話**：非前綴的片段重用。新內容的 query（Sparse-Q）自然出現在序列裡，用它們估「哪些重用的 token 需要修正」，一次前向做 Sparse-KV 重算。前幾層用完整注意力，後面的層才改稀疏（摘要，p1）〔全文〕。
- **線性代數的關鍵一步**：拿 Sparse-Q 對所有 key 算注意力權重，每個位置加總成重要性（§3.2，p10）。這是 MInference「last-q 探測」的推廣：把固定的尾端 query 換成重用情境裡自然出現的 query（p5、p7、p10）〔全文〕。
- **成立的前提**：
  - 不用訓練。
  - **整個執行都在 GPU 記憶體裡**；原文明寫「不是在 disk、CPU、GPU 間搬資料的多層快取系統」（p14）。
- **量到的數字**：
  - 平台：8×RTX 4090、vLLM 0.11.0（p18）。
  - 表 1（p19），Qwen3-32B 的 LOCOMO，分數／TTFT：
    - 全部重算 0.68／9.23 s。
    - naive 0.54／0.48 s。
    - CacheBlend 0.60／1.32 s。
    - EPIC 0.55／0.59 s。
    - SparseX（有 full+sparse）0.67／0.87 s。
  - 另有 RULER 四個子任務（表 2）與多 agent（表 3，含 Qwen2.5-7B／72B）〔全文〕。
- **跟 Tiara 的關係**〔判讀〕：它的「稀疏」是**挑哪些 token 重算**，用來修跨片段的注意力；不是用稀疏注意力重算整段前綴，也不碰慢層。
- **與既有整理的關係**：`mechanisms/04_recompute_scheduling.md:31` 與 `07_lit_quickcheck.md:16` 的描述（標摘要級）和全文**一致**；本卡把它升級為〔全文〕，並補上「只在 GPU 記憶體」這一點。

#### RaBitQCache（venue 未查證；arXiv 2606.31519 v1）
- **讀了什麼**：〔摘要＋部分〕摘要、表 2–4 標題（p1、p6）。
- **事實**：用**隨機旋轉後的二值化 K** 做注意力權重的無偏估計（摘要稱有誤差界），配合 binary×INT4 運算與自適應 top-p 取回。評測含 LLaMA-3.1-8B-Instruct 的 LongBench、RULER（8K–64K）〔摘要＋部分〕。
- **跟 Tiara**〔判讀〕：這是「旋轉＋1 bit」當**挑 block 的索引**，和 L7d §1 第 2 點列的 FIER、YAKV 同一類（decode）。repo 已在 A0 用 KSCAN-2b 測過類似的東西（`A0_progressive_restore.md:135`）。

### 4.2 repo 現況
- M12「先近似後補精」（`mechanisms/04_recompute_scheduling.md`）用自己寫的 block-sparse 重算。估計器是「這一塊 512 個 query、這一組 q head 的平均」（`runs/G_M12_sparse_recompute.md` §1），也就是 FlexPrefill ā 的池化估計。
- G-M12 的進度：
  - 速度已量、還沒看（`results/RUNLOG_H200_S3.md:286`）。
  - Q128 的品質自檢沒過，正在診斷（`RUNLOG_H200_S3.md:436` 起）。
  - 本檔不評論它的數字。

### 4.3 〔判讀〕對 Tiara 還原的意義
- **重算**：Cake 的重算是對**同一段**舊 context 再做一次 prefill。這和第一次 prefill 有一個差別：**第一次 prefill 時，真實的區塊級注意力圖已經算出來過了**。
  - 如果寫入時順手把每層、每個 KV 組的區塊分數存下來（Llama 128K：252² 個 block 對 × 32 層 × 8 組 × 2 B ≈ 32.5 MB〔算術〕，約占 16 GiB KV 的 0.2%），回來重算時就可以直接用「記下的遮罩」，不必再估。
  - 需要注意兩點：
    - 重算前段時，後段（由慢層載入）也要用；遮罩只決定前段 query 看哪些 key，和估計器的角色一樣。
    - 如果前段用近似 KV，後面各層的 hidden 會漂移，記下的圖可能失準（和 MInference 的估計同樣的風險）。
  - 本次搜尋沒找到「同一段 context 重算時重用記下的稀疏遮罩」（查詢字串：`reuse recorded sparse attention pattern mask from previous prefill to accelerate recomputation of same context KV cache`；最接近的 SamKV、KVzip 是多文件稀疏化與壓縮，〔摘要〕）。這是 M12 的一個變體，**不是新方向**：它只省掉估計那一步，而 XAttention p6 說估計本身已經比 MInference、FlexPrefill 便宜很多。
- **縮圖**：A0 的挑塊打分（新問題的 q 對粗底的 K 做 softmax，`A0_progressive_restore.md` §1.4）本身就是一個「用便宜的 K 估 QKᵀ」的估計器；它和本節各家的差別在於估的是**新問題**對 context 的需求，而不是 context 自身的圖樣。

---

## 5. 量化前的正交旋轉（Hadamard／學習的旋轉）與低位元的縮圖

**白話**：先把每個 token 的向量乘一個正交矩陣，把少數特別大的維度攤平，再量化，誤差會變小。可是 K 的大值是「固定在某些 channel」，KIVI 已經用每個 channel 各自的 scale 處理掉了；這時再旋轉，反而把固定的結構攪亂。

**數學**
- H 正交（HHᵀ＝I）→ (qH)(kH)ᵀ＝qkᵀ。所以 K、Q 一起旋轉，注意力分數不變；V 的旋轉可以吸收進 W_O。
- 量化誤差：逐 token 量化的 scale 取決於 max|kH|。旋轉會讓 max 接近平均水準，也會把固定 channel 的離群值分攤到所有 channel。

### 5.1 卡片

#### QuaRot（NeurIPS 2024；arXiv 2404.00456 v2）
- **讀了什麼**：〔部分〕§5.3、附錄 A.3（p9、p14）。
- **KV 的部分**：只量化 KV、其他保持高精度，group 128、非對稱。表 6（p14），**Llama-2** 的 WikiText-2 PPL（7B）：
  - 16/16：5.47。
  - K4V4：5.51。
  - K3V3：5.68。
  - K2V2：9.23。
  - 原文說 K 比 V 敏感（p14）〔全文〕。
  - **沒有 Llama-3.x、Qwen2.5 或長 context 的 KV-only 數字**。

#### SpinQuant（ICLR 2025；arXiv 2405.16406）
- **讀了什麼**：〔部分〕§3（p3–5）。
- **事實**：學習旋轉 R1、R2（吸收進權重）；KV 低位元時再加線上 Hadamard R3、R4（p2）；R3 在需要低位元 KV 時插入（p5）〔全文〕。
- **要訓練**：Cayley 最佳化學 R1、R2。結果以 W4A4KV4 的零樣本推理為主，**沒讀到 KV-only 的長 context 數字**〔部分〕。

#### RotateKV（venue 未查證；arXiv 2501.16383 v2）
- **讀了什麼**：〔部分〕§4–5（p3–6）。
- **事實**：在 **RoPE 前**做分組 head 的旋轉（FWHT），並對 attention sink 特別處理。逐 token、group 128（p6）。
- **數字**（表 4，p6），WikiText-2 PPL，LLaMA-3-8B（FP16 5.75）：

| 位元 | QuaRot | KVQuant | RotateKV |
|:--|--:|--:|--:|
| 4-bit | 5.83 | 5.79 | 5.79 |
| 3-bit | 6.21 | 5.95 | 5.94 |
| 2-bit | 21.43 | 6.75 | 6.69 |

  - 長 context 只有 LLaMA-2-7B-80K 的 40K NIAH 與 LongBench 的 8 個子任務〔全文〕。
  - **沒有 Llama-3.1／Qwen2.5 的 RULER**。

#### KVQuant（NeurIPS 2024；arXiv 2401.18079 v6）
- **讀了什麼**：〔部分〕§3.1–3.2（p4）。
- **事實**：K 逐 channel 量化；在 **RoPE 前**量化 K，反量化後再用融合 kernel 套 RoPE；另有非均勻量化與 dense-and-sparse 離群值。模型含 Llama-3（p1、p4）〔全文〕。這不是旋轉，是「換量化軸＋換量化點」。

#### KVLinC（Under review；arXiv 2510.05373）
- **讀了什麼**：〔部分〕§3–4、表 1–3（p4–8），<https://arxiv.org/pdf/2510.05373>，查證 2026-10-27。
- **一句話**：只用 Hadamard 加輕量的線性修正 adapter，把 KV 壓到 2 bit。
- **關鍵發現**（圖 3、觀察 1–2，p4）：
  - 2 bit 時，「K 不旋轉、逐 channel；V 先 Hadamard（後乘）、再逐 token」最好。
  - Hadamard 前乘（混 token）幾乎總是更差。
  - QuaRot 的「K、V 都旋轉＋都逐 token」比 KIVI 的設定差〔全文〕。
- **數字**（2 bit、group 128、最近 128 token 全精度）：
  - 表 1（p7），平均位元 2.46：
    - Qwen2.5-7B：KIVI 的 WikiText 11.2、GSM8K 71.0；QuaRot 3380.0／0.1（壞掉）。
    - Llama-3.1-8B：KIVI 7.8／34.1；QuaRot 7.3／26.8。
  - 表 3（p8），Llama-3.1-8B、**不訓練**的「K 逐 channel＋V 經 Hadamard 逐 token」：7.2／36.9（KIVI 7.8／34.1）。
  - 加上要訓練的修正（KVLinC，2.71 bit）：7.1／40.9。
  - 長 context 只有 RULER 4K、8K（表 2，p7）。表 2 沒有 7B／8B 的 instruct 模型（只有 Llama-3.2-3B、Qwen-2.5-3B、Qwen-3-4B）〔全文〕。
- **跟 Tiara**：見 5.3。

#### XQuant 的 latent（見 §1.2）
- X·U_k（W_K 的左奇異向量）把離群值集中在**第一個 channel**，所有層、多個模型都這樣（p8–9，圖 B.2、B.3）〔全文〕。
- 這正是「轉到 SVD 基底後，前幾維特別大」的現象，repo `L5d_stacking.md:265` 已經整理過。

### 5.2 本次沒找到的
- 「只對 KV 做 Hadamard、INT2–INT4、Llama-3.1-8B 或 Qwen2.5-7B、RULER 64K 以上」的同位元組數字，本次搜尋沒找到（查詢字串：`Hadamard rotation KV cache quantization 2-bit Llama-3.1-8B RULER LongBench KV-only rotation`）。
- 最接近的是 KVLinC（RULER 只到 8K、3–4B 模型）。
- repo 的 `L6_mainstream_practice.md:334` 記了 D-Quant（2609.19880）有 QuaRot 2.25 bit 在 RULER 32K–128K 的基線〔repo 既有，本次沒重讀〕。

### 5.3 〔判讀〕旋轉能不能把 A0 推過「條件 2」？
- **A0 差在哪**：最接近 GO 的 Q128 B24·1/8·int4·ORACLE，對 E0 的點估計 −0.35，CI [−2.05, +1.44]，條件 2 要求下界 ≥ −2，差 0.05 pp（`RUNLOG_H200_S3.md:467`、`A0_progressive_restore.md:190`）。原文的說法是：288 題時配對 CI 的半寬約 2–3 pp，點估計接近 0 也很難過。**這是統計功效的問題，不是方法偏差的問題**。
- **A0 的兩個量化點**：
  1. **精修**＝INT4-KIVI 的原始 KV（`A0_progressive_restore.md:53`）。我們第二階段量過，INT4-KIVI 在 32K–128K 兩個模型都不掉（`L5a_lowrank.md:50`）；文獻在 4 bit 也都接近 FP16（QuaRot 表 6、RotateKV 表 4）。→ 旋轉可以改善的空間接近 0。
  2. **粗底**＝xKV 跨層低秩，係數 K 用 INT4、V 用 INT2（B24）。係數已經在**校準好的 SVD 基底**裡（K 逆 RoPE、不去平均，`A0_progressive_restore.md:44`），而且 K、V 係數都是「每個主成分各自 scale、沿 token 每 32 個一組」（`code/m16_xstack.py:23–26`）。
     - 這已經是「資料自適應的正交旋轉＋依成分給 scale」的形式。
     - Hadamard 會把成分的變異數攪勻，和依成分給 scale 衝突（repo 已寫在 `L5d_stacking.md:270`）。
     - KVLinC 的「V 旋轉後逐 token」之所以有效，是因為原始 V 逐 token 量化時 channel 間不均；A0 的 V 係數是逐成分量化，不是同一個情境。
- **〔判讀〕結論**：
  - 旋轉不是推過條件 2 的合理槓桿。
  - 如果要試，唯一說得通的位置是 B24／B32 的 **V 係數 INT2**：在 Hadamard 前先把各成分除以自己的 σ，旋轉後再逐 token 量化。但這等於換掉 X4 Q-A 選的係數格式，要重新登記。
  - A0-C（864 題，`runs/A0C_confirm.md`，job 22662–22671）會直接回答「條件 2 是不是只差在題數」。在它出來之前，不建議為了 0.05 pp 動格式。

---

## 6. 權重吸收（MLA 式）：把 W_UK 吸收進 W_Q，讓注意力直接在 latent 上算

**白話**：如果 K＝c·W_UK（c 是存下來的 latent），那 q·kᵀ＝(q·W_UKᵀ)·cᵀ，W_UK 可以先乘進 query，就不必把 K 還原出來。可是 RoPE 夾在中間（q·R_m·R_nᵀ·kᵀ），位置相關的旋轉讓 W_UK 吸收不進去。

**數學**：q R_m (c_n W_UK R_n)ᵀ＝q R_m R_nᵀ W_UKᵀ c_nᵀ。R_m R_nᵀ＝R_{m−n} 和 W_UKᵀ 一般不可交換，所以吸收不了。DeepSeek MLA 的做法是另外拉出一小段帶 RoPE 的 K（decoupled RoPE）。

### 6.1 卡片
- **Palu**（ICLR 2025 依 repo `L5d_stacking.md:71`；arXiv 2407.21118 v2）〔部分，repo 已有完整整理〕：
  - V 的重建矩陣吸收進 W_O；K 要「重建 → RoPE」，為此寫了 kernel（L5a §0 第 1、2 點）。
  - 原文摘要：RoPE 型注意力最多快 1.89×、非 RoPE 型 2.2×，加上量化最多 2.91×、6.17×（p1–2）〔全文〕。
- **TransMLA**（venue 未查證；arXiv 2502.07864 v5）〔部分，§1–2（p1–3）〕：
  - RoRoPE：對 K 做 PCA，同頻率、跨 head 旋轉，把位置資訊集中到第一個 head（不變性條件見 §2）。FreqFold 合併相鄰頻率。剩下的 K_nope 丟掉 RoPE，和 V 做平衡後的聯合 PCA（BKV）（p2）。
  - **Llama-2**：完全不訓練時，壓 68.75% 掉 1.65%；壓 93% 要 6B token 微調才大致恢復；vLLM 上最多快 10.6×（p2–3）〔全文〕。
  - 本次沒讀它在 Llama-3／Qwen2.5 上的數字。
- **MHA2MLA**（venue 未查證；arXiv 2502.14837 v2）〔摘要〕：partial-RoPE（移除貢獻小的維度的 RoPE）加 K、V 聯合 SVD，需要 0.6–1% 的資料微調。

### 6.2 〔判讀〕
- 對還原：吸收讓 decode 時不必還原 K，但前提是 RoPE 已被移出大部分維度。這要嘛改架構（MLA），要嘛微調（TransMLA、MHA2MLA）。
- 不訓練、又能用在 Llama-3.1／Qwen2.5 上的，只有 Palu 式「V 吸收、K 重建」，repo 已涵蓋（L5a、L5d）。
- 沒有新的還原手段。

---

## 7. 查不到、沒查證、和既有整理的差異

**查不到**（≠不存在，CLAUDE.md 規則 7；查詢字串寫在各節）：
- GQA 上「存低秩或量化的 hidden state 取代 KV」，XQuant 以外的作品（§1.2）。
- 「同一段 context 重算時，重用寫入時記下的稀疏遮罩」（§4.3）。
- 還原情境中，「同層內，粗底的部分注意力先算、精修到了再 ⊕ 合併」（§3.3）。
- KV-only Hadamard 在 Llama-3.1-8B／Qwen2.5-7B、RULER ≥ 64K 的同位元組數字（§5.2）。

**沒查證**：
- HGCA 圖 6 的模型與 KV 軸單位。
- HybridGen 在 prefill 是否用 CPU。
- XQuant 的實際 GEMM 延遲（原文沒量）。
- Flash-Decoding 原始部落格（只採 FlashInfer 的轉述）。
- Grouped Value Attention（2609.13285）。
- TransMLA 在 Llama-3／Qwen2.5 的數字。
- SampleAttention 的實驗細節。
- EPIC、XAttention、FlashInfer、NEO、MagicPIG、InstInfer、HGCA、XQuant、RotateKV、TransMLA、MHA2MLA、SampleAttention 的 venue。

**與既有整理的差異**：
- `mechanisms/04_recompute_scheduling.md:30` 把 FlexPrefill、XAttention 標〔未查證〕。本檔已讀方法段，兩者都是「第一次 prefill 的稀疏化」，原本的描述正確。
- `:31` 的 SparseX 摘要級描述和全文一致；補「只在 GPU 記憶體」。
- `arith.md` §7 的交叉點 7／11：本檔用通式 q\*＝C／(BW·g) 重算得 7.9／11.1，一致（arith 取整數部分）。

---

## 8. 參考文獻（IEEE 格式；查證日期 2026-10-27）

[1] S. Gao, Y. Chen, J. Shu, "Fast State Restoration in LLM Serving with HCache," EuroSys 2025. arXiv:2410.05004. https://arxiv.org/abs/2410.05004
[2] A. Tomar, C. Hooper, et al., "XQuant: Breaking the Memory Wall for LLM Inference with KV Cache Rematerialization," arXiv:2508.10395, 2025. https://arxiv.org/abs/2508.10395
[3] J. Yao et al., "CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion," EuroSys 2025. arXiv:2405.16444. https://arxiv.org/abs/2405.16444
[4] D. Ma et al., "Block-Attention for Efficient Prefilling," ICLR 2025. arXiv:2409.15355. https://arxiv.org/abs/2409.15355
[5] J. Yang, B. Hou, et al., "KVLink: Accelerating Large Language Models via Efficient KV Cache Reuse," NeurIPS 2025. arXiv:2502.16002. https://arxiv.org/abs/2502.16002
[6] J. Hu et al., "EPIC: Efficient Position-Independent Caching for Serving Large Language Models," arXiv:2410.15332 (v3). https://arxiv.org/abs/2410.15332
[7] Q. Zhang et al., "SparseX: Efficient Segment-Level KV Cache Sharing for Interleaved LLM Serving," arXiv:2606.01751 (v2), 2026. https://arxiv.org/abs/2606.01751
[8] Z. Ye et al., "FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving," arXiv:2501.01005 (v2). https://arxiv.org/abs/2501.01005
[9] J. He, J. Zhai, "FastDecode: High-Throughput GPU-Efficient LLM Serving using Heterogeneous Pipelines," arXiv:2403.11421, 2024. https://arxiv.org/abs/2403.11421
[10] X. Jiang, Y. Zhou, et al., "NEO: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference," arXiv:2411.01142. https://arxiv.org/abs/2411.01142
[11] X. Pan et al., "InstInfer: In-Storage Attention Offloading for Cost-Effective Long-Context LLM Inference," arXiv:2409.04992. https://arxiv.org/abs/2409.04992
[12] Z. Chen, R. Sadhukhan, Z. Ye, et al., "MagicPIG: LSH Sampling for Efficient LLM Generation," arXiv:2410.16179 (v4). https://arxiv.org/abs/2410.16179
[13] Y. Chen et al., "RetroInfer: A Vector Storage Engine for Scalable Long-Context LLM Inference," PVLDB 19(5), 2026. arXiv:2505.02922. https://arxiv.org/abs/2505.02922
[14] W. Deng, Y. Yang, P. Du, L. Xiang, et al., "HGCA: Hybrid GPU-CPU Attention for Long Context LLM Inference," arXiv:2507.03153, 2025. https://arxiv.org/abs/2507.03153
[15] Q. Zhang et al., "ScoutAttention: Efficient KV Cache Offloading via Layer-Ahead CPU Pre-computation for LLM Inference," arXiv:2603.27138, 2026. https://arxiv.org/abs/2603.27138
[16] M. Lin, X. Wang, G. Cox, "HybridGen: Efficient LLM Generative Inference via CPU-GPU Hybrid Computing," arXiv:2604.18529, 2026. https://arxiv.org/abs/2604.18529
[17] H. Jiang et al., "MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention," NeurIPS 2024. arXiv:2407.02490. https://arxiv.org/abs/2407.02490
[18] R. Xu, G. Xiao, H. Huang, et al., "XAttention: Block Sparse Attention with Antidiagonal Scoring," arXiv:2503.16428. https://arxiv.org/abs/2503.16428
[19] X. Lai et al., "FlexPrefill: A Context-Aware Sparse Attention Mechanism for Efficient Long-Sequence Inference," ICLR 2025. arXiv:2502.20766. https://arxiv.org/abs/2502.20766
[20] Q. Zhu et al., "SampleAttention: Near-Lossless Acceleration of Long Context LLM Inference with Adaptive Structured Sparse Attention," arXiv:2406.15486 (v3). https://arxiv.org/abs/2406.15486
[21] W. Li, J. Dong, H. Zhang, et al., "RaBitQCache: Rotated Binary Quantization for KVCache in Long Context LLM Inference," arXiv:2606.31519, 2026. https://arxiv.org/abs/2606.31519
[22] S. Ashkboos et al., "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs," NeurIPS 2024. arXiv:2404.00456. https://arxiv.org/abs/2404.00456
[23] Z. Liu et al., "SpinQuant: LLM Quantization with Learned Rotations," ICLR 2025. arXiv:2405.16406. https://arxiv.org/abs/2405.16406
[24] Z. Su et al., "RotateKV: Accurate and Robust 2-Bit KV Cache Quantization for LLMs via Outlier-Aware Adaptive Rotations," arXiv:2501.16383 (v2). https://arxiv.org/abs/2501.16383
[25] C. Hooper, S. Kim, H. Mohammadzadeh, M. W. Mahoney, Y. S. Shao, K. Keutzer, A. Gholami, "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization," NeurIPS 2024. arXiv:2401.18079. https://arxiv.org/abs/2401.18079
[26] U. Saxena, K. Roy, "KVLinC: KV Cache Quantization with Hadamard Rotation and Linear Correction," arXiv:2510.05373, 2025. https://arxiv.org/abs/2510.05373
[27] C.-C. Chang et al., "Palu: KV-Cache Compression with Low-Rank Projection," arXiv:2407.21118 (v2). https://arxiv.org/abs/2407.21118
[28] F. Meng, P. Tang, X. Tang, Z. Yao, X. Sun, M. Zhang, "TransMLA: MLA Is All You Need," arXiv:2502.07864 (v5). https://arxiv.org/abs/2502.07864
[29] T. Ji et al., "Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs," arXiv:2502.14837 (v2). https://arxiv.org/abs/2502.14837

（作者名單只寫到 PDF 首頁看得到的部分；[8][17][19][20][22][23][27][29] 的作者名是依 arXiv 首頁慣例簡寫的第一作者，沒逐一核對全名〔未查證〕。）
