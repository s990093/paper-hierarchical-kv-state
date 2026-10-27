# LG　不改模型、只改「解壓」的線性代數：能不能把縮圖的重建 GEMM 拿掉？

> **日期**：2026-10-27。**執行者**：文獻＋推導 agent G。只讀、只上網；這台 Mac 只做 PDF 轉文字（`pdftotext -layout`）與〔算術〕（`python3 -I -c` 代公式），**沒有跑任何實驗或模擬**。只寫這一個檔。
> **使用者的問題（10/27 釐清）**：模型不動（Llama-3.1-8B、Qwen2.5-7B／14B；GQA、RoPE、softmax；不訓練），只在**壓縮後 KV 的反量化／解壓**上用線性代數技巧（對角化、結合律、吸收……「像 Mamba 那樣」）。
> **原文**：PDF 在 `<SCRATCH>/litG/pdf/<arXiv id>.pdf`，文字檔 `<SCRATCH>/litG/txt/`；頁碼一律是 **PDF 頁**（以 form feed 計）。GitHub 文件在 `<SCRATCH>/litG/gh/`（commit 9d7200c）。
> **標記**：〔全文〕〔部分〕〔摘要〕〔程式碼〕〔文件〕〔二手〕〔算術〕〔推導〕〔判讀〕〔未查證〕。〔推導〕＝我自己寫的代數推導，沒有任何論文替它背書；〔算術〕＝公式代入 repo 已量的常數，**不是量測**。本檔沒有一篇逐頁讀完，所以不標〔全文〕。
> **審查性質**：self-review，不是 cross-model review。
> **遵守的盲測**：G-M11 登記要求「沒看 VQ 對 INT4 的比」（`runs/G_M11_vq.md` 偏離 1）。本檔**沒有打開** `results/s3_h200/b0/gm11/*.csv`，也沒有算那個比。

---

## 0. 一頁結論（白話）

**一句話**：七招裡，真正能「不把縮圖還原成 BF16 就直接算注意力」的只有兩招（V 吸收、和 RoPE 可交換的 K 基底），而且**只在 decode（一次一個 token）時划算**。我們回來那一輪是 128–512 個 token 的問題 prefill，這時「先重建再算」在 FLOPs 上反而便宜 2–4 倍；而「在 latent 裡算」本身也是一個吃 tensor core 的 GEMM，**一樣會被算力型背景拖慢**。所以線性代數換不掉那個怕爭用的 GEMM，只能換它的形狀。

1. **痛點有多大**〔算術，§9〕：Llama-3.1-8B・128K、B24 縮圖，重建 K＋V 要 **11.8 TFLOP**。用 repo 實測的 H200 matmul 788.1 TFLOPS 算是 15 ms，用 arith 慣例的 50% 效率是 30 ms，再乘 GPU 忙時的 2.8 倍是 42–84 ms。這和新問題本身的注意力（n_q＝128 時 8.8 TFLOP，arith §6）同一個量級，也和 arith §9「逐層流水」的整個 TTFT（0.028 s，**該表沒算反量化／重建**）同一個量級。→ arith §9 的 TTFT 低估了縮圖路徑，重建這一項要補進去。
2. **招 1（V 吸收）和招 2（K 的 RoPE 可交換基底）的交叉點**〔推導＋算術，§2、§3、§9〕：在 latent 裡算，每個 q head 的成本是原本的 r/d 倍；重建是固定成本、所有 q head 共用。latent 比較便宜的條件是
   **n_q·g·(r − d) < r·d**，也就是 **n_q < n_q\* ＝ r·d ／ (g·(r − d)) ≈ d/g**（r ≫ d 時）。
   - 我們的跨層基底 r 很大（B24：K 544、V 832，對 d＝128 是 4.25、6.5 倍），所以 n_q\* 只有 **26–47**（Llama 38／42、Qwen-7B 26／35、Qwen-14B 30／34；V／K）。
   - 我們的問題 128–512 token → 重建便宜：Llama B24 n_q＝128，重建＋注意力 20.6 TFLOP，全 latent 47.3 TFLOP。
   - decode（n_q＝1）反過來：全 latent 0.37 TFLOP，若每步重建則 11.9 TFLOP（32 倍）。
   - **跨層共用不能換成「每個窗口算一次 P·A」**：P 每個 q head、每層都不同，而且下一層的 P 要等這一層算完。能做的只是把一層的 H·g 個 q head 疊成一個 GEMM（M＝H·g·n_q），FLOPs 不變。
3. **招 2 的「逐頻率、跨 head／層的複數基底」有沒有前作？有，而且結果是負的**：
   - 原理已知：TransMLA 的 RoRoPE（同頻率、跨 head、實部虛部用同一個實正交矩陣，p7 式 (19)）、CommVQ（每個 2 維子空間的碼字是 [[x, y], [−y, x]]＝複數純量，p3 Property 1）、RAP（以 RoPE 對為單位剪枝）。
   - **最直接的前作**是一個 GitHub 研究（PseudoHunt/rope-kv-compression，2026-09-23，非同儕審查）〔文件〕：Llama-2-7B-chat（MHA）、**同一層內**跨 32 head 的逐頻率複數 PCA、RoPE 完全精確、query 吸收、不重建。K 預算 25% 時保留 88.9% 能量（不受限的 RoPE 前 PCA 是 97.7%），注意力輸出誤差是 RoPE 前 PCA 的 **2.69 倍** → 照它自己的登記停在 Gate 3。原因：**K 的低秩大多是「跨頻率」的**，任何和 RoPE 精確可交換的線性映射都用不到。
   - **跨層窗口版**（我們的 W＝4、32 個 head-層）本次沒找到。JoLT（2607.12550 v4，p9）明寫「把旋轉摺進因子打分」是 kernel 端的主要未解問題。
   - 〔判讀〕repo 的 K1（每 head RoPE 前 95% 能量在 36–40 維，`results_analysis.md:87`）就是「同一 head 內跨頻率」的冗餘，正好是逐頻率家族用不到的那一塊；K1b（相鄰層同一 token cosine ≈ 0，`results_analysis.md:93`）暗示跨層的同頻率分量也不太相關。**預期跨層版一樣輸**，但只要用既有 K dump 算能量就能驗（§3.5，不需要 Q、不需要 GPU）。
4. **招 3（scale／zero 吸收）數學上精確，但對我們沒東西可省**：純量反量化本來就只要 0.031 ms／64 MiB（`F3_dequant_fp8.md:56`）〔複核 2026-10-27：原未寫平台，F3 是 **MI300X**（F3 §4.1 標題）；H200 的融合 kernel 是 0.028–0.087 ms（`results/m10_sweep_h200/dqsweep/A/kernels_llama.csv`），結論不變〕。它也碰不到低秩 GEMM。唯一的觀察：QServe 量到「融合的 KV4 注意力在 A100 上會被 CUDA core 卡住」（p10 表 1），所以解壓融進注意力不一定免費。
5. **招 4（不需反轉的正交轉換）**：縮圖的基底本來就是正交欄（`np_xl_calib` 取 eigh 的特徵向量），重建就是乘 Uᵀ，沒有反矩陣可省。能不能吸收進 q，就回到招 1／2。INT4 精修塊上的 Hadamard：LC §5.3 已判沒有空間。
6. **招 5（PQ／VQ 查表）是唯一完全不用 tensor core GEMM 的**，但條件很窄：查表要放得進 SRAM。G-M11 的格式每個（q 列, head）查表 8,192 項；prefill 時每個 KV head 有 n_q·g＝512 列，放不下。所以它只適合 decode。而且要換掉縮圖格式；再加上 RoPE 時，CommVQ 的可交換碼本把成本降到約 **(R＋1)/2 倍的普通注意力**（R＝11 → 約 6 倍，p6）〔複核 2026-10-27：原寫「(R＋1) 倍」，原文（arXiv HTML 的 TeX 為 \frac{R+1}{2}）是 (R＋1)/2，改為 (R＋1)/2〕，不是比普通注意力便宜。repo 的 G-M11 時間判定 NOT_DETERMINED。
7. **招 6（重建融進注意力的 tile，BF16 不寫回 HBM）**：
   - 省的是 HBM：Llama 128K 寫＋讀一次 BF16 約 8.2 ms〔算術〕。FLOPs 不省。
   - **prefill 時還會變多**：每個 query tile 都要重建一次同一塊 KV，倍數 ⌈n_q·g／B_M⌉。B_M＝128 時，Llama n_q＝128 是 4 倍。
   - 文獻也顯示融合版只在 decode＋稀疏時贏：
     - xKV 全部重建的 kernel 在 A100 上一律慢於 FA2，122K 時約 0.4×（p7）。
     - JoLT 融合 kernel 是 SDPA 的 7.5–65.9 倍慢（p8）。
     - Palu 的 1.89× 是對 HF eager、MHA、RTX 4090（p9）。
8. **招 7（直接在壓縮域選塊）**：RoPE 前 latent 打分不精確。SALS 的重疊分數：第 2–29 層 >90%，第 0、1 層 <50%（p5 圖 2）。其餘已在 L7d。

**建議**〔判讀〕：
- **不要為了「拿掉重建 GEMM」改縮圖格式**。
  - 在我們的 prefill 長度下沒有一招贏，而且 latent 版本身也是 GEMM。
  - repo 的「忙時最快格式翻轉」（X3）和「還原給多少 SM」（B1）仍然是處理這個痛點的正確層次。
- 要縮小這一項，只有兩條路：
  - 少重建一點：選 B32（比 B24 少 23% FLOPs，§9）；被挑中升級的塊不重建 V。
  - 把重建時間算進 arith §9 與 C0 的成本模型。
- **可以便宜做的檢查**（不是閘；要做照慣例先登記，交給 H200 session）：
  - 用既有 K dump 算「逐頻率複數 PCA（跨 32 head-層）vs 現在的實數聯合基底」在同秩下的能量。
  - 判準照 PseudoHunt 自己的教訓：比**損失能量的比值**，不是比保留能量的百分點；比值 ≤2.25 才繼續。
  - 過了，decode 時 K、V 都能在 latent 裡算（§3）。
- **decode 時可以直接用的**：V 吸收（招 1）不用改格式，現在的實數 V 基底就能用（§2.4）。但它只幫 decode，不幫回來那一輪的 TTFT。

---

## 1. 總表

d＝128（head dim）、g＝GQA 組大小（Llama 4、Qwen-7B 7、Qwen-14B 5）、W＝4 層一窗、H＝KV head 數、r＝窗口聯合秩（B24 Llama：r_K 544、r_V 832）、n＝context、n_q＝問題長度（decode＝1）。

| # | 技巧 | 數學（一兩行） | 條件（精確） | 省掉什麼 | 代價 | 前作（出處） | 能不能用在我們的縮圖 | 拿掉怕爭用的重建 GEMM？ |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| 1 | **V 基底吸收**（不重建 V） | o＝P·(A·B_{ℓ,h})＝(P·A)·B_{ℓ,h}；再把 B 吸收進 W_O 只在 r_head < d 時划算 | 只要 V 沒有位置相關的轉換（V 不帶 RoPE ✓）；latent 便宜 ⇔ n_q·g·(r−d) < r·d | 重建 V 的 GEMM（2·n·r·H·d／層）＋BF16 materialize | 每個 q head 的 PV 從 d 變 r：r/d＝6.5（Llama B24） | Palu 式 (5) p4（每 head）；CommVQ 式 (22) p6；JoLT §4.5 p8（rank space 累加）；MLA 吸收 W_UV（DeepSeek-V2 p7） | **能，不用改格式**；但只在 n_q < 26–40 時省（§9） | decode：是（換成 M＝H·g 的小 GEMM）；prefill：**否**，latent 版 FLOPs 更多，而且也是 GEMM |
| 2 | **K 的 RoPE 可交換基底**（逐頻率、複數、跨 head／層） | RoPE 在頻率 j 對所有 head-層都乘同一個 e^{inθ_j}；ζ^{(j)}≈V_j c^{(j)} ⇒ RoPE(ζ)＝V_j(e^{inθ_j}c^{(j)})；q 吸收 V_j | 基底**必須**按頻率分塊、且每塊是複線性（§3.2 證明）；同樣的 n_q\* | 重建 K 的 GEMM、RoPE 只對 latent 逐元素做（每窗一次） | 秩被迫按頻率分配，丟掉跨頻率結構：MHA 前作在 25% 預算下輸出誤差 2.69 倍 | TransMLA RoRoPE（實數、單層、其餘去 RoPE、要微調）；CommVQ（VQ 版）；RAP（剪枝版）；PseudoHunt（複數、單層，負結果）；跨層版：沒找到 | 要**換基底並重新校準**；品質預期變差（〔判讀〕）；可用 K dump 先驗能量 | 同招 1：decode 是、prefill 否 |
| 3 | **scale／zero 吸收**（KIVI 式） | q·k̂＝Σ_c(q_c s_{c,γ})Q_{n,c}＋Σ_c q_c z_{c,γ}；o＝Σ(p_n s_n)Q_n＋Σp_n z_n；全 token 共用的 K 平均在 softmax 裡消失 | 平均或 zero 要對**所有** key 相同才會消失；RoPE 前的平均在 RoPE 後不再相同（§4.2） | 反量化的乘加；用 INT8 MMA 時省 FP 轉換 | q 也要量化（INT8 MMA）→ 多一份誤差；token 組間的 bias 要另算 | SageAttention 1 式 (6) p5；SageAttention2（Q 平滑＋GEMV）；QServe SmoothAttention 式 (7)–(9) p6；HACK 式 (4) p6；TurboAttention；BitDecoding；KIVI kernel | 縮圖係數可以，但純量反量化本來就約 0.03 ms／chunk，**沒東西可省** | **否**（不碰低秩 GEMM） |
| 4 | **不需反轉的正交轉換** | (qH)(kH)ᵀ＝qkᵀ；V 的 H 吸收進 W_O | H 正交、同一個 head、RoPE 之後（或與 RoPE 可交換） | 反轉（inverse）那一步 | 和 KIVI 逐 channel、和 SVD 逐成分 scale 衝突（LC §5） | QuaRot、Loki 引理 4.1 p5（任何正交 P）；RotateKV | 縮圖基底已正交，沒反矩陣可省；精修塊：LC §5.3 判沒空間 | **否** |
| 5 | **PQ／VQ 非對稱距離（查表）** | s＝Σ_子空間 LUT_s[code]，LUT＝q·codebook；V：o＝Σ_c C[c]·(Σ_{code＝c} p_n) | 查表放得進 SRAM：(q 列數)×m×K×bytes ≤ SRAM；有 RoPE 時碼本要和 RoPE 可交換 | 整個解壓（無 tensor core） | gather 走 CUDA core／shared memory（bank 衝突）；prefill 放不下；要換格式 | PQCache p9；MILLION p4、p6；VQ-LLM（HPCA'25）p1、p5；CommVQ p5–6、表 5 p9 | 要換掉低秩縮圖；repo G-M11 格式查表 8,192 項／列 | 拿掉 tensor core GEMM，但改吃 CUDA core（G-M11 正在量，NOT_DETERMINED） |
| 6 | **重建融進注意力 tile** | 在 SRAM 裡 K_tile＝A_tile·B_{ℓ,h}＋RoPE，再 QKᵀ | KV tile 只被一個 CTA 重建一次（n_q·g ≤ B_M，decode 成立） | BF16 寫回＋讀回 HBM（Llama 128K 約 8.2 ms） | FLOPs 不省；prefill 會重算 ⌈n_q·g／B_M⌉ 倍 | Palu Triton kernel p7；ShadowKV（L5a:128）；xKV p7；JoLT p8；SALS 融合 recover kernel | 可以，但只在 decode 有利 | **否**（只換掉 HBM 流量） |
| 7 | **壓縮域直接選塊** | 分數≈q_lat·a_n（RoPE 前、不轉） | 只要求排序對；RoPE 前打分忽略相對旋轉 → 不精確 | 選塊時不重建 | 早期層錯很多 | SALS p5 圖 2；Loki；LOCKS／SAKI（L7d） | A0 用的是重建後的粗底 K（精確 RoPE）；latent 打分是另一個近似 | 選塊那份的重建可以省；**最後的注意力仍要重建或 latent** |

---

## 2. 招 1：V 基底吸收——不重建 V

**白話**：注意力輸出是「機率加權的 V 平均」。V＝係數 × 基底，所以可以先用機率把係數加權平均，最後再乘一次基底。等於把「每個 token 都解壓」換成「只解壓一次平均值」。可是我們的係數比 V 本身還寬（832 對 128），所以「平均係數」本身就比較貴。

### 2.1 〔推導〕跨層聯合基底下的兩種算法

縮圖的 V（`code/m16_xstack.py` `np_xl_half_cal`，第 899–930 行；校準 `np_xl_calib` 第 933 行〔程式碼〕）：
- 窗口 w 的 V 疊成 Z ∈ ℝ^{n×(W·H·d)}。
- 係數 A＝Z·U_w ∈ ℝ^{n×r}，其中 U_w 的欄是正交的。
- 重建 Ẑ＝A·U_wᵀ。

第 ℓ 層、第 h 個 KV head 的那一段：V̂_{ℓ,h}＝A·B_{ℓ,h}，其中 B_{ℓ,h}＝(U_w 的對應列)ᵀ ∈ ℝ^{r×d}。

**(a) 先重建再算**（現在的做法；xKV 也是逐層重建，xKV 附錄 D.6 式 (4) p19）：
- 重建：2·n·r·H·d FLOPs／層。所有 q head 共用。
- PV：2·n_q·n·d FLOPs／q head，每層 H·g 個 q head。

**(b) 在 latent 裡算**：
- o_{ℓ,h,i}＝(P_{ℓ,h,i}·A)·B_{ℓ,h}。
- P·A 是 2·n_q·n·r／q head；乘 B 是 2·n_q·r·d／q head（和 n 無關，可略）。

**交叉**：(b) < (a) ⇔ H·g·n_q·n·r < n·r·H·d＋H·g·n_q·n·d ⇔ **n_q·g·(r − d) < r·d**。

所以 **n_q\* ＝ r·d／(g·(r − d))**；r ≫ d 時趨近 **d/g**。白話：「共用同一個 KV head 的 query 列數（n_q·g）少於約 d＝128 列」時，latent 才便宜。數值見 §9。

### 2.2 〔推導〕能不能讓跨層 latent 變便宜？

- **「每個窗口算一次 P·A、讓 head 共用」：不行。**
  - P 每個 q head 都不同，而且同一窗的 4 層之間有先後依賴：第 ℓ＋1 層的 P 要等第 ℓ 層的輸出。
  - 能做的只是把同一層的 H·g 個 q head 疊起來，[P_{ℓ,1,1}; …; P_{ℓ,H,g}]·A，變成一個 M＝H·g·n_q 的 GEMM。FLOPs 不變，但 A 每層只讀一次。decode 時 M＝32（Llama），比每 head 一個 GEMV 有效率得多。
- **「把 B_{ℓ,h} 再分解」：不行。**
  - B_{ℓ,h} 是 r×d，秩 ≤ d。若截斷成 B_{ℓ,h}≈E_h·F_h（E_h 是 r×ρ_h），可以先算 A·E_h（n×ρ_h）再在 ρ_h 維裡做注意力。
  - 但「先算 A·E_h」本身就是一個 2·n·r·ρ_h 的重建型 GEMM。只有 ρ_h ≪ d 才省。
  - V 的每 head 譜很平（K1：95% 能量要 108–110 維，`results_analysis.md:87`）→ ρ_h ≈ d，省不到〔判讀〕。
- **「B 吸收進 W_O」（Palu 的做法）：在我們這裡更貴。**
  - (P·A)·(B_{ℓ,h}·W_O^{(h,i)}) 的第二步是 2·n_q·r·d_model／q head。原本的路徑是先乘 B（2·n_q·r·d），再走原本的 W_O（2·n_q·d·d_model）。
  - 兩者差 (r − d)·d_model，r > d 時吸收更貴〔算術〕。
  - Palu 吸收划算，是因為它每 head 的 r_i < d_h（50% 壓縮時 V 只壓 25%，p8）。
- **softmax 正規化不影響吸收**：latent 累加器 Σ exp(s − m)·A_n 是線性的，online softmax 的重新縮放照常（JoLT §4.5 p8 就是這樣做的）〔部分〕。
- **V 的平均可以免費加回**：若 V 先去平均 μ_V，因為 Σ_n p_n＝1，所以 o＝P·(A·B)＋μ_V，**精確**〔推導〕。repo 的 xKV 校準不去平均（`A0_progressive_restore.md:44`），這只是給以後的選項。

### 2.3 卡片

#### Palu：KV-Cache Compression with Low-Rank Projection（ICLR 2025，依 repo `L5d_stacking.md:71`；arXiv 2407.21118 v2）
- **讀了什麼**：〔部分〕§3.1–3.2（p4–6）、§4.1 kernel（p7）、§4.5 延遲（p8–9），<https://arxiv.org/pdf/2407.21118>，查證 2026-10-27。repo 已有完整卡（L5a）。
- **一句話**：權重 SVD 後快取 latent；V 的重建矩陣吸收進 W_O，K 要「重建 → RoPE」。
- **線性代數的關鍵一步**：
  - 結合律 a_i W_o＝(p_i H_{v,i})(B_{v,i} W_o)（式 (5) p4）。
  - 無 RoPE 時 K 也能吸收進 W_q（式 (6) p4）；RoPE「prevents the matrix fusion」（p4）。
- **成立的前提**：不訓練。原文自己指出聯合 head（J-LRD）的重建 FLOPs 是每 head 分解的 n 倍、吸收後的矩陣也大 n 倍（p5），所以提出分組（G-LRD，p6）。**這正是我們的跨層聯合基底在 latent 算時變貴的同一個原因**〔判讀〕。
- **量到的數字**（p8–9）：
  - 設定：Llama-2-7B（MHA）、RTX 4090、權重 4 bit、50% 壓縮（K 75%、V 25%）、基線是 HF 預設實作、**沒用 FlashAttention**。
  - RoPE 注意力 64K：1.89×；V 再量化成 4 bit：2.91×。
  - 非 RoPE：2.20×、6.17×。
  - 4K 時幾乎沒加速。
- **跟 Tiara 的關係**〔判讀〕：V 吸收的數學可以直接用。但它的加速來自「每 head 的 r 小於 d」＋decode，我們兩者都不成立。L5a:300 另記 Palu kernel 遇到 GQA 形狀會報錯。
- **與既有整理的關係**：L5a §0 第 1、2 點、LC §6.1 已有；本卡只補 J-LRD 的成本敘述（p5）與基線條件（p9）。

#### JoLT（arXiv 2607.12550 v4；comment「Under review at ICLR 2027」依 `L5d_stacking.md:187`）
- **讀了什麼**：〔部分〕§4.5（p8）、§5 結尾（p9），<https://arxiv.org/pdf/2607.12550>。
- **線性代數的關鍵一步**：V 在 rank space 裡累加 softmax 權重，跨 block 合併後才「每 head 展開一次」；K 因為 RoPE 和 token-mode 分解不可交換，要逐 tile 完整重建（p8）。
- **原文的話**：把旋轉摺進因子打分「is the primary open problem on the kernel side」（p9）。
- **量到的數字**（p8）：
  - A100 80GB、Mistral-7B 單層、batch 1、context 512–8,192：融合 kernel 是 SDPA（常駐 BF16）的 7.5–65.9 倍時間。
  - 對「先重建再算」：0.7–2.2 倍，只有 8,192 時 < 1。
  - RULER NIAH 8K 端到端：20.7 s、18.2 s，對不壓 1.6 s。
- **與既有整理的關係**：`L5d_stacking.md:34,38` 已記 7.5–65.9 倍與「RoPE 是主要延遲」。本卡補「V 已在 rank space」與「open problem」原句。

#### CommVQ 的 V 部分（見 §6 卡）
- 式 (22)（p6）：Self-Attn＝Softmax(A)·S_V·C_V，先乘 0／1 的碼矩陣 S_V，再乘碼本 C_V。原文：複雜度從 O(d·N_c·N＋d·N) 降到 O(N_c·N＋d·N_c)〔部分〕。

#### DeepSeek-V2（MLA；arXiv 2405.04434 v5）
- **讀了什麼**：〔部分〕§2.1.2–2.1.3（p7–8）。
- **事實**：
  - W_UK 可吸收進 W_Q、W_UV 可吸收進 W_O（p7）。
  - 「RoPE is incompatible with low-rank KV compression」（p8），所以另外拉出一小段帶 RoPE 的解耦 key。
- **前提**：從頭訓練的架構，**不適用**我們的模型。只當「吸收」的出處。

### 2.4 〔判讀〕對 Tiara
- **回來那一輪（n_q＝128–512）**：n_q 遠大於 n_q\*（26–40），V 吸收比重建多 2.5 倍 FLOPs（Llama B24 n_q＝128：只把 V 換成 latent，重建＋注意力從 20.6 變 37.7 TFLOP，§9）。**不要用**。
- **回來之後的 decode**（縮圖常駐、沒挑到的塊一直用縮圖）：
  - latent V 不需要改格式，馬上能用，FLOPs 降 32 倍。
  - 但要注意：latent 的 decode 算術強度變高。Llama B24 的 V 是 53,248 FLOP／312 B ≈ 171 FLOP/B，接近 H200 的 189 FLOP/B 脊點（arith §8）〔算術〕。原本 BF16 decode 是頻寬型（arith §8：2.6 FLOP/B），換成 latent 後**變得有點怕算力爭用**。
- **拿掉怕爭用的 GEMM？** decode：換成一個比較小、但接近脊點的 GEMM。prefill：沒有拿掉，而且變大。

---

## 3. 招 2：讓 K 的基底和 RoPE 可交換——逐頻率、跨 head／層的複數基底

**白話**：RoPE 對每一對維度（同一個頻率）轉一個角度，而且**所有 head、所有層在同一個位置轉的角度都一樣**。所以如果基底只在「同一個頻率」裡面混合不同 head／層，旋轉就可以等到最後才對係數做。這樣 q 可以先乘進基底，K 永遠不用還原。可惜 K 真正的低秩結構是「跨頻率」的，這個限制把它丟掉了。

### 3.1 〔推導〕可交換的形式

- 把窗口裡 32 個 head-層（Llama：W·H＝4·8）的 RoPE 前 K 寫成複數：頻率 j 的那一對維度 → ζ_n^{(j)} ∈ ℂ^{32}，每個 head-層一個複數。
- RoPE 在位置 n：ζ ↦ e^{i·n·θ_j}·ζ。**對整個 ℂ^{32} 是同一個純量**。前提是 θ_j 對每個 head、每一層都相同；Llama-3.1 的 llama3 rope scaling 與 Qwen 的 RoPE 都是如此〔判讀：同一個 config 套全部層〕。
- 取基底 V_j ∈ ℂ^{32×r_j}、係數 c_n^{(j)} ∈ ℂ^{r_j}：ζ^{(j)}≈V_j c^{(j)}。則
  RoPE(V_j c^{(j)})＝e^{inθ_j}·V_j c^{(j)}＝V_j·(e^{inθ_j} c^{(j)})。
  **旋轉移到係數上**，每個係數逐元素乘一個相位，每窗只做一次，4 層、所有 head 共用。
- 分數（q head (ℓ,h,i)，已 RoPE 的 q̃）：
  s＝Σ_j Re( conj(q̃_j)·v_{ℓh,j}ᵀ·c̃_n^{(j)} )＝Σ_j Re( w_j·c̃_n^{(j)} )，其中 w_j＝conj(q̃_j)·v_{ℓh,j} ∈ ℂ^{r_j}（**q 吸收基底**）。
  - 每個 (query, key) 的成本＝2·Σ_j r_j 個實數乘加＝r（實數維度）。
  - 和招 1 一樣：每 q head 是原本的 r/d 倍，**交叉點 n_q\* 同一個公式**（Llama B24 的 K：r/d＝4.25、n_q\*＝41.8）。
- 實數形式：每個頻率塊是一個 64×2r_j 的實矩陣，形如 [[X, −Y], [Y, X]]（複線性）。

### 3.2 〔推導〕這個形式是**唯一**可交換的
- 要讓「先轉再投影」＝「先投影再轉」對所有位置 n 成立，需要基底的欄空間 S 對所有 R_n 不變。
- R_n 的頻譜分解是 Σ_j e^{inθ_j}·P_j（複數形式，P_j 投影到頻率 j 的 ℂ^{32}）。
- 若各 θ_j 互不相同，取足夠多個 n，Vandermonde 論證可以從 {R_n} 的線性組合得到每個 P_j，以及每個頻率塊內的 90° 旋轉 J_j。
- 所以 S 不變 ⇔ S＝⊕_j (S ∩ 頻率 j) 且每塊對 J_j 不變（複線性）。
- **任何混合不同頻率的基底**都不可交換，包括 repo 現在的實數聯合基底（`np_xl_calib` 對 4096 維做 eigh）。
- 兩個頻率碰巧相等時，可以合併成一塊（這是 TransMLA FreqFold 的「近似」版本，它合併的是**相鄰但不相等**的頻率，原文 App. C.4 自己說會偏離精確 RoPE〔二手，PseudoHunt PRIOR_ART.md 引述〕）。
- 〔未查證〕：Llama-3.1 的 llama3 scaling 後 64 個 θ_j 是否仍兩兩不同（推測單調所以不同）。〔複核 2026-10-27：已查。transformers `modeling_rope_utils.py` `_compute_llama3_parameters`（main 9ed172cd91，L655–660）＋HF config（rope_theta 500000、factor 8、low 1、high 4、orig 8192）代入：64 個 θ_j 嚴格遞減、相鄰比 ≥1.2276、全在 (3.1e-7, 1] rad → 兩兩不同，也沒有 θ_j≡−θ_k 或 ≡0、π（mod 2π），§3.2 的前提成立〔算術〕。Qwen2.5-1M 的 rope_scaling 是 null〕

### 3.3 卡片

#### PseudoHunt/rope-kv-compression（GitHub，建立 2026-09-23，commit 9d7200c；非同儕審查、作者匿名帳號）
- **讀了什麼**：〔文件〕README、`docs/FINDINGS.md`、`docs/PRIOR_ART.md`、`docs/FINDINGS_KERNEL.md`（用 `gh api` 取回原文），查證 2026-10-27。**沒跑它的程式碼**（規則 7），數字照文件抄。
- **一句話**：兩個事先登記閘門的研究，在不訓練的前提下壓 K 並維持精確或近精確的 RoPE。
- **研究 1（逐頻率複數跨 head PCA）**：
  - 方法：同一層內、跨 32 個 head、每個頻率一個複數 PCA（Hermitian Σ_i 的特徵向量）；跨頻率全域分配秩；快取已旋轉的 latent，q 吸收，不重建。
  - 和我們 §3.1 的推導**完全同一個家族**，只是在單層、MHA，不是跨層窗口。
  - 能量（held-out，層平均，FINDINGS 表）：

| K 預算 ρ | (a) RoPE 前 PCA | (d) RoPE 後 PCA | (c) 逐頻率複數 | (b) 逐頻率實數（RoRoPE 基底） |
|:--|--:|--:|--:|--:|
| 50% | 99.5 | 96.2 | 94.9 | 94.2 |
| 25% | 97.7 | 91.8 | 88.9 | 87.4 |
| 12.5% | 95.3 | 88.1 | 83.8 | 81.5 |

  - 輸出誤差（單層替換，relL2，c／a）：50% 2.92×、25% **2.69×**、12.5% 2.38×；32 層裡 0 層在 1.5× 以內 → 照登記停止。
  - 原文的拆解（ρ＝25%，層中位數）：
    - 跨頻率結構約 8.0 點（任何 RoPE 精確的線性映射都拿不到）。
    - 頻率內的複線性限制約 1.4 點。
    - 複數比實數只多約 1.5 點。
  - 秩的分配：大部分給慢頻率（i ≥ 46）。這 18 對在中後層約佔 57% 的 K 能量。
  - 原文教訓：要比「損失能量的比值」，不是保留能量的百分點（97.7 vs 88.9 是 4.8 倍的損失能量）。
- **研究 2（把 RoPE 的位置核分解成低秩）**：
  - e^{iΔθ_i}≈ΦΓ（秩 p），讓**不受限**的低秩 K 基底也能吸收進 q。
  - Llama-2-7B-chat、ρ＝25%、2K–4K：p＝16 時輸出誤差是「重建再 RoPE」的 1.10 倍；WikiText-2 困惑度 p＝32 時 +0.77%、p＝16 時 +2.49%（只在 ctx 2048）。
  - 第 0、1 層最難。
  - Φ 表的大小隨長度線性成長（L＝4096、p＝16 時每 (層, head) 0.5 MB fp32）。
  - 「No latency is reported anywhere」（README）。
- **跟 Tiara 的關係**〔判讀〕：
  - 研究 1 直接預告我們的跨層逐頻率版大概也會輸。K1 的「每 head 36–40 維就有 95% 能量」是 head 內跨頻率的冗餘，正是被丟掉的那塊。
  - 研究 2 在我們這裡更不划算：每個 (q, key) 是 p·r＝16·544＝8,704 個乘加；重建攤到每個 (q 列, key) 是 r·d／(g·n_q)＋d，n_q＝128 時只有 264，n_q＝1 時 17,536〔算術〕。所以它只在 decode 省一半。而且 128K 時 Φ 表是 4K 時的 32 倍。
- **與既有整理的關係**：repo 沒有任何檔提過（`grep rope-kv-compression docs` 0 筆）。

#### TransMLA：MLA Is All You Need（NeurIPS 2025 spotlight，依 PseudoHunt PRIOR_ART〔二手〕；arXiv 2502.07864 v5）
- **讀了什麼**：〔部分〕§4.2（p2、p6 圖 2、p7 式 (17)–(19)）、§5.2 圖 3b（p9）。LC §6.1 已有卡。
- **關鍵一步**：
  - 同頻率、跨 head 的正交 U_l，實部和虛部用同一個 U_l，內積不變（式 (19) p7）。
  - 每個頻率只留一個主成分在「第一個 head」帶 RoPE，**其餘的 RoPE 被拿掉**（變成 NoPE，和 V 一起聯合 PCA；p7）。
  - 所以 TransMLA 是「近似＋微調」，不是「截斷＋精確 RoPE」。
- **數字**：LLaMA 3 8B 逐步移除 RoPE 成分，90% 移除時 RoRoPE＋4D-FreqFold 的 log-perplexity 約 2、MHA2MLA 約 6（p9 圖 3b）〔部分〕。
- **跟 Tiara**：證明「同頻率、跨 head」可交換。但它只在單層內，用實數 U（PseudoHunt 讀了程式碼確認是實數 eigh〔二手〕），而且要微調。

#### CommVQ：Commutative Vector Quantization for KV Cache Compression（ICML 2025，PMLR v267；arXiv 2506.18879 v1）
- **讀了什麼**：〔部分〕§3.2–4.2（p3–6）、§5.6 表 5（p9）。
- **關鍵一步**：
  - 2×2 的 [[x, y], [−y, x]] 和 RoPE 的 2×2 旋轉可交換（Property 1，式 (7)(8) p3）。這就是「碼字是複數純量」。
  - 每個 2 維子空間 j 各有一組這種碼字（式 (17)(18) p5）。
  - 分數 α_i＝Σ_{j,l}(q^j R_t^j) C_K^{jlᵀ} R_i^{jᵀ}[s_i^j＝l]ᵀ，其中 (q^j R_t^j) C_K^{jlᵀ} 可以跨 key 重用（式 (19) p5）。
- **成立的前提**：
  - 模型不訓練。但編碼器（線性層＋Gumbel-softmax）和碼本要用 FineWeb-Edu 訓練（p4、p6）。
  - 1-bit 設定 N_c'＝64、R＝11、g＝64（p6）。
- **量到的數字**：
  - 複雜度從「N_c 倍的普通注意力」降到「約 (R＋1)/2 倍」（p6）〔複核 2026-10-27：原寫「約 R＋1 倍」，pdftotext 把分數印成「R+12」，HTML 的 TeX 是 \frac{R+1}{2}，改為 (R＋1)/2〕。
  - 表 5（p9，每層每 token，單位 ms；樸素 → 最佳化）：8K 2.4 → 0.4、32K 9.2 → 1.1、128K 36.6 → 3.8（6.0×／8.4×／9.6×）。
  - **表 5 沒和 FP16 注意力比**。
- **跟 Tiara**〔判讀〕：證明「可交換」可以讓 RoPE 留在壓縮域。但它是 VQ（逐 head、逐頻率對），不是跨 head／層的低秩；成本仍是普通注意力的約 6 倍〔複核 2026-10-27：原寫「約 12 倍」，依 (R＋1)/2、R＝11 改為約 6 倍〕。
- **與既有整理**：`L8a:343`（〔摘要〕級）、`03_bandwidth_compute.md:170`。本卡補式號、成本與表 5。

#### RAP：RoPE-Aligned Pruning for KV-Cache Compression（arXiv 2602.02599 v4，2026-07-30；venue 未查證）
- **讀了什麼**：〔摘要〕＋p1 引言。
- **事實**：
  - 以 RoPE 對為單位剪 W_K，剪完可以吸收進 W_q，decode 不用逐步重建。
  - 不微調只能剪約 10%（ρ＝0.9）；30% 要 LoRA 知識蒸餾（p1）。
- **跟 Tiara**：是 §3.1 家族的「0／1 選擇、不混 head」特例。

#### EliteKV（arXiv 2503.01586 v1；venue 未查證）
- **讀了什麼**：〔部分〕摘要（p1）、§4.1–4.2（p5）。
- **事實**：
  - RoPElite 逐 head 挑保留 RoPE 的頻率塊，其餘維度「恢復線性」（去掉 RoPE），再對 NoPE-K 與 V 做聯合低秩（J-LRD）。
  - 要用不到 0.6% 的預訓練 token 做 uptraining；實驗是 LLaMA2-7B／13B（MHA）（p5）。
- **跟 Tiara**：要訓練，不適用。

#### SALS（NeurIPS 2025 poster，依搜尋結果〔二手〕；arXiv 2510.24273 v1）
- 引理 1（p6）：多 head 聯合投影能抓到的能量 ≥ 每 head 分塊投影。
- 同一個論證套到我們這裡〔推導〕：逐頻率複線性家族是不受限實數基底的子集，所以同秩下**能量只會更少**。§3.1 的限制一定有代價，只是大小要量。

### 3.4 本次沒找到的
- **跨層窗口的逐頻率 RoPE 可交換基底**（不論實數或複數、不論快取還是權重）：本次沒找到。
  - 查詢字串：`RoPE commuting low-rank key cache per frequency complex basis across heads layers attention in latent space without reconstruction`、`"RoPE" KV cache compression complex-valued PCA per rotary frequency shared across layers query absorption 2026`。
  - PseudoHunt 的 PRIOR_ART.md 自己列了 17 個查詢、22 篇全文，也沒找到單層的複數版。
- 「慢頻率近似可交換、快頻率精確」的混合版：PseudoHunt 列為建議、沒跑。本次沒找到有人做。

### 3.5 〔判讀〕repo 的 K1 怎麼讀，以及能不能便宜地驗
- **K1**：RoPE 前每 head 95% 能量在 36–40 維（`results_analysis.md:87`）。這是「同一 head 內、128 維之間」的冗餘，主要跨頻率。逐頻率家族在每個 head 每個頻率只有 1 個複數，**一個 head 內完全不能壓**；所有壓縮都要靠「同一頻率、不同 head／層之間」的相關。
- **K1b**：相鄰層同一 token 的 cosine ≈ 0，原因是每層有自己的 W_K、座標不同（`results_analysis.md:93`）。同頻率、跨層的分量也來自不同的 W_K 列〔判讀〕，沒有理由特別相關。所以把窗口從「單層 8 head」擴成「4 層 32 head-層」，在逐頻率家族裡大概幫助有限。這和 xKV 共享的是「token 子空間」不同：那是 A 的那一側，逐頻率限制管的是基底那一側。
- **怎麼驗**（不需要 Q、不需要生成）：
  - 用既有 K dump，每窗、每頻率 j 做 Z^{(j)} ∈ ℂ^{T×32} 的 Hermitian PCA，全域分配秩到 r_K＝544 實數維。
  - 和現在的實數聯合基底比**損失能量比值**。
  - PseudoHunt 的換算：relL2 比 ≈ √(損失能量比)，1.5 倍門檻對應 ≤2.25。
  - 這是 CPU 上的特徵分解，但規則上仍交給 H200 session，並先登記。

### 3.6 拿掉怕爭用的 GEMM？
和招 1 相同：decode 拿掉，prefill 沒有。另外要**換縮圖的 K 基底、重新校準**，品質預期變差。

---

## 4. 招 3：scale／zero 吸收與 softmax 平移不變

**白話**：量化值＝scale×整數＋zero。可以把 scale 先乘進 q，把 zero 變成每個 token 組一個加項，就不必先把整數還原。如果某個加項對**所有** key 都一樣，softmax 會把它整個消掉。

### 4.1 〔推導〕
- **K 逐 channel（KIVI；group＝γ 個 token）**：k̂_{n,c}＝s_{c,γ}Q_{n,c}＋z_{c,γ}。
  q·k̂_n＝Σ_c (q_c s_{c,γ})Q_{n,c}＋Σ_c q_c z_{c,γ}。
  - 第二項在同一個 token 組內是常數，組與組之間不同 → **不會**被 softmax 消掉，每組要加一個 bias（每 q 列每組 d 次乘加）。
  - 第一項：q⊙s_γ 每組算一次；要用 INT8 MMA 時 q 也要量化（HACK、QServe、SageAttention 都是這樣），會多一份誤差。
- **V 逐 token**：v̂_n＝s_n Q_n＋z_n → o＝Σ_n (p_n s_n) Q_n＋Σ_n p_n z_n。
- **softmax 平移不變**：σ(q(K − 1μᵀ)ᵀ)＝σ(qKᵀ − q·μ)＝σ(qKᵀ)（SageAttention 式 (6) p5 的推導）。μ 要對**所有 key 相同**。
- **RoPE 前的平均在 RoPE 後就不同了**〔推導〕：q̃·R_n μ＝Re Σ_j conj(q̃_j) μ_j e^{inθ_j}，隨 n 變。只有 θ_j·n ≈ 0 的慢頻率近似不變。
  - 所以在 RoPE 前去平均的 K（Palu、KVTC 一類）不能靠 softmax 白白丟掉平均。
  - SageAttention 是在 RoPE 之後、對實際的 K 去平均（p2 說它的 kernel 融合了 RoPE 與量化〔部分〕），所以精確。
- **縮圖係數的 KIVI 式 scale**（每個主成分、每 32 token 一組，`m16_xstack.py` 檔頭〔程式碼〕）：
  - 要把 scale 摺進基底 B，每組要做一次 diag(s_γ)·B（r×W·H·d）＝每 32 token 多 1/32 的重建量。不如在暫存器裡直接反量化〔算術〕。

### 4.2 卡片

#### SageAttention（ICLR 2025；arXiv 2410.02367 v9）／SageAttention2（ICML 2025；arXiv 2411.10958 v7）／SageAttention3（arXiv 2505.11594 v3）
- **讀了什麼**：〔部分〕Sage1 p1–2、p5 式 (6)；Sage2 摘要（p1）、p2、p4、p6；Sage3 摘要（p1）。
- **關鍵一步**：
  - Sage1：K 減去 token 平均（式 (6) p5），INT8 的 QKᵀ。
  - Sage2：Q 也平滑，補回項用一個 GEMV 算 ΔS（p4、p6 的演算法）；QKᵀ 用 INT4，PV 用 FP8。
  - Sage3：Blackwell 的 FP4 tensor core。
- **數字**：
  - Sage1：OPS 是 FA2 的約 2.1 倍、xformers 的 2.7 倍；RTX 4090 headdim 64 時 340 TOPS（p1–2）。
  - Sage2：約 3×／4.5×；在 Hopper 上和 FA3(fp8) 一樣快（p1）。原文說 Hopper 沒有原生 INT4 tensor core（p2）。
  - Sage3：RTX 5090 上 1038 TOPS，是最快 FlashAttention 的 5 倍（p1）。
- **跟 Tiara**〔判讀〕：這些都是「從 BF16 即時量化 Q、K、V 再算」的注意力 kernel，**不是**讀取預先存好的壓縮快取。它對我們的意義只有兩點：softmax 平移不變的精確用法；H200（Hopper）沒有 INT4 MMA。

#### QServe（MLSys 2025；arXiv 2405.04532 v3）
- **讀了什麼**：〔部分〕摘要（p1）、§3.2（p4）、SmoothAttention（p6）、§5.3 與表 1（p10）。
- **關鍵一步**：
  - SmoothAttention：Z＝(QΛ)·(KΛ⁻¹)ᵀ（式 (7) p6）。
  - 為了和 RoPE 可交換，硬性規定 λ_i＝λ_{i＋D/2}，也就是同一對 RoPE 維度用同一個縮放（式 (9) p6）。這是 §3.2「可交換必須在每個 RoPE 對裡是純量」的最簡單特例〔推導〕。
- **數字**（表 1 p10，A100，decode 注意力）：
  - 8-bit KV：128 token 0.09 ms … 1536 token 0.62 ms。
  - 樸素 KV4：0.87–0.90×（**更慢**）。
  - QServe KV4：1.29–1.51×。
  - 原文的解釋（p10）：反量化一個 INT4 要 5 個 ALU 運算；A100 的 FP32 CUDA core 脊點只有 9.8 Ops/Byte，所以融合的 KV4 注意力會變成算力型。
  - p4：A100、H100 上 FP32 CUDA core 的峰值只有 INT4 tensor core 的 2%。
- **跟 Tiara**〔判讀〕：解壓融進注意力不是自動免費的，會吃 CUDA core。這和 repo G-M11 想量的「不吃 tensor core 的解碼會不會比較不怕爭用」是同一個問題。

#### HACK（arXiv 2502.03589 v1；venue 未查證）
- **讀了什麼**：〔部分〕摘要（p1）、§2（p2）、§5.2 式 (4)（p6）。
- **關鍵一步**：
  - 式 (4)（p6）：Σ_z a_{iz}b_{zj} ≈ s_{ai}s_{bj}Σ a′b′＋m_{bj}s_{ai}Σ a′＋m_{ai}s_{bj}Σ b′＋Z·m_{ai}m_{bj}。
  - 第一項用 INT8 算；其餘 9MN＋MZ＋NZ 次運算（原文的計數）。
  - KV 是 2 bit 隨機量化。
- **數字**：反量化最高佔 JCT 37.9%（p2）；JCT 比分離式基線少最多 70.9%、比 SOTA KV 量化少最多 52.3%（p1）。
- **與既有整理**：`L7b_recheck_contention_format.md:71`（〔摘要〕＋L2 B6，17.2–37.9%）。一致。

#### TurboAttention（arXiv 2412.08585 v3；venue 未查證）
- **讀了什麼**：〔摘要〕p1；p4 的一句。
- **事實**：
  - FlashQ（逐 head 量化、整數執行 QKᵀ 與 PV）＋SAS（指數不必反量化到 FP32）。
  - 注意力 1.2–1.8×、KV 小 4.4 倍以上、最大吞吐 2.37×（p1）。
  - p4：「FP32 CUDA cores deliver only ∼3%」of FP16 tensor core 的效能。

#### BitDecoding（arXiv 2503.18773 v3；venue 未查證）
- **讀了什麼**：〔摘要〕p1。
- **事實**：
  - tensor core 做矩陣乘、CUDA core 做反量化，軟體流水線。
  - 平均是 FP16 FlashDecoding-v2 的 7.5 倍；Blackwell NVFP4 最多 8.6 倍。
  - LLaMA-3.1-8B 128K 單 batch decode 延遲降 3 倍（p1）。
  - **只有 decode**。
- **與既有整理**：`03_bandwidth_compute.md:101`、`07_lit_quickcheck.md:20`。

#### KIVI（ICML 2024；arXiv 2402.02750 v2）
- **讀了什麼**：〔部分〕p5–6。
- **事實**：反量化和矩陣乘融合（「Q_Matmul」，p5）；2.35–3.47× 吞吐來自更大的 batch（p2、p8），不是 kernel 本身更快。

### 4.3 〔判讀〕對 Tiara
- 精修塊是 INT4-KIVI，F3 量到反量化一個 64 MiB chunk 0.027–0.031 ms（`F3_dequant_fp8.md:69`）。ρ＝1/8 的 Llama 128K 約 31 個 chunk → 約 1 ms。〔複核 2026-10-27：原未寫，F3 是 MI300X、INT4 是對稱 per-(token, head) 不是 KIVI；H200 上 int4_kivi 的融合 kernel（k1、k3:8:2）是 0.049–0.073 ms（`kernels_llama.csv`）→ 31 塊約 1.5–2.3 ms，量級不變〕招 3 最多省這 1 ms。
- **完全不碰低秩重建的 GEMM。**

---

## 5. 招 4：不需反轉的正交轉換

**白話**：如果 K 和 q 乘同一個正交矩陣，內積不變，所以「存轉過的 K、用轉過的 q 去算」不需要再轉回來。我們的縮圖基底本來就是正交的，所以沒有「反轉」可以省；能不能省重建，又回到招 1、招 2。

- **數學**：H 正交 ⇒ (qH)(kH)ᵀ＝qkᵀ。Loki 引理 4.1（p5）：任何正交 P 都成立〔部分〕。
  - 截斷後（只留前 d′ 維）才是近似（Loki 引理 4.2 p5）。
- **條件**：同一個 head、同一個座標系。
  - 在 RoPE 之後做：精確，但 post-RoPE 的低秩差。Loki 量到 Llama3-8B 用 post-rotary PCA 時困惑度從約 5 跳到 10 以上（p7）；`L5a:26` 已記。
  - 在 RoPE 之前做：要和 RoPE 可交換 → 招 2。
- **我們的縮圖**：
  - U_w 是 G＝ΣZᵀZ 的特徵向量（`np_xl_calib`〔程式碼〕）→ 正交欄。
  - 係數 A＝Z·U、重建 Z＝A·Uᵀ，本來就沒有反矩陣。「不需反轉」對我們沒有新東西。
- **精修塊**：
  - QuaRot 式「post-RoPE 的 H·k、H·q」數學上精確，但 INT4-KIVI 在 32K–128K 已經不掉分（`L5a:50`），LC §5.3 判沒有空間。
  - 另外 QServe 指出：注意力運算子裡的 Hadamard 是算力密集的，讓 4-bit KV 很難真的比 KV8 快（p10）〔部分〕。
- **拿掉怕爭用的 GEMM？** 否。

---

## 6. 招 5：PQ／VQ 的非對稱距離（查表），完全不解壓

**白話**：把 K 切成很多小段，每段用一本小字典的編號表示。算分數時，先把 q 和字典每一項的內積算好放成一張表，之後每個 key 只要「查表＋加總」。不需要矩陣乘、不需要還原 K。但那張表要放得進晶片上的小記憶體，問題一長（prefill）就放不下。

### 6.1 〔推導〕成本與條件
設 m 個子空間、每本 K 項、子空間維度 d_m＝d/m。
- **建表**：每個 (q 列, q head) 要 K·d 次乘加（和 n 無關）。
- **打分**：每個 (q 列, key) 要 m 次 gather＋加；普通注意力是 d 次乘加 → 運算數少 d_m 倍（G-M11：d_m＝4）。
- **但 gather 走 CUDA core／shared memory**。TurboAttention（p4）說 FP32 CUDA core 只有 FP16 tensor core 的約 3%，QServe（p4）說 2%（對 INT4 tensor core，A100／H100）。
  - 若 gather＋加的吞吐和 FP32 CUDA core 同級（〔判讀〕，沒查到 H200 的數字），運算數少 4 倍抵不過吞吐差約 30 倍。prefill（算力型）時查表會**輸** tensor core 約 8 倍〔算術：(1/0.03)/4〕。
  - decode（頻寬型）時，查表靠位元組贏。
- **SRAM 條件**：G-M11 格式 m＝32、K＝256 → 每 (q 列, head) 8,192 項（FP16 16 KB；MILLION 用 int8 表〔部分，p4〕是 8 KB）。
  - prefill n_q＝128、g＝4 → 每個 KV head 512 列 → 4–8 MB，不可能放進 SRAM（H200 每 SM 的 shared memory 大小本次〔未查證〕，Hopper 公開值是數百 KB 等級）。
  - decode 每個 KV head 只有 g＝4–7 列 → 放得下。
- **V 這一側**：o＝Σ_s Σ_c C_s[c]·(Σ_{n: code＝c} p_n) → 先把 p 依編號做 histogram（每 q 列 n·m 次 scatter-add），再一個小矩陣乘。
- **有 RoPE 時**：碼本要和 RoPE 可交換（CommVQ）。這時每個 key 每個子空間還要乘一次相位，成本變成約 (R＋1)/2 倍的普通注意力（CommVQ p6）〔複核 2026-10-27：原寫「(R＋1) 倍」，改為 (R＋1)/2〕。

### 6.2 卡片

#### PQCache（SIGMOD 2025，依 repo `LE_nla_conditions.md:192`；arXiv 2407.12820 v2）
- **讀了什麼**：〔部分〕§3（p9）。
- **事實**：
  - decode 時先 (h, m, 1, d_m)×(h, m, d_m, 2^b) 算 q 對中心點的內積，再用 PQ 碼 gather＋reduce。
  - PQ 部分的複雜度 O(2^b·d²/(h·m)＋h_kv·m·s)（p9）。
  - 用來**挑 top-k**，KV 本體在 CPU；K-means 在 CPU 和 GPU 計算重疊（p9）。
- **與既有整理**：LE §13、`LE:192` 已有卡；本卡只補 p9 的 ADC 公式。

#### MILLION（arXiv 2504.03661 v2；venue 未查證）
- **讀了什麼**：〔部分〕p2、p4、p6。
- **事實**：
  - 把 q×碼本當成查表載入 shared memory，用 int8 做 16 次 shared memory 查表；V 的部分仍是反量化（p4）。
  - A40、32K：注意力 2.01×、端到端 2.09×（p6）。
  - 引 QServe：4-bit 整數反量化佔推論延遲 20–90%（p2）。

#### VQ-LLM（HPCA 2025；arXiv 2503.02236 v2）
- **讀了什麼**：〔部分〕摘要（p1）、§III（p4–5）。
- **事實**：
  - 主要的低效來自碼本存取。256 項 vs 32 個 shared memory bank 會 bank 衝突（p5）。
  - 提出「codebook cache」：依存取頻率放在 global／shared／register（p1）。
  - 比既有開源 VQ 實作延遲少 64.36–99.1%；和 AWQ、QoQ 同位元時延遲接近或更好（p1）。
  - 分析用 RTX 4090（p4）。

#### CommVQ（見 §3.3）

#### repo 的 G-M11（`runs/G_M11_vq.md`）
- PQ：d_m＝4、每（層, K/V, 子空間）256 項、head 共用、2 bit；解碼＝PyTorch gather（不用 tensor core）。
- 判定：兩次 run 都 **NOT_DETERMINED**（閒時 dq_int4 雙峰，`RUNLOG_H200_S3.md:395`）。
- B0 另報：GB0-STATIC 不成立**只因為** dq_vq（`B0_interference.md` 結果節）。本檔不解讀這一點，也沒看 VQ 對 INT4 的比（盲測）。
- 注意 G-M11 量的是「解碼成 BF16」，不是查表打分（ADC）。

### 6.3 〔判讀〕對 Tiara
- 要用就得把縮圖從「低秩＋純量量化」換成 PQ，而且 K 要 RoPE 可交換的碼本（CommVQ 式，要訓練編碼器）。
- 只對 decode 有利。回來那一輪是 prefill，查表放不下 SRAM，運算又在 CUDA core。
- **拿掉怕爭用的 GEMM？** 拿掉 tensor core 的 GEMM，換成 CUDA core＋shared memory 的負載。它怕不怕算力型背景，正是 G-M11 要回答、目前 NOT_DETERMINED 的問題。

---

## 7. 招 6：重建融進注意力的 tile（BF16 不寫回 HBM）

**白話**：與其先把整份 KV 還原寫回 HBM 再讀回來算注意力，不如在晶片上的小記憶體裡邊還原邊算。這省的是「搬到 HBM 再搬回來」，不是計算。在 prefill 時，同一塊 KV 會被好幾個 query tile 各自還原一次，計算反而變多。

### 7.1 〔推導〕
- **未融合**：重建一次（2·n·r·H·d／層）＋寫 BF16＋讀 BF16。
  - Llama 128K：BF16 KV 17.18 GB，寫＋讀 34.4 GB ÷ 4,173.9 GB/s ≈ **8.2 ms**〔算術；HBM 是 repo 實測的裝置內複製，arith §8〕。
- **融合**：每個 CTA 負責 B_M 個 query 列（GQA 時通常把 g 個 head 的列打包），對每個 KV tile 在 SRAM 裡重建。
  - 同一個 KV tile 會被 ⌈n_q·g／B_M⌉ 個 CTA 各重建一次。
  - B_M＝128 時：Llama n_q＝128 → 4 倍、Qwen-7B → 7 倍、Qwen-14B → 5 倍；n_q＝512 時再乘 4。
  - Llama 128K B24、n_q＝128：融合版重建約 4×11.8＝47 TFLOP，未融合版 11.8 TFLOP＋8.2 ms HBM。
  - **prefill 時未融合比較好**；decode（n_q·g ≤ B_M）時融合好〔算術〕。
  - 除非一個 CTA 吃下一個 KV head 的全部 512 列（暫存器／SRAM 放不下，〔判讀〕）。

### 7.2 卡片（已有的只引用）
- **ShadowKV**：decode 時只重建被選中的 chunk，A100 上每個 block 1.25–1.51 ms，和取 V 的 1.66–1.84 ms 重疊（`L5a_lowrank.md:128`，ShadowKV 表 13）。
- **Palu**：Triton kernel 把「重建 K、RoPE、乘 q」融成一次呼叫（p7）。數字見 §2.3。
- **xKV**（ICML 2026，依 PDF 頁首「Proceedings of the 43rd ICML」；arXiv 2503.18893 v2）〔部分，p4、p7、p19〕：
  - 全部重建在 A100 上「compute-bound」，對 FA2 的 kernel 加速一律 <1.0×，122K 時約 0.4×（p7）。
  - 選擇性重建（SR，只重建選中的 token，式 (3) p4）：注意力最多 3.5×、端到端吞吐 4.23×（122K）（p7）。
  - 跨層用的秩比單層大，所以每層重建時內維更大，計算略增（p7、附錄 D.6）。
  - 全文 grep「absorb」0 筆：**xKV 沒有做 V 吸收**。
- **JoLT**：見 §2.3（融合 kernel 是 SDPA 的 7.5–65.9 倍時間）。
- **SALS**：融合的 recover kernel（p5 圖 3）。表 6（p10）：bs 8、4K、稀疏 1/8 時 0.439–0.530 ms，FA2 是 1.630 ms；GPU 只寫「ampere architecture」（p7）。

### 7.3 〔判讀〕
- 融合是「省 HBM、decode 用」的工程，**不拿掉怕爭用的 GEMM**。
- 文獻裡贏的版本全都靠稀疏（只重建少數 token），不是靠代數。

---

## 8. 招 7：直接在壓縮域選塊

**白話**：挑塊時不重建，直接用 latent 打分。但 RoPE 前的 latent 少了位置旋轉，分數不精確；最前面幾層錯得最多。

- **數學**：精確分數是 q^T R_{n−m} k_n。RoPE 前 latent 打分用的是 q_preᵀ U Uᵀ k_pre，少了 R_{n−m}。
  - 只有 θ_j·(n−m) ≈ 0 的慢頻率近似成立。
  - 用 §3 的逐頻率基底就能精確；用 PseudoHunt 研究 2 的核分解可以近似。
- **已知的量化**：
  - SALS（p5 圖 2）：RoPE 前 latent 的 top-N_c 抓到的注意力質量（overlap score）在 LLaMA2-7B-chat、LLaMA3-8B、Mistral-7B 的第 2–29 層多半 >90%，第 0、1 層 <50%〔部分；N_c 的值圖上沒讀到〕。
  - RULER 128K、Llama-3.1-8B-Instruct、25% 壓縮、稀疏 8×：平均 80.81，不壓 81.60（p9–10）〔部分〕。
  - SAKI、LOCKS、ShadowKV 表 1 的 Loki：`L7d_recheck_lowrank_index.md:38–45` 已整理（共用基底的召回只有 0.58–0.79）。不重抄。
- **A0 的做法**：BASE-ATTN／ONE-PASS 用的是「重建後的粗底 K（精確 RoPE）」打分（`A0_progressive_restore.md` §1.4）。改成 latent 打分是另一層近似，要另外登記。
- **拿掉怕爭用的 GEMM？** 只拿掉「打分用的那份重建」。沒挑到的塊在最後的注意力裡仍要重建或走 latent（招 1／2）。

---

## 9. 成本公式與交叉點〔算術〕

**常數**：模型層數／KV head／q head 取自 `code/s3_mech_arith.py:37–43`〔程式碼〕。H200 matmul 788.1 TFLOPS、HBM 4,173.9 GB/s 取自 repo 實測（arith §8、`LF_hw_isa.md:130`）。秩取自 `A0_progressive_restore.md` §1.3：B16 k20/v30、B24 k17/v26、B32 k13/v20，r＝rhe·W·H、W＝4。**全部是理想 FLOPs，不含效率、不含 RoPE 與反量化的 ALU**。

### 9.1 公式（每層）
- 重建（K＋V）：C_rec＝2·n·(r_K＋r_V)·H·d。和 n_q 無關，所有 q head 共用。
- 普通注意力：C_att＝4·n_q·n·H·g·d（QKᵀ 與 PV 各一半）。
- latent 注意力：C_lat＝2·n_q·n·(r_K＋r_V)·H·g。K 那一半只在招 2 的基底下成立。
- 交叉點（每一半各自）：**n_q\*＝r·d／(g·(r − d))**。
- 融合 tile 的重建倍數：⌈n_q·g／B_M⌉（§7.1）。

### 9.2 交叉點 n_q\*（低於它 latent 便宜）

| 模型 | g | 設定 | r_K | r_V | r_K/d | r_V/d | n_q\*（K） | n_q\*（V） |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|
| Llama-3.1-8B | 4 | B16 | 640 | 960 | 5.00 | 7.50 | 40.0 | 36.9 |
| Llama-3.1-8B | 4 | B24 | 544 | 832 | 4.25 | 6.50 | 41.8 | 37.8 |
| Llama-3.1-8B | 4 | B32 | 416 | 640 | 3.25 | 5.00 | 46.2 | 40.0 |
| Qwen2.5-7B | 7 | B16 | 320 | 480 | 2.50 | 3.75 | 30.5 | 24.9 |
| Qwen2.5-7B | 7 | B24 | 272 | 416 | 2.12 | 3.25 | 34.5 | 26.4 |
| Qwen2.5-7B | 7 | B32 | 208 | 320 | 1.62 | 2.50 | 47.5 | 30.5 |
| Qwen2.5-14B | 5 | B16 | 640 | 960 | 5.00 | 7.50 | 32.0 | 29.5 |
| Qwen2.5-14B | 5 | B24 | 544 | 832 | 4.25 | 6.50 | 33.5 | 30.3 |
| Qwen2.5-14B | 5 | B32 | 416 | 640 | 3.25 | 5.00 | 37.0 | 32.0 |

讀法：問題長度 128–512 全部遠大於 n_q\* → **回來那一輪用「先重建再算」**。decode（n_q＝1）全部小於 n_q\* → decode 用 latent。

### 9.3 一次完整的 TFLOP 數（全部層、整段 context）

| 模型・長度 | 設定 | 重建（TFLOP） | 重建時間：788 T／50% 效率（ms） | n_q＝1：重建＋注意力 vs 全 latent | n_q＝128：重建＋注意力 vs 全 latent | n_q＝512：重建＋注意力 vs 全 latent |
|:--|:--|--:|:--|:--|:--|:--|
| Llama-3.1-8B・128K | B16 | 13.74 | 17.4／34.9 | 13.81 vs 0.43 | 22.54 vs 54.98 | 48.93 vs 219.9 |
| Llama-3.1-8B・128K | **B24** | **11.82** | **15.0／30.0** | 11.89 vs 0.37 | **20.62 vs 47.28** | 47.00 vs 189.1 |
| Llama-3.1-8B・128K | B32 | 9.07 | 11.5／23.0 | 9.14 vs 0.28 | 17.87 vs 36.28 | 44.26 vs 145.1 |
| Qwen2.5-7B・256K | B24 | 5.17 | 6.6／13.1 | 5.28 vs 0.28 | 18.64 vs 36.20 | 59.05 vs 144.8 |
| Qwen2.5-14B・128K | B24 | 17.73 | 22.5／45.0 | 17.86 vs 0.69 | 34.22 vs 88.65 | 83.70 vs 354.6 |
| Qwen2.5-14B・256K | B24 | 35.46 | 45.0／90.0 | 35.72 vs 1.39 | 68.44 vs 177.3 | 167.4 vs 709.2 |

- **只把 V 換成 latent、K 照樣重建**（不用改格式），Llama B24：n_q＝1 4.93、n_q＝128 37.66、n_q＝512 136.6 TFLOP。對照「全部重建」的 11.89／20.62／47.00 → 只有 decode 划算。
- **和 arith §9 對照**：Llama 128K CPU 層 n_q＝128 的「逐層流水」TTFT 是 0.028 s，**沒算重建**。B24 重建在 50% 效率下就要 0.030 s，GPU 忙（×2.8，`results_analysis.md:325`）時 0.084 s。
  - Qwen-14B 256K：§9 是 0.095 s，重建 0.090 s（閒、50%）。
  - → 縮圖路徑的 TTFT 估計大約要加倍。這一項應該進 C0 的成本模型〔判讀〕。
- **B32 對 B24**：重建少 23%（9.07 vs 11.82）。是縮小這一項最直接的槓桿，品質代價看 A0／A0-C。

### 9.4 decode 時的算術強度（Llama B24，每層每 context token）

| 路徑 | FLOP | 讀的位元組 | FLOP/B | 對 H200 脊點 189 |
|:--|--:|--:|--:|:--|
| BF16 常駐（只算 PV） | 2·d·H·g＝8,192 | 2·H·d＝2,048 | 4 | 頻寬型 |
| latent V（招 1） | 2·r_V·H·g＝53,248 | 832×0.375 B＝312 | 171 | 接近脊點 |
| latent K（招 2） | 2·r_K·H·g＝34,816 | 544×0.625 B＝340 | 102 | 偏頻寬型 |
| 每步都重建 V | 2·r_V·H·d＝1,703,936 | 312 | 5,461 | 算力型 |

- 位元組假設〔算術〕：K 係數 int4＋每 32 token 一組的 BF16 scale／zero（0.5＋0.125 B），V 係數 int2（0.25＋0.125 B）。
- 驗算：(544×0.625＋832×0.375)×8 窗／(4,096 B × 32 層)＝0.0398，等於 A0 表 B24 的 f＝0.0398 ✓。
- latent 每一步 Llama 128K：0.37 TFLOP（788 T 下 0.47 ms），係數每層讀一次共 2.73 GB（0.65 ms）。BF16 KV 讀一次約 4.1 ms〔算術〕。

---

## 10. 查不到、沒查證、與既有整理的差異

**本次沒找到**（≠不存在，CLAUDE.md 規則 7；查詢字串寫在 §3.4 與下面）：
- 跨層窗口的逐頻率 RoPE 可交換 K 基底（§3.4）。
- 在**還原（resume）情境**、n_q 為數百的 prefill 上量「latent 注意力 vs 先重建」的交叉點。所有前作都是 decode。
- 2026 年「dequantization-free attention」直接讀預存低位元 KV、用 tensor core 的新作。查詢字串 `dequantization-free attention quantized KV cache integer tensor core 2026 arXiv long context`，只回到 BitDecoding 與 Apple Metal 上的 Open-TQ-Metal（2604.16957，**沒讀**）。

**沒查證**：
- Llama-3.1 llama3 rope scaling 後的 θ_j 是否兩兩相異（§3.2）。〔複核 2026-10-27：已查證為兩兩相異，見 §3.2 的複核註〕
- H200 每 SM 的 shared memory 大小；H200 上 CUDA core 對 tensor core 的吞吐比（§6.1 借用 QServe／TurboAttention 的 A100／H100 說法）。
- Loki 的 pre-rotary 轉換在算分數時是套在 RoPE 前還是 RoPE 後的 K（p5 只說兩種 PCA 都當候選）。
- SALS 圖 2 的 N_c、表 6 的確切 GPU。
- LoRC App. C（p15）說把「旋轉與 key 投影併進 query 的計算」，但沒說位置相關的旋轉怎麼避免逐 key 處理。本檔**不採信**它能免重建〔部分〕。
- Eigen Attention 在 RoPE 模型上把 K「transform back to a high dimension before applying the RoPE」（§4.3 p5）〔部分〕。PseudoHunt PRIOR_ART 說它在 Llama-3 上有延遲懲罰〔二手〕，我沒讀到那張圖。
- Tucker Attention（2603.30033）、KV-Latent（2507.11273）、MHA2MLA 細節：只有 PseudoHunt PRIOR_ART 的轉述〔二手〕。
- venue：EliteKV、RAP、HACK、TurboAttention、BitDecoding、MILLION、LoRC、Eigen Attention、SageAttention3 未查證；SALS（NeurIPS 2025）、TransMLA（NeurIPS 2025）是〔二手〕。
- PseudoHunt 的數字沒有同儕審查，程式碼沒跑。

**與既有整理的關係**：
- LC §6（權重吸收）說「碰到 RoPE 就吸收不了；不訓練只剩 Palu 那一類」。本檔補上兩件事：
  - 精確的可交換條件（逐頻率、複線性，§3.2）。
  - 唯一直接的前作是負結果（PseudoHunt）。
  兩者一致，只是把「吸收不了」細化成「只有一個很小的家族吸收得了，而且它在單層 MHA 上輸 2.7 倍」。
- LC §2.3 末段說「TransMLA 的不變性條件提供另一條路……GQA 下每層只有 4–8 個 KV head，自由度很小」。本檔的跨層版把自由度提到 32 個 head-層，但 K1b 與 PseudoHunt 的結果讓我同樣不建議投入（只建議 §3.5 的便宜能量檢查）。
- `L5d_stacking.md:34,38`（JoLT）一致；本檔補「V 已在 rank space」。
- `L8a:343` 的 CommVQ 是〔摘要〕級；本檔讀到式 (19)(22)、成本「約 (R＋1)/2 倍」〔複核 2026-10-27：原寫「約 R＋1 倍」〕與表 5，和摘要級描述不衝突。
- arith §9 的 TTFT 註明「反量化不計」，但沒有提到低秩重建這一項。本檔 §9.3 指出它和 TTFT 同量級。**不是錯，是漏項**。

---

## 11. 參考文獻（IEEE 格式；查證日期 2026-10-27）

[1] C.-C. Chang *et al.*, "Palu: KV-Cache Compression with Low-Rank Projection," ICLR 2025. arXiv:2407.21118 (v2). https://arxiv.org/abs/2407.21118
[2] R. Krishnan, V. Schulz, "A JoLT for the KV Cache: Near-Lossless KV Cache Compression via Joint Rank-Bit Allocation," arXiv:2607.12550 (v4), 2026. https://arxiv.org/abs/2607.12550
[3] J. Li, Y. Zhang, M. Y. Hassan, T. Chafekar, *et al.*, "CommVQ: Commutative Vector Quantization for KV Cache Compression," ICML 2025 (PMLR 267). arXiv:2506.18879 (v1). https://arxiv.org/abs/2506.18879
[4] DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model," arXiv:2405.04434 (v5). https://arxiv.org/abs/2405.04434
[5] F. Meng, P. Tang, X. Tang, Z. Yao, X. Sun, M. Zhang, "TransMLA: MLA Is All You Need," arXiv:2502.07864 (v5). https://arxiv.org/abs/2502.07864
[6] PseudoHunt, "rope-kv-compression: Training-free RoPE-aware low-rank KV-cache compression on Llama-2-7B," GitHub repository, commit 9d7200c, 2026-09-23. https://github.com/PseudoHunt/rope-kv-compression
[7] J. Xin, T. Lyu, D. Keyes, H. Ltaief, M. Canini, "RAP: RoPE-Aligned Pruning for KV-Cache Compression," arXiv:2602.02599 (v4), 2026. https://arxiv.org/abs/2602.02599
[8] Y. Zhou, S. Song, B. Liu, Z. Xi, S. Jin, X. Fan, Z. Zhang, W. Li, X. Huang, "EliteKV: Scalable KV Cache Compression via RoPE Frequency Selection and Joint Low-Rank Projection," arXiv:2503.01586 (v1), 2025. https://arxiv.org/abs/2503.01586
[9] J. Mu, H. Huang, J. Zhang, *et al.*, "SALS: Sparse Attention in Latent Space for KV Cache Compression," arXiv:2510.24273 (v1), 2025. https://arxiv.org/abs/2510.24273
[10] P. Singhania, S. Singh, S. He, S. Feizi, A. Bhatele, "Loki: Low-rank Keys for Efficient Sparse Attention," NeurIPS 2024. arXiv:2406.02542 (v2). https://arxiv.org/abs/2406.02542
[11] C.-C. Chang, W.-C. Lin, *et al.*, "xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction," ICML 2026 (per PDF header). arXiv:2503.18893 (v2). https://arxiv.org/abs/2503.18893
[12] H. Sun, L.-W. Chang, *et al.*, "ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference," ICML 2025. arXiv:2410.21465 (v3). https://arxiv.org/abs/2410.21465
[13] J. Zhang *et al.*, "SageAttention: Accurate 8-Bit Attention for Plug-and-Play Inference Acceleration," ICLR 2025. arXiv:2410.02367 (v9). https://arxiv.org/abs/2410.02367
[14] J. Zhang *et al.*, "SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization," ICML 2025. arXiv:2411.10958 (v7). https://arxiv.org/abs/2411.10958
[15] J. Zhang *et al.*, "SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-bit Training," arXiv:2505.11594 (v3). https://arxiv.org/abs/2505.11594
[16] Y. Lin, H. Tang, S. Yang, *et al.*, "QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving," MLSys 2025. arXiv:2405.04532 (v3). https://arxiv.org/abs/2405.04532
[17] Z. Zhang, H. Shen, S. Vargaftik, R. Ben Basat, M. Mitzenmacher, M. Yu, "HACK: Homomorphic Acceleration via Compression of the Key-Value Cache for Disaggregated LLM Inference," arXiv:2502.03589 (v1), 2025. https://arxiv.org/abs/2502.03589
[18] H. Kang, S. Bharadwaj, J. Hensman, T. Krishna, V. Rühle, S. Rajmohan, "TurboAttention: Efficient Attention Approximation for High Throughputs LLMs," arXiv:2412.08585 (v3). https://arxiv.org/abs/2412.08585
[19] D. Du, S. Cao, J. Cheng, L. Mai, T. Cao, M. Yang, "BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache," arXiv:2503.18773 (v3). https://arxiv.org/abs/2503.18773
[20] Z. Liu, J. Yuan, H. Jin, S. Zhong, *et al.*, "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache," ICML 2024. arXiv:2402.02750 (v2). https://arxiv.org/abs/2402.02750
[21] H. Zhang *et al.*, "PQCache: Product Quantization-based KVCache for Long Context LLM Inference," arXiv:2407.12820 (v2). https://arxiv.org/abs/2407.12820
[22] Z. Wang, P. Xu, *et al.*, "MILLION: MasterIng Long-Context LLM Inference Via Outlier-Immunized KV Product QuaNtization," arXiv:2504.03661 (v2). https://arxiv.org/abs/2504.03661
[23] "VQ-LLM: High-performance Code Generation for Vector Quantization Augmented LLM Inference," HPCA 2025. arXiv:2503.02236 (v2). https://arxiv.org/abs/2503.02236
[24] R. Zhang, K. Wang, L. Liu, *et al.*, "LoRC: Low-Rank Compression for LLMs KV Cache with a Progressive Compression Strategy," arXiv:2410.03111 (v1). https://arxiv.org/abs/2410.03111
[25] U. Saxena, G. Saha, S. Choudhary, K. Roy, "Eigen Attention: Attention in Low-Rank Space for KV Cache Compression," arXiv:2408.05646 (v2). https://arxiv.org/abs/2408.05646

（[23] 的作者名單沒有逐一核對；[3][9][11][12][13][14][16][21][22][24] 只寫到 PDF 首頁看得到的部分〔未查證全名〕。）
