# LE_nla_conditions：「難但條件成立就大幅加速」的數值線性代數技巧——條件 → 加速對照表

> 2026-10-27，文獻 agent litE。規則見同資料夾 `RULES.md`。
> 問題（使用者原話大意）：「有很多進階、困難的線性代數技巧，或只有在很難達到的條件下才大幅加速的技巧——在什麼條件下才會大幅加速？」
> 範圍：attention／KV cache／長 context 推論上的 NLA 技巧，判斷條件在我們的設定（Llama-3.1-8B、Qwen2.5-7B／14B，GQA、RoPE、128K–256K、暫停後從 CPU／SSD 還原、H200）成不成立。
> **不重做**：S4／Mamba／SSD、線性注意力、DeltaNet、LoLCATs、Performer、HyperAttention、Alman & Song 2023（bounded entries）→ `LA_ssm_structure.md`（agent A）；混合模型狀態快取 → agent B；HCache、RoPE 對角、LSE 合併／CPU attention、稀疏 prefill 估計、Hadamard、MLA → `LC_attention_structure.md`（agent C）；2026 SOTA → agent D。
> 暫存：`<SCRATCH>/litE/`（PDF、`pdftotext -layout` 文字檔、torch `_lowrank.py` v2.5.0 原始碼、PyTorch CUDA notes HTML）。**本檔沒有在 Mac 上跑任何實驗**；只有〔算術〕（公式與代入值都寫出來）。

**證據等級用法**：〔部分〕＝下載 arXiv PDF、`pdftotext -layout` 後讀了卡片列出的頁與節，數字照 PDF 文字抄；沒有一篇逐頁讀完，所以本檔不標〔全文〕。〔摘要〕＝只讀到 PDF 第 1 頁摘要。頁碼一律是 **PDF 頁**。〔程式碼〕＝本 repo 程式（檔名＋行號）或上游原始碼。〔實測〕＝本 repo 已有的量測（附檔案）。〔判讀〕＝我的推論。

---

## §0 一頁結論（白話）

1. **大部分「理論上快幾個數量級」的技巧，需要的條件是 softmax attention 在長 context「平滑、分散」——而我們最在乎的長 context 檢索正好相反：注意力是「遠處一個尖峰」。** 所以以「位置距離」切塊的 H-matrix／FMM（H-Transformer-1D、FMMformer、Fast Multipole Attention）、低秩 attention（Linformer、Nyströmformer、Skyformer）、多項式草圖（PolySketchFormer）全部要**從頭訓練或改架構**；對訓練好的 Llama／Qwen，本次沒找到任何在 128K 量過「遠處區塊低秩」的證據（查詢字串見 §2.1）。Scatterbrain 把條件講得最清楚：注意力**熵高**時低秩好、**熵低**（尖峰）時只能稀疏（Thm 1，PDF p.5）。
2. **能在不訓練的情況下用的「FMM 精神」，是在「key 向量空間」而不是「token 位置」上分遠近**：Multipole Attention（NeurIPS 2025）把 key 分群，接近 query 的群做精確 attention、其餘用群中心近似（PDF p.1–2）。這幾乎就是 Tiara「縮圖還原」的形狀：縮圖＝群中心、還原時精確搬的是被挑中的 block。它量到的是 decode attention **最多 4.5 倍**（A6000、batch 16、90–95% 稀疏；A100 最多 2.8 倍，PDF p.9、p.16）。
3. **讓「重算（prefill）」變成次二次的路，條件不成立或沒人實作**：Alman–Yu（ICLR 2025）證明任何次二次模型解不了某類文件相似度任務（SETH 下，Thm 1.1–1.3，PDF p.3）；Alman–Song 的 RoPE 版本（2025）只有理論、原文自己說沒有任何實驗（PDF p.16）。而且就算 attention 免費，128K prefill 最多只快 3.1 倍（agent A 的 Amdahl 算術，`LA_ssm_structure.md:314`）。Strassen／AlphaTensor 在 GPU 上量到的是 **5–24%**，且都是 FP32 或未標精度、沒有 tensor core 版本的實測（§2.6）。Ozaki／誤差補償是「用低精度模擬高精度」，方向和「比 BF16 更便宜的重算」相反（〔算術〕§2.8）。
4. **真正條件成立、而且是 repo 自己的瓶頸的，是「寫入時的 SVD」**。repo 用的 `torch.svd_lowrank` **本來就是** Halko–Martinsson–Tropp 的隨機 SVD（`code/m16_xstack.py:1539`，niter＝4、多取 16 個，:104），不是 full SVD。H200 上 32K 量到 K 2.02 s＋V 3.29 s（`results/icpe_h200/p11/time/wcost.csv`）＝**同設定 FP32 峰值下界的約 6 倍**〔算術〕。三個條件同時成立：(a) 矩陣很瘦（T ≫ n＝4096）、(b) PyTorch 預設 FP32 matmul **不用** TF32 tensor core（PyTorch 1.12 起，〔文件〕；repo 沒改設定〔程式碼〕）、(c) 秩很高（每窗口 r＝928–1408，約 n 的 1/4–1/3）。在這三點下，「串流累加 Gram 矩陣 ZᵀZ（prefill 時逐 block 加）＋4096×4096 的小特徵分解」比隨機 SVD 少約 3.4 倍 GEMM 運算、而且可以用 TF32／BF16 tensor core，**下界從 0.89 s 降到 0.036 s（32K、Llama、K＋V）**〔算術，§2.3〕。但 repo 的事先校準基底（`:cal`）已經只要 0.042–0.131 s（`results/RUNLOG_H200_ICPE.md:310`），所以這條只在「每題 SVD 的品質確實比 `:cal` 好」時才有價值（目前兩個模型都 UNDETERMINED，同檔 :224）。**而且它是寫入成本，不是還原成本。**
5. **對還原本身，現實的大加速只有一種來源：少搬位元組**——低秩＋量化（repo 已做，xKV）或只搬被挑中的 block（縮圖還原，A0＝GA0-WEAK）。NLA 能幫的是讓「挑」更準（key 空間的多極近似、把沒搬的 block 用中心補上，而不是直接丟），不是讓 attention 本身換複雜度。
6. **可以在 H200 用既有 KV dump 驗證的條件**（§3）：K 的 ℓ∞ 與 logit 最大值（bounded entries）、注意力熵與 top-k 質量（Scatterbrain／RetrievalAttention 的條件）、xKV 窗口的奇異值衰減與 Gram 的特徵值間隙（隨機 SVD／Gram 路徑）、群中心補全的輸出誤差（Multipole）。**注意：dump 只有 K、V，沒有 Q**（`code/m16_kvstruct.py` 檔頭）；「遠處 attention 區塊的秩」需要所有位置的 Q，要在 GPU 上加 hook，不能只用既有 dump。

---

## §1 THE TABLE

欄位：技巧｜需要的條件（數學上精確寫）｜條件成立時的加速（複雜度 X→Y；原文實測倍數＋設定）｜我們的設定成立嗎（已知證據／要量什麼）｜難度（實作／理論）｜證據等級

| # | 技巧 | 需要的條件 | 條件成立時的加速 | 我們的設定成立嗎 | 難度 | 證據等級 |
|:--|:--|:--|:--|:--|:--|:--|
| 1 | **H-matrix／HSS／FMM，以 token 位置分遠近**（H-Transformer-1D、FMMformer、Fast Multipole Attention） | 注意力矩陣 A（或 A 減掉帶狀近場 D）的遠處非對角子塊在容差 ε 下數值秩 r ≪ 子塊大小，且 r 隨距離不增；FMM 的解析版要 A(i,j)＝g(‖qᵢ−kⱼ‖) 且 g 平滑、可分（FMMformer (A1)–(A3)、Def. 1 well-separated，PDF p.3） | 理論 O(n²)→O(n) 或 O(n log n)（FMMformer 摘要 p.1；H-Transformer-1D §4 p.2–3）。**實測只有訓練過的小模型**：H-Transformer-1D LRA 平均 61.41 vs Transformer 54.39（Table 1，p.9）；1B Word 144M 模型 ppl 20.25 vs 24.8（Table 2，p.9）。沒有對預訓練 LLM 的 wall-clock | **不成立（判讀）**。原文自己的證據：H-Transformer-1D 只說「我們假設」此結構也存在於 attention（p.4）；FMMformer 承認 ML 裡很難驗證 (A1)–(A3)（p.3），其低秩證據是 WikiText-103 上訓練的 Transformer、**n＝256**（Fig. 3，p.4）。反證：retrieval head 在所有長 context 模型都有、佔 <5% head、會看任意遠處的資訊（Wu et al. 摘要 p.1）→ 遠處區塊含尖峰。RoPE 的「長距衰減」只是上界（RoFormer §3.4.3 式 (35)–(37)，p.8），不是實際分數。要量：§3 S1 | 實作高（要改 attention kernel＋重訓）；理論成熟 | 〔部分〕 |
| 2 | **FMM／多極，以 key 向量空間分遠近（分群＋中心近似）**（Multipole Attention、Squeezed Attention） | 遠群中的 key 對當前 q 的分數可由群中心近似（群內 key 彼此接近、離 q 遠）；重要 key 集中在少數群（注意力稀疏） | decode attention：A6000 上 batch 1／4／16 為 2.5／4.1／4.5 倍（Qwen3-8B、90–95% 稀疏、每 16 token 一個中心，Fig. 5，p.9）；A100 為 1.3／2.1／2.8 倍（Fig. 6，p.16）；prefill 分群額外 13–15%（App. E，p.16）。是 **attention-only**，不是端到端 | **最接近成立**。注意力集中有證據：Llama-3-8B、100K token KV 檢索，top-1000 token（1%）平均回收 89% attention（RetrievalAttention Fig. 2，p.3–4）。不需訓練。和 Tiara 縮圖還原同形（縮圖＝中心）。要量：§3 S3、S8 | 實作中（分群在 prefill 做、還原時用）；理論弱（無保證） | 〔部分〕 |
| 3 | **稀疏＋低秩（Scatterbrain）** | 依 softmax「溫度」β 與 log n 的關係：β＝o(log n／log d) 時低秩夠；(1−Δ²) log n ≤ β ≤ O(log n) 時要稀疏＋低秩；β＝Ω(log n) 時低秩誤差 Ω(e^{β(1−Δ²)})，只能稀疏（Thm 1，Process 1 群聚生成模型，p.4–5） | 參數 n^{1+o(1)}（Thm 1）；預訓練 BigGAN／T2T-ViT 不微調直接換，近似誤差比基線低 2.1 倍（摘要 p.1） | **條件在檢索 head 上偏向「低溫」→ 只能稀疏**（判讀，依 §3 S2 的 log n 門檻）。要量：每列注意力熵 H／ln n、top-k 質量 | 實作中；理論是生成模型下的界，不是對真實模型 | 〔部分〕 |
| 4 | **低秩 attention 矩陣（Linformer、Nyströmformer、Skyformer）** | 注意力矩陣 P 的奇異值長尾（Linformer Fig. 1 用 RoBERTa、n＝512，p.3）；Linformer Thm 1 是「對固定一個 w，存在 Θ(log n) 秩的 P̃ 使 ‖P̃wᵀ−Pwᵀ‖ 小」的 JL 型機率界（p.3） | O(n²)→O(n)（各摘要）。Nyströmformer 要在 Wikipedia＋BookCorpus 重新預訓練（p.1）；Skyformer 改用 Gaussian kernel（p.2） | **不成立**：全部要訓練；Scatterbrain Thm 1 指出低溫下低秩誤差指數大。另見 agent A 的 HyperAttention 直接替換 LongBench 崩掉（`LA_ssm_structure.md` §0 第 6 點） | 實作中；理論弱 | 〔部分〕（Linformer、Skyformer）〔摘要〕（Nyströmformer） |
| 5 | **Butterfly／Monarch／Kaleidoscope 結構矩陣** | 權重矩陣 W 接近結構類 𝓜：Monarch 投影 argmin_{M∈𝓜}‖A−M‖_F 有解析解（Thm 1，O(n^{5/2})，p.6；Alg. 1＝把 A 重排成 m⁴ 張量、每個 m×m 區塊取秩 1，p.7） | Monarch：GPT-2 預訓練 2 倍、BERT D2S 微調 1.5–1.7 倍（Table 8，p.12：BERT-large GLUE 80.4→79.6）。**D2S 是投影後再微調** | **不成立且不對題**：(a) 加速的是權重 GEMM，不是 KV 還原；(b) 都要微調，ProcrustesGPT 原文說預訓練權重不微調就用結構矩陣表示「不實際」（摘要 p.1）；(c) 換了 W_K／W_V 就換了 KV，不是精確重算。要量（低優先）：‖W−Π_𝓜(W)‖_F／‖W‖_F，§3 S9 | 實作高；理論成熟 | 〔部分〕（Monarch）〔摘要〕（Kaleidoscope、M2、ProcrustesGPT） |
| 6 | **隨機 SVD（HMT 2011），用在 xKV 寫入** | 奇異值衰減：誤差 E‖A−QQ*A‖_F ≤ (1＋k/(p−1))^{1/2}(Σ_{j>k}σⱼ²)^{1/2}（Thm 10.5，p.56）；冪迭代 q 次時譜範數因子以 ^{1/(2q+1)} 收斂到 σ_{k+1}（Cor. 10.10，p.60–61）；原文說 q＝1 或 2 通常夠（p.9） | 成本 (2q＋2)·ℓ·T_mult＋O(ℓ²(m+n))（p.9）；對 m×n 稠密矩陣 T_mult＝2mn flop。**repo 已在用**（torch.svd_lowrank） | **條件成立**（K1：RoPE 前 K 每 head 95% 能量 36–40 維，`docs/stage3_20261019/results_analysis.md:85–87`）。但 repo 的秩很高（每窗口 r＝928–1408），niter＝4 → 成本 10·2·T·n·ℓ；H200 32K 實測是 FP32 峰值下界的約 6 倍（§2.3）。改 q＝1 可省 GEMM 2.5 倍〔算術〕 | 實作低（改參數）；理論成熟 | 〔部分〕＋〔程式碼〕＋〔實測〕 |
| 7 | **串流 Gram（ZᵀZ 逐 block 累加）＋小特徵分解**；Frequent Directions；Brand 增量 SVD | Gram：n 小到 n×n 放得下（n＝4096 → 64 MB FP32），且需要的特徵值 λ_r 與 λ_{r+1} 的間隙遠大於捨入誤差 ‖E‖≈u·λ₁（判讀，Davis–Kahan 型）。FD：‖AᵀA−BᵀB‖₂ ≤ ‖A−A_k‖_F²/(ℓ−k)（Ghashami et al. Thm 1.1，p.4），時間 O(ndℓ)（Thm 1.5，p.4）。Brand：秩 r ≤ √min(p,q) 時 O(pqr)（摘要 p.1） | Gram：一次 GEMM 2Tn² flop、可放 tensor core、和 prefill 重疊；32K Llama K＋V 下界 0.036 s（TF32）vs 隨機 SVD FP32 0.89 s〔算術〕 | **Gram 條件成立**（n＝2048–4096 ≪ T）；FD、Brand 不需要：FD 的好處是 n 也大時省記憶體，我們 n 小；Brand 要 r ≤ √4096＝64，但 xKV 每窗口 r＝928 起 → **不成立**〔算術〕。要量：§3 S4、S5 | 實作低；理論成熟 | 〔部分〕（FD）〔摘要〕（Brand）〔算術〕 |
| 8 | **多項式核＋草圖（PolySketchFormer；TensorSketch）** | 把 softmax 換成高次多項式注意力（架構改變），再用草圖近似（摘要 p.1） | 線性時間；GPT-2 small 級訓練延遲（Fig. 1，p.1） | **不成立**：要訓練（換掉 softmax） | 實作高 | 〔摘要〕 |
| 9 | **Toeplitz／circulant＋FFT** | 分數矩陣 S 的 (m,n) 元素只依賴 n−m（Toeplitz），才能嵌入 2 倍大的 circulant 用 FFT 做 O(n log n) 乘法（Fast RoPE §技術概述，p.5） | O(n²)→O(n log n) | **不成立**：RoPE 的分數 q_mᵀk_n＝x_mᵀW_q R_{n−m} W_kᵀx_n（RoFormer 式 (16)，p.5）——旋轉核只依賴 n−m，但前後乘上隨 m、n 變的內容 → S 不是 Toeplitz。Alman–Song 的 RoPE 演算法正是為此改用「重新縮放的 Toeplitz 之和」（p.5） | 理論高；無實作 | 〔部分〕 |
| 10 | **多項式方法（bounded entries）＋RoPE 版** | ‖Q‖_∞,‖K‖_∞,‖V‖_∞ ≤ B＝o(√log n)，d＝O(log n)（Fast RoPE Thm 1.3，p.4；原 Alman–Song 見 agent A）；n＝131,072 時 √ln n≈3.43，262,144 時 ≈3.53〔算術〕 | n^{1+o(1)}，1/poly(n) 加性誤差（Thm 1.3，p.4）；**原文無任何實驗**（Limitations，p.16）。B＝Ω(√log n) 時 SETH 下不可能次二次（Thm 1.4，p.4） | **預期不成立**：要把注意力集中到一個 token 需要 Θ(log n) 的 logit 差（agent A 的算術，`LA_ssm_structure.md:149`），正是下界區間。要量：§3 S2 | 理論高；實作無 | 〔部分〕 |
| 11 | **Tucker／Tensor-train／Kronecker** | 多出來的模態（層、head、通道）要共用座標 | — | **repo 量過：沒有額外幫助**（K1b、R8：`docs/stage3_design_20261024/mechanisms/05_rejected.md:15`；`results_analysis.md:97`）。新事實只有 ProcrustesGPT 用 Kronecker 壓**權重**（不微調，Llama2 ppl 上升，Table 1，p.8）——對 KV 還原不相關 | — | 〔實測〕＋〔摘要〕 |
| 12 | **快速矩陣乘法（Strassen、AlphaTensor）** | 矩陣夠大且接近方形；GEMM 為運算瓶頸；可接受只有 normwise（grade C）誤差保證（Strassen 類演算法不滿足元素級穩定性，2511.13778 p.6 引 Miller） | 每層遞迴乘法數 8→7（≤12.5%／層）。實測：V100 FP32 SGEMM 1 層 1.11 倍、2 層 1.19 倍（20,480 方陣，1808.07984 摘要 p.1）；AlphaTensor V100 8,192 方陣 8.5%（Strassen-square 4.3%）、20,480 方陣 23.9%／21.3%（Fig. 5a，p.6，見 §2.6 的配對說明） | **不成立（實務上）**：prefill 的 GEMM 是 T×4096 × 4096×{4096, 14336}，內維 4096；本次沒找到 Hopper／Ampere **tensor core** 上 Strassen 的實測（查詢字串見 §2.6）；改變 KV 的捨入 → 不是逐位元重算 | 實作高；理論成熟 | 〔部分〕 |
| 13 | **MIPS／ANN 子線性 top-k（RetrievalAttention、PQCache）** | 注意力集中在少數 key；索引要處理 q 與 k 分布不同（OOD）：一般 IVF 要掃 30–50% key 才 recall>0.95，k→k 只要 1–5%（RetrievalAttention p.2、p.5） | RetrievalAttention 只需存取 1–3% key（摘要 p.1）；decode 用 | **部分成立**（集中性有：上列 89%；OOD 是真的）。repo：Quest 摘要不夠（C1b 停，`results_analysis.md` 4.10）；詳見 agent C 的 MagicPIG／RetroInfer 與 `docs/stage3_20261019/lit/L7d_recheck_lowrank_index.md:92`。要量：§3 S7 | 實作中高 | 〔部分〕 |
| 14 | **混合精度模擬（Ozaki、Ootomo 誤差補償）** | 用多個低精度 GEMM 拼出高精度乘積；只有在「低精度 tensor core 吞吐 ÷ 拆分項數 > 目標精度原生吞吐」時才划算 | Ootomo：A100 上以 FP16／TF32 tensor core 達 FP32 精度 51／33 TFLOP/s，超過 FP32 SIMT 峰值 19.5（摘要 p.1）；Ozaki II：GH200 上 FP64 模擬 56.6–80.2 TFLOPS（摘要 p.1） | **對重算不成立**：目標是 BF16 等級，H200 FP8 只比 BF16 快 2 倍（dense 1,979 vs 989.5 TFLOPS），補償要 ≥2–3 個 FP8 GEMM → 不會比 BF16 快〔算術〕。**對寫入 SVD 成立**：SVD 的 FP32 GEMM 可改 TF32（或 3×TF32 補償到 FP32 精度），H200 TF32 dense 494.5 vs FP32 67〔文件＋算術〕 | 實作低（寫入端）；理論成熟 | 〔摘要〕＋〔文件〕＋〔算術〕 |
| 15 | **次二次替代物的下界（Alman–Yu）** | — （這是「不可能」的條件） | SETH 下，文件相似度類任務 LSD／MSD 無 O(n^{2−ε}) 解（Thm 1.1–1.3，ℓ＝c log n，p.3） | 對 Tiara：任何要在還原時「把 128K 重算成次二次」的近似，在這類任務上必有失敗例〔判讀〕 | 理論 | 〔部分〕 |

---

## §2 卡片

### 2.1 位置上的階層低秩（H-matrix／FMM）

#### H-Transformer-1D：Fast One-Dimensional Hierarchical Attention for Sequences（ACL 2021；arXiv 2107.11906）
- **讀了什麼**：〔部分〕v1，https://arxiv.org/abs/2107.11906 ，讀 PDF p.1–4（§4–5）、p.8–9（Table 1–2），查證 2026-10-27
- **一句話**：把 H-matrix 的「近處全秩、遠處低秩」當 attention 的歸納偏置，從頭訓練。
- **線性代數的關鍵一步**：A＝A⁽²⁾＋A⁽¹⁾＋A⁽⁰⁾ 的多層分解（式 (16)–(18)，p.4），遠處子塊用固定數值秩 N_r；token 階層靠逐層平均（coarsening，p.4–5）。
- **成立的前提**：原文明說這是「假設」：它假設物理問題常見的階層低秩結構也存在於 NLP 的 attention，並把好的 benchmark 結果當成「正面證據」（p.4）——這是間接證據，不是在任何預訓練模型上量 A 的子塊秩。要從頭訓練。
- **量到的數字**：LRA 平均 61.41（Transformer 54.39；Table 1，p.9）；One-Billion Word：53M 模型 ppl 23.95 vs 30.04、144M 模型 20.25 vs 24.8（N_r＝16，Table 2，p.9）。無 wall-clock 對 FlashAttention 的比較（本次讀的頁沒有）。
- **跟 Tiara 的關係**〔判讀〕：要改模型，不能套在 Llama／Qwen 的還原上。
- **與既有整理的關係**：repo 無（grep `H-Transformer` 0 筆）。

#### FMMformer：Efficient and Flexible Transformer via Decomposed Near-field and Far-field Attention（NeurIPS 2021 依使用者；venue 〔未查證〕；arXiv 2108.02347）
- **讀了什麼**：〔部分〕v1，https://arxiv.org/abs/2108.02347 ，讀 PDF p.1–5（§2.1–3.2、Fig. 3），查證 2026-10-27
- **一句話**：attention＝帶狀近場（精確）＋低秩遠場（線性注意力核），訓練出來的模型。
- **線性代數的關鍵一步**：FMM 的 well-separated 條件：存在代表點 k* 與 δ∈(0,1) 使 ‖kⱼ−k*‖ ≤ δ‖qᵢ−k*‖（Def. 1，p.3）；在 (A1) A(i,j)＝g(‖qᵢ−kⱼ‖)、(A2) g 平滑、(A3) g 可分下，遠場子塊可被秩 p ≥ C|log_δ ε| 近似（Lemma 1，p.3）。實作改用 diagonal-plus-semi-separable 的定義（Def. 2，p.3）。
- **成立的前提**：原文自己說在機器學習裡 (A1)–(A3) 很難驗證（p.3）。它的「A−D 低秩」證據：WikiText-103 訓練的小 Transformer、256×256 attention 矩陣、奇異值門檻 1e-6（Fig. 3，p.4）。**n＝256、訓練過的、非 RoPE 長 context LLM**。
- **量到的數字**：LRA 五項平均 60.74% vs 標準 58.70%（摘要 p.1）。
- **跟 Tiara 的關係**〔判讀〕：注意 FMM 原始條件是「**向量空間距離**」g(‖q−k‖)，不是 token 位置差；把它換成「位置差」是 H-Transformer／FMMformer 的設計選擇。對我們有意義的反而是向量空間版本（→ Multipole Attention）。
- **與既有整理的關係**：repo 無。

#### Fast Multipole Attention（Kang, Tran, De Sterck；arXiv 2310.11960 v4）
- **讀了什麼**：〔摘要〕v4（2025-09-18），https://arxiv.org/abs/2310.11960 ，查證 2026-10-27
- **一句話**：O(log n) 層的「學出來的」階層，近處全解析、遠處逐層粗化；O(n log n)／O(n)。
- **成立的前提**：階層基底是學出來的 → 要訓練（摘要）。
- **跟 Tiara 的關係**：不能直接用。

**有沒有「預訓練 LLM 在長 context 的遠處 attention 子塊是低秩」的實測？** 本次搜尋沒找到（查詢字串：`off-diagonal blocks attention matrix numerical rank pretrained LLM long context hierarchical low-rank Llama empirical analysis`）。找到的相關只有：LRQK 說 QKᵀ 可低秩近似（〔二手：搜尋摘要〕，arXiv 2510.23649；repo 已在 `L5a_lowrank.md` 讀過 LRQK）——但 **QKᵀ（exp 之前）的秩本來就 ≤ d＝128**（Q、K 都是 n×128；RoPE 後 q̃ₘ＝R_m W_q xₘ，仍是 n×128，RoFormer 式 (14)–(16)，p.5），H-matrix 要的是 **exp 之後** 的子塊低秩，兩件事不同〔判讀〕。

**反證（為什麼預期不成立）**：
- Retrieval head：4 個模型家族、6 個規模都有，佔 <5% 的 head，延長到 32–128K 仍是同一批 head，砍掉就檢索失敗（Wu et al. 2404.15574，摘要 p.1，〔摘要〕）。檢索時一列 attention 的質量集中在遠處一個（或少數）位置 → 遠處子塊裡「每列一個尖峰、尖峰的欄位隨列變」，這種子塊的秩隨不同尖峰欄位的數目成長〔判讀〕。
- RoPE 的長距衰減（RoFormer §3.4.3，p.8）：用 Abel 求和證的是 |Σ hᵢ e^{i(m−n)θᵢ}| 的**上界**隨 |m−n| 衰減，不是說實際分數衰減；retrieval head 的存在就說明實際分數可以在遠處很大〔判讀〕。

### 2.2 key 空間的多極近似

#### Multipole Attention for Efficient Long Context Reasoning（NeurIPS 2025；arXiv 2506.13059 v2）
- **讀了什麼**：〔部分〕v2，https://arxiv.org/abs/2506.13059 ，讀 PDF p.1–2、p.9–10（§5.3、Fig. 5）、p.15–17（App. D–E），查證 2026-10-27
- **一句話**：key 分群；靠近 q 的群做精確 attention，其他群用中心的分數近似；可多層（越遠越粗）。
- **線性代數的關鍵一步**：FMM 的「遠場用代表點」，但「遠近」用 q 與群中心的分數判定（Fig. 1 說明，p.2）；建立在 Squeezed Attention（Hooper et al. 2024a）的分群上。
- **成立的前提**：不用訓練；在 Qwen3-8B、DeepSeek-R1-Distil-Qwen2.5-14B 上評估（摘要 p.1）。需要 prefill 時分群（overhead 13–15%，App. E，p.16）。
- **量到的數字**：A6000、Qwen3-8B、90%／95% 稀疏、每 16 token 一個中心：attention 對 FlashDecoding 的加速 batch 1／4／16 為 2.5／4.1／4.5 倍（Fig. 5，p.9）；A100 為 1.3／2.1／2.8 倍（Fig. 6、App. E，p.16）；生成時的快速重分群 overhead 約 3–5%（p.16）。品質：GSM-Infinite 上比 Squeezed Attention 更接近基線（p.9，〔部分〕，表中數字未抄）。
- **跟 Tiara 的關係**〔判讀〕：這是本檔裡**唯一「不用訓練、條件看起來成立、又和還原同形」**的 NLA 技巧。對應關係：Tiara 縮圖 ≈ 群中心；還原時精確搬的 block ≈ 精確 attention 的群；**沒搬的 block 不丟、用中心補上**——這是 A0（GA0-WEAK，`results/RUNLOG_H200_S3.md:463–467`）的 ORACLE 選擇之外可以加的一件事。它省的是「要搬多少位元組才不掉分」，加速倍數上限＝1／搬運比例。
- **與既有整理的關係**：repo 無（grep `Multipole` 0 筆）。Squeezed Attention repo 也沒有。

### 2.3 隨機／串流數值線性代數：xKV 的寫入成本

#### HMT：Finding Structure with Randomness（SIAM Review 53(2) 2011；arXiv 0909.4061）
- **讀了什麼**：〔部分〕https://arxiv.org/abs/0909.4061 ，讀 PDF p.9–10（§1.6、Thm 1.2、成本公式）、p.27（Alg. 4.4）、p.56–57（Thm 10.5–10.6）、p.60–61（Cor. 10.10），查證 2026-10-27
- **關鍵事實**（照原文）：
  - 成本：T_randSVD＝(2q＋2)·k·T_mult＋O(k²(m＋n))，只需 2(q＋1) 次掃過矩陣；q＝1 或 2 通常就夠（p.9）。
  - Thm 1.2（p.9–10）：取秩 2k 的分解，E‖A−UΣV*‖ ≤ [1＋4√(2 min{m,n}/(k−1))]^{1/(2q+1)} σ_{k+1}。
  - Thm 10.5（p.56）：E‖(I−P_Y)A‖_F ≤ (1＋k/(p−1))^{1/2}(Σ_{j>k}σⱼ²)^{1/2}，k ≥ 2、p ≥ 2。
  - Thm 10.6（p.57）：譜範數 ≤ (1＋√(k/(p−1)))σ_{k+1}＋(e√(k+p)/p)(Σ_{j>k}σⱼ²)^{1/2}。
  - Cor. 10.10（p.60–61）：冪迭代把額外因子以 ^{1/(2q+1)} 壓到 1；q∼log min{m,n} 時誤差 ∼σ_{k+1}。
  - Alg. 4.4（p.27）：每次乘 A 或 A* 之後都做 QR（穩定版）。
- **repo 現況**〔程式碼〕：`torch.svd_lowrank(Z, q=r+16, niter=4)`（`code/m16_xstack.py:104`、`:1539`、`:1621`）；Z＝T×(W·H·D)，W＝4 層、K 先逆 RoPE（:1531–1538）。torch v2.5.0 的實作（`torch/_lowrank.py` 的 `get_approximate_basis`）：先 A·R、QR，之後每次迭代 Aᴴ·Q→QR→A·Q→QR，最後 B＝Qᴴ·A、對 B 做 `torch.linalg.svd`；矩陣寬時先轉置（`_svd_lowrank`）——即 HMT Alg. 4.4＋Alg. 5.1，**每次迭代 2 次瘦長 QR**。
- **repo 實測**〔實測〕：
  - MI300X、16K token（32 block）、GPU 閒：Llama `xl:k22:v34` K 9.40 s＋V 15.23 s；Qwen K 3.56 s＋V 6.02 s（`results/m16_stage3_mi300x/x3_time/v2/write_cost.csv`）；同表「÷ 同長度重算」4.75–13.8 倍（Llama）。
  - H200、32K、閒：Llama `k29v44` K 2.02 s、V 3.29 s（`results/icpe_h200/p11/time/wcost.csv`，3 次中位數）；全部 2.13–5.33 s，`:cal` 投影 0.042–0.131 s；每題 SVD 是同長度 prefill 的 1.14–2.51 倍（`docs/h200/icpe/P1-1_xkv_int4_h200.md:141`）。
- **〔算術〕成本下界**（只算 flop ÷ 峰值，是時間的**下界**；H200 峰值：FP32 67、TF32 dense 494.5、BF16 dense 989.5 TFLOPS＝NVIDIA 規格表「with sparsity」數字 ÷ 2〔文件〕）：
  - 符號：T＝token 數；n＝W·H·D（Llama、Qwen-14B：4·8·128＝4096；Qwen-7B：4·4·128＝2048，H＝4 由 repo 的基底位元組反算吻合：7 窗口 ×(11+17)·16·2048·2 B＝12,845,056，同 write_cost.csv）；ℓ＝r＋16；窗口數＝層數／4。
  - 隨機 SVD（niter＝q）：GEMM (2q＋2)·2·T·n·ℓ＋瘦長 QR (q＋1)·2·T·ℓ²＋q·2·n·ℓ²＋小 SVD 4·n·ℓ²。
  - Gram：2·T·n²（每窗口，K、V 各一次）；之後 n×n 的特徵分解與 T 無關（時間〔未量〕）。
  - 代入：

    | 設定 | 隨機 SVD q＝4（repo）FP32 下界 | 同，q＝1 FP32 | Gram FP32 | Gram TF32 | Gram BF16 |
    |:--|--:|--:|--:|--:|--:|
    | Llama 32K、k29v44（ℓ＝944／1424），K＋V | 0.89 s | 0.36 s | 0.26 s | 0.036 s | 0.018 s |
    | Llama 128K、k29v44 | 3.51 s | 1.41 s | 1.05 s | 0.14 s | 0.071 s |
    | Qwen-7B 128K、k29v44（ℓ＝480／720） | 0.78 s | 0.31 s | 0.23 s | 0.031 s | 0.016 s |
    | Qwen-14B 128K、k29v44（48 層〔未查證：config〕） | 5.27 s | 2.11 s | 1.58 s | 0.21 s | 0.11 s |
    | Llama 128K、**總秩 64**（ℓ＝80，委託者問的設定） | 0.21 s | 0.08 s | 1.05 s | 0.14 s | 0.071 s |

  - **實測 ÷ 下界**：Llama 32K K 2.02 s ÷ 0.343 s＝5.9 倍；V 3.29 ÷ 0.548＝6.0 倍。
  - 判讀：(1) 低秩（總秩 64）時隨機 SVD 本來就便宜，Gram 反而貴；(2) **repo 的實際秩（每窗口 928–1408，約 n 的 1/4–1/3）下，Gram 的 GEMM flop 比 niter＝4 的隨機 SVD 少 3.4 倍**（0.89／0.26），而且不需要瘦長 QR（GPU 上 QR 效率差，這是 MI300X 時就猜過的瓶頸，`docs/stage3_20261019/experiment_log.md:181`〔推論，未驗〕）；(3) PyTorch 1.12 起 `allow_tf32` 預設 False（PyTorch CUDA notes，〔文件〕，https://docs.pytorch.org/docs/2.14/notes/cuda.html ），repo 沒有設 `allow_tf32`／`set_float32_matmul_precision`（grep 0 筆，〔程式碼〕）→ 目前的 SVD GEMM 跑在 FP32 CUDA core（67 TF），不是 tensor core（假設沒有環境變數覆寫，〔未查證〕）。
- **成立的前提（Gram 路徑）**：要的是「前 r 個主方向的子空間」，誤差以 Frobenius 能量衡量；Gram 把條件數平方，若 λ_r／λ₁（＝(σ_r/σ₁)²）接近 u（FP32 u≈6e-8；TF32 輸入只有 10 位尾數）則第 r 個方向不準。它影響的是尾端方向；尾端方向的能量小，所以對重建誤差的影響可能小——**要量**（§3 S5）。repo 的 `:cal` 校準已經是 eigh(G)（`code/m16_xstack.py:2063`、`:2125`），K1 分析也用 Gram 串流（`code/m16_kvstruct.py` 檔頭「Gram 矩陣串流累加」）→ 程式上是現成的。
- **跟 Tiara 的關係**〔判讀〕：降的是**寫入**成本（暫停時），不是還原。只有在「每題 SVD 的品質 > `:cal`」成立時才值得做（目前 UNDETERMINED，`results/RUNLOG_H200_ICPE.md:224`）。

#### Frequent Directions（Liberty, KDD 2013, arXiv 1206.0594；Ghashami, Liberty, Phillips, Woodruff, SIAM J. Comput. 2016, arXiv 1501.01711）
- **讀了什麼**：〔部分〕1501.01711 PDF p.4（Thm 1.1、1.2、1.5）；1206.0594〔摘要〕＋p.1–2 的界，查證 2026-10-27
- **關鍵事實**：逐列處理 A∈ℝ^{n×d}，得 B∈ℝ^{ℓ×d}，0 ≤ ‖Ax‖²−‖Bx‖² ≤ ‖A−A_k‖_F²/(ℓ−k)（Thm 1.1）；‖A−π_B^k(A)‖_F² ≤ (1＋k/(ℓ−k))‖A−A_k‖_F²（Thm 1.2）；時間 O(ndℓ)（Thm 1.5）。Liberty 2013 的原始界是 ‖AᵀA−BᵀB‖ ≤ ε‖A‖_F²（1206.0594 p.1）。
- **跟 Tiara 的關係**〔判讀〕：FD 的價值在「d 也很大、存不下 d×d」時；我們 d＝n＝4096（全部 32 層串起來也才 32,768，Gram 4.3 GB FP32，H200 放得下）→ 精確 Gram 比 FD 好（無近似、同一次掃描）。**不需要**。

#### Brand：Fast Low-Rank Modifications of the Thin SVD（Linear Algebra Appl. 415(1) 2006；MERL TR2006-059）
- **讀了什麼**：〔摘要〕MERL 技術報告封面頁，https://www.merl.com/publications/docs/TR2006-059.pdf ，查證 2026-10-27
- **關鍵事實**：單次掃描的串流 thin SVD；秩 r 的 p×q 矩陣在 r ≤ √min(p,q) 時 O(pqr)（摘要）。
- **跟 Tiara 的關係**〔算術〕：√4096＝64；xKV 每窗口 r＝22·32＝704 起（`k22`）→ 條件不成立。

### 2.4 結構矩陣（Butterfly／Monarch／Kaleidoscope）

#### Monarch：Expressive Structured Matrices for Efficient and Accurate Training（ICML 2022；arXiv 2204.00595）
- **讀了什麼**：〔部分〕v1，https://arxiv.org/abs/2204.00595 ，讀 PDF p.1–2、p.6–7（§3.3、Thm 1、Alg. 1）、p.11–12（§5.3、Table 8），查證 2026-10-27
- **線性代數的關鍵一步**：M＝P L Pᵀ R（兩個區塊對角＋置換）；重排成 m×m×m×m 張量後是 m² 批秩 1 矩陣 → 投影＝每批取 SVD 的秩 1（p.6–7），整體 O(n^{5/2})（Thm 1，p.6）。
- **量到的數字**：BERT-base GLUE 78.6→78.3、BERT-large 80.4→79.6，微調快 1.5／1.7 倍（Table 8，p.12）——是**投影後再微調**（§5.3，p.11）。
- **本次沒找到**「不微調直接把 Llama 的 W_K／W_V／W_O 投影到 Monarch」的準確度（查詢字串：`Monarch projection pretrained LLM weights post-hoc structured matrix approximation Llama accuracy`）。ProcrustesGPT（Findings ACL 2025 依搜尋結果，〔未查證〕；arXiv 2506.02818）摘要說預訓練權重不微調就能用結構矩陣準確表示「似乎不實際」，所以先做正交變換（〔摘要〕p.1）。
- **跟 Tiara 的關係**〔判讀〕：加速權重 GEMM（重算的線性層部分），不是 attention、不是搬運；改了 W 就改了 KV → 不是精確重算。不建議。
- Kaleidoscope（ICLR 2020；arXiv 2012.14966）與 Monarch Mixer（NeurIPS 2023；arXiv 2310.12109）〔摘要〕：都是端到端學出來的結構（K-matrix 自動學習、M2 新架構）。

### 2.5 Toeplitz／FFT 與 fine-grained complexity

#### Fast RoPE Attention：Combining the Polynomial Method and Fast Fourier Transform（Alman & Song；arXiv 2505.11892 v1）
- **讀了什麼**：〔部分〕v1（2025-05-17），https://arxiv.org/abs/2505.11892 ，讀 PDF p.1–5（Def. 1.1、Thm 1.3–1.4、技術概述）、p.16（Limitations），查證 2026-10-27
- **一句話**：在 bounded entries 下給 RoPE attention 一個 n^{1+o(1)} 近似演算法。
- **線性代數的關鍵一步**：一般化 RoPE：A_{ij}＝exp(Qᵢ W_{i−j} Kⱼᵀ/√d)，W 的支撐 |S|＝O(d)（Def. 1.1，p.4）。exp 之前的矩陣在 d＝1 時可以是任意 Toeplitz、不是低秩，所以舊的多項式方法（f(低秩) 近似低秩）不能用；改證「f(少數個重新縮放 Toeplitz 之和) ≈ 少數個重新縮放 Toeplitz 之和」，再用 FFT（p.5）。
- **條件**：ε＝1/poly(n)、B＝o(√log n)、d＝O(log n)（Thm 1.3，p.4）；B＝Ω(√log n) 時 SETH 下不可能（Thm 1.4，p.4）。
- **量到的數字**：無。原文 Limitations 明說沒有任何實驗驗證實際效能（p.16）。
- **我的〔判讀〕（和原文框架不同，請獨立核對）**：標準 RoPE（不是 Def. 1.1 的一般化）中，R_{n−m}＝R_mᵀR_n（RoFormer 式 (16) 後一行，p.5），所以 q_mᵀk_n＝(R_m W_q x_m)ᵀ(R_n W_k x_n)——就是把 Q、K 各自旋轉後的**標準** attention，exp 之前的矩陣秩 ≤ d。每個 2×2 旋轉讓一個座標的絕對值最多變成 √(x²＋y²) ≤ √2·max(|x|,|y|)，所以旋轉後的 ℓ∞ 界 ≤ √2·B。依此，原本 Alman–Song 2023 的演算法對標準 RoPE 應該直接適用（B 換成 √2B）。原文的「舊技巧不能用」適用於它的一般化 W_{i−j}。不論哪個框架，**瓶頸都是 B＝o(√log n) 這個條件**（§3 S2）。另：Def. 1.1 寫 /√d，但 Remark 1.2 又說 exp 裡是 1/d 的正規化（p.4），原文前後不一致。
- **與既有整理的關係**：Alman–Song 2023 在 `LA_ssm_structure.md:140–152`（agent A）；本卡只補 RoPE 版本。

#### RoFormer（Su et al., Neurocomputing 2024；arXiv 2104.09864）—— 只核一句話
- **讀了什麼**：〔部分〕PDF p.5（式 (14)–(16)）、p.8（§3.4.3），查證 2026-10-27
- **「RoPE 分數不是 Toeplitz」的精確說法**：q_mᵀk_n＝x_mᵀW_q R^d_{Θ,n−m} W_k x_n（式 (16)，p.5）。旋轉部分只依賴 n−m（區塊 Toeplitz 的算子族），但分數矩陣 S_{mn} 還依賴內容 x_m、x_n，所以 S 不是 Toeplitz；只有在所有 x 相同（內容常數）時才是〔判讀，依式 (16)〕。原文沒有寫「不是 Toeplitz」這句話——它只是式 (16) 的直接推論。Alman–Song 的「重新縮放 Toeplitz」說法（D₁CD₂，p.5）是 d＝1 的特例。
- **FFT 加速要什麼**：整個分數矩陣（或 exp 之後的矩陣）是 Toeplitz／circulant，或可寫成少數個重新縮放 Toeplitz 之和（Fast RoPE p.5）。加性相對位置偏置（ALiBi、T5 bias）本身是 Toeplitz，但加上 QKᵀ 後 exp(QKᵀ＋B) 不是〔判讀；ALiBi、T5 原文本次未讀〕。只有「整個混合矩陣不依賴內容」的架構（例如長卷積／SSM 的核）才直接適用 → 見 agent A 的 S4 卡。

#### Fundamental Limitations on Subquadratic Alternatives to Transformers（Alman & Yu，ICLR 2025；arXiv 2410.04271）
- **讀了什麼**：〔部分〕PDF p.1–4（摘要、Thm 1.1–1.3），查證 2026-10-27
- **關鍵事實**：SETH 下，最相似／最不相似文件對（MSD／LSD）在 ℓ＝c log n 維時沒有 O(n^{2−ε}) 演算法（Thm 1.1–1.3，p.3）；所以任何次二次模型（含近似 attention、線性替代）都解不了明示或隱含需要文件相似度的任務；標準 transformer 可以（p.3–4）。
- **跟 Tiara 的關係**〔判讀〕：給「還原時改用次二次近似重算」一個硬上限：不是工程沒做好，是有任務類別一定會失敗。

#### The Fine-Grained Complexity of Gradient Computation（Alman & Song，arXiv 2402.04497）、RoPE Attention Can Be Trained in Almost Linear Time（Cao et al.，arXiv 2412.17316）
- 〔摘要〕兩篇都是**訓練時梯度**的 bounded-entries 結果，與推論還原無關，只記錄已查。

### 2.6 快速矩陣乘法

#### Implementing Strassen's Algorithm with CUTLASS on NVIDIA Volta GPUs（Huang, Yu, van de Geijn；FLAME Working Note #88；arXiv 1808.07984）
- **讀了什麼**：〔部分〕PDF p.1–3、grep 全文，查證 2026-10-27
- **量到的數字**：V100，1 層 Strassen 對 cublasSgemm 最多 1.11 倍、交叉點 1,536；2 層（要額外 workspace）1.19 倍、交叉點 7,680（摘要 p.1）。**基準是 SGEMM（FP32 CUDA core）**，不是 tensor core。
- **本次沒找到** Ampere／Hopper tensor core 上 Strassen 的實測（查詢字串：`Strassen algorithm tensor cores GPU measured speedup CUTLASS arXiv`、`Strassen tensor core FP16 Ampere Hopper performance numerical error fast matrix multiplication 2023 2024`）。
- **數值**：Strassen 類演算法只能滿足 normwise（grade C）誤差界；Miller 證明強穩定性需要 O(n³)（2511.13778 p.6 的轉述，〔部分〕）。

#### AlphaTensor：Discovering faster matrix multiplication algorithms with reinforcement learning（Nature 610, 2022；doi 10.1038/s41586-022-05172-4）
- **讀了什麼**：〔部分〕Nature PDF p.5–6（「Rapid tailored algorithms」段、Fig. 5），查證 2026-10-27
- **量到的數字**：對 4×4 區塊（8,192 方陣，每塊 2,048）在 V100 與 TPU v2 上找硬體量身的演算法，不遞迴（p.5）；相對 cuBLAS 的加速，V100 8,192：8.5% 與 4.3%；20,480：23.9% 與 21.3%（Fig. 5a，p.6）。**配對說明**：pdftotext 的圖文字層讀到這幾個數字與圖例「Strassen-square／AlphaTensor」，我依「AlphaTensor 勝過 Strassen-square」（p.5 正文）推定較大者屬 AlphaTensor〔部分＋判讀〕。精度（FP32？）本次讀的頁沒寫〔未查證〕，在 Supplementary。
- **跟 Tiara 的關係**〔判讀〕：prefill 的 GEMM 是 (T×4096)·(4096×4096 或 ×14336)，內維只有 4096，最多遞迴 1–2 層；最好情況 5–20%，而且沒有 tensor core 的證據、改變捨入。**不值得**。

### 2.7 子線性 top-k（MIPS／ANN）

#### RetrievalAttention（arXiv 2409.10516 v3；venue 〔未查證〕）
- **讀了什麼**：〔部分〕PDF p.1–5（§2.2–3.1、Fig. 2–3），查證 2026-10-27
- **關鍵事實**：Llama-3-8B、100K token 的 KV 檢索、連續生成 20 token：動態選 top-1000 平均回收 89% attention；固定用第一步選的 top-1000 降到 71%（Fig. 2，p.3–4）。q 與 k 分布不同（Mahalanobis 距離，Fig. 3b）：IVF 要掃約 30–50% key 才 recall>0.95，用 k 當查詢只要 1–5%（p.5）；它的索引用 prefill 的 q 建 q→k 連結（p.5–6）。只存取 1–3% key（摘要 p.1）。
- **條件**：注意力集中（上面的 89%）＋索引對 OOD 有處理。
- **與既有整理的關係**：repo 已有：`docs/stage3_20261019/lit/L1_eviction_selective_loading.md:413`（C7）、`L7d_recheck_lowrank_index.md:92、:280、:442`。本卡只補「集中性」與「OOD」兩個數字。

#### PQCache（PACMMOD／SIGMOD 2025 依 repo `L8a:112`；arXiv 2407.12820）
- 〔摘要〕prefill 時對 K 做 PQ（K-means 分子空間），decode 用 PQ 碼與中心找重要 token 再取 KV（摘要 p.1）。repo 已有：`docs/stage3_design_20261024/lit/L8a_progressive_query_aware_restore.md:112`。

### 2.8 混合精度模擬

#### Ootomo & Yokota：Recovering single precision accuracy from Tensor Cores while surpassing the FP32 theoretical peak（arXiv 2203.03341 v3）
- **讀了什麼**：〔摘要〕PDF p.1，查證 2026-10-27
- **關鍵事實**：把 FP32 拆成高低兩個半精度、用額外 GEMM 補償；A100 上 FP16 tensor core（限指數範圍）51 TFLOP/s、TF32（全 FP32 範圍）33 TFLOP/s，精度等於 FP32 SIMT，且超過 FP32 SIMT 峰值 19.5（摘要 p.1）。

#### Ozaki Scheme II（Ozaki, Uchino, Imamura；arXiv 2504.08009 v4）
- **讀了什麼**：〔摘要〕PDF p.1，查證 2026-10-27
- **關鍵事實**：用中國剩餘定理拆成多個 INT8 GEMM 模擬 FP64；GH200 上 56.6–80.2 TFLOPS，超過原生 FP64（摘要 p.1）。
- **〔算術〕對重算（prefill）**：目標精度是 BF16。H200 dense：BF16 989.5、FP8 1,979 TFLOPS（規格 ÷ 2）。用 FP8 拼出接近 BF16 的乘積至少要 A_hi·B_hi＋A_hi·B_lo＋A_lo·B_hi 三個 GEMM → 3／2＝1.5 個 BF16 GEMM 的時間；只留兩項也是 1.0 → **不會比 BF16 快**。這類技巧的方向是「用便宜的單元模擬貴的精度」，只在目標精度的原生單元很慢時（FP32 SIMT 67 vs TF32 494.5、FP64）才划算。
- **〔算術〕對寫入 SVD**：目前 FP32 SIMT（67 TF）。改 TF32（494.5 TF dense）理論 7.4 倍；要 FP32 等級精度則用 Ootomo 式 3×TF32 → 494.5／3≈165 TF，仍 2.5 倍於 FP32 SIMT。PyTorch 文件的 A100 例子：10240² matmul TF32 0.016 s vs FP32 0.11 s，相對誤差 0.0022 vs 0.000039（〔文件〕）。

---

## §3 哪些條件可以在 H200 上用既有 KV dump 驗證

**先講資料**（〔程式碼〕）：
- `code/m16_s1_kvdump.py`：`kv_bf16.u16`＝[n_blocks][L'][2][H][512][D]，K 是 **RoPE 後**、V 無 RoPE；128K 只存部分層（`code/m16_kvstruct.py` 檔頭「dump 的格式」）。RoPE 前 K 用 `m16_kvstruct.py` 的逆旋轉（有探針核對）。
- **沒有 Q**。問題 token 的 post-RoPE q 與「問題 token 對每個 context key 的機率」在 `code/m16_c1a_predict.py:285–316`（`qcap`，GPU forward 內取得，不落地）。**context 自己的 Q（prefill 的所有列）沒有任何地方存**——S1 要新寫 hook。
- 已量：K1（每 head 的譜、r90／r95／r99、校準→測試，`results_analysis.md:85–87`）、K1b（跨層子空間重疊，`:93–97`）、C1a（FULLK／QUEST 的 block 質量與 recall@k）、xKV 寫入時間（上述 wcost）。

| 編號 | 條件（對應 §1 列） | 要算的統計量（精確） | 理論需要的門檻 | 資料夠不夠 | repo 量過嗎 |
|:--|:--|:--|:--|:--|:--|
| S1 | 遠處 attention 子塊低秩（列 1） | 每（層, q head），prefill 的機率矩陣 P＝softmax(QKᵀ/√d)，切 b＝512 的 block；對距離級別 ℓ（|I−J|＝1, 2–3, 4–7, …）的非對角子塊 P[I,J] 算 ε-秩 r_ε＝#{σᵢ > ε·σ₁}（ε＝1e-2、1e-3）與相對 Frobenius 尾能量；另算「尖峰列比例」＝子塊中 max_j P_ij > 0.1 的列數／b。（P 與 exp(S) 只差左右對角縮放，非對角子塊的秩相同〔判讀〕） | H-matrix 有效需要 r_ε ≪ b（例如 ≤ b/16＝32）且隨 ℓ 不增；FMMformer 的 Fig. 3 量法是門檻 1e-6（p.4） | **不夠**：要 context 的 Q → 新 hook（每層 q_proj 輸出＋RoPE）；128K 的 P 不能整個存，要在 GPU 上逐子塊算 SVD 只存 r_ε | 沒量 |
| S2 | bounded entries（列 10） | 每（層, head）：‖K̃‖_∞＝max |K̃_{t,c}|（RoPE 後；可另報 RoPE 前與「去掉 sink token」版本）；‖q̃‖_∞（問題 token）；logit 最大值與「最大 − 第 2 大」及「最大 − 中位數」：(q̃·k̃)/√d | Alman–Song：B＝o(√ln n)：n＝131,072 → 3.43、262,144 → 3.53；集中到單一 token 需 logit 差 ≥ ln(n−1)≈11.8／12.5（agent A 的算術）。若實測 logit 差常 ≥ 12 → 正好在下界區 | K 夠（dump）；q 用 C1a 的 `qcap` 路徑（GPU） | ‖K‖_∞ 沒單獨量（K1 量的是能量比與平均方向，`m16_kvstruct.py` 檔頭 1.） |
| S3 | 注意力集中／溫度（列 2、3、13） | 對問題 token：每列熵 H(pᵢ)／ln n；top-k 質量 Σ_{top-k} p（k＝n 的 0.1%、1%、5%），在 token 與 block（512）兩種粒度；按（層, head）分布，標出 retrieval-like head（top-1 質量 > 0.5 的列比例） | Scatterbrain：低熵 → 稀疏夠、低秩不行（Thm 1，p.5）；參考值 RetrievalAttention top-1%≈89%（Llama-3-8B、100K，Fig. 2） | 夠（C1a 的機率加總已在 GPU 算；只需多輸出熵與 token 粒度 top-k） | 部分：C1a 有 block 粒度 FULLK 質量與 recall@k；熵沒量 |
| S4 | 隨機 SVD 的奇異值衰減（列 6） | xKV 每窗口 Z（RoPE 前 K、V，T×n）：精確譜（由 eigh(ZᵀZ)）σⱼ；在 repo 的秩 r（k22／k29／k34 → r＝r_he·W·H）算 σ_{r+1}/σ₁、尾能量 Σ_{j>r}σⱼ²／Σσⱼ²；svd_lowrank 在 niter＝0／1／2／4 時的 Frobenius 誤差 ÷ Eckart–Young 最佳 | HMT Thm 10.5：期望超出最佳 ≤ (1＋r/(p−1))^{1/2}；p＝16、r＝928 → ≤ 7.9 倍（無冪迭代的界〔算術〕：√(1＋928/15)）；冪迭代收斂到 σ_{r+1}（Cor. 10.10）。實用門檻（建議）：niter＝1 的誤差 ÷ 最佳 ≤ 1.01 | 夠（dump，CPU 或 GPU 都可；H200 上做） | 每 head 的譜有（K1）；**跨層窗口的譜與 niter 敏感度本次沒查到**（grep `niter` 只見程式設定） |
| S5 | Gram 路徑的數值條件（列 7、14） | 同一 Z：λ_r／λ₁、間隙 (λ_r − λ_{r+1})／λ₁；Gram 用 FP32 累加、TF32／BF16 輸入時，前 r 個特徵向量張成的子空間 vs FP64 參考的主角度（sinΘ 最大值）與**重建能量差** | 判讀門檻：間隙 ≫ u_in·√n（TF32 u_in≈2^-11、BF16 2^-8）；實用門檻：重建誤差差 ≤ 0.5% 相對能量 | 夠 | 沒量（`:cal` 用 eigh 但在校準題上、FP32） |
| S6 | 每題 vs 校準基底（列 6 的價值前提） | 已有：品質 UNDETERMINED（`RUNLOG_H200_ICPE.md:224`）。可補：同秩下每題 SVD vs `:cal` 的重建能量差的分布 | — | 夠 | 品質已量；能量差在 `results_analysis.md:274` 附近有探索性比較 |
| S7 | MIPS 的 OOD（列 13） | 問題 q 到 K 分布的 Mahalanobis 距離 vs K 到 K；只用 K 建的索引（IVF、k-means）在掃 1%／5%／30% 時對真 top-100 的 recall | RetrievalAttention：IVF 要 30–50% 才 >0.95（p.5） | 夠（K dump＋C1a q） | Quest 摘要不夠（C1b）；OOD 距離沒量 |
| S8 | key 空間多極補全（列 2） | 每（層, q head）：把 block（或 16-token 群）的 K 用中心 c 代表；問題 q 的輸出 o 的相對誤差：(a) 只精確算 top-k block、其餘丟掉；(b) top-k 精確＋其餘用中心（分數 exp(q·c/√d)·群大小、值用群平均 V）。在同樣的「搬運位元組」下比 (a)(b) | 若 (b) 的誤差在 f＝1/8 搬運下顯著小於 (a)，才值得做 A0 的變體 | 夠（K、V dump＋q） | 沒量（A0 是 ORACLE 選擇＋丟棄／縮圖，沒有中心補全） |
| S9 | Monarch 投影誤差（列 5，低優先） | ‖W−Π_𝓜(W)‖_F／‖W‖_F，W＝W_K、W_V、W_O，每層；Alg. 1（每批秩 1 SVD） | 無理論門檻；ProcrustesGPT 預期大 | 權重檔即可（H200 上 CPU） | 沒量 |

（S1–S9 都只是規格；**沒有在 Mac 上跑**。）

---

## §4 判讀：哪 1–3 個有現實的大加速路徑，閘門是什麼

1. **寫入端：串流 Gram＋tensor core＋小特徵分解（或至少 niter 4→1、開 TF32）**——條件已知成立（瘦矩陣、FP32 SIMT、高秩），〔算術〕下界從 0.89 s 到 0.036 s（Llama 32K），實測目前 5.3 s。
   - **但它不是還原加速**，而且 `:cal` 已把寫入降到 0.04–0.13 s。只在「每題基底的品質顯著好於 `:cal`」時才有意義。
   - 閘門 G-W（建議）：(a) S4、S5：Gram（TF32 輸入、FP32 累加）與 niter＝1 的重建能量 ÷ Eckart–Young 最佳 ≤ 1.01；(b) H200 32K 實測每題基底時間 ≤ 0.1 倍同長度 prefill（≤ 0.21 s，prefill 2.12 s 出自 `P1-1_xkv_int4_h200.md:141`）；(c) 前提：每題 vs `:cal` 的品質差 CI 下界 > 0。任一不過 → 不做，寫入繼續用 `:cal`。
2. **還原端：key 空間的多極近似（中心補全）接在縮圖還原上**——唯一不用訓練、條件有證據（注意力集中）、又和 Tiara 同形的 NLA 技巧。加速倍數的來源仍是「少搬」，NLA 的貢獻是讓同樣搬運量下更準，從而能搬更少。
   - 閘門 G-MP（建議）：先做 S3（集中性：熵、top-k）與 S8（同搬運位元組下，中心補全 vs 丟棄的輸出誤差）；只有 S8 在 f ≤ 1/8 時誤差顯著下降，才把「中心補全」加進 A0 的配置重跑生成品質（A0 的條件 2 差 0.05 pp，`results/RUNLOG_H200_S3.md:467`）。查新要先做：Multipole／Squeezed Attention 是 decode 加速，不是暫停後的分層還原——「還原時用中心補沒搬的 block」是否已有人做，本次沒查。
3. **讓重算次二次（H-matrix、低秩、多項式方法、Toeplitz／FFT、Strassen）——建議關閉，用一個便宜的負面閘門結案**：S2（‖K‖_∞ 與 logit 差）若如預期遠超 √ln n／ln n，就在文件裡把「次二次重算」標成條件不成立；加上 agent A 的 Amdahl 上限（128K 最多 3.1 倍、256K 5.3 倍，`LA_ssm_structure.md:314–315`）與 Alman–Yu 的下界，這條沒有現實路徑。

**漸近 vs 實測要分開看**：本檔列的「O(n²)→O(n)」全部只在訓練過的小模型或純理論上成立；有 wall-clock 的只有 Multipole（attention-only，A6000 最多 4.5 倍）、Strassen（V100 FP32 1.11–1.19 倍）、AlphaTensor（V100 4–24%）、Ootomo／Ozaki（模擬高精度）與 repo 自己的 SVD 寫入時間。

---

## 沒查到／沒做的

- 預訓練 Llama／Qwen 在 ≥32K 的 attention 遠處子塊秩：本次搜尋沒找到（查詢字串見 §2.1）。
- Strassen／AlphaTensor 在 Hopper tensor core 上的實測：本次搜尋沒找到（§2.6）。
- 不微調的 Monarch 投影在 Llama 上的準確度：本次搜尋沒找到（§2.4）。
- AlphaTensor Fig. 5 的精度設定與數字配對：〔未查證〕（在 Supplementary）。
- HiP Attention（arXiv 2406.09827，階層剪枝、不用訓練）、Hierarchical Sparse Plus Low Rank Compression（arXiv 2601.07839）只在搜尋結果看到標題與一句描述，沒讀 → 〔未查證〕；前者屬稀疏 prefill 估計（agent C 範圍）。
- Qwen2.5-14B 的層數 48、KV head 8：〔未查證〕（沒讀 config；只用於 §2.3 的一列算術）。
- Gram 的 n×n 特徵分解在 H200 上的實際時間：〔未量〕；§2.3 的 Gram 欄只含 GEMM。

---

## References

1. N. Halko, P.-G. Martinsson, J. A. Tropp, "Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions," *SIAM Review* 53(2), 2011. https://arxiv.org/abs/0909.4061
2. Z. Zhu, R. Soricut, "H-Transformer-1D: Fast one-dimensional hierarchical attention for sequences," ACL 2021. https://arxiv.org/abs/2107.11906
3. T. M. Nguyen *et al.*, "FMMformer: Efficient and flexible transformer via decomposed near-field and far-field attention," arXiv 2108.02347. https://arxiv.org/abs/2108.02347
4. Y. Kang, G. Tran, H. De Sterck, "Fast Multipole Attention," arXiv 2310.11960. https://arxiv.org/abs/2310.11960
5. C. Hooper *et al.*, "Multipole Attention for efficient long context reasoning," NeurIPS 2025. https://arxiv.org/abs/2506.13059
6. B. Chen, T. Dao *et al.*, "Scatterbrain: Unifying sparse and low-rank attention approximation," arXiv 2110.15343. https://arxiv.org/abs/2110.15343
7. S. Wang *et al.*, "Linformer: Self-attention with linear complexity," arXiv 2006.04768. https://arxiv.org/abs/2006.04768
8. Y. Xiong *et al.*, "Nyströmformer," arXiv 2102.03902. https://arxiv.org/abs/2102.03902
9. Y. Chen *et al.*, "Skyformer: Remodel self-attention with Gaussian kernel and Nyström method," arXiv 2111.00035. https://arxiv.org/abs/2111.00035
10. P. Kacham, V. Mirrokni, P. Zhong, "PolySketchFormer," arXiv 2310.01655. https://arxiv.org/abs/2310.01655
11. T. Dao *et al.*, "Monarch: Expressive structured matrices for efficient and accurate training," ICML 2022. https://arxiv.org/abs/2204.00595
12. T. Dao *et al.*, "Kaleidoscope: An efficient, learnable representation for all structured linear maps," arXiv 2012.14966. https://arxiv.org/abs/2012.14966
13. D. Y. Fu *et al.*, "Monarch Mixer," arXiv 2310.12109. https://arxiv.org/abs/2310.12109
14. E. Grishina, M. Gorbunov, M. Rakhuba, "ProcrustesGPT: Compressing LLMs with structured matrices and orthogonal transformations," arXiv 2506.02818. https://arxiv.org/abs/2506.02818
15. M. Ghashami, E. Liberty, J. M. Phillips, D. P. Woodruff, "Frequent Directions: Simple and deterministic matrix sketching," arXiv 1501.01711. https://arxiv.org/abs/1501.01711
16. E. Liberty, "Simple and deterministic matrix sketching," KDD 2013. https://arxiv.org/abs/1206.0594
17. M. Brand, "Fast low-rank modifications of the thin singular value decomposition," *Linear Algebra Appl.* 415, 2006; MERL TR2006-059. https://www.merl.com/publications/docs/TR2006-059.pdf
18. J. Alman, Z. Song, "Fast RoPE Attention: Combining the polynomial method and fast Fourier transform," arXiv 2505.11892. https://arxiv.org/abs/2505.11892
19. J. Alman, H. Yu, "Fundamental limitations on subquadratic alternatives to transformers," ICLR 2025. https://arxiv.org/abs/2410.04271
20. J. Alman, Z. Song, "The fine-grained complexity of gradient computation for training large language models," arXiv 2402.04497. https://arxiv.org/abs/2402.04497
21. Y. Cao *et al.*, "RoPE attention can be trained in almost linear time," arXiv 2412.17316. https://arxiv.org/abs/2412.17316
22. J. Su *et al.*, "RoFormer: Enhanced transformer with rotary position embedding," arXiv 2104.09864. https://arxiv.org/abs/2104.09864
23. W. Wu *et al.*, "Retrieval head mechanistically explains long-context factuality," arXiv 2404.15574. https://arxiv.org/abs/2404.15574
24. J. Huang, C. D. Yu, R. A. van de Geijn, "Implementing Strassen's algorithm with CUTLASS on NVIDIA Volta GPUs," arXiv 1808.07984. https://arxiv.org/abs/1808.07984
25. A. Fawzi *et al.*, "Discovering faster matrix multiplication algorithms with reinforcement learning," *Nature* 610, 2022. https://www.nature.com/articles/s41586-022-05172-4
26. "Guaranteed DGEMM accuracy while using reduced precision tensor cores through extensions of the Ozaki scheme," arXiv 2511.13778. https://arxiv.org/abs/2511.13778
27. D. Liu *et al.*, "RetrievalAttention: Accelerating long-context LLM inference via vector retrieval," arXiv 2409.10516. https://arxiv.org/abs/2409.10516
28. H. Zhang *et al.*, "PQCache: Product quantization-based KVCache for long context LLM inference," arXiv 2407.12820. https://arxiv.org/abs/2407.12820
29. H. Ootomo, R. Yokota, "Recovering single precision accuracy from Tensor Cores while surpassing the FP32 theoretical peak performance," arXiv 2203.03341. https://arxiv.org/abs/2203.03341
30. K. Ozaki, Y. Uchino, T. Imamura, "Ozaki Scheme II: A GEMM-oriented emulation of floating-point matrix multiplication using an integer modular technique," arXiv 2504.08009. https://arxiv.org/abs/2504.08009
31. NVIDIA H200 Tensor Core GPU 規格頁（查證 2026-10-27）. https://www.nvidia.com/en-us/data-center/h200/
32. PyTorch CUDA semantics notes（TF32 段，v2.14 頁，查證 2026-10-27）. https://docs.pytorch.org/docs/2.14/notes/cuda.html
33. PyTorch `torch/_lowrank.py`（v2.5.0 tag，查證 2026-10-27）. https://github.com/pytorch/pytorch/blob/v2.5.0/torch/_lowrank.py
