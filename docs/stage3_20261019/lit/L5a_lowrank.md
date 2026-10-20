# L5a　KV cache 的低秩／降維壓縮：第三階段文獻地圖

> **日期**：2026-10-20（UTC）
> **執行者**：文獻 agent（只讀、只上網；CPU 只用來把 PDF／HTML 轉成文字，`nice -n 19 taskset -c 0-3`。**沒有開任何 GPU 行程、沒有 import torch／vllm**，第二階段正在 GPU 上量時間）。只寫這一個檔。
> **範圍**：把「低秩／降維」加進第三階段動作空間之前要知道的事：哪些作法不用訓練、在 Llama-3.1-8B／Qwen2.5-7B 上有沒有同位元組的證據、重建要花多少、跟卸載／分層的關係、2.5 天內能補的缺口。跨層合併（MiniCache 一類）在 `L5b_merge_crosslayer.md`，卸載時壓縮在 `L5c_compress_on_offload.md`（兩檔與本檔同時撰寫，我沒讀完，只避免重複）。
> **先讀過的本專案檔案**：`research_map.md`、`lit/L1`（C3 InfiniGen、C5 ShadowKV、C18 YAKV）、`lit/L2`（B3 KVTC、D2 ShadowKV、D3 FreeKV）、`docs/PAPER_GUIDE.md`（ShadowKV、InfiniGen 卡）、`stage2_factcheck_and_toolbox.md` §3、`results/m10_longctx_mi300x/analysis/tables_longctx.md`、`results/RUNLOG_MI300X.md` 10.10／11.2f、`code/m10_codec.py`、`code/m10_codec_x.py`、`code/m7_model.py`、`code/m16_s1_kvdump.py`、`code/m16_kvstruct.py`（10/20 新寫、未 commit 的 K1 分析器，只讀了檔頭說明）。
> **檔案**：PDF 在 `/mlsteam/data/tiara/papers/stage3_L5/*.pdf`，arXiv HTML 在 `…/stage3_L5/html/`，ShadowKV 的圖在 `…/stage3_L5/fig/`；ShadowKV 的 PDF 用既有的 `/mlsteam/data/tiara/papers/shadowkv2025_arXiv2410.21465.pdf`。同目錄裡 `2405.14366.*` 等檔是 L5b agent 下載的。

**標記**
- 閱讀深度：〔全文〕＝PDF 轉文字後主文＋附錄逐段讀過（Palu、xKV、ShadowKV、LRQK）；〔部分〕＝讀了方法段與我指定的表／附錄（KVTC、Loki、Eigen Attention、ThinK、SVDq、LoRC、ReCalKV 讀 PDF；PuzzleKV、VarRate、KV-COBRA、OjaKV、S4R、JoLT 讀 arXiv HTML 的段落）；〔摘要〕＝只讀 arXiv API 回傳的摘要與 comment 欄。
- 來源：〔README〕＝官方 repo 的 README；〔程式碼〕＝本專案程式碼（附行號）；〔本專案實測〕＝本 repo 已有的量測（附檔案）；〔判讀〕＝我讀原文後的解讀；〔推論〕＝原文沒寫、我推的（含算術）；**NOT_FOUND**＝查了沒找到（≠不存在，CLAUDE.md 規則 7）；**NOT_MEASURED**＝沒人量過／我們沒量過。
- arXiv 編號都用 arXiv API 取回標題、作者、comment／journal_ref 核過；venue 只在 arXiv comment／journal_ref、會議官網、ACL Anthology 查得到時才寫，其他寫「venue 未查證」（§9）。
- 不同硬體、batch、context、壓縮比的數字**只並列、不排名**。
- 本檔是 **self-review**，不是 cross-model review（CLAUDE.md §5）。

---

## 0. 一頁結論

1. **低秩有三種做法，兩種不用訓練**〔判讀〕。
   - (a) **權重分解、快取 latent**：Palu、LoRC、ReCalKV、ASVD。離線對 W_K／W_V 做 SVD；V 的重建矩陣可吸收進 W_O，K 因為 RoPE 仍要「重建 → RoPE」。
   - (b) **直接投影 KV 活化值**：每個 prompt 線上 SVD（ShadowKV、xKV、SVDq、PuzzleKV、AttSVD），或離線校準 PCA（KVTC、Loki、Eigen Attention、VarRate、KV-COBRA）。
   - (c) **低秩當量化前的座標轉換**：SVDq、KVTC、KV-COBRA、JoLT（GEAR 反過來用低秩補量化誤差）。
   - GQA→MLA 的轉換（MHA2MLA、TransMLA、X-EcoMLA、MLAfication）與 MatryoshkaKV、EchoKV、BELT 都要訓練或蒸餾 → **本次不可行**。
2. **RoPE 是分水嶺；V 不適合大幅降秩**。
   - pre-RoPE 的 K 最低秩（ShadowKV 圖 1）〔全文〕。Loki：Llama-3-8B 若用 post-RoPE 的 PCA，困惑度從約 5 跳到 10 以上〔部分〕。KVTC 校準前先拿掉 RoPE〔部分〕。
   - V 的譜很平（ShadowKV 圖 1、KV-COBRA App. F、xKV D.3）。
   - 代價：K 不能吸收進權重，要「重建 → 套 RoPE」，Palu、ShadowKV 都為此寫了 kernel。
   - 唯一例外：每 32 token 一頁各自 SVD（PuzzleKV），用 post-RoPE；改用 pre-RoPE 只多 0.48 分〔部分〕。
3. **同位元組下，每層／每 head 的「只截斷秩」輸給量化；跨層、每 prompt 的 SVD（xKV）和 KIVI-2 打平或略贏；最穩的是「轉座標再分位元」**。
   - 例外先講：xKV 表 1（RULER 64K）〔全文〕：
     - Llama：xKV（8.03×）88.50，KIVI-2（7.1×）86.87。
     - 表 1 的 Qwen 列：90.19 vs 90.45。
     - LongBench（Llama）：42.27 vs 41.47。
     - 注意：它的 KIVI-2 用 128 一組（2.25 bit），和我們的 KIVI（32 一組、INT2 是 3 bit）不同。
     - 同一張表的單層 SVD（Single SVD，8.4×）只有 45.71 → 好處來自**跨層＋每 prompt**，不是低秩本身。
   - VarRate，LongBench（≤16K、20% 預算）〔部分〕：
     - Qwen2.5-7B-Instruct：均勻秩的低秩 30.50、KIVI-2（19%）46.82、不壓 48.79。
     - Llama-3.1-8B：43.47、44.82、46.01。
   - Palu 自己的表，Mistral-7B〔全文〕：只壓 2× 的 Palu-50% 得 36.23，比壓 5× 的 KIVI-2（41.45）還低。
   - KVTC App. B.9〔部分〕：純刪主成分到 16×，LITM 掉到 69.2（kvtc 99.4）。
   - 「轉到 SVD 座標再依奇異值分位元」才贏直接量化：SVDq 在 Qwen2.5-7B-Instruct、64K RULER（只壓 K）〔部分〕：
     - 3-bit 直接 per-channel：39.9。
     - SVDq 3-bit：69.7（不壓 70.6）。
   - 唯一在**我們同一個模型**上的低秩數字：S4R 表 2，Qwen2.5-7B-Instruct-1M、RULER 128K、壓 2.5×（位元組比 0.4，比 INT4-KIVI 的 0.3125 寬鬆）〔部分〕：
     - 每 prompt、跨層的 xKV 93.47、S4R 93.61，幾乎等於不壓（93.97）。
     - 權重分解的 Palu 82.22。
     - → 在 0.4 這個位元組點，「活化值 SVD」可行、「權重 SVD」不行；但沒有和 INT4 在同位元組比。
4. **我們的位置：INT4 位元組下沒有空間，INT2／K4V2 位元組下才有**。
   - 第二階段 INT4-KIVI（0.3125 位元組）在 32K–128K 兩個模型都不掉〔本專案實測〕。所以在這個位元組下，低秩最好也只是打平。
   - 有空間的是更低的位元組（`tables_longctx.md`）〔本專案實測〕：
     - INT2-KIVI（0.1875）：Llama 128K −7.2 個百分點；Qwen 32K −12.9、128K −12.5。
     - K4V2（0.25）：Qwen 32K −5.1。
   - → 同位元組測試放在 **0.25 與 0.1875**；0.3125 只當健全性檢查。
5. **「低秩＋卸載」只有 decode 時的稀疏取回與 host 存放；沒人把重建成本放進分層決策，也沒人量 GPU 忙**。
   - 已有的作法〔全文／部分〕：
     - ShadowKV：低秩 K 留 GPU、V 放 CPU，重建 K 與搬 V 重疊。
     - xK-SR：同 ShadowKV，K 改成跨層低秩。
     - LRQK：低秩只當索引，搬的是全精度 KV。
     - KVTC：接進 LMCache，存在 host memory。
   - 全部假設 GPU 只跑自己。
   - 唯一的多 client 端到端數字是 KVTC App. B.16：同一張 GPU 做壓縮與解壓，4–10 個 client 時 TTFT 比不壓**慢 32–42%**；要等 host memory 放不下才贏〔部分＋推論：算術〕。
   - 沒找到任何一篇把「重建是一個 GEMM、會和計算搶 GPU」放進分層或重算的決策。這和研究地圖的缺口 #1（干擾）同一條線，是可能的新意。
6. **在 MI300X 上、2.5 天內，只能自己寫**。
   - 官方碼都跑不了：ShadowKV、LRQK、Palu 量化路徑是 CUDA。Palu 發布版不支援 Llama-3.1〔README〕；它的融合 kernel 遇到 GQA 形狀會報錯（VarRate 附錄）〔部分〕。
   - KVTC、SVDq、PuzzleKV、VarRate、KV-COBRA 找不到官方碼（NOT_FOUND）。
   - 校準 PCA（每層每 KV head、pre-RoPE、去平均）或每 512-token block 的 SVD，都只要幾十行 PyTorch〔推論〕。
   - 已經有一半的量測工具：`code/m16_kvstruct.py`（K1，numpy-only）會量譜、有效秩、同一份 dump 內的「校準 → 測試」。
7. **建議（§8）**：
   - **E-L1**：K1 加四項（跨 prompt 基底、每 block SVD、同位元組對 KIVI 的誤差、跨層 CKA）。只用 CPU，GPU 0–0.2 h（S1 匯出已排）。
   - **E-L2**：同位元組答對率，32K，約 1 GPU 小時〔估計〕。E-L1 過閘才做。
   - **E-L3**：重建 vs 反量化在 GPU 忙時各慢多少，約 0.3 GPU 小時〔估計〕。
   - 停止條件：
     - E-L1：低秩誤差在 ≥80% 的（層, head）都大於同位元組量化誤差 → 不做 E-L2。
     - E-L2：0.25／0.1875 兩點都贏不了 KIVI（配對差 CI 下界 ≤0）→ 低秩不當獨立的儲存格式，只留「低秩 K 當還原時的選 block 索引」，併進 C1a。
     - E-L3：重建的忙時慢化和反量化差 <20% → 不另立成本項。

---

## 1. 已經有的，這裡不重寫

| 已有 | 在哪 | 本檔補什麼 |
|:--|:--|:--|
| ShadowKV 的機制、吞吐、CUDA-only | `PAPER_GUIDE.md`、L1 C5、L2 D2 | 全文細節：SVD 是**每層把所有 KV head 串起來**做（rank 160／1024）、跨 prompt 子空間相似度（讀圖）、prefill SVD 成本、decode 時重建 K 與取 V 的時間 |
| KVTC：PCA＋DP 量化＋DEFLATE、TTFT | L2 B3 | PCA 細節：校準 160K token、**跨 32 層 × 8 head 串成 32,768 維**、去 RoPE、純截斷 vs 量化（B.9）、只用單層 PCA（B.10）、每 prompt vs 一次校準（B.11）、LMCache 多 client（B.16） |
| InfiniGen（SVD 扭轉權重，只拿來選 token） | L1 C3、`PAPER_GUIDE.md` | 不重寫 |
| YAKV：低秩 K 當選擇器，在抽取型任務選錯 | L1 C18 | 只引用（§6.5） |
| FreeKV：ShadowKV 在 Qwen-2.5 重複輸出 | L2 D3 | 只引用（Qwen 的 K 不好壓） |
| Loki 只確認存在 | L1 參考表 | 補卡（§2 B2） |
| xKV 的跨層部分、MiniCache | L5b | 本檔只寫它的低秩與 SR／卸載 |
| 第二階段的量化格式與答對率 | `stage2_factcheck_and_toolbox.md` §3.4–3.5、`tables_longctx.md` | 只拿位元組比，以及 INT4／K4V2／INT2 的結果當同位元組對照（§5） |

---

## 2. 方法卡

每張卡的七個欄位：
- ① 低秩的是什麼（權重 or 活化值；K 在 RoPE 前或後；每 head／每組／每層／跨層）
- ② 要不要訓練；校準要多少資料
- ③ GQA；測過哪些模型
- ④ 原文的壓縮比與答對率；有沒有 ≥32K
- ⑤ 重建成本：在哪算、什麼 kernel、多久；有沒有和量化疊加
- ⑥ 有沒有和卸載／搬運結合
- ⑦ 程式碼、授權、ROCm 判讀

### A. 直接投影 KV 活化值、每個 prompt 線上 SVD

#### A1　ShadowKV（ICML 2025）〔全文〕

- **①** pre-RoPE K 的活化值。
  - 每層把 h_kv 個 KV head 串成 s×(h_kv·d) 後做 SVD（Alg. 1：A∈R^{s×r}、B∈R^{h_kv×r×d}）。所以是「每層、跨 head 聯合」。
  - rank 160：Llama 是 1,024 維，K 約壓 6.4 倍。
  - V 不壓，整份卸到 CPU。
  - post-RoPE K 每 8 個 token 取平均當 landmark；最不像平均的 48 個 chunk 當 outlier，留在 GPU。
- **②** 不訓練、不校準；每個 prompt 在 prefill 時做 SVD。
  - 理由：不同 prompt 的子空間不同。圖 1 中（rank-256 子空間相似度，我讀圖）〔全文〕：
    - 同一序列延長：每層都 ≥0.98。
    - 不同序列：第 0 層約 0.95，中層 0.82–0.86，深層多在 0.74–0.82。
- **③** GQA 有：Llama-3.1-8B（8 KV head）、GLM-4-9B-1M（4）、Yi-9B-200K（4）。Qwen2-7B-128K 只測 NIAH；**沒有 Qwen2.5**。
- **④** 數字〔全文〕：
  - GPU 上的 KV 約少 6 倍（App. A.2 的理論值 7.08×）。
  - Llama-3.1-8B，稀疏預算 1.56%：RULER 128K 85.53 → 83.57、LongBench 48.96 → 48.13。
  - 秩：到 160 左右之後持平（圖 9）。
  - 圖 5 左：pre-RoPE K 壓 6 倍以上 needle 不掉；post-RoPE K 和 V 要高得多的秩。
  - 權重與 KV 換成 FP8（e5m2）也維持（App. A.4）。
- **⑤** 重建成本（A100）〔全文〕：
  - decode 時只重建被選中的 chunk（Gather(A,I)·B，再套 RoPE），用融合的 CUDA kernel。
  - Llama-3-8B-1M 的一個 Transformer block（batch×context＝48×64K 到 6×512K）：重建 K 1.25–1.51 ms，和取 V 的 1.66–1.84 ms 重疊（表 13）。
  - prefill 的 SVD（表 12）：64K 每層 17.2 ms（佔 attention＋FFN 的 6.65%）；128K 26.6 ms（3.25%）。
  - 沒和量化疊加；SVDq 疊上去的結果見 A3。
- **⑥** **有卸載**〔全文〕：
  - V 放 CPU，K 的低秩留 GPU；取 V 與重建 K 用 CUDA multi-stream 重疊。
  - 等效頻寬式：eB＝2S·B_GPU ／ (S/C＋2(K+O)C＋(1−α)KC·B_GPU/B_PCIe)。A100 的例子是 7.2 TB/s。
  - 假設 GPU 只跑自己。
- **⑦** github.com/ByteDance-Seed/ShadowKV，Apache-2.0；flash-attn＋FlashInfer＋自寫 CUDA → **CUDA-only**（L2 D2）。
- **旁證**：
  - FreeKV 報 ShadowKV 在 Qwen-2.5 重複輸出（L2 D3）。
  - YAKV：rank 160 在抽取型任務選錯（L1 C18）。
  - xKV 表 5：把 ShadowKV 的 K 壓到 1.71×（GPU 省 11.6×）時，RULER 掉到 43.59〔全文〕。

#### A2　xKV（ICML 2026）〔全文〕

- **①** pre-RoPE K 與 V 的活化值，**跨層**。
  - 相鄰 W 層（預設 4）水平串接後做一次 SVD，得到共用的 token 基底 A_k（L×r），以及每層一個重建矩陣 B_ℓ（r×d，d＝H_kv·d_h）。也是跨 head 聯合。
  - 動機：相鄰層逐 token 的 cosine 低，但 CKA 高（主奇異向量對齊）。圖 2c：串越多層，達 95% 累積特徵值所需的秩比例越低。
- **②** 不訓練、不校準；prefill 時線上 SVD（torch.svd_lowrank，或自寫的隨機 SVD kernel，快 4 倍）。
  - 成本，表 3（Llama-3.1-8B，W＝4）：64K 0.57 s（prefill 9.97 s 的 5.72%）；128K 0.80 s（2.50%）。
  - **原文自己不一致**〔判讀〕：§3.2 寫「128K 約 3.9%」、C.1 寫「64K 3.17%」，都和表 3 對不上；C.1 文字寫 A6000、表標題寫 A100。數字以表為準。
- **③** GQA 有：Llama-3.1-8B-Instruct、Qwen2.5-7B-Instruct-1M（**和我們同一個模型**）、Qwen3-4B；附錄有 DeepSeek-Coder-V2-Lite（MLA）。
  - **又一處不一致**〔判讀〕：表 1 的 Qwen 列印成「Qwen2.5-14B-Instruct-1M」，正文寫 7B-1M（89.22%）。
- **④** 數字〔全文〕：
  - RULER 64K，Llama：不壓 91.89；KIVI-2（7.1×）86.87；SnapKV（8×）89.68；Single SVD（8.4×）**45.71**；xKV（8.03×）88.50。
  - RULER 64K，表 1 的 Qwen 列：不壓 93.36；KIVI-2（6.4×）90.45；Single SVD 71.79；xKV-4 90.19。
  - LongBench，Llama：43.20 → xKV 42.27、KIVI-2 41.47、Single SVD 33.22。
  - 窗口大小 1→2→4→8：45.71→75.15→88.50→88.91（表 8）。
  - 疊 RTN 4-bit：25.7×、87.64（表 9）。
  - 多輪 NIAH：5 輪都穩；SnapKV、PyramidKV 從第 2 輪開始掉。
- **⑤** 重建成本〔全文〕：
  - dense 重建每層 L·r·d FLOPs；attention kernel 速度 <1.0×，122K 約 0.4×（變成 compute-bound）。
  - 所以 decode 要配 Selective Reconstruction：沿用 ShadowKV 的 landmark 選 chunk，只重建選中的列。
  - xKV-SR 在 A100、122K 端到端是 FlashAttention-2 的 4.23 倍，比 ShadowKV 高約 30%。
  - 跨層共用基底的 r 較大，每層重建的內積維度跟著變大 → xK-SR 比 ShadowKV 略慢。
- **⑥** 兩個變體〔全文〕：
  - **xK-SR**：K 跨層低秩留 GPU、V 卸到 CPU（沿用 ShadowKV 的重疊）。
  - **xKV-SR**：K、V 都壓，全部留 GPU。
  - 結論是「壓得夠小就別卸載」：xKV-SR 的吞吐高於 xK-SR 和 ShadowKV。沒量 GPU 忙。
- **⑦** github.com/abdelfattah-lab/xKV，Apache-2.0；依賴 flash-attn 2.7.4、MInference（KIVI kernel 打補丁改成 Triton）、fast-hadamard-transform（選用）〔README〕。〔推論〕dense 的準確率路徑（torch.svd_lowrank）可在 ROCm 跑；SR 引擎沿用 ShadowKV 的 CUDA kernel，不行。

#### A3　SVDq（venue 未查證）〔部分〕

- **①** K 的活化值。
  - 圖 2 的流程是「重建 → RoPE → 內積」→ 壓的是 pre-RoPE K〔判讀〕。
  - 每層把所有 head 串接（Llama d＝1,024、Qwen2.5-7B d＝512），去平均後對 prefill 的 K 做 SVD。
  - latent channel 分 8 組，依奇異值給 8／4／2／1／0 bit。例：(8,4,4,4,2,2,0,0) 是平均 3 bit。
  - V 不壓（原文：V 的低秩性弱）。
- **②** 不訓練；每個 prompt 在 prefill 時做 SVD，要有整段 K。
- **③** GQA：Llama-3.1-8B-Instruct、Qwen2.5-3B／7B／14B-Instruct。
- **④** RULER 64K（只壓 K，V 留 BF16；表 2）：

  | 模型 | 不壓 | 直接 per-channel 3-bit | ThinK（3-bit 等價） | SVDq 3-bit | SVDq 2-bit |
  |:--|--:|--:|--:|--:|--:|
  | Llama-3.1-8B-Instruct | 90.0 | 79.9 | 77.2 | 89.7 | 86.1 |
  | **Qwen2.5-7B-Instruct** | **70.6** | **39.9** | 52.9 | **69.7** | 62.7 |

  - 疊 ShadowKV 的稀疏＋SVDq 1.25-bit：K 壓 410 倍（稀疏 32× 乘 12.8×），Llama 86.7。
  - Thm 4.1：誤差比直接 per-channel 量化小約一個數量級。
- **⑤** 重建＝反量化＋乘 V^H＋加平均。沒有 kernel、沒有時間數字（NOT_FOUND）。
- **⑥** 沒有卸載；只在疊 ShadowKV 稀疏的設定裡間接有 V 卸載。
- **⑦** 論文沒有程式碼連結（NOT_FOUND）。〔推論〕演算法簡單，自寫可在 ROCm 跑。

#### A4　PuzzleKV（venue 未查證）〔部分〕

- **①** K、V 的活化值。**每層每 KV head 切成 32-token 一頁，每頁各自截斷 SVD**。
  - 預設 (r_K, r_V)＝(16, 14)；儲存比 ρ＝(r_K＋r_V)(P＋d)／(2Pd)，約 0.6。
  - 第一頁（sink）與最近的頁保持 dense。
  - 用 post-RoPE K；改用 pre-RoPE 只從 90.40 到 90.88（Llama 16K，App. D.3）。
- **②** 不訓練、不校準。
- **③** GQA：Llama-3.1-8B-Instruct、Qwen3-8B。
- **④** 約 60% 儲存〔部分〕：
  - RULER 16K／32K：超過不壓的 96%。
  - LongBench，Llama：45.83 → 45.33（Global SVD 44.26、Palu 42.21、H2O 45.61）。
  - 因子再做 INT4：剩 18.7% 儲存，仍超過不壓的 93%。
  - 只測到 32K。
- **⑤** 不重建：自寫 mixed-attention 直接在因子上算 attention，再用 online softmax 合併。GH200 上：32K 的 TTFT 多 20.9%（分解在 prefill），TPOT 只多 0.075 ms。
- **⑥** 沒有。
- **⑦** 程式碼 NOT_FOUND。〔判讀〕「每頁是獨立的壓縮單位」和我們以 512-token block 為單位的還原最相容（§6.3）。

#### A5　其他每個 prompt 線上算的（短卡）

| 方法 | ① 低秩對象 | ② 訓練／校準 | ③ 模型 | ④ 原文數字 | ⑤ 重建 | ⑥ 卸載 | ⑦ 碼 | 深度 |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| **OjaKV**（Findings of ACL 2026） | K、V 投影到低秩子空間：先用校準集初始化，prefill 做完整 Oja 更新，decode 定期小更新；最前與最近的 token 保持全秩 | 不訓練；校準＋線上更新 | Llama-3.1-8B-Instruct（GQA）等 | 動機實驗：WikiText 校準的基底，同領域 RER 0.035，換到 MultiNews 變 0.255；Oja 更新後 0.097；SO 0.597 → 0.653。32K TTFT 2,102 → 2,801 ms，記憶體 16 → 11.6 GB（60% 壓縮） | 和 FlashAttention 相容（重建後算） | 沒有 | github.com/zzbright1998/OjaKV（授權未查）；README 要 CUDA GPU、flash_attn | 〔部分〕 |
| **S4R**（arXiv comment：AACL-IJCNLP 2026 Main） | 從抽樣的 token 建 K／V 基底（prompt-aware 初始化），decode 只稀疏重建 | 不訓練 | Llama-3.2-1B、Qwen3-4B、Llama-3.1-8B、Qwen2.5-14B；RULER 用 **Qwen2.5-7B-Instruct-1M**（和我們同一個） | 最多 5×、接近不壓。**它獨立重跑的基線**：(1) LongBench 表 1，Llama-3.1-8B、KV 壓到 20%：不壓 55.18、**Palu 3.78**、ShadowKV 50.11、xKV-4 54.19、S4R 53.03。(2) **RULER 128K、Qwen2.5-7B-Instruct-1M、2.5×（位元組比 0.4）**（表 2）：不壓 93.97、xKV 93.47、S4R 93.61、ShadowKV 89.61、SnapKV 88.81、**Palu 82.22**、StreamingLLM 48.39 | 稀疏重建。表 3（Qwen3-4B、120K、**沒有任何 kernel 最佳化**）：TTFT 不壓 24.7 s、Palu 24.6 s、S4R 27.7 s、**xKV 79.9 s**；512 token 延遲 54.9 s vs xKV 260.8 s | 沒有（表 3 的 ShadowKV 刻意把 V 留在 GPU） | github.com/Draconis98/S4R（授權未查） | 〔部分〕 |
| **AttSVD** | 每個 prompt 的截斷 SVD，只留 attention 會讀的方向；每 head | 不訓練 | 「多個模型」 | LongBench 全套＋agentic benchmark，用到 50% KV 記憶體時和不壓相當 | NOT_FOUND | 沒有 | NOT_FOUND | 〔摘要〕 |
| **KQ-SVD** | 直接對 attention 矩陣做最佳低秩分解（閉式解） | 不訓練 | LLaMA、Mistral | 只報「投影品質」 | NOT_FOUND | 沒有 | NOT_FOUND | 〔摘要〕 |

### B. 校準 PCA（活化值、離線基底）

#### B1　KVTC（ICLR 2026）〔部分；機制見 L2 B3〕

- **①** K（**先拿掉 RoPE**）與 V 的活化值，**一次校準**。
  - 取校準集的 token，把 l 層 × h head 的向量串成 p＝l·h·d 維（Llama 3.1 8B 是 32×8×128＝32,768 維）。
  - 去平均後做隨機 SVD（截到 10K 維）；K、V 各一個基底。
  - 最前 4 個 sink 與最近 128 個 token 不壓。
- **②** 不訓練〔部分〕。
  - 校準 160K token（Qwen 200K）：FineWeb 與 OpenR1Math 各半，1–8K 與 8–32K 的文件各半；12B 模型在 H100 上 10 分鐘內。
  - 基底不小：Llama 3.1 8B 是 3.28 億參數，等於模型的 8.7%（表 21）。
- **③** GQA：Llama 3.1 8B、Llama 3.3 70B、Mistral NeMo 12B、MN-Minitron 8B、R1-distilled Qwen 2.5 1.5B／7B。**不是** Qwen2.5-7B-Instruct-1M。
- **④** 數字〔部分〕：
  - 表 20（LongBench／RULER 子集，Llama 3.1 8B 平均）：不壓 59.9；kvtc 16–20× 59.2、32× 58.5、64× 57.6；對照 KIVI（5×）58.8、xKV（4–6×）55.7。
  - **只刪主成分、不做量化**（-DPQ，視窗 16；B.9）：

    | 壓縮倍數 | 任務 | kvtc | -DPQ |
    |:--|:--|--:|--:|
    | 8× | RULER-VT | 98.2 | 93.0 |
    | 16× | LITM | 99.4 | 69.2 |
    | 64× | LITM | 80.0 | 5.1 |

  - **只用 1 層做 PCA**（-DPQ，8×；B.10）：GSM8K 27.5、LITM 49.1、RULER-VT 70.8；串 32 層是 56.9／98.6／93.0。
  - **每個 prompt 自己算基底**（B.11）：要存基底，壓縮比只剩 1.0–12.4×；模擬多輪對話時 GSM8K 56.8 → 31.4。
- **⑤** 解壓＝反量化＋乘 V^T；可以逐層用 V^T 的子矩陣，讓生成提早開始。
  - H100、BS 8、8K：壓 379 ms、解 267 ms（L2 B3）。
  - 熵編碼用 nvCOMP DEFLATE（CUDA）。
- **⑥** **有**：目標就是存放／傳輸（GPU 內、host memory、卸載儲存）。
  - B.16 的設定：vLLM＋LMCache；Llama 3.3 70B FP8、2×H100 TP；host 每張 GPU 128 GiB；context 約 64K。
  - 1–2 個 client：一樣。
  - 4 個 client：TTFT 1.2 → 1.7 s。
  - 10 個 client：3.4 → 4.5 s。原文：同一張 GPU 做 prefill、生成、壓縮、解壓「can result in increased latency」。
  - 12 個 client 以上：不壓的那組 host memory 不夠，開始重算（136.6 vs 5.6 s）〔部分〕。
- **⑦** 官方碼 NOT_FOUND（L2 B3：第三方 OnlyTerp/kvtc NOT_VERIFIED）。DEFLATE 段 CUDA-only；〔推論〕PCA 與量化可移植。

#### B2　Loki（NeurIPS 2024）〔部分〕

- **①** K 的活化值；**每層每 head** 用校準集（WikiText-103）做 PCA。
  - pre-RoPE、post-RoPE 兩種基底都試，每個模型挑好的那個。
  - 存全部主成分（**不省記憶體**）；只用前 d（25–50%）維算近似分數、選 top-k（12.5–25%）token，再用全維算精確 attention。
- **②** 不訓練；校準換成 C4、BookCorpus 結果基本一樣。
- **③** GQA：Llama-3-8B／70B、Mistral、Mixtral。
- **④** 數字〔部分〕：
  - Rank@90（解釋 90% 變異所需的秩）平均約 80／128。pre-RoPE 比 post-RoPE 低。三個校準集一致。
  - Llama-3-8B、Mistral-7B 用 post-RoPE 基底：困惑度約 5 → 10 以上。pre-RoPE 沒有這問題。
  - 長上下文：ShadowKV 在 RULER 128K 重跑的 Loki（Llama-3.1-8B）只有 35.52，不壓 85.53〔全文，ShadowKV 表 1〕。
- **⑤** Triton kernel；Llama2-13B 的 attention 最多快 45%。
- **⑥** 沒有卸載（本來就不省記憶體）。
- **⑦** github.com/hpcgroup/loki，MIT；Triton kernel，綁特定 transformers 版本〔README〕。〔推論〕Triton 有機會在 ROCm 跑；但它不壓縮，對我們只有「PCA 指標」的價值。

#### B3　Eigen Attention（Findings of EMNLP 2024）〔部分〕

- **①** Q、K 共用一個基底，V 一個基底。
  - 每層每 head：校準（WikiText 子集）的表示矩陣做 SVD，依門檻 ε_th 取秩。
  - 投影吸收進 W_Q／W_K／W_V／W_O。
  - 有 RoPE 的模型：query 保持全維；K 存低維，attention 前用 (U^K)^T 還原再套 RoPE；U^K 跨 head 共用。
  - 逐層門檻用輸出誤差預算來搜尋。
- **②** 不訓練（另有 LoRA 微調版）；要校準。
- **③** GQA：Llama-2-70B、Llama-3-8B／70B；另有 OPT、MPT。
- **④** 只有短任務。Llama-3-8B，KV 剩 0.6×：WikiText 困惑度 6.14 → 8.51，zero-shot 平均 0.69 → 0.62；LoRA 微調後困惑度 7.60〔部分〕。沒有 ≥32K。
- **⑤** attention 延遲最多降 60%（原文摘要）。
- **⑥** 沒有。
- **⑦** github.com/UtkarshSaxena1/EigenAttn；**GitHub 沒有標授權**（API 回 license＝None）。

#### B4　VarRate（venue 未查證）〔部分〕

- **①** 每層一個校準基底。
  - 把 K 反旋轉成 pre-RoPE，和 V 串成每個 token 一個向量 z＝[k̃; v]（D＝2·G·d_h，所有 head 共用一個基底）。
  - 每隔 stride 個 token 留一個 anchor 不壓；其他 token 用殘差在基底上編碼。
  - **依 query 顯著度給每個 token 不同的秩**（下限 16）。
- **②** 不訓練；每層基底用 6 個無標註的 LongBench context 校準一次。換三組校準集，結果穩定（A.8）。
- **③** GQA：Llama-3.1-8B-Instruct、Qwen2.5-7B-Instruct（另有 Mistral-7B）。
- **④** **同記憶體比較**（表 1：20% 預算，LongBench 16 個任務）〔部分〕：

  | 模型 | 不壓 | VarRate（20%） | KIVI-2（19%） | 均勻秩 flat（20%） | KIVI-4（31%，不是同記憶體） |
  |:--|--:|--:|--:|--:|--:|
  | Llama-3.1-8B-Instruct | 46.01 | 45.69 | 44.82 | 43.47 | 45.94 |
  | **Qwen2.5-7B-Instruct** | 48.79 | 48.00 | 46.82 | **30.50** | 48.98 |

  - 上下文到 16K（A.21 另測更長，我沒讀）。
- **⑤** 在 KVPress 裡「重建後放回 cache」；decode 速度不變（37.4 vs 37.5 tok/s）；要省記憶體得靠融合 kernel（沒寫）。prefill 成本實測 +2.6% 到 +41.8%（依任務）。
- **⑥** 沒有。
- **⑦** 程式碼 NOT_FOUND。
- **副產品**：它重現 Palu 在 GQA 模型上的延遲時，Palu 的融合 kernel 假設 MHA 的權重形狀（num_heads×rank×head_dim），遇到 Llama-3.1-8B 的 GQA 形狀直接報錯；用 MHA 設定才重現到 attention 快 1.28×（8K）〔部分〕。

#### B5　KV-COBRA（venue 未查證）〔部分〕

- **①** 校準時對每個 head 的 K 做 SVD（附錄另有 post-RoPE 的實驗）。
  - **每個 head 自己決定秩 r 與位元 b**：讓截斷與量化的邊際失真相等；再跨 head 重新分配預算。
  - 用 Hadamard 旋轉把通道變異拉平。
- **②** 不訓練；要校準。
- **③** GQA：Llama-3.1-8B、Mistral-7B-v0.3、Qwen2.5-7B-Instruct。
- **④** 數字〔部分〕：
  - 0.5–2 bpd 時，贏所有 SVD／旋轉類方法。
  - **均勻 2 bpd 下，每個 head 的失真跨 256 個 KV head 差 100 倍**（圖 1b）。
  - **V 的譜平：截秩對 V 幾乎沒好處，V 只做位元分配**（App. F）。
  - RULER（App. J）我沒讀。
- **⑤** 用同一類 SVD／旋轉 kernel；原文說沒有逐 token 的額外成本。
- **⑥** 沒有。
- **⑦** NOT_FOUND。〔判讀〕「每個 head 很不一樣」正是 E-L1 要量的東西。

#### B6　ECKVH（Effectively Compress KV Heads；venue 未查證）〔摘要〕

用校準 PCA 把 MHA 的 head 壓成 GQA，對 RoPE 的 K 另有處理。Palu 說它需要 LoRA 微調。**我們的兩個模型本來就是 GQA → 關係小。**

### C. 權重分解、快取 latent

#### C1　Palu（ICLR 2025）〔全文〕

- **①** 把 W_K、W_V 做 SVD（W≈AB），快取 latent h＝xA。
  - **分組**：每 s 個 head 一起分解（G-LRD，預設 s＝4），介於每 head 各自（M-LRD）與全部一起（J-LRD）之間。
  - V 的 B 吸收進 W_O。K 因為 RoPE 不能吸收進 W_Q，要「重建 → RoPE → 內積」，三步融合在一個 Triton kernel。
  - 用 Fisher 資訊分配各層各矩陣的秩。
  - Walsh-Hadamard 旋轉吸收進 A、B，壓掉 SVD 造成的 latent outlier，讓 latent 能量化到 2–3 bit。
- **②** 不訓練（LoRA 是選用）；要校準〔全文〕。
  - Fisher：WikiText-2，2,048 條 × 1,024 token。
  - SVD 用 SVD-LLM 的 truncation-aware 版本，需要活化值做白化。
- **③** GQA：Llama-3-8B-Instruct、Mistral-7B-v0.2。MHA：Llama-2、LongChat。
- **④** 數字〔全文〕：
  - 50% 低秩（G-LRD），Llama-3-8B-Inst：WikiText 困惑度 8.28 → 10.11，zero-shot 67.99 → 64.45。J-LRD 9.12／66.19；M-LRD 12.38／58.40。
  - LongBench，Mistral-7B-v0.2（最長 31.5K）：

    | 設定 | 壓縮倍數 | 平均 | Synthetic |
    |:--|--:|--:|--:|
    | 不壓 | 1× | 42.54 | 44.87 |
    | Palu-30% | 1.43× | 41.55 | — |
    | Palu-50% | 2× | **36.23** | **16.88** |
    | KIVI-2（3.16 bit） | 5.05× | **41.45** | — |
    | Palu-30%＋3 bit | 7.59× | 40.77 | — |

- **⑤** RTX 4090、Llama-2-7B（MHA）〔全文〕：
  - RoPE attention 在 64K 快 1.89×（50% 低秩）；V 再量化成 4-bit 快 2.91×。
  - 重建 kernel 在 gs-4 最多 1.95×；**超過 16K 後加速變小，因為重建變成 compute-bound**。
- **⑥** 沒有。
- **⑦** github.com/shadowpa0327/Palu，MIT。
  - README 的待辦還沒勾「升級 transformers ≥4.43.3 以支援 Llama3.1」→ 發布版不支援 Llama-3.1〔README〕。
  - 量化路徑要 fast-hadamard-transform（CUDA）。
  - 第三方重跑：VarRate 回報融合 kernel 遇 GQA 形狀報錯〔部分〕；S4R 在 η＝0.2 時 Llama-3.1-8B LongBench 只剩 3.78（不壓 55.18）〔部分〕。
  - 〔推論〕本次不建議跑 Palu 本身。

#### C2　LoRC（venue 未查證）〔部分〕

- **①** W_K、W_V 的 SVD，快取 latent。依「累積條件數」逐層決定壓縮維度：淺層留較多；條件數過高的層不壓。對 RoPE 有調整（App. C）。
- **②** 不訓練、不需要任務資料（只對權重做 SVD）。
- **③** GQA：Llama-3-8B-Instruct、Llama-3-70B-Instruct。MHA：Llama-2-13B。
- **④** 只有短任務（BoolQ、XSum、OpenBookQA、GSM8K），KV 預算掃 55–100%。
  - 原文說「60% 壓縮比」時多數資料集幾乎不掉；但圖的 x 軸是「保留比例」，**無法確定是保留 60% 還是壓掉 60%**〔部分〕。
  - 沒有長上下文。
- **⑤** NOT_FOUND（沒有 kernel、沒有時間）。
- **⑥** 沒有。
- **⑦** 程式碼 NOT_FOUND。

#### C3　ReCalKV（venue 未查證）〔部分〕

- **①** K：用 CKA 把相似的 head 重排成組，再分組 SVD（HSR，組大小 4）。V：對 W_V 做 SVD 後用校準資料離線修正，右因子吸收進 W_O（OVC）。
- **②** 不訓練；校準 256 條 WikiText。
- **③** GQA：Mistral-7B-Instruct-v0.2。MHA：Llama-2-7B／13B、LongChat-7B。Llama-3.1 的結果在補充材料（我沒讀）。
- **④** LongBench，Mistral-7B-v0.2（不壓 49.24）〔部分〕：

  | 壓縮 | Palu | ReCalKV |
  |:--|--:|--:|
  | 50% | 44.96 | 45.57 |
  | 60% | 38.39 | 41.19 |
  | 70% | 27.06 | 29.82 |

  **只壓 2 倍就掉 3.7 分。**
- **⑤** NOT_FOUND。
- **⑥** 沒有。
- **⑦** github.com/XIANGLONGYAN/ReCalKV，Apache-2.0（GitHub API）。

#### C4　ASVD（venue 未查證）〔摘要〕

activation-aware 的權重 SVD；摘要稱不訓練就能把 KV cache 減 50% 而不掉效能。沒有長上下文或 GQA 的資訊。

### D. 低秩當索引，搭配卸載

#### D1　LRQK（NeurIPS 2025）〔全文〕

- **①** **不是壓縮 KV 的存放**。
  - prefill 時把（每 head 的）Q、K 聯合分解成 rank-r 因子：交替更新的閉式解，迭代 2 次。
  - decode 時用 r 維的 proxy 分數挑 top-k token。
  - **全精度 KV 放 CPU**；GPU 只放「選中的 top-k＋最近 64 個」，hit/miss 機制只搬缺的。
  - 〔判讀〕Q、K 從 HF 的 attention 拿，應是 post-RoPE（原文沒明寫）。
- **②** 不訓練、不校準；但超參數（r、top-k、lite token 數）要依任務調（原文自承）。
- **③** GQA：Llama-3-8B-1M、Llama-3.1-8B、Qwen2.5-7B-Instruct、Mistral-7B、Qwen2.5-14B／32B。
- **④** 數字〔全文，表 5、6〕：
  - RULER 128K，Llama-3-8B-1M（r＝32、top-2048）：多數任務接近不壓，但 S1 81.00、MK2 42.00（不壓 100／98.96）。
  - **Llama-3.1-8B LongBench**（r＝32、top-1024）：LCC 67.31 → 24.60、SAMSum 29.84 → 12.02。
  - **Qwen2.5-7B RULER VT**：8K 97.4 → 70.0；16K 98.8 → 64.2。
- **⑤** 每一步多一次 O(lr) 的 proxy 分數，加上幾個 r×r 的小反矩陣。原文 Limitations：主要瓶頸是 **CPU 端的 index 操作**，不是 PCIe。
- **⑥** **有卸載**〔全文〕：
  - CPU 放全 KV、只搬 miss；miss 率平均約 0.40，傳輸省約 60%。
  - A100-40G、HF：64K decode 6.80 tok/s；全部卸到 CPU 0.50 tok/s；全放 GPU OOM。
  - RTX 3090、32K 摘要：LRQK 489–646 tok/s；ShadowKV 1,083.55；vanilla（兩張卡）1,775.70。
- **⑦** github.com/tenghuilee/LRQK，MIT；要編 C++/CUDA plugin〔README〕。〔推論〕ROCm 要 hipify，本次不做。

#### D2　SALS（venue 未查證）〔摘要〕

- **①** KV 投影到低秩的 latent；在 latent 裡用「不含 RoPE 的 Q·K」選 token，只重建選中的少數。兩個觀察：RoPE 讓 K 的變異變大、秩變高；轉進 latent 後多數層的表示穩定。
- **②** 投影從哪來（權重或校準）：摘要沒寫，我沒讀全文。
- **③** Llama-2-7B-chat、Mistral-7B；RULER-128K 用 Llama-3.1-8B-Instruct 驗證。
- **④** KV 壓 6.4 倍；attention 運算子 4K 時是 FlashAttention-2 的 5.7 倍；端到端 4K 1.4×、32K 4.5×（對 gpt-fast）。
- **⑤** 只重建選中的 token。
- **⑥** 沒有。
- **⑦** NOT_FOUND（論文只連到 gpt-fast）。

### E. 低秩＋量化疊加（SVDq、KVTC、KV-COBRA 已在上面）

| 方法 | ① | ② | ③ | ④ | ⑤ | ⑥ | ⑦ | 深度 |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| **GEAR**（venue 未查證） | 先量化，再用低秩矩陣近似量化誤差、稀疏矩陣補 outlier（低秩是補誤差，不是存放格式） | 不訓練 | 摘要沒列 | 4-bit 近乎無損，吞吐最多 2.38×、峰值記憶體最多少 2.29×；KVTC 表 20 的 GEAR（5×）在 Llama 3.1 8B 59.7（不壓 59.9） | NOT_FOUND | 沒有 | github.com/HaoKang-Timmy/GEAR（授權未查） | 〔摘要〕 |
| **JoLT**（arXiv comment：Under review at ICLR 2027） | prefill cache 當四階張量，沿 token 與 feature 做 partial Tucker（head、layer 不動）；截斷殘差用旋轉後的低 bit 量化；一個 Lagrangian 在總位元組限制下分配各組的秩與殘差位元 | 不訓練 | Mistral-7B-v0.3、Llama-2-13B、Llama-3.1-8B、Qwen2.5-14B、OLMoE | 2–3× 困惑度 <0.2%；RULER 64K、Llama-3.1-8B：3× 內近乎無損，4× −0.90、5× −2.40 個百分點 | 融合 Triton decode kernel，直接在因子上算 attention，不展開 | 沒有 | github.com/rahulk98/JoLT-Master-Thesis（授權未查） | 〔部分〕 |

### F. 降維（選通道、剪枝）

#### F1　ThinK（ICLR 2025 Spotlight）〔部分〕

- **①** 不是低秩分解，而是**依 query 剪 K 的通道**。
  - 每個 head 留 (1−λ)D 個通道；分數＝‖Q[−S_obs:, j] K[:, j]^T‖_F，Q 用 prompt 最後一段（像 SnapKV 的觀察窗）。
  - 存剪過的 K 加一個二元遮罩。
  - 〔判讀〕分數用 attention 的 Q、K，所以剪的是 post-RoPE K。
- **②** 不訓練、不校準；但**依賴當下的 query**。
- **③** GQA：Llama-3-8B／70B-Instruct、Mistral-7B。
- **④** 數字〔部分〕：
  - LongBench：疊在 SnapKV（KV-size 2048）上再剪 40% 的 K 通道，41.58 → 41.91；整體 KV 記憶體再少 20% 以上。疊 KIVI：峰值記憶體少 2.8 倍。
  - 第三方（SVDq 表 2）：3-bit 等價時 Qwen2.5-7B 64K 只有 52.9（不壓 70.6）。
- **⑤** 不用重建：剪掉的通道對應的 query 通道一起拿掉。
- **⑥** 沒有。
- **⑦** github.com/SalesforceAIResearch/ThinK，Apache-2.0；KIVI 版的 README 寫「目前只支援 LLaMA-2」〔README〕。
- 〔推論〕依第一輪的 query 決定剪哪些通道，和「寫入時依問題刪 token」是同一類風險：第 2 輪問別的事可能失準（L1 §4 的多輪 eviction 問題）。

#### F2　RAP（venue 未查證）〔摘要〕

剪 W_K、W_V 的通道，但以 RoPE 配對為單位（整對拿掉，不破壞旋轉）。Llama／Mistral／Qwen 3B–14B，保留 70% 時維持準確率。程式碼未查。

### G. 要訓練或蒸餾 → 本次不可行

| 方法 | 做什麼 | 要多少訓練 | 和我們的關係 | 深度 |
|:--|:--|:--|:--|:--|
| **MHA2MLA**（ACL 2025） | partial-RoPE（拿掉貢獻小的維度的 RoPE）＋用預訓練權重做 K、V 聯合 SVD 初始化 | 微調，用預訓練資料的 0.3–0.6% | Llama2-7B KV 少 92.19%，LongBench 掉 0.5% | 〔摘要〕 |
| **TransMLA**（NeurIPS 2025；會議版標題 *Migrating GQA Models to MLA with Full DeepSeek Compatibility and Speedup*） | GQA → MLA，可直接用 DeepSeek 的程式碼與 vLLM／SGLang | 微調 6B token 才恢復 | LLaMA-2-7B KV 壓 93%，8K 時快 10.6 倍 | 〔摘要〕 |
| **X-EcoMLA**（venue 未查證） | 蒸餾 upcycling 成混合 MLA | Llama3.2-1B：3.6B token、**AMD MI300 上 70 GPU 小時**（6.4×）；10.6× 要 7B token、140 GPU 小時 | 證明 MLA 轉換能在 MI300 上訓練，但 1B 模型就要 70 小時 | 〔摘要〕 |
| **MLAfication**（venue 未查證） | MHA2MLA 的延伸，支援 QKNorm、參數高效蒸餾 | 蒸餾 | Qwen3-30B-A3B 吞吐 5.35×；疊 4-bit KV 量化，KV 少 94.73% | 〔摘要〕 |
| **MatryoshkaKV**（venue 未查證） | 可訓練的正交投影（蒸餾目標） | 訓練 | 摘要：直接用 PCA 投影時，在它們的設定下「significant performance degradation」 | 〔摘要〕 |
| **EchoKV**（venue 未查證） | 輕量網路從部分 KV 重建被丟掉的部分 | 兩階段微調，7B 模型在一張 A100 上幾分鐘 | 最接近可行，但要另寫訓練與推論路徑，期限內不做 | 〔摘要〕 |
| **BELT**（venue 未查證） | 對齊 attention 分布的閉式低秩＋學習出來的秩分配器 | 訓練分配器 | 最多 70% 低秩壓縮 | 〔摘要〕 |
| **CSKV**（arXiv comment：NeurIPS ENLSP-IV 2024 workshop） | 通道縮減 | training-efficient 的微調 | — | 〔摘要〕 |
| **LRKV**（venue 未查證） | 共用全秩 KV＋每 head 低秩殘差 | 從頭預訓練 | — | 〔摘要〕 |

---

## 3. 總表

「L／Q」＝有沒有測過 Llama-3.x-8B／Qwen2.5-7B（「Q*」＝Qwen2.5-7B-Instruct-1M，和我們同一個模型）。

| # | 方法（venue） | 低秩對象／粒度 | K 的 RoPE | 訓練／校準 | L／Q | ≥32K | 重建／kernel | 卸載 | 碼／ROCm | 深度 |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| A1 | ShadowKV（ICML'25） | 活化值，每層跨 head，只壓 K | pre | 每 prompt SVD | L／— | 128K | 只重建選中的，CUDA | **V 放 CPU** | Apache-2.0，CUDA-only | 全文 |
| A2 | xKV（ICML'26） | 活化值，**跨 4 層**＋跨 head，K、V | pre | 每 prompt SVD | L／Q* | 64K | dense 慢（0.4×），要配 SR | xK-SR：V 放 CPU | Apache-2.0；dense 路徑〔推論〕可移植 | 全文 |
| A3 | SVDq | 活化值，每層跨 head，只壓 K，＋混合位元 | pre | 每 prompt SVD | L／Q | 64K | 反量化＋乘基底，無 kernel | 無 | NOT_FOUND | 部分 |
| A4 | PuzzleKV | 活化值，**每 32-token 頁**、每 head | post | 不需要 | L／— | 32K | 直接在因子上算 attention | 無 | NOT_FOUND | 部分 |
| A5 | OjaKV（Findings ACL'26） | 活化值，線上 PCA | — | 校準＋線上更新 | L／— | 有 | 重建後算 | 無 | 有碼，CUDA | 部分 |
| A5 | S4R（AACL'26，comment） | 活化值，抽樣 token 建基底 | — | 不訓練 | L／Q* | 128K（Q*，2.5×） | 稀疏重建 | 無 | 有碼 | 部分 |
| B1 | KVTC（ICLR'26） | 活化值，**跨全部層×head**，K、V，＋DP 位元＋DEFLATE | 去 RoPE | 校準 160K token | L／— | RULER 子集 | 解壓 267 ms（8K×8，H100） | **存 host／儲存** | 官方 NOT_FOUND | 部分 |
| B2 | Loki（NeurIPS'24） | 活化值，每 head，只拿來選 token | pre／post 擇一 | 校準 | L／— | 128K（ShadowKV 重跑） | Triton | 無（不省記憶體） | MIT，Triton | 部分 |
| B3 | Eigen Attention（Findings EMNLP'24） | 校準基底吸收進權重，每 head | pre | 校準 | L／— | 無 | 還原後 RoPE | 無 | 無授權欄 | 部分 |
| B4 | VarRate | 活化值，每層 K⊕V 聯合，每 token 不同秩 | pre | 6 個 context 校準 | L／Q | ≤16K（主表） | 重建後放回 | 無 | NOT_FOUND | 部分 |
| B5 | KV-COBRA | 每 head 的秩×位元分配 | pre（附錄有 post） | 校準 | L／Q | 有（App. J，未讀） | — | 無 | NOT_FOUND | 部分 |
| C1 | Palu（ICLR'25） | **權重**，每 4 head 一組 | 重建後 RoPE | Fisher 校準 | L(3)／— | ≤31.5K | Triton＋CUDA | 無 | MIT；不支援 3.1、GQA kernel 報錯 | 全文 |
| C2 | LoRC | 權重，逐層 | 有調整 | 不需要 | L(3)／— | 無 | — | 無 | NOT_FOUND | 部分 |
| C3 | ReCalKV | 權重，head 重排分組 | — | 256 條校準 | —（3.1 在補充）／— | ≤32K | — | 無 | Apache-2.0 | 部分 |
| D1 | LRQK（NeurIPS'25） | Q、K 聯合低秩，**只當索引** | post〔判讀〕 | 不需要 | L／Q | 128K | proxy 分數 | **KV 放 CPU，只搬 miss** | MIT，C++/CUDA | 全文 |
| D2 | SALS | latent 中不含 RoPE 的選擇 | 不含 RoPE | — | L／— | 128K | 只重建選中的 | 無 | NOT_FOUND | 摘要 |
| E | GEAR／JoLT | 低秩＋量化 | —／pre | 不訓練 | —／L | —／64K | —／Triton | 無 | 有碼 | 摘要／部分 |
| F1 | ThinK（ICLR'25） | 依 query 剪 K 通道 | post〔判讀〕 | 不需要（但依 query） | L(3)／— | — | 不重建 | 無 | Apache-2.0；KIVI 版只支援 LLaMA-2 | 部分 |
| G | MHA2MLA／TransMLA／X-EcoMLA／MLAfication／MatryoshkaKV／EchoKV／BELT／CSKV／LRKV | — | — | **要訓練** | — | — | — | — | — | 摘要 |

---

## 4. 現象：在真實 KV dump 上可以便宜量的東西

### 4.1 原文用的指標（我們照著量）

| 指標 | 誰用 | 怎麼算 | 原文的觀察 |
|:--|:--|:--|:--|
| 相對奇異值曲線 | ShadowKV 圖 1 | σ_i／σ_1，對數軸 | pre-RoPE K 最陡；post-RoPE K、V、X、W_K、W_V 都平得多 |
| Rank@v | Loki；xKV 圖 2c | 去平均共變異的特徵值累積到 v 所需的秩（Loki v＝90%，每層每 head；xKV v＝95%，串多層） | Loki：平均約 80／128；pre-RoPE 比 post-RoPE 低；淺層特別低 |
| 子空間相似度 | ShadowKV 的 D(H1,H2)＝⟨H1,H2⟩_F／r；OjaKV 的 SO＝Tr(U1ᵀU2U2ᵀU1)／r（兩者等價） | 兩個 rank-r 基底的投影矩陣內積 | ShadowKV（r＝256）：同序列延長 ≥0.98，不同序列 0.74–0.95；OjaKV：校準基底對測試的 oracle 基底 0.597 |
| 殘差能量比 RER／跨分布重建誤差 | OjaKV；KVTC 圖 6 左（A → B） | 在 A 上擬合基底，看 B 的 ‖X−X̂‖²／‖X‖² | OjaKV：同領域 0.035 → 跨領域 0.255；KVTC：10K → 100K 校準 token 時誤差大幅下降 |
| 誤差隨位置 | KVTC 圖 6 中 | 依 context 位置分桶 | 前面的 token（含 sink）誤差較高 → KVTC 不壓前 4 個 |
| 每頁所需秩 | PuzzleKV 圖 2 | 每頁達 80% 重建精度所需的秩 | 每層每 head 都低於滿秩 |
| 每 head 的失真分布 | KV-COBRA 圖 1b | 同一個 bpd 下，每個 head 的 D(r,b) | 跨 256 個 head 差 100 倍 |
| 跨層 CKA | xKV 圖 2b | 中心化 Gram 矩陣的對齊度 | 逐 token 的 cosine 低，但 CKA 高 → **只看 cosine 會低估跨層冗餘** |

### 4.2 我們已經有的工具：`code/m16_kvstruct.py`（K1）

- 這是 10/20 新寫、未 commit 的檔；我只讀了檔頭說明〔程式碼，檔頭〕。
- 已經會做的：
  - 每層每 KV head 的奇異值譜（Gram 串流累加）。
  - rank r 的能量比與相對誤差；centered／uncentered。
  - r90／r95／r99、stable rank。
  - 「**同一份 dump 內**，前 `--calib-tokens` 個 token 當校準 → 套到其後的 token」。
  - `head=all`：整層的 H 個 head 一起算。
  - 跨層 cosine（MiniCache 的量法）、相鄰 token 的 cosine。
  - **RoPE 逆旋轉與核對**：用 kvdump 10/20 補的 `rope_probe`、`inv_freq`；資料自身的 M 統計，照規則 6 寫成門檻。
- 資料來源：`code/m16_s1_kvdump.py`（S1，GPU 窗口跑）。
  - 格式：post-RoPE BF16，`[n_blocks][L'][2][H][512][D]`，block i 的第 t 個 token 在位置 i·512＋t。
  - 範圍：Llama／Qwen × T4（自然文字）／T2（合成句子）；32K 存全部層，128K 只存 5 層。

### 4.3 E-L1：在 K1 上補的四項（都只用 CPU，numpy）

1. **跨 prompt／跨任務的基底**（KVTC 的 A→B、ShadowKV 的 inter-context、OjaKV 的 RER／SO）。
   - K1 的「校準 → 測試」在同一份 dump 裡，等於 ShadowKV 的「同序列延長」，預期很好看（≥0.98）。
   - 要另外做：在 dump A 上擬合（每層每 head，去平均），在 dump B 上算 RER 與 SO。
   - 組合：T2 → T4、T4 → T2、T4（seed X）→ T4（seed Y）。
   - 同 seed 換文件需要多一份 dump：GPU 約 5–10 分鐘。
   - 這一項決定 E-L2 用「校準基底」還是「每 prompt／每 block 基底」。
2. **每 512-token block 的局部 SVD**（PuzzleKV 式，粒度換成我們的 block）。
   - 在 r＝19／26／32 時的能量。r 的取法：block 局部 SVD 的位元組比是 r(512＋128)／(512·128)＝r／102.4，所以 r＝19／26／32 分別對應 0.186／0.254／0.3125。
   - post-RoPE 與 pre-RoPE 都算。PuzzleKV 說 32-token 頁時差很小；512-token 的 block 裡高頻維度會轉很多圈，**post-RoPE 可能吃虧**〔推論〕。
3. **同位元組對 KIVI 的重建誤差**（KVTC 用 Frobenius 誤差當代理）。
   - 每（層, head）、K 與 V 分開，算相對 Frobenius 誤差。
   - 低秩組：校準 PCA r＝24／32／40（位元組比 r／128）、block SVD r＝19／26／32。
   - 量化組：INT2-KIVI（0.1875）、K4V2（0.25）、INT4-KIVI（0.3125）。
     - S1 已經存了 `int4_kivi` 的位元組（`m16_s1_kvdump.py` 的 FMTS_DEFAULT）〔程式碼〕。
     - INT2-KIVI 照 `m10_codec_x.py` 的分組，在 numpy 重算。
   - 輸出：每個位元組點，「低秩誤差 < 量化誤差」的 head 比例。
4. **跨層 CKA 與「串 W 層所需秩」**（xKV 圖 2b／2c）。
   - K1 只量逐 token 的 cosine；xKV 指出這會低估跨層冗餘。
   - 和 L5b 共用：只要 32K 的全層 dump。

另外順手報：KV-COBRA 式的每 head 誤差分布（最小／中位數／最大，以及前 10% 的 head 在哪幾層），以及 KVTC 式的誤差隨 block 位置。

**成本**〔推論〕：Llama 32K 的 dump 是 4 GiB。
- 每（層, head）的 Gram：32,768×128 → 128×128，約 1 GFLOP；全部 512 個（層, head）× K／V 約 1 TFLOP。4 核 numpy 約數十秒到幾分鐘。
- block 局部 SVD：每 block 每（層, head）一次 512×128 的 SVD（用 128×128 的 Gram 特徵分解），共 64×512×2＝65,536 次，約數分鐘。
- 全部遵守 `OMP_NUM_THREADS=4 nice -n 19 taskset -c 0-3`。

### 4.4 文獻給的先驗（是預測，不是結果）

- pre-RoPE K 的能量集中度應明顯高於 post-RoPE K 與 V（ShadowKV、Loki、SALS、KV-COBRA 一致）。
- 每 head 只用 128 維：Loki 的 Rank@90 約 80，所以 r＝24–40 的校準 PCA，**每 head 能量可能只有八成上下**。
- 跨 head 聯合（ShadowKV 1,024 維用 r＝160，相當於每 head 20）或跨層（KVTC）好得多。→ E-L2 的 PCA 可能要用「每層跨 head 聯合」〔推論〕。
- Qwen 的 K 有 k_proj bias，平均方向很大：第二階段 Qwen 的 fp8_head 壞掉（32K 39.5% vs E0 77.4%），先減平均的 fp8k0_head 就沒事（−1.0）〔本專案實測，`tables_longctx.md`〕。→ **PCA 一定要去平均，平均另存**。
- 校準基底跨任務會退化（OjaKV 0.035 → 0.255；ShadowKV 0.74–0.95）。KVTC 靠 16 萬 token、跨 32 層串接才壓住。

---

## 5. 同位元組品質測試（E-L2 的設計）

### 5.1 位元組算術

位元組比＝存的位元組 ÷ BF16。

| 格式 | 位元組比 | 出處 |
|:--|--:|:--|
| INT4-KIVI | 0.3125（每值 4 bit＋每 32 個一組的 FP16 scale／zero＝5 bit） | 〔程式碼〕`m10_codec.py:253` |
| K4V2（kv-int4_kivi-int2_kivi） | 0.25 | 〔程式碼〕`m10_codec_x.py:324` |
| INT2-KIVI | 0.1875（2＋1 bit） | 〔程式碼〕`m10_codec_x.py:321` |
| 校準 PCA，rank r，BF16 係數 | r／128（r＝40 → 0.3125；32 → 0.25；24 → 0.1875） | 〔推論〕。基底＋平均每個模型只存一次：Llama 32 層×2×8 head×128×128×2 B＝16 MiB（約 1/4 個 block），Qwen 7 MiB |
| K 用 PCA r、V 用 INT4-KIVI | (r／128＋0.3125)／2；r＝24 → 0.25 | 〔推論〕 |
| block 局部 SVD，rank r，BF16 | r(512＋128)／(512·128)＝r／102.4（r＝32 → 0.3125；26 → 0.254；19 → 0.186） | 〔推論〕 |
| 丟 token（block 粒度） | 保留比例 | S2／44e |

### 5.2 組別（新的只有 3–4 組；對照組都已經有列）

| 位元組點 | 新組（要跑） | 對照（已有的列） | 預期〔推論〕 |
|:--|:--|:--|:--|
| 0.3125 | **P40**：K（pre-RoPE、去平均）與 V 都用校準 PCA，r＝40 | INT4-KIVI（ALL） | 最好打平；**比 INT4 差太多就先查實作與 RoPE，不是結論**（規則 2） |
| 0.25 | **K24V4**：K 用 PCA r＝24，V 用 INT4-KIVI | K4V2（Qwen 32K −5.1）、丟 1/4（S2） | 文獻說 K 好壓、V 不好壓（xKV D.3、KV-COBRA），這組最有機會贏 |
| 0.1875 | **P24**：K、V 都用 PCA，r＝24 | INT2-KIVI（Llama 128K −7.2；Qwen 32K −12.9） | 每 head／每層的純截斷多半輸（VarRate flat、KVTC B.9／B.10、xKV 的 Single SVD）；跨層每 prompt（xKV）在它的設定下和 KIVI-2 打平。只有 E-L1 顯示誤差贏才跑 |
| 0.3125（選用） | **B32**：block 局部 SVD，post-RoPE，r＝32，不用校準 | INT4-KIVI | 只有 E-L1 顯示 block 局部比校準 PCA 好才加 |

若 E-L1 顯示「每層跨 head 聯合」的誤差明顯比每 head 小，就把 P／K24 換成聯合版：Llama 1,024 維、Qwen 512 維；位元組比同樣是 r_joint／(H·128)。

若 E-L1 顯示校準基底跨 prompt 退化很多（跨 dump 的 RER 遠高於同 dump），P 組改用**每 prompt 的基底**（ShadowKV／xKV 式：prefill 後對整段做 SVD；S4R 表 2 顯示這類作法在我們的 Qwen-1M、128K、2.5× 幾乎不掉）。代價是 block 不再獨立（§6.3），而且基底本身要算進位元組。

**比 KIVI 更有機會的第五組**（不在最小版內）：PCA 旋轉後，依 SVDq 式的位元表做逐座標量化，位元組比 0.1875。這是 SVDq／KVTC／KV-COBRA 都指向的方向，但實作與調參多一步。E-L2 的純截斷組全輸時，再決定要不要做。

### 5.3 RoPE、平均、sink 怎麼處理

- **RoPE**：壓之前用 block 的絕對位置把 K 反旋轉成 pre-RoPE，重建後再轉回去。
  - cos／sin 直接用 `m7_model` 前向用的 `rotary_emb` 算（GPU job 裡），避免在 numpy 重寫一次 llama3 的 rope_scaling（K1 檔頭有逐行對照，但 GPU 路徑沒必要重寫）。
  - Qwen 的 theta＝10,000,000，config 有 DCA 設定；harness 用的是 HF Qwen2 的 rotary_emb〔判讀〕。
  - 注意：VarRate／SVDq／KV-COBRA 用的是 Qwen2.5-7B-Instruct（theta 1e6），**和我們的 1M 版不是同一套 RoPE**〔判讀〕。
- **平均**：每（層, head）的 K、V 平均另存（Qwen 的 K bias 必須如此，§4.4）。
- **sink 與最近的 token**：KVTC 不壓前 4 個與最近 128 個（App. B.3／B.7）。我們的對照組（INT4／INT2／K4V2 的 ALL 列）沒有例外，所以新組也用 ALL，保持同條件；另外記錄「block 0 的誤差」。
- **校準資料**：不能和測試題重疊。
  - 用 HotpotQA 段落或 LongBench 文件，換一段不相交的 seed。
  - 4–8 條 × 32K（13–26 萬 token），參考 KVTC 的 160K。
  - 每（層, head）的平均與共變異用 FP32 累加，128×128 的特徵分解在 CPU 或 GPU 都可以。

### 5.4 最簡單的實作與題組

- **實作**：context prefill 完、作答之前，對 KVModel 的 KV 緩衝原地做一次「壓 → 解」（就是 VarRate、Palu 準確率路徑的「重建後放回」）。
  - 等同第二階段的讀法 A（全部載入）。
  - 不用改 `m10_codec` 的介面：PCA 要知道位置，而 codec 的 `encode(x, fmt)` 不帶位置〔程式碼，`m10_codec.py`〕。
  - 位元組照 §5.1 的公式記，並在程式裡 assert 實際存的 tensor 大小。
- **題組**：沿用長 context 篩選的 32K 題組（`tables_longctx.md`，每個長度 n＝48）。它已經有 E0、ALL int4_kivi、int2_kivi、kv-int4_kivi-int2_kivi 的列，可以逐題配對；32K 有訊號才延伸到 64K／128K。
- **判定**：對同位元組的 KIVI 組做逐題配對的答對率差；照 `m10_analyze.py eps_check` 的 bootstrap（題為單位，10,000 次）報 95% CI。同時報對 E0 的差。

### 5.5 GPU 時間

- 〔估計，未量〕：新組 3–4 組 × 2 個模型 × 48 題 × 32K。
- 第二階段的 33b 品質段：3 個配置 × 2 種讀法、T2–T4 各 180 題＋T5 60 題，兩個模型 2 張卡共 3 小時 49 分（RUNLOG 10.10）〔本專案實測〕。
- 照比例推，加上校準 prefill 與冒煙，約 **1 GPU 小時（±50%）**。
- 照第二階段的規矩，先每個模型試跑 2 題量每題秒數，再定 n。
- 128K 每題的成本約是 32K 的 4–8 倍 → 不在本次預算內。

---

## 6. 跟分層的關係

### 6.1 搬運量：同位元組就是同搬運量

Llama 32K 的 KV 是 4 GiB BF16（64 個 block × 64 MiB）；Qwen 是 1.75 GiB（乘 0.4375）。

| 位元組比 | 量 | CPU 層（實測 49.9 GiB/s） | SSD 級（參數 6.98 GiB/s） | NFS 級（參數 0.331 GiB/s） |
|:--|--:|--:|--:|--:|
| 1.0（BF16） | 4 GiB | 80 ms | 0.57 s | 12.1 s |
| 0.3125（INT4-KIVI／PCA r40） | 1.25 GiB | 25 ms | 0.18 s | 3.8 s |
| 0.25（K4V2／K24V4／丟 3/4） | 1.0 GiB | 20 ms | 0.14 s | 3.0 s |
| 0.1875（INT2-KIVI／PCA r24） | 0.75 GiB | 15 ms | 0.11 s | 2.3 s |
| 0.125（PCA r16） | 0.5 GiB | 10 ms | 0.07 s | 1.5 s |

〔推論：算術〕。頻寬：CPU 層是實測（`stage2_factcheck_and_toolbox.md` §3.1）；SSD、NFS 級是限速器參數，不是量測（同檔 §1）。

- 低秩的秩是連續的旋鈕，可以讓位元組剛好配合某一層的頻寬或預取提前量。
- 但「依頻寬選壓縮比」已經有 KVServe 的公式與 KVTC 的多個壓縮比（L2 B2、B3）→ 這一點的新意低。

### 6.2 重建成本：GEMM，不是逐元素的反量化

**在哪算**：
- V：用每 head 固定基底時，可以吸收進 W_O，不用重建（Palu）；但要自寫 attention 路徑。
- K：因為 RoPE，必須「重建 → RoPE」（Palu、ShadowKV、Eigen、SVDq）。不重建的只有兩種：直接在因子上算 attention（PuzzleKV、JoLT），或只拿 latent 選 token（SALS、LRQK、Loki）。

**量級**〔推論：算術〕：Llama 一個 block、校準 PCA r＝40：
- 512 個（層, K/V, head）各做一次 [512×40]×[40×128] 的 GEMM，共 2.7 GFLOP。
- 讀 20 MiB、寫 64 MiB。
- 就算只拿到 100 TFLOPS，計算也只要 27 µs → 是 memory-bound。

**和反量化比**：
- 我們的 INT4-KIVI 融合反量化：Llama 每 block 0.101 ms、Qwen 0.076 ms〔本專案實測，RUNLOG 11.2f〕。換算有效頻寬約 0.87 TB/s（84 MiB／0.101 ms）。
- 同樣的位元組流量下，重建大概也在 0.1 ms／block 這個量級。**NOT_MEASURED。**

**和搬運比**〔推論：算術〕：
- 閒時：CPU 層搬一個 0.3125 的 block 要 20 MiB／49.9 GiB/s＝0.39 ms，反量化約是它的 26%。
- 忙時：H2D 慢 46–50% → 0.57–0.59 ms；FP8 反量化慢 2.4–3.8 倍（L2 §0.1）→ 若 INT4 反量化也一樣，是 0.24–0.38 ms，占搬運的 42–67%。
- → **在 CPU 層、GPU 忙時，解碼成本和搬運同一個量級**。SSD／NFS 級則可忽略。這和第二階段「INT4 只輸在真實 CPU 層」一致。

**新的問題**〔推論，待 E-L3〕：低秩重建是 GEMM，會用到矩陣單元；我們的「忙」是 BF16 matmul 背景（`m7_busy.py`）。GEMM 型解碼在忙時被拖慢的程度，可能和逐元素的反量化不同。**沒找到有人量過。**

**文獻裡的跡象**：
- Palu：重建 kernel 超過 16K 就變 compute-bound，加速縮小〔全文〕。
- xKV：dense 重建讓 attention 變成 0.4 倍〔全文〕。
- KVTC B.16：同一張 GPU 做壓縮解壓，TTFT 慢 32–42%〔部分〕。
- S4R 表 3（Qwen3-4B、120K、沒有 kernel 最佳化）：xKV 的 TTFT 79.9 s、不壓 24.7 s；生成 512 token 260.8 s vs 54.9 s〔部分〕。沒有專用 kernel 時，線上 SVD 與 dense 重建的成本可以大過 prefill 本身。

### 6.3 基底的相依性：決定能不能「一個 block 一個 block」地搬

| 基底 | 例子 | 對 block 粒度的還原（Cake、G5b、分層） | 多輪 |
|:--|:--|:--|:--|
| **校準（每個模型一份）** | KVTC、Loki、Eigen、VarRate、KV-COBRA | 每個 block 獨立編碼，最相容；基底 16 MiB 常駐 GPU 即可〔推論〕 | 跨任務會退化（OjaKV、ShadowKV 圖 1） |
| **每 prompt** | ShadowKV、xKV、SVDq | 基底要等整段 prefill 完才算得出來（ShadowKV 128K 佔 3.25%、xKV 2.5%），是所有 block 共用的相依物件 | KVTC B.11：每 prompt 的基底對後續對話壓不好（GSM8K 56.8 → 31.4）；ShadowKV+ 把新 token 投到舊基底上，在它們的測試裡沒事 |
| **每 block／頁** | PuzzleKV | 完全獨立；基底每 block 多付 128／512＝25% | 新 block 各自分解 |
| **跨層** | xKV（W＝4）、KVTC（全層） | token 係數跨層共用：要重建第 ℓ 層，得先有該窗口的共用係數 → 逐層載入的管線要改；Cake 以 block 為單位，不受影響〔推論〕 | 同「每 prompt」或「校準」 |

### 6.4 前人的成本分析：做了什麼、缺什麼

| 誰 | 分析了什麼 | 沒有的 |
|:--|:--|:--|
| ShadowKV | 等效頻寬式；重建 K 與取 V 重疊（A100、PCIe） | GPU 忙、SSD、重算這個選項 |
| xKV | 卸載（xK-SR）vs 全留 GPU（xKV-SR）的吞吐 | 同上 |
| LRQK | 吞吐、miss 率；瓶頸是 CPU 端 index | 同上 |
| KVTC | 解壓 vs 重算的 TTFT；LMCache 多 client（B.16） | 解壓和計算搶 GPU 的分析（只有一句話）；SSD／NFS |
| KVServe（L2 B2） | 損益兩平頻寬 B\*＝(1−1/cr)·s | 加總模型、沒有爭用；也沒有低秩 codec |
| Palu | kernel 在長序列變 compute-bound | 卸載 |

### 6.5 可能的新意（依新意 × 成本排序）

- **N1｜解碼方式 × GPU 爭用**：GEMM 重建 vs 逐元素反量化 vs 不壓縮，在 GPU 忙時的還原時間。
  - 延伸第二階段 11 §3d 與 L2 G2；工具都已有。新意中，成本低（E-L3）。
- **N2｜低秩 K 當還原時的 block 選擇器**：把 ShadowKV 的想法從 decode 的 token 粒度搬到「還原」的 block 粒度。
  - 快層只放 rank-r 的 pre-RoPE K（r＝16 時只占 K 的 1/8、KV 的 1/16），第 2 輪問題來時先打分，再決定從慢層載哪些 block。
  - 併進 C1a 當一種挑法，不另開 GPU。
  - 風險：YAKV 指出低秩 K 在抽取型任務選錯（L1 C18），LRQK 在 Qwen2.5-7B 的 VT 也掉很多。
- **N3｜同位元組的「丟 vs 降精度 vs 降秩」三方比較**：S2 的擴充。
  - VarRate、xKV、SVDq 做過一部分，但 ≥32K、兩輪、Qwen2.5-7B-1M、0.25／0.1875 這些點沒人做。
  - 新意低到中，多半是乾淨的負面結果。
- **N4｜秩當分層的連續旋鈕**：依層的頻寬或預取提前量選 r。新意低（KVServe、KVTC 已依頻寬選壓縮比）。

---

## 7. 缺口與停止條件

| # | 缺口 | 最接近的前作 | 我們 2.5 天內能做的 | **結果是 X 就不值得做** |
|:--|:--|:--|:--|:--|
| G-L1 | 我們的模型、≥32K、兩輪下，同位元組比較「降秩 vs 降精度 vs 丟」 | VarRate（≤16K、Llama-3.1／Qwen2.5-7B-Instruct、KIVI-2）、xKV（64K、8×）、SVDq（64K、只壓 K）、S4R 表 2（Qwen2.5-7B-1M、128K、2.5×，但沒有量化組） | E-L2（32K、0.3125／0.25／0.1875） | 0.25 與 0.1875 兩點、兩個模型，低秩組對同位元組 KIVI 的配對差 95% CI 下界都 ≤0 → 低秩不當獨立的儲存格式（動作空間不加） |
| G-L2 | 重建（GEMM）在 GPU 忙時的成本，以及它和反量化的差別 | 沒有（KVTC B.16 只有現象） | E-L3 | 忙／閒的慢化倍數和 INT4 融合反量化相差 <20%，而且忙時重建 <25% 的 CPU 層同位元組搬運時間 → 重建只是「另一種反量化」，沿用既有的成本項 |
| G-L3 | 還原時用低秩 K 選 block | ShadowKV、xK-SR、LRQK、Loki（都是 decode 時、token 粒度） | C1a 加一種挑法（rank r＝8／16 的 pre-RoPE K 摘要） | recall@k 比 Quest 的 min／max 摘要高不到 5 個百分點，或比隨機高不到 10 個百分點 → 不用 |
| G-L4 | 多輪對話裡基底的漂移（校準 vs 每 prompt） | KVTC B.11、OjaKV、ShadowKV 圖 1 | E-L1 的第 1 項；若 dump 了第 2 輪，加算 turn1 基底 → turn2 token 的 RER | 跨任務 RER ≤ 同 dump 內 RER 的 1.2 倍 → 沒有漂移問題，「每 prompt 還是校準」不構成議題 |
| G-L5 | 我們模型的低秩結構本身（閘門） | Loki、ShadowKV、KV-COBRA（別的模型或別的設定） | E-L1 | 見 §8 E-L1 的閘門 |

**不做的**：
- 官方 Palu／xKV／ShadowKV／LRQK 的重現：CUDA，或不支援 Llama-3.1／GQA。
- MLA 轉換與所有要訓練的方法。
- 128K 的品質：每題成本是 32K 的 4–8 倍，32K 沒訊號就不碰。

---

## 8. 建議的最小實驗

### E-L1　KV 的低秩結構與同位元組重建誤差（**閘門**）

- **GPU**：0 h；另外最多 0.2 h。
  - S1 的 KV 匯出（約 10 分鐘）已經在第三階段窗口的第一個 job。
  - 若要「同任務、換文件」的跨 prompt 基底，多 dump 一份 T4（另一個 seed），約 5–10 分鐘。
- **CPU**：K1（`m16_kvstruct.py analyze`）＋§4.3 的四項。全部 `OMP_NUM_THREADS=4 nice -n 19 taskset -c 0-3`，numpy only。預計 10–30 分鐘〔估計〕。
- **產出**：每（模型、任務、層、head、K_pre／K_post／V）在 r＝16／19／24／26／32／40 的能量與相對誤差；跨 dump 的 RER／SO；每 block SVD 的能量；同位元組對 INT2／K4V2／INT4-KIVI 的誤差比。
- **閘門**（在看任何答對率之前寫死）：
  - 對每個位元組點、每個模型，算「低秩誤差 < 同位元組 KIVI 誤差」的（層, head）比例，K、V 分開算。
  - K、V 在 0.25 與 0.1875 兩點都 <20% → **不做 E-L2**。紀錄成「在這兩個模型上，降秩的重建誤差被同位元組量化支配」。
  - 只有 K 過（≥50%）→ E-L2 只跑 K24V4。
  - 都過 → 照 §5.2 全跑。
- **注意**：重建誤差不等於答對率（KVTC 自己說 Frobenius 只是代理）。所以閘門只用來「省掉看起來必輸的組」，不用來宣稱「會贏」。

### E-L2　同位元組答對率（32K）

- **GPU**：約 1 GPU 小時（±50%）〔估計〕。新組 3–4 個 × 2 模型 × 48 題；先每個模型試跑 2 題定 n。品質 run：污染時保留分數、`latency_ms` 作廢（CLAUDE.md §3）。
- **組**：P40、K24V4、P24（B32 選用）；對照是長 context 題組既有的 INT4-KIVI、K4V2、INT2-KIVI 列，以及 S2 的丟 1/4。
- **停止條件**：
  - **繼續**（低秩進動作空間的候選）：至少一個模型、0.25 或 0.1875 至少一點，低秩組對同位元組 KIVI 的配對差 CI 下界 >0，而且另一個模型同一點的 CI 下界 > −2 個百分點。接著才考慮 64K、PCA＋位元分配（第五組）、低秩的還原速度。
  - **停**：兩點、兩個模型的 CI 下界都 ≤0 → 低秩不當獨立的儲存格式。只留 N2（併入 C1a）與 E-L3 的重建成本結論。
  - **先查、不是結論**：P40 在 0.3125 就比 INT4-KIVI 差（CI 上界 <0）→ 先查 RoPE 逆旋轉、去平均、位元組計算（規則 2）。

### E-L3　重建 vs 反量化，在 GPU 閒／忙時

- **GPU**：約 0.3 GPU 小時〔估計〕。**量時間，要獨佔計時卡**；污染就重量。
- **做法**：每個 block（Llama 64 MiB、Qwen 28 MiB）量四種解碼，閒時與 busy2（`m7_busy.py --duty 0.7`，重算慢 2.07 倍的那一級）各 200 次：
  - (a) INT4-KIVI 融合反量化（`m10_dequant_fast.py`）；
  - (b) PCA r＝24／40 的重建（batched matmul＋RoPE 回轉）；
  - (c) block SVD r＝32 的重建；
  - (d) 不解碼（只有同位元組的 H2D）。
  - 可以用 `m16_i4_copybench.py`／`m10_speed.Busy` 的殼。
- **停止條件**：
  - (b)／(c) 的忙／閒倍數比 (a) 高 ≥1.5 倍，**或**忙時 (b) ≥ CPU 層同位元組 H2D 的 50% → 寫進成本模型，成為「解碼方式 × 爭用」的新項（N1）。
  - 倍數差 <20%，而且忙時 (b) < 25% 的 H2D → 重建只是另一種反量化，不另立項。

**合計**：GPU 約 1.3–1.5 小時〔估計〕。E-L1 不過閘時，只剩 E-L3 的 0.3 小時。

---

## 9. 參考清單

「核」：✓＝arXiv API 核過編號、標題、第一作者。venue 欄只寫查得到的來源，其餘是「未查證」。

| # | 方法 | 連結 | venue（來源） | 程式碼（授權） | 核 | 深度 |
|:--|:--|:--|:--|:--|:--|:--|
| 1 | Palu（Chang et al.） | https://arxiv.org/abs/2407.21118 | ICLR 2025（proceedings.iclr.cc、iclr.cc 海報頁 poster/29993） | https://github.com/shadowpa0327/Palu（MIT） | ✓ | 全文 |
| 2 | xKV（Chang et al.） | https://arxiv.org/abs/2503.18893 | ICML 2026（arXiv comment；PDF 頁首 PMLR 306） | https://github.com/abdelfattah-lab/xKV（Apache-2.0） | ✓ | 全文 |
| 3 | ShadowKV（Sun et al.） | https://arxiv.org/abs/2410.21465 | ICML 2025（L1／L2 已核；LRQK 參考文獻同列） | https://github.com/ByteDance-Seed/ShadowKV（Apache-2.0） | ✓ | 全文 |
| 4 | LRQK（Li et al.） | https://arxiv.org/abs/2510.23649 | NeurIPS 2025（arXiv journal_ref；neurips.cc 海報頁 118451） | https://github.com/tenghuilee/LRQK（MIT） | ✓ | 全文 |
| 5 | KVTC（Staniszewski, Łańcucki） | https://arxiv.org/abs/2511.01815 | ICLR 2026（arXiv comment；PDF 頁首） | 官方 NOT_FOUND | ✓ | 部分 |
| 6 | Loki（Singhania et al.） | https://arxiv.org/abs/2406.02542 | NeurIPS 2024（arXiv comment） | https://github.com/hpcgroup/loki（MIT） | ✓ | 部分 |
| 7 | Eigen Attention（Saxena et al.） | https://arxiv.org/abs/2408.05646 | Findings of EMNLP 2024（ACL Anthology 2024.findings-emnlp.899） | https://github.com/UtkarshSaxena1/EigenAttn（無授權欄） | ✓ | 部分 |
| 8 | ThinK（Xu et al.） | https://arxiv.org/abs/2407.21018 | ICLR 2025 Spotlight（arXiv comment；PDF 頁首） | https://github.com/SalesforceAIResearch/ThinK（Apache-2.0） | ✓ | 部分 |
| 9 | SVDq（Hong et al.） | https://arxiv.org/abs/2502.15304 | 未查證 | NOT_FOUND | ✓ | 部分 |
| 10 | LoRC（Zhang et al.） | https://arxiv.org/abs/2410.03111 | 未查證 | NOT_FOUND | ✓ | 部分 |
| 11 | ReCalKV（Yan et al.） | https://arxiv.org/abs/2505.24357 | 未查證 | https://github.com/XIANGLONGYAN/ReCalKV（Apache-2.0） | ✓ | 部分 |
| 12 | PuzzleKV（Wang et al.） | https://arxiv.org/abs/2608.23843 | 未查證 | NOT_FOUND | ✓ | 部分 |
| 13 | VarRate（Esmat et al.） | https://arxiv.org/abs/2607.15498 | 未查證 | NOT_FOUND | ✓ | 部分 |
| 14 | KV-COBRA（Ha et al.） | https://arxiv.org/abs/2609.24298 | 未查證 | NOT_FOUND | ✓ | 部分 |
| 15 | OjaKV（Zhu et al.） | https://arxiv.org/abs/2509.21623 | Findings of ACL 2026（ACL Anthology 2026.findings-acl.494） | https://github.com/zzbright1998/OjaKV（授權未查） | ✓ | 部分 |
| 16 | S4R（Han et al.） | https://arxiv.org/abs/2608.00528 | arXiv comment：AACL-IJCNLP 2026 Main（會議頁未查） | https://github.com/Draconis98/S4R（授權未查） | ✓ | 部分 |
| 17 | JoLT（Krishnan, Schulz） | https://arxiv.org/abs/2607.12550 | arXiv comment：Under review at ICLR 2027 | https://github.com/rahulk98/JoLT-Master-Thesis（授權未查） | ✓ | 部分 |
| 18 | SALS（Mu et al.） | https://arxiv.org/abs/2510.24273 | 未查證 | NOT_FOUND | ✓ | 摘要 |
| 19 | AttSVD（Abdali et al.） | https://arxiv.org/abs/2610.06927 | 未查證 | NOT_FOUND | ✓ | 摘要 |
| 20 | KQ-SVD（Lesens et al.） | https://arxiv.org/abs/2512.05916 | 未查證 | NOT_FOUND | ✓ | 摘要 |
| 21 | GEAR（Kang et al.） | https://arxiv.org/abs/2403.05527 | 未查證 | https://github.com/HaoKang-Timmy/GEAR（授權未查） | ✓ | 摘要 |
| 22 | ASVD（Yuan et al.） | https://arxiv.org/abs/2312.05821 | 未查證 | 未查 | ✓ | 摘要 |
| 23 | ECKVH（Yu et al.） | https://arxiv.org/abs/2406.07056 | 未查證 | 未查 | ✓ | 摘要 |
| 24 | RAP（Xin et al.） | https://arxiv.org/abs/2602.02599 | 未查證 | 未查 | ✓ | 摘要 |
| 25 | MHA2MLA（Ji et al.） | https://arxiv.org/abs/2502.14837 | ACL 2025（arXiv comment） | https://github.com/JT-Ushio/MHA2MLA（GitHub 轉到 JT-Ushio/MLAfication，Apache-2.0） | ✓ | 摘要 |
| 26 | TransMLA（Meng et al.） | https://arxiv.org/abs/2502.07864 | NeurIPS 2025（neurips.cc 海報頁 117839，標題不同） | https://github.com/MuLabPKU/TransArch（原 fxmeng/TransMLA；無授權欄） | ✓ | 摘要 |
| 27 | X-EcoMLA（Li et al.） | https://arxiv.org/abs/2503.11132 | 未查證 | https://github.com/AMD-AGI/AMD-Hybrid-Models（授權未查） | ✓ | 摘要 |
| 28 | MLAfication（Wang et al.） | https://arxiv.org/abs/2610.17202 | 未查證 | https://github.com/JT-Ushio/MLAfication（Apache-2.0） | ✓ | 摘要 |
| 29 | MatryoshkaKV（Lin et al.） | https://arxiv.org/abs/2410.14731 | 未查證 | 未查 | ✓ | 摘要 |
| 30 | EchoKV（Ji et al.） | https://arxiv.org/abs/2603.22910 | 未查證 | 未查 | ✓ | 摘要 |
| 31 | BELT（Liu et al.） | https://arxiv.org/abs/2610.17966 | 未查證 | 未查 | ✓ | 摘要 |
| 32 | CSKV（Wang et al.） | https://arxiv.org/abs/2409.10593 | arXiv comment：NeurIPS ENLSP-IV 2024 workshop | 未查 | ✓ | 摘要 |
| 33 | LRKV（O'Neill et al.） | https://arxiv.org/abs/2601.11471 | 未查證 | 未查 | ✓ | 摘要 |
| 34 | SpectrumKV（Yang） | https://arxiv.org/abs/2606.08635 | 未查證 | 未查 | ✓ | 摘要 |

**34 號的備註**：它不是低秩，而是 PD 分離傳輸時逐 token 分配精度。摘要說「Qwen2.5-7B catastrophically fails under INT4 KV quantization」——它的 INT4 量法沒寫在摘要；和我們 Qwen 的 int4_token 全壞、int4_kivi 沒事〔本專案實測，08 §2／`tables_longctx.md`〕對照看，又是「Qwen 的 K 需要逐通道或去平均」的旁證〔推論〕。

**本專案檔案**：
- `docs/stage3_20261019/research_map.md`、`lit/L1_eviction_selective_loading.md`、`lit/L2_compression_offload_prefetch.md`、`docs/PAPER_GUIDE.md`
- `docs/stage3_20261019/stage2_factcheck_and_toolbox.md`
- `results/m10_longctx_mi300x/analysis/tables_longctx.md`、`results/RUNLOG_MI300X.md`（10.10、11.2f）
- `code/m10_codec.py:253`、`code/m10_codec_x.py:321,324`、`code/m7_model.py:90–107`、`code/m16_s1_kvdump.py`、`code/m16_kvstruct.py`（檔頭）
