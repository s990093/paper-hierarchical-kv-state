# L1　KV eviction／token 重要度／選擇性載入（Cascade）文獻地圖（第三階段選題用）

> **日期**：2026-10-19（repo HEAD `c27ec61`）。**執行者**：文獻 agent（不碰 GPU、不跑耗 CPU 的工作、沒有裝任何套件；只寫這一個檔）。
> **範圍**：方向 A＝eviction、token 重要度、逐層／逐 head 保留率；Cascade＝選擇性載入（稀疏 attention、分層＋top-k、還原時只載一部分）。
> **查證方式**：
> - 每個 arXiv ID 都在 10/19 用 `curl https://arxiv.org/abs/<id>` 抓 `citation_title` 確認存在，標題、首位作者、arXiv comment（venue）照抄。
> - 實驗條件從 `arxiv.org/html/<id>` 全文轉純文字後 grep（硬體、模型、長度、batch、限制）。
> - 官方實作看 GitHub API `/languages`（各語言位元組數）和 README，只讀不裝。
> - 腳本與快取在 scratchpad，不進 repo。
> **閱讀深度**：〔部分〕＝摘要＋從全文抓實驗設定與主要主張的段落；〔摘要〕＝只讀摘要、會議頁或搜尋摘要。**本檔沒有任何一篇是〔全文〕**。
> **數字規則**：所有數字都是原論文在**自己的**模型／硬體／長度／batch 下量的，**不能跨列比較或排名**。條件沒找到就寫 NOT_FOUND。repo 自己量到的數字一律附檔名。
> **審查性質**：self-review，非 cross-model（CLAUDE.md §5）。

---

## 0. 先講結論

1. **和 repo 已有文獻的差別**。已有的：P08（kvpress 分數定義）、14 號（DS1／DS2／D3 計畫）、`PAPER_GUIDE.md`（Quest、ShadowKV、InfiniGen、ArkVale、QEvict、YAKV 卡）、`SOTA_ANALYSIS_20261006.md`（分層與逐出系統）。本檔補四塊：
   - **(a)** 2025 下半年到 2026 的多輪／agent eviction：A15–A23，共 9 篇。
   - **(b)** 2026 的「稀疏 attention × 分層記憶體」系統：HiSparse、SparseServe、NOSA、SparDA、Fathom、SAC、AVSG、OasisKV。
   - **(c)** 還原時只載一部分的前作：ContiguousKV、CacheTune（IMPRESS 已在 repo）。
   - **(d)** 「邏輯上少看」和「實體上少搬、少佔」不一樣的直接證據：kvpress 的假 key 遮罩、vToken、Tangram、AgentKV、RocketKV-MT。
2. **文獻已經有的共識**（各自在自己的設定下）：
   - 寫入時依問題挑的刪除，多輪會壞：SCBench、KVzip、RocketKV-MT、EpiCache。
   - agent 下「每輪立刻壓縮」會傷品質，延後到拿到之後的 query 再壓可救回大部分（2608.00902）；壓縮還會讓 agent 多走好幾輪、重複搜尋。
   - 業界普遍用 KV 量化，很少用 eviction（YAKV 引述 Li et al.）。
   - 「不載入但之後可召回」要成立，快層必須留一份 key 摘要：Quest 的 min/max、ShadowKV 的低秩 K、YAKV 的 1–2 bit K、SpeCache／Fathom 的低位元 K。YAKV 指出這些摘要在需要大量抽取資訊的任務上會挑錯。
   - 2026 年稀疏載入已進到生產級框架（HiSparse 合進 SGLang 上游），但全部在 NVIDIA 或 Ascend 上做。
3. **和第一、二階段最接得上的空白（§4 G1）**：
   - 在 Cake 式還原（前段重算、後段載入）下，前段 block 本來就會被精確重算，丟掉它既不省時間、也不傷品質。所以「有損的丟」只在**載入區**有意義。
   - 在載入區，丟掉要和「同樣位元組數的低精度」比才公平。這個比較我在文獻裡**沒找到**。
4. **ROCm**：沒有任何一個官方實作在 MI300X 上被驗證過（§1.1、§2.1 最後一欄）。
   - 本機 venv（`/mlsteam/workspace/venv/tiara-v028`）有 ROCm 版 PyTorch、Triton、transformers 5.17、vLLM 0.28.0，**沒有** flash-attn，也**沒有** FlashInfer。
   - repo 的 `code/m10_drop.py` 已重寫 StreamingLLM、SnapKV、TOVA、H2O（近似）、PyramidKV、AdaKV、Expected Attention、KNorm、KeyDiff、QRY 的分數。這是期限內唯一能直接用的路。
5. **期限前的建議順序**（§5）：
   1. X1：同樣位元組數下，丟掉 vs 降精度。最便宜，也最能判斷「丟是不是一層」。
   2. X2＋X4：寫入時的分數能不能預測第 2 輪要用的 block；block 摘要（Quest min/max）在 512-token block 上準不準。
   3. X3：用 D3 的速度資料算「被丟的 block 有多少落在載入區」，並分 GPU 閒、忙兩種情況。

---

## 1. 方向 A：KV eviction／token 重要度／保留率分配

### 1.1 總表

欄位說明：
- 「何時決定」：寫入時（prefill 後一次）、每步（decode 時持續）、離線（訓練或校準）。
- 「需問題」：打分時是否需要知道這一輪的問題。
- 「被丟的去哪」：本表全部是**永久刪除**，除了 A22 QEvict（INT2 可升回）和 A18 IntentKV（導到 dead slot，仍是刪除）。

| # | 方法（年；venue） | arXiv（10/19 已確認） | 何時決定 | 粒度／預算分配 | 需問題 | 官方實作與 ROCm | 深度 |
|:--|:--|:--|:--|:--|:--|:--|:--|
| A1 | H2O（2023；arXiv comment 沒寫 venue，repo P08 記 NeurIPS'23，未重查） | 2306.14048 | 每步（累積 attention） | token；heavy hitter＋最近 | 否 | FMInference/H2O：全 Python（建在 FlexGen／HF）；ROCm 未驗證 | 部分 |
| A2 | Scissorhands（2023） | 2305.17118 | 每步 | token | 否 | 未查 | 摘要 |
| A3 | StreamingLLM（ICLR'24） | 2309.17453 | 依位置 | token；sink＋最近 | 否 | 全 Python（23 KB）；ROCm 未驗證 | 部分 |
| A4 | TOVA（2024，preprint） | 2401.06104 | 每步（最後一個 query） | token | 否 | 全 Python（20 KB）；ROCm 未驗證 | 部分 |
| A5 | SnapKV（2024；venue 未查證） | 2404.14469 | 寫入時 | 每個 head 的 token＋pooling | **原文：是**（觀察窗是 prompt 尾端，含指令）；kvpress 預設：否 | Python（monkeypatch 舊版 transformers）；ROCm 未驗證 | 部分 |
| A6 | PyramidKV（2024；venue 未查證） | 2406.02069 | 寫入時 | 逐層預算（低層多） | 同 SnapKV | KVCache-Factory：Python＋3 KB CUDA | 部分 |
| A7 | Ada-KV（NeurIPS'25） | 2407.11550 | 寫入時 | 逐 head 自適應預算 | 有問題、無問題兩種都測 | Python＋3 KB CUDA（變長更新）＋flash-attn varlen | 部分 |
| A8 | DuoAttention（2024；venue 未查證） | 2410.10819 | 離線學每個 head 的角色 | retrieval head 全留；streaming head 只留 sink＋最近 | 否 | 要 CUDA toolkit、flash-attn 2.6.3、FlashInfer、Block-Sparse-Attention → **只能 CUDA** | 部分 |
| A9 | Expected Attention＋kvpress（2025） | 2510.00636 | 寫入時或 decode | token | 否 | kvpress：全 Python（transformers hook）；逐 head 版用假 key 遮罩；ROCm 未驗證 | 部分 |
| A10 | KVzip（NeurIPS'25 Oral） | 2505.23416 | 寫入時（用重建原文打分） | KV pair；各 head 預算不均 | **否**（query-agnostic） | Python＋10 KB CUDA＋flash-attn 2.7.4；ROCm 未驗證 | 部分 |
| A11 | SCBench（ICLR'25） | 2412.10319 | 評測 | — | — | — | 部分 |
| A12 | DefensiveKV（2025）／CriticalKV（ICML'26） | 2510.13334／2502.03805 | 寫入時 | 逐層／逐 head | 否（先壓縮再給問題） | Python＋4 KB CUDA；kvpress 有 CriticalKVPress | 部分／摘要 |
| A13 | KVP（ICML'26） | 2602.10238 | 離線 RL | 每個 head 一個 agent | 否 | 有 code 連結；要訓練 | 部分 |
| A14 | The Pitfalls of KV Cache Compression（ACL'26） | 2510.00231 | 評測 | — | — | — | 摘要 |
| A15 | Online KV Cache Compaction for LLM Agents（2026） | 2608.00902 | 每輪結束，或延後 | token（TE）／擬合（AM） | 用替代 query | PyTorch＋HF sdpa；程式碼釋出 NOT_FOUND | 部分 |
| A16 | AgentKV（2026） | 2609.14872 | 每輪；每個 phase 一組 query | page；實體壓縮 | 否 | Mini-SGLang；README 要求 NVIDIA GPU | 部分 |
| A17 | CommitKV（2026） | 2608.07855 | tool call 的 commit 前後各量一次 | page | 否 | 在 SGLang 上做；程式碼 NOT_FOUND | 部分 |
| A18 | IntentKV（2026） | 2606.09916 | 每輪；跨輪的意圖記憶（學習） | token（slot 重導） | 跨輪 | NOT_FOUND | 摘要 |
| A19 | EpiCache（ICML'26） | 2509.17396 | 分塊 prefill＋依 episode | token | 以 episode 為中心 | NOT_FOUND | 部分 |
| A20 | TaskPress（EMNLP'26） | 2608.03276 | 寫入時；task guide 當 meta-query | token | 任務層級 | NOT_FOUND | 部分 |
| A21 | RestoreKV（2026） | 2608.01247 | 寫入後＋學到的 restore token | — | 否 | kvpress 有 `RestoreKVPress`；要 LoRA 訓練 | 部分 |
| A22 | QEvict（2026） | 2608.05326 | decode 時持續 | window；全精度／INT2／刪除三階 | 否 | PyTorch 2.6＋flash-attn 2.8.3 | 部分 |
| A23 | MemDecay（2026） | 2607.10582 | 依區域先驗＋衰減 | page；可釘住 | 否 | NOT_FOUND | 摘要 |

**只列不寫卡**（都已確認 arXiv 存在；和「分數 ≠ 任務重要度」有關的放在 §3.3 討論）：
- VATP（2406.12335，EMNLP'24）：「Value 也要算」。
- K-VEC（2606.29563）：以覆蓋率為主的逐出。
- Make Each Token Count（2605.09649）：學出來的全域保留分數；full attention 未必最好。
- KeyDiff（2504.15364，NeurIPS'25）、KNorm（2406.11430，EMNLP'24）、Q-Filters（2503.02812）、LagKV（2504.04704）、KVzap（2601.07891）。
- HeadKV（2410.19258，ICLR'25）、RazorAttention（2407.15891）、Retrieval Head（2404.15574）。
- FastGen（2310.01801，ICLR'24）、CAKE（2503.12491，ICLR'25）、DynamicKV（2412.14838）、PyramidInfer（2405.12532，ACL'24）。
- R-KV（2505.24133）、DMS（2506.05345，NeurIPS'25；要訓練）。
- FlowKV（2505.15347，NeurIPS'25 workshop）、Tangram（2606.06302）、Rethinking KV Compression（2503.24000，MLSys'25）、Semantic Integrity Matters（2502.01941，ICML'26）。

### 1.2 卡片

> 每張卡固定七項：問題／方法與假設／原文數字（條件）／成本、限制、失效條件／實作與 ROCm／已有證據 vs 缺的比較／期限前最小版本。

#### A1　H2O：Heavy-Hitter Oracle〔部分〕
- **連結**：arXiv 2306.14048；`github.com/FMInference/H2O`。
- **問題**：生成時 KV 隨長度和 batch 線性長大，限制吞吐。
- **方法與假設**：
  - 少數「heavy hitter」token 拿走大部分 attention。保留 heavy hitter＋最近的 token，每步用累積 attention 更新。
  - 關鍵假設：用局部統計（每步把前面 token 的 attention 加總）算出的分數，和「考慮未來 token 的 attention」一樣有效。原文寫的是「empirically observe … equally effective」。
- **原文數字（條件）**：
  - 預算：20% heavy hitter。
  - 吞吐：比 DeepSpeed Zero-Inference、HF Accelerate、FlexGen 高最多 29×、29×、3×。條件：OPT-6.7B／30B；NVIDIA T4 16GB（放不下時開 CPU 卸載）；合成資料、prompt 全部 padding 到同長度，例如「512＋32」；end-to-end，含建 cache 的時間。
  - 延遲：同 batch 下比 FlexGen 最多低 1.9×。條件：A100 80GB，序列 4K–10K（例如 7000＋1024）。
- **成本、限制、失效條件**：
  - 打分要 attention 權重（kvpress 的 ObservedAttention 需要 eager attention）。
  - 品質主要在 lm-eval-harness／HELM 的短任務上量。
  - 永久刪除。
- **實作與 ROCm**：全 Python（約 2.2 MB），建在 FlexGen／HF 上；ROCm 未驗證。repo 已有近似版：只用 8 個觀察窗（14 號 §8.2）。
- **缺的比較**：多輪的第 2 輪；32K 以上的長 context 任務。
- **期限前**：已在 DS1（H2O 近似），不另做。

#### A2　Scissorhands〔摘要〕
- **連結**：arXiv 2305.17118。
- **問題與方法**：提出「重要性持續」假說：在某一步影響大的 token，之後也會影響大。依此在固定預算下以較高機率保留這些 token。
- **原文數字**：KV 記憶體最多省 5×；搭配 4-bit 量化最多 20×。模型、硬體、batch：NOT_FOUND（只讀摘要）。
- **和本題的關係**：之後有三篇的實測都和這個假說衝突：
  - QEvict（A22）：重要性會隨生成漂移。
  - SCBench（A11）：不同 query 之間差很多。
  - DefensiveKV（A12）：「穩定性假設」很脆弱。
- **期限前**：不做。

#### A3　StreamingLLM〔部分〕
- **連結**：arXiv 2309.17453（ICLR 2024）。
- **問題**：讓有限視窗訓練的模型做無限長的串流生成。
- **方法與假設**：保留開頭幾個 token（attention sink）＋最近視窗，其他刪除。sink 拿到大 attention 是 softmax 必須加總為 1 的副作用，和語意無關。
- **原文數字（條件）**：
  - Llama-2、MPT、Falcon、Pythia 做語言建模，到 400 萬 token 仍穩定。
  - 每 token 解碼最多比「滑動視窗＋重算」快 22.2×。條件：Llama-2-7B／13B，單張 NVIDIA A6000，HF Transformers。
- **限制**：不是長 context 理解（中段一律丟）。原文 Table 6 自己說 cache 變大不一定讓 perplexity 變好。
- **實作與 ROCm**：全 Python；ROCm 未驗證。repo 有 STREAM（DS1）。
- **期限前**：已在 DS1。

#### A4　TOVA〔部分〕
- **連結**：arXiv 2401.06104（arXiv comment：preprint）。
- **方法**：只用最後一個 query 的 attention（所有 head 平均）決定刪誰；把 transformer 看成有界的多狀態 RNN。
- **原文數字**：有些設定只用 1/8 cache 就接近完整模型，換算吞吐高 4.8×。模型：LLaMA-2、Mistral、Yi（7B）＋fine-tune 過的長文版本。硬體、batch：NOT_FOUND。
- **期限前**：已在 DS1。

#### A5　SnapKV〔部分〕
- **連結**：arXiv 2404.14469；`github.com/FasterDecoding/SnapKV`。
- **問題**：長 prompt 的 KV 太大。
- **方法與假設**：
  - 每個 head 在生成時關注的 prompt 位置是穩定的，可以用 prompt 尾端的「觀察窗」事先看出來。
  - 觀察窗對前面每個位置的 attention 加總，再 pooling 成群，每個 head 留前幾名，觀察窗本身保留。
  - **原版是 query-aware**：原文 §3 用「不同指令＋同一份文件」量命中率，觀察窗包含指令。
  - kvpress 預設把問題分開，所以在 kvpress 裡它變成 query-agnostic。
- **原文數字（條件）**：
  - 16K 輸入時生成速度 3.6×、記憶體效率 8.2×。條件：LWM-Text-Chat-1M（7B）；單張 A100-80GB；SnapKV 的 KV 上限 2048；生成長度固定 512；多個 batch size（Fig. 7）。
  - 單張 A100-80GB 可做到 380K（NIAH，KV 1024、pooling kernel 5、觀察窗 16），只掉一點準確度。
- **限制**：單輪；問題要在觀察窗裡才準。多輪會壞（A11 SCBench）。
- **實作與 ROCm**：Python（約 63 KB）＋monkeypatch 某一版 transformers；和本機 transformers 5.17 相不相容：NOT_VERIFIED。
- **期限前**：DS1 的 SNAP（無問題）和 QRY（有問題）已經是這篇的兩種讀法。

#### A6／A7　PyramidKV、Ada-KV：每層、每個 head 留多少〔部分〕
- **連結**：PyramidKV arXiv 2406.02069（KVCache-Factory）；Ada-KV arXiv 2407.11550（NeurIPS 2025；`github.com/FFY0/AdaKV`）。
- **方法與假設**：
  - PyramidKV：資訊在低層分散、高層集中（「金字塔式匯流」），所以低層多留、高層少留。
  - Ada-KV：從「刪除前後 attention 輸出差」的上界出發，讓同一層的 head 一起分預算。
- **原文數字（條件）**：
  - PyramidKV：
    - 條件：LongBench 17 個資料集；LLaMa-3-8B-Instruct、LLaMa-3-70B-Instruct、Mistral-7B。
    - 留 12% 時和完整 KV 相當。
    - 只留 0.7% 時，TREC 上最多比別的方法高 20.5 個絕對點。
    - NIAH：Llama-3-70B 只留 128 個 entry 就 100.0。
    - 硬體：NOT_FOUND。
  - Ada-KV：摘要沒有單一數字。在 RULER 13 個＋LongBench 16 個資料集上，有問題、無問題兩種設定都有改善；作者說無問題時優勢更大。硬體：NOT_FOUND。
- **限制**：
  - 每個 head 留的長度不同，需要變長 FlashAttention＋自訂 CUDA kernel。
  - kvpress 的 `AdaKVPress` 是用假 key 遮罩做的。原始碼註解直說「不省峰值記憶體、還稍微變慢」（`kvpress/attention_patch.py`，見 §3.2）。
  - Tangram（2606.06302）在 vLLM 上量到：每個 head 預算不同會造成 page 碎片；回收碎片最多佔 prefill 時間的 25%；decode 延遲最多膨脹 1.7×（4×A100）。
- **和分層儲存的衝突**：分層一次搬一個 block（我們是 512 token，所有層、所有 head 一起）。每個 head 各自挑的 token **不能直接換成少搬的位元組**。14 號 §8.3 已經寫了：只有 B 粒度能換成還原速度。
- **期限前**：DS1 已有 PYR、ADA（只在 T 粒度）；不另做。

#### A8　DuoAttention〔部分〕
- **連結**：arXiv 2410.10819；`github.com/mit-han-lab/duo-attention`。
- **方法與假設**：
  - 只有少數 retrieval head 需要看全部 token；其他 streaming head 只看 sink＋最近。
  - 每個 head 有一個 gate，用合成資料做蒸餾、優化 2,000 步，再依門檻分成兩類。原文說全部訓練可在 8×A100 的機器上做。
- **原文數字（條件）**：
  - 條件：單張 A100；BF16；KV 事先配好；Llama-2-7B 用 25% retrieval head，Llama-3-8B 用 50%。
  - 記憶體：MHA 最多省 2.55×、GQA 最多省 1.67×。
  - decode：最多快 2.18×（MHA）、1.50×（GQA）。prefill：最多快 1.73×、1.63×。
  - 加上 8-bit 權重＋4-bit KV 量化：Llama-3-8B 在單張 A100 上可解碼 330 萬 context，容量是 FP16 全 attention 的 6.4×。
- **限制**：
  - 每個模型都要先學出 head 分類。CLAUDE.md §8 不訓練，所以只能用官方釋出的分類；我們兩個模型（Llama-3.1-8B、Qwen2.5-7B-1M）有沒有釋出：NOT_VERIFIED。
  - KVzip 的 head 層級比較圖只畫到 cache 比例 0.4，理由是 DuoAttention 可達的下限是 0.32。
- **實作與 ROCm**：CUDA-only（理由見表）。
- **和分層的關係**：streaming head 的 KV 在寫入時就能丟。要存、要搬的只有 retrieval head 的 KV，省下的位元組約等於 streaming head 的比例。這是「逐 head 丟」裡**少數能直接換成搬運量**的做法：丟的是整個 head 的整段 KV，不是零散 token。
- **期限前**：不做（要 head 分類）。

#### A9　Expected Attention＋kvpress〔部分〕
- **連結**：arXiv 2510.00636；`github.com/NVIDIA/kvpress`。
- **方法與假設**：
  - hidden state 和 query 近似常態分布（原文 Fig. 1）。
  - 用「未來 query 的平均與共變異數」以封閉形式算每個 KV 的預期 attention，乘上 ‖V‖ 當分數。
  - 不需要真的 attention 矩陣，所以和 FlashAttention 相容。
- **原文數字（條件）**：
  - 摘要只說「多數情況贏過 baseline」，沒有單一數字（NOT_EXTRACTED）。
  - 模型：Llama3.1-8B（128k）、Qwen3-8B（32k）、Gemma3-12B（128k）。
  - prefill：LongBench、RULER 4K／16K、NIAH。decode：AIME25、MATH-500。
  - baseline：SnapKV、TOVA、KeyDiff、DuoAttention。
  - 硬體：NOT_FOUND。
- **kvpress 本身**：
  - README 列了 40 多個 press（含 wrapper），含 KVzip、KVzap、RestoreKV、CriticalKV、DMS、DecodingPress。
  - 預設只壓 context、問題另外給，所以預設是 query-agnostic 評測。
  - 有 HF leaderboard（RULER 4K）。
- **限制**：
  - 逐 head 的 press 用「假 key」讓 attention underflow 成 0。原始碼註解：「does not reduce peak memory and slightly increases runtime」（`kvpress/attention_patch.py`，10/19 讀）。也就是說，kvpress 的逐 head 結果只能當品質證據，不能當記憶體或速度證據。
  - FAQ 說記憶體約省 `compression_ratio × kv_cache_size`；速度要自己用 notebook 量。
- **實作與 ROCm**：全 Python（約 546 KB），掛在 transformers 的 forward hook 上；flash-attn 是選配；ROCm 未驗證。repo 決定不跑它，照原始碼重寫分數（P08）。
- **期限前**：DS1 已有 EXPATT。

#### A10　KVzip：query-agnostic、用重建原文打分〔部分〕
- **連結**：arXiv 2505.23416（NeurIPS 2025 Oral）；`github.com/snu-mllab/KVzip`。
- **方法與假設**：
  - 讓模型從 cached KV「重建原文」，每個 KV pair 在重建時拿到的最大 attention 就是它的重要度，低的刪。
  - 假設：重建原文需要的 KV，就是任何未來問題可能需要的 KV。
  - 分兩種：
    - 和 context 有關（每段 context 打一次分）。
    - 和 context 無關（只用一本 88K 英文書算 head 分數，借 DuoAttention 的 head 策略；「一分鐘內幾次 forward」，對比 DuoAttention 要 8 GPU 跑數小時）。
- **原文數字（條件）**：
  - KV 小 3–4×；FlashAttention 解碼延遲約快 2×。
  - 效率實驗：LLaMA3.1-8B、124K context、FP16、A100。batch：NOT_FOUND。
  - 模型：Qwen2.5-7B-1M、LLaMA3.1-8B、Gemma3-12B；最長到 170K；單張 A100 80GB。
  - **多 query**（SCBench，Qwen2.5-7B-1M，Fig. 9）：query-aware 的 baseline 在保留 90% 時就明顯掉分；KVzip 保留 30% 時還撐得住，只有 Retr.Prefix-Suffix 例外。這是文字描述，沒有逐格數字。
- **成本、限制**：
  - 打分約是一般 prefill 的 2 倍計算，額外記憶體 <2%；用 2K chunk 打分，複雜度變成線性。
  - Gemma3 沒有最佳化 kernel（README）。
  - 和 context 無關的版本壓縮率有限。
- **實作與 ROCm**：Python＋10 KB CUDA（改自 AdaKV 的 kernel）＋flash-attn 2.7.4.post1；ROCm 未驗證。kvpress 有 `KVzipPress`。
- **和分層的關係**：KVzip 是「寫一次、讀很多次」設計，和 Tiara 的還原情境完全對上。2× prefill 的打分成本可以在**寫入時**付一次，把分數存成 metadata，還原時拿來排載入順序（§4 G2）。
- **期限前**：不在 3 天內做（每題要多跑一次約 2× prefill，又要寫程式）。只有 X2 顯示「寫入時分數有預測力」時才值得做。

#### A11　SCBench：共享 context、多輪的 KV 生命週期〔部分〕
- **連結**：arXiv 2412.10319（ICLR 2025）。
- **做什麼**：
  - 12 個任務；兩種共享模式：Multi-turn（同一 session 內共用）、Multi-request（跨 session 共用）。
  - 13 種方法，8 個模型（Llama-3.1-8B／70B、Qwen2.5-72B／32B、Llama-3-8B-262K、GLM-4-9B-1M、Codestral Mamba、Jamba-1.5-mini）。
  - 4×A100，greedy BF16，HF 或 vLLM＋FA2。
  - 分類：Quest、RetrievalAttention、MagicPIG 屬於「KV cache loading」；SnapKV、PyramidKV、StreamingLLM 屬於「dropping」。
- **原文發現**：
  1. 解碼時記憶體低於 O(n) 的方法，只有第 1 個請求好，之後掉分；保留 O(n) 記憶體的稀疏方法在多個 query 下都穩。
  2. 壓到 1/4 時，低於 O(n) 的方法急掉。
  3. **「重要的 KV 在同一輪內很穩定，但不同 query 之間差很多」**（Retr.KV 的 attention map）。
- **和本題的關係**：G4 的主要證據。我們的 DS2 判準 3（第 2 輪減第 1 輪）就是照這篇設計的（14 號 §4.4）。
- **期限前**：X2 在 block 粒度上重現它的第 3 點。

#### A12　DefensiveKV／CriticalKV〔DefensiveKV 部分；CriticalKV 摘要〕
- **連結**：arXiv 2510.13334；2502.03805（ICML 2026）；`github.com/FFY0/DefensiveKV`。
- **方法與假設**：
  - 一般做法「打分→平均」靠的是「穩定性假設」（固定一組 token 一直重要），平均遇到極端情況很脆弱。
  - DefensiveKV 改成兩步、線性時間、控制最壞情況的彙整。
  - CriticalKV 用「刪掉後輸出的擾動」（W_o·V 的 L1）修正分數。
- **原文數字（條件）**：
  - 7 個任務領域、18 個資料集，20% cache：生成品質損失比最強 baseline 少 2.3×（DefensiveKV）、4.3×（Layer-DefensiveKV）。
  - 設定是**先壓縮、再給問題**（原文寫「Following the settings in (NVIDIA, 2024; Feng et al., 2025)」，前者是 kvpress），作者說這比較像多輪問答或固定前綴。
  - 開銷：A100 80GB、Llama-3.1-8B、20% cache，TTFT 和 decode 延遲與 CriticalKV 幾乎相同。
- **和本題的關係**：直接衝著「分數的平均 ≠ 實際影響」。但它沒有測多輪的第 2 輪。
- **期限前**：不做。kvpress 有 CriticalKVPress，必要時可照 P08 的方式重寫。

#### A13　KVP：用 RL 學「未來有沒有用」〔部分〕
- **連結**：arXiv 2602.10238（ICML 2026）。
- **方法**：每個 head 一個輕量 RL agent，只看 K、V，在事先算好的生成 trace 上學排名；獎勵來自「未來有沒有用」。
- **原文數字（條件）**：
  - 評測：RULER（到 128K）、多輪對話 OASST2-4k；並在 BoolQ、LongBench passage retrieval、GovReport 上零樣本測試。
  - agent 訓練：單節點 8×H100，<30 分鐘。速度：B200。
  - repo `SOTA_ANALYSIS` §3.1 記錄：收 trace 用 7 台×8 GPU，約 1.2 TB。
- **期限前**：不做（要訓練，CLAUDE.md §8）。

#### A14　The Pitfalls of KV Cache Compression〔摘要〕
- **連結**：arXiv 2510.00231（ACL 2026）；arXiv 沒有 HTML 版，只讀了摘要。
- **發現**：
  - 條件：IFEval 多指令 prompt；Llama3.1-8B、Qwen2.5-14B；5 種方法（StreamingLLM、SnapKV、TOVA、H2O、K-Norm）。
  - 某些指令在壓縮下掉得特別快，等於被模型忽略。
  - 系統 prompt 外洩是一個案例。
  - 影響因素：壓縮方法、指令順序、逐出偏差。
- **和本題的關係**：分數和「這個 token 對任務有多重要」不一致的實例。agent 的系統 prompt 在最前面，剛好碰到 sink 和逐出偏差。
- **期限前**：不做。

#### A15　Practical Online KV Cache Compaction for LLM Agents〔部分〕
- **連結**：arXiv 2608.00902（2026-08）。
- **問題**：agent 的 context 是一輪一輪長出來的，壓縮必須在「還不知道之後要問什麼」時就做（online）。
- **方法**：
  - 兩種壓縮：token 刪除（TE）、attention matching（AM）。
  - 比較幾種便宜的替代 query：
    - 這一輪結束時的邊界 query；
    - 重跑一次 prefill；
    - **延後**：等 agent 自己下一輪生成後，用那些 query。
- **原文發現（條件）**：
  - 條件：BrowseComp-Plus、WideSearch；Qwen3.5、Gemma-4（混合 attention，只壓 full-attention 層）；NVIDIA H200；PyTorch＋HF sdpa（cuDNN 後端）。
  - 立刻壓縮常常傷品質；延後到用 agent 之後的 query 可救回大部分落差。
  - 替代 query 不準時，TE 比 AM 穩。
  - TE 把 KV 減 80% 時保住大部分準確度。
  - **壓縮會改變 agent 的行為**：Qwen3.5-4B 不壓縮時，中位數 20 輪（BrowseComp-Plus）、24 輪（WideSearch）；延後壓縮的設定變成 30–39 輪、29–39 輪。搜尋 query 和之前的重複度變高（Table 3），也就是模型在把丟掉的資訊再找一次。
- **和本題的關係**：
  - 「延後決定比較好」和第一階段的結論（放置可以延後）同方向。
  - 在分層情境裡，「延後」＝先放在下層、等下一輪的問題來了再決定載入哪些，正好是 §4 G2／G4 的設定。
  - 「軌跡變長」代表丟資料的代價不只是答錯，還有多出來的輪數和 KV。
- **期限前**：X2 只做靜態兩輪，不做 agent。

#### A16　AgentKV：agent 的未來 query 是多個 phase 的混合〔部分〕
- **連結**：arXiv 2609.14872；`github.com/LiuTaowen-Tony/agentkv`。
- **方法與假設**：
  - 一般逐出用最近 token 當代表 query，假設未來的 attention 像最近的 attention。
  - agent 生成的未來 query 是 think／act／tool 等 phase 的混合，各 phase 的 query 落在不同子空間（principal angle 分析）。所以用最近 token 打分，會低估接下來其他 phase 要用的 key。
  - 做法：每個 phase 留一個小 query buffer，用它們的聯集打分；跨輪保存壓縮後的 KV，並在線上做實體壓實。
- **原文數字（條件）**：
  - 品質：Qwen3-8B／32B；BFCL、τ²-bench；6 個任務領域 × 3 種預算。平均比 R-KV 高 5.5 分、比 Tri-attention 高 5.3 分。
  - 吞吐：最多是上游 SGLang 0.5.12（FlashInfer）的 1.80×。條件：H100 NVL 或 RTX A6000；memory_ratio 0.90、max_prefill_length 8192、開 CUDA graph。
- **關鍵句**：每次壓縮後，留下的 KV 必須實體壓實、被丟的 page 要還給 allocator，「否則較小的邏輯預算不會減少 decode 時的記憶體流量」。
- **實作與 ROCm**：README 寫明要 NVIDIA GPU＋CUDA toolkit（JIT 編 CUDA／Triton kernel）→ ROCm 不能直接跑。
- **期限前**：不做。

#### A17　CommitKV：區分「暫時沒用」和「已經用完」〔部分〕
- **連結**：arXiv 2608.07855。
- **方法與假設**：
  - 這一輪 attention 低，不代表之後沒用。
  - 把完成的 agent 事件切成 page，比較同一個 page 在 tool-call commit 前、拿到 observation 後的「刪除影響」，藉此分出：
    - **dormant**：暫時不活躍，之後可能還要用；
    - **completed**：已經用完。
  - 只退休「一起刪也影響有限」的 page。
- **原文數字（條件）**：
  - 條件：8×NVIDIA H20 96GB；在 SGLang 上做；KV 預算 2048／4096／8192；6 個模型、8 個基準（推理類與 tool-use 類）。
  - 平均準確度最多比最強的壓縮 baseline 高 22.24 個百分點；峰值 KV 記憶體最多省 5.00×；end-to-end 最多快 5.62×。
- **和分層的關係**：這是目前最接近「分層語意」的 eviction 論文：dormant 應該下放到下層，completed 才該丟。它本身只有「留在 GPU」和「刪除」兩種狀態。
- **期限前**：不做（要 agent 負載）。

#### A18　IntentKV〔摘要〕
- **連結**：arXiv 2606.09916。
- **方法**：
  - 每個 session 保存跨輪意圖的 QueryMemory，用 memory-attention 打分。
  - 加一個零初始化的殘差 head（要學）。
  - 刪除是 slot map 重導：被刪的位置指向一個 dead slot，留下的 K／V、RoPE 相位、slot 編號都不動，所以和 prefix cache 相容。
- **原文數字（條件）**：
  - 8k 預算：平均峰值請求 token 數，Qwen3-8B 少 23.9%、Qwen2.5-14B 少 30.7%，準確度幾乎不掉。
  - Qwen2.5-14B 上最長的 100 個 BCP 查詢：最差情況的峰值 token 從 92.3k 降到 20.5k，原始 KV 讀取量從 411M 降到 31M。
- **期限前**：不做（要訓練）。

#### A19　EpiCache〔部分〕
- **連結**：arXiv 2509.17396（ICML 2026）。
- **方法與假設**：
  - 大部分逐出要等整段 prefill 完才做，峰值記憶體沒有上限。改成分塊 prefill，讓 cache 有界。
  - 把對話歷史分群成 episode，依 episode 做逐出。
  - 「依問題逐出會把 cache 的語意收窄到單一問題，多輪時失敗」。
- **原文數字（條件）**：
  - LongMemEval、Realtalk、LoCoMo：準確度最多高 30%；4–6× 壓縮時接近完整 cache；延遲最多低 2.4×、峰值記憶體最多低 3.7×。
  - 原文另指出：把 KVzip 放進分塊 prefill，LongConvQA 分數在各預算下都大幅下降。
  - 硬體：Fig. 1 用 LLaMA-3.2-3B＋A100。
- **期限前**：不做。

#### A20　TaskPress〔部分〕
- **連結**：arXiv 2608.03276（EMNLP 2026）。
- **方法**：
  - 用高層次的「任務說明」當 meta-query，在 prefill 時先過濾，問題還沒來就決定。
  - 用量化的 scale factor 當零成本的 outlier 訊號。
- **原文數字（條件）**：
  - 摘要沒有單一數字。
  - 一次性校準 25.2 分鐘（5 次 Gemini-2.0-Flash API 呼叫＋單張 A100 40GB 跑校準集）。
  - 效率：LLaMA-3.1-8B、64K、單張 A100 80GB、FlashAttention。
- **和本題的關係**：介於 query-aware 和 query-agnostic 之間的「任務層級」先驗。agent 寫入時通常知道任務類型。
- **期限前**：不做。

#### A21　RestoreKV〔部分〕
- **連結**：arXiv 2608.01247。
- **注意**：名稱裡的 restore 是「補回被丟的資訊」，**不是** Tiara 的「從下層還原 KV」。
- **方法**：
  - 不改原本的打分和刪除規則。prefill 後加幾個 restore token，用 LoRA 讀完整 KV 一次，產生一小段補償 cache。
  - LoRA 用完整 cache 的模型做自蒸餾，只訓練 0.4% 的參數。
- **原文數字（條件）**：
  - Qwen3-4B：同預算的 60 組設定裡改善 59 組（5 種 base 刪除法）。
  - 5% 預算時，KVzip 在 RULER-4K 從 38.2 提到 73.2。
  - KVzip+ 在 KVPress Benchmark、16× 壓縮下 RULER 86.4。
  - 32K 時一次性建構開銷 <0.5%。
  - 硬體：單張 NVIDIA RTX PRO 6000 Blackwell（96 GB）。
- **期限前**：不做（要訓練）。

#### A22　QEvict：可回復的三階逐出〔部分〕
- **連結**：arXiv 2608.05326。repo 已有卡（`PAPER_GUIDE.md`）；本地 PDF 在 `/mlsteam/data/tiara/papers/08_QEvict_arXiv2608.05326.pdf`。
- **方法與假設**：
  - 解碼時重要度會漂移，被刪的狀態之後常常拿到大量 attention。
  - 提出兩個診斷指標：Future Missed Mass（被丟掉的狀態之後拿到多少 attention）、Global LIR（不活躍的區域之後重新被用到的程度）。另外用 Selection Churn 量選擇的不穩定。
  - 以 window 為單位分三階：全精度／INT2（可升回）／刪除。第一次降級時量化一次，之後重用，避免反覆量化累積誤差。
- **原文數字（條件）**：
  - LongBench、RULER、GSM8K；3 個 instruction-tuned 模型；8×A100 80GB（部分在 GB10）。
  - repo `PAPER_GUIDE` 摘錄（本次未重核）：eager 模式 TTFT 多 0.5%、TPOT 少 9.3%；FA2 下峰值記憶體 29.54 GB 降到 20.78 GB。
- **和本題的關係**：最接近「丟只是最低一階」的論文。但它的階都在 GPU 裡（精度階），沒有 CPU、SSD，也沒有重算。
- **期限前**：不做；X1 的「丟 vs 量化」就是把它的想法搬到還原的位元組預算上。

#### A23　MemDecay〔摘要〕
- **連結**：arXiv 2607.10582。
- **方法**：依區域（系統指令、計畫、使用者輪、檢索文件、工具輸出、草稿推理）給基礎優先度和衰減率；被 attention 到就重置衰減；可釘住關鍵區域。
- **原文數字（條件）**：
  - 條件：Qwen2.5-1.5B／3B；context 只有約 450 與 1,700 token（很短）。
  - attention 壽命：系統 token 半衰期 148–189 個 decode 步；草稿 token 14–16 步。
  - 釘住系統區：每個設定都保持完整 cache 的準確度；baseline 最多只保住 24 題中的 13 題。
- **期限前**：不做（規模太小，只當「結構訊號」的旁證）。

---

## 2. Cascade：選擇性載入（不載入，但 KV 還在）

### 2.1 總表

欄位說明：
- 「沒選的 KV 在哪」：GPU＝全部留在 HBM，只少讀；CPU／SSD／CXL＝放在下層，要用時才搬。
- 「需訓練」：是否要訓練模型或額外模組（CLAUDE.md §8：不訓練）。

| # | 方法（年；venue） | arXiv | 沒選的 KV 在哪 | 打分用什麼（放在哪） | 需訓練 | 官方實作與 ROCm | 深度 |
|:--|:--|:--|:--|:--|:--|:--|:--|
| C1 | Quest（ICML'24） | 2406.10774 | GPU（只少讀，不省容量） | 每個 page 的 K min/max（GPU） | 否 | 238 KB CUDA（FlashInfer 系）→ CUDA-only | 部分 |
| C2 | InfLLM（2024；venue 未查證） | 2402.04617 | 記憶單元（可放 CPU） | 代表 token | 否 | Python＋Triton 多段 flash attention → 可能可移植，未驗證 | 部分 |
| C3 | InfiniGen（OSDI'24） | 2406.19707 | CPU | 用下一層部分權重預演 | 離線改權重（SVD skew） | Python（FlexGen 系）；YAKV 指出不支援 GQA | 部分 |
| C4 | ArkVale（NeurIPS'24） | 無 arXiv（NeurIPS proceedings） | CPU 備份，可召回 | page 的外框盒子摘要 | 否 | 程式碼 NOT_FOUND | 摘要 |
| C5 | ShadowKV（ICML'25，PMLR v267） | 2410.21465 | V 在 CPU；低秩 K 在 GPU | chunk landmark＋outlier | 否 | flash-attn＋FlashInfer＋自訂 kernel → CUDA-only | 部分 |
| C6 | MagicPIG（2024；venue 未查證） | 2410.16179 | CPU（attention 也在 CPU 算） | LSH 取樣 | 否 | Python＋C++（CPU）；GPU 端 FlashInfer 整合「進行中」 | 部分 |
| C7 | RetrievalAttention（2024） | 2409.10516 | CPU | ANNS 索引（CPU） | 否 | CUDA 12.4＋flash-attn＋FlashInfer＋自訂 kernel → CUDA-only | 部分 |
| C8 | RetroInfer（VLDB'26） | 2505.02922 | CPU | wave index | 否 | `microsoft/RetroInfer` 回 404，程式碼位置 NOT_FOUND | 部分 |
| C9 | PQCache（2024；venue 未查證） | 2407.12820 | CPU | K 的 PQ code（K-means 在 CPU） | 否 | 未查 | 部分 |
| C10 | ClusterKV（2024；venue 未查證） | 2412.03213 | CPU，以語意群召回 | 群中心 | 否 | 323 KB CUDA → CUDA-only | 部分 |
| C11 | OmniKV（ICLR'25） | 無 arXiv（ICLR proceedings） | 可卸載；不刪 | 少數 filter 層挑、其他層沿用 | 否 | Python；triton_flashdecoding 模式；ROCm 未驗證 | 摘要 |
| C12 | RocketKV（ICML'25） | 2502.14051 | 第一階段永久刪；**MT 版保留全部** | SnapKV++＋混合 top-k | 否 | Python＋flash-attn 2.6.3；ROCm 未驗證 | 部分 |
| C13 | Twilight（NeurIPS'25 spotlight） | 2502.02770 | GPU | top-p 自適應預算 | 否 | 論文說建在 FlashInfer 上（CUDA）；repo 未查 | 部分 |
| C14 | HashAttention（ICML'25） | 2412.14468 | GPU | 學到的 Hamming 碼（每 token 32 bit） | 要學映射 | 未查 | 部分 |
| C15 | SeerAttention（2024）／-R（2025） | 2410.13276／2506.08889 | GPU | 學到的 block gate | 要學 gate | Python＋CUDA＋Triton | 摘要 |
| C16 | NSA／MoBA／DSA／InfLLM-V2（原生稀疏） | 2502.11089／2502.13189／2512.02556／2509.24663 | GPU（預設） | 模型自己的 indexer／router | 預訓練或續訓 | NSA 無官方碼（fla-org 有 Triton 版）；MoBA 依賴 flash-attn；DSA 用 DeepGEMM／FlashMLA（CUDA）；InfLLM-V2 稀疏推論要 `infllmv2_cuda_impl` | 摘要 |
| C17 | SpeCache（2025）／FreeKV（2025） | 2503.16163／2505.13109 | CPU | GPU 上的低位元副本／預測式召回 | 否 | FreeKV：126 KB CUDA；SpeCache 未查 | 部分 |
| C18 | YAKV（2026，preprint） | 2604.08426 | CPU | GPU 上 1–2 bit 量化 K | 否 | 未查 | 部分 |
| C19 | NOSA（EMNLP'26） | 2510.13602 | CPU | 有區域性限制的可訓練稀疏 | 要訓練 | Python 為主＋6 KB CUDA | 部分 |
| C20 | SparseServe（2025） | 2509.24626 | DRAM＋HBM 快取 | 任一動態稀疏法 | 否 | 未查（建在 vLLM 上） | 部分 |
| C21 | HiSparse（2026；SGLang 上游） | 2608.07009 | host 全量＋固定大小 GPU 快取（LRU） | 模型的 indexer（DSA／NSA／Quest） | 否 | 融合 CUDA kernel；只在 NVIDIA | 部分 |
| C22 | SparDA（2026） | 2606.04511 | CPU | 新增的 Forecast 投影 | 要訓練（<0.5% 參數） | 380 KB CUDA → CUDA-only | 部分 |
| C23 | Fathom（2026） | 2609.17652 | host（K 的 4-bit 位元平面） | 每個 query 決定讀幾個位元平面 | 否 | 有 code 連結；未查 | 部分 |
| C24 | SAC（2026）／AVSG（2026） | 2606.19746／2609.37538 | CXL 記憶體／host | DSA indexer | 否 | SAC：SGLang（HiSparse 系）；AVSG：Ascend | 部分 |
| C25 | OasisKV（2026） | 2608.08097 | host／遠端記憶體 | 用投機解碼的 lookahead token 預取 | 否 | 建在 vLLM 上；未查 | 摘要 |
| C26 | KVSwap（2025）／LeoAM（2025）／KVDrive（2026） | 2511.11907／2506.20187／2605.18071 | 磁碟／SSD | 壓縮 K／chunk 摘要／依 attention 行為 | 否 | 未查 | 摘要 |
| C27 | 還原時只載一部分：IMPRESS（FAST'25，repo 已收）、ContiguousKV（2026）、CacheTune（2026） | —／2601.13631／2605.24022 | SSD／CPU 上的 prefix KV | probe head／attention／頻域 | 否 | 未查 | 部分 |
| C28 | The Sparse Frontier（2025；評測） | 2504.17768 | — | — | — | Python 為主 | 摘要 |
| C29 | ParisKV（ICML'26） | 2602.07721 | CPU（UVA） | 碰撞式候選＋量化內積重排 | 否 | 65 KB CUDA → CUDA-only | 部分 |

### 2.2 卡片

#### C1　Quest：依 query 只讀前 K 個 page〔部分〕
- **連結**：arXiv 2406.10774（ICML 2024）；`github.com/mit-han-lab/Quest`。
- **方法與假設**：
  - token 重不重要要看當下的 query。
  - 每個 page 存 K 的逐維最小、最大值；用當下的 query 算每個 page 能拿到的 attention 上界，只讀前 K 名。
  - 前兩層的稀疏度 <10%，所以不套用（原文 Fig. 3）。
- **原文數字（條件）**：
  - 32K context：self-attention 延遲最多降 7.03×，end-to-end 推論比 FlashInfer 快 2.23×。
  - kernel 測試：Llama2-7B、RTX4090、CUDA 12.2。
  - 準確度：LongChat-7b-v1.5-32k、Yarn-Llama-2-7b-128k；PG19、passkey、LongBench 6 個。
  - **注意**：arXiv abs 頁的摘要把這兩個數字寫反了（「2.23× self-attention、7.03× latency」）。全文 HTML 的摘要與貢獻列表都是「7.03× self-attention、2.23× 推論」。引用時以全文為準。
- **限制**：
  - **只少讀 HBM，不省容量、不省 PCIe**（KV 全在 GPU）。
  - 原文：「加速和稀疏比例成正比（等於記憶體搬移的減少）」。
  - 模型是舊的 MHA。
- **實作與 ROCm**：FlashInfer 系的 CUDA kernel → 不能直接跑。上界演算法用 PyTorch 幾行就能寫，只量品質沒問題。
- **期限前**：X4 用 Quest 的上界當 512-token block 的「摘要」，看它能不能找回 QRY 選的 block。

#### C2　InfLLM〔部分〕
- **連結**：arXiv 2402.04617；`github.com/thunlp/InfLLM`。
- **方法**：遠處的 context 切成記憶單元，每個單元用幾個代表 token；每步查跟當下相關的單元。GPU 快取大小是每步載入單元數的 2 倍，用頻率分數衰減管理。
- **原文數字**：不訓練就讓幾千 token 預訓練的模型處理到 1,024K。硬體：A100 或 A800。
- **實作與 ROCm**：全 Python（約 152 KB）。README 說 inf-llm／stream-llm 的多段 flash attention 是用 Triton 寫的 → 理論上可移植到 ROCm Triton，未驗證。
- **期限前**：不做（要接模型）。

#### C3　InfiniGen〔部分〕
- **連結**：arXiv 2406.19707（OSDI 2024）。repo 已有卡。
- **方法**：
  - 用這一層的輸入、加上下一層「部分 query 權重＋部分 K」做小規模預演，猜出下一層重要的 token，只從 CPU 預取那些。
  - 要離線把 Q／K 矩陣做 SVD「扭轉」。
- **原文數字（條件）**：在卸載式推論系統上最多比先前的 KV 管理方法快 3.00×。條件：RTX A6000 48GB；OPT、Llama-2。
- **限制**：
  - 準確度在短任務上量（repo `RELATED_WORK_WEAKNESSES`）。
  - YAKV：原生不支援 GQA。
- **期限前**：不做（我們兩個模型都是 GQA）。

#### C4　ArkVale：被逐出的 page 在 CPU 留備份，可召回〔摘要〕
- **連結**：NeurIPS 2024（proceedings 與 mlanthology 有收）；**找不到 arXiv**；程式碼 NOT_FOUND。repo `PAPER_GUIDE` 有詳卡。
- **方法**：page 填滿就非同步複製到 CPU；每個 page 存 key 的外框盒子當摘要；每步用摘要估重要度，重要的召回、不重要的逐出。
- **原文數字（repo 卡＋搜尋摘要）**：
  - 條件：LongChat-7b-v1.5-32k，1×A100 80GB。
  - 2k–4k 預算時準確度幾乎不掉；decode 延遲最多快 2.2×，batch 吞吐最多高 4.6×。
- **限制**：
  - 原文 §7 自承：prefill 時的備份延遲很難藏住。
  - 前兩層不套用。
  - YAKV：外框盒子在需要大量抽取資訊的任務上會誤判。
- **和本題的關係**：「刪了但下層有備份、可召回」的原型，也就是 G1 說的「丟＝最低一層，但留了退路」。

#### C5　ShadowKV〔部分〕
- **連結**：arXiv 2410.21465（ICML 2025，PMLR v267 sun25b）；`github.com/bytedance/ShadowKV`。repo 已有卡。
- **方法**：
  - prefill 時把 V 卸載到 CPU，GPU 只留低秩的 pre-RoPE K（rank 160）＋landmark＋outlier。
  - decode 時用 landmark 選 chunk，再把選中的 KV 即時重建或搬回。
- **原文數字（條件）**：
  - A100、Llama-3.1-8B：batch 最多大 6×、吞吐最多高 3.04×。
  - 其他模型的吞吐：Llama-3-8B-1M 2.97×、GLM-4-9B-1M 2.56×、Yi-9B-200K 2.66×。
  - 稀疏 KV 預算約 1.56%（repo 卡；搜尋到的 ICML 投影片也這樣寫）。
- **限制**：YAKV 指出 rank 160 在需要大量抽取資訊的任務上不夠，landmark 會挑錯。
- **實作與 ROCm**：flash-attn＋FlashInfer＋自訂 kernel → CUDA-only。

#### C6　MagicPIG〔部分〕
- **連結**：arXiv 2410.16179；`github.com/Infini-AI-Lab/MagicPIG`。
- **方法與假設**：
  - TopK attention 本身在某些任務會掉分，因為 attention 沒有想像中那麼稀疏。
  - 改用 LSH 取樣（有理論保證）。hash 表和 attention 計算都放在 CPU：**不把 KV 搬到 GPU，改把計算搬到 KV 所在的地方**。
- **原文數字（條件）**：
  - decode 吞吐最多高 5×（A100、L20、RTX 4090）。
  - 單張 RTX 4090、Llama-3.1-8B-Instruct、96K context，decode 延遲 54 ms。
- **限制**：CPU 算力成為瓶頸。
- **實作與 ROCm**：Python＋C++（CPU 端）。CPU 部分理論上和 GPU 廠商無關，但指令集需求 NOT_VERIFIED；README 寫 FlashInfer 整合「進行中」。

#### C7／C8　RetrievalAttention、RetroInfer〔部分〕
- **連結**：arXiv 2409.10516；2505.02922（VLDB 2026）。
- **方法**：
  - RetrievalAttention：在 CPU 記憶體建 KV 的 ANNS 索引；處理 query 和 key 分布不同（OOD）的問題。
  - RetroInfer：wave index（三段式 attention 近似、有界誤差估計、分段分群）＋GPU／CPU 緩衝管理（wave buffer）。
- **原文數字（條件）**：
  - RetrievalAttention：只需讀 1–3% 的資料就接近完整 attention；單張 RTX4090 24GB 跑 8B 模型、128K，每 token 0.188 秒。另有 A100 結果。
  - RetroInfer：
    - 120K 時 decode 吞吐最多是完整 attention 的 4.4×；1M 時最多是稀疏 baseline 的 12.2×。
    - 分析條件：Llama3-8B-1048K、A100 80GB。
    - 原文：A100 在 128K 時最多只能跑 batch 4。
- **實作與 ROCm**：
  - `microsoft/RetrievalAttention`：CUDA 12.4、flash-attn、FlashInfer 0.2.4、自訂 kernel → CUDA-only。
  - RetroInfer：`microsoft/RetroInfer` 回 404。RetrievalAttention 的 README 提到 GPU–CPU 搬運 kernel，是不是 RetroInfer：未查證。

#### C9　PQCache〔部分〕
- **連結**：arXiv 2407.12820。
- **方法**：prefill 時對每層、每個 head 的 K 做 product quantization（K-means 在 CPU）；decode 時用 PQ code 估重要 token，再抓對應的 KV。
- **原文數字（條件）**：
  - InfiniteBench 上比既有方法高 4.60%。
  - 預設硬體：RTX 4090 24GB＋2×Xeon Gold 6330＋500GB DRAM；原文寫「PCI-e 1.0（x16）」。
- **期限前**：不做。

#### C10　ClusterKV〔部分〕
- **連結**：arXiv 2412.03213；`github.com/sjtu-zhao-lab/ClusterKV`。
- **方法與假設**：依位置切 page 來召回會掉品質，改成以「語意群」為單位召回。
- **原文數字（條件）**：32K context、1k–2k 預算時準確度幾乎不掉；延遲最多快 2×、decode 吞吐最多高 2.5×。硬體：NVIDIA Ada 6000。
- **和分層的衝突**：語意群在儲存上不連續 → 從 SSD 讀會有讀取放大（見 C27 ContiguousKV）。
- **實作與 ROCm**：323 KB CUDA → CUDA-only。

#### C11　OmniKV〔摘要〕
- **連結**：ICLR 2025（iclr.cc 海報頁、proceedings PDF）；找不到 arXiv；`github.com/antgroup/OmniKV`。
- **方法與假設**：
  - 作者認為「這一步的 attention 分數不能可靠預測之後哪些 token 會重要」，所以**完全不刪**。
  - 同一步裡，相鄰層的重要 token 高度相似：少數 filter 層挑，其他層沿用；完整 KV 保留（可卸載）。
- **原文數字（ICLR 摘要）**：不掉效能下快 1.68×；配合卸載，KV 記憶體最多省 75%；Llama-3-8B 在單張 A100 上可處理的長度從 128K 提到 450K。
- **實作與 ROCm**：Python 為主；只有 `triton_flashdecoding` 模式支援 OmniKV；依賴 flash-attn；ROCm 未驗證。

#### C12　RocketKV（含 RocketKV-MT）〔部分〕
- **連結**：arXiv 2502.14051（ICML 2025）；`github.com/NVlabs/RocketKV`。
- **方法**：兩階段：第一階段粗粒度**永久**刪除 prompt token（SnapKV++），第二階段做細粒度的混合 top-k 稀疏 attention。
- **原文數字（條件）**：壓縮比最多 400×；end-to-end 最多快 3.7×；decode 峰值記憶體最多省 32.6%。條件：NVIDIA A100，對比完整 KV。
- **RocketKV-MT（多輪版）**：
  - 原文：多輪時，之前的輪被刪的 token 之後可能必要（引 SCBench），所以**第一階段不刪，全部留給之後的輪**。每一輪的解碼仍只在第一階段篩過的集合裡選。
  - 原文直說：這樣「省到一樣的記憶體流量，但不省儲存」。
- **和本題的關係**：「不載入 vs 淘汰」與「流量 vs 容量」最乾淨的一個例子（§3）。
- **實作與 ROCm**：Python（約 747 KB）＋flash-attn 2.6.3；ROCm 未驗證。

#### C13　Twilight〔部分〕
- **連結**：arXiv 2502.02770（NeurIPS 2025 spotlight）。
- **方法**：把 top-p（nucleus）套到稀疏 attention，讓每步的預算自適應。可以疊在任何稀疏 attention 上。
- **原文數字（條件）**：最多剪掉 98% 的 token；self-attention 最多快 15.4×；end-to-end 每 token 延遲最多快 3.9×。條件：單張 A100；LongBench 的 Qasper、GovReport、LCC，prompt 10k–30k。
- **和分層的關係**：預算隨步變動 → 每步要搬的位元組也跟著變，在有爭用的分層裡比較難預測。原文沒有評估這一點。
- **實作**：論文說建在 FlashInfer 上，反量化沿用 QServe／FasterTransformer → CUDA。repo 沒查。

#### C14　HashAttention〔部分〕
- **連結**：arXiv 2412.14468（ICML 2025）。
- **方法**：把找重要 token 當成推薦問題；學一個映射，把 q、k 編成 Hamming 碼，用位元運算挑；每 token 額外 32 bit。
- **原文數字（條件）**：
  - 用一般資料訓練：token 最多少 16×；任務專屬 fine-tune：32×。
  - A100、32× 稀疏：attention 延遲在 GPT-FAST 最多快 4.3×、FlashDecode 最多快 2.54×；GPT-FAST 吞吐最多高 3.12×。
  - Fig. 1 條件：Llama-3.1-8B-Instruct、RULER@32K、batch＝1。
- **期限前**：不做（要學映射）。

#### C15　SeerAttention／SeerAttention-R〔摘要〕
- **連結**：arXiv 2410.13276；2506.08889；`github.com/microsoft/SeerAttention`。
- **方法**：學一個 block 層級的 gate（類似 MoE），配 block-sparse FlashAttention kernel。原版針對 prefill，-R 版針對長推理的 decode。
- **實作與 ROCm**：Python＋CUDA＋Triton。要學 gate，或用官方釋出的 gate；我們兩個模型有沒有：NOT_VERIFIED。

#### C16　原生稀疏：NSA、MoBA、DSA、InfLLM-V2〔摘要〕
- **NSA**（2502.11089）：
  - 粗粒度壓縮＋細粒度選擇＋滑動視窗，可端到端預訓練；64K 時 decode、forward、backward 都比全 attention 快。
  - 沒有官方程式碼；fla-org 有 Triton 重寫版（README 的範例用 `device='cuda'`，ROCm 版 PyTorch 會對到 HIP），ROCm 未驗證。
- **MoBA**（2502.13189）：MoE 式的 block 路由；已用在 Kimi 的長 context 服務；kernel 依賴 flash-attn 2.6.3。
- **DSA**（DeepSeek-V3.2，2512.02556）：
  - lightning indexer＋細粒度 token 選擇。
  - vLLM Day-0 支援走 DeepGEMM／FlashMLA 的 CUDA kernel。
  - SGLang 有給 MI350X／MI355X 的 ROCm image（搜尋結果）；MI300X：NOT_VERIFIED。
  - 模型規模不適合我們的 2 張卡。
- **InfLLM-V2**（2509.24663）：
  - 密集／稀疏可切換；比密集快 4×，保住 98.1%／99.7% 效能（長 context 理解／CoT）；MiniCPM4.1-8B 已釋出。
  - 稀疏推論需要 `infllmv2_cuda_impl`（模型卡）→ CUDA。
- **和本題的關係**：
  - 原生稀疏讓「選擇性載入」變成預設模式。
  - indexer 的 key 必須常駐快層（一份 metadata 層）；HiSparse、SAC、AVSG 都建在這上面。
  - 原生稀疏模型要預訓練或續訓，3 天內不可行。

#### C17　SpeCache、FreeKV〔部分〕
- **SpeCache**（2503.16163）：
  - 完整 KV 卸載到 CPU；GPU 上留一份**低位元 KV 副本**來量重要度。
  - 用投機的下一個 token 預測下一步要的 KV，先預取。
  - LongBench、NIAH 上 10× 壓縮仍不需重訓。硬體：NOT_FOUND。
- **FreeKV**（2505.13109）：
  - 預測式召回移出關鍵路徑＋細粒度修正；CPU／GPU 用不同版面，避免碎片化傳輸；雙緩衝串流召回。
  - 最多比 SOTA 的 KV 檢索方法快 13×；附錄有 Ascend NPU 評估。
  - 126 KB CUDA → CUDA-only。
- **和第二階段的關係**：「低位元副本當索引」＝把第二階段的精度階層拿來當選擇用的 metadata。Fathom（C23）把這個想法推得更極端。

#### C18　YAKV／KV Cache Offloading for Context-Intensive Tasks〔部分〕
- **連結**：arXiv 2604.08426（preprint）。本地 PDF：`/mlsteam/data/tiara/papers/yakv2026_arXiv2604.08426.pdf`。repo 已有卡（`PAPER_GUIDE.md`）。
- **方法**：
  - 做了 Text2JSON 評測（需要從原文抽很多資訊）。
  - 找出兩個失敗原因：K 的低秩投影不準、group landmark 不可靠。
  - YAKV：從 ShadowKV 出發簡化——不分組、不用 landmark，在 GPU 留極度量化（HIGGS 1–2 bit）的 K 來決定要從 RAM 載哪些 KV，也不需要 outlier。
- **評估方式**：
  - 所有方法用**同樣的每 token PCIe 預算（GiB）**比較：以 ShadowKV 推薦參數為基準，調 YAKV、LRQK、InfiniGen 到同預算。這是「等搬運量比較」的範例，X1 照這個做。
  - InfiniGen 原生不支援 GQA，原文另外處理。
- **原文數字**：Llama 3、Qwen 3 上既有方法大幅掉分；YAKV 在多個模型家族明顯改善。吞吐在單張 H200 上量（合成 65K＋強制解碼；真實資料含 prefill＋continuous batching），細節表格本次沒抄。
- **原文引述**：KV 量化在實務上很普遍；KV eviction 在產業部署中不常見（引 Li et al. 2025b），因為某些任務會大幅掉分。
- **期限前**：X4 的「摘要保真度」就是 YAKV 的 Takeaway B 在我們 block 粒度上的版本。

#### C19　NOSA〔部分〕
- **連結**：arXiv 2510.13602（EMNLP 2026 main）；`github.com/thunlp/NOSA`。
- **方法與假設**：
  - 不訓練的卸載法在長生成時有訓練／推論不一致的問題；可訓練的稀疏 attention 選的 block 又沒有限制，可能逼出大量 CPU→GPU 傳輸。
  - NOSA 在訓練時就**限制 CPU–GPU 的傳輸量**（利用選擇的區域性）；搭配推論系統 NOSI。
- **原文數字（條件）**：decode 吞吐最多是 FullAttn 的 5.04×、InfLLMv2 的 1.92×、ShadowKV 的 1.83×。條件：1B／3B／8B 模型；A800-80GB（Fig. 2）。
- **期限前**：不做（要 NOSA 訓練過的模型）。

#### C20　SparseServe：稀疏 attention 下 batch 太大反而變慢〔部分〕
- **連結**：arXiv 2509.24626。
- **問題**：用了動態稀疏 attention，瓶頸從 HBM 頻寬變成 HBM 容量；沒被選的 KV 也得留在 HBM，batch 上不去。
- **方法**：
  - 碎片化感知傳輸：GPU 直接載入（FlashH2D）、CPU 協助存回（FlashD2H）。
  - 依工作集大小即時調 batch。
  - 分層段的 prefill：prefill 時 HBM 只用到一層。
- **原文數字（條件）**：
  - 條件：A100 40GB、PCIe Gen4（32 GB/s）、AMD EPYC 7J13、256GB DRAM；LWM-7B、Llama3-8B-262K；Poisson 到達。
  - 平均 TTFT 最多低 9.26×；生成吞吐最多高 3.14×。
  - **動機實驗**：batch 從 2 到 6，吞吐升 2.07×；從 6 到 12 反而**降 1.73×**，因為每次迭代平均載入的 KV block 數增加 21.36×（HBM 快取互搶）。
  - 零散小 block 用標準 cudaMemcpy 只有 <4 GB/s 的有效頻寬。
- **和本題的關係**：G3 的核心證據：選擇性載入的收益在多請求爭用下會反轉。

#### C21　HiSparse〔部分〕
- **連結**：arXiv 2608.07009；LMSYS 部落格 2026-04-10。
- **方法**：
  - 精確（不改模型輸出）、和 indexer 無關的分層 KV。
  - 每個請求的完整 KV 放 host，GPU 只放固定大小的快取。
  - 一個融合 CUDA kernel 在 decode 的 CUDA graph 裡完成「命中檢查＋LRU 替換＋host→device 抓取」。
  - 跨層共用選擇的模型，做精確的逐層預取。
- **原文數字（條件）**：
  - DSA、NSA、Quest 三種稀疏；H200、B200、GH200；長 context 下峰值生成吞吐最多高 4.7×。
  - 沒有 IO 的 oracle 顯示：解析機制本身不增加每 token 成本，**唯一的代價是 host–device IO**。
  - 部落格：GLM-5.1-FP8、32k 輸入／8k 輸出、8×H200、256 併發時吞吐超過 baseline 3×；2×H20 的 PD 分離掃描中，長 context 最多 5×。
- **限制（部落格）**：
  - 低併發時多出的 IO 會造成小額開銷。
  - 目前只支援 DSA 家族（DeepSeek-V3.2、GLM-5.1），屬實驗性。
  - 沒有提到 AMD／ROCm。
- **和本題的關係**：「不載入但可拿回」的生產級版本，已合進 SGLang 上游；低併發有開銷、高併發才賺（G3）。

#### C22　SparDA〔部分〕
- **連結**：arXiv 2606.04511；`github.com/NVlabs/SparDA`。
- **方法**：
  - 每層多一個 Forecast 投影，預測**下一層**要的 KV block，讓 CPU→GPU 預取和這一層的計算重疊。
  - 每個 GQA group 一個 Forecast head；只訓練 Forecast（<0.5% 參數）。
- **原文數字（條件）**：
  - 兩個稀疏預訓練的 8B 模型：prefill 最多快 1.25×、decode 最多快 1.7×（對比稀疏＋卸載 baseline）。
  - 因為單卡能跑更大的 batch，decode 吞吐最多是不卸載稀疏 baseline 的 5.3×。
  - A100 吞吐結果。
- **期限前**：不做（要訓練、CUDA）。

#### C23　Fathom：每個 query 決定讀幾位元〔部分〕
- **連結**：arXiv 2609.17652。
- **問題**：百萬 token、多 session 同時在線時，KV 和索引都在 host；為了 top-k 掃描全部 n 個 key 的流量成為瓶頸。
- **方法**：4-bit K 以 channel 為主的位元平面存放；前 t 個平面剛好是 t-bit 量化；每個 query 依 channel 重要度做 reverse water-filling，分配要讀的位元。
- **原文數字（條件）**：
  - A100、Qwen3-8B、100 萬 token：decode 一步的 GPU 時間比 136-bit 掃描（Double Sparsity、Loki、SparQ r＝32）快 1.67×。
  - 同樣 GPU 時間下，比 SparQ 的 68-bit 讀取少 18% 位元組，7 個設定中有 6 個 attention 誤差更低。
  - 真實 coding agent session：92 bit 就達到最準的 136-bit 掃描的 step agreement。
  - 原文自承：索引放在 GPU 記憶體時**不會比較快**。
- **和第二階段的關係**：精度階層可以拿來省「索引掃描」的位元組，不只省 KV 本體。

#### C24　SAC、AVSG〔部分〕
- **SAC**（2606.19746）：
  - 用 CXL 的 cache-line 粒度 load／store，只抓需要的 top-k KV。
  - DeepSeek-V3.2（AWQ 4-bit）、8×H20＋2TB CXL 記憶體池，對比 RDMA 式整段抓取：吞吐 2.1×、TTFT 低 9.7×、TBT 低 1.8×。
  - SGLang 上的 HiSparse 系實作。
- **AVSG**（2609.37538）：
  - Ascend NPU；DSA 相鄰步選到的 KV 有大量重疊 → 在 device 上重用。
  - 向量化 hash 比對比逐一比對快 2.80×；只搬未命中的，有效 H2D 頻寬最多高 34.44×；8K 個 entry 的緩衝在真實請求上命中 94.83%。
  - 生產資料集的 5P1D 服務：TPOT 少 39%、吞吐 1.27×。
- **對照**：Dynamic Sparse Attention: Access Patterns（2603.13430）量到「一個 KV 留在某層 top-k 集合裡的步數」呈指數衰減，平均不到 2 步（作者在幾個開源 backbone 上自建 DSA 式 indexer 記錄）。這和 AVSG 的 94.83% 命中**定義不同**：一個是精確 top-k 集合的持續，一個是 LRU 緩衝＋較大容量的命中。兩者不能直接比。

#### C25　OasisKV〔摘要〕
- **連結**：arXiv 2608.08097（本地 PDF 在 `papers_ssd/`）。
- **方法**：用投機解碼草擬的 lookahead token 預測下一步的重要 KV block，從 host 或遠端記憶體預取到 HBM。建在 vLLM 上。
- **原文數字（條件）**：
  - 2,048-token KV 預算時，準確度和全 attention 差在 0.7 分內。
  - 推理負載：吞吐是 dense vLLM 的 1.69×（準確度損失 0.1 分）；多 GPU 長 context 最多 2.1×。
  - PD 分離：約 2× dense 吞吐；每個請求少收 6.5–9.7× 的 KV；decode 節點 host 記憶體少 2.2–2.6×。
- **和本題的關係**：「預測下一步要用什麼」用的是**未來的 token**（投機），不是寫入時的分數。

#### C26　KVSwap、LeoAM、KVDrive：KV 放磁碟＋選擇性預載〔摘要〕
- **KVSwap**（2511.11907）：
  - 端側裝置（CPU／GPU 共用記憶體），完整 KV 放磁碟；記憶體裡用很小的 metadata 預測要預載哪些。
  - 讀取模式配合裝置特性。原文指出 NVMe、UFS、eMMC 控制器有讀取放大（讀整個 NAND page）。
- **LeoAM**（2506.20187）：
  - 單張消費級 GPU 上的 GPU–CPU–磁碟分層。
  - 依各層 attention 偏斜度切變長 chunk；磁碟上存每個 chunk 的「KV 摘要」，抽摘要而不是讀完整 KV。
  - 平均延遲快 3.46×；較大 batch 最多 5.47×。repo 有本地 PDF。
- **KVDrive**（2605.18071）：GPU、DRAM、SSD 三層，從系統面協調放置、pipeline、跨層搬運；吞吐最多比 SOTA 高 1.74×。repo 有本地 PDF。
- **和本題的關係**：三篇都是 **decode 時**的選擇性載入，不是一次性的還原。

#### C27　還原時只載一部分：IMPRESS、ContiguousKV、CacheTune〔部分〕
- **IMPRESS**（FAST 2025；repo `lit_A` 已核 venue）：
  - 依重要度做多層 prefix KV 儲存。
  - 多存一份 probe head 的 K（約 1.2%），還原時便宜地挑重要 KV（repo `lit_A_summary` §2 N4）。
- **ContiguousKV**（2601.13631）：
  - 問題 1：剪枝演算法以 token 為單位挑，儲存系統卻以固定大 block 做 I/O（IMPRESS 用 64-token、1.8 MB 的 chunk），為了幾 KB 有用的資料要讀幾 MB → 讀取放大。
  - 問題 2：「先找出重要 token、再載入」的先後依賴，造成 I/O 和計算的空檔。
  - 做法：
    - ContiguousChunk：讓剪枝粒度和 I/O 粒度一致。
    - 利用相鄰層的重要 chunk 很相似，做跨週期的非同步預取。
    - attention 引導的記憶體保留。
  - 原文數字：Re-Prefill 階段比 IMPRESS 快 3.85×。條件：Qwen2.5 系列；A800 80GB；Samsung 990 Pro NVMe（實測讀取峰值約 7.45 GB/s）；PCIe 4.0 ×16（實測雙向約 32 GB/s）。
- **CacheTune**（2605.24022）：
  - 非前綴的 KV 重用：離線用頻域分析找出對跨 chunk attention 最關鍵的 KV；線上只重算這些，其他沿用；只傳沒被重算的 KV（稀疏傳輸）；依硬體自動調重算比例。
  - 原文數字：TTFT 快 3.72–4.86×；吞吐高 3.93–6.21×；cache 放在 I/O 很慢的 SSD／HDD 時仍有 2.34–2.36× 的 TTFT 加速。
  - 條件：Mistral-v0.3-7B、Llama-3-8B、Qwen2.5-32B；2×A100-40GB 與 2×RTX 4090 兩個平台；PCIe Gen3 ×16；分層實驗固定重算 15%。
- **和本題的關係**：**還原時依重要度只載一部分**已經有前作（G2）。但三篇都沒有和 Cake 的「前段算、後段載」合起來，也都沒量 GPU 忙的情況。

#### C28　The Sparse Frontier〔摘要〕
- **連結**：arXiv 2504.17768。
- **做什麼**：至今最大規模的不訓練稀疏 attention 評測：6 種方法、多個模型家族與大小、最長 128K、稀疏度到 0.95、9 個任務。
- **發現**：
  - 同成本下，大的稀疏模型贏小的密集模型。
  - prefill 時細粒度的逐 query 估計不實際（估計本身的成本＋缺少能把細粒度稀疏換成牆鐘時間的 kernel）。
  - decode 時 token 對 page 的選擇可行。
  - 越長的序列越耐得住高稀疏 → 固定預算不是最佳。

#### C29　ParisKV〔部分〕
- **連結**：arXiv 2602.07721（ICML 2026）。
- **方法**：GPU 原生、對分布漂移穩健的 KV 檢索（碰撞式候選＋量化內積重排）；百萬 token 時用 UVA 存取 CPU 上的 KV，依需要抓 top-k。
- **原文數字**：
  - 百萬 token 時 decode 延遲比 MagicPIG 低 17×、比 PQCache 低 44×（摘要；內文寫 17×、45×，兩處不一致，照抄）。
  - 在完整 attention 跑得動的範圍內，吞吐最多高 2.8×。
  - 硬體：NOT_FOUND（grep 沒找到）。

---

## 3. 整理：不載入 vs 淘汰、搬運量 vs 邏輯量、分數 vs 任務重要度

### 3.1 沒被選的 KV 去哪了

| 狀態 | 例子 | 之後還拿得回來嗎 | 拿回來的成本 | 省了什麼 |
|:--|:--|:--|:--|:--|
| **還在 GPU，只是這步不讀** | Quest、Twilight、HashAttention、SeerAttention、原生稀疏的預設、RocketKV-MT | 下一步就能讀 | 0 | 只省 HBM 讀取與 attention 計算；**不省容量** |
| **在下層（CPU／CXL／SSD），可召回** | ArkVale、ShadowKV（V）、InfLLM、RetrievalAttention、RetroInfer、PQCache、ClusterKV、OmniKV、SpeCache、FreeKV、YAKV、HiSparse、SparseServe、SAC、OasisKV、KVSwap、LeoAM、KVDrive | 能 | 下層→GPU 的位元組 ÷ 頻寬，還要先有快層的索引或摘要 | GPU 容量；每步只搬選中的 |
| **低精度留著，可升回** | QEvict（INT2） | 能，但有量化誤差 | 反量化 | 容量（部分） |
| **刪了，但可重算** | Pensieve、CacheFlow 的「被逐出只能重算」、Tiara 動作空間的 DROP＋重算 | 能，而且無損 | 重算成本隨位置增加；**中段 block 要精確重算，需要它前面全部的 KV** | 容量與儲存 |
| **刪了，不再拿回（有損）** | H2O、TOVA、SnapKV、PyramidKV、Ada-KV、KVzip、DefensiveKV、DuoAttention 的 streaming head、RocketKV 第一階段、所有 agent 壓縮（A15–A23） | 否 | 品質損失 | 容量、儲存、讀取、計算 |

**對 Tiara 最重要的兩點**〔判讀〕：
1. **在 Cake 式還原下，前段的「刪了可重算」和「有損刪除」分不開**。
   - 計算線本來就從 token 0 往後算，前段 block 一定會被精確算回來。
   - 所以前段 block 不存、不載，**品質完全不變，時間也不變**（14 號 §4.1 已寫：Cake 會把丟在前面的算回來）。
   - 有損刪除只對**載入區**（會合點之後）有意義。
2. **想在還原時「不載入但之後還能拿」，快層必須先有東西可以打分**。
   - 要嘛把 K 全部載入（BF16 下約佔一半的位元組），要嘛留一份摘要：Quest min/max、ShadowKV 低秩 K、YAKV 1–2 bit K、SpeCache 低位元副本、Fathom 位元平面、IMPRESS probe-head K。
   - 摘要放在哪一層、多大、準不準，決定了「選擇性還原」划不划算。YAKV 說摘要在需要大量抽取資訊的任務上會挑錯（G2、X4）。

### 3.2 實體上真的少搬、少佔，還是只是邏輯上少看

| 證據 | 內容 | 出處 |
|:--|:--|:--|
| 遮罩不省記憶體 | kvpress 逐 head 的 press 用假 key 讓 attention 變 0：「does not reduce peak memory and slightly increases runtime」 | `kvpress/attention_patch.py`（10/19 讀） |
| 不壓實就不省 | AgentKV：每次壓縮後必須實體壓實、歸還 page，否則較小的邏輯預算不會減少 decode 的記憶體流量 | 2609.14872（設計段落） |
| page 內碎片 | vToken：token 粒度的逐出遇上 PagedAttention 的固定 block，留下大量回收不了的記憶體；重新打包後，每請求保留的 block 少 27.2%–72.3%，SLA 下吞吐最多 1.37×（vLLM；H2O、Random、Scissorhands） | 2608.13263 摘要 |
| head 預算不均造成碎片 | Tangram：回收碎片最多佔 prefill 時間 25%；decode 延遲最多膨脹 1.7×，或每步花 15–20% 重新規劃 | 2606.06302 摘要 |
| 讀取放大 | ContiguousKV：64-token、1.8 MB 的 I/O 單位只為幾 KB 有用的資料；KVSwap：嵌入式儲存讀整個 NAND page | 2601.13631；2511.11907 |
| 零散小傳輸 | SparseServe：零散 KV block 用 cudaMemcpy 只有 <4 GB/s，PCIe 峰值 32 GB/s | 2509.24626 |
| 流量省了、容量沒省 | RocketKV-MT：「省到一樣的記憶體流量，但不省儲存」；Quest：只省 HBM 讀取 | 2502.14051；2406.10774 |
| 壓縮讓輸出變長 | Rethinking KV Compression：壓縮可能讓輸出變長，end-to-end 延遲反而增加；FlashAttention／PagedAttention 的實作沒為壓縮後的 cache 最佳化 | 2503.24000（MLSys'25）摘要 |
| 壓縮讓 agent 多走幾輪 | Online compaction：Qwen3.5-4B 的軌跡從中位 20／24 輪變成 30–39／29–39 輪，搜尋重複度變高 | 2608.00902 |
| 索引在 GPU 時不賺 | Fathom 自承：索引常駐 GPU 時不會比較快 | 2609.17652 |
| 低併發有開銷 | HiSparse：低併發時多出的 IO 是額外開銷 | LMSYS 部落格 |

**對 repo 的意涵**〔判讀〕：
- 14 號 D 的遮罩實作（`KVModel.key_keep`）只能證明**品質**。
- 只有 **B 粒度**（512-token block、所有層與 head 同一組）的刪除能換成「少載的位元組」；T、P 粒度的品質結果不能拿來推速度。14 號 §8.3 已這樣規定，本節的文獻支持這個規定。
- 想從 T 粒度推論位元組，至少要先解決 vToken／Tangram 說的碎片與壓實問題。

### 3.3 attention 分數 ≠ 對任務的重要度

| 落差 | 證據 |
|:--|:--|
| 不同問題要的 token 不同 | SCBench：重要 KV 在同一輪內穩、不同 query 之間差很多；KVzip：query-aware 的方法在多 query 下保留 90% 就掉分 |
| 重要度隨生成漂移 | QEvict 的 Future Missed Mass；DefensiveKV：「穩定性假設」很脆弱；DSA access patterns：top-k 持續平均不到 2 步 |
| 這一輪不重要 ≠ 之後沒用 | CommitKV（暫時沒用 vs 已用完）；AgentKV（未來 query 是多個 phase 的混合）；Online compaction（立刻壓縮傷品質） |
| attention 權重不等於對輸出的影響 | VATP（Value 也要算）；CriticalKV、kvpress 的 DropKVPress（以輸出擾動打分）；Expected Attention（乘 ‖V‖） |
| 指令、結構位置 | Pitfalls（某些指令被忽略、系統 prompt 外洩）；MemDecay（區域先驗，釘住系統區） |
| 覆蓋率 | K-VEC：掉分和「唯一 token 覆蓋率下降」有關 |
| full attention 未必最好 | Make Each Token Count：長 context 中無關 token 會稀釋 attention，學出來的逐出可以比完整 cache 好 |
| attention 不夠稀疏 | MagicPIG：TopK 本身在某些任務掉分 |
| 選擇器本身不準 | YAKV：低秩 K、landmark 在需要大量抽取資訊的任務上挑錯 |
| **repo 自己** | 09 海選、43 複賽：用「寫入時的 attention」挑哪些 block 壓成低精度，**沒有贏隨機**（`docs/precision_20261013/08_report_mi300x.md` §2.9，14 號 §4.6 引用） |

---

## 4. 研究缺口

> 判斷「有沒有人做過」的搜尋範圍：arXiv 與網路搜尋（WebSearch 約 12 次）＋本檔列出的論文全文 grep。**查不到不等於沒有**（CLAUDE.md §1 第 7 條）；寫「沒找到」的地方都是這個意思。

### G1　被丟的 KV 在還原時才發現需要：drop 是不是只是最低一層的放置狀態

- **有沒有人做過**：
  - 把「刪」當可回復的一階：QEvict（GPU 內的精度階）、ArkVale（CPU 備份＋召回）、SpeCache、FreeKV（完整 KV 在 CPU）。
  - 多層＋選擇：KVDrive、LeoAM、KVSwap（decode 時）；IMPRESS、ContiguousKV（還原時，但被剪掉的 token 直接不用）。
  - 「刪了只能重算」：CacheFlow（`SOTA_ANALYSIS` §2.2）、Pensieve。
  - 「把策略寫成狀態轉移」：KV Survey 附錄 G.2 只是提出方向，沒有實作（repo `PAPER_GUIDE`）。
  - repo D2：vLLM 0.28 接慢的 fs 層時，回來的請求寧可等 fs（最多 83 s）也不重算（同樣 33K 重算約 2–7 s；`docs/research_20261010_directions/D2_vllm_drop.md` §4.3）。
  - 第一、二階段：只載 ÷ 只算的比，真實 CPU 層是 0.007–0.022，b48 是 0.40–2.6（14 號開頭引 11 號 §3c）。
  - **沒找到**：
    - 在 Cake 式還原下，把「有損刪除」「刪除＋重算」「低精度」「慢層」放在**同一個位元組預算**下比較品質與時間。
    - 「被丟的中段 block 若要精確救回，需要它前面全部的 KV」這個不對稱，被當成放置決策的一部分。
- **可能的新意**〔判讀〕：
  - 刪除的代價隨位置不對稱。前段：刪除＋重算是無損的，而且計算線本來就會算。後段：只能有損，或從慢層讀回，或把計算線一路延長到那個位置（成本隨位置增加）。
  - 所以在 Tiara 的動作空間裡，「有損刪除」只該出現在載入區；它值不值得，要看它在等位元組下有沒有贏過低精度。
  - 文獻說量化在實務上更常用（YAKV），第二階段也看到 32K 時好的低精度格式不傷答對率。如果等位元組下量化全面勝出，「有損刪除」就**不是**一個有用的層；這本身是乾淨的負面結果。
- **最小證偽實驗**：§5 X1。
- **什麼時候停止投入**：
  - 三個位元組預算（約 1/2、1/4、1/8）下，最好的寫入時刪除法（DS1 選出的前 2 名）都沒有贏過同位元組的均勻低精度：配對差的 CI 上界 ≤ 0，或贏不到 ε。
  - 這時寫「在本設定下，有損刪除被量化支配」，第三階段不再做刪除的放置策略。
  - 只有在 ORC／QRY 以外的方法贏過量化時才繼續。

### G2　還原時用重要度決定載入順序，或只載一部分（selective restore）

- **有沒有人做過**：
  - 還原時只載一部分：IMPRESS（probe head）、ContiguousKV（剪枝與 I/O 粒度對齊，Re-Prefill 3.85×）、CacheTune（只重算關鍵 token、其他稀疏傳輸）、SparseX（2606.01751，非前綴片段只重算稀疏 token 集合，只看了摘要）、MTDS（整段的部分載入，repo `SOTA_ANALYSIS` §2.6）。
  - decode 時的版本：Quest、ShadowKV、YAKV、HiSparse 等（§2）。
  - 寫入時一次打分、之後多次重用：KVzip、TaskPress。
  - **沒找到**：
    - 「依重要度排載入順序」和 Cake「從後往前載、從前往後算」的結合：載入線改成依分數順序、可隨時中止（anytime）的還原。
    - 還原時 query-aware 打分要先有 K 的代價：摘要放哪一層、要先載多少位元組。
- **可能的新意**〔判讀〕：
  - 把第二階段的精度階層重新定位成**索引**：低位元 K 當摘要（SpeCache、Fathom、YAKV 走這條路，但都是 decode 時）。
  - 還原時：先載低位元 K（約 1/8 的 K）→ 用下一輪的問題打分 → 依分數載入全精度 KV，時間到就停。
- **最小證偽實驗**：§5 X2＋X4（先證明分數與摘要有預測力；不賺就不用做速度）。
- **什麼時候停止投入**：
  - 寫入時的分數（SNAP、EXPATT、KNORM、KEYDIFF、H2O 近似）對第 2 輪答案所在 block 的召回率，和隨機的 CI 重疊；**而且**
  - Quest min/max 摘要在 512-token block 上對 QRY 前 k 名的召回也≈隨機。
  - 兩者都成立 → 還原時能用的只剩位置（就是現在的 Cake 規則），G2 停止。

### G3　GPU 忙、多請求時，選擇性載入的收益會不會變

- **有沒有人做過**：
  - decode 時：SparseServe（batch 6→12 吞吐反降 1.73×，HBM 快取互搶）、HiSparse（低併發有額外開銷，高併發才賺）、DSA access patterns（工作集零散、難預取）、AVSG（相鄰步重疊可重用）。
  - 壓縮在真實框架吞吐不佳：Rethinking（MLSys'25）。
  - 還原時的 GPU 使用率掃描：Cake（12.5%–100%）、CacheFlow（批次 DP）。repo `SOTA_ANALYSIS` §2.1／§2.2 已整理。
  - **沒找到**：還原階段（prefill 前一次性還原）在 GPU 爭用下，選擇性載入的淨收益——要扣掉打分本身也要用 GPU。
- **可能的新意**〔判讀〕：
  - Cake 的會合點會隨 GPU 忙碌往前移：GPU 越忙，載入區越大。
  - 所以同樣丟 50% 的 block，**落在載入區的比例**會隨忙碌度上升，省下的時間也跟著上升。
  - 但 query-aware 打分要 GPU，忙的時候打分變慢。兩個效應方向相反，誰大是可量的。
- **最小證偽實驗**：§5 X3。
- **什麼時候停止投入**：
  - GPU 忙時，每少載 1 GiB 省下的還原時間沒有比閒時多（CI 重疊）；**或**
  - 只看位置的刪除（例如丟最新的 B）在同樣佔比下拿到 ≥80% 的省時，同時品質差異由 X1／X2 決定。
  - 這時「爭用」不是選擇性載入的獨立賣點，併回 G1／G2。

### G4　多輪／agent：query-agnostic 的重要度能不能預測下一輪會用到什麼

- **有沒有人做過**：
  - 多輪會壞：SCBench、KVzip、RocketKV-MT、EpiCache、FlowKV。
  - agent：Online compaction（延後用未來 query 可救回）、AgentKV（phase 混合）、CommitKV（暫時沒用 vs 已用完）、IntentKV（跨輪意圖，要學）、MemDecay（區域先驗）、TaskPress（任務先驗）、KVP（OASST2 多輪）。
  - **沒找到**：
    - 「下一輪實際會用到哪些 block」的實測存取資料：IntentKV 等只報任務分數；搜尋「next-turn access」沒有結果。
    - 在「不刪、只排順序或預取」的分層情境下評估 query-agnostic 分數，也就是把逐出分數改用來排還原或預取的順序。
- **可能的新意**〔判讀〕：
  - 逐出文獻問的是「刪掉後第 2 輪會不會壞」。分層問的是「第 2 輪的問題來之前，先預取哪些」。後者答錯的代價只是多等，不是答錯。
  - 所以 query-agnostic 分數只要**比位置好**，就有用，不需要好到能安全刪除。
  - Online compaction 的「延後」結論等於說：最好的分數要等下一輪的問題 → 在分層裡就是「先留在下層，問題來了再選」。
- **最小證偽實驗**：§5 X2。
- **什麼時候停止投入**：
  - 寫入時的分數預測第 2 輪的能力≈隨機（同 G2）；**而且**
  - 第 1 輪問題的 QRY 對第 2 輪也≈隨機。
  - 這表示在我們的任務上，跨輪的重要度不可預測，唯一的策略是無損保留＋問題來了再選（RocketKV-MT 式）。

---

## 5. 期限前（約 3 天、2×MI300X）可測的最小實驗

> **前提**：
> - 全部建在 14 號已經寫好的程式上：`code/m10_drop.py` 的分數、`KVModel.key_keep` 遮罩、`Restorer.drop`、`m10_speed.py` 的 drop 配置、`bw_<m>.json` 的慢層。
> - 44e 系列（DS1、DS2、D3、M1、M2）在 10/19 寫這份文件時**還沒有任何結果**（`results/RUNLOG_MI300X.md` 只有排程）。
> - 下面只寫設計，**我沒有跑任何東西**。
> - 時間全部是〔估計〕，單價用 14 號 §8.6 的「每題約 240 個 teacher forcing＋分數約 15 秒、約 2 分鐘／題」，即每個配置每題約 0.5 秒。實際要以 44e0 冒煙量到的為準。
> - 要改程式的地方留給執行 agent，並照 14 號的方式先登記再跑。

### X1　等位元組：有損刪除 vs 降精度（G1）

- **假說 H-X1**：只在載入區（block 32–63）動手、搬的位元組相同時，均勻低精度的品質 ≥ 最好的寫入時刪除法；刪除法只有用 QRY（知道問題）或 ORC 時才可能贏。
- **配置**：B 粒度，block 0 一律保留，只動 block 32–63。用 BF16 的比例對齊位元組，忽略 scale 的額外位元組，報告時照實列出：

  | 搬的位元組（相對 BF16） | 刪除組 | 低精度組 |
  |:--|:--|:--|
  | 約 1/2 | 丟 50%：DS1 選出的前 2 名寫入時方法、QRY、RND0 | FP8 或 INT8 |
  | 約 1/4 | 丟 75%：同上 | INT4 KIVI |
  | 約 1/8 | 丟 87.5%：同上 | INT2 KIVI 或 K4V2 |

  另加：E0、MASK0、ORC（丟 50%）當上下界。
- **對照**：同一批題目（沿用 DS1 的題目與 seed 60000 起，配對比較）；全段版（block 1–63）當補充。
- **指標**：
  - 對 E0 的 gold_lp 差、KL、first_ok（DS1 同一套）。
  - 以題目為單位做配對 bootstrap 95% CI：每個位元組預算比「最好的刪除 − 量化」。
  - 一併報 MASK0 的雜訊。
- **時間**〔估計〕：
  - 載入區版：4 個刪除法 × 3 個預算＋3 個量化＋3 個對照 ＝ 18 個配置；加上全段版約 36 個配置 × 192 題 × 0.5 s ≈ 1 小時 GPU，不含每題 prefill。
  - 如果直接併進 DS1 的 shard（prefill 共用），兩張卡合計約 0.5–1 小時。
  - 要寫的程式：在 `m10_drop.py screen` 加「區域」參數，並允許和 `m10_sweep` 的格式並列。
- **停止條件**：見 G1。

### X2（＋X4）　寫入時的分數、block 摘要能不能預測第 2 輪（G2、G4）

- **假說 H-X2**：
  - 寫入時的分數對第 2 輪答案所在 block 的召回≈隨機（和 08 §2.9 一致）。
  - 第 1 輪 QRY 對第 2 輪高於隨機，但明顯低於第 2 輪 QRY（SCBench 的「不同 query 差很多」）。
- **假說 H-X4**：在 512-token block 上，Quest 的 min/max 上界對「第 2 輪 QRY 的前 k 個 block」的召回，明顯低於用完整 K 算的 QRY。T5（多跳）類最差。
- **做法**：
  - 用 DS2 的兩輪題目（新 seed 70000 起：T2–T4 各 45 題、T5 15 題），每個模型每題算三組東西：
    1. 寫入時的分數：SNAP（無問題）、EXPATT、KNORM、KEYDIFF、H2O 近似、STREAM、NEW、RND；
    2. 第 1 輪問題的 QRY、第 2 輪問題的 QRY；
    3. 每個 block、每個（層, KV head）的 K 逐維 min/max（post-RoPE，照快取裡存的），再用第 2 輪問題的 query 算 Quest 上界。
  - 只做打分，不生成。
- **指標**：
  - 第 2 輪答案所在 block 落在前 k 名（k＝16、32 個 block，約 25%、50%）的比例，對隨機的期望 k/63 做 bootstrap CI。
  - 各分數和第 2 輪 QRY 的 block 排名 Spearman ρ。
  - X4：摘要前 k 名和第 2 輪 QRY 前 k 名的重疊率。
- **時間**〔估計〕：每題一次 32K prefill＋兩個問題的短 forward＋打分，約 10–20 s；150 題 × 2 模型 ≈ 1–1.5 GPU 小時。可以放在 44e2 之後、只用一張卡。
- **停止條件**：見 G2、G4。H-X2、H-X4 都成立 → 選擇性還原只能靠位置與下一輪的問題；第三階段不做「寫入時分數」的放置。

### X3　選擇性載入在 GPU 忙時的收益：落在載入區的比例（G3）

- **假說 H-X3**：同樣丟 50%，GPU 忙（busy2）時省下的還原時間比閒時多，因為會合點前移、被丟的 block 更多落在載入區。但純 CPU 層（只載 ÷ 只算 0.007–0.022，讀法 A 全部載入）不會有這個效應。
- **做法**：
  - **主要是分析**：用 44e3（D3）已排定的速度格子：底＝E0／E1s；丟 F、B、R × 25／50%；真實 CPU、b16、b32、b48、nfs；GPU 閒／忙。
  - 從各格的會合點算「被丟的 block 落在載入區的比例」，以及「每少載 1 GiB 省下的時間」，比較閒 vs 忙。
  - **可選加做**（要改程式）：把 DS1 在 B 粒度選出的 block 集合（SNAP、QRY 各取幾題的實際集合）丟進同一個速度格子重跑；並量 QRY 打分本身在 busy2 下要多久。
- **時間**〔估計〕：分析是 CPU 工作，<1 小時；可選加做約 0.5 GPU 小時（一張卡、計時中不能插別的工作）。
- **停止條件**：見 G3。

### 建議順序與總量

1. **X1** 最便宜，而且直接回答「drop 是不是一層」：併進 DS1，約 0.5–1 GPU 小時。
2. **X2＋X4**：44e2 之後，約 1–1.5 GPU 小時。
3. **X3**：等 D3 結果出來做分析；可選加做約 0.5 GPU 小時。

合計約 2–3 GPU 小時〔估計〕，在 44e 原本 8–11 小時的上限（14 號 §8.6）之外。X1、X2 的結果若都觸發停止條件，第三階段的「eviction／選擇性載入」方向可以在 3 天內收掉，留下負面結果。

---

## 6. 參考清單

> - arXiv ID、標題、首位作者：10/19 從 `arxiv.org/abs/<id>` 的 meta 讀到。
> - venue：只寫 arXiv comment 或會議頁有寫的；沒寫的標「venue 未查證」。
> - 「深度」同前。

### 方向 A

| ID | 首位作者 | 標題 | venue | 深度 |
|:--|:--|:--|:--|:--|
| 2306.14048 | Zhang, Zhenyu | H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs | 未查證（repo P08 記 NeurIPS'23） | 部分 |
| 2305.17118 | Liu, Zichang | Scissorhands: Exploiting the Persistence of Importance Hypothesis … | 未查證 | 摘要 |
| 2309.17453 | Xiao, Guangxuan | Efficient Streaming Language Models with Attention Sinks | ICLR 2024 | 部分 |
| 2401.06104 | Oren, Matanel | Transformers are Multi-State RNNs（TOVA） | preprint | 部分 |
| 2404.14469 | Li, Yuhong | SnapKV: LLM Knows What You are Looking for Before Generation | 未查證 | 部分 |
| 2406.02069 | Cai, Zefan | PyramidKV | 未查證 | 部分 |
| 2407.11550 | Feng, Yuan | Ada-KV | NeurIPS 2025 | 部分 |
| 2410.10819 | Xiao, Guangxuan | DuoAttention | 未查證 | 部分 |
| 2510.00636 | Devoto, Alessio | Expected Attention（附 kvpress） | 未查證 | 部分 |
| 2505.23416 | Kim, Jang-Hyun | KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction | NeurIPS 2025 Oral | 部分 |
| 2412.10319 | Li, Yucheng | SCBench | ICLR 2025 | 部分 |
| 2510.13334 | Feng, Yuan | Taming the Fragility of KV Cache Eviction（DefensiveKV） | 未查證 | 部分 |
| 2502.03805 | Feng, Yuan | CriticalKV | ICML 2026 | 摘要 |
| 2602.10238 | Moschella, Luca | Learning to Evict from Key-Value Cache（KVP） | ICML 2026 | 部分 |
| 2510.00231 | Chen, Alex | The Pitfalls of KV Cache Compression | ACL 2026 | 摘要 |
| 2608.00902 | Liu, Yujian | Practical Online KV Cache Compaction for LLM Agents | 未查證 | 部分 |
| 2609.14872 | Liu, Taowen Tony | AgentKV: Phase-Aware KV Eviction for Agentic LLMs | 未查證 | 部分 |
| 2608.07855 | Huang, Weizhong | CommitKV | 未查證 | 部分 |
| 2606.09916 | Li, Junjie | IntentKV | 未查證 | 摘要 |
| 2509.17396 | Kim, Minsoo | EpiCache | ICML 2026 | 部分 |
| 2608.03276 | Park, Wonpyo | TaskPress | EMNLP 2026 | 部分 |
| 2608.01247 | Baek, Changwoo | RestoreKV | 未查證 | 部分 |
| 2608.05326 | Garg, Ayushman | QEvict | 未查證 | 部分 |
| 2607.10582 | Matam, Venkatesha | MemDecay | 未查證 | 摘要 |
| 2606.06302 | Kim, Hyungmin | Tangram | 未查證 | 摘要 |
| 2608.13263 | Gao, Yuanhang | vToken: Token-Level Virtualization for Reclaimable KV Caches | 未查證 | 摘要 |
| 2503.24000 | Gao, Wei | Rethinking Key-Value Cache Compression Techniques for LLM Serving | MLSys 2025 | 摘要 |
| 2406.12335 | Guo, Zhiyu | Attention Score is not All You Need …: Value Also Matters（VATP） | EMNLP 2024 | 摘要 |
| 2606.29563 | Roy, Shuvendu | Coverage-Driven KV Cache Eviction（K-VEC） | 未查證 | 摘要 |
| 2605.09649 | Bui, Ngoc | Make Each Token Count | 未查證 | 摘要 |
| 2505.15347 | Liu, Xiang | FlowKV | NeurIPS 2025 Workshop | 摘要（只列） |
| 2504.15364、2406.11430、2503.02812、2504.04704、2601.07891、2410.19258、2407.15891、2404.15574、2310.01801、2503.12491、2412.14838、2405.12532、2505.24133、2506.05345、2502.01941 | — | KeyDiff、KNorm、Q-Filters、LagKV、KVzap、HeadKV、RazorAttention、Retrieval Head、FastGen、CAKE、DynamicKV、PyramidInfer、R-KV、DMS、Semantic Integrity Matters | 見 §1.1 | 只確認存在 |

### Cascade

| ID | 首位作者 | 標題 | venue | 深度 |
|:--|:--|:--|:--|:--|
| 2406.10774 | Tang, Jiaming | Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference | ICML 2024 | 部分 |
| 2402.04617 | Xiao, Chaojun | InfLLM | 未查證 | 部分 |
| 2406.19707 | Lee, Wonbeom | InfiniGen | OSDI 2024 | 部分 |
| （無 arXiv） | Chen, Renze | ArkVale: Efficient Generative LLM Inference with Recallable Key-Value Eviction | NeurIPS 2024（proceedings） | 摘要 |
| 2410.21465 | Sun, Hanshi | ShadowKV | ICML 2025（PMLR v267） | 部分 |
| 2410.16179 | Chen, Zhuoming | MagicPIG | 未查證 | 部分 |
| 2409.10516 | Liu, Di | RetrievalAttention | 未查證 | 部分 |
| 2505.02922 | Chen, Yaoqi | RetroInfer | VLDB 2026 | 部分 |
| 2407.12820 | Zhang, Hailin | PQCache | 未查證 | 部分 |
| 2412.03213 | Liu, Guangda | ClusterKV | 未查證 | 部分 |
| （無 arXiv） | Hao 等 | OmniKV: Dynamic Context Selection for Efficient Long-Context LLMs | ICLR 2025（proceedings） | 摘要 |
| 2502.14051 | Behnam, Payman | RocketKV | ICML 2025 | 部分 |
| 2502.02770 | Lin, Chaofan | Twilight | NeurIPS 2025（spotlight） | 部分 |
| 2412.14468 | Desai, Aditya | HashAttention | ICML 2025 | 部分 |
| 2410.13276／2506.08889 | Gao, Yizhao | SeerAttention／SeerAttention-R | 未查證 | 摘要 |
| 2502.11089 | Yuan, Jingyang | Native Sparse Attention | 未查證 | 摘要 |
| 2502.13189 | Lu, Enzhe | MoBA | 未查證 | 摘要 |
| 2512.02556 | DeepSeek-AI | DeepSeek-V3.2 | 技術報告 | 摘要 |
| 2509.24663 | Zhao, Weilin | InfLLM-V2 | 未查證 | 摘要 |
| 2503.16163 | Jie, Shibo | SpeCache | 未查證 | 部分 |
| 2505.13109 | Liu, Guangda | FreeKV | 未查證 | 部分 |
| 2604.08426 | Bocharnikov, Andrey | KV Cache Offloading for Context-Intensive Tasks（YAKV） | preprint | 部分 |
| 2510.13602 | Huang, Yuxiang | NOSA | EMNLP 2026 main | 部分 |
| 2509.24626 | Zhou, Qihui | SparseServe | 未查證 | 部分 |
| 2503.00392 | Zhou, Qihui | Progressive Sparse Attention（PSA） | 未查證 | 摘要（只列） |
| 2608.07009 | Xie, Zhiqiang | HiSparse | 未查證；另有 LMSYS 部落格 2026-04-10 | 部分 |
| 2606.04511 | Fu, Yaosheng | SparDA | 未查證 | 部分 |
| 2609.17652 | Kalyanarangan, Vivek | Fathom | 未查證 | 部分 |
| 2606.19746 | Ma, Ruiyang | SAC | 未查證 | 部分 |
| 2609.37538 | Kuang, Wenwei | AVSG | 未查證 | 部分 |
| 2603.13430 | Levy, Noam | Dynamic Sparse Attention: Access Patterns and Architecture | 未查證 | 部分 |
| 2608.08097 | Xiao, Can | OasisKV | 未查證 | 摘要 |
| 2511.11907 | Zhang, Huawei | KVSwap | 未查證 | 摘要 |
| 2506.20187 | Sun, He | LeoAM（Breaking the Boundaries of Long-Context LLM Inference） | 未查證 | 摘要 |
| 2605.18071 | Lin, Jian | KVDrive | 未查證 | 摘要 |
| 2601.13631 | Zou, Jing | ContiguousKV | 未查證 | 部分 |
| 2605.24022 | Li, Fei | Adaptive KV Cache Reuse for Fast Long-Context LLM Serving（CacheTune） | 未查證 | 部分 |
| 2606.01751 | Zhang, Quqing | SparseX | 未查證 | 摘要（只列） |
| 2504.17768 | Nawrot, Piotr | The Sparse Frontier | 未查證 | 摘要 |
| 2602.07721 | Qi, Yanlin | ParisKV | ICML 2026 | 部分 |
| 2406.02542、2411.02886、2502.08910、2408.07092、2312.04985、2502.14866、2502.12574、2502.04077 | — | Loki（NeurIPS'24）、TokenSelect（EMNLP'25）、InfiniteHiP、Double Sparsity、SparQ、LServe（MLSys'25）、HeadInfer、AttentionPredictor（NeurIPS'25） | 見 arXiv comment | 只確認存在 |

### repo 內引用（不重抄）

- `docs/precision_20261013/cards/P08_token_drop_literature.md`：kvpress 分數定義。
- `docs/precision_20261013/14_match_drop_plan.md`：DS1／DS2／D3／M1／M2 計畫。
- `docs/SOTA_ANALYSIS_20261006.md`：Cake、CacheFlow、Strata、Pensieve、AsymCache、When Fancy Eviction Fails（2609.28870，10/19 再確認存在）。
- `docs/PAPER_GUIDE.md`：ArkVale、QEvict、InfiniGen、Quest、ShadowKV、YAKV 卡。
- `docs/research_20261009_explore/lit_A_summary.md`：IMPRESS、Krul、TierKV。
- `docs/research_20261010_directions/D2_vllm_drop.md`：vLLM 0.28 的丟資料率與等慢層。

### 官方實作查證紀錄（10/19，GitHub API `/languages`，單位：位元組）

| repo | 語言比例 |
|:--|:--|
| mit-han-lab/Quest | Cuda 238,668／Python 184,640 |
| microsoft/RetrievalAttention | Python 1,551,250／C++ 76,306／Cuda 60,934 |
| thunlp/InfLLM | Python 152,293（README：Triton 多段 flash attention） |
| mit-han-lab/duo-attention | Python 222,947（README 要 CUDA toolkit、flash-attn、FlashInfer、Block-Sparse-Attention） |
| NVlabs/RocketKV | Python 747,224（flash-attn 2.6.3） |
| microsoft/SeerAttention | Python 1,115,868／Cuda 73,615 |
| MoonshotAI/MoBA | Python 28,540（kernel 依賴 flash-attn 2.6.3） |
| fla-org/native-sparse-attention | Python 142,367（Triton） |
| snu-mllab/KVzip | Python 163,140／Cuda 10,348 |
| NVIDIA/kvpress | Python 546,446 |
| FFY0/AdaKV | Python 155,023／Cuda 3,274 |
| FFY0/DefensiveKV | Python 281,507／Cuda 4,259 |
| Zefan-Cai/KVCache-Factory | Python 718,582／Cuda 3,177 |
| mit-han-lab/streaming-llm | Python 23,735 |
| schwartz-lab-NLP/TOVA | Python 19,544 |
| FasterDecoding/SnapKV | Python 62,896 |
| sjtu-zhao-lab/ClusterKV | Cuda 322,976／Python 228,013 |
| sjtu-zhao-lab/FreeKV | Python 229,536／Cuda 126,012 |
| thunlp/NOSA | Python 3,676,866／Cuda 6,024 |
| NVlabs/SparDA | Python 3,393,761／C++ 1,080,627／Cuda 379,692 |
| amy-77/ParisKV | Python 557,926／Cuda 64,702 |
| antgroup/OmniKV | Python 2,464,841 |
| LiuTaowen-Tony/agentkv | Python 45,995（README 要 NVIDIA GPU） |
| snu-comparch/InfiniGen | Python 877,643 |
| Infini-AI-Lab/MagicPIG | Python 502,074／C++ 58,548 |
| bytedance/ShadowKV | API 回 301（repo 搬家）；README：flash-attn＋FlashInfer＋自建 kernel |
| microsoft/RetroInfer | API 回 404 |

本機（`/mlsteam/workspace/venv/tiara-v028`）的 site-packages：
- 有：torch 2.12.1+rocm7.2、triton 3.8.0／triton_rocm 3.7.1、transformers 5.17.0、vllm 0.28.0+rocm722。
- **沒有**：flash-attn、FlashInfer、kvpress、sglang。
