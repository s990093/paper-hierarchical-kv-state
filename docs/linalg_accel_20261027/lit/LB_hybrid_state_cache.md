# LB：hybrid／SSM 模型的狀態快取、卸載與還原（2026-10-27）

> 範圍：production 模型改用 hybrid 架構（多數層是線性注意力／SSM，只有少數 softmax 注意力層）之後，「暫停後回來」的還原問題怎麼變。查的是：各 hybrid 模型的 KV 與狀態大小、引擎（vLLM、SGLang、LMCache、Jenga）怎麼存、怎麼卸載狀態，以及有沒有人已經做了「搬 vs 重算」或「檢查點間隔」的最佳化。
> 規則：`RULES.md`。全部是讀原文、讀程式碼與算術，**沒有跑任何實驗**。證據等級標在每個事實後面。
> 已在 repo 的整理只指過去、不重抄：`docs/research_20261009_explore/lit_C_architectures.md` §3.5（狀態大小表、Cake 在 hybrid 上要不要中間狀態的結構論證、各系統檢查點規則表）、`cards/A_Marconi_補充.md`、`A_SparsePrefixCaching.md`、`A_SuffixReplay.md`、`A_Jenga.md`、`C_GLMHybridRecovery.md`、`C_LinearKV.md`、`C_TailReplay.md`、`C_KVBuffer.md`；`docs/stage3_design_20261024/lit/L8c_practice_and_trends.md` §3 的模型表；`docs/DIRECTIONS_20260927.md` 方向 8；`docs/PAPERS_BY_LEVEL.md` L0（Marconi、Sparse Prefix Caching、DASC、Tail-Replay）。

---

## §0 一頁白話結論

**一句話**：hybrid 模型讓還原「變小、但多出一個要嘛整個搬、要嘛整段重算的大物件」。引擎已經會把這個物件搬到 CPU，甚至搬到儲存層。但到今天，還沒有人用成本去決定「搬還是重算」，也沒有人用成本去決定「檢查點隔多久存一份」。檢查點間隔只有固定旋鈕和一篇只算重算量的 DP。

1. **大小（算術，§1）**：128K 時，Llama-3.1-8B 的 KV 是 16 GiB。主流 hybrid 只有 0.75–8 GiB：Qwen3.5-35B-A3B 2.5 GiB、Kimi-Linear 0.98 GiB、Nemotron-3-Nano 0.75 GiB。另外每個序列多一份固定狀態，大小 27 MiB 到 870 MiB；最大的是 Kimi-K3 的 429 MiB 與 Nemotron-H-56B 的 870 MiB。換算下來，128K 時狀態只占還原總 bytes 的 0.2%–14.5%；256K 時再減半。**128K–256K 的還原仍然是 KV 主導**，狀態是一個固定的附加費。也有例外：Falcon-H1 每一層都有注意力，34B 的 KV 是 144 KiB/token，比 Llama 還大；OLMo-Hybrid-7B 用 MHA，120 KiB/token，跟 Llama 差不多。「hybrid」不等於 KV 小。
2. **狀態的性質**：只在被存下來的那個位置有效（原地覆寫，不能退回）。用到它只有兩條路：(a) 整份載回；(b) 從最近一份檢查點，把後面每個 token 都過一遍**全模型**重算，連注意力層一起。所以它不能像 KV 那樣「前段算、後段搬」切開。〔判讀；結構論證見 lit_C §3.5〕
3. **對 Tiara 的直接含義**〔判讀〕：純暫停／回來（append-only，續問同一個 session）時，回來的位置是已知的「prompt 結尾」。這時只需要結尾那一份狀態（暫停時本來就存），加上注意力層的 KV。Cake 的「前段重算＋後段搬」只作用在注意力 KV，狀態一律整份搬。這種情境下**沒有檢查點間隔的問題**。中間檢查點只在三種情況才有用：分叉（共用前綴）、反向 Cake（先搬前段、後段重算）、尾段被 DROP。
4. **引擎現況（程式碼，2026-10-27 HEAD）**：
   - **vLLM**：只在 block 邊界存狀態（`align` 模式）。OffloadingConnector 已經會把 Mamba 狀態跟 KV 一起卸到 CPU／分層儲存。稀疏保留用一個**固定**旋鈕 `prefix_cache_retention_interval`。逐出是所有 group 共用一條 LRU。
   - **SGLang**：radix 樹在 chunk 對齊點存狀態，decode 時每 256 token 存一份（`mamba_track_interval`）。有 int8 檢查點池（2026-06 合入），有 host 層 `MambaPoolHost`，L3 儲存把狀態當 `TRAILING_PAGES`。
   - **LMCache**：把狀態當不透明 page 存。
   - 以上**全部是「有就搬」**：沒有一個在命中時比較「從 CPU／SSD 搬這份狀態」和「從更早的檢查點重算」哪個便宜。〔程式碼；搜尋 `recompute`／`cost` 只找到失敗後 fallback 重算〕
5. **最接近「成本感知」的前作**：
   - **Marconi**（MLSys'25）：逐出分數 = 新近度 + α·FLOP efficiency（省下的 FLOPs ÷ 狀態 bytes）。單一容量層，沒有頻寬項，也沒有 CPU／SSD。〔全文〕
   - **Sparse Prefix Caching**（2026-04）：給定檢查點數量預算，用 DP 選位置，目標只有重算 token 數。〔repo 卡〕
   - **vLLM RFC（2026-09）**：讓應用程式自己標檢查點位置。〔文件〕
6. **Young／Daly 能不能直接套**〔判讀，§3 C9〕：形式上可以寫出同樣的 √ 律，T* = √(2·存一份的代價·…/(命中率·每 token 重算成本))。但前提差很多：Young／Daly 的失敗是隨機的、無記憶的，損失的工作和位置無關。我們的「回來點」在 append-only 情境是**已知**的，最佳解退化成「只在結尾存一份」。只有分叉負載的分叉深度才是隨機變數，而且分布由負載決定、不是指數分布，所以該用 Sparse Prefix Caching 那種 k-median DP，不是 √ 公式。另外，從檢查點重算的成本隨絕對位置變大（注意力層），不是常數。
7. **重算有多快（量到的數字）**：
   - Mamba-2 的 SSD 單層 kernel 在 A100 上 2K 起比 FlashAttention-2 快，16K 快 6 倍（Fig. 10）。
   - FLA 在 GB200 上，T=16384 時 chunk_gdn 1.03 ms，flash_attn 5.04 ms（B=2、H=16、D=128，前向）。
   - 整個模型的 prefill：Kimi Linear 比全 MLA 快 2.3 倍（512K）、2.9 倍（1M）（Fig. 7a，batch 1）。
   - 但重算「一份狀態」要過全模型（注意力、MoE 都要算），單層 scan 的速度不是瓶頸。〔全文／文件；判讀〕

**空白（判讀，詳見 §4）**：
- 「多層（GPU／CPU／SSD）× 狀態與 KV 一起 × 用頻寬與 FLOPs 決定搬或重算、決定檢查點放哪一層」這個組合，本次搜尋沒找到。
- 但對 Tiara 的主情境（append-only 暫停／回來）價值有限：狀態只占 0.2–15% 的 bytes，而且只有一份、必搬。
- 真正會改變 Tiara 結論的是另一件事：KV bytes/token 變小 5–20 倍，κ（重算 vs 搬的成本比）會大幅往「搬」偏。這點 lit_C §2 的 R0／R32 已經算過。

---

## §1 模型表：注意力 KV（每 token）與固定狀態（每序列）

**白話**：先看每種 hybrid 模型「每多一個 token 要多存多少 KV」，以及「不管多長都固定要存的狀態」有多大。再拿 Llama-3.1-8B 在 128K、256K 的 KV 來比。

### 1.1 公式〔算術〕

- **GQA 注意力 KV／token（BF16）** = 注意力層數 × 2（K、V）× KV head 數 × head_dim × 2 B。
- **MLA KV／token** = MLA 層數 × (kv_lora_rank + qk_rope_head_dim) × 2 B = 層數 × 576 × 2 B。
  - Kimi-Linear 與 Kimi-K3 的 config 是 `mla_use_nope: true`，vLLM 的 Kimi-K3 程式碼也斷言「NoPE-only」，但投影仍輸出 512+64 維（`vllm/models/kimi_k3/amd/linear.py` 約 L395–431）。快取實際存 512 還是 576 沒有核到，所以 576 是**上界**〔程式碼，部分〕。
- **狀態大小**用 vLLM 的形狀公式（`vllm/model_executor/layers/mamba/mamba_utils.py`，commit `115b59a`，2026-10-27）〔程式碼〕：
  - Mamba1（Jamba）：conv (d_inner, k−1) ＋ ssm (d_inner, d_state)（L171–183）。
  - Mamba2（Nemotron-H、Falcon-H1、Granite-4.0-H）：conv (d_inner + 2·n_groups·d_state, k−1) ＋ ssm (heads, head_dim, d_state)（L186–211）。
  - GDN（Qwen3-Next／3.5／3.6、OLMo-Hybrid）：conv (2·k_heads·d_k + v_heads·d_v, k−1) ＋ ssm (v_heads, d_v, d_k)（L272–293）。
  - KDA（Kimi）：conv (3·heads·d, k−1) ＋ ssm (heads, d, d)（L296–319）。
  - lightning（MiniMax-01／M1）：(heads, d, d)，沒有 conv（HF `modeling_minimax_text_01.py` L297：`torch.zeros(b, h, d, e)…float32`，commit `a7351bf`）〔程式碼〕。
  - k = conv kernel = 4（所有 config 都是 4）。
- **狀態的 dtype**（rule 6：不看欄位名，看程式碼）：conv 一律跟模型 dtype（BF16）。ssm 狀態如下〔程式碼〕：
  - NemotronH 系列：vLLM 預設 **FP32**，「Only float32 is known to have no accuracy issues」（`vllm/model_executor/models/config.py` L812–832）。Nemotron-3 的 config 也寫明 `mamba_ssm_cache_dtype: float32`。
  - Qwen3.5／3.6：config 有 `mamba_ssm_dtype: float32`，vLLM 照用（同檔 L953–976）。
  - KDA：`auto` 時是 **FP32**（`mamba_utils.py` L133–149）。
  - MiniMax lightning：HF 參考實作是 FP32。
  - Qwen3-Next、Jamba、Falcon-H1、Granite：vLLM 的 `auto` 跟模型 dtype，所以是 BF16（`mamba_utils.py` L98–110）。Qwen3-Next 另列 FP32 一行對照。
  - OLMo-Hybrid 的 config 沒有 dtype 欄位，以 BF16 計〔未查證〕。
- **狀態占比** = 狀態 ÷ (KV/token × 長度 + 狀態)。「狀態 ≈ 幾個 token 的 KV」= 狀態 ÷ KV/token。

### 1.2 表〔算術；config 皆為 HF 官方 `raw/main/config.json`，2026-10-27 抓取，下載副本在 `<SCRATCH>/litB/cfg/`〕

| 模型（HF commit，建立日） | 層組成（注意力／線性） | KV B/token | KV @128K | KV @256K | 每序列狀態 | 狀態 ≈ 幾個 token 的 KV | 狀態占比 @128K／@256K | Llama@128K ÷ 本模型@128K |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| **Llama-3.1-8B**（對照；值見 L8c §3） | 32 GQA (8×128)／0 | 131,072 | 16.00 GiB | 32.00 GiB | 0 | — | — | 1.00× |
| Qwen3-Next-80B-A3B（`9c7f2fb`，2025-09） | 12 gated attn (kv2×256)／36 GDN | 24,576 | 3.00 GiB | 6.00 GiB | 37.7 MiB（BF16 ssm）；73.7 MiB（FP32） | 1,608；3,144 | 1.2%／0.6%；2.3%／1.2% | 5.3× |
| Qwen3.5-35B-A3B（`59d61f3`，2026-02）＝Qwen3.6-35B-A3B（`995ad96`，2026-04，欄位相同） | 10／30 GDN | 20,480 | 2.50 GiB | 5.00 GiB | 61.4 MiB（FP32） | 3,144 | 2.3%／1.2% | 6.3× |
| Qwen3.5-27B（`fc05dae`，2026-02） | 16 (kv4×256)／48 GDN (v48) | 65,536 | 8.00 GiB | 16.00 GiB | 146.8 MiB（FP32） | 2,349 | 1.8%／0.9% | 2.0× |
| Kimi-Linear-48B-A3B（`e1df551`，2025-10） | 7 MLA／20 KDA (32×128) | ≤8,064 | 0.98 GiB | 1.97 GiB | 41.4 MiB（FP32） | 5,384 | 3.9%／2.0% | 15.6× |
| **Kimi-K3**（`f831ab6`，2026-06；2.78T 參數） | 24 MLA／69 KDA (96×128) | ≤27,648 | 3.38 GiB | 6.75 GiB | **428.6 MiB**（FP32） | 16,253 | 11.0%／5.8% | 4.2× |
| Nemotron-H-8B-Base-8K（`94ea861`）† | 4 (kv8)／24 Mamba2 (128×64, N=128) | 16,384 | 2.00 GiB | 4.00 GiB | 97.4 MiB（FP32） | 6,234 | 4.5%／2.3% | 7.6× |
| Nemotron-H-56B-Base-8K（`1fb0949`）† | 10／54 Mamba2 (256×64, N=256) | 40,960 | 5.00 GiB | 10.00 GiB | **870.3 MiB**（FP32） | 22,280 | 14.5%／7.8% | 2.7× |
| Nemotron-Nano-9B-v2（`6533e8d`） | 4／27 Mamba2 (128×80) | 16,384 | 2.00 GiB | 4.00 GiB | 136.9 MiB（FP32） | 8,762 | 6.3%／3.2% | 7.5× |
| Nemotron-3-Nano-30B-A3B（`bf77c31`，2025-12） | 6 (kv2)／23 Mamba2 (64×64) | 6,144 | 0.75 GiB | 1.50 GiB | 46.8 MiB（FP32） | 7,989 | 5.7%／3.0% | 20.1× |
| Nemotron-3-Super-120B-A12B（`2dc98e2`，2026-03） | 8 (kv2)／40 Mamba2 (128×64) | 8,192 | 1.00 GiB | 2.00 GiB | 162.3 MiB（FP32） | 20,780 | 13.7%／7.3% | 13.8× |
| Jamba2-Mini（`24cbbd2`，2026-01）≈ Jamba-1.5-Mini 結構‡ | 4 (kv8)／28 Mamba1 (8192, N=16) | 16,384 | 2.00 GiB | 4.00 GiB | 8.3 MiB（BF16） | 532 | 0.4%／0.2% | 8.0× |
| Jamba-1.5-Large（論文 §2）‡ | 9 (kv8×128)／63 Mamba1 | 36,864 | 4.50 GiB | 9.00 GiB | 37.4 MiB（BF16；假設同 v0.1 的 expand 2、N=16，〔未查證〕） | 1,064 | 0.8%／0.4% | 3.5× |
| MiniMax-Text-01（`a7351bf`）／M1-80k（`8d1494b`） | 10 softmax (kv8)／70 lightning (64×128) | 40,960 | 5.00 GiB | 10.00 GiB | 280.0 MiB（FP32） | 7,168 | 5.2%／2.7% | 3.0× |
| Falcon-H1-34B（`6e6890e`）§ | **72 注意力 ∥ 72 Mamba2**（每層都有） | **147,456** | 18.00 GiB | 36.00 GiB | 146.1 MiB（BF16） | 1,039 | 0.8%／0.4% | **0.88×** |
| Falcon-H1-7B（`41e72f2`）§ | 44 ∥ 44 | 45,056 | 5.50 GiB | 11.00 GiB | 66.9 MiB（BF16） | 1,557 | 1.2%／0.6% | 2.9× |
| Granite-4.0-H-Small（`b8c0982`） | 4 (kv8)／36 Mamba2 (128×64) | 16,384 | 2.00 GiB | 4.00 GiB | 73.7 MiB（BF16） | 4,719 | 3.5%／1.8% | 7.7× |
| Granite-4.0-H-Tiny（`791e0d3`） | 4 (kv4)／36 Mamba2 (48×64) | 8,192 | 1.00 GiB | 2.00 GiB | 27.7 MiB（BF16） | 3,544 | 2.6%／1.3% | 15.6× |
| OLMo-Hybrid-7B（HF `allenai/OLMo-Hybrid-7B`） | 8 **MHA** (30×128)／24 GDN (30, d_k 96, d_v 192) | 122,880 | 15.00 GiB | 30.00 GiB | 26.9 MiB（BF16 假設） | 230 | 0.2%／0.1% | 1.07× |

註：
- † Nemotron-H Base 的 `max_position_embeddings` 是 8192，128K／256K 只是算術外推，模型本身不支援。
- ‡ Jamba-1.5 的 ai21labs 原檔要授權（HTTP 401）。Jamba2-Mini 與 Jamba-v0.1 的結構欄位相同。Jamba-1.5-Large 的層數、KV head 取自論文 §2：9 個 block × 8 層，a:m = 1:7，hidden 8192，64 個 q head、8 個 KV head〔全文，arXiv 2408.12570 p.2〕。
- § Falcon-H1 的每一層同時有注意力分支與 Mamba2 分支，兩者平行再相加：config 的 `attn_layer_indices: null`，vLLM 的 `FalconH1ParallelHybrid` 寫明「processed in parallel through both the self-attention branch and the SSM (Mamba) branch」（`vllm/model_executor/models/falcon_h1.py` L313–355）〔程式碼〕。
- Nemotron 的 pattern 字元意義：`M`＝Mamba、`*`＝注意力、`-`＝MLP、`E`＝MoE（`nemotron_h.py` L548–553）〔程式碼〕。MiniMax 的 `attn_type_list`：0＝lightning、1＝softmax（HF modeling L978–982）〔程式碼〕。
- **rule 6 交叉驗證**〔算術 vs 全文〕：
  - Jamba-1.5 論文 Table 1 寫 256K、16-bit 的 KV：Jamba-1.5-Mini 4GB、Large 9GB、LLaMA-3.1 8B 32GB（p.2）。本表算出 4.00／9.00／32.00 GiB，三個都對上，表示層數與 head 的解讀正確。
  - Kimi Linear 論文說 KV 用量最多降 75%（p.1）。本表 1 − 7/27 = 74%，對上。
  - Marconi 的 7B hybrid：狀態是單 token KV 的 N/2 = 64 倍（Table 1，p.15）。
- **與 repo 既有整理不一致（以程式碼為準）**：
  - lit_C §3.5 的狀態表把所有模型都以 BF16 計：Nemotron-H-8B 49.4、Nano-9B-v2 69.4、Kimi-Linear 21.4、MiniMax 140 MiB。依 vLLM HEAD，NemotronH 與 KDA 的 ssm 預設是 FP32，MiniMax 參考實作也是 FP32，實際應為 97.4／136.9／41.4／280 MiB，約兩倍。
  - lit_C 用的是 vLLM 0.19.1（本機 build）；NemotronH 的 FP32 預設在 0.19.1 是否已存在〔未查證〕。
  - DIRECTIONS_20260927 方向 8 的 77.3 MB（Qwen3-Next）＝本表的 73.7 MiB（FP32），一致。
  - L8c §3 只給元素數，沒換成 bytes。
- vLLM HEAD 已經不支援 MiniMax-Text-01：`MiniMaxText01ForCausalLM` 列在 `_PREVIOUSLY_SUPPORTED_MODELS`，最後支援版本 0.23.0（`vllm/model_executor/models/registry.py` L838、L859）〔程式碼〕。

**判讀**：
- 128K 時，主流 hybrid 的 KV 是 Llama 的 1/2 到 1/20。狀態的大小相當於 230–22,000 個 token 的 KV。
- 所以在 Tiara 的 128K–256K 範圍裡，狀態是「一個固定大物件＋小很多的 KV 流」。
- 在短 context（幾千 token 以內）時反過來，狀態主導。Marconi 量到的「狀態塞爆快取」就發生在這個區間（LMSys／ShareGPT 多在 10K 以下）。

---

## §2 卡片

### C1 Marconi：Prefix Caching for the Era of Hybrid LLMs（MLSys 2025；arXiv 2411.19379）——只補新事實
- **讀了什麼**：〔全文〕v3 PDF，https://arxiv.org/pdf/2411.19379 ，這次讀 p.4、p.6、p.15（PDF 頁），查證 2026-10-27。repo 已有完整卡（E05 評測卡、`A_Marconi_補充.md`），這裡只補沒寫到的。
- **一句話**：hybrid 的前綴快取；准入只存分叉點與最後一個 token 的狀態，逐出看「省下的 FLOPs ÷ bytes」。
- **線性代數的關鍵一步**：沒有用新的線性代數。它利用 SSM 狀態「大小固定、和代表幾個 token 無關」這個性質，算出 FLOP efficiency = 所有層省下的 FLOPs ÷ 所有狀態的 bytes（式 1，p.6）〔全文〕。
- **成立的前提**：只對 hybrid／SSM 有差。原文說純 Transformer 的 KV FLOP efficiency 近乎常數，所以不用考慮（p.6）〔全文〕。
- **量到／原文給的數字（新補）**：
  - 細粒度 checkpoint 的浪費：block 32 時，25.0% 的 token block 的 KV 會被之後的請求重用，但 SSM 狀態只有 0.4% 被重用，差 65.3 倍；block 128 時狀態重用率 3.3%（Fig. 3a，p.4）〔全文〕。
  - 一個 7B hybrid、10K token 的單一序列，用細粒度 checkpoint 就吃掉 17.4 GB，是同尺寸 Transformer 的 3.3 倍（Fig. 3b，p.4）〔全文〕。
  - Table 1（p.15）〔全文〕：
    - 每層狀態大小：注意力 4LD bytes、SSM 2DN bytes（FP16，D＝d_model、N＝d_state、L＝序列長）。
    - 每 byte 省下的 FLOPs：注意力 L+2D，SSM L·(6D/N + 8 + 5/(DN))。7B 模型代入後是 L+8192 對 200L。
    - SSM 狀態是單 token KV 的 N/2 = 64 倍（D=4096、N=128）。
    - conv 狀態只占 6.1%，表中略去，但實驗有算（p.15 A.1）。
  - 原文提到狀態可能塞爆「GPU HBM（甚至 CPU RAM）」（p.4），但系統本身只有單一容量層〔全文〕。
- **跟 Tiara 的關係**〔判讀〕：
  - FLOP efficiency 是「省多少計算 ÷ 占多少容量」，沒有頻寬，也沒有層級。
  - 換到 Tiara 的多層，自然的延伸是「省下的時間 ÷（容量成本＋載回時間）」，也就是把 κ 放進效用分數。這個延伸本次沒看到有人做。
- **與既有整理的關係**：與 E05、A_Marconi_補充 一致；17.4 GB、65.3×、Table 1 的公式是新補的。

### C2 vLLM：hybrid KV cache manager、Mamba 前綴快取、OffloadingConnector（程式碼 HEAD `115b59a`，2026-10-27）
- **讀了什麼**：
  - 〔程式碼〕`vllm/config/cache.py`、`vllm/v1/kv_cache_interface.py`、`vllm/v1/core/single_type_kv_cache_manager.py`、`vllm/model_executor/layers/mamba/{mamba_utils,checkpoint}.py`、`vllm/model_executor/models/config.py`、`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`。
  - 〔文件〕`docs/design/hybrid_kv_cache_manager.md`、`docs/features/automatic_prefix_caching.md`。
  - 都在同一個 commit。
- **一句話**：狀態只在 block 邊界存；可以卸到 CPU 等外部層；稀疏保留是固定間隔；沒有成本模型。
- **事實**：
  - `mamba_cache_mode` 只有 `"none"` 與 `"align"`。`align` 的意思是「只存每個排程步最後一個 token 的狀態，而且這個 token 要剛好在 i×block_size 上」（cache.py L176、L304–311）。0.19.1 時還有的 `"all"` 模式在 HEAD 已經不存在。開 prefix caching 時自動設成 `align`，而且必須開 chunked prefill；`mamba_block_size` 預設等於注意力的 block_size（models/config.py L765–787）〔程式碼〕。
  - 命中只能落在 block 邊界上（APC 文件 L16–18）。另有 `--enable-mamba-shared-prefix-checkpoint`，會在分叉點多存一份，但只在 EAGLE／MTP 打開時才生效（cache.py L312–317；APC 文件 L18–35）〔程式碼／文件〕。
  - GPU 上，`align` 模式每個請求同時只常駐 2＋(speculative block 數)＋(prefill checkpoint block 數) 份狀態，較早的被清掉（kv_cache_interface.py L1135–1152）〔程式碼〕。
  - **page 大小對齊**：所有 group 共用同一個 page 大小，所以注意力的 block_size 要放大到「block_size × 注意力 kv_hidden_size ≥ 一份 mamba 狀態」，再把狀態 pad 到這個大小。文件自己承認這可能讓 block 超過 400 token（hybrid 設計文件 L127–147）〔文件〕。
  - **逐出**：「For now, we use one LRU queue for all kv cache groups」（同文件 L203）。也就是**沒有** Marconi 那種狀態專用的效用分數〔文件〕。
  - **稀疏保留間隔** `prefix_cache_retention_interval`（cache.py L275–281；預設 0）〔程式碼〕：
    - 0：只留「語意上會用到的」檢查點，也就是 replay boundary（prompt 結尾）與分叉點。
    - 正整數：另外每隔這麼多 token 留一份。
    - None：每個 block 都留。
    - `MambaManager.reachable_block_mask` 的 docstring 把 replay boundary 與分叉點稱為「Marconi-style APC」，並明說這兩點「always kept」（single_type_kv_cache_manager.py L1549–1600）。
    - 這是一個**手設的固定間隔**，沒有從成本推導。
  - **卸載**：OffloadingConnector 會處理 `MambaSpec` 的 `align` group〔程式碼〕：
    - 命中窗口要向下取整到 offload chunk 大小（scheduler.py L137–155、L916–921）。
    - Mamba group 的載入範圍是「從本地已算到的位置，到命中結尾」，整份狀態一起載（L1487–1490）。
    - 儲存時同樣套用「SWA/Mamba sparsity + retention interval」的遮罩（L1966–1975）。
    - 外部層有 CPU，也有 `vllm/v1/kv_offload/tiering/{fs,obj,p2p}`。
    - 全目錄搜 `recompute`：只在「遠端失敗或 miss 時改成本地重算」出現（例：`tiering/manager.py` L1160、`p2p/manager.py` L540）。**沒有「搬 vs 重算」的成本比較**。
  - **與 repo 不一致**：lit_C §5 說 0.19.1 的 OffloadingConnector 遇到 SSM hybrid 會報錯〔程式碼判讀〕。HEAD 已支援，DIRECTIONS_20260927 也記錄 v0.28.0 已支援。版本差，不是錯。
  - **成熟度**：issue #63565（2026-10-25 開，仍開著）〔文件，WebFetch 摘要，未逐字核對〕：
    - 稀疏保留下，每個請求只在 prompt 最後一個 block 留一份狀態。
    - 重訪時若在該點之前分叉，CPU 層有注意力 block，卻「沒有在它之下的 Mamba 狀態」，offload 命中變成 0，整段約 16K token 重算。
    - 環境：Qwen3.6-35B-A3B、576-token block、vLLM 0.30.0。
    - 修法 PR #63566 未合併。另有未合併的 PR #52527（回報「因缺稀疏保留檢查點而損失的共用前綴 token」）與 #55403（block priority）〔`gh pr list`〕。
- **線性代數的關鍵一步**：無（純記憶體管理）。
- **成立的前提**：hybrid 必須有 full attention 層，而且只支援「full + 一種其他類型」（hybrid 設計文件 L201）〔文件〕。
- **量到的數字**：無（這些是程式碼與文件）。
- **跟 Tiara 的關係**〔判讀〕：
  - vLLM 已經提供「狀態跟 KV 一起卸載」的機制，Tiara 在 hybrid 上可以直接用 OffloadingConnector。
  - 缺的是決策：存哪幾份（只有固定間隔）、放哪一層（沒有選擇）、回來時搬或重算（沒有比較）。
  - #63565 說明一個實際風險：狀態保留太稀疏時，**已經卸載的 KV 會整段失效**。這是 hybrid 才有的「KV 與狀態的共同命中」耦合。

### C3 Jenga：Effective Memory Management for Serving LLM with Heterogeneity（SOSP'25〔二手〕；arXiv 2503.18292）——只補新事實
- **讀了什麼**：〔部分〕PDF p.7–8，https://arxiv.org/pdf/2503.18292 ，查證 2026-10-27。SOSP'25 只有清華翟季冬的論文頁（dblp 轉出）這個來源，沒打開 ACM DL〔二手〕。repo 卡 `A_Jenga.md`。
- **一句話**：用 LCM 大 page 管異質層；Mamba 層每 512 token 存一份狀態。
- **新補事實**〔全文〕：
  - 如果改用「取最大 page」的做法：「Jamba 52B 的大 mamba 狀態需要每個 self-attention page 放 1344 個 token」才不會內部碎片。這比 ShareGPT 的平均請求長 1085.04 還大（p.7）。
  - Mamba 層「only caches the state of every 512 tokens」，所以只能在 512 的倍數命中；只有最後一個快取 token 的存取時間會更新；原文說 Marconi 的選擇演算法「can be integrated」（p.8）。
- **rule 6 核對**〔算術〕：用 Jamba-v0.1 的欄位算，一層 Mamba1 狀態（BF16）是 311,296 B，一層注意力每 token 的 KV 是 4,096 B，比值 76。28 層狀態對 4 層 KV/token 是 532 token。兩者都不是 1344。原文沒寫 1344 怎麼算的〔未查證〕。
- **跟 Tiara 的關係**：固定 512 是第三個「手設間隔」的例子（另外兩個是 vLLM 的 retention interval、SGLang 的 track interval）〔判讀〕。
- **與既有整理的關係**：A_Jenga 已有「每 512 token」（p.8）；1344 與 rule 6 的不一致是新補的。

### C4 SGLang：Mamba radix cache／Unified Radix Cache／HiCache（程式碼 HEAD `3927a73`，2026-10-27；LMSYS blog 2026-08-11）
- **讀了什麼**：
  - 〔程式碼〕`python/sglang/srt/mem_cache/{mamba_checkpoint_pool.py, pool_host/mamba.py, hicache_storage.py, unified_cache/components/mamba.py, evict_policy.py}`、`arg_groups/fields/exec_.py`、`arg_groups/overrides.py`、`runtime_context.py`、`managers/schedule_batch.py`。
  - 〔文件〕LMSYS blog「Unified Radix Cache: One Tree for Hybrid Model Prefix Caching」（Huang, Bao, Zhang, Ouyang, Pan，2026-08-11；WebFetch 摘要）。
- **一句話**：在 chunk 對齊點與 decode 每 256 token 存狀態；可以 int8 壓縮；可以卸到 host 與 L3。
- **事實**〔程式碼〕：
  - **prefill 的檢查點粒度**：`mamba_cache_chunk_size` = max(模型的 mamba_chunk_size，沒有時用 FLA 的 CHUNK_SIZE 64；page_size)（overrides.py L1897–1925）。命中後的「分叉點」= Full-KV 命中長度向下取整到 `mamba_checkpoint_grid`（= lcm(chunk, page)），只在它比現有狀態邊界更深時才存（unified_cache/components/mamba.py L172–193；runtime_context.py L1958–1979）。
  - **decode 的追蹤間隔**：`mamba_track_interval` 預設 **256**（「The interval to track the mamba state during decode」，exec_.py L530–533；schedule_batch.py L3803–3819 用 `seq_lens % interval` 判斷）。
  - **每條路徑的狀態上限**：`mamba_max_states_per_path`，超過時刪「最淺的內部狀態」，但保留它們的 full KV（exec_.py L497–503）。
  - **int8 檢查點池**：PR #28185，2026-06-18 合入〔`gh api`〕。
    - 只有快取中（閒置）的狀態用 int8，per-(head, k-channel) 對稱量化；活躍狀態維持全精度。
    - 命中時反量化一次，複製到新的 slot（copy-on-write）。
    - 原文理由：只會四捨五入一次，不會在遞迴裡反覆量化；int8 比 fp8 好，因為狀態分布接近均勻。
    - 效果寫作「~2x more cached states than bf16」，並說會和 host offload 疊加（mamba_checkpoint_pool.py L15–40）。
    - 旗標 `--enable-int8-mamba-checkpoint`（exec_.py L534–541）。
  - **host 與 L3 層**：
    - 有 `MambaPoolHost(HostKVCache)`（pool_host/mamba.py L75），以及 mamba 專用的 transfer kernel。
    - L3 儲存的命中規則把 Mamba／SWA 設為 `TRAILING_PAGES`：只要前綴的「最後 N 個 page」存在就算命中（hicache_storage.py L124–135、L325–345）。
    - 命中時，如果 device 上的狀態已被逐出、但 host 有備份，就觸發 load_back（unified_cache/components/mamba.py L186–191）。
    - 註解寫：HiCache 下新分叉狀態的增量持久化目前「write-through only」，write-back 逐出可能直接丟掉只在 device 上的狀態（同檔 L176–179）。
  - **逐出**：mamba 有自己的 LRU list（同檔 L127–149、L238）；通用策略有 LRU、LFU、FIFO、MRU、FILO、Priority、TLRU、SLRU（evict_policy.py）。**沒有用 FLOPs 或頻寬加權**。
  - blog〔文件〕：
    - 循環狀態「只在精確的前綴檢查點有效」。
    - SWA 與 MAMBA 先掃自己可重用區裡沒被引用的條目。
    - 現行逐出策略涵蓋 GPU L1 與 host L2，**不涵蓋 L3**。
    - Rust 原型支援 FULL、SWA、MAMBA，但不支援 HiCache。
    - 唯一的 hybrid SSM 數字：Qwen3-Next-80B-A3B、TP4、200 輪合成對話（每輪 100 in／100 out），Rust tree 比 Python tree 的 TTFT 低 5%（最後 25 輪低 7%）。
- **與既有整理不一致**：
  - SuffixReplay（arXiv 2609.33477，2026-09-27）說 SGLang 的狀態檢查點只能留在 HBM，「its states do not page」（p.12，見 `A_SuffixReplay.md`）。
  - HEAD 已有 `MambaPoolHost` 與 L3 的 TRAILING_PAGES。`gh api` 顯示 `pool_host/mamba.py` 在 2026-08-18 由 PR #31180 從別處搬過來，所以 host 層更早就有。
  - 可能是 SuffixReplay 用的設定（沒開 `--enable-hierarchical-cache`）或版本不同〔判讀〕。
- **線性代數的關鍵一步**：int8 的 scale 軸選 per-k-channel，理由是它和每通道的衰減 diag(α) 對齊，大值保持接近 bf16 精度（docstring）〔程式碼註解，非論文〕。
- **跟 Tiara 的關係**〔判讀〕：
  - 「狀態的精度」這一軸已經有人在 production 做了（int8，以及 DASC 的壓縮）。DIRECTIONS_20260927 方向 8 的「把 FP32 checkpoint 降成 BF16／FP8」已不是空白。
  - 「狀態的多層放置」機制也有了（host＋L3），只是決策都是 LRU，或「有就 load_back」。

### C5 LMCache：Hybrid-Attention Models 與 Qwen3.5 recipe（文件，沒有日期）
- **讀了什麼**：〔文件〕https://docs.lmcache.ai/mp/hybrid_models.html 、https://docs.lmcache.ai/recipes/qwen3_5.html （WebFetch 摘要，查證 2026-10-27；頁面只有「© 2024」，沒有版本或日期）。
- **一句話**：把 GDN／Mamba 的 conv＋SSM 狀態當成不透明 page 存。
- **事實**〔文件〕：
  - 「LMCache reinterprets that state as an opaque page at registration time」。
  - 必須用 vLLM 的 `--mamba-cache-mode align`（GDN 不支援 `all`），伺服器要開 `--separate-object-groups`。
  - chunk 大小跟著 vLLM 的「unified block size」走，例如 Qwen3.6-27B 是 784、Qwen3.5-0.8B 是 544。
  - 因為 page 不透明，CacheGen 與 CacheBlend 都不適用。
  - cached 與 fresh 的生成「not bit-exact」，因為 GDN backend 不支援 batch-invariant 模式。
  - 標「Validated with LMCache」（vLLM）的有 Qwen3.8-27B、Qwen3.6-27B、Qwen3.5-0.8B。
- **跟 Tiara 的關係**〔判讀〕：
  - LMCache 把狀態當普通 bytes 搬，沒有「搬 vs 重算」。
  - 「not bit-exact」對 Tiara 的無損定義（`CLAUDE.md` §0）有影響：在 hybrid 上，位元組無損不保證輸出逐字相同。DIRECTIONS_20260927 方向 8 已經記過 RFC #40865 的 14/20，這裡是第二個獨立來源。
- **與既有整理的關係**：`C_GLMHybridRecovery.md`（arXiv 2609.15030）是 vLLM＋LMCache 修補 11 個檔案的案例，固定每 1,792 token 存一份。現在官方 recipe 已經支援 align 模式，不需要修補〔文件，判讀〕。

### C6 vLLM RFC：Application-Directed Prefix Checkpoints for Hybrid GDN/Mamba（discuss.vllm.ai，2026-09-09，@nicholaskh-ai）
- **讀了什麼**：〔文件〕https://discuss.vllm.ai/t/rfc-pr-stack-application-directed-prefix-checkpoints-for-hybrid-gdn-mamba-up-to-7-6x-speedup-on-l40s/2949 （WebFetch 摘要）。
- **一句話**：檢查點放哪讓應用程式在 prompt 裡用 `<|mamba_checkpoint|>` 標出來，不讓引擎猜。
- **事實**〔文件〕：
  - 動機：遞迴架構「realistically afford only 1 or 2」個檢查點／序列，用啟發式幾乎不可能預測那個最有價值的邊界。
  - 量的是 grouped GDN prefill kernel，不是端到端：L40S、PyTorch 2.11；1 個 producer 配 16 個 consumer、共 1,536 token 時，12,698.95 µs 降到 1,674.81 µs，7.58 倍。
  - 沒有討論卸載，也沒有討論間隔。
- **跟 Tiara 的關係**〔判讀〕：對 Tiara 的暫停／回來，「應用程式知道的邊界」就是 session 結尾，引擎本來就會存。這篇解決的是分叉負載，不是 Tiara 的主情境。

### C7 llm-d：Serving Hybrid Models at Scale（blog，2026-06-13，IBM／Red Hat）
- **讀了什麼**：〔文件〕https://llm-d.ai/blog/serving-hybrid-models-at-scale-in-llm-d （WebFetch 摘要）。
- **事實**：
  - 只有一句說 Mamba／線性層的「checkpoints of intermediate states are maintained and offloaded」，沒有細節。
  - 所有數字都是 gpt-oss（SWA＋full），不是 SSM：例如 128K、10 個並發 prompt，HMA-aware 讀取快 1.8–1.9 倍（H100，IBM Storage Scale）〔文件〕。
- **跟 Tiara 的關係**：沒有 SSM 狀態的量測，不能引來支持任何 hybrid SSM 的結論〔判讀〕。

### C8 Sparse Prefix Caching（arXiv 2605.05219）——只指過去
- repo 卡 `A_SparsePrefixCaching.md` 已完整整理：在檢查點數量預算下用 O(NM) DP 選位置（一維加權 k-median），命中成本分三項（CPU 載入、重播、KV 載入），但**最佳化目標只算重算量**，checkpoint 只放 CPU。
- 這次搜尋（查詢字串見 §4）沒找到第二篇做「檢查點位置／間隔最佳化」的論文。所以它仍是唯一一篇，而且它的目標函數不含層級與頻寬〔判讀〕。

### C9 Young（1974）／Daly（2006）：最佳檢查點間隔，以及能不能套到 SSM 狀態
- **讀了什麼**：
  - 原文**沒讀到**：Young 在 ACM 付費牆後；Daly 在 ens-lyon 的連結是 404。
  - 公式取自 Aupy, Robert, Vivien, Zaidouni, "Checkpointing algorithms and fault prediction"，arXiv 1302.3752v2，p.1 與 §3（PDF 頁 1、4–5）〔二手〕。
  - 書目：Young, CACM 17(9):530–531, 1974；Daly, FGCS 22(3):303–312, 2006。Aupy 的參考文獻把 Daly 的年份寫成 2004，其他來源（LANL repository、acolyer blog）都是 2006〔二手，不一致記錄在此〕。
- **一句話**：在失敗隨機（MTBF μ）、每存一次要 C 時間的前提下，最小化浪費的週期是 T ≈ √(2μC)＋C（Young）。Daly 把恢復時間 R 加進去，變成 √(2(μ+R)C)＋C（Aupy p.1 式；Daly 一階式 T = √(2(μ+(D+R))C)＋C，Aupy 式 (9)）〔二手〕。
- **線性代數的關鍵一步**：無；是「檢查點成本 ∝ 1/T」與「失敗損失 ∝ T/2」的權衡，取一階近似後得到 √ 律〔二手〕。
- **映射到 hybrid 狀態**〔算術＋判讀〕：
  - 設一份狀態占 S bytes，容量的單價是 p（每 byte 的機會成本），序列長 L，每隔 T 個 token 存一份。
  - 若「需要回到中間某點」的位置在 [0, L] 上**均勻**出現，頻率 h，從最近的檢查點重算平均要 T/2 個 token，每 token 成本 c。
  - 每序列成本 f(T) = p·S·L/T ＋ h·c·T/2，所以 T* = √(2·p·S·L/(h·c))，和 Young 同形。
  - **但前提有四處對不上**：
    1. **回來點不是隨機的**：Tiara 的暫停／回來（append-only）回來點就是 prompt 結尾，事先知道。最佳解直接退化成「只在結尾存一份」，h（需要中間點的頻率）≈ 0，所以 T* → ∞。這正是 vLLM retention=0 與 SGLang「請求結束時存」的做法〔程式碼見 C2、C4〕。
    2. **分叉深度的分布不是無記憶的**：只有分叉型負載（共用 system prompt、同文件多題）才需要中間點。分叉深度的分布由負載決定（Sparse Prefix Caching 用經驗分布），不是 Poisson，所以應該用加權 k-median DP，不是 √ 公式。
    3. **重算成本跟位置有關**：從檢查點 c 重算到 k 要過全模型，包括注意力層。注意力層每 token 的成本隨絕對位置線性增加（Tiara 已知的 Cake 現象）。所以「損失 ∝ T/2」只在注意力占比很小時近似成立。
    4. **「存」的代價主要是容量，不是時間**：Young／Daly 的 C 是每一期都要付的時間。這裡的狀態在 prefill 時本來就會算出來，存一份主要占容量（GPU／CPU／SSD 都有價）與寫入頻寬。還多了一個 Young／Daly 沒有的選擇：檢查點放在哪一層，載回時間 = S／該層頻寬。
  - **結論**〔判讀〕：Young／Daly 的 √ 律可以當「分叉深度近似均勻」時的直覺上界，但不是 Tiara 主情境的正確模型。真正需要的是「k-median 放置（Sparse Prefix Caching）× 每份放哪一層 × 命中時搬或從更早的檢查點重算」，三者合起來。
- **與既有整理的關係**：repo 沒有 Young／Daly 的卡（grep `young.*daly` 無結果）。

### C10 Mamba-2／SSD：Transformers are SSMs（ICML 2024；arXiv 2405.21060）
- **讀了什麼**：〔部分〕PDF p.2、p.17–18（§6、Thm 6.1、Total Cost）、p.32（§9.3、Fig. 10），https://arxiv.org/pdf/2405.21060 ，查證 2026-10-27。
- **一句話**：把 SSM 寫成半可分（semiseparable）矩陣，用分塊演算法讓計算以 matmul 為主。
- **線性代數的關鍵一步**：
  - 序列切成長 Q 的 chunk：對角塊用二次的注意力形式（matmul）；非對角塊是低秩，用「狀態傳遞」處理。
  - chunk 之間只剩一個長度 T/Q 的純量 scan。
  - N=P=Q 時，總 FLOPs O(TN²)，記憶體 O(TN)，工作量以 (N,N) matmul 為主（p.18 Total Cost）。
  - Thm 6.1：訓練 FLOPs O(TN²)，推論 O(TN)，推論記憶體 O(N²)（p.17）〔全文〕。
- **成立的前提**：模型本身要是 SSM／SSD 層；對 softmax 的 Llama／Qwen 不適用〔全文的前提，判讀〕。
- **量到的數字**〔全文〕：
  - Fig. 10（A100 80GB PCIe，state dim 64，序列長 512–512K）：SSD 比 Mamba 的 fused scan 快 2–8 倍；從序列長 2K 起比 FlashAttention-2 快。
  - p.2 寫「crossing over at sequence length 2K and 6× faster at sequence length 16K」。
  - batch、head 數等設定原文沒寫〔未查證〕。
- **跟 Tiara 的關係**〔判讀〕：
  - 「從檢查點重算到結尾」在 SSD 層是 chunk 平行的，只有一段很短的跨 chunk scan，所以 SSM 層本身不是重算的瓶頸。
  - 但 hybrid 重算要過全模型，注意力層、MLP／MoE 都要算。單層 kernel 的速度不能直接換成還原時間。

### C11 FLA（flash-linear-attention）README 基準（GB200）
- **讀了什麼**：〔文件〕https://raw.githubusercontent.com/fla-org/flash-linear-attention/main/README.md 的 §Benchmarks（約 L480–520），抓取時 main = `cad028c`（2026-10-27）。表頭寫的是 `main[0a484709]`，GB200、CUDA 12.9、PyTorch 2.9.0。
- **量到的數字**〔文件〕（fwd，ms；B／T／H／D 依序列出）：

| B | T | H | D | chunk_gdn | flash_attn |
|--:|--:|--:|--:|--:|--:|
| 1 | 8192 | 96 | 128 | 1.265 | 3.753 |
| 2 | 16384 | 16 | 128 | 1.029 | 5.035 |
| 4 | 4096 | 64 | 128 | 1.581 | 2.560 |
| 4 | 2048 | 16 | 128 | 0.753 | 0.346 |

  - README 沒說 flash_attn 是否 causal〔未查證〕。
  - 最長只到 16K，**沒有 128K 的數字**。
- **跟 Tiara 的關係**：單層 GDN 前向在 16K 時約是 FA2 的 1/5。到 128K 只能外推，不能當作量測〔判讀〕。

### C12 Kimi Linear（arXiv 2510.26692）：整個模型的 prefill 時間
- **讀了什麼**：〔部分〕PDF p.12–13（§5.6、Fig. 7）、p.16，https://arxiv.org/pdf/2510.26692 ，查證 2026-10-27。
- **量到的數字**〔全文〕：
  - 三個模型都用 Kimi Linear 48B 的設定（相同層數與 head 數），batch size 1：全 MLA、GDN-H、Kimi Linear。
  - prefill：4K–16K 時和 MLA 相當；Kimi Linear 比 MLA 快 2.3 倍（512K）、2.9 倍（1M）（Fig. 7a）。
  - decode：1M 時 TPOT 1.84 ms 對 11.48 ms，6.3 倍（Fig. 1b、p.16）。
  - 硬體原文沒寫〔未查證〕。
- **跟 Tiara 的關係**〔判讀〕：
  - 這是「hybrid 整個模型重算一段長前綴」最接近的量測。512K 時只比全注意力快 2.3 倍，遠低於 bytes 的縮小（KV 少 74%，加上 MLA 本來就小）。
  - 換句話說，hybrid 讓「搬」便宜的幅度大於讓「重算」便宜的幅度，κ 往搬偏。這和 lit_C §2 的 R0／R32 預測同向，但這裡只是兩個不同量測的比較，不是 κ 的實測。

### C13 Gated DeltaNet（ICLR 2025；arXiv 2412.06464）
- **讀了什麼**：〔部分〕PDF p.8–9（Fig. 3）。
- **量到的數字**〔全文〕：
  - 1.3B 模型的**訓練**吞吐，單張 H100，2K×16 到 16K×2。
  - Gated DeltaNet 和 DeltaNet 差不多，比 Mamba2 慢 2–3K tokens/s。
  - 2K 時 Transformer++ 最快（Fig. 3，p.9）。
- **跟 Tiara 的關係**：這是訓練吞吐，不是推論 prefill，也不到 128K，只能當背景〔判讀〕。

---

## §3 對 Tiara「暫停後回來」的影響（判讀，整理自 §1–§2）

1. **還原的組成**：還原 = 注意力 KV（隨長度成長，比 Llama 小 2–20 倍；Falcon-H1、OLMo-Hybrid 例外）＋ 一份結尾狀態（固定，27–870 MiB，必須整份搬，或者整段重算）。
2. **Cake 怎麼套**：Cake 只套在注意力 KV 上：前段 [0,b) 全模型重算，後段 [b,n) 只搬注意力 KV，再搬結尾狀態。前段重算時算出的遞迴狀態用不到，因為 decode 只要位置 n 的狀態（lit_C §3.5 的結構論證）。所以 hybrid 版的 Cake 多了一個固定的「搬狀態」時間 S/BW，可以和計算線重疊。
3. **κ 的移動**：bytes/token 縮 5–20 倍，但 FLOPs/token 只按 active 參數縮。所以會合點 b 會往前移，搬的比例變大。lit_C §2 已經對 Nemotron、Granite、Qwen3-Next 等算過 R0／R32（算術預測，非量測）。
4. **檢查點間隔只在三種情況有內容**：反向 Cake（先搬前段、後段重算，需要會合點的狀態）、DROP 尾段、分叉負載。Tiara 的 DROP＋重算動作在 hybrid 上要改寫：DROP 掉位置 c 之後的 block，回來時要有位置 c 的狀態，否則遞迴層得從 0 重算。這把「哪個位置有檢查點」和「DROP 哪些 block」綁在一起。
5. **無損**：LMCache 文件與 RFC #40865 都說 hybrid 的 cached 和 fresh 不 bit-exact。§6.5 那種逐字比對在 hybrid 上要改用「同一份狀態 bytes、同一 backend」的比對方式。

---

## §4 空白在哪（判讀）

**本次搜尋字串**：
- 「hybrid LLM SSM state offload CPU SSD prefix cache recompute vs load linear attention state 2026」
- 「Mamba state checkpoint interval prefix caching hybrid model optimal checkpoint placement arXiv」
- 「"linear attention" OR "Gated DeltaNet" state checkpoint tiered storage SSD restore recompute cost model hybrid serving paper」
- 「LMCache hybrid model Mamba state support Qwen3-Next offload」

另外在 vLLM、SGLang 原始碼中搜 `recompute`、`cost`、`interval`。

| 問題 | 已經有人做 | 還沒看到 |
|:--|:--|:--|
| 狀態要不要存（准入）、丟哪個（逐出） | Marconi（FLOP/byte，單層）；SGLang／vLLM 用 LRU；DASC 的雙層（dense／壓縮）LRU | 把頻寬、層級放進效用分數 |
| 狀態存在哪些位置 | Sparse Prefix Caching（DP，只算重算）；vLLM 固定 retention interval；SGLang 256／chunk 對齊＋每路徑上限；Jenga 每 512；RFC 由應用程式標記 | 用「容量價格×層級×重算成本」推導間隔或位置；本次沒找到 Young／Daly 被用在 SSM 狀態上的任何文獻 |
| 狀態的精度 | SGLang int8 檢查點池（2026-06）；DASC | 精度和放置層級聯合決定（屬故事 1 的延伸，非空白核心） |
| 狀態卸到 CPU／SSD | vLLM OffloadingConnector（align）；SGLang MambaPoolHost＋L3；LMCache 不透明 page；llm-d（無數字） | 這些全部是「有就搬」 |
| **命中時「搬狀態」vs「從更早的檢查點重算」vs「整段重算」** | 無 | **本次沒找到任何系統或論文做這個比較** |
| hybrid 上的 Cake 式還原（注意力 KV 分段＋狀態必搬） | 無（lit_C §3.5 有結構論證，沒有實作或量測） | 沒找到 |

**判讀**：
- 真正的空白是最後兩列：還原時的「搬 vs 重算」決策，以及它和檢查點位置、層級的聯合。
- 但對 Tiara 的主情境（append-only 暫停／回來、128K–256K）：只需要一份結尾狀態，它只占 0.2–15% 的 bytes，而且結構上必搬。這個空白的**實際收益可能很小**：
  - 狀態占比最高的是 Nemotron-H-56B 與 Kimi-K3，但前者原生只有 8K、後者有 2.78T 參數，都不是 Tiara 在 H200 上能跑的 128K 模型。
- 對 Tiara 影響較大的不是狀態，而是 hybrid 讓 KV bytes/token 變小 5–20 倍，κ 往「搬」偏。這要在 H200 上用一個 hybrid 模型（例如 Qwen3.5-35B-A3B 或 Nemotron-3-Nano-30B-A3B）實測 f(i) 才能確認〔NOT_MEASURED〕。
- 如果 Tiara 要在 hybrid 上保留 DROP＋重算，「DROP 尾段需要位置 c 的狀態」是一個 hybrid 特有、而且現有系統（vLLM #63565 就是這類失效）處理得很差的點。這比「間隔最佳化」更貼近 Tiara。

---

## §5 未能查證的項目

- Young（1974）與 Daly（2006）原文沒讀到（付費牆、404），公式取自 Aupy 等人 arXiv 1302.3752〔二手〕；Daly 的年份 2004／2006 有不一致。
- Jamba-1.5-Mini／Large 的 HF config 要授權；Large 的 Mamba 參數（expand、d_state）以 Jamba-v0.1 代入〔未查證〕。
- Kimi-Linear／Kimi-K3 的 MLA 快取實際存 512 還是 576 維（NoPE）沒核到；表中是上界。
- Qwen3-Next 的 ssm 狀態在 HF 參考實作的 dtype 沒核；表中給 BF16（vLLM auto）與 FP32 兩行。OLMo-Hybrid 沒有 dtype 欄位。
- Jenga 的「1344 token」用 config 算不出來；SOSP'25 只有二手來源。
- Mamba-2 Fig. 10 的 batch／head 設定、Kimi Linear Fig. 7 的硬體、FLA 的 flash_attn 是否 causal：原文都沒寫。
- LMCache 兩個文件頁、vLLM issue #63565、RFC、LMSYS blog、llm-d blog 都是 WebFetch 摘要，可能有改寫；關鍵數字（7.58×、5%／7%、1.8–1.9×）沒有逐字再核。
- SuffixReplay 說 SGLang「states do not page」，與 SGLang HEAD 的 MambaPoolHost 不一致；原因（版本或設定）〔未查證〕。
- 所有大小與比例都是算術，沒有在任何 GPU 上量過〔NOT_MEASURED〕。

---

## §6 參考資料（查證日 2026-10-27）

**模型 config（HF，`raw/main/config.json`）**
- R1 Qwen3-Next-80B-A3B-Instruct：https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/config.json （`9c7f2fb`）
- R2 Qwen3.5-35B-A3B：https://huggingface.co/Qwen/Qwen3.5-35B-A3B/raw/main/config.json （`59d61f3`）；Qwen3.5-27B（`fc05dae`）；Qwen3.6-35B-A3B（`995ad96`），路徑同形
- R3 Kimi-Linear-48B-A3B-Instruct：https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/raw/main/config.json （`e1df551`）
- R4 Kimi-K3：https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json （`f831ab6`）
- R5 Nemotron-H-8B-Base-8K（`94ea861`）、Nemotron-H-56B-Base-8K（`1fb0949`）、NVIDIA-Nemotron-Nano-9B-v2（`6533e8d`）、NVIDIA-Nemotron-3-Nano-30B-A3B-BF16（`bf77c31`）、NVIDIA-Nemotron-3-Super-120B-A12B-BF16（`2dc98e2`）：https://huggingface.co/nvidia/<name>/raw/main/config.json
- R6 Jamba-v0.1（`9efd115`）、AI21-Jamba2-Mini（`24cbbd2`）：https://huggingface.co/ai21labs/<name>/raw/main/config.json
- R7 MiniMax-Text-01（`a7351bf`）、MiniMax-M1-80k（`8d1494b`）：https://huggingface.co/MiniMaxAI/<name>/raw/main/config.json ；modeling：https://huggingface.co/MiniMaxAI/MiniMax-Text-01/raw/main/modeling_minimax_text_01.py
- R8 Falcon-H1-34B-Instruct（`6e6890e`）、Falcon-H1-7B-Instruct（`41e72f2`）：https://huggingface.co/tiiuae/<name>/raw/main/config.json
- R9 granite-4.0-h-small（`b8c0982`）、granite-4.0-h-tiny（`791e0d3`）：https://huggingface.co/ibm-granite/<name>/raw/main/config.json
- R10 OLMo-Hybrid-7B：https://huggingface.co/allenai/OLMo-Hybrid-7B/raw/main/config.json

**論文**
- R11 R. Pan et al., "Marconi: Prefix Caching for the Era of Hybrid LLMs," MLSys 2025. https://arxiv.org/abs/2411.19379
- R12 C. Zhang et al., "Jenga: Effective Memory Management for Serving LLM with Heterogeneity," SOSP 2025（二手）. https://arxiv.org/abs/2503.18292 ；venue 來源：https://pacman.cs.tsinghua.edu.cn/~zjd/publication/dblp-confsosp-zhang-dlkmwlyllz-25
- R13 T. Dao, A. Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality," ICML 2024. https://arxiv.org/abs/2405.21060
- R14 Kimi Team, "Kimi Linear: An Expressive, Efficient Attention Architecture," 2025. https://arxiv.org/abs/2510.26692
- R15 S. Yang, J. Kautz, A. Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta Rule," ICLR 2025. https://arxiv.org/abs/2412.06464
- R16 Jamba Team, "Jamba-1.5: Hybrid Transformer-Mamba Models at Scale," 2024. https://arxiv.org/abs/2408.12570
- R17 J. W. Young, "A first order approximation to the optimum checkpoint interval," Commun. ACM 17(9):530–531, 1974. https://doi.org/10.1145/361147.361115 （原文未讀）
- R18 J. T. Daly, "A higher order estimate of the optimum checkpoint interval for restart dumps," Future Gener. Comput. Syst. 22(3):303–312, 2006. https://laro.lanl.gov/esploro/outputs/journalArticle/A-higher-order-estimate-of-the/9916364420003761 （原文未讀）
- R19 G. Aupy, Y. Robert, F. Vivien, D. Zaidouni, "Checkpointing algorithms and fault prediction," arXiv 1302.3752v2. https://arxiv.org/abs/1302.3752 （R17、R18 的二手來源）
- R20 Sparse Prefix Caching（arXiv 2605.05219）、SuffixReplay（2609.33477）、DASC（2608.30386）、GLM hybrid recovery（2609.15030）、LinearKV（2608.11231）：見 repo 卡 `docs/research_20261009_explore/cards/` 與 `docs/PAPERS_BY_LEVEL.md` L0

**程式碼與文件**
- R21 vLLM，commit `115b59a2da86c79dc1795f5845adcc59163084ff`（2026-10-27）：https://github.com/vllm-project/vllm/tree/115b59a2da86c79dc1795f5845adcc59163084ff （檔案與行號見 §1、C2）
- R22 vLLM issue #63565：https://github.com/vllm-project/vllm/issues/63565 ；PR #52527、#55403、#58549、#62932（`gh pr list`，皆未合併）
- R23 vLLM RFC「Application-Directed Prefix Checkpoints」：https://discuss.vllm.ai/t/rfc-pr-stack-application-directed-prefix-checkpoints-for-hybrid-gdn-mamba-up-to-7-6x-speedup-on-l40s/2949
- R24 SGLang，commit `3927a73a340708192fb41a4cc2f6c578df734ce0`（2026-10-27）：https://github.com/sgl-project/sglang/tree/3927a73a340708192fb41a4cc2f6c578df734ce0 ；int8 檢查點池 PR #28185（2026-06-18 合入）
- R25 LMSYS，"Unified Radix Cache: One Tree for Hybrid Model Prefix Caching," 2026-08-11：https://www.lmsys.org/blog/2026-08-11-unified-radix-cache
- R26 LMCache 文件：https://docs.lmcache.ai/mp/hybrid_models.html ；https://docs.lmcache.ai/recipes/qwen3_5.html
- R27 llm-d，"Serving Hybrid Models at Scale in llm-d," 2026-06-13：https://llm-d.ai/blog/serving-hybrid-models-at-scale-in-llm-d
- R28 FLA README Benchmarks（main `cad028c`，表頭 `0a484709`）：https://github.com/fla-org/flash-linear-attention/blob/main/README.md#benchmarks
