# Lit-C 模型架構與模態：KV 大小、重算成本、寫入時才有的東西

> **一句話**：用 config.json 算 KV 大小與 prefill 計算量，再用 Llama-3.1-8B 在 MI300X 上實測的 f(i) 校準，預測各模型的 b/n。只有 MHA（LongAlpaca、Qwen1.5）明顯讓 b 變大；MoE 的優勢在 32K 被注意力計算吃掉；近年的主流架構（KV head 少的 GQA、MLA、SSM／線性混合）都讓 b 變小。「寫入時才有」的東西確實存在（SSM 中間狀態、hidden state、vision embedding），但 SSM 檢查點的位置選擇已經有人做了（Sparse Prefix Caching），而且 Cake 還原同一段 prefix 並**不需要**中間的 SSM 狀態。

**日期**：2026-10-09
**作者**：Lit-C agent（不碰 GPU）
**上游**：[10_breakthrough_plan.md](../phase1_20261008/10_breakthrough_plan.md) §3A、§3B、§4（H2、H5、H6、H7、H9、H14）；[11_round1_plan.md](../phase1_20261008/11_round1_plan.md)
**證據標記**：〔原文 p.X〕論文 PDF 實體頁；〔程式碼〕附檔名與行號；〔算術〕自己用公式算；〔判讀〕推論；〔未查證〕。**本文件沒有任何新的 GPU 量測**；唯一用到的量測是既有的 `calib_c1.csv`（run_id `20261008-130316-m7-c1`）。

---

## 0. 重點

1. **只有 MHA 明顯讓 b 變大**〔算術〕。以 Llama 實測 f(i) 校準後，32K 時 LongAlpaca-7B／Qwen1.5-7B 的預測 b/n 是 0.61（3.69 GiB/s）、0.17（35.4 GiB/s）；Llama-3.1-8B 是 0.31、0.05。**沒有任何模型在 35.4 GiB/s 時達到 30%**。
2. **11 的判準要注意**：Llama-3.1-8B 自己用實測 f(i) 算，3.69 GiB/s 時 b/n 已經是 0.312〔算術，`calib_c1.csv`〕，所以「3.69 或 35.4 時 b/n ≥30%」在 3.69 這一格，基準本身就過關了。建議改看 35.4，或看「相對 Llama 的增幅」〔判讀〕。
3. **MoE 的「重算便宜」在長 context 會消失**〔算術〕。Qwen3-30B-A3B 每 token 的線性計算只有 Llama 的 0.39 倍，但它有 48 層 × 32 個 query head，注意力 FLOPs 是 Llama 的 1.5 倍。比值 R（KV bytes ÷ 每 token FLOPs，相對 Llama）在短 context 是 1.92，32K 平均只剩 0.92；預測 b/n 0.33 對 Llama 的 0.31，幾乎一樣。這還沒算 HF 的 MoE kernel 比 dense GEMM 慢。
4. **「寫入時才有」的東西，依架構不同**：SSM／線性層的中間狀態（N1）、每層的 hidden state（N1＋N4，只有 MHA、Zamba2、Gemma-2／3 比 KV 小）、vision embedding（N4，是 KV 的 1/8 到 22%）、Llama-3.2-Vision 的 cross-attention KV（和位置無關）。詳見 §3、§4。
5. **H7 的動機要修正**〔判讀〕：Cake 還原「同一段 prefix」時，計算線從 token 0 往後算，本來就會算出 SSM 狀態；載入線只載注意力層的 KV；接著 decode 只要最後一個狀態。所以**不需要會合點的 SSM 狀態**。中間檢查點只在「分叉／部分重疊」「反向 Cake（先載前段）」「尾段被丟掉要重算」時才需要。Lit-A 的 [A_SparsePrefixCaching](cards/A_SparsePrefixCaching.md) 也得到同樣的結論；本文件 §3.5 給出結構上的理由。
6. **H7 的空白縮小了**：「寫入時選哪些位置存 SSM 檢查點」已經有 DP 最佳解（Sparse Prefix Caching，2026-04）；有損的替代做法是存 hidden 再重播（Tail-Replay、SuffixReplay）。還沒人做的是：**SSM 檢查點的分層放置**（SGLang 的檢查點只能留在 HBM，原文說「its states do not page」〔SuffixReplay 原文 p.12〕），以及它和 Cake 式還原的互動。
7. **ROCm**：本機的 vLLM 0.19.1+rocm722 除了 Llama-3.2-Vision（Mllama 已移除）以外，表上的架構都有註冊；Mamba1 的 CUDA op 有編進本機的 ROCm build，Mamba2／GDN／KDA 走 Triton。HF transformers 5.17 在本機沒有 mamba_ssm／causal_conv1d／fla，SSM 會走純 torch 路徑。**接上 OffloadingConnector 時，vLLM 會關掉混合 KV 管理器（HMA）**：滑動視窗層改存全部 KV，SSM 混合模型照程式碼會直接報錯〔程式碼判讀，未實測〕。詳見 §5。
8. **推薦下一批量 f(i) 的 3 個模型**（§6）：① Qwen1.5-7B-Chat（MHA、原生 32K）② Gemma-3-12B（全部層 KV 384 KiB/token，只看全域層是 64 KiB）③ Llama-3.2-3B（小模型、harness 不用改）。

---

## 1. 方法（全部是算術；校準用既有量測）

### 1.1 KV／token（BF16）
- GQA／MHA／MQA：每個注意力層 2 × KV head 數 × head_dim × 2 bytes，乘以注意力層數。
- MLA（DeepSeek-V2／V3、Kimi-K2、Kimi-Linear 的 MLA 層）：每層存一個 (kv_lora_rank＋qk_rope_head_dim)＝512＋64＝576 維的 latent，× 2 bytes＝1,152 bytes。vLLM 與 HF transformers 5.17 都存壓縮後的 latent〔程式碼：transformers `models/deepseek_v3/modeling_deepseek_v3.py` L456–471「Cache read / write is performed while latent KV is still compressed」〕。
- 滑動視窗（SWA）：表上的 KV/token 是「全部層都存」的大小；括號內是只存全域層的大小。
- SSM／線性注意力層：沒有 per-token KV，只有固定大小的狀態，另列在 §3.5。
- Llama-3.2-Vision 的 8 個 cross-attention 層：文字 token 不存 KV；圖片 token 的 cross-attn KV 另列在 §3.7。

### 1.2 每 token 的 prefill 計算量
- 線性部分：2 × N_active（非嵌入、**不含 LM head**；重算 KV 不需要 logits）。MoE 只算 top-k 專家＋共享專家＋router；Zamba2 的共享區塊按呼叫次數算（13 次）。
- 注意力部分：位置 p 的 token，每個注意力層 2 × q head 數 ×（d_qk＋d_v）× min(p, W) FLOPs（W 是視窗；全域層 W＝∞）。MLA 用 prefill 解壓形式（d_qk＝192、d_v＝128）。
- SSM／線性層的 scan：粗估（Mamba2：6·H·P·N＋chunk 內 2·(chunk/2)·H·(N＋P)；Mamba1：9·d_inner·d_state；GDN／KDA／lightning：4·H·d_k·d_v＋2·32·H·(d_k＋d_v)）。都不到總量的 10%〔算術〕。
- R0、R32：(KV bytes ÷ 每 token FLOPs) 除以 Llama-3.1-8B 的同一個量。R0 取 p→0（只有線性部分），R32 取位置 0–32K 的平均。R>1 表示「載入相對重算比 Llama 貴」，κ 較大，b 較大。

### 1.3 用 Llama 實測 f(i) 校準，預測 b/n
- 資料：`results/m7_write_policy_mi300x/calib_c1.csv`，run_id `20261008-130316-m7-c1`，80 個 chunk × 3 次，取中位數。
- 模型：f(i)＝α·(線性 FLOPs of chunk i)＋β·(注意力 FLOPs of chunk i)。最小平方得 α＝3.96e-12 ms/FLOP（等效 252 TFLOPS）、β＝6.46e-12 ms/FLOP（等效 155 TFLOPS），80 個點的最大相對誤差 0.6%〔算術〕。
- 預測：把各模型的 FLOPs 代入同一組 α、β 得 f_pred(i)；ℓ＝512 × KV/token ÷ 頻寬；用和 `code/m7_restore_harness.py` L218–227 `write_boundary` 相同的式子算 b，n＝64（32K）。
- 核對：Llama 用實測 f 與用 f_pred 算出的 b/n 在五個頻寬都相同（0.797／0.578／0.312／0.141／0.047）〔算術〕。
- **限制**〔判讀〕：假設每個模型的 MFU 和 Llama 在 m7 harness 上一樣。實際上小模型（hidden 小、GEMM 小）、MoE（每個專家只分到少數 token）、HF 純 torch 的 SSM 路徑，MFU 都可能更低 → f 更大 → b **更小**。所以這些模型的預測是**偏樂觀的上限**。大模型可能反過來。
- 公式的交叉驗證（CLAUDE.md 規則 6 的精神）：用同一套公式算出的**總參數數**，和 HF API 的 safetensors 參數總數比對，純文字的 dense、MoE、MLA、SSM 混合模型都在 1% 內。差異較大的都有原因：DeepSeek-V3（671.0B 對 684.5B）與 Qwen3-Next（79.7B 對 81.3B）的 safetensors 含 MTP 層；Zamba2 少算 LoRA adapter（7.30B 對 7.36B）；VLM 與 Gemma-3 只算文字部分，差額等於 vision tower（例：Qwen2.5-VL-7B 7.62B 對 8.29B，差 0.67B，對上 ViT 630M＋merger 45M）〔算術〕。

---

## 2. 大表

欄位說明：
- 「層數／注意力層」：L＝總層數；`a×(q head, KV head, head_dim)` 表示有 a 個這樣的注意力層；`W=` 是滑動視窗。
- 「預測 b/n」：32K、Cake 會合點落在前段的比例；**算術預測，不是量測**（§1.3）。SWA 模型這欄假設全部層的 KV 都存（vLLM 接 connector 時的實際行為，§5），只存全域層的結果見 §3.4。
- 「config 來源」：本機路徑的前綴都是 `/mlsteam/data/tiara/hf-cache/hub/`，後接 `/snapshots/*/config.json`。HF 原檔需要授權的（meta-llama、google、CohereLabs、ai21labs、Zyphra/Zamba2-7B），改用表中的鏡像，結構欄位與原檔是否完全一致〔未查證〕。
- 「ROCm／HF 支援」：指本機 vLLM 0.19.1+rocm722（`/mlsteam/workspace/src/vllm`，commit `b1388b1`）的註冊與 kernel 狀況，以及本機 transformers 5.17.0。**除了 Llama-3.1-8B，全部沒有在 MI300X 上實際跑過（NOT_MEASURED）**。依據見 §5。
- Llama-2-7B（`NousResearch/Llama-2-7b-hf`）的數字和 LongAlpaca-7B 完全相同，不重列。

| 模型 | 架構 | 層數／注意力層×(q head, KV head, head_dim) | KV/token（KiB） | active（B，非嵌入） | GFLOP/token（p→0／32K 平均） | R0／R32 | 預測 b/n @3.69／@35.4 | config 來源 | ROCm／HF 支援 |
|:--|:--|:--|--:|--:|--:|--:|--:|:--|:--|
| Llama-3.1-8B | GQA（基準） | L=32；32×(32,8,128) | 128 | 6.98 | 14.0／22.5 | 1.00／1.00 | 0.31／0.05 | 本機 `models--unsloth--Llama-3.1-8B-Instruct` | vLLM ✓（第一階段實測）；HF ✓（m7 harness 實測） |
| LongAlpaca-7B | MHA | L=32；32×(32,32,128) | 512 | 6.48 | 13.0／21.5 | 4.31／4.19 | 0.61／0.17 | 本機 `models--Yukang--LongAlpaca-7B` | vLLM：註冊✓；HF ✓ |
| Qwen2.5-7B-1M | GQA | L=28；28×(28,4,128) | 56 | 6.53 | 13.1／19.6 | 0.47／0.50 | 0.19／0.02 | 本機 `models--Qwen--Qwen2.5-7B-Instruct-1M` | vLLM：註冊✓；HF ✓ |
| Qwen2.5-14B-1M | GQA | L=48；48×(40,8,128) | 192 | 13.21 | 26.4／42.5 | 0.79／0.80 | 0.27／0.03 | 本機 `models--Qwen--Qwen2.5-14B-Instruct-1M` | vLLM：註冊✓；HF ✓ |
| Mistral-Nemo-12B | GQA | L=40；40×(32,8,128) | 160 | 10.91 | 21.8／32.5 | 0.80／0.87 | 0.28／0.03 | 本機 `models--mistralai--Mistral-Nemo-Instruct-2407` | vLLM：註冊✓；HF ✓ |
| Nemotron-8B-UltraLong-1M | GQA | L=32；32×(32,8,128) | 128 | 6.98 | 14.0／22.5 | 1.00／1.00 | 0.31／0.05 | 本機 `models--nvidia--Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct` | vLLM：註冊✓；HF ✓ |
| Seed-OSS-36B | GQA（大） | L=64；64×(80,8,128) | 256 | 34.56 | 69.1／112.1 | 0.40／0.40 | 0.17／0.02 | 本機 `models--ByteDance-Seed--Seed-OSS-36B-Instruct` | vLLM：註冊✓；HF ✓ |
| Qwen3-30B-A3B | MoE＋GQA | L=48；48×(32,4,128) | 96 | 2.73 | 5.5／18.3 | 1.92／0.92 | 0.33／0.08 | 本機 `models--Qwen--Qwen3-30B-A3B-Instruct-2507` | vLLM：註冊✓；HF ✓ |
| Llama-3.2-1B | GQA（小） | L=16；16×(32,8,64) | 32 | 0.97 | 1.9／4.1 | 1.79／1.38 | 0.39／0.08 | [unsloth/Llama-3.2-1B-Instruct](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/blob/main/config.json) | vLLM：註冊✓；HF ✓ |
| Llama-3.2-3B | GQA（小） | L=28；28×(24,8,128) | 112 | 2.82 | 5.6／11.3 | 2.17／1.75 | 0.44／0.09 | [unsloth/Llama-3.2-3B-Instruct](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct/blob/main/config.json) | vLLM：註冊✓；HF ✓ |
| Qwen2.5-0.5B | GQA（小） | L=24；24×(14,2,64) | 12 | 0.36 | 0.7／2.1 | 1.83／0.99 | 0.34／0.08 | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/config.json) | vLLM：註冊✓；HF ✓ |
| Qwen2.5-1.5B | GQA（小） | L=28；28×(12,2,128) | 28 | 1.31 | 2.6／5.4 | 1.17／0.91 | 0.31／0.06 | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json) | vLLM：註冊✓；HF ✓ |
| Qwen2.5-3B | GQA（小） | L=36；36×(16,2,128) | 36 | 2.77 | 5.5／10.4 | 0.71／0.61 | 0.23／0.03 | [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/config.json) | vLLM：註冊✓；HF ✓ |
| Llama-3.1-70B | GQA（大） | L=80；80×(64,8,128) | 320 | 68.45 | 136.9／179.9 | 0.25／0.31 | 0.12／0.02 | [unsloth/Meta-Llama-3.1-70B-Instruct](https://huggingface.co/unsloth/Meta-Llama-3.1-70B-Instruct/blob/main/config.json) | vLLM：註冊✓；HF ✓ |
| Qwen2.5-72B | GQA（大） | L=80；80×(64,8,128) | 320 | 70.21 | 140.4／183.4 | 0.25／0.31 | 0.11／0.00 | [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/config.json) | vLLM：註冊✓；HF ✓ |
| Qwen3-8B | GQA | L=36；36×(32,8,128) | 144 | 6.95 | 13.9／23.6 | 1.13／1.08 | 0.33／0.06 | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json) | vLLM：註冊✓；HF ✓ |
| LongAlpaca-13B | MHA | L=40；40×(40,40,128) | 800 | 12.69 | 25.4／38.8 | 3.44／3.63 | 0.58／0.16 | [Yukang/LongAlpaca-13B](https://huggingface.co/Yukang/LongAlpaca-13B/blob/main/config.json) | vLLM：註冊✓；HF ✓ |
| Qwen1.5-7B | MHA | L=32；32×(32,32,128) | 512 | 6.48 | 13.0／21.5 | 4.31／4.19 | 0.61／0.17 | [Qwen/Qwen1.5-7B-Chat](https://huggingface.co/Qwen/Qwen1.5-7B-Chat/blob/main/config.json) | vLLM：註冊✓；HF ✓ |
| Qwen1.5-14B | MHA | L=40；40×(40,40,128) | 800 | 12.61 | 25.2／38.6 | 3.46／3.65 | 0.58／0.16 | [Qwen/Qwen1.5-14B-Chat](https://huggingface.co/Qwen/Qwen1.5-14B-Chat/blob/main/config.json) | vLLM：註冊✓；HF ✓ |
| Mistral-7B-v0.1 | SWA（全部層 4096） | L=32；32×(32,8,128) W=4096 | 128（全域層 0） | 6.98 | 14.0／16.0 | 1.00／1.41 | 0.33／0.05 | [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json) | vLLM：註冊✓，ROCm 後端支援 sliding_window；接 KV connector 時 HMA 關閉 |
| Gemma-2-9B | SWA 1:1（4096） | L=42；21×(16,8,256) W=4096＋21×(16,8,256) | 336（全域層 168） | 8.32 | 16.6／23.6 | 2.20／2.51 | 0.48／0.11 | [unsloth/gemma-2-9b-it](https://huggingface.co/unsloth/gemma-2-9b-it/blob/main/config.json) | vLLM：註冊✓，ROCm 後端支援 sliding_window；接 KV connector 時 HMA 關閉 |
| Gemma-2-2B | SWA 1:1（4096） | L=26；13×(8,4,256) W=4096＋13×(8,4,256) | 104（全域層 52） | 2.02 | 4.0／6.2 | 2.80／2.95 | 0.53／0.12 | [unsloth/gemma-2-2b-it](https://huggingface.co/unsloth/gemma-2-2b-it/blob/main/config.json) | 同上 |
| Gemma-3-1B | SWA 5:1（512）＋MQA | L=26；22×(4,1,256) W=512＋4×(4,1,256) | 26（全域層 4） | 0.70 | 1.4／1.7 | 2.03／2.68 | 0.50／0.09 | [unsloth/gemma-3-1b-it](https://huggingface.co/unsloth/gemma-3-1b-it/blob/main/config.json) | 同上 |
| Gemma-3-4B | SWA 5:1（1024） | L=34；29×(8,4,256) W=1024＋5×(8,4,256) | 136（全域層 20） | 3.21 | 6.4／7.3 | 2.31／3.27 | 0.55／0.11 | [unsloth/gemma-3-4b-it](https://huggingface.co/unsloth/gemma-3-4b-it/blob/main/config.json) | 同上 |
| Gemma-3-12B | SWA 5:1（1024） | L=48；40×(16,8,256) W=1024＋8×(16,8,256) | 384（全域層 64） | 10.76 | 21.5／24.3 | 1.95／2.78 | 0.50／0.09 | [unsloth/gemma-3-12b-it](https://huggingface.co/unsloth/gemma-3-12b-it/blob/main/config.json) | 同上 |
| Gemma-3-27B | SWA 5:1（1024） | L=62；52×(32,16,128) W=1024＋10×(32,16,128) | 496（全域層 80） | 25.60 | 51.2／54.7 | 1.06／1.60 | 0.38／0.05 | [unsloth/gemma-3-27b-it](https://huggingface.co/unsloth/gemma-3-27b-it/blob/main/config.json) | 同上 |
| Command-R7B | SWA 3:1（4096） | L=32；24×(32,8,128) W=4096＋8×(32,8,128) | 128（全域層 32） | 6.98 | 14.0／17.6 | 1.00／1.28 | 0.33／0.05 | [mlx-community/c4ai-command-r7b-12-2024-bf16](https://huggingface.co/mlx-community/c4ai-command-r7b-12-2024-bf16/blob/main/config.json) | 同上 |
| Falcon-7B | MQA | L=32；32×(71,1,64) | 8 | 6.63 | 13.3／22.8 | 0.07／0.06 | 0.03／0.00 | [tiiuae/falcon-7b](https://huggingface.co/tiiuae/falcon-7b/blob/main/config.json) | vLLM：註冊✓；HF ✓ |
| Mixtral-8x7B | MoE＋GQA | L=32；32×(32,8,128) | 128 | 12.62 | 25.2／33.8 | 0.55／0.67 | 0.22／0.03 | [mistralai/Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json) | vLLM：註冊✓；HF ✓ |
| gpt-oss-20b | MoE＋SWA 1:1（128） | L=24；12×(64,8,64) W=128＋12×(64,8,64) | 48（全域層 24） | 3.03 | 6.1／9.3 | 0.86／0.91 | 0.28／0.05 | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json) | vLLM：註冊✓，sinks 時 ROCm kernel 退回 Triton；HF ✓ |
| gpt-oss-120b | MoE＋SWA 1:1（128） | L=36；18×(64,8,64) W=128＋18×(64,8,64) | 72（全域層 36） | 4.55 | 9.1／14.0 | 0.86／0.91 | 0.28／0.05 | [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json) | 同上 |
| DeepSeek-V2-Lite | MLA＋MoE | L=27；27×MLA(16 head, latent 576) | 30.4 | 2.24 | 4.5／9.0 | 0.74／0.59 | 0.23／0.03 | [deepseek-ai/DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/config.json) | vLLM：註冊✓，MLA 走 TRITON_MLA（本機無 aiter） |
| DeepSeek-V3 | MLA＋MoE | L=61；61×MLA(128 head, latent 576) | 68.6 | 35.70 | 71.4／153.3 | 0.10／0.08 | 0.05／0.00 | [deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json) | 同上 |
| Kimi-K2 | MLA＋MoE | L=61；61×MLA(64 head, latent 576) | 68.6 | 30.51 | 61.0／102.0 | 0.12／0.12 | 0.06／0.00 | [moonshotai/Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json) | 同上 |
| Jamba-1.5-Mini | Mamba1＋注意力 1:7＋MoE | L=32；4×(32,8,128)；SSM 28 層 | 16 | 11.57 | 23.2／24.3 | 0.08／0.12 | 0.03／0.00 | [FlagRelease/AI21-Jamba-1.5-Mini-nvidia-FlagOS](https://huggingface.co/FlagRelease/AI21-Jamba-1.5-Mini-nvidia-FlagOS/blob/main/config.json)（ai21labs 原檔需授權；`ai21labs/Jamba-v0.1` 結構欄位相同） | vLLM：註冊✓，Mamba1 CUDA op 有編進本機 ROCm build；HF：純 torch 路徑 |
| Zamba2-7B | Mamba2＋共享注意力（13 次） | L=81；13×(32,32,224)；SSM 81 層 | 364 | 10.86 | 22.2／28.3 | 1.78／2.26 | 0.47／0.09 | [Zyphra/Zamba2-7B-Instruct](https://huggingface.co/Zyphra/Zamba2-7B-Instruct/blob/main/config.json) | vLLM：註冊✓，Mamba2 Triton；HF：純 torch 路徑（無 mamba_ssm） |
| Falcon-H1-7B | 平行混合（每層注意力∥Mamba2） | L=44；44×(12,2,128)；SSM 44 層 | 44 | 6.79 | 13.9／18.3 | 0.35／0.42 | 0.16／0.02 | [tiiuae/Falcon-H1-7B-Instruct](https://huggingface.co/tiiuae/Falcon-H1-7B-Instruct/blob/main/config.json) | 同上 |
| Nemotron-H-8B | Mamba2 24＋注意力 4＋MLP 24 | L=52；4×(32,8,128)；SSM 24 層 | 16 | 7.03 | 14.3／15.4 | 0.12／0.18 | 0.06／0.00 | [nvidia/Nemotron-H-8B-Base-8K](https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K/blob/main/config.json) | 同上 |
| Nemotron-Nano-9B-v2 | Mamba2 27＋注意力 4＋MLP 25 | L=56；4×(40,8,128)；SSM 27 層 | 16 | 7.71 | 15.7／17.1 | 0.11／0.17 | 0.06／0.00 | [nvidia/NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2/blob/main/config.json) | 同上 |
| Granite-4.0-H-Small | Mamba2 36＋注意力 4＋MoE | L=40；4×(32,8,128)；SSM 36 層 | 16 | 8.39 | 17.2／18.3 | 0.10／0.15 | 0.05／0.00 | [ibm-granite/granite-4.0-h-small](https://huggingface.co/ibm-granite/granite-4.0-h-small/blob/main/config.json) | 同上 |
| Granite-4.0-H-Tiny | Mamba2 36＋注意力 4＋MoE | L=40；4×(12,4,128)；SSM 36 層 | 8 | 1.31 | 2.8／3.2 | 0.31／0.44 | 0.14／0.02 | [ibm-granite/granite-4.0-h-tiny](https://huggingface.co/ibm-granite/granite-4.0-h-tiny/blob/main/config.json) | 同上 |
| Qwen3-Next-80B-A3B | Gated DeltaNet 36＋gated 注意力 12＋MoE | L=48；12×(16,2,256)；線性 36 層 | 24 | 3.25 | 6.6／9.8 | 0.40／0.43 | 0.17／0.02 | [Qwen/Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json) | vLLM：註冊✓，FLA Triton；HF：純 torch 路徑（無 fla） |
| MiniMax-Text-01 | Lightning 線性注意力 70＋softmax 10＋MoE | L=80；10×(64,8,128)；線性 70 層 | 40 | 45.94 | 92.5／97.8 | 0.05／0.07 | 0.02／0.00 | [MiniMaxAI/MiniMax-Text-01](https://huggingface.co/MiniMaxAI/MiniMax-Text-01/blob/main/config.json) | vLLM：註冊✓（lightning Triton）；HF ✓ |
| Kimi-Linear-48B-A3B | KDA 20＋MLA 7＋MoE | L=27；7×MLA(32 head, latent 576)；線性 20 層 | 7.9 | 2.73 | 5.5／7.9 | 0.16／0.18 | 0.08／0.00 | [moonshotai/Kimi-Linear-48B-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/config.json) | vLLM：註冊✓，FLA Triton；HF：純 torch 路徑（無 fla） |
| Qwen2.5-VL-7B（LLM 部分） | VLM（GQA 骨幹） | L=28；28×(28,4,128) | 56 | 6.53 | 13.1／19.6 | 0.47／0.50 | 0.19／0.02 | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/config.json) | vLLM：註冊✓；HF ✓ |
| Qwen3-VL-8B（LLM 部分） | VLM（GQA 骨幹） | L=36；36×(32,8,128) | 144 | 6.95 | 13.9／23.6 | 1.13／1.08 | 0.33／0.06 | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json)（本機資料夾是空的） | vLLM：註冊✓；HF ✓ |
| LLaVA-OV-7B（LLM 部分） | VLM（GQA 骨幹） | L=28；28×(28,4,128) | 56 | 6.53 | 13.1／19.6 | 0.47／0.50 | 0.19／0.02 | [llava-hf/llava-onevision-qwen2-7b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf/blob/main/config.json) | vLLM：註冊✓；HF ✓ |
| InternVL3-8B（LLM 部分） | VLM（GQA 骨幹） | L=28；28×(28,4,128) | 56 | 6.53 | 13.1／19.6 | 0.47／0.50 | 0.19／0.02 | [OpenGVLab/InternVL3-8B](https://huggingface.co/OpenGVLab/InternVL3-8B/blob/main/config.json) | vLLM：註冊✓；HF ✓ |
| Llama-3.2-11B-Vision（文字自注意力） | VLM（cross-attn 8 層） | L=40；32×(32,8,128)＋8 個 cross-attn 層 | 128 | 8.66 | 17.3／25.9 | 0.81／0.87 | 0.28／0.05 | [unsloth/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/unsloth/Llama-3.2-11B-Vision-Instruct/blob/main/config.json) | vLLM ✗（Mllama 已移除）；HF ✓ |
| Falcon-Mamba-7B | 純 Mamba1（無 KV） | L=64；無注意力；SSM 64 層 | 0 | 6.74 | 13.6／13.6 | — | — | [tiiuae/falcon-mamba-7b](https://huggingface.co/tiiuae/falcon-mamba-7b/blob/main/config.json) | vLLM：註冊✓，Mamba1 op 有編進；HF：純 torch 路徑 |

**讀表**〔判讀〕：
- 讓 R 變大的只有兩種：KV head 多（MHA、Zamba2 的共享 MHA、Gemma 的 8 個 KV head × head_dim 256）、或 active 參數少而注意力不多（小模型、短 context 的 MoE）。
- 讓 R 變小的：KV head 少的 GQA（Qwen2.5 的 4／28、2／16）、MLA（約 0.1）、SSM／線性混合（0.05–0.44）、大模型（0.25–0.4）。
- 近兩年新出的大型模型幾乎都在「R 變小」那一邊。**「依位置放」最有用的條件（κ 接近 1）在主流新架構上越來越少見**。這是寫論文時要正面回答的威脅。

---

## 3. 各架構族：有沒有「寫入時才有、之後就沒有」的東西

N1＝寫完就消失的資訊；N4＝寫成什麼格式決定讀取時能做什麼（[10](../phase1_20261008/10_breakthrough_plan.md) §2）。

### 3.1 MHA／GQA／MQA（dense）
- **每層的 hidden state**（N1＋N4）：只在 forward 時存在；存成 KV 之後就回不去〔判讀，同 HCache〕。存 hidden state 是否比 KV 小，只看「hidden_size ÷（2 × KV head × head_dim）」，見 §4。MHA 是 0.5 倍（省一半），GQA 是 2–4 倍（更大），MQA（Falcon-7B）是 35 倍。
- **prefill 時的注意力分數**（N1）：所有架構都有，寫完就沒了（除非另存）。SnapKV、MuKV 這類有損壓縮靠它〔03_paper_map B 組〕。和第一階段的無損放置無關。
- 已讀的相關系統：HCache（E03）、HybridServe（[A_HybridServe](cards/A_HybridServe.md)，MHA、單批次卸載）、XQuant（[C_XQuant](cards/C_XQuant.md)，原文確認 Llama-3.1-8B 直接存 X 會多 2 倍記憶體〔原文 p.7〕）。

### 3.2 MLA（DeepSeek-V2／V3、Kimi-K2）
- **沒有**值得一提的 N1／N4：MLA 的 KV latent 本身就是 hidden state 的低秩投影，比 hidden 小 3.6–12.4 倍〔算術，§4〕，所以「改存 hidden」只會更大。
- R 很小（DeepSeek-V3 0.08、Kimi-K2 0.12）：載入非常便宜、重算非常貴，b≈0〔算術〕。這是 10 §3A 說的「反方向的對照」，算術上確認。

### 3.3 MoE（Qwen3-30B-A3B、Mixtral、gpt-oss、DeepSeek-V3）
- **N1**：router 的選擇是 hidden state 的確定性函數，重算會得到同樣的結果，不是「寫完就沒有」的資訊〔判讀〕。
- **「active 參數少 → 重算便宜」只在短 context 成立**〔算術〕：

| 模型 | R0（p→0） | R32（32K 平均） | 32K 時注意力占 FLOPs | b/n @3.69（8K／32K） | b/n @35.4（8K／32K） |
|:--|--:|--:|--:|--:|--:|
| Llama-3.1-8B（基準） | 1.00 | 1.00 | 38% | 0.38／0.31 | 0.00／0.05 |
| Qwen3-30B-A3B | 1.92 | 0.92 | 70% | 0.44／0.33 | 0.06／0.08 |
| gpt-oss-20b | 0.86 | 0.91 | 35% | 0.31／0.28 | 0.00／0.05 |
| Mixtral-8x7B | 0.55 | 0.67 | 25% | 0.25／0.22 | 0.00／0.03 |
| DeepSeek-V2-Lite | 0.74 | 0.59 | 50% | 0.25／0.23 | 0.00／0.03 |

（8K 時 n＝16，b/n 的刻度是 1/16＝0.0625。）

- Qwen3-30B-A3B 的注意力很重：48 層 × 32 q head × 128，每對 (q,k) 是 786,432 FLOPs，Llama 是 524,288〔算術〕。所以 32K 時它的優勢幾乎消失。
- **MoE 的實作效率**：m7 harness 走 HF transformers，`M7_EXPERTS_IMPL` 可選 eager（逐專家迴圈）或 grouped_mm〔程式碼：`code/m7_model.py` L42–43〕。512 token 的 chunk 分給 128 個專家、每 token 選 8 個，平均每個專家只有 32 個 token〔算術〕，GEMM 效率會比 dense 差，實測 f 會比本表預測大 → b 更小〔判讀〕。11 已經寫了這一點；本表給出「即使效率相同，32K 時也只有 0.33」的上限。

### 3.4 滑動視窗與局部／全域混合（Gemma 2／3、gpt-oss、Mistral 7B v0.1、Command-R7B）
- **N2／N4 型的決定，不是 N1**：局部層的舊 KV 是否需要，取決於「這個 chunk 離 session 結尾多遠」——還原整段 session 時，局部層只需要最後 W 個 token 的 KV〔判讀；Jenga 的命中規則：滑動視窗層只要最後 sliding_window_size 個 token 還在就算命中〔原文 p.8〕〕。
- 但延後版也做得到：搬到慢層時再把局部層的舊 KV 丟掉即可。寫入時決定只能省「寫入量」（N2）〔判讀；Lit-A 的 [A_Jenga](cards/A_Jenga.md) 也這麼判〕。
- **能省多少**〔算術，config〕：

| 模型 | 局部：全域 | 視窗 W | 全部層 KV/token | 只存全域層 | 32K 時舊 chunk 可省 | b/n @3.69（全部層／只全域） | b/n @35.4（全部層／只全域） |
|:--|:--|--:|--:|--:|--:|--:|--:|
| Gemma-2-9B | 1:1 | 4096 | 336 KiB | 168 KiB | 50% | 0.48／0.33 | 0.11／0.05 |
| Gemma-3-12B | 5:1 | 1024 | 384 KiB | 64 KiB | 83% | 0.50／0.14 | 0.09／0.02 |
| Gemma-3-27B | 5:1 | 1024 | 496 KiB | 80 KiB | 84% | 0.38／0.08 | 0.05／0.00 |
| Command-R7B | 3:1 | 4096 | 128 KiB | 32 KiB | 75% | 0.33／0.11 | 0.05／0.00 |
| gpt-oss-20b | 1:1 | 128 | 48 KiB | 24 KiB | 50% | 0.28／0.17 | 0.05／0.02 |
| Mistral-7B-v0.1 | 全部局部 | 4096 | 128 KiB | 0 | 只需最後 4096 token | 0.33／0 | 0.05／0 |

- 局部／全域的層型怎麼數：Gemma 2 是 `(i+1) % 2` 為真的層用滑動視窗（即 0、2、4… 層）〔程式碼：transformers `models/gemma2/configuration_gemma2.py` L95–97〕；Gemma 3 與 Command-R7B 用 config 的 `sliding_window_pattern`（6、4），每 pattern 層有 1 個全域層；gpt-oss 直接寫在 config 的 `layer_types`（12／12、18／18）〔config〕。
- **現況**：vLLM 只要設了 `--kv-transfer-config`，預設就關掉 HMA，滑動視窗層改成 FullAttentionSpec，「不會丟掉視窗外的 KV」（log 原話：「we do not enable any optimizations for saving KV cache memory (e.g., dropping the KV cache outside the sliding window)」）〔程式碼：vLLM `vllm/config/vllm.py` L1227–1244、`vllm/v1/core/kv_cache_utils.py` L1160–1216〕。所以**用 OffloadingConnector 卸載 Gemma／gpt-oss 時，局部層的舊 KV 會整份被存、被搬**〔程式碼判讀，未實測〕。H9 的節省在現行實作裡是真的拿得到的。
- **Mistral 7B v0.1 是特例**：全部層都是 4096 視窗，還原只需要最後 4096 token 的 KV；但要**重算**這 4096 token 的 KV，需要往前 32 層 × 4096 的感受野〔判讀，V01 的 ReKV 也有同樣的觀察〕。Cake 的「前段重算、後段載入」在這裡不成立：前段的 KV 根本不用還原。

### 3.5 SSM／線性注意力混合（Jamba、Zamba2、Falcon-H1、Nemotron-H、Granite-4.0-H、Qwen3-Next、MiniMax-01、Kimi-Linear）
- **N1：中間位置的遞迴狀態**。遞迴層原地更新 h_t＝F(h_{t−1}, x_t)，prefill 經過位置 c 之後就拿不到 h_c 了〔Sparse Prefix Caching 原文 p.1–2；GLM 案例原文 p.1「Reducing a token count cannot reconstruct an earlier recurrent state」〕。
- **狀態有多大**〔算術，config；vLLM 的狀態形狀公式見 `vllm/model_executor/layers/mamba/mamba_utils.py` L99–232；預設 dtype 跟模型 dtype（BF16），L55–67〕：

| 模型 | 遞迴層數 | 一份完整狀態（BF16） | 相當於幾個 token 的注意力 KV | 注意力 KV/token |
|:--|--:|--:|--:|--:|
| Jamba-1.5-Mini（Mamba1） | 28 | 8.3 MiB | 532 | 16 KiB |
| Falcon-H1-7B | 44 | 66.9 MiB | 1,557 | 44 KiB |
| Nemotron-H-8B | 24 | 49.4 MiB | 3,162 | 16 KiB |
| Nemotron-Nano-9B-v2 | 27 | 69.4 MiB | 4,442 | 16 KiB |
| Granite-4.0-H-Small | 36 | 73.7 MiB | 4,719 | 16 KiB |
| Zamba2-7B | 81 | 74.3 MiB | 209 | 364 KiB |
| Qwen3-Next-80B-A3B（GDN） | 36 | 37.7 MiB | 1,608 | 24 KiB |
| Kimi-Linear-48B-A3B（KDA） | 20 | 21.4 MiB | 2,783 | 7.9 KiB |
| MiniMax-Text-01（lightning） | 70 | 140 MiB | 3,584 | 40 KiB |

  - KVBuffer 說 Qwen3-Next 一個 GDN 層的狀態約 2 MB〔原文 p.1〕；本表算出 BF16 每層 1 MiB，FP32 時 2 MiB，對得上〔算術〕。SuffixReplay 說 Qwen3.5-4B 一個檢查點約 49 MiB、一個 token 的 KV 約 32 KiB〔Lit-A 卡引原文 p.2〕，數量級和本表一致。
  - 〔判讀〕一份檢查點相當於 200–4,700 個 token 的注意力 KV。所以「每個 block 都存一份」很貴（Marconi 的 vLLM+ 基線就是這樣），存哪些位置、存在哪一層是真的要決定的事。

- **對 10 §3A／H7 的修正**〔判讀，結構論證〕：
  - 10 §3A 寫「Cake 要從後面載入，需要會合點的 SSM 狀態」。這不對。理由：注意力層第 ℓ 層在 token t 的 K、V 只依賴 token ≤t 經過下面各層（含遞迴層）的結果；Cake 的計算線從 token 0、零狀態開始往後算 [0, b)，一路就把遞迴層的狀態算出來了，**不需要任何存下來的中間狀態**。載入線只載 [b, n) 的注意力層 KV。接下來 decode 只需要位置 n 的遞迴狀態（最後一份，加上 conv 的最後 k−1 個輸入），這份在寫入時本來就會存。
  - **中間檢查點只在這三種情況需要**：(a) 新請求只和舊的共用前 k 個 token（分叉、同一份文件問不同問題、system prompt）；(b) 反向 Cake（先載前段、後段重算）需要會合點的狀態；(c) 後段被丟掉、要從中間重算。
  - 第一階段的 doc 負載如果是「每次在同一份 32K 文件後面**接著**問」，只要最後一份狀態；如果是「每題都從文件結尾分叉」，需要的是**文件結尾那一點**的狀態——這正是第一次 prefill 結束時的狀態，SGLang（請求結束時存）、vLLM（align 模式）都會存〔原文／程式碼見下〕。所以只有「重疊深度不固定」的負載（Sparse Prefix Caching 的 NarrativeQA、System Prompts）才讓位置選擇有內容〔判讀〕。
  - Lit-A 的 [A_SparsePrefixCaching](cards/A_SparsePrefixCaching.md) 得到同樣結論；本節補上結構論證，並確認「Cake 在混合模型上不需要中間狀態」。
- **現行系統怎麼存檢查點**（都是寫入時的固定規則）：

| 系統 | 存在哪些位置 | 存在哪一層 | 出處 |
|:--|:--|:--|:--|
| vLLM 0.19.1 `mamba_cache_mode="all"` | 每個 block 邊界（block 大小選成「一頁注意力 KV ≥ 一份 mamba 狀態」，例如程式碼註解舉的 512 token） | GPU；實驗性功能 | 〔程式碼〕`vllm/config/cache.py` L25、L97–118；`vllm/model_executor/models/config.py` L243–296、L437–476 |
| vLLM 0.19.1 `"align"` | 每個排程步的最後一個 token，且落在 block 邊界時 | GPU | 同上 |
| SGLang（SuffixReplay 讀到的版本） | prefill chunk 邊界、請求結束（向下對齊 256）、某個 prefix 第一次被分叉延續時；預設每 8192 token 一份 | **HBM 固定 slot pool，不隨 KV 一起搬到 host**（「its states do not page」） | 〔SuffixReplay 原文 p.3、p.8、p.12〕 |
| Jenga | 每 512 token 一份 | GPU | 〔Jenga 原文 p.8〕 |
| Marconi | 依命中情境分類：投機插入時的分叉點、最後一個 decode token | 單層容量（模擬） | 〔E05 評測卡；[A_Marconi_補充](cards/A_Marconi_補充.md)〕 |
| Sparse Prefix Caching | 依重疊深度分布用 DP 選 M 個位置 | CPU RAM（原型） | 〔[A_SparsePrefixCaching](cards/A_SparsePrefixCaching.md)〕 |
| vLLM＋LMCache（GLM-5.3-Flash 案例） | 固定每 1,792 token | 外部 CPU cache（修補 11 個檔案） | 〔[C_GLMHybridRecovery](cards/C_GLMHybridRecovery.md) 原文 p.2〕 |
| Tail-Replay／SuffixReplay（有損） | 不存檢查點，改存 FA 輸出 hidden 或稀疏 anchor | host（SuffixReplay 的 anchor 隨 KV page 搬） | 〔[C_TailReplay](cards/C_TailReplay.md)、[A_SuffixReplay](cards/A_SuffixReplay.md)〕 |

- **Marconi 的 venue**：10 §5.6 寫「MLSys'25？要查證」。已查證：arXiv 2411.19379v3 的 comment 寫「MLSys 2025 camera-ready version」，PDF 頁腳是 MLSys'25 proceedings〔E05 評測卡；[A_Marconi_補充](cards/A_Marconi_補充.md)〕。正確。
- **H7 還剩的空白**〔判讀〕：(1) 檢查點的分層放置（GPU／CPU／SSD）——SGLang 的狀態不會分頁，vLLM 的 OffloadingConnector 遇到混合模型會報錯（§5）；(2) 檢查點位置和 Cake 式還原一起決定：從檢查點 c 重算到 k，成本是 (k−c) 個 token 的全模型計算，而且越後面越貴（注意力層）；(3) 無損。三者合起來還沒看到有人做。

### 3.6 小模型（1B–3B）與大模型（70B）
- 小模型的 R 介於 0.6（Qwen2.5-3B，KV head 只有 2）到 2.2（Llama-3.2-3B）〔算術〕。**同樣是 3B，KV head 的配置讓 R 差 3 倍**；「小模型重算便宜」不是一般規則。
- 小模型的 MFU 風險：Llama-3.2-1B 預測一個 512-token chunk 只要 4 ms〔算術〕，m7 harness 的 Python 逐層迴圈與 kernel 啟動開銷會占很大比例 → 實測 f 會比預測大 → b 比預測小〔判讀〕。
- 70B／72B：R≈0.25–0.31，b/n @3.69 只有 0.11–0.12〔算術〕。BF16 權重約 141 GB，放得進一張 MI300X（192 GB），但 KV 空間只剩約 50 GB〔算術，未扣 activation〕。
- 寫入時才有的東西：和 §3.1 相同，沒有額外的。

### 3.7 VLM（Qwen2.5-VL／Qwen3-VL、LLaVA-OneVision、InternVL3、Llama-3.2-Vision）
- **N4：存 vision embedding 還是存 KV**。vision embedding 是 vision encoder（加 merger／projector）的輸出，是 LLM 的輸入；存它，讀取時省掉 vision encoder，但 LLM prefill 仍要重做。和 hidden state 一樣，存成 KV 之後就回不去〔判讀〕。
- **大小與成本**〔算術，config；vision encoder 只算線性部分，注意力另計〕：

| 模型 | vision encoder 參數 | 每個 LLM 視覺 token 對應幾個 patch | 每個視覺 token 的 vision encoder GFLOPs | 同 token 的 LLM GFLOPs（p→0） | 存 embedding（每 token） | 存 KV（每 token） | embedding ÷ KV | 每張圖幾個 token |
|:--|--:|--:|--:|--:|--:|--:|--:|:--|
| Qwen2.5-VL-7B | 630M（32 層 ViT，SwiGLU） | 4 | 5.13 | 13.1 | 7 KiB | 56 KiB | 12.5% | (H/28)(W/28)，4–16,384（preprocessor min／max_pixels 3,136／12,845,056） |
| Qwen3-VL-8B | 411M（27 層） | 4 | 3.61（含 3 個 DeepStack merger） | 13.9 | 32 KiB（主 embedding＋3 份 DeepStack） | 144 KiB | 22% | (H/32)(W/32)，64–16,384（shortest／longest_edge 65,536／16,777,216 像素） |
| LLaVA-OV-7B（圖） | 396M（SigLIP 26 層） | 1 | 0.83 | 13.1 | 7 KiB | 56 KiB | 12.5% | 每個 384px crop 729；anyres_max_9 |
| LLaVA-OV-7B（影片） | 396M | 3.72 | 2.98 | 13.1 | 7 KiB | 56 KiB | 12.5% | 每格 196（V01：0.5 FPS） |
| InternVL3-8B | 302M（InternViT 24 層） | 4（pixel shuffle 0.5） | 2.47 | 13.1 | 7 KiB | 56 KiB | 12.5% | 每個 448px tile 256；1–12 tiles＋縮圖 |
| Llama-3.2-11B-Vision | 786M（32＋8 層） | 1 | 1.77（含 projector 與 8 層 cross-attn 的 K/V 投影） | — | 8 KiB（投影後 4,096 維） | 32 KiB（**cross-attn** KV，8 層） | 25% | 每 tile 1,601；≤4 tiles＝≤6,404 |

  - 來源：各 config 的 `vision_config`；Qwen2.5-VL 的 ViT MLP 是 gate／up／down 三個矩陣〔程式碼：transformers `models/qwen2_5_vl/modeling_qwen2_5_vl.py` L83–91〕；Qwen3-VL 的 ViT MLP 是兩個矩陣，DeepStack 有 3 個 merger，各輸出 out_hidden_size＝4,096，加到 LLM 前幾層〔程式碼：`models/qwen3_vl/modeling_qwen3_vl.py` L75–82、L674–684、L848–851〕。preprocessor 數字取自 HF 的 `preprocessor_config.json`。
- **判讀**：
  - 重算一個圖片 token 比重算一個文字 token 多付 6%（LLaVA-OV 圖）到 39%（Qwen2.5-VL）的 vision encoder 成本 → f 變大 → κ 變小 → **b 變小**，除非存了 embedding。所以 VLM 讓「依位置」更不重要，讓「存什麼格式」更重要（H5 的重點應該放在 N4）。
  - Qwen3-VL 因為 DeepStack，embedding 要存 4 份，省得比較少（22%）。
  - **Llama-3.2-Vision 的 cross-attn KV 和位置無關**：圖片 token 不在自注意力序列裡，它的 cross-attn KV 只依賴圖片本身〔Jenga 原文 p.3：32 個自注意力層存文字 token 的 KV，8 個 cross-attention 層存圖片 token 的 KV〕。這段 KV 的重算成本不隨位置變，Cake 的位置邏輯不適用；但它的 bytes／FLOP 是 Llama 文字 token（p→0）的約 2 倍〔算術：32,768／1.77e9 對 131,072／13.96e9〕，而且「存投影後的 vision feature（8 KiB）再補 K/V 投影」只要 0.13 GFLOP／token〔算術〕——一個很乾淨的 HCache 型格式選擇。可惜 vLLM 已經不支援 Mllama（§5）。
- **現行系統**：vLLM 有 GPU 上的 encoder cache，用 mm_hash 跨請求共用 vision embedding，配置時空間不夠就淘汰最舊、沒人引用的〔程式碼：`vllm/v1/core/encoder_cache_manager.py` L35–39〕；另有 encoder cache 的傳輸介面，範例實作把 embedding 用 safetensors 存到 `shared_storage_path`（預設 `/tmp`）〔程式碼：`vllm/distributed/ec_transfer/ec_connector/example_connector.py` L56、L98–117〕。所以「把 vision embedding 存到磁碟」在 vLLM 已有範例，但不是分層快取。MPIC 是「上傳時就算好圖片 KV 並同時寫磁碟、讀取時並行算與載」〔[A_MPIC](cards/A_MPIC.md)〕。
- 補正 Lit-A 的 [A_MPIC](cards/A_MPIC.md)：它寫「硬體：讀過的頁沒看到 GPU 型號（未查證）」。原文 p.7 §5.1：「1 NVIDIA H800-80 GB GPU, 20-core Intel(R) Xeon(R) Platinum CPUs, and 100GB DRAM」〔原文 p.7〕。另外原文說單張圖的 KV 可達 1 GB、所以大多存在本機磁碟〔原文 p.2〕；以 LLaVA-1.6-vicuna-7B（MHA，512 KiB/token）× LLaVA 1.6 的 2,304 個圖片 token〔原文 p.1〕＝1.125 GiB，對得上〔算術〕。

### 3.8 影片 LLM（一次寫很多、之後多次問答）
- 這是 doc 型負載（第一階段唯一贏過的類型），寫入量很大〔10 §3B〕：
  - LLaVA-OV-7B、0.5 FPS、每格 196 token：一小時 352,800 token、18.8 GiB〔V01 卡〔計算〕，已用 ReKV 的 log 核對 GB 實為 GiB〕。
  - Qwen2.5-VL-7B（舉例，解析度與 FPS 是假設）：448×448、2 FPS，兩格合成一個 temporal patch（temporal_patch_size＝2），每 2 格 256 token → 每秒 256 token → 一小時 921,600 token × 56 KiB＝49.2 GiB〔算術〕。
- 寫入時才有的東西：vision embedding（同 §3.7）；ReKV 存的是**還沒套 RoPE** 的 K（V01 卡〔程式碼〕），也是寫入時選的格式（N4）。
- 位置邏輯的限制：ReKV 用 15K 滑動視窗編碼，每格的重算成本不隨位置增加；問答時依問題挑散落的格，不是連續前綴（V01 綜合判讀 3）。所以 Cake 的「前段重算」在影片上要重新定義〔判讀〕。

---

## 4. HCache 類「存 hidden state 再投影回 KV」：各架構的大小比較

每個注意力層、每個 token：hidden state＝hidden_size × 2 bytes；KV＝2 × KV head × head_dim × 2 bytes（MLA：576 × 2 bytes）〔算術，config〕。

| 模型 | 架構 | hidden／層 | KV／層 | hidden ÷ KV | 格式選擇有沒有空間 |
|:--|:--|--:|--:|--:|:--|
| LongAlpaca-7B、Qwen1.5-7B | MHA | 8 KiB | 16 KiB | **0.50** | 有：存 hidden 省一半（HCache 的前提） |
| Qwen1.5-14B、LongAlpaca-13B | MHA | 10 KiB | 20 KiB | **0.50** | 有 |
| Zamba2-7B（共享 MHA 區塊） | Mamba2＋MHA | 7 KiB（只存 h；另一半輸入是原始 embedding，可由 token 重查） | 28 KiB | **0.25**〔判讀：假設 embedding 部分不存〕 | 有，而且更大；但 context 上限 4,096 |
| Gemma-2-9B | GQA，head_dim 256 | 7 KiB | 8 KiB | 0.88 | 小幅 |
| Gemma-3-12B | GQA，head_dim 256 | 7.5 KiB | 8 KiB | 0.94 | 小幅 |
| Gemma-3-27B | GQA | 10.5 KiB | 8 KiB | 1.31 | 沒有 |
| Llama-3.1-8B、Qwen3-30B-A3B、Nemotron-H-8B、Qwen3-Next | GQA | — | — | 2.00 | 沒有 |
| Mistral-Nemo-12B、Seed-OSS-36B | GQA | 10 KiB | 4 KiB | 2.50 | 沒有 |
| gpt-oss-20b | GQA，head_dim 64 | 5.6 KiB | 2 KiB | 2.81 | 沒有 |
| Qwen2.5-7B | GQA 4／28 | 7 KiB | 2 KiB | 3.50 | 沒有 |
| DeepSeek-V2-Lite | MLA | 4 KiB | 1.125 KiB | 3.56 | 沒有 |
| Llama-3.1-70B、Kimi-Linear 的 MLA 層 | GQA／MLA | — | — | 4.00 | 沒有 |
| Falcon-H1-7B | GQA 2 head | 6 KiB | 1 KiB | 6.00 | 沒有 |
| DeepSeek-V3、Kimi-K2 | MLA | 14 KiB | 1.125 KiB | 12.4 | 沒有 |
| Falcon-7B | MQA | 8.9 KiB | 0.25 KiB | 35.5 | 沒有 |

- **結論**〔算術＋判讀〕：H6（寫入時選 KV 或 hidden）只在 MHA 上有位元組上的好處；Zamba2 也有但 context 太短；Gemma-2／3 只省 6–12%。GQA 上 hidden 是 KV 的 2–4 倍，MLA 是 3.6–12.4 倍。
- **SVD 也救不了 GQA**：XQuant 對 GQA 先把 X 投影到 d/g 維，記憶體「和 GQA 的 KV 一樣大」〔[C_XQuant](cards/C_XQuant.md) 原文 p.8〕；好處只在量化（有損）。
- **SSM 層的 hidden**：遞迴層沒有 per-token KV 可以取代，存 per-token hidden 是為了能在任意位置重播出狀態（Tail-Replay、SuffixReplay）；SuffixReplay 算的「每 token 每層都存」額外成本是 120–480 KiB/token，是 SGLang 每 8192 token 一份檢查點的 20–28 倍〔[A_SuffixReplay](cards/A_SuffixReplay.md)；SuffixReplay 原文 p.4 表 1〕。
- **HCache 本身的範圍**：原文只做 MHA（Llama2-7B／13B、OPT-30B），GQA 要先投影到低秩、屬範圍外〔E03 評測卡卡 4，HCache 原文 p13〕。

---

## 5. ROCm（MI300X）與 HF transformers 的支援狀況

**本機環境**：`/mlsteam/workspace/venv/tiara`，torch 2.10.0+rocm7.0（HIP 7.0.51831）、vLLM 0.19.1+rocm722（從 `/mlsteam/workspace/src/vllm` 安裝，commit `b1388b1`，2026-04-17）、transformers 5.17.0、triton 3.8.0／triton-rocm 3.6.0。**沒有安裝** mamba_ssm、causal_conv1d、flash-linear-attention（fla）、`kernels`、aiter〔`pip list`；`import aiter` 失敗〕。

### 5.1 vLLM 0.19.1（本機 build）
| 項目 | 狀況 | 出處 |
|:--|:--|:--|
| ROCm 的不支援清單 | 空的（`_ROCM_UNSUPPORTED_MODELS = []`、`_ROCM_PARTIALLY_SUPPORTED_MODELS = {}`） | 〔程式碼〕`vllm/platforms/rocm.py` L52–56 |
| Llama-3.2-Vision（Mllama） | **已移除**，最後支援 0.10.2（V0 淘汰時拿掉 encoder-decoder） | 〔程式碼〕`vllm/model_executor/models/registry.py` L645–654 |
| 表上其他架構 | 全部在 registry（Falcon、DeepseekV2／V3／V32、Mixtral、GptOss、Qwen3Moe、Gemma2／3、Mistral、Cohere2、Jamba、Zamba2、FalconH1、NemotronH、Qwen3Next、MiniMaxText01、KimiLinear、GraniteMoeHybrid、FalconMamba、Mamba2、Qwen2.5-VL、Qwen3-VL、LlavaOnevision、InternVLChatModel、SeedOss） | 〔程式碼〕`registry.py`（逐一 grep） |
| Mamba1（Jamba、Falcon-Mamba） | `selective_scan_fwd.cu` 在共同的 `VLLM_EXT_SRC`；binding 不在 `#ifndef USE_ROCM` 區塊內；本機 `_C.abi3.so` 有 406 個 selective_scan 符號 → **有編進 ROCm build**。mamba_mixer 有 ROCm 專用的 contiguous 修正（表示有人在 ROCm 上跑過） | 〔程式碼〕`CMakeLists.txt` L283；`csrc/torch_bindings.cpp` L635–655；`nm -D vllm/_C.abi3.so`；`vllm/model_executor/layers/mamba/mamba_mixer.py` L198–222 |
| Mamba2（Zamba2、Falcon-H1、Nemotron-H、Granite-H） | SSD 與 causal_conv1d 都是 Triton kernel | 〔程式碼〕`vllm/model_executor/layers/mamba/ops/ssd_*.py`、`causal_conv1d.py` |
| GDN／KDA（Qwen3-Next、Kimi-Linear） | vendored 的 FLA Triton kernel | 〔程式碼〕`vllm/model_executor/layers/fla/ops/` |
| MLA | ROCm 的 MLA 後端：ROCM_AITER_MLA、TRITON_MLA、ROCM_AITER_TRITON_MLA；本機沒有 aiter → 只剩 TRITON_MLA | 〔程式碼〕`vllm/platforms/rocm.py` L317–326 |
| 滑動視窗 | ROCm 的 rocm_attn、triton_attn、rocm_aiter_fa 都處理 `sliding_window` | 〔程式碼〕`vllm/v1/attention/backends/rocm_attn.py` 等 |
| attention sinks（gpt-oss） | ROCm 自訂 kernel 不支援 sinks，會退回 Triton | 〔程式碼〕`rocm_attn.py` L198–199 |
| **接 KV connector（OffloadingConnector）時** | 只要設 `--kv-transfer-config` 且沒有明確要求，**HMA 預設關閉**；滑動視窗層轉成 FullAttentionSpec（不丟視窗外的 KV）；不能統一成同一種 spec 的（注意力＋Mamba）會丟 `ValueError("Hybrid KV cache manager is disabled but failed to convert the KV cache specs to one unified type.")`。OffloadingConnector 沒有實作 `SupportsHMA`（只有 SimpleCPUOffloadConnector、NixlConnector 有）。scheduler 端只取第一個 KV cache group | 〔程式碼〕`vllm/config/vllm.py` L1227–1244；`vllm/v1/core/kv_cache_utils.py` L1160–1216；`kv_connector/v1/base.py` L84–120；`simple_cpu_offload_connector.py` L45；`offloading/scheduler.py` L154–155 |
| 混合模型的 prefix caching | 實驗性；`mamba_cache_mode` 三種：none／all／align（§3.5 表） | 〔程式碼〕`vllm/config/cache.py` L25、L97–118 |
| SSM 狀態 dtype | 預設跟模型 dtype；MiniMax 的線性注意力不支援 FP32 狀態 | 〔程式碼〕`mamba_utils.py` L21–30、L55–67 |

**判讀**：在 vLLM 路徑上，SSM／線性混合模型的 kernel 在 ROCm 上「有東西可以跑」，但**本研究要用的 OffloadingConnector 路徑對混合模型照程式碼會直接報錯**，滑動視窗模型則會存全部 KV。全部 **NOT_MEASURED**，第一次要用時先跑一個最小的啟動測試。

### 5.2 HF transformers 5.17（m7 harness 走這條）
- 所有家族都有實作（jamba、zamba2、falcon_h1、nemotron_h、qwen3_next、minimax、kimi_linear、granitemoehybrid、deepseek_v2／v3、gpt_oss、gemma2／3、cohere2、mllama、qwen2_5_vl、qwen3_vl、llava_onevision、internvl、falcon_mamba、mamba2、mixtral、seed_oss）〔本機 `transformers/models/` 目錄〕。
- SSM／線性層的 kernel 優先順序：HF hub kernels（有要求才用）→ 原始套件（mamba_ssm、causal_conv1d、fla）→ **純 torch 路徑**〔程式碼：`transformers/integrations/hub_kernels.py` L829–859〕。本機三個原始套件都沒裝 → 走純 torch，正確但慢〔判讀，未量〕。
- 上游套件：state-spaces/mamba 與 Dao-AILab/causal-conv1d 的 README 都有「Additional Prerequisites for AMD cards」，ROCm 6.0 要打 patch、6.1 以後不用〔README，2026-10-09 抓取〕。也就是**官方支援在 ROCm 上編譯**；在 MI300X＋ROCm 7.0 上能不能編過、跑多快：NOT_MEASURED。
- **m7 harness 的相容性**〔程式碼：`code/m7_model.py` L35–107〕：自管 KV 的前向是 Llama 式（q/k/v/o_proj、input_layernorm、post_attention_layernorm、可選 q_norm），所以 Llama、Qwen2／Qwen1.5（有 bias，`q_proj` 自帶）、Qwen3、Mistral 系列可直接換 `M7_MODEL_GLOB`；Gemma（四個 norm、query_pre_attn_scalar、滑動視窗遮罩）、MLA、SSM 混合、VLM 都要改程式〔判讀〕。

---

## 6. 推薦下一批在 MI300X 上量 f(i) 的 3 個模型〔判讀〕

前提：LongAlpaca-7B 與 Qwen3-30B-A3B 已在第 1 輪 H2（[11](../phase1_20261008/11_round1_plan.md)）。本文件對它們的預測（可拿來對照實測）：LongAlpaca-7B b/n＝0.61／0.17（3.69／35.4 GiB/s）；Qwen3-30B-A3B＝0.33／0.08，8K 時 0.44／0.06。

依「最可能讓 b 變大」排序：

| 順位 | 模型 | 預測 b/n @3.69／@35.4（32K） | 為什麼 | 成本與風險 |
|:--|:--|:--|:--|:--|
| 1 | **Qwen1.5-7B-Chat**（MHA，原生 32K） | 0.61／0.17（R32＝4.19） | R 最大；和 LongAlpaca-7B 同尺寸，但 **原生 32K**（max_position 32,768、rope θ＝1e6，不用 LongAlpaca 的線性 RoPE ×8）。可以檢驗「MHA → b 變大」是不是 LongAlpaca 特有。要看尺寸效應可換 Qwen1.5-14B-Chat（0.58／0.16） | 未授權限制（HF 不需登入）；Qwen2 架構，m7 harness 只換 `M7_MODEL_GLOB`；約 15 GB 權重。風險：MHA 是 2024 年以前的架構，審稿人可能說不代表現在的模型 |
| 2 | **Gemma-3-12B** | 全部層：0.50／0.09；只存全域層：0.14／0.02 | 唯一「存什麼」會讓 b 差 3.5 倍的主流新模型。vLLM 接 OffloadingConnector 時實際上會存全部層（§5.1），所以「全部層」是真實部署的情形；「只存全域層」是寫入時（或搬移時）的格式決定。可同時回答 H2 與 H9 | 要改 harness（Gemma 的 norm、scaling、滑動視窗遮罩、head_dim 256）；HF 原檔需授權，可用 `unsloth/gemma-3-12b-it`；權重約 24 GB |
| 3 | **Llama-3.2-3B** | 0.44／0.09（R0＝2.17、R32＝1.75） | 小模型裡 R 最大的；和 Llama-3.1-8B 同一家族，**harness 不用改**，最便宜的一筆。實測可以直接回答「小模型的 MFU 會不會把理論上的 κ 優勢吃掉」 | `unsloth/Llama-3.2-3B-Instruct`；約 6.4 GB；預測是上限（§1.3） |

- **為什麼不選**：Gemma-2-9B（R 高但 context 只有 8K）、Zamba2-7B（R 高但 context 只有 4K，且 SSM）、Qwen3-8B（R≈1.1，和 Llama 差不多）、70B（b 只會更小）、MLA 與 SSM 混合（b≈0，只能當反方向對照）。
- **如果目的是 H7 的可行性，不是 b**：另外量一個 SSM 混合模型「能不能在 ROCm 上跑、純 torch 路徑多慢」。建議 Nemotron-H-8B（Mamba2，HF 不需授權、8B 好放）；Falcon-H1-7B 是第二選擇（每層都有注意力，KV 較大）。這兩個的 KV 很小，**不會讓 b 變大**。

---

## 7. 對 §4 各假設的影響

| H | 本文件的發現 | 狀態〔判讀〕 |
|:--|:--|:--|
| **H2**（重算便宜的模型讓 b 變大） | 只有 MHA 明顯有效（R≈4）；MoE 在 32K 優勢消失（R32 0.92）；小模型 R 0.6–2.2 看 KV head 配置；沒有模型在 35.4 GiB/s 達到 30%；Llama 自己在 3.69 GiB/s 已經 0.31 | 活著，但範圍窄：只剩 MHA（和 Gemma 類「存全部層」）。判準要修（§0 第 2 點） |
| **H5**（VLM／影片） | vision encoder 讓 f 變大 → b 變小；embedding 只有 KV 的 12.5–22%；Llama-3.2-Vision 的 cross-attn KV 和位置無關；vLLM 有 GPU encoder cache 和存磁碟的範例 connector；MPIC 已做「上傳時寫磁碟、讀取時並行算與載」 | 重點應從「位置」改成「格式（N4）」；對照組至少要有 MPIC 型與 vLLM encoder cache |
| **H6**（寫入時選格式） | 只在 MHA（0.5×）、Zamba2（0.25–0.5×）、Gemma-2／3（0.88–0.94×）有位元組上的好處；GQA 2–4×、MLA 3.6–12.4×；XQuant 的 SVD 也只到打平；HybridServe 已在 MHA 上做「依比例選格式」 | 只能在 MHA 上做；Lit-A 已把 H6 降到最低優先，本文件的算術支持這個判斷 |
| **H7**（SSM 檢查點） | Cake 還原同一段 prefix 不需要中間狀態（§3.5）；位置選擇已有 DP（Sparse Prefix Caching）；有損替代（Tail-Replay、SuffixReplay）；SGLang 狀態不分頁；vLLM OffloadingConnector 對混合模型會報錯；ROCm 上 kernel 有、但沒量過 | 只剩「檢查點的分層放置＋分叉型負載＋無損」；要先做可行性（§6 的補充建議） |
| **H9**（滑動視窗層的舊 KV 不存） | 可省 50–84%（Gemma-3 5:1 省 83%），遠超 20% 的停損線；vLLM 接 connector 時現行實作會存全部；但延後版（搬移時才丟）也做得到 → 只有 N2（寫入量） | 活著，但屬於 N2；要和「搬移時才丟」比 |
| **H14**（寫入時選精度） | 不在 Lit-C 主範圍。附帶：XQuant 顯示同位元下「存量化 X」比量化 KV 準，H14 若做要把它列為對照（有損，要老師同意） | 未變 |

---

## 8. 卡片與文獻

### 8.1 本次新寫的卡（Lit-C，5 張）
| 卡 | 論文 | 讀到 | 一句話 |
|:--|:--|:--|:--|
| [C_TailReplay](cards/C_TailReplay.md) | Tail-Replay（arXiv 2608.30310） | p.1–4／7 | 存 FA KV＋FA 輸出 hidden，不存線性層檢查點，命中時重播最後 5–10%（有損） |
| [C_KVBuffer](cards/C_KVBuffer.md) | KVBuffer（arXiv 2605.19049） | p.1–2、p.7 | GDN 層狀態約 2 MB；短 context 只存 KV 不建狀態（GPU 內的格式選擇） |
| [C_GLMHybridRecovery](cards/C_GLMHybridRecovery.md) | GLM-5.3-Flash＋vLLM＋LMCache 的混合狀態還原（arXiv 2609.15030） | p.1–3／8 | 固定間隔 1,792 存到 CPU；狀態位置錯一格就靜默改變輸出 |
| [C_XQuant](cards/C_XQuant.md) | XQuant（arXiv 2508.10395） | p.1–2、p.7–8 | 存 X 代替 KV；GQA 直接存 X 多 2 倍，SVD 後才打平 |
| [C_LinearKV](cards/C_LinearKV.md) | LinearKV（arXiv 2608.11231） | p.1–5、p.7 | 混合模型的位置無關快取：每個 chunk 存 FA KV＋結尾狀態（有損） |

### 8.2 Lit-A 已做卡、本文件引用的（不重做）
[A_SparsePrefixCaching](cards/A_SparsePrefixCaching.md)、[A_SuffixReplay](cards/A_SuffixReplay.md)、[A_Jenga](cards/A_Jenga.md)、[A_MPIC](cards/A_MPIC.md)、[A_HybridServe](cards/A_HybridServe.md)、[A_Marconi_補充](cards/A_Marconi_補充.md)、[A_HCache_補充](cards/A_HCache_補充.md)。本文件另外讀了 Jenga、MPIC、Sparse Prefix Caching、SuffixReplay 的原文，補充：
- Jenga：Mamba 層「每 512 token 存一份狀態」〔原文 p.8〕；若用最大的 embedding 當 page，Jamba 52B 的 mamba 狀態會讓每個注意力 page 要 1,344 token〔原文 p.7〕；VLM 的 cross-attn 與 vision embedding 管理〔原文 p.3〕；硬體 H100 80GB 與 L4〔原文 p.10〕。
- MPIC：硬體 H800-80GB〔原文 p.7〕（補正 A_MPIC 的「未查證」）。
- SuffixReplay：SGLang 的狀態「do not page」〔原文 p.12〕；容量實驗 host 預算 128 GB〔原文 p.9〕；單張 H800 80GB、SGLang 原生檢查點規則〔原文 p.8〕。
- Sparse Prefix Caching：檢查點在另一次不計時的 run 擷取，因為 FLA 的 Python API 不提供非最後位置的狀態〔原文 p.8〕；卸載成本（CPU→GPU 載入檢查點）以 Fig. 4 的截距呈現〔原文 p.10–11〕。

### 8.3 看到但沒做卡的（皆未讀全文）
從 Semantic Scholar 的 Marconi 與 HCache 引用清單（2026-10-09 查詢）看到、和本題相關的：HYPIC（arXiv 2607.01299，混合模型的位置無關快取）、DASC（2608.30386，混合線性注意力的狀態壓縮）、Bole（2608.01651，混合模型的樹狀 speculation）、Irminsul（2605.05696，MLA 原生的位置無關快取）、CONDUIT（2609.05821，VLM 的選擇性重算，只讀了摘要 p.1：10% refresh 預算下保留 97.0–99.5%，有損）、VLCache（2512.12977）、Kamera（2606.23581）、CoMem（2607.28263，跨查詢重用中間 residual）。影片的 InfiniPot-V、StreamMem、LiveVLM、StreamingVLM 仍是 V01 列的〔未查證〕。

---

## 9. 未查證與限制

- **所有 b/n 都是算術預測**，假設各模型的 MFU 等於 Llama-3.1-8B 在 m7 harness 上的值；沒有任何新模型的實測 f(i)。
- HF 需授權的模型改用鏡像 config（unsloth、mlx-community、FlagRelease、NousResearch、Zyphra 的 Instruct 版）；鏡像與原檔的結構欄位是否完全相同〔未查證〕。gpt-oss 的 HF API 參數數是 MXFP4 打包後的值，公式的總數碰巧接近，不代表 dtype。
- vLLM 與 ROCm 的所有結論來自讀本機原始碼，**沒有實際啟動**任何非 Llama 模型。「OffloadingConnector＋混合模型會報錯」是程式碼判讀。
- SSM scan、vision encoder 注意力的 FLOPs 是粗估；不影響量級。
- MLA 的注意力 FLOPs 用解壓形式；vLLM 在 prefill 實際用哪種形式〔未查證〕。
- Zamba2 的「只存 h、不存 embedding」是推論；Zamba2 共享區塊的輸入確實是 concat(h, 原始 embedding)（config 的 attention_hidden_size＝2×hidden_size），但 vLLM／HF 的實作細節沒逐行核對。
- Jenga 的 SOSP'25：本文件讀的是 arXiv v1，沒有核對 ACM 定稿（Lit-A 卡同樣標註）。
- Tail-Replay、SuffixReplay、LinearKV、KVBuffer、Sparse Prefix Caching、GLM 案例都是 arXiv 預印本，venue 未查證。

---

## 附錄 A：重現本文件算術的方法

計算腳本放在本 session 的 scratchpad（不在 repo；依規則本次只能寫這份文件與卡片）。要重現，照下面三步寫一支腳本即可：

1. **讀 config**：本機 8 個模型讀 `/mlsteam/data/tiara/hf-cache/hub/models--*/snapshots/*/config.json`；其他用 `https://huggingface.co/<repo>/resolve/main/config.json`（表中的 repo）。只讀 config，不下載權重。
2. **每個模型算三個量**（§1.1、§1.2）：
   ```text
   kv_bytes  = Σ_注意力層 2·n_kv·d·2           （MLA：每層 (kv_lora_rank+qk_rope_head_dim)·2）
   lin_flops = 2·N_active_nonembed（不含 LM head；MoE 只算 top-k＋shared＋router）
   attn(p)   = Σ_注意力層 2·n_q·(d_qk+d_v)·min(p, W)
   chunk i 的 FLOPs：lin = 512·lin_flops；att = Σ_{q=1..512} attn(512·i+q)
   ```
   用 HF API 的 `https://huggingface.co/api/models/<repo>?expand[]=safetensors` 取總參數數，和公式算出的總數核對。
3. **校準與 b**：從 `results/m7_write_policy_mi300x/calib_c1.csv`（`item=f_chunk`，每個 chunk_idx 取 3 次中位數）對 Llama-3.1-8B 做兩參數最小平方 f(i)＝α·lin_i＋β·att_i；對每個模型 f_pred(i)＝α·lin_i＋β·att_i、ℓ＝512·kv_bytes／(頻寬·2^30)·1000 ms；用 `code/m7_restore_harness.py` 的 `write_boundary(n=64, f_pred, ℓ)` 算 b。檢查點：Llama 用實測 f 與 f_pred 的 b/n 應相同（0.797／0.578／0.312／0.141／0.047）。
