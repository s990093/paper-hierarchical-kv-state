# D0 — 模型自帶的稀疏注意力索引（InfLLM-v2）能不能當還原時的挑塊器（05 §7；品質，不量時間）

> 05 §7 的量法 1（聯集佔比）與量法 2（答案塊 recall），照這份寫死的定義做。L8d §6 的第 5、6 步（只還原前 1/4、1/2 的品質；用早層 hidden 預先決定）**不在 D0 裡**：NOT_MEASURED，GD0 過了再另外登記。
> 換了模型：數字不能直接和 Llama／Qwen 的 A0 比，只能比「挑法離上界多遠」（05 §7 的注意）。

## 0. 前提（P0f 冒煙，job 22307）

冒煙 PASS 才跑：每層都是 `MiniCPMInfLLMv2Attention`、32K 的 forward 真的呼叫 `infllmv2_attn_stage1` 與帶 `topk_idx` 的 `infllmv2_attn_varlen_func`、flash_attn 轉接層和 SDPA 對得上（RUNLOG S3-D0-0）。冒煙記下的 `topk_idx` 形狀決定第一維是 KV head 還是 q head（照實寫進結果）；下面寫「head」的地方指那一維。

## 1. 模型與環境（固定）

- `openbmb/MiniCPM4.1-8B` revision `3a8dfed9c79a45e07dbff95bcd49d792343fa1a3`，BF16，1 張 H200；transformers 4.56.1（job 本地 pylibs）；`infllm_v2`＝OpenBMB/infllmv2_cuda_impl `93cf2ec2`。
- `sparse_config`＝README 那一組（kernel 32、stride 16、init_blocks 1、block 64、window 2048、topk 64、use_nope false、dense_len 8192）→ 每個 query token 選 96 個 64-token 塊（topk 64＋local 32；含 init 塊）。
- **稀疏模式**＝上面那組；**dense 模式**＝同一份權重、同一個類別，只把 `dense_len` 設成大於序列長度（每層都走 `_flash_attention_forward_dense`）。

## 2. 題目

- `m16_xstack.plan_interleaved_cpu(["T2","T3","T4"], 64, 76000)` → **192 題**（兩輪題；種子 76000–76063，和 A0 的 tf／gen1／gen2、`a0.USED_SEEDS` 都不重疊，10/25 查過）。T5（全域，沒有位置帶）不做。
- 題目參數：`m10_tasks` 的預設（MiniCPM 沒有調過的參數檔；D0 量的是索引的 recall，不是答對率）。
- 序列＝第 2 輪的完整序列（`context_ids`〔64×512＝32,768 token〕＋第 1 輪問題＋第 2 輪的 history＋第 2 輪問題，`m10_tasks.full_prompt_ids`）。**Q2**＝第 2 輪問題的 token（`turns[1]["query_ids"]`，序列最後那一段）。
- 答案所在的 512-token 塊：`m16_c1a_predict.verify_answer_blocks(tok, it, 1)`（三條路徑驗算；任何一題不過 → 停，規則 2）。候選塊＝1..63（塊 0 是 sink，永遠保留，和 A0／C1a 相同）。

## 3. 量法（每題）

一次稀疏模式的完整 prefill、一次 dense 模式的完整 prefill（都是整條序列一次送進去，`logits_to_keep=1`）。

**原生索引（稀疏模式）**：包住 `infllm_v2.infllmv2_attn_varlen_func`，每層一次（呼叫順序＝層的順序，每次 forward 必須剛好 32 次帶 `topk_idx` 的呼叫，否則停），取 `topk_idx` 裡 Q2 那幾列：每層 ℓ、每個 head h、每個 Q2 token t 選到的 64-token 塊 c（−1＝無效，丟掉）→ 512-token 塊 c // 8。

- **聯集佔比**（量法 1）：U＝所有 (ℓ, h, t) 選到的 512-token 塊 ∩ {1..63}；`union_frac`＝|U| / 63。
  只報告：每層各自的聯集佔比的平均（`union_frac_per_layer`）；64-token 粒度的聯集佔比（`union64_frac`＝選到的 64-token 塊 ∩ 塊 1..63 的範圍 ÷ 504）；local window 蓋到的 512-token 塊數（`n_local_blocks`，第一個 Q2 token 往前 2048 個 token 碰到的塊）。
- **NATIVE-HITS**（排名）：s[b]＝所有 (ℓ, h, t, 選的那一格) 落在 512-token 塊 b 的次數。只報告：**NATIVE-HITS-L**（同上，略過第 0、1 層）。

**對照（dense 模式，同一份權重）**：在 flash_attn 轉接層拿到每層 post-RoPE 的 q、k（只取 Q2 那幾列的 q），fp32 計算：

- **FULLK**（完整 K 上界，05 §7）：每層 ℓ ≥ 2、每個 q head、每個 Q2 token，對它看得到的所有 key 做 softmax(q·kᵀ/√D)，每個 512-token 塊的機率總和 → 對 Q2 token 平均 → 對 (ℓ, q head) 平均。和 C1a 的 FULLK_MEAN 同一個定義（GQA：q head i 對 KV head i // 16）。
- **QUEST-512**：每層、每個 KV head、每個 512-token 塊的 post-RoPE K 逐維 min／max；U[b]＝Σ_d max(q_d·min_d[b], q_d·max_d[b]) / √D，在 64 個塊上 softmax → 對 Q2 token 平均 → 略過前 2 層 → 對 (ℓ, q head) 平均。和 C1a 的 QUEST_MEAN 同一個定義。
- **RND0–2**：三個隨機排列（種子＝題目種子＋r）。

排名：分數由大到小；同分用固定的隨機排列打破（種子＝題目種子），不看任何分數決定。

**指標**：recall@k＝答案塊落在前 k 名的比例。**k＝8（1/8）是判定用**；k＝16（1/4）一起報、不判定；k＝4、32 只報告。另報 NATIVE-HITS 與 FULLK 前 k 名的 Jaccard。

**健全性檢查（任何一個不過 → 停）**：答案塊驗算；稀疏模式每次 forward 剛好 32 次 `infllmv2_attn_stage1`、32 次帶 `topk_idx` 的 varlen；dense 模式 0 次帶 `topk_idx` 的呼叫；`topk_idx` 的值在 [−1, 64-token 塊數)；每個 (層, q head, Q2 token) 的 FULLK 機率在所有 key 上加總 ＝1（誤差 ≤ 1e−3）。

## 4. GD0（05 §7，寫死）

統計：以題目為單位的 bootstrap（10,000 次）平均與 95% CI；判定看點估計，CI 跨過門檻的另外標 **BORDERLINE**（照樣判定、照實回報）。

1. **聯集**：`union_frac` 的平均 > 0.80 → **UNION-NO-SAVINGS**（「只讀被選的」省不了位元組，D 只剩 A 的挑法角色）；否則 **UNION-SAVES**（報佔比）。
2. **recall**：gap＝每題 [recall@8(FULLK) − recall@8(NATIVE-HITS)] 的平均（配對）。gap > 10 個百分點 → **STOP**；否則 **SELECTOR-OK**。
3. 總判定：2 是 STOP → **GD0-STOP**；否則 1 是 UNION-NO-SAVINGS → **GD0-SELECTOR-ONLY**；否則 **GD0-CONTINUE**。

## 5. 產出

- `results/s3_h200/d0/raw/d0_<run_id>.csv`：每題 × 每個排名一列（run_id、ts、item_id、task、seed、method、recall@{4,8,16,32}、jaccard8／16 對 FULLK、ans_blocks、top8；NATIVE 那幾列另有 union_frac、union_frac_per_layer、union64_frac、n_local_blocks、n_stage1_calls、n_sparse_calls、topk_shape、own_gpu_intruders、host_level）。
- `results/s3_h200/d0/d0_report.json`：§4 的統計與判定。
- 成本〔估計〕：192 題 × 兩次 32K prefill，0.5–1 GPU 小時（05 估 4–8，包含編譯與冒煙）。

## 登記

- 登記時間（`date -u`）：2026-10-25T14:32:50Z；commit 前沒有任何 D0 的數字（D0 冒煙 22307 還沒跑；看過的只有 MiniCPM 的 modeling 程式、config、README）。

## 偏離（登記之後）

（無）

## 結案（2026-10-28，使用者決定；沒有任何 D0 的數字）

- **不做**。原因：(1) infllm_v2 編不起來，錯在上游 cutlass fork 的標頭檔（RUNLOG S3-D0-FAIL-1）；(2) D0 是方向 A 的挑塊器，A0／A0-C 判 STOP（RUNLOG S3-A0C）之後不影響 06 §4 的決策表。使用者 10/28「照建議」→ D0 量測 job 22536 取消，GD0 記 **NOT_MEASURED**。
