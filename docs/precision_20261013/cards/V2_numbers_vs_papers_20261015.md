# V2｜今天在 MI300X 上量到的數字，和論文一不一樣？（2026-10-15）

> 回到 [README](../README.md)。這一頁回答使用者的問題：「這些數據跟其他 paper、跟所有參考的 paper 一樣嗎？」
> **標記**：〔原文 p.X〕＝我這次自己讀了原文的那一頁；〔二手〕＝只讀到別人轉述，或 arXiv HTML 經 WebFetch 摘要；〔文件〕＝GitHub issue、部落格這類沒有同儕審查的來源；〔算術〕〔判讀〕照 03 的用法。查不到就寫「查不到」，不估、不猜。
> **判定**：✅ 一致／⚠️ 不同但解釋得通（寫原因）／❌ 矛盾（要停下來查）／❓ 沒有可比的數字。
> 我們的數字全部是 MI300X 彩排（`results/m10_pilot_mi300x/`，RUNLOG_MI300X 的 Milestone 10-pilot 兩節），不進任何判定。

**一句話**：能對照的大多一致。最漂亮的一致是 Qwen 的 K 偏移：最大值、出現在哪幾層、哪幾個 channel，都和第三方量測幾乎一樣。重建誤差也和 KVTuner 對 Llama 的表差不到 0.1 個百分點。唯一的矛盾是 CWE（常見詞），只影響 T5：我們的 48% 不是出題造成的，上游原版題目在我們的環境反而只有 1–6%，低於 RULER 論文換算出的下限 8.2%（§5b）。

**判定統計（26 列）**：✅ 16、⚠️ 4、❓ 5、❌ 1（第 23 列，10/15 由 ⚠️ 改判，只影響 T5）。

---

## 1. Qwen 的 K：最大值與固定偏移

| # | 我們量到的 | 論文／外部的數字 | 判定 |
|:--|:--|:--|:--|
| 1 | Qwen2.5-7B-Instruct-1M（BF16 權重），RoPE 後、5 個 32K prompt：max\|K\|＝422；第 27 層 422、第 0 層 172，其餘 26 層都 ＜80；max\|V\|＝82〔`kv_stats/summary_kv_qwen.json`、`kv_head_qwen.csv`，run `20261015-030452-m10-kvstats2-qwen`〕 | 第三方 fork issue #28：Qwen2.5-7B-Instruct-**GPTQ-Int4**、gfx1100、128／2K／8K token：「max\|K\| 419-423, max\|V\| 72.5」；逐層「27 (423), 0 (173), everything else under 80」〔文件 https://github.com/ThePie88/vLLM-ROCm-Windows/issues/28 ，10/15 讀〕。原文沒明說 RoPE 前或後，但它討論「channels that barely rotate over 8k positions」，看起來是 RoPE 後〔判讀〕 | ✅ 幾乎一樣（422 對 423、172 對 173）。差別：不同的微調版本（1M 對一般 Instruct）、權重 BF16 對 GPTQ（GPTQ 只量化權重矩陣，bias 照原值〔判讀〕）、長度 32K 對 ≤8K。**來源不是論文** |
| 2 | 每（KV head, channel）取全部層的 max\|K\|：512 組裡 22 組 ＞100、中位數 8.9、p95 90.5；＞100 的集中在 head 0、1 的 channel 57–63、122–127，另有 head 2 的 channel 125、127（109、108）〔本頁計算，raw `kv_chan_qwen.csv`，指令見附錄〕 | 同一個 issue：「median 6.9, p95 93, 22 of 512 pairs above 100」，全部在「channels 58-63 and 123-127 of kv heads 0 and 1」〔文件 issue #28〕 | ✅ 22 組對 22 組。我們多出 head 2 的兩組、channel 57 與 122，量級差不多 |
| 3 | 這些 channel 幾乎是常數：最大的一個（第 27 層 head 1 channel 127）平均 415.1、標準差 1.69；14,336 個 channel 中 645 個「\|平均\|／標準差 ＞3」，219 個 ＞10 | issue #28：「a large, almost constant vector across positions (the k_proj bias...) plus a small residual」〔文件〕；沒有給標準差 | ✅（定性一致；他們沒有可比的數值） |
| 4 | 原因：固定偏移來自 k_proj bias？我們的 channel 都在每個 RoPE 半段的最後幾維（最低頻）。Qwen 的 rope_theta＝10⁷，最低頻那一維在 32K token 內只轉 0.004 弧度〔算術：10⁷^(−126/128)×32768〕，所以 RoPE 前的常數在 RoPE 後幾乎不變。沒有 QKV bias 的 Llama 最大 \|平均\| 只有 24.9〔`summary_kv_llama.json`〕 | Qwen2.5-1M 技術報告 §2 Architecture：模型用「QKV bias ([Su, 2023]) in the attention mechanism」〔二手：arXiv 2501.15383 HTML 經 WebFetch〕。Massive Values（ICML 2025，arXiv 2502.01563）：Q、K 的集中大值出現在「low-frequency regions」，前半和後半各一群，**歸因於 RoPE**；分析過的模型含 Qwen2.5-7B、Llama3-8B，沒提 bias〔二手：arXiv HTML 經 WebFetch〕。「是 bias 造成的」只是 issue 作者的推測〔文件〕 | ⚠️ **位置**和論文一致（低頻 channel、兩個半段各一群）。**原因**有兩種說法：bias 造成「常數」、RoPE 低頻造成「不被轉掉」，兩者可以同時成立〔判讀〕。目前沒有任何論文直接證明是 bias |

## 2. Llama-3.1-8B 的 K、V

| # | 我們量到的 | 論文的數字 | 判定 |
|:--|:--|:--|:--|
| 5 | max\|K\|＝34.5（第 29 層）、max\|V\|＝7.625；每（head, channel）的 max\|K\| 中位數 10.7、p95 19.9、沒有 ＞100 的〔run `20261015-030742-m10-kvstats2-llama`；附錄〕 | 查不到 Llama-3.1-8B 的 max\|K\| 數值（KIVI、KVQuant、KVTuner 都只給圖或相對誤差） | ❓ |
| 6 | 有 channel 型的離群值：32,768 個 channel 中 1,059 個「\|平均\|／標準差 ＞3」、167 個 ＞10；最大 \|平均\| 24.9 | KIVI（ICML 2024）：「For key cache, there are a few fixed channels whose magnitudes are very large」，Figure 2 是 Llama-2-13B、Falcon-7B〔原文 p.2、p.4〕。KVQuant（NeurIPS 2024）：LLaMA-7B 的 K 在 RoPE 前有固定的離群 channel，RoPE 後「less consistent」〔原文 p.4 Figure 2〕。KVTuner（ICML 2025）：「key cache has strong channel-wise outliers」〔原文 p.4〕 | ✅ 定性一致；模型版本不同（Llama-2／LLaMA-1 對 Llama-3.1）。我們量的是 RoPE 後，Llama 的離群值比 Qwen 溫和得多 |

## 3. 重建誤差

| # | 我們量到的（相對 L2，K／V；5 個 32K prompt、每 block 算再平均）〔`summary_kv_*.json`〕 | 論文的數字 | 判定 |
|:--|:--|:--|:--|
| 7 | FP8 e4m3（每 head／每 token scale）：Qwen K 2.45–2.52%、Llama 2.36–2.63%，V 兩個模型 2.53–2.66% | e4m3 只有 3 位尾數，捨入誤差是半格，正常數的相對誤差均方根約 2.65%〔算術；本頁另用高斯與對數均勻分布模擬，都是 2.65%，指令見附錄〕。8/31 的高斯分析 2.64%〔`code/quant_error.py`，01 §3〕 | ✅ 和理論值一致。Qwen 每 channel scale 是 1.69%，偏低是因為相對 L2 被大值主導〔判讀，沒細查〕，不影響結論 |
| 8 | INT8 每 token 每 head（對稱）：Llama K 1.18%、V 0.79%；Qwen K 1.16%、V 0.74% | KVTuner Table 9：Llama-3.1-8B-Instruct、20 個 GSM8K prompt，INT8 per-token-asym：e_k 0.012280、e_v 0.007865〔原文 p.17〕 | ✅ Llama 幾乎一樣（1.18 對 1.23、0.79 對 0.79）。差別：對稱或非對稱、誤差定義（他們是 mean(\|K−K̂\|／\|K\|)〔原文 p.4〕）、prompt 長度 |
| 9 | INT4 每 token（對稱）：Llama K 21.1%、V 14.1%；Qwen K 11.4%、V 13.0% | KVTuner Table 9，Llama KV4 per-token-asym：e_k 0.196476、e_v 0.126894〔原文 p.17〕 | ✅ Llama 接近（對稱的格子較少，略大是預期的）。Qwen 沒有論文數字，見第 11 列 |
| 10 | KIVI 式 INT4（K 每 channel、V 每 token，各 32 個一組，非對稱）：Llama K 3.67%、V 8.25%；Qwen K 0.81%、V 8.06% | KVTuner Table 9，Llama KV4 per-channel-asym：e_k 0.080991、e_v 0.125457〔原文 p.17〕。這張表沒寫分組大小；它的 KIVI 實驗另寫「residual length … 32 and the group size to 32」〔原文 p.17〕 | ⚠️ 方向一致：K 每 channel 比每 token 準很多（我們 3.7 對 21.1，他們 8.1 對 19.6）。絕對值不同，因為我們的 V 也是 32 個一組、K 沿 token 32 個一組，他們 Table 9 的分組沒寫 |
| 11 | Qwen 的 FP8／INT8／INT4 重建誤差（第 7–10 列的 Qwen 數字） | 查不到任何論文報 Qwen2.5 的 KV 重建誤差（KVTuner Table 9 只有 Llama） | ❓ |
| 12 | 「扣掉平均後」的 K 誤差：Qwen FP8（每 head）18.0%、先減偏移（K0）2.7%；Llama 4.2%、2.7% | issue #28 第 27 層 attention 輸出誤差（rms-rel）：FP8 per-tensor 1.0 是 0.237；「mean removed + per-(head,channel) residual scale」是 0.017；INT8 非對稱每 (head, channel) 0.003；KIVI 式 int4 0.071〔文件〕 | ✅ 方向一致：減掉平均後誤差降一個量級（我們約 7 倍）。指標不同：他們是第 27 層的 attention 輸出、GPTQ 權重；我們是全部層的 K 重建 |
| 13 | 每 channel scale 的 FP8：Qwen 扣平均後 12.4%（比每 head 的 18.0% 好一些） | issue #28：每 (head, channel) scale 在第 27 層反而變差（rms-rel 0.426 對 per-tensor 0.237）〔文件〕 | ⚠️ 方向不同，但指標不同（attention 輸出 vs K 重建；單一層 vs 全部層），而且兩邊都同意「只改 scale 修不好，要減平均」 |

## 4. 九月的撈針舊實測 vs 論文

| # | 我們的（9 月；01 §3） | 論文的數字 | 判定 |
|:--|:--|:--|:--|
| 14 | Qwen INT4 撈針 0/20（3090 AWQ 權重、32K：run `20260831-181930-m5-needle`；MI300X BF16：run `20260916-065920-m5-b-qwen7b-needle`） | KVTuner Table 5：Qwen2.5-7B-Instruct per-token-asym GSM8K 平均：BF16 0.7755、KV4 0.0078〔原文 p.8〕；原文「sensitive even to int4 key cache quantization」〔原文 p.5〕 | ✅ 一樣是「INT4 就壞掉」（任務不同：撈針 vs GSM8K） |
| 15 | Qwen INT8 19/20、18/20 | KVTuner Table 5：KV8 0.7712〔原文 p.8〕（幾乎沒掉） | ✅ |
| 16 | Llama 五種精度（含 INT4）撈針都 20/20（MI300X，64K／129K；D6 §4.5） | KVTuner Table 5：Llama-3.1-8B-Instruct BF16 0.8038、KV4 per-token-asym 0.7765、KIVI-4 0.8011〔原文 p.8〕 | ✅ Llama 對 4-bit 不敏感。我們的撈針太簡單（碰天花板，05 §3），只能說方向一樣 |
| 17 | Qwen INT4 0/20 | DiffKV（SOSP 2025）：Qwen2.5-7B 在 K4V8 時準確率「drops to nearly zero」〔二手：cards/P04 §4，arXiv 2412.03131v3 §7.2 經 WebFetch 摘要〕 | ✅ |
| 18 | Qwen FP8 撈針 1/20、2/20 | 查不到任何論文量過 Qwen2.5 的 FP8 KV。issue #28：一般 FP8「garbles at token 0, all lengths」，減平均後恢復〔文件〕 | ❓ 沒有論文；第三方的觀察方向一致 |
| 19 | （這一列不是我們的數字）vLLM 部落格：Hopper 上 128K 撈針 BF16 91%、FP8 13%、修正後 89% | 〔文件 https://vllm.ai/blog/2026-04-22-fp8-kvcache ，10/15 讀〕：沒寫是哪個模型；原因是 FA3 的 FP8 Tensor Core「imprecise FP32 accumulation」，不是 scale | ⚠️ 和我們的失敗**不是同一件事**：那是「連 attention 都用 FP8 算」的累加精度問題，我們的計劃是先反量化成 BF16 再算（03 §1），不受影響。但 H200 若真的開 vLLM 原生 FP8，要記得這件事（03 §9 已列） |

## 5. BF16 正確率（調難度之前的預設設定，32K）

我們的數字：第 1 輪 `e0_tune/p1_default_*.csv`（run `20261015-030858-m10-e0p1-def-llama`、`-030854-…-qwen`），每個任務 60 題（T5 30 題），兩輪合併；T4 的 qa_f1 見 `e0_tune/t4_breakdown.csv`。**我們和上游的差別**（`code/m10_tasks.py` 檔頭）：haystack 用 noise、vt 的 1-shot 縮到 block 0 裡、T4 是兩段 gold 段落放進同一個位置帶。

| # | 我們的（Llama-3.1-8B-Instruct） | 論文的數字 | 判定 |
|:--|:--|:--|:--|
| 20 | T1 多鍵撈針 100%、T2 一鍵多值 97.5% | RULER（COLM 2024）Table 13：Meta-Llama-3.1-8B-Instruct（Table 4 列的 checkpoint〔原文 p.17〕），8 個 NIAH 任務平均 32K：99.6〔原文 p.25〕。Double-P（arXiv 2602.05191v1）Table 5，LLaMA-3.1-8B 的完整 attention 列，32K：mk3 100.00、mv 97.38〔原文 p.12〕 | ✅ |
| 21 | T3 變數追蹤（4 跳）99.3% | RULER Table 14：VT 32K 98.8〔原文 p.26〕；Double-P：vt 99.80〔原文 p.12〕 | ✅ |
| 22 | T4 HotpotQA：string_match_part 46.7%、qa_f1 62.4% | RULER Table 16：QA 兩項（SQuAD、HotpotQA）平均 32K：65.8〔原文 p.27〕；Double-P：qa_1 75.50、qa_2（HotpotQA）54.00〔原文 p.12〕 | ✅ 和 qa_2 的 54 差 7 分，在我們 60 題的誤差內（95% CI 約 ±13）；構造也不同 |
| 23 | T5 常見詞（freq_cw 30）47.7%；30 題裡 12 題 0 分，答的是「the、and、a、to…」這種英文常用字。**10/15 追加的對照（見 §5b）**：同一個模型、同一套環境，上游 RULER 產生器＋上游 meta-llama3 模板只有 **1.0%**（照上游 HF 路徑加 BOS）／**6.0%**（不加），HF 原生 generate 和我們的還原器逐字相同 | RULER Table 15：Aggregation（CWE、FWE 平均）32K：54.1〔原文 p.26〕，**沒有單獨的 CWE**。FWE 最高 100 分，所以 CWE ≥ 2×54.1−100＝**8.2**〔算術〕。Double-P：cwe 95.17、fwe 50.40〔原文 p.12〕，但它沒寫是 Base 還是 Instruct〔原文 p.7「LLaMA3.1-8B」〕，兩項平均 72.8 也和 RULER 的 54.1 對不上。RULER 原文描述過同一種失敗：「ignore the contextual information and instead use parametric knowledge … returns high frequency words, such as "the", "an", "a"」〔原文 p.9〕 | ❌（10/15 由 ⚠️ 改判）**48% 不是我們出題造成的**：上游出題在我們的環境更低（1–6%），而且清單的規模和上游幾乎一樣（§5b）。**但上游原版題目在我們的環境只拿 1.0%，低於 RULER 論文換算出的下限 8.2%，這是和論文的矛盾**。還原器已排除（HF 原生 generate 逐字相同）。未解的可能原因：RULER 程式版本（論文 2024 年，我們的 clone 是 2026-07 的 c3f5e3b）、推論框架與硬體（論文跑 vLLM／CUDA〔未查證〕，我們是 HF transformers 5.17＋ROCm）。**影響範圍只有 CWE**：其他任務和論文一致（第 20–22 列），T5 調難度後用的是我們自己的出題（freq_cw 40，82%） |
| 24 | （我們只有 5 個任務） | RULER Table 3：Llama3.1 (8B) 13 項平均 32K：87.4〔原文 p.7〕 | ❓ 13 項平均，不能和 5 項比 |
| 25 | Qwen2.5-7B-Instruct-1M：T1 93.3%、T2 93.3%、T3 91.7%、T4 51.7%、T5 73.3% | Qwen2.5-1M 技術報告 Table 4：RULER 32K 平均 91.1〔二手：arXiv 2501.15383 HTML 經 WebFetch〕；RULER README 同一個數字，標「reported by the authors」〔文件 github.com/NVIDIA/RULER〕。查不到逐任務分數 | ❓ |

LongBench 的 HotpotQA 分數**不拿來比**：構造和計分都不同（03 §3.1）；Double-P 的 LongBench 表（Table 7〔原文 p.13〕）抽出來的欄位對不上，而且模型可能是 Base 版。

### 5b. 第 23 列的追加對照：上游 RULER 出題 vs 我們的出題（10/15）

**問題**：Llama 在 T5 只有 48%，是我們出題的差異造成的嗎？
**做法**：`code/m10_ruler_cwe_check.py`。Llama-3.1-8B-Instruct（`unsloth/…`，revision 4699cc75）、32K、全 BF16，各 30 題，生成最多 120 token，計分 string_match_all（和上游 evaluate 相同）。
- 上游：直接跑 RULER `scripts/data/synthetic/common_words_extraction.py`（commit c3f5e3b），參數和 `prepare.py` 對 `cwe` 組出來的相同：
  - template＝`Templates['meta-llama3']`（config_models.sh 對 llama3.1-8b-chat 的設定）包住上游 cwe 模板＋answer prefix；
  - freq_cw 30、freq_ucw 3、num_cw 10、1-shot、random_seed 42、`--max_seq_length 32768 --tokens_to_generate 120`。
  - MLSteam 上的 RULER clone 沒拉 Git LFS，`json/english_words.json` 只是指標檔，所以在副本裡換成 GitHub LFS 的真檔（sha256 和指標檔記的一樣；32K 用不到這個檔，只是上游程式一開頭就讀它）。
- 我們：`m10_tasks` T5 預設參數、seed 1000–1029，和第 1 輪同一批。結果和第 1 輪 30／30 題逐字相同，可重現。
- 輸出：`results/m10_pilot_mi300x/e0_tune/ruler_cwe_upstream_llama.csv`（＋`.summary.json`、`_hfgen.csv`）；run `20261015-061335-m10-ruler-cwe-llama`、`20261015-065849-m10-ruler-cwe-hfgen`。第一次 run `20261015-054607-…` 因為 LFS 指標檔失敗（JSONDecodeError）。

| 組別 | 正確率 | 0 分／滿分 | 回答成英文常用字（≥3 個） | prompt token（中位數） | 清單項目／不同詞／答案詞出現次數（中位數） |
|:--|:--|:--|:--|:--|:--|
| 上游，照上游 HF 路徑 tokenize（`tokenizer(prompt)` 預設加 BOS → 開頭 2 個 BOS，全文 3 個） | **1.0%** | 29／0 | 29 | 32,555 | 5,997／1,909／30 |
| 上游，不另加 BOS（開頭 1 個，全文 2 個） | **6.0%** | 27／0 | 27 | 32,554 | 5,997／1,909／30 |
| 上游，HF 原生 `model.generate`（整段 prefill，不經過我們的還原器；前 10 題） | **0%** | 10／0 | 10 | — | 同上 |
| 我們的 m10_tasks T5 | **47.7%** | 12／5 | 11 | 32,805 | 6,035／1,927／30 |

**讀法**
1. **不是出題差異**：清單的規模、不同詞數、答案詞次數兩邊幾乎一樣，上游的題目反而只有 1–6%。
2. **不是還原器**：同一批上游 prompt，HF 原生 generate 和還原器 10／10 題逐字相同。
3. **差在 prompt 的包裝**〔判讀〕。上游的 1-shot 範例是一整段 chat（自己的 `<|begin_of_text|>`、user／assistant 標頭、答案），後面再接一段完整的 chat；我們是 Llama 官方 chat template（含系統段「Cutting Knowledge Date… Today Date: 26 Jul 2024」），範例放在同一個 user 訊息裡。兩邊清單一樣，正確率差 40 個百分點以上，表示這個模型在 32K 的 CWE 上對 prompt 包裝非常敏感，失敗方式都是 RULER 記錄過的「改用常識回答」。這一點沒有另外做消融，只是推論。
4. **和論文的矛盾還在**：RULER 論文的 Aggregation 54.1 換算 CWE ≥ 8.2，上游原版題目在我們的環境只有 1.0%（30 題，0／30 滿分）。可能的原因：RULER 程式版本（論文 2024 年、clone 2026-07）、推論框架與硬體（我們是 HF transformers 5.17＋ROCm 7.0）。下一步建議：H200 第一天（CUDA）用 vLLM 跑同一份 30 題上游 jsonl（`/mlsteam/data/tiara/runs/20261015-061335-m10-ruler-cwe-llama/ruler_cwe/cwe/validation.jsonl`）。
5. **對精度實驗的影響**：T5 是全域對照，判準是「同一題對 E0 配對」，絕對難度不進判定。T5 已改用我們自己的出題、freq_cw 40（82%，CANDIDATE_PARAMS.json）。報告裡要寫明：T5 不是 RULER 原版 CWE 的分數，不能和 RULER 的表直接比。

## 6. 其他

| # | 我們量到的 | 參考 | 判定 |
|:--|:--|:--|:--|
| 26 | torch 2.10（ROCm 7.0）、2.12.1（ROCm 7.2）：e4m3fn 超出範圍變 NaN，不會截在 448〔RUNLOG Milestone 10-pilot 異常 3〕 | cards/P04 §3：torch 2.9 變 NaN、2.13 截在 448〔程式碼〕 | ✅ 版本落在兩者之間，行為和 2.9 一樣；codec 一律自己先截，不受影響 |
| — | 第一階段與精度計劃裡引用的其他論文數字（Cake 的頻寬帶、ZipCache probe 消融 52.08 對 52.54 等） | 已在 [V_verification_20261013](V_verification_20261013.md) 複核 85 項，這次不重查 | — |

---

## 所以呢

1. **可以放心的**：
   - **Qwen 的 K 偏移是真的，而且很穩**。不同權重（BF16 對 GPTQ）、不同長度、不同平台量到的最大值、層、channel、甚至「22／512」都一樣。
   - 「減掉偏移就好」在兩邊的指標上都成立。
   - 重建誤差和 KVTuner 對 Llama-3.1 的表在 0.1 個百分點內一致，表示我們的 codec 和量法沒問題。
   - NIAH、VT、HotpotQA 的 BF16 正確率和 RULER、Double-P 一致，表示出題和作答流程沒有明顯偏差。
2. **報告裡要講清楚的**：
   - Qwen 偏移的**位置**有論文支持（Massive Values，ICML 2025）；「**是 k_proj bias 造成的**」目前只有 GitHub issue 的推測，加上我們「沒有 bias 的 Llama 沒有這種常數偏移」的旁證。論文裡要寫成推論，不是事實。
   - Qwen 的 FP8 KV 失敗，**沒有任何論文量過**；論文只量過整數 4-bit（KVTuner、DiffKV）。
   - vLLM 部落格的「91%→13%」是另一個機制（FA3 累加），不要和我們的格式問題混在一起引用。
3. **唯一的 ❌（第 23 列，§5b）只影響 T5**：
   - 已排除：我們的出題（上游出題更低，1–6%）、還原器（HF 原生 generate 逐字相同）。
   - 還沒排除：RULER 程式版本、推論框架與硬體（ROCm 對 CUDA）。下一步是 H200 第一天用 vLLM 跑同一份 30 題。
   - 為什麼不擋精度實驗：T5 是對 E0 的配對比較，絕對難度不進判定；T5 已改用 freq_cw 40（82%）。報告裡要寫明 T5 不是 RULER 原版 CWE 的分數。
4. 其餘沒有 ❌。

## 附錄：本頁自己算的數字怎麼來

```bash
# 第 2、5 列：每 (head, channel) 的 max|K| 統計（讀 runs 目錄的 raw 檔，不進 git）
cd /mlsteam/data/tiara/runs && python3 - <<'EOF'
import csv, collections, statistics
for m, d in (("qwen","20261015-030452-m10-kvstats2-qwen"),("llama","20261015-030742-m10-kvstats2-llama")):
    per_hc = collections.defaultdict(float); per_layer = collections.defaultdict(float)
    for r in csv.DictReader(open(f"{d}/raw/kv_chan_{m}.csv")):
        k = float(r["maxabs_K"]); per_hc[(int(r["head"]), int(r["chan"]))] = max(per_hc[(int(r["head"]), int(r["chan"]))], k)
        per_layer[int(r["layer"])] = max(per_layer[int(r["layer"])], k)
    v = sorted(per_hc.values())
    print(m, len(v), statistics.median(v), v[int(0.95*len(v))-1], sum(x > 100 for x in v), sorted(per_layer.items(), key=lambda x: -x[1])[:3])
EOF
# 輸出（2026-10-15）：qwen 512 8.88 90.5 22 [(27, 422.0), (0, 172.0), (3, 75.0)]；llama 1024 10.66 19.9 0 [(29, 34.5), …]

# 第 7 列：e4m3 的理論相對誤差（模擬）
python -c "import torch; x=torch.randn(1<<24); y=x.to(torch.float8_e4m3fn).float(); print(float((x-y).norm()/x.norm()))"   # 0.0265
```

- 論文原文這次讀的檔：KIVI `papers/kivi2024_arXiv2402.02750.pdf`、KVTuner `papers/kvtuner2025_arXiv2502.04420.pdf`、KVQuant `papers_d6/arXiv2401.18079.pdf`（都在 `/mlsteam/data/tiara/`）；RULER arXiv 2404.06654v3、Double-P arXiv 2602.05191v1（WebFetch 下載 PDF 後抽文字）。
- 沒有自己讀原文的：Qwen2.5-1M 報告、Massive Values、DiffKV（都標〔二手〕）。
