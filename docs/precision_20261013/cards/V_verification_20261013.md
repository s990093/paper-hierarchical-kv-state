# V　獨立複核：precision_20261013 的事實查核

- **日期**：2026-10-13
- **誰做的**：zero-context verifier（沒有寫過這批文件；只回原文、原始碼、官方頁面核對；**不參考** P01–P05 卡片的結論，卡片只用來找「它引用的是哪個 repo／哪一版」）
- **範圍**：
  - 主要：`02_papers.md` §1.1 表、§1.2、§2.1 表、§2.2、§3.1、§3.2 表、§4.1 表、§5.1 表、§5.2。另順手查了 §3.3 S3、§4.2 兩句和 SemKV／SparKV 有關的判讀（任務點名要查）。
  - 次要：`01_advisor_feedback.md` §3、`03_plan_h200.md` §1 的算術、`04_story.md` §5 的投稿日期。
- **打開過的來源**（下載到 `/private/tmp/claude-501/verif_20261013_dl/`，只讀不執行）：
  - Cake：PMLR v267 jin25d 正式版 PDF（13 頁，`raw.githubusercontent.com/mlresearch/v267/main/assets/jin25d/jin25d.pdf`）、arXiv 2410.03065 v1（16 頁）、v2（12 頁，和本機 `papers/cake2025.pdf` 逐位元相同）。GitHub API：`jsw-zorro/vllm`、`xenshinu/LMCache`（dev、patch-1 兩個分支）。
  - CacheGen：arXiv 2310.07240v6（19 頁）；`UChi-JCL/CacheGen@6bed34c` 的 `src/utils.py`、`run_adaptation.py`、`LMCache/.../cachegen_encoder.py`；`LMCache/LMCache@240506e`（2026-10-13 的 commit）的 `cachegen_basics.py`（舊版與 v1）、`docs/source/mp/cachegen.rst`、`docs/design/v1/distributed/l2_adapters/serde_wrapper.md`。
  - 位置文獻：StreamingLLM 2309.17453、MuKV 2605.22269v1＋`IMBALDY/MuKV@2127b87`、KVQuant 2401.18079v6、KIVI 2402.02750v2、SKVQ 2405.06219v3、Kitty 2511.18643v1、InnerQ 2602.23200v2、KVSink 2508.04257v1、ZipCache 2405.14256v1（venue 用 OpenAlex 的 NeurIPS DOI 10.52202/079017-2181 核對）、SemKV 2608.28911v1、Lost in the Middle 2307.03172、Found in the Middle 2406.16008v2。
  - 最接近的做法：PatchKV 2609.26219v1、SparKV 2604.21231v3、HA-RAG 2510.20878v1、Cocktail 2503.23294、MoQAE 2506.07533、PM-KVQ 2505.18610v2、DiffKV 2412.03131v3、KVTuner 2502.04420v5、SCBench 2412.10319v2。各篇 arXiv 頁面的 Comments 欄（venue）。
  - vLLM：tag `v0.28.0`＝commit `2cf0a6915ce5`（GitHub API 確認）的 `attention.py`、`kv_cache.py`、`auto_awq.py`、`config/cache.py`、`fa_utils.py`、`flash_attn.py`、`flashinfer.py`、`triton_attn.py`、`flex_attention.py`、`turboquant_attn.py`、`triton_reshape_and_cache_flash.py`、`platforms/cuda.py`、`csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh`；移除前的 commit `0cf49a5`（2026-08-03 14:59）的 `attention.py`、`envs.py`；PR #49389 的 GitHub API 資料。
  - FlashInfer：tag `v0.6.16.post3`＝commit `9dc1b2495b40` 的 `include/flashinfer/attention/prefill.cuh`、`decode.cuh`、`flashinfer/prefill.py`。
  - 第三方：`ThePie88/vLLM-ROCm-Windows` issue #28 本文。
  - 專案內：本 worktree 的 git（`c4d24f7`、`52803bc^:results/m5_quality/needle_pilot_32k.csv`）、`results/m8_directions/d6_quality_inventory.csv`、`docs/research_20261010_directions/D6_precision.md`、`docs/research_20261010_followup/F3_dequant_fp8.md`。
  - 其他：HF `Qwen/Qwen2.5-7B-Instruct-1M` 的 `config.json`；ICPE 2027 官網 `/important-dates/`、`/tracks-and-submissions/research-paper-track/`；CCGrid 2027 EasyChair CFP；CVPR 2027 `/Dates` 與 `/CallForPapers`。
- **判定記號**：✅一致／⚠️需修正（事實大致對，但有缺漏、出處或措辭會誤導）／❌錯誤／❓未能查證。

---

## 查核表

| # | 文件與位置 | 原句（短） | 判定 | 正確內容與出處 |
|:--|:--|:--|:--|:--|
| 1 | 02 §1.1 方向 | 計算從 chunk 0 往後，I/O 從最後往前 | ✅ | ICML App. A 第 2–5 步、Algorithm 1（p.12） |
| 2 | 02 §1.1 會合點 | 不事先規劃；算完一個 chunk 就查下一個是否已載入；vLLM 每個 engine step 後用 `is_resident`；沒有估計器 | ✅ | Algorithm 1 第 3 行 `IsInCPUMemory`（p.12）；App. B.2 第 4–6 步（p.13）；「incorporating an estimation mechanism」列為 future work（§5.4，p.7）；§5.5 的 fallback（p.7） |
| 3 | 02 §1.1 粒度 | 計算端 budget 512；I/O 端 128；兩者怎麼對齊原文沒寫 | ⚠️ | 512、128 都對（§5.1，p.5）。但 App. B.2 第 1–2 步（p.13）寫「依 scheduled token budget 切 chunk，把這些 chunk 的 key 反序推進佇列」，和 128 有張力。應寫成「原文兩處說法不一致」，不是單純「沒寫」 |
| 4 | 02 §1.1 GPU 忙 | budget 佔比（512 給 256＝50%）；排程優先序 | ✅ | §5.1 p.6；優先序在 p.5（decode → 非前綴 chunked prefill → 前綴 chunked prefill） |
| 5 | 02 §1.1 頻寬 | 延遲注入；頻寬點對到雲端 egress、SSD | ✅ | §5.1「I/O Bandwidth Control」p.5；Table 2 p.6（7／25／32／56／100 Gbps） |
| 6 | 02 §1.1 實作 | 約 1,000 行；vLLM v0.6.2、LMCache v0.1.4 | ✅ | 1,000 行在 App. B 開頭（p.12）；版本在 §5.1 p.5 |
| 7 | 02 §1.1 下一輪 | 事先算好存好；寫回與下一輪原文沒寫 | ✅ | 「precompute and store all requests' KV cache in advance」（p.5）。三版全文搜 write back／next turn 都沒有 |
| 8 | 02 §1.1 程式碼 | 沒公開；一作 vLLM fork 0 commit；二作 LMCache fork 只有 raw-byte serde | ✅ | GitHub compare：`jsw-zorro/vllm` main 相對上游 ahead 0。`xenshinu/LMCache` dev 分支本人 commit 只有 2024-09 的 `fast_serde.py`（`@46bdb8f` L12–29：`view(torch.uint8)…tobytes()`／`frombuffer(bf16)`）；patch-1 分支只多一個 2025-01 的 `local_backend.py` 小改。三版論文都沒有程式碼連結 |
| 9 | 02 §1.2 位置 | v2 §5.5／Table 6／p.7；ICML §5.6／Table 7／p.8；ICML 多了 §5.3 | ✅ | 兩版都核對過。ICML §5.3 是 chunk 大小，Table 4 是新增的 |
| 10 | 02 §1.2 用了什麼 | 「最常見的 8-bit」和「KVQuant 的 3-bit」；8-bit 歸屬有歧義 | ✅ | ICML §5.1 p.6 原句可兩讀 |
| 11 | 02 §1.2 設定 | LongAlpaca-13B、16K、512、2×A100 TP、100%、7／25／32／56；16-bit 欄＝Table 3 同列 | ✅ | Table 7 標題（p.8）；TP 在 §5.2（p.6）；Table 3 的 2×A100 100% 列 8.10\1.10、3.03\1.43、2.63\1.59、1.97\2.00，和 Table 7 16-bit 欄逐格相同 |
| 12 | 02 §1.2 品質 | 三個版本都沒有品質數字 | ✅ | v1、v2、ICML 全文搜 accuracy／perplexity／F1／ROUGE／quality／lossless：只有引文與倫理段，沒有任何品質數字 |
| 13 | 02 §1.2 前段精度 | 前段重算精度、何時量化、在哪反量化，原文都沒寫 | ✅ | 確認沒寫。補充：ICML p.5 有一句「We use BF16 as the default kv cache data type」，可當〔判讀〕「前段是 BF16」的旁證 |
| 14 | 02 §1.2 推算 | 只載入基線：16→8 縮 1.88–1.94 倍，16→3 縮 4.21–5.02 倍；常數項沒變大 | ✅ | 我用 T_io/T_comp＝S_io/S_comp 重算：8-bit 1.884／1.895／1.914／1.940；3-bit 4.206／4.503／4.644／5.022。對 1/BW 直線擬合的截距（以 T_comp 為單位）：16-bit 0.069、8-bit 0.057、3-bit 0.060 |
| 15 | 02 §1.2 ICML §6 | 「agnostic to its specific representation or compression scheme」；證據只有 TTFT | ✅ | ICML §6 p.9 原句 |
| 16 | 02 §2.1 寫入 | 先 prefill，沿 token 切 chunk（預設 1.5K），每個 chunk 多個等級、可獨立解碼 | ✅ | §5.3 p.6 右欄起、p.7（1.5K 在 p.7）。出處可寫成 p.6–7 |
| 17 | 02 §2.1 等級內部 | 10 token 一組，anchor 8-bit，其餘差值；層分三組淺層較細；算術編碼 | ✅ | §5.2 p.6（「ten contiguous tokens」「8-bit… anchor token」「three layer groups」）；App. C.2 p.19（bins 0.5／1／1.5） |
| 18 | 02 §2.1 讀取 | 用上一個 chunk 的吞吐量；挑損失最小且在 SLO 內；送文字重算是損失最小 | ✅ | p.7「least compression loss (i.e., text format or lowest encoding level)」；Algorithm 1 先判斷 `time_recompute ≤ remaining_time`（p.19） |
| 19 | 02 §2.1 重算 | 文字 chunk 用前一個已解碼 chunk 的 KV 算，所以依序 | ✅ | p.7 |
| 20 | 02 §2.1 重疊 | 只寫「傳 i 時解碼 i−1」；重算能否並行沒寫 | ✅ | §6 p.8。全文搜 recompute／in parallel／overlap，沒有重算與傳輸並行的敘述 |
| 21 | 02 §2.1 位置 | 沒有位置規則；Fig. 7 被重算的是中間 chunk；依 token 位置分組熵下降小 | ✅ | Fig. 7（p.6，t=2s 切到 KV compute、t=4s 切回）；§5.1.3 p.5 |
| 22 | 02 §2.1 敏感度 | 量的是層，淺層最敏感 | ✅ | Insight 2、Fig. 4（p.5） |
| 23 | 02 §2.1 品質 | ≤2% accuracy、F1 <0.1%、PPL <0.1；兩次稱 8-bit 幾乎無損 | ✅ | §7.2 p.9；p.2「nearly lossless」、p.9「almost lossless」 |
| 24 | 02 §2.1 代價 | 8.5K token、Llama-13B 全部版本約 5 GB | ✅ | App. E p.19 |
| 25 | 02 §2.2 artifact 順序 | 先試編碼等級，三級都不行才送文字 | ✅ | `src/utils.py:303–312`（`for quant_level in np.arange(3,0,-1)`，`if not found_cache` 才用 `chunk_delay`） |
| 26 | 02 §2.2 頻寬 | 直接取軌跡上該 chunk 的值 | ✅ | `src/utils.py:300`（`bw = all_bws[chunk_id]`） |
| 27 | 02 §2.2 重算成本 | 固定 0.2 秒 | ⚠️ | 事實對，但出處不在 `utils.py`：`chunk_delay = 0.2` 在 `run_adaptation.py:77`，`utils.py:311` 才使用它 |
| 28 | 02 §2.2 重算用原始 KV | 「重算」的 chunk 直接拿全精度 KV | ✅ | `src/utils.py:365–366`（`split_kv(orig_kv, …)`） |
| 29 | 02 §2.2 沒有 anchor | 編碼器沒有 anchor／差值 | ✅ | `LMCache/.../cachegen_encoder.py` 只有逐 token amax 量化（L42–57），搜 anchor／delta 無結果 |
| 30 | 02 §2.2 現行 LMCache | 只剩一組固定 codec；MP mode 寫「尚未實作」 | ✅ | `cachegen_basics.py` 的 `CacheGenConfig.from_model_name` 每個模型一組固定 bins；舊版與 v1 都搜不到 bandwidth／level／recompute；`docs/source/mp/cachegen.rst`「not implemented yet」 |
| 31 | 02 §3.1 StreamingLLM 機制 | 「大部分 attention 落在最前面幾個 token」；softmax 總和 1；最前面對所有 token 可見 | ⚠️ | 機制對（§1 p.2；§3.1 p.4–5）。但原文是「a surprisingly large amount」，不是「大部分」。建議改措辭 |
| 32 | 02 §3.1 Table 1 | Llama-2-13B：0+1024 5158.07；4+1020 5.40；4"\n"+1020 5.60 | ✅ | Table 1 p.5 |
| 33 | 02 §3.1 留幾個 | 預設 4；Falcon-7B 1 個就夠；Llama-2-7B 要 4 | ✅ | Table 2 p.5（Falcon 17.90→12.12 後持平；Llama-2-7B 11.88／10.51／9.59／9.54）；§4 開頭 p.6「default to using four initial tokens」 |
| 34 | 02 §3.1 只有驅逐 | 沒有量化實驗；App. D 的 1750+1750 是驅逐，歸因於提示資訊 | ✅ | 全文 quantiz 只出現在 p.4 一句類比與參考文獻；App. D p.17「likely due to the loss of crucial initial input prompt information」 |
| 35 | 02 §3.1 MuKV 剪枝 | 做的是剪枝，不是降精度 | ✅ | §2.2 p.2「we focus on pruning」；§3.3 p.4；Table 3 的 DCP(67%) 是丟掉 67% token |
| 36 | 02 §3.1 MuKV 位置偏差 | 是單獨編碼的視覺區塊內順序（49／196／784） | ✅ | `model/mukv_dualsignal_compression.py:474–479`：每個 block 只接 `init_kv_cache` 單獨跑；粒度 49／196／784 在 L62–64 |
| 37 | 02 §3.1 MuKV 分數 | 因果遮罩下沒正規化的欄加總 | ✅ | 同檔 L343：`attn.mean(0).mean(0).sum(0)`（對 query 維度加總） |
| 38 | 02 §3.1 MuKV 消融 | 位置偏差是 p.8 的事後解讀；設計理由是 overfit（p.4）；Att／Freq／兩者 55.9／56.6／57.3 | ✅ | arXiv 2605.22269v1：p.4「may overfit」；p.7 Table 3；p.8「calibrate this positional bias」。頁碼是 arXiv 版（CVF 版未核對） |
| 39 | 02 §3.2 KVQuant | 第 1 個 token 留 FP16；2-bit 23.34→9.59；4-bit 4.62→4.60 | ✅ | §3.5 p.6「keeping only the first token in fp16」；App. K、Table 13 p.20（nuq2、nuq4，dense-only 列）。NeurIPS 2024（arXiv Comments） |
| 40 | 02 §3.2 KIVI | 最近 128 全精度；Llama-2-13B GSM8K 12.21／20.77／22.67 | ✅ | R＝128（p.6）；Table 3 p.8。12.21 是 fake「2bit (K-C, V-T)」那列 |
| 41 | 02 §3.2 SKVQ | 最近 128＋最前 5；加窗口 +10.18，再加 sink +0.15 | ✅ | §4.3 Table 3 p.8。注意中間還隔了 clipping（+0.71）與 reorder（+1.55），sink 是第 4 步 |
| 42 | 02 §3.2 Kitty | 前 32 個；2-bit Qwen3-8B MATH-Algebra 47.29→74.92 | ✅ | §3.1 p.3–4；Table 2 p.4（KIVI-K2V2 vs KIVI-K2V2*） |
| 43 | 02 §3.2 InnerQ | 32＋96 vs 0＋128；短任務 +1.7；LongBench +0.2；越長越小 | ✅ | p.8（設定）、p.9（+1.7、+0.2、「diminishes as the sequence length increases」）；§6.1 p.13 |
| 44 | 02 §3.2 KVSink | sink 不一定在最前（第 14 個）；INT4 保留前 5 個 59.5，預測的 5 個 5.8 | ✅ | Fig. 1 p.2（tokens 0 與 14，0 起算）；§5.2 p.9。建議補上條件：LLaMA2-70B、per-token static INT4 |
| 45 | 02 §3.2 ZipCache | 累積 attention 除以非零個數；GSM8K CoT LLaMA3-8B 55.88／47.99／53.75 | ✅ | §4.2 Eq. 8 p.6；Table 3 p.8。NeurIPS 2024（OpenAlex DOI 10.52202/079017-2181） |
| 46 | 02 §3.2 SemKV | 比較隨機和 8 種指標選 25%；Llama：8 種指標沒差；Mistral：隨機 Δ＝−0.0213 | ⚠️ | 數字對（p.2、p.14、p.24）。但 Llama 主比較（§4.3、Table 3 p.17）只有 8 種指標，**沒有隨機**；Mistral 上隨機是**顯著變差**（p<10⁻³），8 種指標都安全。兩級是 2.585／2.322 bit，不是 8-bit。表格寫法容易被讀成「隨機和指標一樣好」 |
| 47 | 02 §3.2 Lost in the Middle | 答案文件位置 → U 形；全文沒有量化 | ✅ | §2.3 p.5；全文搜 quantiz 無結果 |
| 48 | 02 §3.2 Found in the Middle | U 形偏差、校正要 query；Recall@3 0.2052→0.6832 | ✅ | Table 3 p.6（K＝20）。Findings ACL 2024（arXiv Comments） |
| 49 | 02 §3.3 S3（加查） | ZipCache、Found in the Middle、SemKV「都需要 query；寫入時沒有 query」 | ❌ | SemKV 的 8 種指標中 7 種（hidden-state 4 種、key-norm、value-norm、self-information）只用 token i 自己與之前的狀態（p.7），因果模型下不受後面的 query 影響，寫入時就能算。只有 accumulated attention（Σ_{j≥i}）要看後面的 token |
| 50 | 02 §4.1 PatchKV | 重算集不存；只有載入集量化，逐 block 選 FP16／K8V8／K8V4 | ⚠️ | 卸載時**所有** suffix block 都依凍結的 tag 量化並存進 CPU checkpoint（p.6「During offload, a fused quantize_gather kernel…」）；恢復時 R＝BlockRound(D∪H) 重算、不讀存的那份，只有 F 讀回並反量化（Eq. 12，p.5）。所以是「重算集不讀」，不是「不存」 |
| 51 | 02 §4.1 SparKV | 逐層非均勻量化；Cake＋載入段量化當 baseline；位置是 chunk 成本的差勁代理（同位置計算時間差 3–5 倍） | ⚠️ | baseline 對（「Strong Hybrid [25]」，p.8，串流段用同樣量化）。但 3–5 倍是 **sparse attention（SpargeAttention）** 下的計算時間（p.4 引 §III-B）；dense attention 下同位置成本相同。而且它講的是**成本**，不是品質敏感度 |
| 52 | 02 §4.1 CacheGen | 頻寬＋SLO；讀取時逐 chunk；可送文字重算、依序 | ✅ | 見 #16–#21 |
| 53 | 02 §4.1 HA-RAG | 熱度（熱→INT8、次熱→E4M3→E5M2…）；離線；4 層；沒有重算；沒和均勻 8-bit 比品質 | ⚠️ | 熱度分級對：最熱 10% INT8、再 10% E4M3、再 10% E5M2、其餘 GSE-8（p.8）；4 層 GPU／pinned／pageable／disk；沒有重算。但它**有**逐格式單獨的 ROUGE-1 對 TurboRAG（Fig. 10–11，p.7）；只是混精度整體只和未壓縮 TurboRAG 比（Fig. 13，p.9）。PDF 首頁是 HPCA 2026 |
| 54 | 02 §4.1 LMCache fp8 serde | 依存放層；寫到該層時；沒有重算決策 | ✅ | `SerdeL2AdapterWrapper` 把 serde 包在 L2 adapter 的 store path（`serde_wrapper.md`） |
| 55 | 02 §4.1 Cocktail | query 和 32-token chunk 的相似度；推論前 | ✅ | §III p.3；chunk 32 在 §IV。DATE 2025（arXiv Comments） |
| 56 | 02 §4.1 MoQAE | 學出來的 router；第一個 chunk 固定 FP16 | ✅ | 「routing-freezing mechanism to the first chunk… FP16」。ACL 2025 |
| 57 | 02 §4.1 PM-KVQ | 記憶體壓力＋逐層敏感度；最前與最近 128 保持 16-bit；快滿才降 | ✅ | 原文「first and most recent 128 tokens in INT16」本身有歧義，另一處寫「storing the first token in INT16」，所以是第 1 個＋最近 128。ICLR 2026（arXiv Comments） |
| 58 | 02 §4.1 DiffKV | 每 token 收到的平均 attention；最近 64；prompt 一次、decode 降級 | ✅ | 分數＝「averaging the N−i attention scores it receives」；W＝64；Algorithm 1 的降級路徑。SOSP 2025 |
| 59 | 02 §4.1 KVTuner | 逐層敏感度；離線 | ✅ | 標題與「offline calibrated layer-wise」。ICML 2025 |
| 60 | 02 §4.1 SemKV | 8 種指標，刻意不用位置；prefill 後 | ✅ | p.3「rather than committing to… fixed positional heuristic」 |
| 61 | 02 §4.2 反方證據（加查） | 「SemKV 說品質懸崖之上，隨機選和各種指標差不多」 | ❌ | 和原文相反。Llama 懸崖之上只比了 8 種指標（彼此沒差），隨機沒在這個點報告；Mistral 上隨機顯著變差（Δ＝−0.0213，p<10⁻³），指標都安全（p.24、Fig. 16）；懸崖內 r＝50% 時 delta-norm 比隨機差（Δ＝−0.0189，p.15）。SemKV 對「隨機是強對照」是混合證據 |
| 62 | 02 §4.2 反方證據（加查） | 「SparKV 說位置是差勁的代理」→ 隨機 25% 可能是強對照 | ⚠️ | SparKV 講的是 sparse attention 下的**成本**代理（#51），和精度敏感度無關，不能支持「R25 會很強」 |
| 63 | 02 §4.2 PatchKV 5.6% | 混精度只比均勻 INT8 快 5.6% | ✅ | p.8：固定修復集、只換表示法，TTFT 比 INT8 快 5.6%，F1 差 0.05 內（Fig. 9，Llama-3.3-70B） |
| 64 | 02 §4.2 SCBench | 只測 KIVI 2-bit，只報平均 | ✅ | 量化只有 KIVI（2 bit、group 32）；逐輪的 Fig. 9–10 圖例沒有 KIVI。ICLR 2025 |
| 65 | 02 §5.1 scale 來源 | 只有兩種：checkpoint 的 per-tensor scale，沒有就 1.0；BF16、AWQ 都 1.0 | ⚠️ | 對 `fp8`／`fp8_e4m3` 而言對（`attention.py:124-145、:580-592`；`kv_cache.py:99-111`）。但 v0.28.0 還有第三種：`fp8_per_token_head`／`int8_per_token_head`／`int4_per_token_head` 在寫入時逐 (token, head) 動態算 scale（`kv_cache.py:82-93`；`config/cache.py:32-34`）。AWQ 對 attention 層回傳 `None` 在 `auto_awq.py:359`，不在 :285-330 |
| 66 | 02 §5.1 `--calculate-kv-scales` | 已移除（PR #49389，2026-08-03）；移除前「第一個 batch、整個 tensor 一個值、max÷200」 | ⚠️ | 移除與日期對（PR API：merged_at 2026-08-03T15:15:13Z）。但除數不全是 200：`Q_SCALE_CONSTANT`＝200、`K_SCALE_CONSTANT`＝200、**`V_SCALE_CONSTANT`＝100**（`vllm@0cf49a5 envs.py:1328-1333`；`attention.py:148-150、584-594`） |
| 67 | 02 §5.1 3090 誰收 FP8 | 只有 FlashInfer；FA 在 sm_86 是 FA2 不收；Triton per-tensor FP8 寫入要 SM89 | ✅ | `fa_utils.py:223-243`（只有 FA3@SM90／FA4@SM100）；`flashinfer.py:408-417`；`triton_reshape_and_cache_flash.py:397-401`。更強的證據：`triton_attn.py:540-549` 在 SM89 以下拒收**所有** `fp8*` KV（含 `fp8_per_token_head`）；FLEX 不收 FP8 |
| 68 | 02 §5.1 同一路徑嗎 | BF16 預設 FA2，FP8 走 FlashInfer | ✅ | `platforms/cuda.py` 非 SM100 的優先序：FLASH_ATTN → FLASHINFER → TRITON_ATTN → FLEX |
| 69 | 02 §5.1 FlashInfer 在 sm_86 | Q 保持 BF16；prefill 把 FP8 K/V 轉 BF16 再算，decode 轉 FP32 | ✅ | vLLM `flashinfer.py:953-961`；FlashInfer `prefill.cuh:1095-1126`（`repack_fp8_tile_to_bf16`）、`decode.cuh:72-79`（`vec_t<float>` cast_load） |
| 70 | 02 §5.1 超過 ±448 | 截在 448（`__NV_SATFINITE`） | ✅ | 檔案是 vLLM 的 `csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh:490-503`（建議把 repo 與路徑寫全） |
| 71 | 02 §5.1 混精度 | 沒有依位置／逐 block；只有 `--kv-cache-dtype-skip-layers` | ✅ | `config/cache.py:114-116`。`*_diffkv` 後端是「K、V head 維度不同」，不是混精度 |
| 72 | 02 §5.1 q_scale 的坑 | 帶 q_scale 時 Q 沒量化，FlashInfer 仍乘進 softmax 縮放 | ✅ | 程式碼支持：FlashInfer `flashinfer/prefill.py:2720-2721`（`sm_scale *= q_scale`），vLLM `flashinfer.py:2154` 傳入。仍未實測（原文已標） |
| 73 | 02 §5.2 不是溢位 | 第三方 max\|K\| 419–423（GPTQ、未重現）；MI300X 動態 FP8 0/20 | ✅ | fork issue #28 本文（Qwen2.5-7B-Instruct-GPTQ-Int4、gfx1100）；0/20 見 #77 |
| 74 | 02 §5.2 K 偏移 | 400 附近格寬 32；只改 scale 第 27 層 0.71–0.94；先減每 channel 平均降到 0.031 | ⚠️ | 格寬 32 對（e4m3 在 [256,512) 的間距 2⁸/8）；0.709–0.937 對。但 0.031 是「減每 (head, channel) 平均**再加**每 (head, channel) scale」；只減平均、scale 1.0 是 0.070，減平均＋per-tensor scale 是 0.066 |
| 75 | 01 §3 3090 撈針 | BF16 20/20、FP8 1/20、INT8 19/20、INT4 0/20；AWQ 權重 | ✅ | `git show 52803bc^:results/m5_quality/needle_pilot_32k.csv`：80 列、單一 run_id `20260831-181930-m5-needle`、qwen-awq、32,245 token；kv_dtype 為 auto／fp8／int8_per_token_head／int4_per_token_head |
| 76 | 01 §3 MI300X 撈針數字 | BF16 20/20、FP8 2/20、FP8 動態 0/20、INT8 18/20、INT4 0/20 | ✅ | `results/m8_directions/d6_quality_inventory.csv:249-253`（原 CSV 遺失，由 stdout 摘要轉錄） |
| 77 | 01 §3 MI300X 長度 | 「129K 撈針」 | ⚠️ | 盤點檔只寫「requested --needle-ctx=131072 (actual prompt tokens only in lost CSV)」。129,024 是同設定 **Llama** run 的實際長度（D6 §4.5）。Qwen 的實際長度查不到 |
| 78 | 01 §3 kernel 混淆 | 動態 FP8 走 TRITON_ATTN（F3 §4.4），比靜態 FP8 還差 | ✅ | F3 §4.4 表：fp8_ptk、int8、int4 都走 TRITON_ATTN。補充：所以 fp8_ptk（0/20）和 int8（18/20）是**同一個 kernel**，這組對照沒有 kernel 混淆 |
| 79 | 01 §3 c4d24f7 | 高斯合成；靜態 2.64% vs 動態 2.56%；INT8 0.65% | ✅ | commit 訊息與 `code/quant_error.py:93`（`torch.randn`） |
| 80 | 01 §3 Llama | MI300X 64K／129K 五種精度都 20/20 | ✅ | D6 §4.5 表；盤點檔 240–248、602– 列（runs 20260916-062139、20260917-103133） |
| 81 | 03 §1 KV 大小 | 28 層 × 2 × 4 head × 128 × 2 B＝56 KiB；512 token＝28 MiB | ✅ | HF config：`num_hidden_layers` 28、`num_key_value_heads` 4、`hidden_size` 3584／`num_attention_heads` 28＝128（config 沒有 `head_dim` 欄）。57,344 B＝56 KiB；×512＝28 MiB |
| 82 | 03 §1 FP8 block | 14 MiB＋scale 896 B | ✅ | 28 MiB÷2；28×2×4×4 B（FP32 scale）＝896 B |
| 83 | 04 §5 ICPE 2027 | 摘要 11/9、全文 11/16 AoE；和 CVPR 全文同一天 | ✅ | ICPE 官網 Important Dates（Research Track：Nov 9／Nov 16, 2026，「All deadlines are AoE」）；CVPR 2027 Dates：Submission Deadline Nov 16 '26 AoE |
| 84 | 04 §5 CCGrid 2027 | 摘要 11/24、全文 12/1 AoE；10 頁含參考文獻；雙盲；有 LLM 系統 track | ✅ | EasyChair CFP：Nov 24／Dec 1, 2026（AoE）；「no more than 10… pages, including references」；「double-blind」；Track 4「Systems for LLM Applications and Agentic AI」 |
| 85 | venue 標註（散見 02） | StreamingLLM ICLR 2024、KVQuant NeurIPS 2024、KIVI ICML 2024、SKVQ COLM 2024、KVSink COLM 2025、ZipCache NeurIPS 2024、MuKV CVPR 2026、CacheGen SIGCOMM 2024、SparKV IoT-J、Found in the Middle Findings ACL 2024、Cocktail DATE 2025、MoQAE ACL 2025、DiffKV SOSP 2025、KVTuner ICML 2025、SCBench ICLR 2025、Cake ICML 2025 | ✅ | PDF 頁首或 arXiv Comments；ZipCache 用 OpenAlex。Lost in the Middle 的 TACL 2024 沒有另外開頁核對（arXiv 無 jref），見「沒查到的」 |

**統計**：✅ 71　⚠️ 12　❌ 2　❓ 0（共 85 列）（另有「沒查到的」清單）

---

## 需要修改的句子

以下是建議的替換文字（只寫要換的那一格或那一句）。

**#3　02 §1.1「粒度」**
> 計算端＝vLLM token budget，預設 512；**I/O 端 128 token**〔原文 ICML §5.1 / p.5〕。但 App. B.2 又寫「依 scheduled token budget 切 chunk、把 key 反序推進佇列」〔原文 ICML App. B.2 / p.13〕，兩處說法不一致，原文沒說怎麼對齊。

**#27　02 §2.2 第三點**
> 重算成本是固定 0.2 秒，**不隨位置變**〔程式碼 UChi-JCL/CacheGen@6bed34c `run_adaptation.py:77`；在 `src/utils.py:311` 使用〕；

**#31　02 §3.1 StreamingLLM 第一點**
> 大量 attention（原文「a surprisingly large amount」）落在最前面幾個 token，和這些 token 跟任務有沒有關係無關。……

**#46　02 §3.2 SemKV 列**
> | SemKV（arXiv 2608.28911） | 不看位置；用 8 種內部指標選 25% token 拿較高一級（2.585 bit，其餘 2.322 bit） | Llama-3.1-8B：懸崖之上 8 種指標彼此沒差（這個點沒報隨機）；Mistral：8 種指標都安全，**隨機選顯著變差** Δ＝−0.0213（p<10⁻³） | 〔原文 p.2、p.14、p.17 Table 3、p.24〕 |

**#49　02 §3.3 S3**
> ✅ S3 | **和 query 相關的內容要高精度** | ZipCache、Found in the Middle 需要 query，寫入時只能當上界〔判讀〕。**SemKV 不一樣**：8 種指標裡 7 種只用 token 自己與之前的狀態，寫入時就能算〔原文 SemKV p.7〕；它在 Mistral 上比隨機好，是「寫入時可用的內容訊號」的證據。

**#50　02 §4.1 PatchKV 列「和重算／載入」格**
> 卸載時**所有** suffix block 都依凍結的 tag 量化存起來（FP16／K8V8／K8V4）；恢復時重算集 R 不讀存的那份，只有載入集 F 讀回並反量化〔原文 p.5 Eq. 12、p.6〕

**#51　02 §4.1 SparKV 列「和重算／載入」格**
> 把「Cake＋載入段用同樣 KV 量化」當 baseline（Strong Hybrid）；說位置是 chunk **成本**的差勁代理：在 sparse attention 下，同位置 chunk 的計算時間可差 3–5 倍〔原文 p.4、p.8〕。講的是成本，不是品質敏感度。

**#53　02 §4.1 HA-RAG 列**
> | | **HA-RAG**（HPCA 2026；arXiv 2510.20878） | 存取熱度：最熱 10% INT8、次 10% E4M3、再 10% E5M2、其餘 GSE-8 | 離線 | 4 層放置，**沒有重算**；有逐格式單獨的品質（Fig. 10–11），但混精度整體只和未壓縮的 TurboRAG 比，沒和均勻 INT8 並排 | 〔原文 p.7–9，B 級〕 |

**#61　02 §4.2「要誠實面對的反方證據」第一點**
> - SparKV 說位置是 chunk **成本**的差勁代理（sparse attention 下），和精度敏感度無關。SemKV 在 Llama 懸崖之上只顯示「8 種指標彼此沒差」；在 Mistral 上**隨機顯著比指標差**。所以文獻沒有直接說「隨機 25% 會很強」。R25 仍要當主要對照，因為它是「同位元組、無資訊」的基準，不是因為文獻預測它強。

**#62　02 §6 第 7 列（同一個問題）**
> | SemKV（Mistral 上隨機變差）、SparKV（只談成本） | R25 是「無資訊」基準，位置策略至少要贏它；文獻沒有預測它強或弱 | … |

**#65　02 §5.1「scale 從哪來」列**
> | scale 從哪來 | 對 `fp8`／`fp8_e4m3`（per-tensor）：只有兩種——checkpoint 的 per-tensor scale，沒有就 1.0。純 BF16 或 AWQ 權重都是 1.0。另有 `*_per_token_head`（fp8／int8／int4）在寫入時逐 (token, head) 動態算 scale，但只有 TRITON_ATTN 支援，而它在 SM89 以下拒收所有 fp8*，所以 3090 上只有 INT8 動態版能用 | 〔程式碼 vllm@v0.28.0 `attention.py:124-145、:580-592`；`kv_cache.py:82-111`；`auto_awq.py:359`；`triton_attn.py:540-549`〕 |

**#66　02 §5.1 `--calculate-kv-scales` 列**
> **已移除**（PR #49389，2026-08-03 合併）。移除前是「第一次 forward 算一次、整個 tensor 一個值、Q 和 K 是 max÷200、**V 是 max÷100**」〔文件 PR #49389；程式碼 vllm@0cf49a5 `attention.py:148-150、584-594`、`envs.py:1328-1333`〕

**#74　02 §5.2 第二點後半**
> ……只改 scale 的各種做法，第 27 層 attention 的最大相對誤差仍有 0.71–0.94；**先減掉每 (head, channel) 平均、再對殘差用每 (head, channel) scale，降到 0.031**（只減平均、scale 1.0 是 0.070）〔文件 fork issue #28〕。

**#77　01 §3 第二點**
> - MI300X（9/16，BF16 權重，要求 131,072 token；實際 prompt 長度在遺失的 CSV 裡，同設定的 Llama 是 129,024；撈針 20 題）：……

**補充建議（不是錯，但會讓證據更強）**
- #13：把 ICML p.5「We use BF16 as the default kv cache data type」加進前段 BF16 的〔判讀〕出處。
- #44：KVSink 那格補上「LLaMA2-70B、per-token static INT4」。
- #67：把 `triton_attn.py:540-549` 加進出處；它比 reshape kernel 的 assert 更早擋下所有 fp8*。
- #70：把 `quant_utils.cuh` 寫成 `vllm@v0.28.0 csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh`。
- #78：可以加一句「fp8_ptk 和 int8 在同一個 TRITON_ATTN kernel 上是 0/20 對 18/20，這組比較沒有 kernel 混淆，支持『格式』解釋」。

---

## 沒查到的

1. 02 開頭一句話「**沒有人在 FP8 量過**」sink／位置規則：這是對整個文獻的否定句，無法窮舉查證。我讀過的 16 篇裡沒看到，但 PM-KVQ 有 16→8→4→2 的漸進路徑（INT，不是 FP8），HA-RAG 有 E4M3 但依熱度。只能寫成「我們查到的文獻裡沒有」。
2. 02 §4.2「PatchKV……沒有兩頭並行」：原文提到 CUDA streams 與非阻塞 H2D（p.6），但沒有明說重算與傳輸是否重疊；我沒有讀完 PatchKV 的全部實作細節。
3. 02 §4.1 標 B 級的 MoQAE、PM-KVQ、DiffKV、KVTuner：只核對了表格裡那幾個短語，沒有全文重讀。
4. 02 §7.1 舊卡更正表：只順手核了 Cake（Table 7／§5.6／p.8、v1 的 8-bit＋CacheGen／14K／單張 A100）、CacheGen（p.6 與 p.8 的機率分布說法矛盾、Algorithm 1 看剩下全部 chunk）、ZipCache（60% 是 4-bit）。**D6_QEvict 的「正文 32、附錄 128／64／25%」與 MuKV「論文頻率公式和腳本不同」沒查**。
5. MuKV 的頁碼是 arXiv 2605.22269v1 的；CVF Open Access 版的頁碼沒核對。
6. Lost in the Middle 的 venue（TACL 2024）沒有開期刊頁核對（arXiv 頁沒有 journal-ref）。
7. 第三方 fork issue #28 的數字只核對「issue 本文怎麼寫」，沒有重現（原文也標了未重現）。
8. #72 的 q_scale 問題、#65 的 `int8_per_token_head` 在 sm_86 可用，都是讀程式碼得出，沒有在 3090 上實跑。
9. MI300X 撈針的原始 CSV 已遺失，#76 只能核到盤點檔（stdout 摘要）這一層，無法回到逐題資料。
