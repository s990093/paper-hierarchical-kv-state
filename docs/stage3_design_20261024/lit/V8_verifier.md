# V8 — 文獻與 CFP 事實查核（zero-context verifier）

- **日期**：2026-10-24
- **執行者**：zero-context verifier（Claude subagent，沒有讀本專案任何既有 docs，只看原始來源）
- **方法**：每一條都回到原始來源查。arXiv 抓 abs 頁（標題、第一作者、版本歷史）與 PDF，用 `pdftotext -layout` 轉文字後全文搜尋，再逐頁讀相關段落；USENIX FAST'26 抓 usenix.org 上的 PDF；vLLM／TensorRT-LLM 用 `gh` 讀 PR 本文與 diff，並從 raw.githubusercontent.com 與 nvidia.github.io 取官方文件；ICPE、ISPASS、HotOS 讀官方網站的 CFP 與日期頁。下載的檔案只當資料，沒有執行。原始檔暫存在 scratchpad 的 `v8_pdfs/`，不進 repo。
- **頁碼慣例**：寫「PDF p.N」的是 PDF 第 N 頁。Bidaw 的 PDF 第 1 頁是 USENIX 封面，所以另外附上 proceedings 頁碼。
- **審查性質**：這是 **self-review，不是 cross-model review**。查核者與原作者同屬 Claude 系列模型，只是沒有共用前文脈絡。
- **「沒有」類的主張**（例如「沒有任何實驗改變 GPU 負載」）：是用全文關鍵字搜尋再讀相關章節得到的結論，不是逐字通讀全文的證明。

---

## 0. 總表

| # | 主張（簡寫） | 判定 | 位置 | 修正（FIX 才填） |
|---|---|---|---|---|
| 1 | Janus 2609.36938：在 sparse 模型上預測 indexer 會讀哪些 KV，recall >96%，無損，TTFT 平均快 1.22–1.85× | OK | §4.2 PDF p.7；§5.2 PDF p.10；abs p.1 | — |
| 2 | ReadKV 2610.11245：8-bit 漸進碼；query 決定每個 K channel 讀幾 bit，attention 決定每個 V token 讀幾 bit；prefill 用 dense；Discussion 提到較慢、較大的記憶體 | OK | §3 PDF p.4；§5 PDF p.10；§7 PDF p.14–15 | — |
| 3 | Xie et al. 2503.18869：記憶體控制器把 KV 存成 bit-plane；用 Quest 排序 page；top page 讀 BF16，其次讀 FP8/FP4 prefix；排名外的 page 不讀 | **FIX** | §II-C Table II PDF p.3；§III Fig.5 PDF p.4；§IV-B PDF p.6–7 | 見 §1-3 |
| 4 | Fathom 2609.17652：4-bit K 以 channel-major bit-plane 存在 host memory；每個 query 決定每個 channel 讀幾個 plane；用來排序、選出 top-k | OK | abs；§1 PDF p.1–2 | — |
| 5 | Cascade 2608.06557：用 budget 決定每層要還原多少 bytes，其餘重算（Eq.6–7）；負載用 active decode 請求；γ 為離線固定值（Eq.3）；主要結果來自擴充的 Vidur；作者機構 | OK | §IV-B/C PDF p.6–7；§V-A PDF p.8–9；作者欄 p.1 | — |
| 6 | Bidaw（FAST'26）：第二層 KV 讀取用 HRRN 排序（Eq.1）；重建放在低優先權 CUDA stream，理由是 >30% SM 閒置 | OK | §3.2 Eq.1 PDF p.7（proc. p.106）；§4 PDF p.10（proc. p.109） | — |
| 7 | vLLM PR #42212：CPU→GPU onload 在 cuMemcpyBatchAsync 與 Triton kernel 之間二選一；門檻 n≥16、payload<28 KiB；固定 12 SMs，因為 12/16/20 的端到端 req/s 相同 | OK | PR 本文、diff `swap_blocks_triton.py`、`gpu_worker.py` | — |
| 8 | TensorRT-LLM NVFP4 cold-page：只在 Host/Disk 編成 NVFP4，GPU 端維持原精度；只支援 SM100/103/107；由記憶體壓力觸發，不看 GPU 負載 | OK | 官方 docs `features/kv-cache-compression`；`examples/kv_cache_compression/nvfp4_cold_page.md`；PR #18091 | — |
| 9 | KVServe 2605.13734（SIGCOMM'26）：選壓縮方式的 context 不含 GPU 使用率；GPU 負載漂移靠 EWMA 殘差加 ε-greedy 吸收；沒有任何實驗改變 GPU 負載 | OK | §3.1 PDF p.4；§6.2 PDF p.8–9；§7 PDF p.10–12 | — |
| 10 | NVIDIA 文件警告 Qwen2.5-7B-Instruct 用 FP8 KV 會明顯掉準確度，原因是「某些層」的 KV 值很大 | **FIX** | TensorRT Edge-LLM〈FP8 KV Cache〉頁的 Warning 區塊 | 見 §1-10 |
| 11 | 2604.08426（Bocharnikov et al.）：低秩 K（ShadowKV SVD）在資訊密集任務上「選錯」，載入約 10× token 也一樣（§4.1）；2-bit 逐 token 量化 K 做選擇勝過低秩 proxy（§4.3） | **FIX** | §4.1 PDF p.5；§4.3 PDF p.7（v5） | 見 §1-11 |
| 12 | ICPE 2027 research track：Regular 與 EERCS 兩種；只有 reproduction 寫「正、負結果都歡迎」；評審含 novelty；≤10 頁不含參考文獻；摘要 11/09、論文 11/16 AoE；Emerging ≤6 頁、2027-02-01、歡迎負面結果 | OK | icpe2027.spec.org 的 research-paper-track／important-dates／emerging-research-track 頁 | — |
| 13 | ISPASS 2027 CFP 列有主題 "Confirmations or refutations of important prior results"；摘要 2026-12-11、論文 2026-12-18 | OK | ispass.org/2027/cfp/ | — |
| 14 | HotOS XXI（2027）：論文截止 2027-02-01；≤5 頁不含參考文獻；地點 Burghausen（德國） | OK | sigops.org/s/conferences/hotos/2027/ 首頁與 cfp.html | — |

---

## 1. 逐條說明

### 1. Janus — arXiv 2609.36938 — **OK**
- **書目**：標題 *Efficient Agentic LLM Serving over SSD-based Sparse KV Storage*（系統名 Janus，見 abs 與 §1）。第一作者 Wenhao He（SJTU／Infrawaves），通訊作者 Fangcheng Fu。v1 為 2026-09-29，目前只有 v1。✔
- **證據**：§4.2（PDF p.7）寫道 "the recall metric is above 96% for all three models"。
- **機制**：把較早一層 ℓ−d 的 hidden state 餵給目標層 ℓ 自己的 indexer（Eq.2，PDF p.7），DeepSeek-V4 用 d=2，GLM-5.2 用 d=4。預測到的部分先讀（predictive reads），與計算重疊；漏掉的在 attention 執行前補讀（corrective reads）。論文明寫 lossless（PDF p.2、p.7）。模型是 DeepSeek-V4-Flash、DeepSeek-V4-Pro、GLM-5.2（§5.1，PDF p.10）。
- **數字**：§5.2（PDF p.10）是「對 Tutti 最多 1.57×（平均 1.22×）、對 LMCache 最多 3.69×（平均 1.85×）」，比的是 mean TTFT。對 Janus_base 是平均 1.27×，不在主張的範圍內。主張寫「對 Tutti/LMCache 平均 1.22–1.85×」與原文相符。
- **補充**：實驗時把 CPU DRAM 限制在 16 GB（PDF p.10）；多輪 agent 的 append prefill 是論文的明確焦點（abs）。

### 2. ReadKV — arXiv 2610.11245 — **OK**（附註）
- **書目**：標題 *Read What Matters: Query-Adaptive Quantization for KV Caches*；第一作者 Siddharth Bhandari（Amazon）；v1 為 2026-10-08。✔
- **證據**：§7 Discussion（PDF p.15 第一段）寫道 "tiered memory that keeps the less frequently fetched refinement levels in slower, larger memory"。原文前面加了「可能」，屬推測性的延伸，不是論文做過的事。
- **機制**：§3／Fig.2（PDF p.4）。第一批讀取由目前的 query 決定每個 key channel 讀多深的 prefix；把 key 重建、算出 attention 後，第二批讀取再由 attention 決定每個 value token 讀多深。用的是 Gaussian Lloyd tree 的巢狀（nested）非均勻純量碼。
- **prefill**：§5（PDF p.10）說所有方法的 prompt prefill 都用 Dense，附錄（PDF p.18、42）也有相同說明。✔
- **附註**：儲存深度 W 是參數。主要設定是 W=8（abstract：「從 8-bit cache 平均讀 4 bit」），但論文也有 W=4 的設定，例如 ReadKV (4, 2)。所以「8-bit」應理解為主要設定，不是方法的固定性質。

### 3. Xie et al. — arXiv 2503.18869 — **FIX**（範圍與措辭）
- **書目**：標題 *Reimagining Memory Access for LLM Inference: Compression-Aware Memory Controller Design*；第一作者 Rui Xie（RPI）；v1 為 2025-03-24，v3 為 2025-04-21。v1 與 v3 的 Table II 內容相同。✔
- **證據**：Table II（PDF p.3）有一列 "Dynamic Quant. (Top 5 pages in BF16, Next 3 in FP8, Next 2 in FP4)"。
- **哪些成立**：
  - §III／Fig.5（PDF p.4）：記憶體控制器把權重與 KV 拆成 bit-plane，讀取時可以只取高位 plane，例如 FP8 只取 top 8 plane。
  - Table II 比較了 Quest（Top-5 page 用 BF16，PPL 12.49）與動態精度（11.87 / 11.60）。page 大小是 16 token，模型 Llama-3.1-8B，資料集 BookSum。
- **哪些不成立或過度推論**：
  - (a) page 排序的方式，原文只寫成依 token 的相關性（relevance），並拿 Quest 當比較基準。原文沒有明說「用 Quest 排序」。
  - (b) 原文沒有明說排名外的 page 不讀。這只能從「與 Quest 的 top-5 設定相比」推論出來。
  - (c) Table II 屬於 §II-C 背景／動機段的 perplexity 實驗，不是在記憶體控制器上實作評估的機制。§IV-B 的 DRAMSim3 動態量化評估對象是**權重**（MoDE/MoE），不是 KV page。
  - (d) 文中數字與表格對不上：內文說「其次 5 頁 FP8、再 3 頁 FP4」得到 11.60，但表格把 11.60 標給「其次 5 頁 FP8」那一列。
- **建議改寫**：Xie et al. 提出在記憶體控制器中把 KV／權重拆成 bit-plane 存放，讓低精度讀取只抓高位 plane（§III）。在 §II-C 的動機實驗中（Table II，Llama-3.1-8B／BookSum，16-token page），top-5 page 保留 BF16、其後 page 降為 FP8/FP4，perplexity 優於 Quest 的 top-5 BF16（11.87／11.60 對 12.49）。論文沒有明說排序方法即 Quest、也沒有明說排名外 page 不讀；硬體端的動態量化評估（§IV-B）只針對權重。

### 4. Fathom — arXiv 2609.17652 — **OK**
- **書目**：標題 *Fathom: Per-Query Read Depth for Sparse Decoding over Offloaded KV Caches*；作者 Vivek Kalyanarangan（單一作者）；v1 為 2026-09-15，v3 為 2026-09-18。v1 與 v3 的 abstract 對這些點的描述相同。✔
- **證據**：abstract（PDF p.1）寫道 "The 4-bit K cache is stored channel-major as bit planes"。
- **對照主張**：
  - KV 與排序索引都在 host memory（abstract）。✔
  - 每個 query 用 reverse water-filling 決定每個 channel 讀幾個 plane（abstract、§1）。✔
  - 用途是先掃描、排序，再選出 top-k（§1、§2）。✔
- **附註**：沒有 QK-norm 的模型（Llama-3.1、Qwen2.5）用的是 KLT 轉換後的 plane，不是原始 channel（§3.5，見 p.2 的 contributions）。作者也寫明，索引放在 GPU HBM 時，這個方法不會更快。

### 5. Cascade — arXiv 2608.06557 — **OK**
- **書目**：標題 *Cascade: Exploiting SLO-Aware latency budget for fair and high goodput LLM inference serving*；第一作者 Muhammad Adnan；v1 為 2026-08-06。✔
- **機構**：PDF p.1 的作者欄列出 The University of British Columbia、Microsoft Azure Research、NVIDIA。✔
- **證據**：§IV-B（PDF p.6）寫道 "γ absorbs latency estimation errors, memory bus contention, and queuing variance"。原文接著說 γ 對每個模型只設一個值，離線決定，對所有 trace 與負載都不變（Eq.3）。✔
- **對照主張**：
  - §IV-B：TTFT estimator 的輸入是 context 長度、prefix 命中情形，以及 instance 的 active load，原文括號註明就是 active decode-phase requests。✔
  - §IV-C（PDF p.7）：Eq.6 是每層傳輸時間 T_k(x)；Eq.7 由剩餘 budget 反推每個 request 從 tier k 可以還原的位元組上限 M_{r,k}；超過的部分退回 GPU prefill 重算。✔
  - §V-A（PDF p.8–9）：作者擴充 Vidur 模擬器來模擬多層記憶體，所有 policy 都在擴充後的 Vidur 模擬器框架內評估；硬體 profiling 在 GB200 NVL72 上做。✔
- **附註**：Algorithm 1 的註解把 Eq. 編號寫成 (2)/(3)，與內文的 (3)/(4) 不一致。這是原文的內部筆誤，不影響判定。

### 6. Bidaw — USENIX FAST'26 — **OK**（附註）
- **書目**：標題 *Bidaw: Enhancing Key-Value Caching for Interactive LLM Serving via Bidirectional Computation–Storage Awareness*；第一作者 Shipeng Hu（清華大學）。來源 https://www.usenix.org/conference/fast26/presentation/hu-shipeng ，PDF 為 fast26-hu-shipeng.pdf。✔
- **證據**：§4（PDF p.10，proc. p.109）寫道 "we assign the transformation to a low-priority CUDA stream"。同一段的理由是觀察到超過 30% 的 GPU SM 處於閒置。✔
- **HRRN**：§3.2（PDF p.7，proc. p.106）的 disk-HRRN 公式是 Eq.1：Response ratio = 1 + 等待時間 / KV size。它只用來排序「preparing queue」裡、要對 capacity layer（SSD）發出的 KV 讀取。✔
- **附註**：
  - 這裡的「重建」，具體是把快取的 storage-efficient tensor（tensor 6，即 normalized activation）一步轉成 KV。論文寫明只對 MHA 模型有益；GQA 模型改回直接快取 KV。
  - 主張寫「~p.10」，對應的是 PDF 頁碼；proceedings 頁碼是 109。

### 7. vLLM PR #42212 — **OK**（附註）
- **書目**：標題 "[Perf] Triton fast path for small CPU→GPU `swap_blocks_batch` in the offloading connector"；作者 Itay Etelis（@Etelis）；建立於 2026-05-10，**merged 2026-06-03T10:38:17Z**。✔
- **證據**：PR 本文 "SM count" 段寫道 "E2E req/s is identical for sm12 / sm16 / sm20"，接著說取最小的 slice，因為對 decode 的爭用最少。
- **程式碼**（diff `vllm/v1/kv_offload/cpu/swap_blocks_triton.py`）：`NUM_SMS = 12`、`THRESHOLD_BYTES = 28 * 1024`、`MIN_N = 16`。n < 16 會退回 DMA（`ops.swap_blocks_batch`，也就是 cuMemcpyBatchAsync）。`gpu_worker.py` 的 `_select_swap_blocks_fn` 寫明 GPU→CPU 一律走 DMA。✔
- **附註**：
  - 選擇 kernel 的條件不只兩個：還要 Triton 可用、page size 是 8 bytes 的倍數，且 page size 的上限是在 handler 初始化時就決定的。
  - 「12 SMs」實際上是 launch `min(12, n)` 個 Triton program，不是把工作綁在特定 SM 上。
  - 程式碼註解給的理由不同：NUM_SMS 是「8–32 KB 時，頻寬在峰值 5% 以內的最小 SM 數」。端到端 req/s 相同是 PR 本文的說法。

### 8. TensorRT-LLM「NVFP4 cold-page KV Cache Compression」— **OK**
- **官方名稱與來源**：功能名是 "Cold-page quantization (NVFP4)"，屬於 KV cache compression framework；設定類別是 `ColdPageQuantizationCompressionConfig`，`algorithm: quantization_for_cold_page`。
  - 文件：https://nvidia.github.io/TensorRT-LLM/latest/features/kv-cache-compression.html （HTTP 200；原始檔在 `docs/source/features/kv-cache-compression.md`），以及 `examples/kv_cache_compression/nvfp4_cold_page.md`。
  - 主要 PR：**#18091**「Add NVFP4 as a cold-page KV Cache Compression Method」，merged 2026-09-01T20:45:25Z。後續有 #18783（DeepSeek-V4，merged 2026-09-16）與 #19391（RoPE precision，merged 2026-10-10）。
- **證據**：「Cold-Page Quantization」一節寫道 "while its Pages reside in Host or Disk memory"。同段說 GPU cache 繼續用模型原本的 runtime KV 型別（FP16/BF16/FP8）。✔
- **硬體**：footnote `general-requirements` 寫明需要 compute capability 為 SM100、SM103 或 SM107 的 GPU，example 頁也這樣寫。PR #18091 本文的說法是 SM100 家族的硬體。✔
- **觸發條件**：「When Compression Runs → Storage-Bound Methods」一節說，編碼在 page 跨層搬移時進行（offload 到 Host/Disk 時編碼，onboard 回來時解碼）；搬移由 KVCM 負責。example 頁的動機段寫「真正的 cold-tier pressure」時效益最大。兩份文件都沒有任何以 GPU 負載、使用率或爭用作為觸發的描述（關鍵字搜尋 utiliz／load／contention／idle 無結果）。
- **用語建議**：比「記憶體壓力觸發」更精確的說法是「由 KVCM 的 page 跨層搬移觸發（容量／淘汰驅動）」。

### 9. KVServe — arXiv 2605.13734 — **OK**
- **書目**：標題 *KVServe: Service-Aware KV Cache Compression for Communication-Efficient Disaggregated LLM Serving*；第一作者 Zedong Liu（UCAS／ICT CAS）；v1 為 2026-05-13；arXiv Comments 欄註明已被 SIGCOMM 2026 接受。✔
- **證據**：§6.2（PDF p.8）寫道 "GPU load and queue contention change the actual (de)compression throughput"。這被當成「離線到線上的漂移」，交給 §6.2 的 Residual-Corrected Bandit 處理：每個候選 profile 維護一個 EWMA 殘差 δ̄，再做 ε-greedy 選擇（PDF p.9）。✔
- **context 定義**：§3.1（PDF p.4）定義 c = (w, B, T_SLO, q_min)，也就是 workload、有效頻寬、SLO、品質下限，沒有 GPU 使用率。✔
- **「沒有實驗改變 GPU 負載」**：§7（PDF p.10–12）改變的是硬體等級、頻寬（5–100 Gbps、rate limiting）、資料集，以及 0–60 秒的頻寬波動（Fig.16 右）。全文搜尋 load／concurren／interfer／contention，沒有找到任何改變 GPU 負載的實驗。✔（屬「沒有」類主張，方法見開頭說明）

### 10. NVIDIA 文件：Qwen2.5-7B-Instruct 用 FP8 KV — **FIX**
- **確切頁面**：NVIDIA **TensorRT Edge-LLM** 使用手冊〈FP8 KV Cache〉，https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/features/FP8KV.html ，位於 "Model-Specific Accuracy Issues" 的 Warning 區塊。0.6.0 版起的頁面都有同樣文字（`/0.6.0/user_guide/features/FP8KV.html`）。這不是 TensorRT-LLM 主文件。
- **證據**：原文寫道 "the KV cache of the last layer has very large values"。
- **差異**：
  - NVIDIA 說原因在**最後一層**，不是「某些層」。
  - 受影響模型除了 Qwen2.5-7B-Instruct，還有 Qwen2.5-VL-7B-Instruct。
  - 頁面說這是模型本身的問題，在其他框架也會出現；詳細討論在 GitHub `NVIDIA/TensorRT-LLM` issue #4218。
- **建議改寫**：NVIDIA TensorRT Edge-LLM 的〈FP8 KV Cache〉文件警告，Qwen2.5-7B-Instruct 與 Qwen2.5-VL-7B-Instruct 使用 FP8 KV cache 會明顯掉準確度，原因是**最後一層**的 KV 值非常大、量化損失大；文件稱這是模型特性，其他框架也有，並引用 TensorRT-LLM issue #4218。

### 11. arXiv 2604.08426（Bocharnikov et al.）— **FIX**
- **書目**：標題 *KV Cache Offloading for Context-Intensive Tasks*；第一作者 Andrey Bocharnikov（HSE／Yandex）；v1 為 2026-04-09，最新 v5 為 2026-09-01。以下依 v5。✔
- **證據**：§4.1（PDF p.5）寫道 "even when we load 10× as many tokens"。
- **§4.1 的錯誤**：
  - §4.1 的題目是 **key 的壓縮**（Analyzing Key Compression & Alternatives），不是「選擇」。它說 ShadowKV 預設的 rank-160 SVD key 壓縮即使載入 10 倍 token，也追不上 full attention；原因歸於壓縮造成的 retrieval error（PDF p.5）。這組實驗的選擇仍由 ShadowKV 的 landmark 負責。
  - 論文也說這不是 SVD 的根本限制：rank 512 會好一些，只是壓縮率還不如 FP8；HIGGS-4bit 的權衡更好。
  - 論文的用語是 context-intensive tasks，全文沒有 information density 這種說法。
- **§4.3 成立**（PDF p.7）：在相同 GPU 記憶體下，2-bit HIGGS、chunk size 1（逐 token）的選擇明顯勝過 ShadowKV 的 chunk-8 landmark，也勝過 LRQK rank-32 的低秩 proxy（後者佔用的記憶體更多）。
- **版本附註**：v1 的 §4.3（"Improving KV Selection"）只和 ShadowKV 的 chunk 設定比較，**沒有 LRQK 低秩 proxy 的比較**。「勝過低秩 proxy」只在後續版本（已確認 v5 有）成立。
- **建議改寫**：§4.1：ShadowKV 預設的 rank-160 SVD **key 壓縮**在 context-intensive 任務上，即使載入 10 倍 token 也追不上 full attention（rank 512 較好，但壓縮率不如 FP8；HIGGS 4-bit 量化的權衡較佳）。§4.3（v5）：2-bit 逐 token（chunk 1）HIGGS 量化 key 做 KV 選擇，勝過 ShadowKV 的 chunk-8 landmark 與 LRQK rank-32 低秩 proxy。

### 12. ICPE 2027 — **OK**（附註）
- **來源**：https://icpe2027.spec.org/tracks-and-submissions/research-paper-track/ 、 /important-dates/ 、 /tracks-and-submissions/emerging-research-track/ 。
- **證據**：research track 頁的 EERCS 說明寫道 "Both positive and negative reproduction studies are welcome."
- **Research track**：
  - 分 Regular Research Papers 與 EERCS（Empirical / Experience / Reproduction / Case Study）兩種。✔
  - 評審標準是 novelty、relevance、technical quality、scientific soundness 與 clarity。✔
  - 篇幅 ≤10 頁（含圖表），**不含參考文獻與附錄**。✔
  - 「正負結果都歡迎」只出現在 reproduction 的語境；research track 頁沒有一般性歡迎負面結果的句子。✔
- **日期**（important-dates 頁，research track 頁本身沒列日期）：abstract 為 Monday, November 9, 2026；paper 為 Monday, November 16, 2026；全部 AoE。✔
- **Emerging Research Track**：Position 與 WIP 兩種都是 ≤6 頁（參考文獻與匿名附錄不算）；paper submission 為 Monday, February 1, 2027；明文歡迎能讓社群學到東西的負面結果。✔

### 13. ISPASS 2027 — **OK**
- **來源**：https://ispass.org/2027/cfp/ 。
- **證據**：topic 清單中有 "Confirmations or refutations of important prior results"。✔
- **日期**：Key dates 為 December 11, 2026 AoE（paper abstract）與 December 18, 2026 AoE（full paper）。✔
- **補充**：2027 年 5 月 9–11 日在紐約市舉行，由 NYU 主辦，必須親自出席報告。投稿網站尚未開放。

### 14. HotOS XXI（2027）— **OK**
- **來源**：https://www.sigops.org/s/conferences/hotos/2027/ （首頁）與 /cfp.html 。usenix.org 的 hotos27 路徑回 404。
- **證據**：cfp.html 寫道 "no longer than 5 pages including figures and tables, plus as many pages"，後接參考文獻頁數不限。✔
- **截止日**：cfp.html 寫 2027 年 2 月 1 日（週一），並註明是 hard deadline。✔
- **地點**：首頁寫 May 24–26, 2027, Burghausen, Bavaria, Germany，由 TU Munich 系統研究組主辦。✔

---

## 2. 統計

| 判定 | 數量 | 條目 |
|---|---|---|
| OK | 11 | 1, 2, 4, 5, 6, 7, 8, 9, 12, 13, 14 |
| FIX | 3 | 3, 10, 11 |
| CANNOT_VERIFY | 0 | — |

OK 之中，2、6、7、8、12 附有措辭或範圍上的附註，詳見 §1。
