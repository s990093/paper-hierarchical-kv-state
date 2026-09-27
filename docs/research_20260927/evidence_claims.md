# 2026-09-27 查詢的逐條證據（自動產生）

> 由 `deep-research` workflow（run `wf_00cb5dd9-097`，110 個子 agent）的 journal 產生。每個來源列出 fetch agent 擷取的 claim（通常 5 條）、原文引句、以及這條 claim 是否進入三票對抗式驗證與結果。
> **標記**：「三票驗證」= 3 個獨立 subagent 嘗試推翻，≥2 票推翻即否決（都是 Claude subagent，**不是 cross-model**）；「未進入驗證」= 只有擷取 agent 讀過，數字與解讀都可能錯；「抽查」= 我在 2026-09-27 用 WebFetch 重讀原始頁面，核對標題、作者、日期與關鍵句。
> **所有數字都是別人在別的硬體上量的，不是本專案的量測**（`CLAUDE.md` 規則 1）。擷取者在 claim 裡加的「對本研究：…」「推論」是擷取者的判讀，不是原文。

## S01. https://arxiv.org/abs/2605.05219

- 角度：C: Hybrid linear-attention/SSM/SWA models, tiering recurrent state vs KV
- 來源性質：primary；日期：2026-04-17

**S01.1**（central；三票驗證：支持 3／推翻 0 → 通過）

Shirokikh & Nikolenko（arXiv 2605.05219）提出 sparse prefix caching，是介於「不重用」與「dense per-token KV 重用」之間的新設計點。做法是 recurrent/SSM 層只在稀疏的 checkpoint 位置存「精確」recurrent state，命中時從最深的 checkpoint 恢復，再精確重算剩下的 suffix，輸出無損。checkpoint 放置被形式化成「給定 overlap depth 分佈的放置問題」，有精確的 O(NM) dynamic program：N 是 cached prefix 長度，M 是 checkpoint 預算。內文指出，overlap 均勻分佈時，最佳期望重算量是 N/(2(M+1))+O(1)。

> 原文：State-space models change the structure of the problem: a recurrent layer can resume from a single stored state rather than requiring the entire token history. This asymmetry opens a new design point between no reuse and dense caching: store exact recurrent states at a sparse set of checkpoint positions and, on a cache hit, resume from the deepest stored checkpoint and recompute the remaining suffix exactly. We formalize sparse prefix caching as checkpoint placement under a distribution over overlap depths, yielding an exact O(NM) dynamic program.

**S01.2**（central；三票驗證：支持 3／推翻 0 → 通過）

命中成本模型分三項：(i) checkpoint 從 CPU 載入 GPU；(ii) recurrent 層重播 suffix；(iii) attention 層載入 cached KV。實測 wall-clock 約等於「待重算 token 數的線性函數」加上載入造成的小常數偏移。也就是說，recurrent state 只放在 CPU DRAM 一層。擷取到的內文沒提到 SSD tier，也沒有精度層級（BF16/FP8/INT4）。KV 壓縮只被列為可以另外搭配的現有技術。因此「recurrent state 與 KV 在 HBM/DRAM/SSD＋精度之間的聯合 tiering」這個方向仍未被這篇涵蓋。

> 原文：The wall-clock cost of a cache hit consists of: (i) loading the stored checkpoint from CPU to GPU, (ii) replaying the suffix through the recurrent layers, and (iii) for the attention layers, loading the cached KV state. ... wall-clock time is approximately linear in the number of tokens to recompute, with a small constant offset due to cache loading

**S01.3**（supporting；未進入驗證（單一 agent 擷取））

在 QuALITY 與 System Prompts 兩組資料上（System Prompts 是真實 provider 的 system prompt 配 ShareGPT 查詢），distribution-aware 的 DP 放置在實測的 layer-group Pareto frontier 上勝過所有固定預算 baseline。和最強的 heuristic（block caching）相比，它追平或更好，而且通常用的 checkpoint 少得多。預算越低、overlap 分佈越不均勻，增益越大。擷取到的內文沒有給出相對 block caching 的具體百分比。

> 原文：Across QuALITY and System Prompts, distribution-aware placement dominates every fixed-budget baseline on the measured layer-group Pareto frontier and matches or outperforms the strongest heuristic (block caching) while typically using substantially fewer checkpoints, with the largest gains at low checkpoint budgets where the overlap distribution is most non-uniform.

**S01.4**（supporting；未進入驗證（單一 agent 擷取））

實驗規模小，也不是端到端服務量測。計時只做在 Qwen-3.5-0.8B 的一個代表性 layer group 上（1 層 full attention + 3 層 GatedDeltaNet），據內文硬體是 RTX 2080 Super 8 GB。checkpoint 是在另外的未計時 run 裡擷取的，因為 Flash Linear Attention 函式庫沒有提供擷取 non-last checkpoint 的 Python API，雖然底層 kernel 已經會在 block-64 位置回傳 state。完整模型與整機評估留待未來。所以大模型、真實 serving engine（vLLM/SGLang）上的整合與量測仍然空白。

> 原文：Wall-clock time was measured on a representative layer group (1 full attention layer and 3 GatedDeltaNet layers) of the Qwen-3.5-0.8B model. ... the current Flash Linear Attention library does not expose a Python API for non-last checkpoint extraction, although the underlying low-level kernels already return recurrent states at block-64 positions. ... We view removing this off-critical-path separation as a straightforward engineering task and defer it, together with the full-device evaluation, to future work.

**S01.5**（supporting；未進入驗證（單一 agent 擷取））

和 Marconi 的差別：Marconi 改進的是 exact recurrent cache entry 的 admission/eviction；這篇則直接改變「重用單位」，存稀疏的精確 checkpoint，讓部分 prefix overlap 也能被利用。適用範圍有限，只在很多請求共享「大量但不完全相同」的 prefix 時才有意義。內文也承認，每次都帶完整歷史、只往後追加的多輪對話，只要快取最後一個 checkpoint 就夠了。

> 原文：rather than improving the management of exact recurrent cache entries, we change the unit of reuse itself by storing sparse exact checkpoints so that partial prefix overlap becomes useful. ... The method is most relevant when many requests share a substantial but not identical prefix within a retained cache entry.

## S02. https://github.com/sgl-project/sglang/issues/40865

- 角度：C: Hybrid linear-attention/SSM/SWA models, tiering recurrent state vs KV
- 來源性質：forum；日期：2026-09-23
- ✅ 抽查：RFC，作者 curnane-lab，2026-09-23，Closed；Qwen3.8-27B hybrid（48 GDN + 16 FA）、2×910B3；pool 16 時 93% retreat（gap p50/max 32K/49K）；replay 撤回；每個 checkpoint slot 78 MB；state 重用 exact match 14/20——一致

**S02.1**（central；未進入驗證（單一 agent 擷取））

在 hybrid GDN + full-attention (FA) 模型上，SGLang UnifiedRadixCache 限制前綴重用的是 Mamba/GDN state checkpoint pool，而不是 full-KV 容量。若 full-KV 命中位置 H 背後的 state checkpoint 被驅逐，引擎會「retreat」到較早的 checkpoint C，並對 [C,H) 做全層 re-prefill，即使這段 KV 仍在快取中。以下為作者自報的 A/B 結果：150-request agentic trace，first-admission semantics，環境標為「Qwen3.8-27B hybrid, 2×910B3」；但內文又稱「Qwen3.5 interleaves GGGF (48 GDN + 16 FA)」，型號名稱前後不一致。Mamba pool=16 時 93% 請求 retreat（gap p50/max 為 32K/49K tokens，TTFT p50 49.3 s）；pool=24 為 79%，pool=32 為 4%，pool=48 為 0%。12 sessions 下，pool=32 與 pool=16 分別為 92% 與 96%。只有 full-KV 容量壓力（112K）時 retreat 為 0%，但 TTFT p50 仍升到 46.4 s（control 為 2.5 s）。

> 原文：On hybrid models, when the Mamba checkpoint behind a full-KV hit is missing or evicted, the engine retreats the whole hit to that checkpoint and re-prefills the gap through all layers — even though the gap's full-KV is already cached. ... The problem requires Mamba pool pressure, not full-KV pressure: full-KV eviction removes whole nodes (H == C by construction) and hides the gap entirely.

**S02.2**（central；未進入驗證（單一 agent 擷取））

在 interleaved hybrid 架構上，「保留已快取的 FA KV、只重建 GDN recurrent state」的 tail-replay 在經濟上不可行，作者因此撤回 replay 執行路徑（issue 已 Closed）。理由有二。(a) split-prefix extend：FA 一旦跳過 [C,H)，後面的 GDN 層就沒有 hidden states 可用；若照樣跑 FA，成本與現況相同，省下為 0。(b) ring-buffer：記錄中間態約需 0.8 MB/token（跨 48 層），覆蓋實測的 32K gap 約要 25 GB/請求，換來的只是省掉約 13 s 的 re-prefill。作者的結論是：現行全層 re-prefill 對此架構已是 compute-optimal，浪費在於 gap 存在本身，而不在修補 gap 的方式。

> 原文：Once FA skips `[C, H)`, later GDN layers have no hidden states to replay with. Running FA over `[C, H)` anyway costs the same as today — saving zero. ... Ring records ≈ 0.8 MB/token across 48 layers → covering a measured 32K gap costs ~25 GB per request, vs ~13s of re-prefill compute. Memory cost exceeds the compute it avoids by orders of magnitude. ... The corollary: today's retreat (all-layer re-prefill of `[C, P)`) is already compute-optimal for this architecture — every FLOP in it is necessary. The waste is not in _how_ the gap is repaired but in _that the gap exists at all_.

**S02.3**（central；未進入驗證（單一 agent 擷取））

作者認為 recurrent-state checkpoint 的保留政策才是「真正的槓桿」。依作者數字，此模型每個 GDN checkpoint slot 約 78 MB；每個 active session 釘住最近 K=4 個 checkpoint，100 個 session 約需 31 GB。這是用記憶體換掉約 30% re-prefill 的可調取捨。這項政策該放在 MambaComponent 的 eviction ordering，還是 scheduler-side pinning，RFC 中仍是開放問題；RFC 只詢問上游是否有興趣，沒有提出實作結果。

> 原文：Checkpoint retention policy (the real lever): keep the gap small instead of repairing it — pin the most recent K checkpoints per active session (78 MB/slot at this model size, so pinning K=4 for 100 sessions ≈ 31 GB — a tunable trade vs 30% re-prefill).

**S02.4**（supporting；未進入驗證（單一 agent 擷取））

以下數字是作者用模型估算的，並非實測。在 Mamba-pool 壓力下，gap re-prefill 時間中可歸因於「已快取 FA KV 被重算」的部分約占 30%：mb24 為 30.5%；mb16 三個 seed 為 31.5–31.7%（例如 1422 s → 973 s）；12 sessions 下 pool 32/16 分別為 30.5%/30.3%。pool=32 時只有 4.3%（314 s → 301 s）。由於作者後來論證這部分無法透過 replay 回收，這約 30% 只是理論上界，而且只在 state pool 很小時才顯著。

> 原文：mb32 | 156K | 314s → 301s | 4.3% ... mb16 (seed 0) | 4.12M | 1422s → 973s | 31.6% ... 12 sessions, pool 16 | 8.16M | 1891s → 1318s | 30.3%

**S02.5**（supporting；未進入驗證（單一 agent 擷取））

在 hybrid 模型上重用 recurrent state，輸出不保證與 fresh prefill 相同。32-token greedy continuation 的測試中，兩次清空快取的 fresh prefill 為 28/28 exact match；而 GDN 取自 donated state 的 leaf-checkpoint seed（clean full hit）只有 14/20 exact match，且在 23 tokens 內就開始分歧。instrumented state diff 顯示：只有在 seed position 與 reference 的 live position 不同時才會分歧，位置相同時則不會。作者歸因於兩點：GDN state 的數值對 chunk 分解敏感；checkpoint 的 track placement 在相同執行之間不具決定性（同一 prompt 分別得到 2816 / 2688 / none）。作者並指出，這兩點已影響現行的 clean-hit 路徑。

> 原文：Leaf-checkpoint seed | clean full hit, GDN from donated state | 14/20 | diverge ≤ 23 tok ... divergence occurs iff the seed position ≠ the reference's live position ... track placement is nondeterministic across identical runs (same prompt: 2816 / 2688 / none) ... GDN state numerics are chunk-decomposition sensitive ... Both already affect today's clean-hit path.

## S03. https://github.com/sgl-project/sglang/pull/39436

- 角度：C: Hybrid linear-attention/SSM/SWA models, tiering recurrent state vs KV
- 來源性質：primary；日期：2026-09-14

**S03.1**（central；三票驗證：支持 1／推翻 2 → **被否決，不可引用**）

SGLang HiCache 對 hybrid（attention + Mamba/KDA）模型，每個 prefill chunk 與每個完成請求各存一個 Mamba state checkpoint，作為 host KV 的還原錨點。預設依 device 端 bytes 比例把 `--hicache-size` 切給 KV 與 Mamba host pool；device Mamba pool 只按執行中的請求配置，所以很小，導致 host checkpoint slot 遠少於 KV tier 所需的錨點。某 prefix 的最後一個 checkpoint 被逐出後，它在 host 上的 KV rows 就無法還原，成為 dead memory。意涵：hybrid 模型的 KV 可重用性受另一個 pool（recurrent-state checkpoint）是否存活所約束，只看單一 pool 的容量規劃或 LRU 會系統性失效。

> 原文：Hybrid (attention + Mamba/KDA) models back up one Mamba state checkpoint per prefill chunk plus one per finished request next to the KV rows of every cached token. `build_hybrid_mamba_stack` splits the fixed `--hicache-size` budget between the KV and Mamba host pools in proportion to their _device_ bytes, and the device Mamba pool is small (sized for running requests), so the host Mamba pool ends up with far fewer checkpoint slots than the KV host tier needs anchors. Once a prefix's last checkpoint is evicted its KV rows on host cannot be restored and sit there as dead memory.

**S03.2**（central；三票驗證：支持 3／推翻 0 → 通過）

測試設定為 GLM-5.3-Flash TP4（4× RTX PRO 6000 Blackwell、`--hicache-size 32`、write_through、chunked prefill 4096）。預設切分只有 565 個 host Mamba slot，對應 2,680,896-token 的 KV host tier（coverage 82%）。灌入 200 個 8,448-token prompt 後，重放 8 個已被逐出的 prompt：0/8 host hit，TTFT 0.82 s（等同 cold）；當時 Mamba pool 是 565/565 全滿，KV pool 卻還有 35% 是空的。改用 `--hicache-mamba-size-gb 14`（734 slots）後：8/8 host hit，TTFT 0.15 s，從 host 還原 8,192 tokens。推論：8,192 < 8,448，還原停在與 chunk 對齊的 checkpoint 邊界，尾段仍要重算，和我們的「load front, recompute tail」一致。限制：只有單一設定、8 次重放，未經同儕審查。

> 原文：Reproducible on GLM-5.3-Flash TP4 (4x RTX PRO 6000 Blackwell, `--enable-hierarchical-cache --hicache-size 32 --hicache-write-policy write_through`, chunked prefill 4096): the default split gives 565 host Mamba slots against a 2,680,896-token KV host tier (coverage 82%). Fill 200 prompts of 8,448 tokens (3 checkpoints each), then replay 8 evicted ones: 0/8 host hits, replay TTFT 0.82 s (cold), with the Mamba pool at 565/565 and the KV pool 35% empty. The same cell with `--hicache-mamba-size-gb 14` (734 slots): 8/8 host hits, replay 0.15 s, `cached_tokens` 8,192 from host.

**S03.3**（supporting；未進入驗證（單一 agent 擷取））

PR 的 `auto` 模式用啟發式公式 `slots = ceil(kv_host_tokens / chunked_prefill_size) + 4 x max_running_requests`（two-pass fixed point）決定 checkpoint slot 數，所以 Mamba 的容量需求隨 host 快取 token 數線性成長。在 32 GB 預算下切成 KV 19.12 GB / Mamba 12.88 GB（由引文推算：約 40% 預算給 recurrent-state checkpoint），676 slots，coverage 109%。灌入 36 個 65,536-token prompt 後重放 8 次：8/8 host hit，TTFT 0.42 s，cold 則是 5.98 s（推算：host 還原比重算快約 14 倍，是 hybrid 模型在 Blackwell 上一個類似 κ 的資料點）。意涵：在 host tier 上，recurrent state 不是「常數大小、可以忽略」，而是和 KV 搶容量的一級消費者；而且這個切分只是啟發式，沒有 MRC 或成本模型的依據。

> 原文：`auto` solves `slots = ceil(kv_host_tokens / chunked_prefill_size) + 4 x max_running_requests` as a two-pass fixed point with the KV share reduced accordingly. ... With `auto` the split is KV 19.12 GB / Mamba 12.88 GB, 676 slots, coverage 109%, and a 36 x 65,536-token fill + 8 replays gives 8/8 host hits at 0.42 s vs 5.98 s cold.

**S03.4**（supporting；未進入驗證（單一 agent 擷取））

在這個 PR 之前，SGLang 的 `/metrics` 只有讀取單一 host pool 的 anchor gauges（`sglang:hicache_host_used_tokens` / `_total_tokens`），Mamba pool 的佔用與逐出完全看不到。PR 新增三個 per-pool 指標：`sglang:hicache_host_pool_used_tokens{pool}`、`sglang:hicache_host_pool_total_tokens{pool}`、`sglang:hicache_host_pool_evicted_tokens_total{pool}`。它也明確定義：在 mamba 壓力下被計入逐出的 KV，就是連同 checkpoint 一起遺失的 host prefix。意涵：生產級多 pool KV tiering 連跨 pool 的連帶逐出（orphaned/cascading eviction）都沒有遙測可看，量測方法本身就是缺口（對應方向 E）。

> 原文：Nothing on `/metrics` showed this: the anchor gauges `sglang:hicache_host_used_tokens` / `_total_tokens` read one host pool, and the Mamba pool's occupancy and its evictions were invisible. ... KV counted under mamba pressure is a host prefix lost with its checkpoints.

**S03.5**（tangential；未進入驗證（單一 agent 擷取））

此 PR 於 2026-09-14 開啟，擷取時仍是 Open、尚未合併。SGLang 對 hybrid 模型 host tier 的 KV/Mamba 切分方式還沒有定案：另一個 PR #29037 提議在設定 `--hicache-size` 時按比例放大 Mamba host pool；本 PR 則保留預設的 byte-proportional 切分，只加上手動旋鈕與指標。意涵（novelty 風險評估）：生產系統目前以啟發式或手動旋鈕處理「KV 與 recurrent-state checkpoint 的聯合 tiering 與逐出」，但社群正積極修補，搶先發表的時間窗有限。

> 原文：Related: [#29037](https://github.com/sgl-project/sglang/pull/29037) proposes scaling the Mamba host pool proportionally when `--hicache-size` is set; this PR keeps the default split and adds an explicit knob plus the metrics to see when it is needed.

## S04. https://llm-d.ai/blog/serving-hybrid-models-at-scale-in-llm-d

- 角度：C: Hybrid linear-attention/SSM/SWA models, tiering recurrent state vs KV
- 來源性質：blog；日期：2026-06-13
- ✅ 抽查：Toledo/Ozeri/Harnik/Etelis/Brill/Ayoub，2026-06-13；「Without HMA awareness, an offloading connector turns the HMA off」、HMA-aware 讀回快 1.8–1.9×（H100、gpt-oss、IBM Storage Scale）——一致

**S04.1**（central；未進入驗證（單一 agent 擷取））

In vLLM, an offloading connector that is not aware of the Hybrid Memory Allocator (HMA) turns HMA off for hybrid models such as gpt-oss (full-attention plus sliding-window layers). Offloading still works, but the cache is managed as uniform full-attention blocks. That throws away the GPU-memory savings of the small-footprint layers and makes every KV load move more bytes than needed. The post calls this a silent slowdown, not an error. [Relevance, our inference: this is another case where the software path, not the hardware, sets transfer cost. Tiering measurements on hybrid models taken through a connector that is not HMA-aware would silently overstate bytes moved and understate GPU capacity.]

> 原文：Prior to our changes, an offloading connector built for the uniform world handles a hybrid cache in one of two bad ways: Either forgo offloading altogether or forgo the use of HMA. The second is not a failure but a silent slowdown: vLLM falls back to managing the cache as uniform attention blocks, so offloading still succeeds, but the GPU memory savings are lost, and every KV load ends up moving more data than it should.

**S04.2**（central；未進入驗證（單一 agent 擷取））

HMA-aware offloading makes batch KV reload 1.8-1.9x faster on both the CPU and storage tiers. The mechanisms are per-layer-group tensor subsets, a separate offload file per group per chunk (two files per chunk for gpt-oss, moved in parallel by the I/O pool), and partial transfers that load only the in-window sliding-window blocks, often starting at an offset inside a file. Setup: gpt-oss-20b (TP=2) and gpt-oss-120b (TP=4) on H100 with IBM Storage Scale; 10 concurrent 128K-token prompts; GPU cache disabled on the hot pass; 16-token blocks on the CPU tier and 256-token blocks on the storage tier. The whole gain comes from sliding-window layers loading only the last 128 tokens. The write side saw neither benefit nor penalty because all KV data was still offloaded. [Relevance, our inference: under this design the hybrid architecture does not reduce SSD write volume, which leaves a write-budget-aware knob open. Per-group files also multiply transfers per chunk, which could eat into the byte savings on a descriptor-cost-bound path such as the one we measured. Untested.]

> 原文：The results (in Figure 2) show that HMA-aware reads are 1.8x to 1.9x faster on both tiers and both model sizes. The source of these benefits is in the sliding window layers that only need to load the last 128 tokens worth of KV cache instead of the full attention for the entire 128K tokens. On the offload side (after initial prefill) we did not see this benefit, nor any penalty, as we still chose to offload all the KV data.

**S04.3**（supporting；未進入驗證（單一 agent 擷取））

For gpt-oss-120b, vLLM's own log messages predict 1.77x more KV cache capacity with HMA on. This is a logged prediction, not a measured capacity. The cause is that sliding-window layers keep only the last 128 tokens, while full-attention layers keep every token (all 128K for a 128K request). Setup of the user-scalability sweep: previously seen 16K-token prompts, single-token decode, max 80 QPS and 80 concurrency, gpt-oss-120b TP=4 on H100. In that sweep, HMA moves the GPU and CPU tiers' saturation cliffs to higher user counts and raises the throughput plateau with storage added. Storage was large enough that throughput never collapsed to prefill level in the tested range. [Relevance, our inference: because footprint differs by layer, the capacity cliff moves. Capacity-cliff arithmetic and any a-priori 'is tiering worth it' criterion must be computed per layer group.]

> 原文：For example, for a gpt-oss-120b model, vLLM predicts an increase of 1.77x in KV cache capacity when turning the HMA configuration flag on (as seen in vLLM's log messages). This is again due to the smaller KV footprint of the sliding-window layers.

**S04.4**（supporting；未進入驗證（單一 agent 擷取））

Setup: a 16-server gpt-oss-120b fleet (TP=1, one H100 per server); 250 prefix groups x 5 prompts; 16K-token system prompt, 256-token question, 256-token output; 5-40 QPS. Results: the baseline that routes on GPU cache only saturates past about 20 QPS, with TTFT in the tens of seconds. Adding a CPU offload tier while still routing on GPU state raises throughput about 75% at 40 QPS. Routing that scores both the GPU and CPU tiers raises throughput about 115%, with TTFT essentially flat. This routing uses KV-events extended to the CPU tier that carry HMA-group information, so full-attention layers count as resident after the sliding-window layers roll out. [Relevance and caveat, our inference: placement across replicas (routing) may hold headroom that placement within a single instance does not. However, the workload is heavily prefix-shared and the baseline has no offload tier, so only about 40 percentage points are attributable to routing awareness. This is a vendor benchmark, not independently reproduced.]

> 原文：Past about 20 QPS the GPU-only baseline saturates and its TTFT climbs into the tens of seconds as requests queue. Adding a CPU offload tier helps: at 40 QPS it lifts throughput about 75% over the baseline while holding TTFT low. But the real win comes from managing both tiers together. With GPU+CPU KV management, throughput rises roughly 115% over the baseline at 40 QPS, and TTFT stays essentially flat across the whole sweep. [...] Capacity is a property of an instance; throughput is a property of placement.

**S04.5**（central；未進入驗證（單一 agent 擷取））

The post leaves recurrent-state tiering and multi-window models unevaluated or unsolved. (1) For Mamba and linear-attention layers (Jamba, Qwen3.5) it says only that checkpoints of intermediate states are maintained and offloaded. It gives no checkpoint interval, state sizes or measurements, and every experiment uses gpt-oss (full attention + 128-token sliding window) only. (2) Routing for purely sliding-window models, or models mixing several window sizes, needs a dedicated scorer that is still in progress. (3) It says HMA 'is not a solved problem', because each new attention type is another cache group; DeepSeek V4 is cited as a 'five-way cache stack'. (4) Its HMA results rely on vLLM PR #37885, which is not merged. The same HMA-aware path also exists in a new multi-tier connector upstreamed to vLLM, where CPU acts as a hub for hot data and staging for storage. [Relevance, our inference: characterizing how recurrent-state checkpoints should be tiered versus KV is an open gap with production motivation. The tradeoff is checkpoint interval vs prefix-hit granularity vs write volume vs precision.]

> 原文：Mamba and linear-attention layers (Jamba, Qwen3.5) hold a single fixed-size state instead of per-token KV. In such models, rather than keeping a KV stream, checkpoints of intermediate states are maintained and offloaded. [...] Models that are purely sliding-window, or that mix several window sizes, are the harder case, and a dedicated scorer to handle them optimally is in progress

## S05. https://arxiv.org/abs/2608.30386

- 角度：C: Hybrid linear-attention/SSM/SWA models, tiering recurrent state vs KV
- 來源性質：primary；日期：2026-08-31

**S05.1**（central；三票驗證：支持 3／推翻 0 → 通過）

Hybrid linear-attention 模型（Qwen3-Next-80B 的 Gated DeltaNet、Kimi-Linear-48B 的 KDA）做 prefix caching 時，需要一種 KV 以外的新快取物件。recurrent state 是就地覆寫的，不保留較早的 prefix 邊界，所以 SGLang 每隔固定 token 間隔就要物化一份涵蓋所有 linear-attention 層的完整 state checkpoint，與 full-attention KV block 並存。沒快取到的 checkpoint 只能靠 prefill 重算。checkpoint 間隔因此是「HBM 佔用 vs. replay／重複 prefill」的取捨，性質和 KV block 不同：單一大物件，而且只在邊界上有效。依附錄 C.1 的 per-rank 預算推算（Kimi-KDA 694,681,600 B、Qwen-GDN 1,236,271,104 B，各等於 128 個 dense slot），一份 dense checkpoint 在 TP8 下每個 rank 約 5.43 MB（KDA）或 9.66 MB（GDN）。這是我的除法推算，論文沒有直接寫出這兩個數字。

> 原文：Unlike full-attention KV, a recurrent state is overwritten as tokens arrive and does not preserve earlier prefix boundaries. Prefix caching therefore materializes full state checkpoints at regular token intervals alongside KV blocks (Zheng et al., 2024; Dao, 2026). Because each state checkpoint contains large state matrices for all linear-attention layers, frequent state checkpointing quickly exhausts HBM, while sparse state checkpointing increases replay or repeated prefill. [...] The per-rank budgets are 694,681,600 bytes for Kimi-KDA and 1,236,271,104 bytes for Qwen-GDN, each equal to 128 dense physical slots and therefore 127 deployable dense slots.

**S05.2**（central；三票驗證：支持 3／推翻 0 → 通過）

DASC 的範圍只到 HBM。所有 serving 實驗都是「matched-HBM」設定：固定 per-rank state-checkpoint 預算、LRU 淘汰、SGLang、TP8、Hopper GPU。它把 GPU↔host memory 的階層式 KV 系統（CachedAttention、RAGCache）列為「互補」，full-attention KV 完全不動。我對 HTML 全文做檢索，offload、CPU、SSD、DRAM 皆 0 次命中。它把 hybrid-state 快取的前作分成三條軸：Kimi K3 持久化 KDA state、Marconi 做 admission/eviction、ReplaySSM 快取 SSM 輸入並按需重建 state；DASC 自己改的是第四軸，也就是每份 checkpoint 的表示大小。所以「把 recurrent-state checkpoint 和 KV 一起分層放到 HBM/DRAM/SSD，並決定何時重算」這件事，此文完全沒有處理。

> 原文：Hierarchical systems extend reusable KV across GPU and host memory (Gao et al., 2024; Jin et al., 2024); quantization, eviction, and sparsification further reduce token-indexed KV storage [...]. These methods are complementary to DASC, which leaves full-attention KV unchanged and compresses the recurrent state checkpoint stored alongside it. [...] Kimi K3 persists KDA states for prefix reuse (Kimi Team, 2026); Marconi instead improves which hybrid prefix entries are admitted and evicted based on reuse and compute–memory utility (Pan et al., 2025). ReplaySSM caches recent SSM inputs and reconstructs states on demand, targeting decode-time state traffic and rollback (Dao, 2026). DASC changes a different axis—the representation size of each persistent boundary state checkpoint [...] They use Route A, ragged state checkpoint storage in the head-aware extra_buffer pool, LRU eviction, mem-fraction-static=0.80, the overlap scheduler disabled, and cache telemetry enabled.

**S05.3**（supporting；未進入驗證（單一 agent 擷取））

在固定的 state-checkpoint HBM 預算下，壓縮倍率取決於架構的 decay 粒度：channel 粒度的 KDA 在 W_max=16–512 間達 2.63–28.04×，head 粒度的 GDN 只有 1.10–2.48×。KDA 在 W_max=16 時，mean TTFT 從 567.6 ms 降到 326.0 ms（−42.6%），input throughput 從 33.25 升到 55.98 k tok/s（+68.4%）。GDN 在 W_max=512 時，TTFT 從 614.4 ms 降到 374.5 ms。要注意負載：這是從 12,032 個 MMLU-Pro 短 prompt 衍生的受控 reuse-distance 負載（concurrency 96、chunked-prefill 8192、每請求只輸出 1 token），不是長上下文負載。所以收益是在「state 佔滿 HBM」的短 prompt 高併發情境下量到的。

> 原文：With TP-balanced placement, channel-wise KDA achieves 2.63–28.04× compression across W_max=16–512, while head-wise GDN achieves 1.10–2.48×. [...] At W_max=16, DASC-nr reduces TTFT from 567.6 to 326.0 ms and raises throughput from 33.25 to 55.98 k tokens/s. GDN shows a similar trend: at W_max=512, DASC-nr reduces TTFT from 614.4 to 374.5 ms and improves throughput from 30.56 to 50.96 k tokens/s. [...] The controlled reuse-distance workload is derived from all 12,032 MMLU-Pro test prompts [...] All arms use concurrency 96, chunked-prefill-size=8192, one deterministic output token

**S05.4**（central；三票驗證：支持 1／推翻 2 → **被否決，不可引用**）

state 與 KV 的品質敏感度很不對稱。在 RULER NIAH-S1 上，把 full-attention KV 歸零後準確率變成 0；把 recurrent-state checkpoint 歸零，準確率最多只差 1 點（GDN 0.990、KDA 1.000）。不過這只是單一 needle 的檢索。在較難的任務上，zero-fill（dasc-nr）隨 W_max 增大會掉分，例如 MMLU-Pro 從 66.94% 掉到 60.17%。「只丟部分 state、再從有界 suffix 重算」的 dasc-wr 則維持在 66.59–67.19%（Dense 為 67.50%）。W_max=128 時，它用 +23.0 ms 的 TTFT 換回 4.43 點。附錄 Table 8 給出每次 hit 的攤提重算成本：重放 16 token 約 67.0 ms，重放 841 token 約 426.0 ms。這個數字包含 concurrency 96 下的排隊。這些結果支持兩件事：state 和 KV 應各有不同的品質預算 ε，而且「部分丟棄＋有界重算」可以成為動作空間裡的新動作。

> 原文：Removing full-attention KV reduces accuracy to zero, whereas removing recurrent state checkpoints changes it by at most one point (Table 1 (a)). [...] DASC-nr decreases from 66.94% to 60.17% as W_max grows, whereas DASC-wr remains at 66.59–67.19%, close to the 67.50% Dense baseline. [...] At W_max=128, DASC-wr recovers 4.43 points for 23.0 ms of additional TTFT. At W_max=512, it recovers 7.02 points and remains within 0.31 points of Dense while retaining 25.4% lower TTFT.

**S05.5**（supporting；未進入驗證（單一 agent 擷取））

recurrent state 的精度階梯和 KV 不同。SGLang 預設用 FP32 存 temporal state、BF16 存 convolution state，比 KV 的 BF16 還寬。論文只把 cached temporal state 量化成對稱 INT8（計算用的 active state 仍是 FP32），再與 DASC W_max=16 的 channel 選擇疊加，得到 8.11× 的 state-checkpoint 容量。五項 Kimi-KDA end task 與 dense 相近：AIME 0.6073→0.6219、MMLU-Pro 0.6750→0.6748、GPQA 0.6187→0.6130、IMO 0.2787→0.2781。這表示「位元數」（量化）和「保留哪些 unit」（DASC）是兩個正交的壓縮維度。state 的量化空間（FP32→INT8）比 KV 的 BF16→INT4 更大，但論文沒有測 INT4/FP8 state，也沒有測跨 tier 搬移。

> 原文：Recurrent state checkpoints use SGLang's default mixed precision: the temporal state is FP32 and the convolution state is BF16 [...] Quantization reduces the bits per retained value, whereas DASC reduces the number of retained units, making the two complementary. We combine W_max=16 channel selection, INT8 state checkpoint storage, and DASC-nr on the five Kimi-KDA end tasks, yielding 8.11× state checkpoint capacity relative to the dense mixed-precision reference.

## S06. https://arxiv.org/abs/2604.03143

- 角度：A: Cross-model, cross-adapter and cross-agent KV reuse (non-prefix)
- 來源性質：primary；日期：2026-04-03

**S06.1**（central；三票驗證：支持 1／推翻 2 → **被否決，不可引用**）

Multi-agent All-Gather 回合會造成大量跨 agent 的 KV 冗餘。8-agent GenerativeAgents 回合中，pairwise block similarity 為 91%–97%。在 Qwen2.5-14B、單張 A100 80GB 上，multi-agent workload 的 KV 佔 41.5 GiB（pool 的 99.3%），相同條件下的獨立請求只佔 24.8 GiB（59.2%）。這些共享輸出區塊在各 agent prompt 裡不是共同 prefix，要靠 RoPE re-rotation 對齊位置，所以 prefix-only 命中語意沒辦法去重。→ 對本研究：這是 agentic workload 上的直接量化證據，支持 non-prefix／段級重用（我方 finding 4）。

> 原文：in an 8-agent GenerativeAgents round, pairwise block similarity ranges from 91% to 97% ... The multi-agent workload consumes 41.5 GiB of KV Cache storage—99.3% of the pool—while independent requests use only 24.8 GiB (59.2%).

**S06.2**（central；三票驗證：支持 1／推翻 2 → **被否決，不可引用**）

Diff-Aware Storage 只保留一份 dense master copy。其餘 sibling agent 的 KV 存成 block-sparse K/V diff（block index 加上 block 級修正值），在 layerwise 傳輸路徑上做 fused restore，不會實體化 dense 副本。在代表性 workload 上壓縮率為 11–17×，per-agent KV 最多降 17.5×。作者自己承認壓縮率取決於 workload 結構：各 agent 收到的共享輸出子集差異越大，correction 就越大，收益也隨之遞減。

> 原文：Its Diff-Aware Storage encodes sibling caches as block-sparse diffs against a single master copy, achieving 11-17x compression on representative workloads. ... If requests diverge more strongly—for example, in a system where each agent receives a substantially different subset of the shared outputs—then the Mirror corrections grow larger and the storage benefit diminishes.

**S06.3**（supporting；未進入驗證（單一 agent 擷取））

KV Collector 把 non-prefix reuse 的固定開銷（RoPE re-rotation，加上用 key-difference 分析挑出要重算的位置，後端沿用 CacheBlend 方法）改成每個回合集體做一次，因此共享 block 的 reuse 成本與 agent 數無關。在 GenerativeAgents 與 AgentSociety 上、1500 ms round-latency SLO 下，它比 vLLM prefix caching 最多多撐 2.7× 並行 agents，prefill 比 per-request position-independent caching 最多快 1.9×。

> 原文：TokenDance's KV Collector performs KV Cache reuse over the full round in one collective step, so the cost of reusing a shared block is paid once regardless of agent count. ... Evaluation on GenerativeAgents and AgentSociety shows that TokenDance supports up to 2.7x more concurrent agents than vLLM with prefix caching under SLO requirement, reduces per-agent KV Cache storage by up to 17.5x, and achieves up to 1.9x prefill speedup over per-request position-independent caching.

**S06.4**（supporting；未進入驗證（單一 agent 擷取））

近似的 non-prefix 重用（CacheBlend 式 PIC）在多回合 agent 模擬中不保證輸出一致。與 vLLM prefix caching 逐回合比對（temperature=0、輸入相同），8 個 GenerativeAgents 場景中只有 3 個整條 trace 完全一致，其餘 5 個的 round 數差 3.3%–11.9%。作者把原因歸於 PIC 在選擇性重算位置造成的數值擾動：這些擾動在 greedy decoding 下會翻轉單一 token，接著跨回合級聯。→ 對本研究：在 agentic 多回合情境下，品質誤差 ε 會跨回合累積，只看單回合 accuracy 不夠。quality-aware 的放置／重用應該用 trajectory 級指標來量。

> 原文：To verify this, we run both TokenDance and vLLM with prefix caching on the same set of agent rounds, setting temperature to 0 ... In three of the eight scenarios (Meet and Greet, Valentine's Day Party, and Information Outbreak), both systems produce identical output for the entire trace (Δ=0.0%). In the remaining five scenarios, the round counts differ by 3.3%–11.9%. ... under greedy decoding, such perturbations can eventually flip a single token choice, which then cascades through subsequent rounds.

**S06.5**（supporting；未進入驗證（單一 agent 擷取））

評估範圍很窄：只用單張 NVIDIA A100 80GB 與 Qwen2.5-7B/14B，實作是包在 LMCache storage backend 外層的 diff-aware backend（vLLM + LMCache）。TokenDance 的設計本身不含 KV 量化（KIVI／KVQuant 只出現在 related work），也不含 CPU／SSD offloading（CacheGen、Mooncake 只在 related work 提到）。→ 對本研究的空白（我方推論）：有兩件事它都沒有探討，一是 diff-encoded KV 跨 HBM／DRAM／SSD 的階層放置，例如 master 放 HBM、diff 放低階；二是 block-sparse diff 與 FP8／INT4 精度階的組合。另外，它用的模型正好是 Qwen2.5 家族，也就是我方量到在 FP8／INT4 下崩潰的那一族。

> 原文：We evaluate TokenDance on NVIDIA A100 80GB GPU. We use two models, Qwen2.5-7B and Qwen2.5-14B ... We add a diff-aware backend that wraps the normal LMCache storage backend.

## S07. https://arxiv.org/abs/2608.30310

- 角度：C: Hybrid linear-attention/SSM/SWA models, tiering recurrent state vs KV
- 來源性質：primary；日期：2026-08-31

**S07.1**（central；三票驗證：支持 0／推翻 3 → **被否決，不可引用**）

Tail-Replay（TeleAI／上海交大，ML for Systems @ NeurIPS 2026 workshop 論文）處理 hybrid 模型（full-attention 層與 Gated DeltaNet linear-attention 層交錯）的 prefix cache 時完全不存 recurrent-state checkpoint。它只保留精確的 FA KV。命中 m 個 token 時，它把 linear-attention 狀態設為零，再重播 matched prefix 最後 k=⌈r·m⌉ 個 token，近似重建該狀態。因此重用邊界由共享 token 決定，不必對齊 checkpoint 位置（對照 Marconi 與 Sparse Prefix Caching）。對方向 C 的意涵：hybrid 模型的 recurrent state 多了一個「DROP 加上 replay 預算 r 可調的有損近似重算」動作，選項不再只有「存或不存精確 checkpoint」。

> 原文：Tail-Replay exploits this property by caching the exact full-attention key-value cache while omitting recurrent-state checkpoints. On a cache hit, it reconstructs the linear-attention states by replaying a short, recent suffix of the matched prefix. As a result, the reuse boundary is determined by the shared tokens rather than by recurrent-state checkpoints.

**S07.2**（central；三票驗證：支持 3／推翻 0 → 通過）

近似重建造成的品質損失明顯依模型與任務而異，與我們「ε 由模型×任務決定」的發現一致。r=5% 時，LongBench 保留 full prefill 的 92.8%（OLMo-Hybrid-7B）到 98.9%（Qwen3.5-4B），RULER 保留 93.1–99.9%。r=10% 時，LongBench 保留 93.9–98.1%，RULER 保留 96.7–99.9%。r 加大不一定更好：Qwen3.5-4B 的 LongBench 從 98.9% 降到 98.1%。最差的單一格是 OLMo-Hybrid-7B 的 LongBench qasper@16K，從 .394 降到 .308（約 78%）。對照組 zero-only 只重用 FA KV、把 recurrent state 歸零，其 RULER 平均掉到 0.215–0.415（full prefill 為 0.812–0.987），可見 recurrent state 不能直接丟。評測範圍窄：RULER 只測 8K／16K，LongBench 多數只測 16K（narrativeqa 最長 64K），平台是 H100 上的 PyTorch 2.9.1 原型。

> 原文：We evaluate three Gated DeltaNet-based hybrid LLMs—OLMo-Hybrid-7B, Qwen3.5-4B, and Qwen3.6-27B—on NVIDIA H100 GPUs using PyTorch 2.9.1. ... Tail-Replay retains 92.8–98.9% of full-prefill quality on LongBench at r=5% and 93.9–98.1% at r=10%; on RULER, it retains 93.1–99.9% and 96.7–99.9%, respectively.

**S07.3**（supporting；未進入驗證（單一 agent 擷取））

快取的 FA KV 放在 host（CPU）記憶體。H2D 傳輸在獨立的 copy stream 上與 replay 同時進行，只在 query forward 前同步一次；再加上 replay 時跳過每個 group 最後一層的 FFN，32K 時 TTFT 比序列化 H2D 再少 18–42%。matched prefix 約 32K 時的 TTFT 加速：OLMo-Hybrid-7B 9.8×（1068.3→108.8 ms）、Qwen3.5-4B 9.1×（786.4→86.2 ms）、Qwen3.6-27B 14.3×（3605.2→251.8 ms）。replay 預算從 5% 提高到 10% 時，32K 的 TTFT 上升（Qwen3.6-27B 從 251.8 升到 371.3 ms），此時 replay 成為主要成本。TTFT 從「開始傳輸或 replay」起算，取 20 次平均；依此定義，應不含排隊、查找，也不含從 SSD 取回。以下是推論，不是論文原文：命中路徑的 TTFT 從 8K 到 16K 幾乎不變（例如 Qwen3.5-4B 75.2→75.0 ms），代表固定開銷占主導。這與我們「傳輸成本由軟體路徑決定」的發現相容；本文也等於把「載入」與「重算」從二選一改成兩者重疊。

> 原文：Relative to serialized H2D transfer, OVL+skip further reduces TTFT by 18–42% at 32K. Increasing the replay budget to 10% has little effect at shorter contexts but raises TTFT at 32K, where replay becomes the dominant cost.

**S07.4**（supporting；未進入驗證（單一 agent 擷取））

快取物件仍以 token 為單位定址，因此可以像一般 KV 一樣分塊搬到 CPU 或 SSD。每個 token 存 FA 層的 K/V，另外每個 FA 層再存一個 output hidden。模型切成多個 group，每個 group 由一個 FA 層和其後連續的 linear-attention 層組成；各 group 獨立 replay，誤差只留在 group 內。代價是每個 token、每個 FA 層都多存一個 hidden 向量。但全文沒有任何以 bytes 或容量計的比較（'memory' 一詞只出現在 'host memory'），沒有 CPU 以外的 tier，沒有容量受限下的淘汰策略，也沒有整合進 vLLM 或 SGLang（vLLM 在全文出現 0 次）。這正是一篇 tiering 論文可以切入的空缺：在容量限制下，於「recurrent checkpoint」、「FA hidden 加 replay」與「完整重算」之間做放置決策。

> 原文：Specifically, we cache the FA output hidden of each FA layer and partition the hybrid architecture into groups, each comprising one FA layer and the consecutive linear-attention layers that follow it. This ensures that the input hidden of the first linear-attention layer in every group exactly matches its value during the original prefill, thereby confining replay error within each group.

**S07.5**（supporting；未進入驗證（單一 agent 擷取））

本文給出 hybrid 模型 prefix 與 position-independent caching 的相關工作地圖。Marconi（MLSys 2025）決定跨 prefix 保留哪些 recurrent state。Sparse Prefix Caching（Shirokikh & Nikolenko，arXiv 2605.05219）決定 checkpoint 放在 prefix 內的哪些位置。LinearKV（arXiv 2608.11231）拿快取的 local linear state 當 PIC 的初始值；它的作者與本文重疊，同屬 TeleAI 團隊。另外引用了 HYPIC（arXiv 2607.01299，hybrid-attention 的 PIC）、ProphetKV（arXiv 2602.02579）與 EPIC（ICML 2025）。novelty 風險：同一團隊在一個月內發了兩篇，方向 C（hybrid state 快取）競爭活躍。本文與以上各篇都沒有處理多層 tier 的放置問題（依本文的相關工作描述判斷）。

> 原文：For hybrid LLMs, Marconi [15] and Sparse Prefix Caching [20] retain recurrent checkpoints, while LinearKV [10] uses cached local linear states as initializers; Tail-Replay instead reconstructs the matched-prefix state from a short recent hidden suffix without storing recurrent checkpoints.

## S08. https://arxiv.org/abs/2605.24022

- 角度：A: Cross-model, cross-adapter and cross-agent KV reuse (non-prefix)
- 來源性質：primary；日期：2026-05-20

**S08.1**（central；三票驗證：支持 0／推翻 3 → **被否決，不可引用**）

CacheTune 用逐層成本模型決定非前綴（non-prefix）KV 重用時「載入多少、重算多少」：T(r)=max(rN·tc,(1−r)N·ti)+to，解析最佳點 r0=ti/(tc+ti)，等同 r0=1/(1+κ)，其中 κ=tc/ti 是重算對傳輸的成本比。之後再用 10 筆 SAMSum 樣本做 golden-section search 微調，每個硬體或儲存設定約需 3–4 分鐘。在「用 κ 驅動 DROP+重算決策、並隨硬體調整」這件事上，這是目前最接近的先前工作，對 Tiara 的 κ 主張構成 novelty risk。與 Tiara 的差異有三：傳輸成本只是每 token 線性項加每層固定開銷 to，沒有 per-descriptor 的軟體開銷項；決策只是單一比例 r，不是在多個 tier 之間放置；依擷取到的全文，沒有精度維度（BF16/FP8/INT4）。

> 原文：the per-layer latency is bounded by the slower path: T^(ℓ)(r)=max(rN·tc,(1−r)N·ti)+to ... The crossover point at which the transfer time equals the recomputation time yields the first-order analytical optimum: r₀=ti/(tc+ti).

**S08.2**（central；三票驗證：支持 1／推翻 2 → **被否決，不可引用**）

最佳重算比取決於 KV 放在哪一層儲存，而且儲存越慢，重算比應越高。固定重算比 15% 時，HDD 與 SSD 相對 Full Recompute 的 TTFT 加速平均只有 1.92× 與 2.05×。改用硬體感知校準後，HDD 選出 36.4%、SSD 選出 30.9%，加速提升到 2.36× 與 2.34×。論文明言慢速儲存上「不應盲目最大化 KV 重用」，這和 Tiara 動作空間的 DROP+重算、以及「load front, recompute tail」的分界直接重疊。

> 原文：The calibration search selects recomputation ratios of 36.4% and 30.9% for HDD and SSD, respectively, increasing the TTFT speedup over Full Recompute to 2.36× and 2.34× on average. ... when the cache medium is slow, the system should not blindly maximize KV reuse but instead increase the online recomputation ratio appropriately to reduce data read from slow external storage.

**S08.3**（supporting；未進入驗證（單一 agent 擷取））

非前綴重用並非無損。CacheTune 先離線沿序列維度對 KV 做 FFT 頻域分析，挑出語意關鍵的 token 重算，其餘直接重用，並設品質下限 r_min=15%。相對 Full Recompute，TTFT 降低 3.72×–4.86×，但在 5 個資料集（SAMSum、MuSiQue、WikiMQA、HotpotQA、Multi-News）與 3 個模型（Mistral-7B v0.3、Llama-3-8B、Qwen2.5-32B）上平均只保留 94.8% 的生成品質，也就是平均約 5% 的品質損失。這表示非前綴命中本身就是一種有損動作，應和量化一起計入同一個品質約束 ε 的預算。

> 原文：CacheTune achieves accuracy close to Full Recompute, retaining on average 94.8% of the generation quality across five datasets and three models. ... To avoid overly small ratios that harm cross-attention recovery on high-bandwidth media (i.e., GPU or CPU memory), we impose a quality-preserving lower bound rmin=15%.

**S08.4**（supporting；未進入驗證（單一 agent 擷取））

CacheTune 的 SSD/HDD 層評估用的是非常慢的裝置。依 fio 量測，SSD 讀寫約 535/445 MB/s，屬 SATA 等級；HDD 約 205/201 MB/s。平台是 2×RTX 4090，以及以 PCIe Gen3×16 連接的 2×A100-40GB。頻寬是用 fio 量的，不是量服務軟體路徑；依擷取到的全文，也沒有提到 NVMe 或 GPUDirect Storage。因此它的 tier 結論能否外推到 NVMe/GDS，或推到瓶頸在軟體路徑、由 per-descriptor 開銷主導的層（Tiara 發現 2），都沒有驗證過。

> 原文：The RTX 4090 platform uses an SSD-backed cache pool, with fio measuring effective read and write bandwidths of the SSD-backed offload path of approximately 535 MB/s and 445 MB/s.

**S08.5**（supporting；未進入驗證（單一 agent 擷取））

在 KV 放 CPU 記憶體的設定下，CacheTune 相對 CacheBlend、CacheSlide、vLLM Prefix Caching，平均 TTFT 分別降低 25.3%、11.9%、74.4%。它讓 KV 能在任意位置重用的做法是：cache pool 存旋轉前（pre-RoPE）的 key，線上依真實的全域位置再套 RoPE（deferred positional-encoding recovery）；傳輸時也只搬不重算的 (1−r)N 個 token（sparse KV transfer）。CacheTune 是 CacheBlend 後繼系列（CacheBlend、EPIC、CacheSlide、KVLink）目前最新的一個，和 Tiara 的 segment/non-prefix 命中提案直接相鄰。

> 原文：Compared with CacheBlend, CacheSlide, and vLLM Prefix Caching under the same CPU-offloading setting, CacheTune reduces the average TTFT by 25.3%, 11.9%, and 74.4%, respectively. ... Instead of caching post-RoPE Keys during offline KV generation, CacheTune stores raw pre-RoPE Keys in the cache pool.

## S09. https://arxiv.org/abs/2602.01053

- 角度：A: Cross-model, cross-adapter and cross-agent KV reuse (non-prefix)
- 來源性質：primary；日期：2026-02-01 (v1; v2 revised 2026-05-31; Journal ref: ICML 2026 Poster)

**S09.1**（central；三票驗證：支持 1／推翻 2 → **被否決，不可引用**）

LRAgent（ICML 2026 poster）把 multi-LoRA agent 的 KV cache 拆成兩部分：一份所有 agent 共用的 base cache（由 pretrained W0 產生），加上每個 agent 各一份 low-rank LR cache（X·A_i，rank r=8，只套在 Q/V projection）。理論上總 KV 大小降為 non-shared 的 1/N + r/d_out ≈ 1/N。N=3 agents、Ministral-8B、66.4k tokens 時實測約為 Non-Shared 的 1/3，與 FullShared 相差 <1 GB。附錄 Table 22（總記憶體，含 14.95 GB 權重）在 66.4k 時：Non-Shared 39.84 GB，BaseLRShared 23.74 GB。【我方觀察】同表 132.0k 列的 Non-Shared 是 64.34 GB，超過 48 GB A6000 的容量，所以這張表至少有一部分是推估值，不是實測。對分層 KV 的意涵：要放置的物件從「N 份完整 KV」變成「1 份共享 base + N 份極小的 low-rank delta」。

> 原文：Because each LR cache is smaller than the full cache by a factor of r/d_out ≪ 1, the total KV cache size is reduced to 1/N + r/d_out ≃ 1/N of the non-shared scheme. … Additionally, BaseShared and BaseLRShared reduce KV cache memory by nearly 1/3 compared to Non-Shared baseline, as shown in Figure 5, which is comparable to other cache-sharing baselines and only marginally higher than FullShared within 1GB.

**S09.2**（central；三票驗證：支持 0／推翻 3 → **被否決，不可引用**）

同一 context 下，不同 LoRA agent 之間的 KV 差異主要來自 adapter output。Table 1 用 128 個 2k-token HotpotQA 樣本、3 組 agent pair 量測：base cache 的跨 agent 平均 cosine similarity 是 0.9726（LLaMA-3.1-8B）/ 0.9530（Ministral-8B），adapter output 只有 0.0538 / 0.0225，接近正交。key cache 的平均相似度是 0.9922 / 0.9840，所以作者讓所有 agent 直接共用整個 key cache，只有 value cache 需要各 agent 分開處理。這代表 K 和 V 的可共享性不對稱，可以作為 K/V 分開決定 tier 或精度的依據。

> 原文：The average similarity is 0.9922 for LLaMA-3.1-8B-Instruct and 0.9840 for Ministral-8B-Instruct, and even the minimum similarity across agent pairs is higher than the corresponding average base-cache similarity: 0.9726 and 0.9530, respectively. This indicates that the primary cross-agent differences come from the value cache, mainly through the adapter-induced component. Therefore, we simply share the entire key cache across agents in all of our schemes.

**S09.3**（central；三票驗證：支持 3／推翻 0 → 通過）

跨 adapter 重用 KV 有品質代價（相當於一個新動作「借用其他 adapter 的 KV」，有自己的 ε）。在 HotpotQA / ScienceQA agentic QA 上，直接共用完整 KV、不做任何重算的 FullShared，平均準確率最多下降 5.3%；DroidSpeak（重算最敏感的 33% 層）最多下降 2.6%；BaseShared 最多下降 0.7%，BaseLRShared 最多下降 1.5%（各跑 20 次，std 0.16–0.45%）。作者也指出品質損失會回頭推高成本：準確率低的方法會讓 agent 走更多步、累積更長的 context，延遲和 KV 佔用都跟著增加。

> 原文：We observe that latency in agent systems depends on both the cache sharing method’s efficiency and the system accuracy, since lower-accuracy methods tend to take more steps and accumulate longer contexts. … BaseShared stays close to Non-Shared, with an average accuracy drop of at most 0.7%. BaseLRShared also maintains strong accuracy, with an average drop of at most 1.5%. In contrast, FullShared and DroidSpeak exhibit larger average drops, up to 5.3% and 2.6%, respectively.

**S09.4**（supporting；未進入驗證（單一 agent 擷取））

效能測試在單張 NVIDIA A6000 48GB 上進行，用的是 emulated trace：工具取回的 context 從 1k 到 64k tokens，總長 2k–66k，所有 agent 的 prefix 相同，因此 KVLink / KVFlow / KVComm 都退化成 FullShared。BaseLRShared 需要 shared-A 加 Flash-LoRA-Attention kernel，吞吐量最高 2.46×、TTFT 最多降 4.44×，接近 FullShared 的上限。BaseShared 為 1.42× / 1.63×，DroidSpeak 為 1.36× / 1.56×。Non-Shared 在 66.4k 時 OOM，附錄 D.7 說 Section 4.3 的 OOM 主要來自記憶體碎片化。【我方觀察】評估全部在 GPU 記憶體內進行；我們搜尋 v2 HTML 全文，找不到 offload、CPU、SSD、vLLM、SGLang 等字樣。所以 base/LR 分解後的 cache 要怎麼跨 HBM/DRAM/SSD 放置或換精度，這篇完全沒有處理。

> 原文：With Flash-LoRA-Attention enabled, BaseShared achieves up to a 1.42× gain and BaseLRShared achieves up to a 2.46× gain in throughput, approaching the upper bound of full cache sharing with FullShared. … For TTFT, BaseShared and BaseLRShared provide up to 1.63× and 4.44× reductions, respectively, both exceeding DroidSpeak which achieves up to a 1.56× reduction. … We conducted experiments on a single NVIDIA A6000 48GB GPU.

**S09.5**（supporting；未進入驗證（單一 agent 擷取））

適用範圍有兩個限制。(i) BaseLRShared 需要 shared-A 的 multi-LoRA，也就是用共用的 A 矩陣重新訓練 adapter。已經部署、權重不能修改的 multi-LoRA 系統（例如現成的 LoRA 集合）無法直接套用。(ii) 收益取決於跨 agent 的 context overlap。Table 21（LLaMA-3.1-8B，33.7k trajectory）中，Non-Shared 是 683.2 tok/s；BaseLRShared 在 100% overlap 時 1678.1，20% 時 764.9，0% 時 681.6，收益完全消失。所以跨 adapter KV 共用值不值得，要看真實 workload 的 overlap 分佈；這篇的 overlap 是人工設定的。

> 原文：We acknowledge that, in systems where a multi-LoRA model has already been deployed and the model weights cannot be modified, BaseLRShared may not be directly applicable because it requires shared-A to avoid accuracy degradation. … As expected, the throughput advantage of KV cache sharing decreases as the overlap ratio becomes smaller and converges toward the non-shared baseline when the overlap ratio approaches 0%.

## S10. https://arxiv.org/abs/2411.02820

- 角度：A: Cross-model, cross-adapter and cross-agent KV reuse (non-prefix)
- 來源性質：primary；日期：2024-11-05 (arXiv v1); v4 2025-07-14; USENIX NSDI 2026

**S10.1**（central；三票驗證：支持 0／推翻 3 → **被否決，不可引用**）

Across LLMs that share an architecture but have different fine-tuned weights, reusing another model's entire KV cache causes a large accuracy loss. On average only 11% of layers are 'critical'. DroidSpeak recomputes just those layers and reuses the other model's KV for the rest, with negligible quality loss. This is the closest prior work for cross-fine-tune / cross-agent KV reuse.

> 原文：On average across all pairs of models, we identify 11% of layers to be critical. [...] Reusing the whole KV cache between models leads to a huge loss in accuracy. [...] Inspired by the findings, we present DroidSpeak, which selectively recomputes a few layers of the KV cache produced by another LLM and reuses the remaining layers, with negligible quality loss.

**S10.2**（central；三票驗證：支持 2／推翻 1 → 通過）

Evaluation covered 3 datasets and 8 model pairs: fine-tuned versions of Mistral-7B, Mistral-24B, Llama-3-8B, Llama-3.1-8B, Phi-3.5-mini-instruct, Llama-3-70B and Llama-3.1-70B. It ran on two Azure A100 VMs connected by InfiniBand. Against a baseline that allows no cross-model sharing, prefill latency (TTFT) dropped 1.7-3.1x and throughput rose up to 4x (Section 5.3 says 2-4x). Quality loss in F1, Rouge-L and code similarity was negligible.

> 原文：Across three datasets and eight model pairs, DroidSpeak can reduce the prefill latency by 1.7–3.1× without compromising accuracy. [...] Experiments on diverse datasets and model pairs demonstrate that DroidSpeak achieves up to 4x throughput improvement and about 3.1x faster prefill (time to first token), with negligible loss of quality in F1 scores, Rouge-L or code similarity score, compared to the baseline which does not allow any sharing across models.

**S10.3**（supporting；未進入驗證（單一 agent 擷取））

Cross-model reuse adds a new kind of state beyond KV: the E cache, which holds hidden states at transition layers. At every layer where reuse switches to recompute, the sender must store and transmit its E cache. For the Mistral-7B and Llama-3-8B families the E cache can be up to twice the size of the KV cache. That directly changes capacity and transfer budgets for tiered KV management (HBM/CPU/SSD).

> 原文：For any transition layer (between reuse and recompute), the sender model must store and transmit the E cache to the receiver model. [...] The E cache is typically large, reaching up to twice the size of the KV cache for the Mistral-7B or Llama-3-8B model families

**S10.4**（supporting；未進入驗證（單一 agent 擷取））

The implementation is about 3K lines of Python on PyTorch 2.0, vLLM and LMCache 0.1.4. It splits KV/E cache per layer into a key-value store in GPU memory. It fetches from a remote GPU node with torch.distributed on a separate CUDA stream so transfer overlaps compute. Pipelining layer-wise recompute with KV loading gives about 2x TTFT improvement. The evaluated data path is GPU-to-GPU over InfiniBand. KV compression and offloading appear only as related work (Section 7) and are not part of the system, so interaction with CPU/SSD tiers and precision tiers is not studied.

> 原文：We split the KV or E cache into layers, and store it in a key-value store in GPU memory. [...] Both fetch_kv and fetch_e are implemented with torch.distributed to fetch KV or E cache from a remote GPU node. All transmission will be placed on a CUDA Stream different from PyTorch's default computation stream, enabling us to overlap transmission. [...] Some work focuses on compressing or offloading KV cache for reduced memory and transmission costs.

**S10.5**（supporting；未進入驗證（單一 agent 擷取））

Limits of the approach: it only works for models derived from the same foundation model (same architecture) and does not support sharing across different foundation models. The critical-layer set must be profiled offline for each model pair using a 'training' dataset. For Llama-3-8B (32 layers) this takes 3 hours on one A100. These limits leave room for new work on online or per-request selection of reuse/recompute layers and on cross-foundation-model reuse.

> 原文：DroidSpeak as-is does not support KV cache sharing across LLMs originating from different foundation models [...] For each model pair, we use a 'training' dataset to determine the critical layer group. [...] For example, for Llama-3-8B with 32 layers, it takes three hours on an A100 GPU.

## S11. https://arxiv.org/abs/2512.16822

- 角度：A: Cross-model, cross-adapter and cross-agent KV reuse (non-prefix)
- 來源性質：primary；日期：2025-12-18

**S11.1**（central；三票驗證：支持 2／推翻 1 → 通過）

既有的 position-independent caching（PIC）系統（CacheBlend、EPIC）對每個請求各自做 selective recomputation 與位置編碼調整，所以同一個 chunk 的 KV 在不同請求間會分歧，無法在 paged KV cache 中做 page 對齊或共享。結果是即使很多請求重用同一段內容，HBM 也只省下有限的量。換句話說，非前綴命中若沿用「每個請求各自修補」的做法，快取裡會存多份同一 chunk 的 KV，這直接影響分層快取的有效容量與去重。

> 原文：because recomputation and positional adjustment are performed independently per request, the resulting KV representations diverge across requests and cannot be page-aligned or shared in the paged KV cache.

**S11.2**（central；三票驗證：支持 0／推翻 3 → **被否決，不可引用**）

MEPIC 以 NoPE（positional-encoding-free）格式存 chunk KV，materialize 時不套 RoPE，改在 fused RoPE-attention kernel 裡即時套上旋轉偏移。已快取的 chunk 只重算第一個 KV block，其餘 block 當作 canonical KV，可跨位置、請求與批次共享。因此非前綴命中的重算成本從 token 級、與位置成線性，變成每個 chunk 固定一個 block。這對「load front, recompute tail」只適用前綴命中的前提構成直接反例，也是 segment／非前綴命中方向最接近的前作。

> 原文：Newly encountered chunks are fully recomputed, while cached chunks require recomputation of only the first KV block, with the remaining blocks reused as canonical KV. / MEPIC instead stores KV in a positional-encoding-free (NoPE) format, omitting RoPE during KV materialization.

**S11.3**（supporting；未進入驗證（單一 agent 擷取））

量化結果：在不同 QPS 下，MEPIC 的 HBM 用量比 CacheBlend 低 5.74 倍、比 EPIC 低 5.25 倍；在各種 chunk size 下低 2.97 到 5.21 倍（摘要的標題數字則是比 SOTA PIC 最多低 2 倍、長 prompt 最多低 5 倍）。端到端延遲比 EPIC 低 9.1%、比 CacheBlend 低 11.48%；F1／Rouge-L 準確度與 baselines 相當或略高。

> 原文：It lowers HBM usage by 5.74× relative to CacheBlend and 5.25× relative to EPIC. ... achieving 9.1% lower latency than EPIC and 11.48% lower latency than CacheBlend.

**S11.4**（central；三票驗證：支持 0／推翻 3 → **被否決，不可引用**）

評估範圍很窄：只有一個模型（Mistral-7B-Instruct-v0.3），在 Ascend 910B NPU（64 GB HBM）上跑，用四個 QA 資料集（SQuAD、NewsQA、NarrativeQA、emrQA）各 300 個請求，平均請求長度只有 1,435 到 2,224 tokens。系統雖然透過 LMCache 做 CPU／磁碟持久層（lazy LRU donation），但從 HTML 全文萃取時沒找到 CPU／磁碟層的載入延遲量測，也沒找到 32K 以上長上下文或 NVIDIA/AMD GPU 的結果。所以「NoPE canonical chunk KV 在多層儲存與長上下文下的成本」這塊還是空的。

> 原文：Experiments are conducted on Ascend 910B NPUs, each equipped with 64 GB of HBM ... All experiments use the Mistral-7B-Instruct-v0.3 model. ... resolving memory pressure via lazy LRU donation and integrating with LMCache for off-device persistence.

**S11.5**（supporting；未進入驗證（單一 agent 擷取））

作者自己把「chunk-aware cache 與模型量化或記憶體壓縮整合」和「dynamic chunk prioritization／heat-aware eviction」列為未來工作。可見 PIC 與 KV 精度分層（FP8/INT4）、驅逐策略的結合，這篇還沒處理；這正好和我們「INT4 取代放置」「LRU 已接近 oracle」的發現交會。新穎性風險在於 Huawei 同一團隊可能接著做。

> 原文：Integrating the chunk-aware cache with model quantization or memory compression techniques could reduce overall memory footprint

## S12. https://arxiv.org/abs/2609.10266

- 角度：A: Cross-model, cross-adapter and cross-agent KV reuse (non-prefix)
- 來源性質：primary；日期：2026-09-09

**S12.1**（central；三票驗證：支持 0／推翻 3 → **被否決，不可引用**）

KVShareArena（Xi Shi & Qian Lou, University of Central Florida；arXiv 2609.10266 v1, 2026-09-09, cs.CL, 未經同儕審查）指出，現有 KV-cache benchmark 只測 exact-prefix reuse。它提出一個 benchmark，測 non-prefix reuse 與跨 model checkpoint reuse。non-prefix 情境有兩種：RAG 每個 query 重新組合的 retrieved chunks，以及 multi-agent coordinator 讀取其他 agent 寫出的 reports。品質指標是 PGR = (S_method − S_floor)/(S_ceiling − S_floor)，其中 floor = 不用 cache，ceiling = full recomputation。成本分三軸：compute（PrefillWorkSaved = 1−R/D）、memory（KVBytesSaved）、cache 已在手上時的 per-request TTFT。一次性建 cache 的成本另外報告。harness 以 pip 套件 kvsharearena 發佈，並附 public leaderboard。對本研究的意義：它是 finding (4)（segment/non-prefix hit）最接近的 prior work，也可直接拿來當品質評測框架。

> 原文：existing benchmarks test only exact-prefix reuse, where nothing is lost. KVShareArena benchmarks KV-cache reuse across prompt contexts and model checkpoints on retrieved chunks and agent reports. It scores every method by the fraction of the gap it recovers between no cache and full recomputation, and charges compute, memory, and per-request latency with the cache in hand, reporting the one-time cost of building a cache separately.

**S12.2**（central；三票驗證：支持 0／推翻 3 → **被否決，不可引用**）

不需重算的 position correction（free position alignment）在只依賴單一來源的問題上就夠用。一旦問題要同時結合多個來源（multi-hop QA），只有付出代價的方法能收回約 1/2 到 2/3 的 gap。代價有兩種：一是部分 re-encode，例如 CacheBlend 實測重算 17%，把 multi-hop 拉回 .39–.46 的 gap；二是訓練，例如 KVPacket。沒修復的 cache 可能比完全不用 cache 還差：在 agent reports 軌上，naive assembly 的 PGR 是 −.82。對本研究的意義：non-prefix hit 不是免費命中，放置與命中決策必須把 repair 重算量算進預算。這推翻了「hit = 省下整段 prefill」的假設，也表示 DROP+recompute 與 load 應該能在同一個 segment 上部分共存。

> 原文：We find that correcting positions, which needs no recomputation, is enough until a question needs several sources at once. There, only methods that pay, by re-encoding part of the cache or by training, recover half to two thirds of the gap; unrepaired caches can be worse than no cache. [...] pulling multi-hop back to .39–.46 of the gap [...] CacheBlend pays a measured 17% while still significantly beating free [...] naive assembly is toxic (−.82): misplaced KV is worse than no context

**S12.3**（central；三票驗證：支持 0／推翻 3 → **被否決，不可引用**）

在記憶體軸上，只有 compression 類方法能省位元組，而且要付出品質。在實際 tensor 上量，4-bit quantization 只省 71.9%，不是名目的 75%，因為 scale factors 也要存。所有修復品質的方法都保留完整 cache，KVPacket 與 MiniPIC 還額外多用約 3%。這些 compression 方法在單一 prompt 上近乎無害，但在新寫出的 agent reports 上，每一列 compression 都顯著低於 free position alignment，排在最前面的則是以重算為基礎的 repair。benchmark 的 compression 家族包含 SnapKV、TOVA、StreamingLLM、Knorm 與 4-bit quantization；是否每一種都在 agent reports 軌上評測過，沒有逐一核對。對本研究的意義：這與 finding (1)「INT4 取代 placement」有張力——在 cross-agent / non-prefix reuse 下，INT4 可能不再免費，precision tier 與 reuse 模式之間有交互作用。71.9% 這個數字也可以用來校正容量算術。

> 原文：Cache-compression methods that are harmless on a single prompt fall significantly behind position correction on freshly written agent reports. [...] Only compression saves bytes, and it pays quality for them. Measured on actual tensors, 4-bit quantization saves 71.9%, not the nominal 75%, because scale factors must be stored. [...] Every method that repairs quality stores the full cache; KVPacket and MiniPIC add about 3%. [...] Every compression row is significantly below free alignment while recomputation-based repair holds the top

**S12.4**（supporting；未進入驗證（單一 agent 擷取））

跨 checkpoint 重用的實驗設定是：cache 由同架構的另一個 checkpoint 寫出，可能是進一步 post-train 的 specialist sibling，也可能是 receiver 的 base-weight 前身。結果是多數 training-free 方法幾乎不受影響。例外有兩個：用 producer cache 訓練的 adapter KVPacket，在六格中有四格顯著下降（最多 −.135）；LegoLink（EPIC）的零重算變體在 multi-hop QA 上崩潰（−.231）。這個結論只涵蓋 architecture、tokenizer 與 RoPE base 三者都相同的 producer。對本研究（angle A）的意義：同家族 fine-tuned 變體之間共享 KV tier 的品質風險，可以用 training-free 修復控制住；但不同架構或不同 tokenizer 之間的跨模型重用，目前沒有證據。LoRA 變體沒有被明確測過。

> 原文：When a different checkpoint wrote the cache, training-free methods are barely affected, while an adapter trained on one checkpoint's caches loses quality. [...] a further-trained specialist sibling (the deployment case: another post-trained model wrote the cache) or the receiver's base-weight predecessor [...] KVPacket drops significantly in four of six cells (up to −.135), and LegoLink's zero-recompute variant collapses on multi-hop QA (−.231). [...] The checkpoint factor covers matched architecture, tokenizer, and RoPE base, and says nothing about producers that differ in any of the three.

**S12.5**（supporting；未進入驗證（單一 agent 擷取））

KVShareArena 的量測範圍沒有涵蓋階層式儲存與長上下文。具體限制如下：
- 輸入上限 29K tokens。
- 主榜排名只根據一個未具名的 8B 模型；確認榜是第二家族 Llama-3.1-8B-Instruct 與一個 4B sibling，只複現「模式」，不複現排名。
- N=100 時，最小可偵測配對差約 .10。
- 延遲是在 cache 已在手上時量的 per-request TTFT，每一列 reuse 約比 dense prefill 低 .9。
- 從較慢儲存串流 cache 的成本，只用解析式 B_method/bandwidth 加上去，沒有實測。
對本研究的意義：兩件事都是空白——一是長上下文（>29K，到 128K–262K）下的 non-prefix reuse，二是它與 CPU/SSD tier 真實載入成本的交互作用，後者包括我們量到的 software-path per-descriptor 開銷。這可以做成一個新方向，同時也標出了 novelty 風險的邊界。

> 原文：Per-row rankings rest on one 8B model; confirmation boards replicate patterns, not rankings. [...] the input cap is 29K tokens. [...] At N=100 the minimum detectable paired difference is roughly .10 [...] per-request TTFT with the cache in hand is ≈.9 below dense prefill for every reuse row [...] a cache that streams from slower storage adds B_method/bandwidth to every read, and the memory axis supplies B_method in bytes. [...] A second family (Llama-3.1-8B-Instruct) gets its own frozen anchors and method runs

## S13. https://arxiv.org/abs/2605.17613

- 角度：E: Precision tiers as speculative draft/verify; long outputs from reasoning models
- 來源性質：primary；日期：2026-05-17

**S13.1**（central；三票驗證：支持 3／推翻 0 → 通過）

VeriCache（University of Chicago / Tensormesh / Samsung Semiconductor / Microsoft Research；建於 vLLM + LMCache，約 8K LoC Python/C++）讓「有損 KV」和「完整 KV」同時存在，不再二擇一：GPU HBM 放壓縮 KV 負責 draft，完整 KV 放在 CPU DRAM（或 storage／遠端節點）負責 verify，輸出與 full-KV decoding 完全相同。壓縮方式透過統一的 compressor 介面接入，涵蓋 token-dropping（KVzip、KVzap、ExpectedAttention、SnapKV）與量化（KIVI、KVQuant、RotateKV，測 8/4/2 bits）。對本研究的意義：『把精度層當 speculative draft/verify』這個方向已被此文佔走，novelty 風險高。它也推翻了我們動作空間『每個 block 只待在一個 tier』的前提，例如 INT4@HBM 與 BF16@DRAM 可以並存。

> 原文：VeriCache uses the compressed KV cache to draft tokens, then verifies them against the full KV cache.

**S13.2**（central；未進入驗證（單一 agent 擷取））

有損 KV 的誤差會隨輸出長度累積：序列層級 KL 隨 decode 步數線性成長。KVzip 4× 每步約 0.023 nats，250 步後累積約 6 nats。在需要逐字精確的長輸出 agentic 任務上會崩潰：KVzip 4× compaction 下，ComplexFuncBench 的 function-call accuracy 低於 10%，SWE-bench Lite 的 code-format accuracy（輸出須為合法 git diff）接近 0；論文稱此時表面相似度指標仍高，但功能正確性已崩潰。注意：這些失敗數字來自 token-dropping（KVzip），不是量化。對本研究 finding (3)（ε 隨 model×task 而異）的意義：用短輸出或相似度指標量出的 ε，可能大幅低估 tool-calling、長輸出、reasoning 工作負載的品質損失。

> 原文：Function call accuracy—which requires every call name and argument to match exactly—drops below 10% under KVzip 4× compaction

**S13.3**（supporting；未進入驗證（單一 agent 擷取））

核心系統洞察：用壓縮 KV 做 draft（逐 token decode）受 HBM 頻寬限制；verify 則要把完整 KV 從 CPU 經 PCIe、或從 storage 經網路搬回 GPU，受 interconnect 頻寬與 FLOPs 限制。兩者可以平行重疊，而夠長的 draft horizon 能攤提每次 full-KV swap。長上下文情境下完整 KV 放在 CPU memory，每次 verify 都經 CPU–GPU interconnect 重新載入。評估設定用鏈路速率描述：CPU↔GPU 為 PCIe 5.0 ×16（64 GB/s），local storage 40 GB/s，remote storage 1.2 GB/s。對本研究 finding (2) 的意義：如果實際傳輸由軟體路徑決定（MI300X 每 descriptor 約 13 µs；3090 的 SSD 層只有 raw NVMe 的 1/18），這個攤提條件在 OffloadingConnector 這類路徑上未必成立。『用實測的軟體路徑頻寬重新界定 draft/verify 何時划算』是此文沒有處理的角度。

> 原文：(1) compressed-KV decoding can be parallelized with full-KV swap, because one is HBM-bandwidth-bound and the other is PCIe/network-bound

**S13.4**（supporting；未進入驗證（單一 agent 擷取））

實測數字。長上下文 decoding：比 Full KV 快 1.92×–2.73×（Llama-70B 為 256 vs 102 tok/s）。remote prefix caching：快 1.33×–2.11×（Llama-70B 為 485 vs 240 tok/s）。摘要裡的 headline『up to 4×』是上限。4× compaction 下 acceptance length 約 19（Qwen-32B）與約 23（Llama-70B），EAGLE 約 11–22，小型 draft model 只有約 3。其中一條 pipeline 用 KIVI 4-bit 當 drafter，draft length 40。評估硬體只列 NVIDIA RTX PRO 6000（96 GB）與 2×H100 NVL（TP=2），模型為 Mistral-24B、Qwen-32B、Llama-70B；沒有 AMD MI300X 或 24 GB 消費卡的數據。

> 原文：VeriCache delivers 1.92×–2.73× over Full KV (e.g., 256 vs. 102 tok/s on Llama-70B)

**S13.5**（central；未進入驗證（單一 agent 擷取））

作者在 §10 自列的限制可以直接當成新方向：(i) 記憶體開銷：完整 KV 放 CPU/storage，GPU 上還有一份壓縮 KV，等於存兩份；(ii) draft length 固定，需要根據早期 accept/reject 結果做 per-request 自適應；(iii) 現有壓縮器優化的是直接服務時的 accuracy，而『大 draft horizon 下的 acceptance length』是另一個目標；(iv) 非 prefix 的 KV 重用（CacheBlend）同樣有損，也會偏離 full-KV 輸出，同樣需要驗證。對本研究：(iii) 對應『以 acceptance length 而非 ε 來選精度層』；(iv) 對應 finding (4) 的 segment/non-prefix hit，可以把 segment hit 當 draft、用完整 KV 或重算來驗證，讓非 prefix 重用變成無損；(i) 則把雙份儲存帶進 DRAM/SSD 容量與寫入預算的權衡。

> 原文：Existing compressors optimize direct-serving accuracy; a compressor designed to maximize acceptance length at large draft horizons—a different objective—could push VeriCache's throughput further.

## S14. https://arxiv.org/abs/2502.10424

- 角度：E: Precision tiers as speculative draft/verify; long outputs from reasoning models
- 來源性質：primary；日期：2025-02-05 (arXiv v1); ICML 2025, PMLR v267 pp. 59668-59686 (July 2025)

**S14.1**（central；未進入驗證（單一 agent 擷取））

QuantSpec 的 hierarchical quantized KV cache 把 INT8 KV 拆成 upper-4-bit 與 lower-4-bit 兩個 INT4 plane（INT8 = INT4 + INT4 residual）。draft 只載入 upper 4-bit（等同 INT4 KV），target 驗證時兩個 plane 都載入，重建成 INT8。所以同一份位元組同時是 INT4 與 INT8 兩個精度階，不必另存一份 INT4 副本。這是「巢狀精度階共用儲存」的直接先例，可以對應我們的 GPU-INT4 / GPU-FP8 精度階。整個設計都在 GPU HBM 內；讀過的內容裡沒看到任何 CPU/SSD offload。把 lower nibble 下放到較慢層、只在驗證時取回的「bit-plane 跨層」做法，看起來還沒有人做。

> 原文：This effectively allows us to use a hierarchical design to represent the KV cache of the draft model in INT4 and the target model in INT8 at the same time, removing the need to store a separate INT4 copy.

**S14.2**（central；未進入驗證（單一 agent 擷取））

條件：LWM-Text-Chat-128k、Multi-LexSum、128K context、batch size 1、8×RTX A6000。QuantSpec 的 draft acceptance rate 94.31%，端到端加速 2.49×，peak GPU memory 61.22 GB；兩個 sparse-KV self-speculative baseline（StreamingLLM、SnapKV）都 OOM。摘要層級的主張是：acceptance >90%、最高約 2.5× 加速、比 sparse-KV 替代方案省約 1.3× 記憶體。限制：只測了 Llama-2 系模型（Llama-2-7B-32K-Instruct、LWM），也只測 batch=1，沒有涵蓋 serving 批次，也沒有涵蓋對 FP8/INT4 敏感的 Qwen2.5 這類模型。

> 原文：QuantSpec maintains high acceptance rates (>90%) and reliably provides consistent end-to-end speedups upto ∼2.5×, outperforming other self-speculative decoding methods that use sparse KV cache for long-context LLM inference. QuantSpec also reduces the memory requirements by ∼1.3× compared to these alternatives.

**S14.3**（supporting；未進入驗證（單一 agent 擷取））

在需要全文的任務上，sparse-KV draft（丟 token）比量化 draft 損失大得多。Llama-2-7B-32K-Instruct 跑 Multi-LexSum 時，SnapKV draft 的 acceptance 在 8K 只有 55.55%、32K 為 72.54%；QuantSpec 則是 91.23% / 91.16%。這支持我們兩個已有發現：量化可以取代放置/逐出，以及 attention-importance 訊號不可靠。推論（論文沒做）：speculative acceptance rate 或許可以當成低精度 KV 階的線上、逐請求品質代理訊號，不必事先離線量 ε。

> 原文：for such tasks where the whole context is important, sparse KV cache methods are much more lossy, whereas quantization preserves most of the information in the context.

**S14.4**（supporting；未進入驗證（單一 agent 擷取））

加速的來源會隨 context 長度移轉：短 context 主要靠 weight 量化，中等長度兩者都有貢獻，長 context 則以 KV 量化最有效（KV 隨 context 線性成長，decode 延遲由 KV 載入主導）。這和我們用 Amdahl 份額判斷「何時值得做 KV 放置」的準則方向一致。推論（論文沒測 reasoning 模型）：長輸出 decode 是 KV 精度階最有利的區間。

> 原文：showing that for short contexts most of the speedup comes from quantizing weights, for medium length prompts both weight and KV cache quantization contribute to the final speedup, and KV cache quantization is most effective for long contexts.

**S14.5**（supporting；未進入驗證（單一 agent 擷取））

QuantSpec 的 target 用的不是 FP16 KV，而是 INT8 重建的 KV，外加一個 full-precision double buffer，保留最近 G 到 2G 個 token。所以它的「無損」只是相對於 INT8-KV 模型。報告的品質代價（Llama-2 系）：WikiText-2 perplexity 6.4595→6.4696，C4 7.2617→7.2620。量化與 KV 搬移每 G 個 decode step 才做一次，用來攤提開銷。可檢驗的新角度：把 INT4 階的品質風險轉成 acceptance（也就是速度）風險，並把較高精度的驗證副本放在較慢層。但在我們觀察到 FP8/INT4 會崩的 Qwen2.5 上，INT8 target 是否仍然近乎無損，論文沒有驗證。

> 原文：When verifying the drafted tokens using the target model, we utilize both the upper and lower 4-bit representations to reconstruct the KV cache in the higher INT8 precision.

## S15. https://github.com/vllm-project/vllm/issues/39321

- 角度：E: Precision tiers as speculative draft/verify; long outputs from reasoning models
- 來源性質：forum；日期：2026-04-08
- ✅ 抽查：作者 wenxinzhang0，2026-04-08，Closed；APC 快取 thinking token、剝除後 hash 鏈分岔、QwQ-32B 每條死鏈 ~1.3 GB（假設每輪 ~5000 thinking token）——一致。頁面內容沒有列出 linked PR（擷取提到的 #39806、#41939 未抽查）

**S15.1**（central；未進入驗證（單一 agent 擷取））

vLLM（main，issue 作者於 2026-04-08 驗證）以 `--reasoning-parser` 服務 reasoning model（DeepSeek-R1、QwQ 等）時，Automatic Prefix Caching 會雜湊並快取全部輸出 token，包括 `<think>` 思考 token。請求結束後，這些 block 只做 ref-count 遞減，仍留在 `cached_block_hash_to_block` 當 eviction 候選。客戶端若照 DeepSeek API 文件在下一輪剝除思考內容，這些 block 就不可能再被任何未來前綴命中，只能等 LRU 把它們擠出去。換句話說，LRU 管理的 KV pool 裡有一類 block 在寫入當下就能判定為 dead-on-arrival，LRU 卻把它當成最近使用的熱資料，讓它和有用的項目競爭空間。

> 原文：When serving reasoning models (DeepSeek-R1, QwQ, etc.) with `--reasoning-parser`, all output tokens — including `<think>` tokens — are hashed and cached via Automatic Prefix Caching (APC). If the user does not include thinking tokens when constructing the next turn's prompt (e.g., DeepSeek's API docs explicitly say not to), these cached blocks will never be matched by any future prefix, becoming dead weight that wastes GPU memory until LRU eviction. ... When the request finishes, these blocks are freed (ref count decremented) but remain in `cached_block_hash_to_block` as eviction candidates — not deleted.

**S15.2**（central；未進入驗證（單一 agent 擷取））

每個 block 的 hash 都依賴 `parent_block_hash`（`kv_cache_utils.py` 的 `hash_block_tokens`）。所以 turn 2 以 [Q1, A1, Q2]（已剝除 T1）送進來時，只有 Q1 的 block 會命中。A1 的 KV 明明還在快取裡，卻「卡在 T1 之後」，只能重算；turn 1 的 T1→A1 鏈從此永久不可達，而且每一輪都會重複累積。這是 prefix-only 命中語意在多輪 reasoning 對話中造成「有 KV 卻得重算」的實際生產案例，對應我們的發現 (4)。但有一個限制：同一作者後續的 vLLM PR #39806 和 SGLang PR #23315 都指出，A1 是在位置 [input_len+thinking_len, ...] 計算的，剝除 T1 後直接重用會出現 RoPE 位置錯位（另外，第 2 層以上的 K/V 已經混入對 T1 的 attention，這點是我們的分析）。因此要回收 A1，只能靠 CacheBlend 類的非前綴、位置校正式重用，本質上有損、需要評估 ε，不是無損命中。issue 裡「只快取 origin_input_ids + answer_tokens」的提案忽略了這一點。

> 原文：Only `Q1` blocks match. `A1` must be recomputed despite its KV being in the cache (stranded behind `T1` in the hash chain). ... The old `T1 → A1` blocks from turn 1 are now permanently unreachable. This repeats every turn, accumulating dead blocks.

**S15.3**（supporting；未進入驗證（單一 agent 擷取））

issue 給出的浪費量級是紙上算術，不是實測。以每輪約 5000 個 thinking token 計，每條死鏈在 QwQ-32B 約 5000×256 KB ≈ 1.3 GB，在 DeepSeek-R1-Distill-70B 約 5000×320 KB ≈ 1.6 GB；50 個並行對話各留一條近期死鏈，就有約 65–80 GB 的死 KV 在競爭 eviction。每 token 的 KV 大小經驗算一致：QwQ-32B 是 64 層×8 KV heads×128 dim×2(K/V)×2 B = 256 KiB，70B 是 80 層 → 320 KiB。但「每輪 5000 token」和「50 個並行」都是假設。目前找到的實測只有兩筆，都出自後續 PR（不是本 issue）：#39806 引用 SGLang 修正在 QwQ-32B、2×B300 上的結果，cache hit rate 11.1%→17.1%，TTFT p50 9.98 s→8.96 s；#41939 在 Qwen3-14B、20 個相同請求下，峰值 KV 使用率 100%→92.4%，hit rate 維持 76.0%。實際效益與工作負載高度相關，還沒有 trace 級的量化。

> 原文：With ~5000 thinking tokens per turn (typical for DeepSeek-R1), each dead chain wastes: QwQ-32B: 5000 × 256 KB/token ≈ 1.3 GB / DeepSeek-R1-Distill-70B: 5000 × 320 KB/token ≈ 1.6 GB / 50 concurrent conversations with 1 recent dead chain each: 65-80 GB of dead KV cache competing with useful entries for eviction.

**S15.4**（central；未進入驗證（單一 agent 擷取））

issue 的根因診斷是：「快取什麼」和「未來請求會送什麼」由兩個互不相關的元件各自決定。vLLM 的 chat completion API 已經用 `--reasoning-parser` 把 `reasoning_content` 和 `content` 分開，快取層卻不分角色地雜湊所有輸出 token。提案是：啟用 reasoning parser 時，讓 `update_block_hashes` 跳過 thinking token，另以 `--cache-thinking-tokens` 讓自訂 prompt 構造的使用者 opt-in。對本研究的含意是：token 的角色（prompt / thinking / answer / tool output）在寫入當下就拿得到，可以用來判斷之後會不會被跨請求重用。這和我們的發現 (7)（attention importance 預測跨請求重用的 AUC 約 0.5）正好形成對照，可以發展成依協定與角色做 admission 和 tier placement 的方向。反例在 PR #39806 的審查討論裡：reviewer 指出支援 interleaved thinking 的 agentic 工作流會把 reasoning 帶進後續請求，所以可達性取決於客戶端協定和 chat template，不能一律丟棄。

> 原文：The core issue is that what gets cached and what gets sent in future requests are decided independently. For example, vLLM's own chat completion API separates `reasoning_content` from `content` in responses via `--reasoning-parser`, yet the caching layer hashes all output tokens indiscriminately. ... When `--reasoning-parser` is set, `update_block_hashes` should skip thinking tokens from the hash chain, so that only `origin_input_ids + answer_tokens` are cached.

**S15.5**（supporting；未進入驗證（單一 agent 擷取））

同樣的問題也存在於 SGLang 的 RadixAttention（平行 issue sgl-project/sglang#22373）。兩邊的修正狀態不一樣（已逐頁查證）。SGLang 在 PR #23315（2026-04-21 merged）加入 opt-in、預設關閉的 `--strip-thinking-cache`：只提交 prompt 前綴，thinking 和 answer 一起丟掉，理由就是 RoPE 位移。vLLM 這邊，issue #39321 標為 Closed，但沒有連結任何 PR；兩個修正 PR（#39806「Fixes #39321」、#41939 `--strip-thinking-tokens-from-cache`）都沒合併，因閒置被關閉。所以說「已由 PR #23315 修正」只適用於 SGLang，而且預設沒開。另外查了 vLLM main 的 OffloadingConnector 原始碼：`_build_store_jobs` 以 `num_computed_tokens + num_scheduled_tokens` 決定可卸載量，所以 decode 產生的 block 也會被卸載，除非設定 `offload_prompt_only`。由此推論（未量測）：在預設的 CPU/SSD 階層設定下，死掉的 thinking block 很可能也會被寫進下層，白白消耗傳輸路徑和 SSD 寫入預算，連到我們的發現 (2)(5)。

> 原文：Note: filed a parallel issue on SGLang (sgl-project/sglang#22373) where the same problem exists with RadixAttention.

## S16. https://arxiv.org/abs/2512.01278

- 角度：E: Precision tiers as speculative draft/verify; long outputs from reasoning models
- 來源性質：primary；日期：2025-12-01
- ✅ 抽查（只到摘要）：SparseSpec，Zhao ... Kasikci, Han, Stoica，2025-12-01，cs.LG，無 venue；最多 2.13× 吞吐——一致。offload 細節、消融數字未抽查

**S16.1**（central；未進入驗證（單一 agent 擷取））

長輸出 reasoning 模型的 decode 瓶頸在讀 KV-Cache，而不在計算。證據一：H100 上服務 Qwen3-8B（batch 128、輸出 8192 tokens），每步載入 KV-Cache 平均 21 ms，佔端到端延遲 70% 以上。證據二：在 AIME（平均輸出 12K）上，attention 佔端到端時間 77% 以上（§3.1）。證據三：Table 1 的輸入只有 124–148 tokens，Qwen3-14B 的輸出均值卻達 10K–13K（AIME 為 13185±7626），可見 KV 幾乎全是 decode 階段自己產生的。注意：§2.2 內文寫 13542 對 2593，與 Table 1 的 13185 對 1732 不一致，引用前需擇一並註明。推論：在這個 regime 下，分層的壓力來自 decode 時逐步變大的 KV 和每一步的頻寬需求，不是 prefix-reuse 命中率。

> 原文：For example, when serving Qwen3-8B [...] on an H100 with a batch size of 128 and an output of 8192, loading the KV-Cache takes on average 21 ms per step, accounting for over 70% of the end-to-end latency.

**S16.2**（central；未進入驗證（單一 agent 擷取））

SparseSpec 的 dynamic KV-Cache manager 刻意不預測輸出長度。它的做法是：先超額接納請求，把 GPU 上的 KV 容量用滿；快要 OOM 時，把 KV offload 到 host DRAM，以避免 recomputation。offload 與 reload 都按 FIFO 順序，單位是請求（文中稱 offloaded requests，GPU 一有空間就優先重新排程）。量測方法是與「把 offload 操作換成空 kernel」的基準比較，結果 offload 只讓 cycle time 平均增加 0.5%。文中沒寫明這項量測的模型與硬體，上下文是 Figure 5 的 Qwen3-8B、AIME、H100。設計動機來自 Figure 5：既有做法（oracle、preemption 等基準）若不是 KV 容量用不滿，就是因長度誤判而大量 recompute。推論：這是 reasoning decode 場景下「CPU tier 對上 DROP+recompute」的直接證據，而且放置政策只是 FIFO。

> 原文：instead of optimizing output-length prediction, SparseSpec prefers to aggressively increase request concurrency to fully utilize KV-Cache, while offloading KV-Cache to host once approaching out-of-memory to avoid recomputation. Note that both offloading and loading follow the FIFO order, assuring fairness and avoiding starvation. [...] The results indicate that offloading prolongs cycle time by only 0.5% on average, which is practically negligible.

**S16.3**（central；未進入驗證（單一 agent 擷取））

消融實驗在 AIME、Qwen3-8B 上，從 naive sparse self-speculation 出發，依序開啟 unified batch scheduler、dynamic KV-Cache manager（含 host offload）、delayed verification。三者分別帶來 1.23×、1.61×、1.12×，相乘為 2.22×（1.23×1.61×1.12≈2.218，數字自洽）。所以 KV 容量管理是三個系統元件中貢獻最大的一項。但這是依序累加的歸因，數字會隨開啟順序改變。論文的解釋（§3.3）是：speedup η 會隨可用 KV 容量變小而下降。推論：在 reasoning 服務中，靠 CPU tier 撐大 effective batch 是主要的系統槓桿。

> 原文：incrementally enable the unified batch scheduler, dynamic KV-Cache manager, and delayed verification. [...] Our experiments reveal that three designs boost the performance by 1.23×, 1.61×, and 1.12×, respectively, culminating in an aggregate throughput gain of 2.22×

**S16.4**（supporting；未進入驗證（單一 agent 擷取））

論文對 offload 頻寬的論證只是紙上算術，沒有量測實際傳輸路徑。其算法是：每步產生 128 個新 token，需 18 MB KV（128×128×8×2×2×36 B = 18,874,368 B）；每步約 10 ms；因此只需 18 GB/s，遠低於 PCIe 上限。但照它自己的數字，18.9 MB ÷ 10 ms ≈ 1.9 GB/s，與文中的 18 GB/s 差 10 倍。這個誤差偏保守，不影響「PCIe 夠用」的結論。全文沒有報告 chunk 大小、pinned memory、PCIe 世代或每筆傳輸的固定成本（已全文檢索確認）。它還靠「優先排程已 offload 的請求」，把最壞情況的 CPU 用量限制在 GPU 容量以內（8×H100 為 640 GB）。推論：這可對照我們的量測（vLLM OffloadingConnector 在 MI300X 上每個 descriptor 約 13 µs）。每步小量增量 offload 在 MI300X/3090 上的真實成本仍未被量過，是可以補的空缺。

> 原文：each decoding step generates just 128 new tokens, requiring only 18 MB of KV-Cache memory [...] Since the GPU latency per iteration is on the magnitude of 10 ms, the necessary bandwidth is only 18 GB/s to overlap offloading with GPU computation, which is well below the PCIe bandwidth limit.

**S16.5**（supporting；未進入驗證（單一 agent 擷取））

PillarAttn 是 self-speculative 的 draft，只讀 5% 的 KV（s=0.05）。要讀哪些 token，是拿 verification 階段客製 kernel 吐出的 attention score 做 Top-K 選出，不需額外估計成本。k=8 時平均接受 6.16/8 個 token，NGram 與 EAGLE3 都不到 2 個。端到端實驗在 DGX-H100 上跑 Qwen3-1.7B/8B/14B，資料集為 AIME、OlympiadBench、LiveCodeBench。吞吐量相對 vLLM 最高 2.13×，相對 vLLM-NGram 最高 1.56×，相對 MagicDec 1.36×，相對 TriForce 1.76×。限制：正在執行的請求，完整 KV 仍放在 GPU 上，因為 verification 用 full attention。SparseSpec 不做 KV 量化，也不做 per-token 的跨層放置；「quantization」一詞在全文只出現在參考文獻裡。推論：每步只碰 5% 的熱集合、完整 KV 每 k 步才讀一次，這種存取結構可以用來切入「精度或層級分層 × draft/verify」方向，而本文沒有做這一塊。

> 原文：PillarAttn achieves an average acceptance token length of 6.16 out of 8 tokens, surpassing all other drafting methods. In comparison, both NGram and EAGLE3 can only draft fewer than 2 accepted tokens. [...] We set the sparsity ratio to 0.05, as performance saturates with further increases in selected tokens.

## S17. https://arxiv.org/abs/2503.24000

- 角度：E: Precision tiers as speculative draft/verify; long outputs from reasoning models
- 來源性質：primary；日期：2025-03-31 (arXiv v1; MLSys 2025 proceedings)

**S17.1**（central；未進入驗證（單一 agent 擷取））

有損 KV 壓縮（KIVI-4、GEAR-4 等 4-bit 量化，以及 H2O-512、StreamingLLM-512）會讓輸出長度分布偏向「變長」。實驗條件是 LLaMA-3.1-8B-Instruct、1,000 筆 ShareGPT、sampling T=1.0，結果有逾 20% 的樣本回應長度達 1.5 倍以上。另在 200 筆「壓縮後變長」的條件子集上（LLaMA-7B，Table 4），平均長度增加 1.55–1.76 倍。作者同時指出，多數情境下壓縮帶來的 decode 吞吐提升不到 1.5 倍，所以這些請求的 end-to-end latency 反而變差，GEAR 甚至拉長了 tail latency。可檢驗的弱點（Table 5）：只把溫度改成 0.9 或 1.1，就有 27.5% / 31.4% 的樣本變長 1.5 倍以上，比四種壓縮法的 21.3–27.1% 還高。壓縮真正的特徵是「縮短 2 倍以上」的樣本變少（6.8–16.5%，對照溫度擾動的 20.8–21.3%），「>20%」這個標題數字本身並不能跟抽樣雜訊區分開。此外，這篇沒有在 greedy decoding、reasoning（長 CoT）模型、FP8、或 CPU/SSD tier 上測過。「量化 tier 引發輸出變長」這筆隱性成本，在長輸出情境下仍是空白。

> 原文：Specifically, more than 20% of the samples show at least a 1.5 × increase in response length. Prior throughput analysis shows that compression methods cannot achieve more than a 1.5× increase in the decoding throughput in many scenarios.

**S17.2**（central；未進入驗證（單一 agent 擷取））

這是 per-request 壓縮/精度選擇最接近的先行工作。作者把 attention layer 的 profiling 接進 Vidur 做 throughput predictor（準確率 85.8–88.5%），再加一個 BERT length predictor（87.8–95.7%），用來在 1 張 FP16 GPU 與 3 張壓縮 GPU 之間路由請求。條件是 LLaMA-7B、LMDeploy、4×A6000、1,000 筆 ShareGPT、Poisson 10 req/s。結果平均 E2E latency 比「4 卡跑同一壓縮法」的 baseline 低 1.45–1.80 倍；若只用 length predictor 則是 0.83–1.03 倍，可能反而變慢。作者建議用輕量模型預測任務類型，並採用「varying compression levels」。新穎性風險：這個設計是每張 GPU 靜態固定一種壓縮法再做路由，並不是在 HBM 精度階層（BF16/FP8/INT4）、CPU、SSD 之間做 per-request 或 per-block 的動態精度放置。後者、以及把精度選擇跟 prefix reuse 或 offload 放在一起考慮，仍未被這篇涵蓋。

> 原文：Combining the throughput predictor and length predictor speeds up the latency by 1.45 to 1.80×.

**S17.3**（supporting；未進入驗證（單一 agent 擷取））

在有 PagedAttention 和 FlashAttention 的 serving framework 中（LMDeploy、4×A6000 NVLink、合成請求），4-bit KV 量化的 decode 吞吐大致持平甚至下降。Table 3 中，KIVI-4 的 decode 相對 FP16 在 TP=1/2/4 分別是 0.98×/0.88×/0.90×，GEAR-4 是 1.02×/0.97×/0.97×；H2O 的 prefill 只有 0.51–0.58×。在長 KV 加大 batch 的設定下，量化法的優勢會消失，KV 長度到 8192 時量化法反而 OOM。TP 越大，每張卡的記憶體頻寬爭用越小，壓縮的收益也被稀釋。作者的結論是壓縮只該用在 KV 很重的請求上。換句話說，效益取決於 kernel 實作與 batch×length×TP，而不是壓縮比本身。這跟本專案「頻寬由軟體路徑決定」的發現同構，但這篇沒有涵蓋 CPU/SSD offload 或 prefix reuse。

> 原文：For heavy settings with long KV length and high batch size, sparsity-based methods can maintain their throughput advantages, whereas the benefits of quantization-based methods tend to diminish. We also observe that quantization-based methods even suffer from out-of-memory issues when the KV length reaches up to 8192 in Figure 1 (l).

**S17.4**（supporting；未進入驗證（單一 agent 擷取））

平均精度幾乎不掉，不代表每個樣本都安全。在 LongBench 加 LLaMA-3.1-8B-Instruct 上，以 10% 相對精度損失為門檻，KIVI、GEAR、H2O、StreamingLLM 都產生大量 negative samples（原本答對、壓縮後答壞的樣本）。把多個演算法做 ensemble 能減少這類樣本，但無法消除。最脆弱的是 summarization 和 QA。作者另外用這些失敗樣本組了一個 benchmark；它是刻意挑出來的，有 selection bias，只能拿來比較方法。在上面，QA 從 52.0 掉到 28.7–33.8，summarization 從 31.6 掉到 23.7–24.8，code 從 97.0 掉到 30.0–61.3。這支持用 per-sample 的尾部品質、而不是平均 ε，來約束精度或放置決策。它也跟本專案「ε 隨 model×task 變動」的發現一致。

> 原文：The minor accuracy loss brought by compression algorithms (e.g., KIVI, GEAR) does not mean that each sample suffers from the minor performance loss. Our pinhole observation indicates a high number of negative samples even with a threshold of 10%, revealing the fragility of compression algorithms.

**S17.5**（supporting；未進入驗證（單一 agent 擷取））

量測方法論：在 HF Transformers（TRL）上量到的壓縮加速不可信，應改在有 PagedAttention 和 FlashAttention 的 serving framework 上量（Observation 1）。用固定回應長度量吞吐也不恰當，必須把壓縮造成的長度分布位移算進 end-to-end latency（Missing Piece 2）。作者也指出，既有的壓縮 benchmark 大多只報平均精度，只有 LLM-QBench 量了 prefill/decode 吞吐。這可以作為 KV tiering benchmark 方法論的依據。本專案若用固定輸出長度或平均 ε 去評估 INT4 tier 的收益（例如 INT4 加 LRU 拿到約 92–104% 的 oracle headroom），就可能落入同一個盲點。

> 原文：Thus, measuring the computational efficiency with a fixed response length is not an appropriate approach.

## S18. https://arxiv.org/abs/2503.16163

- 角度：E: Precision tiers as speculative draft/verify; long outputs from reasoning models
- 來源性質：primary；日期：2025-03-20 (arXiv v1; ICML 2025)
- ✅ 抽查（只到摘要）：SpeCache，Jie/Tang/Han/Deng/Han，2025-03-20。arXiv 頁 comments 沒寫 venue（擷取說 ICML 2025，未確認）

**S18.1**（central；未進入驗證（單一 agent 擷取））

SpeCache（ICML 2025，Peking University + Huawei Noah's Ark Lab）讓精度階與放置階疊加使用，兩者並非互相取代。VRAM 只留 1–2 bit（KIVI 式，group g=32/64）的低位元 KV 副本，完整 16-bit KV 全部 offload 到 CPU DRAM。每個 decode step 依低位元副本算出注意力，取回 top-64 個 16-bit KV pair。在 LongBench 上，與 16-bit baseline 的差距只有 2%（Mistral-7B-Instruct-v0.2，32k）和 1%（LLaMA-3-8B-Instruct，8k），GPU 上只留約 10% 的 KV 大小。範圍限制：只測 LLaMA-2-7B/13B-Chat（4k）、Mistral-7B（32k）、LLaMA-3-8B（8k），全文沒有 Qwen；LLaMA-2 在 1-bit 下品質「significantly poor」，只能用 2-bit。這表示低精度的容忍度隨模型而異，與我們的發現 (3) 一致。這是「INT4 常駐 + BF16 放 CPU、按需取 top-k 回補品質」這類方向最接近的先行工作。

> 原文：our method can maintain a performance gap of only 2% and 1% compared to the 16-bit baseline for Mistral-7B-Instruct-v0.2 and LLaMA-3-8B-Instruct, respectively, while retaining only approximately 10% of the KV cache size in the GPU.

**S18.2**（central；未進入驗證（單一 agent 擷取））

作者在 PyTorch 下實測（Mistral-7B-Instruct-v0.2，32k，NVIDIA A6000）：非連續（sparse gather）的 CPU→GPU 傳輸延遲約為連續傳輸的 5 倍；但因為只傳 top-1% 的 KV，整體傳輸延遲仍降低 95%。這佐證了我們的發現 (2)：KV 傳輸成本由軟體路徑決定，不是硬體頻寬。它也顯示先前工作只觀察到 gather 開銷，沒有系統性地量化（例如每個 descriptor 的固定成本、跨模型差異），所以「傳輸軟體開銷量化」這個方向仍有缺口。

> 原文：For non-contiguous memory transferring, mainstream framework such as PyTorch introduces additional time overhead. Therefore, we need to conduct experiments to verify the efficiency advantage of transferring sparse KV cache compared to the full KV cache. As shown in Figure 2 (right), although sparse CPU-GPU transfers can introduce approximately 5 times the latency, we can enhance efficiency by reducing the amount of data transferred, since the attention mechanism maintains a high hit rate even at 1% sparsity. For instance, with Mistral-7B-Instruct-v0.2 model and context length of 32k, transferring only top-1% of the KV pairs would reduce transfer latency by 95%.

**S18.3**（supporting；未進入驗證（單一 agent 擷取））

「speculative token」機制等於把低精度階當成 draft：每一步同時解碼兩個 token。output token 用已預取的 top-k 16-bit KV 計算輸出；speculative token 用 VRAM 裡的低位元副本，猜測下一步會用到的 top-k KV，讓取回與計算重疊進行。Table 4 的設定是 2k context、48GB A6000 上的最大 batch。和非推測式（注意力算完才取回、再重算）相比，每步延遲 2-bit 從 877 降到 643 ms/step，1-bit 從 1144 降到 775 ms/step；LongBench 平均分只從 42.6 降到 42.5、從 42.4 降到 41.9。作者指出，batch 變大時 CPU 載入時間線性成長、計算時間成長較慢，因此大 batch 下 KV 載入成為瓶頸。

> 原文：As the batch size increases, the time required to load KV pairs from the CPU increases linearly, while the computation time increases more slowly due to the improved parallelism. Therefore, during decoding with large batch sizes, the latency of loading KV pairs becomes the dominant factor. This highlights the importance of parallelizing computation and prefetching.

**S18.4**（supporting；未進入驗證（單一 agent 擷取））

吞吐量測試條件：Mistral-7B-Instruct-v0.2、HuggingFace transformers、單張 48GB A6000。32k context 下，full KV 只放得下 batch 3（10.3 tok/s）；SpeCache 2-bit 可到 batch 22（34.6 tok/s，3.4×），1-bit 可到 batch 36（47.3 tok/s，4.6×）。增益來自更大的 batch，不是單一請求的延遲。實作用 PyTorch multi-stream 加 Tensor.copy_()，作者自承「not theoretically optimal」。範圍也有限：只處理 decode，prefill 在 GPU 上逐層量化後把 16-bit KV 全部 offload；沒有 vLLM/paged KV、prefix/跨請求重用、SSD 階或重算動作（全文搜尋 SSD、prefix、vLLM、recompute 皆為 0 筆）。因此它是單一請求內的 tiering，與跨請求放置問題互不重疊。

> 原文：Note that our CPU-GPU interaction code is implemented using pytorch’s multi-stream mechanism and the Tensor.copy_() method, so the parallelism achieved is not theoretically optimal. By customizing lower-level operators, the efficiency of SpeCache can be further improved. As shown in the Table 3, with 2-bit and 1-bit quantization, SpeCache allows the batch size to increase by up to 7× and 12×, respectively, resulting in overall throughput improvements of 3.4× and 4.6× compared to the original setup.

**S18.5**（supporting；未進入驗證（單一 agent 擷取））

作者在 LLaMA-3-8B / PG19（截到 8196 tokens）上量到：注意力高度稀疏，0.5% 的 key 就涵蓋 90% 的注意力，但哪些 key 重要會隨 query 改變。在相同稀疏度下，H2O 式按累積注意力做 greedy eviction 的 hit rate 遠低於逐 query 的 top-k。也就是說，過去的注意力重要性預測不了未來 query 需要哪些 KV。這在單一請求內佐證了我們的發現 (7)（attention-importance 對跨請求重用的 AUC≈0.5），也支持「保留可取回的全精度備份，不做不可逆 eviction」的設計。

> 原文：i) Attention is highly sparse. Only 0.5% of the keys can cover 90% of a query’s attention. ii) The sparsity of attention is query-dependent. While both methods enforce the same level of sparsity, the hit rate for greedy eviction is much lower than that of query-dependent top-k attention. This suggests that different queries tend to focus on distinct sets of keys.

## S19. https://github.com/vllm-project/vllm/issues/55434

- 角度：B: KV transfer software overhead, disaggregated P/D, CXL and GPUDirect Storage tiers
- 來源性質：forum；日期：2026-09-05
- ✅ 抽查：作者 stu-cao，2026-09-05，Open、無 maintainer 回覆；91k–120k descriptors／~4.9 GB、cuda_ipc 4.067 µs vs rc_x IB 0.204 µs、RoCE 0.352 µs、10.3k→41.1k token 時 descriptor 恆為 101、45 KiB 2.4 GB/s vs 連續 778.6 GB/s——皆一致

**S19.1**（central；未進入驗證（單一 agent 擷取））

在 GB200 上以 P/D disaggregation 服務 GLM-5.3（DeepSeek-V3.2 架構：MLA + DSA indexer；vLLM 0.28.0+cu129、nixl 1.3.2）時，NIXL KV transfer 的瓶頸是 descriptor submission，不是 fabric 頻寬。負載是 AgentX C64 agentic load，prefix hit 88–92%。每次 TP-rank transfer 搬約 4.9 GB，要發出 91,000–120,000 個 descriptor；走 cuda_ipc（MNNVL）時，56.9% 的傳輸時間花在 posting descriptor 上。這在另一條路徑（P/D 的 NIXL）和另一種硬體上，獨立印證了我們的發現 (2)：KV 搬移成本由軟體路徑的 per-descriptor 固定成本決定，而不是由硬體頻寬決定。注意：這是單一回報者的量測，沒有經過審查。

> 原文：Under agentic load we observe 91,000–120,000 descriptors per TP-rank transfer for a ~4.9 GB payload, and cuda_ipc spends 56.9% of transfer time posting them.

**S19.2**（central；未進入驗證（單一 agent 擷取））

同一台機器上，各 transport 的 per-descriptor submission 成本差到約 20 倍：cuda_ipc 4.067 µs，rc_x IB 0.204 µs，rc_x RoCE 0.352 µs。所以名義上較快的 MNNVL/NVLink 路徑，端到端反而比 4-rail IB RDMA 慢。每次 rank-transfer 的平均總時間是 cuda_ipc 800.8 ms（其中 post 455.9 ms），IB 4 rails 350.8 ms（其中 post 24.4 ms）。這推翻了「依峰值頻寬替 tier 或鏈路排序」的假設：成本模型每條 transport 都需要一個 per-descriptor 項，κ 類的常數也必須對每條路徑分別量測。

> 原文：The consequence: for this workload shape, the documented GB-series MNNVL configuration is slower end to end than plain RDMA on the same hardware, because cuda_ipc submission costs ~20× more per descriptor than rc_x.

**S19.3**（central；未進入驗證（單一 agent 擷取））

descriptor 數量等於「架構放大係數」乘以「workload 造成的 block 碎片化」。放大係數是 101 個註冊的 KV regions（78 層 MLA 加上 DSA indexer）：block list 裡每一段不連續的 run，每個 region 各要一個 descriptor。碎片化來自 agentic 長上下文大量命中散落在 pool 各處的 prefix-cache blocks。context 長度本身不會讓 descriptor 變多：新配置的 block 是連續的，prompt 從 10.3k 放大到 41.1k tokens，payload 變成 4 倍，descriptor 仍然剛好是 101 個。這揭露了一個隱藏的耦合：prefix 重用率越高，實體 layout 越碎，descriptor 越多，傳輸越慢。這推翻了「傳輸成本只是 bytes/tokens 的函數」的假設，和我們的發現 (4)（prefix-only 命中語意）直接相關，也可以支撐一個新方向：contiguity/fragmentation-aware 的 KV 配置與 compaction。

> 原文：Quadrupling the prompt from 10.3k to 41.1k tokens quadrupled the payload and left the descriptor count at exactly 101 — one per region, because freshly allocated blocks are contiguous and coalesce.

**S19.4**（supporting；未進入驗證（單一 agent 擷取））

PyTorch NVLink peer-copy microbenchmark 顯示，頻寬在 KV-block 粒度下崩潰。單一段連續的 4.29 GB copy 可達 778.6 GB/s。切成 45 KiB（93,206 次 copy）只剩 2.4 GB/s，1 MiB（4,096 次）是 53.1 GB/s，256 MiB（16 次）是 740.9 GB/s。以下兩點是本次從來源數字推算的，原文沒有寫：(i) 四個資料點都符合 t ≈ n×~19 µs + bytes/~779 GB/s，也就是每次 copy 約 19–20 µs 固定成本（45 KiB 點約 19.2 µs，1 MiB 點約 19.7 µs），模型形式和我們在 MI300X OffloadingConnector 量到的約 13 µs per-descriptor 固定成本相同；(ii) 實際傳輸中每個 descriptor 平均約 41–54 KB（4.9 GB ÷ 91k–120k），和 45 KiB 的測試粒度同一量級。

> 原文：At 45 KB granularity NVLink delivers 0.3% of what it does contiguously.

**S19.5**（supporting；未進入驗證（單一 agent 擷取））

回報者提出、但尚未量測的修正有三項：(1) 改用 vLLM 既有的 _register_packed_kv_cache 做 packed 單一 region 註冊，預估把最壞情況從約 112,000 個 descriptor 降到約 1,100 個（約 100 倍）；(2) 採用能讓同一 sequence 的 blocks 保持連續的 allocation policy，或改善 coalescing；(3) 和 NIXL/UCX 一起檢查 cuda_ipc 的 submission 路徑，看能否批次化。這對新方向是 novelty 風險：upstream 已經在 layout 層處理這個問題，論文貢獻必須超越它，例如把 prefix 重用和實體連續性一起最佳化，或做含 per-run descriptor 項的放置成本模型。截至抓取時（2026-09-27），issue 沒有任何留言，也沒有 maintainer 確認。

> 原文：With one region, even a fully fragmented request costs one descriptor per block rather than 101, taking our worst case from ~112,000 to roughly 1,100

## S20. https://vllm.ai/blog/2026-01-08-kv-offloading-connector

- 角度：B: KV transfer software overhead, disaggregated P/D, CXL and GPUDirect Storage tiers
- 來源性質：primary；日期：2026-01-08
- ✅ 抽查：作者 Or Ozeri、Danny Harnik（IBM Research），2026-01-08；「increased the physical block size by a factor of 2*num_layers」、Llama-3.1-8B 32 KB→2 MB、H100 上 TTFT 最多降 4×／吞吐 5×、2 MB 雙向 DMA 83.4 GB/s vs kernel 68.5 GB/s、hybrid 模型「currently not optimized」、storage 階是下一個里程碑——皆與擷取一致

**S20.1**（central；未進入驗證（單一 agent 擷取））

vLLM 0.12.0 changed the KV cache layout so that each 16-token logical block holds the K and V of every layer in one contiguous physical block. This made the GPU-CPU transfer unit 2 x num_layers larger: Llama-3.1-8B-Instruct went from 32 KB to 2 MB, Llama-3.1-70B-Instruct from 8 KB to 1.25 MB, and Qwen2.5-7B-Instruct from 16 KB to 0.87 MB. On the same H100 hardware, OffloadingConnector throughput rose by about an order of magnitude (for Llama-3.1-8B vs 0.11.0: up to 4x lower TTFT and 5x higher throughput). This is first-party evidence that effective KV-offload bandwidth is set by the software layout and transfer granularity, not by the PCIe link. It is the closest prior work to our 'software path, not hardware' finding. However, the post gives no model of fixed cost per transfer or per descriptor (e.g., microseconds per descriptor).

> 原文：This change effectively increased the physical block size by a factor of 2*num_layers, and this in turn increased the throughput of the offloading connector by an order of magnitude.

**S20.2**（central；未進入驗證（單一 agent 擷取））

The authors ran an H100 microbenchmark: one transfer of 1000 blocks, with block sizes from 4 KB to 16 MB. cudaMemcpyAsync (copy-engine DMA) throughput depends strongly on block size and is good only for large blocks. An SM-based custom copy kernel (copies 16-byte words through raw pointers) is much faster for small blocks but noisier. Both peak at about 50 GB/s in one direction. With 2 MB blocks in both directions, DMA reaches 83.4 GB/s and the kernel 68.5 GB/s. The CPU backend is said to support 'CUDA-compatible devices (NVIDIA and AMD)', but every number is from H100 only. Transfer overhead on AMD/ROCm (e.g., hipMemcpyAsync on MI300X) is not characterized.

> 原文：The results confirm that DMA performs well, but only for larger block sizes. For smaller block sizes, the custom kernel achieves significantly better throughput.

**S20.3**（supporting；未進入驗證（單一 agent 擷取））

The copy mechanism competes with model compute. Setup: H100, Llama-3.2-1B-Instruct (0.5 MB physical blocks), 10,000 concurrent 512-token prefill requests. DMA beats the SM-based copy kernel by about 5.5% at a 0% CPU-hit rate, rising to about 15% at 80%. At a 0% hit rate, the SM-based kernel gives 6% lower throughput than running with no offloading at all. For Llama-3.1-8B, DMA gets up to 32% more throughput than the kernel at equal TTFT. Implication: a KV-tier cost model needs a term for interference with compute, not just bytes divided by bandwidth.

> 原文：For 0% hit rate, the custom kernel approach actually yields 6% worse throughput than without using CPU offloading at all.

**S20.4**（supporting；未進入驗證（單一 agent 擷取））

The measured benefit of CPU KV offloading is mainly throughput under concurrency and preemption, not single-request latency. On H100 with Llama-3.1-8B-Instruct, loading KV from CPU cuts single-request TTFT by 2x to 22x, depending on prompt size. With 10,000 prefill requests of 512 tokens, throughput rises by up to 9x, while TTFT at that prompt size drops only 2x. Stores are asynchronous, so cache misses see little TTFT impact. The authors also present offloading as a replacement for recompute on preemption (it avoids re-running prefill for evicted requests). All results were measured with GPU prefix caching disabled.

> 原文：We observe that the throughput increases by up to X9, even though TTFT for this prompt size only decreased by X2. This demonstrates that the major gain in KV cache offloading is throughput maximization.

**S20.5**（supporting；未進入驗證（單一 agent 擷取））

As of this post (January 2026), vLLM's native OffloadingConnector was optimized only for uniform models, where every layer's KV has the same shape. Hybrid models (e.g., mixing SSM or linear-attention layers with attention) were explicitly not optimized. A storage/SSD tier behind the CPU cache was still a future milestone. So offloading hybrid-model state and a native GPU-to-CPU-to-SSD path were not characterized in this system at that time.

> 原文：vLLM also supports hybrid models, which are currently not optimized for the offloading connector. [...] Our next milestone is enabling the CPU KV cache to act as an intermediate tier for storage offloading.

## S21. https://llm-d.ai/blog/networking-for-distributed-inference-llm-d

- 角度：B: KV transfer software overhead, disaggregated P/D, CXL and GPUDirect Storage tiers
- 來源性質：blog；日期：2026-06-23

**S21.1**（central；未進入驗證（單一 agent 擷取））

On the same hardware (two H200 nodes, 400 Gb/s Quantum-2 InfiniBand, 1–8 GB payloads, 1,000 descriptors per transfer), the NIXL backend's software posting path sets the KV-transfer bandwidth. UCCL and UCX reach about 49.5 GB/s (about 99% of line rate), but Mooncake tops out at about 42 GB/s (about 84%). The authors attribute the gap to Mooncake's per-block posting latency: about 2–17 ms of accumulated post time per batch, against about 12 µs for UCCL and about 283 µs for UCX. Dividing by the 1,000 descriptors per batch gives roughly 2–17 µs, 0.01 µs and 0.28 µs per descriptor (my arithmetic, not stated in the post). At 10,000 descriptors per transfer, UCX and UCCL still hold about 49.5 GB/s while Mooncake stays at about 38–42 GB/s. Mooncake also had to be patched before its topology detection picked the right NIC. Caveat: these are the authors' own measurements, not peer-reviewed, and the authors contributed the UCCL backend.

> 原文：When we move to the 400 Gb/s fabric and keep blocks relatively large (1–8 MB, batch of 1000), UCCL and UCX both achieve ~49.5 GB/s (≈99% of the 50 GB/s line rate). We had to patch Mooncake to fix its topology detection in this cluster to ensure that it picks the right interface. Mooncake saturates at ~42 GB/s, roughly 84% of line rate — the gap is consistent with its higher per-block posting latency for read operations (~2–17 ms of accumulated post time per batch, versus ~12 µs and ~283 µs for UCCL and UCX respectively).

**S21.2**（central；未進入驗證（單一 agent 擷取））

Whether software overhead is visible depends on link speed. On 100 Gb/s RoCE (two H100 nodes, ConnectX-7), all three backends (UCCL, UCX, Mooncake) converge to 12.0–12.2 GB/s, within 3% of the 12.5 GB/s line rate, at both 1,000 and 10,000 descriptors per transfer. At this speed the link is the bottleneck and the choice of backend does not show up in bandwidth. Backend differences appear only on the faster 400G InfiniBand fabric.

> 原文：On the 100GbE RoCE network, UCCL, UCX, and Mooncake all converge to roughly 12.0–12.2 GB/s — within 3% of the 12.5 GB/s theoretical line rate (without considering packet header overheads) for both batch sizes as shown in Figure 2. At this network speed, we can easily saturate the link and the choice of backend is effectively invisible in the bandwidth numbers, with only minor differences in latency.

**S21.3**（supporting；未進入驗證（單一 agent 擷取））

Over 100G TCP on the same NICs, GPU-to-GPU transfer time differs about 9x by backend. UCCL moves 1 GB in 211 ms on average (P99 217 ms, about 4.7–4.9 GB/s). UCX takes 1.8 s for the same transfer (about 0.55 GB/s) and 14.4 s at 8 GB. Mooncake does not support GPU-to-GPU over TCP. UCCL's advantage comes from software: it chunks the transfer and pipelines the GPU-to-CPU staging with the TCP send, using NCCL CUDA kernels. For CPU-to-CPU at 1 GB, UCCL takes 216 ms, UCX 263 ms and Mooncake 298 ms.

> 原文：UCCL transfers 1 GB in 211 ms (avg) with a P99 of 217 ms — stable and consistent. UCX takes 1.8 s for the same transfer, nearly 9× slower, degrading further to 14.4 s at 8 GB (Figure 3). While UCX does not provide specific optimizations for GPU-to-GPU transfers over TCP, UCCL leverages NCCL's CUDA kernels in their send/recv APIs, which chunk a single transfer into multiple transfers for GPU-to-CPU and pipeline the network transfer over TCP.

**S21.4**（central；未進入驗證（單一 agent 擷取））

The post measures transfer software overhead only on network point-to-point paths: GPU-to-GPU over RDMA and TCP, and CPU-to-CPU over TCP. All tests use bulk 1–8 GB payloads split into 1,000 or 10,000 descriptors, which works out to about 0.1–8 MB per descriptor (my arithmetic from the stated payload and batch sizes). Not measured: GPU-to-host DRAM offload, GPU-to-SSD offload, and small KB-scale per-page descriptors. The authors themselves say that fast transfer paths for KV management that offloads to CPU memory are still being developed.

> 原文：Considering KV Cache Management: KV Cache management strategies that involve CPU memory offloading require efficient heterogeneous transfer paths, which are an active area of development.

**S21.5**（supporting；未進入驗證（單一 agent 擷取））

In vLLM's NixlConnector, which llm-d uses for prefill/decode disaggregation, the decode worker pulls KV blocks with one-sided RDMA READs. The post describes recomputation only as a failure fallback, set by kv_load_failure_policy: the default 'fail' returns an error, and 'recompute' falls back to local prefill on the decode worker. The post does not describe any per-request, cost-based choice between transferring KV and recomputing it.

> 原文：D-side KV load failures (e.g. a crashed P instance) are handled by the kv_load_failure_policy: fail (default) returns an error immediately, while recompute falls back to local prefill on D.

## S22. https://rocm.blogs.amd.com/software-tools-optimization/amd-infinity-context/README.html

- 角度：B: KV transfer software overhead, disaggregated P/D, CXL and GPUDirect Storage tiers
- 來源性質：blog；日期：2026-07-22

**S22.1**（central；未進入驗證（單一 agent 擷取））

AMD ROCm Infinity Context (AIC) 在 KV cache 階層中、GPU HBM 與 CPU DRAM 之下新增一層 RDMA 網路附加儲存（初版後端為 NFS over RDMA）。KV 經 ROCm hipFile（2026-07 隨 ROCm 7.14 GA）直接從儲存搬進 GPU HBM，不經 host DRAM。堆疊分工：llm-d 負責路由、P/D 放置與 KV locality；vLLM 為 serving 層；LMCache 決定 block 放在 HBM、DRAM 或遠端層；NIXL 負責傳輸，同時支援一般檔案式本地 NVMe 路徑，以及 P/D disaggregation 所需的跨節點 KV 遷移。MI300X Series 經 front-side Ethernet 標示為 Supported，Tech Preview 排在 2026-09 的 Fall ROCm release。這是廠商對自家產品架構的第一手描述，平台 A/B 可以直接對照：GPU-direct 路徑對上 vLLM OffloadingConnector 的 host-bounce 路徑。

> 原文：ROCm AIC provides an additional tier that sits below GPU HBM and CPU DRAM in the KV cache hierarchy … AMD Instinct MI300X Series and MI350 Series GPUs via front-side Ethernet network attach … It supports standard file-based NVMe paths as well as AMD Infinity Storage (AIS) object store paths. … Data moves from networked storage directly into GPU HBM without passing through host CPU memory.

**S22.2**（central；未進入驗證（單一 agent 擷取））

文中宣稱從儲存層取回 KV 只要重算時間的「a fraction」，但全文沒有任何量測數據：沒有 TTFT、GB/s 頻寬、延遲、每 block 或每 descriptor 的軟體開銷、命中率，也沒有 fetch 與 recompute 損益兩平時的 context 長度。TTFT 效益只用「may be achievable」這種保留語氣帶過，benchmark 則推給 github.com/ROCm/rocm-aic 與後續文章。也就是說，至少在這篇廠商公告裡，GPU-direct 遠端 KV 層在 MI300X 上的傳輸軟體路徑開銷還沒有人刻畫過；這一點可以直接量測來證偽。

> 原文：The net result: requests that would have required seconds of GPU compute to recompute KV context can instead be served from the storage tier in a fraction of the time, enabling dramatically lower Time to First Token (TTFT) and higher concurrency at the same GPU footprint. … For shared-context workloads with high cache hit rates, substantial TTFT improvements may be achievable.

**S22.3**（supporting；未進入驗證（單一 agent 擷取））

AIC 的查找語意以 prefix 為鍵，只在位置維度上逐層串接：先查 HBM 和 DRAM，沒命中才去遠端層抓，由 LMCache 以 prefix-aware caching 管理。全文完全沒提到精度層（FP8/INT4）、quantize-on-offload 或 compress-on-offload，也沒提到非 prefix 命中（segment 或與位置無關的 RAG 命中）。關鍵字檢索 quant/FP8/INT4/compress/precision/segment 的出現次數都是 0。但文中又把「RAG 文件 KV 建一次、跨所有查詢共享」列為主要用例，卻沒說明 prefix-only 查找怎麼處理不在前綴位置的文件 KV。換句話說，廠商最新的 KV 階層仍把 KV 當成不透明的位元組，只做位置放置。

> 原文：LMCache tracks block identifiers, handles prefix-aware caching, and orchestrates reuse across requests and across nodes. … LMCache checks whether the required KV prefix is already in GPU HBM or CPU DRAM. If not, it initiates a fetch from the ROCm AIC tier. … RAG pipelines index documents into KV cache once and share them across every query against the same corpus.

**S22.4**（supporting；未進入驗證（單一 agent 擷取））

AMD 主張本地 NVMe 卸載受限於單一節點（node-bound），KV block 無法有效跨節點共享；又主張把原本規劃給本地 NVMe 的 BOM 預算改投共享網路儲存層，就能「dramatically increase cache hit rates」。但文中沒有任何命中率數據，也沒有跟「本地 NVMe + llm-d cache-aware routing」這個對照組比較過；而依文中自己的描述，llm-d 本來就會把請求導向已持有相關 KV 的 replica。這個主張可以用 trace 模擬（例如 Mooncake）來證偽。

> 原文：The common workaround of offloading KV cache to local NVMe via host CPU and RAM works on a single node, but KV blocks cannot be efficiently shared across GPU nodes. … moving this to a dedicated networked storage tier with ROCm AIC can dramatically increase cache hit rates across distributed inference workloads.

**S22.5**（supporting；未進入驗證（單一 agent 擷取））

文中兩個關鍵的量化動機數字都無法從文中重現。第一，HBM3e 在系統層級約 $50/GB，腳註只寫依據是「AMD internal calculations」。第二，單一 1M-token 請求可產生超過 600 GB 的 KV，但沒說是哪個模型、哪種 dtype，也沒說是否考慮 GQA 或 sliding-window；文中舉的例子是 GPT-OSS-120B 與 Qwen3-235B。NVMe 網路儲存的 $/GB 只用「a fraction」帶過，全文也沒提到 SSD 寫入耐久或 RDMA NIC 成本（endurance/write/wear 出現 0 次）。這些數字可以當 cost($)-aware 放置分析的單一廠商參數，但必須找獨立來源交叉驗證。

> 原文：Even a single 1M-token request can generate over 600 GB of KV data, more than the HBM capacity of an entire MI455X node. … At the time of the writing, this blog, HBM3e costs approximately $50/GB at the system level.[1]

## S23. https://arxiv.org/abs/2607.01831

- 角度：B: KV transfer software overhead, disaggregated P/D, CXL and GPUDirect Storage tiers
- 來源性質：primary；日期：2026-07-02
- ✅ 抽查：Lynx，Han/Yeung/Barletta/Toner/Hoste/Barker，2026-07-02，註明原投 SIGCOMM'26；摘要的 1.43×、5.1%、Anchor/Residual 描述一致

**S23.1**（central；未進入驗證（單一 agent 擷取））

Lynx 推翻「KV cache 是必須完整收到才能使用的不可分割單位」這個假設。它把 INT8 量化後的 KV 依位元拆成兩條 stream：高優先的 Anchor（4-bit MSB）與低優先的 Residual（4-bit LSB）。Anchor 一到，decode 就以 speculative 方式開始；Residual 到齊後再驗證，保證輸出等同高精度 decode。論文宣稱 TTFT 接近 INT4，準確度與 BF16 相當；TTFT 比 INT8 最多快 1.43×，準確度比 SOTA（CacheGen）最多高 5.1%。本研究可延伸之處（屬推論）：BF16/FP8/INT4 精度階層不一定要是互斥的整份副本，也可以是拆到不同儲存層的 bit-plane，例如 MSB 放 HBM、residual 放 CPU/SSD。但 Lynx 只評估了 disaggregated P/D 的跨伺服器傳輸。此篇為預印本，作者註明原投 SIGCOMM '26。

> 原文：We challenge the assumption that the KV cache is an indivisible unit that must be fully received before use. ... Lynx achieves Time-to-First-Token (TTFT) comparable to aggressive 4-bit KV quantization, while matching the accuracy of high-precision (BF16) inference, improving TTFT over standard 8-bit KV quantization by up to 1.43× and improving accuracy over state-of-the-art by up to 5.1%.

**S23.2**（central；未進入驗證（單一 agent 擷取））

Lynx 把傳輸時間建模成「payload ÷ 鏈路頻寬」。動機部分用算術推得：Qwen3-235B 在 128K context 下的 BF16 KV 約 23.5 GB，走 100 Gbps TCP/IP 約需 2 秒。實驗環境是 2 台 Atlas A2（Ascend 910B4）加 vLLM-Ascend + LMCache-Ascend，並在 KV connector 上加 rate-limiter，把頻寬人為限制在 10–50 Gbps。也就是說，瓶頸是實驗施加的，不是量出來的。論文沒有量測 connector 軟體路徑本身的開銷，例如 per-descriptor／per-chunk 成本，或實得頻寬與鏈路頻寬的落差；序列化成本只說「被 pipeline 隱藏」。這正對應本研究發現 (2)：頻寬由軟體路徑決定。推論（未經檢驗）：拆成兩條 stream 會讓 descriptor 數加倍，在每個 descriptor 有固定成本（如 MI300X 上約 13 µs）的路徑上，收益可能被吃掉。

> 原文：On a standard 100Gbps TCP/IP interconnect, transferring this 23.5 GB payload requires roughly 2 seconds. ... To emulate the network bottlenecks characteristic of long context scenarios (e.g., >1M tokens) within our testbed's memory constraints, we implement a rate-limiter on the KV connector. We restrict inter-server bandwidth from 10Gbps to 50Gbps.

**S23.3**（supporting；未進入驗證（單一 agent 擷取））

頻寬越高，Lynx 相對 INT8 的收益縮得越快：TT64T 的優勢在 10 Gbps 是 0.86 s，到 50 Gbps 只剩 0.18 s。作者也承認，最大加速只出現在「低頻寬 + 長 context」的情境。推論：搬到 CPU DRAM 階層時（PCIe 頻寬遠高於 50 Gbps）收益可能消失；反倒是實測只有 raw NVMe 約 1/18 的 3090 SSD 階層，才可能落在它的甜蜜點。

> 原文：Lynx consistently achieves lower TT64T than INT8 across the range of 10 to 50Gbps bandwidth, though arguably such gain shrinks from 0.86s to 0.18s for 10 and 50Gbps respectively. In general, Lynx yields maximum speedups under the scenario of low bandwidth and long context.

**S23.4**（supporting；未進入驗證（單一 agent 擷取））

只用 MSB（Anchor）KV 做 speculative decode，產出的 token 多半與高精度 KV 相同。在 MMLU + Qwen workload 上，平均產生 21.43 個 speculative token、接受 19.38 個，且有 64.8% 的機率整段全數被接受。驗證失敗時只修正第一個分歧的 token，不需重算。推論：低精度階層可以當「draft KV」，全精度留在慢層只負責驗證。這樣一來，量化誤差 ε 從永久的品質損失變成延遲／重做成本，是 quality-aware 放置可以加入的新動作語意。

> 原文：In the MMLU Qwen workload, we find that the model generates on average 21.43 speculative tokens, of which 19.38 tokens are accepted; with 64.8% probability, the whole sequence of the speculative tokens is fully accepted. ... Lynx identifies the longest prefix of tokens, accepting valid tokens and correcting the first divergence without re-computation.

**S23.5**（supporting；未進入驗證（單一 agent 擷取））

在 MMLU + Qwen（實驗模型為 Qwen3-32B）上，直接用 INT4 KV 讓準確度下降 8.7%；Lynx-INT4 降 1.7%，CacheGen 降 5.1%。相對地，INT8 與 Lynx（4-bit anchor + 4-bit residual，等效 INT8）和 BF16 的差距都在 ±0.3% 以內。這獨立佐證了本研究發現 (3)：ε 取決於 model×task，也取決於量化方法。「全域 INT4 取代放置」這個結論，對部分模型需要加上品質條件或改成逐模型選精度。

> 原文：INT4, Lynx-INT4 and CacheGen bring accuracy drops of 8.7%, 1.7% and 5.1% respectively on MMLU + Qwen. ... The accuracy of INT8, Lynx, and Lynx-INT8 are of little statistical difference (i.e., within the ±0.3% range) from BF16's accuracy. ... Lynx's prototype uses INT4 for both the anchor and residual streams, yielding an effective INT8 representation.

## S24. https://arxiv.org/abs/2502.07776

- 角度：D: Cost-, SLO- and carbon-aware KV caching, and prompt-caching economics
- 來源性質：primary；日期：2025-02-11
- ✅ 抽查：Gu/Li/Kuditipudi/Liang/Hashimoto，"Auditing Prompt Caching in Language Model APIs"，ICML 2025；7 家全域共享——一致（「per-user 快取保留多數效益但未量測」是內文，未抽查）

**S24.1**（central；未進入驗證（單一 agent 擷取））

Timing audits were run in September and early October 2024 on 17 LLM API providers. They used a one-sided two-sample KS test with alpha = 1e-8, 250 timing samples per procedure and 5,000-token prompts. Prompt (KV-prefix) caching was detected in 8 providers, and 7 of them shared the cache globally across users. The 7 were Azure and OpenAI text-embedding-3-small, plus the Llama-3/3.1-8B APIs on Deep Infra, Fireworks, Lepton, Perplexity and Replicate. Anthropic Claude 3 Haiku and OpenAI GPT-4o mini shared only per organization. Hit-versus-miss timing could be classified with average precision 0.70–1.00, mostly about 0.8. Relevance (our inference, not tested by the paper): in production, the sharing scope of a reusable KV cache (per-user, per-org or global) is a security axis, not just a hit-rate knob. A multi-tier cache (HBM / DRAM / SSD / recompute) could leak which tier holds an entry, not only whether it hit or missed.

> 原文：We conducted audits on real-world LLM API providers in September and October 2024. We detected prompt caching in 8 out of 17 API providers. In 7 of these providers, we detected global cache sharing. On these APIs, an attacker could, in principle, detect cache hits from timing differences to infer that another user sent a prompt that shares a prefix with a given prompt.

**S24.2**（central；未進入驗證（單一 agent 擷取））

The paper's main mitigation is to allow only per-user caching. It asserts that per-user caching keeps 'many of the performance benefits' of global sharing, because different users rarely share long prefixes. The paper never measures this: it has no hit-rate, trace or throughput analysis. So the hit-rate and TTFT cost of each cache-isolation level (per-user vs per-org vs global) is an open question that can be tested on multi-tenant traces that carry user or tenant IDs.

> 原文：To completely prevent any privacy leakage from prompt caching, only per-user caching should be allowed. In per-user caching, an attacker will not be able to produce cache hits on prompts sent by other users. Since it is unlikely that different users will send prompts with long matching prefixes, per-user caching should retain many of the performance benefits from global cache sharing.

**S24.3**（supporting；未進入驗證（單一 agent 擷取））

In OpenAI's text-embedding-3-small API, the same prompt returned a slightly different embedding on fast responses (cache hits) than on normal-speed responses (misses), off by about 1e-4 to 1e-5 per coordinate. The authors only hypothesize the cause: the reused KV cache may be stored at lower floating-point precision than freshly computed KV. They also saw a third embedding variant on some slow responses, which they attribute to different GPU models. Relevance: this is production evidence (inferred, not confirmed) that reused KV may be stored at reduced precision, and that hit and miss outputs are not bit-identical. That bears on precision-tiered KV (BF16/FP8/INT4) and on determinism guarantees across hits and misses.

> 原文：These differences are small, on the order of 10^-4 to 10^-5 in each coordinate. We hypothesize that these differences may arise if the reused KV cache is stored in a lower floating-point precision, resulting in slight discrepancies when the attention KV is computed from scratch in cache misses versus when it is retrieved from the cache in cache hits.

**S24.4**（supporting；未進入驗證（單一 agent 擷取））

How well a cache hit can be told apart from a miss by timing depends on prompt length. On the Fireworks, Perplexity and Replicate Llama APIs, average precision is high and stable for prompts of about 1,000 tokens or more. It drops to random chance as the prompt length, or the fraction of the prefix that matches, shrinks. In most APIs a single victim request was enough to detect caching. Only the OpenAI and Azure embedding APIs needed 25, which the authors attribute to multiple servers with separate caches and random routing. Relevance: the timing side channel is strongest in the long-context regime that tiered KV systems target.

> 原文：When the PromptLength is moderately high ( ≳ 1000 ), the average precision is relatively high and stable. However, as the PromptLength approaches zero, the average precision decreases to random chance.

**S24.5**（supporting；未進入驗證（單一 agent 擷取））

The paper proposes another mitigation: deliberately delay cache-hit responses so they look like misses. The provider still saves GPU processing time, but users lose the latency benefit, so caching's value becomes GPU time and cost rather than TTFT. Relevance (our inference): if hit latency is padded anyway, which tier serves a hit (HBM vs DRAM vs SSD) no longer matters for user-visible latency, only for GPU time and dollars. That turns a latency-driven placement objective into a cost-driven one.

> 原文：Another potential mitigation is to intentionally delay the response time for cache hits so that they look like cache misses. This eliminates the benefits of prompt caching for users, but API providers could still benefit, as cached prompts require less GPU processing time.

## S25. https://platform.claude.com/docs/en/build-with-claude/prompt-caching

- 角度：D: Cost-, SLO- and carbon-aware KV caching, and prompt-caching economics
- 來源性質：primary；日期：未標示（持續更新的線上文件，頁面沒有發布或更新日期）；內文引用 beta header inline-tools-2026-09-15，所以此版本不早於 2026-09-15；擷取於 2026-09-27

**S25.1**（central；未進入驗證（單一 agent 擷取））

Anthropic 官方的 cached-token 定價：5 分鐘 TTL 寫入 = 1.25× base input，1 小時 TTL 寫入 = 2× base input，cache hit/refresh 的標準價是 0.1× base input。最新一代把 hit 折扣再壓低到 0.05×（〔某 Claude 模型〕）和 0.025×（〔某 Claude 模型〕 / 〔某 Claude 模型〕）。定價表裡 〔某 Claude 模型〕 與 〔某 Claude 模型〕 的 base（$10/MTok）、5m 寫入（$12.50）、1h 寫入（$20）、output（$50）完全一樣，只有 hit 從 $1 降到 $0.25/MTok。換算下來，公開的『未命中（重算）: 命中（重用）』價比從 10× 拉大到 40×，寫入溢價沒變（倍數是依表中數字換算的）。（註：這個 URL 只有 Anthropic 的價格；標題提到的 OpenAI/DeepSeek/Gemini 價格不在此頁，本次沒有驗證。）

> 原文：5-minute cache write tokens are 1.25 times the base input tokens price … 1-hour cache write tokens are 2 times the base input tokens price … Cache read tokens are 0.1 times the base input tokens price (see the table footnote for per-model exceptions) … Cache hits and refreshes on 〔某 Claude 模型〕 and 〔某 Claude 模型〕 are priced at 0.025x the base input price. … Cache hits and refreshes on 〔某 Claude 模型〕 are priced at 0.05x the base input price.

**S25.2**（central；未進入驗證（單一 agent 擷取））

Anthropic 聲明 prompt cache 的 KV 表示與內容 hash 只放在記憶體、不落地（not stored at rest）。TTL 是『最低』保證壽命（標準 5 分鐘、延長 1 小時），到期後會 promptly（但不是立即）刪除。快取在組織之間隔離，即使 prompt 完全相同也不共享；在 Claude API / Claude Platform on AWS / Microsoft Foundry 上還細到 workspace 級。（推論，非原文：至少有一家主要供應商的生產 prompt cache 不使用持久化 SSD tier，而且保留語意是『保證下限、到期即刪』的 TTL，不是容量驅動的 LRU。這樣一來，合規（ZDR）與租戶隔離就成了 tier 選擇與可達 hit rate 的外生約束；用全域跨使用者共享的假設去跑 trace，可能會高估 hit rate。）

> 原文：KV (key-value) cache representations and cryptographic hashes of cached content are held in memory only and are not stored at rest. Cached entries have a minimum lifetime of 5 minutes (standard) or 1 hour (extended), after which they are promptly, though not immediately, deleted. Cache entries are isolated between organizations and, on the Claude API, Claude Platform on AWS, and Microsoft Foundry, between workspaces within an organization. … Different organizations never share caches, even if they use identical prompts.

**S25.3**（central；未進入驗證（單一 agent 擷取））

文件保證 prompt caching 不影響輸出：命中快取的回應和不使用快取時『identical』。（推論，非原文：在這個契約下，供應商不能只對被快取的 prefix 使用有損的 KV 表示，例如 cache tier 存 FP8/INT4、未命中路徑用 BF16 計算，否則命中與未命中的輸出會不同。有損精度只能是命中與否都一致的全域選擇。這和本專案『global INT4 可取代 placement』的發現相容，但對『per-request / per-tenant 精度分層』以及『cache tier 降精度』構成產品契約層級的限制。可以在任何有 prefix cache 的系統上，用 temperature 0、同一請求有/無 cache_control 的 A/B 來檢驗。）

> 原文：Prompt caching has no effect on output token generation. The response you receive is identical to what you would get if prompt caching were not used.

**S25.4**（supporting；未進入驗證（單一 agent 擷取））

TTL 從寫入或讀取該 entry 的請求『開始』起算，生成時間也算在壽命裡：回應串流 4 分鐘的話，後續請求要在大約 1 分鐘內送出才會命中；每次命中會免費 refresh。另外 thinking 設定（mode、budget_tokens）會被 render 進 prompt，改了就會讓 message cache 失效。（推論，非原文：長輸出或 reasoning 請求會直接縮短 prefix KV 的可重用窗口，而推理設定本身也是 cache key 的一部分。這是 reasoning 模型與 KV 保留互相影響的一條具體生產規則。）

> 原文：By default, the cache has a 5-minute lifetime. The cache is refreshed for no additional cost each time the cached content is used. The lifetime is measured from the start of the request that writes or reads the cache entry, not from the end of its response. Time spent generating a response counts against the lifetime: if a response takes 4 minutes to stream, a follow-up request that reuses the same cached prefix must start within about 1 minute of that response completing. … The thinking configuration (mode, and `budget_tokens` in extended mode) is rendered into the prompt, so changing it always invalidates message blocks

**S25.5**（supporting；未進入驗證（單一 agent 擷取））

命中語意是用累積 hash 做精確 prefix 比對，由使用者標記，粒度粗。細節：每個 cache_control breakpoint 只寫一個 entry（到該 block 為止的 prefix hash），更早的位置不寫；讀取時每個 breakpoint 最多往回查 20 個 block；每個請求最多 4 個 breakpoint；內容必須 100% 相同（含圖片）；任何 breakpoint 有變動（例如更新 RAG 文件），該段和其後全部失效；最小可快取長度依模型不同，範圍 512–4,096 tokens（〔某 Claude 模型〕 是 4,096，〔某 Claude 模型〕 以後降到 512），不到門檻就靜默不快取；並行請求要等第一個回應開始後 entry 才能用。頁面沒有描述任何 non-prefix / position-independent 的重用機制，這對應本專案發現 (4) 的 prefix-only 限制，以及 CacheBlend 這一類的方向。（推論，非原文：新模型可快取的最短長度變小，小 entry 會變多，每個 entry/descriptor 的固定開銷因此更重要，和發現 (2) 有關。）

> 原文：Marking a block with `cache_control` writes exactly one cache entry: a hash of the prefix ending at that block. The system does not write entries for any earlier position. … The lookback window is 20 blocks. … You can define up to 4 cache breakpoints … Cache hits require 100% identical prompt segments, including all text and images up to and including the block marked with cache control. … Changes at any breakpoint invalidate that segment and everything after it, while earlier cached segments remain valid … 512 tokens for 〔某 Claude 模型〕, 〔某 Claude 模型〕, 〔某 Claude 模型〕, 〔某 Claude 模型〕, 〔某 Claude 模型〕, and 〔某 Claude 模型〕 … 4,096 tokens for 〔某 Claude 模型〕 and 〔某 Claude 模型〕 … For concurrent requests, note that a cache entry only becomes available after the first response begins.

## S26. https://arxiv.org/abs/2505.23970

- 角度：D: Cost-, SLO- and carbon-aware KV caching, and prompt-caching economics
- 來源性質：primary；日期：2025-05-29 (v1, titled 'EmbAdvisor: Adaptive Cache Management for Sustainable LLM Serving'); latest v3 2026-04-11 (University of Waterloo / Purdue; arXiv preprint, no venue listed)
- ✅ 抽查：現標題 "Cache Your Prompt When It's Green: Carbon-Aware Caching for Large Language Model Serving"（v1 名 EmbAdvisor），Tian/Sun/Ding/Liu，v3 2026-04-11；FR 電網平均減碳 ~15%、最高 25%、>90% 請求守住延遲——一致（內文細節未抽查）

**S26.1**（central；未進入驗證（單一 agent 擷取））

把 SSD 的 embodied carbon 攤提進碳會計後，KV caching 的淨效益會依電網碳強度 (CI) 變號。在 Llama-3 70B、1.5 prompts/s、16 TB SSD cache 的條件下：高 CI 的 MISO（485 gCO2e/kWh）減碳 7.5%，低 CI 的 FR（33 gCO2e/kWh）反而增碳 16.5%。同一電網內也會翻轉：CISO 一天內 CI 從 37 變到 232 gCO2e/kWh，快取在清晨 7 點不划算，晚上 8 點效益最大。這推翻了「命中就省」的隱含前提。對我們而言，「何時值得做 KV 放置/快取」的先驗判準（發現 6）可以再加兩個常數：每 TB 的 embodied carbon 與 CI。

> 原文：For instance, a 16 TB cache reduces carbon emissions by 7.5 % in MISO, where the CI reaches 485 gCO2e/kWh. In contrast, low-CI grids (on the left side) benefit less or even incur higher emissions from caching. For example, in FR, where the CI is only 33 gCO2e/kWh, the same cache increases carbon emissions by 16.5 %.

**S26.2**（central；未進入驗證（單一 agent 擷取））

GreenCache（v1 名為 EmbAdvisor）建在 LMCache 上，只調整 SSD 快取的「容量」這一個維度。做法：先 profile 不同 cache size × request rate 下的 TTFT/TPOT 與功耗；以歷史值預測 CI 與負載（最多提前 24 小時）；再用 ILP 在 TTFT/TPOT SLO 約束下每小時重設 cache 大小（1 TB 粒度、上限 16 TB、平均每次決策 7.03 s）。Llama-3 70B（4×L40）在 FR 電網平均減碳 15.1%、最高 25.3%，>90% 請求符合延遲 SLO；在高 CI 的 CISO 最多只減 6.91%。結論：把「carbon-aware KV cache 管理」當新方向的 novelty risk 高，至少 SSD 容量調整這一維已被佔據。

> 原文：Evaluations from real traces demonstrate that GreenCache achieves an average carbon reduction of 15.1 % when serving Llama-3 70B in the FR grid, with reductions reaching up to 25.3 %, while staying within latency constraints for > 90 % of requests. ... The cache has an allocation granularity of 1 TB, with a maximum size of 16 TB. We evaluate the ILP execution time in Section 6.4 — 7.03 s per decision on average, a low overhead compared to the hourly cache resizing frequency.

**S26.3**（central；未進入驗證（單一 agent 擷取））

範圍缺口（已對 v3 HTML 全文檢索）：(1) 評估只用 8k context，超出即截斷；(2) 快取媒介只有 SSD，DRAM、CXL、HDD 僅在註腳宣稱同一碳模型可套用，沒有實測；(3) 全文沒有出現 'quantiz'、'tier'、'endurance' 等詞。也就是說，論文沒有處理 KV 精度層（BF16/FP8/INT4）、HBM/DRAM/SSD 多層放置、DROP+重算動作，也沒有建模 SSD 寫入耐久度：壽命只當作 3–7 年的固定攤提；FairyWREN (OSDI'24) 有被引用，但寫入成本沒有整合進模型。Discussion 另指出 MoE 會降低 operational carbon，但 KV 儲存量不變，因此 embodied carbon 的比重被放大。尚未被佔據的差異化方向：長上下文 × 精度層 × 寫入預算的 carbon/$-aware 多層 KV 放置。

> 原文：Both models have a context window of 8k tokens. When the context goes beyond this limit, we truncate extra context like prior work ... The same carbon modeling also applies to other caching mediums, such as DRAM, CXL-attached memory, and HDD. ... Therefore, MoE models lower operational carbon emissions but amplify the significance of embodied carbon, making the optimizations provided by GreenCache more impactful.

**S26.4**（supporting；未進入驗證（單一 agent 擷取））

GreenCache 提出 Least Carbon Savings (LCS) 替換策略，分數為 Score = (#Token × #Hit)/(Size × Age)，本質上是 GreedyDual-Size-Frequency 型的 cost-aware eviction。與 LMCache 預設的 LRU 相比：小容量時 hit rate 最多高約 9 個百分點（TriviaQA α=0.7、4 TB：0.35 vs 0.26）；到 16 TB 時兩者幾乎相同（ShareGPT：0.71 vs 0.69）。這佐證我們的發現 1（vanilla LRU 已接近 oracle）：cost-aware eviction 的增益只出現在容量受壓的區間。

> 原文：Although LRU indicates similar performance as LCS with 16TB cache size, LCS demonstrates up to 9 % higher hit rate than LRU with smaller cache sizes.

**S26.5**（supporting；未進入驗證（單一 agent 擷取））

論文提供可引用的儲存層 embodied carbon 參數：其平台上 SSD 佔伺服器 embodied carbon 的 76.6%。Table 1 的數字為：最多 16 TB SSD ≈ 480 kgCO2e，4×L40 GPU 106.4 kgCO2e，512 GB DDR4 30.8 kgCO2e。模型採 ACT，預設 5 年攤提；敏感度分析掃過 30–90 kgCO2e/TB 與 3–7 年壽命（3 年壽命時最多省 11.9%；90 kgCO2e/TB 時最多省 25%）。意涵：INT4/FP8 精度層省下的位元組可以直接換算成 SSD/DRAM embodied carbon 的減少，讓『量化取代放置』的發現延伸到碳或 $ 的維度。限制：平台 A（3090）沒有分軌能耗計數器，operational 能耗結論不能在那裡做。

> 原文：In our platform, the embodied carbon from SSDs consumes 76.6 % of the total embodied carbon of the server, similar to the over 75 % fraction reported by a prior study

## S27. https://arxiv.org/abs/2607.19214

- 角度：D: Cost-, SLO- and carbon-aware KV caching, and prompt-caching economics
- 來源性質：primary；日期：2026-07-21 (v1); v2 2026-07-24
- ✅ 抽查：Khailo，"Keeping the Cache Warm Pays: Keepalive Economics for Agentic Workloads"，2026-07-21/24，單一作者 5 頁；12.5×、~4 分鐘間隔、~46/36 分鐘損益兩平、「gives LRU eviction nothing to rank」——一致

**S27.1**（central；未進入驗證（單一 agent 擷取））

Across four frontier APIs (100k-token prefix; efficacy matrix n=8/cell x 3 independent runs, retention sweep n=6/cell), production prompt/KV-cache retention lasts minutes and differs by provider. Anthropic has a hard 5-min TTL: warm at 300 s, gone by 360 s, 0/48 warm at 600 s. DeepSeek (measured via OpenRouter pinned to the DeepInfra backend, not DeepSeek's own disk-backed cache) is gone by 540-600 s. OpenAI decays gradually and is fully cold by 1800 s. Google is a 'routing lottery', 33-83% warm at every gap. So agent tool or approval pauses of minutes evict the prefix, while a client-side keepalive (replaying the prefix every 30 s) held 40/40 warm at 600 s on Anthropic. Caveats: single-author independent preprint; Google keepalive cells only n=3-7.

> 原文：Hard TTL (Anthropic): the baseline is warm through 300 s and evicted to 0/48 samples across the three runs at 600 s, while the keepalive holds 40/40 at the same gap [...] Anthropic falls off a cliff between 300 and 360 s (its 5-minute TTL, no grace period), DeepSeek collapses by 540–600 s, and OpenAI outlives its documented 5–10 minutes, decaying gradually to fully cold by 1800 s. Google never converges (a routing lottery, not a retention curve).

**S27.2**（central；未進入驗證（單一 agent 擷取））

The paper gives a closed-form rent-vs-re-prefill (ski-rental-like) model for cached-token pricing. Holding a prefix through idle I costs (I/tau+1)*r per input token, versus paying w once to let it die. Keepalive spend r/tau falls strictly as the ping interval tau grows, so the economical interval is TTL minus a margin: about 240 s for Anthropic and DeepSeek, about 480 s for OpenAI. The common 30 s convention spends about 8x more (about $3.60/h vs about $0.45/h for a 100k Anthropic prefix). Keepalive only pays inside a 'paying band' between the provider's eviction point and I_max = tau(w/r-1): about 46 min for Anthropic's 5-min tier (r=0.10, w=1.25), about 36 min for OpenAI and DeepSeek (w=1.0), and only about 12 min for Google (r=0.25).

> 原文：Keeping a prefix alive through an idle I costs (I/τ+1) r per input token; letting it die costs w once. [...] Keepalive spend per unit time is r/τ, strictly decreasing in τ [...] Break-even against re-prefill is I_max = τ(w/r−1): ≈46 min for Anthropic’s 5-minute tier at τ∗, ≈36 min for OpenAI and DeepSeek, and only ≈12 min for Google, whose implicit cache reads at 0.25× rather than 0.1×

**S27.3**（supporting；未進入驗證（單一 agent 擷取））

Measured whole-strategy savings are much smaller than the '12.5x' headline. By our arithmetic from the paper's Table 1, 12.5x equals Anthropic's per-request post-pause price ratio w/r = 1.25/0.10; the paper does not state this. At a 30-min pause the keepalive saved 1.56x on Anthropic (tau=240 s, n=8) and 2.45x on OpenAI (tau=480 s, n=6). At a 10-min pause the 30 s convention lost money on every provider (Anthropic 100k: $0.867 vs $0.667 baseline). DeepSeek's cold re-prefill ($0.022) is too cheap to insure, so keepalive is a cost loss at every gap. Pinging past the TTL is 'toxic': on Anthropic, tau>=480 s cost $1.334 vs $0.333 for letting the cache die.

> 原文：The 30 s convention loses money on every provider. [...] the keepalive saves on both non-trivial re-prefillers: 1.56× on Anthropic at τ=240 s and up to 2.45× on OpenAI at its own optimal τ=480 s (1.23× at 240 s). DeepSeek is the honest exception: it evicts, but its cold re-prefill costs $0.022, so seven pings cost more than the eviction they prevent (0.20×), a cost loss at every gap. [...] the “keepalive” costs 4× more than never pinging at all ($1.334 vs. $0.333).

**S27.4**（central；未進入驗證（單一 agent 擷取））

The following is argued, not measured. Providers bill cache residency per read rather than per token-hour, so universal keepalive adoption 'manufactures recency'. LRU then has nothing to rank, and the shared tier degrades for other tenants (shorter effective TTLs, lower hit rates). That is an unpriced congestion externality. The paper predicts lifetime caps, per-account residency quotas, paid long-TTL tiers, and per-token-hour residency metering, which Google's explicit context cache already uses. It explicitly does not measure provider tiers under fleet-wide keepalive load. It calls itself the client-side complement of server-side reuse-aware eviction for coding agents (CacheWise, arXiv 2606.16824). It also cites 'Don't break the cache' (arXiv 2601.06007) and 'KVCache Cache in the Wild' (ATC'25). Our note on relevance: whether reuse-aware or tier-aware (HBM/DRAM/SSD) eviction keeps ranking signal under keepalive load, or whether residency should be priced per tier, is left open.

> 原文：a keepalive manufactures recency, so once every client keeps its prefixes alive, LRU has nothing left to rank and the tier degrades toward first-in-first-out-of-luck. Residency today is priced per read, not per token held per second, and the price does not rise when the tier is hot. [...] metering residency per token-hour, which Google’s explicit context cache already does [...] Retention timescales are measured at the load one measurement client generates; provider tiers under fleet-wide keepalive pressure are the subject of § 6, not of our retention curves.

**S27.5**（supporting；未進入驗證（單一 agent 擷取））

In production, keeping a 100k-token prefix warm saves only about 1.4-3.4 s of TTFT. At an 1800 s pause (n=8) the keepalive cut TTFT from 2.9 to 1.5 s (Anthropic), 4.4 to 1.3 s (OpenAI), and 5.4 to 2.0 s (DeepSeek via DeepInfra). The paper frames a cache-read ping as turning a compute-expensive re-prefill into a nearly free memory read. It costs the client about 1.5x the input price per hour of residency at tau*.

> 原文：at 1800 s (Table 4) the keepalive cuts TTFT from 2.9 to 1.5 s on Anthropic, 4.4 to 1.3 s on OpenAI, and 5.4 to 2.0 s on DeepSeek. [...] (holding a 100k prefix alive at τ∗ costs the client ∼1.5× the input price per hour, recurring rent for the memory), the provider retains the right to evict, and a cache-read ping converts a compute-expensive re-prefill into a nearly free memory read, smoothing their compute demand.

## S28. https://arxiv.org/abs/2503.14647

- 角度：D: Cost-, SLO- and carbon-aware KV caching, and prompt-caching economics
- 來源性質：primary；日期：2025-03-18
- ⚠️ 抽查：標題 "Towards More Economical Context-Augmented LLM Generation by Reusing Stored KV Cache"，Li/Liu/Cheng/Du/Jiang，2025-03-18，cs.NI。**摘要沒有「每小時重用一次以上」這個門檻**；擷取引的是內文句子，引用前要回 PDF 確認

**S28.1**（central；未進入驗證（單一 agent 擷取））

This UChicago poster (Hanchen Li, Yuhan Liu, Yihua Cheng, Kuntai Du, Junchen Jiang; 2 pages, arXiv cs.NI) uses AWS list prices to conclude two things: storing a long context's KV cache in cloud block storage and reloading it costs less than re-running prefill whenever the context is reused more than once per hour, and storage is only a small share of total cost. It is the closest earlier result on the dollar economics of KV caching. Its scope is narrow: one storage tier (EBS io2), uncompressed KV, no quality term, and V100 hardware only.

> 原文：By analyzing the model with real pricing data from AWS, we find that Reusing KV cache could be more economical as long as the context is reused more than once per hour. And that the storage cost is usually only a minimal portion of the total cost.

**S28.2**（central；未進入驗證（單一 agent 擷取））

The cost model simplifies to C_text/C_KV ≈ 1 + ((N−1)/N) · T_prefill(L_context) / (T_decode(L_output) + T_prefill(L_prompt)). The saving is therefore capped by the share of each request's GPU time spent prefilling the reused context, an Amdahl-style bound; storage and transmission terms are dropped from this approximation. Large savings require a long context with a short prompt and a short output (e.g. document QA). This overlaps our finding (6), the a-priori prefill-share criterion, which is a novelty risk. It does not cover memory tiers, KV precision, quality loss ε, or a hardware-dependent recompute/transfer ratio κ.

> 原文：This demonstrates that when context is long while the prompt and output are short, such as in many article Q&A tasks [3], the savings of loading KV cache from storage could be significant.

**S28.3**（supporting；未進入驗證（單一 agent 擷取））

Worked example: the Llama-7B KV cache for a 10K-token context takes 5.2 GB. That matches uncompressed FP16 at about 0.52 MB/token (32 layers × 4096 × 2 × 2 B); this is my own check. On AWS EBS io2 at $0.125/GB-month it costs about $8.8e-4 per hour. One 10K-token prefill on a V100 (footnote: $3/h × 7 s / 3600) costs about $0.0058. The paper says prefill costs 'more than 7 times' the hourly storage cost, but its own footnote numbers give 0.0058/0.00088 ≈ 6.6x, so the claim is slightly overstated. The '$8.8e-4 storage and transmission' figure is really storage only: 0.00017 × 5.2, with transmission and IOPS fees treated as negligible. By my calculation, the break-even storage time is about 6.6 h per reuse, which is the same kind of break-even interval as the five-minute rule.

> 原文：A context of length 10K-token requires a storage space of 5.2 GB. With Amazon Elastic Block Storage(EBS) that could be directly mounted onto instance, the highest tier storage with 4GB/S throughput, called io2, costs 0.125 per GB × month [...] The total storage and transmission cost spent this hour for this 10K-token context is $8.8e−4 [...] On the other hand, the GPU inference cost for prefilling is around $6e−3 for a single request [...] This is already more than 7 times larger than the fixed cost of storage during this hour.

**S28.4**（supporting；未進入驗證（單一 agent 擷取））

Setup: Llama-7B on an EC2 p3.8xlarge (4×V100) using HuggingFace model parallelism, not a serving engine; TriviaQA with 200 contexts, each assumed reused 5 times. Varying input length from 1 to 10K tokens, KV reuse cut end-to-end delay 1.1–2.9x and cloud cost about 1.3–3.6x. Varying output length from 1 to 100 tokens, delay fell 1.6–3.5x and cost 1.7–4.5x. At inputs of about 1K tokens, the time to load the KV cache roughly cancels the prefill saving.

> 原文：When varying the input length from 1–10K, as shown in Figure 2 (a), using KV cache reuse can significantly lower both the end-to-end delay (by 1.1 to 2.9 ×) and the cost of using cloud services (by about 1.3 to 3.6 ×). [...] For shorter inputs (like 1,000 units), the time saved by skipping the prefilling delay does not compromise the delay it takes to load the KV cache. [...] the delay saving ranges from 1.6–3.5× and the cost saving ranges from 1.7–4.5×.

**S28.5**（supporting；未進入驗證（單一 agent 擷取））

KV reuse saves less as outputs get longer, because it only shortens the time to the first token. The paper tested outputs of at most 100 tokens. Its 'validated' model was checked only 'by simulation', with no model-vs-measurement error reported. So the cost case for long-output reasoning models (thousands of decode tokens) and for multi-tier or precision-tiered KV storage is still untested.

> 原文：We observe that the longer the output is, the less saving KV cache reusing can bring. The insight is that KV cache reusing is only able to save delay for the first token's generation, thus, the longer the output is, the more likely the saving will be amortized. [Introduction:] Moreover, we validate this result by simulation under various workloads.
