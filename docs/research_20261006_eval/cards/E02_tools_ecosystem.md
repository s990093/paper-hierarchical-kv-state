# E02：Benchmark 工具與 KV 層系統說明書

> 抽取者：E02 子 agent（2026-10-06 開始、2026-10-07 完成）。狀態：**抽取完成；V02 已於 2026-10-07 獨立複核**（見檔尾「複核紀錄」）。
> 規則依 `../README.md`：每格附出處與證據等級；程式碼一律記 commit SHA 與檔名／行號（行號以該 SHA 為準，換版本會漂移）。

## 範圍

- **A. Benchmark 工具**：vLLM `vllm bench serve`（含 `benchmarks/multi_turn`）、SGLang `sglang.benchmark.serving`、NVIDIA AIPerf（GenAI-Perf 後繼）、GuideLLM、LLMPerf、MLPerf Inference 的 LLM 項目。重點是「指標的程式定義」「預設值」「哪些設定會讓結果不可比」。
- **B. KV 層系統**：LMCache、vLLM KV connector API 與 `OffloadingConnector`、SGLang HiCache、NVIDIA Dynamo KVBM、Mooncake Store／Transfer Engine。
- 不含：論文本身的評測設定（見 E01／E03／E04），資料集統計（見 E10）。

## 來源清單（查證日 2026-10-06／07）

| 代號 | 來源 | 版本 |
|:--|:--|:--|
| [V] | https://github.com/vllm-project/vllm | main `31e2443c90542a33a4a4a293ea7186fba2796c67`（2026-10-06）；另對照 tag `v0.28.0`（`2cf0a6915ce5…`，release 2026-08-26）與 `v0.29.0`（release 2026-09-09） |
| [V-blog] | https://vllm.ai/blog/2026-01-08-kv-offloading-connector | 2026-01-08 |
| [S] | https://github.com/sgl-project/sglang | main `662879e4952190b8649196615e26741448102337`（2026-10-06） |
| [A] | https://github.com/ai-dynamo/aiperf（docs/） | main `f3a76e5a0bd1dd3888052ddcc70be61e53e5cb56`（2026-10-05） |
| [GP] | https://github.com/triton-inference-server/perf_analyzer/blob/main/genai-perf/README.md | main `ee519c63…`（2026-09-15） |
| [G] | https://github.com/vllm-project/guidellm | main `246f3b0f02053b4351f648093461d397ad534cc2`（2026-10-05） |
| [LP] | https://github.com/ray-project/llmperf | `f1d6bed47e4501b0e371082b41601b59ab55269f`（2024-12-08；GitHub 標記 archived） |
| [M-rules] | https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc | `d3eba2f21026d868ad65cdcad2bb81e4a17ce3d3`（2026-08-20） |
| [M-lg] | https://github.com/mlcommons/inference（`loadgen/mlperf.conf`、`loadgen/logging.cc`、`loadgen/results.cc`） | master `3fbc329939999c13d0a7b5e67fb2092287e06047` |
| [M-blog] | https://mlcommons.org/2024/03/mlperf-llama2-70b/ ；https://mlcommons.org/2025/04/llm-inference-v5/ ；https://mlcommons.org/2025/09/deepseek-inference-5-1/ ；https://mlcommons.org/2025/09/small-llm-inference-5-1/ | 網頁，2026-10-06 讀取 |
| [L] | https://github.com/LMCache/LMCache | dev `8c77a6f77b4029269199de39c6f509944294c968`（2026-10-05） |
| [D] | https://github.com/ai-dynamo/dynamo | main `7d7d3dadc06090bfc94479445c122d9c030dd129`（2026-10-06）；KVBM 設計文件取自 tag `v1.4.0` |
| [MC] | https://github.com/kvcache-ai/Mooncake | main `245e710604d46a14c46f8882e4381bd87bcd94ce`（2026-10-06） |
| [PR] | GitHub PR 頁面（`gh pr view`），各條目附編號 | 2026-10-06 讀取 |

## 重點摘要

1. **同一個名字，不同的量。** vLLM 的 TPOT（每請求 `(E2E−TTFT)/(n−1)`）＝ AIPerf 的「ITL」＝ GuideLLM 的「ITL」＝ MLPerf LoadGen 的 TPOT；vLLM／SGLang 的「ITL」是**所有請求的 chunk 間隔攤平後**的分布（AIPerf 叫它 ICL）；GuideLLM 的「TPOT」與 LLMPerf 的「ITL」**把 TTFT 也算進去**（GuideLLM：`(last−start)/n`，n＝輸出 token 數）。〔複核修正〕LLMPerf 的分母不是 token 數，而是 client 收到的 SSE data chunk 數（`tokens_received`，含不帶 content 的 chunk）：除法在 `token_benchmark_ray.py` L121 先做，L124 才用 Llama tokenizer 覆寫 `NUM_OUTPUT_TOKENS`〔程式碼：llmperf `f1d6bed4` `openai_chat_completions_client.py` L83、L112；`token_benchmark_ray.py` L117–124〕。跨工具比數字前必須先對齊定義。〔程式碼〕〔文件〕（§A5 表）
2. **`--random-range-ratio` 預設都是 0.0，意思卻相反。** vLLM：長度取 `[L(1−r), L(1+r)]`，r=0 即固定長度 L；SGLang：取 `[max(L·r,1), L]`，r=0 即 **1 到 L 的均勻分布，平均約 L/2**。SGLang 的 `--gsp-range-ratio` 預設反而是 1.0（固定），程式碼自己加了警告註解。〔程式碼〕〔複核補充〕AIPerf 文件獨立描述了同一差異：`--random-corpus-style sglang` 的窗口是 `[max(1,int(mean·r)), mean]`，「r=0 allows full variability [1, mean]」（aiperf `f3a76e5a` `docs/cli-options.md` L944、L954–955）。
3. **Mooncake trace 的 512-token block，三個工具三種處理。** AIPerf 與 GuideLLM 預設 512（GuideLLM 還驗算 `ceil(input/512)==len(hash_ids)`）；vLLM `timed_trace` 預設 **16**（help 文字有寫 Moonshot 是 512），hash 不夠時**不補長度**，直接用會把 prompt 縮成約 1/32——和本專案 2026-08-31 踩過的錯一模一樣；SGLang 的 mooncake 模式**根本不用 `input_length`**，每個 hash 換成「id＋128 個 hi」。〔程式碼〕〔文件〕〔複核補充〕(a) Mooncake 官方 README 寫 `timestamp` 單位是毫秒、block 512 token（Mooncake `245e7106` `FAST25-release/README.md` L52、L55）；GuideLLM 的 trace 時間戳要求**秒**、mooncake loader 不做換算（guidellm `246f3b0f` `docs/en/guides/datasets.md` L196；`trace_mooncake.py` 無 ms 轉換），所以直接重播 Mooncake trace 會慢 1000 倍，需另設 `time_scale=0.001`〔判讀〕。(b) SGLang 的 mooncake **暖機**請求每個 hash 用 512 個 "hi"，正式請求用 128 個，兩者不一致（sglang `662879e4` `benchmark/serving.py` L1443–1448 vs `datasets/mooncake.py` L84–87）。
4. **vLLM 的時間定義在 2026-09-13 才修正，本專案用的 v0.28.0 不含此修正。** PR #55508 之前：chat 端點的 E2EL 會延到最後那個 usage chunk，completions 端點的 TTFT 用第二次讀時鐘。作者量到的誤差約 24 µs（0.01%），但說「沒有上界」。〔複核修正〕24 µs 是 **chat** 端點 usage chunk 造成的殘差；**completions** 第二次讀時鐘的殘差只有約 −0.35 µs（PR #55508 本文的 real-socket 表格：completions −0.35 µs、chat +24.09 µs；usage chunk 延遲 30 ms 時 chat 變 +31,473 µs）。v0.28.0 原始碼確認仍是舊行為〔複核補充〕v0.29.0（`98dff2a8`）同檔 L236、L421 也仍是舊行為。〔程式碼〕〔PR〕
5. **暖機與「隱形暖機」會汙染 prefix-cache 實驗。** vLLM 的 `--num-warmups`（預設 0）與舊版預設開啟的 ready check 都**重送第一個請求的 prompt**；SGLang 預設 1 個暖機請求，也是第一個請求，且預設**不清快取**。量 KV 重用時，第一筆量測請求會意外命中。〔程式碼〕
6. **到達過程不可比。** vLLM 的 gamma 到達會把總時長**正規化成剛好 N/rate**；SGLang 只有 Poisson、無正規化；AIPerf 有 constant／poisson／gamma 與「每使用者固定間隔」；GuideLLM 的 sweep 內插預設是 **constant 而非 Poisson**；LMCache multi-round QA 是**每使用者固定間隔**而非 Poisson。〔程式碼〕〔文件〕
7. **KV 層系統的寫入策略幾乎都是「寫穿」**：LMCache 把每個完整 chunk 寫進所有已啟用的層；vLLM `OffloadingConnector` 預設 `store_threshold=0`（全存）；SGLang HiCache 預設 `write_through`。只有 Dynamo KVBM 的磁碟層與 Mooncake 的 SSD 層是「依次數／依逐出」才寫。**沒有一個系統把「不存、之後重算」當成寫入選項。**〔程式碼〕〔文件〕〔複核修正〕最後一句說得太滿：機制層面的「不存」開關是有的——LMCache in-process 的每請求 `lmcache.skip_save`（lmcache `8c77a6f7` `integration/vllm/vllm_v1_adapter.py` L332–340；MP 模式文件明說不保證生效，`docs/source/mp/configuration.rst` L42–48）、vLLM `store_threshold≥2` 會跳過未達次數的 chunk、KVBM 磁碟過濾、HiCache `write_back`／`write_through_selective`。查到的系統中沒有一個**依「重算成本 vs 傳輸成本」**決定不存〔判讀：本卡讀過的程式與文件範圍內〕。
8. **Dynamo KVBM 已在 v1.5.0（2026-09-18）宣告棄用，目標 v1.6.0 移除**；Dynamo 改推引擎原生卸載。使用者寄給老師的介紹仍把 Dynamo 列為有卸載機制的系統。〔文件〕〔複核補充〕日期：`deprecations.mdx` 寫 Sep 18, 2026；GitHub release `v1.5.0` 的 publishedAt 是 2026-09-21（`gh release view v1.5.0 --repo ai-dynamo/dynamo`）。release note 也寫 v1.5.0 的 `kvbm` wheel「ship unchanged for this release」，即目前仍可用、只是已棄用。

---

## 範圍 A：Benchmark 工具

### A1. vLLM `vllm bench serve`

入口：`vllm/benchmarks/serve.py`（2418 行）、`vllm/benchmarks/datasets/datasets.py`（4776 行；舊單檔 `datasets.py` 已拆成目錄）、`vllm/benchmarks/lib/endpoint_request_func.py`。舊 `benchmarks/benchmark_serving.py` 現在只印 DEPRECATED 後 `exit(1)`（PR #24411，2025-09-09 完全棄用）。〔程式碼〕[V]

#### A1.1 指標定義

| 指標 | 程式定義 | 出處 |
|:--|:--|:--|
| TTFT | `st=perf_counter()` 在送出 POST 前、**取得 client semaphore 之後**；TTFT＝第一個帶 `choices` 的 SSE chunk 到達時間 − st。chunk 的 text 可以是空字串，仍算第一個 token。〔程式碼〕 | `endpoint_request_func.py` L228–266（completions）、L406–438（chat） [V main] |
| ITL | 每個請求內「相鄰兩個帶 choices 的 chunk」的時間差；所有請求的 list 直接串接（`itls += outputs[i].itl`）再取統計。**是 chunk 間隔，不是 token 間隔**：伺服器一次推多個 token 時會失真。修正 per-token 的 PR #46652 仍為 OPEN。〔程式碼〕〔PR〕 | `endpoint_request_func.py` L270；`serve.py` `calculate_metrics()` L631；PR #5263 說明、#46652 |
| TPOT | 每請求 `(latency − ttft)/(output_len − 1)`，只收 `output_len>1` 的請求；再對請求取 mean／median／std／percentile。終端標題直接寫 "excl. 1st token"。〔程式碼〕 | `serve.py` L625–629、`process_one_metric("tpot", …)` |
| E2EL | `latency = most_recent_timestamp − st`，main 上停在最後一個帶 choices 的 chunk。v0.28.0 的 chat 函式在 usage chunk 也更新時間戳（見 A1.5 #12）。〔程式碼〕 | `endpoint_request_func.py` L287、L462 [V main]；v0.28.0 同檔 L396–423 |
| output_len | 優先用伺服器 `usage.completion_tokens`（每個請求都送 `stream_options.include_usage=True`）；拿不到才用 client tokenizer 重新編碼。prompt_len 也被伺服器回報的 `prompt_tokens` 覆寫。〔程式碼〕 | `endpoint_request_func.py` L214–217、L273–276；`serve.py` L606–620 |
| throughput | `request_throughput = completed/dur`；`output_throughput = Σoutput_len/dur`；`total_token_throughput = (Σinput+Σoutput)/dur`。dur＝`benchmark()` 從開始派送到全部任務完成的牆鐘時間（含第一個到達間隔）。另有 1 秒桶的 peak output tok/s 與 peak concurrent requests。〔程式碼〕 | `serve.py` L683–783 |
| goodput | `--goodput ttft:500 tpot:50 e2el:3000`（單位 ms，key 只能是 ttft／tpot／e2el）。每個**成功**請求要所有 SLO 都滿足（`SLO ≥ 實測`）才算 good；`output_len≤1` 的 TPOT 記為 0；`request_goodput = good/dur`。定義引 DistServe。失敗請求不進分子也不進比較，但分母是時間。〔程式碼〕 | `serve.py` L646–664、`check_goodput_args` L1473、`parse_goodput` L1495；CLI L1821；PR #9338（2024-10-20） |
| percentile | `--percentile-metrics` 預設 `ttft,tpot,itl`（可加 `e2el`、`client_queue_time`、`e2el_including_client_queue`）；`--metric-percentiles` 預設 `"99"`。用 `np.percentile` 預設內插。〔程式碼〕 | `serve.py` L1801–1820 |
| client 排隊 | 開 `--max-concurrency` 時記錄 `client_queue_time = start_time − 到達時間`，另報「含 client 排隊的 E2EL」；**TTFT／E2EL 本身不含 client 排隊**。〔程式碼〕〔複核補充〕「含排隊的 E2EL」只在 `--max-concurrency` 有設**且** `--request-rate` 不是 inf 時才算（L1320–1327）。 | `serve.py` L984、L1318–1330 |

#### A1.2 到達、併發、輸出長度、種子、暖機

| 參數 | 行為與預設 | 出處 |
|:--|:--|:--|
| `--request-rate` | 預設 `inf`：所有請求在時間 0 送出。〔程式碼〕 | CLI L1699 |
| `--burstiness` | 預設 1.0（Poisson）。間隔 ~ Gamma(shape=b, scale=1/(rate·b))，平均 1/rate；b<1 較陣發，b>1 較均勻；`b=inf` 為固定間隔（PR #26941，2025-11-03）。〔程式碼〕間隔的 CV＝1/√b〔計算：gamma 變異數 kθ²〕 | `get_request()` L402–516，抽樣 L479 |
| 總時長正規化 | 沒開 ramp-up 時，把累積延遲整體縮放成最後一個請求**剛好在 N/rate 秒**（註解說是為了消除不同種子 1–2% 的差）。所以實際平均到達率被強制等於設定值，但第一個請求不在 t=0。〔程式碼〕 | L488–499 |
| ramp-up | `--ramp-up-strategy linear`／`exponential`、`--ramp-up-start-rps`、`--ramp-up-end-rps`。〔程式碼〕 | L377–400 |
| `--max-concurrency` | client 端 `asyncio.Semaphore`；到達照常產生，超過上限就在 client 排隊。help 明說實際速率可能低於 `--request-rate`。〔程式碼〕 | CLI L1631 |
| `--probe-request-rate` | 另以固定速率送 1-token 探測請求（不受併發上限），另報其 E2EL，用來看主負載把無關請求卡多久。〔程式碼〕 | CLI L1720 |
| `--ignore-eos` | 預設 False；但 `random`、`random-mm` 在 OpenAI 相容後端**自動設 True**（PR #28227，2025-11-07）；`timed_trace` 強制 True。ShareGPT 等其他資料集預設**不** ignore EOS，實際輸出長度會短於資料集的 completion 長度。〔程式碼〕 | `main_async` L2135–2155 |
| `--seed` | 預設 0；`random.seed`／`np.random.seed`；`RandomDataset` 另用自己的 `default_rng(seed)`。〔程式碼〕 | `datasets.py` L1587；`serve.py` L2013–2014 |
| `--num-prompts` | 預設 1000。〔程式碼〕 | `datasets.py` L72 |
| 暖機 | `--num-warmups` 預設 0（PR #26943，2025-10-16 加入）。暖機請求＝**第一個資料請求重送 N 次**。`--ready-check-timeout-sec` 預設 0（略過）；PR #30975（2026-01-12）之前預設會先送一次第一個請求，PR 作者稱之為 "an ambiguous warm-up"。〔程式碼〕〔PR〕 | `serve.py` L876–920、CLI L1734、L1942 |

#### A1.3 資料集

`--dataset-name` 可選：sharegpt、burstgpt、sonnet、random（預設）、random-mm、random-rerank、hf、custom、custom_audio、custom_image、prefix_repetition、spec_bench、speed_bench、timed_trace。〔程式碼〕`datasets.py` L1594–1614

| 資料集 | 取樣與預設 | 出處 |
|:--|:--|:--|
| sharegpt | 只留 `conversations` ≥2 輪的條目，依 seed 洗牌；**只取第 1 輪人類發言當 prompt、第 1 輪回覆的 token 數當輸出長度**。過濾 `is_valid_sequence`：prompt<4、輸出<4（指定 `--sharegpt-output-len` 時不檢查）、**prompt>1024**、**prompt+輸出>2048** 皆丟棄。註解說門檻沿用舊 `benchmark_serving.py`。⇒ 用 vLLM 跑 ShareGPT **不可能出現 >1024 token 的 prompt**。〔程式碼〕 | `ShareGPTDataset` L1329–1417；`is_valid_sequence` L342–366 |
| random | `--random-input-len` 1024、`--random-output-len` 128、`--random-range-ratio` "0.0"（float，或 JSON `{"input":..,"output":..}`，值域 [0,1)）。長度在 `[floor(L'(1−r)), ceil(L'(1+r))]` 均勻取整數，L'＝L 減 tokenizer special token 數〔複核修正：只有**輸入**減 special token；輸出直接用 L，且下界至少 1，`datasets/utils.py` L68–77〕。內容：`--random-prefix-len`（預設 0）個隨機 token 的**共用前綴**，加上「從 offset 開始的連號 token id」，再 decode／re-encode 修正長度。PR #44708（2026-06-08）起會用伺服器 `/tokenize` 檢查並修正 client／server tokenizer 不一致。〔程式碼〕 | `RandomDataset` L557–772；`datasets/utils.py` L39–99；CLI L1912–1940 |
| prefix_repetition | 前綴 256、後綴 256、前綴組數 10、輸出 128（皆 token）；請求數＝`10 × (num_prompts // 10)`，最後洗牌。內容為全詞表隨機 token。〔程式碼〕 | L4344–4430；CLI L1838–1865 |
| timed_trace | JSONL，欄位名預設 `timestamp`、`input_length`、`output_length`、`hash_ids`；`--timed-trace-chunk-hash-size` **預設 16**（help 說 Moonshot trace 是 512、Qwen／Alibaba 是 16）；`--timed-trace-sec-multiplier` 預設 1（ms trace 要給 0.001）。每個 hash id 展開成 chunk_size 個 token，同 id 共用內容；**hash 用完就停，不補到 `input_length`**。預設依 trace 時間戳送出（`--self-timed`）。〔程式碼〕種子用 Python `hash(str)`，跨行程不保證可重現〔判讀：Python 字串雜湊受 PYTHONHASHSEED 影響〕 | `TimedTrace` L1419–1578，展開 L1482–1513；CLI L1713–1750 |
| burstgpt | 只留 `Model=="GPT-4"` 且 Response tokens>0 的列；**只用長度**，prompt 是合成的連號 token，**不使用 trace 的時間戳**（到達由 `--request-rate` 決定）。〔程式碼〕 | `BurstGPTDataset` L3178–3254 |
| hf | 依 `--dataset-path` 分派到 VisionArena、MTBench、InstructCoder、GSM8K、AIMO、MLPerf 等類別。〔程式碼〕 | `get_samples()` L2076 起，分派於 L2180–2270 |

#### A1.4 多輪 benchmark（`benchmarks/multi_turn/`）

| 項目 | 內容 | 出處 |
|:--|:--|:--|
| 用途 | README 標題就是 "Benchmark KV Cache Offloading with Multi-Turn Conversations"。〔文件〕 | `benchmarks/multi_turn/README.md` |
| 合成對話範例 | 24 段對話；輪數 uniform 12–18；所有對話共用前綴 500 token；每段對話自有前綴 lognormal（平均 1000、上限 5000）；使用者每輪 120–160 token；回答 80–120 token。文字取自 Gutenberg pg1184。分布可選 constant／uniform／lognormal／zipf／poisson。〔文件〕 | `generate_multi_turn.json`；README |
| 負載 | `--num-clients` 預設 1（閉迴路：每個 client 等回覆再送下一輪）；`--max-active-conversations`；`--request-rate` 是**每個 client** 的 Poisson 速率，預設 0＝不等待；`--conversation-sampling` 預設 round_robin；`--warmup-step` 只送每段對話第一輪且不計入。〔程式碼〕 | `benchmark_serving_multi_turn.py` L1380–1470、`poisson_sleep` L561 |
| 指標 | 與 `vllm bench serve` **不同**：若第一個 chunk 含多個 token，TTFT 會扣掉 `(first_chunk_tokens−1)×TPOT`；TPOT＝`(latency−ttft)/(output_tokens−first_chunk_tokens)`；另報以歷史長度估計的 `approx_cached_percent`。〔程式碼〕 | 同檔 L480–508 |
| ShareGPT 版 | 用 `convert_sharegpt_to_openai.py` 轉成多輪（範例 `--max-items=128`）。〔文件〕 | README |

#### A1.5 「同一工具，定義改過」的實例（vLLM）

| # | 日期 | 改了什麼 | 出處 |
|:--|:--|:--|:--|
| 1 | 2024-06-05 | 新增 ITL；TPOT 改成「先算每請求 TPOT 再取統計」，標題 "excl. 1st token"。作者說 ITL 不能拿來比不同伺服器，因為有人會把 token 打包。 | PR #5263 |
| 2 | 2024-08-09 | 修正 ITL 記錄〔複核補充：讀了 diff——修正前 completions 端點把第一個 chunk 的間隔（＝TTFT）也 append 進 ITL，修正後才加 `else:`〕 | PR #7372（`gh pr diff 7372`） |
| 3 | 2024-09-04 | 「input token throughput」換成「total token throughput」 | PR #8164（只看標題） |
| 4 | 2024-10-20 | 加 goodput | PR #9338 |
| 5 | 2024-11-07 | 加 gamma／burstiness | PR #10105 |
| 6 | 2025-01-22 | 輸出 token 數改用伺服器 usage（不再重新 tokenize）；E2E 改到「最後一個 token」而非最後一個 chunk | PR #12288 |
| 7 | 2025-04-10 | `--random-range-ratio`：舊版 `[L·r, L]`、預設 1.0 → 新版 `[L(1−r), L(1+r)]`、預設 0.0、r<1。同一個 r=0.5 在新舊版代表不同分布。 | PR #16126（含 diff） |
| 8 | 2025-10-16 | 加 `--num-warmups` | PR #26943 |
| 9 | 2025-11-07 | random 資料集預設 ignore_eos=True | PR #28227 |
| 10 | 2026-01-12 | ready check 預設關閉（之前每次都多送一個第一請求） | PR #30975 |
| 11 | 2026-02-13 | 修正 random 前綴長度不準 | PR #33907（只看標題） |
| 12 | 2026-09-13 | completions 的 TTFT 改用同一次時鐘讀值；chat／audio 的 E2E 不再延到 usage chunk。**v0.28.0（2026-08-26）與 v0.29.0（2026-09-09）都早於此 PR**；v0.28.0 原始碼確認是舊行為。 | PR #55508；v0.28.0 `endpoint_request_func.py` L235、L396–423〔程式碼〕〔計算：比對日期〕 |

〔複核補充〕12 列的日期皆以 `gh pr view <n> --json mergedAt` 逐一核對一致（UTC）。但並非每列都是「定義改變」：#4（goodput）、#5（gamma）、#8（`--num-warmups`）是**新增**功能；#9、#10 是**預設值**改變；真正改了既有指標算法的是 #1、#2、#3、#6、#7、#11、#12。

### A2. SGLang `sglang.benchmark.serving`

入口：`python -m sglang.benchmark.serving`；`python/sglang/bench_serving.py` 只剩 22 行的棄用轉址（FutureWarning）。〔程式碼〕[S] `bench_serving.py`、`benchmark/serving.py`（2807 行）

| 項目 | 內容 | 出處 |
|:--|:--|:--|
| 資料集 | agentic-trace、sharegpt（**預設**）、custom、openai、random、random-ids、generated-shared-prefix、mmmu、image、mooncake、longbench_v2、speed-bench。〔程式碼〕 | `datasets/__init__.py`；CLI L2284 |
| `--random-range-ratio` 陷阱 | 預設 0.0；`compute_random_lens` 取 `randint(max(int(L·r),1), L+1)`，所以 **r=0 → 1…L 均勻，平均約 L/2**；r=1 才是固定 L。`--gsp-range-ratio` 預設 1.0，原始碼註解明說和 random 的 0.0 不同，是為了向後相容。〔程式碼〕 | `datasets/common.py` L55–63；CLI L2380、L2697 |
| random 內容 | `random`：拿 ShareGPT 的 prompt 重複／截斷到目標長度（長輸入＝同一段話反覆）；`random-ids`：連號 token id。〔程式碼〕 | `datasets/random.py` L84–160 |
| 長度預設 | `--random-input-len` 1024、`--random-output-len` **1024**（vLLM 是 128）；`--num-prompts` 1000。〔程式碼〕 | CLI L2350–2378 |
| ShareGPT 過濾 | prompt<2 或輸出<2 丟棄；只有指定 `--sharegpt-context-len` 才濾長度，**沒有 1024／2048 上限**（與 vLLM 不同）。可 `--apply-chat-template`。〔程式碼〕 | `datasets/sharegpt.py` L101–147 |
| generated-shared-prefix | 64 組 × 每組 16 個 prompt；系統提示 2048、問題 128、輸出 256；`--gsp-num-turns` 1；組的分布 uniform 或 zipf（`--gsp-zipf-alpha`）；預設洗牌。總請求數＝組數×每組數，不看 `--num-prompts`。〔程式碼〕〔文件〕 | `datasets/generated_shared_prefix.py`；CLI L2667–2756 |
| mooncake | 只有 `--backend sglang` 才走 trace 時間排程，**忽略 `--request-rate`**；時間戳除以 1000（ms→s），可 `--mooncake-slowdown-factor`。**prompt 由 hash_ids 組成：每個 id 寫成 `"{id}"＋128 個 "hi"`，不使用 `input_length`**；`--mooncake-num-rounds`>1 時把同一 session 的多輪連續送出，助理回覆用 "story"×輸出長度占位。workload：mooncake（arxiv）、conversation（預設）、synthetic、toolagent。〔程式碼〕 | `datasets/mooncake.py` L55–118；`serving.py` L1546–1556；CLI L2757–2783 |
| 到達 | 只有 Poisson：每送一個請求後睡 `Exp(1/rate)`；第一個請求在 t=0；**沒有 burstiness、沒有總時長正規化**。〔程式碼〕〔複核補充〕`--use-trace-timestamps`（help 說只對 mooncake 有效）在此 SHA **實際不影響排程**：`get_request()` 雖有該參數（L1089–1110），但 `benchmark()` 唯一的呼叫 `get_request(input_requests, request_rate)`（L1560）沒有傳入；該旗標只改輸出的 "Traffic request rate" 標籤（L1674、L1847）。trace 時間排程只取決於 `backend=="sglang"` 且 dataset 為 mooncake（L1548–1554）。 | `get_request()` L1086–1123；`benchmark()` L1548–1560 |
| 指標 | TTFT＝第一個 **text 非空** 的 chunk；ITL＝chunk 間隔串接（投機解碼時用重新 tokenize 攤平）；TPOT＝`(latency−ttft)/(output_len−1)`；E2E latency 在 completions 端點是**最後收到的任何 chunk（含 `[DONE]`）**；output_len 用 usage 的 `completion_tokens`，沒有就**當作請求的 max_tokens**；另報 retokenized 版本與 `concurrency = Σe2e/dur`。**沒有 goodput**。〔程式碼〕 | `calculate_metrics` L1126–1307；completions 函式 L284–400（L350、L363、L379） |
| percentile | 固定 mean／median／std／p90／p95／p99（＋ITL max），不能自選。PR #27662（2026-06-12）之前 TTFT／TPOT 只有 p99、ITL 有 p95／p99、E2E 有 p90／p99。官方文件仍寫舊的組合。〔程式碼〕〔PR〕〔文件〕〔複核補充〕`gh pr diff 27662` 確認新增的是 TTFT／TPOT 的 p90、p95，ITL 的 p90，E2E 的 p95。文件 L240–242 寫 E2E「mean/median/std/p99」、TTFT「…/p99」、ITL「…/p95/p99/max」，E2E 少列了舊版就有的 p90，所以文件既不是新組合也不完全是舊組合。 | `BenchmarkMetrics` L1030–1083；`docs/docs/developer_guide/bench_serving.mdx` L236–244 |
| 預設行為 | `ignore_eos` **預設 True**（全部資料集，`--disable-ignore-eos` 關閉）；temperature 0.0；`--seed` 42；`--warmup-requests` 1（用第一個請求、輸出截到 ≤32 token）；`--ready-check-timeout-sec` 60；`--flush-cache` 預設關（只有加旗標或 `SGLANG_IS_IN_CI` 才在暖機後清快取；對 vLLM 打 `/reset_prefix_cache`，需 `VLLM_SERVER_DEV_MODE=1`）。〔程式碼〕〔文件〕〔複核補充〕`SGLANG_IS_IN_CI` 只在 backend 名稱含 "sglang" 時觸發（L1513–1515）；對 vLLM 打的 `/reset_prefix_cache` **沒有帶 `reset_external=true`**（L1014–1017），所以只清 GPU prefix cache，不清 OffloadingConnector 的 CPU 層（見 PoC 建議 5 的複核補充）。 | `serving.py` L311、L1470–1517、CLI L2500–2660；`bench_serving.mdx` L89 |
| 定義改過 | TPOT 在 PR #3988（2025-03-03）被拿掉、PR #12976（2025-11-11）加回；percentile 組合在 #27662 改；投機解碼的 ITL 在 #12064／#12156（2025-10）改。〔PR〕 | PR 頁面 |

### A3. AIPerf（GenAI-Perf 後繼）、GuideLLM、LLMPerf

#### AIPerf

| 項目 | 內容 | 出處 |
|:--|:--|:--|
| 定位 | GenAI-Perf README 寫「正在淘汰，不再開發新功能，請改用 AIPerf」；AIPerf 文件自稱 successor。〔文件〕 | [GP] README L31；[A] `docs/genai-perf-feature-comparison.md` |
| TTFT | 第一個「內容非空的回應」時間 − 請求開始。包含網路、排隊、prefill。〔文件〕 | `docs/metrics-reference.md` L215–238 |
| ITL | **每請求** `(request_latency − TTFT)/(OSL − 1)`——即 vLLM 的 TPOT；`--per-chunk-usage` 可改用伺服器回報的第一個 chunk token 數修正分母。〔文件〕〔複核補充〕程式碼一致：`src/aiperf/metrics/types/inter_token_latency_metric.py` L54–75；TTFT＝`content_responses[0].perf_ns − start`（`ttft_metric.py` L49–56）；request latency＝`content_responses[-1]`（`request_latency_metric.py` L45–49）〔程式碼，aiperf `f3a76e5a`〕。 | L306–334 |
| ICL | chunk 到達間隔的完整分布——即 vLLM／SGLang 的 ITL。〔文件〕 | L336–353 |
| token 計數 | OSL＝輸出＋reasoning token；**預設用 client tokenizer 計算**（`add_special_tokens=False`），`--use-server-token-count` 才改用 usage。ISL 同樣用 client tokenizer。〔文件〕 | L399–449；`docs/cli-options.md` L606 |
| Request latency | 到最後一個內容回應。〔文件〕 | L1713–1727 |
| goodput | `--goodput "time_to_first_token:100 inter_token_latency:3.40"`；所有 SLO 都滿足才算；goodput＝good/總時長；另有 `good_request_fraction`，**錯誤請求算進分母**。注意這裡的 inter_token_latency 是 AIPerf 的 ITL（＝vLLM TPOT）。〔文件〕 | L1601–1665；`docs/tutorials/goodput.md` |
| 負載模式 | `--request-rate`（`--arrival-pattern` constant／poisson（預設）／gamma，`--arrival-smoothness`＝gamma 形狀參數，文件說與 vLLM `--burstiness` 相容）；`--concurrency`（單獨用＝盡量送滿；和 rate 一起用＝上限）；`--fixed-schedule`（trace 時間戳；給 mooncake_trace 等帶 timestamp 的資料時自動啟用）；`--user-centric-rate`（每使用者固定間隔＝`num_users/QPS`，文件明說用於 KV cache TTL 測試）。〔文件〕 | `docs/benchmark-modes/timing-modes-reference.md`；`docs/tutorials/arrival-patterns.md` L83–148；`docs/tutorials/user-centric-timing.md` |
| 合成長度 | ISL 預設 550、常態分布、stddev 0；OSL **預設不設**（由模型決定何時停，沒有 ignore_eos）。KV 相關：`--shared-system-prompt-length`、`--user-context-prompt-length`、`--prefix-prompt-length`、`--num-prefix-prompts`。〔文件〕〔複核補充〕「沒有 ignore_eos」的佐證：`docs/metrics-reference.md` L1467 建議要讓伺服器遵守 `--osl` 時自己加 `--extra-inputs ignore_eos:true`。另有 `--random-range-ratio` 與 `--random-corpus-style vllm`／`sglang`（預設 vllm）可模仿兩家的長度窗口（`cli-options.md` L942–955）。 | `docs/cli-options.md` L899–968 |
| trace block | `--isl-block-size` 覆寫 loader 預設：**mooncake_trace 512、bailian_trace 16**；長度＝`(n_hash−1)×block＋最後一塊`。timestamp 單位 ms。〔文件〕 | `cli-options.md` L933–935；`docs/benchmark-modes/trace-replay.md` L43–60 |
| 種子與暖機 | `--random-seed` **未設定時用系統熵**（預設不可重現）；預設**沒有暖機**，需 `--warmup-request-count`／`--warmup-duration` 等觸發。〔文件〕 | `cli-options.md` L739–743；`docs/tutorials/warmup.md` L412 |

#### GuideLLM

| 項目 | 內容 | 出處 |
|:--|:--|:--|
| 負載 profile | synchronous、throughput（`max_concurrency`）、concurrent（`streams`）、constant（＝async）、poisson、sweep、replay。sweep＝synchronous → throughput → 中間內插 `sweep_size−2` 個速率，**內插預設用 constant**，可改 `strategy_type=poisson`。〔文件〕 | `docs/en/getting-started/benchmark.md` L106–205 |
| TTFT／ITL／TPOT | TTFT＝第一個 token − request_start；ITL＝`(last−first)/(n−1)`；**TPOT＝`(last−start)/n`，含第一個 token（含 TTFT）**。〔程式碼〕〔複核補充〕n（`output_tokens`）取伺服器 usage 的 `completion_tokens`，沒有 usage 時退回「token iteration 次數」（註解：假設每次 iteration 一個 token）（`request_stats.py` L237–246；`backends/openai/request_handlers.py` L778–835）。若資料指定輸出長度，completions 請求會送 `ignore_eos=True`、`stop=None`，並送 `stream_options.continuous_usage_stats`（`request_handlers.py` L612–624）。 | `src/guidellm/schemas/base/request_stats.py` L264–354 |
| 其他時間 | dispatch delay（排定到實際送出）、scheduled latency（從排定時間算的延遲）、turn predecessor／scheduling delay。〔文件〕 | `docs/en/guides/metrics.md` |
| SLO | SLO attainment（錯誤算不合格；被時長截斷或無法評估的不計）；request goodput（合格請求／秒）。〔文件〕 | 同上 |
| 量測窗 | warmup／cooldown；請求級指標收「生命週期與窗口重疊」的請求，TTFT／ITL 只收事件落在窗口內的。〔文件〕〔複核修正：措辭更精確是——TTFT 收「request start→first token」區間與窗口部分重疊者，first token 早於 `measure_start` 的不收；ITL 收「first token→完成」區間與窗口部分重疊者，`docs/en/guides/metrics.md` L127–134〕 | 同上 |
| 種子 | 文件說預設是固定值（範例 `kind=static,value=42`）。〔文件〕 | `benchmark.md` L76–84 |
| trace | 時間戳**以秒為單位**；mooncake 格式 `hash_id_block_size` 預設 512，並**驗證 `ceil(input/512)==len(hash_ids)`**，不符的列丟出錯誤；`time_scale` 乘在相對時間上。〔文件〕〔程式碼〕〔複核補充〕(a) `InvalidRowError` 的 docstring 寫該列「should be skipped」（`src/guidellm/data/schemas/base.py` L47–48），但 mooncake 路徑上沒看到 catch（`trace_common.py` L357–379），上層是跳過還是中止〔未查證〕。(b) Mooncake 原始 trace 是毫秒，GuideLLM 不換算，需 `time_scale=0.001`〔判讀，見重點摘要 3〕。(c) replay profile 預設 `schedule_turn=idle_gap`：下一個請求在「前一個實際完成＋紀錄的 idle gap」才送，**不是**照 trace 時間戳；要照時間戳需 `schedule_turn=timestamp`（`docs/en/getting-started/benchmark.md` L217–224；`benchmark/profiles/replay.py` L28–30）。 | `docs/en/guides/trace_replay.md` L60–103；`docs/en/guides/datasets.md` L196；`src/guidellm/data/deserializers/trace_mooncake.py` L83–96；`trace_session_timing.py` L3–5 |

#### LLMPerf

| 項目 | 內容 | 出處 |
|:--|:--|:--|
| 狀態 | GitHub 標記 archived；最後 commit 2024-12-08。〔文件〕〔複核補充：2024-12-08 是 committer 時區（−08:00）的日期；UTC 為 2024-12-09T01:52:28Z，`gh api repos/ray-project/llmperf/commits/f1d6bed4`〕 | `gh api repos/ray-project/llmperf` |
| 負載 | `--num-concurrent-requests` 閉迴路；輸入／輸出長度用 mean＋stddev；prompt 從莎士比亞十四行詩抽行；**不論測哪個模型都用 Llama tokenizer 計 token**。〔文件〕〔程式碼〕 | README L15–30；`token_benchmark_ray.py` L63–66 |
| ITL | client 端把「TTFT＋之後每個 token 間隔」加總，再除以輸出 token 數 ⇒ **含 TTFT 的平均**。程式註解自己說這和 E2E 應該一樣。〔程式碼〕〔複核修正〕分子是「TTFT＋之後每個**帶 content 的 chunk** 的間隔」（L92–100），分母**不是 token 數**，而是 client 端的 `tokens_received`＝收到的非 `[DONE]` SSE data 行數（L83，含 role-only、空 content 的 chunk）；`token_benchmark_ray.py` L121 先做除法，L124 才把 `NUM_OUTPUT_TOKENS` 覆寫成 Llama tokenizer 的計數。所以 LLMPerf ITL ≈ (最後一個 content chunk − start)／SSE chunk 數，伺服器一次推多個 token 時與 token 數無關。 | `src/llmperf/ray_clients/openai_chat_completions_client.py` L74–117；`token_benchmark_ray.py` L113–126 |

### A4. MLPerf Inference 的 LLM 基準

| 項目 | 內容 | 出處 |
|:--|:--|:--|
| Scenario | Server／Interactive：Poisson 到達、600 s、tail 99%、指標＝能維持的最大 Poisson 速率。Offline：所有樣本一次送出，至少 24,576 個（資料集較小則為資料集大小 N），指標＝吞吐。Single stream：tail 90%。有 early stopping 規則。LLM 不准跨資料集邊界排序樣本。〔文件〕 | [M-rules] L136–152 |
| 百分位 | `*.Server.target_latency_percentile = 99`；LLM 項目 `use_token_latencies = 1`，`target_latency = 0`，只看 TTFT 與 TPOT。〔程式碼〕 | [M-lg] `mlperf.conf` L83、L96–160 |
| TPOT 定義 | LoadGen 每樣本 `(latency − first_token_latency)/(n_tokens − 1)`，再取第 99 百分位——與 vLLM TPOT 相同。〔程式碼〕〔複核補充〕百分位取法不同：LoadGen 是排序後取 index `sample_count × p`（最近秩，不內插，`results.cc` L105–136），vLLM 用 `np.percentile` 預設線性內插。 | [M-lg] `logging.cc` L475–476；`results.cc` L92–144 |
| Llama2-70B | OpenOrca（max_seq_len 1024），24,576 筆；準確度 99.9% of FP32（rouge1／2／L），每樣本 token 數 >90% 參考值 294.45；Server（規則表寫 Conversational）TTFT／TPOT 2000／200 ms；Interactive 450／40 ms。〔文件〕 | [M-rules] L267 |
| Llama3.1-405B | LongBench、LongDataCollections、RULER、GovReport 的子集，8,313 筆；99% of FP16；token 數在參考值 684.68 的 90–110%；Server 6000／175 ms；Interactive 4500／80 ms；max_new_tokens 20000。部落格：平均輸入約 9,400、輸出約 680 token，context window 128K。〔文件〕 | [M-rules] L268、L376–377；[M-blog] llm-inference-v5 |
| Llama3.1-8B | CNN/DailyMail v3.0.0（max_seq_len 2048）；資料中心 13,368 筆、Edge 5,000 筆；99% 與 99.9% of FP32；Server 2000／100 ms；Interactive 500／30 ms。部落格：平均輸入 778、輸出 73 token。〔文件〕 | [M-rules] L266、L305；[M-blog] small-llm |
| DeepSeek-R1 | mlperf_deepseek_r1，4,388 筆；規則表寫 99% of FP16（EM 81.9132%）；Server 2000／80 ms；Interactive 1500／15 ms；max_new_tokens 20000。部落格：平均輸入 800、輸出 3,880 token。〔文件〕 | [M-rules] L270、L380–381；[M-blog] deepseek |
| GPT-OSS-120B | AIME25、GPQA Diamond、LiveCodeBench v6，6,396 筆；99% of 83.13%；Server 3000／80 ms；Interactive 2000／20 ms；sampling temperature 1.0。〔文件〕〔複核補充〕規則與 LoadGen 設定不一致：[M-lg] `mlperf.conf` L162–164 的 `gpt-oss-120b-interactive.Server.tpot_latency = 15`（不是 20）。實跑以 LoadGen 設定為準還是以規則為準〔未查證〕。 | [M-rules] L271、L893 |
| 其他 2025–26 新項 | Mixtral-8x7B（15,000 筆，2000／200 ms）；Edge 的 Qwen3.6-27B agentic function calling（BFCL v4，只測準確度）；E2E-RAG（FRAMES 824 題，Offline）。〔文件〕 | [M-rules] L269、L277、L309 |
| 為什麼這樣設（原文） | Llama2-70B：以人類閱讀速度為錨，200 ms TPOT 約等於每分鐘 240 字；Interactive 450／40 ms 依業界分析「每秒 20–50 token」的流暢度；405B 的 6 s／175 ms 是在算力需求與回應性間取捨；8B 的 100 ms 約 480 wpm、30 ms 約 1,600 wpm；DeepSeek 的門檻反映推理模型的大思考預算。準確度門檻是為了不偏離 FP32 參考輸出太多。〔文件〕 | [M-blog] 四篇 |
| 規則演變 | 2024-01-11「移除 Llama2 的低延遲限制」；2024-12-16 加 405B；2025-01-08 Llama2-70B 加 interactive。〔文件：commit 標題〕 | `gh api …/commits?path=inference_rules.adoc` |
| 設計理由〔判讀〕 | MLPerf 是唯一把「準確度門檻＋p99 TTFT／TPOT＋固定資料集」綁在一起的基準；但它**不規範 KV 分層與重用**，405B 的長輸入平均也只有約 9.4K，遠低於本研究的 16K–512K。可沿用其「p99＋TPOT 定義＋準確度 gate」的格式，不可直接沿用其門檻數值。〔複核修正〕「不規範重用」不正確：規則 FAQ 明文**禁止跨 query 的快取**——KV cache 只能用在單一 query 內、不能跨 query，每個輸入都必須完整計算（[M-rules] L895–897）。所以標準 MLPerf LLM 項目的數字**本質上是無 prefix 重用**的結果，不能當 KV 重用或分層的 baseline；分層（CPU／SSD）本身確實沒有規範。 | [M-rules] L895–897 |

### A5. 跨工具指標對照（同名不同義）

| 概念 | vLLM bench serve | SGLang | AIPerf | GuideLLM | LLMPerf | MLPerf LoadGen |
|:--|:--|:--|:--|:--|:--|:--|
| 每請求 `(E2E−TTFT)/(n−1)` | **TPOT** | **TPOT** | **ITL** | **ITL** | — | **TPOT** |
| chunk 間隔分布（攤平） | **ITL** | **ITL** | **ICL** | — | — | — |
| 含 TTFT 的 `E2E/n` | — | — | — | **TPOT**（n＝token 數） | **ITL**〔複核修正：n＝SSE chunk 數，不是 token 數〕 | — |
| TTFT 的「第一個 token」 | 第一個帶 choices 的 chunk（可空） | 第一個 text 非空的 chunk | 第一個內容非空的回應 | first_token_iteration | 第一個 content 非空的 delta | SUT 回報 first token |
| 輸出 token 數 | 伺服器 usage（無則重新 tokenize） | usage，無則**用 max_tokens** | **client tokenizer**（可改 usage） | 〔複核補充〕伺服器 usage，無則 token iteration 次數 | Llama tokenizer（但 ITL 分母用 chunk 數） | SUT 回報 n_tokens |
| 預設 percentile | p99（可選） | p90／p95／p99 固定 | p50／p90／p99 等 | 〔複核補充〕p0.1／p1／p5／p10／p25／median／p75／p90／p95／p99／p99.9，另附 CI（`docs/en/guides/metrics.md` L140–175） | 〔複核補充〕p25／p50／p75／p90／p95／p99（pandas quantile，`token_benchmark_ray.py` L231） | p99（Server） |
| goodput | 有（成功請求、時間分母） | 無 | 有（另有含錯誤的 fraction） | 有＋SLO attainment | 無 | 以 p99 門檻判定合格 |

出處：見 A1–A4 各格。〔程式碼〕〔文件〕；本表為彙整〔判讀〕。

---

## 範圍 B：KV 層系統說明書

### B1. LMCache

| 項目 | 內容 | 出處 |
|:--|:--|:--|
| 架構 | GPU → CPU DRAM（pinned，「hot cache」）→ 本地儲存（磁碟、NVMe GDS）→ 遠端（Redis、Mooncake、InfiniStore…）。兩種模式：storage（跨請求、跨 session 持久化）與 transport（PD 分離）。索引以 chunk 為單位做 token 雜湊。〔文件〕 | [L] `docs/source/developer_guide/architecture.rst` |
| 與 vLLM 的接法 | vLLM 內建 `LMCacheConnectorV1`（`kv_connector: "LMCacheConnectorV1"`，實作轉給 `lmcache.integration.vllm.vllm_v1_adapter.LMCacheConnectorV1Impl`，`use_native` 時用 vLLM 內附版本）；以及 `LMCacheMPConnector`（LMCache 另起 server 行程，預設 `tcp://localhost:5555`）。vLLM 的 `--kv-offloading-backend lmcache` 會選 `LMCacheMPConnector`，**且不會把 `--kv-offloading-size` 傳過去**（容量由 LMCache server 決定）。〔程式碼〕 | [V] `kv_connector/v1/lmcache_connector.py` L68–108；`factory.py` L165–176；`vllm/config/vllm.py` L1148–1175 |
| 模式狀態 | in-process 模式的文件標為 deprecated，建議用 MP 模式。〔文件〕 | [L] `docs/source/kv_cache_optimizations/blending.rst`（warning）、`cacheblend.rst` |
| in-process 參數預設 | `chunk_size` 256；`local_cpu` True；`max_local_cpu_size` 5.0（GB）；`local_disk` None；`max_local_disk_size` 0.0；`remote_url` None；`use_layerwise` False；`save_decode_cache` False；`save_unfull_chunk` False；`cache_policy` "LRU"（可選 LRU／LFU／FIFO／MRU）；`enable_blending` False；`blend_min_tokens` 256；`enable_async_loading` False；`min_retrieve_tokens` 0；`lookup_timeout_ms` 3000。〔程式碼〕 | [L] `lmcache/v1/config.py` L90–164、L385–470；`storage_backend/cache_policy/__init__.py` L14–17 |
| MP server 參數 | `--chunk-size` 256；`--l1-size-gb` 必填；`--eviction-policy` 必填（LRU、ARC、IsolatedLRU、noop）；觸發水位 0.8、一次逐出 0.2；`--l2-store-policy default`＝所有 key 存到所有 L2 adapter 且保留 L1；L2 預取預設「讀完就從 L1 刪」，`retain` 才保留。`--engine-type blend` 開 CacheBlend。〔文件〕 | [L] `docs/source/mp/configuration.rst` L79–113、L312–540 |
| 寫入時機 | prefill 過程中每個 step 之後存；只存**完整 chunk**（最後一段不滿 256 的，除非 `save_unfull_chunk=True` 否則丟掉）；**decode 產生的 KV 預設不存**；已存過的部分以 `skip_leading_tokens` 跳過。〔程式碼〕 | [L] `integration/vllm/vllm_v1_adapter.py` L310–361、L552–566 |
| 寫哪幾層 | `batched_put` 對**每一個啟用的 backend**都寫一次（除非指定 `store_location`）＝寫穿到所有層；磁碟與遠端的 put 是非同步，get 是阻塞。CPU 空間不夠時 LRU 逐出。〔程式碼〕〔文件〕 | [L] `storage_backend/storage_manager.py` L386–430；`docs/source/kv_cache/storage_backends/local_storage.rst` L115–125；`cpu_ram.rst` L49–66 |
| 讀取時機 | 排程端 `get_num_new_matched_tokens` 查前綴命中；worker 端 `start_load_kv` 載入；`use_layerwise` 時逐層載入與計算重疊。磁碟有 prefetch 到 CPU 的機制。〔程式碼〕〔文件〕 | 同上 adapter L756、L973、L1360；`local_storage.rst` L124–125 |
| 非前綴重用 | 有：CacheBlend。in-process 版需 `use_layerwise=True`、分隔字串 `" # # "`、在第 1 層挑出約 15% token 重算（範例值）；MP 版用 `--engine-type blend`。〔文件〕 | `blending.rst`；`cacheblend.rst` |
| 官方 benchmark：long_doc_qa | 文件長度 20,000（以 `"hi"` 字數構成）、文件數 8、輸出 100、每份重複 2 次、`--repeat-mode` random／tile／interleave、同時在途 2、`--hit-miss-ratio`；先 warmup 輪（每份送一次＝未命中）再 query 輪，比較兩輪 TTFT。〔程式碼〕 | [L] `benchmarks/long_doc_qa/long_doc_qa.py` 檔頭、L503–573、CLI L611–750 |
| 官方 benchmark：multi_round_qa | `--num-users`、`--num-rounds`、`--qps`（必填）、`--shared-system-prompt`、`--user-history-prompt`、`--answer-len`；README 範例 10 位使用者 × 5 輪、QPS 0.5、系統提示 1000、歷史 2000、回答 100。**每位使用者以固定間隔 `num_users/qps` 發問**（不是 Poisson），使用者以固定間隔陸續加入；prompt 用 `"hi"` 重複組成；`max_tokens=answer_len`，沒有 ignore_eos；TTFT 取第一個非空 chunk。〔程式碼〕〔文件〕 | `benchmarks/multi_round_qa/README.md`；`multi-round-qa.py` L82、L130–167、L222–240、L314–338、L362–367 |
| 官方 benchmark：其他 | multi_doc_qa（CacheBlend：warmup 輪送單篇，之後隨機拼接；範例 100 篇 × 3000 token、每請求 5 篇、輸出 1）；rag；storage_backend_io。新的 `lmcache bench engine` 工作負載：long-doc-qa（文件 10,000、每份 2 問、在途 3；`--kv-cache-volume` 預設 100 GB 決定文件數）、multi-round-chat（系統提示 2000、歷史 10,000、輸入 50、輸出 200、QPS 1.0、60 s）、long-doc-permutator、prefix-suffix-tuner、rag-qa-quality、random-prefill；種子 42；預設有暖機。〔文件〕 | `benchmarks/multi_doc_qa/README.md`；`docs/source/cli/bench.rst` L145–340；`docs/source/getting_started/benchmarking.rst` |

### B2. vLLM KV connector API 與 `OffloadingConnector`

| 項目 | 內容 | 出處 |
|:--|:--|:--|
| 排程端 hook | `get_num_new_matched_tokens(request, num_computed_tokens)`：回傳「已算部分之後還能從外部載入的 token 數」與是否非同步；可回 None 表示稍後再問；註解要求只算**最長可用前綴**。`update_state_after_alloc`、`build_connector_meta`、`update_connector_output`、`request_finished`（可接手延後釋放 block）、`take_events`。〔程式碼〕 | [V] `kv_connector/v1/base.py` 檔頭 L1–40、L496–528 |
| worker 端 hook | `start_load_kv`、`wait_for_layer_load`（逐層等待）、`save_kv_layer`（逐層存）、`wait_for_save`、`get_finished`／`get_transfer_results`、`handle_preemptions`。〔程式碼〕 | 同檔 L281–426 |
| 已註冊 connector | LMCacheConnectorV1、LMCacheMPConnector、NixlConnector（含 Pull／Push）、MultiConnector、OffloadingConnector、SimpleCPUOffloadConnector、MooncakeConnector、MooncakeStoreConnector、FlexKVConnectorV1、HF3FSKVConnector、HiSparseConnector、MoRIIOConnector 等。〔程式碼〕 | `kv_connector/factory.py` L150–250 |
| 啟用方式 | `--kv-offloading-size <GiB>`（`--kv-offloading-backend` 預設 native）→ 自動設 `OffloadingConnector` 與 `cpu_bytes_to_use`；環境變數 `VLLM_USE_SIMPLE_KV_OFFLOAD` 改用 `SimpleCPUOffloadConnector`。部落格寫的舊版設定鍵是 `num_cpu_blocks`。〔程式碼〕〔文件〕 | `vllm/config/cache.py` L256–265；`vllm/config/vllm.py` L1148–1175；[V-blog] |
| extra_config | `cpu_bytes_to_use`（必填）；`block_size` 或 `blocks_per_chunk`（離線 chunk，預設＝GPU block）；`eviction_policy` `lru`（預設）或 `arc`，也可外掛自訂類別；`store_threshold` 預設 0——**≥2 時一個 chunk 要被「提議儲存」達 N 次才真的存**；`max_tracker_size` 64,000；`spec_name` `CPUOffloadingSpec`（預設）或 `TieringOffloadingSpec`。v0.28.0 已有相同的兩個 spec 與 lru／arc。〔程式碼〕〔複核補充〕`store_threshold` 的**計數方式在 v0.28.0 與 main 不同**：main 在 `prepare_store` 時記「被提議儲存」次數（`cpu/manager.py` L122–141、L250–271）；v0.28.0 在 `lookup()` 時記「被查詢」次數（v0.28.0 `cpu/manager.py` L113–121；`cpu/spec.py` L126–128 註解 "must appear in lookup()"），單位也是 block 而非 chunk。本專案用 v0.28.0，baseline 描述要寫 v0.28.0 的語意。 | `vllm/v1/kv_offload/cpu/spec.py` L82–157；`offloading/config.py` L99–130；`kv_offload/factory.py` L31、L62–68；`cpu/policies/factory.py`（main 與 v0.28.0） |
| 分層 | `TieringOffloadingSpec`：CPU 為主層＋`secondary_tiers`（fs、obj、p2p、example 或外掛〔複核補充：main 另註冊了 `kvcr`，`tiering/factory.py` L146–170〕）；存＝GPU→CPU→次層**逐層串下去**，讀＝次層→CPU→GPU（升層）。〔程式碼〕 | `kv_offload/tiering/spec.py` 檔頭；`tiering/base.py` L135–136 |
| 部落格的設計 | CPU 是主要目標也是往外部儲存的跳板；存與載都非同步，使用者請求不等卸載完成〔複核：部落格明說的是**卸載**非同步、請求不必等傳輸完成；「載入也非同步」在部落格中未找到明確句子〔未查證〕〕；0.12.0 把每個 block 改成「所有層連續的一塊」，實體 block 由 KB 級變成約 0.5–2 MB，因為 DMA 在大塊連續拷貝時最快；0.11.0 首次加入、0.14.0 起有 CLI 旗標。測試：H100、Llama-3.1-8B，單請求 TTFT 降 2–22 倍，高命中時吞吐最多 9 倍。部落格沒有談逐出策略。〔文件〕 | [V-blog] |

### B3. SGLang HiCache、NVIDIA Dynamo KVBM、Mooncake Store／Transfer Engine

**SGLang HiCache**（[S] `docs/docs/advanced_features/hicache_design.mdx`；預設值見 `python/sglang/srt/arg_groups/fields/memory.py`）

| 項目 | 內容 | 出處 |
|:--|:--|:--|
| 層級 | L1 GPU、L2 主機記憶體（**每個 instance 私有**，同節點兩個 instance 也不共用）、L3 儲存後端（file、mooncake、hf3fs、nixl、aibrix…，是否跨 instance 共享看後端設定）。〔文件〕 | design §Overall Architecture、§Tier Sharing Scope |
| 粒度與索引 | HiRadixTree，每個節點記錄它在哪幾層；`page_size>1` 時以 page 比對；L3 的 metadata 不常駐，用到時即時查。〔文件〕 | §HiRadixTree、§Local Match |
| 寫入策略 | `write_through`（**預設**）、`write_through_selective`（命中次數門檻 2 才寫）、`write_back`（被逐出時才寫）。L2→L3 只寫 L3 尚未有的資料。〔程式碼〕〔文件〕 | `memory.py` L118–124；`unified_radix_cache.py` L501–503；§Data Write-back |
| 預取 | L3 命中超過 256 token 才預取；策略 best_effort／wait_complete／timeout（**預設 timeout**：2 s＋每 1K token 0.1 s，上限 30 s）。〔程式碼〕〔文件〕 | `memory.py` L166–172；§Prefetch from L3 |
| 其他預設 | `hicache_ratio` cache 模式預設 2.0（主機池＝2×GPU 池）；`hicache_mem_layout` page_first；`hicache_io_backend` kernel；載入時逐層與計算重疊。〔程式碼〕〔文件〕 | `memory.py` L106–144；§Data Transfer Optimization |
| 逐出 | radix tree `--radix-eviction-policy` 預設 lru，另有 lfu、slru、priority、tlru。〔程式碼〕 | `memory.py` L24–42 |
| 引擎 | SGLang 原生（`--enable-hierarchical-cache`）；LMCache 被列為替代方案。〔文件〕 | §Related Parameters、§Unified Interfaces |

**NVIDIA Dynamo KVBM**（[D]）

| 項目 | 內容 | 出處 |
|:--|:--|:--|
| 狀態 | **v1.5.0（2026-09-18）棄用，目標 v1.6.0 移除**；官方建議改用引擎原生卸載做主機／磁碟分層。目前的「本地卸載」教學只列 LMCache、HiCache、FlexKV。〔文件〕 | `docs/fern/pages/reference/general/releases/deprecations.mdx` L20–26；`docs/fern/pages/cli/kv-cache-offloading/overview.mdx` |
| 層級 | G1 GPU → G2 主機 pinned → G3 磁碟 → G4 物件儲存（S3 相容）。以 `DYN_KVBM_CPU_CACHE_GB`、`DYN_KVBM_DISK_CACHE_GB` 等設定大小；文件警告 CPU 層小於 GPU KV 時會反覆卸載而變慢。〔文件〕 | `docs/fern/pages/reference/components/kvbm-configuration.mdx` |
| 粒度 | block（block 大小從 vLLM 執行期讀取）；以 sequence hash 去重。〔文件〕 | v1.4.0 `kvbm-design.md`；`dynamo-v1-0-0.mdx` L276 |
| 寫入策略 | G1→G2、G2→G3 各有 policies 清單（pass_all／presence／presence_lfu，AND 組合；`presence_lfu.min_lfu_count` 預設 8）；磁碟層預設有過濾，只把頻率 ≥2 的 block 寫到 SSD（頻率初值 1、命中加倍、時間衰減減 1），理由是延長 SSD 壽命。〔文件〕〔複核補充〕v1.4.0 `kvbm-guide.md` L271 自稱「KVBM is a write-through cache」，並說 CPU 層小於 GPU KV 時「there will be no benefit」（可用 `DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER=true` 關閉磁碟過濾，L278–282）。 | `kvbm-configuration.mdx`；v1.4.0 `kvbm-guide.md` L276–281 |
| 逐出 | 設計文件只描述 InactivePool 回收，具體順序〔未查證〕。 | v1.4.0 `kvbm-design.md` L110–150 |
| 引擎 | vLLM：`"kv_connector":"DynamoConnector","kv_connector_module_path":"kvbm.vllm_integration.connector"`；也支援 TRT-LLM。〔文件〕 | v1.4.0 `kvbm-guide.md` L97、L318 |

**Mooncake Store／Transfer Engine**（[MC]）

| 項目 | 內容 | 出處 |
|:--|:--|:--|
| 定位 | 分散式 KV 物件儲存（key→value，由使用者決定物件粒度），Master 管空間與 metadata，client 貢獻記憶體 segment；可多副本；可嵌入推論行程或獨立服務。〔文件〕 | `docs/source/design/store/mooncake-store.md` L7–43 |
| 逐出 | 用量達高水位（預設 90%）或配置失敗時，逐出約 5%；**近似 LRU**；有 lease（預設 10 s）或尚未 PutEnd 的物件不逐出。〔文件〕〔複核補充〕另有 soft pin：Put 時可標記（例：system prompt），逐出時先逐出未 pin 的（同檔 §Soft Pin L377–379）。 | 同檔 §Eviction Policy L359–375 |
| SSD 層 | 在 real client 內背景執行：Master 挑出**要從記憶體逐出的物件**，heartbeat 執行緒把它們寫到 SSD；Get 找不到記憶體副本時退回 SSD。SSD 層自身的容量逐出預設關閉。〔文件〕 | `docs/source/design/store/ssd-offload.md` L7、L54–90、L163–167 |
| Transfer Engine | Segment（RAM segment：DRAM／VRAM；NVMeoF segment）＋BatchTransfer；傳輸後端 TCP、RDMA、EFA、NVMeoF、NVLink、HIP、SHM 等。〔文件〕 | `docs/source/design/transfer-engine/index.md` |
| 引擎 | vLLM 內建 `MooncakeStoreConnector`、`MooncakeConnector`（PD）；SGLang HiCache 的 L3 後端；LMCache 的遠端後端。〔程式碼〕〔文件〕 | [V] `factory.py`；[S] hicache_design；[L] `storage_backends/mooncake.rst` |

### B4. 比較表

| 系統 | 層級 | 粒度 | 寫入策略（預設） | 逐出（預設） | 非前綴重用 | 引擎 | 授權 |
|:--|:--|:--|:--|:--|:--|:--|:--|
| LMCache | GPU→CPU→本地碟→遠端（MP：L1＋L2 adapters） | chunk 256 token | 寫穿：prefill 的完整 chunk 寫到所有啟用層；decode 不存 | LRU（in-process；另有 LFU／FIFO／MRU）；MP 必須指定 LRU／ARC／… | 有（CacheBlend） | vLLM（LMCacheConnectorV1、LMCacheMPConnector）、SGLang | Apache-2.0 |
| vLLM OffloadingConnector | GPU→CPU（Tiering：→fs／obj／p2p） | GPU block 或 `block_size` 指定的 chunk | 全存（`store_threshold=0`）；可設 N 次才存 | LRU（可 ARC、自訂） | 無 | vLLM | Apache-2.0 |
| SGLang HiCache | L1 GPU→L2 主機→L3 後端 | page（`--page-size`） | write_through（可 selective＝2 次、write_back） | radix LRU（可 LFU／SLRU／priority／TLRU） | 無〔判讀：文件只談前綴〕 | SGLang | Apache-2.0 |
| Dynamo KVBM（已棄用） | G1→G2→G3→G4 | block | 依 policy；磁碟預設頻率 ≥2 | 〔未查證〕 | 無〔判讀〕 | vLLM（DynamoConnector）、TRT-LLM | Apache-2.0（另有第三方檔案例外） |
| Mooncake Store | 叢集 DRAM 池→本地 SSD | 物件（由整合方決定） | Put 寫入記憶體；SSD 在記憶體逐出時才寫 | 近似 LRU，90% 水位逐出 5% | 無〔判讀〕 | vLLM、SGLang、LMCache 後端 | Apache-2.0 |

出處：B1–B3 各格；授權依 `gh api repos/<repo>` 的 SPDX 與 Dynamo `LICENSE` L7–11。〔程式碼〕〔文件〕

---

## 與既有整理不一致

| 既有檔 | 原寫法 | 本卡查證 | 出處 |
|:--|:--|:--|:--|
| `docs/research_20260924/workloads_eval.md` §4 LMCache 列 | multi_round_qa「QPS（到達分布未查證）」 | 每位使用者以固定間隔 `num_users/qps` 發問、使用者以固定間隔加入，**不是 Poisson**〔複核：✅ 指控成立。workloads_eval L333 原文確為「QPS（到達分布未查證）」；程式 L82、L326–337、L362–367 確認固定間隔，且前一個請求未完成時不送（閉迴路＋最小間隔）〕 | [L] `multi-round-qa.py` L82、L314–338、L362–367 |
| 同檔 §4 SGLang 列 | mooncake 子集與 slowdown 參數描述正確，但沒提 prompt 怎麼生 | mooncake 模式**不用 `input_length`**，每個 hash 換成「id＋128 個 hi」；只在 `--backend sglang` 生效；忽略 `--request-rate`〔複核：✅ 指控成立（屬遺漏，非錯誤）。`datasets/mooncake.py` L84–87、`serving.py` L1548–1554〕 | [S] `datasets/mooncake.py` L80–90；`serving.py` L1546–1556 |
| 同檔 §4 SGLang 列 | 「HiCache 另有其評測（未查證）」 | 設計文件只連到 LMSYS 部落格（2025-09-10），本卡未讀該部落格，仍〔未查證〕〔複核：✅ 一致。設計文件 L12 確實只連到該部落格；複核者也未讀該部落格〕 | [S] hicache_design §Why |
| 同檔 §4 vLLM 列 | timed_trace「預設 16-token chunk」正確 | 補充：用 Mooncake trace 時若不改成 512，prompt 會被截成約 1/32，且不會報錯〔複核：✅ 成立。`_expand_prompt` 每個 hash 只展開 chunk_size 個 token、hash 用完就 break（L1489–1511），`sample()` 不比對 `input_length`（L1551–1565）〕 | [V] `datasets.py` L1482–1513 |
| `SOTA_ANALYSIS_20261006`（sota.txt §2.3） | HiCache 寫入「被存取超過門檻（預設 2）次才備份（預設）」 | 目前 SGLang main 的**預設是 `write_through`**；門檻 2 只用於 `write_through_selective`。Strata 論文描述的若是當時版本，需回原文確認〔未查證〕〔複核：✅ 對 SGLang `662879e4` 而言成立（`memory.py` L118–124 預設 `"write_through"`；`unified_radix_cache.py` L501–503 只有非 write_through 時門檻才是 2）。sota.txt L88–89 原文確為此寫法。Strata 原文未重讀〕 | [S] `memory.py` L118–124；`unified_radix_cache.py` L501–503 |
| 同上 | 「每一層的逐出都用 LRU」 | 預設是 LRU，但 radix 逐出可選 lfu、slru、priority、tlru〔複核：✅ 成立（屬過度簡化）。`memory.py` L25–42；另 L3 的逐出由各儲存後端決定，例如 Mooncake 是近似 LRU＋soft pin〕 | [S] `memory.py` L24–42 |
| `RESEARCH_INTRO_20261006`（intro.txt §1.2） | 「vLLM、SGLang、LMCache、NVIDIA Dynamo 都有把 KV 卸載到 CPU／SSD 的機制」 | Dynamo 的 KVBM 已於 v1.5.0 棄用、預定 v1.6.0 移除；Dynamo 現在的建議是用引擎原生卸載或 LMCache／HiCache／FlexKV〔複核：⚠️ 部分成立。棄用屬實，但 v1.5.0 release note 寫 `kvbm` wheel 本版「ship unchanged」，且 Dynamo 透過 LMCache／HiCache／FlexKV 仍提供卸載；intro.txt L91 的句子目前不算錯，只是 v1.6.0 後會過時。建議改寫成「Dynamo 原有 KVBM（v1.5.0 起棄用），改走引擎原生卸載」〕 | [D] `deprecations.mdx` L20–26 |
| MLCommons 部落格 vs 規則 | DeepSeek-R1 部落格寫「99% of the FP8 reference」；Llama3.1-8B 部落格寫「99% of FP16」 | 規則表分別寫「99% of FP16」與「99% of FP32 and 99.9% of FP32」。以規則為準〔複核：✅ 成立。WebFetch 部落格原句與 [M-rules] L266、L270 皆核對一致〕 | [M-blog]；[M-rules] L266、L270 |
| 〔複核補充〕`workloads_eval.md` §4 SGLang 列 | 「**`--use-trace-timestamps` 只對 mooncake 有效**」 | 在 SGLang `662879e4` 這個旗標**對排程沒有作用**：`benchmark()` 呼叫 `get_request(input_requests, request_rate)` 時沒傳它（L1560），只影響輸出標籤（L1674、L1847）；mooncake 的 trace 時間排程由 `backend=="sglang"` 決定（L1548–1554）。workloads_eval 的寫法照抄了 help 字串 | [S] `benchmark/serving.py` L1086–1123、L1548–1560、L2428–2432 |
| 〔複核補充〕`workloads_eval.md` §4 MLPerf v6.0 列 | 跨請求重用「未規範」 | 標準 LLM 項目的規則 FAQ **明文禁止跨 query 快取**：KV cache 只能用在單一 query 內，每個輸入都要完整計算。所以應寫「禁止」而非「未規範」；CPU／SSD 分層確實未規範。v6.1 Agentic 的 prefix cache 規則本卡未讀〔未查證〕 | [M-rules] `d3eba2f2` L895–897 |

---

## 對 PoC 的建議〔判讀〕

1. **主量測工具用 vLLM 自己的 `vllm bench serve`，版本鎖在本專案的 vLLM（依 repo 的 CLAUDE.md 為 v0.28.0）。** 理由：TPOT 與 MLPerf LoadGen 定義一致、有 goodput、有 `timed_trace` 與 `prefix_repetition`。但 v0.28.0 沒有 #55508 的時間修正，**一律用 `--backend openai`（completions）而不是 chat**，並在 manifest 註明「TTFT 為 completions 舊時鐘（誤差量級約數十 µs，作者實測）」。〔複核修正〕量級寫錯：PR #55508 作者實測 completions 的殘差約 **−0.35 µs**（次微秒），約 24 µs 的是 **chat** 的 usage chunk 殘差。manifest 應寫「completions 端點，TTFT 第二次讀時鐘，PR #55508 實測殘差約 0.35 µs」。另 `timed_trace` 本來就只接受 `vllm`／`openai`（completions）backend（`serve.py` L2144–2149），與此建議一致。
2. **manifest 必填欄位**：工具名＋commit／版本；endpoint 類型；`--dataset-name` 與其長度參數；`--random-range-ratio` 的實際區間（直接抄 log 的 "Sampling input_len from [a, b]"）；`--request-rate`、`--burstiness`、是否 ramp-up、`--max-concurrency`；`--ignore-eos` 是否生效；`--seed`；`--num-warmups` 與 `--ready-check-timeout-sec`；percentile 清單；`--goodput` 字串；伺服器端 prefix caching 開關、`--kv-offloading-size`／kv-transfer-config 全文、LMCache 設定檔全文；量測前是否呼叫 `/reset_prefix_cache`。
3. **不要把 SGLang、AIPerf、GuideLLM 的數字和 vLLM 的直接放同一張表**，除非先換算（§A5）。若要引用別人的 HiCache 結果，先確認他們的 `--random-range-ratio`：SGLang 預設 0 代表平均長度只有一半。
4. **trace 重播要寫斷言**（呼應 CLAUDE.md 規則 6）：自己寫 loader 或用 vLLM `timed_trace` 時，設 `--timed-trace-chunk-hash-size 512`、`--timed-trace-sec-multiplier 0.001`，並檢查每筆 `len(prompt_ids)==input_length`（vLLM 程式不檢查）。GuideLLM 已內建 `ceil(input/512)==len(hash_ids)` 檢查，可以當交叉驗證工具。
5. **暖機與快取隔離**：量 prefix／offload 命中時，(a) 暖機用與量測不重疊的 prompt（vLLM 的暖機只會重送第一個請求，要自己寫），(b) 每輪前清 GPU prefix cache（`VLLM_SERVER_DEV_MODE=1` 的 `/reset_prefix_cache`）並另外確認 CPU／LMCache 層也清空——**`/reset_prefix_cache` 是否清 OffloadingConnector／LMCache 的 CPU 層〔未查證〕**。〔複核補充：已查證程式路徑〕(1) 不帶參數的 `POST /reset_prefix_cache` **只清 GPU prefix cache**；要加 `?reset_external=true` 才會呼叫 connector 的 `reset_cache()`（`vllm/entrypoints/serve/dev/cache/api_router.py` L20–44，main 與 v0.28.0 相同；`v1/core/sched/scheduler.py` main L2805–2870）。(2) `OffloadingConnector` 有實作 `reset_cache()`（清掉所有已存 chunk；v0.28.0 `offloading_connector.py` L201、main L230–233、`offloading/scheduler.py` L1988–1989）。(3) vLLM 內的 `LMCacheConnectorV1` 與 `LMCacheMPConnector` **沒有覆寫** `reset_cache()`，base 回 `None`（`kv_connector/v1/base.py` L751–763），而排程器只在回傳 `False` 時才判失敗（main L2863、v0.28.0 L2566）→ **API 回 `success: true` 但 LMCache 沒被清**。這正是 CLAUDE.md 規則 7「查不到不等於沒有」的情形，LMCache 的清空要走 LMCache 自己的介面〔未查證：LMCache 側的清除 API〕。以上是讀程式碼的推論，未實跑〔判讀〕。
6. **單請求 × context 遞增（本專案主軸）**：用 `--max-concurrency 1 --request-rate inf` 或自寫腳本；報 TTFT 的完整分布而非只報 p99（`--metric-percentiles 50,90,99` 並 `--save-detailed`）。
7. **Cake 式「前段重算＋後段載入」**：vLLM connector 的 `get_num_new_matched_tokens` 語意是「已算部分之後的**連續前綴**」，排程器會算剩下的尾段。Cake 的順序剛好相反（前段算、後段載），在現行 API 下無法直接表達；能直接用的只有「載入與計算逐層重疊」（`wait_for_layer_load`／LMCache `use_layerwise`）。要做 Cake 還原，大概需要改排程器或自訂 connector＋model runner，請與 E03（Cake 原文）交叉確認後再定 PoC 架構。
8. **LMCache 的預設會影響「寫什麼」**：預設不存 decode KV、不存未滿 256 的尾段、所有層都寫。若研究的是「寫入時決定存哪一層」，baseline 要明寫這三個設定；MP 模式的 L2 預取預設不留在 L1，也會影響第二次命中的位置。
9. **用 OffloadingConnector 做 LRU／ARC baseline 時**，`store_threshold` 與 `block_size` 都要寫進 manifest；`store_threshold≥2` 就是一個現成的「依次數才寫」baseline，可以直接當對照組。〔複核補充〕注意 v0.28.0 的計數點是 `lookup()`（被查詢次數），main 是 `prepare_store`（被提議儲存次數），見 B2 extra_config 格；兩版的同一個 N 不等價。

---

## 未查證清單

1. vLLM `/reset_prefix_cache` 是否同時清掉 OffloadingConnector 的 CPU 層與 LMCache 的快取。〔複核：程式路徑已查證，見 PoC 建議 5——預設不清；`reset_external=true` 會清 OffloadingConnector，但 LMCache connector 未實作、仍回 success。未實跑〕
2. GuideLLM 的 output token 數來源（usage 或 client tokenizer）與預設 percentile 清單。〔複核：已查證，見 A3 GuideLLM 格與 A5 表〕
3. LLMPerf 的 percentile 輸出格式。〔複核：已查證，p25／p50／p75／p90／p95／p99，見 A5 表〕
4. AIPerf 的預設統計欄位完整清單（只看到教學範例的 avg／min／max／p99／p90／p50／std）。
5. Dynamo KVBM 主機層與磁碟層的逐出順序。
6. HiCache 的官方效能數字（LMSYS 2025-09-10 部落格未讀）；Strata 論文對 HiCache 預設寫入策略的描述。
7. MLPerf Inference v6.0／v6.1（含 Agentic）結果頁與 v6.1 Agentic 規則：本卡只讀了 `inference_rules.adoc`，其中沒有 Agentic datacenter 項目；`workloads_eval.md` 的 Agentic 描述本卡未重驗。
8. vLLM PR #7372、#8164、#33907、#12064（SGLang）等只讀了標題，未讀 diff。〔複核：#7372 已讀 diff（見 A1.5 #2）；#33907 已讀 PR 本文（前綴改用 decode-encode 修正長度，與標題一致）；#8164、SGLang #12064／#12156 仍只核對了標題與合併日期〕
9. LMCache MP 模式的 decode 寫入、未滿 chunk 處理是否與 in-process 相同（只讀了 in-process adapter）。
10. SGLang HiCache 是否支援非前綴重用（文件未提，表中標〔判讀〕）。
11. vLLM `timed_trace` 的 `hash(str)` 種子跨行程不可重現：依 Python 語意推論，未實跑。

---

## 狀態

- 抽取：完成（E02，2026-10-07）。
- 複核：完成（V02，2026-10-07），見下方「複核紀錄」。

---

## 複核紀錄

- **複核者**：V02（未參與抽取；未讀抽取者的筆記或 `scratchpad/E02/`）
- **日期**：2026-10-07
- **方法**：把卡上寫的每個 SHA 用 `git fetch --depth 1 origin <sha>` 抓到 `scratchpad/V02/`，逐檔逐行讀原始碼與 repo 內文件：vLLM main `31e2443c`、tag v0.28.0 `2cf0a691`、tag v0.29.0 `98dff2a8`；SGLang `662879e4`；AIPerf `f3a76e5a`；GuideLLM `246f3b0f`；LLMPerf `f1d6bed4`；inference_policies `d3eba2f2`；mlcommons/inference `3fbc3299`；LMCache `8c77a6f7`；Dynamo main `7d7d3dad` 與 tag v1.4.0 `03014943`；Mooncake `245e7106`。所有 SHA 都存在且與卡上一致。PR 用 `gh pr view`／`gh pr diff`；release 用 `gh release view`；MLCommons 部落格與 vLLM 部落格用 WebFetch（curl 被 Cloudflare 擋）。**沒有實跑任何工具**，執行期行為都是讀程式碼推論。
- **檢查項數**：179 項（來源清單 14、重點摘要 8、A1 共 44、A2 13、A3 20、A4 13、A5 7、B1 12、B2 7、B3 18、B4 6、與既有整理不一致 8、PoC 建議 9）。
- **結果**：✅ 168、❌ 5、⚠️ 6；另有 〔複核補充〕 約 25 處，新增「與既有整理不一致」2 列。

### ❌（錯誤，已改正）

1. **LLMPerf ITL 的分母**（重點摘要 1、A3 LLMPerf ITL 格、A5 第 3 列，同一個錯在三處，算 3 項）：原文「除以輸出 token 數」→ 改為「除以 client 收到的 SSE data chunk 數（`tokens_received`，含 role-only／空 content 的 chunk）；`token_benchmark_ray.py` L121 先除，L124 才用 Llama tokenizer 覆寫 `NUM_OUTPUT_TOKENS`」。出處：llmperf `f1d6bed4` `src/llmperf/ray_clients/openai_chat_completions_client.py` L74–117；`token_benchmark_ray.py` L113–126。
2. **A4 MLPerf 判讀「不規範 KV 分層與重用」** → 改為「規則 FAQ 明文禁止跨 query 快取，KV cache 只能用在單一 query 內；分層確實未規範」。出處：inference_policies `d3eba2f2` `inference_rules.adoc` L895–897。
3. **PoC 建議 1 的誤差量級**：「completions 舊時鐘誤差約數十 µs」→「completions 殘差約 −0.35 µs；約 24 µs 的是 chat 的 usage chunk 殘差」。出處：vLLM PR #55508 本文的 real-socket 表格。

### ⚠️（不精確、說得太滿或找不到出處，已補註）

1. 重點摘要 4：24 µs 未區分端點 → 補上 completions −0.35 µs vs chat +24.09 µs；另補 v0.29.0 `endpoint_request_func.py` L236、L421 也還是舊行為。
2. 重點摘要 7：「沒有一個系統把不存當成寫入選項」說得太滿 → 補上 LMCache `lmcache.skip_save`（`vllm_v1_adapter.py` L332–340；MP 模式不保證生效，`mp/configuration.rst` L42–48）、vLLM `store_threshold`、KVBM 磁碟過濾、HiCache write_back／selective；改成「沒有依重算 vs 傳輸成本決定不存」。
3. A1.3 random：L' 減 special token 只適用輸入，輸出用 L 且下界至少 1（`datasets/utils.py` L68–77）。
4. A3 GuideLLM 量測窗：TTFT／ITL 的納入規則是「區間與窗口部分重疊」，不是「事件落在窗口內」（`docs/en/guides/metrics.md` L127–134）。
5. B2 部落格：「存與載都非同步」——部落格只明說卸載非同步，載入非同步的句子沒找到 → 標〔未查證〕。
6. 不一致表 intro.txt §1.2 列：Dynamo 的指控只部分成立（v1.5.0 的 `kvbm` wheel「ship unchanged」，Dynamo 仍可透過 LMCache／HiCache／FlexKV 卸載）。

### 三大發現的複核結論

- **指標同名不同義**：✅ 成立。vLLM TPOT（`serve.py` L625–629）＝AIPerf ITL（`inter_token_latency_metric.py` L54–75）＝GuideLLM ITL（`request_stats.py` L337–354）＝LoadGen TPOT（`logging.cc` L475–476），算式都是 `(last−first)/(n−1)`；vLLM／SGLang ITL＝chunk 間隔攤平＝AIPerf ICL；GuideLLM TPOT 含 TTFT。唯一的錯是 LLMPerf ITL 的分母（見 ❌1）。仍要注意 n 的來源不同（vLLM 用伺服器 usage、AIPerf 預設用 client tokenizer、GuideLLM 用 usage 否則 iteration 數），以及百分位取法不同（LoadGen 用最近秩，vLLM 用線性內插）。
- **`--random-range-ratio`**：✅ 成立。vLLM 預設 `"0.0"`、窗口 `[floor(L'(1−r)), ceil(L'(1+r))]`、r∈[0,1)（`datasets.py` L1924–1931；`datasets/utils.py` L64–97）；SGLang 預設 0.0、`randint(max(int(L·r),1), L+1)`（`datasets/common.py` L56–64；CLI L2380–2385）；`--gsp-range-ratio` 預設 1.0 並有 WARN 註解（L2696–2702）；PR #16126 的 diff 確認舊版 vLLM 預設 1.0、窗口 `[L·r, L]`。AIPerf 文件也獨立寫出同一差異。
- **`timed_trace`**：✅ 成立。`--timed-trace-chunk-hash-size` 預設 16（`datasets.py` L1715–1723，help 寫 Moonshot 512），`_expand_prompt` hash 用完就停、不補到 `input_length`（L1482–1513），v0.28.0 相同（L1452）；SGLang mooncake 每個 hash 換成 `"{id}"`＋128 個 "hi"，不用 `input_length`，只有 `--backend sglang` 才照 trace 時間（`datasets/mooncake.py` L84–87；`serving.py` L1548–1554）。另補兩點：GuideLLM 要求秒、Mooncake trace 是毫秒；SGLang 的 `--use-trace-timestamps` 在此 SHA 不影響排程。

### 逐條修改清單（原→新＋出處）

| 位置 | 原內容 | 新內容 | 出處 |
|:--|:--|:--|:--|
| 重點摘要 1 | LLMPerf ITL `(last−start)/n` | n＝SSE chunk 數，不是 token 數 | llmperf `f1d6bed4` client L83、L112；`token_benchmark_ray.py` L117–124 |
| 重點摘要 2 | — | 補充：AIPerf 文件的同一差異描述 | aiperf `cli-options.md` L944、L954–955 |
| 重點摘要 3 | — | 補充：Mooncake 時間單位 ms、GuideLLM 要秒；SGLang 暖機用 512 個 hi | Mooncake `FAST25-release/README.md` L52、L55；guidellm `datasets.md` L196；sglang `serving.py` L1443–1448 |
| 重點摘要 4 | 「誤差約 24 µs」 | completions −0.35 µs／chat +24.09 µs；v0.29.0 也是舊行為 | PR #55508；v0.29.0 `endpoint_request_func.py` L236、L421 |
| 重點摘要 7 | 「沒有一個系統把不存當寫入選項」 | 有機制層面的不存開關，但沒有依重算成本決定的 | LMCache adapter L332–340；`mp/configuration.rst` L42–48 |
| 重點摘要 8 | — | 補充：GitHub release 2026-09-21；kvbm wheel 本版照常出貨 | `gh release view v1.5.0` |
| A1.1 client 排隊 | — | 補充：含排隊 E2EL 需 max-concurrency 且 rate≠inf | `serve.py` L1320–1327 |
| A1.3 random | L'＝L−special（輸入輸出不分） | 只有輸入減；輸出下界 1 | `datasets/utils.py` L68–77 |
| A1.5 #2 | 只看標題 | 讀了 diff：舊版把 TTFT 也 append 進 ITL | `gh pr diff 7372` |
| A1.5 表尾 | — | 補充：12 列日期全數核對；區分新增／預設／算法改變 | `gh pr view` |
| A2 到達 | — | 補充：`--use-trace-timestamps` 不影響排程 | sglang `serving.py` L1089–1110、L1548–1560、L1674、L1847 |
| A2 percentile | 「官方文件仍寫舊的組合」 | 補充：文件 E2E 少列 p90，新舊都不完全符合 | `gh pr diff 27662`；`bench_serving.mdx` L240–242 |
| A2 預設行為 | — | 補充：CI 條件限 sglang backend；對 vLLM 不帶 reset_external | `serving.py` L1013–1017、L1513–1515 |
| A3 AIPerf ITL | 只有〔文件〕 | 補上程式碼出處 | aiperf `metrics/types/*.py` |
| A3 AIPerf 合成長度 | — | 補充：ignore_eos 佐證；`--random-corpus-style` | `metrics-reference.md` L1467；`cli-options.md` L942–955 |
| A3 GuideLLM 指標 | — | 補充：n 的來源；指定輸出長度時送 ignore_eos | `request_stats.py` L237–246；`request_handlers.py` L612–624、L778–835 |
| A3 GuideLLM 量測窗 | 「事件落在窗口內」 | 「區間與窗口部分重疊」 | `metrics.md` L127–134 |
| A3 GuideLLM trace | — | 補充：InvalidRowError 語意、ms→s、replay 預設 idle_gap | `schemas/base.py` L47–48；`benchmark.md` L217–224；`profiles/replay.py` L28–30 |
| A3 LLMPerf 狀態 | 2024-12-08 | 補充：UTC 為 2024-12-09 | `gh api …/commits/f1d6bed4` |
| A3 LLMPerf ITL | 除以輸出 token 數 | 除以 SSE chunk 數 | 同上 ❌1 |
| A4 TPOT 定義 | — | 補充：LoadGen 最近秩 vs np.percentile 內插 | `results.cc` L105–136 |
| A4 GPT-OSS-120B | — | 補充：mlperf.conf interactive TPOT 15 ms ≠ 規則 20 ms | `mlperf.conf` L162–164 |
| A4 判讀 | 「不規範 KV 分層與重用」 | 禁止跨 query 快取；分層未規範 | `inference_rules.adoc` L895–897 |
| A5 第 3 列 | LLMPerf ITL＝E2E/n | n＝chunk 數 | 同 ❌1 |
| A5 輸出 token 數 | GuideLLM〔未查證〕 | usage，無則 iteration 次數 | `request_stats.py` L237–246 |
| A5 預設 percentile | GuideLLM「多種統計」、LLMPerf〔未查證〕 | GuideLLM p0.1–p99.9＋CI；LLMPerf p25–p99 | `metrics.md` L140–175；`token_benchmark_ray.py` L231 |
| B2 extra_config | — | 補充：store_threshold 計數點 v0.28.0（lookup）≠ main（prepare_store） | main `cpu/manager.py` L122–141、L250–271；v0.28.0 `cpu/manager.py` L113–121 |
| B2 分層 | fs／obj／p2p／example | 另有 kvcr | `tiering/factory.py` L146–170 |
| B2 部落格 | 存與載都非同步 | 載入非同步〔未查證〕 | WebFetch vLLM 部落格 |
| B3 KVBM 寫入策略 | — | 補充：v1.4.0 文件自稱 write-through；關閉磁碟過濾的環境變數 | v1.4.0 `kvbm-guide.md` L271、L278–282 |
| B3 Mooncake 逐出 | — | 補充：soft pin | `mooncake-store.md` L377–379 |
| 不一致表 8 列 | 無判定 | 逐列加 ✅／⚠️ | 見各列 |
| 不一致表 新增 2 列 | — | workloads_eval 的 `--use-trace-timestamps`、MLPerf「未規範重用」 | sglang `serving.py` L1560；`inference_rules.adoc` L895–897 |
| PoC 1 | 「約數十 µs」 | 約 0.35 µs（completions） | PR #55508 |
| PoC 5 | `/reset_prefix_cache` 清不清 CPU 層〔未查證〕 | 預設不清；`reset_external=true` 清 OffloadingConnector；LMCache connector 未實作但回 success | `api_router.py` L20–44；`scheduler.py` main L2805–2870、v0.28.0 L2553–2573；`base.py` L751–763 |
| PoC 9 | — | 補充：兩版 store_threshold 語意不同 | 同 B2 |
| 未查證清單 1–3、8 | 未查證 | 標上已查證內容 | 見各條 |

### 因時間或範圍沒有檢查的項目（如實列出）

- Strata 論文對 HiCache 預設寫入策略的描述（屬 E04 範圍，未讀原文）。
- LMSYS HiCache 部落格（2025-09-10）、MLPerf v6.0／v6.1 結果頁與 v6.1 Agentic 規則（未讀）。
- vLLM PR #8164、#9338、#10105、#26941、#26943 的 diff（只核對了標題、狀態與合併日期；#28227、#30975、#12288、#5263、#16126 讀了本文或 diff）；SGLang #12064、#12156 的 diff（只核對了標題與日期）。
- SGLang chat-completions 函式的 TTFT 細節（只讀了 completions 函式 L284–400）。
- LMCache MP 模式的 decode 寫入與未滿 chunk 處理；LMCache 端的清快取 API。
- GuideLLM 上層是否 catch `InvalidRowError`（是跳過該列還是中止）。
- AIPerf 預設輸出的統計欄位完整清單。
- 所有執行期行為（暖機、reset、trace 重播）都是讀程式推論，**沒有實跑**。
