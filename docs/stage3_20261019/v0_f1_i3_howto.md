# V0／F1／F2／I3 操作手冊：m16 vLLM serving 工具

> 寫於 2026-10-19（UTC）。**本頁沒有任何 GPU 量測**：工具只在 CPU 上對假 server 測過（§8）。
> 標〔估計〕的時間都是拿既有的 vLLM 量測推的，附出處；找不到依據的寫 NOT_MEASURED。
> 實驗設計見 `experiment_matrix.md`（V0、F1、I3 三張卡）；F2 是協調者 10/19 追加的（文獻 L3 §7 X2，vLLM issue #44294）。

---

## 0. 檔案

| 檔案 | 用途 | 會不會碰 GPU |
|:--|:--|:--|
| `code/m16_workload.py` | 產生工作負載 JSONL：`gen-prefix`（F1）、`gen-mooncake`、`gen-agentic`（I3）、`gen-burst`（F2）、`mooncake-deltas`、`--self-test` | 不會（純 Python） |
| `code/m16_vllm_bench.py` | `serve`（只組指令）、`wait-ready`、`run`、`v0check`、`count-log` | client 不會；`--guard on` 時呼叫 amd-smi |
| `code/m16_serve_patched.py` | `--rocm-attn-patch` 時包住 `vllm serve`，在每個 server 行程裝上 F1 的 ROCm attention 修法 | **會**（它就是 server） |
| `code/tests/m16_bench_cpu.py` | 假的 OpenAI 相容 streaming server＋端到端測試 | 不會 |

- client 端全部不 import torch／vllm（不打開 `/dev/kfd`，AMD 的 gpu_guard 才不會把它算成入侵者）。設 `M16_FORBID_GPU_IMPORTS=1` 會在載入時裝上 import 阻擋器，測試一律這樣跑。
- Python：`/mlsteam/workspace/venv/tiara-v028/bin/python`（有 aiohttp 3.14.3；**沒有安裝任何東西**）。server 用同一個 venv 的 vLLM 0.28.0。
- 輸出：每個 run 的原始檔在 `$TIARA_RUNS/<run_id>/`（`requests.csv`、`summary.json`、`metrics_before.txt`／`metrics_after.txt`、`gpu_guard.json`、`vllm_server.log`）；乾淨的 run 另外寫 `--out-dir`（`requests_<run_id>.csv`、`summary_<run_id>.json`）。**每個 CSV 都有 `run_id`、`ts` 欄**。

### 0.1 vLLM 0.28 原始碼查到的名稱（`/mlsteam/workspace/src/vllm-v0.28.0`）

| 項目 | 名稱 | 出處 |
|:--|:--|:--|
| connector | `kv_connector="OffloadingConnector"`、`kv_role="kv_both"`、`engine_id` | `config/kv_transfer.py:30,93-94` |
| CPU 層 | `spec_name="CPUOffloadingSpec"`、`cpu_bytes_to_use`（必填）、`eviction_policy`（lru／arc）、`store_threshold`、`max_tracker_size`、`block_size`、`blocks_per_chunk` | `v1/kv_offload/cpu/spec.py:80-132`、`kv_connector/v1/offloading/config.py:68-69` |
| 分層 | `spec_name="TieringOffloadingSpec"`、`secondary_tiers=[{"type":"fs","root_dir":…}]`（另有 `n_read_threads`、`n_write_threads`，預設 16）；**fs 層沒有容量參數** | `v1/kv_offload/tiering/spec.py:1-45`、`tiering/fs/manager.py:109-118` |
| 寫入方式 | 存進 CPU 時「一律寫到所有二級層」；`reset_cache` 不清二級層 | `tiering/manager.py:1-20、821-831` |
| CPU 映射檔 | `/dev/shm/vllm_offload_{engine_id}.mmap` | `v1/kv_offload/cpu/shared_offload_region.py:94` |
| prefix cache metrics | `vllm:prefix_cache_queries`、`vllm:prefix_cache_hits`、`vllm:external_prefix_cache_queries`、`vllm:external_prefix_cache_hits` | `v1/metrics/loggers.py:584-628` |
| prompt token | `vllm:prompt_tokens`、`vllm:prompt_tokens_cached`、`vllm:prompt_tokens_by_source{source=local_compute｜local_cache_hit｜external_kv_transfer}` | `loggers.py:670-702`、`v1/metrics/stats.py:316-320` |
| offload metrics | `vllm:kv_offload_load_bytes`、`_load_time`、`_store_bytes`、`_store_time`、`_allocation_failure`、`vllm:kv_offload_lookup_async_delay_seconds`（histogram）；舊名 `vllm:kv_offload_total_bytes{transfer_type=CPU_to_GPU｜GPU_to_CPU}` | `kv_connector/v1/offloading/metrics.py:22-38、132-134` |
| CPU 層 gauge | `vllm:kv_offload_cpu_cache_usage_perc` 等 | `v1/kv_offload/cpu/common.py:6-11` |
| tiering metrics | `vllm:kv_offload_tiering_read_bytes／write_bytes／chunk_queries／chunk_hits{tier=…}` 等 | `v1/kv_offload/tiering/base.py:37-56` |
| counter 字尾 | prometheus_client 在 counter 名稱後加 `_total`；labeled 子項第一次觀測才出現 | venv `prometheus_client/exposition.py`（`mtype=='counter'` → `+'_total'`） |
| 每請求命中 | `usage.prompt_tokens_details.cached_tokens`（＝本機＋connector），要 `--enable-prompt-tokens-details` | `entrypoints/openai/completion/serving.py:453-456`、`stats.py:284` |
| reset | `POST /reset_prefix_cache?reset_external=true｜false`，回 `{"success": bool}`；有 block 被占用或非同步搬移中時回 false；只在 `VLLM_SERVER_DEV_MODE=1` 時有這個路由 | `entrypoints/serve/dev/cache/api_router.py:20-43` |
| F2 延後邏輯 | `_chunks_being_loaded`（只在開 prefix caching 時建立）；命中的 chunk 正在載入就延後，訊息 `"Delaying request %s since some of its chunks are already being loaded"`（`logger.debug`） | `kv_connector/v1/offloading/scheduler.py:544-546、835-857、1099-1100、1569-1570、1694-1695`（0.19.1 是 `:126-140` 的 `_blocks_being_loaded`，訊息寫 blocks） |
| 其他延後訊息 | `"Delaying request %s since it still has in-flight transfers"`、`"Offloading manager delayed request %s as backend requested"`、`"Request %s hit %s offloaded tokens after %s GPU hit tokens"` | 同檔 `:964`、`:830`、`:860-865` |
| logger 名稱 | `vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler`；用 `VLLM_LOGGING_CONFIG_PATH`（dictConfig）只把它調到 DEBUG | `vllm/logger.py:41-71、156-201` |
| 容量下限 | `--num-gpu-blocks-override` 放不下一個 `max_model_len` 的請求 → 啟動就丟 ValueError | `v1/core/kv_cache_utils.py:731-760、2162-2196` |

---

## 1. 共通：一個 job 的殼

規則（`experiment_matrix.md` §6）：不動第二階段的佇列；用 `/mlsteam/data/tiara/gpu{0,1}.lock`；只在第二階段那張卡的佇列空了才開；每個 job ≤ 2 小時；計時 job 用 `GpuWatcher`。

```bash
#!/usr/bin/env bash
# GPU-NEED: one
# 用法（runner 裡）：m7run m16-<短名> bash <這支腳本> <參數>   → RUN_ID、TIARA_RUNS、TIARA_GPU 由 m7_env.sh／m7run 給
set -euo pipefail
REPO=/mlsteam/workspace/paper-hierarchical-kv-state
PY=/mlsteam/workspace/venv/tiara-v028/bin/python
B="$PY $REPO/code/m16_vllm_bench.py"
RD=$TIARA_RUNS/$RUN_ID
PORT=${PORT:-8016}

$B serve --run-dir "$RD" --model llama --apc on --offload cpu:64 --max-model-len 40960 --port $PORT --label <格子名>
setsid bash "$RD/serve_cmd.sh" &          # 非互動 shell 裡 setsid 不會再 fork → $! 就是 server，也是行程群組 leader
SPID=$!
stop() { kill -TERM -- -$SPID 2>/dev/null || true; for i in $(seq 60); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
         kill -KILL -- -$SPID 2>/dev/null || true; bash "$RD/serve_cleanup.sh"; }
trap stop EXIT
$B wait-ready --url http://127.0.0.1:$PORT --timeout 900 --pid $SPID
grep -E "Using .* backend|Overriding with|M16_ROCMFIX|GPU KV cache size|num_gpu_blocks" "$RD/vllm_server.log" > "$RD/backend.txt" || true
$B run --workload <W.jsonl> --server-config "$RD/serve_config.json" --out-dir <results/...> --guard-own-root $$ ...
```

- **guard 的 own_root**：server 和 bench 都要是同一支 job shell 的子孫（上面的寫法就是），否則 gpu_guard 會把自家 server 算成「同容器入侵者」。`--guard-own-root` 預設就是 bench 的父行程。
- 結束碼：0 完成；2 設定錯（含 server 的 `max_model_len` 和 serve_config 對不上、工作負載放不下）；3 有請求失敗（全部照實記錄後才結束）；4 contaminated（結果只留在 run 目錄、**不寫 `--out-dir`**、要重量）。runner 遇到非 0 照 CLAUDE.md 規則 2 停下來。
- 一個 server 可以跑很多格：每格 `run` 開頭預設會 `/reset_prefix_cache`（有卸載時連 CPU 層一起清；**fs 層不會清**，所以 tier 的格子 fs 目錄要每個 server 一個新的）。
- 每格開頭有 2 個和工作負載無關的暖機請求（`--server-warmup`，讓 Triton JIT 先跑），不計入。

---

## 2. V0 冒煙（`results/m16_stage3_mi300x/v0_vllm_smoke/`）

目標（卡片）：APC、CPU offload、tiering 在 MI300X 上能不能跑，命中 vs 不命中的輸出逐 token 相同。每個設定一個 server、一個 job：

| 步驟 | serve 參數（其餘同 §1） | v0check 的模式（auto 會自己判斷） | 看什麼 |
|:--|:--|:--|:--|
| V0-a | `--apc off` | none：miss vs miss2 | 同一 prompt 兩次輸出相同（確定性基準） |
| V0-b | `--apc on` | apc：miss → gpu_hit → miss2 | GPU 命中輸出相同、`cached_tokens ≥ L−32` |
| V0-c | `--apc on --offload cpu:64` | cpu：miss → gpu_hit → (只清 GPU) cpu_hit → miss2 | CPU 載入輸出相同、`kv_offload_load_bytes` 有增加；`backend.txt` 應該是 TRITON_ATTN |
| V0-d | V0-c ＋ `--rocm-attn-patch` | 同上 | 同上；`backend.txt` 要有 `Using ROCM_ATTN backend (selected via --attention-backend)` 和 `M16_ROCMFIX: registered … K/V tensors` |
| V0-e | `--apc on --offload tier:16:/var/tmp/m16_fs_$RUN_ID:64` | tier：…→ cpu_hit →(清 CPU，fs 留著) fs_hit | fs 載入輸出相同、`tiering_read_bytes` 有增加 |
| V0-f（選做） | V0-e 但 fs 放 NFS：`tier:16:$RD/fs:64` | 同上 | 真的從 NFS 讀（factcheck：之前的「NFS 級」是限速器參數） |

每個 job 裡跑兩次 v0check（亂數 token＋真文字）：

```bash
OUT=$REPO/results/m16_stage3_mi300x/v0_vllm_smoke
$B v0check --server-config "$RD/serve_config.json" --out-dir $OUT --lengths 8192 16384 32768 --n-prompts 3 \
           --max-tokens 32 --settle-s 3 --guard-own-root $$
$B v0check --server-config "$RD/serve_config.json" --out-dir $OUT --n-prompts 3 --max-tokens 32 --settle-s 3 \
           --text-file /mlsteam/data/tiara/datasets/longbench/data/narrativeqa.jsonl --text-chars 30000 --guard-own-root $$
```

- 判定（`v0check_<run_id>.json` 的 `verdict`／`overall`）：`*_match`＝輸出 token id 和 miss 完全相同；`*_cached`＝`cached_tokens ≥ L − 2×16`；`cpu_hit_loaded_bytes`、`fs_hit_tier_read_bytes`＝該段 `/metrics` 差值 > 0；`miss_cold`＝第一次真的沒命中；`determinism_miss_vs_miss2`。抓不到 metrics 的項目是 NOT_AVAILABLE（`overall`＝INCOMPLETE），不會被當成 PASS。
- **不一致時怎麼讀**：CSV 有 `first_diverge`（第幾個 token 開始不同）和 `miss_margin_at_diverge`（miss 那次在該位置 top-1 與 top-2 的 logprob 差）。F1 已經看到：亂數 token 的 prompt 讓第一個輸出 token 的機率很接近，**就算完全沒有 CPU 參與，輸出也偶爾不同**（`docs/research_20261010_followup/F1_offload_slowdown.md:269`）。所以 (1) 先看 V0-a 的 miss vs miss2；(2) margin 很小的分歧要看文字 prompt 那一輪；(3) 真正的「無損」證據是 F1 的逐 block 位元組比對（cpu50 與 rocmfix 都是 511／512，同檔 §4.5）。v0check 不重做位元組比對（那需要在 server 行程裡讀 KV）。
- 判定規則照卡片：V0 不過 → F1、I3 改用我們自己的 Restorer，並在 RUNLOG 寫明。
- **V0 順便量還原時間 R(L)**（I3 的「Nx」提前量要用）：V0-c／V0-d 的 v0check CSV 裡 `phase=cpu_hit` 的 `ttft_s`（3 個長度 × 3 個 prompt）擬合 `R(L)＝a＋b·L`，之後以 `--restore-est linear:<a>:<b>` 傳給 I3。只有 `timing_valid=1` 的列能用。
- 也記下 `backend.txt` 裡的 `GPU KV cache size: N tokens`（預設容量，F1 的「充足」）。

〔估計〕V0 每個 job：server 啟動約 2 分 21 秒（C6 的 `vllm_server.log`：14:13:24 → `Application startup complete` 14:15:45，其中 init engine 79.40 s、torch.compile 19.94 s；run `20261008-141324-m7-c6-vllm`）＋ v0check：每個 prompt 4–5 段，每段約 0.3–4.6 s（C6：8K 冷 443 ms、32K 冷 4,584 ms、32K 從 CPU 載入 1,080 ms；`docs/phase1_20261008/07_report.md:258-262`）＋ 每段 3 s settle → 9 個亂數 prompt＋3 個文字 prompt 約 4–6 分鐘。**每個 V0 job 約 7–9 分鐘，5 個約 40 分鐘**（卡片估 1 h）。

---

## 3. F1 前綴復用地圖（`results/m16_stage3_mi300x/f1_prefix_reuse/`）

### 3.1 格子

| 軸 | 值（卡片） | 工具參數 |
|:--|:--|:--|
| 重疊 | 0／25／50／75／100% | `gen-prefix --overlap 0 0.25 0.5 0.75 1.0`、`--n-groups 4` |
| prompt | 8K／32K | `--prompt-len 8192｜32768`；`--out-len 128` |
| 到達 | Poisson（多個 QPS）＋一次全部到 | `--arrival poisson --rate λ`、`--arrival burst` |
| 配置 | 無 APC、APC、APC＋CPU offload | `serve --apc off`、`--apc on`、`--apc on --offload cpu:<GiB>` |
| KV 容量 | 充足、緊、很緊 | 預設 profiler 容量、`--gpu-blocks ceil(r × WS)`，r＝1.5、0.75 |
| 指標 | 命中率、TTFT P50／P95、TPOT、吞吐、goodput（兩個 TTFT SLO）、CPU 搬運位元組 | `summary_<run_id>.json` 的 `ttft_s`、`tpot_s`、`throughput`、`goodput`、`server.metrics.series` |

建議的請求數：8K 每格 64 個、32K 每格 32 個（一格約 0.5–1M prompt token）。

**容量的定義要先決定（卡片原文有歧義）**：卡片寫「緊＝只放得下約 1.5 倍工作集、很緊＝0.75 倍」。我照字面實作成「GPU 容量＝r × 工作集」，工作集＝該格所有 prompt 的不同 16-token block 數＋輸出（工作負載 meta 的 `workset.ws_blocks_incl_output`）。r＝1.5 時其實全部放得下（考的是管理開銷與併發中的佔用），r＝0.75 才一定要淘汰。如果原意是「工作集是容量的 1.5 倍」（r＝1/1.5），請在跑之前改。
另一個選擇：容量是跟「每一格的工作集」走（每格重開 server，60 次以上的啟動，約 2.5 小時光是啟動），還是跟「同長度、重疊 0% 那格的工作集」走（每個長度一個值，18 個 server）。**建議後者**，分析時用每格自己的 `ws_blocks` 算實際的比值。

算容量的指令：

```bash
$PY code/m16_workload.py gen-prefix --n-req 32 --prompt-len 32768 --overlap 0 --n-groups 4 --out-len 128 --arrival burst --out W0.jsonl
$PY -c "import json,math;m=json.loads(open('W0.jsonl').readline())['_meta']['workset'];print({r:math.ceil(r*m['ws_blocks_incl_output']) for r in (1.5,0.75)})"
# → serve --gpu-blocks <值> --block-size 16（block size 要明寫，容量才算得準）
```

QPS 的取法（事先定、寫進 RUNLOG 再跑）：先跑每個長度「無 APC、充足、重疊 0%、burst」那格，λ_sat＝n_req ÷ makespan（`summary` 的 `throughput.req_per_s`）；Poisson 用 0.5×λ_sat 與 0.9×λ_sat。〔估計〕λ_sat：單請求 32K prefill 2.19 s（ROCM_ATTN、無 connector，`results/m10_restore2_mi300x/q5/vllm_kvdecode/vllm_kvdecode.csv`，run `20261018-153119-m10-rs2-q5d-llama-auto`）→ 約 15K token/s〔算術〕→ 32K 約 0.46 req/s、8K 約 1.8 req/s；開 connector 退回 TRITON_ATTN 時約一半（C6 32K 冷 4.58 s）。批次化之後的吞吐 NOT_MEASURED。
TTFT SLO（兩個，事先定）：建議＝V0 量到的單請求冷 prefill × 1.5 與 × 5（V0 之前的〔估計〕：32K 約 3.3 s／11 s；8K 用 C6 的 0.443 s → 0.66 s／2.2 s，那是 TRITON 路徑）。

### 3.2 展開成 job

一個 job＝一個 server（固定 配置 × 容量 × 長度），裡面依序跑 5 個重疊 × 3 個到達＝15 格：

```bash
for ov in 0 0.25 0.5 0.75 1.0; do
  for arr in "burst" "poisson --rate $L1" "poisson --rate $L2"; do
    W=$RD/w_${L}_${ov}_${arr%% *}.jsonl
    $PY $REPO/code/m16_workload.py gen-prefix --n-req $N --prompt-len $L --overlap $ov --n-groups 4 --out-len 128 \
        --arrival $arr --seed 0 --out $W
    $B run --workload $W --server-config "$RD/serve_config.json" --out-dir $REPO/results/m16_stage3_mi300x/f1_prefix_reuse \
        --slo-ttft $SLO1 $SLO2 --label f1_${CFG}_${CAP}_${L}_${ov}_${arr%% *} --guard-own-root $$
  done
done
```

job 數＝3 配置 × 3 容量 × 2 長度＝18。

〔估計〕每格時間＝暖機＋reset＋到達期間＋收尾 settle：
- burst：prompt token 總數 ÷ 吞吐。8K×64＝524K token ÷ 約 15–18K token/s ≈ 30–35 s（無命中；8K 的 18.5K token/s 是 C6 TRITON 8K 冷 0.443 s 算的）；32K×32＝1.05M token ≈ 70 s（ROCM_ATTN）／147 s（TRITON，C6 32K 冷 4.58 s）。重疊越高越快。
- Poisson：n_req ÷ λ：8K 在 0.9λ／0.5λ 約 40／71 s；32K 約 78／139 s。
- 每格固定開銷：暖機 2 個請求（≤ 8K，約 1 s）＋ 2 次 settle 3 s ＋ reset ≈ 10 s。
- → 8K 每個重疊約 2.5–3 分鐘、32K 約 5–6 分鐘；每個 job：8K 約 15 分＋啟動 2.4 分；32K 約 28 分＋2.4 分。
- **全格約 9×17.4＋9×30.4 ≈ 430 分鐘 ≈ 7.2 小時**（卡片估 6 h）。要壓到 6 h 以內：第一輪只跑重疊 {0, 50, 100}（約 4.5 h），有訊號再補 25／75。

---

## 4. F2 共用前綴 burst（重現 vLLM issue #44294）（`results/m16_stage3_mi300x/f2_shared_prefix_burst/`）

假說來源：#44294 回報 L40S 上 50 個併發共用 10K 前綴，開 OffloadingConnector 11.46 s、不開 5.95 s（協調者轉述，**我沒有讀到 issue 原文**）。機制：vLLM 0.28 仍有同樣的邏輯——命中的 chunk 正被別的請求載入時，整個請求延後（`offloading/scheduler.py:835-857`；0.19.1 是 `:126-140`）。

### 4.1 格子

| 軸 | 值 |
|:--|:--|
| 前綴長度 P | 10K（10240）、32K（32768）；尾巴 256、輸出 64 |
| 併發 N | 1、8、32、64 |
| 設定 | (A) APC、不卸載；(B) APC＋CPU 卸載（加 `--connector-debug`）；(C) 不復用（`--apc off`） |
| 三段 | seed（1 個請求寫入前綴）→ evict（把前綴擠出 GPU）→ burst（N 個同時送） |

```bash
$PY $REPO/code/m16_workload.py gen-burst --prefix-len 10240 --suffix-len 256 --n-concurrent 64 --out-len 64 --seed 0 --out W.jsonl
$B run --workload W.jsonl --server-config "$RD/serve_config.json" --evict reset --phase-gap-s 3 \
       --server-log "$RD/vllm_server.log" --out-dir $REPO/results/m16_stage3_mi300x/f2_shared_prefix_burst \
       --label f2_${CFG}_P10240_N64 --guard-own-root $$
```

- evict 的三種做法：`--evict reset`（`/reset_prefix_cache?reset_external=false`，只清 GPU、CPU 留著，**建議主格用**：確定、瞬間完成）；`--evict fill`（工作負載用 `gen-burst --n-fill K --fill-len 8192` 產生填充請求，要配合小的 `--gpu-blocks` 才擠得掉，否則預設容量（C6 在 0.5 利用率就有 638,400 token）要送幾十個 8K 請求）；`--evict none`（前綴還在 GPU）。另外 `--phases burst` 是冷啟動（沒有 seed，大家同時第一次算這個前綴）。**#44294 量的是哪一種情形我不知道**，所以建議 (B) 的 P＝10K、N＝64 先跑 reset、none、冷啟動三種，看哪一種重現 11.46 vs 5.95 的差，再決定主格。
- (A) 的 reset 等於「前綴不在任何地方 → 重算」，是 (B)「前綴在 CPU → 載入」的對照；(C) 是完全不復用的下限。
- 主指標：burst 段的 TTFT P50／P95／最大、`burst.makespan_s`（對應 issue 的總時間）；(B) 另看 `server.connector_log.delay_chunks_being_loaded`（行數、被延後的請求數）與 burst 段 `/metrics` 的 `offload_lookup_async_delay_*`、`offload_load_bytes`（應該只載入一次前綴）。
- **kernel 混雜**：(B) 開了 connector 會退回 TRITON_ATTN（§7 風險 1），(A)、(C) 是 ROCM_ATTN。要比「延後邏輯」本身，加一組 (A′)：`--apc on --attention-backend TRITON_ATTN`；或 (B) 加 `--rocm-attn-patch`。
- 重複：每格 3 次（每次 run 開頭會 reset，順序打散）。

### 4.2 展開與時間

job＝3 個 server（A、B、C；要做 A′ 就 4 個），每個 server 跑 2 P × 4 N × 3 次＝24 個 run。
〔估計〕(A)、(B) 每個 run：暖機約 1 s＋seed 冷 prefill（10K 約 0.7 s〔算術，15K token/s〕、32K 2.19–4.58 s）＋ 2 × 3 s 間隔＋ burst（只算尾巴＋(B) 載入一次前綴：32K 從 CPU 約 1.08 s，C6）＋ settle 3 s ≈ 15–20 s → 24 個約 8 分鐘。(C) 的 burst 要算 N×P token：每輪 (10K＋32K)×(1＋8＋32＋64)＝約 4.5M token ÷ 15K token/s ≈ 5 分鐘，3 次 15 分鐘＋開銷 ≈ 20 分鐘（N＝64、P＝32K 時 2.1M token 的 KV 放不下 GPU，vLLM 會排隊／搶占，這本身就是結果的一部分）。**加上 3 次啟動，合計約 45 分鐘；有 A′ 約 55 分鐘**。

---

## 5. I3 agent 多輪重播（`results/m16_stage3_mi300x/i3_agentic_replay/`）

### 5.1 工作負載

```bash
# 工具輸出長度的經驗分佈（從 Mooncake toolagent 推；〔判讀〕見下）
$PY $REPO/code/m16_workload.py mooncake-deltas --trace toolagent --out $RD/toolagent_deltas.csv
$PY $REPO/code/m16_workload.py gen-agentic --n-sess 8 --n-turns 6 --base-ctx 16384 \
    --tool-len empirical:$RD/toolagent_deltas.csv:tool_est --pause uniform:1:120 --out-len 128 --max-ctx 40000 --seed 0 --out W.jsonl
```

- 每輪 prompt＝上一輪完整 prompt＋上一輪輸出的替代 token（固定序列，不是 server 真的生成的 token）＋工具輸出。所以上一輪真正留在 cache 的只有上一輪的 prompt；替代輸出那 128 個 token 每輪要重算（warm-up 會先算掉）。
- `mooncake-deltas`：把 toolagent trace 裡「完整包含前一個請求的下一個請求」當成同一個 session 的下一輪，只留一對一的配對。2026-10-19 在 CPU 上跑：23,608 列中 20,339 列找得到父請求、5,962 個一對一配對；`tool_est＝input 差 − 父請求輸出`，11.9% ≤ 0（產生器會夾到 0）。這是**推論**（trace 沒有標 session，也沒說下一輪是否含上一輪的完整輸出），不是 trace 給的數字。載入時一樣逐列驗 512-token 單位，並用 hash 數驗 input 差：(nB−nA−1)×512 < Δ < (nB−nA+1)×512。
- 暫停 1–120 s（卡片）。vLLM 沒有 TTL，KV 會不會被擠掉取決於暫停期間其他 session 的流量，不是時間本身。

### 5.2 格子

| 配置 | serve | run |
|:--|:--|:--|
| G：只用 GPU（被擠掉就重算） | `--apc on --gpu-blocks <緊>` | — |
| C：CPU 卸載 | `--apc on --offload cpu:<夠放全部 session> --gpu-blocks <緊>` | — |
| W：C＋warm-up | 同 C | `--warmup-lead {0,0.5,1,2}x --restore-est linear:<a>:<b> --warmup-noise {0,0.5}` |

- 容量「緊」：要讓 session 在暫停時真的被擠出 GPU，否則 G 一定最好、問題不存在。建議 GPU 容量＝0.5 × 所有 session 最後一輪 context 的總和（工作負載 meta 可算；必須 ≥ max_model_len，否則 vLLM 不啟動）。
- warm-up 的定義（`m16_vllm_bench.py` 檔頭）：在預測的回來時間前 lead 秒，送本輪 prompt 去掉新工具輸出的那段、`max_tokens=1`；實際請求照原定時間送、不等 warm-up。判定 `hidden`／`late`／`evicted`／`unknown`／`failed`；白搬比例＝(late＋evicted)÷(總數−unknown−failed)。卡片的「多搬的位元組」要用 W 格與 C 格的 `offload_load_bytes` 差值算（每個 warm-up 自己搬多少拿不到）。
- `Nx` 的 R(L) 來自 V0（§2）。V0 之前沒有本設定的數字：NOT_MEASURED（參考：C6 32K 從 CPU 載入 1.080 s，TRITON；F1 的 rocmfix 33K 2.049 s）。
- 指標：`summary.ttft_by_turn`、`ttft_s`（不含 warm-up）、`throughput`、`warmup.*`、`server.metrics`。

### 5.3 展開與時間

每個 (S, base_ctx) 組合：2 個 server（G、C）；C 那個 server 依序跑 C＋8 個 W 格（每格開頭 reset）。
〔估計〕每格牆鐘 ≈ 每個 session 的 Σ(暫停＋每輪延遲)：uniform 1–120 平均 60.5 s × 5 個暫停＋6 輪 × 約 1–5 s ≈ 5.3–5.5 分鐘，加開銷約 6 分鐘。10 格約 1 小時＋2 次啟動。S＝8，base_ctx 16K 與 32K 兩組約 2.2 小時；每組 2 個 seed 約 4.4 小時（卡片估 6 h）。暫停可以等比例縮短來省時間，但要在 RUNLOG 寫明（縮短會改變「暫停期間的其他流量」，就是改變擠出壓力）。

---

## 6. 時間總表〔估計〕

| 項目 | job 數 | 每個 job | 合計 | 主要依據 |
|:--|--:|:--|--:|:--|
| V0 | 5（＋選做 1） | 7–9 分 | 約 40 分 | C6 啟動 2 分 21 秒與冷／載入 TTFT（07_report.md:258-262） |
| F1 全格 | 18 | 17–30 分 | 約 7.2 h（只跑 3 個重疊約 4.5 h） | q5d 32K 2.19 s、C6 8K 0.443 s／32K 4.58 s |
| F2 | 3（＋A′） | 8–20 分 | 約 45–55 分 | 同上 |
| I3 | 4（2 組 × G、C） | 約 1 h | 約 2.2 h（2 seed 4.4 h） | 卡片的暫停分佈；每輪延遲同上 |

全部以「單請求」量到的速度外推；批次化、併發、ROCm 上開 connector 之後的實際吞吐都是 NOT_MEASURED，V0 跑完要回頭修這張表。

---

## 7. 已知風險

1. **ROCm 上接 KV connector 就退回 TRITON_ATTN**：vLLM 0.28 的 `RocmAttentionBackend.supports_kv_connector()` 回 False（`rocm_attn.py:211-215`），開卸載後 33K prefill 的 attention 1.367 → 3.277 s、doc C＝1 首輪 2.24 → 4.40 s（`RUNLOG_MI300X.md:176`、F1 §4.2–4.3）。APC 與 APC＋CPU 的比較會混進 kernel 的差。兩條路：
   - `--rocm-attn-patch`（`m16_serve_patched.py`，照 `m9_f1_run.py:72-107`）：強制 ROCM_ATTN，每層 KV 拆成 K、V 兩個 view 註冊。F1 驗過位元組 511／512 和上游一樣，但 copy 次數變 2 倍，33K 從 CPU 載入 1.092 → 2.049 s（F1 §4.4）。修法在 `vllm serve`（多行程）下**還沒在 GPU 上跑過**：launcher 在模組最上層安裝 patch，fork 與 spawn 開的 EngineCore 都應該生效，V0-d 要確認 log 裡每個行程都有 `M16_ROCMFIX`。只處理單一 KV group（Llama、Qwen2.5 都是）；Qwen 沒在 F1 驗過。
   - 或全部設定都 `--attention-backend TRITON_ATTN`，讓 kernel 一致。
2. **vLLM 的 CPU→GPU 只有約 4.1 GiB/s**：C6 用 vLLM 自己的 metrics 量到 8K／16K／32K 4.02／4.09／4.11 GiB/s（`docs/phase1_20261008/07_report.md:258-268`，run `20261008-141324-m7-c6-vllm`），裸 pinned H2D 是 52.9 GiB/s（同檔 :95）。F1 進一步發現成本主要是「每次 copy 的固定成本（約 14.5 µs）× copy 次數」，ROCm 7.2.x 的 `hipMemcpyBatchAsync` 沒有真的批次化（F1 §4.4，上游 #43018）。⚠️ 任務說明寫的 `docs/precision_20261013/07_report.md` 不存在；原文在 `docs/phase1_20261008/07_report.md`（`stage2_factcheck_and_toolbox.md:143` 只寫了檔名）。
3. **亂數 token 的 prompt 會有近似平手**：輸出不一致不一定是 KV 錯（§2）。
4. **reset**：要 `VLLM_SERVER_DEV_MODE=1`（serve 一律設）；有 block 被占用或非同步搬移中會回 `success=false`，bench 每 0.5 s 重試、最多 40 次，還是失敗就結束碼 3。`reset_external=true` 清 CPU 層、不清 fs 層。
5. **/dev/shm 殘留**：CPU 層是 `/dev/shm/vllm_offload_<engine_id>.mmap`，server 收掉後跑 `serve_cleanup.sh`（只刪自己的 engine_id；`m7_c6_vllm.py` 結尾、RUNLOG_MI300X 發現 7、16 都踩過）。
6. **fs 層沒有容量上限**：`tier:<cpu>:<path>:<fsGiB>` 的 fsGiB 只用來在組指令時檢查剩餘空間（不夠就不組）；vLLM 會一直寫。NFS（`/mlsteam/data`）上的大量寫入只排在沒有量時間的 stage（matrix §6）。
7. **gpu_guard（AMD）**：server 必須是 guard own_root 的子孫；容器內「開了 kfd 但還沒配置 VRAM」的行程和容器外的行程會互相抵掉（`gpu_guard.py:317-319` 的已知盲點）。`--guard off` 只給 CPU 測試，該 run 的 `timing_valid=0`。
8. **`--connector-debug` 的 log 量**：每個被延後的請求在每個 scheduler step 都會印一行，N＝64 時很多；只在 F2 (B) 開。沒開時 `connector_log.status＝DEBUG_OFF`，計數不能用。
9. **容量下限**：`--gpu-blocks × block_size` 放不下一個 `max_model_len` 的請求，vLLM 啟動就失敗（`kv_cache_utils.py:731-760、2162-2196`）。F1 的「很緊」和 I3 的「緊」要先檢查。
10. **metrics 的 0 與 NOT_AVAILABLE**：offload 的 counter 是第一次觀測才出現樣本；有 `# TYPE` 行但沒樣本記成 0 並標 `REGISTERED_NO_SAMPLE`；整個家族不存在或 `/metrics` 抓不到才是 NOT_AVAILABLE（不填 0，規則 7）。
11. **client 精度**：CPU 測試（假 server、`nice 19`、4 核）TTFT 誤差中位數約 2 ms、最大 3.5 ms；TPOT 約 0.1–0.3 ms。真實 run 看 `summary.send_lag_s`（排程延遲）；client 不要和大量 CPU 工作擠同一組核心。
12. **Qwen**：用 noDCA 目錄（vLLM 0.28 ROCm 不支援 DCA，`RUNLOG_MI300X.md:868-869`）；Qwen 的 KV 每 token 28 層 × 4 頭 × 128 × 2 × 2 B＝56 KiB〔算術，config.json〕，Llama 是 128 KiB。

---

## 8. CPU 測試結果（2026-10-19，沒有用 GPU）

- `nice -n 19 taskset -c 0-3 /mlsteam/workspace/venv/tiara-v028/bin/python code/m16_workload.py --self-test` → **ALL PASS**（約 5 s）。含：重疊比例恰好、確定性、toolagent／conversation 整個檔案 0 列違反 512-token 斷言、兩條路徑的共享 block 分佈逐請求相同、16-token 單位的假 trace 被擋下、agentic 每輪是上一輪的嚴格前綴延伸、gen-burst 的三段結構。
- `M16_TEST_DIR=<暫存> nice -n 19 taskset -c 0-3 …/python code/tests/m16_bench_cpu.py` → **ALL PASS**（81 項，約 40 s）。含：serve 組出的指令與環境、dry-run、TTFT／TPOT 對假 server 的設定值、per-request cached_tokens 與 `/metrics` 差值、agentic 暫停（server 端量到 0.403–0.404 s，設定 0.4 s）、warm-up 提前量（0.248／0.299 s，設定 0.25／0.3 s）與四種判定、`/metrics` 缺欄位與 404 → NOT_AVAILABLE、F2 三段順序與延後訊息計數（只算該 run 的 log 區間）、v0check 的 PASS 與負對照（CPU 載入後輸出被改 → FAIL、記下分歧位置與 logprob 差）、污染替身 → 結束碼 4 且不寫 `--out-dir`、請求失敗照實記錄。
- 這些只證明**工具的邏輯**；vLLM 真的會不會照原始碼那樣回 metrics、cached_tokens、延後訊息，要 V0 才知道。

---

## 9. 沒做完或不確定

- `m16_serve_patched.py` 只在 CPU 上檢查過語法，沒有執行過（執行會 import torch）。
- #44294 原文沒讀到；三段式是照協調者的描述設計的，哪一種 evict 情境對應 issue 的 11.46 vs 5.95 s 未知（§4.1 的建議是先三種都跑一格）。
- F1 的容量比值定義有歧義（§3.1）。
- 所有時間都是〔估計〕；V0 之後要重估。
- 沒寫 runner（`code/m16_queue/`）；本頁 §1 只給 job 殼。
