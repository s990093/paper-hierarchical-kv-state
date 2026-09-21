# RUNLOG — 平台 B（AMD MI300X @ MLSteam）

> 格式照 `EXPERIMENT_PLAN.md` §7。原始 log 在 `/mlsteam/data/tiara/runs/<run_id>/`。
> 沒量到就寫 `NOT_MEASURED`。平台 A 的流水帳在 `RUNLOG.md`。

---

## 2026-09-15 — 選模與版本決策

**選模報告**：`results/model_selection/REPORT_model_selection_mi300x.md`（run_id `20260915-084837-model-survey`）。
報告裡的數字全部是算術估計，不是量測。

**使用者決定**：
1. 做 dense 與 MoE，小模型與 30B 級都做；hybrid 暫不做。
2. 🔴 **全部改用 vLLM v0.22.0**。原因是 0.19.1 沒有 SSD（fs）階，見下方發現 4。

**模型**（BF16 權重；下載有逐檔大小驗證，run `20260915-092340-model-download`、`...-model-download-par-seedoss`）：

| 設定鍵 | repo | 類別 |
|---|---|---|
| b-llama8b | unsloth/Llama-3.1-8B-Instruct（官方 repo gated，這是公開鏡像） | dense 小 |
| b-ultralong8b-1m | nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct | dense 小，原生 1M |
| b-qwen7b-1m | Qwen/Qwen2.5-7B-Instruct-1M → noDCA 變體（`20260915-093219-make-nodca-qwen7b1m`） | dense 小，與平台 A 同模型 |
| b-qwen3-30b-a3b | Qwen/Qwen3-30B-A3B-Instruct-2507 | MoE 30B |
| b-seedoss36b | ByteDance-Seed/Seed-OSS-36B-Instruct | dense 36B |

⚠️ 第一次下載 run（`20260915-092236-model-download`）**無效**：`--exclude` 接多個 pattern 被解析成檔名，實際什麼都沒下載，腳本卻印出 DONE。
已在該目錄留下 `INVALID` 檔。修正後的腳本會比對每個 safetensors 檔的大小與 HF API 回報值，全部相符才印 `VERIFIED`。

---

## 驗收 A2 — vLLM 在 ROCm 上能起來（v0.19.1）

**狀態**：PASS（第二次）

### 第一次 FAIL：`20260915-093230-m1-b-llama8b-bf16`

```
RuntimeError: Failed to infer device type
```

- **根因**：venv 裡沒有 `amdsmi` Python 模組，`vllm/platforms/__init__.py` 的 `rocm_platform_plugin()` 因而回報
  `No module named 'amdsmi'`，平台落到 `UnspecifiedPlatform`。
- **為什麼 `20_build_vllm.sh` 沒抓到**：它的驗收只做 `import OffloadingConnector`，這一步不需要平台偵測。
- **修正**：
  - 從 `/opt/rocm/share/amd_smi` 安裝綁定（`20260915-093559-infra-pip-amdsmi`）。
  - 寫進 `bin/10_python_stack.sh` 與 `bin/21_build_vllm_v022.sh`。
  - `bin/selftest.sh` 新增一項檢查：平台必須偵測為 `RocmPlatform`。
- **同一個 log 裡的非致命錯誤**：`amd-quark 0.12` 的 vLLM 插件 import `RoutedExperts` 失敗。這個錯誤會被攔下，不影響啟動。

---

## 發現 1 — `gpu_guard.py` 的 AMD 後端在真機上三處都是錯的（已改寫並自測）

| 問題 | 後果 | 修正 |
|---|---|---|
| amd-smi 的記憶體欄位是 `{"value","unit"}`；舊碼 `int(dict)` 丟例外後被 `continue` 靜默略過 | 有 GPU 行程時 `compute_apps()` 仍回傳 `[]`，**被讀成乾淨**（違反 CLAUDE.md 規則 7） | `_amd_val()` 解析，解析不了就丟 `SmiUnavailable` |
| **amd-smi 回報 host PID namespace 的 pid**（實測容器內 46730 ↔ amd-smi 3365442） | 無法用 pid 判斷是不是自己的行程 | 改用數量判斷：amd-smi 裡 VRAM>0 的行程數 − 容器內開著 `/dev/kfd` 的行程數 |
| `gpu_util()`、`free_mib()`、`idle_gpus()` 走 nvidia-smi 或解析錯誤 | 在 AMD 上直接崩潰 | 改走 `amd-smi metric -u/-m --json` |

- **常駐條目**：pid 8110，VRAM=0、gfx=0，07:08 的 hwinfo 就已經存在（早於本容器的任何 GPU 程式）。只記錄，不判為污染。
- **自測**：容器內非本 run 的 torch 行程 → 判為 `same_container` 入侵 ✅；本 run 自己的子行程 → `CLEAN` ✅。
- **容器外**（其他 pod）入侵：**無法注入測試**。已知盲點寫在 `foreign_on()` 的註解裡。
- `host_contention()` 在單一 VF 容器內看不到同一台 host 上其他 pod 對 PCIe／host RAM 的負載，回報 `UNOBSERVABLE_HOST`，**不回報 QUIET**。

---

## 發現 2 — vLLM 0.19.1–0.22.0 沒有 `int8/int4_per_token_head` KV dtype

`vllm/config/cache.py` 的 `CacheDType` 只有 `auto / float16 / bfloat16 / fp8 / fp8_e4m3 / fp8_e5m2 / fp8_inc / fp8_ds_mla`。
所以**平台 B 的 GPU 精度階只量 BF16 與 FP8**。平台 A（v0.28.0）的 INT8／INT4 兩階在這裡是 `NOT_MEASURED`。

---

## Milestone 1（v0.19.1，保留作跨版本對照；正式數字將以 v0.22.0 重量）

**產出**：`results/m1_capacity/capacity_mi300x_vllm0191.csv`

| run_id | 設定 | max_model_len | GPU KV cache size（token） | 結果 |
|---|---|---|---|---|
| 20260915-093626-m1-b-llama8b-bf16 | Llama-3.1-8B BF16 / KV auto | 8,192 | **1,271,232** | ready |
| 同上 | | 131,072（模型上限） | 1,271,696 | ready，`limited_by=model` |
| 20260915-094209-m1-b-llama8b-bf16-kvfp8 | Llama-3.1-8B BF16 / KV fp8 | 8,192 | **2,543,408** | ready |
| 同上 | | 131,072 | 2,543,408 | ready，`limited_by=model` |

- **FP8 / BF16 = 2.0008×**，與平台 A 的「恰好 2 倍」一致。
- **與選模估算比較**：估算 1,227,394（假設 runtime 餘裕 8 GiB），實測高 3.6%。
- **論文 §1 寫的「Llama-3.1-8B 於 MI300X 的 KV 預算 157.8 GiB」**：實測 1,271,232 × 128 KiB = **155.2 GiB**（由 token 數換算，不是 vLLM 直接印出的 GiB 值）。
- 驗證上界（`verify_over`）**沒有做**：容量超過模型可定址上限 131,072，這個模型量不到真正的記憶體懸崖。
- 佇列（`20260915-094209-m1-queue-b`）在 Llama-8B kvfp8 之後就停了，因為決定改用 v0.22.0，避免 0.19.1 的數字混進結果。

---

## 發現 4 — 🔴 vLLM 0.19.1 沒有 SSD（fs）階 → 改用 v0.22.0

`vllm/v1/kv_offload/factory.py` 在 0.19.1 只註冊了 `CPUOffloadingSpec`。逐 tag 檢查：

- **v0.22.0 起**才有 `TieringOffloadingSpec` 與 fs 次階。
- v0.22.0 的 fs 次階類型名稱是 **`fs_python`**，v0.28.0 才改名為 `fs`。
- v0.22.0 **沒有** `chunk_queries` 指標；平台 A 靠這個指標確認資料真的 cascade 到磁碟。
  → M2 改記錄伺服器行程樹的 `/proc/<pid>/io` 讀寫位元組數，以及磁碟階目錄的大小。
- v0.22.0 沒有 `csrc/libtorch_stable/cuda_view.cu`（v0.23.0 起才有，這是先前 v0.26 編不過的原因之一）。
- v0.22.0 需要 torch 2.11.0，ROCm 版只有 nightly `2.11.0.dev20260206+rocm7.0`。

**建置**：`bin/21_build_vllm_v022.sh`，另開 venv `venv/tiara-v022` 與 git worktree `src/vllm-v0.22.0`，不動 0.19.1。
log 在 `logs/build_vllm_v0.22.0-*.log`。

**結果：FAIL**（log `logs/pip-editable-v0.22.0-*.log`）。第一次失敗在 `git fetch` 的 ref 鎖（NFS）；跳過多餘的 fetch 後，第二次在 C++ 核心編譯失敗：
```
csrc/custom_quickreduce.hip:63:27: error: no member named 'getCurrentHIPStreamMasqueradingAsCUDA' in namespace 'at::cuda'
csrc/mamba/mamba_ssm/selective_scan_fwd.hip:421:21: error: use of undeclared identifier 'C10_HIP_CHECK'   (×8)
```
根因：torch nightly `2.11.0.dev20260206` 與 v0.22.0 預期的 torch 2.11.0 正式版 API 不同。

---

## 發現 5 — 🔴 先前「ROCm 沒有 torch 2.13」的結論是錯的 → 改建 v0.28.0（與平台 A 同版）

`bin/20_build_vllm.sh` 的版本調查只查了 `download.pytorch.org/whl/rocm7.0`。
實際上 **`whl/rocm7.2` 有正式版 torch 2.11.0／2.12.1／2.13.0／2.14.0**（本容器是 ROCm 7.2.2）。

**使用者決定：改建 v0.28.0。** torch 版本的選法：v0.28.0 上游有兩個說法彼此不一致——
- `requirements/build/rocm.txt` 寫 torch 2.11.0（rocm7.1）
- **官方 `docker/Dockerfile.rocm_base`** 用 ROCm 7.2.3 + PyTorch release/2.12 + vision v0.27.1

選最接近官方映像的正式 wheel：**torch 2.12.1+rocm7.2、torchvision 0.27.1+rocm7.2**。
不裝 torchaudio：它的 2.11.0 wheel 會把 torch 降回 2.11，而且文字 LLM 用不到。

指令：
```bash
VLLM_REF=v0.28.0 VENV=/mlsteam/workspace/venv/tiara-v028 TORCH_INDEX=https://download.pytorch.org/whl/rocm7.2 \
  TORCH_SPEC="torch==2.12.1+rocm7.2 torchvision==0.27.1+rocm7.2" bash bin/21_build_vllm_v022.sh
```
⚠️ 平台 A 是 v0.28.0 + torch 2.13.0+cu129。**vLLM 同版，但 torch 小版本不同**，κ 比較時要一併註記。
**狀態：進行中**。

v0.28.0 之後，發現 2（沒有 INT8／INT4）與發現 4（fs 次階叫 `fs_python`、沒有 `chunk_queries`）都不再適用：
v0.28.0 的 CacheDType 含 `int8/int4/fp8_per_token_head`，fs 次階叫 `fs`。M1／M2 的預設已改成 `venv/tiara-v028`。

---

## 資料集下載 — PASS

**run_id**：`20260915-100408-dataset-download`，位置 `/mlsteam/data/tiara/datasets/`

| 資料集 | 驗證（CLAUDE.md 規則 6：用資料本身交叉驗證） |
|---|---|
| gsm8k train/test | 7,473／1,319 題，與公開規格相同；每題都有 `####` 標準答案 |
| traces conversation／toolagent／synthetic（Mooncake commit `3cca71da`） | 12,031／23,608／3,993 請求，前兩者與平台 A 的記錄相同；`input_length ÷ len(hash_ids)` 中位數 496.3／487.9／499.6（每個 hash_id 是 512-token block） |
| longbench data.zip + config／metrics／eval／pred（GitHub commit `08ae8446`） | sha256 與 HF LFS oid 相同；7 個英文任務各 150–200 筆 |
| ruler_ref（NVIDIA/RULER `c3f5e3b4`） | clone 成功 |
| yakv_ref（context-intensive-kv-offloading `3475125e`） | clone 成功 |
| scbench（microsoft/SCBench） | 31 個檔案，大小逐一與 HF API 相同 |
| pylibs（`--target /mlsteam/data/tiara/pylibs`）：rouge、fuzzywuzzy、wonderwords、lightgbm 4.7.0 | 全部能 import |

---

## 驗收 A1–A3（v0.28.0）— PASS

**建置**：`bin/21_build_vllm_v022.sh`，參數見發現 5。log 在 `logs/build_vllm_v0.28.0-*.log`，約 25 分鐘。

| 驗收 | 結果 |
|---|---|
| A1 torch | `2.12.1+rocm7.2`，hip `7.0.51831`；GPU 上實際跑過 2048² matmul |
| A2 平台偵測 | `RocmPlatform` |
| A3 卸載連接器 | `OffloadingConnector` 可 import 且 `SupportsHMA=True`；spec 有 `CPUOffloadingSpec`、`TieringOffloadingSpec` |

---

## 發現 6 — gpu_guard 在 v0.28.0 上的兩個誤判（已修）

1. **server 關閉後的假入侵**（`20260915-103500-m1-b-llama8b-bf16`，`CONTAMINATED_DURING_RUN`，峰值 180,012 MiB）
   - 容器內的 EngineCore 已經結束、kfd 也關了，但 amd-smi（host 端）仍回報那 180 GB 約數秒。
   - 數量判斷因此把自家 server 算成容器外的行程。接下來 9 個設定開跑前都判定不乾淨，各在 1 秒內結束。
   - 當時佇列 log 的 `rc=0` **不可信**：`echo "$(date) rc=$?"` 裡先執行的 `$(date)` 把 `$?` 蓋掉了，已改成先存 rc。
   - **修正**：
     - `GpuWatcher.pause()`：關閉 server 期間暫停取樣，暫停區間寫進報告。
     - `wait_until_released()`：確認 amd-smi 已經沒有任何行程持有 VRAM（連續 3 次），才進行下一次啟動。
   - **取證**：v0.28 的 `VLLM::EngineCore` 有開 `/dev/kfd`，server 運作中會被正確認成自家行程。
2. **取樣與暫停的競態**（`20260915-105844-m1-b-llama8b-bf16-kvint8`）
   - 取樣在暫停開始前就進入，12 ms 後才讀到正在關閉的 server，只有這一筆。
   - **修正**：取樣期間只要碰到暫停（旗標或暫停次數改變），整筆丟棄，並記在 `discarded_samples_overlapping_pause`。
   - 這一格已補跑：`...m1-b-llama8b-bf16-kvint8-rerun`。

---

## Milestone 1 — 容量（v0.28.0）— PASS

**狀態**：PASS
**執行時間**：2026-09-15 10:52 → 12:26（佇列 `20260915-105204-m1-queue-b-v028`，加上 kvint8 補跑）
**指令**：`TIARA_PLATFORM=B TIARA_VENV=/mlsteam/workspace/venv/tiara-v028 python code/m1_capacity.py --gpu 0 --config <cfg>`
**產出檔**：`results/m1_capacity/capacity_mi300x.csv`（41 列，`guard_verdict` 全部 CLEAN）

**關鍵數字**（`GPU KV cache size`，`gpu_memory_utilization=0.90`，量測時 `max_model_len=8192`）：

| 模型 | BF16 | FP8 | INT8 | INT4 | 可定址上限 | 限制來自 | run_id（BF16） |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | 1,271,024 | 2,535,008（1.994×） | 2,465,024（1.939×） | 4,771,792（3.754×） | 131,072 | model | 20260915-105209-m1-b-llama8b-bf16 |
| UltraLong-8B-1M | 1,265,520 | 2,531,040（2.000×） | 2,454,336（1.939×） | 4,764,320（3.765×） | 1,073,152 | model | 20260915-110604-m1-b-ultralong8b-1m-bf16 |
| Qwen2.5-7B-1M（noDCA） | 2,889,696 | 5,817,152（2.013×） | 5,604,272（1.939×） | 10,878,880（3.765×） | 262,144 | model | 20260915-112242-m1-b-qwen7b-1m-bf16 |
| Qwen3-30B-A3B（MoE） | 948,176 | 1,896,368（2.000×） | 1,838,912（1.939×） | 3,569,408（3.764×） | 262,144 | model | 20260915-113505-m1-b-qwen3-30b-a3b-bf16 |
| **Seed-OSS-36B** | **413,632** | 827,264（2.000×） | 802,192（1.939×） | 1,557,200（3.765×） | 524,288 | **memory** | 20260915-115706-m1-b-seedoss36b-bf16 |

**Seed-OSS-36B 是唯一由記憶體決定上限的設定**，雙向都驗證過：
- `max_model_len=413,632` 可以啟動。
- `475,676`（1.15×）啟動失敗，錯誤訊息：
  ```
  ValueError: To serve at least one request with the model's max seq len (475676), (116.13 GiB KV cache is needed,
  which is larger than the available KV cache memory (100.99 GiB). ... the estimated maximum model length is 413632.
  ```
- log 另記載：權重 `67.46 GiB`，`Available KV cache memory: 100.99 GiB`，換算 413,632 × 256 KiB = 100.99 GiB。

**UltraLong-8B-1M**：以 `max_model_len=1,073,152` 啟動成功，所以**單卡 BF16 的 1M 上下文可以啟動**。
這裡只驗證了 server 能起來，**還沒實際送 1M 長度的請求**（M2 的重算掃描會送）。

**與論文假設的差異**：
- **精度階容量倍數**：2.000／1.939／3.765（Llama-3.1-8B 為 1.994／1.939／3.754），與平台 A 的 2.00／1.94／3.77 一致。
  INT8／INT4 的固定中繼資料開銷在兩個平台上相同。
- **論文 §1「Llama-3.1-8B 於 MI300X 的 KV 預算 157.8 GiB」**：實測 1,271,024 × 128 KiB = 155.2 GiB，差 1.7%。
- **跨版本**：同一個 Llama-8B BF16，v0.19.1 為 1,271,232、v0.28.0 為 1,271,024，差 0.02%。
- **與選模估算比較**（假設 runtime 餘裕 8 GiB）：Seed-OSS 估 399,152、實測 413,632（+3.6%）；Llama-8B 估 1,227,394、實測 1,271,024（+3.6%）。
  → 實際的 runtime 開銷小於假設的 8 GiB。

**失敗與異常**：見發現 6（兩個假污染，已修並重跑）。

---

## Milestone 2 — 成本常數（v0.28.0）— 進行中

**佇列**：`20260915-123708-m2-queue-b-v028`，產出在 `results/m2_harness_mi300x/`

### 試跑閘門失敗兩次，第三次通過（完整記錄）

1. `20260915-122606-m2-b-smoke`：SSD 階的 warm ≈ cold ≈ 1,305 ms，fs 目錄 0 bytes，**卸載完全沒發生**。
   - log 裡是 `cannot store chunks`。
   - **根因**：平台 B 的 vLLM 預設 `max_num_batched_tokens=16384`，一個 prefill step 要一次交給 CPU 主階 1,024 個 block；
     但沿用平台 A 做法縮到 1 GiB 的 CPU 主階只有 512 個 block，一步都放不下，`prepare_store` 整批拒收。
   - 平台 A 的 24 GB 卡預設批次較小，所以沒遇到。
   - **修正**：新增 `--ssd-cpu-bytes`，設為兩個 step 的量（32,768 token）。一步放得下，但遠小於工作集，仍會被逼著 cascade。
2. `20260915-123210-m2-b-smoke`：fs 8.00 GiB、warm 讀回 8.00 GiB，但仍有 `kv_offload_allocation_failure=4`，
   被「拒收次數必須為 0」的閘門擋下。
   - **判讀**：拒收發生在 cold 第 3、4 個前綴、CPU 主階 100% 滿的時候，屬於暫時拒收，vLLM 之後重試成功。
     `kv_offload_store_bytes=8,589,934,592`，正好等於工作集（4 × 16,384 × 128 KiB）。
   - **判準修正**：改成 `store_bytes`、fs 目錄大小、warm 讀回位元組三者都必須 ≥ 工作集；拒收次數照樣逐列記錄（`store_failures_so_far`、`m_kv_offload_allocation_failure`）。
3. `20260915-123708-m2-b-smoke`：**通過**。store_bytes 8.00 GiB = 工作集，fs 8.00 GiB，warm 讀回 8.00 GiB，暫時拒收 4 次。

⚠️ 兩次試跑中，SSD 階的 warm #0 都約 2,110–2,130 ms，**比 cold（整段重算，約 1,400 ms）還慢**；
warm #1–#3 約 1,120–1,150 ms。可能是第一個前綴在磁碟、其餘部分在 CPU 階。
這只是試跑，要等交錯 3 輪的正式量測與逐階統計，**不下結論**。

### 🔴 發現 7：CPU 階的共享記憶體檔沒被清掉，把 /dev/shm 塞滿 → 大模型的 CPU／SSD 階起不來（2026-09-15 16:50 檢查）

- **現象**：以下 server 啟動就死（rc=1），錯誤 `RuntimeError: Insufficient space in /dev/shm: <N> MiB required, <M> MiB free`
  - qwen7b-1m `cpu_r2`（需 32,768 MiB，剩 16,962）
  - qwen3-30b-a3b `cpu_r0/r1/r2`（需 55,296 MiB，剩 16,962）
  - seedoss36b `cpu_r0/r1/r2`（需 145,408 MiB，剩 7,746）與 `ssd_r0/r1/r2`（需 8,192 MiB，剩 7,746）
  - log：`/mlsteam/data/tiara/runs/20260915-131243-m2-retrieval/`、`20260915-140254-m2-retrieval/`、`20260915-151300-m2-retrieval/`
- **根因**：vLLM v0.28 的 CPU 卸載階在 `/dev/shm/vllm_offload_<uuid>.mmap` 開檔，server 被收掉後檔案留著。
  16:50 時 /dev/shm 179G 用了 172G，共 17 個 `vllm_offload_*.mmap`（12:30–14:52 建立），**沒有任何 process 持有**（查 /proc/*/fd 與 map_files）。
- **第二個問題（harness）**：tier 起不來時 `m2_cost_model.py` 印 NOT_MEASURED 但仍以 rc=0 結束，佇列沒有停下（違反規則 2 的設計）。
- **影響到的結果**：
  - qwen3-30b-a3b：cpu 階 NOT_MEASURED
  - seedoss36b：cpu、ssd 階 NOT_MEASURED
  - qwen7b-1m：cpu 階只有 2 輪（n=8，不是 12）
- **處置**：清除孤兒 mmap 被 auto-mode 權限擋下，**等使用者決定**；harness 修正（收 server 後刪自己的 mmap、tier 失敗回傳非 0）要等目前佇列跑完才改，避免改到執行中的程式。

### ⚠️ 發現 8：96K 上下文的 SSD 階沒有把整個前綴從磁碟讀回 → 這兩列不能當 SSD 成本

| 模型 | 每前綴 KV（算術） | 4 前綴工作集 | fs 目錄 | warm 每請求讀回（中位） | `--ssd-cpu-bytes` |
|---|---|---|---|---|---|
| qwen7b-1m | 5.13 GiB | 20.5 GiB | 13.68 GiB | 1.75 GiB | 1.75 GiB |
| qwen3-30b-a3b | 8.79 GiB | 35.2 GiB | 23.59 GiB | 3.00 GiB | 3.00 GiB |
| llama8b（16K，對照） | 2.00 GiB | 8.00 GiB | 8.00 GiB | 2.00 GiB | 4.00 GiB |

- 每前綴 KV = 層數 × 2 × KV 頭數 × 128 × 2 bytes × token 數（qwen7b 28 層 4 頭；qwen3 48 層 4 頭）。
- 96K 的兩個模型：讀回量**正好等於 CPU 主階大小**，只有前綴的一部分；fs 目錄也小於工作集。
  - 推測：從磁碟載回時要先放進 CPU 主階，放不下的部分就重算。**未驗證**。
  - 這兩個模型 SSD warm（23.4 s／42.4 s）比整段重算（10.7 s／17.9 s）還慢，但因為不是完整讀回，**不能當作 SSD 階成本**。
- 正式量測沒有套用試跑的「讀回 ≥ 工作集」閘門，是 harness 缺口。

### 目前可用的取回成本（warm TTFT 中位數，ms；3 輪交錯）

| 模型 / ctx | GPU 常駐 | GPU FP8 | GPU INT4 | CPU | SSD | DROP（重算） |
|---|---|---|---|---|---|---|
| Llama-3.1-8B / 16,384 | 81.7 | 84.5 | 107.8 | 581.9 | 1,145.3（#0 前綴 2,100） | 847.6 |
| Qwen2.5-7B-1M / 96,000 | 634.8 | 643.0 | 626.5 | 2,839.8（2 輪） | 無效（發現 8） | 10,714.0 |
| Qwen3-30B-A3B / 96,000 | 743.3 | 811.6 | 784.8 | NOT_MEASURED | 無效（發現 8） | 17,897.2 |
| Seed-OSS-36B / 96,000 | 878.9 | 924.9 | 885.8 | NOT_MEASURED | NOT_MEASURED | 63,959.7 |

- 來源：`results/m2_harness_mi300x/retrieval_cost_b-*.csv`（run_id／ts 欄）；guard 全部 CLEAN。
- Llama-8B 在 16K 時，**SSD（1,145 ms）比重算（848 ms）慢**，CPU（582 ms）比重算快。
- INT4 的 cold（寫入 + prefill）特別慢：Seed-OSS 443 s vs BF16 64 s；qwen7b 55 s vs 11 s。
- ⚠️ `gpu_guard_m2_b-llama8b.json` 被後面的 recompute 階段覆寫了（同一檔名）；retrieval 階段的整段 guard 判定檔已不存在；只剩每列的 `foreign_gpu_count`（90 列全為 0）可佐證。

### 發現 7、8 的處置（2026-09-15 16:50–17:00）

- **清 /dev/shm**（使用者授權）：刪除前確認 17 個 `vllm_offload_*.mmap` 都沒有 process 持有；刪後 /dev/shm 179G 可用。清單存在 session scratchpad。
- **發現 8 的根因已由原始碼確認**：`vllm/v1/kv_offload/tiering/base.py` 的 `SecondaryTierManager` 註解寫明
  「Load: secondary → CPU (primary) → GPU (promotion)」，所以 CPU 主階必須放得下整個前綴。
- **harness 修正**（`code/m2_cost_model.py`）：
  1. `Server` 開跑前記下既有 mmap，並檢查 /dev/shm 剩餘空間 ≥ CPU 階；`__exit__` 刪掉這個 server 新建、且沒人持有的 mmap（自測：未持有的刪、持有的保留）。
  2. retrieval 任一 tier 例外 → 回傳 1（佇列會停）。
  3. ssd 每列 warm 的 `req_read_bytes` 須 ≥ 95% 前綴 KV，否則整階另存 `retrieval_cost_INVALID_partial_ssd_read_*.csv` 並回傳 1。
     既有 llama8b 的 12 列 warm 都讀回 2.00 GiB，符合此判準。
  4. guard 檔名加上 stage（`gpu_guard_m2_<model>_<stage>.json`），不再被 recompute 覆寫。
  - 這些修改在第一個佇列執行中寫入，只會影響它剩下的 recompute 步驟的收尾（mmap 清理對 recompute 無作用），不影響量測本身。
- **補量佇列** `20260915-165825-m2-queue-b-v028-rerun`：等第一個佇列（PID 130779）結束後自動開始。
  - qwen7b-1m、qwen3-30b-a3b、seedoss36b 的 `gpu_resident cpu ssd drop`，交錯 3 輪，輸出 `*_v2.csv`
  - SSD 的 CPU 主階 = 2 個前綴的 KV（10.30／17.58／46.88 GiB），是工作集的一半，比例與 llama8b 16K 相同

### 🔴 發現 9：Seed-OSS-36B 重算階段在 P=393,216 逾時，已量的 36 列遺失；佇列 zombie 等待

- `20260915-170743-m2-b-seedoss-recompute` rc=1（18:04:19），`TimeoutError: timed out`，發生在 `stage_recompute` 灌 P=393,216 前綴的第一個請求（900 s 逾時）。
  - server.log 最後一筆 engine 統計在 17:49:16（Running 1 req、GPU KV 7.9%），之後到 18:04 abort 前沒有新統計。**是極慢還是卡住，無法從現有 log 判定**。
  - harness 只在全部位置跑完才寫 CSV → P=0…327,680 的 36 列只剩 stdout，**不寫入 results**（沒有 CSV 列可追溯）。
  - 修正：每個位置量完就寫；灌前綴時間另記 `fill_first_ttft_ms`／`fill_second_ttft_ms`；新增 `--request-timeout`（補量用 5400 s）。
- ultralong8b-1m 重算因佇列停下而**未執行**。
- 補量佇列原本用 `kill -0 130779` 等第一個佇列，但該 bash 結束後成為 zombie，`kill -0` 永遠成功 → 會永遠等。已停掉並移除等待，18:06:26 重新啟動，最後兩步加上 seedoss 重算 v2 與 ultralong 重算。

### 重算成本 vs 位置（2,048 token，ms；3 次）— 已寫入 CSV 的三個模型

| P（已快取前綴） | Llama-8B | Qwen2.5-7B-1M | Qwen3-30B-A3B |
|---|---|---|---|
| 0 | 95 | 79 | 84 |
| 16,384 | 259 | 234 | 284 |
| 49,152 | 572 | 521 | 677 |
| 114,688 | 1,185 | 1,058 | 1,429 |
| 163,840 | — | 2,185 | 3,294 |
| 212,992 | — | 1,924 | 2,604 |
| 258,048 | — | 3,327 | 3,137 |

- P ≤ 114,688：三個模型都隨 P 近似線性成長，3 次之間差 < 4%（P=0 的第 1 次是暖機）。
- ⚠️ P ≥ 163,840：**不單調**（Qwen3 163,840 的 3,294 > 212,992 的 2,604），而且同一位置 3 次會在兩個值之間跳（Qwen3 258,048：3,136／3,137／5,060）。seedoss（只有 stdout）也有同樣的兩值跳動。**原因未查明**，擬合 α、C0 前要先釐清，不下結論。

## Milestone 2 — 成本常數（v0.28.0）— 量測完成，分析中

**狀態**: 取回成本 PASS（gpu_fp8／gpu_int4 只有 llama8b 有）、重算成本 PASS（5 個模型）、一個異常待查（發現 11）
**執行時間**: 2026-09-15 12:37 → 2026-09-16 00:10（兩個佇列）
**run_id**: `20260915-123708-m2-queue-b-v028`、`20260915-165825-m2-queue-b-v028-rerun`
**產出檔**: `results/m2_harness_mi300x/{retrieval_cost_*,recompute_position_*,cost_constants_mi300x.csv,recompute_fit_mi300x.csv}`
**分析腳本**: `code/m2_analyze_b.py`（只做算術，不生資料）
**guard**: 所有 run CLEAN，逐列 `foreign_gpu_count=0`

### 1. 取回成本（warm TTFT 中位數減 gpu_resident 基準）

| 模型 | ctx | GPU FP8 | GPU INT4 | CPU | SSD | DROP（重算） |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 16,384 | +2.7 ms | +26.1 ms | +500.1 ms | +1,063.6 ms | +765.8 ms |
| Qwen2.5-7B-1M | 96,000 | — | — | +2,253.9 | +4,263.1 | +10,061.1 |
| Qwen3-30B-A3B | 96,000 | — | — | **−49.1（異常，見發現 11）** | +2,970.4 | +17,146.0 |
| Seed-OSS-36B | 96,000 | — | — | +5,411.2 | +11,554.4 | +63,050.9 |

換算成等效頻寬（前綴 KV 位元組 ÷ 減基準時間）：

| 模型 | 前綴 KV | CPU | SSD |
|---|---|---|---|
| Llama-8B（16K） | 2.00 GiB | 4.29 GB/s | 2.02 GB/s |
| Qwen-7B（96K） | 5.13 GiB | 2.44 GB/s | 1.29 GB/s |
| Qwen3-MoE（96K） | 8.79 GiB | 異常 | 3.18 GB/s |
| Seed-OSS（96K） | 23.44 GiB | 4.65 GB/s | 2.18 GB/s |

- SSD 階全部通過「每列 warm 完整讀回一個前綴」的判準（`fs_dir_bytes` = 4 個前綴、`req_read_bytes` = 1 個前綴、`store_failures=0`）。
- GPU FP8 的反量化成本極小（16K 前綴 +2.7 ms＝0.17 µs/token）；INT4 是 +26.1 ms（1.59 µs/token），約 FP8 的 10 倍。
  ⚠️ 只有 llama8b 量到這兩階（其餘模型的精度階本來就在同一個 run 裡，但 v2 補量只跑 4 階，fp8/int4 沿用第一批，第一批的 96K 三個模型有 fp8/int4 但沒有可比的 cpu/ssd —— 分析腳本只用同一個 run 的基準，所以標 —）。

### 2. 重算成本 vs 位置：C_recompute(P) = C0 + a·P（每次重算 2,048 token）

用每個位置 3 次的**最小值**擬合（見發現 10，中位數會落在兩個模式之間）：

| 模型 | C0 (ms) | a (µs/千 token) | R² | 掃到 |
|---|---|---|---|---|
| Llama-3.1-8B | 101.5 | 9.25 | 0.9998 | 114,688 |
| UltraLong-8B-1M | 44.8 | 9.75 | 0.9993 | **1,048,576** |
| Qwen2.5-7B-1M | 96.1 | 9.22 | 0.9469 | 258,048 |
| Qwen3-30B-A3B | 89.5 | 11.83 | 0.9999 | 258,048 |
| Seed-OSS-36B | 408.3 | 30.76 | 0.9993 | 393,216 |

- **論文 §3「C_recompute 不是常數」在平台 B 成立**：UltraLong-8B 從 P=0 的 86 ms 長到 P=1,048,576 的 10,350 ms（120 倍），而且線性擬合 R²=0.9993。
- 兩個 8B 模型的斜率幾乎一樣（9.25／9.75 µs/千 token），Seed-OSS-36B 是 3.3 倍。

### 3. κ = 重算 / 傳輸（同一個 ctx、同一批 token）

| 模型 | ctx | κ_cpu | κ_ssd |
|---|---|---|---|
| Llama-3.1-8B | 16,384 | 1.53× | 0.72× |
| Qwen2.5-7B-1M | 96,000 | 4.46× | 2.36× |
| Qwen3-30B-A3B | 96,000 | 異常 | 5.77× |
| Seed-OSS-36B | 96,000 | 11.65× | 5.46× |

- κ_ssd < 1（Llama-8B 16K）代表**在這個位置重算比從 SSD 取回便宜**；κ 隨 ctx 與模型大小上升，因為重算是位置的線性函數而傳輸不是。
- ⚠️ llama8b 的 ctx 是 16,384、其餘是 96,000，**κ 不能跨列直接比**（重算那一項與位置成正比）。要與平台 A 比對時必須用同一個 ctx。

### 4. P*（SSD 與 DROP 的交叉點，chunk = 2,048 token）

| 模型 | SSD ms/chunk | C0 | a | P* |
|---|---|---|---|---|
| Llama-3.1-8B | 132.9 | 101.5 | 9.25 | **3,396 token** |
| Qwen2.5-7B-1M | 90.9 | 96.1 | 9.22 | < 0（SSD 一直較便宜） |
| Qwen3-30B-A3B | 63.4 | 89.5 | 11.83 | < 0 |
| Seed-OSS-36B | 246.5 | 408.3 | 30.76 | < 0 |

- 只有 Llama-8B 有正的交叉點：位置 < 3,396 token 時重算較便宜，之後 SSD 較便宜。
- 其餘三個模型在量測的 ctx 下 SSD 永遠較便宜 —— 但這是用 96K 量到的 SSD 每 token 成本外推到 2,048 token chunk，**外推的合理性還沒驗證**（SSD 取回可能有固定開銷，短 chunk 的每 token 成本會更高）。

### 🔴 發現 10：重算量測在 P ≥ 163,840 時出現兩個模式

- 同一位置 3 次會在兩個值之間跳，比值穩定在約 1.6：
  Qwen3 258,048 → 3,136／3,137／5,060；Seed-OSS 393,216 → 20,455／20,483／12,547；Qwen-7B 212,992 → 2,750／1,910／1,924。
- Llama-8B（只掃到 114,688）與 UltraLong-8B（掃到 1M）**沒有**這個現象，三次差 < 1%。
- 原因未查明。目前的處置：擬合同時報 median 與 min 兩版，P* 用 min（乾淨路徑），並在這裡標明。

### 發現 9 的後續：Seed-OSS 在 P=393,216 不是卡住，是真的要那麼久

- 補量的 `fill_first_ttft_ms` 顯示：P=393,216 的前綴 prefill 花了 **933,959 ms（15.6 分鐘）**，
  P=258,048 是 417,423 ms、P=327,680 是 659,173 ms —— 超線性成長，與 attention 的二次項一致。
- 所以原本 900 s 的逾時**不足**，不是 server 掛住。改用 5400 s 後一次過。

### 🔴 發現 11：Qwen3-30B-A3B（MoE）的 CPU 階 warm 比 gpu_resident 基準還快 —— 已重現，原因未明

**現象**（`retrieval_cost_b-qwen3-30b-a3b_v2.csv`，ctx=96,000、前綴 KV 8.79 GiB）：
CPU 階 warm 中位數 696.0 ms，gpu_resident 基準 745.0 ms → **delta = −49 ms**，等於「把 8.79 GiB 搬回 GPU 不花時間」。
同一批設定下 qwen7b 是 +2,254 ms（5.13 GiB）、seedoss 是 +5,411 ms（23.44 GiB）。

**診斷 1**（`20260916-001321-m2-queue-b-cpu-diag`，GPU pool 縮到 7,000 block = 111,999 token，只放得下 1.17 個前綴，逐出無可避免）：

| 模型 | gpu_resident | cpu warm | delta | drop warm |
|---|---|---|---|---|
| Qwen3-30B-A3B | 747.6 | 709.0 | **−38.6** | 17,884.3 |
| Qwen2.5-7B-1M | 627.3 | 2,872.0 | +2,244.7 | 10,714.2 |

→ **重現**。不是「block 還留在 GPU」造成的（pool 只有 1.17 個前綴，且 drop 階在同一設定下要 17.9 s，證明沒有快取命中）。

**還沒排除的解釋**：96K 的 gpu_resident 基準本身就要 745 ms，可能把非同步搬運整個蓋住（8.79 GiB 若以 50 GB/s 走 host link 只要 ~176 ms）。
若成立，delta ≈ 0 的意思是「搬運被計算完全重疊」，而不是「搬運免費」——這對 Oracle 的成本模型意義完全不同。
**診斷 2 進行中**（`20260916-011918-m2-queue-b-cpu-diag16k`）：同樣兩個模型改用 ctx=16,384（基準約 100 ms 級），看 delta 會不會變正。

**處置**：在原因釐清前，**Qwen3-30B-A3B 的 κ_cpu 標為異常、不採用**（`cost_constants_mi300x.csv` 的 `delta_vs_gpu_resident_ms` 為負，`effective_gb_per_s` 留空）。

### 發現 11（續）：診斷 2 排除「被基準蓋住」；vLLM 自己的指標證實搬運頻寬差 16 倍

**診斷 2**（`20260916-011918-m2-queue-b-cpu-diag16k`，ctx=16,384、GPU pool 1,200 block = 19,200 token）：

| 模型 | 前綴 KV | gpu_resident | cpu warm | delta | drop warm |
|---|---|---|---|---|---|
| Qwen3-30B-A3B | 1.50 GiB | 137.4 | 135.1 | **−2.2** | 898.4 |
| Qwen2.5-7B-1M | 0.88 GiB | 106.1 | 506.8 | +400.7（2.34 GB/s） | 731.8 |

→ 基準只有 137 ms，不可能蓋住 1.5 GiB 的搬運，**「被基準蓋住」的解釋被排除**。

**vLLM 連接器層的直接證據**（server.log 的 `KV Transfer metrics`，與 TTFT 無關的獨立量測）：

| 模型 | load_bytes | load_time | 載入頻寬 | store 頻寬 |
|---|---|---|---|---|
| Qwen3-30B-A3B（16K） | 4.83 GB | 0.128 s | **37.8 GB/s** | 39.2 GB/s |
| Qwen2.5-7B-1M（16K） | 2.82 GB | 1.213 s | **2.3 GB/s** | 2.6 GB/s |
| Seed-OSS-36B（96K） | 50.33 GB | 11.66 s | **4.3 GB/s** | — |

- 兩邊的 TTFT 推算值（qwen7b 2.34 GB/s、seedoss 4.65 GB/s）與連接器指標一致，**兩種獨立量法互相吻合**。
- 所以 Qwen3-MoE 的 CPU 階 delta ≈ 0 是**真的**：它的 KV 搬運跑在 37.8 GB/s（MI300X 的 host link 是 PCIe Gen5 x16，理論約 55 GB/s，這個數字物理上合理），1.5 GiB 只要 ~40 ms，小於量測解析度。
- **已排除的解釋**：GPU 殘留快取（診斷 1）、基準蓋住（診斷 2）、注意力後端不同（兩者都是 TRITON_ATTN）、每次複製的區塊大小（每層每 block 都是 32 KiB）、offload spec 設定（都是 CPUOffloadingSpec、同樣 25.77 GB 的 mmap）。
- **機制仍未確定**：為什麼同一台機器、同一版 vLLM、同一條路徑，不同模型的 KV 搬運頻寬差 16 倍。四個模型裡只有 Qwen3 是 MoE，但「MoE」本身不解釋搬運頻寬。

**對論文的意義**：這反而**強化**「κ 跨硬體／跨模型變動」的主張——同一張卡上，CPU 階的有效頻寬就有 2.3～37.8 GB/s（16 倍）的差距。
但在機制查明前，Qwen3 的 κ_cpu **不列入跨平台比較**；`cost_constants_mi300x.csv` 的該列 delta 為負、`effective_gb_per_s` 留空。

**harness 更新**：`Server.offload_metrics()` 之後會一併記錄 `kv_offload_load_bytes` 與 `kv_offload_load_time`，
往後每一列都能直接拿到連接器層的搬運頻寬，不必再從 log 撈。

---

## Milestone 3 — Tier 0 baselines（v0.28.0）— 進行中

**移植**（`code/m3_baseline.py`，平台 A 的設定原封不動保留）：
- `MODELS_B`：b-llama8b／b-qwen7b-1m／b-qwen3-30b-a3b／b-seedoss36b，容量取自 M1 實測。
- 🔴 **必須縮小 GPU KV 池**：B 的容量是 A 的 10–30 倍，而模型可定址長度反而是硬上限
  （Llama-8B 131,072）。4 個前綴 × 模型上限仍遠小於容量 → **一次都不會逐出**，
  五個 baseline 會量出一模一樣、看起來完全正常的數字（M2 踩過同一個坑）。
  作法：`--num-gpu-blocks-override`，並新增 `effective_capacity()`，`--list` 與守門一律用**有效容量**，不用 M1 全量容量。
- 新增守門：工作集若全部 ≤ 有效容量就中止（rc=7）。
- CPU 階大小改為逐模型（工作集差 4 倍）：`cpu_bytes`（≥ 工作集，當對手的快取）與 `fs_cpu_bytes`（< 工作集，強迫 cascade 到磁碟）。
  ⚠️ 平台 A 的 tier_fs 用 24 GiB ≥ 工作集，磁碟階可能一次都沒被用到 —— **兩平台的 tier_fs 不可直接比較**，要比必須先重量 A。CSV 逐列記 `fs_dir_bytes` 以便驗證。
- 沿用 M2 的三個修正：`/dev/shm` mmap 自動清理（發現 7）、收 server 時暫停 GpuWatcher + `wait_until_released`（發現 6）、ROCm 環境變數。

### 🔴 發現 12：override 太小會讓 server 起不來（不是記憶體不足）

- `20260916-031010-m3-b-llama8b-full_gpu` rc=1：
  `ValueError: To serve at least one request with the model's max seq len (34048), 4.16 GiB KV cache is needed, which is larger than the available KV cache memory (4.0 GiB)`
- 原因：override 2,048 block = 32,768 token = 4.0 GiB，但 vLLM 啟動時無條件檢查
  「KV 池要放得下一個 `max_model_len` 的請求」（`kv_cache_utils._check_enough_kv_cache_memory`），而 `max_len = max(ctx) + 256 + 1024 = 34,048`。
- 修正：override 下限 = `ceil(max_len/16) + 64`，程式自動提高並印出（實際用 2,192 block = 35,072 token）。提高後工作集 131,072 仍遠大於它，逐出照樣發生。

### 裝置頻寬（實測，O_DIRECT 循序，8 GiB × 3 次；`results/m2_harness_mi300x/disk_bw_mi300x.csv`）

| 位置 | 寫 | 讀 | 說明 |
|---|---|---|---|
| `/var/tmp`（本地 overlay） | 1,886 MiB/s | 5,392 MiB/s | SSD 階實際用的位置 |
| `/mlsteam/data/tiara`（NFS） | 683 MiB/s | 289 MiB/s | 讀比本地慢 18.6 倍 |

→ M4 的 `DEVICE_WRITE_MIBPS` 改成從這份 CSV 讀，不寫死。

### ⚠️ LMCache（Tier 0 第 5 個 baseline）在平台 B 尚未量到

- `lmcache` 0.5.5 的依賴含 `cupy-cuda13x`、`cuda-python`、`cufile-python`（CUDA 專屬），ROCm 上不會直接可用。
- 目前 `m3_baseline.py` 遇到缺 venv 會回 rc=6 並印出原因，**不會靜默跳過**。
- 計畫：M3／M4 跑完後，在**獨立 venv** 嘗試 `--no-deps` 安裝並測 `LMCacheConnectorV1` 能否在 ROCm 匯入；失敗就記 NOT_MEASURED 與完整錯誤，不用別的東西頂替。

---

## Milestone 4 — Oracle go/no-go（平台 B）

**狀態**: **GO**（四個模型在記憶體有壓力時都 > 15%）
**執行時間**: 2026-09-16 05:2x → 05:41
**指令**: `python code/m4_oracle.py --model <profile> --pressure 1 2 4 8 --cpu-gib 96`
**產出檔**: `results/m4_oracle_mi300x/`（`cost_model.json`、`simulator_validation.json`、掃描 CSV）

### 1. 模擬器先修了兩處，否則 headroom 不可信

**(a) 重算成本的量測規模不對**（`--recompute-source m3`，預設）
- M2 的 `recompute_position` 量的是「前綴已快取、再補 2,048 個 token」；模擬器要的是「block 被丟掉、跟著整段 prefill 一起重算」。
- vLLM 的 `max_num_batched_tokens=16384`，整段 prefill 一次送 16,384 個 token，2,048 的小塊用不滿 GPU → M2 的斜率比整段 prefill **大 2.2 倍**。
- 改用 M3 的 `full_gpu` cold TTFT 擬合 `T(N)=c+aN+bN²/2`（四個模型 **R²=1.00000**），對 N 微分得每 block 成本。

| 模型 | M2 分塊 base / slope | M3 整段 base / slope | 每請求固定開銷 c |
|---|---|---|---|
| b-llama8b | 0.7459 / 7.42e-5 | 0.5470 / 3.40e-5 | 14.9 ms |
| b-qwen7b-1m | 0.6133 / 9.84e-5 | 0.4717 / 2.69e-5 | 25.4 ms |
| b-qwen3-30b-a3b | 0.6558 / 9.25e-5 | 0.4213 / 5.33e-5 | 20.8 ms |
| b-seedoss36b | 2.9484 / 3.99e-4 | 2.2630 / 1.71e-4 | −5.8 ms |

**(b) Qwen3-MoE 的 CPU 階成本是 0**（發現 11）→ 改用連接器指標
- `vllm:kv_offload_load_bytes / load_time` 是連接器層直接量測，與 TTFT 無關。
- 交叉驗證（`results/m2_harness_mi300x/connector_transfer_mi300x.csv`，25 列）：

| 模型 | 相減法 ms/block | 連接器指標 ms/block | 差 |
|---|---|---|---|
| b-llama8b | 0.4884 | 0.4692 | 4% |
| b-qwen7b-1m | 0.3757 | 0.4047 | 7% |
| b-seedoss36b | 0.9019 | 0.9627 | 7% |
| b-qwen3-30b-a3b | **無效（−49 ms）** | **0.0411** | — |

### 2. 模擬器驗證（比對 M3 實測的 full_gpu / cpu_lru 倍數）

| 模型 | 有鑑別力的點 | 方向一致 | 量級相符（≤50%） | 偏離 |
|---|---|---|---|---|
| b-llama8b | 2 | 2/2 | **2/2** | 14–16% |
| b-qwen7b-1m | 2 | 2/2 | **2/2** | 28–29% |
| b-seedoss36b | 2 | 2/2 | **2/2** | 15–16% |
| b-qwen3-30b-a3b | 2 | 2/2 | 0/2 | 178–192% |

→ 前三個模型的 Oracle **絕對** headroom 可引用；Qwen3-MoE 只能引用趨勢（腳本自身的判準）。

### 3. go/no-go（Zipf α=0.9，壓力 = 工作集 / GPU 預算；CPU 階 96 GiB）

| 模型 | 1× | 2× | 4× | 8× | 判定 |
|---|---|---|---|---|---|
| b-llama8b | 0.0% | 35.2% | 37.6% | 33.5% | **GO** |
| b-qwen7b-1m | 0.0% | 33.2% | 39.4% | 34.0% | **GO** |
| b-qwen3-30b-a3b | 0.0% | 5.9%（MARGINAL） | 39.8% | 43.0% | **GO**（趨勢） |
| b-seedoss36b | 0.0% | 21.9% | 39.7% | 39.8% | **GO** |

- 1× 一律 0.0%：工作集塞得進 GPU，最佳 baseline 就是 full_gpu，**沒有放置決策可做**。這是正確的退化行為，不是失敗。
- 壓力 ≥ 2× 時最佳 baseline 幾乎都是 `cpu_arc`（Seed-OSS 在 4×／8× 是 `tier_fs`）——Oracle 是贏過**最強**的對手，不是最笨的。
- 判準（EXPERIMENT_PLAN §0 規則 4）：> 15% GO、5–15% 問人、< 5% 停。**四個模型的有壓力點都 > 15%**（除 Qwen3 的 2× 為 5.9%）。

**→ M4 判定：GO。**

---

## Milestone 5 — 品質（v0.28.0）— 進行中

### 🔴 發現 13：平台 B 的「無損」驗證不是位元相同（平台 A 是 60/60 完全相同）

- `results/m5_quality_mi300x/lossless_b-llama8b.csv`（GSM8K 60 題 × 3 設定）：
  - `cpu_lru`／`tier_fs` 相對 `full_gpu`：**輸出 sha1 不同 36/120**，其中**最終答案不同 8/120**。
  - 例：idx=2 兩邊答案都是 70000（正確），但生成長度 267 vs 265 token。
- 平台 A 的同一測試是 60/60 逐字元相同，所以 CLAUDE.md 寫「CPU 與 SSD 兩階是無損的位元組搬移」。
- **還不能說平台 B 的卸載是有損的**：缺一個對照——同一個 `full_gpu` 設定跑兩次是否本來就位元相同？
  ROCm 的注意力 kernel 在不同的 prefill 分塊邊界下可能給出不同的浮點結果，`temperature=0` 仍會因此改變取樣。
  - 已排入對照實驗：同設定重跑一次，比對 `out_sha1`。
  - **在對照做完前，這一格記為「待判定」，不寫入任何「無損／有損」的結論。**

### M4 補充 1：真實 trace（Mooncake FAST25，各取前 4,000 個請求）

| 模型 | conversation | toolagent | 最佳 baseline |
|---|---|---|---|
| b-llama8b | **20.9% GO** | **20.7% GO** | tier_fs |
| b-qwen7b-1m | **15.3% GO** | 11.5% MARGINAL | tier_fs |
| b-qwen3-30b-a3b | 9.8% MARGINAL | 9.5% MARGINAL | tier_fs |
| b-seedoss36b | 6.9% MARGINAL | 7.2% MARGINAL | tier_fs |

實際壓力（工作集 / GPU 預算）：conversation 12.3–86.1×、toolagent 6.4–44.5×。

🔴 **headroom 不是壓力越大越高**：合成 Zipf 在 2–8× 時 21–43%，真實 trace 在 6–86× 反而只有 7–21%。
   極高壓力下幾乎沒有重用，最佳 baseline（tier_fs）已經把能拿的都拿到了，Oracle 沒有多少空間。
   模型越大（KV 越大 → 相對壓力越高）headroom 越低，順序與此一致。
   **依 EXPERIMENT_PLAN §0 規則 4，5–15% 屬於「停下來問人」，已回報使用者。**

### M4 補充 2：發現 11 的敏感度（Qwen3-MoE 的 CPU 階成本）

| CPU 成本來源 | ms/block | 壓力 8× 的 headroom |
|---|---|---|
| 連接器指標（`kv_offload_load_*`） | 0.0411 | 43.0% GO |
| M3 端到端推得 | 0.1178 | **41.4% GO** |

→ 兩個獨立估計相差 2.9 倍，但 go/no-go 結論**一致**（都是 GO）。該剖面的結論不因這個未解的異常而改變。

### M5-1 大海撈針（不同 KV 精度，每設定 5 深度 × 4 次 = 20 題）

| 模型 | ctx | BF16 | FP8（靜態） | FP8-ptk（動態） | INT8-ptk | INT4-ptk |
|---|---|---|---|---|---|---|
| b-llama8b | 64,517 | 100% | 100% | 100% | 100% | 100% |
| b-qwen3-30b-a3b（MoE） | 129,067 | 100% | 100% | 100% | 100% | 100% |
| b-seedoss36b | 64,520 | 100% | 100% | 100% | 100% | 100% |
| **b-qwen7b-1m** | 129,067 | 100% | **10%** | **0%** | 90% | **0%** |

Qwen-7B 逐深度（0.05／0.25／0.50／0.75／0.95）：
- BF16 100/100/100/100/100；INT8 100/75/100/100/75
- FP8 0/0/25/25/0；FP8-ptk 全 0；INT4 全 0 → **不是某個深度的問題，是整體失效**

### 🔴 發現 14：A 平台「KV 量化對檢索是毀滅性的」這個結論，**不能跨模型推廣**

平台 A（`results/RUNLOG.md`，qwen-awq、ctx 4K–32K）量到 BF16 100%／FP8 5%／INT8 95%／INT4 0%，
並據此寫下「動作空間裡的 GPU-FP8 與 GPU-INT4 兩階，在檢索型工作負載下實際上不可用」。

平台 B 的四個模型顯示：
1. **同一個模型家族（Qwen2.5-7B）在 B 上重現了 A 的模式**（FP8 10%、INT4 0%、INT8 90%），
   而且是在 129K —— 比 A 的 32K 長 4 倍。跨平台、跨長度、跨權重格式（A 是 AWQ，B 是 BF16）一致。
2. **但 Llama-3.1-8B、Qwen3-30B-A3B、Seed-OSS-36B 在 64K–129K 下，五種精度全部 100%。**
   → 「INT4 會毀掉檢索」是 **Qwen2.5 這一族的性質，不是 KV 量化的普遍性質**。

**另一個推翻**：A 把原因歸給「縮放方式」（靜態 fp8 vs 動態 int8）。
B 加測了 `fp8_per_token_head`（動態縮放的 FP8）：**仍是 0%**，而同樣動態縮放的 INT8 有 90%。
→ 差別不在縮放方式，在**數值格式本身**（FP8 E4M3 的尾數只有 3 bit）。

**對論文的意義**：ε 必須按「模型 × 任務」報告，不能只按精度。
Oracle 若要使用 GPU-FP8／INT4 階，必須對該模型先做這個檢查；對 Qwen2.5 族要禁用，對其餘三族可用。

### M5-2 GSM8K many-shot（64-shot，120 題）

| 模型 | BF16 | FP8 | FP8-ptk | INT8 | INT4 |
|---|---|---|---|---|---|
| b-llama8b | 80.8% | 75.8% | 82.5% | 79.2% | 75.8% |
| b-qwen3-30b-a3b | 95.8% | 97.5% | 96.7% | 95.0% | 95.8% |
| b-seedoss36b | 92.5% | 94.2% | 93.3% | 94.2% | 90.0% |

- 差異都在 ±5pp 內，n=120 時 95% CI 約 ±7pp → **與 0 無法區分**，重現 A 的結論：
  **只看 GSM8K 會得出「KV 量化幾乎免費」的錯誤結論**（同一批設定在撈針上可以掉 100pp）。

### 🔴 發現 15：Qwen2.5-7B-1M 在 129K + 低精度 KV 下**不是品質下降，是語言能力崩潰** → 換掉主力模型

逐題看輸出（`needle_b-qwen7b-1m.csv` 的 `pred` 欄）：

| 精度 | 正解 | 模型輸出 | 輸出長度 |
|---|---|---|---|
| BF16 | 4266 | `4266` | 5 token |
| INT8-ptk | 4266 | `4266` | 5 token |
| FP8 | 4266 | `666666666666…`（重複同一字元） | 446 token |
| FP8-ptk | 4266 | 空字串 | 0 token |
| INT4-ptk | 4266 | `0` + 後續垃圾 | 699 token |

BF16／INT8 回 5 個 token 就停；FP8／INT4 是**退化生成**（重複字元、空輸出、700 token 垃圾）。
→ 量到的「0%」是崩潰，不是「找不到針」。**不能當品質指標引用。**

與文獻一致的部分：低位元 KV 量化的災難性退化在 KIVI／KVQuant 一系的工作中有記載
（K cache 的 outlier channel 被 4-bit 毀掉），且模型間敏感度差異大；FP8 E4M3 只有 3 bit 尾數，
精度低於 INT8，與「同樣動態縮放、FP8 死而 INT8 活」相符。
**不一致的部分**：單純精度不足應是逐漸退化（答錯但句子正常），不該是吐垃圾。

**處置（2026-09-17）**：
1. **長上下文主力改用 `nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct`**（原生 1M，M1 已量 1,265,520 tok）。
   已加入 M3／M4 的模型表，並排入 M2 取回、M3 四個 baseline、無損、撈針（**同樣 129K，直接對照**）、GSM8K。
2. Qwen2.5-7B-1M 的資料**全部保留**，定位改為「與平台 A 交叉重現的異常案例」：
   同一模型家族、跨平台（3090/AWQ ↔ MI300X/BF16）、跨長度（32K ↔ 129K）出現同一模式。
3. 診斷對照已排隊：Qwen2.5-7B-1M 在 4K／16K／32K 的撈針（平台 A 在 4K 就崩），
   以及 Llama-3.1-8B 拉到 129K（確認不是長度本身的問題）。

### 🔴 發現 16：/dev/shm 殘留檔再次塞滿（這次來自 M5）

- 2026-09-17 01:23：176/179 GiB 被 12 個 `vllm_offload_*.mmap` 佔住 → `m2-b-qwen14b-retrieval` 的 CPU 階
  三輪全部起不來（`/dev/shm 只剩 50.8 GiB < CPU 階 106.0 GiB`），佇列依規則 2 停止。
- 根因：發現 7 的修正只加在 `m2_cost_model.py` 與 `m3_baseline.py`，**`m5_quality.py` 漏了**。
- 已修：`m5_quality.Server.__exit__` 也會清掉自己建立且無人持有的 mmap（同一套邏輯）。
- 這次的檢查機制有效：M2 的開跑前檢查把它變成明確錯誤而不是難解的 server 崩潰。

---

## M5-C 注意力重要度 ↔ 重用預測的對齊分析（新增，補上論文表 15 (C) 的 attn_mass 那一格）

**動機**：`m5_predictor.py` 自己標註 `attn_mass` 與 pooled key/value 兩族特徵為 **NOT_AVAILABLE**
（「要跑模型才有」），並註記表 15 的特徵集消融做不到。本實驗把 `attn_mass` 量出來。

**做法**（`code/m5_attention_importance.py`）：
- 用 transformers 的 `AttentionInterface` 註冊自訂 attention：照常用 SDPA 算輸出（不改變模型行為），
  另外只取**最後 32 個 query**（SnapKV 式觀察窗）對所有 key 算分數，當場聚合成 per-block 質量後丟棄。
  成本 O(w·n) 而非 O(n²)——文獻指出具體化 n×n 與 FlashAttention 這類融合 kernel 不相容。
- 工作負載：24 份 LongBench **真實長文件** × 4,096 token，240 個請求依 Zipf(α=0.9) 重複挑文件，
  使同一批 block 同時具有真實注意力與真實重用歷史。真值 = 其後 32 個請求內是否再被存取。

### 先驗證量測本身可信（否則 AUC 沒有意義）

| block 位置 | 注意力質量（中位數） |
|---|---|
| 0（開頭） | **23.59** |
| 1–5 | 3.77–3.81 |
| 127–128（中間） | 3.83 |
| 253 | 5.22 |
| 255（結尾） | **9.05** |

→ 開頭有 6.2× 的尖峰（**attention sink**，StreamingLLM 記載的現象）、結尾單調上升（**recency**），
中間平坦。**與文獻的注意力形狀一致，量測可信。**

### 結果（b-llama8b，55,296 個可評估樣本，正例率 0.769）

| 訊號 | 預測「該 block 會被重用」的 AUC |
|---|---|
| 注意力重要度 `attn_mass` | **0.488**（= 隨機猜） |
| 存取歷史（次數 − 0.01×距上次） | **0.721** |
| 兩者的 Spearman 相關 | **−0.016**（無相關） |

### 🔴 發現 17：注意力重要度**無法**用來預測跨請求重用——兩個訊號正交

補充證據（同一批資料）：
- 同一個 block 位置**跨請求**的變異係數 0.0085；不同位置之間是 0.3306。
  → **位置解釋的變異是請求間變異的 38.8 倍**。
- 同一份文件的兩次不同請求，注意力前 50 名 block 的重疊率 **100%**。

**解釋**：注意力重要度幾乎完全由**位置**決定（sink + recency），對同一份文件的每次請求都一樣；
而跨請求重用取決於**哪份文件又被查詢**，與位置無關。兩者測的是不同的東西：
- H2O／SnapKV／Quest 那一系的重要度 → 單次請求**內**的取捨（要不要驅逐這個 token）
- Tiara 的預測器 → 跨請求的**放置**（這個 block 之後還會不會被別人用到）

**對論文的意義**：
1. 表 15 (C) 的 `attn_mass` 那一格可以填了：**在重用預測這個目標上，它的邊際貢獻為 0**（AUC 0.488）。
   §5.2 若把它列為特徵族，應改為「已量測、對此目標無貢獻」，而不是保留為未驗證的假設。
2. 這反而**支持** Tiara 用存取歷史特徵的設計選擇，也說明為什麼不能把現成的驅逐式重要度訊號
   直接拿來做放置決策。
3. ⚠️ 這**不代表**注意力重要度沒用——它對「單次請求內要保留哪些 token」仍是有效訊號
   （文獻已充分驗證）。本實驗只否證「可用於跨請求重用預測」。

**還在跑**：b-ultralong8b-1m、b-qwen14b-1m 兩個模型的同一實驗，確認這個否定結果跨模型成立。

### UltraLong-8B-1M（新的長上下文主力）的品質結果

| 測項 | 結果 |
|---|---|
| 大海撈針 @ **129,024**（與 Qwen2.5-7B 崩潰的同一長度） | 五種精度**全部 100%** |
| GSM8K 64-shot（120 題） | bf16 68.3／fp8 66.7／fp8_ptk 74.2／int8 67.5／int4 67.5 |
| 無損驗證 | 位元不同 36/120，答案不同 6 |

→ **同樣 129K、同樣五種精度，UltraLong 完好而 Qwen2.5-7B-1M 崩潰**，
   確認發現 15 是**該模型的性質，不是長度效應**，換模型的決定成立。

---

## 🔴 發現 18：端到端的天花板由 prefill 佔比決定 —— 這解釋了 M4 與政策模擬的落差

M4 的壓力掃描給出 21–43% 的 headroom，但政策模擬（`m5_policy_sim.py`）只有 3.76–4.41%。
兩者不矛盾：**M4 只模 prefill，政策模擬含 decode**，而放置決策只能影響 prefill。

| 工作負載 | 模型 | prefill 佔總時間 | oracle 端到端 headroom | ÷ prefill 佔比 = prefill-only |
|---|---|---|---|---|
| toolagent | b-llama8b | 19.1% | 3.76% | 19.7% |
| conversation | b-llama8b | 21.7% | 4.00% | 18.5% |
| toolagent | b-seedoss36b | 17.8% | 4.41% | 24.8% |

→ 推算出的 prefill-only headroom（18.5–24.8%）與 M4 壓力掃描的 21–43% **量級一致**，兩條獨立路徑互相印證。

**對論文的意義**：`main.tex` 引用 headroom 時必須標明是 prefill-only 還是端到端。
平台 B 的 decode 佔 78–82%（Seed-OSS 的 decode 每 token 109.9 ms、Llama 18.8 ms），
所以**端到端的改善上限就是 prefill-only headroom × 約 0.2**。這不是實作缺陷，是問題本身的結構。

## 🔴 發現 19：學習式放置策略在平台 B **沒有**贏過最佳 baseline（負面結果）

`results/m5_predictor_mi300x/policy_sim.csv`，四個設定全部一致：

| 工作負載 | 模型 | oracle 空間 | sym_l2 拿到 | cost_l2 拿到 | 最佳 baseline |
|---|---|---|---|---|---|
| toolagent | b-llama8b | 3.76% | **−33.2%** | −35.0% | tier_fs |
| lc128kz（合成長上下文） | b-llama8b | 8.33% | **−47.3%** | −29.7% | tier_fs |
| conversation | b-llama8b | 4.00% | **−45.3%** | −43.3% | tier_fs |
| toolagent | b-seedoss36b | 4.41% | **−9.7%** | −9.6% | tier_fs |

（負數 = 比最佳 baseline 還差。平台 A 在 lc128kz 上是 **+81.85%**。）

**預測器本身是好的**：AUC 0.917–0.922、ECE 0.003–0.004、正樣本 Spearman 0.99。
問題不在預測品質，而在**可操作空間太小**：

1. 端到端只有 3.76–4.41% 的空間（發現 18），預測器的少數錯誤就足以吃掉全部收益。
2. 平台 B 的各階成本差距被硬體壓縮了：本地 NVMe 讀 5,392 MiB/s、CPU 搬運 2.3–37.8 GB/s，
   使得最簡單的 `tier_fs`（CPU + 磁碟兩階）已經接近最佳 —— 四個 baseline 的端到端差距只有 1.6%。
   平台 A 的 SSD 是 SATA QLC（持續寫 181 MiB/s），階間差距大得多，學習式策略才有發揮空間。
3. Seed-OSS（重算最貴、階間差距最大）的損失最小（−9.7% vs llama 的 −33%），
   方向與上述解釋一致：**階間成本差距越大，學習式放置越有價值**。

**這是負面結果，必須照實寫**。它不否證 M4 的 GO（oracle 確實有 prefill-only 20–40% 的空間），
但它說明：**在階層成本差距被硬體壓縮的平台上，簡單的兩階 baseline 已接近最佳，學習式放置沒有優勢。**
論文的主張應限定在「階間成本差距大」的硬體條件下，而那正是 κ 跨硬體變動 32 倍這個核心論點的另一面。

## 污染疑點的最終處置（對照實驗）

| | 修正前的 run | 對照 run（加了 `WATCHER.pause`） |
|---|---|---|
| guard 判定 | CONTAMINATED_DURING_RUN | **CLEAN** |
| 入侵者 | 1（pid −1、179,679 MiB） | **0** |
| 暫停區間 | 0 | 1 |

同一工作負載、同一張卡，唯一差別是收 server 時有沒有暫停監看 → 證實那五個標記是**自己 server 的 teardown 殘影**。
證據與影響範圍記在 `results/contamination_explained.json`（分數有效、`latency_ms` 作廢），
`code/verify_results_b.py` 會讀它，但**缺 evidence 欄位的條目一律仍判 FAIL**，不是忽略清單。
