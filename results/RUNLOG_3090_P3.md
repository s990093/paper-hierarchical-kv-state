# RUNLOG — 平台 A（RTX 3090，acvlab）第三階段

> 只有 3090 寫這份（MI300X 寫 `results/RUNLOG_MI300X.md`）。格式照 CLAUDE.md §4.2。
> 每個 run 的原始輸出：`/ssd8/hungwei/paper-hkv/runs/<run_id>/`（cmd.sh、context.txt、stdout.log、stderr.log、exit_code、gpu_guard.json）。

---

## Milestone P3-0 — 3090 環境重建

**狀態**: PASS（有例外）——環境、模型、10/11 份資料、所有冒煙測試都過；例外：
LMSYS-Chat-1M **NEEDS_USER**（gated）；`test_m7_correctness.py` 用預設 L=8192 在 24 GB 上 **FAILED（OOM）**（L=2048 過）；
CacheWise 沒有 LICENSE（已下載，授權待確認）。中途 5 個失敗都已記錄原因並修好重跑（見「失敗與異常」）。

**執行時間**: 2026-10-17T17:19:50+08:00（建目錄、裝 Python）→ 2026-10-17T17:47:15+08:00（最後一個冒煙 run 結束）

**run_id**（全部在 `/ssd8/hungwei/paper-hkv/runs/`；GPU 欄是 nvidia-smi index）:

| run_id | GPU | 內容 | exit | Guard | 判定 |
|:--|:--:|:--|:--:|:--|:--|
| 20261017-173416-p3-0-data | — | `setup_3090_data.sh`（全部 11 項） | 1 | — | mooncake FAILED（腳本 bug），其餘 9 OK、1 NEEDS_USER |
| 20261017-173857-p3-0-data-mooncake | — | 修好後只重跑 mooncake | 0 | — | OK |
| 20261017-173543-p3-0-gpusmoke | 0 | `gpu_smoke.py` | 0 | CLEAN | 作廢（當時 libnccl.so.2 是 cu13 套件蓋過去的版本），以 174110 為準 |
| 20261017-173625-p3-0-smoke-triton | 0 | `smoke_3090.py triton` | 0 | CLEAN | 作廢（同上），以 174124 為準 |
| 20261017-173636-p3-0-smoke-libs | 0 | `smoke_3090.py libs` | 1 | CLEAN | FAILED：e5 離線載入失敗；xgboost GPU 靜默退回 CPU |
| 20261017-173955-p3-0-smoke-libs | 0 | 同上（換 xgboost-cu12 後） | 1 | CLEAN | FAILED：`libnccl.so.2` 被刪 |
| 20261017-174110-p3-0-gpusmoke | 0 | `gpu_smoke.py` | 0 | CLEAN | OK |
| 20261017-174124-p3-0-smoke-triton | 0 | `smoke_3090.py triton` | 0 | CLEAN | OK |
| 20261017-174133-p3-0-smoke-libs | 0 | `smoke_3090.py libs` | 0 | CLEAN | OK |
| 20261017-174111-p3-0-prefill-llama-2k | 2 | — | 無 | — | ABORTED（我中止：同一條鏈的下一步輸出路徑寫錯），沒有數字 |
| 20261017-174142-p3-0-prefill-llama-2k | 2 | `smoke_3090.py prefill --ctx 2048`（Llama） | 0 | CLEAN | OK |
| 20261017-174209-p3-0-correct-llama-2k | 2 | `M7_TEST_L=2048 test_m7_correctness.py` | 0 | CLEAN | OK |
| 20261017-174233-p3-0-prefill-llama-32k | 2 | `smoke_3090.py prefill --ctx 32768`（Llama） | 0 | CLEAN | OK |
| 20261017-174343-p3-0-correct-llama-8k | 2 | `test_m7_correctness.py`（預設 L=8192） | 1 | CLEAN | FAILED：OOM |
| 20261017-174142-p3-0-prefill-qwen-2k | 4 | `smoke_3090.py prefill --ctx 2048`（Qwen2.5-7B-1M） | 0 | CLEAN | OK |
| 20261017-174142-p3-0-smoke-vllm | 5 | `smoke_3090.py vllm`（venv/vllm028） | 1 | CLEAN | FAILED：torchcodec 是 CUDA 13 build |
| 20261017-174320-p3-0-smoke-vllm | 5 | 同上（換 CPU torchcodec 後） | 0 | CLEAN | OK |

所有 GPU run 的 GpuWatcher 判定都是 CLEAN，沒有 CONTAMINATED 檔。開跑前 `gpu_guard --idle-gpus` 為 `0 2 4 5 6` 或 `0 1 2 4 5 6`（GPU 1、3 有別人的 htcs 行程）。

**指令**（每一行都包在 `code/run3090.sh` 裡；GPU 的先 `export CUDA_VISIBLE_DEVICES=<i>`）:

```bash
source code/env_3090.sh
TIARA_NO_GPU=1 bash code/run3090.sh p3-0-data bash code/setup_3090_data.sh
TIARA_NO_GPU=1 bash code/run3090.sh p3-0-data-mooncake bash code/setup_3090_data.sh mooncake
CUDA_VISIBLE_DEVICES=0 bash code/run3090.sh p3-0-gpusmoke python code/gpu_smoke.py
CUDA_VISIBLE_DEVICES=0 bash code/run3090.sh p3-0-smoke-triton python code/smoke_3090.py triton
CUDA_VISIBLE_DEVICES=0 bash code/run3090.sh p3-0-smoke-libs env HF_HUB_OFFLINE=1 python code/smoke_3090.py libs
CUDA_VISIBLE_DEVICES=2 bash code/run3090.sh p3-0-prefill-llama-2k python code/smoke_3090.py prefill --ctx 2048
CUDA_VISIBLE_DEVICES=2 bash code/run3090.sh p3-0-correct-llama-2k env M7_TEST_L=2048 bash -c 'python code/test_m7_correctness.py "$RUN_DIR/correctness.json"'
CUDA_VISIBLE_DEVICES=2 bash code/run3090.sh p3-0-prefill-llama-32k python code/smoke_3090.py prefill --ctx 32768
CUDA_VISIBLE_DEVICES=2 bash code/run3090.sh p3-0-correct-llama-8k bash -c 'python code/test_m7_correctness.py "$RUN_DIR/correctness.json"'
CUDA_VISIBLE_DEVICES=4 bash code/run3090.sh p3-0-prefill-qwen-2k env M7_MODEL_GLOB=$M7_QWEN_GLOB python code/smoke_3090.py prefill --ctx 2048
CUDA_VISIBLE_DEVICES=5 bash code/run3090.sh p3-0-smoke-vllm env PATH=$M7_VLLM_VENV/bin:$PATH HF_HUB_OFFLINE=1 \
    $M7_VLLM_VENV/bin/python code/smoke_3090.py vllm --model $TIARA_QWEN3_06B_PATH --gpu-mem 0.5
```

安裝與下載（不產生實驗數字，沒走 run 殼；log 在 `/ssd8/hungwei/paper-hkv/logs/`）：`01_uv_python_install`、`02_tiara_torch`、
`03_tiara_pkgs`、`04_tiara_pycox`、`05_pylibs_wonderwords`、`06_vllm028`、`06b_vllm028_torchcodec_cpu`、`07_tinytex`、`08_tlmgr_install`、
`09_tex_smoke`、`10_dl_models`、`11_verify_ssd7_snapshots`、`12_copy_ssd7_models`、`13_dl_e5_retry`、`14_verify_new_cache`、
`15_tiara_xgboost_cu12`、`16_hf_whoami`。確切指令與版本見 `docs/ENV_3090.md` §2–§4。

**產出檔**:
- repo：`code/env_3090.sh`、`code/run3090.sh`、`code/setup_3090_data.sh`、`code/smoke_3090.py`、`code/m7_env.sh`（加 3090 分支）、
  `.claude/settings.json`（env 區塊）、`docs/ENV_3090.md`、本檔。`_big -> /ssd8/hungwei/paper-hkv`（gitignore）。
- `/ssd8/hungwei/paper-hkv/`：`venv/tiara`、`venv/vllm028`、`hf-cache/hub/`（5 個模型）、`datasets/`（10 份，各有 README.txt）、`texlive/.TinyTeX`、`pylibs/`、
  `logs/pip_freeze_tiara.txt`、`logs/pip_freeze_vllm028.txt`、各 run 目錄裡的 `*.json`。

**關鍵數字**（只寫量到的；冒煙等級，時間與頻寬**不可寫進論文**）:

| 項目 | 值 | run_id |
|:--|:--|:--|
| GPU 能算、算得對 | bf16 8192³ matmul 中位數 70.1 TFLOPS；max\|I@Y−Y\| = 0；HBM d2d 835.9 GB/s；pinned H2D 6.15 GiB/s、D2H 5.98 GiB/s（1 GiB、10 次） | 174110-p3-0-gpusmoke |
| PCIe 連線上限 | nvidia-smi：7 張卡都是 `pcie.link.gen.max=3`、`pcie.link.width.max=16`（閒置時 gen.current=1） | context.txt（同上） |
| Triton | 向量加法 max err 0.0；列 softmax max err 1.49e-8；bf16 matmul 有限值；Triton 3.6.0 | 174124-p3-0-smoke-triton |
| 第三階段套件 | 28 個模組全部 import；lightgbm CPU 訓練準確率 0.9945；xgboost-cu12（build CUDA 12.9）cpu／cuda 都 1.0 且沒有退回 CPU 的 warning；sksurv Cox 可 fit；e5-small 在 cuda:0 encode（dim 384） | 174133-p3-0-smoke-libs |
| Llama-3.1-8B BF16，KVModel，2K prefill | torch peak allocated 15.278 GiB、reserved 15.406 GiB；nvidia-smi 本行程峰值 16,088 MiB；最後 logits 有限 | 174142-p3-0-prefill-llama-2k |
| **Llama-3.1-8B BF16，KVModel，單卡 32K prefill** | **放得下**：KV 緩衝 4.0 GiB；載入後 allocated 18.958 GiB；**peak allocated 19.028 GiB、reserved 19.156 GiB；nvidia-smi 峰值 19,928 MiB**；最後 logits 有限 | 174233-p3-0-prefill-llama-32k |
| Qwen2.5-7B-1M BF16，KVModel，2K prefill | peak allocated 14.418 GiB、reserved 14.545 GiB；nvidia-smi 峰值 15,206 MiB；L=28、KV 頭 4、D=128 | 174142-p3-0-prefill-qwen-2k |
| correctness（L=2048） | hf_twice_bitwise_equal＝true；own_vs_hf_maxabs_logit＝0.125；argmax 相同；top-5 重疊 5；K maxabs（第 0／中／末層）＝0.125／0.28125／0.1875；recompute_same_chunking_bitwise_equal＝true；host_roundtrip_bitwise_equal＝true | 174209-p3-0-correct-llama-2k |
| vLLM 0.28.0+cu129 | Qwen3-0.6B（max_model_len 4096、gpu_memory_utilization 0.5）greedy 生成 16 token：「 Paris. The capital of Italy is Rome. The capital of Spain is Madrid.」；KV cache 94,592 token | 174320-p3-0-smoke-vllm |
| Mooncake 單位（規則 6） | `len(hash_ids)==ceil(input_length/512)` 吻合率：conversation 12,031/12,031、synthetic 3,993/3,993、toolagent 23,608/23,608、arxiv 23,608/23,608（都是 1.000000）；`m8_trace_oracle.load_trace` 斷言全過，chain_bad_pos＝chain_bad_prev＝0 | 173857-p3-0-data-mooncake |
| Bailian 單位（規則 6） | `len(hash_ids)==ceil(input_length/16)` 吻合率：traceA 43,058、traceB 172,800、thinking 10,812、coder 43,011，全部 1.000000 | 173416-p3-0-data |
| TeX | `main.tex` 的複本（不是 repo 裡那份）pdflatex＋bibtex：11 頁、0 overfull、0 undefined、0 font warning | logs/09_tex_smoke.log |
| 磁碟 | 這次寫進 `/ssd8/hungwei/paper-hkv` 的量：84,209,503,029 B（78.4 GiB），不含其他 agent 的 `tmp/p3x*`、`runs/*p3x*`；`/ssd7` 沒寫（仍 106G 可用） | du -sb，17:5x |

**失敗與異常**（照時間順序；都是環境建置中的問題，修法見 `docs/ENV_3090.md` §2–§3）:

1. **e5-small 第一次下載抓錯檔**（logs/10_dl_models.log）：`hf download ... --exclude 'onnx/*' 'openvino/*' 'pytorch_model.bin'` 把後兩個 pattern 當成檔名，
   CLI 印 `UserWarning: Ignoring --exclude since filenames have been explicitly set.`，只抓到 openvino 與 pytorch_model.bin。刪掉該 repo 目錄（本次建立的），
   改成每個 pattern 一個 `--exclude` 重下（logs/13_dl_e5_retry.log）。
2. **tlmgr 要先自我更新**（logs/08_tlmgr_install.log）：`tlmgr itself needs to be updated. ... tlmgr: Terminating; please see warning above!`（exit 255）。
   照它的指示 `tlmgr update --self` 後安裝成功。
3. **run 173416-p3-0-data：mooncake FAILED**，stderr：`IndexError: list index out of range`。原因是腳本 bug：Mooncake 根目錄的授權檔叫 `LICENSE-APACHE`，
   不是 `LICENSE`，tree 查不到。改腳本、刪掉本次建立的半成品 `datasets/mooncake/`、只重跑 mooncake（173857，OK）。
4. **run 173636-p3-0-smoke-libs FAILED**：
   - e5：`OSError("We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files. ...")`。
     原因：用 `--revision <sha>` 下載的 repo 沒有 `refs/main`，`HF_HUB_OFFLINE=1` 時用名稱載入解析不到。改用 snapshot 路徑（`TIARA_E5_PATH`）。
   - xgboost 3.4.1（PyPI 預設，`build_info` CUDA_VERSION＝13.3）：`device="cuda"` 時 stderr 只有
     `WARNING: ... No visible GPU is found, setting device to CPU.`、`Device is changed from GPU to CPU as we couldn't find any available GPU on the system.`，
     **訓練照樣成功、回傳 acc 1.0**——那是 CPU 的數字。冒煙腳本當時沒抓到，已改成把這種 warning 判為錯誤。改裝官方 CUDA 12 變體 `xgboost-cu12==3.4.1`。
5. **run 173955-p3-0-smoke-libs FAILED**：`ImportError('libnccl.so.2: cannot open shared object file: No such file or directory')`（torch、accelerate、peft、pycox、torchtuples、
   sentence_transformers 都 import 不了）。原因：`nvidia-nccl-cu13`（xgboost 3.4.1 的相依）和 `nvidia-nccl-cu12` 裝在同一個檔 `nvidia/nccl/lib/libnccl.so.2`，
   解除安裝 cu13 時把它刪了。`--reinstall-package nvidia-nccl-cu12 nvidia-nccl-cu12==2.27.5` 修回（sha256 和 wheel RECORD 相同；`torch.cuda.nccl.version()`＝(2,27,5)）。
   **連帶**：173543-gpusmoke 與 173625-smoke-triton 跑的時候 `libnccl.so.2` 其實是 cu13 套件的版本，兩個都作廢，以 174110／174124 重跑為準（單卡不用 NCCL，數字差異只是 smoke 雜訊）。
6. **run 174111-p3-0-prefill-llama-2k ABORTED**：同一條指令鏈的下一步（correctness）我把輸出路徑寫成字面上的 `__RUNDIR__`，發現後把整條鏈停掉（run 目錄留 `ABORTED` 檔），
   改用 `bash -c '... "$RUN_DIR/correctness.json"'` 重跑。這個 run 沒有 exit_code、沒有數字。
7. **run 174142-p3-0-smoke-vllm FAILED**：`OSError: libnvrtc.so.13: cannot open shared object file: No such file or directory` →
   `OSError: Could not load this library: .../torchcodec/libtorchcodec_image.so`；呼叫鏈是 `from vllm import LLM` → `vllm/multimodal/video.py:35 from torchcodec.decoders import VideoDecoder`。
   原因：`--index-strategy unsafe-best-match` 從 PyPI 選了 torchcodec 0.17.0（CUDA 13 build）；cu129 index 最高只有 0.16.0+cu129。改裝 `torchcodec==0.17.0+cpu`（174320，OK）。
8. **run 174343-p3-0-correct-llama-8k FAILED（沒有修，照實記錄）**：
   `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 23.68 GiB of which 948.94 MiB is free. Including non-PyTorch memory, this process has 22.74 GiB memory in use. Of the allocated memory 21.99 GiB is allocated by PyTorch, and 461.84 MiB is reserved by PyTorch but unallocated.`
   位置：`test_m7_correctness.py` 第 60 行 `recompute_same_chunking_maxabs`（`ref.float() - km.kv[...].float()` 把整塊 KV 轉 fp32）。
   這是測試本身的記憶體用法在 24 GB 放不下，不是模型或 KVModel 的問題；L=2048 全過。要在 3090 上跑預設 L 需要改測試（逐層比對），這次沒改。
9. **LMSYS-Chat-1M NEEDS_USER**：HF API `gated="auto"`，要使用者登入並同意條款；沒有下載（`docs/ENV_3090.md` §6）。
10. **CacheWise 沒有 LICENSE**：GitHub API license＝null，repo 裡沒有授權檔；已下載，使用前要問作者。

**與論文假設的差異**（這裡是和既有文件的假設比，不是 main.tex 的結果）:
- `docs/migration_20261013/README.md` §2.3：「Llama-3.1-8B 做到 32K 很緊〔算術，約 21–23 GB，沒量過〕」→ 實測 KVModel 單卡 32K prefill **放得下**：
  torch peak allocated 19.028 GiB、nvidia-smi 本行程峰值 19,928 MiB（run 174233）。只量了 prefill（chunk 512、SDPA）；還原、Cake、vLLM 的 32K 沒量（NOT_MEASURED）。
- 同一節：「3090 ... PCIe 是 4.0」→ nvidia-smi 回報 7 張卡 `pcie.link.gen.max=3`（PCIe 3.0 x16）。smoke 的 pinned H2D 中位數 6.15 GiB/s。
  這個數字是 smoke 等級（單 stream、1 GiB、10 次、同一 PCIe switch 上其他卡可能在搬資料而 GpuWatcher 看不到），正式的 H2D 要用 `m7_calib.py c0`（P3-2a）。
- HANDOFF／EXPERIMENT_PLAN §8 只記了「vLLM 的 PyPI wheel 是 CUDA 13」；這次另外踩到兩個同類的坑：PyPI 的 **xgboost**（GPU 靜默退回 CPU）和 **torchcodec**（vLLM import 失敗）。
- 第三階段計畫說內容型預測器（P3-7）可能要使用者同意 WildChat 的條款：實際上 `allenai/WildChat-1M` 目前 gated=False（ODC-BY），已下載。

## Milestone P3-0b — 24 GB 上的正確性檢查：雙卡對照模式（使用者 10/17「OOM 改雙卡」）
**狀態**: PASS（Llama 8K 單卡、Llama 8K 雙卡、Qwen 32K 雙卡、Llama 32K 雙卡第 2 次）；Llama 32K 雙卡第 1 次 FAILED（OOM，見下）
**執行時間**: 2026-10-17 17:59:53 → 18:0x（+08:00）
**run_id**: `20261017-175953-p3-0-corr-llama-8k-1gpu`（GPU 2）、`20261017-175953-p3-0-corr-llama-8k-2gpu`（GPU 4,5）、`20261017-180212-p3-0-corr-qwen-32k-2gpu`（GPU 2,6）、`20261017-180212-p3-0-corr-llama-32k-2gpu`（GPU 4,5，FAILED）、`20261017-180434-p3-0-corr-llama-32k-2gpu-r2`（GPU 4,5）
**指令**: `source code/env_3090.sh; M7_MODEL_GLOB=$M7_LLAMA_GLOB CUDA_VISIBLE_DEVICES=4,5 TIARA_TWO_GPUS=1 m7run p3-0-corr-llama-32k-2gpu-r2 bash -c 'PYTORCH_ALLOC_CONF=expandable_segments:True M7_TEST_L=32768 M7_TEST_REF_DEVICE=cuda:1 python code/test_m7_correctness.py "$RUN_DIR/correctness.json"'`（其他 run 換 L、模型、卡；單卡不設 `M7_TEST_REF_DEVICE`、`TIARA_TWO_GPUS`；`PYTORCH_ALLOC_CONF` 只有 r2 設）
**產出檔**: 各 run 目錄的 `correctness.json`、`gpu_guard.json`（雙卡另有 `gpu_guard_cuda1.json`）
**程式改動**（不是 tensor parallel：受測的 KVModel 整個在第一張卡，和實驗同一條路徑）:
- `code/test_m7_correctness.py`：單卡時 HF 的整段輸出用完就釋放、maxabs 逐層算（值和整塊一次算相同）；`M7_TEST_REF_DEVICE=cuda:1` 時 HF 對照模型、重算前的 KV 副本與所有比對放第二張卡，HF 只算最後一個位置的 logits（`logits_to_keep=1`），只取 logits 的兩次呼叫 `use_cache=False`（JSON 記 `hf_logits_to_keep`、`hf_logit_calls_use_cache`）。
- `code/run3090.sh`：`TIARA_TWO_GPUS=1` 才准 `CUDA_VISIBLE_DEVICES=<i>,<j>`；兩張卡開跑前都 `gpu_guard --check`。單卡行為不變。
- `code/m7_guard_run.py`：看得到幾張卡就各開一個 GpuWatcher（第二張寫 `gpu_guard_cuda1.json`）；一張卡時和原本相同。
**關鍵數字**（全部 bitwise 旗標＝true、argmax 相同、top-5 重疊 5；GpuWatcher 每張卡都 CLEAN）:

| run | L | 模式 | own vs HF logit maxabs | K maxabs（第 0／中／最後層） | 重算逐位元 | 主機往返逐位元 |
|:--|--:|:--|--:|:--|:--|:--|
| llama-8k-1gpu | 8192 | 單卡 | 0.09375 | 0.125／0.3765／0.25 | true | true |
| llama-8k-2gpu | 8192 | 雙卡 | 0.09375 | 0.125／0.3765／0.25 | true | true |
| qwen-32k-2gpu | 32768 | 雙卡 | 0.125 | 0.5／0.3125／4.0 | true | true |
| llama-32k-2gpu-r2 | 32768 | 雙卡 | 0.09375 | 0.125／0.5／0.25 | true | true |

- 8K 單卡與雙卡的每一個數字都相同 → 雙卡模式（含 `logits_to_keep=1`）沒有改變比對結果。
- Qwen 最後一層 K maxabs 4.0 是絕對誤差；Qwen 的 K 有很大的逐 channel 偏移（main.tex §5.3），這次沒有算相對誤差（NOT_MEASURED）。
**失敗與異常**:
- `20261017-180212-p3-0-corr-llama-32k-2gpu` FAILED（exit 1），對照卡（cuda:1＝GPU 5）在第一次 HF 整段前向（測試第 56 行 → `modeling_llama.py` MLP）：
  `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 896.00 MiB. GPU 1 has a total capacity of 23.68 GiB of which 762.94 MiB is free. Including non-PyTorch memory, this process has 22.92 GiB memory in use. Of the allocated memory 20.73 GiB is allocated by PyTorch, and 1.89 GiB is reserved by PyTorch but unallocated.`
  原因：HF 預設 `use_cache=True`，只取 logits 的呼叫也建了一份 32K 的 KV cache，加上碎片。改成這兩次 `use_cache=False`＋`expandable_segments` 後 r2 通過。
- 這個 session 的 shell 還帶著舊 settings 的 `HF_HUB_ENABLE_HF_TRANSFER=1`，stderr 有 FutureWarning；不影響結果（新 settings 要重開 session 才生效）。
**與假設的差異**: P3-0 寫「要在 3090 上跑預設 L 需要改測試」→ 已改；單卡 L=8192 現在放得下，32K 用雙卡對照模式（Llama、Qwen）都過。

### P3-0b 補記：改 `code/run3090.sh` 時打斷了別人正在跑的 run（主 session 的錯）
- 2026-10-17 18:00 前後，主 session 為了雙卡模式改了 `code/run3090.sh`，那時 X2 的 `20261017-175914-p3x2-ph-mc-strict` 正在用它。bash 是邊跑邊讀腳本，所以那次 run 讀到改到一半的檔，報 `line 111: syntax error near unexpected token 'then'`，沒有寫 exit_code。X2 留了 INTERRUPTED，用同一指令重跑成 `20261017-175941-p3x2-ph-mc-strict2`；之後回報確認還波及 X7 的 `20261017-175844-p3x7-busy-qwen`（冒煙 run，沒有 exit_code，X7 留了 INTERRUPTED、不用；X7 的報告誤寫成「被環境 agent 改了」，實際是主 session）。其他 agent 的回報都說自己的 run 有 exit_code。
- 之後要做的（還沒做；要等沒有任何 run 在用這支殼才改，免得再打斷一次）：把 `run3090.sh` 的主體包成 `main() { ...; }; main "$@"; exit $?`，讓 bash 先讀完整支再執行；同樣檢查 `env_3090.sh`。在那之前，**有 run 在跑時不改這兩支檔**。

## Milestone P3-M1 — 壓縮格式表（地圖實驗的輸入；只讀第二階段結果）
**狀態**: PASS（有缺：見下）
**執行時間**: 2026-10-18 00:25 → 02:34（+08:00；中途被 API 用量上限打斷一次）
**run_id**: `20261018-002550-p3fmt-qderive`（補算品質）、`20261018-023443-p3fmt-table`（這一版表）；失敗的 `20261018-023324-p3fmt-table`（exit 1，程式錯，見下）
**指令**: `TIARA_NO_GPU=1 bash code/run3090.sh p3fmt-table python code/p3_formats/build_format_table.py`（品質補算：`code/p3_formats/derive_quality.py`，用凍結的 `m10_analyze.py`）
**產出檔**: `results/p3_formats/format_table.{csv,json}`（30 列＝2 模型 × 15 格式，每格有 run_id 與來源檔）、`quality_pending.csv`、`derived/`；文件 `docs/phase3_20261017/05_format_table.md`
**關鍵數字**（只寫量到的）:
- 位元組比（含 scale／zero，程式算＝存下來的實際大小，逐位元組相同）：FP8 主線 Llama 0.516／Qwen 0.500、INT8 0.516、INT4-KIVI 0.3125、INT2-KIVI 0.1875。
- 32K 生成品質（主 session 抽查，和 main.tex 表 VII 一致）：Llama FP8 −0.47（全載入）／−0.21（Cake 閒）／−0.48（Cake 忙）過；Qwen FP8 −32.84 不過；Qwen INT4-KIVI +0.06／+0.66／+0.41 過。新補算：Llama FP8-K0 +0.15、INT8 −0.12（都過；第二階段沒報過，補算前先做兩個重現檢查，逐欄相同）。
- INT2-KIVI 只有 teacher forcing（Llama −1.04、Qwen −6.25，無 CI），**不能當成「過」**；生成品質待 MI300X stage 43（Llama）、44c（Qwen）。
- 反量化（還原中、真實 CPU 層、32K）：例 Qwen INT4-KIVI MI300X 原寫法 0.411／1.038 ms（閒／忙）、融合 0.082／0.351；3090 原寫法 1.998／3.893。
**失敗與異常**:
- `20261018-023324-p3fmt-table` exit 1：`AttributeError: 'tuple' object has no attribute 'astype'`（`build_format_table.py` 第 285 行把 `dq3.shape` 當成欄位）。改 `dq3["shape"]` 後重跑。
- 對拍發現的矛盾（05 文件 §5）：X7 說 MI300X 沒有單獨的反量化 bench，其實 `results/m10_sweep_mi300x/parts/dequant_fast_bench.json` 有；同一支工具比，3090 原寫法慢 3.6–5.1 倍（X7 寫的 2–4 倍是拿不同量法比）。Qwen INT4 單獨量兩份來源差 1.9 倍，原因未查。
- stage 43 的速度格子不在這個分支，從本地 `origin/main`（1ddba0b）用 `git show` 讀，只取反量化時間。
**與假設的差異**: 04 草稿 §1 寫 Qwen 的 8-bit 用 INT8：表支持（Qwen INT8 +0.20 過、FP8 不過）。04 §3 的 O 會用 INT2 存「不會再被讀」的 session：INT2 的生成品質目前沒有，所以 INT2 只能給 oracle 確定不會被讀的 session（和 04 寫的一致）。

## Milestone P3-M2 — 3090 量測：/ssd8 頻寬、CPU 忙的變慢倍數、補 X7 沒量到的格子
**狀態**: PASS（Llama 64K CPU 層 OOM，見下）
**執行時間**: 2026-10-18 00:19 → 02:44（+08:00；中途被 API 用量上限打斷一次）
**run_id**: `20261018-001935-p3cal-ssd8bw`、`-002700-p3cal-stress-selftest`、`-002928-p3cal-grid-llama-L0`、`-010316-p3cal-grid-llama-L1`、`-011852-p3cal-grid-qwen-L1b`、`-004505-p3cal-fused-qwen`、`-004724-p3cal-fused-llama`、`-004954-p3cal-cpubusy-qwen`、`-023443-p3cal-stress-numa`；失敗 `-013157-p3cal-grid-llama64k-idle`
**指令**: 磁碟 `TIARA_NO_GPU=1 bash code/run3090.sh p3cal-ssd8bw bash /ssd8/hungwei/paper-hkv/logs/p3cal/disk.sh`（`code/disk_bw.py`，16 GiB × 3、O_DIRECT）；GPU 全部經 `bash code/gpu_wait_run.sh --gpus "6" ...`（腳本在 `/ssd8/hungwei/paper-hkv/logs/p3cal/`）
**產出檔**: `results/p3_calib_3090/`（`ssd8_bw*.csv`、`cpubusy_*.csv`、`restore_3090_v2_*.csv`、`lowprec_cpu_v2.csv`、`meet_check_v2.csv`、`constants_3090_v2.json`）；文件 `docs/phase3_20261017/06b_3090_calib.md`
**關鍵數字**（只寫量到的；GPU 6 個 run 全部 CLEAN，GPU 時間 58 分鐘，磁碟測試與 GPU 量測沒有重疊）:
- **/ssd8**（/dev/sdg，Samsung 870 QVO 8TB，SATA QLC，78% 滿）：寫入中位 0.151 GiB/s（155.1 MiB/s）、讀取 0.477 GiB/s；比 MI300X 的本地 SSD 讀慢 14.6 倍、寫慢 12.6 倍。測試期間外人寫入約 2.5–3.7 MiB/s。
- **CPU 忙**（Qwen 32K、真實 CPU 層、GPU 閒；N 個 numpy 複製執行緒）：只載入的還原時間慢 **1.35–1.81 倍**；N ≥32 塞滿推論節點的 CPU 時，GPU 重算也慢 3.1 倍（發 kernel 的 Python 執行緒搶不到 CPU），Cake 的 TTFT 慢 2.3–2.6 倍。壓力程式的記憶體擺在哪個 NUMA 節點沒控制到（沒有 numactl），倍數只能當範圍。
- **Llama 32K 還原格子**（兩次）：真實 CPU 層會合點閒 4、忙 3（和 X7 的模型預測相同），18／18 格模型誤差在 ±1 block 內，兩次差 ≤1%。
- **Qwen 第二次**：只載入 0.248 s（X7 L0 是 0.369 s；X7 那次鄰卡有 PCIe 流量，時間偏高）→ CPU 層 Cake 比只載入慢 60%（不是 X7 寫的 10%）。
- **融合 kernel，CPU 層、只載入、GPU 閒**：Qwen FP8 −48%、INT4 −66%；Llama FP8 −47%、INT4 −67%（對 BF16）。3090 上低精度讓 CPU 層變快，和 MI300X 相反（X7 的發現，這次加上 Llama 與融合版）。
**失敗與異常**:
- `20261018-013157-p3cal-grid-llama64k-idle` exit 1（GPU 6 乾淨，只有自己的 process）：`torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 MiB. GPU 0 has a total capacity of 23.68 GiB of which 58.94 MiB is free. Including non-PyTorch memory, this process has 23.60 GiB memory in use. Of the allocated memory 23.10 GiB is allocated by PyTorch, and 176.49 MiB is reserved by PyTorch but unallocated.`（`m7_restore_harness.py:176 _load`，接著 `KeyError: 'l_end'` at :581）。照規則 2 沒有換設定重試；busy 版沒跑。
- agent 自己的失誤：`pkill -f "logs/p3cal/chain.sh"` 連帶砍到自己的 shell（exit 144），留下孤兒 `gpu_wait_run.sh`，已用 PID 停掉；沒有影響量測。主 session 事後確認沒有殘留的 p3cal 程序。
- NOT_MEASURED：Llama 64K CPU 層；第三次 C0 的 duplex（02:33 起所有卡都被別人佔用）；CPU 忙＋GPU 忙同時；Llama／低精度的 CPU 忙倍數；CPU 忙倍數的重複性（只跑一次）。
**與假設的差異**: 04 草稿 §5 的「3090 的 /ssd8 頻寬」原本 NOT_MEASURED，現在有了：它是 SATA QLC，比地圖主設定用的 MI300X 本地 SSD 慢一個數量級，3090 的真實機器點要用這個值。

## Milestone P3-M3 — 預測器與 loss 開發（**合成資料、方法開發，不是結論**）
**狀態**: PASS（DeepSurv 3 格發散，見下）
**執行時間**: 2026-10-18 00:2x → 03:1x（+08:00；中途被 API 用量上限打斷一次）
**run_id**: `20261018-023915-p3ld-A`、`-025026-p3ld-Areal`（真實格式表重跑 A）、`-024809-p3ld-B`（exit 5）、`-030646-p3ld-Brerun`（exit 5，3 筆 DeepSurv NaN 記成錯誤）、C、D、`D_w03`、E、`E_lam16`（完整清單在 07 文件）
**指令**: `TIARA_NO_GPU=1 bash code/run3090.sh p3ld-<短名> python code/p3_lossdev/run_cell.py ...`；判讀 `code/p3_lossdev/analyze.py`
**產出檔**: `results/p3_lossdev/`（`A.csv.gz`、`A_realtable.csv.gz` 是壓縮版，原檔 >1 MB，放在各自的 run 目錄）；文件 `docs/phase3_20261017/07_lossdev_synth.md`（開頭是「設計選擇與依據」表）、協定草稿 `08_prereg_predictor_draft.md`（未凍結）
**關鍵數字**（合成資料：45 份＝ρ 5 種 × 不回來比例 π 3 種 × 3 seed；主設定 λ＝4；主 session 抽查 `A_summary.csv`）:
- ρ＝1（線索最強）：離散 hazard −8.7% [−14.9, −2.6]、Cox −8.8%、BCE（1 小時）−8.5%（對 LRU 的 regret）；只補回 LRU→oracle 差距的 33–35%；oracle −20 到 −21%。
- ρ＝0（線索沒用）：事後移門檻的方法都和 LRU 打平（+0.3% 到 +0.7%）；唯一明顯輸的是「成本加權＋固定門檻 0.5」（+10.8% [6.6, 15.0]），和 9 月 B6 同方向。
- 模型：logistic regression 和小樹（≤200 棵、≤15 葉）最穩；大模型在 D2 過擬合。推論（batch 1024，每列）：LR 0.29 µs、LightGBM 200 棵 6.7 µs。
- 整體退路（valid 上沒顯著贏 LRU 就改用 LRU）有用：λ＝16、ρ≤0.5 時把 +2% 左右的劣化拉回 0。
- 特徵消融在合成資料上看不出 F2 的貢獻：這是產生器的設計造成的，不能當成真實資料的結論。
**失敗與異常**:
- `20261018-024809-p3ld-B` exit 5：3 格 `ValueError: NaNs detected in inputs, please correct or drop.`（lifelines concordance_index）；原因是 DeepSurv（pycox CoxPH）在 (ρ,π,seed)＝(0.25,0.9,2)、(1.0,0.9,0)、(1.0,0.9,1) 訓練發散、分數全 NaN。加 NaN 檢查重跑（`-030646-p3ld-Brerun`），3 筆記成錯誤；沒有調 DeepSurv 硬跑。用 `B_full.csv`，不要用 `B_summary.*`（缺 3 格）。
- 產生器開發時兩個斷言失敗（scratch）：`1−ndtr` 尾端精度不夠；session 長度重尾 → w 從 0.5 改 0.2（07 §2.10）。
- RSF 太慢（一格 >5 分、10 GB）：在任何 B 結果之前改成最多 6,000 列、葉 ≥30。
- agent 的 `pkill -f` 誤殺自己的 shell（exit 144），試跑重啟，不影響結果。
**與假設的差異**: 9 月 B5「設限補值會壞排序」在這裡沒出現：只有兩個動作（品質過關的最小格式／INT2）時只需要「會不會被讀」，同分不傷；若加入「依回來時間選精度」的動作要重驗。

## Milestone P3-M4 — 統一模擬器（工具與驗證；**沒有跑完整地圖**）
**狀態**: PASS（不變量 (iii) 的第二主指標有 2 格極小違反，照報）
**執行時間**: 2026-10-18 00:2x → 03:3x（+08:00；中途被 API 用量上限打斷一次）
**run_id**: `20261018-002806-p3sim-inputs`、`-030356-p3sim-smoke`（冒煙，數字只在 run 目錄）、`-032839-p3sim-timing`、`-032844-p3sim-test`（28 passed）；失敗 `-025459-p3sim-smoke`、`-030356-p3sim-timing`（exit 1）；中止 `-024805-p3sim-smoke`（ABORTED）
**指令**: `TIARA_NO_GPU=1 bash code/run3090.sh p3sim-test python -m pytest code/p3_sim/test_p3_sim.py`
**產出檔**: `code/p3_sim/`（session 引擎、block 引擎、外層做法設定 `config/outer_policies.json`、壓縮決定、記帳）；文件 `docs/phase3_20261017/06_simulator.md`
**關鍵數字**（不變量，run `20261018-032844-p3sim-test`）:
- (i) 對 X1 的 LRU：144／144 逐位元相同。
- (ii) 對 X4 的佇列預取：8／8 逐位元相同。
- (iii) O 的品質帳全為 0；SSD 讀寫總量 O ≤ 每個品質帳為 0 的固定做法：120／120（block 引擎 10／10）；**SSD 尖峰 p99：118／120**，2 格（synth、E2／E5、16 GiB，對 C3）O 高 5e-5 GiB/s（6.691947 對 6.691897）。
- (iv) O 容量單調：84／84。(v) 同設定同 seed 指紋相同。
- 冒煙：1,096 格，記帳恆等式 7,746 條全部成立；沒有算、也沒有看任何「省多少」。
**失敗與異常**:
- 開發中 (iii) 第一次沒過：`AssertionError: O 的品質帳不是 0：E2 16 1408`（block 引擎 `assert 273 == 0`）。O 只看寫入當下的長度選格式（之後更長的請求會讀到）；block 引擎把只有 promote 的那一輪當成不會再讀。兩處都改了。
- `20261018-025459-p3sim-smoke`：`TraceAssertion('bailian/traceA: 前綴鏈被違反 38 次（同一個 block key 出現在不同位置）')`。統一表裡 traceA 有 52,168 個 id 出現在不只一個位置（coder、thinking 為 0）；改用整段前綴當 key。**資料層發現**：03b 沒有檢查 Bailian 的跨請求鏈性質，要補。
- `20261018-030356-p3sim-timing` exit 1：`TraceAssertion: freeinference/week13: 父請求排在子請求之後（472 ≥ 471）`（16 週只有 1 列，step 7 比 step 6 早 14 ms）。處理：子請求的到達時刻夾到父請求並計數，比例 >1e-4 仍丟錯。**主 session 暫定接受**（1 列／122 萬列、差 14 ms，對結果沒有實質影響；使用者可推翻）。
**與假設的差異**:
- 模擬器只有開迴路重播；03 §5 的預設是閉迴路（要在登記時決定）。
- X6 查到 LMCache 預設是寫穿，比較像 E3，不是 04 寫的 E2；E2 改稱「LRU、滿了才寫（通用）」。
- E6（block 引擎、一張 GPU）在整份 Mooncake 上 GPU 忙碌比例 3.27、Bailian coder 1.19：一張 GPU 撐不住，要決定每張 GPU 分多少 session。
- 完整地圖〔估計，用 timing run 線性外插〕：session 引擎約 160–210 CPU 小時（8 個 process 約 20–26 小時）；E6 約 160–720 CPU 小時。時間幾乎都花在 FreeInference。

## Milestone P3-M5 — 地圖登記 v4 草稿＋模擬器準備（opencode 做；**沒有跑任何地圖的格子**）
**狀態**: PASS（工具與資料統計）；登記 v4 **未凍結**
**執行時間**: 2026-10-18 08:11 → 11:44（+08:00）
**run_id**:
- 審查 v3：`20261018-081143-p3-ocreview-map-v3`（exit 0，但 opencode 的最後答案沒寫出來：`finish=length`，32K 輸出用完；意見從匯出的思考文字整理，`docs/phase3_20261017/review_opencode_20261018_map_v3.md`）。**zero-context same-family review（NOT cross-model）**。
- 模擬器準備（opencode，`anthropic/claude-opus-5-5` variant high）：`20261018-082935-oc-simprep`（exit 0；子 run 見 `/ssd8/hungwei/paper-hkv/opencode/tasks/20261018-082929_simprep/REPORT.md` §2）。
- 主 session 重跑驗證：`20261018-114600-p3sim-verify-test`（46 passed）、`p3data-verify-test`（52 passed）。
**指令**: `TIARA_NO_GPU=1 bash code/run3090.sh oc-simprep bash code/oc_task.sh <任務目錄> --variant high`
**產出檔**: `docs/phase3_20261017/04_prereg_map.md`（v4 草稿，commit `0671f0e`、`e13b0b2`）；`code/p3_sim/`、`code/p3_data/`、`docs/phase3_20261017/06_simulator.md` §11–§17（commit `d0c4658`）；`results/p3_sim/{return_gaps,session_counts,kgpu_*}.csv`、`results/p3_data/raw_id_chain.csv`；`code/oc_task.sh`
**關鍵數字**（不變量與資料統計；沒有任何「省多少」）:
- 預設行為不變：X1 144／144、X4 8／8 逐位元；冒煙裡 C0–C4 的舊 103 欄在 911 格全部相同；O（照規格改成搬去 SSD 時也決定）185 格裡 72 格變了。
- 回來間隔 p99：Bailian coder 3,488.7 s、traceA 2,207.3 s、WildChat 27,511.2 s、Mooncake 1,548.6 s、FreeInference 每週 298.9–2,174.0 s（run `20261018-102541-p3sim-gaps2`）。
- 有效 session（1／Σp²）：FreeInference 第 1 週只有 14.8，4 週合起來 479.3（同一個 run）。
- E6 的 GPU 數：草稿的「每小時最大 ≤0.8」在 FreeInference 第 10–12、14–16 週做不到（單一 session 單獨一張 GPU 就忙到 0.85–1.0，run `20261018-102627-p3sim-kgpu4`）；v4 改成整段平均。
**失敗與異常**:
- 審查的輸出被截斷（上面）。之後 `oc_task.sh` 設 `OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=64000`。
- opencode 在兩個開發用指令裡設了 `CUDA_VISIBLE_DEVICES=""`（沒有 run_id、沒有產生報告用的數字），第三次被權限擋下。
- opencode 中止了 5 個 run（各有 `ABORTED`；理由見 REPORT §8）。
- 第二件工作第一次啟動失敗：`20261018-115552-oc-maprunner` exit 1，API 回 400 `"thinking.type.enabled" is not supported for this model. Use "thinking.type.adaptive" and "output_config.effort" to control thinking behavior.`（opencode 1.17.8 內建的 Sonnet 5.5 variant 是舊格式）。在 `work.jsonc` 改成 adaptive 後重送（`oc-maprunner2`）。
- 新程式沒有比舊程式快（FreeInference week13 session 引擎 +18%；負載不同）；輸入快取讓準備從 561.7 s 降到 8.3 s。
**與假設的差異**: v3 的判定照審查幾乎必然 GO（對手只剩 C0、INT2 不對稱）；v4 改成「讀到沒驗證的格式就重算」＋等待時間護欄 5%＋15 個主格（使用者 10/18 同意）。opencode 花費：審查約 US$2.6、模擬器準備約 US$28.6。
