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
- 2026-10-17 18:00 前後，主 session 為了雙卡模式改了 `code/run3090.sh`，那時 X2 的 `20261017-175914-p3x2-ph-mc-strict` 正在用它。bash 是邊跑邊讀腳本，所以那次 run 讀到改到一半的檔，報 `line 111: syntax error near unexpected token 'then'`，沒有寫 exit_code。X2 留了 INTERRUPTED，用同一指令重跑成 `20261017-175941-p3x2-ph-mc-strict2`；其他 agent 的 run 有沒有受影響，NOT_MEASURED（各 agent 回報時要自己查 exit_code）。
- 之後要做的（還沒做；要等沒有任何 run 在用這支殼才改，免得再打斷一次）：把 `run3090.sh` 的主體包成 `main() { ...; }; main "$@"; exit $?`，讓 bash 先讀完整支再執行；同樣檢查 `env_3090.sh`。在那之前，**有 run 在跑時不改這兩支檔**。
