# 第三階段探索：怎麼重跑（指令與設定）

> 隨實驗進度補。每一步都照 CLAUDE.md §4.1 的記錄殼（`m7run`）：`/mlsteam/data/tiara/runs/<run_id>/` 有 cmd.sh、context.txt、stdout、stderr、exit_code。
> 平台：MI300X @ MLSteam（`environment_snapshot.md`）。其他平台要改的地方見最後一節。

## 0. 環境

```bash
cd /mlsteam/workspace/paper-hierarchical-kv-state
source code/m7_env.sh            # venv tiara（torch 2.10+rocm7.0）、HF_HOME、TIARA_RUNS、m7run；卡號＝${TIARA_GPU:-0}（有 /mlsteam/data/tiara/USE_ROCR 時設 ROCR_VISIBLE_DEVICES 並 unset HIP／CUDA，否則設 HIP／CUDA_VISIBLE_DEVICES；行程內一律 cuda:0）
# vLLM 實驗用另一個 venv：/mlsteam/workspace/venv/tiara-v028（vLLM 0.28.0+rocm722）
```

## 1. 只用 CPU 的部分（任何時候都能跑；第二階段計時期間要 `nice -n 19 taskset -c 0-3`，而且**不能 import torch**）

```bash
# 自測
python -I code/m16_s1_lossless.py --self-test          # S1 分析：25 項 ALL PASS
python3 -I code/m16_mcake.py                            # I1 排程核心：ALL PASS
# I1 事先預測〔模擬〕（run 20261019-134046-m16-i1pred）
m7run m16-i1pred nice -n 19 taskset -c 0-3 python -I code/m16_i1_predict.py --model llama \
  --out results/m16_stage3_mi300x/i1_concurrent_restore/prediction
```

## 2. GPU 部分（第三階段 runner）

### 2.1 runner

```bash
# 量時間的佇列 queue/（GPU-NEED: both；在卡 1；第二階段全空時獨佔兩張卡，或 XGPU 條件下用 WANT_CARD1 清空卡 1）
S3_ROLE=timing  setsid nohup bash code/m16_queue/runner.sh > /mlsteam/data/tiara/stage3/jobs/runner_timing.out 2>&1 &
# 品質的佇列 queue_q/（GPU-NEED: one；第二階段 main 空、沒有 WANT_CARD1 時在卡 1；helper 還在跑時 ≤ 96 GB）
S3_ROLE=quality setsid nohup bash code/m16_queue/runner.sh > /mlsteam/data/tiara/stage3/jobs/runner_quality.out 2>&1 &
# 放 job：把 code/m16_queue/jobs/<NN_name>.sh 複製到 /mlsteam/data/tiara/stage3/jobs/<queue>/（名稱照位元組排序執行）
# 停：touch /mlsteam/data/tiara/stage3/jobs/STOP
```

- 和第二階段的協調（RUNLOG_MI300X 15.13；`code/m16_queue/runner.sh` 檔頭）：量時間的 job 在第二階段兩個佇列都空時獨佔兩張卡；有 `XGPU_OK`、第二階段 main 空時改用 XGPU 條件（`WANT_CARD1` → 等卡 1 清空 → `gpu1.lock`，結果標「卡 0 同時有品質工作」，環境變數 `S3_XGPU_COND=1`）。品質 job 在第二階段 main 空時用卡 1。拿 `/mlsteam/data/tiara/gpu{0,1}.lock`；登記共用顯存帳 `vram_reserved.json`；`MAX-H` 超過 10/23 02:00Z 的不開。測試：`S2_JOBS=<假目錄> S3_JOBS=<假目錄> S3_LOCKDIR=<假目錄> bash code/m16_queue/runner.sh`（見 `experiment_log.md`）。
- 卡號變數（RUNLOG_MI300X 15.16）：第二階段的 `44d_xgpu_check2` 通過後會放 `/mlsteam/data/tiara/USE_ROCR`。之後 runner 與 `m16_vllm_bench.py serve` 都改設 `ROCR_VISIBLE_DEVICES`＝實體卡號、unset HIP／CUDA（只用 HIP 時行程會開兩張卡的 render 節點，amd-smi 兩張卡都列，對方的 guard 會當成入侵者）。測試：`M16_USE_ROCR_FLAG=<假檔> …/tiara-v028/bin/python code/tests/m16_bench_cpu.py`（有／沒有旗子都 ALL PASS）。**10/20 更新（RUNLOG_MI300X 15.17）**：ROCR 驗證也沒過（設了 ROCR 的行程一樣開兩張卡的 render 節點），所以 `USE_ROCR` 不會出現，切換程式留著但不會觸發，照 HIP 做法。第二階段改讓 `code/gpu_guard.py` 從 KFD sysfs（`/sys/class/kfd/kfd/proc/<pid>/vram_<gpu_id>`）讀每張卡分開的 VRAM；第三階段的 guard 全走 `gpu_guard`（`GpuWatcher`、`smi_index`、`host_contention`、`foreign_on`，介面沒變），自動用到新版。`ledger.py card_clear` 看的是整張卡的已用量（`amd-smi metric --mem-usage`），不受行程列表的問題影響。
- **10/20 起預設 `S3_AFTER_S2=1`**（使用者：「基本上是要排在第二階段後」）：任何 GPU job 都要第二階段兩個佇列都空、沒有 WANT_BOTH 才開；XGPU 條件與「helper 還在跑時用卡 1 跑品質」關掉。`S3_AFTER_S2=0` 才回到上面的寬鬆規則。假目錄測試 5 種情況都照預期（helper 忙時嚴格模式不開品質、不立 WANT_CARD1；全空時 timing／quality 都開；寬鬆模式 helper 忙時開品質）。
- job 檔頭：`# GPU-NEED: none|one|both`、`# GPU: 0|1`、`# NEED-GB: N`、`# MAX-H: h`。

### 2.1b 10/20 起：單一順序佇列＋cpu 佇列

```bash
# GPU 佇列（照 v0.4 時間表的順序；品質 job 用 CARDS: 0,1 在 job 內兩卡並行）
S3_ROLE=timing setsid nohup bash code/m16_queue/runner.sh > /mlsteam/data/tiara/stage3/jobs/runner_timing.out 2>&1 &
# CPU 分析佇列 queue_c/（nice 19、0–3 號核心；看到 CPU_QUIET 就先不開新的）
S3_ROLE=cpu    setsid nohup bash code/m16_queue/runner.sh > /mlsteam/data/tiara/stage3/jobs/runner_cpu.out 2>&1 &
```
- 檔頭新增 `CARDS: 0,1`（runner 兩張卡各登記 NEED-GB）與 `AFTER: <job 名>`（等那些 job 在任一佇列 .done）。`quality` 角色停用；舊的 `queue_q/` 兩個 job 移到 `retired_20261020/`，改名為 `05_s1_kvdump`（兩卡並行）與 `35_s1_cpubench`（排他、立 CPU_QUIET、綁 8–23 核）。
- 假目錄測試（10/20）：雙卡切片的登記與釋放、依賴（沒依賴的先跑、cpu job 完成後 GPU job 才開）、cpu 佇列收到 GPU job → failed、CPU_QUIET 擋住新 job，都照預期。

### 2.2 job 清單（`code/m16_queue/jobs/`，10/22 收尾版）

實際執行順序、每個 run_id、失敗與修正都在 `experiment_log.md`；下表只列「要重跑時放哪個 job」。共用函式在 `lib.sh`（`oncard`、`nh`＝寫 NEEDS_HUMAN、`lastrun`）、`vllm_lib.sh`（起停 vLLM server＋guard）、`*_common.sh`。名稱照位元組排序執行；`AFTER:` 檔頭決定依賴。

| job | 實驗 | 主要程式 | 怎麼跑／事先寫死的條件 | 輸出 |
|:--|:--|:--|:--|:--|
| `00_smoke.sh`、`02_smoke_x.sh`、`03_smoke_xtime.sh`、`04_smoke_c1a.sh` | 冒煙 | 各 GPU 程式最小設定 | — | `/mlsteam/data/tiara/stage3/smoke/`（不進 results） |
| `05_s1_kvdump.sh`（兩卡並行）→ `35_s1_cpubench.sh`（排他、CPU_QUIET、綁 8–23 核） | S1 無損壓縮 | `m16_s1_kvdump.py`、`m16_s1_lossless.py analyze／breakeven` | `experiment_matrix.md` S1 卡 | `/var/tmp/tiara_s3/kv_dumps/`（可重生）、`results/m16_stage3_mi300x/s1_lossless/` |
| `06_k1_kvstruct.sh` | K1 KV 結構 | `m16_kvstruct.py ropecheck／analyze` | `k1_kvstruct_howto.md` | `results/.../k1_kvstruct/`（大檔見該目錄 `BIG_FILES.md`） |
| （手動，CPU；10/22 補） | K1b Tucker 與跨層共用 | `m16_k1_tucker.py run`（每份 dump 一個行程，8 核）→ `report` | `experiment_log.md`「K1b」（事先寫死 Q1／Q2）；自測 `--self-test` | `results/.../k1_kvstruct/tucker/`（譜在 `/mlsteam/data/tiara/stage3/k1/tucker_spectra_*.npz`） |
| `20_i4_sdma0.sh`、`21_i4_sdma1.sh` | I4 搬法 | `m16_i4_copybench.py` | `experiment_matrix.md` I4 卡 | `results/.../i4_copy_batching/` |
| `30_i1_interf.sh`、`31_i1_grid.sh` | I1／I2 | `m16_i1_multisession.py`（排程核心 `m16_mcake.py`） | `experiment_matrix.md` I1 卡；預測見 §1 | `results/.../i1_concurrent_restore/` |
| `33a_c1a.sh`、`33b_c1a.sh` → `33z_c1a_report.sh` | C1a 預測第 2 輪 block | `m16_c1a_predict.py run／report`（`c1a_common.sh`） | `c1a_s2_howto.md` | `results/.../c1a_predictive/` |
| `36a–c_x1_screen.sh` → `36z_x1_report.sh` | X1 疊加篩選（TF） | `m16_xstack.py calib／screen`（`x_common.sh`、`x_pool.sh`） | `x_stack_howto.md`；`lit/L5d_stacking.md` | `results/.../x_stack/` |
| `37a–f_x2_gen.sh` → `37z`、`37zz_x2_report_final.sh` | X2 生成答對率 | `m16_xstack.py gen／report` | `x_stack_howto.md` | `results/.../x_stack/` |
| `38b1–4_x2b_gen.sh` → `38bz_x2b_report.sh` | X2b 加題（288 題） | 同上（`x_ext.sh`） | `x_stack_howto.md` | `results/.../x_stack/` |
| `38c1–2_xkv_cal.sh` → `38cz`；`38k1–2_xlcal_more.sh` → `38kz` | xKV 事先校準基底（38c、38k） | `m16_xstack.py gen`、`m16_xlcal3.py plan／check-calib／report`（`xlcal3_common.sh`） | `x_stack_howto.md`；38k 主判定只用新的 192 題 | `results/.../x_stack/` |
| `38e1–3_x4a_tf.sh` → `38ez`；`38f1–4_x4b_gen.sh`、`38fy_x4_lossless.sh` → `38fz`；`38g1–2_x4c_128k.sh`；`38h1–2_x4d_turns.sh` → `38hz` | X4 推到極限（配比、極限、128K、多輪） | `m16_x4.py`（`x4_common.sh`；子命令 tf、select-b、dump、gen、lossless、report） | `x4_limit_plan.md`（不改）；`x_stack_howto.md` §15 | `results/.../x_stack/x4/` |
| `38i1_x4c_256k_qwen.sh`（→ `38iz_x4_report.sh`） | X4c Qwen 256K | 同上 | **失敗、沒有資料**（`m10_tasks._cwe` 詞表不夠長）；兩個 job 都已移到 `retired_20261020/`，X4 的正式報告是 `38hz` | — |
| `39_x3_micro.sh` → `39z_x3_map.sh`（cpu 佇列）→ `39v_x3_validate.sh`；`38a_x3_micro2.sh` → `38z_x3_map2.sh` → `38j_x3_validate2.sh` | X3 還原時間、決策圖、端到端驗證 | `m16_xtime.py micro／map／validate`、`m16_xtime_kern.py`（融合 Triton kernel） | `x3_time_howto.md`；validate 只在 `flip_cells.json` 的 `worth_validate` 為真時跑 | `results/.../x3_time/`（map 大檔見 `BIG_FILES.md`） |
| `38da_c1b.sh`、`38db_c1b.sh` → `38dz_c1b_report.sh` | C1b 選擇性還原 | `m16_c1b_restore.py selfcheck／qtime／run／report`（`c1b_common.sh`） | `c1b_howto.md` | `results/.../c1_selective_restore/` |
| `40_v0.sh` | V0 vLLM 冒煙（ROCm 上各設定能不能跑、輸出一不一致） | `m16_vllm_bench.py`（venv `tiara-v028`；`vllm_lib.sh`） | `v0_f1_i3_howto.md` §1；分歧只在 margin ≤ 1e-3 的平手且其他檢查都過才算過（`tie_ok`） | `results/.../v0_vllm_smoke/` |
| `50_f2.sh` | F2 重現 vLLM #44294 | `m16_vllm_bench.py` | `v0_f1_i3_howto.md` §2；檔頭寫死三段 seed→evict→burst 與 (A)(B)(C)(A′) 四種設定 | `results/.../f2_shared_prefix_burst/` |
| `51_f1_08192_*`、`52_f1_32768_*`（18 個） | F1 前綴復用地圖 | `m16_workload.py`、`m16_vllm_bench.py`、`m16_serve_patched.py`（`f1_common.sh`） | `v0_f1_i3_howto.md` §3 | `results/.../f1_prefix_reuse/` |
| （手動，CPU） | F1 報告 | `m7run m16-f1-report python -I code/m16_f1_report.py` | 成功＝有 APC 收益 <5% 或變差的格子 | `f1_prefix_reuse/f1_report.{md,json}`（run `20261022-132645-m16-f1-report`） |
| （手動，CPU） | I3 agent 暫停〔模擬〕＋換基準重讀 | `m16_pause_sim.py`、`m16_i3_reread.py` | `i3_pause_sim.md` | `results/.../i3_pause_sim/` |
| （手動，CPU） | 結果索引 | `python -I code/m16_index.py` | 每列 run_id＋source；沒跑的寫 NOT_MEASURED | `docs/stage3_20261019/results.csv` |
| `90_report_pdf.sh`（或手動） | 完整 PDF 報告 | `m16_report_pdf.py build`（自測：`m16_report_pdf.py self-test`） | `report/README.md`；文字在 `report/narrative/*.html` | `/mlsteam/data/tiara/stage3/report/stage3_report.pdf` |


**10/22 補跑**（設計與事先寫死的判定：`experiment_log.md`「10/22 15:3xZ 起：補跑」）：

| job | 實驗 | 主要程式 | 輸出 |
|:--|:--|:--|:--|
| `61_x4c_256k_qwen_noT5.sh` → `61z_x4_report_256k.sh` | Qwen 256K（只跑 T2–T4，T5 在 256K 補不滿） | `m16_x4.py gen`（續跑 38i1） | `results/.../x_stack/x4/`（x4_report 重出） |
| `62_x4a_wbig.sh` → `62z_x4w_report.sh` | X4a 每 8／7 層一組（TF） | `m16_x4w.py grid／report`＋`m16_x4.py tf --combos code/m16_xstack_combos_x4w.json` | `results/.../x_stack/x4/wbig/` |
| `62b_x4a_wall.sh` → `62bz_x4wall_report.sh` | X4a 全部層一組（TF） | 同上，`--variant wall`、`m16_xstack_combos_x4wall.json` | `results/.../x_stack/x4/wall/` |
| `63_x3_flipgrid.sh` | X3 翻轉格子（算力型背景） | `m16_x3grid.py build／report`＋`m16_xtime.py validate` | `results/.../x3_time/v3grid/`（中間檔在 `/mlsteam/data/tiara/stage3/x3grid/`） |
| `63b_x3_flipgrid_mem.sh` | X3 翻轉格子（頻寬型背景） | 同上＋`M10_BUSY_SCRIPT=code/m16_busy_mem.py` | `results/.../x3_time/v3grid_mem/` |
| `64_i3g.sh` | I3g：agent 暫停（vLLM 端到端，a／b／c／d） | `m16_workload.py gen-agentic`、`m16_vllm_bench.py`、`m16_i3g_report.py` | `results/.../i3g_agentic/` |
| （手動，CPU） | K1b：跨層子空間、Tucker | `m16_k1_tucker.py run／report`（Llama 要分塊 Gram，見失敗 5） | `results/.../k1_kvstruct/tucker/` |
| （手動，CPU） | 38k 同位元組（事後） | `m16_xlcal_eqbytes.py` | `results/.../x_stack/v3/xlcal_eqbytes.*` |
| （手動，CPU） | 交付 md | `python3 code/m16_narr2md.py`（從 `report/narrative/*.html` 轉出 `results_analysis.md`、`stage3_candidates.md`） | `docs/stage3_20261019/` |

撤下沒跑的：`61b_k1b_more_dumps`／`61bz_k1b_more`（使用者「實驗都可以了」；在 `retired_20261020/`）。

S2（`m16_s2_equalbytes.py`）沒有單獨的 GPU job：X1 篩選時順便產生同位元組的子集（`s2part_*`），由 `36z_x1_report.sh` 的 `m16_s2_equalbytes.py report` 出表（`results/.../x_stack/s2_subset/`）。

### 2.3 結果文件

- `results_analysis.md`：各實驗的解讀（報告 PDF 的文字版）。
- `stage3_candidates.md`：第三階段候選題目排名。
- `report/`：PDF 產生器的說明與敘述文字；PDF 本身不進 git（約 2.7 MB）。

## 3. 換平台要改的

- GPU 廠商相依：`code/gpu_guard.py`（amd-smi／nvidia-smi）、`HIP_VISIBLE_DEVICES`↔`CUDA_VISIBLE_DEVICES`；`HSA_ENABLE_SDMA`（I4）只在 ROCm 有意義。
- 限速層頻寬（b16／b32／b48）是**每個模型、每台機器**用 `m10_speed.py calib-f` 量出來的（`results/m10_precision_mi300x/speed_<m>/bw.json`），換機器要重量。
- runner 的第二階段路徑（`S2_JOBS`）只在這台 MI300X 的這次租約有意義；別的機器設成一個空目錄即可。
