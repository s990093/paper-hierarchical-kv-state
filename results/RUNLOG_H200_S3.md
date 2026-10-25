# RUNLOG — 第三階段（docs/stage3_design_20261024/）在 H200（hgx-cn01）

> 計畫：[05_poc_plan_h200.md](../docs/stage3_design_20261024/05_poc_plan_h200.md)、[mechanisms/06_eval_gates.md](../docs/stage3_design_20261024/mechanisms/06_eval_gates.md)；追蹤表：[06_tracker.md](../docs/stage3_design_20261024/06_tracker.md)。
> runner：`code/r_h200_s3.sh`；新程式在 `code/s3/`；結果在 `results/s3_h200/<代號>/`；每個閘的判準登記在 `docs/stage3_design_20261024/runs/`（登記先、結果後附，登記不改）。
> 沒量到的寫 NOT_MEASURED；失敗照實記、停下回報（CLAUDE.md §1）。

---

## S3-0 — 開工前置（2026-10-25，head node）

**狀態**: 進行中
**git**: 從 `ae60a7f`（main）開始；本 worktree `paper-hkv-stage3` 在 main。

**看到的（照實）**
- `/work/ChihChungHsu/hungweilai/hf-cache/hub/` 是空的（目錄 mtime 2026-10-23 03:12 +08）；`/work/ChihChungHsu/hungweilai/models/` 也是空的（noDCA 變體不在）。
  10/22 的 `RUNLOG_H200_ICPE.md` ICPE-0b 還記著 Llama-3.1-8B（14.97 GiB）、Qwen2.5-7B-1M（14.2 GiB）、noDCA 變體都在。誰、為什麼刪：查不到（NOT_VERIFIED）。
  資料集（`datasets/{hotpotqa,longbench,ruler_ref,scbench,traces}`）還在。
- 群組配額 `quota -s -g ChihChungHsu`：1,487G／1,536G（剩 49G）。我們帳號：DL_Lab4 62G、其他約 4G。
- GPU：lab 帳號上限 4 張（和 hao、chihchung 共用），開工時 4／4（我們 3：icpe-cpu、DL_Lab4 final_ens、J_kd；hao 1）。
- head node 的 `/tmp` 清理：`/usr/lib/tmpfiles.d/tmp.conf` 是 `D /tmp 1777 root root 30d`，`systemd-tmpfiles-clean.timer` 每天一次（30 天沒碰的檔案刪掉、開機清空）。compute node 的設定由 fetch job 記錄（`node_tmp_policy`）。
- 從 head node `ssh hgx-cn01`：連線被拒（只能從 job 裡看 compute node）。

**使用者的決定（2026-10-25）**
1. GPU 排程「排滿：stage3 品質先填空檔」：品質 job 切成 ≤4 小時、可續跑的分片，一有空卡就跑；下一個要開跑的 ICPE 量時間 job 加 `afterany:<正在跑的 stage3 分片>`；hold 中的 ICPE job 不動；stage3 量時間的 job 排在 ICPE 量時間鏈之後（memory `stage3-gpu-scheduling`）。
2. 模型權重「全部只放 `/tmp`，由我通知 ICPE 那個 session 自己改路徑」：hgx-cn01 本地 `/tmp/hungweilai-models/hf-cache`（HF_HOME 的版面）；每個 GPU job 開跑前核對戳記，不在就重抓。
3. 要跟老師要配額（使用者自己談）：篩選閘的權重 57 GiB（Llama 14.96＋Qwen-7B 14.19＋Qwen-14B 27.51，HF API 逐檔加總）；主實驗再加 Mistral-NeMo-12B 22.8（只抓 HF 分片）＋DeepSeek-V2-Lite-Chat 29.26；D0 的 MiniCPM4.1-8B 15.25；X5 的 gpt-oss-20b 12.82／Nemotron-Nano-9B-v2 16.56／Qwen3-Next-80B-A3B-FP8 76.42 或 Qwen3.5-35B-A3B 66.97（GiB）。

**新程式**
- `code/s3/fetch_models.py`：固定 revision 下載（Llama `4699cc75…`、Qwen-7B `e28526f7…`、Qwen-14B `620fad32…`），每個檔和 HF 的雜湊比對（LFS：sha256；一般檔：git blob sha1），寫戳記與 `results/s3_h200/env/models_manifest_<run_id>.json`；`check` 用大小＋inode＋mtime 核對戳記。`--self-test` PASS（head node）。
- `code/r_h200_s3.sh`：stage3 的 runner；品質 part 遇到 ICPE 量時間的 job 在跑就 exit 3；量時間 part 遇到我們任何別的 job 就停。
- 觀察：Qwen2.5-7B-1M 和 14B-1M 的 `tokenizer.json` 是同一個 git blob（`443909a6…`）→ 題目可以兩個模型共用（開跑時再用 tokenizer 本身核對一次）。

**送出與排程（2026-10-25 20:1x +08）**
- `env -u TMPDIR -u CLAUDE_CODE_TMPDIR sbatch --job-name=s3-fetch --time=1:00:00 code/r_h200_s3.sh --part fetch` → job **22235**（排隊：MaxGRESPerAccount）。
- 照「排滿」的規則，ICPE 下一個量時間的 job **22233**（p01brr，沒有 hold）的依賴加上 `afterany:22235`（原本的 6 個依賴 21656、21662、21779、21905、21907、21909 照留）。hold 中的 21655、21663、21670 沒動。
- 觀察：ICPE 的 **21669 p11q** 在 2026-10-25 20:01:20 開跑、16 秒後 FAILED：`FileNotFoundError: /work/ChihChungHsu/hungweilai/hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct-1M/snapshots/e28526f7…`（`paper-hierarchical-kv-state/slurm-icpe-p11q-21669.out`）。不是 stage3 的 job，我沒動；已用 SendMessage 通知 ICPE 的 session（h200 [04d6f4]）：模型位置、`fetch_models.py env／check`、noDCA 變體也不在、22233 加了依賴。

---

## S3-A0-0 — A0（＝G-M01/02）登記與冒煙（2026-10-25）

**狀態**: 登記完成；冒煙排隊中
**登記**: `docs/stage3_design_20261024/runs/A0_progressive_restore.md`，`date -u` 2026-10-25T12:23:54Z，commit `000ff40`（在任何 A0 數字之前；模型還在下載）。
**程式**: `code/s3/a0.py`（CPU：題組、配置、位元組帳、生成配置挑法 G1–G4、判定；`--self-test` PASS，含合成資料的 GO／STOP／WEAK）、`code/s3/a0_gpu.py`（GPU：block 為主的 store、只改差異的組 KV、ctx_mass 打分、可插 hook 的逐層 forward；還沒在 GPU 上跑過）、`code/r_h200_s3.sh` 加 `a0-smoke`／`a0-tf`／`a0-gen`。
**計畫檔**: `results/s3_h200/a0/plan_{L128,Q128,Q256,Q14-256,L32,Q32}.json`（每格 186–187 個 TF 配置；seed 和既有題組 0 重疊）。
**和 05／06 的差異**（登記 §5，登記前決定）：K2 家族不當粗底（倍數上限約 10×，做不到 16–32×）→ 改成 KSCAN-2b 挑法；粗底用校準基底（:cal）；加 ONE-PASS、ORACLE-L、BASE-ATTN-L；生成 192＋96＝288 題；32K 只當 TF 對照列。
**送出**: `sbatch --job-name=s3-a0smoke-L128 --time=1:00:00 --dependency=afterany:22235 code/r_h200_s3.sh --part a0-smoke --cell L128` → **22237**（校準 llama → gpu-selftest → 2 題、16 個配置的 TF 冒煙；輸出在 run 目錄，不進 results）。照「排滿」的規則，ICPE 的 22233 依賴改成 afterany:21662,21905,21907,21909,22235,22236,22237（前 6 個是 ICPE session 設的）；已通知 ICPE session。

---

## S3-FETCH — 模型權重下載到 hgx-cn01 /tmp（job 22235）

**狀態**: PASS（exit 0）　**執行時間**: 2026-10-25 20:28:10 → 20:36:3x（+08:00）　**run_id**: `20261025-202810-s3-fetch`
**指令**: `sbatch --job-name=s3-fetch --time=1:00:00 code/r_h200_s3.sh --part fetch`
**產出檔**: `results/s3_h200/env/models_manifest_20261025-202810-s3-fetch.json`（每個檔的名稱、大小、HF 的 sha256／git blob sha1）
**結果**〔實測〕: Llama 14.97 GiB（下載 130 s、13 個檔全對）、Qwen-7B 14.20 GiB（126 s、15 個）、Qwen-14B 27.52 GiB（226 s、19 個）；`check --full` 三個都 OK。
**compute node 的 /tmp**（manifest 的 `node`）: `D /tmp 1777 root root 30d`（和 head node 相同）、`systemd-tmpfiles-clean.timer` 每天；開機時間 2026-06-15；`/tmp` 在根分割區 `/dev/md0`（1.8T，剩 892G）、**掛載選項 `noatime,nodiratime`**。
**因此改了程式**（看到 manifest 之後）：noatime 下讀檔不會更新存取時間，tmpfiles 只看得到 mtime／ctime → 下載後 30 天（約 11/24）會被刪，用不用都一樣。
- `fetch_models.py check` 改成只比大小與 inode（不比 mtime）；新增 `refresh`（touch blob、symlink、戳記、目錄）。
- `r_h200_s3.sh` 的 `ensure_models` 在 check 通過後 refresh；`ensure_calib` 也 touch 校準檔。
- ICPE 的 runner 只呼叫 `check`：改成不比 mtime 之後，refresh 不會讓它誤判。
**失敗與異常**: 無。

---

## S3-C0-v1 — C0 還原方式預測（head node，CPU；2026-10-25）

**狀態**: FAIL（判準 GC-GO 沒過，照實記）
**登記**: `docs/stage3_design_20261024/runs/C0_cost_model.md`（2026-10-25T12:57:12Z，commit `c9c9970`，評估之前）
**run_id**: `20261025-205728-s3-c0-eval-v1`
**指令**: `RUN_ID=… python3 code/s3/c0.py eval`
**產出檔**: `results/s3_h200/c0/{c0_report.json,tables_c0.md,cells.csv}`
**關鍵數字**〔實測〕: 352 格。平均／最差代價：長度 48.9%／542%、負載 55.1%／415%、平台 56.7%／542%、kernel 33.4%／255%；永遠 Cake 6.6–13.0%／60–108%；G5／G5b 8.4–45.1%（有量的格）。
**失敗與異常**: 無程式錯誤。預測器在「忙」的格子選了全部載入（parts 的忙時 H2D 倍數≈1.0，實測全部載入忙時慢約 6 倍）。判讀見登記文件的結果節。
**之後**: ICPE 的 p01brr、P2 收齊後跑 C0-v2（同程式、同判準）。

---

## S3-GM13 — G-M13 搬運和重算賽跑的重播（head node，CPU；MI300X 的 I3g 資料）

**狀態**: PASS（H0；H1 沒過，照實記）
**登記**: `docs/stage3_design_20261024/runs/G_M13_race_sim.md`（commit `b512650`，算之前）
**run_id**: `20261025-210037-s3-gm13`　**指令**: `RUN_ID=… python3 code/s3/gm13.py run`
**產出檔**: `results/s3_h200/gm13/{gm13_report.json,tables_gm13.md}`
**關鍵數字**〔實測資料的重播〕: Llama 回來那一輪 P95：d 105.14 s → H0 1.40 s（門檻 1.5×c＝2.50 s），H1 2.75 s；重算贏 81%／78%。Qwen：NOT_MEASURED（MI300X 沒有 Qwen 的 vLLM 冷 prefill）。
**限制**: 不模擬額外重算搶 GPU（偏樂觀）、不模擬取消讀取讓 NFS 變空（偏悲觀）。

---

## S3-P0e — 環境（P0e；B0 冒煙 job 22238 的第一步，2026-10-25 20:43 +08）

**狀態**: PASS　**run_id**: `20261025-204305-s3-b0-env`　**指令**: `python code/s3/b0.py env --out $RUN_DIR/env.json`
**產出檔**: `results/s3_h200/env/p0e_env_20261025-204305-s3-b0-env.json`
**結果**: H200、132 個 SM；torch 2.13.0+cu130；CUDA stream 優先權範圍 [0, −3]；**green context 可用**（`torch.cuda.green_contexts.GreenContext.create(num_sms=32)`＋一次 matmul 跑通）；MPS 控制程式 `/usr/bin/nvidia-cuda-mps-control` 在。
**對 B0 的影響**: B0 登記 §4 第 4 點：green context 能用 → 另外登記一節才加進共用方式（還沒加）。
