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

---

## S3-NOISE — 速度判準的「雜訊」補上操作定義（使用者要求；2026-10-25）

**改的檔**：`05_poc_plan_h200.md` §2 的速度判準那一格＋新的 §2.1；`runs/README.md` 的「共同規則」與受影響的 run 表；`runs/B0_interference.md`、`runs/W0_write_cost.md`、`runs/G_fio.md` 各加偏離 1；`runs/C0_cost_model.md`、`runs/G_M13_race_sim.md` 各加一行「事後以 2σ 重算的結果（只供參考）」。程式：`code/s3/b0.py`（每列 `n_rep`、`sigma_rel`、`delta_star`、`p95_sigma_rel`、`p95_delta_star`、`bg_solo_sigma_rel`、`bg_delta_star`；判定加 δ*＋bootstrap CI；2σ>5% 的格子列出、不判定）、`bgengine.py`（背景單獨跑量 6 個時間窗）、`w0.py`（三個欄位；GW0 改成 PASS／FAIL／UNDETERMINED／NOISY）、`gfio.py`（每格 6 次＋三個欄位）、`c0.py ref2sigma`、`gm13.py ref2sigma`（只供參考）。
**套新規則（還沒看過時間數字）**：B0＋G-M08、W0、G-fio（只有欄位，沒有判定）、之後的量時間的閘。
**維持舊判準（已看過結果）**：C0-v1（GC-FAIL）、G-M13（GM13-PASS）；C0-v2 也維持（判準已看過）。品質（A0、G-M04）不變。
**參考值**：C0 事後 2σ：平均代價 22.2–31.6%（kernel 切法 0%），每個切法有 2–53 格 2σ>5%；G-M13：c、H0 的 P95 2σ＝10.4%、18.0%（都 >5%），H0 的 CI 上界 1.69 s ≤ 門檻 2.50 s。

---

## S3-QUEUE-1 — 10/25 晚上的登記與送出（使用者 10/25：「排滿」、四個新候選全加）

**新登記**（都在任何數字之前 commit）：G-M04（併在 A0 TF）、G-M11（併在 B0，B0 偏離 2）、M05（併在 A0 L128 生成）、G-M12（品質＋速度）、G-M06＋M07（4 卡）；D0 先做可行性（P0f，job 22270），量測的登記等讀過 MiniCPM 的 modeling 程式再寫。
**冒煙**：A0 L128（22241）PASS；A0 Q256（22252）PASS（自檢過、71.7 GB；每題 prefill 31 s、每個配置約 0.5 s）；B0（22242）PASS（stream／stream_hi／m08 256K／MPS 都跑通、開頭的正確性核對全過）；A0 Q14-256（22253）跑中；B0 第二次（22267，含 dq_vq、6 次）排隊。
**品質 job（排滿，ICPE 的 22233 會等它們；開跑時照讓位規則）**：A0 TF L128 22250（跑中，每題約 74 s〔實測〕）、Q128 22251、Q256 22254／22255、Q14-256 22256–22259（afterok 冒煙）、L32 22260、Q32 22261；G-M12 品質 L128 22268、Q128 22269；D0 可行性 22270。
**量時間的鏈（s3-T-*，afterany ICPE 的 p11t 21670，一個接一個）**：22271 B0 llama b0 stream → 22272 qwen b0 stream → 22273 llama m08 stream → 22274 qwen m08 stream → 22275 llama b0 MPS → 22276 qwen b0 MPS → 22277 llama m08 MPS → 22278 qwen m08 MPS → 22279 W0＋G-fio → 22280 G-M12 速度 → 22281 G-M06（4 卡）。
**排程規則的補強**：stage-3 量時間的 job 開跑時有我們別的 job 在跑 → 重送排在它們之後（不失敗）；品質的 job 讓位給「只剩在等 stage-3」的 s3-T-*（和 ICPE 同一個規則；`code/s3/sched.py`）。ICPE 量時間的 job 若在 p01brr 結束後 hold 中（等 ICPE session 檢查），60 分鐘內也讓位（ICPE session 的要求）。
**觀察**：DL_Lab4 的 21907（J_kd）、21909（final3）在 2026-10-25 21:03:08 被取消（不是這個 session、也不是 ICPE session 取消的）。

---

## S3-NUMA — 量時間的 run 記 NUMA 位置（只記錄；使用者要求，2026-10-25）

**狀態**: 程式改好；在 lab-3090 上驗過程式（**不是 H200**）；H200 上還沒跑過
**為什麼**: `docs/H200_SLURM.md` 第 158 行寫了兩件事：pinned 記憶體要和卡在同一個 NUMA 節點；每個 job 拿到的卡可能不同，要每次記錄。但 stage-3 的程式沒有記：`b0.py env` 只有 torch、SM、stream 優先權、green context、MPS；pinned 配置（`b0.py` 的 `contig_host`、`m7_model.host_chunk`、`m10_speed.Bench`）也沒有綁 NUMA。
**改的檔**: 新增 `code/s3/numa_probe.py`；`code/s3/b0.py`（`env` 加 `numa`；`run` 每列多 `gpu_numa`、`cpuset_numa`、`buf_numa`、`numa_local`，另存 `<名稱>_numa.json`，含 start／bench／end 三個時點）；`code/s3/w0.py`（同樣四欄＋`_numa.json`）；`code/s3/gfio.py`（`_fs.json` 加 `numa`）。runner 沒改。規則寫在 `docs/stage3_design_20261024/runs/README.md` 的「共同規則：NUMA 位置」。
**判準**: 不變。`numa_local` 不進任何判定；`numa_local≠yes` 的 run 先回報，要不要重量由使用者決定（還沒定規則）。
**程式驗證（lab-3090，driver 550.163.01、kernel 6.8.0-138；沒有 run_id，不進 results，不是 H200 的數字）**
- `numa_probe.py --self-test` 在 Mac（只有邏輯；move_pages 照實印 NOT_AVAILABLE）和 lab-3090 上都 PASS。lab-3090 上，一般記憶體與 libcuda `cuMemHostAlloc` 配的 pinned 記憶體，抽樣的每一頁都查得到節點。
- `torch_gpu`：torch 2.6 只有 `uuid`（沒有 `pci_*` 屬性），用 uuid 對 nvidia-smi，GPU 5 → `0000:8b:00.0` → 節點 1，和 sysfs 一致。
- 用假的 Target 走 B0 的寫列路徑（`measure_block`、`probe_numa`、`NP.save`）：CSV 有四欄，sidecar 有 start／bench／end，end 含 `bench_bf`、`bench_lp`。`taskset -c 0-23`（節點 0）＋GPU 5（節點 1）→ `numa_local=no`；`taskset -c 24-47`（節點 1）→ `yes`。
- **看到的現象（lab-3090）**: `b0.py env` 在 `taskset -c 24-47`（節點 1）、GPU 5（節點 1）下，64 MiB pinned 抽樣 64 頁有 **57 頁在節點 0**。當時節點 1 的 MemFree 0.47 GiB（其餘多為 page cache，FilePages 約 78 GiB），`zone_reclaim_mode=0`，所以 kernel 沒有先回收本地 page cache，而是直接配到節點 0。**cpuset 對了，不代表 pinned 就在同一邊**。這就是 probe 另外記每個節點可用記憶體的原因。
**H200 上還不知道（NOT_VERIFIED）**: H200 的 driver 與 kernel 和 lab-3090 不同，move_pages 查不查得到 pinned 頁要看第一個 `b0-prep`／`b0-run` 的輸出；查不到時會寫 NOT_AVAILABLE（附原因），不會當成 yes。
**合併 main 之後補的（同一天）**: main 新增了兩個量時間的 part，也接上記錄：`gm12.py speed`（只記 sidecar，因為計時區間不碰主機緩衝）、`gm06.py run`（每列 `src_numa`＋sidecar）。
**順手發現的 gm06 問題（沒有改，待決定）**: 在 lab-3090 上跑縮小版 `gm06.cmd_run`（GiB 換成 MiB、2 張卡、拓撲用替身；只驗記錄，不是量測）時，`numa_remote` 要求綁節點 0，但頁實際在**節點 1**。用 torch 2.6 單獨重現後確認原因：
- `set_mempolicy` 本身有效：全新配置的 6 MiB 綁節點 0 → 64／64 頁在節點 0。
- 問題在 gm06 的順序：先 `pinned(4 GiB, local)`、`del`，再 `pinned(4 GiB, remote)`。torch 的 pinned 快取把**同一塊**（位址相同）還回來，所以 remote 拿到的還是 local 那塊。
- 先呼叫 `torch._C._host_emptyCache()` 再配 → 正確落在節點 0。

影響：M07 的「NUMA 本地／遠端差」是登記裡的「另報」項目，不進 G-M06 的判準（k＝2 加速 ≥1.6）。照現在的程式跑，遠端那一列量到的會是本地記憶體。G-M06 在 H200 上已排隊（S3-QUEUE-1 的 22281，量時間的鏈最後一個），還沒跑。改 gm06 的量法不在「只記錄」的範圍內，而且是另一個 session 的程式，所以先不改，待使用者決定。新的 `src_numa` 欄就算不修也會把這件事照實記下來。H200（torch 2.13）的快取行為：NOT_VERIFIED。
**失敗與異常**: 無。

---

## S3-BUG-1 — 讓位規則的 bug：品質 job 全部讓位到計時鏈之後（2026-10-25 21:4x–21:5x；已修）

**看到的**: Q256（22254、22255）、Q14-256（22256–22259）、L32（22260）、Q32（22261）、G-M12 品質（22268、22269）、D0（22270）、B0 冒煙 2（22267）開跑後都「讓位」給 s3-T 計時鏈的 22272–22281，重送成 22282–22294（afterany 整條鏈）。沒有量到任何東西（每個 job 都在第一步之前結束，exit 0）。
**原因**: `sched.py s3-timing-target` 把「等的全是 s3-* job」的計時 job 當成準備好了；鏈上的 22272–22281 等的是前一個 s3-T，也算進去了。
**修正**（`fix(s3)` commit）: 只有等的全是 stage-3 **品質** job（或什麼都不等）的計時 job 才算準備好；在等鏈上前一個 s3-T 或 ICPE 的都不算。
**補救**: 重新送出正確的 job：Q256 22296／22297、Q14-256 22298–22301、L32 22302、Q32 22303、G-M12 品質 22304／22305、B0 冒煙 2 22306、D0 22307（都加進 ICPE 22233 的依賴）。錯誤重送的 22282–22294 還在佇列裡（排在整條計時鏈之後）：取消要使用者同意（scancel 被權限擋下）；不取消的話，它們輪到時題目都已做完，會直接結束。
**同時合進來的**: 另一個 session 的 NUMA 紀錄（main `5d0ae0c`：`code/s3/numa_probe.py`，b0／w0／gfio／gm12／gm06 只多記錄欄位，判定不讀）；合併後所有自檢 PASS。

---

## S3-GUARD-1 — GpuWatcher 在 nvidia-smi 逾時後停止監看，卻照樣報 CLEAN（2026-10-25 21:44:46 +08；已修）

**看到的**: hgx-cn01 上四個同時在跑的 run，stderr 都有同一段 traceback：監看執行緒呼叫 `nvidia-smi --query-gpu=index,uuid` **逾時 20 秒** → `gpu_guard.SmiUnavailable` → 執行緒結束。四個 stderr.log 的最後修改時間都是 **21:44:46.44x**（＝節點上的 nvidia-smi 同時卡住）。之後不再取樣；結束時 `__exit__` 補的那一次取樣成功，`*.gpu_guard.json` 寫 `verdict: CLEAN`、`n_dirty_samples: 0`。
- 受影響的 run：`20261025-210503-s3-a0-tf-L128-s0`（job 22250，還在跑）、`20261025-212904-s3-a0-tf-Q128-s0`（job 22251，還在跑）、`20261025-214036-s3-a0-smoke-Q14-256`（job 22253，冒煙、不進 results）、ICPE 的 `20261025-210417-icpe-p11-gen-llama`（job 22236，另一個 checkout `paper-hierarchical-kv-state`；不是我的 run，只通知）。四個都是品質 run。
**影響**: 品質規則（CLAUDE.md §3）：分數保留，時間欄本來就不用。但 22250／22251 在 **21:44:46 以後**寫的列，`own_gpu_intruders` 記的是 0，實際上那段時間**沒人在看**（NOT_OBSERVED），結束時的 gpu_guard.json 也會寫 CLEAN。這兩個 run 的爭用紀錄要讀成：21:44:46 之前有看、之後沒有。A0 的判準只用分數，不受影響；它們的 `t_cfg_s`／`t_prefill_s` 不得用來下任何時間結論（本來就不用）。
**為什麼重要**: 同一個監看也包住 B0 的量時間 run（`b0.py run` 用 `m16_c1a_predict.start_watch`）。在量時間的 run 裡發生的話，被污染的時間會被標成乾淨（規則 7：查不到 ≠ 沒有）。
**修正**（`fix(s3)` commit）: `GpuWatcher._try_sample`：每次 `SmiUnavailable` 都記進 `smi_failures`（時間＋錯誤），照樣繼續取樣；有任何失敗 → `contaminated=True`、`verdict=UNOBSERVED_DURING_RUN`、報告多 `n_smi_failures`／`smi_failures`；`m16_c1a_predict.intruders()` 在有失敗後每列寫 `UNOBSERVED:<外來者數>`（下游的計數把它算成「有外來者」，是保守的一邊）。`code/test_gpu_guard_slurm.py` 加一個測試（10 個全過）。只改本 worktree（main）；正在跑的 22250／22251 已載入舊程式，不受影響。
**還不知道**: nvidia-smi 為什麼卡住、卡了多久：NOT_MEASURED（監看只知道「超過 20 秒」）。

## S3-A0-SMOKE-Q14 — A0 Q14-256 冒煙（job 22253；2026-10-25 21:35–21:50 +08）

**狀態**: PASS（程式）
**run_id**: `20261025-213508-s3-a0-calib-qwen14`、`20261025-213607-s3-a0-selftest-Q14-256`、`20261025-214036-s3-a0-smoke-Q14-256`
**指令**: `sbatch --job-name=s3-a0smoke-Q14-256 ... code/r_h200_s3.sh --part a0-smoke --cell Q14-256`
**看到的**: 校準 qwen14 exit 0（R＝3072）；gpu-selftest 全部 true（`pass: true`；store＝host、seg＝16；mem 107.8 GB）；TF 冒煙 2 題 × 17 個配置跑完、exit 0（每題 prefill 約 87–88 s、一題 122–154 s）。監看在 21:44:46 掛掉（S3-GUARD-1），冒煙的輸出不進 results。分數沒有看（冒煙只驗程式）。

## S3-A0-GENSMOKE — A0 生成路徑的冒煙（送出 2026-10-25 22:0x +08）

**為什麼**: 生成那條路（含第 2 輪的尾段、L128 的 M05 三個配置）還沒在 GPU 上跑過；TF 挑完配置就要送 288 題的生成分片，先驗程式。
**做法**: `--part a0-gensmoke --cell L128`：`a0.py gen --smoke --max-items 2 --shard 0/192` → 單輪 1 題（`T2_front-back_s72000`）＋第 2 輪 1 題（`T2_front-back_s73000`）；配置是固定的 8 個（E0、U|q:int4_kivi、P|B32|1/8|int4|{ORACLE, ORACLE-L, BASE-ATTN, BASE-ATTN-L, ONE-PASS}、D|1/8|int4|ORACLE）＋M05 的 3 個，**不看任何 TF 分數挑**；輸出在 run 目錄，不進 results。
**送出**: job **22308**（加進 ICPE 22233 的依賴）。ICPE 22233 現在只剩在等 stage-3 的 job，照讓位規則，22308 和其他排隊中的品質分片開跑時會讓位到 22233 之後。

## S3-GUARD-2 — 量時間的閘量測期間沒有監看；B0 被污染的 CSV 會留在 results（2026-10-25 22:1x；已補，都還沒跑）

**怎麼發現的**: ICPE session 稽核 S3-GUARD-1 之後提醒：W0、G-fio、G-M12 速度、G-M06 只有 runner 開跑時的獨佔檢查（只看得到我們自己的 job），量測期間沒有 GpuWatcher，違反 CLAUDE.md §3（gpu_guard 必用）。我查的時候另外發現：B0 有監看，但被污染時只在 run 目錄留 CONTAMINATED（訊息還寫「品質實驗」），CSV 照樣在 `results/s3_h200/b0/raw/`，`b0.py report` 也照讀；每列的 `own_gpu_intruders` 是每個區塊量測**之前**取的值。
**修正**: 新增 `code/s3/guard.py`（`Timing`／`status`／`finish`／`evict`／`drop_contaminated`）。五個量時間的 part（b0.py run、w0.py run、gfio.py run、gm12.py speed、gm06.py run〔4 張卡都看〕）都改用它：監看開不起來 → 停；被污染（含監看期間 nvidia-smi 失敗過）→ 結果寫到／搬到 run 目錄的 `<檔名>.CONTAMINATED`＋`CONTAMINATED`，不進 results；四個 report（b0、w0、gm12、gm06）再按 run_id 排除，記在 `excluded_contaminated_runs`。
**驗證（head node，沒有 GPU，不是量測、沒有 run_id 進 results）**: `guard.py --self-test` PASS（乾淨、監看失敗、開跑前有外來者、開頭查不到 → 丟錯、evict、drop_contaminated）；b0／w0／gfio／gm06／gm12 的 `--self-test` 都 PASS；用假的 nvidia-smi 把 `gfio.py run` 在 head node 整條跑兩次（小檔、輸出在 scratchpad）：乾淨 → CSV 在 results、`own_gpu_intruders=0`；第一次取樣時 nvidia-smi 卡 25 秒 → 記 1 次失敗、照樣繼續取樣（共 215 次查詢）、結果改寫到 run 目錄、`UNOBSERVED:0`、verdict `UNOBSERVED_DURING_RUN`。W0／G-M12 速度／G-M06／B0 的新程式碼要等 GPU job 才會第一次跑到（B0 冒煙 2，22306，會先跑到 B0 這一支）。
**登記**: W0（偏離 2）、G-fio（偏離 2）、G-M12（偏離 1）、G-M06（偏離 1）、B0（偏離 3）各補一段「實作，判準不變」；都在任何時間數字之前。

## S3-D0-0 — D0 冒煙（P0f）原本一定會失敗：照 modeling 程式補齊（2026-10-25 22:2x；job 22307 還沒跑）

**怎麼發現的**: 寫 D0 登記前，在 head node 把 `openbmb/MiniCPM4.1-8B` 固定 revision `3a8dfed9…` 的 `modeling_minicpm.py`（sha256 前 16 碼 `cd81d52e8e0d35d0`）、`config.json`、`README.md` 抓到 scratchpad 讀（只讀，不執行）。
**原本的冒煙會在哪裡失敗（四處）**
1. modeling 程式 import 的是 `infllm_v2`（`infllmv2_cuda_impl` repo 裝出來的套件名），冒煙卻 import／包 `infllmv2_cuda_impl` → import 失敗。
2. `flash_attn` 和 `infllm_v2` 包在同一個裸 `try/except: pass` 裡；conda 環境沒有 flash-attn（torch 2.13＋CUDA 13，沒有現成的 wheel）→ 靜默跳過，呼叫時才 NameError。
3. `config.json` 沒有 `sparse_config` → 整個模型走 dense（README：要自己加）；稀疏類別還要求 `attn_implementation == "flash_attention_2"`。
4. 環境的 transformers 是 5.19，modeling 程式照 4.56.1 寫（config 的 `transformers_version`）；5.19 沒有 `transformers.utils.import_utils.is_torch_fx_available` → import 失敗（head node 實測）。
**改法（D0 專用，不動 conda 環境）**
- `code/s3/d0_shim/flash_attn`：轉接層，`flash_attn_varlen_func`／`flash_attn_func` 走 `infllm_v2.infllmv2_attn_varlen_func(topk_idx=None)`（infllm_v2 是 FA2 分出來的；topk_idx=None → 傳給核心的 block mask 是 None）＋`bert_padding` 三個函式（FA < 2.7 的 4 值介面，modeling 程式就是這樣拆的）＋dist-info（版本 `2.6.3+infllmv2shim`，讓 transformers 的 flash_attention_2 檢查過）。冒煙先拿 torch SDPA 對答案（GQA 32/2，causal，正方形與右下對齊各一），對不上就停。
- runner 加 `D0-PYLIBS`：`pip install --no-deps --target /tmp/hungweilai-s3/d0/pylibs transformers==4.56.1 "huggingface_hub>=0.34,<1.0" "tokenizers>=0.22,<=0.23.0"`，裝了什麼記在 log 與 `.pip_list`；冒煙的 PYTHONPATH＝轉接層:pylibs:原本的。
- `d0.py smoke`：`sparse_config` 照 README（kernel 32／stride 16／init 1／block 64／window 2048／topk 64／use_nope false／dense_len 8192；和程式裡 `.get()` 的預設相同）；確認每層都是 `MiniCPMInfLLMv2Attention`；infllm_v2 的呼叫分 dense／sparse 計數；記第一次 sparse 呼叫的 `topk_idx` 形狀（D0 量法要用）；`logits_to_keep=1`。
**head node 驗證（沒有 GPU；不是量測）**: 轉接層的 pad／unpad 來回一致；scratchpad 側裝 transformers 4.56.1＋hub 0.36.2＋tokenizers 0.22.2 後，transformers 的相依檢查通過，`modeling_minicpm.py` import 成功，`flash_attn_*` 綁到轉接層；`is_flash_attn_greater_or_equal_2_10()`＝True（讀 dist-info）。GPU 上的部分（infllm_v2 編譯、SDPA 對答案、稀疏路徑）等 22307。
**和 05 §7 的差異**: 05 寫「HF transformers ≥4.56＋infllmv2_cuda_impl（feature_infer 分支）」。feature_infer 分支已不在，用 main `93cf2ec2`（10/25 記過）；transformers 用 4.56.1；dense 的 attention 由 infllm_v2 的核心算，不是 flash-attn 套件。

## S3-A0-FAIL-1 — 256K 的 T5（cwe）題產生不出來（job 22314，2026-10-25 23:17；停下回報）

**錯誤（完整）**: `m10_tasks.py:697 _cwe → RuntimeError: cwe 詞表不夠長，補不滿`（`a0_gpu._loop` → `make_item(T5, ..., n_blocks=504)`）；run `20261025-231553-s3-a0-tf-Q256-s1`，exit 1；runner 停在 `TF-Q256-1`。
**原因**: T5 的常見詞任務要用詞表把情境填滿；256K（504 個 512-token 塊）時詞表不夠。Q256／Q14-256 的冒煙只做前 2 題（T2、T3），沒有碰到 T5。
**影響**: Q256、Q14-256（判定格）每個分片遇到 T5 題都會停：22313（Q256 分片 0，跑中）、22315–22318（Q14-256，排隊中）。128K／32K 的格子不受影響（L128、Q128 已做完 96/96）。
**待使用者決定**（判準登記的事，不自己改）：(a) 256K 的格子不做 T5（寫成 A0 偏離，判定只用 T2–T4）；(b) 加長 cwe 詞表（題目會和 128K 不同）；(c) 其他。

## S3-A0-FAIL-2 — Q256 分片 0 在第 40 題 GPU 記憶體不足（job 22313，2026-10-26 00:13；已修，重送）

**錯誤（完整）**: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 414.00 MiB. GPU 0 has a total capacity of 139.79 GiB of which 45.31 MiB is free. ... Of the allocated memory 138.89 GiB is allocated by PyTorch, and 185.09 MiB is reserved by PyTorch but unallocated.`（`run_item → question_pass → forward_hooked → ctx_mass`；題 `T4_mid-front_s71019`）；run `20261025-231545-s3-a0-tf-Q256-s0`；前 39 題正常寫進 `results/s3_h200/a0/raw/a0_tf_Q256_s0.csv.gz`（每題 prefill 約 30 s、一題 50–61 s〔實測〕）。
**判斷**: `ctx_mass` 一個分塊最多約 1.5 GiB；失敗時 PyTorch 已經握著 138.9 GiB 的活張量（保留未用只有 185 MiB → 不是碎片化），冒煙時一題是 71.7 GB → 跨題累積。原因：`ItemState.comp = Composer(self)`、`Composer.st = st`（參照循環），`Composer.stores` 握著這題的原值／INT4 緩衝 → `_loop` 的 `del st` 釋放不了，要等 Python 的循環 GC。
**修正**（`fix(s3-a0)`）: `a0_gpu.free_item(st)` 每題結束打斷循環 → `del st` → `gc.collect()` → `empty_cache()`；每題印 `mem_after_item`（確認不再累積）。G-M12 品質（`gm12.py quality`，同一個 ItemState）同樣處理；gm12 裡 guard 的別名改成 `GD`（避免和函式內的 `import a0_gpu as G` 混淆；原本沒有撞到）。只改記憶體管理，算法與輸出不變。L128、Q128（96/96 做完）沒有受影響。
**重送**: Q256 分片 0（從第 40 題接著做）。分片 1（T3＋T5）等 S3-A0-FAIL-1 的決定。

## S3-A0-DEV3 — 使用者決定：256K 不做 T5；取消多餘的舊副本（2026-10-26 00:3x）

- 使用者 10/26：「256K 不做 T5」→ A0 偏離 3（`runs/A0_progressive_restore.md`）：Q256／Q14-256 的 TF 72 題、生成 192（gen1 T2–T4 各 64）＋96＝288。`a0.py --self-test` 新增 4 項都 PASS；seed 重疊全為 0；Q256 TF 兩個分片各 36 題、Q14-256 四個分片各 18 題（原本全是 T5 的分片 3 也有 18 題）。
- 使用者 10/26：「取消」→ `scancel` 22282–22285、22287–22294（S3-BUG-1 錯誤重送的 12 個副本；22286 不是我們的）。
- 重送 Q256 分片 0 → 22356（S3-A0-FAIL-2 的記憶體修正之後）；Q256 分片 1 → 見下一個 commit。

## S3-A0-SEL-128 — L128／Q128 的生成配置（登記 §1.6 G1–G4；2026-10-26 00:32，head node）

**狀態**: 完成
**輸入**: `results/s3_h200/a0/raw/a0_tf_L128_s0.csv.gz`（run `20261025-210503-s3-a0-tf-L128-s0`，job 22250，96／96 題）、`a0_tf_Q128_s0.csv.gz`（run `20261025-212904-s3-a0-tf-Q128-s0`，job 22251，96／96 題）。兩個 run 在 21:44:46 之後的爭用紀錄是「沒人在看」（S3-GUARD-1）；品質 run，分數照用。
**指令**: `python code/s3/a0.py select-gen --cell L128`、`--cell Q128`
**產出**: `results/s3_h200/a0/gen_selection_L128.json`（run_id `20261026-003235-adhoc`，24 個配置；生成時另加 M05 的 3 個）、`gen_selection_Q128.json`（`20261026-003236-adhoc`，23 個配置）。G1 挑到的（粗底, f, 目標）：L128 B16·1/16·int4、B24·1/8·int4、B32·1/8·int4；Q128 B16·1/16·int4、B24·1/8·int4、B24·1/16·int4。
**下一步**: 生成冒煙（22325）PASS 之後送生成分片（每格 288 題）。

## S3-SCHED-2 — 加進 ICPE 依賴的新 s3 job 讓 p02 白等（2026-10-26 00:4x；改 sched.py，ICPE job 沒動）

**看到的**: 00:3x 送 Q256 分片 0／1（22356、22357）時照規則 `add-dep` 加進 21655（p02）、21663、21670 的依賴。當時 p02 已經沒有別的依賴（只等資源）→ 加了之後 p02 只剩在等這兩個 s3 job：它們一拿到卡就會讓位（不會死結），但在那之前 p02 不能排程。其他 10 個 s3 品質 job 00:4x 開跑後都照規則讓位到 21655 之後（22359–22371）。
**想做但被擋下的**: 把 22356／22357 從三個 ICPE job 的依賴拿掉（`scontrol update JobId=21655 Dependency=` 等）→ 被權限擋下（auto mode：Interfere With Workloads），沒有繞過；p02 會在 22356／22357 開跑讓位後排程。
**修正**: `sched.py add-dep` 在目標已經準備好（`yield-target` 會選它）時跳過、印原因；skill §4 補註。

## S3-B0-GATE — B0 量時間前的冒煙閘（2026-10-26 06:2x；判準不變）

**問題**: 照讓位規則，品質 job 讓位給 s3-T 量時間的 job → B0 冒煙 2（22369；含 `dq_vq`、guard 監看、NUMA 紀錄）會被排到整條 s3-T 鏈之後，B0 的量時間 run 會先跑沒冒煙過的程式（Qwen 的 b0 設定也從沒冒煙過）。用 job 依賴當閘不行：讓位時重送會換編號、afterok 就被滿足了。
**改法**（`code/r_h200_s3.sh` 的 `b0-run`）: 每組（模型, 設定）在這份程式（b0.py、bgengine.py、guard.py、numa_probe.py、gpu_guard.py 的 sha256 前 16 碼）上第一次量時間前，先在同一個獨佔的 allocation 裡跑 `b0.py run --modes stream --bgs idle --reps 6`（這個設定的全部原語），輸出在 run 目錄、不進 results、不判定；過了在 `$TIARA_RUNS/.s3_gates/` 留戳記。失敗 → step 停下（規則 2）。在 MPS daemon 開之前跑。

## S3-B0-FAIL-1 — B0 llama b0 量時間（job 22271）在建 decode:64:8192 背景時 OOM（2026-10-26 06:29；已修，重送）

**閘**: `B0-GATE-llama-b0` PASS（run `20261026-062242-s3-b0-gate-llama-b0`：開頭的正確性核對全為 true，含 `dq_vq_equal_lookup`；126 列；`own_gpu_intruders` 全 0；`numa_local=yes`〔GPU 0、cpuset 節點 0、pinned 緩衝 128 頁抽樣都在節點 0〕——H200 上第一次查到 pinned 頁的位置，move_pages 可用；目標 session 的 GPU 峰值 60.5 GiB）。
**錯誤（完整）**: `bgengine.py:47 ModelStep.__init__ → torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 64.12 GiB. GPU 0 has a total capacity of 139.79 GiB of which 16.00 GiB is free. Including non-PyTorch memory, this process has 123.76 GiB memory in use. Of the allocated memory 120.46 GiB is allocated by PyTorch, with 22.00 MiB allocated in private pools (e.g., CUDA Graphs), and 2.56 GiB is reserved by PyTorch but unallocated.`；run `20261026-062548-s3-b0-llama-b0-stream-stream_hi`；runner 停在 `B0-RUN`。
**原因**: `decode:64:8192`＝64 條 × 8K 的分頁 KV（Llama 64 GiB）。上一個背景沒有釋放：`rate_fn` 的預設參數 `bg=bg` 握著它，`finally` 裡的 `del bg` 不會釋放 → 新背景建好前，舊背景的 KV 還在。冒煙只用過 decode:8 與 cprefill（小），閘只跑 idle，所以沒碰到。
**修正**（`fix(s3-b0)`）: 每個背景結束時 `rate_fn = None`、`del bg`、`gc.collect()`、`empty_cache()`，並印釋放後的已配置量。釋放乾淨時：目標 60.5＋背景 64 ≈ 125 GiB < 139.8。
**部分輸出**: 這個 run 在失敗前寫進 `results/s3_h200/b0/raw/` 的 CSV、gpu_guard.json、_numa.json 搬到 run 目錄（`*.FAILED_RUN`），不進 results；沒有看任何時間數字。整個 run 重量。
**重送**: llama b0 stream,stream_hi 排到量時間鏈最後（afterany:22281）。下一個 22272（qwen b0）開跑時會用修正後的程式（另跑 qwen／b0 的閘）。

## S3-B0-RUN-qwen-b0 — B0 qwen b0（stream, stream_hi）量完（job 22272，2026-10-26 07:05–08:0x）

**狀態**: 完成（只記執行狀態；B0 的判定等全部 8 個 run 做完、`b0.py report` 之後）
**run_id**: 閘 `B0-GATE-qwen-b0`（07:05–07:08 PASS）；量測 `20261026-070828-s3-b0-qwen-b0-stream-stream_hi`
**指令**: `sbatch --job-name=s3-T-b0-qwen-b0-stream --time=3:00:00 --dependency=afterany:22271 code/r_h200_s3.sh --part b0-run --model qwen --setup b0 --modes stream,stream_hi`
**產出**: `results/s3_h200/b0/raw/b0_qwen_b0_stream-stream_hi_20261026-070828-s3-b0-qwen-b0-stream-stream_hi.{csv,gpu_guard.json,_numa.json}`
**完整性**: 2016 列（8 背景 × 2 模式 × 21 原語 × 6 次）；guard `CLEAN`（`n_smi_failures` 0）、每列 `own_gpu_intruders` 0；`numa_local` 全 `yes`；`host_level` 全 `UNOBSERVABLE_HOST`（Slurm cgroup，照規則照實記）；每列都有 `n_rep`＝6、`sigma_rel`、`delta_star`。背景釋放後 GPU 已配置量維持約 40 GiB（S3-B0-FAIL-1 的修正有效）。時間數字還沒看。

## S3-B0-RUN-m08-stream — G-M08 的 llama／qwen（stream, stream_hi）量完（jobs 22273、22274，2026-10-26 08:03–09:0x）

**狀態**: 完成（只記執行狀態；判定等 B0 全部做完）
**run_id**: llama：閘 `B0-GATE-llama-m08` PASS → 量測 `20261026-080412-s3-b0-llama-m08-stream-stream_hi`；qwen：閘 `B0-GATE-qwen-m08` PASS → `20261026-084740-s3-b0-qwen-m08-stream-stream_hi`
**產出**: `results/s3_h200/b0/raw/b0_{llama,qwen}_m08_stream-stream_hi_<run_id>.{csv,gpu_guard.json,_numa.json}`
**完整性**: llama 216 列（3 背景 × 2 模式 × 6 原語 × 6 次）、qwen 288 列（4 × 2 × 6 × 6）；兩個 guard 都 `CLEAN`（smi 失敗 0）、`own_gpu_intruders` 全 0、`numa_local` 全 `yes`、`n_rep`＝6、`sigma_rel`／`delta_star` 都有值。時間數字還沒看。

## S3-B0-FAIL-2 — B0 llama b0 MPS（job 22275）在 MPS 下 illegal memory access（2026-10-26 09:07；停下、診斷中）

**閘**: `B0-GATE-llama-b0` PASS（b0.py 改過 → 新的雜湊，重跑一次）。MPS daemon 開了（`mps_log_22275`）。
**錯誤（完整）**: `terminate called after throwing an instance of 'c10::AcceleratorError' what(): CUDA error: an illegal memory access was encountered`（在 `CachingHostAllocator` free 時才報出來，非同步）→ Aborted（core dumped），exit 134；run `20261026-090451-s3-b0-llama-b0-mps-mps87-mps75-mps50`；runner 停在 `B0-RUN`。
**MPS server log**: 09:07:45.706 背景 process（PID 2115399）先退出 → 09:07:46.023 `Client {PID: 2115041 …} encountered a fatal GPU error`（主程式），server 拆掉這個 client、重設 device 0。
**做到哪裡**: 第一個區塊 `decode:64:8192 × mps` 做完 21 個原語中的 17 個（102 列）；照同一個種子重算的順序，第 18 個是 **`lr_recon`**。`lr_recon` 在 stream 模式下（22272、B0 冒煙）都正常 → 只在 MPS 下出事；這個區塊的背景另一個 process 就用 64 GiB，加上主程式約 60 GiB，整張卡幾乎全滿，也可能有關。原因還不知道（NOT_VERIFIED）。
**處理**: 部分輸出（CSV／guard／numa）搬到 run 目錄（`*.FAILED_RUN`），不進 results，沒有看時間數字；`scontrol hold 22276`（qwen b0 MPS，同樣會跑 lr_recon）；22277／22278（m08 MPS）照鏈排在它後面；W0＋G-fio（22279）的依賴拿掉，先跑。
**診斷**: 新 part `b0-diag`（量時間類、獨佔、輸出只在 run 目錄）：MPS 下只跑 `lr_recon`、`recompute1`，`CUDA_LAUNCH_BLOCKING=1`，背景 decode:8:8192（小）與 decode:64:8192（大）各一次、各記 exit code。

## S3-W0 — W0 寫入端成本（job 22279，2026-10-26 10:28–10:33）：GW0-FAIL；報告那一步的 bug

**狀態**: W0 量完、判定 **GW0-FAIL（llama、qwen）**；G-fio 沒跑到（runner 停在 W0-REPORT）。
**run_id**: `20261026-102851-s3-w0-llama`、`20261026-103153-s3-w0-qwen`（guard 都 `CLEAN`、`numa_local` yes、各 54 列）；報告 `20261026-103416-adhoc`（head node）。
**指令**: `sbatch --job-name=s3-T-w0gfio ... code/r_h200_s3.sh --part w0gfio`；報告 `python code/s3/w0.py report`
**產出**: `results/s3_h200/w0/raw/w0_{llama,qwen}_<run_id>.{csv,gpu_guard.json,_numa.json}`、`results/s3_h200/w0/w0_report.json`
**關鍵數字（實測）**: CPU 16 核壓縮速度 Llama B16 0.283、B32 0.298 GiB/s（需求 3.27）；Qwen 0.529、0.601 GiB/s（需求 1.26）。GPU 側每請求 31–124 ms，期間背景吞吐比 0.13–0.44。輸出位元組和 A0 的 predict_bytes 逐一相同。細節在 `runs/W0_write_cost.md` 的結果節。
**失敗與異常**: `W0-REPORT` 失敗（exit 1）：`UnboundLocalError: cannot access local variable 'G'`——`w0.py cmd_report` 後面有 `G = np.asarray(g)`，把 guard 的別名 `G` 變成區域變數（S3-GUARD-2 引進的，我的錯）。量測不受影響。修正：五支程式裡 guard 的別名都改成 `GD`；自檢都 PASS；報告在 head node 重跑成功。G-fio 重送（`--from GFIO`）。

## S3-B0-FAIL-2（續）— 原因：MPS 下的容量；B0 偏離 4（2026-10-26 10:5x）

**診斷結果**: job 22389（run `20261026-104206-s3-b0-diag` 等，MPS、`CUDA_LAUNCH_BLOCKING=1`、只跑 lr_recon／recompute1）decode:8:8192 與 decode:64:8192 都 `DIAG_EXIT=0` → 沒有重現。回頭看 22275 的完整 stderr：第一個錯誤其實是主程式的 **OOM**（`lr_recon → xl() → torch.cat`：`Tried to allocate 512.00 MiB … Process 2115101 has 60.00 MiB … this process has 59.27 GiB … Process 2115399 has 80.12 GiB`），illegal memory access 是之後冒出來的。背景 process 80.12 GiB＝自己的權重（約 15）＋分頁 KV 64.12 GiB（`step_bytes` 算出 64.12，和 OOM 訊息相同）。
**處理**: B0 偏離 4（`runs/B0_interference.md`）：MPS 區塊開背景前檢查容量，放不下 → NOT_MEASURED（附原因），不讓整個 run 崩。`b0.py --self-test` 加一項（report 遇到 NOT_MEASURED 的列）PASS。b0.py／bgengine.py 改了 → 下一個 B0 job 會重跑閘。
**排程**: 放行 22276（qwen b0 MPS）；llama b0 MPS 重送到鏈尾。

## S3-GFIO — G-fio（job 22417，2026-10-26 10:44）＋ G-M12 速度（job 22280，10:39–10:41）

- **G-fio**：run `20261026-104426-s3-gfio`，36 列，guard `CLEAN`。compute node 上 /tmp＝md0 ext4、/work＝md1 ext4（本地，不是 NFS）。本地 NVMe 常數（/tmp 64 MiB 循序 8 執行緒）＝**8.07 GiB/s**（σ 0.018）；NFS 的列照 01 §3 繼續用限速層。4 個格子 2σ > 5%（不是判定用的那格）。細節：`runs/G_fio.md` 的結果節。
- **G-M12 速度**：runs `20261026-103958-s3-gm12-s-llama`、`20261026-104031-s3-gm12-s-qwen`，各 15 列（5 個位置 × dense／sparse0.1／sparse0.2，每點 6 次），guard 都 `CLEAN`。判定等 G-M12 的品質那一半（gm12q 排在量時間鏈之後）。時間數字還沒看。

## S3-B0-FAIL-3 — qwen b0 MPS（job 22276）：終止背景 process 之後主程式 fatal GPU error（2026-10-26 10:59；已改、重送）

**看到的**: 容量檢查正常（主程式峰值 40.1 GiB、背景另載權重 14.2 GiB）；`decode:128:8192` × mps／mps87／mps75 三個區塊做完（各 126 列），mps75 的最後一列 10:59:15 → 主程式 `proc.terminate()` 終止背景（MPS log：背景 PID 2214985 在 10:59:15.868 退出）→ 10:59:16.060 `Client {PID: 2205735 …} encountered a fatal GPU error`（主程式），在下一個區塊開頭的 `torch.cuda.mem_get_info()` 報出 `CUDA error: an illegal memory access was encountered` → abort（exit 134）。run `20261026-105034-s3-b0-qwen-b0-mps-mps87-mps75-mps50`。前面沒有 OOM。
**和 22275 一樣的模式**: 22275 也是「背景退出 → 0.3 s 後主程式 fatal」（那次是主程式先 OOM、finally 終止背景）。兩次都在 SIGTERM 直接殺掉背景 process 之後；之前好幾次終止都沒事 → 偶發。
**判斷（NOT_VERIFIED）**: 背景 process 在 CUDA graph 還在 GPU 上跑時被殺，MPS 下會連累同一張卡上的另一個 client。
**改法**: `cmd_bgproc` 收到 SIGTERM 只設旗標 → 跳出迴圈 → `bg.stop()`（等在途的步驟）→ `synchronize` → 自己結束（印 STOPPED）；主程式終止背景前先 `synchronize`、`wait` 放寬到 120 s。量法不變。
**部分輸出**: 搬到 run 目錄（`*.FAILED_RUN`），不進 results，沒有看時間數字。**重送**: qwen b0 MPS 排到鏈尾。22277（llama m08 MPS）已用舊程式在跑；22278 之後的 MPS job 用新程式。

## S3-B0-RUN-m08-mps — G-M08 的 llama／qwen（mps, mps87, mps75）量完（jobs 22277、22278，2026-10-26 11:01–11:2x）

**run_id**: llama `20261026-110117-s3-b0-llama-m08-mps-mps87-mps75`（22277，終止背景用的還是舊做法，9 次都沒出事）；qwen `20261026-111118-s3-b0-qwen-m08-mps-mps87-mps75`（22278，第一個用優雅停止的 run）。
**完整性**: llama 324 列（3 背景 × 3 模式 × 6 原語 × 6 次），全部放得下；qwen 342 列：9 格量到（各 36 列），**`decode:6:258048` × mps／mps87／mps75 是 NOT_MEASURED**（B0 偏離 4 的容量檢查：`MPS 背景放不下：背景 process 約 102.9 GiB（權重 14.2＋背景 82.7＋餘量 6）＋主程式還會長 1.4 GiB > 可用 101.1 GiB`）。兩個 guard 都 `CLEAN`、`own_gpu_intruders` 全 0、量到的列 `numa_local` 全 yes（NOT_MEASURED 的列沒有 NUMA 欄位）。主程式峰值：llama 38.3 GiB、qwen 31.5 GiB。時間數字還沒看。

## S3-SCHED-3 — G-M06（4 卡）要整個 lab 帳號：先 hold（2026-10-26 11:2x）

**看到的**: 22281（G-M06，`--gres=gpu:4`）排隊原因 `AssocGrpCpuLimit`：這個分區每張卡固定 16 核 → 4 卡＝64 核＝lab 帳號的 CPU 上限（和 hao、chihchung 共用）→ 只有整個帳號都空著時才跑得起來。當時 llama b0 stream 重量（22384）和 13 個品質 job（22418–22430，讓位重送的）都用 afterany 排在它後面；照讓位規則，品質 job 還會一直讓位給這個「準備好但拿不到資源」的 job → 全部停擺。
**處理**: `scontrol hold 22281`（讓位規則本來就跳過 hold 中的 job）；22384、22418–22430 的依賴拿掉（都是 stage-3 自己的 job）。G-M06 什麼時候跑，待使用者決定（要整個帳號空檔）。

## S3-B0-RUN-llama-b0-stream — llama b0（stream, stream_hi）重量完成（job 22384，2026-10-26 11:4x–12:3x）

**run_id**: `20261026-115242-s3-b0-llama-b0-stream-stream_hi`（閘 llama／b0 先 PASS）。**完整性**: 2016 列（8 背景 × 2 模式 × 21 原語 × 6 次），沒有 NOT_MEASURED；guard `CLEAN`、`own_gpu_intruders` 0、`numa_local` yes、`n_rep` 6；背景釋放後 GPU 已配置約 57 GiB（S3-B0-FAIL-1 的修正有效）。B0／G-M08 的 stream 部分（llama、qwen × b0、m08）全部量完。時間數字還沒看。

## S3-B0-FAIL-4 — llama b0 MPS（job 22431）：25 格被誤判成「放不下」（2026-10-26 12:48–13:12；重量）

**看到的**: run `20261026-124828-s3-b0-llama-b0-mps-mps87-mps75-mps50` 正常結束（exit 0、guard `CLEAN`），但 1512 列裡只有 matmul ×4、decode:8:8192 ×3、idle × mps 量到；其餘 25 格寫 NOT_MEASURED，而且「可用」一路變少：decode:64 時 61.3 GiB → decode:8／cprefill／hbm 時 24.9 → idle／mixed／decode:32 時 19.6 GiB；同時「主程式還會長」只有 0.1–0.4 GiB（主程式配置的量沒變）。連 idle（背景只要權重＋餘量 21 GiB）都被判放不下。
**原因**: MPS 模式每個區塊結束時只終止背景、沒有 `empty_cache()`（stream 模式有）→ 主程式 PyTorch 快取保留的記憶體一路累積，`cudaMemGetInfo` 把它算成已用；背景 process 也真的拿不到那些記憶體。`decode:64:8192` 那一格清了也放不下（S3-B0-FAIL-2）。
**修正**: 檢查前先 `empty_cache()`（`ef6f1b4`，B0 偏離 4 的補充）；qwen b0 MPS（22432）已用這一版。
**處理**: 這個 run 的 CSV／guard／numa 搬到 run 目錄（`*.SUPERSEDED`），不進 results（量到的格子本身沒問題，但和重量混在一起 report 會把兩個 run 的 rep 合在一個格子裡）；沒有看時間數字。llama b0 MPS 重送（afterany:22432）。
- **同一個問題也影響 qwen m08 MPS（22278）**：`decode:6:258048` 被判放不下時記的「可用 101.1 GiB」也是沒清快取量的；主程式峰值 31.5 GiB → 清了大約有 107 GiB，比要的 104.3 多，很可能其實放得下。這個 run 也搬出 results（`*.SUPERSEDED`）、整個重量（afterany 新的 llama b0 MPS）。llama m08 MPS（22277）每格都放得下、沒有 NOT_MEASURED，保留。

## S3-KVS-F0 — F0 跟隨者模擬：尺準 PASS；決定性實驗 CONTINUE〔模擬〕（2026-10-26 14:00–14:01，head node）

**狀態**: F0-CAL PASS；決定性實驗 **CONTINUE**；全網格**沒跑**（使用者指示先回報）。
**run_id**: `20261026-140045-s3-kvs-f0-cal`、`20261026-140113-s3-kvs-f0-dec`、報告 `20261026-140130-kvs-f0-adhoc`
**指令**:
```bash
python3 code/s3/kvs_f0.py --self-test                 # 59 項 ALL PASS
source code/env_h200.sh; export TIARA_ALLOW_NO_SLURM=1  # head node：tiara.sbatch 預設拒絕 Slurm 外的數字指令；使用者指示在 head node 跑
m7run s3-kvs-f0-cal python3 code/s3/kvs_f0.py cal
m7run s3-kvs-f0-dec python3 code/s3/kvs_f0.py run --slice decisive
python3 code/s3/kvs_f0.py report
```
**產出**: `results/s3_h200/kvs_f0/{f0_cal.json, f0_decisive_h200_llama.csv, f0_verdict.json}`；每個請求的明細 `$TIARA_RUNS/20261026-140113-s3-kvs-f0-dec/f0_decisive_requests.json.gz`
**關鍵數字〔模擬〕**: 尺準 16／16 個（面板, tier）比值在 [0.9, 1.1]（0.933–0.989）。12 格最好固定規則：P5×7、P1＝P4×3、P0×1、P4×1（不全同）；P6 對它的改進最大 +0.21%、全部 <5%、最差 −167.5%（nfs 128K ρ_o＝0.8，那一格整體 P95 被別人的工作主導，P0 的共享請求 P95 749.6 s）。P6 對 P4：−99.2%～+87.1%；對 P5：0.0%～+40.0%。細節：`docs/kv_sharing_hiding_20261026/runs/F0_follower_sim.md` 結果節。
**失敗與異常**: `m7run` 第一次在 head node 被 `tiara.sbatch` 擋下（`不在 Slurm job 裡`），改設 `TIARA_ALLOW_NO_SLURM=1` 後照常建 run 目錄；不影響數字。

## S3-KVS-T0 — T0 Mooncake trace 統計：單位檢查 PASS（2026-10-26 14:03，head node）

**狀態**: 完成（只報告；單位檢查 PASS）
**run_id**: `20261026-140345-s3-kvs-t0`
**指令**:
```bash
source code/env_h200.sh; export TIARA_ALLOW_NO_SLURM=1
python3 code/s3/kvs_t0.py --self-test          # 15 項 ALL PASS
m7run s3-kvs-t0 python3 code/s3/kvs_t0.py run
```
**產出**: `results/s3_h200/kvs_t0/t0_summary.csv`（1,546 列）；`$TIARA_RUNS/20261026-140345-s3-kvs-t0/t0_groups_*.json.gz`
**關鍵數字（實測，trace 統計）**: 單位檢查 toolagent S＝6K w＝±1 s **0.3893**（之後 1 s：0.3400）∈ [0.19, 0.76]；conversation 同設定 0.0000；T0-c 跟隨者比例 toolagent S＝6K 0.340（cpu、nvme）／0.340（nfs），conversation ≤0.0007；T0-d λ·W 中位數全部 ≤0.094。toolagent 群內相鄰間隔 p50＝0.0 s（跟隨者大多和領頭同一個時間戳）。細節：`runs/T0_arrivals.md` 結果節。
**失敗與異常**: 無（`m7run` 在 head node 用 `TIARA_ALLOW_NO_SLURM=1`，同 S3-KVS-F0）。

## S3-KVS-S0 — S0-2 兩人聯集 u(2)：L128 送出（2026-10-26 14:0x；排隊中）

**狀態**: 排隊中（job **22455**）；Q128、Q256、Q14-256 照使用者指示**先不送**。
**指令**:
```bash
python3 code/s3/kvs_s0.py --self-test                      # 22 項 ALL PASS（head node）
bash code/r_h200_s3.sh --part kvs-s0 --cell L128 --shard 0/1 --dry-run   # MODELS → KVS-S0-SMOKE-L128（1 題冒煙，輸出在 run 目錄）→ KVS-S0-L128-0
env -u TMPDIR -u CLAUDE_CODE_TMPDIR sbatch --job-name=s3-kvs-s0-L128-0 --time=4:00:00 code/r_h200_s3.sh --part kvs-s0 --cell L128 --shard 0/1   # → 22455
for t in $(python3 code/s3/sched.py next-icpe); do python3 code/s3/sched.py add-dep --to $t --job 22455; done   # next-icpe 是空的（ICPE 沒有量時間的 job）→ 沒加
```
**排程**: 品質型（TIMING＝0），照讓位規則會排在 stage-3 剩下的 B0 MPS 量時間 job（22453、22454）之後。跑完之後：`python3 code/s3/kvs_s0.py report` → `results/s3_h200/kvs_s0/s0_verdict.json`，判準：L128 的 ρ＝1/4 `u2_global`、`u2_layer` 中位數**都** > 0.85 → STOP-S，否則 CONTINUE。

## S3-B0-RUN-qwen-b0-mps — qwen b0（mps, mps87, mps75, mps50）量完（job 22432，2026-10-26 13:15–14:4x）

**run_id**: `20261026-131551-s3-b0-qwen-b0-mps-mps87-mps75-mps50`（閘 qwen／b0 先 PASS；第一個同時有「優雅停止背景」＋「檢查前清快取」的 B0 run）。**完整性**: 4032 列（8 背景 × 4 模式 × 21 原語 × 6 次），全部放得下、沒有 NOT_MEASURED；guard `CLEAN`、`own_gpu_intruders` 0、`numa_local` yes。終止背景 31 次都沒有 fatal GPU error（S3-B0-FAIL-3 的修正有效）。時間數字還沒看。
**B0 剩下**: llama b0 MPS（22453）、qwen m08 MPS（22454）；之後 `b0.py report`。

## S3-B0-RUN-llama-b0-mps — llama b0 MPS 重量完成（job 22453，2026-10-26 14:49–16:1x）

**run_id**: `20261026-144940-s3-b0-llama-b0-mps-mps87-mps75-mps50`。**完整性**: 3612 列＝28 個區塊量到（各 126 列）＋`decode:64:8192` × mps／mps87／mps75／mps50 照 B0 偏離 4 寫 NOT_MEASURED（84 列；`背景 process 約 85.1 GiB（權重 15.0＋背景 64.1＋餘量 6）＋主程式還會長 0.4 GiB > 可用 82.3 GiB`——清了快取之後的可用量，和 S3-B0-FAIL-4 估的 ≈82 相符）；guard `CLEAN`、`own_gpu_intruders` 0、量到的列 `numa_local` yes。時間數字還沒看。

## S3-B0 — B0＋G-M08＋G-M11 的判定（2026-10-26 16:2x，head node 報告）

**狀態**: 8 個量時間 run 全部完成（guard 都 CLEAN）；報告 run_id `20261026-162611-adhoc`。
**指令**: `source code/env_h200.sh; python code/s3/b0.py report` → `results/s3_h200/b0/b0_report.json`
**判定**: **GB0-FLIP 不成立**（2 條線因太晃未判定）；**GB0-SHARE 成立**（18 格；最強 llama cprefill × e2e_load_int4_b32 × mps87 比 stream 快 4.28×、背景比 0.980）；**GB0-STATIC 不成立**（只有 dq_vq 不是固定等級，20／22 是）；**G-M08 STOP**（45 個候選全不過，背景吞吐比 0.33–0.65）；**G-M11 NOT_DETERMINED**（閒時 dq_int4 σ_rel 0.124／0.086）。讀法：B1 只做「還原給多少 SM」。379 格 2σ > 5% 只回報、不判定；84 列 NOT_MEASURED（llama decode:64 × MPS，容量）。細節：`runs/B0_interference.md`、`runs/G_M11_vq.md` 的結果節。

## S3-SCHED-4 — 使用者 10/26「都做掉」：G-M06 放行；讓位規則不讓給卡在帳號上限的量時間 job

- `scontrol release 22281`（G-M06）：排隊原因 `AssocGrpCpuLimit`（要 64 核＝整個 lab 帳號；當時 chihchung 1 個 job 在跑、17 個在排）。
- `sched.py s3-timing-target`：量時間 job 的排隊原因是 `Assoc*`（卡在帳號上限）→ 品質 job **不讓位**給它（讓了也只是一起等別的使用者；它要等帳號整個空出來才排得上，那時本來就不會和我們的 job 重疊）。自檢 PASS。
- G-M11 重量：runner part `gm11-redo`（G-M11 偏離 1）；F0 全網格：head node 背景跑（`--jobs 8`）；S0-2 其他 3 格：等 L128 的判定（登記：L128 STOP-S → 其他 3 格不跑）。

## S3-KVS-F0（續）— F0 全網格：問人（ASK）〔模擬〕（2026-10-26 16:36–16:39，head node）

**run_id**: `20261026-163625-s3-kvs-f0-full`；報告 `20261026-163950-kvs-f0-adhoc`
**指令**: `source code/env_h200.sh; export TIARA_ALLOW_NO_SLURM=1; m7run s3-kvs-f0-full nice -n 10 python3 code/s3/kvs_f0.py run --slice full --jobs 8; python3 code/s3/kvs_f0.py report`
**產出**: `results/s3_h200/kvs_f0/f0_full_{h200,mi300x}_{llama,qwen}.csv.gz`、`f0_verdict.json`
**關鍵數字〔模擬〕**: R*＝P5（兩個平台）；Llama：gain > 15% 的格 H200 36（11.1%）、MI300X 38（11.7%）→ (i) 過；gain < −5% 的格 1／4（最差 −19.4%／−20.5%）→ (ii) 不過；(iii) 過 → **ASK**。Qwen 重現同樣的樣子。細節：`runs/F0_follower_sim.md`。
**失敗與異常**: 全網格的 CSV 每個約 2.2 MB（> 1 MB）→ 改寫成 .csv.gz（`fix(s3-kvs-f0)`：run 寫 .gz、report 兩種都讀）；壓縮後重跑 report 判定相同。

## S3-GM11-REDO — G-M11 四格重量（job 22478，2026-10-26 18:05–18:08）：仍 NOT_DETERMINED

**run_id**: `20261026-180520-s3-gm11-llama`、`20261026-180657-s3-gm11-qwen`；報告 `20261026-180825-adhoc`
**指令**: `sbatch --job-name=s3-T-gm11redo --time=1:00:00 code/r_h200_s3.sh --part gm11-redo`
**產出**: `results/s3_h200/b0/gm11/*.csv`（各 48 列）、`results/s3_h200/b0/gm11_report.json`
**關鍵數字（實測）**: 閒時 dq_int4 σ_rel llama 0.038、qwen 0.102（2σ > 5% → 不判定）；其他三格 σ_rel ≤ 0.003。guard 都 CLEAN。
**判定**: G-M11 **NOT_DETERMINED**（照偏離 1 事先約定）。閒時 INT4 kernel 在兩次 run 都不穩、有背景時穩；時脈切換的解釋 NOT_VERIFIED。後續選項在 `runs/G_M11_vq.md`。

## S3-KVS-S0（續）— L128 判定 CONTINUE；送出另外 3 格（2026-10-26 18:3x）

**run_id**: 冒煙 `20261026-180827-s3-kvs-s0-smoke-L128`；正式 `20261026-180852-s3-kvs-s0-L128-s0`（job 22497＝22455 讓位重送；96／96 題，guard CLEAN）；報告 `20261026-183013-adhoc`
**指令**: `python3 code/s3/kvs_s0.py report` → `results/s3_h200/kvs_s0/s0_verdict.json`
**關鍵數字（實測）**: ρ＝1/4 中位數 u2_global **0.540**、u2_layer **0.543** → 不是 STOP-S（> 0.85 才停）→ **CONTINUE**。ov_global 0.921；rec_own_2 0.898 vs rec_cross_2 0.556（重疊主要是不分問題的塊；答案塊各不相同）；rec_union_2＝rec_own_2。細節：`runs/S0_union.md` 結果節。
**接著送**: Q128（1 片）、Q256（1 片）、Q14-256（2 片；每題約 90 s）——使用者 10/26「都做掉」＋登記 §2 CONTINUE。
- **送出**（2026-10-26 18:3x）：`sbatch --job-name=s3-kvs-s0-<cell>-<i> --time=4:00:00 code/r_h200_s3.sh --part kvs-s0 --cell <cell> --shard <i/n>` → Q128 0/1 **22516**、Q256 0/1 **22517**、Q14-256 0/2 **22518**、1/2 **22519**（next-icpe 為空，沒加依賴）。四格都跑完後 `kvs_s0.py report` 做 11/17 預篩（要 A0 的判定；A0 還沒判完 → `PENDING_A0`）。

## S3-QUOTA-1 — /work 群組配額碰到上限（2026-10-26 18:53–19:1x）；刪 DL_Lab4 的權重（使用者指示）

**看到的**: `quota -s -g ChihChungHsu`：1523G／1536G（limit）。18:53 A0 TF Q14-256 分片 0（job 22498，run `20261026-181019-s3-a0-tf-Q14-256-s0`）在關輸出檔時 `OSError: [Errno 122] Disk quota exceeded` → FAILED；輸出 `a0_tf_Q14-256_s0.csv.gz` 被截斷（`gzip: unexpected end of file`，可讀 855 行）。19:0x head node 上 `git pull` 也 `fatal: error when closing loose object file: Disk quota exceeded`（ICPE session 跑 `git fsck --connectivity-only` 確認共用物件庫沒壞）。19:01 降到 1520G（群組用量在波動；我們帳號 79 GB，其餘是群組其他人的）。
**處理**: 先 `scontrol hold` stage-3 排隊中的 17 個 job（避免連環失敗與寫壞的檔）；通知 ICPE session（它沒有 job、沒有寫 /work）。使用者 10/26：「DL_Lab4 刪掉，那個做好了」→「刪掉權重」→ 只刪 `DL_Lab4/runs/` 下 34 個權重檔（`best_ema.pt`、`last.pt`、`swa_cycle*.pt`、`pretrained.pt`；33.35 GiB）；程式、資料、圖、log、npy／npz／pkl（約 16 GiB）、`.venv` 都沒動。之後群組 **1487G**（剩約 49 GB）、我們帳號 44.8 GB。
**恢復**: 截斷的 `a0_tf_Q14-256_s0.csv.gz` 與同一分片的 `gm04_Q14-256_s0.csv.gz` 搬到 run 目錄（`*.TRUNCATED`、`*.FROM_FAILED_SHARD`，不進 results）；放行 17 個 job；分片 0 重送 → **22527**（從頭做）。

## S3-KVS-F0P — F0P 優先權反轉：INHERIT-ONLY〔模擬〕（2026-10-26 19:02–19:25，H200 head node）

**狀態**: 完成；判定 **INHERIT-ONLY**（Qwen 重現同樣）。依偏離 1 改在 H200 head node 跑（使用者 10/26「不要在 mac 跑」），Mac 的 run 已作廢。
**run_id**: 決定性 `20261026-190229-s3-kvs-f0p-dec`；全網格 `20261026-190431-s3-kvs-f0p-full`；報告 `20261026-192513-kvs-f0p-adhoc`
**指令**:
```bash
git fetch origin claude/llm-inference-deadlock-mapping-e6acb7 && git merge --ff-only FETCH_HEAD   # → a8859d3，push main
source code/env_h200.sh; export TIARA_ALLOW_NO_SLURM=1
python3 code/s3/kvs_f0p.py --self-test                                                   # 141 項 ALL PASS
m7run s3-kvs-f0p-dec nice -n 10 python3 code/s3/kvs_f0p.py run --slice decisive --jobs 8
python3 code/s3/kvs_f0p.py report                                                         # 決定性：CONTINUE
m7run s3-kvs-f0p-full nice -n 10 python3 code/s3/kvs_f0p.py run --slice full --jobs 8
python3 code/s3/kvs_f0p.py report
```
**產出**: `results/s3_h200/kvs_f0p/{f0p_decisive_h200_llama.csv, f0p_full_*_*.csv.gz, f0p_verdict.json}`；共享請求明細 `$TIARA_RUNS/20261026-190229-s3-kvs-f0p-dec/f0p_decisive_requests.json.gz`
**關鍵數字〔模擬〕**: 決定性 12 格 inv4 0.115～4.140（全部 > 5% → CONTINUE）。全網格 Llama：(E) inv4 > 15% 的格 H200 58.7%、MI300X 50.5% ✓；(X) esc > 5% 的格 9.0%、4.5% ✗ → INHERIT-ONLY。esc > 5% 的格全是 C4（自己重算）、全在 nfs；D4 從沒贏過繼承。ρ_b＝0 的 inv4 都是 0.0（健全檢查過）。細節：`docs/kv_sharing_hiding_20261026/runs/F0P_priority.md` 結果節。
**重現檢查（只報告）**: 決定性 CSV 去掉 run_id、ts 後的 sha256 和 Mac 那份**相同**（`b96cf08f…f8a`，72 列）。
**失敗與異常**: 一開始 `git pull` 遇到 /work 群組配額滿（S3-QUOTA-1），配額恢復後重做同步；F0P 本身沒有失敗。

## S3-QUEUE-2 — 把卡在冒煙後面的工作也送進佇列（2026-10-26 19:4x；使用者：「不急，只是把任務送出最大化」）

- runner 的 `a0-gen`、`d0-run` 加冒煙閘（同 S3-B0-GATE 的做法）：這份程式（a0／a0_gpu；d0／d0_shim 的雜湊）在這一格還沒過冒煙 → 先在同一個 job 裡跑（生成：2 題；D0：P0f 冒煙），輸出在 run 目錄、不看分數；過了在 `$TIARA_RUNS/.s3_gates/` 留戳記；失敗 → step 停下。理由：獨立的冒煙 job 會讓位、重送成新編號，afterok 會斷。
- 送出：A0 生成 L128、Q128 各 4 片（288 題；每片 `--budget-min 210`、可續跑）；D0 量測 1 片（192 題）。原本排隊中的 `a0gensmoke-L128`、`d0prep` 照留（先跑到的那個留戳記，另一個就跳過冒煙）。送出前 `quota -s -g ChihChungHsu` 見下一行。
- 配額（送出前）：1487G／1536G。job：A0 生成 L128 0–3／4 → **22528–22531**、Q128 0–3／4 → **22532–22535**；D0 量測 0/1 → **22536**（next-icpe 為空，沒加依賴）。佇列裡我們的 job 共 27 個。

## S3-GM12-FAIL-1 — G-M12 品質 Q128 的 GPU 自檢沒過（job 22505，2026-10-27 02:34；停下回報）

**錯誤**: run `20261027-023424-s3-gm12-selftest-Q128`：`gm12 gpu-selftest 沒過` `{"dense_path_max_abs_diff": 28.3125, "n_select": [2, 3, 3, 3, 10, 25], "ok": false}`；runner 停在 `GM12-SELFTEST-Q128`（exit 1）。
**對照**: L128 的同一個自檢（run `20261027-005002-s3-gm12-selftest-L128`）`dense_path_max_abs_diff` **0.0**（逐位元相同）、ok。
**判斷（NOT_VERIFIED）**: 參考 KV 是 ItemState 的 fa3 prefill（和 L128 相同路徑）。sparse_chunk 在 d＝1 時把 GQA 拆成每個 KV head 一條序列（Qwen 7 個 q head／1 個 KV head；Llama 4／1），KVModel 是整組一起送 fa3。Llama 逐位元相同、Qwen 差 28 → 可能是 fa3 對兩種形狀選了不同的 split-KV（加總順序不同）、在 28 層裡累積，加上 Qwen 的 K 有 bias、數值大（絕對差被放大）；也可能是 Qwen 才有的結構差異。要在 GPU 上量逐層差、相對差、`num_splits=1` 時是否歸零才知道。
**判準**: 登記（`runs/G_M12_sparse_recompute.md` §2）：「d＝1 時的 KV 和 KVModel fa3 prefill 的 KV 最大差 < 0.25，否則停」→ **G-M12 的 Qwen 品質停下**，不改門檻。待使用者決定是否做上面的診斷。

## S3-D0-FAIL-1 — D0 的 infllm_v2 編不起來：上游 cutlass 標頭檔的語法錯（job 22507，2026-10-27 02:38；停下回報）

**錯誤**: run `20261027-023815-s3-d0-build`：`/tmp/hungweilai-s3/d0/src/csrc/cutlass/include/cutlass/cuda_host_adapter.hpp:122:2: error: #else after #else`、`:93: error: the conditional began here`、`:148:2: error: #endif without #if`、`:154:2: error: #endif without #if` → `ERROR: Failed building wheel for infllm_v2`；runner 停在 `D0-BUILD`（exit 2）。
**原因（已查證）**: infllmv2_cuda_impl `93cf2ec2` 的子模組 `csrc/cutlass`＝`zhangyan-didu/cutlass` branch `infllmv2-cuda13` commit `424c5a03`。那一版的 `include/cutlass/cuda_host_adapter.hpp` 第 103 與 145 行是 `if ((CUDACC_VER_MAJOR > 12) || (CUDACC_VER_MAJOR == 12 && CUDACC_VER_MINOR >= 5))`——少了 `#`、巨集名少了底線（應為 `__CUDACC_VER_MAJOR__`），所以前處理器的 #if／#else／#endif 對不起來。是上游的錯，不是這裡的設定（head node 從 raw.githubusercontent.com 抓同一個 commit 的檔案確認）。
**處理**: `scontrol hold 22536`（D0 量測，會在同一步失敗）。可行的做法（待使用者決定，都要寫成 D0 偏離）：(a) 編譯前在 job 本地把那兩行改成 `#if ((__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))`（只修語法，不改演算法）；(b) 換一個能編的 infllmv2 commit；(c) D0 這一輪不做。

## S3-KVS-S0（續 2）— S0-2 四格都完成：預篩 PENDING_A0（2026-10-27 04:28）

**run_id**: Q128 `20261027-024744-s3-kvs-s0-Q128-s0`、Q256 `20261027-030441-s3-kvs-s0-Q256-s0`、Q14-256 `20261027-030624-…-s0`／`20261027-030901-…-s1`（L128 見前節）；報告 `20261027-042845-adhoc`
**指令**: `python3 code/s3/kvs_s0.py report` → `results/s3_h200/kvs_s0/s0_verdict.json`
**關鍵數字（實測）**: ρ＝1/4 中位數 u2_global／u2_layer：L128 0.540／0.543、Q128 0.556／0.557、Q256 0.544／0.543、Q14-256 0.536／0.538 → 4 格都 ≤ 0.75。預篩 **PENDING_A0**（S0 那一半成立，等 A0 GO／STOP／WEAK）。四格都是「重疊高、答案塊各不相同」（rec_cross_2 0.55–0.63 vs rec_own_2 0.90–0.97）。guard 都 CLEAN。細節：`runs/S0_union.md`。

## S3-SCHED-5 — 多卡的量時間 job 不再讓人讓位；256K 的生成配置與分片（2026-10-27 06:3x）

- **看到的**：a0gen-Q128-1（22533）開跑後讓位給 G-M06（22281）、重送成 22605（afterany:22281）——G-M06 的排隊原因當時不是 `AssocGrp*`（S3-SCHED-4 的例外沒擋到）。
- **修正**：`sched.py s3-timing-target` 跳過要 >1 張卡的量時間 job（`gpus_of`，讀 squeue `%b`）：G-M06 要整個帳號的 4 卡，它一開跑、帳號上限就保證我們沒有別的 job 在跑，不需要讓位。自檢 PASS；當下的 target 為空。22605 的依賴拿掉（我們自己的 job）。
- **256K 的 A0 TF 做完**：Q256（22509、22510）、Q14-256（22527、22499、22500、22501）各 72／72 題（A0 偏離 3）；`gzip -t` 都過。`a0.py select-gen`：Q256 23 個配置（run `20261027-063058-adhoc`；G1 B16·1/16·int4、B24·1/8·int4、B32·1/8·int4）、Q14-256 24 個配置（`20261027-063059-adhoc`；G1 B24·1/16·int4、B24·1/8·int4、B32·1/8·int4）→ `gen_selection_{Q256,Q14-256}.json`。
- **生成速度（實測，切分片用）**：Q128 每題約 6 s＋prefill 9.7 s（run `20261027-062635-s3-a0-gen-Q128-s2`）；Q128 生成冒煙 PASS（`20261027-061411-s3-a0-gensmoke-Q128`）。256K：Q256 切 4 片、Q14-256 切 8 片（每片預計 1–1.5 小時〔估計〕；第一片會先跑生成冒煙）。
- 送出（配額 1339G／1536G）：A0 生成 Q256 0–3／4 → **22606–22609**；Q14-256 0–7／8 → **22610–22617**（next-icpe 為空）。
