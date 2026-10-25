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
