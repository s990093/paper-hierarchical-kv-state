# RUNLOG — 平台 C（H200，hgx-cn01）ICPE 2027 補量

> 格式照 CLAUDE.md §4.2。只有 ICPE 補量（`docs/h200/icpe/`）寫這份；第二階段寫 `RUNLOG_H200.md`、第三階段 3b 的移植寫 `RUNLOG_H200_P3.md`。
> 原始 log：`$TIARA_RUNS/<run_id>/`（＝`/work/ChihChungHsu/hungweilai/tiara-runs/`）。沒量到就寫 `NOT_MEASURED`。

---

## ICPE-0 — 前置（head node，CPU）

**狀態**: PASS
**執行時間**: 2026-10-22T02:43Z → （登記 commit 的時間見下）
**做了什麼**:
- 合併 `origin/main`（`09950ab`：MI300X 第三階段 m16_* 的程式與結果）進 `h200`：在暫時 worktree（detached 在 `ea87db9`）`merge --no-ff`，沒有衝突 → `cf31400`，再 `--ff-only` 進共用 checkout。合併時節點上沒有我們這個 repo 的 GPU job。
- 新增 `code/h200_gitpush.sh`：使用者說「push 只用 gitpush.sh、不要 git stash」；原本的 `code/m10_queue_mi300x/gitpush.sh` 寫死 MI300X 的路徑（`/mlsteam/workspace/...`）而且推 `main`，在這台跑不了，也違反「AMD 結束前只推 `origin/h200`」。H200 版照同一個原理（fetch → merge → push，不 stash、不 rebase），推 `origin/h200`。
- 事先登記：`docs/h200/icpe/P0-1_recompute_speed.md`、`P0-2_x1_gpu_validation.md`（這一節的 commit；寫的時候沒有任何 P0 的資料）。
**同節點的狀況**（2026-10-22T02:4xZ，`squeue`）：我們帳號在 hgx-cn01 上有 3 個 GPU job 在跑（DL_Lab4 的 `H_hr480`、`G_long300`、`I_xl192`，各 16 CPU；後面還排了 `G_swa`、`H_swa`、`F3_mim`），lab 上限 4 張卡／64 CPU。P0 的量時間 job 要等它們（見「要問的」）。
**失敗與異常**: 無。

### ICPE-0b — 磁碟與 GPU 排程（使用者 2026-10-22 決定）
**磁碟**（使用者問「目前空間還夠嗎」）：lab 的 `/work` 群組配額 1,533G／1,536G（剩約 3 GB，`quota -s -g ChihChungHsu`，2026-10-22T03:3xZ）；我們帳號 148G（hf-cache 99G、DL_Lab4 34G、conda 8.8G、tiara-runs 1.4G）。
- 使用者選「刪兩個沒用到的模型（建議）」＋另外跟老師要配額當緩衝。刪除（`huggingface_hub.scan_cache_dir().delete_revisions(...).execute()`，只刪沒有被別的模型引用的 blob）：
  - `Qwen/Qwen3-30B-A3B-Instruct-2507` revision `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`（56.89 GiB；第一階段第 1 輪的 MoE，已結束）
  - `Yukang/LongAlpaca-7B` revision `14c462dc860ba1818194ef0b5a31387ba57f0e4d`（12.55 GiB；同上）
  - 之後：群組 1,464G／1,536G、我們帳號 79,451M（2026-10-22T03:52Z）。Llama-3.1-8B（14.97 GiB）、Qwen2.5-7B-1M（14.2 GiB）、noDCA 變體（連到 Qwen 的 blob）都在。要重下載就用上面兩個 revision。
**GPU**：使用者自己的 DL_Lab4 有 3 個 job 在 hgx-cn01 跑（21566 `G_long300` 到 10/23 16:24、21567 `H_hr480` 到 20:24、21568 `I_xl192` 到 22:22，+08:00 的時限），排隊的 3 個（21570 `G_swa`、21571 `H_swa`、21618 `F3_mim`）。
- 使用者選「把排隊的 DL_Lab4 先 hold（建議）」：`scontrol hold 21570 21571 21618`（2026-10-22T03:5xZ）。在跑的 3 個不動；P0 量時間的 job 等它們結束、節點上沒有我們別的 job 才開跑。**P0 量時間做完要 `scontrol release 21570 21571 21618`**。

---

## ICPE-P0 — P0-1、P0-2 的程式與送出（2026-10-22）

**程式**（commit `e6338bd`、`099552f`、`9d7f506`）：`m7_model.py` 的 `M7_ATTN`（沒設＝sdpa，程式路徑不變）、`code/icpe/attn_bench.py`、`p01_tools.py`、`smoke_p01.sh`、`x1_gpu.py`、`p3_fvar.py`、`headrun.sh`、`code/r_h200_icpe.sh`（part p01s1、p01ab、p02、cpu）。NV_PORTING_NOTES 加一列（`M7_ATTN`）。
**P0-1 偏離 1**（寫在任何資料之前，見 P0-1 §8）：G5b 只配 S0／S2，判定用的 Cake／全部載入改在 S4 量、G5b 另在 S0 量。

**head node 的開發檢查**（scratchpad，不進 results）：x1_gpu selftest（k＝16）全過；`p3_fvar` V0 在 conversation／R1／128 GiB 的 36 格和 git 的 `x1_sweep.csv.gz` 逐值相同（差 0.0）；V2 的 X5：f×0.5710、CPU 層校正比值不變、慢層 gate 仍過。

### P0-2 的 CPU 步驟（head node，`code/icpe/headrun.sh`，§4、§7、§6 的 S_orig）
| run_id | 指令 | exit | 產出 |
|:--|:--|:--:|:--|
| `20261022-120540-icpe-p02-select` | `x1_gpu.py select --out results/icpe_h200/p02` | 0 | `select.json`：**k＝8**（k＝4 的 ρ̂＝0.534 > 0.30；k＝8 的 ρ̂＝0.276、O-PF 對 LRU 23.1%／19.9%；k＝16 ρ̂＝0.156、18.5%／16.5%） |
| `20261022-120545-icpe-p02-selftest` | `x1_gpu.py selftest --out … --k 8` | 0 | `selftest.csv`：9／9 過（重播≡Engine.run 5 個做法；L_a＝0 時 HINT-PF≡LRU；L_a 中位 22.9 s；HINT-PF 預取 136.8 GiB；ChanTier） |
| `20261022-120549-icpe-p02-sim-orig` | `x1_gpu.py sim --out … --k 8` | 0 | `sim_requests.csv`（S_orig） |

縮小後（k＝8：1,535 個請求、467 個回來請求；CPU 16／32 GiB）的模擬，回來請求平均 TTFT（s）〔模擬〕：
| 容量 | LRU | FREQ2Q | O-BEL | HINT-PF | O-PF |
|:--|--:|--:|--:|--:|--:|
| 128/8＝16 GiB | 0.3052 | 0.2997 | 0.2669 | 0.2441 | 0.2346 |
| 256/8＝32 GiB | 0.2937 | 0.2936 | 0.2536 | 0.2440 | 0.2352 |
- 〔判讀〕縮小後的 headroom 比全尺寸小（全尺寸 O-PF 對 LRU 52%／47%，這裡 23%／20%）：抽樣 session 也拿掉了跨 session 的前綴共用；GPU 驗證的是「縮小後這個設定的模擬」，論文要寫比例與這個差。

### 送出的 job（全部 `env -u TMPDIR -u CLAUDE_CODE_TMPDIR sbatch`，鏈在一起、不重疊）
| job | 名稱 | 依賴 | 內容 |
|:--|:--|:--|:--|
| 21652 | `icpe-p01-smoke` | — | 冒煙（不量時間、不進 results）：FA2／FA3／FA4／cuDNN 接線、grid 在 fa3＋S4＋nfs |
| 21653 | `icpe-p01s1` | afterok:21652、afterany:21566:21567:21568（DL_Lab4 在跑的 3 個） | P0-1 S1＋勝者的校準 |
| 21654 | `icpe-p01ab` | afterok:21653 | P0-1 (a)(b) |
| 21655 | `icpe-p02` | afterany:21654 | P0-2 冒煙＋12 個重播 |
| 21656 | `icpe-cpu` | afterany:21655 | P0-1 (c)（V0／V1／V2）＋P0-2 全尺寸 HINT-PF |
送出時 Slurm 預估 21652 在 10/22 20:54（+08:00）開始（lab 的 CPU 上限）。量時間的 job 開跑時還會再檢查節點上沒有我們自己的別的 job。

---

## ICPE-P12-D — X4「排隊占比」的拆解（head node，CPU；P1-2 §3，登記 2026-10-22T04:14Z）

**狀態**: PASS（第一次 exit 1，修正後重跑）
**run_id**:
- `20261022-121501-icpe-p12-queue` — **exit 1**：`ValueError: could not convert string to float: 'conversation'`（`p12_queue.py` 用 `dtype == object` 判字串欄；pandas 3 的字串欄是 Arrow 字串型，不是 object → 把 trace 欄當數字）。我的程式錯，沒有產出。改成 `pd.api.types.is_numeric_dtype`（commit 見下）後重跑。
- `20261022-121515-icpe-p12-queue` — exit 0
**指令**: `bash code/icpe/headrun.sh icpe-p12-queue python code/icpe/p12_queue.py --cfg results/p3_explore_h200/x4/x4_verdict_cfg.csv --sweep $TIARA_RUNS/20261021-012002-p3h-x4-sweep/x4_sweep.csv --out results/icpe_h200/p12/x4`
**產出檔**: `results/icpe_h200/p12/x4/{p12_queue_verdict.json, p12_queue_cells.csv}`
**關鍵數字**〔模擬，H200 常數；資料＝H200 已跑的 X4 主掃描 run `20261021-012002-p3h-x4-sweep`〕:
- **D1**：可進判定 262 格，NP 的排隊 p99 ÷ TTFT p99：中位 **0.983**、最小 0.846、p10 0.947 → ≥0.90 → **「p99 尾端仍幾乎全是排隊」成立**（3090 分支：中位 0.995、最小 0.939）。
- **D2**：可進判定且平均 headroom ≥5% 的 63 格裡，43 格「Δ還原 ÷ Δ ≥ 0.5」→ **「平均值的空間主要來自還原變快」**（Δ還原占比中位 0.64）。這 63 格的最好對手多半是 QP（不用預測的佇列預取），其次 GP。
- 只報告：QP 對 NP 的 p99 改善中位 3.1%、最大 25.9%。
**失敗與異常**: 上面的 exit 1（程式錯，已修）。
**與 3090 分支的差異**: D1 方向相同（H200 的排隊占比略低：中位 0.983 vs 0.995）。D2 是 H200 才有的問題（3090 的平均只到「可能」）。

### 10/22 加送：P1-2、P2（登記 commit `72e048a`；程式 `b17ea51`）
| job | 名稱 | 依賴 | 內容 |
|:--|:--|:--|:--|
| 21662 | `icpe-p12` | afterany:21656 | P1-2（CPU）：X1 FI、X5 事後版本 B1–B5、X4 f×0.5（D 已在 head node 算完） |
| 21663 | `icpe-p2` | afterany:21662 | P2（量時間）：G5b CPU 層忙，seed 310／311 |
**注意**：P2（以及之後 P1-1 的量時間）也不能和 DL_Lab4 同時跑。使用者 10/22 只同意 hold 到「P0 量時間做完」；P0 做完時要再問一次：放出 DL_Lab4（P1／P2 的量時間就等它們），還是繼續 hold 到 ICPE 的量時間全部做完。runner 開跑前會檢查，節點上有我們自己的 job 就直接退出（不會污染）。

### 10/22 加送：P1-1（登記 `68d248e`；程式 `9d…`→`1b6a2c2`）
- **head node（CPU）**：`20261022-122600-icpe-p11-plan`（exit 0）→ `results/icpe_h200/p11/p11_selection.json`：Qwen `q:int4_kivi`、`xl:k34:v51:cal+q:int4_kivi`（9.64 倍，每請求不含共用基底）、`xl:k34:v51+q:int4_kivi`（8.01 倍）；Llama `q:int4_kivi`、`xl:k29:v44:cal+…`（11.22 倍）、`xl:k19:v29:cal+…`（17.07 倍）、`xl:k29:v44+…`（7.98 倍）、`xl:k19:v29+…`（12.14 倍）。新題 seed 67000 和 x1／calib／x2／x2b／x4a／x4b／xlcal3／c1a／44c 都 0 重疊。
- **m16_xtime 的改動**：`M16_XKV_EXTRA` 加新段（`_cal`＝共用基底常駐 GPU、還原不搬）。沒設時段清單不變；CPU 自測 97 項的結果和改之前（暫時 worktree 在 HEAD）相同——兩邊都只有 1 項 FAIL，是舊的（「位元組帳 vs X2 實際寫出的 ratio，第一版 x2_unmapped.csv 比了 0 列」：MI300X 37zz 之後那個檔變空，和這次無關）。新段的位元組（coef＋basis）和 `m16_xstack.predict_bytes` 逐項相同（head node 核對）。
| job | 名稱 | 依賴 | 內容 |
|:--|:--|:--|:--|
| 21669 | `icpe-p11q` | afterany:21655 | P1-1 (Q)：PCA／xKV 校準、gpu-selftest、192 題生成（不量時間；和 CPU 的 21656、21662 可以同時） |
| 21663 | `icpe-p2` | **改成** afterany:21662、afterany:21669 | （量時間，不能和 21669 重疊） |
| 21670 | `icpe-p11t` | afterany:21663 | P1-1 (T)：loadpath 冒煙＋8 格＋CPU 層、壓縮時間 |

### 10/22 冒煙 job 加做、S1 改成手動放行
- 21652（冒煙，還沒開跑）加：P0-2 的 GPU 重播（HINT-PF、k＝8、前 50 個請求）、P1-1 的 loadpath（`--smoke`）；時限 1:15（commit `ef98fd1`）。
- 21653（P0-1 S1）的依賴從 `afterok:21652` 改成 `afterany:21652`（不讓冒煙裡別的實驗的失敗自動取消整條鏈），並 **`scontrol hold 21653`**：冒煙跑完、我看過結果才放行。
- 開 `docs/h200/ICPE_EVIDENCE.md` 骨架（只放已核對的數字；待填的列標出是哪個實驗）。

---

## ICPE-SMOKE — job 21652（不量時間、不進 results）
**狀態**: PASS（exit 0）　**執行時間**: 2026-10-22 15:51:32 → 15:54:46（+08:00）　**run_id**: `20261022-155134-icpe-p01-smoke`（輸出在 run 目錄的 `smoke/`）
**看到的**:
- attention：五個候選（sdpa、fa2、fa3、fa4、sdpa_cudnn）在 Llama、Qwen 都載得起來、跑得通；2K／8K 的 argmax 都和 HF 相同，max|Δlogit| ≤ 0.156；第 0 層 KV 和 sdpa 逐位元相同（接線對）。**sdpa 自己在 2K 的 top-5 就和 HF 不同**（兩個模型）→ 登記的「top-5 集合相同」會隨機踢掉候選 → P0-1 §8 偏離 2（改成至少 4 個相同；S1 還沒跑）。fa4（CuTe DSL）在 sm_90 可用。速度只量 8 塊（不算分數）：fa3 每塊的增加量約 sdpa 的一半。
- m10_speed grid（fa3＋S4，CPU 層、nfs；G5b＋S0，CPU 層忙）各 1 次：跑完、`kv_bad` 0；G5b 在 CPU 層忙時又選了全部載入（0.235 s vs Cake 0.123 s）。
- P0-2 重播（HINT-PF、k＝8、前 50 個請求）：跑完；50 個請求 GPU÷S_orig 的 TTFT 平均比 0.969、逐請求中位 0.951（小請求 GPU 較快：模擬把最後不滿的一塊算成整塊）；22 個請求排隊（序列開頭都是第一次的 prefill）。
- P1-1 loadpath（`--smoke`，4 塊 ×16）：兩個模型都跑完、位元組帳斷言過。每題 SVD 版在冒煙裡會被 ×16 放大基底的搬運（只有冒煙這樣，正式 64 塊不縮放）。
**之後**：`scontrol release 21653`（S1 仍等 DL_Lab4 的 3 個 job 結束）。

---

## ICPE-P01-S1 — job 21653（P0-1 §2：最快的 attention＋勝者的校準；量時間）
**狀態**: PASS　**執行時間**: 2026-10-22 20:57:35 → 21:03:42（+08:00）　**開跑時**：DL_Lab4 的 3 個 job 已經提早結束；節點上沒有我們別的 job（runner 檢查過）；別人的 job 6 個（`slurm-icpe-p01s1-21653.out`）。
**run_id**: `20261022-205738-icpe-p01-attn-llama`、`…-205836-icpe-p01-attn-qwen`、`…-205931-icpe-p01-attn-choose`（推算；見 slurm log）、`…-205934-icpe-p01-c1`、`…-205953-icpe-p01-calibf-llama`、`…-210010-icpe-p01-c5`、C4 same＋restart ×10（`…-2100xx`～`…-210309-icpe-p01-c4-*`）、`…-210325-icpe-p01-calibf-qwen`；全部 exit 0。
**產出檔**: `results/icpe_h200/p01/attn/{gate.csv,speed.csv,choice.json,gpu_guard_*.json}`、`results/icpe_h200/p01/calib/{calib_c1.csv,calib_c5.csv,c4_noise.csv,speed_{llama,qwen}/}`
**關鍵數字**〔實測〕:
- 正確性閘（8K、32K）：五個候選兩個模型都過；argmax 都相同、top-5 重疊都是 5／5、max|Δlogit| ≤ 0.125。（偏離 2 的「至少 4 個」在這次沒有作用：全部 5／5。）
- 速度（Σ_{i<64} f(i)，3 輪中位數，ms）：Llama sdpa 2,107.2、fa2 2,049.5、**fa3 1,583.0**、fa4 1,605.5、sdpa_cudnn 2,041.7；Qwen sdpa 1,854.0、fa2 1,806.5、**fa3 1,400.4**、fa4 1,416.8、sdpa_cudnn 1,782.4。→ **勝者 fa3**（Llama 0.751×、Qwen 0.755× sdpa），停損沒觸發。
- 勝者的校準：C1（Llama，fa3）f(0) 18.17、f(63) 30.92 ms，斜率 **0.2104** ms/塊（sdpa 0.4730），Σ64＝1,569.8 ms（sdpa 2,091.4）；calib-f Σ64：Llama 1.5741 s、Qwen 1.3934 s；C5 duty 0.6／1.0 變慢 1.536／**1.970** 倍（sdpa 1.511／1.945）；C4 δ 0.02–0.03% → δ\*＝5%。
- 新的 harness ÷ vLLM 預設 32K（只報告）：Llama 1.574／1.2106＝**1.30**、Qwen 1.393／1.0590＝**1.32**（sdpa 時 1.75／1.76）→ 最快的 attention 只補回一部分；「跟 vLLM 一樣快」只能靠 (c) 的 V2。
**失敗與異常**: 無。

---

## ICPE-P01-AB — job 21654（P0-1 (a) B 主設定、(b) 還原格子；量時間）
**狀態**: (a) PASS；(b) **暫定**（有兩個 run 和我們自己的 job 重疊，要照重疊重跑）
**執行時間**: 2026-10-22 21:03:42 → 23:45:27（+08:00），exit 0
**重疊**: DL_Lab4 的 21570 `G_swa`（hold 在 23:28:31 被放掉，不是這個 session 放的）23:41:25 開跑 → 和 (b) 最後兩步重疊：`20261022-233934-icpe-p01b-qwen-s4-fa3`（重疊 23:41:25–23:42:06）、`20261022-234206-icpe-p01b-qwen-g5b-fa3`（全程）。照重疊選（memory h200-no-self-cotenancy-for-timing），兩個 run 目錄放了 `OVERLAP_OWN_JOB`，列在 `results/icpe_h200/p01/grid/EXCLUDED_RUNS.csv`，分析排除它們；重跑用 runner 的 `--part p01brr`（同一個 seed，launch0 501），**還沒送**。(a) 的 B（21:04–22:41）和其他 (b) 的 run 都沒有重疊。
**(a)**（`20261022-234955-icpe-p01a-analyze`，head node）：S5 對 S4+（回來請求 TTFT 中位數，rep 0–5 合併，正＝S5 快）cpu 0.25／0.5／1.0 ＝ **+0.07%／−0.47%／+0.03%**，δ\*＝5%（C4 δ 0.03%）→ **三格都成立 → (a) 成立**。平均值版 +0.74／−0.09／+0.03%；S5 對最好的簡單策略 −16.2%（S4）／−26.0%（S1）／−0.05%（S2b）；舊合併組（sdpa）−0.03／−0.12／+0.03%。污染列 0、kv_bad 0。產出 `p01a_table.csv`、`p01a_verdict.json`、`b/b_share.csv.gz`。
**(b) 暫定**（`20261022-235022-icpe-p01b-analyze-provisional` 含重疊的 run；`…-icpe-p01b-analyze-excl` 排除後）：24 格 **0 翻轉**（兩種都是）；弱變化 1 格：Llama CPU 層閒 E0，sdpa 平手（Cake 0.0844／全部載入 0.0852 s）→ fa3 全部載入較快（Cake 0.1002／0.0849：fa3 時 Cake 反而變慢，原因沒查）。慢層每格都還是 Cake 最好（fa3 的 Cake 快 4–19%）。G5b ÷ 最好（只報告，S0）：sdpa 1.21、fa3 1.33。排除重疊的 run 後 Qwen fa3 每格只剩 5 次 → 判定要等 `p01brr` 重跑才定。
**排程（2026-10-22 23:5x）**：使用者要 lab 的 GPU 先給 DL_Lab4（另一個 session 照使用者的要求 hold 了 21655、21656、21662、21663、21669、21670）。我只把 21656、21669 的依賴改回 `afterany:21655`（放行時 P0-2 先自己跑，不和 CPU／品質 job 重疊），沒有放行、沒有重送。使用者說可以之後：先送 `p01brr`（afterany 無），再 `scontrol release`，並讓 21655 接在 `p01brr` 後面。

---

## ICPE 恢復排程（2026-10-25T11:34Z，使用者：「先跑重跑 p01brr，確認後再 release，繼續 ICPE」）
**節點狀態（`date -u` 11:31Z）**：DL_Lab4 還有 2 個在跑（21907 `J_kd`，時限到 10/26 06:35 +08；21779 `final_ens`，10/25 19:26 +08 剛開跑，時限 16 h），2 個排著（21905 `final2` afterany:21779；21909 `final3` afterany:21907,21905）。照 memory h200-no-self-cotenancy-for-timing（DL_Lab4 算自己的），量時間的 `p01brr` 現在開跑只會被 runner 擋下退出 → **量時間的 job 改排在 DL_Lab4 這 4 個之後**；不量時間的先跑（和 DL_Lab4 同時跑不違反規則：DL_Lab4 不量時間）。
**改動**（沒有改任何 DL_Lab4 的 job）：
| job | 名稱 | 依賴（新） | 狀態 |
|:--|:--|:--|:--|
| 21656 | `icpe-cpu` | 無（原 afterany:21655） | release → 11:34Z 開跑 |
| 21669 | `icpe-p11q` | 無（原 afterany:21655） | release → 等 account 的 4 卡上限（MaxGRESPerAccount） |
| 21662 | `icpe-p12` | afterany:21656（不變） | release |
| **22233** | `icpe-p01brr`（新送，0:30） | afterany:21656:21662:21669（ICPE 不量時間的）＋21779:21905:21907:21909（DL_Lab4） | 不 hold |
| 21655 | `icpe-p02` | **afterany:22233** | **仍 hold**：p01brr 跑完、我看過（含 sacct 對照有沒有和自己的 job 重疊）才放 |
| 21663 | `icpe-p2` | **afterany:21655**（原 afterany:21662,21669） | 仍 hold |
| 21670 | `icpe-p11t` | afterany:21663（不變） | 仍 hold |
- account `ChihChungHsu` 的上限：gres/gpu＝4、cpu＝64（`sacctmgr`），lab 其他人的 job 也算在裡面。`cpu`／`p12` 只用 CPU 但 runner 要一張卡（smi_index 檢查），所以也占一格。
- 若 DL_Lab4 之後又送新的 job 到 hgx-cn01，量時間的 part 開跑時會被 runner 擋下（exit 2），跑到一半才開始的只能事後用 sacct 對照、照重疊重跑。
- origin/main 從上次 merge 後多了 24 個 commit（含 `code/m10_speed.py`：`M10_BUSY_SCRIPT` 環境變數，沒設時行為不變）。**ICPE 的量時間跑完之前不 merge**（p01brr 要和原本 (b) 同一份程式）；寫 ICPE_EVIDENCE 之前再 merge。
- **更正（同日 11:5xZ，使用者：「h200 已經併入 main，之後結果直接送 main」）**：上一行「量時間跑完之前不 merge」作廢。已 `git merge origin/main` 進 h200（`dd9e4bc`；main 多的程式只有 m16_*／s3_mech_arith 與 m10_speed 的 `M10_BUSY_SCRIPT`，沒設時行為不變、只在 busy JSON 多一個 `script` 欄——ICPE 的 CSV 不受影響，p01brr 仍和原本 (b) 同一條程式路徑）；`code/h200_gitpush.sh` 改成推 origin/main（origin/h200 跟著同一個 commit，`916ae50`）。
- ICPE_EVIDENCE.md：A1、A1''、A2、A3、A4、A5（暫定）、C1、D2、D3、E2 補上 run_id（MI300X 的從 main 的結果檔讀；H200 的都核對過 run 目錄在、exit 0）。D2 改成只算 INT4（舊值 21.5–54.5%／22–57% 含 FP8）。

---

## ICPE-P11Q 第一次 — job 21669 FAILED（2026-10-25 20:01:20 → 20:01:36 +08:00）
**狀態**: FAIL（exit 1；第 2 步）　**run_id**: `20261025-200124-icpe-p11-plan`（exit 0；`p11_selection.json` 和 10/22 的逐位元相同，git 沒有變動）、`20261025-200128-icpe-p11-calib-qwen`（**exit 1**）
**錯誤**（`slurm-icpe-p11q-21669.out`）：
```
File ".../code/m7_model.py", line 39, in model_path
    raise FileNotFoundError(g)
FileNotFoundError: /work/ChihChungHsu/hungweilai/hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct-1M/snapshots/e28526f7bb80e2a9c8af03b831a9af3812f18fba
```
**原因**：`/work/ChihChungHsu/hungweilai/hf-cache/hub/` 整個是空的（目錄 mtime 2026-10-23 03:12 +08）；`$HOME/models/`（noDCA）也是空的。10/22 的 P0-1 S1／(a)／(b) 都在這之前跑完（載得到模型）。誰刪的：查不到（NOT_VERIFIED；stage3 session 的 `RUNLOG_H200_S3.md` S3-0 也記了同一件事）。群組配額剩 49G。
**使用者的決定（10/25，經 stage3 session 轉達，寫在 `RUNLOG_H200_S3.md` S3-0）**：權重全部只放 hgx-cn01 本地 `/tmp/hungweilai-models/hf-cache`（HF_HOME 的版面），由 stage3 的 `code/s3/fetch_models.py` 下載**同一個 revision**（Llama `4699cc75…`、Qwen `e28526f7…`）、逐檔對 HF 的 sha256／git blob sha1；ICPE 自己改路徑。
**改動**（`code/r_h200_icpe.sh`，寫新檔再 mv，在跑的 21656 不受影響）：source env_h200.sh 之前 `HF_HOME=$ICPE_MODEL_ROOT`（預設上面那個 /tmp 路徑）並 unset `M7_*_GLOB` → 三個 GLOB 跟著新路徑；要載模型的 part（cpu、p12 以外）開跑前 `fetch_models.py check --models llama qwen`（大小＋inode＋mtime 對戳記），不在就 exit 3；再核對 env_h200 算出的路徑和 `fetch_models.py env` 逐字相同。這支只檢查、不下載（fetch_models 沒有鎖，不和 s3-fetch 搶著寫）。ICPE 的程式都不用 noDCA（那是 vLLM 用的；E1 的 vLLM 分母是 10/21 量的）。
**對可比性**：同一個 revision、逐檔雜湊驗過 → 權重位元組和 10/22 相同；載入時間不在任何量測裡。p01brr 重跑和原本 (b) 仍是同一份模型、同一條程式路徑。
**重送**：
| job | 名稱 | 依賴 | 內容 |
|:--|:--|:--|:--|
| 22235 | `s3-fetch`（stage3 session 送的） | — | 權重 → hgx-cn01 /tmp |
| **22236** | `icpe-p11q` | afterany:22235 | `--from P11Q-CALIB-qwen`（PLAN 已在 21669 跑完） |
| 22233 | `icpe-p01brr` | afterany:21656:21662:21779:21905:21907:21909:22235:**22236** | 21669 換成 22236；22235 是 stage3 session 加的（使用者 10/25「排滿：stage3 品質先填空檔」：下一個 ICPE 量時間的 job 等正在跑的 stage3 job） |
