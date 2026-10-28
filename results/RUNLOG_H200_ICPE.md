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

---

## ICPE-P01-C — job 21656（P0-1 (c) X1／X5 換常數 V0／V1／V2＋P0-2 全尺寸 HINT-PF；只用 CPU）
**狀態**: PASS　**執行時間**: 2026-10-25 19:34:17 → 20:23:36（+08:00），exit 0　**同節點**：DL_Lab4 的 J_kd、final_ens（不量時間的 job，允許）
**run_id**: `20261025-193420-icpe-p01c-x1-sweep-V0`、`…-194544-…-x1-an-V0`、`…-194549-…-x1-h1800-V0`、`…-194554-…-x1-h3600-V0`、`…-194600-…-x5-grid-V0`、`…-195015-…-x5-an-V0`；V1：`20261025-195031`～`…-200641`（同 6 步）；V2：`20261025-200651`～`…-202304`；`20261025-202314-icpe-p02-fullsim`。全部 exit 0。
**分析**（head node）：`20261025-202403-icpe-p01c-analyze` exit 1（我用了系統 python，`ModuleNotFoundError: No module named 'numpy'`）；`20261025-202411-icpe-p01c-analyze` exit 1（`TypeError: dict() got multiple values for keyword argument 'variant'`：X5 的 key 欄有 `variant`，和我記版本的欄撞名；我的程式錯）→ `p01_tools.py` 版本欄改名 `fvar`；`20261025-202422-icpe-p01c-analyze` exit 0。
**指令**: `bash code/icpe/headrun.sh icpe-p01c-analyze ~/.conda/envs/tiara/bin/python code/icpe/p01_tools.py analyze-c`
**產出檔**: `results/icpe_h200/p01/sim_{V0,V1,V2}/{x1,x5}/`、`p01c_verdict.json`、`p01c_crossings.csv`；`results/icpe_h200/p02/fullsim_hint.csv`
**關鍵數字**〔模擬〕:
- V0 回歸：和 git 的 `results/p3_explore_h200/{x1,x5}` 逐值相同（X1 1,152 列、X5 384 列，最大差 0；判定字相同）。
- 判定字（X1 原登記／X1 截尾 H＝3,600／X5 登記；只報告 H＝1,800）：V0、V1、V2 全部「有看頭」→ **(c) 成立**。
- X1（rule＝min，384 格）≥15%：167／168／169 格；X5（48 格）≥15%：11／16／17 格，最大 37.4／40.6／41.5%。跨門檻 85 列（X5 只往上）。
- P0-2 全尺寸 HINT-PF（只報告，24 格）：HINT-PF 對 LRU 5.6–47.8%、O-PF 對 LRU 5.5–52.4%；HINT-PF÷O-PF 的改善比例 SSD 中位 0.939、b32 0.949、NFS 1.117；6 格 > 1（NFS 128／256 GiB 四格 1.25–1.45）。
**失敗與異常**: 上面兩次分析失敗（叫錯／程式錯，已修）。沒做：V2 對第二階段 §5.5 `whatif_recompute.json` 的對照（只報告項）。
**P0-1 總判定**：(a) 成立、(c) 成立、(b) 暫定 0 翻轉 → 等 p01brr（22233）重跑後 `p01_tools.py analyze-b`＋`verdict`。

---

## ICPE-P12 第一次 — job 21662 FAILED（2026-10-25 20:23:37 → 20:41:16 +08:00；只用 CPU）
**完成的步驟**（全部 exit 0）：`20261025-202341`～ P12-VERIFY、A2（X1 FreeInference sweep＋兩個 analyze）、B1（strict）、B2（FI）；B3 的 x5 run `20261025-203700-icpe-p12-x5-cens1800`（exit 0）。
**失敗**：`20261025-204109-icpe-p12-x5-cens1800-an`，exit 1：
```
File ".../code/p3x/x5_analyze.py", line 103, in bootstrap
    sa = np.bincount(inv, weights=a, minlength=nu)
ValueError: The weights and list don't have the same length.
```
**原因（我的 runner 錯）**：`x5_analyze.bootstrap` 用環境變數 `X5_CENSOR_H` 重建設限遮罩（ret ∧ prevT ≤ T_end − H，和 `x5_run.apply_censor` 同一式），`x5_run` 則用 `--censor-H`。runner 的 `x5v` 只把 `--censor-H` 給了 run，analyze 沒有設 `X5_CENSOR_H` → 用沒設限的 session 清單對設限後的 TTFT 陣列。3090 分支的同一步（`20261017-220051-…-cens1800` 等）exit 0，所以當時有設這個變數。
**修正**：`x5v` 從 run 參數取 `--censor-H` 的值，analyze 用 `env X5_CENSOR_H=<H>`（沒設限的步驟 H＝0，和之前一樣）。寫新檔再 mv（22236 正在用這支）。
**重送**：**22240** `icpe-p12 --from P12-B3-CENS1800`（B3 的 x5 run 重跑一次，raw 目錄先刪，所以輸出和 analyze 一致）。p01brr 22233 的依賴加上 22240（現在：afterany:21907:21909:22236:22238:22240；已滿足的 Slurm 會自己拿掉）。
- **10/25 12:5xZ runner 小改**（stage3 session 告知 main `23e509e`：stage3 的 job 開跑時若看到沒 hold、只等 s3-* 或沒有依賴的 ICPE 量時間 job，就重送自己到它後面、exit 0）：量時間 part 的「節點上有自己的 job → 退出」改成每 60 s 再查、最多 15 次，還有才退出。目的：兩邊同一輪排程一起開跑時，等 stage3 讓開，不讓量時間鏈一路 exit 2。量測仍然只在節點上沒有我們別的 job 時才開始。

## ICPE-P12 第二次 — job 22240 FAILED（2026-10-25 20:47:01 → 21:00:15 +08:00；只用 CPU）
**完成的步驟**（全部 exit 0）：B3 四個（cens1800、strict-cens1800、fi-cens1800、fi-cens3600 的 run＋analyze；`X5_CENSOR_H` 修正有效）、B4（loadonly）。
**失敗**：`20261025-210010-icpe-p12-x5-le32k`，exit 1：
```
File ".../code/p3x/x5_short_only.py", line 31, in main
    h = pd.read_csv(os.path.join(a.out_dir, f"x5_headroom{sfx}.csv"))
FileNotFoundError: [Errno 2] No such file or directory: '.../results/icpe_h200/p12/x5/x5_headroom.csv'
```
**原因（我的 runner 錯）**：`x5_short_only` 從同一個 `--out-dir` 讀兩組的 `x5_headroom{sfx}.csv`；主格子（sfx ''）是 P0-1 (c) V0 算的，檔在 `results/icpe_h200/p01/sim_V0/x5/`，沒有放到 p12 的 out-dir。
**修正**：B5 這一步先 `cp results/icpe_h200/p01/sim_V0/x5/x5_headroom.csv results/icpe_h200/p12/x5/`（複製的檔保留原 run_id 欄＝V0 的 `20261025-195015-icpe-p01c-x5-an-V0`；它和 git 的 H200 主格子逐值相同，見 ICPE-P01-C），再跑原本的 `x5s`。登記（P1-2 §2 B5：主格子 raw＋B1 strict raw）不變。
**重送**：**22249** `icpe-p12 --from P12-B5-LE32K`（B5、C1 sweep、C1 analyze）；p01brr 22233 的依賴 22240 換成 22249。
- **10/25 13:0xZ 排程變化（不是這個 session 做的）**：DL_Lab4 的 21907 `J_kd`、21909 `final3` 在 21:03:08（+08）被取消（sacct：CANCELLED）。p01brr 22233 現在只等 22236（p11q）、22249（p12）和 stage3 session 加的 s3-* job（22242、22250–22261）。照 stage3 的自動讓位（main 23e509e），p11q／p12 結束後 p01brr 只剩 s3-* 依賴，之後開跑的 s3 job 會自己重送到 p01brr 後面 → p01brr 最多等那時已經在跑的 s3 分片（≤4 h）。

---

## ICPE-P11Q — job 22236（P1-1 (Q) 品質；不量時間）
**狀態**: PASS　**執行時間**: 2026-10-25 20:37:30 → 21:52:41（+08:00），exit 0　**同節點**：p12、DL_Lab4 J_kd（到 21:03 被取消）、stage3 的 s3-a0smoke／b0prep（都不量時間）
**run_id**: `20261025-203736-icpe-p11-calib-qwen`、`…-203809-…-calibxl-qwen`、`…-203841-…-gpuselftest-qwen`（all_pass）、`…-203913-icpe-p11-gen-qwen`；Llama：`…-2102xx`～`…-210323-…-gpuselftest-llama`（all_pass）、`…-210417-icpe-p11-gen-llama`；`20261025-215237-icpe-p11-report-q`。全部 exit 0。
**產出檔**: `results/icpe_h200/p11/{raw/p11_gen_{qwen,llama}.csv.gz, calib/, p11q_report.json, tables_p11q.md}`
**關鍵數字**〔實測，192 題，seed 67000，INCOMPLETE 0〕: Qwen 主配置 `xl:k34:v51:cal+q:int4_kivi`（9.64 倍）對 E0 −0.6 [−2.9, +1.8]，MLPerf 99% 過（67.73 vs 界 67.65）；Llama 主配置 11.22 倍 −0.6 [−3.3, +2.0]、17.07 倍 −0.7 [−3.4, +2.1]（MLPerf 過、ε 沒過）；X4b 同配置三個都和 MI300X 的 CI 有交集，判定字 Qwen 相同、Llama 兩個不同（H200 +0.1／+0.4 過 ε；MI300X −3.3／−3.6）；校準 vs 每題 SVD 兩個模型都 UNDETERMINED。細節：`docs/h200/icpe/P1-1_xkv_int4_h200.md` §7.1。
**失敗與異常**: 無（第一次 21669 的模型路徑問題見上）。

## ICPE-P12 第三次 — job 22249 PASS（2026-10-25 21:03:09 → 21:04:59 +08:00）＋並排
**run_id**: `20261025-210316-icpe-p12-x5-le32k`、`…-210326-icpe-p12-x4f05-sweep`、`…-210449-icpe-p12-x4f05-an`（exit 0）；並排 `20261025-215527-icpe-p12-compare`（head node，`code/icpe/p12_compare.py`，exit 0）→ `results/icpe_h200/p12/p12_compare.{csv,json}`
**關鍵數字**〔模擬，H200 常數〕: 15 項判定字 **14 項和 3090 分支一致**；不一致：X4 f×0.5 的 p90（只報告；H200 可能、3090 死路）。登記的 p99 兩邊都死路。表：`docs/h200/icpe/P1-2_cpu_symmetric_reruns.md` §6。

---

## ICPE-GUARD — gpu_guard 在 nvidia-smi 逾時後報 CLEAN（2026-10-25，stage3 session 發現；我核對＋補修 ICPE 用的路徑）
**事件**：2026-10-25 21:44:46 +08，hgx-cn01 上 nvidia-smi 全節點卡住 >20 s。舊的 `GpuWatcher` 監看執行緒遇到 `SmiUnavailable` 就結束，之後不再取樣；`__exit__` 補的那次取樣成功 → `gpu_guard.json` 照樣 CLEAN（違反 CLAUDE.md 規則 7）。
**核對**：`grep -l SmiUnavailable $TIARA_RUNS/*/stderr.log` → 只有 4 個 run：`20261025-210417-icpe-p11-gen-llama`（ICPE，品質，分數不受影響；P1-1 §7.1 已註明）和 stage3 的 3 個。**ICPE 所有量時間的 run（P0-1 S1、(a)、(b)，10/22）以及 tiara-runs 裡其他 H200 run 都沒有這個訊號** → 不用回頭重判。
**修正**：
1. main `6c450c9`（stage3 session）：`GpuWatcher` 每次 `SmiUnavailable` 記進 `smi_failures`、繼續取樣；`contaminated`／`verdict()` 把失敗算成 UNOBSERVED（髒）。h200 已 fast-forward 合進來。
2. **我補的**：ICPE 量時間用的路徑多半不讀 `watch.contaminated`，而是自己用 `intruders`＋`started_clean` 算（要排除自己的 busy 背景 pid），所以 (1) 傳不過去。這些地方都加上 `or bool(getattr(watch, "smi_failures", None))`：`m10_speed.py`（grid：p01brr、p2）、`m10_dqsweep.py`、`m7_write_policy.py`（gpu_contaminated＋CONTAMINATED 檔也記 smi_failures）、`m16_xtime.py` 三處（gpu_loadpath 每列、run 級、wcost 計時 ctx：p11t）、`code/icpe/p11_time.py`（wcost）。`attn_bench.py`、`x1_gpu.py` 本來就用 `w.contaminated`，不用改。
   - 一旦失敗過，之後每一列都算髒（不知道缺口影響哪幾列，保守）。nvidia-smi 沒失敗時行為逐位元相同（式子多一個 False）。
   - 驗證：五個檔 py_compile 過；式子在「乾淨／失敗過／沒有這個屬性的舊 watcher／開始就髒」四種假物件上結果都對；讀 CONTAMINATED 檔的程式都只看存在或前 90 字。

---

## ICPE-P01BRR — job 22233（P0-1 (b) 照重疊重跑；量時間）
**狀態**: PASS　**執行時間**: 2026-10-25 23:09:43 → 23:15:41（+08:00），exit 0　**同節點**：只有別人的 job（runner 檢查過）；sacct：這段時間沒有我們自己的別的 job（22313／22314 是之後才開跑）
**run_id**: `20261025-230946-icpe-p01b-qwen-s4-fa3-rr`、`20261025-231221-icpe-p01b-qwen-g5b-fa3-rr`（exit 0；沒有 CONTAMINATED、stderr 沒有 SmiUnavailable；模型從 /tmp，同 revision）
**之後**：15:2xZ `scontrol update 21655 Dependency=afterany:22313:22314`（p01brr 結束後開跑的兩個 stage3 job；若它們照讓位規則退出就馬上滿足）→ `scontrol release 21655 21663 21670`。(b) 的定稿分析（`p01_tools.py analyze-b`＋`verdict`）還沒跑。
- **P0-1 定稿（head node，2026-10-26 00:34 +08）**：`20261026-003415-icpe-p01b-analyze-final`（exit 0）→ (b) **24 格 0 翻轉、成立**（排除重疊的 2 個 run、加入重跑的 2 個，Qwen fa3 每格 10 次）；`20261026-003427-icpe-p01-verdict`（exit 0）→ **P0-1 總判定成立**（a、b、c 都 True）。注意：重跑時 m10_speed 改寫了 `grid/qwen_{s4,g5b}_fa3/` 的 `summary.csv`、`meets.json`、`spec_grid.json`（換成重跑那次的），`restores.csv` 是附加；分析只讀 `restores.csv`（按 run_id 排除），舊版在 git 歷史（`ce34075`）。
- **量時間鏈（2026-10-26 00:3x +08）**：21655 p02 等 stage3 的 22356、22357（stage3 session 照「排滿」規則加的；它們開跑時會讓位）。hgx-cn01 的 8 張卡全被別人的 job 佔著（Priority），還沒有我們的 job 在跑。

## ICPE-P02 — job 21655（P0-2 GPU 重播：冒煙＋12 個 run；量時間）
**狀態**: PASS（模擬可信，三條判準都成立）　**執行時間**: 2026-10-26 00:48:04 → 04:06:29（+08:00），exit 0，git `cbe8aee`　**同節點**：只有別人的 job；sacct：這段時間沒有我們自己的別的 job（stage3 的 22356／22357 讓位，00:48:04 已結束）
**run_id**: 冒煙 `20261026-004808-icpe-p02-smoke`；正式 `20261026-004945-icpe-p02-256-O-BEL-r0`、`…-010551-…-256-O-PF-r0`、`…-012215-…-256-FREQ2Q-r0`、`…-013828-…-128-LRU-r0`、`…-015512-…-128-O-BEL-r0`、`…-021119-…-128-LRU-r1`、`…-022804-…-128-FREQ2Q-r0`、`…-024413-…-256-LRU-r0`、`…-030054-…-256-HINT-PF-r0`、`…-031720-…-128-HINT-PF-r0`、`…-033341-…-256-LRU-r1`、`…-035022-…-128-O-PF-r0`（全部 exit 0；`gpu_guard_*.json` 都 `contaminated: false`、intruders／smi_failures 空；沒有 CONTAMINATED、stderr 沒有 SmiUnavailable）
**之後（head node）**:
- `20261026-040738-icpe-p02-sim-real` **exit 1**：`FileNotFoundError: '/ssd8/hungwei/paper-hkv/datasets/mooncake/traces/conversation_trace.jsonl'`。head node 的 shell 沒設 `X1_TRACE_DIR`、`P3_CONST`。在 `load_P` 就失敗，沒有寫進 `sim_requests.csv`（行數 15,351 不變）。
- 照 10/22 sim-orig 的環境（`P3_CONST=h200 X1_TRACE_DIR=$HOME/datasets/traces M7_RES=… TIARA_RUNS=…`）重跑 → `20261026-040756-icpe-p02-sim-real` exit 0。S_real 12 個 run 附加到 `sim_requests.csv`（33,771 行）。
- `20261026-040806-icpe-p02-analyze` exit 0 → 判定 passed。
- 加了只報告的「含排隊」欄位後重跑 `20261026-040946-icpe-p02-analyze` exit 0，判定欄位逐值相同（P0-2 §9）。
**產出檔**: `results/icpe_h200/p02/{gpu_requests.csv.xz（原檔 4.8 MB gitignore；xz 880 KB，讀回逐值相同）, gpu_smoke.csv, gpu_guard_*.json, sim_requests.csv.gz（含 S_real）, p02_cells.csv, p02_pairs.csv, p02_signs.csv, p02_verdict.json}`
**關鍵數字**〔實測，縮小 k＝8：1,535 請求／467 回來請求，CPU 16／32 GiB，Llama-3.1-8B，SDPA〕:
- 回來請求平均 TTFT 實測÷S_orig：10 格 0.911–0.975（全在 [0.9, 1.1]）。
- 排序：S_orig 差 ≥5% 的 15 對全同；5 對 <5% 不判。
- headroom 正負號：8 個全同。O-PF 對 LRU：模擬 23.1%／19.9%，GPU 17.7%／16.0%。HINT-PF 對 LRU：20.0%／16.9% → 14.3%／12.9%。
- 對 S_real：0.966–1.013。兩次 LRU 差 0.13%。
- 回來請求平均排隊 1.19–1.26 s（TTFT 不含排隊，照登記）；含排隊時 O-PF 對 LRU 7.9%／4.6%（只報告）。
- 細節：`docs/h200/icpe/P0-2_x1_gpu_validation.md` §8。
**失敗與異常**: S_real 第一次的環境變數（上面）。量時間的部分無。
**與論文假設的差異**: X1 模擬在縮小 trace 上偏慢 2.5–9%（GPU 比模擬快），對 LRU 偏最多 → 模擬高估 headroom（GPU 上是模擬的 0.72–0.94 倍）。符號與排序不變。
**接著**: 21663 p2 等 hgx-cn01 的空卡（8 張都被別人佔著，Priority），之後是 21670 p11t。

## ICPE-P01-C 補：V2 對第二階段 §5.5（2026-10-26，只報告，只讀檔）
不跑新的模擬，只讀已有的兩個檔做算術：`docs/h200/report_stage2_h200/derived/whatif_recompute.json` 和 P0-1 §1 的表。
- f 的縮小倍數：V2 ×0.5710／×0.5678，§5.5 ×0.5750／×0.5731（Llama／Qwen）。
- 方向一致：重算變快 → 重算分到的工作變多。
- §5.5 估 CPU 層 Llama 會翻成 Cake，但 (b) 的 fa3 實測同一格是全部載入較快。vLLM 級的這一格沒有實測。
- 細節：P0-1 §7.4 最後一點。

## ICPE-P2 — job 21663（P2：G5b 在 CPU 層、GPU 忙，換 seed 310／311；量時間）
**狀態**: PASS（跑完、乾淨；判定＝**部分穩定**）　**執行時間**: 2026-10-26 05:58:53 → 06:04:56（+08:00），exit 0，git `0489306`　**同節點**：只有別人的 job；sacct：這段時間沒有我們自己的別的 job
**run_id**: `20261026-055856-icpe-p2-llama-fp8-L310`、`…-055944-…-llama-int4-L310`、`…-060031-…-qwen-fp8-L310`、`…-060113-…-qwen-int4-L310`、`…-060154-…-llama-fp8-L311`、`…-060245-…-llama-int4-L311`、`…-060332-…-qwen-fp8-L311`、`…-060415-…-qwen-int4-L311`（全部 exit 0）。乾淨度：
- `gpu_guard` 8 個都 CLEAN；
- `contaminated_gpu` 全 False，`kv_bad` 0；
- 沒有 CONTAMINATED，沒有 SmiUnavailable。
分析 `20261026-060616-icpe-p2-analyze`（head node，exit 0）。
**指令**: `sbatch --job-name=icpe-p2 code/r_h200_icpe.sh --part p2`（P2 §2 的 m10_speed grid，`M10_MEET=g5b`，CPU 層、busy、E0/E1/R25 × cake/claim/loadall × 10 次）
**產出檔**: `results/icpe_h200/p2/{<m>_<fmt>_cpu_g5b_L{310,311}/, p2_cells.csv, p2_verdict.json}`
**關鍵數字**〔實測〕:
- 原本選錯的 10 格：穩 4（Llama int4 E1、Qwen fp8 E1、Qwen int4 E1、Qwen int4 R25）、不穩 5、沒重現 1（Llama int4 R25）。照 §3 → 部分穩定。
- 對照 2 格這次兩個 seed 都 >1.05。
- 24 個（格 × seed）：全部載入都比 Cake 慢 1.52–3.52 倍；G5b 選全部載入 0.3–1.0。
- 細節：`docs/h200/icpe/P2_g5b_cpu_busy_reseed.md` §6。
**失敗與異常**: 無。
**與論文假設的差異**: ICPE_EVIDENCE E3 原本寫「穩定的平台差異」，不成立（部分穩定）。可以寫的是「每格都常選錯、多慢隨 seed 變」。
**接著**: 21670 p11t（afterany:21663，已滿足；等空卡）。

## ICPE-P11T — job 21670（P1-1 (T)：慢層的低秩＋INT4 對 INT4；量時間）
**狀態**: PASS（跑完、乾淨；判定＝**不成立**，`:cal` 6/8）　**執行時間**: 2026-10-26 06:12:29 → 06:19:10（+08:00），exit 0，git `64868d8`（含開跑前的 report-t 修正，P1-1 §8）　**同節點**：只有別人的 job；sacct：這段時間沒有我們自己的別的 job
**run_id**: `20261026-061232-icpe-p11t-smoke-llama`、`…-061250-icpe-p11t-lp-llama`、`…-061535-icpe-p11t-wc-llama`、`…-061709-…-smoke-qwen`、`…-061722-…-lp-qwen`、`…-061837-…-wc-qwen`、`…-061907-icpe-p11-report-t`（全部 exit 0）。乾淨度：
- `gpu_guard` 6 個都 CLEAN；
- `loadpath.csv` 48 列都沒有污染；
- 沒有 CONTAMINATED，沒有 SmiUnavailable。
**產出檔**: `results/icpe_h200/p11/{time/loadpath.csv, time/loadpath_smoke.csv, time/wcost.csv, time/gpu_guard_*.json, p11t_cells.csv, p11t_report.json}`
**關鍵數字**〔實測，32K〕:
- `:cal` 的全部載入 ÷ INT4 端到端最好（倍數＝INT4 ÷ 低秩）：
  - b32 閒 2.20–3.28 倍、NFS 級閒 1.14–1.73、NFS 級忙 1.48–2.44；
  - **b32 忙 0.57–0.63**（兩個模型都不成立）。
- 每題 SVD 版 5/8。
- GPU 忙時低秩的全部載入有約 0.71–0.77 s 的底（b32、CPU 層都是）。
- CPU 層忙慢 21–43 倍。
- 壓縮：每題 SVD 2.13–5.33 s（閒），`:cal` 0.042–0.131 s。
- 細節：`docs/h200/icpe/P1-1_xkv_int4_h200.md` §7.2。
**失敗與異常**: 無。
**與論文假設的差異**: 「低秩＋INT4 在慢層比 INT4 快」在 H200 要加條件（GPU 閒，或慢層慢到蓋過重建）。ICPE_EVIDENCE D4-H200、§F。

---

## ICPE 量時間鏈結束（2026-10-26 06:19 +08）
P0-1、P0-2、P1-1、P1-2、P2 都跑完。`docs/h200/ICPE_EVIDENCE.md` 沒有「待」的列，§F 列出要改寫或加條件的主張。stage3 的 s3-T-* 鏈（22271 起，afterany:21670）現在可以開跑。

---

## ICPE-T1-START — ICPE 改成只用 H200：CPU 層換成 H200 的常數，重跑受影響的模擬（登記 `docs/h200/icpe/T1_h200_cpu_tier.md`，2026-10-27T22:50:34Z，commit a9c11bd）

- **緣由**：使用者 10/28「這個新主張是給 ICPE 只會有一個平台 都h200」「好登記 ok」，層級常數選「只換 CPU 層」（SSD、NFS 維持限速模擬的裝置層）。原本「H200 常數」的模擬，CPU／SSD／NFS 都用 MI300X 的 `tier_params.json`（03 §R2）。
- **新常數**（`code/icpe/t1.py build`，run `20261028-065334-icpe-t1-build` → `results/icpe_h200/t1/tier_params_h200cpu.json`）：照 `code/m7_params.py` 的規則，用 H200 的 C0 `20261016-061254-m7h-c0` 與 duplex `20261016-061303-m7h-duplex`：cpu 讀 43.5888、寫 41.1687 GiB/s、k_read 1.002（MI300X：35.4159、35.5482、1.161）。local、nfs 逐位元照抄。和登記 §1 的值相同（build 內斷言）。
- **開關**：`P3_TIER=h200cpu`（`code/p3_platform.py`：TIER_PARAMS 換新檔、`explore_out` 改到 `results/icpe_h200/t1/explore/`；`code/s3/e2e.py` 的輸出改到 `results/icpe_h200/t1/e2e/`）。沒設＝原本的行為。commit e208ca4。
- **自檢**：`20261028-065511-icpe-t1-selftest` **PASS**（12 項：沒設時讀 MI300X 的值、輸出在原位；設了只有 cpu 改變；X5 的成本常數設不設開關逐值相同 → X5 不重跑；輸出都在 t1 底下）。第一次 `20261028-065354-icpe-t1-selftest` exit 1：只有 2d（X4）不過，原因是自檢的 dump 先 import 了 `x3_restore`（它會載入 `m7_write_policy`，那時 `M7_TIER_PARAMS` 還是 env_h200.sh 的值）；實際的 X4 由 `x4_sim` 當主程式、先設好再 import，不受影響。dump 改成先 import `x4_sim`、runner 也直接 export 新的 `M7_TIER_PARAMS` 當保險。是我的自檢程式的錯，沒有產生模擬數字。
- **報告程式**：`t1.py report`（commit 8d3eee7），在任何 T1 數字之前寫好；舊對舊測過：0 差異，B1／B2／B3／B4／B7／C1／C2 的 summary 重現 ICPE_EVIDENCE 的值。
- **compute node**：`env -u TMPDIR -u CLAUDE_CODE_TMPDIR sbatch --job-name=icpe-t1 -o $TIARA_RUNS/slurm-%x-%j.out code/r_h200_t1.sh --part t1` → **job 22762**（2026-10-28 07:0x +08 開跑，hgx-cn01；不量時間；ICPE 沒有在等的 job，`sched.py next-icpe` 空；配額 1356G／1536G）。步驟：自檢 → X1 V0／V1／V2（sweep、analyze、H1800、H3600）→ fullsim → X1 FreeInference（前 6 週）→ X3（validate、tables、leads、hide）→ X4（trace、validate、sweep、analyze）。
- **之後（head node）**：`p12_queue.py`（T1 的 X4）→ `t1.py report`；E2E-C 的 T1 版等現在那份 E2E-C（FreeInference 還在跑）跑完再跑（登記 §6）。

## ICPE-T1-FAIL-1 — job 22762 在 T1-X3-LEADS 失敗（2026-10-28 07:43；runner 設定錯，已修、從 T1-X3-EVENTS 重送 job 22772）

- **跑完的步驟**（06:58–07:43，都 exit 0）：T1-SELFTEST；T1-X1-V0／V1／V2 的 SWEEP、AN、H1800、H3600；T1-FULLSIM；T1-P12-VERIFY、X1FI-SWEEP、X1FI-AN、X1FI-AN-USER；T1-EX-NEED、EX-VERIFY；T1-X3-VALIDATE、X3-TABLES、X3-TABLES-RUN、X3-EVENTS。log：`$TIARA_RUNS/slurm-icpe-t1-22762.out`。
- **失敗**：`T1-X3-LEADS` 的第一個來源 `x3_leads.py consts`：
  `OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/work/ChihChungHsu/hungweilai/hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct-1M/snapshots/e28526f7bb80e2a9c8af03b831a9af3812f18fba'. Use `repo_type` argument if needed.`（`AutoTokenizer.from_pretrained` 找不到目錄，當成 repo id）
- **原因**：`x3_leads.py` 的 tokenizer 路徑＝`$TIARA_HF_HUB/models--…/snapshots/<固定 revision>`；`r_h200_t1.sh` 沒有像 `r_h200_icpe.sh` 那樣在 source env_h200.sh 之前把 `HF_HOME` 指到 hgx-cn01 的 `/tmp/hungweilai-models/hf-cache`，所以指到 /work 的 hf-cache（10/23 起是空的）。10/21 原本那次 X3 跑的時候 /work 的 hf-cache 還在。我的 runner 漏抄，不是模擬的問題。
- **修正**（commit 見 `fix(icpe-t1)`）：runner 開頭設 `ICPE_MODEL_ROOT`／`HF_HOME`（同 r_h200_icpe.sh），開跑時 `fetch_models.py check --models llama qwen`。revision 相同（Llama `4699cc75…`、Qwen `e28526f7…`，fetch_models 逐檔對過 HF 的雜湊）→ 輸入不變。
- **重送**：`env -u TMPDIR -u CLAUDE_CODE_TMPDIR sbatch --job-name=icpe-t1b -o $TIARA_RUNS/slurm-%x-%j.out code/r_h200_t1.sh --part t1 --from T1-X3-EVENTS` → **job 22772**（排隊中：帳號 4 卡滿；同使用者的 kc-la08-* 是品質 run、不計時，可以同節點）。T1-X3-EVENTS 會把舊的事件目錄改名、重建。

## ICPE-T1 — CPU 層換成 H200 的常數：B、C 節判定字全部不變（jobs 22762／22772／22788＋head node 報告，2026-10-28 06:58–08:20 +08）

**狀態**: PASS（照登記跑完；E2E-C 的 T1 版還沒跑，等登記的那份 E2E-C 跑完）
**執行時間**: 2026-10-28T06:58:20+08:00 → 08:20:16（compute node 三段＋head node）
**run_id**: 自檢 `20261028-065820-icpe-t1-selftest`；X1 `20261028-065829`～`…-073308-icpe-t1-x1-*`（V0／V1／V2 各 sweep、an、h1800、h3600）；fullsim `…-073312-icpe-t1-p02-fullsim`；X1 FreeInference `…-073351`～`…-074221-icpe-t1-p12-x1fi*`；X3 `…-074241`～`…-081248-icpe-t1-x3-*`；X4 `…-081502`、`…-081741`、`…-081922-icpe-t1-x4-*`；排隊拆解 `…-082005-icpe-t1-p12-queue`；報告 `…-082016-icpe-t1-report`
**指令**: `env -u TMPDIR -u CLAUDE_CODE_TMPDIR sbatch --job-name=icpe-t1 -o $TIARA_RUNS/slurm-%x-%j.out code/r_h200_t1.sh --part t1`（22762）、`… --from T1-X3-EVENTS`（22772）、`… --from T1-X4-SWEEP`（22788）；head node `source code/env_h200.sh; export P3_CONST=h200 P3_TIER=h200cpu TIARA_ALLOW_NO_SLURM=1; m7run icpe-t1-p12-queue python code/icpe/p12_queue.py --cfg results/icpe_h200/t1/explore/x4/x4_verdict_cfg.csv --sweep $TIARA_RUNS/20261028-081741-icpe-t1-x4-sweep/x4_sweep.csv --out results/icpe_h200/t1/p12/x4; m7run icpe-t1-report python code/icpe/t1.py report`
**產出檔**: `results/icpe_h200/t1/`；並排 `results/icpe_h200/t1/t1_compare.{csv,json}`
**關鍵數字**〔模擬〕（舊＝MI300X 的 CPU 層 → 新＝H200 的 CPU 層）:
- **判定字全部不變**：X1（V0／V1／V2 × 原登記／H1800／H3600）9 個「有看頭」；P0-1 (c) 成立；X1 FreeInference 死路；X4 的 30 個判定相同；B4 的 D1、D2 同結論；X3 原登記／加 FreeInference「有看頭」。
- B1（conversation、SSD、閒、Cake，128／256 GiB）：O-PF 48.9／44.6% → 50.5／46.5%；O-ORD 25.4／27.0% → 27.65／31.0%；O-DROP 27.0／32.8% → 27.6／34.9%。
- B2：O-DROP 最大 45.9%（48 格 ≥15%）→ 47.7%（58 格）；截尾 H＝3,600 仍 0.0%（0 格）；O-PF 91.8 → 92.4%、O-ORD 99.9 → 99.9%。
- B3（X1）：FreeInference 0.28% → 0.30%（使用者發起 1.48% → 1.52%）。X5（2.44%）不受影響、沒重跑。
- B4：D1 中位 0.983 → 0.983（最小 0.846 → 0.837；可判 262 → 264 格）；D2 43／63 → 44／65。
- B7：X1 ≥15% 的格 167／168／169 → 190／192／189。
- C1：36／36、37／54，NFS 級最好的格 98.74% → 98.74%。
- C2：HINT-PF 對 LRU 5.6–47.8% → 6.9–49.1%；O-PF 對 LRU 5.5–52.4% → 7.0–54.2%；HINT-PF÷O-PF 中位 SSD 0.939 → 0.933、b32 0.949 → 0.941、NFS 1.117 → 1.118；>1 的格 6 → 4。
- 跨門檻（rule＝min，0.05／0.15）：271 列，全部往上跨；逐格在 `t1_compare.json`。
**失敗與異常**: ICPE-T1-FAIL-1（tokenizer 路徑，runner 的錯，已修）；`T1-X4-VALIDATE` exit 1——V1 54/160（參考值是舊常數算的；沒設開關時 160/160），V2–V4 過 → 偏離 1（`docs/h200/icpe/T1_h200_cpu_tier.md`），不當停止條件、從 sweep 接著跑。
**與原本的差異**: 照登記 §5，ICPE（只用 H200）改用新值；舊值＝「CPU 層以 35.4 GiB/s 限速時」的敏感度。沒重跑、列為限制：X5；CPU 層在 GPU 上被限速成 35.4 GiB/s 的 GPU 實測 run（A1'、A1''、A2、A3、B6、C2 的 GPU 縮小重播）；SSD、NFS 級。`docs/h200/ICPE_EVIDENCE.md` 開頭加了一行指向 T1 的結果。

## ICPE-T1G-START — CPU 層換成 H200 的常數，重量 GPU 實測的 run（第一批；登記 `docs/h200/icpe/T1G_gpu_rerun.md`，2026-10-28T08:25:20Z，commit dd43fcb）

- 使用者 10/28：GPU 不要空著，「兩批都排，T1 先」；和 kc-\* 共存「照規則，等 kc-\* 空檔」。
- CPU 前置（head node，T1 常數）：P02 selftest `20261028-162428-icpe-t1g-p02-selftest` 9/9 PASS；S_orig `20261028-162505-icpe-t1g-p02-sim-orig`（開跑）。
- 送出：job **22882** `icpe-t1g`（`env -u TMPDIR -u CLAUDE_CODE_TMPDIR sbatch --job-name=icpe-t1g code/r_h200_t1g.sh --part all`；A1'' → A1' → A2 → P02，原本的時間合計約 10 小時；`--time 1-12:00:00`）。送出時 PENDING（Priority）。
- 排程：每一步開跑前等節點上沒有我們自己的別的 job；跑完用 sacct 查重疊（`results/icpe_h200/t1/gpu/overlap.csv`），重疊就重跑、最多 3 次（登記 §4）。

## ICPE-T1G2-PREP — A3 的第 1 輪模擬在 T1 常數下重選存活設定（head node，2026-10-28 16:31–16:36；登記 `docs/h200/icpe/T1G2_round1_rerun.md`，2026-10-28T08:30:41Z，commit 1833593）

- **第一次是我設定錯，作廢、沒有看輸出**：`20261028-163102-icpe-t1g2-r1-sim`、`…-163232-…-r1-sim-an` 沒有設 `M7_D2H_GIBPS`，`m7_sim.py` 退回 44.8（MI300X 的值）；原本的 H200 run 是 49.5（`r_h200_cpu.sh` 從 H200 C0 算的）。輸出被下一次覆寫。
- 重跑：`M7_D2H_GIBPS=49.5`（用 `r_h200_cpu.sh` 同一段程式從 `results/m7_write_policy_h200/calib_c0.csv` 重算，等於 49.5）`M7_TIER_PARAMS=results/icpe_h200/t1/tier_params_h200cpu.json M7_EXPLORE=results/icpe_h200/t1/gpu/a3`：`20261028-163333-icpe-t1g2-r1-sim`（`m7_round1_sim.py --model llama31_8b`）→ `20261028-163501-icpe-t1g2-r1-sim-an`（`m7_round1_analyze.py sim`）。
- **回歸**：同一條指令、不換 tier params（`20261028-163538-icpe-t1g2-r1-sim-regress`，輸出在 scratchpad）→ 和 git 的 `results/m7_explore_h200/r1_sim_llama31_8b.csv` 4,800 列逐值相同（只差 run_id、ts）。
- **存活設定〔模擬〕**：原本 3 個（11.6／hold／nfs／0.5、35.42／free／local／0.5、35.42／hold／nfs／0.25）→ T1 常數下 **2 個**：11.6／hold／doc／nfs／0.5（3/5 seed，g_nwt 中位 26.9%，best_wt S5L 5/5）、43.59／hold／doc／nfs／0.25（4/5，6.9%，S5c 4/5）。35.42→43.59／free／local／0.5 不再存活。
- 計畫：`python3 code/icpe/t1g.py a3-plan --dir results/icpe_h200/t1/gpu/a3 --tier-params …` → `a3_plan.tsv`。runner 加 `--part a3`（不在 `all` 裡；寫 `.new` 再 mv）。
- 送出：job **22886** `icpe-t1g2`（`--dependency=afterany:22882`，10 個 run，原本的速度約 5.5 小時）。
- 排程：22882 `icpe-t1g` 改成 `afterany:22876`（使用者 10/28：kv-cliffs 排好的 LA01 → LA03 → LQ2 先跑完再讓 Tiara）。

## ICPE-A4' — A4 的 H200 列（不用新的 run）

- Llama INT4 的 Step 2 在 H200 早就算過：10/18「δ\* 用 5% 的對照」（RUNLOG_H200 該節）的 `results/m10_precision_h200/analysis/sensitivity_delta005/compare_mi300x_path/int4/step2_llama_int4.json`，資料＝H200 封存的 `quality_sealed/int4mix_llama_*`。F25／M25／B25 對 R25（讀法 B、b32，閒／忙）−0.47～+0.08 個百分點、CI 都含 0 → stable_gap_1_2＝False → **K2**（MI300X ±0.33）。δ\* 只影響「不比 R25 慢」那一條，品質門檻是 ≥2 個百分點，所以和 δ\* 無關。
- 加進 `docs/h200/ICPE_EVIDENCE.md` 的 A4' 列。A4 不需要 GPU。

## ICPE-T1-E2E — E2E-C 的 T1 版：判定和登記版相同（E2E-NOGO）（head node，2026-10-28 16:07 → 2026-10-29 02:19 +08）

**狀態**: PASS（只報告；登記 `docs/h200/icpe/T1_h200_cpu_tier.md` §5）
**執行時間**: 2026-10-28T16:07:54+08:00 → 2026-10-29T02:19:00+08:00
**run_id**: `20261028-160754-icpe-t1-e2e-dcheck`；`20261028-160904-icpe-t1-e2e-llama-conv`、`20261028-160905-icpe-t1-e2e-qwen-conv`、`20261028-160906-icpe-t1-e2e-llama-fi`、`20261028-160907-icpe-t1-e2e-qwen-fi`；`20261029-021758-icpe-t1-e2e-report`；`20261029-021836-icpe-t1-e2e-curve`；`20261029-021851-icpe-t1-report`
**指令**: `source code/env_h200.sh; export P3_CONST=h200 P3_TIER=h200cpu TIARA_ALLOW_NO_SLURM=1; m7run icpe-t1-e2e-dcheck python code/s3/e2e.py dcheck; m7run icpe-t1-e2e-<model>-<trace> python code/s3/e2e.py run --model <llama|qwen> --trace <conversation|freeinference>`（四個在背景同時跑）；跑完 `m7run icpe-t1-e2e-report python code/s3/e2e.py report; m7run icpe-t1-e2e-curve python code/s3/e2e_curve.py --root results/icpe_h200/t1/e2e; m7run icpe-t1-report python code/icpe/t1.py report`
**產出檔**: `results/icpe_h200/t1/e2e/`；`results/icpe_h200/t1/t1_compare.{csv,json}`
**關鍵數字**〔模擬〕:
- 判定 E2E-NOGO：0／32 格過、ASK 0、NOT_MEASURED 8（Qwen × cprefill），m\* 全空——和登記版（`results/s3_h200/e2e/e2e_report.json`）相同。
- MPS 取捨（只報告）：不設上限的 MPS 對 K2 的 TTFT 改善 30.7–62.9%（登記版 32.9–63.0%），B 比 stream 少 0～6.5 個點（登記版 0～6.7）。
- `t1_compare`：除 E2E 各項補上值以外，和 10/28 那份相同。
**失敗與異常**: 第一次送出 `20261028-1608{22,23,24,26}-*` exit 127（我在 zsh 下用 `$var` 拆字串，沒有跑任何東西）；改用 bash 重送。
**與原本的差異**: 無（判定字相同）。
