# P2　H200 才有的爭用差異：CPU 層、GPU 忙時 G5b 選錯，換 seed 還會選錯嗎？

> ICPE 補量 P2（[README](README.md)）。平台 H200（hgx-cn01），量時間。**§1–§5 是事先登記**，結果寫在 §6 以後。
> **登記寫於 2026-10-22T04:14:28Z（`date -u`），repo HEAD `b0f0fba`**；寫的時候沒有任何這個實驗的新資料（上表是 job 21109 的舊值）。
> 標記同 repo 慣例。

---

## 1. 問題

第二階段的還原第二輪（13 號 Q2，job 21109）在 H200 上：G5b（開始前用線上量到的每塊載入時間與第 0 塊重算時間模擬 Cake 兩條線，再和「全部載入」比）整體離最好 11.5%（`STAGE2_H200_VS_MI300X.md` §4.3；MI300X 0.4%），幾乎全來自 **CPU 層、GPU 忙**：H200 的「GPU 忙」是 duty 1.0，全部載入每塊多排一次隊（約 3.7 ms／塊），G5b 卻有 47–77% 的次數選了全部載入。

`results/m10_restore2_h200/report_13.json` 的 q2，G5b 在 CPU 層、GPU 忙的 12 格（G5b ÷ min(Cake, 全部載入)；選全部載入的比例）：

| 模型 | 格式 | E0 | E1 | R25 |
|:--|:--|:--|:--|:--|
| Llama | fp8_token | **1.305**（0.47） | **2.038**（0.60） | 1.006（0.30） |
| Llama | int4_kivi | 1.009（0.20） | **3.110**（0.60） | **1.988**（0.63） |
| Qwen | fp8_head | **3.447**（0.67） | **3.507**（0.77） | **3.437**（0.67） |
| Qwen | int4_kivi | **3.424**（0.60） | **3.480**（0.70） | **3.429**（0.57） |

粗體＝ > 1.05（「選錯」）的 10 格；另外 2 格（Llama fp8 R25、Llama int4 E0）當對照。原本每格是 3 個 launch（seed 300–302）× 10 次。

**要回答**：這個「H200 上 G5b 在 CPU 層忙時選錯」是穩定的平台差異（換 seed 還是錯），還是那一次的偶然（例如 G5b 的線上紀錄被某幾次量測帶偏）？論文的跨平台那一節要寫哪一個。

## 2. 做法（和 43c 的 G5b 同一條指令，只換 seed、只跑 CPU 層忙）

```
K=$(python code/m10_dqsweep.py env S0 --model <m> --fmt <F>)        # H200 的 K*（M10_SWEEP_DIR＝results/m10_sweep_h200）
env $K M10_MEET=g5b python code/m10_speed.py grid --model <m> --fmt <F> --base-fmt <fmt_of m fp8> \
    --bw-json results/m10_precision_h200/speed_<m>/bw.json --configs E0 E1 R25 --refs --lengths \
    --bw-names cpu --gpu-states busy --extra-modes loadall claim --launches 1 --reps 10 --launch0 <L> \
    --out-dir results/icpe_h200/p2/<m>_<tag>_cpu_g5b_L<L>
```

- 模型 × 格式：Llama fp8_token、Llama int4_kivi、Qwen fp8_head、Qwen int4_kivi（各模型 `fmt_choice.json` 的選擇，和 43c 相同）。
- **2 個新 seed**：launch0＝310、311（43c 是 300–302；R25 的擺法也跟著 seed 換）。每個 seed 一個 launch × 10 次。
- GPU 忙＝duty 1.0（`results/m7_write_policy_h200/busy_levels.json` 的 busy2，和 43c 相同）。程式、K*、切法都和 43c 相同（`M7_ATTN` 不設＝sdpa）。
- 順序：seed 310 的 4 組、seed 311 的 4 組。
- 和 43c 唯一的差：43c 同一個 process 先跑 GPU 閒、再跑 GPU 忙；這裡只跑忙。G5b 的線上紀錄以（層, GPU 狀態）分開（`Restorer.g5_key`），閒時的紀錄不進忙時的決定，所以判定用的東西相同〔判讀〕。

## 3. 判準（每格、每個新 seed 各算一次）

- 比值 r＝G5b（`mode=cake`、`meet_rule_used` 照實）的還原時間中位數 ÷ min(Cake〔`claim`〕中位數, 全部載入〔`loadall`〕中位數)，同一個 launch 內的 10 次。
- 每格：兩個新 seed 都 r > 1.05 → **穩（選錯可重現）**；只有一個 → **不穩**；兩個都 ≤ 1.05 → **沒重現**。
- **總判定**：原本選錯的 10 格裡 ≥ 8 格「穩」→ 「H200 上 G5b 在 CPU 層、GPU 忙時選錯」是**穩定的平台差異**；≤ 2 格「穩」→ **偶然**；其他 → **部分穩定**（逐格列）。
- 只報告：每格每 seed 選全部載入的比例、Cake 與全部載入各自的中位數（看「全部載入在 H200 忙時就是慢」是否每次都成立）、2 個對照格的 r、和 43c 原值並排。

## 4. 停損、失敗

- 指令失敗：完整錯誤記下、停下來回報。gpu_guard 不是 CLEAN 的 launch：不進判定，重量一次；第二次還污染就停。`kv_bad` 不是 0 → 停。
- 量時間：不和我們自己的任何 job 同時在 hgx-cn01 跑（排在其他量時間的 job 後面）。

## 5. 時間〔估計〕

每個 launch：載模型約 1 分＋G5b 的 profiler＋3 個配置 × 3 種方式 × 10 次 × 每次 0.05–0.25 s，約 2–4 分；8 個 launch 約 30–40 分。

---

## 6. 結果

**總判定：部分穩定**。原本選錯的 10 格裡，**4 格穩、5 格不穩、1 格沒重現**。
- 判定檔：`results/icpe_h200/p2/p2_verdict.json`，`problems: []`，run `20261026-060616-icpe-p2-analyze`。
- 逐格：`p2_cells.csv`。

### 6.1 執行〔實測〕
- **job 21663**：2026-10-26 05:58:53 → 06:04:56（+08:00），exit 0，git `0489306`。模型從 hgx-cn01 的 /tmp（pinned revision，`fetch_models.py check` 過）。
- **8 個 launch** 全部 exit 0：`20261026-055856-icpe-p2-llama-fp8-L310` … `20261026-060415-icpe-p2-qwen-int4-L311`。每個 launch 約 45 s，比 §5 估的 2–4 分短。
- **乾淨度**：
  - 8 個 `gpu_guard_*.json` 都是 CLEAN（started_clean、intruders 空、smi 失敗 0）。
  - `restores.csv` 的 `contaminated_gpu` 全 False，`kv_bad` 合計 0。
  - 沒有 CONTAMINATED，stderr 沒有 SmiUnavailable。
  - sacct：這段時間沒有我們自己的別的 job。
- **GPU 忙**：duty 1.0（`busy_*.json`：matmul 1.380 ms，約 447 次／s）。

### 6.2 判定（r＝G5b 中位數 ÷ min(Cake, 全部載入) 中位數，同一 launch 10 次）

| 模型 | 格式 | 配置 | 43c 原值 | seed 310 | seed 311 | 判定 |
|:--|:--|:--|--:|--:|--:|:--|
| Llama | fp8_token | E0 | 1.305 | 1.005 | 1.523 | 不穩 |
| Llama | fp8_token | E1 | 2.038 | 0.999 | 2.061 | 不穩 |
| Llama | int4_kivi | E1 | 3.110 | 3.098 | 3.100 | **穩** |
| Llama | int4_kivi | R25 | 1.988 | 0.993 | 1.030 | 沒重現 |
| Qwen | fp8_head | E0 | 3.447 | 1.001 | 3.434 | 不穩 |
| Qwen | fp8_head | E1 | 3.507 | 2.194 | 3.484 | **穩** |
| Qwen | fp8_head | R25 | 3.437 | 3.491 | 1.010 | 不穩 |
| Qwen | int4_kivi | E0 | 3.424 | 1.001 | 2.221 | 不穩 |
| Qwen | int4_kivi | E1 | 3.480 | 2.229 | 3.507 | **穩** |
| Qwen | int4_kivi | R25 | 3.429 | 3.463 | 3.445 | **穩** |
| 對照：Llama | fp8_token | R25 | 1.006 | 1.994 | 2.009 | （兩個都 >1.05） |
| 對照：Llama | int4_kivi | E0 | 1.009 | 1.528 | 1.537 | （兩個都 >1.05） |

### 6.3 只報告
- **全部載入在 H200 忙時每次都慢**：24 個（格 × seed）都是全部載入比 Cake 慢。
  - 全部載入的中位數 0.232–0.237 s，Cake 0.066–0.155 s，全部載入 ÷ Cake＝1.52–3.52〔算術，`p2_cells.csv`〕。
  - 和 43c 的「全部載入每塊多排一次隊」一致：64 塊 × 約 3.7 ms ≈ 0.237 s〔算術〕。
- **G5b 選全部載入的比例**：24 個（格 × seed）都在 0.3–1.0。換句話說，每格每個 seed 都有至少 3 成的次數選了較慢的那個。
  - Llama fp8 的 seed 311 三個配置都是 10／10。
  - 43c 原值是 0.20–0.77。
- **r 是「選錯有沒有過半」的二分**〔判讀，從上表〕：
  - 選全部載入的比例 ≤0.4 時 r 都 ≤1.03（中位數落在 Cake）；
  - ≥0.6 時 r ≈ 全部載入 ÷ Cake（1.5–3.5）；
  - 剛好 0.5 時 r 在中間（2.19、2.22、2.23）。
  - 所以登記的判準（中位數 r > 1.05）量到的是「10 次裡選錯有沒有超過一半」，而 G5b 選錯的次數本身每格都不少。
- **兩個對照格**：43c 的 r ≤1.05（選全部載入 0.30、0.20），這次兩個 seed 都 >1.05（0.8／1.0、0.7／0.6）。也就是說，43c 的「沒選錯」這兩格同樣不穩。
- **對論文的意思**〔判讀〕：
  - 不能寫成「穩定的平台差異（逐格可重現）」。
  - 可以寫：H200 上 CPU 層、GPU 忙（duty 1.0）時，全部載入每次都比 Cake 慢 1.5–3.5 倍，但 G5b 在每格都有 30–100% 的次數選了全部載入；選錯有沒有過半、中位數多慢，隨 seed 變。
  - MI300X 的 0.4% 是另一個平台的整體值，不在這次比較裡。

## 7. 偏離登記

（沒有。程式、指令、判準都照 §2–§3。）
