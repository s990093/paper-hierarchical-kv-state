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

（還沒有。）

## 7. 偏離登記

（還沒有。）
