# G-M12：先近似、後補精——Cake 的前段改用稀疏注意力重算，品質掉多少、重算的斜率降多少

> **性質**：判準登記（事先寫死）。來源：[mechanisms/04 M12](../mechanisms/04_recompute_scheduling.md)、[mechanisms/06 §2](../mechanisms/06_eval_gates.md)（G-M12：「品質下界 ≥ −2 且 斜率 ≤0.3 倍 → 過」）。使用者 10/25 選「G-M12 稀疏重算」。
> 程式：`code/s3/gm12.py`。品質不量時間（可以和別人同時跑）；速度量時間、獨佔，照 [05 §2.1](../05_poc_plan_h200.md)。

## 1. 稀疏重算的定義（MInference／FlexPrefill 那一類的 block-sparse prefill；我們自己的實作，不是那兩個的程式）
- 逐塊重算（和 Cake 的重算線同一個順序）。重算第 i 塊時，每層、每個 KV head 只看：block 0（sink）、block i−1（local）、block i 自己（causal），加上 block 1..i−2 裡分數最高的若干塊；分數＝這一塊 512 個 query、這一組 q head 的平均 q（RoPE 後）· 每塊 K 的平均（RoPE 後；重算出來的塊用重算後的 K）。看的塊數合計＝max(3, round(d·(i+1)))，d＝密度 ∈ {0.1, 0.2}。
- 實作：選中的 key 接成每個 KV head 一條序列，fa3 varlen（causal 以右下角對齊），計算量和選中的 key 數成正比。自檢：d＝1 時的 KV 和 KVModel fa3 prefill 的 KV 最大差 < 0.25（BF16 的幾個刻度），否則停。
## 2. 品質（TF）
- 判定格：A0 的 L128（Llama 128K）、Q128（Qwen-7B 128K）；題目＝A0 的 TF 題組（seed 71000、96 題）。
- 配置：E0（精確 KV）；前段比例 F ∈ {0.5, 1.0}（block 1..round(F·nb)−1 用稀疏重算；block 0 精確；其餘是原值＝Cake 從慢層載入的）× d ∈ {0.1, 0.2}。
- 量：TF 的 Δgold_lp＝配置 − E0（nats，答案 token 的 log 機率加總；同 A0、X4a 的定義），以題目為單位 bootstrap 10,000 次。
## 3. 速度（量時間、獨佔）
- 每個模型，128K（252 塊）的 context，重算第 i 塊（i ∈ {16, 64, 128, 192, 248}）的時間：dense（KVModel fa3）vs 稀疏 d；每點 1 次暖機＋6 次。斜率＝每塊時間對 i 的最小平方直線斜率；CI＝對每點的 6 次重抽 10,000 次。
## 4. 判準（事先寫死）
- **品質過**（某個 d）：兩個判定格在 F＝1.0（最壞的情況）的 Δgold_lp 平均的 95% CI 下界 ≥ −2（nats；06 §2 的「品質下界 ≥ −2」照字面用 TF 的 Δgold_lp）。
- **速度過**（某個 d）：兩個模型的稀疏斜率 ÷ dense 斜率 ≤ 0.3，且稀疏斜率 CI 的上界 ÷ dense 斜率 ≤ 0.3；任何一點 2σ > 5% → 先回報、不判定（05 §2.1）。
- **G-M12-PASS**：存在一個 d 同時品質過、速度過。否則 **G-M12-STOP**（M12 卡：品質下界 < −2 或斜率沒降到 ≤0.3 倍 → 停）。
- 另報：F＝0.5；「找答案」與「彙整」分開；稀疏重算實際讀到的 key 比例（keys_frac）。**解讀上的限制**：Δgold_lp 不是答對率；PASS 之後，主實驗要用生成答對率（≥288 題、ε＝2 pp）再確認。

## 登記
- 登記時間（`date -u`）：**2026-10-25T13:38:00Z**；commit 訊息開頭 `docs(s3-gm12): prereg`。登記時沒有任何 G-M12 的數字（程式只過了 CPU 自測）。

## 偏離（登記之後）

### 偏離 1（2026-10-25T14:23:30Z；實作，判準不變；G-M12 速度 還沒有任何時間數字）：量測期間加 GpuWatcher，被污染的 run 不寫進 results

- **改什麼**：`code/s3/guard.py`（RUNLOG S3-GUARD-2）。量測開始前開監看（開不起來 → 這個 run 直接停，不在看不到的情況下量）；結束時被污染（有外來 process，或監看期間 nvidia-smi 查詢失敗過，S3-GUARD-1）→ 結果寫到 run 目錄的 `<檔名>.CONTAMINATED`、留 `CONTAMINATED` 檔，**不寫進 results**，要重量；每列多 `own_gpu_intruders`（0／外來者數／`UNOBSERVED:<n>`）；report 再按 run_id 排除一次（`excluded_contaminated_runs`）。品質（TF）那一半不受影響（照品質規則：分數保留、逐列記爭用）。
- **為什麼**：CLAUDE.md §3（gpu_guard 必用；contaminated 的 run 不得寫進 results）。原本的程式只靠 runner 開跑時的獨佔檢查，量測期間沒有監看（ICPE session 10/25 提醒）。
- **影響的 run**：全部（都還沒跑）。

---

## 結果（2026-10-28 01:58，head node 報告；以上各節未改）

**判定：G-M12-STOP**（不需要 Q128 就判得出來）。
- **Q128 品質沒跑**：GPU 自檢在 default 設定下過不了（RUNLOG S3-GM12-FAIL-1；診斷 S3-GM12-DIAG-RESULT：差異全來自 fa3 的 split-KV 加總順序）；使用者 10/28 決定不另寫偏離、不跑（06 §4 決策表裡用到 M12 的 P1、P2 已被 G-M08 STOP 排除）→ Q128 品質 NOT_MEASURED。
- **為什麼仍判得出 STOP**：§2 的「品質過」要兩格都過、「速度過」要兩個模型都過。L128 的品質兩個 d 都不過，Qwen 的速度（2σ ≤ 5%、可判定）兩個 d 都不過 → 不論 Q128 品質如何，沒有任何 d 能過。
- **指令**：`source code/env_h200.sh; export TIARA_ALLOW_NO_SLURM=1; m7run s3-gm12-report python code/s3/gm12.py report` → `results/s3_h200/gm12/gm12_report.json`（run_id `20261028-015835-s3-gm12-report`）。報告程式改成三值判定（缺的格／太晃的點＝判不了；任何量到的不過＝不過），並補上 §2「2σ > 5% 不判定」的檢查（原本的程式沒做）；前一次 `20261028-015735-s3-gm12-report` 用的是過渡版（缺格一律 NOT_MEASURED），已被覆蓋。
- **資料**：品質 `raw/gm12q_L128_s0.csv.gz`（job 22304 的 run `20261027-005028-s3-gm12-q-L128-s0`，96 題，guard CLEAN）；速度 `raw/gm12s_{llama,qwen}_*.csv`（S3-GFIO 一節，guard CLEAN）。

| L128 品質（TF Δgold_lp，nats，96 題） | F＝0.5 | F＝1.0（判定用） |
|:--|:--|:--|
| d＝0.1 | −4.63 [−5.81, −3.48] | **−14.56 [−17.24, −11.93]** |
| d＝0.2 | −3.01 [−3.95, −2.12] | **−8.64 [−10.38, −6.92]** |

| 速度（ms／塊的斜率；比值＝稀疏 ÷ dense） | dense | sparse 0.1 | sparse 0.2 | 判定 |
|:--|:--|:--|:--|:--|
| qwen | 0.1741 | 0.0818（0.47×，2σ 0.7%） | 0.1625（0.93×，2σ 0.9%） | 兩個 d 都 > 0.3× → 不過 |
| llama | 0.2038 | 0.1774（0.87×，2σ 34.5%） | 0.3231（1.59×，2σ 26.4%） | 2σ > 5% → 不判定（只報告） |
