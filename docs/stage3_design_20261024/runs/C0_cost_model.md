# C0：能外推的還原方式預測（只用 CPU、只用既有資料）

> **性質**：判準登記（事先寫死）。登記時間見文末。來源：[05 §6](../05_poc_plan_h200.md)（演算法 C、四種切法、GC-GO／PARTIAL）。
> **只用 CPU**（head node），資料全部是已經 commit 的還原量測；不跑新的 GPU。程式：`code/s3/c0.py`。

## 1. 資料（格子）
- 來源（都是 Restorer 預設的逐塊反量化 S0，或沒有反量化的全 BF16）：
  - `results/m10_precision_{h200,mi300x}/speed_{llama,qwen}/restores.csv.gz`：32K、sdpa，Cake（`cake`）對全部載入（`load_only`，所有塊都有存時＝全部載入）；
  - `results/m10_restore2_{h200,mi300x}/q2/*`（32K、CPU 層）、`q1/*_P0`、`q1/*_G5`（64K、128K、CPU 層）、`q4/*P0*`（32K）：Cake（`claim`／`cake` 且 meet_rule_used＝claim）對全部載入（`loadall`）；
  - `results/icpe_h200/p01/grid/*_g5b_{sdpa,fa3}`：32K、CPU／b32／NFS 層、sdpa 與 fa3。
- 排除：`smoke` 目錄、`contaminated_gpu` 為真或 `kv_bad`>0 的列、q4duty（duty 和微基準的忙不同）、S2／S4 等其他拆法的目錄、MPS 目錄。
- **格子**＝（平台, 來源, 模型, 長度, 格式, 配置, 下層, GPU 狀態, duty, attention kernel）；Cake 與全部載入各 ≥5 次才算；值＝t_restore 的中位數。最好＝兩者較小的。

## 2. 預測器（演算法 C；常數全部來自「測試那一格的平台、模型、kernel」的微基準，不用任何還原格子的時間）
- f(i)＝f₀＋α·i：那個平台、模型、kernel 的 calib-f（64 塊，32K）每塊中位數的最小平方直線；長度更長時直接外推。sdpa：`m10_precision_*/speed_*/bw.json`；fa3：`icpe_h200/p01/calib/speed_*/bw.json`。
- B：限速層用列上的 `bw_GiBps`；CPU 層用那個平台、模型的 `bw.json` 的 `real_pinned_h2d_GiBps`；NFS 層用列上的值。
- 每塊平均位元組 s̄＝全部載入那幾列的 bytes_read ÷ N；低精度塊的比例 q＝全部載入那幾列的低精度塊數 ÷ N（位元組帳，不是時間）。
- 每塊反量化時間 d：`m10_sweep_{h200,mi300x}/parts/parts_<model>.json` 那個格式、閒時的 `t_dequant_torch_ms` 中位數（S0 是 PyTorch 的寫法）。
- 忙時的倍數：m_c＝那個平台 calib_c5 在那個 duty 的 slowdown 中位數；m_l＝parts 忙／閒的 t_h2d 中位數比；m_d＝parts 忙／閒的 t_dequant 中位數比（parts 的「忙」＝各平台的標準 duty：H200 1.0、MI300X 0.7，和格子的忙相同）。
- 預測：T_cake＝min_b max(m_c·Σ_{i<b} f(i), (N−b)·(m_l·s̄/B＋m_d·q·d))＋c₀；T_all＝N·(m_l·s̄/B＋m_d·q·d)＋c₁；選小的那一個。
- **唯一從訓練格子學的**：c₀、c₁＝訓練格子上「實測 − 物理預測」的中位數（各一個數）。不做殘差模型（05 §6.2 第 5 點是選做，這次不做）。

## 3. 四種切法（各自報；c₀、c₁ 只用訓練格子）
1. **長度**：每個平台用 32K 的格子訓練，測 64K、128K。
2. **負載**：每個平台用閒的格子訓練，測忙的格子。
3. **平台**：用 MI300X 的格子訓練，測 H200（H200 的格子用 H200 自己的微基準常數）。
4. **kernel**：用 H200 sdpa 的格子訓練，測 H200 fa3 的格子（fa3 的 f(i)）。

## 4. 判準（事先寫死）
- 每個測試格子的選錯代價＝t_實測(選的)／t_實測(最好)−1。
- **GC-GO**：四種切法的平均代價都 ≤1%、最差都 ≤5%。
- **GC-PARTIAL**：只有部分切法過 → 寫清楚在哪裡失效。
- 對照（同一批測試格子）：G5／G5b 有量到的格子（它們實際量到的時間 ÷ 最好）；「永遠 Cake」、「永遠全部載入」。舊值：G5b 32K 平均 0.4%、64K／128K 2.7%（最差 20%）、H200 忙 11.5%、換 kernel 1.21→1.33（05 §6.1）。

## 登記
- 登記時間（`date -u`）：**2026-10-25T12:57:12Z**；commit 訊息開頭 `docs(s3-c0): prereg`。登記時沒有跑過評估（只建了格子表：352 格；排除污染列 320、和自己的 job 重疊的 run 的列）。
- 資料的時間點：這次用 10/25 已 commit 的量測（C0-v1）；ICPE 的 p01brr（重疊 run 的重跑）與 P2（新 seed）收齊後用同一支程式、同一個判準再跑一次（C0-v2），兩次都報，不挑。
