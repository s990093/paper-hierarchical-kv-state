# 第二階段摘要核實＋第三階段工具箱盤點（MI300X，2026-10-19）

> **做法**：只讀檔，沒有跑 GPU、沒有跑分析腳本、沒有改任何既有檔案或佇列。每個數字都附「檔案:行號」或 CSV 的列與 run_id。
> 估計值一律標出處；我自己推的標〔推論〕。這是 **self-review，不是 cross-model review**（CLAUDE.md §5）。
> **階段名詞**（`docs/phase1_20261008/02_story.md:105-112`、`results/RUNLOG_MI300X.md:8,336`）：
> - 第一階段＝M7：BF16、寫入時依位置放置（S5）vs 簡單策略，還原用 Cake（10/08–10/12）。
> - 第二階段＝M10：精度（FP8／INT4）依位置放，判定 K0／K1／K3／K2（10/15–10/16）；M11–M14 是它之後的探索（大掃描、長 context、兩層混合、還原第二輪、drop）。
> - 原本的「第三階段」是預測器（會不會回來、何時回來、GPU 多忙），見 `02_story.md:112`。

---

## §1　十條「第二階段既有觀察」逐條核實

**總結**：第 2–10 條**全部是第一階段（M7，寫入位置）的結果**，不是第二階段（精度）。第 9 條另有 10/10 方向探索 D4 的 trace 佐證。第 1 條的硬體兩階段都對，但第二階段用了**兩個**模型。

| # | 使用者寫的 | 出處（檔案:行號／CSV:列／run_id） | 原文數字與量測條件 | 判定 | 階段 |
|:--|:--|:--|:--|:--|:--|
| 1 | 硬體 MI300X、模型 Llama-3.1-8B | `docs/phase1_20261008/07_report.md:54-56`；`docs/precision_20261013/08_report_mi300x.md:29` | 第一階段：MI300X（gfx942、192 GiB）＋ Llama-3.1-8B-Instruct（unsloth），權重與 KV 都是 BF16。第二階段：**Llama-3.1-8B-Instruct ＋ Qwen2.5-7B-Instruct-1M**，32K。第一階段第 1 輪另外用過 LongAlpaca-7B、Qwen3-30B-A3B（`RUNLOG_MI300X.md:88-93`）。10/19 起有兩張 MI300X（`RUNLOG:893`） | **部分符合**：第二階段少了 Qwen | 兩階段 |
| 2 | Cake baseline 完成，28 張圖中 27 張趨勢符合原論文 | `07_report.md:21,164-181`；`results/m7_write_policy_mi300x/verdict.json`（A0：`"cond3_cake_not_slower_(2%tol)": false, "PASS": false`）；`summary_a0.csv:3-4`；run `20261008-133544-m7-a0` | 「28」是**格**（4 種長度 × 7 種頻寬），不是圖。三個趨勢條件：條件 1（7／7 頻寬）、條件 2（4／4 長度，只在兩端成立，中間不單調）、條件 3「Cake 不比只算或只載慢」27／28 格成立；例外 4K × 100 Gbps，Cake 78.1 ms vs 只載 73.8 ms。判準是 05 §6 事先寫的趨勢條件，**不是逐張對原論文**；和原論文只對了一個數（Cake Table 6 的 0.98×，`07:181`）。照判準 **A0 嚴格說未通過**（`07:21`）。每格 6 次 | **部分符合** | 第一階段 |
| 3 | BF16 固定，restore 維持 Cake baseline | `RUNLOG:8`（「BF16，Cake 還原」）；`07_report.md:56` | 第一階段：BF16 固定、Cake 還原。第二階段：精度正是變數（E0／E1／F25…），讀法 B 才是 Cake（`08:31`）；13 號另外量到真實 CPU 層「全部載入」比 Cake 快 14–55%（`RUNLOG:716`） | 第一階段**符合**；若當第二階段描述**不符** | 第一階段 |
| 4 | 32K 時 CPU restore 約 202 ms、NFS restore 約 2720 ms | `07_report.md:202-206`；`summary_a1.csv:237`（CPUall 0.2022 s，CI 0.190–0.268）、`:238`（S0 2.7198 s）；`summary_a0.csv:60`（NFS 參數、Cake 2.7227 s）；run `20261008-134526-m7-a1`、`-133544-m7-a0` | 條件：A1、Llama、32K、GPU 閒、「寫完才回來」（gap＝inf）、每格 6 次。**是 TTFT，不是純還原時間**：含新一輪 256 token（t_new≈75 ms）。「CPU」＝限速 35.4 GiB/s 的 CPU 層、Cake 仍重算 4 塊（meet＝4，t_rec 125 ms）；「NFS」＝**pinned 記憶體＋限速器**，參數取 NFS 實測最慢 ×0.9（0.331 GiB/s），不是真的讀 NFS；S0 重算 50 塊、載 14 塊。同條件但本地 SSD 那組的 CPUall 是 189 ms（`summary_a1.csv:117`）。第二階段的對照：Llama 32K、真實 CPU 層 E0 還原 0.095 s、TTFT 0.171 s（`speed_llama/summary_fp8.csv:97`）；全部放 NFS 級 BF16 還原 2.64 s（`08:377`） | **部分符合**（數字對，量的是 TTFT、NFS 是限速模擬） | 第一階段 |
| 5 | 特定測試中 S5 與全 CPU 延遲相同；S5 只有少量資料用 SSD | `07_report.md:205-206,228-229`；`summary_a1.csv:237,241` | NFS 參數、GPU 閒、gap＝inf、32K：S5 201.8 ms vs CPUall 202.2 ms；S5 寫入時 3／64 塊進 SSD 層（w_ssd＝3），還原時 0 塊從 SSD 讀（那 3 塊落在重算段，meet＝4）。**反例**：本地 SSD 參數同條件 219 vs 189 ms（+16%，`summary_a1.csv:117,121`）；NFS、GPU 忙 2、32K 是 546 vs 400 ms（+37%，`summary_a1.csv:197,201`；`07:229`），因為 b 是照「GPU 閒」算的 | **符合**（限那一格） | 第一階段 |
| 6 | 不同 write position 的延遲差異不超過約 0.3% | `07_report.md:25,35`；`summary_b_verdict.csv:8-10` | S5（寫入時依位置）vs S4+（延後寫入＋依位置逐出），B 主設定（8 session、NFS 參數、share 干擾、seed 0）：CPU 容量 25%／50%／100% ＝ +0.07%／+0.26%／−0.30%（6／6／3 次）。**所有 17 個設定是 −3.0%～+0.3%**（`07:25`）。另外注意：第二階段 Step 2 的「F25／M25／B25 對 R25 在 ±0.33 個百分點內」（`08:327-334`）是**答對率**，不是延遲 | **部分符合**（≤0.3% 只在主設定） | 第一階段 |
| 7 | 等寫入完成再返回，與對照組延遲接近；提早返回並非穩定受益 | `07_report.md:22,208-211`；`summary_a1_pairs.csv`（S3 vs S2a，各 24 格）；`13_advisor_update_20261012.md:28` | 「回來」指**使用者的下一個請求**，不是寫入 API 提早 return。比較的是 S3（一開始不寫前段）vs S2a（先寫再丟），最後留下的東西相同。寫完才回來（inf）：0／24 格 S3 顯著較快；寫入還在進行就回來（gap＝0）：6／24 格快 3.7–13%（CI 不重疊），其中 3 格超過 δ＝6.2%。原因：只要還有寫入在跑，讀就變慢，寫多寫少差不多（A2：本地 3%、NFS 0.2%，`07:249`）。小問題：`07:209` 寫「inf 時差距最大 9%」，但 `summary_a1_pairs.csv` 有一格 NFS 4K 閒 S3 比 S2a **慢** 35.9%（CI 不重疊），報告漏寫 | **部分符合**（語意要改寫） | 第一階段 |
| 8 | GPU busy 可使延遲增加約 13–66% | `07_report.md:23,213-226`；`summary_a1_pairs.csv:54-65,80-81`（S3 vs S0、gap＝inf） | **比較對象不是「忙 vs 閒」**，而是 GPU 忙時「一開始不存前段（S3）」比「全存（S0）」多等多少：CI 不重疊的格子 +13.2%～+66.4%（本地忙 2、32K 最大）。忙＝背景 matmul duty 0.4（重算慢 1.48×）／0.7（2.07×）（`07:133`、`busy_levels.json`），每格 6 次。`07:34` 寫「16K 以上」，但 §0 表（`07:23`）寫的才完整：忙 2 時本地全部長度、NFS 8K 以上都顯著（本地忙 2 的 4K +45.6%、8K +29.2%）。GPU 忙本身對 S0 的影響是另一組數（例：NFS 32K 閒／忙 1／忙 2 ＝ 2.72／3.57／4.59 s，`summary_a1.csv:238,158,198`） | **部分符合**（範圍對，比較對象錯） | 第一階段 |
| 9 | 延遲寫入可避免部分未再讀取的 SSD 寫入 | `07_report.md:348`；`summary_b.csv:68-69,74-75`；run `20261008-164801-m7-b-nfs-seed1`；`docs/research_20261010_directions/SUMMARY.md:30,83` | seed 1（8 個 session 有 5 個中途離開）、NFS 參數當 SSD 層、3 次：CPU 25% 時寫穿 S1 寫 20.0 GiB、延後 S4 寫 12.0 GiB；50% 時 20.0 vs 6.0 GiB。**TTFT 一樣**（1.071 vs 1.073 s）。D4（Mooncake trace，run `20261010-054225-d4-char`）：75–78% 的 block 寫了就沒人再讀；同樣的知識延到淘汰時才用，結果一樣 | **符合**（但沒有換到延遲） | 第一階段＋10/10 D4 |
| 10 | NFS 頻寬約 0.33 GiB/s，local SSD 約 6.98 GiB/s | `results/m7_write_policy_mi300x/tier_params.json`；`07_report.md:99-111`；run `20261008-130257-m7-c0`、`-130438-m7-c2` | 這是**限速器參數**（實測最慢 ×0.9，只有讀）：local 讀 6.98（寫 1.91）、NFS 讀 0.331（寫 0.601）GiB/s。原始實測（O_DIRECT、每 chunk 一個檔、3 次 × 64 個 64 MiB）：本地 SSD 讀 7,943–9,152 MiB/s、NFS 讀 377–447 MiB/s。NFS 同一天稍後有 500–830 MiB/s，所以 0.331 是「慢 NFS」（`07:123`）。第二階段 12／14 號沿用（`code/m10_tier2.py:50-55`） | **部分符合**（是參數，不是平均頻寬） | 第一階段（第二階段沿用） |

---

## §2　第二階段（精度）、長 context、M13／M14 的現有結論與佇列

**主線判定（07 凍結 2026-10-15T16:13:06Z；判定程式 `code/m10_analyze.py`）**
- K0（FP8）不成立：限速層全 FP8 比全 BF16 快，Llama 22.7–37.2%、Qwen 22.0–39.2%；真實 CPU 層 Llama 閒快 24.8%、忙慢 3.5%，Qwen 閒／忙慢 5.1%／12.6%（`RUNLOG:361-373,447`）。
- Step 1：Llama 全 FP8 對 E0 讀法 A −0.5 [−1.1, +0.2] 個百分點 → **K1 成立**；Qwen −32.8 → 不過；Step 2b：Qwen E1-INT8 +0.2 → **K3 成立**（問題在 FP8 的 K 格式，不在位置）（`RUNLOG:478-483`）。
- INT4：限速層快 36–57%；品質 24 格 ε 全過；只輸「真實 CPU 層、原寫法反量化」的速度 → 判「均勻不夠」→ Step 2 在 Llama 做（`RUNLOG:495-515`）。
- Step 2（Llama INT4）：F25／M25／B25 對 R25 的穩定差距都在 ±0.33 個百分點內（門檻 2）→ **K2 成立，主線停**：依位置決定精度沒有幫助（`RUNLOG:540-543`；`08:19`）。
- 換融合反量化 kernel（逐位元相同，只報告）後，真實 CPU 層 24 個速度檢查全過（`RUNLOG:665`）。
- 大掃描複賽 43：12 個選手 **0 個成立**（`RUNLOG:733-736`）。

**長 context 44c（RUNLOG 15.9，`RUNLOG:967-973`；表 `results/m10_longctx_mi300x/analysis/tables_longctx.md`；探索、不判定）**
- Llama 生成答對率 E0：32K 68.2、64K 66.7、128K 40.3%；全 FP8／INT8／INT4-KIVI 在三個長度都和 E0 差 ≤2 個百分點（`tables_longctx.md:9-11`）。
- Qwen E0 77.4／61.9／56.6%；全 fp8_head 39.5／17.0／1.5%（越長越壞）；INT8、INT4-KIVI 和 E0 差 ≤2.5（`tables_longctx.md:142-144`）。
- 25% 前／中／後對隨機：
  - Llama 三種主要格式都在 ±2.6 個百分點內（`tables_longctx.md:33-63`）。
  - Qwen：INT4-KIVI 在 ±3.1 內；INT8 在 128K 有 F +4.5 [+0.8, +9.4]。
  - **Qwen 的 fp8_head（本身就壞的格式）放哪裡差很多**：128K 時 F +19.5、M +28.0、B −9.5 個百分點，32K B +8.2，CI 都不含 0（`tables_longctx.md:166-197`）。**RUNLOG 15.9 只寫了 Llama，沒寫這一點**。〔推論〕格式本身傷品質時，位置才開始有作用；這和 14 號「丟 token 會真的傷品質，挑法才拉得開」（`14_match_drop_plan.md:159`）同一個邏輯。
- 傷害隨長度（↑＝變差且 CI 不含 0）：Llama 標 ↑ 的是 fp8e5_head、int4_token、nvfp4（`tables_longctx.md:17-29`；RUNLOG 15.9 寫的「INT2、MXFP4、int2h_g32」在表上沒有標 ↑）。Qwen 只有 fp8_head 標 ↑（128K−32K −17.2）；另外 6 種格式（fp8e5_head、int2h_g32、int4_token、int4h_token、mxfp4、nvfp4）三個長度的差都剛好等於 −E0，也就是答對率 0%，本來就全壞（`tables_longctx.md:150-162`）。
- 速度（只有 FP8，INT4 是 NOT_MEASURED，等 44ca）：128K b32 閒 E1 比 E0 快 37.1%（Llama）／38.6%（Qwen）；128K 真實 CPU 層閒 E1 反而快 14.3%／13.2%（`tables_longctx.md:120,124,253,257`）。
- **發現一個分析錯誤**：`tables_longctx.md` §5 的「32K・閒」列取到的是 **8K** 的列（例 Llama b16 閒 E0 0.15453 s＝`speed_llama/summary_fp8.csv` 的 `length=8192` 列；32K 實際是 0.57818 s）。原因：`code/m10_longctx_analyze.py:186-188` 沒有依 `length` 篩選，主線 summary 有 8K／16K／32K 三種長度，「閒」的 8K 列在檔尾把 32K 蓋掉。「忙」的列只有 32K，所以是對的。RUNLOG 15.9 還沒拿 §5 下結論，但寫報告前要修。

**M13（12 號兩層混合，stage 44，`RUNLOG:810-815`；`08:373-385`）**
- 慢層存 INT4 比 BF16 快（NFS 36／36、SSD 31／36 個比較）。NFS 級「舊的先降」最快、「新的先降」最慢（Llama 降最新 25% 到 NFS 要 2.64 s，等於全部放 NFS）；SSD 級沒有一致方向。原本預期舊 25% 會被重算蓋掉，結果錯了：Cake 只重算 3–5 塊，16 塊大多還是要從慢層讀。
- IO 牆用實測更新：SSD 級全 INT4 比推算慢 37–39%；損益兩平點 Llama・忙 18.1、Qwen・閒 29.9、Qwen・忙 17.6 GiB/s。

**M14（13 號還原第二輪，`RUNLOG:833-870`）**：GPU 忙時「全部搬完一次拆（S4）＋全部載入」比原本快 21–56%，背景 duty 0.4→0.9 幾乎不變；Cake 隨背景暴增。會合規則 G5b 對每格最好只差 0.4%（幾何平均 1.004）。拆融合進 attention：我們的 Triton kernel 每 token 多花時間；vLLM 0.28 的 FP8 KV decode 反而快 4–29%。

**14 號 drop（DS1／DS2／D3／M1／M2，`docs/precision_20261013/14_match_drop_plan.md`）**：**一個都還沒在 GPU 跑**；`results/m10_drop_mi300x/`、`m10_match_mi300x/` 都不存在。只有 CPU 測試過（`tests/match_drop_cpu.py` 126 項、`restore2_cpu_shim.py` 含 drop，`RUNLOG:919`）。

**佇列現況（2026-10-19 13:06Z 讀；佇列在另一個 session 的 `/tmp/claude-0/-mlsteam-workspace/31610192-*/scratchpad/jobs/`，只讀）**

| 卡 | stage | 內容 | 狀態 | 估計時間（出處） |
|:--|:--|:--|:--|:--|
| main（a6:00.0） | 44ca | 長 context 速度剩下的 4 格 INT4（Llama／Qwen × 128K／64K，2 launch × 5 rep），兩卡獨佔 | 10:47:21Z 開跑，跑中 | 44a 全部 8 格原估 6 h（`10_longctx_plan.md:89`）；FP8 四格實際 16:35→23:43Z ＝7.1 h（`runner.log`、run 目錄時間）→〔推論〕INT4 四格同量級，約 10/19 18Z 前後 |
| main | 44cb | 兩卡干擾實驗（過了才立 `XGPU_OK`） | 排隊 | 檔案裡沒有估計 |
| main | 44cz | CPU：pick-c → 寫 `44d_rs_longctx.sh` | 排隊 | 幾分鐘（CPU） |
| main | 44d | 13 號 Q1：64K／128K 的還原方法（P0、S2k8+G4、P3、G5b、nodq） | 還沒寫出 | 2.2 h（`13_restore2_plan.md:126`） |
| main | 44e0→44e5、44ez | 14 號：冒煙 → DS1 → DS2 → D3 → M2 → M1 → 表 | 排隊 | 冒煙 0.3、DS1 2–3、DS2 1.5–2、D3 0.5、M2 2、M1 1 h；合計 8–9 h、上限 11 h（`14_match_drop_plan.md:131,227-228`） |
| main | 70 | vLLM 0.28 只算 TTFT（預設切法 vs 512） | 排隊 | 檔案裡沒有估計 |
| main | 75 | 上游 RULER CWE 用 vLLM 重查 | 排隊 | 檔案裡沒有估計 |
| helper（86:00.0） | 55 | T1、LongBench（FP8 階，只報告） | 10:28Z 開、10:47:22Z 被搶先（10 個 job 開了 4 個），之後照題號續跑 | 檔案裡沒有估計 |
| helper | 60 | 逐 block 敏感度地圖 | 排隊 | 原設計（50 題）約 1 h GPU〔推論〕（`03_plan_h200.md:417`）；stage 60 的設定不同 |
| helper | 88 | INT4 的 T1、LongBench（只報告） | 排隊 | 檔案裡沒有估計 |

- 45、46 已跑完（`runner_g0.log`：06:42Z、10:28Z），輸出在 `results/m10_precision_mi300x/quality_sealed/fp8mix_*`、`int4mix_qwen_*`，**RUNLOG 和 08 還沒有結果段**。
- 只要 main 跑兩卡獨佔的 stage（44ca、44cb、44e0、44e3–44e5、70、44d），helper 就停（`WANT_BOTH`）。44cb 沒過的話，55／60／88 只能在這些 stage 之間的空檔跑。

---

## §3　可重用的工具箱（第三階段會用到）

### 3.1 KV 還原／載入
- **限速層 `Tier`**（`code/m7_restore_harness.py:37-96`）：資料全放 pinned 記憶體，時間＝`c_s + bytes/read_Bps`；`mode="fifo"`（讀寫共用一條隊）或 `"share"`（寫入進行中讀取慢 `k_read` 倍，k 由第一階段 A2 真實裝置校準：CPU 1.16、本地 SSD 3.31、NFS 1.67）；背靠背讀取 <0.5 ms 不算閒置（`:33,60-62`，C7 自檢修正）。真的做 H2D，等到 max(限速 done, 真實複製完成)。
- **`Restorer`**（`:106-601`）：
  - `restore(ids, n, host, loc, mode)`，mode＝`cake`／`load_only`／`compute_only`（`:507-601`）。
  - `loc[i]` 是每個 block 自己的 Tier → 可以混合 CPU／慢層（12 號 tier2 就是這樣用）。
  - `host[i]` 可以是 BF16 pinned tensor 或 `m10_codec.Enc`（低精度：限速器照實際位元組計時，資料到齊後在 I/O stream 上反量化，`:195-219`）。
  - `set_options(unit, sched, group, compute_prio, meet, io_prio)`（`:141-157`）：unit＝還原單位（≥32 token）；sched＝serial／pipe（S1 邊搬邊拆）／group（S2 多塊一起拆）／defer（S4 全搬完一次拆）／defer_meet（S5）／nodq（只搬不拆）；meet＝claim（Cake）／estimate（G4）／none（全部載入）／g5／g5b（`:469-496` 模擬兩條線再決定）。
  - `drop`（`:139,400-403,571-574`）：載入線跳過被丟的 block；落在重算段的還是會被算回來。**`load_only` 模式不看 drop。**
  - `new_turn`（`:603-615`）：新一輪 prefill＋第一個 token。`write_boundary`（`:620-629`）：演算法 1 的分界 b。
  - 規則：一個 process 只建一個 Restorer（`RUNLOG:243`）；每格第一次做 KV 逐位元檢查（`kv_bad`）。
- **`code/m10_speed.py`**（速度格子）：
  - 配置（`:67`、`layout()` `:116-145`）：E0 全 BF16；E1 全低精度；E1s 只有 block 0 是 BF16；E1sr block 0 和最後一塊是 BF16；F25＝block 1–16、M25＝24–39、B25＝48–63、R25＝從 1–63 隨機 16 塊（seed＝launch×1000＋rep）。n 不是 64 時照比例。另有 `<底>-D<F|B|R><佔比>`（drop，`drop_set()` `:98-113`）。
  - 層：b16／b32／b48＝**讓 E0 的 Cake 會合點落在 64 塊裡第 16／32／48 塊的限速頻寬**，B＝(64−b\*)×chunk_bytes÷Σ_{i<b\*} f(i)（`:4-6`）；Llama 5.27／1.47／0.42、Qwen 2.77／0.76／0.22 GiB/s（`RUNLOG:349`；`speed_<m>/bw.json`）。cpu＝不限速、真實 pinned H2D（Llama 49.9、Qwen 48.6 GiB/s）。nfs＝0.331 GiB/s（44e0 才會寫進 `results/m10_match_mi300x/bw_<m>.json`）。`m10_tier2.py:44,50-55` 另有 ssd 6.98 與 w1／w3／w12／w20。
  - CLI（`:770-813`）：`grid --model llama|qwen --fmt F --base-fmt G --bw-json --bw-names b16 b32 b48 cpu [nfs] --gpu-states idle busy --busy-key busy2 --configs … --refs … --lengths … --launches 3 --reps 10 --new-tokens 256 --extra-modes loadall claim --extra-tiers cpu --launch0 N --main-length L --blocks-out`；另有 `calib-f`、`noise`、`k0`、`k0-verdict`、`summary`。
  - 環境變數（`:149-156,365-368`）：`M10_DQ_KERNEL／M10_DQ_SCHED／M10_DQ_GROUP／M10_DQ_PRIO／M10_UNIT／M10_MEET／M10_RS_PRIO`、`M10_CODEC_X=1`（新格式）、`M10_DEQUANT_FAST=1`（融合 kernel）、`M10_BUSY_CU_MASK`。方法代號 P0–P5 由 `code/m10_restore2.py env <P> --model --fmt` 印出（`m10_restore2.py:33-35`）。
- **兩層混合**：`code/m10_tier2.py`（`grid/smoke/summary`；配置 `T<old|new|rnd><25|50|75>-<bf16|int4>`）。

### 3.2 GPU 忙碌模擬
- `code/m7_busy.py --duty D [--period-ms 20] [--n 8192]`：每 20 ms 週期做 D×20 ms 的 BF16 8192² matmul、其餘睡；印 `READY`／`DONE n= s=`（背景吞吐）（`:1-53`）。
- 校準：duty 0.4＝busy1（重算慢 1.48×）、0.7＝busy2（2.07×）（`results/m7_write_policy_mi300x/busy_levels.json`；run `20261008-130834-m7-c5`、`-130956-m7-c5b`）。第二階段的「忙」＝busy2。
- `m10_speed.Busy`（`m10_speed.py:160-215`）包住它：可加 `HSA_CU_MASK`（P4a／P4b 留 CU）；`M10_BUSY_TEARDOWN_PAUSE=1` 收尾時暫停 gpu_guard，避開 amd-smi 殘影誤報。

### 3.3 token／block drop
- `KVModel.key_keep`（`code/m7_model.py:63-66,110-113,124-141`）：`[max_len]` bool（全部層共用，page／block 粒度）或 `[L, H, max_len]`（每層每個 KV head）；False＝attention 看不到（等於從 cache 拿掉，RoPE 位置不變）。None 時路徑逐字不變。
- `Restorer.drop`（見 3.1）只影響載入線。
- `code/m10_drop.py`（kvpress 0.5.5 的分數定義**重寫**，不是直接用它的程式，`:1-48`）：
  - 挑法：STREAM、NEW、RND0–2、SNAP、TOVA、H2O（8 個觀察窗近似）、PYR、ADA、EXPATT、KNORM、KEYDIFF、QRY（還原時用第 1 輪問題）、ORC（作弊上限）（`:42-48,127-254`）。
  - 粒度 T（每層每 head 的 token）／P（16 token）／B（512 token，唯一能換成還原速度的）；佔比 10／25／50／75／90%。
  - 子命令：`screen`、`screen-report`、`pick`、`confirm-report`、`speed-report`（`:686-713`）。`screen` 沒有 `--n-blocks`，只能 32K。
  - 狀態：CPU 測試過，**GPU 還沒驗**（44e0 冒煙排隊中）。
- 精度放法的挑法（ATT／AXE／ERR／ORC，`code/m10_sweep.py`）在 41／43 用過，GPU 驗過。

### 3.4 量化格式與反量化 kernel
- `code/m10_codec.py:191`（主線，07 凍結）：bf16、fp8_head／channel／token、fp8k0_head／channel／token（K 先減平均）、int8_token、int4_token、int4_kivi。e4m3fn 自己先截 ±448（torch 2.10／2.12 ROCm 超出會變 NaN）。
- `code/m10_codec_x.py:48`（大掃描）：fp8e5_head／token、mxfp4、nvfp4、int2_kivi、int4h_token、int2h_g32，加 `kv-<K>-<V>`（K8V4、K4V2）。MXFP4／NVFP4 對 torchao 0.18 逐值相同（`RUNLOG:570-571`）。
- 融合反量化（Triton）：`code/m10_dequant_fast.py`（fp8_\*、int8_token、int4_token、int4_kivi；`decode_fast` `:145`、`decode_multi` `:292`、`decode_variant` `:365`、`install` `:396`），GPU 逐位元相同（`RUNLOG:597,604`；int32 位址溢位已修）；`code/m10_dequant_fast_x.py:22`（新格式，不含 nvfp4、int4h、int2h），GPU 24／24 逐位元相同（`RUNLOG:707`）。
- 拆進 attention 的對照組：`code/m10_attn_q.py`（只能當對照，比讀 BF16 慢）；vLLM 的 FP8 KV：`code/m10_vllm_kvdecode.py`。
- 選定格式：Llama fp8_token／int4_kivi；Qwen fp8_head／int4_kivi（`results/m10_precision_mi300x/fmt_select/fmt_choice.json`）。

### 3.5 品質評測
- `code/m10_tasks.py`（只用 tokenizer，`:1-70`）：
  - T1＝RULER niah_multikey_3（整片海都是干擾針）、T2＝niah_multivalue（一個鍵 4 個值）、T3＝vt（4 跳變數鏈）、T4＝HotpotQA（兩段 gold 段落）、T5＝cwe（全文詞頻，沒有位置帶）（`:96`）。
  - 計分：string_match_all／part、qa_f1（`:154-189`）；生成長度 run 時 T5 120、其他 64（`m10_quality.py` 檔頭；上游值 `:111`）。
  - context 剛好 n_blocks×512；位置帶 前 1–16、中 24–39、後 48–63（`:82`）；`bands(n_blocks)` 照比例縮放（`:85-94`）。兩輪：第 2 輪歷史固定成標準答案。
  - 難度參數：`results/m10_pilot_mi300x/e0_tune/CANDIDATE_PARAMS.json`。HotpotQA：`/mlsteam/data/tiara/datasets/hotpotqa/…parquet`（`:114-115`）。
- `code/m10_quality.py`：`run`（配置＋@格式、讀法 A 全載入／B 重放 meets.json、兩輪、`--shard`、逐題續跑；`SW-*`、`DS-*` 配置）、`sensmap`、`selfcheck`、`tune`（`:700-745`）。
  - KL＝在標準答案 token 上 teacher forcing，KL(E0‖X) 對答案 token 平均（`:355-365,387-396`）；另記 gold_logprob、same_as_e0。
  - **`Answerer` 固定 32K**（`:58-59`），64K／128K 要走 `m10_sweep.py screen --n-blocks 128|252 --gen-all --gen-pos`（`m10_sweep.py:502-504`）。
- 判定：`code/m10_analyze.py` `eps_check`（`:141-160`）：對 E0 的配對答對率差，題目為單位 bootstrap（10,000 次）95% CI 下界 > −2 個百分點，且每個任務族、每一輪點估計 ≥ −5（`:44-49`）。
- LongBench：`code/m10_longbench.py`（6 任務 × 50，官方截斷）。

### 3.6 vLLM
- **版本要分清楚**：
  - venv `tiara`（harness 用）裡裝的是 vLLM 0.19.1+rocm722，editable 指到 `/mlsteam/workspace/src/vllm`（commit b1388b1），**但 harness 不用 vLLM**。
  - **所有 vLLM 實驗都用 venv `tiara-v028`**：vLLM 0.28.0+rocm722（editable 指到 `/mlsteam/workspace/src/vllm-v0.28.0`）、torch 2.12.1+rocm7.2（`07_report.md:55`、`RUNLOG:868`、佇列 70／75 的指令）。
  - 另有 `tiara-v022`（只有 torch）。
- 70（`code/m10_vllm_ttft.py`）：`LLM(enable_prefix_caching=False)`，預設切法 vs `max_num_batched_tokens=512`，`max_tokens=1`、隨機 token → 純重算 TTFT。Qwen 用 noDCA 目錄（`:27-29`）。
- 75（`code/m10_ruler_cwe_vllm.py:39-41`）：`enable_prefix_caching=False`、`max_model_len=33792`、greedy 120 token，跑上游 RULER CWE jsonl（只有 E0）。
- **用過 prefix caching＋KV offloading**：
  - `code/m7_c6_vllm.py`：`vllm serve` ＋ `OffloadingConnector`（CPUOffloadingSpec、48 GiB、LRU）；用 `/reset_prefix_cache` 逼它從 CPU 層載入。
  - `code/m8_vllm_drop.py:477-491`、`code/m9_f1_run.py:331-347`：`enable_prefix_caching=True` ＋ OffloadingConnector（CPUOffloadingSpec，`tier50` 時用 TieringOffloadingSpec 加 fs 二級層）；同一行程 monkeypatch，不改 venv。
  - 已知問題：ROCm 上只要接 KV connector，attention 就從 ROCM_ATTN 退回較慢的 TRITON_ATTN（33K attention 1.367→3.277 s，`RUNLOG:176`）；`m9_f1_run.py:74` 有繞過的 patch。vLLM 的 CPU→GPU 實得約 4.1 GiB/s（`07_report.md:260-268`）。
- **LMCache：沒有安裝**（三個 venv 都沒有），只在 H0 讀過原始碼（run `20261009-164921-h0-src`）。

### 3.7 gpu_guard、pool、runner
- `code/gpu_guard.py`：
  - `with GpuWatcher(gpu=g, out_path=…) as w:` → `w.contaminated`（`:337-360`）；`w.pause(reason)` 暫停取樣。
  - CLI：`--check N`、`--idle-gpus`、`--watch N`、`--host`（`:574-581`）。
  - 10/19 起依行程的 `HIP_VISIBLE_DEVICES`／`ROCR_VISIBLE_DEVICES` 分卡；沒設的照舊算入侵（`RUNLOG:981-984`）。
  - 規則：計時 run 被標污染就重量；品質 run 留分數、`latency_ms` 作廢（CLAUDE.md §3）。
- 佇列（副本 `code/m10_queue_mi300x/`；正在跑的那份在 `/tmp/claude-0/…31610192…/scratchpad/jobs/`，**別碰**）：
  - `runner.sh`：`ROLE=main` 讀 `queue/`、`TIARA_GPU=1`（a6:00.0，計時卡）；`ROLE=helper` 讀 `queue_g0/`、`TIARA_GPU=0`（86:00.0，只跑品質）。名稱用 `LC_ALL=C` 位元組排序。
  - 腳本開頭寫 `# GPU-NEED: one` 就只拿自己那張卡的鎖；沒寫＝both：立 `WANT_BOTH`、依序 `flock gpu0.lock gpu1.lock`。有 `XGPU_OK` 時 both 也只拿自己那張（`runner.sh:35-44`）。
  - helper 的 pool 看到 `WANT_BOTH` 就停掉 job、exit 75，runner 把 stage 放回 `.sh`，之後照題號續跑（`:47`；`pool.py:5-29`）。
  - `pool.py jobs.json P`：jobs＝`[{"name","cmd","need_gb"}]`，cmd 在 bash 裡跑（已 source `m7_env.sh`）；所有 pool 共用顯存帳 `vram_reserved.json`（已登記＋need＋8 GB ≤ 總量，`pool.py:44-88`）；main 的 job 放不下 a6 就放 86（`:92-102`）。
  - `lib.sh`：`REPO`、`JOBS`、`OUT`、`P`（難度參數）、`fmt_of m tier`、`lastrun`、`pool`，並 `export M10_BUSY_TEARDOWN_PAUSE=1`。
  - 慣例：每個 run 用 `m7run <短名> <指令>`（`$TIARA_RUNS/<run_id>/` 有 cmd.sh、stdout、stderr、exit_code）；失敗就寫 `$JOBS/NEEDS_HUMAN`、跳過該段其餘部分、`exit 0`（`RUNLOG:878`）。
  - 看狀態：`make status／q／gpu／jobs／nh／done／tail／watch`（`RUNLOG:948-959`）。

### 3.8 環境、模型、資料
- `source code/m7_env.sh`（`:1-13`）：
  - venv `/mlsteam/workspace/venv/tiara`：torch 2.10.0+rocm7.0、transformers 5.17.0、triton-rocm 3.6.0。
  - `HF_HOME=/mlsteam/data/tiara/hf-cache`、`HIP_VISIBLE_DEVICES=${TIARA_GPU:-0}`、`TIARA_RUNS=/mlsteam/data/tiara/runs`、`m7run`。
  - 注意：要遮 GPU，必須在 source 之後才 export（`RUNLOG:530`）；`M7_MODEL_GLOB` 要在 import 之前設（`RUNLOG:325`）。
- 模型：
  - Llama `/mlsteam/data/tiara/hf-cache/hub/models--unsloth--Llama-3.1-8B-Instruct`、Qwen `…/models--Qwen--Qwen2.5-7B-Instruct-1M`（`m10_tasks.py:116-119`）。
  - vLLM 用的 `/mlsteam/data/tiara/models/Qwen2.5-7B-Instruct-1M-noDCA`。
  - hub 裡另有 Qwen2.5-14B-1M、Qwen3-30B-A3B、LongAlpaca-7B、Mistral-Nemo、Nemotron-8B-UltraLong-1M、Seed-OSS-36B、Qwen3-VL-8B。
- 資料：`/mlsteam/data/tiara/datasets/{hotpotqa, longbench（含上游 eval／metrics）, ruler_ref, scbench, traces（Mooncake conversation／synthetic／toolagent）, gsm8k, yakv_ref}`。
- 硬體：2 × MI300X 192 GB；`/mlsteam/data/tiara` 是 NFS4（NetApp）；第一階段的本地 SSD 路徑是 `/var/tmp`（overlay）。

---

## §4　缺口

**共同限制**（任何方向都會碰到）：
- m10 的「慢層」都是 pinned 記憶體＋限速器，沒有真的檔案 I/O。真實裝置只在第一階段量過（C2、A2、C7 錨點），而且 `m7_calib.py c2` 的磁碟統計仍寫死 `sda`（`RUNLOG:251`）。
- 一次只服務一個請求，沒有併發和寫入積壓（`12_round1_report.md:310`）。
- harness 的重算比預設 vLLM 慢 1.7–2 倍（`07_report.md:267`），會影響 κ 和會合點；70 會量，但還在排隊。
- GPU 忙＝合成的 matmul。
- 長 context 速度表有 32K・閒取錯列的 bug（§2）。

**方向 1：壓縮／drop**
- 有：13 種格式＋K／V 混合、融合 kernel、key_keep 遮罩、13 種挑法、DS1→DS2 流程。
- 缺：
  - (a) drop 路徑還沒在 GPU 上驗（44e0）。
  - (b) 丟掉之後 cache 不會變短：KVModel 是固定長度的緩衝，attention 仍跑整段，所以量不到 decode／TTFT 的好處（`14_match_drop_plan.md:139`）。
  - (c) 只有讀法 A（全載入）的品質，沒有和 Cake 重算組合的品質（`:140`）。
  - (d) `m10_drop.py screen` 和 `m10_quality` 都只能 32K。
  - (e) per-head token drop 對不到還原速度，只有 B 粒度可以。
  - (f) 「降精度 vs 丟掉」同存量的品質比較還沒設計。

**方向 2：選擇性載入**
- 有：QRY（知道問題後的上限）、`Restorer.drop` 跳過載入、G5b 在 Cake 和全載入之間自動選、還原單位可到 128（`M10_UNIT`）。
- 缺：
  - (a) 便宜的線上挑法：QRY 要用完整的 BF16 attention，前提是 KV 已經全部在 GPU，和「選擇性載入」本身互相矛盾。需要 Quest 式的 page 摘要（min／max key）之類，目前沒有。
  - (b) 挑選的時間沒有算進還原時間（DS1 另記 `t_score_s`）。
  - (c) 第 2 輪需要被丟掉的 block 時，沒有「之後再補載」的路徑。
  - (d) 沒有 decode 時的稀疏 attention kernel。
  - (e) 單位 128／256 在慢層的效果還沒量（44e5）。

**方向 3：prefix 復用**
- 有：Mooncake trace 的前綴共享分析與 SCBench 結構分析（`code/m8_trace_oracle.py` `char`／`scbench`）、SCBench 資料在磁碟上、vLLM prefix caching＋OffloadingConnector 腳本（C6／D2／F1）、第一階段的多 session 工作負載與虛擬時鐘模擬器（`m7_write_policy.py b`、`m7_sim.py`）。
- 缺：
  - (a) harness 沒有跨請求共用前綴：每次還原只是同一個 session 自己的連續前綴，沒有 refcount、radix tree、去重。
  - (b) 沒有非前綴復用（CacheBlend 式，要重新處理位置）。
  - (c) 品質題目只有「同一 context 兩輪」，沒有 SCBench 的 runner。
  - (d) LMCache 沒裝。
  - (e) ROCm 上 vLLM 一接 connector 就走慢路徑，公平比較要先套 F1 的 patch。

**方向 4：分層預取 overlap**
- 有：Cake 兩條線本身就是重疊；每個 block 有自己的層（tier2）；S1 邊搬邊拆、S4 延後拆；I/O stream 優先權（`M10_RS_PRIO`）；CU 遮罩；第一階段的 share／fifo 干擾模型。
- 缺：
  - (a) 請求來之前的預取：沒有預測器，也沒有「閒的時候把 NFS 的 block 先搬到 CPU」的路徑。
  - (b) 還原器只做「某一層 → GPU」直接 H2D，沒有 NFS→CPU→GPU 的多段搬移。
  - (c) 沒有逐層（layer-wise）管線：還原單位是一個 block 的全部層，LMCache 式的逐層重疊沒做。
  - (d) 預取的準確度、浪費的位元組還沒有定義指標。
  - (e) 12 號已經顯示降級順序要配合還原方向（NFS 級「舊的先降」快 7–59%，`08:383-385`），但「預取順序要怎麼配合 Cake 的兩條線」還沒有任何設計。
