# ICPE 2027（只用 H200）：論文版的證據總表

> 2026-10-30 編。只讀 `results/` 與 `docs/`，沒有跑新實驗。self-review，不是 cross-model review。
> 論文版 markdown 的每一個數字都從這張表抄；表裡沒有的數字不寫。

## 怎麼用

1. **版本規則**：B、C 節的模擬以 **T1**（CPU 層＝H200 自己量的 43.59／41.17 GiB/s）為主結果；舊值（CPU 層 35.4 GiB/s，MI300X 量的常數）只能寫成「CPU 層以 35.4 GiB/s 限速時」的敏感度（T1 §5，登記時決定）。
2. A 節的 GPU 實測、B6、C2 的 GPU 縮小重播以 **T1G／T1G-2** 重量的為主；沒重量的列在「敏感度」欄寫明。
3. **T3**（STAIR，我們照 CacheFlow 的想法寫的簡化規劃器）一律放「敏感度」欄，不當主數字。
4. 標記：〔實測〕GPU 量的；〔模擬〕trace 模擬；〔算術〕；〔判讀〕。模擬的數字一律帶〔模擬〕。
5. 核對：✓＝我打開原始檔（CSV／JSON），數字對得上（多數是從原始列重算）；⚠＝和文件的說法不一致，或只在文件裡找到，見文末「註」；NOT_VERIFIED＝找不到原始檔。
6. headroom＝1 − T_oracle ÷ T_對手，T＝回來請求的平均 TTFT；對手＝最好的對稱（不預知）策略；rule＝min 是 R1、R2 兩種 session 重建規則取小。
7. 「有看頭」＝某元件在同一（trace, 慢層, GPU）、Cake、≥2 種 CPU 容量下 ≥15%；「死路」＝所有格 <5%；其他＝「可能」。
8. 只用 H200。MI300X 的數字列在「H200 上沒有的」，不能寫進論文。
9. ⏳：目前沒有在跑的實驗（T4、T5 已完成，見 B3-T4、B3-T5）；⏳待決＝數字已核對、論文措辭等使用者決定（「待決」一節）。
10. 寫的人在動筆前先讀「不能寫的」與文末「差異清單」。

---

## 0. 設定表

| 項目 | 值 | 來源＋run_id | 核對 |
|:--|:--|:--|:--|
| GPU | NVIDIA H200，143,771 MiB，PCIe Gen5 x16；單卡（不做 TP） | `results/hw_h200.json`（`20261017-000734-r0-h200-probe`） | ✓ |
| 軟體 | vLLM 0.28.0；torch 2.13.0+cu130 | 同上 | ✓ |
| 模型 | Llama-3.1-8B-Instruct；Qwen2.5-7B-Instruct-1M（vLLM 端用 noDCA） | `results/m10_precision_h200/speed_{llama,qwen}/bw.json` 的 model_path（`20261017-074151-m10h-calibf-llama`、`…-074209-…-qwen`） | ✓ |
| 各節用哪個模型 | A1''／A1'／A2／A3＝Llama；A4＝Qwen（FP8）；A4'＝Llama（INT4）；A5、D2、D3、D4、E1–E3＝兩個；X1（B1–B3、B6、B7、C2）、X4（B4）、T4＝Llama 的 f(i)；X5（B5）主格子＝Llama（Qwen 只報告）；X3（C1）兩個；E2E-C 兩個（Qwen×cprefill NOT_MEASURED）；E2E GPU 附錄只有 Llama | 程式（`code/p3x/x1_sim.py` constants、`code/m7_model.py`）與各登記 | ✓ |
| block | 512 token；32K＝64 塊；BF16 每塊 Llama 67,108,864 B、Qwen 29,360,128 B | `results/m10_tier_h200/iowall.json` | ✓ |
| trace：Mooncake conversation | 12,031 個請求；回來的 3,511（R1）／2,357（R2） | `results/icpe_h200/t1/p01/sim_V0/x1/x1_sweep.csv.gz`（`20261028-065829-icpe-t1-x1-sweep-V0`） | ✓ |
| trace：Mooncake toolagent（X1、X5 網格裡也有） | 23,608 個請求；回來的 7,132（R1）／2,378（R2） | 同上 | ✓ |
| trace：FreeInference 前 6 週（2026-05-17～06-27） | 129,717 個請求；回來的 128,072；使用者發起的回來 5,242 | `results/icpe_h200/t1/p12/x1/x1_sweep_fi.csv.gz`（`20261028-073351-icpe-t1-p12-x1fi-sweep`） | ✓ |
| FreeInference 全 16 週（只當背景） | 1,222,199 請求、14,036 session；使用者回來 73,472、工具回來 1,134,688 | `docs/phase3_20261017/03b_data_report.md` §4（run `20261017-221053-p3data-crosscheck`，3090 分支；資料性質，不是平台量測） | ⚠ 只在文件 |
| CPU 層（T1） | 讀 43.5888、寫 41.1687 GiB/s、k_read 1.002（H200 C0 最慢一次 ×0.9） | `results/icpe_h200/t1/tier_params_h200cpu.json`（`20261028-065334-icpe-t1-build`；來源 `20261016-061254-m7h-c0`、`20261016-061303-m7h-duplex`） | ✓ |
| SSD 級（local） | 讀 6.981、寫 1.911 GiB/s、k_read 3.306——**MI300X 量的**，不是 H200 | 同上檔：`t1.derived_from`＝`results/m7_write_policy_mi300x/tier_params.json`、`unchanged`＝[local, nfs, derate]、`k_read_source`「local/nfs: copied from the MI300X tier_params.json」（逐位元照抄，使用者選擇；限速模擬層，不是真的 SSD） | ✓ |
| NFS 級 | 讀 0.331、寫 0.601 GiB/s、k_read 1.673——**MI300X 量的**，不是 H200 | 同上 | ✓ |
| 模擬常數的出處（總結） | H200 量的：CPU 層（T1）、f(i) 重算時間、忙倍數、反量化、b\* 限速層的目標頻寬、每塊位元組。**MI300X 量的：SSD 級與 NFS 級的讀寫頻寬與 k_read**。所有 SSD／NFS 級的格子（含 B1、B2、B3-T5 的 NFS 5.6–11.8%、B6、C2、T4 的 NFS 列）都依賴這兩組 MI300X 常數。論文只能寫「CPU 層與重算的常數在 H200 上量；SSD／NFS 級沿用 MI300X 的量測」 | `tier_params_h200cpu.json`；`code/p3x/x1_sim.py` `constants()` | ✓ |
| 重算時間 f(i) 的量測範圍 | X1／X4（V0）用的 H200 C1：只量 block 0–79（pos_end 最大 40,960 token，約 40K），GPU 閒、每塊 3 次、取中位數；**i ≥ 80 用這 80 個點的最小平方直線外插**。V1（fa3）的 C1 同樣只到 block 79。X5 另用 `bw.json` 的 64 塊（X5 文件：FreeInference 外插到 2,500 塊） | `results/m7_write_policy_h200/calib_c1.csv`（`20261016-061311-m7h-c1`，f_chunk chunk_idx 0–79）；`results/icpe_h200/p01/calib/calib_c1.csv`（`20261022-205934-icpe-p01-c1`）；`code/m8_trace_oracle.py` `load_f()`（第 153–161 行）、`code/p3x/x1_sim.py` 第 126 行 | ✓ |
| FreeInference 回來請求 >128K token 的比例 | **前 6 週（X1、T4、T5 用的）：NOT_VERIFIED**（沒有任何 run 算過）。文件裡的 34.8% 是**另一週** 2026-07-26～08-01（X5 FreeInference 用的那一週；84,524 請求、83,326 回來）的 `frac_ret_keff_gt256`＝0.3479，也就是可重用前綴 k_eff > 256 塊（>131,072 token）的回來比例；同一週 >32K（k_eff >64 塊）88.2% | `results/icpe_h200/p12/x5/x5_trace_meta_fi.csv`（`20261025-203345-icpe-p12-x5-fi`）；`docs/phase3_20261017/explore/X5_precision_capacity.md` §4.7（3090 分支同一週同值） | NOT_VERIFIED（6 週）；✓（那一週的 34.8%） |
| X1 模擬的還原底 | `restore="cake"` 對整個儲存的前綴（CPU BF16、CPU 上預取的 INT4、慢層 INT4 混在一起）用 Cake：從前面重算、從後面依 layout 逐塊讀 CPU 或慢層，會合點以塊為單位；`"load"`＝全部載入。**沒有**實作 A5 的「每層固定規則」（CPU 層 INT4 全部載入＋慢層 Cake）。對 headroom 的影響 NOT_VERIFIED | `code/p3x/x1_sim.py` `View.restore()`（第 310 行起） | ✓（讀程式） |
| X1 的對手集合 | 每個元件只和「同樣動作」的不預知策略比：O-ORD {LRU, LRU2, ARC, S3-FIFO, GDSF, PENS}；O-DROP {LRU, LRU+NONE, LRU+2HW, LRU+2HA}；O-NOW {LRU, LRU+NONE, LRU+2HW}；**O-PF {LRU, LRU+RPF}**（見 B1-R） | `code/p3x/x1_sim.py` `RIVALS`（第 96–101 行） | ✓（讀程式） |
| 打字提示的提前量 | **合成的，不是 trace 裡的訊號**（Mooncake 沒有按鍵時間）：L＝min(打字時間, think time)；打字時間＝max(新輸入 token − 10, 0) × 每 token 字元數（Llama 英文 4.269）÷（51.56 WPM × 5 字元 ÷ 60 s）；51.56 WPM＝Dhakal et al. CHI'18 的桌機平均（照抄句子的速度）；think time＝兩請求間隔 − 估計的生成時間（TPOT 50 ms）。X3 另有手機 36.17 WPM 的敏感度 | `code/icpe/x1_gpu.py` `leads()`（第 48、112–134 行）；`code/p3x/x3_hide.py` 第 40–44 行；`docs/phase3_20261017/explore/X3_short_horizon_prefetch.md` §5、L21、L78 | ✓（讀程式） |
| CPU 容量÷工作集（`cpu_frac_ws`，只報告） | CPU 128 GiB：Mooncake conversation 1.20%、toolagent 1.28%；FreeInference 逐週 0.78／1.11／1.60／2.08／7.27／8.06%（6 週裡 2 週比 Mooncake 吃緊）；FreeInference 16 GiB 逐週 0.10–1.01%。→「CPU 相對工作集吃緊」不能區分 FreeInference 與 Mooncake；真正的差別要用命中率（B3-T4：128 GiB CPU 命中 99.29%）〔判讀〕 | `results/icpe_h200/t1/p01/sim_V0/x1/x1_sweep.csv.gz`（`20261028-065829-icpe-t1-x1-sweep-V0`）；`results/icpe_h200/t5/p12/x1/x1_sweep_fi.csv.gz`（`20261030-102447-icpe-t5-x1fi-sweep`） | ⚠ 註13 |
| b\* 慢層（重算＝載入的會合點在第 b 塊） | Llama b16／b32／b48＝8.83／2.49／0.722 GiB/s；Qwen 4.33／1.23／0.358 GiB/s；X1 的 b32＝2.49（讀寫、k＝1） | `results/m10_precision_h200/speed_*/bw.json` | ✓ |
| 門檻 | δ\*＝max(5%, 雜訊)；雜訊定義依實驗（A1：C4 雜訊；D2：各 launch 中位數的 (max−min)÷中位；第三階段：2σ）；≥15%「有看頭」、<5%「死路」；品質 ε＝95% CI 下界 ≥ −2 個百分點（第二階段另要求輔助任務 ≥ −5 pp、四種讀法都過） | 各登記；`code/m10_analyze.py`（EPS_CI −0.02、CONDS 4 種） | ✓ |
| GPU 驗證（B6）設定 | 縮小 k＝8：Mooncake conversation 依 session 抽 1/8（1,535 請求、467 個回來）；SSD 級；CPU 16／32 GiB（＝128／256 ÷ 8）；Llama-3.1-8B；Cake；5 個策略 × 2 容量＋LRU 重複 2 次＝12 個乾淨 run；TTFT 不含排隊 | `results/icpe_h200/t1/gpu/p02/gpu_requests.csv.xz`（T1G，`20261028-233954-…`～`20261029-024041-icpe-t1g-p02-128-O-PF-r0`） | ✓ |

---

## A. 寫入時預知＝沒有收益〔實測〕

| ID | 主張（一句） | 主數字（版本） | 敏感度（其他版本） | 標記 | 來源檔＋run_id | 核對 |
|:--|:--|:--|:--|:--|:--|:--|
| A1'' | 寫入時依位置決定（S5）沒贏延後版（S4+），重算用 fa3 | （T1G）S5 對 S4+（回來請求 TTFT 中位數，正＝S5 快）cpu_frac 0.25／0.5／1.0＝−0.37／−0.37／+0.01%；rep 6；C4 雜訊 0.03% → δ\* 5% → 成立；污染列 0 | CPU 層 35.4：+0.07／−0.47／+0.03% | 〔實測〕 | `results/icpe_h200/t1/gpu/a1pp/b_share.csv.gz`（`20261028-171820-icpe-t1g-a1pp-b-nfs-gap0`）→ `analysis/verdict.json`（`20261029-023018-icpe-t1g-analyze-a1a2`）；舊 `results/icpe_h200/p01/p01a_verdict.json`（`20261022-234955-icpe-p01a-analyze`） | ✓ 原始列重算 |
| A1' | 同上，sdpa | （T1G）+1.16／+0.09／−0.02%；C4 雜訊 1.76% → δ\* 5% → 成立 | CPU 層 35.4（合併組）：−0.03／−0.12／+0.03% | 〔實測〕 | `…/t1/gpu/a1p/b_share.csv.gz`（`20261028-185448-icpe-t1g-a1p-b-nfs-gap0`）；舊 `results/m7_write_policy_h200_merged/`（`20261016-063929-m7h-b-nfs-gap0`＋`20261016-172445-m7h-b-nfs-gap0-r36-rr`） | ✓ 原始列重算 |
| A2 | 08 消融：寫入時決定同時贏「最好的不看位置」與 S4B 各 ≥5% | （T1G）GPU 0/16；同 run 的模擬列 0/16 | CPU 層 35.4：GPU 0/16 | 〔實測〕 | `…/t1/gpu/a2/analysis/b2_verdict.csv`（`20261028-204748-icpe-t1g-a2-b2-g1-doc-hold-cpu3.69-s0`、`20261028-220610-…-s1`、`20261028-230008-…-g2-chat-hold-s0`） | ✓ 註12 |
| A3 | 第 1 輪模擬存活的設定上 GPU 確認 | （T1G-2）T1 常數下模擬存活 2 個（11.6／hold／doc／NFS／0.5，S5L；43.59／hold／doc／NFS／0.25，S5c）→ GPU 各 2/5 seed（要 ≥3/5）→ **存活 0/2** | CPU 層 35.4：H200 自己 3 格 0/3 | 〔實測〕 | `…/t1/gpu/a3/r1_gpu_verdict_*.csv`（10 個 run `20261029-025653-icpe-t1g2-r1h-…-s0`～`20261029-084254-…-s4`）→ `analysis/verdict.json`（`20261029-091732-icpe-t1g-analyze-a3`）；模擬 `20261028-163501-icpe-t1g2-r1-sim-an`；舊 `results/m7_explore_h200/r1_gpu_verdict_{A,B,C}.csv` | ✓ 註1 |
| A4 | 依位置決定精度沒有幫助（H200 自己的判定路徑：Qwen、FP8 階 Step 2） | F25 對 R25 品質 +17.3／+18.8 pp（閒／忙）但慢 8.0%／9.9%（>δ\* 5%）；M25 −4.9／−4.4、B25 −3.4／−1.9 pp → 穩定差距 1、2 都不成立 → **K2**。〔判讀〕F25 品質好是因為那段沒被讀到。注意：登記的是**品質**檢定，不是 TTFT 檢定；B25 還原比 R25 快 6.2%／4.3%（閒／忙，沒判雜訊；不能寫的 23） | — | 〔實測〕 | `results/m10_precision_h200/analysis/fp8/step2_qwen_seedA.json`；資料 `20261018-074632-m10h-f8m-qwen-T24-s0`～`…-085357-…-s5`、`20261018-092427-m10h-f8m-qwen-T5` | ✓ 註2 |
| A4' | 同上，Llama INT4（「δ\* 用 5% 的對照」路徑，H200 封存資料） | F25／M25／B25 對 R25（讀法 B、b32、閒／忙）−0.47～+0.08 pp，6 個 CI 都含 0 → K2 | — | 〔實測〕 | `…/analysis/sensitivity_delta005/compare_mi300x_path/int4/step2_llama_int4.json`（資料 `quality_sealed/int4mix_llama_*`；RUNLOG_H200 10/18「δ\* 用 5% 的對照」） | ✓ 註2 |
| A5 | 每層最佳還原方式在重算變快（fa3，0.75 倍）時不翻轉 | 24 格 0 翻轉；1 格弱變化（Llama、CPU 層閒、E0：平手→全部載入）；慢層 16 格全是 Cake。Cake 比全部載入快（〔實測〕單次 32K 還原的微基準，只有 b32 與 NFS 級，不是 TTFT）：INT4（E1）8 格 sdpa 11.0–67.2%、fa3 16.5–71.9%；全部格式 16 格 sdpa 11.0–85.9%、fa3 16.5–87.5%；CPU 層：INT4（E1）全部載入勝或平手，**BF16（E0）GPU 忙時 Cake 勝**（Llama 0.152 vs 0.238 s、Qwen 0.068 vs 0.236 s）→ P0-1 總判定成立 | sdpa 同字；fa3 的 Cake 比 sdpa 快 3.4–18.8%；T1G 沒有重量（不在 T1 §4 清單） | 〔實測〕 | `results/icpe_h200/p01/p01b_cells.csv`、`p01b_verdict.json`（`20261026-003415-icpe-p01b-analyze-final`；重跑 `20261025-230946-…-qwen-s4-fa3-rr`、`…-231221-…-qwen-g5b-fa3-rr`）；`p01_verdict.json`（`20261026-003427-icpe-p01-verdict`） | ⚠ 註3 |

## B. 預知「回來時間」：只在窄區域有用

| ID | 主張（一句） | 主數字（版本） | 敏感度（其他版本） | 標記 | 來源檔＋run_id | 核對 |
|:--|:--|:--|:--|:--|:--|:--|
| B1 | 聊天 trace 上 oracle 天花板高（X1 最大的設定：conversation、SSD 級、GPU 閒、Cake） | （T1，Cake）CPU 128／256 GiB：O-PF 50.5／46.5%、O-ORD 27.6／31.0%、O-DROP 27.6／34.9%、O-NOW 8.0／7.4%；X1 判定「有看頭」（R1、R2、min 都是）；這幾格都沒有寫入飽和 | T3（STAIR）：42.9／39.4、20.9／24.2、20.9／27.4、5.5／5.1%；CPU 層 35.4：48.9／44.6、25.4／27.0、27.0／32.8、8.0／7.4% | 〔模擬〕 | `results/icpe_h200/t1/p01/sim_V0/x1/x1_headroom_components.csv`（`20261028-070955-icpe-t1-x1-an-V0` ← `…-065829-…-sweep-V0`）；T3 `…/t3/p01/sim_V0/x1/`（`20261029-125151-icpe-t3-x1-an-V0`）；舊 `…/p01/sim_V0/x1/`（`20261025-194544-icpe-p01c-x1-an-V0`） | ✓ 原始列重算 |
| B1-ORD | 聊天的淘汰順序（O-ORD）上界，以及沒有免費訊號和它比過 | （T1，conversation、Cake、rule＝min，24 格）O-ORD 8.5–65.5%（中位 27.3%），20/24 ≥15%；最大 65.5% 在寫入飽和的 NFS 格（512 GiB、忙、對 GDSF）；不飽和的最大 31.0%（＝B1 的 256 GiB、SSD、閒，對 LRU2）；B1 兩格 27.6／31.0%；截尾 H3600 後 conversation 仍 6 個 O-ORD 熱組（4 個不靠飽和格）。**HINT-PF 只和 O-PF（預取）比過；沒有任何免費訊號對 O-ORD 的比較 → NOT_MEASURED**；聯合 oracle NOT_MEASURED | T3：conversation O-ORD 最大 48.7%；B1 兩格 20.9／24.2% | 〔模擬〕 | `results/icpe_h200/t1/p01/sim_V0/x1/x1{,h3600}_headroom_components.csv`、`x1h3600_hot_groups.csv`（`20261028-070955-icpe-t1-x1-an-V0`、`20261028-071003-icpe-t1-x1-h3600-V0`）；T3 `20261029-125151-icpe-t3-x1-an-V0` | ✓ 原始列重算 |
| B1-R | 只報告：O-PF 換一組對手（不是登記的數字） | B1 兩格 O-PF 對 LRU2（O-PF 登記對手集合外、但在這兩格是最快的不預知策略）：CPU 128／256 GiB＝45.3／40.3%（R1、R2 取小；登記值 50.5／46.5% 是對 LRU／LRU+RPF）。R2、128 GiB：LRU 0.5154 s、LRU2 0.4667 s（快 9.4%）、O-PF 0.2551 s | T3：38.3／33.9%（登記值 42.9／39.4%） | 〔模擬〕 | `results/icpe_h200/t1/p01/sim_V0/x1/x1_sweep.csv.gz`（`20261028-065829-icpe-t1-x1-sweep-V0`）；T3 `results/icpe_h200/t3/p01/sim_V0/x1/x1_sweep.csv.gz`（`20261029-121338-icpe-t3-x1-sweep-V0`） | ✓ 原始列重算 |
| B2 | 「不回來就丟」（O-DROP）大半是 trace 太短；截尾後剩預取與降級順序 | （T1，V0，rule＝min，384 格＝**conversation＋toolagent × Cake＋全部載入** × 4 元件；不是只有 chat）不截尾 O-DROP 最大 47.7%（≥15% 的格 58；只算 Cake：47.4%、28 格）；**H＝3,600 s：O-DROP 0.0%（0 格、0 熱組）**；O-PF、O-ORD 最大 92.4%、99.9% 不變；H1800／H3600 判定都「有看頭」（O-ORD 9、O-PF 11 個熱組）。H＝1,800：O-DROP 最大 24.1%（9 格），**1 個熱組**（conversation、NFS、GPU 忙、CPU 512／1,024；其中 1 格寫入飽和） | T3：47.7%（57）／H1800 19.4%（11 格，熱組 3，全飽和）／H3600 0.0；CPU 層 35.4：45.9%（48）／H1800 22.6%（4 格，0 熱組）／H3600 0.0 | 〔模擬〕 | `…/t1/p01/sim_V0/x1/x1{,h1800,h3600}_{headroom_components,verdict,hot_groups}.csv`（`…-070955`、`…-070959-…-h1800-V0`、`…-071003-…-h3600-V0`） | ⚠ 註4 |
| B8 | Mooncake **toolagent**（tool／agent trace，和 conversation 同在 X1 網格）上 X1 天花板高 | （T1，Cake、rule＝min）判定「有看頭」；≥15% 的格 **40/96**（CPU 128:14、256:14、512:6、1,024:6；NFS 22、SSD 11、b32 7）；各元件最大 O-ORD **55.5%**（256 GiB、NFS、忙，對 LRU2，寫入飽和）、O-PF 41.6%（512、NFS、閒，飽和）、O-DROP 41.1%（512、NFS、忙，不飽和）、O-NOW 7.1%；不飽和的 ≥15% 格 23；熱組 O-DROP 4、O-ORD 3、O-PF 5。**截尾 H＝3,600 s＋只算不飽和格：仍 12 格 ≥15%**（全在 CPU 128／256 GiB、SSD 級／b32；O-PF 7 格最大 30.0%、O-ORD 5 格最大 17.0%）；H3600 熱組 O-ORD 3、O-PF 5（共 8；不靠飽和格的 1＋3） | T3：28/96 ≥15%，最大 O-ORD 36.2%；H3600＋不飽和 4 格（全 O-PF）；H3600 熱組 O-ORD 2、O-PF 4 | 〔模擬〕 | `results/icpe_h200/t1/p01/sim_V0/x1/x1{,h3600}_{headroom_components,hot_groups,verdict}.csv`（`20261028-070955-icpe-t1-x1-an-V0`、`20261028-071003-icpe-t1-x1-h3600-V0`）；T3 `…/t3/p01/sim_V0/x1/`（`20261029-125151`、`20261029-125229-icpe-t3-x1-h3600-V0`） | ✓ 原始列重算 |
| B9 | Mooncake 兩條 trace 的 X1 ≥15% 格**不只**在小 CPU、慢的慢層 | （T1，conversation＋toolagent，Cake、rule＝min，192 格）≥15% **95/192**：依 CPU 128:28、256:31、**512:24、1,024:12**；依慢層 NFS 42、**SSD 級 29**、b32 24。**截尾 H3600＋只算不飽和：36 格**（128:16、256:12、**512:8**、1,024:0；SSD 19、b32 17、NFS 0）；512 GiB 那 8 格全是 conversation、SSD／b32 × 閒／忙 × O-ORD／O-PF，最大 O-PF 30.3%（SSD、閒，對 LRU+RPF）、O-ORD 24.2%（SSD、閒，對 GDSF） | T3：86/192（512:24、1,024:11；SSD 22）；H3600＋不飽和 28 格（512:8） | 〔模擬〕 | 同 B8 | ✓ 原始列重算 |
| B3-X1 | 真實 agent trace（FreeInference 前 6 週）上 X1 天花板消失 | （T1，CPU 128–1,024 GiB）四元件最大 0.30%（O-ORD）；只算使用者發起 1.52% → **死路**。**適用範圍**（加上 B3-T5）：X1 的 4 個元件在 CPU 16–1,024 GiB 每一格都 <15%；5–12% 只出現在 O-ORD＋NFS 級、CPU 16–32 GiB；CPU ≥128 GiB 時 ≤0.30%。X5 是 ≤2.44%、只量 128–1,024 GiB（B3-X5）。**只對 FreeInference 成立**：Mooncake toolagent 的上界高（B8）。論文措辭 ⏳待決（待決 4） | T3：0.30／1.52%；CPU 層 35.4：0.28／1.48% | 〔模擬〕 | `…/t1/p12/x1/x1fi{,user}_verdict.csv`（`20261028-074217-icpe-t1-p12-x1fi-an`、`…-074221-…-x1fiuser-an`） | ✓ |
| B3-X5 | 同上，「知道誰不回來」（X5） | 最大 2.44%（24 主格，0 格 ≥5%）→ **死路**；截尾 H1800 2.44%、H3600 2.45%。**資料是另一週**：2026-07-26～08-01 一週（84,524 請求、83,326 回來），**不是** B3-X1／T4／T5 用的前 6 週。**只量了 CPU 128／256／512／1,024 GiB**（各容量最大 2.44／0.50／0.17／0.09%）；**CPU 16–64 GiB 的 X5＝NOT_MEASURED**（T5 只擴了 X1 的 4 個元件），不能寫成 <15% | X5 不讀 CPU 層常數 → T1 不重跑（T1 §2 自檢 3）；T3 沒跑 | 〔模擬〕 | `results/icpe_h200/p12/x5/x5_verdict_fi{,_cens1800,_cens3600}.json`（`20261025-203614-icpe-p12-x5-fi-an`、`…-205709-…`、`…-205915-…`）；CPU 範圍 `x5_headroom_fi.csv`、`x5_cells_fi.csv`（`20261025-203345-icpe-p12-x5-fi`） | ✓ |
| B3-T4 | 橋接 Liu 等人：命中率差距 vs TTFT 差距（同一段 FreeInference） | **BRIDGE-NO-GAP**（主設定 ssd、GPU 閒、回來請求）：CPU 8／16／32／64／128 GiB 的 G_hit＝3.03／6.01／4.85／0.83／0.07 pp，G_rec（沒命中就重算）＝1.4／5.8／15.9／11.8／0.9%，G_cake＝1.6／2.9／2.6／0.4／0.02%；沒有一個容量同時 G_hit ≥5 pp 且 G_rec ≥15%；128 GiB 時 CPU 命中 99.29%；有慢層時 CPU＋慢層命中 99.42%（每個容量） | 12 種（慢層×GPU×子集）全部 NO-GAP；只報告：NFS 級 G_cake 16／32 GiB＝17.7／27.7%（閒）、19.1／29.2%（忙），64 GiB 7.7%、128 GiB 0.3%。**注意：G_cake 的對手是 LRU**；同樣這 4 格對「最好的對稱對手」（S3-FIFO）只剩 5.6–11.8%（B3-T5）→ 18–29% 大部分是 LRU 太弱（待決 4） | 〔模擬〕 | `results/icpe_h200/t4/t4_verdict.json`、`t4_cells.csv`（analyze `20261030-101546-icpe-t4-analyze` ← sweep `20261030-100335-icpe-t4-sweep`） | ✓ 由 ttft 欄重算 |
| B3-T5 | FreeInference 上 CPU 縮到 16／32／64 GiB，X1 對最好的對稱對手仍沒有 ≥15% | 判定（X1 原登記規則）：全部回來的請求 **「可能」**、只算使用者發起 **「死路」**；熱組 0。Cake 96 格（4 元件×CPU 16／32／64／128×3 慢層×閒／忙）裡 4 格 ≥5%、0 格 ≥15%：全是 O-ORD（oracle O-BELS）、NFS 級、CPU 16／32 GiB＝5.59／5.92%、11.62／11.81%（閒／忙），對手都是 S3-FIFO，沒有寫入飽和。其他最大：O-PF 4.89%（32 GiB、b32、閒）、O-DROP ≤0.65%、O-NOW ≤0.06%；O-ORD 64 GiB 3.64%；SSD 級 O-ORD ≤1.66%。使用者發起：O-ORD ≤3.92%、O-PF ≤1.44%。只報告：O-PF 在 16 GiB 每格都是負的（最大 −4.38%），全網格最差 −29.5%（32 GiB、NFS、忙）；oracle 比某個對手慢的列 53（O-PF 18、O-NOW 14、O-ALL(report) 21） | CPU 128 GiB 的 828 列和 T1 逐值相同（自檢 PASS）；STAIR、其他 trace 沒跑。論文措辭 ⏳待決（待決 4） | 〔模擬〕 | `results/icpe_h200/t5/p12/x1/x1fi{,user}_{headroom_components,verdict,violations}.csv`（analyze `20261030-104731-icpe-t5-x1fi-an`、`20261030-104745-icpe-t5-x1fiuser-an` ← sweep `20261030-102447-icpe-t5-x1fi-sweep`）；`t5_check.json`（`20261030-104713-icpe-t5-check`）、`t5_report.json`（`20261030-104800-icpe-t5-report`） | ✓ 原始列重算 |
| B4 | 併發時 p99 尾端是排隊，不是預測 | （T1）排隊占 NP p99 的中位 0.983（最小 0.837；264 格）→ D1 成立；平均 headroom ≥5% 的 65 格裡 44 格主要來自「還原變快」（還原占比中位 0.629）；只報告：QP 對 NP 的 p99 改善中位 3.5%（最大 28.1%） | CPU 層 35.4：0.983（0.846；262 格）、43／63；T3 沒跑 | 〔模擬〕 | `results/icpe_h200/t1/p12/x4/p12_queue_verdict.json`、`p12_queue_cells.csv`（`20261028-082005-icpe-t1-p12-queue` ← `20261028-081741-icpe-t1-x4-sweep`） | ✓ |
| B5 | 預知值 ≥15% 的條件（X5「知道誰不回來」） | 登記判定「有看頭」：Llama 主格子 48 格裡 11 格 ≥15%（10 格 CI 下界也 ≥15%），最大 37.4%；11 格的分布：CPU 128（6）、256（4）、**512（1，15.2%）**；慢層 b48（9，0.722 GiB/s）、b32（2，2.49 GiB/s）、b16 0；GPU 閒 7／忙 4；**conversation 6、toolagent 5（最大 37.4% 在 toolagent）** | 事後版：嚴格規則 20 格／48.6%；截尾 H1800 11 格／37.1%；只載入 14 格／59.6%；只算 ≤32K：11→9 格、嚴格 20→15 格；Qwen（只報告）4/48、最大 27.6%；X5 不受 T1 影響；T3 沒跑 | 〔模擬〕 | `results/icpe_h200/p01/sim_V0/x5/x5_verdict.json`（`20261025-195015-icpe-p01c-x5-an-V0`；＝`20261021-012558-p3h-x5-analyze` 同值）；事後 `results/icpe_h200/p12/x5/x5_verdict_{strict,cens1800,loadonly}.json`、`x5_headroom_le32k.csv`（`20261025-203333`、`…-205114`、`…-210002`、`…-210316-icpe-p12-x5-le32k`） | ⚠ 註5 |
| B6 | X1 模擬在 GPU 上站得住 | （T1G）實測÷S_orig 10 格 0.911–0.987（全在 [0.9, 1.1]）；S_orig 差 ≥5% 的 15 對排序全同；8 個 headroom 正負號全同 → **成立**。GPU 上 headroom 比模擬小：O-PF 對 LRU 17.9／16.2%（模擬 24.2／21.1%）；實測÷S_real 0.967–1.021。**TTFT 不含排隊**：回來請求平均排隊 1.19–1.26 s；含排隊時 O-PF 對 LRU 只剩 8.1／4.7%（只報告；重播不做 batching） | CPU 層 35.4：0.911–0.975；O-PF 對 LRU 17.7／16.0%（模擬 23.1／19.9）；含排隊 7.9／4.6% | 〔實測〕對〔模擬〕 | `results/icpe_h200/t1/gpu/p02/gpu_requests.csv.xz`（12 run）→ `analysis/p02_{cells,pairs,signs}.csv`（`20261029-030016-icpe-t1g-analyze-p02`）；舊 `results/icpe_h200/p02/`（`20261026-040946-icpe-p02-analyze`） | ✓ 原始列重算 |
| B7 | 重算跟 vLLM 一樣快時判定字不變 | （T1）X1 V0／V1（fa3 實測 f）／V2（vLLM 級，f 乘 0.571／0.568，模擬）× 原登記／H1800／H3600：9 個字全「有看頭」；≥15% 的格（384）190／192／189。X5（不重跑）≥15% 主格 11／16／17、最大 37.4／40.6／41.5% | T3：181／179／174，字全同；CPU 層 35.4：167／168／169 | 〔模擬〕 | `…/t1/p01/sim_V{0,1,2}/x1/*_verdict.csv`（`…-070955`、`20261028-072128-icpe-t1-x1-an-V1`、`20261028-073300-icpe-t1-x1-an-V2`）；X5 `…/p01/sim_V{0,1,2}/x5/x5_verdict.json`（`20261025-195015`、`…-200641`、`…-202304`）；P0-1 (c) `20261025-202422-icpe-p01c-analyze` | ✓ |

## C. 預取用免費的打字提示就夠

| ID | 主張（一句） | 主數字（版本） | 敏感度（其他版本） | 標記 | 來源檔＋run_id | 核對 |
|:--|:--|:--|:--|:--|:--|:--|
| C1 | 打字提示的提前量夠藏住聊天的慢層還原 | （T1）X3 判定「有看頭」：原登記 36/36 主格、加 FreeInference 37/54；最好的 NFS 格＝**Bailian traceA**、打字提示、Llama INT4、閒：時間加權藏住 98.74%。論文自己的 **Mooncake conversation**、NFS 級、full：62.2–72.0%（2 模型×BF16／INT4×閒／忙） | CPU 層 35.4：同字、同 98.74%；T3 沒跑 | 〔模擬〕 | `results/icpe_h200/t1/explore/x3/x3_verdict.json`、`x3_hide_summary.csv.gz`（`20261028-081248-icpe-t1-x3-hide`） | ⚠ 註6 |
| C2 | 打字提示預取（HINT-PF）拿到 oracle 預取（O-PF）的大部分（全尺寸） | （T1，Cake）24 格（conversation、R1、SSD／b32／NFS×閒／忙×128–1,024 GiB）：HINT-PF 對 LRU 6.9–49.1%、O-PF 對 LRU 7.0–54.2%；HINT-PF÷O-PF 改善比例中位 SSD 0.933／b32 0.941／NFS 1.118；>1 的格 4（全是 NFS、128／256 GiB）；逐格最小 0.814 | T3：6.6–41.0%、6.7–46.5%；中位 0.924／0.933／0.932；>1 的 2 格；最小 0.796。CPU 層 35.4：5.6–47.8%、5.5–52.4%；0.939／0.949／1.117；6 格 | 〔模擬〕 | `results/icpe_h200/t1/p02/fullsim_hint.csv`（`20261028-073312-icpe-t1-p02-fullsim`）；T3 `…/t3/p02/fullsim_hint.csv`（`20261029-140918-icpe-t3-p02-fullsim`）；舊 `…/p02/fullsim_hint.csv`（`20261025-202314-icpe-p02-fullsim`） | ✓ 註11 |
| C2-GPU | 同上，GPU 縮小重播 | （T1G）回來請求平均 TTFT LRU 0.278／0.272、HINT-PF 0.237／0.237、O-PF 0.228／0.228 s（CPU 16／32 GiB）；HINT-PF 對 LRU 14.5／12.9%、O-PF 對 LRU 17.9／16.2% → **HINT-PF 拿到 O-PF 改善的 0.81／0.79**〔算術〕（同設定模擬 0.86／0.85）；O-PF 比 HINT-PF 快 4.0／3.8%（<5%，不判） | CPU 層 35.4：0.81／0.81（模擬 0.86／0.85） | 〔實測〕 | 同 B6（`20261029-030016-icpe-t1g-analyze-p02`） | ✓ 原始列重算 |

## D. 不用預知就有用的：把位元組變小

| ID | 主張（一句） | 主數字（版本） | 敏感度（其他版本） | 標記 | 來源檔＋run_id | 核對 |
|:--|:--|:--|:--|:--|:--|:--|
| D1 | INT4 每塊位元組小 3.2 倍 | Llama 67,108,864 → 20,971,520 B；Qwen 29,360,128 → 9,175,040 B（都是 3.2 倍） | — | 〔算術〕 | `results/m10_tier_h200/iowall.json`（RUNLOG_H200 10/19 晚） | ✓ 註2 |
| D2 | INT4 讓限速慢層的還原變快 | INT4（E1）對 BF16（E0），32K，b16／b32／b48×閒／忙：Llama 37.4–49.1%、Qwen 36.7–54.5%；12 格全「E1 贏」、污染列 0。**CPU 層 INT4 沒有變快**：Llama 忙 −39.4%、閒 +13.0%（未過 δ\*）；Qwen 閒 −21.9%、忙 −80.0% | δ\*：Llama 28.4%（來自 CPU 忙那格的雜訊）、Qwen 5% | 〔實測〕 | `results/m10_precision_h200/speed_{llama,qwen}/k0_verdict_int4.json`（`20261017-094511-m10h-grid-int4-llama`、`20261017-103540-m10h-grid-int4-qwen`） | ✓ |
| D2-Q | INT4 的品質在 ε 內 | (a) 32K、192 題（T2–T5 各 48，seed 67000）：INT4 對 E0 Qwen +0.4 [−1.1, +2.2]、Llama +1.0 [−0.4, +2.7] pp → 兩個都過 ε。(b) 多鍵 NIAH（T1，180 題）＋LongBench-6（「對照」路徑）：Llama INT4 E1 四種讀法都 0.0 pp、LongBench +0.2 [−1.2, +1.5]；**Qwen INT4 E1 讀法 A −0.8 [−1.9, 0.0] 過，讀法 B-busy／B-i2b −0.8 [−2.2, +0.3] 沒過 ε**；LongBench +0.2 [−0.6, +1.1] | (b) 的 E1s／E1sr 見原檔 | 〔實測〕 | (a) `results/icpe_h200/p11/p11q_report.json`（`20261025-215237-icpe-p11-report-q` ← `20261025-203913-icpe-p11-gen-qwen`、`20261025-210417-icpe-p11-gen-llama`）；(b) `results/m10_precision_h200/analysis/report_only/report_only.json`（RUNLOG_H200 10/21） | ⚠ 註7 |
| D3 | 低秩＋INT4（約 8–17 倍）的品質 | 32K、192 題、seed 67000：Qwen `:cal` 9.64 倍 −0.6 [−2.9, +1.8]（ε 沒過；MLPerf 99% 過，邊際 0.08 pp）、每題 SVD 8.01 倍 −0.4 [−2.7, +1.8]（ε 沒過、MLPerf 過）；Llama `:cal` 11.22 倍 −0.6 [−3.3, +2.0]、17.07 倍 −0.7 [−3.4, +2.1]（ε 沒過、MLPerf 過）；Llama 每題 SVD 7.98 倍 +0.1 [−2.0, +2.3]、12.14 倍 +0.4 [−1.8, +2.7]（ε 過）；`:cal` 對每題 SVD：UNDETERMINED | — | 〔實測〕 | `results/icpe_h200/p11/p11q_report.json`、`tables_p11q.md`（同上） | ✓ |
| D4 | 低秩＋INT4 在慢層比 INT4 快 | **不成立**（登記：`:cal` 版慢層 8 格都要快 >5%）：6/8。GPU 閒：b32 2.20–3.28 倍、NFS 級 1.14–1.73 倍；NFS 級忙 1.48–2.44 倍；**b32 忙 0.56–0.63 倍（INT4 較快）**；每題 SVD 版 5/8；CPU 層忙 0.024–0.048 倍（慢 21–43 倍） | — | 〔實測〕 | `results/icpe_h200/p11/p11t_cells.csv`、`p11t_report.json`（`20261026-061907-icpe-p11-report-t` ← `20261026-061250-icpe-p11t-lp-llama`、`…-061722-…-lp-qwen`、`…-061535-…-wc-llama`、`…-061837-…-wc-qwen`） | ⚠ 註8 |

## S. 固定 SM 預留（MPS）：是取捨，不是免費的增益

| ID | 主張（一句） | 主數字（版本） | 敏感度（其他版本） | 標記 | 來源檔＋run_id | 核對 |
|:--|:--|:--|:--|:--|:--|:--|
| S1 | 微基準：還原給固定 SM 份額比 stream 快、背景幾乎不掉 | GB0-SHARE 成立：18 格（模型, 背景, 動作, MPS 等級）比 stream 快 ≥30% 且背景吞吐比 ≥0.91；最強 Llama × cprefill 背景 × e2e_load_int4_b32 × mps87＝4.28 倍（CI [4.26, 4.29]）、背景比 0.980。GB0-STATIC 不成立（22 個動作裡 20 個的最佳等級固定；dq_vq 不固定）；GB0-FLIP 不成立（0 翻轉）。379 格 2σ>5% 不判；84 列 NOT_MEASURED（Llama decode:64×MPS，容量） | — | 〔實測，微基準〕 | `results/s3_h200/b0/b0_report.json`（`20261026-162611-adhoc`） | ✓ |
| S2 | 端到端（模擬）：固定 MPS 預留守不住背景吞吐 | **E2E-NOGO**：32 格裡 24 格可判、**0/24 過**（status FAIL 24）、ASK 0；另 8 格是這 32 格**之內**的 NOT_MEASURED（Qwen×cprefill）；四組（模型, 背景）都選不出 m\*（開發集 B 最高 0.945 < 0.95；登記版 0.944）。只報告（T1，測試集中位）：不設上限 MPS 對 K2（stream）TTFT 改善 30.7–62.9%，背景 B 少 −0.04～6.5 pp；stream 的 B：conversation 0.70–0.80、FreeInference 0.93–0.96 | 登記版（CPU 層 35.4）：同判定；30 → 32.9–63.0%、0～6.7 pp | 〔模擬〕 | T1：`results/icpe_h200/t1/e2e/e2e_report.json`（`20261029-021758-icpe-t1-e2e-report`）、`mps_tradeoff.csv`（`20261029-021836-icpe-t1-e2e-curve`）；登記版 `results/s3_h200/e2e/e2e_report.json`（`20261028-161142-s3-e2e-report`）、`mps_tradeoff.csv`（`20261028-165753-s3-e2e-curve`） | ✓ |
| S3 | MPS 取捨曲線的 GPU 附錄 | **MPS-GPU-INVALID**：可量 9 格；(b) 7 格裡 4 格 GPU 與模擬的 gain 差 >10 pp；7 格正負號全同，GPU 的 gain 全部比模擬大 4.6–31.3 pp；(a) 6/9、(c) 1/9；stream 基準 GPU÷模擬 1.22／1.23；idle 0.986。設定：Llama、conversation k＝8、SSD 級、CPU 16 GiB、HINT-PF、T1 常數 | — | 〔實測〕對〔模擬〕 | `results/s3_h200/e2e_gpu/report.json`（`20261030-021318-s3-e2eg-report`；15 個 GPU run 見 `gpu_runs.csv`；模擬 `20261030-021241-s3-e2eg-sim`） | ✓ |

## E. 效度

| ID | 主張（一句） | 主數字（版本） | 敏感度（其他版本） | 標記 | 來源檔＋run_id | 核對 |
|:--|:--|:--|:--|:--|:--|:--|
| E1 | harness 的重算比 vLLM 慢（sdpa） | 32K 只算：harness ≈2.12／1.865 s ÷ vLLM 預設切法 1.2106／1.0590 s＝Llama 1.75、Qwen 1.76 倍（vLLM 6 次中位、guard 全 CLEAN） | — | 〔實測〕 | vLLM `results/m10_precision_h200/vllm_ttft_guarded/vllm_ttft.csv`（`20261021-025538-m10h-vllmg1-llama-default`、`20261021-025716-m10h-vllmg1-qwen-default`）；harness `results/m10_match_h200/tables_speed_14.md` M2 compute_only 閒（`20261020-*-m10h-md-m2-*`） | ✓ |
| E2 | 換 H200 最快的 attention（fa3）後的 harness ÷ vLLM | Σ64 f(i)：Llama 1.574、Qwen 1.393 s → ÷ vLLM 1.30／1.32 倍；fa3＝sdpa 的 0.751／0.755 | — | 〔實測〕 | `results/icpe_h200/p01/attn/choice.json`（`20261022-205931-icpe-p01-attn-choose`）；`results/icpe_h200/p01/calib/speed_{llama,qwen}/bw.json`（`20261022-205953-icpe-p01-calibf-llama`、`20261022-210325-icpe-p01-calibf-qwen`） | ✓ |
| E3 | 線上還原選擇器（G5b）在 CPU 層、GPU 忙時常選錯 | 原本 12 格裡 10 格 G5b÷最好 1.31–3.51 倍；新 seed 310／311：4 格兩個都 >1.05、5 格只有一個、1 格都沒有 → **部分穩定**；兩個對照格兩 seed 都 >1.05；24 個（格×seed）全部載入比 Cake 慢 1.52–3.52 倍，G5b 選全部載入 30–100% 的次數 | 原本「整體 11.5%」NOT_VERIFIED | 〔實測〕 | `results/icpe_h200/p2/p2_cells.csv`、`p2_verdict.json`（`20261026-060616-icpe-p2-analyze` ← `20261026-055856-icpe-p2-llama-fp8-L310`～`20261026-060415-icpe-p2-qwen-int4-L311`）；原本 `results/m10_restore2_h200/report_13.json`（job 21109） | ⚠ 註9 |
| E4 | 兩個量時間 job 同時跑在不同 NUMA 節點的卡上不互相污染 | CAL-OK：8 個指標變慢 −2.19～+0.94%，95% CI 上界最大 +4.02%（δ\* 5%）；同 NUMA 沒量 | — | 〔實測〕 | `results/icpe_h200/calib/calib_report.json`（`20261029-124037-icpe-cal-analyze`） | ✓ |

## G. 第三階段「不靠預知、但更聰明的還原」閘（ICPE 可選；每個一行）

| ID | 主張（一句） | 主數字（版本） | 敏感度（其他版本） | 標記 | 來源檔＋run_id | 核對 |
|:--|:--|:--|:--|:--|:--|:--|
| G-A0 | 漸進還原（A0）→ 確認性實驗 | A0 GA0-WEAK → A0-C **A0C-STOP**：Q128 P−E0 −2.58 [−3.87, −1.34]、P−U\* +1.28 [−0.27, +2.88]（要 +2） | A0 舊題上 −0.35、+3.47（贏家詛咒） | 〔實測〕 | `results/s3_h200/a0/a0_report.json`（`20261027-155423-adhoc`）、`a0c_report.json`（`20261028-012712-s3-a0c-report`） | ✓ |
| G-M08 | 借鄰居的算力（長 context decode 背景） | **STOP**：45 個候選 0 個過；背景吞吐比 0.35–0.74（要 ≥0.95）；動作速度最高閒時的 0.763 | — | 〔實測〕 | `results/s3_h200/b0/b0_report.json`（`20261026-162611-adhoc`） | ⚠ 註10 |
| G-M12 | 近似重算 | **STOP**：L128 品質 Δgold_lp（F＝1.0）d＝0.1 −14.56 [−17.24, −11.93]、d＝0.2 −8.64 [−10.38, −6.92]（要 ≥−2）；Qwen 速度比 0.47 倍（要 ≤0.3）；Q128 品質 NOT_MEASURED | — | 〔實測〕 | `results/s3_h200/gm12/gm12_report.json`（`20261028-015835-s3-gm12-report`） | ✓ |
| G-M06 | 多一條還原路徑（借鄰卡 H2D＋NVLink） | **PASS**：16 GiB 還原，k＝2 對 k＝1 快 1.986 倍（CI 下界 1.985）；只有 2 卡、鄰卡閒；鄰卡 TPOT 影響 NOT_MEASURED；k＝4 NOT_MEASURED | — | 〔實測〕 | `results/s3_h200/gm06/gm06_report.json`（`20261028-044359-s3-gm06-report`） | ✓ |

## R. 穩健性總表

| ID | 主張（一句） | 主數字（版本） | 敏感度（其他版本） | 標記 | 來源檔＋run_id | 核對 |
|:--|:--|:--|:--|:--|:--|:--|
| R-T1 | CPU 層常數換成 H200 的，B、C 節判定字不變 | 判定字 0 個改變（X1 9 個、B3、B4 D1／D2、X4 30 個、C1、E2E）；單格跨 5%／15% 門檻 271 列，**全部往上** | 沒重跑：X5（B3-X5、B5、B7 的 X5；不讀 CPU 層常數） | 〔模擬〕 | `results/icpe_h200/t1/t1_compare.{csv,json}`（`20261029-021851-icpe-t1-report`） | ✓ |
| R-T1G | GPU 實測的 run 用 H200 CPU 層常數重量，判定字不變 | A1''、A1'、A2、B6／C2：字全同，數字變化 ≤1.2 pp；A3：0/2（原 0/3）；18＋10 步全部「重疊：無」、沒污染。別的使用者的 job（只報告）：拿掉暴露的單位後沒有判定字不同 | 沒重量：A4／A4'（品質）、A5、D 節（CPU 層是真的 H2D） | 〔實測〕 | `results/icpe_h200/t1/gpu/t1g_compare.json`、`overlap.csv`、`exposure/`（`20261029-023333-icpe-t1g-exposure-a1a2`、`20261029-030108-icpe-t1g-exposure-p02`、`20261029-091830-icpe-t1g-exposure-a3`） | ✓ |
| R-T3 | 換更強的規劃器（STAIR），判定字不變 | 11 個判定檔 0 個字改變；B1 headroom 少 2.3–7.6 pp；≥15% 的格少 9–15；唯一變多：H1800 的 O-DROP ≥15% 格 9／7／6 → 11／9／8。**異常 1**：SSD 級回來請求裡 STAIR 比 Cake 慢的 20/3,511（CPU 128）、16/3,511（256），最多 2.05／2.28 倍；NFS 0 個 → 99.4% 以上的 SSD 回來請求 STAIR ≤ Cake。STAIR÷Cake 中位 SSD 0.953／0.935、NFS 0.491／0.625。NFS 級 Cake 格寫入飽和 36/64（T1）→ 61/64（T3） | 沒重跑：X3（C1）、X4（B4）、X5（B5）、E2E-C、所有 GPU 實測 | 〔模擬〕 | `results/icpe_h200/t3/t3_compare.{csv,json}`（`20261029-152352-icpe-t3-report`）、`t3_ratio.csv`（`20261029-141742-icpe-t3-ratio`） | ✓ 註11 |
| R-B7 | 重算速度（sdpa／fa3／vLLM 級）不改判定 | 見 B7 | — | 〔模擬〕 | 見 B7 | ✓ |
| R-B2 | trace 截尾不改判定（但改元件） | 見 B2：H3600 下 O-DROP 全部消失、O-PF／O-ORD 不變 | — | 〔模擬〕 | 見 B2 | ⚠ 註4 |
| R-B6 | 模擬對 GPU | 見 B6 | — | 〔實測〕 | 見 B6 | ✓ |
| R-E2E | MPS 曲線的 GPU 檢查 | 見 S3：正負號一致、大小不一致 → 曲線只能寫〔模擬〕＋GPU 反例 | — | 〔實測〕 | 見 S3 | ✓ |
| R-T2 | 平行跑量時間 job 的效度 | 見 E4 | — | 〔實測〕 | 見 E4 | ✓ |

---

## M. 組合地圖（決策地圖：回來的請求在 H200 上，哪個還原組合最好、值多少）

> **單位與範圍**：一次回來請求的還原時間 t_restore（s，中位數；不含新一輪的 prefill、不含排隊），context 32K＝64 塊（還原的位元組：Llama BF16 4.0 GiB／INT4 1.25 GiB；Qwen BF16 1.75 GiB／INT4 0.55 GiB）。〔實測〕GPU 是 H200 單卡。
> **來源**：CPU 層、b32、NFS 級＝A5 的格子 `results/icpe_h200/p01/p01b_cells.csv`（`20261026-003415-icpe-p01b-analyze-final`；sdpa 為主，每種做法 10 次；資料 job 21654＋重跑 22233）；b16、b48 與「全部重算」＝D2 的格子 `results/m10_precision_h200/speed_{llama,qwen}/summary.csv`（`20261017-094511-m10h-grid-int4-llama`、`20261017-103540-m10h-grid-int4-qwen`；sdpa，30 次）；低秩＋INT4＝D4 `results/icpe_h200/p11/p11t_cells.csv`（`20261026-061907-icpe-p11-report-t`；只有全部載入，xKV 沒有 Cake 路徑）。
> **注意**：(1)「INT4 全部載入」在 CPU／b32／NFS 是「全部載入＋一次拆完」（A5），在 b16／b48 是 D2 的逐塊反量化（較慢的實作）——兩者不同。(2) 層：CPU 層＝真實 pinned H2D；b16／b32／b48＝依模型定的限速（Llama 8.83／2.49／0.722、Qwen 4.33／1.23／0.358 GiB/s）；NFS 級 0.331 GiB/s（MI300X 的常數）。SSD 級（6.98 GiB/s）**沒有**在還原格子裡量過 → NOT_MEASURED。(3)「忙」＝合成的背景負載（duty 1.0），不是 MPS。(4) 領先幅度＝1 − 最好÷第二名；判斷用 A5 登記的平手規則 5%；Llama CPU 層忙那格在 D2 的雜訊是 28.4%（`speed_llama/noise.json`）。

### M1. 地圖（〔實測〕；s；sdpa）

| 模型／層／GPU | BF16 全部載入 | BF16 Cake | INT4 全部載入 | INT4 Cake | 全部重算 | 低秩＋INT4 全部載入：ε 過的最快（ε 沒過的最快） | 最好（不含低秩） | 領先第二名 | >5%？ | 核對 |
|:--|--:|--:|--:|--:|--:|:--|:--|--:|:--|:--|
| Llama／CPU／閒 | 0.0852 | 0.0844 | 0.0291 | 0.0290 | 2.104 | k19v29 0.0207（`:cal` 0.0191） | INT4 Cake≈INT4 全部載入 | 0.5% | 否（平手） | ✓ |
| Llama／CPU／忙 | 0.2383 | 0.1208 | 0.0342 | 0.0445 | 4.051 | k19v29 0.7205（0.7183） | INT4 全部載入 | 23.1% | 是（<雜訊 28.4%） | ✓ |
| Llama／b16／閒 | 0.4530 | 0.3505 | 0.1915\* | 0.1883 | 2.105 | NOT_MEASURED | INT4 Cake≈INT4 全部載入\* | 1.7% | 否（平手） | ✓ |
| Llama／b16／忙 | 0.4530 | 0.4008 | 0.2696\* | 0.2414 | 4.051 | NOT_MEASURED | INT4 Cake | 10.4% | 是 | ✓ |
| Llama／b32／閒 | 1.6071 | 0.8227 | 0.5067 | 0.3772 | 2.106 | k19v29 0.1430（0.1152） | INT4 Cake | 25.6% | 是 | ✓ |
| Llama／b32／忙 | 1.6070 | 1.0651 | 0.5127 | 0.4511 | 4.048 | k29v44 0.7710、k19v29 0.7446（0.7184） | INT4 Cake | 12.0% | 是 | ✓ |
| Llama／b48／閒 | 5.5389 | 1.4007 | 1.7881\* | 0.8765 | 2.106 | NOT_MEASURED | INT4 Cake | 37.4%（對 BF16 Cake） | 是 | ✓ |
| Llama／b48／忙 | 5.5389 | 2.1110 | 1.8569\* | 1.1846 | 4.050 | NOT_MEASURED | INT4 Cake | 36.2% | 是 | ✓ |
| Llama／NFS／閒 | 12.082 | 1.7054 | 3.7803 | 1.2412 | NOT_MEASURED | k19v29 0.9998（0.7166） | INT4 Cake | 27.2%（對 BF16 Cake） | 是 | ✓ |
| Llama／NFS／忙 | 12.082 | 2.8340 | 3.7868 | 1.7737 | NOT_MEASURED | k19v29 1.0121（0.7277） | INT4 Cake | 37.4%（對 BF16 Cake） | 是 | ✓ |
| Qwen／CPU／閒 | 0.0392 | 0.0540 | 0.0141 | 0.0228 | 1.848 | 沒有 ε 過的（k34v51 0.0140；`:cal` 0.0133） | INT4 全部載入 | 37.9% | 是 | ✓ |
| Qwen／CPU／忙 | 0.2355 | 0.0689 | 0.0166 | 0.0364 | 3.562 | 沒有（0.7069） | INT4 全部載入 | 54.2% | 是 | ✓ |
| Qwen／b16／閒 | 0.4043 | 0.3091 | 0.1600\* | 0.1723 | 1.848 | NOT_MEASURED | INT4 全部載入\* | 7.1% | 是 | ✓ |
| Qwen／b16／忙 | 0.4043 | 0.3553 | 0.2482\* | 0.2250 | 3.558 | NOT_MEASURED | INT4 Cake | 9.4% | 是 | ✓ |
| Qwen／b32／閒 | 1.4238 | 0.7252 | 0.4475 | 0.3337 | 1.848 | 沒有（0.1516） | INT4 Cake | 25.4% | 是 | ✓ |
| Qwen／b32／忙 | 1.4237 | 0.9384 | 0.4486 | 0.3993 | 3.558 | 沒有（0.7067） | INT4 Cake | 11.0% | 是 | ✓ |
| Qwen／b48／閒 | 4.8858 | 1.2315 | 1.5630\* | 0.7654 | 1.848 | NOT_MEASURED | INT4 Cake | 37.9%（對 BF16 Cake） | 是 | ✓ |
| Qwen／b48／忙 | 4.8858 | 1.8620 | 1.6263\* | 1.0356 | 3.557 | NOT_MEASURED | INT4 Cake | 36.3% | 是 | ✓ |
| Qwen／NFS／閒 | 5.2864 | 1.3221 | 1.6548 | 0.7837 | NOT_MEASURED | 沒有（0.5524） | INT4 Cake | 40.7%（對 BF16 Cake） | 是 | ✓ |
| Qwen／NFS／忙 | 5.2864 | 1.9586 | 1.6556 | 1.0428 | NOT_MEASURED | 沒有（0.7070） | INT4 Cake | 37.0% | 是 | ✓ |
| 任何模型／SSD 級（6.98 GiB/s）／閒、忙 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | — | — | — | — |
| 任何模型／> 32K（64K、128K） | 只有 CPU 層有（C0 的格子，含 MI300X 訓練，不放進地圖） | | | | | | | | | — |

\* b16／b48 的 INT4 全部載入是逐塊反量化（D2），不是 A5 的「一次拆完」。

- **只准 BF16 時**：CPU 層閒 Llama 平手（0.0852 vs 0.0844）、Qwen 全部載入（快 27.4%）；CPU 層忙 Cake（Llama 0.121 vs 0.238、Qwen 0.069 vs 0.236 s）；所有慢層 Cake。
- **全部重算**在 32K 的每一格都不是最好（閒 1.85–2.11 s、忙 3.56–4.05 s；同一 GPU 狀態下各層相同 → NFS 級沒量但〔判讀〕同值）。
- **fa3 敏感度**（A5 的 fa3 欄）：最好的做法 12 格都不變；INT4 Cake 的時間在慢層少 3.4–14.3%（例 Llama b32 閒 0.3645、NFS 閒 1.0641 s），CPU 層 −0.4～+3.3%；唯一的弱變化是 Llama CPU 閒 BF16（平手 → 全部載入 0.0849 vs 0.1002）。
- **低秩＋INT4 的代價**：ε 過的只有 Llama 每題 SVD（k29v44 7.98 倍、k19v29 12.14 倍）；每個 32K 請求在寫入時要做 SVD：k19v29 1.23＋2.01＝3.24 s、k29v44 5.31 s（GPU 閒；忙時 6.21／10.00 s），Qwen 2.13 s；`:cal` 只要投影 0.042–0.131 s，但 ε 沒過（`results/icpe_h200/p11/time/wcost.csv`，`20261026-061535-icpe-p11t-wc-llama`、`20261026-061837-icpe-p11t-wc-qwen`）✓。GPU 忙時低秩在 b32 與 CPU 層有約 0.71–0.77 s 的底（比 INT4 慢）。

### M2. 格式的品質（ε＝CI 下界 ≥ −2 pp）

| 格式 | Llama-3.1-8B | Qwen2.5-7B | 來源＋run_id | 核對 |
|:--|:--|:--|:--|:--|
| BF16 | 參考（E0） | 參考（E0） | — | — |
| INT4（KIVI） | ✓：32K 192 題 +1.0 [−0.4, +2.7]；NIAH-T1 讀法 A（全部載入）、B-idle、B-busy（Cake 的會合點）、B-i2b 全是 0.0；LongBench +0.2 | ✓ 32K 192 題 +0.4 [−1.1, +2.2]；NIAH-T1 讀法 A −0.8 [−1.9, 0.0] ✓、B-idle −0.6 ✓、**B-busy／B-i2b −0.8 [−2.2, +0.3] ✗**；LongBench +0.2 ✓ → **Cake＋GPU 忙的讀法沒過 ε**（點估計同） | D2-Q（`p11q_report.json` `20261025-215237-icpe-p11-report-q`；`report_only.json`） | ⚠ 註7 |
| 低秩＋INT4 | 每題 SVD 7.98 倍 +0.1 [−2.0, +2.3] ✓、12.14 倍 +0.4 [−1.8, +2.7] ✓；`:cal` 11.22 倍 −0.6 [−3.3, +2.0] ✗、17.07 倍 −0.7 [−3.4, +2.1] ✗（MLPerf 99% 都過） | 每題 SVD 8.01 倍 −0.4 [−2.7, +1.8] ✗、`:cal` 9.64 倍 −0.6 [−2.9, +1.8] ✗（MLPerf 99% 過） | D3（同上） | ✓ |
| 64K／128K 的 INT4 品質 | NOT_MEASURED（這張地圖只到 32K） | NOT_MEASURED | — | — |

### M3. 為什麼一張靜態的表就夠（穩定性）

| ID | 主張（一句） | 主數字（版本） | 敏感度（其他版本） | 標記 | 來源檔＋run_id | 核對 |
|:--|:--|:--|:--|:--|:--|:--|
| M3-a | 重算變快，最好的還原方式不翻 | A5：24 格 0 翻轉；1 格弱變化（Llama、CPU、閒、BF16：平手→全部載入） | — | 〔實測〕 | `p01b_verdict.json`（`20261026-003415-icpe-p01b-analyze-final`） | ✓ |
| M3-b | GPU 爭用下最快的計畫不翻（GB0-FLIP 不成立） | 可判定的（模型×層×背景）線上 0 條翻轉；2 條因 2σ>5% 沒判（Llama CPU×cprefill:2048:65536、Qwen CPU×decode:128:8192）；整份報告標 provisional（379 格 2σ>5% 只回報、84 列 NOT_MEASURED） | — | 〔實測〕 | `results/s3_h200/b0/b0_report.json`（`20261026-162611-adhoc`） | ✓ |
| M3-c | 最好的 MPS 等級是固定的（GB0-STATIC） | 22 個（模型, 動作）裡 20 個固定；不固定的 2 個都是 dq_vq（Llama、Qwen）→ 登記判定「不成立」只因為 dq_vq | — | 〔實測〕 | 同上 `static` | ✓ |
| M3-d | 給還原固定的 SM 份額有用（GB0-SHARE） | 18 格比 stream 快 ≥30% 且背景吞吐比 ≥0.91；最強 Llama×cprefill×e2e_load_int4_b32×mps87＝4.28 倍（CI [4.26, 4.29]），背景比 0.980；另 5 格點估計過、CI 沒過 | — | 〔實測，微基準〕 | 同上 `share`、`share_inconclusive` | ✓ |
| M3-e | 線上選擇器（G5b）在 CPU 層、GPU 忙時常選錯 | 30–100% 的次數選了較慢的全部載入（比 Cake 慢 1.52–3.52 倍）；部分穩定 | — | 〔實測〕 | 見 E3 | ✓ |
| M3-f | 用微基準常數的成本模型選 Cake／全部載入（C0）選不準 | **GC-FAIL**（v1、v2 相同）。預測器＝min_b max(重算, 載入)＋c，常數來自微基準（f(i) 直線、頻寬、反量化、忙倍數）。只用 H200 的切法（sdpa→fa3，24 個測試格）平均代價 33.4%、最差 256%；「永遠 Cake」6.7%／60.5%。最差的格都是 H200 CPU 層、GPU 忙、預測器選全部載入（例 Qwen 64K INT4：Cake 0.0739 vs 全部載入 0.4746 s，代價 542%）。〔判讀，未驗證〕忙時的時間片等待不在單塊 H2D 的常數裡 | 長度、負載兩種切法混了 MI300X 的測試格，平台切法是 MI300X→H200 → 不能用；v2 發現 p01 那批格子開了融合反量化、預測器的 d 被高估（kernel 切法受影響最大） | 〔實測〕＋〔判讀〕 | `results/s3_h200/c0/v2/c0_report.json`（`20261027-164036-s3-c0-eval-v2`）、`cells.csv`（`20261027-164026-s3-c0-cells-v2`）；`docs/stage3_design_20261024/runs/C0_cost_model.md` | ⚠ 註14 |

### M4. MPS 選單（〔模擬〕＋GPU 檢查）

背景只拿 87／75／50% 的 SM（`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`），還原的 process 不限；mps＝MPS 不設上限。數字＝E2E-C 測試集 4 格的中位數；gain＝對 K2（stream＋INT4＋打字提示）的回來請求 TTFT 改善；B＝背景吞吐 ÷ 背景單獨跑。T1 版（CPU 層 H200 常數）為主，括號是登記版（CPU 層 35.4 GiB/s）。SSD、NFS 級的爭用倍數用 b32 的代用。

| （模型, 背景, trace） | stream：B | mps：gain／B | mps87：gain／B | mps75：gain／B | mps50：gain／B |
|:--|--:|:--|:--|:--|:--|
| Llama, cprefill, conversation | 0.801（0.802） | 50.4%／0.777（51.7／0.778） | 71.7%／0.668（72.4／0.668） | 77.8%／0.643（78.1／0.642） | NOT_MEASURED |
| Llama, cprefill, FreeInference | 0.937（0.935） | 62.9%／0.937（63.0／0.936） | 76.0%／0.772（76.2／0.771） | 79.5%／0.752（79.7／0.751） | NOT_MEASURED |
| Llama, mixed, conversation | 0.697（0.698） | 33.2%／0.632（34.4／0.631） | 33.7%／0.607（34.9／0.607） | 36.3%／0.602（37.5／0.601） | 44.7%／0.470（45.5／0.470） |
| Llama, mixed, FreeInference | 0.929（0.927） | 39.8%／0.889（40.0／0.888） | 39.9%／0.829（40.1／0.828） | 42.0%／0.811（42.3／0.809） | 46.8%／0.622（47.1／0.621） |
| Qwen, mixed, conversation | 0.701（0.702） | 30.7%／0.648（32.9／0.648） | 32.2%／0.624（34.3／0.624） | 33.3%／0.607（35.4／0.606） | 42.4%／0.487（44.0／0.487） |
| Qwen, mixed, FreeInference | 0.957（0.955） | 31.5%／0.915（34.3／0.913） | 32.2%／0.866（35.0／0.864） | 33.4%／0.803（36.3／0.802） | 41.0%／0.630（45.0／0.629） |
| Qwen, cprefill（全部） | NOT_MEASURED | | | | |

來源：`results/icpe_h200/t1/e2e/mps_tradeoff.csv`（`20261029-021836-icpe-t1-e2e-curve`）；登記版 `results/s3_h200/e2e/mps_tradeoff.csv`（`20261028-165753-s3-e2e-curve`）✓。登記的端到端判準（B ≥0.95 且改善 >15%）：0/24 可判格過（S2）。

**GPU 檢查**（S3；Llama、conversation k＝8、SSD 級、CPU 16 GiB）：gain GPU／模擬——cprefill：mps 60.1／28.8%、mps87 68.5／62.2%、mps75 76.8／72.3%；mixed：mps 30.9／18.1%、mps87 38.9／18.8%、mps75 42.0／22.4%、mps50 45.6／38.5%（cprefill×mps50 NOT_MEASURED）。7 格正負號全同、GPU 都比模擬大 4.6–31.3 pp → **MPS-GPU-INVALID**：選單的形狀（等級越緊、TTFT 越好、B 越低）在 GPU 上方向對，絕對值不能用模擬的。`results/s3_h200/e2e_gpu/report.json`（`20261030-021318-s3-e2eg-report`）✓

### M5. 四個方向，各一個主實驗（＋跨方向）

| ID | 主張（一句） | 主數字（版本） | 敏感度（其他版本） | 標記 | 來源檔＋run_id | 核對 |
|:--|:--|:--|:--|:--|:--|:--|
| M5-壓縮-a | 位元組變小（INT4）：限速慢層還原變快 | 見 D2、M1：慢層 36.7–54.5%；CPU 層沒有變快（逐塊反量化時） | — | 〔實測〕 | D2 | ✓ |
| M5-壓縮-b | 低秩＋INT4 | 品質見 D3／M2；速度見 D4（6/8，b32 忙時 INT4 較快） | — | 〔實測〕 | D3、D4 | ✓ |
| M5-壓縮-c | 查表式 VQ 解碼（G-M11） | **NOT_DETERMINED**：兩個模型閒時 dq_int4 的參考格太晃（2σ>5%），重量 4 格後仍是；結案不再量 | — | 〔實測〕 | `results/s3_h200/b0/gm11_report.json`（`20261026-180825-adhoc`） | ✓ |
| M5-壓縮-d | 降級時才在 CPU 壓縮來得及嗎（W0；粗底＋INT4 精修，16 核） | **GW0-FAIL**（兩個模型、兩種粗底）：CPU 壓縮速度 Llama B16 0.283 [0.280, 0.286]、B32 0.298 [0.292, 0.301] GiB/s，需求 3.27 GiB/s（vLLM 冷 prefill 32K 1.224 s 的 KV 產生速度）；Qwen 0.529 [0.518, 0.536]、0.601 [0.588, 0.604] GiB/s，需求 1.26 GiB/s（harness fa3，寬鬆）→ 差約 11 倍／2.1–2.4 倍 → 要在 GPU 卸載時就壓。只報告（文件）：GPU 側每個 32K 請求的編碼 31–124 ms，壓縮時背景 decode 吞吐比 0.13–0.44 | — | 〔實測〕 | `results/s3_h200/w0/w0_report.json`（`20261026-103416-adhoc`；量測 `20261026-102851-s3-w0-llama`、`20261026-103153-s3-w0-qwen`）；GPU 側數字只在 `docs/stage3_design_20261024/runs/W0_write_cost.md` 結果節 | ✓（CPU）；⚠（GPU 側只在文件） |
| M5-聚焦-a | 漸進還原（A0：先讀粗底＋只精修重要的塊）→ 確認 | A0 GA0-WEAK → A0-C **A0C-STOP**：Q128 P 57.60% 對 E0 −2.58 [−3.87, −1.34]、對 U\* +1.28 [−0.27, +2.88]；L128 −1.86 [−3.37, −0.35]、+2.32 [+0.53, +4.10] | A0 舊題：Q128 −0.35、+3.47（贏家詛咒） | 〔實測〕 | `results/s3_h200/a0/a0c_report.json`（`20261028-012712-s3-a0c-report`） | ✓ |
| M5-聚焦-b | 打字時就挑塊（G-M04：用問題的前半段挑塊） | **沒有判定**：登記了、資料收了（`results/s3_h200/a0/raw/gm04_*.csv.gz`，不進 git），沒有出報告 → NOT_REPORTED | — | — | `docs/stage3_design_20261024/runs/G_M04_prefix_select.md`（沒有結果節） | NOT_VERIFIED |
| M5-聚焦-c | 依注意力頭還原（M05：DuoAttention 頭表，串流頭只留 sink＋最近窗口） | **沒有判定**（登記的 CI 判準沒算）。只有 L128 的答對率（288 題）：E0 52.8%、p＝0.25 50.8%、0.5 42.8%、0.75 7.2%（點估計）。Qwen NOT_MEASURED（沒有頭表） | — | 〔實測〕 | `results/s3_h200/a0/a0_report.json` `cells.L128.judge.acc`（`20261027-155423-adhoc`） | ⚠ 註15 |
| M5-重用 | 近似（稀疏）重算（G-M12） | **STOP**：見 G-M12 | — | 〔實測〕 | `gm12_report.json` | ✓ |
| M5-擴展-a | 預知（回來時間、誰不回來、寫入時放置） | 見 A、B 節：寫入時 0 收益（A1–A3）；回來時的上界依 trace 與容量（B1、B8、B9、B3） | — | 〔實測〕／〔模擬〕 | A、B 節 | ✓ |
| M5-擴展-b | 多一條還原路徑（G-M06：借鄰卡 H2D＋NVLink） | **PASS**：k＝2 快 1.986 倍（CI 下界 1.985）；只有 2 卡、鄰卡閒；鄰卡 TPOT NOT_MEASURED | — | 〔實測〕 | `gm06_report.json` | ✓ |
| M5-跨方向 | 借鄰居的算力（G-M08） | **STOP**：0/45；背景吞吐比 0.35–0.74 | — | 〔實測〕 | `b0_report.json` `m08` | ⚠ 註10 |

## 待決（措辭要使用者決定；數字已核對）

1. **C2 措辭**。建議：「HINT-PF 拿到 O-PF 改善的 ≥0.92（中位；SSD／b32 兩種規劃器都是 0.92–0.94；NFS 級 Cake 1.118、STAIR 0.932）」。核對：SSD 0.933／0.924、b32 0.941／0.933、NFS 1.118／0.932 → 數字正確。但要同時寫：(i) 這是**全尺寸模擬的中位數**，逐格最小 0.814（Cake）／0.796（STAIR）；(ii) **GPU 實測（縮小）是 0.81／0.79**（T1G；你給的「0.81」是舊常數的 0.81／0.81）；(iii) NFS 級 >1 的格（O-PF 不是上界）在 T1 是 4 格、T3 是 2 格，而且 NFS 級多數格寫入飽和（註11）。
2. **STAIR 例外**：「SSD 級回來請求 ≥99.4% STAIR ≤ Cake」正確（20/3,511＝0.57%、16/3,511＝0.46%）；最多慢 2.05／2.28 倍；NFS 0 個。
3. **MPS 曲線**：標〔模擬〕＋GPU 檢查。核對：可比的 7 格正負號全同，GPU gain 大 **4.6–31.3 pp**（你寫的 5–31 是四捨五入，可用）；判定 MPS-GPU-INVALID（4/7 差 >10 pp）。
4. ⏳**待決：B3 與「預知何時值得做」判準的論文措辭**（使用者還沒決定；T5 登記是「先回報」）。已核對的事實：T4 的 NFS 級、CPU 16／32 GiB G_cake＝17.7–29.2% 是**對 LRU**；T5 對最好的對稱對手（S3-FIFO）同樣這幾格只剩 5.6–11.8%，沒有任何格 ≥15%，這幾格沒有寫入飽和。T5 文件建議的寫法（未定案）：「agent trace 上，CPU 16–1,024 GiB 都沒有任何預知元件 ≥15%；CPU ≥128 GiB 時 ≤0.30%」，判準不加 agent trace 的條件。**照原始檔，這句要改三處**：(i)「agent trace」要改成「FreeInference」——同樣是 tool／agent trace 的 Mooncake toolagent 上界高（B8：40/96 格 ≥15%，截尾＋不飽和後仍 12 格）；(ii)「任何預知元件」只對 X1 的 4 個元件成立，X5 只量了 128–1,024 GiB（16–64 GiB NOT_MEASURED）；(iii)「≥128 GiB 時 ≤0.30%」只對 X1，X5 是 ≤2.44%（而且 X5 用的是 2026-07-26 那一週，不是前 6 週，差異 17）。T4 的登記說法仍有效：不能寫「差距被分層＋Cake 吸收」。同樣待決：「預知 ≥15% 的條件」若從 X1 歸納，CPU 512／1,024 GiB 與 SSD 級都有 ≥15% 的格（B9），只能說「X5 的 11 格」的條件（待決 5）。
5. **B5 的條件句**（新發現，註5）：「CPU ≤256 GiB、慢層慢、聊天型」三個條件只有「慢層慢」完全成立；要改成例如「11 格裡 10 格在 CPU ≤256 GiB、全部在 b32／b48 慢層；conversation 與 toolagent 都有」。
6. **INT4 品質措辭**（新發現，註7）：建議主引 D2-Q (a)（兩模型都過 ε）；(b) 要寫明 Qwen 在 GPU 忙的讀法沒過 ε。
7. **C1 的 trace**（新發現，註6）：98.74% 不是論文那兩條 trace；Mooncake conversation 是 62–72%。

## H200 上沒有的（只有 MI300X 的證據，論文不能用）

- A1（MI300X 的 B 主設定 +0.07／+0.26／−0.30%）；A2 的 MI300X 模擬列 1/16；A3 的 MI300X 那格（35.42／free／local，0/5；T1 常數下不是 H200 的存活設定，沒量）。
- A4 的 MI300X 版（Llama INT4 K2，±0.33 pp）。H200 有 A4'（同一條「對照」路徑、H200 封存資料），但那不是 H200 自己的登記路徑。
- B1'、B3 的 MI300X 常數版（0.22／1.06%、X5 0.64%）。另外：舊的「CPU 層 35.4 GiB/s」各值是 H200 的 run＋MI300X 量的 CPU 常數 → 只能當敏感度。
- C1a（寫入時分數只比隨機好 11–24 分）。
- D2 的 MI300X 數字（Llama 36.1–55.2%、Qwen 38.1–56.9%）；D3 的 MI300X X4b（Qwen 8.01 倍 −0.7、Llama −3.3／−3.6）；D4 的 MI300X validate（SSD 級 2.5–3.4 倍）。
- E1 的 MI300X 比值（1.67／1.72）；E3 的 MI300X 0.4%；第二階段 K3（Qwen INT8）在 MI300X 成立、H200 不成立。
- G-M13（搬運與重算賽跑的重播）用的是 MI300X 的 I3g 資料。
- **SSD 級、NFS 級的讀寫頻寬與 k_read 是 MI300X 量的**（`tier_params_h200cpu.json` 的 `derived_from`＝`results/m7_write_policy_mi300x/tier_params.json`，逐位元照抄；使用者選擇沿用）。這兩組常數本身不是 H200 的證據；用到它們的 H200 模擬與 GPU 重播（B1、B2、B6、B8、B9、B3-T5 的 NFS 格、C2、T4 的 NFS 列、S3）可以用，但論文要寫數值與「沿用 MI300X 的量測」，不能寫「模擬常數都在 H200 上量」。

## 寫的人不能寫的

1. 「差距被分層＋Cake 吸收了」（T4＝BRIDGE-NO-GAP）。登記的說法：B3 的低天花板＝「容量夠（CPU ≥128 GiB 時 CPU 命中 99.3%），在這個模型裡本來就沒差距」。
2. 「agent trace 上天花板 ≤0.30%」而不加「CPU ≥128 GiB」；CPU 16–32 GiB＋NFS 級時 O-ORD 有 5.6–11.8%（B3-T5）。也不能把 T4 對 LRU 的 17.7–29.2% 寫成預知的上界（對最好的對稱對手只有 5.6–11.8%）。措辭 ⏳待決（待決 4）。
3. 「固定 SM 預留是不用預測的端到端增益」：E2E-NOGO＋MPS-GPU-INVALID → 只能寫「延遲換背景吞吐的取捨」〔模擬〕＋GPU 反例。
4. 「Qwen INT4 品質在 ε 內」不加讀法條件；「INT4 讓還原變快」不加「限速慢層」（CPU 層 INT4 較慢）。
5. 「低秩＋INT4 在慢層比 INT4 快」不加條件（6/8；b32、GPU 忙時 INT4 快）。
6. 「HINT-PF 拿到 ≥0.92」當成實測或逐格保證（GPU 0.79–0.81；逐格最小 0.80–0.81）。
7. 「模擬在 GPU 上驗證過」而不說 TTFT 不含排隊；含排隊時 O-PF 對 LRU 只剩 4.7–8.1%。
8. 「STAIR 處處 ≤ Cake」「STAIR 就是 CacheFlow」「X1 的上界涵蓋批次規劃／多卡」。
9. 「還原方式是固定規則：CPU 層全部載入」不加格式條件（BF16、GPU 忙時 Cake 勝，A5）。
10. 「G5b 選錯在每格可重現」（E3 部分穩定）。
11. 「SSD／NFS 是真實裝置」；「vLLM 級重算有實測」（V2 是模擬的縮放，只有 V1＝fa3 是實測）。
12. 「H＝1,800 s 截尾後 O-DROP 完全消失」（T1 下還有 1 個熱組；用 H＝3,600 s 的說法）。
13. 第三階段 3b 的 README 規則：「模擬的『有看頭』只能篩，不能單獨進論文」——B 節的模擬上界必須和 B6 的 GPU 驗證一起寫（B6 只驗了 X1 conversation；X5、toolagent 沒有 GPU 驗證）。
14. 「模擬器的常數都在 H200 上量」：SSD／NFS 級是 MI300X 的（設定表）。
15. 「agent trace 上界低」不說是 FreeInference：Mooncake toolagent 上界高（B8）；論文用了三條 trace（conversation、toolagent、FreeInference），不是兩條。
16. 「預知 ≥15% 的格子都在 CPU ≤256 GiB、慢層慢」當成全部元件的結論：X1 在 CPU 512／1,024 GiB、SSD 級都有 ≥15%（B9；截尾＋不飽和後 512 GiB 仍 8 格）。
17. 「打字提示拿走了大部分、留給預測器的只剩一小段」不限定在預取：淘汰順序 O-ORD 的上界（B1 兩格 27.6／31.0%）沒有任何免費訊號比過，NOT_MEASURED（B1-ORD）。
18. 「O-PF 對最好的不預知策略 50.5／46.5%」：登記的對手只有 LRU、LRU+RPF；對 LRU2 是 45.3／40.3%（B1-R，只報告）。寫「對同動作、最好的對稱對手」。
19. 「X5 在 FreeInference 的 CPU 16–64 GiB 也 <15%」：NOT_MEASURED（B3-X5）。
20. 「模擬的底＝Cake＋固定的分層規則」：X1 的 Cake 用在整個前綴（設定表「X1 模擬的還原底」）。
21. 「打字提示的提前量來自 trace」：是用 token 數和 51.56 WPM 算出來的（設定表）；GPU 的 0.81／0.79 是實測時間，但提示時間本身是合成的。
22. 「E2E-C 0/32 格過、另 8 格 NOT_MEASURED」：是 32 格裡 24 格可判、0/24 過，8 格 NOT_MEASURED 在 32 格之內（S2）。
23. 「依位置決定精度對 TTFT 沒有收益」：A4／A4' 登記的是**品質**檢定（K2＝沒有穩定的品質差距）；原始檔裡 B25 的還原時間比 R25 快 4.3–8.6%（Llama 0.6708 vs 0.7342 s 閒、0.8926 vs 0.9430 s 忙；Qwen 0.6255 vs 0.6666、0.8089 vs 0.8451 s），沒有判雜訊。只能寫「同樣位元組下沒有穩定的品質收益」；TTFT 的說法只留給層級（A1''／A1'／A2／A3）。
24. 組合地圖（M1）：「全部重算」在 32K 有用（每格都最慢或近最慢）；「SSD 級的最好組合」（SSD 級沒有還原格子，NOT_MEASURED）；「64K／128K 也一樣」（地圖只有 32K）；b16／b48 的「INT4 全部載入」寫成和 CPU／b32／NFS 同一個實作（那兩層是逐塊反量化）。
25. 「低秩＋INT4 是最快的還原」不加條件：ε 過的只有 Llama 每題 SVD，寫入時每個 32K 請求要 3.2–5.3 s 的 SVD（GPU 閒）；Qwen 沒有 ε 過的配置；GPU 忙時在 b32、CPU 層有約 0.71–0.77 s 的底，比 INT4 慢。
26. 「Qwen 的 INT4 Cake 在 GPU 忙時品質過 ε」：B-busy 讀法 CI 下界 −2.2 pp（M2）。
27. 「線上成本模型能選對還原方式」：C0＝GC-FAIL（只能引 H200 的 kernel 切法 33.4%／256%；其他切法混了 MI300X）。
28. 「G-M04、M05 的結論」：兩個都沒有判定（M5）。
29. 「MPS 選單的絕對值在 GPU 上成立」：GPU gain 比模擬大 4.6–31.3 pp（MPS-GPU-INVALID）；只能寫形狀＋〔模擬〕。conversation 上連 stream 的 B 都只有 0.70–0.80。
30. 「W0 的 GPU 側成本（31–124 ms、背景吞吐比 0.13–0.44）」當成已核對：只在文件裡，`w0_report.json` 沒有這些欄。

## 差異清單（文件說法 vs 原始檔）

1. **INT4 品質**：`docs/icpe_story_20261028/README.md` §1.4「Qwen INT4 … ε 過」只在讀法 A（與 B-idle）成立；`report_only.json` 的 Qwen INT4 E1 B-busy、B-i2b `eps_pass=false`（CI 下界 −2.22 pp）；照 `m10_analyze.CONDS` 四種讀法都要過的規則，`compare_mi300x_path/int4/decide_int4.json` 裡 Qwen INT4 三個配置都是 false。
2. **B5 條件**：ICPE_EVIDENCE B5「只在 CPU ≤256 GiB、慢層慢時」與 story §1.5「聊天型的回來」：H200 V0 的 11 格裡有 1 格在 512 GiB（15.2%），5 格是 toolagent（含最大 37.4%）。「11 格全部 ≤256 GiB」是 3090 分支（`docs/phase3_20261017/explore/X5_precision_capacity.md`）的結果，H200 不成立。
3. **B2 H1800**：ICPE_EVIDENCE B2「H＝1,800 的 O-DROP 最大 22.6% 但 0 個熱組」是舊常數；T1（主結果）是 24.1%、1 個熱組（T3 2–3 個，全飽和）。H3600 三個版本都是 0。
4. **C1**：98.7% 是 Bailian traceA（不在論文的兩條 trace 裡）；Mooncake conversation、NFS 級是 62.2–72.0%。
5. **D4-H200**：「b32 忙 0.57–0.63 倍」→ 原始檔 0.565–0.628（Qwen 0.564994），應寫 0.56–0.63。
6. **A5**：「fa3 的 Cake 快 4–19%」→ 原始檔 3.4–18.8%；「CPU 層全部載入」只對 INT4（與 BF16 閒）成立，BF16 在 GPU 忙時 Cake 勝（第二階段 H200 報告也這樣寫）。
7. **G-M08**：RUNLOG_H200_S3 §S3-B0、`B0_interference.md`「背景吞吐比 0.33–0.65」→ `b0_report.json` 的 1 − bg_slowdown＝0.354–0.741。
8. **E3**：「整體 11.5%」在 `report_13.json` 裡沒有找到對應欄位 → NOT_VERIFIED（其他 E3 數字 ✓）。
9. **D1 出處**：ICPE_EVIDENCE 引 `results/m10_tier_mi300x/iowall.json`；H200 有自己的 `results/m10_tier_h200/iowall.json`，位元組相同 → 論文引 H200 那份（X1 程式本身讀的是 MI300X 那份，兩份逐值相同，程式有 BF16 斷言）。
10. **story README**「H200 上沒有：A4 的 Llama INT4 K2」：H200 有 A4'（對照路徑），要寫明不是 H200 的登記路徑。
11. **story README／ICPE_EVIDENCE 的 B、C 節**數字全是舊常數（CPU 層 35.4）；B6「0.911–0.975」→ T1G 0.911–0.987；C2-GPU「0.81／0.81」→ 0.81／0.79；SM 預留「值得（端到端待定）」→ E2E-NOGO、MPS-GPU-INVALID。
12. **FreeInference 請求數**：story §6 的 1,222,199 是 16 週合計；B3 只用前 6 週（129,717 請求、128,072 回來）。
13. **CPU÷工作集**：`AUDIT.md` #10 寫「FreeInference 128 GiB 只占工作集 0.78–1.11%、16 GiB 0.10–0.14%」→ 原始檔逐週是 128 GiB 0.78–8.06%、16 GiB 0.10–1.01%（0.78／1.11% 只是第 6、第 1 週）。稽核的結論方向（FreeInference 有幾週比 Mooncake 1.20–1.28% 更吃緊，所以「容量相對工作集」分不出兩者）仍成立，但範圍要照原始檔寫。
14. **trace 數**：ICPE_EVIDENCE／story 寫「兩條真實 trace」；X1、X5 網格實際有三條（conversation、toolagent、FreeInference），B5 的最大值 37.4% 就在 toolagent。
15. **E2E-C 計數**：RUNLOG／T1 文件的「0／32 格過、NOT_MEASURED 8」容易讀成 32＋8；`e2e_report.json` 是 32 格＝FAIL 24＋NOT_MEASURED 8。
16. **檔內缺 run_id 欄**（CLAUDE.md §4.3）：`step2_qwen_seedA.json`、`step2_llama_int4.json`、`report_only.json`、`iowall.json`、`a3/r1_sim_configs.csv`；run_id 只能從 RUNLOG 抄。
17. **X5 FreeInference 的資料週**：B3-X5 的 2.44% 來自 2026-07-26～08-01 一週（`x5_trace_meta_fi.csv` 只有這一列），B3-X1、T4、T5 用前 6 週（2026-05-17～06-27）；文件與 PAPER 把兩者並列成「FreeInference 上」時要寫明不同週。「34.8% > 128K」也是那一週的值（k_eff 定義），前 6 週 NOT_VERIFIED。

## 註

1. A3：`r1_sim_configs.csv` 只有 ts、沒有 run_id 欄（run_id 抄自 RUNLOG「ICPE-T1G2-PREP」）。別的使用者的 job（只報告）：43.59 那格 seed 0、2 有暴露單位，照附加 1 這一格的存活字是 NM（剩 3 個 seed 裡 2 個過）；判定照登記用全部單位。
2. A4、A4'、D1、D2-Q(b) 的 JSON 檔內沒有 run_id，數字 ✓，run_id 是從 RUNLOG／ICPE_EVIDENCE 抄的。
3. A5：數字 ✓；⚠ 是說法——見差異 6。
4. B2：數字 ✓；⚠ 是說法——見差異 3；X1 的登記要求「只靠寫入飽和格成立時要寫明靠 free 語意」。
5. B5：數字 ✓；⚠ 是條件句——見差異 2。
6. C1：數字 ✓；⚠ 是 trace——見差異 4。
7. D2-Q：數字 ✓；⚠ 是 ε 的說法——見差異 1。(b) 是「對照」路徑（H200 的登記路徑沒走 INT4 階，資料是 H200 量的）。
8. D4：除了 0.57（應 0.56）以外 ✓。
9. E3：「11.5%」NOT_VERIFIED，其餘 ✓。
10. G-M08：判定 ✓；背景吞吐比的範圍和文件不同——見差異 7。
11. NFS 級寫入飽和（ρ_w ≥1，「free」語意讓還沒寫完的資料就能讀）：X1 rule＝min 的 Cake 格，NFS 級 T1 36/64、T3 61/64；SSD 級、b32 都是 0。B1 的格子（SSD）不受影響；C2、T4 的 NFS 數字要當較弱的證據（`fullsim_hint.csv`、`t4_cells.csv` 沒有飽和欄 → NOT_VERIFIED）。
12. A2：別的使用者的 job（只報告）：A2 chat g2-s0 暴露的重複對沒暴露的中位 +8.0%（8 點裡 6 點變慢）；那幾列本來就輸最好的對手 10–217%，判定不翻。
13. CPU÷工作集：數字 ✓（`cpu_frac_ws`＝CPU 容量 ÷ `P["ws_GiB"]`，`code/p3x/x1_sim.py` 第 1002 行）；⚠ 是 AUDIT.md 寫的範圍和原始檔不同——見差異 13。
14. C0：GC-FAIL 與 H200 的最差格 ✓；但長度、負載兩種切法的平均代價把 H200 與 MI300X 的測試格放在一起算（`fits` 各 48＋48、100＋76），平台切法用 MI300X 訓練 → 這三個數字不能當 H200 的結果；只用 kernel 切法（H200 sdpa→fa3）。另外 C0 的「全部載入」來自 G5b 的格子，和 M1 的「一次拆完」不是同一個實作。
15. M05：答對率 ✓（從 `a0_report.json` 讀），但登記的判準（每個 p 的配對差 95% CI 下界 ≥ −2 pp）沒有算過 → 沒有判定；G-M04 連答對率以外的報告都沒有。
