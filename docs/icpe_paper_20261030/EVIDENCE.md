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
9. ⏳：目前沒有在跑的列（T4 已完成，見 B3-T4）；「待決」一節是要使用者決定的措辭。
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
| SSD 級（local） | 讀 6.981、寫 1.911 GiB/s、k_read 3.306 | 同上（**逐位元抄自 MI300X 的 tier_params**，使用者選擇；限速模擬層，不是真的 SSD） | ✓ |
| NFS 級 | 讀 0.331、寫 0.601 GiB/s、k_read 1.673 | 同上（同上，抄自 MI300X） | ✓ |
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
| A4 | 依位置決定精度沒有幫助（H200 自己的判定路徑：Qwen、FP8 階 Step 2） | F25 對 R25 品質 +17.3／+18.8 pp（閒／忙）但慢 8.0%／9.9%（>δ\* 5%）；M25 −4.9／−4.4、B25 −3.4／−1.9 pp → 穩定差距 1、2 都不成立 → **K2**。〔判讀〕F25 品質好是因為那段沒被讀到 | — | 〔實測〕 | `results/m10_precision_h200/analysis/fp8/step2_qwen_seedA.json`；資料 `20261018-074632-m10h-f8m-qwen-T24-s0`～`…-085357-…-s5`、`20261018-092427-m10h-f8m-qwen-T5` | ✓ 註2 |
| A4' | 同上，Llama INT4（「δ\* 用 5% 的對照」路徑，H200 封存資料） | F25／M25／B25 對 R25（讀法 B、b32、閒／忙）−0.47～+0.08 pp，6 個 CI 都含 0 → K2 | — | 〔實測〕 | `…/analysis/sensitivity_delta005/compare_mi300x_path/int4/step2_llama_int4.json`（資料 `quality_sealed/int4mix_llama_*`；RUNLOG_H200 10/18「δ\* 用 5% 的對照」） | ✓ 註2 |
| A5 | 每層最佳還原方式在重算變快（fa3，0.75 倍）時不翻轉 | 24 格 0 翻轉；1 格弱變化（Llama、CPU 層閒、E0：平手→全部載入）；慢層 16 格全是 Cake（比全部載入快 16.5–87.5%）；CPU 層：INT4（E1）全部載入勝或平手，**BF16（E0）GPU 忙時 Cake 勝**（Llama 0.152 vs 0.238 s、Qwen 0.068 vs 0.236 s）→ P0-1 總判定成立 | sdpa 同字；fa3 的 Cake 比 sdpa 快 3.4–18.8%；T1G 沒有重量（不在 T1 §4 清單） | 〔實測〕 | `results/icpe_h200/p01/p01b_cells.csv`、`p01b_verdict.json`（`20261026-003415-icpe-p01b-analyze-final`；重跑 `20261025-230946-…-qwen-s4-fa3-rr`、`…-231221-…-qwen-g5b-fa3-rr`）；`p01_verdict.json`（`20261026-003427-icpe-p01-verdict`） | ⚠ 註3 |

## B. 預知「回來時間」：只在窄區域有用

| ID | 主張（一句） | 主數字（版本） | 敏感度（其他版本） | 標記 | 來源檔＋run_id | 核對 |
|:--|:--|:--|:--|:--|:--|:--|
| B1 | 聊天 trace 上 oracle 天花板高（X1 最大的設定：conversation、SSD 級、GPU 閒、Cake） | （T1，Cake）CPU 128／256 GiB：O-PF 50.5／46.5%、O-ORD 27.6／31.0%、O-DROP 27.6／34.9%、O-NOW 8.0／7.4%；X1 判定「有看頭」（R1、R2、min 都是）；這幾格都沒有寫入飽和 | T3（STAIR）：42.9／39.4、20.9／24.2、20.9／27.4、5.5／5.1%；CPU 層 35.4：48.9／44.6、25.4／27.0、27.0／32.8、8.0／7.4% | 〔模擬〕 | `results/icpe_h200/t1/p01/sim_V0/x1/x1_headroom_components.csv`（`20261028-070955-icpe-t1-x1-an-V0` ← `…-065829-…-sweep-V0`）；T3 `…/t3/p01/sim_V0/x1/`（`20261029-125151-icpe-t3-x1-an-V0`）；舊 `…/p01/sim_V0/x1/`（`20261025-194544-icpe-p01c-x1-an-V0`） | ✓ 原始列重算 |
| B2 | 「不回來就丟」（O-DROP）大半是 trace 太短；截尾後剩預取與降級順序 | （T1，V0，rule＝min，384 格）不截尾 O-DROP 最大 47.7%（≥15% 的格 58）；**H＝3,600 s：O-DROP 0.0%（0 格、0 熱組）**；O-PF、O-ORD 最大 92.4%、99.9% 不變；H1800／H3600 判定都「有看頭」（O-ORD 9、O-PF 11 個熱組）。H＝1,800：O-DROP 最大 24.1%（9 格），**1 個熱組**（conversation、NFS、GPU 忙、CPU 512／1,024；其中 1 格寫入飽和） | T3：47.7%（57）／H1800 19.4%（11 格，熱組 3，全飽和）／H3600 0.0；CPU 層 35.4：45.9%（48）／H1800 22.6%（4 格，0 熱組）／H3600 0.0 | 〔模擬〕 | `…/t1/p01/sim_V0/x1/x1{,h1800,h3600}_{headroom_components,verdict,hot_groups}.csv`（`…-070955`、`…-070959-…-h1800-V0`、`…-071003-…-h3600-V0`） | ⚠ 註4 |
| B3-X1 | 真實 agent trace（FreeInference 前 6 週）上 X1 天花板消失 | （T1）四元件最大 0.30%（O-ORD）；只算使用者發起 1.52% → **死路** | T3：0.30／1.52%；CPU 層 35.4：0.28／1.48% | 〔模擬〕 | `…/t1/p12/x1/x1fi{,user}_verdict.csv`（`20261028-074217-icpe-t1-p12-x1fi-an`、`…-074221-…-x1fiuser-an`） | ✓ |
| B3-X5 | 同上，「知道誰不回來」（X5） | 最大 2.44%（24 主格，0 格 ≥5%）→ **死路**；截尾 H1800 2.44%、H3600 2.45% | X5 不讀 CPU 層常數 → T1 不重跑（T1 §2 自檢 3）；T3 沒跑 | 〔模擬〕 | `results/icpe_h200/p12/x5/x5_verdict_fi{,_cens1800,_cens3600}.json`（`20261025-203614-icpe-p12-x5-fi-an`、`…-205709-…`、`…-205915-…`） | ✓ |
| B3-T4 | 橋接 Liu 等人：命中率差距 vs TTFT 差距（同一段 FreeInference） | **BRIDGE-NO-GAP**（主設定 ssd、GPU 閒、回來請求）：CPU 8／16／32／64／128 GiB 的 G_hit＝3.03／6.01／4.85／0.83／0.07 pp，G_rec（沒命中就重算）＝1.4／5.8／15.9／11.8／0.9%，G_cake＝1.6／2.9／2.6／0.4／0.02%；沒有一個容量同時 G_hit ≥5 pp 且 G_rec ≥15%；128 GiB 時 CPU 命中 99.29%；有慢層時 CPU＋慢層命中 99.42%（每個容量） | 12 種（慢層×GPU×子集）全部 NO-GAP；只報告：NFS 級 G_cake 16／32 GiB＝17.7／27.7%（閒）、19.1／29.2%（忙），64 GiB 7.7%、128 GiB 0.3%（待決 4） | 〔模擬〕 | `results/icpe_h200/t4/t4_verdict.json`、`t4_cells.csv`（analyze `20261030-101546-icpe-t4-analyze` ← sweep `20261030-100335-icpe-t4-sweep`） | ✓ 由 ttft 欄重算 |
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
| S2 | 端到端（模擬）：固定 MPS 預留守不住背景吞吐 | **E2E-NOGO**：0/32 格過、ASK 0、NOT_MEASURED 8（Qwen×cprefill）；四組（模型, 背景）都選不出 m\*（開發集 B 最高 0.945 < 0.95；登記版 0.944）。只報告（T1，測試集中位）：不設上限 MPS 對 K2（stream）TTFT 改善 30.7–62.9%，背景 B 少 −0.04～6.5 pp；stream 的 B：conversation 0.70–0.80、FreeInference 0.93–0.96 | 登記版（CPU 層 35.4）：同判定；30 → 32.9–63.0%、0～6.7 pp | 〔模擬〕 | T1：`results/icpe_h200/t1/e2e/e2e_report.json`（`20261029-021758-icpe-t1-e2e-report`）、`mps_tradeoff.csv`（`20261029-021836-icpe-t1-e2e-curve`）；登記版 `results/s3_h200/e2e/e2e_report.json`（`20261028-161142-s3-e2e-report`）、`mps_tradeoff.csv`（`20261028-165753-s3-e2e-curve`） | ✓ |
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

## 待決（措辭要使用者決定；數字已核對）

1. **C2 措辭**。建議：「HINT-PF 拿到 O-PF 改善的 ≥0.92（中位；SSD／b32 兩種規劃器都是 0.92–0.94；NFS 級 Cake 1.118、STAIR 0.932）」。核對：SSD 0.933／0.924、b32 0.941／0.933、NFS 1.118／0.932 → 數字正確。但要同時寫：(i) 這是**全尺寸模擬的中位數**，逐格最小 0.814（Cake）／0.796（STAIR）；(ii) **GPU 實測（縮小）是 0.81／0.79**（T1G；你給的「0.81」是舊常數的 0.81／0.81）；(iii) NFS 級 >1 的格（O-PF 不是上界）在 T1 是 4 格、T3 是 2 格，而且 NFS 級多數格寫入飽和（註11）。
2. **STAIR 例外**：「SSD 級回來請求 ≥99.4% STAIR ≤ Cake」正確（20/3,511＝0.57%、16/3,511＝0.46%）；最多慢 2.05／2.28 倍；NFS 0 個。
3. **MPS 曲線**：標〔模擬〕＋GPU 檢查。核對：可比的 7 格正負號全同，GPU gain 大 **4.6–31.3 pp**（你寫的 5–31 是四捨五入，可用）；判定 MPS-GPU-INVALID（4/7 差 >10 pp）。
4. **T4 的 NFS 觀察（只報告）**：NFS 級、CPU 16／32 GiB 時 O-BEL 對 LRU（分層＋Cake）G_cake＝17.7／27.7%（GPU 閒）、19.1／29.2%（忙）——B3「agent trace 天花板低」只在 X1 的網格（CPU ≥128 GiB）成立。故事怎麼用要你決定。（注意：這是對 LRU，不是 X1 的「最好的對稱對手」；這幾格有沒有寫入飽和 NOT_VERIFIED。）
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
- SSD 級、NFS 級的頻寬常數本身是 MI300X 量的（使用者選擇沿用；論文要寫數值與出處，不能寫成 H200 的裝置量測）。

## 寫的人不能寫的

1. 「差距被分層＋Cake 吸收了」（T4＝BRIDGE-NO-GAP）。登記的說法：B3 的低天花板＝「容量夠（CPU ≥128 GiB 時 CPU 命中 99.3%），在這個模型裡本來就沒差距」。
2. 「agent trace 上預知沒用」而不加「CPU ≥128 GiB（X1 的網格）」（待決 4）。
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
13. 第三階段 3b 的 README 規則：「模擬的『有看頭』只能篩，不能單獨進論文」——B 節的模擬上界必須和 B6 的 GPU 驗證一起寫。

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
13. **檔內缺 run_id 欄**（CLAUDE.md §4.3）：`step2_qwen_seedA.json`、`step2_llama_int4.json`、`report_only.json`、`iowall.json`、`a3/r1_sim_configs.csv`；run_id 只能從 RUNLOG 抄。

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
