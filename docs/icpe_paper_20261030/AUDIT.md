# PAPER.md 獨立稽核（2026-10-30）

> 對象：`docs/icpe_paper_20261030/PAPER.md`（371 行）、`EVIDENCE.md`、`figs/*.sources.json`。
> 方法：逐一打開 EVIDENCE.md 引用的原始檔（CSV／JSON，必要時 `code/` 讀定義），用 pandas 重算。沒有改任何其他檔，沒有跑新實驗。
> 本稽核由一個新開、沒有前文的 Claude subagent 執行——**同一模型家族，不是 cross-model review**。
> 行號＝PAPER.md 的行號。

## 摘要

| 嚴重度 | 數量 | 定義 |
|:--|--:|:--|
| 高 | 3 | 原始資料直接牴觸論文的說法 |
| 中 | 13 | 範圍說過頭、沒有證據支撐、或計數措辭錯 |
| 低 | 17 | 措辭、標記、圖說、一致性 |

**數字忠實度本身很好**：PAPER.md 裡的數字除了它自己聲明「只出自 sidecar」的幾個（表 4 LRU 命中率、36.8%、每格 30 對、≤1.3 pp），全部能在 EVIDENCE.md 找到；我抽查並從原始檔重算了 41 組數字（見附錄 A），**沒有一個數值抄錯**。問題集中在**主張的範圍**：

1. 「agent trace 上界低」只對 FreeInference 成立；同時在用的 **Mooncake toolagent**（X1、X5 網格都有）上界很高，論文沒報告 X1 toolagent 的結果。
2. §7「預知 ≥15% 的格子都同時滿足慢層慢＋CPU ≤256 GiB」只從 X5 的 11 格歸納；X1 自己的 Mooncake 格子（含截尾、排除寫入飽和後）在 CPU 512 GiB、SSD 級都有 ≥15%。
3. 「模擬器的常數在 H200 上量」：SSD／NFS 級常數是從 MI300X 抄的。

## 問題表

| # | 嚴重度 | 位置（§／行號） | 問題 | 證據（檔案、數字） | 建議修法 |
|--:|:--|:--|:--|:--|:--|
| 1 | 高 | 摘要 L16；§1 L33、L39；§5 標題 L136、主張 L138；§11 L350；表 7 L314「只有一條 agent trace」；表 2 L89 只把 FreeInference 標成 agent | 「agent trace 上，CPU 16–1,024 GiB 都沒有任何回來時的 oracle 元件達到 15%」「agent 上界低」。但論文同時在用的 **Mooncake toolagent**（表 2 L88；X1、X5 網格）是 tool／agent trace，而且上界很高；X1 toolagent 的結果全文沒有報告（§5.2 的 B2 數字把它混在「chat trace」底下，見 #11）。照一般讀法，資料直接牴觸這句話 | `results/icpe_h200/t1/p01/sim_V0/x1/x1_headroom_components.csv`（run `20261028-070955-icpe-t1-x1-an-V0`，rule＝min、restore＝cake）：toolagent ≥15% 的格 **40/96**（O-ORD 最大 **55.5%**＝256 GiB、NFS、忙；O-PF 41.6%；O-DROP 41.1%）。`x1h3600_headroom_components.csv` 截尾 H＝3,600 s 且排除 write_saturated 後仍有 **12 格** toolagent ≥15%（SSD 級／b32、CPU 128–256 GiB）。`x1h3600_hot_groups.csv`：toolagent O-ORD 3、O-PF 5 個熱組（「有看頭」）。X5 `results/icpe_h200/p01/sim_V0/x5/x5_verdict.json`：toolagent 5 格 ≥15%，最大 37.4% | 全文把「agent trace」改成「FreeInference」；在 §5 補一段 X1 toolagent 的結果（或說明為什麼 toolagent 不算 agent trace）；表 7「只有一條 agent trace」改寫 |
| 2 | 高 | 摘要 L16「預知值 ≥15% 的區域很窄：只在…慢層慢且 CPU 容量吃緊時」；§1 L35、貢獻 4 L41（「可證偽的條件」）；§7 L224–227；§11 L350 | §7 開頭寫「回來時的預知對最好的對稱對手值 ≥15% 的**格子，都**同時滿足」慢層慢＋CPU 吃緊，但證據只取 X5 的 11 格。X1（同樣是回來時的 oracle、同樣兩條 Mooncake trace）的 ≥15% 格子大量落在 CPU 512／1,024 GiB 與 SSD 級（讀 6.98 GiB/s，接近 X5 裡 0 格的 b16＝8.83 GiB/s）。「區域很窄」與 B7 自己的「≥15% 的格 190/384」也矛盾 | 同上 X1 檔，Cake、rule＝min：≥15% **95/192** 格；CPU 512 GiB 24 格、1,024 GiB 12 格；SSD 級 29 格。截尾 H＝3,600 s＋排除寫入飽和後仍 **36 格**，其中 CPU 512 GiB 8 格（conversation、SSD 級、閒：O-PF 30.3% 對 LRU+RPF、O-ORD 24.2% 對 GDSF）。`results/icpe_h200/t3/t3_compare.json` summary_t1 V0.x1 n_ge15_all＝190 | 把條件明確限定為「X5（知道誰不回來）的 11 格」；或用全部元件重新歸納條件；刪掉「很窄」，或改成「X5 元件 ≥15% 的區域」 |
| 3 | 高 | 摘要 L16「模擬器的常數在 H200 上量」；閱讀指引 L6「常數是 H200 量的」；§3 L92「成本常數來自 H200 的微基準（表 2）」；圖 2 圖說 L96；圖 4、圖 6 圖內文字「H200 constants」 | SSD 級、NFS 級的讀寫頻寬與 k_read 是**從 MI300X 抄的**（L10、表 2 自己也這樣寫）。摘要的說法和 EVIDENCE「H200 上沒有的」最後一條（不能寫成 H200 的裝置量測）衝突；FreeInference 的 5.6–11.8%（NFS 級）與所有 SSD／NFS 格都依賴這些常數 | `results/icpe_h200/t1/tier_params_h200cpu.json`：`k_read_source`＝「local/nfs: copied from the MI300X tier_params.json」；`derived_from`＝`results/m7_write_policy_mi300x/tier_params.json` | 改成「CPU 層與重算的常數在 H200 上量；SSD／NFS 級的頻寬常數沿用 MI300X 的量測」；圖內文字同改 |
| 4 | 中 | 摘要 L16；§1 L33「打字提示拿走大部分」；§5 主張 L138；§5.3「發現與含意」L205「留給預測器的只剩 oracle 與打字提示之間的那一段」；§11 L350 | HINT-PF 只和 **O-PF（預取）** 比過。chat 的 O-ORD（淘汰順序）上界在代表格就有 27.6／31.0%，截尾後仍有 9 個熱組；沒有任何免費訊號對 O-ORD 的全尺寸比較；聯合 oracle NOT_MEASURED。「只剩那一段」說過頭 | `x1_headroom_components.csv`：conversation Cake O-ORD 最大 65.5%；`x1h3600_hot_groups.csv` O-ORD 9 組。GPU 縮小重播只有部分支持：`results/icpe_h200/t1/gpu/p02/analysis/p02_cells.csv` HINT-PF 0.2373 vs O-BEL 0.2486 s（16 GiB）、0.2368 vs 0.2326 s（32 GiB） | 限定為「預取的上界」：「chat 的預取上界高，但打字提示拿到它的大部分；淘汰順序的上界（27.6–31.0%）沒有對應的免費訊號，NOT_MEASURED」 |
| 5 | 中 | 摘要 L16「CPU 16–1,024 GiB 都沒有任何回來時的 oracle 元件達到 15%」；§1 L33、L39；§5 主張 L138「處處」；表 6 第 4 列 L291；§10 L336；§11「處處」 | X5（知道誰不回來，表 1 L71 列為回來時的 oracle）在 FreeInference 上**只掃了 128–1,024 GiB**；T5 只把 X1 的 4 個元件擴到 16–64 GiB。CPU 小正是 X5 在 Mooncake 上出現 ≥15% 的區域，所以 16–64 GiB 的 X5 是 NOT_MEASURED，不是 <15%。§5.1 待決 L184 自己有注意到，但摘要與 §1 沒寫 | `results/icpe_h200/p12/x5/x5_verdict_fi.json`：n_main_cells 24（CPU 128–1,024）；`results/icpe_h200/t5/p12/x1/x1fi_headroom_components.csv`：只有 O-ORD／O-DROP／O-NOW／O-PF | 改成「X1 的四個元件在 16–1,024 GiB 都 <15%；X5 只在 128–1,024 GiB 量過（≤2.44%）」 |
| 6 | 中 | §5.3 L195–197（C1、C2、C2-GPU）；摘要的 0.81／0.79 | 打字提示的提前量是**模型算出來的**，不是 trace 裡的訊號：提前量＝新輸入 token 數 × 每 token 字元數 ÷ 桌機打字速度 51.56 WPM（Dhakal et al.），再以 think time 封頂。Mooncake 沒有按鍵時間戳。論文沒有揭露；GPU 重播的 0.81／0.79 標〔實測〕，但提示時間本身是合成的 | `code/icpe/x1_gpu.py` `leads()`（`typing = max(new_raw − TEMPLATE_OV, 0) * cpt / cps`，`cps = WPM_DESKTOP * CHARS_PER_WORD / 60`）；`code/p3x/x3_hide.py` L40 `WPM = {"desktop": 51.56, "mobile": 36.17}`；`results/icpe_h200/t1/explore/x3/x3_hide_summary.csv.gz` Mooncake 列 `wpm=desktop`、`human_typing=1` | §3 或 §5.3 寫明提前量的模型與參數；報告 mobile WPM（X3 已有 sensitivity 列）的結果 |
| 7 | 中 | 摘要 L16「量它對最好的不預知對手的 TTFT 上界」 | 每個元件的對手只在「同樣動作」的小集合裡挑（§3 L62 有寫，摘要沒寫）。O-PF 的對手只有 {LRU, LRU+RPF}；在 B1 那格，不預知的 LRU2 比 LRU 快 9.4%，所以 50.5% 是對 LRU 的值，不是對最好的不預知策略 | `code/p3x/x1_sim.py` L96–101 `RIVALS`；`x1_sweep.csv.gz`（conversation、SSD、閒、Cake、R2、128 GiB）：LRU 0.5154 s、LRU2 0.4667 s、O-PF 0.2551 s → 對 LRU2 為 45.3%（256 GiB：40.3%），論文寫 50.5／46.5% | 摘要改成「對同動作、最好的對稱（不預知）對手」；或另報對全部不預知策略的值 |
| 8 | 中 | §2 L51；§3 L62「底（Cake＋固定規則）」；摘要「在 Cake 加上固定的分層規則」 | 模擬器的底**不是**文中的固定分層規則：X1 的 `restore="cake"` 對整個 CPU／慢層混合 layout 都用 Cake（含 CPU BF16、CPU INT4），沒有「CPU 層 INT4 全部載入、BF16 閒時全部載入」。A5 量到 CPU 層 INT4 全部載入比 Cake 快 26–39%。對 headroom 的影響 NOT_VERIFIED（CPU 讀很快，可能不大） | `code/p3x/x1_sim.py` `View.restore()`（mode＝cake 時兩端會合，對 lay∈{1,2,3} 一視同仁）；`results/icpe_h200/p01/p01b_cells.csv`：Llama CPU 忙 E1 Cake 0.0460 vs 全部載入 0.0337 s；Qwen CPU 閒 E1 0.0229 vs 0.0140 s | 照實描述模擬的底（「Cake 用在整個前綴」），或用固定規則重跑並比較 |
| 9 | 中 | §4 主張 L116「依位置決定層級或**精度**，對 TTFT 沒有可量到的收益」；表 3 L126–127 | A4／A4' 登記的檢定是**品質**（同樣位元組下，品質要好 ≥2 pp 且速度不輸 δ\*；K2＝沒有穩定的品質差距），不是 TTFT 檢定。原始檔裡 B25（後段低精度）的還原時間比 R25 快 5–9%，而且 Llama 的品質 CI 含 0——沒判雜訊，但不能寫成「TTFT 沒有收益」 | `code/m10_analyze.py` `cmd_step2`（cond1 條件）；`.../compare_mi300x_path/int4/step2_llama_int4.json`：B25 t_x 0.6708 vs t_R25 0.7342 s（閒，−8.6%）、0.8926 vs 0.9430 s（忙，−5.3%），品質 −0.38 pp [−0.82, +0.04]；`.../fp8/step2_qwen_seedA.json`：B25 0.6255 vs 0.6666 s（閒，−6.2%），但品質 −3.4 pp | 精度那半句改成「依位置決定精度，在同樣位元組下沒有穩定的品質收益（K2）」；TTFT 的說法只留給層級（A1''／A1'／A2／A3） |
| 10 | 中 | §7 L227 條件 2「CPU 容量**相對工作集**吃緊」；L232「FreeInference 在 CPU ≥128 GiB 時命中 99.29%，條件 2 不成立」 | 照模擬器自己的 `cpu_frac_ws`（CPU 容量÷工作集），FreeInference 128 GiB 只占工作集 0.78–1.11%，比 Mooncake 128 GiB（1.20–1.28%）**更吃緊**；16 GiB 更只有 0.10–0.14%。所以照字面，條件 2 對 FreeInference 是**成立**的；真正的差別是命中率／重用距離，不是容量相對工作集 | `results/icpe_h200/t5/p12/x1/x1_sweep_fi.csv.gz`、`results/icpe_h200/t1/p12/x1/x1_sweep_fi.csv.gz`、`results/icpe_h200/t1/p01/sim_V0/x1/x1_sweep.csv.gz` 的 `cpu_frac_ws` 欄 | 條件 2 改用命中率（如「LRU 的 CPU 命中率 < X%」）定義，或寫明「工作集」的意思；「條件 2 不成立」那句同改 |
| 11 | 中 | §5.2 標題 L186「chat trace（Mooncake conversation）」下的 B2 那一點 L189；表 6 第 6 列 L293；表 7 L310「≥15% 的格 190／192／189」 | B2 的數字是**兩條 Mooncake trace、Cake＋全部載入兩種還原**一起算的：O-DROP 最大 47.7%、58 格裡含全部載入的格；只算 Cake 是 47.4%、28 格。截尾後的「O-ORD 9、O-PF 11 個熱組」有 3＋5 個是 toolagent。放在「chat trace」底下會誤導 | `x1_headroom_components.csv`（rule＝min）：O-DROP 全部 0.4768／58 格，Cake 0.4742／28 格；`x1h3600_hot_groups.csv` 按 trace 分組：O-ORD conversation 6＋toolagent 3、O-PF 6＋5 | 依 trace 與還原方式分開報；或在句中寫明「含 toolagent 與全部載入」 |
| 12 | 中 | 摘要 L16「用兩條真實 trace（Mooncake conversation、FreeInference）」；貢獻 1 L38 | 實際用了**三條**：toolagent 在 X1、X5 網格，B5 的最大值 37.4% 就在 toolagent；摘要後半又說「只在 Mooncake 的兩條 trace 上」，前後不一致 | 表 2 L87–89；`x5_verdict.json` cells_ge15 的 trace 欄 | 摘要與貢獻 1 改成三條 trace，並說明各自的角色 |
| 13 | 中 | §10 L338「（表 4：本來就沒差距，不是被吸收）」 | 沒加範圍。表 4 本身：16 GiB 的命中率差距 6.01 pp（≥5）、32 GiB 的 G_rec 15.9%（≥15%）、64 GiB 11.8%。登記的判定只說「沒有一個容量同時兩者都過」；「本來就沒差距」只在 ≥128 GiB 成立（§5.1 L165 有加範圍） | `results/icpe_h200/t4/t4_cells.csv`（ssd、idle、ret）：G_hit_pp 6.01197（16）、G_rec 0.15945（32）、0.11778（64） | 改成「沒有一個容量同時有命中率差距與 TTFT 差距；CPU ≥128 GiB 時兩者都接近 0」 |
| 14 | 中 | 貢獻 4 L41；§7；表 6 第 8 列 L295 | B5（X5）的「有看頭」是**純模擬**、沒有 GPU 重播（§7 L233 自己承認）；EVIDENCE「不能寫的」#13：模擬的有看頭只能篩、不能單獨進論文，B 節的模擬上界要和 B6 一起寫，而 B6 驗的是 X1 chat | EVIDENCE.md L164；PAPER L233 | 在貢獻 4 與表 6 第 8 列標「〔模擬〕，未經 GPU 驗證」；或補 X5 的 GPU 重播 |
| 15 | 中 | §5.1 待決 L184 建議寫法「沒有任何預知元件…CPU ≥128 GiB 時 ≤0.30%」；§7 L232「上界 ≤0.30%」 | 同一個量（FreeInference、CPU ≥128 GiB 的回來時上界）在不同地方寫 ≤2.44%（摘要、§1、§11）與 ≤0.30%（§7、待決建議）。0.30% 只是 X1；X5 是 2.44%。建議寫法若照抄，「任何預知元件 ≤0.30%」就是錯的 | `results/icpe_h200/t1/p12/x1/x1fi_verdict.csv` 最大 0.003028；`results/icpe_h200/p12/x5/x5_verdict_fi.json` main_headroom_max 0.024365 | 統一寫「X1 ≤0.30%、X5 ≤2.44%」；修正待決 4 的建議句 |
| 16 | 中 | §8.2 L259「0/32 格過登記門檻（另 8 格 NOT_MEASURED）」；貢獻 5 L42；表 6 第 13 列 L300 | 計數錯：總共 32 格，**其中** 8 格 NOT_MEASURED、24 格 FAIL；不是 32 格之外另有 8 格。可判的是 0/24 | `results/icpe_h200/t1/e2e/e2e_report.json`：`cells` 長度 32，status FAIL 24、NOT_MEASURED 8；`n_not_measured` 8 | 「32 格中 24 格可判、0 格過；8 格 NOT_MEASURED（Qwen×cprefill）」 |
| 17 | 低 | 圖 9 圖說 L265；`figs/fig6_mps.sources.json` notes | 曲線畫的是登記版（CPU 層 35.4 GiB/s），內文數字（30.7–62.9%）是 T1 版；閱讀指引 L8 與「數字出處」L369 說模擬用 T1。sidecar 說 T1 是「report-only」，EVIDENCE S2 卻把 T1 當主數字，兩者對版本規則的說法互相矛盾。曲線只有 Llama＋conversation（測試半），30.7–62.9% 卻跨 2 個模型 × 2 條 trace，圖說沒寫 | `fig6_mps.sources.json` notes[0]、notes[1]；`results/icpe_h200/t1/e2e/mps_tradeoff.csv` | 改畫 T1 曲線，或在圖說寫明版本與「只畫 Llama、conversation」；統一 sidecar 和 EVIDENCE 的版本規則 |
| 18 | 低 | 圖 4 圖說 L142 | 粗體主句「agent trace 上每一種回來時 oracle 都在 5% 線左邊」沒寫 CPU ≥128 GiB（後面才補「24 格（CPU 128–1,024 GiB）」）；圖 6 在 32 GiB 就有 11.8%。違反「不能寫的」#2 的精神。另外 chat 面板標題寫「SSD class, GPU idle」，但面板裡的 X5 長條是 b16／b32／b48 × 閒／忙 × CPU 的最大值（argmax＝b48、閒、128 GiB） | `fig3_scorecard.png`；sidecar `chat_X5_argmax` | 主句加「CPU 128–1,024 GiB」；X5 長條另註慢層與 GPU 狀態 |
| 19 | 低 | 圖 6 圖說 L182 | 「每一格的 oracle … 都 <15%」，但圖只畫 O-ORD 與 O-PF（O-DROP、O-NOW 沒畫）；「空心圈明顯高於實線」在 128 GiB 不成立（兩者都是 0.30%，因為那格最好的對手就是 LRU） | `fig9_fi_capacity.sources.json` plotted_values（只有 O-ORD、O-PF）；`t4_cells.csv` 128 GiB NFS G_cake 0.00303 | 圖說改成「O-ORD 與 O-PF」；「明顯高於」加「CPU 16–64 GiB」 |
| 20 | 低 | §3 L64 headroom 定義 | 「每格取 R1、R2 中較小的值」只適用 Mooncake 的 X1。FreeInference 只有一種規則（`session_id`）；X5 用 subtree 重建、對手是 R1–R6 的最好者 | `x1fi_headroom_components.csv` rule 欄只有 `session_id`；`x5_headroom_le32k.csv` rule＝subtree／strict | 定義旁註明各實驗的 session 規則 |
| 21 | 低 | 表 2 L83 | 「CPU 層〔實測〕讀 43.5888、寫 41.1687」：這是量到的最小值 × 0.9（derate），不是直接量到的值 | `tier_params_h200cpu.json` `rule`＝「min(measured) * 0.9」 | 加「（最慢一次 ×0.9）」 |
| 22 | 低 | §7 L231 | 「重算加速到 fa3／vLLM 級時 11／16／17 格，最大 37.4／40.6／41.5%」：兩個標籤配三個值（11／37.4 是 sdpa 基準 V0） | `results/icpe_h200/p01/sim_V{0,1,2}/x5/x5_verdict.json` | 改成「sdpa／fa3／vLLM 級：11／16／17 格」 |
| 23 | 低 | §6 標題 L209〔實測〕；§8 標題 L240〔實測〕；圖 2 圖說 L96 | §6 含 STAIR（〔模擬〕），§8.2 的端到端是〔模擬〕；圖 2 說「不用預知的收益直接在 GPU 上量」，但 MPS 端到端是模擬 | EVIDENCE R-T3、S2 的標記欄 | 標題改〔實測＋模擬〕或拿掉；圖 2 圖說修正 |
| 24 | 低 | §2 L48 vs 表 6 第 11 列 L298、§8.1 | 底已經「以 INT4 降到慢層」，INT4 卻又列成「加在 Cake 上的東西」，容易被讀成重複計算 | PAPER L48、L298 | 寫明 D2 是在隔離的還原微基準上量 INT4，X1 的底已經內含它 |
| 25 | 低 | §10 L338「花俏的策略贏不了 LRU…『簡單策略就夠』不是本文的新發現」 vs §5.1 L175「大部分是 LRU 太弱」 | 本文自己的 T5：S3-FIFO 在 NFS 級、16–32 GiB 比 LRU 快 14.7–20.4%（TTFT）——跟「花俏策略贏不了 LRU」方向相反（指標不同：TTFT vs 命中率），文中沒有調和 | `results/icpe_h200/t5/p12/x1/x1_sweep_fi.csv.gz` 依週加總：S3FIFO 對 LRU 0.147／0.162（16 GiB 閒／忙）、0.186／0.204（32 GiB）；64 GiB 為 −3.9／−3.4% | §10 加一句說明兩者的差別（指標、容量區間） |
| 26 | 低 | §3 L108「出現外來 process 就作廢重量」；表 7 L316 | 同卡的外來 process 才作廢；同節點別的使用者 job 開始／結束的「暴露」單位**保留**並照登記用全部單位判定（A1'' 180 個單位裡 29 個暴露；A2 chat g2-s0 暴露的重複中位慢 +8.0%，EVIDENCE 註 12）。表 7 沒寫 | `results/icpe_h200/t1/gpu/exposure/t1g_exposure.json`（a1pp n_units 180、n_exposed 29）；EVIDENCE L195 | 表 7「剩下的風險」加「同節點他人 job 的暴露單位保留（拿掉後判定字不變）」 |
| 27 | 低 | §6 L213「最快的還原計畫也和閒時相同（0 翻轉）[S1]」 | B0 報告整體標 `provisional_because_noisy: true`；2 個 FLIP 組合因雜訊或沒量而跳過（含 Llama CPU 層＋cprefill 背景），379 格 2σ>5% 不判、84 列 NOT_MEASURED | `results/s3_h200/b0/b0_report.json` `skipped`、`noisy`、`not_measured` | 加「（2 個組合因雜訊未判）」 |
| 28 | 低 | §8.3 表 5 L278 vs 表 2 L80「單卡，不做 TP」 | G-M06 的 PASS 用了 2 張卡（借鄰卡 H2D＋NVLink），和設定表的單卡不一致（表 5 有寫「只有 2 卡」） | `results/s3_h200/gm06/gm06_report.json` | 表 2 註明 §8.3 例外 |
| 29 | 低 | 摘要 L16「0/16 個設定」；貢獻 2 | 16 是 GPU 列數（chat 4 列＋doc 兩個 seed 各 6 列），不是 16 個不同設定 | `results/icpe_h200/t1/gpu/a2/analysis/b2_verdict.csv`（src＝gpu 16 列） | 改「16 個 GPU 列」 |
| 30 | 低 | 摘要 L16「免費的打字提示在 GPU 上拿到 oracle 改善的 0.81／0.79」 | 沒寫範圍：1/8 縮小重播、SSD 級、Llama、CPU 16／32 GiB、TTFT 不含排隊。含排隊時 HINT-PF／O-PF 對 LRU 是 6.0／3.9% vs 8.1／4.7%（比例 0.75／0.83） | `p02_signs.csv` `headroom_gpu_incl_queue` | 摘要加「（縮小重播、SSD 級）」 |
| 31 | 低 | 圖 8 圖說 L254「收益來自位元組變小」 | 這是解讀，沒標〔判讀〕 | — | 加〔判讀〕 |
| 32 | 低 | 待決清單 L356–365；§7 待決 L236 | 待決 8、9 在 §8.1、§5.3 內文沒有 ⏳ 標記；§7 待決建議「不加 FreeInference 的條件」，但摘要、§7 L229、§11 已經寫了「只在 Mooncake 的兩條 trace 上」，內文先替待決做了決定 | PAPER L229、L350 | 內文補 ⏳，或把已採用的寫法從待決移除 |
| 33 | 低 | 摘要 L16；§1 L22–24；§10 表 8、L336–338 | 文獻性的說法（PBKV 等的對照組是「沒命中就重算」、Liu 等人的發現、*Where Should the KV Cache Live?* 說預取 oracle 沒用、Cake＝ICML'25）不在 EVIDENCE.md 裡，本稽核 NOT_VERIFIED | — | 交給 citation-audit／novelty-check |

## 附錄 A：自己從原始檔重算過的數字（41 組，全部 ✓，除非另註）

| PAPER 位置 | 數字 | 原始檔 | 重算結果 |
|:--|:--|:--|:--|
| 摘要、§3 | 實測÷模擬 0.911–0.987、10 格在 [0.9, 1.1] | `t1/gpu/p02/analysis/p02_cells.csv` | 0.9106–0.9867，in_band 全 True ✓ |
| §3 | 15 對排序、8 個正負號全同 | `p02_pairs.csv`、`p02_signs.csv` | judged＝15 全 same_order；8 列 same_sign ✓ |
| §3、§5.2 | O-PF 對 LRU 17.9／16.2%（模擬 24.2／21.1）；含排隊 8.1／4.7%；排隊 1.19–1.26 s | 同上 | 0.1792／0.1624（0.2424／0.2106）；0.0806／0.0466；ret_queue_mean 1.189–1.263 ✓ |
| §3 | 12 個乾淨 run | `p02_verdict.json` | n_clean_runs 12 ✓ |
| 摘要、表 3 | A2 GPU 0/16、模擬 0/16 | `t1/gpu/a2/analysis/b2_verdict.csv` | gpu 16 列 0 過、sim 16 列 0 過 ✓ |
| 表 3 | A1'' −0.37／−0.37／+0.01%，雜訊 0.03%，6 次 | `t1/gpu/a1pp/analysis/verdict.json` | −0.370／−0.366／+0.012；c4 0.0297；n_reps 6 ✓ |
| 表 3 | A1' +1.16／+0.09／−0.02%，雜訊 1.76% | `t1/gpu/a1p/analysis/verdict.json` | +1.160／+0.090／−0.021；1.762 ✓ |
| 表 3 | A3 各 2/5 seed、存活 0/2 | `t1/gpu/a3/analysis/verdict.json` | seeds_pass 2、2；n_survive 0 ✓ |
| 表 3 | A4 +17.3／+18.8 pp、慢 8.0／9.9%、M25 −4.9／−4.4、B25 −3.4／−1.9 | `m10_precision_h200/analysis/fp8/step2_qwen_seedA.json` | 0.1733／0.1881；0.71988/0.6666＝+8.0%、0.92872/0.84509＝+9.9%；−0.0493／−0.0438；−0.0336／−0.0188 ✓ |
| 表 3 | A4' −0.47～+0.08 pp、6 個 CI 含 0 | `.../int4/step2_llama_int4.json` | −0.0047～+0.0008，6 個 CI 跨 0 ✓ |
| §6 | 24 格 0 翻轉；慢層 16 格 Cake 快 16.5–87.5%；0.152 vs 0.238、0.068 vs 0.236 s | `icpe_h200/p01/p01b_cells.csv`、`p01b_verdict.json` | n_flips 0；0.1653–0.8749；0.1524/0.2379、0.0681/0.2365 ✓ |
| §5.2、圖 4 | B1：O-PF 50.5／46.5、O-ORD 27.6／31.0、O-DROP 27.6／34.9、O-NOW 8.0／7.4%，沒飽和 | `t1/p01/sim_V0/x1/x1_headroom_components.csv` | rule＝min 逐值相同；write_saturated 全 False ✓ |
| §5.2 | STAIR O-PF 42.9／39.4% | `t3/t3_compare.json` summary_t3.B1 | 0.4288／0.3940 ✓ |
| §5.2 | B2 47.7%（58 格）、H3600 0.0%、H1800 24.1%＋1 熱組、O-ORD 9／O-PF 11 熱組 | `x1{,h1800,h3600}_headroom_components.csv`、`*_hot_groups.csv` | 0.4768／58；0.0；0.2406／1；9／11 ✓（範圍問題見 #11） |
| §5.1 | X1 FI 0.30%、使用者發起 1.52% | `t1/p12/x1/x1fi{,user}_verdict.csv` | 0.003028、0.015189 ✓ |
| §5.1 | X5 FI 2.44%、H1800 2.44%、H3600 2.45%、0 格 ≥5% | `p12/x5/x5_verdict_fi{,_cens1800,_cens3600}.json` | 0.02437／0.02443／0.02450；reg_cells_ge5 0 ✓ |
| 表 4 | G_hit、G_rec、G_cake、LRU 命中率 5 列 | `t4/t4_cells.csv` | 逐值相同；由 ttft 欄重算 G_cake、G_rec，最大誤差 7e−6 ✓ |
| §5.1 | NFS 17.7／27.7%（閒）、19.1／29.2%（忙）、64 GiB 7.7%、128 GiB 0.3% | `t4/t4_cells.csv` | ✓；另用 T5 sweep 交叉驗算 O-BEL 對 LRU：0.1772／0.2767／0.1906／0.2919 ✓ |
| §5.1 | CPU＋慢層命中 99.42%（每個容量）、12 種全 NO-GAP | `t4_cells.csv`、`t4_verdict.json` | hr_tier_lru 99.4211（ret）；robustness 12 個 NO-GAP ✓ |
| §5.1 | T5：4 格 ≥5%（5.59／5.92／11.62／11.81%）、對手 S3-FIFO、沒飽和、0 格 ≥15% | `t5/p12/x1/x1fi_headroom_components.csv` | 0.0559／0.0592／0.1162／0.1181；S3FIFO；saturated 0 ✓ |
| §5.1 | O-PF 最大 4.89%、16 GiB 最大 −4.38%、最差 −29.5%；O-ORD 64 GiB 3.64%、SSD ≤1.66%；使用者 3.92／1.44% | 同上與 `x1fiuser_*` | 0.04886；−0.04379；−0.2949；0.03641；0.01660；0.03918／0.01437 ✓ |
| §5.1 | violations 53；828 列自檢 | `x1fi_violations.csv`、`t5_check.json` | 53（O-PF 18、O-NOW 14、O-ALL 21）；n_rows 828、mismatch 0 ✓ |
| 表 2 | FI 129,717／128,072／5,242 | `t1/p12/x1/x1_sweep_fi.csv.gz` 依週加總 | ✓ |
| 表 2 | conversation 12,031／3,511／2,357；toolagent 23,608／7,132／2,378 | `t1/p01/sim_V0/x1/x1_sweep.csv.gz` | ✓ |
| §7 | B5：11/48、10 格 CI ≥15%、最大 37.4%（toolagent）、CPU 128:6／256:4／512:1（15.2%）、b48 9／b32 2、閒 7／忙 4、conversation 6／toolagent 5 | `icpe_h200/p01/sim_V0/x5/x5_verdict.json` cells_ge15 | 逐格數過 ✓ |
| §7 | 嚴格 20／48.6%、H1800 11／37.1%、≤32K 9、Qwen 4/48／27.6% | `p12/x5/x5_verdict_{strict,cens1800}.json`、`x5_headroom_le32k.csv`、`tables_x5.md` | ✓ |
| §7、表 7 | B7 11／16／17 格、37.4／40.6／41.5%；X1 ≥15% 190／192／189 | `p01/sim_V{0,1,2}/x5/x5_verdict.json`、`t3/t3_compare.json` | ✓ |
| §5.2 | B4 0.983、最小 0.837、264 格、44/65 | `t1/p12/x4/p12_queue_verdict.json` | 0.9828、0.8371、264、44/65 ✓ |
| §5.3 | C1 62.2–72.0% | `t1/explore/x3/x3_hide_summary.csv.gz`（mooncake_conversation、nfs、a_typing、full） | 0.6217–0.7199 ✓ |
| §5.3 | C2 6.9–49.1%、7.0–54.2%、中位 0.933／0.941／1.118、最小 0.814、>1 的 4 格 | `t1/p02/fullsim_hint.csv` | 逐格重算中位 ✓ |
| §5.3、圖 7 | STAIR 最小 0.796、>1 的 2 格（全 NFS） | `t3/p02/fullsim_hint.csv` | 0.7957；NFS 128 閒／忙 ✓ |
| §5.3 | GPU：LRU 0.278／0.272、HINT 0.237／0.237、O-PF 0.228／0.228 s → 0.81／0.79（模擬 0.86／0.85）；O-PF 比 HINT 快 4.0／3.8% | `p02_cells.csv`、`p02_signs.csv` | 0.1452/0.1792＝0.810、0.1290/0.1624＝0.794；0.862／0.849；0.0398／0.0383 ✓ |
| §8.1 | D1 位元組、3.2 倍 | `m10_tier_h200/iowall.json` | 67,108,864／20,971,520；29,360,128／9,175,040 ✓ |
| §8.1、圖 8 | D2 Llama 37.4–49.1%、Qwen 36.7–54.5%；CPU −39.4／+13.0／−21.9／−80.0%；δ\* 28.4% | `m10_precision_h200/speed_{llama,qwen}/k0_verdict_int4.json` | ✓ |
| §8.1 | D2-Q(a) Qwen +0.4 [−1.1, +2.2]、Llama +1.0 [−0.4, +2.7] | `icpe_h200/p11/p11q_report.json` | +0.44 [−1.09, +2.16]、+1.02 [−0.35, +2.74] ✓ |
| §8.1 | D4 6/8、b32 忙 0.56–0.63 倍 | `icpe_h200/p11/p11t_cells.csv`、`p11t_report.json` | n_faster 6；0.565–0.628 ✓ |
| §8.2 | S1 18 格、背景 ≥0.91、4.28 倍 [4.26, 4.29]、0.980 | `s3_h200/b0/b0_report.json` | 18、0.912、[4.260, 4.290]、0.9797 ✓ |
| §8.2 | S2 30.7–62.9%、−0.04～6.5 pp、0.945、stream B 0.70–0.80 | `t1/e2e/mps_tradeoff.csv`、`e2e_report.json` | 0.3072–0.6285；−0.042～6.53；0.9449；0.697–0.801 ✓（計數見 #16） |
| §8.2 | S3 7 格同號、GPU 大 4.6–31.3 pp、4 格 >10 pp | `s3_h200/e2e_gpu/report.json` | 4.56–31.32、4 ✓ |
| §6 | E3 10/12、1.31–3.51、4／5／1、1.52–3.52 倍、30–100% | `icpe_h200/p2/p2_cells.csv`、`p2_verdict.json` | 1.305–3.507；穩 4、不穩 5、沒重現 1；1.521–3.524；0.3–1.0 ✓ |
| §3 | E4 −2.19～+0.94%、CI 上界 +4.02% | `icpe_h200/calib/calib_report.json` | d −0.0219～+0.0094；ci_hi 0.0402 ✓ |
| 表 5 | G-A0 −2.58 [−3.87, −1.34]、+1.28 [−0.27, +2.88]；G-M08 0/45、0.35–0.74；G-M12 −14.56／−8.64、0.47 倍；G-M06 1.986 | `s3_h200/a0/a0c_report.json`、`b0/b0_report.json`、`gm12/gm12_report.json`、`gm06/gm06_report.json` | ✓（0.0818/0.1741＝0.47） |
| 表 2 | CPU 43.5888／41.1687、SSD 6.981／1.911、NFS 0.331／0.601；b16/b32/b48；143,771 MiB、Gen5 x16、vLLM 0.28.0、torch 2.13.0+cu130 | `t1/tier_params_h200cpu.json`、`speed_*/spec_grid_int4.json`（經 sidecar）、`hw_h200.json` | ✓ |
| 表 7 | 271 列跨門檻全部往上；判定字 0 改變 | `t1/t1_compare.json` | 271 列全 new>old；word_changes 只在 best_rival／oracle_policy 等非判定欄 ✓ |
| §6、表 7 | STAIR 少 2.3–7.6 pp；SSD STAIR÷Cake 中位 0.953／0.935；慢 20/3,511、16/3,511，最多 2.05／2.28 倍 | `t3/t3_compare.json`、`t3_ratio.csv` | ✓ |

## 附錄 B：NOT_VERIFIED

- STAIR÷Cake 的 NFS 中位 0.491／0.625、「NFS 級 0 個比 Cake 慢」：沒打開 NFS 列。
- E1 的 harness 2.12／1.865 s（`m10_match_h200/tables_speed_14.md`）：沒打開；vLLM 1.2106 s ✓。
- R-T1G「數字變化 ≤1.2 pp」：沒逐列重算。
- D2-Q(b) 只核對了 Qwen INT4 E1 四種讀法的 eps_pass（A、B-idle 過；B-busy、B-i2b 沒過），沒重算 CI。
- #8 模擬的底與固定規則的差別對 headroom 的影響。
- 全部文獻說法（#33）。
- 圖 3、圖 5、圖 7 只核對 sidecar 與圖說，沒有逐點目視 PNG；圖 4、圖 6、圖 9 有看 PNG。
