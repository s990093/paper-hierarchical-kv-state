# 跑前登記（PREREG）：下一輪擴大驗證

> **性質**：事先登記。適用於 `h200`、`icpe-sim`、`icpe-data` 三個 session 這一輪的**所有正式結果**（NEXT_PLAN_20261104 的 W1–W7）。
> **生效**：本檔出現在 `origin/icpe-next-a` 的那一刻（以 push 後的 commit 為準）。在那之前只准試跑（計時、除錯），試跑輸出不進 `results/`（NEXT_PLAN §3b）。
> **怎麼定的**：NEXT_PLAN §6b 的草案由 `h200` session 逐項問使用者（2026-11-04，用選擇題），每一項的選擇記在 §2、§3；W6 的部分照使用者 11/4「H200 的 GPU 以後只做 W6 實測，現在不排。也排一下」一併登記（§5）。對手論文的數字有核對原文（arXiv HTML 全文），引文、版本與 sha256 見 [`W8_source_check.md`](W8_source_check.md)。
> **只有 `h200` session 改本檔**。其他 session 要改，寫進自己的 STATUS 檔，由使用者決定；改動一律寫進文末「偏離」。
> 標記：〔實測〕〔模擬〕〔算術〕〔判讀〕同 repo 慣例。

---

## 0. 登記時已經看過什麼

- **看過**：所有舊結果（X1 Mooncake、FreeInference 的 T1／T3／T4／T5、B6、C2、ICPE_EVIDENCE、`figs/ceiling_by_capacity.csv`）；`results/icpe_h200/agent/trace_stats.csv` 的紀錄特性（run `20261029-104721-icpe-agent-traces-char`；兩輪間隔中位數 1.1–2.9 s、p90 34–49 s，FreeInference／TraceLab／SemiAnalysis）；H200 上 vLLM 的 KV 池 log（§2 第 1 項）。
- **沒看過**：任何新紀錄（TraceLab、CacheWise、SemiAnalysis、Bailian）的上限模擬（NEXT_PLAN §5.1：從來沒跑過）；任何 GPU 層、TTL、8.07 GiB/s、慢層容量上限、雜訊的模擬數字；W1 的命中率（還沒算）。

## 1. 假設與判準（沿用，不變）

- **H1**：有簡單線索的地方，簡單線索拿走完美預取改善量的大部分。
- **H2**：對最強的不預測做法，剩下的空間集中在淘汰順序，不在預取。
- **H3**：上限隨 CPU 容量變大而下降。
- **推翻條件**：只要有一份真實的工具類紀錄，對最強的不預測做法上限 ≥15%，又沒有簡單線索能拿走大部分，結論改成「在這類工作負載上預測仍有空間」。
- **門檻**（使用者確認「Keep 15% / 5%」）：上限＝1 − TTFT(④) ÷ TTFT(②)，**≥15% 才算值得做**；兩個做法差 **≥5%** 才算有差異。主指標＝回來請求的平均 TTFT；p90（`p90_ret`）只報告。

## 2. §6b 的八項（使用者逐項確認）

| # | 設定 | 登記值 | 草案 | 依據／核對 |
|:--|:--|:--|:--|:--|
| 1 | **GPU 層的 KV 預算** | `gpu_memory_utilization`＝**0.92**（vLLM 0.28.0 的預設）→ KV＝0.92×139.79 − 15.81 − 1.69＝**111.11 GiB**〔算術〕→ 取整到 512-token 塊：**1,777 塊**（BF16 每塊 64 MiB）＝111.06 GiB＝909,824 token。存 **BF16** | 「vLLM 預設比例」 | 預設值：tiara 環境的 `vllm/config/cache.py:69`（`Field(default=0.92)`）；草案以為是 0.9，不是。算式各項取自 H200 上 Llama-3.1-8B 的 vLLM log：run `20261016-145911-m7h-r1-vllmf2-llama31-8b`（u＝0.85：總 139.79 GiB、權重＋非 torch 15.81、activation 峰值 1.69、KV 101.33 GiB＝830,080 token）〔實測〕、run `20261018-130851-m10h-cwe-vllm`（u＝0.80：KV 94.34 GiB＝772,832 token）〔實測〕；算式重現兩者到 0.01 GiB（101.32、94.33）。規則 6 驗算：830,080 token × 128 KiB＝101.33 GiB。**0.92 本身沒量過**；activation 是在 max_model_len 37,888／33,792 量的，更長的 max_model_len 下的 activation＝NOT_MEASURED |
| 2 | **預取搬到哪** | **主結果：CPU**（同現狀）。**敏感度：直接搬回 GPU**，跑在 §3 A 的全部舊格子上 | CPU（現狀，保守） | 登記時指出：對「預測沒多大優勢」這個主張，只預取到 CPU **不保守**——oracle 回來時還要付 CPU→GPU 的載入，上限會偏低，和「去對手最有利的地方測」相反；所以加 GPU 敏感度。GPU 層只存 BF16（第 1 項），預取進 GPU 的資料要先反量化、佔 GPU 預算（實作細節由 `icpe-sim` 在 G1 前寫進 STATUS_SIM）。新紀錄（W2）要不要也跑這個敏感度：**本檔未登記**，要跑先補登記 |
| 3 | **TTL 值** | **0.3、1、3、10、30、100、300 s**（7 個）。② 的 GPU 層在 {LRU、每個 TTL 值、前綴快取式} 裡逐格取最快（和現有 `RIVALS` 同一種「事後挑最快」，對對手有利）。**語意**：回合結束後這個對話的 KV 在 GPU 釘住 τ 秒；到期移到 CPU（BF16，進 CPU 層的正常規則）；GPU 滿而全部被釘住時，先放掉「對話開始時間最晚」的那個（照 Continuum §5.2 防死結的規則），移到 CPU | 0.3 s＋幾個更長的值 | CacheScout（2608.14624v1）§5.1：「Following the observed inter-turn latency in AutoGen, we set the TTL to 0.3 s」——0.3 s 是 AutoGen 自己的間隔，不是通用值。SAGA（2605.00528v2）§4.2（Algorithm 1）：TTL＝工具延遲的 p95、上限 `TTL_max=300s`。我們的 agent 紀錄間隔中位數 1.1–2.9 s、p90 34–49 s（`trace_stats.csv`），所以用對數間距從 0.3 s 掃到 300 s。Continuum 在「程式最後一步」主動放掉釘住——紀錄裡要預知才知道是最後一步，**不做** |
| 4 | **8.07 GiB/s 那一格** | 慢層讀 **8.07**、寫 8.07 × (1.9107 ÷ 6.9812)＝**2.209 GiB/s**〔算術〕、`k_read` 3.306（同 SSD 級）。其他設定和 SSD 級完全一樣；每個跑 SSD 級的正式格子都加跑這一格；只當敏感度 | 照 SSD 級的讀寫比例 | 8.07＝H200 計算節點 /tmp 的 64 MiB 循序讀、8 執行緒、6 次中位數（`docs/stage3_design_20261024/runs/G_fio.md`）〔實測〕；**H200 上沒量寫入**，寫入是比例換算。SSD 級讀寫取自 `results/icpe_h200/t1/tier_params_h200cpu.json` 的 `local`（6.98115、1.91065 GiB/s） |
| 5 | **慢層容量中間點** | 慢層上限＝⌊0.5 × `n_blocks_full`⌋ 個 INT4 塊（`n_blocks_full`＝這份紀錄所有不同的 512-token 塊數，和 `cpu_frac_ws` 的分母同一個塊數）。**主結果仍是慢層沒有上限**。慢層滿時：② ＝ O-ORD 那組固定規則（LRU、LRU2、ARC、S3FIFO、GDSF、PENS）用在慢層、逐格取最快；④ ＝ 最晚回來的先丟（Belady 式）。被丟的對話回來時全部重算。跑在全部正式格子上（第二批） | 全部 INT4 資料的一半 | NEXT_PLAN §5.9「慢層容量」 |
| 6 | **雜訊（W4）** | **兩種模型都跑**，主模型＝P。**模型 P（照 PBKV）**：每次抽雜訊時，以機率 λ 把這個對話的「下次回來時間」換成從**同一份紀錄**的經驗分佈隨機抽的值（分佈＝所有回合結束事件到該對話下一次請求的時間，「之後不再回來」照它的經驗比例算進去），否則用真值；λ＝**0、10、20、30、40%**。**模型 M（乘法誤差）**：剩餘時間 ×(1＋ε)，ε～U[−λ, λ]，λ＝**0、10、20、30%**；「不再回來」維持不再回來（這個模型不會猜錯生死）。細節見 §3 B | 回來時間 ×(0／10／20／30% 誤差) | PBKV（2605.06472v1）附錄 E.1（定義）、E.3（掃描）：雜訊是 P̃＝(1−λ)P＋λU，**把預測的下一個 agent 分佈和均勻分佈混合**，不是時間的乘法誤差；文中 λ 到 40%；「standard deviation over 10 trials」。所以只有模型 P 能寫「照 PBKV」 |
| 7 | **並行數** | **N＝16、32、64、72**（CacheWise、SemiAnalysis；固定並行：同時保持 N 個對話，一個結束就從佇列補下一個；對話內的間隔一律用紀錄的真值）。佇列順序見 §3 C；CacheWise 的複本見 §3 D；SemiAnalysis 不做複本（同 Fancy Eviction）。TraceLab：原始到達，另把各對話的開始時間壓縮 1×／2×／4×（NEXT_PLAN §5.6）。Bailian：原始時間，每份各自跑 | 16／32／64 | Fancy Eviction（2609.28870v2）§4.6：SemiAnalysis「artificially schedule sessions to maintain a steady concurrency of 16, while preserving the exact timing between requests within each session」；附錄 B 掃 8–32（HBM）到 1,024（pool）。PBKV §6.1：並行上限 72／24／48「to induce memory pressure」。72＝PBKV 的最大值 |
| 8 | **門檻** | 不變：上限 ≥15%、差異 5% | 不變 | 見 §1 |

## 3. 登記時補問的細節

- **A　舊紀錄用新模擬器重跑哪些格子**（使用者：「基本核心都把可以做的都做這樣之後好辦事」→ 全部）：**全部舊格子**。Mooncake＝`results/icpe_h200/t1/p01/sim_V1/x1/x1_sweep.csv.gz` 的整個網格（conversation／toolagent × R1／R2 × CPU 128／256／512／1,024 GiB × 全部策略 × ssd／b32／nfs × GPU 閒／忙 × cake／load）；FreeInference＝T1（`results/icpe_h200/t1/p12/`）、T4（`results/icpe_h200/t4/`）、T5（`results/icpe_h200/t5/`）的網格。每一格都跑新功能（GPU 層 1,777 塊、GPU 層的 ②／④、8.07 那格）；第 2 項的 GPU 預取敏感度也跑在這些格子上。
- **B　雜訊怎麼抽**（同上 → 全部）：**兩種抽法都跑**——B1「每個回合結束抽一次，到這個對話下一次請求前不變」、B2「oracle 每次被問都重抽」。兩種模型 × 兩種抽法 × 每個 λ>0 各 **10 個 seed**；λ＝0 只跑一次，且必須和沒加雜訊的 oracle 逐值相同（自檢）。雜訊套在 oracle 的每一種用途：預取（O-PF）、淘汰順序（O-BEL／O-BELS、GPU 層的 Belady、慢層的 Belady）、丟不丟（O-DROP、O-NOW）。報 seed 間的中位數、最小–最大、平均±標準差。範圍：④ 對 ② 上限 ≥15% 的每一格、每個元件（NEXT_PLAN W4）。〔`h200` 的解讀：B1 當主線，因為同一對話的排序前後一致；使用者沒有另外指定主線。〕
- **C　固定並行時的佇列順序**（使用者選「Shuffled, 3 seeds」）：對話順序用 **3 個 seed 隨機打亂**，報中位數與範圍。和雜訊的 seed 交叉（3 × 10）。
- **D　CacheWise 的複本**（使用者選「Always 5 copies (470 sessions)」）：每個 N 都用 **5 份複本、470 個對話**；每份換新的 session id、複本之間不共享前綴（照 Continuum 把紀錄重複 1–5 次的上限）。

## 4. 沿用計畫已定的（不在 §6b，但開跑前一併凍結）

- **紀錄**：必跑＝TraceLab、CacheWise、SemiAnalysis、Bailian（NEXT_PLAN §5.5）。選跑的（mini-SWE-agent、OpenHands、Terminal-Bench、ShareGPT＋Pensieve、InferCept、Exgentic、Codex、LMCache）**不在本檔**，要跑之前另補登記。CacheWise 沒有授權：原始資料、轉換後的紀錄、逐對話的數字都不進 git。
- **紀錄轉換**：照 `code/p3x/x1_trace.py`、`code/p3x/x1_fi.py`：512-token chunk、完整長度；每個載入函式寫「用 A 欄驗算 B 欄」（CLAUDE.md 規則 6）。
- **常數**：Llama-3.1-8B；`P3_CONST=h200 P3_TIER=h200cpu M7_TIER_PARAMS=results/icpe_h200/t1/tier_params_h200cpu.json`；重算 f(i) 用 V1（`results/icpe_h200/p01/calib`，第 80 塊以後線性外插）；還原用 Cake（`X1_PLANNER` 不設），只載入（load）當敏感度。
- **上下文長度**（NEXT_PLAN §5.8）：主結果用完整長度；敏感度 1＝每個對話遇到第一個 >128K 的請求就截斷（報被截掉的請求比例）；敏感度 2＝容量也用「佔工作集的比例」（`cpu_frac_ws`）報。
- **慢層頻寬**：SSD 級 6.98／1.91、b32 2.49／2.49、NFS 級 0.33／0.60 GiB/s（讀／寫），加第 4 項的 8.07 那格。
- **CPU 容量**：8、16、32、64、128、256、512、1,024 GiB（涵蓋 T4／T5 的 8–64 與 X1 的 128–1,024），加參考點「主機層＝GPU KV 預算」＝1,777 塊＝111.06 GiB（NEXT_PLAN §5.9，對齊 HiCache 預設）。
- **GPU 閒／忙**：都跑。
- **比較層級**（NEXT_PLAN §5.10）：① LRU（GPU 層也 LRU）；② 每個動作在固定規則裡取最快（`x1_sim.py` 的 `RIVALS`＋GPU 層的 LRU、第 3 項的 TTL、前綴快取式），慢層存 INT4、Cake 還原；③ 打字就預取（只有聊天）；④ 完美預測的四個元件（O-PF、O-BEL／O-BELS、O-DROP、O-NOW；O-ALL 只報告）＋GPU 層的 Belady；⑤ 第 6 項的雜訊。
- **資料怎麼流動**（NEXT_PLAN §5.10）：滿了才逐出；回合結束留在 GPU；GPU 滿 → 逐出到 CPU（BF16）；CPU 滿 → 慢層（寫 INT4）或丟；回來時從所在的層用 Cake 搬回 GPU；預取照第 2 項。
- **前綴快取式（vLLM APC）在 session 粒度模擬器裡的確切定義**：由 `icpe-sim` 在 G1 前寫進 STATUS_SIM；若改變 ② 的組成，照文末「偏離」程序補登記，而且要在看任何正式結果之前。
- **標記**：每個結果標明到達方式（原始／固定並行 N／壓縮 k×），合成到達的結果和原始到達的分開報。
- **不在本檔、要跑之前另補登記**：排隊模型、W3 的線索 (a)、W5、W7、選跑的紀錄。

## 5. W6（GPU 實測；登記，**現在不排**）

使用者 11/4：「H200 的 GPU 以後只做 W6 實測，現在不排。」排程時照記憶的規則：計時的工作不和自己的其他 GPU 工作同時跑（`--dependency=afterany` 串起來）、實驗室上限 4 張卡、`gpu_guard` 必用、被污染的 run 不進 `results/`。

- **順序**（使用者選）：**W6c → W6a → W6b**。
- **W6c　重算校準延伸到 128K**：照 P0-1 的 C1（`m7_calib.py c1 --max-chunks 80 --reps 3`，run `20261016-061311-m7h-c1`）的量法，只把塊數改成 **256**（＝128K token）。**用法**（使用者選「Sensitivity; sims don't wait」）：主結果維持現在的常數（V1，40K 以後外插），正式模擬不等 W6c；W6c 出來後，用實測的 f(i) 重跑代表格當敏感度。**代表格是哪些：本檔未定**，要在看 W6c 結果之前補登記。128K 以上仍是外插。
- **W6a　C2「打字就預取」的穩定度**（使用者選「6 reps」）：設定同 C2 的 GPU 實測（`docs/h200/ICPE_EVIDENCE.md` 的 C2 列：縮小 k＝8、CPU 16／32 GiB、GPU job 21655），每個設定 **6 次**重複；報中位數與範圍。**只要有任何一次**完美預取比打字提示快 ≥5%，就照實寫；6 次全部 <5% 才寫「打字提示拿走大部分」。
- **W6b　Mooncake 對話「淘汰順序」的 GPU 縮小重播**（使用者選「Include, after code is ready」）：要先讓 `code/icpe/x1_gpu.py` 支援 oracle 淘汰順序。〔更正：問題裡說這支程式歸 `icpe-sim`；其實它在 `code/icpe/`，不在 NEXT_PLAN §3b 任何一個 session 的歸屬裡，由誰改要等排 W6 時問使用者。〕判準沿用 P0-2 §8 的三條（實測÷模擬在 [0.9, 1.1]；模擬差 ≥5% 的配對排序相同；headroom 正負號相同）。

## 6. 偏離程序

本檔 push 之後的任何改動：在文末「偏離」加一節，寫 `date -u` 時間、改什麼、為什麼、改的當下已經看過哪些正式結果。只有 `h200` session 寫本檔；`icpe-sim`、`icpe-data` 要改的寫進自己的 STATUS 檔，由使用者決定。

---

## 登記

- 登記時間（`date -u`）：2026-11-04T15:12:02Z（開始寫本檔時；問完使用者之後）
- 登記時 repo HEAD：`77c25f6`（本檔隨下一個 commit 進 git，push 到 `origin/icpe-next-a` 時生效）

## 偏離（登記之後）

### 偏離 1（2026-11-04T15:16:11Z；使用者補答登記時未定的三項；`h200` 這邊當下沒有看過任何正式結果，W1 也還沒算）

使用者原話（回覆 STATUS_A「需要使用者決定」的三題）：「都做 gpu 的話 h200 做」。

- **第 2 項（預取搬到哪）**：直接搬回 GPU 的敏感度**也跑新紀錄（W2）的全部正式格子**（原本只登記了 §3 A 的舊紀錄格子）。
- **§5 W6c**：W6c 出來後的敏感度重跑＝**全部正式格子**（舊紀錄＋新紀錄）。只把 f(i) 換成 W6c 實測的值（量到 256 塊；256 塊以後仍是外插），其他設定不變。
- **§5 W6b**：`code/icpe/x1_gpu.py` 由 **`h200` session** 改（GPU 的工作都歸 `h200`）。W6 仍然**現在不排**。

### 偏離 2（2026-11-04T15:22:48Z；使用者在 `icpe-sim` 決定、在 `h200` 確認要登記；當下 `h200` 看過的正式結果只有 W1（用舊輸出算的命中率，和本偏離的項目無關）；任何新模擬的正式結果都還沒有——G1 還沒做）

來源：`origin/icpe-next-sim` @ `fabb81d` 的 `docs/icpe_brief_20261030/next/STATUS_SIM.md`。使用者在 `h200` 回答「Yes, register all 3」（下面 1–3），另外兩題選「Unrounded: 8.068 / 2.2081」（4）、「Yes, record it」（5）。

1. **TTLB 加進 ②**。STATUS_SIM 2026-11-04T23:19+08:00 原文：「**TTLB 加進 ②**，請 `h200` 在 PREREG 寫偏離（要在看任何正式結果之前）。→ ② 的 GPU 規則＝{LRU、TTLA:τ、TTLB:τ（τ＝0.3／1／3／10／30／100／300 s）}；④＝BEL。」語意（同節上一段 23:17）：TTLA＝到期就逐出到 CPU（＝第 3 項原本登記的語意）；TTLB＝到期後照 LRU，GPU 滿時才逐出。兩者用同一條防死結規則（GPU 滿而全部被釘住時，放掉「對話開始時間最晚」的那個；對話開始時間＝該 session 第一個請求的到達時間）。
2. **直寫包含 `wt` 登記成敏感度，跑全部格子**。原文：「**直寫包含 `wt` 登記成敏感度，跑全部格子**（請 `h200` 寫偏離）。」主結果仍是回寫互斥 `wb`（§4 的資料流）。`wt`＝CPU 層照舊，GPU 另外多存一份，從 GPU 逐出不用寫回。
3. **APC 分兩批**。原文：「**APC：第一批＝GPU-LRU（session 粒度下完全相同，不另外跑）；第二批再加 block 粒度**。……請 `h200` 現在就登記『block 粒度 APC 做好後加進 ②；在那之前的結果標明 ② 不含 block 粒度 APC，做好後重算』。」→ 第一批的每個結果都要標「② 不含 block 粒度 APC」；block 粒度 APC 做好後，受影響的格子全部重算。
4. **8.07 那格改用未四捨五入的實測值**：慢層讀 **8.068 GiB/s**（`results/s3_h200/gfio/gfio_20261026-104426-s3-gfio.csv`，/tmp、循序 64 MiB、8 執行緒、6 次中位數〔實測〕），寫 8.068 × (1.9106543 ÷ 6.9811523)＝**2.2081 GiB/s**〔算術〕，`k_read` 3.306 不變。取代第 4 項的 8.07／2.209（四捨五入後的值；`icpe-sim` 23:17 用的是讀 8.068、寫 2.2087，兩個數字不是同一個讀值算的）。
5. **補充說明（不改設定）：② 的組合方式**。STATUS_SIM 23:17 使用者決定 4 原文：「比較：**每個 CPU 對手都配**（GPU 規則 × CPU 規則全部組合，各取最快）。」→ ② 在每一格是「GPU 層規則 × CPU 層規則」全部組合裡最快的那個。

### 偏離 3（2026-11-04T15:36:14Z；`h200` 準備 W6 的程式時發現 §5 有兩處引用錯誤，更正；另把 W6c、W6a 的實作細節寫死。W6 還沒有任何資料，GPU 沒有排）

使用者 11/4 對「接下來做什麼」回「你決定」；`h200` 決定先把 W6 的 runner 與分析寫好（不送出）。

1. **更正 §5 W6c 引用的校準 run**：模擬用的 V1 常數（`results/icpe_h200/p01/calib/calib_c1.csv`）來自 run **`20261022-205934-icpe-p01-c1`**（Slurm job 21653，`code/r_h200_icpe.sh` 的 `P01-C1`）：`M7_ATTN=fa3 M7_RES=<輸出目錄> python code/m7_calib.py c1 --reps 3 --max-chunks 80`（fa3＝P0-1 §2.3 的 Llama 勝者）。§5 原本寫的 `20261016-061311-m7h-c1` 是 V0（sdpa）的校準，**不是**模擬用的那份。→ W6c＝同一條指令（`M7_ATTN=fa3`），只改 `--max-chunks 256`；另外用 `code/m7_guard_run.py` 包 GpuWatcher（原本那次沒有包），被污染就照暴露重跑、最多 3 次。輸出 `results/icpe_next/a/w6c/`。
2. **W6c 敏感度怎麼換 f(i)**（偏離 1 說「只把 f(i) 換成 W6c 實測的值」；`h200` 的解讀，使用者可以改）：**整份換**——第 0–255 塊都用 W6c 這一次量的中位數，256 塊以後用 W6c 256 個點的線性外插（`p3_fvar.c1_f` 同一套）；**不**把 V1 的 0–79 塊和 W6c 的 80–255 塊接在一起（兩天量的混在一條曲線上）。GPU 忙倍數照用 V1 的 `calib_c5.csv`。0–79 塊 W6c 對 V1 的比值只報告。給模擬用的目錄＝`results/icpe_next/a/w6c/calib_for_sim/`（`p3_fvar.py --variant V1 --calib <這個目錄>`）。
3. **更正 §5 W6b**：§5 寫「要先讓 `code/icpe/x1_gpu.py` 支援 oracle 淘汰順序」——**它已經支援**：`x1_gpu.py` 的 `POLICIES5`＝LRU、FREQ2Q、O-BEL、HINT-PF、O-PF，T1G 已經在 GPU 上跑過 O-BEL（`results/icpe_h200/t1/gpu/p02/`，k＝8、CPU 128／256 GiB 縮小成 16／32）。W6b 真正缺的是 **O-ORD 的其他固定對手**（LRU2、ARC、S3FIFO、GDSF、PENS）不能在重播裡跑。W6b 要跑哪些做法、幾次：在 `x1_gpu.py` 改好之後、跑之前另外登記。
4. **W6a 的實作細節**（不改判準）：新的 6 次（rep 0–5）寫到新的目錄 `results/icpe_next/a/w6a/`，**不**沿用 T1G 舊的 rep 0；跑的順序＝6 輪、每輪 6 個設定（CPU 128／256 × LRU／HINT-PF／O-PF）、輪內用 seed 20261105＋輪次打亂；被污染照暴露重跑、最多 3 次；分析程式 `code/icpe/next_a_w6_analyze.py w6a`（每個 rep 內配對：gap＝1 − TTFT(O-PF) ÷ TTFT(HINT-PF)，判定字 ANY_GE_5／ALL_LT_5／INCOMPLETE）。
5. **runner**：`code/icpe/next_a_w6.sh`（`--part w6c|w6a`；`--dry-run` 已在 head node 跑過）。檔名不在 NEXT_PLAN §3b 的 `next_a_*.py` 樣式內（是 `.sh`），但只有 `h200` 用。

### 偏離 4（2026-11-04T16:15:11Z（H200 head node 的 `date -u`；STATUS_DATA 節名的 00:20+08:00 是 acvlab 的鐘）；使用者在 `icpe-data` 決定「方案 A」、在 `h200` 確認要登記；當下 `h200` 看過的正式結果只有 W1（和本偏離無關）；`icpe-data` 寫明正式模擬還沒開始、G1 還沒過）

來源：`origin/icpe-next-data` @ `3ca8334` 的 `docs/icpe_brief_20261030/next/STATUS_DATA.md`，2026-11-05T00:20+08:00、00:15、2026-11-04T23:55 三節（以那份為準，下面是摘錄）。使用者原話（00:20 節）：「太久了 不行 目標5 天內搞定」「11/9 要搞定 注意喔」。使用者在 `h200` 回答「Yes, register all」，另外兩題選「Representative cells」（第 6 點）、「No, keep it queued」（第 7 點）。

**取代**本檔先前「跑全部格子」的幾處：第 2 項與偏離 1 的「預取進 GPU 跑全部格子」、偏離 2 第 2 點的「`wt` 跑全部格子」、第 4 項的「每個跑 SSD 級的格子都加跑 8.07」、§3 A 的「舊紀錄全部格子都跑新功能」、偏離 1 的「W6c 敏感度跑全部正式格子」→ 一律改成下面第 4 點的代表格。

1. **CPU 層規則 23 → 17**：拿掉 6 個事後穩健性版本（`O-DROP-H1800／-H3600`、`O-NOW-H1800／-H3600`、`O-ALL-H1800／-H3600`）。17＝LRU、LRU2、ARC、S3FIFO、GDSF、PENS、FREQ2Q、LRU+NONE、LRU+2HW、LRU+2HA、LRU+RPF、O-BEL、O-BELS、O-DROP、O-NOW、O-PF、O-ALL。
2. **TTL 值 7 → 3**：**0.3、10、300 s**（TTLA、TTLB 都是）；GPU 層規則＝LRU、BEL、TTLA:{0.3, 10, 300}、TTLB:{0.3, 10, 300}（8 個）。少了 TTL 值 → ② 只會變弱 → 上限只會偏高。**事先登記的補跑規則**：主結果裡 ④ 對 ② 上限 **≥10%** 的每一格，用第 3 項原登記的 7 個 TTL（0.3／1／3／10／30／100／300 s）補跑 ②，報告用補跑後的值。〔`h200` 註：② 只會隨 TTL 值變多而變快，所以 3 個 TTL 下 <10% 的格子，7 個 TTL 下也 <10%，不會跨過 15% 門檻。〕
3. **主結果＝所有登記的紀錄變體 × 主時間設定**（Cake × 慢層 ssd／b32／nfs × GPU 閒／忙）：CacheWise 12（N 16／32／64／72 × 佇列 seed 1／2／3；前綴＝結構推得）、TraceLab 3（開始時間壓縮 1×／2×／4×；前綴＝結構推得）、SemiAnalysis 12、Bailian 3（traceA、thinking、coder）、Mooncake 4（conversation／toolagent × R1／R2）、FreeInference 6 週；CPU 9 個容量照 §4（8–1,024 GiB＋111.0625）。
4. **敏感度只跑代表格**：CacheWise N16 s1、TraceLab c1、SemiAnalysis N16 s1、Bailian coder、Mooncake conversation R1 與 toolagent R1、FreeInference 第 1 週（`2026-05-17_2026-05-23`）× CPU **111.0625、128 GiB**。敏感度＝nvme807（Cake）、只載入（4 個慢層）、`wt`、預取進 GPU、快取命中前綴（只有 CacheWise／TraceLab）、截斷 128K（§4 敏感度 1）。
5. **紀錄的處理**（STATUS_DATA 23:55、00:15）：
   - CacheWise 可用對話＝**80**（12 個原始檔 0 byte、2 個沒有 LLM 呼叫；108 個 session＝80 主＋28 子 agent）→ §3 D 的 5 份複本＝**400** 個對話（§3 D 的「470」是用 94 算的）。
   - CacheWise／TraceLab 的「可以還原的前綴」：**主結果＝結構推得**（輸入沒有變短、中間沒有 compact_boundary → 上一輪整段輸入都算前綴；否則退回快取命中）；**敏感度＝快取命中**（NEXT_PLAN §5.6 原文的做法；供應商快取 5 分鐘／1 小時沒用就失效，長間隔回來會被當成沒有前綴）。
   - Claude Code resume 複本**不去重**（照上游作者的做法）；論文限制寫明 CacheWise 16.6%、TraceLab 2.2% 的請求是複本。
   - **Bailian traceB 跳過**：172,800 個請求全部 turn＝1、parent＝−1，沒有「回來的請求」，主指標沒有定義；論文寫明。
6. **W6c 敏感度**（使用者在 `h200` 選）：跑第 4 點的代表格，不跑全部正式格子。
7. **W6 不趕 11/9**（使用者在 `h200` 選）：job 24443／24444 照原樣排在 DL_Lab5 後面；趕不上的寫進 CCGrid 或限制。
8. 不在方案 A 裡、另外算的：W4 雜訊、W3、第二批功能（慢層容量、block 粒度 APC、排隊）。照 STATUS_DATA 00:20 節，它們的格子另外定、跑之前登記。

### 偏離 5（2026-11-05T00:59:15Z（acvlab 的 `date -u`）；**偏離 4 第 2 項的釐清**，不改偏離 4 原文；使用者在 `h200` 選「交給 acvlab 寫」（`origin/icpe-next-a` @ 20fe61a 的 STATUS_A「09:05」節），由 `icpe-data` 寫；當下 `icpe-data` 已經看過的正式結果列在本節最後）

**釐清**：偏離 4 第 2 項「用第 3 項原登記的 7 個 TTL 補跑 ②」——補跑**只擴充 ② 的 GPU 層不預測規則**。
- 擴充的內容：在原本的 LRU、TTLA／TTLB:{0.3, 10, 300} 之外，加 TTLA／TTLB 的 1、3、30、100 s，配 10 個不預測的 CPU 規則（`x1_sim.NON_ORACLE`）。
- **④ 的 GPU 規則集合維持偏離 4 的 3 個 TTL**。
- 受影響的只有 4 個 CPU 元件（O-ORD、O-DROP、O-NOW、O-PF）：它們的 ④ 也在 GPU 不預測規則裡取最快。O-GPU（④＝GPU BEL）和 O-ALL(report)（④＝O-ALL × BEL）的 ④ 不含 GPU 層的 TTL，不受影響。
- 分析指令：`next_sim_analyze.py ... --rival-only-gpu TTLA:1 TTLA:3 TTLA:30 TTLA:100 TTLB:1 TTLB:3 TTLB:30 TTLB:100`（`icpe-sim` 9acfec1；每次執行都自檢：格子集合不變、沒有任何上限變大，否則停）。

**為什麼要釐清**：`next_sim_analyze.py` 原本兩邊用同一組 GPU 規則。補跑只加在 ② 時，補跑過的格子裡 ④ 缺新 TTL 的列，4 個 CPU 元件就被整列不算。CacheWise 因此剩 2,896 列，補跑前是 3,888 列（run `20261105-084414-icpe-data-analyze-cw_main-ttl7`）。原文沒寫 ④ 要不要跟著加。

**理由**：
1. 偏離 4 原文寫的是「補跑 ②」。
2. 偏離 4 的 `h200` 註「② 只會隨 TTL 值變多而變快，所以 3 個 TTL 下 <10% 的格子，7 個 TTL 下也 <10%」**只有在 ④ 不變時才成立**。若 ④ 也加 TTL，上限可能變大，沒有補跑的 <10% 格子就沒有保證，≥10% 的選格規則就失效。
3. 方向保守：④ 不變、② 只會變快 → 補跑後的上限只會變小或不變。
4. 不需要多跑模擬：已跑的 `cw_main_ttl7`（STATUS_DATA 07:20、09:00）就是這個集合。
- 另一種做法（④ 也加 7 個 TTL，要多跑預測的 CPU 規則 × 8 個 TTL）**沒有跑**：NOT_MEASURED。

**出處**：STATUS_DATA 9f9e291（請求 2）；STATUS_A 20fe61a（09:05）；`icpe-sim` 9acfec1（STATUS_SIM 08:57，驗證 run `20261105-085502-icpe-sim-analyze-rival-only`）。

**寫本節時 `icpe-data` 已經看過的正式結果**（照實列出；寫本節的動機是上面那個「整列不算」的機制問題，不是看了數字之後挑方向）：
- CacheWise 主結果 3 個 TTL 的分析（run `20261105-071035-icpe-data-analyze-cw_main-ttl3`；先前的 `…070937…` 內容相同、run_id 寫錯已作廢）：
  - 12 個變體 × 元件的摘要：格數、最大值、中位數、≥15% 與 ≥5% 的格數。
  - ≥10% 的格子：225 格，以及各 CPU 容量的分布。
  - violations 221 列的摘要（最多 6.4%）。
- 同一批加上補跑、沒有開新選項的分析（run `20261105-084414…`）：列數 2,896、被丟掉的是哪 4 個元件；72 列上限變小、0 列變大；violations 42 列；stdout 最後一列摘要。
- 各 sweep 進度列裡個別 job 的 `mean_ret`：CacheWise 補跑、TraceLab c1 前幾十個 job、xcheck。
- 其他紀錄（TraceLab、SemiAnalysis、Mooncake、FreeInference、Bailian）的上限：都還沒有結果，沒有看過。

### 偏離 6（2026-11-05T16:09:01Z（acvlab 的 `date -u`）；**補回偏離 4 砍掉的格子（P1–P4，照優先度）＋換用 Rust 引擎的紀錄**；使用者在 `icpe-ana` 決定（`origin/icpe-next-ana` @ 1ea44c9 的 STATUS_ANA「2026-11-06T00:20」節），由 `icpe-data` 寫；P1 的「④ 的 GPU 規則要不要也擴」由使用者在 `icpe-data` 回答；當下 `icpe-data` 已經看過的正式結果列在本節最後）

使用者的決定：
- STATUS_ANA 00:20 原文：「被砍掉的格子**全部補**，照優先度：P1 → P2 → P3 → P4。」「PREREG 偏離 6 由 icpe-data 寫」。
- 使用者在 `icpe-data` 的回答（AskUserQuestion，11/6）：「④ 也擴 7 個」。

**1. 換用 Rust 引擎**（不改任何格子的結果，所以不改分析計畫，只記錄）
- 用法：`x1 sweep --engine rust`（或 `$X1_ENGINE=rust`）。程式在 `code/p3x/x1fast/`，是 `icpe-ana` v3 原樣複製（`origin/icpe-next-ana` aee1527 的 `code/icpe/ana_x1fast/v3_prefix_fork`）。不支援的功能（W4 雜訊、慢層容量上限、STAIR）由該 job 自動改用 Python。
- 從哪裡開始用：`icpe-sim` 審查、包裝，並在 STATUS_SIM 寫明 G1（新功能關閉的 4,416 列 × 29 欄逐位元相同）和 xcheck（xc_sa、xc_bl）PASS 的那個 commit 起。各機器 merge 之後還要再做自己的 G0（acvlab）或同等的對拍（TRX40）。開始用的 commit 寫進各自的 STATUS 和每個 run 的 context。
- 只用在還沒開始的 block。已經用 Python 跑完或正在跑的 block 不重跑。
- 依據（`icpe-ana` 的對拍，`results/icpe_next/ana/`，STATUS_ANA 23:58 節）：
  - Bailian coder 278／278 個 job、Mooncake conversation 68／68 個 job，逐請求陣列與 `summarize` 的每一欄逐位元相同。
  - SemiAnalysis N16 s1 整個正式 block：7,344／7,344 列和 `origin/icpe-next-trx` 的 `sa_main/sa_N16_s1` 在 CSV 的精度內相同。

**2. 補回偏離 4 砍掉的格子**（取代偏離 4 第 1、2 點，以及第 4 點裡 nvme807、只載入、`wt`、預取進 GPU 這 4 項的縮小；照 P1 → P2 → P3 → P4 的順序執行）
- **P1**
  - **截尾版規則**：CPU 層規則加回偏離 4 第 1 點拿掉的 6 個：`O-DROP-H1800／-H3600`、`O-NOW-H1800／-H3600`、`O-ALL-H1800／-H3600`。
    - 照 `x1_sim.py`／`next_sim_analyze.py` 的 `--dead-h` 既有定義，**只報告**，不改主判定。
    - 範圍：主結果的全部格子（偏離 4 第 3 點）。GPU 規則和 P1 的其他規則相同。
  - **② 用 7 個 TTL**：主結果的**全部格子**的 ② 都用第 3 項原登記的 7 個 TTL（0.3／1／3／10／30／100／300 s，TTLA、TTLB）。這取代偏離 4 第 2 項「只補跑上限 ≥10% 的格子」。
  - **④ 也擴到同樣的 7 個 TTL**（使用者在 `icpe-data` 選）：
    - 4 個 CPU 元件（O-ORD、O-DROP、O-NOW、O-PF）的 ④，在 {LRU、TTLA／TTLB × 7 個 TTL} 裡取最快。這和 ② 的 GPU 不預測規則集合相同。
    - O-GPU（④＝GPU BEL）和 O-ALL(report) 不受影響。
    - 分析：主結果**不加** `--rival-only-gpu`（兩邊用同一組 GPU 規則）。偏離 5 的算法（④ 維持 3 個 TTL，加 `--rival-only-gpu`）從同一批輸出也算得出來，**只並列報告**。
  - **為什麼 ④ 也擴**：
    - 偏離 5 讓 ④ 維持 3 個 TTL，理由是保證 ≥10% 的選格規則成立。全部格子都補跑之後不需要選格，這個理由不適用。
    - ② 比 ④ 多 GPU 規則可挑時，上限會被算低，而上限低對「預測空間不大」的主張有利。偏離 5 寫的「方向保守」只就選格規則而言，對主張其實不保守。兩邊規則集合相同，才是公平比較。
    - 要多跑的：已跑的 0.3／10／300 之外，TTLA／TTLB 的 1／3／30／100 s（8 個 GPU 規則），配 10 個不預測的 CPU 規則（`x1_sim.NON_ORACLE`）和 5 個元件的 CPU 規則（O-BEL、O-BELS、O-DROP、O-NOW、O-PF；`x1_sim.COMPONENT_POLICIES`）。
- **P2**：nvme807（cake）與只載入（ssd／b32／nfs／nvme807），從偏離 4 第 4 點的代表格擴到主結果的全部格子（全部登記的紀錄變體 × CPU 9 個容量 × GPU 閒／忙）。
- **P3**：`wt` 與預取進 GPU，同樣擴到主結果的全部格子。
- **P4**：W4 雜訊，照第 6 項與 §3 B（兩種模型、兩種抽法、λ、λ>0 各 10 個 seed、λ＝0 自檢；和 §3 C 的佇列 seed 交叉）。範圍照 §3 B：④ 對 ② 上限 ≥15% 的每一格、每個元件。
- **`icpe-data` 的解讀**（使用者可以改）：
  - P2、P3 的規則集合和 `sens_rep` 相同：偏離 4 的 17 個 CPU 規則 × 8 個 GPU 規則（3 個 TTL）。P1 的 7 個 TTL 只用在主時間設定。
  - P4 的 ≥15% 用 P1 之後的上限選格。
  - 某份紀錄的某個優先度還沒做完時，它的報告值照偏離 4／5（含 ≥10% 補跑），並標明是哪一版。做完之後，報告值換成本節的版本，舊版保留。
- **不在本節**：偏離 4 第 4 點的快取命中前綴、截斷 128K，以及第 6 點（W6c）。三者維持代表格。

**3. 判準不變**：15%／5%，主指標＝回來請求的平均 TTFT。第一批的結果仍標「② 不含 block 粒度 APC」（偏離 2 第 3 點）。

**寫本節時 `icpe-data` 已經看過的正式結果**（偏離 5 列過的之外，照實列出）：
- **CacheWise 最終上限表**（偏離 5 的算法；run `20261105-085942-icpe-data-analyze-cw_main-final`）：
  - stdout 的摘要：12 個變體 × 6 個元件的格數、最大值、中位數、≥15% 與 ≥5% 的格數。
  - 3,888 列、0 列上限變大、213 列變小；violations 369 列。
- **TraceLab c1**（`tl_main/tl_c1`）、**Mooncake**（`old_main/mooncake`）、**`sens_rep`** 19 個 block、**xcheck**：
  - 只看了列數、表頭、TraceLab c1 的第一列，以及 sweep 進度列裡個別 job 的 `mean_ret`。
  - xcheck 另外和 TRX40 逐值比較（0 格不同）。
  - 都**沒有算上限**。
- **SemiAnalysis**（`icpe-trx` 的 `sa_main`）：沒有打開。STATUS_TRX 只讀了進度。
- **STATUS_ANA 23:58、00:20 兩節**：讀到文字描述，例如近似方法「會翻 15%／5% 判定」。沒有打開它的數字檔。
- **本節的動機**：
  - 補格子是使用者的決定。
  - ④ 的問題是 `icpe-ana` 草稿提出的機制問題，不是看了數字之後挑方向。
  - ④ 擴充後，上限只會變大或不變，方向對我們的主張不利。

### 偏離 6 補記（2026-11-05T23:49:08Z（acvlab 的 `date -u`）；**改偏離 6 第 1 點「只用在還沒開始的 block」**；使用者在 `icpe-data` 決定（AskUserQuestion，11/6 03:25 提問、07:4x 回答），由 `icpe-data` 寫；不改任何格子、規則、判準）

使用者的決定：選項「兩個都停，全改 Rust」——停掉 acvlab 上兩個正在跑的 Python 佇列，剩下的 block 全部用一個 Rust 佇列跑。

**改什麼**
- 偏離 6 第 1 點「只用在還沒開始的 block。已經用 Python 跑完或正在跑的 block 不重跑」改成：
  - 已經用 Python **跑完**的 block 不重跑（不變）。
  - 正在用 Python 跑的 `tl_main/tl_c4`、`old_main/fi6w` **停掉、整個 block 用 Rust 從頭重跑**。
  - Python 跑到一半的輸出留在 /ssd6 的 run 目錄，**不收、不用**。同一個 block 不會一部分用 Python、一部分用 Rust。
- 新增：`icpe-sim` 在 STATUS_SIM 2026-11-06T06:14 節寫明 W4 雜訊 B1 也進 Rust（commit 1304345；G1、14 個 xcheck、雜訊格 Python 對 Rust 逐格都是 0 個不同）。`icpe-data` merge 並做 G0 之後，B1 雜訊的 sweep 也用 Rust。B2、慢層容量上限、STAIR 仍由該 job 自動改用 Python。

**為什麼**（CPU 預算，不是看了結果）
- 使用者定的 CPU 總用量約 44。兩個 Python 佇列（40＋4 worker）佔滿時，Rust 沒有 CPU 可用。
- 提問時（03:25）`tl_c4` 做到 454／6,174；Python 還要約 8–11 h〔推算〕，Rust 約 30 min〔推算，依 STATUS_SIM 03:15 量到的 TraceLab 每 job 時間〕。
- 使用者回答時（07:4x）`tl_c4` 3,835／6,174、`fi6w` 34,993／37,044。停掉仍比等 Python 跑完快〔推算〕。
- 兩個引擎的結果逐位元相同：G1、14 個 xcheck、TraceLab c1 整個 block 7,344 列與 6,174 個 npz（STATUS_SIM 04:32），以及 acvlab 的 G0。所以這個改動只影響時間，不影響數字。

**寫本節時 `icpe-data` 已經看過的正式結果**（偏離 6 列過的之外）：
- `tl_main/tl_c2`：只看了列數（7,344）和 manifest。**沒有算上限**。
- `tl_main/tl_c4` 與 `old_main/fi6w` 的 Python 半成品：只看了 sweep 進度列。
- STATUS_SIM 03:15、04:32、06:14 三節：都是 Python 對 Rust 的逐格比對、時間和記憶體，沒有上限。

### 偏離 7（2026-11-06T00:53:20Z（acvlab 的 `date -u`）；**判定方式的釐清**：PREREG §1 推翻條件與 §4 標記沒寫到的幾點；使用者在 `icpe-main` 決定（AskUserQuestion，11/6 08:4x），並選「登記，由 icpe-main（我）寫」（11/6 09:0x）；不改任何格子、規則、門檻、主指標；當下 `icpe-main` 已經看過的正式結果列在本節最後）

使用者的決定（選項原文）：
1. **「原始／合成各下一個判定字」**：推翻條件「只要有一份真實的工具類紀錄，對最強的不預測做法上限 ≥15%」，原始到達與合成到達的紀錄**各自判定、不合併**（PREREG §4「合成到達的結果和原始到達的分開報」的延伸）。
   - 原始到達：TraceLab c1、Mooncake、FreeInference、Bailian。
   - 合成到達：CacheWise、SemiAnalysis（固定並行 N）、TraceLab c2／c4（開始時間壓縮 2×／4×）。
2. **「R1、R2 各自下判定」**：Mooncake 的兩種重建規則各自判定。`next_sim_analyze.py` 輸出的 `rule=min` 列只報告。這和先前論文、`FIG1_DATA.md` 用 `rule=min` 的慣例不同。
3. **「照算、另外標格數」**：寫入頻寬飽和（`write_saturated`）的格子不排除；另報「≥15% 的格子裡有幾格飽和」。

`icpe-main` 照既有文件寫下的解讀（不是新決定，寫明以免歧義；使用者可以改）：
- 算判定的元件＝O-ORD、O-DROP、O-NOW、O-PF、O-GPU；O-ALL(report) 與截尾版只報告（§4、偏離 6）。
- 工具類＝NEXT_PLAN W2／§5.2 的分法：TraceLab、CacheWise、SemiAnalysis、FreeInference、Mooncake toolagent。Mooncake conversation（聊天）與 Bailian（NEXT_PLAN §5.2「是聊天／API 服務紀錄，不是工具 agent」）只報告。
- 「上限 ≥15%」照字面：一個變體（紀錄變體 × rule）只要任一算判定的元件、任一格（CPU × 慢層 × GPU 閒／忙）≥15% 就算。
- 推翻條件的後半「沒有簡單線索能拿走大部分」：簡單線索（W3 的線索 (a)）沒登記也沒跑 → NOT_MEASURED。所以 ≥15% 成立時，推翻條件寫「未定」；≥15% 不成立時，推翻條件不成立。
- 實作：`code/icpe/main_summary.py`（只彙總、套門檻，不重算上限）。

**為什麼要釐清**：要下判定就必須定這幾點，PREREG 沒寫。

**寫本節時 `icpe-main` 已經看過的正式結果**（照實列出）：
- **使用者做決定 1–3 的時候**（08:4x）：只看過 manifest 的 `stdout_tail` 欄（只印了前 600 字元）：
  - `analysis/sa_main_ttl3/manifest.json`（run `20261106-074822-icpe-trx-analyze-sa_main-ttl3`）：SemiAnalysis N16 s1 的 O-ALL(report)、O-DROP、O-GPU、O-NOW、O-ORD 五列摘要。O-GPU、O-ORD、O-ALL(report) 有 ≥15% 的格子。
  - `analysis/cw_main_ttl3`、`cw_main_final` 的 manifest：CacheWise N16 s1、s2 的部分元件列，最大都 <1.2%。
  - 提問時我已向使用者寫明「已經看到 SemiAnalysis N16 s1（合成到達）有 ≥15% 的格子」。
- **使用者選「登記」的時候**（09:0x）：版本 A（3 個 TTL，run `20261106-084846-icpe-main-verA-ttl3`）的全部判定表、by_trace、by_cpu 摘要，已回報使用者。決定 1–3 在看過版本 A 之後**沒有改**。

### 偏離 7 更正（2026-11-06T00:57:25Z（acvlab 的 `date -u`）；只更正時間標記，不改偏離 7 的內容；`icpe-main` 寫）

- 偏離 7「寫本節時已經看過的正式結果」第二點的「使用者選『登記』的時候（**09:0x**）」是筆誤，實際是 **08:51–08:53**（acvlab 本地時間）：版本 A 在 08:50:41 跑完並回報使用者之後、偏離 7 的 `date -u` 2026-11-06T00:53:20Z 之前。先後順序不變（看過版本 A 之後才決定登記；決定 1–3 時只看過 manifest 的摘要）。icpe-data 指出；核對方式見 STATUS_MAIN 08:58 節。

### 偏離 8（2026-11-06T04:41:49Z（acvlab 的 `date -u`）；**P4（W4 雜訊）選格的解讀**；使用者在 `icpe-main` 決定，由 `icpe-main` 寫；號碼由 `h200` 確認可用）

- 原文：§3 B「範圍：④ 對 ② 上限 ≥15% 的每一格、每個元件（NEXT_PLAN W4）」；偏離 6 的 P4「範圍照 §3 B」，`icpe-data` 的解讀「P4 的 ≥15% 用 P1 之後的上限選格」。沒寫的是：用哪一版的上限表、只報告的 O-ALL(report) 算不算、Mooncake 的 rule＝min 怎麼處理。
- **1. 選格用的上限**＝版本 B（P1 正式值；偏離 6 `icpe-data` 的解讀「用 P1 之後的上限選格」，版本 B 的算法是使用者在 `icpe-main` 同意的計畫）：每類紀錄 main＋`_p1` 的 `_ttl`，④ 和 ② 都有 7 個 TTL，沒有 `--rival-only-gpu`，不含 `cw_main_ttl7`；主指標＝回來請求的平均 TTFT，不截尾（run `20261106-110854-icpe-main-verB-p1`，`results/icpe_next/main/verB/*_p1_headroom_components.csv.xz`）。偏離 5 的算法、`--dead-h`、p90 都不拿來選格。
- **2. 元件**＝O-ORD、O-DROP、O-NOW、O-PF、O-GPU，**加上 O-ALL(report)**（使用者選「加上 O-ALL(report)」）。O-ALL(report) 加雜訊的結果仍然只報告、不進判定（§4、偏離 6）。
- **3. Mooncake 只列 R1、R2**（`icpe-main` 的說明，不改要跑的格子）：rule＝min 是兩者取小，min ≥15% 一定代表 R1、R2 都 ≥15%，所以不另列、也不少跑（程式自檢：322 筆 min ≥15%，R1、R2 都在清單裡）。
- **4. 「≥15%」**＝headroom ≥ 0.15（和 §1 相同）。寫入頻寬飽和的格子照列，另標 `write_saturated`（偏離 7 第 3 點）。
- **結果（選格清單）**：2,582 個（格, 元件）、975 個不同的格；依元件 O-ALL(report) 888、O-GPU 781、O-ORD 434、O-PF 282、O-DROP 197、O-NOW 0；依紀錄 Mooncake 891／216 格、SemiAnalysis 849／389、CacheWise 369／179、Bailian 272／98、TraceLab 201／93、FreeInference 0。清單：`results/icpe_next/main/p4/{p4_cells.csv.xz,p4_cells_distinct.csv,p4_summary.csv}`（run `20261106-124122-icpe-main-p4cells`，`code/icpe/main_p4cells.py`）。只挑 5 個判定元件時是 1,694 個（格, 元件）、868 個格。
- **不改**：門檻（15%／5%）、主指標、規則集合；雜訊的模型、抽法、λ、seed 數、λ＝0 自檢、和 §3 C 佇列 seed 的交叉，全部照 §3 B／偏離 6。

**寫本節時 `icpe-main` 已經看過的正式結果**（照實列出）：
- 版本 A（3 個 TTL，run `20261106-084846-icpe-main-verA-ttl3`）的全部表（偏離 7 已列）。
- **使用者做決定（第 2 點、要登記）的時候**：版本 B（run `20261106-110854-icpe-main-verB-p1`）的全部輸出都已經回報使用者，包括正式判定、偏離 5 的算法、`--dead-h 1800／3600`、p90、論文兩個 Mooncake 舊測試的新舊並列，以及各元件 ≥15% 的格數。提問時已寫明兩種選法各有幾格（1,694 對 2,582）。決定的時間在版本 B 跑完（acvlab 本地 11:15:21）之後、commit baa508d（12:40:25）之前。
- W4 雜訊的結果：還沒有，一個都沒看過（還沒跑）。

### 偏離 9（2026-11-07T00:36:50Z（RUX3DPRO 的 `date -u`）；**W3 線索 (a) 的規則**（推翻條件的後半）；使用者決定登記、由 `icpe-rux` 寫；號碼由 `h200` 給；只登記 W3，APC、W5、排隊不在本節）

**經過**
- 使用者的決定由 `icpe-main` 轉達（11/7 約 08:45）：W3 的規則要在 TRX40 跑任何 W3 sweep 之前登記，由 `icpe-rux` 寫。
- PREREG、NEXT_PLAN 沒有依據的三點，使用者在 `icpe-rux` 用 AskUserQuestion 定了（11/7 約 08:50）：第 3 點的比例、第 4 點的門檻與彙總。使用者也同意由 `icpe-rux` 直接寫進本檔。
- 為什麼要登記：§1 的推翻條件（「只要有一份真實的工具類紀錄，對最強的不預測做法上限 ≥15%，又沒有簡單線索能拿走大部分」）的後半，偏離 7 寫「W3 沒登記也沒跑 → NOT_MEASURED → 未定」。TraceLab c1 是原始到達、工具類、上限 ≥15% 的變體。先看結果再定規則，結果就不能當正式證據，所以先登記。

**1. 線索 (a) 的定義**（依據 NEXT_PLAN W3 第 42 行「工具一開始執行，就把這個對話的 KV 留在／搬回 CPU（不需預測，相當於 Continuum 的 TTL 的極端版）」、§4 表 ③「工具一開始執行就留在 GPU／搬回 CPU」）
- **每個請求的旗標** cue[r] ∈ {0, 1}＝請求 r 這一輪的輸出有工具呼叫。回合結束時系統就看得到工具要開始執行，不用預測。
  - TraceLab（release v0.0.2，`syfi_coding_trace.jsonl.gz` sha256 `11ce51ec0a25e3d1d95b025bca2f7d1647e47571eb7cc968acd5fc64d4b4fb65`）：cue[r]＝該輪原始紀錄的 `tools` 非空。
    - 請求順序照 `code/icpe/next_b_convert.py` 的 `conv_tracelab`＋`finalize`。
    - 程式：`code/icpe/rux_x1fast/v8_features/make_cue.py tracelab`。
    - 跑之前必須用 `make_cue.py check` 對 nb 檔核對 t、sess 的 sha256，要印 `CUE_CHECK PASS`。
    - tracelab_c1：665,453 個請求、cue＝1 有 574,485 個。TRX40 與 RUX3DPRO 各自產生，sha 相同。
  - FreeInference（HF `harvardMadsys/freeinference_agentic_trace` @ `16d3c04d`）：cue[r]＝同一 session 的下一個請求 `step_trigger`＝tool。
    - FreeInference 只記觸發來源，只有 tool／user 兩種。session 的最後一個請求＝0。
    - 程式：`make_cue.py fi`，順序直接取自 `x1_fi.prepare_fi`。
  - 沒有欄位的紀錄（Mooncake toolagent、SemiAnalysis）：不跑，推翻條件的後半維持「未定」（偏離 7）。
  - CacheWise：這次不跑（使用者 11/7「CacheWise 都先不跑」）。之後要跑，得另寫 cue 程式（`trigger`），並補登記 cue 的定義；其他照本節。
- **③ 的兩個 GPU 規則**（引擎 `code/icpe/rux_x1fast/v8_features/`，v8 `src/main.rs` sha256 `48e4efce…`；sweep 用 `sweep_v8.py`＋`RUX_CUE_FILE`）：
  - **CUEP**：cue＝1 的回合結束後，這個對話的 KV 釘在 GPU，直到它的下一個請求；cue＝0 的回合照 GPU LRU。GPU 滿而且全部釘住時，放掉對話開始時間最晚的那個（同 §2 第 3 項的 TTL 規則）。
  - **CUEO**：cue＝1 的回合結束時立刻逐出 GPU（wb：寫進 CPU，整個對話寫一次）；cue＝0 照 GPU LRU。
- **驗證**（`smoke_v8.py`，3 份紀錄）：
  - cue 全 0 時，CUEP、CUEO＝GPU LRU，逐位元相同；
  - cue 全 1 時，CUEP＝TTLB:∞，逐位元相同；
  - 不開新功能時，引擎和 Python 參考逐位元相同。
  - 另外有一個只讀的獨立審查，見 `docs/icpe_brief_20261030/next/rux/V8_FEATURES.md`。

**2. 跑哪些紀錄、哪些格**
- **紀錄**：
  - TraceLab c1（原始到達）、c2、c4（合成到達）；
  - FreeInference 6 週：2026-05-17_2026-05-23、05-24_05-30、05-31_06-06、06-07_06-13、06-14_06-20、06-21_06-27（原始到達）。
  - 原始、合成各自判定（偏離 7 第 1 點）。
- **格**＝主結果的格（偏離 4 第 3 項）：
  - CPU 9 個容量（8、16、32、64、128、256、512、1024、111.0625 GiB）× 慢層 ssd／b32／nfs × GPU 閒／忙 × Cake；
  - GPU 111.0625 GiB、wb、預取到 CPU。
- **CPU 規則**：`x1_sim.NON_ORACLE` 的 10 個，同 ② 的組成（偏離 2 第 5 點「每個 CPU 對手都配」）。
- 一個變體約 450 個 job：CPU 10 × GPU {CUEP, CUEO} × CPU 9 個容量，時間相關的規則依時間設定拆。

**3. 怎麼算**
- 主指標：回來請求的平均 TTFT（`mean_ret`），不截尾（§1、偏離 8 第 1 點）。
- T③（每格）＝min over {CPU ∈ NON_ORACLE} × {GPU ∈ {CUEP, CUEO}} 的 mean_ret：事後挑最快，同 ②（§2 第 3 項）。
  - 一格要 20 個組合都跑了才算；少了，那格＝NOT_MEASURED。
- T②、T④c＝推翻條件前半用的同一份上限表，現在是版本 B（`results/icpe_next/main/verB/{tl,fi}_p1_headroom_components.csv.xz`，run `20261106-110854-icpe-main-verB-p1`，偏離 8 第 1 點）。
  - 跑 ③ 的機器要先證明環境和版本 B 相同：同一個變體重算版本 B 的表，逐值相同（`explore_apc.py` 的重現檢查，0 個不同）。不同就停，不判。
- **比例**（使用者選「相對 ②（PR:21）」）：f_c＝(T② − T③) ÷ (T② − T④c)，每個 (格, 元件 c) 一個。
  - c＝O-ORD、O-DROP、O-NOW、O-PF、O-GPU（偏離 7：算判定的元件）。O-ALL(report) 也算，只報告。
  - NEXT_PLAN 第 43 行的 C2 算法 (T① − T③) ÷ (T① − T④O-PF) 只報告，不進判定。
- 程式：`code/icpe/rux_x1fast/v8_features/analyze_w3.py --sweep <③ 的 x1_sweep.csv> --headroom <版本 B 的表>`，輸出 `w3_rows.csv`、`w3_verdict.csv`。

**4. 判定**（使用者選：門檻「f > 0.5」、彙總「逐格，任一格成立就算」）
- (格, 元件) 的上限 ≥ 0.15（§1、偏離 8 第 4 點），而且 ③ 齊全，才判。
- 「簡單線索拿走大部分」＝該 (格, 元件) 的 f_c > 0.5。
- 一個變體（紀錄變體 × rule；FreeInference 每週一個變體）：
  - 只要有任一個「上限 ≥15%、③ 齊全」的 (格, 元件) f_c ≤ 0.5 → 推翻條件的後半**成立**（沒有簡單線索能拿走大部分）。
  - 所有「上限 ≥15%、③ 齊全」的 (格, 元件) 都 f_c > 0.5，而且沒有 ③ 不齊的 ≥15% 格 → 後半**不成立**（簡單線索拿走大部分）。
  - 沒有 ≥15% 的 (格, 元件) → 前半就不成立（偏離 7）。
  - 有 ≥15% 的格但 ③ 都不齊、或只剩不齊的格 → **未定**。
- 推翻條件：前半與後半都成立的變體，只要有一個（「只要有一份真實的工具類紀錄」）→ 推翻條件成立。原始、合成各自下判定字（偏離 7 第 1 點）。
- 寫入頻寬飽和（`write_saturated`）的格照算，另報「上限 ≥15%、f ≤ 0.5 的格裡有幾格飽和」（偏離 7 第 3 點）。
- 多個 CPU 容量、慢層、GPU 閒／忙不合併，照上面逐格判。
- Mooncake 的 R1／R2：沒有欄位，不適用，見第 1 點。
- p90 等其他指標只報告。

**5. 方向（照實寫）**
- ③ 是兩個規則 × 10 個 CPU 規則事後挑最快，對線索有利 → 線索比較容易「拿走大部分」→ 比較不容易推翻，偏向保住「預測沒有多大優勢」的結論。
- 反過來，CUEO 在 wb 下每次都把整個對話重寫進 CPU，讓 ③ 略弱。

**寫本節時 `icpe-rux` 已經看過的結果**（照實列出）：
- **版本 B**：
  - 讀過 Mooncake、SemiAnalysis、Bailian 的 `*_p1_headroom_components.csv.xz` 全部列，用來重現與探索。
  - 透過研究報告看過 STATUS_MAIN 的摘要：原始到達的工具類變體 ≥15% 的是 Mooncake toolagent R1 57.8%、R2 62.4%、TraceLab c1 38.9%。
  - **沒有打開** TraceLab、FreeInference 的版本 B 表。
- **W3**：
  - **沒有看過任何 TraceLab、FreeInference 的 ③ 或 f**（還沒跑）。
  - 看過 TraceLab 原始紀錄的線索統計（runs `20261107-033551-icpe-rux-w3-cue-stats-tracelab`、`20261107-045446-icpe-rux-w3-make-cue-tracelab`）：
    - 「輸出有工具呼叫」和「下一輪由工具結果觸發」一致 96.9%；
    - cue＝1 的比例 86.3%。
  - 看過 SemiAnalysis N16 s1 用代用 cue 的管線測試（runs `20261107-045622-icpe-rux-w3-pipeline-sa-proxy`、`20261107-083542-icpe-rux-w3-analyze-recheck-sa-proxy`）：
    - 代用 cue＝下一個請求不是使用者觸發，SemiAnalysis 幾乎全是 1，不是真的線索。
    - 3 個 CPU 容量下，③ 在每一格都比 ② 慢，f 為負，「後半成立」。
    - 只當管線測試，不是證據；SemiAnalysis 沒有欄位，不在第 2 點的範圍裡。
- **其他探索**（不是正式結果，`docs/icpe_brief_20261030/next/rux/V8_FEATURES.md` §4、`results/icpe_next/v8_explore/`）：
  - APC／BELP、W5：Mooncake toolagent、SemiAnalysis N16 s1、Bailian coder；
  - 排隊：SemiAnalysis N16／N32／N64／N72 s1。
  - 使用者在 08:50 做決定前，已看過這些的摘要。
- TRX40 的 TraceLab c1 APC／BELP 探索在本節寫好時還在跑，結果沒有看過。

### 偏離 9 更正（2026-11-07T00:40:03Z（RUX3DPRO 的 `date -u`）；更正偏離 9 的兩個時間，並照實補一句「方向」；規則、門檻、判定都不改；`icpe-rux` 寫，使用者同意補方向（AskUserQuestion，11/7 約 08:40））

- **時間 1**：偏離 9「經過」寫「icpe-main 轉達（11/7 約 08:45）」，是錯的，我照抄了 `icpe-main` 給的時間。
  - 實際：使用者在 `icpe-main` 決定「W3 先登記、由 icpe-rux 寫」是 acvlab 本地時間約 08:28–08:30。
  - `icpe-main` 的訊息在 08:31:28（+08:00）之前送出。
  - `icpe-main` 指出並提供時間。
- **時間 2**：偏離 9「經過」寫「使用者在 `icpe-rux` 用 AskUserQuestion 定了（11/7 約 08:50）」，是錯的，我估錯了。照 `icpe-rux` 這邊的紀錄：
  - 定三點、同意直接寫入：在 08:29:19（commit `45583298`）之後、08:35:26 之前。08:35:26 是 run `20261107-083526-icpe-rux-w3-analyze-recheck-sa-proxy` 開始的時間，這個 run 用的已經是照答案改好的 `analyze_w3.py`。
  - 同意 push：在 08:37:10（commit `8f04a91c`）之後、push 之前。
- **先後順序不變**：
  1. 使用者決定（08:28–08:35）；
  2. 登記（偏離 9 的 `date -u` 00:36:50Z＝08:36:50，commit 08:37:10）；
  3. push；
  4. TRX40 收到 `8f04a91c` 之後才跑 W3。到本更正為止，TRX40 還沒開始任何 W3 sweep，它的 TraceLab c1 還在跑 APC／BELP 的重現檢查。
- **方向，補充偏離 9 第 5 點**：
  - 第 4 點的彙總「逐格，任一個上限 ≥15%、③ 齊全的 (格, 元件) f_c ≤ 0.5 就算後半成立」會讓後半容易成立，推翻條件也就比較容易成立，偏向「預測仍有空間」。
  - 這和第 5 點寫的「③ 事後挑最快，對線索有利」方向相反，影響可能更大。
  - 規則照使用者的選擇，不改。`icpe-main` 指出。
- **寫本更正時看過的結果**：同偏離 9。另外仍然沒有看過任何 TraceLab、FreeInference 的 W3 結果。

### 偏離 9 補記（2026-11-07T00:52:12Z（RUX3DPRO 的 `date -u`）；**偏離 9 第 3 點「先證明環境和版本 B 相同」要重算哪些格**；使用者決定（約 08:47，在 `icpe-main`；`icpe-next-trx` 轉達），使用者在 `icpe-rux` 指示「你做掉」（11/7 約 08:52），由 `icpe-rux` 寫；規則、門檻、判定都不改）

- **原文**：偏離 9 第 3 點「跑 ③ 的機器要先證明環境和版本 B 相同：同一個變體重算版本 B 的表，逐值相同（`explore_apc.py` 的重現檢查，0 個不同）。不同就停，不判。」沒寫要重算哪些格。
- **決定**：重現檢查要涵蓋 W3 會判的**全部格**，也就是偏離 9 第 2 點的格。
  - 範圍：9 個 CPU 容量（8、16、32、64、128、256、512、1024、111.0625 GiB）× 慢層 ssd／b32／nfs × GPU 閒／忙 × Cake。
  - 規則集合同版本 B：CPU 17 × GPU 16，不含 APC、BELP。
  - 每個變體各自做。
  - 每一格、每一個元件都要逐值相同：T_oracle、T_rival、上限、最佳對手的 CPU 與 GPU 規則、oracle 的 CPU 與 GPU 規則。
  - 任何一格不同 → 那個變體停，不判（＝NOT_MEASURED）。
- **已經發生的事**（照實寫）：
  - TraceLab c1 在本補記之前，只重現了 3 個容量（8、111.0625、1,024 GiB；108／108、0 個不同），就跑了 W3 並算出判定（icpe-next-trx `9e1064f0`）。
  - 依本補記，這個判定**待確認**：16–512 GiB 這 6 個容量的重現也 0 個不同，才算數；有任何一格不同，c1 就不判。
  - W3 的 sweep 和 `analyze_w3.py` 的結果不必重跑，只是先決條件要補齊。
- **寫本補記時已經看過的結果**：
  - 除了偏離 9 與其更正列的之外，`icpe-rux` 和使用者都看過 TraceLab c1 的 W3 結果（使用者在 08:47 決定之前已經看過）：
    - 上限 ≥15% 的 (格, 元件) 有 22 個（O-GPU 18、O-ORD 4），全部 f_c ≤ 0.5，f_c 中位數 −7.41；
    - `w3_verdict.csv` 的判定字是「後半成立」。
  - TraceLab c1 的 APC／BELP 探索也看過了（icpe-next-trx `0cfa480b`）。
  - 這些結果都**不影響**本補記的內容：本補記只把先決條件定得更嚴，不改規則、門檻、判定。
- **方向**：先決條件更嚴 → 可以判的變體只會變少（較多「未定」），不改任何一個 f。

### 偏離 10（2026-11-07T00:59:09Z（RUX3DPRO 的 `date -u`）；**block 粒度 APC 加進 ②、④ 的 O-GPU 加 BELP、APC 版成為正式主表**（PR:95 承諾的第二批）；使用者在 `icpe-rux` 決定（AskUserQuestion，11/7 08:56:16 之後、08:58:48 之前：前者是發問前寫入的檔、後者是照答案寫的草稿的修改時間），由 `icpe-rux` 寫；號碼由 `h200` 給）

**經過**
- 偏離 2 第 3 點（PR:95）：「APC：第一批＝GPU-LRU（session 粒度下完全相同，不另外跑）；第二批再加 block 粒度」，並登記「block 粒度 APC 做好後加進 ②；在那之前的結果標明 ② 不含」。
- PR:55：APC 的確切定義要登記。PR:126：第二批的格子另外定、跑之前登記。
- 語意：使用者 11/7 授權 `icpe-rux` 決定第 2–5 項的細節（`docs/icpe_brief_20261030/next/rux/SEMANTICS_2to5.md` §1）。
- 本節的三個決定：使用者 11/7 在 `icpe-rux` 選的，選項原文見下面第 3、4、5 點。

**1. APC 的定義**（SEMANTICS 2-1–2-4、2-7、2-8）
- **逐出的順序**：
  - GPU 滿了才逐出。
  - 先選回合結束時刻最早的 session（session LRU），從它的尾端一塊一塊逐出（512-token 塊），直到 GPU 用量 ≤ 預算（PR:28 的 1,777 塊）。
  - 共用的塊要等 GPU 上所有持有者都放掉才真的空出來。
  - 依據：
    - vLLM APC 的 free queue：釋放的塊照相反順序排，從頭逐出最久沒用的；同一個請求尾端先出。
    - 一次服務一個請求時，這和「全域 block LRU＋尾端先出」選到同樣的塊。
- **wb**（主結果）：
  - session 回合結束後第一次被逐出塊時，整個 session 寫進 CPU（BF16，PR:54 的資料流）；session 進 CPU 層，CPU、慢層的規則照舊。
  - GPU 上留下的前段變成乾淨的副本，之後從尾端逐出、不用再寫。
  - 比逐塊寫回早寫一點 → ② 略弱 → 上限略偏高。
- **wt**：CPU 照舊（回合結束就寫），GPU 副本照同樣的順序逐出、不寫。
- **TTL** 維持 session 粒度（§2 第 3 項）。
- 預取進 GPU × APC 不支援，P3 的「預取進 GPU」格子不放 APC。
- **引擎**：`code/icpe/rux_x1fast/v8_features/`（v8，`src/main.rs` sha256 `48e4efce…`）；sweep 用 `sweep_v8.py --gpu-policies … APC BELP`。
- **驗證**（沒有 Python 參考；使用者 11/7 決定只寫 Rust）：
  - 一次逐出整個 session 時＝v7 的 GPU LRU／BEL，逐位元相同；
  - GPU 夠大時＝v7；
  - 每 200 個請求從頭重算 GPU 引用數、CPU 引用數與用量的稽核；
  - 新功能關閉時和 Python 參考 0 個不同；
  - 一次只讀的獨立審查（`V8_FEATURES.md` §3b）。

**2. BELP**
- 同樣「從尾端一塊一塊逐出」，但選 session 照 Belady：下次回來最晚的先逐出，鍵同 GPU BEL。是完美預測（oracle）。
- 只進 ④ 的 O-GPU，不進 ②，也不進其他元件的 ④。

**3. ②、④ 怎麼組成**（使用者選「加 BELP」）
- APC 是不預測的 GPU 規則。
  - ② 的每個元件、以及 O-ORD、O-DROP、O-NOW、O-PF 的 ④，GPU 不預測規則集合都多一個 APC（PR:177–182「兩邊規則集合相同」）。
  - 也就是 {LRU, TTLA×7, TTLB×7, APC}。
- O-GPU：
  - ④＝NON_ORACLE × {BEL, BELP} 取最快；
  - ②＝NON_ORACLE × {LRU, TTLA×7, TTLB×7, APC}。
  - 依據：② 有 block 粒度的不預測規則，④ 沒有 block 粒度的完美規則，上限會被低估。探索：TraceLab c1 上 O-GPU 上限最大值，④ 只有 BEL 時 39%→5%，加 BELP 是 39.2%。
- O-ALL(report)：照舊（只報告，④ 固定 (O-ALL, BEL, BEL)）。
- 程式：`code/icpe/rux_x1fast/v8_features/analyze_v8.py`。
  - 就是 `next_sim_analyze.py`，只多一點：BELP 不進任何不預測集合，O-GPU 的 ④ 含 BELP。
  - 檢查：O-GPU 以外的元件＝拿掉 BELP 列後原本的分析；O-GPU＝獨立算法；都 0 個不同。

**4. 重算哪些格**（使用者選「全部登記變體的主結果格」）
- **紀錄變體**：偏離 4 第 3 項的全部 40 個，即 CacheWise 12、TraceLab 3、SemiAnalysis 12、Bailian 3、Mooncake 4、FreeInference 6。
  - CacheWise 在使用者恢復之後才跑（使用者 11/7「CacheWise 都先不跑」）。
- **格**：主結果的格，即 CPU 9 個容量 × 慢層 ssd／b32／nfs × GPU 閒／忙 × Cake，GPU 111.0625 GiB、wb、預取到 CPU。
- **規則**：
  - CPU 17 個（偏離 4 第 1 項）；
  - GPU 18 個＝版本 B 的 16 個（LRU、BEL、TTLA／TTLB × 7）＋ APC ＋ BELP。
  - 每個變體重跑全部 18 個 GPU 規則，不和舊的 sweep 合併。
- **重現檢查**：同一個變體只用版本 B 的 16 個 GPU 規則重算上限表，要和版本 B 的表（`results/icpe_next/main/verB/*_p1_headroom_components.csv.xz`）全部格、全部元件、7 欄逐值相同（同偏離 9 補記）。不同就停，那個變體不重算。
- **分工**：
  - TraceLab、FreeInference：TRX40（icpe-next-trx）；
  - SemiAnalysis、Mooncake、Bailian：RUX3DPRO（icpe-rux）；
  - CacheWise：之後在 acvlab。
- **結果**：`results/icpe_next/apc/<變體>/`，內容：
  - `x1_sweep.csv.xz`；
  - analyze_v8 的 `*_headroom_components.csv`；
  - `MANIFEST.json`，記 run、引擎 code_sha、檔案 sha256。
- 彙總與判定字由 `icpe-main` 照版本 B 的流程做（`main_summary.py`）。

**5. 新表的地位**（使用者選「成為正式主表」）
- APC 版的上限表（以下稱「版本 C」）取代版本 B，成為 §1 推翻條件前半的表。
  - 判定方式照偏離 7：只判 5 個元件；原始、合成各自判；R1／R2 各自判；寫入飽和照算另標。
- 偏離 9 的 W3 用「前半同一份表」，所以 T②、T④ 改用版本 C 重算 f。
- P4 的選格（偏離 8 第 1 點）改用版本 C：
  - 版本 C 下 ≥15%、版本 B 沒選到的格，補跑 P4；
  - 版本 B 選到、版本 C 下 <15% 的格，已經有的結果保留並標明。
- 版本 B 保留，標「② 不含 block 粒度 APC」，和版本 C 並列。
- 門檻（15%／5%）、主指標（回來請求的平均 TTFT，不截尾）都不改。

**6. 方向**
- APC 進 ② → ② 變強 → 上限下降。
- BELP 進 ④ 的 O-GPU → ④ 變強 → 上限上升。
- 兩者一起是為了公平（第 3 點）。
- wb 的「整個 session 一次寫進 CPU」讓 ② 略弱（上限略偏高）。

**寫本節時 `icpe-rux` 與使用者已經看過的結果**（照實列出；都不是正式結果）：
- **APC／BELP 探索**（CPU 8、111.0625、1,024 GiB；`docs/icpe_brief_20261030/next/rux/V8_FEATURES.md` §4a、`results/icpe_next/v8_explore/apc/`）：
  - Mooncake toolagent R1／R2、Bailian coder：上限最多變 0.03 個百分點。
  - SemiAnalysis N16 s1：② 最多快 28%；O-GPU（④ 加 BELP）上限中位數 8.5%→4.6%，≥15% 的格 6→4。
  - TraceLab c1（TRX40，icpe-next-trx `0cfa480b`）：
    - O-GPU 上限最大值：④ 只有 BEL 時 39.0%→5.1%，加 BELP 是 39.2%；
    - O-ORD 23.3%→21.9%。
- 使用者做三個決定時，已經看過上面這些摘要。
- 版本 A、B 的表與判定（偏離 7、8 已列）；W3 的結果（偏離 9 補記已列）；W5、排隊的探索摘要。

### 偏離 11（2026-11-07T01:05:49Z（RUX3DPRO 的 `date -u`）；**W5 與排隊兩個敏感度**（PR:57「排隊模型、W3、W5…要跑之前另補登記」）；使用者在 `icpe-rux` 決定（AskUserQuestion，11/7 09:01:38 之後、09:03:39 之前：前者是 RUX3DPRO 開始正式 APC 重算、確認它在跑之後才發問，後者是照答案之後的第一個 commit `7a89a9b5`），由 `icpe-rux` 寫；號碼由 `h200` 給；主結果、門檻、主指標都不改）

**經過**
- 兩項的語意由使用者授權 `icpe-rux` 決定（11/7），見 `docs/icpe_brief_20261030/next/rux/SEMANTICS_2to5.md`：W5 是 §4，排隊是 §2b。
- 排隊的 Q3-a（④ 知不知道排隊造成的新順序），使用者答「我不知道你決定」，由 `icpe-rux` 定為「④ 只知道紀錄的順序」（§2b）。
- 本節四個決定的選項原文見下面。
- 引擎：`code/icpe/rux_x1fast/v8_features/`（v8，`src/main.rs` sha256 `48e4efce…`，和偏離 10 同一版）。
- 規則集合：同偏離 10 的版本 C，即 CPU 17 × GPU 18（版本 B 的 16 個＋APC＋BELP），分析用 `analyze_v8.py`。

**A. W5（CPU 層 INT4＋一次拆完）**
1. **定義**（SEMANTICS 5-1–5-4）：
   - CPU 層的每個塊都是 INT4：占 i4 位元組，容量照這個算。
   - 還原 CPU 上的塊時，每塊讀 i4 ÷ r_cpu，再加「一次拆完的殘差」d1＝(T_meas − 64·i4 ÷ r_cpu) ÷ 64。
     - T_meas：P0-1b S4（fa3）在 32K、CPU、INT4、全部載入時的還原中位數，GPU 閒 0.02914 s、忙 0.03327 s。
     - 出處：`results/icpe_h200/p01/grid/llama_s4_fa3/summary.csv`，run `20261022-232344-icpe-p01b-llama-s4-fa3`。
     - 32K 的總時間和實測相同；長度依賴 NOT_MEASURED。
   - 預取進 CPU 的 INT4 塊也用 d1。CPU → 慢層是直接複製，不再量化。
   - 量化時間不計，和現有的慢層 INT4 寫入一致。
   - 還原照舊用 Cake。
   - sweep 用 `sweep_v8.py` 並設 `RUX_W5=1`，CSV 多一欄 `cpu_fmt`。
   - 驗證：參數設成等於沒開時和 v7 逐位元相同（17 × 16 個規則的 sweep 254,592 格 0 個不同）；位元組不變量；獨立審查（F2 已改）。
2. **格**（使用者選「7 個代表變體 × 111／128／256」）：
   - 變體：PR:118 的 7 個代表變體，即 CacheWise N16 s1、TraceLab c1、SemiAnalysis N16 s1、Bailian coder、Mooncake conversation R1、Mooncake toolagent R1、FreeInference 第 1 週（`2026-05-17_2026-05-23`）。CacheWise 在使用者恢復之後才跑。
   - CPU 容量：111.0625、128、256 GiB，是 NP:52 和 PR:118 的聯集。
   - 時間設定：主時間設定（ssd／b32／nfs × 閒／忙 × Cake），GPU 111.0625 GiB、wb、預取到 CPU。
   - 只跑 W5 那一半；BF16 那一半直接用偏離 10 正式重算的同一個 sweep（同一版引擎）。
3. **怎麼算**（使用者選「敏感度，每組事後挑較快格式」）：
   - 每個 (格, CPU 規則, GPU 規則) 的 mean_ret 取 BF16 和 W5 較小的那個，② 和 ④ 都一樣。這和 ② 的「事後挑最快」是同一種。
   - 之後用 `analyze_v8.py` 算 W5 的上限表，和版本 C 同一格比較。
   - **主結果不變**，W5 只報告：每個變體 × 元件的上限、≥15% 的格數、和版本 C 的差。
4. **NP:53 的「若上升，要查原因」**：
   - 上限比版本 C 升高超過 5 個百分點的 (格, 元件)，都要報 ④ 和 ② 的最佳組合在回來請求上「CPU 載入／慢層載入／重算」次數的變化。
   - 程式：`v8_features/w5_mechanism.py`（門檻 5）。
5. **分工**：
   - TraceLab c1、FreeInference 第 1 週：TRX40；
   - SemiAnalysis、Bailian、Mooncake：RUX3DPRO；
   - 結果放 `results/icpe_next/w5/<變體>/`。

**B. 排隊（一階近似 `qpost`）**
1. **定義**（SEMANTICS §2b）：
   - 快取的決定，包括所有 oracle，都在沒有排隊、照紀錄順序的模擬裡做（④ 只知道紀錄的順序）。
   - 之後每個時間設定各做一次工作守恆的 closed-loop FCFS：
     - 每個 session 的第一個請求在紀錄的時刻到；之後的請求到達＝紀錄的時刻＋這個 session 上一個請求「開始服務比紀錄晚多少」。
     - 單一 GPU，照到達先後服務，同時到照請求序號。
     - 服務時間＝這個時間設定的 TTFT（還原＋prefill；解碼不算，NP:276）。
     - 回報的 TTFT＝等待＋服務。
   - sweep 用 `sweep_v8.py` 並設 `RUX_QPOST=1`，CSV 多 `queue`、`qwait_mean`、`qwait_p90`、`qwait_frac` 四欄。
   - 驗證：
     - 到達拉開時＝沒排隊；
     - Python 精確重算逐位元相同；
     - 快取狀態不變（每份紀錄 48 項）；
     - sweep 端到端只有 TTFT 衍生的欄位不同。
2. **格**（使用者選「全部格，重負載只報告」）：
   - 固定並行的紀錄：SemiAnalysis 12 個變體（RUX3DPRO），CacheWise 12 個（使用者恢復後在 acvlab 跑）。
   - × 主結果的格（CPU 9 個容量 × ssd／b32／nfs × 閒／忙 × Cake）× 版本 C 的規則集合。
3. **負載分級**（每格依 (LRU, LRU) 那一組的平均等待）：輕 < 2 s、中 2–20 s、重 ≥ 20 s。
   - 重負載的格服務順序重排很多，例如 nfs、GPU 忙時服務排名的位移中位數約 2,000（run `20261107-080926-icpe-rux-v8-qpost-reorder-sa`），一階近似不可靠。
   - **重負載的格只報告、標「不可靠」，不拿來下結論。**
4. **地位**（使用者選「敏感度，主結果標『不含排隊』」）：
   - 固定並行紀錄的主判定維持不排隊（版本 C），照 NP:266 標「上限（不含排隊）」。
   - 排隊結果另外報：每個變體 × 元件在輕、中負載格的上限、≥15% 的格數、和版本 C 的差。
   - 也照實報 NP:266「不排隊對預測有利」這個方向在輕、中負載格上是否成立。
5. 結果放 `results/icpe_next/queue/<變體>/`，分析用 `v8_features/explore_queue.py`（負載分級同上）。

**C. 方向（照實寫）**
- W5：② 和 ④ 都可以事後挑格式，對兩邊一樣。CPU 層變大時，oracle 也一起變強；探索裡看到上限可能上升。
- 排隊：④ 不知道排隊造成的新順序 → ④ 略弱 → 上限略偏低，這是 §0 說不可以的方向。輕負載時重排小、偏差小；重負載的格不下結論。

**寫本節時 `icpe-rux` 與使用者已經看過的結果**（照實列出；都不是正式結果；`docs/icpe_brief_20261030/next/rux/V8_FEATURES.md` §4、`results/icpe_next/v8_explore/`）：
- **W5 探索**（CPU 111.0625、128、256）：
  - Mooncake toolagent R1／R2：O-GPU 上限中位數降 6–8 個百分點；O-DROP、O-ORD 上升，nfs 的格最多 +39。上升的格裡，④ 的慢層載入減 79–99.5%，② 減 13–26%。
  - SemiAnalysis N16 s1：各元件下降，O-GPU ≥15% 的格 4→0。
  - Bailian coder：多數下降，O-GPU ≥15% 的格 8→2。
- **排隊探索**（SemiAnalysis N16／N32／N64／N72 s1，CPU 8、111.0625、1,024）：
  - 輕負載上限幾乎不變，中位數 −0.3 個百分點。
  - 負載越重上限越大，O-GPU ≥15% 的格在 N64、N72 變成 18／18。
- 使用者做四個決定時已經看過上面的摘要。
- 其他：版本 A、B；APC 探索（偏離 10 已列）；W3 TraceLab c1 的結果（偏離 9 補記已列，9 個容量的重現 TRX40 已補完、0 個不同）。
