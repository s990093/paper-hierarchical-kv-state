# 第三階段前置探索：實驗矩陣（v0.2，2026-10-19；13:4xZ 補 L0，14:0xZ 依 L1–L3 改）

> **狀態**：v0.2。使用者 10/19 同意方案、先做 CPU 部分。文獻地圖 L0–L3 已回（`lit/`，彙整在 `research_map.md`）；L4（Cascade、CacheSmith、CacheFlow 全文查新）跑中。第二階段十條觀察的核實：`stage2_factcheck_and_toolbox.md`。
> **規則**：本頁所有「預期」都是〔預測〕，不是結果；沒量到的一律 NOT_MEASURED。引用的舊數字都附出處。

---

## 0. 時間與資源（已和使用者確認）

| 項目 | 值 |
|:--|:--|
| 租約結束 | **10/23 17:00 台灣＝10/23 09:00Z**（伺服器時鐘是 UTC） |
| 現在 | 10/19 13:3xZ（台灣 21:3x）→ 剩約 91 小時 |
| GPU 優先序 | **第二階段佇列先跑完**（使用者 10/19 決定）；第三階段只用剩下的 GPU 時間 |
| 第二階段剩下 | main：44ca（跑中）→ 44cb → 44cz → 44d → 44e0–44ez（drop／M1／M2，14 號估 8–11 h）→ 70 → 75；helper：55（被搶先、待續）→ 60 → 88 |
| 〔估計〕第三階段 GPU 窗口 | main 約 **10/20 10–14Z** 清空（不確定性 ±6 h）；helper 的 55／60／88 可能更晚 → 前幾小時可能只有卡 1 |
| 收尾 | **10/23 02:00Z 後不開新的長實驗**；02–09Z 整理、commit、push、備份 |
| 硬體 | 2× MI300X 192 GB（gfx942）、EPYC 9684X（pod 配額 32 核、435 GiB；主機有 2.2 TB 但容器只能用 435 GiB）、本地磁碟（overlay，sda GBT3916 3.5 TB；**看不到 /dev/nvme\***，所以 hipFile 直通不能用）、NFS（/mlsteam，nfs4）。**沒有 QAT／IAA／DSA／AMX 的對應硬體**（L0 §3） |
| 軟體 | ROCm 7.2.2、torch 2.10+rocm7.0、vLLM 0.19.1（tiara venv）與 **vLLM 0.28.0（tiara-v028 venv，有 OffloadingConnector、simple_cpu_offload、kv_offload/tiering、LMCache／Mooncake／MoRI connector 的原始碼）** |
| 資料 | RULER 式 T1–T5（`code/m10_tasks.py`，32K／64K／128K）、SCBench（多輪共享 context）、Mooncake trace（conversation／toolagent；**hash_ids＝512 token**，CLAUDE.md 規則 6） |

---

## 1. 設計主軸：前兩階段告訴我們什麼、第三階段要測什麼

前兩階段的共同結論是**「位置」不是有用的訊號**：

- 第一階段：寫入時依位置放哪一層 → 延後版追得上（`docs/phase1_20261008/13_advisor_update_20261012.md`）。
- 第二階段：依位置決定精度 → FP8 全部用就夠；INT4 時放前／中／後也不比隨機好（K2，`docs/precision_20261013/08_report_mi300x.md` §0）。好的低精度格式在 32K–128K 本身就不太傷答對率（RUNLOG_MI300X 15.9）。
- 第二階段也看到：**單一請求、真實 CPU 層的還原已經很快**（Llama 32K E0 閒 0.095 s，08 §2.1），只有限速層／NFS 才慢；GPU 忙會把會合點往前移。

**核實（X1，`stage2_factcheck_and_toolbox.md`）改掉的兩件事**：
- 使用者摘要的第 2–10 條（202／2720 ms、寫入位置 ≤0.3%、GPU 忙 13–66% 等）都出自**第一階段**（M7 寫入位置），不是第二階段的精度實驗；202／2720 ms 是含 256 token 新一輪的 TTFT。
- 「NFS 0.33 GiB/s、SSD 6.98 GiB/s」是**限速器參數**（實測最慢 ×0.9），不是真的從 NFS 讀；實測讀取：本地 7,943–9,152 MiB/s、NFS 377–447 MiB/s。→ 第三階段凡標「NFS 級」都要寫明是限速模擬；S1、I1 各加一條**真的從 NFS／本地檔案讀**的對照。

所以第三階段要測**另外兩種訊號**，加上一個「省位元組」的對照：

| 訊號 | 問題 | 對應的影片方向 |
|:--|:--|:--|
| **內容（query-aware 重要性）** | 回來的那個問題需要哪些 block？只載入那些、其餘先不載（不是丟掉），能省多少搬運、掉多少品質？ | Cascade（＋Shrink 的 drop） |
| **系統狀態（併發、GPU 爭用、容量壓力）** | 多個 session 同時回來、GPU 在忙、KV 容量不夠時，單請求最佳（Cake）的結論還成立嗎？ | Infinity、Fuse |
| **位元組（壓縮）** | 無損／有損壓縮在什麼鏈路上划算？同樣位元組下，壓縮、丟棄、卸載誰好？ | Shrink |

**agent 情境**：這次只當**工作負載**（多輪、工具呼叫暫停、前綴重疊），不建完整 agent 系統（使用者的「場景改 agent 感覺太急」）。

---

## 2. 實驗總表（依資訊價值÷成本排序；v0.2 依文獻改）

「前人」欄：✗＝沒找到（≠不存在）、△＝部分做過、✓＝做過；出處見 `research_map.md`。

| ID | 方向 | 一句話 | 前人 | 需要 | 〔估〕GPU | 優先 |
|:--|:--|:--|:--|:--|--:|:--|
| **S1** | Shrink | 無損壓縮 KV 的盈虧點；含 Intel KV Shrink 的作法（DEFLATE、只壓高位元組） | △ SplitZip（H200，1.31–1.33×）、KVServe 公式；**AMD／CPU 端解壓沒人量** | 程式已寫好（自測過）；KV 匯出 10 分鐘 GPU | 0.2 | ★★★ |
| **I4** | Infinity（DSA） | 零碎 copy 批次化＋**SDMA／blit × GPU 忙**（L2 X3 併入）：vLLM CPU 層 4.1 GiB/s vs pinned 50；GPU 忙時 H2D 慢 46–50% 的原因 | ✗（ROCm PR #11614 未合併；ConCCL 是訓練通訊） | 1 卡、量時間 | 2 | ★★★ |
| **F2** | Fuse | **重現 vLLM #44294**：命中但正在載入 → 延後；共用前綴的併發突發，卸載反而更慢 | △ 單點 issue（L40S）；MI300X 沒人量 | V0 之後；1 卡、量時間 | 1 | ★★★ |
| **C1a** | Cascade（閘門） | **分數有沒有預測力**：寫入時分數、Quest 式 block 摘要、第 1 輪 QRY，能不能找到第 2 輪答案所在的 block（recall@k、Spearman） | ✗ 「下一輪實際用到哪些 block」的實測（L1 G2／G4） | 1 卡、只 forward | 1.5 | ★★★ |
| **S2** | Shrink＋drop | **等位元組：載入區的有損刪除 vs 降精度**（1/2、1/4、1/8 預算；L1 X1） | ✗ 同位元組、Cake 還原下的比較（QEvict、ArkVale 最近） | 1 卡、品質；可併進 44e 的 DS 程式 | 1 | ★★★ |
| **I1** | 系統狀態 | 多 session 同時還原：各自 Cake vs 聯合決定 vs 順序；**實測計算與搬運互相干擾**（CacheFlow 假設不干擾） | △ CacheFlow、Strata、CacheSmith（L4 查全文中） | 1 卡、量時間 | 3 | ★★ |
| **V0** | 閘門 | vLLM 0.28 ROCm：APC、CPU offload、tiering 能跑、命中前後輸出相同 | — | 1 卡 | 1 | ★★（F1／F2／I3 前提） |
| **F1** | Fuse | 復用邊界：重疊 × 容量 × 併發 × {不復用／APC／APC＋卸載}；同時看平均與 goodput 的最佳設定會不會翻轉 | △ py-kvcache 只有單請求前沿；**併發的邊界沒人量** | 1 卡、量時間 | 3（子集） | ★★ |
| **I3s** | Infinity／agent | **工具暫停四選一（GPU／CPU／SSD／丟）的模擬**，用 repo 實測成本；比 Continuum、Tokencake、InferCept；看決策會不會隨硬體翻轉 | △ 都只做二／三選一、κ 不實測；**Cascade 待查全文** | **CPU（現在做，agent 跑中）** | 0 | ★★ |
| **I3g** | Infinity／agent | I3s 的 GPU 驗證點＋warm-up 預取（vLLM multi_turn 重播） | 同上 | 1 卡、量時間 | 2 | ★（I3s 有訊號才做） |
| **C1b** | Cascade | 選擇性還原的速度（只載前 k%＋需要時補載 vs 丟 vs 完整 Cake） | △ IMPRESS、ContiguousKV、CacheTune（都沒和 Cake 結合） | 1 卡、品質＋時間 | 3 | ★（C1a 過了才做） |
| **I2** | Infinity | 背景搬運拖不拖慢正在 decode 的請求 | △ HCache 只報 TBT +4% | 1 卡、量時間 | 1 | ★ |
| D* | 深入 | 只對有訊號的 1–2 個：GPU 忙（duty 0.3／0.7）、128K、更多併發、Qwen | — | 1–2 卡 | 12 | 視結果 |

**v0.2 改了什麼（相對 v0）**：
- C1 拆成 C1a（先證明分數有預測力，只要 forward）與 C1b（速度）；C1a 不過，C1b 不做（L1 G2／G4 的停止條件）。
- S2 改成 L1 的 X1：**只動載入區**（前段反正會被重算，丟了不省時間也不傷品質）；位元組預算 1/2、1/4、1/8。
- 新增 F2（vLLM #44294 重現，L3 X2），便宜、直接對應「復用不划算的區域」。
- I3 拆成 I3s（CPU 模擬，現在就做）與 I3g（GPU 驗證）。
- I4 併入 L2 的 X3（SDMA／blit × GPU 忙），一次回答「vLLM 的 CPU 層為什麼慢」和「GPU 忙時 H2D 為什麼慢」。
- 已知的新意風險：L2 G4（壓縮＋丟棄＋卸載的聯合模型）很可能退化成「全部 INT4＋G5b」這條規則 → 不單獨開實驗，由 S1、S2 的結果判定。

GPU 合計〔估〕：主表（不含 D）約 21 卡·時；D 約 12；緩衝 10。兩張卡平行約 22 小時牆鐘，在約 62 小時的窗口內。

**和第二階段佇列的關係**：44e（DS1／DS2 的 drop 品質、D3 的 drop 還原速度）已經在量「寫入時 ATT／還原時 QRY／位置／隨機」的丟棄。**S2、C1 直接讀 44e 的結果，不重量**；只補 44e 沒有的組合（延遲載入、第 2 輪、量化＋丟棄）。

---

## 3. 各實驗卡

### S1　無損壓縮 KV 的盈虧點（Shrink；對應影片的 QAT 壓縮）

- **假說**：BF16 KV 的指數位元低熵，無損壓縮能把卸載／傳輸的位元組減到夠低，在 NFS 這類慢鏈路上縮短還原；但在 CPU DRAM 層（H2D）不划算；而且**被第二階段已驗證的 INT4 KIVI（4 倍、品質不掉）支配**。
- **對照**：不壓縮 BF16；FP8／INT4 KIVI（有損，第二階段已有品質）。**處理**（`code/m16_s1_lossless.py`，自測 ALL PASS）：codec＝zstd（−3、1、3）、zlib（1、6；DEFLATE，和 QAT 同格式）；變體＝raw、bytes2（高低位元組分開）、**bytes2hi（只壓高位元組＝Intel `intel-accel-for-llm` 的 IAXL_KV_DATA_SHUFFLE＋IAXL_ZIP_EXPONENT_ONLY，L0）**、planes（exponent／sign＋mantissa）、planes_e（只壓 exponent）；order-0 熵（理論上限）；也對 FP8、INT4 的碼做一次。GPU 端的 SplitZip（純 Triton，r ≤ 4/3，L0）列為選做。
- **固定**：Llama-3.1-8B、Qwen2.5-7B；KV 由 `code/m16_s1_kvdump.py` 在 GPU 窗口匯出（32K 全層；128K 選 5 層），T4（HotpotQA 自然文字）與 T2（RULER 重複句子）分開看。**原本想用 CPU prefill 先拿 KV，取消**：第二階段要求計時期間不能有任何行程開 /dev/kfd，而沒辦法事先證明 CPU 上的 torch 不會開它（無法用 mount namespace 隔離）。
- **指標**：壓縮比（逐層、K／V 分開）、壓縮／解壓 GB/s（1、8、16 核）；〔計算〕各層級（CPU、本地 NVMe、NFS）的還原時間＝max(讀, 解壓) 或相加，標明假設。
- **成功**：BF16 壓縮比 ≥ 1.3 且解壓速度 > NFS 頻寬 → 列為 NFS 層的候選。**停止**：BF16 壓縮比 < 1.2，或在每個層級都被 INT4 支配 → 記為負面結果，不再投入。
- **產出**：`results/m16_stage3_mi300x/s1_lossless/`。

### V0　vLLM 0.28 ROCm 功能閘門

- 確認 `--enable-prefix-caching`、`OffloadingConnector`（CPU）、`kv_offload/tiering`（檔案層）在 MI300X 上能跑、KV 逐位元正確（同一 prompt 命中 vs 不命中的輸出相同，temperature 0）。
- 不過 → F1、I3 改用我們自己的 Restorer（失去「真實 serving 引擎」，但實驗還能做），並在 log 寫明。
- **產出**：`results/m16_stage3_mi300x/v0_vllm_smoke/`。

### I1　多個 session 同時還原（系統狀態；任務 §5 方向四、§6 第 1、5、7 點）

- **假說**：Cake 的會合點是**單一請求**在「整條鏈路＋整張 GPU」下算的。N 個 session 同時回來時，鏈路和 GPU 都被分掉，各自照 Cake 切會：(a) 大家都搶著重算前段 → GPU 成為瓶頸；或 (b) 都搶鏈路。**聯合決定**（例：全域的會合點、或讓某些 session 全載入、某些全重算）能降低 P95。
- **對照**：各自 Cake（現行）、全部載入、全部重算。**處理**：全域會合（把 N 個請求當成一個大請求切）、先到先服務的序列化、最短優先（SRPT）。
- **固定**：Llama 32K BF16（主）、INT4 KIVI（副）；層級 CPU、b32（限速 1.47 GiB/s）、NFS 級；N ∈ {1, 2, 4, 8}；GPU 閒（主）／忙 duty 0.7（副）。
- **指標**：每個 session 的還原延遲 P50／P95、全部完成時間、有效 H2D 頻寬、GPU 使用率；N=1 要對回第二階段的值（尺準檢查）。
- **重複**：2 launch × 5 rep；中位數、四分位距、配對 bootstrap CI。
- **成功**：N=8 時，最好的聯合策略比「各自 Cake」的 P95 少 ≥ 15%。**停止**：< 5% → 併發不改變單請求的結論（也是有用的負面結果：第二階段的結論在併發下成立）。
- **需要的程式**：Restorer 多 session 版（現在是單請求；X1 盤點後確定要改多少）。
- **產出**：`results/m16_stage3_mi300x/i1_concurrent_restore/`。

### C1a　閘門：分數能不能預測下一輪要用的 block（L1 X2＋X4）

- **假說**（L1 寫死）：寫入時的分數（SNAP、EXPATT、KNORM、KEYDIFF、H2O 近似、STREAM、NEW、RND）對第 2 輪答案所在 block 的召回≈隨機；第 1 輪 QRY 高於隨機但明顯低於第 2 輪 QRY；Quest 式 min/max 摘要（512-token block、post-RoPE K）對第 2 輪 QRY 前 k 名的召回明顯低於完整 K。
- **做法**：DS2 的兩輪題目（seed 70000 起：T2–T4 各 45 題、T5 15 題）× 2 模型；每題一次 32K prefill＋兩個問題的短 forward＋打分，不生成。重用 `code/m10_drop.py` 的分數（L1：「唯一能在期限內直接用的路」）。
- **指標**：第 2 輪答案 block 落在前 k（k＝16、32）的比例 vs 隨機期望 k/63（bootstrap CI）；各分數與第 2 輪 QRY 的 Spearman ρ；摘要與第 2 輪 QRY 前 k 名的重疊率。
- **停止**：寫入時分數≈隨機、且摘要也≈隨機 → 選擇性還原只能靠位置與下一輪的問題，C1b 不做（寫成負面結果）。
- **產出**：`results/m16_stage3_mi300x/c1a_predictive/`。

### C1b　選擇性還原：不載入 ≠ 丟掉（只在 C1a 過了才做）

- **假說**：回來的問題只需要少數 block。用 query-aware 分數（Quest 式：每個 block 的 key 最小／最大值當摘要，常駐 GPU，很小）只載入前 k%，其餘**留在下層**、decode 時分數超過門檻才補載，可以在品質不掉的情況下大幅減少搬運；而**第 2 輪換問題**時，「丟掉」會壞、「留在下層」不會。
- **對照**：完整還原（Cake）、第二階段 44e 的丟棄（ATT／QRY／位置／隨機）。**處理**：只載前 k%（k ∈ {10, 25, 50}）＋(a) 不補載（＝丟）、(b) 補載。
- **工作負載**：T2–T5（資訊在前／中／後帶）、32K（主）與 128K；**2 輪題**：第 1 輪問 A 區、第 2 輪問 B 區（同一份 context）；另用 SCBench 的多輪共享 context 子集當外部效度（只跑 Llama）。
- **指標**：答對率（配對差、95% CI，題數照第二階段的 n 規則，主格 ≥ 48 題）、實際搬運位元組、首 token 時間、補載次數與補載位元組；**分數 vs 實際需要**：被選中的 block 有沒有蓋到 needle 的位置（我們有 needle 的 token 位置）。
- **成功**：k=25% 時答對率對完整還原差 ≥ −2 個百分點（ε，沿用第二階段），搬運位元組 ≤ 30%；且第 2 輪時 (b) 明顯好過 (a)。**停止**：k=50% 都掉超過 ε → query-aware 選擇在這些任務不可靠。
- **產出**：`results/m16_stage3_mi300x/c1_selective_restore/`。

### F1　前綴復用地圖（Fuse）

- **假說**：復用的收益不是常數；在「低重疊＋KV 容量緊＋高併發」的區域，APC 與 CPU offload 的收益趨近 0 甚至為負（多出的管理與搬運成本、命中但被擠掉）。
- **工作負載**：可控重疊的產生器（共同前綴佔 0／25／50／75／100%，prompt 8K／32K，輸出 128）；到達：Poisson（多個 QPS）＋一次全部到。
- **配置**：無 APC、APC、APC＋CPU offload（V0 過了才有）；KV 容量：充足、緊（`--num-gpu-blocks-override` 讓它只放得下約 1.5 倍工作集）、很緊（0.75 倍）。
- **指標**：命中率（vLLM 的 metrics）、TTFT P50／P95、TPOT、吞吐、goodput（TTFT SLO 事先定兩個值）、CPU offload 的搬運位元組。
- **成功**：找到「收益 < 5%」或「變差」的區域，而且能用（重疊、容量、併發）三個變數說清楚邊界。**停止**：全部區域收益都很大且單調 → 沒有研究問題，只當 baseline。
- **產出**：`results/m16_stage3_mi300x/f1_prefix_reuse/`。

### I3（I3s 模擬＋I3g 驗證）　agent 多輪重播：工具暫停時的 KV 去留＋預測預取（Infinity＋Fuse）

- **v0.2**：先做 I3s——CPU 模擬（`code/m16_pause_sim.py`，agent 寫中），成本常數全部從 repo 的實測檔讀（附 run_id），間隔用 Mooncake toolagent trace 推得＋參數化分布；對手＝vLLM 預設、Continuum TTL、Tokencake、InferCept、κ 感知四選一、oracle。證偽：κ 感知比最好的現成規則 P95 好不到 5%，或兩平台選的動作完全相同。過了才做下面的 GPU 版（I3g）。外部 agent trace（Exgentic、SemiAnalysis、TraceLab，合計約 3 GB）**要先問使用者才下載**。

- **假說**：工具暫停期間，KV 該留 GPU、卸載 CPU、還是丟掉重算，取決於暫停長度、context 長度、容量壓力；而**在預計回來前先送一個「只 prefill、不生成」的 warm-up 請求**（不用改 vLLM，就能把 KV 從 CPU 拉回 GPU）能藏掉大部分還原時間，代價是預測錯時白搬。
- **工作負載**：S 個 session × T 輪；context 每輪加工具輸出（長度分佈取自 Mooncake toolagent trace，**以 512-token block 換算，載入時驗算長度**）；暫停時間取 1–120 s 的分佈。
- **配置**：GPU only（被擠掉就重算）、CPU offload、CPU offload＋warm-up（提前量 Δ ＝ 0、0.5、1、2 倍的還原時間；預測誤差 0／±50%）。
- **指標**：每輪 TTFT P50／P95、吞吐、多搬的位元組、白搬的比例。
- **成功**：warm-up 在 Δ ≥ 1 倍還原時間時藏掉 ≥ 50% 的 TTFT，白搬 < 20%。**停止**：CPU offload 本身已讓 TTFT 很小（還原不是瓶頸）→ 這個方向在 MI300X 上沒問題可解。
- **產出**：`results/m16_stage3_mi300x/i3_agentic_replay/`。

### S2　等位元組：載入區的有損刪除 vs 降精度（L1 X1）

- **為什麼只動載入區**（L1 G1）：Cake 還原時前段反正會被精確重算，丟了不省時間也不傷品質；有損刪除只在載入區有意義。所以只動 block 32–63（block 0 一律保留）。
- **問題**：同樣位元組預算（相對 BF16 約 1/2、1/4、1/8），哪一種最好：丟 50%／75%／87.5%（DS1 選出的前 2 名寫入時方法、QRY、RND0） vs FP8 或 INT8／INT4 KIVI／INT2 KIVI 或 K4V2；E0、MASK0、ORC 當上下界。「drop 是不是只是最低一階的放置狀態」。
- **大部分沿用**第二階段的量化品質與 44e 的丟棄品質；只補「量化＋丟棄」的組合。
- **指標**：答對率（T2–T5，32K／128K）、還原時間（CPU、b32、NFS 級）、存的位元組。畫品質—位元組的 Pareto 前緣。
- **成功／停止**：若 INT4 均勻在每個預算都支配丟棄 → 丟棄只在 < 25% 的預算才值得討論；反之則 drop 進入第三階段的動作空間。
- **產出**：`results/m16_stage3_mi300x/s2_equal_bytes/`。

### I4　零碎 copy 批次化（DSA 的對應；L0 §3、§4）

- **事實（舊量測，附出處）**：vLLM 0.28 的 CPU offload 層只有 4.0–4.1 GiB/s；同一台的 pinned H2D 約 50 GiB/s；F1 的分析是每次 copy 約 3.5 µs × copy 次數（`docs/research_20261010_followup/F1_offload_slowdown.md` §4.6，run `20261010-112119-f1-s1-rocmfixcoal`）。
- **假說**：瓶頸是送出 copy 的固定成本，不是頻寬。把 vLLM 的 block（16 token × 層）攤成一次 gather kernel 或批次送出，CPU 層的有效頻寬能接近 pinned H2D。
- **對照**：逐片 `copy_`（SDMA，預設）；**處理**：`HSA_ENABLE_SDMA=0`（blit kernel）、自寫的 gather kernel（Triton，從 pinned host 讀到 GPU 的分頁）、先在 CPU 端合成連續緩衝再一次 H2D。上游對應：ROCm PR #11614（`hipMemcpy3DBatchAsync`，未合併）。
- **固定**：片段大小＝vLLM 的 block（Llama：16 token × 32 層 × 2 × 8 × 128 × 2 B），片數＝32K／128K 的量；GPU 閒／忙。
- **指標**：有效 GiB/s、每片成本（µs）、CPU 使用率。
- **成功**：最好的處理 ≥ 3 倍於逐片 → F1／I3 的 vLLM 數字要分「引擎現況」和「修好之後」兩組報告，且這本身是一個可寫的系統問題。**停止**：差 < 1.5 倍。
- **產出**：`results/m16_stage3_mi300x/i4_copy_batching/`。

### F2　重現 vLLM #44294：命中但正在載入 → 延後（L3 X2）

- **事實（前人）**：OffloadingConnector 遇到「命中的 block 正被別的請求載入」時把請求延後到下一輪排程；L40S、50 併發共用 10K 前綴，開卸載 11.46 s、不開 5.95 s（issue #44294，2026-06-02）。0.19.1 的位置是 `offloading/scheduler.py:126-140`；0.28 是否還在由 harness agent 用原始碼確認。
- **做法**：先一個請求寫入前綴 P ∈ {10K, 32K}，再把它擠出 GPU（填充請求或 `--num-gpu-blocks-override`），最後同時送 N ∈ {1, 8, 32, 64} 個共用該前綴的請求。設定：(A) APC 不卸載、(B) APC＋CPU 卸載、(C) 不復用。
- **指標**：TTFT P50／P95、吞吐、connector log 裡延後訊息的次數。
- **停止**：所有 N 下 (B) ≤ (A) → MI300X／0.28 不受影響，停。成立 → 最小改法是「命中正在載入的 block 時，等同一個載入或改重算」＝ κ 決策在併發下的版本；但單一 issue 只值一個工程修正，要和 F1 的邊界一起看才有論文價值（L3 缺口 2）。
- **產出**：`results/m16_stage3_mi300x/f2_delayed_hit/`。

### I2　背景搬運對 decode 的干擾

- 一個 batch 在 decode（量 TPOT），同時在背景做還原／預取（CPU→GPU、限速與不限速），看 TPOT 變多少。決定 I3 的預取是不是「免費」。第二階段 13 號在 S4 量過「背景工作不受影響」（08 §0 第 1 列），這裡換成 decode 負載再確認。
- **產出**：`results/m16_stage3_mi300x/i2_interference/`。

---

## 4. 四大方向之外（任務 §6），先列、文獻回來再評

| # | 候選 | 從哪個實驗得到證據 |
|:--|:--|:--|
| E1 | 負載感知的還原排程（多 session 聯合決定載入或重算、誰先） | I1、D |
| E2 | 執行中才知道的 κ：GPU 爭用、鏈路爭用讓「重算 vs 搬運」的比值變動 | I1、I2 |
| E3 | 選擇性還原＋延遲載入（不載入 ≠ 丟掉），多輪下的重要性漂移 | C1 |
| E4 | agent 工具暫停期間的 KV 去留（TTL）與預測預取 | I3 |
| E5 | 以 goodput／P95 為目標的 KV 放置與 admission | F1、I3 |
| E6 | 壓縮—丟棄—卸載的統一成本模型（等位元組階梯） | S1、S2 |
| E7 | 搬運的「送出成本」而非頻寬才是 ROCm 上 KV 卸載的瓶頸（零碎 copy） | I4 |

每個候選最後要回答（任務 §6）：有沒有人做過、新意在哪、要什麼硬體與軟體、MI300X 上能否 PoC、最小證偽實驗、什麼結果出現就停。

---

## 5. 時程（UTC；台灣＝＋8）

| 階段 | 時間 | 做什麼 |
|:--|:--|:--|
| A | 10/19 13Z → GPU 窗口開（〔估〕10/20 10–14Z） | 文獻與核實（agent）；S1（CPU，挑第二階段跑品質的時段做，避開量時間的 stage）；寫 I1、C1、F1、I3 的程式與 CPU 測試；寫 runner |
| B | 窗口開 → ＋7 h | S1 的 KV 匯出（第一個 job，10 分鐘）、V0、I4、I1、I2 |
| C | → 10/21 18Z | C1、F1、I3、S2 |
| D | 10/21 18Z → 10/23 02Z | 有訊號的 1–2 個加深；最後 8 小時只排保證跑得完的 |
| E | 10/23 02–09Z | 分析、`stage3_candidates.md`、`reproduction.md`、commit／push、備份到 /mlsteam/data |

---

## 6. 執行規則（和第二階段共用機器）

- **不動第二階段的佇列**（不移、不刪、不改順序）。第三階段用自己的 runner（`code/m16_queue/`），拿同一組 `/mlsteam/data/tiara/gpu{0,1}.lock`；**只在第二階段那張卡的佇列已空時才開新 job**，每個 job ≤ 2 小時，所以第二階段之後新加的 stage 最多等一個 job。
- 量時間的 job 照第二階段的規矩：兩張卡獨佔，除非 44cb 判出 `XGPU_OK`；`GpuWatcher` 包住，`contaminated` 的時間欄作廢重量（CLAUDE.md §3）。
- CPU 工作在第二階段量時間時：綁 ≤ 4 核、`nice 19`；NFS 的大量讀寫只在沒有量時間的 stage 時做（44e3 會量 NFS 級還原）。
- 每個 run：`/mlsteam/data/tiara/runs/<run_id>/`（cmd、context、stdout、stderr、exit_code）；CSV 一律有 `run_id`、`ts`。
- 交付物（任務 §10）都放 `docs/stage3_20261019/`；數字表在 `results/m16_stage3_mi300x/`；RUNLOG 開 Milestone 16。
