# 第三階段實驗紀錄（逐實驗）

> 規則：每個數字都附 run_id 與輸出檔；〔模擬〕＝用實測常數算的，不是量測；NOT_MEASURED＝還沒量。原始 log：`/mlsteam/data/tiara/runs/<run_id>/`（cmd.sh、context.txt、stdout、stderr、exit_code）。
> 時間一律 UTC（台灣＝＋8）。

---

## X0　環境快照（10/19 13:01–13:34Z）

- 產出：`environment_snapshot.md`。只用唯讀指令，沒開 GPU 行程。
- 失敗與異常：無。`unshare -m`（想用 mount namespace 隔離 /dev/kfd）→ `Operation not permitted`；改成「計時期間不 import torch」的規則。

## X1　第二階段十條觀察的核實＋工具箱（10/19 13:0x–13:2xZ，agent，self-review）

- 產出：`stage2_factcheck_and_toolbox.md`。
- 結論：十條裡第 2–10 條出自**第一階段**（M7）；202／2720 ms 是含新一輪的 TTFT、「NFS」是限速模擬；0.33／6.98 GiB/s 是限速參數。詳見 `experiment_matrix.md` §1。
- 順帶發現第二階段的分析 bug（`m10_longctx_analyze.py` 沒篩長度）→ 已轉告第二階段 session，修好並重跑（commit 79cbdc7，RUNLOG_MI300X 15.11）。

## X2　文獻地圖（10/19 13:0x–13:5xZ，4 個 agent，self-review）

- 產出：`lit/L0`–`L3`、彙整 `research_map.md`。L4（全文查新：Cascade、CacheSmith、CacheFlow）跑中。

---

## S1　無損壓縮（程式就緒，資料等 GPU）

- `code/m16_s1_lossless.py --self-test`：ALL PASS（25 項；5 個變體 × zstd／zlib 的速度測試都逐位元相同）。過程抓到一個自己的 bug：zstd context 被兩個 thread 共用 → "Data corruption detected"，改成每個 thread 一份。
- `code/m16_s1_kvdump.py`：還沒跑（要 GPU；第三階段窗口的第一個 job）。
- 關鍵數字：NOT_MEASURED。

## I1　多 session 同時還原：事先預測〔模擬〕（10/19 13:40Z）

- **run_id**：`20261019-134046-m16-i1pred`（之前兩次 `-133930-`、`-134035-` 是同一支程式的草稿版與一次 bug（變數遮蔽，exit 1），以這次為準）。
- **指令**：`m7run m16-i1pred nice -n 19 taskset -c 0-3 python -I code/m16_i1_predict.py --model llama --out results/m16_stage3_mi300x/i1_concurrent_restore/prediction`
- **產出**：`results/m16_stage3_mi300x/i1_concurrent_restore/prediction/prediction_llama.{csv,json}`。
- **模型**：`code/m16_mcake.py`（多 session 的 Cake 排程；一條計算線、每層一條載入線；自測 ALL PASS）。常數全部讀檔：f(i)、chunk、pinned H2D 49.858 GiB/s、b16／b32／b48（`speed_llama/bw.json`，run `20261015-154558-m10-calibf-llama`）；新一輪 0.0755 s（閒）／0.235 s（忙）、單 session 實測還原（`speed_llama/summary_fp8.csv`，run `20261015-161226-m10-grid-fp8-llama`）；GPU 忙時重算 ×2.07（`busy_levels.json`）。「載入在計算忙時 +50%」只是敏感度參數（08 §2.1 的算術，不是本實驗的量測）。
- **尺準**（N＝1，模擬 vs 第二階段實測，GPU 閒）：cpu 0.0889 vs 0.0954 s（×0.932）、b16 0.5688 vs 0.5782（×0.984）、b32 1.3635 vs 1.3802（×0.988）、b48 2.3832 vs 2.4215（×0.984）。
- **關鍵數字〔模擬〕**（N＝4、8；四種工作負載 × 閒／忙 × 有無干擾）：
  - **A：從「各自 Cake＋輪流」到「各請求各自最佳」（第二階段的 G5b＋先到先服務／最短優先）**：P95 少 4–42%、平均少 10–58%。來源是兩個已知的東西：G5b（CPU 層全部載入比 Cake 快，第二階段 11 問題三）與排隊順序（處理器共享 vs 先到先服務的教科書結果）。
  - **B：從「各請求各自最佳」到「同時考慮所有 session」（joint，每個 session 指定全部重算或全部載入、讓兩條線總量平衡）**：**每一格都是負的**（−14% 到 −60%）。在這個模型下，聯合決定沒有好處。
- **判讀**：I1 原本的新意假說（「聯合決定比各自 Cake 好」）在「互不干擾」的模型下**不成立**；能拿到的好處都來自已知的規則與排隊順序。剩下唯一的問題是「真實的干擾會不會讓模型錯」→ GPU 實測降級成 1 小時的驗證（N＝4、8，cpu／b32，閒；看實測和預測差多少）。如果實測和預測差 <10%，I1 這個方向收掉（負面結果：第二階段的單請求規則＋排隊順序就夠）。
- **失敗與異常**：無（第二次執行的 exit 1 是自己的變數遮蔽 bug，已修，見上）。
- **限制**：joint 只試了「每個 session 全算或全載」這一種；更細的聯合（例如每個 session 的會合點一起調）沒試。

## I3s　agent 工具暫停時 KV 去留的模擬〔模擬〕（10/19 14:1xZ，agent 寫、我核對）

- **run_id**：`20261019-141254-m16-pausesim`（exit 0，11,232 次模擬）。第一次 `20261019-141044-m16-pausesim` exit 1：帳目斷言抓到「下一輪截斷歷史時，舊副本記成整段長度」的 bug，已修、加回歸測試（`FAILED_NOTE.txt`）。
- **程式**：`code/m16_pause_sim.py`（自測 74 項 ALL PASS；只用 CPU，結尾斷言沒開 /dev/kfd）。報告：`i3_pause_sim.md`。結果：`results/m16_stage3_mi300x/i3_pause_sim/`。
- **常數**：MI300X（Llama、Qwen）全部讀檔、附 run_id（`costs_sources.csv`）；延伸處已標（Qwen 新一輪 <32K 用 32K、GPU 忙的倍數 <32K 用 32K、Qwen 的 D2H 沿用 Llama）。3090 缺 D2H、忙的倍數、原始 H2D，而且舊量測是 HEAVY 污染 → 只進靜態表（NOT_MEASURED）。
- **判定（照 L3 缺口 1 事先寫的證偽條件）**：事前凍結的 κ 感知四選一，主掃描 **0／96 格比最好的現成規則好 ≥5%**（中位 −2.8%，最好 +1.3%）→ **缺口不成立，照 CLAUDE.md 規則 4 停**。
- **探索性（不進判定）**：
  - 最好的現成規則多半是 vLLM 0.28 原生的 CPU＋SSD 串寫；勝負取決於「有沒有快的下層接住 CPU 放不下的 KV」。
  - 下層從本地 SSD 換成 NFS：31／32 組最好的規則換人；把本地 SSD 上的贏家搬到 NFS，P95 中位 +60%（最多 +482%）——「哪個規則最好」隨硬體翻轉，但 κ 感知本身打不贏該硬體上最好的規則。
  - 事後加的 `kappa_tier` 只在 CPU 層縮成 0.25×HBM 時明顯贏（中位 +49%），一部分來自 vLLM 沒有的寫入路徑；事後候選，不能下判定。
  - 真實 HBM（192 GB）下沒有容量壓力：C＝32 時容量是參考工作集的 4.2 倍（Llama）／9.5 倍（Qwen）。
  - 最大的改善空間是「知道 session 已經結束」（NFS 設定下中位 +26%），比 κ 大。
- **限制**：這是模擬（常數是實測的，排程與佇列是模型）；GPU 驗證點沒有跑。
- **對 agent 方向的結論**：使用者的「場景改 agent 感覺太急」在這個模擬裡得到支持——κ 感知的去留決策沒有空間。剩下兩個可能的改題（「依實測 κ 自動選規則」「應用層告知 session 結束」）要使用者決定要不要考慮。

## I3r　I3 模擬換基準重讀〔模擬〕（10/20 04:02Z）

- **為什麼**：使用者質疑探索「太多否定」；我（10/20）說過「I3 只跟神諭比，換成固定作法結論會翻」。這次照實重算檢查那句話。
- **run_id**：`20261020-040244-m16-i3reread`（同一支程式的前一次 `-040223-` 少了 vllm_tier 基準，以這次為準）。只讀 `20261019-141254-m16-pausesim` 的 `cells.csv`／`regret.csv`，不重跑模擬。
- **指令**：`m7run m16-i3reread nice -n 19 taskset -c 0-3 python3 -I code/m16_i3_reread.py`
- **產出**：`results/m16_stage3_mi300x/i3_pause_sim/reread_baselines.{csv,json}`。
- **關鍵數字〔模擬〕**（回來那一輪 TTFT P95；gain＝1 − 事前凍結的 κ ÷ 基準，>0＝κ 較好；中位數）：

| 設定組 | 對神諭（每格最好） | 對 vLLM 原生卸載（vllm_tier，固定） | 對不卸載的 vLLM 預設 |
|:--|--:|--:|--:|
| main（本地 SSD、CPU 大，96 格） | −2.8% | −1.7% | +9.8% |
| GPU 外加忙（busy_exo） | −48.9% | −48.9% | +67.7% |
| Cake 還原 | −65.5% | −65.5% | +68.8% |
| 下層換 NFS（ssd_nfs） | −5.3% | **+27.3%** | +69.7% |
| CPU 層小（cpu_small） | +35.6% | **+35.6%** | +50.9% |
| CPU 層小＋NFS | +7.1% | **+38.2%** | +14.1% |

- **搬錯硬體**（regret.csv）：本地 SSD→NFS 方向，κ 比「原封搬過去」好 31／32 組（regret 中位 5.3% vs 60.3%）；**反方向 NFS→本地 SSD 只好 3／32**（κ 57.3% vs 25.7%）；GPU 閒↔忙兩個方向都是 0／32（原封搬過去幾乎沒損失）。
- **判讀**：
  - 我 10/20 說的「換基準結論就翻」**大部分不成立**：在本地 SSD、CPU 層夠大的設定（main、忙、Cake、sigma2），連可部署的固定規則（vLLM 原生卸載）κ 都打不贏，原判定（停）在這些設定成立。
  - **只有在不尋常的硬體**（下層是 NFS、或 CPU 層小）κ 才明顯贏 vLLM 原生卸載（中位 +27～38%）；但這些設定裡現成的 Tokencake 也贏 vLLM 原生卸載，κ 對神諭仍差 5%（ssd_nfs）。也就是：價值在「硬體不尋常時別用 vLLM 預設」，不在 κ 本身。
  - 「κ 能避免選錯」只在 SSD→NFS 一個方向成立，不能一般化。
  - 對「不卸載」的預設，任何卸載規則都大贏，這個基準太弱，不能拿來支持 κ。
- **限制**：同 I3s（模擬；成本常數是第二階段 harness 路徑，vLLM 實際 CPU 層約 4.1 GiB/s 沒代入）。

## 第三階段 GPU 窗口開始（10/20 13:59Z）

- 第二階段兩個佇列 13:59:17Z 清空（session a 通知）；runner 13:59:24Z 開 `00_smoke`。
- **失敗 1**：`00_smoke` 第一步 `m16_s1_kvdump`（run `20261020-135924-m16-smoke-kvdump`，exit 1）：
  `RuntimeError: T4 ['mid'] seed=70000：10 次都放不好：band mid 放不下：需要 812 token、只有 512`。
  原因：冒煙用 `--n-blocks 4`（2K token），`m10_tasks.bands(4)` 的 mid 帶只有 1 個 block，放不下 T4 的段落。是冒煙參數的 bug，不是實驗結果。
  修法：冒煙改 `--n-blocks 16`（mid 帶 4 個 block），輸出 `smoke/kv_llama_t4_16`；`06_k1_kvstruct.sh` 的 RoPE 核對路徑同步。失敗的 job 檔與 log 另存 `retired_20261020/`，NEEDS_HUMAN 改名 `.handled_20261020T1406Z`。14:00:51Z 重開 `00_smoke`。
- `00_smoke` 14:05:19Z 全過（kvdump 16 block、S1 分析、I4、I1 interf／grid 的最小設定）。
- **失敗 2**：`02_smoke_x` 的 Qwen `gpu-selftest`（run `20261020-140604-m16-x-gpuselftest-qwen`）48／49 過，`proxy_masked_eq_subset` rel＝7.1e-5 > 門檻 1e-5；同一項 Llama 7.8e-8（49／49 過）。
  判讀：兩條路都是 FP32，只差 GEMM 形狀（加總順序）；Qwen 第 0 層 K 有大值，softmax 放大捨入差。遮罩寫錯（少 10 個 block）會是大得多的差。
  修法：門檻改 1e-3，另記「完全不遮」的差當反向對照（`code/m16_xstack.py`，理由寫在註解）。Llama 那半照跑完：calib、screen --smoke（每題 65.8 s、peak 41.7 GiB）、gen 32K、gen 128K 都 exit 0。14:13:40Z 重開 `02_smoke_x`。
- 冒煙全過：`02_smoke_x`（重開後 Qwen 49／49；proxy 遮罩 vs 子集 7.1e-5、完全不遮 3.9e-2）、`03_smoke_xtime`、`04_smoke_c1a`（own_gpu_intruders 0）。`05_s1_kvdump` 14:24:56–14:27:41Z（兩卡並行，約 52 GB 在 /var/tmp/tiara_s3/kv_dumps）。

## I4　KV 從 CPU 搬回 GPU 的搬法（10/20 14:27–14:36Z，CLEAN）〔實測，初步〕

- run：`20261020-142741-m16-i4-sdma0`、`20261020-143218-m16-i4-sdma1`。產出 `results/m16_stage3_mi300x/i4_copy_batching/copybench_sdma{0,1}.{csv,json}`，20 格 × 2，全部 ok、未污染。原估 2 h，實際 8 分鐘。
- 有效 H2D（GiB/s，每格一個彙總值）：散在小塊（scatter）時，逐塊 copy（py_loop／foreach）1.1–1.4（SDMA 關）、2.2–3.0（SDMA 開）；合併（coalesce）在 scatter 也只有 1.3–3.0；CPU 端先收集再一次 copy（stage）17.7–20.3；GPU kernel 直接讀 pinned host（zerocopy）53.1，SDMA 開關與 GPU 閒／busy2 都一樣。連續（contig）時 coalesce 49–53。
- 初步判讀（待 results_analysis 細看）：逐塊 copy 的量級與 vLLM ROCm CPU 層約 4.1 GiB/s 一致；zerocopy 比逐塊快 18–45 倍。busy2 對各搬法幾乎沒影響，和第二階段「GPU 忙時 H2D +46–50%」不同（設定不同：這裡是單純搬運的 microbench），要對照 I1 再說，不下結論。

## K1　KV 結構（10/20 14:28–14:39Z，CPU）〔實測：KV 幾何量，不是品質〕

- run：`20261020-142819-m16-k1-kvstruct`；RoPE 核對 6 份 dump 全部 PASS_PROBE。產出 `results/m16_stage3_mi300x/k1_kvstruct/`（lowrank、lowrank_eff、xlayer、adjtok 的 CSV／JSON、summary.md）。
- **低秩**（rank 32＝維度 1/4 的相對重建誤差，層×head 中位數；T4 32K）：K 在 RoPE 前 Llama 0.253／Qwen 0.241，RoPE 後 0.461／0.423；V 0.665／0.649。95% 能量需要的 rank：K_pre 40／36、K_post 84／82、V 110／108.5（滿 128）。用前 4096 token 的基底套到後面：K_pre 0.268／0.253（幾乎不變差），K_post 0.616／0.580（明顯變差）。128K 與 32K 相近（K_pre r32：0.240／0.237）。
- **跨層**（同一 token、相鄰層的 cosine，深半段中位數）：K、V、中心化前後、32K 與 128K、兩個模型全部在 −0.016～+0.022；整層 head 串接也是 ≈0（最大 0.066）。→ 照 L5b 寫死的條件（深半段中心化 cosine 中位數 <0.5），MiniCache／KVSharer 式「逐 token 直接共用」在這兩個模型上沒有依據，停。xKV 式（共用子空間）不受這條影響，交給 X1 量品質。
- **相鄰 token**：K 的 cosine 中位數看起來高（RoPE 後 0.87），但主要來自共同的平均方向：減掉平均後中位數 0.52／0.55，≥0.9 的相鄰對只有 0.3%／0.7%；V 中位數 0.42、≥0.9 約 0.1–0.2%。T2（重複句子）去平均後更低（0.20）。→ 合併相似 token 能利用的天然冗餘很少。
- 這些是 KV 本身的幾何量；能不能換成答對率，要看 X1／X2。

## I1　計算／搬運／decode 互相拖慢＋多 session 實測 vs 模擬（10/20 14:36–14:53Z，CLEAN）〔實測〕

- run：`20261020-143616-m16-i1-interf`、`20261020-143840-m16-i1-grid`（171 列，全部未污染）。產出 `results/m16_stage3_mi300x/i1_concurrent_restore/`。
- interf（同時跑／單獨跑的時間比，中位數）：GPU 閒：重算 ×1.063、H2D ×1.021、decode ×1.055（和 H2D 同時）、H2D ×1.003（和 decode 同時）；busy2：重算 ×1.047、decode ×1.040，H2D ×0.944／×0.844（同時跑反而比單獨快，原因未查，先不解讀）。→ 在這組設定下，還原的計算與搬運互相拖慢 ≤6%。
- grid（實測平均 TTFT ÷ 同行程模擬，每格 3 次的中位數；GPU 閒；N＝1、4、8；cpu／b32／混合；cake／g5b／joint × rr／fcfs／srpt）：全部在 0.973–1.050。
- **判定（照事先寫的條件：實測／模擬 ≤ 1.10 → 模擬可信、I1 收掉）**：成立。I1 的事先預測（「聯合決定」每一格都比各請求各自最佳差 14–60%）因此可信；好處只來自第二階段的 G5b 規則與排隊順序。I1 方向收掉（負面結果）。

## C1a　寫入時的分數能不能猜中第 2 輪要的 block（10/20 14:53–15:10Z；report 15:5xZ）〔實測〕

- run：`20261020-145315-m16-c1a-{llama,qwen}-s{0,1,2}`、`20261020-150131-m16-c1a-{llama,qwen}-s{3,4,5}`；每個模型 T2–T4 135 題＋T5 15 題，齊全、逐題檢查沒有 false。產出 `results/m16_stage3_mi300x/c1a_predictive/`（tables_c1a.md、c1a_summary.json）。
- recall@16 比隨機（25.4%）高幾個百分點（T2–T4，95% CI）：寫入時分數 Llama KNORM +23.6、KEYDIFF +14.7、SNAP +10.8；Qwen KEYDIFF +19.2、KNORM +13.3、SNAP +13.1。第 1 輪問題（QRY1）+14.6／+21.3。還原時用新問題打 block 摘要（QUEST_MEAN）+34.7／+44.5；完整 K（FULLK_MEAN，上界）+61.1／+67.6。隨機對照（RND0）−0.1／−2.0。
- **判定（G2、G4 照原文）**：兩個模型都不觸發（寫入時分數與 Quest 摘要都明顯高於隨機）→ C1b（選擇性還原，量省多少時間與答對率）照計畫可以做。
- 判讀：寫入時就決定「哪些重要」只抓得到一部分；等下一輪問題來了再用便宜的 block 摘要挑，召回率高出一大截。T5（全域任務）沒有目標 block，表上是「—」。

## S1　無損壓縮比與 CPU 解壓速度（10/20 15:09–15:54Z，GPU 閒置、CPU_QUIET）〔實測〕

- run：`20261020-150947-m16-s1-analyze`（＋breakeven）。產出 `results/m16_stage3_mi300x/s1_lossless/`。
- BF16 KV 的無損壓縮比（最好的 zstd:1 變體）：自然文字 32K 1.47（兩個模型）、128K 1.51–1.52；重複句子（T2）1.67–1.72。Intel KV Shrink 的「只壓高位元組」（bytes2hi）1.46，和全部壓幾乎一樣。照 L5c STOP-1（<1.15 停）：不觸發，無損這一軸保留。
- CPU 解壓速度（壓縮比約 1.46 的格式）：0.5–2.1 GB/s（1–16 thread）。
- 盈虧（pipelined，相對不壓縮的時間）：NFS 級（0.33–0.37 GiB/s）省 32%；限速 b32（1.47 GiB/s）省 21–25%；本地 SSD（7.8 GiB/s）沒好處（Qwen 最好 +1.2%，Llama 全負）；CPU 層（49.9 GiB/s）大幅變慢。→ 無損壓縮在 CPU 上解，只對慢的下層（NFS、限速）划算。

## X1　壓縮疊加的 teacher-forcing 篩選（10/20 15:54–21:22Z；report 21:23Z）〔實測：TF 篩選，不是生成答對率〕

- run：`20261020-155515/-155519/-173921/-193056-m16-x1-{llama,qwen}-s*`（21 個），report `results/m16_stage3_mi300x/x_stack/x1_report.json`（tables_x1.md、pareto_{llama,qwen}.png、x2_selection.json、s2_subset/）。Llama 61 題、Qwen 96 題，各 161 個配置，範圍＝block 1–63（全段）。
- **判定（S1-b，事先寫死）**：0.125、0.0625 兩個預算、兩個模型，「最好組合」都明顯好過「最好單一方法」→ S1-b 不成立 → **X2 GO**。TF 版 r（ε 內最小位元組比，組合 ÷ 單一）：Llama 0.32（0.040 vs 0.123）、Qwen 0.35（0.082 vs 0.233）。
- 每個家族（同預算約 0.25）：量化 Llama q:kv-int4_kivi-int2_kivi Δgold_lp +0.05、Qwen −0.54；**丟 block（KNORM／KEYDIFF 0.75）−19.0／−26.4**；每層低秩 −0.11／−1.39；**跨層聯合低秩（xKV 式，每題 SVD）**最好；**合併 merge:4 −72.8／−70.9**（first_ok −29.5／−12.5 個百分點）。
- 最好的組合幾乎都是「xKV 式跨層低秩＋係數再量化」：Llama `xl:k11:v17+q:int4_kivi` 位元組比 0.048（約 21 倍）Δgold_lp +0.03、first_ok 不變；Qwen `xl:k22:v34+q:int4_kivi` 0.082（約 12 倍）−0.77、`xl:k28:v42+q:int4_kivi` 0.103 −0.53。Qwen 對壓縮比 Llama 敏感。
- S2 的問題（同位元組下丟 vs 量化）：丟 block 在每個預算都輸量化一大截 → 照 L1 G1：「在本設定下，有損刪除被量化支配」。
- 代理指標（以配置為單位的 Spearman，對 KL）：relerr_k +0.50、relerr_v +0.59；崩潰 AUROC 0.54–0.60 → 代理弱，只能當粗篩（和 L5d 的預測一致）。
- 限制：teacher forcing、第 1 輪；xKV 的基底是每題用整段 KV 算的（壓縮在 prefill 之後做），壓縮本身的成本（每 4 層一次 SVD）沒有算進位元組或時間；答對率以 X2 的生成為準。

## X2　挑出組合的生成答對率（32K core／x3／ext；10/20 21:41Z–10/21 02:52Z；report 02:52Z）〔實測〕

- run：`20261020-214139…20261021-020023-m16-x2-*`（28 個，見 tables_x2.md）；report `20261021-025230-m16-x2-report`。產出 `results/m16_stage3_mi300x/x_stack/{x2_report.json,tables_x2.md,x2_feasible.csv,x2_unmapped.csv}`。第 1 輪生成答對率；E0：Llama 79.5%、Qwen 75.7%（各 96 題；部分配置 61 題）。37d（ext 後半）、37e／37f（128K）還在跑，37zz 會出最終版。
- **判定（M1，事先寫死：任一模型 r＜0.85 → GO）**：Qwen r＝0.33（ε 內最小位元組：組合 `xl:k28:v42+q:int4_kivi` 0.103 vs 單一 `q:int4_kivi` 0.3125）→ **GO**。Llama：沒有任何組合通過 ε（r＝∞）；MLPerf 99% 門檻下 r＝1.12（單一 `xl:k11:v17` 0.123 已達 99%）。
- 主要數字（對 E0 的百分點，95% CI）：Llama `xl:k11:v17`（約 8 倍）+0.3 [−3.2, +4.0]；`xl:k11:v17+q:int4_kivi`（約 21 倍）−2.7 [−6.4, +0.8]；`xl:k22:v34+q:int4_kivi`（約 10 倍）−2.4 [−7.2, +1.9]；`q:int4_kivi` −0.8 [−3.7, +1.4]；`q:int2_kivi` −1.0 [−6.4, +4.7]。Qwen `xl:k28:v42+q:int4_kivi`（約 10 倍）+1.9 [−1.5, +5.5]、`xl:k22:v34+q:int4_kivi`（約 12 倍）+0.9 [−3.0, +4.9]、`xl:k11:v17+q:int4_kivi`（約 24 倍）−6.8 [−13.5, −0.3]。
- 崩掉的：每層低秩（lr:k16–40 不聯合）3–21%、合併 3–11%、丟 block（KEYDIFF／KNORM 0.875）13–21%，和隨機丟差不多。
- M2：1/16 預算兩個模型都顯著（組合 − 最好單一 +36.4／+37.6 個百分點，Holm 拒絕）；1/8 Qwen +7.1 [+1.3, +13.1]（Holm p 0.0081 對門檻 0.00625，差一點）；1/4 Llama 不顯著。
- **限制／注意**：ε 規則要 CI 下界 ≥ −2 個百分點，96 題時 CI 半寬約 3–4 個百分點，連 INT4 KIVI 都過不了 → Llama 的「沒有組合過 ε」主要是統計力不夠，不是組合變差；要更多題才能分辨。xKV 的基底每題用整段 KV 算，壓縮本身的 SVD 成本沒算。

## X3　壓縮組合的還原時間 × 連線 × GPU 忙（micro 10/20 21:22–21:4xZ；map 10/21 02:52Z）〔micro 實測＋map 模擬〕

- run：`20261020-212244-m16-x3-micro-llama`、`20261020-213232-m16-x3-micro-qwen`、map `20261021-025231-m16-x3-map`。產出 `results/m16_stage3_mi300x/x3_time/`（map_report.md、決策圖 SVG）。
- **判定（事先寫死）**：STOP-0 成立（允許第二階段驗證過的有損格式時，每一格最佳都是 INT4 一次拆，翻轉 0 格）；模式 L（只准無損）與 Q：STOP-2a（「GPU 忙」這個軸對選擇沒有價值，A 對 S 每格都 <5%）；模式 X（疊加可行集）：STOP-2b（A 對最好固定格式只好 2.2%）→ NO-GO；validate（N5）不跑。
- 無損在慢層（b32、b48、NFS）對不壓縮省 11–19%，在 SSD、CPU 層沒好處（和 S1 一致）。
- **E-L3**：低秩重建在 GPU 忙（duty 0.9）時變慢 2.79–2.80 倍（Llama），INT4 反量化只慢 1.34 倍；忙時重建 ÷ H2D 0.30–0.37 → 判「新成本項」。Qwen 2.64–2.78 倍 vs INT4 2.05 倍（中間）。
- **缺口**：X2 最好的組合（xKV 式、K／V 不同秩＋INT4 係數）在 X3 沒有對應的解碼段（x2_unmapped.csv），所以模式 X 的 NO-GO **沒有涵蓋最好的疊加候選**（Qwen `xl:k28:v42+q:int4_kivi` 可行但 unmapped）。要補一個解碼段再跑一次 micro＋map 才算完整。
