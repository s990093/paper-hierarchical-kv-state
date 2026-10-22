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

## 10/21 加深（使用者選了四條）：程式與排程

- 使用者 10/21 03:0xZ 選了：補 X3 缺口、C1b 選擇性還原、xKV 實用性（事先校準基底＋壓縮成本）、X2 加題數；之後又要求「把多層低秩＋量化推到極限」（X4，設計見 `x4_limit_plan.md`，看到數字前寫死）。
- 三個 agent 寫程式（都沒開 GPU）：X3 補缺口（`m16_xtime.py` 新 7 段＋寫入成本＋map v2；self-test 97／97）、C1b（`m16_c1b_restore.py`；self-test 46／46）、X2b＋xKV 校準（`m16_xstack.py` 新增功能、兩卡工作池 `x_pool.sh`；self-test 147／147，順手修了 self-test 的 `ok &= f(...)` 吞掉子函式失敗的 bug）。
- 決定：X2b 多加 3 個單一方法（兩個模型的 `q:int2_kivi`、Qwen 的 `xl:k22:v34`），避免合併後的池子缺口讓 r 偏小；xKV 校準照「同一行程配對」不改規則（不省那 2.6 卡小時）；`37zz` 會用新的 X2→X3 對應重寫第一版 `x2_feasible.csv`／`x2_unmapped.csv`（7 個 xKV 組合從 unmapped 移到 feasible），這是 37zz 本來就要做的「完整資料取代初版」，記在這裡。
- 新的品質 job 改成兩卡共用工作池（使用者 10/21 問「為何都只用一張卡」：量時間的 job 刻意只用卡 1；量品質的 job 原本「卡 0 Llama、卡 1 Qwen」，先做完的那張卡就閒到 job 結束）。
- 佇列順序：37d（跑中）→ 37e、37f（128K）→ 38a（X3 補缺口）→ 38b1–4（X2b）→ 38c1–2（xKV 校準）→ 38da、38db（C1b）→ X4（寫程式中）→ 39v → 40_v0、50_f2、51/52 F1。

## X2　128K 切片與最終報告（37e／37f 04:16–07:55Z；37zz 07:55Z）〔實測；128K 只探索、不進判定〕

- 128K 用第二階段 44c 的題組（每任務 12 題）。**實際題數（10/22 更正，報告產生器核對 x2_report.json 時抓到）：Llama 每個配置只有 18 題（37e／37f 的時間預算用完）、Qwen 48 題**；原本寫「每個模型 48 題」是錯的。E0 答對率：Llama 35.3%、Qwen 56.6%（128K 本身就難）。Llama 的 128K 數字只有 18 題、CI 很寬，只能當方向。
- Llama：`xl:k22:v34+q:int4_kivi`（位元組比 0.075，約 13 倍）+0.8 [−5.6, +6.3]，過 MLPerf 99%；`xl:k22:v34+q:int8_token`（約 9 倍）−0.2 [−3.2, +3.4]；`xl:k11:v17+q:int4_kivi`（約 27 倍）−6.9 [−21.0, +5.0]；`xl:k5:v9` 0%（崩）。
- Qwen：`xl:k28:v42+q:int4_kivi`（0.090，約 11 倍）−0.5 [−4.0, +2.7]，過 MLPerf 99%；`xl:k11:v17+q:int4_kivi`（約 28 倍）−2.7 [−6.9, +1.4]；只降秩 `xl:k11:v17`（約 9 倍）−18.1（128K 比 32K 掉更多），`xl:k5:v9` 0.4%（崩）。
- 判讀：「多層低秩＋INT4」在 128K 仍在約 11–13 倍時撐住（MLPerf 99%）；同樣倍數下只靠降秩在 128K 掉得比 32K 多，疊加的好處在長上下文更明顯。128K 的位元組比比 32K 小一點（每題基底攤在更多 token 上）。題數少、CI 寬，只當探索。
- 最終版 M1（32K）不變：Qwen r＝0.33（GO）、Llama r＝∞（統計力不夠，X2b 在補）。
- **失敗 3**（10/21 08:27Z）：`38b1_x2b_gen` 開跑 1 秒 rc=1：`x_ext.sh: line 49: m: unbound variable`。原因：`local m=$1 s0 out=…${m}…` 同一個 local 裡先設再用，bash 先展開（set -u）。同樣的寫法在第 85 行（xlcal 冒煙）也有。修法：分成兩行（全部 job 檔用程式掃過，只有這兩處）。plan 那步已 exit 0、沒有其他副作用。08:3xZ 重開 38b1。

## X3 補缺口（38a，07:55–08:27Z，CLEAN）〔實測〕

- run：`20261021-075914-m16-x3-micro2-llama`、`20261021-080256-m16-x3-micro2-qwen`（新 7 個 xKV 解碼段，reps 5）、寫入成本 `20261021-080628-m16-x3-wcost-llama`、`20261021-082103-m16-x3-wcost-qwen`。產出 `results/m16_stage3_mi300x/x3_time/v2/`、`v2/wcost/`。決策圖 v2（38z）等 X2b 的報告。
- **寫入（壓縮）成本**：xKV 每題的 `svd_lowrank`（W＝4、niter 4、多取 16），區域 T＝16,384 token、全部窗口（Llama 8、Qwen 7），GPU 閒：Llama `k11v17` K 4.0 s＋V 6.8 s＝10.8 s、`k22v34` 24.6 s、`k28v42` 31.3 s；Qwen `k11v17` 3.9 s、`k22v34` 9.6 s、`k28v42` 12.7 s。GPU 忙（duty 0.9）：Llama `k11v17` 33.1 s（約 3 倍）、Qwen `k11v17` 14.8 s。係數量化本身只要 1–4 ms。
- 判讀：每題算 SVD 的壓縮成本是**數十秒**，比同長度的 prefill 本身還久很多 → 每題 SVD 的 xKV 在寫入路徑上不實用，除非（a）事先校準的基底可用（38c 在測），或（b）換更快的作法（例如 Gram 矩陣＋eigh；目前 `svd_lowrank` 在 ROCm 上可能被 QR 拖慢，〔推論，未驗〕）。

## X2b　加題數（38b1–38b4，08:29–12:31Z；report 12:31Z）〔實測〕

- run：`20261021-0829…-m16-x2b-*`（工作池）、report `20261021-123141-m16-x2b-report`。產出 `results/m16_stage3_mi300x/x_stack/v2/`（tables_x2b.md、x2b_report.json、x2_feasible.csv、x2_unmapped.csv）。X2b 每個模型 192 題（seed 63000 起），和 X2 的 96 題合併成 288 題（變體 D，主要讀）。
- **Qwen（D，288 題）**：`xl:k22:v34+q:int4_kivi`（約 12 倍）+0.3 [−1.9, +2.5] **過 ε**；`xl:k28:v42+q:int4_kivi`（約 10 倍）+0.5 [−1.8, +2.7] 過 ε；`xl:k11:v17+q:int4_kivi`（約 24 倍）−4.5 [−7.6, −1.6]（顯著變差）；`q:int4_kivi`（3.2 倍）−0.3 過；`q:int2_kivi` −6.9。M1：r＝0.082÷0.233＝0.35 → GO；M2 在 1/8 預算：組合 − 最好單一 +4.0 [+1.2, +6.9]，Holm 拒絕（顯著）。
- **Llama（D，288 題）**：`xl:k28:v42+q:int4_kivi`（約 8 倍）−1.2 [−2.8, +0.2]、`xl:k22:v34+q:int4_kivi`（約 10 倍）−1.4 [−3.6, +0.7]、`xl:k11:v17+q:int4_kivi`（約 21 倍）−2.2 [−4.2, −0.3]（顯著但小）；`xl:k11:v17`（8 倍）−0.6 [−2.5, +1.2]、`xl:k22:v34+q:int8_token`（7.3 倍）+0.0 [−2.0, +2.0]；`q:int4_kivi` −0.2 過 ε、K4V2（4 倍）+0.5 過 ε。M1：沒有組合的 CI 下界 ≥ −2 → r＝∞；MLPerf 99%：r＝1.12；MLPerf 99.9%：r＝0.55。
- 判讀：題數加到 288 後 CI 半寬約 2 個百分點。Qwen 的疊加約 12 倍在 ε 內，確定贏過最好的單一方法；Llama 的疊加在 8–10 倍時約 −1.2～−1.4 個百分點，嚴格的 ε 規則仍判不過（CI 下界剛好掉到 −2.8～−3.6），21 倍時顯著掉 2.2 個百分點。

## X3 決策圖 v2（38z，12:31Z）〔模擬：零件實測〕

- run：`20261021-123143-m16-x3-map2`，輸出 `results/m16_stage3_mi300x/x3_time/v2/`（map_report.md、map_decision.csv、決策圖 SVG）。用了新 xKV 解碼段與 X2b 的可行集。
- 判定：模式 Q STOP-0 照舊成立；L、Q 的 STOP-2a 照舊（GPU 忙碌軸沒價值）；**模式 X：STOP-2c（A 對最好固定格式 +6.5%，介於 5–15% → 照規則停下來問使用者）**；A 對「只看連線」的 S 只 +0.9%；T-d 成立 → map 判 validate 要跑（8 格）。
- 還原時間（模擬，64 block，秒）：Qwen 在慢層，`xkv_k22v34_int4` 對單用 INT4：NFS 0.3 vs 0.9（BF16 1.6）、b32 0.2 vs 0.5、本地 SSD（閒）0.04 vs 0.1 → 快 61–69%；CPU pinned 只差 0–12%、vLLM CPU 路徑 0%（被逐片 copy 綁住）；GPU 忙時在 CPU 層 INT4 反而最好（低秩重建忙時慢 2.8 倍）。Llama 的可行集沒有 xKV 組合（沒過 ε），最佳是 K4V2，對 INT4 快 8–20%。不含寫入（SVD）成本。

## xKV 事先校準的基底 vs 每題 SVD（38c1–38c2，12:31–14:35Z；report 14:36Z）〔實測〕

- run：calib `20261021-123131-m16-xlcal-calib-{llama,qwen}`（seed 61000 的 8 題、258,048 token）、TF／生成 `20261021-123858…141255-m16-xlcal-*`；產出 `results/m16_stage3_mi300x/x_stack/v2/{tables_xlcal.md,xlcal_report.json,calib_xl_*.json}`。和 X2 逐值重現 384/384、TF 錨點 48/48。
- **判定（事先寫死）**：兩個模型都 **UNDETERMINED**（96 題分辨不了）。
- Llama（生成，校準 − 每題 SVD）：`xl:k22:v34+int4` +2.6 [−1.6, +7.5]、`xl:k28:v42+int4` +3.5 [−0.6, +8.2]（這兩對 CAN_PRECOMPUTE）；只降秩 `xl:k11:v17` −3.7 [−7.6, +0.2]。校準版 `xl:k28:v42:cal+q:int4_kivi`（位元組比 0.085，約 11.7 倍；基底另算 147 MB 共用）對 E0 +1.3 [−1.7, +4.7]，過 ε。
- Qwen：`xl:k22:v34+int4` −2.1 [−6.0, +2.0]、`xl:k28:v42+int4` −2.2 [−5.5, +1.1]、只降秩 −5.2 [−10.3, +0.3]（都 UNDETERMINED）；TF 上 INT4 組合兩者沒差（Δgold_lp 差 +0.37、+0.06，CI 跨 0）。
- 判讀：加了 INT4 係數量化之後，事先校準的共用基底大致和每題 SVD 一樣好（Llama 點估計還更好、Qwen 約 −2 個百分點不顯著）；只降秩時校準基底較差。事先算好基底就不用每題做數十秒的 SVD（壓縮只剩一次投影），每請求位元組也更小（基底共用）。統計上還不能下結論。

## 10/21 14:5xZ：xKV 校準基底加題數（38k*）排進佇列

- 使用者：「要正式證明事先校準不輸每題 SVD，需要更多題目 → 好繼續實驗」。只比有 INT4 的兩對（`xl:k22:v34[:cal]+q:int4_kivi`、`xl:k28:v42[:cal]+q:int4_kivi`）＋E0，新題 seed 66000 起、每個模型 192 題；校準基底沿用 38c（sha256 核對）。程式 `code/m16_xlcal3.py`（agent；self-test 10/10），排在 X4 與 38j 之後。
- **主要判定改成「只用新的 192 題」**（在看到任何新資料之前改定）：「只比兩對」是看過 38c 結果之後才決定的；光這個限定就會讓 38c 的 96 題下 Llama 從 UNDETERMINED 變成 CAN_PRECOMPUTE（差別全在拿掉的只降秩那對）。所以只有新的 192 題是乾淨的獨立檢驗；合併 288 題只當參考並列。判定門檻照 XLCAL_TEXT 原文，aggregate 從三對改兩對（寫在報告裡）。
- **失敗 4**（10/21 14:53Z）：`38da_c1b` 工作池領 llama shard 3 時失敗：`m16_c1a_predict.read_rows` 讀到別的 worker 正在寫的 `.csv.gz`（EOFError，只是跳過檢查用）時把警告印到 stdout，job 用 `u=$(pool-claim …)` 把它當成領到的工作名 → 指令錯（`env: '…csv.gz:': No such file`）、`pool-done` 解析失敗，工作池立 STOP。已在跑的 5 個單位（llama s0–s2、qwen s0–s2 中做完的）正常完成、結果有效。修法：`cmd_pool_claim` 領取期間 `redirect_stdout(sys.stderr)`（原子替換檔案；self-test 46/46）。15:10Z 重排 38da（逐題續跑）。
- 10/21 15:12Z NEEDS_HUMAN（資訊）：38da 重開時 qtime qwen（run `20261021-151145-m16-c1b-qtime-qwen`）CONTAMINATED_DURING_RUN → 照規則沒寫進 results。不需重量：第一次開跑時兩個模型的 qtime 都是 CLEAN（`20261021-143835-…-qtime-llama`、`20261021-143911-…-qtime-qwen`），`results/…/c1_selective_restore/qtime_qwen.csv` 是那次的；llama 重開那次也 CLEAN。

## C1b　選擇性還原（38da 14:35–15:07Z 失敗後重開 15:08–16:33Z；38db 立即結束；report 16:33Z）〔實測〕

- run：24 個工作池單位（`c1b_summary.json`）；題目同 C1a（每個模型 T2–T4 135 題＋T5 15 題），逐題檢查沒有 false；FULL（同一條遮罩路徑）和 E0_SDPA 一致（+0.0）。產出 `results/m16_stage3_mi300x/c1_selective_restore/`；逐列原始檔在 `/mlsteam/data/tiara/stage3/c1b/`。
- **判定（事先寫死）：兩個模型、設計卡（QUEST_MEAN）與可部署版（QUEST_BLIND）都是 STOP**——k＝50% 時兩種 query-aware 選法補不補載都掉超過 ε。
- 主要數字（第 2 輪答對率對 FULL 的差，百分點，T2–T4）：Llama QUEST_MEAN@25 −27.7、@50 −7.9；QUEST_BLIND@50 −7.6；Qwen QUEST_MEAN@25 −18.5、@50 −9.6；QUEST_BLIND@50 −12.4。寫入時分數（WRITE）、第 1 輪問題（QRY1）、位置（POS）、隨機（RND）都更差（@50：Llama −14～−36）。補載 (b) 比不補載 (a) 好 +1～+14 個百分點（多數顯著），但補不回 ε。
- **上界（FULLK_MEAN，用完整 K 打分，不可部署）**：Qwen @25 +0.5 [−2.5, +3.4]、@50 +1.3；Llama @25 −6.9、@50 −1.6 [−4.0, +0.2]。→ 「只搬需要的 block」本身有空間（Qwen 搬 1/4 就不掉），失敗的是 Quest 的 512-token block min/max 摘要不夠準。
- 判讀：照設計卡寫成負面結果：「在這些任務上，以 Quest 摘要做的 block 級選擇性還原不可靠」。C1a 的 recall@16 高於隨機，但答案 block 只要漏一塊答案就錯，召回 30–70% 不夠。可能的下一步（未測、只是方向）：用更準又便宜的 K 摘要（例如 RoPE 前的低秩 K，K1／X 系列顯示 K 在 RoPE 前很低秩）當選擇索引，逼近 FULLK 上界。

## X4　把「xKV＋量化」推到極限（38e1–38h2，10/21 16:33Z–23:29Z；report 23:29Z）〔實測〕

- 60 個 run（`tables_x4.md`、`x4_report.json`、`x4_best_allocation.csv`、`lossless_x4.json`、Pareto 圖），輸出 `results/m16_stage3_mi300x/x_stack/x4/`。X4a TF：每個模型 96 題 × 71 個配置（seed 64000）；X4b 生成：192 題（seed 65000，**和 X2／X2b 都不重疊，是獨立樣本**）；X4c 128K：48 題；X4d 兩輪：96 題（C1a 題組）。
- **目標倍數 → 最佳配比（X4b 生成，對 E0，百分點）**：
  - Llama：8× `xl:k11:v17` −2.0 [−5.1, +1.0]；12× `xl:k19:v29+int4` −3.6 [−6.9, −0.5]；16× `xl:k16:v24+drop:KNORM:0.125+int4` −4.1；20× `xl:k12:v18+int4` −2.6 [−6.0, +0.6]；24× −4.3；32× `xl:k9:v14+K4V2` −6.8。
  - Qwen：8× `xl:k34:v51+int4` −0.7 [−3.5, +2.0]（過 MLPerf 99%）；12× `xl:k23:v35+int4` −2.6 [−6.0, +0.7]；16× `xl:k17:v26+int4` −3.6；20× −3.2；24× `xl:k14:v21+K4V2` −4.6；32× `xl:k11:v17+K4V2` −7.8。
- **極限（事先寫死的 ε：CI 下界 ≥ −2）**：32K 兩個模型都只有 INT4（3.2 倍）過 ε；MLPerf 99% 下 Qwen 8 倍。128K：Llama `xl:k19:v29+int4`（15.5 倍）+2.8 [−0.9, +8.1] 撐住、Qwen 9–18 倍 −1.1～−2.4（CI 下界超過 −2，判「掉」）。
- **和 X2b 的落差（照實）**：X2b（288 題）Qwen `xl:k22:v34+int4`（12 倍）+0.3 [−1.9, +2.5]；X4b（獨立 192 題）相近配置 `xl:k23:v35+int4`（11.7 倍）−2.6 [−6.0, +0.7]。兩個樣本的點估計差約 3 個百分點，都在彼此 CI 內；合起來看 10–12 倍的真實影響大約是 −1～−3 個百分點，**不是「零損失」**。之前「Qwen 約 12 倍過 ε」的說法要改成「約 8 倍以內接近無損（MLPerf 99%），12 倍約掉 1–3 個百分點」。
- **Q-A 哪樣配（TF，帶內配對）**：量化係數 vs 只降秩：每個倍數都大幅值得（Qwen +2.3～+47 gold_lp；Llama 12 倍起 +2.4～+37）；INT4 係數 vs INT8：INT4 較好（多留秩、少給位元）；低於 4 bit：8–20 倍時中性或略差，24–32 倍時 INT2／K4V2 值得；W＝4 優於 W＝2（高倍數差很多）；輕度丟：Qwen 不值得、Llama 混合；K 少 V 多：大致中性。無損（實際係數位元組 zstd:3）：INT4 係數組合再多 1.10–1.20 倍，BF16 係數 1.40 倍 → 值得疊（只看位元組，速度由 X3 決定）。
- **Q-B 配比怎麼變**：8–20 倍：降秩（W＝4、V:K≈1.5）＋INT4 係數；24–32 倍：降秩＋K4V2（K INT4、V INT2）；輕度丟只在 Llama 16／24 倍出現（差異不顯著）；再乘無損約 ×1.1–1.2。
- **疊加優於單降秩（生成）**：Llama 12 倍 +7.8 [+2.5, +13.0]（成立）；Qwen 16 倍 +25.4 [+20.2, +30.8]、8 倍 +4.2 [+0.5, +7.8]（成立）。
- **多輪**：第 2 輪對 E0：xKV＋INT4 Qwen 8 倍 +1.3、12 倍 +0.9，Llama 12 倍 −1.1 [−4.0, +1.2]；丟一半 block 的對照 −17～−19。→ 疊加壓縮在多輪幾乎不傷（判定標「壓縮傷多輪」是因 CI 下界超過 −2，點估計接近 0），遠好於丟 token。
- **256K（38i1）**：失敗——T5（CWE）在 256K 出題時「cwe 詞表不夠長，補不滿」（`m10_tasks._cwe`），Qwen shard 1 rc=1；shard 0 跑完。報告寫「沒有資料」。GPU runner 因此在 00:46Z 停下，01:18Z 我把 38i1 移出佇列、重啟（GPU 空了約 32 分鐘）。

## X3 validate v2（38j，10/22 01:18–01:31Z，CLEAN）〔實測：端到端還原時間〕

- run：`20261022-011846-m16-x3-validate2`（8 格、25 列、0 列污染）；產出 `results/m16_stage3_mi300x/x3_time/v2/validate/`。
- **判定（事先寫死，STOP-3：≥2 格實測和預測差 >10%，或實測 A 對 F 幾何平均 <5% → 停）：STOP-3 成立**——預測普遍低估 5–26%（多格 >10%），所以「模擬決策圖」不可信到可以直接用。
- 但實測本身（秒，64 block）：Qwen 本地 SSD、GPU 忙 0.9：`xkv_k22v34`（BF16 係數）0.065、`xkv_k22v34_int4` 0.088、BF16 原樣 0.224（快 2.5–3.4 倍）；Qwen CPU pinned、GPU 閒：`xkv_k22v34` 0.015、INT4 0.018、BF16 0.039；**Qwen CPU pinned、GPU 忙 0.9：INT4 0.020、`xkv_k22v34_int4` 0.083、BF16 0.060**——GPU 忙時低秩重建慢到比不壓縮還慢；duty 0.4 時 xKV＋INT4 0.021 vs INT4 0.019、0.7 時 0.041 vs 0.019。Llama CPU pinned 閒：K4V2 0.032、INT4 0.033、BF16 0.084。實測的「依情況選」對「固定最好的一種」（A 對 F，4 對）幾何平均 +27.6%。
- 判讀：照規則，模擬圖停用；但實測顯示「最好的壓縮格式會隨 GPU 忙碌翻轉」在快的下層真的發生（閒時 xKV 最快、忙時 INT4 最快、xKV＋INT4 甚至比不壓縮慢），和 E-L3（低秩重建忙時慢 2.8 倍）一致。這是 B（依連線＋GPU 忙選格式）在「疊加組合」下才出現的空間，只有 4 對實測，要另做實驗確認。

## xKV 校準基底加題數（38k1–38k2，10/22 01:34–04:04Z；report 04:04Z）〔實測〕

- run：`20261022-013407…-m16-xlcal3-*`（工作池）；report `results/m16_stage3_mi300x/x_stack/v3/tables_xlcal3.md`。校準基底 sha256 和 38c 相同。
- **主要判定（新的 192 題，獨立檢驗）：兩個模型都 UNDETERMINED**。Llama：`k22v34+int4` 校準 − 每題 +2.1 [+0.0, +4.4]（CAN）、`k28v42+int4` −1.5 [−4.3, +1.2]；Qwen：−1.3 [−4.1, +1.5]、−0.5 [−3.2, +2.3]。合併 288 題（參考）同樣 UNDETERMINED。
- 判讀：四對點估計 −1.5～+2.1，平均約 0，CI 半寬約 2.5–3 個百分點 → 事先校準的共用基底和每題 SVD 差在 ±2–4 個百分點內，**看不出系統性變差，但也證明不了「不輸」**。Llama 校準版 `xl:k22:v34:cal+q:int4_kivi`（約 14.6 倍、每請求位元組比 0.068）對 E0 −0.3 [−2.0, +1.4]（合併 288 題，過 ε）。

## V0　vLLM 0.28（ROCm）APC／CPU 卸載／分層能不能跑、命中 vs 不命中輸出是否相同

- **失敗 5**（10/22 04:09Z）：`40_v0` 的 V0-b（APC on）文字題 `v0check` FAIL（run `20261022-040931-m16-v0b-txt`，rc=3）：3 題裡 2 題 gpu_hit 的輸出在第 7、15 個 token 和 miss 分歧，**分歧位置 miss 的 top-1／top-2 logprob 差都是 0.0**（數值平手）；miss_cold、gpu_hit_cached、determinism（miss vs miss2）全 PASS；V0-a（APC off）與 V0-b 亂數題全 PASS。判讀：命中路徑（只算最後幾個 token）和完整 prefill 的加總順序不同，平手時 argmax 會翻；這是 `v0_f1_i3_howto.md` §2「不一致時怎麼讀」預料的情況，不是 KV 錯（真正的無損證據是 F1 的逐 block 位元組比對，第一階段已做）。
- 修法：`40_v0.sh` 加 `tie_ok`：只在「非 match 的檢查全 PASS、而且每個分歧位置 margin ≤ 1e-3」時記 NEEDS_HUMAN（資訊）並當成通過，其他情況照樣停。用這次的真實 json 測：判「只是數值平手」；把一個 margin 改成 0.5 的反向對照判「不是」。04:1xZ 重跑 V0。
- V0 重跑（10/22 04:11–04:28Z）完成：a（APC off）、b（APC on）、c（APC＋CPU 卸載）、d（＋ROCM_ATTN patch）、e（fs 分層）都能跑；不一致的只有 b-txt、c-rnd、d-txt、e-rnd，**分歧位置 margin 全部 ≤ 1e-3（數值平手）**，照 tie_ok 當成通過、記成資訊 NEEDS_HUMAN。

## F2　共用前綴 burst（重現 vLLM #44294；10/22 04:28–05:46Z，全部 CLEAN）〔實測〕

- run：`20261022-0430…-m16-f2-*`（A＝APC、ROCM_ATTN；A′(Ap)＝APC、TRITON_ATTN；B＝APC＋CPU 卸載（connector 會強制 TRITON_ATTN）；C＝不復用；每格 3 次、evict＝reset）。產出 `results/m16_stage3_mi300x/f2_shared_prefix_burst/`（summary_*.json、requests_*.csv）。
- burst 的 makespan（秒，3 次中位數；P＝前綴長、N＝同時請求數）：P＝10K、N＝64：A 2.20、A′ 4.41、B 4.42、C 49.2；P＝32K、N＝64：A 5.44、A′ 14.50、B 11.53、C 382；P＝32K、N＝32：A 4.04、A′ 12.19、B 8.36。B 的 connector log：N＝64 時 63 個請求被「chunks being loaded」延後過。
- 判讀：**#44294 的「開卸載慢一倍」在 MI300X／vLLM 0.28 沒有重現**——和同後端的 A′ 比，B 在 10K 時打平、32K 時更快（延後邏輯有觸發，但淨效果不負）。B 對 A 慢約 2 倍，**全部來自 connector 強制改用 TRITON_ATTN**（A′ 對 A 一樣慢約 2 倍）。→ ROCm 上開 CPU 卸載的真正代價是注意力後端的退回，不是延後邏輯；這也支持候選 3（AMD 上的還原路徑）。

## F1　前綴復用地圖（51_*、52_*，10/22 05:46–13:24Z，全部 CLEAN）〔實測〕

- 18 個 job、54 格 × 3 種設定（off＝不復用、apc、apccpu＝APC＋CPU 卸載；全部 TRITON_ATTN，事先寫死以免後端混雜），L＝8K／32K × 容量 ample／c075／c040 × 重疊 0／0.5／1.0 × 到達 burst／Poisson 0.5λ_sat／0.9λ_sat。校準：8K t1＝0.455 s、λ_sat＝1.97 req/s；32K t1＝4.73 s、λ_sat＝0.197 req/s（`calib_*.json`）。報告 `code/m16_f1_report.py`（run `20261022-132645-m16-f1-report`）→ `results/m16_stage3_mi300x/f1_prefix_reuse/f1_report.{md,json}`。
- **判定（照事先條件的字面）：SUCCESS**——54 格裡 18 格 APC 收益 <5%；**但這 18 格全是重疊 0%（沒有東西可復用），屬於當然結果**。重疊 50%：APC 讓平均 TTFT 少 20–62%；重疊 100%：少 76–93%；容量（ample → c040）與到達率幾乎不改變這個結論。→ 照實寫：APC 的收益只由重疊決定、單調，**F1 原本想找的「非當然的不划算區域」沒有找到**（接近停止條件的精神）。
- **CPU 卸載對 APC 的邊界（有意思的部分）**：8K 時 CPU 卸載幾乎都讓 TTFT 變差（−1%～−112%，容量緊＋高負載時最差：c040、重疊 0、0.9λ 平均 −112%、P95 −136%）；32K 時幾乎都變好（+3%～+15%）。後端已固定為 TRITON_ATTN，所以這是卸載本身的成本（寫入 CPU、排程）對上省下的重算：**提示越短，卸載越不划算，而且在容量緊、負載高時會反過來傷害**。
- 三種指標（平均、P95、goodput@1.5×t1）選出不同最佳設定的格子 6 個，都在 8K、重疊 0 這類差異很小的格子。

## 收尾（10/22 13:30Z 起）

- 交付文件：`results_analysis.md`（各實驗解讀，報告 PDF 的文字版）、`stage3_candidates.md`（候選排名）；`reproduction.md` §2.2 補齊全部 job；`results/RUNLOG_MI300X.md` 加 Milestone 16 指標。commit `89dff4b`。
- 最終 PDF：run `20261022-133248-m16-report-pdf`（git 89dff4b），77 頁、2.71 MB，驗收 ok（字型只有 Noto Sans CJK TC、頁尾齊、目錄 48 條、NOT_MEASURED 圖表 0）。字型缺字 3 個照替換表處理（U+2717→×、U+FE0F 去掉），來源是內嵌的 `experiment_matrix.md`／`research_map.md`。
- 備份（只在資料盤，不進 git）：
  - `/mlsteam/data/tiara/backup_stage3_mi300x_20261022.tar.gz`（1.09 GB，10,888 個檔）＝`runs/*m16*`＋`stage3/`（排除 `stage3/smoke/kv_*`，可重生）＋`results/m16_stage3_mi300x/`；`gzip -t` 通過。打包在最終 PDF 之前，所以另包
  - `/mlsteam/data/tiara/backup_stage3_mi300x_20261022_final_report.tar.gz`（8.5 MB）＝`stage3/report/`＋最終 PDF 的 run 目錄。
  - 校驗碼：`/mlsteam/data/tiara/backup_stage3_mi300x_20261022.sha256`。
  - 沒備份：`/var/tmp/tiara_s3/kv_dumps/`（本地盤，`05_s1_kvdump.sh` 可重生）。

## K1b　Tucker（多維分解）與「跨層共用的到底是什麼」（10/22 加；只用 CPU）

- **為什麼**（使用者 10/22）：「4.1 K1 多一個維度壓縮（Tucker）？」「相鄰層同一 token 的 KV 完全不像，這樣不能跨層？」「TD-MoE……有機會結合？」
- **查新**（10/22，WebSearch／WebFetch）：查不到叫 TuckerMoE 的論文；最接近的是 TD-MoE（Xu et al., ICLR 2026）：把一層所有專家的**權重**疊成 3 維張量做 Tucker，摘要寫「data-aware」、多線性白化、3 維秩分配；摘要沒寫是否需要訓練（NOT_VERIFIED）。Tucker Attention（arXiv 2603.30033）、TPA 分解的是 attention **權重／架構**；arXiv 2606.03465（Zagitov et al.）摘要：張量分解假設的「共用子空間」和 LLM 學到的異質表示不合。**沒查到**把 Tucker 用在 KV 活化值（層×head×token×維度）上、不訓練的作法（沒查到≠沒有）。query 白化的閉式低秩 KV 已有 KQ-SVD（L5a 第 20 筆）。
- **設計**：S1 的 dump（Llama／Qwen × T4／T2，32K token，全部層；K 用 m16_kvstruct 的 RoPE 逆轉回 RoPE 前，要先過 rope_check）。不去平均（照 m16_xstack 的 xKV）。位元組一律同精度（只比結構，量化另外乘）。指標＝相對 Frobenius 誤差（X1：和答對率只中度相關，只能篩選）。程式 `code/m16_k1_tucker.py`。
- **事先寫死（看數字前）**：
  - Q1 跨層：相鄰層「同 token cosine」 vs 「token 子空間重疊」（每層前 64 個左奇異向量；重疊＝‖U_lᵀU_{l+1}‖²_F／64；隨機≈64／32768）與 CKA。**去平均版**為準（不去平均時，各層共同的平均方向會讓重疊虛高）。判讀：去平均重疊（層平均）≥0.5 且不去平均 cosine（K1 報的那個）≤0.1 → 「跨層共用的是 token 子空間、不是向量」；<0.2 → 不能這樣解釋。
  - Q2 Tucker 多一個維度：在 xKV 每 4 層（X 系列用的）操作點的同樣位元組下（K：每 head 每層等效秩 8／11／16／22／28／32／42；V：8／17／22／34／42／48／64），比 4 維 Tucker（token×層×head×維度，HOSVD 上界＋任何 Tucker 的下界）、3 維 Tucker（token×層×(head·維度)）、兩側分解（token 因子跨 G 層共用＋每層自己的通道基底，G＝4、8〔Qwen 7〕）、xKV 不同分組 G。**GO（值得做 GPU 答對率）**：兩個模型的 K 都至少有一個操作點，Tucker 上界或兩側分解的誤差 ≤ 0.9 × xKV-G4（兩題平均）。否則 STOP。Tucker 下界也 > xKV-G4 → 「連最樂觀的 Tucker 都輸」。

## 10/22 15:3xZ 起：補跑（使用者選：X3 翻轉實測格子、X4a 每 8／7 層一組、Qwen 256K、38k 同位元組重新分析、I3g 全跑含 NFS、K1b Llama 診斷後重跑）

- **K1b 失敗 5**（10/22 14:12Z）：Llama 兩份（`20261022-135022-m16-k1b-tucker-llama_{T4,T2}_64`）都 Segmentation fault（exit 139），算完 G＝16、在 G＝32（全部層一組）時崩。Qwen 兩份正常完成。診斷（faulthandler，scratchpad 的 k1b_diag.py）：崩在 `Xg.T @ Xg`（32768×32768 的 float64 矩陣乘，OpenBLAS），不是特徵值分解。修法：分塊算 Gram（每塊 4096 欄），再重跑兩份 Llama；判定條件不變。（我的進度監看 14:22Z 到期，晚約 1 小時才發現。）
- **Qwen 256K（61_x4c_256k_qwen_noT5）**：38i1 失敗的原因是 T5（cwe）在 256K 補不滿（`m10_tasks._cwe`：上游固定詞表）。**256K 只跑 T2–T4**，不改第二階段的 `m10_tasks.py`（硬補等於造新版 T5，和 32K／128K 的 T5 不能比）。38i1 已做完 s0 的 T2 6 題＋T4 6 題、s1 的 T3 1 題；續跑只剩 T3 5 題。計畫檔、題目、配置不變；判讀照 X4c（對 E0 的差與 CI，題數少照實寫），報告由 61z 重出（x4_report）。
- **X4a W 大（每 8／7 層一組）**：當初為了時間從 X4a 縮掉（x_stack_howto §15.6）。現在補：同一批 X4a 的 96 題（seed 64000 起），T∈{20, 24, 32}，INT4 與 K4V2 兩種係數位元，W＝8（Llama）／7（Qwen）的配置照 m16_x4 的 `wbig` 規則（K 秩上限 16、倍數落在 T±10%），同一個行程裡也跑同一 T、同係數的 W＝4 配置當配對。**事先寫死**：逐題配對差（W 大 − W4 的 Δgold_lp，bootstrap 95% CI）；某模型在至少一個 T 的 CI 下界 > 0 → 「W 大較好」；所有 T 的 CI 上界 < 0 → 「W 大較差」；其他 → 「分不出」。TF 只是篩選（照 X1 的規矩）。
- **38k 同位元組重新分析（只用 CPU）**：**事後分析**（看過 38k 結果之後才提的），只當探索，不當判定。比「共用基底的高秩」對「每題 SVD 的低秩」：每請求位元組 cal k28＋INT4（0.085）< 每題 SVD k22＋INT4（0.096），用 38k 新 192 題逐題配對。
- X3 翻轉格子、I3g 的設計與判定：讀完程式後另寫在下面，排進佇列之前寫。
- **X3 翻轉格子（63_x3_flipgrid）設計與事先寫死的判定**（排進佇列前寫）：38j 只有 4 對實測支撐「最快的還原格式隨 GPU 忙碌翻轉」。補一個完整格子：2 模型 × 3 層（cpu_pinned＝真實 pinned H2D、b16、ssd_meas〔限速器〕）× 4 個 GPU 忙碌度（duty 0／0.4／0.7／0.9）× 6 個動作（raw＝BF16、int4_kivi@oneshot、k4v2@perblock、xkv_k22v34@oneshot、xkv_k22v34_int4@oneshot、xkv_k11v17_int4@oneshot）＝144 個端到端量測。量法＝`m16_xtime.py validate`（和 38j 同一條路徑；格子由 `code/m16_x3grid.py build` 產生，預測值取 map v2）。NFS 不放（32K Llama 原樣還原一次要十幾秒，而且慢層上壓縮一定贏，不是翻轉要看的地方）。**判定**：每條（模型, 層），duty 0 與 0.9 的最快動作不同、而且兩端各自最快的都比對方的選擇快 ≥5% → 翻轉；兩個模型都至少一條翻轉 → FLIP_REPRODUCED；只有一個 → FLIP_PARTIAL；都沒有 → NO_FLIP。另報「依情況選」對「固定最好的一種」的幾何平均好處。污染列不算。
- **I3g（64_i3g）設計與事先寫死的判定**（排進佇列前寫；使用者選「全跑（含 NFS）」）：驗 `i3_pause_sim.md` §7 的「翻轉」前提，不是驗 κ。Llama、卡 1、vLLM 0.28、四個設定都用 TRITON_ATTN：a＝APC、b＝APC＋CPU、c＝APC＋CPU＋檔案層 `/var/tmp/m16_i3g_fs`（本地）、d＝同 c 但檔案層在 `/mlsteam/data/tiara/stage3/i3g_fs`（NFS）。工作負載 `m16_workload.py gen-agentic`：64 個 session、6 輪、起始 8K、工具輸出長度＝Mooncake toolagent 推得的經驗分布、暫停 lognormal 中位 30 s σ＝1（模擬的 LN30）、session 開始時間 uniform 0–300 s（中段約 33 個同時活著、最多 47）、上限 24K；workload sha256 59f397fe…。容量：參考工作集 W_ref＝32 × 平均 session 最大長度（12,039）＝385K token，HBM＝0.25×＝6,020 個 16-token block；CPU 卸載 33 GiB（≈2.8×HBM）；檔案層 150 GiB（只用來檢查空間；NFS 最多寫約 100 GiB，跑完照實記錄寫了多少再刪掉自己的快取檔）。模擬預測（回來那一輪 TTFT P95）：a 4.33、b 3.00、c 0.97、d 13.39 s。**判定**：RANK＝c 最低且 d 最高 → SIM_RANK_CONFIRMED，否則 SIM_RANK_REJECTED；FLIP＝P95(c) ≤ 0.95×P95(b) 且 P95(d) ≥ 1.05×P95(b) → FLIP_CONFIRMED，否則 FLIP_NOT_CONFIRMED；實測／預測的 ±30% 只當參考（模擬用 harness 的常數）。污染的 run 不算。報告 `code/m16_i3g_report.py`。
- **38k 同位元組重新分析（事後探索；run `20261022-154352-m16-xlcal-eqbytes`，只讀 38k 的逐題檔）**：新的 192 題，答對率差（百分點）：Llama「共用基底 k28＋INT4」對「每題 SVD k22＋INT4」＋0.4 [−1.7, +2.6]（每請求位元組 0.0854 vs 0.0961，共用基底少 11%）→ 下界過 −2；Qwen −0.1 [−2.8, +2.8]（0.0854 vs 0.0822，共用基底多 4%）→ 分不出。**只當探索**（看過 38k 才提），38k 的判定不變（UNDETERMINED）。產出 `results/m16_stage3_mi300x/x_stack/v3/xlcal_eqbytes.{md,json}`。
