# I3（C-I3）：agent 工具暫停期間 KV 去留的 κ 感知模擬（探索性，只用 CPU）

> **狀態**：探索性模擬，**不是實測**。只用 CPU（numpy＋標準庫，`nice -n 19 taskset -c 0-3`），沒有任何行程開 `/dev/kfd`（程式結尾有斷言）。
> **run_id**：`20261019-141254-m16-pausesim`（`/mlsteam/data/tiara/runs/20261019-141254-m16-pausesim/`：cmd.sh、context.txt、stdout.log、stderr.log、exit_code、scan.csv、decisions.csv）。
> 前一次 `20261019-141044-m16-pausesim` **失敗**（exit_code 1），見 §8。
> **程式**：`code/m16_pause_sim.py`（`--self-test` 74 項 ALL PASS）。**產出**：`results/m16_stage3_mi300x/i3_pause_sim/`。
> **審查**：self-review，**不是 cross-model review**（CLAUDE.md §5）。
> 設計依據：`lit/L3_reuse_serving_agentic.md` §5 缺口 1、§7 X3；`experiment_matrix.md` I3 卡；常數的真假之分照 `stage2_factcheck_and_toolbox.md`。

---

## 0. 先講結論

（全部是模擬算出來的；只有 MI300X 能跑動態模擬。主要指標＝**回來那一輪的 TTFT P95**；gain＝1 − 候選 ÷ 該格最好的現成規則，>0 表示候選較好。）

1. **事前凍結的候選 `kappa`：照 L3 的證偽條件，缺口不成立。** 主掃描 96 格中 **0 格好 ≥5%**（中位數 −2.8%，最好 +1.3%），42 格差 ≥5%；P95 JCT 0／96 格好 ≥5%。它輸給 vLLM 0.28 本來就有的 CPU＋SSD 串寫（`vllm_tier`），C=32、HBM 緊時 P95 是後者的 3 倍以上。
2. **事後加的 `kappa_tier`（κ 把關串寫）也沒有在主掃描贏**：96 格中 1 格 ≥5%、0 格 ≥10%（中位數 −1.6%，最好 +9.4%）；在一半以上的主掃描格子它完全不寫 SSD（`vllm_tier` 的 SSD 寫入量中位 235 GiB），P95 只差幾 %。只有 **CPU 層縮到 0.25×HBM**（`cpu_small`）時 32／32 格好 ≥10%（中位 +49%），但一部分來自「HBM 直接串流寫 SSD」這個 vLLM 0.28 沒有的能力（它的次層一定經過 CPU 主層，CPU 太小時 ≥64K 的 session 只能丟），不全是 κ 的功勞；而且是事後候選，不能拿來下判定。
3. **決策翻轉：有，但翻的是「哪個現成規則最好」，不是 κ 能贏它。** 把 SSD 級從本地 SSD（實測讀 8,054 MiB/s）換成 NFS（讀 392 MiB/s）：32 組配對中 **31 組最好的現成規則換人**（本地 SSD＝`vllm_tier`；NFS＝`tokencake`／`infercept`／`vllm_cpu`）。把本地 SSD 上最好的規則原封搬到 NFS，P95 中位 **+60%**（最多 +482%，29／32 組 ≥10%）；反方向中位 +26%。κ 把關的 `kappa_tier` 對每個設定的最佳規則只差中位 +3.1%（NFS）／+5.6%（本地 SSD）。**GPU 閒→外加忙：最好的規則幾乎不換（2／32），搬過去的遺憾中位 0%。** Llama↔Qwen：κ 的「回來時在哪裡」460 個帶中 2 個翻轉。
4. **真實 MI300X 的 HBM 在這個工作負載下根本沒壓力**：Mooncake toolagent、C=32 時，實測的 vLLM KV 容量是參考工作集的 4.2 倍（Llama）／9.5 倍（Qwen）；`real` 容量的格子 κ 對最好的現成規則中位 −1.8%（最差 −17.2%，來自已結束 session 的 KV 慢慢堆滿 HBM 之後的淘汰順序），`kappa_end` 中位 −0.1%。
5. **最大的可用空間不是 κ，而是「知道 session 已經結束」**：`kappa_end`（只多知道這一件事）在 NFS 設定中位 +26%、`cpu_small_nfs` 中位 +27%、主掃描最多 +22.6%。完美資訊的變體（`oracle*`）共用 κ 的結構、沒有 SSD 串寫，**不是下界**：C=32 緊時比 `vllm_tier` 差約 2 倍。
6. **靜態 κ 表（只看還原端）**：MI300X harness 路徑 CPU > SSD > 重算；換成 NFS 變 CPU > 重算 > NFS，**Qwen 在 ≥64K 又翻回 NFS > 重算**；MI300X 的 vLLM 引擎路徑與 3090（vLLM 路徑，HEAVY 污染）都是 CPU > 重算 > SSD。同一台 MI300X，**換軟體路徑排名就翻**。但動態模擬裡 NFS 的寫入時間在關鍵路徑上，κ 在 ≥64K 也選重算，所以這個依長度的翻轉沒有出現在動態結果。
7. **建議的 GPU 驗證點不是驗 κ 策略，而是驗「翻轉」這個前提**（§7）：vLLM 0.28 原生就能跑的 {APC、APC＋CPU、APC＋CPU＋檔案層（本地 /var/tmp）、同左（NFS）}，Llama、C=32、HBM 壓到參考工作集的 0.25 倍、Mooncake toolagent 重播；模擬預測本地 SSD 時串寫最好、NFS 時串寫最差（P95 0.97 vs 13.4 s）。

---

## 1. 問題與判準

- **問題**：agent 工具暫停（秒到分鐘）期間，KV 留在 GPU／卸到 CPU／卸到 SSD 級／丟掉重算；用實測的成本比 κ 做決定，能不能比現成規則好？決策會不會隨硬體翻轉？
- **判準（照 L3 §5 缺口 1，跑之前就寫在 L3）**：
  - 證偽：κ 感知比最好的現成規則 **P95 好不到 5%** → 缺口不成立（CLAUDE.md §1 規則 4：<5% 停、5–15% 問人）。
  - 值得 GPU 驗證：**≥10% 且有決策翻轉**。
  - L3 原文用的是 **P95 JCT**；本模擬的 JCT 含暫停本身（幾分鐘到幾十分鐘），KV 決策只影響其中幾秒，所以另外報 **回來那一輪的 TTFT P95（主要指標）**、全部輪的 TTFT P95、平均。兩種都照 5%／10% 判。
- 「最好的現成規則」＝每一格裡 7 個現成規則（§2.4）中該指標最好的那一個（事後挑最好的，對 κ 不利）。

---

## 2. 模型

### 2.1 工作負載：Mooncake toolagent 推 session

- 檔案 `/mlsteam/data/tiara/datasets/traces/toolagent_trace.jsonl`（23,608 列）。**單位斷言（規則 6）**：`(len(hash_ids)−1)×512 < input_length ≤ len(hash_ids)×512`，23,608／23,608 列成立（conversation 12,031／12,031 也成立）。
- **推 session 的方法**（`infer_sessions()`，同 `code/m8_trace_oracle.py` D4 的規則，再鏈化）：
  1. B 的 parent＝更早的請求 A，A 的最後一個完整 block（A 至少 2 個完整 block，不靠大家共用的 block 0）出現在 B.hash_ids 的同一位置；Mooncake 的 hash 是前綴鏈，同位置同 hash ⇒ 整段前綴相同。取最深的。
  2. 一個 parent 有多個 child（平行工具呼叫／分支）時，最早的 child 當「下一輪」，其餘另起 session（冷啟動；不模擬跨 session 共用前綴）。
  3. **自我檢查（斷言）**：每一對都滿足 `B.hash_ids[:nfull_A] == A.hash_ids[:nfull_A]`、`B.ts ≥ A.ts`。**交叉欄位檢查（報告）**：B.input ≥ A.input＋A.output 的比例 **0.778**（9,167 對中）；其餘 22% 的下一輪沒有包含上一輪的全部輸出 → 可重用的歷史只算 A 的完整 block（`reuse＝nfull_A×512`），超出的副本作廢（§8 的失敗就出在這裡）。
- 結果：16,400 對 parent–child；2,567 個 parent 有多個 child；14,441 個 session，其中多輪 **4,571** 個（13,738 輪、9,167 次暫停）。**暫停**（下一輪時間戳 − 這一輪時間戳，解析度 1 s）：P10 0、P25 0、**P50 3.0 s**、P75 99 s、P90 258 s、P95 396 s、P99 624 s；**48.2% 是 0**（同一秒內就回來）。每個 session 最大 context：P50 7,087、P90 17,528、P99 59,849、最大 123,783 token（都在成本模型的有效範圍 ≤129,024 內）。最後一輪的比例 p_end＝0.333。數字在 `trace_sessions.json`。
- **暫停的定義**：直接用 trace 的時間差（含原系統當時的 prefill＋decode＋工具時間）；模擬是閉迴路：下一輪到達＝本輪在模擬裡結束＋暫停。
- **參數化間隔**（敏感度）：lognormal 中位 5／30／120 s、σ＝1（σ 是工作負載參數，不是成本常數；另跑 σ＝2）；`GapDist` 也支援 const 與經驗分位點（沒進主掃描）。用參數化間隔時，session 的長度結構仍取自 Mooncake，只換間隔。
- 每格：C 個 slot，每個 slot 依序跑抽到的 session（不重複抽樣，共 max(200, 12C) 個）；slot 開場與 session 之間的空檔也從同一個間隔分佈抽。所有策略用同一份工作負載（同 seed），3 個 seed。

### 2.2 系統

- **資源**（各自 FIFO、單一伺服器；事件模擬）：GPU 計算、H2D、D2H、SSD 讀、SSD 寫。HBM 與 CPU 容量以 byte 計。
- **一輪**＝(HBM 准入：需要 (input＋output)×bytes/token) → (還原歷史) → prefill 新 token＋第一個 token。**TTFT＝到達 → prefill 完成**（含等 HBM、還原、GPU 排隊）。**decode 不模擬**（TPOT 在這條路徑 NOT_MEASURED；KV 在 decode 期間一定要在 HBM，所有策略相同）。
- 准入 FIFO（head-of-line）；放不下時請策略趕人；都趕不動時照 vLLM 的作法搶占（丟最晚到、KV 還在 HBM 的那個）。
- **還原路徑**：GPU（0）；CPU（H2D）；SSD 級（讀完即可算，H2D 與讀重疊，沒另加）；DROP（從位置 0 重算）；有部分副本時＝載入副本＋重算其餘。`--restore cake` 時 CPU／SSD 改成 Cake 式（前段重算＋後段載入，分割點用閒置成本選）。
- **副本**：只寫還沒有副本的部分（增量）；下一輪截斷歷史時，舊副本只認共同前綴（epoch）。CPU 層滿了 LRU 淘汰副本。SSD 容量不設限（本地碟 3.5 TB）。
- **容量**：W_ref＝C ×（本次抽到的 session 最大 context 的平均）× bytes/token。HBM：緊 0.25、中 0.5、鬆 1.0 × W_ref，下限＝本次最大的單一 session（否則跑不了）；**real**＝vLLM 在 MI300X 實測的 KV token 數。CPU：照真實機器的 CPU:HBM 比例縮放＝HBM ×（容器記憶體上限 434.8 GiB ÷ 實測 HBM KV 容量 154.7 GiB ≈ 2.8）；real 時＝434.8 GiB；敏感度 `cpu_small`＝0.25 × HBM。
  - 注意：C≤8 時「緊」與「中」常常都被下限（最大的單一 session）墊住而相同（`cells.csv` 的 `hbm_over_W` 欄）。
- **GPU 忙**：`endo`（主）＝H2D 開始時 GPU 正在算 → 該次 H2D 乘實測的「忙/閒」倍數；GPU 計算是 FIFO，所以別人的 prefill 對重算的影響以排隊呈現。`exo`＝外加一個背景負載（第二階段的 busy＝`m7_busy` duty 0.7），計算與 H2D 都乘實測倍數。`off`＝都不乘（對照）。D2H 與 SSD 在 GPU 忙時的放大 **NOT_MEASURED → 不乘**。

### 2.3 動作的記法（「各動作被選的比例」）

暫停開始時每個策略恰好記一個決定：**KEEP**＝只留在 HBM（不寫下層副本）；**CPU／SSD**＝寫一份到該層（若同時立刻釋放 HBM 記為 eager）；**DROP**＝立刻釋放、不寫副本。另外記「有人要空間時趕到哪裡」（evict）與「回來時 KV 在哪裡」（return）。

### 2.4 策略

| 名稱 | 規則（出處） |
|:--|:--|
| `vllm` | vLLM APC：暫停時留著；放不下 LRU 淘汰＝丟，回來重算 |
| `vllm_cpu` | APC＋CPU 卸載（OffloadingConnector／LMCache 式）：每輪算完寫一份到 CPU，HBM 淘汰時有完整副本就瞬間釋放 |
| `vllm_tier` | vLLM 0.28 `TieringOffloadingSpec`：寫進 CPU 的 block 一律串寫到次層（原始碼 `v1/kv_offload/tiering/manager.py` 檔頭「Always offload to all tiers」）；CPU LRU |
| `continuum`／`continuum_cpu` | TTL：τ\*＝argmax_τ F(τ)·(T̄·η＋PR(n)) − τ（**照 L3 §H5 對原文 Eq.1–2 的轉述實作，本 agent 沒讀原文 PDF**）；F＝已知的間隔分佈；T̄＝被趕出者回來時的平均等待（線上，初值 0）；η＝1（原文冷啟動值）；PR＝有 CPU 卸載時載回時間、否則重算時間；TTL 內釘住、過期回到 LRU；死鎖時從最晚開始的 program 解釘 |
| `tokencake` | T_fc＝α·t_req＋(1−α)·t_hist（EWMA；**α 原文未給，取 0.5**）；T_fc ≤ T_offload＋T_upload 不卸；有請求在等 HBM 才卸到 CPU，並在預測回來前 T_upload 主動載回；重算不是選項（L3 §H6） |
| `infercept` | 浪費（記憶體×時間）：Discard＝T_rec·(n＋batch)、Preserve＝T_INT·n、Swap＝2·T_swap·batch（L3 §H1）；T_INT 用已暫停時間估 → 保留到 e\*·n＝min(Discard, Swap) 再換成 Swap（CPU）或 Discard；swap budget、分層 pipeline 不模擬 |
| **`kappa`**（事前凍結的候選） | 見 §2.5 |
| `kappa_tier`（**事後**加入） | `kappa`＋「κ 把關的串寫」：寫 CPU 副本時，只有用實測成本算的「從 SSD 級還原 < 重算」（依這個 session 的長度）才串寫到 SSD 級。**看過 `vllm_tier` 的結果才加**，結果另標 |
| `kappa_end`（歸因用） | `kappa`＋知道 session 已結束（最後一輪立刻丟、清副本） |
| `oracle_nopf`／`oracle`（歸因用） | 知道每個 session 確切回來時間（含不會再回來）；`oracle` 另外在確切回來前預取。**共用 GPU／鏈路時這是貪婪規則，不是下界**（§4.4） |

### 2.5 κ 感知（`kappa`）的定義與設計演進

**定義（v4，主掃描前凍結）**：每個暫停中的 session v（已暫停 e 秒、KV b bytes）
- C_X(v)＝用實測成本算的還原時間，X ∈ {CPU（要有空間）, SSD 級, DROP}；**X\*＝argmin C_X**（κ 在這裡起作用）。
- R̄(v)＝E[G−e | G>e]（已知的間隔分佈）；P(回來)＝由間隔分佈與 p_end 算的條件機率；W̄＝被趕出者回來時平均等 HBM 的時間（線上）。
- **score(v)＝P(回來)·(C_X\*＋W̄) ÷ (b·R̄)**＝「趕它出去，每釋放 1 byte·秒 HBM 的預期 TTFT 代價」。
- 暫停開始（四選一）：若 X\*＝CPU 且 R̄ > max(寫副本時間, T̄_ev)（T̄_ev＝這一格裡暫停中的 session 平均撐多久才被趕；沒趕過＝∞＝不寫）→ 背景寫 CPU 副本（記 CPU，HBM 先當快取）；否則 KEEP。**不在暫停開始就釋放 HBM**。
- 有人要空間：依 score 由小到大趕；目的地＝argmin{搬移時間＋C_X（CPU/SSD）, C_DROP}。不預取。

**設計演進（誠實記錄；都是在掃描前、只看過少數單一 seed 的格子時改的）**：
1. v1：有壓力（HBM 空閒 <10% 或有人等）時，若自己是最便宜的可趕對象就立刻卸。問題：C=1 時「自己佔滿 HBM」也算壓力 → 自己把自己趕出去（自測 C=1 抓到）；「壓力」改成只算「有請求在等 HBM」（Tokencake 原文的說法，所有策略共用）。
2. v2：加長期的 HBM 影子價格 μ̄（被趕者實際付的代價 ÷ 不在 HBM 的 byte·秒），用五分鐘法則式打平比較決定立刻卸。問題：μ̄ 由自己的趕人行為估出，會收斂到「平手」；而且立刻卸掉的 session 若馬上回來要重新排隊拿 HBM，結果幾乎全部立刻卸、P95 變差。
3. v3：J(X) 加上 W̄（重新排隊的等待）。仍然幾乎全卸。
4. v4（凍結）：理由——有 CPU 副本時「等有人要空間再放」弱支配「現在就放」，所以暫停開始只決定**副本寫到哪一層**，釋放一律延後；副本只在「預期比通常被趕的時間更晚回來」時才寫（避免間隔→0 時白寫）。
- 另外為了公平，**加了 `vllm_tier` 這個現成規則**（vLLM 0.28 本來就有的 CPU＋檔案層串寫），之後才加 `kappa_tier`（事後）。

---

## 3. 成本常數與出處

全部由 `load_costs()` 從檔案讀出；逐項（含 run_id）在 `costs_sources.csv`，缺的在 `costs_missing.csv`。

### 3.1 MI300X（harness 路徑＝第二階段的 Restorer；**唯一能跑動態模擬的平台**）

| 常數 | Llama-3.1-8B | Qwen2.5-7B-1M | 出處（run_id） | 註 |
|:--|--:|--:|:--|:--|
| KV bytes/token | 131,072 | 57,344 | HF config.json；與 `bw.json` 的 chunk_bytes÷512 交叉驗證（斷言） | |
| 重算 f(i)＝α＋β·i（ms／512-token chunk） | α 28.874、β 0.8846 | α 23.843、β 0.7695 | `m10_precision_mi300x/speed_<m>/calib_f.csv`、`bw.json`（`20261015-154558-m10-calibf-llama`、`…-154647-…-qwen`） | 64 個 chunk 中位數擬合，逐 chunk 殘差 ≤1.1%／1.4% |
| 擬合預測 vs 實測 compute_only（閒） | 32K −0.75%、64K −0.72%、128K +0.39% | −1.73%、−0.74%、+0.20% | 32K：`speed_<m>/summary_fp8.csv`；64K／128K：`m10_longctx_mi300x/speed_<m>_{65536,129024}_*/summary.csv` | **樣本外驗證**（64K、128K 沒進擬合）；斷言 <3% |
| H2D 還原（CPU 層全部載入，BF16） | 48.17 GiB/s（截距 −0.3 ms→0） | 45.95 GiB/s（截距 0.31 ms） | 32K／64K／128K 的 `load_only` E0 cpu idle（`…-grid-fp8-*`、`…-lc-grid-*`） | `bw.json` 的單塊 pinned H2D 49.9／48.6 交叉參考 |
| GPU 忙的放大（H2D） | 1.50／1.49／1.75（32K/64K/128K） | 1.43／1.66／2.10 | 同上的 busy／idle | 忙＝duty 0.7；<32K 用 32K 值（常數延伸，標 flag） |
| GPU 忙的放大（重算） | 2.12／1.90／1.71 | 2.11／1.92／1.70 | compute_only busy／idle | 第一階段 C5 的 2.07（`busy_levels.json`）只當交叉參考 |
| 新一輪（256 token＋第一個 token，閒） | 8K 0.0354、16K 0.0477、32K 0.0757、64K 0.1318、128K 0.2373 s | 32K 0.0662、64K 0.1143、128K 0.2073 s | ttft − t_restore | **Qwen 的 8K／16K 列 contaminated_rows=10，不用** → <32K 用 32K 值（常數延伸） |
| 新一輪（忙） | 0.234／0.333／0.695 s | 0.202／0.296／0.608 s | 同上 busy | <32K 用 32K 的忙/閒比例延伸 |
| D2H | 42.88 GiB/s | 同左（**只量過 Llama 的 64 MiB chunk 形狀**） | `m7_write_policy_mi300x/calib_c0.csv`（`20261008-130257-m7-c0`），rep≥1 中位數 | |
| H2D 遇到同時 D2H 的變慢 | 1.161× | 同左 | `calib_c0_duplex.csv`（`20261008-132007-m7-duplex`） | |
| SSD 級（本地，真實裝置） | 讀 8,053.8、寫 2,178.9 MiB/s | 同左 | `calib_c2.csv` 持續讀寫中位數（`20261008-130438-m7-c2`，各 n=3） | 第二階段「SSD 級」是限速參數 6.98 GiB/s（實測最慢×0.9）；**本模擬用實測中位數**，參數只列對照 |
| 寫入中讀取變慢 | 3.306×（本地）、1.673×（NFS） | 同左 | `tier_params.json`（第一階段 A2 真實裝置） | |
| NFS（敏感度用的「SSD 級」） | 讀 391.7、寫 689.2 MiB/s | 同左 | `calib_c2.csv` | 真實裝置，不是限速參數 |
| 真實 HBM KV 容量（vLLM BF16） | 1,267,504 token（154.7 GiB） | 2,889,696 token（154.3 GiB） | `audit_20261010/m5_server_paperwork_mi300x.csv`（m5 的 server log） | |
| 容器記憶體上限（設定值） | 434.8 GiB | | `/sys/fs/cgroup/memory.max` | experiment_matrix §0 寫 435 GiB |

### 3.2 NOT_MEASURED（因此不跑動態模擬）

- **3090（平台 A）**：舊結果在 git `52803bc^`（工作樹已刪，「推倒重來」），只讀。有：vLLM 路徑的 CPU／SSD（SATA、NVMe）還原與重算（16K、Llama BF16 權重），重算位置只量到 24,576。**缺：D2H（卸載寫出）、GPU 忙的放大、原始 pinned H2D** → 動態模擬拒跑（`Costs.require()`）。所有列都是 `host_contention=HEAVY`（舊 RUNLOG 發現 5：PCIe 路徑被平行 job 灌水 26–52%）。Qwen 只有 AWQ 權重（`qwen-awq`），和 MI300X 的 BF16 權重不可通約 → 整組 NOT_MEASURED。
- **MI300X 的 vLLM 引擎路徑**（`audit_20261010/cost_constants_mi300x_rebuilt.csv`）：只有還原端（一個 ctx 點），store／D2H、GPU 忙放大 NOT_MEASURED。
- 兩者都只進 §4.5 的**靜態 κ 表**（只看還原端）。
- 全平台：decode 的 TPOT（這條路徑）、D2H／SSD 在 GPU 忙時的放大。

---

## 4. 結果（run `20261019-141254-m16-pausesim`；11,232 次模擬，0 次失敗）

**掃描**：主掃描＝MI300X × {Llama, Qwen} × C {1, 8, 32} × 容量 {緊, 中, 鬆, real} × 間隔 {Mooncake 推得, LN 中位 5／30／120 s} × 3 seed（96 格）。敏感度（C {8, 32} × 容量 {緊, 中} × 4 種間隔 × 2 模型 × 3 seed，各 32 格）：`busy_exo`、`busy_off`、`ssd_nfs`（SSD 級換 NFS）、`cake`（Cake 還原）、`cpu_small`（CPU＝0.25×HBM）、`cpu_small_nfs`、`sigma2`（σ＝2，24 格）。每格 12 個策略。逐格數字在 `cells.csv`；彙總在 `summary.json`；跨設定的遺憾在 `regret.csv`；決策帶的翻轉在 `flips.csv`；原始逐次結果在 run 目錄的 `scan.csv`（7.9 MB）、`decisions.csv`（58 MB）。

### 4.1 候選對「每格最好的現成規則」（回來那一輪的 TTFT P95）

| 掃描 | 格數 | `kappa` 中位／最好 | `kappa` ≥5%／≥10%／≤−5% | `kappa_tier` 中位／最好 | `kappa_tier` ≥5%／≥10%／≤−5% | `kappa_end` 中位 | 完美資訊最佳 中位／最好 |
|:--|--:|:--|:--|:--|:--|--:|:--|
| main | 96 | −2.8%／+1.3% | 0／0／42 | −1.6%／+9.4% | 1／0／27 | +0.0% | +0.0%／+28.5% |
| ssd_nfs | 32 | −5.3%／+0.4% | 0／0／16 | −3.1%／+3.3% | 0／0／14 | +26.1% | +27.1%／+70.0% |
| cpu_small | 32 | +35.6%／+58.9% | 25／21／6 | +49.4%／+78.7% | 32／32／0 | +53.5% | +60.6%／+69.8% |
| cpu_small_nfs | 32 | +7.1%／+35.4% | 21／15／3 | +7.1%／+35.4% | 21／15／3 | +27.0% | +31.5%／+59.9% |
| busy_exo | 32 | −48.9%／−1.3% | 0／0／31 | −4.1%／+3.3% | 0／0／15 | −0.4% | −0.2%／+4.7% |
| busy_off | 32 | −57.6%／−10.8% | 0／0／32 | −4.8%／+9.4% | 1／0／15 | +0.8% | +0.8%／+28.5% |
| cake | 32 | −65.5%／−6.9% | 0／0／32 | −5.3%／+7.8% | 1／0／17 | −0.5% | +0.3%／+21.9% |
| sigma2 | 24 | −40.8%／−13.6% | 0／0／24 | −9.3%／+0.8% | 0／0／16 | −0.3% | +0.7%／+6.2% |

- **P95 JCT（L3 原本的指標）**：所有掃描裡 `kappa` 0 格好 ≥5%；`kappa_tier` 只有 `cpu_small` 4／32 格 ≥5%。JCT 由暫停本身主導，各掃描的 gain 中位數絕對值都 <0.5%。
- **全部輪的 TTFT P95** 與 **回來那一輪的平均** 結論相同（`summary.json` 的 `ttft_p95`、`ttft_ret_mean`）。
- 主掃描依容量（C>1 的 16 格）：

| 容量 | `kappa` 中位（最差／最好） | `kappa_tier` 中位（最差／最好） | `kappa_end` 中位（最好） |
|:--|:--|:--|:--|
| 緊 | −132%（−371%／−11.9%） | −8.0%（−29.9%／+9.4%） | −13.3%（+22.6%） |
| 中 | −31.1%（−191%／−10.7%） | −5.3%（−29.9%／+1.6%） | +2.7%（+22.6%） |
| 鬆 | −6.0%（−17.2%／+1.3%） | −4.5%（−17.2%／+0.9%） | −2.3%（+8.4%） |
| real | −1.8%（−17.2%／+0.4%） | −1.8%（−17.2%／+0.4%） | −0.1%（+0.0%） |

- C=1 的格子：容量被「最大的單一 session」墊住（HBM／W_ref＝6.7，real＝134），κ 對最好的規則差 0～−1.4%（3 seed 平均）。C≤8 的「緊」與「中」被下限墊成同一個容量（HBM／W_ref＝0.84）。

**最好的現成規則是誰**（各格計數，`summary.json` 的 `best_baseline_counts`）：main：`vllm_tier` 45、`vllm` 38（多為 C=1 與 real）、`tokencake` 7、`infercept` 5、`vllm_cpu` 1；ssd_nfs：`tokencake` 22、`infercept` 5、`vllm_cpu` 4、`vllm_tier` 1；cpu_small／busy_*／cake／sigma2：幾乎全是 `vllm_tier`。

### 4.2 代表性格子（回來那一輪的 TTFT P95，秒；3 seed 平均；粗體＝該格最好的現成規則）

| 設定 | vllm | vllm_cpu | vllm_tier | continuum | continuum_cpu | tokencake | infercept | kappa | kappa_tier | kappa_end | 完美資訊最佳 | HBM／W_ref |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| main Llama C=8 緊 Mooncake | 2.32 | 0.85 | **0.50** | 2.32 | 0.85 | 0.61 | 0.85 | 0.86 | 0.50 | 0.39 | 0.36 | 0.84 |
| main Llama C=32 緊 Mooncake | 4.33 | 3.00 | **0.97** | 4.33 | 3.00 | 2.99 | 2.98 | 3.05 | 0.88 | 2.00 | 1.94 | 0.26 |
| main Qwen C=32 緊 Mooncake | 3.38 | 2.49 | **0.68** | 3.38 | 2.49 | 2.44 | 2.48 | 2.51 | 0.69 | 1.46 | 1.44 | 0.26 |
| main Llama C=32 中 LN30 | 8.20 | 2.23 | **1.16** | 8.20 | 2.23 | 1.77 | 2.23 | 2.27 | 1.21 | 0.98 | 0.96 | 0.50 |
| main Llama C=32 real Mooncake | 0.67 | **0.65** | **0.65** | 0.67 | **0.65** | 0.66 | 0.68 | 0.66 | 0.66 | 0.65 | 0.65 | 4.16 |
| ssd_nfs Llama C=32 緊 Mooncake | 4.33 | 3.00 | 13.39 | 4.33 | 3.00 | 2.99 | **2.98** | 3.05 | 3.05 | 2.00 | 1.94 | 0.26 |
| ssd_nfs Qwen C=32 緊 Mooncake | 3.38 | 2.49 | 3.75 | 3.38 | 2.49 | **2.44** | 2.48 | 2.51 | 2.45 | 1.46 | 1.44 | 0.26 |
| cpu_small Llama C=32 緊 LN30 | 9.29 | 9.14 | **7.21** | 9.29 | 9.14 | 8.50 | 9.31 | 3.89 | 2.65 | 3.30 | 2.83 | 0.26 |
| cpu_small_nfs Llama C=32 緊 LN30 | 9.29 | 9.14 | 20.43 | 9.29 | 9.14 | **8.50** | 9.31 | 9.52 | 9.52 | 8.82 | 8.67 | 0.26 |
| cpu_small Qwen C=32 緊 Mooncake | 3.38 | 3.39 | **2.77** | 3.38 | 3.39 | 3.37 | 3.59 | 1.44 | 0.84 | 1.32 | 1.25 | 0.26 |
| busy_exo Llama C=32 緊 LN30 | 37.03 | 26.07 | **7.03** | 37.03 | 26.07 | 26.93 | 28.09 | 26.62 | 7.11 | 14.08 | 14.08 | 0.26 |

讀法：
- 「緊／中」時決定勝負的是**有沒有一個夠快的下層可以接住 CPU 放不下的 KV**：本地 SSD 讀 8 GB/s 級、Llama 重算 32K 要 3.6 s，從 SSD 讀 4 GiB 約 0.5 s，所以「全部串寫到 SSD」（`vllm_tier`）大勝；`kappa`（事前凍結版）沒有背景串寫，CPU 滿了只能丟 → 輸。
- 同一格換成 NFS：`vllm_tier` 從最好變成最差（Llama 0.97 → 13.39 s），因為從 0.38 GiB/s 的 NFS 讀比重算還慢；κ 把關的兩個候選都不用 NFS（和 CPU-only 規則同級）。
- `cpu_small`：`vllm_tier` 的次層要經過 CPU 主層，CPU 只有 0.25×HBM 時大的 session（≥64K）副本寫不進去，被趕時只能丟（決策帶：≥64K 的趕出 100% DROP）；κ 系列從 HBM 經小緩衝直接串流寫 SSD（≥64K 100% 趕到 SSD）。所以這裡的大勝一部分是**寫入路徑的能力差**，不是 κ。
- GPU 外加忙（exo）時重算×1.7–2.1、H2D×1.4–2.1（實測），更偏向「從 SSD 讀」，`vllm_tier` 與 `kappa_tier` 同級。

### 4.3 決策翻轉與「搬錯規則」的代價（`regret.csv`）

A 設定下最好的現成規則原封搬到 B 設定，比 B 的最好規則差多少（回來那一輪的 TTFT P95；各 32 組配對）：

| 換什麼 | 最好的規則換人 | 搬過去的遺憾 中位（最大；≥10% 組數） | `kappa` 的遺憾 中位（≥10%） | `kappa_tier` 的遺憾 中位（≥10%） |
|:--|--:|:--|:--|:--|
| 下層 本地 SSD → NFS | 31／32 | +60.3%（+482%；29） | +5.3%（11） | +3.1%（9） |
| 下層 NFS → 本地 SSD | 31／32 | +25.7%（+353%；26） | +57.3%（32） | +5.6%（13） |
| CPU 小，本地 SSD → NFS | 28／32 | +50.6%（+211%；26） | −7.1%（3） | −7.1%（3） |
| CPU 小，NFS → 本地 SSD | 28／32 | +19.0%（+131%；26） | −35.6%（6） | −49.4%（0） |
| GPU 閒 → 外加忙 | 2／32 | +0.0%（+0.8%；0） | +48.9%（22） | +4.1%（4） |
| GPU 外加忙 → 閒 | 2／32 | +0.0%（+4.9%；0） | +57.6%（32） | +4.8%（13） |

（負的遺憾＝比該設定最好的現成規則還好。）

**κ 自己的決策帶翻轉**（`flips.csv`；同一個 C／容量／間隔／context 帶／真實暫停帶，多數動作不同才算）：
- 下層 本地 SSD ↔ NFS（CPU 小）：`kappa` 的趕出目的地 163／418 個帶翻轉（本地 SSD→趕到 SSD；NFS→丟）。這是設計上預期的。
- Llama ↔ Qwen：`kappa` 回來時位置 2／460 個帶翻轉、趕出目的地 0／361。GPU 閒 ↔ 外加忙：回來時位置 18／320、趕出目的地 0／429。
- 靜態表（§4.5）預測 Qwen 在 NFS、≥64K 應該改存 NFS；動態模擬裡 κ 在 ≥64K 仍 100% 選丟（cpu_small_nfs 的 ≥64K 趕出 189／189 DROP），因為被趕時 NFS 的寫入（0.67 GiB/s）在關鍵路徑上。

### 4.4 歸因：完美資訊能多拿多少、從哪裡來

- `kappa_end`（只多知道「這個 session 結束了」）：main C>1 中位 +0.0%、最好 +22.6%；ssd_nfs 中位 +26.1%；cpu_small_nfs 中位 +27.0%。也就是說，在下層慢的時候，**最大的空間來自「別把已結束的 session 的 KV 留著佔位」**，是應用層的提示（Continuum／Leyline 一類的方向），不是 κ。
- 完美資訊的三個變體共用 κ 的結構（沒有 SSD 背景串寫），**不是下界**：main C=32 緊時比 `vllm_tier` 差約 2 倍（Llama 1.94 vs 0.97 s）。自測只在兩種情況把 oracle 當下界檢查：靜態逐次暫停（嚴格）與 C=1 動態（等待＋還原＝0）；C>1 動態裡 oracle 輸給 `vllm_tier` 的情形照實列出（自測 INFO 1／32）。
- 預取（`oracle` 對 `oracle_nopf`）有時反而變差（佔 H2D 與 HBM），差距都在幾 % 內。

### 4.5 靜態 κ 表（只看還原端、GPU 閒；`decision_map_static.csv`）

每 token 還原成本（µs／token）與「最便宜→最貴」：

| 平台／路徑 | 模型 | 下層 | context | CPU | SSD 級 | 重算 | 排名 |
|:--|:--|:--|--:|--:|--:|--:|:--|
| MI300X harness | Llama | 本地 SSD | 8K → 128K | 2.53 | 15.5 | 69 → 273 | CPU > SSD > 重算（全部長度） |
| MI300X harness | Llama | NFS | 8K → 128K | 2.53 | 319 | 69 → 273 | CPU > 重算 > NFS（全部長度；128K 時 κ_ssd＝0.86） |
| MI300X harness | Qwen | 本地 SSD | 8K → 128K | 1.17–1.20 | 6.79 | 58 → 235 | CPU > SSD > 重算 |
| MI300X harness | Qwen | NFS | 8K–32K | 1.17–1.20 | 140 | 58–94 | CPU > 重算 > NFS |
| MI300X harness | Qwen | NFS | 64K、128K | 1.17 | 140 | 142、235 | **CPU > NFS > 重算（翻轉）** |
| MI300X vLLM 引擎路徑 | Llama | vLLM fs 層 | 16K | 30.5 | 64.9 | 46.7 | CPU > 重算 > SSD |
| MI300X vLLM 引擎路徑 | Qwen | vLLM fs 層 | 96K | 23.5 | 44.4 | 104.8 | CPU > SSD > 重算 |
| 3090 vLLM 路徑（HEAVY 污染） | Llama | SATA | 16K | 36.8 | 346 | 326 | CPU > 重算 > SSD |
| 3090 vLLM 路徑（HEAVY 污染） | Llama | NVMe | 16K | 33.9 | 393 | 324 | CPU > 重算 > SSD |

- 同一台 MI300X、同一個 Llama 16K：harness 路徑 SSD 比重算便宜 5.4 倍，vLLM 引擎路徑 SSD 比重算貴 1.4 倍——**翻轉來自軟體路徑**（vLLM 的 CPU 路徑只有約 4 GiB/s、harness 的重算比 vLLM 慢 1.7–2 倍），不是硬體。
- 3090 與 MI300X 在 vLLM 路徑下排名相同（CPU > 重算 > SSD）；κ_cpu 3090 是 8.9–9.5、MI300X 是 1.5（Llama 16K）。這是靜態、只看還原端的比較，3090 的數字有 HEAVY 污染，**不能當作動態決策會翻轉的證據**。

---

## 5. 判定（照 L3 §5 缺口 1）

| 判準 | 結果 | 判定 |
|:--|:--|:--|
| κ 感知 vs 最好的現成規則，P95 好不到 5% → 缺口不成立 | 事前凍結的 `kappa`：主掃描 0／96 格 ≥5%（TTFT P95 與 JCT P95 都是） | **成立（缺口不成立）**——照 CLAUDE.md §1 規則 4（<5% 停），這個候選停 |
| ≥10% 且有翻轉 → 值得 GPU 驗證 | `kappa` 只在 `cpu_small`（敏感度）有 ≥10%；`kappa_tier`（事後）只在 `cpu_small`／`cpu_small_nfs`；而且那裡的優勢一部分是寫入路徑能力 | **不成立**（事前候選）；事後候選的訊號需要事前登記再驗 |
| 決策會不會隨硬體翻轉 | 隨**下層裝置**翻：最好的現成規則 31／32 組換人，搬錯中位 +26～+60%；隨**GPU 忙**幾乎不翻（2／32）；隨**模型**幾乎不翻；靜態表另顯示隨**軟體路徑**翻 | **有翻轉，但 κ 感知的價值是「不會選錯」（遺憾中位 3–6%），不是「贏過該硬體上最好的規則」** |

**一句話**：在這個模型與這些實測常數下，「用 κ 在四個動作間選」打不過「在該硬體上挑對的那一個現成規則」；它的價值是可攜（換硬體不必重挑規則）。缺口 1 若要繼續，題目應改成「**規則選擇（或參數）要依實測 κ 自動調**」並把「應用層告知 session 結束」一併考慮，而不是「κ 感知策略打敗現成規則」。這一點需要使用者決定（5–15% 區間的精神：停下來問人）。

---

## 6. 限制（這是模擬，不是實測）

1. **成本來自 harness 路徑**（第二階段的 Restorer，單請求量的）：harness 的重算比預設 vLLM 慢 1.7–2 倍（toolbox §4），vLLM 的 CPU 載回只有約 4.1 GiB/s（harness 48 GiB/s）。用 vLLM 的常數，κ 小很多、排名可能不同（§4.5）。
2. **併發的資源模型是假設**：GPU、H2D、D2H、SSD 讀、SSD 寫各自 FIFO 單一伺服器；GPU 忙的放大只在 H2D 開始時看一次；D2H 與 SSD 在 GPU 忙時的放大 NOT_MEASURED（沒乘）；SSD「寫入中讀變慢 3.3×」來自第一階段 A2（單一 session 的量測）。
3. **限速參數 vs 真實裝置**：本模擬的 SSD 級與 NFS 用第一階段 C2 的**真實裝置實測中位數**（本地讀 8,053.8／寫 2,178.9 MiB/s；NFS 讀 391.7／寫 689.2 MiB/s）。第二階段的「SSD 級 6.98 GiB/s」「NFS 級 0.331 GiB/s」是限速器參數（實測最慢×0.9），本模擬沒用；本地 SSD 的讀取量測在 overlay 上，有沒有被 page cache 影響，C2 的檔頭寫 O_DIRECT，本 agent 沒有再驗。
4. **外推與延伸**：重算擬合在 64K、128K 樣本外驗證誤差 <1%；但新一輪的固定開銷在 <8K（Llama）／<32K（Qwen，8K／16K 列被污染）用最小量測點的值常數延伸；GPU 忙的倍數在 <32K 常數延伸；D2H 只量過 Llama 的 chunk 形狀。
5. **工作負載**：Mooncake toolagent 的 session 是用 hash 前綴鏈推的（22% 的下一輪不含上一輪的完整輸出）；分支（2,567 個 parent 有多個 child）另起 session、不共用前綴；時間戳解析度 1 s（48% 的暫停是 0）；暫停用 trace 的時間差（含原系統的 prefill／decode）；只取多輪 session。lognormal 的 σ 是自選參數。外部 trace（Exgentic、SemiAnalysis、TraceLab）沒有下載。
6. **沒模擬的**：decode（TPOT NOT_MEASURED）、跨 session 前綴共用、block 粒度的部分淘汰（以整個 session 為單位）、vLLM 的分塊 prefill 交錯、InferCept 的 swap budget 與分層 pipeline、Continuum 的程式結構預測（η 固定 1）。
7. **策略實作**：Continuum、Tokencake、InferCept 都是照 L3 的轉述實作，本 agent 沒讀原文 PDF；Tokencake 的 α＝0.5 是自選。κ 感知經過 4 版設計（§2.5），`kappa_tier` 是看過結果後才加的。
8. **平台 A（3090）的動態模擬 NOT_MEASURED**：缺 D2H、GPU 忙放大、原始 H2D；而且舊量測全部 HEAVY 污染。所以「NVIDIA 選卸載、MI300X 選重算」這種跨廠商的動態翻轉，本模擬**回答不了**。

---

## 7. GPU 驗證點建議

- **要驗的不是 `kappa`（已照判準停），而是「翻轉」這個前提**：同一個工作負載，vLLM 0.28 的四種原生設定在「下層＝本地 SSD」與「下層＝NFS」之間，最好的那個會不會換人。這是缺口 1 剩下唯一有訊號的部分，而且**不需要寫新的排程器**（`code/m16_vllm_bench.py` 已有 `--offload cpu:<GiB>`／`tier:<cpuGiB>:<fs_path>:<fsGiB>`）。
- **設定**：Llama-3.1-8B（BF16 KV）、單卡 MI300X、vLLM 0.28；C＝32 個同時活著的 agent（Mooncake toolagent 推得的 session 重播，或 `benchmarks/multi_turn` 以 lognormal 中位 30 s 的思考時間）；HBM 用 `--num-gpu-blocks-override` 壓到參考工作集的 0.25 倍（模擬中約 10 GiB KV；實際值照抽到的 session 算）；CPU 卸載 ≈ 2.8×HBM（照真實比例）；四個設定：(a) APC、(b) APC＋CPU、(c) APC＋CPU＋檔案層 `/var/tmp`、(d) 同 (c) 但檔案層在 `/mlsteam/data`（NFS）。
- **模擬的預測**（回來那一輪的 TTFT P95，Llama C=32 緊 Mooncake）：(a) 4.33、(b) 3.00、(c) 0.97、(d) 13.39 s。驗證標準照 L3 X3：**排名一致**（(c) 最好、(d) 最差），差距容忍 ±30%。注意 vLLM 的重算比 harness 快 1.7–2 倍、CPU 路徑慢約 12 倍，(b) 與 (a) 的差距可能縮小。
- 規模：模擬裡一格 384 個 session，在 LN30 下約 40 分鐘 wall-clock；GPU 上建議 128 個 session × 4 設定 × 1 seed（約 1–2 小時，量時間要包 `GpuWatcher`）。NFS 那組要避開第二階段量 NFS 的 stage（experiment_matrix §6）。
- 若使用者想保留 κ 方向：另一個值得的 GPU 點是 `cpu_small`（CPU 卸載 0.25×HBM）＋本地 SSD，比較 (c) 與「HBM 直接串流寫 SSD」——但後者 vLLM 0.28 沒有，需要寫程式，期限內不建議。

---

## 8. 失敗紀錄與重現

- **失敗的 run**：`20261019-141044-m16-pausesim`，exit_code 1。scan 跑到約第 80／936 格時，帳目檢查 `check_books()` 斷言「副本不可能比 KV 長」失敗（完整 traceback 在該目錄 `stderr.log`）。原因：下一輪截斷歷史（22% 的情況）時，舊內容的 SSD／CPU 副本寫入仍在進行，完成時把整段長度記成有效副本。修法：副本加 epoch，完成時只認和新內容共同的前綴；CPU 保留量改由「已完成副本與進行中寫入的最大值」重算；加回歸測試（截斷＋NFS 慢寫，已確認會走到 12 次「舊寫入被裁短」的路徑）。該 run 沒有寫出任何結果檔。修正後重跑＝`20261019-141254-m16-pausesim`（exit_code 0）。
- **重現**（只用 CPU）：
  ```bash
  cd /mlsteam/workspace/paper-hierarchical-kv-state
  PY=/mlsteam/workspace/venv/tiara/bin/python; N="nice -n 19 taskset -c 0-3"
  $N $PY code/m16_pause_sim.py --self-test
  RUN_ID=<run_id> $N $PY code/m16_pause_sim.py costs --out results/m16_stage3_mi300x/i3_pause_sim
  RUN_ID=<run_id> $N $PY code/m16_pause_sim.py trace --out results/m16_stage3_mi300x/i3_pause_sim
  RUN_ID=<run_id> $N $PY code/m16_pause_sim.py decision-map --out results/m16_stage3_mi300x/i3_pause_sim
  RUN_ID=<run_id> $N $PY code/m16_pause_sim.py scan --out results/m16_stage3_mi300x/i3_pause_sim --raw /mlsteam/data/tiara/runs/<run_id> --sens --procs 4
  RUN_ID=<run_id> $N $PY code/m16_pause_sim.py analyze --out results/m16_stage3_mi300x/i3_pause_sim --raw /mlsteam/data/tiara/runs/<run_id>
  # 單格：$N $PY code/m16_pause_sim.py one --C 32 --cap tight --gap mooncake [--ssd-dev nfs] [--cpu-mult 0.25] [--busy exo]
  ```
  完整指令在 run 目錄的 `cmd.sh`；context（程式 sha256、python／numpy 版本、cgroup 上限）在 `context.txt`。整個掃描 4 核約 2 分鐘。
- **產出檔**（`results/m16_stage3_mi300x/i3_pause_sim/`，每個 CSV 都有 `run_id`、`ts`）：`costs_sources.csv`、`costs_missing.csv`、`trace_sessions.json`、`decision_map_static.csv`、`cells.csv`、`regret.csv`、`flips.csv`、`summary.json`。
