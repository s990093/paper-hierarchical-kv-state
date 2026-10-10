# 一頁大綱：延後也追得上（負面結果論文）

> F5，2026-10-10（同日依 F1–F4、F6 更新）。草稿全文：[draft.md](draft.md)；英文摘要與投稿場所：[abstract_en.md](abstract_en.md)。self-review，非 cross-model。
> 數字的來源標記與代號（A0、B2、R1GPU、F1、F6…）見 draft.md 附錄 B。這一頁不產生任何圖，只寫「每張圖要從哪個 CSV 的哪些欄位來」。〔待重量〕＝絕對值取決於 harness 的重算速度（draft §3.1）。

## 定位（輕量版 Brainstorm）

- **一句話**：讀取端會自適應（Cake）時，寫入當下依位置決定 KV 放哪一層，在單卡、實測參數下都被「同一條規則延後套用」追平；延後版只在 SSD 持續寫不完時會輸，而那在單卡上不出現。
- **誰在乎**：做 KV 分層（vLLM、LMCache、SGLang、Dynamo）和寫入時放置（Krul、TierKV、Lachesis）的人。
- **結構性的缺口**：寫入時放置的前作幾乎都沒和延後雙胞胎比過〔Lit-A §0〕。
- **命名的概念**：「延後測試」（deferral test）：寫入時的想法要同時贏過延後、背景、預先清理、寫穿、讀取時五種雙胞胎。
- **故事線**：Cake 有一段有用的頻寬帶 → 在帶子裡，寫入時放置仍被延後版追平 → 原因是位置延後也看得到 → 唯一例外是持續寫不完 → 單卡到不了（新 seed 事先登記確認）→ 給設計者的建議與方法教訓。旁支：真實 vLLM 的 kernel 陷阱。
- **最大的兩個缺口**：harness 重算比預設 vLLM 慢 1.7–2 倍（重算變快時模擬裡有少數存活者）；多卡共用 SSD 的頻寬沒量。

## 主張 → 章節 → 證據

| # | 主張 | 章節 | 主要證據 |
|:--|:--|:--|:--|
| C1 | Cake 只在一段頻寬帶明顯有用：harness 上 0.33–3.73 GiB/s 1.38–1.93×，11.6 GiB/s 1.05×；帶子的絕對位置〔待重量〕 | §4.1 | 〔實測 A0〕；〔實測 F1〕（harness 偏慢） |
| C2 | 寫入時放置在 38 組 GPU 比較（17＋16＋5）裡，沒有一次同時贏過全部延後雙胞胎 ≥5%；位置延後也拿得到。重算變快時證據最弱（模擬 3／30 存活，未上 GPU） | §4.2、§4.5 | 〔實測 B、B2、R1GPU〕；〔模擬 SIM08，F5 重算〕；D1–D8 |
| C3 | 贏家由回來順序決定，不由 κ 決定（模擬 17／64 組贏家不變） | §4.3 | 〔模擬 D3〕〔實測 B、B2、R1GPU〕 |
| C4 | 延後版唯一會輸的「持續寫不完」區域，在單卡實測參數下不出現：ρ_KV ≤0.99；新 seed 事先登記重跑 ρ_KV<1 的 216 格 0 格通過；過載角落只剩 1 格、只在中位數上贏；vLLM 跳過 0／133,504 | §4.4 | 〔模擬 D1SW、F6〕〔實測 D2〕〔實測 D7〕 |
| C5 | 方法：延後測試、模擬門檻 > 模擬器相對偏差（約 5–6 點）、事後修正要用新 seed 事先登記重跑、確認引擎實際用的 kernel | §5.3 | 12 §5.4、RT、D1、F6、F1 |
| C6（工程） | vLLM 0.28 在 ROCm 上開任何 KV connector 就把 ROCM_ATTN 換成 TRITON_ATTN；33K prefill 的 attention 1.367 → 3.277 s；K/V 拆分修法無損拿掉 45–74%；上游未修（#60316） | §4.6 | 〔實測 F1〕 |

## 版面（6 頁 workshop；Architecture）

| 節 | 頁 | 主張 | 圖表 |
|:--|:--|:--|:--|
| 1 引言 | 0.75 | 問題、缺口、C1–C6、範圍 | — |
| 2 背景 | 0.5 | KV 分層、Cake、κ(i)＝f(i)÷ℓ、延後雙胞胎 | 策略表（小） |
| 3 方法 | 0.75 | harness 正確性與**速度更正**、限速層、模擬器準度與偏差、延後測試、事先登記 | — |
| 4.1 | 0.5 | C1 | **圖 1** |
| 4.2 | 0.75 | C2（含重算變快的敏感度） | **表 1** |
| 4.3 | 0.5 | C3 | **圖 2** |
| 4.4 | 0.75 | C4 | **圖 3** |
| 4.5–4.6 | 0.5 | 8 個方向表；C6 | **表 2** |
| 5 討論 | 0.5 | 什麼時候寫入時放置會重要（含能耗）；建議；方法教訓 | — |
| 6–7 | 0.5 | 相關研究、限制（含 main.tex 稽核） | — |

## 4 張關鍵圖表＋1 張表（全部從現有 CSV 來，不要新跑）

**圖 1：Cake 的頻寬帶**（C1）
- 檔案：`results/m7_write_policy_mi300x/summary_a0_speedup.csv`（run `20261008-133544-m7-a0`；逐次資料在 `a0.csv`）。
- 欄位：x＝`GiBps`（對數軸）；y＝min(`speedup_vs_compute`, `speedup_vs_load`)；每條線一個 `L`（4096、8192、16384、32768）。y＝1 畫一條參考線。
- 可選疊加：`results/m7_explore_mi300x/kappa_screen.csv` 的 `model`∈{llama31_8b, longalpaca7b}、`gibps`、`speedup_vs_best`（**公式值**，虛線，標〔算術〕）。vLLM 的 CPU 路徑頻寬 `summary_c6_vllm_bw.csv` 的 `gibps`（4.02–4.11）畫成直線（這個摘要檔沒有 `run_id` 欄，run_id 在 `c6_vllm.csv`）。
- 圖說要寫：harness 重算比預設 vLLM 慢 1.7–2 倍（`summary_a0.csv` 的 compute_only 對 `results/m9_followup/f1_metrics.csv` 的 off `first_ttft_med`），帶子的絕對位置〔待重量〕。
- 標題：**Cake 只在一段頻寬帶裡明顯有用；重算越快，帶子越往高頻寬移。**

**表 1：延後測試計分板**（C2）
- 第一階段：`results/m7_write_policy_mi300x/summary_b_verdict.csv`，17 列；欄位 `gain_pct`（S5 對最好簡單策略）、`s5_vs_s4plus_pct`（S5 對它的延後雙胞胎 S4+）、`verdict`。
- 08 消融：`results/m7_write_policy_mi300x/b2_verdict.csv`，篩 `src`＝gpu（16 列）；欄位 `strategy`、`best_nwt`、`gain_nwt`、`gain_s4b`、`pass`。
- 第 1 輪 GPU：`results/m7_explore_mi300x/r1_gpu_verdict.csv`，篩 `stat`＝median（5 列）；欄位 `g_nwt`、`g_S4B`、`g_S4W`、`g_S4C`、`pass`。
- 併發模擬（另起一欄，標〔模擬〕）：`results/m8_directions/d1_verdict_cfg.csv`、`d1_posthoc_cfg.csv`、`d1_admitfirst_cfg.csv`、`d1_x4_S5T_cfg.csv` 的 `pass10` 計數（5／3／0／0），加一列 F6：`results/m9_followup/f6_summary.csv` 的 `pass10_rho_lt1`（S5T 0、S1D 0，`cells_rho_lt1`＝216）。**事先登記、事後修正、新 seed 重跑三個判定並列**。
- 重算變快的敏感度（另起一小塊，標〔模擬，F5 重算〕）：`results/m7_write_policy_mi300x/sim_sweep_gain.csv` 依 `f_scale`×`release` 數「`gain_nwt`≥5 且 `gain_s4b`≥5」的格；`sim_probe.csv` 用 ≥3／5 規則的存活設定（3／30）。
- 可改畫成點圖：`results/m8_directions/d3_write_cells.csv`，篩 `is_gpu`＝True（28 列），y＝`lead_WTP_over_NWT`，顏色＝`set`（P1／A08／R1），+5% 畫線。最高 +0.66%。

**圖 2：贏家由回來順序決定**（C3）
- (a) `results/m8_directions/d3_posthoc.csv`，篩 `rule` 開頭是 `PH seed-flip`、`dataset`∈{SIMR1_llama31_8b, SIMR1_longalpaca7b}；長條＝`hits`÷`n`（「5 個 seed 標籤都一樣」的比例）；分組＝`rule` 的尾巴（best_NWT、matters、R3_meas、R2_meas）。重點那根：best_NWT 17／64、8／64。
- (b) GPU：`results/m7_explore_mi300x/r1_gpu_summary.csv`，x＝`wl_seed`（0–4），y＝`median`，每條線一個 `strategy`；標出每個 seed 的最低者（4 種不同贏家）。也可加 `d3_write_cells.csv` 的 P1 列（`wl_seed`、`spread`）：同條件 seed 0／1／2 的策略差 29.5%／32.6%／0.1%。
- 標題：**同樣的硬體常數，只換回來順序，最好的策略就換人。**

**圖 3：「沒有空閒」的區域**（C4）
- 主圖（新 seed，事先登記）：x＝`rho_kv`（10 個 seed 平均），y＝`med_min_gain`（候選對最緊對手的增益，seed 中位數），篩 `metric`＝median。
  - `results/m9_followup/f6_main_S5T_cfg.csv`、`f6_main_S1D_cfg.csv`（各 192 列）。
  - `f6_local8_S5T_cfg.csv`、`f6_local8_S1D_cfg.csv`（各 96 列）。
  - `f6_l8q16_S1D_cfg.csv`（96 列），`pass10`＝True 的 2 點另外標。
  - 畫 y＝0.10 與 x＝1 兩條線。只有 x≈2.1 的 2 點越過 y＝0.10。
- 對照（舊 seed 0–4）：`results/m8_directions/d1_x4_S5T_cfg.csv`、`d1_afq16_S1D_cfg.csv`（同欄位；`rho_kv` 是 5 個 seed 平均）。
- 側表 1（過載角落）：`results/m9_followup/f6_c_overload.csv`，欄位 `d1_n10_of5`、`f6_n10_of10`、`f6_med_min_gain`、`f6_worst_min_gain`、`f6_rho_kv`，分 `metric`＝median／mean／p90。
- 側表 2（真實 vLLM）：`results/m8_directions/d2_summary.csv` 的 `group`、`offered`、`skipped`、`skip_ratio`（主判定 0／133,504）。
- 標題：**單卡、實測參數下 ρ<1，寫入時決定不贏；只在假設的 ρ≈2.1 角落、只在中位數上贏。** 全部〔模擬〕，沒有 GPU 驗證。

**表 2：真實 vLLM 開 CPU 卸載的 kernel 陷阱**（C6，工程發現）
- 重現：`results/m9_followup/f1_repro.csv`（`workload`、`conc`、`metric`、`seed`、`off_s`、`cpu50_s`、`slowdown`）。
- 拆解：`f1_decomp.csv`（`off_s`、`off_triton_s`、`cpu50_s`、`cpu50_rocmfix_s`、`backend_explained_frac`、`rocmfix_removed_frac`）。
- kernel 時間：`f1_profile_summary.csv`（`cfg`、`backend`、`attention_s`、`gemm_s`、`memcpy_s`）。
- 載入 vs 重算：`f1_ret_split.csv`（`cfg`、`group`、`ttft_med`）。
- NFS 層等待：`f1_tier50_summary.csv`（`ret_ttft_max`、`ret_defer_s_max`）。
- 標題：**開 connector 換掉 attention kernel，才是「開卸載反而變慢」的主因。** 放正文還是附錄，看投稿場所的頁數。

## 能耗（不放主圖；§5.1 一段文字）

- `results/m9_followup/f2_verdict.csv`（`accounting`＝A：`tier`、`saving_pct`、`time_cost_pct`）、`f2_probe_summary.csv`（P_b、P_wait）、`f2_cost.csv`（`breakeven_gpu_usd_per_hr`）。一個 seed，只量 GPU。

## 佔位狀態

- F1、F2、F3、F4、F6 都已填入 draft.md（HTML 註解保留追蹤）。
- 還沒解決、標在文中的：〔待重量〕（κ、b、Cake 頻寬帶、f(i)、能耗裡的重算能量）；〔待稽核〕（main.tex 的 κ、headroom、品質數字，本文不用）。
