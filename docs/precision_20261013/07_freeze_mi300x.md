# 07　事先登記凍結版（MI300X 主實驗）

> 回到 [README](README.md)。依據：[03](03_plan_h200.md) §1–§2，加上開頭的「10/15 改動」與「10/15 晚改動」。這一頁把所有要寫死的**具體數值**收在一起。
> **凍結時間**：2026-10-15T16:13:06Z（`date -u`）。**到這時為止，沒有看過任何低精度配置的答對率**（見 §7「凍結前已經看過的東西」）。
> 凍結之後要改任何一項：另起一節寫「改了什麼、為什麼、在看到什麼結果之前或之後」，不覆蓋這一頁。

## 1. 機器與軟體

| 項目 | 值 | 來源 |
|:--|:--|:--|
| GPU | AMD MI300X（MLSteam），單卡 192 GB | `REPORT_MI300X.md` |
| 軟體 | torch 2.10.0+rocm7.0（HIP 7.0.51831）、transformers 5.17.0、venv `/mlsteam/workspace/venv/tiara` | 凍結時印出 |
| 程式 | 這一頁的 commit；判定程式 `code/m10_analyze.py`（commit 0e79da7 起，selftest 過） | git |
| 記錄殼 | `m7run`／`runsh`（`$TIARA_RUNS/<run_id>/`），GPU 工作排成佇列、每段拿 `gpu.lock` | `code/m10_queue_mi300x/`（跑完後放進 repo） |

## 2. 模型、題目、生成

| 項目 | 值 |
|:--|:--|
| 模型 | Llama-3.1-8B-Instruct（`unsloth`，snapshot `4699cc75…`）；Qwen2.5-7B-Instruct-1M（snapshot `e28526f7…`）；都是 BF16 權重 |
| 長度 | context 64 個 block × 512 token＝32,768 token，問題與答案接在後面 |
| 難度參數 | `results/m10_pilot_mi300x/e0_tune/CANDIDATE_PARAMS.json`（sha256 `00c1034f…74412e2`） |
| 題目 | seed 10000 起。T1–T4 各 180 題（三種輪替各 60），T5 60 題，共 780 題；LongBench 6 個資料集各 50 題（只做讀法 A） |
| 主判準的任務 | T2–T5（T1 只報告） |
| 生成 | greedy；T5 最多 120 token、T1–T4 最多 64、LongBench 用官方 `dataset2maxlen` |

## 3. 低精度格式

20 題小測（seed 4000 起，T2–T5 各 5 題），只看 E1（讀法 A）對 E0 的平均 KL，每個模型、每一階各選最小的（`results/m10_precision_mi300x/fmt_select/fmt_choice.json`）。

| 模型 | FP8：head／channel／token 的平均 KL | **選定 FP8** | INT4：token／kivi 的平均 KL | **選定 INT4** |
|:--|:--|:--|:--|:--|
| Llama | 0.00434／0.00412／0.00236 | **fp8_token** | 0.0283／0.00521 | **int4_kivi** |
| Qwen | 3.401／8.816／11.780 | **fp8_head** | 10.47／0.0125 | **int4_kivi** |

- 每格 35 列（20 題、T2–T4 兩輪＋T5 一輪）。run_id：`20261015-160426-m10-fmtsel-llama-s0`、`-160541-…-llama-s1`、`-160656-…-qwen-s0`、`-160811-…-qwen-s1`。
- 所以 **E1-K0**：Llama＝E1s@fp8k0_token、Qwen＝E1s@fp8k0_head；**E1-INT8**＝E1s@int8_token（兩個模型）。

- **E1-K0**＝E1s@fp8k0_〈選定粒度〉；**E1-INT8**＝E1s@int8_token（Step 2b，block 0 BF16）。
- 扣平均後的 K 重建誤差（彩排量的，只報告）：Llama fp8_head／channel／token＝4.18／4.16／3.74%；Qwen＝18.0／12.4／18.5%。

## 4. 速度

| 項目 | Llama | Qwen | 來源 |
|:--|:--|:--|:--|
| f(i) 總和（64 個 chunk，GPU 閒） | 3.631 s | 3.077 s | `speed_<m>/bw.json` |
| 真實 pinned H2D | 49.9 GiB/s | 48.6 GiB/s | 同上 |
| 限速 b16／b32／b48（GiB/s） | 5.275／1.467／0.420 | 2.771／0.764／0.218 | 同上（B＝(64−b*)×chunk÷Σ_{i<b*} f(i)） |
| CPU 錨點 | 不限速、真的 H2D | 同左 | |
| GPU 忙 | duty 0.7（第一階段 busy2） | 同左 | `results/m7_write_policy_mi300x/busy_levels.json` |
| δ（E0，4 頻寬 × 閒／忙，3 process × 10 次；run `…-154722-m10-noise-llama`、`…-155626-m10-noise-qwen`） | 2.97%（最大的格子：b48 閒） | 9.43%（最大的格子：b16 忙，3 個 process 的中位數 0.609／0.554／0.611 s） | `speed_<m>/noise.json` |
| **δ\*＝max(5%, δ)** | **5%** | **9.43%** | 同上 |
| 速度格子 | 8 配置＋只算＋只載（E0、E1 的存法）× 4 頻寬 × 閒／忙，3 process × 10 次，配置順序隨機打散；另加 8K、16K（E0、E1、B25、R25，閒） | 同左 | `code/m10_speed.py grid` |
| 讀法 B 的會合點 | 速度格子（32K、cake、沒被標污染的列）每個配置在 b32 閒／忙的會合點中位數 | 同左 | `speed_<m>/meets.json` |

## 5. 判準與判定流程（照 `code/m10_analyze.py`）

- **ε（非劣性）**：主任務 T2–T5 所有列（兩輪合併），X 對 E0 的配對正確率差；以「題」為單位重抽 10,000 次（seed 0），95% CI 下界 > −2 個百分點；**而且**每個主任務族、每一輪的點估計差 ≥ −5 個百分點。
- **檢定力**：E1 對 E0（讀法 A）的 CI 半寬 > 1.5 個百分點 → 先加題，不判「沒過」。
- **情境**：A（強制全載入）、B-閒、B-忙、B-閒→忙（第 2 輪換成忙再還原一次）；B 都用 b32。
- **速度不輸**：每個（頻寬, GPU 狀態）格子，X 的還原時間中位數 ≤ 最快的混合配置（F25、M25、B25、R25）×（1＋δ*）。
- **K0**：E1 在每一格都沒有「贏」E0（中位數差 > δ* 且配對差的 bootstrap CI 不含 0）→ 精度對還原時間沒有賺頭 → 停掉位置策略。用速度格子裡的 E0、E1 配對列算（`m10_speed.py k0-verdict`）。
- **K1**：E1 四種情境都 ε 過、速度不輸。**K1s**：K1 不成立，E1s 或 E1sr 四種情境都過、速度不輸。「模型過」＝K1 或 K1s。
- **兩個模型合併**：03 §1.1 的表（`m10_analyze.py decide`）。
- **K3**（FP8 階）：E1-K0 或 E1-INT8 四種情境都 ε 過、速度不輸（Step 2b 的速度是另一次 grid，跨 run 比，報告寫明）。
- **第 2 階（INT4）**：第 1 階「不需要位置策略」才走；判準、題目、ε、δ* 完全相同。
- **K2／穩定差距**（Step 2）：F25、M25、B25 對 R25，讀法 B（b32 閒或忙）主任務的品質差 ≥ 2 個百分點、配對 CI 不含 0、還原時間不比 R25 慢超過 δ*；方向在 T1–T4 至少 2 族一致；新 seed 重跑重現。都不成立 → K2：位置沒有貢獻。

## 6. 先跑、封存的工作（不改判準、不改判定順序）

為了不讓 GPU 空等（使用者 10/15 要求），下面這些在判定流程走到之前就先跑，結果放 `results/m10_precision_mi300x/quality_sealed/`，**判定流程走到才分析**；沒走到的照實記錄「跑了、沒分析」：
- Step 2b 的 E1-K0、E1-INT8（和 Step 1 同一批題、同一個 process，列寫在 Step 1 的 CSV 裡，但 Step 1 的判定只讀 E1、E1s、E1sr）
- 第 2 階 INT4 的速度格子、端點（E1、E1s、E1sr）、混合配置（F25、M25、B25、R25）
- 03 §3.3b 逐 block 敏感度地圖（只報告）

## 7. 凍結前已經看過的東西（照實揭露）

- E0（全 BF16）的答對率：彩排調難度（seed 1000–、2000–）。
- K／V 分布與各格式的重建誤差（彩排）。
- 速度彩排（Qwen、縮小規模）：b32 時 E1 比 E0 快 32%（閒）、38%（忙）；CPU 層 E1 反而慢 6–10%。這是 K0 的方向，K0 的判準沒有因此改。
- 這一輪的 f(i)、頻寬、δ（§4）。
- 20 題小測的 KL（只用來選格式；答對率欄沒有讀、沒有印）。看到的是：Qwen 的 FP8 三種粒度 KL 都很大（3.4–11.8），INT4-KIVI 只有 0.0125；Llama 全部 < 0.03。判準和判定程式在這之前就已 commit（0e79da7），沒有因此改。
- 兩個 smoke test（scratch，不進 results）：`20261015-160602-m10-smoke-bpath`（讀法 B 只取 b32 的列形狀，`--blind`）、`-160602-m10-smoke-sensmap`（敏感度地圖 64 列；看過第一列的 KL 數字，只確認欄位）。

## 8. 和 03 的差異（全部在凍結前決定）

見 03 開頭的「10/15 晚改動」：主平台 MI300X、§R 不做、§5.2 這輪不修、讀法 B 只用 b32、K0 從速度格子算、先跑封存。
