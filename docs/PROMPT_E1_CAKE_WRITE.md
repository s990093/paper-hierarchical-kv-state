# 執行指令書：E1「依位置寫入」模擬實驗（Cake 延伸方向的 go/no-go）

> 用法：在 MI300X 的 Lab 裡開一個新的 Claude Code session，把下面「==== 開始 ====」到「==== 結束 ====」整段貼給它。
> 背景與動機見 `docs/DIRECTION_CAKE_EXTENSION.md` §3.1、§8、§9。

==== 開始 ====

你是這個 repo 的實驗執行 agent。任務是跑 **E1：容量 × 命中率**，回答一個問題：

> **Cake 讀取時本來就會重算前段。如果寫入時就不存前段、把後段依重算成本放到較快的層，在容量有限時，命中率與 TTFT 能比「全存＋Cake 讀取」好多少？**

這是 Cake 延伸方向的停損點。結果不好就停，不要為了讓方向成立而調參數。

## 0. 開始前必讀（照順序）

1. `CLAUDE.md`：特別是 §1 的不可協商規則（不准編數字、不准跳過失敗、每個數字都要有 run_id、外部 trace 的單位要交叉驗證）與 §4 記錄協定。
2. `docs/MI300X_MLSTEAM.md`：平台 B 的路徑。大檔與 run 原始輸出放 `/mlsteam/data/tiara/runs/<RUN_ID>/`，不要放 repo。
3. `docs/DIRECTION_CAKE_EXTENSION.md`：§2（Cake 做法）、§3.1（要驗證的主張與反面）、§7（風險）、§8（第一步）。
4. `code/m4_oracle.py`：現有模擬器。重點看：
    - `CostModel.cost(tier, position_tokens)`：重算成本隨位置線性成長；
    - `load_cost_model()`、`calibrate_recompute_from_m3()`：成本常數的來源；
    - `mooncake_trace()`：Mooncake 的 `hash_ids` 是 **512-token** block，載入時已有斷言；
    - `--lookup prefix|per-block`：命中語意；
    - `--validate`：模擬器對實測的校驗。
5. `results/m4_oracle_mi300x/cost_model_b-*.json` 與 `simulator_validation.json`：MI300X 的實測成本常數與校驗結果。

讀完後先用 5 行以內回報：你打算新增哪些檔案、要重用 `m4_oracle.py` 的哪些函式。**不要修改 `m4_oracle.py` 既有行為**（其他結果依賴它）；需要的東西用 import 重用，或在新檔案裡寫。

## 1. 要實作的東西

新檔案 `code/m6_cake_write.py`（模擬器）與 `code/test_m6_cake_write.py`（測試）。

### 1.1 Cake 雙向讀取的模型

對一個請求的 context（n 個 block，第 p 個 block 的重算成本 R(p) 用 `CostModel` 算，隨位置變貴）：

- GPU 從 block 0 往後重算，I/O 從 block n-1 往前載入，兩邊平行進行。
- 會合點 m：前 m 個 block 重算，後 n-m 個載入。
  TTFT_restore(m) = max( Σ_{p<m} R(p), Σ_{p≥m} L(tier_p) )，取使其最小的 m。
- L(tier) 是該層的每 block 載入成本（含每筆搬運的固定成本，用實測常數）。已在 GPU 的 block，L = 0。
- **因果約束**：重算 block p 需要 0..p-1 的 KV。所以沒存的 block 一定要落在重算區：m ≥（最後一個沒存的 block 位置）+ 1。
- **退路**：如果只重算或只載入比雙向更快，就取較快的那個。三者取最小值，並記錄用了哪一種。

### 1.2 命中語意

前段不存時，後段在 vLLM 原生的「連續前綴」查找下永遠不會命中。所以本實驗要用「**區段命中**」：block 只要存在任何一層就算命中，沒存的由重算補。

- 這是模擬器假設，不是 vLLM 現有行為。輸出裡要標註。
- P0（現狀）照舊用 prefix 語意，其餘策略用區段命中。報告時要分開說明這個差異的影響。

### 1.3 要比較的策略

E1 只開 **BF16**（不開 FP8／INT4），避免量化效果混進來。INT4 另外在 §3 的風險檢查跑。

| 代號 | 寫入 | 放置／逐出 | 讀取 | 用途 |
|---|---|---|---|---|
| P0 | 全存 | LRU，prefix 命中 | 只載入 | 現狀 |
| P1 | 全存 | 單層（只用最慢層），LRU | Cake 雙向 | **Cake，主要比較對象** |
| P1b | 全存 | CPU＋SSD 多層，LRU | Cake 雙向 | 分離「多層」本身的效果 |
| P2 | 全存 | 多層，從前段逐出（Pensieve 式） | Cake 雙向 | 分離「前段逐出」的效果 |
| P3 | 前段不存 | 單層，LRU | Cake 雙向 | 分離「寫入時不存前段」的效果 |
| P4 | 前段不存 | 依位置分層（越後面越快的層）＋前段逐出 | Cake 雙向 | **本提案** |
| OPT | — | 現有 Oracle（`m4_oracle.py` 的 Bélády／成本貪婪） | — | 上界參考，可選 |

寫入分界 b 的決定：用寫入當下的成本常數，算「如果整段都存在目標層，Cake 會合點會落在哪」，前 b 個 block 不存。**不能用未來資訊**（P4 是線上策略）。

### 1.4 測試（先寫、先過，再跑實驗）

`code/test_m6_cake_write.py` 至少要有：

1. 容量無限時，P3 與 P4 的 TTFT 不得比 P1 差（寫入分界等於會合點，讀取不受影響）。
2. 容量無限、且所有 block 都存時，P1 的會合點要跟手算的小例子一致（例如 n=4，自訂 R 與 L）。
3. κ 極大（載入極便宜）時，b → 0，P3 退化成 P1。
4. 因果約束：構造一個「中間有一格沒存」的例子，會合點必須在它之後。
5. Mooncake 單位：`mooncake_trace()` 載入後，用請求長度欄位驗算 `hash_ids` 數 × 512（CLAUDE.md 第 6 條）。

同時跑既有的 `code/test_m4_regression.py`，確認沒有被你弄壞。

## 2. 要跑的掃描

全部用 MI300X 的實測成本常數（`--device local`）。

1. **模型（κ 的兩端）**：從 `results/m4_oracle_mi300x/cost_model_b-*.json` 找出 κ 最小與最大的兩組模型／長度設定，在 RUNLOG 寫明是哪兩個、κ 實際多少。**不要沿用文件裡的 0.72／11.65，要自己算出來並對照**；對不上就停下來回報。
2. **Trace**：`mooncake`（真實）為主，`conversation`／`toolagent`（Zipf 合成）為輔。
3. **容量壓力**：用 `--pressure` 的方式掃，至少 5 點，從「全部放得下」到「只放得下 10–20%」。
4. **種子**：至少 3 個 seed，報平均與範圍。

每格輸出：
- 命中率（block 層級）、整段重算的請求比例；
- TTFT p50／p99／平均（模擬值）；
- 每請求寫入位元組、各層峰值使用量；
- 讀取時走雙向／只重算／只載入的比例；
- 相對 P1 的變化（%）。

## 3. 風險檢查（同一支腳本，另跑一輪）

1. **INT4 會不會吃掉效益**：允許 GPU 上 FP8／INT4（品質門檻照 `load_quality()` 的既有結果），重跑 P1 與 P4。記錄 P4 相對 P1 的優勢剩多少。
2. **E4 穩健性**：寫入決定固定，讀取時把載入頻寬與可用算力各乘 0.5 與 2，重算 TTFT。記錄 P4 比 P1 差的情況占多少、退路觸發幾次。
3. **SSD 敏感度**：MI300X 沒有本地 NVMe（`local` 是 overlay）。另外用平台 A 的 `nvme` 常數跑一組，**標為敏感度分析，不是 MI300X 量測**。

## 4. 判準（照實寫，不要美化）

主要比較：**P4 vs P1**，在容量吃緊（命中率 < 80%）的壓力點上。

| 結果 | 判定 |
|---|---|
| TTFT（p50 或平均）改善 ≥ 15%，或寫入量／容量省 ≥ 30% 且 TTFT 不比 P1 差超過 2% | 🟢 GO：可以進 E2（真實寫入干擾量測） |
| 改善 5–15% | 🟡 停下來，把數字交給人判斷 |
| 改善 < 5%，或只在不切實際的壓力下才成立 | 🔴 NO-GO：這個方向停止，結果寫成「何時不值得做」 |

另外分別回報 P1b、P2、P3 相對 P1 的改善，說明 P4 的好處主要來自哪一項（多層、前段逐出、不存前段）。如果 P2（Pensieve 式）就拿走大部分好處，要明講，這直接影響新穎性。

## 5. 記錄與產出

- 每次 run 照 CLAUDE.md §4.1 的殼：`RUN=/mlsteam/data/tiara/runs/<RUN_ID>`，存 `cmd.sh`、`context.txt`（含 `rocm-smi`、repo HEAD）、stdout／stderr、`exit_code`。
- 量測期間不需要 GPU（純模擬），但 `env_fingerprint` 仍要記。
- 小結果放 `results/m6_cake_write_mi300x/`：
    - `e1_summary.csv`、`e1_risk.csv`：每列都要有 `run_id`、`ts`、`model`、`kappa`、`trace`、`pressure`、`seed`、`policy`；
    - `e1_verdict.json`：判準、數字、判定。
- `results/RUNLOG_MI300X.md` 補一段，格式照 CLAUDE.md §4.2，「與文件假設的差異」一欄要對照 `DIRECTION_CAKE_EXTENSION.md` §3.1 的算術上限（κ 小約 2.4 倍、κ 大約 1.09 倍）。
- **不准改 `main.tex` 或 `docs/DIRECTION_CAKE_EXTENSION.md`**。結論寫在 results 與 RUNLOG，文件由另一個 session 根據結果更新。
- commit 訊息用 `results(m6): ...`；程式用 `infra(m6): ...`。測試通過、一輪掃描完成後就 commit 並 push，不要累積。
- audit 用 self-review 時，RUNLOG 要寫明「此次 audit 為 self-review，非 cross-model」。

## 6. 遇到這些情況要停下來回報，不要繞過

- `--validate` 校驗不過，或 `simulator_validation.json` 顯示模擬器與實測對不上；
- 成本常數檔缺少或讀不到（不准用預設值）；
- 算出的 κ 與文件差很多；
- 任何測試失敗；
- 結果落在 🟡 區間。

## 7. 最後回報的格式

1. 判定（🟢／🟡／🔴）與一句話理由；
2. 主表：兩個 κ × 壓力點 × 策略的 TTFT 與命中率（相對 P1 %）；
3. 好處拆解：P1b／P2／P3 各貢獻多少；
4. 風險檢查三項的結果；
5. 所有 run_id 與輸出檔路徑；
6. 沒做到或不確定的地方。

==== 結束 ====

## 備註（給人看，不用貼）

- E1 是純模擬，大約一到兩天可以完成；只有 🟢 才值得花時間做 E2（MI300X 上用 vLLM `OffloadingConnector` 實測寫入與載入互相干擾）。
- E1 開始前，建議另外用 `novelty-check` skill 重做一次查新（上次是 9/24），確認沒有人已經做掉「寫入時依位置決定＋多層」這個組合。
- P2（Pensieve 式）是最需要注意的對照：如果它就拿走大部分好處，新穎性會變得更窄。
