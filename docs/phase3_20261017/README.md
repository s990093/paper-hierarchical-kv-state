# 第三階段計畫：延後決定時，預測「誰會回來」值多少？（草稿，2026-10-17）

> **狀態**：草稿，等你看過。§8 的判準還沒有凍結（凍結＝寫進 `01_prereg_p3.md` 並 commit 之後才跑 P3-3）。
> **10/17 更新（09:35Z）**：使用者要求先大量探索、找出問題；同日的零前文審查（[`review_opencode_20261017.md`](review_opencode_20261017.md)）指出 §8 有多個漏洞。**§8 暫緩凍結**，先跑 [`explore/README.md`](explore/README.md) 的 9 個方向，收斂後再改寫 §1、§8。
> **分支**：`p3-predictor`（說明見 §11）。**平台**：先 3090，之後 MI300X／H200 只換成本常數重跑。
> **標記**：〔實測 / 模擬 / 算術 / 原文 / 判讀 / 估計〕同 repo 慣例。本文**沒有任何第三階段的結果**；引用的數字都來自第一、二階段，附出處。
> 文獻與 trace 的查證由另一個 agent 進行中，寫在 [`research/LIT_AND_TRACES.md`](research/LIT_AND_TRACES.md)；本文標「待 research」的地方等它回來再補。

---

## 0. 一頁摘要

**現在能不能開始第三階段？可以。**
- 第一階段（放置）已收斂：36 組 GPU 比較，寫入時規則 0 組同時贏過最好的位置無關策略和自己的延後雙胞胎〔main.tex §4.2〕。
- 第二階段（精度）主線判定流程已走完：FP8 兩個模型都不需要位置策略；INT4 統一格式在 32K 兩個模型都守住 2 分界線；25% INT4 放前／中／後和隨機一樣（K2）〔RUNLOG_MI300X 10.12、main.tex §5〕。
- MI300X 還在跑的都是**只報告**或**長 context** 的工作（stage 42–46、44 兩層還原、64K／128K）。第三階段只有兩處會用到它們，§5.3 列出、先用暫定值、結果出來再換。

**第三階段要回答的一個問題**：
> 論文的設計是「寫入簡單、滿了才降級、讀取時決定」。延後的決定仍然要選**降級誰、何時把誰預取回來、丟掉誰**——這些選擇都在猜「誰會回來、何時回來」。
> **在讀取端會自適應（Cake）、降級時才量化的設計下，完美知道回來時間能比 LRU 快多少？學得到其中多少？用什麼 loss 學最好？**

**為什麼這個問題接得上前兩步**：
1. 第一階段發現**贏家由回來順序決定，不是由硬體常數決定**（同條件只換 seed，Llama 64 組只有 17 組贏家不變）〔D3；main.tex §4.4〕。回來順序正是預測器要猜的東西。
2. 第一階段的「延後測試」可以原封不動套到預測上：**在寫入時（回合結束）預測一次，vs 在要決定時（降級／預取當下）用最新狀態重新預測**。這是 §1 的 Q3。
3. Cake 讓「放錯層」的代價變小（慢層的後段可以和前段重算重疊）。所以 Cake 可能**縮小**預測的價值。這是 §1 的 Q1b，目前查到的前作都沒問過（待 research 確認）。

**最重要的停損點（CLAUDE.md §1 規則 4 的精神）**：先用模擬算 oracle headroom（P3-3，只用 CPU）。`>15%` 才開始訓練；`5–15%` 停下來問你；`<5%` 停，寫成負面結果。
- 已有線索指向「空間可能很窄」：D4 在 Mooncake 上，完美知道重用的 oracle 對最好的延後版，TTFT 差距中位數只有 0.015%／0.002%；只有 CPU ≤10% 工作集時才有差：延後版若不用 SSD 慢 8.6–12.5%，用 SSD 則要多寫約 10 TiB；慢層換成 NFS 參數時 oracle 快 7.0–17.5%〔模擬 D4 §0〕。但 D4 沒有做**預取**、沒有**慢層 INT4**、指標是全部請求的 TTFT 總和（含所有策略都一樣的第一次重算）。P3-3 補這三點。

**你問的「主要訓練、找好 loss」**：放在 P3-4。CLAUDE.md §8 規定 **oracle 出來之前不准訓練任何模型或調超參數**，所以順序是：資料與模擬器的採坑（P3-1、P3-2）→ oracle 判定（P3-3）→ 訓練與 loss 比較（P3-4、P3-5）。採坑階段可以用**合成資料**把訓練管線整條跑通，結果只放 scratch、不進 `results/`。

**3090 多卡**：可以，但用途是「一張卡一個 job、平行掃設定」與「小模型訓練」，不是 tensor parallel（CLAUDE.md §3）。P3-1～P3-5 主要吃 CPU（96 核、440 GB RAM）；GPU 用在 3090 成本常數重量（P3-2a）、內容型預測器訓練（P3-7）、GPU 驗證（P3-6）。

---

## 1. 要回答的問題

| # | 問題 | 怎麼答 | 對應 milestone |
|:--|:--|:--|:--|
| **Q1** | 延後設計下，完美知道回來時間比 LRU 類策略快多少？ | 模擬：oracle vs 最好的非 oracle 延後策略，掃容量、慢層頻寬、GPU 忙閒 | P3-3（停損點） |
| Q1a | 這個空間來自哪一種預測？降級順序、預取時機、還是丟不丟？ | 把 oracle 拆成三個元件，一次只開一個 | P3-3 |
| **Q1b** | Cake 會不會縮小預測的價值？ | 同一格用 Cake 還原 vs 只載入還原，各算一次 headroom | P3-3 |
| Q1c | 預測要多準才有用？需要的是「會不會回來」「在 H 秒內回不回來」還是「確切順序」？ | 部分資訊的 oracle（只知道會不會回來、只知道 H 內）＋加雜訊的 oracle | P3-3 |
| **Q2** | 哪一種訓練目標／loss 變成的決策最好（不是 AUC 最好）？ | 同特徵、同模型族，只換 loss；看 regret 與拿到的 headroom 比例 | P3-4 |
| **Q3** | 預測在寫入時做一次，vs 決定當下重新做，哪個好？ | 延後測試套在預測上 | P3-5 |
| Q4 | 預測失效時會不會比 LRU 差很多？換 trace 還能用嗎？ | 退回 LRU 的機制、跨 trace、雜訊注入、冷啟動 | P3-5 |
| Q5 | 模擬的結論在 GPU 上成立嗎？換硬體呢？ | 3090 harness 多 session 重播；之後 MI300X／H200 換常數 | P3-6 |

Q1c 直接決定 Q2 要比哪些 loss：如果「只知道會不會回來」就拿到大部分空間，分類（BCE）就夠；如果需要確切順序，就要排序或存活分析的 loss。**先量需求，再挑 loss**，不要反過來。

---

## 2. 預測器驅動的決策（動作空間）

設計固定為論文 §6.1 的建議，第三階段**不加任何寫入時動作**（第一階段已判死）：

```
寫入：每個 block BF16 進 CPU（不決定）
                │
   CPU 滿了 ──▶ D1 降級：選哪個 session 搬到慢層（搬的時候量化成 INT4）   ← 預測：誰最晚回來
   慢層滿了 ──▶ D2 丟掉：選哪個 session 整個丟、回來時重算               ← 預測：誰不會回來
   閒置時   ──▶ D3 預取：選哪個慢層 session 先搬回 CPU、搬多少             ← 預測：誰快回來、多快
回來時    ──▶ Cake 還原（讀取時決定重算多少，不需要預測）
```

- **D3 預取多少**可以用到第一階段的位置結論：從慢層還原時，前段反正會被 Cake 重算，所以只需要預取會合點之後的後段（「部分預取」）。這是位置資訊在延後設計裡唯一的新用途，要和「整份預取」比。
- **回來時 GPU 多忙**（第一階段 02 §9 的第三個預測目標）影響會合點：GPU 忙 → 會合點往前 → 要載入的 block 變多 → 預取要多搬。先當特徵與敏感度，不單獨做預測器，除非 P3-3 顯示它值得。
- 慢層存 INT4 是第二階段的結論（慢層省 36–57% 還原時間、容量 3.2 倍〔main.tex §5.1、§6.3〕）；同時保留「慢層存 BF16」當敏感度。

---

## 3. 對手與 oracle（P3-3 開跑前凍結）

| 類別 | 策略 | 代表誰 | 用到預測嗎 |
|:--|:--|:--|:--|
| 位置無關、看近期 | **LRU** | vLLM／LMCache／Mooncake／SGLang HiCache 的預設 | 否（只看閒置時間） |
| | LRU＋2-hit 准入慢層 | SGLang `write_through_selective`、Dynamo KVBM | 否 |
| | ARC、S3-FIFO | 經典強基準 | 否 |
| 看成本 | Pensieve 式（重算成本 ÷ 閒置時間） | Pensieve；第一階段已有 | 否 |
| 簡單預測（老師說的「強 baseline＋簡單 heuristic」） | **WA 查表**：請求類別×第幾輪×閒置時間 → 回來機率 | KVCache-in-the-wild（ATC'25） | 查表，不訓練 |
| | 上一輪輸出長度 → 下次間隔 | Bidaw（FAST'26） | 一條規則 |
| | 工具名稱的歷史時長 → TTL | Continuum（agent trace 才有） | 一條規則 |
| 學習式（P3-4 以後） | LightGBM＋退回 LRU | LARU（2509.20979） | 是 |
| | 文字嵌入＋MLP＋時間衰減 | LPC（NeurIPS'25） | 是（要有文字） |
| 預取的對手 | 不預取（回來才還原＝Cake 原樣） | Cake | — |
| | 佇列看得到才預取（零提前量） | CachedAttention | 否 |
| | 閒置時依近期把 CPU 填滿 | 簡單 heuristic | 否 |
| **Oracle** | Belady 降級＋oracle 丟掉＋oracle 預取（提前量＝還原時間） | 上限 | 完美 |
| 部分 oracle（Q1c） | 只知道「會不會回來」／只知道「H 秒內回不回來」／確切時間加 log-normal 雜訊 σ | 需求分析 | — |

- 「最好的非 oracle 策略」＝每一格裡上表非 oracle 列中最快的那個。**對手集合事先寫死**；第一階段 D1、D4 都踩過「看到候選贏了才加對手」（main.tex §6.4「Post-hoc rivals」），這次不再犯。
- WA 查表與 Bidaw 規則是「不訓練的預測」，放在非 oracle 那一邊。這樣 P3-3 的 headroom 已經扣掉簡單 heuristic 能拿到的部分，符合老師 9/19 的要求（ADVISOR_REPLY_20260919 §「strongest baseline 加一個簡單 heuristic」）。

---

## 4. 工作負載與資料

**原則**：學習的主張只能用**真實回來間隔**的 trace；合成資料只用來測機制、測程式。Bidaw 把時間戳換成 Poisson 之後效果消失〔D4 §5 表，原文 p.12〕；9 月 M5 的 B11 也發現獨立事件太少時「學習」是假的〔PAPER_DELTAS B11〕。

| trace | 用途 | 狀態 |
|:--|:--|:--|
| Mooncake conversation／toolagent | 真實時間戳（毫秒、解析度約 1 s）、512-token block、前綴鏈可重建 session〔D4 §4.1〕；58.9 分鐘、12,031／23,608 請求 | 重新下載中（環境 agent） |
| Qwen-Bailian（ATC'25） | 若有 chat／parent／turn 欄位，是最直接的「回來」標籤 | 待 research 確認欄位 |
| CacheWise（Claude Code 使用者） | 真實 agent 工作階段；只有 94 個 session〔PAPER_DELTAS〕，獨立事件少 | 下載中；只當 agent 情境的補充 |
| WildChat | 若每輪有時間戳＋文字，是內容型預測器（P3-7）唯一的來源 | 待 research；可能要你在 HF 同意條款 |
| 第一階段的 chat／doc 工作負載 | 可控、和第一階段 GPU 結果相同的設定；回來順序由 seed 決定 | repo 裡有（`m7_write_policy.make_workload`） |
| 合成（P3-1 新寫） | 已知的回來分布（快回來的工具呼叫／幾分鐘的人／不回來，三群混合），用來驗證 oracle 與預測器行為 | 待寫 |

**長度**：真實 trace 多半短（Mooncake 中位約 6.3K〔HANDOFF〕）。要讓慢層與 Cake 有作用，可能要把 session 的 context 長度映射到 8K–32K。這是**合成變換**，結果一律和「原始長度」分開列，不能混成一個數字。

---

## 5. 模擬器與成本常數

### 5.1 重用什麼

| 東西 | 來源 | 驗證狀態 |
|:--|:--|:--|
| Cake 兩條線的還原模型 | `m7_sim.sim_restore`、`m8_conc_sim.restore_v2` | 會合點 28／28 格在一個 block 內〔main.tex §2.2〕 |
| 寫入／降級的狀態機、限速層 | `m7_write_policy.BState`、`m7_restore_harness.Tier` | 第一階段 114 格誤差中位 0.4%、最大 5.0%；**策略之間的差距誤差約 5–6 個百分點**〔main.tex §3.3〕 |
| trace 載入與單位斷言、Belady、ARC、S3-FIFO、2-hit | `m8_trace_oracle.py`（D4） | 單元測試（D4 §3「策略的檢查」） |
| 併發、到達過程、GPU 排隊、SSD 寫入佇列 | `m8_conc_sim.py`（D1） | **沒有 GPU 驗證**〔main.tex §3.3〕 |

新寫 `code/p3_sim.py`：import 上面這些，不改它們（第一、二階段的結果要能逐位元重現）。新增的只有：session 層的降級／丟掉／預取、慢層 INT4、部分預取、預測器介面（每次決定時呼叫，傳入當下狀態）。

**不變量（每次 sweep 自動檢查，比照舊 `m4_invariants`）**：
1. 預測器換成「上次存取時刻」時，`p3_sim` 必須**逐位元**等於 LRU。
2. oracle 不慢於任何策略；部分 oracle 介於 LRU 與完整 oracle 之間（不保證，但違反要報告）。
3. 容量單調：任一層加大，oracle 不變慢。
4. 關掉預取、慢層 BF16、單一 session 時，逐格重現第一階段 `m7_sim` 的數字。

### 5.2 成本常數

| 常數 | MI300X（已量） | 3090 |
|:--|:--|:--|
| 重算 f(i) | 第一個 block 28.9 ms、block 80 為 98.8 ms，約 0.89 ms／block〔main.tex §2.2〕 | NOT_MEASURED → P3-2a |
| CPU 層載入 | 1.76 ms／block（35.4 GiB/s）；pinned H2D 原始 52.9 GiB/s〔main.tex §2.2、§3.2〕 | NOT_MEASURED（PCIe 4.0）→ P3-2a |
| 慢層 | 限速層；本地 SSD 6.98／1.91 GiB/s（讀／寫）、NFS 0.331／0.601〔main.tex §3.2〕 | 沿用限速層的設定值（不是裝置結論） |
| 寫入干擾 | SSD 3.31×、NFS 1.67×、CPU 1.16×〔main.tex §3.2〕 | 沿用 |
| GPU 忙 | 背景 matmul 0.7 duty → 重算慢 2.07×〔main.tex §3.4〕 | NOT_MEASURED → P3-2a |
| INT4 反量化 | 表 VI：CPU 層會變慢〔main.tex §5.1〕 | NOT_MEASURED → P3-2a |
| harness vs vLLM | harness 重算慢 1.7–2.0×〔main.tex §3.1〕 | NOT_MEASURED |

**跨硬體是免費的附帶結果**：同一個 P3-3 用 MI300X 常數和 3090 常數各跑一次，就能回答「預測的價值隨硬體變不變」。第一階段說贏家主要看回來順序、不看常數（D3）；如果 headroom 也不隨常數變，這句話就延伸到預測。

### 5.3 會用到的第二階段未完成結果

| 等什麼 | 用在哪 | 暫定做法 |
|:--|:--|:--|
| stage 44：CPU 存 BF16、慢層存 INT4 的兩層還原實測 | 模擬器的混合層還原時間 | 先用 `m10_tier2` 的推算〔推算〕，44 出來後換實測、重跑 P3-3 |
| 64K／128K 的 INT4 品質（10 號計畫） | 慢層 INT4 在長 context 能不能用 | P3 先限 ≤32K；長 context 結果出來再決定要不要延伸 |

---

## 6. 預測器與 loss（P3-4 的內容，現在只寫設計）

### 6.1 學什麼

- **單位是「session 的一次回合結束」，不是 block**（9 月 M5 的 B11：一份 131K 文件的 8,192 個 block 只帶一個事件的資訊）。
- **標籤**：同一 session 下一輪的到達時間 T（相對於上一輪結束）；trace 結束前沒回來＝**右設限**（censored），不是「T＝視窗長度」。9 月 M5 的 B5 就是把設限當成同分，排序整個壞掉（AUC 0.9995、正樣本內 Spearman 只有 0.233）〔PAPER_DELTAS B5〕。
- **只用負載訊號**（phase1 02 §9）：輪數、上一輪輸入／輸出長度、session 歷史間隔、目前已閒置多久、請求類別／工具名（有的話）、時段、整體到達率。文字內容只在 P3-7。
- 切分：**依時間**（前段訓練、後段測試）且**依 session**（同一 session 不跨切分）。LPC 的程式碼是對逐輪樣本隨機切，同一對話的不同輪可能同時落在訓練與驗證〔E06 卡片〕，我們不這樣做。

### 6.2 loss 候選（同特徵、同模型族，只換 loss）

| # | loss | 預測什麼 | 怎麼處理「不回來」 | 為什麼要比 |
|:--|:--|:--|:--|:--|
| L0 | 不訓練：LRU、WA 查表、Bidaw 規則 | — | — | 下界與簡單 heuristic |
| L1 | BCE：H 秒內回不回來 ＋ 事後移門檻（Elkan） | 機率 | H 內沒回來＝0 | 9 月 M5 的 B6：移門檻贏過成本加權訓練 |
| L2 | log T 迴歸，設限值補成視窗長度 | 點估計 | 補值（錯的） | **負對照**：重現 B5 的失敗 |
| L3 | 存活分析：Cox、AFT（log-normal）、離散時間 hazard | 整條 S(t) | 正確處理右設限 | 直接修 B5；S(t) 也給預取的提前量 |
| L4 | 排序：LambdaRank／pairwise，組＝每次降級當下在 CPU 的 session | 順序 | 不回來排最後 | 策略吃的就是順序（B5 的建議） |
| L5 | L4 加權：每一對的權重＝排錯時在 Cake 成本模型下多出的 TTFT | 順序 | 同 L4 | 成本感知；要和 L1＋移門檻正面比（B6 的教訓） |
| L6 | pinball（分位數）：預取提前量 | T 的 q 分位 | 設限要另外處理 | 和 L3 推出的分位數比，看專門的 loss 有沒有用 |
| L7（可選） | 決策導向（SPO+ 或模擬器的平滑 regret） | — | — | 只有 L1–L6 留下明顯差距才做 |

- 模型族：LightGBM／XGBoost（CPU、微秒級推論；XGBoost 內建 `survival:cox`、`survival:aft`）與小 MLP（GPU，loss 好改）。**換 loss 時模型族固定**，換模型族時 loss 固定。
- 決策介面固定兩種，讓不同 loss 能公平比：(i) 只用順序（降級預測最晚回來的）；(ii) 用機率／分布算「降級的期望代價 ÷ 位元組」（LHD／Greedy-Dual 的形式）。只輸出順序的 loss（Cox、LambdaRank）只能用 (i)。
- **退路**：分數離散度太低，或最近被降級的 session 很快回來（LARU 的誤判訊號），就退回 LRU。9 月 M5 的 B10：沒有退路時比 LRU 差 2.7–5.6 倍〔PAPER_DELTAS B10〕。

### 6.3 指標

- **主指標**：回來請求的 TTFT（中位、平均、p90）對 oracle 的 regret，與「拿到的 headroom 比例」＝(T_最好基準 − T_策略)／(T_最好基準 − T_oracle)。
- 次指標：慢層寫入量、預取位元組、CPU 占用。
- 診斷（不拿來判勝負）：C-index、time-dependent AUC、integrated Brier、校準、正樣本內 Spearman、LRB 的 good decision ratio。9 月 M5 的教訓是 AUC 高不代表策略好（B5、ADVISOR_REPLY 發現 19）。
- 統計：每個 trace 切成多個不重疊的時間段當重複；成對 bootstrap（10,000 次）；至少 3 段。

---

## 7. Milestones

| # | 內容 | 硬體 | 產出 | 關卡 |
|:--|:--|:--|:--|:--|
| P3-0 | 3090 環境、模型、資料（環境 agent 進行中） | — | `docs/ENV_3090.md`、`results/RUNLOG_3090_P3.md` | 冒煙測試全過 |
| P3-1 | trace 載入（單位斷言、session 重建驗算）、回來間隔描述統計、合成產生器 | CPU | `results/p3_traces/` | 每個 trace 的「用 A 欄驗算 B 欄」斷言過（規則 6） |
| P3-2 | `p3_sim.py`＋不變量；P3-2a 量 3090 成本常數（f(i)、H2D、GPU 忙、INT4 反量化） | CPU；P3-2a 一張空閒卡數小時〔估計〕 | `results/p3_sim/`、`results/p3_calib_3090/` | §5.1 四條不變量全過 |
| **P3-3** | **oracle headroom（停損點）**：Q1、Q1a、Q1b、Q1c | CPU | `results/p3_oracle/`、判定 JSON | **§8：>15% GO／5–15% 問你／<5% NO-GO** |
| P3-4 | loss 比較（L0–L6），LightGBM＋MLP | CPU＋1–2 張卡（MLP） | `results/p3_predictor/` | 只有 P3-3 GO（或你同意的 MARGINAL）才做 |
| P3-5 | 寫入時 vs 決定時預測；退路；跨 trace；雜訊；冷啟動 | CPU | 同上 | — |
| P3-6 | GPU 驗證：前 2 名＋LRU＋oracle 在 3090 harness 上多 session 重播；之後 MI300X／H200 | 多張卡平行（一卡一設定） | `results/p3_gpu_3090/` | 模擬與 GPU 的策略排序一致；差距誤差在第一階段的範圍內 |
| P3-7（可選） | 內容型預測器（LPC 式：凍結 e5-small＋MLP；或小 LM＋存活 head 的 LoRA） | 多卡訓練 | — | 只有 P3-4 的負載特徵留下明顯差距、且有帶時間戳的文字資料才做 |

- **你說的「在 3090 上採坑」**落在 P3-1、P3-2：資料單位、session 重建、設限、切分洩漏、模擬器不變量、3090 的 24 GB 能不能跑 Llama 32K。這些都不需要先有 oracle。
- **訓練（你說的主力）**從 P3-4 開始。合成資料上的管線試跑可以提前，但輸出只放 scratch。
- 時程〔估計〕：P3-1～P3-3 主要是寫程式與 CPU 模擬，約一週；P3-4、P3-5 約一到兩週；P3-6 看 GPU 空閒程度。CCGrid（12/1）要第三階段的結果會很緊，ICPE（11/16）來不及。老師 10/13 說不用逼這個月出 paper。

---

## 8. P3-3 的判準（草稿，凍結前可以改）

- **headroom**＝(T_最好非 oracle − T_oracle) ÷ T_最好非 oracle；T＝**回來請求**的 TTFT 中位數（平均、p90 一起報告，不改判定）。只算回來請求，避開 D4 那種「所有策略都一樣的第一次重算」把差距稀釋掉的問題。
- **格子**：trace（真實：Mooncake conversation、toolagent；Bailian 若可用）× CPU 容量（工作集的 2、5、10、25、50%）× 慢層頻寬（第二階段的 b\*＝16／32／48 三個限速值，加 NFS 0.331 GiB/s）× GPU（閒／忙）× 還原（Cake／只載入）× 慢層精度（INT4／BF16）× 常數（MI300X／3090）。
- **區域**：小容量（≤10%）、中（25%）、大（50%）分開判。
- **判定**（對 Cake＋INT4＋MI300X 常數這個主設定）：
  - **GO**：至少一個真實 trace、至少一個區域，各時間段 headroom 的中位數 ≥15%，且 ≥2／3 的時間段 ≥10%（10% 是模擬器策略差距誤差 5–6 個百分點的兩倍左右，防止模擬偏差造成的假陽性）。
  - **NO-GO**：所有真實 trace、所有區域的中位數 <5%。
  - 其他＝**MARGINAL：停下來問你**（CLAUDE.md §7）。
  - 合成 trace 只報告，不能單獨讓判定變 GO。
- Q1b（Cake 是否縮小預測價值）、Q1a（拆元件）、Q1c（部分 oracle、雜訊）都只報告，不改判定，但會決定 P3-4 的 loss 清單。
- **NO-GO 不是失敗**：它本身就是一條很好接進論文的結果——「延後設計＋Cake 下，LRU 已在 oracle 的 X% 以內，預測不值得做」，和 D4、When Fancy Eviction Fails 的方向一致。

---

## 9. 3090 採坑清單（P3-1、P3-2 逐條確認）

| # | 坑 | 怎麼防 | 來源 |
|:--|:--|:--|:--|
| 1 | trace 單位看錯 | 載入函式裡寫「用 A 欄驗算 B 欄」的斷言 | CLAUDE.md 規則 6（hash_ids 少算 32 倍那次） |
| 2 | Mooncake 用前綴鏈重建 session，共用 system prompt 的不同 session 被併成一個 | 用 Bailian 的明確 parent 欄位（若有）對拍重建規則 | D4 §4.2：block 0 被全部 12,031 個請求共用 |
| 3 | trace 尾端的 session 被誤當「不回來」 | 右設限；只評估有完整觀察窗的決定 | B5 |
| 4 | 切分洩漏 | 依時間＋依 session 切 | E06（LPC 程式碼） |
| 5 | 獨立事件太少 | 報 session 數，不報 block 數 | B11 |
| 6 | 排序壞掉但 AUC 很好 | 報排序指標 | B5 |
| 7 | 沒有退路 | §6.2 的退路一定要有 | B10 |
| 8 | 換 trace 校準崩掉 | 重擬校準器，報跨 trace | B13 |
| 9 | 模擬器的策略差距偏差 | 模擬只篩、GPU 才定論；GO 門檻高於偏差 | main.tex §3.3、§6.4 |
| 10 | harness 比 vLLM 慢 | 重算加速 2×、1.67× 的敏感度（同 F7） | main.tex §4.6 |
| 11 | 共用機器的污染 | 計時一律 `gpu_guard`；訓練不計時、不受限 | CLAUDE.md §3 |
| 12 | 24 GB 放不下 Llama 32K | 環境 agent 量 peak VRAM；不行就 Llama 16K＋Qwen 32K | migration §2.3 |
| 13 | 合成時間戳把訊號洗掉 | 學習的主張只用真實間隔 | Bidaw |
| 14 | 閉迴路重播的「下次請求時間」含伺服器回應時間 | 標籤用「使用者思考時間」（回應結束到下次請求） | E06（LARU 卡片） |

---

## 10. 怎麼對應到論文

現在的 `main.tex` 是兩步（放置、精度），結論是「寫入簡單、延後降級、讀取時決定」。第三階段依 P3-3 的結果分兩條路：

| P3-3 結果 | 論文怎麼寫 |
|:--|:--|
| NO-GO（<5%） | 在現有論文加一小節「Step 3：延後的決定需要預測嗎？」：延後設計＋Cake 下，LRU 已接近 oracle；拆出預取、降級、丟掉各自的上限；Cake 縮小（或不縮小）預測的價值。補強現有論點，不改主線 |
| MARGINAL | 帶數字找你和老師；通常是「只在小容量／慢層」成立的窄結果，寫成 §6.2「Where an early decision could still pay」的延伸 |
| GO（>15%） | 另開一篇預測器論文：前兩步當動機（為什麼要延後、延後之後剩什麼要猜），主體是 loss 比較、寫入時 vs 決定時、退路、GPU 驗證、跨硬體。舊 Tiara 的 §5.2／§5.3（GBDT、成本敏感 loss、校準、退路）可以拿回來用，加上 9 月的 B5–B13 教訓 |

**和前作的差別（初稿，待 research 補機制細節）**：學習式快取（LRB、LARU、LPC、SAECache、WA、Bidaw）都假設 miss 的代價一樣、只有一層、沒有重算；KV 系統的預取（CachedAttention、KVFlow）沒有和 oracle 比。我們的問題是**多層、miss 代價隨位置與層而不同、讀取端會自適應**，而且預測放在延後的決定點上。

**論文的修改要另開工作階段、先有 `results/` 的證據**（CLAUDE.md 規則 5）。第三階段的實驗 agent 不改 `main.tex`。

---

## 11. 分支：為什麼要開、怎麼用

**一句話**：分支是一條平行的 commit 線。MI300X 那邊的 agent 一直往 `main` 推第二階段的結果和論文；3090 的第三階段是探索，常常要試錯。分開之後兩邊不會互相干擾，等第三階段有站得住的結果再合回 `main`。

- 已建立：`p3-predictor`（從 `main` 的 `4718768` 分出來）。目前只在本機，還沒推上 GitHub。
- 日常：在 `p3-predictor` 上 commit（`infra(p3):`、`results(p3):`、`docs(p3):`）。
- 拿 MI300X 的新結果：`git fetch origin && git merge origin/main`（第二階段的檔案和第三階段不重疊，通常不會衝突；`results/RUNLOG_MI300X.md` 只有 MI300X 寫，3090 寫 `results/RUNLOG_3090_P3.md`）。
- 合回去：每個 milestone 判完，開 PR 合進 `main`。`main.tex` 不在分支上改。
- 備份：要推的話是 `git push -u origin p3-predictor`（repo 是 public，推之前確認沒有大檔與個資）。

---

## 12. opencode（你問的）

- 這台有 opencode 1.17.8，設定的是 Anthropic（`~/.config/opencode/opencode.jsonc`，走 API 額度計費），預設 Sonnet 5.5。`--variant xhigh` 對 `anthropic/claude-opus-5-5` 有支援（`opencode models anthropic --verbose` 的 variants 有 low／medium／high／xhigh／max）。
- **適合的用途**：當**零前文的審查者**——拿事先登記的判準、實驗報告給它挑錯。指令形如
  `opencode run -m anthropic/claude-opus-5-5 --variant xhigh --agent plan --dir <只放要審的檔案的資料夾> "..."`。
  它和我是同一家族的模型，所以只能標「**zero-context same-family review（NOT cross-model）**」（CLAUDE.md §5 的規矩：不能假裝是跨模型審查）。
- **不適合的用途**：長時間跑實驗、盯佇列。那些用本機的記錄殼和佇列就好；opencode 每一步都在花 API 額度。
- 這次計畫凍結前，我打算用它審一次 §8 的判準（唯讀，只給它計畫書和相關文件）。

---

## 13. 需要你決定的事

1. **§8 的判準**（門檻、主設定、區域劃分）可以嗎？凍結之後就不改。
2. **P3-3 之前不訓練**（CLAUDE.md §8）：同意照這個順序嗎？還是你要改 CLAUDE.md 的這條規則？
3. **WildChat／LMSYS 這類要同意條款的資料**：需要你用自己的 HF 帳號按同意（環境 agent 會列出來）。
4. **分支要不要推上 GitHub**（備份用）。
5. **大檔案根目錄改到 `/ssd8/hungwei/paper-hkv/`**（`/ssd7` 只剩 106 GB、99%）：環境 agent 已照這個做；CLAUDE.md §2 要跟著改，我等環境 agent 回報後一起改。
