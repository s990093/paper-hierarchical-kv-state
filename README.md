---
type: report
created: 2026-08-28
updated: 2026-08-30
tags: [paper, latex, draft, kv-cache, amd, mi300x, rtx3090]
related_ideas: ["[[idea-20260828-hierarchical-kv-state]]"]
---

# Tiara — Cost-Asymmetric Hierarchical KV State Management

**成本不對稱下以品質為約束的可恢復階層式 KV State 管理**

長上下文 LLM 推論的 KV cache 放置策略。核心主張：**重算與傳輸的成本比 κ 跨硬體變動達 32 倍**
（MI300X 6–21×、RTX 3090 54–190×），因此「單一 KV 放置策略適用所有硬體」的隱含前提不成立。

> ⚠️ **本專案為論文初稿 + 進行中的實驗。§6 是實驗計畫，不是結果。**
> 表 8 全部是灰色佔位符。**不要在這個狀態投稿。**

---

## 🚦 現在的狀態

| 面向 | 狀態 |
|---|---|
| 論文初稿 | ✅ 完成，16 頁，0 warning 建置乾淨 |
| 工作環境 | ✅ 7×RTX 3090、TeX、vLLM 0.28.0+cu129、兩個模型 |
| Skill 工具鏈 | ✅ 22 個 vendored（寫作 + 實驗 + 誠實性稽核） |
| 環境驗收 A1–A3 | ✅ **全過**（含決定性的 A3 卸載連接器） |
| **Milestone 1**（容量懸崖） | ✅ **完成**，4 個設定、兩個邊界都驗證過 |
| **Milestone 3**（Tier 0 baselines） | ✅ **Llama 5/5 完成**；Qwen 進行中 |
| Milestone 2（成本常數 2×5 矩陣） | 🔴 未開始 |
| **Milestone 4（Oracle）** | 🔴 **未開始 ← 這是整個研究的停損點** |
| 論文表 8 | 🔴 仍是灰色佔位符 |

**時程**：MLSys 2027 截止 2026-10-30。**9 月底若還沒有 Oracle 結果，改投 EuroMLSys 2027。**
見 [`VENUES.md`](VENUES.md)。

---

## 📊 目前量到的東西

完整記錄與方法說明見 [`results/RUNLOG.md`](results/RUNLOG.md)。
**所有數字都可追溯到一條指令與一個輸出檔**；沒量到的一律寫 `NOT_MEASURED`。

### M1 — GPU KV 容量懸崖（單卡 24 GB，BF16 權重）

| 設定 | **懸崖（token）** | KV GiB | 在懸崖啟動 | 超出 1.15× |
|---|---|---|---|---|
| Llama-3.1-8B, KV BF16 | **41,648** | 5.084 | ✅ | ✅ 如預期失敗 |
| Llama-3.1-8B, **KV FP8** | **83,312** | 5.085 | ✅ | ✅ 如預期失敗 |
| Qwen2.5-7B-1M, KV BF16 | **106,512** | 5.688 | ✅ | ✅ 如預期失敗 |
| Qwen2.5-7B-1M, **KV FP8** | **213,040** | 5.689 | ✅ | ✅ 如預期失敗 |

FP8 在兩個模型上都給出**恰好 2 倍**的 token 數而位元組佔用不變——沒量錯的強證據。
補回權重差後，與論文 §2.5 的算術值差 **6–10%**。

### M3 — Tier 0 baselines（Llama，warm 相對 cold 的 TTFT 改善）

工作負載是**兩輪共享前綴**：cold 送 4 個不同的長前綴，warm 再送一次同樣的。
`4 × ctx` 跨過 M1 量到的 41,648，逼出逐出。**Δ% 就是那一階卸載的價值。**

| baseline | ctx 16K（工作集 65,536） | ctx 32K（工作集 131,072） |
|---|---|---|
| `full_gpu`（無第二階） | **−1.4%** | **−0.3%** |
| `cpu_lru` | 84.6% | **91.5%** |
| `cpu_arc` | 84.7% | 64.5% |
| `tier_fs`（CPU+磁碟） | 89.2% | **92.1%** |
| `lmcache` ⚠️ | 58.0% | 74.2% |

⚠️ `lmcache` 的 wheel 沒有編譯擴充（走 torch baseline），**這兩個數字是被 handicap 的**，
不能用來宣稱我們比它好。
⚠️ 這批是 `concurrency_mode = parallel`，**絕對毫秒數要等 `--serial` 重跑**才可進論文；
相對關係（誰贏誰、轉折在哪）可信。

### 三個推翻原計畫的實測發現

1. **Ampere 可以用 FP8 KV cache。** 計畫書說 sm_86 量不到 `GPU-FP8` 那一階，錯了——
   混淆了 FP8 **運算**（Ampere 沒有）與 FP8 **儲存**（可以）。論文的動作空間是儲存狀態，
   所以六階裡有五階在平台 A 就量得到。
2. **vLLM 0.28.0 跑不了 Qwen2.5-1M 的 Dual Chunk Attention。** V1 engine 沒有對應的
   attention backend。已建 no-DCA 變體，評測上限因此是 **262,144**（該模型真正訓練的
   長度）而非宣稱的 1M——這對品質歸因**反而更乾淨**。
3. **M1 的容量數字準確預測了 M3 的行為轉折。** `full_gpu` 在工作集 32,768 時 warm
   仍快 95.2%，在 65,536 時掉到 −1.4%，轉折正好落在獨立量到的 41,648 上。
   **論文 §2.5 從算術主張變成可觀測現象。**

---

## 📁 檔案地圖

**先讀哪一份，取決於你要做什麼：**

| 你要做的事 | 讀這份 |
|---|---|
| **在這台機器上動手** | **[`CLAUDE.md`](CLAUDE.md)** ← agent 的操作手冊，先讀這個 |
| 跑實驗 | [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) ← 執行 agent 的唯一指令書 |
| 了解論文主張 | `main.pdf` / `main.tex` |
| 知道哪些問題是「刻意暫緩」的 | [`OPEN_ISSUES.md`](OPEN_ISSUES.md) |
| 決定投哪裡 | [`VENUES.md`](VENUES.md) |
| 知道裝了哪些 skill、為什麼 | [`docs/SKILLS.md`](docs/SKILLS.md) |
| 看實驗做到哪 | [`results/RUNLOG.md`](results/RUNLOG.md) |
| **快速看懂目前的量測狀態** | **[視覺化報告](https://claude.ai/code/artifact/abef2cd7-4c8c-4dcf-8089-335218b4813d)**（原始碼 `docs/measurement-log.html`） |

> **不要把 `main.pdf` 交給執行 agent。** 那是主張文件，不是執行文件。
> 執行 agent 只該拿到 `CLAUDE.md` + `EXPERIMENT_PLAN.md`。

```
├── main.tex refs.bib Makefile main.pdf   論文（45 筆參考文獻、2 張 TikZ 圖）
├── CLAUDE.md                             ★ agent 操作手冊
├── EXPERIMENT_PLAN.md                    ★ PoC 執行計畫（Milestone 1–5）
├── OPEN_ISSUES.md                        已知但刻意暫緩的問題（A/B/C 三級）
├── VENUES.md                             投稿場所與截止日
├── docs/SKILLS.md                        skill 選型紀錄
├── code/                                 實驗 harness
├── results/                              ★ 只放摘要級結果（CSV/JSON/Markdown）
│   ├── RUNLOG.md  env.json
│   └── m1_capacity/ m2_harness/ m3_baseline/ m4_oracle/
├── .claude/skills/                       22 個 vendored skill
└── _big -> /ssd7/hungwei/paper-hkv       大檔案（gitignore）
```

---

## 🔨 編譯

```bash
export PATH=/ssd7/hungwei/paper-hkv/texlive/.TinyTeX/bin/x86_64-linux:$PATH
make          # xelatex → bibtex → xelatex ×2，並印出警告統計
make clean    # 清中間檔
make check    # 列出所有 TODO 與佔位符
```

**必須用 XeLaTeX**（內文中文，使用 `xeCJK`）。

字型已移植到 Linux：`Noto Serif CJK TC` / `Noto Sans CJK TC` + `TeX Gyre Termes/Heros/Cursor`。
（原始 macOS 版本用 `Songti TC` / `Heiti TC` / `Times New Roman`。）

**目前建置狀態：0 字型警告、0 overfull box、0 未定義引用、16 頁。**
⚠️ macOS 版是 15 頁——字型度量差異造成。**投稿前若有頁數上限，要用投稿機器的字型重新確認。**

---

## 🖥 執行環境

| | |
|---|---|
| GPU | 7 × RTX 3090 24 GB（sm_86, Ampere） |
| Driver / CUDA | 550.163.01 / 12.4 |
| 大檔案 | `/ssd7/hungwei/paper-hkv/`（TeX、venv、模型、原始 log、profile） |
| sudo | 無（所有安裝都是 user-level） |

**Ampere 的限制**（2026-08-30 實測修正）：
1. ~~不支援原生 FP8~~ → **這條原本是錯的。** `--kv-cache-dtype fp8` 在 sm_86 可用，
   實測給出恰好 2 倍的 KV 容量。錯在混淆 FP8 **運算**（Ampere 沒有）與 FP8 **儲存**（可以）。
2. **無記憶體/計算分軌能耗計數器** → 論文 §6.7 的能耗分析不適用於平台 A（仍成立）
3. **vLLM 0.28.0 的 V1 engine 沒有可用的 DCA 路徑** → Qwen2.5-7B-1M 要用 no-DCA 變體，
   評測上限 262,144

**這是共用機器**（`/ssd7` 下有二十幾個使用者），所以每次量測都要防插隊：
`code/gpu_guard.py`（GPU 佔用）與 `code/shm_gc.py`（`/dev/shm` 洩漏）是必用的。

詳見 [`CLAUDE.md`](CLAUDE.md) §3。

---

## 📝 各章節完成度

| 章節 | 狀態 |
|---|---|
| §1 Introduction | ✅ 完整 |
| §2 Background & Motivation | ✅ 完整，數據都經查證（KV footprint 由 HuggingFace config 算出） |
| §3 Related Work | ✅ 完整，含 15 篇的特徵矩陣 |
| §4 Problem Formulation | ✅ 完整（式 1–4） |
| §5 Design | ✅ 完整。§5.2 預測器與 §5.3 成本敏感損失函數皆有 11 個既有系統的比對佐證 |
| §6 Evaluation | 🔴 **是「實驗計畫」，不是結果。表 8 全部是灰色佔位符** |
| §7 Limitations | ✅ 完整，誠實列出六項限制 |

論文已在 Abstract、Introduction 結尾與 §7 三處明確標註「本文為初稿、實驗尚未執行」。

### 兩張 TikZ 圖

- **Fig. 1（`fig:ladder`）**：動作空間階梯。把「位置分層」與「精度分層」畫成同一條階梯，
  並讓 `DROP → recompute` 成為其中一層而非旁支。
- **Table 7（`tab:predictors`）**：11 個既有學習式快取／放置預測器的設計比對。
  最後一欄是重點——只有 PARROT 與 Sibyl 處理成本不對稱，且都不是針對異質動作成本。
- **Fig. 2（`fig:arch`）**：系統架構。灰色 = 既有基礎設施（vLLM / OffloadingConnector），
  橘色 = 本文貢獻（插在 `CachePolicy` 介面）。這張圖回應「為什麼這不只是一個 plugin」。

---

## 🔴 不可協商的規則

適用於每一個 agent、每一次 session（完整版見 [`CLAUDE.md`](CLAUDE.md) §1）：

1. **不准編造任何數字。** 沒跑出來就寫 `NOT_MEASURED`。**論文裡的數字是假設，不是結果。**
2. **不准跳過失敗。** 指令失敗 → 記完整錯誤 → 停下來回報。
3. **每一個數字都要能追溯到一條指令與一個輸出檔。**
4. **不准跳過 Milestone 4 的 go/no-go。** `< 5%` headroom = 停止。
   **NO-GO 不是失敗，是省下數個月的有價值負面結果。**
5. **實驗 agent 不准改 `main.tex`。**

---

## 下一步

1. ~~環境驗收 A1–A3~~ ✅ 全過
2. ~~Milestone 1：容量懸崖~~ ✅ 完成
3. **Milestone 3 收尾**：Qwen 的五個 baseline + **用 `--serial` 重跑定稿數字**
   （目前的絕對毫秒數是平行跑的，PCIe 被自己的其他 job 共用）
4. **Milestone 2**：成本模型 **2×5 矩陣**（平時成本 vs 被需要時的成本）+ `recompute_chain.csv`
   ← FP8 那一階現在**量得到**了，見上面的發現 1
5. **🔴 Milestone 4：Oracle** ← 決定性。`> 15%` GO、`5–15%` 停下來問人、`< 5%` **停止**
6. 品質評測（multi-fact extraction）目前是 `NOT_MEASURED`——M3 只量了延遲
7. AWQ-INT4 權重待決策：`Qwen2.5-7B-Instruct-1M` 沒有官方 AWQ，
   用社群版 vs 自行量化需要決定
8. 投稿前需補：作者資訊、CRediT、Funding、COI（§Statements 已留空格）

## 相關筆記
- [[idea-20260828-hierarchical-kv-state]]
- [[report-hierarchical-kv-state-20260828]]（gap 分析）
- [[report-hierarchical-kv-state-amd-feasibility-20260828]]（AMD 可行性）
- [[compare-hierarchical-kv-state-gap-analysis]]

> ℹ️ 上面這幾份筆記在原 workspace 的上層目錄，**不在這個 repo 裡**。
