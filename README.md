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
| 工作環境 | ✅ 建好（7×RTX 3090、TeX、vLLM） |
| Skill 工具鏈 | ✅ 22 個 vendored（寫作 + 實驗 + 誠實性稽核） |
| **實驗結果** | 🔴 **尚未取得任何數字** |
| 下一個決定點 | 🔴 **Milestone 4 的 Oracle go/no-go** |

**時程**：MLSys 2027 截止 2026-10-30。**9 月底若還沒有 Oracle 結果，改投 EuroMLSys 2027。**
見 [`VENUES.md`](VENUES.md)。

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

**Ampere 的兩個硬限制**（會影響論文哪些章節能在此驗證）：
1. **不支援原生 FP8** → 動作空間裡的 `GPU-FP8` 在平台 A 量不到，標 `NOT_SUPPORTED`，不要用估的填
2. **無記憶體/計算分軌能耗計數器** → 論文 §6.7 的能耗分析不適用於平台 A

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

1. **完成環境驗收 A1–A3**，特別是 **A3（`OffloadingConnector` 可用）**——
   若失敗，整個計畫要重新設計
2. **Milestone 1**：容量懸崖實測，比對論文 §2.5 的算術值
3. **Milestone 2**：成本模型 **2×5 矩陣**（平時成本 vs 被需要時的成本）+ `recompute_chain.csv`
4. **Milestone 3**：Tier 0 baselines
5. **🔴 Milestone 4：Oracle** ← 決定性。有 headroom 才動手實作 §5 的預測器
6. 投稿前需補：作者資訊、CRediT、Funding、COI（§Statements 已留空格）

## 相關筆記
- [[idea-20260828-hierarchical-kv-state]]
- [[report-hierarchical-kv-state-20260828]]（gap 分析）
- [[report-hierarchical-kv-state-amd-feasibility-20260828]]（AMD 可行性）
- [[compare-hierarchical-kv-state-gap-analysis]]

> ℹ️ 上面這幾份筆記在原 workspace 的上層目錄，**不在這個 repo 裡**。
