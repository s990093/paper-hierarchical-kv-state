---
type: report
created: 2026-08-28
tags: [paper, latex, draft, kv-cache, amd, mi300x]
related_ideas: ["[[idea-20260828-hierarchical-kv-state]]"]
---

# 論文初稿：Adaptive Hierarchical KV State Management

（此資料夾只對應單一 idea：[[idea-20260828-hierarchical-kv-state]]）

## 這是什麼

一份 **8 頁的 systems 論文初稿**，LaTeX 原始檔，正文中文、技術名詞保留英文。
內容來自本 workspace 的兩份調查報告：
- [[report-hierarchical-kv-state-20260828]]（gap 分析）
- [[report-hierarchical-kv-state-amd-feasibility-20260828]]（AMD 可行性）

## 怎麼編譯

```bash
make          # 完整編譯（xelatex → bibtex → xelatex ×2）並印出警告統計
make clean    # 清掉中間檔
make check    # 列出所有 TODO 與佔位符
```

**必須用 XeLaTeX**（內文含中文，使用 `xeCJK`）。
字型目前設定為 macOS 內建的 **Songti TC / Heiti TC**。在 Linux 上請把 `main.tex` 開頭改成 `Noto Serif CJK TC` / `Noto Sans CJK TC`。

**目前建置狀態：0 字型警告、0 overfull box、0 未定義引用、15 頁。**

## 檔案

| 檔案 | 內容 |
|---|---|
| `main.tex` | 論文本體（含 2 張 TikZ 圖） |
| `refs.bib` | 45 筆參考文獻，**全部經直接抓取查證**，預印本以 `[PREPRINT]` 標註 |
| `Makefile` | 編譯與檢查 |
| `main.pdf` | 產出 |
| **`EXPERIMENT_PLAN.md`** | **3090 PoC 執行計畫（給執行 agent 用，不要給它 PDF）** |
| **`OPEN_ISSUES.md`** | **已知但暫緩處理的問題清單（A/B/C 三級 + 已查證事項）** |
| **`VENUES.md`** | **投稿場所與截止日調查（含歐洲中階場所）** |

## ⚠️ 這份初稿的狀態（重要）

**尚無任何實驗結果。** 各章節的完成度不同：

| 章節 | 狀態 |
|---|---|
| §1 Introduction | ✅ 完整 |
| §2 Background & Motivation | ✅ **完整，且數據都經查證**（KV footprint 由 HuggingFace config 算出；硬體比值有出處） |
| §3 Related Work | ✅ **完整**，含 15 篇的特徵矩陣 |
| §4 Problem Formulation | ✅ 完整（式 1–4） |
| §5 Design | ✅ **完整**。§5.2 預測器（預測什麼／特徵／模型／標籤）與 §5.3 **成本敏感損失函數**皆有 11 個既有系統的比對佐證 |
| §6 Evaluation | 🔴 **是「實驗計畫」，不是結果**。表 8 全部是灰色佔位符 |
| §7 Limitations | ✅ 完整，且誠實列出六項限制 |

**論文中已在 Abstract、Introduction 結尾與 §7 三處明確標註「本文為初稿、實驗尚未執行」。** 不要在這個狀態投稿。

## 兩張 TikZ 圖

- **Fig. 1（`fig:ladder`）**：動作空間階梯。五個狀態 + 升降級箭頭 + 右側標註存取成本。這張圖的重點是把「位置分層」與「精度分層」畫成同一條階梯，並讓 `DROP → recompute` 成為其中一層而非旁支。
- **Table 7（`tab:predictors`）**：11 個既有學習式快取／放置預測器的設計比對，最後一欄是重點——只有 PARROT 與 Sibyl 處理成本不對稱，且都不是針對異質動作成本。
- **Fig. 2（`fig:arch`）**：系統架構。灰色 = 既有基礎設施（vLLM / OffloadingConnector），橘色 = 本文貢獻（插在 `CachePolicy` 介面）。這張圖直接回應「為什麼這不只是一個 plugin」——因為貢獻邊界畫得很清楚：機制屬於 vLLM，策略屬於論文。

## 下一步

1. **先跑 §6.6 的 Oracle**。若 headroom < 5%，這篇論文的方向要改，此時改成本最低。
2. Oracle 有 headroom 才動手實作 §5 的預測器。
3. 表 8（主結果）與表 7（消融）填上真實數字後，這份初稿就接近可投稿。
4. 投稿前需補：作者資訊、CRediT、Funding、COI（§Statements 已留空格）。

## 相關筆記
- [[idea-20260828-hierarchical-kv-state]]
- [[report-hierarchical-kv-state-20260828]]
- [[report-hierarchical-kv-state-amd-feasibility-20260828]]
- [[compare-hierarchical-kv-state-gap-analysis]]
