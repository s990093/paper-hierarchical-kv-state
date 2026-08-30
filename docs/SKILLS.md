# Skill 選型紀錄

**日期**：2026-08-30
**目的**：在動手跑實驗之前，先把「論文寫作」與「實驗執行／誠實性稽核」的工具鏈裝好。
**方法**：網路調查候選 → 逐一 clone 檢視 SKILL.md → 依本專案需求挑選 → vendored 進 repo。

上游 clone 保留在 `/ssd7/hungwei/paper-hkv/vendor-skills/`（不進 git），供日後 `git pull` 比對更新。

---

## 1. 候選調查

| 專案 | 星等定位 | License | 判定 |
|---|---|---|---|
| **[SNL-UCSB/paper-writing-skill](https://github.com/SNL-UCSB/paper-writing-skill)** | UCSB Systems & Networking Lab 的寫作方法論，由 6 篇論文 / 8 次投稿 / 7,600+ 次 Overleaf 編輯 / 5 輪 peer review 的**取證式分析**萃取而成 | MIT | ✅ **採用（全部）** |
| **[wanshuiyin/Auto-claude-code-research-in-sleep (ARIS)](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep)** | 82 個純 Markdown skill，涵蓋 idea → 實驗 → 論文 → rebuttal 全鏈，強調 **cross-model 對抗審查** | MIT | ✅ **採用（挑 21 個）** |
| [Imbad0202/academic-research-skills](https://github.com/imbad0202/academic-research-skills) | 12-agent 論文流水線，含 citation 幻覺防治、整合性 gate | **CC BY-NC 4.0** | ❌ 不 vendor |
| [hameefy/claude-latex-skill](https://github.com/hameefy/claude-latex-skill) | 計算/應用數學向的 LaTeX（定理、證明、收斂分析） | — | ❌ 領域不符 |
| [ndpvt-web/latex-document-skill](https://github.com/ndpvt-web/latex-document-skill) | 27 個通用 LaTeX 模板 | — | ❌ 本專案已有可編譯的 `main.tex` |
| [flonat/flonat-research](https://github.com/flonat/flonat-research) | PhD 研究者的 skills/agents/hooks 基礎設施 | — | 🔶 觀望 |

### 不採用 `academic-research-skills` 的原因

**不是品質問題**——它的 citation 幻覺防治（Zhao et al. 2026 對 111M 筆參考文獻的稽核）與
AI-research 失敗模式 checklist 都做得很好。三個具體理由：

1. **License**：CC BY-NC 4.0（禁商業用途）。本 repo 是 **public** 的，vendored 進去會讓
   整個 repo 的下游使用條件變複雜。上面兩個是 MIT，乾淨。
2. **重量**：58 MB，帶 `pyproject.toml` / hooks / evals / MCP 基礎設施，是一個要「安裝與維護」
   的系統，不是可以直接 copy 的 Markdown。
3. **功能重疊**：它最有價值的 citation-audit 與 integrity gate，ARIS 有對應的
   `citation-audit` / `experiment-audit`，且是純 Markdown。

> 若日後需要它的 `ai_research_failure_modes.md` checklist，用 plugin marketplace 裝在
> **user 層**（`~/.claude/`）而不是 vendored 進本 repo，即可避開 license 問題。

---

## 2. 採用清單（22 個，`.claude/skills/`，共 772 KB）

### 2.1 `paper-writing`（SNL-UCSB，MIT）

唯一一個從**真實修改歷史**逆向工程出來的寫作 skill——不是「寫作建議」，是可執行的 gate。

五階段流水線：
`Brainstorm (34 問 / 6 phase) → Architecture (章節+claim+圖表+頁數預算) → Section Drafts (強制順序：Intro 兩次 → Eval → Design → Background → Related Work → Abstract) → Integration → Compression (7 種操作，目標壓縮 30–50%)`

**為什麼對本專案特別合用：**

- **目標 venue 完全命中**：systems（SIGCOMM/NSDI/CoNEXT）+ ML（NeurIPS/ICLR/ICML）——
  本文投 MLSys / EuroSys / EuroMLSys，正在這個交集上。
- **強制的 Style Audit Gate**：`gate_mechanical.md`（M1–M18，可 grep 的機械檢查）+
  `gate_semantic.md`（S1–S31，讀者判斷檢查）。每次編輯都要過。
- **`red_team_protocol.md`**：獨立的 fresh-reader 批判。本文最脆弱的地方正是
  「§6 是計畫不是結果」——red team 會先於審稿人抓到。
- **`figure_synthesis_guide.md` + `figure_templates/tikz_skeletons.md`**：
  `main.tex` 已有兩張手寫 TikZ 圖（Fig.1 動作空間階梯、Fig.2 系統架構），
  後續補主結果圖時可沿用同一套 venue style。
- **`section_rhetorical_moves/evaluation.md`**：Oracle 結果出來後，
  §6 要從「計畫」改寫成「結果」，這是最需要結構指引的一步。

**客製化位置**：`author_profile/`（editorial_principles / craft_reference / compression_patterns /
rhetorical_moves / intervention_types）。目前**維持上游預設未改**——先照原樣用，
確認哪幾條與本文的中文寫作衝突後再改，改動要記在本檔。

### 2.2 ARIS 誠實性 gate（★ 本專案的核心需求）

`EXPERIMENT_PLAN.md` §0 的五條禁令（不准編造數字／不准跳過失敗／每個數字可追溯／
不准跳過 go-no-go／不准美化）本來只是**文字約定**。這批 skill 把它變成**可執行的檢查**：

| Skill | 對應本專案的哪條禁令 |
|---|---|
| `experiment-audit` | 查 fake ground truth、score normalization fraud、**phantom results**、scope 不足 → 禁令 1 |
| `result-to-claim` | 判定結果支持／不支持哪些 claim、還缺什麼 → 禁令 5（不准美化） |
| `paper-claim-audit` | **zero-context** 逐一核對論文每個數字 vs 原始結果檔 → 禁令 3 |
| `citation-audit` | `refs.bib` 45 筆的作者／年份／venue／**引用語境是否被原文支持** |
| `kill-argument` | 兩執行緒對抗審查：先寫最強拒稿理由，再逐點反駁 |

`refs.bib` 的 header 宣稱「全部經直接抓取查證」——`citation-audit` 正是用來**驗證這個宣稱本身**。

### 2.3 ARIS 實驗執行

| Skill | 為什麼選 |
|---|---|
| **`experiment-bridge`** | description 明寫「**Reads EXPERIMENT_PLAN.md**, implements experiment code, deploys to GPU, collects initial results」——本專案剛好就有一份 `EXPERIMENT_PLAN.md`。**主入口** |
| **`experiment-queue`** | 「multi-seed/multi-config + **OOM-aware retry** + stale-screen cleanup」——M1 的容量懸崖二分搜尋**本質上就是刻意去撞 OOM**，這正是它的形狀 |
| `run-experiment` | 單一 job 部署 |
| `monitor-experiment` | 長跑 job 輪詢（可配 `/loop`） |
| `analyze-results` | 統計與比較表 |
| `ablation-planner` | Oracle = GO 之後，設計消融矩陣（尤其表 8 的 (B) 區塊） |
| `experiment-plan` | 需要把新想法變成 claim-driven roadmap 時 |

### 2.4 ARIS 論文輔助

`writing-systems-papers`（systems venue 段落級藍圖與頁數配置）、`paper-compile`、
`paper-figure`、`figure-spec`、`rebuttal`。

`writing-systems-papers` 與 `paper-writing` **不衝突，是不同層級**：
前者給 systems venue 的**結構骨架**（頁數配置、段落角色），後者給**流程與句子級 gate**。

### 2.5 文獻

`arxiv`、`semantic-scholar`、`openalex`、`novelty-check`。
`novelty-check` 用來定期複查主張——`OPEN_ISSUES.md` 已記錄 KVTuner（ICML'25）用的正是
本文的兩個主模型，這類重疊需要持續監看。

---

## 3. ⚠️ 已知落差：沒有 Codex MCP

ARIS 有 46 個 skill 在 `allowed-tools` 宣告 `mcp__codex__codex`。我們採用的 21 個裡，
**7 個**依賴它：`experiment-audit`、`result-to-claim`、`paper-claim-audit`、
`citation-audit`、`kill-argument`、`ablation-planner`、`novelty-check`。

**它的用途不是「更聰明的模型」，是「沒有前文脈絡的審查者」**——
用來規避同一個 agent 既做實驗又審自己實驗的確認偏誤。這是設計上的要點，不是實作細節。

**本機沒有裝 Codex MCP。** 替代方案依偏好順序：

1. **開一個全新的 Claude subagent 當 zero-context reviewer**——效果最接近原設計。
   ⚠️ 依本專案的 agent 規則，**spawn subagent 需使用者明確要求**。
2. 裝 ARIS 附帶的 MCP server：`vendor-skills/aris/mcp-servers/` 下有 `claude-review`、
   `gemini-review`、`manual-review`、`llm-chat` 可選。
3. 降級成 self-review，**且必須在 `results/RUNLOG.md` 明確標註
   「此次為 self-review，非 cross-model」**。

**不要假裝跑了 cross-model review。** 這正是這批 skill 要防的失敗模式。

---

## 4. 更新流程

```bash
cd /ssd7/hungwei/paper-hkv/vendor-skills/aris && git pull
cd /ssd7/hungwei/paper-hkv/vendor-skills/paper-writing-skill && git pull
# 比對本 repo 的 vendored 版本，逐個確認差異後再 copy：
diff -ru <repo>/.claude/skills/<name> /ssd7/hungwei/paper-hkv/vendor-skills/aris/skills/<name>
```

**不要無腦覆蓋。** 若本專案改過某個 skill（例如替換 Codex 依賴），要在本檔記錄，
更新時保留客製化。目前**尚未修改任何 vendored skill**。

---

## 5. 授權

| 來源 | License | 檔案 |
|---|---|---|
| SNL-UCSB/paper-writing-skill | MIT © 2026 Arpit Gupta | `.claude/skills/paper-writing/LICENSE` |
| wanshuiyin/ARIS | MIT © 2026 wanshuiyin | `.claude/skills/LICENSE.ARIS` |

兩者皆 MIT，允許 vendored 進本 public repo，保留原授權聲明即可。
