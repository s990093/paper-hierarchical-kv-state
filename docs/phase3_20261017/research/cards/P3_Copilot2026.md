# P3 卡：Agentic Coding in the Wild（GitHub Copilot trace，2026）

- **出處**：Banruo Liu, Haoran Qiu, Íñigo Goiri, Rodrigo Fonseca, Ricardo Bianchini, Esha Choukse（所屬機構未查證；資料由 Azure 公開資料集釋出）。arXiv 2608.00101v1（2026-07-30），venue〔未查證〕。<https://arxiv.org/abs/2608.00101>
- **資料**：<https://github.com/Azure/AzurePublicDataset/blob/master/GitHubCopilotCodingAgentDataset2026.md>（md commit `6ada3303`，2026-09-30）；release `ghcp-coding-agent-2026`（2026-10-01），7 個 tar.gz 共約 3.22 GB〔計算〕，CC-BY。
- **讀了哪裡**：PDF p.16–18（§9 閒置與預測）；md、`DATASET_CARD.md`、`schema.json`〔文件〕。沒有下載資料。

## trace
- 公開的是取樣版：301,026 session、1,189,581 使用者回合、9,310,255 次 LLM 呼叫、8,681,190 次工具呼叫、37 個匿名模型；2026-06-01 至 06-07〔文件〕。論文分析的是全量：320 萬使用者、1,300 萬 session〔摘要〕。
- 欄位：回合裡的 `llm_calls`（`timestamp`、`duration_ms`、`initiator_type` user／agent、`tokens` 含 cached）與 `tool_batches`（時長、各工具時長與狀態）；**沒有 user id、文字、block hash**〔文件〕。

## 原文的觀察
- KV 命中率回合內平均約 90%，跨回合邊界降到 55%〔摘要〕。
- 回合內的閒置很短（容器 5.8 s、KV 1.2 s），跨回合的長（243 s、172 s）；超過 90% 的閒置區間在回合內；回合之間的使用者閒置中位 1,512 s（25.2 分鐘），尾端超過一天〔原文 p.17〕。
- 原文提到 Claude 模型的快取預設保留 300 s；預測閒置跨過這個界線時，可以在到期前送一個便宜的 keep-alive 請求〔原文 p.18〕。

## 預測器（§9.2）
- **預測什麼**：回合邊界後，到下一回合的閒置時間的**存活曲線** S(t)＝Pr(idle>t)〔原文 p.17〕。
- **loss**：LightGBM **分位數迴歸**（pinball），目標是 log 閒置時間；12 個分位數各一個 400 棵樹的模型，合起來約 2 MB；一個 16 vCPU 節點訓練不到一分鐘，推論 <3 ms〔原文 p.17〕。
- **條件更新**：過了 t₀ 還沒回來，用 S(t|idle>t₀)＝S(t)/S(t₀)，不必重訓〔原文 p.17〕。
- **特徵**：回合層（時長、LLM 次數、token、工具時間、失敗、成功率）、session 層（回合序號、已經過時間、上一次閒置、平均閒置）、時間（是否平日）；最重要的是平均累積閒置與回合序號〔原文 p.17〕。
- **切分**：一週 150K session 訓練、**下一週** 50K 評估〔原文 p.17〕。
- **準確度**：「>60 s」的 AUC 0.73；always-positive 0.5；只看上一次閒置的啟發式 0.58；換模型族（例如決策樹）只有微小差別〔原文 p.17〕。過了越久越難：準確度與 F1 從 30 s 的約 80–88% 降到 30 分鐘的 25–43%；但「抓到的閒置時間比例」一直在 86–90%〔原文 p.17〕。
- **設限**：原文沒有提〔原文未見〕。**沒有端到端的策略評估、沒有 oracle**。

## 和第三階段的關係〔判讀〕
- 分位數→存活曲線→條件更新，是 L6（pinball）和 L3（存活）的混合；正好能同時回答降級（順序）、丟掉（尾端機率）、預取（低分位數）。
- 「抓到的閒置時間」和 AUC 走勢相反，再次說明分類指標不等於決策品質（我們 9 月的教訓）。
- 沒有 block hash，不能模擬前綴共用；適合 X2（訊號在不在）與跨 trace 測試。
