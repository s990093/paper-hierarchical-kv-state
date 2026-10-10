# D8 卡片：TideRL: Boosting Agentic RL Goodput with Readiness-Aware Scheduling

- **連結**：https://arxiv.org/abs/2608.10402
- **venue／年份**：arXiv 2608.10402（2026-08）；venue 未查證
- **讀了哪裡**：arXiv 摘要頁
- **三行摘要**：
  1. 多輪 agentic RL 的 rollout 會為了外部環境一再暫停、帶著更長的 context 回來〔摘要〕。
  2. 原文說：GPU 等待與「重複的 prefill 重算」都是純浪費〔摘要〕。
  3. 訓練 goodput 比同步基線最多 5.6 倍；KV 命中率高 1.58 倍〔摘要〕。
- **和 D8 的關係**：方向 6、7：RL 裡的 KV 重算已被當成主要浪費。
