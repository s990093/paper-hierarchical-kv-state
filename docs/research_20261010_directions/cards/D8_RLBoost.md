# D8 卡片：RLBoost: Harvesting Preemptible Resources for Cost-Efficient Reinforcement Learning on LLMs

- **連結**：https://arxiv.org/abs/2510.19225
- **venue／年份**：arXiv 2510.19225（2025-10）；venue 未查證
- **讀了哪裡**：arXiv 摘要頁
- **三行摘要**：
  1. 用雲端 spot（可被搶佔）GPU 跑 rollout〔摘要〕。
  2. 被搶佔時用 token 級的回應收集與遷移處理〔摘要〕。
  3. 訓練吞吐 1.51–1.97 倍、成本效率好 28–49%〔摘要〕。
- **和 D8 的關係**：方向 6：「搶佔」在 RL 裡有兩種意思（引擎內 KV 不夠、雲端 GPU 被收回），D8 只看前者。
