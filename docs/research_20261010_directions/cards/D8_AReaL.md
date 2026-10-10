# D8 卡片：AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning

- **連結**：https://arxiv.org/abs/2505.24298
- **venue／年份**：arXiv 2505.24298；搜尋結果顯示 NeurIPS 2025 proceedings 有此文，但本 agent 沒開該頁 → venue 未查證
- **讀了哪裡**：PDF p.2–4、p.24（本機 d8_20261010/areal2025_arXiv2505.24298.pdf）
- **三行摘要**：
  1. 完全非同步 RL：rollout 一直生成，訓練端隨時更新權重〔摘要〕。
  2. 更新權重時「rollout worker 丟掉用舊權重算的 KV，用新權重重算」，然後繼續生成〔原文 p.4〕。
  3. 實驗設定：每個 prompt 16 個答案，最大生成長度 27,648 token〔原文 p.24〕。
- **和 D8 的關係**：方向 7（過期 KV）：代表「全部重算」那一端。
