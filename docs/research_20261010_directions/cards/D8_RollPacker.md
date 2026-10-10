# D8 卡片：RollPacker: Mitigating Long-Tail Rollouts for Fast, Synchronous RL Post-Training

- **連結**：https://arxiv.org/abs/2509.21009
- **venue／年份**：arXiv 2509.21009（2025-09）；venue 未查證
- **讀了哪裡**：arXiv 摘要頁
- **三行摘要**：
  1. 把會產生長尾回答的 prompt 集中到少數「長回合」，其他回合只有短回答〔摘要〕。
  2. 配合 rollout 的彈性平行度、獎勵階段排程、串流訓練〔摘要〕。
  3. 128 張 H800 上比 veRL 快 2.03–2.56 倍〔摘要〕。
- **和 D8 的關係**：方向 6 的對照：用排程避免搶佔，而不是讓搶佔變便宜。
