# D8 卡片：Where Does the Energy Go? Profiling LLM Agent Inference on Blackwell GPUs

- **連結**：https://arxiv.org/abs/2609.29707
- **venue／年份**：arXiv 2609.29707（2026-08）；venue 未查證
- **讀了哪裡**：arXiv 摘要頁
- **三行摘要**：
  1. 同時用 NVML（GPU）、RAPL（CPU／DRAM）、IPMI（整機）量 agent 推論的能耗〔摘要〕。
  2. 只看 GPU 計數器會漏掉整機能耗的 41–45%〔摘要〕。
  3. 推理型負載在每張 GPU 400–600 W 功耗上限下吞吐不變，300 W 時大掉〔摘要〕。
- **和 D8 的關係**：方向 1：能耗的「計算邊界」（只算 GPU 還是整機）會改變結論；本機 container 讀不到 RAPL（見主文 §4.1）。
