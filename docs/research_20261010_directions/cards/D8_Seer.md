# D8 卡片：Seer: Online Context Learning for Fast Synchronous LLM Reinforcement Learning

- **連結**：https://arxiv.org/abs/2511.14617
- **venue／年份**：arXiv 2511.14617（2025-11）；venue 未查證
- **讀了哪裡**：PDF p.1–2（本機 d8_20261010/seer2025_arXiv2511.14617.pdf）
- **三行摘要**：
  1. RL rollout 時，CoT 請求的 KV 從幾百 MB 長到幾十 GB；系統只能縮 batch 或搶佔（preempt），「搶佔特別貴，因為要重新 prefill」〔原文 p.1〕。
  2. Seer 把同一 prompt 群組的請求拆成小段排程，用改自 Mooncake 的全域 KV 池在實例間搬，「省掉重算」〔原文 p.2〕。
  3. rollout 吞吐最多 2.04 倍，長尾延遲少 72–94%〔摘要〕。
- **和 D8 的關係**：方向 6（RL 搶佔）的前作：證明搶佔成本是真問題；它的做法是「載入」不是「重算」。
