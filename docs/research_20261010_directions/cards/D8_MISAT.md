# D8 卡片：Scheduling Mixed RL Rollouts Beyond Prefix Locality (MISA-T)

- **連結**：https://arxiv.org/abs/2608.11152
- **venue／年份**：arXiv 2608.11152（2026-08）；venue 未查證
- **讀了哪裡**：arXiv 摘要頁
- **三行摘要**：
  1. RLVR、RLHF、agentic rollout 共用一個非同步推論服務時，KV 駐留時間差很多〔摘要〕。
  2. MISA-T：路由層的准入策略，按工作負載分配 KV 容量、按駐留時間記帳〔摘要〕。
  3. rollout 吞吐比調過的 cache-aware vLLM Router 高 53.3%／43.6%〔摘要〕。
- **和 D8 的關係**：方向 6：RL rollout 的 KV 容量競爭已經有人從准入角度做。
