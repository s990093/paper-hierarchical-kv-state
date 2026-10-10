# D8 卡片：Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live

- **連結**：https://arxiv.org/abs/2511.02230
- **venue／年份**：arXiv 2511.02230（v7）；venue 未查證
- **讀了哪裡**：PDF p.1–5（本機 d8_20261010/continuum2025_arXiv2511.02230.pdf）
- **三行摘要**：
  1. agent 呼叫工具時，把 KV 釘在 GPU 一段 TTL（存活時間），TTL 由「重新載入／重算的成本」和「被趕出去之後的排隊延遲」決定〔原文 p.1–2〕。
  2. 原文明說：有 LMCache 這種快速非同步卸載時，重新載入很便宜，但「就算瞬間載回，回來的請求還是要排隊等 GPU 記憶體」〔原文 p.5〕。
  3. 摘要報告平均 job 完成時間改善 8 倍以上（SWE-Bench、BFCL、OpenHands）〔摘要〕。
- **和 D8 的關係**：方向 9（工具呼叫 TTL＋Cake）的最近前作，也是殺掉它的主要理由：排隊才是大頭，Cake 只讓載回更快。
