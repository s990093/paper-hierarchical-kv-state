# D8 卡片：Attention Is All You Need for KV Cache in Diffusion LLMs (Elastic-Cache)

- **連結**：https://arxiv.org/abs/2510.14973
- **venue／年份**：arXiv 2510.14973（2025-10）；venue 未查證
- **讀了哪裡**：arXiv 摘要頁
- **三行摘要**：
  1. 擴散式 LLM 每一步都重算所有 token、所有層的 QKV，但大部分步驟 KV 變化很小，淺層尤其小〔摘要〕。
  2. Elastic-Cache 決定「何時刷新」（看最受關注 token 的漂移）與「刷新哪裡」（從某一層開始往深層重算）〔摘要〕。
  3. 權重固定，變的是輸入（遮罩 token）〔判讀〕。
- **和 D8 的關係**：方向 7：「部分刷新過期 KV」在別的場景有先例；RL 權重更新會改每一層，深層規律不一定適用〔判讀〕。
