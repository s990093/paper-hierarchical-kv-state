# D8 卡片：Understanding Bottlenecks for Efficiently Serving LLM Inference with KV Offloading（能耗補充）

- **連結**：https://arxiv.org/abs/2601.19910
- **venue／年份**：arXiv 2601.19910；本機檔名寫 MLSys26，但本 agent 沒查會議頁 → venue 未查證（03_paper_map 已收此文）
- **讀了哪裡**：本機 PDF p.1、p.2、p.8（`/mlsteam/data/tiara/papers/07_Bottlenecks_MLSys26_arXiv2601.19910.pdf`）
- **三行摘要**：
  1. KV 卸載到 CPU 時，99% 的延遲花在傳輸上〔原文 p.1–2〕。
  2. 服務卸載請求時，GPU 平均只用到 TDP 的 28%（ShareGPT）與 22%（NarrativeQA）〔原文 p.8〕。
  3. 作者的結論是電力與散熱基礎設施閒置，單位工作成本變高〔原文 p.8〕。
- **和 D8 的關係**：方向 1：等待載入時 GPU 的功耗不是零，也不是滿載——能耗帳要算這段。
