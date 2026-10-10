# D8 卡片：RaReCache: Bridging the Gap in Cross-Model KV Cache Reuse via Rank disagreement-based Selective Recomputation

- **連結**：https://arxiv.org/abs/2610.11358
- **venue／年份**：arXiv 2610.11358（2026-10-08）；venue 未查證
- **讀了哪裡**：arXiv 摘要頁
- **三行摘要**：
  1. 小模型先 prefill，用線性映射把 KV 轉給大模型；轉換失敗集中在少數資訊密集的 token〔摘要〕。
  2. 用 rank disagreement 指標挑出關鍵位置重算〔摘要〕。
  3. Qwen3-0.6B→14B 重算 30% 位置保留 95–99% 準確度；最多 3.04 倍 prefill 加速〔摘要〕。
- **和 D8 的關係**：方向 12：跨模型 KV 重用＋選擇性重算是 2026 年 10 月的熱點，有損。
