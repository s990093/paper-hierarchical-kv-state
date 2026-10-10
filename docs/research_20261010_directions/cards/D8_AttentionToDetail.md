# D8 卡片：Attention to Detail: Evaluating Energy, Performance, and Accuracy Trade-offs Across vLLM Configurations

- **連結**：https://arxiv.org/abs/2607.09172
- **venue／年份**：arXiv 2607.09172（2026-07）；venue 未查證
- **讀了哪裡**：arXiv 摘要頁
- **三行摘要**：
  1. 控制實驗：attention kernel、prefix caching、chunked prefill 三個 vLLM 選項，5 個模型 × 5 個任務，共 9,000 次 run〔摘要〕。
  2. 能耗與效能主要受 attention 種類與 prefix caching 影響〔摘要〕。
  3. 沒有一組設定到處最好〔摘要〕。
- **和 D8 的關係**：方向 1：prefix caching（也就是「載入代替重算」）對能耗的影響有人量過，但是在 vLLM 設定層級，不是 κ 層級。
