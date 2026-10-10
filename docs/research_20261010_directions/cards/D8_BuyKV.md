# D8 卡片：Can I Buy Your KV Cache?

- **連結**：https://arxiv.org/abs/2606.13361
- **venue／年份**：arXiv 2606.13361（2026-06，單一作者）；venue 未查證
- **讀了哪裡**：arXiv 摘要頁（PDF 已下載，未細讀）
- **三行摘要**：
  1. 提議文件發布者先算好 KV，其他 agent 付費載入、跳過 prefill；載入後逐 token 與重算一致〔摘要〕。
  2. Qwen3-4B 上重用比 prefill 省 9–50 倍計算〔摘要〕。
  3. 把 KV 傳出去不划算（幾乎壓不動，流量費比省下的 prefill 貴）；留在供應商端才划算〔摘要〕。
- **和 D8 的關係**：方向 2：用「錢」看存 vs 算，但沒有分層、沒有部分重算。
