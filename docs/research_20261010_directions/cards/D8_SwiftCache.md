# D8 卡片：SwiftCache: Efficient LLM Serving for Multi-turn Conversations with Heterogeneous KV Cache Sharing

- **連結**：https://arxiv.org/abs/2606.16135
- **venue／年份**：arXiv 2606.16135（2026-06）；venue 未查證
- **讀了哪裡**：arXiv 摘要頁
- **三行摘要**：
  1. KV 需求低的模型把閒置 GPU 記憶體借給需求高的模型存 prefix cache，經 NVLink 傳，避開 PCIe〔摘要〕。
  2. 只把目前這一層的 KV 留在本地 GPU〔摘要〕。
  3. P99 TTFT 最多少 69%〔摘要〕。
- **和 D8 的關係**：方向 5：多卡節點裡「快的層」可以不是主機 DRAM（同類想法）。
