# D8 卡片：Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve

- **連結**：https://arxiv.org/abs/2403.02310
- **venue／年份**：arXiv 2403.02310；03_paper_map 記為 OSDI'24，本 agent 未開會議頁 → 未查證
- **讀了哪裡**：arXiv 摘要頁
- **三行摘要**：
  1. prefill 吃滿算力、decode 算力利用率低〔摘要〕。
  2. chunked prefill：把 prefill 切成小塊，搭著 decode 一起跑，不暫停 decode（stall-free）〔摘要〕。
  3. 大 batch 提高吞吐，同時壓住延遲〔摘要〕。
- **和 D8 的關係**：方向 4：Cake 的重算線本質上就是 chunked prefill，可以搭 decode 的便車。
