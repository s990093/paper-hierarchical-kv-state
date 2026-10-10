# D8 卡片：DroidSpeak: KV Cache Sharing for Cross-LLM Communication and Multi-LLM Serving

- **連結**：https://arxiv.org/abs/2411.02820
- **venue／年份**：arXiv 2411.02820（2024-11）；venue 未查證
- **讀了哪裡**：arXiv 摘要頁（PDF 已下載到 d8_20261010，未細讀）
- **三行摘要**：
  1. 同架構的不同 LLM（例如同一 base 的微調版）共用 prefix KV〔摘要〕。
  2. 選擇性重算少數幾層，其餘層沿用別的模型的 KV；並把逐層重算和載入 pipeline 起來〔摘要〕。
  3. 吞吐最多 4 倍、prefill 快約 3.1 倍，品質損失可忽略〔摘要〕。
- **和 D8 的關係**：方向 12（跨模型／LoRA）：「一邊重算一邊載入」在層這個維度已經有人做了，D8 只剩「位置×層」二維。
