# D8 卡片：Activated LoRA: Fine-tuned LLMs for Intrinsics

- **連結**：https://arxiv.org/abs/2504.12397
- **venue／年份**：arXiv 2504.12397（2025-04）；venue 未查證
- **讀了哪裡**：arXiv 摘要頁
- **三行摘要**：
  1. 一般 LoRA 切換時，整段歷史的 KV 都要用 LoRA 權重重算〔摘要〕。
  2. aLoRA 只改「被呼叫之後」的 token 的權重，所以能直接吃 base model 的 KV〔摘要〕。
  3. 已併入 Hugging Face PEFT〔摘要〕。
- **和 D8 的關係**：方向 12：從模型設計上讓跨變體 KV 重用變成無損，削弱「系統端部分重算」的需要。
