# D8 卡片：CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving

- **連結**：https://arxiv.org/abs/2310.07240
- **venue／年份**：SIGCOMM 2024（arXiv comment「SIGCOMM'24」）
- **讀了哪裡**：PDF p.2、p.6（本機 d8_20261010/cachegen2024_arXiv2310.07240.pdf）
- **三行摘要**：
  1. KV 切 chunk 串流傳輸，每個 chunk 依頻寬選壓縮等級，讓網路延遲在 SLO 內〔原文 p.2〕。
  2. 頻寬太低時，某個 chunk 可以改傳文字，讓 LLM 自己重算那段 KV〔原文 p.2、p.6〕。
  3. 所以「逐 chunk 在載入與重算之間選」早就有（但它的主軸是有損壓縮）〔判讀〕。
- **和 D8 的關係**：方向 3 的對照組之一（逐 chunk 靜態選擇）。
