# D8 卡片：MatKV: Trading Compute for Flash Storage in LLM Inference

- **連結**：https://arxiv.org/abs/2512.22195
- **venue／年份**：ICDE 2026（arXiv comment 自述「Accepted for publication in ICDE 2026」；會議頁未開）
- **讀了哪裡**：PDF p.1–4、p.6（本機 `/mlsteam/data/tiara/papers/d8_20261010/matkv2026_arXiv2512.22195.pdf`）
- **三行摘要**：
  1. RAG 文件的 KV 先算好存在 flash，用的時候載入，不在 GPU 重算〔原文 p.1〕。
  2. 「十天規則」：仿 Gray 的五分鐘規則，用 GPU 價格、載入時間、SSD 價格算出「至少每 10 天被讀一次就值得存」〔原文 p.3–4，式 (1)〕。
  3. 能耗：H100 重算 1,024 token「約 175 J（峰值功耗 350 W）」，看起來就是 350 W × 0.5 s 算出來的〔原文 p.3；「算出來」是判讀〕；SSD 讀取只算 SSD 本身 7 W，所以說能耗低 1,200 倍〔原文 p.3、p.6〕。沒有算「等待載入時 GPU 閒置也在耗電」，也沒有「一半重算一半載入」的混合。
- **和 D8 的關係**：方向 1（κ_E）與方向 2（κ_$）最近的前作。D8 要補的是：GPU 閒置功耗、混合還原（Cake）、DRAM 層、跨硬體的 κ。
