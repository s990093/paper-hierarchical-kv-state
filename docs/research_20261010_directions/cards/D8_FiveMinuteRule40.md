# D8 卡片：Five-Minute Rule 40 Years Later: A First-Principles Revisit for Modern Memory Hierarchy

- **連結**：https://arxiv.org/abs/2511.03944
- **venue／年份**：ISCA 2026（arXiv journal-ref；會議頁未開）
- **讀了哪裡**：PDF p.1–3、p.6、p.10、p.14 參考文獻（本機 d8_20261010/fiveminute2026_arXiv2511.03944.pdf）
- **三行摘要**：
  1. 重推 Gray & Putzolu 1987 的五分鐘規則，加上主機成本、DRAM 頻寬／容量、SSD 物理模型〔原文 p.1–3〕。
  2. 結論：GPU 主機配高 IOPS SSD 時，DRAM↔flash 的損益兩平時間從「分鐘」縮到「秒」〔原文 p.1、p.6〕；粗粒度 KV tensor 的兩平時間更短〔原文 p.10〕。
  3. 只比 DRAM 和 flash 兩種「存」；沒有把「丟掉、需要時重算」當成第三個選項（全文搜 recompute 只在別的意思出現）〔判讀〕。
- **和 D8 的關係**：方向 2 的前作。D8 的差別：把重算（以及 Cake 的部分重算）當成一層，並用兩個平台實測的 κ。也提供 Gray & Putzolu 1987 的書目〔原文 p.14 參考文獻 [19]〕。
