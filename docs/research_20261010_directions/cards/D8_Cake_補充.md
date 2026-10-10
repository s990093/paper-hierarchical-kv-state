# D8 卡片：Compute Or Load KV Cache? Why Not Both?（Cake）——波動與排程補充

- **連結**：https://proceedings.mlr.press/v267/ （本機 PMLR PDF）；arXiv https://arxiv.org/abs/2410.03065
- **venue／年份**：ICML 2025（PMLR 267；Lit-A 已查證）
- **讀了哪裡**：本機 PDF p.1–9（`/mlsteam/data/tiara/papers/02_Cake_ICML25_PMLR_v267.pdf`）
- **三行摘要**：
  1. §5.7：隨機抽「算力預算 0–512 token」與「I/O 0–25 Gbps」的波動軌跡，只給一張軌跡圖（Fig. 5），沒有 p99 之類的分布統計〔原文 p.8〕。
  2. §5.8：一個 16K 的 prefix 請求加 22 個短請求，Cake 的適應式排程把完成時間從 1.5 s 降到 1.19 s（吞吐 +26%）〔原文 p.8〕。§6：說可和推測解碼、P/D 分離並用〔原文 p.9〕。
  3. Table 2 的頻寬設定是 7–100 Gbps（雲端 egress、SSD、RoCE）〔原文 p.6〕。
- **和 D8 的關係**：方向 3、4 的「已經做過什麼」：波動只有定性展示；和其他請求共用算力只有一個例子。
