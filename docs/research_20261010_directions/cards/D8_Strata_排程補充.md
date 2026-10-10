# D8 卡片：Strata（OSDI'26）——排程部分補充

- **連結**：本機 `/mlsteam/data/tiara/papers/04_Strata_OSDI26.pdf`；arXiv https://arxiv.org/abs/2508.18572
- **venue／年份**：OSDI 2026（PDF 頁尾「20th USENIX Symposium on Operating Systems Design and Implementation」）
- **讀了哪裡**：本機 PDF p.2–8
- **三行摘要**：
  1. 排程器把「CPU→GPU 頻寬」當一級資源：組 batch 時讓載入配上足夠的計算來遮住載入時間〔原文 p.5、p.7〕。
  2. 載入還是卡住時，在空泡裡插入 decode 等有用的計算〔原文 p.7〕；並處理 delay hit（同一段 context 的多個請求在 miss 還沒解決時到達）〔原文 p.5、p.7–8〕。
  3. Mooncake toolagent trace 裡 38% 的請求在 1 秒內和別人共用 ≥6K token 的 prefix〔原文 p.5〕。
- **和 D8 的關係**：方向 15（跨請求 Cake）：跨請求「載入配計算」已經有了；Strata 的計算是別人的新 token，不是重算自己的快取。
