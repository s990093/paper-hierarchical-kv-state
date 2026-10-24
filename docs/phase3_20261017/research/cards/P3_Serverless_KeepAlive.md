# P3 卡：serverless 的 keep-alive／pre-warm（Serverless in the Wild、FaasCache）

> 和「KV 在 CPU 留多久、何時從慢層預取回來」是同構問題，所以把機制與評估方式查清楚。

## Serverless in the Wild（Shahrad et al., USENIX ATC'20）
- **連結**：<https://www.usenix.org/system/files/atc20-shahrad.pdf>；讀了 PDF p.8–10、p.13。
- **決定什麼**：每個應用的 pre-warm 視窗（執行結束後多久重新載入）與 keep-alive 視窗（載入後留多久）；pre-warm＝0 表示執行完不卸載〔原文 p.8〕。
- **怎麼從分布變成決策**〔原文 p.8〕：
  - 每個應用一個閒置時間（IT）直方圖，1 分鐘一格，最長追蹤 4 小時（可設定）；超過的算 out-of-bound。
  - 頭＝第 5 百分位 → pre-warm 視窗；尾＝第 99 百分位 → keep-alive 視窗；落在格內時頭往下取整、尾往上取整；再留 10% 餘裕（pre-warm 少 10%、keep-alive 多 10%）。
- **退路**〔原文 p.8–9〕：
  - 直方圖不具代表性（觀察的 IT 太少，或 bin 計數的 CV 低於門檻，表示分布太散）→ 標準 keep-alive：pre-warm＝0、keep-alive＝直方圖範圍（例如 4 小時）；CV 用 Welford 線上演算法追蹤。
  - out-of-bound 很多的低頻應用 → ARIMA 預測下次 IT，留 15% 餘裕（例：預測 5 小時 → pre-warm 4.25 小時、keep-alive 1.5 小時）。
- **為什麼主動卸載**：FaaS 按配置時間計費，所以主動卸載，不等別人需要空間；作者說最接近「新存取會重設 TTL 的 TTL 快取」，而 pre-warm 相當於「時間上的預取」，這兩者以前的快取研究沒有同時考慮〔原文 p.13〕。
- **評估**〔原文 p.9–10〕：用 Azure Functions 第一週的真實呼叫時間做模擬；指標是第 3 四分位應用的冷啟動百分比 vs 正規化的浪費記憶體時間（應用載入但沒在執行的時間），畫成 Pareto；固定 keep-alive 從 10 分鐘拉到 1 小時，75 百分位應用的冷啟動從 50.3% 降到 25%；另在 Apache OpenWhisk 實作。

## FaasCache（Fuerst & Sharma, ASPLOS'21）
- **連結**：<https://homes.luddy.indiana.edu/prateeks/papers/faascache-asplos21.pdf>；讀了 PDF p.1、p.5、p.10。
- **機制**：把 keep-alive 當快取，Greedy-Dual-Size-Frequency：Priority＝Clock＋Freq×Cost／Size；容量滿時逐出優先權最低的，Clock 設成被逐出者的優先權〔原文 p.5〕。
- **和 Shahrad 的關係**：作者重現了直方圖策略（HIST），稱它「實際上是 TTL＋Prefetching」：用 IAT 直方圖預測下一次呼叫、主動逐出；CoV≤2 時用應用自己的預載與 TTL，否則用 2 小時通用 TTL〔原文 p.10〕。
- **結果**：比現行做法降低冷啟動開銷 3 倍以上〔原文 p.1〕；原文指出當平均 IAT 很短（<36 ms）時，work-conserving 與否的差別被熱門函式蓋過〔原文 p.10〕。

## 和第三階段的對應〔判讀〕
| serverless | 第三階段 |
|:--|:--|
| 冷啟動 | 回來時要從慢層還原（或重算） |
| keep-alive 視窗 | 留在 CPU 多久才降級 |
| pre-warm（時間上的預取） | 何時把慢層的 session 預取回 CPU |
| 浪費的記憶體時間 | CPU 被沒回來的 session 占用的時間 |
| 按配置時間計費 → 主動卸載（Shahrad） | **不適用**：CPU 層是容量滿才降級（FaasCache 式的 work-conserving） |
- 「5 百分位 pre-warm、99 百分位 keep-alive」需要的是回來時間的**分布**，不是點估計；這是 L3（存活）或 L6（分位數）比 L1（二元）多出來的用處。
- 「直方圖不具代表性就退回標準 keep-alive」是一個不需要學習的退路，可以直接搬：session 歷史太少時退回 LRU。
- 評估應該畫兩個目標的 Pareto（回來 TTFT vs 預取／慢層位元組），不是只報一個數字。
