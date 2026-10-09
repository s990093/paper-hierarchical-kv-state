# B_EvictionPlacement Eviction Based Cache Placement for Storage Caches

- **出處**：Zhifeng Chen、Yuanyuan Zhou（UIUC）、Kai Li（Princeton），USENIX ATC 2003 General Track（San Antonio，2003-06-09～14），pp.269–282（頁碼取自 Gill FAST'08 參考文獻 [6]）。
  連結：https://static.usenix.org/events/usenix03/tech/full_papers/chen/chen.pdf
  頁碼規則：p.X＝PDF 第 X 頁，印刷頁＝p.X＋267（PDF p.1 是封面）。讀了 p.2、p.8–9。
  **滾雪球第 1 輪**：從 Gill FAST'08 的參考文獻 [6] 追到。
- **寫入時做了什麼決定**：論文把「**什麼時候放進下層 cache**」直接當成變數〔原文 p.2〕：
  - access-based placement：存取當下就放（等於「寫入時」放），傳統做法，也是 inclusive 的來源；
  - **eviction-based placement**：延後到「上層把這塊逐出」時才放進 storage cache。
  - 延後放的資料要重新取得。他們**不用** client 的 DEMOTE，改成讓 storage 自己從 disk **reload**〔原文 p.8–9〕。
  - 存取次數低於門檻的塊不 reload。reload 以低優先權排在獨立佇列，只在沒有 demand 請求搶同一顆 disk 時才發。論文明說「reload 沒做也完全沒關係」〔原文 p.9〕。
- **用什麼資訊做決定？寫完之後還在不在？（N1）**：用的是上層的逐出事件（透過 client content tracking table 取得）與存取次數〔原文 p.2、p.9〕。這個資訊在存取當下**還不存在**，延後才有，和 N1 剛好相反：等待帶來了資訊。
- **有沒有和延後版、寫穿版、背景版比較？**
  - eviction-based 對 access-based：模擬中命中率最多高 500%；OLTP 交易率 1.2 倍〔原文 p.2 摘要〕。
  - 他們列出 DEMOTE（同步的延後搬）的三個缺點〔原文 p.8–9〕：
    1. client→storage 流量幾乎加倍；
    2. **client 的 miss 太突發、蓋不住 demote 成本時，一個 miss 要等 demote 做完、拿到空的 buffer，才能送出讀取**。這就是本研究的 hold；
    3. demote 的時間窗太短，無法排程或批次。
  - 取捨條件：SAN 頻寬大於 disk 總頻寬的環境裡，DEMOTE 可能比較好〔原文 p.9〕。
- **硬體**：storage system 接 Microsoft SQL Server；文中以 60–100 顆 SCSI disk、1 Gbps client–storage 互連做頻寬推論〔原文 p.9〕。
- **模型架構、模態**：不適用。
- **和本研究的關係**：
  - **威脅（整體）**：在多層快取裡，「延後放置」被證明比「存取時放置」好，原因是存取時放會和上層重複。
  - **它就是 10 §2 的「背景版」**：放棄同步 demote，改成低優先、可以不做的背景搬移，理由正是「同步搬移卡在請求路徑上」。背景版成立的前提是 disk 頻寬有閒置。對應 H3（N3：沒有空閒時，背景版就失效）。
  - KV 的類比〔判讀〕：KV 丟了可以重算，所以「背景搬沒做也沒關係」在 KV 上比儲存區塊更成立。這讓背景版更強。
- **證據等級**：數字與引述〔原文 p.X〕；和 H3／背景版的對應〔判讀〕。
