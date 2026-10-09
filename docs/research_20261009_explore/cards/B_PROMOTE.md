# B_PROMOTE On Multi-level Exclusive Caching: Offline Optimality and Why promotions are better than demotions

- **出處**：Binny S. Gill（IBM Almaden），FAST '08（6th USENIX Conference on File and Storage Technologies），印刷 pp.49–64。
  連結：https://www.usenix.org/legacy/event/fast08/tech/full_papers/gill/gill.pdf
  頁碼規則：p.X＝PDF 第 X 頁，印刷頁＝p.X＋48。讀了 p.1–4、p.6–7、p.13–14、p.16。
- **寫入時做了什麼決定**：PROMOTE 是**「資料經過時就決定放哪一層」**，不是逐出時才往下搬：
  - READ 回覆往上傳時帶一個 promoteHint。每一層依機率 probPromote 決定要「擁有」這頁（留下來，並把 hint 設成 false），還是繼續往上送；最上層的 probPromote＝0〔原文 p.6–7〕。
  - 所以新資料可能在**較低層就被留下**，不一定先進最上層。常被命中的頁會一層層往上爬〔原文 p.7〕。
  - probPromote 會自適應，目標是讓各層「被逐出的頁的有用程度」相等（讓各層的 cache life 相等）〔原文 p.7〕。
- **用什麼資訊做決定？寫完之後還在不在？（N1）**：用機率，加上上層週期回報的 cache life。沒有用到「寫完就消失」的資訊。
- **有沒有和延後版、寫穿版、背景版比較？**（PROMOTE 是「放置時決定」，DEMOTE 是「延後版」）
  - **頻寬不限、demote 免費（對 DEMOTE 最有利）**：trace P1 上，PROMOTE 只比 DEMOTE 好**最多 4%**（LRU）／**5%**（ARC）。其他 trace 平均只好 **0.3%**（LRU）／**1.5%**（ARC）〔原文 p.13〕。
  - **頻寬受限**（300 blocks/s，是「沒有命中也沒有 demote 時所需」的 1.5 倍）：平均回應時間 PROMOTE 3.42 ms（LRU）、3.21 ms（ARC），DEMOTE 5.05 ms、5.43 ms。DEMOTE 比單純 LRU 還差；cache 小時甚至比完全不快取（tm＝5 ms）還差〔原文 p.13〕。
  - DEMOTE 用掉的層間頻寬是 PROMOTE 的 2 倍〔原文 p.3、p.13〕。圖 12 把頻寬分成「頻寬敏感」與「頻寬不敏感」兩區，頻寬不充裕時 DEMOTE 比 LRU 差〔原文 p.14〕。
  - 對背景版的看法：「idle time should never be considered free」〔原文 p.3〕。
  - 對 hold 的描述：頻寬緊或負載高時，「reads stall until demotions … can create space for the new page」〔原文 p.3〕。
  - 離線最佳的上下界 OPT-UB、OPT-LB，兩層時相差 2.18%，三層時相差 2.83%〔原文 p.1〕。
- **硬體**：trace 驅動的模擬，回應時間模型〔原文 p.4、p.13〕。
- **模型架構、模態**：不適用。
- **和本研究的關係**：
  - **這是 S5（放置時決定）對 S4（延後 demote）最直接的經典對照**。結論和 08 一樣：**傳輸免費時，兩者差距 ≤5%**。只有層間頻寬是瓶頸（N2）時，放置時決定才大贏。
  - 支持的 H：只在 N2 成立時（H1 寫入量、H12 三層、H3 併發時共用頻寬）。
  - **KV 的關鍵差異**〔判讀〕：Gill 的 DEMOTE 多付的是一次額外的跨層傳輸（client→array）。KV 的延後版（S4）不會多寫一次 SSD：前段的 KV 在 S5 和 S4 都要寫一次 SSD，S4 多的只是一次 CPU DRAM 讀，以及時間點（可能卡在請求路徑上）。所以 Gill 的 N2 優勢在 KV 上大多不存在，**除非** DRAM 頻寬、host→NVMe 路徑或 PCIe 是瓶頸（E4、E5）。這也解釋了 08 為什麼 S4B 追得上。
- **證據等級**：數字〔原文 p.X〕；KV 類比〔判讀〕。
