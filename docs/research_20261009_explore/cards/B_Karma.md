# B_Karma Karma: Know-it-All Replacement for a Multilevel cAche

- **出處**：Gala Yadgar（Technion）、Michael Factor（IBM Haifa）、Assaf Schuster（Technion），FAST '07（5th USENIX Conference on File and Storage Technologies），印刷 pp.169–184。
  連結：https://www.usenix.org/legacy/event/fast07/tech/full_papers/yadgar/yadgar.pdf
  頁碼規則：p.X＝PDF 第 X 頁，印刷頁＝p.X＋168。讀了 p.1–3、p.5。
- **寫入時做了什麼決定**：放哪一層，在資料被讀取（放入 cache）時就決定：
  - 應用程式（資料庫的 query optimizer）給 hint，把 disk block 分成多個 range（循序、迴圈、隨機、存取頻率）。每個 range 依 marginal gain 分到某一層的一塊分區，**marginal gain 越高的 range 放越上層**〔原文 p.2〕。
  - 讀取分成 READ（下層刪掉自己的副本）和 READ-SAVE（下層保留）〔原文 p.3〕。某個 range 在上層沒有配到空間時，用 READ-SAVE 讀，讀完直接丟，**不做 DEMOTE**，「to avoid unnecessary overhead」〔原文 p.5〕。
  - hint 改變時「lazy repartitioning」：不實際搬資料，只把不屬於新分區的塊標成優先逐出〔原文 p.5〕。
- **用什麼資訊做決定？寫完之後還在不在？（N1）**：用應用程式的 hint，在存取前就知道。**下層 cache 自己拿不到這些資訊**：論文把「下層缺少 block 的屬性（屬於哪個檔案、哪個應用）」列為多層快取的三大問題之一〔原文 p.2〕。這是一種**資訊不對稱**：寫入者知道，下層與之後的觀察者不知道。
- **有沒有和延後版、寫穿版、背景版比較？**
  - 對手：LRU、2Q、ARC、MultiQ、LRU-SP、Demote，比的是加權 I/O 成本。查詢排列的 trace 上，Karma 平均比 LRU 好最多 85%；相對於 LRU，Karma 比最好的 LRU 類策略（Demote）再多好 50%，比最好的 informed 策略（LRU-SP）再多好 25%〔原文 p.2〕。
  - **aggregate cache 很小（<3% 資料集）時，Karma 也被 DEMOTE 的成本拖累**：demote 下去的塊還沒被讀就被丟了〔原文 p.2〕。
  - 成本模型：demote 成本等於下一層的存取成本，C1＝D1＜C2＝D2＜…＜CDisk〔原文 p.3〕。
- **硬體**：模擬兩層 cache 加一層 storage。PostgreSQL 的 trace（用 explain 當 hint 來源），以及 Zipf 合成 trace〔原文 p.2〕。
- **模型架構、模態**：不適用。
- **和本研究的關係**：
  - **支持 N1 的形式**：寫入者手上有「之後的層拿不到」的資訊時，在放置時就決定，能省掉一次 demote（N2）。
  - **但 KV 的位置資訊不是這種**〔判讀〕：位置一直都在，S4B 延後也看得到，所以 Karma 的資訊優勢搬不到「依位置放」。它只適用於真的只在寫入時才知道的資訊，例如 H4（RAG／共享前綴：寫入的系統知道「這段會被很多請求讀」）、H8（應用層知道 session 會不會回來）。
  - 補充：Gill FAST'08 把 ULC（Jiang & Zhang，ICDCS 2004）描述成「由最上層對下層發 RETRIEVE 與 DEMOTE 命令來控制整個階層」〔Gill 原文 p.2〕。ULC **未讀原文**。
- **證據等級**：〔原文 p.X〕；和 H4／H8 的對應〔判讀〕。
