# B_Sibyl Sibyl: Adaptive and Extensible Data Placement in Hybrid Storage Systems Using Online Reinforcement Learning

- **出處**：Gagandeep Singh、Rakesh Nadig、Jisung Park、Rahul Bera、Nastaran Hajinazar、David Novo、Juan Gómez-Luna、Sander Stuijk、Henk Corporaal、Onur Mutlu，ISCA '22（New York，2022-06-18～22）〔原文 p.1〕。
  本機：`/mlsteam/data/tiara/papers/sibyl2022_arXiv2205.07394.pdf`（arXiv 2205.07394）。讀了 p.1、p.3–6、p.9、p.12。
  不在 `03_paper_map.md` 的 61 篇裡。
- **寫入時做了什麼決定**：**每個 I/O 請求都當下決定放快的或慢的裝置**，用線上強化學習。reward 裡有一項 **eviction penalty**：快的層空間不夠時，會在背景把頁逐出到慢的層，逐出花的時間要扣分。這讓 agent「learn to restrain itself from placing」非關鍵的頁在快的層〔原文 p.5–6〕。
- **用什麼資訊做決定？寫完之後還在不在？（N1）**：用請求的特徵（大小、類型、存取次數等）、目前的放置位置、裝置狀態，再加上執行時的回饋。這些資訊之後大多還在，所以不是 N1。
- **有沒有和延後版、寫穿版、背景版比較？**
  - 快的層限制為工作集的 10%，確保會發生逐出〔原文 p.3〕。
  - 對手〔原文 p.3–4〕：
    - CDE：熱的或隨機的寫入放快的層，冷的或循序的放慢的層；
    - **HPS**：依存取次數，**週期性把冷頁搬到慢的層**（背景搬）；
    - Archivist、RNN-HSS（學習式）；
    - Oracle。
  - H&M 組態（兩個裝置延遲差距小）：比 CDE 好 28.1%、比 HPS 好 23.2%。H&L 組態（差距大）：比 CDE 好 19.9%、比 HPS 好 45.9%。達到 Oracle 的 80%〔原文 p.9〕。
  - 原文判讀：「the larger the latency gap between HSS devices, the higher the expected benefits of avoiding the eviction penalty by placing only performance-critical pages in the fast storage」；「aggressive placement in the fast storage is not beneficial for long-term performance」〔原文 p.9〕。
  - 逐出次數：H&M 下比 CDE 少 68.4%、比 HPS 少 43.2%。CDE 放進快的層的資料多、逐出也多，但兩個裝置差距大（H&L）時，CDE 在其他對手中最好〔原文 p.12〕。
- **硬體**（真實系統）〔原文 p.9 表 3〕：
  - H：Intel Optane SSD P4800X（375 GB，PCIe 3.0 NVMe）；
  - M：1.92 TB SATA TLC SSD；
  - L：Seagate HDD ST1000DM010（1 TB，7200 RPM）；
  - 組成 H&M（效能導向）、H&L（成本導向）兩種組態，另有三層組態。
- **模型架構、模態**：不適用。
- **和本研究的關係**：
  - **支持「寫入時決定」的一種條件**：快的層很小（10% 工作集）、逐出要花時間時，寫入當下就別把不關鍵的資料放進快的層，可以省下逐出（N2＋N3）。
  - **但它不是 10 §2 的延後測試**〔判讀〕：HPS（背景版）用的是**另一條**規則（存取次數），不是「同一條規則延後套用」。Sibyl 贏 HPS 23–46%，可能來自規則不同，不能歸因於「寫入時」。
  - 方法上可以借用：把「之後的逐出成本」直接放進寫入時決定的目標函數。這和 S5 想表達的一致，只是 S5 用的是位置規則。
- **證據等級**：〔原文 p.X〕；與延後測試的差異〔判讀〕。
