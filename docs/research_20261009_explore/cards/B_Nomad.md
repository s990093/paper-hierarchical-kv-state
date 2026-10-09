# B_Nomad Nomad: Non-Exclusive Memory Tiering via Transactional Page Migration

- **出處**：Lingfeng Xiang、Zhen Lin、Weishu Deng、Hui Lu、Jia Rao（UT Arlington）、Yifan Yuan、Ren Wang（Intel Labs），OSDI '24（USENIX 頁面 https://www.usenix.org/conference/osdi24/presentation/xiang，網路搜尋確認）。
  **讀的是 arXiv 2401.13154v2（2024-06-18）**，17 頁：讀了 p.1–3、p.12、p.14。
- **寫入時做了什麼決定**：Nomad 本身不在配置時決定；它是搬移機制，有兩個設計：
  - **non-exclusive tiering**：頁從慢的層 promote 到快的層後，慢的層保留一份 shadow copy；
  - **transactional page migration**：複製時不 unmap，複製完再檢查這頁有沒有被寫過，寫過就放棄重來〔原文 p.1–2〕。
  - 結果是「page demotion is made less expensive by simply remapping a page if it is not dirty and its shadow copy exists on the capacity tier」〔原文 p.2〕。
- **用什麼資訊做決定？寫完之後還在不在？（N1）**：用 dirty bit。
- **有沒有和延後版、寫穿版、背景版比較？**
  - 記憶體有壓力時，最多比 TPP 快 6 倍〔原文 p.1〕。
  - 動機實驗（16 GB local DRAM＋16 GB CXL）：「**no migration is consistently and substantially better than TPP in progress**」。工作集超過快的層時，TPP 一直到不了穩定狀態，陷入 thrashing。另一方面，初始放置不好、而且熱資料放得進快的層時，搬移就很關鍵〔原文 p.3〕。
  - YCSB（存取大多隨機）：**所有搬移方法都輸給「完全不搬」**〔原文 p.12〕。
  - 討論：「**When the program's working set exceeds the capacity of the fast tier, the most effective strategy is to access pages directly from their initial placement, completely disabling page migration**」〔原文 p.14〕。
- **硬體**：local DRAM 16 GB＋CXL memory 16 GB〔原文 p.3〕。
- **模型架構、模態**：不適用。
- **和本研究的關係**（兩面都有）：
  - **支持 S5 的前提**：工作集大於快的層、搬移成本不低時，**初始（配置時）放置就是最終放置**，之後搬不如不搬。這是本研究條件 2（快的層緊）加上 hold。但 Nomad 的情境是 byte-addressable，可以直接從慢的層讀；KV 的對應是 Cake 從哪一層都能還原〔判讀〕。
  - **威脅（寫穿版）**：在慢的層留一份副本，逐出 clean 頁就免費。**KV 永遠是 clean**，所以寫穿（S1、S2b）的逐出永遠免費。這是 OS 文獻對 08「hold 下寫穿最快」的直接解釋。
  - 對 H3：搬移本身有成本，背景搬移在記憶體有壓力時也可能變成負擔（「TPP in progress」比不搬還差）。
- **證據等級**：〔原文 p.X〕（arXiv v2 的頁碼）；KV 類比〔判讀〕。
