# B_Colloid Tiered Memory Management: Access Latency is the Key!（Colloid）

- **出處**：Midhul Vuppalapati、Rachit Agarwal（Cornell），SOSP '24（Austin，2024-11-04～06），DOI 10.1145/3694715.3695968〔原文 p.1〕。論文標題沒有「Colloid」，Colloid 是系統名。
  連結：https://www.cs.cornell.edu/~ragarwal/pubs/colloid.pdf
  讀了 p.1–2、p.11–12（共 16 頁）。
- **寫入時做了什麼決定**：**不在配置時決定**。它是持續的、搬移式的放置：
  - 原則是「page placement across tiers should be performed so as to balance their average (loaded) access latencies」，而不是把最熱的頁塞滿 default tier〔原文 p.1–2〕；
  - 每個 quantum 有搬移上限，接近平衡點時改用動態上限〔原文 p.12〕。
- **用什麼資訊做決定？寫完之後還在不在？（N1）**：用**執行時**各層的 loaded latency：用 Little's Law，從 CPU 到記憶體路徑上的佇列占用與到達率算出來〔原文 p.2〕。這個資訊**寫入時還不存在**，而且會隨爭用程度改變。
- **有沒有和延後版、寫穿版、背景版比較？**
  - 中等負載下，default tier 的延遲最多膨脹 5 倍，比 alternate tier 慢 2.5 倍；這時現有 tiering 系統比最佳差 2.3 倍〔原文 p.1–2〕。
  - 接到 HeMem、TPP、MEMTIS 上，三者都接近最佳〔原文 p.1–2〕。
  - 收斂時間：熱集合突然改變後，HeMem＋Colloid 比 HeMem 多約 3 秒才收斂；**TPP 要數百秒**，因為 page table 掃描慢、存取資訊不精確〔原文 p.12〕。穩態時，HeMem＋Colloid 的搬移量 <0.7% 的應用吞吐〔原文 p.12〕。
- **硬體**（§2 的動機實驗）：雙 socket 的 Intel Xeon Platinum 8362。default tier 是本地 socket 的記憶體（32 GB，空載延遲 70 ns）；alternate tier 是另一個 socket 的記憶體（96 GB，135 ns），兩者之間是 UPI，每個方向理論 75 GB/s〔原文 p.3〕。§5 的評測硬體**未逐頁讀**。
- **模型架構、模態**：不適用。
- **和本研究的關係**：
  - **概念上是 Cake 的雙胞胎**：最佳點是「兩層一起用，讓兩邊的（帶負載）延遲相等」。Cake 在讀取時、每個請求各自找會合點〔判讀〕。
  - **威脅**：如果最佳的分配取決於執行時的帶負載頻寬（例如多張 GPU 搶同一個 CPU DRAM／PCIe，E4），寫入時就固定下來的分配追不上；讀取時調整（Cake）、持續搬移（Colloid）追得上。S5 的 b 是寫入時用「當時的頻寬」算的，回來時頻寬不同，b 就過時了；S4B 和 Cake 用的是當下的值〔判讀〕。
  - **也指出延後版的弱點**（N3）：負載變得比 tiering 系統收斂得快時，搬移式管理會失效（TPP 要數百秒）。
  - 對 E4：「快的層在負載下不一定快」有文獻根據，值得在 MI300X 上量（見 lit_B_summary §3）。
- **證據等級**：〔原文 p.X〕；和 Cake 的類比〔判讀〕。
