# B_TPP TPP: Transparent Page Placement for CXL-Enabled Tiered-Memory

- **出處**：Hasan Al Maruf、Hao Wang、Abhishek Dhanotia、Johannes Weiner、Niket Agarwal、Pallab Bhattacharya、Chris Petersen、Mosharaf Chowdhury、Shobhit Kanaujia、Prakash Chauhan。ASPLOS '23（Vol. 3），pp.742–755，DOI 10.1145/3582016.3582063。venue 與頁碼取自 NSF PAR 紀錄與 arXiv 的 DOI（網路搜尋）；Colloid 的參考文獻 [35] 也寫 ASPLOS 2023。
  **讀的是 arXiv 2206.02878v2（2023-05-28）**，14 頁：讀了 p.1、p.7–8、p.10–11。
- **寫入時做了什麼決定**：
  1. **配置與回收的水位線分開（背景提早搬）**：local node 的背景回收（demote 到 CXL）要一直做到空頁數達到 demotion_watermark 才停；新配置只要空頁數滿足較低的 allocation_watermark 就可以進行。demotion 水位線一定設得比 allocation 和 low 水位線高，以保留空閒的餘裕〔原文 p.7–8〕。觸發門檻由 `/proc/sys/vm/demote_scale_factor` 控制，預設 2%〔原文 p.8〕。
  2. **依頁的類型在配置時決定放哪（page type-aware allocation）**：file cache、tmpfs 等優先**直接配置在 CXL node**；anon 頁維持原本的配置方式。file cache 之後變熱，才 promote 回 local〔原文 p.8〕。
- **用什麼資訊做決定？寫完之後還在不在？（N1）**：用頁的類型（file 或 anon），配置當下就知道。觀察到的現象是：新配置的頁常是**短命而且熱的**〔原文 p.1、p.8〕；file cache 在暖機期大量產生，之後很少被存取〔原文 p.8〕。類型之後也還在，所以不是 N1。
- **有沒有和延後版、寫穿版、背景版比較？**
  - **Linux 預設**（等於「滿了再搬」）：空頁數低於 low watermark 才開始回收；新的 local 配置要停下來，等回收補到 high watermark。配置速率高時，回收跟不上，頁就落到 CXL node〔原文 p.7〕。
  - **不分開水位線**：配置速率被回收速率卡住，突發的配置全部落到 CXL；promotion 幾乎停擺；困在 CXL 的頁占 55% 的記憶體流量，吞吐掉 12%。分開之後，local 的配置速率提高 1.6 倍〔原文 p.10–11〕。
  - **類型感知配置**：表 2 只列出開啟之後的結果（相對於全部放 local 的基準）：Web1 2:1 時 97% 流量在 local、吞吐 99.5%；Cache1 1:4 時 85%、99.8%；Cache2 1:4 時 72%、98.5%〔原文 p.11〕。原文的表**沒有**「TPP 不開類型感知」的對照。
  - 對其他系統：NUMA Balancing 的回收慢 42 倍，promotion 慢 11 倍，吞吐掉 17.2%〔原文 p.11〕。
- **硬體**：Meta 生產環境，早期支援 CXL 1.1 的 x86 樣品〔原文 p.1〕；local:CXL 容量比 2:1、1:4。
- **模型架構、模態**：不適用。
- **和本研究的關係**：
  - **這是 10 §1.3 要求的「背景提早搬（水位線）」的標準做法**。證據：新寫入是突發的時候，「滿了才回收」會讓新資料落到慢的層，背景保留餘裕就能修好（N3）。**威脅**：在 hold 下，背景水位線可能追平 S5（H3 的停損條件）。
  - **類型感知配置是 OS 裡最接近 S5 的做法**：把「預測會冷」的資料在配置時就直接放到慢的層，理由是避免不必要的搬移、把快的層留給熱資料（N2）。但它用的是類型標籤，不是位置；也沒有和「同一條規則延後套用」比較〔判讀〕。
  - Linux 已經有同類旋鈕：`watermark_scale_factor` 控制 kswapd 多積極，配置突發時可以調高（見 lit_B_summary §3 的 E8）。
- **證據等級**：〔原文 p.X〕（arXiv v2 的頁碼）；和 H3 的對應〔判讀〕。
