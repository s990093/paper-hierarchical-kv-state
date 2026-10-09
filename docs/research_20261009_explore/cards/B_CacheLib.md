# B_CacheLib The CacheLib Caching Engine: Design and Experiences at Scale

- **出處**：Benjamin Berg、Daniel S. Berger、Sara McAllister、Isaac Grosof、Sathya Gunasekar、Jimmy Lu、Michael Uhlar、Jim Carrig、Nathan Beckmann、Mor Harchol-Balter、Gregory R. Ganger，OSDI '20（2020-11-04～06），印刷 pp.769–786。
  連結：https://www.usenix.org/system/files/osdi20-berg.pdf
  頁碼規則：p.X＝PDF 第 X 頁，印刷頁＝p.X＋767（PDF p.1 是封面）。讀了 p.8、p.12、p.15。
- **寫入時做了什麼決定**：
  - 新物件一律在 DRAM 配置。DRAM 逐出時，才決定「放進 flash」或「丟掉」〔原文 p.8〕。所以 flash 是在**逐出時**才填，等於延後版。
  - 物件已經在 flash 上、在 DRAM 期間又沒被改過，就**不再寫回 flash**〔原文 p.8〕。
  - admission 策略〔原文 p.8、p.12、p.15〕：
    - 預設：固定機率 p；
    - reject-first：被 DRAM 逐出的前 n 次都拒絕；
    - 進階 ML 版（類 Flashield，但特徵取自 DRAM 生命期之外：用 Bloom filter 記錄過去 6 小時的存取）。
- **用什麼資訊做決定？寫完之後還在不在？（N1）**：Flashield 的特徵是「在 DRAM 期間被命中幾次」。但 Facebook 的 DRAM 生命期太短：L2 Lookaside 裡，被考慮放進 flash 的物件只有 **14%** 在 DRAM 期間被讀或寫過〔原文 p.15〕。所以改用「在 DRAM 之外收集的歷史」。
- **有沒有和延後版、寫穿版、背景版比較？**
  - DRAM 逐出的物件如果全部放進 flash，寫入率會比「達到目標壽命所允許的寫入率」高 50%〔原文 p.15〕。
  - 進階 admission 對預設的機率式 admission：SocialGraph 上寫入 flash 的 bytes **少 44%**，命中率沒有下降〔原文 p.12、p.15〕。
  - flash 一般超額配置 50%；裝置層寫入放大 1.1×（Lookaside）到 1.4×（Storage）〔原文 p.12〕。
- **硬體**：Facebook 生產環境的 hybrid cache（DRAM＋flash）。
- **模型架構、模態**：不適用。
- **和本研究的關係**：
  - **「延後觀察」在快的層停留時間短時會失效**。要改用 admission 當下就拿得到的歷史資訊。KV 的條件 2（CPU 空間緊）就是這種情形，所以頻率感知對照組（10 §1.3）在緊的 CPU 上可能看不到足夠的讀取〔判讀〕。這間接支持 H4／H8：用寫入當下就有的 metadata（session 類型、共享前綴）。
  - **威脅（寫穿版）**：「已經在 flash、又沒改過就不重寫」就是 immutable 資料的寫穿好處。KV 永遠不會被改，所以一旦下層有副本，從 CPU 逐出就是免費的。這和 08「hold 時寫穿最快」一致。
  - **支持 H1**：寫入量本身是一級指標（壽命），admission 能少寫 44% 而命中率不變。
- **證據等級**：〔原文 p.X〕；和 H 的對應〔判讀〕。
