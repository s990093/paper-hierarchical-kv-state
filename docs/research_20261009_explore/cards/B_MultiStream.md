# B_MultiStream The Multi-streamed Solid-State Drive

- **出處**：Jeong-Uk Kang、Jeeseok Hyun、Hyunjoo Maeng、Sangyeun Cho（Samsung Memory Solutions Lab），HotStorage '14。
  連結：https://www.usenix.org/system/files/conference/hotstorage14/hotstorage14-paper-kang.pdf
  PDF 本身沒有寫 venue；HotStorage '14 是從 USENIX 的 URL 路徑判斷的〔判讀〕。5 頁，讀了 p.1–2。
- **寫入時做了什麼決定**：host 在**寫入時**替每筆寫入標上 stream ID，依「預期壽命」分類。SSD 保證同一個 stream 的資料寫在一起（同一個 erase unit），而且和其他 stream 分開。理想上，GC 時就不必搬有效資料〔原文 p.1〕。
- **用什麼資訊做決定？寫完之後還在不在？（N1）**：用的是應用程式在寫入當下就知道的資料壽命。
  - 裝置自己從存取歷史去推 hot／cold 很難：需要可觀的資源記錄歷史；存取模式一變（例如 log-structured FS），準確度就下降〔原文 p.2〕。
  - 原文：「robustly deriving accurate information about data hotness and future access patterns is hard」〔原文 p.2〕。
  - **這是 N1 的「資訊不對稱」形式**：寫入者知道，之後的觀察者（SSD）只能猜。
- **有沒有和延後版、寫穿版、背景版比較？**
  - 「之後再修正」就是 GC：SSD 越滿、越碎，GC 越頻繁，每次 erase 前要複製越多有效頁〔原文 p.1〕。
  - Cassandra 的最差更新吞吐，在 Normal SSD 上隨資料持續更新而下降約 56%。multi-stream 把最差吞吐改善將近 56%〔原文 p.1–2〕。
  - TRIM 有幫助，但追不上 multi-stream 的效果〔原文 p.2〕。
  - 後續評論：FairyWREN（OSDI '24）指出，stream 能把壽命相近的資料放在一起，但應用程式不能直接控制 GC，所以裝置層的寫入放大仍然明顯〔FairyWREN 原文 p.6〕。
- **硬體**：真實的 multi-stream SSD 原型（Samsung）。
- **模型架構、模態**：不適用。
- **和本研究的關係**：
  - **裝置層的 N2**：放錯之後再修正，要付額外寫入（GC 複製）。只有 SSD 寫入額度或 GC 停頓是瓶頸時才重要（E6）。
  - KV 的可能用法〔判讀〕：同一個 session 的 KV 會一起被淘汰（壽命相同），寫入時可以依 session 或「預期回來時間」分 stream，減少 GC 複製。這屬於 H1（寫入量）的延伸，**不會**縮短 TTFT，除非 GC 停頓正好落在讀取路徑上。
  - 實務限制：NVMe 的 streams／FDP 在 MI300X 機器的 SSD 上有沒有支援，**未查證**。
- **證據等級**：〔原文 p.X〕；KV 用法〔判讀〕。
