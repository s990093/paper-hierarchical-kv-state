# B_Tachyon Tachyon: Reliable, Memory Speed Storage for Cluster Computing Frameworks

- **出處**：Haoyuan Li、Ali Ghodsi、Matei Zaharia、Scott Shenker、Ion Stoica，SoCC '14（Seattle，2014-11-03～05），DOI 10.1145/2670979.2670985〔原文 p.1〕。
  連結：https://www.cs.berkeley.edu/~haoyuan/papers/2014_SOCC_tachyon.pdf
  讀了 p.1–2、p.5–6（共 15 頁）。
- **寫入時做了什麼決定**：
  - 寫入只寫記憶體，不做同步複製；遺失的資料靠 lineage 重新執行算回來〔原文 p.1〕。
  - **大小超過門檻的 dataset，寫入時就同步寫到 disk，不放記憶體**〔原文 p.6〕。這是依大小的 write-around。
  - 其他資料在**背景、低優先**做 checkpoint，用 Edge 演算法挑：先 checkpoint lineage DAG 的葉子；存取次數 >2 的熱檔案優先；盡量不 checkpoint 暫存檔〔原文 p.5–6〕。
  - 記憶體被還沒 checkpoint 的檔案塞滿時，就**同步** checkpoint〔原文 p.6〕。
- **用什麼資訊做決定？寫完之後還在不在？（N1）**：用 lineage、檔案大小、存取次數。lineage 之後一直都在，這正是可以延後的原因。
- **有沒有和延後版、寫穿版、背景版比較？**
  - 對 in-memory HDFS（同步複製，等於寫穿）：寫入吞吐高 110 倍；真實工作流程的端到端延遲快 4 倍〔原文 p.1〕。
  - 核心論點：「lineage enables us to asynchronously checkpoint in the background, without stalling writes」。沒有 lineage 的系統（例如 key-value store）只能同步 checkpoint〔原文 p.5〕。
  - **背景版能成立的條件**：「most data has time to be checkpointed due to the bursty behavior of frameworks」，所以逐出還沒 checkpoint 的檔案很少發生〔原文 p.6〕。反過來，「any fixed checkpoint interval can lead to unbounded recovery times if data is written faster than the available disk bandwidth」〔原文 p.2〕。
  - Facebook 有超過 70% 的資料在一天內刪除；Edge 先 checkpoint 葉子，讓暫存資料有機會在 checkpoint 之前就被刪掉〔原文 p.5–6〕。
- **硬體**：叢集（細節在評測章，未讀）。
- **模型架構、模態**：不適用（資料處理的 job pipeline）。
- **和本研究的關係**：
  - **最接近 KV「丟了可以重算」的經典系統**。結論：**能重算，就能把持久化從同步（寫穿）變成可延後的背景工作**。這是對「寫入時決定」的**威脅**，背景版因此更強。
  - **但它自己寫出了背景版失效的條件**：要有突發之間的空閒（N3），而且長期寫入速率要低於 disk 頻寬；否則要退回同步寫（等於 hold）。對應 H3。
  - **依大小在寫入時 write-around**：最大的 dataset 直接寫 disk。KV 的類比〔判讀〕：doc 型負載一次寫一大段，寫入時就決定前段不進 CPU，和 09 的「只有 doc 型有機會」一致。
  - 「先 checkpoint 葉子、讓暫存資料自然死掉」：KV 的類比是 session 結束得早的資料，延後寫就能省下寫入（H1、H8）〔判讀〕。
- **證據等級**：〔原文 p.X〕；KV 類比〔判讀〕。
