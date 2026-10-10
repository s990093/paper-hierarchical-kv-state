# D5_DroidSpeak DroidSpeak: KV Cache Sharing for Cross-LLM Communication and Multi-LLM Serving（只補 H11 相關段落）

> DroidSpeak 已在專案 `docs/research_20260927/verified_findings.md`（方向 11、2-1，記為 NSDI'26）出現過，那裡記的是**品質**。這張卡只補「**從遠端節點抓 KV＋部分重算＋兩者管線化**」這一面，和 H11 直接相關。

- **出處**：Yuhan Liu, Yuyang Huang, Jiayi Yao, Shaoting Feng, Zhuohan Gu, Kuntai Du, Hanchen Li, Yihua Cheng 等（UChicago 等）。arXiv 2411.02820。venue 沿用專案已記的 NSDI'26，**這次沒有重查**〔未查證〕。讀了 PDF 實體頁 p.2、p.8、p.12。
- **場景**：同一底座、不同 fine-tune 的兩個模型共用 KV。重算的理由是**品質**（某些「關鍵層」直接沿用別的模型的 KV 會掉分），不是省頻寬。
- **寫入時做了什麼決定**：離線 profiling 決定哪些層要重算〔原文 p.2〕；store 時把 KV 或 E cache（論文用語；〔判讀〕應是 transition 層的輸入 hidden state）按層存進 GPU 記憶體裡的 key-value store〔原文 p.8〕。
- **有沒有「接收端重算一部分、只傳另一部分」**：**有，沿著「層」切**，而且**跨網路**。
  - 重算關鍵層、其餘層沿用別的模型的 KV；KV 和 E cache 都用 torch.distributed **從遠端 GPU 節點抓**〔原文 p.8〕。
  - 「smart KV cache loading」：把 KV 的載入和關鍵層的重算管線化，盡量藏住從遠端節點載入的延遲〔原文 p.2〕。範例：先傳 transition 層的 E cache，就能開始重算第 4–10 層，同時並行傳第 1–3 層的 KV；總 TTFT 從 30 降到 17（示意單位）〔原文 p.8〕。
  - 傳輸放在獨立的 CUDA stream，和重算重疊〔原文 p.8〕。
- **報告的網路頻寬**：Fig. 20 比較不同頻寬下「管線化」與「先傳再算」，頻寬很高時絕對改善變小〔原文 p.12〕。（確切頻寬範圍在圖上，我沒讀出數值〔未查證〕。）
- **關鍵的一句（Limitation）**：「§4.3 只依**系統負載**調整重算比例。未來工作可以讓調整演算法考慮**網路頻寬的變化**，例如頻寬有限時擴大要重算的關鍵層範圍」〔原文 p.12〕。
- **後續**：Semantic Cache Distillation（arXiv 2606.07684，PDF 頁尾印 ICML 2026、PMLR 306）在「producer 裝置 → consumer 裝置」的分離式設定下，頻寬掃 100 Gbps–1 Tbps，拿「DroidSpeak 式選擇性重算」（多數層用傳來的 KV、少數關鍵層在 consumer 重算）當基線〔SCD 原文 p.7〕。它的問題設定也寫明 consumer「要嘛本地重算（受算力限制），要嘛從 producer 抓（受頻寬限制）」〔SCD 原文 p.3〕。但它同樣是**異質模型**的情境。
- **和 D5(d) 的關係**：
  - 「跨網路、接收端重算一部分、同時載入其餘、兩者管線化」的**機制**，在 DroidSpeak 已經有了（沿層切、跨模型）。H11 不能主張「第一個在網路上同時做部分重算與部分傳輸」。〔判讀〕
  - DroidSpeak 自己把「依網路頻寬調整重算比例」列為未來工作〔原文 p.12〕——表示**「依頻寬決定切點」在同模型 P/D 情境下仍是空白**（至少在 DroidSpeak 這條線上）。〔判讀〕
  - 同模型的 P/D 裡，沒有品質理由要重算；重算唯一的理由就是省頻寬，這正是 H11 的論點。〔判讀〕
- **證據等級**：〔原文〕；venue〔未查證〕（沿用專案紀錄）。
