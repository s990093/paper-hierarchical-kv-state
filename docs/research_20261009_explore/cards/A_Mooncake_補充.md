# A_Mooncake_補充 Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving（補充卡）

> Mooncake 已在 `03_paper_map.md` A5（E04）與 `02_sota_write_techniques.md`。這張卡**不重做**，只補和 **H11（P/D 分離：prefill 節點決定傳哪些、讓對方重算哪些）** 直接相關的段落。

- **出處**：Ruoyu Qin, Zheming Li, Weiran He, Mingxing Zhang, Yongwei Wu, Weimin Zheng, Xinran Xu（Moonshot AI、Tsinghua）。**FAST 2025**，會議版標題是「Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot」，pp.155–170（依本機 Bidaw FAST'26 PDF 與 Strata OSDI'26 PDF 的參考文獻；**本機的是 arXiv 版 2407.00079，標題不同**，用 arXiv abs 頁核對過 v3 標題）。這次讀了 arXiv 版 p.5、p.10–11（PDF 實體頁）。

## 補充 1：prefill 節點的「寫入」與傳輸，全部做、不挑
- 工作流程：(1) 從遠端 CPU 記憶體載入可重用的 prefix cache；(2) incremental prefill：算完後把**新產生的增量 KVCache 存回 CPU 記憶體**；(3) KVCache transfer：與 (2) 重疊、**逐層串流**到目的 decode 節點的 CPU 記憶體〔原文 p.5〕。
- 〔判讀〕prefill 節點對「新產生的 KV」沒有任何選擇：全部存回、全部傳給 decode。10 §3C 說「這次傳輸本身就是一次寫入」——在 Mooncake 裡它確實是，但沒有 H11 想做的「挑哪些傳、哪些讓對方重算」。

## 補充 2：「算還是載」的決定在讀取時（排程時）做
- 請求不一定被送到 prefix 最長的 prefill 節點；若額外的 prefill 時間比傳輸時間短，Conductor 把快取位置轉給另一個節點，由它主動抓來並存一份；「若最佳的遠端 prefix 長度不超過本地可重用 prefix 乘以一個門檻，**我們偏好直接計算**」——門檻目前手調〔原文 p.11〕。這兩個策略順便達成熱點快取的自動複製〔原文 p.11〕。
- 原文說預測未來使用量來排程「在 MaaS 的動態負載下不可能準確」，所以用啟發式〔原文 p.11〕。
- 〔判讀〕Mooncake 的 compute-or-load 是**讀取時**、以「整段 prefix 長度」為單位的二選一；不是 Cake 式的同時算與載，也不是寫入時決定。

## 和本研究的關係
- **H11**：在這次的有限搜尋裡，**沒有找到**「prefill 節點在寫入／傳輸時就決定哪些 chunk 不傳、讓 decode 端重算」的論文（WebSearch 一次、arXiv 關鍵字搜尋兩次，見 summary §5）。最接近的是：
  - Mooncake 的讀取時 compute-or-fetch〔原文 p.11〕；
  - KVServe（本機 PDF，SIGCOMM'26，`docs/research_20260924/workloads_eval.md` 已列）與 Semantic Cache Distillation（arXiv 2606.07684，ICML'26，只讀摘要）——都是 P/D 傳輸時**有損壓縮**，後者摘要說它在頻寬受限時支配「量化與選擇性重算」基線；
  - OasisKV（arXiv 2608.08097，只看 web 搜尋摘要）——P/D 傳輸時只傳第一步 decode 需要的部分，其餘之後網路預取（不是重算）。
  - 〔判讀〕H11 的空白可能還在，但「查不到」不等於沒有（CLAUDE.md §1-7）。H11 需要兩個節點，本機只能模擬（10 §4）。
- **證據等級**：〔原文 p.X〕；搜尋範圍與空白判斷〔判讀〕。
