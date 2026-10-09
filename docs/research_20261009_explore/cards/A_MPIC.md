# A_MPIC MPIC: Position-Independent Multimodal Context Caching System for Efficient MLLM Serving

- **出處**：Shiju Zhao, Junhao Hu, Rongxiao Huang, Jiaqi Zheng, Guihai Chen（Nanjing University、Peking University）。arXiv 2502.01960v2（v1 2025-02-04）；comment「17 pages, 13 figures, the second version」。venue：未查證（PDF 沒有寫會議）。<https://arxiv.org/abs/2502.01960>。讀了 p.1–3、p.5–7（PDF 實體頁）。和 EPIC（ICML'25）同一作者群（Junhao Hu）。
- **寫入時做了什麼決定**：**什麼時候算、存到哪、存什麼格式**。
  - 使用者**上傳檔案時**（還沒提問），MPIC 就算出該圖片／檔案的 KV，放 GPU 記憶體服務，**同時複製到磁碟**，到期後刪除〔原文 p.1 摘要、p.6〕。
  - 存的是**和位置無關**（position-independent）的 KV，讓它之後能被接到任何位置的 prompt 裡〔原文 p.1、p.5〕。
- **用什麼資訊做決定？（N1）**：「使用者上傳了這個檔案」這個事件——它預告了之後會被問〔判讀〕；事後仍在，不是 N1。
- **有沒有和延後版、寫穿版、背景版比較？**
  - 讀取時：查到的圖片 KV 有的在 GPU、有的在 CPU 或磁碟，缺的（過期被刪）就重算；**計算與載入並行**；從磁碟載入失敗就退回全部重算〔原文 p.6 §3.2〕。〔判讀〕這是 Cake 式「算與載並行」在多模態上的版本，但以圖片為單位。
  - 重用時用 selective attention 只重算一小部分 token 來減少準確度損失（類似 CacheBlend）〔原文 p.3、p.6〕。
  - 對照是 prefix caching 與全部重算等；回應時間最多少 54%、吞吐 2 倍〔原文 p.1〕。**沒有「上傳時不算、第一次提問時才算」的延後版**〔判讀〕。
- **硬體**：讀過的頁沒看到 GPU 型號（未查證）。
- **模型架構、模態**：LLaVA-1.6-vicuna-7B、LLaVA-1.6-mistral-7B（圖文）〔原文 p.7〕。**有損**（位置無關的 KV 拼接，靠部分重算補）〔原文 p.1〕。
- **和本研究的關係**
  - **H5（VLM）**：「上傳時就把圖片 KV 算好並寫到磁碟」是現成的寫入時做法〔原文 p.6〕；讀取時並行算與載〔原文 p.6〕。H5 若要做，對照組至少要有 MPIC 這種「上傳時全寫、讀取時並行」〔判讀〕。
  - **N4**：位置無關的格式是寫入時選的；選了才能在讀取時接到任意位置〔原文 p.1、p.5〕。CachedAttention 的「存 RoPE 之前的 KV」也是同一類（見 [A_CachedAttention_補充](A_CachedAttention_補充.md)）〔判讀〕。
  - Lit-C 負責模態；這張卡只記和 KV 系統寫入有關的部分。
- **證據等級**：〔原文 p.X〕；判讀如標示。
