# D5_Vista Vista: Scene-Aware Optimization for Streaming Video Question Answering under Post-Hoc Queries

- 出處：Haocheng Lu, Nan Zhang, Wei Tao, Xiaoyang Qu, Guokuan Li, Jiguang Wan, Jianzong Wang（華中科大、平安科技）。arXiv 2602.08448v1（2026-02-09）。**AAAI 2026**：PDF 第 1 頁有 AAAI 2026 版權頁腳〔原文 p.1〕，arXiv comment 寫「Accepted to AAAI 2026 (Main Technical Track)」；ShallowStream 的參考文獻給了 AAAI 卷 40 頁碼 7539–7547（沒有打開 AAAI 網站）。<https://arxiv.org/abs/2602.08448>。讀了 p.1–6（全文 9 頁）。
- 寫入時做了什麼決定：**存什麼格式、放哪裡**。
  - 影格依場景切段。一段結束時：做一個壓縮的場景 token（pooled embedding）留在 **GPU** 當索引；**原始高解析度的影格特徵**卸載到 **CPU 記憶體**（或 disk）〔原文 p.1 摘要、p.3、p.4〕。
  - 存的是影格／encoder 特徵，**不是 KV**〔原文 p.3「original high-resolution frame features are offloaded to CPU memory」〕。原文有時寫「frames」、有時寫「frame features」，到底是像素還是 encoder 輸出沒有講清楚〔判讀〕。
- 用什麼資訊做決定？寫完之後還在不在？（N1）：用的是場景邊界（相鄰影格相似度）〔原文 p.3–4〕。事後仍可算，不是 N1〔判讀〕。
- 有沒有和延後版、寫穿版、背景版比較？
  - 沒有。
  - 問題來時：用場景 token 打分、選 top-k 場景，把它們的完整影格從 CPU／disk 拿回來，和最近的影格一起當成**新的輸入**送進 VLM（等於從特徵重算 LLM prefill）〔原文 p.4–5〕。
  - 效率只和「全部影格輸入」「均勻取樣」比 TTFT 和記憶體〔原文 p.6 圖 3〕；**沒有**和 ReKV 這種「讀 KV」的做法比延遲（ReKV 只出現在相關研究）〔原文 p.2〕。
- 硬體、各層頻寬、模型、模態：4×RTX 4090D 24GB、i9-14900K、125GB RAM〔原文 p.5〕；LLaVA-OneVision-7B 等〔原文 p.5〕；StreamingBench 等；串流影片。頻寬沒寫〔未查證〕。
- 和 D5(a) 的關係：
  - **部分威脅**：「影片的歷史不存 KV、改存影格／特徵在 CPU（或 disk），提問時重算」已經有人做，而且是 AAAI 2026〔原文 p.1、p.4〕。
  - **沒做的**：沒有選格式（固定存特徵）、沒有 KV 這個選項、沒有並行還原、沒有量讀取 vs 重算的成本比。
  - 加上 [D5_ShallowStream](D5_ShallowStream.md)：串流影片社群已經偏向「存 embedding／特徵、讀時重算」，主要理由是 GPU 記憶體與串流時計算，不是 κ〔判讀〕。
- 證據等級：〔原文 p.X〕；「像素還是特徵」為〔判讀〕。
