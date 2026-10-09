# A_HBF_LRUK Enabling High-Bandwidth Flash for Generative Recommendation Serving with Write-Aware KV Cache Policy

- **出處**：Danni Peng, Kai Wu, Tianyu Zuo, Pengfei Xia, Hui Zang（Huawei Technologies）。arXiv 2609.07175v1（2026-09-07），cs.AR，5 頁；PDF 註明「已投 IEEE」〔原文 p.1〕。venue：未查證。<https://arxiv.org/abs/2609.07175>。讀了 p.1–2（PDF 實體頁）。
- **寫入時做了什麼決定**：**存不存**（寫入准入）。用 admission-controlled LRU-K：一個使用者的 KV 要等到第 K 次請求才准入快取；傳統 LRU 是「每次 miss 就寫入整份 KV、插到 MRU」〔原文 p.2〕。原文的說法是「把 KV 寫入和 cache miss 解耦」〔原文 p.1 摘要、p.2〕。
- **用什麼資訊做決定？（N1）**：使用者的請求次數（存取歷史），事後仍在，不是 N1。
- **有沒有和延後版、寫穿版、背景版比較？**
  - 比的是 LRU（寫入跟著 miss）vs LRU-K（准入過濾）。10K 使用者、HSTU-10B 的負載下，7 個 HBF stack 在 LRU 下只撐約 1 年；LRU-10 延長到 6 年以上，吞吐相當或略好〔原文 p.1 摘要、p.2〕。HBF 系統比只用 HBM 的吞吐高 3.8–4.7 倍〔原文 p.1〕。
  - **全部是分析模型**（roofline 式的延遲模型加上寫入流量與壽命模型），不是實機〔原文 p.2 Eq. 1–2〕。
- **硬體（模型參數）**：HBF 每 stack 512 GB、約 1.6 TB/s 讀頻寬；SLC 約 100K P/E；1 個 HBM4 stack 放常更新的 activation，7 個 HBF stack 放權重與使用者 KV〔原文 p.1–2〕。
- **模型架構、模態**：生成式推薦（HSTU、OpenOneRec），不是 LLM 對話；使用者層級的 KV 重用〔原文 p.1–2〕。
- **和本研究的關係**
  - **支持 N2／H1**：當慢層的稀缺資源是**寫入額度**時，「不寫」或「晚點寫」可以換來數倍壽命〔原文 p.1〕。和 [A_Lachesis](A_Lachesis.md)、Dynamo KVBM 的磁碟過濾（文件明說是為了延長 SSD 壽命，見 summary §4）是同一個論點。
  - **但它是頻率型准入，不看位置**，也就是 10 §1.3 說的「頻率感知」對照組；H1 若要主張 S5 少寫，必須贏過這種准入〔判讀〕。
- **證據等級**：〔原文 p.X〕（分析模型）；意義〔判讀〕。
