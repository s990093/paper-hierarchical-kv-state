# D5_DASC DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving

- **出處**：Yanqi Yu, Pingwei Sun, Jianchao Tan, Tao Zhang, Yuchen Xie, Xunliang Cai, Yao Liu（華東師大、美團、華南理工）。arXiv 2608.30386v1，2026-08-31，cs.LG。venue：未查證。<https://arxiv.org/abs/2608.30386>。讀了 p.1–3、p.5–6、p.14（PDF 實體頁；共 18 頁，其餘沒讀）。
- **寫入時做了什麼決定**：**狀態檢查點要存哪些「單元」**（GDN 的 head、KDA 的 (head, k) channel）。離線從模型權重算出每個單元的「保留時距」（retention horizon），存檢查點時只留長時距的單元，打包成不規則（ragged）的檢查點格式；讀取時被省略的單元補零，或從一段有限長的 suffix 重算〔原文 p.1 摘要、p.2 圖 1〕。
  - 動機：前綴快取要在固定間隔存完整的狀態檢查點；密了吃光 HBM，疏了要多重算〔原文 p.1〕。
  - 它不改「存在哪些位置」，也不改全注意力層的 KV〔原文 p.3 相關研究段〕。
- **用什麼資訊做決定？寫完後還在不在？（N1）**：決定依據是模型權重（與輸入無關，離線算一次）〔原文 p.2 圖 1〕，寫完後還在，不是 N1。被丟掉的單元在寫完之後就拿不回來，只能補零或重算 suffix〔原文 p.1〕——這部分是 N1 的形狀〔判讀〕。
- **有沒有和延後版、寫穿版、背景版比較？**：沒有。比的是同一個 SGLang 快取路徑下的 dense 與 DASC 各種壓縮設定，在**相同的 HBM 檢查點預算**下比較〔原文 p.14 C.1〕。結果：Kimi-Linear 上保守設定把 KDA 狀態檢查點壓 2.63 倍，品質接近全存；相同預算下平均 TTFT −42.6%、輸入吞吐 +68.4%〔原文 p.1〕。
- **硬體、各層頻寬、模型**：Hopper GPU〔原文 p.6〕；SGLang、TP8、BF16 權重；狀態用 SGLang 預設的混合精度（temporal FP32、conv BF16）〔原文 p.14〕。每 rank 的檢查點預算：Kimi-KDA 694,681,600 bytes、Qwen-GDN 1,236,271,104 bytes，各等於 128 個 dense slot〔原文 p.14〕。**只有 HBM 一層**；相關研究段提到跨 GPU 與 host 的階層式系統（原文引 Gao et al., 2024；Jin et al., 2024）是「互補的」，沒有自己做 host／SSD〔原文 p.3〕。模型：Kimi-Linear（KDA）、Qwen（GDN）〔原文 p.6〕。
- **品質**：**有損**（補零或近似重算）〔原文 p.1〕。
- **和 D5(b) 的關係**
  - 它是「寫入時決定存什麼格式」（N4 類），不是「存在哪一層」。和 SuffixReplay（存 hidden-state anchor）、SGLang 的 int8 檢查點池（見 D5_SGLangHiCacheMamba）同一類：都在縮小每份檢查點的大小〔判讀〕。
  - 對 H7 的影響：檢查點越小，「全部寫到 CPU」越不稀缺，延後版／寫穿版越便宜——這會削弱 H7「寫入頻寬是稀缺資源」的論點（如果接受有損）〔判讀〕。無損是硬條件時，DASC 不適用。
  - **沒有多層放置**：查證過的範圍（p.1–3、5–6、14）內沒有 host／SSD 的實驗〔原文 p.14 只有 matched-HBM 協定〕。
- **證據等級**：〔原文 p.X〕；對 H7 的影響〔判讀〕。p.7–13、15–18 沒讀。
