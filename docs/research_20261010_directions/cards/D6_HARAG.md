# D6_HARAG HA-RAG: Hotness-Aware RAG Acceleration via Mixed Precision and Data Placement

- 出處：Danying Ge, Jianhua Gao, Yixue Yang, Weixing Ji，HPCA 2026（p.1 頁首）。https://arxiv.org/abs/2510.20878 。查「引用 Cake 的論文」時找到的。PDF 在 `/mlsteam/data/tiara/papers_d6/arXiv2510.20878.pdf`。讀了 p.1–2、p.5、p.7（grep 定位）。

- 精度決定：
  - **做什麼**：RAG 知識庫預先算好的 KV chunk 依存取頻率排序分四組，用四種 8-bit 格式（INT8、FP8 E4M3、FP8 E5M2、GSE-8）。最熱的 chunk 用誤差最小的格式，最冷的用解壓最快的格式〔原文 p.5〕。
  - **何時**：離線預先計算 KV chunk 時壓縮，載入時解壓〔原文 p.1、p.5〕。
  - **粒度**：KV chunk。
  - **用什麼資訊**：存取頻率（hotness）〔原文 p.5〕。

- 有沒有和重算／載入並行結合？精度跟位置有關嗎？
  - 我讀的段落裡沒有重算與載入並行；baseline 是 TurboRAG（預先算 KV、按需載入）〔原文 p.1、p.6〕。
  - 精度跟 token 位置無關，看熱度。

- 反量化／解碼成本：Table II 列出解壓時間：FP8 E4M3 284.86 s、FP8 E5M2 254.72 s、INT8 20.23 s、GSE-8 17.38 s，壓縮比都約 2.0〔原文 p.7〕。原文說是在「大小 1024 的資料集」上量的，但單位和每個樣本的大小在我讀的段落裡不清楚，硬體也沒找到（**未查證**）。至少可以看出 FP8 的解壓比 INT8 慢一個數量級。

- 品質與位置的關係：只比了各格式的 RMSE：INT8 最小，E4M3 次之，GSE-8 最大〔原文 p.7〕。

- 有沒有和延後版比？沒有。

- 和 D6 的關係〔判讀〕：
  - 「依 chunk 的某種屬性分配不同精度、而且把**解壓成本**當成選格式的依據」已經有人做了（依熱度）。D6 若改依位置，換的只是分配依據。
  - 「FP8 解壓可能比 INT8 慢很多」這點值得在 Tiara 平台上自己量（**NOT_MEASURED**），因為 CLAUDE.md 記錄 3090 沒有原生 FP8 運算。

- 證據等級：B（Table II 的單位存疑）。
