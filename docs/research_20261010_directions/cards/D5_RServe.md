# D5_RServe RServe: Overlapping Encoding and Prefill for Efficient LMM Inference

- 出處：Tianyu Guo, Tianming Xu, Xianjie Chen, Junru Chen, Nong Xiao, Xianwei Zhang（中山大學、小紅書）。arXiv 2509.24381v1（2025-09-29）。venue：未查證（arXiv 無 comment）。<https://arxiv.org/abs/2509.24381>。讀了 p.1–5、p.7–8（PDF 實體頁）。
- 寫入時做了什麼決定：無。它是**讀取（服務）時的排程**。
  - 有一個 embedding tracker，用 request ID 當 key、存該請求的 embedding，只是排程用的暫存〔原文 p.7〕。
- 用什麼資訊做決定？寫完之後還在不在？（N1）：不適用。
- 有沒有和延後版、寫穿版、背景版比較？沒有（不是寫入時的設計）。
  - 核心做法：encoder 由左到右細粒度地編碼，**部分 embedding 一出來，LLM 就開始 prefill**（intra-request pipeline），再加上跨請求的 micro-batch 平衡〔原文 p.2、p.5〕。
  - 對手是 vLLM、DistServe 式的 encoder 分離系統；延遲最多降 66%、吞吐最多 +109%〔原文 p.1〕。
- 硬體、各層頻寬、模型、模態：
  - 8×H100 80GB（NVLink）；另測 4×A100 40GB（PCIe）〔原文 p.8〕。Qwen2.5-VL 7B／32B／72B；MMMU＋SGLang benchmark；1K、2K 解析度平均輸入 8K、12K token〔原文 p.8〕。
  - **encoder 佔比（Q3 用）**：單一請求、兩張圖、4×H100，encoder 佔 TTFT 的 11%–26%，隨解析度（1K→8K）上升〔原文 p.2 圖 2〕。圖說沒寫是哪個模型〔判讀：主實驗用 Qwen2.5-VL，但圖 2 未標〕。
- 和 D5(a) 的關係：
  - **威脅「存 embedding」的價值**：如果還原時從像素重算，encoder 可以和 LLM prefill 管線化，encoder 的時間有一部分能被藏起來〔原文 p.2〕。所以 D5(a) 的 ΔT（要不要付 encoder）應該拿「管線化之後」的 encoder 成本來算，不是 encoder＋prefill 直接相加〔判讀〕。
  - **不威脅分層**：沒有 KV／embedding 的儲存與放置，沒有 load。
  - 它的管線形狀（前面的 token 先算、後面的還在產生）和 Cake 的「前算後讀」不同：這裡兩條都是計算〔判讀〕。
- 證據等級：〔原文 p.X〕；圖 2 的模型為〔判讀〕。
