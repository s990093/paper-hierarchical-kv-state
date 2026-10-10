# D5_EPD Efficiently Serving Large Multimodal Models Using EPD Disaggregation

- 出處：Gursimran Singh, Xinglu Wang, Yifan Hu 等（Huawei Technologies Canada、SFU、Huawei Cloud）。arXiv 2501.05460v4（2025-06-28）。**ICML 2025**：PDF 第 1 頁註腳「Proceedings of the 42nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025」〔原文 p.1〕。程式碼 <https://github.com/vbdi/epdserve>（沒有打開）。<https://arxiv.org/abs/2501.05460>。讀了 p.1–5、p.8 的一部分（全文 17 頁）。
- 寫入時做了什麼決定：無（和儲存放置無關）。
  - encoder worker 算完的多模態 token 放在 MM cache，傳到 prefill worker 的 MM cache 之後就清掉〔原文 p.4〕。這個 cache 是**傳輸用的暫存**，不是跨請求重用。
- 用什麼資訊做決定？寫完之後還在不在？（N1）：不適用。
- 有沒有和延後版、寫穿版、背景版比較？不適用。對手是 vLLM、DistServe（不分離 encoder）〔原文 p.6–7〕。
- 硬體、各層頻寬、模型、模態：三個 LMM（MiniCPM-V 2.6、InternVL2 等，詳見附錄 E.2）〔原文 p.2、p.5〕；圖片、影片；另有 NPU 實驗〔原文 p.9〕。讀過的頁沒有每 token 的 encoder vs prefill 時間〔判讀：沒找到〕。
  - 結果：峰值記憶體最多低 15×、batch 最多大 22×、TTFT 最多低 71%〔原文 p.1〕。
- 和 D5(a) 的關係：
  - 背景：「encoder 是獨立的一個階段，可以拆出去」的代表作。它說明 encoder 會增加 token 數與計算〔原文 p.2〕，但沒量化成 r_v。
  - 不威脅：沒有 embedding 的長期保存、沒有分層、沒有還原。
- 證據等級：〔原文 p.X〕；只讀了部分頁。
