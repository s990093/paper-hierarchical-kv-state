# D5_PACE PACE: A Unified Condense-and-Extract Paradigm for Fast VLM Inference

- 出處：Junjie Liu, Shengyuan Ye, Xu Chen（中山大學等）。arXiv 2608.27206v2（2026-09-21）。arXiv comment：「Accepted to Findings of EMNLP 2026」（只看了 arXiv comment，沒看會議網站）。<https://arxiv.org/abs/2608.27206>。讀了 p.1–2、p.6–8（PDF 實體頁）。
- 寫入時做了什麼決定：無。它是 encoder 前後的視覺 token 壓縮（推論加速），沒有快取。
- 用什麼資訊做決定？寫完之後還在不在？（N1）：不適用。
- 有沒有和延後版、寫穿版、背景版比較？不適用。
- 硬體、各層頻寬、模型、模態：單張 RTX 4090；Qwen2.5-VL-7B（另有 3B）；圖文（DocVQA、TextVQA 等）〔原文 p.7–8〕。
  - **這張卡只為 Q3 記數字**（vision encoder vs LLM prefill，實測）：
    - 表 2，Qwen2.5-VL-7B、fixed-resolution（MinPix＝MaxPix＝2048×28×28〔原文 p.6〕；Qwen2.5-VL 每個 LLM 視覺 token 對應 28×28 像素，所以約每張圖 2048 個視覺 token〔判讀〕）、RTX 4090：encoder 148.84 ms、LLM prefill 217.05 ms、TTFT 365.89 ms（未壓縮的 vanilla）〔原文 p.7–8 表 2〕。encoder÷prefill ≈ 0.69〔計算〕。
    - 圖 2（Qwen2.5-VL-7B，依解析度）：512²：encoder 36 ms／prefill 33 ms；1024²：197／33 ms；2048²：853／88 ms；4096²：2.82 s／315 ms〔原文 p.2 圖 2，已把圖畫出來確認藍色＝Vision Encoder〕。Qwen2.5-VL-3B：35／42、111／44、562／100 ms、1.86 s／359 ms〔原文 p.2 圖 2〕。
    - 注意：圖 2 在 1024² 的 prefill（33 ms）比表 2 在 2048 token 的 prefill（217 ms）小很多。圖 2 沒寫硬體和 token 數，可能有 max_pixels 上限或不同設定〔判讀：兩者對不上，優先用表 2〕。
- 和 D5(a) 的關係：
  - 只提供 r_v 的旁證：Qwen2.5-VL 在高解析度時 encoder 比 prefill 貴（圖 2：6–10 倍）〔原文 p.2〕；在 2048 token／張時兩者同一個量級（0.69）〔計算〕。和 VLCache 的 Qwen2.5-VL 數字方向一致（見 [D5_VLCache](D5_VLCache.md)）。
  - 不涉及儲存、放置、還原。
- 證據等級：〔原文 p.X〕；比值為〔計算〕；圖 2 與表 2 不一致為〔判讀〕。
