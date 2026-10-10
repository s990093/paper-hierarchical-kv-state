# D5_Kamera Kamera: Unified Position-Invariant Multimodal KV Cache for Training-Free Reuse

- 出處：Bole Ma, Jan Eitzinger, Harald Köstler, Gerhard Wellein（NHR@FAU, Erlangen）。arXiv 2606.23581v1（2026-06-22）。venue：無，PDF 標 "Preprint. Under review."〔原文 p.1〕。<https://arxiv.org/abs/2606.23581>。讀了 p.1–9、p.13、p.19–20（PDF 實體頁）。
- 寫入時做了什麼決定：**存什麼格式**。
  - 每個 chunk（一段影格／一張圖）存成「和位置無關的 canonical KV(B|∅)」＋一個小的低秩 patch（rank-m）〔原文 p.2 式 1〕。
  - patch 要在寫入時（compile 時）多跑一次「有前文」的 forward 才算得出來；之後重用不必再 forward〔原文 p.4、p.13〕。
  - 淘汰時可以丟掉「有條件的 KV」、只留 canonical；原文說 canonical 也可以「從標準的 vision-embedding cache 用約 1/8 的位元組重算」〔原文 p.2〕。
- 用什麼資訊做決定？寫完之後還在不在？（N1）：patch 需要「B 前面接的是哪個 A」的資訊，也就是寫入當下的上下文〔原文 p.2、p.6〕。召回時舊 patch 會過期（前文換了），要在「已固定的較早上下文」上重做一個新 patch〔原文 p.6〕。所以「當時的上下文」是寫入時才有的，但重做 patch 只需要較早、仍然存在的上下文〔判讀：不是嚴格的 N1〕。
- 有沒有和延後版、寫穿版、背景版比較？
  - 沒有。比較的是「每次重用都重新 prefill」vs「一次 forward 做 patch、之後每次 patch-apply」：約 9 次重用後回本〔原文 p.8、p.20 圖 11c〕。圖 11c 標的是「N×430 ms」vs「一次形成＋N×2.6 ms」（H100、MLA、MoonViT）〔原文 p.20〕。
  - 所有東西都在 HBM：服務時「只從 HBM 讀 content KV 和小的 factor」〔原文 p.13〕。沒有 CPU／SSD 層。
  - 原文明說：「召回連鎖（recall cascade）、這個放置問題」留給未來〔原文 p.9〕。
- 硬體、各層頻寬、模型、模態：H100〔原文 p.8、p.20〕；SGLang 正式 kernel〔原文 p.1〕；Kimi-VL（MLA）、Qwen2.5-VL（GQA）、Qwen3-VL（GQA interleaved）、Qwen3-Omni（MoE）等六個 backbone〔原文 p.1、p.20〕。模態：影片、圖片、文件、音訊（音訊效果較小）〔原文 p.8〕。
  - 第 1 頁有一個數字：「編碼 1024-token 影片段 ≈230 ms vision-tower 計算，重播存好的 KV ≈5 ms」，後面引的是 SGLang 與 vLLM 論文〔原文 p.1〕；這兩個數字在哪個硬體量的，原文沒寫〔判讀：來源不明，不要當實測引用〕。
- 和 D5(a) 的關係：
  - **支持一點**：它把「embedding cache」當成比 KV 便宜約 8 倍位元組、可以重算 canonical 的來源〔原文 p.2〕，和 D5(a) 的 e（embedding÷KV）同一個方向。
  - **不威脅分層放置**：只在 HBM、沒有 κ、沒有並行還原；作者自己把放置問題列為未來工作〔原文 p.9〕。
  - **提醒**：多段影片拼接時，直接重用獨立存的 KV 會讓多跳問題準確率減半〔原文 p.1–2〕。D5(a) 若要「讀 KV」，也要處理這個品質問題，或只做整段 session 原位還原（prefix 情境，沒有這個問題）〔判讀〕。
- 證據等級：〔原文 p.X〕；p.1 的 230 ms／5 ms 為〔判讀：來源不明〕。
