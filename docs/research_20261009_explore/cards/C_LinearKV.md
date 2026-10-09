# C_LinearKV LinearKV: One Cached State Suffices for Position-Independent Caching in Hybrid LLMs

- **出處**：Yirui Liu, Ruoling Qi, Longwen Wang, Xuaner Wu, Jian Chen, Yuxin Jin, Jiawei Shao, Xuelong Li（China Telecom TeleAI、SJTU、XJTU、University at Buffalo）。arXiv 2608.11231v1，2026-07-31，cs.AI。venue：未查證。<https://arxiv.org/abs/2608.11231>。讀了 p.1–5、p.7 的實驗設定（全文 14 頁，PDF 實體頁）。
- **寫入時做了什麼決定**：**每個 chunk 存什麼**。可重用的 chunk 先離線各自 prefill，存成一份「混合快取」：FA 層的 per-chunk KV，加上每個遞迴層在該 chunk 結尾的狀態 S_local〔原文 p.4 圖 2〕。
  - 讀取時：FA 層照位置無關快取（PIC）的做法把 KV 接起來；每個遞迴層把 K 個 chunk 的狀態併成一個初始狀態。原文的結論是**只用最後一個 chunk 的狀態**當初始值就夠了，不必精確合成〔原文 p.1、p.4〕。
- **用什麼資訊做決定？寫完之後還在不在？（N1）**：S_local 是 chunk 單獨 prefill 時的結尾狀態，只在那次 forward 存在〔判讀〕。精確合成另外需要每個 chunk 的轉移矩陣，原文的精確版也要存〔原文 p.4 演算法 1〕。
- **有沒有和延後版、寫穿版、背景版比較？**：沒有。比的是兩種初始化（最後一塊 vs 精確合成，後者是同期的 HYPIC 的做法）× 三種 PIC 選擇器（CacheBlend、EPIC、ProphetKV）〔原文 p.1〕。
  - GDN 模型上兩者打平，最多恢復約 92% 品質；Mamba-2 模型（Granite）上精確合成崩掉：EPIC 下只恢復 46.6%，最後一塊初始化是 86.8%〔原文 p.1〕。
  - 最後一塊初始化的 TTFT 是 full prefill 的 0.46 倍；精確合成再多 5–17%〔原文 p.1〕。
- **硬體**：NVIDIA H800 80 GB，HuggingFace Transformers，bfloat16；TTFT 在單張獨占 GPU 上量〔原文 p.7〕。
- **模型架構、模態**：Granite（Mamba-2）、OLMo（GDN）、Qwen3.6-27B（GDN）；文字；LongBench QA、RULER，8K–32K〔原文 p.1、p.5〕。**有損**（PIC 本身有損）。
- **和本研究的關係**
  - 和 H7 只是旁支：它示範了「每個 chunk 存一份結尾狀態」這種寫入時的格式，讓遞迴層也能做位置無關重用〔原文 p.4〕。但本研究第一階段是無損的前綴重用，PIC 不在範圍內〔判讀〕。
  - 給 H7 的提醒：Mamba-2 和 GDN 的狀態行為不同，在 GDN 上成立的近似在 Mamba-2 上可能崩〔原文 p.1〕。H7 若選模型，要說清楚是哪一族〔判讀〕。
- **證據等級**：〔原文 p.X〕；判讀如標示。只讀到 p.7，後面的消融沒讀。
