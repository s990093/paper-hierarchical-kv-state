# C_XQuant XQUANT: Breaking the Memory Wall for LLM Inference with KV Cache Rematerialization

- **出處**：Aditya Tomar, Coleman Hooper, Minjae Lee, Haocheng Xi, Rishabh Tiwari, Wonjun Kang, Luca Manolache, Michael W. Mahoney, Kurt Keutzer, Amir Gholami（UC Berkeley、FuriosaAI、ICSI、LBNL）。arXiv 2508.10395v1，2025-08-14，cs.LG。venue：未查證。<https://arxiv.org/abs/2508.10395>。讀了 p.1–2、p.7–8（全文 24 頁，PDF 實體頁）。Semantic Scholar 把它列為引用 HCache 的論文。
- **寫入時做了什麼決定**：**存什麼格式**——不存 KV，改存每層的輸入 activation X（量化後），decode 時再把 K、V 算回來（rematerialize）〔原文 p.1 摘要〕。
  - MHA 模型：X 本來就是 KV 的一半，所以「立刻省 2 倍」〔原文 p.1〕。
  - GQA 模型：離線對 W_k、W_v 做 SVD，prefill 時把 X 投影到 d/g 維的 latent（X·U_k、X·U_v），**記憶體和 GQA 的 KV 一樣大**〔原文 p.8〕。好處在於這個 latent 比較好量化〔原文 p.8〕。
  - XQuant-CL：利用相鄰層 X 很像，存跨層差值，更進一步壓縮〔原文 p.1〕。
- **用什麼資訊做決定？寫完之後還在不在？（N1）**：X 只在 forward 經過時存在〔判讀，與 HCache 同類〕；格式在寫入時就定了。
- **有沒有和延後版、寫穿版、背景版比較？**：沒有。比的是同位元寬下的 KV 量化（KIVI、KVQuant）的 perplexity 與壓縮倍數〔原文 p.1–2 圖 1〕。沒有分層儲存。**有損**（低位元量化）。
- **硬體**：讀過的頁沒有實機延遲量測的設定（未查證）。
- **模型架構、模態**：Llama-2-7B（MHA，圖 1）〔原文 p.2〕；GQA 的例子用 Llama-3.1-8B〔原文 p.7〕；文字。
- **原文給的 GQA 算術**：Llama-3.1-8B 的 hidden 是 4,096、每 4 個 query head 共用一組 KV，所以 X 每 token 4K 維、KV 每 token 2K 維；**直接存 X 會多 2 倍記憶體**〔原文 p.7〕。
  - 這和 [10_breakthrough_plan](../../phase1_20261008/10_breakthrough_plan.md) §3A 的算術一致（Llama-3.1-8B：hidden state 8 KiB／層，KV 4 KiB／層）。
- **和本研究的關係**
  - **H6 的範圍被再次確認**：在 GQA 上，「存 hidden state／X」在無損 BF16 下沒有位元組上的好處；SVD latent 最多和 KV 打平〔原文 p.7–8〕。所以寫入時選格式（KV vs hidden）只對 MHA 這類模型有意義（本文件 §4 的表另外算出 Gemma-2／3、Zamba2 也有小幅空間）〔算術〕。
  - **威脅 H14（若容許有損）**：同樣的位元預算下，存量化後的 X 比量化 KV 準〔原文 p.1–2〕。H14 若做，對照組要包含「存量化 X」這種格式〔判讀〕。
  - 和分層放置無關：它只談 GPU 記憶體容量與頻寬〔原文 p.1〕。
- **證據等級**：〔原文 p.X〕；對 H6、H14 的意義〔判讀〕。
