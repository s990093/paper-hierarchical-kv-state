# A_WhereKVLive Where Should the KV Cache Live? Placement Policies Across GPU, CPU, and SSD for Long-Lived Sessions

- **出處**：Srikanta Datta Tumkur, Jay Iyer, Mehar Simhadri, Sai Pavan Kumar, Sai Kapil Kumar, Ramesh Nampelly（Vizuara）。arXiv 2609.16215v1，2026-09-14，cs.AI。venue：未查證（沒有會議頁首）。<https://arxiv.org/abs/2609.16215>。本機 PDF `09_WhereShouldKVLive_arXiv2609.16215.pdf`，讀了全文 p.1–8（PDF 實體頁）。
- **補充說明**：這篇已在 `phase1_20261008/report/related_papers_update.md` §c 和 `07_report.md` 提過（只讀 p.1–3），但 `03_paper_map.md` 沒收。本卡補上全文讀到的「它的策略在什麼時候做決定」與限制。
- **寫入時做了什麼決定**：**沒有**。演算法 1 是「存取時」的路徑：miss 就往上抓、提升到 HBM；HBM 滿了就把分數最低的降到 DRAM，再串到 SSD〔原文 p.3 Algorithm 1、§III-B〕。新 block 怎麼放，原文沒有另外寫——〔判讀〕等於「先放最上層、滿了才往下降」，也就是延後版。
- **用什麼資訊做決定？（N1）**：recency、reuse frequency、「predicted reuse」（作者發現 repo 附的版本和 recency 逐位元相同）、以及另外加的 EWMA next-touch 預測器〔原文 p.1 摘要、p.4、p.8〕。都是事後仍在的存取紀錄，不是 N1。
- **有沒有和延後版、寫穿版、背景版比較？**
  - 比的是 placement policy（都是延後版）× prefetch lookahead，對照 GPU-only（回來就重算）、full-CPU-offload（FlexGen 式）、prefix-reuse（RadixAttention 式）〔原文 p.4〕。
  - 結論 1：tiering 讓每張 GPU 的 session 數 ×73.02、每 session 成本 ÷62.04，但這是層的容量倍數（1+8+64），**不是放置策略的效果**〔原文 p.1、p.4〕。
  - 結論 2：batch 1 時 decode 是算力瓶頸，策略幾乎不影響吞吐，只影響 PCIe 搬移量和 TTFT；chat 用 recency 最好（搬移量比 reuse-frequency 少 2.30 倍），agent 與 document QA 用 reuse-frequency 最好〔原文 p.1〕。
  - 結論 4：**prefetch 不划算**——在策略 × 快取大小的網格上，連會讀未來的 oracle prefetch 都沒有一格贏過不 prefetch〔原文 p.1〕；原因是它的模型讓投機抓取和需求抓取共用同一條被爭用的連結〔原文 p.8〕。
  - 沒有寫穿版、沒有寫入時版。
- **硬體**：**全部是模擬**。離散事件模擬器，用 random-forest 執行時間預測器校準；預設層形狀 1:8:64〔原文 p.1、p.4〕。批次 1、單 GPU；fetch stall 不到 TPOT 的 1%〔原文 p.8〕。
- **模型架構、模態**：原文主文未寫明模型（〔判讀〕看 p.1–8 沒找到）；合成的 chat、agent、document QA 產生器〔原文 p.4、p.8〕。
- **限制（原文自己寫的）**：合成負載的重用結構「依構造有利於對應的策略」，真實 trace 上差距會縮小；batch 1；DRAM/SSD 的 stall 分攤是估的，不是直接量的〔原文 p.8〕。
- **和本研究的關係**
  - **威脅重要性（不是新穎性）**：「好處主要來自容量，不是放置策略」〔原文 p.1〕。這和第一階段 Q5「容量才是主因」方向一致（`07_report.md` §0 第 6 點已引用）。
  - **對「讀取時版」對照組的提醒**：它發現 prefetch 在頻寬被爭用時不划算〔原文 p.1、p.8〕。我們 §2 的「讀取時版（預取）」不一定是強對照組；但這只是在它的模擬模型下〔判讀〕。
  - **沒有支持任何 H**；它也沒有測 Cake 式的重算＋載入，所以它的「策略不重要」不能直接套到「會合點受位置影響」的情境〔判讀〕。
- **證據等級**：〔原文 p.X〕（模擬、合成負載）；對本研究的意義〔判讀〕。依 `docs/research_20260924/novelty_sota.md` 的紅線，只能當「領域現象」引用。
