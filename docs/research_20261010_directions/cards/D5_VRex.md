# D5_VRex V-Rex: Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval

- 出處：Donghyuk Kim, Sejeong Yang, Wonjin Shin, Joo-Young Kim（KAIST）。arXiv 2512.12284v3（2025-12-24）。**HPCA 2026**：PDF 第 1 頁頁首是「2026 IEEE International Symposium on High-Performance Computer Architecture (HPCA)」〔原文 p.1〕；arXiv comment 也寫 accepted by HPCA 2026。<https://arxiv.org/abs/2512.12284>。讀了 p.1–4、p.6–8（PDF 實體頁）。
- 寫入時做了什麼決定：**放哪、怎麼排（layout）**。
  - 最近的 KV 留在加速器記憶體；超過容量時，最舊的卸載到 CPU 記憶體或 storage〔原文 p.7〕。
  - 卸載時把同一個 hash cluster 的 token 放在**連續位址**，讓之後一次 PCIe 傳輸能搬多個 token〔原文 p.7〕。分群只在「最近的 KV」上做，不用回去讀已卸載的資料〔原文 p.7〕。
- 用什麼資訊做決定？寫完之後還在不在？（N1）：用 key 的 hash 分群〔原文 p.7〕。事後仍可算，不是 N1〔判讀〕。分群在「要卸載時」做，本身就是延後版／背景版的形狀〔判讀〕。
- 有沒有和延後版、寫穿版、背景版比較？沒有。
  - 動機量測：A100 上 40K KV、InfiniGen 做 prefill、token budget 10K 時，KV 檢索佔總延遲 85%，其中 39% 是從 CPU 記憶體抓 KV〔原文 p.4〕。
  - 讀取時有逐層 prefetch（為下一層先抓選中的 KV，藏在計算後面）〔原文 p.4〕。沒有重算這個選項。
- 硬體、各層頻寬、模型、模態：
  - 模擬的加速器。邊緣：4 GB/s PCIe＋M.2 NVMe SSD 當卸載層、LPDDR5 204.8 GB/s；伺服器：HBM2e 1935 GB/s、32 GB/s PCIe 卸載到 DDR4 CPU 記憶體〔原文 p.8〕。SSD 用 MQSim、DRAM 用 DRAMSim3 模擬〔原文 p.8〕。對照 AGX Orin、A100〔原文 p.8〕。
  - Llama-3 8B＋SigLIP-ViT-L-384〔原文 p.8〕；COIN；串流影片。
  - 結果：邊緣 3.9–8.3 FPS；比 AGX Orin 快 1.9–19.7×〔原文 p.1〕。
- 和 D5(a) 的關係：
  - 是「影片 KV 放 CPU／SSD、讀時選擇性取回」的硬體版；**只有 KV 一種格式**，沒有 embedding、像素、重算〔原文 p.7–8〕。
  - 不威脅 D5(a) 的多格式問題；但它的 A100 量測（抓 KV 佔 39%）說明影片 KV 卸載的 I/O 很重〔原文 p.4〕，可以當 D5(a) 的動機引用。
- 證據等級：〔原文 p.X〕；硬體結果是模擬器數字〔原文 p.8〕。
