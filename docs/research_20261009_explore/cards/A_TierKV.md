# A_TierKV TierKV: Long-Context On-Device LLMs via Predictive Multi-Tier KV Caching

- **出處**：Zhihao Shu, Md Musfiqur Rahman Sanim, Jie Hu, Kun Yuan, Minghai Qin, Gagan Agrawal, Wei Niu（UGA、PKU、Western Digital Research）。arXiv 2609.21172v2（v1 2026-09-18）。PDF 頁首寫 **EuroSys '27**（April 19–23, 2027, Rabat），並有 DOI 10.1145/3842654.3848586〔原文 p.1、p.4 頁首〕；未在 ACM DL 核對。<https://arxiv.org/abs/2609.21172>。**引用 Cake**（ref [29]），但只在相關研究裡把它歸為「縮短 TTFT 的行動端系統」〔原文 p.14〕。讀了 p.1–5、p.8、p.14（PDF 實體頁）。
- **寫入時做了什麼決定**：**依位置把 token 切成三層**（prefill 結束、decode 開始前決定）：
  - Tier-0 (0, n]：不壓縮（保護對壓縮敏感的位置）；Tier-1 (n, m]：SVD 低秩壓縮；Tier-2 (m, L]：放 flash，保持全秩〔原文 p.4〕。
  - 每個請求決定兩個邊界 n、m 和一個整體壓縮等級；每層的秩 R_l 是離線校準〔原文 p.4〕。
  - 「因為快取配置發生在 decode 之前，邊界是事先用 prefill 階段的訊號選的」〔原文 p.4〕。
- **用什麼資訊做決定？寫完之後還在不在？（N1）**
  - 用 **prefill 的 hidden state** 預測這個請求之後的快取需求（總長度 L）〔原文 p.1 摘要、p.3〕。原文：「prefill 在行動端是免費的預測器……伺服器端要拿到類似訊號，通常要另外跑預測器」〔原文 p.3〕。〔判讀〕這是 N1 的一種弱形式：資訊在寫入當下免費，之後要再付一次計算。
  - 原文說這種事先規劃「消除了反應式淘汰的循環依賴」〔原文 p.1〕。
- **有沒有和延後版、寫穿版、背景版比較？**：對照是 llama.cpp、MNN-LLM 等行動端框架、固定平均長度配置（Static Mean Length）與淘汰類方法〔原文 p.1、p.8〕。prefill 吞吐最多 17.6 倍（對 MNN-LLM）、RAM 內的 KV 少 12.5–34%，準確度小幅下降〔原文 p.1〕。**沒有「事後再降級到 flash」的延後版**。
- **硬體**：OnePlus 12（Snapdragon 8 Gen 3、Adreno 750、12 GB RAM）為主，另有 OnePlus 11（8 Gen 2、16 GB）、Pixel 8（Tensor G3、8 GB）〔原文 p.8〕。行動端 host-to-device 只有 1.66 GB/s，卸載 9.6K token 的 Llama-3.2-3B 每步多 ~189 ms〔原文 p.4〕。
- **模型架構、模態**：8 個文字、視覺、語音模型（例：Llama-3.2-3B）〔原文 p.1、p.4〕。
- **範圍**：**單一請求的 decode**，不是跨請求重用；Tier-1 有損〔原文 p.4〕。
- **和本研究的關係**
  - **對 S5 新穎性是部分威脅**：「寫入時（prefill 後）依 token 位置把 KV 分到不同層」已經有人做，只是方向和目的都不同——它把**尾段**放 flash（原文：Tier-2 只放「long tail」，目的是限制 RAM 用量〔原文 p.4〕），S5 是把**前段**給重算或慢層；它是單請求、有損、行動端〔原文 p.4；比較是〔判讀〕〕。
  - **支持 N1 的論點**：「prefill 當下的 hidden state 是免費的預測訊號」〔原文 p.3〕。如果 S5 要用寫入時才有的資訊（例如注意力分布）決定位置，這是先例〔判讀〕。
  - 相關研究要引用並區分（三點：單請求 vs 跨請求、有損 vs 無損、尾巴放慢層 vs 前段給重算）〔判讀〕。
- **證據等級**：〔原文 p.X〕；威脅評估〔判讀〕。
