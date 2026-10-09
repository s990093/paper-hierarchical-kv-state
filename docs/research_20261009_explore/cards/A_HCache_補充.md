# A_HCache_補充 Fast State Restoration in LLM Serving with HCache（補充卡）

> HCache 已在 `03_paper_map.md` A1（E03）與 `write_path_20261008/02_sota_write_techniques.md`。這張卡**不重做**，只補和 **H6（寫入時選格式）** 與 **S5（依位置）** 直接相關、而 03／02 沒寫到細節的兩點，再加上後續論文到哪了。

- **出處**：Shiwei Gao, Youmin Chen, Jiwu Shu（Tsinghua）。EuroSys 2025（PDF p.1 頁首「EuroSys '25, March 30–April 3, 2025, Rotterdam」）。本機 PDF `hcache2025_arXiv2410.05004.pdf`，這次讀了 p.6–7、p.12–13（PDF 實體頁）。

## 補充 1：HCache 比過「依 token 位置切」，結果比「依層切」慢
- HCache 考慮兩種切法：token-wise（前幾個 token 用 hidden state，後面的用其他方法，例如 KV 卸載）與 layer-wise（前幾層用 hidden state，其他層用別的方法）〔原文 p.6–7〕。
- 原文說 token-wise 切法會產生不規則的 GEMM 形狀，cuBLAS 對它最佳化較差，「token 較少的 GEMM 可能和 token 較多的一樣久」，即使對齊到最近的最佳大小，計算與傳輸之間仍有 bubble〔原文 p.7〕。
- 實測（13B、1,024 token 的歷史、1×A100＋1×SSD）：layer-wise 方案是 31 層 hidden state＋9 層重算；naive token-wise 是 794 個 token 用 hidden state、230 個重算，**慢 12%**；對齊到 768 的 token-wise **仍慢 7%**〔原文 p.12〕。
- 〔判讀〕這是「依位置切」在**格式選擇**（hidden state vs 重算）上的直接負面證據：在 HCache 的設定裡，位置維度不如層維度。對 S5／H6 的意義：如果要做「依位置選格式」，必須說明為什麼這裡的 GEMM 形狀問題不成立（例如 chunk 夠大、512 token 以上）。

## 補充 2：GQA 明確在範圍外
- 「MQA、GQA……可以先把 hidden state 投影到低秩表示再存，但這要改模型結構，超出本文範圍。HCache 目前支援不改模型結構的 MHA 模型」〔原文 p.13〕。
- 與 §3A 的算術一致：Llama-3.1-8B（GQA）的 hidden state 是 KV 的 2 倍。

## 後續論文：H6 現在做到哪
| 論文 | 做了什麼 | 和 H6 的差距 | 出處 |
|:--|:--|:--|:--|
| HybridServe（ICCD'25） | 寫入時依比例把 block 存成 KV 或 activation，讀取時一部分重算、一部分載入；MHA（OPT） | 單一批次的卸載、不跨請求、單層 host | [A_HybridServe](A_HybridServe.md) |
| Krul（arXiv 2507.08045） | 引用 HCache，說 HCache 要支援 MQA 得改 hidden state 的表示 | 走的是跨層 KV 共享（有損），不是 hidden state；但它在寫入時就決定每層前段不存、讀取時重算 | [A_Krul](A_Krul.md)〔Krul 原文 p.12（引用 HCache）、p.8–9（不存前段）〕 |
| SuffixReplay（arXiv 2609.33477） | 混合模型上改存稀疏的 hidden state anchor，讀取時重播 | 有損；只針對線性注意力層 | [A_SuffixReplay](A_SuffixReplay.md) |
| Apt-Serve（arXiv 2504.07494） | KV cache＋「hidden cache」的混合快取，增加 batch | 未讀原文，只看摘要；摘要說模型 13B–66B | arXiv 摘要 |
| KVPR（ACL Findings'25，已在 03） | 讀取時前段傳 activation 重算、其餘傳 KV | 讀取時決定、單一批次 | 03 A4 |

## 和本研究的關係
- **H6 威脅**：H6 的停損條件之一是「新穎性不夠」。目前「寫入時選格式（KV／activation）＋讀取時混合重算與載入」已有 HCache（依層）與 HybridServe（依比例）兩個前例，而且 HCache 量過 token-wise 比較慢〔原文 p.12〕。剩下的空白只有「跨請求 × 多層 × 依位置」，而依位置那一維已有負面證據〔判讀〕。
- **證據等級**：〔原文 p.X〕；判讀如標示。
