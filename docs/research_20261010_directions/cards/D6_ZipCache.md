# D6_ZipCache ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification

- 出處：Yefei He, Luoming Zhang, Weijia Wu, Jing Liu, Hong Zhou, Bohan Zhuang（浙江大學、NUS、Monash），arXiv 2405.14256v1（2024-05-23，頁首寫「Preprint. Under review.」；正式 venue 未查證）。https://arxiv.org/abs/2405.14256 ，ID 已用 PDF 標題確認。讀了 p.1–2、p.6–10（grep 定位）。

- 精度決定：
  - **做什麼**：混合精度，salient token 用 4-bit，其他 token 用 2-bit；Key 按 channel，Value 用 channel-separable tokenwise 量化〔原文 p.8〕。
  - **何時**：prefill 時決定；decode 時採 streaming，每多生成 100 個 token 就重做一次壓縮〔原文 p.8〕。原文沒寫重做時是否保留全精度副本（**未查證**）。
  - **粒度**：每個 token。
  - **用什麼資訊**：正規化 attention 分數，也就是每一欄的累積 attention 除以該欄的非零個數，用來修正下三角遮罩造成的「越前面的 token 分數越高」偏差〔原文 p.6〕。為了能用 FlashAttention，只對 10% 的 probe token 算完整 attention 分數（5% 最近的 token + 5% 隨機）〔原文 p.7、p.9〕。
  - salient 比例是手動指定的，不會自動調整（原文自己列為限制）〔原文 p.10〕。

- 有沒有和重算／載入並行結合？精度跟位置有關嗎？
  - 沒有重算或載入，GPU 內方法。
  - 精度**不直接看位置**，但原文明確指出位置偏差：累積分數會讓最早的 token 永遠比最後的 token 高分〔原文 p.2、p.6〕。修正後，GSM8K CoT 範例的 salient token 集中在**prompt 結尾**（也就是要回答的問題）〔原文 p.6〕。
  - 原文也批評「保留最近 token 為全精度」（KIVI 的做法）並非最佳，因為最近的 token 不一定最重要〔原文 p.2、p.6〕。

- 反量化／解碼成本：只跟 MiKV 比（A100、LLaMA3-8B、4096 長度）：prefill 延遲 −37.3%、decode 延遲 −56.9%、GPU 記憶體 −19.8%〔原文 p.9〕。**沒有單獨的反量化時間**。

- 品質與位置的關係：
  - Table 3（GSM8K CoT）：Mistral-7B FP16 41.62，ZipCache（4/2 bit、60% 用 2-bit、壓縮 4.98×）41.24。LLaMA2-7B：FP16 14.18、KIVI 13.19、MiKV 9.02、ZipCache 13.50〔原文 p.8〕。原文說 KIVI 在 LLaMA3-8B 掉 7.89%〔原文 p.8〕。
  - probe 選法（Table 2，LLaMA3-8B GSM8K，40% 4-bit／60% 2-bit）：全部 token 52.54、隨機 47.46、特殊 token 46.78、最近 51.10、隨機加最近 52.08〔原文 p.7〕。

- 有沒有和延後版比？沒有。

- 和 D6 的關係〔判讀〕：
  - **威脅「純依位置分精度」**：它的結論是重要性看內容，不看位置；純位置規則（例如尾端 BF16）會漏掉不在尾端的重要 token。
  - 但它的 saliency 需要問題的 query 才算得準（salient token 是問題本身）。對「寫入時就決定」的 Tiara 來說，存前綴時還不知道未來的問題。這代表依 saliency 的精度只有在**讀取時**（或延後）才算得出來，間接支持延後版。
  - 跟 Cake 的 b 無關。

- 證據等級：B。
