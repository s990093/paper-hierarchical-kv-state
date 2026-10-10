# D6_KVQuant KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization

- 出處：Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W. Mahoney, Yakun Sophia Shao, Kurt Keutzer, Amir Gholami（UC Berkeley 等），NeurIPS 2024（p.1）。https://arxiv.org/abs/2401.18079 （讀的是 v6）。PDF 在 `/mlsteam/data/tiara/papers_d6/arXiv2401.18079.pdf`。讀了 p.5–7、p.9–10、p.19–20、p.26（grep 定位後細讀）。

- 精度決定：
  - **做什麼**：Key 在 RoPE 前按 channel 量化、Value 按 token 量化；非均勻資料型態（nuqX）；dense-and-sparse（例如 1% outlier 存全精度）；第一個 token 保持 fp16〔原文 p.5–7〕。
  - **何時**：非均勻資料型態和 Key 的 per-channel scale 在**離線**用 calibration 資料算好；Value 的 per-token scale 和 outlier 門檻在**線上**、每個新 token 進來時算〔原文 p.5–6〕。KV 在 GPU 內寫入時就量化。
  - **粒度**：per-channel／per-token；每層一個資料型態。
  - **用什麼資訊**：離線 calibration（16 筆 Wikitext-2 樣本，2K 長度）〔原文 p.7〕；位置資訊只用在「第一個 token」。

- 有沒有和重算／載入並行結合？精度跟位置有關嗎？
  - 沒有重算或載入。
  - **跟位置有關（只有第一個 token）**：Attention Sink-Aware Quantization，只把第一個 token 留在 fp16〔原文 p.6〕。

- 反量化／解碼成本（A6000，LLaMA-2-7B-32K，batch 1，單位 µs）：
  - Table 22：每個新 token 的「打包／壓縮」Key 4.5 µs、Value 4.1 µs，不隨序列長度變〔原文 p.26〕。
  - 融合反量化的 Key matvec 總時間在 l=2K／4K／16K 時為 25.6／39.9／126.3 µs；fp16 matvec 為 33.3／59.1／219.4 µs〔原文 p.10 Table 6、p.26 Table 22〕。也就是 decode 時的融合 kernel 比 fp16 還快（1.2–1.7×）〔原文 p.9〕。
  - 注意：這是 decode 階段的 matvec，不是「把一大段 KV 一次反量化成 BF16 放進 paged cache」的成本。後者才是 Cake 式還原的情況，這篇沒有量。

- 品質與位置的關係：
  - **第一個 token 的精度影響很大**：Table 13，Wikitext-2 困惑度，nuq2（2-bit、沒有 sparse）時，Llama-2-13B 不保護 sink 為 23.34，只保護第一個 token 為 9.59（fp16 為 4.57）；Llama-2-7B 為 11.20 → 7.03〔原文 p.20〕。位元數越低，效果越明顯〔原文 p.19〕。
  - 4-bit 時差距很小，例如 nuq4 Llama-2-13B 4.62 → 4.60〔原文 p.20〕。

- 有沒有和延後版比？沒有。

- 和 D6 的關係〔判讀〕：
  - **支持「前段要高精度」**，但真正敏感的只有極少數 token（第一個）。
  - **和 Cake 配起來是好消息**：Cake 永遠從最前面開始重算，所以 sink 永遠是精確的 BF16（見 D6_Cake）。D6「前段不存」的政策正好也不會傷到 sink。換句話說，「前段高精度」這條理由在 Cake 底下已經自動滿足，D6 不用再為它付出儲存。
  - Cake 的 §5.5 用的 8-bit／3-bit 就是 KVQuant 的方法〔Cake 原文 p.6〕。

- 證據等級：B（相關段落與表格細讀）。
