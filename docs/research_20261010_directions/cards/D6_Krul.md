# D6_Krul Krul: Efficient State Restoration for Multi-turn Conversations with Dynamic Cross-layer KV Sharing

- 出處：Junyi Wen, Junyuan Liang, Zicong Hong 等（中山大學、HKUST 等），arXiv 2507.08045v2（2025-08-26）。PDF 用 PVLDB 樣板，但卷期是樣板佔位（Vol. 14、2020），正式 venue 未查證。也是查「引用 Cake 的論文」時找到的。PDF 在 `/mlsteam/data/tiara/papers_d6/arXiv2507.08045.pdf`。讀了 p.1–2、p.5、p.8–9、p.12。

- 精度決定：
  - **不是量化**，是跨層 KV 共享（相鄰兩層 attention 很像時共用一份 KV，方法沿用 MiniCache）〔原文 p.8 腳註〕。
  - **何時**：對話結束、變成非活躍時，先壓縮再 offload 到 CPU 記憶體〔原文 p.5〕。屬於**寫入（offload）時決定**。
  - **粒度**：每個對話選一組要共享的層對（layer pair）〔原文 p.1〕。
  - **用什麼資訊**：每個對話自己的層間 attention 相似度（preemptive selector）〔原文 p.1〕。

- 有沒有和重算／載入並行結合？精度跟位置有關嗎？
  - **有，而且壓縮跟重算／載入的切分在寫入時就綁在一起**：
    - 先解出重算比例 r_c，讓重算時間 T_C 和載入時間 T_L 越接近越好〔原文 p.8 式 1〕。
    - **再依 r_c 分配儲存空間：總空間 C = c×(1−r_c)×N**。也就是會被重算的那部分**根本不存**〔原文 p.8〕。
    - 還原時用 pyramid 形狀：每層載入量隨層數線性增加，重算量隨層數線性減少；深層有共享的 KV，所以多分配載入〔原文 p.9〕。
  - 原文直接點出問題：KV 一被壓縮，原本重算和載入的平衡就被破壞，兩條流會出現 bubble，需要新的排程〔原文 p.5〕。
  - 原文對 Cake 的描述：Pensieve、Cake、MobileLLM 都是「重算前段、非同步載入後段」〔原文 p.12〕。
  - 精度（這裡是共享與否）跟 token 位置無直接關係；切分是 token × 層的 pyramid。

- 反量化／解碼成本：不適用（不是量化）。硬體為 4×A100 80GB、128 GB DRAM、PCIe Gen4〔原文 p.9〕。

- 品質與位置的關係：沒有量位置。

- 有沒有和延後版比？沒有。baseline 是全部重算、全部載入（AttentionStore）等〔原文 p.9〕。

- 和 D6 的關係〔判讀〕：
  - **D6「前段不存、因為會被重算」這個想法的最接近先例**：Krul 在 offload 時就依 r_c 只存 (1−r_c) 的 KV〔原文 p.8〕，並且承認壓縮會改變重算／載入的平衡〔原文 p.5〕。
  - **跟 D6 的差別**：(1) 壓縮手段是跨層共享，不是 INT8／FP8／INT4；(2) 切分是 token × 層的 pyramid，不是 Cake 的 token 方向會合點 b；(3) 沒有「依離 b 的距離給不同精度」。
  - **同一個風險**：寫入時就把 r_c 定死、不存重算那段，之後硬體負載變了就不能多載一點。Krul 用 calibration 解 r_c〔原文 p.8〕，沒有討論 r_c 過時的情況（我讀的段落裡沒看到）。

- 證據等級：B。
