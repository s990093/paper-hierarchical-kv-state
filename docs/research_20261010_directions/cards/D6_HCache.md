# D6_HCache Fast State Restoration in LLM Serving with HCache

- 出處：Shiwei Gao, Youmin Chen, Jiwu Shu（清華大學），EuroSys 2025（p.1）。https://arxiv.org/abs/2410.05004 。讀了 p.1–2、p.6–7、p.13（grep 定位）。

- 精度決定：**沒有**。HCache 標榜無損：「HCache is a lossless method for state restoration」〔原文 p.13〕。
  - 存的是中間激活（hidden states），還原時從激活重算 KV；儲存空間比存 KV 小 1.92–2.40×〔原文 p.1〕。

- 有沒有和重算／載入並行結合？精度跟位置有關嗎？
  - **有並行，但是按「層」切，不是按 token 切**：bubble-free 排程器把模型的層分成兩群。算力快的平台上，前 L_O 層從 token 重算，其餘 L_H 層用 HCache（傳激活再算 KV）；I/O 快的平台上，前 L_H 層用 HCache，後 L_O 層直接傳 KV。L_H、L_O 用離線 profile 的速度解出來，讓傳輸和計算同時結束〔原文 p.7〕。
  - 它也考慮過按 token 切，但因為 cuBLAS 對不規則的 GEMM 大小效率差、仍有 bubble，最後選了按層切〔原文 p.7〕。
  - 精度跟位置無關（全部無損）。

- 反量化／解碼成本：不適用。

- 品質與位置的關係：不適用（無損）。

- 有沒有和延後版比？沒有。

- 和 D6 的關係〔判讀〕：
  - 原文說量化「可以用在 HCache 上，減少 hidden state 的大小」，但沒有做〔原文 p.13〕。所以「激活 + 量化 + 重算／載入並行」是它留下的空白。
  - 它選按層切而不按 token 切的理由（GEMM 效率）〔原文 p.7〕，對 D6 的 token 位置政策是一個實作風險：Cake 用 512-token chunk，應該沒這個問題，但若 D6 的精度邊界切得很細，要注意 kernel 效率。

- 證據等級：B。
