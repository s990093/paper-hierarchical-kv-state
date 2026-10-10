# D6_MiniKV MiniKV: Pushing the Limits of 2-Bit KV Cache via Compression and System Co-Design for Efficient Long Context Inference

- 出處：Akshat Sharma, Hangliang Ding, Jianping Li, Neel Dani, Minjia Zhang（UIUC），arXiv 2411.18077v3（2025-06-08），venue 未查證（PDF 第 1 頁沒有標）。https://arxiv.org/abs/2411.18077 ，ID 已用 PDF 標題確認。讀了 p.1–6、p.9、p.16（grep 定位）。

- 精度決定：
  - **做什麼**：先用類 H2O 的方式（pyramid 層預算）挑出 heavy hitter（HH）和最近窗口（RW），沒挑中的直接逐出；留下來的都壓成 INT2（Key 按 sub-channel、Value 按 token）〔原文 p.2 Fig. 1、p.6〕。
  - **何時**：prefill 結束時挑一次 heavy hitter，之後整個生成過程都不更新〔原文 p.4〕。decode 期間新產生的 KV 先放 FP16 buffer，每 n_r 步壓一次〔原文 p.6〕。
  - **粒度**：每個 token 二選一：留下（INT2）或逐出。
  - **用什麼資訊**：prefill 時的累積 attention 分數，加上位置（最近窗口）。

- 有沒有和重算／載入並行結合？精度跟位置有關嗎？
  - 沒有重算或載入，GPU 內方法。
  - 跟位置的關係：位置決定的是「留不留」（最近窗口一定留），不是「用幾 bit」。留下的都是 INT2；只有 decode 時的新 token buffer 是 FP16〔原文 p.6〕。

- 反量化／解碼成本：
  - decode 時把反量化融合進後面的矩陣乘法〔原文 p.6〕。**沒有單獨的反量化時間**。
  - Table 4 是它的 prefill selective flash-attention kernel，不是反量化：序列長度 1K–8K 時 5.21–187.83 ms；標準 attention 為 10.46–130.51 ms，8K 時 OOM〔原文 p.9〕。
  - Fig. 15 是 decode 每個 token 的延遲拆解（MiniKV vs KIVI），只有圖〔原文 p.16〕。

- 品質與位置的關係：
  - 在總預算 50% 下調整 RW 和 HH 的比例：有的資料集偏好 HH（Passage Count），有的偏好較長的 RW（TriviaQA）；**只用 RW 或只用 HH 在某些任務會崩**（Lcc、TriviaQA）〔原文 p.4〕。原文結論是至少要保留 5–10% 的 HH／RW〔原文 p.4〕。

- 有沒有和延後版比？沒有。

- 和 D6 的關係〔判讀〕：
  - 它的位置效應是「保留 vs 丟」，跟 D6 的「位置 → 精度」不同。但「只看位置不夠、要搭配內容重要性」這個結論也適用於 D6：純位置的精度政策可能在某些任務崩掉。
  - 跟 Cake 的 b 無關。

- 證據等級：B。
