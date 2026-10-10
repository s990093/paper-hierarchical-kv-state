# D6_SparKV SparKV: Overhead-Aware KV Cache Loading for Efficient On-Device LLM Inference

- 出處：Hongyao Liu, Liuqun Zhai, Junyi Wang, Zhengru Fang, Jingshu Chen, Jun Huang，IEEE Internet of Things Journal（p.1 頁首）。https://arxiv.org/abs/2604.21231 （v3，2026-10-08）。這篇是查「引用 Cake 的論文」時找到的（Semantic Scholar citations API）。PDF 在 `/mlsteam/data/tiara/papers_d6/arXiv2604.21231.pdf`。讀了 p.2、p.4–9（p.6、p.9 只讀 grep 定位的段落）。

- 精度決定：
  - 雲端串流的 KV 用「layer-wise 非均勻量化 + Huffman 編碼」壓成 bitstream〔原文 p.7〕。動機實驗用的是均勻 5-bit + Huffman〔原文 p.4〕。
  - **精度沒有隨 chunk 是串流還是本地算而變**：它把壓縮後的大小當成既定輸入（t_stream(c) = b_c/bw + t_proc），排程只決定每個 chunk 要串流還是本地算〔原文 p.5〕。

- 有沒有和重算／載入並行結合？精度跟位置有關嗎？
  - **有，而且直接以 Cake 為基礎**：原文引 Cake 的「bidirectional」做法〔原文 p.2〕，並把它強化成「Strong Hybrid」baseline：早的 chunk 本地算、晚的 chunk 串流，**串流那段用跟 SparKV 一樣的量化 + Huffman**〔原文 p.8〕。換句話說，「Cake + 載入那段量化」已經是別人論文裡的一個 baseline。
  - 排程單位是（token chunk、層、head），每 1024 token 一個 chunk；目標是最小化每個階段 max(串流時間, 計算時間) 的總和，並遵守 token 和層的相依關係〔原文 p.5〕。
  - **明確反對用位置排程**：原文說既有的混合方案「用固定的位置順序，前面算、後面串流」，但同位置的 chunk 計算時間可差 3–5×，壓縮後大小也差很多，所以「位置是 chunk 成本的差勁代理」〔原文 p.4〕。
  - 精度跟位置無關。

- 反量化／解碼成本：t_proc（解密 + 解碼）有放進模型〔原文 p.5〕，但我讀的段落沒有給數字。本地 latency 預測的 MLP 每個 chunk 2.6 ms〔原文 p.6，grep 結果〕。平台是 RTX 5080 筆電 GPU 和 Jetson AGX，模型都是 4-bit 權重〔原文 p.7〕。

- 品質與位置的關係：沒有量位置。

- 有沒有和延後版比？沒有。

- 結果：比 Local Prefill 快 2.9–5.1×，比 CacheGen 快 1.8–2.2×；Strong Hybrid（Cake + 量化）仍比 SparKV 慢約 1.3×〔原文 p.8〕。

- 和 D6 的關係〔判讀〕：
  - **最直接的威脅之一**：「Cake 式並行 + 載入那段量化」已經被拿來當 baseline，所以 D6 不能把「量化讓 b 往前移」當貢獻。
  - **也是反方證據**：它主張位置不是好的排程依據。D6 若主張「依位置分精度」，要回答為什麼位置在 Tiara 的伺服器設定（512-token chunk、SSD/CPU）下是夠好的代理。Cake 自己的 Fig. 4 顯示伺服器上 chunk 計算時間跟 index 線性相關〔Cake 原文 p.4〕，跟 SparKV 在邊緣裝置用稀疏 attention 的觀察不同。
  - 仍是空白：SparKV 不依「串流還是重算」或「離交會點多遠」選精度。

- 證據等級：B。
