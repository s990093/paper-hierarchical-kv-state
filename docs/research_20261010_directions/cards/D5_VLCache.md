# D5_VLCache VLCache: Computing 2% Vision Tokens and Reusing 98% for Vision-Language Inference

- 出處：Shengling Qin, Hao Yu, Chenxin Wu 等（Qwen Team, Alibaba；TairKVCache Team, Alibaba Cloud）。arXiv 2512.12977v2（2025-12-18）。venue：無（PDF 標 "Preprint."，arXiv 無 comment）〔原文 p.1〕。<https://arxiv.org/abs/2512.12977>。讀了全文 14 頁（PDF 實體頁）。
- 寫入時做了什麼決定：
  - 圖片第一次出現（cache miss）時，**encoder 輸出和 KV 兩種格式都存**到後端 KVCache store〔原文 p.5〕。實作上 embedding 存在分散式 KV store（Tair KVCache Store）〔原文 p.6〕。
  - KV 存的是**套 RoPE 之前**的版本，讓之後能放到別的位置〔原文 p.6〕。
  - 沒有「只存一種」的選擇，也沒有放哪一層（GPU／CPU／SSD）的決定。store 背後的層級沒寫〔未查證〕。
- 用什麼資訊做決定？寫完之後還在不在？（N1）：用的是「這張圖的 pixel hash 有沒有見過」〔原文 p.6〕。這資訊事後仍在，不是 N1〔判讀〕。
- 有沒有和延後版、寫穿版、背景版比較？
  - 沒有。寫入策略固定是「兩種都存」。
  - 但它量了**三條還原路徑**的 TTFT〔原文 p.7 表 1–2〕：Origin（從像素：ViT＋LLM prefill）、w/o ViT（從 embedding：只跑 LLM prefill）、Static r=0（KV 全部重用）。這是目前看到最接近「像素／embedding／KV 三選一」的實測。
  - 沒有拆出讀取（從 store 搬回來）的時間，也沒有變動頻寬〔判讀：原文未提〕。
- 硬體、各層頻寬、模型、模態：
  - NVIDIA H20-3e；Qwen3-VL-8B、Qwen3-VL-32B〔原文 p.7〕；附錄加 Qwen2.5-VL-7B／32B〔原文 p.12–13 表 6、8〕、Qwen3-VL-30B-A3B〔原文 p.14 表 10〕。附錄沒再寫硬體〔判讀：應同 p.7〕。
  - 圖文（多張圖片＋20 個文字 token）。頻寬：無。
  - 量到的 ViT／LLM 拆分（〔計算〕＝Origin − w/o ViT；TTFT 秒數取自〔原文 p.7 表 1、p.13 表 6〕）：

| 模型 | 圖片 token | ViT (s) | LLM prefill (s) | ViT÷LLM |
|:--|--:|--:|--:|--:|
| Qwen3-VL-8B | 1K | 0.106 | 0.286 | 0.37 |
| Qwen3-VL-8B | 20K | 1.219 | 7.565 | 0.16 |
| Qwen3-VL-32B | 20K | 1.730 | 18.549 | 0.09 |
| Qwen2.5-VL-7B | 1.4K | 0.313 | 0.322 | 0.97 |
| Qwen2.5-VL-7B | 20K | 56.50 | 6.251 | 9.0 |

  - 原文自己說：Qwen3-VL 的加速「主要來自 KV 重用」，因為它的 ViT 比較有效率〔原文 p.7〕。
  - 注意：Static r=0（KV 全部重用、不重算）在 Qwen3-VL-8B、20K 時 TTFT 仍有 4.430 s〔原文 p.7 表 1〕。原文沒說這 4.4 s 花在哪〔判讀：可能含從 store 讀 KV 的時間，未查證〕。
- 和 D5(a) 的關係：
  - **威脅（部分）**：「embedding 和 KV 都存在一個外部 store、讀取時跳過 ViT」已經有人做了，而且是 Qwen 團隊在 SGLang 上做〔原文 p.6〕。
  - **沒做的**：(1) 沒有「依頻寬／容量決定存哪一種」；(2) 沒有分層放置；(3) 沒有 Cake 式「一邊算、一邊讀」的並行還原；(4) 沒有和延後版比較。
  - **支持**：它的數字顯示 r_v（ViT÷LLM）強烈依模型而定：Qwen3-VL 約 0.1–0.37，Qwen2.5-VL 約 1–9〔計算〕。所以「要不要存 embedding」對 Qwen2.5-VL 值得，對 Qwen3-VL 價值小。
- 證據等級：〔原文 p.X〕為主；ViT／LLM 拆分為〔計算〕；store 層級〔未查證〕。
