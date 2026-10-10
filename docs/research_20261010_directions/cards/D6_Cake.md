# D6_Cake Compute Or Load KV Cache? Why Not Both?

- 出處：Shuowei Jin, Xueshen Liu, Qingzhao Zhang, Z. Morley Mao（密西根大學），ICML 2025（PMLR 267）。https://arxiv.org/abs/2410.03065 。兩個版本都讀了：arXiv v2（2025-02-20，`/mlsteam/data/tiara/papers/02b_Cake_arXiv2410.03065.pdf`）的 p.2–7；ICML 版（`02_Cake_ICML25_PMLR_v267.pdf`）的 p.7–9。以下頁碼沒特別標的是 arXiv v2。

- 精度決定：
  - **Cake 本身不做精度決定**。預設 KV dtype 是 BF16〔原文 p.5〕。
  - 評估時「搭配」KVQuant 的 8-bit 與 3-bit 量化，當成正交技術〔原文 p.6〕。原文沒有說只量化載入的那段，也沒有說量化在何時做；Table 6 只標「low-precision compression」〔原文 p.7〕。
  - ICML 版明說：「Cake treats the KV cache as data and is agnostic to its specific representation or compression scheme.」〔ICML 版 p.9〕

- 有沒有和重算／載入並行結合？精度跟位置有關嗎？
  - **核心機制就是並行**：GPU 從第一個 chunk 往後算，I/O 從最後一個 chunk 往前載，兩邊在中間會合〔原文 p.4，Fig. 2〕。
  - 依據的觀察：越後面的 token 算起來越貴（要 attend 前面所有 token），但每個 token 的 KV 大小一樣，所以 I/O 成本跟位置無關〔原文 p.4〕。Fig. 4 用 512-token chunk 量到每個 chunk 的 prefill 時間隨 index 線性增加〔原文 p.4〕。
  - 會合點會自己跟著算力／頻寬變動移動〔原文 p.5〕。
  - **量化有和並行結合，但是均勻量化**：Table 6（2×A100、100% 利用率、chunk 512、16K、LongAlpaca-13B）顯示精度越低，Cake 相對 compute-only 的加速越大。例如 32 Gbps 時，16-bit／8-bit／3-bit 分別是 1.59／2.12／3.73×；相對 I/O-only 則是 2.63／1.85／1.37×〔原文 p.7；ICML 版 p.8 §5.6〕。
  - **精度跟位置無關**：沒有任何位置相關或「依 chunk 會被載入還是重算」而定的精度。

- 反量化／解碼成本：原文沒有量。Cake 自己的 runtime 開銷是「每步只檢查下一個 chunk 到了沒」，對 vLLM 的 step time 影響可忽略〔ICML 版 p.9，Fig. 7〕。

- 品質與位置的關係：沒有量。量化實驗只報加速倍數，沒有報任何準確率／困惑度（全文搜尋 accuracy／quality／perplexity 都沒有相關結果）。

- 有沒有和延後版比？沒有。

- 和 D6 的關係〔判讀〕：
  - **威脅**：D6 的第一個想法（「量化載入的那段 → 位元組變少 → 載入變快 → b 往前移」）在 Cake 的 Table 6 已經有均勻量化的版本。b 往前移這件事本身不新。
  - **支持**：Cake 只試了均勻精度，也沒量品質。「精度依位置變、而且跟 b 綁在一起」以及「在 Cake 設定下量品質」都沒人做。
  - **關鍵提醒**：Cake 的賣點是會合點會自己適應〔原文 p.5〕。如果 D6 在寫入時就決定「前段不存（因為會被重算）」，那等到 GPU 很忙、需要多載一點時，前段就沒得載了。**位置相依的寫入決定會吃掉 Cake 的適應性**，延後版（全存 BF16、讀的時候再決定）天生保有這個彈性。
  - 正面副作用：Cake 一定從第一個 chunk 開始重算，所以 attention sink（第一個 token）永遠是精確的 BF16。KVQuant 量到 sink 對量化特別敏感（見 D6_KVQuant）。這算是 Cake 白送的保護。

- 證據等級：A（方法與 §5.5 兩個版本都讀過；品質相關部分以全文搜尋確認「沒有」）。
