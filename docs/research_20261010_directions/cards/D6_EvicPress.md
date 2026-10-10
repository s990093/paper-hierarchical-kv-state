# D6_EvicPress EvicPress: Joint KV-Cache Compression and Eviction for Efficient LLM Serving

- 出處：Shaoting Feng, Yuhan Liu, Hanchen Li, … Junchen Jiang（芝加哥大學、Tensormesh 等），arXiv 2512.14946v1（2025-12-16），venue 未查證。https://arxiv.org/abs/2512.14946 。讀了 p.1–12（正文全部）；附錄 p.13–16 只掃過。注意：結論段把系統叫成「CacheServe」〔原文 p.12〕，跟標題名稱不一致。

- 精度決定：
  - **做什麼**：每個 context 選一組（存哪一層、用哪種壓縮法、壓縮比）〔原文 p.2〕。
  - **何時**：新的 KV 算出來後，先完整存到假設容量無限的遠端磁碟〔原文 p.5〕；存進 CPU DRAM 後，manage 函式決定放哪一層〔原文 p.7〕。**某一層滿了**就要更新那層上的 KV：要嘛壓得更兇，要嘛降到下一層，用貪婪法解 multi-choice knapsack〔原文 p.7〕。另外定期 re-profile〔原文 p.5、p.7〕。所以主要是**降級／逐出時**做決定，屬於延後型。
  - **粒度**：每個 context。原文明確拒絕 chunk 粒度和 token 粒度，理由是 context 粒度才看得到整段的重要 token 分布〔原文 p.6〕。
  - **用什麼資訊**：utility = (α·quality − TTFT)·frequency；quality 由 GPT-5 產生的離線問題集 profile 出來〔原文 p.6〕。
  - **注意**：實際評估的壓縮法只有三種丟 token 的方法（keydiff、knorm、snapkv），**沒有量化**；量化列為未來工作〔原文 p.12〕。

- 有沒有和重算／載入並行結合？精度跟位置有關嗎？
  - 沒有並行。重算只是一個 baseline（「Prefill」，用 vLLM v0.11.2 全部重算）〔原文 p.8〕。
  - 精度跟 token 位置無關，是整個 context 一個設定。只有「放哪一層」會影響壓縮比：磁碟上的 KV 平均壓得比 CPU 上的兇，因為磁碟讀得慢〔原文 p.10〕。

- 反量化／解碼成本：沒有量。只提到 CacheGen、KIVI 這類方法有解壓開銷，H2O 可以直接用〔原文 p.3〕。硬體為 1×H100 80GB、80 GB DRAM、800 GB SSD〔原文 p.8〕。

- 品質與位置的關係：沒有量位置。量的是 **context 之間**敏感度差很多（同一資料集內變異係數 0.078–0.394）〔原文 p.4〕。

- 有沒有和延後版比？
  - EvicPress **本身就是延後版**：滿了才壓、才降，而且遠端磁碟留有完整副本〔原文 p.5〕。
  - 比較的 baseline 是「所有 context 用同一種壓縮法和壓縮比 + LRU 逐出」〔原文 p.8〕。這同時改了「何時壓」和「每個 context 是否自適應」兩件事，所以**不是乾淨的延後 vs 寫入時對照**。

- 和 D6 的關係〔判讀〕：
  - 它就是任務描述裡說的延後版（先存全精度，降級時才壓）。D6 任何「寫入時就決定精度」的想法都要跟它比。
  - 它的粒度是 context、沒有位置、沒有量化、沒有重算並行，所以「依位置、跟 Cake 的 b 綁在一起的精度」不在它的設計空間裡。
  - 它保留完整副本，代表「寫入時不可逆」的問題它用儲存空間換掉了（遠端磁碟假設無限〔原文 p.5〕）。如果 Tiara 的 SSD 不是無限，這個前提要重新檢查。

- 證據等級：A（正文全部讀過）。
