# 查新報告：以 Cake 為基底的「寫入時依位置決定」（2026-10-01）

> 上一次查新是 2026-09-24（見 `docs/DIRECTION_CAKE_EXTENSION.md` §6）。這次的重點是 9/24 之後的新論文，以及同時涵蓋「寫入時 × 依位置 × 多層 × 雙向還原」的工作。
>
> **審查方式**：self-review，不是 cross-model。這台機器沒有 Codex MCP；依 CLAUDE.md §5，開 subagent 當審查者需要使用者明確要求，所以這次沒有開。
>
> **論文核實**：表中每一篇都直接抓過 arXiv 摘要頁，確認標題、作者與日期；repo 裡沒有 `verify_papers.py`。「是否涵蓋某點」的判斷依據是摘要，CacheFlow 與 EfficientAgent 另外讀了內文。只讀摘要的條目已標註。

## 提案

prefill 算完 KV 後，逐個 chunk 依位置決定存不存、存哪一層：GPU（BF16／FP8／INT4）、CPU、SSD，或不存。前段重算便宜，不存；後段依重算成本放到對應的層。依據是：Cake 讀取時的會合點，就是寫入時的分界。讀取時沿用 Cake 的雙向還原，改成多層來源，並加上退路。

## 核心主張與最接近的前作

1. **寫入時決定存不存**
    - 最接近：EfficientAgent（arXiv 2609.33762，2026-09-27）。它的 capacity-conditioned write admission 在快取層壓力大時，整個請求的新 KV 都不寫。
    - 差別：EfficientAgent 以「整個請求」為單位，只看工作集壓力。不看位置，不看重算成本，只有 host memory 一層，也不考慮與讀取時的重算重疊〔內文〕。
2. **依位置的重算成本決定快取去留**
    - 最接近：AsymCache（arXiv 2606.02964，2026-06-01）的逐出策略同時考慮命中率與位置相關的重算成本；Pensieve（EuroSys'25）優先保留後段。
    - 差別：兩者都是**逐出**時的決定，而且只有 GPU 或 GPU＋CPU，沒有寫入時決定，也不考慮雙向還原〔AsymCache 依摘要〕。
3. **讀取時重算與載入並行，並決定切點**
    - 最接近：Cake（ICML'25）；CacheFlow（arXiv 2604.25080，2026-09-26 修訂）用 DP 規劃在 token × layer 空間「哪些重算、哪些載入」。
    - 差別：CacheFlow 內文明寫假設 KV 已存好（「load previously materialized KV states from CPU memory, SSDs, or remote machines」），只在讀取時決定，也不改寫入策略〔內文〕。
4. **多層放置（GPU／CPU／SSD）**
    - 最接近：「Where Should the KV Cache Live?」（arXiv 2609.16215，2026-09-14）、KVDrive（arXiv 2605.18071）、Predictive Multi-Tier（arXiv 2604.26968）、Bounded-State Restoration（arXiv 2608.17826）。
    - 差別：這些都依熱度、重用預測或注意力決定放哪層。摘要裡都沒有「不存＋重算」這一態，也沒有依位置的重算成本〔依摘要〕。
5. **寫入分界 ＝ 讀取會合線**
    - 查到的論文裡，沒有一篇把讀取時的會合點拿來決定寫入。CacheFlow 與 Cake 都把會合或切點當成讀取時的排程問題。

## 最接近的前作

| 論文 | 時間 | 出處 | 重疊 | 主要差別 |
|---|---|---|---|---|
| Cake: Compute Or Load KV Cache? Why Not Both? | 2025 | ICML'25 | 讀取時雙向還原 | 假設全存、單層、I/O 模擬 |
| CacheFlow (2604.25080) | 2026-04，09-26 修訂 | arXiv | 讀取時決定重算／載入，staircase 切分 | 只在讀取時；假設 KV 已存好 |
| EfficientAgent (2609.33762) | 2026-09-27 | arXiv | 寫入時決定不寫 | 以請求為單位、看壓力，不看位置；單一 host 層 |
| AsymCache (2606.02964) | 2026-06-01 | arXiv | 位置相關的重算成本 | 用在逐出，不在寫入；無 SSD；無雙向還原 |
| Pensieve (2312.05516) | 2025 | EuroSys'25 | 後段優先保留、前段丟掉再重算 | 逐出時；無 SSD；無寫入決定 |
| py-kvcache (2609.11744) | 2026-09 | arXiv | 依硬體算損益平衡，決定存不存 | 以整段前綴長度判斷，不是 block 位置 |
| Where Should the KV Cache Live? (2609.16215) | 2026-09-14 | arXiv | GPU／CPU／SSD 三層放置 | 依熱度／重用預測；無「不存」；不看位置 |
| KVDrive (2605.18071) | 2026-05-18 | arXiv | 多層＋計算與 I/O 重疊 | decode 階段、依注意力；無寫入時依位置 |
| CacheTune (2605.24022) | 2026-05-20 | arXiv | 多層、硬體感知的重算比例 | 非前綴重用的語意修補；不談寫入 |
| HCache | 2025 | EuroSys'25 | 部分不存、要用時重算 | 以「層」為單位，不是位置 |

## 整體判斷

- **分數**：5／10。有明確的鄰居，但差別可以辯護，值得做 pilot。
- **建議**：**PROCEED WITH CAUTION**。
- **主要差異點**：查到的工作中，沒有一篇在**寫入時**用**依位置的重算成本**，同時決定「不存／哪一層」，並把這個分界跟 Cake 式**讀取會合點**連起來。
- **差別為什麼薄，具體在哪**：
    1. Pensieve／AsymCache 的「前段先逐出」，跑一段時間之後的穩態，可能跟「寫入時就不存前段」幾乎一樣。兩者只差在逐出之前，是否已經浪費了寫入頻寬和暫時的容量。E1 的 P2 對 P4 就是在量這個差距。如果 P2 已經拿走大部分好處，主要貢獻就只剩寫入頻寬。
    2. EfficientAgent（9/27）已經把「寫入時選擇性不寫」做成可運作的系統。我們不能把「寫入時決定」本身當成新穎點，要強調的是「依位置」與「跟讀取會合點連動」。
- **審稿人最可能引的前作**：Pensieve、AsymCache、EfficientAgent、CacheFlow。

## 什麼能讓差別站得住

1. E1 中 P4 相對 P2（Pensieve 式）有實質差距，而不只是相對 P1（Cake）。
2. 有一個 workload 或硬體條件，讓「逐出時才處理」明顯不夠。例如寫入頻寬吃緊、SSD 寫入跟載入搶頻寬（E2）。
3. 「寫入分界 ＝ 讀取會合線」寫成有推導的成本模型，並用實測驗證會合點在擾動下的穩定度（E4）。

## 建議定位（一句話，審稿人可驗證）

> 既有系統在讀取時決定重算與載入的切點（Cake、CacheFlow），或在逐出時考慮位置成本（Pensieve、AsymCache）；本工作把讀取時的切點提前到寫入時使用，逐 block 決定不存或存到哪一層，並量化它相對逐出式做法省下的寫入量與容量。

## 新增的觀察（9/24 之後）

- **CacheFlow 9/26 修訂版**：讀取端已經做到 token × layer 的 DP 規劃。讀取端的改進空間更小了，本方向應該把重心放在寫入端。
- **EfficientAgent 9/27**：寫入准入成為熱門題目。這是一場競賽而不是否決，但定位要更精確。
- **RelaxKV（2609.33503）、Tutti（2605.03375）**：與本方向的重疊低，前者做非前綴重用，後者做 SSD I/O 路徑。列出備查。

## 來源

- https://arxiv.org/abs/2410.03065 （Cake）
- https://arxiv.org/abs/2604.25080 （CacheFlow）
- https://arxiv.org/abs/2609.33762 （EfficientAgent）
- https://arxiv.org/abs/2606.02964 （AsymCache）
- https://arxiv.org/abs/2312.05516 （Pensieve）
- https://arxiv.org/abs/2609.11744 （py-kvcache）
- https://arxiv.org/abs/2609.16215 （Where Should the KV Cache Live?）
- https://arxiv.org/abs/2605.18071 （KVDrive）
- https://arxiv.org/abs/2605.24022 （CacheTune）
- https://arxiv.org/abs/2604.26968 （Predictive Multi-Tier）
- https://arxiv.org/abs/2608.17826 （Bounded-State Restoration）
- https://arxiv.org/abs/2605.03375 （Tutti）
- https://arxiv.org/abs/2609.33503 （RelaxKV）
