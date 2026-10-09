# A_IMPRESS IMPRESS: An Importance-Informed Multi-Tier Prefix KV Storage System for Large Language Model Inference

- **出處**：Weijian Chen, Shuibing He, Haoyang Qu, Ruidong Zhang, Siling Yang, Ping Chen（Zhejiang University）, Yi Zheng, Baoxing Huai（Huawei Cloud）, Gang Chen（ZJU）。**FAST 2025**（USENIX，pp.187–201；PDF p.1 是會議封面）。<https://www.usenix.org/conference/fast25/presentation/chen-weijian-impress>。讀了 p.1–6、p.9–13（PDF 實體頁；p.1 是封面，正文從 p.2 開始）。
- **補充說明**：`related_papers_update.md` §c 提到它在評測卡 E05，但 `03_paper_map.md` 沒收。本卡是第一次在這批文件裡讀全文。
- **寫入時做了什麼決定**
  - **布局（但不是寫入時做）**：KV reordering——依 token 重要度把一個 radix 節點內的 token 重新排序、重新打包成 chunk，讓重要的 KV 集中在少數 chunk；**定期（例：每 10 分鐘）在背景非同步做**，不在關鍵路徑上〔原文 p.9〕。只在節點內重排，不跨節點，以免破壞 radix tree〔原文 p.9〕。
  - **格式（寫入時多存一份）**：為了只讀 probe head 的 key 來判斷重要度，把 probe head 的 key 另外冗餘存一份，占全部 prefix KV 的 1.2%〔原文 p.13〕。
  - **放哪（存入／淘汰時）**：score-based admission——每個 chunk 的分數＝存取頻率 × 重要 token 比例，高分放 GPU、低分放 CPU〔原文 p.10〕。
- **用什麼資訊做決定？寫完之後還在不在？（N1）**：重要度是用 probe head 的 key 和 query 算的，跨 head 的重要 token 集合高度相似〔原文 p.7〕。key 存在磁碟上，事後仍可算，**不是 N1**。
- **有沒有和延後版、寫穿版、背景版比較？**
  - 對照組：ReComp（全部重算）、AS-like（重做的 AttentionStore：非同步存、非同步載入全部 prefix KV，加 GPU LRU 快取）、AS+H2O+LRU、AS+H2O+LFU〔原文 p.11〕。
  - TTFT 最多快 2.8 倍〔原文 p.2 摘要〕。消融：KV reordering 讓載入的 chunk 平均少 1.2 倍；score-based 快取讓 GPU 命中率從 68% 升到 80%〔原文 p.12〕。reordering 整個實驗的執行時間不到 1 分鐘，在關鍵路徑外〔原文 p.13〕。
  - 沒有「寫入時就排好」對「背景重排」的比較〔判讀，原文只做背景版〕。
- **硬體**：1×A100 80GB、2×EPYC 7763、128 GB DRAM、2 TB Intel SSD（讀約 5 GB/s）、PCIe 4.0 x16；GPU 快取 10 GB、CPU 快取 32 GB〔原文 p.10–11〕。
- **模型架構、模態**：OPT-6.7B／13B／30B（MHA）；文字，PIQA、RTE、COPA、OpenBookQA 加 few-shot 當共享 prefix，prefix 平均 4.8K–5.7K token〔原文 p.10–11〕。**有損**（只載重要的 KV），保留比例 50%→5%〔原文 p.11〕。
- **和本研究的關係**
  - **威脅 H10（寫入布局）**：IMPRESS 證明「布局」可以**事後在背景重排**，而且不在關鍵路徑上〔原文 p.9、p.13〕。所以布局不是結構上只能在寫入時決定的東西；延後版（背景重排）存在，代價是多一次讀寫〔判讀〕。H10 若要成立，要說明背景重排在我們的情境為什麼不行（例：沒有空閒時間＝N3，或 SSD 寫入額度＝N2）。
  - **支持 N4 的一個實例**：多存一份 probe head 的 key（1.2%），讀取時才能便宜地判斷要載哪些〔原文 p.13〕——寫入形式決定讀取時能做什麼。
  - **H4（一寫多讀）**：它的 admission 用「存取頻率 × 重要度」，頻率要先看到存取才知道〔原文 p.10〕。這是 H4 說的「頻率感知要先看到幾次讀取」的現成做法〔判讀〕。
- **證據等級**：〔原文 p.X〕；H10／H4 的判斷〔判讀〕。
