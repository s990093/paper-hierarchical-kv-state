# A_Marconi_補充 Marconi: Prefix Caching for the Era of Hybrid LLMs（補充卡）

> Marconi 已在 `os_mapping_20261008/03_paper_map.md` A1（評測卡 E05）。這張卡**不重做**，只補 03 沒寫到、而和 **H7（SSM 檢查點）** 直接相關的三點。

- **出處**：Rui Pan, Zhuang Wang, Zhen Jia, Can Karakus, Luca Zancato, Tri Dao 等。MLSys 2025（arXiv 2411.19379v3 的 comment 寫「MLSys 2025 camera-ready version」）。本機 PDF `marconi2025_arXiv2411.19379.pdf`，這次讀了 p.1、p.5–6、p.8（PDF 實體頁）。

## 補充 1：檢查點在 prefill 當下才拿得到，延後要重跑（N1 的直接證據）
- SSM 狀態原地更新，序列結尾的狀態無法回推到 prefix〔原文 p.1〕。
- Marconi 在 prefill **之前**做一次 speculative insertion（把輸入 token 試插進 radix tree），看會不會產生新的中間節點；會的話就在 prefill 時把那個分支點的狀態存下來〔原文 p.5〕。
- 取得中間狀態的兩種方法：有 chunked state passing 的模型，存「倒數第二個 chunk」的狀態（例：要 token 80，chunk 32，就存 token 64 的）；沒有的模型（Mamba1、Jamba）做**兩段 prefill**（先跑前 80 個 token 拿狀態，再從那裡跑剩下的 20 個）〔原文 p.5–6〕。
- 代價：「純輸入」的 prefix（system prompt、few-shot、長文件 QA）要到**第 3 次出現**才開始受益，因為第 2 次出現才被認出來並存狀態〔原文 p.6〕。〔判讀〕這等於量化了「延後版」的代價：第一次寫入時錯過的狀態，要等下一次 prefill 才補得回來。
- 每個序列最多只准入兩個 SSM 狀態（分支點＋最後一個 decode token）〔原文 p.6〕。

## 補充 2：和「每 block 都存」的對照（寫入時「存哪些位置」的現成比較）
- 對照組 vLLM+：每個 token block 都存一個狀態，block 32（vLLM 支援的最大值）〔原文 p.8〕。SGLang+：用 Marconi 的准入、但 LRU 淘汰〔原文 p.8〕。
- 摘要：token hit rate 最多高 34.4 倍（P95 TTFT 最多少 71.1%／617 ms）〔原文 p.1〕。
- 原文 p.1 指出「為了任意未來負載的重用，要每 256 token 存一個細粒度檢查點」，但這會讓快取充滿低命中的大條目〔原文 p.1〕。

## 補充 3：只有單層（GPU）
- 實驗在 AWS p4d.24xlarge（8×A100-40GB、1,152 GB DDR4）；主結果是 7B 混合模型（{4,24,28} 層 {Attention, SSM, MLP}），TTFT 用 Jamba-1.5-Mini（12B active／52B total）在 4×A100-40GB 上〔原文 p.8〕。
- 〔判讀〕快取層是 GPU 記憶體（原文 p.3 提到可以 provision GPU/CPU 記憶體區塊，但沒有多層放置策略）。**「檢查點放 CPU 還是 SSD」沒有人在這篇做**。

## 和本研究的關係
- **H7**：Marconi（存哪些條目）＋ Sparse Prefix Caching（條目內存哪些位置，見 [A_SparsePrefixCaching](A_SparsePrefixCaching.md)）已經把「寫入時決定存哪些 SSM 狀態」做了兩半。H7 剩下能做的是「多層」與「和 Cake 還原的互動」〔判讀〕。
- 但要注意 [A_SparsePrefixCaching](A_SparsePrefixCaching.md) 卡裡的判讀：整段 session 還原（chat、doc 續問）只需要最後一個狀態，Cake 不需要會合點的 SSM 狀態。H7 只在分叉型負載才有 N1。
- **證據等級**：〔原文 p.X〕；H7 的意義〔判讀〕。
