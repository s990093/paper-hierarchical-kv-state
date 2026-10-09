# A_Strata_補充 Strata: Hierarchical Context Caching for Long Context Language Model Serving（補充卡）

> Strata 已在 `03_paper_map.md` A5（E04）與 `02_sota_write_techniques.md`。這張卡**不重做**，只補和 **H0（hold 是不是真的）** 與 **H10（寫入布局）** 直接相關的原文段落。

- **出處**：Zhiqiang Xie（Stanford／NVIDIA）, Ziyi Xu（SJTU）, Mark Zhao（CU Boulder）, Yuwei An（CMU）, Vikram Sharma Mailthody, Scott Mahlke, Michael Garland（NVIDIA）, Christos Kozyrakis 等。**OSDI 2026**（PDF p.1 是 USENIX OSDI '26 封面，July 13–15, 2026）。本機 PDF `04_Strata_OSDI26.pdf`，這次讀了 p.7–9、p.13（PDF 實體頁）。arXiv 版 2508.18572 當時標「under peer review」。

## 補充 1：三種寫入策略的取捨，原文自己的話（H0）
- 「寫入每個生成的 KV page 能最大化重用與持久性，但對一次性的請求會浪費 host 記憶體與寫入頻寬。**只在淘汰時寫入可以把寫入流量降到最低，但在 GPU 記憶體有壓力時，可能把阻塞的工作加到關鍵路徑上**」〔原文 p.9〕。
- write-back「只在 KV 即將被淘汰時備份……適合資源受限的環境，但可能帶來額外的執行期阻塞」；write-through「每次產生新 KV 就備份，適合要全部持久保存的對話」；selective-write-through 是**預設**，radix 節點的存取次數超過門檻才備份，寫入頻寬充足時門檻是 2〔原文 p.9〕。各層預設都是 LRU〔原文 p.9〕。
- 〔判讀〕這是 H0 的文字證據：一個 OSDI 系統的作者明說 write-back（延後寫）會把阻塞放到關鍵路徑上。但原文沒有給三種策略的量化比較（至少 p.9–13 沒看到）。Screen 的 H0 原始碼結論（`../H0_offload_blocking.md` §0：SGLang 的 write_back 會在排程執行緒上 `synchronize()` 等 GPU→host 拷貝）和這段一致。

## 補充 2：寫入布局在寫入路徑上順便轉換（H10）
- GPU 裡用 layer-first 布局（對計算友善），但一個邏輯 page（一個 token 所有層的 K/V）會被拆成 L 個不連續、各只有幾 KB 的片段，對傳輸最不利；page-first 布局把一個 page 的所有層放在一起，傳輸效率高〔原文 p.7〕。
- Strata 用 GPU 協助 I/O 在搬運時順便轉換：每個 thread 對自己的 offset 多做一次算術就能算出目的地址，「開銷可忽略」；GPU 保持 layer-first，host 記憶體與外部儲存用 page-first〔原文 p.7〕。
- 效果：在 H20-storage 平台、DeepSeek-V3 上，page-first 布局讓平均 TTFT 快 2.1 倍、吞吐高 1.3 倍（相對於「已經相當大的 page size」）〔原文 p.13 §5.3.5〕。因為 H200 平台的磁碟頻寬 <1 GiB/s，這組實驗另外在 H20-storage 上做〔原文 p.13〕。
- 〔判讀〕對 H10：
  - **支持**「寫入布局影響讀取」：2.1 倍 TTFT〔原文 p.13〕。
  - **但威脅「必須在寫入時決定」**：Strata 的轉換是搬運時免費做的，等於寫入時版本已經是現有系統的標準做法；而 [A_IMPRESS](A_IMPRESS.md) 顯示布局也可以在背景重排。H10 剩下的空間只有「為 Cake 從後面讀而反序排」，而我們的 harness 是每層每 chunk ≥2 MiB 的大 I/O，Strata 的問題（每片幾 KB）在我們這裡可能不存在。10 §4 H10 的停損測法（fio 量 2 MiB 以上的差距）正好可以確認。

## 和本研究的關係
- **H0**：支持（文字層面）〔原文 p.9〕。
- **H10**：支持「布局重要」，威脅「寫入時才能做」〔判讀〕。
- **證據等級**：〔原文 p.X〕；判讀如標示。
