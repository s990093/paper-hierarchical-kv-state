# A_DualBlade DUAL-BLADE: Dual-Path NVMe-Direct KV-Cache Offloading for Edge LLM Inference

- **出處**：Bodon Jeong, Hongsu Byun, Youngjae Kim, Weikuan Yu, Kyungkeun Lee, Jihoon Yang。arXiv 2604.26557v1（2026-04-29）；arXiv comment：「To appear in IEEE ICDCS 2026」（未在 IEEE 核對）。<https://arxiv.org/abs/2604.26557>。讀了 p.1–3、p.5、p.7（PDF 實體頁）。
- **寫入時做了什麼決定**：**走哪條寫入路徑**——每層的 K/V tensor 在配置時就綁定到 page-cache 路徑（檔案系統＋mmap）或 NVMe-direct 路徑（繞過檔案系統、直接對應到連續的 LBA 區段）〔原文 p.1 摘要、p.5〕。
  - 演算法 1 用一個旋鈕 X（允許進 page cache 的總位元組）把第 1 到 n₁ 層放 Group 1（page cache），其餘放 Group 2（NVMe-direct）〔原文 p.5〕。原文說目前對所有 tensor 一視同仁，但設計可以接排序器，讓重要的才走 page cache〔原文 p.5〕。
  - X 依**執行時的記憶體可用量**動態決定〔原文 p.1 摘要〕。
- **用什麼資訊做決定？寫完之後還在不在？（N1）**：用的是當下的 host 記憶體可用量，不是 N1。但**一旦寫進 page cache，就占著 DRAM**，直到核心決定回寫或淘汰〔原文 p.2–3〕——這是 N4 的作業系統版本：寫入路徑決定之後的記憶體占用與讀取路徑〔判讀〕。
- **有沒有和延後版、寫穿版、背景版比較？**
  - 基線是原版 FlexLLMGen：GPU↔CPU 走 pinned DMA，CPU↔檔案走 mmap，「寫入落在 page cache、非同步刷出」〔原文 p.2〕。
  - **page cache 的問題（實測）**：host 記憶體上限從 11 GB 掃到 2 GB（KV 共 8.57–9.11 GB）。低於約 6 GB 時 prefill 延遲單調上升，原因是「page cache 變小，迫使核心在背景回寫之前就做同步淘汰，寫入 I/O 被卡住」〔原文 p.3〕。decode 的 page cache 命中率在某點從 42% 崩到 <1%，2–7 GB 之間形成 thrashing 區〔原文 p.3〕。原文說這是首次在 LLM KV 卸載上實驗指出 page-cache thrashing〔原文 p.3〕。
  - DUAL-BLADE 讓 prefill 延遲最多少 33.1%、decode 少 42.4%，SSD 使用率最多 2.2 倍〔原文 p.1〕。
- **硬體**：單 GPU：RTX 5060 Ti 16 GB、Intel Core Ultra 7 265K、**16 GB host 記憶體**；兩顆 NVMe：Samsung PM9D3a（PCIe Gen5）、Samsung 990 PRO（Gen4）；Ubuntu 24.04、Linux 6.8、liburing 2.5〔原文 p.7〕。
- **模型架構、模態**：OPT-6.7B（MHA），prompt 512、生成 32、batch 32；只卸載 KV，不卸載權重；文字〔原文 p.7〕。
- **範圍**：單一批次的 decode 卸載（FlexLLMGen），不是跨請求重用；邊緣裝置〔原文 p.2、p.7〕。
- **和本研究的關係**
  - **直接支持 H13 的前提**：用 buffered I/O 寫 SSD 時，資料其實先待在 page cache（占 DRAM），記憶體緊時寫入會被同步淘汰卡住〔原文 p.2–3〕。這就是 10 §3E 的 E1（雙重快取）和 E2（dirty writeback 節奏）。
  - **但也是 H13 的「更簡單的做法」**：解法是改寫入路徑（O_DIRECT／NVMe-direct），和「寫入時依位置決定放哪」無關〔判讀〕。H13 的停損條件「都用 O_DIRECT」若成立，H13 就只剩工程意義。要先查 vLLM／LMCache 的磁碟後端是否 O_DIRECT（H0 的 Screen 文件可能已經看到）。
  - 我們的 MI300X 節點和 7 個鄰居共用記憶體（10 §3E 的 E8），比它的 16 GB 邊緣機更不容易壓到 thrashing 區，所以效果大小不能直接搬〔判讀〕。
- **證據等級**：〔原文 p.X〕；對 H13 的意義〔判讀〕。
