# 07　這輪的快速查新（摘要級）

> **日期**：2026-10-24。**深度**：網頁搜尋結果與 arXiv 摘要頁，**沒有讀全文、沒有標頁碼**。和 `lit/L8a–L8e`（全文）不同等級。
> 正式寫進論文或給老師的新意主張之前，每一條都要照 memory `paper-claims-read-original-double-check` 讀全文、再過獨立 verifier。
> 「沒查到」只代表這輪沒找到（CLAUDE.md 規則 7）。回到核心頁：[README](README.md)。

---

## 1. 查到、而且會改變判斷的

| 論文 | 出處 | 這輪讀到什麼 | 影響哪張卡 | 影響 |
|:--|:--|:--|:--|:--|
| **DualPath** | [arXiv 2602.21548](https://arxiv.org/abs/2602.21548)（2026-02-25） | P/D 分離、agentic 多輪：prefill 端儲存網卡滿、decode 端閒；新增「儲存 → decode 端 → RDMA → prefill 端」的路，全域排程平衡兩條路；離線吞吐最多 1.87 倍、線上平均 1.96 倍（摘要） | M06 | 「借閒著的路」在叢集層級已經有人做；M06 的新意降為「同一台機器內的 PCIe 上行、主機記憶體層」 |
| **PPD** | [ICML 2026 poster](https://icml.cc/virtual/2026/poster/64036) | 多輪時把第 2 輪以後的請求依情況派到 decode 節點做 append prefill；第 2 輪以後 TTFT 少 68%（搜尋結果摘要） | R1 | 「回來的請求在 decode 節點做 prefill」已被做掉 → 否決 |
| NVIDIA Dynamo conditional disaggregation | [Dynamo 文件](https://docs.nvidia.com/dynamo/latest/architecture/disagg_serving.html) | prefill 短或 decode 端前綴命中高時，就在 decode 引擎本地 prefill | R1 | 同上（業界作法） |
| **SparseX** | [arXiv 2606.01751](https://arxiv.org/pdf/2606.01751)（2026-06） | 非前綴片段重用：Sparse-Q 挑要修正的 token，一次前向做 Sparse-KV 重算；前幾層完整注意力、後面稀疏（搜尋結果） | M12 | 最接近「近似重算」；但它稀疏的是「哪些 token 要重算」，不是用稀疏注意力重算整段前綴，也沒有後補精 |
| **Lynx** | [arXiv 2607.01831](https://arxiv.org/pdf/2607.01831) | KV 還沒傳完就開始 decode（P/D 傳輸），主張一般線性量化太粗、要逐 channel | M15 | M15 只是把 Lynx 搬到慢層 → 不建議主做 |
| Tzachristas, certified top-k | [arXiv 2512.07647](https://arxiv.org/pdf/2512.07647) | top-k 截斷的 total variation 誤差有確定性上界；提出誤差預算滿足就停的挑選演算法；實驗少掃 2–4 倍的 key（搜尋結果） | M03 | 「有證書的挑選」在 decode 已有雛形；M03 的差別是粗底＋殘差範數、讀回邊界 |
| RaBitQCache | [emergentmind 2606.31519](https://www.emergentmind.com/papers/2606.31519) | 隨機旋轉二值量化估注意力權重，宣稱無偏且有誤差上界，用在 top-p（只看到摘要，未核對） | M03 | 同上 |
| BitDecoding | [arXiv 2503.18773](https://arxiv.org/html/2503.18773v3) | 低位元 KV 的 decode：tensor core 做矩陣乘、CUDA core 做反量化（搜尋結果） | M08 | 融合的是**同一個請求**自己的解壓；不是別的 session 的還原 → M08 仍有空間 |
| ECHO | [OSDI'26](https://www.usenix.org/system/files/osdi26-liu-guangda.pdf) | 原生稀疏注意力模型：利用 index 分數的可預測性預取 KV；decode 時和 indexer 重疊，prefill 時和逐 query 處理重疊（搜尋結果） | M02（原生稀疏模型時） | 在原生稀疏模型上，「prefill 時邊算邊預取」已有人做；M02 的對象是 dense 模型＋粗底精修 |
| Beluga | [arXiv 2511.20172](https://arxiv.org/pdf/2511.20172) | CXL 記憶體做 KV 管理；量到跨 PCIe switch 的 GPU–NIC P2P 寫 33、讀 23 GB/s，瓶頸在 CPU root complex（搜尋結果） | M06 | 風險警告：PCIe 頻寬不能直接相加 |

## 2. 查到、但不改變判斷的

| 論文 | 出處 | 讀到什麼 | 卡 |
|:--|:--|:--|:--|
| Harvest | [arXiv 2602.00328](https://arxiv.org/abs/2602.00328)（2026-01-30） | 把鄰卡沒用到的 HBM 當暫時的快取層，放權重與 KV；吞吐超過 2 倍（摘要） | M06（借的是容量，不是 PCIe） |
| ObjectCache | [arXiv 2605.22850](https://arxiv.org/pdf/2605.22850) | 在 A100、H100 量不同大小的 pinned H2D 吞吐（搜尋結果） | M02（小傳輸的效率可參考） |
| NOSA | [arXiv 2510.13602](https://arxiv.org/html/2510.13602v1) | 可訓練、為卸載設計的稀疏注意力；CPU↔GPU 傳輸仍是主要成本 | M01（decode 時的對照） |
| QuantSpec | [arXiv 2502.10424](https://arxiv.org/html/2502.10424v1) | 4-bit KV＋4-bit 權重當草稿的自我推測解碼，接受率 >90%（搜尋結果） | M15 |
| LiveMind | [arXiv 2406.14319](https://arxiv.org/html/2406.14319v2) | 使用者還在輸入時就對部分輸入推論 | M04（用在算答案，不是挑 KV） |
| Stream2LLM | [MLSys 2026](https://proceedings.mlsys.org/paper_files/paper/2026/file/3b3889d313ba9476c12c2d77ea66b24f-Paper-Conference.pdf) | 檢索到的 context 串流進 prefill，處理 context 修改時的快取失效 | M04 |
| HUOZIIME | [arXiv 2604.14159](https://arxiv.org/pdf/2604.14159) | 裝置端輸入法，KV 存成壓縮前綴樹，打字修改時重用 | M04 |
| DuoAttention | [arXiv 2410.10819](https://arxiv.org/abs/2410.10819) | 檢索頭／串流頭；GQA 記憶體少 1.67 倍、MHA 2.55 倍（摘要）；搜尋沒找到「只卸載檢索頭」的論文 | M05 |
| CacheOPT（Mitigating KV Cache Competition） | [arXiv 2503.13773](https://arxiv.org/pdf/2503.13773) | 搶佔時在 swap 與重算之間選延遲較短的（搜尋結果的描述，未核對） | M13（選一個，不是賽跑） |

## 3. 搜了但沒找到的（＝這輪的空白，不代表不存在）

| 搜什麼 | 結果 | 卡 |
|:--|:--|:--|
| 暫停的 session 在 GPU 留壓縮副本，回來時從 CPU 補全精度的塊 | 只找到一般的卸載文件（BentoML、Ray、Dynamo KVBM 的動機頁） | M01 |
| 同一台機器內，多張卡的 PCIe 一起還原 KV、經 NVLink 轉送 | 只有 DualPath（叢集）、Harvest（容量）、Beluga（警告） | M06 |
| 用部分問題（打字中）挑要讀的 KV 塊 | 沒有 | M04 |
| 搬運和重算賽跑、輸的取消 | 沒有；Cascade 是預算決定、CacheOPT 是二選一 | M13 |
| 把別的 session 的還原計算塞進 decode 的空檔 | 沒有；BitDecoding、KVComp 是自己的解壓 | M08 |

## 4. 這輪讀過的模型設定（HF config.json，10/24）

| 模型 | 層 | q heads | KV heads | hidden | 上限 |
|:--|--:|--:|--:|--:|:--|
| [Qwen2.5-14B-Instruct-1M](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-1M/raw/main/config.json) | 48 | 40 | 8 | 5120 | `max_position_embeddings` 1,010,000；DCA `chunk_size` 262,144、`original_max_position_embeddings` 262,144 |
| [Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407/raw/main/config.json) | 40 | 32 | 8 | 5120（head_dim 128） | 131,072 |
