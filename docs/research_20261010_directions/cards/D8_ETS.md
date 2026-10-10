# D8 卡片：ETS: Efficient Tree Search for Inference-Time Scaling

- **連結**：https://arxiv.org/abs/2502.13575
- **venue／年份**：arXiv 2502.13575（2025-02）；venue 未查證
- **讀了哪裡**：arXiv 摘要頁
- **三行摘要**：
  1. 樹搜尋的分支越分散，KV 共享越少、記憶體越大〔摘要〕。
  2. 用線性規劃挑要保留的節點，鼓勵 KV 共享，同時保留語意不同的分支〔摘要〕。
  3. 平均 KV 少 1.8 倍、吞吐高 1.4 倍〔摘要〕。
- **和 D8 的關係**：方向 10：樹搜尋的 KV 壓力已有人用「剪枝」處理，不是用分層。
