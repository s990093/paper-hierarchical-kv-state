# B_Checkmate Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization

- **出處**：Paras Jain、Ajay Jain、Aniruddha Nrusimha、Amir Gholami、Pieter Abbeel、Kurt Keutzer、Ion Stoica、Joseph E. Gonzalez（UC Berkeley），MLSys 2020（Proceedings of the 3rd Conference on Machine Learning and Systems，Austin）〔原文 p.1〕。
  讀的是 arXiv 1910.02653v3（2020-05-14），讀了 p.1–2、p.9。
- **寫入時做了什麼決定**：forward 時產生的每個 activation，要**留著**還是**丟掉、backward 時再重算**（rematerialization）。用 MILP 求最佳排程，或用兩階段 LP rounding 求近似解，**在訓練之前離線算好**〔原文 p.1〕。
- **用什麼資訊做決定？寫完之後還在不在？（N1）**：用靜態的計算圖，加上依加速器 profile 出來的每個運算的成本〔原文 p.1〕。這些資訊事先就有，之後也還在。
  - **延後做不到的原因不是資訊，是時間點**〔判讀〕：記憶體峰值發生在 forward 期間，丟不丟要在 activation 產生之後、峰值之前決定。
- **有沒有和延後版、寫穿版、背景版比較？**
  - 只有單一層記憶體，沒有 tier，所以沒有延後搬可比。對照組是其他 rematerialization 啟發式：Chen et al. 的 √n、greedy 等。這些假設每個節點的成本都一樣〔原文 p.1〕。
  - 16 GB V100 預算下的 U-Net：比最好的 baseline（linearized greedy）快 1.20 倍，比 linearized √n 快 1.38 倍〔原文 p.9 圖 5 說明〕。
  - 最多可以用 5.1 倍大的輸入〔原文 p.1〕。
- **硬體**：NVIDIA V100（16 GB）〔原文 p.9〕。
- **模型架構、模態**：CNN（VGG16、MobileNet、U-Net 等），訓練〔原文 p.9〕。
- **和本研究的關係**：
  - **支持「存還是算，要看各段的重算成本」**：重算成本不一樣時，最佳解和「成本都一樣」的啟發式差很多。KV 的重算成本隨位置變，這是 H2、H6 的背景〔判讀〕。
  - **提供一種「延後做不到」的形式**：決定點被資源峰值逼在寫入當下（N3 的一種）。KV 的對應〔判讀〕：GPU 記憶體在 prefill 期間是瓶頸時，「這段 KV 要不要留、留什麼格式」只能在 prefill 時決定（H6、H8）。但本研究現在的 S5 決定的是 CPU 或 SSD，不受 GPU 峰值限制，所以不直接適用。
- **證據等級**：〔原文 p.X〕；和 H2／H6／H8 的對應〔判讀〕。
