# D8 卡片：Efficient Remote KV Cache Reuse with GPU-native Video Codec (KVCodec)

- **連結**：https://arxiv.org/abs/2602.09725
- **venue／年份**：SIGCOMM 2026（arXiv comment「Accepted by SIGCOMM 2026」）
- **讀了哪裡**：PDF p.1–5（本機 d8_20261010/kvcodec2026_arXiv2602.09725.pdf）
- **三行摘要**：
  1. 遠端 KV 重用只有在「抓取比重算快」時才划算；為了省錢，現代服務多跑在「幾十 Gbps 或更低」的網路上，租用儲存伺服器在 AWS 上受限於 19 Gbps〔原文 p.3〕（≈2.2 GiB/s，落在 Cake 的甜蜜區〔算術〕）。
  2. 用 GPU 內建的影片編解碼器做無損壓縮（跳過 DCT 與量化），約 10 倍壓縮；pipeline 化抓取、解碼、還原，宣稱能掩蓋網路抖動〔原文 p.1–2〕。
  3. 原文指出 Mooncake 的逐層抓取「沒有處理網路抖動的機制」〔原文 p.5〕。TTFT 最多快 3.51 倍〔摘要〕。
- **和 D8 的關係**：方向 3：壓縮讓頻寬變大（和 Cake 正交，可以疊加）；也說明「頻寬受限＋抖動」是 2026 年的真問題。
