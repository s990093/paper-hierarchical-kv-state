# D6_CacheGen CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving

- 出處：Yuhan Liu, Hanchen Li, Yihua Cheng, … Junchen Jiang（芝加哥大學、Microsoft、Stanford），SIGCOMM 2024。https://arxiv.org/abs/2310.07240（讀的是 v6，2024-07-19）。PDF 存在 `/mlsteam/data/tiara/papers_d6/arXiv2310.07240.pdf`。已讀 p.1–11、p.17–19（附錄 A–E）；p.12–16 是參考文獻，只掃過。

- 精度決定：
  - **寫入時（離線）先把每個 chunk 編成「好幾個等級」**：先算整段 context 的 KV，沿 token 維度切成 chunk，每個 chunk 用多個量化等級各編一份，彼此獨立可解碼〔原文 p.6–7〕。「This can be done offline」〔原文 p.4〕。
  - **讀取時才挑等級**：一個 chunk 一個 chunk 送；用上一個 chunk 的實測 throughput 估頻寬，挑「SLO 內壓縮損失最小」的設定〔原文 p.7〕。演算法見 Algorithm 1〔原文 p.19〕。
  - 粒度：每個 chunk 預設 1.5K tokens〔原文 p.7〕。第一個 chunk 沒頻寬資訊時用中等等級（Llama-7B 每 chunk 140 MB）〔原文 p.7〕。
  - 等級內部：層分三組（前 1/3、中 1/3、後 1/3），量化 bin 依序 0.5／1／1.5（淺層較精）〔原文 p.6、p.19〕；每 10 個 token 一組，組內第 1 個 anchor token 用 8-bit，其餘存 delta〔原文 p.6〕；算術編碼的機率分布按「channel×layer」離線 profile〔原文 p.6〕。
  - 用到的資訊：層的敏感度（淺層較敏感）〔原文 p.5〕、即時頻寬、SLO。

- 有沒有和重算／載入並行結合？精度跟位置有關嗎？
  - **有「每 chunk 退回送文字、讓 GPU 重算」**：每個 chunk 可選某個編碼等級或文字格式〔原文 p.6–7〕。Algorithm 1：若重算時間 ≤ 剩餘時間就送文字，否則挑能在剩餘時間內送完的等級〔原文 p.19〕。
  - **但不是 Cake 式的兩頭並行**：文字 chunk 是「根據前一個已收到並解碼的 chunk 的 KV」來算〔原文 p.7〕，是一條 chunk 序列逐一處理；原文沒有描述「前段重算、後段同時載入」。只寫了「傳 chunk i 的同時解碼 chunk i−1」〔原文 p.8〕。
  - 精度跟位置：**沒有刻意依位置分配**。唯一的位置因素是每 10 token 的 anchor 用 8-bit〔原文 p.6〕。它還量到「依 token 位置分組」的熵下降比依 channel／layer 分組小很多〔原文 p.5〕。chunk 等級只隨頻寬變，不隨 chunk 在序列中的位置變。
  - context 短於約 1K 時會自動退回全送文字〔原文 p.10〕。

- 反量化／解碼成本：
  - 正文**沒給數字**，只說解碼在 GPU 上做、跟傳輸 pipeline 化，「對端到端延遲影響很小」（Fig. 14a，只有圖）〔原文 p.11〕。
  - 編碼延遲約 200 ms（Fig. 14c，Mistral-7B）〔原文 p.11〕。
  - 硬體：NVIDIA A40 ×4 伺服器〔原文 p.9〕。
  - 存多個版本的儲存成本「與量化 baseline 相當」（Fig. 14d）〔原文 p.11〕；8.5K token、Llama-13B 的全部版本約 5 GB〔原文 p.19〕。

- 品質與位置的關係：
  - 量到的是**層**的效應：淺層加損失，準確率掉很多；深層掉很少（Fig. 4）〔原文 p.5〕。
  - 沒有量 token 位置對品質的影響。

- 有沒有和延後版比？
  - 沒有。它的作法本身繞開了「寫入時決定、無法回頭」：寫入時把所有等級都存下來，讀取時才挑〔原文 p.6–7〕。代價是多存幾份〔原文 p.11、p.19〕。

- 和 D6 的關係〔判讀〕：
  - **威脅「頻寬自適應、每 chunk 選精度、可退回重算」的新穎性**：CacheGen 已經是 per-chunk、依頻寬選等級、每 chunk 可改成重算的系統。
  - **支持 D6 的空白點**：(1) 它的重算是序列式的，不是 Cake 式兩頭並行；(2) 等級隨頻寬變，不隨「這個 chunk 離交會點 b 多遠」變；(3) 它用「多存幾個版本」避開了寫入時不可逆的問題，所以「單一版本 + 依位置的精度」這個組合它沒碰。
  - 提醒：KVServe 用同一份 CacheGen 演算法測到相對準確率只有 65.76%（Qwen2.5-7B，見 D6_KVServe）。CacheGen 自己的論文說掉幅不超過 2%〔原文 p.9〕。兩篇的設定不同，品質數字不能直接搬。

- 證據等級：A（全文 PDF，正文與附錄逐頁讀過）。
