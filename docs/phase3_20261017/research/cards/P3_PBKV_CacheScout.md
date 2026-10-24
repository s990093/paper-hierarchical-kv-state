# P3 卡：PBKV 與 CacheScout（agent 工作流的預測式逐出＋預取）

## PBKV：Efficient Serving for Dynamic Agent Workflows with Prediction-based KV-Cache Management
- **出處**：Haoyu Zheng, Fangcheng Fu, Jia Wu, Binhang Yuan 等。arXiv 2605.06472v1（2026-05-07），venue〔未查證〕。<https://arxiv.org/abs/2605.06472>
- **讀了哪裡**：PDF p.1–6、附錄 E（p.17–18）、p.19。
- **預測什麼**：每個工作流接下來 K 步會呼叫哪些 agent；輸入是 prefix 的 agent 序列與 prefill embedding〔原文 p.4〕。
- **loss**：**cross entropy**，標籤是接下來 K 個 agent，⟨END⟩ 之後的位置 mask；約 350K 參數〔原文 p.4〕。
- **決策**：階層式逐出——先回收「已退休」（工作流已結束）的 cache，再依 K 步前瞻的重用分數排序，否則退回 LRU；預測終止之後的步數不計入分數〔原文 p.2、p.5〕。
- **預取（保守）**：不為預取趕走 active cache；只用閒置的 GPU 空位；只在純 decode 批次（>90%）時用閒置的 PCIe；排序只看一步〔原文 p.6〕。
- **保證**：證明效能對預測誤差是 Lipschitz 連續〔原文 p.2、p.6；證明在附錄 G，沒讀〕。
- **結果**：動態工作流上比 LRU 快最多 1.85×、命中率最多 2.55×；靜態工作流上比 KVFlow 快最多 1.26×〔原文 p.1〕。全文檢索 oracle／ground truth 沒有結果〔原文〕。
- **附錄 E（激進 vs 保守預取）**：激進版允許為預取趕走 active cache。在無雜訊時預取空間 ρ 增大，命中率先升後降、而且變異隨 ρ 單調增加；只有雜訊 λ＝0 與 10% 時激進版勝過保守版，λ＝20% 被超過，**λ≥30% 時比完全不預取（PBKV-HE）還差**；保守版在所有雜訊下都勝過不預取〔原文 p.17–18〕。

## CacheScout：Learning Agent Execution for KV-Cache Management in Agentic Serving
- **出處**：Rui Zhang, Chaeeun Kim, Shaoting Feng, Kuntai Du, Yuhan Liu 等。arXiv 2608.14624（2026-07-16），venue〔未查證〕。<https://arxiv.org/abs/2608.14624>
- **讀了哪裡**：PDF p.2、p.6–9。
- **預測什麼**：K 個排程步內會不會再用到某個 agent anchor 的「存活機率」p_surv＝Pr(∃k∈[1,K]: A_{t+k}＝a | A_t＝a₀)〔原文 p.7〕。
- **怎麼學**：線上轉移矩陣（每次 dispatch 加一個計數），不離線訓練、不需要工作流圖；修剪 P(b|a)<τ（0.5）的邊，用 BFS 跳數近似存活機率〔原文 p.6–7〕。
- **決策**：Score(b)＝(p̃_surv＋δ)·e^{−λ·age(b)}＋δ·|b|，低分先逐出；預測變平時近期項主導，「自然退回 LRU」；δ 讓新 block 在冷啟動時不會馬上被踢〔原文 p.7〕。
- **預取**：送一個只含預測 agent 的系統提示與工具定義的暖機請求（**用重算建 KV**，不是從下層搬）〔原文 p.7〕。
- **結果**：和 vLLM、Continuum（TTL 設 0.3 s）比；命中率 +10–18 個百分點；消融：預測式逐出單獨 +18–22 點，**預取單獨最多 +1 點**，因為中位數的回合沒有可暖機的 prefix，暖機算出的 block 又會在 agent 回來前被 LRU 踢掉；但和預測式逐出一起用時，GAIA 上每回合延遲比只有逐出再降 28%〔原文 p.2、p.9–10〕。

## 和第三階段的關係〔判讀〕
- 兩篇都把「預測的可信度」做成連續的退路：PBKV 用階層式（先回收確定沒用的，再排序，再 LRU）；CacheScout 用乘法混合（預測變平就是 LRU）。比偵測誤判再切換（LARU）簡單。
- 預取的教訓一致：PBKV 的激進預取在中度雜訊下反噬，CacheScout 的預取幾乎沒用。D3 預取要照 PBKV 的保守原則，並在 P3-3 單獨量它值多少。
- 兩篇的「回來」都是 agent 步數，不是秒數，也不跨使用者的人為停頓；和我們的使用者回合尺度不同。
