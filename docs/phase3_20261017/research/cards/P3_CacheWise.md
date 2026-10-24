# P3 卡：CacheWise（coding agent 的預測式 KV 逐出）＋ CATraces

- **出處**：Tiwari, Chugh, Rickert, Peter, Mahajan（UW）、Shen（UVA）。arXiv 2606.16824v1（2026-06-15），venue〔未查證〕。<https://arxiv.org/abs/2606.16824>
- **資料與程式**：<https://github.com/cachewise-project/cachewise-coding-traces>（commit `181c435a`，2026-06-14；GitHub API 的 license 為 `null`）。
- **讀了哪裡**：PDF p.1–12（全文）；repo 的 `parsed_traces/manifest.json`、`project_001/session_0001/events.json`、`tool_duration_prediction/` 的全部 Python〔程式碼〕。

## trace（CATraces）
- 94 個 session、11 個專案、20,634 次 LLM 呼叫、11,617 次工具呼叫〔文件 manifest〕；論文 Table 1 寫約 10M token〔原文 p.3〕。
- 事件欄位：`llm_call`（時間、模型、input／output／cache_creation／cache_read token、是否用工具）、`tool_call`（工具名、參數已遮蔽、`execution_time_ms`、`result_timestamp`）〔文件〕。
- session 時長中位 36 分鐘、尾端 >2.6 小時；工具觸發的請求是使用者觸發的 20 倍（中位）〔原文 p.3〕。
- 不一致：README 指向 `release/raw_traces/`，實際在根目錄 `raw_traces/`；論文說 80% session 拿去訓練〔原文 p.8〕，repo 的 split 是 34／94〔程式碼〕。
- 時間戳看起來被平移（session_0001 是 2020-07-07，但模型是 `claude-opus-4-1-20250805`）〔判讀〕。

## 預測與決策
- **預測什麼**：工具呼叫的剩餘時間 E[τ | τ−T_i > t−T_i]〔原文 p.6–7〕。
- **怎麼學**：不訓練參數。每個工具名一組 TF-IDF（最多 5,000 詞）＋MiniBatchKMeans（每叢集 ≥30 筆、≤1,000 叢集），每叢集存經驗的「仍在執行」曲線；推論時以梯形積分算條件期望；相似度 <0.3 退回整體曲線〔程式碼 `build_clusters.py`、`infer.py`〕。使用者停頓插成 `HumanPause` 假工具（≥2 s、<1 天）〔程式碼 `human_pauses.py`〕。
- **原文的理由**：只需要相對順序；「找出 τ 最大的 session 就夠了」〔原文 p.6〕。
- **決策**：block 引用數歸零時帶著 session 的 metadata 進逐出堆，以 E[τ] 為優先權；預測會過期，所以每 N_rebuild＝3 個引擎迭代重建一次堆〔原文 p.8〕。
- **oracle**：CacheWise\*（真實工具時長）。CacheWise 和它相近，有時略好，原文歸因於排程的連鎖效應〔原文 p.9〕。
- **退路**：無；漂移的穩健性「留給未來」〔原文 p.8〕。

## 結果
- N>10 時 session 完成時間比 vLLM、InferCept 快 2.7–3.5 倍，逐出少 2–2.6 倍〔原文 p.9〕。
- 叢集越細越好：Point Estimate → Tool Name → C20／C50／C100，C100 最好，session 完成時間最多改善 19%〔原文 p.11〕。
- **換成 CPU 卸載（不重算）時只快 1.19 倍**；原文：PCIe 比重算快得多，每次逐出的懲罰較小，空間也較小〔原文 p.11〕。

## 和第三階段的關係〔判讀〕
- 1.19 倍 vs 2.7–3.5 倍是「還原便宜 → 預測價值小」的直接例子，支持 Q1b 的假設方向。
- session 數太少（94），不能當主要學習資料；它的「條件剩餘時間＋定期重建」是 Q3（決定當下重新預測）的現成做法，但原文沒有做「不重建」的對照。
- Ask-the-Tool 重做了它的預測器：只有 HBM 時比 LRU 差 22.7%，加 DRAM 層後比 LRU 好 14.9%（見 P3_AskTheTool 卡）。
