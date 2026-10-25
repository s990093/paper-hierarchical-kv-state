# M05：依注意力頭還原——檢索頭全還原、串流頭只還原 sink＋最近的窗口

> **性質**：判準登記（事先寫死）。來源：[mechanisms/02 M05](../mechanisms/02_bytes_quality.md)（定位：A／M01 的「乘數」，不當貢獻）。使用者 10/25 選「D0＋M05」。
> **品質（不量時間）**：**併在 A0 的 L128 生成裡跑**（同一批 288 題、同一份 prefill；`code/s3/a0_gpu.py` 的 `kind=duo` 配置），只在 L128。

## 1. 頭表與設定
- 頭表：DuoAttention 釋出的 `attn_patterns/Meta-Llama-3.1-8B-Instruct/lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10/`（GitHub `mit-han-lab/duo-attention` commit `53b606ce`，MIT 授權），存在 `results/s3_h200/m05/duo_patterns/`（`config.json` sha256 `b0babbb2…`、`full_attention_heads.tsv` sha256 `9e615b9a…`）：32 層 × 8 個 KV head 的分數。
- **Qwen2.5-7B-1M、Qwen-14B：NOT_MEASURED**（沒有釋出的頭表；自己算要做 DuoAttention 的門控最佳化＝訓練，CLAUDE.md §8：oracle 之前不訓練）。
- 串流頭＝分數最低的 p 比例（p ∈ {0.25, 0.5, 0.75}；平手照（層, head）順序）。串流頭對 context 只看 [0, 128)（sink，頭表 config 的 sink_size）與 [ctx_end−256, ctx_end)（recent_size），問題與生成的 token 照常看得到；檢索頭看全部。所有 KV 都是 BF16（不壓縮）。實作：每（層, KV head）的 key_keep（KVModel 的遮罩路徑）。
- 位元組（區內）：f＝(1−p)＋p×256／區內 token 數（p＝0.25／0.5／0.75 → 0.751／0.501／0.252）。
## 2. 判準（事先寫死）
- 每個 p：生成答對率對 E0 的配對差（288 題，bootstrap 10,000 次）95% CI 下界 ≥ −2 個百分點 → 那個 p **撐住**。
- **M05-PASS**：p＝0.25 撐住（可以當 A 的乘數；報最大撐住的 p）；**M05-STOP**：p＝0.25 也撐不住（M05 卡：「CI 下界 < −2 → 停」）。
- 另報：「找答案」（T2–T4）與「彙整」（T5）分開、單輪與第 2 輪分開；和 A0 的同位元組均勻壓縮對照（只報告）。

## 登記
- 登記時間（`date -u`）：**2026-10-25T13:34:56Z**；commit 訊息開頭 `docs(s3-m05): prereg`。登記時 A0 的生成還沒開始（L128 只有 TF 在跑）；沒有任何 M05 的數字。
