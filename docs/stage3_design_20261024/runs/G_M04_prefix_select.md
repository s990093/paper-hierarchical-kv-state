# G-M04：打字時就挑塊——用「問題的前半段」挑塊，準不準

> **性質**：判準登記（事先寫死）。登記時間見文末。來源：[mechanisms/02 M04](../mechanisms/02_bytes_quality.md)、[mechanisms/06 §2](../mechanisms/06_eval_gates.md)（G-M04）。
> **品質（不量時間）**：併在 A0 的 TF 篩選裡跑（同一題、同一份 prefill 的 KV；`code/s3/a0_gpu.py` 的 `--gm04`），輸出另一個檔 `results/s3_h200/a0/raw/gm04_<cell>_s*.csv.gz`，不影響 A0 的列。

## 1. 量法
- 題目＝A0 的 TF 題組（seed 71000，T2–T5 各 24 題）；判定格＝A0 的 L128、Q128、Q256、Q14-256（32K 對照列也量，只報告）。
- 問題的文字部分：在 `query_ids` 裡找到 `tokenize(question)` 那一段（逐 token 的子序列；找不到 → 那一題記 NOT_MEASURED）。前綴比例 p ∈ {0.25, 0.5, 0.75, 1.0}：只用「問題文字前面的 token（user 開頭）＋問題文字的前 ⌈p·n⌉ 個 token」（還沒按送出：沒有 assistant 開頭與答案前綴）。
- 挑塊＝A0 的 ORACLE（全域）：這些 token 在 BF16 KV 上、對 context key 的 attention 機率加總（A0 登記 §1.4 的同一個式子），取前 m 塊，m＝round(ρ·nreg)，ρ ∈ {1/8, 1/4}。
- 指標：recall_p＝答案所在的塊（T2–T4 的 needle blocks）被挑中的比例；overlap_p＝和 p＝1.0（完整的文字＋assistant 開頭＝A0 的 ORACLE 用的那一串）挑中的集合重疊的比例。
## 2. 判準（事先寫死）
- 每個判定格、ρ＝1/4：R＝平均 recall_{0.5} ÷ 平均 recall_{完整}（完整＝A0 的 ORACLE 用的整串 `query_ids`）。
- **GM04-PASS**：所有有量到的判定格 R ≥ 0.8；**GM04-STOP**：任何一格 R < 0.8 → 只剩「打字時先讀粗底」（M01 本來就有）。
- 完整問題的平均 recall < 0.10 的格子：R 不穩，記 NOT_MEASURED（不算 PASS 也不算 STOP），照實報告。
- 另報：ρ＝1/8、p＝0.25／0.75、overlap、T5（沒有答案位置，只有 overlap）。

## 登記
- 登記時間（`date -u`）：**2026-10-25T12:52:35Z**；commit 訊息開頭 `docs(s3-gm04): prereg`。登記時沒有任何數字（A0 的 TF 還沒開始）。
