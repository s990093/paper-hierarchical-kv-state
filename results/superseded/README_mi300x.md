# 已被取代的平台 B 結果（保留供追溯，不可引用）

| 檔案 | 為什麼被取代 | 取而代之 |
|---|---|---|
| `retrieval_cost_b-qwen7b-1m.csv`、`retrieval_cost_b-qwen3-30b-a3b.csv`、`retrieval_cost_b-seedoss36b.csv` | SSD 階的 CPU 主階小於一個前綴，warm 只讀回一部分（RUNLOG 發現 8）；且部分 CPU 階因 /dev/shm 被塞滿而 NOT_MEASURED（發現 7） | 同名 `_v2.csv` |
| `recompute_position_b-mistral-nemo12b.csv` | 模型對隨機 token 提示立刻吐 EOS，11/27 列沒有 TTFT（已加 `min_tokens` 修正） | `recompute_position_b-mistral-nemo12b_v2.csv` |
| `oracle.csv` | 固定檔名導致 7 個模型互相覆蓋，只剩最後一次的 20 列（違反規則 3） | `oracle_<model>_<workload>.csv` |
