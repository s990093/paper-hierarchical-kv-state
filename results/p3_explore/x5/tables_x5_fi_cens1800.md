# X5 表（`code/p3x/x5_analyze.py`，run 20261017-220105-p3x5-analyze-fi-cens1800；cells run 20261017-215650-p3x5-fi-cens1800）

判定（事先登記，主設定 llama／K0／reg）：**死路**；excess 校正（只報告）：NOT_RUN


## 主設定 headroom — conversation（Llama、原寫法反量化 K0、reg 校正；回來請求完整 TTFT 平均；括號＝session bootstrap 95% CI）

| CPU GiB (占工作集) | 慢層 | GPU | 最好規則 | T_規則 (s) | 最好 oracle | T_oracle (s) | headroom | 只算還原 | 中位數 | p90 | 部分 oracle（只知會不會回來） |
|--:|:--|:--|:--|--:|:--|--:|--:|--:|--:|--:|--:|

## 主設定 headroom — toolagent（Llama、原寫法反量化 K0、reg 校正；回來請求完整 TTFT 平均；括號＝session bootstrap 95% CI）

| CPU GiB (占工作集) | 慢層 | GPU | 最好規則 | T_規則 (s) | 最好 oracle | T_oracle (s) | headroom | 只算還原 | 中位數 | p90 | 部分 oracle（只知會不會回來） |
|--:|:--|:--|:--|--:|:--|--:|--:|--:|--:|--:|--:|

## 全部格子的 headroom 範圍（只報告；variant × 模型 × 反量化 × trace）

| variant | 模型 | 反量化 | trace | 格子 | headroom 最小 | 中位 | 最大 | ≥15% 的格子 | ≥5% 的格子 | 最常見的最好規則 |
|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|:--|
| reg | llama | k0 | freeinference | 24 | 0.0% | 0.0% | 0.6% | 0 | 0 | R6 (24) |

## oracle 比某個規則慢的格子：0

