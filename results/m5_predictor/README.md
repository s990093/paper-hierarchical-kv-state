# M5 第二階段 — 未來效用預測器（實作 + 訓練）

`EXPERIMENT_PLAN.md` §6 的第二步（第一步是「成本模型 + recency，不訓練」）。
產生這批數字的是 `code/m5_predictor.py`（特徵／標籤／訓練／校準）
與 `code/m5_policy_sim.py`（把訓練好的模型接回 M4 的模擬器）。

**這是 trace 驅動的模擬，不是端到端系統量測。** 特徵只有存取歷史那三族；
`pooled key/value` 與 `attn_mass` 要跑模型才有，此處為 `NOT_AVAILABLE`。
表 15 的 (C) 區塊（特徵集消融）**不能**用這裡的數字填。

## 檔案

| 檔 | 一列是什麼 | 關鍵欄位 |
|---|---|---|
| `samples.csv` | 一次特徵抽取 | `n_samples`、`positive_rate`、機制 (a)/(b) 的計數、`run_dir` |
| `predictor_metrics.csv` | (trace × 損失 × 門檻規則) | `ece`、`auc`、`cost_ms`、`cost_us_per_decision`、`lat_median_us` |
| `calibration_bins.csv` | 可靠度表的一格 | `mean_p_hat` vs `empirical_rate`（§B.6 要求的校準證據） |
| `cost_by_position.csv` | (損失 × 門檻規則 × 位置分箱) | `kappa_median`、`p_star_median`、`cost_ms` |
| `feature_importance.csv` | 一個特徵 | `gain` |
| `policy_sim.csv` | 一個策略在一段 trace 上 | `total_ms`、`vs_best_baseline_pct`、`headroom_captured_pct`、`ssd_write_mibps` |

每一列都有 `run_id`，對應 `/ssd7/hungwei/paper-hkv/runs/<run_id>/`
（特徵矩陣 `X.npy`、標籤 `labels.npz`、模型 `model_*.txt`、校準表 `calib_*.npz`）。

## 這批數字怎麼驗

```bash
PYTHONPATH=/ssd7/hungwei/paper-hkv/pylibs python code/test_m5_predictor.py
python code/m5_policy_sim.py --check-shim     # 記帳與 M4 完全一致才可並列
```

`--check-shim` 是這裡最重要的一道檢查：把預測換成「上次存取時刻」（＝LRU）之後，
`m5_policy_sim` 的迴圈必須**逐位元**重現 `m4_oracle.Sim.run_online("lru")`。
對不上就代表線上策略與 oracle 的比較是在比兩把不同的尺。
