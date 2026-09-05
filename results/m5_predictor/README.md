# M5 第二階段 — 未來效用預測器（實作 + 訓練 + 消融）

`EXPERIMENT_PLAN.md` §6 的第二步（第一步是「成本模型 + recency，不訓練」）。
程式：`code/m5_predictor.py`（特徵／標籤／訓練／校準／指標）、
`code/m5_policy_sim.py`（把模型接回 M4 的模擬器）、
`code/m5_summary.py`（判定材料）、`notebooks/m5_figures.py`（四張圖）。

## 🔴 在哪個區間量的（這一條先讀）

第一輪在 **Mooncake／端到端**上量——那裡 oracle headroom 只有 8.49%，
等於在方法自己宣稱沒用的區間檢驗它。第二輪起主結果改在
**論文 §6.8 headroom 最高的那個設定**上量：

| | 設定 | 論文回報 | 本檔重現 |
|---|---|---|---|
| `longctx-zipf` | `surface:L131072:rq10`：12 份 131K 文件、120 請求、tail 2%、α=0.9 | 重用 89.06%、壓力 6.41×、**headroom 34.65%**（prefill-only） | 重用 89.06%、壓力 6.41× ✅ |

另外兩個工作負載：
* `longctx-session`——多輪會談（一份長文件被連續追問，`concurrency` 個會談交錯）。
  128K 級請求下 GPU 只裝得下約兩個請求，**唯一會發生重用的形態就是多輪對話**。
* `mooncake`——真實 trace，保留作為「真實但 headroom 低」的對照。

兩個合成負載都讓**位置 32K–128K（κ 37–60）第一次有正樣本**；
Mooncake 上 8K 以上的位置一個正樣本都沒有，(B) 區塊因此無從檢驗。

## 🔴 這是 trace 驅動的模擬，不是端到端系統量測

特徵只有存取歷史三族（deltas / EDC / 靜態）。
`pooled key/value` 與 `attn_mass` 要跑模型才有，此處 `NOT_AVAILABLE`——
表 15 的 (C) 區塊（完整特徵集消融）**不能**用這裡的數字填，
`cost_by_position.csv` 與 `feature_importance.csv` 給的是**簡化版**。

## 檔案

| 檔 | 一列是什麼 | 關鍵欄位 |
|---|---|---|
| `samples.csv` | 一次特徵抽取 | `workload`、`window_accesses`、`sample_rate`、`seed`、`label_mode`、`positive_rate`、`reuse_rate` |
| `predictor_metrics.csv` | (工作負載 × 損失 × 門檻 × 容量 × 特徵族 × seed) | `auc`、`ece`、**`spearman_positives`**、`cost_ms`、`lat_*`、`calib_split` |
| `calibration_bins.csv` | 可靠度表的一格 | `mean_p_hat` vs `empirical_rate`（§B.6 要求的證據） |
| `cost_by_position.csv` | 位置分箱 | `kappa_median`、`p_star_median`、`cost_ms`（檢驗 §5.3 的可證偽預測） |
| `cross_workload.csv` | 於 A 訓練、於 B 測試 | `trace` → `test_trace` 的退化幅度 |
| `feature_importance.csv` | 一個特徵 | `gain` |
| `policy_sim.csv` | 一個策略在一段 trace 上 | `total_ms`、`vs_best_baseline_pct`、`headroom_captured_pct`、`drop_cost_rule` |

每一列都有 `run_id`，對應 `/ssd7/hungwei/paper-hkv/runs/<run_id>/`
（`X.npy`／`labels.npz`／`model_*.txt`／`calib_*.npz`／`test_pred_*.npz`）。
`features_index.json` 是「設定 → 特徵目錄」的索引。

## 這批數字怎麼驗

```bash
export PYTHONPATH=/ssd7/hungwei/paper-hkv/pylibs
python code/test_m5_predictor.py          # 八組手算單元測試
python code/m5_policy_sim.py --check-shim # 記帳必須逐位元等於 m4_oracle
python code/m5_summary.py                 # 全部判定材料
python notebooks/m5_figures.py            # 四張圖
```

`--check-shim` 是最重要的一道：把預測換成「上次存取時刻」（＝LRU）之後，
`m5_policy_sim` 的迴圈必須**逐位元**重現 `m4_oracle.Sim.run_online("lru")`。
對不上就代表線上策略與 oracle 的比較是在比兩把不同的尺。
