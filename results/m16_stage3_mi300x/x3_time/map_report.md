# X3：壓縮組合的還原時間 × 連線速度 × GPU 忙（map，run 20261021-025231-m16-x3-map）

> 產生時間 2026-10-21T02:52:31+00:00；micro：llama 20261020-212244-m16-x3-micro-llama, qwen 20261020-213232-m16-x3-micro-qwen。 全部是〔模擬〕：零件是 micro 的實測，還原規則照第二階段的還原器（m10_sweep_sim）。

## 判定（門檻在看數字前寫死，x3_time_howto.md §4）

- STOP-0（模式 Q）：成立；翻轉 0 格。
- STOP-1（模式 L）：STOP-1 不成立（）
- 模式 L：STOP-2a：每一格 A 對 S 都好不到 5% → GPU 忙碌這個軸沒有價值（退回前作）
- 模式 Q：STOP-2a：每一格 A 對 S 都好不到 5% → GPU 忙碌這個軸沒有價值（退回前作）
- 模式 X：STOP-2b：A 對 F 的幾何平均只好 2.2%（<5%）→ NO-GO
- T-a：False；T-b：None；T-c：False；T-d：False
- E-L3 llama|lr_r24：新成本項（重建在忙時比反量化敏感，或和搬運同量級）（重建忙/閒 2.79×、INT4 1.34×、忙時重建÷H2D 0.37）
- E-L3 llama|lr_r40：新成本項（重建在忙時比反量化敏感，或和搬運同量級）（重建忙/閒 2.80×、INT4 1.34×、忙時重建÷H2D 0.30）
- E-L3 qwen|lr_r24：中間（照實報，不下結論）（重建忙/閒 2.78×、INT4 2.05×、忙時重建÷H2D 0.25）
- E-L3 qwen|lr_r40：中間（照實報，不下結論）（重建忙/閒 2.64×、INT4 2.05×、忙時重建÷H2D 0.19）

validate：不跑——沒有模式判 GO_N5 → 照 L5c §5.5 不做 N5 驗證（L：STOP-2a；Q：STOP-2a；X：STOP-2b；STOP-0 成立）

格點版本（L5c §5.4 的連線格點，只當敏感度，不拿來判定）：L：STOP-2a；Q：STOP-2a；X：STOP-2b

## 聚合（每個模式；gain＝1 − T_A ÷ T_基準 的幾何平均）

| 模式 | 格子 | A 對 D | A 對 F | A 對 S | A 對 S ≥5% 的格子 | >15% | F（每個模型） |
|:--|--:|--:|--:|--:|--:|--:|:--|
| L | 68 | +10.1% | +10.1% | +0.0% | 0.00 | 0.00 | {'llama': 'raw', 'qwen': 'raw'} |
| Q | 68 | +56.6% | +0.0% | +0.0% | 0.00 | 0.00 | {'llama': 'int4_kivi@oneshot', 'qwen': 'int4_kivi@oneshot'} |
| X | 68 | +59.7% | +2.2% | +0.1% | 0.01 | 0.00 | {'llama': 'k4v2@perblock', 'qwen': 'int4_kivi@oneshot'} |

## 模式 L：每格的最佳動作

| 模型 | 層 | GiB/s | duty | A | T_A (ms) | 第二名 | 差 | S | F | 對 D | 對 F | 對 S |
|:--|:--|--:|--:|:--|--:|:--|--:|:--|:--|--:|--:|--:|
| llama | b16 | 5.27 | 0 | raw | 568.76 | ll:bytes2hi|zstd:1|t16 | +110.5% | raw | raw | +0.0% | +0.0% | +0.0% |
| llama | b16 | 5.27 | 0.4 | raw | 616.16 | ll:bytes2hi|zstd:1|t16 | +128.0% | raw | raw | +0.0% | +0.0% | +0.0% |
| llama | b16 | 5.27 | 0.7 | raw | 717.29 | ll:bytes2hi|zstd:1|t16 | +113.5% | raw | raw | +0.0% | +0.0% | +0.0% |
| llama | b16 | 5.27 | 0.9 | raw | 679.56 | ll:bytes2hi|zstd:1|t16 | +141.4% | raw | raw | +0.0% | +0.0% | +0.0% |
| llama | b32 | 1.47 | 0 | ll:bytes2hi|zstd:1|t16 | 1197.39 | ll:bytes2hi|zstd:1|t8 | +0.0% | ll:bytes2hi|zstd:1|t16 | raw | +12.2% | +12.2% | +0.0% |
| llama | b32 | 1.47 | 0.4 | ll:bytes2hi|zstd:1|t16 | 1404.98 | ll:bytes2hi|zstd:1|t8 | +0.0% | ll:bytes2hi|zstd:1|t16 | raw | +14.0% | +14.0% | +0.0% |
| llama | b32 | 1.47 | 0.7 | ll:bytes2hi|zstd:1|t16 | 1531.26 | ll:bytes2hi|zstd:1|t8 | +0.0% | ll:bytes2hi|zstd:1|t16 | raw | +16.6% | +16.6% | +0.0% |
| llama | b32 | 1.47 | 0.9 | ll:bytes2hi|zstd:1|t16 | 1640.23 | ll:bytes2hi|zstd:1|t8 | +0.6% | ll:bytes2hi|zstd:1|t16 | raw | +19.0% | +19.0% | +0.0% |
| llama | b48 | 0.42 | 0 | ll:planes|zstd:1|t16 | 2121.81 | ll:planes|zstd:1|t4 | +0.0% | ll:planes|zstd:1|t16 | raw | +11.0% | +11.0% | +0.0% |
| llama | b48 | 0.42 | 0.4 | ll:planes|zstd:1|t16 | 2627.01 | ll:planes|zstd:1|t4 | +0.0% | ll:planes|zstd:1|t16 | raw | +16.0% | +16.0% | +0.0% |
| llama | b48 | 0.42 | 0.7 | ll:planes|zstd:1|t16 | 3132.20 | ll:planes|zstd:1|t4 | +0.0% | ll:planes|zstd:1|t16 | raw | +19.1% | +19.1% | +0.0% |
| llama | b48 | 0.42 | 0.9 | ll:bytes2hi|zstd:1|t1 | 3697.20 | ll:bytes2hi|zstd:1|t16 | +0.0% | ll:planes|zstd:1|t16 | raw | +19.0% | +19.0% | +0.0% |
| llama | cpu_pinned | — | 0 | raw | 79.34 | ll:bytes2hi|zstd:1|t16 | +1409.1% | raw | raw | +0.0% | +0.0% | +0.0% |
| llama | cpu_pinned | — | 0.4 | raw | 99.89 | ll:bytes2hi|zstd:1|t16 | +1306.5% | raw | raw | +0.0% | +0.0% | +0.0% |
| llama | cpu_pinned | — | 0.7 | raw | 115.42 | ll:bytes2hi|zstd:1|t16 | +1226.7% | raw | raw | +0.0% | +0.0% | +0.0% |
| llama | cpu_pinned | — | 0.9 | raw | 129.70 | ll:bytes2hi|zstd:1|t16 | +1164.6% | raw | raw | +0.0% | +0.0% | +0.0% |
| llama | nfs_meas | 0.383 | 0 | ll:bytes2hi|zstd:1|t1 | 2174.84 | ll:bytes2hi|zstd:1|t16 | +0.0% | ll:bytes2|zstd:1|t1 | raw | +11.4% | +11.4% | +0.0% |
| llama | nfs_meas | 0.383 | 0.4 | ll:bytes2hi|zstd:1|t1 | 2762.89 | ll:bytes2hi|zstd:1|t16 | +0.0% | ll:bytes2|zstd:1|t1 | raw | +15.5% | +15.5% | +0.5% |
| llama | nfs_meas | 0.383 | 0.7 | ll:planes|zstd:1|t16 | 3325.00 | ll:planes|zstd:1|t4 | +0.0% | ll:bytes2|zstd:1|t1 | raw | +17.5% | +17.5% | +0.2% |
| llama | nfs_meas | 0.383 | 0.9 | ll:bytes2hi|zstd:1|t1 | 3865.30 | ll:bytes2hi|zstd:1|t16 | +0.0% | ll:bytes2|zstd:1|t1 | raw | +18.6% | +18.6% | +0.0% |
| llama | nfs_param | 0.331 | 0 | ll:planes|zstd:1|t1 | 2304.93 | ll:planes|zstd:1|t16 | +0.0% | ll:planes|zstd:1|t1 | raw | +12.8% | +12.8% | +0.0% |
| llama | nfs_param | 0.331 | 0.4 | ll:planes|zstd:1|t1 | 2945.19 | ll:planes|zstd:1|t16 | +0.0% | ll:planes|zstd:1|t1 | raw | +13.3% | +13.3% | +0.0% |
| llama | nfs_param | 0.331 | 0.7 | ll:planes|zstd:1|t1 | 3585.45 | ll:planes|zstd:1|t16 | +0.0% | ll:planes|zstd:1|t1 | raw | +17.4% | +17.4% | +0.0% |
| llama | nfs_param | 0.331 | 0.9 | ll:bytes2hi|zstd:1|t1 | 4210.09 | ll:bytes2hi|zstd:1|t16 | +0.0% | ll:planes|zstd:1|t1 | raw | +17.8% | +17.8% | +0.0% |
| llama | ssd_meas | 7.87 | 0 | raw | 444.51 | ll:bytes2hi|zstd:1|t16 | +169.4% | raw | raw | +0.0% | +0.0% | +0.0% |
| llama | ssd_meas | 7.87 | 0.4 | raw | 437.06 | ll:bytes2hi|zstd:1|t16 | +221.5% | raw | raw | +0.0% | +0.0% | +0.0% |
| llama | ssd_meas | 7.87 | 0.7 | raw | 480.96 | ll:bytes2hi|zstd:1|t16 | +218.4% | raw | raw | +0.0% | +0.0% | +0.0% |
| llama | ssd_meas | 7.87 | 0.9 | raw | 471.22 | ll:bytes2hi|zstd:1|t16 | +248.1% | raw | raw | +0.0% | +0.0% | +0.0% |
| llama | ssd_param | 6.98 | 0 | raw | 485.05 | ll:bytes2hi|zstd:1|t16 | +146.9% | raw | raw | +0.0% | +0.0% | +0.0% |
| llama | ssd_param | 6.98 | 0.4 | raw | 491.52 | ll:bytes2hi|zstd:1|t16 | +185.8% | raw | raw | +0.0% | +0.0% | +0.0% |
| llama | ssd_param | 6.98 | 0.7 | raw | 557.68 | ll:bytes2hi|zstd:1|t16 | +174.6% | raw | raw | +0.0% | +0.0% | +0.0% |
| llama | ssd_param | 6.98 | 0.9 | raw | 572.97 | ll:bytes2hi|zstd:1|t16 | +186.3% | raw | raw | +0.0% | +0.0% | +0.0% |
| llama | vllm_cpu | — | 0 | raw | 974.01 | ll:bytes2hi|zstd:1|t16 | +120.0% | raw | raw | +0.0% | +0.0% | +0.0% |
| llama | vllm_cpu | — | 0.7 | raw | 1006.49 | ll:bytes2hi|zstd:1|t16 | +112.9% | raw | raw | +0.0% | +0.0% | +0.0% |
| qwen | b16 | 2.77 | 0 | ll:raw|zstd:1|t16 | 402.99 | ll:raw|zstd:1|t8 | +0.0% | ll:raw|zstd:1|t16 | raw | +14.9% | +14.9% | +0.0% |
| qwen | b16 | 2.77 | 0.4 | ll:raw|zstd:1|t16 | 448.27 | ll:raw|zstd:1|t8 | +0.0% | ll:raw|zstd:1|t16 | raw | +17.8% | +17.8% | +0.0% |
| qwen | b16 | 2.77 | 0.7 | ll:raw|zstd:1|t16 | 441.06 | ll:raw|zstd:1|t8 | +0.0% | ll:raw|zstd:1|t16 | raw | +22.4% | +22.4% | +0.0% |
| qwen | b16 | 2.77 | 0.9 | ll:raw|zstd:1|t16 | 449.77 | ll:raw|zstd:1|t8 | +0.0% | ll:raw|zstd:1|t16 | raw | +27.3% | +27.3% | +0.0% |
| qwen | b32 | 0.763 | 0 | ll:bytes2hi|zstd:1|t16 | 915.67 | ll:bytes2hi|zstd:1|t4 | +0.0% | ll:bytes2hi|zstd:1|t16 | raw | +20.1% | +20.1% | +0.0% |
| qwen | b32 | 0.763 | 0.4 | ll:bytes2hi|zstd:1|t16 | 1040.30 | ll:bytes2hi|zstd:1|t4 | +0.0% | ll:bytes2hi|zstd:1|t16 | raw | +23.2% | +23.2% | +0.0% |
| qwen | b32 | 0.763 | 0.7 | ll:bytes2hi|zstd:1|t16 | 1145.90 | ll:bytes2hi|zstd:1|t4 | +0.0% | ll:bytes2hi|zstd:1|t16 | raw | +23.8% | +23.8% | +0.0% |
| qwen | b32 | 0.763 | 0.9 | ll:bytes2hi|zstd:1|t16 | 1286.79 | ll:bytes2hi|zstd:1|t4 | +0.0% | ll:bytes2hi|zstd:1|t16 | raw | +25.2% | +25.2% | +0.0% |
| qwen | b48 | 0.217 | 0 | ll:planes|zstd:1|t1 | 1786.38 | ll:planes|zstd:1|t16 | +0.0% | ll:planes|zstd:1|t1 | raw | +11.2% | +11.2% | +0.0% |
| qwen | b48 | 0.217 | 0.4 | ll:planes|zstd:1|t1 | 2211.70 | ll:planes|zstd:1|t16 | +0.0% | ll:planes|zstd:1|t1 | raw | +16.2% | +16.2% | +0.0% |
| qwen | b48 | 0.217 | 0.7 | ll:bytes2hi|zstd:1|t1 | 2603.82 | ll:bytes2hi|zstd:1|t16 | +0.0% | ll:planes|zstd:1|t1 | raw | +17.2% | +17.2% | +0.0% |
| qwen | b48 | 0.217 | 0.9 | ll:planes|zstd:1|t1 | 2977.29 | ll:planes|zstd:1|t16 | +0.0% | ll:planes|zstd:1|t1 | raw | +21.1% | +21.1% | +0.0% |
| qwen | cpu_pinned | — | 0 | raw | 35.17 | ll:raw|zstd:-3|t16 | +499.0% | raw | raw | +0.0% | +0.0% | +0.0% |
| qwen | cpu_pinned | — | 0.4 | raw | 43.21 | ll:raw|zstd:-3|t8 | +397.6% | raw | raw | +0.0% | +0.0% | +0.0% |
| qwen | cpu_pinned | — | 0.7 | raw | 52.58 | ll:raw|zstd:-3|t4 | +293.5% | raw | raw | +0.0% | +0.0% | +0.0% |
| qwen | cpu_pinned | — | 0.9 | raw | 56.96 | ll:raw|zstd:-3|t4 | +276.6% | raw | raw | +0.0% | +0.0% | +0.0% |
| qwen | nfs_meas | 0.383 | 0 | ll:planes|zstd:1|t1 | 1354.31 | ll:planes|zstd:1|t16 | +0.0% | ll:planes|zstd:1|t1 | raw | +17.6% | +17.6% | +0.0% |
| qwen | nfs_meas | 0.383 | 0.4 | ll:bytes2hi|zstd:1|t1 | 1623.95 | ll:bytes2hi|zstd:1|t16 | +0.0% | ll:planes|zstd:1|t1 | raw | +18.9% | +18.9% | +0.0% |
| qwen | nfs_meas | 0.383 | 0.7 | ll:planes|zstd:1|t1 | 1837.99 | ll:planes|zstd:1|t16 | +0.0% | ll:planes|zstd:1|t1 | raw | +22.1% | +22.1% | +0.0% |
| qwen | nfs_meas | 0.383 | 0.9 | ll:bytes2hi|zstd:1|t1 | 2067.75 | ll:bytes2hi|zstd:1|t16 | +0.0% | ll:planes|zstd:1|t1 | raw | +23.2% | +23.2% | +0.0% |
| qwen | nfs_param | 0.331 | 0 | ll:planes|zstd:1|t1 | 1452.94 | ll:planes|zstd:1|t16 | +0.0% | ll:planes|zstd:1|t1 | raw | +16.2% | +16.2% | +0.0% |
| qwen | nfs_param | 0.331 | 0.4 | ll:bytes2hi|zstd:1|t1 | 1765.95 | ll:bytes2hi|zstd:1|t16 | +0.0% | ll:planes|zstd:1|t1 | raw | +17.8% | +17.8% | +0.0% |
| qwen | nfs_param | 0.331 | 0.7 | ll:planes|zstd:1|t1 | 2011.77 | ll:planes|zstd:1|t16 | +0.0% | ll:planes|zstd:1|t1 | raw | +21.4% | +21.4% | +0.0% |
| qwen | nfs_param | 0.331 | 0.9 | ll:bytes2hi|zstd:1|t1 | 2310.71 | ll:bytes2hi|zstd:1|t16 | +0.0% | ll:planes|zstd:1|t1 | raw | +21.8% | +21.8% | +0.0% |
| qwen | ssd_meas | 7.87 | 0 | ll:raw|zstd:-3|t16 | 210.66 | ll:raw|zstd:-3|t4 | +0.0% | ll:raw|zstd:-3|t4 | raw | +0.0% | +0.0% | +0.0% |
| qwen | ssd_meas | 7.87 | 0.4 | ll:raw|zstd:-3|t4 | 219.93 | ll:raw|zstd:-3|t8 | +0.0% | ll:raw|zstd:-3|t4 | raw | +1.2% | +1.2% | +0.0% |
| qwen | ssd_meas | 7.87 | 0.7 | ll:raw|zstd:-3|t4 | 206.93 | ll:raw|zstd:-3|t8 | +0.0% | ll:raw|zstd:-3|t4 | raw | +0.8% | +0.8% | +0.0% |
| qwen | ssd_meas | 7.87 | 0.9 | ll:raw|zstd:-3|t4 | 214.48 | ll:raw|zstd:-3|t8 | +0.0% | ll:raw|zstd:-3|t4 | raw | +0.0% | +0.0% | +0.0% |
| qwen | ssd_param | 6.98 | 0 | ll:raw|zstd:-3|t1 | 240.71 | ll:raw|zstd:-3|t16 | +0.0% | ll:raw|zstd:-3|t16 | raw | +0.0% | +0.0% | +0.0% |
| qwen | ssd_param | 6.98 | 0.4 | ll:raw|zstd:-3|t16 | 226.28 | ll:raw|zstd:-3|t4 | +0.0% | ll:raw|zstd:-3|t16 | raw | +0.4% | +0.4% | +0.0% |
| qwen | ssd_param | 6.98 | 0.7 | ll:raw|zstd:-3|t16 | 247.78 | ll:raw|zstd:-3|t4 | +0.0% | ll:raw|zstd:-3|t16 | raw | +1.2% | +1.2% | +0.0% |
| qwen | ssd_param | 6.98 | 0.9 | ll:raw|zstd:-3|t16 | 247.78 | ll:raw|zstd:-3|t4 | +0.0% | ll:raw|zstd:-3|t16 | raw | +1.2% | +1.2% | +0.0% |
| qwen | vllm_cpu | — | 0 | ll:raw|zstd:-3|t1 | 818.00 | ll:raw|zstd:-3|t16 | +0.0% | ll:raw|zstd:-3|t1 | raw | +0.0% | +0.0% | +0.0% |
| qwen | vllm_cpu | — | 0.7 | ll:raw|zstd:-3|t1 | 832.26 | ll:raw|zstd:-3|t16 | +0.0% | ll:raw|zstd:-3|t1 | raw | +0.0% | +0.0% | +0.0% |

## 模式 Q：每格的最佳動作

| 模型 | 層 | GiB/s | duty | A | T_A (ms) | 第二名 | 差 | S | F | 對 D | 對 F | 對 S |
|:--|:--|--:|--:|:--|--:|:--|--:|:--|:--|--:|--:|--:|
| llama | b16 | 5.27 | 0 | int4_kivi@oneshot | 220.09 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +61.3% | +0.0% | +0.0% |
| llama | b16 | 5.27 | 0.4 | int4_kivi@oneshot | 228.54 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +62.9% | +0.0% | +0.0% |
| llama | b16 | 5.27 | 0.7 | int4_kivi@oneshot | 241.52 | int4_kivi@perblock | +2.3% | int4_kivi@oneshot | int4_kivi@oneshot | +66.3% | +0.0% | +0.0% |
| llama | b16 | 5.27 | 0.9 | int4_kivi@oneshot | 241.84 | int4_kivi@perblock | +5.2% | int4_kivi@oneshot | int4_kivi@oneshot | +64.4% | +0.0% | +0.0% |
| llama | b32 | 1.47 | 0 | int4_kivi@oneshot | 655.87 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +51.9% | +0.0% | +0.0% |
| llama | b32 | 1.47 | 0.4 | int4_kivi@oneshot | 726.36 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +55.5% | +0.0% | +0.0% |
| llama | b32 | 1.47 | 0.7 | int4_kivi@oneshot | 722.86 | int4_kivi@perblock | +0.7% | int4_kivi@oneshot | int4_kivi@oneshot | +60.7% | +0.0% | +0.0% |
| llama | b32 | 1.47 | 0.9 | int4_kivi@oneshot | 787.96 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +61.1% | +0.0% | +0.0% |
| llama | b48 | 0.42 | 0 | int4_kivi@oneshot | 1444.61 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +39.4% | +0.0% | +0.0% |
| llama | b48 | 0.42 | 0.4 | int4_kivi@oneshot | 1712.91 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +45.2% | +0.0% | +0.0% |
| llama | b48 | 0.42 | 0.7 | int4_kivi@oneshot | 1942.64 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +49.8% | +0.0% | +0.0% |
| llama | b48 | 0.42 | 0.9 | int4_kivi@oneshot | 2163.55 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +52.6% | +0.0% | +0.0% |
| llama | cpu_pinned | — | 0 | int4_kivi@oneshot | 30.26 | int4_kivi@perblock | +5.3% | int4_kivi@oneshot | int4_kivi@oneshot | +61.9% | +0.0% | +0.0% |
| llama | cpu_pinned | — | 0.4 | int4_kivi@oneshot | 32.60 | int4_kivi@perblock | +5.8% | int4_kivi@oneshot | int4_kivi@oneshot | +67.4% | +0.0% | +0.0% |
| llama | cpu_pinned | — | 0.7 | int4_kivi@oneshot | 33.91 | int4_kivi@perblock | +16.4% | int4_kivi@oneshot | int4_kivi@oneshot | +70.6% | +0.0% | +0.0% |
| llama | cpu_pinned | — | 0.9 | int4_kivi@oneshot | 34.69 | int4_kivi@perblock | +36.3% | int4_kivi@oneshot | int4_kivi@oneshot | +73.3% | +0.0% | +0.0% |
| llama | nfs_meas | 0.383 | 0 | int4_kivi@oneshot | 1537.45 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +37.4% | +0.0% | +0.0% |
| llama | nfs_meas | 0.383 | 0.4 | int4_kivi@oneshot | 1793.11 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +45.1% | +0.0% | +0.0% |
| llama | nfs_meas | 0.383 | 0.7 | int4_kivi@oneshot | 2050.30 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +49.1% | +0.0% | +0.0% |
| llama | nfs_meas | 0.383 | 0.9 | int4_kivi@oneshot | 2304.84 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +51.4% | +0.0% | +0.0% |
| llama | nfs_param | 0.331 | 0 | int4_kivi@oneshot | 1653.25 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +37.4% | +0.0% | +0.0% |
| llama | nfs_param | 0.331 | 0.4 | int4_kivi@oneshot | 1957.58 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +42.4% | +0.0% | +0.0% |
| llama | nfs_param | 0.331 | 0.7 | int4_kivi@oneshot | 2271.13 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +47.7% | +0.0% | +0.0% |
| llama | nfs_param | 0.331 | 0.9 | int4_kivi@oneshot | 2595.62 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +49.3% | +0.0% | +0.0% |
| llama | ssd_meas | 7.87 | 0 | int4_kivi@oneshot | 152.61 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +65.7% | +0.0% | +0.0% |
| llama | ssd_meas | 7.87 | 0.4 | int4_kivi@oneshot | 163.01 | int4_kivi@perblock | +1.2% | int4_kivi@oneshot | int4_kivi@oneshot | +62.7% | +0.0% | +0.0% |
| llama | ssd_meas | 7.87 | 0.7 | int4_kivi@oneshot | 163.46 | int4_kivi@perblock | +3.4% | int4_kivi@oneshot | int4_kivi@oneshot | +66.0% | +0.0% | +0.0% |
| llama | ssd_meas | 7.87 | 0.9 | int4_kivi@oneshot | 163.78 | int4_kivi@perblock | +7.7% | int4_kivi@oneshot | int4_kivi@oneshot | +65.2% | +0.0% | +0.0% |
| llama | ssd_param | 6.98 | 0 | int4_kivi@oneshot | 182.44 | int4_kivi@perblock | +0.6% | int4_kivi@oneshot | int4_kivi@oneshot | +62.4% | +0.0% | +0.0% |
| llama | ssd_param | 6.98 | 0.4 | int4_kivi@oneshot | 180.05 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +63.4% | +0.0% | +0.0% |
| llama | ssd_param | 6.98 | 0.7 | int4_kivi@oneshot | 183.58 | int4_kivi@perblock | +3.0% | int4_kivi@oneshot | int4_kivi@oneshot | +67.1% | +0.0% | +0.0% |
| llama | ssd_param | 6.98 | 0.9 | int4_kivi@oneshot | 180.50 | int4_kivi@perblock | +8.9% | int4_kivi@oneshot | int4_kivi@oneshot | +68.5% | +0.0% | +0.0% |
| llama | vllm_cpu | — | 0 | int4_kivi@oneshot | 925.41 | int4_kivi@perblock | +0.1% | int4_kivi@oneshot | int4_kivi@oneshot | +5.0% | +0.0% | +0.0% |
| llama | vllm_cpu | — | 0.7 | int4_kivi@oneshot | 924.97 | int4_kivi@perblock | +0.6% | int4_kivi@oneshot | int4_kivi@oneshot | +8.1% | +0.0% | +0.0% |
| qwen | b16 | 2.77 | 0 | int4_kivi@oneshot | 181.43 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +61.7% | +0.0% | +0.0% |
| qwen | b16 | 2.77 | 0.4 | int4_kivi@oneshot | 185.58 | int4_kivi@perblock | +0.3% | int4_kivi@oneshot | int4_kivi@oneshot | +66.0% | +0.0% | +0.0% |
| qwen | b16 | 2.77 | 0.7 | int4_kivi@oneshot | 199.78 | int4_kivi@perblock | +2.6% | int4_kivi@oneshot | int4_kivi@oneshot | +64.9% | +0.0% | +0.0% |
| qwen | b16 | 2.77 | 0.9 | int4_kivi@oneshot | 200.44 | int4_kivi@perblock | +6.4% | int4_kivi@oneshot | int4_kivi@oneshot | +67.6% | +0.0% | +0.0% |
| qwen | b32 | 0.763 | 0 | int4_kivi@oneshot | 547.30 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +52.2% | +0.0% | +0.0% |
| qwen | b32 | 0.763 | 0.4 | int4_kivi@oneshot | 595.80 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +56.0% | +0.0% | +0.0% |
| qwen | b32 | 0.763 | 0.7 | int4_kivi@oneshot | 634.82 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +57.8% | +0.0% | +0.0% |
| qwen | b32 | 0.763 | 0.9 | int4_kivi@oneshot | 629.45 | int4_kivi@perblock | +12.4% | int4_kivi@oneshot | int4_kivi@oneshot | +63.4% | +0.0% | +0.0% |
| qwen | b48 | 0.217 | 0 | int4_kivi@oneshot | 1218.47 | int4_kivi@perblock | +0.1% | int4_kivi@oneshot | int4_kivi@oneshot | +39.4% | +0.0% | +0.0% |
| qwen | b48 | 0.217 | 0.4 | int4_kivi@oneshot | 1419.69 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +46.2% | +0.0% | +0.0% |
| qwen | b48 | 0.217 | 0.7 | int4_kivi@oneshot | 1646.33 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +47.6% | +0.0% | +0.0% |
| qwen | b48 | 0.217 | 0.9 | int4_kivi@oneshot | 1833.37 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +51.4% | +0.0% | +0.0% |
| qwen | cpu_pinned | — | 0 | int4_kivi@oneshot | 15.45 | int4_kivi@perblock | +20.9% | int4_kivi@oneshot | int4_kivi@oneshot | +56.1% | +0.0% | +0.0% |
| qwen | cpu_pinned | — | 0.4 | int4_kivi@oneshot | 16.48 | int4_kivi@perblock | +17.4% | int4_kivi@oneshot | int4_kivi@oneshot | +61.9% | +0.0% | +0.0% |
| qwen | cpu_pinned | — | 0.7 | int4_kivi@oneshot | 17.85 | int8_token@oneshot | +22.6% | int4_kivi@oneshot | int4_kivi@oneshot | +66.1% | +0.0% | +0.0% |
| qwen | cpu_pinned | — | 0.9 | int4_kivi@oneshot | 18.67 | int8_token@oneshot | +21.3% | int4_kivi@oneshot | int4_kivi@oneshot | +67.2% | +0.0% | +0.0% |
| qwen | nfs_meas | 0.383 | 0 | int4_kivi@oneshot | 871.80 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +47.0% | +0.0% | +0.0% |
| qwen | nfs_meas | 0.383 | 0.4 | int4_kivi@oneshot | 980.67 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +51.0% | +0.0% | +0.0% |
| qwen | nfs_meas | 0.383 | 0.7 | int4_kivi@oneshot | 1068.07 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +54.7% | +0.0% | +0.0% |
| qwen | nfs_meas | 0.383 | 0.9 | int4_kivi@oneshot | 1184.25 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +56.0% | +0.0% | +0.0% |
| qwen | nfs_param | 0.331 | 0 | int4_kivi@oneshot | 960.26 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +44.6% | +0.0% | +0.0% |
| qwen | nfs_param | 0.331 | 0.4 | int4_kivi@oneshot | 1100.83 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +48.7% | +0.0% | +0.0% |
| qwen | nfs_param | 0.331 | 0.7 | int4_kivi@oneshot | 1225.32 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +52.1% | +0.0% | +0.0% |
| qwen | nfs_param | 0.331 | 0.9 | int4_kivi@oneshot | 1286.79 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +56.5% | +0.0% | +0.0% |
| qwen | ssd_meas | 7.87 | 0 | int4_kivi@oneshot | 70.72 | int4_kivi@perblock | +3.2% | int4_kivi@oneshot | int4_kivi@oneshot | +66.4% | +0.0% | +0.0% |
| qwen | ssd_meas | 7.87 | 0.4 | int4_kivi@oneshot | 70.84 | int4_kivi@perblock | +1.3% | int4_kivi@oneshot | int4_kivi@oneshot | +68.2% | +0.0% | +0.0% |
| qwen | ssd_meas | 7.87 | 0.7 | int4_kivi@oneshot | 71.97 | int4_kivi@perblock | +7.2% | int4_kivi@oneshot | int4_kivi@oneshot | +65.5% | +0.0% | +0.0% |
| qwen | ssd_meas | 7.87 | 0.9 | int4_kivi@oneshot | 71.49 | int4_kivi@perblock | +19.5% | int4_kivi@oneshot | int4_kivi@oneshot | +66.7% | +0.0% | +0.0% |
| qwen | ssd_param | 6.98 | 0 | int4_kivi@oneshot | 79.52 | int4_kivi@perblock | +4.1% | int4_kivi@oneshot | int4_kivi@oneshot | +67.0% | +0.0% | +0.0% |
| qwen | ssd_param | 6.98 | 0.4 | int4_kivi@oneshot | 80.03 | int4_kivi@perblock | +3.6% | int4_kivi@oneshot | int4_kivi@oneshot | +64.8% | +0.0% | +0.0% |
| qwen | ssd_param | 6.98 | 0.7 | int4_kivi@oneshot | 80.77 | int4_kivi@perblock | +6.4% | int4_kivi@oneshot | int4_kivi@oneshot | +67.8% | +0.0% | +0.0% |
| qwen | ssd_param | 6.98 | 0.9 | int4_kivi@oneshot | 81.43 | int4_kivi@perblock | +15.7% | int4_kivi@oneshot | int4_kivi@oneshot | +67.5% | +0.0% | +0.0% |
| qwen | vllm_cpu | — | 0 | int4_kivi@oneshot | 798.28 | int4_kivi@perblock | +0.4% | int4_kivi@oneshot | int4_kivi@oneshot | +2.4% | +0.0% | +0.0% |
| qwen | vllm_cpu | — | 0.7 | int4_kivi@oneshot | 797.53 | int8_token@oneshot | +0.5% | int4_kivi@oneshot | int4_kivi@oneshot | +4.2% | +0.0% | +0.0% |

## 模式 X：每格的最佳動作

| 模型 | 層 | GiB/s | duty | A | T_A (ms) | 第二名 | 差 | S | F | 對 D | 對 F | 對 S |
|:--|:--|--:|--:|:--|--:|:--|--:|:--|:--|--:|--:|--:|
| llama | b16 | 5.27 | 0 | k4v2@oneshot | 185.91 | k4v2@perblock | +0.0% | k4v2@oneshot | k4v2@perblock | +67.3% | +0.0% | +0.0% |
| llama | b16 | 5.27 | 0.4 | k4v2@oneshot | 183.00 | k4v2@perblock | +8.3% | k4v2@oneshot | k4v2@perblock | +70.3% | +7.6% | +0.0% |
| llama | b16 | 5.27 | 0.7 | k4v2@oneshot | 194.18 | k4v2@perblock | +0.9% | k4v2@oneshot | k4v2@perblock | +72.9% | +0.9% | +0.0% |
| llama | b16 | 5.27 | 0.9 | k4v2@oneshot | 196.69 | k4v2@perblock | +11.1% | k4v2@oneshot | k4v2@perblock | +71.1% | +10.0% | +0.0% |
| llama | b32 | 1.47 | 0 | k4v2@oneshot | 526.49 | k4v2@perblock | +0.0% | k4v2@oneshot | k4v2@perblock | +61.4% | +0.0% | +0.0% |
| llama | b32 | 1.47 | 0.4 | k4v2@oneshot | 606.37 | k4v2@perblock | +0.0% | k4v2@oneshot | k4v2@perblock | +62.9% | +0.0% | +0.0% |
| llama | b32 | 1.47 | 0.7 | k4v2@oneshot | 636.48 | k4v2@perblock | +0.0% | k4v2@oneshot | k4v2@perblock | +65.4% | +0.0% | +0.0% |
| llama | b32 | 1.47 | 0.9 | k4v2@oneshot | 679.56 | k4v2@perblock | +0.0% | k4v2@oneshot | k4v2@perblock | +66.4% | +0.0% | +0.0% |
| llama | b48 | 0.42 | 0 | k4v2@oneshot | 1268.51 | k4v2@perblock | +0.0% | k4v2@oneshot | k4v2@perblock | +46.8% | +0.0% | +0.0% |
| llama | b48 | 0.42 | 0.4 | k4v2@oneshot | 1479.97 | k4v2@perblock | +0.0% | k4v2@oneshot | k4v2@perblock | +52.7% | +0.0% | +0.0% |
| llama | b48 | 0.42 | 0.7 | k4v2@oneshot | 1642.79 | k4v2@perblock | +0.4% | k4v2@oneshot | k4v2@perblock | +57.6% | +0.4% | +0.0% |
| llama | b48 | 0.42 | 0.9 | k4v2@oneshot | 1792.74 | k4v2@perblock | +5.4% | k4v2@oneshot | k4v2@perblock | +60.7% | +5.1% | +0.0% |
| llama | cpu_pinned | — | 0 | k4v2@perblock | 27.97 | int4_kivi@oneshot | +8.2% | int4_kivi@oneshot | k4v2@perblock | +64.7% | +0.0% | +7.6% |
| llama | cpu_pinned | — | 0.4 | k4v2@perblock | 32.18 | int4_kivi@oneshot | +1.3% | int4_kivi@oneshot | k4v2@perblock | +67.8% | +0.0% | +1.3% |
| llama | cpu_pinned | — | 0.7 | int4_kivi@oneshot | 33.91 | int4_kivi@perblock | +16.4% | int4_kivi@oneshot | k4v2@perblock | +70.6% | +16.5% | +0.0% |
| llama | cpu_pinned | — | 0.9 | int4_kivi@oneshot | 34.69 | int4_kivi@perblock | +36.3% | int4_kivi@oneshot | k4v2@perblock | +73.3% | +35.4% | +0.0% |
| llama | nfs_meas | 0.383 | 0 | k4v2@oneshot | 1350.32 | k4v2@perblock | +1.0% | k4v2@oneshot | k4v2@perblock | +45.0% | +1.0% | +0.0% |
| llama | nfs_meas | 0.383 | 0.4 | k4v2@oneshot | 1556.30 | k4v2@perblock | +0.1% | k4v2@oneshot | k4v2@perblock | +52.4% | +0.1% | +0.0% |
| llama | nfs_meas | 0.383 | 0.7 | k4v2@oneshot | 1760.67 | k4v2@perblock | +0.4% | k4v2@oneshot | k4v2@perblock | +56.3% | +0.4% | +0.0% |
| llama | nfs_meas | 0.383 | 0.9 | k4v2@oneshot | 1925.05 | k4v2@perblock | +5.2% | k4v2@oneshot | k4v2@perblock | +59.4% | +4.9% | +0.0% |
| llama | nfs_param | 0.331 | 0 | k4v2@oneshot | 1465.20 | k4v2@perblock | +0.0% | k4v2@oneshot | k4v2@perblock | +44.6% | +0.0% | +0.0% |
| llama | nfs_param | 0.331 | 0.4 | k4v2@oneshot | 1712.91 | k4v2@perblock | +0.0% | k4v2@oneshot | k4v2@perblock | +49.6% | +0.0% | +0.0% |
| llama | nfs_param | 0.331 | 0.7 | k4v2@oneshot | 1942.64 | k4v2@perblock | +0.1% | k4v2@oneshot | k4v2@perblock | +55.3% | +0.1% | +0.0% |
| llama | nfs_param | 0.331 | 0.9 | k4v2@oneshot | 2163.55 | k4v2@perblock | +0.0% | k4v2@oneshot | k4v2@perblock | +57.7% | +0.0% | +0.0% |
| llama | ssd_meas | 7.87 | 0 | k4v2@oneshot | 123.47 | k4v2@perblock | +7.3% | k4v2@oneshot | k4v2@perblock | +72.2% | +6.8% | +0.0% |
| llama | ssd_meas | 7.87 | 0.4 | k4v2@oneshot | 132.76 | k4v2@perblock | +0.2% | k4v2@oneshot | k4v2@perblock | +69.6% | +0.2% | +0.0% |
| llama | ssd_meas | 7.87 | 0.7 | k4v2@oneshot | 129.26 | k4v2@perblock | +10.8% | k4v2@oneshot | k4v2@perblock | +73.1% | +9.7% | +0.0% |
| llama | ssd_meas | 7.87 | 0.9 | k4v2@oneshot | 134.24 | k4v2@perblock | +16.3% | k4v2@oneshot | k4v2@perblock | +71.5% | +14.0% | +0.0% |
| llama | ssd_param | 6.98 | 0 | k4v2@oneshot | 147.80 | k4v2@perblock | +0.5% | k4v2@oneshot | k4v2@perblock | +69.5% | +0.5% | +0.0% |
| llama | ssd_param | 6.98 | 0.4 | k4v2@oneshot | 148.86 | k4v2@perblock | +2.0% | k4v2@oneshot | k4v2@perblock | +69.7% | +1.9% | +0.0% |
| llama | ssd_param | 6.98 | 0.7 | k4v2@oneshot | 149.53 | k4v2@perblock | +6.5% | k4v2@oneshot | k4v2@perblock | +73.2% | +6.1% | +0.0% |
| llama | ssd_param | 6.98 | 0.9 | k4v2@oneshot | 150.34 | k4v2@perblock | +14.6% | k4v2@oneshot | k4v2@perblock | +73.8% | +12.7% | +0.0% |
| llama | vllm_cpu | — | 0 | k4v2@perblock | 922.64 | int4_kivi@oneshot | +0.3% | int4_kivi@oneshot | k4v2@perblock | +5.3% | +0.0% | +0.3% |
| llama | vllm_cpu | — | 0.7 | int4_kivi@oneshot | 924.97 | int4_kivi@perblock | +0.6% | int4_kivi@oneshot | k4v2@perblock | +8.1% | +0.7% | +0.0% |
| qwen | b16 | 2.77 | 0 | int4_kivi@oneshot | 181.43 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +61.7% | +0.0% | +0.0% |
| qwen | b16 | 2.77 | 0.4 | int4_kivi@oneshot | 185.58 | int4_kivi@perblock | +0.3% | int4_kivi@oneshot | int4_kivi@oneshot | +66.0% | +0.0% | +0.0% |
| qwen | b16 | 2.77 | 0.7 | int4_kivi@oneshot | 199.78 | int4_kivi@perblock | +2.6% | int4_kivi@oneshot | int4_kivi@oneshot | +64.9% | +0.0% | +0.0% |
| qwen | b16 | 2.77 | 0.9 | int4_kivi@oneshot | 200.44 | int4_kivi@perblock | +6.4% | int4_kivi@oneshot | int4_kivi@oneshot | +67.6% | +0.0% | +0.0% |
| qwen | b32 | 0.763 | 0 | int4_kivi@oneshot | 547.30 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +52.2% | +0.0% | +0.0% |
| qwen | b32 | 0.763 | 0.4 | int4_kivi@oneshot | 595.80 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +56.0% | +0.0% | +0.0% |
| qwen | b32 | 0.763 | 0.7 | int4_kivi@oneshot | 634.82 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +57.8% | +0.0% | +0.0% |
| qwen | b32 | 0.763 | 0.9 | int4_kivi@oneshot | 629.45 | int4_kivi@perblock | +12.4% | int4_kivi@oneshot | int4_kivi@oneshot | +63.4% | +0.0% | +0.0% |
| qwen | b48 | 0.217 | 0 | int4_kivi@oneshot | 1218.47 | int4_kivi@perblock | +0.1% | int4_kivi@oneshot | int4_kivi@oneshot | +39.4% | +0.0% | +0.0% |
| qwen | b48 | 0.217 | 0.4 | int4_kivi@oneshot | 1419.69 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +46.2% | +0.0% | +0.0% |
| qwen | b48 | 0.217 | 0.7 | int4_kivi@oneshot | 1646.33 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +47.6% | +0.0% | +0.0% |
| qwen | b48 | 0.217 | 0.9 | int4_kivi@oneshot | 1833.37 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +51.4% | +0.0% | +0.0% |
| qwen | cpu_pinned | — | 0 | int4_kivi@oneshot | 15.45 | int4_kivi@perblock | +20.9% | int4_kivi@oneshot | int4_kivi@oneshot | +56.1% | +0.0% | +0.0% |
| qwen | cpu_pinned | — | 0.4 | int4_kivi@oneshot | 16.48 | int4_kivi@perblock | +17.4% | int4_kivi@oneshot | int4_kivi@oneshot | +61.9% | +0.0% | +0.0% |
| qwen | cpu_pinned | — | 0.7 | int4_kivi@oneshot | 17.85 | int8_token@oneshot | +22.6% | int4_kivi@oneshot | int4_kivi@oneshot | +66.1% | +0.0% | +0.0% |
| qwen | cpu_pinned | — | 0.9 | int4_kivi@oneshot | 18.67 | int8_token@oneshot | +21.3% | int4_kivi@oneshot | int4_kivi@oneshot | +67.2% | +0.0% | +0.0% |
| qwen | nfs_meas | 0.383 | 0 | int4_kivi@oneshot | 871.80 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +47.0% | +0.0% | +0.0% |
| qwen | nfs_meas | 0.383 | 0.4 | int4_kivi@oneshot | 980.67 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +51.0% | +0.0% | +0.0% |
| qwen | nfs_meas | 0.383 | 0.7 | int4_kivi@oneshot | 1068.07 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +54.7% | +0.0% | +0.0% |
| qwen | nfs_meas | 0.383 | 0.9 | int4_kivi@oneshot | 1184.25 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +56.0% | +0.0% | +0.0% |
| qwen | nfs_param | 0.331 | 0 | int4_kivi@oneshot | 960.26 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +44.6% | +0.0% | +0.0% |
| qwen | nfs_param | 0.331 | 0.4 | int4_kivi@oneshot | 1100.83 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +48.7% | +0.0% | +0.0% |
| qwen | nfs_param | 0.331 | 0.7 | int4_kivi@oneshot | 1225.32 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +52.1% | +0.0% | +0.0% |
| qwen | nfs_param | 0.331 | 0.9 | int4_kivi@oneshot | 1286.79 | int4_kivi@perblock | +0.0% | int4_kivi@oneshot | int4_kivi@oneshot | +56.5% | +0.0% | +0.0% |
| qwen | ssd_meas | 7.87 | 0 | int4_kivi@oneshot | 70.72 | int4_kivi@perblock | +3.2% | int4_kivi@oneshot | int4_kivi@oneshot | +66.4% | +0.0% | +0.0% |
| qwen | ssd_meas | 7.87 | 0.4 | int4_kivi@oneshot | 70.84 | int4_kivi@perblock | +1.3% | int4_kivi@oneshot | int4_kivi@oneshot | +68.2% | +0.0% | +0.0% |
| qwen | ssd_meas | 7.87 | 0.7 | int4_kivi@oneshot | 71.97 | int4_kivi@perblock | +7.2% | int4_kivi@oneshot | int4_kivi@oneshot | +65.5% | +0.0% | +0.0% |
| qwen | ssd_meas | 7.87 | 0.9 | int4_kivi@oneshot | 71.49 | int4_kivi@perblock | +19.5% | int4_kivi@oneshot | int4_kivi@oneshot | +66.7% | +0.0% | +0.0% |
| qwen | ssd_param | 6.98 | 0 | int4_kivi@oneshot | 79.52 | int4_kivi@perblock | +4.1% | int4_kivi@oneshot | int4_kivi@oneshot | +67.0% | +0.0% | +0.0% |
| qwen | ssd_param | 6.98 | 0.4 | int4_kivi@oneshot | 80.03 | int4_kivi@perblock | +3.6% | int4_kivi@oneshot | int4_kivi@oneshot | +64.8% | +0.0% | +0.0% |
| qwen | ssd_param | 6.98 | 0.7 | int4_kivi@oneshot | 80.77 | int4_kivi@perblock | +6.4% | int4_kivi@oneshot | int4_kivi@oneshot | +67.8% | +0.0% | +0.0% |
| qwen | ssd_param | 6.98 | 0.9 | int4_kivi@oneshot | 81.43 | int4_kivi@perblock | +15.7% | int4_kivi@oneshot | int4_kivi@oneshot | +67.5% | +0.0% | +0.0% |
| qwen | vllm_cpu | — | 0 | int4_kivi@oneshot | 798.28 | int4_kivi@perblock | +0.4% | int4_kivi@oneshot | int4_kivi@oneshot | +2.4% | +0.0% | +0.0% |
| qwen | vllm_cpu | — | 0.7 | int4_kivi@oneshot | 797.53 | int8_token@oneshot | +0.5% | int4_kivi@oneshot | int4_kivi@oneshot | +4.2% | +0.0% | +0.0% |

## 組成模型對第二階段實測（CPU 層、全部載入）

| 模型 | 第二階段 | 配置 | 動作 | duty | 實測 (ms) | 組成 (ms) | 差 | 可用 |
|:--|:--|:--|:--|--:|--:|--:|--:|:--|
| llama | q4/llama_int4_P3（20261018-101744-m10-rs2-q4-llama-int4-P3） | E0 | raw | 0 | 83.81 | 79.34 | -5.3% | 是 |
| llama | q4/llama_int4_P3（20261018-101744-m10-rs2-q4-llama-int4-P3） | E0 | raw | 0.7 | 123.75 | 115.42 | -6.7% | 是 |
| llama | q4/llama_int4_P3（20261018-101744-m10-rs2-q4-llama-int4-P3） | E1 | int4_kivi@oneshot | 0 | 33.12 | 30.74 | -7.2% | 是 |
| llama | q4/llama_int4_P3（20261018-101744-m10-rs2-q4-llama-int4-P3） | E1 | int4_kivi@oneshot | 0.7 | 35.76 | 33.91 | -5.2% | 是 |
| llama | q4/llama_int4_P0（20261018-100535-m10-rs2-q4-llama-int4-P0） | E1 | int4_kivi@perblock | 0 | 35.73 | 31.85 | -10.8% | 是 |
| llama | q4/llama_int4_P0（20261018-100535-m10-rs2-q4-llama-int4-P0） | E1 | int4_kivi@perblock | 0.7 | 54.72 | 39.48 | -27.8% ⚠ | 是 |
| llama | q4/llama_fp8_P3（20261018-094951-m10-rs2-q4-llama-fp8-P3） | E1 | fp8@oneshot | 0 | 45.55 | 43.38 | -4.8% | 是 |
| llama | q4/llama_fp8_P3（20261018-094951-m10-rs2-q4-llama-fp8-P3） | E1 | fp8@oneshot | 0.7 | 49.33 | 47.44 | -3.8% | 是 |
| llama | q4duty/llama_int4_P3（20261018-113107-m10-rs2-q4duty-llama-P3） | E1 | int4_kivi@oneshot | 0.4 | 35.23 | 32.60 | -7.4% | 是 |
| llama | q4duty/llama_int4_P3（20261018-113107-m10-rs2-q4duty-llama-P3） | E1 | int4_kivi@oneshot | 0.7 | 36.08 | 33.91 | -6.0% | 是 |
| llama | q4duty/llama_int4_P3（20261018-113107-m10-rs2-q4duty-llama-P3） | E1 | int4_kivi@oneshot | 0.9 | 36.70 | 34.69 | -5.5% | 是 |
| qwen | q4/qwen_int4_P3（20261018-110958-m10-rs2-q4-qwen-int4-P3） | E0 | raw | 0 | 38.95 | 35.17 | -9.7% | 是 |
| qwen | q4/qwen_int4_P3（20261018-110958-m10-rs2-q4-qwen-int4-P3） | E0 | raw | 0.7 | 55.82 | 52.58 | -5.8% | 是 |
| qwen | q4/qwen_int4_P3（20261018-110958-m10-rs2-q4-qwen-int4-P3） | E1 | int4_kivi@oneshot | 0 | 17.51 | 15.45 | -11.7% | 是 |
| qwen | q4/qwen_int4_P3（20261018-110958-m10-rs2-q4-qwen-int4-P3） | E1 | int4_kivi@oneshot | 0.7 | 18.89 | 17.85 | -5.5% | 是 |
| qwen | q4/qwen_int4_P0（20261018-105843-m10-rs2-q4-qwen-int4-P0） | E1 | int4_kivi@perblock | 0 | 21.07 | 18.69 | -11.3% | 是 |
| qwen | q4/qwen_int4_P0（20261018-105843-m10-rs2-q4-qwen-int4-P0） | E1 | int4_kivi@perblock | 0.7 | 42.90 | 23.02 | -46.3% ⚠ | 是 |
| qwen | q4/qwen_fp8_P3（20261018-104426-m10-rs2-q4-qwen-fp8-P3） | E1 | fp8@oneshot | 0 | 21.01 | 19.10 | -9.1% | 是 |
| qwen | q4/qwen_fp8_P3（20261018-104426-m10-rs2-q4-qwen-fp8-P3） | E1 | fp8@oneshot | 0.7 | 22.21 | 21.47 | -3.3% | 是 |
| qwen | q4duty/qwen_int4_P3（20261018-113436-m10-rs2-q4duty-qwen-P3） | E1 | int4_kivi@oneshot | 0.4 | 19.29 | 16.48 | -14.6% | 是 |
| qwen | q4duty/qwen_int4_P3（20261018-113436-m10-rs2-q4duty-qwen-P3） | E1 | int4_kivi@oneshot | 0.7 | 19.73 | 17.85 | -9.5% | 是 |
| qwen | q4duty/qwen_int4_P3（20261018-113436-m10-rs2-q4duty-qwen-P3） | E1 | int4_kivi@oneshot | 0.9 | 20.78 | 18.67 | -10.1% | 是 |

## 常數（出處見 constants_used.csv）

| 模型 | 常數 | 值 | 單位 | 檔 | run_id |
|:--|:--|:--|:--|:--|:--|
| llama | f_idle_s | 64 個值；總和 3.63127 | s/chunk | results/m10_precision_mi300x/speed_llama/bw.json | 20261015-154558-m10-calibf-llama |
| llama | chunk_bytes | 67108864 | B | results/m10_precision_mi300x/speed_llama/bw.json | 20261015-154558-m10-calibf-llama |
| llama | real_pinned_h2d_GiBps | 49.858 | GiB/s | results/m10_precision_mi300x/speed_llama/bw.json | 20261015-154558-m10-calibf-llama |
| llama | link_b16 | 5.2746 | GiB/s | results/m10_precision_mi300x/speed_llama/bw.json | 20261015-154558-m10-calibf-llama |
| llama | link_b32 | 1.4668 | GiB/s | results/m10_precision_mi300x/speed_llama/bw.json | 20261015-154558-m10-calibf-llama |
| llama | link_b48 | 0.4196 | GiB/s | results/m10_precision_mi300x/speed_llama/bw.json | 20261015-154558-m10-calibf-llama |
| llama | mult_f_stage2_d0.7 | 2.10723 | × | results/m10_precision_mi300x/speed_llama/summary_int4.csv | 20261016-034902-m10-grid-int4-llama |
| llama | fp8_fmt | fp8_token | - | results/m10_precision_mi300x/fmt_select/fmt_choice.json | 20261015-160426-m10-fmtsel-llama-s0;20261015-160541-m10-fmtsel-llama-s1;20261015-160656-m10-fmtsel-qwen-s0;20261015-160811-m10-fmtsel-qwen-s1 |
| qwen | f_idle_s | 64 個值；總和 3.07731 | s/chunk | results/m10_precision_mi300x/speed_qwen/bw.json | 20261015-154647-m10-calibf-qwen |
| qwen | chunk_bytes | 29360128 | B | results/m10_precision_mi300x/speed_qwen/bw.json | 20261015-154647-m10-calibf-qwen |
| qwen | real_pinned_h2d_GiBps | 48.644 | GiB/s | results/m10_precision_mi300x/speed_qwen/bw.json | 20261015-154647-m10-calibf-qwen |
| qwen | link_b16 | 2.7712 | GiB/s | results/m10_precision_mi300x/speed_qwen/bw.json | 20261015-154647-m10-calibf-qwen |
| qwen | link_b32 | 0.7635 | GiB/s | results/m10_precision_mi300x/speed_qwen/bw.json | 20261015-154647-m10-calibf-qwen |
| qwen | link_b48 | 0.2175 | GiB/s | results/m10_precision_mi300x/speed_qwen/bw.json | 20261015-154647-m10-calibf-qwen |
| qwen | mult_f_stage2_d0.7 | 2.1217 | × | results/m10_precision_mi300x/speed_qwen/summary_int4.csv | 20261016-052143-m10-grid-int4-qwen |
| qwen | fp8_fmt | fp8_head | - | results/m10_precision_mi300x/fmt_select/fmt_choice.json | 20261015-160426-m10-fmtsel-llama-s0;20261015-160541-m10-fmtsel-llama-s1;20261015-160656-m10-fmtsel-qwen-s0;20261015-160811-m10-fmtsel-qwen-s1 |
| - | link_ssd_meas | 7.86504 | GiB/s | results/m7_write_policy_mi300x/calib_c2.csv | 20261008-130438-m7-c2 |
| - | link_nfs_meas | 0.38252 | GiB/s | results/m7_write_policy_mi300x/calib_c2.csv | 20261008-130438-m7-c2 |
| - | link_ssd_param | 6.98115 | GiB/s | results/m7_write_policy_mi300x/tier_params.json | 20261008-130257-m7-c0;20261008-130438-m7-c2 |
| - | link_nfs_param | 0.331084 | GiB/s | results/m7_write_policy_mi300x/tier_params.json | 20261008-130257-m7-c0;20261008-130438-m7-c2 |
| - | mult_f_phase1_d0.4 | 1.48 | × | results/m7_write_policy_mi300x/busy_levels.json | 20261008-130834-m7-c5;20261008-130956-m7-c5b |
| - | duty_idle | 0 | duty | results/m10_restore2_mi300x/busy_levels_q4.json | - |
| - | duty_busy1 | 0.4 | duty | results/m10_restore2_mi300x/busy_levels_q4.json | - |
| - | duty_busy2 | 0.7 | duty | results/m10_restore2_mi300x/busy_levels_q4.json | - |
| - | duty_busy3 | 0.9 | duty | results/m10_restore2_mi300x/busy_levels_q4.json | - |
| llama | vllm_c6_T_s | 0.974006 | s | results/m7_write_policy_mi300x/c6_vllm.csv | 20261008-141324-m7-c6-vllm |
| llama | vllm_c6_bytes | 4.29497e+09 | B | results/m7_write_policy_mi300x/c6_vllm.csv | 20261008-141324-m7-c6-vllm |

## 注意

- llama duty 0：重算慢化 1.000（定義）
- llama duty 0.4：重算慢化 1.498（micro recompute 探針（20261020-212244-m16-x3-micro-llama，9 個位置））
- llama duty 0.7：重算慢化 2.185（micro recompute 探針（20261020-212244-m16-x3-micro-llama，9 個位置））
- llama duty 0.9：重算慢化 3.088（micro recompute 探針（20261020-212244-m16-x3-micro-llama，9 個位置））
- qwen duty 0：重算慢化 1.000（定義）
- qwen duty 0.4：重算慢化 1.478（micro recompute 探針（20261020-213232-m16-x3-micro-qwen，9 個位置））
- qwen duty 0.7：重算慢化 2.094（micro recompute 探針（20261020-213232-m16-x3-micro-qwen，9 個位置））
- qwen duty 0.9：重算慢化 2.939（micro recompute 探針（20261020-213232-m16-x3-micro-qwen，9 個位置））
- 組成模型和第二階段實測差 >15% 的格子見「組成模型對第二階段實測」表；那種動作的預測要先打折看
