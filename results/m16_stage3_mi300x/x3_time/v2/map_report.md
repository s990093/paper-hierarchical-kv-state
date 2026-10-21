# X3：壓縮組合的還原時間 × 連線速度 × GPU 忙（map，run 20261021-123143-m16-x3-map2）

> 產生時間 2026-10-21T12:31:44+00:00；micro：llama 20261020-212244-m16-x3-micro-llama;20261021-075914-m16-x3-micro2-llama;20261021-080628-m16-x3-wcost-llama, qwen 20261020-213232-m16-x3-micro-qwen;20261021-080256-m16-x3-micro2-qwen;20261021-082103-m16-x3-wcost-qwen。 全部是〔模擬〕：零件是 micro 的實測，還原規則照第二階段的還原器（m10_sweep_sim）。
> micro 目錄：{'llama': ['results/m16_stage3_mi300x/x3_time', 'results/m16_stage3_mi300x/x3_time/v2', 'results/m16_stage3_mi300x/x3_time/v2/wcost'], 'qwen': ['results/m16_stage3_mi300x/x3_time', 'results/m16_stage3_mi300x/x3_time/v2', 'results/m16_stage3_mi300x/x3_time/v2/wcost']}；X2 品質可行集：{'run_id': '20261021-123141-m16-x2b-report', 'file': 'results/m16_stage3_mi300x/x_stack/v2/x2_feasible.csv'}。

## 判定（門檻在看數字前寫死，x3_time_howto.md §4）

- STOP-0（模式 Q）：成立；翻轉 0 格。
- STOP-1（模式 L）：STOP-1 不成立（）
- 模式 L：STOP-2a：每一格 A 對 S 都好不到 5% → GPU 忙碌這個軸沒有價值（退回前作）
- 模式 Q：STOP-2a：每一格 A 對 S 都好不到 5% → GPU 忙碌這個軸沒有價值（退回前作）
- 模式 X：STOP-2c：介於中間（A 對 F 幾何平均 6.5%；A 對 S >15% 的格子 1%）→ 停下來問使用者
- T-a：False；T-b：False；T-c：False；T-d：True
- E-L3 llama|lr_r24：新成本項（重建在忙時比反量化敏感，或和搬運同量級）（重建忙/閒 2.79×、INT4 1.34×、忙時重建÷H2D 0.37）
- E-L3 llama|lr_r40：新成本項（重建在忙時比反量化敏感，或和搬運同量級）（重建忙/閒 2.80×、INT4 1.34×、忙時重建÷H2D 0.30）
- E-L3 qwen|lr_r24：中間（照實報，不下結論）（重建忙/閒 2.78×、INT4 2.05×、忙時重建÷H2D 0.25）
- E-L3 qwen|lr_r40：中間（照實報，不下結論）（重建忙/閒 2.64×、INT4 2.05×、忙時重建÷H2D 0.19）

validate：要跑——模式 ['X'] 判 GO_N5（或 X 的 T-d 成立）；挑了 8 格

格點版本（L5c §5.4 的連線格點，只當敏感度，不拿來判定）：L：STOP-2a；Q：STOP-2a；X：STOP-2c

## 聚合（每個模式；gain＝1 − T_A ÷ T_基準 的幾何平均）

| 模式 | 格子 | A 對 D | A 對 F | A 對 S | A 對 S ≥5% 的格子 | >15% | F（每個模型） |
|:--|--:|--:|--:|--:|--:|--:|:--|
| L | 68 | +10.1% | +10.1% | +0.0% | 0.00 | 0.00 | {'llama': 'raw', 'qwen': 'raw'} |
| Q | 68 | +56.6% | +0.0% | +0.0% | 0.00 | 0.00 | {'llama': 'int4_kivi@oneshot', 'qwen': 'int4_kivi@oneshot'} |
| X | 68 | +73.3% | +6.5% | +0.9% | 0.06 | 0.01 | {'llama': 'k4v2@perblock', 'qwen': 'xkv_k22v34_int4@oneshot'} |

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
| qwen | b16 | 2.77 | 0 | xkv_k22v34_int4@oneshot | 55.55 | xkv_k28v42_int4@oneshot | +24.2% | xkv_k22v34_int4@oneshot | xkv_k22v34_int4@oneshot | +88.3% | +0.0% | +0.0% |
| qwen | b16 | 2.77 | 0.4 | xkv_k22v34_int4@oneshot | 57.12 | xkv_k28v42_int4@oneshot | +24.0% | xkv_k22v34_int4@oneshot | xkv_k22v34_int4@oneshot | +89.5% | +0.0% | +0.0% |
| qwen | b16 | 2.77 | 0.7 | xkv_k22v34_int4@oneshot | 59.01 | xkv_k28v42_int4@oneshot | +24.7% | xkv_k22v34_int4@oneshot | xkv_k22v34_int4@oneshot | +89.6% | +0.0% | +0.0% |
| qwen | b16 | 2.77 | 0.9 | xkv_k22v34_int4@oneshot | 77.11 | xkv_k28v42_int4@oneshot | +10.4% | xkv_k22v34_int4@oneshot | xkv_k22v34_int4@oneshot | +87.5% | +0.0% | +0.0% |
| qwen | b32 | 0.763 | 0 | xkv_k22v34_int4@oneshot | 181.43 | xkv_k22v34_int4@perblock | +16.1% | xkv_k22v34_int4@oneshot | xkv_k22v34_int4@oneshot | +84.2% | +0.0% | +0.0% |
| qwen | b32 | 0.763 | 0.4 | xkv_k22v34_int4@oneshot | 185.58 | xkv_k22v34_int4@perblock | +19.0% | xkv_k22v34_int4@oneshot | xkv_k22v34_int4@oneshot | +86.3% | +0.0% | +0.0% |
| qwen | b32 | 0.763 | 0.7 | xkv_k22v34_int4@oneshot | 195.23 | xkv_k22v34_int4@perblock | +6.7% | xkv_k22v34_int4@oneshot | xkv_k22v34_int4@oneshot | +87.0% | +0.0% | +0.0% |
| qwen | b32 | 0.763 | 0.9 | xkv_k22v34_int4@oneshot | 197.38 | xkv_k22v34_int4@perblock | +21.0% | xkv_k22v34_int4@oneshot | xkv_k22v34_int4@oneshot | +88.5% | +0.0% | +0.0% |
| qwen | b48 | 0.217 | 0 | xkv_k22v34_int4@oneshot | 510.13 | xkv_k22v34_int4@perblock | +0.0% | xkv_k22v34_int4@oneshot | xkv_k22v34_int4@oneshot | +74.6% | +0.0% | +0.0% |
| qwen | b48 | 0.217 | 0.4 | xkv_k22v34_int4@oneshot | 545.40 | xkv_k22v34_int4@perblock | +1.2% | xkv_k22v34_int4@oneshot | xkv_k22v34_int4@oneshot | +79.3% | +0.0% | +0.0% |
| qwen | b48 | 0.217 | 0.7 | xkv_k22v34_int4@oneshot | 568.53 | xkv_k22v34_int4@perblock | +11.7% | xkv_k22v34_int4@oneshot | xkv_k22v34_int4@oneshot | +81.9% | +0.0% | +0.0% |
| qwen | b48 | 0.217 | 0.9 | xkv_k22v34_int4@oneshot | 619.06 | xkv_k22v34_int4@perblock | +0.5% | xkv_k22v34_int4@oneshot | xkv_k22v34_int4@oneshot | +83.6% | +0.0% | +0.0% |
| qwen | cpu_pinned | — | 0 | xkv_k22v34@oneshot | 13.63 | xkv_k22v34_int4@oneshot | +4.8% | int4_kivi@oneshot | xkv_k22v34_int4@oneshot | +61.3% | +4.5% | +11.8% |
| qwen | cpu_pinned | — | 0.4 | int4_kivi@oneshot | 16.48 | int4_kivi@perblock | +17.4% | int4_kivi@oneshot | xkv_k22v34_int4@oneshot | +61.9% | +30.2% | +0.0% |
| qwen | cpu_pinned | — | 0.7 | int4_kivi@oneshot | 17.85 | int8_token@oneshot | +22.6% | int4_kivi@oneshot | xkv_k22v34_int4@oneshot | +66.1% | +54.4% | +0.0% |
| qwen | cpu_pinned | — | 0.9 | int4_kivi@oneshot | 18.67 | int8_token@oneshot | +21.3% | int4_kivi@oneshot | xkv_k22v34_int4@oneshot | +67.2% | +75.8% | +0.0% |
| qwen | nfs_meas | 0.383 | 0 | xkv_k22v34_int4@oneshot | 335.71 | xkv_k22v34_int4@perblock | +0.0% | xkv_k22v34_int4@oneshot | xkv_k22v34_int4@oneshot | +79.6% | +0.0% | +0.0% |
| qwen | nfs_meas | 0.383 | 0.4 | xkv_k22v34_int4@oneshot | 355.88 | xkv_k22v34_int4@perblock | +0.0% | xkv_k22v34_int4@oneshot | xkv_k22v34_int4@oneshot | +82.2% | +0.0% | +0.0% |
| qwen | nfs_meas | 0.383 | 0.7 | xkv_k22v34_int4@oneshot | 379.87 | xkv_k22v34_int4@perblock | +0.0% | xkv_k22v34_int4@oneshot | xkv_k22v34_int4@oneshot | +83.9% | +0.0% | +0.0% |
| qwen | nfs_meas | 0.383 | 0.9 | xkv_k22v34_int4@oneshot | 368.88 | xkv_k22v34_int4@perblock | +15.5% | xkv_k22v34_int4@oneshot | xkv_k22v34_int4@oneshot | +86.3% | +0.0% | +0.0% |
| qwen | nfs_param | 0.331 | 0 | xkv_k22v34_int4@oneshot | 368.90 | xkv_k22v34_int4@perblock | +0.7% | xkv_k22v34_int4@oneshot | xkv_k22v34_int4@oneshot | +78.7% | +0.0% | +0.0% |
| qwen | nfs_param | 0.331 | 0.4 | xkv_k22v34_int4@oneshot | 401.46 | xkv_k22v34_int4@perblock | +0.0% | xkv_k22v34_int4@oneshot | xkv_k22v34_int4@oneshot | +81.3% | +0.0% | +0.0% |
| qwen | nfs_param | 0.331 | 0.7 | xkv_k22v34_int4@oneshot | 392.59 | xkv_k22v34_int4@perblock | +12.3% | xkv_k22v34_int4@oneshot | xkv_k22v34_int4@oneshot | +84.7% | +0.0% | +0.0% |
| qwen | nfs_param | 0.331 | 0.9 | xkv_k22v34_int4@oneshot | 442.95 | xkv_k22v34_int4@perblock | +1.5% | xkv_k22v34_int4@oneshot | xkv_k22v34_int4@oneshot | +85.0% | +0.0% | +0.0% |
| qwen | ssd_meas | 7.87 | 0 | xkv_k22v34_int4@oneshot | 22.00 | xkv_k28v42_int4@oneshot | +21.0% | xkv_k22v34_int4@oneshot | xkv_k22v34_int4@oneshot | +89.6% | +0.0% | +0.0% |
| qwen | ssd_meas | 7.87 | 0.4 | xkv_k22v34_int4@oneshot | 23.62 | xkv_k28v42_int4@oneshot | +22.7% | xkv_k22v34_int4@oneshot | xkv_k22v34_int4@oneshot | +89.4% | +0.0% | +0.0% |
| qwen | ssd_meas | 7.87 | 0.7 | xkv_k22v34_int4@oneshot | 39.16 | xkv_k28v42_int4@oneshot | +9.1% | xkv_k22v34_int4@oneshot | xkv_k22v34_int4@oneshot | +81.2% | +0.0% | +0.0% |
| qwen | ssd_meas | 7.87 | 0.9 | xkv_k22v34@oneshot | 59.66 | int4_kivi@oneshot | +19.8% | xkv_k22v34_int4@oneshot | xkv_k22v34_int4@oneshot | +72.2% | +22.6% | +22.6% |
| qwen | ssd_param | 6.98 | 0 | xkv_k22v34_int4@oneshot | 23.93 | xkv_k28v42_int4@oneshot | +25.1% | xkv_k22v34_int4@oneshot | xkv_k22v34_int4@oneshot | +90.1% | +0.0% | +0.0% |
| qwen | ssd_param | 6.98 | 0.4 | xkv_k22v34_int4@oneshot | 25.88 | xkv_k28v42_int4@oneshot | +23.2% | xkv_k22v34_int4@oneshot | xkv_k22v34_int4@oneshot | +88.6% | +0.0% | +0.0% |
| qwen | ssd_param | 6.98 | 0.7 | xkv_k22v34_int4@oneshot | 39.16 | xkv_k28v42_int4@oneshot | +9.1% | xkv_k22v34_int4@oneshot | xkv_k22v34_int4@oneshot | +84.4% | +0.0% | +0.0% |
| qwen | ssd_param | 6.98 | 0.9 | xkv_k22v34@oneshot | 66.21 | xkv_k22v34_int4@oneshot | +16.5% | xkv_k22v34_int4@oneshot | xkv_k22v34_int4@oneshot | +73.6% | +14.1% | +14.1% |
| qwen | vllm_cpu | — | 0 | xkv_k22v34@oneshot | 796.46 | xkv_k22v34_int4@oneshot | +0.1% | int4_kivi@oneshot | xkv_k22v34_int4@oneshot | +2.6% | +0.1% | +0.2% |
| qwen | vllm_cpu | — | 0.7 | int4_kivi@oneshot | 797.53 | int8_token@oneshot | +0.5% | int4_kivi@oneshot | xkv_k22v34_int4@oneshot | +4.2% | +2.6% | +0.0% |

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

## micro 跨 run 漂移（10/21：多個 micro 目錄合併；錨點＝重疊的 segment，用第一個目錄的值，這裡只看差多少）

差 >15% 打 ⚠（不停；那個 duty 下兩次 run 拼起來的比較要打折看）。

| 模型 | duty | segment | op | 第一個 (ms) | 後來 (ms) | 後÷前 | run |
|:--|--:|:--|:--|--:|--:|--:|:--|
| llama | 0 | bf16 | h2d_region | 39.671 | 39.637 | 0.999 | 20261020-212244-m16-x3-micro-llama → 20261021-075914-m16-x3-micro2-llama |
| llama | 0 | int4_kivi | h2d_stack | 13.678 | 13.691 | 1.001 | 20261020-212244-m16-x3-micro-llama → 20261021-075914-m16-x3-micro2-llama |
| llama | 0 | int4_kivi | dq_perblock | 2.248 | 2.215 | 0.985 | 20261020-212244-m16-x3-micro-llama → 20261021-075914-m16-x3-micro2-llama |
| llama | 0 | int4_kivi | dq_oneshot | 1.694 | 1.691 | 0.999 | 20261020-212244-m16-x3-micro-llama → 20261021-075914-m16-x3-micro2-llama |
| llama | 0 | xkv_g4_f8 | h2d_region | 7.297 | 7.260 | 0.995 | 20261020-212244-m16-x3-micro-llama → 20261021-075914-m16-x3-micro2-llama |
| llama | 0 | xkv_g4_f8 | dq_oneshot | 3.962 | 3.933 | 0.993 | 20261020-212244-m16-x3-micro-llama → 20261021-075914-m16-x3-micro2-llama |
| llama | 0.4 | bf16 | h2d_region | 49.945 | 49.866 | 0.998 | 20261020-212244-m16-x3-micro-llama → 20261021-075914-m16-x3-micro2-llama |
| llama | 0.4 | int4_kivi | h2d_stack | 14.262 | 14.337 | 1.005 | 20261020-212244-m16-x3-micro-llama → 20261021-075914-m16-x3-micro2-llama |
| llama | 0.4 | int4_kivi | dq_perblock | 2.991 | 3.330 | 1.113 | 20261020-212244-m16-x3-micro-llama → 20261021-075914-m16-x3-micro2-llama |
| llama | 0.4 | int4_kivi | dq_oneshot | 2.040 | 2.036 | 0.998 | 20261020-212244-m16-x3-micro-llama → 20261021-075914-m16-x3-micro2-llama |
| llama | 0.4 | xkv_g4_f8 | h2d_region | 10.809 | 11.105 | 1.027 | 20261020-212244-m16-x3-micro-llama → 20261021-075914-m16-x3-micro2-llama |
| llama | 0.4 | xkv_g4_f8 | dq_oneshot | 5.113 | 5.132 | 1.004 | 20261020-212244-m16-x3-micro-llama → 20261021-075914-m16-x3-micro2-llama |
| llama | 0.7 | bf16 | h2d_region | 57.710 | 57.317 | 0.993 | 20261020-212244-m16-x3-micro-llama → 20261021-075914-m16-x3-micro2-llama |
| llama | 0.7 | int4_kivi | h2d_stack | 14.689 | 14.821 | 1.009 | 20261020-212244-m16-x3-micro-llama → 20261021-075914-m16-x3-micro2-llama |
| llama | 0.7 | int4_kivi | dq_perblock | 5.052 | 4.826 | 0.955 | 20261020-212244-m16-x3-micro-llama → 20261021-075914-m16-x3-micro2-llama |
| llama | 0.7 | int4_kivi | dq_oneshot | 2.265 | 2.213 | 0.977 | 20261020-212244-m16-x3-micro-llama → 20261021-075914-m16-x3-micro2-llama |
| llama | 0.7 | xkv_g4_f8 | h2d_region | 17.379 | 17.167 | 0.988 | 20261020-212244-m16-x3-micro-llama → 20261021-075914-m16-x3-micro2-llama |
| llama | 0.7 | xkv_g4_f8 | dq_oneshot | 6.006 | 6.284 | 1.046 | 20261020-212244-m16-x3-micro-llama → 20261021-075914-m16-x3-micro2-llama |
| llama | 0.9 | bf16 | h2d_region | 64.850 | 66.394 | 1.024 | 20261020-212244-m16-x3-micro-llama → 20261021-075914-m16-x3-micro2-llama |
| llama | 0.9 | int4_kivi | h2d_stack | 14.921 | 15.180 | 1.017 | 20261020-212244-m16-x3-micro-llama → 20261021-075914-m16-x3-micro2-llama |
| llama | 0.9 | int4_kivi | dq_perblock | 8.723 | 8.480 | 0.972 | 20261020-212244-m16-x3-micro-llama → 20261021-075914-m16-x3-micro2-llama |
| llama | 0.9 | int4_kivi | dq_oneshot | 2.425 | 2.468 | 1.018 | 20261020-212244-m16-x3-micro-llama → 20261021-075914-m16-x3-micro2-llama |
| llama | 0.9 | xkv_g4_f8 | h2d_region | 23.708 | 23.633 | 0.997 | 20261020-212244-m16-x3-micro-llama → 20261021-075914-m16-x3-micro2-llama |
| llama | 0.9 | xkv_g4_f8 | dq_oneshot | 7.082 | 7.087 | 1.001 | 20261020-212244-m16-x3-micro-llama → 20261021-075914-m16-x3-micro2-llama |
| llama | 0 | bf16 | h2d_region | 39.671 | 39.679 | 1.000 | 20261020-212244-m16-x3-micro-llama → 20261021-080628-m16-x3-wcost-llama |
| qwen | 0 | bf16 | h2d_region | 17.584 | 17.577 | 1.000 | 20261020-213232-m16-x3-micro-qwen → 20261021-080256-m16-x3-micro2-qwen |
| qwen | 0 | int4_kivi | h2d_stack | 7.132 | 7.082 | 0.993 | 20261020-213232-m16-x3-micro-qwen → 20261021-080256-m16-x3-micro2-qwen |
| qwen | 0 | int4_kivi | dq_perblock | 2.211 | 2.205 | 0.997 | 20261020-213232-m16-x3-micro-qwen → 20261021-080256-m16-x3-micro2-qwen |
| qwen | 0 | int4_kivi | dq_oneshot | 0.594 | 0.593 | 0.999 | 20261020-213232-m16-x3-micro-qwen → 20261021-080256-m16-x3-micro2-qwen |
| qwen | 0 | xkv_g4_f8 | h2d_region | 3.643 | 3.623 | 0.994 | 20261020-213232-m16-x3-micro-qwen → 20261021-080256-m16-x3-micro2-qwen |
| qwen | 0 | xkv_g4_f8 | dq_oneshot | 1.376 | 1.383 | 1.004 | 20261020-213232-m16-x3-micro-qwen → 20261021-080256-m16-x3-micro2-qwen |
| qwen | 0.4 | bf16 | h2d_region | 21.603 | 21.536 | 0.997 | 20261020-213232-m16-x3-micro-qwen → 20261021-080256-m16-x3-micro2-qwen |
| qwen | 0.4 | int4_kivi | h2d_stack | 7.393 | 7.337 | 0.992 | 20261020-213232-m16-x3-micro-qwen → 20261021-080256-m16-x3-micro2-qwen |
| qwen | 0.4 | int4_kivi | dq_perblock | 2.280 | 2.262 | 0.992 | 20261020-213232-m16-x3-micro-qwen → 20261021-080256-m16-x3-micro2-qwen |
| qwen | 0.4 | int4_kivi | dq_oneshot | 0.845 | 0.837 | 0.990 | 20261020-213232-m16-x3-micro-qwen → 20261021-080256-m16-x3-micro2-qwen |
| qwen | 0.4 | xkv_g4_f8 | h2d_region | 6.551 | 5.989 | 0.914 | 20261020-213232-m16-x3-micro-qwen → 20261021-080256-m16-x3-micro2-qwen |
| qwen | 0.4 | xkv_g4_f8 | dq_oneshot | 1.893 | 2.029 | 1.071 | 20261020-213232-m16-x3-micro-qwen → 20261021-080256-m16-x3-micro2-qwen |
| qwen | 0.7 | bf16 | h2d_region | 26.291 | 26.300 | 1.000 | 20261020-213232-m16-x3-micro-qwen → 20261021-080256-m16-x3-micro2-qwen |
| qwen | 0.7 | int4_kivi | h2d_stack | 7.706 | 7.588 | 0.985 | 20261020-213232-m16-x3-micro-qwen → 20261021-080256-m16-x3-micro2-qwen |
| qwen | 0.7 | int4_kivi | dq_perblock | 3.803 | 3.841 | 1.010 | 20261020-213232-m16-x3-micro-qwen → 20261021-080256-m16-x3-micro2-qwen |
| qwen | 0.7 | int4_kivi | dq_oneshot | 1.219 | 1.172 | 0.962 | 20261020-213232-m16-x3-micro-qwen → 20261021-080256-m16-x3-micro2-qwen |
| qwen | 0.7 | xkv_g4_f8 | h2d_region | 10.391 | 9.909 | 0.954 | 20261020-213232-m16-x3-micro-qwen → 20261021-080256-m16-x3-micro2-qwen |
| qwen | 0.7 | xkv_g4_f8 | dq_oneshot | 2.697 | 2.655 | 0.985 | 20261020-213232-m16-x3-micro-qwen → 20261021-080256-m16-x3-micro2-qwen |
| qwen | 0.9 | bf16 | h2d_region | 28.479 | 28.631 | 1.005 | 20261020-213232-m16-x3-micro-qwen → 20261021-080256-m16-x3-micro2-qwen |
| qwen | 0.9 | int4_kivi | h2d_stack | 7.787 | 7.629 | 0.980 | 20261020-213232-m16-x3-micro-qwen → 20261021-080256-m16-x3-micro2-qwen |
| qwen | 0.9 | int4_kivi | dq_perblock | 7.952 | 9.253 | 1.164 ⚠ | 20261020-213232-m16-x3-micro-qwen → 20261021-080256-m16-x3-micro2-qwen |
| qwen | 0.9 | int4_kivi | dq_oneshot | 1.547 | 1.556 | 1.006 | 20261020-213232-m16-x3-micro-qwen → 20261021-080256-m16-x3-micro2-qwen |
| qwen | 0.9 | xkv_g4_f8 | h2d_region | 19.947 | 18.552 | 0.930 | 20261020-213232-m16-x3-micro-qwen → 20261021-080256-m16-x3-micro2-qwen |
| qwen | 0.9 | xkv_g4_f8 | dq_oneshot | 3.354 | 3.518 | 1.049 | 20261020-213232-m16-x3-micro-qwen → 20261021-080256-m16-x3-micro2-qwen |
| qwen | 0 | bf16 | h2d_region | 17.584 | 17.570 | 0.999 | 20261020-213232-m16-x3-micro-qwen → 20261021-082103-m16-x3-wcost-qwen |

## 寫入路徑成本：xKV 壓縮（10/21；不進還原決策圖）

每題一次：K、V 兩半的 torch.svd_lowrank（每 4 層一個窗口、K 先逆 RoPE；niter、多取幾個、seed 照 m16_xstack）＋係數量化。載入區形狀（最後 n 個 block）。「÷ 重算」＝除以同一段閒時重算的總時間（bw.json，只是比例尺）。

| 模型 | 段 | 組合 | duty | SVD K (ms) | SVD V (ms) | 量化 (ms) | 合計 (ms) | 每 block (ms) | 位元組比 | ÷ 重算 |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| llama | xkv_k11v17 | xl:k11:v17 | 0 | 3994.2 | 6779.0 | 0.2 | 10773.4 | 336.7 | 0.1367 | 4.75 |
| llama | xkv_k11v17 | xl:k11:v17 | 0.9 | 12650.2 | 20447.3 | 0.2 | 33097.8 | 1034.3 | 0.1367 | 14.59 |
| llama | xkv_k22v34 | xl:k22:v34 | 0 | 9400.5 | 15229.7 | 0.3 | 24630.5 | 769.7 | 0.2734 | 10.86 |
| llama | xkv_k22v34 | xl:k22:v34 | 0.9 | 27576.0 | 44321.1 | 0.6 | 71897.8 | 2246.8 | 0.2734 | 31.70 |
| llama | xkv_k28v42 | xl:k28:v42 | 0 | 12241.9 | 19041.4 | 0.4 | 31283.7 | 977.6 | 0.3418 | 13.79 |
| llama | xkv_k28v42 | xl:k28:v42 | 0.9 | 35335.8 | 55467.1 | 0.4 | 90803.4 | 2837.6 | 0.3418 | 40.04 |
| llama | xkv_k11v17_int4 | xl:k11:v17+q:int4_kivi | 0 | 3994.2 | 6779.0 | 1.8 | 10775.0 | 336.7 | 0.0615 | 4.75 |
| llama | xkv_k11v17_int4 | xl:k11:v17+q:int4_kivi | 0.9 | 12650.2 | 20447.3 | 6.8 | 33104.4 | 1034.5 | 0.0615 | 14.60 |
| llama | xkv_k22v34_int4 | xl:k22:v34+q:int4_kivi | 0 | 9400.5 | 15229.7 | 3.4 | 24633.6 | 769.8 | 0.1230 | 10.86 |
| llama | xkv_k22v34_int4 | xl:k22:v34+q:int4_kivi | 0.9 | 27576.0 | 44321.1 | 9.7 | 71906.8 | 2247.1 | 0.1230 | 31.71 |
| llama | xkv_k28v42_int4 | xl:k28:v42+q:int4_kivi | 0 | 12241.9 | 19041.4 | 4.1 | 31287.5 | 977.7 | 0.1538 | 13.80 |
| llama | xkv_k28v42_int4 | xl:k28:v42+q:int4_kivi | 0.9 | 35335.8 | 55467.1 | 10.2 | 90813.1 | 2837.9 | 0.1538 | 40.05 |
| llama | xkv_k22v34_int8 | xl:k22:v34+q:int8_token | 0 | 9400.5 | 15229.7 | 2.6 | 24632.8 | 769.8 | 0.1646 | 10.86 |
| llama | xkv_k22v34_int8 | xl:k22:v34+q:int8_token | 0.9 | 27576.0 | 44321.1 | 5.4 | 71902.5 | 2247.0 | 0.1646 | 31.71 |
| qwen | xkv_k11v17 | xl:k11:v17 | 0 | 1184.0 | 2733.9 | 0.1 | 3918.0 | 122.4 | 0.1230 | 2.03 |
| qwen | xkv_k11v17 | xl:k11:v17 | 0.9 | 5424.0 | 9423.6 | 0.1 | 14847.7 | 464.0 | 0.1230 | 7.69 |
| qwen | xkv_k22v34 | xl:k22:v34 | 0 | 3558.6 | 6019.1 | 0.2 | 9577.9 | 299.3 | 0.2461 | 4.96 |
| qwen | xkv_k22v34 | xl:k22:v34 | 0.9 | 11800.3 | 19086.3 | 0.2 | 30886.8 | 965.2 | 0.2461 | 15.99 |
| qwen | xkv_k28v42 | xl:k28:v42 | 0 | 5093.4 | 7641.2 | 0.2 | 12734.8 | 398.0 | 0.3076 | 6.59 |
| qwen | xkv_k28v42 | xl:k28:v42 | 0.9 | 16558.3 | 23916.8 | 0.2 | 40475.3 | 1264.9 | 0.3076 | 20.96 |
| qwen | xkv_k11v17_int4 | xl:k11:v17+q:int4_kivi | 0 | 1184.0 | 2733.9 | 1.0 | 3918.9 | 122.5 | 0.0479 | 2.03 |
| qwen | xkv_k11v17_int4 | xl:k11:v17+q:int4_kivi | 0.9 | 5424.0 | 9423.6 | 4.2 | 14851.8 | 464.1 | 0.0479 | 7.69 |
| qwen | xkv_k22v34_int4 | xl:k22:v34+q:int4_kivi | 0 | 3558.6 | 6019.1 | 1.6 | 9579.3 | 299.4 | 0.0957 | 4.96 |
| qwen | xkv_k22v34_int4 | xl:k22:v34+q:int4_kivi | 0.9 | 11800.3 | 19086.3 | 4.9 | 30891.5 | 965.4 | 0.0957 | 16.00 |
| qwen | xkv_k28v42_int4 | xl:k28:v42+q:int4_kivi | 0 | 5093.4 | 7641.2 | 1.9 | 12736.5 | 398.0 | 0.1196 | 6.60 |
| qwen | xkv_k28v42_int4 | xl:k28:v42+q:int4_kivi | 0.9 | 16558.3 | 23916.8 | 6.9 | 40482.0 | 1265.1 | 0.1196 | 20.96 |
| qwen | xkv_k22v34_int8 | xl:k22:v34+q:int8_token | 0 | 3558.6 | 6019.1 | 1.2 | 9578.9 | 299.3 | 0.1377 | 4.96 |
| qwen | xkv_k22v34_int8 | xl:k22:v34+q:int8_token | 0.9 | 11800.3 | 19086.3 | 2.9 | 30889.5 | 965.3 | 0.1377 | 15.99 |

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
- 跨 run 漂移：1 個錨點的兩次 micro 差 >15%（見報告的漂移表）
- llama：micro 由 3 個目錄合併（results/m16_stage3_mi300x/x3_time、results/m16_stage3_mi300x/x3_time/v2、results/m16_stage3_mi300x/x3_time/v2/wcost；第一個為主，後面只補沒有的 segment）
- qwen：micro 由 3 個目錄合併（results/m16_stage3_mi300x/x3_time、results/m16_stage3_mi300x/x3_time/v2、results/m16_stage3_mi300x/x3_time/v2/wcost；第一個為主，後面只補沒有的 segment）
- 組成模型和第二階段實測差 >15% 的格子見「組成模型對第二階段實測」表；那種動作的預測要先打折看
