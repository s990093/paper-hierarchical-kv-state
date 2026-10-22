# X3 翻轉格子（run 20261022-183752-m16-x3gridmem-report）

來源：20261022-175930-m16-x3gridmem-validate；144／144 列可用（污染 0）

**判定：NO_FLIP（沒有任何一條翻轉）**（翻轉 0 條）

規則：每條（模型, 層）：duty 0 與 0.9 的最快動作不同，而且兩端最快的都比對方的選擇快 ≥5% → 翻轉。兩個模型都至少一條翻轉 → FLIP_REPRODUCED；只有一個 → FLIP_PARTIAL；都沒有 → NO_FLIP。另報：依情況選（每個 duty 各挑最快）對「固定最好的一種」的幾何平均好處。污染列不算。

## llama cpu_pinned：翻轉 否；固定最好＝xkv_k11v17_int4@oneshot；依情況選好 0.0%

| duty | raw | int4_kivi@oneshot | k4v2@perblock | xkv_k22v34@oneshot | xkv_k22v34_int4@oneshot | xkv_k11v17_int4@oneshot | 最快 |
|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.0 | 0.0837 | 0.0331 | 0.0325 | 0.0342 | 0.0287 | 0.0211 | xkv_k11v17_int4@oneshot |
| 0.4 | 0.0855 | 0.0336 | 0.0343 | 0.0374 | 0.0329 | 0.0239 | xkv_k11v17_int4@oneshot |
| 0.7 | 0.0872 | 0.0347 | 0.0354 | 0.0414 | 0.0356 | 0.0264 | xkv_k11v17_int4@oneshot |
| 0.9 | 0.0880 | 0.0351 | 0.0359 | 0.0420 | 0.0376 | 0.0290 | xkv_k11v17_int4@oneshot |

## llama b16：翻轉 否；固定最好＝xkv_k11v17_int4@oneshot；依情況選好 0.0%

| duty | raw | int4_kivi@oneshot | k4v2@perblock | xkv_k22v34@oneshot | xkv_k22v34_int4@oneshot | xkv_k11v17_int4@oneshot | 最快 |
|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.0 | 0.5735 | 0.2252 | 0.2256 | 0.1988 | 0.0855 | 0.0453 | xkv_k11v17_int4@oneshot |
| 0.4 | 0.6176 | 0.2442 | 0.2209 | 0.2006 | 0.0870 | 0.0466 | xkv_k11v17_int4@oneshot |
| 0.7 | 0.6913 | 0.2423 | 0.2286 | 0.2026 | 0.0888 | 0.0475 | xkv_k11v17_int4@oneshot |
| 0.9 | 0.7297 | 0.2389 | 0.2255 | 0.2040 | 0.0910 | 0.0489 | xkv_k11v17_int4@oneshot |

## llama ssd_meas：翻轉 否；固定最好＝xkv_k11v17_int4@oneshot；依情況選好 0.0%

| duty | raw | int4_kivi@oneshot | k4v2@perblock | xkv_k22v34@oneshot | xkv_k22v34_int4@oneshot | xkv_k11v17_int4@oneshot | 最快 |
|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.0 | 0.4143 | 0.1585 | 0.1291 | 0.1372 | 0.0615 | 0.0333 | xkv_k11v17_int4@oneshot |
| 0.4 | 0.4587 | 0.1649 | 0.1334 | 0.1392 | 0.0642 | 0.0353 | xkv_k11v17_int4@oneshot |
| 0.7 | 0.4901 | 0.1651 | 0.1294 | 0.1406 | 0.0640 | 0.0367 | xkv_k11v17_int4@oneshot |
| 0.9 | 0.4820 | 0.1617 | 0.1293 | 0.1419 | 0.0664 | 0.0375 | xkv_k11v17_int4@oneshot |

## qwen cpu_pinned：翻轉 否；固定最好＝xkv_k11v17_int4@oneshot；依情況選好 0.8%

| duty | raw | int4_kivi@oneshot | k4v2@perblock | xkv_k22v34@oneshot | xkv_k22v34_int4@oneshot | xkv_k11v17_int4@oneshot | 最快 |
|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.0 | 0.0390 | 0.0175 | 0.0196 | 0.0147 | 0.0155 | 0.0132 | xkv_k11v17_int4@oneshot |
| 0.4 | 0.0395 | 0.0180 | 0.0205 | 0.0162 | 0.0177 | 0.0155 | xkv_k11v17_int4@oneshot |
| 0.7 | 0.0403 | 0.0187 | 0.0212 | 0.0175 | 0.0200 | 0.0175 | xkv_k22v34@oneshot |
| 0.9 | 0.0406 | 0.0190 | 0.0217 | 0.0182 | 0.0214 | 0.0188 | xkv_k22v34@oneshot |

## qwen b16：翻轉 否；固定最好＝xkv_k11v17_int4@oneshot；依情況選好 0.0%

| duty | raw | int4_kivi@oneshot | k4v2@perblock | xkv_k22v34@oneshot | xkv_k22v34_int4@oneshot | xkv_k11v17_int4@oneshot | 最快 |
|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.0 | 0.4789 | 0.1853 | 0.1721 | 0.1506 | 0.0560 | 0.0292 | xkv_k11v17_int4@oneshot |
| 0.4 | 0.5127 | 0.1866 | 0.1707 | 0.1510 | 0.0565 | 0.0301 | xkv_k11v17_int4@oneshot |
| 0.7 | 0.5704 | 0.2010 | 0.1720 | 0.1518 | 0.0582 | 0.0307 | xkv_k11v17_int4@oneshot |
| 0.9 | 0.5537 | 0.2002 | 0.1715 | 0.1523 | 0.0582 | 0.0311 | xkv_k11v17_int4@oneshot |

## qwen ssd_meas：翻轉 否；固定最好＝xkv_k11v17_int4@oneshot；依情況選好 0.0%

| duty | raw | int4_kivi@oneshot | k4v2@perblock | xkv_k22v34@oneshot | xkv_k22v34_int4@oneshot | xkv_k11v17_int4@oneshot | 最快 |
|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.0 | 0.2152 | 0.0729 | 0.0575 | 0.0555 | 0.0222 | 0.0160 | xkv_k11v17_int4@oneshot |
| 0.4 | 0.2203 | 0.0711 | 0.0575 | 0.0564 | 0.0227 | 0.0183 | xkv_k11v17_int4@oneshot |
| 0.7 | 0.2237 | 0.0733 | 0.0575 | 0.0573 | 0.0236 | 0.0204 | xkv_k11v17_int4@oneshot |
| 0.9 | 0.2237 | 0.0734 | 0.0576 | 0.0577 | 0.0249 | 0.0221 | xkv_k11v17_int4@oneshot |

