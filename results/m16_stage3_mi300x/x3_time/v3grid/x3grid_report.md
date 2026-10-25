# X3 翻轉格子（run 20261022-175920-m16-x3grid-report）

來源：20261022-171510-m16-x3grid-validate；144／144 列可用（污染 0）

**判定：FLIP_REPRODUCED（兩個模型都至少一條翻轉）**（翻轉 3 條）

規則：每條（模型, 層）：duty 0 與 0.9 的最快動作不同，而且兩端最快的都比對方的選擇快 ≥5% → 翻轉。兩個模型都至少一條翻轉 → FLIP_REPRODUCED；只有一個 → FLIP_PARTIAL；都沒有 → NO_FLIP。另報：依情況選（每個 duty 各挑最快）對「固定最好的一種」的幾何平均好處。污染列不算。

## llama cpu_pinned：翻轉 是；固定最好＝int4_kivi@oneshot；依情況選好 13.6%

| duty | raw | int4_kivi@oneshot | k4v2@perblock | xkv_k22v34@oneshot | xkv_k22v34_int4@oneshot | xkv_k11v17_int4@oneshot | 最快 |
|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.0 | 0.0834 | 0.0330 | 0.0323 | 0.0342 | 0.0287 | 0.0212 | xkv_k11v17_int4@oneshot |
| 0.4 | 0.1019 | 0.0350 | 0.0425 | 0.0480 | 0.0434 | 0.0304 | xkv_k11v17_int4@oneshot |
| 0.7 | 0.1239 | 0.0356 | 0.0547 | 0.0666 | 0.0679 | 0.0420 | int4_kivi@oneshot |
| 0.9 | 0.1395 | 0.0371 | 0.0672 | 0.0860 | 0.1259 | 0.0836 | int4_kivi@oneshot |

## llama b16：翻轉 否；固定最好＝xkv_k11v17_int4@oneshot；依情況選好 0.0%

| duty | raw | int4_kivi@oneshot | k4v2@perblock | xkv_k22v34@oneshot | xkv_k22v34_int4@oneshot | xkv_k11v17_int4@oneshot | 最快 |
|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.0 | 0.5729 | 0.2250 | 0.1923 | 0.1988 | 0.0854 | 0.0454 | xkv_k11v17_int4@oneshot |
| 0.4 | 0.6364 | 0.2436 | 0.2463 | 0.2047 | 0.1124 | 0.0641 | xkv_k11v17_int4@oneshot |
| 0.7 | 0.6649 | 0.2409 | 0.2614 | 0.2104 | 0.1435 | 0.0842 | xkv_k11v17_int4@oneshot |
| 0.9 | 0.7595 | 0.2439 | 0.2675 | 0.2199 | 0.1773 | 0.1223 | xkv_k11v17_int4@oneshot |

## llama ssd_meas：翻轉 否；固定最好＝xkv_k11v17_int4@oneshot；依情況選好 0.0%

| duty | raw | int4_kivi@oneshot | k4v2@perblock | xkv_k22v34@oneshot | xkv_k22v34_int4@oneshot | xkv_k11v17_int4@oneshot | 最快 |
|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.0 | 0.4141 | 0.1578 | 0.1290 | 0.1372 | 0.0615 | 0.0333 | xkv_k11v17_int4@oneshot |
| 0.4 | 0.4619 | 0.1653 | 0.1478 | 0.1402 | 0.0849 | 0.0438 | xkv_k11v17_int4@oneshot |
| 0.7 | 0.4828 | 0.1643 | 0.1738 | 0.1492 | 0.1222 | 0.0646 | xkv_k11v17_int4@oneshot |
| 0.9 | 0.5097 | 0.1656 | 0.1906 | 0.1610 | 0.1530 | 0.1051 | xkv_k11v17_int4@oneshot |

## qwen cpu_pinned：翻轉 是；固定最好＝int4_kivi@oneshot；依情況選好 5.9%

| duty | raw | int4_kivi@oneshot | k4v2@perblock | xkv_k22v34@oneshot | xkv_k22v34_int4@oneshot | xkv_k11v17_int4@oneshot | 最快 |
|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.0 | 0.0388 | 0.0174 | 0.0193 | 0.0150 | 0.0156 | 0.0136 | xkv_k11v17_int4@oneshot |
| 0.4 | 0.0474 | 0.0182 | 0.0261 | 0.0205 | 0.0212 | 0.0204 | int4_kivi@oneshot |
| 0.7 | 0.0558 | 0.0187 | 0.0420 | 0.0335 | 0.0418 | 0.0409 | int4_kivi@oneshot |
| 0.9 | 0.0596 | 0.0207 | 0.0518 | 0.0498 | 0.0828 | 0.0743 | int4_kivi@oneshot |

## qwen b16：翻轉 否；固定最好＝xkv_k11v17_int4@oneshot；依情況選好 0.0%

| duty | raw | int4_kivi@oneshot | k4v2@perblock | xkv_k22v34@oneshot | xkv_k22v34_int4@oneshot | xkv_k11v17_int4@oneshot | 最快 |
|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.0 | 0.4788 | 0.1853 | 0.1587 | 0.1506 | 0.0560 | 0.0293 | xkv_k11v17_int4@oneshot |
| 0.4 | 0.5269 | 0.2017 | 0.1938 | 0.1526 | 0.0746 | 0.0425 | xkv_k11v17_int4@oneshot |
| 0.7 | 0.6102 | 0.1986 | 0.1916 | 0.1531 | 0.1008 | 0.0621 | xkv_k11v17_int4@oneshot |
| 0.9 | 0.6327 | 0.2025 | 0.2040 | 0.1539 | 0.1309 | 0.1050 | xkv_k11v17_int4@oneshot |

## qwen ssd_meas：翻轉 是；固定最好＝xkv_k11v17_int4@oneshot；依情況選好 5.7%

| duty | raw | int4_kivi@oneshot | k4v2@perblock | xkv_k22v34@oneshot | xkv_k22v34_int4@oneshot | xkv_k11v17_int4@oneshot | 最快 |
|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.0 | 0.2146 | 0.0726 | 0.0574 | 0.0555 | 0.0222 | 0.0161 | xkv_k11v17_int4@oneshot |
| 0.4 | 0.2061 | 0.0731 | 0.0765 | 0.0608 | 0.0348 | 0.0210 | xkv_k11v17_int4@oneshot |
| 0.7 | 0.2236 | 0.0743 | 0.0944 | 0.0639 | 0.0520 | 0.0421 | xkv_k11v17_int4@oneshot |
| 0.9 | 0.2178 | 0.0743 | 0.1042 | 0.0660 | 0.0884 | 0.0834 | xkv_k22v34@oneshot |

