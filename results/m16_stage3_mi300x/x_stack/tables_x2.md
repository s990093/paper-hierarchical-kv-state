# X2：挑出的組合的生成答對率（第 1 輪；判定照 L5d §6.3，MLPerf 門檻只報）

run：20261020-214139-m16-x2-llama-32K-core-s0、20261020-214139-m16-x2-llama-32K-core-s1、20261020-214139-m16-x2-llama-32K-core-s2、20261020-214139-m16-x2-qwen-32K-core-s0、20261020-214139-m16-x2-qwen-32K-core-s1、20261020-214139-m16-x2-qwen-32K-core-s2、20261020-214139-m16-x2-qwen-32K-core-s3、20261020-232731-m16-x2-llama-32K-core-s0、20261020-232731-m16-x2-llama-32K-core-s1、20261020-232731-m16-x2-llama-32K-core-s2、20261020-232731-m16-x2-qwen-32K-core-s3、20261020-233535-m16-x2-qwen-32K-x3-s0、20261020-233535-m16-x2-qwen-32K-x3-s1、20261020-233535-m16-x2-qwen-32K-x3-s2、20261020-233535-m16-x2-qwen-32K-x3-s3、20261021-004556-m16-x2-qwen-32K-ext-s0、20261021-004556-m16-x2-qwen-32K-ext-s1、20261021-004556-m16-x2-qwen-32K-ext-s2、20261021-004556-m16-x2-qwen-32K-ext-s3、20261021-011044-m16-x2-llama-32K-core-s0、20261021-011044-m16-x2-llama-32K-core-s1、20261021-011137-m16-x2-qwen-32K-ext-s0、20261021-011137-m16-x2-qwen-32K-ext-s1、20261021-011137-m16-x2-qwen-32K-ext-s2、20261021-011137-m16-x2-qwen-32K-ext-s3、20261021-020023-m16-x2-llama-32K-x3-s0、20261021-020023-m16-x2-llama-32K-x3-s1、20261021-020023-m16-x2-llama-32K-x3-s2；不齊全的題（沒用）：0

## llama 32K（E0 答對率 79.5%）

| 組合 | 角色 | 位元組比 | 答對率 | 對 E0（百分點，CI） | ε | ≥99% E0 | ≥99.9% E0 |
|:--|:--|--:|--:|--:|:--|:--|:--|
| drop:KNORM:0.875+lr:k32:v32:joint+q:int2_kivi | naive_x1 | 0.0060 | 23.4 | -56.1 [-63.7, -48.5] | 沒過 | 沒過 | 沒過 |
| xl:k11:v17+q:int4_kivi | primary | 0.0481 | 76.8 | -2.7 [-6.4, +0.8] | 沒過 | 沒過 | 沒過 |
| xl:k22:v34+drop:KEYDIFF:0.5+q:int4_kivi | front | 0.0614 | 61.1 | -18.4 [-24.9, -12.5] | 沒過 | 沒過 | 沒過 |
| xl:k5:v9 | best_single|family_best:xlayer | 0.0616 | 39.2 | -40.3 [-48.7, -31.8] | 沒過 | 沒過 | 沒過 |
| drop:RND0:0.9375 | random_drop | 0.0635 | 10.1 | -69.4 [-75.9, -62.5] | 沒過 | 沒過 | 沒過 |
| xl:k22:v34+q:int4_kivi | front|primary | 0.0961 | 77.1 | -2.4 [-7.2, +1.9] | 沒過 | 沒過 | 沒過 |
| lr:k40:v40+q:int4_kivi | naive | 0.0977 | 16.9 | -62.6 [-69.6, -55.4] | 沒過 | 沒過 | 沒過 |
| xl:k28:v42+q:int4_kivi | front | 0.1202 | 77.3 | -2.2 [-5.8, +0.7] | 沒過 | 沒過 | 沒過 |
| xl:k11:v17 | best_single|family_best:xlayer | 0.1233 | 79.8 | +0.3 [-3.2, +4.0] | 沒過 | 過 | 過 |
| lr:k16:v16 | x3_segment | 0.1250 | 3.3 | -77.3 [-85.2, -67.6] | 沒過 | 沒過 | 沒過 |
| drop:RND0:0.875 | random_drop | 0.1270 | 18.0 | -61.5 [-69.0, -53.7] | 沒過 | 沒過 | 沒過 |
| xl:k22:v34+q:int8_token | front|primary | 0.1376 | 79.0 | -0.5 [-4.7, +3.4] | 沒過 | 過 | 沒過 |
| xl:k16:v16 | best_single|family_best:xlayer|x3_segment | 0.1409 | 77.3 | -2.2 [-6.4, +1.8] | 沒過 | 沒過 | 沒過 |
| merge:2+q:int4_kivi | x3_segment | 0.1562 | 10.6 | -70.0 [-78.9, -59.6] | 沒過 | 沒過 | 沒過 |
| lr:k24:v24 | x3_segment | 0.1875 | 6.6 | -74.1 [-82.4, -64.2] | 沒過 | 沒過 | 沒過 |
| q:int2_kivi | x3_segment | 0.1875 | 79.6 | -1.0 [-6.4, +4.7] | 沒過 | 沒過 | 沒過 |
| drop:KEYDIFF:0.25+q:int4_kivi | naive | 0.2331 | 73.9 | -5.6 [-10.3, -1.6] | 沒過 | 沒過 | 沒過 |
| lr:k24+q:kv-bf16-int4_kivi | x3_segment | 0.2500 | 6.6 | -74.1 [-82.4, -64.2] | 沒過 | 沒過 | 沒過 |
| lr:k32:v32 | x3_segment | 0.2500 | 6.6 | -74.1 [-82.4, -64.2] | 沒過 | 沒過 | 沒過 |
| merge:4 | family_best:merge|x3_segment | 0.2500 | 6.6 | -74.1 [-82.4, -64.2] | 沒過 | 沒過 | 沒過 |
| q:kv-int4_kivi-int2_kivi | family_best:quant|x3_segment | 0.2500 | 83.5 | +2.9 [-0.7, +7.4] | 過 | 過 | 過 |
| drop:RND0:0.75 | random_drop | 0.2540 | 31.9 | -47.6 [-55.4, -39.6] | 沒過 | 沒過 | 沒過 |
| xl:k32:v32 | x3_segment | 0.2817 | 78.8 | -1.8 [-7.9, +4.0] | 沒過 | 沒過 | 沒過 |
| lr:k40:v40 | x3_segment | 0.3125 | 20.8 | -59.8 [-68.8, -50.2] | 沒過 | 沒過 | 沒過 |
| q:int4_kivi | anchor_int4|x3_segment | 0.3125 | 78.7 | -0.8 [-3.7, +1.4] | 沒過 | 沒過 | 沒過 |
| lr:k64:v64 | x3_segment | 0.5000 | 67.0 | -13.6 [-22.9, -5.1] | 沒過 | 沒過 | 沒過 |
| merge:2 | x3_segment | 0.5000 | 11.0 | -69.6 [-78.6, -59.1] | 沒過 | 沒過 | 沒過 |
| q:vllm_fp8 | deploy | 0.5000 | 78.8 | -0.7 [-2.4, +0.7] | 沒過 | 過 | 沒過 |
| q:fp8_token | x3_segment | 0.5156 | 80.5 | -0.1 [-1.1, +0.7] | 過 | 過 | 沒過 |
| q:int8_token | x3_segment | 0.5156 | 79.5 | -1.1 [-2.7, +0.0] | 沒過 | 沒過 | 沒過 |
| none | anchor_mask0|x3_segment | 1.0000 | 79.1 | -0.4 [-1.2, +0.2] | 過 | 過 | 沒過 |

## qwen 32K（E0 答對率 75.7%）

| 組合 | 角色 | 位元組比 | 答對率 | 對 E0（百分點，CI） | ε | ≥99% E0 | ≥99.9% E0 |
|:--|:--|--:|--:|--:|:--|:--|:--|
| xl:k11:v17+q:int4_kivi | primary | 0.0411 | 68.9 | -6.8 [-13.5, -0.3] | 沒過 | 沒過 | 沒過 |
| drop:KEYDIFF:0.5+xl:k22:v34+q:int4_kivi | front | 0.0475 | 61.3 | -14.4 [-21.1, -7.9] | 沒過 | 沒過 | 沒過 |
| xl:k22:v34+drop:KEYDIFF:0.5+q:int4_kivi | front | 0.0475 | 60.3 | -15.4 [-22.4, -8.5] | 沒過 | 沒過 | 沒過 |
| xl:k5:v9 | best_single|family_best:xlayer | 0.0582 | 32.5 | -43.2 [-51.5, -34.8] | 沒過 | 沒過 | 沒過 |
| drop:KEYDIFF:0.5+xl:k11:v17 | front | 0.0608 | 56.1 | -19.6 [-27.9, -11.5] | 沒過 | 沒過 | 沒過 |
| lr:k8:v8:joint | family_best:lowrank | 0.0625 | 3.1 | -72.6 [-78.6, -65.9] | 沒過 | 沒過 | 沒過 |
| merge:16 | family_best:merge | 0.0625 | 3.2 | -72.4 [-78.6, -65.7] | 沒過 | 沒過 | 沒過 |
| drop:KEYDIFF:0.9375 | family_best:drop | 0.0635 | 12.8 | -62.9 [-69.7, -55.6] | 沒過 | 沒過 | 沒過 |
| drop:RND0:0.9375 | random_drop | 0.0635 | 8.0 | -67.7 [-74.5, -60.0] | 沒過 | 沒過 | 沒過 |
| lr:k32:v32:joint+q:int4_kivi | front | 0.0781 | 65.5 | -10.2 [-15.7, -4.9] | 沒過 | 沒過 | 沒過 |
| xl:k22:v34+q:int4_kivi | front | 0.0822 | 76.6 | +0.9 [-3.0, +4.9] | 沒過 | 過 | 過 |
| lr:k64:v64+q:int2_kivi | front | 0.0938 | 48.3 | -27.3 [-33.8, -21.0] | 沒過 | 沒過 | 沒過 |
| lr:k40:v40+q:int4_kivi | naive | 0.0977 | 36.3 | -39.4 [-46.9, -31.4] | 沒過 | 沒過 | 沒過 |
| xl:k28:v42+q:int4_kivi | front|primary | 0.1028 | 77.6 | +1.9 [-1.5, +5.5] | 過 | 過 | 過 |
| xl:k11:v17 | best_single|family_best:xlayer | 0.1163 | 70.4 | -5.3 [-11.7, +0.9] | 沒過 | 沒過 | 沒過 |
| lr:k16:v16 | x3_segment | 0.1250 | 3.1 | -72.6 [-78.6, -65.9] | 沒過 | 沒過 | 沒過 |
| lr:k16:v16:joint | family_best:lowrank | 0.1250 | 34.5 | -41.1 [-48.8, -33.6] | 沒過 | 沒過 | 沒過 |
| merge:8 | family_best:merge | 0.1250 | 3.2 | -72.4 [-78.6, -65.7] | 沒過 | 沒過 | 沒過 |
| drop:KEYDIFF:0.875 | family_best:drop | 0.1270 | 21.4 | -54.3 [-61.5, -47.1] | 沒過 | 沒過 | 沒過 |
| drop:RND0:0.875 | random_drop | 0.1270 | 17.3 | -58.3 [-66.0, -50.3] | 沒過 | 沒過 | 沒過 |
| xl:k16:v16 | x3_segment | 0.1329 | 73.4 | -2.2 [-8.2, +3.5] | 沒過 | 沒過 | 沒過 |
| merge:2+q:int4_kivi | x3_segment | 0.1562 | 7.1 | -68.5 [-74.7, -61.8] | 沒過 | 沒過 | 沒過 |
| lr:k24:v24 | x3_segment | 0.1875 | 2.3 | -73.3 [-79.2, -66.8] | 沒過 | 沒過 | 沒過 |
| q:int2_kivi | x3_segment | 0.1875 | 68.5 | -7.1 [-13.2, -1.1] | 沒過 | 沒過 | 沒過 |
| xl:k22:v34 | family_best:xlayer | 0.2326 | 75.6 | -0.1 [-4.3, +4.3] | 沒過 | 過 | 沒過 |
| drop:KEYDIFF:0.25+q:int4_kivi | front|naive | 0.2331 | 67.5 | -8.2 [-13.3, -3.5] | 沒過 | 沒過 | 沒過 |
| lr:k24+q:kv-bf16-int4_kivi | x3_segment | 0.2500 | 3.2 | -72.4 [-78.4, -65.8] | 沒過 | 沒過 | 沒過 |
| lr:k32:v32 | x3_segment | 0.2500 | 13.5 | -62.1 [-68.5, -55.5] | 沒過 | 沒過 | 沒過 |
| lr:k32:v32:joint | family_best:lowrank | 0.2500 | 61.1 | -14.6 [-20.3, -9.2] | 沒過 | 沒過 | 沒過 |
| merge:4 | family_best:merge|x3_segment | 0.2500 | 2.2 | -73.5 [-79.5, -66.9] | 沒過 | 沒過 | 沒過 |
| q:kv-int4_kivi-int2_kivi | best_single|family_best:quant|x3_segment | 0.2500 | 73.5 | -2.2 [-5.9, +1.2] | 沒過 | 沒過 | 沒過 |
| drop:KEYDIFF:0.75 | family_best:drop | 0.2540 | 40.2 | -35.5 [-43.9, -27.3] | 沒過 | 沒過 | 沒過 |
| drop:RND0:0.75 | random_drop | 0.2540 | 33.7 | -42.0 [-50.4, -33.4] | 沒過 | 沒過 | 沒過 |
| drop:KNORM:0.25+q:kv-int8_token-int2_kivi | naive_x1 | 0.2623 | 65.8 | -9.9 [-15.4, -4.6] | 沒過 | 沒過 | 沒過 |
| xl:k32:v32 | x3_segment | 0.2659 | 76.0 | +0.3 [-3.4, +4.2] | 沒過 | 過 | 過 |
| lr:k40:v40 | x3_segment | 0.3125 | 36.5 | -39.2 [-47.0, -30.9] | 沒過 | 沒過 | 沒過 |
| q:int4_kivi | anchor_int4|best_single|x3_segment | 0.3125 | 76.9 | +1.2 [-0.4, +3.1] | 過 | 過 | 過 |
| lr:k64:v64 | x3_segment | 0.5000 | 60.5 | -15.2 [-20.8, -9.6] | 沒過 | 沒過 | 沒過 |
| merge:2 | x3_segment | 0.5000 | 7.6 | -68.1 [-74.4, -61.1] | 沒過 | 沒過 | 沒過 |
| q:vllm_fp8 | deploy_broken | 0.5000 | 18.5 | -57.1 [-65.2, -48.6] | 沒過 | 沒過 | 沒過 |
| q:fp8_head | x3_segment | 0.5000 | 35.8 | -39.9 [-48.1, -31.7] | 沒過 | 沒過 | 沒過 |
| q:int8_token | deploy|x3_segment | 0.5156 | 76.2 | +0.5 [-1.0, +2.3] | 過 | 過 | 過 |
| none | anchor_mask0|x3_segment | 1.0000 | 75.7 | -0.0 [-0.5, +0.5] | 過 | 過 | 過 |

## 給 X3 的可行集（32K；feasible＝ε 判定，MLPerf 99% 只另列）

{"x2_feasible.csv": {"rows": 34, "feasible": 6}, "x2_unmapped.csv": {"rows": 40, "feasible": 1}}

## 判定（事先寫死；32K）

```
{
 "models": {
  "llama": {
   "M1": {
    "bstar_single": 0.25,
    "bstar_combo": null,
    "r": Infinity
   },
   "M1_mlperf0.99": {
    "bstar_single": 0.123264,
    "bstar_combo": 0.137641,
    "r": 1.1166358385254413
   },
   "M1_mlperf0.999": {
    "bstar_single": 0.123264,
    "bstar_combo": null,
    "r": null
   },
   "M2": {
    "0.25": {
     "primary": "xl:k22:v34+q:int8_token",
     "best_single": "xl:k16:v16",
     "naive": "drop:KEYDIFF:0.25+q:int4_kivi",
     "room": true,
     "diff_primary_minus_single": {
      "mean": 0.017113095238095236,
      "lo": -0.0043154761904761925,
      "hi": 0.04151785714285714,
      "p_le0": 0.0633,
      "n": 96
     },
     "supports_nogo": true,
     "naive_minus_primary": {
      "mean": -0.050892857142857136,
      "lo": -0.10312499999999998,
      "hi": -0.001560639880952405,
      "p_le0": 0.9785,
      "n": 96
     },
     "naive_as_good": false,
     "holm": {
      "p": 0.0633,
      "threshold": 0.0125,
      "reject": false
     }
    },
    "0.125": {
     "primary": "xl:k22:v34+q:int4_kivi",
     "best_single": "xl:k11:v17",
     "naive": "lr:k40:v40+q:int4_kivi",
     "room": false,
     "diff_primary_minus_single": {
      "mean": -0.027455357142857146,
      "lo": -0.07455357142857143,
      "hi": 0.013244047619047619,
      "p_le0": 0.8977,
      "n": 96
     },
     "supports_nogo": null,
     "naive_minus_primary": {
      "mean": -0.6017113095238096,
      "lo": -0.6722470238095237,
      "hi": -0.5287909226190478,
      "p_le0": 1.0,
      "n": 96
     },
     "naive_as_good": false,
     "holm": {
      "p": 0.8977,
      "threshold": 0.025,
      "reject": false
     }
    },
    "0.0625": {
     "primary": "xl:k11:v17+q:int4_kivi",
     "best_single": "xl:k5:v9",
     "naive": null,
     "room": true,
     "diff_primary_minus_single": {
      "mean": 0.3763392857142857,
      "lo": 0.28816964285714286,
      "hi": 0.4611607142857143,
      "p_le0": 0.0,
      "n": 96
     },
     "supports_nogo": false,
     "holm": {
      "p": 0.0,
      "threshold": 0.004166666666666667,
      "reject": true
     }
    }
   }
  },
  "qwen": {
   "M1": {
    "bstar_single": 0.3125,
    "bstar_combo": 0.10281,
    "r": 0.328992
   },
   "M1_mlperf0.99": {
    "bstar_single": 0.232639,
    "bstar_combo": 0.082248,
    "r": 0.3535434729344607
   },
   "M1_mlperf0.999": {
    "bstar_single": 0.265873,
    "bstar_combo": 0.082248,
    "r": 0.309350705035863
   },
   "M2": {
    "0.25": {
     "primary": "xl:k28:v42+q:int4_kivi",
     "best_single": "q:kv-int4_kivi-int2_kivi",
     "naive": "drop:KEYDIFF:0.25+q:int4_kivi",
     "room": true,
     "diff_primary_minus_single": {
      "mean": 0.040625,
      "lo": 0.0015625000000000025,
      "hi": 0.08438802083333316,
      "p_le0": 0.0195,
      "n": 96
     },
     "supports_nogo": false,
     "naive_minus_primary": {
      "mean": -0.10052083333333334,
      "lo": -0.16041666666666668,
      "hi": -0.046875,
      "p_le0": 1.0,
      "n": 96
     },
     "naive_as_good": false,
     "holm": {
      "p": 0.0195,
      "threshold": 0.008333333333333333,
      "reject": false
     }
    },
    "0.125": {
     "primary": "xl:k28:v42+q:int4_kivi",
     "best_single": "xl:k11:v17",
     "naive": "lr:k40:v40+q:int4_kivi",
     "room": true,
     "diff_primary_minus_single": {
      "mean": 0.07135416666666666,
      "lo": 0.013020833333333334,
      "hi": 0.13125,
      "p_le0": 0.0081,
      "n": 96
     },
     "supports_nogo": false,
     "naive_minus_primary": {
      "mean": -0.41250000000000003,
      "lo": -0.4864583333333334,
      "hi": -0.3416666666666666,
      "p_le0": 1.0,
      "n": 96
     },
     "naive_as_good": false,
     "holm": {
      "p": 0.0081,
      "threshold": 0.00625,
      "reject": false
     }
    },
    "0.0625": {
     "primary": "xl:k11:v17+q:int4_kivi",
     "best_single": "xl:k5:v9",
     "naive": null,
     "room": true,
     "diff_primary_minus_single": {
      "mean": 0.3640625,
      "lo": 0.27708333333333335,
      "hi": 0.4515624999999999,
      "p_le0": 0.0,
      "n": 96
     },
     "supports_nogo": false,
     "holm": {
      "p": 0.0,
      "threshold": 0.005,
      "reject": true
     }
    }
   }
  }
 },
 "holm": {
  "llama|0.0625": {
   "p": 0.0,
   "threshold": 0.004166666666666667,
   "reject": true
  },
  "qwen|0.0625": {
   "p": 0.0,
   "threshold": 0.005,
   "reject": true
  },
  "qwen|0.125": {
   "p": 0.0081,
   "threshold": 0.00625,
   "reject": false
  },
  "qwen|0.25": {
   "p": 0.0195,
   "threshold": 0.008333333333333333,
   "reject": false
  },
  "llama|0.25": {
   "p": 0.0633,
   "threshold": 0.0125,
   "reject": false
  },
  "llama|0.125": {
   "p": 0.8977,
   "threshold": 0.025,
   "reject": false
  }
 },
 "M1_verdict": "GO",
 "M2_supports_nogo": false,
 "M2_no_room_anywhere": false,
 "completeness_flags": {
  "llama|0.25": false,
  "llama|0.125": false,
  "qwen|0.25": false,
  "qwen|0.125": false
 },
 "n_items_32K": {
  "llama": 96,
  "qwen": 96
 },
 "complete": true
}
```

- **M1**：（L5d §6.3）每個家族在 ε 內（生成答對率對 E0 的配對差 95% CI 下界 ≥ −2 個百分點）能到的最小位元組 B*；r＝B*(最好的組合)／B*(最好的單一方法)。兩個模型都 r > 0.95（沒有組合過 ε 也算）→ NO-GO（疊加沒有加乘）；沒有模型 < 0.85、但有 0.85–0.95 → 停下來問人；任一模型 r < 0.85 → GO（進第三階段）。
- **M2**：（L5d §6.3）每個預算：主要組合（X1 事先選的前緣第 1 名）− 最好的單一方法（X1 選的），生成答對率配對差。E0 − 最好的單一方法 < 2 個百分點的預算記成「沒有空間」；有空間的預算全部「差 < 2 個百分點或 CI 含 0」→ 支持 NO-GO。Holm 校正（3 預算 × 2 模型，單邊 α＝0.025）。
- **completeness**：（L5d §6.3）最好的組合在 M1／M2 贏，但天真疊加（naive）本身已經一樣好（naive − 主要組合的 CI 含 0 或 ≥ −ε）→ 結論改寫成「疊加有用，但不需要聯合分配」。
- **MLPerf**：（L6 §5.2，只報、不改判定）MLPerf 式門檻：答對率 ≥ E0 的 99%（以及 99.9%），點估計；另報同樣門檻下的 B*。
