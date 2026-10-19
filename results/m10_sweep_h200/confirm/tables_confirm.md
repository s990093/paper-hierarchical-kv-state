# 大掃描複賽（09 §4.3；新 seed 50000 起）

## llama（δ*＝0.2842；300 題；run：20261019-033607-m10-confirm-llama-T24-s0、20261019-033722-m10-confirm-llama-T24-s1、20261019-033837-m10-confirm-llama-T24-s2、20261019-060440-m10-confirm-llama-T24-s3、20261019-060956-m10-confirm-llama-T24-s4、20261019-061417-m10-confirm-llama-T5）

| 配置 | 角色 | ε 四種情境 | 贏隨機（讀法 A，Holm） | 對隨機的差 A（百分點，95% CI） | 速度不輸 | 複賽成立 |
|:--|:--|:--|:--|:--|:--|:--|
| SW-ERR-25@int2_kivi | 選手 | 沒過 | 否 | +1.1 [-0.2, +2.5] | True | **否** |
| SW-ERR-20@int2_kivi | 選手 | 沒過 | 否 | +0.9 [-0.3, +2.0] | True | **否** |
| SW-ERR-15@kv-int4_kivi-int2_kivi | 選手 | 過 | 否 | +0.2 [-0.5, +1.0] | True | **否** |
| SW-ERR-25@kv-int4_kivi-int2_kivi | 選手 | 過 | 否 | +0.0 [-0.8, +0.8] | True | **否** |
| SW-ATT-10@nvfp4 | 選手 | 過 | 否 | +0.1 [-0.7, +0.9] | True | **否** |
| SW-AXE-10@nvfp4 | 選手 | 過 | 否 | -0.1 [-0.9, +0.8] | True | **否** |
| E1s@kv-int4_kivi-int2_kivi | 前緣點 | 沒過 | — | — | —（ALL：沒有同位元組的混合配置） | — |
| SW-M-90@int2_kivi | 前緣點 | 沒過 | — | -0.3 [-1.7, +1.0] | True | — |

## qwen（δ*＝0.0500；300 題；run：20261019-074225-m10-confirm-qwen-T24-s0、20261019-083256-m10-confirm-qwen-T24-s1、20261019-083411-m10-confirm-qwen-T24-s2、20261019-084033-m10-confirm-qwen-T24-s3、20261019-095818-m10-confirm-qwen-T24-s4、20261019-111032-m10-confirm-qwen-T5）

| 配置 | 角色 | ε 四種情境 | 贏隨機（讀法 A，Holm） | 對隨機的差 A（百分點，95% CI） | 速度不輸 | 複賽成立 |
|:--|:--|:--|:--|:--|:--|:--|
| SW-AXE-40@kv-int4_kivi-int2_kivi | 選手 | 過 | 否 | +1.2 [+0.2, +2.1] | False | **否** |
| SW-ATT-40@kv-int4_kivi-int2_kivi | 選手 | 過 | 否 | +1.1 [+0.1, +2.1] | False | **否** |
| SW-ATT-10@kv-int8_token-int4_kivi | 選手 | 過 | 否 | -0.2 [-0.7, +0.3] | True | **否** |
| SW-AXE-10@kv-int8_token-int4_kivi | 選手 | 過 | 否 | -0.3 [-0.7, +0.2] | True | **否** |
| SW-ERR-10@int8_token | 選手 | 過 | 否 | -0.1 [-0.6, +0.4] | True | **否** |
| SW-ATT-35@int4_kivi | 選手 | 過 | 否 | +0.3 [-0.3, +0.9] | False | **否** |
| E1s@kv-int4_kivi-int2_kivi | 前緣點 | 沒過 | — | — | —（ALL：沒有同位元組的混合配置） | — |
| E1s@int4_kivi | 前緣點 | 過 | — | — | —（ALL：沒有同位元組的混合配置） | — |

