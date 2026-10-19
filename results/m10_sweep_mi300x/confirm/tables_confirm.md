# 大掃描複賽（09 §4.3；新 seed 50000 起）

## llama（δ*＝0.0500；300 題；run：20261017-134426-m10-confirm-llama-T24-s0、20261017-134541-m10-confirm-llama-T24-s1、20261017-134656-m10-confirm-llama-T24-s2、20261017-134811-m10-confirm-llama-T24-s3、20261017-223132-m10-confirm-llama-T24-s4、20261017-223247-m10-confirm-llama-T5）

| 配置 | 角色 | ε 四種情境 | 贏隨機（讀法 A，Holm） | 對隨機的差 A（百分點，95% CI） | 速度不輸 | 複賽成立 |
|:--|:--|:--|:--|:--|:--|:--|
| SW-ERR-20@int2_kivi | 選手 | 沒過 | 是 | +2.0 [+0.8, +3.1] | True | **否** |
| SW-ERR-20@kv-int4_kivi-int2_kivi | 選手 | 過 | 否 | -0.3 [-1.0, +0.4] | True | **否** |
| SW-ERR-25@kv-int4_kivi-int2_kivi | 選手 | 過 | 否 | -0.4 [-1.2, +0.4] | True | **否** |
| SW-ERR-15@int2_kivi | 選手 | 沒過 | 否 | +0.5 [-0.6, +1.6] | True | **否** |
| SW-ERR-30@int4_kivi | 選手 | 過 | 否 | -0.3 [-1.0, +0.4] | True | **否** |
| SW-ATT-20@mxfp4 | 選手 | 沒過 | 否 | -0.2 [-1.7, +1.2] | False | **否** |
| E1s@int2_kivi | 前緣點 | 沒過 | — | — | —（ALL：沒有同位元組的混合配置） | — |
| E1s@kv-int4_kivi-int2_kivi | 前緣點 | 沒過 | — | — | —（ALL：沒有同位元組的混合配置） | — |

## qwen（δ*＝0.0943；300 題；run：20261017-225319-m10-confirm-qwen-T24-s0、20261017-225536-m10-confirm-qwen-T24-s1、20261017-225651-m10-confirm-qwen-T24-s2、20261018-035637-m10-confirm-qwen-T24-s3、20261018-035752-m10-confirm-qwen-T24-s4、20261018-045826-m10-confirm-qwen-T5）

| 配置 | 角色 | ε 四種情境 | 贏隨機（讀法 A，Holm） | 對隨機的差 A（百分點，95% CI） | 速度不輸 | 複賽成立 |
|:--|:--|:--|:--|:--|:--|:--|
| SW-ATT-40@kv-int4_kivi-int2_kivi | 選手 | 過 | 否 | +0.5 [-0.4, +1.4] | False | **否** |
| SW-AXE-40@kv-int4_kivi-int2_kivi | 選手 | 過 | 否 | +0.5 [-0.3, +1.5] | False | **否** |
| SW-ATT-25@int4_kivi | 選手 | 過 | 否 | +0.1 [-0.4, +0.6] | True | **否** |
| SW-ATT-10@kv-int8_token-int4_kivi | 選手 | 過 | 否 | -0.3 [-0.8, +0.2] | True | **否** |
| SW-AXE-10@kv-int8_token-int4_kivi | 選手 | 過 | 否 | -0.2 [-0.8, +0.2] | True | **否** |
| SW-AXE-25@int4_kivi | 選手 | 過 | 否 | +0.0 [-0.4, +0.5] | True | **否** |
| E1s@kv-int4_kivi-int2_kivi | 前緣點 | 沒過 | — | — | —（ALL：沒有同位元組的混合配置） | — |
| E1s@int4_kivi | 前緣點 | 過 | — | — | —（ALL：沒有同位元組的混合配置） | — |

