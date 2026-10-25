# X4a 補跑：W 大（每 8／7 層一組）vs W＝4（TF，逐題配對）

run：20261022-171534-m16-x4wall-report；來源 run：20261022-163917-m16-x4awall-llama-s0, 20261022-163917-m16-x4awall-llama-s1, 20261022-163917-m16-x4awall-llama-s2, 20261022-163917-m16-x4awall-qwen-s0, 20261022-163917-m16-x4awall-qwen-s1, 20261022-165103-m16-x4awall-llama-s3, 20261022-165108-m16-x4awall-llama-s4, 20261022-165118-m16-x4awall-qwen-s2, 20261022-165118-m16-x4awall-qwen-s3, 20261022-170103-m16-x4awall-llama-s5, 20261022-170248-m16-x4awall-llama-s6, 20261022-170314-m16-x4awall-llama-s7

| 模型 | T | 係數 | W 大（倍數） | W4（倍數） | 題數 | gold_lp 差（W大−W4）[95% CI] | first_ok 差 | relerr_k W大／W4 | relerr_v W大／W4 |
|:--|--:|:--|:--|:--|--:|:--|--:|:--|:--|
| llama | 20 | int4 | `xl:k4:v6:w32+q:int4_kivi`（19.272） | `xl:k12:v18+q:int4_kivi`（19.417） | 96 | -0.204 [-0.682, +0.274] | +0.010 | +0.078／+0.077 | +0.201／+0.217 |
| llama | 20 | k4v2 | `xl:k4:v6:w32+q:kv-int4_kivi-int2_kivi`（20.425） | `xl:k14:v21+q:kv-int4_kivi-int2_kivi`（20.067） | 96 | -0.337 [-0.954, +0.235] | +0.010 | +0.078／+0.073 | +0.390／+0.389 |
| llama | 24 | int4 | `xl:k3:v5:w32+q:int4_kivi`（24.089） | `xl:k10:v15+q:int4_kivi`（23.3） | 96 | -0.600 [-1.185, -0.058] | +0.010 | +0.085／+0.082 | +0.214／+0.233 |
| llama | 24 | k4v2 | `xl:k3:v5:w32+q:kv-int4_kivi-int2_kivi`（25.595） | `xl:k12:v18+q:kv-int4_kivi-int2_kivi`（23.412） | 96 | -0.836 [-1.563, -0.165] | +0.000 | +0.085／+0.077 | +0.397／+0.396 |
| qwen | 20 | int4 | `xl:k7:v11:w28+q:int4_kivi`（18.789） | `xl:k14:v21+q:int4_kivi`（19.453） | 96 | +0.140 [-0.386, +0.639] | +0.000 | +0.019／+0.020 | +0.145／+0.167 |
| qwen | 20 | k4v2 | `xl:k7:v11:w28+q:kv-int4_kivi-int2_kivi`（20.898） | `xl:k17:v26+q:kv-int4_kivi-int2_kivi`（19.818） | 96 | -0.125 [-0.998, +0.706] | -0.031 | +0.019／+0.018 | +0.405／+0.412 |
| qwen | 24 | int4 | `xl:k6:v9:w28+q:int4_kivi`（22.547） | `xl:k11:v17+q:int4_kivi`（24.317） | 96 | +0.500 [-0.080, +1.102] | -0.021 | +0.020／+0.022 | +0.154／+0.181 |
| qwen | 24 | k4v2 | `xl:k6:v9:w28+q:kv-int4_kivi-int2_kivi`（25.026） | `xl:k14:v21+q:kv-int4_kivi-int2_kivi`（24.301） | 96 | -0.828 [-1.791, +0.079] | +0.000 | +0.020／+0.020 | +0.408／+0.417 |
| qwen | 32 | int4 | `xl:k4:v6:w28+q:int4_kivi`（33.82） | `xl:k8:v12+q:int4_kivi`（34.044） | 96 | +1.757 [+0.703, +2.849] | +0.000 | +0.022／+0.025 | +0.174／+0.205 |
| qwen | 32 | k4v2 | `xl:k5:v8:w28+q:kv-int4_kivi-int2_kivi`（28.959） | `xl:k11:v17+q:kv-int4_kivi-int2_kivi`（30.466） | 96 | +0.266 [-0.547, +1.066] | -0.010 | +0.021／+0.022 | +0.410／+0.423 |

判定：llama **UNDETERMINED**（96 題）；qwen **W_BIG_BETTER**（96 題）

規則：W 大 − W4 的 gold_lp 逐題差（bootstrap 95% CI）：至少一個 (T, 位元) 下界 > 0 → W_BIG_BETTER；全部上界 < 0 → W_BIG_WORSE；其他 → UNDETERMINED；最少的一對 < 48 題 → INCOMPLETE。TF 只是篩選。
有 GPU 爭用紀錄的列：2304（TF 的分數不受影響；時間欄不用）。
