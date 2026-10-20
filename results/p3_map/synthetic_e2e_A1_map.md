# 地圖判定（full）

- 判定：**ASK**
- 理由：沒有任何進判定的真實資料（有效 session <100、不完整或只有合成資料）
- 進判定的資料：（無）；沒進的：{"synth": "合成"}
- bootstrap 1000 次、seed 20261018、max-T 同時 95% CI；等待時間護欄 1.05×C0；run `20261018-130635-p3sim-anasyn2`

- **合成資料**（不是判定）：如果當成真實資料，規則會給 ASK（沒有滿足 GO，也不是全部上界 <5%（最高的 CI 下界 54.8%））

## 15 個主格：省多少＝(最好的對手 − O) ÷ 最好的對手；截尾 H（判定用）；點估計［同時 CI 下界, 上界］

| 外層做法 | 記憶體 GiB | synth（合成） |
|:--|--:|:--|
| E2 | 63 | 3.6% [2.5%, 4.6%]<br>最好：C0 |
| E2 | 170 | 18.9% [13.9%, 23.9%]<br>最好：C0 |
| E2 | 435 | 61.9% [54.8%, 69.1%]<br>最好：D-len-8192 |
| E3 | 63 | 3.3% [2.4%, 4.3%]<br>最好：C0 |
| E3 | 170 | 6.0% [4.0%, 7.9%]<br>最好：C0 |
| E3 | 435 | 6.6% [4.9%, 8.4%]<br>最好：C0 |
| E4 | 63 | 3.3% [2.4%, 4.2%]<br>最好：C0 |
| E4 | 170 | 5.9% [4.0%, 7.8%]<br>最好：C0 |
| E4 | 435 | 6.5% [4.8%, 8.3%]<br>最好：C0 |
| E5-T1800 | 63 | 3.6% [2.5%, 4.6%]<br>最好：C0 |
| E5-T1800 | 170 | 18.7% [13.8%, 23.6%]<br>最好：C0 |
| E5-T1800 | 435 | 39.0% [32.1%, 45.9%]<br>最好：C0 |
| E6 | 63 | 3.5% [2.8%, 4.3%]<br>最好：C0 |
| E6 | 170 | 20.9% [16.6%, 25.1%]<br>最好：C0 |
| E6 | 435 | 60.6% [54.0%, 67.3%]<br>最好：C0 |

## 15 個主格：省多少＝(最好的對手 − O) ÷ 最好的對手；截尾 2H（只報告）；點估計［同時 CI 下界, 上界］

| 外層做法 | 記憶體 GiB | synth（合成） |
|:--|--:|:--|
| E2 | 63 | 3.8% [2.9%, 4.7%] |
| E2 | 170 | 18.7% [13.9%, 23.5%] |
| E2 | 435 | 62.2% [54.8%, 69.6%] |
| E3 | 63 | 3.6% [2.8%, 4.4%] |
| E3 | 170 | 5.7% [3.9%, 7.6%] |
| E3 | 435 | 6.5% [4.7%, 8.3%] |
| E4 | 63 | 3.5% [2.7%, 4.3%] |
| E4 | 170 | 5.7% [3.9%, 7.5%] |
| E4 | 435 | 6.5% [4.7%, 8.2%] |
| E5-T1800 | 63 | 3.8% [2.9%, 4.7%] |
| E5-T1800 | 170 | 18.5% [13.8%, 23.3%] |
| E5-T1800 | 435 | 38.9% [31.8%, 45.9%] |
| E6 | 63 | 3.6% [2.8%, 4.4%] |
| E6 | 170 | 20.7% [16.4%, 25.1%] |
| E6 | 435 | 60.8% [54.0%, 67.6%] |

## 整張地圖（只有點估計，H；只報告）

### E2@block

| 記憶體 GiB | synth |
|--:|--:|
| 63 | 3.9% |
| 170 | 19.9% |
| 435 | 59.7% |

### E2@session

| 記憶體 GiB | synth |
|--:|--:|
| 63 | 3.6% |
| 170 | 18.9% |
| 435 | 61.9% |

### E3@session

| 記憶體 GiB | synth |
|--:|--:|
| 63 | 3.3% |
| 170 | 6.0% |
| 435 | 6.6% |

### E4@session

| 記憶體 GiB | synth |
|--:|--:|
| 63 | 3.3% |
| 170 | 5.9% |
| 435 | 6.5% |

### E5-T1800@session

| 記憶體 GiB | synth |
|--:|--:|
| 63 | 3.6% |
| 170 | 18.7% |
| 435 | 39.0% |

### E6@block

| 記憶體 GiB | synth |
|--:|--:|
| 63 | 3.5% |
| 170 | 20.9% |
| 435 | 60.6% |

## 敏感度（只報告；點估計；半合成標在 kind）

| kind | 資料 | 外層 | 記憶體 | 最好的對手 | 省多少 |
|:--|:--|:--|--:|:--|--:|
| sens_3090 | synth | E2@session | 63 | C2 | -7406.5% |
| sens_3090_ssd8 | synth | E2@session | 63 | C2 | -7406.5% |
| sens_cpu1.35 | synth | E2@session | 63 | C0 | 3.6% |
| sens_cpu1.81 | synth | E2@session | 63 | C0 | 3.6% |
| sens_3090 | synth | E2@session | 170 | C2 | -632.2% |
| sens_3090_ssd8 | synth | E2@session | 170 | C2 | -632.2% |
| sens_cpu1.35 | synth | E2@session | 170 | C0 | 18.9% |
| sens_cpu1.81 | synth | E2@session | 170 | C0 | 18.9% |
| sens_3090 | synth | E2@session | 435 | C2 | -104.6% |
| sens_3090_ssd8 | synth | E2@session | 435 | C2 | -104.6% |
| sens_cpu1.35 | synth | E2@session | 435 | D-idle-300 | -113.1% |
| sens_cpu1.81 | synth | E2@session | 435 | D-idle-300 | -113.1% |
| sens_3090 | synth | E3@session | 63 | C2 | -5955.1% |
| sens_3090_ssd8 | synth | E3@session | 63 | C2 | -5955.1% |
| sens_cpu1.35 | synth | E3@session | 63 | C0 | 3.3% |
| sens_cpu1.81 | synth | E3@session | 63 | C0 | 3.3% |
| sens_3090 | synth | E3@session | 170 | C2 | -530.2% |
| sens_3090_ssd8 | synth | E3@session | 170 | C2 | -530.2% |
| sens_cpu1.35 | synth | E3@session | 170 | C0 | 6.0% |
| sens_cpu1.81 | synth | E3@session | 170 | C0 | 6.0% |
| sens_3090 | synth | E3@session | 435 | C2 | -229.3% |
| sens_3090_ssd8 | synth | E3@session | 435 | C2 | -229.3% |
| sens_cpu1.35 | synth | E3@session | 435 | C0 | 6.6% |
| sens_cpu1.81 | synth | E3@session | 435 | C0 | 6.6% |
| sens_3090 | synth | E4@session | 63 | C2 | -6073.6% |
| sens_3090_ssd8 | synth | E4@session | 63 | C2 | -6073.6% |
| sens_cpu1.35 | synth | E4@session | 63 | C0 | 3.3% |
| sens_cpu1.81 | synth | E4@session | 63 | C0 | 3.3% |
| sens_3090 | synth | E4@session | 170 | C2 | -536.5% |
| sens_3090_ssd8 | synth | E4@session | 170 | C2 | -536.5% |
| sens_cpu1.35 | synth | E4@session | 170 | C0 | 5.9% |
| sens_cpu1.81 | synth | E4@session | 170 | C0 | 5.9% |
| sens_3090 | synth | E4@session | 435 | C2 | -230.0% |
| sens_3090_ssd8 | synth | E4@session | 435 | C2 | -230.0% |
| sens_cpu1.35 | synth | E4@session | 435 | C0 | 6.5% |
| sens_cpu1.81 | synth | E4@session | 435 | C0 | 6.5% |
| sens_3090 | synth | E5-T1800@session | 63 | C2 | -7365.9% |
| sens_3090_ssd8 | synth | E5-T1800@session | 63 | C2 | -7365.9% |
| sens_cpu1.35 | synth | E5-T1800@session | 63 | C0 | 3.6% |
| sens_cpu1.81 | synth | E5-T1800@session | 63 | C0 | 3.6% |
| sens_3090 | synth | E5-T1800@session | 170 | C2 | -538.5% |
| sens_3090_ssd8 | synth | E5-T1800@session | 170 | C2 | -538.5% |
| sens_cpu1.35 | synth | E5-T1800@session | 170 | C0 | 18.7% |
| sens_cpu1.81 | synth | E5-T1800@session | 170 | C0 | 18.7% |
| sens_3090 | synth | E5-T1800@session | 435 | C1 | -165.4% |
| sens_3090_ssd8 | synth | E5-T1800@session | 435 | C1 | -165.4% |
| sens_cpu1.35 | synth | E5-T1800@session | 435 | C0 | 39.0% |
| sens_cpu1.81 | synth | E5-T1800@session | 435 | C0 | 39.0% |

