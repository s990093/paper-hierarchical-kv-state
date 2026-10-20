# 地圖判定（full）

- 判定：**ASK**
- 理由：沒有任何進判定的真實資料（有效 session <100、不完整或只有合成資料）
- 進判定的資料：（無）；沒進的：{"synth": "合成"}
- 護欄的參考（判定用）：**C0**（命令列指定 C0（不是 04 §9 的值）（合成資料：判定參數可以用命令列選））；兩種參考都在下面「護欄的參考」節
- 完美預測（判定用的 O）：**greedy**（O；另一個在 report_oracle_alt）；重算計價：**pass_size**
- 能判定的格子（C0 的 SSD 使用率 ≥5%、護欄輸入沒有 NOT_MEASURED；其餘格子 GO、NO-GO 都不算）：None
- bootstrap 1000 次、seed 20261018、max-T 同時 95% CI；等待時間護欄 1.05×參考（截尾版 ttftq_H_*）；GO 另要求 O 過護欄；run `20261021-020034-p3h-anasyn-mi300x-C0-greedy-pass_size`

- **合成資料**（不是判定）：如果當成真實資料，規則會給 NO-GO（所有進判定的資料（synth）、所有能判定的主格（{'synth': 5}；其餘格子 SSD 不忙或護欄輸入有 NOT_MEASURED，不算），同時 CI 上界都 <5%）

## 15 個主格：省多少＝(最好的對手 − O) ÷ 最好的對手；截尾 H（判定用）；點估計［同時 CI 下界, 上界］

| 外層做法 | 記憶體 GiB | synth（合成） |
|:--|--:|:--|
| E2 | 63 | -6.5% [-8.8%, -4.1%]<br>最好：C4r-32768（4.03e+03 GiB/h、使用率 18.2%）；O：4.29e+03 GiB/h、19.0%<br>重算 token：最好 3.14e+08／O 3.15e+08 |
| E2 | 170 | 14.2% [9.9%, 18.4%]<br>最好：D-idle-1800（396 GiB/h、使用率 3.3%）；O：340 GiB/h、2.5%<br>重算 token：最好 1.64e+08／O 1.6e+08<br>⚑ SSD 不忙（C0 使用率 3.6%<5%；GO、NO-GO 都不算） |
| E2 | 435 | -29.4% [-39.4%, -19.3%]<br>最好：C1-2（51.5 GiB/h、使用率 0.5%）；O：66.6 GiB/h、0.7%<br>重算 token：最好 1.59e+08／O 1.54e+08<br>⚑ SSD 不忙（C0 使用率 2.3%<5%；GO、NO-GO 都不算） |
| E3 | 63 | -6.6% [-8.9%, -4.2%]<br>最好：C4r-32768（4.04e+03 GiB/h、使用率 18.4%）；O：4.31e+03 GiB/h、19.3%<br>重算 token：最好 3.22e+08／O 3.22e+08 |
| E3 | 170 | 4.7% [2.8%, 6.6%]<br>最好：C4r-16384（451 GiB/h、使用率 4.2%）；O：430 GiB/h、3.8%<br>重算 token：最好 1.65e+08／O 1.61e+08<br>⚑ SSD 不忙（C0 使用率 4.1%<5%；GO、NO-GO 都不算） |
| E3 | 435 | -243.0% [-289.5%, -196.4%]<br>最好：C1-2（65 GiB/h、使用率 0.7%）；O：223 GiB/h、3.0%<br>重算 token：最好 1.59e+08／O 1.54e+08<br>⚑ SSD 不忙（C0 使用率 3.2%<5%；GO、NO-GO 都不算） |
| E4 | 63 | -6.4% [-8.8%, -4.1%]<br>最好：C4r-32768（4e+03 GiB/h、使用率 18.2%）；O：4.26e+03 GiB/h、19.0%<br>重算 token：最好 3.34e+08／O 3.35e+08 |
| E4 | 170 | 4.7% [2.8%, 6.6%]<br>最好：C4r-16384（443 GiB/h、使用率 4.1%）；O：422 GiB/h、3.8%<br>重算 token：最好 1.65e+08／O 1.61e+08<br>⚑ SSD 不忙（C0 使用率 4.0%<5%；GO、NO-GO 都不算） |
| E4 | 435 | -241.5% [-288.5%, -194.4%]<br>最好：C1-2（63.6 GiB/h、使用率 0.7%）；O：217 GiB/h、2.9%<br>重算 token：最好 1.59e+08／O 1.54e+08<br>⚑ SSD 不忙（C0 使用率 3.1%<5%；GO、NO-GO 都不算） |
| E5-T1800 | 63 | -6.5% [-8.8%, -4.1%]<br>最好：C4r-32768（4.03e+03 GiB/h、使用率 18.2%）；O：4.29e+03 GiB/h、19.0%<br>重算 token：最好 3.14e+08／O 3.15e+08 |
| E5-T1800 | 170 | 14.5% [10.2%, 18.7%]<br>最好：D-idle-1800（399 GiB/h、使用率 3.4%）；O：342 GiB/h、2.6%<br>重算 token：最好 1.64e+08／O 1.6e+08<br>⚑ SSD 不忙（C0 使用率 3.6%<5%；GO、NO-GO 都不算） |
| E5-T1800 | 435 | 34.4% [27.4%, 41.3%]<br>最好：D-idle-1800（216 GiB/h、使用率 2.5%）；O：142 GiB/h、1.3%<br>重算 token：最好 1.57e+08／O 1.55e+08<br>⚑ SSD 不忙（C0 使用率 2.7%<5%；GO、NO-GO 都不算） |
| E6 | 63 | -7.2% [-8.7%, -5.7%]<br>最好：C4r-32768（3.11e+03 GiB/h、使用率 14.4%）；O：3.34e+03 GiB/h、14.9%<br>重算 token：最好 2.09e+08／O 2.07e+08 |
| E6 | 170 | 13.4% [10.0%, 16.8%]<br>最好：D-idle-1800（381 GiB/h、使用率 3.3%）；O：330 GiB/h、2.5%<br>重算 token：最好 1.43e+08／O 1.4e+08<br>⚑ SSD 不忙（C0 使用率 3.5%<5%；GO、NO-GO 都不算） |
| E6 | 435 | 55.4% [48.3%, 62.5%]<br>最好：D-len-32768（181 GiB/h、使用率 2.3%）；O：80.6 GiB/h、0.9%<br>重算 token：最好 1.36e+08／O 1.35e+08<br>⚑ SSD 不忙（C0 使用率 2.6%<5%；GO、NO-GO 都不算） |

## 15 個主格：省多少＝(最好的對手 − O) ÷ 最好的對手；截尾 2H（只報告）；點估計［同時 CI 下界, 上界］

| 外層做法 | 記憶體 GiB | synth（合成） |
|:--|--:|:--|
| E2 | 63 | -6.8% [-9.0%, -4.7%] |
| E2 | 170 | 13.9% [9.9%, 18.0%] |
| E2 | 435 | -28.9% [-38.2%, -19.7%] |
| E3 | 63 | -6.9% [-9.1%, -4.8%] |
| E3 | 170 | 4.7% [2.8%, 6.7%] |
| E3 | 435 | -243.2% [-292.1%, -194.4%] |
| E4 | 63 | -6.8% [-8.9%, -4.7%] |
| E4 | 170 | 4.7% [2.7%, 6.7%] |
| E4 | 435 | -241.7% [-291.0%, -192.4%] |
| E5-T1800 | 63 | -6.8% [-9.0%, -4.7%] |
| E5-T1800 | 170 | 14.2% [10.1%, 18.3%] |
| E5-T1800 | 435 | 34.1% [27.0%, 41.2%] |
| E6 | 63 | -7.5% [-8.9%, -6.1%] |
| E6 | 170 | 13.3% [9.9%, 16.8%] |
| E6 | 435 | 55.6% [48.2%, 63.0%] |

## 護欄的參考（兩種都報告；點估計；H；主格）：C0＝對手要 ≤1.05×C0；min(C0, O)＝要 ≤1.05×C0 與 O 逐項取小

| 外層做法 | 記憶體 GiB | 參考 | synth |
|:--|--:|:--|:--|
| E2 | 63 | C0 | -6.5%（C4r-32768） |
| E2 | 63 | min_C0_O | -6.5%（C4r-32768） |
| E2 | 170 | C0 | 14.2%（D-idle-1800） ⚑SSD 不忙（C0 使用率 3.6%<5%；GO、NO-GO 都不算） |
| E2 | 170 | min_C0_O | 14.2%（D-idle-1800） ⚑SSD 不忙（C0 使用率 3.6%<5%；GO、NO-GO 都不算） |
| E2 | 435 | C0 | -29.4%（C1-2） ⚑SSD 不忙（C0 使用率 2.3%<5%；GO、NO-GO 都不算） |
| E2 | 435 | min_C0_O | -29.4%（C1-2） ⚑SSD 不忙（C0 使用率 2.3%<5%；GO、NO-GO 都不算） |
| E3 | 63 | C0 | -6.6%（C4r-32768） |
| E3 | 63 | min_C0_O | -6.6%（C4r-32768） |
| E3 | 170 | C0 | 4.7%（C4r-16384） ⚑SSD 不忙（C0 使用率 4.1%<5%；GO、NO-GO 都不算） |
| E3 | 170 | min_C0_O | 4.7%（C4r-16384） ⚑SSD 不忙（C0 使用率 4.1%<5%；GO、NO-GO 都不算） |
| E3 | 435 | C0 | -243.0%（C1-2） ⚑SSD 不忙（C0 使用率 3.2%<5%；GO、NO-GO 都不算） |
| E3 | 435 | min_C0_O | -243.0%（C1-2） ⚑SSD 不忙（C0 使用率 3.2%<5%；GO、NO-GO 都不算） |
| E4 | 63 | C0 | -6.4%（C4r-32768） |
| E4 | 63 | min_C0_O | -6.4%（C4r-32768） |
| E4 | 170 | C0 | 4.7%（C4r-16384） ⚑SSD 不忙（C0 使用率 4.0%<5%；GO、NO-GO 都不算） |
| E4 | 170 | min_C0_O | 4.7%（C4r-16384） ⚑SSD 不忙（C0 使用率 4.0%<5%；GO、NO-GO 都不算） |
| E4 | 435 | C0 | -241.5%（C1-2） ⚑SSD 不忙（C0 使用率 3.1%<5%；GO、NO-GO 都不算） |
| E4 | 435 | min_C0_O | -241.5%（C1-2） ⚑SSD 不忙（C0 使用率 3.1%<5%；GO、NO-GO 都不算） |
| E5-T1800 | 63 | C0 | -6.5%（C4r-32768） |
| E5-T1800 | 63 | min_C0_O | -6.5%（C4r-32768） |
| E5-T1800 | 170 | C0 | 14.5%（D-idle-1800） ⚑SSD 不忙（C0 使用率 3.6%<5%；GO、NO-GO 都不算） |
| E5-T1800 | 170 | min_C0_O | 14.5%（D-idle-1800） ⚑SSD 不忙（C0 使用率 3.6%<5%；GO、NO-GO 都不算） |
| E5-T1800 | 435 | C0 | 34.4%（D-idle-1800） ⚑SSD 不忙（C0 使用率 2.7%<5%；GO、NO-GO 都不算） |
| E5-T1800 | 435 | min_C0_O | 34.4%（D-idle-1800） ⚑SSD 不忙（C0 使用率 2.7%<5%；GO、NO-GO 都不算） |
| E6 | 63 | C0 | -7.2%（C4r-32768） |
| E6 | 63 | min_C0_O | -7.2%（C4r-32768） |
| E6 | 170 | C0 | 13.4%（D-idle-1800） ⚑SSD 不忙（C0 使用率 3.5%<5%；GO、NO-GO 都不算） |
| E6 | 170 | min_C0_O | 13.4%（D-idle-1800） ⚑SSD 不忙（C0 使用率 3.5%<5%；GO、NO-GO 都不算） |
| E6 | 435 | C0 | 55.4%（D-len-32768） ⚑SSD 不忙（C0 使用率 2.6%<5%；GO、NO-GO 都不算） |
| E6 | 435 | min_C0_O | 55.4%（D-len-32768） ⚑SSD 不忙（C0 使用率 2.6%<5%；GO、NO-GO 都不算） |

## 寬鬆版雙胞胎（只報告；點估計；H；O 與所有對手都用寬鬆品質規則＝32K 過關的格式任何長度都過關）

| 外層做法 | 記憶體 GiB | synth |
|:--|--:|:--|
| E2 | 63 | 11.7%（C2） |
| E2 | 170 | 32.8%（C2） ⚑SSD 不忙（C0 使用率 3.6%<5%；GO、NO-GO 都不算） |
| E2 | 435 | 40.6%（C2） ⚑SSD 不忙（C0 使用率 2.3%<5%；GO、NO-GO 都不算） |
| E3 | 63 | 0.4%（C2） |
| E3 | 170 | 0.6%（C2） ⚑SSD 不忙（C0 使用率 4.1%<5%；GO、NO-GO 都不算） |
| E3 | 435 | -30.3%（C1-2） ⚑SSD 不忙（C0 使用率 3.2%<5%；GO、NO-GO 都不算） |
| E4 | 63 | 0.4%（C2） |
| E4 | 170 | 0.6%（C2） ⚑SSD 不忙（C0 使用率 4.0%<5%；GO、NO-GO 都不算） |
| E4 | 435 | -29.8%（C1-2） ⚑SSD 不忙（C0 使用率 3.1%<5%；GO、NO-GO 都不算） |
| E5-T1800 | 63 | 11.4%（C2） |
| E5-T1800 | 170 | 18.8%（C2） ⚑SSD 不忙（C0 使用率 3.6%<5%；GO、NO-GO 都不算） |
| E5-T1800 | 435 | 18.8%（C1） ⚑SSD 不忙（C0 使用率 2.7%<5%；GO、NO-GO 都不算） |
| E6 | 63 | 11.8%（C2） |
| E6 | 170 | 31.9%（C2） ⚑SSD 不忙（C0 使用率 3.5%<5%；GO、NO-GO 都不算） |
| E6 | 435 | 39.5%（C2） ⚑SSD 不忙（C0 使用率 2.6%<5%；GO、NO-GO 都不算） |

## 護欄敏感度（只報告；點估計；H；主格；參考＝C0）：護欄 1.05 是登記的（判定用），其他只用來看判定有多依賴 5%

| 外層做法 | 記憶體 GiB | 護欄 | synth |
|:--|--:|:--|:--|
| E2 | 63 | 1.05 | -6.5%（C4r-32768） |
| E2 | 63 | 1.10 | -6.5%（C4r-32768） |
| E2 | 63 | 1.20 | -6.5%（C4r-32768） |
| E2 | 63 | 不設 | -1798.0%（C2） |
| E2 | 170 | 1.05 | 14.2%（D-idle-1800） |
| E2 | 170 | 1.10 | 7.7%（C4r-32768） |
| E2 | 170 | 1.20 | 7.7%（C4r-32768） |
| E2 | 170 | 不設 | -454.6%（C2） |
| E2 | 435 | 1.05 | -29.4%（C1-2） |
| E2 | 435 | 1.10 | -29.4%（C1-2） |
| E2 | 435 | 1.20 | -29.4%（C1-2） |
| E2 | 435 | 不設 | -102.9%（C2） |
| E3 | 63 | 1.05 | -6.6%（C4r-32768） |
| E3 | 63 | 1.10 | -6.6%（C4r-32768） |
| E3 | 63 | 1.20 | -6.6%（C4r-32768） |
| E3 | 63 | 不設 | -19.3%（C4-32768） |
| E3 | 170 | 1.05 | 4.7%（C4r-16384） |
| E3 | 170 | 1.10 | -3.1%（C4r-32768） |
| E3 | 170 | 1.20 | -3.1%（C4r-32768） |
| E3 | 170 | 不設 | -36.3%（C1-2） |
| E3 | 435 | 1.05 | -243.0%（C1-2） |
| E3 | 435 | 1.10 | -243.0%（C1-2） |
| E3 | 435 | 1.20 | -243.0%（C1-2） |
| E3 | 435 | 不設 | -243.0%（C1-2） |
| E4 | 63 | 1.05 | -6.4%（C4r-32768） |
| E4 | 63 | 1.10 | -6.4%（C4r-32768） |
| E4 | 63 | 1.20 | -6.4%（C4r-32768） |
| E4 | 63 | 不設 | -19.5%（C4-32768） |
| E4 | 170 | 1.05 | 4.7%（C4r-16384） |
| E4 | 170 | 1.10 | -2.9%（C4r-32768） |
| E4 | 170 | 1.20 | -2.9%（C4r-32768） |
| E4 | 170 | 不設 | -35.4%（C1-2） |
| E4 | 435 | 1.05 | -241.5%（C1-2） |
| E4 | 435 | 1.10 | -241.5%（C1-2） |
| E4 | 435 | 1.20 | -241.5%（C1-2） |
| E4 | 435 | 不設 | -241.5%（C1-2） |
| E5-T1800 | 63 | 1.05 | -6.5%（C4r-32768） |
| E5-T1800 | 63 | 1.10 | -6.5%（C4r-32768） |
| E5-T1800 | 63 | 1.20 | -6.5%（C4r-32768） |
| E5-T1800 | 63 | 不設 | -1769.7%（C2） |
| E5-T1800 | 170 | 1.05 | 14.5%（D-idle-1800） |
| E5-T1800 | 170 | 1.10 | 7.7%（C4r-32768） |
| E5-T1800 | 170 | 1.20 | 7.7%（C4r-32768） |
| E5-T1800 | 170 | 不設 | -157.5%（C2） |
| E5-T1800 | 435 | 1.05 | 34.4%（D-idle-1800） |
| E5-T1800 | 435 | 1.10 | 34.4%（D-idle-1800） |
| E5-T1800 | 435 | 1.20 | -34.6%（C1-2） |
| E5-T1800 | 435 | 不設 | -34.6%（C1-2） |
| E6 | 63 | 1.05 | -7.2%（C4r-32768） |
| E6 | 63 | 1.10 | -7.2%（C4r-32768） |
| E6 | 63 | 1.20 | -7.2%（C4r-32768） |
| E6 | 63 | 不設 | -1654.4%（C2） |
| E6 | 170 | 1.05 | 13.4%（D-idle-1800） |
| E6 | 170 | 1.10 | 12.0%（C4r-32768） |
| E6 | 170 | 1.20 | 12.0%（C4r-32768） |
| E6 | 170 | 不設 | -1596.5%（C2） |
| E6 | 435 | 1.05 | 55.4%（D-len-32768） |
| E6 | 435 | 1.10 | 55.4%（D-len-32768） |
| E6 | 435 | 1.20 | 55.4%（D-len-32768） |
| E6 | 435 | 不設 | -8119.9%（C2） |

## 整張地圖（只有點估計，H；只報告）

### E2@block

| 記憶體 GiB | synth |
|--:|--:|
| 63 | -7.3% |
| 170 | 13.9% |
| 435 | 55.5% |

### E2@session

| 記憶體 GiB | synth |
|--:|--:|
| 63 | -6.5% |
| 170 | 14.2% |
| 435 | -29.4% |

### E3@session

| 記憶體 GiB | synth |
|--:|--:|
| 63 | -6.6% |
| 170 | 4.7% |
| 435 | -243.0% |

### E4@session

| 記憶體 GiB | synth |
|--:|--:|
| 63 | -6.4% |
| 170 | 4.7% |
| 435 | -241.5% |

### E5-T1800@session

| 記憶體 GiB | synth |
|--:|--:|
| 63 | -6.5% |
| 170 | 14.5% |
| 435 | 34.4% |

### E6@block

| 記憶體 GiB | synth |
|--:|--:|
| 63 | -7.2% |
| 170 | 13.4% |
| 435 | 55.4% |

## 其他視角（只報告；點估計；H；主視角以外的時間設定與 GPU 忙）

| 資料 | 外層 | 記憶體 | 視角 | 最好的對手 | 省多少 |
|:--|:--|--:|:--|:--|--:|
| synth | E2@session | 63 | mi300x|cake|gpu=busy|cpu=idle|ssd/1|mi300x_local | C4r-32768 | -6.5% |
| synth | E2@session | 63 | mi300x|cake|gpu=busy|cpu=idle|ssd/2|mi300x_local | C4r-32768 | -6.5% |
| synth | E2@session | 63 | mi300x|cake|gpu=busy|cpu=idle|ssd/4|mi300x_local | C4r-32768 | -6.5% |
| synth | E2@session | 63 | mi300x|cake|gpu=busy|cpu=idle|ssd/8|mi300x_local | C4r-32768 | -6.5% |
| synth | E2@session | 63 | mi300x|cake|gpu=idle|cpu=idle|ssd/2|mi300x_local | C4r-32768 | -6.5% |
| synth | E2@session | 63 | mi300x|cake|gpu=idle|cpu=idle|ssd/4|mi300x_local | C4r-32768 | -6.5% |
| synth | E2@session | 63 | mi300x|cake|gpu=idle|cpu=idle|ssd/8|mi300x_local | C4r-32768 | -6.5% |
| synth | E2@session | 170 | mi300x|cake|gpu=busy|cpu=idle|ssd/1|mi300x_local | D-idle-1800 | 14.2% |
| synth | E2@session | 170 | mi300x|cake|gpu=busy|cpu=idle|ssd/2|mi300x_local | D-idle-1800 | 14.2% |
| synth | E2@session | 170 | mi300x|cake|gpu=busy|cpu=idle|ssd/4|mi300x_local | D-idle-1800 | 14.2% |
| synth | E2@session | 170 | mi300x|cake|gpu=busy|cpu=idle|ssd/8|mi300x_local | C4r-32768 | 7.7% |
| synth | E2@session | 170 | mi300x|cake|gpu=idle|cpu=idle|ssd/2|mi300x_local | C4r-32768 | 7.7% |
| synth | E2@session | 170 | mi300x|cake|gpu=idle|cpu=idle|ssd/4|mi300x_local | C4r-32768 | 7.7% |
| synth | E2@session | 170 | mi300x|cake|gpu=idle|cpu=idle|ssd/8|mi300x_local | C4r-32768 | 7.7% |
| synth | E2@session | 435 | mi300x|cake|gpu=busy|cpu=idle|ssd/1|mi300x_local | D-idle-3600 | 12.1% |
| synth | E2@session | 435 | mi300x|cake|gpu=busy|cpu=idle|ssd/2|mi300x_local | D-idle-3600 | 12.1% |
| synth | E2@session | 435 | mi300x|cake|gpu=busy|cpu=idle|ssd/4|mi300x_local | C1-2 | -29.4% |
| synth | E2@session | 435 | mi300x|cake|gpu=busy|cpu=idle|ssd/8|mi300x_local | C1-2 | -29.4% |
| synth | E2@session | 435 | mi300x|cake|gpu=idle|cpu=idle|ssd/2|mi300x_local | C1-2 | -29.4% |
| synth | E2@session | 435 | mi300x|cake|gpu=idle|cpu=idle|ssd/4|mi300x_local | C1-2 | -29.4% |
| synth | E2@session | 435 | mi300x|cake|gpu=idle|cpu=idle|ssd/8|mi300x_local | C1-2 | -29.4% |
| synth | E3@session | 63 | mi300x|cake|gpu=busy|cpu=idle|ssd/1|mi300x_local | C4r-32768 | -6.6% |
| synth | E3@session | 63 | mi300x|cake|gpu=busy|cpu=idle|ssd/2|mi300x_local | C4r-32768 | -6.6% |
| synth | E3@session | 63 | mi300x|cake|gpu=busy|cpu=idle|ssd/4|mi300x_local | C4r-32768 | -6.6% |
| synth | E3@session | 63 | mi300x|cake|gpu=busy|cpu=idle|ssd/8|mi300x_local | C4r-32768 | -6.6% |
| synth | E3@session | 63 | mi300x|cake|gpu=idle|cpu=idle|ssd/2|mi300x_local | C4r-32768 | -6.6% |
| synth | E3@session | 63 | mi300x|cake|gpu=idle|cpu=idle|ssd/4|mi300x_local | C4r-32768 | -6.6% |
| synth | E3@session | 63 | mi300x|cake|gpu=idle|cpu=idle|ssd/8|mi300x_local | C4r-32768 | -6.6% |
| synth | E3@session | 170 | mi300x|cake|gpu=busy|cpu=idle|ssd/1|mi300x_local | C0 | 6.0% |
| synth | E3@session | 170 | mi300x|cake|gpu=busy|cpu=idle|ssd/2|mi300x_local | C0 | 6.0% |
| synth | E3@session | 170 | mi300x|cake|gpu=busy|cpu=idle|ssd/4|mi300x_local | C4r-16384 | 4.7% |
| synth | E3@session | 170 | mi300x|cake|gpu=busy|cpu=idle|ssd/8|mi300x_local | C4r-32768 | -3.1% |
| synth | E3@session | 170 | mi300x|cake|gpu=idle|cpu=idle|ssd/2|mi300x_local | C4r-16384 | 4.7% |
| synth | E3@session | 170 | mi300x|cake|gpu=idle|cpu=idle|ssd/4|mi300x_local | C4r-32768 | -3.1% |
| synth | E3@session | 170 | mi300x|cake|gpu=idle|cpu=idle|ssd/8|mi300x_local | C4r-32768 | -3.1% |
| synth | E3@session | 435 | mi300x|cake|gpu=busy|cpu=idle|ssd/1|mi300x_local | D-len-16384 | 4.1% |
| synth | E3@session | 435 | mi300x|cake|gpu=busy|cpu=idle|ssd/2|mi300x_local | D-len-16384 | 4.1% |
| synth | E3@session | 435 | mi300x|cake|gpu=busy|cpu=idle|ssd/4|mi300x_local | C1-2 | -243.0% |
| synth | E3@session | 435 | mi300x|cake|gpu=busy|cpu=idle|ssd/8|mi300x_local | C1-2 | -243.0% |
| synth | E3@session | 435 | mi300x|cake|gpu=idle|cpu=idle|ssd/2|mi300x_local | C1-2 | -243.0% |
| synth | E3@session | 435 | mi300x|cake|gpu=idle|cpu=idle|ssd/4|mi300x_local | C1-2 | -243.0% |
| synth | E3@session | 435 | mi300x|cake|gpu=idle|cpu=idle|ssd/8|mi300x_local | C1-2 | -243.0% |
| synth | E4@session | 63 | mi300x|cake|gpu=busy|cpu=idle|ssd/1|mi300x_local | C4r-32768 | -6.4% |
| synth | E4@session | 63 | mi300x|cake|gpu=busy|cpu=idle|ssd/2|mi300x_local | C4r-32768 | -6.4% |
| synth | E4@session | 63 | mi300x|cake|gpu=busy|cpu=idle|ssd/4|mi300x_local | C4r-32768 | -6.4% |
| synth | E4@session | 63 | mi300x|cake|gpu=busy|cpu=idle|ssd/8|mi300x_local | C4r-32768 | -6.4% |
| synth | E4@session | 63 | mi300x|cake|gpu=idle|cpu=idle|ssd/2|mi300x_local | C4r-32768 | -6.4% |
| synth | E4@session | 63 | mi300x|cake|gpu=idle|cpu=idle|ssd/4|mi300x_local | C4r-32768 | -6.4% |
| synth | E4@session | 63 | mi300x|cake|gpu=idle|cpu=idle|ssd/8|mi300x_local | C4r-32768 | -6.4% |
| synth | E4@session | 170 | mi300x|cake|gpu=busy|cpu=idle|ssd/1|mi300x_local | C0 | 5.9% |
| synth | E4@session | 170 | mi300x|cake|gpu=busy|cpu=idle|ssd/2|mi300x_local | C0 | 5.9% |
| synth | E4@session | 170 | mi300x|cake|gpu=busy|cpu=idle|ssd/4|mi300x_local | C4r-16384 | 4.7% |
| synth | E4@session | 170 | mi300x|cake|gpu=busy|cpu=idle|ssd/8|mi300x_local | C4r-32768 | -2.9% |
| synth | E4@session | 170 | mi300x|cake|gpu=idle|cpu=idle|ssd/2|mi300x_local | C4r-32768 | -2.9% |
| synth | E4@session | 170 | mi300x|cake|gpu=idle|cpu=idle|ssd/4|mi300x_local | C4r-32768 | -2.9% |
| synth | E4@session | 170 | mi300x|cake|gpu=idle|cpu=idle|ssd/8|mi300x_local | C4r-32768 | -2.9% |
| synth | E4@session | 435 | mi300x|cake|gpu=busy|cpu=idle|ssd/1|mi300x_local | D-len-16384 | 4.2% |
| synth | E4@session | 435 | mi300x|cake|gpu=busy|cpu=idle|ssd/2|mi300x_local | D-len-16384 | 4.2% |
| synth | E4@session | 435 | mi300x|cake|gpu=busy|cpu=idle|ssd/4|mi300x_local | C1-2 | -241.5% |
| synth | E4@session | 435 | mi300x|cake|gpu=busy|cpu=idle|ssd/8|mi300x_local | C1-2 | -241.5% |
| synth | E4@session | 435 | mi300x|cake|gpu=idle|cpu=idle|ssd/2|mi300x_local | C1-2 | -241.5% |
| synth | E4@session | 435 | mi300x|cake|gpu=idle|cpu=idle|ssd/4|mi300x_local | C1-2 | -241.5% |
| synth | E4@session | 435 | mi300x|cake|gpu=idle|cpu=idle|ssd/8|mi300x_local | C1-2 | -241.5% |
| synth | E5-T1800@session | 63 | mi300x|cake|gpu=busy|cpu=idle|ssd/1|mi300x_local | C4r-32768 | -6.5% |
| synth | E5-T1800@session | 63 | mi300x|cake|gpu=busy|cpu=idle|ssd/2|mi300x_local | C4r-32768 | -6.5% |
| synth | E5-T1800@session | 63 | mi300x|cake|gpu=busy|cpu=idle|ssd/4|mi300x_local | C4r-32768 | -6.5% |
| synth | E5-T1800@session | 63 | mi300x|cake|gpu=busy|cpu=idle|ssd/8|mi300x_local | C4r-32768 | -6.5% |
| synth | E5-T1800@session | 63 | mi300x|cake|gpu=idle|cpu=idle|ssd/2|mi300x_local | C4r-32768 | -6.5% |
| synth | E5-T1800@session | 63 | mi300x|cake|gpu=idle|cpu=idle|ssd/4|mi300x_local | C4r-32768 | -6.5% |
| synth | E5-T1800@session | 63 | mi300x|cake|gpu=idle|cpu=idle|ssd/8|mi300x_local | C4r-32768 | -6.5% |
| synth | E5-T1800@session | 170 | mi300x|cake|gpu=busy|cpu=idle|ssd/1|mi300x_local | D-idle-1800 | 14.5% |
| synth | E5-T1800@session | 170 | mi300x|cake|gpu=busy|cpu=idle|ssd/2|mi300x_local | D-idle-1800 | 14.5% |
| synth | E5-T1800@session | 170 | mi300x|cake|gpu=busy|cpu=idle|ssd/4|mi300x_local | C4r-32768 | 7.7% |
| synth | E5-T1800@session | 170 | mi300x|cake|gpu=busy|cpu=idle|ssd/8|mi300x_local | C4r-32768 | 7.7% |
| synth | E5-T1800@session | 170 | mi300x|cake|gpu=idle|cpu=idle|ssd/2|mi300x_local | C4r-32768 | 7.7% |
| synth | E5-T1800@session | 170 | mi300x|cake|gpu=idle|cpu=idle|ssd/4|mi300x_local | C4r-32768 | 7.7% |
| synth | E5-T1800@session | 170 | mi300x|cake|gpu=idle|cpu=idle|ssd/8|mi300x_local | C4r-32768 | 7.7% |
| synth | E5-T1800@session | 435 | mi300x|cake|gpu=busy|cpu=idle|ssd/1|mi300x_local | D-idle-1800 | 34.4% |
| synth | E5-T1800@session | 435 | mi300x|cake|gpu=busy|cpu=idle|ssd/2|mi300x_local | D-idle-1800 | 34.4% |
| synth | E5-T1800@session | 435 | mi300x|cake|gpu=busy|cpu=idle|ssd/4|mi300x_local | D-idle-1800 | 34.4% |
| synth | E5-T1800@session | 435 | mi300x|cake|gpu=busy|cpu=idle|ssd/8|mi300x_local | D-idle-1800 | 34.4% |
| synth | E5-T1800@session | 435 | mi300x|cake|gpu=idle|cpu=idle|ssd/2|mi300x_local | D-idle-1800 | 34.4% |
| synth | E5-T1800@session | 435 | mi300x|cake|gpu=idle|cpu=idle|ssd/4|mi300x_local | D-idle-1800 | 34.4% |
| synth | E5-T1800@session | 435 | mi300x|cake|gpu=idle|cpu=idle|ssd/8|mi300x_local | D-idle-1800 | 34.4% |

## 敏感度（只報告；點估計；半合成標在 kind）

| kind | 資料 | 外層 | 記憶體 | 最好的對手 | 省多少 |
|:--|:--|:--|--:|:--|--:|
| sens_3090 | synth | E2@session | 63 | C2 | -1798.0% |
| sens_3090_ssd8 | synth | E2@session | 63 | C2 | -1798.0% |
| sens_cpu1.35 | synth | E2@session | 63 | C4r-32768 | -6.5% |
| sens_cpu1.81 | synth | E2@session | 63 | C4r-32768 | -6.5% |
| sens_3090 | synth | E2@session | 170 | C2 | -454.6% |
| sens_3090_ssd8 | synth | E2@session | 170 | C2 | -454.6% |
| sens_cpu1.35 | synth | E2@session | 170 | C4r-32768 | 7.7% |
| sens_cpu1.81 | synth | E2@session | 170 | C4r-32768 | 7.7% |
| sens_3090 | synth | E2@session | 435 | C2 | -102.9% |
| sens_3090_ssd8 | synth | E2@session | 435 | C2 | -102.9% |
| sens_cpu1.35 | synth | E2@session | 435 | C1-2 | -29.4% |
| sens_cpu1.81 | synth | E2@session | 435 | C1-2 | -29.4% |
| sens_3090 | synth | E3@session | 63 | C4-32768 | -19.3% |
| sens_3090_ssd8 | synth | E3@session | 63 | C4-32768 | -19.3% |
| sens_cpu1.35 | synth | E3@session | 63 | C4r-32768 | -6.6% |
| sens_cpu1.81 | synth | E3@session | 63 | C4r-32768 | -6.6% |
| sens_3090 | synth | E3@session | 170 | C1 | -15.5% |
| sens_3090_ssd8 | synth | E3@session | 170 | C1-2 | -36.3% |
| sens_cpu1.35 | synth | E3@session | 170 | C4r-32768 | -3.1% |
| sens_cpu1.81 | synth | E3@session | 170 | C4r-32768 | -3.1% |
| sens_3090 | synth | E3@session | 435 | C1-2 | -243.0% |
| sens_3090_ssd8 | synth | E3@session | 435 | C1-2 | -243.0% |
| sens_cpu1.35 | synth | E3@session | 435 | C1-2 | -243.0% |
| sens_cpu1.81 | synth | E3@session | 435 | C1-2 | -243.0% |
| sens_3090 | synth | E4@session | 63 | C4-32768 | -19.5% |
| sens_3090_ssd8 | synth | E4@session | 63 | C4-32768 | -19.5% |
| sens_cpu1.35 | synth | E4@session | 63 | C4r-32768 | -6.4% |
| sens_cpu1.81 | synth | E4@session | 63 | C4r-32768 | -6.4% |
| sens_3090 | synth | E4@session | 170 | C1 | -14.8% |
| sens_3090_ssd8 | synth | E4@session | 170 | C1-2 | -35.4% |
| sens_cpu1.35 | synth | E4@session | 170 | C4r-32768 | -2.9% |
| sens_cpu1.81 | synth | E4@session | 170 | C4r-32768 | -2.9% |
| sens_3090 | synth | E4@session | 435 | C1-2 | -241.5% |
| sens_3090_ssd8 | synth | E4@session | 435 | C1-2 | -241.5% |
| sens_cpu1.35 | synth | E4@session | 435 | C1-2 | -241.5% |
| sens_cpu1.81 | synth | E4@session | 435 | C1-2 | -241.5% |
| sens_3090 | synth | E5-T1800@session | 63 | C2 | -1769.7% |
| sens_3090_ssd8 | synth | E5-T1800@session | 63 | C2 | -1769.7% |
| sens_cpu1.35 | synth | E5-T1800@session | 63 | C4r-32768 | -6.5% |
| sens_cpu1.81 | synth | E5-T1800@session | 63 | C4r-32768 | -6.5% |
| sens_3090 | synth | E5-T1800@session | 170 | C2 | -157.5% |
| sens_3090_ssd8 | synth | E5-T1800@session | 170 | C2 | -157.5% |
| sens_cpu1.35 | synth | E5-T1800@session | 170 | C4r-32768 | 7.7% |
| sens_cpu1.81 | synth | E5-T1800@session | 170 | C4r-32768 | 7.7% |
| sens_3090 | synth | E5-T1800@session | 435 | C1 | -7.1% |
| sens_3090_ssd8 | synth | E5-T1800@session | 435 | C1-2 | -34.6% |
| sens_cpu1.35 | synth | E5-T1800@session | 435 | D-idle-1800 | 34.4% |
| sens_cpu1.81 | synth | E5-T1800@session | 435 | D-idle-1800 | 34.4% |

