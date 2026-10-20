# 地圖判定（full）

- 判定：**ASK**
- 理由：沒有任何進判定的真實資料（有效 session <100、不完整或只有合成資料）
- 進判定的資料：（無）；沒進的：{"synth": "合成"}
- 護欄的參考（判定用）：**C0**（命令列指定 C0（不是 04 §9 的值）（合成資料：判定參數可以用命令列選））；兩種參考都在下面「護欄的參考」節
- 完美預測（判定用的 O）：**tier**（O-tier；另一個在 report_oracle_alt）；重算計價：**none**
- 能判定的格子（C0 的 SSD 使用率 ≥5%、護欄輸入沒有 NOT_MEASURED；其餘格子 GO、NO-GO 都不算）：None
- bootstrap 1000 次、seed 20261018、max-T 同時 95% CI；等待時間護欄 1.05×參考（截尾版 ttftq_H_*）；GO 另要求 O 過護欄；run `20261021-020039-p3h-anasyn-mi300x-C0-tier-none`

- **合成資料**（不是判定）：如果當成真實資料，規則會給 NO-GO（所有進判定的資料（synth）、所有能判定的主格（{'synth': 5}；其餘格子 SSD 不忙或護欄輸入有 NOT_MEASURED，不算），同時 CI 上界都 <5%）

## 15 個主格：省多少＝(最好的對手 − O) ÷ 最好的對手；截尾 H（判定用）；點估計［同時 CI 下界, 上界］

| 外層做法 | 記憶體 GiB | synth（合成） |
|:--|--:|:--|
| E2 | 63 | -6.5% [-8.7%, -4.2%]<br>最好：C4r-32768（4.02e+03 GiB/h、使用率 18.1%）；O：4.28e+03 GiB/h、18.9%<br>重算 token：最好 3.14e+08／O 3.15e+08 |
| E2 | 170 | 14.2% [10.1%, 18.4%]<br>最好：D-idle-1800（394 GiB/h、使用率 3.3%）；O：338 GiB/h、2.5%<br>重算 token：最好 1.64e+08／O 1.6e+08<br>⚑ SSD 不忙（C0 使用率 3.6%<5%；GO、NO-GO 都不算） |
| E2 | 435 | -111.7% [-156.3%, -67.2%]<br>最好：C1-2（31.3 GiB/h、使用率 0.5%）；O：66.2 GiB/h、0.7%<br>重算 token：最好 1.59e+08／O 1.54e+08<br>⚑ SSD 不忙（C0 使用率 2.3%<5%；GO、NO-GO 都不算） |
| E3 | 63 | -6.1% [-8.2%, -3.9%]<br>最好：C4r-32768（4.03e+03 GiB/h、使用率 18.4%）；O：4.28e+03 GiB/h、18.9%<br>重算 token：最好 3.22e+08／O 3.22e+08 |
| E3 | 170 | 23.5% [17.8%, 29.1%]<br>最好：C4r-16384（451 GiB/h、使用率 4.2%）；O：345 GiB/h、2.6%<br>重算 token：最好 1.65e+08／O 1.61e+08<br>⚑ SSD 不忙（C0 使用率 4.1%<5%；GO、NO-GO 都不算） |
| E3 | 435 | -76.2% [-104.4%, -47.9%]<br>最好：C1-2（44.7 GiB/h、使用率 0.7%）；O：78.8 GiB/h、0.9%<br>重算 token：最好 1.59e+08／O 1.54e+08<br>⚑ SSD 不忙（C0 使用率 3.2%<5%；GO、NO-GO 都不算） |
| E4 | 63 | -6.0% [-8.1%, -3.9%]<br>最好：C4r-32768（4e+03 GiB/h、使用率 18.2%）；O：4.24e+03 GiB/h、18.7%<br>重算 token：最好 3.34e+08／O 3.35e+08 |
| E4 | 170 | 22.9% [17.3%, 28.6%]<br>最好：C4r-16384（442 GiB/h、使用率 4.1%）；O：341 GiB/h、2.6%<br>重算 token：最好 1.65e+08／O 1.61e+08<br>⚑ SSD 不忙（C0 使用率 4.0%<5%；GO、NO-GO 都不算） |
| E4 | 435 | -77.3% [-106.1%, -48.5%]<br>最好：C1-2（43.5 GiB/h、使用率 0.6%）；O：77.2 GiB/h、0.9%<br>重算 token：最好 1.59e+08／O 1.54e+08<br>⚑ SSD 不忙（C0 使用率 3.1%<5%；GO、NO-GO 都不算） |
| E5-T1800 | 63 | -6.5% [-8.7%, -4.2%]<br>最好：C4r-32768（4.02e+03 GiB/h、使用率 18.1%）；O：4.28e+03 GiB/h、18.9%<br>重算 token：最好 3.14e+08／O 3.15e+08 |
| E5-T1800 | 170 | 14.9% [10.7%, 19.1%]<br>最好：D-idle-1800（398 GiB/h、使用率 3.4%）；O：339 GiB/h、2.5%<br>重算 token：最好 1.64e+08／O 1.6e+08<br>⚑ SSD 不忙（C0 使用率 3.6%<5%；GO、NO-GO 都不算） |
| E5-T1800 | 435 | 34.4% [27.6%, 41.2%]<br>最好：D-idle-1800（215 GiB/h、使用率 2.5%）；O：141 GiB/h、1.3%<br>重算 token：最好 1.57e+08／O 1.55e+08<br>⚑ SSD 不忙（C0 使用率 2.7%<5%；GO、NO-GO 都不算） |
| E6 | 63 | -7.5% [-9.0%, -5.9%]<br>最好：C4r-32768（3.11e+03 GiB/h、使用率 14.3%）；O：3.34e+03 GiB/h、14.9%<br>重算 token：最好 2.09e+08／O 2.07e+08 |
| E6 | 170 | 12.6% [9.3%, 15.8%]<br>最好：D-idle-1800（378 GiB/h、使用率 3.2%）；O：330 GiB/h、2.5%<br>重算 token：最好 1.43e+08／O 1.4e+08<br>⚑ SSD 不忙（C0 使用率 3.5%<5%；GO、NO-GO 都不算） |
| E6 | 435 | 55.3% [48.3%, 62.2%]<br>最好：D-len-32768（180 GiB/h、使用率 2.3%）；O：80.6 GiB/h、0.9%<br>重算 token：最好 1.36e+08／O 1.35e+08<br>⚑ SSD 不忙（C0 使用率 2.6%<5%；GO、NO-GO 都不算） |

## 15 個主格：省多少＝(最好的對手 − O) ÷ 最好的對手；截尾 2H（只報告）；點估計［同時 CI 下界, 上界］

| 外層做法 | 記憶體 GiB | synth（合成） |
|:--|--:|:--|
| E2 | 63 | -6.8% [-8.9%, -4.7%] |
| E2 | 170 | 14.0% [10.0%, 18.0%] |
| E2 | 435 | -111.6% [-159.0%, -64.2%] |
| E3 | 63 | -6.4% [-8.4%, -4.5%] |
| E3 | 170 | 23.7% [18.0%, 29.4%] |
| E3 | 435 | -77.2% [-107.0%, -47.4%] |
| E4 | 63 | -6.3% [-8.3%, -4.4%] |
| E4 | 170 | 23.2% [17.5%, 28.9%] |
| E4 | 435 | -78.4% [-108.7%, -48.0%] |
| E5-T1800 | 63 | -6.8% [-8.9%, -4.7%] |
| E5-T1800 | 170 | 14.7% [10.6%, 18.7%] |
| E5-T1800 | 435 | 34.1% [27.1%, 41.2%] |
| E6 | 63 | -7.8% [-9.3%, -6.4%] |
| E6 | 170 | 12.5% [9.1%, 15.8%] |
| E6 | 435 | 55.4% [48.1%, 62.7%] |

## 護欄的參考（兩種都報告；點估計；H；主格）：C0＝對手要 ≤1.05×C0；min(C0, O)＝要 ≤1.05×C0 與 O 逐項取小

| 外層做法 | 記憶體 GiB | 參考 | synth |
|:--|--:|:--|:--|
| E2 | 63 | C0 | -6.5%（C4r-32768） |
| E2 | 63 | min_C0_O | -6.5%（C4r-32768） |
| E2 | 170 | C0 | 14.2%（D-idle-1800） ⚑SSD 不忙（C0 使用率 3.6%<5%；GO、NO-GO 都不算） |
| E2 | 170 | min_C0_O | 14.2%（D-idle-1800） ⚑SSD 不忙（C0 使用率 3.6%<5%；GO、NO-GO 都不算） |
| E2 | 435 | C0 | -111.7%（C1-2） ⚑SSD 不忙（C0 使用率 2.3%<5%；GO、NO-GO 都不算） |
| E2 | 435 | min_C0_O | -111.7%（C1-2） ⚑SSD 不忙（C0 使用率 2.3%<5%；GO、NO-GO 都不算） |
| E3 | 63 | C0 | -6.1%（C4r-32768） |
| E3 | 63 | min_C0_O | -6.1%（C4r-32768） |
| E3 | 170 | C0 | 23.5%（C4r-16384） ⚑SSD 不忙（C0 使用率 4.1%<5%；GO、NO-GO 都不算） |
| E3 | 170 | min_C0_O | 23.5%（C4r-16384） ⚑SSD 不忙（C0 使用率 4.1%<5%；GO、NO-GO 都不算） |
| E3 | 435 | C0 | -76.2%（C1-2） ⚑SSD 不忙（C0 使用率 3.2%<5%；GO、NO-GO 都不算） |
| E3 | 435 | min_C0_O | -76.2%（C1-2） ⚑SSD 不忙（C0 使用率 3.2%<5%；GO、NO-GO 都不算） |
| E4 | 63 | C0 | -6.0%（C4r-32768） |
| E4 | 63 | min_C0_O | -6.0%（C4r-32768） |
| E4 | 170 | C0 | 22.9%（C4r-16384） ⚑SSD 不忙（C0 使用率 4.0%<5%；GO、NO-GO 都不算） |
| E4 | 170 | min_C0_O | 22.9%（C4r-16384） ⚑SSD 不忙（C0 使用率 4.0%<5%；GO、NO-GO 都不算） |
| E4 | 435 | C0 | -77.3%（C1-2） ⚑SSD 不忙（C0 使用率 3.1%<5%；GO、NO-GO 都不算） |
| E4 | 435 | min_C0_O | -77.3%（C1-2） ⚑SSD 不忙（C0 使用率 3.1%<5%；GO、NO-GO 都不算） |
| E5-T1800 | 63 | C0 | -6.5%（C4r-32768） |
| E5-T1800 | 63 | min_C0_O | -6.5%（C4r-32768） |
| E5-T1800 | 170 | C0 | 14.9%（D-idle-1800） ⚑SSD 不忙（C0 使用率 3.6%<5%；GO、NO-GO 都不算） |
| E5-T1800 | 170 | min_C0_O | 14.9%（D-idle-1800） ⚑SSD 不忙（C0 使用率 3.6%<5%；GO、NO-GO 都不算） |
| E5-T1800 | 435 | C0 | 34.4%（D-idle-1800） ⚑SSD 不忙（C0 使用率 2.7%<5%；GO、NO-GO 都不算） |
| E5-T1800 | 435 | min_C0_O | 34.4%（D-idle-1800） ⚑SSD 不忙（C0 使用率 2.7%<5%；GO、NO-GO 都不算） |
| E6 | 63 | C0 | -7.5%（C4r-32768） |
| E6 | 63 | min_C0_O | -7.5%（C4r-32768） |
| E6 | 170 | C0 | 12.6%（D-idle-1800） ⚑SSD 不忙（C0 使用率 3.5%<5%；GO、NO-GO 都不算） |
| E6 | 170 | min_C0_O | 12.6%（D-idle-1800） ⚑SSD 不忙（C0 使用率 3.5%<5%；GO、NO-GO 都不算） |
| E6 | 435 | C0 | 55.3%（D-len-32768） ⚑SSD 不忙（C0 使用率 2.6%<5%；GO、NO-GO 都不算） |
| E6 | 435 | min_C0_O | 55.3%（D-len-32768） ⚑SSD 不忙（C0 使用率 2.6%<5%；GO、NO-GO 都不算） |

## 寬鬆版雙胞胎（只報告；點估計；H；O 與所有對手都用寬鬆品質規則＝32K 過關的格式任何長度都過關）

| 外層做法 | 記憶體 GiB | synth |
|:--|--:|:--|
| E2 | 63 | 11.9%（C2） |
| E2 | 170 | 32.9%（C2） ⚑SSD 不忙（C0 使用率 3.6%<5%；GO、NO-GO 都不算） |
| E2 | 435 | 38.1%（C1-2） ⚑SSD 不忙（C0 使用率 2.3%<5%；GO、NO-GO 都不算） |
| E3 | 63 | 15.2%（C2） |
| E3 | 170 | 35.0%（C2） ⚑SSD 不忙（C0 使用率 4.1%<5%；GO、NO-GO 都不算） |
| E3 | 435 | 9.0%（C1-2） ⚑SSD 不忙（C0 使用率 3.2%<5%；GO、NO-GO 都不算） |
| E4 | 63 | 15.0%（C2） |
| E4 | 170 | 35.0%（C2） ⚑SSD 不忙（C0 使用率 4.0%<5%；GO、NO-GO 都不算） |
| E4 | 435 | 9.1%（C1-2） ⚑SSD 不忙（C0 使用率 3.1%<5%；GO、NO-GO 都不算） |
| E5-T1800 | 63 | 11.6%（C2） |
| E5-T1800 | 170 | 18.9%（C2） ⚑SSD 不忙（C0 使用率 3.6%<5%；GO、NO-GO 都不算） |
| E5-T1800 | 435 | 18.9%（C1） ⚑SSD 不忙（C0 使用率 2.7%<5%；GO、NO-GO 都不算） |
| E6 | 63 | 11.8%（C2） |
| E6 | 170 | 31.9%（C2） ⚑SSD 不忙（C0 使用率 3.5%<5%；GO、NO-GO 都不算） |
| E6 | 435 | 39.5%（C2） ⚑SSD 不忙（C0 使用率 2.6%<5%；GO、NO-GO 都不算） |

## 護欄敏感度（只報告；點估計；H；主格；參考＝C0）：護欄 1.05 是登記的（判定用），其他只用來看判定有多依賴 5%

| 外層做法 | 記憶體 GiB | 護欄 | synth |
|:--|--:|:--|:--|
| E2 | 63 | 1.05 | -6.5%（C4r-32768） |
| E2 | 63 | 1.10 | -6.5%（C4r-32768） |
| E2 | 63 | 1.20 | -6.5%（C4r-32768） |
| E2 | 63 | 不設 | -4375.9%（C2） |
| E2 | 170 | 1.05 | 14.2%（D-idle-1800） |
| E2 | 170 | 1.10 | 8.3%（C4r-32768） |
| E2 | 170 | 1.20 | 8.3%（C4r-32768） |
| E2 | 170 | 不設 | -592.5%（C2） |
| E2 | 435 | 1.05 | -111.7%（C1-2） |
| E2 | 435 | 1.10 | -111.7%（C1-2） |
| E2 | 435 | 1.20 | -111.7%（C1-2） |
| E2 | 435 | 不設 | -111.7%（C1-2） |
| E3 | 63 | 1.05 | -6.1%（C4r-32768） |
| E3 | 63 | 1.10 | -6.1%（C4r-32768） |
| E3 | 63 | 1.20 | -6.1%（C4r-32768） |
| E3 | 63 | 不設 | -412.4%（C1-2） |
| E3 | 170 | 1.05 | 23.5%（C4r-16384） |
| E3 | 170 | 1.10 | 17.1%（C4r-32768） |
| E3 | 170 | 1.20 | 15.3%（D-len-32768） |
| E3 | 170 | 不設 | -302.7%（C1-2） |
| E3 | 435 | 1.05 | -76.2%（C1-2） |
| E3 | 435 | 1.10 | -76.2%（C1-2） |
| E3 | 435 | 1.20 | -76.2%（C1-2） |
| E3 | 435 | 不設 | -76.2%（C1-2） |
| E4 | 63 | 1.05 | -6.0%（C4r-32768） |
| E4 | 63 | 1.10 | -6.0%（C4r-32768） |
| E4 | 63 | 1.20 | -6.0%（C4r-32768） |
| E4 | 63 | 不設 | -413.0%（C1-2） |
| E4 | 170 | 1.05 | 22.9%（C4r-16384） |
| E4 | 170 | 1.10 | 16.7%（C4r-32768） |
| E4 | 170 | 1.20 | 15.2%（D-len-32768） |
| E4 | 170 | 不設 | -305.4%（C1-2） |
| E4 | 435 | 1.05 | -77.3%（C1-2） |
| E4 | 435 | 1.10 | -77.3%（C1-2） |
| E4 | 435 | 1.20 | -77.3%（C1-2） |
| E4 | 435 | 不設 | -77.3%（C1-2） |
| E5-T1800 | 63 | 1.05 | -6.5%（C4r-32768） |
| E5-T1800 | 63 | 1.10 | -6.5%（C4r-32768） |
| E5-T1800 | 63 | 1.20 | -6.5%（C4r-32768） |
| E5-T1800 | 63 | 不設 | -4324.3%（C2） |
| E5-T1800 | 170 | 1.05 | 14.9%（D-idle-1800） |
| E5-T1800 | 170 | 1.10 | 8.3%（C4r-32768） |
| E5-T1800 | 170 | 1.20 | 8.3%（C4r-32768） |
| E5-T1800 | 170 | 不設 | -377.4%（C2） |
| E5-T1800 | 435 | 1.05 | 34.4%（D-idle-1800） |
| E5-T1800 | 435 | 1.10 | 34.4%（D-idle-1800） |
| E5-T1800 | 435 | 1.20 | -240.1%（C1-2） |
| E5-T1800 | 435 | 不設 | -240.1%（C1-2） |
| E6 | 63 | 1.05 | -7.5%（C4r-32768） |
| E6 | 63 | 1.10 | -7.5%（C4r-32768） |
| E6 | 63 | 1.20 | -7.5%（C4r-32768） |
| E6 | 63 | 不設 | -9402.3%（C2） |
| E6 | 170 | 1.05 | 12.6%（D-idle-1800） |
| E6 | 170 | 1.10 | 11.8%（C4r-32768） |
| E6 | 170 | 1.20 | 11.8%（C4r-32768） |
| E6 | 170 | 不設 | -7280.3%（C2） |
| E6 | 435 | 1.05 | 55.3%（D-len-32768） |
| E6 | 435 | 1.10 | 55.3%（D-len-32768） |
| E6 | 435 | 1.20 | 55.3%（D-len-32768） |
| E6 | 435 | 不設 | -27790.4%（C2） |

## 整張地圖（只有點估計，H；只報告）

### E2@block

| 記憶體 GiB | synth |
|--:|--:|
| 63 | -7.6% |
| 170 | 13.0% |
| 435 | 55.3% |

### E2@session

| 記憶體 GiB | synth |
|--:|--:|
| 63 | -6.5% |
| 170 | 14.2% |
| 435 | -111.7% |

### E3@session

| 記憶體 GiB | synth |
|--:|--:|
| 63 | -6.1% |
| 170 | 23.5% |
| 435 | -76.2% |

### E4@session

| 記憶體 GiB | synth |
|--:|--:|
| 63 | -6.0% |
| 170 | 22.9% |
| 435 | -77.3% |

### E5-T1800@session

| 記憶體 GiB | synth |
|--:|--:|
| 63 | -6.5% |
| 170 | 14.9% |
| 435 | 34.4% |

### E6@block

| 記憶體 GiB | synth |
|--:|--:|
| 63 | -7.5% |
| 170 | 12.6% |
| 435 | 55.3% |

## 其他視角（只報告；點估計；H；主視角以外的時間設定與 GPU 忙）

| 資料 | 外層 | 記憶體 | 視角 | 最好的對手 | 省多少 |
|:--|:--|--:|:--|:--|--:|
| synth | E2@session | 63 | mi300x|cake|gpu=busy|cpu=idle|ssd/1|mi300x_local | C4r-32768 | -6.5% |
| synth | E2@session | 63 | mi300x|cake|gpu=busy|cpu=idle|ssd/2|mi300x_local | C4r-32768 | -6.5% |
| synth | E2@session | 63 | mi300x|cake|gpu=busy|cpu=idle|ssd/4|mi300x_local | C4r-32768 | -6.5% |
| synth | E2@session | 63 | mi300x|cake|gpu=busy|cpu=idle|ssd/8|mi300x_local | C1-2 | -416.1% |
| synth | E2@session | 63 | mi300x|cake|gpu=idle|cpu=idle|ssd/2|mi300x_local | C4r-32768 | -6.5% |
| synth | E2@session | 63 | mi300x|cake|gpu=idle|cpu=idle|ssd/4|mi300x_local | C4r-32768 | -6.5% |
| synth | E2@session | 63 | mi300x|cake|gpu=idle|cpu=idle|ssd/8|mi300x_local | C1-2 | -416.1% |
| synth | E2@session | 170 | mi300x|cake|gpu=busy|cpu=idle|ssd/1|mi300x_local | D-idle-1800 | 14.2% |
| synth | E2@session | 170 | mi300x|cake|gpu=busy|cpu=idle|ssd/2|mi300x_local | D-idle-1800 | 14.2% |
| synth | E2@session | 170 | mi300x|cake|gpu=busy|cpu=idle|ssd/4|mi300x_local | D-idle-1800 | 14.2% |
| synth | E2@session | 170 | mi300x|cake|gpu=busy|cpu=idle|ssd/8|mi300x_local | C4r-32768 | 8.3% |
| synth | E2@session | 170 | mi300x|cake|gpu=idle|cpu=idle|ssd/2|mi300x_local | C4r-32768 | 8.3% |
| synth | E2@session | 170 | mi300x|cake|gpu=idle|cpu=idle|ssd/4|mi300x_local | C4r-32768 | 8.3% |
| synth | E2@session | 170 | mi300x|cake|gpu=idle|cpu=idle|ssd/8|mi300x_local | C4r-32768 | 8.3% |
| synth | E2@session | 435 | mi300x|cake|gpu=busy|cpu=idle|ssd/1|mi300x_local | D-idle-3600 | -5.3% |
| synth | E2@session | 435 | mi300x|cake|gpu=busy|cpu=idle|ssd/2|mi300x_local | D-idle-3600 | -5.3% |
| synth | E2@session | 435 | mi300x|cake|gpu=busy|cpu=idle|ssd/4|mi300x_local | C1-2 | -111.7% |
| synth | E2@session | 435 | mi300x|cake|gpu=busy|cpu=idle|ssd/8|mi300x_local | C1-2 | -111.7% |
| synth | E2@session | 435 | mi300x|cake|gpu=idle|cpu=idle|ssd/2|mi300x_local | C1-2 | -111.7% |
| synth | E2@session | 435 | mi300x|cake|gpu=idle|cpu=idle|ssd/4|mi300x_local | C1-2 | -111.7% |
| synth | E2@session | 435 | mi300x|cake|gpu=idle|cpu=idle|ssd/8|mi300x_local | C1-2 | -111.7% |
| synth | E3@session | 63 | mi300x|cake|gpu=busy|cpu=idle|ssd/1|mi300x_local | C4r-32768 | -6.1% |
| synth | E3@session | 63 | mi300x|cake|gpu=busy|cpu=idle|ssd/2|mi300x_local | C4r-32768 | -6.1% |
| synth | E3@session | 63 | mi300x|cake|gpu=busy|cpu=idle|ssd/4|mi300x_local | C4r-32768 | -6.1% |
| synth | E3@session | 63 | mi300x|cake|gpu=busy|cpu=idle|ssd/8|mi300x_local | C1-2 | -412.4% |
| synth | E3@session | 63 | mi300x|cake|gpu=idle|cpu=idle|ssd/2|mi300x_local | C4r-32768 | -6.1% |
| synth | E3@session | 63 | mi300x|cake|gpu=idle|cpu=idle|ssd/4|mi300x_local | C4r-32768 | -6.1% |
| synth | E3@session | 63 | mi300x|cake|gpu=idle|cpu=idle|ssd/8|mi300x_local | C1-2 | -412.4% |
| synth | E3@session | 170 | mi300x|cake|gpu=busy|cpu=idle|ssd/1|mi300x_local | C0 | 24.5% |
| synth | E3@session | 170 | mi300x|cake|gpu=busy|cpu=idle|ssd/2|mi300x_local | C0 | 24.5% |
| synth | E3@session | 170 | mi300x|cake|gpu=busy|cpu=idle|ssd/4|mi300x_local | C4r-16384 | 23.5% |
| synth | E3@session | 170 | mi300x|cake|gpu=busy|cpu=idle|ssd/8|mi300x_local | C4r-32768 | 17.1% |
| synth | E3@session | 170 | mi300x|cake|gpu=idle|cpu=idle|ssd/2|mi300x_local | C4r-16384 | 23.5% |
| synth | E3@session | 170 | mi300x|cake|gpu=idle|cpu=idle|ssd/4|mi300x_local | C4r-32768 | 17.1% |
| synth | E3@session | 170 | mi300x|cake|gpu=idle|cpu=idle|ssd/8|mi300x_local | C4r-32768 | 17.1% |
| synth | E3@session | 435 | mi300x|cake|gpu=busy|cpu=idle|ssd/1|mi300x_local | D-len-16384 | 65.2% |
| synth | E3@session | 435 | mi300x|cake|gpu=busy|cpu=idle|ssd/2|mi300x_local | D-len-16384 | 65.2% |
| synth | E3@session | 435 | mi300x|cake|gpu=busy|cpu=idle|ssd/4|mi300x_local | C1-2 | -76.2% |
| synth | E3@session | 435 | mi300x|cake|gpu=busy|cpu=idle|ssd/8|mi300x_local | C1-2 | -76.2% |
| synth | E3@session | 435 | mi300x|cake|gpu=idle|cpu=idle|ssd/2|mi300x_local | C1-2 | -76.2% |
| synth | E3@session | 435 | mi300x|cake|gpu=idle|cpu=idle|ssd/4|mi300x_local | C1-2 | -76.2% |
| synth | E3@session | 435 | mi300x|cake|gpu=idle|cpu=idle|ssd/8|mi300x_local | C1-2 | -76.2% |
| synth | E4@session | 63 | mi300x|cake|gpu=busy|cpu=idle|ssd/1|mi300x_local | C4r-32768 | -6.0% |
| synth | E4@session | 63 | mi300x|cake|gpu=busy|cpu=idle|ssd/2|mi300x_local | C4r-32768 | -6.0% |
| synth | E4@session | 63 | mi300x|cake|gpu=busy|cpu=idle|ssd/4|mi300x_local | C4r-32768 | -6.0% |
| synth | E4@session | 63 | mi300x|cake|gpu=busy|cpu=idle|ssd/8|mi300x_local | C1-2 | -413.0% |
| synth | E4@session | 63 | mi300x|cake|gpu=idle|cpu=idle|ssd/2|mi300x_local | C4r-32768 | -6.0% |
| synth | E4@session | 63 | mi300x|cake|gpu=idle|cpu=idle|ssd/4|mi300x_local | C4r-32768 | -6.0% |
| synth | E4@session | 63 | mi300x|cake|gpu=idle|cpu=idle|ssd/8|mi300x_local | C1-2 | -413.0% |
| synth | E4@session | 170 | mi300x|cake|gpu=busy|cpu=idle|ssd/1|mi300x_local | C0 | 24.0% |
| synth | E4@session | 170 | mi300x|cake|gpu=busy|cpu=idle|ssd/2|mi300x_local | C0 | 24.0% |
| synth | E4@session | 170 | mi300x|cake|gpu=busy|cpu=idle|ssd/4|mi300x_local | C4r-16384 | 22.9% |
| synth | E4@session | 170 | mi300x|cake|gpu=busy|cpu=idle|ssd/8|mi300x_local | C4r-32768 | 16.7% |
| synth | E4@session | 170 | mi300x|cake|gpu=idle|cpu=idle|ssd/2|mi300x_local | C4r-32768 | 16.7% |
| synth | E4@session | 170 | mi300x|cake|gpu=idle|cpu=idle|ssd/4|mi300x_local | C4r-32768 | 16.7% |
| synth | E4@session | 170 | mi300x|cake|gpu=idle|cpu=idle|ssd/8|mi300x_local | C4r-32768 | 16.7% |
| synth | E4@session | 435 | mi300x|cake|gpu=busy|cpu=idle|ssd/1|mi300x_local | D-len-16384 | 65.0% |
| synth | E4@session | 435 | mi300x|cake|gpu=busy|cpu=idle|ssd/2|mi300x_local | D-len-16384 | 65.0% |
| synth | E4@session | 435 | mi300x|cake|gpu=busy|cpu=idle|ssd/4|mi300x_local | C1-2 | -77.3% |
| synth | E4@session | 435 | mi300x|cake|gpu=busy|cpu=idle|ssd/8|mi300x_local | C1-2 | -77.3% |
| synth | E4@session | 435 | mi300x|cake|gpu=idle|cpu=idle|ssd/2|mi300x_local | C1-2 | -77.3% |
| synth | E4@session | 435 | mi300x|cake|gpu=idle|cpu=idle|ssd/4|mi300x_local | C1-2 | -77.3% |
| synth | E4@session | 435 | mi300x|cake|gpu=idle|cpu=idle|ssd/8|mi300x_local | C1-2 | -77.3% |
| synth | E5-T1800@session | 63 | mi300x|cake|gpu=busy|cpu=idle|ssd/1|mi300x_local | C4r-32768 | -6.5% |
| synth | E5-T1800@session | 63 | mi300x|cake|gpu=busy|cpu=idle|ssd/2|mi300x_local | C4r-32768 | -6.5% |
| synth | E5-T1800@session | 63 | mi300x|cake|gpu=busy|cpu=idle|ssd/4|mi300x_local | C4r-32768 | -6.5% |
| synth | E5-T1800@session | 63 | mi300x|cake|gpu=busy|cpu=idle|ssd/8|mi300x_local | C1-2 | -416.1% |
| synth | E5-T1800@session | 63 | mi300x|cake|gpu=idle|cpu=idle|ssd/2|mi300x_local | C4r-32768 | -6.5% |
| synth | E5-T1800@session | 63 | mi300x|cake|gpu=idle|cpu=idle|ssd/4|mi300x_local | C4r-32768 | -6.5% |
| synth | E5-T1800@session | 63 | mi300x|cake|gpu=idle|cpu=idle|ssd/8|mi300x_local | C1-2 | -416.1% |
| synth | E5-T1800@session | 170 | mi300x|cake|gpu=busy|cpu=idle|ssd/1|mi300x_local | D-idle-1800 | 14.9% |
| synth | E5-T1800@session | 170 | mi300x|cake|gpu=busy|cpu=idle|ssd/2|mi300x_local | D-idle-1800 | 14.9% |
| synth | E5-T1800@session | 170 | mi300x|cake|gpu=busy|cpu=idle|ssd/4|mi300x_local | C4r-32768 | 8.3% |
| synth | E5-T1800@session | 170 | mi300x|cake|gpu=busy|cpu=idle|ssd/8|mi300x_local | C4r-32768 | 8.3% |
| synth | E5-T1800@session | 170 | mi300x|cake|gpu=idle|cpu=idle|ssd/2|mi300x_local | C4r-32768 | 8.3% |
| synth | E5-T1800@session | 170 | mi300x|cake|gpu=idle|cpu=idle|ssd/4|mi300x_local | C4r-32768 | 8.3% |
| synth | E5-T1800@session | 170 | mi300x|cake|gpu=idle|cpu=idle|ssd/8|mi300x_local | C4r-32768 | 8.3% |
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
| sens_3090 | synth | E2@session | 63 | C2 | -4375.9% |
| sens_3090_ssd8 | synth | E2@session | 63 | C2 | -4375.9% |
| sens_cpu1.35 | synth | E2@session | 63 | C4r-32768 | -6.5% |
| sens_cpu1.81 | synth | E2@session | 63 | C4r-32768 | -6.5% |
| sens_3090 | synth | E2@session | 170 | C2 | -592.5% |
| sens_3090_ssd8 | synth | E2@session | 170 | C2 | -592.5% |
| sens_cpu1.35 | synth | E2@session | 170 | C4r-32768 | 8.3% |
| sens_cpu1.81 | synth | E2@session | 170 | C4r-32768 | 8.3% |
| sens_3090 | synth | E2@session | 435 | C1-2 | -111.7% |
| sens_3090_ssd8 | synth | E2@session | 435 | C1-2 | -111.7% |
| sens_cpu1.35 | synth | E2@session | 435 | C1-2 | -111.7% |
| sens_cpu1.81 | synth | E2@session | 435 | C1-2 | -111.7% |
| sens_3090 | synth | E3@session | 63 | C1 | -204.9% |
| sens_3090_ssd8 | synth | E3@session | 63 | C1-2 | -412.4% |
| sens_cpu1.35 | synth | E3@session | 63 | C4r-32768 | -6.1% |
| sens_cpu1.81 | synth | E3@session | 63 | C4r-32768 | -6.1% |
| sens_3090 | synth | E3@session | 170 | C1 | -134.1% |
| sens_3090_ssd8 | synth | E3@session | 170 | C1-2 | -302.7% |
| sens_cpu1.35 | synth | E3@session | 170 | C4r-32768 | 17.1% |
| sens_cpu1.81 | synth | E3@session | 170 | C4r-32768 | 17.1% |
| sens_3090 | synth | E3@session | 435 | C1-2 | -76.2% |
| sens_3090_ssd8 | synth | E3@session | 435 | C1-2 | -76.2% |
| sens_cpu1.35 | synth | E3@session | 435 | C1-2 | -76.2% |
| sens_cpu1.81 | synth | E3@session | 435 | C1-2 | -76.2% |
| sens_3090 | synth | E4@session | 63 | C1 | -205.3% |
| sens_3090_ssd8 | synth | E4@session | 63 | C1-2 | -413.0% |
| sens_cpu1.35 | synth | E4@session | 63 | C4r-32768 | -6.0% |
| sens_cpu1.81 | synth | E4@session | 63 | C4r-32768 | -6.0% |
| sens_3090 | synth | E4@session | 170 | C1 | -135.8% |
| sens_3090_ssd8 | synth | E4@session | 170 | C1-2 | -305.4% |
| sens_cpu1.35 | synth | E4@session | 170 | C4r-32768 | 16.7% |
| sens_cpu1.81 | synth | E4@session | 170 | C4r-32768 | 16.7% |
| sens_3090 | synth | E4@session | 435 | C1-2 | -77.3% |
| sens_3090_ssd8 | synth | E4@session | 435 | C1-2 | -77.3% |
| sens_cpu1.35 | synth | E4@session | 435 | C1-2 | -77.3% |
| sens_cpu1.81 | synth | E4@session | 435 | C1-2 | -77.3% |
| sens_3090 | synth | E5-T1800@session | 63 | C2 | -4324.3% |
| sens_3090_ssd8 | synth | E5-T1800@session | 63 | C2 | -4324.3% |
| sens_cpu1.35 | synth | E5-T1800@session | 63 | C4r-32768 | -6.5% |
| sens_cpu1.81 | synth | E5-T1800@session | 63 | C4r-32768 | -6.5% |
| sens_3090 | synth | E5-T1800@session | 170 | C2 | -377.4% |
| sens_3090_ssd8 | synth | E5-T1800@session | 170 | C2 | -377.4% |
| sens_cpu1.35 | synth | E5-T1800@session | 170 | C4r-32768 | 8.3% |
| sens_cpu1.81 | synth | E5-T1800@session | 170 | C4r-32768 | 8.3% |
| sens_3090 | synth | E5-T1800@session | 435 | C1 | -98.6% |
| sens_3090_ssd8 | synth | E5-T1800@session | 435 | C1-2 | -240.1% |
| sens_cpu1.35 | synth | E5-T1800@session | 435 | D-idle-1800 | 34.4% |
| sens_cpu1.81 | synth | E5-T1800@session | 435 | D-idle-1800 | 34.4% |

