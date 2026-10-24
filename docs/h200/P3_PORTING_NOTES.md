# 第三階段搬到 H200：每一個和 3090／MI300X 不同的地方（2026-10-19）

> 規矩同 `NV_PORTING_NOTES.md`（memory h200-nv-porting-notes）：每一項寫「改了什麼、會不會改數字（⚠／·）、寫論文時怎麼寫」。
> 這份只管第三階段（`origin/p3-predictor` 搬來的東西）；合回 `h200` 時在 `NV_PORTING_NOTES.md` 加一行指到這裡。執行記錄：`results/RUNLOG_H200_P3.md`。

| # | 項目 | 改了什麼 | 數字 | 論文怎麼寫 |
|:--|:--|:--|:--:|:--|
| 1 | 成本常數的來源 | `P3_CONST=h200`（`code/p3_platform.py`）：f(i)、GPU 忙、b\* 頻寬、還原中的反量化讀 H200 第一、二階段的實測；限速層（cpu／local／nfs GiB/s）與每 block 位元組沿用 MI300X 的 | ⚠ | 「H200 的模擬只換 GPU 量出來的常數；儲存層的頻寬是同一組限速值」——兩台的差別只來自 GPU |
| 2 | GPU 忙倍數 | H200 busy2＝1.945（MI300X 2.07）；背景負載的 duty 在兩台的實際值不同（NV_PORTING_NOTES 的 busy duty 項） | ⚠ | GPU 忙的情境要附各自的倍數，不要寫成「同一個忙的程度」 |
| 3 | b\* 限速頻寬 | b\* 是「會合點在第 16／32／48 個 block」的頻寬，由各台自己的 f(i) 定義：H200 b32＝2.49 GiB/s（MI300X 1.47） | ⚠ | 用 b\* 的名字比，不要用 GiB/s 比；同名 b\* 在兩台是不同的 GiB/s |
| 4 | CPU 忙的 H2D 倍數 | 3090 量到 1.35–1.81；H200 **決定不量**（1-GPU job 只有 16 CPU 的 cpuset，做不出「整台主機忙」；整台加壓會影響同節點別人；只進敏感度） | ⚠ | H200 的 CPU 忙敏感度寫 NOT_MEASURED；不能借 3090 的值 |
| 5 | 格式表 | `results/p3_formats_h200`：同一支 build_format_table、同一套規則；沒有 3090 欄；Qwen 沒有重量（regrid）；43 的選手是 H200 自己的決賽；INT4 端點品質取 `compare_mi300x_path` | ⚠ | 表的每一格附 run_id；H200 有、MI300X 沒有的格式（nvfp4）與反過來的（mxfp4）分開列 |
| 6 | Llama FP8-K0／INT8 品質 | H200 補算：重現檢查對主任務欄位逐位元相同；H200 已提交的分析是在 T1／LB shard 寫完前算的，所以含 T1 列的 `mean_lp_read` 與 run_ids 不同 → 照記不擋（RUNLOG_H200_P3 P3H-0b）。兩格四種情境都過 | · | 30／30 列的判定和 MI300X 版相同；H200 的 Llama 8-bit 判定有自己的 run_id |
| 7 | 地圖的主視角與敏感度 | `P3_MAP_PLATFORM=h200`：主視角換 H200 常數＋H200 格式表；敏感度沒有 3090 列 | · | 只有 H200 平台（memory h200-phase1-reproduction） |
| 8 | 真實裝置 | H200 沒有量本地碟／NFS（使用者 10/16「都用模擬限速」）；3090 地圖的 `/ssd8` 真實視角在 H200 不存在 | · | 只報限速層 |
| 9 | 套件版本 | numpy 2.3.5、torch 2.13.0+cu130、transformers 5.19.0（3090：2.2.6／2.10.0+cu126／5.17.0）；sklearn、statsmodels 等照 3090 同版。job 21116 的 DATA-CMP：FI 服務時間迴歸（QuantReg）的 A／C 和 3090 差 6.7e-5／7.4e-5（相對），Mooncake 的估計服務時間取整數 ms → think_ms 差 ±1 ms（summary 3 格：tool-rule-d4 p50 1016→1015 等）；其餘資料層逐位元相同（RUNLOG_H200_P3 P3H-1）。地圖：合成資料的服務時間也用這組常數 → 合成端到端（MI300X 常數）和 3090 差約 1e-3；用 3090 的常數重建後 summary／verdict 逐位元相同、tradeoff 只有 ttftq_H_mean_s 末位（≤3.5e-16）不同；C0 使用率逐值相同（P3H-1d、P3H-1e） | ⚠ | H200 用自己的迴歸常數（不借 3090 的）；Mooncake 驅動的 H200 結果和 3090 比時先看這項。DATA-DIAG-CONST3090 與 mapdiag 確認差異只來自這組常數（P3H-1b、P3H-1e） |
| 10 | 計時檔 | `run_map` 的工作順序用 3090 的計時 run（在 /ssd8）；H200 沒有 → 用請求數排 | · | 只影響先後，不影響數字 |
