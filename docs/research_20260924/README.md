# 2026-09-24 查詢報告（子 agent 產出）

這個資料夾是 `../DIRECTIONS_20260924.md` 的證據附件，由 4 個查詢子 agent 在 2026-09-24 產出，未經對抗式交叉審查（self-review 等級）。

| 檔案 | 內容 |
|---|---|
| `novelty_sota.md` | 新穎性審查（審稿人視角）、35 篇的 baseline 引用圖、程式碼與整合狀態、Semantic Scholar 引用數 |
| `os_toolbox.md` | OS／儲存／快取理論工具箱：約 50 條、128 個 URL；三條「一句話定律」（我們的推導）與「先做理論模型」的最小路線 |
| `kappa_map.md` | 18 篇論文的評測平台放上同一條 κ 軸；規格附 URL；DeepSeek 推理系統的 KV 設計 |
| `workloads_eval.md` | 評測設定矩陣、命中率定義稽核、公開 trace 清單（自行計算）、統一評測協定草案 v0 |
| `restore_modes.py` | 三種讀回方式（前綴語意／雙向重疊／裸頻寬）的算術，常數取自 `main.tex` 表 costmodels。**只做算術，不是量測** |

**路徑說明**：報告內的 `scratchpad/...` 路徑是當次 session 的暫存資料夾，已不存在。
- `scratchpad/papers_txt/<key>.txt`：由 `papers/<key>.pdf` 抽出的純文字（頁碼標記 `=== [page N] ===`）。
- `scratchpad/remote/...`：GitHub `origin/main` 上 2026-09-21 的檔案（`results/RUNLOG_MI300X.md`、`EXPERIMENTS_20260919.md` 等），當時本地尚未 pull。
- `scratchpad/extra_txt/`、`research/s2/`、`research/kappa_map_work/`：子 agent 下載的額外論文、Semantic Scholar 原始回應與規格擷取，未保留。
