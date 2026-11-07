# v8 探索的結果（不是正式結果；語意還沒登記）

由 icpe-rux 定的放法，其他 session（icpe-next-trx 等）照同一套。

**位置**
- repo：`paper-hierarchical-kv-state`。每個 session 放自己的分支：icpe-rux 在 `icpe-next-rux`，TRX40 在 `icpe-next-trx`。
- 路徑：`results/icpe_next/v8_explore/<實驗>/<變體>/`
  - 實驗：`apc`（APC／BELP）、`w3`（線索）、`w5`、`queue`（排隊的一階近似 qpost）、`slowcap`
  - 變體：紀錄名稱，例如 `semianalysis_N16_s1`、`mooncake_toolagent`（R1、R2 在同一個檔）、`bailian_coder`、`tracelab_c1`、`fi_2026-05-17_2026-05-23`

**每個目錄放什麼**
- sweep 的輸出：`x1_sweep.csv.xz`。W5 有兩份：`x1_sweep_bf16.csv.xz`、`x1_sweep_w5.csv.xz`。
- 分析的輸出，每列都有 run_id、ts、host：
  - APC：`explore_apc_rows.csv`、`explore_ogpu_belp_rows.csv`、`*_headroom_components.csv`
  - W3：`w3_rows.csv`
  - W5：`explore_w5_rows.csv`、`w5_mechanism_rows.csv`
  - 排隊：`queue_rows.csv`
- `MANIFEST.json`：run_id、ts、host、repo commit、v8 `main.rs` 的 sha256、sweep 裡的 code_sha、每個檔的 sha256 與列數。
- 超過 1 MB 的檔用 xz 壓。

**不放進 git**：原始資料、nb 檔、`cue.u8`、逐請求的 npz（`req/`）、job 目錄。

**收進來的工具**：`code/icpe/rux_x1fast/v8_features/collect_v8.py <實驗> <變體> --sweep … --files … --runs …`

說明與結論：`docs/icpe_brief_20261030/next/rux/V8_FEATURES.md`、`SEMANTICS_2to5.md`。
