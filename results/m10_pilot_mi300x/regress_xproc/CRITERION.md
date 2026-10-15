# 跨 process 回歸的判準（跑之前寫，2026-10-15T02:41:16Z）
- 每個 process 只建一個 Restorer（--impls new 或 old），交替 new1 old1 new2 old2 new3 old3，每格 5 次
- 通過：所有 run 的 KV 逐位元相同；6 格各自的 t_restore 中位數（3 個 process × 5 次合併）新/舊 ∈ [0.95, 1.05]；會合點集合相同
- 原本同 process 的判準（restore_regress_bf16_llama.json）照實記為 1/6 格沒過；這一份是看到那個結果、做完診斷之後加的（事後加的檢查）
