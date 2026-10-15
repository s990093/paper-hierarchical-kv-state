# 還原器加 H2D 計時（time_h2d，預設關）之後的 BF16 回歸（跑之前寫，2026-10-15T07:22:31Z）
- 判準同 ../regress_xproc/CRITERION.md：每 process 一個 Restorer，交替 new1 old1 new2 old2，6 格 × 5 次；KV 全部逐位元相同；每格 t_restore 中位數新/舊 ∈ [0.95, 1.05]；會合點集合相同
- 舊＝git f6595f0 的 m7_restore_harness.py
