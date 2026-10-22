# GPU-NEED: none
# AFTER: 38f1_x4b_gen
# X4 的無損（只用 CPU；cpu 佇列）：38f1 dump 到本地盤 $KVD/x4 的實際係數＋基底位元組（X4a 的 4 題 × X4b 的 xKV 配置），用 m16_s1_lossless 的
# zstd（-3／1／3；BF16 的 part 用 planes 變體）量真實壓縮比（每次解回來逐位元比對）與解壓速度（4 MiB frame、1／4 個 thread）
# → results/m16_stage3_mi300x/x_stack/x4/lossless_x4.json。判定：zstd:3 ≥ 1.05 倍 → 值得疊（x4_limit_plan.md §3）。
# dump 不在（容器重啟、本地盤被清）→ NEEDS_HUMAN、exit 1（要重跑 38f1 的 dump 那一步）。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_common.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_pool.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x4_common.sh
JOB=38fy_x4_lossless
export OMP_NUM_THREADS=4
m7run m16-x4-lossless $X4PY lossless --dumps $X4_DUMP --out-dir $X4R || { nh "$JOB：lossless 失敗（$(lastrun m16-x4-lossless)）"; exit 1; }
echo "$JOB 結束 $(date -u +%FT%TZ)"
