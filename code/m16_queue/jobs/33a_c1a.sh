# GPU-NEED: one
# CARDS: 0,1
# NEED-GB: 120
# MAX-H: 1
# AFTER: 04_smoke_c1a
# C1a 正式（前半：shard 0–2／6）。本體見 c1a_common.sh；輸出 results/m16_stage3_mi300x/c1a_predictive/c1a_<模型>_s<i>.csv。
# NEED-GB：每張卡 3 個行程 × ≤ 40 GiB〔估，c1a_s2_howto.md §8 第 12 條；04 的 peak 量到之後再調〕。
# MAX-H 1〔估〕：每張卡 75 題、3 個行程；每題 20–40 s（howto §5）→ 3 個行程完全不互相拖慢 8–17 分鐘、完全沒有平行效益 25–50 分鐘，加載模型。
JOB=33a_c1a; SHARDS="0 1 2"
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/c1a_common.sh
