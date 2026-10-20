# GPU-NEED: one
# CARDS: 0,1
# NEED-GB: 120
# MAX-H: 1
# AFTER: 04_smoke_c1a
# C1a 正式（後半：shard 3–5／6）。本體見 c1a_common.sh；NEED-GB、MAX-H 的依據同 33a_c1a.sh。
JOB=33b_c1a; SHARDS="3 4 5"
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/c1a_common.sh
