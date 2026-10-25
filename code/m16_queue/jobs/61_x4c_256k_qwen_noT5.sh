# GPU-NEED: one
# CARDS: 0,1
# NEED-GB: 170
# MAX-H: 1
# X4c 的 Qwen 256K 補跑（10/22 使用者：「Qwen 256K 補跑」）。38i1 失敗原因：T5（cwe）在 256K 補不滿
# （m10_tasks._cwe「cwe 詞表不夠長，補不滿」，詞表是上游固定的）。做法：256K 只跑 T2–T4，**不改第二階段的 m10_tasks.py**；
# 硬把 cwe 補到 256K 等於造一個新版 T5，和 32K／128K 的 T5 本來就不能比。計畫檔、題目（seed 40000 起、每任務 6 題）、配置都和 38i1 一樣。
# 38i1 已做完：s0 的 T2 6 題＋T4 6 題、s1 的 T3 1 題（done_union 續跑會跳過）→ 這次只剩 T3 5 題〔估〕約 20–25 分鐘。
# 程式和 38i1 同一版（x4_check_smoke x4c 核對 sha；38i1 的冒煙 10/21 已過，不重跑冒煙）。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_common.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_pool.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x4_common.sh
JOB=61_x4c_256k_qwen_noT5
x4_check_smoke x4c || exit 1
q256_fill_noT5() { x4_units $1 x4q256 q256 256K 0 $Q256_SHARDS "T2 T3 T4" $Q256_N $X4C_SEED0 gen --selection $X4C_SEL --n-blocks 504; }
x4_run_pool 60 q256_fill_noT5 || exit 1
echo "$JOB 結束 $(date -u +%FT%TZ)"
