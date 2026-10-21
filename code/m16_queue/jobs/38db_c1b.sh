# GPU-NEED: one
# CARDS: 0,1
# NEED-GB: 150
# MAX-H: 2
# AFTER: 38da_c1b
# C1b 第 2 段：接 38da_c1b 沒做完的題（逐題續跑；本體 c1b_common.sh）。38da 已全部做完 → 不佔 GPU、直接 exit 0。
# 開頭照樣先跑 selfcheck（失敗 → NEEDS_HUMAN、exit 1）；qtime 只在 38da 沒有留下乾淨的 results/…/qtime_<模型>.csv 時重量（DO_QTIME＝auto）。
# 這一段之後還有題沒做完 → 記 NEEDS_HUMAN（給排程的人決定要不要再加一段；38dz 只要每個模型 T2–T4 ≥ 48 題就能下判定）。
JOB=38db_c1b; BUDGET_MIN=105; DO_QTIME=auto
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/c1b_common.sh
st=$($PY pool-status --n $N --raw-dir $CR | head -1)
[ "$st" = "ALL_COMPLETE" ] || nh "38db_c1b: C1b 還有題沒做完（$st；時間預算用完）。38dz 照樣會跑，題數夠（每個模型 T2–T4 ≥ 48）就能下判定"
exit $rc
