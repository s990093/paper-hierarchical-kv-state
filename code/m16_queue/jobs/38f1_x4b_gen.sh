# GPU-NEED: one
# CARDS: 0,1
# NEED-GB: 170
# MAX-H: 2
# AFTER: 37z_x2_report
# X4b 第 1 段（x_stack_howto.md §15）：照事先寫死的規則從 X4a 挑配置（m16_x4 select-b：每個 T 帶內 TF 最好的＋最多 2 個「只降秩」對照＋
# q:int4_kivi；X4a 每個模型 < 48 題 → 不挑、NEEDS_HUMAN、exit 1）→ 冒煙（每卡一個模型：生成 1 題）→ dump（X4a 的 4 題 × 前緣的 xKV 配置的
# 實際係數位元組 → 本地盤 $KVD/x4，給 38fy 的無損）→ 兩卡共用工作池（新種子 65000 起、T2–T5 各 48 題＝192 題；Llama 12 份、Qwen 8 份）。
# GPU 時間〔估〕：挑選＋冒煙＋dump 10–20 分鐘；X4b 全部約 4.3–5.9 卡小時（Llama 2.5–3.7、Qwen 1.8–2.2；看挑到的秩與 W）→ 兩卡約 2.2–3.0 小時；
# 本段 ≤ 約 1.9 小時。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_common.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x_pool.sh
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/x4_common.sh
JOB=38f1_x4b_gen
T0=$(date +%s)
x4_check_smoke x4a || exit 1
m7run m16-x4-selectb $X4PY select-b --raw "$X4D/tf/x4a_*_32K_s*.csv.gz" --out $X4B_SEL \
  || { nh "$JOB：select-b 失敗（$(lastrun m16-x4-selectb)；X4a 題數不夠？）"; exit 1; }
prep() {
  x4_smoke_gen $1 $X4B_SEL 64 1 T2 $X4B_SEED0 x4b || return 1
  m7run m16-x4-dump-$1 $X4PY dump --model $1 --selection $X4B_SEL --out-dir $X4_DUMP \
    || { nh "$JOB：$1 dump 失敗（$(lastrun m16-x4-dump-$1)）"; return 1; }
}
oncard 0 prep llama & p0=$!
oncard 1 prep qwen & p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
[ $r0 = 0 ] && [ $r1 = 0 ] || exit 1
x4_mark_smoke x4b
x4_run_pool $(( X4_BUDGET - ($(date +%s) - T0) / 60 )) x4b_fill || exit 1
echo "$JOB 結束 $(date -u +%FT%TZ)"
