# GPU-NEED: both
# GPU: 1
# NEED-GB: 80
# MAX-H: 1
# 第三階段 GPU 程式的冒煙（每支都用最小的設定跑一次；輸出到 $SMOKE，不進 results）。任何一支失敗 → NEEDS_HUMAN、exit 1（後面的 job 都靠這幾支程式）。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
mkdir -p $SMOKE
m7run m16-smoke-kvdump python code/m16_s1_kvdump.py --model llama --task T4 --n-blocks 4 --out $SMOKE/kv_llama_t4_4 \
  || { nh "00_smoke: m16_s1_kvdump 失敗（$(lastrun m16-smoke-kvdump)）"; exit 1; }
m7run m16-smoke-lossless taskset -c 0-3 python code/m16_s1_lossless.py analyze --dump $SMOKE/kv_llama_t4_4 --out $SMOKE/s1 \
  --max-blocks 2 --speed-blocks 1 --reps 1 --threads 1 --speed-codecs zstd:1 \
  || { nh "00_smoke: m16_s1_lossless 失敗（$(lastrun m16-smoke-lossless)）"; exit 1; }
m7run m16-smoke-i4 python code/m16_i4_copybench.py --blocks 64 --reps 1 --gpu-states idle --py-loop-max 64 --out $SMOKE/i4 \
  || { nh "00_smoke: m16_i4_copybench 失敗（$(lastrun m16-smoke-i4)）"; exit 1; }
m7run m16-smoke-i1interf python code/m16_i1_multisession.py interf --seconds 1 --gpu-states idle --out $SMOKE/i1 \
  || { nh "00_smoke: i1 interf 失敗（$(lastrun m16-smoke-i1interf)）"; exit 1; }
m7run m16-smoke-i1grid python code/m16_i1_multisession.py grid --N 1 2 --workloads homog_cpu homog_b32 --modes cake g5b \
  --orders rr fcfs --reps 1 --gpu-states idle --out $SMOKE/i1 \
  || { nh "00_smoke: i1 grid 失敗（$(lastrun m16-smoke-i1grid)）"; exit 1; }
echo "00_smoke 全過 $(date -u +%FT%TZ)"
