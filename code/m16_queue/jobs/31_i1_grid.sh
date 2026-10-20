# GPU-NEED: both
# GPU: 1
# NEED-GB: 75
# MAX-H: 2
# I1 (b)：N 個 session 同時還原，實測 vs 同一 process 量的常數做的「互不干擾」預測。判準（事先寫）：實測 ÷ 預測的平均 TTFT 在每格都 ≤ 1.10 → 模型夠用，I1 收掉。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
OUT=$RES/i1_concurrent_restore; mkdir -p $OUT
m7run m16-i1-grid python code/m16_i1_multisession.py grid --model llama --N 1 4 8 \
  --workloads homog_cpu homog_b32 mix_cpu_b32 --modes cake g5b joint --orders rr fcfs srpt --reps 3 --gpu-states idle --out $OUT \
  || { nh "31_i1_grid 失敗"; exit 1; }
