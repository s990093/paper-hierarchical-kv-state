# GPU-NEED: both
# GPU: 1
# NEED-GB: 30
# MAX-H: 1.5
# I4：KV 從 CPU 載回 GPU 的搬法 × GPU 閒／busy2；HSA_ENABLE_SDMA=0（blit kernel）。量時間 → 兩張卡獨佔。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
export HSA_ENABLE_SDMA=0
OUT=$RES/i4_copy_batching; mkdir -p $OUT
m7run m16-i4-sdma0 python code/m16_i4_copybench.py --reps 5 --gpu-states idle busy2 --out $OUT \
  || { nh "20_i4_sdma0 失敗"; exit 1; }
