# GPU-NEED: both
# GPU: 1
# NEED-GB: 30
# MAX-H: 1.5
# I4：KV 從 CPU 載回 GPU 的搬法 × GPU 閒／busy2；HSA_ENABLE_SDMA=1（預設：copy engine）。量時間 → 兩張卡獨佔。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
export HSA_ENABLE_SDMA=1
OUT=$RES/i4_copy_batching; mkdir -p $OUT
m7run m16-i4-sdma1 python code/m16_i4_copybench.py --reps 5 --gpu-states idle busy2 --out $OUT \
  || { nh "21_i4_sdma1 失敗"; exit 1; }
