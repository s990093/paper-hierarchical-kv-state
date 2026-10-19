# GPU-NEED: both
# GPU: 1
# NEED-GB: 40
# MAX-H: 1
# I1 (a)：計算線與載入線同時跑時互相拖慢多少（k_rec、k_load），GPU 閒／busy2。量時間 → 兩張卡獨佔。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
OUT=$RES/i1_concurrent_restore; mkdir -p $OUT
m7run m16-i1-interf python code/m16_i1_multisession.py interf --model llama --seconds 6 --gpu-states idle busy2 --out $OUT \
  || { nh "30_i1_interf 失敗"; exit 1; }
