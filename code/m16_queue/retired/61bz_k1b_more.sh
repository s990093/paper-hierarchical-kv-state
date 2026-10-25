# GPU-NEED: none
# AFTER: 61b_k1b_more_dumps
# K1b 加題的 CPU 分析：6 份新 dump 各一個行程（每個 8 核、nice 19；不 import torch），然後 report（10 份一起）。
# 輸出 results/m16_stage3_mi300x/k1_kvstruct/tucker/（tucker_<dump>.json、summary.*、op_points.csv、xlayer_subspace.csv、圖）。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
JOB=61bz_k1b_more
OUT=$RES/k1_kvstruct/tucker
c=64; pids=()
for d in llama_T3_64 llama_T5_64 llama_T4b_64 qwen_T3_64 qwen_T5_64 qwen_T4b_64; do
  ( OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 m7run m16-k1b-tucker-$d nice -n 19 taskset -c $c-$((c+7)) python -I -X faulthandler code/m16_k1_tucker.py run \
      --dump $KVD/$d --out $OUT --spectra /mlsteam/data/tiara/stage3/k1/tucker_spectra_$d.npz ) & pids+=($!)
  c=$((c+8))
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
[ $rc = 0 ] || { nh "$JOB：有 dump 的 K1b 失敗（看 $TIARA_RUNS/*m16-k1b-tucker-*）"; exit 1; }
m7run m16-k1b-report python -I code/m16_k1_tucker.py report --in $OUT || { nh "$JOB：report 失敗（$(lastrun m16-k1b-report)）"; exit 1; }
echo "$JOB 結束 $(date -u +%FT%TZ)"
