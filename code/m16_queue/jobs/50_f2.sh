# GPU-NEED: both
# GPU: 1
# NEED-GB: 180
# MAX-H: 1.5
# F2：重現 vLLM #44294（命中的 chunk 正被別的請求載入 → 整個請求延後）。三段：seed → evict → burst。
# 先在 (B) P＝10K、N＝64 試三種 evict（reset、none、冷啟動），再跑主格（evict＝reset）：P∈{10K,32K} × N∈{1,8,32,64} × 3 次，
# 設定 (A) APC 不卸載、(B) APC＋CPU 卸載（connector debug）、(C) 不復用、(A′) APC＋TRITON_ATTN（和 B 同一個 kernel）。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
JOBNAME=50_f2
source $REPO/code/m16_queue/jobs/vllm_lib.sh
OUT=$RES/f2_shared_prefix_burst; mkdir -p $OUT
WD=/mlsteam/data/tiara/stage3/workloads/f2; mkdir -p $WD
for P in 10240 32768; do for N in 1 8 32 64; do
  [ -f $WD/b_${P}_${N}.jsonl ] || $W gen-burst --prefix-len $P --suffix-len 256 --n-concurrent $N --out-len 64 --seed 0 \
    --out $WD/b_${P}_${N}.jsonl || { nh "50_f2：gen-burst 失敗"; exit 1; }
done; done
grid() {   # $1＝設定名，其餘＝每個 run 額外參數
  local cfg=$1; shift
  for rep in 1 2 3; do for P in 10240 32768; do for N in 1 8 32 64; do
    bench_run f2-$cfg-P$P-N$N-r$rep run --workload $WD/b_${P}_${N}.jsonl --evict reset --phase-gap-s 3 \
      --server-log $SRD/vllm_server.log --out-dir $OUT --label f2_${cfg}_P${P}_N${N}_r$rep "$@" || return 1
  done; done; done
}
srv_start f2B --apc on --offload cpu:64 --connector-debug --max-model-len 34816 || exit 1
for ev in reset none; do
  bench_run f2-B-probe-$ev run --workload $WD/b_10240_64.jsonl --evict $ev --phase-gap-s 3 --server-log $SRD/vllm_server.log \
    --out-dir $OUT --label f2_B_probe_$ev || exit 1
done
bench_run f2-B-probe-cold run --workload $WD/b_10240_64.jsonl --phases burst --server-log $SRD/vllm_server.log \
  --out-dir $OUT --label f2_B_probe_cold || exit 1
grid B || exit 1
srv_start f2A --apc on --max-model-len 34816 || exit 1;                              grid A || exit 1
srv_start f2Ap --apc on --attention-backend TRITON_ATTN --max-model-len 34816 || exit 1; grid Ap || exit 1
srv_start f2C --apc off --max-model-len 34816 || exit 1;                             grid C || exit 1
srv_stop
echo "50_f2 結束 $(date -u +%FT%TZ)"
