# GPU-NEED: both
# GPU: 1
# NEED-GB: 180
# MAX-H: 1.5
# V0：vLLM 0.28（ROCm）的 APC、CPU 卸載、tiering 能不能跑、命中與不命中輸出是否相同（docs/stage3_20261019/v0_f1_i3_howto.md §2）。
# a、b、c 不過 → F1／F2／I3g 不能用 vLLM，停（NEEDS_HUMAN）；d（ROCM_ATTN patch）、e（fs 層）不過只記下，不擋後面。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
JOBNAME=40_v0
source $REPO/code/m16_queue/jobs/vllm_lib.sh
OUT=$RES/v0_vllm_smoke; mkdir -p $OUT
TXT=/mlsteam/data/tiara/datasets/longbench/data/narrativeqa.jsonl
v0() {
  local tag=$1; shift
  srv_start v0$tag --max-model-len 34816 "$@" || return 1
  bench_run v0$tag-rnd v0check --out-dir $OUT --lengths 8192 16384 32768 --n-prompts 3 --max-tokens 32 --settle-s 3 \
    --label v0$tag-rnd || return 1
  bench_run v0$tag-txt v0check --out-dir $OUT --n-prompts 3 --max-tokens 32 --settle-s 3 --text-file $TXT --text-chars 30000 \
    --label v0$tag-txt || return 1
}
v0 a --apc off || exit 1
v0 b --apc on || exit 1
v0 c --apc on --offload cpu:64 || exit 1
v0 d --apc on --offload cpu:64 --rocm-attn-patch || nh "40_v0：V0-d（ROCM_ATTN patch）沒過 → F1／F2 不用 patch（不擋）"
rm -rf /var/tmp/m16_fs_v0e
v0 e --apc on --offload tier:16:/var/tmp/m16_fs_v0e:64 || nh "40_v0：V0-e（fs 層）沒過（不擋）"
srv_stop; rm -rf /var/tmp/m16_fs_v0e
echo "40_v0 結束 $(date -u +%FT%TZ)"
