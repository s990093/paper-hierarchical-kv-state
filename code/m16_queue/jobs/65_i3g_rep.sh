# GPU-NEED: both
# GPU: 1
# CARDS: 0,1
# NEED-GB: 180
# MAX-H: 0.95
# I3g 重跑（10/23 00:5xZ 使用者選「I3g b/c 重跑」）：I3g 的 c 對 b（本地檔案層好 18%）每組只跑一次、session bootstrap 的 CI 碰到 1。
# 同一份工作負載、同一組 serve 參數，b、c 各再跑兩次；兩張卡同時、交叉順序：卡 0 b→c、卡 1 c→b（兩張卡同時跑會共用主機記憶體與 PCIe，
# 和 64_i3g 單卡跑的條件不同 → 照實寫）。判定寫在 experiment_log.md「I3g 重跑（65）」（排進佇列前寫）。輸出 i3g_agentic/rep/（不動 64 的檔）。
# GPU 時間〔估〕：每個設定約 17 分（server 2–3 分＋工作負載約 14 分）→ 約 35 分。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
JOBNAME=65_i3g_rep
MODEL=llama
source $REPO/code/m16_queue/jobs/vllm_lib.sh
trap - EXIT
OUT=$RES/i3g_agentic/rep; mkdir -p $OUT
WF=/mlsteam/data/tiara/stage3/workloads/i3g/i3g_w.jsonl
SHA=$($PY -c "import json;print(json.loads(open('$WF').readline())['_meta']['sha256'])") || { nh "$JOBNAME：讀不到工作負載 $WF"; exit 1; }
[ "$SHA" = 59f397fe741adb44c018f3fee9cca1e928a9ba53cf1d5245ed3ec07d45e29100 ] || { nh "$JOBNAME：工作負載的 sha256 不是事先登記的那份（$SHA）"; exit 1; }
lane() {   # lane <卡> <port> <rep 標籤> <設定…>
  local g=$1 p=$2 r=$3; shift 3
  export TIARA_GPU=$g PORT=$p URL=http://127.0.0.1:$p
  if [ -e /mlsteam/data/tiara/USE_ROCR ]; then export ROCR_VISIBLE_DEVICES=$g; unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
  else unset ROCR_VISIBLE_DEVICES; export HIP_VISIBLE_DEVICES=$g CUDA_VISIBLE_DEVICES=$g; fi
  trap srv_stop EXIT
  local FS=/var/tmp/m16_i3g_fs_g$g
  for k in "$@"; do
    case $k in
      b) OFF=cpu:33; F="" ;;
      c) OFF=tier:33:$FS:150; F=$FS ;;
    esac
    if [ -n "$F" ]; then rm -rf "$F"; mkdir -p "$F"; fi
    srv_start i3g-${k}-$r --apc on --offload $OFF --gpu-blocks 6020 --block-size 16 --attention-backend TRITON_ATTN --max-model-len 26624 || return 1
    bench_run i3g-${k}-$r run --workload $WF --out-dir $OUT --label i3g_${k}_$r --timeout-s 2400 || { srv_stop; return 1; }
    srv_stop
    if [ -n "$F" ]; then
      echo "$k $r $(du -sb "$F" | cut -f1) bytes $(find "$F" -type f | wc -l) files $(date -u +%FT%TZ) $F" > $OUT/fs_bytes_${k}_$r.txt
      rm -rf "$F"
    fi
  done
}
( lane 0 8016 r2 b c ) > $OUT/lane_g0.log 2>&1 &
P0=$!
sleep 20
( lane 1 8017 r3 c b ) > $OUT/lane_g1.log 2>&1 &
P1=$!
wait $P0; R0=$?
wait $P1; R1=$?
rm -rf /var/tmp/m16_i3g_fs_g0 /var/tmp/m16_i3g_fs_g1
echo "lane0 rc=$R0 lane1 rc=$R1"
[ $R0 -eq 0 ] && [ $R1 -eq 0 ] || { nh "$JOBNAME：lane rc $R0／$R1（$OUT/lane_g*.log）"; exit 1; }
echo "$JOBNAME 結束 $(date -u +%FT%TZ)"
