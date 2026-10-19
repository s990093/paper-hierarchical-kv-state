# vLLM 實驗（V0／F1／F2／I3g）共用：一個 server 的啟動與關閉。由 job source（在 lib.sh 之後）。
# 用法：srv_start <短名> <serve 參數…>  → 設 SRD（server 的 run 目錄）、SPID、URL；結束時 trap 會關掉 server。
#       bench_run <短名> <run 參數…>   → 每格一個 m7run（自己的 RUN_ID），rc＝4（污染）重跑一次，還是 4 就 NEEDS_HUMAN。
PY=/mlsteam/workspace/venv/tiara-v028/bin/python
B="$PY $REPO/code/m16_vllm_bench.py"
W="$PY $REPO/code/m16_workload.py"
PORT=${PORT:-8016}
URL=http://127.0.0.1:$PORT
SPID=""
srv_stop() {
  [ -n "$SPID" ] || return 0
  kill -TERM -- -$SPID 2>/dev/null || true
  for i in $(seq 60); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
  kill -KILL -- -$SPID 2>/dev/null || true
  [ -f "$SRD/serve_cleanup.sh" ] && bash "$SRD/serve_cleanup.sh" || true
  SPID=""
  sleep 5                                    # 等 kfd 關、amd-smi 不再列（gpu_guard 的殘影）
}
trap srv_stop EXIT
srv_start() {
  local short=$1; shift
  srv_stop
  SRID="$(date -u +%Y%m%d-%H%M%S)-m16-srv-$short"; SRD=$TIARA_RUNS/$SRID; mkdir -p $SRD
  $B serve --run-dir $SRD --model ${MODEL:-llama} --port $PORT --label $short "$@" > $SRD/serve_build.log 2>&1 \
    || { nh "$JOBNAME: serve 組指令失敗（$SRD）"; exit 1; }
  setsid bash $SRD/serve_cmd.sh > /dev/null 2>&1 < /dev/null &
  SPID=$!
  $B wait-ready --url $URL --timeout 900 --pid $SPID >> $SRD/serve_build.log 2>&1 \
    || { nh "$JOBNAME: server 沒有起來（$SRD/vllm_server.log）"; exit 1; }
  grep -E "Using .* backend|Overriding with|M16_ROCMFIX|GPU KV cache size|num_gpu_blocks|Maximum concurrency" \
    "$SRD/vllm_server.log" > "$SRD/backend.txt" 2>/dev/null || true
  echo "server $short 起來了：$SRD（pid $SPID）"; cat $SRD/backend.txt
}
bench_run() {
  local short=$1; shift
  local rc
  for try in 1 2; do
    m7run m16-$short $B "$@" --server-config $SRD/serve_config.json --guard-own-root $$; rc=$?
    [ $rc -eq 4 ] || break
    echo "$short：gpu_guard 污染（rc=4），重跑（第 $try 次）"
  done
  if [ $rc -ne 0 ]; then nh "$JOBNAME: $short rc=$rc（$(lastrun m16-$short)）"; return $rc; fi
}
