#!/bin/bash
# run_diff.sh <trace> <matrix> [--no-ref]   （§4.1 的殼；單一行程、nice 19）
B=/ssd6/hungwei/icpe-next/ana-bench
tr=$1; mx=$2; extra=$3; tag=${TAG:-}
RUN_ID=$(date +%Y%m%d-%H%M%S)-icpe-ana-diff$tag-$mx-$(echo $tr | tr ':' '_'); RUN=/ssd6/hungwei/icpe-next/runs/$RUN_ID; mkdir -p "$RUN"
cp $B/scripts/difftest.py $B/rust/x1fast/src/main.rs "$RUN"/
cat > "$RUN"/cmd.sh <<EOC
source $B/scripts/benv.sh
export RUN_ID=$RUN_ID X1_TRACE_DIR=/ssd8/hungwei/paper-hkv/datasets/mooncake/traces
cd $B/scripts
nice -n 19 /ssd6/hungwei/icpe-next/venv/bin/python -I $B/scripts/difftest.py $B/src/code $B/scripts/K_V1.json $tr $mx $RUN/diff.jsonl ${BIN:-$B/rust/x1fast/target/release/x1fast} $B/work $extra
EOC
{ date -Is; echo host=acvlab; cat $B/src/SOURCE.txt; sha256sum ${BIN:-$B/rust/x1fast/target/release/x1fast}; uptime; } > "$RUN"/context.txt
bash "$RUN"/cmd.sh > "$RUN"/stdout.log 2> "$RUN"/stderr.log; echo $? > "$RUN"/exit_code
echo "$RUN exit=$(cat "$RUN"/exit_code)"; tail -3 "$RUN"/stdout.log; tail -3 "$RUN"/stderr.log
