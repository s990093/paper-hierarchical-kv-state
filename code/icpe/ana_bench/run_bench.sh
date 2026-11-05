#!/bin/bash
# 用法：run_bench.sh <label> <python> <code_dir> [reps]   （§4.1 的殼；單一行程、nice 19）
B=/ssd6/hungwei/icpe-next/ana-bench
label=$1; py=$2; code=$3; reps=${4:-3}
RUN_ID=$(date +%Y%m%d-%H%M%S)-icpe-ana-bench${BTAG:-}-$label; RUN=/ssd6/hungwei/icpe-next/runs/$RUN_ID; mkdir -p $RUN
cat > $RUN/cmd.sh <<EOC
source $B/scripts/benv.sh
export RUN_ID=$RUN_ID
cd $B/scripts
nice -n 19 $py -I $B/scripts/bench.py $label $code $B/scripts/K_V1.json ${TRACE:-nb:semianalysis_N16_s1} $reps $RUN/bench.jsonl
EOC
{ date -Is; echo host=acvlab; echo "src=$(cat $B/src/SOURCE.txt)"; $py -VV; uptime; free -g; } > $RUN/context.txt 2>&1
cp $B/scripts/bench.py $RUN/
bash $RUN/cmd.sh > $RUN/stdout.log 2> $RUN/stderr.log; echo $? > $RUN/exit_code
echo "$RUN exit=$(cat $RUN/exit_code)"
