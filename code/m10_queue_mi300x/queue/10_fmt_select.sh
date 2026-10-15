# 20 題小測：挑 FP8 粒度與 INT4 分組（只看 KL，03 開頭 10/15 晚改動）
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
mkdir -p $OUT/fmt_select
CF="E0 E1@fp8_head E1@fp8_channel E1@fp8_token E1@int4_token E1@int4_kivi"
python3 - <<PY
import json
jobs=[]
for m,need in (("llama",48),("qwen",30)):
    for s in (0,1):
        jobs.append(dict(name=f"fmtsel-{m}-s{s}", need_gb=need,
          cmd=f"m7run m10-fmtsel-{m}-s{s} python $REPO/code/m10_quality.py run --model {m} --params-file $P --seed0 4000 --n 5 --tasks T2 T3 T4 T5 --modes A --configs $CF --shard {s}/2 --out $OUT/fmt_select/mini_{m}_s{s}.csv"))
json.dump(jobs,open("jobs_10.json","w"),indent=1)
PY
pool jobs_10.json 4 || exit 1
python3 $REPO/code/m10_fmt_select.py --csv $OUT/fmt_select/mini_*_s*.csv --out $OUT/fmt_select/fmt_choice.json || exit 1
