# 每個 stage 開頭 source 這支
REPO=/mlsteam/workspace/paper-hierarchical-kv-state
JOBS=/tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs
cd $JOBS
source $REPO/code/m7_env.sh
OUT=$REPO/results/m10_precision_mi300x
P=$REPO/results/m10_pilot_mi300x/e0_tune/CANDIDATE_PARAMS.json
lastrun() { ls -d $TIARA_RUNS/*-"$1" | sort | tail -1 | xargs basename; }
fmt_of() { python3 -c "import json;print(json.load(open('$OUT/fmt_select/fmt_choice.json'))['choice']['$1']['$2'])"; }
pool() { python3 $JOBS/pool.py "$@"; }      # pool <jobs.json> <最多幾個同時>
