# 每個 stage 開頭 source 這支
REPO=/mlsteam/workspace/paper-hierarchical-kv-state
JOBS=/tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs
cd $JOBS
source $REPO/code/m7_env.sh
OUT=$REPO/results/m10_precision_mi300x
P=$REPO/results/m10_pilot_mi300x/e0_tune/CANDIDATE_PARAMS.json
lastrun() { ls -d $TIARA_RUNS/*-"$1" | sort | tail -1 | xargs basename; }
fmt_of() { python3 -c "import json;print(json.load(open('$OUT/fmt_select/fmt_choice.json'))['choice']['$1']['$2'])"; }
pool() { python3 $JOBS/pool.py "$@"; local rc=$?; [ $rc -eq 75 ] && exit 75; return $rc; }      # pool <jobs.json> <最多幾個同時>；75＝helper 讓卡（10/19 雙卡），整個 stage 交回 runner 重排
# 2026-10-15 19:3xZ 起（stage 21 之後的 stage）：m7_busy 收尾時暫停 gpu_guard 取樣，避開 amd-smi 殘留條目的誤報（m10_speed.Busy）
export M10_BUSY_TEARDOWN_PAUSE=1
