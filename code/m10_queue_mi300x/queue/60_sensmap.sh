# 03 §3.3b 逐 block 敏感度地圖（只報告）：兩個模型 × 兩階選定的格式，T1–T5 各 10 題（seed 10000 起）、第 1 輪
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
[ -n "$(git -C $REPO log --oneline -1 -- docs/precision_20261013/07_freeze_mi300x.md)" ] || { echo "事先登記還沒凍結"; exit 1; }
mkdir -p $OUT/quality_sealed
python3 - <<PY
import json
def fmt(m,t): return json.load(open("$OUT/fmt_select/fmt_choice.json"))["choice"][m][t]
jobs=[]
for m,need in (("llama",32),("qwen",24)):
    for tier in ("fp8","int4"):
        F=fmt(m,tier)
        for s in range(2):
            jobs.append(dict(name=f"sens-{m}-{F}-s{s}", need_gb=need, cmd=f"m7run m10-sens-{m}-{tier}-s{s} python $REPO/code/m10_quality.py sensmap --model {m} --fmt {F} --params-file $P --n 10 --shard {s}/2 --out $OUT/quality_sealed/sensmap_{m}_{F}_s{s}.csv"))
json.dump(jobs,open("jobs_60.json","w"),indent=1)
PY
pool jobs_60.json 8 || exit 1
