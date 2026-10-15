# 第 2 階（INT4）混合配置 F25、M25、B25、R25（Step 2）；讀法 A＋B（b32）；T2–T5（T1 在 55）。先跑、封存；主模型確定後可只留一個模型（MODELS_51）
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
[ -n "$(git -C $REPO log --oneline -1 -- docs/precision_20261013/07_freeze_mi300x.md)" ] || { echo "事先登記還沒凍結"; exit 1; }
mkdir -p $OUT/quality_sealed
MODELS=${MODELS_51:-"llama qwen"}
python3 - <<PY
import json
def fmt(m,t): return json.load(open("$OUT/fmt_select/fmt_choice.json"))["choice"][m][t]
per={}
for m,need in (("llama",32),("qwen",24)):
    if m not in "$MODELS".split(): continue
    G=fmt(m,"fp8"); I=fmt(m,"int4")
    C=f"F25@{I} M25@{I} B25@{I} R25@{I}"
    base=f"python $REPO/code/m10_quality.py run --model {m} --params-file $P --fmt {G} --configs {C} --modes A B --bws b32 --meets $OUT/speed_{m}/meets.json"
    js=[dict(name=f"i4m-{m}-T24-s{s}", need_gb=need, cmd=f"m7run m10-i4m-{m}-T24-s{s} {base} --tasks T2 T3 T4 --n 180 --shard {s}/6 --out $OUT/quality_sealed/int4mix_{m}_T24_s{s}.csv") for s in range(6)]
    js.append(dict(name=f"i4m-{m}-T5", need_gb=need, cmd=f"m7run m10-i4m-{m}-T5 {base} --tasks T5 --n 60 --out $OUT/quality_sealed/int4mix_{m}_T5.csv"))
    per[m]=js
lists=list(per.values())
jobs=[j for grp in zip(*lists) for j in grp] if len(lists)>1 else lists[0]
json.dump(jobs,open("jobs_51.json","w"),indent=1)
PY
pool jobs_51.json 8 || exit 1
