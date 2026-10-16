# 第 2 階（INT4）端點品質：E1、E1s、E1sr @INT4；讀法 A＋B（b32）；T2–T5（T1、LongBench 在 55）。先跑、封存
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
[ -n "$(git -C $REPO log --oneline -1 -- docs/precision_20261013/07_freeze_mi300x.md)" ] || { echo "事先登記還沒凍結"; exit 1; }
mkdir -p $OUT/quality_sealed
python3 - <<PY
import json
def fmt(m,t): return json.load(open("$OUT/fmt_select/fmt_choice.json"))["choice"][m][t]
per={}
for m,need in (("llama",32),("qwen",24)):
    G=fmt(m,"fp8"); I=fmt(m,"int4")
    C=f"E1@{I} E1s@{I} E1sr@{I}"
    base=f"python $REPO/code/m10_quality.py run --model {m} --params-file $P --fmt {G} --configs {C} --modes A B --bws b32 --meets $OUT/speed_{m}/meets.json"
    js=[dict(name=f"i4e-{m}-T24-s{s}", need_gb=need, cmd=f"m7run m10-i4e-{m}-T24-s{s} {base} --tasks T2 T3 T4 --n 180 --shard {s}/5 --out $OUT/quality_sealed/int4end_{m}_T24_s{s}.csv") for s in range(5)]
    js.append(dict(name=f"i4e-{m}-T5", need_gb=need, cmd=f"m7run m10-i4e-{m}-T5 {base} --tasks T5 --n 60 --out $OUT/quality_sealed/int4end_{m}_T5.csv"))
    per[m]=js
jobs=[j for a,b in zip(per["llama"],per["qwen"]) for j in (a,b)]
json.dump(jobs,open("jobs_50.json","w"),indent=1)
PY
pool jobs_50.json 8 || exit 1
