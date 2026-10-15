# 只報告的部分（不進判準）：T1（碰天花板）與 LongBench（外部效度），同樣的題目、同樣的配置，只是排在決定性的部分之後
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
git -C $REPO log --oneline -- docs/precision_20261013/07_freeze_mi300x.md | grep -q . || { echo "事先登記還沒凍結"; exit 1; }
MODELS=${MODELS_51:-"llama qwen"}
python3 - <<PY
import json
def fmt(m,t): return json.load(open("$OUT/fmt_select/fmt_choice.json"))["choice"][m][t]
LB="LB-narrativeqa LB-qasper LB-multifieldqa_en LB-hotpotqa LB-2wikimqa LB-musique"
jobs=[]
for m,need in (("llama",42),("qwen",28)):
    G=fmt(m,"fp8"); g=G[4:]; I=fmt(m,"int4")
    q=f"python $REPO/code/m10_quality.py run --model {m} --params-file $P --fmt {G} --bws b32 --meets $OUT/speed_{m}/meets.json"
    s1=f"--configs E0 E1 E1s E1sr E1s@fp8k0_{g} E1s@int8_token"
    i4=f"--configs E1@{I} E1s@{I} E1sr@{I}"
    for s in range(2):
        jobs.append(dict(name=f"s1-{m}-T1-s{s}", need_gb=need, cmd=f"m7run m10-s1-{m}-T1-s{s} {q} {s1} --modes A B --tasks T1 --n 180 --shard {s}/2 --out $OUT/quality/step1_{m}_T1_s{s}.csv"))
        jobs.append(dict(name=f"s1-{m}-LB-s{s}", need_gb=need, cmd=f"m7run m10-s1-{m}-LB-s{s} {q} {s1} --modes A --tasks {LB} --lb-n 50 --shard {s}/2 --out $OUT/quality/step1_{m}_LB_s{s}.csv"))
    jobs.append(dict(name=f"i4e-{m}-T1", need_gb=need, cmd=f"m7run m10-i4e-{m}-T1 {q} {i4} --modes A B --tasks T1 --n 180 --out $OUT/quality_sealed/int4end_{m}_T1.csv"))
    jobs.append(dict(name=f"i4e-{m}-LB", need_gb=need, cmd=f"m7run m10-i4e-{m}-LB {q} {i4} --modes A --tasks {LB} --lb-n 50 --out $OUT/quality_sealed/int4end_{m}_LB.csv"))
    if m in "$MODELS".split():
        jobs.append(dict(name=f"i4m-{m}-T1", need_gb=need, cmd=f"m7run m10-i4m-{m}-T1 {q} --configs F25@{I} M25@{I} B25@{I} R25@{I} --modes A B --tasks T1 --n 180 --out $OUT/quality_sealed/int4mix_{m}_T1.csv"))
json.dump(jobs,open("jobs_55.json","w"),indent=1)
PY
pool jobs_55.json 8 || exit 1
