# 從 55 拆出來（19:2xZ）：INT4 端點與混合配置的 T1／LongBench（只報告、封存）。要用 80 那次 grid 寫進 meets.json 的 @int4 會合點，所以排在 80 之後。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
[ -n "$(git -C $REPO log --oneline -1 -- docs/precision_20261013/07_freeze_mi300x.md)" ] || { echo "88：事先登記還沒凍結 → 跳過" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }
MODELS=${MODELS_51:-"llama qwen"}
mkdir -p $OUT/quality_sealed
python3 - <<PY
import json
def fmt(m,t): return json.load(open("$OUT/fmt_select/fmt_choice.json"))["choice"][m][t]
LB="LB-narrativeqa LB-qasper LB-multifieldqa_en LB-hotpotqa LB-2wikimqa LB-musique"
jobs=[]
for m,need in (("llama",42),("qwen",28)):
    G=fmt(m,"fp8"); I=fmt(m,"int4")
    q=f"python $REPO/code/m10_quality.py run --model {m} --params-file $P --fmt {G} --bws b32 --meets $OUT/speed_{m}/meets.json"
    i4=f"--configs E1@{I} E1s@{I} E1sr@{I}"
    jobs.append(dict(name=f"i4e-{m}-T1", need_gb=need, cmd=f"m7run m10-i4e-{m}-T1 {q} {i4} --modes A B --tasks T1 --n 180 --out $OUT/quality_sealed/int4end_{m}_T1.csv"))
    jobs.append(dict(name=f"i4e-{m}-LB", need_gb=need, cmd=f"m7run m10-i4e-{m}-LB {q} {i4} --modes A --tasks {LB} --lb-n 50 --out $OUT/quality_sealed/int4end_{m}_LB.csv"))
    if m in "$MODELS".split():
        jobs.append(dict(name=f"i4m-{m}-T1", need_gb=need, cmd=f"m7run m10-i4m-{m}-T1 {q} --configs F25@{I} M25@{I} B25@{I} R25@{I} --modes A B --tasks T1 --n 180 --out $OUT/quality_sealed/int4mix_{m}_T1.csv"))
json.dump(jobs,open("jobs_88.json","w"),indent=1)
PY
pool jobs_88.json 8 || { echo "88：pool jobs_88.json 有 job 失敗（$JOBS/pool_*.log）→ 跳過這段的其餘部分" >> $JOBS/NEEDS_HUMAN; exit 0; }
