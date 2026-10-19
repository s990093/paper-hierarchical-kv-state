# 只報告的部分（不進判準）：T1（碰天花板）與 LongBench（外部效度），同樣的題目、同樣的配置，只是排在決定性的部分之後。
# 19:2xZ 拆開：INT4 的 T1／LB 要用 80 那次 grid 的會合點（E1@int4 等鍵），移到 88；這裡只留 FP8 的，加上 FP8 混合配置的 T1（封存）。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
[ -n "$(git -C $REPO log --oneline -1 -- docs/precision_20261013/07_freeze_mi300x.md)" ] || { echo "55：事先登記還沒凍結 → 跳過" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }
mkdir -p $OUT/quality_sealed
python3 - <<PY
import json
def fmt(m,t): return json.load(open("$OUT/fmt_select/fmt_choice.json"))["choice"][m][t]
LB="LB-narrativeqa LB-qasper LB-multifieldqa_en LB-hotpotqa LB-2wikimqa LB-musique"
jobs=[]
for m,need in (("llama",42),("qwen",28)):
    G=fmt(m,"fp8"); g=G[4:]
    q=f"python $REPO/code/m10_quality.py run --model {m} --params-file $P --fmt {G} --bws b32 --meets $OUT/speed_{m}/meets_fp8tier.json"
    s1=f"--configs E0 E1 E1s E1sr E1s@fp8k0_{g} E1s@int8_token"
    for s in range(2):
        jobs.append(dict(name=f"s1-{m}-T1-s{s}", need_gb=need, cmd=f"m7run m10-s1-{m}-T1-s{s} {q} {s1} --modes A B --tasks T1 --n 180 --shard {s}/2 --out $OUT/quality/step1_{m}_T1_s{s}.csv"))
        jobs.append(dict(name=f"s1-{m}-LB-s{s}", need_gb=need, cmd=f"m7run m10-s1-{m}-LB-s{s} {q} {s1} --modes A --tasks {LB} --lb-n 50 --shard {s}/2 --out $OUT/quality/step1_{m}_LB_s{s}.csv"))
    jobs.append(dict(name=f"f8m-{m}-T1", need_gb=need, cmd=f"m7run m10-f8m-{m}-T1 {q} --configs F25 M25 B25 R25 --modes A B --tasks T1 --n 180 --out $OUT/quality_sealed/fp8mix_{m}_T1.csv"))
json.dump(jobs,open("jobs_55.json","w"),indent=1)
PY
pool jobs_55.json 8 || { echo "55：pool jobs_55.json 有 job 失敗（$JOBS/pool_*.log）→ 跳過這段的其餘部分" >> $JOBS/NEEDS_HUMAN; exit 0; }
