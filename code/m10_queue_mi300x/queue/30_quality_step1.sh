# Step 1 品質（決定性的部分先跑）：FP8 端點（E0、E1、E1s、E1sr）＋ Step 2b（先跑、封存）；讀法 A＋B（b32）；
# 主判準的任務 T2–T4 各 180、T5 60。T1（只報告）與 LongBench（只報告）移到 55（同樣的題目，只是順序往後）
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
git -C $REPO log --oneline -- docs/precision_20261013/07_freeze_mi300x.md | grep -q . || { echo "事先登記還沒凍結（07_freeze_mi300x.md 沒 commit）"; exit 1; }
mkdir -p $OUT/quality
python3 - <<PY
import json
def fmt(m,t): return json.load(open("$OUT/fmt_select/fmt_choice.json"))["choice"][m][t]
per={}
for m,need in (("llama",42),("qwen",28)):
    G=fmt(m,"fp8"); g=G[4:]
    C=f"E0 E1 E1s E1sr E1s@fp8k0_{g} E1s@int8_token"
    base=f"python $REPO/code/m10_quality.py run --model {m} --params-file $P --fmt {G} --configs {C} --modes A B --bws b32 --meets $OUT/speed_{m}/meets.json"
    js=[dict(name=f"s1-{m}-T24-s{s}", need_gb=need, cmd=f"m7run m10-s1-{m}-T24-s{s} {base} --tasks T2 T3 T4 --n 180 --shard {s}/6 --out $OUT/quality/step1_{m}_T24_s{s}.csv") for s in range(6)]
    js.append(dict(name=f"s1-{m}-T5", need_gb=need, cmd=f"m7run m10-s1-{m}-T5 {base} --tasks T5 --n 60 --out $OUT/quality/step1_{m}_T5.csv"))
    per[m]=js
jobs=[j for a,b in zip(per["llama"],per["qwen"]) for j in (a,b)]
json.dump(jobs,open("jobs_30.json","w"),indent=1)
PY
pool jobs_30.json 8 || exit 1
