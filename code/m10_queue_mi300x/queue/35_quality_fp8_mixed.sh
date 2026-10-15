# FP8 階的混合配置 F25、M25、B25、R25（Step 2：穩定差距、K2）；讀法 A＋B（b32）；T2–T5。先跑、封存（07 §6）。
# 為什麼提前（19:2xZ 排）：Qwen 的 E1／E1s／E1sr 在 CPU 層 GPU 忙那格速度不輸沒過（analysis/fp8/speed_qwen.json），
# 照凍結的判定（m10_analyze model_verdict：ε 過「而且」速度不輸）Qwen 的 K1／K1s 不會成立 → 走到 Step 2 的機率高。
# 哪個模型進 Step 2 要等 Step 1／2b 的判定；Qwen 的工作排前面。判定後不需要的模型：把 "f8m-<model>" 寫進 pool_skip.txt。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
[ -n "$(git -C $REPO log --oneline -1 -- docs/precision_20261013/07_freeze_mi300x.md)" ] || { echo "事先登記還沒凍結"; exit 1; }
mkdir -p $OUT/quality_sealed
python3 - <<PY
import json
def fmt(m,t): return json.load(open("$OUT/fmt_select/fmt_choice.json"))["choice"][m][t]
jobs=[]
for m,need in (("qwen",24),("llama",32)):
    G=fmt(m,"fp8")
    base=f"python $REPO/code/m10_quality.py run --model {m} --params-file $P --fmt {G} --configs F25 M25 B25 R25 --modes A B --bws b32 --meets $OUT/speed_{m}/meets.json"
    jobs+=[dict(name=f"f8m-{m}-T24-s{s}", need_gb=need, cmd=f"m7run m10-f8m-{m}-T24-s{s} {base} --tasks T2 T3 T4 --n 180 --shard {s}/6 --out $OUT/quality_sealed/fp8mix_{m}_T24_s{s}.csv") for s in range(6)]
    jobs.append(dict(name=f"f8m-{m}-T5", need_gb=need, cmd=f"m7run m10-f8m-{m}-T5 {base} --tasks T5 --n 60 --out $OUT/quality_sealed/fp8mix_{m}_T5.csv"))
json.dump(jobs,open("jobs_35.json","w"),indent=1)
PY
pool jobs_35.json 8 || exit 1
