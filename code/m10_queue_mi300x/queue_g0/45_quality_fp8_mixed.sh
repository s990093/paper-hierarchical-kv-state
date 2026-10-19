# FP8 階的混合配置 F25、M25、B25、R25（Step 2：穩定差距、K2）；讀法 A＋B（b32）；T2–T5。先跑、封存（07 §6）。
# 19:2xZ 提前的理由（依第一次 Qwen 速度量測：E1／E1s／E1sr 在 CPU・忙沒過 → Qwen 的 K1／K1s 不會成立）在重量後只剩一半：
# 重量（stage 22）E1s 過了，INT4 那條路又走得到。所以 stage 31 判完 Step 1 會調順序：兩個模型都過 → INT4（33a–c）排在這段前面；
# Step 2 要在某個模型上做 → f8m_first.txt 寫那個模型，這段先跑它（預設 Qwen 先）。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
[ -n "$(git -C $REPO log --oneline -1 -- docs/precision_20261013/07_freeze_mi300x.md)" ] || { echo "45：事先登記還沒凍結 → 跳過" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }
mkdir -p $OUT/quality_sealed
python3 - <<PY
import json
def fmt(m,t): return json.load(open("$OUT/fmt_select/fmt_choice.json"))["choice"][m][t]
jobs=[]
import os
first=open("$JOBS/f8m_first.txt").read().strip() if os.path.exists("$JOBS/f8m_first.txt") else "qwen"
order=sorted((("qwen",24),("llama",32)), key=lambda x: x[0]!=first)
print("[35] 先跑", order[0][0], flush=True)
for m,need in order:
    G=fmt(m,"fp8")
    base=f"python $REPO/code/m10_quality.py run --model {m} --params-file $P --fmt {G} --configs F25 M25 B25 R25 --modes A B --bws b32 --meets $OUT/speed_{m}/meets_fp8tier.json"
    jobs+=[dict(name=f"f8m-{m}-T24-s{s}", need_gb=need, cmd=f"m7run m10-f8m-{m}-T24-s{s} {base} --tasks T2 T3 T4 --n 180 --shard {s}/6 --out $OUT/quality_sealed/fp8mix_{m}_T24_s{s}.csv") for s in range(6)]
    jobs.append(dict(name=f"f8m-{m}-T5", need_gb=need, cmd=f"m7run m10-f8m-{m}-T5 {base} --tasks T5 --n 60 --out $OUT/quality_sealed/fp8mix_{m}_T5.csv"))
json.dump(jobs,open("jobs_35.json","w"),indent=1)
PY
pool jobs_35.json 8 || { echo "45：pool jobs_35.json 有 job 失敗（$JOBS/pool_*.log）→ 跳過這段的其餘部分" >> $JOBS/NEEDS_HUMAN; exit 0; }
