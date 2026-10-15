# 大掃描（09）試跑：零件（記憶體、載入、反量化；GPU 閒／忙）→ 融合反量化微測 → 海選試跑（每個模型 2 個 process、每任務 2 題）
# → sweep_plan_n.py 照 09 §4.1 的規則決定海選題數，寫出 41_sweep_screen.sh。探索實驗，不碰主線判定。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
SW=$REPO/results/m10_sweep_mi300x; RAW=/mlsteam/data/tiara/sweep_m10
mkdir -p $SW/parts $RAW
for m in llama qwen; do
  m7run m10-sweep-parts-$m python $REPO/code/m10_sweep.py parts --model $m --out-dir $SW/parts || exit 1
done
m7run m10-dqfast-bench python $REPO/code/m10_dequant_fast.py --self-test --bench --out $SW/parts/dequant_fast_bench.json || exit 1
python3 - <<PY
import json
G={"llama":"/mlsteam/data/tiara/runs/20261015-161226-m10-grid-fp8-llama/blocks.csv",
   "qwen":"/mlsteam/data/tiara/runs/20261015-194150-m10-regrid-fp8-qwen/blocks.csv"}
jobs=[]
for m,need in (("llama",40),("qwen",30)):
    for s in range(2):
        jobs.append(dict(name=f"swp-{m}-s{s}", need_gb=need,
            cmd=f"m7run m10-sweep-pilot-{m}-s{s} python $REPO/code/m10_sweep.py screen --model {m} --params-file $P --n 2 --shard {s}/2 "
                f"--parts $SW/parts/parts_{m}.json --grid-blocks {G[m]} --out $RAW/pilot_{m}_s{s}.csv.gz"))
json.dump(jobs,open("jobs_40.json","w"),indent=1)
PY
date -u +%s > $RAW/pilot_start
pool jobs_40.json 4 || exit 1
date -u +%s > $RAW/pilot_end
python3 $JOBS/sweep_plan_n.py || exit 1
