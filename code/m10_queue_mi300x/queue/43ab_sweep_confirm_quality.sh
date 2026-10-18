# 10/17 13:3xZ 接續 43aa：速度格子 7/7 完成（21 個 guard 全 CLEAN；跑的那份寫成 16，筆誤）；品質第一個 job 因 jobs_43q.json 裡寫成字面 $P（lib.sh 的 P 沒 export，pool 的 bash -c 看不到）→ --params-file 空 → rc=2。
# 修正：m10_confirm.py plan 改寫實際路徑；jobs_43q.json 已換成實際路徑（和沙盒重新 plan 出來的逐字相同；舊檔 jobs_43q.BAD_literalP.json）。這裡只跑 43aa 的第 2、3 步，不重跑速度。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
C=$REPO/results/m10_sweep_mi300x/confirm
# 0. 品質冒煙：每個模型拿第一個正式 job 的指令，只跑 T2 × 1 題（輸出放 scratch，不進 results）
SM=$JOBS/qsmoke43; mkdir -p $SM
python3 - $JOBS/jobs_43q.json $SM <<'PY'
import json, re, sys
jobs, sm = json.load(open(sys.argv[1])), sys.argv[2]
out, seen = [], set()
for j in jobs:
    m = j["name"].split("-")[1]
    if m in seen or "T24" not in j["name"]: continue
    seen.add(m)
    c = j["cmd"].replace(f"m10-confirm-{m}-T24-s0", f"m10-confirm-qsmoke-{m}")
    c = re.sub(r"--tasks T2 T3 T4 --n 90 --shard \S+ --out \S+", f"--tasks T2 --n 1 --out {sm}/qsmoke_{m}.csv", c)
    assert "--n 1 " in c and "$" not in c, c
    out.append(dict(name=f"cfq-smoke-{m}", need_gb=j["need_gb"], cmd=c))
json.dump(out, open(f"{sm}/jobs_43qsmoke.json", "w"), indent=1)
print(len(out), "smoke jobs")
PY
pool $SM/jobs_43qsmoke.json 2 || { echo "43ab 品質冒煙失敗（$JOBS/pool_cfq-smoke-*.log）→ 跳過 43 品質" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }
# 2. 品質（新 seed 50000 起；T2–T4 各 90、T5 30；讀法 A＋B（b32 閒／忙，實測會合點））
pool $JOBS/jobs_43q.json 8 || { echo "43ab 品質 pool 有 job 失敗（$JOBS/pool_cf-*.log）" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }
# 3. 判定（CPU）
python $REPO/code/m10_confirm_analyze.py --dir $C > $C/analyze.out 2>&1 || echo "43 confirm analyze 失敗（$C/analyze.out）" >> $JOBS/NEEDS_HUMAN
exit 0
