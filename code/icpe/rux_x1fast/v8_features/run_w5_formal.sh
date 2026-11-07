#!/usr/bin/env bash
# PREREG 偏離 11 A 的 W5 敏感度：一個代表變體 × CPU 111.0625／128／256 × CPU 17 × GPU 18（版本 C 的規則），只跑 W5（INT4）那一半；
# BF16 那一半用偏離 10 正式 APC 重算的同一個 sweep（同一版引擎）。
# 用法（rux_run.sh 的 run 目錄裡）：bash run_w5_formal.sh <--traces> <--rules（單一個）> <變體目錄名> <偏離 10 的 x1_sweep.csv(.xz)> [workers]
# 例：bash run_w5_formal.sh toolagent R1 mooncake_toolagent_R1 $REPO/results/icpe_next/apc/mooncake_toolagent/x1_sweep.csv.xz 16
# 步驟：1. sweep（RUX_W5=1）；2. BF16 sweep 篩到同一個 rule、同 3 個容量；3. explore_w5.py（allowed＝每組事後挑較快格式＝登記的算法）；
#       4. w5_mechanism.py（allowed 比 base（＝版本 C 同格）升高 >5 個百分點的格）；5. collect → results/icpe_next/w5/<變體>/。逐請求 npz 不存。
set -euo pipefail
TR=$1; RU=$2; VAR=$3; BASE=$4; W=${5:-16}
: "${RUN:?要在 rux_run.sh 的 run 目錄裡跑}"
V8=$REPO/code/icpe/rux_x1fast/v8_features
source $REPO/code/icpe/rux_x1fast/v6_slowcap/env_sim.sh
cd $REPO
P17="LRU LRU2 ARC S3FIFO GDSF PENS FREQ2Q LRU+NONE LRU+2HW LRU+2HA LRU+RPF O-BEL O-BELS O-DROP O-NOW O-PF O-ALL"
G18="LRU BEL TTLA:0.3 TTLA:1 TTLA:3 TTLA:10 TTLA:30 TTLA:100 TTLA:300 TTLB:0.3 TTLB:1 TTLB:3 TTLB:10 TTLB:30 TTLB:100 TTLB:300 APC BELP"
mkdir -p $RUN/a
echo "== W5 sweep $(date -Is) load=$(cut -d' ' -f1-3 /proc/loadavg)"
RUX_W5=1 RUN_DIR="" /usr/bin/time -v nice -n 5 $ROOT/venv/bin/python -I $V8/sweep_v8.py --variant V1 --calib $CALIB --out-root $RUN/out x1 sweep \
  --traces $TR --rules $RU --cpu 111.0625 128 256 --policies $P17 --slows ssd b32 nfs --modes cake --gpu-gib 111.0625 --gpu-write wb \
  --gpu-policies $G18 --pf-target cpu --workers $W --engine rust > $RUN/sweep.log 2> $RUN/sweep.err
grep -E '^(engine=|jobs=|sweep done)' $RUN/sweep.log
grep -q "py_fallback_jobs=0" $RUN/sweep.log || { echo "有 job 退回 Python → 停"; exit 1; }
$ROOT/venv/bin/python -I - "$BASE" "$RU" $RUN/base_bf16.csv <<'PY'
import sys, pandas as pd
b = pd.read_csv(sys.argv[1]); b = b[(b.rule == sys.argv[2]) & b.cpu_GiB.astype(float).isin([111.0625, 128.0, 256.0])]
assert len(b), "BF16 sweep 裡沒有這些格"
b.to_csv(sys.argv[3], index=False); print("BF16 列", len(b))
PY
( cd $RUN && $ROOT/venv/bin/python -I $V8/explore_w5.py $RUN/base_bf16.csv $RUN/out/x1/x1_sweep.csv $RUN/a ) | tee $RUN/explore_w5.log
( cd $RUN && $ROOT/venv/bin/python -I $V8/w5_mechanism.py $RUN/a 5 allowed ) | tee $RUN/w5_mechanism.log
$ROOT/venv/bin/python -I - $RUN <<'PY'
import sys, os, pandas as pd
r = sys.argv[1]; x = pd.read_csv(os.path.join(r, "w5_mechanism_rows.csv")); c = open(os.path.join(r, "context.txt")).read().splitlines()
x.insert(0, "host", c[1].strip()); x.insert(0, "ts", c[0].strip()); x.insert(0, "run_id", os.path.basename(r)); x.to_csv(os.path.join(r, "a", "w5_mechanism_rows.csv"), index=False)
PY
$ROOT/venv/bin/python -I $V8/collect_v8.py w5 $VAR --base results/icpe_next/w5 --sweep $RUN/base_bf16.csv=x1_sweep_bf16 $RUN/out/x1/x1_sweep.csv=x1_sweep_w5 \
  --files $RUN/a/explore_w5_rows.csv $RUN/a/base_headroom_components.csv $RUN/a/allowed_headroom_components.csv $RUN/a/forced_headroom_components.csv $RUN/a/w5_mechanism_rows.csv \
  --runs $RUN --note "PREREG 偏離 11 A 的 W5 敏感度；登記的算法＝allowed（每組事後挑 BF16／W5 較快）；BF16＝偏離 10 的正式 sweep：$BASE"
echo "W5_FORMAL_DONE $VAR"
