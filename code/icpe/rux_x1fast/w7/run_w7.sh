#!/usr/bin/env bash
# W7 正式執行（PREREG 偏離 12）：LHS 90 點 × 3 seed＝270 份合成紀錄 × 版本 C 全套規則（CPU 17 × GPU 18）× CPU 9 個水準（WS_ref 的 2^k，k＝−6..2）
# × ssd／b32／nfs × 閒／忙 × Cake，GPU 111.0625 GiB wb、預取到 CPU。
# 用法（rux_run.sh 的 run 目錄裡）：bash run_w7.sh [workers=16] [之前中斷的 run 目錄 ...（已完成的紀錄直接沿用）]
# 只給煙霧測試用的環境變數（正式跑不設）：W7_POINTS（預設 0..89）、W7_SEEDS（預設 1 2 3）、W7_NBDIR（預設 $ROOT/datasets/x1_nb_w7）、
#   W7_BASE（collect 的上層目錄，預設 results/icpe_next/w7；可給絕對路徑）。
# 步驟（任何一步失敗就停）：
#   0. 編 w7gen（cargo --offline），重產 design.csv，sha256 必須等於登記的 DESIGN_SHA。
#   1. 每份紀錄：w7gen gen → w7_pack.py（獨立檢查＋x1_nb.write_nb）→ $ROOT/datasets/x1_nb_w7/w7_pPPP_sS.npz（不進 git）。
#   2. 每份紀錄：sweep_v8（py_fallback_jobs＝0）→ analyze_v8 → per/<名>/（上限表＋sweep；做完寫 DONE）。
#   3. w7_map.py → a/（w7_cells、w7_map、w7_summary、w7_dims）。
#   4. collect（只放彙總；270 份 sweep 不進 git，MANIFEST 記 sha256）。
set -euo pipefail
W=${1:-16}; shift || true
PRIOR=("$@")
: "${RUN:?要在 rux_run.sh 的 run 目錄裡跑}"
DESIGN_SHA=ab39ad31c11972ecb6e236fb1399f64c1051bd4b81482f7e2752c20584720fe8
W7=$REPO/code/icpe/rux_x1fast/w7
V8=$REPO/code/icpe/rux_x1fast/v8_features
source $REPO/code/icpe/rux_x1fast/v6_slowcap/env_sim.sh
export X1_NB_DIR=${W7_NBDIR:-$ROOT/datasets/x1_nb_w7}
PTS=${W7_POINTS:-$(seq -s' ' 0 89)}; SDS=${W7_SEEDS:-1 2 3}
NEXP=$(( $(echo $PTS | wc -w) * $(echo $SDS | wc -w) ))
mkdir -p $X1_NB_DIR $RUN/per $RUN/a
P17="LRU LRU2 ARC S3FIFO GDSF PENS FREQ2Q LRU+NONE LRU+2HW LRU+2HA LRU+RPF O-BEL O-BELS O-DROP O-NOW O-PF O-ALL"
G18="LRU BEL TTLA:0.3 TTLA:1 TTLA:3 TTLA:10 TTLA:30 TTLA:100 TTLA:300 TTLB:0.3 TTLB:1 TTLB:3 TTLB:10 TTLB:30 TTLB:100 TTLB:300 APC BELP"
echo "== 0. build＋design $(date -Is)"
( cd $W7 && $X1_CARGO build --release --offline --locked 2>&1 | tail -2 )
B=$W7/target/release/w7gen
sha256sum $W7/src/main.rs $B | tee $RUN/w7gen_sha.txt
$B design --n 90 --out $RUN/design.csv
GOT=$(sha256sum $RUN/design.csv | cut -d' ' -f1)
[ "$GOT" = "$DESIGN_SHA" ] || { echo "design.csv 的 sha256 $GOT ≠ 登記的 $DESIGN_SHA → 停"; exit 1; }
echo "== 1. 產生＋檢查 $(date -Is)"
gen1() {  # $1=point $2=seed
  local NM=w7_p$(printf %03d $1)_s$2
  if [ -f $X1_NB_DIR/$NM.npz ] && [ -f $X1_NB_DIR/$NM.meta.json ]; then echo "W7_PACK_SKIP $NM"; return 0; fi
  $B gen --design $RUN/design.csv --point $1 --seed $2 --out-dir $RUN/raw/$NM 2>/dev/null
  $ROOT/venv/bin/python -I $W7/w7_pack.py --raw $RUN/raw/$NM --name $NM --out-dir $X1_NB_DIR
}
export -f gen1; export B RUN X1_NB_DIR W7 ROOT REPO
for P in $PTS; do for S in $SDS; do echo "$P $S"; done; done | xargs -P 8 -n 2 bash -c 'gen1 "$0" "$1"' > $RUN/gen.log 2>&1
N_OK=$(grep -cE "^W7_PACK_(OK|SKIP)" $RUN/gen.log)
[ "$N_OK" = "$NEXP" ] || { echo "只有 $N_OK／$NEXP 份紀錄通過檢查 → 停"; tail -20 $RUN/gen.log; exit 1; }
# 同一份紀錄（同點同 seed）重產一定逐位元相同：抽 3 份重產比 npz 的 sha256（不同 run 之間也成立）
RG=$(for P in $PTS; do for S in $SDS; do echo w7_p$(printf %03d $P)_s$S; done; done | awk 'NR==1{a=$0} {l[NR]=$0} END{print a, l[int((NR+1)/2)], l[NR]}')
for NM in $RG; do
  P=$((10#${NM:4:3})); S=${NM: -1}
  $B gen --design $RUN/design.csv --point $P --seed $S --out-dir $RUN/regen/$NM 2>/dev/null
  $ROOT/venv/bin/python -I $W7/w7_pack.py --raw $RUN/regen/$NM --name $NM --out-dir $RUN/regen_nb > /dev/null
  $ROOT/venv/bin/python -I - $X1_NB_DIR/$NM.npz $RUN/regen_nb/$NM.npz <<'PY'
import sys, numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
for k in ("t", "sess", "parent", "kind", "nall", "full_off", "full_key", "user_trigger"):
    assert np.array_equal(a[k], b[k]), k
print("regen same", sys.argv[1].split("/")[-1])
PY
done
rm -rf $RUN/regen $RUN/regen_nb
echo "== 2. sweep＋上限表 $(date -Is)"
i=0
for P in $PTS; do for S in $SDS; do
  NM=w7_p$(printf %03d $P)_s$S; D=$RUN/per/$NM; i=$((i+1))
  for PR in "${PRIOR[@]}"; do
    if [ -f $PR/per/$NM/DONE ] && [ ! -f $D/DONE ]; then mkdir -p $D; cp -a $PR/per/$NM/. $D/; echo "從 $PR 沿用 $NM" >> $D/DONE; fi
  done
  [ -f $D/DONE ] && continue
  rm -rf $D; mkdir -p $D
  CPU=$($ROOT/venv/bin/python -I -c "
import json; w=json.load(open('$X1_NB_DIR/$NM.meta.json'))['WS_ref_GiB']
c=[max(1, round(w * 2.0**k * 16)) / 16 for k in range(-6, 3)]
assert len(set(c)) == 9, c
print(' '.join(repr(x) for x in c))")
  RUN_DIR="" nice -n 5 $ROOT/venv/bin/python -I $V8/sweep_v8.py --variant V1 --calib $CALIB --out-root $D/out x1 sweep \
    --traces nb:$NM --rules nb --cpu $CPU --policies $P17 --slows ssd b32 nfs --modes cake --gpu-gib 111.0625 --gpu-write wb \
    --gpu-policies $G18 --pf-target cpu --workers $W --engine rust > $D/sweep.log 2> $D/sweep.err
  grep -q "py_fallback_jobs=0" $D/sweep.log || { echo "$NM 有 job 退回 Python → 停"; exit 1; }
  ( cd $D && $ROOT/venv/bin/python -I $V8/analyze_v8.py --sweep $D/out/x1/x1_sweep.csv --out-dir $D --out-prefix w7 ) > $D/analyze.log 2>&1
  xz -T4 -9 $D/out/x1/x1_sweep.csv
  echo "$(date -Is) $(grep -m1 'sweep done' $D/sweep.log)" > $D/DONE
  echo "[$i/$NEXP] $NM $(cat $D/DONE)"
done; done
echo "== 3. 地圖 $(date -Is)"
( cd $RUN && $ROOT/venv/bin/python -I $W7/w7_map.py --design $RUN/design.csv --nb-dir $X1_NB_DIR --runs-glob "$RUN/per/*/w7_headroom_components.csv" --out $RUN/a --seeds $SDS ) | tee $RUN/w7_map.log
grep -q "W7_MAP_RESULT" $RUN/w7_map.log
echo "== 4. collect $(date -Is)"
$ROOT/venv/bin/python -I - $RUN $X1_NB_DIR <<'PY'
import sys, os, glob, json, hashlib, pandas as pd
run, nbd = sys.argv[1:3]
rows = []
for f in sorted(glob.glob(os.path.join(nbd, "w7_p*_s*.meta.json"))):
    m = json.load(open(f)); nm = os.path.basename(f)[:-10]
    sw = os.path.join(run, "per", nm, "out", "x1", "x1_sweep.csv.xz")
    rows.append(dict(name=nm, point=m["point"], seed=m["seed"], n_req=m["n_req"], n_sess=m["n_sess"], n_fork_req=m["n_fork_req"],
                     full_chunks_total=m["full_chunks_total"], span_s=m["span_s"], WS_ref_GiB=m["WS_ref_GiB"], WS_max_GiB=m["WS_max_GiB"],
                     **{f"gen_{k}": v for k, v in m["gen_stats"].items()},
                     npz_sha256=hashlib.sha256(open(os.path.join(nbd, nm + ".npz"), "rb").read()).hexdigest(),
                     sweep_xz_sha256=hashlib.sha256(open(sw, "rb").read()).hexdigest() if os.path.exists(sw) else ""))
pd.DataFrame(rows).to_csv(os.path.join(run, "a", "w7_traces.csv"), index=False)
print("traces", len(rows))
PY
cd $REPO
CS=$($ROOT/venv/bin/python -I - $RUN <<'PY'
import sys, glob, lzma, csv
s = set()
for f in glob.glob(sys.argv[1] + "/per/*/out/x1/x1_sweep.csv.xz"):
    with lzma.open(f, "rt") as fh:
        for r in csv.DictReader(fh):
            s.add(r["code_sha"])
print(" ".join(sorted(s)))
PY
)
echo "sweep code_sha：$CS"
$ROOT/venv/bin/python -I $V8/collect_v8.py w7 formal --base ${W7_BASE:-results/icpe_next/w7} \
  --files $RUN/design.csv $RUN/a/w7_traces.csv $RUN/a/w7_cells.csv $RUN/a/w7_map.csv $RUN/a/w7_summary.csv $RUN/a/w7_dims.csv $RUN/w7gen_sha.txt \
  --code-sha $CS --runs $RUN "${PRIOR[@]}" --note "PREREG 偏離 12 的 W7 合成參數地圖；x1_sweep 不進 git（RUX3DPRO 的 run 目錄 per/*/out/x1/x1_sweep.csv.xz，sha256 在 w7_traces.csv）"
echo W7_FORMAL_DONE
