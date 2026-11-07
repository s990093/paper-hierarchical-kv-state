#!/usr/bin/env bash
# main_d14_family.sh — icpe-main：一個家族的偏離 14 正式版本 B 表（第一批）。在 §4.1 的 run 殼裡呼叫：
#   bash code/icpe/main_d14_family.sh <fam> <run_dir>      fam ∈ bl fi mc tl sa
# 做法照版本 B（run 20261106-110854-icpe-main-verB-p1），只是 sweep 先用 main_d14patch.py 把偏離 14 重跑的列換進去：
#   p1（正式）、p1dev5（並列）、p1h1800／h3600（只報告）、p1_p90（只報告）。
# 檢查（不過就停）：rowcheck；p1＝icpe-data 的 b/analysis_d14/<fam>_p1（去掉 run_id／ts／host 後逐格相同）；
#   O-ORD／O-DROP／O-NOW／O-GPU 和舊版本 B 表 eqsel 相同（5 張表都比）；版本 B 流程裡的逐格關係（dev5＝p1 的 O-GPU／O-ALL；截尾＝不截尾的 O-ORD／O-PF／O-GPU）。
set -euo pipefail
f=$1; R=$2
PY="nice -n 19 /ssd6/hungwei/icpe-next/venv/bin/python -I"
O=$R/out; I=$R/in; mkdir -p $O $I
B=results/icpe_next/b; VB=results/icpe_next/main/verB; AN=code/icpe/next_sim_analyze.py
LIST=results/icpe_next/sim/dev14_rerun_list.csv
TTL4="TTLA:1 TTLA:3 TTLA:30 TTLA:100 TTLB:1 TTLB:3 TTLB:30 TTLB:100"
case $f in
  bl) PM=bl_main; BM="bailian3"; PP=bl_p1; BT="bailian3_ttl"; BR="bailian3_trunc";;
  fi) PM=old_main; BM="fi6w"; PP=old_p1; BT="fi6w_ttl"; BR="fi6w_trunc";;
  mc) PM=old_main; BM="mooncake"; PP=old_p1; BT="mooncake_ttl"; BR="mooncake_trunc";;
  tl) PM=tl_main; BM="$(cd $B/tl_main && ls -d */ | tr -d / | tr '\n' ' ')"; PP=tl_p1; BT="$(cd $B/tl_p1 && ls -d *_ttl | tr '\n' ' ')"; BR="$(cd $B/tl_p1 && ls -d *_trunc | tr '\n' ' ')";;
  sa) PM=sa_main; BM="$(cd $B/sa_main && ls -d */ | tr -d / | tr '\n' ' ')"; PP=sa_p1; BT="$(cd $B/sa_p1 && ls -d *_ttl | tr '\n' ' ')"; BR="$(cd $B/sa_p1 && ls -d *_trunc | tr '\n' ' ')";;
  *) echo "不認得的家族 $f" >&2; exit 2;;
esac
echo "== $f $(date -Is)：main $PM {$BM}；P1 $PP {$BT} {$BR}"
# 1. 補列（icpe-data next_d14_patch.py 的獨立實作）
$PY code/icpe/main_d14patch.py --list $LIST --root $B --plan $PM --block $BM --out-dir $I
$PY code/icpe/main_d14patch.py --list $LIST --root $B --plan $PP --block $BT $BR --out-dir $I/p1
M=$(for b in $BM; do echo $I/$PM/$b/x1_sweep.csv; done); T=$(for b in $BT; do echo $I/p1/$PP/$b/x1_sweep.csv; done); Rt=$(for b in $BR; do echo $I/p1/$PP/$b/x1_sweep.csv; done)
# 2. 列數
$PY code/icpe/main_rowcheck.py --label ${f}_d14_main --sweep $M --expect-cells 54 --expect-cpu 17 --expect-gpu 8 --out $O/rowcheck.csv
$PY code/icpe/main_rowcheck.py --label ${f}_d14_p1_ttl --sweep $T --expect-cells 54 --expect-cpu 15 --expect-gpu 8 --out $O/rowcheck.csv
$PY code/icpe/main_rowcheck.py --label ${f}_d14_p1_trunc --sweep $Rt --expect-cells 54 --expect-cpu 6 --expect-gpu 16 --out $O/rowcheck.csv
# 3. 上限表（next_sim_analyze.py 不改；和版本 B 同樣的 5 種）
$PY $AN --sweep $M $T --out-dir $O --out-prefix ${f}_p1 --pf-target cpu
$PY $AN --sweep $M $T --out-dir $O --out-prefix ${f}_p1dev5 --pf-target cpu --rival-only-gpu $TTL4
$PY $AN --sweep $M $T $Rt --out-dir $O --out-prefix ${f}_p1h1800 --pf-target cpu --dead-h 1800
$PY $AN --sweep $M $T $Rt --out-dir $O --out-prefix ${f}_p1h3600 --pf-target cpu --dead-h 3600
$PY $AN --sweep $M $T --out-dir $O --out-prefix ${f}_p1_p90 --pf-target cpu --metric p90_ret
# 4. 和 icpe-data 的檢查表逐格相同
$PY - $O/${f}_p1_headroom_components.csv $B/analysis_d14/${f}_p1/${f}_p1_headroom_components.csv.xz <<'PYEOF'
import sys, pandas as pd
k = ["trace", "rule", "cpu_GiB", "slow", "gpu", "restore", "gpu_GiB", "gpu_write", "slow_cap_frac", "component"]
a, b = (pd.read_csv(p).drop(columns=["run_id", "ts", "host"]).sort_values(k).reset_index(drop=True) for p in sys.argv[1:])
pd.testing.assert_frame_equal(a, b, check_exact=True)
print(f"偏離 14 版本 B：我的 p1 和 icpe-data 的檢查表逐格相同（{len(a)} 列）← {sys.argv[2]}")
PYEOF
# 5. 不該變的元件＝舊版本 B（5 張表都比）；變的元件另外列差異
for t in p1 p1dev5 p1h1800 p1h3600 p1_p90; do
  $PY code/icpe/main_ab.py cells --name ${f}_${t}_d14_eqsel_old_nonPF --x $O/${f}_${t}_headroom_components.csv --y $VB/${f}_${t}_headroom_components.csv.xz --rel eqsel --component O-ORD O-DROP O-NOW O-GPU --out $O/cellcheck.csv
done
# 6. 版本 B 流程裡必然成立的逐格關係
$PY code/icpe/main_ab.py cells --name ${f}_d14_B_eq_dev5_OGPU_OALL --x $O/${f}_p1_headroom_components.csv --y $O/${f}_p1dev5_headroom_components.csv --rel eq --component O-GPU "O-ALL(report)" --out $O/cellcheck.csv
for h in 1800 3600; do
  $PY code/icpe/main_ab.py cells --name ${f}_d14_h${h}_eq_B_untruncated --x $O/${f}_p1h${h}_headroom_components.csv --y $O/${f}_p1_headroom_components.csv --rel eq --component O-ORD O-PF O-GPU --out $O/cellcheck.csv
done
# 7. O-PF／O-ALL 和舊版本 B 的差（只報告）
$PY - $f $O $VB <<'PYEOF'
import sys, os, socket
from datetime import datetime
import pandas as pd
f, O, VB = sys.argv[1:]
k = ["trace", "rule", "cpu_GiB", "slow", "gpu", "restore", "gpu_GiB", "gpu_write", "slow_cap_frac", "component"]
rows = []
for t in ["p1", "p1dev5", "p1h1800", "p1h3600", "p1_p90"]:
    n = pd.read_csv(f"{O}/{f}_{t}_headroom_components.csv"); o = pd.read_csv(f"{VB}/{f}_{t}_headroom_components.csv.xz")
    m = n.merge(o, on=k, suffixes=("", "_old"))
    assert len(m) == len(n) == len(o), f"{t}：新舊格數不同 {len(n)} {len(o)} {len(m)}"
    for c, g in m.groupby("component"):
        d = g.headroom - g.headroom_old
        rows.append(dict(table=t, component=c, n_cells=len(g), n_changed=int((d != 0).sum()), d_min=d.min(), d_median=d.median(), d_max=d.max(),
                         n_ge15_old=int((g.headroom_old >= 0.15).sum()), n_ge15_new=int((g.headroom >= 0.15).sum()),
                         max_old=g.headroom_old.max(), max_new=g.headroom.max(),
                         n_cross_up=int(((g.headroom_old < 0.15) & (g.headroom >= 0.15)).sum()),
                         n_cross_down=int(((g.headroom_old >= 0.15) & (g.headroom < 0.15)).sum())))
D = pd.DataFrame(rows)
for i, (kk, v) in enumerate(dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"), host=socket.gethostname(), family=f).items()):
    D.insert(i, kk, v)
D.to_csv(f"{O}/{f}_d14_vs_verB.csv", index=False)
pd.set_option("display.width", 250)
print(D[D.table.isin(["p1"])].drop(columns=["run_id", "ts", "host"]).round(4).to_string(index=False))
PYEOF
echo "== $f done $(date -Is) =="
