#!/usr/bin/env bash
# main_d14_nb.sh — icpe-main：對話資料集（偏離 13）／agent 軌跡（偏離 17）的偏離 14 正式版本 B 上限表（第一批），只有 p1（同之前的做法）。
#   bash code/icpe/main_d14_nb.sh <prefix> <run_dir> [show]      prefix ∈ sg wc uc lm（chat_*）、ms oh（agent_*）
# 補列（main_d14patch.py）→ rowcheck（main 17×8、_ttl 15×8）→ next_sim_analyze p1 → 檢查：
#   p1＝icpe-data 的 b/analysis_d14/<prefix>_p1（去掉 run_id／ts／host 後逐格相同）；
#   O-ORD／O-DROP／O-NOW／O-GPU 和原本的 b/analysis/<prefix>_p1 eqsel 相同。
# 對話資料集的判定照使用者 11/7 等第二批（版本 C）之後才算；這裡**只印檢查結果和變動格數，不印上限的值**（第三個參數 show 才印）。
set -euo pipefail
x=$1; R=$2; SHOW=${3:-}
PY="nice -n 19 /ssd6/hungwei/icpe-next/venv/bin/python -I"
O=$R/out; I=$R/in; mkdir -p $O $I
B=results/icpe_next/b; AN=code/icpe/next_sim_analyze.py; LIST=results/icpe_next/sim/dev14_rerun_list.csv
case $x in sg|wc|uc|lm) PM=chat_main; PP=chat_p1;; ms|oh) PM=agent_main; PP=agent_p1;; *) echo "不認得 $x" >&2; exit 2;; esac
BM=$(cd $B/$PM && ls -d ${x}_* | grep -v _ttl | tr '\n' ' '); BT=$(cd $B/$PP && ls -d ${x}_*_ttl | tr '\n' ' ')
echo "== $x $(date -Is)：$PM {$BM}；$PP {$BT}"
$PY code/icpe/main_d14patch.py --list $LIST --root $B --plan $PM --block $BM --out-dir $I
$PY code/icpe/main_d14patch.py --list $LIST --root $B --plan $PP --block $BT --out-dir $I/p1
M=$(for b in $BM; do echo $I/$PM/$b/x1_sweep.csv; done); T=$(for b in $BT; do echo $I/p1/$PP/$b/x1_sweep.csv; done)
# 原本的版本 B 表：有的家族存 .csv.xz、有的存 .csv（wc），兩個都找、必須剛好一個
OLD=$(ls $B/analysis/${x}_p1/${x}_p1_headroom_components.csv $B/analysis/${x}_p1/${x}_p1_headroom_components.csv.xz 2>/dev/null || true)
if [ "$(echo $OLD | wc -w)" != 1 ]; then echo "原本的 ${x}_p1 上限表不是剛好一個：[$OLD]" >&2; exit 3; fi
NEWCHK=$(ls $B/analysis_d14/${x}_p1/${x}_p1_headroom_components.csv $B/analysis_d14/${x}_p1/${x}_p1_headroom_components.csv.xz 2>/dev/null || true)
if [ "$(echo $NEWCHK | wc -w)" != 1 ]; then echo "icpe-data 的 ${x}_p1 檢查表不是剛好一個：[$NEWCHK]" >&2; exit 3; fi
$PY code/icpe/main_rowcheck.py --label ${x}_d14_main --sweep $M --expect-cells 54 --expect-cpu 17 --expect-gpu 8 --out $O/rowcheck.csv
$PY code/icpe/main_rowcheck.py --label ${x}_d14_p1_ttl --sweep $T --expect-cells 54 --expect-cpu 15 --expect-gpu 8 --out $O/rowcheck.csv
$PY $AN --sweep $M $T --out-dir $O --out-prefix ${x}_p1 --pf-target cpu > $O/${x}_analyze_stdout.log   # next_sim_analyze 會印摘要表 → 不進主 log
echo "next_sim_analyze ${x}_p1 完成（摘要在 out/${x}_analyze_stdout.log，未顯示）"
$PY - $O/${x}_p1_headroom_components.csv $NEWCHK <<'PYEOF'
import sys, pandas as pd
k = ["trace", "rule", "cpu_GiB", "slow", "gpu", "restore", "gpu_GiB", "gpu_write", "slow_cap_frac", "component"]
a, b = (pd.read_csv(p).drop(columns=["run_id", "ts", "host"]).sort_values(k).reset_index(drop=True) for p in sys.argv[1:])
pd.testing.assert_frame_equal(a, b, check_exact=True)
print(f"偏離 14 版本 B：我的 p1 和 icpe-data 的檢查表逐格相同（{len(a)} 列）← {sys.argv[2]}")
PYEOF
$PY code/icpe/main_ab.py cells --name ${x}_p1_d14_eqsel_old_nonPF --x $O/${x}_p1_headroom_components.csv --y $OLD --rel eqsel --component O-ORD O-DROP O-NOW O-GPU --out $O/cellcheck.csv
$PY - $x $O $OLD "$SHOW" <<'PYEOF'
import sys, os, socket
from datetime import datetime
import pandas as pd
x, O, OLD, show = sys.argv[1:]
k = ["trace", "rule", "cpu_GiB", "slow", "gpu", "restore", "gpu_GiB", "gpu_write", "slow_cap_frac", "component"]
n = pd.read_csv(f"{O}/{x}_p1_headroom_components.csv"); o = pd.read_csv(OLD)
m = n.merge(o, on=k, suffixes=("", "_old"))
assert len(m) == len(n) == len(o)
rows = []
for c, g in m.groupby("component"):
    d = g.headroom - g.headroom_old
    rows.append(dict(component=c, n_cells=len(g), n_changed=int((d != 0).sum()), d_min=d.min(), d_median=d.median(), d_max=d.max(),
                     n_ge15_old=int((g.headroom_old >= 0.15).sum()), n_ge15_new=int((g.headroom >= 0.15).sum()),
                     max_old=g.headroom_old.max(), max_new=g.headroom.max(),
                     n_cross_up=int(((g.headroom_old < 0.15) & (g.headroom >= 0.15)).sum()),
                     n_cross_down=int(((g.headroom_old >= 0.15) & (g.headroom < 0.15)).sum())))
D = pd.DataFrame(rows)
for i, (kk, v) in enumerate(dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"), host=socket.gethostname(), family=x).items()):
    D.insert(i, kk, v)
D.to_csv(f"{O}/{x}_d14_vs_verB.csv", index=False)
pd.set_option("display.width", 250)
print(D[["component", "n_cells", "n_changed"] + (["d_min", "d_max", "n_ge15_old", "n_ge15_new", "max_old", "max_new", "n_cross_up", "n_cross_down"] if show else [])].round(4).to_string(index=False))
PYEOF
echo "== $x done $(date -Is) =="
