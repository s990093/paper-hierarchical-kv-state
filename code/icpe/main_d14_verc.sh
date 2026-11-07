#!/usr/bin/env bash
# main_d14_verc.sh — icpe-main：偏離 14 第二批（修正版版本 C，results/icpe_next/apc_d14/<變體>/）的獨立核對。
#   bash code/icpe/main_d14_verc.sh <label> <run_dir> <變體名> ...      例：bl bailian_coder bailian_thinking bailian_traceA
# 對每個變體：
#   1. rowcheck 新 sweep（54 格、CPU 17 × GPU 18、cake／wb／預取到 CPU）；
#   2. 拿掉 BELP 的列，用 next_sim_analyze.py 原樣重算（和版本 C 當初同一個獨立核對）：O-ORD／O-DROP／O-NOW／O-PF／O-ALL 要和新的 verC 表 eqsel 相同；O-GPU 不含 BELP ≤ 含 BELP；
#   3. 新 verC 對舊 apc/ 的 verC：O-ORD／O-DROP／O-NOW／O-GPU eqsel 相同；O-PF／O-ALL 的變動另外列（只印變動格數，show 才印值）。
set -euo pipefail
lab=$1; R=$2; shift 2; SHOW=${SHOW:-}
PY="nice -n 19 /ssd6/hungwei/icpe-next/venv/bin/python -I"
O=$R/out; I=$R/in; mkdir -p $O $I
A=results/icpe_next/apc; N=results/icpe_next/apc_d14; AN=code/icpe/next_sim_analyze.py
for v in "$@"; do [ -f $N/$v/verC_headroom_components.csv ] && [ -f $N/$v/x1_sweep.csv.xz ] || { echo "缺 $N/$v 的 verC 表或 sweep" >&2; exit 3; }; done
$PY code/icpe/main_rowcheck.py --label ${lab}_d14_verC --sweep $(for v in "$@"; do echo $N/$v/x1_sweep.csv.xz; done) --expect-cells 54 --expect-cpu 17 --expect-gpu 18 --expect-value restore=cake gpu_write=wb pf_target=cpu --out $O/rowcheck.csv
cat_tables() { $PY - "$@" <<'PYEOF'
import sys, pandas as pd
out, *src = sys.argv[1:]
d = pd.concat([pd.read_csv(p) for p in src], ignore_index=True)
assert not d.duplicated(["trace","rule","cpu_GiB","slow","gpu","restore","gpu_GiB","gpu_write","slow_cap_frac","component"]).any()
d.to_csv(out, index=False); print(f"{len(src)} 個檔 → {out}（{len(d)} 列）")
PYEOF
}
cat_tables $O/${lab}_d14_verC_headroom_components.csv $(for v in "$@"; do echo $N/$v/verC_headroom_components.csv; done)
cat_tables $O/${lab}_old_verC_headroom_components.csv $(for v in "$@"; do echo $A/$v/verC_headroom_components.csv; done)
$PY code/icpe/main_rowfilter.py --in $(for v in "$@"; do echo $N/$v/x1_sweep.csv.xz; done) --col gpu_policy --drop BELP --out-dir $I/$lab > $O/${lab}_rowfilter.log
tail -1 $O/${lab}_rowfilter.log
$PY $AN --sweep $(ls $I/$lab/*.csv | grep -v manifest) --out-dir $O --out-prefix ${lab}_d14_CnoBELP --pf-target cpu --metric mean_ret > $O/${lab}_analyze_stdout.log
echo "next_sim_analyze ${lab}_d14_CnoBELP 完成（摘要未顯示）"
$PY code/icpe/main_ab.py cells --name ${lab}_d14_verC_eqsel_CnoBELP_nonGPU --x $O/${lab}_d14_CnoBELP_headroom_components.csv --y $O/${lab}_d14_verC_headroom_components.csv --rel eqsel --component O-ORD O-DROP O-NOW O-PF "O-ALL(report)" --out $O/cellcheck.csv
$PY code/icpe/main_ab.py cells --name ${lab}_d14_CnoBELP_le_verC_OGPU --x $O/${lab}_d14_CnoBELP_headroom_components.csv --y $O/${lab}_d14_verC_headroom_components.csv --rel le --component O-GPU --out $O/cellcheck.csv
$PY code/icpe/main_ab.py cells --name ${lab}_d14_verC_eqsel_old_nonPF --x $O/${lab}_d14_verC_headroom_components.csv --y $O/${lab}_old_verC_headroom_components.csv --rel eqsel --component O-ORD O-DROP O-NOW O-GPU --out $O/cellcheck.csv
$PY - $lab $O "$SHOW" <<'PYEOF'
import sys, os, socket
from datetime import datetime
import pandas as pd
lab, O, show = sys.argv[1:]
k = ["trace", "rule", "cpu_GiB", "slow", "gpu", "restore", "gpu_GiB", "gpu_write", "slow_cap_frac", "component"]
n = pd.read_csv(f"{O}/{lab}_d14_verC_headroom_components.csv"); o = pd.read_csv(f"{O}/{lab}_old_verC_headroom_components.csv")
m = n.merge(o, on=k, suffixes=("", "_old")); assert len(m) == len(n) == len(o)
rows = []
for c, g in m.groupby("component"):
    d = g.headroom - g.headroom_old
    rows.append(dict(component=c, n_cells=len(g), n_changed=int((d != 0).sum()), n_rival_changed=int((g.T_rival != g.T_rival_old).sum()),
                     d_min=d.min(), d_max=d.max(), n_ge15_old=int((g.headroom_old >= 0.15).sum()), n_ge15_new=int((g.headroom >= 0.15).sum()),
                     max_old=g.headroom_old.max(), max_new=g.headroom.max(),
                     n_cross_up=int(((g.headroom_old < 0.15) & (g.headroom >= 0.15)).sum()), n_cross_down=int(((g.headroom_old >= 0.15) & (g.headroom < 0.15)).sum())))
D = pd.DataFrame(rows)
for i, (kk, v) in enumerate(dict(run_id=os.environ.get("RUN_ID", "adhoc"), ts=datetime.now().astimezone().isoformat(timespec="seconds"), host=socket.gethostname(), family=lab).items()):
    D.insert(i, kk, v)
D.to_csv(f"{O}/{lab}_d14_verC_vs_old.csv", index=False)
pd.set_option("display.width", 250)
print(D[["component", "n_cells", "n_changed", "n_rival_changed"] + (["d_min", "d_max", "n_ge15_old", "n_ge15_new", "max_old", "max_new", "n_cross_up", "n_cross_down"] if show else [])].round(4).to_string(index=False))
PYEOF
echo "== $lab done $(date -Is) =="
