# icpe-rux：x1 sweep 用的環境（照 STATUS_RUX 對拍時的 cmd.sh）。source 之後再跑 sweep。
export P3_CONST=h200
export P3_TIER=h200cpu
export M7_TIER_PARAMS=$REPO/results/icpe_h200/t1/tier_params_h200cpu.json
export M7_RES=$REPO/results/m7_write_policy_h200
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CALIB=$REPO/results/icpe_h200/p01/calib
PYREF="$ROOT/venv/bin/python -I $REPO/code/icpe/p3_fvar.py --variant V1 --calib $CALIB"
RSV6="$ROOT/venv/bin/python -I $REPO/code/icpe/rux_x1fast/v6_slowcap/sweep_v6.py --variant V1 --calib $CALIB"
CMP="$ROOT/venv/bin/python -I $REPO/code/icpe/rux_x1fast/v6_slowcap/cmp_v6.py"
