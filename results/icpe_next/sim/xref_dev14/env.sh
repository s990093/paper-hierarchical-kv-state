cd /ssd6/hungwei/icpe-next/g1-63b8e32
export RUN_ID=20261107-155727-icpe-sim-xref-dev14 P3_CONST=h200 P3_TIER=h200cpu CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RUN_DIR=
unset X1_MAXTASKS X1_ENGINE X1_RUST_BIN
export M7_TIER_PARAMS=/ssd6/hungwei/icpe-next/g1-63b8e32/results/icpe_h200/t1/tier_params_h200cpu.json
export M7_RES=/ssd6/hungwei/icpe-next/g1-63b8e32/results/m7_write_policy_h200
export X1_TRACE_DIR=/ssd8/hungwei/paper-hkv/datasets/mooncake/traces
export X1_NB_DIR=/ssd6/hungwei/icpe-next/datasets/x1_nb
export X1_FI_DIR=/ssd8/hungwei/paper-hkv/datasets/freeinference_agentic_trace/release
export X1_RUST_WORK=/ssd6/hungwei/icpe-next/x1fast-work
PY=/ssd6/hungwei/icpe-next/venv/bin/python
SW="nice -n 10 $PY -I code/icpe/p3_fvar.py --variant V1 --calib results/icpe_h200/p01/calib"
ON="--pf-expire --noise-overdue resample --pf-gpu-recheck"
P8="LRU LRU+RPF O-BEL O-BELS O-DROP O-NOW O-PF O-ALL"
# 使用者 11/7 選 A：拿掉 wt × 預取到 GPU（旗標關也有的舊 assert；正式 block 都不用；STATUS_SIM 15:56）
XQB="--cpu 8 64 --policies $P8 --slows b32 --modes cake --gpu-gib 0 111.0625 --gpu-write wb --gpu-policies LRU BEL TTLA:10 TTLB:30 --pf-target cpu gpu"
XQT="--cpu 8 64 --policies $P8 --slows b32 --modes cake --gpu-gib 111.0625 --gpu-write wt --gpu-policies LRU BEL TTLA:10 TTLB:30 --pf-target cpu"
XN="--cpu 8 64 --policies LRU+RPF O-BEL O-DROP O-NOW O-PF O-ALL --slows b32 --modes cake --gpu-gib 111.0625 --gpu-write wb --gpu-policies BEL TTLB:30 --pf-target cpu gpu --noise-models P M --noise-lambdas 0.3 --noise-draws B1 B2 --noise-seeds 0"
TSA="--traces nb:semianalysis_N16_s1 --rules nb"
TMC="--traces conversation --rules R1"
CHK="$PY -I /ssd6/hungwei/icpe-next/runs/20261107-155727-icpe-sim-xref-dev14/scripts/dev14_check.py"
