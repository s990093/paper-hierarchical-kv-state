# 第三階段 job 的共用設定（由每個 job source）。runner 已設 TIARA_GPU 與卡號變數（GPU job；有 USE_ROCR 旗子時是 ROCR_VISIBLE_DEVICES，否則 HIP／CUDA_VISIBLE_DEVICES）。
set -uo pipefail
REPO=/mlsteam/workspace/paper-hierarchical-kv-state
cd $REPO
source code/m7_env.sh                      # venv tiara、HF_HOME、卡號（USE_ROCR → ROCR_VISIBLE_DEVICES，否則 HIP_VISIBLE_DEVICES）=${TIARA_GPU:-0}、m7run
S3=${S3_JOBS:-/mlsteam/data/tiara/stage3/jobs}
RES=$REPO/results/m16_stage3_mi300x
SMOKE=/mlsteam/data/tiara/stage3/smoke     # 冒煙的輸出不進 results
KVD=/var/tmp/tiara_s3/kv_dumps             # 本地盤（overlay）；可重生，容器重啟會不見
nh() { echo "$(date -u +%FT%TZ) $*" | tee -a $S3/NEEDS_HUMAN; }
lastrun() { /bin/ls -t $TIARA_RUNS | grep -m1 -- "$1"; }
