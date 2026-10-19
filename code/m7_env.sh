# source 這支：第一階段 m7 的環境
if [ -d /mlsteam/workspace/venv/tiara ]; then   # 平台 B（MI300X / MLSteam）：原本的設定，不變
source /mlsteam/workspace/venv/tiara/bin/activate
export HF_HOME=/mlsteam/data/tiara/hf-cache
# 10/19 起兩張卡：TIARA_GPU 選實體卡（amd-smi／HIP 編號；1＝a6:00.0＝10/18 以前所有計時用的那張）。沒設＝0（原本的行為）
export HIP_VISIBLE_DEVICES=${TIARA_GPU:-0} CUDA_VISIBLE_DEVICES=${TIARA_GPU:-0}
export TIARA_RUNS=/mlsteam/data/tiara/runs
export PYTHONUNBUFFERED=1
R=/mlsteam/workspace/paper-hierarchical-kv-state/results/m7_write_policy_mi300x
m7run() { local short=$1; shift; local id="$(date +%Y%m%d-%H%M%S)-$short"; RUN_ID=$id /mlsteam/workspace/bin/runsh "${short}" env RUN_ID=$id "$@"; }
else                                            # 平台 C（H200 / Slurm）：路徑、R、m7run 全部由 env_h200.sh 給（03_plan_h200 §R）
source "$(dirname "${BASH_SOURCE[0]}")/env_h200.sh"
fi
