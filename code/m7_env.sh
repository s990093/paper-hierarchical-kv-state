# source 這支：第一階段 m7 的環境（平台 B）
source /mlsteam/workspace/venv/tiara/bin/activate
export HF_HOME=/mlsteam/data/tiara/hf-cache
export HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0
export TIARA_RUNS=/mlsteam/data/tiara/runs
export PYTHONUNBUFFERED=1
R=/mlsteam/workspace/paper-hierarchical-kv-state/results/m7_write_policy_mi300x
m7run() { local short=$1; shift; local id="$(date +%Y%m%d-%H%M%S)-$short"; RUN_ID=$id /mlsteam/workspace/bin/runsh "${short}" env RUN_ID=$id "$@"; }
