# H200 才有的 stage（13_h200_repro §14 B）：13 號 Q4 的 P4（留運算單元給還原）用 CUDA MPS 代替 ROCm 的 HSA_CU_MASK。只報告。量時間的 stage。exit 0。
#   開 MPS daemon（job 自己的 pipe／log 目錄）→ 限制檢查（h200_mps.py check：背景單獨跑的 matmul 時間要變慢）→ gpu_guard 冒煙（MPS server 算自己的）
#   → P0mps（MPS、不限）、P4a_mps（背景 7/8 SM）、P4b_mps（背景 3/4 SM）× 2 模型 × fp8／int4；grid 指令和 43b 的 P0 一字不改 → 報告。
source __JOBS__/lib.sh
set -uo pipefail
R=$REPO/results/m10_restore2_h200; Q=$R/q4mps; mkdir -p $Q
command -v nvidia-cuda-mps-control >/dev/null || { echo "h2：找不到 nvidia-cuda-mps-control → P4-MPS 不跑" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_${SLURM_JOB_ID} CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_${SLURM_JOB_ID}
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-smi --query-gpu=index,compute_mode --format=csv | tee $Q/compute_mode.txt
nvidia-cuda-mps-control -d || { echo "h2：MPS daemon 起不來 → P4-MPS 不跑" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }
trap 'echo quit | nvidia-cuda-mps-control; sleep 2; mkdir -p $Q/mps_log; cp $CUDA_MPS_LOG_DIRECTORY/* $Q/mps_log/ 2>/dev/null; rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY' EXIT
export GPU_GUARD_JOB_CGROUP_OWN=1
m7run m10h-mps-check python $REPO/code/h200_mps.py check --out $Q/mps.json \
  || { echo "h2：MPS 限制沒生效或 server 沒起來（$Q/mps.json、runs/*-m10h-mps-check）→ P4-MPS 不跑" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }
PA=$(python3 -c "import json;print(json.load(open('$Q/mps.json'))['pct']['p4a'])")
PB=$(python3 -c "import json;print(json.load(open('$Q/mps.json'))['pct']['p4b'])")
E=$(python3 $REPO/code/m10_restore2.py env P0 --model qwen --fmt int4_kivi)
m7run m10h-mps-guardsmoke env $E bash -c "python $REPO/code/m10_speed.py grid --model qwen --fmt int4_kivi --base-fmt fp8_head --out-dir $Q/_guardsmoke --bw-json $OUT/speed_qwen/bw.json --bw-names cpu --configs E1 --refs --lengths --launches 1 --reps 1 --launch0 300"
python3 - $Q/_guardsmoke <<'PY' || { echo "h2：MPS 底下 gpu_guard 不是 CLEAN（$Q/_guardsmoke）→ P4-MPS 不跑" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }
import glob, json, sys
gs = [json.load(open(g)) for g in glob.glob(sys.argv[1] + "/gpu_guard_*.json")]
print([g.get("verdict") for g in gs])
sys.exit(0 if gs and all(g.get("verdict") == "CLEAN" for g in gs) else 1)
PY
for m in llama qwen; do
  G=$(fmt_of $m fp8)
  for tag in fp8 int4; do
    F=$(fmt_of $m $tag); E=$(python3 $REPO/code/m10_restore2.py env P0 --model $m --fmt $F)
    for meth in P0mps P4a_mps P4b_mps; do
      case $meth in P0mps) X="" ;; P4a_mps) X="M10_BUSY_MPS_PCT=$PA" ;; P4b_mps) X="M10_BUSY_MPS_PCT=$PB" ;; esac
      m7run m10h-mps-q4-$m-$tag-$meth env $E $X bash -c "python $REPO/code/m10_speed.py grid --model $m --fmt $F --base-fmt $G --out-dir $Q/${m}_${tag}_$meth --bw-json $OUT/speed_$m/bw.json --bw-names cpu --configs E0 E1 --refs --lengths --extra-modes loadall --launches 3 --reps 10 --launch0 300 --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" \
        || echo "h2：P4-MPS $m $tag $meth 失敗（runs/*-m10h-mps-q4-$m-$tag-$meth）" >> $JOBS/NEEDS_HUMAN
    done
  done
done
python3 $REPO/code/h200_mps.py report > /dev/null || echo "h2：報告失敗" >> $JOBS/NEEDS_HUMAN
exit 0
