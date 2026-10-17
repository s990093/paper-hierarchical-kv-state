# 13 號計畫（docs/precision_20261013/13_restore2_plan.md）§8：新程式的 GPU 冒煙（逐位元、遮罩生效、G5／none／nodq、優先權、新格式 kernel）。
# 只寫 results/m10_restore2_mi300x/smoke/（冒煙的時間不是結果）。沒過的部分：用到它的 stage 移到 hold/、寫 NEEDS_HUMAN；其他照跑。exit 0。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
R=$REPO/results/m10_restore2_mi300x; S=$R/smoke; mkdir -p $S
hold() { for f in "$@"; do [ -f $JOBS/queue/$f ] && mv $JOBS/queue/$f $JOBS/hold/ && echo "移到 hold：$f"; done; }
# 1. 新格式的融合 kernel：GPU 上和原寫法逐位元相同（13 §5）
m7run m10-rs2-dqx-selftest python $REPO/code/m10_dequant_fast_x.py --self-test --device cuda --out $S/dqx_selftest_gpu.json \
  || { echo "43a：新格式融合 kernel 在 GPU 上和原寫法不是逐位元相同（$S/dqx_selftest_gpu.json）→ 43d 擱置" | tee -a $JOBS/NEEDS_HUMAN; hold 43d_rs2_newfmt.sh; }
# 2. attention 對照組：讀壓縮＝同一 kernel 讀拆好的 BF16（逐位元）、對 SDPA 的誤差（13 §6 (b)）
m7run m10-rs2-attn-selftest python $REPO/code/m10_attn_q.py selftest --device cuda --out $S/attn_selftest_gpu.json \
  && touch $S/OK_attn || echo "43a：attention 對照組自測沒過（$S/attn_selftest_gpu.json）→ 43e 只跑 (a)" | tee -a $JOBS/NEEDS_HUMAN
# 3. CU 遮罩有沒有生效（13 §3.1）
m7run m10-rs2-cumask python $REPO/code/m10_restore2.py cumask --out $S/cumask.json \
  && touch $S/OK_cumask || echo "43a：HSA_CU_MASK 沒生效或量不到（$S/cumask.json）→ 43b 不跑 P4a／P4b" | tee -a $JOBS/NEEDS_HUMAN
# 4. 真模型（Qwen、int4_kivi、1 launch × 1 rep）：每格第一次都檢查 KV
G=$(fmt_of qwen fp8)
K=$(python3 $REPO/code/m10_dqsweep.py env S0 --model qwen --fmt int4_kivi)
GRID="python $REPO/code/m10_speed.py grid --model qwen --fmt int4_kivi --base-fmt $G --bw-json $OUT/speed_qwen/bw.json --refs --lengths --launches 1 --reps 1 --launch0 390"
m7run m10-rs2-smoke-g5 env $K M10_MEET=g5 bash -c "$GRID --out-dir $S/g5 --bw-names cpu b32 --gpu-states idle --configs E0 E1 R25 --extra-modes loadall claim --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv"
m7run m10-rs2-smoke-nodq env $K M10_MEET=none M10_DQ_SCHED=nodq bash -c "$GRID --out-dir $S/nodq --bw-names cpu --gpu-states idle --configs E0 E1 --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv"
if [ -f $S/OK_cumask ]; then MK=$(python3 -c "import json;print(json.load(open('$S/cumask.json'))['masks']['p4b'])"); else MK=""; fi
K8=$(python3 $REPO/code/m10_dqsweep.py env S2k8 --model qwen --fmt int4_kivi)
m7run m10-rs2-smoke-p5mask env $K8 M10_RS_PRIO=1 M10_BUSY_CU_MASK=$MK bash -c "$GRID --out-dir $S/p5mask --bw-names cpu --gpu-states busy --configs E0 E1 --extra-modes loadall --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv"
# 5. 新格式融合 kernel 走大掃描的速度路徑（Llama int2_kivi）
M10_CODEC_X=1 M10_DEQUANT_FAST=1 M10_DEQUANT_FAST_X=1 m7run m10-rs2-smoke-dqx bash -c "python $REPO/code/m10_confirm.py speed --model llama --fmt int2_kivi --configs E0 E1s SW-R0-20 --out-dir $S/dqx --bw-json $OUT/speed_llama/bw.json --bw-names cpu --gpu-states idle --launches 1 --reps 1 --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv"
python3 $REPO/code/m10_restore2.py smoke-check --dir $S \
  || { echo "43a：還原第二輪的冒煙沒過（見 $S、43a log）→ 43b 43c 43e 擱置（43d 看第 1 項）" | tee -a $JOBS/NEEDS_HUMAN; hold 43b_rs2_busy.sh 43c_rs2_g5.sh 43e_rs2_attn.sh 44cz_rs2_plan44d.sh; }
exit 0
