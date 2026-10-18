# 09 §4.3 大掃描複賽（由 m10_confirm.py plan 照海選的 finalists_<m>.json 寫出；12 個品質 process）
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
C=/mlsteam/workspace/paper-hierarchical-kv-state/results/m10_sweep_mi300x/confirm; mkdir -p $C
# 0. 冒煙：每個模型 1 題、全部配置（失敗 → NEEDS_HUMAN，跳過 43 的其餘部分，不擋後面）
for m in llama qwen; do
  M10_CODEC_X=1 m7run m10-confirm-smoke-$m python /mlsteam/workspace/paper-hierarchical-kv-state/code/m10_confirm.py smoke --model $m > $C/smoke_$m.log 2>&1 \
    || { echo "43 confirm smoke $m 失敗（$C/smoke_$m.log）→ 跳過 43" | tee -a $JOBS/NEEDS_HUMAN; tail -20 $C/smoke_$m.log; exit 0; }
done
# 0b. 速度路徑的冒煙（真模型：每個模型第一個格式的全部配置，含 ATT／AXE；CPU 層、閒、1 launch × 1 rep）
M10_CODEC_X=1 m7run m10-confirm-ssmoke-llama bash -c "python /mlsteam/workspace/paper-hierarchical-kv-state/code/m10_confirm.py speed --model llama --fmt int2_kivi --configs E0 SW-ERR-20 SW-ERR-15 E1s SW-R0-15 SW-R1-15 SW-R2-15 SW-R3-15 SW-R4-15 SW-R0-20 SW-R1-20 SW-R2-20 SW-R3-20 SW-R4-20 --out-dir $C/smoke_speed_llama --bw-json $OUT/speed_llama/bw.json --bw-names cpu --gpu-states idle --launches 1 --reps 1 --blocks-out $C/smoke_speed_llama/blocks.csv" > $C/smoke_speed_llama.log 2>&1 || { echo "43 confirm speed smoke llama 失敗（$C/smoke_speed_llama.log）→ 跳過 43" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }
M10_CODEC_X=1 m7run m10-confirm-ssmoke-qwen bash -c "python /mlsteam/workspace/paper-hierarchical-kv-state/code/m10_confirm.py speed --model qwen --fmt kv-int4_kivi-int2_kivi --configs E0 SW-ATT-40 SW-AXE-40 E1s SW-R0-40 SW-R1-40 SW-R2-40 SW-R3-40 SW-R4-40 --out-dir $C/smoke_speed_qwen --bw-json $OUT/speed_qwen/bw.json --bw-names cpu --gpu-states idle --launches 1 --reps 1 --blocks-out $C/smoke_speed_qwen/blocks.csv" > $C/smoke_speed_qwen.log 2>&1 || { echo "43 confirm speed smoke qwen 失敗（$C/smoke_speed_qwen.log）→ 跳過 43" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }
# 1. 速度格子（每個模型 × 格式；會合點給讀法 B、速度判準用）
M10_CODEC_X=1 m7run m10-confirm-grid-llama-int2_kivi bash -c "python /mlsteam/workspace/paper-hierarchical-kv-state/code/m10_confirm.py speed --model llama --fmt int2_kivi --configs E0 SW-ERR-20 SW-ERR-15 E1s SW-R0-15 SW-R1-15 SW-R2-15 SW-R3-15 SW-R4-15 SW-R0-20 SW-R1-20 SW-R2-20 SW-R3-20 SW-R4-20 --out-dir $C/speed_llama --bw-json $OUT/speed_llama/bw.json --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" || { echo "43 confirm grid llama int2_kivi 失敗" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }
M10_CODEC_X=1 m7run m10-confirm-grid-llama-kv-int4_kivi-int2_kivi bash -c "python /mlsteam/workspace/paper-hierarchical-kv-state/code/m10_confirm.py speed --model llama --fmt kv-int4_kivi-int2_kivi --configs SW-ERR-20 SW-ERR-25 E1s SW-R0-20 SW-R1-20 SW-R2-20 SW-R3-20 SW-R4-20 SW-R0-25 SW-R1-25 SW-R2-25 SW-R3-25 SW-R4-25 --out-dir $C/speed_llama --bw-json $OUT/speed_llama/bw.json --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" || { echo "43 confirm grid llama kv-int4_kivi-int2_kivi 失敗" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }
M10_CODEC_X=1 m7run m10-confirm-grid-llama-int4_kivi bash -c "python /mlsteam/workspace/paper-hierarchical-kv-state/code/m10_confirm.py speed --model llama --fmt int4_kivi --configs SW-ERR-30 SW-R0-30 SW-R1-30 SW-R2-30 SW-R3-30 SW-R4-30 E1s --out-dir $C/speed_llama --bw-json $OUT/speed_llama/bw.json --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" || { echo "43 confirm grid llama int4_kivi 失敗" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }
M10_CODEC_X=1 m7run m10-confirm-grid-llama-mxfp4 bash -c "python /mlsteam/workspace/paper-hierarchical-kv-state/code/m10_confirm.py speed --model llama --fmt mxfp4 --configs SW-ATT-20 SW-R0-20 SW-R1-20 SW-R2-20 SW-R3-20 SW-R4-20 E1s --out-dir $C/speed_llama --bw-json $OUT/speed_llama/bw.json --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" || { echo "43 confirm grid llama mxfp4 失敗" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }
M10_CODEC_X=1 m7run m10-confirm-grid-qwen-kv-int4_kivi-int2_kivi bash -c "python /mlsteam/workspace/paper-hierarchical-kv-state/code/m10_confirm.py speed --model qwen --fmt kv-int4_kivi-int2_kivi --configs E0 SW-ATT-40 SW-AXE-40 E1s SW-R0-40 SW-R1-40 SW-R2-40 SW-R3-40 SW-R4-40 --out-dir $C/speed_qwen --bw-json $OUT/speed_qwen/bw.json --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" || { echo "43 confirm grid qwen kv-int4_kivi-int2_kivi 失敗" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }
M10_CODEC_X=1 m7run m10-confirm-grid-qwen-int4_kivi bash -c "python /mlsteam/workspace/paper-hierarchical-kv-state/code/m10_confirm.py speed --model qwen --fmt int4_kivi --configs SW-ATT-25 SW-AXE-25 E1s SW-R0-25 SW-R1-25 SW-R2-25 SW-R3-25 SW-R4-25 --out-dir $C/speed_qwen --bw-json $OUT/speed_qwen/bw.json --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" || { echo "43 confirm grid qwen int4_kivi 失敗" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }
M10_CODEC_X=1 m7run m10-confirm-grid-qwen-kv-int8_token-int4_kivi bash -c "python /mlsteam/workspace/paper-hierarchical-kv-state/code/m10_confirm.py speed --model qwen --fmt kv-int8_token-int4_kivi --configs SW-ATT-10 SW-AXE-10 SW-R0-10 SW-R1-10 SW-R2-10 SW-R3-10 SW-R4-10 E1s --out-dir $C/speed_qwen --bw-json $OUT/speed_qwen/bw.json --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" || { echo "43 confirm grid qwen kv-int8_token-int4_kivi 失敗" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }
# 2. 品質（新 seed 50000 起；T2–T4 各 90、T5 30；讀法 A＋B（b32 閒／忙，實測會合點））
pool /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/jobs_43q.json 8 || exit 1
# 3. 判定（CPU；m10_confirm_analyze.py 在 43 跑完前寫好；沒有就寫 NEEDS_HUMAN）
if [ -f /mlsteam/workspace/paper-hierarchical-kv-state/code/m10_confirm_analyze.py ]; then
  python /mlsteam/workspace/paper-hierarchical-kv-state/code/m10_confirm_analyze.py --dir $C > $C/analyze.out 2>&1 || echo "43 confirm analyze 失敗" >> $JOBS/NEEDS_HUMAN
else
  echo "m10_confirm_analyze.py 還沒寫（43 的資料齊了）" >> $JOBS/NEEDS_HUMAN
fi
exit 0
