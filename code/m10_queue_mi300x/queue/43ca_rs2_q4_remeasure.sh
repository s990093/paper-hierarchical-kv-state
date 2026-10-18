# 10/18 11:4xZ：43b 的 P2（S2k8）× Llama × FP8 第 3 個 launch（L302）gpu_guard CONTAMINATED_DURING_RUN（1 個樣本、pid 3965560、peak 0 MiB、09:49:27Z；
# 已經不在了，查不到是誰）→ 那一格整格不能用 → pick-q4 少了 P2（3／4 格）、挑了 P3。照 CLAUDE.md §3：計時被污染要重量。
# 這裡只重量那一格（同一個指令、同一組 launch 編號；新的 run 會接在同一個 restores.csv 後面，latest_rows 只讀最新的 run），再重新 pick-q4。
# 新的 pick 不是 P0／P3 → 補新 pick 的背景強度掃描（和 43b 同一個指令）。舊的 q4_pick.json 留成 q4_pick_43b.json。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
R=$REPO/results/m10_restore2_mi300x; Q=$R/q4
m=llama; tag=fp8; meth=P2
G=$(fmt_of $m fp8); F=$(fmt_of $m $tag)
E=$(python3 $REPO/code/m10_restore2.py env $meth --model $m --fmt $F) || { echo "43ca：$meth $m $tag 的環境變數失敗" >> $JOBS/NEEDS_HUMAN; exit 0; }
m7run m10-rs2-q4-$m-$tag-$meth-remeasure env $E bash -c "python $REPO/code/m10_speed.py grid --model $m --fmt $F --base-fmt $G --out-dir $Q/${m}_${tag}_$meth --bw-json $OUT/speed_$m/bw.json --bw-names cpu --configs E0 E1 --refs --lengths --extra-modes loadall --launches 3 --reps 10 --launch0 300 --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" \
  || { echo "43ca：Q4 重量 $m $tag $meth 失敗（見 runs/*-m10-rs2-q4-$m-$tag-$meth-remeasure）" >> $JOBS/NEEDS_HUMAN; exit 0; }
cp $R/q4_pick.json $R/q4_pick_43b.json
OLD=$(python3 -c "import json;print(json.load(open('$R/q4_pick_43b.json')).get('pick') or 'P0')")
python3 $REPO/code/m10_restore2.py pick-q4 || { echo "43ca：pick-q4 失敗（重量後）" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }
PICK=$(python3 -c "import json;print(json.load(open('$R/q4_pick.json')).get('pick') or 'P0')")
echo "pick：43b=$OLD → 重量後=$PICK"
python3 -c "import json;d=json.load(open('$R/q4_pick.json'));import sys;sys.exit(0 if d['table']['P2']['n_cells']==4 else 1)" \
  || echo "43ca：重量後 P2 還是不到 4 格（又被污染？）→ 照現在的 q4_pick.json 走" >> $JOBS/NEEDS_HUMAN
BL=$R/busy_levels_q4.json
if [ "$PICK" != P0 ] && [ "$PICK" != "$OLD" ]; then
  for m in llama qwen; do
    G=$(fmt_of $m fp8); E=$(python3 $REPO/code/m10_restore2.py env $PICK --model $m --fmt int4_kivi)
    m7run m10-rs2-q4duty-$m-$PICK env $E bash -c "python $REPO/code/m10_speed.py grid --model $m --fmt int4_kivi --base-fmt $G --out-dir $R/q4duty/${m}_int4_$PICK --bw-json $OUT/speed_$m/bw.json --bw-names cpu --gpu-states busy1 busy2 busy3 --busy-levels $BL --configs E0 E1 --refs --lengths --extra-modes loadall --launches 2 --reps 5 --launch0 350 --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" \
      || echo "43ca：背景強度掃描 $m $PICK 失敗" >> $JOBS/NEEDS_HUMAN
  done
fi
exit 0
