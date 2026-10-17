# 13 §3（Q4，使用者 10/17：最優先）：GPU 忙時的「拆」。P0–P5 × 2 模型 × 2 格式；真實 CPU 層；閒／忙；E0、E1；
# cake（Cake，照各方法的拆法）＋ loadall（全部載入）；3 launch × 10 rep、launch 從 300 起。背景吞吐記在每個目錄的 busy_*.json。
# 遮罩沒生效（43a 沒有 OK_cumask）→ 不跑 P4a／P4b。之後：pick-q4（13 §3.2）→ 背景強度掃描（13 §3.3）。
# 計時的 stage：跑的時候不在 GPU 上插任何東西。失敗的 grid 寫 NEEDS_HUMAN、跑下一個；exit 0。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
R=$REPO/results/m10_restore2_mi300x; S=$R/smoke; Q=$R/q4; mkdir -p $Q
METHODS="P0 P1 P2 P3 P4a P4b P5"; [ -f $S/OK_cumask ] || METHODS="P0 P1 P2 P3 P5"
for m in llama qwen; do
  G=$(fmt_of $m fp8)
  for tag in fp8 int4; do
    F=$(fmt_of $m $tag)
    for meth in $METHODS; do
      E=$(python3 $REPO/code/m10_restore2.py env $meth --model $m --fmt $F) || { echo "43b：$meth $m $tag 的環境變數失敗" >> $JOBS/NEEDS_HUMAN; continue; }
      m7run m10-rs2-q4-$m-$tag-$meth env $E bash -c "python $REPO/code/m10_speed.py grid --model $m --fmt $F --base-fmt $G --out-dir $Q/${m}_${tag}_$meth --bw-json $OUT/speed_$m/bw.json --bw-names cpu --configs E0 E1 --refs --lengths --extra-modes loadall --launches 3 --reps 10 --launch0 300 --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" \
        || echo "43b：Q4 $m $tag $meth 失敗（見 runs/*-m10-rs2-q4-$m-$tag-$meth）" >> $JOBS/NEEDS_HUMAN
    done
  done
done
python3 $REPO/code/m10_restore2.py pick-q4 || { echo "43b：pick-q4 失敗 → 背景強度掃描和 44d 用 P0" | tee -a $JOBS/NEEDS_HUMAN; }
PICK=$(python3 -c "import json;print(json.load(open('$R/q4_pick.json')).get('pick') or 'P0')" 2>/dev/null || echo P0)
# 13 §3.3 背景強度掃描：P0 和挑出來的方法；int4_kivi；duty 0.4／0.7／0.9；2 launch × 5 rep
BL=$R/busy_levels_q4.json
[ -f $BL ] || echo '{"idle": 0.0, "busy1": 0.4, "busy2": 0.7, "busy3": 0.9, "_note": "13 §3.3：busy1、busy2 同主線；busy3 沒校準過，只當更忙的方向"}' > $BL
for meth in $(echo P0 $PICK | tr ' ' '\n' | sort -u); do
  for m in llama qwen; do
    G=$(fmt_of $m fp8); E=$(python3 $REPO/code/m10_restore2.py env $meth --model $m --fmt int4_kivi)
    m7run m10-rs2-q4duty-$m-$meth env $E bash -c "python $REPO/code/m10_speed.py grid --model $m --fmt int4_kivi --base-fmt $G --out-dir $R/q4duty/${m}_int4_$meth --bw-json $OUT/speed_$m/bw.json --bw-names cpu --gpu-states busy1 busy2 busy3 --busy-levels $BL --configs E0 E1 --refs --lengths --extra-modes loadall --launches 2 --reps 5 --launch0 350 --blocks-out \$TIARA_RUNS/\$RUN_ID/blocks.csv" \
      || echo "43b：背景強度掃描 $m $meth 失敗" >> $JOBS/NEEDS_HUMAN
  done
done
exit 0
