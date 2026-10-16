# 大掃描海選（09 §4.1）；題數與組合由 sweep_plan_n.py 照試跑決定：{'n': 24, 'policies': ['F', 'M', 'B', 'R0', 'R1', 'R2', 'R3', 'R4', 'ERR', 'ATT', 'AXE', 'ORC'], 'fracs': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9], 'label': '全部', 'est_h': 9.87, 'pilot_wall_h': 0.823, 'per_item_h': 0.0514}
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
pool /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/jobs_41.json 4 || exit 1
# 彙整、挑選手（09 §4.2）、圖表：CPU；出錯不擋後面的 GPU 工作（寫 NEEDS_HUMAN）
for m in llama qwen; do
  python /mlsteam/workspace/paper-hierarchical-kv-state/code/m10_sweep_analyze.py --model $m --csv /mlsteam/data/tiara/sweep_m10/screen_${m}_s*.csv.gz --out-dir /mlsteam/workspace/paper-hierarchical-kv-state/results/m10_sweep_mi300x/analysis     > /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/queue/41_analyze_$m.out 2>&1 || echo "sweep analyze $m 失敗（queue/41_analyze_$m.out）" >> /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/NEEDS_HUMAN
done
