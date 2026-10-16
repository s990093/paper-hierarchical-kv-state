# INT4 階 Step 2（33c 的混合配置）跑完 → 凍結的 m10_analyze.py 判穩定差距第 1、2 條與 K2（analyze_step2_int4.sh）。10/16 10:5xZ 加的；只用 CPU、幾秒。
# K2 成立（第 1、2 條沒有任何配置成立）→ 03 §2.4「位置沒有貢獻」→ 主線整個停，佇列照跑（後面都是探索與只報告）。
# 第 1、2 條有配置成立 → 第 3 條要新 seed 重跑（03 §2.4）→ 寫 NEEDS_HUMAN，由人寫重跑的 stage；佇列照跑，GPU 不空著。
# 分析出錯 → NEEDS_HUMAN、exit 0。
source /tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs/lib.sh
set -uo pipefail
C=$(ls $JOBS/queue/33c_quality_int4_mixed.done $JOBS/queue/33c_quality_int4_mixed.running $JOBS/queue/33c_quality_int4_mixed.sh 2>/dev/null | head -1)
m=$(sed -n 's/^MODELS="\([a-z]*\)".*/\1/p' "$C" | head -1)
[ -n "$m" ] || { echo "33cz：從 $C 讀不到 Step 2 的模型（33c 沒被 33bz 改成單一模型？）" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }
bash $JOBS/analyze_step2_int4.sh $m > $JOBS/queue/33cz_analyze.out 2>&1 || { echo "analyze_step2_int4 $m 失敗（見 33cz_analyze.out）" | tee -a $JOBS/NEEDS_HUMAN; tail -30 $JOBS/queue/33cz_analyze.out; exit 0; }
python3 - $OUT/analysis/int4/step2_${m}_int4.json $JOBS/NEEDS_HUMAN <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for c in d["cond1"]:
    print(f"[33cz] {c['config']:4s} {c['gpu_state']:4s} 對 R25 {c['diff_vs_R25']:+.4f} CI {c['ci95']} t {c['t_x']:.4f}/{c['t_R25']:.4f} 第 1 條={c['cond1_pass']}")
print(f"[33cz] stable_gap_1_2={d['stable_gap_1_2']} K2={d['K2']}")
if d["stable_gap_1_2"]:
    open(sys.argv[2], "a").write(f"33cz：INT4 階 {d['model']} 穩定差距第 1、2 條成立（{list(d['cond2'])}）→ 03 §2.4 第 3 條要新 seed（20000 起）同樣題數重跑，請寫重跑的 stage\n")
PY
exit 0
