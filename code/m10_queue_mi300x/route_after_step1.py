"""Step 1 判定出來後，只調「後面 GPU 工作的順序」（不動判準、不動任何數字）。stage 31／32z 呼叫。

判定一律由凍結的 m10_analyze.py 算（analyze_step1.sh、analyze_2b.sh）；這支只讀 decide_fp8.json／k3_<m>.json：
  - 檢定力擋判定（power_blocks）→ 排 07 §9 寫死的加題（同題數、seed 20000 起、只加一次），加完再判一次（32z）。
  - 兩個模型都過 → 往 INT4：80／85／86 搬到 35 前面（33a／33b／33c）。
  - 有模型沒過 → 在 decide 指定的模型上判 K3（Step 2b，資料已在 Step 1 那批）；K3 不成立 → stage 35 先跑那個模型（f8m_first.txt）。
  - 其他（資料不全、程式出錯、加題後仍擋）→ 寫 NEEDS_HUMAN，不動佇列（35 照預設順序跑，GPU 不空著）。
用法：route_after_step1.py [--after-addon]
"""
import json
import os
import subprocess
import sys

J = os.path.dirname(os.path.abspath(__file__))
Q = os.path.join(J, "queue")
REPO = "/mlsteam/workspace/paper-hierarchical-kv-state"
OUT = os.environ.get("M10_OUT", f"{REPO}/results/m10_precision_mi300x")
AFTER_ADDON = "--after-addon" in sys.argv


def log(x):
    print(f"[route] {x}", flush=True)


def human(why):
    open(os.path.join(J, "NEEDS_HUMAN"), "a").write(why + "\n")
    log("NEEDS_HUMAN: " + why)
    sys.exit(0)


def mv(a, b):
    src, dst = os.path.join(Q, a + ".sh"), os.path.join(Q, b + ".sh")
    if os.path.exists(src):
        os.rename(src, dst)
        log(f"queue {a} → {b}")
    else:
        log(f"queue {a}.sh 不在（已跑過或已搬）")


ADDON = r'''# 07 §9 寫死的加題：同樣題數（T2–T4 各 180、T5 60）、新 seed（20000 起），和原題合併重判；只加一次。stage 31 自動排的。
source {J}/lib.sh
set -uo pipefail
[ -n "$(git -C $REPO log --oneline -1 -- docs/precision_20261013/07_freeze_mi300x.md)" ] || {{ echo "事先登記還沒凍結"; exit 1; }}
python3 - <<PY
import json
def fmt(m,t): return json.load(open("$OUT/fmt_select/fmt_choice.json"))["choice"][m][t]
per={{}}
for m,need in {models}:
    G=fmt(m,"fp8"); g=G[4:]
    C=f"E0 E1 E1s E1sr E1s@fp8k0_{{g}} E1s@int8_token"
    base=f"python $REPO/code/m10_quality.py run --model {{m}} --params-file $P --fmt {{G}} --configs {{C}} --modes A B --bws b32 --meets $OUT/speed_{{m}}/meets.json --seed0 20000"
    js=[dict(name=f"s1a-{{m}}-T24-s{{s}}", need_gb=need, cmd=f"m7run m10-s1a-{{m}}-T24-s{{s}} {{base}} --tasks T2 T3 T4 --n 180 --shard {{s}}/6 --out $OUT/quality/step1_{{m}}_addon_T24_s{{s}}.csv") for s in range(6)]
    js.append(dict(name=f"s1a-{{m}}-T5", need_gb=need, cmd=f"m7run m10-s1a-{{m}}-T5 {{base}} --tasks T5 --n 60 --out $OUT/quality/step1_{{m}}_addon_T5.csv"))
    per[m]=js
jobs=[j for js in per.values() for j in js]
json.dump(jobs,open("jobs_32.json","w"),indent=1)
PY
pool jobs_32.json 8 || exit 1
'''

GATE2 = '''# 加題跑完後再判一次（只加一次；仍擋 → NEEDS_HUMAN）。stage 31 自動排的。
source {J}/lib.sh
set -uo pipefail
bash $JOBS/analyze_step1.sh > $JOBS/queue/32z_analyze.out 2>&1 || {{ echo "analyze_step1 失敗（見 32z_analyze.out）" | tee -a $JOBS/NEEDS_HUMAN; tail -30 $JOBS/queue/32z_analyze.out; exit 0; }}
python3 $JOBS/route_after_step1.py --after-addon
'''


def main():
    d = json.load(open(f"{OUT}/analysis/fp8/decide_fp8.json"))
    log("next = " + d["next"])
    log("next_frozen_code = " + d["next_frozen_code"])
    blk = [m for m in ("llama", "qwen") if d[m].get("power_blocks")]
    if blk:
        if AFTER_ADDON:
            human(f"加題之後仍然檢定力不夠：{blk}（07 §9：只加一次）→ 要人判斷")
        need = {"llama": 42, "qwen": 28}
        models = tuple((m, need[m]) for m in blk)
        open(os.path.join(Q, "32_quality_step1_addon.sh"), "w").write(ADDON.format(J=J, models=repr(models)))
        open(os.path.join(Q, "32z_decide_step1_after_addon.sh"), "w").write(GATE2.format(J=J))
        log(f"排了加題 32（{blk}）與再判 32z")
        return
    L, Qp = d["llama"]["model_pass"], d["qwen"]["model_pass"]
    if L is None or Qp is None:
        human("decide 說資料不全：" + d["next"])
    if L and Qp:
        mv("80_grid_int4", "33a_grid_int4")
        mv("85_quality_int4_endpoints", "33b_quality_int4_endpoints")
        mv("86_quality_int4_mixed", "33c_quality_int4_mixed")
        mv("35_quality_fp8_mixed", "45_quality_fp8_mixed")
        log("兩個模型都過 → INT4 的格子、端點、混合配置排到 35 前面；FP8 混合配置變成只報告 → 排到大掃描（40–42）後面")
        return
    m = "qwen" if (L and not Qp) else "llama"      # 照 m10_analyze.cmd_decide 的分支（兩個都沒過 → Llama）
    log(f"Step 2b（K3）在 {m} 上判")
    r = subprocess.run(["bash", os.path.join(J, "analyze_2b.sh"), m], capture_output=True, text=True)
    open(os.path.join(Q, f"31_analyze_2b_{m}.out"), "w").write(r.stdout + r.stderr)
    if r.returncode != 0:
        human(f"analyze_2b.sh {m} 失敗（rc={r.returncode}，見 queue/31_analyze_2b_{m}.out）")
    k3 = json.load(open(f"{OUT}/analysis/fp8/k3_{m}.json")).get("K3")
    log(f"K3({m}) = {k3}")
    if k3 is None:
        human(f"K3({m}) 判不出來（None）")
    if k3:
        mv("35_quality_fp8_mixed", "45_quality_fp8_mixed")
        log("K3 成立 → 流程在 FP8 階停；FP8 混合配置變成只報告 → 排到大掃描（40–42）後面（使用者 10/15：主線關鍵段先、掃描在只報告的工作前）")
        return
    open(os.path.join(J, "f8m_first.txt"), "w").write(m + "\n")
    log(f"K3 不成立 → Step 2 在 {m} 上做：stage 35 先跑 {m}（f8m_first.txt）")


if __name__ == "__main__":
    main()
