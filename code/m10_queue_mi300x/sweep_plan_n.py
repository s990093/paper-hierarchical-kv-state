"""09 §4.1：用試跑的牆鐘時間決定海選題數，寫 queue/41_sweep_screen.sh。規則（事先寫死）：
n ∈ {30, 24, 18, 12}（每任務），兩個模型合計的牆鐘時間 ≤ 12 小時取最大的；12 都超過 → 先砍 R3、R4，再把佔比改成每 10%。
吞吐＝試跑的題數 ÷ 試跑的牆鐘時間（4 個 process 同時跑；海選也用同樣 4 個 process）。"""
import json, os
J = os.path.dirname(os.path.abspath(__file__)); RAW = "/mlsteam/data/tiara/sweep_m10"
REPO = "/mlsteam/workspace/paper-hierarchical-kv-state"
t0, t1 = (int(open(f"{RAW}/pilot_{x}").read()) for x in ("start", "end"))
wall_h = (t1 - t0) / 3600
pilot_items = 2 * 4 * 2                                    # 2 個模型 × 4 個任務 × 2 題
per_item_h = wall_h / pilot_items
FR17 = [round(0.10 + 0.05 * i, 2) for i in range(17)]
FR9 = [round(0.10 + 0.10 * i, 2) for i in range(9)]
POL = ["F", "M", "B", "R0", "R1", "R2", "R3", "R4", "ERR", "ATT", "AXE", "ORC"]
plan = None
for pols, fracs, label in ((POL, FR17, "全部"), ([p for p in POL if p not in ("R3", "R4")], FR17, "砍 R3、R4"),
                           ([p for p in POL if p not in ("R3", "R4")], FR9, "砍 R3、R4＋佔比每 10%")):
    scale = (len(pols) * len(fracs)) / (len(POL) * len(FR17))     # 每題的前向次數大約和組合數成正比
    for n in (30, 24, 18, 12):
        est = 2 * 4 * n * per_item_h * scale
        if est <= 12:
            plan = dict(n=n, policies=pols, fracs=fracs, label=label, est_h=round(est, 2))
            break
    if plan:
        break
if plan is None:
    plan = dict(n=12, policies=[p for p in POL if p not in ("R3", "R4")], fracs=FR9, label="最小設定仍超過 12 小時（照實記）",
                est_h=round(2 * 4 * 12 * per_item_h * (10 * 9) / (12 * 17), 2))
plan.update(pilot_wall_h=round(wall_h, 3), per_item_h=round(per_item_h, 4))
json.dump(plan, open(f"{REPO}/results/m10_sweep_mi300x/screen_plan.json", "w"), indent=1, ensure_ascii=False)
print("[sweep_plan_n]", plan, flush=True)
G = {"llama": "/mlsteam/data/tiara/runs/20261015-161226-m10-grid-fp8-llama/blocks.csv",
     "qwen": "/mlsteam/data/tiara/runs/20261015-194150-m10-regrid-fp8-qwen/blocks.csv"}
pol = " ".join(plan["policies"]); fr = " ".join(str(x) for x in plan["fracs"])
jobs = []
for m, need in (("llama", 40), ("qwen", 30)):
    for s in range(2):
        jobs.append(dict(name=f"sws-{m}-s{s}", need_gb=need,
                         cmd=f"m7run m10-sweep-screen-{m}-s{s} python {REPO}/code/m10_sweep.py screen --model {m} "
                             f"--params-file {REPO}/results/m10_pilot_mi300x/e0_tune/CANDIDATE_PARAMS.json --n {plan['n']} "
                             f"--shard {s}/2 --policies {pol} --fracs {fr} --parts {REPO}/results/m10_sweep_mi300x/parts/parts_{m}.json "
                             f"--grid-blocks {G[m]} --out {RAW}/screen_{m}_s{s}.csv.gz"))
json.dump(jobs, open(f"{J}/jobs_41.json", "w"), indent=1)
open(f"{J}/queue/41_sweep_screen.sh", "w").write(f"""# 大掃描海選（09 §4.1）；題數與組合由 sweep_plan_n.py 照試跑決定：{plan}
source {J}/lib.sh
set -uo pipefail
pool {J}/jobs_41.json 4 || exit 1
""")
print("[sweep_plan_n] 寫了 queue/41_sweep_screen.sh", flush=True)
