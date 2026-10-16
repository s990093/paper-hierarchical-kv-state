"""h200_longctx_plan.py — 10_longctx_plan §3.3 的試跑工作與題數規則，H200 版。

照抄 code/m10_queue_mi300x/longctx_plan_n.py（MI300X 的 44b／44c），規則一字不改；只換：
  - 路徑（results/*_h200、$TIARA_RUNS、原始 CSV 放 $LONGCTX_RAW）；
  - run 名稱前綴 m10-lc-* → m10h-lc-*；
  - 32K 的讀法 B 參考線：H200 主線 FP8 格子（job m10h-speed 的 m10h-grid-fp8-<m>）的 blocks.csv；
  - 不寫佇列腳本：plan 寫 jobs_screen.json，由 code/r_h200_lc.sh 用 h200_pool.py 跑。
NEED（GPU 記憶體）沿用 MI300X 39a 的實測峰值換算；H200 的峰值照實記在每個 run 的 stdout（m10_sweep screen 印 "peak"）。

  python code/h200_longctx_plan.py jobs-pilot   → $LONGCTX_JOBS/jobs_pilot.json（每個模型 × 長度 1 個 process，每任務 1 題）
  python code/h200_longctx_plan.py plan         → $LONGCTX_JOBS/jobs_screen.json、results/m10_longctx_h200/screen_plan.json

規則（同 MI300X）：n ∈ {12, 9, 6, 3}（每任務每長度），估計牆鐘＝Σ_{模型,長度} 4 個任務 × n × 試跑單題時間 × 縮放 ÷ 3 ≤ 14 小時，
取最大的 n。都超過 → 依序：拿掉讀法 B → 佔比只留 25%、50%、90% → 拿掉 ERR、AXE。縮放＝（放法數 × 佔比數 × 讀法數）÷（9 × 5 × 3）。
正式工作的順序：252 → 128 → 64 個 block（最重的先；使用者 10/17「先跑 128」也是這個順序）。
"""
import glob
import json
import os
import re
import statistics
import sys

REPO = os.environ.get("REPO") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = os.environ.get("LONGCTX_JOBS") or sys.exit("要設 LONGCTX_JOBS（jobs_*.json 放哪裡）")
OUT = os.path.join(REPO, "results", "m10_precision_h200")
LC = os.environ.get("LONGCTX_LC", os.path.join(REPO, "results", "m10_longctx_h200"))
SW = os.path.join(REPO, "results", "m10_sweep_h200")
RAW = os.environ.get("LONGCTX_RAW") or sys.exit("要設 LONGCTX_RAW（逐列 CSV 放哪裡，不進 git）")
RUNS = os.environ.get("TIARA_RUNS") or sys.exit("要設 TIARA_RUNS")
P = os.path.join(REPO, "results", "m10_pilot_mi300x", "e0_tune", "CANDIDATE_PARAMS.json")
NBS = (64, 128, 252)
TASKS = ["T2", "T3", "T4", "T5"]
POL = ["F", "M", "B", "R0", "R1", "ERR", "ATT", "AXE", "ORC"]
FR = [0.1, 0.25, 0.5, 0.75, 0.9]
NEED = {("llama", 64): 40, ("llama", 128): 50, ("llama", 252): 78, ("qwen", 64): 30, ("qwen", 128): 32, ("qwen", 252): 45}
SHARDS = {64: 1, 128: 2, 252: 3}


def fmt_of(m, tier):
    return json.load(open(f"{OUT}/fmt_select/fmt_choice.json"))["choice"][m][tier]


def lastrun(name):
    ds = sorted(glob.glob(f"{RUNS}/*-{name}"))
    return ds[-1] if ds else None


def grid_blocks(m, nb):
    d = lastrun(f"m10h-grid-fp8-{m}") if nb == 64 else lastrun(f"m10h-lc-grid-{m}-{nb * 512}-{fmt_of(m, 'fp8')}")
    return os.path.join(d, "blocks.csv") if d and os.path.exists(os.path.join(d, "blocks.csv")) else None


def cmd(m, nb, n, pols, fracs, use_b, shard, out, run_name):
    G = fmt_of(m, "fp8")
    c = (f"m7run {run_name} python {REPO}/code/m10_sweep.py screen --model {m} --params-file {P} --n-blocks {nb} "
         f"--tasks {' '.join(TASKS)} --n {n} --shard {shard} --policies {' '.join(pols)} --fracs {' '.join(map(str, fracs))} "
         f"--gen-all --gen-pos {G} int8_token int4_kivi --out {out}")
    gb, parts = grid_blocks(m, nb), f"{SW}/parts/parts_{m}.json"
    if use_b and gb and os.path.exists(parts):
        c += f" --parts {parts} --grid-blocks {gb}"
    return c, bool(use_b and gb and os.path.exists(parts))


def jobs_pilot():
    jobs, missing_b = [], []
    for m in ("llama", "qwen"):
        for nb in NBS:
            c, has_b = cmd(m, nb, 1, POL, FR, True, "0/1", f"{RAW}/pilot_{m}_{nb}.csv.gz", f"m10h-lc-pilot-{m}-{nb}")
            if not has_b:
                missing_b.append((m, nb))
            jobs.append(dict(name=f"lcp-{m}-{nb}", need_gb=NEED[(m, nb)], cmd=c))
    json.dump(jobs, open(f"{J}/jobs_pilot.json", "w"), indent=1)
    print("[longctx] 試跑工作", len(jobs), "；沒有讀法 B 的（缺 parts 或該長度的格子）：", missing_b or "無", flush=True)
    if missing_b:          # MI300X 版只印；H200 版停下：試跑沒有讀法 B，單題時間會偏短，題數規則就算錯（規則 2）
        sys.exit("試跑缺讀法 B 的輸入 → 停（先跑 r_h200_lc.sh --part speed）")


def per_item_seconds(m, nb):
    d = lastrun(f"m10h-lc-pilot-{m}-{nb}")
    xs = [float(x) for x in re.findall(r"n_blocks=\d+ .*? (\d+\.\d)s \(", open(os.path.join(d, "stdout.log")).read())] if d else []
    return statistics.mean(xs) if xs else None


def plan():
    dt = {(m, nb): per_item_seconds(m, nb) for m in ("llama", "qwen") for nb in NBS}
    if any(v is None for v in dt.values()):
        raise SystemExit(f"試跑的單題時間讀不到：{dt}")
    base = sum(4 * v for v in dt.values()) / 3 / 3600         # n＝1、完整設定的牆鐘（小時）
    levels = [("完整", POL, FR, True),
              ("拿掉讀法 B", POL, FR, False),
              ("拿掉讀法 B＋佔比 25/50/90%", POL, [0.25, 0.5, 0.9], False),
              ("拿掉讀法 B＋佔比 25/50/90%＋拿掉 ERR、AXE", [p for p in POL if p not in ("ERR", "AXE")], [0.25, 0.5, 0.9], False)]
    choice = None
    for label, pols, fracs, use_b in levels:
        scale = len(pols) * len(fracs) * (3 if use_b else 1) / (len(POL) * len(FR) * 3)
        for n in (12, 9, 6, 3):
            est = base * n * scale
            if est <= 14:
                choice = dict(label=label, n=n, policies=pols, fracs=fracs, readout_b=use_b, est_h=round(est, 2))
                break
        if choice:
            break
    if choice is None:
        label, pols, fracs, use_b = levels[-1]
        scale = len(pols) * len(fracs) / (len(POL) * len(FR) * 3)
        choice = dict(label=label + "（仍超過 14 小時，照實記）", n=3, policies=pols, fracs=fracs, readout_b=False,
                      est_h=round(base * 3 * scale, 2))
    choice.update(pilot_seconds_per_item={f"{m}|{nb}": round(v, 1) for (m, nb), v in dt.items()},
                  platform="H200", rule="10_longctx_plan §3.3（MI300X 的 longctx_plan_n.py 同一條）")
    os.makedirs(LC, exist_ok=True)
    json.dump(choice, open(f"{LC}/screen_plan.json", "w"), indent=1, ensure_ascii=False)
    print("[longctx plan]", choice, flush=True)
    jobs = []
    for nb in (252, 128, 64):
        for m in ("llama", "qwen"):
            k = SHARDS[nb]
            for s in range(k):
                c, _ = cmd(m, nb, choice["n"], choice["policies"], choice["fracs"], choice["readout_b"], f"{s}/{k}",
                           f"{RAW}/screen_{m}_{nb}_s{s}.csv.gz", f"m10h-lc-screen-{m}-{nb}-s{s}")
                jobs.append(dict(name=f"lcs-{m}-{nb}-s{s}", need_gb=NEED[(m, nb)], cmd=c))
    json.dump(jobs, open(f"{J}/jobs_screen.json", "w"), indent=1)
    print(f"[longctx plan] 寫了 {J}/jobs_screen.json（{len(jobs)} 個 process）", flush=True)


if __name__ == "__main__":
    {"jobs-pilot": jobs_pilot, "plan": plan}[sys.argv[1]]()
