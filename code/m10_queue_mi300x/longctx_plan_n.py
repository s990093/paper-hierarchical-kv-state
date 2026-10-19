"""10_longctx_plan §3.3：長 context 的試跑工作與題數規則（事先寫死）。
  jobs-pilot：寫 jobs_91.json（每個模型 × 長度 1 個 process，每任務 1 題，正式的設定）。
  plan      ：讀試跑每題的時間，照規則決定 n 與設定，寫 jobs_92.json、queue/44c_longctx_screen.sh、results/m10_longctx_mi300x/screen_plan.json。
規則：n ∈ {12, 9, 6, 3}（每任務每長度），估計牆鐘＝Σ_{模型,長度} 4 個任務 × n × 試跑單題時間 × 縮放 ÷ 3（同時 3 個 process）≤ 14 小時，
取最大的 n。都超過 → 依序：拿掉讀法 B → 佔比只留 25%、50%、90% → 拿掉 ERR、AXE。縮放＝（放法數 × 佔比數 × 讀法數）÷（9 × 5 × 3），
和 09 §4.1 一樣是「前向次數大約和組合數成正比」的近似（prefill、生成是固定成本，所以這個近似偏保守）。
"""
import glob
import json
import os
import re
import statistics
import sys

J = os.environ.get("LONGCTX_JOBS", os.path.dirname(os.path.abspath(__file__)))   # 測試時改到別的目錄，不碰真的佇列
REPO = "/mlsteam/workspace/paper-hierarchical-kv-state"
OUT = f"{REPO}/results/m10_precision_mi300x"
LC = os.environ.get("LONGCTX_LC", f"{REPO}/results/m10_longctx_mi300x")
SW = f"{REPO}/results/m10_sweep_mi300x"
RAW = "/mlsteam/data/tiara/longctx_m10"
RUNS = os.environ.get("TIARA_RUNS", "/mlsteam/data/tiara/runs")
P = f"{REPO}/results/m10_pilot_mi300x/e0_tune/CANDIDATE_PARAMS.json"
NBS = (64, 128, 252)
TASKS = ["T2", "T3", "T4", "T5"]
POL = ["F", "M", "B", "R0", "R1", "ERR", "ATT", "AXE", "ORC"]
FR = [0.1, 0.25, 0.5, 0.75, 0.9]
# 10/16 13:1xZ 照實測峰值改（單一 process 的 GPU 記憶體峰值，m10_sweep screen 印的 "peak"）：
#   64 個 block：Qwen 20.1、Llama 28.4 GiB（stage 39，run 20261016-123319-…-screen-qwen、-123519-…-screen-llama）
#   252 個 block：Qwen 35.4、Llama 63.1 GiB（stage 39a，run 20261016-125625-m10-lc-smoke-screen-qwen、-125821-…-llama）
#   128 個 block 沒量，用 64 與 252 之間的直線推（Qwen 25.3、Llama 40.2）。need＝峰值 × 1.2 ＋ 2，64 個 block 保留原值（已經比這個大）。
NEED = {("llama", 64): 40, ("llama", 128): 50, ("llama", 252): 78, ("qwen", 64): 30, ("qwen", 128): 32, ("qwen", 252): 45}
SHARDS = {64: 1, 128: 2, 252: 3}
G32 = {"llama": "/mlsteam/data/tiara/runs/20261015-161226-m10-grid-fp8-llama/blocks.csv",
       "qwen": "/mlsteam/data/tiara/runs/20261015-194150-m10-regrid-fp8-qwen/blocks.csv"}


def fmt_of(m, tier):
    return json.load(open(f"{OUT}/fmt_select/fmt_choice.json"))["choice"][m][tier]


def lastrun(name):
    ds = sorted(glob.glob(f"{RUNS}/*-{name}"))
    return ds[-1] if ds else None


def grid_blocks(m, nb):
    if nb == 64:
        return G32[m]
    d = lastrun(f"m10-lc-grid-{m}-{nb * 512}-{fmt_of(m, 'fp8')}")
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
            c, has_b = cmd(m, nb, 1, POL, FR, True, "0/1", f"{RAW}/pilot_{m}_{nb}.csv.gz", f"m10-lc-pilot-{m}-{nb}")
            if not has_b:
                missing_b.append((m, nb))
            jobs.append(dict(name=f"lcp-{m}-{nb}", need_gb=NEED[(m, nb)], cmd=c))
    json.dump(jobs, open(f"{J}/jobs_91.json", "w"), indent=1)
    print("[longctx] 試跑工作", len(jobs), "；沒有讀法 B 的（缺 parts 或該長度的格子）：", missing_b or "無", flush=True)


def per_item_seconds(m, nb):
    d = lastrun(f"m10-lc-pilot-{m}-{nb}")
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
    choice.update(pilot_seconds_per_item={f"{m}|{nb}": round(v, 1) for (m, nb), v in dt.items()})
    os.makedirs(LC, exist_ok=True)
    json.dump(choice, open(f"{LC}/screen_plan.json", "w"), indent=1, ensure_ascii=False)
    print("[longctx plan]", choice, flush=True)
    jobs = []
    for nb in (252, 128, 64):                                  # 最重的先排，pool 依剩餘 VRAM 開
        for m in ("llama", "qwen"):
            k = SHARDS[nb]
            for s in range(k):
                c, _ = cmd(m, nb, choice["n"], choice["policies"], choice["fracs"], choice["readout_b"], f"{s}/{k}",
                           f"{RAW}/screen_{m}_{nb}_s{s}.csv.gz", f"m10-lc-screen-{m}-{nb}-s{s}")
                jobs.append(dict(name=f"lcs-{m}-{nb}-s{s}", need_gb=NEED[(m, nb)], cmd=c))
    json.dump(jobs, open(f"{J}/jobs_92.json", "w"), indent=1)
    open(f"{J}/queue/44c_longctx_screen.sh", "w").write(f"""# GPU-NEED: one　（10/19 雙卡：只量品質的 pool，只用 main 那張卡）
# 10_longctx_plan §3.3 正式海選；題數與設定由 longctx_plan_n.py 照試跑決定：{choice['label']}，n＝{choice['n']}，估計 {choice['est_h']} 小時
source {J}/lib.sh
set -uo pipefail
pool {J}/jobs_92.json 4 || {{ echo "44c：pool jobs_92.json 有 job 失敗（{J}/pool_lcs-*.log）→ 跳過彙整" >> {J}/NEEDS_HUMAN; exit 0; }}
# 彙整（CPU）；出錯不擋（寫 NEEDS_HUMAN）
if [ -f {REPO}/code/m10_longctx_analyze.py ]; then
  python {REPO}/code/m10_longctx_analyze.py --raw {RAW} --out-dir {LC}/analysis > {J}/queue/44c_analyze.out 2>&1 \\
    || echo "longctx analyze 失敗（queue/44c_analyze.out）" >> {J}/NEEDS_HUMAN
else
  echo "m10_longctx_analyze.py 還沒寫" >> {J}/NEEDS_HUMAN
fi
""")
    print("[longctx plan] 寫了 queue/44c_longctx_screen.sh", flush=True)


if __name__ == "__main__":
    {"jobs-pilot": jobs_pilot, "plan": plan}[sys.argv[1]]()
