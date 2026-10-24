"""m10_confirm.py — 大掃描複賽（docs/precision_20261013/09_sweep_plan.md §4.3；規則 10/15 22:50 UTC 寫死）

子命令：
  plan     讀海選選出的 results/m10_sweep_mi300x/analysis/finalists_<m>.json → 每個模型的配置清單
           （選手、前緣點、同格的 R0–R4 與 ALL、E0）→ 寫 confirm_plan.json、jobs_43q.json、queue/43_sweep_confirm.sh
  speed    一個模型、一個格式：主線 m10_speed 的格子（4 頻寬 × 閒／忙，3 launch × 10 rep），只把放法換成大掃描的定義
           （SW-<放法>-<佔比%>；ERR／ATT／AXE 用這次合成 token 的 prefill 算 err、att；R 用 seed＝launch×1000＋rep）。
           會合點寫進 <out-dir>/meets.json（鍵＝"SW-ERR-35@int2h_g32"、"E1s@<格式>"、"E0"），m10_quality 讀法 B 用。
  smoke    真模型冒煙：每個模型 1 題（T2），全部配置、讀法 A＋B，檢查跑得完、低精度集合的大小對（43 開頭跑）
  _sworker 子 process（patch m10_speed 的 layout／Bench 後呼叫 m10_speed.worker）

主線的程式一行都沒改行為：m10_quality 只多了「SW- 開頭的配置」與「M10_CODEC_X=1 時裝新格式」兩個入口（預設不會走到）。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.dirname(HERE)
SW_DIR = os.environ.get("M10_SWEEP_DIR") or os.path.join(REPO, "results", "m10_sweep_mi300x")   # （H200：環境變數，沒設時不變；2026-10-17）
CTX = {}
NEED_GB = {"llama": (28, 4.5), "qwen": (22, 2.0)}   # 品質 process：基本＋每個格式（每個格式在 ItemState 留 64 個解碼後的 block）


def sw_name(policy: str, frac: float) -> str:
    return f"SW-{policy}-{int(round(frac * 100))}"


def configs_from_finalists(fin: dict) -> list[str]:
    """09 §4.2 第 2–4 條：選手＋前緣點＋每格的對照組（R0–R4 五個 seed、ALL）；ALL＝block 1 以後全部低精度＝主線 E1s。"""
    out = []

    def add(s):
        if s not in out:
            out.append(s)
    for f in fin["finalists"]:
        add(f"{sw_name(f['policy'], f['frac'])}@{f['fmt']}")
    for f in fin["frontier"]:
        add(f"E1s@{f['fmt']}" if f["policy"] == "ALL" else f"{sw_name(f['policy'], f['frac'])}@{f['fmt']}")
    for c in fin["controls"]:
        if float(c["frac"]) < 1.0:
            for p in c["policies"]:
                add(f"{sw_name(p, float(c['frac']))}@{c['fmt']}")
        add(f"E1s@{c['fmt']}")
    return out


# ====================================================================== speed（子 process 裡 patch m10_speed）
def _sworker(spec: dict):
    import types
    import m10_speed as S
    import m10_sweep as SW
    from m10_codec import rel_err_centered

    class CBench(S.Bench):
        def capture(self, n):
            super().capture(n)
            from m10_codec import decode
            km, torch = self.km, self.torch
            err = []
            for i in range(n):
                x = km.chunk_view(i)
                y = decode(self.lp[i].to(km.device))
                err.append(rel_err_centered(x, y)["K"])
            CTX["err"] = err
            if spec.get("need_att"):
                ans = types.SimpleNamespace(km=km, CHUNK=self.CHUNK)
                bf = [km.chunk_view(i).clone() for i in range(n)]
                shim = types.SimpleNamespace(ans=ans, ids=self.ids[:n * self.CHUNK], bf16=bf, src=["bf16"] * n)
                CTX["att"] = SW.Item.attention_mass(shim)
                torch.cuda.synchronize()
            else:
                CTX["att"] = None

    orig_layout = S.layout

    def layout(config, n, seed):
        if not config.startswith("SW-"):
            return orig_layout(config, n, seed)
        _, policy, pct = config.split("-")
        assert n == SW.N_BLOCKS
        s = SW.lp_set(policy, SW.k_of(int(pct) / 100), {"seed": seed, "needle": [], "err": CTX["err"], "att": CTX["att"]})
        return [i in s for i in range(n)]

    S.Bench, S.layout = CBench, layout
    S.worker(spec)


def cmd_speed(a):
    import argparse as ap
    import m10_speed as S
    os.makedirs(a.out_dir, exist_ok=True)
    sa = ap.Namespace(bw_json=a.bw_json, bw_names=a.bw_names, gpu_states=a.gpu_states, busy_levels=a.busy_levels,
                      busy_key=a.busy_key)
    need_att = any(c.split("-")[1] in ("ATT", "AXE") for c in a.configs if c.startswith("SW-"))
    spec = dict(model=a.model, fmt=a.fmt, new_tokens=a.new_tokens, lengths=[a.main_length], main_length=a.main_length,
                configs=a.configs, refs=[], tiers=S.load_tiers(sa), gpu_states=S.load_states(sa), reps=a.reps,
                out_dir=a.out_dir, blocks_out=a.blocks_out, verify=True, need_att=need_att)
    json.dump({**spec, "argv": sys.argv, "run_id": S.run_id()},
              open(os.path.join(a.out_dir, f"spec_confirm_{a.fmt}.json"), "w"), indent=1, ensure_ascii=False)
    for launch in range(a.launches):
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "_sworker", json.dumps(dict(spec, launch=launch))])
        if r.returncode != 0:
            raise SystemExit(f"launch {launch} 失敗（exit {r.returncode}）；照規則 2 停下來")
    rows = [r for r in S._rows(os.path.join(a.out_dir, "restores.csv")) if r["run_id"] == S.run_id()]
    S.write_meets(ap.Namespace(out_dir=a.out_dir, model=a.model, main_length=a.main_length, base_fmt="__none__"), rows)
    S.cmd_summary(ap.Namespace(out_dir=a.out_dir, only_run=S.run_id()))
    os.replace(os.path.join(a.out_dir, "summary.csv"), os.path.join(a.out_dir, f"summary_{a.fmt}.csv"))


# ====================================================================== plan：寫 stage 43
def cmd_plan(a):
    global SW_DIR
    SW_DIR = a.sw_dir
    J = a.jobs
    plan, qjobs = {}, []
    for m in ("llama", "qwen"):
        p = os.path.join(SW_DIR, "analysis", f"finalists_{m}.json")
        fin = json.load(open(p))
        cfgs = configs_from_finalists(fin)
        by_fmt = {}
        for c in cfgs:
            name, _, fmt = c.partition("@")
            by_fmt.setdefault(fmt, []).append(name)
        plan[m] = dict(finalists_json=p, n_finalists=len(fin["finalists"]), n_frontier=len(fin["frontier"]),
                       configs=cfgs, speed_grids=by_fmt)
        allc = " ".join(["E0"] + cfgs)
        base = (f"python {REPO}/code/m10_quality.py run --model {m} --params-file {a.params_file} --configs {allc} --modes A B "
                f"--bws b32 --meets {SW_DIR}/confirm/speed_{m}/meets.json --seed0 50000")
        b0, per = NEED_GB[m]
        need = int(b0 + per * len(by_fmt))
        for s in range(a.shards):
            qjobs.append(dict(name=f"cf-{m}-T24-s{s}", need_gb=need,
                              cmd=f"M10_CODEC_X=1 m7run m10-confirm-{m}-T24-s{s} {base} --tasks T2 T3 T4 --n 90 "
                                  f"--shard {s}/{a.shards} --out {SW_DIR}/confirm/quality_{m}_T24_s{s}.csv"))
        qjobs.append(dict(name=f"cf-{m}-T5", need_gb=need,
                          cmd=f"M10_CODEC_X=1 m7run m10-confirm-{m}-T5 {base} --tasks T5 --n 30 "
                              f"--out {SW_DIR}/confirm/quality_{m}_T5.csv"))
    json.dump(dict(rule="09 §4.3（2026-10-15 22:50 UTC 寫死）", seed0=50000, n={"T2-T4": 90, "T5": 30}, models=plan),
              open(os.path.join(SW_DIR, "confirm_plan.json"), "w"), indent=1, ensure_ascii=False)
    json.dump(qjobs, open(os.path.join(J, "jobs_43q.json"), "w"), indent=1)
    L = [f"# 09 §4.3 大掃描複賽（由 m10_confirm.py plan 照海選的 finalists_<m>.json 寫出；{len(qjobs)} 個品質 process）",
         f"source {J}/lib.sh", "set -uo pipefail", f"C={SW_DIR}/confirm; mkdir -p $C",
         "# 0. 冒煙：每個模型 1 題、全部配置（失敗 → NEEDS_HUMAN，跳過 43 的其餘部分，不擋後面）",
         "for m in llama qwen; do",
         f"  M10_CODEC_X=1 m7run m10-confirm-smoke-$m python {REPO}/code/m10_confirm.py smoke --model $m > $C/smoke_$m.log 2>&1 \\",
         '    || { echo "43 confirm smoke $m 失敗（$C/smoke_$m.log）→ 跳過 43" | tee -a $JOBS/NEEDS_HUMAN; tail -20 $C/smoke_$m.log; exit 0; }',
         "done",
         "# 0b. 速度路徑的冒煙（真模型：每個模型第一個格式的全部配置，含 ATT／AXE；CPU 層、閒、1 launch × 1 rep）"]
    for m, pm in plan.items():
        fmt, names = next(iter(pm["speed_grids"].items()))
        L.append(f"M10_CODEC_X=1 m7run m10-confirm-ssmoke-{m} bash -c \"python {REPO}/code/m10_confirm.py speed --model {m} "
                 f"--fmt {fmt} --configs E0 {' '.join(names)} --out-dir $C/smoke_speed_{m} --bw-json $OUT/speed_{m}/bw.json "
                 f"--bw-names cpu --gpu-states idle --launches 1 --reps 1 --blocks-out $C/smoke_speed_{m}/blocks.csv\" "
                 f"> $C/smoke_speed_{m}.log 2>&1 || {{ echo \"43 confirm speed smoke {m} 失敗（$C/smoke_speed_{m}.log）→ 跳過 43\" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }}")
    L.append("# 1. 速度格子（每個模型 × 格式；會合點給讀法 B、速度判準用）")
    for m, pm in plan.items():
        first = True
        for fmt, names in pm["speed_grids"].items():
            cl = " ".join((["E0"] if first else []) + names)
            first = False
            L.append(f"M10_CODEC_X=1 m7run m10-confirm-grid-{m}-{fmt} bash -c \"python {REPO}/code/m10_confirm.py speed "
                     f"--model {m} --fmt {fmt} --configs {cl} --out-dir $C/speed_{m} --bw-json $OUT/speed_{m}/bw.json "
                     f"--blocks-out \\$TIARA_RUNS/\\$RUN_ID/blocks.csv\" || {{ echo \"43 confirm grid {m} {fmt} 失敗\" | tee -a $JOBS/NEEDS_HUMAN; exit 0; }}")
    L += ["# 2. 品質（新 seed 50000 起；T2–T4 各 90、T5 30；讀法 A＋B（b32 閒／忙，實測會合點））",
          f"pool {J}/jobs_43q.json {a.pool} || {{ echo \"43 品質 pool 有 job 失敗\" >> $JOBS/NEEDS_HUMAN; exit 0; }}",
          "# 3. 判定（CPU；m10_confirm_analyze.py 在 43 跑完前寫好；沒有就寫 NEEDS_HUMAN）",
          f"if [ -f {REPO}/code/m10_confirm_analyze.py ]; then",
          f"  python {REPO}/code/m10_confirm_analyze.py --dir $C > $C/analyze.out 2>&1 || echo \"43 confirm analyze 失敗\" >> $JOBS/NEEDS_HUMAN",
          "else", '  echo "m10_confirm_analyze.py 還沒寫（43 的資料齊了）" >> $JOBS/NEEDS_HUMAN', "fi", "exit 0"]
    out = os.path.join(J, "queue", "43_sweep_confirm.sh")
    open(out, "w").write("\n".join(L) + "\n")
    print(json.dumps(plan, ensure_ascii=False, indent=1))
    print("寫了", out)


# ====================================================================== smoke
def cmd_smoke(a):
    """真模型、1 題 T2（seed 50000）、全部配置、讀法 A＋B（會合點用假的：每個配置 b32 閒 20、忙 10）。
    檢查：跑得完；每個 SW- 配置的低精度集合大小＝k_of(佔比)、不含 block 0；ATT／AXE 有算 attention。"""
    import m10_quality as Q
    import m10_sweep as SW
    import m10_tasks as T
    if os.environ.get("M10_CODEC_X") == "1":
        import m10_codec_x
        m10_codec_x.install()
    plan = json.load(open(os.path.join(SW_DIR, "confirm_plan.json")))["models"][a.model]
    cfgs = [Q.parse_config(c, "fp8_head") for c in ["E0"] + plan["configs"]]
    ans = Q.Answerer(a.model)
    P = json.load(open(a.params_file))["params"][a.model]
    task, b1, b2, seed = next(iter(Q.plan_items(["T2"], 1, 50000)))
    it = T.make_item(task, b1, b2, seed, ans.tok, P.get(task))
    st = Q.ItemState(ans, it)
    meets = {spec: {"b32": {"idle": 20, "busy": 10}} for spec, _, _ in cfgs}
    rows = Q.item_rows(st, cfgs, ["A", "B"], meets, cache={})
    bad = []
    for spec, name, fmt in cfgs:
        if not name.startswith("SW-"):
            continue
        _, policy, pct = name.split("-")
        s = Q.sweep_placement(st, name, fmt)
        k = SW.k_of(int(pct) / 100)
        if len(s) != k or 0 in s:
            bad.append((spec, len(s), k))
    have = {r["config"] for r in rows}
    miss = [c for c, _, _ in cfgs if c not in have]
    print(f"smoke {a.model}：{len(rows)} 列、{len(have)}／{len(cfgs)} 個配置有列；集合大小不對 {bad}；沒有列 {miss}")
    if bad or miss:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("_sworker")
    w.add_argument("spec")
    p = sub.add_parser("plan")
    p.add_argument("--jobs", default="/tmp/claude-0/-mlsteam-workspace/31610192-64b7-4462-bdf0-b22684a43855/scratchpad/jobs")
    p.add_argument("--shards", type=int, default=5)
    p.add_argument("--pool", type=int, default=8)
    p.add_argument("--sw-dir", default=SW_DIR, help="測試時改到沙盒（讀 analysis/finalists_<m>.json、寫 confirm_plan.json）")
    p.add_argument("--params-file", default=os.path.join(REPO, "results", "m10_pilot_mi300x", "e0_tune", "CANDIDATE_PARAMS.json"))  # 10/17：原本寫成字面 $P（沒人設）→ 43aa 品質第一個 job 參數空掉
    s = sub.add_parser("speed")
    s.add_argument("--model", choices=["llama", "qwen"], required=True)
    s.add_argument("--fmt", required=True)
    s.add_argument("--configs", nargs="+", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--bw-json", required=True)
    s.add_argument("--bw-names", nargs="+", default=["b16", "b32", "b48", "cpu"])
    s.add_argument("--gpu-states", nargs="+", default=["idle", "busy"])
    s.add_argument("--busy-levels", default=os.environ.get("M10_BUSY_LEVELS")
                   or os.path.join(REPO, "results", "m7_write_policy_mi300x", "busy_levels.json"))   # （H200：環境變數，沒設時不變；2026-10-17）
    s.add_argument("--busy-key", default="busy2")
    s.add_argument("--main-length", type=int, default=32768)
    s.add_argument("--launches", type=int, default=3)
    s.add_argument("--reps", type=int, default=10)
    s.add_argument("--new-tokens", type=int, default=256)
    s.add_argument("--blocks-out")
    k = sub.add_parser("smoke")
    k.add_argument("--model", choices=["llama", "qwen"], required=True)
    k.add_argument("--params-file", default=os.path.join(REPO, "results", "m10_pilot_mi300x", "e0_tune", "CANDIDATE_PARAMS.json"))
    a = ap.parse_args()
    if a.cmd == "_sworker":
        _sworker(json.loads(a.spec))
        return
    {"plan": cmd_plan, "speed": cmd_speed, "smoke": cmd_smoke}[a.cmd](a)


if __name__ == "__main__":
    main()
