"""t1g_exposure.py — T1G 附加 1（docs/h200/icpe/T1G_gpu_rerun.md；只報告，不改判定）：同一個節點上別的使用者的 job。

§4 只擋我們自己的 job；$USER 以外的 job（含同 lab 帳號的人）擋不了。每一步（overlap.csv 的時間窗）用 sacct -a 分成：
  常駐鄰居：窗開始前就在跑、窗結束後還在跑 → 只記數量，不算暴露（各臂、各次重複都一樣在它旁邊，順序又是隨機的）
  事件：在窗裡開始或結束（負載改變的時刻：開始時載權重、結束時釋放）
暴露單位＝同一個 run_id、同一組設定欄（KEYS 裡檔案有的）、同一個 rep 的那些列；時間＝它們 ts 的〔最早 − 120 s，最晚 + 120 s〕，
有事件落在裡面就算暴露。拿掉暴露單位，用 t1g.py 同一支分析（an_*）在影子目錄重算，列出拿掉前後的判定字：
  某個量測點（設定欄相同、只差 rep）拿掉後重複次數變少而且不到 6 次 → 那一點 NOT_MEASURED；判定列跟著標 NM、不比
  sacct 查不到 → 那一步 NOT_MEASURED（不是「沒有暴露」；CLAUDE.md §1 規則 7）→ 那一個 part 的敏感度 NOT_MEASURED
另外（附加 1 的偏離 1，只報告、不進判定）：同一個量測點裡，暴露的重複對沒暴露的重複——每個單位取回來請求（round > 1）TTFT 的平均，
〔暴露的中位 ÷ 沒暴露的中位 − 1〕，每一步報點數、中位、四分位、變慢的點數（P02 每點只有 1–2 次，不算）。
  判定字不同 → 停下來回報使用者（附加 1「怎麼用」），這支只報告

  python code/icpe/t1g_exposure.py scan                       → results/icpe_h200/t1/gpu/exposure/exposure_steps.csv
  python code/icpe/t1g_exposure.py analyze --part a1pp|a1p|a2|p02|a3|all [--regress]
      要先有 t1g.py analyze 的輸出與 scan；→ exposure/units_<part>.csv、exposure/t1g_exposure.json（影子目錄在 run 目錄）
      --regress：什麼都不拿掉，影子目錄的判定必須和 t1g.py analyze 的逐值相同（只差 run_id、ts）→ 驗影子目錄這條路
  python code/icpe/t1g_exposure.py old --label L --csv F [F ...] [--main-b]  原本 ICPE run 的暴露比例（窗＝每個 run_id 的 ts 範圍；只報告）
提交的 CSV 不寫別人的帳號名與 job 名（repo 是公開的），只記 job id 與「是不是同 lab 帳號」；完整的 sacct 輸出在 run 目錄。
（要 pandas：在 env_h200.sh 的 conda 環境裡跑）
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t1g  # noqa: E402

EXP = os.path.join(t1g.T1G, "exposure")
MARGIN = timedelta(seconds=120)
MIN_REPS = 6
KEYS = ["strategy", "policy", "cpu_frac", "cpu_cap_chunks", "ssd_frac", "ssd_dev", "gap_s", "wl_seed", "workload", "cpu_gibps",
        "release", "k", "cpu_GiB"]
ARM = ("strategy", "policy")
LOCAL = datetime.now().astimezone().tzinfo
META = {"run_id", "ts", "src_run", "src_runs"}
PARTS = dict(a1pp=("T1G-A1PP", "a1pp"), a1p=("T1G-A1P", "a1p"), a2=("T1G-A2-", "a2"), p02=("T1G-P02-", "p02"), a3=("T1G-A3-", "a3"))


def _t(s):
    return None if s in ("", "Unknown", "None") else datetime.fromisoformat(s).replace(tzinfo=LOCAL)


def _sacct(args):
    p = subprocess.run([os.path.join(t1g.SLURM_BIN, "sacct")] + args, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"sacct exit {p.returncode}: {p.stderr.strip()}")
    return p.stdout


_ACCT = {}


def own_account(job):
    if job not in _ACCT:
        _ACCT[job] = _sacct(["-j", str(job), "-X", "-n", "-P", "-o", "Account"]).split("\n")[0].strip().lower()
    return _ACCT[job]


def window_jobs(node, t0, t1):
    """t0、t1：有時區的 datetime。$USER 以外、和窗重疊的 job；查不到 → 丟例外（規則 7）。"""
    fmt = "%Y-%m-%dT%H:%M:%S"
    out = _sacct(["-a", "-N", node, "-X", "-P", "-n", "-S", t0.astimezone(LOCAL).strftime(fmt), "-E", t1.astimezone(LOCAL).strftime(fmt),
                  "-o", "JobID,User,Account,JobName,Start,End,State"])
    me, jobs = os.environ["USER"], []
    for ln in out.splitlines():
        jid, user, acct, name, st, en, state = ln.split("|", 6)
        if user == me or _t(st) is None:
            continue
        jobs.append(dict(job=jid, user=user, account=acct.lower(), name=name, start=_t(st), end=_t(en), state=state))
    return jobs


def classify(jobs, t0, t1):
    steady, events = [], []
    for j in jobs:
        if j["start"] < t0 and (j["end"] is None or j["end"] > t1):
            steady.append(j)
            continue
        for kind in ("start", "end"):
            if j[kind] is not None and t0 <= j[kind] <= t1:
                events.append(dict(j, t=j[kind], kind=kind))
    return steady, events


def _detail(name, obj):
    d = os.environ.get("RUN_DIR")
    if d:
        json.dump(obj, open(os.path.join(d, name), "w"), ensure_ascii=False, indent=1, default=str)


def cmd_scan(a):
    os.makedirs(EXP, exist_ok=True)
    rid, ts = t1g.run_id(), t1g.ts()
    rows, detail = [], {}
    for r in csv.DictReader(open(os.path.join(t1g.T1G, "overlap.csv"))):
        t0, t1_ = datetime.fromisoformat(r["t_start"]), datetime.fromisoformat(r["t_end"])
        row = dict(run_id=rid, ts=ts, step_run=r["run_id"], step=r["step"], node=r["node"], t_start=r["t_start"], t_end=r["t_end"])
        try:
            mine = own_account(r["job"])
            steady, events = classify(window_jobs(r["node"], t0, t1_), t0, t1_)
        except Exception as e:                   # noqa: BLE001 — 規則 7：查不到就是 NOT_MEASURED
            print(f"[exposure] {r['run_id']}：sacct 查不到：{e}", file=sys.stderr)
            rows.append(dict(row, status="NOT_MEASURED", n_steady="", n_steady_same_account="", n_events="", n_events_same_account="",
                             events=""))
            continue
        detail[r["run_id"]] = dict(steady=steady, events=events)
        rows.append(dict(row, status="ok", n_steady=len(steady), n_steady_same_account=sum(j["account"] == mine for j in steady),
                         n_events=len(events), n_events_same_account=sum(e["account"] == mine for e in events),
                         events=json.dumps([[e["t"].isoformat(), e["kind"], e["job"], e["account"] == mine] for e in events])))
    p = os.path.join(EXP, "exposure_steps.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    _detail("sacct_detail.json", detail)
    print("→", p, len(rows), "步；NOT_MEASURED", sum(r["status"] != "ok" for r in rows))
    return 0


def _n(v):
    try:
        return f"{float(v):.6g}"
    except (TypeError, ValueError):
        return str(v)


def _steps_events():
    p = os.path.join(EXP, "exposure_steps.csv")
    if not os.path.exists(p):
        raise SystemExit("[exposure] 先跑 scan")
    out = {}
    for r in csv.DictReader(open(p)):
        out[r["step_run"]] = None if r["status"] != "ok" else [datetime.fromisoformat(e[0]) for e in json.loads(r["events"])]
    return out


def _valid_runs(part):
    prefix, _ = PARTS[part]
    val, _tried = t1g._valid(prefix)
    if part == "p02":
        val.pop("T1G-P02-SMOKE", None)
    return {r: s for s, r in val.items() if r}, sorted(s for s, r in val.items() if r is None)


def _sources(part):
    d = os.path.join(t1g.T1G, PARTS[part][1])
    if part in ("a1pp", "a1p"):
        return [os.path.join(d, "b_share.csv")]
    if part == "a2":
        return [os.path.join(d, "b2_share.csv")]
    if part == "p02":
        return [os.path.join(d, "gpu_requests.csv")]
    return [os.path.join(d, f"r1h_gpu_llama_doc_{tag}.csv") for tag in _a3_tags()]


def _a3_tags():
    p = os.path.join(t1g.T1G, "a3", "a3_plan.tsv")
    return [ln.split("\t")[0] for ln in open(p) if ln.strip() and not ln.startswith("#")]


def units(part, ev):
    """→ (各來源檔要拿掉的列〔原檔的列號〕, 單位表, 量測點表, NOT_MEASURED 的步)。"""
    import pandas as pd
    runs, nm_steps = _valid_runs(part)
    nm_steps += sorted(s for r, s in runs.items() if ev.get(r, None) is None)
    drop, us = {}, []
    for src in _sources(part):
        df = pd.read_csv(src)
        df = df[df.run_id.isin(runs)]
        drop[src] = []
        keys = [c for c in KEYS if c in df.columns]
        t = pd.to_datetime(df.ts, utc=True)
        g = df.assign(_t=t).groupby(["run_id"] + keys + ["rep"], dropna=False)
        for k, x in g:
            k = dict(zip(["run_id"] + keys + ["rep"], k if isinstance(k, tuple) else (k,)))
            lo, hi = x._t.min().to_pydatetime() - MARGIN, x._t.max().to_pydatetime() + MARGIN
            evs = ev.get(k["run_id"])
            hit = None if evs is None else [e.isoformat() for e in evs if lo <= e <= hi]
            if hit:
                drop[src] += list(x.index)
            ret = x[x["round"] > 1].ttft.mean() if {"round", "ttft"} <= set(x.columns) else float("nan")
            us.append(dict(src=os.path.basename(src), step=runs[k["run_id"]], **k, t_first=x._t.min().isoformat(), t_last=x._t.max().isoformat(),
                           n_rows=len(x), exposed=(None if hit is None else bool(hit)), events_hit=json.dumps(hit), ret_ttft_mean=ret))
    U = pd.DataFrame(us)
    pts = []
    if len(U):
        pk = ["src"] + [c for c in KEYS if c in U.columns]
        for k, x in U.groupby(pk, dropna=False):
            before = x.rep.nunique()
            after = x[x.exposed == False].rep.nunique()  # noqa: E712 — None（查不到）不算乾淨
            nm = (after < before and after < MIN_REPS)
            pts.append(dict(zip(pk, k), reps_before=before, reps_after=after, nm=nm, tags=sorted({s for s in x.step})))
    return drop, U, pts, nm_steps


def _words(part, root):
    """→ [(鍵, 字, 遮罩欄)]：列層的判定字帶設定欄（給 NM 遮罩用）；遮罩欄是 None ＝整個 part 的字。"""
    import pandas as pd
    W = []
    d = os.path.join(root, PARTS[part][1])
    if part in ("a1pp", "a1p"):
        v = json.load(open(os.path.join(d, "analysis", "verdict.json")))
        W.append((("passed",), v.get("passed"), None))
        if os.path.exists(os.path.join(d, "analysis", "table.csv")):
            for _, r in pd.read_csv(os.path.join(d, "analysis", "table.csv")).iterrows():
                W.append((("row", _n(r.cpu_frac)), bool(r.passed), dict(cpu_frac=_n(r.cpu_frac))))
    elif part == "a2":
        v = json.load(open(os.path.join(d, "analysis", "verdict.json")))
        W.append((("gpu_pass",), v.get("gpu_pass"), None))
        t = pd.read_csv(os.path.join(d, "analysis", "b2_verdict.csv"))
        kc = ["workload", "release", "cpu_gibps", "ssd_dev", "io_model", "gap_s", "ssd_frac", "wl_seed", "cpu_frac", "strategy"]
        for _, r in t[t.src == "gpu"].iterrows():
            m = {c: _n(r[c]) for c in kc if c in t.columns and c not in ARM}
            W.append((("row",) + tuple(_n(r[c]) for c in kc if c in t.columns), bool(r["pass"]), m))
    elif part == "p02":
        v = json.load(open(os.path.join(d, "analysis", "p02_verdict.json")))
        W.append((("passed",), v.get("passed"), None))
        for k, x in sorted(v.get("criteria", {}).items()):
            W.append((("criteria", k), x, None))
    else:
        v = json.load(open(os.path.join(d, "analysis", "verdict.json")))
        W.append((("n_survive",), v.get("n_survive"), None))
        for c in v.get("cells", []):
            W.append((("survive", c["tag"]), c["survive"], dict(tag=c["tag"])))
            p = os.path.join(d, f"r1_gpu_verdict_{c['tag']}.csv")
            if os.path.exists(p):
                for _, r in pd.read_csv(p).iterrows():
                    W.append((("row", c["tag"], _n(r.wl_seed), r.stat), bool(r["pass"]), dict(tag=c["tag"], wl_seed=_n(r.wl_seed))))
    return W


def _masked(mask, nm_pts, any_nm):
    if mask is None:
        return any_nm
    for p in nm_pts:
        if "tag" in mask and not any(s.startswith(f"T1G-A3-{mask['tag']}-") for s in p["tags"]):
            continue
        shared = [c for c in mask if c != "tag" and c in p]
        if all(_n(p[c]) == mask[c] for c in shared):
            return True
    return False


def _shadow(part, base, drop):
    """影子目錄：overlap.csv＋這個 part 的目錄（不含 analysis/），來源檔拿掉 drop 的列。"""
    sh = os.path.join(base, "gpu")
    os.makedirs(sh, exist_ok=True)
    shutil.copy2(os.path.join(t1g.T1G, "overlap.csv"), sh)
    sub = PARTS[part][1]
    dst = os.path.join(sh, sub)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(os.path.join(t1g.T1G, sub), dst, ignore=shutil.ignore_patterns("analysis"))
    for src, idx in drop.items():
        if idx:
            import pandas as pd
            full = pd.read_csv(src)
            full[~full.index.isin(idx)].to_csv(os.path.join(dst, os.path.basename(src)), index=False)
    return sh


def _run_part(part, root):
    old = t1g.T1G
    t1g.T1G = root
    try:
        return {"a1pp": lambda: t1g.an_a1("a1pp"), "a1p": lambda: t1g.an_a1("a1p"), "a2": t1g.an_a2, "p02": t1g.an_p02, "a3": t1g.an_a3}[part]()
    finally:
        t1g.T1G = old


def _strip(x):
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if k not in META}
    if isinstance(x, list):
        return [_strip(v) for v in x]
    return x


def exposed_vs_clean(part, U):
    """附加 1 的偏離 1：同一個量測點裡〔暴露的中位 ÷ 沒暴露的中位 − 1〕，按步驟分開報。"""
    import numpy as np
    if part == "p02" or not len(U):
        return dict(status="NOT_MEASURED", reason="P02 每點只有 1–2 次重複" if part == "p02" else "沒有單位")
    pk = ["src"] + [c for c in KEYS if c in U.columns]
    per = {}
    for _, x in U.groupby(pk, dropna=False):
        e = x[x.exposed == True].ret_ttft_mean.dropna()  # noqa: E712
        c = x[x.exposed == False].ret_ttft_mean.dropna()  # noqa: E712
        if len(e) and len(c):
            per.setdefault(x.step.iloc[0], []).append(float(e.median() / c.median() - 1))
    out = {}
    for st, r in sorted(per.items()):
        r = np.asarray(r)
        out[st] = dict(n_points=len(r), median_pct=100 * float(np.median(r)), p25_pct=100 * float(np.percentile(r, 25)),
                       p75_pct=100 * float(np.percentile(r, 75)), n_slower=int((r > 0).sum()))
    return dict(status="ok" if out else "NOT_MEASURED", by_step=out)


def one(part, ev, base, regress):
    import pandas as pd
    drop, U, pts, nm_steps = units(part, ev)
    if len(U):
        U = U.rename(columns={"run_id": "src_run"})
        U.insert(0, "ts", t1g.ts())
        U.insert(0, "run_id", t1g.run_id())
        U.to_csv(os.path.join(EXP, f"units_{part}.csv"), index=False)
    n_exp = 0 if not len(U) else int((U.exposed == True).sum())  # noqa: E712
    sh = _shadow(part, os.path.join(base, part), {} if regress else drop)
    try:
        _run_part(part, sh)
    except (SystemExit, Exception) as e:     # noqa: BLE001 — 拿掉之後分析跑不動 → 這個 part 的敏感度 NOT_MEASURED
        if regress:
            raise
        return dict(part=part, status="NOT_MEASURED", reason=f"分析失敗：{e}", n_units=len(U), n_exposed=n_exp, nm_steps=nm_steps,
                    exposed_vs_clean=exposed_vs_clean(part, U))
    if regress:
        paths = {"a1pp": ["a1pp/analysis/verdict.json", "a1pp/analysis/table.csv"], "a1p": ["a1p/analysis/verdict.json", "a1p/analysis/table.csv"],
                 "a2": ["a2/analysis/verdict.json", "a2/analysis/b2_verdict.csv"], "p02": ["p02/analysis/verdict.json", "p02/analysis/p02_verdict.json"],
                 "a3": ["a3/analysis/verdict.json"] + [f"a3/r1_gpu_verdict_{t}.csv" for t in _a3_tags()]}[part]
        bad = []
        for p in paths:
            x, y = os.path.join(t1g.T1G, p), os.path.join(sh, p)
            if p.endswith(".json"):
                same = _strip(json.load(open(x))) == _strip(json.load(open(y)))
            else:
                A, B = pd.read_csv(x), pd.read_csv(y)
                cols = [c for c in A.columns if c not in META]
                same = A[cols].equals(B[cols])
            if not same:
                bad.append(p)
        return dict(part=part, status="REGRESS_PASS" if not bad else "REGRESS_FAIL", differ=bad)
    nm_pts = [p for p in pts if p["nm"]]
    any_nm = bool(nm_pts) or bool(nm_steps)
    main = {k: (v, m) for k, v, m in _words(part, t1g.T1G)}
    new = {k: (v, m) for k, v, m in _words(part, sh)}
    rows, diffs = [], []
    for k in sorted(set(main) | set(new), key=str):
        mv, mm = main.get(k, ("MISSING", None))
        nv, nmask = new.get(k, ("MISSING", mm))
        nm = bool(nm_steps) or _masked(mm if k in main else nmask, nm_pts, any_nm)
        rows.append(dict(key=list(k), main=mv, noexp=("NOT_MEASURED" if nm else nv)))
        if not nm and mv != nv:
            diffs.append(list(k))
    status = "NOT_MEASURED" if nm_steps else ("DIFFERENT" if diffs else ("SAME_SOME_NM" if any_nm else "SAME"))
    return dict(part=part, status=status, n_units=len(U), n_exposed=n_exp, nm_steps=nm_steps,
                nm_points=[{k: v for k, v in p.items() if k != "nm"} for p in nm_pts], words=rows, diffs=diffs,
                exposed_vs_clean=exposed_vs_clean(part, U))


def cmd_analyze(a):
    os.makedirs(EXP, exist_ok=True)
    ev = _steps_events()
    base = os.path.join(os.environ.get("RUN_DIR") or os.path.join(EXP, "_shadow"), "shadow")
    parts = list(PARTS) if a.part == "all" else [a.part]
    p = os.path.join(EXP, "t1g_exposure.json")
    doc = json.load(open(p)) if os.path.exists(p) else {}
    rc = 0
    for part in parts:
        r = one(part, ev, base, a.regress)
        r.update(run_id=t1g.run_id(), ts=t1g.ts())
        if a.regress:
            doc.setdefault("regress", {})[part] = r
            rc |= r["status"] != "REGRESS_PASS"
        else:
            doc[part] = r
        print(f"[exposure] {part}：", json.dumps({k: v for k, v in r.items() if k not in ("words", "nm_points")}, ensure_ascii=False, default=str))
    doc["rule"] = "T1G 附加 1：事件＝別的使用者的 job 在步驟窗裡開始或結束；單位〔最早 ts − 120 s，最晚 + 120 s〕碰到事件＝暴露；判定照 §3（全部單位）"
    json.dump(doc, open(p, "w"), ensure_ascii=False, indent=1, default=str)
    return rc


def cmd_old(a):
    import numpy as np
    import pandas as pd
    os.makedirs(EXP, exist_ok=True)
    rid, ts = t1g.run_id(), t1g.ts()
    out, detail = [], {}
    for f in a.csv:
        df = pd.read_csv(f, usecols=lambda c: c in set(KEYS) | {"run_id", "ts", "rep"} | (set(t1g.MAIN_B) if a.main_b else set()))
        if a.main_b:                         # 只留主設定（t1g.MAIN_B＝p01_tools.MAIN_B；A1''／A1' 的判定只用這些列）
            for k, v in t1g.MAIN_B.items():
                if k in df.columns:
                    df = df[df[k].astype(str) == str(v)] if isinstance(v, str) else df[np.isclose(df[k].astype(float), float(v))]
        keys = [c for c in KEYS if c in df.columns]
        df["_t"] = pd.to_datetime(df.ts, utc=True)
        for run, x in df.groupby("run_id"):
            t0, t1_ = x._t.min().to_pydatetime(), x._t.max().to_pydatetime()
            row = dict(run_id=rid, ts=ts, label=a.label, src=os.path.relpath(os.path.abspath(f), t1g.REPO), src_run=run,
                       t_first=t0.astimezone(LOCAL).isoformat(), t_last=t1_.astimezone(LOCAL).isoformat())
            try:
                steady, events = classify(window_jobs(a.node, t0, t1_), t0, t1_)
            except Exception as e:           # noqa: BLE001 — 規則 7
                print(f"[exposure] {run}：sacct 查不到：{e}", file=sys.stderr)
                out.append(dict(row, status="NOT_MEASURED"))
                continue
            detail[run] = dict(steady=steady, events=events)
            n_u = n_e = 0
            for _, u in x.groupby(keys + ["rep"], dropna=False):
                lo, hi = u._t.min().to_pydatetime() - MARGIN, u._t.max().to_pydatetime() + MARGIN
                n_u += 1
                n_e += any(lo <= e["t"] <= hi for e in events)
            out.append(dict(row, status="ok", n_steady=len(steady), n_events=len(events), n_units=n_u, n_exposed=n_e))
    p = os.path.join(EXP, "exposure_old.csv")
    new = not os.path.exists(p)
    cols = ["run_id", "ts", "label", "src", "src_run", "t_first", "t_last", "status", "n_steady", "n_events", "n_units", "n_exposed"]
    with open(p, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerows(out)
    _detail(f"sacct_old_{a.label}.json", detail)
    print("→", p, len(out), "個 run")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("scan")
    s = sp.add_parser("analyze"); s.add_argument("--part", required=True, choices=list(PARTS) + ["all"]); s.add_argument("--regress", action="store_true")
    s = sp.add_parser("old"); s.add_argument("--label", required=True); s.add_argument("--csv", nargs="+", required=True)
    s.add_argument("--node", default="hgx-cn01"); s.add_argument("--main-b", action="store_true")
    a = ap.parse_args()
    sys.exit({"scan": cmd_scan, "analyze": cmd_analyze, "old": cmd_old}[a.cmd](a))


if __name__ == "__main__":
    main()
