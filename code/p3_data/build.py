"""build.py — 把各來源轉成統一表（03_data_design.md §8）。

  python -I code/p3_data/build.py <source> [--variants v1 v2 ...]     # 轉一個來源（預設全部 variant）
  python -I code/p3_data/build.py fi-constants                        # 服務時間公式的常數（FreeInference 訓練週）
  python -I code/p3_data/build.py summarize                           # 由所有 manifest 重建 results/p3_data/summary.csv
來源：freeinference bailian mooncake wildchat burstgpt cachewise synth

要用 TIARA_NO_GPU=1 bash code/run3090.sh <短名> <這行指令> 包起來（CLAUDE.md §4.1）；run_id 從 RUN_ID 環境變數來。
斷言不過（schema.DataAssertionError）→ 不寫 turns.parquet，在 variant 目錄寫 FAILED.json（完整 checks），exit 5。
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:  # python -I 不會把 script 目錄放進 sys.path；這裡只放自己的程式目錄
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import schema as S  # noqa: E402

# P3_DATA_RESULTS：平台 C（H200）寫到 results/p3_data_h200，和 3090 的逐位元對拍（code/r_h200_p3.sh）；沒設＝原本
RESULTS = os.environ.get("P3_DATA_RESULTS", os.path.join(S.REPO, "results", "p3_data"))
CONST_PATH = os.path.join(RESULTS, "service_constants.json")


def load_constants():
    if not os.path.exists(CONST_PATH):
        raise FileNotFoundError(f"{CONST_PATH} 不存在：先跑 build.py fi-constants（規則 7：查不到要停）")
    return json.load(open(CONST_PATH, encoding="utf-8"))


def _write_failed(source, variant, err, checks):
    d = S.variant_dir(source, variant)
    os.makedirs(d, exist_ok=True)
    for f in ("turns.parquet", "manifest.json"):  # 舊的成功輸出不能留著冒充
        p = os.path.join(d, f)
        if os.path.exists(p):
            os.replace(p, p + ".stale")
    S.write_manifest(os.path.join(d, "FAILED.json"), dict(
        source=source, variant=variant, run_id=os.environ.get("RUN_ID", ""), ts=S.now_iso(), error=err, checks=checks,
        code=S.code_fingerprint()))


def run_one(source, variant, fn):
    """fn() → (df, checks, inputs, extra[, writer])。"""
    S.log(f"build {source}/{variant} start")
    checks_rows = []
    try:
        res = fn()
        df, ck, inputs, extra = res[:4]
        writer = res[4] if len(res) > 4 else None
        checks_rows = ck.rows
        tier = extra.pop("tier", "R")
        man = S.write_variant(df, source, variant, ck, inputs, extra, tier=tier, writer=writer)
        fp = os.path.join(S.variant_dir(source, variant), "FAILED.json")
        if os.path.exists(fp):
            os.replace(fp, fp + ".old")
        st = man["stats"]
        S.log(f"build {source}/{variant} OK rows={st['n_rows']} sessions={st['n_sessions']} "
              f"cens={st['censored_frac']:.4f} think_p50={st['think_ms_p50']}")
        for r in ck.rows:
            S.log(f"  check {'PASS' if r['passed'] else 'FAIL'}{'' if r['hard'] else '(report)'} {r['check']} = {r['value']} [{r['threshold']}]")
        return 0
    except S.DataAssertionError as e:
        _write_failed(source, variant, str(e), e.checks)
        S.log(f"build {source}/{variant} ASSERTION FAILED: {e}")
        return 5
    except Exception as e:  # noqa: BLE001 —— 程式錯誤：完整 traceback 留在 FAILED.json 與 stderr
        tb = traceback.format_exc()
        _write_failed(source, variant, tb, checks_rows)
        S.log(f"build {source}/{variant} ERROR\n{tb}")
        return 6


# ----------------------------------------------------------------------------- 各來源
def build_freeinference(variants):
    import load_freeinference as L
    ws = [f"week{i:02d}" for i in range(1, L.N_WEEKS + 1)]
    rc = 0
    for v in variants or ws:
        wn = int(v[4:])
        rc = max(rc, run_one(L.SOURCE, v, lambda wn=wn: L.load_week(wn)))
    return rc


def build_fi_constants():
    import load_freeinference as L
    c = L.service_constants()
    c.update(run_id=os.environ.get("RUN_ID", ""), ts=S.now_iso(), code=S.code_fingerprint())
    os.makedirs(RESULTS, exist_ok=True)
    S.write_manifest(CONST_PATH, c)
    S.log(json.dumps({k: c[k] for k in ("A", "B", "C", "n_fit", "frac_pred_le_dur_all")}))
    return 0


def build_bailian(variants):
    import load_bailian as L
    const = load_constants()
    rc = 0
    for v in variants or L.VARIANTS:
        rc = max(rc, run_one(L.SOURCE, v, lambda v=v: L.load(v, const)))
    return rc


def build_mooncake(variants):
    import load_mooncake as L
    const = load_constants()
    rc = 0
    for v in variants or L.VARIANTS:
        rc = max(rc, run_one(L.SOURCE, v, lambda v=v: L.load(v, const)))
    return rc


def build_wildchat(variants):
    import load_wildchat as L
    const = load_constants()
    rc = 0
    for v in variants or L.VARIANTS:
        rc = max(rc, run_one(L.SOURCE, v, lambda v=v: L.load(v, const)))
    return rc


def build_burstgpt(variants):
    import load_burstgpt as L
    rc = 0
    for v in variants or L.VARIANTS:
        rc = max(rc, run_one(L.SOURCE, v, lambda v=v: L.load(v)))
    return rc


def build_cachewise(variants):
    import load_cachewise as L
    rc = 0
    for v in variants or L.VARIANTS:
        rc = max(rc, run_one(L.SOURCE, v, lambda v=v: L.load(v)))
    return rc


def build_synth(variants):
    import synth as L
    rc = 0
    for v in variants or list(L.PRESETS):
        rc = max(rc, run_one(L.SOURCE, v, lambda v=v: L.build_variant(v)))
    return rc


# ----------------------------------------------------------------------------- summary.csv
SUMMARY_COLS = ["run_id", "ts", "source", "variant", "tier", "status", "n_sessions", "n_users", "n_rows", "n_sessions_ge2",
                "n_turns_user", "n_turns_tool", "n_turns_unknown", "censored_frac", "censored_short_frac",
                "think_ms_p10", "think_ms_p50", "think_ms_p90", "think_ms_p99", "n_think_human",
                "tool_gap_ms_p50", "tool_gap_ms_p90", "n_think_tool", "think_negative", "think_source", "t_end_source",
                "asserts_hard_pass", "asserts_hard_total", "asserts_failed", "n_train", "n_valid", "n_test", "path"]


def _row_from_manifest(m):
    st = m["stats"]
    cs = m["checks_summary"]
    return dict(run_id=m["run_id"], ts=m["ts"], source=m["source"], variant=m["variant"], tier=m["tier"], status="OK",
                n_sessions=st["n_sessions"], n_users=st["n_users"], n_rows=st["n_rows"], n_sessions_ge2=st["n_sessions_ge2"],
                n_turns_user=st["n_turns_user"], n_turns_tool=st["n_turns_tool"], n_turns_unknown=st["n_turns_unknown"],
                censored_frac=st["censored_frac"], censored_short_frac=st["censored_short_frac"],
                think_ms_p10=st["think_ms_p10"], think_ms_p50=st["think_ms_p50"], think_ms_p90=st["think_ms_p90"],
                think_ms_p99=st["think_ms_p99"], n_think_human=st["n_think_human"],
                tool_gap_ms_p50=st["tool_gap_ms_p50"], tool_gap_ms_p90=st["tool_gap_ms_p90"], n_think_tool=st["n_think_tool"],
                think_negative=st["think_negative"], think_source=st["think_source"], t_end_source=st["t_end_source"],
                asserts_hard_pass=cs["n_hard_pass"], asserts_hard_total=cs["n_hard"], asserts_failed="",
                n_train=st["split_counts"]["train"], n_valid=st["split_counts"]["valid"], n_test=st["split_counts"]["test"],
                path=m["output"])


def summarize():
    rows = []
    for mp in sorted(glob.glob(os.path.join(S.UNIFIED, "*", "*", "manifest.json"))):
        rows.append(_row_from_manifest(json.load(open(mp, encoding="utf-8"))))
    for fp in sorted(glob.glob(os.path.join(S.UNIFIED, "*", "*", "FAILED.json"))):
        f = json.load(open(fp, encoding="utf-8"))
        hard = [c for c in f.get("checks", []) if c.get("hard")]
        rows.append(dict(run_id=f["run_id"], ts=f["ts"], source=f["source"], variant=f["variant"], status="ASSERT_FAIL"
                         if hard and not all(c["passed"] for c in hard) else "ERROR",
                         asserts_hard_pass=sum(c["passed"] for c in hard), asserts_hard_total=len(hard),
                         asserts_failed=";".join(c["check"] for c in hard if not c["passed"]), path=fp))
    # FreeInference 全部 16 週合起來一列（由各週 parquet 重算；run_id＝這次 summarize）
    fi = sorted(glob.glob(os.path.join(S.UNIFIED, "freeinference", "week*", "turns.parquet")))
    if len(fi) == 16:
        cols = ["session_id", "user_id", "trigger", "t_end_ms", "think_ms", "next_trigger", "censored", "obs_end_ms",
                "t_end_source", "think_source", "split"]
        df = pd.concat([pd.read_parquet(p, columns=cols) for p in fi], ignore_index=True)
        st = S.describe(df)
        mans = [json.load(open(p.replace("turns.parquet", "manifest.json"))) for p in fi]
        rows.append(dict(run_id=os.environ.get("RUN_ID", ""), ts=S.now_iso(), source="freeinference", variant="ALL(week01-16)",
                         tier="R", status="OK" if all(m["checks_summary"]["n_hard"] == m["checks_summary"]["n_hard_pass"] for m in mans)
                         else "ASSERT_FAIL", **{k: st[k] for k in ("n_sessions", "n_users", "n_rows", "n_sessions_ge2", "n_turns_user",
                                                                    "n_turns_tool", "n_turns_unknown", "censored_frac",
                                                                    "censored_short_frac", "think_ms_p10", "think_ms_p50",
                                                                    "think_ms_p90", "think_ms_p99", "n_think_human",
                                                                    "tool_gap_ms_p50", "tool_gap_ms_p90", "n_think_tool",
                                                                    "think_negative", "think_source", "t_end_source")},
                         asserts_hard_pass=sum(m["checks_summary"]["n_hard_pass"] for m in mans),
                         asserts_hard_total=sum(m["checks_summary"]["n_hard"] for m in mans), asserts_failed="",
                         n_train=st["split_counts"]["train"], n_valid=st["split_counts"]["valid"], n_test=st["split_counts"]["test"],
                         path=os.path.join(S.UNIFIED, "freeinference", "week*")))
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "summary.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in SUMMARY_COLS})
    S.log(f"summary: {len(rows)} 列 → {out}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("--variants", nargs="*")
    a = ap.parse_args()
    fn = {"freeinference": build_freeinference, "bailian": build_bailian, "mooncake": build_mooncake,
          "wildchat": build_wildchat, "burstgpt": build_burstgpt, "cachewise": build_cachewise, "synth": build_synth}
    if a.what == "fi-constants":
        return build_fi_constants()
    if a.what == "summarize":
        return summarize()
    return fn[a.what](a.variants)


if __name__ == "__main__":
    sys.exit(main())
