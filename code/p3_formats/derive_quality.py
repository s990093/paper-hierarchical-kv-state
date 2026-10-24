"""p3_formats/derive_quality.py — 第三階段格式表：補算 Llama 的 FP8-K0／INT8 生成品質（只用 CPU、只讀既有列）。

為什麼要補：第二階段 Step 1（stage 30）的品質列裡，Llama 也跑了 E1s@fp8k0_token、E1s@int8_token，
但 Step 2b（K3）只在 Qwen 上判（08 §2.4），所以 Llama 這兩個格式的「對 E0 配對答對率差＋95% CI」從來沒有算過。
這支用**凍結的** `code/m10_analyze.py quality`（一行不改，subprocess 呼叫）在同一批既有列上算，
並先做兩個重現檢查（同一支程式、同一批檔，結果必須和 repo 裡已經提交的 JSON 逐欄相同）：
  1. Qwen E1s@fp8k0_head／E1s@int8_token 對 `analysis/fp8/quality_qwen_step2b.json`
  2. Llama E1 對 `analysis/fp8/quality_llama_step1.json`
任一不同 → 停（CLAUDE.md 規則 2），不寫 derived。

輸入（只讀）：results/m10_precision_mi300x/quality/step1_<m>_*.csv.gz（run_id 見每列）
輸出：$RUN_DIR/csv/（解壓的暫存，不進 git）、$RUN_DIR/repro_*/、results/p3_formats/derived/quality_llama_step2b{.json,_table.csv}、
      results/p3_formats/derived/derive_quality.json（重現檢查結果、run_id、ts）

用法：TIARA_NO_GPU=1 bash code/run3090.sh p3fmt-qderive python code/p3_formats/derive_quality.py
"""
from __future__ import annotations

import datetime
import glob
import gzip
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
Q = os.path.join(REPO, "results", "m10_precision_mi300x", "quality")
A8 = os.path.join(REPO, "results", "m10_precision_mi300x", "analysis", "fp8")
OUT = os.path.join(REPO, "results", "p3_formats", "derived")
RUN_ID = os.environ.get("RUN_ID", "adhoc")
RUN_DIR = os.environ.get("RUN_DIR") or os.path.join("/ssd8/hungwei/paper-hkv/runs", RUN_ID)
# P3_CONST=h200（2026-10-19）：同一支、同樣的重現檢查，讀 H200 的品質列與分析，寫 results/p3_formats_h200/derived。沒設＝原本
sys.path.insert(0, os.path.join(REPO, "code"))
import p3_platform as PF  # noqa: E402
if PF.PLAT != "mi300x":
    Q = PF.rdir("m10_precision_mi300x", "quality")
    A8 = PF.rdir("m10_precision_mi300x", "analysis", "fp8")
    OUT = os.path.join(REPO, "results", f"p3_formats_{PF.PLAT}", "derived")
ANALYZE = os.path.join(REPO, "code", "m10_analyze.py")
KEYS = ("n_rows", "n_items", "diff", "ci95", "ci_halfwidth", "main_pass", "aux_pass", "eps_pass", "aux", "flips",
        "same_as_e0", "mean_kl_vs_e0", "mean_lp_read", "b_values")


def unzip(model: str) -> list[str]:
    d = os.path.join(RUN_DIR, "csv")
    os.makedirs(d, exist_ok=True)
    out = []
    for p in sorted(glob.glob(os.path.join(Q, f"step1_{model}_*.csv.gz"))):
        q = os.path.join(d, os.path.basename(p)[:-3])
        with gzip.open(p, "rb") as fi, open(q, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        out.append(q)
    assert out, f"找不到 step1_{model}_*.csv.gz"
    return out


def analyze(model: str, csvs: list[str], configs: list[str], tag: str, out_dir: str) -> dict:
    cmd = [sys.executable, ANALYZE, "quality", "--model", model, "--csv", *csvs, "--configs", *configs,
           "--tag", tag, "--out-dir", out_dir]
    print("$", " ".join(cmd[:4]), "…", " ".join(cmd[-6:]), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout, flush=True)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr, flush=True)
        raise SystemExit(f"m10_analyze.py quality 失敗 rc={r.returncode}（規則 2：停）")
    return json.load(open(os.path.join(out_dir, f"quality_{model}_{tag}.json")))


# H200（2026-10-19）：repo 裡 H200 的 step1／step2b 分析是在 T1／LB 兩個「只報告」的 shard 還沒寫完時算的（Qwen 用同樣 9 個檔重跑：
# 主任務 T2–T5 的欄位逐位元相同，只有含 T1／LB 列的 e0_accuracy、n_items、mean_lp_read 不同；T1 每格 n=9 vs 現在 30，RUNLOG_H200_P3 P3H-0b）。
# 所以 H200 只對主任務算的欄位要求相同；mean_lp_read（含 T1 列的平均）與 run_ids（多了後來寫完的 shard）照記、不擋。MI300X 照原本全部要相同。
REPORT_ONLY = {"mean_lp_read"} if PF.PLAT != "mi300x" else set()
NOTED = []                         # 照記、不擋的差異（寫進 derive_quality.json）


def same(new: dict, old: dict, configs: list[str]) -> list[str]:
    bad = []
    for c in configs:
        for cond, e_old in old["configs"][c].items():
            e_new = new["configs"].get(c, {}).get(cond)
            if e_new is None:
                bad.append(f"{c}/{cond}: 新的沒有")
                continue
            for k in KEYS:
                if k in e_old and e_new.get(k) != e_old.get(k):
                    (NOTED if k in REPORT_ONLY else bad).append(f"{c}/{cond}/{k}: {e_new.get(k)!r} != {e_old.get(k)!r}")
    if sorted(new["run_ids"]) != sorted(old["run_ids"]):
        (NOTED if PF.PLAT != "mi300x" else bad).append(f"run_ids 不同：{new['run_ids']} vs {old['run_ids']}")
    return bad


def main():
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    rep = {"run_id": RUN_ID, "ts": ts, "tool": "code/m10_analyze.py quality（凍結版，未修改；subprocess）", "checks": {}}
    csv_q, csv_l = unzip("qwen"), unzip("llama")
    # 重現 1：Qwen step2b
    c1 = ["E1s@fp8k0_head", "E1s@int8_token"]
    new = analyze("qwen", csv_q, c1, "step2b", os.path.join(RUN_DIR, "repro_qwen"))
    bad = same(new, json.load(open(os.path.join(A8, "quality_qwen_step2b.json"))), c1)
    rep["checks"]["repro_qwen_step2b"] = {"ref": os.path.relpath(os.path.join(A8, "quality_qwen_step2b.json"), REPO),
                                          "configs": c1, "identical": not bad, "diffs": bad}
    # 重現 2：Llama E1（step1）
    c2 = ["E1"]
    new = analyze("llama", csv_l, c2, "step1", os.path.join(RUN_DIR, "repro_llama"))
    bad2 = same(new, json.load(open(os.path.join(A8, "quality_llama_step1.json"))), c2)
    rep["checks"]["repro_llama_step1_E1"] = {"ref": os.path.relpath(os.path.join(A8, "quality_llama_step1.json"), REPO),
                                             "configs": c2, "identical": not bad2, "diffs": bad2}
    if PF.PLAT != "mi300x":         # MI300X 的輸出不加這個鍵（和原本相同）
        rep["noted_not_blocking"] = {"keys": sorted(REPORT_ONLY) + ["run_ids"], "diffs": NOTED,
                                     "why": "H200 的已提交分析在 T1／LB shard 寫完之前算的；主任務欄位要求相同（RUNLOG_H200_P3 P3H-0b）"}
    os.makedirs(OUT, exist_ok=True)
    if bad or bad2:
        json.dump(rep, open(os.path.join(RUN_DIR, "derive_quality.FAILED.json"), "w"), indent=1, ensure_ascii=False)
        raise SystemExit(f"重現檢查沒過（規則 2：停）：{bad + bad2}")
    # 補算：Llama E1s@fp8k0_token／E1s@int8_token（從來沒有算過）
    c3 = ["E1s@fp8k0_token", "E1s@int8_token"]
    tmp = os.path.join(RUN_DIR, "derived_llama")
    res = analyze("llama", csv_l, c3, "step2b", tmp)
    for fn in ("quality_llama_step2b.json", "quality_llama_step2b_table.csv"):
        shutil.copy(os.path.join(tmp, fn), os.path.join(OUT, fn))
    rep["derived"] = {"file": os.path.relpath(os.path.join(OUT, "quality_llama_step2b.json"), REPO), "configs": c3,
                      "source_run_ids": res["run_ids"],
                      "note": "第二階段沒有報過這兩格（Step 2b 只判 Qwen）。同一支凍結程式、同一批列；"
                              "JSON 裡的 csv 路徑指向本 run 解壓的暫存檔（內容＝repo 的 .csv.gz）。"}
    json.dump(rep, open(os.path.join(OUT, "derive_quality.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in rep.items() if k != "derived"}, ensure_ascii=False, indent=1))
    print("OK")


if __name__ == "__main__":
    main()
