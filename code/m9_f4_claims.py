#!/usr/bin/env python3
"""F4 稽核：把三份 main.tex 數字核對片段合成一張 claims_table.csv，並數各狀態。

輸入（都是本次稽核新建的檔）：
  results/audit_20261010/fragments/claims_A.csv   平台 A：§觀察三尾～觀察四、§Evaluation、§Limitations、附錄 A
  results/audit_20261010/fragments/claims_B.csv   平台 B：§平台 B 實測、tab:models、附錄 B 設計段
  results/audit_20261010/fragments/claims_M.csv   摘要、引言、背景表格（kvsize/fit/ratio/kappa）、結論、跨段發現
輸出：
  docs/audit_20261010/claims_table.csv            每列加 audit_run_id、ts、fragment
  results/audit_20261010/claims_counts.csv        依 status × kind × 區段的計數
用法：source code/m7_env.sh; HIP_VISIBLE_DEVICES= m7run f4-claims python code/m9_f4_claims.py
"""
from __future__ import annotations

import csv
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FR = REPO / "results/audit_20261010/fragments"
OUT_TABLE = REPO / "docs/audit_20261010/claims_table.csv"
OUT_COUNTS = REPO / "results/audit_20261010/claims_counts.csv"
RID = os.environ.get("RUN_ID", "NO_RUN_ID")
NOW = datetime.now().astimezone().isoformat(timespec="seconds")
STATUSES = ["一致", "不一致", "找不到來源", "來源有疑慮"]
HEADER = ["claim_id", "section", "line", "claim_text", "paper_value", "kind", "source_path",
          "source_run_id", "evidence_value", "status", "tag", "note"]


def first_line(v: str) -> int:
    s = str(v).split("-")[0].split("–")[0].strip()
    try:
        return int(s)
    except ValueError:
        return 10 ** 6


def main() -> int:
    rows = []
    for frag in ("M", "A", "B"):
        p = FR / f"claims_{frag}.csv"
        if not p.exists():
            print(f"[f4] 🔴 缺 {p}")
            continue
        for r in csv.DictReader(p.open()):
            bad = [k for k in HEADER if k not in r]
            if bad:
                raise SystemExit(f"🔴 {p.name} 缺欄位 {bad}")
            if r["status"] not in STATUSES:
                raise SystemExit(f"🔴 {p.name} {r['claim_id']} 狀態不合法：{r['status']!r}")
            rows.append({"audit_run_id": RID, "ts": NOW, "fragment": frag,
                         **{k: r[k] for k in HEADER}})
    rows.sort(key=lambda r: (first_line(r["line"]), r["claim_id"]))
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TABLE.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[f4] wrote {len(rows)} rows -> {OUT_TABLE.relative_to(REPO)}")

    def region(r):
        n = first_line(r["line"])
        if n <= 109:
            return "1_abstract"
        if n <= 142:
            return "2_intro"
        if n <= 459:
            return "3_background"
        if n <= 1450:
            return "4_eval_A"
        if n <= 1605:
            return "5_platB"
        if n <= 1618:
            return "6_conclusion"
        if n <= 1786:
            return "7_appendixA"
        return "8_appendix_notyet"

    cnt = Counter()
    for r in rows:
        for scope in ("ALL", region(r)):
            for kind in ("ALL", r["kind"]):
                cnt[(scope, kind, r["status"])] += 1
    out = []
    for (scope, kind, st), n in sorted(cnt.items()):
        out.append({"audit_run_id": RID, "ts": NOW, "scope": scope, "kind": kind, "status": st, "n": n})
    with OUT_COUNTS.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"[f4] wrote {len(out)} rows -> {OUT_COUNTS.relative_to(REPO)}")
    tot = Counter(r["status"] for r in rows)
    print("[f4] 全部：", {s: tot.get(s, 0) for s in STATUSES}, "共", len(rows))
    for scope in sorted({region(r) for r in rows}):
        c = Counter(r["status"] for r in rows if region(r) == scope)
        print(f"[f4] {scope:18s}", {s: c.get(s, 0) for s in STATUSES})
    ck = Counter(r["status"] for r in rows if r["kind"] == "實測")
    print("[f4] 只看 kind=實測：", {s: ck.get(s, 0) for s in STATUSES})
    # 判準字面版：「引用檔不存在、只剩 stdout」的「一致」照字面應算「來源有疑慮」
    kw = ("stdout", "遺失", "重建", "不存在", "已刪")
    lit = [r for r in rows if r["status"] == "一致"
           and any(k in (r["source_path"] + r["note"] + r["tag"]) for k in kw)]
    print(f"[f4] 「一致」裡只靠 stdout／遺失檔重建的：{len(lit)} 列"
          f"（照判準字面改判「來源有疑慮」時，一致 {tot['一致'] - len(lit)}、"
          f"來源有疑慮 {tot['來源有疑慮'] + len(lit)}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
