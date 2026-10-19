"""adapt_table.py — 把 results/p3_formats/format_table.json（格式表 agent 的輸出）轉成本目錄用的四格式表。**只讀，不改原檔。**

  python -I code/p3_lossdev/adapt_table.py [--model llama]
規則（照 04_prereg_map §1、§3、§8）：
  * 四個格式：bf16、8-bit（Llama＝fp8_token；Qwen＝int8_token）、int4_kivi、int2_kivi。
  * ratio＝bytes_per_block_code ÷ bf16 的 bytes_per_block_code（status 必須是 measured）。
  * 品質：quality_verdict_32k＝PASS／REFERENCE ⇒ 在 ≤32K 過；8-bit 在 >32K 也算過（04 §8 只把「INT4 以下」標未驗證）；
    INT4 在 >32K 不算過（未驗證）；不是 PASS（例如 INT2 的 UNVERIFIED_TF_ONLY）⇒ 讀取一律不過（只能給不會再被讀的）。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

PICK = {"llama": {"bf16": "bf16", "int8": "fp8_token", "int4": "int4_kivi", "int2": "int2_kivi"},
        "qwen": {"bf16": "bf16", "int8": "int8_token", "int4": "int4_kivi", "int2": "int2_kivi"}}


def main():
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "llama"
    src = C.REAL_TABLE
    t = json.load(open(src, encoding="utf-8"))
    rows = {r["fmt"]: r for r in t["rows"] if r["model"] == model}
    out = dict(label=f"由 {os.path.relpath(src, C.REPO)}（run {t['run_id']}）轉出；ratio＝bytes_per_block_code÷BF16；品質照 quality_verdict_32k（規則見 adapt_table.py）",
               source=os.path.relpath(src, C.REPO), source_run_id=t["run_id"], source_sha256=hashlib.sha256(open(src, "rb").read()).hexdigest(),
               model=model, formats={})
    b0 = rows["bf16"]["cells"]["bytes_per_block_code"]
    if b0["status"] != "measured":
        raise SystemExit(f"bf16 bytes status={b0['status']}（規則 2：停下來）")
    for k, fmt in PICK[model].items():
        c = rows[fmt]["cells"]
        b = c["bytes_per_block_code"]
        if b["status"] != "measured":
            raise SystemExit(f"{fmt} bytes status={b['status']}")
        v = c["quality_verdict_32k"]["value"]
        if v in ("PASS", "REFERENCE"):
            pm = None if k in ("bf16", "int8") else 32768
        else:
            pm = 0
        out["formats"][k] = dict(ratio=b["value"] / b0["value"], pass_max_ctx=pm, src_fmt=fmt, quality_verdict_32k=v,
                                 bytes_per_block=b["value"], bytes_run_id=b.get("run_id"))
    fn = os.path.join(C.RES, f"format_table_from_p3formats_{model}.json")
    json.dump(out, open(fn, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(out["formats"], ensure_ascii=False))
    C.load_table(fn)   # 驗證格式
    print("wrote", fn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
