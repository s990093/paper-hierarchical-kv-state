#!/usr/bin/env python3
"""`m4_oracle.load_precision_tiers` 的守門測試（假輸入，不碰 GPU）。

兩個失敗模式各一條，都是實際踩過的：
  1. 污染的量測（host_contention != QUIET）被當成有效常數用掉。
  2. n 小時把「階內全距」讀成「階間訊號」——2026-08-31 的污染資料
     換算出 int4 的反量化比 fp8 便宜，物理上說不通。

用法：python code/test_precision_tiers.py
"""
import csv, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m4_oracle as M

COLS = ["run_id", "ts", "model_key", "tier", "gpu", "ctx", "n_prefixes",
        "round", "kv_dtype", "ttft_ms", "gpu_kv_cache_tokens", "host_contention"]


def write(path: Path, rows: list[tuple[str, float]], contention: str) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, COLS)
        w.writeheader()
        for tier, ttft in rows:
            w.writerow({"run_id": "TEST", "ts": "2026-09-01T00:00:00+08:00",
                        "model_key": "llama", "tier": tier, "gpu": 0,
                        "ctx": 16384, "n_prefixes": 1, "round": "warm",
                        "kv_dtype": "auto", "ttft_ms": ttft,
                        "gpu_kv_cache_tokens": 48128,
                        "host_contention": contention})


def run(rows, contention, tmp: Path) -> dict:
    for c in list(tmp.glob("retrieval_cost_precision_tiers*.csv")):
        c.unlink()
    write(tmp / "retrieval_cost_precision_tiers_quiet.csv", rows, contention)
    M.M2 = tmp
    return M.load_precision_tiers("nvme")


def main() -> int:
    fails = []
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)

        # ── 1. 污染 → 全部 NOT_MEASURED，且說得出原因
        clean_shape = [("gpu_resident", 140.0), ("gpu_resident", 143.0),
                       ("gpu_resident", 147.0), ("gpu_fp8", 200.0),
                       ("gpu_fp8", 205.0), ("gpu_fp8", 210.0)]
        r = run(clean_shape, "HEAVY", tmp)
        if r["fp8"]["dequant_ms_per_block"] != "NOT_MEASURED":
            fails.append("HEAVY 的資料沒有被擋下")
        if "reason" not in r["fp8"]:
            fails.append("擋下污染資料時沒有說明原因")

        # ── 2. QUIET 且全距不重疊 → 量得到
        r = run(clean_shape, "QUIET", tmp)
        if r["fp8"].get("distinguishable") is not True:
            fails.append("全距不重疊時應判為可區分")
        exp = (205.0 - 143.0) / (16384 / M.BLOCK)
        if r["fp8"]["dequant_ms_per_block"] != round(exp, 5):
            fails.append(f"反量化成本算錯：{r['fp8']['dequant_ms_per_block']} != {round(exp, 5)}")

        # ── 3. QUIET 但全距重疊 → 與零無法區分，不得給數字
        overlap = [("gpu_resident", 140.0), ("gpu_resident", 143.0),
                   ("gpu_resident", 147.0), ("gpu_fp8", 138.0),
                   ("gpu_fp8", 145.0), ("gpu_fp8", 149.0)]
        r = run(overlap, "QUIET", tmp)
        if r["fp8"]["dequant_ms_per_block"] != "NOT_MEASURED":
            fails.append("全距重疊時仍給出數字（把雜訊讀成訊號）")
        if r["fp8"].get("distinguishable") is not False:
            fails.append("全距重疊時應判為不可區分")

        # ── 4. 沒有檔案 → NOT_MEASURED，不得丟例外
        for c in list(tmp.glob("*.csv")):
            c.unlink()
        r = M.load_precision_tiers("nvme")
        if r["int4"]["dequant_ms_per_block"] != "NOT_MEASURED":
            fails.append("沒有檔案時應回 NOT_MEASURED")

    for f in fails:
        print(f"✗ {f}")
    print("✓ load_precision_tiers 的四條守門全部通過" if not fails
          else f"🔴 {len(fails)} 條失敗")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
