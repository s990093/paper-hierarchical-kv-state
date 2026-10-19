"""〔算術〕使用者 10/18 的想法：「拆融合進 attention 只用在 prefill，decode 維持原本（BF16）」。只用已經量好的零件，不跑 GPU。

A（現在最好）＝S4＋全載入的還原（43b 的 P3）＋ BF16 的 prefill attention（256 個新 query，所有層）
B（想法）    ＝只搬不拆的還原（43e (a) nodq）＋ 讀壓縮的 prefill attention（同一個 Triton kernel，43e (b)）
之後的 decode 兩邊都讀 BF16（B 要在背景把 KV 拆成 BF16，那段不算在關鍵路徑上，所以對 B 有利）。
32K、真實 CPU 層、E1；只算 attention，MLP 等其他部分兩邊一樣。
用法：python code/m10_fused_prefill_arith.py → results/m10_restore2_mi300x/q5/fused_prefill_arith.md
"""
import csv, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m10_restore2 as R                       # noqa: E402

ATTN = os.path.join(R.R2, "q5", "attn")
MODE = {"fp8": "fp8", "int4": "int4_kivi"}


def attn_ms(m, mode, length, q, gs):
    for r in csv.DictReader(open(os.path.join(ATTN, f"attn_{m}.csv"))):
        if r["mode"] == mode and r["length"] == str(length) and r["queries"] == str(q) and r["gpu_state"] == gs:
            return float(r["median_s"]) * 1000
    return None


def restore_ms(d, gs):
    rows = R.latest_rows(d)
    if not R.usable(d, rows):
        raise SystemExit(f"{d} 不能用（guard 不是 CLEAN）")
    vals = [R.med(rows, mode=mo, config="E1", gpu_state=gs, bw_name="cpu") for mo in ("loadall", "cake")]
    return [v * 1000 if v else None for v in vals]


def main():
    L = ["# 拆融合進 attention 只用在 prefill（〔算術〕；`code/m10_fused_prefill_arith.py`）", "",
         "32K、真實 CPU 層、E1，ms。A＝S4＋全載入的還原＋BF16 prefill attention；B＝只搬不拆的還原＋讀壓縮的 prefill attention（256 個 query，所有層）。",
         "來源：`q4/<m>_<fmt>_P3`、`q5/nodq_<m>_<fmt>`、`q5/attn/attn_<m>.csv`（全部 guard CLEAN）。", "",
         "| 模型 | 格式 | GPU | A 還原 | A prefill | A 合計 | B 還原 | B prefill | B 合計 | B − A | decode 每 token：BF16／讀壓縮 |",
         "|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|:--|"]
    for m in ("llama", "qwen"):
        for tag in ("fp8", "int4"):
            for gs in ("idle", "busy"):
                a_r = restore_ms(os.path.join(R.R2, "q4", f"{m}_{tag}_P3"), gs)[0]
                b_r = min(v for v in restore_ms(os.path.join(R.R2, "q5", f"nodq_{m}_{tag}"), gs) if v)
                a_p, b_p = attn_ms(m, "bf16", 32768, 256, gs), attn_ms(m, MODE[tag], 32768, 256, gs)
                d_b, d_f = attn_ms(m, "bf16", 32768, 1, gs), attn_ms(m, MODE[tag], 32768, 1, gs)
                L.append(f"| {m} | {tag} | {'閒' if gs == 'idle' else '忙'} | {a_r:.1f} | {a_p:.1f} | {a_r + a_p:.1f} | {b_r:.1f} | {b_p:.1f} | "
                         f"{b_r + b_p:.1f} | {b_r + b_p - a_r - a_p:+.1f} | {d_b:.2f}／{d_f:.2f} |")
    out = os.path.join(R.R2, "q5", "fused_prefill_arith.md")
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
