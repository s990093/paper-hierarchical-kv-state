"""大掃描複賽（43）的補充表：讀法 A 的答對率、對 E0 的 ε 細節、隨機 5 個 seed 的平均、存的記憶體（只報告，不改判定）。

判定在 m10_confirm_analyze.py（confirm_<m>.json）；這支只把同一批列攤開，方便看「為什麼沒過」。
用法：python code/m10_confirm_detail.py [--dir results/m10_sweep_mi300x/confirm]  → <dir>/detail_A.md
"""
import argparse, collections, glob, os, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m10_analyze as A                    # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BITS = {"int2_kivi": 3.00, "kv-int4_kivi-int2_kivi": 4.00, "int4_kivi": 5.00, "mxfp4": 4.25,
        "kv-int8_token-int4_kivi": 6.62}   # 位元／值（09 §2.1 的表）
BF16_GIB = {"llama": 4.0, "qwen": 1.75}    # 32K、64 個 block 的 BF16 KV（GiB）


def mem_gib(m, name, fmt):
    """block 0 一律 BF16；SW-<放法>-<佔比>：k＝round(佔比×63) 個低精度；E1s：block 1 以後全部。"""
    k = 63 if name == "E1s" else round(int(name.split("-")[2]) / 100 * 63)
    r = BITS[fmt] / 16
    return BF16_GIB[m] * (k * r + (64 - k)) / 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(REPO, "results", "m10_sweep_mi300x", "confirm"))
    a = ap.parse_args()
    L = ["# 大掃描複賽：讀法 A 的細節（m10_confirm_detail.py；只報告）", "",
         "- 答對率＝讀法 A、T2–T5 兩輪合併的平均分數；ε＝m10_analyze.eps_check（CI 下界 > −2 且每個任務族、每一輪 ≥ −5）。",
         "- R(5)＝同格式、同佔比的 R0–R4 各自平均後再平均。記憶體＝存的 KV（GiB），block 0 一律 BF16。", ""]
    for m in ("llama", "qwen"):
        rows = A.load_rows(sorted(glob.glob(os.path.join(a.dir, f"quality_{m}_*.csv"))))
        ref = A.cond_rows(rows, "E0", "A")
        acc = collections.defaultdict(list)
        for r in rows:
            if r["mode"] == "A":
                acc[r["config"]].append(r["score"])
        e0 = st.mean(acc["E0"])
        L += [f"## {m}（E0 {e0 * 100:.1f}%，{len(acc['E0'])} 列；BF16 {BF16_GIB[m]} GiB）", "",
              "| 配置 | 記憶體 GiB | 答對率 | 對 E0（點，95% CI） | ε 主 | ε 逐族 | 沒過的族 | R(5) 答對率 |",
              "|:--|--:|--:|:--|:--|:--|:--|--:|"]
        for c in sorted(k for k in acc if k != "E0" and not k.startswith("SW-R")):
            name, _, fmt = c.partition("@")
            e = A.eps_check(ref, A.cond_rows(rows, c, "A"))
            bad = ", ".join(f"{x['task']} 第 {x['turn']} 輪 {x['diff'] * 100:+.1f}" for x in e["aux"] if not x["pass"]) or "—"
            rr = ""
            if name.startswith("SW-"):
                frac = name.split("-")[2]
                seeds = [st.mean(acc[f"SW-R{i}-{frac}@{fmt}"]) for i in range(5) if acc.get(f"SW-R{i}-{frac}@{fmt}")]
                rr = f"{st.mean(seeds) * 100:.1f}" if seeds else ""
            L.append(f"| {c} | {mem_gib(m, name, fmt):.2f} | {st.mean(acc[c]) * 100:.1f} | {e['diff'] * 100:+.1f} "
                     f"[{e['ci95'][0] * 100:+.1f}, {e['ci95'][1] * 100:+.1f}] | {'過' if e['main_pass'] else '沒過'} | "
                     f"{'過' if e['aux_pass'] else '沒過'} | {bad} | {rr} |")
        L.append("")
    out = os.path.join(a.dir, "detail_A.md")
    open(out, "w").write("\n".join(L))
    print("\n".join(L))
    print("寫了", out)


if __name__ == "__main__":
    main()
