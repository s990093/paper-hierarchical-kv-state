"""analyze_v8.py — icpe-rux：`next_sim_analyze.py` 加上 v8 的 GPU 規則（code/icpe/next_sim_analyze.py 一行不改）。
用法同 next_sim_analyze.py（參數原樣轉給它）。
和原本的差別（SEMANTICS 2-5）：
  - BELP（block 粒度的 GPU Belady）是**完美預測**：不放進任何「GPU 不預測」的集合（② 和其他元件的 ④ 都不用）。
  - O-GPU 的 ④＝min over {CPU ∈ NON_ORACLE} × {BEL, BELP}（有哪個用哪個）；② 不變（NON_ORACLE × GPU 不預測，含 APC）。
  - APC 照原本的規則（BEL 以外都算不預測）＝② 和 ④ 都有。
做法：analyze() 的原始碼只換兩處（精確取代，每處剛好一次），在原模組的命名空間重新定義。"""
import inspect
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
for p in (os.path.join(REPO, "code"), os.path.join(REPO, "code", "icpe"), os.path.join(REPO, "code", "p3x")):
    sys.path.insert(0, p)
import next_sim_analyze as NSA  # noqa: E402

src = inspect.getsource(NSA.analyze)
R = [
    ('gp_no = [x for x in gp_all if x != "BEL"]', 'gp_no = [x for x in gp_all if x not in ("BEL", "BELP")]'),
    ('''        if "BEL" in gp_all:
            comps.append(("O-GPU", combos(XS.NON_ORACLE, ["BEL"], sp_no), rival_all))''',
     '''        gp_ogpu = [x for x in ("BEL", "BELP") if x in gp_all]
        if gp_ogpu:
            comps.append(("O-GPU", combos(XS.NON_ORACLE, gp_ogpu, sp_no), rival_all))'''),
]
for a, b in R:
    assert src.count(a) == 1, ("next_sim_analyze.analyze 和預期的不同", a[:60])
    src = src.replace(a, b)
exec(compile(src, NSA.__file__ + ":analyze[v8]", "exec"), NSA.__dict__)
assert NSA.analyze.__code__.co_filename.endswith("[v8]")

if __name__ == "__main__":
    print("analyze_v8: next_sim_analyze.analyze 換成 v8 版（BELP＝oracle，只進 O-GPU 的 ④）", flush=True)
    NSA.main()
