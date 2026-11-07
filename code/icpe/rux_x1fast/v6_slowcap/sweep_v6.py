"""sweep_v6.py — icpe-rux：用 v6（慢層容量上限）跑 `x1 sweep --engine rust`，code/p3x 一行不改。
用法同 code/icpe/p3_fvar.py（參數原樣轉給它）：
  python -I code/icpe/rux_x1fast/v6_slowcap/sweep_v6.py --variant V1 --calib <dir> --out-root <dir> x1 sweep ... --engine rust
做法：
  1. sys.modules["x1_rust"]＝本目錄的 x1_rust_v6（x1_sim 的 `import x1_rust` 拿到的就是它）。
  2. x1_sim._rust_sweep 的原始碼只改一處——write_job 多傳 sc=j[7]（原本沒傳，因為 v5 不支援慢層上限）——在 x1_sim 的
     命名空間重新定義（inspect.getsource → 字串取代，取代處必須剛好出現一次）。其他（job 展開、摘要、CSV、npz）全是 x1_sim 原本的。
  3. 交給 p3_fvar.main()。"""
import inspect
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
for p in (os.path.join(REPO, "code"), os.path.join(REPO, "code", "icpe"), os.path.join(REPO, "code", "p3x"), HERE):
    sys.path.insert(0, p)

import x1_rust_v6  # noqa: E402
sys.modules["x1_rust"] = x1_rust_v6
import x1_sim as XS  # noqa: E402

_OLD = "_XR.write_job(me, P, j[3], j[2] * GiB, views, _K, _gpu_kw(j[5]), nz=j[6])"
_src = inspect.getsource(XS._rust_sweep)
assert _src.count(_OLD) == 1, "x1_sim._rust_sweep 和預期的不同（write_job 那行）"
exec(compile(_src.replace(_OLD, _OLD[:-1] + ", sc=j[7])"), XS.__file__ + ":_rust_sweep[v6]", "exec"), XS.__dict__)
assert XS._rust_sweep.__code__.co_filename.endswith("[v6]"), "_rust_sweep 沒有換成 v6 版"

import p3_fvar  # noqa: E402

if __name__ == "__main__":
    print(f"sweep_v6: x1_rust={x1_rust_v6.__file__} SRC={x1_rust_v6.SRC}", flush=True)
    p3_fvar.main()
