"""sweep_v8_dev14.py — 用 v8_dev14 引擎（v8＋PREREG 偏離 14 的規則）跑 `x1 sweep --engine rust`。其他全部照 sweep_v8.py（先 import 它）。
用法同 sweep_v8.py：python -I code/icpe/rux_x1fast/v8_dev14/sweep_v8_dev14.py --variant V1 --calib <dir> --out-root <dir> x1 sweep ... --engine rust
偏離 14 的旗標（照 icpe-sim 的 x1_sim `--pf-expire`／`--noise-overdue`；本分支的 x1_sim 沒有這兩個參數，所以用環境變數）：
  RUX_PFX=1             → 每個 job 加 `pfx=1`（引擎只在 O-PF／O-ALL 這類 pf＝opf 的規則開）；CSV 每列多 pf_expire=True。
  RUX_PGR=1             → 每個 job 加 `pgr=1`（SIM_CHANGES 23：預取到 GPU 時騰完空間再查候選）；CSV 每列多 pf_gpu_recheck=True。
  RUX_NZO=resample      → 每個 job 加 `nzo=resample`（只影響有雜訊、模型 P 的 job）；CSV 每列多 noise_overdue=resample。
                          重抽用亂數、B1 也照 CPython 3.12 的 set 順序 → 只在 Python 3.12 跑（同 x1_rust.unsupported）。
都沒設＝sweep_v8 換成 v8_dev14 的引擎（和 v8 逐位元相同；smoke_v8_dev14.py D1）。可以和 RUX_W5、RUX_QPOST、RUX_CUE_FILE 一起用（sweep_v8 的路徑）。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.join(HERE, "..", "v8_features")
if os.environ.get("RUX_X1FAST_SRC") and not os.path.samefile(os.environ["RUX_X1FAST_SRC"], HERE):
    raise SystemExit(f"RUX_X1FAST_SRC={os.environ['RUX_X1FAST_SRC']} 不是 v8_dev14")
os.environ["RUX_X1FAST_SRC"] = HERE
sys.path.insert(0, V8)
sys.path.insert(0, HERE)
import sweep_v8 as S8  # noqa: E402  （套用 v8 的全部修改；x1_rust_v6.SRC＝本目錄）

XS, XR = S8.XS, S8.x1_rust_v6
assert os.path.samefile(XR.SRC, HERE), XR.SRC

import dev14_flags  # noqa: E402

dev14_flags.apply(XS, XR)

if __name__ == "__main__":
    assert os.path.samefile(XR.SRC, os.environ["RUX_X1FAST_SRC"])
    print(f"sweep_v8_dev14: x1_rust={XR.__file__} SRC={XR.SRC} {dev14_flags.banner()}", flush=True)
    S8.p3_fvar.main()
