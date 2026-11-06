"""sweep_v8.py — icpe-rux：用 v8 跑 `x1 sweep --engine rust`，多兩個 GPU 規則 APC、BELP（block 粒度；只有 Rust）。code/p3x 一行不改。
用法同 code/icpe/p3_fvar.py：
  python -I code/icpe/rux_x1fast/v8_features/sweep_v8.py --variant V1 --calib <dir> --out-root <dir> x1 sweep ... --gpu-policies LRU APC BELP --engine rust
做法（照 sweep_v6.py）：
  1. RUX_X1FAST_SRC 沒設時指到本目錄（v8 的 main.rs）；sys.modules["x1_rust"]＝v6 目錄的 x1_rust_v6。
  2. x1_sim._rust_sweep 的 write_job 多傳 sc=j[7]（同 sweep_v6）。
  3. x1_sim.parse_gpol 多認 APC、BELP（不依時間 → 一個 job 跑多個時間設定，同 LRU／BEL）；CSV 的 gpu_policy 欄就寫 APC／BELP。
  4. Python 的 Engine 遇到 APC／BELP 就停（沒有 Python 版；不支援的 job 不可以退回 Python）。
W5：環境變數 RUX_W5=1 → 每個 job 都加 w5（CPU 層 INT4＋一次拆完的殘差），CSV 多一欄 cpu_fmt＝int4_oneshot（輸出放另一個目錄）。
W3 線索：RUX_CUE_FILE=<cue.u8>（make_cue.py）→ GPU 規則 CUEP、CUEO（③；分析用 analyze_w3.py）。排隊還沒接進 sweep。"""
import inspect
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
os.environ.setdefault("RUX_X1FAST_SRC", HERE)
V6 = os.path.join(REPO, "code", "icpe", "rux_x1fast", "v6_slowcap")
for p in (os.path.join(REPO, "code"), os.path.join(REPO, "code", "icpe"), os.path.join(REPO, "code", "p3x"), V6):
    sys.path.insert(0, p)

import x1_rust_v6  # noqa: E402
sys.modules["x1_rust"] = x1_rust_v6
import x1_sim as XS  # noqa: E402

_OLD = "_XR.write_job(me, P, j[3], j[2] * GiB, views, _K, _gpu_kw(j[5]), nz=j[6])"
_src = inspect.getsource(XS._rust_sweep)
assert _src.count(_OLD) == 1, "x1_sim._rust_sweep 和預期的不同（write_job 那行）"
exec(compile(_src.replace(_OLD, _OLD[:-1] + ", sc=j[7])"), XS.__file__ + ":_rust_sweep[v8]", "exec"), XS.__dict__)
assert XS._rust_sweep.__code__.co_filename.endswith("[v8]"), "_rust_sweep 沒有換成 v8 版"

NEW_GPU = ("APC", "BELP", "CUEP", "CUEO")
_parse_gpol = XS.parse_gpol


def parse_gpol(spec):
    if spec in NEW_GPU:
        return spec.lower(), None
    return _parse_gpol(spec)


XS.parse_gpol = parse_gpol
_init = XS.Engine.__init__


def _engine_init(self, *a, **k):
    assert k.get("gpu_policy", "LRU") not in NEW_GPU, f"GPU 規則 {k.get('gpu_policy')} 只有 Rust 版（v8），不可以退回 Python"
    _init(self, *a, **k)


XS.Engine.__init__ = _engine_init

# W3 線索（RUX_CUE_FILE=<cue.u8>，make_cue.py 產生）：GPU 規則 CUEP／CUEO 的 job 加 `cue=1`＋這個檔（長度必須＝請求數）。
# 一次 sweep 只能一份紀錄（cue 檔是那份紀錄的）。CUEP／CUEO 是 ③（簡單線索），分析用 analyze_w3.py，不可以混進 ②。
CUE_FILE = os.environ.get("RUX_CUE_FILE", "")
CUE = ("CUEP", "CUEO")

# W5（RUX_W5=1）：每個 job 都是「CPU 層 INT4＋一次拆完的殘差」（SEMANTICS 5-1）；CSV 多一欄 cpu_fmt＝int4_oneshot。
# 殘差 d1＝(T_meas − 64·i4 ÷ r_cpu) ÷ 64，T_meas＝P0-1b S4（fa3）32K、CPU、INT4 全部載入的還原中位數（閒／忙）。
W5 = os.environ.get("RUX_W5", "0") in ("1", "degen")       # degen＝驗證用：走 W5 的路徑但 w5＝BF、dc1＝0、dc2＝d（＝沒開）
W5_DEGEN = os.environ.get("RUX_W5", "0") == "degen"
import numpy as _np
_T = {}
if W5:
    import csv as _csv
    for _r in _csv.DictReader(open(os.path.join(REPO, "results/icpe_h200/p01/grid/llama_s4_fa3/summary.csv"))):
        if _r["length"] == "32768" and _r["bw_name"] == "cpu" and _r["config"] == "E1" and _r["mode"] == "loadall":
            _T[_r["gpu_state"]] = float(_r["median_t_restore"])
    assert set(_T) == {"idle", "busy"}, _T
_CUE_ARR = _np.fromfile(CUE_FILE, _np.uint8) if CUE_FILE else None
_write_job = x1_rust_v6.write_job


def write_job(XS_, P, policy, cap_bytes, views, K, gkw, nz=None, sc=None):
    jd = _write_job(XS_, P, policy, cap_bytes, views, K, gkw, nz=nz, sc=sc)
    extra = []
    if gkw.get("gpu_cap", 0) and gkw.get("gpu_policy") in CUE:
        assert _CUE_ARR is not None, "GPU 規則 CUEP／CUEO 要設 RUX_CUE_FILE"
        assert len(_CUE_ARR) == len(P["full"]), ("cue 檔的長度和請求數不同", len(_CUE_ARR), len(P["full"]))
        _CUE_ARR.tofile(os.path.join(jd, "cue.u8"))
        extra.append("cue=1")
    if W5:
        assert gkw.get("pf_target", "cpu") == "cpu", "W5 不支援預取進 GPU"
        i4 = int(K["int4_bytes"])
        w = []
        for v in views:
            if W5_DEGEN:
                w += [0.0, float(v.d)]
                continue
            d1 = (_T[v.gpu] - 64 * i4 / v.cpu.r) / 64
            assert d1 > 0, (v.gpu, d1)
            w += [d1, d1]
        _np.array(w, _np.float64).tofile(os.path.join(jd, "w5.f64"))
        extra.append(f"w5={(64 << 20) if W5_DEGEN else i4}")
    if extra:
        jp = os.path.join(jd, "jobs.txt")
        ln = open(jp).read().strip()
        open(jp, "w").write(ln + " " + " ".join(extra) + "\n")
    return jd


x1_rust_v6.write_job = write_job
if W5:
    _job_rows = XS._job_rows

    def _job_rows_w5(*a, **k):
        rows = _job_rows(*a, **k)
        for r in rows:
            r["cpu_fmt"] = "w5_degen" if W5_DEGEN else "int4_oneshot"
        return rows

    XS._job_rows = _job_rows_w5

import p3_fvar  # noqa: E402

if __name__ == "__main__":
    assert os.path.samefile(x1_rust_v6.SRC, os.environ["RUX_X1FAST_SRC"])
    print(f"sweep_v8: x1_rust={x1_rust_v6.__file__} SRC={x1_rust_v6.SRC}", flush=True)
    p3_fvar.main()
