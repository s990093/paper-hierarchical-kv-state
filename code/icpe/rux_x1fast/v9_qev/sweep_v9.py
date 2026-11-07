"""sweep_v9.py — icpe-rux：用 v9 引擎（v8_dev14＋二階排隊 qev=1；PREREG 偏離 19）跑 `x1 sweep --engine rust`。其他全部照 sweep_v8.py（先 import 它，套用它的修改）。
用法同 sweep_v8.py：python -I code/icpe/rux_x1fast/v9_qev/sweep_v9.py --variant V1 --calib <dir> --out-root <dir> x1 sweep ... --engine rust
RUX_QEV=1 → 二階排隊：
  - 每個 job 只放 1 個時間設定（服務順序取決於時間設定，不能共用一次模擬）：_rust_sweep 收到的 job 依時間設定拆開；
  - jobs.txt 加 `qev=1`；CSV 多欄 queue＝closed_event、qwait_mean／qwait_p90／qwait_frac（等待，秒）、qsvc_mean（服務時間）。
  - mean_ret 等 TTFT 欄位＝等待＋服務（v9 的 TTFT 輸出已經加好）。
RUX_QPOST2=1 → 一階近似、到達不早於上一個完成（v9 的 `qpost=2`；偏離 19 的對照）：照 sweep_v8 的 RUX_QPOST=1 路徑，jobs.txt 的 qpost=1 換成 qpost=2，
  CSV 的 queue＝closed_post2（其他欄同 qpost）。
偏離 14 的旗標照 v8_dev14/dev14_flags.py（RUX_PFX=1、RUX_PGR=1）。
沒有 RUX_QEV／RUX_QPOST2 時＝sweep_v8 換成 v9 的引擎（v9 沒開 qev 和 v8_dev14 逐位元相同；smoke_v9.py T2）。RUX_QEV 不可以和 RUX_QPOST(2)、RUX_W5、RUX_CUE_FILE 同時開。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = os.path.join(HERE, "..", "v8_features")
if os.environ.get("RUX_X1FAST_SRC") and not os.path.samefile(os.environ["RUX_X1FAST_SRC"], HERE):
    raise SystemExit(f"RUX_X1FAST_SRC={os.environ['RUX_X1FAST_SRC']} 不是 v9_qev")
os.environ["RUX_X1FAST_SRC"] = HERE
QEV = os.environ.get("RUX_QEV", "0") == "1"
QPOST2 = os.environ.get("RUX_QPOST2", "0") == "1"
if QPOST2:
    assert not QEV, "RUX_QPOST2 和 RUX_QEV 不能同時開"
    os.environ["RUX_QPOST"] = "1"          # 走 sweep_v8 的 qpost 路徑（加 qpost=1、讀等待），下面再換成 qpost=2
if QEV:
    assert os.environ.get("RUX_QPOST", "0") != "1", "RUX_QEV 和 RUX_QPOST 不能同時開"
    assert os.environ.get("RUX_W5", "0") == "0", "RUX_QEV 不支援 W5"
    assert not os.environ.get("RUX_CUE_FILE"), "RUX_QEV 不支援 W3 線索"
sys.path.insert(0, V8)
sys.path.insert(0, os.path.join(HERE, "..", "v8_dev14"))
import sweep_v8 as S8  # noqa: E402  （套用 v8 的全部修改；x1_rust_v6.SRC＝本目錄）
import numpy as np  # noqa: E402

XS, XR = S8.XS, S8.x1_rust_v6
assert os.path.samefile(XR.SRC, HERE), XR.SRC
import dev14_flags  # noqa: E402  （偏離 14 的旗標：RUX_PFX／RUX_PGR；RUX_NZO 是雜訊用，qev 不支援雜訊）
dev14_flags.apply(XS, XR)

if QEV:
    _write_job = XR.write_job

    def write_job(XS_, P, policy, cap_bytes, views, K, gkw, nz=None, sc=None):
        assert len(views) == 1, "qev：一個 job 只能 1 個時間設定"
        assert nz is None and sc is None, "qev 不支援雜訊／慢層上限"
        jd = _write_job(XS_, P, policy, cap_bytes, views, K, gkw, nz=nz, sc=sc)
        jp = os.path.join(jd, "jobs.txt")
        ln = open(jp).read().strip()
        open(jp, "w").write(ln + " qev=1\n")
        return jd

    XR.write_job = write_job
    _read_outputs = XR.read_outputs

    def read_outputs(jd, P, views, gkw, policy):
        eng = _read_outputs(jd, P, views, gkw, policy)
        od = os.path.join(jd, "out", "j0")
        n = len(P["full"])
        w = np.fromfile(os.path.join(od, "v0_qe_wait.f64"), np.float64)
        sv = np.fromfile(os.path.join(od, "v0_qe_svc.f64"), np.float64)
        o = np.fromfile(os.path.join(od, "v0_qe_order.i64"), np.int64)
        assert len(w) == len(sv) == len(o) == n, (len(w), len(sv), len(o), n)
        assert np.array_equal(np.sort(o), np.arange(n)), "qev：服務順序不是 0..n−1 的排列"
        views[0]._qw = (float(w.mean()), float(np.percentile(w, 90)), float((w > 0).mean()), float(sv.mean()))
        return eng

    XR.read_outputs = read_outputs
    _job_rows = XS._job_rows

    def _job_rows_v9(*a, **k):
        rows = _job_rows(*a, **k)
        views = a[2]
        assert len(rows) == len(views) == 1, (len(rows), len(views))
        r, v = rows[0], views[0]
        assert r["slow"] == v.slow_name and r["gpu"] == v.gpu
        r["queue"] = "closed_event"
        r["qwait_mean"], r["qwait_p90"], r["qwait_frac"], r["qsvc_mean"] = v._qw
        return rows

    XS._job_rows = _job_rows_v9
    _rust_sweep = XS._rust_sweep

    def _rust_sweep_v9(JR, workers, put):
        J1 = []
        for j in JR:
            for k in j[4]:
                x = list(j)
                x[4] = [k]
                J1.append(type(j)(x))
        print(f"sweep_v9 qev：{len(JR)} 個 job 依時間設定拆成 {len(J1)} 個", flush=True)
        return _rust_sweep(J1, workers, put)

    XS._rust_sweep = _rust_sweep_v9

if QPOST2:
    _write_job2 = XR.write_job

    def write_job_q2(XS_, P, policy, cap_bytes, views, K, gkw, nz=None, sc=None):
        jd = _write_job2(XS_, P, policy, cap_bytes, views, K, gkw, nz=nz, sc=sc)
        jp = os.path.join(jd, "jobs.txt")
        x = open(jp).read().split()
        assert x.count("qpost=1") == 1, x
        x[x.index("qpost=1")] = "qpost=2"
        open(jp, "w").write(" ".join(x) + "\n")
        return jd

    XR.write_job = write_job_q2
    _job_rows_q = XS._job_rows

    def _job_rows_q2(*a, **k):
        rows = _job_rows_q(*a, **k)
        for r in rows:
            assert r["queue"] == "closed_post"
            r["queue"] = "closed_post2"
        return rows

    XS._job_rows = _job_rows_q2

if __name__ == "__main__":
    assert os.path.samefile(XR.SRC, os.environ["RUX_X1FAST_SRC"])
    print(f"sweep_v9: x1_rust={XR.__file__} SRC={XR.SRC} qev={QEV} qpost2={QPOST2} {dev14_flags.banner()}", flush=True)
    S8.p3_fvar.main()
