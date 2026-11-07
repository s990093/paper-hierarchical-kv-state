"""dev14_flags.py — 偏離 14 的三個旗標（照 icpe-sim 的 x1_sim --pf-expire／--noise-overdue／--pf-gpu-recheck）給 sweep 外殼共用（sweep_v8_dev14.py、v9 的 sweep_v9.py）。
環境變數：RUX_PFX=1 → 每個 job 加 `pfx=1`、每列 pf_expire=True；RUX_NZO=resample → `nzo=resample`、noise_overdue=resample（只在 Python 3.12）；
RUX_PGR=1 → `pgr=1`、pf_gpu_recheck=True。都沒設＝不改任何東西。apply(XS, XR) 要在 sweep_v8 的修改之後呼叫。"""
import os
import sys

PFX = os.environ.get("RUX_PFX", "0") == "1"
NZO = os.environ.get("RUX_NZO", "clamp")
PGR = os.environ.get("RUX_PGR", "0") == "1"
assert NZO in ("clamp", "resample"), NZO
if NZO != "clamp":
    assert sys.version_info[:2] == (3, 12), "RUX_NZO=resample 要 Python 3.12（B1／B2 的亂數順序照 CPython 3.12 的 set）"
ON = PFX or NZO != "clamp" or PGR


def apply(XS, XR):
    if not ON:
        return
    _write_job = XR.write_job

    def write_job(XS_, P, policy, cap_bytes, views, K, gkw, nz=None, sc=None):
        jd = _write_job(XS_, P, policy, cap_bytes, views, K, gkw, nz=nz, sc=sc)
        jp = os.path.join(jd, "jobs.txt")
        ln = open(jp).read().strip()
        open(jp, "w").write(ln + (" pfx=1" if PFX else "") + ("" if NZO == "clamp" else f" nzo={NZO}") + (" pgr=1" if PGR else "") + "\n")
        return jd

    XR.write_job = write_job
    _job_rows = XS._job_rows

    def _job_rows_d14(*a, **k):
        rows = _job_rows(*a, **k)
        for r in rows:
            if PFX:
                r["pf_expire"] = True
            if NZO != "clamp":
                r["noise_overdue"] = NZO
            if PGR:
                r["pf_gpu_recheck"] = True
        return rows

    XS._job_rows = _job_rows_d14


def banner():
    return f"deviation14 pf_expire={PFX} noise_overdue={NZO} pf_gpu_recheck={PGR}"
