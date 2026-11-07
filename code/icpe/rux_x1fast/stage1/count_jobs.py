"""icpe-rux 第 1 段（a）：慢層容量上限的 job 數，直接用 x1_sim.jobs_for 數（不跑模擬、不讀紀錄）。
用法：python -I code/icpe/rux_x1fast/stage1/count_jobs.py > <run>/stdout.log
每列：集合名、每個變體的 job 數、列數（job × 時間設定）、對應的「沒有上限」job 數與列數。"""
import json, os, socket, sys
from datetime import datetime
REPO = os.environ["REPO"]
sys.path.insert(0, os.path.join(REPO, "code", "p3x"))
import x1_sim as XS

CPU17 = ["LRU", "LRU2", "ARC", "S3FIFO", "GDSF", "PENS", "FREQ2Q", "LRU+NONE", "LRU+2HW", "LRU+2HA", "LRU+RPF",
         "O-BEL", "O-BELS", "O-DROP", "O-NOW", "O-PF", "O-ALL"]                                  # PREREG 偏離 4 第 1 點
CPU23 = CPU17 + ["O-DROP-H1800", "O-DROP-H3600", "O-NOW-H1800", "O-NOW-H3600", "O-ALL-H1800", "O-ALL-H3600"]   # 偏離 6 P1
TTL7 = ["0.3", "1", "3", "10", "30", "100", "300"]
GP_NO = ["LRU"] + [f"TTLA:{t}" for t in TTL7] + [f"TTLB:{t}" for t in TTL7]                     # GPU 不預測 15 個
GP16 = GP_NO + ["BEL"]
for p in CPU23:
    assert p in XS.POLICIES, p
assert set(XS.NON_ORACLE) <= set(CPU17)
CPUS = [8, 16, 32, 64, 111.0625, 128, 256, 512, 1024]                                            # PREREG §4：8–1,024＋111.0625
SLOWS = ["ssd", "b32", "nfs"]
SP7 = XS.SLOW_POLICIES
SP_NO = [p for p in SP7 if p != "BEL"]
run_id = os.environ.get("RUN_ID", "adhoc")
ts = datetime.now().astimezone().isoformat(timespec="seconds")
host = socket.gethostname()


def count(cpu_pols, gpols, spols):
    gs = XS.gpu_specs([111.0625], ["wb"], gpols)
    ss = XS.slow_specs([0.5], spols) if spols else None
    J = XS.jobs_for(["T"], ["R1"], CPUS, cpu_pols, modes=["cake"], slows=SLOWS, gspecs=gs, sspecs=ss)
    return len(J), sum(len(j[4]) for j in J)


sets = [
    ("full_cpu23_gpu16_slow7", CPU23, GP16, SP7),
    ("cpu17_gpu16_slow7", CPU17, GP16, SP7),
    ("oslow_only_nonoracle10_gpuno15_slow7", XS.NON_ORACLE, GP_NO, SP7),
    ("oslow_4_only_nonoracle10_gpuno15_BEL", XS.NON_ORACLE, GP_NO, ["BEL"]),
    ("oslow_2_only_nonoracle10_gpuno15_slow6", XS.NON_ORACLE, GP_NO, SP_NO),
]
for name, cp, gp, sp in sets:
    nj, nr = count(cp, gp, sp)
    nj0, nr0 = count(cp, gp, None)
    # 按慢層規則分：GDSF 的 job 拆成 (慢層, GPU) 各一個
    per = {}
    for p in sp:
        per[p] = count(cp, gp, [p])
    print(json.dumps(dict(run_id=run_id, ts=ts, host=host, set=name, n_cpu=len(cp), n_gpu=len(gp), n_slow=len(sp), cpus=len(CPUS),
                          jobs_per_variant=nj, rows_per_variant=nr, nocap_jobs_per_variant=nj0, nocap_rows_per_variant=nr0,
                          jobs_by_slow_policy={k: v[0] for k, v in per.items()},
                          rows_by_slow_policy={k: v[1] for k, v in per.items()})), flush=True)
