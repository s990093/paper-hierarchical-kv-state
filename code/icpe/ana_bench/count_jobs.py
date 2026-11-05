"""PREREG 偏離 6 的 P1–P3：每個紀錄變體多出幾個 job（直接呼叫參考實作的 jobs_for 數，不是估），再乘每 job 的 Rust 秒數推算時間。
用法：python count_jobs.py <code_dir> <out.json>"""
import sys, os, json
code, out = sys.argv[1:3]
sys.path.insert(0, os.path.join(code, "p3x")); sys.path.insert(0, code)
import x1_sim as XS
CPUS = [8, 16, 32, 64, 128, 256, 512, 1024, 111.0625]
P17 = ["LRU", "LRU2", "ARC", "S3FIFO", "GDSF", "PENS", "FREQ2Q", "LRU+NONE", "LRU+2HW", "LRU+2HA", "LRU+RPF", "O-BEL", "O-BELS",
       "O-DROP", "O-NOW", "O-PF", "O-ALL"]
P15 = XS.NON_ORACLE + ["O-BEL", "O-BELS", "O-DROP", "O-NOW", "O-PF"]
H6 = [f"{p}-H{h}" for h in (1800, 3600) for p in ("O-DROP", "O-NOW", "O-ALL")]
TTL3 = [f"{k}:{v}" for k in ("TTLA", "TTLB") for v in ("0.3", "10", "300")]
TTL_NEW = [f"{k}:{v}" for k in ("TTLA", "TTLB") for v in ("1", "3", "30", "100")]
G8 = ["LRU", "BEL"] + TTL3
G16 = G8 + TTL_NEW
SL3 = ["ssd", "b32", "nfs"]


def n(policies, gpols, slows, modes, writes=("wb",), pft=("cpu",)):
    gs = XS.gpu_specs([111.0625], list(writes), gpols, list(pft))
    return len(XS.jobs_for(["t"], ["r"], CPUS, policies, modes=list(modes), slows=slows, gspecs=gs))


base = n(P17, G8, SL3, ["cake"])
items = {
    "main_now（主結果、主時間設定，現在跑的）": base,
    "P1a 7 個 TTL：新 8 個 GPU 規則 × 15 個 CPU 規則": n(P15, TTL_NEW, SL3, ["cake"]),
    "P1b 截尾版：6 個規則 × 16 個 GPU 規則": n(H6, G16, SL3, ["cake"]),
    "P2a nvme807（cake）：17 × 8": n(P17, G8, ["nvme807"], ["cake"]),
    "P2b 只載入（ssd／b32／nfs／nvme807）：17 × 8": n(P17, G8, SL3 + ["nvme807"], ["load"]),
    "P3a wt（主時間設定）：17 × 8": n(P17, G8, SL3, ["cake"], writes=("wt",)),
    "P3b 預取進 GPU（主時間設定；只有預取規則）": n(P17, G8, SL3, ["cake"], pft=("gpu",)) - 0,
}
# 預取進 GPU：jobs_for 對沒有預取的規則跳過 pf=gpu，所以上式就是只有預取規則的 job 數
# 每個紀錄變體的 Rust 秒／job（單執行緒）：實測或推算
VARIANTS = {  # 名稱: (變體數, 秒/job, 來源)
    "CacheWise": (12, None, "NOT_MEASURED 整體平均；diff 中位數 34.6× → Python 11 s/34.6≈0.32〔推算〕"),
    "TraceLab": (3, 6.0, "實測 4.0–8.6 s（run 20261106-001155-icpe-ana-tlprobe），取 6.0〔推算〕"),
    "SemiAnalysis": (12, 0.67, "實測：294 個 job 196.4 s（run 20261105-225652-icpe-ana-variants-sa fast_t1）"),
    "Bailian": (3, 0.06, "Python 約 2 s/job（STATUS_DATA 推算）÷35〔推算〕"),
    "Mooncake": (4, 0.6, "Python 20.7 s/job（G0 實測，多視角）÷35〔推算〕"),
    "FreeInference": (6, 0.14, "Python 約 5 s/job（H200 T1／T5）÷35〔推算〕"),
}
VARIANTS["CacheWise"] = (12, 0.32, VARIANTS["CacheWise"][2])
THREADS, EFF = 40, 0.72    # 16 執行緒實測效率 11.5/16
res = {"jobs_per_variant": items, "variants": {k: dict(n=v[0], sec_per_job=v[1], source=v[2]) for k, v in VARIANTS.items()},
       "threads": THREADS, "thread_efficiency": EFF}
tot = {}
for it, nj in items.items():
    if it.startswith("main_now"):
        continue
    cpu = sum(v[0] * nj * v[1] for v in VARIANTS.values())
    tot[it] = dict(jobs=sum(v[0] for v in VARIANTS.values()) * nj, cpu_h=round(cpu / 3600, 1),
                   wall_h_40threads_one_machine=round(cpu / (THREADS * EFF) / 3600, 2),
                   tracelab_share=round(VARIANTS["TraceLab"][0] * nj * VARIANTS["TraceLab"][1] / cpu, 3))
res["estimate"] = tot
json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
