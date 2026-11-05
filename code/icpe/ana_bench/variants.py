"""Rust 編譯版本與執行緒數的跑分：同一組 job（difftest 建好的 run 目錄），每個版本各跑一次，比輸出檔是否逐位元相同、量時間。
用法：python variants.py <trace_dir> <src_run_dir> <out.jsonl> <label=bin:threads> ...
不是 sweep、不進 results。"""
import sys, os, json, time, hashlib, shutil, subprocess, resource
tdir, src, out = sys.argv[1:4]
specs = sys.argv[4:]
base = os.path.dirname(out)


def tree_hash(d):
    h = hashlib.sha256()
    for root, _, files in sorted(os.walk(d)):
        for f in sorted(files):
            p = os.path.join(root, f)
            h.update(os.path.relpath(p, d).encode())
            h.update(open(p, "rb").read())
    return h.hexdigest()[:16]


ref = None
for sp in specs:
    label, rest = sp.split("=", 1)
    binp, thr = rest.rsplit(":", 1)
    rd = os.path.join(base, f"v_{label}")
    if os.path.exists(rd):
        shutil.rmtree(rd)
    os.makedirs(rd)
    for f in os.listdir(src):
        if f.endswith((".txt", ".f64")):
            shutil.copy(os.path.join(src, f), rd)
    env = dict(os.environ, X1FAST_THREADS=thr)
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN); w0 = time.perf_counter()
    cp = subprocess.run(["nice", "-n", "19", binp, tdir, rd], env=env, capture_output=True, text=True)
    wall = time.perf_counter() - w0; r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (r1.ru_utime + r1.ru_stime) - (r0.ru_utime + r0.ru_stime)
    if cp.returncode != 0:
        print("FAILED", label, cp.stderr[-2000:]); sys.exit(2)
    tim = {}
    for line in open(os.path.join(rd, "timing.txt")):
        p = line.split()
        tim[p[0]] = float(p[1])
    jobs = {k: v for k, v in tim.items() if k.startswith("j")}
    hsh = tree_hash(os.path.join(rd, "out"))
    ref = ref or hsh
    rec = dict(label=label, bin=binp, threads=int(thr), n_jobs=len(jobs), sum_job_s=round(sum(jobs.values()), 3),
               all_jobs_wall_s=round(tim.get("all_jobs_wall_s", float("nan")), 3), trace_load_s=round(tim["trace_load_s"], 3),
               proc_wall_s=round(wall, 3), proc_cpu_s=round(cpu, 3), out_hash=hsh, same_as_first=(hsh == ref),
               ts=time.strftime("%Y-%m-%dT%H:%M:%S%z"), host="acvlab", run_id=os.environ.get("RUN_ID", "adhoc"))
    with open(out, "a") as fo:
        fo.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
