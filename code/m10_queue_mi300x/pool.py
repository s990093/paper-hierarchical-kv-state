"""品質 run 的工作池：最多 P 個同時跑，VRAM 夠才開下一個；任何一個失敗 → 不再開新的、等跑中的結束、exit 1（規則 2）。
jobs.json：[{"name":..., "cmd":..., "need_gb":...}]；cmd 在 bash 裡跑（已 source m7_env.sh，可用 m7run）。"""
import json, subprocess, sys, time
jobs = json.load(open(sys.argv[1])); P = int(sys.argv[2]); MARGIN = 8
ENV = "source /mlsteam/workspace/paper-hierarchical-kv-state/code/m7_env.sh; "
def free_gb():
    out = subprocess.run(["amd-smi", "metric", "--mem-usage", "--json"], capture_output=True, text=True).stdout
    d = json.loads(out)
    d = d["gpu_data"][0] if isinstance(d, dict) else d[0]
    return d["mem_usage"]["free_vram"]["value"] / 1024
running, failed, done = {}, [], []
pending = list(jobs)
last_launch = 0
while pending or running:
    for n, (p, log) in list(running.items()):
        rc = p.poll()
        if rc is not None:
            log.close(); del running[n]
            (done if rc == 0 else failed).append((n, rc))
            print(f"[pool] {'done' if rc == 0 else 'FAILED'} {n} rc={rc} {time.strftime('%H:%M:%S')}", flush=True)
    if failed:
        pending = []
    if pending and len(running) < P and time.time() - last_launch > 75:
        j = pending[0]
        fg = free_gb()
        if fg >= j["need_gb"] + MARGIN:
            pending.pop(0)
            log = open(f"pool_{j['name']}.log", "w")
            running[j["name"]] = (subprocess.Popen(["bash", "-c", ENV + j["cmd"]], stdout=log, stderr=subprocess.STDOUT), log)
            last_launch = time.time()
            print(f"[pool] start {j['name']} free={fg:.0f}GB running={len(running)} {time.strftime('%H:%M:%S')}", flush=True)
    time.sleep(5)
print(f"[pool] finished done={len(done)} failed={failed}", flush=True)
sys.exit(1 if failed else 0)
