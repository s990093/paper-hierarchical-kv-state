"""品質 run 的工作池：最多 P 個同時跑，VRAM 夠才開下一個；任何一個失敗 → 不再開新的、等跑中的結束、exit 1（規則 2）。
jobs.json：[{"name":..., "cmd":..., "need_gb":...}]；cmd 在 bash 裡跑（已 source m7_env.sh，可用 m7run）。"""
import json, os, signal, subprocess, sys, time
jobs = json.load(open(sys.argv[1])); P = int(sys.argv[2]); MARGIN = 8
# 10/19 雙卡：helper（第二條 runner）上的 pool 看到 WANT_BOTH（main 要開量時間的 stage）→ 停掉自己的 job、exit 75，
# runner 把 stage 放回佇列；之後重跑時品質 job 照題號續跑（m10_quality／m10_drop／m10_sweep 都跳過已寫的題）。
WANT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "WANT_BOTH")
HELPER = os.environ.get("ROLE") == "helper"


def preempt(running):
    for n, (p, log, _g) in running.items():
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    t0 = time.time()
    while any(p.poll() is None for p, _l, _g in running.values()) and time.time() - t0 < 90:
        time.sleep(2)
    for n, (p, log, _g) in running.items():
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        log.write(f"\n[pool] PREEMPTED（WANT_BOTH）{time.strftime('%H:%M:%S')}\n")
        log.close()
    print(f"[pool] PREEMPTED：停掉 {list(running)}（main 要兩張卡）{time.strftime('%H:%M:%S')}", flush=True)
    sys.exit(75)
ENV = "source /mlsteam/workspace/paper-hierarchical-kv-state/code/m7_env.sh; "
def free_gb(g: int) -> float:
    out = subprocess.run(["amd-smi", "metric", "--mem-usage", "--json"], capture_output=True, text=True).stdout
    d = json.loads(out)
    d = d["gpu_data"] if isinstance(d, dict) else d
    d = next(x for x in d if int(x.get("gpu", -1)) == g)
    return d["mem_usage"]["free_vram"]["value"] / 1024


# 10/19（使用者同意）：main 的品質 pool 也用第二張卡空出來的顯存——先放自己那張（TIARA_GPU），塞不下再放另一張。
# 每張卡最多 P 個；helper 只用自己那張。量時間的 stage 不用 pool，不受影響。
OWN = int(os.environ.get("TIARA_GPU", "0"))
GPUS = [OWN] + ([g for g in (0, 1) if g != OWN] if os.environ.get("ROLE") == "main" else [])
running, failed, done = {}, [], []
pending = list(jobs)
last_launch = {g: 0 for g in GPUS}
while pending or running:
    if HELPER and os.path.exists(WANT):
        preempt(running)
    for n, (p, log, _g) in list(running.items()):
        rc = p.poll()
        if rc is not None:
            log.close(); del running[n]
            (done if rc == 0 else failed).append((n, rc))
            print(f"[pool] {'done' if rc == 0 else 'FAILED'} {n} rc={rc} {time.strftime('%H:%M:%S')}", flush=True)
    if failed:
        pending = []
    if pending:
        j = pending[0]
        try:                              # pool_skip.txt：一行一個名稱前綴；判定流程走不到的封存工作，開跑前可略過（照實記 skipped）
            skip = [x.strip() for x in open("pool_skip.txt") if x.strip() and not x.startswith("#")]
        except FileNotFoundError:
            skip = []
        if any(j["name"].startswith(x) for x in skip):
            pending.pop(0)
            print(f"[pool] skipped {j['name']} (pool_skip.txt) {time.strftime('%H:%M:%S')}", flush=True)
            continue
        for g in GPUS:                    # 每次換卡前等 75 秒（上一個 job 的顯存要先長出來，free 才準）
            if sum(1 for _p, _l, gg in running.values() if gg == g) >= P or time.time() - last_launch[g] <= 75:
                continue
            fg = free_gb(g)
            if fg < j["need_gb"] + MARGIN:
                continue
            pending.pop(0)
            log = open(f"pool_{j['name']}.log", "a")
            log.write(f"[pool] launch {time.strftime('%Y-%m-%dT%H:%M:%S')} TIARA_GPU={g} ROLE={os.environ.get('ROLE', '')}\n")
            log.flush()
            running[j["name"]] = (subprocess.Popen(["bash", "-c", f"export TIARA_GPU={g}; " + ENV + j["cmd"]], stdout=log,
                                                   stderr=subprocess.STDOUT, start_new_session=True), log, g)
            last_launch[g] = time.time()
            print(f"[pool] start {j['name']} gpu{g} free={fg:.0f}GB running={len(running)} {time.strftime('%H:%M:%S')}", flush=True)
            break
    time.sleep(5)
print(f"[pool] finished done={len(done)} failed={failed}", flush=True)
sys.exit(1 if failed else 0)
