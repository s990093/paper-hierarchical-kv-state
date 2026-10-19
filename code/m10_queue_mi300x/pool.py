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
def _mem(g: int):
    out = subprocess.run(["amd-smi", "metric", "--mem-usage", "--json"], capture_output=True, text=True).stdout
    d = json.loads(out)
    d = d["gpu_data"] if isinstance(d, dict) else d
    d = next(x for x in d if int(x.get("gpu", -1)) == g)
    return d["mem_usage"]["free_vram"]["value"] / 1024, d["mem_usage"]["total_vram"]["value"] / 1024


def free_gb(g: int) -> float:
    return _mem(g)[0]


# 10/19 03:39Z：44c 的第 3 個 Llama 128K job（need 78 GB）開的時候卡上還空 93 GB，但前兩個還沒長到峰值 → OOM。
# 只看「現在空多少」不夠：每個 job 開的時候登記 need_gb（峰值估計），所有 pool（main、helper）共用一本帳，
# 同一張卡「已登記＋這個」要 ≤ 總量 − MARGIN，而且現在空的也要 ≥ need＋MARGIN（帳外的行程也算進去）。
LEDGER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vram_reserved.json")


def alive(pid: int) -> bool:
    """10/19 05:5xZ：容器的 PID 1 不收殭屍。pool 被停掉後留下的 job 跑完會變成殭屍，/proc/<pid> 還在。
    44c 的 s0、s1 就這樣在帳上多佔了 156 GB，卡 1 閒了幾分鐘才被發現。所以狀態是 Z／X 也算「不在」。"""
    try:
        st = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return st not in ("Z", "X")


def _ledger(fn):
    import fcntl
    with open(LEDGER + ".lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            d = json.load(open(LEDGER))
        except (FileNotFoundError, ValueError):
            d = {}
        d = {k: v for k, v in d.items() if alive(v["pid"])}     # 行程不在了（或是殭屍）就不算
        out = fn(d)
        json.dump(d, open(LEDGER + ".tmp", "w"))
        os.replace(LEDGER + ".tmp", LEDGER)
        return out


def reserve(name: str, g: int, need: float, total: float) -> bool:
    def f(d):
        used = sum(v["need"] for v in d.values() if v["gpu"] == g)
        if used + need + MARGIN > total:
            return False
        d[name] = {"gpu": g, "need": need, "pid": os.getpid(), "ts": time.time()}
        return True
    return _ledger(f)


def release(name: str):
    _ledger(lambda d: d.pop(name, None))


def reserved_by(name: str, pid: int):
    _ledger(lambda d: d[name].update(pid=pid) if name in d else None)


# 10/19（使用者同意）：main 的品質 pool 也用第二張卡空出來的顯存——先放自己那張（TIARA_GPU），塞不下再放另一張。
# 每張卡最多 P 個；helper 只用自己那張。量時間的 stage 不用 pool，不受影響。
OWN = int(os.environ.get("TIARA_GPU", "0"))
QMAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "queue")


def gpus_now():
    """main：自己那張＋另一張。helper：自己那張；main 的佇列全部做完（沒有 .sh、.running）時也用另一張（收尾，10/19）。"""
    if os.environ.get("ROLE") == "main":
        return [OWN] + [g for g in (0, 1) if g != OWN]
    if os.environ.get("ROLE") == "helper":
        main_idle = not any(f.endswith((".sh", ".running")) for f in os.listdir(QMAIN)) if os.path.isdir(QMAIN) else False
        return [OWN] + ([g for g in (0, 1) if g != OWN] if main_idle else [])
    return [OWN]


GPUS = gpus_now()
running, failed, done = {}, [], []
pending = list(jobs)
last_launch = {g: 0 for g in (0, 1)}
while pending or running:
    if HELPER and os.path.exists(WANT):
        preempt(running)
    for n, (p, log, _g) in list(running.items()):
        rc = p.poll()
        if rc is not None:
            log.close(); del running[n]
            release(n)
            (done if rc == 0 else failed).append((n, rc))
            print(f"[pool] {'done' if rc == 0 else 'FAILED'} {n} rc={rc} {time.strftime('%H:%M:%S')}", flush=True)
    if failed:
        pending = []
    if pending:
        GPUS = gpus_now()
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
            fg, tot = _mem(g)
            if fg < j["need_gb"] + MARGIN or not reserve(j["name"], g, j["need_gb"], tot):
                continue
            pending.pop(0)
            log = open(f"pool_{j['name']}.log", "a")
            log.write(f"[pool] launch {time.strftime('%Y-%m-%dT%H:%M:%S')} TIARA_GPU={g} ROLE={os.environ.get('ROLE', '')}\n")
            log.flush()
            running[j["name"]] = (subprocess.Popen(["bash", "-c", f"export TIARA_GPU={g}; " + ENV + j["cmd"]], stdout=log,
                                                   stderr=subprocess.STDOUT, start_new_session=True), log, g)
            reserved_by(j["name"], running[j["name"]][0].pid)        # 帳上記 job 的行程（pool 死了 job 還在時照樣算）
            last_launch[g] = time.time()
            print(f"[pool] start {j['name']} gpu{g} free={fg:.0f}GB running={len(running)} {time.strftime('%H:%M:%S')}", flush=True)
            break
    time.sleep(5)
print(f"[pool] finished done={len(done)} failed={failed}", flush=True)
sys.exit(1 if failed else 0)
