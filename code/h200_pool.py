#!/usr/bin/env python3
"""品質 run 的工作池（H200 版；MI300X 版是 code/m10_queue_mi300x/pool.py，邏輯照抄，只換「剩多少記憶體」的查法）。

  python code/h200_pool.py <jobs.json> <最多幾個同時>
  jobs.json：[{"name": ..., "cmd": ..., "need_gb": ...}]；cmd 在 bash 裡跑（已 source env_h200.sh，可用 m7run）。

- 同一個 Slurm job 的那一張卡上，最多 P 個同時跑；剩餘記憶體 ≥ need_gb＋8 GB、且離上一個開跑 ≥ 75 s 才開下一個
  （75 s：讓上一個載完模型、記憶體用量穩定下來，同 MI300X）。
- 任何一個失敗 → 不再開新的、等跑中的結束、exit 1（CLAUDE.md 規則 2）。
- 剩餘記憶體查不到 → 丟錯（規則 7：查不到不等於有空間）。
- 每個工作的輸出：<cwd>/pool_<name>.log。
"""
import json
import os
import subprocess
import sys
import time

MARGIN = 8
GAP_S = 75
REPO = os.environ.get("REPO") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = f"source {REPO}/code/env_h200.sh; "


def free_gb() -> float:
    """這個 job 看得到的那張卡（Slurm 只給一張，cuda:0）的剩餘記憶體。用 nvidia-smi，不開 CUDA context。"""
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", "0"],
                         capture_output=True, text=True, timeout=60)
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(f"nvidia-smi 查不到剩餘記憶體（rc={out.returncode}）：{out.stderr.strip()!r}")
    return float(out.stdout.strip().splitlines()[0]) / 1024


def main():
    jobs = json.load(open(sys.argv[1]))
    P = int(sys.argv[2])
    running, failed, done = {}, [], []
    pending = list(jobs)
    last_launch = 0.0
    while pending or running:
        for n, (p, log) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                log.close()
                del running[n]
                (done if rc == 0 else failed).append((n, rc))
                print(f"[pool] {'done' if rc == 0 else 'FAILED'} {n} rc={rc} {time.strftime('%H:%M:%S')}", flush=True)
        if failed:
            pending = []
        if pending and len(running) < P and time.time() - last_launch > GAP_S:
            j = pending[0]
            fg = free_gb()
            if fg >= j["need_gb"] + MARGIN:
                pending.pop(0)
                log = open(f"pool_{j['name']}.log", "w")
                running[j["name"]] = (subprocess.Popen(["bash", "-c", ENV + j["cmd"]], stdout=log,
                                                       stderr=subprocess.STDOUT), log)
                last_launch = time.time()
                print(f"[pool] start {j['name']} free={fg:.0f}GB running={len(running)} {time.strftime('%H:%M:%S')}",
                      flush=True)
        time.sleep(5)
    print(f"[pool] finished done={len(done)} failed={failed}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
