#!/usr/bin/env python3
"""品質 run 的工作池（H200 版；MI300X 版是 code/m10_queue_mi300x/pool.py，邏輯照抄，只換「剩多少記憶體」的查法）。

  python code/h200_pool.py <jobs.json> <最多幾個同時>
  jobs.json：[{"name": ..., "cmd": ..., "need_gb": ...}]；cmd 在 bash 裡跑（已 source env_h200.sh，可用 m7run）。

- 同一個 Slurm job 的那一張卡上，最多 P 個同時跑；離上一個開跑 ≥ 75 s（讓上一個載完模型，同 MI300X），而且兩個條件都要成立才開：
  (a) 剩餘記憶體 ≥ need_gb＋8 GB（同 MI300X）；
  (b) 預約：卡的總記憶體 − 跑中工作的 need_gb 總和 ≥ need_gb＋8 GB（H200 加的）。
  只看 (a) 不夠：跑中的工作記憶體會漲會退（逐題載入格式時漲、題目之間放掉），剛好退下來的時候別的工作被放進來，
  它下一題再漲就 OOM。job 20705（10/18 02:50）就是這樣：Llama 252 個 block 的試跑在 52 GB 時被放進兩個 Qwen，
  下一題載入格式 OOM。MI300X 192 GB 沒遇到；H200 141 GB 會。
- 開哪一個：照 jobs.json 的順序，第一個放得下的（前面的放不下時，後面較小的可以先開）。
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


def total_gb() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", "0"],
                         capture_output=True, text=True, timeout=60)
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(f"nvidia-smi 查不到總記憶體（rc={out.returncode}）：{out.stderr.strip()!r}")
    return float(out.stdout.strip().splitlines()[0]) / 1024


def main():
    jobs = json.load(open(sys.argv[1]))
    P = int(sys.argv[2])
    running, failed, done = {}, [], []
    pending = list(jobs)
    total = total_gb()
    too_big = [j["name"] for j in jobs if j["need_gb"] + MARGIN > total]
    if too_big:
        sys.exit(f"[pool] 這些工作的 need_gb＋{MARGIN} 超過整張卡（{total:.0f} GB），永遠開不了：{too_big}")
    need = {}
    last_launch = 0.0
    while pending or running:
        for n, (p, log) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                log.close()
                del running[n]
                del need[n]
                (done if rc == 0 else failed).append((n, rc))
                print(f"[pool] {'done' if rc == 0 else 'FAILED'} {n} rc={rc} {time.strftime('%H:%M:%S')}", flush=True)
        if failed:
            pending = []
        if pending and len(running) < P and time.time() - last_launch > GAP_S:
            fg, left = free_gb(), total - sum(need.values())
            j = next((j for j in pending if fg >= j["need_gb"] + MARGIN and left >= j["need_gb"] + MARGIN), None)
            if j is not None:
                pending.remove(j)
                need[j["name"]] = j["need_gb"]
                log = open(f"pool_{j['name']}.log", "w")
                running[j["name"]] = (subprocess.Popen(["bash", "-c", ENV + j["cmd"]], stdout=log,
                                                       stderr=subprocess.STDOUT), log)
                last_launch = time.time()
                print(f"[pool] start {j['name']} need={j['need_gb']}GB free={fg:.0f}GB reserved={sum(need.values())}/{total:.0f}GB "
                      f"running={len(running)} {time.strftime('%H:%M:%S')}",
                      flush=True)
        time.sleep(5)
    print(f"[pool] finished done={len(done)} failed={failed}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
