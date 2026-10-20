"""x7_busy.py — X7：3090 上「GPU 忙」讓重算慢幾倍（第一階段 C5 的定義），另記背景負載實際做到的 duty。

為什麼另寫（不改 m7_write_policy.busy_calib）：
  1. busy_calib 只量 block 32、只傳 --duty 給 m7_busy.py（週期 20 ms、8192² matmul 寫死成預設）。MI300X 上一次 8192² BF16
     matmul 多數 1.65–1.67 ms（m10 dqsweep busy JSON 的 READY 行），duty 的粒度夠細；3090 上一次 matmul 慢一個數量級，「做 duty×週期 的 matmul」會整個 matmul 超出，
     實際 duty 不是設定值。所以每個設定都另量「只有背景在跑時」nvidia-smi 的 utilization.gpu（kernel 在跑的時間比例），
     並記 m7_busy 自己印的 matmul_ms、DONE 的次數與秒數。
  2. 結果寫進 run 目錄，不寫 results/m7_write_policy_*（busy_calib 會寫到 $M7_RES/calib_c5.csv）。

  python code/p3x/x7_busy.py --model qwen --out $RUN_DIR/busy.csv --cfg 0.7:20:8192 0.4:20:8192 0.7:200:8192 0.7:20:4096
cfg＝duty:period_ms:n（m7_busy.py 的 --duty --period-ms --n）。0.7:20:8192＝MI300X 用的設定（m10_speed／m7_write_policy 只傳 --duty）。

量法（同 busy_calib）：prefill 64 個 block；每個位置 i∈--chunks，forward_span(block i) 量 10 次取中位數＝一個 rep；
GPU 閒的基準先量；每個背景設定 3 個 rep。slowdown＝忙的 rep ÷ 同位置閒的中位數。
污染：本程式不另開 GpuWatcher（外層 run3090.sh → m7_guard_run.py 已經包住整個 process tree；背景 m7_busy 是子孫，白名單）。
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)

import torch  # noqa: E402

from gpu_guard import smi_index  # noqa: E402


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


class UtilSampler:
    """nvidia-smi utilization.gpu，每 200 ms 一筆（查詢失敗照實記，不當成 0：規則 7）。"""

    def __init__(self, idx):
        self.idx, self.vals, self.err = idx, [], None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "-i", str(self.idx), "--query-gpu=utilization.gpu",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                if out.returncode != 0:
                    raise RuntimeError(out.stderr.strip())
                self.vals.append(int(out.stdout.strip()))
            except Exception as ex:  # noqa: BLE001
                self.err = repr(ex)
            self._stop.wait(0.2)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["llama", "qwen"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", nargs="+", default=["0.7:20:8192"])
    ap.add_argument("--chunks", type=int, nargs="+", default=[0, 32, 63])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--inner", type=int, default=10)
    ap.add_argument("--bg-alone-s", type=float, default=4.0)
    a = ap.parse_args()
    os.environ["M7_MODEL_GLOB"] = os.environ[f"M7_{a.model.upper()}_GLOB"]
    from m7_model import CHUNK, KVModel
    rid = os.environ.get("RUN_ID", "NO_RUN_ID")
    gidx = smi_index(0)
    km = KVModel(max_len=64 * CHUNK + 512)
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(1000, min(120000, km.cfg.vocab_size), (64 * CHUNK + 512,), generator=g).to(km.device)
    km.prefill_chunked(ids[:64 * CHUNK])
    torch.cuda.synchronize()

    def meas(i):
        r = []
        for _ in range(a.inner):
            torch.cuda.synchronize()
            t = time.perf_counter()
            km.forward_span(ids[i * CHUNK:(i + 1) * CHUNK], i * CHUNK)
            torch.cuda.synchronize()
            r.append(time.perf_counter() - t)
        return statistics.median(r)

    fields = ["run_id", "ts", "model", "gpu_smi_index", "cfg", "duty", "period_ms", "n_mat", "matmul_ms_ready",
              "bg_alone_util_mean", "bg_alone_util_n", "bg_util_err", "chunk_idx", "rep", "ms", "idle_median_ms", "slowdown",
              "bg_n_matmul", "bg_seconds", "busy_pid"]
    new = not os.path.exists(a.out)
    f = open(a.out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    if new:
        w.writeheader()
    for i in a.chunks:                                     # 暖機
        meas(i)
    idle = {i: [meas(i) for _ in range(a.reps)] for i in a.chunks}
    idle_med = {i: statistics.median(v) for i, v in idle.items()}
    for i, v in idle.items():
        for rep, m in enumerate(v):
            w.writerow(dict(run_id=rid, ts=ts(), model=a.model, gpu_smi_index=gidx, cfg="idle", duty=0.0, chunk_idx=i, rep=rep,
                            ms=round(m * 1e3, 3), idle_median_ms=round(idle_med[i] * 1e3, 3),
                            slowdown=round(m / idle_med[i], 4)))
    f.flush()
    for cfg in a.cfg:
        duty, period, n = cfg.split(":")
        p = subprocess.Popen([sys.executable, os.path.join(CODE, "m7_busy.py"), "--duty", duty, "--period-ms", period,
                              "--n", n], stdout=subprocess.PIPE, text=True)
        line = p.stdout.readline()
        if not line.startswith("READY"):
            p.kill()
            raise SystemExit(f"m7_busy 沒有啟動（{cfg}）：{line!r}")
        mm = float(line.split("matmul_ms=")[1])
        time.sleep(1.0)
        with UtilSampler(gidx) as us:                     # 只有背景在跑（本 process 不送 kernel）
            time.sleep(a.bg_alone_s)
        util = round(statistics.mean(us.vals), 2) if us.vals else "NOT_MEASURED"
        rows = []
        for i in a.chunks:
            for rep in range(a.reps):
                m = meas(i)
                rows.append(dict(chunk_idx=i, rep=rep, ms=round(m * 1e3, 3), idle_median_ms=round(idle_med[i] * 1e3, 3),
                                 slowdown=round(m / idle_med[i], 4)))
        p.terminate()
        out, _ = p.communicate(timeout=60)
        nmat = secs = "NOT_MEASURED"
        for ln in out.splitlines():
            if ln.startswith("DONE"):
                kv = dict(x.split("=") for x in ln.split()[1:])
                nmat, secs = int(kv["n"]), float(kv["s"])
        for r in rows:
            w.writerow(dict(run_id=rid, ts=ts(), model=a.model, gpu_smi_index=gidx, cfg=cfg, duty=float(duty),
                            period_ms=float(period), n_mat=int(n), matmul_ms_ready=mm, bg_alone_util_mean=util,
                            bg_alone_util_n=len(us.vals), bg_util_err=us.err or "", bg_n_matmul=nmat, bg_seconds=secs,
                            busy_pid=p.pid, **r))
        f.flush()
        print(cfg, "matmul_ms", mm, "bg_util", util, [(r["chunk_idx"], r["slowdown"]) for r in rows], flush=True)
        time.sleep(2.0)
    f.close()


if __name__ == "__main__":
    main()
