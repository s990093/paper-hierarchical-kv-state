"""m7_c6_vllm.py — C6：真實系統對照（vLLM v0.28 ＋ OffloadingConnector CPU 層，只載入）

同一個 L，量：
  cold  ＝ 第一次送 prompt（整段 prefill，同時寫入 CPU 層）
  warm  ＝ reset GPU prefix cache 後再送同一個 prompt（從 CPU 層載入，只算最後不滿一個 block 的部分）
和 harness 的 compute_only、load_only（CPU 真實 H2D）對照。
TTFT 用 streaming、max_tokens=1，量到第一個 chunk 回來。
"""
import argparse
import csv
import json
import os
import random
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

VENV = "/mlsteam/workspace/venv/tiara-v028"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m7_model import model_path  # noqa: E402


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def post(url, body, stream=False):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        if stream:
            for line in r:
                if line.startswith(b"data:") and b"[DONE]" not in line:
                    return time.perf_counter() - t0
        r.read()
    return time.perf_counter() - t0


def metrics(port):
    txt = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10).read().decode()
    return {l.split(" ")[0]: l.split(" ")[-1] for l in txt.splitlines() if "offload" in l and not l.startswith("#")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--L", type=int, nargs="*", default=[8192, 16384, 32768])
    ap.add_argument("--reps", type=int, default=6)
    a = ap.parse_args()
    rd = os.path.join(os.environ.get("TIARA_RUNS", "/mlsteam/data/tiara/runs"), os.environ.get("RUN_ID", "adhoc"))
    os.makedirs(rd, exist_ok=True)
    port = free_port()
    cfg = {"kv_connector": "OffloadingConnector", "kv_role": "kv_both",
           "kv_connector_extra_config": {"spec_name": "CPUOffloadingSpec", "cpu_bytes_to_use": 48 * 2**30,
                                         "eviction_policy": "lru"}}
    cmd = [f"{VENV}/bin/vllm", "serve", model_path(), "--port", str(port), "--max-model-len", "40960",
           "--gpu-memory-utilization", "0.5", "--kv-transfer-config", json.dumps(cfg), "--served-model-name", "m"]
    open(os.path.join(rd, "vllm_cmd.txt"), "w").write(" ".join(shlex.quote(c) for c in cmd))
    env = dict(os.environ, VLLM_SERVER_DEV_MODE="1", PATH=f"{VENV}/bin:/opt/rocm/bin:" + os.environ["PATH"])
    log = open(os.path.join(rd, "vllm_server.log"), "w")
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        for _ in range(900):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2); break
            except Exception:
                if p.poll() is not None:
                    raise RuntimeError("vllm exited; see vllm_server.log")
                time.sleep(2)
        else:
            raise RuntimeError("vllm did not become healthy")
        new = not os.path.exists(a.out)
        f = open(a.out, "a", newline="")
        w = csv.writer(f)
        if new:
            w.writerow(["run_id", "ts", "L", "rep", "kind", "ttft_s", "offload_metrics"])
        base = f"http://127.0.0.1:{port}"
        rng = random.Random(0)
        # 暖機
        post(base + "/v1/completions", {"model": "m", "prompt": [rng.randrange(1000, 120000) for _ in range(512)],
                                        "max_tokens": 1, "stream": True}, stream=True)
        for L in a.L:
            for rep in range(a.reps):
                ids = [rng.randrange(1000, 120000) for _ in range(L)]
                body = {"model": "m", "prompt": ids, "max_tokens": 1, "temperature": 0, "stream": True}
                t_cold = post(base + "/v1/completions", body, stream=True)
                time.sleep(1.0)     # 讓背景的 GPU→CPU 寫入完成
                post(base + "/reset_prefix_cache", {})
                m0 = metrics(port)
                t_warm = post(base + "/v1/completions", body, stream=True)
                m1 = metrics(port)
                dm = {k: (float(m1[k]) - float(m0.get(k, 0))) for k in m1 if k in m0 or True}
                dm = {k: v for k, v in dm.items() if v}
                ts = datetime.now().astimezone().isoformat(timespec="seconds")
                rid = os.environ.get("RUN_ID")
                w.writerow([rid, ts, L, rep, "cold_compute", round(t_cold, 5), ""])
                w.writerow([rid, ts, L, rep, "warm_cpu_load", round(t_warm, 5), json.dumps(dm)[:1500]])
                f.flush()
                print(L, rep, round(t_cold, 4), round(t_warm, 4), json.dumps(dm)[:300], flush=True)
                post(base + "/reset_prefix_cache", {})
    finally:
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(60)
        except Exception:
            os.killpg(p.pid, signal.SIGKILL)
        # /dev/shm 的殘留（RUNLOG_MI300X 發現 7、16）
        subprocess.run("ls /dev/shm | head", shell=True)


if __name__ == "__main__":
    main()
