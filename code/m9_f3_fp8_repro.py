#!/usr/bin/env python
"""m9_f3_fp8_repro.py — F3 工作 B：九月品質 run 的 FP8 KV 到底有沒有生效？

做法：用九月同一個 vLLM（venv tiara-v028 = vLLM 0.28.0）、同一個模型、同樣的 server 參數，
依序開 auto（兩次）、fp8、fp8_e4m3 四個 server，每個都送同一批長 prompt（token id 完全相同），
greedy 生成 32 個 token，記錄：
  * server.log 裡的 KV 容量（GPU KV cache size: N tokens）、可用 KV 記憶體、attention backend、
    "Using fp8 data type to store kv cache"；
  * 每個生成 token 的 id 與 top-1 logprob（greedy → 選中的就是 top-1）。
然後逐 token 位元比較。

用法（一定要包 GPU 鎖與 guard）：
  m7run f3-fp8 flock /mlsteam/data/tiara/gpu.lock <py> code/m7_guard_run.py <py> code/m9_f3_fp8_repro.py --out-dir <repo>/results/m9_followup
"""
import argparse
import csv
import json
import os
import random
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RUN_ID = os.environ.get("RUN_ID", "no-run-id")
RUN_DIR = os.path.join(os.environ.get("TIARA_RUNS", "/mlsteam/data/tiara/runs"), RUN_ID)
TS = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", required=True)
ap.add_argument("--venv", default="/mlsteam/workspace/venv/tiara-v028")
ap.add_argument("--model", default="unsloth/Llama-3.1-8B-Instruct")
ap.add_argument("--max-model-len", type=int, default=129536)
ap.add_argument("--configs", default="A1:auto,A2:auto,F1:fp8,F2:fp8_e4m3")
ap.add_argument("--lens", default="32768,49152,65536,65536,122880")
ap.add_argument("--depths", default="0.5,0.3,0.25,0.75,0.5")
ap.add_argument("--max-tokens", type=int, default=32)
ap.add_argument("--n-judge", type=int, default=8, help="判定用前幾個生成 token")
args = ap.parse_args()
OUT_DIR = os.path.abspath(args.out_dir)
os.makedirs(RUN_DIR, exist_ok=True)
os.chdir(RUN_DIR)   # ROCm 當掉時的 gpucore.* 留在 run 目錄

HF_HOME = os.environ.get("HF_HOME", "/mlsteam/data/tiara/hf-cache")


# ───────────────────────────── prompt ─────────────────────────────

WORDS = ("river mountain lantern harvest copper meadow whisper engine orchard signal granite violet "
         "harbor ledger compass thunder pepper glacier cabinet saddle marble fabric quarry beacon "
         "tunnel feather canyon velvet anchor walnut pigeon kettle summit parcel ribbon tavern ember "
         "falcon prairie crystal bramble shutter pebble lagoon timber barrel cobalt hollow orbit").split()


def filler_sentence(rng):
    n = rng.randint(8, 16)
    w = [rng.choice(WORDS) for _ in range(n)]
    return " ".join(w).capitalize() + "."


def build_prompts(tok):
    lens = [int(x) for x in args.lens.split(",")]
    depths = [float(x) for x in args.depths.split(",")]
    assert len(lens) == len(depths)
    out = []
    for i, (n, d) in enumerate(zip(lens, depths)):
        rng = random.Random(20261010 + i)
        city = ["Taipei", "Lisbon", "Nairobi", "Oslo", "Quito", "Hanoi"][i % 6]
        code = str(rng.randint(1_000_000, 9_999_999))
        header = tok.encode(f"Document {i} (archive {rng.randint(10**5, 10**6)}). Read carefully; one line hides a number.\n",
                            add_special_tokens=False)
        needle = tok.encode(f" The special magic number for {city} is {code}. ", add_special_tokens=False)
        question = tok.encode(f"\n\nQuestion: What is the special magic number for {city}?\n"
                              f"Answer: The special magic number for {city} is", add_special_tokens=False)
        body_n = n - 1 - len(header) - len(needle) - len(question)
        sents = []
        ids = []
        while len(ids) < body_n:
            chunk = " ".join(filler_sentence(rng) for _ in range(200))
            ids += tok.encode(" " + chunk, add_special_tokens=False)
        ids = ids[:body_n]
        pos = int(len(ids) * d)
        full = [tok.bos_token_id] + header + ids[:pos] + needle + ids[pos:] + question
        assert len(full) == n, (len(full), n)
        out.append({"idx": i, "target_len": n, "depth": d, "city": city, "code": code,
                    "ids": full, "needle_pos": 1 + len(header) + pos})
        del sents
    return out


# ───────────────────────────── server ─────────────────────────────

def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def parse_log(text):
    g = lambda pat: (re.search(pat, text).group(1) if re.search(pat, text) else None)  # noqa: E731
    kv_tokens = g(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
    return {
        "vllm_version": g(r"version (\d+\.\d+\.\d+\S*)"),
        "kv_tokens": int(kv_tokens.replace(",", "")) if kv_tokens else None,
        "kv_mem_gib": float(g(r"Available KV cache memory:\s*([\d.]+)\s*GiB") or "nan"),
        "backend": g(r"Overriding with (\w+)") or g(r"Using (\w+) backend"),
        "backend_line": g(r"(\[rocm\.py:\d+\][^\n]*backend[^\n]*)"),
        "fp8_store_msg": bool(re.search(r"Using fp8 data type to store kv cache", text)),
        "engine_kv_cache_dtype": g(r"kv_cache_dtype=(\w+)"),
    }


class Server:
    def __init__(self, name, kv_dtype):
        self.name, self.kv_dtype = name, kv_dtype
        self.dir = os.path.join(RUN_DIR, name)
        os.makedirs(self.dir, exist_ok=True)
        self.port = free_port()

    def __enter__(self):
        cmd = [os.path.join(args.venv, "bin/vllm"), "serve", args.model, "--port", str(self.port),
               "--max-model-len", str(args.max_model_len), "--gpu-memory-utilization", "0.90"]
        if self.kv_dtype != "auto":
            cmd += ["--kv-cache-dtype", self.kv_dtype]
        with open(os.path.join(self.dir, "cmd.txt"), "w") as f:
            f.write(" ".join(cmd) + "\n")
        e = dict(os.environ)
        e.update({"HIP_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": "0", "HF_HOME": HF_HOME,
                  "PATH": f"{args.venv}/bin:/opt/rocm/bin:" + e.get("PATH", "")})
        self.log = open(os.path.join(self.dir, "server.log"), "w")
        self.p = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=e,
                                  cwd=self.dir, start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < 900:
            if self.p.poll() is not None:
                raise RuntimeError(f"server {self.name} died rc={self.p.returncode}; see {self.dir}/server.log")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2)
                self.info = parse_log(open(os.path.join(self.dir, "server.log"), errors="replace").read())
                self.info["startup_s"] = round(time.time() - t0, 1)
                return self
            except Exception:  # noqa: BLE001
                time.sleep(2)
        raise TimeoutError(f"server {self.name} not ready")

    def ask(self, ids):
        body = json.dumps({"model": args.model, "prompt": ids, "max_tokens": args.max_tokens,
                           "temperature": 0.0, "seed": 12345, "logprobs": 1,
                           "return_tokens_as_token_ids": True}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=1800) as r:
            js = json.load(r)
        js["_latency_s"] = time.time() - t0
        return js

    def __exit__(self, *exc):
        try:
            if self.p.poll() is None:
                os.killpg(os.getpgid(self.p.pid), signal.SIGTERM)
                self.p.wait(timeout=120)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(os.getpgid(self.p.pid), signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
        self.log.close()
        try:
            from gpu_guard import wait_until_released
            ok, left = wait_until_released(0, timeout_s=300)
            print(f"[f3b] {self.name}: GPU released={ok} left_mib={left}", flush=True)
        except Exception:  # noqa: BLE001
            print(f"[f3b] wait_until_released failed:\n{traceback.format_exc()}", flush=True)
            time.sleep(15)


# ───────────────────────────── main ─────────────────────────────

def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = build_prompts(tok)
    with open(os.path.join(RUN_DIR, "prompts_meta.json"), "w") as f:
        json.dump([{k: v for k, v in p.items() if k != "ids"} for p in prompts], f, indent=1)
    print(f"[f3b] prompts: {[len(p['ids']) for p in prompts]}", flush=True)

    configs = [c.split(":") for c in args.configs.split(",")]
    results = {}   # name -> {"info":..., "gens":[...]}
    gen_rows = []
    for name, dt in configs:
        print(f"[f3b] === {name} kv_cache_dtype={dt}", flush=True)
        with Server(name, dt) as s:
            print(f"[f3b]   info {json.dumps(s.info)}", flush=True)
            gens = []
            for p in prompts:
                js = s.ask(p["ids"])
                ch = js["choices"][0]
                lp = ch["logprobs"]
                toks = [int(t.split(":")[1]) for t in lp["tokens"]]
                gens.append({"idx": p["idx"], "text": ch["text"], "token_ids": toks,
                             "token_logprobs": lp["token_logprobs"],
                             "prompt_tokens": js["usage"]["prompt_tokens"], "latency_s": js["_latency_s"]})
                found = p["code"] in ch["text"]
                print(f"[f3b]   p{p['idx']} len={js['usage']['prompt_tokens']} found={found} "
                      f"text={ch['text'][:60]!r} lp0..3={lp['token_logprobs'][:4]}", flush=True)
                gen_rows.append({"run_id": RUN_ID, "ts": TS, "config": name, "kv_cache_dtype": dt,
                                 "prompt_idx": p["idx"], "prompt_tokens": js["usage"]["prompt_tokens"],
                                 "needle_depth": p["depth"], "needle_found": found,
                                 "kv_tokens": s.info["kv_tokens"], "kv_mem_gib": s.info["kv_mem_gib"],
                                 "backend": s.info["backend"], "fp8_store_msg": s.info["fp8_store_msg"],
                                 "engine_kv_cache_dtype": s.info["engine_kv_cache_dtype"],
                                 "vllm_version": s.info["vllm_version"],
                                 "first8_token_ids": " ".join(map(str, toks[:args.n_judge])),
                                 "first8_logprobs": " ".join(repr(x) for x in lp["token_logprobs"][:args.n_judge]),
                                 "text": ch["text"].replace("\n", "\\n")[:200]})
            results[name] = {"dtype": dt, "info": s.info, "gens": gens}
        with open(os.path.join(RUN_DIR, "f3b_results.json"), "w") as f:
            json.dump(results, f, indent=1)

    # ── 比較：每個 config 對 A1 ──
    base = results[configs[0][0]]
    cmp_rows = []
    for name, dt in configs[1:]:
        r = results[name]
        for gb, gr in zip(base["gens"], r["gens"]):
            n = args.n_judge
            ids_same = gb["token_ids"] == gr["token_ids"]
            first_div = next((k for k, (a, b) in enumerate(zip(gb["token_ids"], gr["token_ids"])) if a != b), None)
            # logprob 只在 token 相同的位置比（token 不同，logprob 本來就不能比）
            m = min(n, first_div if first_div is not None else n)
            diffs = [abs(a - b) for a, b in zip(gb["token_logprobs"][:m], gr["token_logprobs"][:m])]
            bit_same = all(a == b for a, b in zip(gb["token_logprobs"][:m], gr["token_logprobs"][:m]))
            cmp_rows.append({"run_id": RUN_ID, "ts": TS, "base": configs[0][0], "config": name,
                             "kv_cache_dtype": dt, "prompt_idx": gb["idx"],
                             "prompt_tokens": gb["prompt_tokens"],
                             "kv_tokens_base": base["info"]["kv_tokens"], "kv_tokens": r["info"]["kv_tokens"],
                             "kv_ratio": round(r["info"]["kv_tokens"] / base["info"]["kv_tokens"], 4)
                             if r["info"]["kv_tokens"] and base["info"]["kv_tokens"] else None,
                             "text_identical": gb["text"] == gr["text"],
                             "token_ids_identical_all32": ids_same,
                             "first_divergent_token": first_div if first_div is not None else "",
                             "n_logprobs_compared": m,
                             "logprobs_bit_identical_firstN": bit_same,
                             "max_abs_logprob_diff_firstN": max(diffs) if diffs else "",
                             "logprob_diffs_firstN": " ".join(f"{x:.3e}" for x in diffs)})
    for fn, rows in (("f3_fp8_repro_gens.csv", gen_rows), ("f3_fp8_repro_compare.csv", cmp_rows)):
        for d in (OUT_DIR, RUN_DIR):
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, fn), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    for r in cmp_rows:
        print(f"[f3b] {r['config']} vs {r['base']} p{r['prompt_idx']} ({r['prompt_tokens']} tok): "
              f"kv_ratio={r['kv_ratio']} text_same={r['text_identical']} ids_same={r['token_ids_identical_all32']} "
              f"first_div={r['first_divergent_token']} lp_bit_same(first{r['n_logprobs_compared']})="
              f"{r['logprobs_bit_identical_firstN']} max|dlp|={r['max_abs_logprob_diff_firstN']}", flush=True)
    return 0


if __name__ == "__main__":
    t0 = time.time()
    try:
        rc = main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        rc = 1
    print(f"[f3b] wall {time.time() - t0:.1f} s", flush=True)
    sys.exit(rc)
