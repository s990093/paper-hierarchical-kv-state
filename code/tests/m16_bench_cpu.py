#!/usr/bin/env python3
"""m16_bench_cpu.py — m16_vllm_bench.py 的 CPU 端到端測試（假的 OpenAI 相容 streaming server，不碰 GPU）。

假 server（同一支檔案 --fake-server 起在子行程，aiohttp）：
  - /v1/completions：token-id 或文字 prompt、stream、stream_options.include_usage、return_token_ids、logprobs；
    TTFT＝ttft_base＋prefill_per_tok×(沒命中的 token)＋load_per_tok×(從 CPU／fs 載入的 token)，之後每 tpot 秒一個 token
    （絕對時間排程，不累積誤差）。輸出 token 由 prompt 的 sha256 決定（所以命中與否輸出相同；corrupt_cpu 時故意改掉）。
  - 16-token block 的前綴快取（GPU LRU，可設容量）＋CPU／fs 層；「命中的 block 正在被別的請求載入」時延後並寫一行
    和 vLLM 0.28 一樣的 DEBUG 訊息（offloading/scheduler.py:855-856），給 F2 的計數測試用。
  - /metrics：vLLM 0.28 的名稱（counter 加 _total）；offload 家族像 prometheus_client 一樣「第一次觀測才有樣本」；
    metrics_mode＝no_offload（offload 家族整個不存在）／404。
  - /reset_prefix_cache?reset_external=…（reset_external 只清 CPU、不清 fs，同 tiering/manager.py:821-831）。
  - 每個請求寫一行 JSON log（伺服器端的收到時間、命中、預定 TTFT／TPOT、完成時間）。

全部子行程都帶 M16_FORBID_GPU_IMPORTS=1（import torch／vllm 會直接失敗），--guard off 或替身，不呼叫 amd-smi。
用法：nice -n 19 taskset -c 0-3 /mlsteam/workspace/venv/tiara-v028/bin/python code/tests/m16_bench_cpu.py
印 ALL PASS 或失敗明細（結束碼 0／1）。
"""
from __future__ import annotations

import asyncio
import collections
import csv
import glob
import hashlib
import json
import os
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request

os.environ["M16_FORBID_GPU_IMPORTS"] = "1"
CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
import m16_workload as W  # noqa: E402  （裝上 import 阻擋器）

PY = sys.executable
BENCH = os.path.join(CODE, "m16_vllm_bench.py")
WL = os.path.join(CODE, "m16_workload.py")

FAKE_DEFAULT = dict(ttft_base=0.15, prefill_per_tok=0.0004, load_per_tok=0.0002, tpot=0.02, block=16,
                    gpu_cap_blocks=None, apc=True, offload="cpu", metrics_mode="full", corrupt_cpu=False,
                    prompt_details=True, max_model_len=40960, bpt=131072, fail_ids=[], reset_fail_n=0,
                    evict_after_warm_s=None, log_path=None, server_log=None)


# ═════════════════════════════════ 假 server ═════════════════════════════════
class Fake:
    def __init__(self):
        self.configure({})

    def configure(self, cfg):
        self.cfg = dict(FAKE_DEFAULT, **cfg)
        self.gpu: collections.OrderedDict = collections.OrderedDict()
        self.cpu: set = set()
        self.fs: set = set()
        self.loading: dict = {}
        self.c: collections.Counter = collections.Counter()
        self.observed: set = set()
        self.reset_calls = 0

    def slog(self, msg, line):
        p = self.cfg["server_log"]
        if p:
            with open(p, "a") as f:
                f.write(f"(EngineCore pid=1) DEBUG {time.strftime('%m-%d %H:%M:%S')} [scheduler.py:{line}] {msg}\n")

    def rlog(self, rec):
        p = self.cfg["log_path"]
        if p:
            with open(p, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def gpu_insert(self, keys):
        cap = self.cfg["gpu_cap_blocks"]
        for d in keys:
            if d in self.gpu:
                self.gpu.move_to_end(d)
            else:
                self.gpu[d] = True
            while cap is not None and len(self.gpu) > cap:
                self.gpu.popitem(last=False)

    def inc(self, k, v):
        self.c[k] += v
        self.observed.add(k)

    async def completions(self, request):
        from aiohttp import web
        m0 = time.perf_counter()
        t_recv = time.time()
        body = await request.json()
        cfg = self.cfg
        rid = body.get("request_id", "noid")
        if any(f in rid for f in cfg["fail_ids"]):
            self.rlog(dict(rid=rid, t_recv=t_recv, failed=True))
            return web.json_response({"error": "fake failure"}, status=500)
        p = body["prompt"]
        toks = p if isinstance(p, list) else [1000 + (ord(ch) % 50000) for ch in p]
        L, B = len(toks), cfg["block"]
        mt = int(body.get("max_tokens") or 16)
        if L + mt > cfg["max_model_len"]:
            return web.json_response({"error": "too long"}, status=400)
        ds = W._chain_digests(toks, B)
        polls = 0
        while True:
            kg = 0
            if cfg["apc"]:
                while kg < len(ds) and ds[kg] in self.gpu:
                    kg += 1
            ke = kg
            if cfg["apc"] and cfg["offload"] != "none":
                while ke < len(ds) and (ds[ke] in self.cpu or ds[ke] in self.fs):
                    ke += 1
            if not any(d in self.loading for d in ds[kg:ke]):
                break
            polls += 1
            self.slog(f"Delaying request {rid} since some of its chunks are already being loaded", 855)
            await asyncio.sleep(0.02)
        cap = ((L - 1) // B) * B
        cached_gpu = min(kg * B, cap)
        cached = min(ke * B, cap)
        ext_tok = cached - cached_gpu
        keys = ds[kg:kg + ext_tok // B]
        from_fs = sum(1 for d in keys if d not in self.cpu)
        for d in ds[:kg]:
            self.gpu.move_to_end(d)
        load_t = cfg["load_per_tok"] * ext_tok
        if ext_tok:
            self.slog(f"Request {rid} hit {ext_tok} offloaded tokens after {cached_gpu} GPU hit tokens", 860)
            for d in keys:
                self.loading[d] = True

            async def finish():
                await asyncio.sleep(load_t)
                for d in keys:
                    self.loading.pop(d, None)
                self.gpu_insert(keys)
            asyncio.ensure_future(finish())
        t_first = time.perf_counter() + cfg["ttft_base"] + cfg["prefill_per_tok"] * (L - cached) + load_t
        intended_ttft = t_first - m0
        # 計數（vLLM 0.28 的語意）
        if cfg["apc"]:
            self.inc("vllm:prefix_cache_queries", L)
            self.inc("vllm:prefix_cache_hits", cached_gpu)
        if cfg["offload"] != "none" and cfg["apc"]:
            self.inc("vllm:external_prefix_cache_queries", L - cached_gpu)
            self.inc("vllm:external_prefix_cache_hits", ext_tok)
        self.inc("vllm:prompt_tokens", L)
        self.inc("vllm:prompt_tokens_cached", cached)
        self.inc("src:local_compute", L - cached)
        self.inc("src:local_cache_hit", cached_gpu)
        self.inc("src:external_kv_transfer", ext_tok)
        if ext_tok:
            self.inc("vllm:kv_offload_load_bytes", ext_tok * cfg["bpt"])
            self.inc("vllm:kv_offload_load_time", load_t)
            if from_fs:
                self.inc("tier:read", from_fs * B * cfg["bpt"])
        hsh = hashlib.sha256(json.dumps(toks).encode()).digest()
        out = [1000 + int.from_bytes(hashlib.sha256(hsh + bytes([i % 256, i // 256])).digest()[:4], "little") % 100000
               for i in range(mt)]
        if cfg["corrupt_cpu"] and ext_tok:
            out[0] += 1
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        t_first_wall = None
        for i, tk in enumerate(out):
            d = t_first + i * cfg["tpot"] - time.perf_counter()
            if d > 0:
                await asyncio.sleep(d)
            if i == 0:
                t_first_wall = time.time()
                if cfg["apc"]:
                    self.gpu_insert(ds)
                if cfg["offload"] != "none" and cfg["apc"]:
                    new = [x for x in ds if x not in self.cpu]
                    self.cpu.update(ds)
                    self.inc("vllm:kv_offload_store_bytes", len(new) * B * cfg["bpt"])
                    self.inc("vllm:kv_offload_store_time", 0.001 * len(new))
                    if cfg["offload"] == "tier":
                        newfs = [x for x in ds if x not in self.fs]
                        self.fs.update(ds)
                        self.inc("tier:write", len(newfs) * B * cfg["bpt"])
            ch = {"index": 0, "text": f"t{tk}", "logprobs": None, "finish_reason": "length" if i == mt - 1 else None}
            if body.get("return_token_ids"):
                ch["token_ids"] = [tk]
                if i == 0:
                    ch["prompt_token_ids"] = toks
            if body.get("logprobs") is not None:
                ch["logprobs"] = {"tokens": [f"token_id:{tk}"], "token_logprobs": [-0.1],
                                  "top_logprobs": [{f"token_id:{tk}": -0.1, f"token_id:{tk + 1}": -2.6}]}
            await resp.write(("data: " + json.dumps({"id": rid, "object": "text_completion", "model": "m",
                                                     "choices": [ch]}) + "\n\n").encode())
        self.inc("vllm:generation_tokens", mt)
        if (body.get("stream_options") or {}).get("include_usage"):
            u = {"prompt_tokens": L, "completion_tokens": mt, "total_tokens": L + mt}
            if cfg["prompt_details"]:
                u["prompt_tokens_details"] = {"cached_tokens": cached}
            await resp.write(("data: " + json.dumps({"id": rid, "choices": [], "usage": u}) + "\n\n").encode())
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        self.rlog(dict(rid=rid, t_recv=t_recv, t_first=t_first_wall, t_done=time.time(), L=L, max_tokens=mt,
                       cached=cached, cached_gpu=cached_gpu, ext_tok=ext_tok, intended_ttft=intended_ttft,
                       intended_tpot=cfg["tpot"], polls=polls))
        if cfg["evict_after_warm_s"] is not None and rid.endswith("-warm"):
            async def clear():
                await asyncio.sleep(cfg["evict_after_warm_s"])
                self.gpu.clear()
            asyncio.ensure_future(clear())
        return resp

    def render(self):
        cfg, c = self.cfg, self.c
        lab = 'engine="0",model_name="m"'
        out = []

        def fam(name, typ, samples):
            out.append(f"# HELP {name} fake")
            out.append(f"# TYPE {name} {typ}")
            for sname, extra, v in samples:
                out.append(f"{sname}{{{lab}{(',' + extra) if extra else ''}}} {float(v)}")
        for b in ("vllm:prefix_cache_queries", "vllm:prefix_cache_hits", "vllm:external_prefix_cache_queries",
                  "vllm:external_prefix_cache_hits", "vllm:prompt_tokens", "vllm:prompt_tokens_cached",
                  "vllm:generation_tokens", "vllm:num_preemptions"):
            fam(b + "_total", "counter", [(b + "_total", "", c[b])])
        fam("vllm:prompt_tokens_by_source_total", "counter",
            [("vllm:prompt_tokens_by_source_total", f'source="{s}"', c["src:" + s])
             for s in ("local_compute", "local_cache_hit", "external_kv_transfer")])
        if cfg["offload"] != "none" and cfg["metrics_mode"] != "no_offload":
            for b in ("vllm:kv_offload_load_bytes", "vllm:kv_offload_load_time", "vllm:kv_offload_store_bytes",
                      "vllm:kv_offload_store_time", "vllm:kv_offload_allocation_failure"):
                fam(b + "_total", "counter", [(b + "_total", "", c[b])] if b in self.observed else [])
            fam("vllm:kv_offload_total_bytes_total", "counter",
                [("vllm:kv_offload_total_bytes_total", 'transfer_type="CPU_to_GPU"', c["vllm:kv_offload_load_bytes"]),
                 ("vllm:kv_offload_total_bytes_total", 'transfer_type="GPU_to_CPU"', c["vllm:kv_offload_store_bytes"])]
                if "vllm:kv_offload_load_bytes" in self.observed or "vllm:kv_offload_store_bytes" in self.observed else [])
            fam("vllm:kv_offload_lookup_async_delay_seconds", "histogram", [])
            if cfg["offload"] == "tier":
                fam("vllm:kv_offload_tiering_read_bytes_total", "counter",
                    [("vllm:kv_offload_tiering_read_bytes_total", 'tier="fs"', c["tier:read"])] if "tier:read" in self.observed else [])
                fam("vllm:kv_offload_tiering_write_bytes_total", "counter",
                    [("vllm:kv_offload_tiering_write_bytes_total", 'tier="fs"', c["tier:write"])] if "tier:write" in self.observed else [])
        return "\n".join(out) + "\n"


def fake_main(port: int):
    from aiohttp import web
    fk = Fake()
    app = web.Application(client_max_size=2 ** 28)

    async def health(_):
        return web.Response(text="")

    async def models(_):
        return web.json_response({"object": "list", "data": [{"id": "m", "max_model_len": fk.cfg["max_model_len"]}]})

    async def metrics(_):
        if fk.cfg["metrics_mode"] == "404":
            return web.Response(status=404, text="nope")
        return web.Response(text=fk.render(), content_type="text/plain")

    async def reset(req):
        fk.reset_calls += 1
        if fk.reset_calls <= fk.cfg["reset_fail_n"]:
            return web.json_response({"success": False})
        fk.gpu.clear()
        if req.query.get("reset_external") == "true":
            fk.cpu.clear()        # fs 層不清（同 vLLM tiering manager）
        return web.json_response({"success": True})

    async def configure(req):
        fk.configure(await req.json())
        return web.json_response({"ok": True})
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/metrics", metrics)
    app.router.add_post("/reset_prefix_cache", reset)
    app.router.add_post("/v1/completions", fk.completions)
    app.router.add_post("/fake/configure", configure)
    web.run_app(app, host="127.0.0.1", port=port, print=None, access_log=None)


# ═════════════════════════════════ 測試工具 ═════════════════════════════════
FAILS: list[str] = []
NOTES: list[str] = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def post_json(url, obj):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


class Env:
    def __init__(self, work):
        self.work = work
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.n = 0
        self.proc = subprocess.Popen([PY, os.path.abspath(__file__), "--fake-server", str(self.port)],
                                     stdout=open(os.path.join(work, "fake_server.out"), "w"), stderr=subprocess.STDOUT,
                                     env=self.env())
        for _ in range(200):
            try:
                urllib.request.urlopen(self.url + "/health", timeout=1)
                return
            except Exception:  # noqa: BLE001
                time.sleep(0.1)
        raise RuntimeError("假 server 起不來")

    def env(self, **extra):
        e = dict(os.environ, M16_FORBID_GPU_IMPORTS="1", TIARA_RUNS=os.path.join(self.work, "runs"))
        e.pop("TIARA_GPU", None)
        e.pop("M16_GUARD_STUB", None)
        e.pop("RUN_ID", None)
        e.update(extra)
        return e

    def configure(self, **cfg):
        self.n += 1
        tag = f"c{self.n:02d}"
        cfg.setdefault("log_path", os.path.join(self.work, f"fake_{tag}.jsonl"))
        post_json(self.url + "/fake/configure", cfg)
        return cfg

    def server_log(self, cfg):
        recs = {}
        if os.path.exists(cfg["log_path"]):
            for ln in open(cfg["log_path"]):
                r = json.loads(ln)
                recs[r["rid"]] = r
        return recs

    def run(self, args, env_extra=None, expect=0, name=""):
        self.n += 1
        rid = f"t{self.n:02d}-{name or 'x'}"
        env = self.env(RUN_ID=rid, **(env_extra or {}))
        p = subprocess.run([PY, BENCH] + args, env=env, capture_output=True, text=True, timeout=600)
        if p.returncode != expect:
            print(f"   [{name}] exit {p.returncode}（期望 {expect}）\n   stdout: {p.stdout[-1500:]}\n   stderr: {p.stderr[-2500:]}")
        return p, rid, os.path.join(self.work, "runs", rid)

    def close(self):
        self.proc.send_signal(signal.SIGTERM)
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def gen(args, out):
    p = subprocess.run([PY, WL] + args + ["--out", out], capture_output=True, text=True,
                       env=dict(os.environ, M16_FORBID_GPU_IMPORTS="1"), timeout=300)
    assert p.returncode == 0, p.stderr
    return out


def read_rows(path):
    return list(csv.DictReader(open(path)))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def write_sc(work, name, **kw):
    sc = dict(model="fake", apc="on", offload="cpu:1", offload_info={"kind": "cpu"}, gpu_blocks=None, kv_cache_gib=None,
              gpu_mem_util=None, rocm_attn_patch=False, attention_backend=None, max_model_len=40960,
              connector_debug=False, label=name)
    sc.update(kw)
    p = os.path.join(work, f"sc_{name}.json")
    json.dump(sc, open(p, "w"))
    return p


# ═════════════════════════════════ 測試 ═════════════════════════════════
def t_serve(E: Env):
    print("── serve（只組指令，不啟動）")
    fsdir = os.path.join(E.work, "fs_tier")
    cases = [
        ("llama-apc-off", ["--model", "llama", "--apc", "off", "--max-model-len", "40960", "--gpu", "1"], 0),
        ("llama-cpu", ["--model", "llama", "--apc", "on", "--offload", "cpu:64", "--gpu-blocks", "4096",
                       "--max-model-len", "40960", "--gpu", "1", "--block-size", "16"], 0),
        ("qwen-tier-patch", ["--model", "qwen", "--apc", "on", "--offload", f"tier:32:{fsdir}:1", "--rocm-attn-patch",
                             "--connector-debug", "--max-model-len", "40960", "--gpu", "0", "--kv-cache-gib", "20"], 0),
        ("bad-apcoff-offload", ["--model", "llama", "--apc", "off", "--offload", "cpu:8", "--max-model-len", "40960",
                                "--gpu", "0"], 2),
        ("bad-nogpu", ["--model", "llama", "--apc", "on", "--max-model-len", "40960"], 2),
    ]
    for name, args, exp in cases:
        p, rid_used, rd = E.run(["serve"] + args + ["--run-dir", os.path.join(E.work, "runs", f"serve-{name}")],
                                expect=exp, name=name)
        check(p.returncode == exp, f"serve {name}: 結束碼 {p.returncode}（期望 {exp}）")
        if exp != 0:
            continue
        rd = os.path.join(E.work, "runs", f"serve-{name}")
        sc = json.load(open(os.path.join(rd, "serve_config.json")))
        sh = open(os.path.join(rd, "serve_cmd.sh")).read()
        cmd = sc["cmd"]
        check("--enable-prompt-tokens-details" in cmd and sc["env"]["VLLM_SERVER_DEV_MODE"] == "1",
              f"serve {name}: 有 --enable-prompt-tokens-details 與 VLLM_SERVER_DEV_MODE=1")
        ex_lines = [ln for ln in sh.splitlines() if ln.startswith("exec ")]
        check(len(ex_lines) == 1 and ex_lines[0].endswith("vllm_server.log 2>&1") and ex_lines == sh.splitlines()[-1:],
              f"serve {name}: serve_cmd.sh 最後一行 exec、輸出到 vllm_server.log")
        check(sc["env"]["HF_HOME"] == "/mlsteam/data/tiara/hf-cache", f"serve {name}: HF_HOME＝平台 B 的值（不沿用 shell 的 /ssd7）")
        if name == "llama-apc-off":
            check("--no-enable-prefix-caching" in cmd and "--kv-transfer-config" not in cmd
                  and cmd[0].endswith("/bin/vllm") and sc["env"]["HIP_VISIBLE_DEVICES"] == "1",
                  "serve llama-apc-off：--no-enable-prefix-caching、沒有 connector、HIP_VISIBLE_DEVICES=1")
        if name == "llama-cpu":
            ktc = json.loads(cmd[cmd.index("--kv-transfer-config") + 1])
            ex = ktc["kv_connector_extra_config"]
            check(ktc["kv_connector"] == "OffloadingConnector" and ex["spec_name"] == "CPUOffloadingSpec"
                  and ex["cpu_bytes_to_use"] == 64 * 2 ** 30 and ex["eviction_policy"] == "lru"
                  and ktc["engine_id"] == f"m16-{rid_used}",
                  "serve llama-cpu：CPUOffloadingSpec、cpu_bytes_to_use＝64 GiB、lru、engine_id＝m16-<run>")
            check(cmd[cmd.index("--num-gpu-blocks-override") + 1] == "4096" and "--enable-prefix-caching" in cmd,
                  "serve llama-cpu：--num-gpu-blocks-override 4096、--enable-prefix-caching")
            check("rm -f /dev/shm/vllm_offload_m16-" in open(os.path.join(rd, "serve_cleanup.sh")).read(),
                  "serve llama-cpu：serve_cleanup.sh 只刪自己的 /dev/shm/vllm_offload_<engine_id>.mmap")
        if name == "qwen-tier-patch":
            ktc = json.loads(cmd[cmd.index("--kv-transfer-config") + 1])
            ex = ktc["kv_connector_extra_config"]
            check(ex["spec_name"] == "TieringOffloadingSpec" and ex["secondary_tiers"] == [{"type": "fs", "root_dir": fsdir}],
                  "serve tier：TieringOffloadingSpec＋fs 二級層 root_dir")
            check(sc["offload_info"]["fs_capacity_enforced"] is False, "serve tier：明寫 fs 容量 vLLM 不會限制")
            check(cmd[1].endswith("m16_serve_patched.py") and cmd[2] == "serve"
                  and cmd[cmd.index("--attention-backend") + 1] == "ROCM_ATTN",
                  "serve --rocm-attn-patch：走 m16_serve_patched.py、--attention-backend ROCM_ATTN")
            check("Qwen2.5-7B-Instruct-1M-noDCA" in sc["model_path"], "serve qwen：用 noDCA 目錄")
            lc = json.load(open(sc["logging_config"]))
            check(lc["loggers"]["vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler"]["level"] == "DEBUG"
                  and lc["loggers"]["vllm"]["level"] == "INFO" and sc["env"]["VLLM_LOGGING_CONFIG_PATH"] == sc["logging_config"],
                  "serve --connector-debug：只有 offloading scheduler 是 DEBUG（VLLM_LOGGING_CONFIG_PATH）")
            check(cmd[cmd.index("--kv-cache-memory-bytes") + 1] == str(20 * 2 ** 30), "serve --kv-cache-gib 20 → bytes")
    src = open(os.path.join(CODE, "m16_serve_patched.py")).read()
    try:
        compile(src, "m16_serve_patched.py", "exec")   # 只檢查語法，不執行（執行會 import torch／vllm）
        check(True, "m16_serve_patched.py 語法正確（compile，不執行）")
    except SyntaxError as e:
        check(False, f"m16_serve_patched.py 語法錯誤：{e}")


def t_dry(E: Env):
    print("── dry-run（不連 server）")
    w1 = gen(["gen-prefix", "--n-req", "10", "--prompt-len", "64", "--overlap", "0.5", "--n-groups", "2",
              "--out-len", "4", "--arrival", "poisson", "--rate", "5", "--seed", "1"], os.path.join(E.work, "w_dry1.jsonl"))
    p, rid, rd = E.run(["run", "--workload", w1, "--url", "http://127.0.0.1:9", "--dry-run"], name="dry1")
    check(p.returncode == 0 and os.path.exists(os.path.join(rd, "dryrun_schedule.csv")), "dry-run open-loop：結束碼 0、有排程檔")
    w2 = gen(["gen-agentic", "--n-sess", "2", "--n-turns", "3", "--base-ctx", "64", "--tool-len", "const:16",
              "--pause", "const:1", "--out-len", "4"], os.path.join(E.work, "w_dry2.jsonl"))
    p, rid, rd = E.run(["run", "--workload", w2, "--url", "http://127.0.0.1:9", "--dry-run", "--warmup-lead", "0.5x",
                        "--restore-est", "fixed:1.0", "--dry-service-s", "0.2"], name="dry2")
    rows = read_rows(os.path.join(rd, "dryrun_schedule.csv")) if p.returncode == 0 else []
    warm = [r for r in rows if r["is_warmup"] == "1"]
    act = {r["req_id"]: float(r["t_sched_s"]) for r in rows if r["is_warmup"] == "0"}
    ok = len(warm) == 4 and all(abs(act[w["req_id"][:-5]] - float(w["t_sched_s"]) - 0.5) < 1e-9 for w in warm)
    check(p.returncode == 0 and ok, "dry-run agentic：4 個 warm-up，各比實際送出早 0.5×1.0 s")
    p, rid, rd = E.run(["run", "--workload", w2, "--url", "http://127.0.0.1:9", "--dry-run", "--warmup-lead", "1x"],
                       expect=2, name="dry3")
    check(p.returncode == 2, "dry-run：lead 用倍數卻沒給 --restore-est → 結束碼 2")


def t_openloop(E: Env):
    print("── open-loop（prefix_overlap）：TTFT／TPOT 精度、/metrics 差值、命中率")
    cfg = E.configure(ttft_base=0.10, prefill_per_tok=0.0004, tpot=0.02, offload="cpu")
    w = gen(["gen-prefix", "--n-req", "16", "--prompt-len", "320", "--overlap", "0.5", "--n-groups", "2",
             "--out-len", "8", "--arrival", "fixed", "--rate", "6", "--seed", "3"], os.path.join(E.work, "w_ol.jsonl"))
    sc = write_sc(E.work, "ol")
    out = os.path.join(E.work, "out_ol")
    p, rid, rd = E.run(["run", "--workload", w, "--url", E.url, "--server-config", sc, "--guard", "off",
                        "--out-dir", out, "--slo-ttft", "0.2", "1.0", "--settle-s", "0.3", "--server-warmup", "0"],
                       name="ol")
    check(p.returncode == 0, "open-loop：結束碼 0")
    if p.returncode != 0:
        return
    rows = read_rows(os.path.join(out, f"requests_{rid}.csv"))
    summ = json.load(open(os.path.join(out, f"summary_{rid}.json")))
    srv = E.server_log(cfg)
    check(len(rows) == 16 and all(r["run_id"] == rid and r["ts"] for r in rows), "CSV：16 列、每列都有 run_id 與 ts")
    errs_ttft, errs_tpot = [], []
    for r in rows:
        s = srv.get(f"{rid}-{r['req_id']}")
        if s:
            errs_ttft.append(fnum(r["ttft_s"]) - s["intended_ttft"])
            errs_tpot.append(fnum(r["tpot_s"]) - s["intended_tpot"])
    med_t, max_t = statistics.median(abs(x) for x in errs_ttft), max(abs(x) for x in errs_ttft)
    med_p, max_p = statistics.median(abs(x) for x in errs_tpot), max(abs(x) for x in errs_tpot)
    NOTES.append(f"TTFT 誤差 |中位數| {med_t * 1000:.1f} ms、最大 {max_t * 1000:.1f} ms；TPOT |中位數| {med_p * 1000:.2f} ms、最大 {max_p * 1000:.2f} ms")
    check(len(errs_ttft) == 16 and med_t < 0.015 and max_t < 0.05,
          f"TTFT 和假 server 的設定值：|誤差| 中位數 {med_t * 1000:.1f} ms（< 15）、最大 {max_t * 1000:.1f} ms（< 50）")
    check(min(errs_ttft) > -0.002, f"TTFT 不會比 server 端還短（最小誤差 {min(errs_ttft) * 1000:.2f} ms）")
    check(med_p < 0.002 and max_p < 0.006,
          f"TPOT 和設定值 20 ms：|誤差| 中位數 {med_p * 1000:.2f} ms（< 2）、最大 {max_p * 1000:.2f} ms（< 6）")
    check(all(r["n_out"] == "8" for r in rows) and summ["n_out_mismatch"] == 0, "n_out＝max_tokens＝8")
    # 命中：同組第一個之後應命中 ≥ 共享部分對齊 16（160 → 160）
    cached = {r["req_id"]: int(r["cached_tokens"]) for r in rows}
    exp = {f"po{i:05d}": (0 if i < 2 else 160) for i in range(16)}
    check(cached == exp, "per-request cached_tokens：每組第一個 0、其餘 160（＝共享 160 token）")
    met = summ["server"]["metrics"]["series"]
    check(met["prefix_cache_queries"]["delta"] == 16 * 320 and met["prefix_cache_hits"]["delta"] == 14 * 160,
          f"/metrics 差值：queries {met['prefix_cache_queries']['delta']}（期望 5120）、hits {met['prefix_cache_hits']['delta']}（期望 2240）")
    hr = summ["server"]["metrics"]["derived"]["local_prefix_hit_rate"]
    check(abs(hr - 2240 / 5120) < 1e-12, f"命中率＝{hr:.4f}（期望 0.4375）")
    ob = met["offload_store_bytes"]
    check(ob["status"] == "before=REGISTERED_NO_SAMPLE;after=OK" and ob["delta"] > 0,
          f"offload store bytes：開跑前「有註冊沒樣本」記 0 並標示、結束後 OK（{ob['status']}）")
    check(met["offload_load_bytes"]["delta"] == 0 and "REGISTERED_NO_SAMPLE" in met["offload_load_bytes"]["status"],
          "offload load bytes：沒有載入 → 0，但狀態寫明是 REGISTERED_NO_SAMPLE（不是查不到）")
    gp = summ["goodput"]
    check(len(gp) == 2 and gp[0]["n_meet"] < gp[1]["n_meet"] == 16, f"goodput：SLO 0.2 s 滿足 {gp[0]['n_meet']}、1.0 s 滿足 16")
    check(summ["client_cached"]["sum_cached"] == 14 * 160, "client 端 cached_tokens 加總＝2240")
    check(summ["timing_valid"] is False and all(r["timing_valid"] == "0" for r in rows),
          "--guard off：summary 與每列 timing_valid＝0")


def t_agentic(E: Env):
    print("── agentic：暫停、warm-up 時間點與判定")
    cfg = E.configure(ttft_base=0.05, prefill_per_tok=0.0002, tpot=0.01, offload="cpu")
    w = gen(["gen-agentic", "--n-sess", "2", "--n-turns", "3", "--base-ctx", "200", "--tool-len", "const:40",
             "--pause", "const:0.4", "--out-len", "4", "--seed", "2"], os.path.join(E.work, "w_ag.jsonl"))
    sc = write_sc(E.work, "ag")
    p, rid, rd = E.run(["run", "--workload", w, "--url", E.url, "--server-config", sc, "--guard", "off",
                        "--settle-s", "0.2"], name="ag-pause")
    srv = E.server_log(cfg)
    gaps = []
    for s in range(2):
        for t in range(1, 3):
            a_ = srv.get(f"{rid}-ag-s{s:03d}-t{t:02d}")
            b_ = srv.get(f"{rid}-ag-s{s:03d}-t{t - 1:02d}")
            if a_ and b_:
                gaps.append(a_["t_recv"] - b_["t_done"])
    NOTES.append(f"agentic 暫停（設定 0.4 s，server 端量）：{[round(g, 3) for g in gaps]}")
    check(p.returncode == 0 and len(gaps) == 4 and all(0.395 <= g <= 0.45 for g in gaps),
          f"agentic：下一輪在上一輪結束＋0.4 s 才送（server 端量到 {[round(g, 3) for g in gaps]}）")

    # warm-up：秒數提前量
    cfg = E.configure(ttft_base=0.05, prefill_per_tok=0.0002, tpot=0.01, offload="cpu")
    w = gen(["gen-agentic", "--n-sess", "2", "--n-turns", "3", "--base-ctx", "200", "--tool-len", "const:40",
             "--pause", "const:0.6", "--out-len", "4", "--seed", "2"], os.path.join(E.work, "w_ag2.jsonl"))
    for lead_args, lead_s, nm in ((["--warmup-lead", "0.25s"], 0.25, "s"),
                                  (["--warmup-lead", "1x", "--restore-est", "fixed:0.3"], 0.3, "x")):
        cfg = E.configure(ttft_base=0.05, prefill_per_tok=0.0002, tpot=0.01, offload="cpu")
        p, rid, rd = E.run(["run", "--workload", w, "--url", E.url, "--server-config", sc, "--guard", "off",
                            "--settle-s", "0.2"] + lead_args, name=f"ag-warm-{nm}")
        srv = E.server_log(cfg)
        rows = read_rows(os.path.join(rd, "requests.csv")) if p.returncode == 0 else []
        warm = [r for r in rows if r["is_warmup"] == "1"]
        leads = []
        for r in warm:
            act = srv.get(f"{rid}-{r['warm_of']}")
            wr = srv.get(f"{rid}-{r['warm_of']}-warm")
            if act and wr:
                leads.append(act["t_recv"] - wr["t_recv"])
        check(p.returncode == 0 and len(warm) == 4 and len(leads) == 4 and all(abs(x - lead_s) < 0.04 for x in leads),
              f"warm-up（lead {nm}）：比實際請求早 {lead_s} s 送出（server 端量到 {[round(x, 3) for x in leads]}）")
        okp = all(int(r["prompt_len"]) == int(r["warm_prefix_len"]) and r["max_tokens"] == "1" for r in warm)
        check(okp, f"warm-up（lead {nm}）：prompt＝warm_prefix_len 個 token、max_tokens＝1")
        check(all(r["warm_status"] == "hidden" for r in warm), f"warm-up（lead {nm}）：全部 hidden（實得 {[r['warm_status'] for r in warm]}）")

    # noise：預測誤差
    cfg = E.configure(ttft_base=0.05, prefill_per_tok=0.0002, tpot=0.01, offload="cpu")
    p, rid, rd = E.run(["run", "--workload", w, "--url", E.url, "--server-config", sc, "--guard", "off",
                        "--settle-s", "0.2", "--warmup-lead", "0.2s", "--warmup-noise", "0.5", "--warmup-seed", "3"],
                       name="ag-noise")
    srv = E.server_log(cfg)
    rows = read_rows(os.path.join(rd, "requests.csv")) if p.returncode == 0 else []
    warm = [r for r in rows if r["is_warmup"] == "1"]
    dev = []
    for r in warm:
        e = float(r["warm_pred_err"])
        s_, t_ = int(r["session"]), int(r["turn"])
        prev = srv.get(f"{rid}-ag-s{s_:03d}-t{t_ - 1:02d}")
        wr = srv.get(f"{rid}-{r['warm_of']}-warm")
        if prev and wr:
            dev.append((wr["t_recv"] - prev["t_done"]) - max(0.0, 0.6 * (1 + e) - 0.2))
    errs = [float(r["warm_pred_err"]) for r in warm]
    check(len(warm) == 4 and all(-0.5 <= x <= 0.5 for x in errs) and len(set(errs)) == 4,
          f"noise 0.5：預測誤差都在 ±50% 內且各不相同（{[round(x, 3) for x in errs]}）")
    check(len(dev) == 4 and all(-0.005 < d < 0.05 for d in dev),
          f"noise 0.5：warm-up 送出時間＝上一輪結束＋0.6×(1+e)−0.2（偏差 {[round(d, 3) for d in dev]}）")

    # late：warm-up 太慢
    E.configure(ttft_base=0.3, prefill_per_tok=0.0002, tpot=0.01, offload="cpu")
    p, rid, rd = E.run(["run", "--workload", w, "--url", E.url, "--server-config", sc, "--guard", "off",
                        "--settle-s", "0.2", "--warmup-lead", "0.1s"], name="ag-late")
    summ = json.load(open(os.path.join(rd, "summary.json"))) if p.returncode == 0 else {}
    wu = summ.get("warmup", {})
    check(wu.get("status_counts") == {"late": 4} and wu.get("wasted_frac") == 1.0,
          f"late：warm-up 要 0.3 s 但只提前 0.1 s → 4/4 late、白搬比例 1.0（{wu.get('status_counts')}）")
    # evicted：warm-up 完成後 GPU 被清掉
    E.configure(ttft_base=0.05, prefill_per_tok=0.0002, tpot=0.01, offload="none", evict_after_warm_s=0.05)
    p, rid, rd = E.run(["run", "--workload", w, "--url", E.url, "--server-config", write_sc(E.work, "ag_noff", offload="none",
                        offload_info={}), "--guard", "off", "--settle-s", "0.2", "--warmup-lead", "0.3s"], name="ag-evict")
    summ = json.load(open(os.path.join(rd, "summary.json"))) if p.returncode == 0 else {}
    wu = summ.get("warmup", {})
    check(wu.get("status_counts") == {"evicted": 4} and wu.get("wasted_numerator") == 4,
          f"evicted：warm-up 完成後 KV 被擠掉 → 4/4 evicted（{wu.get('status_counts')}）")
    # unknown：server 沒回 cached_tokens
    E.configure(ttft_base=0.05, prefill_per_tok=0.0002, tpot=0.01, offload="cpu", prompt_details=False)
    p, rid, rd = E.run(["run", "--workload", w, "--url", E.url, "--server-config", sc, "--guard", "off",
                        "--settle-s", "0.2", "--warmup-lead", "0.25s"], name="ag-unknown")
    summ = json.load(open(os.path.join(rd, "summary.json"))) if p.returncode == 0 else {}
    wu = summ.get("warmup", {})
    check(wu.get("status_counts") == {"unknown": 4} and summ.get("client_cached", {}).get("status") == "NOT_AVAILABLE",
          f"沒有 cached_tokens：warm-up 判 unknown、client 命中率 NOT_AVAILABLE（{wu.get('status_counts')}）")


def t_metrics_missing(E: Env):
    print("── /metrics 缺欄位 → NOT_AVAILABLE（不是 0）")
    w = gen(["gen-prefix", "--n-req", "4", "--prompt-len", "64", "--overlap", "0.5", "--n-groups", "1",
             "--out-len", "2", "--arrival", "burst"], os.path.join(E.work, "w_m.jsonl"))
    sc = write_sc(E.work, "m")
    E.configure(ttft_base=0.02, tpot=0.005, offload="cpu", metrics_mode="no_offload")
    p, rid, rd = E.run(["run", "--workload", w, "--url", E.url, "--server-config", sc, "--guard", "off",
                        "--settle-s", "0.1"], name="m-nooff")
    summ = json.load(open(os.path.join(rd, "summary.json"))) if p.returncode == 0 else {}
    ser = summ.get("server", {}).get("metrics", {}).get("series", {})
    check(ser.get("offload_load_bytes", {}).get("delta") == "NOT_AVAILABLE"
          and "NOT_REGISTERED" in ser.get("offload_load_bytes", {}).get("status", "")
          and ser.get("prefix_cache_queries", {}).get("delta") == 4 * 64,
          "offload 家族不存在 → offload_load_bytes＝NOT_AVAILABLE（NOT_REGISTERED），其他照常")
    der = summ.get("server", {}).get("metrics", {}).get("derived", {})
    check(der.get("offload_load_gib_per_s_vllm") == "NOT_AVAILABLE", "衍生的 vLLM 載入頻寬也是 NOT_AVAILABLE")
    E.configure(ttft_base=0.02, tpot=0.005, offload="cpu", metrics_mode="404")
    p, rid, rd = E.run(["run", "--workload", w, "--url", E.url, "--server-config", sc, "--guard", "off",
                        "--settle-s", "0.1"], name="m-404")
    summ = json.load(open(os.path.join(rd, "summary.json"))) if p.returncode == 0 else {}
    ser = summ.get("server", {}).get("metrics", {}).get("series", {})
    vals = {k: v["delta"] for k, v in ser.items()}
    check(p.returncode == 0 and vals and all(v == "NOT_AVAILABLE" for v in vals.values())
          and summ["server"]["metrics"]["derived"]["local_prefix_hit_rate"] == "NOT_AVAILABLE",
          f"/metrics 回 404 → {len(vals)} 個計數全部 NOT_AVAILABLE、命中率 NOT_AVAILABLE（沒有任何 0）")


def t_f2(E: Env):
    print("── F2 三段式（seed → evict → burst）與延後訊息計數")
    w = gen(["gen-burst", "--prefix-len", "256", "--suffix-len", "32", "--n-concurrent", "6", "--out-len", "4",
             "--n-fill", "3", "--fill-len", "512"], os.path.join(E.work, "w_f2.jsonl"))
    slog = os.path.join(E.work, "fake_vllm_server.log")
    open(slog, "w").write("INFO 開頭雜訊（不屬於這次 run）\n(EngineCore pid=1) DEBUG x [scheduler.py:855] Delaying request "
                          "OLD since some of its chunks are already being loaded\n")
    # GPU 只放得下 40 個 block：seed（18 block）之後 3 個填充（各 32 block）會把前綴擠出 GPU；CPU 留著
    cfg = E.configure(ttft_base=0.02, prefill_per_tok=0.0002, load_per_tok=0.002, tpot=0.005, offload="cpu",
                      gpu_cap_blocks=40, server_log=slog)
    sc = write_sc(E.work, "f2", connector_debug=True)
    p, rid, rd = E.run(["run", "--workload", w, "--url", E.url, "--server-config", sc, "--guard", "off",
                        "--settle-s", "0.2", "--phase-gap-s", "0.2", "--server-log", slog], name="f2-fill")
    summ = json.load(open(os.path.join(rd, "summary.json"))) if p.returncode == 0 else {}
    srv = E.server_log(cfg)
    ph = [x["phase"] for x in summ.get("server", {}).get("phases", [])]
    check(p.returncode == 0 and ph == ["seed", "evict", "burst"], f"三段依序（{ph}）")
    seed = srv.get(f"{rid}-sp-seed")
    fills = [v for k, v in srv.items() if "sp-fill" in k]
    burst = [v for k, v in srv.items() if "-sp-b" in k]
    check(seed and len(fills) == 3 and len(burst) == 6 and max(f["t_done"] for f in fills) < min(b["t_recv"] for b in burst)
          and seed["t_done"] < min(f["t_recv"] for f in fills), "seed 完成 → 填充全部完成 → burst 才開始（server 端時間）")
    spread = max(b["t_recv"] for b in burst) - min(b["t_recv"] for b in burst)
    check(spread < 0.03, f"burst 的 6 個請求同時送出（server 收到時間差 {spread * 1000:.1f} ms）")
    delayed = sum(1 for b in burst if b["polls"] > 0)
    cl = summ.get("server", {}).get("connector_log", {})
    d = cl.get("delay_chunks_being_loaded", {})
    check(delayed == 5 and d.get("distinct_requests") == 5 and d.get("lines") == sum(b["polls"] for b in burst),
          f"延後訊息：假 server 延後 {delayed} 個請求，bench 數到 {d}（只數這次 run 的 log 區間，不含 OLD）")
    check(cl.get("hit_offloaded_tokens", {}).get("distinct_requests") == 1,
          "只有第一個 burst 請求真的從 CPU 載入（其他等它載完變成 GPU 命中）")
    bm = summ.get("server", {}).get("phases", [{}] * 3)[2].get("metrics", {}).get("series", {})
    check(bm.get("offload_load_bytes", {}).get("delta") == 256 * 131072,
          f"burst 段的 /metrics：只載入一次前綴 256 token × 128 KiB（實得 {bm.get('offload_load_bytes', {}).get('delta')}）")
    check(summ.get("n_measured") == 6 and "burst" in summ, "主指標只算 burst 段的 6 個請求")
    p2 = subprocess.run([PY, BENCH, "count-log", "--log", slog], capture_output=True, text=True, env=E.env())
    tot = json.loads(p2.stdout)["delay_chunks_being_loaded"]["lines"] if p2.returncode == 0 else None
    check(tot == d.get("lines", -99) + 1, f"count-log 整個檔案＝這次 run＋1 行舊的（{tot}）")
    # evict=reset、debug 沒開
    w2 = gen(["gen-burst", "--prefix-len", "256", "--suffix-len", "32", "--n-concurrent", "4", "--out-len", "4"],
             os.path.join(E.work, "w_f2b.jsonl"))
    E.configure(ttft_base=0.02, prefill_per_tok=0.0002, load_per_tok=0.002, tpot=0.005, offload="cpu", server_log=slog)
    p, rid, rd = E.run(["run", "--workload", w2, "--url", E.url, "--server-config", write_sc(E.work, "f2b"),
                        "--guard", "off", "--settle-s", "0.2", "--phase-gap-s", "0.2", "--evict", "reset",
                        "--server-log", slog], name="f2-reset")
    summ = json.load(open(os.path.join(rd, "summary.json"))) if p.returncode == 0 else {}
    cl = summ.get("server", {}).get("connector_log", {})
    rs = [r for r in summ.get("server", {}).get("resets", []) if r.get("when") == "evict"]
    check(p.returncode == 0 and rs and rs[0]["ok"] and cl.get("delay_chunks_being_loaded", {}).get("distinct_requests") == 3,
          "evict=reset：用 /reset_prefix_cache 清 GPU、CPU 留著 → 4 個 burst 裡 3 個被延後")
    check(str(cl.get("status", "")).startswith("DEBUG_OFF"), "serve 沒開 --connector-debug → 計數標 DEBUG_OFF")
    # --phases burst：冷啟動（沒有 seed）
    cfg = E.configure(ttft_base=0.02, prefill_per_tok=0.0002, load_per_tok=0.002, tpot=0.005, offload="cpu")
    p, rid, rd = E.run(["run", "--workload", w, "--url", E.url, "--server-config", write_sc(E.work, "f2c"),
                        "--guard", "off", "--settle-s", "0.2", "--phase-gap-s", "0.1", "--phases", "burst"], name="f2-cold")
    srv = E.server_log(cfg)
    check(p.returncode == 0 and sorted(k.split("-sp-")[1] for k in srv if "-sp-" in k) == [f"b{i:04d}" for i in range(6)],
          "--phases burst：只送 6 個 burst（沒有 seed、沒有填充）")
    p, rid, rd = E.run(["run", "--workload", w, "--url", E.url, "--guard", "off", "--phases", "burst,seed"],
                       expect=2, name="f2-badphase")
    check(p.returncode == 2, "--phases 順序不對 → 結束碼 2")


def t_v0(E: Env):
    print("── v0check：命中 vs 不命中輸出逐 token 比對")
    sc = write_sc(E.work, "v0cpu")
    E.configure(ttft_base=0.02, prefill_per_tok=0.00005, tpot=0.003, offload="cpu", reset_fail_n=2)
    out = os.path.join(E.work, "out_v0")
    p, rid, rd = E.run(["v0check", "--url", E.url, "--server-config", sc, "--guard", "off", "--lengths", "300", "700",
                        "--n-prompts", "2", "--max-tokens", "8", "--settle-s", "0.05", "--out-dir", out], name="v0-cpu")
    res = json.load(open(os.path.join(out, f"v0check_{rid}.json"))) if p.returncode == 0 else {}
    check(p.returncode == 0 and res.get("overall") == "PASS" and res.get("mode") == "cpu"
          and set(res.get("verdict", {})) >= {"gpu_hit_match", "cpu_hit_match", "cpu_hit_loaded_bytes",
                                               "determinism_miss_vs_miss2", "miss_cold"},
          f"v0check cpu：全部 PASS（{res.get('verdict')}）；/reset_prefix_cache 前 2 次 success=false 有重試")
    rows = read_rows(os.path.join(out, f"v0check_{rid}.csv")) if p.returncode == 0 else []
    check(len(rows) == 4 * 4 and all(r["run_id"] == rid and r["ts"] for r in rows), "v0check CSV：4 prompt × 4 phase、有 run_id／ts")
    E.configure(ttft_base=0.02, prefill_per_tok=0.00005, tpot=0.003, offload="cpu", corrupt_cpu=True)
    p, rid, rd = E.run(["v0check", "--url", E.url, "--server-config", sc, "--guard", "off", "--lengths", "300",
                        "--n-prompts", "1", "--max-tokens", "8", "--settle-s", "0.05"], expect=3, name="v0-corrupt")
    res = json.load(open(os.path.join(rd, f"v0check_{rid}.json"))) if os.path.exists(os.path.join(rd, f"v0check_{rid}.json")) else {}
    row = [r for r in read_rows(os.path.join(rd, f"v0check_{rid}.csv")) if r["phase"] == "cpu_hit"] if res else []
    check(p.returncode == 3 and res.get("verdict", {}).get("cpu_hit_match") == "FAIL" and res.get("overall") == "FAIL"
          and row and row[0]["first_diverge"] == "0" and abs(float(row[0]["miss_margin_at_diverge"]) - 2.5) < 1e-9,
          "負對照：CPU 載入後輸出被改 → cpu_hit_match FAIL、第 0 個 token 分歧、記下 miss 的 logprob 差 2.5")
    sct = write_sc(E.work, "v0tier", offload="tier:1:/x:1", offload_info={"kind": "tier"})
    E.configure(ttft_base=0.02, prefill_per_tok=0.00005, tpot=0.003, offload="tier")
    p, rid, rd = E.run(["v0check", "--url", E.url, "--server-config", sct, "--guard", "off", "--lengths", "300",
                        "--n-prompts", "1", "--max-tokens", "8", "--settle-s", "0.05"], name="v0-tier")
    res = json.load(open(os.path.join(rd, f"v0check_{rid}.json"))) if p.returncode == 0 else {}
    check(res.get("overall") == "PASS" and res.get("verdict", {}).get("fs_hit_tier_read_bytes") == "PASS",
          f"v0check tier：fs 載入也 PASS（{res.get('verdict')}）")
    E.configure(ttft_base=0.02, prefill_per_tok=0.00005, tpot=0.003, offload="cpu")
    p, rid, rd = E.run(["v0check", "--url", E.url, "--server-config", sc, "--guard", "off", "--text-file", WL,
                        "--text-chars", "800", "--n-prompts", "2", "--max-tokens", "6", "--settle-s", "0.05"], name="v0-text")
    res = json.load(open(os.path.join(rd, f"v0check_{rid}.json"))) if p.returncode == 0 else {}
    check(res.get("overall") == "PASS", "v0check 文字 prompt（server 端 tokenize）：PASS")


def t_guard_and_failures(E: Env):
    print("── 污染、guard、失敗處理")
    w = gen(["gen-prefix", "--n-req", "4", "--prompt-len", "64", "--overlap", "0.5", "--n-groups", "1",
             "--out-len", "2", "--arrival", "burst"], os.path.join(E.work, "w_g.jsonl"))
    sc = write_sc(E.work, "g")
    E.configure(ttft_base=0.02, tpot=0.005, offload="cpu")
    p, rid, rd = E.run(["run", "--workload", w, "--url", E.url, "--server-config", sc], expect=2, name="g-nogpu")
    check(p.returncode == 2 and "TIARA_GPU" in p.stderr, "--guard on 但沒有 TIARA_GPU → 結束碼 2、不量（規則 7）")
    out = os.path.join(E.work, "out_g")
    p, rid, rd = E.run(["run", "--workload", w, "--url", E.url, "--server-config", sc, "--out-dir", out,
                        "--settle-s", "0.1"], env_extra={"M16_GUARD_STUB": "contaminated"}, expect=4, name="g-stub")
    summ = json.load(open(os.path.join(rd, "summary.json"))) if os.path.exists(os.path.join(rd, "summary.json")) else {}
    check(p.returncode == 4 and os.path.exists(os.path.join(rd, "CONTAMINATED"))
          and not glob.glob(os.path.join(out, f"*{rid}*")) and summ.get("timing_valid") is False
          and summ.get("guard", {}).get("contaminated") is True,
          "contaminated：結束碼 4、run 目錄有 CONTAMINATED、--out-dir 沒有這個 run、summary timing_valid＝false")
    p, rid, rd = E.run(["run", "--workload", w, "--url", E.url, "--server-config", sc, "--guard", "off"],
                       env_extra={"M16_GUARD_STUB": "clean"}, expect=2, name="g-stubclean")
    check(p.returncode == 2, "M16_GUARD_STUB 不能設成 clean（替身只能判髒）")
    E.configure(ttft_base=0.02, tpot=0.005, offload="cpu", fail_ids=["po00002"])
    p, rid, rd = E.run(["run", "--workload", w, "--url", E.url, "--server-config", sc, "--guard", "off",
                        "--settle-s", "0.1"], expect=3, name="g-fail")
    rows = read_rows(os.path.join(rd, "requests.csv")) if os.path.exists(os.path.join(rd, "requests.csv")) else []
    bad = [r for r in rows if r["ok"] != "1"]
    check(p.returncode == 3 and len(rows) == 4 and len(bad) == 1 and bad[0]["req_id"] == "po00002"
          and "HTTP 500" in bad[0]["error"], "一個請求 HTTP 500 → 照實記錄（ok=0、錯誤訊息）、其他照常、結束碼 3")
    wa = gen(["gen-agentic", "--n-sess", "1", "--n-turns", "3", "--base-ctx", "64", "--tool-len", "const:8",
              "--pause", "const:0.05", "--out-len", "2"], os.path.join(E.work, "w_gf.jsonl"))
    E.configure(ttft_base=0.02, tpot=0.005, offload="cpu", fail_ids=["ag-s000-t01"])
    p, rid, rd = E.run(["run", "--workload", wa, "--url", E.url, "--server-config", sc, "--guard", "off",
                        "--settle-s", "0.1"], expect=3, name="g-agfail")
    rows = read_rows(os.path.join(rd, "requests.csv")) if os.path.exists(os.path.join(rd, "requests.csv")) else []
    st = {r["req_id"]: (r["ok"], r["error"][:24]) for r in rows}
    check(st.get("ag-s000-t02", ("", ""))[1].startswith("SKIPPED_PREV_TURN_FAILED") and st.get("ag-s000-t00", ("",))[0] == "1",
          f"agentic 某輪失敗 → 後面的輪記成 SKIPPED（不是靜默消失）：{st}")
    E.configure(ttft_base=0.02, tpot=0.005, offload="cpu", max_model_len=50)
    p, rid, rd = E.run(["run", "--workload", w, "--url", E.url, "--guard", "off"], expect=2, name="g-mml")
    check(p.returncode == 2, "server 回報的 max_model_len（50）放不下工作負載（66）→ 結束碼 2，不送")


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--fake-server":
        fake_main(int(sys.argv[2]))
        return 0
    base = os.environ.get("M16_TEST_DIR") or tempfile.mkdtemp(prefix="m16_bench_cpu_")
    os.makedirs(base, exist_ok=True)
    print(f"工作目錄：{base}")
    E = Env(base)
    try:
        for fn in (t_serve, t_dry, t_openloop, t_agentic, t_metrics_missing, t_f2, t_v0, t_guard_and_failures):
            try:
                fn(E)
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                check(False, f"{fn.__name__} 例外：{type(e).__name__}: {e}")
    finally:
        E.close()
    check(not [m for m in sys.modules if m.split(".")[0] in W.GPU_MODULES], "測試行程沒有載入 torch／vllm")
    print("── 備註")
    for n in NOTES:
        print("   " + n)
    if FAILS:
        print(f"FAILED {len(FAILS)} 項：")
        for f in FAILS:
            print("  - " + f)
        return 1
    if not os.environ.get("M16_TEST_DIR"):
        shutil.rmtree(base, ignore_errors=True)
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
