#!/usr/bin/env python3
"""smoke_3090.py — 平台 A（3090）環境重建後的冒煙測試（P3-0，2026-10-17）

子命令（都在 code/run3090.sh 的記錄殼裡跑；輸出 JSON 到 $RUN_DIR/<子命令>.json）：
  triton    編一個 Triton kernel（向量加法＋列 softmax）執行、和 torch 對答案
  libs      第三階段套件 import＋版本；xgboost／lightgbm 小資料訓練（xgboost 另試 device=cuda）；
            e5-small（sentence-transformers）在 GPU 上 encode
  prefill   m7_model.KVModel 載入 M7_MODEL_GLOB 的模型，chunk 化 prefill --ctx 個 token，記 peak VRAM
            （torch allocator 的 max_allocated／max_reserved，以及 nvidia-smi 看到的本行程用量峰值）
  vllm      （venv/vllm028 跑）vLLM 0.28 載入小模型生成幾個 token

⚠️ smoke 等級：時間只記數量級、不可寫進論文。peak VRAM 是這台卡、這版 torch 的實測，可以引用但要帶 run_id。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime

GiB = 1 << 30
HERE = os.path.dirname(os.path.abspath(__file__))


def out_path(name):
    d = os.environ.get("RUN_DIR") or "."
    return os.path.join(d, f"{name}.json")


def dump(name, res):
    res.setdefault("ts", datetime.now().astimezone().isoformat(timespec="seconds"))
    res.setdefault("run_id", os.environ.get("RUN_ID", "NO_RUN_ID"))
    print(json.dumps(res, indent=1, default=str))
    with open(out_path(name), "w") as f:
        json.dump(res, f, indent=1, default=str)


class SmiPeak:
    """背景每 0.2 s 讀 nvidia-smi 的本行程顯存（MiB），記峰值。查不到就記錯誤，不當成 0（規則 7）。"""

    def __init__(self, pid):
        self.pid, self.peak, self.n, self.err = pid, None, 0, None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _read(self):
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            raise RuntimeError(out.stderr.strip() or f"nvidia-smi rc={out.returncode}")
        for line in out.stdout.strip().splitlines():
            p, m = [x.strip() for x in line.split(",")]
            if int(p) == self.pid:
                return int(m)
        return None

    def _loop(self):
        while not self._stop.wait(0.2):
            try:
                v = self._read()
            except Exception as e:  # noqa: BLE001
                self.err = repr(e)
                continue
            self.n += 1
            if v is not None:
                self.peak = v if self.peak is None else max(self.peak, v)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=5)
        try:
            v = self._read()
            if v is not None:
                self.peak = v if self.peak is None else max(self.peak, v)
        except Exception as e:  # noqa: BLE001
            self.err = repr(e)


def dev_info(torch):
    p = torch.cuda.get_device_properties(0)
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "device": p.name, "arch": f"sm_{p.major}{p.minor}", "total_memory_gib": round(p.total_memory / GiB, 2),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


# ---------------------------------------------------------------------------------------------------
def cmd_triton(a):
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def add_k(x, y, o, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        tl.store(o + i, tl.load(x + i, mask=m) + tl.load(y + i, mask=m), mask=m)

    @triton.jit
    def softmax_k(x, o, stride, ncol, BLOCK: tl.constexpr):
        r = tl.program_id(0)
        c = tl.arange(0, BLOCK)
        v = tl.load(x + r * stride + c, mask=c < ncol, other=-float("inf"))
        v = v - tl.max(v, axis=0)
        e = tl.exp(v)
        tl.store(o + r * stride + c, e / tl.sum(e, axis=0), mask=c < ncol)

    res = dev_info(torch) | {"triton": triton.__version__, "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR")}
    d = torch.device("cuda:0")
    n = 1 << 24
    x, y = torch.randn(n, device=d), torch.randn(n, device=d)
    o = torch.empty_like(x)
    t0 = time.perf_counter()
    add_k[(triton.cdiv(n, 1024),)](x, y, o, n, BLOCK=1024)
    torch.cuda.synchronize()
    res["add_first_call_s_incl_compile"] = round(time.perf_counter() - t0, 3)
    res["add_max_abs_err"] = float((o - (x + y)).abs().max())
    X = torch.randn(4096, 1000, device=d)
    O = torch.empty_like(X)
    softmax_k[(X.shape[0],)](X, O, X.stride(0), X.shape[1], BLOCK=1024)
    torch.cuda.synchronize()
    res["softmax_max_abs_err"] = float((O - torch.softmax(X, dim=1)).abs().max())
    # 一個 torch 端的 matmul（driver 太舊時 is_available 會是 True 但 kernel 會爆：EXPERIMENT_PLAN §8）
    A = torch.randn(4096, 4096, device=d, dtype=torch.bfloat16)
    res["matmul_bf16_finite"] = bool(torch.isfinite(A @ A).all())
    res["ok"] = res["add_max_abs_err"] == 0.0 and res["softmax_max_abs_err"] < 1e-5 and res["matmul_bf16_finite"]
    dump("triton", res)
    return 0 if res["ok"] else 1


def cmd_libs(a):
    import importlib

    import numpy as np
    res = {"versions": {}, "errors": {}}
    for m in ["torch", "triton", "transformers", "tokenizers", "safetensors", "accelerate", "peft", "datasets",
              "huggingface_hub", "hf_xet", "hf_transfer", "numpy", "pandas", "scipy", "pyarrow", "matplotlib",
              "sklearn", "lightgbm", "xgboost", "lifelines", "sksurv", "pycox", "torchtuples", "statsmodels",
              "sentence_transformers", "tqdm", "pytest", "numba"]:
        try:
            mod = importlib.import_module(m)
            res["versions"][m] = getattr(mod, "__version__", "?")
        except Exception as e:  # noqa: BLE001
            res["errors"][m] = repr(e)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 16)).astype(np.float32)
    yb = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    try:
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(n_estimators=20, verbose=-1).fit(X, yb)
        res["lightgbm_cpu_train_acc"] = float((clf.predict(X) == yb).mean())
    except Exception as e:  # noqa: BLE001
        res["errors"]["lightgbm_train"] = repr(e)
    try:
        import xgboost as xgb
        res["xgboost_build_info"] = {k: v for k, v in xgb.build_info().items() if "CUDA" in k.upper() or "NCCL" in k.upper()}
        import warnings
        for dev in ("cpu", "cuda"):
            try:
                with warnings.catch_warnings(record=True) as ws:
                    warnings.simplefilter("always")
                    m = xgb.XGBClassifier(n_estimators=20, device=dev, tree_method="hist").fit(X, yb)
                    acc = float((m.predict(X) == yb).mean())
                fell = [str(w.message)[:300] for w in ws if "GPU" in str(w.message)]
                if dev == "cuda" and fell:
                    # 2026-10-17 第一次 libs smoke：xgboost 3.4.1 是 CUDA 13.3 build，driver 550（CUDA 12.4）下
                    # 只發 warning 就靜默改用 CPU。那個 acc 是 CPU 的，不能當成 GPU 可用（規則 7）
                    res["errors"]["xgboost_cuda_train"] = "SILENT_CPU_FALLBACK: " + " | ".join(fell)
                    res["xgboost_cuda_train_acc_actually_cpu"] = acc
                else:
                    res[f"xgboost_{dev}_train_acc"] = acc
            except Exception as e:  # noqa: BLE001
                res["errors"][f"xgboost_{dev}_train"] = repr(e)[:2000]
    except Exception as e:  # noqa: BLE001
        res["errors"]["xgboost"] = repr(e)
    try:
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from sksurv.util import Surv
        t = rng.exponential(size=2000) * np.exp(-X[:, 0])
        ev = rng.random(2000) < 0.8
        cox = CoxPHSurvivalAnalysis().fit(X[:, :4], Surv.from_arrays(ev, t))
        res["sksurv_cox_coef0"] = float(cox.coef_[0])
    except Exception as e:  # noqa: BLE001
        res["errors"]["sksurv_fit"] = repr(e)
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        # 用 snapshot 路徑（TIARA_E5_PATH，env_3090.sh）：用 --revision 下載的 repo 沒有 refs/main，離線時用名稱載入會失敗
        p = os.environ.get("TIARA_E5_PATH") or "intfloat/multilingual-e5-small"
        st = SentenceTransformer(p, device="cuda" if torch.cuda.is_available() else "cpu")
        emb = st.encode(["query: KV cache 放在哪一層？", "passage: The KV cache is offloaded to CPU memory."],
                        normalize_embeddings=True)
        res["e5_small"] = {"path": p, "device": str(st.device), "dim": int(emb.shape[1]), "cos01": float(emb[0] @ emb[1]),
                           "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE")}
    except Exception as e:  # noqa: BLE001
        res["errors"]["e5_small"] = repr(e)[:2000]
    res["ok"] = not res["errors"]
    dump("libs", res)
    return 0 if res["ok"] else 1


def cmd_prefill(a):
    import torch
    sys.path.insert(0, HERE)
    from m7_model import CHUNK, KVModel, model_path

    res = dev_info(torch) | {"ctx": a.ctx, "model_glob": os.environ.get("M7_MODEL_GLOB"), "model_path": model_path(),
                             "chunk": CHUNK, "max_len": a.ctx}
    assert a.ctx % CHUNK == 0 and a.ctx >= 2 * CHUNK
    with SmiPeak(os.getpid()) as sp:
        try:
            t0 = time.perf_counter()
            km = KVModel(max_len=a.ctx)
            torch.cuda.synchronize()
            res["load_s"] = round(time.perf_counter() - t0, 1)
            res["kv_buffer_gib"] = round(km.kv.numel() * km.kv.element_size() / GiB, 3)
            res["after_load_allocated_gib"] = round(torch.cuda.memory_allocated() / GiB, 3)
            res["model"] = {"L": km.L, "H_kv": km.H, "H_q": km.Hq, "D": km.D, "chunk_bytes": km.chunk_bytes}
            torch.cuda.reset_peak_memory_stats()
            g = torch.Generator().manual_seed(0)
            ids = torch.randint(1000, min(120000, km.cfg.vocab_size), (a.ctx,), generator=g).to(km.device)
            t0 = time.perf_counter()
            km.prefill_chunked(ids[:a.ctx - CHUNK])
            logit = km.forward_span(ids[a.ctx - CHUNK:], a.ctx - CHUNK, want_logits=True)
            torch.cuda.synchronize()
            res["prefill_s_smoke"] = round(time.perf_counter() - t0, 2)
            res["last_logits_finite"] = bool(torch.isfinite(logit).all())
            res["last_argmax"] = int(logit.argmax())
            res["status"] = "OK"
        except torch.OutOfMemoryError as e:
            res["status"] = "OOM"
            res["error"] = str(e)[:3000]
        res["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / GiB, 3)
        res["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / GiB, 3)
    res["smi_peak_process_mib"] = sp.peak
    res["smi_samples"] = sp.n
    res["smi_error"] = sp.err
    res["sdpa_backends_enabled"] = {"flash": torch.backends.cuda.flash_sdp_enabled(),
                                    "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                                    "math": torch.backends.cuda.math_sdp_enabled()}
    dump(f"prefill_{a.ctx}", res)
    return 0 if res["status"] == "OK" else 5


def cmd_vllm(a):
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    res = dev_info(torch) | {"vllm": vllm.__version__, "model": a.model}
    t0 = time.perf_counter()
    llm = LLM(model=a.model, max_model_len=4096, gpu_memory_utilization=a.gpu_mem, enforce_eager=a.eager, seed=0)
    res["load_s"] = round(time.perf_counter() - t0, 1)
    outs = llm.generate(["The capital of France is"], SamplingParams(temperature=0.0, max_tokens=16))
    res["text"] = outs[0].outputs[0].text
    res["n_tokens"] = len(outs[0].outputs[0].token_ids)
    res["ok"] = res["n_tokens"] > 0
    dump("vllm", res)
    return 0 if res["ok"] else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("triton")
    sub.add_parser("libs")
    p = sub.add_parser("prefill")
    p.add_argument("--ctx", type=int, default=2048)
    v = sub.add_parser("vllm")
    v.add_argument("--model", default="Qwen/Qwen3-0.6B")
    v.add_argument("--gpu-mem", type=float, default=0.5)
    v.add_argument("--eager", action="store_true")
    a = ap.parse_args()
    return {"triton": cmd_triton, "libs": cmd_libs, "prefill": cmd_prefill, "vllm": cmd_vllm}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
