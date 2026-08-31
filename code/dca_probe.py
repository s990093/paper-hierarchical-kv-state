#!/usr/bin/env python3
"""決定性實測：vLLM 0.28 的 V1 engine 能不能真的跑 Dual Chunk Attention？

## 為什麼這件事決定整個長上下文方向

`graelo/Qwen2.5-7B-Instruct-1M-AWQ` 的 config 是：

    max_position_embeddings     = 1,010,000
    dual_chunk_attention_config = {chunk_size: 262144, local_size: 8192,
                                   original_max_position_embeddings: 262144}

若 DCA 可用，這顆模型在 3090 上（AWQ 權重 + FP8 KV，容量 599K token）
就能做**品質有效**的 512K–768K 實驗。
若不可用，位置超過 262,144 就落在 RoPE 未訓練區間，只能量延遲不能量品質，
而且**換更大的卡也解決不了**（限制在模型不在記憶體）。

## 讀程式碼得到的預期（可能是錯的，所以要實測）

`DualChunkRotaryEmbedding` 完整存在，它產生一個 **5 倍寬的 query**
（query / query_succ / query_inter / 兩個 critical）。
但 `v1/attention/backends/` 底下**沒有任何一支消化這個 5 倍 query**，
`dual_chunk_attention_config` 只傳到 RoPE 層就沒有下文。
所以預期是「起得來但結果錯」或「shape 不合而崩潰」。

## 測法

大海撈針：在填充文字的指定深度插入一個魔術數字，然後問模型那個數字是多少。
關鍵在於**針的位置要超過 262,144**——那正是非 DCA 的有效範圍上限。
若 DCA 有效，模型答得出來；若無效，位置在未訓練區間，答案會是垃圾。

同時在 262,144 以內放一根針當**對照組**：若連對照組都答錯，
代表問題出在別的地方（模型、prompt 格式），不是 DCA。

用法：
  python code/dca_probe.py --gpu 0
"""
from __future__ import annotations
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_guard import GpuWatcher, host_contention, wait_until_free  # noqa: E402

BIG = Path(os.environ.get("PAPER_HKV_BIG", "/ssd7/hungwei/paper-hkv"))
VENV = BIG / "venv/vllm"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results/m1_capacity"
MODEL = "graelo/Qwen2.5-7B-Instruct-1M-AWQ"

FILLER = (
    "The archives of the northern observatory record the passage of comets, "
    "the drift of ice shelves, and the slow rotation of distant galaxies. "
    "Each observation is logged with the date, the instrument used, and the "
    "name of the astronomer on duty. The logs span several decades and are "
    "kept in bound volumes on the third floor. "
)
NEEDLE = "The magic access code for the north tower is {code}."
QUESTION = ("\n\nQuestion: What is the magic access code for the north tower? "
            "Answer with the number only.\n\nAnswer:")


def free_port() -> int:
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def build_prompt(tok, total_tokens: int, needle_pos: int,
                 code: str) -> tuple[str, int, int]:
    """做一個 total_tokens 長的 prompt，把針放在 needle_pos 附近。

    回傳 (prompt, 實際 token 數, **針的實際 token 位置**)。

    🔴 為什麼要回報實際位置：`decode` 之後再編碼，token 數會因為 BPE 在
       邊界重新合併而改變（實測漂移約 1.5%）。M3 的 `make_prefix` 踩過同一個
       坑，導致 ctx=32,768 的 prompt 超過 max_model_len、整批 HTTP 400。
       這裡的判定是「針在不在 262,144 之外」，用要求值會判錯邊界情形，
       所以一律以實際量到的位置為準。
    """
    unit = tok(FILLER, add_special_tokens=False)["input_ids"]
    need = tok(NEEDLE.format(code=code), add_special_tokens=False)["input_ids"]
    tail = tok(QUESTION, add_special_tokens=False)["input_ids"]
    body = total_tokens - len(tail)
    reps = body // len(unit) + 2
    ids = (unit * reps)[:body]
    ids = ids[:needle_pos] + need + ids[needle_pos:]
    ids = ids[:body] + tail
    text = tok.decode(ids, skip_special_tokens=True)
    final = tok(text, add_special_tokens=False)["input_ids"]
    at = next((i for i in range(len(final) - len(need) + 1)
               if final[i:i + len(need)] == need), -1)
    return text, len(final), at


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-len", type=int, default=524_288)
    ap.add_argument("--kv-dtype", default="fp8")
    ap.add_argument("--probes", type=int, nargs="*",
                    default=[131_072, 393_216],
                    help="針要放在哪些 token 位置。預設一個在 262,144 之內"
                         "（對照組），一個在之外（真正的 DCA 測試）")
    ap.add_argument("--prompt-tokens", type=int, default=460_000)
    ap.add_argument("--out", default=str(OUT / "dca_probe.json"))
    a = ap.parse_args()

    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-dca-probe"
    root = BIG / "runs" / run_id
    root.mkdir(parents=True, exist_ok=True)
    ok, got = wait_until_free(a.gpu, need_mib=22 * 1024, timeout_s=600)
    if not ok:
        print(f"🔴 GPU {a.gpu} 只有 {got} MiB 可用，不開跑")
        return 5

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)

    port = free_port()
    cmd = [str(VENV / "bin/vllm"), "serve", a.model,
           "--port", str(port), "--max-model-len", str(a.max_len),
           "--gpu-memory-utilization", "0.90",
           "--kv-cache-dtype", a.kv_dtype]
    (root / "cmd.sh").write_text(" ".join(shlex.quote(c) for c in cmd) + "\n")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    env.setdefault("PATH", "")
    env["PATH"] = f"{VENV / 'bin'}:{env['PATH']}"
    log = (root / "server.log").open("w")
    print(f"[dca] 啟動 {a.model}  max_len={a.max_len:,}  kv={a.kv_dtype}")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)

    res: dict = {"run_id": run_id, "model": a.model, "max_len": a.max_len,
                 "kv_dtype": a.kv_dtype, "log": str(root / "server.log"),
                 "host_contention": host_contention(exclude_gpu=a.gpu)["level"],
                 "ts": datetime.now().astimezone().isoformat()}
    t0 = time.time()
    ready = False
    with GpuWatcher(gpu=a.gpu, out_path=str(root / "gpu_guard.json")) as w:
        while time.time() - t0 < 1200:
            if proc.poll() is not None:
                res["verdict"] = "SERVER_DIED"
                res["exit_code"] = proc.returncode
                tail = (root / "server.log").read_text(errors="replace").splitlines()
                res["error_tail"] = [l for l in tail[-40:]
                                     if "Error" in l or "error" in l or "raise" in l][:8]
                break
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                ready = True
                break
            except Exception:  # noqa: BLE001
                time.sleep(3)
        else:
            res["verdict"] = "STARTUP_TIMEOUT"

        if ready:
            res["startup_s"] = round(time.time() - t0, 1)
            txt = (root / "server.log").read_text(errors="replace")
            m = re.search(r"GPU KV cache size: ([\d,]+) tokens", txt)
            res["kv_cache_tokens"] = int(m.group(1).replace(",", "")) if m else None
            res["dca_warning"] = "dual_chunk" in txt.lower()
            print(f"[dca] server up in {res['startup_s']}s，"
                  f"KV {res.get('kv_cache_tokens')}")
            probes = []
            for i, pos in enumerate(a.probes):
                code = f"{7000 + i * 137}"
                prompt, n, at = build_prompt(tok, a.prompt_tokens, pos, code)
                if at < 0:
                    print(f"[dca] 🔴 針在 decode/encode 之後找不到了，"
                          f"跳過位置 {pos:,}（BPE 邊界合併把它切碎了）")
                    probes.append({"needle_pos_requested": pos,
                                   "needle_pos_actual": None,
                                   "error": "needle not found after re-encode"})
                    continue
                beyond = at > 262_144
                print(f"[dca] 針要求在 {pos:,}、實際在 {at:,}"
                      f"（{'超出' if beyond else '在內'} 262,144）"
                      f"，prompt {n:,} token …", flush=True)
                body = json.dumps({"model": a.model, "prompt": prompt,
                                   "max_tokens": 16, "temperature": 0.0}).encode()
                t1 = time.time()
                try:
                    req = urllib.request.Request(
                        f"http://127.0.0.1:{port}/v1/completions", data=body,
                        headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=3600) as r:
                        out = json.load(r)["choices"][0]["text"]
                    err = None
                except Exception as e:  # noqa: BLE001
                    out, err = "", f"{type(e).__name__}: {e}"
                hit = code in (out or "")
                probes.append({"needle_pos_requested": pos,
                               "needle_pos_actual": at,
                               "beyond_262144": beyond,
                               "prompt_tokens": n, "expected": code,
                               "output": (out or "")[:120], "correct": hit,
                               "latency_s": round(time.time() - t1, 1),
                               "error": err})
                print(f"       -> {'✅ 答對' if hit else '🔴 答錯'}  "
                      f"輸出={(out or err or '')[:60]!r}  "
                      f"{probes[-1]['latency_s']}s")
            res["probes"] = probes
            probes = [p for p in probes if "beyond_262144" in p]
            inside = [p for p in probes if not p["beyond_262144"]]
            outside = [p for p in probes if p["beyond_262144"]]
            if inside and not all(p["correct"] for p in inside):
                res["verdict"] = "CONTROL_FAILED"   # 對照組就錯，問題不在 DCA
            elif outside and all(p["correct"] for p in outside):
                res["verdict"] = "DCA_WORKS"
            elif outside:
                res["verdict"] = "DCA_BROKEN"
            else:
                res["verdict"] = "NO_OUTSIDE_PROBE"
        res["contaminated"] = w.contaminated

    try:
        os.killpg(os.getpgid(proc.pid), 15)
    except Exception:  # noqa: BLE001
        pass
    proc.wait(timeout=120)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    print(f"\n判定：{res.get('verdict')}\nwrote {p}")
    print({"DCA_WORKS": "→ 512K–768K 可做**品質有效**的實驗，不需換卡換模型",
           "DCA_BROKEN": "→ 位置 >262,144 只能量延遲，品質無效。"
                         "要做有效的長上下文品質實驗必須換模型（不是換卡）",
           "CONTROL_FAILED": "→ 連 262,144 以內都答錯，問題不在 DCA，先查 prompt 與模型",
           "SERVER_DIED": "→ 起不來，看 error_tail",
           }.get(res.get("verdict"), ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
