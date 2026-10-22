"""attn_bench.py — ICPE P0-1 步驟 S1（docs/h200/icpe/P0-1_recompute_speed.md §2）：H200 上最快、而且算得對的 attention。

  python code/icpe/attn_bench.py run --model llama --out-dir results/icpe_h200/p01/attn [--cands ...] [--rounds 3]
  python code/icpe/attn_bench.py choose --out-dir results/icpe_h200/p01/attn      # 兩個模型都量完：每個模型的勝者 → choice.json

run：
  §2.2 正確性閘：固定 seed 的隨機 token，8K、32K；HF 整段 prefill（logits_to_keep=1）當參考；
       每個候選以 512 一塊依序 prefill，比最後一個位置的 logits（argmax、top-5 重疊數、max|Δ|）。→ gate.csv
       （§8 偏離 2，10/22 冒煙之後、正式資料之前：top-5 從「集合相同」改成「至少 4 個相同」——冒煙裡 sdpa 自己在 2K 就不相同）
       接線檢查：第 0 層的 K、V（attention 之前算的）必須和 sdpa 逐位元相同，不同就停。
  §2.3 速度：C1 協定（暖機 4 塊，chunk 0–63 每塊前後 synchronize），rounds 輪，每輪候選順序輪換。→ speed.csv
choose：§2.2 的過關規則＋§2.3 的分數（Σ_{i<64} 每塊各輪中位數）→ 每個模型的勝者；勝者沒有比 sdpa 快 ≥5% → stop_loss。
每列都有 run_id、ts；GpuWatcher 全程監看，污染時 contaminated 欄＝1，並在 run 目錄留 CONTAMINATED。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)

CANDS = ["sdpa", "fa2", "fa3", "fa4", "sdpa_cudnn"]
LENS = (8192, 32768)
N_SPEED = 64          # chunk 0–63（32K）


def run_id():
    return os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc"


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


class Out:
    def __init__(self, path, fields):
        new = not os.path.exists(path)
        self.f = open(path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=["run_id", "ts"] + fields, extrasaction="ignore")
        if new:
            self.w.writeheader()

    def row(self, **kw):
        kw.update(run_id=run_id(), ts=ts())
        self.w.writerow(kw)
        self.f.flush()
        print(json.dumps(kw, ensure_ascii=False), flush=True)


def cmd_run(a):
    import torch
    import m10_tasks
    os.environ["M7_MODEL_GLOB"] = m10_tasks.MODELS[a.model]
    from gpu_guard import GpuWatcher, smi_index
    from m7_model import CHUNK, KVModel

    os.makedirs(a.out_dir, exist_ok=True)
    rdir = os.path.join(os.environ.get("TIARA_RUNS", "."), run_id())
    gate = Out(os.path.join(a.out_dir, "gate.csv"),
               ["model", "cand", "length", "status", "argmax_same", "top5_same", "top5_overlap", "max_abs_dlogit", "hf_argmax", "cand_argmax",
                "kv_l0_same_as_sdpa", "kv_last_max_abs_vs_sdpa", "error", "contaminated"])
    speed = Out(os.path.join(a.out_dir, "speed.csv"),
                ["model", "cand", "round", "pos_in_round", "chunk_idx", "pos_end", "ms", "contaminated"])
    with GpuWatcher(gpu=smi_index(0), own_root=os.getpid(), out_path=os.path.join(a.out_dir, f"gpu_guard_{run_id()}.json")) as w:
        lens = tuple(a.lens)
        maxL = max(lens)
        km = KVModel(max_len=maxL + CHUNK)
        vocab = km.cfg.vocab_size
        g = torch.Generator().manual_seed(0)
        ids = torch.randint(1000, min(120000, vocab), (maxL,), generator=g).to(km.device)
        ok = {}
        # ---- §2.2 正確性閘
        for L in lens:
            with torch.inference_mode():
                ref = km.hf(input_ids=ids[:L].view(1, L), logits_to_keep=1, use_cache=False).logits[0, -1].float()
            torch.cuda.synchronize()
            l0_sdpa = last_sdpa = None
            for c in a.cands:
                row = dict(model=a.model, cand=c, length=L)
                try:
                    km.set_attn(c)
                    km.kv.zero_()
                    with torch.inference_mode():
                        for s in range(0, L, CHUNK):
                            lg = km.forward_span(ids[s:s + CHUNK], s, want_logits=(s + CHUNK == L))
                    torch.cuda.synchronize()
                    lg = lg.float()
                    top_ref = set(ref.topk(5).indices.tolist())
                    top_c = set(lg.topk(5).indices.tolist())
                    l0 = km.kv[0, :, :, :L].clone()
                    last = km.kv[-1, :, :, :L].float()
                    if c == "sdpa":
                        l0_sdpa, last_sdpa = l0, last.clone()
                    row.update(status="ok", argmax_same=int(int(ref.argmax()) == int(lg.argmax())), top5_same=int(top_ref == top_c),
                               top5_overlap=len(top_ref & top_c),
                               max_abs_dlogit=round(float((lg - ref).abs().max()), 6), hf_argmax=int(ref.argmax()),
                               cand_argmax=int(lg.argmax()),
                               kv_l0_same_as_sdpa=("" if l0_sdpa is None else int(torch.equal(l0, l0_sdpa))),
                               kv_last_max_abs_vs_sdpa=("" if last_sdpa is None else round(float((last - last_sdpa).abs().max()), 6)))
                    del l0, last
                except Exception as e:  # noqa：候選出局，錯誤照記（§2.1）
                    import traceback
                    row.update(status="error", error=(type(e).__name__ + ": " + str(e))[:500])
                    traceback.print_exc()
                row["contaminated"] = int(w.contaminated)
                gate.row(**row)
                if row.get("kv_l0_same_as_sdpa") == 0:
                    raise SystemExit(f"[attn_bench] {c} 的第 0 層 KV 和 sdpa 不同（attention 之前就不同）→ 接線錯，停")
                ok.setdefault(c, True)
                ok[c] = ok[c] and row["status"] == "ok"
            del ref
            torch.cuda.empty_cache()
        # ---- §2.3 速度（只量載入、呼叫都沒出錯的候選）
        cands = [c for c in a.cands if ok.get(c)]
        for rnd in range(a.rounds):
            order = cands[rnd % len(cands):] + cands[:rnd % len(cands)]
            for pos, c in enumerate(order):
                km.set_attn(c)
                with torch.inference_mode():
                    for i in range(4):
                        km.forward_span(ids[i * CHUNK:(i + 1) * CHUNK], i * CHUNK)
                    torch.cuda.synchronize()
                    for i in range(a.n_speed):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        km.forward_span(ids[i * CHUNK:(i + 1) * CHUNK], i * CHUNK)
                        torch.cuda.synchronize()
                        speed.row(model=a.model, cand=c, round=rnd, pos_in_round=pos, chunk_idx=i, pos_end=(i + 1) * CHUNK,
                                  ms=round((time.perf_counter() - t0) * 1e3, 3), contaminated=int(w.contaminated))
    if w.contaminated:
        os.makedirs(rdir, exist_ok=True)
        open(os.path.join(rdir, "CONTAMINATED"), "w").write(json.dumps(sorted(w.intruders)) + "\n")
        print("[attn_bench] CONTAMINATED：這次的時間不可用", flush=True)
        sys.exit(3)


def cmd_choose(a):
    gate = list(csv.DictReader(open(os.path.join(a.out_dir, "gate.csv"))))
    speed = list(csv.DictReader(open(os.path.join(a.out_dir, "speed.csv"))))
    if any(r["contaminated"] == "1" for r in gate + speed):
        raise SystemExit("[attn_bench choose] 有污染的列（contaminated=1）→ 停，要重量")
    out = {"run_id": run_id(), "ts": ts(), "rule": "P0-1 §2.2–§2.3（＋§8 偏離 2）：過閘＝argmax 相同、top-5 至少 4 個相同、max|Δlogit| ≤ max(0.25, 2×sdpa)；"
           "分數＝Σ_{i<64} 每塊各輪中位數；勝者沒有比 sdpa 快 ≥5% → stop_loss", "models": {}}
    for m in sorted({r["model"] for r in gate}):
        g = [r for r in gate if r["model"] == m]
        sd = {int(r["length"]): float(r["max_abs_dlogit"]) for r in g if r["cand"] == "sdpa" and r["status"] == "ok"}
        if len(sd) != len(LENS):
            raise SystemExit(f"[attn_bench choose] {m}：sdpa 自己沒有過閘的資料 → 停")
        passed, why = {}, {}
        for c in sorted({r["cand"] for r in g}):
            rows = [r for r in g if r["cand"] == c]
            p = len(rows) == len(LENS) and all(
                r["status"] == "ok" and r["argmax_same"] == "1" and int(r["top5_overlap"]) >= 4      # §8 偏離 2：top-5 至少 4 個相同
                and float(r["max_abs_dlogit"]) <= max(0.25, 2 * sd[int(r["length"])]) for r in rows)
            passed[c] = p
            why[c] = [{k: r.get(k) for k in ("length", "status", "argmax_same", "top5_same", "top5_overlap", "max_abs_dlogit", "error")} for r in rows]
        sc = {}
        for c in sorted({r["cand"] for r in speed if r["model"] == m}):
            per = {}
            for r in speed:
                if r["model"] == m and r["cand"] == c:
                    per.setdefault(int(r["chunk_idx"]), []).append(float(r["ms"]))
            if sorted(per) != list(range(N_SPEED)):
                continue
            sc[c] = round(sum(statistics.median(v) for v in per.values()), 3)
        elig = {c: s for c, s in sc.items() if passed.get(c)}
        win = min(elig, key=elig.get) if elig else None
        stop = win is None or "sdpa" not in sc or elig[win] > 0.95 * sc["sdpa"]
        out["models"][m] = dict(passed=passed, gate_detail=why, score_ms=sc, winner=win,
                                winner_vs_sdpa=(round(elig[win] / sc["sdpa"], 4) if win and "sdpa" in sc else None),
                                stop_loss=bool(stop))
    json.dump(out, open(os.path.join(a.out_dir, "choice.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps({m: {k: v[k] for k in ("passed", "score_ms", "winner", "winner_vs_sdpa", "stop_loss")}
                      for m, v in out["models"].items()}, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--model", choices=["llama", "qwen"], required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--cands", nargs="+", default=CANDS)
    r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--lens", type=int, nargs="+", default=list(LENS), help="正確性閘的長度（正式＝8192 32768；冒煙才改）")
    r.add_argument("--n-speed", type=int, default=N_SPEED, help="速度量幾塊（正式＝64；冒煙才改）")
    c = sp.add_parser("choose")
    c.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    {"run": cmd_run, "choose": cmd_choose}[a.cmd](a)


if __name__ == "__main__":
    main()
