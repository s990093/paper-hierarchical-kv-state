#!/usr/bin/env python3
"""gm12.py — 第三階段 G-M12（mechanisms/04 M12）：「先近似、後補精」的第一步——Cake 的前段改用稀疏注意力重算，品質掉多少、
每塊重算的斜率降多少。判準：docs/stage3_design_20261024/runs/G_M12_sparse_recompute.md（登記，不改）。

稀疏重算（逐塊，和 Cake 的重算線同一個順序）：重算第 i 塊（512 個 token）時，每層、每個 KV head 只看：
  block 0（sink）、block i−1（local）、block i 自己（causal），加上 block 1..i−2 裡分數最高的若干塊；
  分數＝這一塊 512 個 query、這一組 q head 的平均 q（RoPE 之後）· 每塊 K 的平均（RoPE 之後）。
  看的塊數合計＝max(3, round(d·(i+1)))（i ≥ 2）；d＝密度。
實作：把每個 KV head 選中的 key 接成一條序列（最後是自己這塊），用 fa3 的 varlen（每個 KV head 一條序列、rep 個 q head、1 個 KV head），
causal 以右下角對齊 → 前面的塊全部看得到、自己這塊 causal。計算量和選中的 key 數成正比（不是遮罩，真的少算）。d＝1 時＝dense（自檢）。

  quality --cell L128|Q128 [--shard i/n]  TF：E0（精確 KV）vs 前段（F＝0.5、1.0）用 d＝0.1、0.2 稀疏重算 → Δgold_lp
  speed   --model llama|qwen               每塊重算時間：dense fa3 vs 稀疏 d（block 16、64、128、192、248；6 次）（量時間、獨佔）
  report                                   判定（登記 §2）
  gpu-selftest --cell L128                 d＝1 的稀疏路徑＝KVModel fa3 的 KV（容差內）；選塊數正確
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import a0 as A  # noqa: E402
import guard as GD  # noqa: E402（量時間的 run 的 GPU 監看，CLAUDE.md §3）
import numa_probe as NP  # noqa: E402（只記錄：speed 是 GPU 上的重算，計時區間不碰主機緩衝 → 只記卡與 cpuset 在哪個節點）

CHUNK = 512
OUT = os.path.join(A.REPO, "results", "s3_h200", "gm12")
DENS = (0.1, 0.2)
FRONT = (0.5, 1.0)
CELLS = ("L128", "Q128")
SPEED_BLOCKS = (16, 64, 128, 192, 248)
QF = ["run_id", "ts", "cell", "model", "item_id", "task", "seed", "turn", "cid", "d", "front", "n_sparse_blocks", "keys_frac",
      "gold_lp", "gold_lp_e0", "d_gold_lp", "kl", "top1", "first_ok", "t_recompute_s", "own_gpu_intruders", "host_level", "n_rows_item"]


def n_select(i: int, d: float) -> int:
    """重算第 i 塊時看的塊數（含 sink、local、自己）。"""
    if i <= 2:
        return i + 1
    return min(i + 1, max(3, int(round(d * (i + 1)))))


# ============================================================================ GPU
def sparse_chunk(km, ids, i: int, d: float, pooled, stats=None):
    """重算第 i 塊（位置 i·512..）：逐層、每個 KV head 選塊，fa3 varlen。寫進 km.kv，並更新 pooled[l, :, i]（K 的平均）。"""
    import torch
    from vllm.vllm_flash_attn import flash_attn_varlen_func as fa
    m = km.hf.model
    n, start = CHUNK, i * CHUNK
    end = start + n
    H, Hq, D = km.H, km.Hq, km.D
    rep = Hq // H
    x = m.embed_tokens(ids[start:end].view(1, n))
    pos = torch.arange(start, end, device=km.device).view(1, n)
    cos, sin = m.rotary_emb(x, pos)
    ns = n_select(i, d)
    cu_q = torch.arange(0, (H + 1) * n, n, dtype=torch.int32, device=km.device)
    for li, layer in enumerate(m.layers):
        a = layer.self_attn
        h = layer.input_layernorm(x)
        q = a.q_proj(h).view(1, n, Hq, D)
        k = a.k_proj(h).view(1, n, H, D)
        if hasattr(a, "q_norm"):
            q, k = a.q_norm(q), a.k_norm(k)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        v = a.v_proj(h).view(1, n, H, D).transpose(1, 2)
        q, k = km.apply_rope(q, k, cos, sin)
        km.kv[li, 0, :, start:end].copy_(k[0])
        km.kv[li, 1, :, start:end].copy_(v[0])
        pooled[li, :, i] = k[0].float().mean(1)
        # 選塊（每個 KV head）
        if ns >= i + 1:
            sel = [list(range(i + 1))] * H
        else:
            qbar = q[0].float().mean(1).view(H, rep, D).mean(1)                         # [H, D]
            sc = torch.einsum("hd,hjd->hj", qbar, pooled[li, :, 1:i - 1])               # 候選 block 1..i−2
            top = sc.topk(ns - 3, dim=-1).indices + 1
            sel = [sorted([0] + top[hh].tolist() + [i - 1, i]) for hh in range(H)]
        Ks, Vs, lens = [], [], [0]
        for hh in range(H):
            idx = torch.cat([torch.arange(b * CHUNK, (b + 1) * CHUNK, device=km.device) for b in sel[hh]])
            Ks.append(km.kv[li, 0, hh].index_select(0, idx))
            Vs.append(km.kv[li, 1, hh].index_select(0, idx))
            lens.append(lens[-1] + idx.numel())
        Kc = torch.cat(Ks).unsqueeze(1)                                                 # [Σ, 1, D]
        Vc = torch.cat(Vs).unsqueeze(1)
        cu_k = torch.tensor(lens, dtype=torch.int32, device=km.device)
        qq = q[0].view(H, rep, n, D).permute(0, 2, 1, 3).reshape(H * n, rep, D).contiguous()
        o = fa(qq, Kc, Vc, max_seqlen_q=n, cu_seqlens_q=cu_q, max_seqlen_k=max(b - a_ for a_, b in zip(lens, lens[1:])),
               cu_seqlens_k=cu_k, softmax_scale=D ** -0.5, causal=True, fa_version=3)          # [H·n, rep, D]
        o = o.view(H, n, rep, D).permute(1, 0, 2, 3).reshape(1, n, Hq * D)
        x = x + a.o_proj(o)
        x = x + layer.mlp(layer.post_attention_layernorm(x))
        if stats is not None:
            stats["keys"] += sum(lens[j + 1] - lens[j] for j in range(H))
            stats["dense"] += H * end
    return None


def init_pooled(km, nb: int):
    import torch
    P = torch.zeros(km.L, km.H, nb, km.D, dtype=torch.float32, device=km.device)
    for b in range(nb):
        P[:, :, b] = km.kv[:, 0, :, b * CHUNK:(b + 1) * CHUNK].float().mean(2)
    return P


def sparse_front(km, ids, nb: int, d: float, front: float) -> dict:
    """block 1..b−1 用稀疏重算（block 0 本來就只看自己＝精確）；b＝round(front·nb)。block b 之後是原值（Cake 從慢層載入的）。"""
    import torch
    b = max(1, int(round(front * nb)))
    pooled = init_pooled(km, nb)
    st = dict(keys=0, dense=0)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        for i in range(1, b):
            sparse_chunk(km, ids, i, d, pooled, st)
    torch.cuda.synchronize()
    return dict(n_sparse=b - 1, keys_frac=(st["keys"] / st["dense"]) if st["dense"] else 1.0, t=time.time() - t0)


def cmd_quality(a):
    import torch
    import a0_gpu as G
    from m16_c1a_predict import finish_watch, host_level, intruders, start_watch
    c = A.CELLS[a.cell]
    model, nb = c["model"], c["nb"]
    out = a.out or os.path.join(OUT, "raw", f"gm12q_{a.cell}_s{a.shard.split('/')[0]}.csv.gz")
    paths = sorted(set(glob.glob(os.path.join(OUT, "raw", f"gm12q_{a.cell}_s*.csv.gz"))) | {out})
    todo = A.pending(["tf"], a.shard, [p for p in paths if os.path.exists(p)])
    if a.max_items:
        todo = todo[:a.max_items]
    print(f"[gm12 quality] {a.cell}：還有 {len(todo)} 題 → {out}", flush=True)
    if not todo:
        return
    sess = G.Session(model, nb, store="gpu", calib_xl=a.calib_xl or G.calib_path(model))
    rid = A.run_id()
    watch = start_watch(out)
    fo, w = A.open_out(out, QF)
    t_start = time.time()
    try:
        for kk, r in enumerate(todo):
            if a.budget_min and (time.time() - t_start) / 60 > a.budget_min:
                print(f"[gm12] 時間預算到了（做完 {kk} 題）", flush=True)
                break
            it = sess.T.make_item(r["task"], r["b1"], r["b2"], r["seed"], sess.ans.tok, sess.P.get(r["task"]), n_blocks=nb)
            st = G.ItemState(sess, it, r["turn"])
            km = sess.km
            ids = torch.tensor(it["context_ids"], device=km.device)
            lp0 = st.score()
            m0 = G.ItemState.metrics(lp0, lp0, st.gold_ids)
            rows = [dict(cid="E0", d="", front="", n_sparse_blocks=0, keys_frac=1.0, t_recompute_s="", **m0)]
            for d in DENS:
                for front in FRONT:
                    st.comp.apply(G.want_array(sess, [], "global", G.ORIG, G.ORIG))
                    info = sparse_front(km, ids, nb, d, front)
                    st.comp.unknown()
                    mm = G.ItemState.metrics(st.score(), lp0, st.gold_ids)
                    rows.append(dict(cid=f"S|{d}|{front}", d=d, front=front, n_sparse_blocks=info["n_sparse"], keys_frac=round(info["keys_frac"], 5),
                                     t_recompute_s=round(info["t"], 3), **mm))
            lvl, intr = host_level(), intruders(watch)
            for x in rows:
                x.update(run_id=rid, ts=A.ts(), cell=a.cell, model=model, item_id=it["item_id"], task=it["task"], seed=it["seed"],
                         turn=r["turn"], gold_lp_e0=m0["gold_lp"], d_gold_lp=x["gold_lp"] - m0["gold_lp"], own_gpu_intruders=intr,
                         host_level=lvl, n_rows_item=len(rows))
                w.writerow({f: x.get(f, "") for f in QF})
            fo.flush()
            print(f"[{kk + 1}/{len(todo)}] {it['item_id']} " + " ".join(f"{x['cid']}:{x.get('t_recompute_s', '')}s" for x in rows[1:]), flush=True)
            G.free_item(st)
            del st
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        fo.close()
        finish_watch(watch, out)


def cmd_speed(a):
    """每塊重算時間（量時間、獨佔）：位置 i ∈ SPEED_BLOCKS，dense（KVModel fa3）vs 稀疏 d；每點 1 次暖機＋6 次（05 §2.1）。"""
    import torch
    os.environ["M7_MODEL_GLOB"] = os.environ.get(f"M7_{a.model.upper()}_GLOB", "")
    from m7_model import KVModel
    from m16_c1a_predict import host_level
    nb = 252
    km = KVModel(max_len=nb * CHUNK + 1024)
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(1000, min(120000, km.cfg.vocab_size), (nb * CHUNK,), generator=g).to(km.device)
    km.set_attn("fa3")
    km.prefill_chunked(ids)
    torch.cuda.synchronize()
    pooled = init_pooled(km, nb)
    rows, rid = [], A.run_id()
    numa = NP.probe(torch.cuda.current_device())
    print("[gm12] NUMA：", json.dumps(numa["summary"], ensure_ascii=False), flush=True)
    os.makedirs(os.path.join(OUT, "raw"), exist_ok=True)
    p = a.out or os.path.join(OUT, "raw", f"gm12s_{a.model}_{rid}.csv")
    gw = GD.Timing(p)
    for i in SPEED_BLOCKS:
        for arm in ["dense"] + [f"sparse{d}" for d in DENS]:
            ts_ = []
            for rep in range(7):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    if arm == "dense":
                        km.forward_span(ids[i * CHUNK:(i + 1) * CHUNK], i * CHUNK)
                    else:
                        sparse_chunk(km, ids, i, float(arm[6:]), pooled)
                torch.cuda.synchronize()
                if rep:
                    ts_.append(time.perf_counter() - t0)
            sr = statistics.stdev(ts_) / statistics.median(ts_)
            rows.append(dict(run_id=rid, ts=A.ts(), model=a.model, block=i, arm=arm, n_rep=len(ts_), t_med_ms=round(1e3 * statistics.median(ts_), 4),
                             reps_ms=";".join(f"{1e3 * t:.4f}" for t in ts_), sigma_rel=round(sr, 6), delta_star=round(max(0.05, 2 * sr), 6),
                             host_level=host_level()))
            print(json.dumps(rows[-1]), flush=True)
    p = gw.finish()                                               # 污染 → run 目錄，不進 results
    for r in rows:
        r["own_gpu_intruders"] = gw.status()
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("[gm12] NUMA 存在", NP.save(p, rid, start=numa), flush=True)


def cmd_gpu_selftest(a):
    import torch
    import a0_gpu as G
    c = A.CELLS[a.cell]
    model, nb = c["model"], c["nb"]
    sess = G.Session(model, nb, store="gpu", calib_xl=a.calib_xl or G.calib_path(model))
    r = A.plan_set("tf")[0]
    it = sess.T.make_item(r["task"], r["b1"], r["b2"], r["seed"], sess.ans.tok, sess.P.get(r["task"]), n_blocks=nb)
    st = G.ItemState(sess, it, 0)
    km = sess.km
    ids = torch.tensor(it["context_ids"], device=km.device)
    ref = km.kv[:, :, :, :32 * CHUNK].clone()                                   # fa3 prefill 的 KV（前 32 塊）
    pooled = init_pooled(km, nb)
    with torch.inference_mode():
        for i in range(1, 32):
            sparse_chunk(km, ids, i, 1.0, pooled)
    diff = float((km.kv[:, :, :, :32 * CHUNK].float() - ref.float()).abs().max())
    out = dict(dense_path_max_abs_diff=diff, n_select=[n_select(i, 0.1) for i in (1, 2, 3, 10, 100, 251)],
               ok=diff < 0.25 and [n_select(i, 0.1) for i in (1, 2, 3)] == [2, 3, 3])
    st.comp.unknown()
    print(json.dumps(out), flush=True)
    if a.out:
        json.dump(dict(out, run_id=A.run_id(), ts=A.ts()), open(a.out, "w"), indent=1)
    if not out["ok"]:
        raise SystemExit("gm12 gpu-selftest 沒過")


def cmd_report(a):
    import numpy as np
    res = dict(run_id=A.run_id(), ts=A.ts(), quality={}, speed={})
    for cell in CELLS:
        rows = A.read_rows(sorted(glob.glob(os.path.join(OUT, "raw", f"gm12q_{cell}_s*.csv.gz"))))
        tab = A.table(rows, "d_gold_lp")
        res["quality"][cell] = {cid: A.boot(v.values()) for cid, v in tab.items() if cid != "E0"}
    res["excluded_contaminated_runs"] = []
    for p in sorted(glob.glob(os.path.join(OUT, "raw", "gm12s_*.csv"))):
        rows, bad = GD.drop_contaminated(list(csv.DictReader(open(p))))
        res["excluded_contaminated_runs"] += bad
        if not rows:
            continue
        m = rows[0]["model"]
        by = {}
        for r in rows:
            by.setdefault(r["arm"], []).append((int(r["block"]), [float(x) for x in r["reps_ms"].split(";")], float(r["sigma_rel"])))
        slope = {}
        rng = np.random.default_rng(0)
        for arm, pts in by.items():
            xs = np.array([b for b, _, _ in pts], dtype=float)
            ys = np.array([statistics.median(v) for _, v, _ in pts])
            boots = []
            for _ in range(10000):
                yb = np.array([np.median(rng.choice(v, len(v))) for _, v, _ in pts])
                boots.append(np.polyfit(xs, yb, 1)[0])
            slope[arm] = dict(slope_ms_per_block=float(np.polyfit(xs, ys, 1)[0]), ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                              max_2sigma=max(2 * s for _, _, s in pts))
        res["speed"][m] = slope
    # 判定（登記 §2）
    # 三值：True／False／None（量不到或太晃，判不了）。「兩格都要過」→ 任何一個量到的不過就是 False，缺的只在其餘都過時才讓結果變 None
    def all3(xs):
        return False if False in xs else (None if None in xs else True)
    verdict = {}
    for d in DENS:
        qs = []
        for cell in CELLS:
            b = res["quality"].get(cell, {}).get(f"S|{d}|1.0")
            qs.append(None if not b else b["lo"] >= -2.0)          # Q128 沒跑（使用者 10/28 收掉）→ None
        ss = []
        for m in ("llama", "qwen"):
            sl = res["speed"].get(m, {})
            dn, sp = sl.get("dense"), sl.get(f"sparse{d}")
            if not (dn and sp):
                ss.append(None)
            elif max(dn["max_2sigma"], sp["max_2sigma"]) > 0.05:      # 登記 §2：任何一點 2σ > 5% → 先回報、不判定
                ss.append(None)
            else:
                ss.append(sp["slope_ms_per_block"] / dn["slope_ms_per_block"] <= 0.3 and sp["ci"][1] / dn["slope_ms_per_block"] <= 0.3)
        q_ok, s_ok = all3(qs), all3(ss)
        verdict[str(d)] = dict(quality_by_cell=dict(zip(CELLS, qs)), speed_by_model=dict(zip(("llama", "qwen"), ss)),
                               quality_ok=q_ok, speed_ok=s_ok, passed=(False if False in (q_ok, s_ok) else (None if None in (q_ok, s_ok) else True)))
    ps = [v["passed"] for v in verdict.values()]
    res["verdict"] = dict(by_density=verdict, missing_quality=[cell for cell in CELLS if not res["quality"].get(cell)],
                          G_M12="PASS" if True in ps else ("STOP" if all(p is False for p in ps) else "NOT_DETERMINED"))
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "gm12_report.json"), "w"), indent=1, ensure_ascii=False, default=str)
    print(json.dumps(res["verdict"], ensure_ascii=False))


def cmd_diag(a):
    """只報告（RUNLOG S3-GM12-FAIL-1 的診斷；自檢與判準都不改）。d＝1 時逐層比較 sparse_chunk 和 fa3 prefill 的 KV，
    兩輪：default、兩條路徑都強制 fa3 num_splits=1。第 0 層的 K／V 不經過 attention → 若只是加總順序造成的捨入差，第 0 層必為 0。"""
    import gc
    import torch
    import vllm.vllm_flash_attn as VF
    import a0_gpu as G
    c = A.CELLS[a.cell]
    model, nb = c["model"], c["nb"]
    sess = G.Session(model, nb, store="gpu", calib_xl=a.calib_xl or G.calib_path(model))
    r = A.plan_set("tf")[0]
    it = sess.T.make_item(r["task"], r["b1"], r["b2"], r["seed"], sess.ans.tok, sess.P.get(r["task"]), n_blocks=nb)
    km = sess.km
    ids = torch.tensor(it["context_ids"], device=km.device)
    orig = VF.flash_attn_varlen_func
    out = dict(cell=a.cell, model=model, run_id=A.run_id(), ts=A.ts(), item=it["item_id"], modes={})
    for mode in ("default", "splits1"):
        if mode == "splits1":
            def fa1(*x, **k):
                k["num_splits"] = 1
                return orig(*x, **k)
            VF.flash_attn_varlen_func = fa1          # KVModel.set_attn 與 sparse_chunk 都在呼叫時才從模組取
        else:
            VF.flash_attn_varlen_func = orig
        try:
            st = G.ItemState(sess, it, 0)                                      # fa3 prefill（這一輪的 fa 設定）
            ref = km.kv[:, :, :, :32 * CHUNK].clone()
            pooled = init_pooled(km, nb)
            with torch.inference_mode():
                for i in range(1, 32):
                    sparse_chunk(km, ids, i, 1.0, pooled)
            new = km.kv[:, :, :, :32 * CHUNK]
            layers = []
            for li in range(km.L):
                row = dict(layer=li)
                for kv, name in ((0, "k"), (1, "v")):
                    rf, nw = ref[li, kv].float(), new[li, kv].float()
                    d = (nw - rf).abs()
                    j = int(d.argmax())
                    v = float(rf.flatten()[j])
                    ulp = 2.0 ** (math.floor(math.log2(abs(v))) - 7) if v != 0 else 2.0 ** -133
                    row.update({f"{name}_max_diff": round(float(d.max()), 6), f"{name}_max_abs_ref": round(float(rf.abs().max()), 4),
                                f"{name}_rel": round(float(d.max()) / max(1e-12, float(rf.abs().max())), 8),
                                f"{name}_ulps_at_max": round(float(d.max()) / ulp, 3),
                                f"{name}_frac_neq": round(float((d > 0).float().mean()), 6)})
                layers.append(row)
            out["modes"][mode] = dict(max_diff=max(max(l_["k_max_diff"], l_["v_max_diff"]) for l_ in layers),
                                      first_layer_with_diff=next((l_["layer"] for l_ in layers if max(l_["k_max_diff"], l_["v_max_diff"]) > 0), None),
                                      layers=layers)
            print(mode, json.dumps({k: v for k, v in out["modes"][mode].items() if k != "layers"}), flush=True)
            G.free_item(st)
            del st, ref, new
            gc.collect()
            torch.cuda.empty_cache()
        finally:
            VF.flash_attn_varlen_func = orig
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({m: {k: v for k, v in x.items() if k != "layers"} for m, x in out["modes"].items()}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    for name in ("quality", "gpu-selftest", "diag"):
        s = sub.add_parser(name)
        s.add_argument("--cell", choices=CELLS, required=True)
        s.add_argument("--shard", default="0/1")
        s.add_argument("--budget-min", type=float, default=0)
        s.add_argument("--max-items", type=int, default=0)
        s.add_argument("--calib-xl", default="")
        s.add_argument("--out", default="")
    s = sub.add_parser("speed")
    s.add_argument("--model", choices=["llama", "qwen"], required=True)
    s.add_argument("--out", default="")
    sub.add_parser("report")
    a = ap.parse_args()
    if a.self_test:
        ok = [n_select(i, 0.1) for i in (1, 2, 3, 10, 100, 251)] == [2, 3, 3, 3, 10, 25]
        ok &= [n_select(i, 0.2) for i in (10, 100)] == [3, 20]
        print("self-test", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    {"quality": cmd_quality, "speed": cmd_speed, "gpu-selftest": cmd_gpu_selftest, "diag": cmd_diag, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
