"""m10_drop.py — 14 號 §8：直接丟掉 token 的大規模掃描（DS），只報告（docs/precision_20261013/14_match_drop_plan.md）。

被丟的位置用 KVModel.key_keep 遮住（＝從 cache 拿掉；RoPE 位置不變）。分數的定義照 NVIDIA kvpress 0.5.5 的寫法重寫
（cards/P08_token_drop_literature.md）；我們的 KV 緩衝是自己管的，所以不能直接掛它的 press。

子命令：
  screen         DS1 海選（teacher forcing、第 1 輪；09 海選同一套 Item）→ 逐列 CSV（放 runs，不進 git）
  screen-report  DS1 的表（每個（粒度, 挑法, 佔比）對 E0、對 RND 的差＋bootstrap CI）
  pick           14 §8.5：照 DS1 寫出 DS2 的配置與 jobs（CPU）
  confirm-report DS2 的判讀（ε、對 RND0＋Holm、第 2 輪減第 1 輪、MASK0）
  speed-report   D3、M1、M2 的速度表（讀 m10_speed 的 restores.csv）

挑法（分數高的留下）：見 METHODS；粒度 T＝per (層, KV head) token、P＝16 token 的 page、B＝512 token 的 block。
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import random
import statistics
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.dirname(HERE)

N_SINK = 4                 # StreamingLLM／ExpectedAttention 的 sink（kvpress 預設）
WINDOW = 64                # SnapKV 的觀察窗（kvpress 預設）
POOL = 5                   # SnapKV 的 avg-pool kernel（kvpress 預設）
PYR_BETA = 20              # PyramidKV（kvpress 預設）
ADA_ALPHA = 0.20           # AdaKV 的每個 head 最少保留比例（kvpress 預設）
EXP_FUTURE = 512           # ExpectedAttention 的未來位置數（kvpress 預設）
H2O_BLOCKS = (7, 15, 23, 31, 39, 47, 55, 63)   # H2O 近似：這些 block 的最後 64 個 token 當觀察窗（14 §8.2）
FRACS = [0.10, 0.25, 0.50, 0.75, 0.90]
UNIT = {"T": 1, "P": 16, "B": 512}
WRITE_TIME = ["SNAP", "TOVA", "H2O", "PYR", "ADA", "EXPATT", "KNORM", "KEYDIFF"]
POSITION = ["STREAM", "NEW", "RND0", "RND1", "RND2"]
METHODS = {"T": POSITION + WRITE_TIME + ["QRY"],
           "P": POSITION + [m for m in WRITE_TIME if m not in ("PYR", "ADA")] + ["QRY", "ORC"],
           "B": POSITION + [m for m in WRITE_TIME if m not in ("PYR", "ADA")] + ["QRY", "ORC"]}
SCORED = ["SNAP", "TOVA", "H2O", "EXPATT", "KNORM", "KEYDIFF", "QRY"]     # 要從 KV 算分數的（PYR、ADA 用 SNAP 的）


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ============================================================================ 分數（每個（層, KV head, context token）一個）
class Ctx:
    """一題的 context：km（緩衝裡 [0, ctx_len) 是原本 prefill 的 BF16 KV）、ids、每個 block 的 BF16 備份（觀察窗算完要放回）。"""

    def __init__(self, km, ids, ctx_len: int, bf16_blocks: list, chunk: int, seed: int, needle_blocks: list, query_ids: list):
        self.km, self.ids, self.ctx_len, self.bf16, self.chunk = km, ids, ctx_len, bf16_blocks, chunk
        self.seed, self.needle, self.query_ids = seed, list(needle_blocks), list(query_ids)
        self._scores: dict = {}

    def restore_span(self, s: int, e: int):
        """把 [s, e) 放回原本的 BF16（觀察窗的前向會寫進緩衝）。"""
        c = self.chunk
        for b in range(s // c, (min(e, self.ctx_len) - 1) // c + 1):
            self.km.chunk_view(b).copy_(self.bf16[b])


def _window_probs(cx: Ctx, ids_win, start: int, key_end: int):
    """觀察窗 ids_win（位置 start..start+w−1）跑過每一層；回傳每層、每個 query head 的 attention 機率
    在 key [0, key_end) 上對觀察窗 query 的**加總** [L, Hq, key_end]（float32），以及每個 key 被幾個 query 看到 [key_end]。
    softmax 在 [0, start+w) 上做（causal），和 kvpress compute_window_attention 相同；key_keep 必須是 None。"""
    import torch
    import torch.nn.functional as F
    from torch.nn.attention.bias import causal_lower_right
    km = cx.km
    assert km.key_keep is None
    w = len(ids_win)
    end = start + w
    m = km.hf.model
    rep = km.Hq // km.H
    out = torch.zeros(km.L, km.Hq, key_end, device=km.device, dtype=torch.float32)
    ids = torch.as_tensor(ids_win, device=km.device).view(1, w)
    with torch.inference_mode():
        x = m.embed_tokens(ids)
        pos = torch.arange(start, end, device=km.device).view(1, w)
        cos, sin = m.rotary_emb(x, pos)
        mask = causal_lower_right(w, end)
        causal = torch.ones(w, w, device=km.device, dtype=torch.bool).tril()
        for li, layer in enumerate(m.layers):
            a = layer.self_attn
            h = layer.input_layernorm(x)
            q = a.q_proj(h).view(1, w, km.Hq, km.D)
            k = a.k_proj(h).view(1, w, km.H, km.D)
            if hasattr(a, "q_norm"):
                q, k = a.q_norm(q), a.k_norm(k)
            q, k = q.transpose(1, 2), k.transpose(1, 2)
            v = a.v_proj(h).view(1, w, km.H, km.D).transpose(1, 2)
            q, k = km.apply_rope(q, k, cos, sin)
            km.kv[li, 0, :, start:end].copy_(k[0])
            km.kv[li, 1, :, start:end].copy_(v[0])
            K = km.kv[li, 0, :, :end]
            for hk in range(km.H):
                qs = q[0, hk * rep:(hk + 1) * rep].float()                      # [rep, w, D]
                s = qs @ K[hk].float().T / math.sqrt(km.D)                      # [rep, w, end]
                s[:, :, start:].masked_fill_(~causal, float("-inf"))
                out[li, hk * rep:(hk + 1) * rep] += s.softmax(-1)[:, :, :key_end].sum(1)
            o = F.scaled_dot_product_attention(q, km.kv[li, 0, :, :end].unsqueeze(0), km.kv[li, 1, :, :end].unsqueeze(0),
                                               attn_mask=mask, enable_gqa=True)
            x = x + a.o_proj(o.transpose(1, 2).reshape(1, w, km.Hq * km.D))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
    seen = (torch.arange(start, end, device=km.device).view(-1, 1) >= torch.arange(key_end, device=km.device).view(1, -1)).sum(0)
    if start < cx.ctx_len:
        cx.restore_span(start, end)
    return out, seen.float()


def _pool_group(p, km):
    """SnapKV：每個 query head 的分數 avg-pool（kernel 5、padding 2、stride 1），再在 GQA 組內平均 → [L, H, k]。"""
    import torch.nn.functional as F
    L, Hq, k = p.shape
    p = F.avg_pool1d(p.reshape(L * Hq, 1, k), kernel_size=POOL, padding=POOL // 2, stride=1).view(L, Hq, k)
    return p.view(L, km.H, Hq // km.H, k).mean(2)


def score_snap(cx: Ctx):
    import torch
    n = cx.ctx_len
    s0 = n - WINDOW
    p, _ = _window_probs(cx, cx.ids[s0:n].tolist(), s0, s0)
    sc = _pool_group(p / WINDOW, cx.km)
    return torch.cat([sc, torch.full_like(sc[..., :WINDOW], float(sc.max()) + 1)], -1)        # 觀察窗保留


def score_tova(cx: Ctx):
    import torch
    n = cx.ctx_len
    p, _ = _window_probs(cx, cx.ids[n - 1:n].tolist(), n - 1, n - 1)
    sc = p.mean(1, keepdim=True).expand(-1, cx.km.H, -1)                                       # 所有 head 平均，每個 KV head 一樣
    return torch.cat([sc, torch.full_like(sc[..., :1], float(sc.max()) + 1)], -1).contiguous()


def score_h2o(cx: Ctx):
    """ObservedAttention（H2O）近似：8 個均勻的 64-token 觀察窗，加總 ÷ 看得到的 query 數，GQA 組內平均。"""
    import torch
    km, n, c = cx.km, cx.ctx_len, cx.chunk
    tot = torch.zeros(km.L, km.Hq, n, device=km.device)
    cnt = torch.zeros(n, device=km.device)
    for b in H2O_BLOCKS:
        e = (b + 1) * c
        if e > n:
            continue
        s0 = e - WINDOW
        p, seen = _window_probs(cx, cx.ids[s0:e].tolist(), s0, e)
        tot[..., :e] += p
        cnt[:e] += seen
    sc = tot / cnt.clamp(min=1)
    return sc.view(km.L, km.H, km.Hq // km.H, n).mean(2)


def score_qry(cx: Ctx):
    """還原時知道第 1 輪問題：問題的 token 當觀察窗（SnapKV 的算法；所有 context token 都有分數）。"""
    q = cx.query_ids
    p, _ = _window_probs(cx, q, cx.ctx_len, cx.ctx_len)
    return _pool_group(p / len(q), cx.km)


def score_knorm(cx: Ctx):
    import torch
    K = cx.km.kv[:, 0, :, :cx.ctx_len]
    return -torch.linalg.vector_norm(K, dim=-1, dtype=torch.float32)


def score_keydiff(cx: Ctx):
    import torch
    import torch.nn.functional as F
    K = cx.km.kv[:, 0, :, :cx.ctx_len].float()
    anchor = F.normalize(K, p=2, dim=-1).mean(dim=2, keepdim=True)
    return -F.cosine_similarity(K, anchor, dim=-1)


def score_expatt(cx: Ctx):
    """ExpectedAttention（kvpress）：context 全部 query（去掉前 4 個）的 RoPE 前平均、共變異數 → 接下來 512 個位置的平均 RoPE →
    exp(K·μ/√d ＋ ½ KᵀΣK/d) 在 key（去掉 sink）上的 softmax → GQA 組內平均 → × ‖V‖；sink 保留。
    query 要每一層的 hidden state：照 prefill 的切法再跑一次 context（K／V 直接用緩衝裡原本的，不寫回）。"""
    import torch
    import torch.nn.functional as F
    from torch.nn.attention.bias import causal_lower_right
    km, n, c = cx.km, cx.ctx_len, cx.chunk
    m = km.hf.model
    D, Hq, H = km.D, km.Hq, km.H
    s1 = torch.zeros(km.L, Hq, D, device=km.device, dtype=torch.float64)
    s2 = torch.zeros(km.L, Hq, D, D, device=km.device, dtype=torch.float64)
    cnt = 0
    with torch.inference_mode():
        for s in range(0, n, c):
            e = min(n, s + c)
            w = e - s
            x = m.embed_tokens(cx.ids[s:e].view(1, w))
            pos = torch.arange(s, e, device=km.device).view(1, w)
            cos, sin = m.rotary_emb(x, pos)
            mask = causal_lower_right(w, e)
            keep = slice(max(0, N_SINK - s), w)
            for li, layer in enumerate(m.layers):
                a = layer.self_attn
                h = layer.input_layernorm(x)
                q = a.q_proj(h).view(1, w, Hq, D)
                if hasattr(a, "q_norm"):
                    q = a.q_norm(q)
                qq = q[0, keep].double()                                         # [m, Hq, D]（RoPE 前）
                s1[li] += qq.sum(0)
                s2[li] += torch.einsum("nhi,nhj->hij", qq, qq)
                q = q.transpose(1, 2)
                q, _ = km.apply_rope(q, q, cos, sin)
                o = F.scaled_dot_product_attention(q, km.kv[li, 0, :, :e].unsqueeze(0), km.kv[li, 1, :, :e].unsqueeze(0),
                                                   attn_mask=mask, enable_gqa=True)
                x = x + a.o_proj(o.transpose(1, 2).reshape(1, w, Hq * D))
                x = x + layer.mlp(layer.post_attention_layernorm(x))
            cnt += w - keep.start
        mu = s1 / cnt                                                            # [L, Hq, D]
        cov = s2 / cnt - torch.einsum("lhi,lhj->lhij", mu, mu)                   # 母體共變異數（kvpress 除以 n）
        # 平均 RoPE：R＝mean_t(cos_t·I ＋ sin_t·P)，P 把後半維換成 −前半（rotate_half 的矩陣）
        pos = torch.arange(n, n + EXP_FUTURE, device=km.device).view(1, -1)
        cos, sin = m.rotary_emb(torch.zeros(1, 1, D, device=km.device, dtype=torch.float32), pos)
        cos, sin = cos[0].double(), sin[0].double()                              # [T, D]
        I = torch.eye(D, device=km.device, dtype=torch.float64)
        P = torch.zeros(D, D, device=km.device, dtype=torch.float64)
        P[D // 2:, :D // 2], P[:D // 2, D // 2:] = torch.eye(D // 2), -torch.eye(D // 2)
        R = (cos.unsqueeze(-1) * I + sin.unsqueeze(-1) * P).mean(0)              # [D, D]
        mu = mu @ R.T
        cov = R @ cov @ R.T
        out = torch.empty(km.L, H, n, device=km.device, dtype=torch.float32)
        rep = Hq // H
        for li in range(km.L):
            K = km.kv[li, 0, :, N_SINK:n].double()                               # [H, k, D]
            Vn = torch.linalg.vector_norm(km.kv[li, 1, :, N_SINK:n].float(), dim=-1)    # [H, k]
            for hk in range(H):
                acc = 0
                for r in range(rep):
                    hq = hk * rep + r
                    lin = K[hk] @ mu[li, hq] / math.sqrt(D)
                    quad = ((K[hk] @ cov[li, hq]) * K[hk]).sum(-1) / D / 2
                    acc = acc + torch.softmax(lin + quad, dim=-1)
                out[li, hk, N_SINK:] = (acc / rep).float() * Vn[hk]
        out[..., :N_SINK] = float(out[..., N_SINK:].max()) + 1
    return out


SCORERS = {"SNAP": score_snap, "TOVA": score_tova, "H2O": score_h2o, "QRY": score_qry, "KNORM": score_knorm,
           "KEYDIFF": score_keydiff, "EXPATT": score_expatt}


def scores(cx: Ctx, method: str):
    if method in ("PYR", "ADA"):
        method = "SNAP"
    if method not in cx._scores:
        cx._scores[method] = SCORERS[method](cx)
    return cx._scores[method]


# ============================================================================ 挑 → 遮罩
def n_kept(k_len: int, frac: float) -> int:
    """kvpress compute_n_kept：max(1, int(k_len × (1 − ratio)))；ratio ≥ 1 時 0。"""
    return 0 if frac >= 1 else max(1, int(k_len * (1 - frac)))


def pyr_budget(k_len: int, frac: float, n_layers: int) -> list:
    """kvpress PyramidKV.get_layer_budget（window 64、beta 20）；不合條件時退回 SnapKV 的預算。"""
    out = []
    mcp = WINDOW + k_len * (1 - frac)
    mn = (mcp - WINDOW) / PYR_BETA
    mx = (mcp - WINDOW) * 2 - mn
    if mx >= k_len - WINDOW:
        mx = k_len - WINDOW
        mn = (mcp - WINDOW) * 2 - mx
    if not (k_len >= mx >= mn >= WINDOW):
        return [max(1, round(k_len * (1 - frac)))] * n_layers
    steps = (mx - mn) / (n_layers - 1)
    for li in range(n_layers):
        out.append(int(mx - li * steps) + WINDOW)
    return out


def keep_T(cx: Ctx, method: str, frac: float):
    """per (層, KV head) token 的保留遮罩 [L, H, ctx_len]（bool）。"""
    import torch
    km, n = cx.km, cx.ctx_len
    L, H = km.L, km.H
    keep = torch.zeros(L, H, n, dtype=torch.bool, device=km.device)
    nk = n_kept(n, frac)
    if method in POSITION:
        npr = n - nk
        if method == "STREAM":
            keep[:] = True
            keep[..., N_SINK:N_SINK + npr] = False
        elif method == "NEW":
            keep[..., :nk] = True
        else:
            import zlib
            g = torch.Generator(device="cpu").manual_seed(zlib.crc32(f"{cx.seed}|T|{method}".encode()))
            r = torch.rand(L, H, n, generator=g).to(km.device)
            keep.scatter_(-1, r.topk(nk, dim=-1).indices, True)
        return keep
    sc = scores(cx, method).clone()
    if method == "PYR":
        for li, b in enumerate(pyr_budget(n, frac, L)):
            keep[li].scatter_(-1, sc[li].topk(min(b, n), dim=-1).indices, True)
        return keep
    if method == "ADA":
        n_safe = max(1, int(nk * ADA_ALPHA)) if nk > 0 else 0
        for li in range(L):
            s = sc[li]
            s.scatter_(-1, s.topk(n_safe, dim=-1).indices, torch.finfo(s.dtype).max)
            pr = H * (n - nk)
            idx = (-s.reshape(-1)).topk(pr).indices
            k = torch.ones(H * n, dtype=torch.bool, device=km.device)
            k[idx] = False
            keep[li] = k.view(H, n)
        return keep
    keep.scatter_(-1, sc.topk(nk, dim=-1).indices, True)
    return keep


def unit_order(cx: Ctx, gran: str, method: str) -> list:
    """P／B：單位 1..n−1 的「先丟」順序（單位 0 一律保留）。分數類：每個（層, head）上單位內 token 分數平均的百分位排名，
    再對所有（層, head）平均（14 §8.3）。"""
    import torch
    u = UNIT[gran]
    nu = cx.ctx_len // u
    cand = list(range(1, nu))
    if method == "STREAM":
        return cand
    if method == "NEW":
        return cand[::-1]
    if method.startswith("RND") or method == "ORC":
        r = random.Random(f"{cx.seed}|{gran}|{'RND0' if method == 'ORC' else method}").sample(cand, len(cand))
        if method == "ORC":
            nd = {t // u for b in cx.needle for t in range(b * cx.chunk, (b + 1) * cx.chunk)}
            r = [i for i in r if i not in nd] + [i for i in r if i in nd]
        return r
    sc = scores(cx, method)                                                       # [L, H, n]
    L, H, n = sc.shape
    um = sc[..., :nu * u].reshape(L, H, nu, u).mean(-1)                          # [L, H, nu]
    rk = um.argsort(-1).argsort(-1).float() / max(1, nu - 1)                     # 百分位排名
    agg = rk.mean((0, 1)).tolist()
    return sorted(cand, key=lambda i: (agg[i], i))


def keep_mask(cx: Ctx, gran: str, method: str, frac: float):
    """回傳 (key_keep 張量（長度 km.max_len，context 之後全 True）, 實際留下的比例, 答案所在 block 的 token 留下的比例)。"""
    import torch
    km, n = cx.km, cx.ctx_len
    if method == "NONE":
        kk = torch.ones(km.max_len, dtype=torch.bool, device=km.device) if gran != "T" else \
            torch.ones(km.L, km.H, km.max_len, dtype=torch.bool, device=km.device)
        return kk, 1.0, 1.0
    nd = [t for b in cx.needle for t in range(b * cx.chunk, (b + 1) * cx.chunk)]
    if gran == "T":
        k = keep_T(cx, method, frac)
        kk = torch.ones(km.L, km.H, km.max_len, dtype=torch.bool, device=km.device)
        kk[..., :n] = k
        kept = float(k.float().mean())
        nk = float(k[..., nd].float().mean()) if nd else ""
        return kk, kept, nk
    u = UNIT[gran]
    nu = n // u
    drop = unit_order(cx, gran, method)[:int(round(frac * (nu - 1)))]
    k = torch.ones(n, dtype=torch.bool, device=km.device)
    for i in drop:
        k[i * u:(i + 1) * u] = False
    kk = torch.ones(km.max_len, dtype=torch.bool, device=km.device)
    kk[:n] = k
    return kk, float(k.float().mean()), (float(k[nd].float().mean()) if nd else "")


def parse_ds(name: str):
    """'DS-T-SNAP-25' → ('T', 'SNAP', 0.25)；'DS-B-NONE-0'＝MASK0、'DS-T-NONE-0'＝MASK0H。"""
    _, g, m, pct = name.split("-")
    assert g in UNIT and (m == "NONE" or m in METHODS[g]), name
    return g, m, int(pct) / 100


# ============================================================================ DS1 海選
SCREEN_FIELDS = ["run_id", "ts", "model", "item_id", "task", "band", "seed", "needle_blocks", "gran", "method", "frac",
                 "kept_frac", "needle_kept", "kl", "gold_lp", "gold_lp_e0", "top1", "first_ok", "first_ok_e0",
                 "t_score_s", "own_gpu_intruders"]


def cmd_screen(a):
    import torch
    import m10_sweep as SW
    from m10_quality import Answerer, make_items, _watch
    ans = Answerer(a.model)
    a.lb_n = 0
    items = make_items(a, ans.tok)
    si, sk = (int(x) for x in a.shard.split("/"))
    done = set()
    if os.path.exists(a.out):
        with (gzip.open(a.out, "rt") if a.out.endswith(".gz") else open(a.out)) as fin:
            done = {r["item_id"] for r in csv.DictReader(fin)}
    fo = gzip.open(a.out, "at", newline="") if a.out.endswith(".gz") else open(a.out, "a", newline="")
    w = csv.DictWriter(fo, fieldnames=SCREEN_FIELDS)
    if not done:
        w.writeheader()
    watch = _watch(a.out)
    rid = os.environ.get("RUN_ID", "adhoc")
    grans = a.grans
    fracs = a.fracs or FRACS
    t_all, n_items = time.time(), 0
    for kk_, mk in enumerate(items):
        if kk_ % sk != si:
            continue
        it = mk()
        if it["item_id"] in done:
            continue
        t0 = time.time()
        km = ans.km
        km.key_keep = None
        st = SW.Item(ans, it)
        tr = it["turns"][0]
        base = dict(model=a.model, item_id=it["item_id"], task=it["task"], band=tr["band"], seed=it["seed"],
                    needle_blocks=json.dumps(st.needle))
        lp0 = st.score()
        m0 = SW.Item.metrics(lp0, lp0, st.gold_ids)
        rows = [dict(base, gran="", method="E0", frac=0.0, kept_frac=1.0, needle_kept=1.0, gold_lp_e0=m0["gold_lp"],
                     first_ok_e0=m0["first_ok"], t_score_s="", **m0)]
        cx = Ctx(km, st.ids, st.ctx_len, st.bf16, ans.CHUNK, it["seed"], st.needle, tr["query_ids"])
        tsc = {}
        for meth in (m for m in SCORED if any(m in METHODS[g] or (m == "SNAP" and g == "T") for g in grans)):
            t1 = time.time()
            scores(cx, meth)
            torch.cuda.synchronize()
            tsc[meth] = round(time.time() - t1, 3)
        n_tf = 1
        cells = [("B", "NONE", 0.0), ("T", "NONE", 0.0)] + [(g, m, f) for g in grans for m in METHODS[g] for f in fracs]
        for g, meth, fr in cells:
            kk, kept, ndk = keep_mask(cx, g, meth, fr)
            km.key_keep = kk
            mm = SW.Item.metrics(st.score(), lp0, st.gold_ids)
            km.key_keep = None
            n_tf += 1
            rows.append(dict(base, gran=g, method=("MASK0" if g == "B" else "MASK0H") if meth == "NONE" else meth, frac=fr,
                             kept_frac=round(kept, 5), needle_kept=round(ndk, 5) if ndk != "" else "",
                             gold_lp_e0=m0["gold_lp"], first_ok_e0=m0["first_ok"],
                             t_score_s=tsc.get("SNAP" if meth in ("PYR", "ADA") else meth, ""), **mm))
        intr = "NOT_CHECKED" if watch is None else len(watch.intruders)
        for r in rows:
            r.update(run_id=rid, ts=ts(), own_gpu_intruders=intr)
            w.writerow(r)
        fo.flush()
        n_items += 1
        dt = time.time() - t0
        print(f"[{kk_ + 1}] {it['item_id']} rows={len(rows)} tf={n_tf} score={sum(tsc.values()):.1f}s {dt:.1f}s "
              f"avg {(time.time() - t_all) / n_items:.1f}s/題 peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        del st, cx
        torch.cuda.empty_cache()
    fo.close()
    if watch is not None:
        watch.__exit__(None, None, None)


# ============================================================================ 報告（CPU）
def _boot(diffs_by_item: dict, n_boot=2000, seed=0):
    import numpy as np
    items = sorted(diffs_by_item)
    if not items:
        return None, None, None
    v = np.array([statistics.mean(diffs_by_item[i]) for i in items])
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _read(paths):
    rows = []
    for p in paths:
        with (gzip.open(p, "rt") if p.endswith(".gz") else open(p)) as f:
            rows += [r for r in csv.DictReader(f) if r["run_id"] != "run_id"]
    return rows


def screen_table(rows):
    """{(model, gran, method, frac): {d_gold_lp, d_first_ok, kl_med, vs_rnd_gold_lp (mean, lo, hi), kept, needle_kept, n}}"""
    from collections import defaultdict
    by = defaultdict(dict)                                                     # (model, gran, frac) → method → item → row
    for r in rows:
        if r["method"] == "E0":
            continue
        by[(r["model"], r["gran"], float(r["frac"]))].setdefault(r["method"], {})[r["item_id"]] = r
    out = {}
    for (m, g, fr), meths in by.items():
        rnd = [meths[k] for k in ("RND0", "RND1", "RND2") if k in meths]
        for meth, items in meths.items():
            dg = {i: [float(r["gold_lp"]) - float(r["gold_lp_e0"])] for i, r in items.items()}
            df = {i: [int(r["first_ok"]) - int(r["first_ok_e0"])] for i, r in items.items()}
            vs = {}
            for i, r in items.items():
                rr = [x[i] for x in rnd if i in x]
                if rr and not meth.startswith("RND"):
                    vs[i] = [float(r["gold_lp"]) - statistics.mean(float(x["gold_lp"]) for x in rr)]
            out[(m, g, meth, fr)] = dict(n=len(items), d_gold_lp=_boot(dg), d_first_ok=_boot(df),
                                         kl_med=statistics.median(float(r["kl"]) for r in items.values()),
                                         kept=statistics.mean(float(r["kept_frac"]) for r in items.values()),
                                         needle_kept=statistics.mean(float(r["needle_kept"]) for r in items.values()
                                                                      if r["needle_kept"] != "") if any(
                                             r["needle_kept"] != "" for r in items.values()) else None,
                                         vs_rnd_gold_lp=_boot(vs) if vs else None)
    return out


def cmd_screen_report(a):
    import glob
    rows = _read(sorted(glob.glob(a.raw)))
    tab = screen_table(rows)
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump({"|".join(map(str, k)): v for k, v in tab.items()}, open(os.path.join(a.out_dir, "ds1_table.json"), "w"), indent=1)
    rids = sorted({r["run_id"] for r in rows})
    L = ["# DS1 海選：直接丟掉 token（14 §8.4；teacher forcing、第 1 輪；只報告、沒有多重比較校正）", "",
         f"run：{'、'.join(rids)}；{len({(r['model'], r['item_id']) for r in rows})} 題", "",
         "gold_lp＝標準答案 log 機率對 E0 的差（越接近 0 越好）；first_ok 是百分點；對 RND＝gold_lp 對 RND0–2 平均的配對差（正＝比隨機好）。", ""]
    for m in sorted({k[0] for k in tab}):
        for g in ("T", "P", "B"):
            ks = [k for k in tab if k[0] == m and k[1] == g]
            if not ks:
                continue
            L += [f"## {m}・{g}（{'per-head token' if g == 'T' else ('16-token page' if g == 'P' else '512-token block')}）", "",
                  "| 挑法 | " + " | ".join(f"{int(f * 100)}%：gold_lp／first_ok／對 RND" for f in FRACS) + " |",
                  "|:--|" + "--:|" * len(FRACS)]
            for meth in ["MASK0" if g == "B" else "MASK0H"] + METHODS[g]:
                cells = []
                for f in FRACS:
                    v = tab.get((m, g, meth, f)) or tab.get((m, g, meth, 0.0)) if meth.startswith("MASK") else tab.get((m, g, meth, f))
                    if not v:
                        cells.append("—")
                        continue
                    vr = v["vs_rnd_gold_lp"]
                    cells.append(f"{v['d_gold_lp'][0]:+.2f}／{100 * v['d_first_ok'][0]:+.1f}／"
                                 + (f"{vr[0]:+.2f} [{vr[1]:+.2f}, {vr[2]:+.2f}]" if vr else "—"))
                    if meth.startswith("MASK"):
                        cells += ["（同上）"] * (len(FRACS) - 1)
                        break
                L.append(f"| {meth} | " + " | ".join(cells) + " |")
            L.append("")
    open(os.path.join(a.out_dir, "tables_ds1.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:40]))


def cmd_pick(a):
    """14 §8.5：每個模型、粒度 T、B：25%、50% 兩個佔比的 gold_lp 對 E0 平均差（兩個佔比平均）排名，寫入時挑法取前 2，
    加 STREAM、RND0、QRY、ORC（B 才有）；兩個佔比；加 E0、MASK0、MASK0H。寫 ds2_plan.json 和 jobs json。"""
    tab = {tuple(k.split("|")[:3]) + (float(k.split("|")[3]),): v
           for k, v in json.load(open(os.path.join(a.out_dir, "ds1_table.json"))).items()}
    plan, jobs = {}, []
    for m in a.models:
        cfgs = ["E0", "DS-B-NONE-0", "DS-T-NONE-0"]
        rank = {}
        for g in ("T", "B"):
            sc = {}
            for meth in WRITE_TIME:
                if meth not in METHODS[g]:
                    continue
                v = [tab.get((m, g, meth, f)) for f in (0.25, 0.50)]
                if all(v):
                    sc[meth] = statistics.mean(x["d_gold_lp"][0] for x in v)
            top = sorted(sc, key=lambda k: -sc[k])[:2]
            rank[g] = dict(scores=sc, top2=top)
            for meth in top + ["STREAM", "RND0", "QRY"] + (["ORC"] if g == "B" else []):
                for pct in (25, 50):
                    cfgs.append(f"DS-{g}-{meth}-{pct}")
        plan[m] = dict(configs=cfgs, rank=rank)
        base = (f"python {REPO}/code/m10_quality.py run --model {m} --params-file {a.params_file} --configs {' '.join(cfgs)} "
                f"--modes A --seed0 70000")
        for s in range(a.shards):
            jobs.append(dict(name=f"ds2-{m}-T24-s{s}", need_gb=a.need_gb,
                             cmd=f"m7run m10-ds2-{m}-T24-s{s} {base} --tasks T2 T3 T4 --n 45 --shard {s}/{a.shards} "
                                 f"--out {a.out_dir}/quality_{m}_T24_s{s}.csv"))
        jobs.append(dict(name=f"ds2-{m}-T5", need_gb=a.need_gb,
                         cmd=f"m7run m10-ds2-{m}-T5 {base} --tasks T5 --n 15 --out {a.out_dir}/quality_{m}_T5.csv"))
    json.dump(dict(rule="14 §8.5（10/19 01:5xZ 寫死）", models=plan), open(os.path.join(a.out_dir, "ds2_plan.json"), "w"),
              indent=1, ensure_ascii=False)
    json.dump(jobs, open(a.jobs_out, "w"), indent=1)
    print(json.dumps(plan, indent=1, ensure_ascii=False))


def cmd_confirm_report(a):
    """14 §8.5 判準 1–4（只報告）。"""
    import glob
    import m10_analyze as A
    out = {}
    L = ["# DS2 複賽：直接丟掉 token（14 §8.5；生成、兩輪、讀法 A；只報告）", ""]
    for m in a.models:
        rows = A.load_rows(sorted(glob.glob(os.path.join(a.out_dir, f"quality_{m}_*.csv"))))
        if not rows:
            continue
        ref = A.cond_rows(rows, "E0", "A")
        cfgs = sorted({r["config"] for r in rows} - {"E0"})
        res = {}
        for c in cfgs:
            x = A.cond_rows(rows, c, "A")
            e = A.eps_check(ref, x)
            t1 = A.paired(ref, x, turn=1)[0]
            t2 = A.paired(ref, x, turn=2)[0]
            d1 = sum(sum(v) for v in t1.values()) / max(1, sum(len(v) for v in t1.values()))
            d2 = sum(sum(v) for v in t2.values()) / max(1, sum(len(v) for v in t2.values())) if t2 else None
            res[c] = dict(eps=e, turn1=d1, turn2=d2, turn2_minus_turn1=(d2 - d1) if d2 is not None else None)
        # 對 RND0：同一個（粒度, 佔比）的其他配置，Holm（α＝0.025 單邊）
        for c in cfgs:
            if not c.startswith("DS-") or "-NONE-" in c or "-RND0-" in c or "-ORC-" in c:
                continue
            g, meth, f = parse_ds(c)
            rc = f"DS-{g}-RND0-{int(round(f * 100))}"
            if rc not in res:
                continue
            d, n, *_ = A.paired(A.cond_rows(rows, rc, "A"), A.cond_rows(rows, c, "A"))
            pt, lo, hi = A.cluster_boot(d)
            import numpy as np
            items = sorted(d)
            s = np.array([sum(d[i]) for i in items]); nn = np.array([len(d[i]) for i in items])
            rng = np.random.default_rng(0)
            idx = rng.integers(0, len(items), size=(10000, len(items)))
            p1 = float(((s[idx].sum(1) / nn[idx].sum(1)) <= 0).mean())
            res[c]["vs_rnd0"] = dict(diff=pt, ci95=[lo, hi], p_one_sided=p1)
        fam = {}
        for c, v in res.items():
            if "vs_rnd0" in v:
                g, meth, f = parse_ds(c)
                fam.setdefault((g, f), []).append(c)
        for key, cs in fam.items():
            cs = sorted(cs, key=lambda c: res[c]["vs_rnd0"]["p_one_sided"])
            for j, c in enumerate(cs):
                thr = 0.025 / (len(cs) - j)
                rej = res[c]["vs_rnd0"]["p_one_sided"] <= thr and all(
                    res[cs[i]]["vs_rnd0"]["p_one_sided"] <= 0.025 / (len(cs) - i) for i in range(j))
                res[c]["vs_rnd0"]["holm"] = dict(threshold=thr, reject=bool(rej))
        out[m] = res
        L += [f"## {m}（E0 列數 {len(ref)}）", "",
              "| 配置 | 對 E0（點，95% CI） | ε | 第 1 輪 | 第 2 輪 | 第 2 − 第 1 | 對 RND0（點，CI，Holm） |",
              "|:--|:--|:--|--:|--:|--:|:--|"]
        for c in cfgs:
            v = res[c]
            e = v["eps"]
            vr = v.get("vs_rnd0")
            t2 = "" if v["turn2"] is None else f"{100 * v['turn2']:+.1f}"
            dd = "" if v["turn2_minus_turn1"] is None else f"{100 * v['turn2_minus_turn1']:+.1f}"
            rv = (f"{100 * vr['diff']:+.1f} [{100 * vr['ci95'][0]:+.1f}, {100 * vr['ci95'][1]:+.1f}] "
                  f"{'Holm 過' if vr['holm']['reject'] else '—'}") if vr else "—"
            L.append(f"| {c} | {100 * e['diff']:+.1f} [{100 * e['ci95'][0]:+.1f}, {100 * e['ci95'][1]:+.1f}] | "
                     f"{'過' if e['eps_pass'] else '沒過'} | {100 * v['turn1']:+.1f} | {t2} | {dd} | {rv} |")
        L.append("")
    json.dump(out, open(os.path.join(a.out_dir, "ds2_report.json"), "w"), indent=1, default=str)
    open(os.path.join(a.out_dir, "tables_ds2.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def cmd_speed_report(a):
    """D3、M1、M2：每個 restores.csv 的（層, GPU, 配置, 模式）還原時間中位數；只用 gpu_guard CLEAN、沒有污染列的 run。"""
    import glob
    sys.path.insert(0, HERE)
    import m10_restore2 as R2
    L = ["# 14 號的速度（D3、M1、M2；只報告）", "", "中位數（s）；只用 gpu_guard 全 CLEAN 的 run（CLAUDE.md §3）。", ""]
    out = {}
    for d in sorted(glob.glob(os.path.join(a.root, "*", "*"))):
        if not os.path.exists(os.path.join(d, "restores.csv")):
            continue
        rows = R2.latest_rows(d)
        ok = R2.usable(d, rows)
        name = os.path.relpath(d, a.root)
        keys = sorted({(r["length"], r["bw_name"], r["gpu_state"], r["config"], r["mode"]) for r in rows})
        tab = {}
        for k in keys:
            v = [float(r["t_restore"]) for r in rows if (r["length"], r["bw_name"], r["gpu_state"], r["config"], r["mode"]) == k]
            b = sorted({r["b"] for r in rows if (r["length"], r["bw_name"], r["gpu_state"], r["config"], r["mode"]) == k})
            tab["|".join(k)] = dict(median=statistics.median(v), n=len(v), b=b)
        out[name] = dict(usable=ok, run_id=rows[0]["run_id"] if rows else "", cells=tab)
        L += [f"## {name}（{'可用' if ok else '**不能用**'}；{out[name]['run_id']}）", "",
              "| 長度 | 層 | GPU | 配置 | 模式 | 中位數 | n | 會合點 b |", "|:--|:--|:--|:--|:--|--:|--:|:--|"]
        for k, v in tab.items():
            L.append("| " + " | ".join(k.split("|")) + f" | {v['median']:.4f} | {v['n']} | {' '.join(v['b'][:6])} |")
        L.append("")
    json.dump(out, open(os.path.join(a.root, "speed_14.json"), "w"), indent=1)
    open(os.path.join(a.root, "tables_speed_14.md"), "w").write("\n".join(L) + "\n")
    print(f"寫了 {len(out)} 個目錄的表")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("screen")
    s.add_argument("--model", choices=["llama", "qwen"], required=True)
    s.add_argument("--params-file", default=os.path.join(REPO, "results", "m10_pilot_mi300x", "e0_tune", "CANDIDATE_PARAMS.json"))
    s.add_argument("--tasks", nargs="+", default=["T2", "T3", "T4", "T5"])
    s.add_argument("--n", type=int, default=24)
    s.add_argument("--seed0", type=int, default=60000)
    s.add_argument("--grans", nargs="+", default=["T", "P", "B"])
    s.add_argument("--fracs", nargs="+", type=float)
    s.add_argument("--shard", default="0/1")
    s.add_argument("--out", required=True)
    r = sp.add_parser("screen-report")
    r.add_argument("--raw", required=True, help="DS1 逐列 CSV 的萬用字元")
    r.add_argument("--out-dir", required=True)
    p = sp.add_parser("pick")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--models", nargs="+", default=["llama", "qwen"])
    p.add_argument("--params-file", default=os.path.join(REPO, "results", "m10_pilot_mi300x", "e0_tune", "CANDIDATE_PARAMS.json"))
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--need-gb", type=int, default=40)
    p.add_argument("--jobs-out", required=True)
    c = sp.add_parser("confirm-report")
    c.add_argument("--out-dir", required=True)
    c.add_argument("--models", nargs="+", default=["llama", "qwen"])
    v = sp.add_parser("speed-report")
    v.add_argument("--root", required=True)
    a = ap.parse_args()
    {"screen": cmd_screen, "screen-report": cmd_screen_report, "pick": cmd_pick, "confirm-report": cmd_confirm_report,
     "speed-report": cmd_speed_report}[a.cmd](a)


if __name__ == "__main__":
    main()
