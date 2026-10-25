"""a0_gpu.py — A0 的 GPU 端（由 a0.py 的 calib／tf／gen／gpu-selftest 呼叫）。登記：docs/stage3_design_20261024/runs/A0_progressive_restore.md。

結構（每題）：
  ItemState   fa3 prefill → 切 sdpa →（第 2 輪題）用完整 KV 接上第 1 輪問題＋歷史＝尾段 → 區內原值存成 block 為單位的 store
  Store       [nreg, L, 2, H, 512, D]（block 為主的排法：每塊、每層都是連續的一片）；放 GPU 或 host（14B 256K）；INT4 可以現算
  Composer    記住 KV 緩衝裡每個（層, 區內 block）現在放的是哪一份（原值／INT4／粗底），換配置時只複製不同的地方
  打分        一律 ctx_mass：每層、每個 q head、每個問題 token 在 [0, ctx_end) 的 key 上 softmax(q·k/√D)，加總落在區內各塊的機率
  forward     forward_hooked＝KVModel.forward_span（sdpa、沒有 key_keep）的逐行拷貝，多一個「寫完這層的 K、V、算 attention 之前」的 hook
              （ONE-PASS 在 hook 裡挑這層的塊、換成精修；打分的 forward 在 hook 裡算 ctx_mass）。hook＝None 時和 forward_span 逐位元相同（gpu-selftest）。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import a0 as A  # noqa: E402
import m16_xstack as XS  # noqa: E402

CHUNK = A.CHUNK
ORIG, INT4, BASE = 0, 1, 2
NAMES = {ORIG: "orig", INT4: "int4", BASE: "base", -1: "unknown"}


def sync():
    torch.cuda.synchronize()


# ============================================================================ 模型
def calib_path(model: str) -> str:
    return os.path.join(A.LOCAL, "xstack", f"calib_xl_{model}.pt")


def params_file_for(model: str, src: str) -> str:
    """qwen14 沿用 qwen 的題目參數（登記 §1.1）：寫一份補上 qwen14 鍵的副本到本地（內容＝原檔＋這一個鍵）。"""
    if model != "qwen14":
        return src
    d = json.load(open(src))
    d["params"]["qwen14"] = d["params"][A.PARAMS_OF["qwen14"]]
    os.makedirs(os.path.join(A.LOCAL, "params"), exist_ok=True)
    p = os.path.join(A.LOCAL, "params", "CANDIDATE_PARAMS_with_qwen14.json")
    json.dump(d, open(p, "w"), indent=1)
    return p


class Session:
    def __init__(self, model: str, nb: int, store: str = "auto", seg_blocks: int = 16, calib_xl: str = ""):
        from m10_quality import Answerer
        import m10_tasks as T
        self.T = T
        self.model, self.nb, self.nreg = model, nb, nb - 1
        self.info = A.info_of(model)
        self.ans = Answerer(model, extra=nb * CHUNK + 4096 - T.N_BLOCKS * T.BLOCK if nb * CHUNK > T.N_BLOCKS * T.BLOCK else 4096)
        self.km = km = self.ans.km
        XS.check_info(km, self.info)
        self.P = json.load(open(params_file_for(model, A.PARAMS)))["params"][A.PARAMS_OF[model]]
        self.store_mode = store if store != "auto" else ("host" if model == "qwen14" and nb > 256 else "gpu")
        self.seg = seg_blocks
        self.cx = XS.TCtx(km, nb * CHUNK)
        self.calib_xl = calib_xl or calib_path(model)
        self.cx.xlcal = None
        self.region_bf16 = A.region_bf16(self.info, self.nreg)

    def load_calib(self, eval_keys: set):
        d = torch.load(self.calib_xl, map_location=self.km.device)
        meta = d["meta"]
        if meta["model"] != self.model or (meta["L"], meta["H"], meta["D"], meta["W"]) != (self.km.L, self.km.H, self.km.D, 4):
            raise SystemExit(f"xKV 校準檔 {self.calib_xl} 是 {meta['model']} L{meta['L']}/H{meta['H']}/D{meta['D']}/W{meta['W']}，對不上")
        ov = {tuple(x) for x in meta["item_keys"]} & eval_keys
        if ov:
            raise SystemExit(f"xKV 校準題和評測題重疊：{sorted(ov)[:5]}")
        self.cx.xlcal = d
        self.calib_rid = meta["run_id"]
        return meta


# ============================================================================ store
class Store:
    """區內 nreg 塊，每塊 [L, 2, H, 512, D] BF16。mode：gpu｜host（pinned；開不起來就用一般記憶體，照實印出）。"""

    def __init__(self, sess, mode: str, name: str):
        km = sess.km
        shape = (sess.nreg, km.L, 2, km.H, CHUNK, km.D)
        self.name, self.mode, self.dev = name, mode, km.device
        if mode == "gpu":
            self.t = torch.empty(shape, dtype=torch.bfloat16, device=km.device)
        else:
            try:
                self.t = torch.empty(shape, dtype=torch.bfloat16, pin_memory=True)
                self.pinned = True
            except RuntimeError as ex:
                print(f"[store {name}] pin_memory 失敗（{ex!r}）→ 用一般的 host 記憶體（比較慢）", flush=True)
                self.t = torch.empty(shape, dtype=torch.bfloat16)
                self.pinned = False

    def put(self, j: int, x):
        self.t[j].copy_(x, non_blocking=self.mode == "host")

    def block(self, j: int):
        return self.t[j] if self.mode == "gpu" else self.t[j].to(self.dev, non_blocking=True)

    def layer(self, j: int, l: int):
        return self.t[j, l] if self.mode == "gpu" else self.t[j, l].to(self.dev, non_blocking=True)


class Int4Fly:
    """INT4 精修（int4_kivi）：從原值的 store 現算（14B 256K 放不下第三份）。和 XS.t_full_q(…, "int4_kivi", 512) 的一塊相同。"""

    def __init__(self, orig: Store):
        import m10_codec_x as X10
        self.X10, self.orig, self.mode, self._c = X10, orig, "fly", {}

    def block(self, j: int):
        if j not in self._c:
            if len(self._c) > 4:
                self._c.clear()
            x = self.orig.block(j).contiguous()
            self._c[j] = self.X10.decode(self.X10.encode(x, "int4_kivi"))
        return self._c[j]

    def layer(self, j: int, l: int):
        return self.block(j)[l]


def int4_block(x):
    import m10_codec_x as X10
    return X10.decode(X10.encode(x.contiguous(), "int4_kivi"))


# ============================================================================ composer
class Composer:
    def __init__(self, st):
        import numpy as np
        self.np, self.st, self.km = np, st, st.sess.km
        self.state = np.zeros((self.km.L, st.sess.nreg), dtype=np.int8)       # prefill 完：全部是原值
        self.stores = {ORIG: st.orig, INT4: st.int4, BASE: None}
        self.n_block_copies = self.n_layer_copies = 0

    def set_base(self, base_store):
        self.stores[BASE] = base_store
        self.state[self.state == BASE] = -1          # 換了粗底 → 原本放粗底的位置不算數

    def unknown(self):
        self.state[:] = -1

    def apply(self, want):
        """want：[L, nreg] 的來源碼。只複製和現在不同的（層, 塊）。"""
        np, km = self.np, self.km
        diff = want != self.state
        if not diff.any():
            return
        for j in np.nonzero(diff.any(0))[0].tolist():
            col, d = want[:, j], diff[:, j]
            s, e = (j + 1) * CHUNK, (j + 2) * CHUNK
            if d.all() and (col == col[0]).all():
                km.kv[:, :, :, s:e].copy_(self.stores[int(col[0])].block(j))
                self.n_block_copies += 1
            else:
                for l in np.nonzero(d)[0].tolist():
                    km.kv[l, :, :, s:e].copy_(self.stores[int(col[l])].layer(j, l))
                    self.n_layer_copies += 1
        self.state[:] = want

    def put_layer(self, l: int, blocks, code: int):
        """ONE-PASS 在 forward 裡用：第 l 層的這些塊換成 code。"""
        km = self.km
        for j in blocks:
            if self.state[l, j] != code:
                s = (j + 1) * CHUNK
                km.kv[l, :, :, s:s + CHUNK].copy_(self.stores[code].layer(j, l))
                self.state[l, j] = code
                self.n_layer_copies += 1


# ============================================================================ forward
@torch.inference_mode()
def forward_hooked(km, ids, start: int, hook=None, want_logits: bool = False, n_logits: int = 1):
    """KVModel.forward_span（sdpa、key_keep＝None）的逐行拷貝；hook(li, q) 在第 li 層寫完 K、V、算 attention 之前呼叫。"""
    from torch.nn.attention.bias import causal_lower_right
    assert km.key_keep is None and km.attn == "sdpa"
    n = ids.shape[-1]
    end = start + n
    assert end <= km.max_len
    m = km.hf.model
    x = m.embed_tokens(ids.view(1, n))
    pos = torch.arange(start, end, device=km.device).view(1, n)
    cos, sin = m.rotary_emb(x, pos)
    key = (n, end)
    mask = km._masks.get(key)
    if mask is None:
        mask = causal_lower_right(n, end)
        km._masks[key] = mask
    for li, layer in enumerate(m.layers):
        a = layer.self_attn
        h = layer.input_layernorm(x)
        q = a.q_proj(h).view(1, n, km.Hq, km.D)
        k = a.k_proj(h).view(1, n, km.H, km.D)
        if hasattr(a, "q_norm"):
            q, k = a.q_norm(q), a.k_norm(k)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        v = a.v_proj(h).view(1, n, km.H, km.D).transpose(1, 2)
        q, k = km.apply_rope(q, k, cos, sin)
        km.kv[li, 0, :, start:end].copy_(k[0])
        km.kv[li, 1, :, start:end].copy_(v[0])
        if hook is not None:
            hook(li, q[0])
        K = km.kv[li, 0, :, :end].unsqueeze(0)
        V = km.kv[li, 1, :, :end].unsqueeze(0)
        o = F.scaled_dot_product_attention(q, K, V, attn_mask=mask, enable_gqa=True)
        o = o.transpose(1, 2).reshape(1, n, km.Hq * km.D)
        x = x + a.o_proj(o)
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    if want_logits:
        if n_logits > 1:
            return km.hf.lm_head(m.norm(x[:, -n_logits:]))[0]
        return km.hf.lm_head(m.norm(x[:, -1:]))[0, -1]
    return None


def ctx_mass(km, q, K, nq: int, nreg: int, rows: int = 512):
    """q：[Hq, n, D]（取前 nq 個＝問題 token）；K：[H, ctx_end, D]（block 0＋區內＋尾段）。回傳 [nreg] float64：
    Σ_{h,t} softmax_{k<ctx_end}(q·k/√D) 落在區內第 j 塊（token 512 起）的機率。"""
    H, Hq, D = km.H, km.Hq, km.D
    rep = Hq // H
    out = torch.zeros(nreg, dtype=torch.float64, device=km.device)
    sc = 1.0 / math.sqrt(D)
    for hk in range(H):
        Kh = K[hk].float()
        qs = q[hk * rep:(hk + 1) * rep, :nq].float().reshape(-1, D)
        for r0 in range(0, qs.shape[0], rows):
            p = (qs[r0:r0 + rows] @ Kh.T * sc).softmax(-1)
            out += p[:, CHUNK:CHUNK * (nreg + 1)].sum(0).view(nreg, CHUNK).sum(-1).double()
        del Kh
    return out


def topm(score, m: int) -> list:
    """分數由大到小取前 m 個（平手取編號小的）；score：長度 nreg 的 list／tensor。"""
    s = score.tolist() if hasattr(score, "tolist") else list(score)
    return sorted(range(len(s)), key=lambda j: (-s[j], j))[:m]


# ============================================================================ 一題
class ItemState:
    def __init__(self, sess: Session, it: dict, turn: int):
        km, T = sess.km, sess.T
        self.sess, self.it, self.turn = sess, it, turn
        n = len(it["context_ids"])
        assert n == sess.nb * CHUNK, (n, sess.nb)
        self.ctx_len = n
        km.key_keep = None
        ids = torch.tensor(it["context_ids"], device=km.device)
        sync()
        t0 = time.time()
        km.set_attn("fa3")
        try:
            km.prefill_chunked(ids)
        finally:
            km.set_attn("sdpa")
        sync()
        self.t_prefill = time.time() - t0
        pos = n
        if turn == 1:                                        # 暫停前：第 1 輪問題＋第 1 輪標準答案的歷史，用完整 KV 接上（＝尾段）
            q0 = it["turns"][0]["query_ids"]
            h1 = it["turns"][1]["history_ids"]
            sess.ans.feed(q0 + h1, pos)
            pos += len(q0) + len(h1)
        self.ctx_end = pos
        tr = it["turns"][turn]
        self.tr = tr
        self.q_ids = list(tr["query_ids"])
        self.gold_ids = T._enc(sess.ans.tok)(tr["gold_text"])
        self.needle = [b - 1 for b in tr["needle_blocks"] if 1 <= b < sess.nb]
        self.orig = Store(sess, sess.store_mode, "orig")
        for j in range(sess.nreg):
            self.orig.put(j, km.chunk_view(j + 1))
        if sess.store_mode == "gpu":
            self.int4 = Store(sess, "gpu", "int4")
            for j in range(sess.nreg):
                self.int4.put(j, int4_block(km.chunk_view(j + 1)))
        else:
            self.int4 = Int4Fly(self.orig)
        sync()
        self.comp = Composer(self)
        self._write_order = None
        self._minmax = None

    # ---- teacher forcing ----------------------------------------------------
    def tf_seq(self):
        return self.q_ids + self.gold_ids[:-1]

    @torch.inference_mode()
    def score(self, hook=None):
        """問題＋標準答案（去掉最後一個）一次餵進去；回傳答案位置的 log 機率 [m, vocab]（m10_sweep.Item.score 的同一件事）。
        hook＝None：KVModel.forward_span（有 key_keep 就走遮罩路徑；超過 512 個 token 時前面先接上）；有 hook：forward_hooked 一整段。"""
        km = self.sess.km
        seq = self.tf_seq()
        m = len(self.gold_ids)
        ids = torch.tensor(seq, device=km.device)
        if hook is not None:
            lg = forward_hooked(km, ids, self.ctx_end, hook=hook, want_logits=True, n_logits=m)
        elif len(seq) > CHUNK:
            cut = len(seq) - CHUNK
            self.sess.ans.feed(seq[:cut], self.ctx_end)
            lg = km.forward_span(ids[cut:], self.ctx_end + cut, want_logits=True, n_logits=m)
        else:
            lg = km.forward_span(ids, self.ctx_end, want_logits=True, n_logits=m)
        lg = lg if lg.dim() == 2 else lg.unsqueeze(0)
        return lg.float().log_softmax(-1)

    @staticmethod
    def metrics(lp, lp0, gold):
        g = torch.tensor(gold, device=lp.device)
        ar = torch.arange(len(gold), device=lp.device)
        a, a0 = lp.argmax(-1), lp0.argmax(-1)
        return dict(kl=float((lp0.exp() * (lp0 - lp)).sum(-1).mean()), gold_lp=float(lp[ar, g].sum()),
                    top1=float((a == a0).float().mean()), first_ok=int(a[0] == g[0]))

    # ---- 生成 ---------------------------------------------------------------
    @torch.inference_mode()
    def generate(self, hook=None):
        from m10_quality import gen_len
        sess = self.sess
        if hook is not None:
            logits = forward_hooked(sess.km, torch.tensor(self.q_ids, device=sess.km.device), self.ctx_end, hook=hook, want_logits=True)
        else:
            logits = sess.ans.feed(self.q_ids, self.ctx_end)
        out, stop = sess.ans.generate(logits, self.ctx_end + len(self.q_ids), gen_len(self.it))
        return sess.T.score(self.tr["metric"], sess.ans.decode(out), self.tr["gold"]), stop, out

    # ---- 打分 ---------------------------------------------------------------
    @torch.inference_mode()
    def question_pass(self, want_q: bool = True):
        """KV 緩衝現在的樣子（原值或粗底）上跑一次新問題：每層的 ctx_mass [L, nreg]，以及每層的 q [L, Hq, nq, D]（FP32）。"""
        km, nreg, nq = self.sess.km, self.sess.nreg, len(self.q_ids)
        mass = torch.zeros(km.L, nreg, dtype=torch.float64, device=km.device)
        qcap = torch.empty(km.L, km.Hq, nq, km.D, dtype=torch.float32, device=km.device) if want_q else None

        def hook(li, q):
            mass[li] = ctx_mass(km, q, km.kv[li, 0, :, :self.ctx_end], nq, nreg)
            if want_q:
                qcap[li] = q[:, :nq].float()
        forward_hooked(km, torch.tensor(self.q_ids, device=km.device), self.ctx_end, hook=hook)
        return mass, qcap

    @torch.inference_mode()
    def mass_from_q(self, qcap, kfn):
        """給定每層的 q，對每層的 K（kfn(li) → [H, ctx_end, D]）算 ctx_mass。"""
        km, nreg, nq = self.sess.km, self.sess.nreg, qcap.shape[2]
        mass = torch.zeros(km.L, nreg, dtype=torch.float64, device=km.device)
        for li in range(km.L):
            mass[li] = ctx_mass(km, qcap[li], kfn(li), nq, nreg)
        return mass

    def orig_k_layer(self, li: int):
        """第 li 層、[0, ctx_end) 的原值 K（block 0、尾段直接取緩衝；區內取原值的 store）。"""
        km = self.sess.km
        reg = torch.cat([self.orig.layer(j, li)[0] for j in range(self.sess.nreg)], dim=1)       # [H, Tr, D]
        return torch.cat([km.kv[li, 0, :, :CHUNK], reg, km.kv[li, 0, :, self.ctx_len:self.ctx_end]], dim=1)

    @torch.inference_mode()
    def kscan_mass(self, qcap):
        """KSCAN-2b：區內 K 換成逐 token 2-bit（int2h_g32）解碼值；block 0、尾段是原值（反正一律讀 BF16）。全域分數 [nreg]。"""
        import m10_codec_x as X10
        km = self.sess.km

        def kfn(li):
            K = self.orig_k_layer(li)
            reg = K[:, CHUNK:self.ctx_len].unsqueeze(0)                                          # [1, H, Tr, D]
            p = X10.enc_half(reg.contiguous(), "int2h_g32", "k")
            dec = X10.dec_half(p, "int2h_g32", "k", tuple(reg.shape))[0].to(torch.bfloat16)
            return torch.cat([K[:, :CHUNK], dec, K[:, self.ctx_len:]], dim=1)
        return self.mass_from_q(qcap, kfn).sum(0)

    def minmax(self):
        """Quest 的摘要：每塊、每（層, KV head）的原值 post-RoPE K 逐維 min／max（含 block 0）→ [L, H, nb, D] ×2（FP32）。"""
        if self._minmax is None:
            km = self.sess.km
            mins = [km.kv[:, 0, :, :CHUNK].amin(dim=2).float()]
            maxs = [km.kv[:, 0, :, :CHUNK].amax(dim=2).float()]
            for j in range(self.sess.nreg):
                b = self.orig.block(j)[:, 0]
                mins.append(b.amin(dim=2).float())
                maxs.append(b.amax(dim=2).float())
            self._minmax = (torch.stack(mins, dim=2), torch.stack(maxs, dim=2))
        return self._minmax

    @torch.inference_mode()
    def quest_score(self, qcap, skip: int = 2):
        """C1a 的 QUEST_MEAN：U＝Σ_d max(q·min, q·max)/√D，每（層, q head, 問題 token）對所有塊（含 block 0）softmax → 對 token 平均
        → 略過前 skip 層 → 對（層, q head）平均 → 取區內塊。"""
        km = self.sess.km
        kmin, kmax = self.minmax()
        rep = km.Hq // km.H
        kmx = kmax.repeat_interleave(rep, dim=1)
        kmn = kmin.repeat_interleave(rep, dim=1)
        U = (qcap.clamp(min=0) @ kmx.transpose(-1, -2) + qcap.clamp(max=0) @ kmn.transpose(-1, -2)) / math.sqrt(km.D)
        P = U.softmax(-1).mean(2)                                                                 # [L, Hq, nb]
        return P[skip:].reshape(-1, P.shape[-1]).mean(0)[1:].double()

    def write_order(self):
        """寫入時分數（Llama KNORM、Qwen KEYDIFF）→ m10_drop.unit_order 的彙總（百分位排名平均）→ 保留順序（區內編號）。"""
        if self._write_order is None:
            km, nb = self.sess.km, self.sess.nb
            meth = A.WRITE_METHOD[self.sess.model]
            rk_sum = torch.zeros(nb, dtype=torch.float64, device=km.device)
            for li in range(km.L):
                K = km.kv[li, 0, :, :self.ctx_len].float()                                       # [H, n, D]（prefill 後還是原值）
                if meth == "KNORM":
                    sc = -torch.linalg.vector_norm(K, dim=-1)
                else:
                    anchor = F.normalize(K, p=2, dim=-1).mean(dim=1, keepdim=True)
                    sc = -F.cosine_similarity(K, anchor, dim=-1)
                um = sc.reshape(km.H, nb, CHUNK).mean(-1)                                         # [H, nb]
                rk_sum += (um.argsort(-1).argsort(-1).double() / max(1, nb - 1)).sum(0)
            agg = (rk_sum / (km.L * km.H)).tolist()
            drop_first = sorted(range(1, nb), key=lambda i: (agg[i], i))
            self._write_order = [i - 1 for i in drop_first[::-1]]
        return self._write_order


# ============================================================================ 粗底／均勻配置的編碼（分段，和整段一次編碼相同）
@torch.inference_mode()
def encode_spec(sess: Session, st: ItemState, spec: str, sink):
    """把區內的原值用 spec 編碼、解碼；每塊的解碼結果交給 sink(j, [L,2,H,512,D])。回傳實際位元組（逐項）並和 predict_bytes 核對。"""
    info, nreg = sess.info, sess.nreg
    stages = XS.validate(XS.parse_spec(spec), info, nreg)
    if any(s["op"] == "drop" for s in stages):
        raise XS.SpecError("A0 的均勻配置不丟塊")
    tot, shared = {}, {}
    for j0 in range(0, nreg, sess.seg):
        j1 = min(nreg, j0 + sess.seg)
        X = torch.cat([st.orig.block(j) for j in range(j0, j1)], dim=3)                         # [L, 2, H, (j1−j0)·512, D]
        pos = torch.arange((j0 + 1) * CHUNK, (j1 + 1) * CHUNK, device=sess.km.device)
        dec, idx, by, dropped = XS.t_pipeline(stages, X, pos, sess.cx, info)
        assert idx is None and not dropped
        for k, v in by.items():
            if k.startswith("shared."):
                if shared.setdefault(k, v) != v:
                    raise RuntimeError(f"{spec}：共用的 {k} 每段不同（{shared[k]} vs {v}）")
            else:
                tot[k] = tot.get(k, 0) + v
        for j in range(j0, j1):
            sink(j, dec[:, :, :, (j - j0) * CHUNK:(j - j0 + 1) * CHUNK])
        del X, dec
    pred = XS.predict_bytes(stages, info, nreg)
    got = dict(tot, **shared)
    if got != pred:
        raise RuntimeError(f"{spec}：實際位元組 {got} ≠ 預測 {pred}（規則 2：停）")
    return got


def write_uniform(sess, st, spec):
    """均勻配置直接寫進 KV 緩衝的區內。"""
    km = sess.km

    def sink(j, x):
        km.kv[:, :, :, (j + 1) * CHUNK:(j + 2) * CHUNK].copy_(x)
    by = encode_spec(sess, st, spec, sink)
    st.comp.unknown()
    return by


def build_base(sess, st, spec):
    store = Store(sess, sess.store_mode, "base")
    by = encode_spec(sess, st, spec, store.put)
    sync()
    return store, by


# ============================================================================ 挑塊與組 KV
def want_array(sess, sel, gran: str, code: int, fill: int):
    import numpy as np
    w = np.full((sess.km.L, sess.nreg), fill, dtype=np.int8)
    if gran == "global":
        w[:, sel] = code
    else:
        for l in range(sess.km.L):
            w[l, sel[l]] = code
    return w


def sel_hash(sel) -> str:
    return hashlib.sha1(json.dumps(sel).encode()).hexdigest()[:12]


def needle_cover(sel, gran: str, needle: list):
    if not needle:
        return ""
    nd = set(needle)
    if gran == "global":
        return round(len(nd & set(sel)) / len(nd), 4)
    return round(sum(len(nd & set(s)) / len(nd) for s in sel) / len(sel), 4)


def keep_mask(sess, st, sel, gran):
    """丟塊對照：沒挑中的區內塊遮掉（block 0、尾段、問題都看得到）。global → [max_len]；layer → [L, H, max_len]。"""
    km = sess.km
    if gran == "global":
        kk = torch.ones(km.max_len, dtype=torch.bool, device=km.device)
        kk[CHUNK:st.ctx_len] = False
        for j in sel:
            kk[(j + 1) * CHUNK:(j + 2) * CHUNK] = True
        return kk
    kk = torch.ones(km.L, km.H, km.max_len, dtype=torch.bool, device=km.device)
    kk[:, :, CHUNK:st.ctx_len] = False
    for l in range(km.L):
        for j in sel[l]:
            kk[l, :, (j + 1) * CHUNK:(j + 2) * CHUNK] = True
    return kk


class Selections:
    """一題、一個粗底下各挑法的分數（同一題的 ORACLE 分數跨粗底共用）。"""

    def __init__(self, sess, st):
        self.sess, self.st = sess, st
        self.mass = {}        # 名稱 → [L, nreg]（逐層）或 [nreg]（全域）

    def pick(self, sel: str, m: int):
        nreg = self.sess.nreg
        if sel in ("ORACLE", "EXACT-Q", "BASE-ATTN"):
            return topm(self.mass[sel].sum(0), m)
        if sel == "ORACLE-NEEDLE":                          # 偏離 2（診斷、不參與判定）：答案所在的塊先、其餘照 ORACLE 的分數
            nd = sorted(set(self.st.needle))
            rest = [j for j in topm(self.mass["ORACLE"].sum(0), nreg) if j not in set(nd)]
            return (nd + rest)[:m]
        if sel in ("ORACLE-L", "BASE-ATTN-L"):
            M = self.mass[sel[:-2]]
            return [topm(M[l], m) for l in range(M.shape[0])]
        if sel in ("QUEST-512", "KSCAN-2b"):
            return topm(self.mass[sel], m)
        if sel == "WRITE-TIME":
            return self.st.write_order()[:m]
        if sel == "RANDOM":
            return random.Random(f"{self.st.it['seed']}|a0|RANDOM").sample(range(nreg), nreg)[:m]
        if sel == "RECENT":
            return list(range(nreg - 1, nreg - 1 - m, -1))
        if sel == "SEQ":
            return list(range(m))
        raise KeyError(sel)


def onepass_hook(sess, st, m: int, code: int, nq: int, record: list):
    """ONE-PASS：第 li 層用這層的 q（來自已經組好的下層）在這層的 K（粗底）上打分，挑前 m 塊換成精修，再算這層的 attention。"""
    km = sess.km

    def hook(li, q):
        mass = ctx_mass(km, q, km.kv[li, 0, :, :st.ctx_end], nq, sess.nreg)
        s = topm(mass, m)
        st.comp.put_layer(li, s, code)
        record.append(s)
    return hook


# ============================================================================ 一題的 TF
def row_base(sess, st, cell, c):
    info = sess.info
    return dict(cell=cell, model=sess.model, n_blocks=sess.nb, item_id=st.it["item_id"], task=st.it["task"], seed=st.it["seed"],
                band=st.tr["band"], turn=st.turn, cid=c["cid"], kind=c["kind"], spec=c.get("spec", ""), base=c.get("base", ""),
                rho=c.get("rho", ""), m=c.get("m", ""), target=c.get("target", ""), selector=c.get("selector", ""),
                gran=c.get("gran", ""), deployable=int(c["deployable"]), bytes_base=c["bytes_base"], bytes_refine=c["bytes_refine"],
                bytes_index=c["bytes_index"], bytes_shared=c["bytes_shared"], bytes_region_bf16=A.region_bf16(info, sess.nreg),
                f=round(c["f"], 6), multiple=round(1 / c["f"], 4))


def run_item(sess, st, cfgs, cell, mode: str):
    """mode＝tf：每個配置 teacher forcing；gen：每個配置生成。回傳列（不含 run_id、ts、監控欄）。"""
    km = sess.km
    nq = len(st.q_ids)
    rows, sels = [], Selections(sess, st)
    tgt_code = {"int4": INT4, "bf16": ORIG}
    t_item = time.time()

    def measure(hook=None):
        if mode == "tf":
            return dict(lp=st.score(hook=hook))
        sc, stop, out = st.generate(hook=hook)
        return dict(score=sc, stop=stop, gen=out)

    def finish(c, res, t0, sel=None, gran=None):
        r = row_base(sess, st, cell, c)
        if mode == "tf":
            r.update(**ItemState.metrics(res["lp"], lp0, st.gold_ids))
            r.update(gold_lp_e0=m0["gold_lp"], d_gold_lp=r["gold_lp"] - m0["gold_lp"])
        else:
            r.update(score=res["score"], score_e0=e0_gen["score"], stop=res["stop"], n_gen=len(res["gen"]),
                     gen_sha=hashlib.sha1(json.dumps(res["gen"]).encode()).hexdigest()[:12],
                     same_as_e0=int(res["gen"] == e0_gen["gen"]))
        if sel is not None:
            r.update(needle_cover=needle_cover(sel, gran, st.needle), sel_hash=sel_hash(sel))
        sync()
        r["t_cfg_s"] = round(time.time() - t0, 3)
        rows.append(r)
    # E0（緩衝＝全部原值）
    t0 = time.time()
    res = measure()
    if mode == "tf":
        lp0 = res["lp"]
        m0 = ItemState.metrics(lp0, lp0, st.gold_ids)
    else:
        e0_gen = res
    finish(next(c for c in cfgs if c["cid"] == "E0"), res, t0)
    need = {c.get("selector") for c in cfgs if c["kind"] in ("prog", "drop")}
    if need & {"ORACLE", "ORACLE-L", "EXACT-Q", "ORACLE-NEEDLE"}:
        sels.mass["ORACLE"], q_exact = sels.st.question_pass(want_q="EXACT-Q" in need)
    else:
        q_exact = None
    # 均勻配置（直接寫進緩衝）
    for c in cfgs:
        if c["kind"] != "uniform":
            continue
        t0 = time.time()
        by = write_uniform(sess, st, c["spec"])
        assert sum(v for k, v in by.items() if not k.startswith("shared.")) == c["bytes_base"], (c["cid"], by)
        finish(c, measure(), t0)
    # 每個粗底：編碼一次 → 打分 → 這個粗底的漸進配置。丟塊對照的 BASE-ATTN 用 B32 第一趟的分數（登記 §1.5 第 6 點）
    drop_cfgs = [c for c in cfgs if c["kind"] == "drop"]
    bases = [b for b in A.BASES if any(c.get("base") == b for c in cfgs if c["kind"] == "prog")]
    if any(c["selector"] == "BASE-ATTN" for c in drop_cfgs) and A.DROP_BASE not in bases:
        bases.append(A.DROP_BASE)
    for b in bases:
        store, by = build_base(sess, st, A.BASES[b])
        st.comp.set_base(store)
        st.comp.apply(want_array(sess, [], "global", BASE, BASE))
        want_b = {c["selector"] for c in cfgs if c["kind"] == "prog" and c["base"] == b}
        if b == A.DROP_BASE and any(c["selector"] == "BASE-ATTN" for c in drop_cfgs):
            want_b.add("BASE-ATTN")
        if "EXACT-Q" in want_b:
            sels.mass["EXACT-Q"] = st.mass_from_q(q_exact, lambda li: km.kv[li, 0, :, :st.ctx_end])
        if want_b & {"BASE-ATTN", "BASE-ATTN-L", "QUEST-512", "KSCAN-2b"}:
            sels.mass["BASE-ATTN"], q_base = st.question_pass(want_q=bool(want_b & {"QUEST-512", "KSCAN-2b"}))
            if b == A.DROP_BASE:
                sels.mass["BASE-ATTN@" + b] = sels.mass["BASE-ATTN"]
            if "QUEST-512" in want_b:
                sels.mass["QUEST-512"] = st.quest_score(q_base)
            if "KSCAN-2b" in want_b:
                sels.mass["KSCAN-2b"] = st.kscan_mass(q_base)
            del q_base
        for c in cfgs:
            if c["kind"] != "prog" or c["base"] != b:
                continue
            t0 = time.time()
            code = tgt_code[c["target"]]
            if c["selector"] == "ONE-PASS":
                st.comp.apply(want_array(sess, [], "global", BASE, BASE))
                rec = []
                res = measure(hook=onepass_hook(sess, st, c["m"], code, nq, rec))
                finish(c, res, t0, sel=rec, gran="layer")
                continue
            sel = sels.pick(c["selector"], c["m"])
            st.comp.apply(want_array(sess, sel, c["gran"], code, BASE))
            finish(c, measure(), t0, sel=sel, gran=c["gran"])
        st.comp.set_base(None)
        del store
        torch.cuda.empty_cache()
    # 丟塊對照（沒挑中的區內塊用 key_keep 遮掉；那些位置放什麼都看不到，組 KV 時放原值）
    for c in drop_cfgs:
        t0 = time.time()
        if c["selector"] == "BASE-ATTN":
            sel = topm(sels.mass["BASE-ATTN@" + A.DROP_BASE].sum(0), c["m"])
        else:
            sel = sels.pick(c["selector"], c["m"])
        st.comp.apply(want_array(sess, sel, c["gran"], tgt_code[c["target"]], ORIG))
        km.key_keep = keep_mask(sess, st, sel, c["gran"])
        try:
            res = measure()
        finally:
            km.key_keep = None
        finish(c, res, t0, sel=sel, gran=c["gran"])
    return rows, time.time() - t_item


# ============================================================================ 指令
def _cell_args(a):
    if not a.cell:
        raise SystemExit("要 --cell")
    c = A.CELLS[a.cell]
    if a.model and a.model != c["model"]:
        raise SystemExit(f"--model {a.model} 和 --cell {a.cell}（{c['model']}）不一致")
    return c["model"], c["nb"]


def _eval_keys():
    return {(r["task"], r["seed"]) for k in A.SETS for r in A.plan_set(k)}


def _session(a, model, nb):
    sess = Session(model, nb, store=a.store, seg_blocks=a.seg_blocks, calib_xl=a.calib_xl)
    meta = sess.load_calib(_eval_keys())
    print(f"[a0] {model} nb={nb} store={sess.store_mode} seg={sess.seg} calib={sess.calib_xl}（run {meta['run_id']}，R={meta['R']}）", flush=True)
    return sess


def _loop(a, kind: str):
    from m16_c1a_predict import finish_watch, host_level, intruders, start_watch
    model, nb = _cell_args(a)
    plan = json.load(open(os.path.join(A.OUT, f"plan_{a.cell}.json")))
    cfgs_all = plan["configs"]
    if kind == "gen":
        sel = json.load(open(os.path.join(A.OUT, f"gen_selection_{a.cell}.json")))
        by = {c["cid"]: c for c in cfgs_all}
        cfgs = [by[cid] for cid in sel["configs"]]
        sets, fields = ["gen1", "gen2"], A.GEN_FIELDS
    else:
        cfgs, sets, fields = cfgs_all, ["tf"], A.TF_FIELDS
    if a.smoke:
        keep = {"E0", "U|q:int4_kivi", f"U|{A.BASES['B32']}", "P|B32|1/8|int4|ORACLE", "P|B32|1/8|int4|ORACLE-L",
                "P|B32|1/8|int4|EXACT-Q", "P|B32|1/8|int4|BASE-ATTN", "P|B32|1/8|int4|BASE-ATTN-L", "P|B32|1/8|int4|ONE-PASS",
                "P|B32|1/8|int4|QUEST-512", "P|B32|1/8|int4|KSCAN-2b", "P|B32|1/8|int4|WRITE-TIME", "P|B16|1/16|bf16|ORACLE",
                "P|B32|1/8|int4|ORACLE-NEEDLE",
                "D|1/8|bf16|ORACLE", "D|1/8|int4|ORACLE-L", "D|1/8|bf16|BASE-ATTN"}
        cfgs = [c for c in cfgs if c["cid"] in keep]
        if not a.out:
            raise SystemExit("--smoke 要給 --out（冒煙的輸出不進 results）")
    si, sk = A.parse_shard(a.shard)
    out = a.out or os.path.join(A.RAW, f"a0_{kind}_{a.cell}_s{si}.csv.gz")
    paths = sorted(set(glob_out(kind, a.cell)) | {out}) if not a.smoke else [out]
    todo = A.pending(sets, a.shard, [p for p in paths if os.path.exists(p)])
    if a.max_items:
        todo = todo[:a.max_items]
    print(f"[a0 {kind}] {a.cell}：{len(cfgs)} 個配置、這個分片還有 {len(todo)} 題 → {out}", flush=True)
    if not todo:
        return
    sess = _session(a, model, nb)
    rid = A.run_id()
    watch = start_watch(out)
    fo, w = A.open_out(out, fields)
    t_start = time.time()
    try:
        for k, r in enumerate(todo):
            if a.budget_min and (time.time() - t_start) / 60 > a.budget_min:
                print(f"[a0 {kind}] 時間預算 {a.budget_min} 分鐘到了 → 停在題與題之間（做完 {k} 題）", flush=True)
                break
            it = sess.T.make_item(r["task"], r["b1"], r["b2"], r["seed"], sess.ans.tok, sess.P.get(r["task"]), n_blocks=nb)
            if it["item_id"] != r["item_id"]:
                raise RuntimeError(f"item_id {it['item_id']} ≠ 計畫的 {r['item_id']}")
            st = ItemState(sess, it, r["turn"])
            rows, t_item = run_item(sess, st, cfgs, a.cell, kind)
            lvl, intr = host_level(), intruders(watch)
            for x in rows:
                x.update(run_id=rid, ts=A.ts(), calib_run_id=sess.calib_rid, t_prefill_s=round(st.t_prefill, 2),
                         own_gpu_intruders=intr, host_level=lvl, n_rows_item=len(rows))
                if kind == "gen":
                    x["set"] = r["set"]
                w.writerow({f: x.get(f, "") for f in fields})
            fo.flush()
            e0 = rows[0]
            print(f"[{k + 1}/{len(todo)}] {it['item_id']} t{r['turn']} prefill {st.t_prefill:.1f}s 題 {t_item:.1f}s "
                  f"{len(rows)} 列 copies={st.comp.n_block_copies}/{st.comp.n_layer_copies} "
                  + (f"E0 gold_lp {e0['gold_lp']:.3f}" if kind == "tf" else f"E0 score {e0['score']}"), flush=True)
            del st
            torch.cuda.empty_cache()
    finally:
        fo.close()
        finish_watch(watch, out)


def glob_out(kind, cell):
    import glob
    return glob.glob(A.out_glob(kind, cell))


def cmd_tf(a):
    _loop(a, "tf")


def cmd_gen(a):
    _loop(a, "gen")


def cmd_calib(a):
    """m16_xstack.cmd_calib_xl 原樣（同一套校準題、窗口 4）；rank_max 96（同位元組配對的 rv 最大到 96）；qwen14 補模型資訊與題目參數。"""
    model = a.model
    if model not in XS.MODEL_INFO:
        XS.MODEL_INFO[model] = dict(A.INFO[model])                    # 只在這個 process 裡補（m16_xstack 的檔不改）
    ns = SimpleNamespace(model=model, tasks=["T2", "T3", "T4", "T5"], n=2, seed0=XS.CALIB_SEED0,
                         params_file=params_file_for(model, a.params_file), w=4, rank_max=96,
                         out=a.out or calib_path(model), summary_dir=os.path.join(A.OUT, "calib"), force=a.force)
    XS.cmd_calib_xl(ns)


def cmd_gpu_selftest(a):
    """組 KV、forward、位元組的自我檢查（不計分、不寫 results）。"""
    model, nb = _cell_args(a)
    sess = _session(a, model, nb)
    km = sess.km
    plan = A.plan_set("tf")[0]
    it = sess.T.make_item(plan["task"], plan["b1"], plan["b2"], plan["seed"], sess.ans.tok, sess.P.get(plan["task"]), n_blocks=nb)
    st = ItemState(sess, it, 0)
    out = {}
    lp0 = st.score()
    # 1. forward_hooked（hook＝None）＝KVModel.forward_span
    ids = torch.tensor(st.tf_seq(), device=km.device)
    a1 = km.forward_span(ids, st.ctx_end, want_logits=True, n_logits=len(st.gold_ids))
    kv1 = km.kv[:, :, :, st.ctx_end:st.ctx_end + len(ids)].clone()
    a2 = forward_hooked(km, ids, st.ctx_end, hook=None, want_logits=True, n_logits=len(st.gold_ids))
    kv2 = km.kv[:, :, :, st.ctx_end:st.ctx_end + len(ids)].clone()
    out["forward_hooked_equal"] = bool(torch.equal(a1, a2) and torch.equal(kv1, kv2))
    # 2. 有 hook 但不改 KV：logits 也要相同
    a3 = forward_hooked(km, ids, st.ctx_end, hook=lambda li, q: None, want_logits=True, n_logits=len(st.gold_ids))
    out["noop_hook_equal"] = bool(torch.equal(a1, a3))
    # 3. 組 KV：寫均勻 → 換回全部原值 → 和 E0 逐位元相同
    write_uniform(sess, st, A.BASES["B32"])
    st.comp.apply(want_array(sess, [], "global", ORIG, ORIG))
    out["compose_back_to_orig_equal_e0"] = bool(torch.equal(st.score(), lp0))
    # 4. 分段編碼 vs 整段編碼（區內前 32 塊）：位元組要相同；cuBLAS 依列數挑 GEMM 演算法，fp32 捨入可能不同 → 不要求逐位元，
    #    要求重建誤差（‖X̂−X‖／‖X‖）的差 ≤1%（偏離 1：所有格子固定分段 16 塊）
    stages = XS.validate(XS.parse_spec(A.BASES["B16"]), sess.info, sess.nreg)
    nb32 = min(32, sess.nreg)
    X = torch.cat([st.orig.block(j) for j in range(nb32)], dim=3)
    pos = torch.arange(CHUNK, (nb32 + 1) * CHUNK, device=km.device)
    d_full, _, by_full, _ = XS.t_pipeline(stages, X, pos, sess.cx, sess.info)
    segs, by_seg = [], {}
    for j0 in range(0, nb32, 8):
        Xs = torch.cat([st.orig.block(j) for j in range(j0, j0 + 8)], dim=3)
        ps = torch.arange((j0 + 1) * CHUNK, (j0 + 9) * CHUNK, device=km.device)
        d, _, b, _ = XS.t_pipeline(stages, Xs, ps, sess.cx, sess.info)
        segs.append(d)
        for k, v in b.items():
            by_seg[k] = v if k.startswith("shared.") else by_seg.get(k, 0) + v
    d_seg = torch.cat(segs, dim=3)
    nx = X.float().norm()
    e_full, e_seg = float((d_full.float() - X.float()).norm() / nx), float((d_seg.float() - X.float()).norm() / nx)
    out["segmented_bytes_equal"] = by_full == by_seg
    out["segmented_relerr_full"], out["segmented_relerr_seg"] = round(e_full, 6), round(e_seg, 6)
    out["segmented_frac_values_differ"] = round(float((d_full != d_seg).float().mean()), 6)
    out["segmented_relerr_ok"] = abs(e_seg - e_full) <= 0.01 * e_full
    del X, d_full, segs, d_seg
    # 5. INT4 store＝t_full_q 的一塊
    x = st.orig.block(3)
    y, _ = XS.t_full_q(x.contiguous(), "int4_kivi", CHUNK)
    out["int4_equal_t_full_q"] = bool(torch.equal(st.int4.block(3), y))
    # 6. ONE-PASS：m＝0 時和「全部粗底」的 TF 相同；m＝nreg、精修 BF16 時和 E0 相同（所有塊都換回原值）
    store, by = build_base(sess, st, A.BASES["B32"])
    st.comp.set_base(store)
    st.comp.apply(want_array(sess, [], "global", BASE, BASE))
    lp_base = st.score()
    st.comp.apply(want_array(sess, [], "global", BASE, BASE))
    lp_m0 = st.score(hook=onepass_hook(sess, st, 0, ORIG, len(st.q_ids), []))
    out["onepass_m0_equal_base"] = bool(torch.equal(lp_m0, lp_base))
    st.comp.apply(want_array(sess, [], "global", BASE, BASE))
    lp_all = st.score(hook=onepass_hook(sess, st, sess.nreg, ORIG, len(st.q_ids), []))
    out["onepass_all_bf16_equal_e0"] = bool(torch.equal(lp_all, lp0))
    # 7. 挑法的形狀與 ORACLE 的分數加總（每個 q 在 [0, ctx_end) 的機率和＝1 → 區內的和 ≤ L·Hq·nq）
    st.comp.apply(want_array(sess, [], "global", ORIG, ORIG))
    mass, q = st.question_pass()
    tot = float(mass.sum())
    out["oracle_mass_sum"] = round(tot, 3)
    out["oracle_mass_bound"] = km.L * km.Hq * len(st.q_ids)
    out["oracle_mass_ok"] = 0 < tot <= km.L * km.Hq * len(st.q_ids) + 1e-6
    out["needle_blocks"] = st.needle
    out["oracle_top8"] = topm(mass.sum(0), 8)
    out["copies"] = [st.comp.n_block_copies, st.comp.n_layer_copies]
    out["mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
    must = ("forward_hooked_equal", "noop_hook_equal", "compose_back_to_orig_equal_e0", "segmented_bytes_equal", "segmented_relerr_ok",
            "int4_equal_t_full_q", "onepass_m0_equal_base", "onepass_all_bf16_equal_e0", "oracle_mass_ok")
    ok = all(out[k] is True for k in must)
    out["pass"] = ok
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
    if a.out:
        json.dump(dict(out, run_id=A.run_id(), ts=A.ts(), cell=a.cell), open(a.out, "w"), indent=1, ensure_ascii=False)
    if not ok:
        raise SystemExit("gpu-selftest 沒過（規則 2：停）")
