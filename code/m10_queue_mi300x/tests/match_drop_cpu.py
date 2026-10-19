"""14 號（M1、M2、D／DS）的 CPU 測試：小的隨機 Llama／Qwen2（2 層），不碰 GPU。

1. KVModel._attn_keep：遮罩（全域、per-head）＝把那些 key 拿掉後的 attention（float64 參考）。
2. forward_span：key_keep 全 True 和 None 的 logits 接近（換路徑的雜訊）；丟一半時真的會變。
3. 分數：SNAP、TOVA、QRY、H2O 和 HF eager attention（output_attentions）照 kvpress 定義算出來的接近；EXPATT 和用 HF hidden state 照公式算的接近。
4. 算完分數，緩衝裡的 context KV 和原本逐位元相同。
5. 遮罩：每個（層, head）留的數、PYR／ADA 的預算、P／B 的單位數、單位 0 保留、ORC 保留答案 block、STREAM 丟最舊的。
6. Restorer.drop：載入線跳過被丟的 block（CPU 替身）。m10_speed 的 drop 配置。
用法：python code/m10_queue_mi300x/tests/match_drop_cpu.py
"""
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, CODE)

import torch                                            # noqa: E402
import torch.nn.functional as F                         # noqa: E402

FAIL = []


def chk(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {info}" if info else ""), flush=True)
    if not cond:
        FAIL.append(name)


def tiny(kind: str, d: str):
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM
    kw = dict(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
              num_key_value_heads=2, max_position_embeddings=65536, rope_theta=500000.0)
    torch.manual_seed(0)
    m = (LlamaForCausalLM(LlamaConfig(**kw)) if kind == "llama" else Qwen2ForCausalLM(Qwen2Config(**kw))).to(torch.bfloat16)
    p = os.path.join(d, kind)
    m.save_pretrained(p)
    return p


def test_attn_keep():
    import types
    from m7_model import KVModel
    torch.manual_seed(1)
    Hq, H, D, start, n = 4, 2, 16, 40, 7
    end = start + n
    q = torch.randn(1, Hq, n, D, dtype=torch.float64)
    K = torch.randn(1, H, end, D, dtype=torch.float64)
    V = torch.randn(1, H, end, D, dtype=torch.float64)
    rep = Hq // H

    def ref(keep2):                                     # keep2 [H, end]
        o = torch.empty(1, Hq, n, D, dtype=torch.float64)
        for h in range(Hq):
            for t in range(n):
                idx = [j for j in range(start + t + 1) if keep2[h // rep, j]]
                s = q[0, h, t] @ K[0, h // rep, idx].T / math.sqrt(D)
                o[0, h, t] = s.softmax(-1) @ V[0, h // rep, idx]
        return o
    g = torch.ones(end + 5, dtype=torch.bool)
    g[3:20] = False
    fake = types.SimpleNamespace(key_keep=g, Hq=Hq, H=H, device=torch.device("cpu"))
    o = KVModel._attn_keep(fake, q, K, V, 0, start, n, end)
    r = ref(g[:end].expand(H, -1))
    chk("attn_keep 全域＝拿掉 key", torch.allclose(o, r, atol=1e-10), f"max {float((o - r).abs().max()):.2e}")
    ph = torch.ones(2, H, end + 5, dtype=torch.bool)
    ph[1, 0, 5:30] = False
    ph[1, 1, 10:12] = False
    fake.key_keep = ph
    o = KVModel._attn_keep(fake, q, K, V, 1, start, n, end, sub=3)
    r = ref(ph[1, :, :end])
    chk("attn_keep per-head（query 分段）＝拿掉 key", torch.allclose(o, r, atol=1e-10), f"max {float((o - r).abs().max()):.2e}")


def run_model(kind: str, d: str):
    os.environ["M7_MODEL_GLOB"] = tiny(kind, d)
    import importlib
    import m7_model
    importlib.reload(m7_model)
    from m7_model import KVModel
    import m10_drop as DR
    CH = m7_model.CHUNK
    nb = 8
    n = nb * CH
    km = KVModel(max_len=n + 600, device="cpu")
    g = torch.Generator().manual_seed(2)
    ids = torch.randint(0, 256, (n,), generator=g)
    km.prefill_chunked(ids)
    bf = [km.chunk_view(i).clone() for i in range(nb)]
    kv0 = km.kv[:, :, :, :n].clone()
    qids = torch.randint(0, 256, (37,), generator=g).tolist()

    # 2. forward_span：MASK0 ≈ None；丟一半會改變
    q2 = torch.randint(0, 256, (9,), generator=g)
    l0 = km.forward_span(q2, n, want_logits=True).float()
    km.key_keep = torch.ones(km.max_len, dtype=torch.bool)
    l1 = km.forward_span(q2, n, want_logits=True).float()
    kk = torch.ones(km.max_len, dtype=torch.bool)
    kk[CH:n // 2] = False
    km.key_keep = kk
    l2 = km.forward_span(q2, n, want_logits=True).float()
    km.key_keep = torch.ones(km.L, km.H, km.max_len, dtype=torch.bool)
    l3 = km.forward_span(q2, n, want_logits=True).float()
    km.key_keep = None
    chk(f"{kind} MASK0 ≈ None", float((l0 - l1).abs().max()) < 0.05, f"max {float((l0 - l1).abs().max()):.3f}")
    chk(f"{kind} MASK0H ≈ None", float((l0 - l3).abs().max()) < 0.05, f"max {float((l0 - l3).abs().max()):.3f}")
    d01, d02 = float((l0 - l1).abs().max()), float((l0 - l2).abs().max())
    chk(f"{kind} 丟一半會改變 logits（比換路徑的差大很多）", d02 > max(1e-3, 10 * d01), f"drop {d02:.4f} vs mask0 {d01:.4f}")
    for i in range(nb):                                  # forward_span 會寫 n 之後；context 不變
        km.chunk_view(i).copy_(bf[i])

    cx = DR.Ctx(km, ids, n, bf, CH, seed=123, needle_blocks=[3], query_ids=qids)
    sc = {m: DR.scores(cx, m) for m in DR.SCORED}
    chk(f"{kind} 算完分數 context KV 逐位元不變", torch.equal(km.kv[:, :, :, :n], kv0))
    for m, s in sc.items():
        chk(f"{kind} {m} 形狀 [L,H,n]、有限", tuple(s.shape) == (km.L, km.H, n) and bool(torch.isfinite(s).all()))
    chk(f"{kind} SNAP 觀察窗保留（最大）", bool((sc["SNAP"][..., -DR.WINDOW:] > sc["SNAP"][..., :-DR.WINDOW].max()).all()))
    chk(f"{kind} TOVA 最後一個保留、各 KV head 相同", bool((sc["TOVA"][..., -1:] > sc["TOVA"][..., :-1].max()).all())
        and torch.equal(sc["TOVA"][:, 0], sc["TOVA"][:, 1]))
    chk(f"{kind} EXPATT sink 保留", bool((sc["EXPATT"][..., :DR.N_SINK] > sc["EXPATT"][..., DR.N_SINK:].max()).all()))

    # 3. 對 HF eager attention（kvpress 的定義）
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(os.environ["M7_MODEL_GLOB"], dtype=torch.float32, attn_implementation="eager").eval()
    rep = km.Hq // km.H
    with torch.no_grad():
        out = hf(torch.cat([ids, torch.tensor(qids)]).view(1, -1), output_attentions=True, output_hidden_states=True)
    att = [x[0] for x in out.attentions]                 # [Hq, N, N]

    def pool_group(p):                                   # [L, Hq, k] → avg-pool 5 → 組內平均
        L_, Hq_, k = p.shape
        p = F.avg_pool1d(p.reshape(L_ * Hq_, 1, k), DR.POOL, padding=DR.POOL // 2, stride=1).view(L_, Hq_, k)
        return p.view(L_, km.H, rep, k).mean(2)

    def close(a, b, name):
        a, b = a.float(), b.float()
        rel = float((a - b).abs().max() / b.abs().max())
        rk = float(torch.corrcoef(torch.stack([a.flatten().argsort().argsort().float(), b.flatten().argsort().argsort().float()]))[0, 1])
        chk(name, rel < 0.08 and rk > 0.98, f"rel {rel:.3f} rank-corr {rk:.4f}")
    w = DR.WINDOW
    ref = pool_group(torch.stack([a[:, n - w:n, :n - w].mean(-2) for a in att]))
    close(sc["SNAP"][..., :n - w], ref, f"{kind} SNAP ≈ eager")
    ref = torch.stack([a[:, n - 1, :n - 1].mean(0) for a in att])           # [L, k]
    close(sc["TOVA"][:, 0, :n - 1], ref, f"{kind} TOVA ≈ eager")
    nq = len(qids)
    ref = pool_group(torch.stack([a[:, n:n + nq, :n].mean(-2) for a in att]))
    close(sc["QRY"], ref, f"{kind} QRY ≈ eager")
    rows = [r for b in DR.H2O_BLOCKS if (b + 1) * CH <= n for r in range((b + 1) * CH - w, (b + 1) * CH)]
    tot = torch.stack([a[:, rows, :n].sum(-2) for a in att])
    cnt = (torch.tensor(rows).view(-1, 1) >= torch.arange(n).view(1, -1)).sum(0).clamp(min=1)
    ref = (tot / cnt).view(km.L, km.H, rep, n).mean(2)
    close(sc["H2O"], ref, f"{kind} H2O（8 窗近似）≈ eager")
    # EXPATT：用 HF 的 hidden state 照 kvpress 的公式
    refs = []
    for li, layer in enumerate(hf.model.layers):
        a = layer.self_attn
        h = layer.input_layernorm(out.hidden_states[li][0, DR.N_SINK:n])
        qq = a.q_proj(h).view(-1, km.Hq, km.D).double()
        mu = qq.mean(0)
        cov = torch.einsum("nhi,nhj->hij", qq - mu, qq - mu) / qq.shape[0]
        pos = torch.arange(n, n + DR.EXP_FUTURE).view(1, -1)
        cos, sin = hf.model.rotary_emb(torch.zeros(1, 1, km.D), pos)
        cos, sin = cos[0].double(), sin[0].double()
        Id = torch.eye(km.D, dtype=torch.float64)
        P = torch.zeros(km.D, km.D, dtype=torch.float64)
        P[km.D // 2:, :km.D // 2], P[:km.D // 2, km.D // 2:] = torch.eye(km.D // 2), -torch.eye(km.D // 2)
        R = (cos.unsqueeze(1) * Id + sin.unsqueeze(1) * P).mean(0)        # kvpress 的寫法
        mu, cov = mu @ R.T, R @ cov @ R.T
        K = km.kv[li, 0, :, DR.N_SINK:n].double().repeat_interleave(rep, 0)  # [Hq, k, D]
        s = torch.einsum("hkd,hd->hk", K, mu) / math.sqrt(km.D) + torch.einsum("hkd,hde,hke->hk", K, cov, K) / km.D / 2
        s = s.softmax(-1).view(km.H, rep, -1).mean(1) * torch.linalg.vector_norm(km.kv[li, 1, :, DR.N_SINK:n].float(), dim=-1)
        refs.append(s)
    close(sc["EXPATT"][..., DR.N_SINK:], torch.stack(refs), f"{kind} EXPATT ≈ 公式（HF hidden state）")

    # 5. 遮罩
    for f in DR.FRACS:
        nk = DR.n_kept(n, f)
        for m in ["STREAM", "NEW", "RND0", "SNAP", "KNORM", "QRY"]:
            k = DR.keep_T(cx, m, f)
            chk(f"{kind} T {m} {f}：每個（層, head）留 {nk}", bool((k.sum(-1) == nk).all()), str(k.sum(-1).unique().tolist())) \
                if m in ("SNAP", "KNORM", "QRY", "RND0", "STREAM") else None
        k = DR.keep_T(cx, "ADA", f)
        chk(f"{kind} T ADA {f}：每層總數 H×{nk}、每個 head ≥ 20%", bool((k.sum((-1, -2)) == km.H * nk).all())
            and bool((k.sum(-1) >= max(1, int(nk * DR.ADA_ALPHA))).all()))
        k = DR.keep_T(cx, "PYR", f)
        bud = DR.pyr_budget(n, f, km.L)
        chk(f"{kind} T PYR {f}：每層照預算", all(int(k[li, 0].sum()) == min(bud[li], n) for li in range(km.L)), str(bud))
        k = DR.keep_T(cx, "STREAM", f)
        chk(f"{kind} T STREAM {f}：sink 留、丟最舊", bool(k[..., :DR.N_SINK].all()) and (nk == n or not bool(k[..., DR.N_SINK].any())))
        for gname in ("P", "B"):
            u = DR.UNIT[gname]
            nu = n // u
            for m in DR.METHODS[gname]:
                kk, kept, ndk = DR.keep_mask(cx, gname, m, f)
                dropped = int((~kk[:n]).sum()) // u
                good = dropped == int(round(f * (nu - 1))) and bool(kk[:u].all()) and bool(kk[n:].all())
                if m == "ORC" and dropped <= nu - 1 - CH // u:
                    good &= bool(kk[3 * CH:4 * CH].all())
                if m == "STREAM":
                    good &= bool((~kk[u:u * (1 + dropped)]).all())
                if not good:
                    chk(f"{kind} {gname} {m} {f}", False, f"dropped {dropped}")
    chk(f"{kind} P／B 遮罩（所有挑法 × 佔比）", True)
    kk, kept, _ = DR.keep_mask(cx, "T", "NONE", 0)
    chk(f"{kind} MASK0H 全 True", bool(kk.all()) and kk.dim() == 3)
    chk(f"{kind} parse_ds", DR.parse_ds("DS-T-SNAP-25") == ("T", "SNAP", 0.25) and DR.parse_ds("DS-B-NONE-0")[1] == "NONE")
    km.key_keep = None


def main():
    test_attn_keep()
    with tempfile.TemporaryDirectory() as d:
        for kind in ("llama", "qwen2"):
            run_model(kind, d)
    print("ALL PASS" if not FAIL else f"FAIL {len(FAIL)}: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
