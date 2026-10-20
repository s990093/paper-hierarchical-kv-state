#!/usr/bin/env python3
"""m16_s1_kvdump.py — 第三階段 S1：把真的 KV（BF16）與各低精度格式的實際位元組存到磁碟，給 m16_s1_lossless.py 分析。

計畫：docs/stage3_20261019/experiment_matrix.md 的 S1。**要用 GPU**（KVModel 的 prefill），只能在第三階段的 GPU 窗口跑
（第二階段的佇列跑完之後；規則見 experiment_matrix.md §6）。分析那一支只用 numpy＋zstandard，不碰 torch。

  python code/m16_s1_kvdump.py --model llama --task T4 --n-blocks 64 --seed 70000 --out <dir>
  python code/m16_s1_kvdump.py --model llama --task T4 --n-blocks 252 --layers 0,8,16,24,31 --out <dir>

存什麼（<out>/）：
  kv_bf16.u16                    原始 BF16（uint16），依 block 順序串接，每個 block 是 [L', 2, H, 512, D] 連續（L'＝選的層）
  <fmt>.<part>.bin               m10_codec.encode(block, fmt) 每個 part 的原始位元組，依 block 順序串接
  meta.json                      形狀、每個 part 每個 block 的 dtype／shape／bytes、題目、run_id、git commit
內容：T4（HotpotQA 段落＝自然文字）與 T2（RULER noise haystack＝重複的合成句子）兩種，分析時分開看，
因為重複文字的 KV 可能特別好壓，不能拿來代表一般情況。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FMTS_DEFAULT = ["fp8_token", "fp8_head", "int8_token", "int4_kivi"]


def rope_meta(m, ids, n_tok: int, chunk: int, torch) -> dict:
    """10/20 補（給 m16_kvstruct.py 的 K1，算 RoPE 前的 K 要用）：實際用的 RoPE 參數與 inv_freq，以及第 0 層 K 在
    RoPE 前／後的探針（規則 6：分析端用它核對「dump 是 RoPE 後、位置從 0 起算、逆旋轉正確」）。
    探針用和 prefill 相同的形狀（整個 512-token chunk）重算第 0 層的 embedding → RMSNorm → k_proj（和上下文無關），
    取幾個位置存下來。任何一步失敗只記錯誤字串，不讓 dump 失敗（分析端看到 error 會改用資料自身的核對）。"""
    out: dict = {}
    try:
        rot = m.hf.model.rotary_emb
        rp = getattr(m.cfg, "rope_parameters", None)
        out["rope"] = dict(rope_type=str(getattr(rot, "rope_type", None)), attention_scaling=float(rot.attention_scaling),
                           inv_freq=[float(x) for x in rot.inv_freq.detach().float().cpu().tolist()],
                           inv_freq_dtype=str(rot.inv_freq.dtype), pairing="rotate_half",
                           rope_parameters=json.loads(json.dumps(dict(rp), default=str)) if rp else None)
    except Exception as e:                                     # noqa: BLE001
        out["rope"] = dict(error=f"{type(e).__name__}: {e}")
    try:
        hf, l0 = m.hf.model, m.hf.model.layers[0]
        pos = sorted({p for p in (0, 1, chunk - 1, chunk, n_tok // 2, n_tok - 1) if 0 <= p < n_tok})
        pre, post, cs, sn = [], [], [], []
        with torch.inference_mode():
            for c in sorted({p // chunk for p in pos}):
                s = c * chunk
                x = hf.embed_tokens(ids[s:s + chunk].view(1, chunk))
                k = l0.self_attn.k_proj(l0.input_layernorm(x)).view(1, chunk, m.H, m.D)
                if hasattr(l0.self_attn, "k_norm"):
                    k = l0.self_attn.k_norm(k)
                k = k.transpose(1, 2)                                                    # [1,H,512,D]，同 forward_span
                cos, sin = hf.rotary_emb(x, torch.arange(s, s + chunk, device=m.device).view(1, chunk))
                _, kr = m.apply_rope(k, k, cos, sin)
                for p in pos:
                    if p // chunk == c:
                        t = p - s
                        pre.append(k[0, :, t]); post.append(kr[0, :, t]); cs.append(cos[0, t]); sn.append(sin[0, t])

        def hx(ts):
            return torch.stack(ts).contiguous().view(torch.int16).cpu().numpy().tobytes().hex()
        out["rope_probe"] = dict(layer=0, positions=pos, shape_k=[len(pos), m.H, m.D], shape_cs=[len(pos), m.D],
                                 k_pre_bf16_hex=hx(pre), k_post_bf16_hex=hx(post), cos_bf16_hex=hx(cs), sin_bf16_hex=hx(sn))
    except Exception as e:                                     # noqa: BLE001
        out["rope_probe"] = dict(error=f"{type(e).__name__}: {e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["llama", "qwen"], required=True)
    ap.add_argument("--task", default="T4")
    ap.add_argument("--band", default="mid", help="needle 放的位置帶（T5 不用）")
    ap.add_argument("--n-blocks", type=int, default=64)
    ap.add_argument("--seed", type=int, default=70000)
    ap.add_argument("--layers", default="all", help="all 或逗號分隔的層號（128K 只存部分層，省空間）")
    ap.add_argument("--fmts", nargs="*", default=FMTS_DEFAULT)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import m10_tasks as T
    # M7_MODEL_GLOB 必須在 import m7_model 之前設（RUNLOG 10/15：否則靜默載入 Llama）
    os.environ["M7_MODEL_GLOB"] = os.environ.get(f"M7_{a.model.upper()}_GLOB") or T.MODELS[a.model]
    import numpy as np
    import torch
    import m10_codec as C
    from m7_model import CHUNK, KVModel

    os.makedirs(a.out, exist_ok=True)
    tok = T.load_tokenizer(os.environ["M7_MODEL_GLOB"])
    band = None if a.task in ("T5", "cwe") else a.band
    item = T.make_item(a.task, band, None, a.seed, tok, n_blocks=a.n_blocks)
    ids_list = item["context_ids"]
    n_tok = len(ids_list)
    assert n_tok == a.n_blocks * CHUNK, (n_tok, a.n_blocks)

    m = KVModel(max_len=n_tok, device="cuda:0")
    layers = list(range(m.L)) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    assert all(0 <= li < m.L for li in layers), layers
    ids = torch.tensor(ids_list, dtype=torch.long, device=m.device)
    t0 = dt.datetime.now(dt.timezone.utc)
    m.prefill_chunked(ids)
    torch.cuda.synchronize()
    t_prefill = (dt.datetime.now(dt.timezone.utc) - t0).total_seconds()

    lidx = torch.tensor(layers, device=m.device)
    files = {"bf16": open(os.path.join(a.out, "kv_bf16.u16"), "wb")}
    part_meta: dict[str, dict] = {}
    for i in range(a.n_blocks):
        blk = m.chunk_view(i).index_select(0, lidx).contiguous()          # [L',2,H,512,D] BF16
        assert torch.isfinite(blk.float()).all(), f"block {i} 有非有限值"
        files["bf16"].write(blk.view(torch.int16).cpu().numpy().tobytes())
        for f in a.fmts:
            e = C.encode(blk, f)
            for k, t in e.parts.items():
                key = f"{f}.{k}"
                if key not in files:
                    files[key] = open(os.path.join(a.out, f"{key}.bin"), "wb")
                    part_meta[key] = dict(dtype=str(t.dtype), shape=list(t.shape), bytes_per_block=t.numel() * t.element_size())
                assert list(t.shape) == part_meta[key]["shape"], (key, t.shape)
                files[key].write(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    for fh in files.values():
        fh.close()

    git = subprocess.run(["git", "-C", os.path.dirname(HERE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = dict(
        run_id=os.environ.get("RUN_ID", "NOT_SET"), ts=dt.datetime.now(dt.timezone.utc).isoformat(), git=git,
        model=a.model, model_path=os.environ["M7_MODEL_GLOB"], task=a.task, band=band, seed=a.seed,
        item_id=item.get("item_id"), n_blocks=a.n_blocks, n_tokens=n_tok, chunk=CHUNK,
        L_total=m.L, layers=layers, H=m.H, D=m.D, bf16_block_shape=[len(layers), 2, m.H, CHUNK, m.D],
        bf16_bytes_per_block=len(layers) * 2 * m.H * CHUNK * m.D * 2, fmts=a.fmts, parts=part_meta,
        t_prefill_s=t_prefill, hip_visible_devices=os.environ.get("HIP_VISIBLE_DEVICES"),
        torch=torch.__version__, numpy=np.__version__,
        # 10/20 補（K1 用）：排列、位置、K 是 RoPE 後（m7_model.forward_span 在 apply_rope 之後才寫進 kv）
        kv_layout="kv_bf16.u16=[n_blocks][len(layers)][2: 0=K,1=V][H][chunk][D] BF16 bits little-endian; "
                  "token t of block i is at position pos_start + i*chunk + t",
        pos_start=0, k_rope="post", v_rope="none",
        transformers=getattr(sys.modules.get("transformers"), "__version__", None),
        **rope_meta(m, ids, n_tok, CHUNK, torch))
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: meta[k] for k in ("model", "task", "n_tokens", "layers", "t_prefill_s")}, ensure_ascii=False))
    print("rope_probe:", meta["rope_probe"].get("error", "ok"), "| rope:", meta["rope"].get("error", "ok"))


if __name__ == "__main__":
    main()
