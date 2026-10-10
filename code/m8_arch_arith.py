"""m8_arch_arith.py — D5（新架構與新模態）的算術。不碰 GPU（不 import torch）。

  RUN_ID=<id> python code/m8_arch_arith.py --cfg-vl <dir> --cfg <dir> [--out results/m8_directions]

輸入：
  * results/m7_write_policy_mi300x/calib_c1.csv（run_id 20261008-130316-m7-c1，Llama-3.1-8B 每個 512-token chunk 的重算時間）
  * results/m7_write_policy_mi300x/tier_params.json（各層的讀寫頻寬）
  * --cfg-vl：Qwen3-VL-8B-Instruct 的 config.json（只下 config，不下權重）
  * --cfg：其他模型的 config.json（Gemma-3、gpt-oss、Nemotron-H、Qwen3-Next…）
輸出（每列有 run_id、ts、src_run）：
  d5_calib_check.csv  校準 α、β，並核對 Lit-C 的 Llama b/n
  d5_vlm.csv          (a) Qwen3-VL-8B：encoder vs LLM、三種格式、Cake 還原時間
  d5_ssm.csv          (b) 混合 SSM：檢查點大小、占比、寫出頻寬、載入 vs 重算
  d5_swa.csv          (c) 滑動視窗：全部層 vs 只存需要的，位元組與還原時間
  d5_pd.csv           (d) P/D 分離：不重疊／逐 chunk 管線／Cake 分割
  d5_mla.csv          (e) MLA／MQA：b/n（用 Lit-C 表的欄位重建 f）

模型（全部是算術，假設各模型的 MFU 與 Llama 在 m7 harness 上相同；Lit-C §1.3 的限制照樣適用）：
  f(i) = α·(線性 FLOPs of chunk i) + β·(注意力 FLOPs of chunk i)
  Cake 還原時間 T = min_b max(Σ_{i<b} f_i, Σ_{i≥b} ℓ_i)   （與 code/m7_restore_harness.py L218–227 write_boundary 的目標式相同）
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES7 = os.path.join(HERE, "..", "results", "m7_write_policy_mi300x")
GiB = 1 << 30
MiB = 1 << 20
KiB = 1 << 10
CHUNK = 512
TS = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
RUN_ID = os.environ.get("RUN_ID")
CALIB_RUN = "20261008-130316-m7-c1"


# ----------------------------------------------------------------------------- 共用
def write_boundary(n: int, f: list[float], ell: float) -> int:
    """與 code/m7_restore_harness.py L218–227 相同：回傳使 max(sum f[0:m], (n-m)*ell) 最小的 m。"""
    best, best_m, R = None, 0, 0.0
    for m in range(0, n + 1):
        if m > 0:
            R += f[m - 1]
        T = max(R, (n - m) * ell)
        if best is None or T < best:
            best, best_m = T, m
    return best_m


def cake_time(f: np.ndarray, ell: np.ndarray) -> tuple[float, int]:
    """一般化的 Cake：前段 [0,b) 重算、後段 [b,n) 載入（每個 chunk 的 ℓ 可以不同）。回傳 (T, b)。"""
    n = len(f)
    F = np.concatenate([[0.0], np.cumsum(f)])          # F[b] = Σ_{i<b} f_i
    Lr = np.concatenate([np.cumsum(ell[::-1])[::-1], [0.0]])  # Lr[b] = Σ_{i≥b} ℓ_i
    T = np.maximum(F, Lr)
    b = int(np.argmin(T))
    return float(T[b]), b


def attn_chunk_flops(i: int, per_key_flops: float, window: int | None = None) -> float:
    """chunk i（token 位置 512i+1 … 512i+512）的注意力 FLOPs；per_key_flops = Σ_層 2·n_q·(d_qk+d_v)。"""
    p = np.arange(CHUNK * i + 1, CHUNK * i + CHUNK + 1, dtype=np.float64)
    if window is not None:
        p = np.minimum(p, window)
    return float(per_key_flops * p.sum())


def calib(path: str):
    c = pd.read_csv(path)
    assert set(c.run_id) == {CALIB_RUN}, set(c.run_id)
    c = c[c.item == "f_chunk"]
    med = c.groupby("chunk_idx").ms.median().sort_index()
    # 規則 6：用 pos_end 欄驗算 chunk_idx（pos_end = 512·(i+1)）
    pe = c.groupby("chunk_idx").pos_end.first().sort_index()
    assert (pe.values == CHUNK * (pe.index.values + 1)).all(), "pos_end 與 chunk_idx 對不上"
    f = med.values.astype(float)
    n = len(f)
    lin_llama = 2 * llama_lin_params()          # FLOPs/token
    perkey_llama = 32 * 2 * 32 * (128 + 128)    # 32 層 × 2 × 32 q head × (d_qk+d_v)
    X = np.array([[CHUNK * lin_llama, attn_chunk_flops(i, perkey_llama)] for i in range(n)])
    (a, b), *_ = np.linalg.lstsq(X, f, rcond=None)
    pred = X @ np.array([a, b])
    return f, a, b, pred, lin_llama, perkey_llama


def llama_lin_params() -> float:
    h, inter, L = 4096, 14336, 32
    per = h * h * 2 + h * 1024 * 2 + 3 * h * inter
    return L * per


def bw_list():
    tp = json.load(open(os.path.join(RES7, "tier_params.json")))
    # 第一階段用過的頻寬（GiB/s）
    return tp, [("nfs", tp["nfs"]["read_GiBps"]), ("7Gbps", 0.814907), ("1GiBps", 1.0), ("25Gbps", 2.910383),
                ("vllm_cpu", 3.69), ("local_ssd", tp["local"]["read_GiBps"]), ("100Gbps", 11.641532),
                ("cpu_pinned", tp["cpu"]["read_GiBps"])]


def row(**kw):
    d = {"run_id": RUN_ID, "ts": TS}
    d.update(kw)
    return d


# ----------------------------------------------------------------------------- (0) 校準
def part_calib(out, f_all, a, b, pred_all):
    rows = []
    n = 64                       # 32K（calib_c1 有 80 個 chunk；α、β 用全部 80 個擬合，b/n 只看前 64 個，同 Lit-C）
    f, pred = f_all[:n], pred_all[:n]
    kv_tok = 131072
    for name, bw in bw_list()[1]:
        ell = CHUNK * kv_tok / (bw * GiB) * 1e3
        bm = write_boundary(n, list(f), ell)
        bp = write_boundary(n, list(pred), ell)
        rows.append(row(src_run=CALIB_RUN, item="llama_bn", bw_name=name, GiBps=bw, ell_ms=ell,
                        b_meas=bm, b_pred=bp, bn_meas=bm / n, bn_pred=bp / n,
                        T_cake_meas_ms=cake_time(f, np.full(n, ell))[0],
                        T_compute_ms=float(f.sum()), T_load_ms=n * ell))
    rows.append(row(src_run=CALIB_RUN, item="alpha_ms_per_flop", value=a))
    rows.append(row(src_run=CALIB_RUN, item="beta_ms_per_flop", value=b))
    rows.append(row(src_run=CALIB_RUN, item="max_rel_err_80", value=float(np.max(np.abs(pred_all - f_all) / f_all))))
    rows.append(row(src_run=CALIB_RUN, item="sum_f_ms_64", value=float(f.sum())))
    rows.append(row(src_run=CALIB_RUN, item="n_chunks_calib", value=len(f_all)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "d5_calib_check.csv"), index=False)
    return df


# ----------------------------------------------------------------------------- (a) VLM
def part_vlm(out, cfg_path, a, b, src_run):
    cfg = json.load(open(cfg_path))
    tc, vc = cfg["text_config"], cfg["vision_config"]
    # LLM 線性參數（非嵌入、不含 LM head）
    h, inter, L = tc["hidden_size"], tc["intermediate_size"], tc["num_hidden_layers"]
    nq, nkv, hd = tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
    per = h * nq * hd + 2 * h * nkv * hd + nq * hd * h + 3 * h * inter
    llm_lin = 2 * L * per                    # FLOPs/token
    llm_perkey = L * 2 * nq * (hd + hd)
    kv_tok = L * 2 * nkv * hd * 2            # bytes/token BF16
    assert abs(L * per / 1e9 - 6.95) < 0.02, L * per / 1e9   # Lit-C 表：Qwen3-VL-8B 的 active 6.95B
    # vision encoder（transformers qwen3_vl/modeling_qwen3_vl.py：Conv3d patch embed、27 個 block、merger＋3 個 deepstack merger）
    vh, vi, depth, vheads = vc["hidden_size"], vc["intermediate_size"], vc["depth"], vc["num_heads"]
    ps, tps, ms = vc["patch_size"], vc["temporal_patch_size"], vc["spatial_merge_size"]
    out_h = vc["out_hidden_size"]
    n_ds = len(vc["deepstack_visual_indexes"])
    patches_per_tok = ms * ms
    pe = 2 * (3 * tps * ps * ps) * vh                     # patch embed / patch
    blk = 2 * (vh * 3 * vh + vh * vh + 2 * vh * vi)        # qkv＋proj＋MLP(fc1,fc2) / patch / block
    merger = 2 * ((vh * ms * ms) ** 2 + vh * ms * ms * out_h)  # fc1 (4608²)＋fc2 (4608→4096) / LLM token
    enc_lin_tok = patches_per_tok * (pe + depth * blk) + (1 + n_ds) * merger
    vhd = vh // vheads
    enc_attn_per_patch_per_key = depth * 2 * vheads * (vhd + vhd)
    emb_bytes = (1 + n_ds) * out_h * 2                    # 主 embedding＋3 份 deepstack，BF16
    pix_bytes_raw = patches_per_tok * 3 * tps * ps * ps   # uint8 原始像素
    rows = []
    base = dict(src_run=src_run, model="Qwen3-VL-8B-Instruct", kv_bytes_tok=kv_tok, emb_bytes_tok=emb_bytes,
                pix_raw_bytes_tok=pix_bytes_raw, emb_over_kv=emb_bytes / kv_tok, pix_over_kv=pix_bytes_raw / kv_tok,
                llm_lin_gflop_tok=llm_lin / 1e9, enc_lin_gflop_tok=enc_lin_tok / 1e9)
    tp, bws = bw_list()
    for S, label in [(784, "frame448"), (4096, "img1024"), (16384, "img2048"), (65536, "img4096_max")]:
        enc_attn_tok = patches_per_tok * enc_attn_per_patch_per_key * S
        enc_ms_tok = a * enc_lin_tok + b * enc_attn_tok
        for n in (24, 64):   # 12,288 token（預設影片上限）與 32,768 token
            f_llm = np.array([a * CHUNK * llm_lin + b * attn_chunk_flops(i, llm_perkey) for i in range(n)])
            f_pix = f_llm + CHUNK * enc_ms_tok
            r_v = (CHUNK * enc_ms_tok) / f_llm.mean()
            for name, bw in bws:
                ell = CHUNK * kv_tok / (bw * GiB) * 1e3
                e_ell = ell * emb_bytes / kv_tok
                T_pix, b_pix = cake_time(f_pix, np.full(n, ell))
                T_emb, b_emb = cake_time(f_llm, np.full(n, ell))
                # embedding 和 KV 共用同一條 I/O：前段每個重算的 chunk 還要讀 embedding
                best = None
                for bb in range(n + 1):
                    T = max(f_llm[:bb].sum(), (n - bb) * ell + bb * e_ell)
                    if best is None or T < best[0]:
                        best = (T, bb)
                T_embs, b_embs = best
                rows.append(row(**base, scenario=label, patches_per_frame=S, n_chunks=n, tokens=n * CHUNK,
                                enc_attn_gflop_tok=enc_attn_tok / 1e9, enc_ms_per_chunk=CHUNK * enc_ms_tok,
                                f_llm_mean_ms=float(f_llm.mean()), r_v=r_v, bw_name=name, GiBps=bw, ell_ms=ell,
                                T_compute_pix_ms=float(f_pix.sum()), T_compute_emb_ms=float(f_llm.sum()),
                                T_load_ms=n * ell, T_cake_pix_ms=T_pix, b_pix=b_pix, T_cake_emb_sep_ms=T_emb,
                                b_emb_sep=b_emb, T_cake_emb_shared_ms=T_embs, b_emb_shared=b_embs,
                                gain_emb_sep=1 - T_emb / T_pix, gain_emb_shared=1 - T_embs / T_pix))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "d5_vlm.csv"), index=False)
    return df


# ----------------------------------------------------------------------------- (b) SSM 混合
def mamba2_layer_bytes(d_inner, n_groups, d_state, n_heads, head_dim, conv_k, dtype_bytes=2):
    # vLLM vllm/model_executor/layers/mamba/mamba_utils.py L127–151（tp=1）
    conv = (conv_k - 1) * (d_inner + 2 * n_groups * d_state)
    ssm = n_heads * head_dim * d_state
    return (conv + ssm) * dtype_bytes


def gdn_layer_bytes(nk, nv, dk, dv, conv_k, dtype_bytes=2):
    # vLLM mamba_utils.py L177–200
    conv = (conv_k - 1) * (dk * nk * 2 + dv * nv)
    ssm = nv * dv * dk
    return (conv + ssm) * dtype_bytes


# Lit-C 表（docs/research_20261009_explore/lit_C_architectures.md L95–104、L171–181）的欄位；〔算術，Lit-C〕
LITC_HYBRID = [
    # name, n_mamba, n_attn, attn_kv_layer_tok_bytes, state_MiB(全部遞迴層), kv_KiB_tok, G0, G32, chunk, max_ctx
    ("Jamba-1.5-Mini", 28, 4, 4096, 8.3, 16, 23.2, 24.3, 1024, 262144),
    ("Falcon-H1-7B", 44, 44, 1024, 66.9, 44, 13.9, 18.3, 256, 262144),
    ("Nemotron-H-8B", 24, 4, 4096, 49.4, 16, 14.3, 15.4, 128, 8192),
    ("Nemotron-Nano-9B-v2", 27, 4, 4096, 69.4, 16, 15.7, 17.1, 128, 131072),
    ("Granite-4.0-H-Small", 36, 4, 4096, 73.7, 16, 17.2, 18.3, 256, 131072),
    ("Qwen3-Next-80B-A3B", 36, 12, 2048, 37.7, 24, 6.6, 9.8, 64, 262144),
    ("Kimi-Linear-48B-A3B", 20, 7, 1152, 21.4, 7.9, 5.5, 7.9, 64, 1048576),
]


def part_ssm(out, cfgdir, a, b, src_run):
    tp = json.load(open(os.path.join(RES7, "tier_params.json")))
    # 規則 6 式的交叉驗算：兩個模型從 config 重算一份狀態大小，必須對上 Lit-C
    nh = json.load(open(os.path.join(cfgdir, "nvidia_Nemotron-H-8B-Base-8K.json")))
    nm = nh["hybrid_override_pattern"].count("M")
    na = nh["hybrid_override_pattern"].count("*")
    nh_layer = mamba2_layer_bytes(nh["expand"] * nh["hidden_size"], nh["n_groups"], nh["ssm_state_size"],
                                  nh["mamba_num_heads"], nh["mamba_head_dim"], nh["conv_kernel"])
    assert (nm, na) == (24, 4), (nm, na)
    assert abs(nm * nh_layer / MiB - 49.4) < 0.1, nm * nh_layer / MiB
    qn = json.load(open(os.path.join(cfgdir, "Qwen_Qwen3-Next-80B-A3B-Instruct.json")))
    qn_layer = gdn_layer_bytes(qn["linear_num_key_heads"], qn["linear_num_value_heads"], qn["linear_key_head_dim"],
                               qn["linear_value_head_dim"], qn["linear_conv_kernel_dim"])
    n_lin = qn["num_hidden_layers"] - qn["num_hidden_layers"] // qn["full_attention_interval"]
    assert abs(n_lin * qn_layer / MiB - 37.7) < 0.1, n_lin * qn_layer / MiB
    rows = []
    for (name, n_m, n_a, attn_layer_tok, st_mib, kv_kib, G0, G32, mchunk, max_ctx) in LITC_HYBRID:
        S = st_mib * MiB
        per_layer_state = S / n_m
        kv_tok = kv_kib * KiB
        # vLLM all 模式的 block 大小（vllm/model_executor/models/config.py L244–263）
        attn_tok_per_state = math.ceil(per_layer_state / attn_layer_tok)
        ch = math.lcm(mchunk, 16)
        blk = ch * math.ceil(attn_tok_per_state / ch)
        page = blk * attn_layer_tok                      # 每層每 block 一頁（mamba 狀態補齊到這個大小）
        bytes_per_block_all = (n_m + n_a) * page
        frac_all = n_m / (n_m + n_a)
        # 每 token 的重算時間（32K 平均）：線性部分用 α、注意力部分用 β（G32−G0 是 0–32K 的平均注意力）
        t_tok_ms = a * G0 * 1e9 + b * (G32 - G0) * 1e9
        tok_per_s = 1e3 / t_tok_ms
        f_chunk_ms = CHUNK * t_tok_ms
        for interval_name, I in [("vllm_all_block", blk), ("sglang_8192", 8192), ("every_512", 512), ("every_4096", 4096)]:
            # 實際位元組（不含 vLLM 為了對齊頁大小補的 padding）
            frac = S / (S + I * kv_tok)
            wr_GiBps = (S / I + kv_tok) * tok_per_s / GiB
            # vLLM all 模式在 GPU 上的實際占用：每層每 block 一頁，mamba 狀態補齊到注意力頁大小
            frac_pad = frac_all if interval_name == "vllm_all_block" else float("nan")
            wr_pad = bytes_per_block_all / blk * tok_per_s / GiB if interval_name == "vllm_all_block" else float("nan")
            rows.append(row(src_run=src_run, model=name, n_mamba=n_m, n_attn=n_a, state_MiB=st_mib, kv_KiB_tok=kv_kib,
                            max_ctx=max_ctx, vllm_all_block_tokens=blk, interval=interval_name, interval_tokens=I,
                            state_over_chunk_kv=S / (CHUNK * kv_tok),
                            ckpt_frac_bytes=frac, ckpt_frac_bytes_vllm_padded=frac_pad,
                            write_GiBps_vllm_padded=wr_pad, t_tok_ms=t_tok_ms, prefill_tok_per_s=tok_per_s,
                            write_GiBps_needed=wr_GiBps, local_ssd_write_GiBps=tp["local"]["write_GiBps"],
                            nfs_write_GiBps=tp["nfs"]["write_GiBps"],
                            ratio_vs_local_ssd_write=wr_GiBps / tp["local"]["write_GiBps"],
                            ckpt_load_ms_cpu=S / (tp["cpu"]["read_GiBps"] * GiB) * 1e3,
                            ckpt_load_ms_vllm_cpu=S / (3.69 * GiB) * 1e3,
                            ckpt_load_ms_local_ssd=S / (tp["local"]["read_GiBps"] * GiB) * 1e3,
                            ckpt_load_ms_nfs=S / (tp["nfs"]["read_GiBps"] * GiB) * 1e3,
                            f_chunk_ms=f_chunk_ms,
                            exp_recompute_ms_uniform_branch=(I / 2) * t_tok_ms,
                            cpu_GiB_per_32k_session_ckpts=(32768 / I) * S / GiB,
                            kv_GiB_per_32k_session=32768 * kv_tok / GiB))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "d5_ssm.csv"), index=False)
    return df


# ----------------------------------------------------------------------------- (c) 滑動視窗
def part_swa(out, cfgdir, a, b, src_run):
    rows = []
    specs = []
    for fn, name in [("unsloth_gemma-3-12b-it.json", "Gemma-3-12B"), ("unsloth_gemma-3-27b-it.json", "Gemma-3-27B")]:
        c = json.load(open(os.path.join(cfgdir, fn)))
        c = c.get("text_config", c)          # Gemma-3 的多模態 config 把文字部分放在 text_config
        Lh, h, inter = c["num_hidden_layers"], c["hidden_size"], c["intermediate_size"]
        nq, nkv, hd, W, pat = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["sliding_window"], c["sliding_window_pattern"]
        n_glob = sum(1 for i in range(Lh) if (i + 1) % pat == 0)
        n_loc = Lh - n_glob
        per = h * nq * hd + 2 * h * nkv * hd + nq * hd * h + 3 * h * inter
        lin = 2 * Lh * per
        specs.append((name, n_glob, n_loc, 2 * nkv * hd * 2, W, lin, 2 * nq * (hd + hd)))
    for fn, name in [("openai_gpt-oss-20b.json", "gpt-oss-20b"), ("openai_gpt-oss-120b.json", "gpt-oss-120b")]:
        c = json.load(open(os.path.join(cfgdir, fn)))
        lt = c["layer_types"]
        n_glob, n_loc = lt.count("full_attention"), lt.count("sliding_attention")
        Lh, h = c["num_hidden_layers"], c["hidden_size"]
        nq, nkv, hd, W = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["sliding_window"]
        # 線性：注意力投影＋top-k 專家（每個專家 gate/up/down，intermediate_size）＋router
        attn_p = h * nq * hd + 2 * h * nkv * hd + nq * hd * h
        moe_p = c["num_experts_per_tok"] * 3 * h * c["intermediate_size"] + h * c["num_local_experts"]
        lin = 2 * Lh * (attn_p + moe_p)
        specs.append((name, n_glob, n_loc, 2 * nkv * hd * 2, W, lin, 2 * nq * (hd + hd)))
    litc_kv = {"Gemma-3-12B": (384, 64), "Gemma-3-27B": (496, 80), "gpt-oss-20b": (48, 24), "gpt-oss-120b": (72, 36)}
    litc_lin = {"Gemma-3-12B": 10.76, "Gemma-3-27B": 25.60, "gpt-oss-20b": 3.03, "gpt-oss-120b": 4.55}
    tp, bws = bw_list()
    for (name, ng, nl, kv_layer, W, lin, perkey_layer) in specs:
        kv_all = (ng + nl) * kv_layer
        kv_glob = ng * kv_layer
        assert (kv_all // KiB, kv_glob // KiB) == litc_kv[name], (name, kv_all, kv_glob)
        assert abs(lin / 2 / 1e9 - litc_lin[name]) < 0.05, (name, lin / 2 / 1e9)
        for ctx in (32768, 131072):
            n = ctx // CHUNK
            f = np.array([a * CHUNK * lin + b * (attn_chunk_flops(i, ng * perkey_layer) +
                                                 attn_chunk_flops(i, nl * perkey_layer, W)) for i in range(n)])
            bytes_all = ctx * kv_all
            bytes_need = ctx * kv_glob + min(ctx, W) * nl * kv_layer
            # 每個 chunk 要載入的位元組：只存需要的 → 全域層＋（落在最後 W 個 token 內的 chunk 才有）局部層
            last_w_chunks = math.ceil(W / CHUNK)
            per_chunk_need = np.array([CHUNK * kv_glob + (CHUNK * nl * kv_layer if i >= n - last_w_chunks else 0)
                                       for i in range(n)], dtype=float)
            for bname, bw in bws:
                ell_all = np.full(n, CHUNK * kv_all / (bw * GiB) * 1e3)
                ell_need = per_chunk_need / (bw * GiB) * 1e3
                Ta, ba = cake_time(f, ell_all)
                Tn, bn = cake_time(f, ell_need)
                rows.append(row(src_run=src_run, model=name, n_global=ng, n_local=nl, window=W, ctx=ctx,
                                kv_all_KiB_tok=kv_all / KiB, kv_global_KiB_tok=kv_glob / KiB,
                                GiB_all=bytes_all / GiB, GiB_needed=bytes_need / GiB, waste_frac=1 - bytes_need / bytes_all,
                                bw_name=bname, GiBps=bw, T_compute_ms=float(f.sum()),
                                T_load_all_ms=float(ell_all.sum()), T_load_need_ms=float(ell_need.sum()),
                                T_cake_all_ms=Ta, b_all=ba, T_cake_need_ms=Tn, b_need=bn,
                                cake_gain_need_vs_all=1 - Tn / Ta))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "d5_swa.csv"), index=False)
    return df


# ----------------------------------------------------------------------------- (d) P/D
def part_pd(out, f, src_run):
    """P/D：prefill 節點算 n 個 chunk，KV 經網路送到 decode 節點。
    phi = decode 端拿來重算的 GPU 速度比例（decode 節點本來在做 decode，不是閒著；phi=1 表示完全閒置，
    這時「decode 端全部自己算」＝不分離，結果是退化的，只當上限參考）。"""
    n = len(f)
    kv_chunk = CHUNK * 131072
    F = np.concatenate([[0.0], np.cumsum(f)])   # F[i+1] = chunk i 在 prefill 節點算完的時間
    rows = []
    for gbps in (2, 4, 6, 8, 10, 25, 50, 100, 200, 400):   # 每個 stream 實際分到的頻寬
        G = gbps * 1e9 / 8 / GiB
        ell = kv_chunk / (G * GiB) * 1e3
        T_noov = F[n] + n * ell
        s_ = 0.0
        for i in range(n):          # 逐 chunk 管線：chunk i 算完才能送，鏈路一次送一個
            s_ = max(F[i + 1], s_) + ell
        T_pipe = s_
        for phi in (1.0, 0.5, 0.25):
            fd = np.concatenate([[0.0], np.cumsum(f / phi)])   # decode 端重算 [0,b) 的完成時間
            best = None
            for bb in range(n + 1):
                s2 = 0.0
                for i in range(bb, n):
                    s2 = max(F[i + 1], s2) + ell
                T = max(fd[bb], s2)
                if best is None or T < best[0]:
                    best = (T, bb)
            T_cake, b_cake = best
            rows.append(row(src_run=src_run, model="Llama-3.1-8B", ctx=n * CHUNK, net_Gbps_per_stream=gbps, GiBps=G,
                            ell_ms=ell, prefill_ms=float(F[n]), T_no_overlap_ms=T_noov, T_chunk_pipe_ms=T_pipe,
                            phi_decode_gpu=phi, T_cake_split_ms=T_cake, b_cake=b_cake,
                            exposed_no_overlap_frac=(T_noov - F[n]) / T_noov,
                            exposed_chunk_pipe_frac=(T_pipe - F[n]) / T_pipe,
                            cake_gain_vs_chunk_pipe=1 - T_cake / T_pipe,
                            decode_recompute_ms=float(fd[b_cake])))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "d5_pd.csv"), index=False)
    return df


# ----------------------------------------------------------------------------- (e) MLA／MQA
# Lit-C 表 L61、L88、L92–94（KV KiB/token、GFLOP/token p→0、32K 平均、預測 b/n @3.69／@35.4）〔算術，Lit-C〕
LITC_MLA = [
    ("Llama-3.1-8B", 128, 14.0, 22.5, 0.31, 0.05),
    ("DeepSeek-V2-Lite", 30.4, 4.5, 9.0, 0.23, 0.03),
    ("DeepSeek-V3", 68.6, 71.4, 153.3, 0.05, 0.00),
    ("Kimi-K2", 68.6, 61.0, 102.0, 0.06, 0.00),
    ("Falcon-7B", 8, 13.3, 22.8, 0.03, 0.00),
    ("LongAlpaca-7B", 512, 13.0, 21.5, 0.61, 0.17),
]


def part_mla(out, a, b):
    rows = []
    n = 64
    for (name, kv_kib, G0, G32, bn37_litc, bn354_litc) in LITC_MLA:
        # 注意力 FLOPs 與位置成正比：attn(p)=k·p，0–32K 的平均 = k·16384 → k = (G32−G0)/16384
        k = (G32 - G0) * 1e9 / 16384
        f = [a * CHUNK * G0 * 1e9 + b * attn_chunk_flops(i, k) for i in range(n)]
        for bw, litc in ((0.331084, float("nan")), (3.69, bn37_litc), (35.4159, bn354_litc)):
            ell = CHUNK * kv_kib * KiB / (bw * GiB) * 1e3
            bb = write_boundary(n, f, ell)
            T, _ = cake_time(np.array(f), np.full(n, ell))
            rows.append(row(src_run="lit_C_table+" + CALIB_RUN, model=name, kv_KiB_tok=kv_kib, G0=G0, G32=G32,
                            GiBps=bw, ell_ms=ell, f0_ms=f[0], f63_ms=f[-1], b=bb, bn=bb / n, bn_litc=litc,
                            T_compute_ms=float(sum(f)), T_load_ms=n * ell, T_cake_ms=T,
                            cake_gain_vs_best=1 - T / min(sum(f), n * ell)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "d5_mla.csv"), index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg-vl", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "m8_directions"))
    args = ap.parse_args()
    assert RUN_ID, "要用 m7run 包起來（需要 RUN_ID）"
    os.makedirs(args.out, exist_ok=True)
    f, a, b, pred, _, _ = calib(os.path.join(RES7, "calib_c1.csv"))
    cal = part_calib(args.out, f, a, b, pred)
    print(cal[cal.item == "llama_bn"][["bw_name", "GiBps", "bn_meas", "bn_pred"]].to_string())
    print("alpha", a, "beta", b)
    src_vl = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(args.cfg_vl))))
    src_cfg = os.path.basename(os.path.dirname(os.path.abspath(args.cfg)))
    part_vlm(args.out, args.cfg_vl, a, b, src_vl)
    part_ssm(args.out, args.cfg, a, b, src_cfg + "+lit_C_table")
    part_swa(args.out, args.cfg, a, b, src_cfg)
    part_pd(args.out, f[:64], CALIB_RUN)
    part_mla(args.out, a, b)
    print("done")


if __name__ == "__main__":
    main()
