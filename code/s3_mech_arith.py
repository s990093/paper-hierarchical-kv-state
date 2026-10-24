#!/usr/bin/env python3
"""s3_mech_arith.py — 第三階段「新機制候選」用到的所有算術推估（10/24，使用者要 128K–256K、8B–14B 的有利設定）。

不跑 GPU、不讀結果檔：只把已經量到的常數（每個都註明出處）代進公式，印成 markdown 表。
文件 docs/stage3_design_20261024/mechanisms/ 裡標〔算術〕的數字全部來自這支的輸出（arith.md）。
這些都是推估，不是量測；量到之前一律當 NOT_MEASURED 的預測。

  python code/s3_mech_arith.py > docs/stage3_design_20261024/mechanisms/arith.md
  python code/s3_mech_arith.py --self-test
"""
from __future__ import annotations

import argparse
import math
import sys

GiB = 1024 ** 3

# ---- 已量到的常數（H200，origin/h200） ----
H2D_GIBPS = 53.55          # RUNLOG_H200.md:63，pinned H2D 1 GiB 中位數
HBM_GBPS = 4173.9          # 同上，HBM 裝置內複製
MATMUL_TFLOPS = 788.1      # 同上，BF16 matmul
VLLM_COLD_S = {8192: 0.209, 16384: 0.482, 32768: 1.224}   # RUNLOG_H200.md:148-150，vLLM 冷（重算），Llama
HARNESS_OVER_VLLM = 1.30   # origin/h200 commit 98598c5（fa3 後 harness÷vLLM）
F_SLOPE_MS = 0.210         # 同上，fa3 重算斜率 ms／塊（Llama，512-token 塊）
F_SUM64_MS = 1583.0        # 同上，Σ64 f（Llama fa3，32K＝64 塊）
BLOCK = 512

# ---- 慢層頻寬 ----
TIERS = {                  # GiB/s
    "CPU（H200 實測 H2D）": H2D_GIBPS,
    "本地 NVMe（MI300X tier_params，H200 md0 未量）": 6.981,   # RUNLOG_H200_P3.md:54
    "NFS（MI300X tier_params）": 0.331,                          # 同上
}

# ---- 模型（HF config.json；14B、NeMo 在 H200 沒有載入驗證過） ----
MODELS = {
    # 名稱: (層, KV heads, head_dim, 參數量 B, q heads)
    "Llama-3.1-8B": (32, 8, 128, 8.03, 32),
    "Qwen2.5-7B-1M": (28, 4, 128, 7.62, 28),
    "Mistral-NeMo-12B": (40, 8, 128, 12.25, 32),
    "Qwen2.5-14B-1M": (48, 8, 128, 14.77, 40),
}
CTX = {"128K": 131072, "256K": 262144}
MAXCTX = {"Llama-3.1-8B": 131072, "Qwen2.5-7B-1M": 262144, "Mistral-NeMo-12B": 131072, "Qwen2.5-14B-1M": 262144}

HBM_GIB = 141e9 / GiB      # H200 規格 141 GB
UTIL = 0.90                # vLLM gpu_memory_utilization 預設
ACT_RESERVE_GIB = 6.0      # 〔假設〕activation／CUDA graph 預留，沒量
INT4_RATIO = 3.2           # X4：單用 INT4 為 3.2×（02_observations O5）


def kv_bytes_per_token(m):
    L, kvh, hd, _, _ = MODELS[m]
    return 2 * L * kvh * hd * 2


def kv_gib(m, n):
    return kv_bytes_per_token(m) * n / GiB


def weights_gib(m):
    return MODELS[m][3] * 1e9 * 2 / GiB


def kv_budget_gib(m):
    return HBM_GIB * UTIL - weights_gib(m) - ACT_RESERVE_GIB


def f0_ms():
    # Σ_{i<64} (f0 + α i) = 64 f0 + α·2016
    return (F_SUM64_MS - F_SLOPE_MS * 63 * 64 / 2) / 64


def recompute_prefix_ms(nb, density=1.0):
    """harness 的逐塊重算時間（Llama fa3）：前 nb 塊，注意力那一項乘 density（稀疏 prefill 的假設）。"""
    return f0_ms() * nb + density * F_SLOPE_MS * nb * (nb - 1) / 2


def cake_meet(nblocks, block_mib, gibps, density=1.0):
    """Cake：前段重算、後段載入，兩邊同時跑；回傳 (會合點 b, 時間 s)。"""
    per_block_s = block_mib / 1024 / gibps
    best = None
    for b in range(nblocks + 1):
        t = max(recompute_prefix_ms(b, density) / 1000, (nblocks - b) * per_block_s)
        if best is None or t < best[1]:
            best = (b, t)
    return best


def vllm_cold_fit():
    """用 16K、32K 兩點擬合 t＝a·n＋c·n²（vLLM 冷 prefill，Llama）。"""
    n1, n2 = 16384, 32768
    t1, t2 = VLLM_COLD_S[n1], VLLM_COLD_S[n2]
    c = (t2 - 2 * t1) / (n2 * n2 - 2 * n1 * n1)
    a = (t1 - c * n1 * n1) / n1
    return a, c


def fmt(x, nd=2):
    return f"{x:.{nd}f}"


def section_kv():
    print("## 1. 每個 session 的 KV 大小與 HBM 放得下幾個（H200）\n")
    print(f"HBM 規格 141 GB＝{fmt(HBM_GIB, 1)} GiB；×{UTIL}；activation 預留 {ACT_RESERVE_GIB} GiB〔假設〕。\n")
    print("| 模型 | KV／token | 權重 GiB | KV 預算 GiB | 長度 | 一個 session 的 KV GiB | 完整放得下 | 16× 粗底放得下 | 32× 粗底放得下 |")
    print("|:--|--:|--:|--:|:--|--:|--:|--:|--:|")
    for m in MODELS:
        for c, n in CTX.items():
            if n > MAXCTX[m]:
                continue
            s = kv_gib(m, n)
            bud = kv_budget_gib(m)
            print(f"| {m} | {kv_bytes_per_token(m) // 1024} KiB | {fmt(weights_gib(m), 1)} | {fmt(bud, 1)} | {c} | {fmt(s, 1)} | "
                  f"{int(bud // s)} | {int(bud // (s / 16))} | {int(bud // (s / 32))} |")
    print()


def section_restore():
    print("## 2. 回來那一輪要搬多少、搬多久（只算傳輸，不含 prefill 與反量化）\n")
    print("ρ＝升級的塊比例；refine＝只把挑中的塊升級成 INT4（3.2×）。粗底常駐 HBM 時，只有 refine 要過 PCIe／慢層。\n")
    print("| 模型・長度 | 層 | 全部 BF16 (s) | 全部 INT4 (s) | 粗底常駐＋ρ=1/4 INT4 (s) | 粗底常駐＋ρ=1/8 INT4 (s) | 比全部 BF16 快 |")
    print("|:--|:--|--:|--:|--:|--:|--:|")
    for m, c in (("Llama-3.1-8B", "128K"), ("Qwen2.5-7B-1M", "256K"), ("Qwen2.5-14B-1M", "128K"), ("Qwen2.5-14B-1M", "256K")):
        s = kv_gib(m, CTX[c])
        for t, bw in TIERS.items():
            full = s / bw
            i4 = s / INT4_RATIO / bw
            r4 = s / INT4_RATIO / 4 / bw
            r8 = s / INT4_RATIO / 8 / bw
            print(f"| {m}・{c} | {t} | {fmt(full)} | {fmt(i4)} | {fmt(r4, 3)} | {fmt(r8, 3)} | {fmt(full / r4, 1)}× |")
    print()


def section_multipath():
    print("## 3. 借鄰卡 PCIe 的多路還原（理想值：不計主機端瓶頸、NVLink 轉送不計時間）\n")
    print("每條路徑 = 一張卡自己的 PCIe（H200 實測 53.55 GiB/s）。HGX 上兩張卡可能共用一個 PCIe switch 的上行（拓撲 NOT_MEASURED），"
          "所以 k 是「獨立上行」的數量，不是卡數。Beluga（2511.20172）量到跨 switch 的 P2P 只有 23–33 GB/s，是上限的警告。\n")
    print("| 模型・長度 | KV GiB | k＝1 (s) | k＝2 (s) | k＝4 (s) |")
    print("|:--|--:|--:|--:|--:|")
    for m, c in (("Llama-3.1-8B", "128K"), ("Qwen2.5-14B-1M", "128K"), ("Qwen2.5-14B-1M", "256K")):
        s = kv_gib(m, CTX[c])
        print(f"| {m}・{c} | {fmt(s, 1)} | " + " | ".join(fmt(s / (H2D_GIBPS * k), 3) for k in (1, 2, 4)) + " |")
    print()


def section_recompute():
    print("## 4. 重算有多貴：128K 以上，注意力那一項主導\n")
    a, c = vllm_cold_fit()
    print(f"vLLM 冷 prefill（Llama，H200 實測 16K {VLLM_COLD_S[16384]} s、32K {VLLM_COLD_S[32768]} s）擬合 t＝a·n＋c·n²："
          f"a＝{a:.3e} s/token、c＝{c:.3e} s/token²。\n")
    print("| 長度 | vLLM 冷 prefill 外推 (s) | 其中 n² 項佔比 | 從 CPU 載入 BF16 (s) | 重算÷載入 |")
    print("|:--|--:|--:|--:|--:|")
    for n in (32768, 65536, 131072):
        t = a * n + c * n * n
        load = kv_gib("Llama-3.1-8B", n) / H2D_GIBPS
        print(f"| {n // 1024}K | {fmt(t)} | {fmt(c * n * n / t * 100, 0)}% | {fmt(load, 3)} | {fmt(t / load, 1)}× |")
    print()
    print(f"harness 的逐塊重算（fa3 校準）：f(i)＝f₀＋α·i，α＝{F_SLOPE_MS} ms／塊，由 Σ64＝{F_SUM64_MS:.0f} ms 解出 f₀＝{fmt(f0_ms(), 1)} ms。"
          f"第 i 塊注意力那一項佔比＝α·i÷f(i)：i＝63 → {fmt(F_SLOPE_MS * 63 / (f0_ms() + F_SLOPE_MS * 63) * 100, 0)}%，"
          f"i＝251（128K 最後一塊）→ {fmt(F_SLOPE_MS * 251 / (f0_ms() + F_SLOPE_MS * 251) * 100, 0)}%。\n")


def section_sparse_recompute():
    print("## 5. 近似重算（稀疏 prefill）會讓 Cake 的會合點往後移多少（Llama 128K＝252 塊，每塊 64 MiB）\n")
    print("density＝重算時注意力只算原本的幾成（MInference／FlexPrefill 類，品質未量）。慢層頻寬用 H200 calib-f 的 Llama 限速層"
          "（RUNLOG_H200.md:879：b16 8.83、b32 2.49、b48 0.72 GiB/s）與 NFS 0.331 GiB/s。\n")
    print("| 層 | 精確重算：會合點 b／時間 (s) | density 0.2 | density 0.1 | 0.1 比精確快 |")
    print("|:--|:--|:--|:--|--:|")
    for t, bw in (("b16", 8.83), ("b32", 2.49), ("b48", 0.72), ("NFS", 0.331)):
        e = cake_meet(252, 64, bw, 1.0)
        d2 = cake_meet(252, 64, bw, 0.2)
        d1 = cake_meet(252, 64, bw, 0.1)
        print(f"| {t} | {e[0]}／{fmt(e[1])} | {d2[0]}／{fmt(d2[1])} | {d1[0]}／{fmt(d1[1])} | {fmt((1 - d1[1] / e[1]) * 100, 0)}% |")
    print("\n對照（實測，harness E0 Cake 中位數，GPU 閒）：Llama 128K b16 1.459、b32 3.917、b48 8.433 s（`results/m10_longctx_h200/analysis/tables_longctx.md` §5）。"
          "算術用的是理想的兩線並行，比實測快，只拿來看「會合點往後移」的方向與幅度。\n")


def section_question_prefill():
    print("## 6. 回來那一輪：新問題在粗底上 prefill 一次要多少算力（挑塊用）\n")
    print(f"注意力 FLOPs ≈ 4·n_q·n_ctx·(q heads·128)·層數；線性層 ≈ 2·參數·n_q。效率假設 50% 的實測 matmul {MATMUL_TFLOPS} TFLOPS〔假設〕。\n")
    print("| 模型・長度 | n_q | 注意力 TFLOP | 線性 TFLOP | 時間 (s) |")
    print("|:--|--:|--:|--:|--:|")
    for m, c in (("Llama-3.1-8B", "128K"), ("Qwen2.5-7B-1M", "256K"), ("Qwen2.5-14B-1M", "256K")):
        L, _, hd, p, qh = MODELS[m]
        for nq in (128, 512):
            att = 4 * nq * CTX[c] * qh * hd * L / 1e12
            lin = 2 * p * 1e9 * nq / 1e12
            t = (att + lin) / (MATMUL_TFLOPS * 0.5)
            print(f"| {m}・{c} | {nq} | {fmt(att, 1)} | {fmt(lin, 1)} | {fmt(t, 3)} |")
    print()


def section_cpu_attention():
    print("## 7. 被否決的：在 CPU 上直接對 CPU 層的 KV 算新問題的注意力（不搬 KV）\n")
    peak = 16 * 3.1e9 * 64   # 〔假設〕每個 GPU job 16 核 EPYC 9554、AVX-512 BF16 每週期 64 FLOP，沒量
    print(f"每個 GPU job 只有 16 個 EPYC 9554 核（H200_SLURM.md），沒有 AMX；峰值假設 {peak / 1e12:.1f} TFLOPS〔假設〕。\n")
    print("| 模型・長度 | 搬 KV 過 PCIe (s) | CPU 每個問題 token 的注意力 (s) | 打平的問題長度 |")
    print("|:--|--:|--:|--:|")
    for m, c in (("Qwen2.5-7B-1M", "256K"), ("Qwen2.5-14B-1M", "256K")):
        L, _, hd, _, qh = MODELS[m]
        per_tok = 4 * CTX[c] * qh * hd * L / peak
        load = kv_gib(m, CTX[c]) / H2D_GIBPS
        print(f"| {m}・{c} | {fmt(load, 3)} | {fmt(per_tok, 4)} | {math.floor(load / per_tok)} token |")
    print()


def section_decode():
    print("## 8. 長 context 的 decode 是 HBM 頻寬型（batch 1 的下界）\n")
    print(f"每步至少讀一次權重＋全部 KV；HBM 用實測裝置內複製 {HBM_GBPS} GB/s。\n")
    print("| 模型・長度 | 每步讀 GB | TPOT 下界 (ms) | 每步的 FLOPs÷位元組 |")
    print("|:--|--:|--:|--:|")
    for m, c in (("Llama-3.1-8B", "128K"), ("Qwen2.5-7B-1M", "256K"), ("Qwen2.5-14B-1M", "256K")):
        L, _, hd, p, qh = MODELS[m]
        byt = p * 1e9 * 2 + kv_bytes_per_token(m) * CTX[c]
        flops = 2 * p * 1e9 + 4 * CTX[c] * qh * hd * L
        print(f"| {m}・{c} | {fmt(byt / 1e9, 1)} | {fmt(byt / (HBM_GBPS * 1e9) * 1000, 1)} | {fmt(flops / byt, 1)} |")
    print(f"\nH200 的 matmul÷HBM 比（實測）＝{fmt(MATMUL_TFLOPS * 1e12 / (HBM_GBPS * 1e9), 0)} FLOP／byte：decode 遠低於它，tensor core 大多閒著。\n")


def prefill_s(m, c, nq):
    L, _, hd, p, qh = MODELS[m]
    return (4 * nq * CTX[c] * qh * hd * L + 2 * p * 1e9 * nq) / (MATMUL_TFLOPS * 1e12 * 0.5)


def section_resume_ttft():
    print("## 9. 回來那一輪的 TTFT：全部載入 vs 粗底常駐（兩趟 prefill）vs 粗底常駐＋逐層流水（一趟）\n")
    print("全部載入＝搬完 BF16 再 prefill 問題；兩趟＝粗底上 prefill 挑塊 → 搬 ρ=1/4 的 INT4 → 再 prefill 一次；"
          "逐層流水＝每一層先用下層已精修的結果算 q、挑塊、搬這一層的精修，和這一層的計算重疊，"
          "時間取 max(prefill, 精修傳輸)＋一層的傳輸。prefill 時間用第 6 節的公式；反量化不計。\n")
    print("| 模型・長度 | 層 | n_q | 全部載入 (s) | 兩趟 (s) | 逐層流水 (s) | 逐層流水比全部載入快 |")
    print("|:--|:--|--:|--:|--:|--:|--:|")
    for m, c in (("Llama-3.1-8B", "128K"), ("Qwen2.5-14B-1M", "256K")):
        L = MODELS[m][0]
        s = kv_gib(m, CTX[c])
        for t, bw in TIERS.items():
            for nq in (128, 512):
                pf = prefill_s(m, c, nq)
                full = s / bw + pf
                ref = s / INT4_RATIO / 4 / bw
                two = pf + ref + pf
                pipe = max(pf, ref) + ref / L
                print(f"| {m}・{c} | {t.split('（')[0]} | {nq} | {fmt(full, 3)} | {fmt(two, 3)} | {fmt(pipe, 3)} | {fmt(full / pipe, 1)}× |")
    print()


def self_test():
    assert kv_bytes_per_token("Llama-3.1-8B") == 131072
    assert kv_bytes_per_token("Qwen2.5-7B-1M") == 57344
    assert kv_bytes_per_token("Qwen2.5-14B-1M") == 196608
    assert abs(f0_ms() - 18.1) < 0.1
    a, c = vllm_cold_fit()
    assert abs(a * 16384 + c * 16384 ** 2 - 0.482) < 1e-9
    assert abs(a * 32768 + c * 32768 ** 2 - 1.224) < 1e-9
    b, t = cake_meet(252, 64, 0.72, 1.0)
    assert 0 < b < 252 and t > 0
    b2, t2 = cake_meet(252, 64, 0.72, 0.1)
    assert b2 >= b and t2 <= t
    print("self-test ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    print("# 新機制候選的算術推估（〔算術〕，不是量測）\n")
    print("> 產生：`python code/s3_mech_arith.py > docs/stage3_design_20261024/mechanisms/arith.md`。常數出處寫在程式註解。\n")
    section_kv()
    section_restore()
    section_multipath()
    section_recompute()
    section_sparse_recompute()
    section_question_prefill()
    section_cpu_attention()
    section_decode()
    section_resume_ttft()


if __name__ == "__main__":
    sys.exit(main())
