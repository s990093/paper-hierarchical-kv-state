"""m10_kv_stats.py — D0 的 K／V 分布量測（03_plan_h200.md §1「先量 K／V 的分布」）＋真實 KV 上的 codec 重建誤差

為什麼：外部量測說 Qwen2.5-7B 的 max|K| 約 423（< 448），而且少數 channel 有很大的固定偏移（k_proj bias）〔P04 §4〕。
這決定 FP8 失敗是「溢位」還是「偏移 × 只有 3 位尾數」。

做法：用 m10_tasks 產生 32K 的 prompt（T4 兩題＝真實文字、T1、T3、T5 各一題），chunk 化 prefill（和還原器同一條路徑），
在 KV 緩衝上直接量 RoPE 之後的 K、V：
  每（層, KV head, channel）：max|K|、平均 K、K 的標準差（對 32K 個 token）
  每（層, KV head）：max|V|、V 的標準差
  每（prompt, block, 格式）：m10_codec 的重建相對誤差（K、V 分開）；FP8 格式另記「落在 e4m3 次正規區」的比例
只量 BF16 的 KV 和 codec 的重建誤差，不跑任何題目、不看任何答對率（事先登記凍結之前可以做，03 §1 的 D1 用重建誤差挑 scale 粒度）。

輸出（--out 目錄）：
  kv_head_<model>.csv      每（層, head）的摘要（進 git）
  kv_topchan_<model>.csv   |平均 K| 最大的 64 個 channel（進 git）
  codec_err_<model>.csv    每（prompt, 格式）的誤差摘要（進 git）
  summary_<model>.json
  完整的逐 channel、逐 block 檔放 --raw 目錄（runs，不進 git）
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m10_tasks as T  # noqa: E402
from m10_codec import FORMATS, FP8, decode, encode, rel_err, rel_err_centered  # noqa: E402

FP8_MIN_NORMAL = 2.0 ** -6
PROMPTS = [("T4", "front", "back", 0), ("T4", "mid", "front", 1), ("T1", "back", "mid", 0),
           ("T3", "mid", "back", 0), ("T5", None, None, 0)]


def main(a):
    os.environ["M7_MODEL_GLOB"] = T.MODELS[a.model]
    from m7_model import CHUNK, KVModel     # M7_MODEL_GLOB 要在 import 前設好
    tok = T.load_tokenizer(a.model)
    n = T.N_BLOCKS
    km = KVModel(max_len=n * CHUNK)
    L, H, D = km.L, km.H, km.D
    rid = os.environ.get("RUN_ID", "adhoc")
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.raw, exist_ok=True)
    # 跨 prompt 累積
    kmax = torch.zeros(L, H, D, device=km.device)
    vmax = torch.zeros(L, H, device=km.device)
    kmeans, kstds, vstds = [], [], []
    err_rows, blk_rows = [], []
    for task, b1, b2, seed in PROMPTS:
        it = T.make_item(task, b1, b2, seed, tok)
        ids = torch.tensor(it["context_ids"], device=km.device)
        t0 = time.time()
        km.prefill_chunked(ids)
        torch.cuda.synchronize()
        K = km.kv[:, 0, :, :n * CHUNK].float()          # [L,H,T,D]
        V = km.kv[:, 1, :, :n * CHUNK].float()
        kmax = torch.maximum(kmax, K.abs().amax(dim=2))
        vmax = torch.maximum(vmax, V.abs().amax(dim=(2, 3)))
        kmeans.append(K.mean(dim=2))
        kstds.append(K.std(dim=2))
        vstds.append(V.std(dim=(2, 3)))
        del K, V
        # codec 重建誤差：每個 block、每種格式
        for fmt in FORMATS:
            if fmt == "bf16":
                continue
            ks, vs, subn, kc, vc = [], [], [], [], []
            for i in range(n):
                x = km.chunk_view(i)
                e = encode(x, fmt)
                y = decode(e)
                er, ec = rel_err(x, y), rel_err_centered(x, y)
                ks.append(er["K"])
                vs.append(er["V"])
                kc.append(ec["K"])
                vc.append(ec["V"])
                if fmt.startswith("fp8"):
                    q = e.parts["q"].view(FP8).float().abs()
                    subn.append(float(((q > 0) & (q < FP8_MIN_NORMAL)).float().mean()))
                blk_rows.append(dict(run_id=rid, model=a.model, item_id=it["item_id"], fmt=fmt, block=i,
                                     rel_err_K=er["K"], rel_err_V=er["V"], cen_err_K=ec["K"], cen_err_V=ec["V"],
                                     frac_subnormal=subn[-1] if subn else ""))
            kt, vt = torch.tensor(ks), torch.tensor(vs)
            err_rows.append(dict(run_id=rid, model=a.model, item_id=it["item_id"], task=task, fmt=fmt,
                                 nbytes_ratio=round(e.nbytes / (km.chunk_bytes), 5),
                                 relK_mean=float(kt.mean()), relK_max=float(kt.max()), relK_block0=ks[0],
                                 relV_mean=float(vt.mean()), relV_max=float(vt.max()), relV_block0=vs[0],
                                 cenK_mean=sum(kc) / n, cenK_max=max(kc), cenV_mean=sum(vc) / n,
                                 frac_subnormal_mean=(sum(subn) / len(subn)) if subn else ""))
        print(f"{it['item_id']}: prefill+codec {time.time() - t0:.1f}s", flush=True)

    kmean = torch.stack(kmeans)                         # [P,L,H,D]
    kstd = torch.stack(kstds)
    km_avg, ks_avg = kmean.mean(0), kstd.mean(0)
    offset = km_avg.abs() / ks_avg.clamp_min(1e-6)      # |平均| ÷ 標準差：大＝固定偏移
    # 逐 channel 完整檔（runs）
    with open(os.path.join(a.raw, f"kv_chan_{a.model}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "model", "layer", "head", "chan", "maxabs_K", "mean_K", "std_K",
                    "mean_K_min_over_prompts", "mean_K_max_over_prompts", "absmean_over_std"])
        for l in range(L):
            for h in range(H):
                for d in range(D):
                    w.writerow([rid, a.model, l, h, d, f"{kmax[l, h, d]:.4f}", f"{km_avg[l, h, d]:.4f}",
                                f"{ks_avg[l, h, d]:.4f}", f"{kmean[:, l, h, d].min():.4f}",
                                f"{kmean[:, l, h, d].max():.4f}", f"{offset[l, h, d]:.3f}"])
    with open(os.path.join(a.raw, f"codec_err_blocks_{a.model}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(blk_rows[0]))
        w.writeheader()
        w.writerows(blk_rows)
    # 每（層, head）摘要（git）
    with open(os.path.join(a.out, f"kv_head_{a.model}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "model", "layer", "head", "maxabs_K", "maxabs_V", "max_absmean_K", "median_std_K",
                    "n_chan_offset_gt3", "n_chan_absmean_gt_maxabs_half"])
        for l in range(L):
            for h in range(H):
                w.writerow([rid, a.model, l, h, f"{kmax[l, h].max():.3f}", f"{vmax[l, h]:.3f}",
                            f"{km_avg[l, h].abs().max():.3f}", f"{ks_avg[l, h].median():.4f}",
                            int((offset[l, h] > 3).sum()),
                            int((km_avg[l, h].abs() > 0.5 * kmax[l, h].max()).sum())])
    flat = [(float(km_avg[l, h, d].abs()), l, h, d) for l in range(L) for h in range(H) for d in range(D)]
    flat.sort(reverse=True)
    with open(os.path.join(a.out, f"kv_topchan_{a.model}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "model", "rank", "layer", "head", "chan", "absmean_K", "std_K", "maxabs_K",
                    "absmean_over_std"])
        for r, (m, l, h, d) in enumerate(flat[:64]):
            w.writerow([rid, a.model, r, l, h, d, f"{m:.3f}", f"{ks_avg[l, h, d]:.4f}", f"{kmax[l, h, d]:.3f}",
                        f"{offset[l, h, d]:.2f}"])
    with open(os.path.join(a.out, f"codec_err_{a.model}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(err_rows[0]))
        w.writeheader()
        w.writerows(err_rows)
    by_fmt = {}
    for r in err_rows:
        by_fmt.setdefault(r["fmt"], []).append(r)
    summ = {
        "run_id": rid, "model": a.model, "model_path": km.cfg._name_or_path, "L": L, "H": H, "D": D,
        "prompts": [f"{t}_{b1}-{b2}_s{s}" for t, b1, b2, s in PROMPTS], "tokens_per_prompt": n * CHUNK,
        "maxabs_K_overall": float(kmax.max()), "maxabs_V_overall": float(vmax.max()),
        "maxabs_K_over_448": float(kmax.max()) / 448.0,
        "n_chan_total": L * H * D, "n_chan_offset_gt3": int((offset > 3).sum()),
        "n_chan_offset_gt10": int((offset > 10).sum()),
        "max_absmean_K": float(km_avg.abs().max()),
        "codec_relerr_mean_over_prompts": {f: {"K": sum(r["relK_mean"] for r in v) / len(v),
                                               "V": sum(r["relV_mean"] for r in v) / len(v),
                                               "K_worst_block": max(r["relK_max"] for r in v),
                                               "K_centered": sum(r["cenK_mean"] for r in v) / len(v),
                                               "K_centered_worst_block": max(r["cenK_max"] for r in v),
                                               "V_centered": sum(r["cenV_mean"] for r in v) / len(v)}
                                           for f, v in by_fmt.items()},
        "torch": torch.__version__, "hip": torch.version.hip, "cuda": torch.version.cuda,
        "note": "BF16 KV 與 codec 重建誤差；沒有任何答對率。MI300X 彩排，不進判定。",
    }
    json.dump(summ, open(os.path.join(a.out, f"summary_kv_{a.model}.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summ.items() if k != "codec_relerr_mean_over_prompts"}, ensure_ascii=False))
    for f, v in summ["codec_relerr_mean_over_prompts"].items():
        print(f"{f:14s} K {v['K']:.4f} (worst block {v['K_worst_block']:.4f})  V {v['V']:.4f}  "
              f"| 扣平均後 K {v['K_centered']:.4f} (worst {v['K_centered_worst_block']:.4f})  V {v['V_centered']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(T.MODELS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", required=True, help="完整逐 channel、逐 block 的檔（放 runs 目錄）")
    main(ap.parse_args())
