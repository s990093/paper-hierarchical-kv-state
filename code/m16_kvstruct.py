#!/usr/bin/env python3
"""m16_kvstruct.py — 第三階段 K1：真實 KV 的結構（低秩、跨層相似、相鄰 token 相似）。只用 numpy＋標準函式庫，不碰 torch／GPU。

為什麼：「低秩／降維」（Palu、LoRC、Eigen Attention 一類）與「token 合併／跨層共用」（MiniCache、KVMerger 一類）兩個新方向的
第一步——先量真的 KV 有沒有那種結構、有多少。資料＝m16_s1_kvdump.py 的 dump（GPU 窗口跑）；這支只讀檔。

  python code/m16_kvstruct.py --self-test
  python code/m16_kvstruct.py ropecheck --dump <dir> [<dir> ...]
  python code/m16_kvstruct.py analyze --dump <dir> [<dir> ...] --out <outdir> [--layers 0 1 2] [--heads 0 1] [--max-blocks 32]
        [--calib-tokens 4096] [--ranks 8 16 32 64 96] [--thresholds 0.75 0.8 0.9 0.95 0.99] [--no-joint] [--no-pre-rope] [--no-centered]
        [--spectra <path.npz>] [--max-mem-gb 16] [--skip-first-tokens 0]

執行規矩（共用機器、第二階段在 GPU 上量時間）：`OMP_NUM_THREADS=4 nice -n 19 taskset -c 0-3 python ...`；
沒設 OMP_NUM_THREADS 時這支自己設成 4（在 import numpy 之前）。開頭與結尾都檢查 torch／vllm／transformers 沒有被 import。

## dump 的格式（10/20 讀原始碼確認；行號是 10/20 的 HEAD）
- `kv_bf16.u16`＝[n_blocks][L'][2][H][chunk][D] 的 BF16 位元（uint16 little-endian）。L'＝meta.layers（128K 只存部分層）；
  [2] 的 0＝K、1＝V；chunk＝512。來源：m16_s1_kvdump.py 的 `m.chunk_view(i).index_select(0, lidx)`（[L,2,H,512,D]，
  m7_model.py:59 的 kv 緩衝＝[L, 2, H, max_len, D]），依 block 順序串接。
- 位置：block i 的第 t 個 token＝i·512＋t（m7_model.py:144–149 `prefill_chunked(start=0)`）。
- **K 是 RoPE 之後的**：m7_model.py:105 `q, k = self.apply_rope(q, k, cos, sin)` 之後，:106 才 `self.kv[li, 0].copy_(k[0])`。
  V 沒有 RoPE。
- 10/20 起 kvdump 在 meta 補了 `pos_start`、`k_rope`、`rope`（實際用的 inv_freq）、`rope_probe`（第 0 層 K 在 RoPE 前／後的
  幾個位置）。舊 dump 沒有這些欄位時：位置照上面推定、inv_freq 由 config.json 的公式算，**並且一定要通過資料自身的核對**（下面）。

## RoPE（逆旋轉用；照 transformers 5.17.0 的原始碼，tiara venv，只讀原始碼、不 import）
- inv_freq（預設，Qwen2.5）：models/qwen2/modeling_qwen2.py:86、models/llama/modeling_llama.py:108
  `inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))`（float32）。
- inv_freq（llama3，Llama-3.1）：modeling_rope_utils.py:642–660 `_compute_llama3_parameters`：
  :642 同上；:649–650 low/high_freq_wavelen＝old_ctx／low（high）_freq_factor；:652 `wavelen = 2*math.pi / inv_freq`；
  :655 wavelen > low_freq_wavelen → inv_freq / factor；:657 `smooth = (old_ctx / wavelen - low) / (high - low)`；
  :658 `smoothed = (1 - smooth) * inv_freq_llama / factor + smooth * inv_freq_llama`；:659–660 介於兩者之間的用 smoothed。
  注意 torch 的 `scalar / tensor` 是 `tensor.reciprocal() * scalar`（torch/_tensor.py:1119–1122），這裡照樣模擬。
  attention_factor＝1.0（:639）。
- cos／sin：modeling_llama.py:122–127（qwen2 :97–102）——`freqs = inv_freq @ pos` 在 float32（K=1 的 matmul＝一次捨入），
  `emb = cat(freqs, freqs)`，`cos = emb.cos() * attention_scaling`，最後 `.to(x.dtype)`＝**BF16**。
- 套用：modeling_llama.py:130–134 `rotate_half(x) = cat(-x2, x1)`（x1＝前半 D/2、x2＝後半；**第 j 維和第 j+D/2 維配對**，
  不是相鄰兩維）；:159 `k_embed = (k * cos) + (rotate_half(k) * sin)`，每一步都在 BF16（float32 算、捨入回 BF16）。
  KVModel 對 Qwen 也用 Llama 的 apply_rotary_pos_emb（m7_model.py:42），兩者的式子相同（qwen2 :105–134）。
- 逆旋轉：每一對 (x1_j, x2_j) 是 [[c, −s], [s, c]]，用同一組 BF16 的 c、s 做精確的矩陣反（除以 c²＋s²，float64）。
  RoPE 在實數上是正交旋轉、可精確逆；**實際的 dump 有 BF16 捨入**（每個值相對誤差 ≤ 2^-8），所以逆回來的 K 和真正的
  RoPE 前 K 差在 BF16 捨入的量級（自測量給你看），不是逐位元相同。

## RoPE 核對（規則 6：用資料自身驗算，不看欄位名推斷；不過就停，規則 2）
1. 探針（meta.rope_probe，kvdump 10/20 起才有）：
   (a) dump 第 0 層 K 在探針位置＝探針的 RoPE 後 K（逐位元，容許 ≤ 2 ulp）→ 核對排列、位置、K 是 RoPE 後；
   (b) 這支模擬的 BF16 cos／sin＝torch 算的（≤ 1 ulp）→ 核對 inv_freq、float32 乘法、cat(freqs, freqs)；
   (c) 這支模擬的 BF16 前向旋轉（探針的 RoPE 前 K）＝探針的 RoPE 後 K（≤ 1 ulp）→ 核對 rotate_half 的配對與正負號；
   (d) 這支的逆旋轉（dump）≈ 探針的 RoPE 前 K（每個向量的相對誤差 ≤ 0.02）→ 核對分析實際走的路徑。
2. 資料自身（每個 dump 都做，舊 dump 沒探針時是唯一的門檻）：假設 RoPE 前的 K 有不隨位置轉的平均方向（Qwen2 的 k_proj 有 bias，
   modeling_qwen2.py:190；Llama 沒有 bias，靠的是文獻〔KIVI 等〕觀察到的 K 固定大值通道——**這是假設**，不成立時結果是
   INCONCLUSIVE），RoPE 後的平均會被轉散。M＝‖平均 token‖²／平均‖token‖²（逆旋轉後）。
   正確的逆旋轉應該讓 M 最大：對照「不逆轉」「轉反方向」「相鄰兩維配對」「θ×2、θ×½」「llama3 不縮放」「inv_freq 捨成 BF16」。
   PASS＝期望的 M ≥ 每個對照 −0.001，且比不逆轉高 ≥ 0.005；高不到 0.005 → INCONCLUSIVE（資料裡沒有夠大的平均方向，無法判斷）。
   有探針且探針過了，這一項只當旁證（不過會寫進 warnings）；沒有探針時，這一項不是 PASS → 停（exit 2）。
   要跳過 RoPE 前的量測、只做其餘的 → `--no-pre-rope`。

## 量什麼（每層、每個 KV head；K、V 分開；K 另算 RoPE 前）
1. 低秩（Gram 矩陣串流累加，記憶體與 token 數無關）：
   - 奇異值譜（＝Gram 的特徵值），rank r 保留的能量比 Σ_{i≤r}σ²／Σσ²、相對重建誤差 ‖X−X_r‖_F／‖X‖_F＝√(1−能量)（Eckart–Young）；
     另算「先減平均再降秩」（centered；平均向量另存，誤差仍除以 ‖X‖_F）；有效秩 r90／r95／r99、stable rank、平均方向的能量比、
     單一 token 佔的最大能量比（attention sink 會不會主導）。
   - **校準→測試**：用前 `--calib-tokens` 個 token 的基底（uncentered：前 r 個特徵向量；centered：校準平均＋校準共變異的前 r 個），
     套到其後的 token，誤差除以測試 token 的 ‖X‖_F；同時給測試集自己的最佳（oracle）。兩者的差＝基底的泛化損失。
   - `head=all`：整層 H 個 head 串成 H·D 維一起降秩，rank＝r·H（和逐 head 同參數量）。這是這份資料上「同參數量的任何線性投影」
     （含逐 head、分組 head）誤差的下限，不是某個方法的實測。
   - 壓縮比（參考，不是量測）：rank r 的係數＝r/D 的位元組（BF16 係數；基底 r·D 另計，T ≫ D 時可忽略）。
2. 跨層相似（MiniCache 的量法）：同一 token、同一 head，層 l 與 l+1 的 K（或 V）先各自除以長度（方向正規化）再內積＝cosine；
   另算整層 H 個 head 串起來的 cosine（`head=all`）。**K 的跨層 cosine 在 RoPE 前後相同**（同一位置、同一個正交旋轉；自測有驗），
   所以只算一次（標成 K）；另用 RoPE 前的 K 再算一次當旁證（最大差寫進 summary）。只有 l 與 l+1 都在 dump 裡才算
   （128K 的 dump 只存部分層 → 沒有相鄰層，寫 NOT_MEASURED）。
3. 相鄰 token 相似：同一層同一 head，token t 與 t+1 的 cosine（K_post、K_pre、V）；block 邊界只在兩個 block 都讀到時才算。
4. 中心化（第二輪，`--no-centered` 關掉）：先減掉第一輪算出的每（層, head）對 token 的平均向量（K 用 RoPE 前的），再算 2、3。
   K 有固定的大通道（Qwen2 另有 k_proj bias），未中心化的 cosine 可能只是「大家都指向平均方向」；中心化版看扣掉之後還像不像。
   cosine 的分布：平均、標準差、角距離 arccos/π 的平均（MiniCache 的 d）、分位數 p05–p95（0.001 寬的直方圖內插，誤差 ≤ 0.001）、
   ≥ 門檻的比例（逐值精確比較，不經直方圖；預設 0.75＝KVMerger 的門檻、0.8、0.9、0.95、0.99）。長度為 0 的向量不算，另記 `n_zero`。

## 輸出（--out）
`lowrank.csv`（每 dump × 層 × head × 種類 × rank 一列）、`lowrank_eff.csv`（有效秩等）、`xlayer.csv`、`adjtok.csv`、
`lowrank.json`／`xlayer.json`／`adjtok.json`（彙總）、`rope_check.json`、`summary.json`（設定、時間、記憶體）、`summary.md`（給人看的表）。
每個 CSV 都有 `run_id`、`ts`（RUN_ID 環境變數；沒設＝NOT_SET）。`--spectra` 另存完整特徵值（npz，大，放 runs 目錄，不進 git）。
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")                       # 必須在 import numpy 之前
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime as dt  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

FORBIDDEN = ("torch", "vllm", "transformers")
NBINS = 2000                                     # cosine 直方圖：[-1, 1] 切 2000 格（0.001 寬）
QS = (0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95)


def assert_no_gpu_libs():
    bad = [m for m in FORBIDDEN if m in sys.modules]
    assert not bad, f"不准 import {bad}（會開 /dev/kfd，第二階段在量時間）"


# ---------------------------------------------------------------- BF16
def bf16_to_f32(u16) -> np.ndarray:
    return (np.asarray(u16, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def f32_to_bf16(x) -> np.ndarray:
    """float32 → BF16 位元，round-to-nearest-even（和 torch 的 .to(bfloat16) 相同）。"""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = ((u + (np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1)))) >> np.uint32(16)).astype(np.uint16)
    nan = (u & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
    if nan.any():
        r[nan] = ((u[nan] >> np.uint32(16)) | np.uint32(0x40)).astype(np.uint16)
    return r


def bf16r(x) -> np.ndarray:
    """捨入成 BF16 可表示的值（回傳 float32）。"""
    return bf16_to_f32(f32_to_bf16(x))


def bf16_ulp_diff(a_u16, b_u16) -> np.ndarray:
    """兩組 BF16 位元相差幾個 ulp（+0 與 −0 視為相同）。"""
    def key(u):
        u = np.asarray(u, dtype=np.uint16).astype(np.int32)
        return np.where(u & 0x8000, -(u & 0x7FFF), u & 0x7FFF)
    return np.abs(key(a_u16) - key(b_u16))


# ---------------------------------------------------------------- RoPE（模擬 transformers 5.17.0；行號見檔頭）
def rope_params(src: dict) -> dict:
    """統一成 {rope_type, rope_theta, (llama3 的 factor 等)}。src＝meta['rope']（有 rope_parameters，v5）或 config.json
    （v4：rope_theta＋rope_scaling；v5 存的 config 也可能直接有 rope_parameters）。"""
    if src.get("rope_parameters"):
        p = dict(src["rope_parameters"])
    else:
        p = dict(src.get("rope_scaling") or {})
        p.setdefault("rope_theta", src.get("rope_theta"))
    p["rope_type"] = p.get("rope_type") or p.get("type") or "default"
    if p.get("rope_theta") is None:
        raise ValueError(f"沒有 rope_theta：{src}")
    if p["rope_type"] not in ("default", "llama3"):
        raise NotImplementedError(f"rope_type={p['rope_type']}：這支只照原始碼實作了 default 與 llama3")
    prf = float(p.get("partial_rotary_factor", src.get("partial_rotary_factor", 1.0)) or 1.0)
    if prf != 1.0:
        raise NotImplementedError(f"partial_rotary_factor={prf}")
    return p


def inv_freq_hf(p: dict, head_dim: int) -> np.ndarray:
    """float32 的 inv_freq，逐步照 torch 的 float32 運算（scalar / tensor＝reciprocal × scalar）。"""
    dim = head_dim
    ex = np.arange(0, dim, 2, dtype=np.int64).astype(np.float32) / np.float32(dim)
    inv = (np.float32(1.0) / (np.float32(p["rope_theta"]) ** ex)).astype(np.float32)   # reciprocal×1.0＝1/x（都是正確捨入）
    if p["rope_type"] == "default":
        return inv
    factor = np.float32(p["factor"])
    lff, hff = float(p["low_freq_factor"]), float(p["high_freq_factor"])
    old = float(p["original_max_position_embeddings"])
    low_wl, high_wl = np.float32(old / lff), np.float32(old / hff)                       # :649–650（python float → 比較時 float32）
    wavelen = ((np.float32(1.0) / inv) * np.float32(2 * math.pi)).astype(np.float32)    # :652
    inv_l = np.where(wavelen > low_wl, (inv / factor).astype(np.float32), inv).astype(np.float32)   # :655
    smooth = ((((np.float32(1.0) / wavelen) * np.float32(old)).astype(np.float32) - np.float32(lff))
              / np.float32(hff - lff)).astype(np.float32)                              # :657
    smoothed = ((((np.float32(1.0) - smooth) * inv_l).astype(np.float32) / factor).astype(np.float32)
                + (smooth * inv_l).astype(np.float32)).astype(np.float32)               # :658
    med = ~(wavelen < high_wl) & ~(wavelen > low_wl)                                    # :659
    return np.where(med, smoothed, inv_l).astype(np.float32)                            # :660


def inv_freq_ref64(p: dict, head_dim: int) -> np.ndarray:
    """獨立的 float64 參考實作（Meta llama-models 的 apply_scaling 寫法：逐頻率 if／elif），自測拿來對照 inv_freq_hf。"""
    f = 1.0 / (float(p["rope_theta"]) ** (np.arange(0, head_dim, 2) / head_dim))
    if p["rope_type"] == "default":
        return f
    out = []
    old, lff, hff, fac = (float(p["original_max_position_embeddings"]), float(p["low_freq_factor"]),
                          float(p["high_freq_factor"]), float(p["factor"]))
    for x in f:
        wl = 2 * math.pi / x
        if wl < old / hff:
            out.append(x)
        elif wl > old / lff:
            out.append(x / fac)
        else:
            sm = (old / wl - lff) / (hff - lff)
            out.append((1 - sm) * x / fac + sm * x)
    return np.array(out)


def rope_cos_sin(inv32: np.ndarray, pos, attention_scaling: float = 1.0):
    """回傳 [T, D/2] 的 (cos, sin)，float64，值＝torch 實際用的 BF16（modeling_llama.py:122–127）。"""
    ang = np.asarray(inv32, dtype=np.float32)[None, :] * np.asarray(pos, dtype=np.float32)[:, None]   # float32 乘一次
    c = np.cos(ang.astype(np.float64)).astype(np.float32)
    s = np.sin(ang.astype(np.float64)).astype(np.float32)
    if attention_scaling != 1.0:
        c, s = c * np.float32(attention_scaling), s * np.float32(attention_scaling)
    return bf16r(c).astype(np.float64), bf16r(s).astype(np.float64)


def rope_apply64(x: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """精確（float64）的 RoPE：x [..., T, D]，c／s [T, D/2]；rotate_half 配對（j 與 j+D/2）。"""
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)


def rope_apply_bf16(x: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """逐步模擬 torch 的 BF16 前向（:159 `(k*cos) + (rotate_half(k)*sin)`；每個運算 float32 算、捨入回 BF16）。
    x、c、s 的值都必須是 BF16 可表示的數。回傳 float32（BF16 值）。"""
    h = x.shape[-1] // 2
    x = np.asarray(x, dtype=np.float32)
    c, s = np.asarray(c, dtype=np.float32), np.asarray(s, dtype=np.float32)
    x1, x2 = x[..., :h], x[..., h:]
    o1 = bf16r(bf16r(x1 * c) + bf16r(-x2 * s))
    o2 = bf16r(bf16r(x2 * c) + bf16r(x1 * s))
    return np.concatenate([o1, o2], axis=-1)


def rope_invert(y: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """[[c,−s],[s,c]] 的精確反（除以 c²+s²；BF16 的 c、s 讓 c²+s² 不是剛好 1）。float64。"""
    h = y.shape[-1] // 2
    y1, y2 = y[..., :h], y[..., h:]
    det = c * c + s * s
    return np.concatenate([(y1 * c + y2 * s) / det, (y2 * c - y1 * s) / det], axis=-1)


# ---------------------------------------------------------------- 讀 dump
class Dump:
    def __init__(self, d: str):
        self.dir = d.rstrip("/")
        self.name = os.path.basename(self.dir)
        self.meta = m = json.load(open(os.path.join(d, "meta.json")))
        self.nb, self.chunk = int(m["n_blocks"]), int(m.get("chunk", 512))
        self.layers = [int(x) for x in m["layers"]]
        self.H, self.D = int(m["H"]), int(m["D"])
        shape = (len(self.layers), 2, self.H, self.chunk, self.D)
        # 規則 6：形狀用兩條路徑核對（meta 的 block shape、bytes／block、n_tokens、檔案大小）
        assert tuple(m["bf16_block_shape"]) == shape, ("bf16_block_shape 和 layers／H／chunk／D 對不上", m["bf16_block_shape"], shape)
        assert int(m["bf16_bytes_per_block"]) == int(np.prod(shape)) * 2, "bf16_bytes_per_block 不符"
        assert int(m["n_tokens"]) == self.nb * self.chunk, ("n_tokens ≠ n_blocks × chunk", m["n_tokens"])
        path = os.path.join(d, "kv_bf16.u16")
        assert os.path.getsize(path) == self.nb * int(np.prod(shape)) * 2, ("kv_bf16 大小不符 meta", path)
        self.bf16 = np.memmap(path, dtype=np.uint16, mode="r", shape=(self.nb,) + shape)
        if "pos_start" in m:
            self.pos_start, self.pos_source = int(m["pos_start"]), "meta"
        else:
            self.pos_start, self.pos_source = 0, "推定：m7_model.prefill_chunked(start=0)（舊 dump 沒存；靠 RoPE 核對驗證）"
        self.k_rope = m.get("k_rope", "post")
        if self.k_rope != "post":
            raise NotImplementedError(f"k_rope={self.k_rope}：這支只處理 RoPE 後的 K（m7_model 的 dump）")

    def positions(self, bi: int) -> np.ndarray:
        return self.pos_start + bi * self.chunk + np.arange(self.chunk, dtype=np.int64)

    def model_config(self, override: str | None = None) -> tuple[dict | None, str]:
        if override:
            return json.load(open(override)), override
        ps = sorted(glob.glob(self.meta.get("model_path", "") or "/nonexistent"))
        if ps and os.path.exists(os.path.join(ps[-1], "config.json")):
            p = os.path.join(ps[-1], "config.json")
            return json.load(open(p)), p
        return None, "找不到 config.json（meta.model_path＝%r）" % self.meta.get("model_path")


def sample_blocks(lo: int, hi: int, k: int | None) -> list[int]:
    """[lo, hi) 之間平均取 k 個（含兩端）；k 為 0／None 或 ≥ 個數 → 全部。"""
    n = hi - lo
    if n <= 0:
        return []
    if not k or k >= n:
        return list(range(lo, hi))
    return sorted(set(int(round(x)) for x in np.linspace(lo, hi - 1, k)))


# ---------------------------------------------------------------- RoPE 的來源與核對
def resolve_rope(dp: Dump, model_config: str | None = None) -> dict:
    """決定要用的 inv_freq：meta.rope.inv_freq（kvdump 存的實際 buffer）優先，否則用 config.json 的公式。兩者都有就互相核對。"""
    mr = dp.meta.get("rope") or {}
    cfg, cfg_src = dp.model_config(model_config)
    out = dict(warnings=[], config_source=cfg_src)
    params, psrc = None, None
    if mr.get("rope_parameters"):
        params, psrc = rope_params(mr), "meta.rope.rope_parameters"
    elif cfg is not None:
        params, psrc = rope_params(cfg), cfg_src
    formula = inv_freq_hf(params, dp.D) if params else None
    if mr.get("inv_freq") is not None and "error" not in mr:
        inv = np.array(mr["inv_freq"], dtype=np.float32)
        out.update(inv_freq_source="meta.rope.inv_freq（kvdump 存的實際 buffer，dtype=%s）" % mr.get("inv_freq_dtype"),
                   attention_scaling=float(mr.get("attention_scaling", 1.0)))
    elif formula is not None:
        inv = formula
        out.update(inv_freq_source=f"公式（{psrc}）", attention_scaling=1.0)
        if mr.get("error"):
            out["warnings"].append("kvdump 存 rope 時失敗：" + mr["error"])
    else:
        raise RuntimeError(f"沒有 inv_freq 也沒有 RoPE 參數（{cfg_src}）；用 --model-config 指定 config.json")
    assert inv.shape == (dp.D // 2,), ("inv_freq 長度不是 D/2", inv.shape)
    out.update(inv_freq=inv, params=params, params_source=psrc)
    if formula is not None and inv is not formula:
        rel = float(np.max(np.abs(inv.astype(np.float64) - formula) / formula))
        out["formula_max_rel_diff"] = rel
        out["formula_match"] = rel <= 5e-7                 # ≈ 4 個 float32 ulp
        if not out["formula_match"]:
            out["warnings"].append(f"meta 的 inv_freq 和 config 公式差 {rel:.3g}（相對）；分析用 meta 的（實際用的）")
    return out


def probe_check(dp: Dump, rope: dict) -> dict:
    pr = dp.meta.get("rope_probe")
    if not pr:
        return dict(available=False, reason="meta 沒有 rope_probe（10/20 以前的 kvdump）")
    if "error" in pr:
        return dict(available=False, reason="kvdump 端失敗：" + pr["error"])
    P, H, D = pr["shape_k"]
    assert (H, D) == (dp.H, dp.D), ("探針形狀和 dump 不同", pr["shape_k"])
    pos = np.array(pr["positions"], dtype=np.int64)

    def dec(key, shape):
        return np.frombuffer(bytes.fromhex(pr[key]), dtype="<u2").reshape(shape)
    kpre, kpost = dec("k_pre_bf16_hex", (P, H, D)), dec("k_post_bf16_hex", (P, H, D))
    cosu, sinu = dec("cos_bf16_hex", (P, D)), dec("sin_bf16_hex", (P, D))
    r: dict = dict(available=True, positions=pos.tolist(), layer=int(pr.get("layer", 0)))
    # (b) cos／sin
    c, s = rope_cos_sin(rope["inv_freq"], pos, rope["attention_scaling"])
    em_c = f32_to_bf16(np.concatenate([c, c], -1).astype(np.float32))
    em_s = f32_to_bf16(np.concatenate([s, s], -1).astype(np.float32))
    u = np.concatenate([bf16_ulp_diff(em_c, cosu).ravel(), bf16_ulp_diff(em_s, sinu).ravel()])
    r["b_cos_sin"] = dict(max_ulp=int(u.max()), frac_exact=float((u == 0).mean()),
                          ok=bool(u.max() <= 1 and (u == 0).mean() >= 0.99))
    # (c) 前向模擬（用探針自己的 cos／sin，只驗套用那一步）
    ph = D // 2
    fwd = f32_to_bf16(rope_apply_bf16(bf16_to_f32(kpre), bf16_to_f32(cosu[:, None, :ph]), bf16_to_f32(sinu[:, None, :ph])))
    u = bf16_ulp_diff(fwd, kpost)
    r["c_forward_emul"] = dict(max_ulp=int(u.max()), frac_exact=float((u == 0).mean()),
                               ok=bool(u.max() <= 1 and (u == 0).mean() >= 0.99))
    # (a)、(d) 要 dump 有第 0 層
    if r["layer"] not in dp.layers:
        r["a_dump_match"] = r["d_inverse"] = dict(ok=None, reason=f"dump 沒有第 {r['layer']} 層")
        r["ok"] = None
        return r
    li = dp.layers.index(r["layer"])
    rel = pos - dp.pos_start
    assert (rel >= 0).all() and (rel < dp.nb * dp.chunk).all(), ("探針位置超出 dump", pos.tolist())
    dumpk = np.stack([np.asarray(dp.bf16[p // dp.chunk, li, 0, :, p % dp.chunk, :]) for p in rel])     # [P,H,D]
    u = bf16_ulp_diff(dumpk, kpost)
    r["a_dump_match"] = dict(max_ulp=int(u.max()), frac_exact=float((u == 0).mean()),
                             ok=bool(u.max() <= 2 and (u == 0).mean() >= 0.99))
    inv = rope_invert(bf16_to_f32(dumpk).astype(np.float64), c[:, None, :], s[:, None, :])
    ref = bf16_to_f32(kpre).astype(np.float64)
    nr = np.linalg.norm(ref, axis=-1)
    e = np.linalg.norm(inv - ref, axis=-1) / np.where(nr > 0, nr, 1.0)
    r["d_inverse"] = dict(max_rel_err=float(e.max()), median_rel_err=float(np.median(e)), ok=bool(e.max() <= 0.02))
    r["ok"] = bool(r["a_dump_match"]["ok"] and r["b_cos_sin"]["ok"] and r["c_forward_emul"]["ok"] and r["d_inverse"]["ok"])
    return r


def _mean_coherence(K: np.ndarray, pos: np.ndarray, f: np.ndarray | None, pairing: str, sign: float) -> np.ndarray:
    """K [H, T, D]（RoPE 後）用頻率 f 逆轉，回傳每個 head 的 M＝‖平均‖²／平均‖·‖²。f=None＝不逆轉。"""
    h = K.shape[-1] // 2
    z = K[..., :h] + 1j * K[..., h:] if pairing == "half" else K[..., 0::2] + 1j * K[..., 1::2]
    if f is not None:
        z = z * np.exp(-1j * sign * (pos.astype(np.float64)[:, None] * np.asarray(f, np.float64)[None, :]))
    mean = z.mean(axis=1)                                                   # [H, D/2]
    return (np.abs(mean) ** 2).sum(-1) / (np.abs(z) ** 2).sum((1, 2)) * K.shape[1]


def data_check(dp: Dump, rope: dict, n_blocks: int = 16, layers: list[int] | None = None) -> dict:
    """資料自身的 RoPE 核對（見檔頭）。用平均取樣的 block（位置分散，低頻也分得出來）。"""
    blocks = sample_blocks(0, dp.nb, n_blocks)
    pos = np.concatenate([dp.positions(b) for b in blocks])
    inv = rope["inv_freq"].astype(np.float64)
    cands: dict[str, tuple] = {"expected": (inv, "half", 1.0), "none": (None, "half", 1.0),
                               "reverse": (inv, "half", -1.0), "interleaved_pairs": (inv, "inter", 1.0)}
    p = rope.get("params")
    if p:
        for nm, fac in (("theta_x2", 2.0), ("theta_x0.5", 0.5)):
            q = dict(p, rope_theta=float(p["rope_theta"]) * fac)
            cands[nm] = (inv_freq_hf(q, dp.D).astype(np.float64), "half", 1.0)
        if p["rope_type"] == "llama3":
            cands["no_llama3_scaling"] = (inv_freq_hf(dict(p, rope_type="default"), dp.D).astype(np.float64), "half", 1.0)
    b16 = bf16r(rope["inv_freq"]).astype(np.float64)
    if not np.array_equal(b16, inv):
        cands["inv_freq_bf16"] = (b16, "half", 1.0)
    lis = [dp.layers.index(x) for x in (layers or dp.layers)]
    per = {k: [] for k in cands}
    for li in lis:
        K = bf16_to_f32(np.asarray(dp.bf16[blocks, li, 0])).astype(np.float64)   # [nbS, H, chunk, D]
        K = K.transpose(1, 0, 2, 3).reshape(dp.H, -1, dp.D)
        for k, (f, pr, sg) in cands.items():
            per[k].append(_mean_coherence(K, pos, f, pr, sg))
    M = {k: float(np.mean(np.concatenate(v))) for k, v in per.items()}
    others = max(v for k, v in M.items() if k != "expected")
    margin = M["expected"] - M["none"]
    if M["expected"] >= others - 1e-3 and margin >= 0.005:
        st = "PASS"
    elif M["expected"] < others - 1e-3:
        st = "FAIL"
    else:
        st = "INCONCLUSIVE"
    return dict(status=st, M=M, margin_vs_none=margin, n_tokens=int(pos.size), blocks=blocks,
                layers=[dp.layers[i] for i in lis],
                per_layer_expected=[float(np.mean(x)) for x in per["expected"]],
                per_layer_none=[float(np.mean(x)) for x in per["none"]])


def rope_check(dp: Dump, model_config: str | None = None, data_blocks: int = 16) -> dict:
    rope = resolve_rope(dp, model_config)
    pc = probe_check(dp, rope)
    dc = data_check(dp, rope, data_blocks)
    warn = list(rope["warnings"])
    if pc.get("ok") is True:
        status = "PASS_PROBE"
        if dc["status"] != "PASS":
            warn.append(f"探針過了，但資料自身的核對＝{dc['status']}（M={dc['M']}）")
    elif pc.get("ok") is False:
        status = "FAIL"
    else:                                              # 沒有探針，或 dump 沒有第 0 層
        if rope.get("formula_match") is False:
            status = "FAIL"
            warn.append("沒有探針可驗，而 meta 的 inv_freq 和公式不符")
        elif pc.get("available") and not (pc["b_cos_sin"]["ok"] and pc["c_forward_emul"]["ok"]):
            status = "FAIL"
        else:
            status = "PASS_DATA" if dc["status"] == "PASS" else "FAIL"
    rj = {k: v for k, v in rope.items() if k != "inv_freq"}
    rj["inv_freq_first_last"] = [float(rope["inv_freq"][0]), float(rope["inv_freq"][-1])]
    return dict(dump=dp.name, status=status, warnings=warn, rope=rj, probe=pc, data=dc,
                pos_start=dp.pos_start, pos_source=dp.pos_source, k_rope=dp.k_rope), rope


# ---------------------------------------------------------------- 累加器
class GramAcc:
    """一組（層, 種類）：每個 head 的 Gram [D,D]、和、token 數，校準（0）／測試（1）分開；可選整層串接的 [H·D, H·D]。"""

    def __init__(self, H: int, D: int, joint: bool):
        self.G = np.zeros((2, H, D, D))
        self.s = np.zeros((2, H, D))
        self.n = np.zeros(2, dtype=np.int64)
        self.maxe = np.zeros(H)
        self.maxpos = np.full(H, -1, dtype=np.int64)
        self.joint = joint
        if joint:
            self.Gj = np.zeros((2, H * D, H * D))
            self.sj = np.zeros((2, H * D))
            self.maxe_j, self.maxpos_j = 0.0, -1

    def add(self, X: np.ndarray, pos: np.ndarray, bucket: int):
        """X [H, T, D] float64。"""
        H, T, D = X.shape
        self.G[bucket] += np.matmul(X.transpose(0, 2, 1), X)
        self.s[bucket] += X.sum(1)
        self.n[bucket] += T
        e = np.einsum("htd,htd->ht", X, X)
        i = e.argmax(1)
        v = e[np.arange(H), i]
        up = v > self.maxe
        self.maxe[up], self.maxpos[up] = v[up], pos[i[up]]
        if self.joint:
            Xj = np.ascontiguousarray(X.transpose(1, 0, 2)).reshape(T, H * D)
            self.Gj[bucket] += Xj.T @ Xj
            self.sj[bucket] += Xj.sum(0)
            ej = e.sum(0)
            k = int(ej.argmax())
            if ej[k] > self.maxe_j:
                self.maxe_j, self.maxpos_j = float(ej[k]), int(pos[k])


def eig_desc(G: np.ndarray, vectors: bool = True):
    """由大到小的特徵值（負的捨入誤差截成 0）與特徵向量。V 要轉成連續記憶體：負 stride 的陣列做 matmul 不走 BLAS，
    1024 維慢 10 倍以上（10/20 bench 抓到）。"""
    Gs = (G + G.T) / 2
    if not vectors:
        return np.clip(np.linalg.eigvalsh(Gs)[::-1], 0, None), None
    w, V = np.linalg.eigh(Gs)
    return np.clip(w[::-1], 0, None), np.ascontiguousarray(V[:, ::-1])


def lowrank_metrics(Gc, sc, nc, Gt, st, nt, ranks) -> dict:
    """Gram 矩陣算的低秩量測（純函式，自測直接對照 SVD）。回傳每個 rank 的 dict 與有效秩。"""
    G, s, n = Gc + Gt, sc + st, nc + nt
    D = G.shape[0]
    tot = float(np.trace(G))
    w, _ = eig_desc(G, False)
    mu = s / n
    C = G - n * np.outer(mu, mu)
    wc, _ = eig_desc(C, False)
    trC = float(np.trace(C))
    out = dict(n=int(n), n_calib=int(nc), n_test=int(nt), total_energy=tot, eigvals=w,
               mean_energy_frac=float(n * mu @ mu / tot) if tot > 0 else float("nan"),
               stable_rank=float(tot / w[0]) if w[0] > 0 else float("nan"), ranks={})
    cum = np.cumsum(w)
    tail = lambda r: max(0.0, tot - float(cum[min(r, D) - 1])) if r > 0 else tot          # noqa: E731
    cumc = np.cumsum(wc)
    tailc = lambda r: max(0.0, trC - float(cumc[min(r, D) - 1])) if r > 0 else max(0.0, trC)  # noqa: E731
    for p in (0.90, 0.95, 0.99):
        out[f"r{int(p * 100)}"] = min(D, int(np.searchsorted(cum / tot, p - 1e-12) + 1)) if tot > 0 else None
        rr = [r for r in range(0, D + 1) if tailc(r) <= (1 - p) * tot + 1e-12 * tot]
        out[f"r{int(p * 100)}_centered"] = rr[0] if rr else D
    do_x = nc > 0 and nt > 0
    if do_x:
        _, Vc = eig_desc(Gc)
        totT = float(np.trace(Gt))
        wt, _ = eig_desc(Gt, False)
        cumt = np.cumsum(wt)
        muc = sc / nc
        _, Vcc = eig_desc(Gc - nc * np.outer(muc, muc))
        GS = Gt - np.outer(st, muc) - np.outer(muc, st) + nt * np.outer(muc, muc)
        trGS = float(np.trace(GS))
        mut = st / nt
        wct, _ = eig_desc(Gt - nt * np.outer(mut, mut), False)
        cumct = np.cumsum(wct)
        trCt = float(np.trace(Gt - nt * np.outer(mut, mut)))
    for r in ranks:
        rr = min(r, D)
        d = dict(rank=r, rank_eff=rr, energy=float(cum[rr - 1] / tot) if tot > 0 else float("nan"),
                 rel_err=math.sqrt(tail(rr) / tot) if tot > 0 else float("nan"),
                 rel_err_centered=math.sqrt(tailc(rr) / tot) if tot > 0 else float("nan"))
        if do_x and totT > 0:
            Vr, Vcr = Vc[:, :rr], Vcc[:, :rr]
            kept = float(np.sum((Gt @ Vr) * Vr))
            keptc = float(np.sum((GS @ Vcr) * Vcr))
            d.update(xfer_err=math.sqrt(max(0.0, 1 - kept / totT)),
                     xfer_oracle_err=math.sqrt(max(0.0, totT - float(cumt[rr - 1])) / totT),
                     xfer_err_centered=math.sqrt(max(0.0, trGS - keptc) / totT),
                     xfer_oracle_err_centered=math.sqrt(max(0.0, trCt - float(cumct[rr - 1])) / totT))
        else:
            d.update(xfer_err="NOT_MEASURED", xfer_oracle_err="NOT_MEASURED", xfer_err_centered="NOT_MEASURED",
                     xfer_oracle_err_centered="NOT_MEASURED")
        out["ranks"][r] = d
    return out


class SimAcc:
    """cosine 分布的累加：G 組（每個 head＋可選的 all）× 值；精確的 ≥ 門檻計數＋0.001 寬直方圖。"""

    def __init__(self, G: int, thr: list[float]):
        self.G, self.thr = G, list(thr)
        self.n = np.zeros(G, dtype=np.int64)
        self.nz = np.zeros(G, dtype=np.int64)
        self.s1 = np.zeros(G)
        self.s2 = np.zeros(G)
        self.ge = np.zeros((G, len(thr)), dtype=np.int64)
        self.hist = np.zeros((G, NBINS), dtype=np.int64)
        self.sa = np.zeros(G)                                    # Σ 角距離 arccos(cos)/π（MiniCache 的 d）

    def add(self, c: np.ndarray, valid: np.ndarray):
        """c、valid：[G, N]。"""
        self.n += valid.sum(1)
        self.nz += (~valid).sum(1)
        c0 = np.where(valid, c, 0.0)
        self.s1 += c0.sum(1)
        self.s2 += (c0 * c0).sum(1)
        self.sa += np.where(valid, np.arccos(np.clip(c0, -1.0, 1.0)) / np.pi, 0.0).sum(1)
        for k, t in enumerate(self.thr):
            self.ge[:, k] += ((c >= t) & valid).sum(1)
        idx = np.clip(np.floor((c0 + 1.0) * (NBINS / 2.0)).astype(np.int64), 0, NBINS - 1)
        idx = idx + (np.arange(self.G) * NBINS)[:, None]
        self.hist += np.bincount(idx[valid], minlength=self.G * NBINS).reshape(self.G, NBINS)

    def stats(self, g) -> dict:
        """g：組號，或 None＝全部合併。"""
        sl = slice(None) if g is None else slice(g, g + 1)
        n, nz = int(self.n[sl].sum()), int(self.nz[sl].sum())
        out = dict(n=n, n_zero=nz)
        if n == 0:
            out.update(mean="NOT_MEASURED")
            return out
        m = float(self.s1[sl].sum() / n)
        out.update(mean=m, std=math.sqrt(max(0.0, float(self.s2[sl].sum() / n) - m * m)),
                   mean_angdist=float(self.sa[sl].sum() / n))
        out.update(hist_quantiles(self.hist[sl].sum(0), QS))
        for k, t in enumerate(self.thr):
            out[f"frac_ge_{t:g}"] = float(self.ge[sl, k].sum() / n)
        return out


def hist_quantiles(h: np.ndarray, qs) -> dict:
    """[-1,1] 上 NBINS 格的直方圖 → 分位數（格內線性內插）。"""
    n = h.sum()
    cum = np.cumsum(h)
    w = 2.0 / NBINS
    out = {}
    for q in qs:
        t = q * n
        b = int(np.searchsorted(cum, t, side="left"))
        b = min(b, NBINS - 1)
        prev = cum[b - 1] if b > 0 else 0
        frac = (t - prev) / h[b] if h[b] > 0 else 0.0
        out[f"p{int(round(q * 100)):02d}"] = float(-1.0 + (b + frac) * w)
    return out


def cos_rows(A: np.ndarray, B: np.ndarray):
    """A、B [..., D] → cosine [...] 與 valid（兩邊長度都 > 0）。"""
    na, nb = np.linalg.norm(A, axis=-1), np.linalg.norm(B, axis=-1)
    valid = (na > 0) & (nb > 0)
    c = np.einsum("...d,...d->...", A, B) / np.where(valid, na * nb, 1.0)
    return np.where(valid, c, 0.0), valid


# ---------------------------------------------------------------- 分析一個 dump
def mem_estimate_gb(nl: int, nk: int, H: int, D: int, joint: bool, n_xl: int, L_read: int, chunk: int) -> float:
    g = nl * nk * (2 * H * D * D + 2 * H * D) * 8
    if joint:
        g += nl * nk * (2 * (H * D) ** 2 + 2 * H * D) * 8
    sims = (nl * nk * H + n_xl * 2 * (H + 1)) * NBINS * 8
    blk = L_read * 2 * H * chunk * D * (2 + 4) + 16 * H * chunk * D * 8          # 一個 block＋每層的暫存
    eig = (H * D) ** 2 * 8 * 6 if joint else 0                                    # 結尾 eigh 的暫存
    return (g + sims + blk + eig) / 2 ** 30


def iter_blocks(dp: Dump, blocks: list[int], lis: list[int], sel_layers: list[int], hsel: np.ndarray,
                rope: dict | None, skip: int):
    """依序讀 block：產出 (block, 位置, 逐層的 {種類: [H, T, D] float64})；第二輪（中心化）和第一輪讀到的完全一樣。"""
    for bi in blocks:
        pos = dp.positions(bi)
        U = np.asarray(dp.bf16[bi, lis])                         # [L'', 2, H, chunk, D] uint16（只讀選的層）
        if rope is not None:
            c, s = rope_cos_sin(rope["inv_freq"], pos, rope["attention_scaling"])
        keep = (pos >= skip) if pos[0] < skip else None          # --skip-first-tokens：只會切掉 block 0 的開頭
        pos_k = pos if keep is None else pos[keep]

        def layers(U=U, c=c if rope is not None else None, s=s if rope is not None else None, keep=keep):
            for j, L in enumerate(sel_layers):
                Kp = bf16_to_f32(U[j, 0][hsel]).astype(np.float64)   # [H, T, D]
                X = {"K_post": Kp, "V": bf16_to_f32(U[j, 1][hsel]).astype(np.float64)}
                if c is not None:
                    X["K_pre"] = rope_invert(Kp, c[None], s[None])
                if keep is not None:
                    X = {k: v[:, keep] for k, v in X.items()}
                yield L, X
        yield bi, pos_k, layers()


def _adj_add(acc: SimAcc, last: dict, key, bi: int, X: np.ndarray):
    """同層同 head 的 t、t+1；上一個讀到的 block 剛好是 bi−1 時，也算 block 邊界那一對。"""
    cc, vv = cos_rows(X[:, :-1], X[:, 1:])                                     # [H, T-1]
    lb = last.get(key)
    if lb is not None and lb[0] == bi - 1:
        c0, v0 = cos_rows(lb[1], X[:, 0])
        cc, vv = np.concatenate([c0[:, None], cc], 1), np.concatenate([v0[:, None], vv], 1)
    acc.add(cc, vv)
    last[key] = (bi, X[:, -1].copy())


def _xl_add(acc: SimAcc, A: np.ndarray, B: np.ndarray):
    """同 token、層 l 與 l+1：每個 head 一組，最後一組＝整層 head 串接。"""
    ch, vh = cos_rows(A, B)                                                    # [H, T]
    ca, va = cos_rows(A.transpose(1, 0, 2).reshape(A.shape[1], -1), B.transpose(1, 0, 2).reshape(B.shape[1], -1))
    acc.add(np.concatenate([ch, ca[None]], 0), np.concatenate([vh, va[None]], 0))
    return ch, vh


def analyze_dump(dp: Dump, a, rope: dict | None, run_id: str, ts: str) -> dict:
    t0 = time.perf_counter()
    sel_layers = sorted(a.layers) if a.layers else list(dp.layers)
    miss = [x for x in sel_layers if x not in dp.layers]
    assert not miss, f"dump {dp.name} 沒有層 {miss}（有 {dp.layers}）"
    lis = [dp.layers.index(x) for x in sel_layers]
    heads = sorted(a.heads) if a.heads else list(range(dp.H))
    assert all(0 <= h < dp.H for h in heads), heads
    H = len(heads)
    cb = a.calib_tokens // dp.chunk
    assert a.calib_tokens % dp.chunk == 0, "--calib-tokens 必須是 chunk 的倍數"
    assert 0 <= a.skip_first_tokens < dp.chunk, "--skip-first-tokens 必須小於一個 chunk"
    notes = []
    if cb >= dp.nb:
        cb = dp.nb // 2
        notes.append(f"--calib-tokens={a.calib_tokens} ≥ dump 的 token 數 {dp.nb * dp.chunk}；改用前 {cb} 個 block 當校準")
    test = sample_blocks(cb, dp.nb, a.max_blocks)
    blocks = sorted(set(range(cb)) | set(test))
    do_pre = rope is not None
    kinds = ["K_post", "K_pre", "V"] if do_pre else ["K_post", "V"]
    ckinds = (["K_pre", "V"] if do_pre else ["V"]) if a.centered else []      # 中心化（減掉每層每 head 的 token 平均）
    xl_pairs = [(x, x + 1) for x in sel_layers if x + 1 in sel_layers]
    est = mem_estimate_gb(len(lis), len(kinds) + len(ckinds), H, dp.D, a.joint, 2 * len(xl_pairs), len(lis), dp.chunk)
    print(f"[{dp.name}] 層 {sel_layers}、head {heads}、block {len(blocks)}（校準 {cb}）、種類 {kinds}、中心化 {ckinds}、"
          f"跨層配對 {len(xl_pairs)}、估計記憶體 {est:.2f} GiB", flush=True)
    if est > a.max_mem_gb:
        print(f"估計記憶體 {est:.2f} GiB > --max-mem-gb {a.max_mem_gb}：用 --no-joint 或少選層", file=sys.stderr)
        sys.exit(3)
    gram = {(L, k): GramAcc(H, dp.D, a.joint) for L in sel_layers for k in kinds}
    adj = {(L, k): SimAcc(H, a.thresholds) for L in sel_layers for k in kinds}
    xl = {(L, k): SimAcc(H + 1, a.thresholds) for (L, _) in xl_pairs for k in ("K", "V")}
    last: dict = {}
    xl_inv = 0.0                                                 # 跨層 cosine：RoPE 前 vs 後的最大差（旁證）
    hsel = np.array(heads)
    step = max(1, len(blocks) // 10)
    # ---- 第一輪：Gram、相鄰 token、跨層
    for n_done, (bi, pos_k, layers) in enumerate(iter_blocks(dp, blocks, lis, sel_layers, hsel, rope, a.skip_first_tokens)):
        bucket = 0 if bi < cb else 1
        prev = None
        for L, X in layers:
            for k in kinds:
                gram[(L, k)].add(X[k], pos_k, bucket)
                _adj_add(adj[(L, k)], last, (L, k), bi, X[k])
            if prev is not None and prev[0] == L - 1:
                ch, vh = _xl_add(xl[(L - 1, "K")], prev[1]["K_post"], X["K_post"])
                _xl_add(xl[(L - 1, "V")], prev[1]["V"], X["V"])
                if do_pre:                                       # 旁證：RoPE 前算的跨層 cosine 應該幾乎相同
                    cpre, vpre = cos_rows(prev[1]["K_pre"], X["K_pre"])
                    if (vh & vpre).any():
                        xl_inv = max(xl_inv, float(np.abs(cpre - ch)[vh & vpre].max()))
            prev = (L, X)
        if (n_done + 1) % step == 0 or n_done + 1 == len(blocks):
            print(f"  [{dp.name}] 第一輪 block {n_done + 1}/{len(blocks)}  {time.perf_counter() - t0:.1f}s", flush=True)
    t_acc = time.perf_counter() - t0
    # ---- 第二輪（中心化）：減掉第一輪算出的每（層, head）token 平均，再算相鄰 token 與跨層 cosine。
    #      K 用 RoPE 前的（平均方向是 RoPE 前的固定通道；RoPE 後減全域平均沒有意義），V 直接用。
    adj_c = {(L, k): SimAcc(H, a.thresholds) for L in sel_layers for k in ckinds}
    xl_c = {(L, k): SimAcc(H + 1, a.thresholds) for (L, _) in xl_pairs for k in ckinds}
    if ckinds:
        mu = {(L, k): (gram[(L, k)].s[0] + gram[(L, k)].s[1]) / gram[(L, k)].n.sum() for L in sel_layers for k in ckinds}
        last_c: dict = {}
        for n_done, (bi, pos_k, layers) in enumerate(iter_blocks(dp, blocks, lis, sel_layers, hsel, rope, a.skip_first_tokens)):
            prev = None
            for L, X in layers:
                Xc = {k: X[k] - mu[(L, k)][:, None, :] for k in ckinds}
                for k in ckinds:
                    _adj_add(adj_c[(L, k)], last_c, (L, k), bi, Xc[k])
                if prev is not None and prev[0] == L - 1:
                    for k in ckinds:
                        _xl_add(xl_c[(L - 1, k)], prev[1][k], Xc[k])
                prev = (L, Xc)
            if (n_done + 1) % step == 0 or n_done + 1 == len(blocks):
                print(f"  [{dp.name}] 第二輪（中心化） block {n_done + 1}/{len(blocks)}  {time.perf_counter() - t0:.1f}s",
                      flush=True)
    t_acc2 = time.perf_counter() - t0
    # ---- 彙總成列
    base = dict(run_id=run_id, ts=ts, dump=dp.name, model=dp.meta.get("model"), task=dp.meta.get("task"),
                n_tokens_dump=dp.nb * dp.chunk)
    lr_rows, eff_rows, adj_rows, xl_rows, spectra = [], [], [], [], {}
    for (L, k), g in gram.items():
        groups = [(heads[i], g.G[0, i], g.s[0, i], g.G[1, i], g.s[1, i], g.maxe[i], g.maxpos[i], a.ranks)
                  for i in range(H)]
        if g.joint:
            groups.append(("all", g.Gj[0], g.sj[0], g.Gj[1], g.sj[1], g.maxe_j, g.maxpos_j, [r * H for r in a.ranks]))
        for hd, Gc, sc, Gt, st, me, mp, rks in groups:
            m = lowrank_metrics(Gc, sc, int(g.n[0]), Gt, st, int(g.n[1]), rks)
            spectra[f"{dp.name}|L{L}|{k}|h{hd}"] = m["eigvals"]
            for r, d in m["ranks"].items():
                lr_rows.append(dict(base, layer=L, head=hd, kind=k, rank=r, rank_eff=d["rank_eff"],
                                    energy=d["energy"], rel_err=d["rel_err"], rel_err_centered=d["rel_err_centered"],
                                    xfer_err=d["xfer_err"], xfer_oracle_err=d["xfer_oracle_err"],
                                    xfer_err_centered=d["xfer_err_centered"],
                                    xfer_oracle_err_centered=d["xfer_oracle_err_centered"],
                                    n_tokens=m["n"], n_calib=m["n_calib"], n_test=m["n_test"]))
            eff_rows.append(dict(base, layer=L, head=hd, kind=k, dim=Gc.shape[0], r90=m["r90"], r95=m["r95"], r99=m["r99"],
                                 r90_centered=m["r90_centered"], r95_centered=m["r95_centered"],
                                 r99_centered=m["r99_centered"], stable_rank=m["stable_rank"],
                                 mean_energy_frac=m["mean_energy_frac"],
                                 top_token_energy_frac=float(me / m["total_energy"]) if m["total_energy"] > 0 else "NOT_MEASURED",
                                 top_token_pos=int(mp), n_tokens=m["n"]))
    for accs, suffix in ((adj, ""), (adj_c, "_centered")):
        for (L, k), acc in accs.items():
            for i, hd in enumerate(heads):
                adj_rows.append(dict(base, layer=L, head=hd, kind=k + suffix, **acc.stats(i)))
            adj_rows.append(dict(base, layer=L, head="pooled", kind=k + suffix, **acc.stats(None)))
    for accs, ren in ((xl, {"K": "K", "V": "V"}), (xl_c, {"K_pre": "K_centered", "V": "V_centered"})):
        for (L, k), acc in accs.items():
            for i, hd in enumerate(heads):
                xl_rows.append(dict(base, layer=L, layer_next=L + 1, kind=ren[k], head=hd, **acc.stats(i)))
            xl_rows.append(dict(base, layer=L, layer_next=L + 1, kind=ren[k], head="all", **acc.stats(H)))
    not_meas = [f"{x}→{x + 1}" for x in sel_layers if x + 1 < int(dp.meta.get("L_total", 10 ** 9)) and x + 1 not in sel_layers]
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20               # Linux：KiB → GiB
    info = dict(dump=dp.name, meta={k: dp.meta.get(k) for k in ("model", "task", "n_tokens", "layers", "run_id", "item_id", "seed")},
                layers=sel_layers, heads=heads, kinds=kinds, centered_kinds=[k + "_centered" for k in ckinds],
                calib_blocks=cb, calib_tokens=cb * dp.chunk, skip_first_tokens=a.skip_first_tokens,
                test_blocks=[b for b in blocks if b >= cb], n_blocks_used=len(blocks), ranks=a.ranks,
                joint_ranks=[r * H for r in a.ranks] if a.joint else None, thresholds=a.thresholds,
                xlayer_pairs=[f"{x}→{y}" for x, y in xl_pairs], xlayer_not_measured=not_meas, notes=notes,
                xlayer_K_pre_vs_post_max_abs_diff=xl_inv if (do_pre and xl_pairs) else "NOT_MEASURED",
                pre_rope="measured" if do_pre else "NOT_MEASURED（RoPE 核對沒過或 --no-pre-rope）",
                mem_estimate_gb=round(est, 3), peak_rss_gb=round(peak, 3), t_pass1_s=round(t_acc, 1),
                t_pass2_s=round(t_acc2 - t_acc, 1), t_total_s=round(time.perf_counter() - t0, 1))
    return dict(info=info, lowrank=lr_rows, eff=eff_rows, adj=adj_rows, xl=xl_rows, spectra=spectra)


# ---------------------------------------------------------------- 輸出
def write_csv(path: str, rows: list[dict]):
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="NOT_MEASURED")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.7g}" if isinstance(v, float) else v) for k, v in r.items()})


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _mean(vals):
    v = [x for x in (_num(x) for x in vals) if x is not None and not math.isnan(x)]
    return float(np.mean(v)) if v else "NOT_MEASURED"


def aggregate(res: dict) -> dict:
    """給 json／md：每個（種類, 層）對 head 平均（不含 head=all），另列 head=all。"""
    info = res["info"]
    lr: dict = {}
    for r in res["lowrank"]:
        key = (r["kind"], "all" if r["head"] == "all" else "head_mean")
        d = lr.setdefault(key[0], {}).setdefault(key[1], {}).setdefault(str(r["layer"]), {}).setdefault(str(r["rank"]), [])
        d.append(r)
    lowrank = {}
    for kind, gs in lr.items():
        for g, layers in gs.items():
            for L, rks in layers.items():
                for rk, rows in rks.items():
                    lowrank.setdefault(kind, {}).setdefault(g, {}).setdefault(L, {})[rk] = {
                        f: _mean(x[f] for x in rows) for f in ("energy", "rel_err", "rel_err_centered", "xfer_err",
                                                               "xfer_oracle_err", "xfer_err_centered",
                                                               "xfer_oracle_err_centered")}
    eff = {}
    for r in res["eff"]:
        g = "all" if r["head"] == "all" else "head_mean"
        eff.setdefault(r["kind"], {}).setdefault(g, {}).setdefault(str(r["layer"]), []).append(r)
    eff = {k: {g: {L: {f: _mean(x[f] for x in rows) for f in ("r90", "r95", "r99", "r90_centered", "r95_centered",
                                                                 "r99_centered", "stable_rank", "mean_energy_frac",
                                                                 "top_token_energy_frac")}
                   for L, rows in ls.items()} for g, ls in gs.items()} for k, gs in eff.items()}
    adj = {}
    for r in res["adj"]:
        if r["head"] == "pooled":
            adj.setdefault(r["kind"], {})[str(r["layer"])] = {k: v for k, v in r.items()
                                                              if k not in ("run_id", "ts", "dump", "model", "task")}
    xl = {}
    for r in res["xl"]:
        g = "all_heads_concat" if r["head"] == "all" else "head"
        xl.setdefault(r["kind"], {}).setdefault(f"{r['layer']}→{r['layer_next']}", {}).setdefault(g, []).append(r)
    xlo = {}
    for k, ps in xl.items():
        for p, gs in ps.items():
            o = {}
            for g, rows in gs.items():
                fs = [f for f in rows[0] if f.startswith(("mean", "p", "frac_ge_")) and f not in ("model",)]
                o[g if g != "head" else "head_mean"] = {f: _mean(x.get(f) for x in rows) for f in fs}
            xlo.setdefault(k, {})[p] = o
    return dict(info=info, lowrank=lowrank, effrank=eff, adjtok=adj, xlayer=xlo)


def _f(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, float) else ("NOT_MEASURED" if v in (None, "") else str(v))


def write_md(path: str, run_id: str, ts: str, aggs: list[dict], rchecks: list[dict], argv: list[str]):
    L = [f"# K1：KV 結構（低秩／跨層／相鄰 token）摘要", "",
         f"- run_id：`{run_id}`　ts：`{ts}`", f"- 指令：`{' '.join(argv)}`",
         "- 數字全部來自同目錄的 CSV（每列有 run_id）；表內是對 head 的平均（`all`＝整層 head 串接）。",
         "- 相對誤差＝‖X−X_r‖_F／‖X‖_F；xfer＝前 calib 個 token 的基底套到其後 token；oracle＝測試 token 自己的最佳 rank r。", ""]
    L += ["## RoPE 核對", "", "| dump | 狀態 | inv_freq 來源 | 探針 (a/b/c/d) | 資料自身 M（期望／不逆轉） | 警告 |", "|:--|:--|:--|:--|:--|:--|"]
    for rc in rchecks:
        p = rc["probe"]
        ps = ("／".join(str(p.get(k, {}).get("ok")) for k in ("a_dump_match", "b_cos_sin", "c_forward_emul", "d_inverse"))
              if p.get("available") else "無：" + p.get("reason", ""))
        d = rc["data"]
        L.append(f"| {rc['dump']} | **{rc['status']}** | {rc['rope'].get('inv_freq_source')} | {ps} | "
                 f"{d['status']}：{_f(d['M']['expected'])}／{_f(d['M']['none'])} | {'；'.join(rc['warnings']) or '無'} |")
    L.append("")
    for ag in aggs:
        info = ag["info"]
        L += [f"## {info['dump']}（{info['meta'].get('model')}、{info['meta'].get('task')}、{info['meta'].get('n_tokens')} token）", "",
              f"- 層 {info['layers']}；head {info['heads']}；用了 {info['n_blocks_used']} 個 block（校準 {info['calib_tokens']} token）；"
              f"略過位置 < {info['skip_first_tokens']} 的 token；"
              f"RoPE 前的 K：{info['pre_rope']}；時間 {info['t_total_s']} s；峰值 RSS {info['peak_rss_gb']} GiB。",
              f"- 旁證：跨層 K 的 cosine 用 RoPE 前／後算的最大差＝{_f(info['xlayer_K_pre_vs_post_max_abs_diff'])}"
              f"（理論上只差 BF16 捨入；大 → 逆旋轉有問題）。"]
        for n in info["notes"]:
            L.append(f"- 注意：{n}")
        L.append("")
        ranks = [str(r) for r in info["ranks"]]
        ref = "32" if "32" in ranks else ranks[len(ranks) // 2]
        for kind in info["kinds"]:
            lk = ag["lowrank"].get(kind, {}).get("head_mean", {})
            ek = ag["effrank"].get(kind, {}).get("head_mean", {})
            if not lk:
                continue
            L += [f"### 低秩：{kind}（每個 head；rank r 的相對誤差，越小越好）", "",
                  "| 層 | " + " | ".join(f"r={r}" for r in ranks) + f" | r={ref} centered | r95 | 平均方向能量 | 最大單 token 能量 | xfer r={ref}（oracle） |",
                  "|--:|" + "--:|" * (len(ranks) + 5)]
            for Ls in sorted(lk, key=int):
                row = lk[Ls]
                x32 = row.get(ref, {})
                L.append(f"| {Ls} | " + " | ".join(_f(row.get(r, {}).get("rel_err")) for r in ranks) +
                         f" | {_f(x32.get('rel_err_centered'))} | {_f(ek.get(Ls, {}).get('r95'), 1)} | "
                         f"{_f(ek.get(Ls, {}).get('mean_energy_frac'))} | {_f(ek.get(Ls, {}).get('top_token_energy_frac'))} | "
                         f"{_f(x32.get('xfer_err'))}（{_f(x32.get('xfer_oracle_err'))}） |")
            la = ag["lowrank"].get(kind, {}).get("all", {})
            if la:
                L += ["", f"整層串接（rank＝r×H）：" + "；".join(
                    f"L{Ls}: " + "、".join(f"{rk}→{_f(v.get('rel_err'))}" for rk, v in sorted(la[Ls].items(), key=lambda z: int(z[0])))
                    for Ls in sorted(la, key=int))]
            L.append("")
        L += ["### 跨層相似（同 token、l 與 l+1；cosine；head 平均）", "",
              "K 的 cosine 在 RoPE 前後相同；「中心化」＝先減掉每（層, head）對 token 的平均向量（K 用 RoPE 前的），"
              "看「很像」是不是只來自固定的大通道。", ""]
        X_ = ag["xlayer"]
        if X_:
            def g_(kind, p, grp, f):
                return X_.get(kind, {}).get(p, {}).get(grp, {}).get(f)
            L += ["| 配對 | K 平均 | K p50 | K ≥0.9 | K 整層 | K 中心化 | V 平均 | V p50 | V ≥0.9 | V 整層 | V 中心化 |",
                  "|:--|" + "--:|" * 10]
            pairs = sorted(set().union(*[set(v) for v in X_.values()]), key=lambda p: int(p.split("→")[0]))
            for p in pairs:
                L.append(f"| {p} | " + " | ".join(_f(g_(k, p, grp, f)) for k, grp, f in (
                    ("K", "head_mean", "mean"), ("K", "head_mean", "p50"), ("K", "head_mean", "frac_ge_0.9"),
                    ("K", "all_heads_concat", "mean"), ("K_centered", "head_mean", "mean"),
                    ("V", "head_mean", "mean"), ("V", "head_mean", "p50"), ("V", "head_mean", "frac_ge_0.9"),
                    ("V", "all_heads_concat", "mean"), ("V_centered", "head_mean", "mean"))) + " |")
            for k in ("K", "V", "K_centered", "V_centered"):
                rk = sorted(((v.get("head_mean", {}).get("mean"), p) for p, v in X_.get(k, {}).items()
                             if isinstance(v.get("head_mean", {}).get("mean"), float)), reverse=True)[:3]
                if rk:
                    L.append(f"\n最像的 3 對（{k}，head 平均 cosine）：" + "、".join(f"{p}（{m:.4f}）" for m, p in rk))
        else:
            L.append("NOT_MEASURED（dump 裡沒有相鄰的兩層）")
        if info["xlayer_not_measured"]:
            L.append(f"\n沒有量（缺 l+1 層）：{', '.join(info['xlayer_not_measured'])}")
        L += ["", "### 相鄰 token 相似（同層同 head，t 與 t+1；所有 head 合併）", ""]
        thr = [f"{t:g}" for t in info["thresholds"]]
        L += ["| 層 | 種類 | 平均 | p50 | " + " | ".join(f"≥{t}" for t in thr) + " |", "|--:|:--|--:|--:|" + "--:|" * len(thr)]
        for kind in info["kinds"] + info.get("centered_kinds", []):
            for Ls, st in sorted(ag["adjtok"].get(kind, {}).items(), key=lambda z: int(z[0])):
                L.append(f"| {Ls} | {kind} | {_f(st.get('mean'))} | {_f(st.get('p50'))} | " +
                         " | ".join(_f(st.get(f'frac_ge_{t}')) for t in thr) + " |")
        L.append("")
    open(path, "w").write("\n".join(L) + "\n")


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def cmd_ropecheck(a) -> int:
    rcs = []
    for d in a.dump:
        rc, _ = rope_check(Dump(d), a.model_config, a.data_blocks)
        rcs.append(rc)
        print(json.dumps(rc, ensure_ascii=False, default=_json_default, indent=1))
    return 0 if all(rc["status"] in ("PASS_PROBE", "PASS_DATA") for rc in rcs) else 2


def cmd_analyze(a) -> int:
    assert_no_gpu_libs()
    run_id = os.environ.get("RUN_ID", "NOT_SET")
    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    os.makedirs(a.out, exist_ok=True)
    dumps = [Dump(d) for d in a.dump]
    rchecks, ropes = [], []
    for dp in dumps:                                          # 先全部做 RoPE 核對（快），不過就停（規則 2）
        if a.no_pre_rope:
            rchecks.append(dict(dump=dp.name, status="SKIPPED（--no-pre-rope）", warnings=[], rope={}, probe={"available": False,
                                "reason": "--no-pre-rope"}, data=dict(status="SKIPPED", M={"expected": None, "none": None})))
            ropes.append(None)
            continue
        rc, rope = rope_check(dp, a.model_config, a.data_blocks)
        rchecks.append(rc)
        ropes.append(rope if rc["status"] in ("PASS_PROBE", "PASS_DATA") else None)
        print(f"[{dp.name}] RoPE 核對：{rc['status']}  警告：{rc['warnings'] or '無'}", flush=True)
    json.dump(dict(run_id=run_id, ts=ts, checks=rchecks), open(os.path.join(a.out, "rope_check.json"), "w"),
              indent=1, ensure_ascii=False, default=_json_default)
    bad = [rc["dump"] for rc, r in zip(rchecks, ropes) if r is None and not a.no_pre_rope]
    if bad:
        print(f"RoPE 核對沒過：{bad}。細節在 {a.out}/rope_check.json。停（規則 2）；"
              f"只要其餘量測就加 --no-pre-rope 重跑。", file=sys.stderr)
        return 2
    allres = [analyze_dump(dp, a, rope, run_id, ts) for dp, rope in zip(dumps, ropes)]
    write_csv(os.path.join(a.out, "lowrank.csv"), [r for x in allres for r in x["lowrank"]])
    write_csv(os.path.join(a.out, "lowrank_eff.csv"), [r for x in allres for r in x["eff"]])
    write_csv(os.path.join(a.out, "adjtok.csv"), [r for x in allres for r in x["adj"]])
    xrows = [r for x in allres for r in x["xl"]]
    if xrows:
        write_csv(os.path.join(a.out, "xlayer.csv"), xrows)
    else:
        write_csv(os.path.join(a.out, "xlayer.csv"), [dict(run_id=run_id, ts=ts, dump=x["info"]["dump"], layer="", layer_next="",
                                                           kind="", head="", mean="NOT_MEASURED") for x in allres])
    aggs = [aggregate(x) for x in allres]
    for nm, key in (("lowrank", "lowrank"), ("xlayer", "xlayer"), ("adjtok", "adjtok")):
        json.dump(dict(run_id=run_id, ts=ts, dumps={ag["info"]["dump"]: ag[key] for ag in aggs},
                       effrank={ag["info"]["dump"]: ag["effrank"] for ag in aggs} if key == "lowrank" else None),
                  open(os.path.join(a.out, f"{nm}.json"), "w"), indent=1, ensure_ascii=False, default=_json_default)
    if a.spectra:
        os.makedirs(os.path.dirname(os.path.abspath(a.spectra)), exist_ok=True)
        np.savez_compressed(a.spectra, **{k: v for x in allres for k, v in x["spectra"].items()})
    summary = dict(run_id=run_id, ts=ts, argv=sys.argv, numpy=np.__version__, omp_threads=os.environ.get("OMP_NUM_THREADS"),
                   affinity=sorted(os.sched_getaffinity(0)), loadavg=os.getloadavg(), spectra=a.spectra,
                   rope_status={rc["dump"]: rc["status"] for rc in rchecks}, dumps=[x["info"] for x in allres])
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, ensure_ascii=False, default=_json_default)
    write_md(os.path.join(a.out, "summary.md"), run_id, ts, aggs, rchecks, sys.argv)
    assert_no_gpu_libs()
    print("寫好", a.out)
    return 0


# ---------------------------------------------------------------- 自測（合成資料，只驗程式，不產生任何結果）
LLAMA3 = dict(rope_type="llama3", rope_theta=500000.0, factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0,
              original_max_position_embeddings=8192)
QWEN = dict(rope_type="default", rope_theta=10000000.0)


def make_fake_dump(d: str, *, nb=6, chunk=64, layers=(0, 1, 2, 4), H=2, D=32, params=LLAMA3, rotate=True, probe=True,
                   meta_rope=True, seed=0, r0=3):
    """寫一個假 dump（格式和 m16_s1_kvdump.py 相同）。結構（已知答案）：
    - K（RoPE 前）＝平均方向 μ＋rank r0＋1e-3 雜訊；RoPE 後＝BF16 前向模擬（rotate=False → 直接存 RoPE 前，給門檻測試）。
    - V：rank 4＋1e-3 雜訊；最後一層的 V 是緩慢漂移的序列（相鄰 token cosine 高）。
    - 第 1 層＝第 0 層的 K、V（RoPE 前）＋小擾動 → 1→2 很像；第 0 層和第 1 層獨立 → 0→1 不像。"""
    rng = np.random.default_rng(seed)
    os.makedirs(d, exist_ok=True)
    T = nb * chunk
    pos = np.arange(T)
    inv = inv_freq_hf(rope_params(dict(rope_parameters=params)), D)
    c, s = rope_cos_sin(inv, pos)
    Lp = len(layers)
    kpre = np.zeros((Lp, H, T, D), np.float32)
    v = np.zeros((Lp, H, T, D), np.float32)
    for li in range(Lp):
        for h in range(H):
            mu = rng.standard_normal(D) * 2.0
            A, B = rng.standard_normal((T, r0)), rng.standard_normal((r0, D))
            kpre[li, h] = mu + A @ B + 1e-3 * rng.standard_normal((T, D))
            A, B = rng.standard_normal((T, 4)), rng.standard_normal((4, D))
            v[li, h] = A @ B + 1e-3 * rng.standard_normal((T, D))
    i1, i2 = layers.index(1), layers.index(2)
    kpre[i2] = kpre[i1] + 0.01 * rng.standard_normal(kpre[i1].shape)
    v[i2] = v[i1] + 0.01 * rng.standard_normal(v[i1].shape)
    lastv = np.cumsum(0.05 * rng.standard_normal((H, T, D)), axis=1) + rng.standard_normal((H, 1, D))
    v[-1] = lastv
    kpre_b = bf16r(kpre)
    kpost_b = rope_apply_bf16(kpre_b, c.astype(np.float32), s.astype(np.float32)) if rotate else kpre_b
    kv = np.stack([kpost_b, bf16r(v)], axis=1)                                  # [L', 2, H, T, D]
    blk = kv.reshape(Lp, 2, H, nb, chunk, D).transpose(3, 0, 1, 2, 4, 5)       # [nb, L', 2, H, chunk, D]
    f32_to_bf16(np.ascontiguousarray(blk)).tofile(os.path.join(d, "kv_bf16.u16"))
    meta = dict(run_id="SELFTEST", model="synthetic", model_path="/nonexistent", task="none", n_blocks=nb, n_tokens=T,
                chunk=chunk, L_total=max(layers) + 2, layers=list(layers), H=H, D=D, item_id=None, seed=seed,
                bf16_block_shape=[Lp, 2, H, chunk, D], bf16_bytes_per_block=Lp * 2 * H * chunk * D * 2, parts={},
                pos_start=0, k_rope="post")
    if meta_rope:
        meta["rope"] = dict(rope_type=params["rope_type"], attention_scaling=1.0, inv_freq=[float(x) for x in inv],
                            inv_freq_dtype="torch.float32", pairing="rotate_half", rope_parameters=params)
    if probe:
        pp = sorted({0, 1, chunk - 1, chunk, T // 2, T - 1})
        li0 = layers.index(0)
        pre_u = f32_to_bf16(kpre_b[li0][:, pp].transpose(1, 0, 2))               # [P, H, D]
        post_true = rope_apply_bf16(kpre_b, c.astype(np.float32), s.astype(np.float32))  # 「torch」算的（不管 dump 存什麼）
        post_u = f32_to_bf16(post_true[li0][:, pp].transpose(1, 0, 2))
        cc = f32_to_bf16(np.concatenate([c[pp], c[pp]], -1).astype(np.float32))
        ss = f32_to_bf16(np.concatenate([s[pp], s[pp]], -1).astype(np.float32))
        meta["rope_probe"] = dict(layer=0, positions=pp, shape_k=[len(pp), H, D], shape_cs=[len(pp), D],
                                  k_pre_bf16_hex=pre_u.astype("<u2").tobytes().hex(), k_post_bf16_hex=post_u.astype("<u2").tobytes().hex(),
                                  cos_bf16_hex=cc.astype("<u2").tobytes().hex(), sin_bf16_hex=ss.astype("<u2").tobytes().hex())
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"))
    return dict(kpre=kpre_b, v=bf16r(v), pos=pos, inv=inv, c=c, s=s)


def self_test() -> bool:
    import tempfile
    ok = True
    n_items = 0

    def check(name, cond, info=""):
        nonlocal ok, n_items
        n_items += 1
        print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info else ""))
        ok &= bool(cond)

    rng = np.random.default_rng(0)
    assert_no_gpu_libs()
    # ---- BF16
    x = np.array([1.0, 1.0 + 2 ** -8, 1.0 + 3 * 2 ** -8, -2.5, 0.0, 1e-40, 3.0e38], dtype=np.float32)
    check("BF16 捨入：tie 到偶數、進位、負數、0", np.array_equal(bf16r(x)[:5], np.array([1.0, 1.0, 1.0 + 2 ** -6, -2.5, 0.0], np.float32)),
          str(bf16r(x)))
    y = rng.standard_normal(100000).astype(np.float32)
    yb = bf16r(y)
    re_ = float(np.max(np.abs(yb - y) / np.abs(y)))
    check("BF16 捨入誤差 ≤ 2^-8 相對（半個 ulp；8 位有效位元）", re_ <= 2 ** -8 + 1e-12, f"{re_:.3e}")
    check("BF16 位元往返", np.array_equal(f32_to_bf16(bf16_to_f32(f32_to_bf16(y))), f32_to_bf16(y)))
    check("ulp 距離：相鄰值＝1、±0＝0", bf16_ulp_diff(np.array([0x3F80, 0x0000]), np.array([0x3F81, 0x8000])).tolist() == [1, 0])
    # ---- inv_freq
    for nm, p in (("llama3", LLAMA3), ("qwen default", QWEN)):
        f32 = inv_freq_hf(rope_params(dict(rope_parameters=p)), 128)
        ref = inv_freq_ref64(rope_params(dict(rope_parameters=p)), 128)
        rel = float(np.max(np.abs(f32 - ref) / ref))
        check(f"inv_freq（{nm}）float32 模擬 vs 獨立 float64 參考：相對差 ≤ 1e-6", rel <= 1e-6, f"{rel:.2e}")
    f = inv_freq_hf(rope_params(dict(rope_parameters=LLAMA3)), 128)
    f0 = inv_freq_hf(rope_params(dict(rope_parameters=dict(rope_type="default", rope_theta=500000.0))), 128)
    wl = 2 * np.pi / f0
    check("llama3：波長 < 2048 不變、> 8192 除以 8、中間介於兩者",
          np.array_equal(f[wl < 2048], f0[wl < 2048]) and np.allclose(f[wl > 8192], f0[wl > 8192] / 8, rtol=1e-6)
          and bool(np.all((f[(wl >= 2048) & (wl <= 8192)] <= f0[(wl >= 2048) & (wl <= 8192)])
                          & (f[(wl >= 2048) & (wl <= 8192)] >= f0[(wl >= 2048) & (wl <= 8192)] / 8))),
          f"不變 {int((wl < 2048).sum())}、÷8 {int((wl > 8192).sum())}、中間 {int(((wl >= 2048) & (wl <= 8192)).sum())}")
    cfg_llama = dict(rope_theta=500000.0, rope_scaling=dict(LLAMA3, rope_type="llama3"))
    cfg_qwen = dict(rope_theta=10000000.0, rope_scaling=None)
    check("config.json（v4 形式）與 rope_parameters（v5 形式）解析一致",
          np.array_equal(inv_freq_hf(rope_params(cfg_llama), 128), f)
          and np.array_equal(inv_freq_hf(rope_params(cfg_qwen), 128), inv_freq_hf(rope_params(dict(rope_parameters=QWEN)), 128)))
    # ---- RoPE 旋轉
    D = 128
    pos = rng.integers(0, 131072, size=300)
    c, s = rope_cos_sin(f, pos)
    X = rng.standard_normal((3, 300, D))
    Y = rope_apply64(X, c, s)
    e = float(np.max(np.abs(rope_invert(Y, c, s) - X)))
    check("RoPE 旋轉→逆旋轉（float64）還原 ≤ 1e-12", e <= 1e-12, f"{e:.2e}")
    c64, s64 = np.cos(pos[:, None] * f[None].astype(np.float64)), np.sin(pos[:, None] * f[None].astype(np.float64))
    nrm = float(np.max(np.abs(np.linalg.norm(rope_apply64(X, c64, s64), axis=-1) - np.linalg.norm(X, axis=-1))))
    check("RoPE（精確 cos/sin）保長度（正交）", nrm <= 1e-10, f"{nrm:.2e}")
    e1 = np.zeros(D)
    e1[0] = 1.0
    cc, ss = rope_cos_sin(f, np.array([7]))
    r1 = rope_apply64(e1[None], cc, ss)[0]
    check("rotate_half 配對：e_0 轉到 (cos, 0…, sin@D/2)", abs(r1[0] - cc[0, 0]) < 1e-15 and abs(r1[D // 2] - ss[0, 0]) < 1e-15
          and np.count_nonzero(r1) == 2)
    q, k = rng.standard_normal(D), rng.standard_normal(D)

    def rot64(v, p):
        return rope_apply64(v[None], np.cos(p * f.astype(np.float64))[None], np.sin(p * f.astype(np.float64))[None])[0]
    d1, d2 = rot64(q, 1000) @ rot64(k, 990), rot64(q, 50010) @ rot64(k, 50000)
    check("RoPE 相對位置性質：⟨R_m q, R_n k⟩ 只看 m−n", abs(d1 - d2) < 1e-9, f"{d1:.6f} vs {d2:.6f}")
    Xb = bf16r(X).astype(np.float32)
    Yb = rope_apply_bf16(Xb, c.astype(np.float32), s.astype(np.float32))
    check("BF16 前向模擬的輸出都是 BF16 值", np.array_equal(bf16r(Yb), Yb))
    rel = np.linalg.norm(rope_invert(Yb.astype(np.float64), c, s) - Xb, axis=-1) / np.linalg.norm(Xb, axis=-1)
    check("BF16 旋轉→逆旋轉：每個向量相對誤差 ≤ 2^-7（BF16 捨入量級）", float(rel.max()) <= 2 ** -7,
          f"max {rel.max():.2e}、中位數 {np.median(rel):.2e}")
    # ---- 低秩
    T, Dm = 400, 32
    U, _ = np.linalg.qr(rng.standard_normal((T, Dm)))
    Vq, _ = np.linalg.qr(rng.standard_normal((Dm, Dm)))
    sv = np.array([10.0, 5, 3, 2, 1] + [0.0] * (Dm - 5))
    M = (U * sv) @ Vq.T
    G = M.T @ M
    m = lowrank_metrics(G * 0, np.zeros(Dm), 0, G, M.sum(0), T, [1, 3, 5, 8])
    want = {r: math.sqrt((sv[r:] ** 2).sum() / (sv ** 2).sum()) for r in (1, 3, 5)}
    check("已知奇異值：rank r 的相對誤差＝√(Σ_{i>r}σ²/Σσ²)", all(abs(m["ranks"][r]["rel_err"] - want[r]) < 1e-9 for r in want),
          str({r: round(m["ranks"][r]["rel_err"], 6) for r in want}))
    check("秩 5 的矩陣：r=5、8 誤差≈0、r99≤5", m["ranks"][5]["rel_err"] < 1e-6 and m["ranks"][8]["rel_err"] < 1e-6 and m["r99"] <= 5)
    sv_np = np.linalg.svd(M, compute_uv=False)
    check("Gram 特徵值＝SVD 的 σ²", np.allclose(m["eigvals"][:5], sv_np[:5] ** 2, rtol=1e-9))
    Z = rng.standard_normal((T, Dm)) @ np.diag(np.linspace(3, 0.1, Dm))
    blocks = np.array_split(Z, 7)
    Gs = sum(b.T @ b for b in blocks)
    m1 = lowrank_metrics(Gs * 0, np.zeros(Dm), 0, Gs, Z.sum(0), T, [4, 16])
    Uz, Sz, Vz = np.linalg.svd(Z, full_matrices=False)
    direct = np.linalg.norm(Z - (Uz[:, :4] * Sz[:4]) @ Vz[:4]) / np.linalg.norm(Z)
    check("分 block 串流累加的 Gram＝整段直接 SVD", abs(m1["ranks"][4]["rel_err"] - direct) < 1e-9, f"{direct:.6f}")
    # 校準→測試：同子空間 → xfer≈oracle≈0；正交子空間 → xfer＝1
    Bs = np.linalg.qr(rng.standard_normal((Dm, Dm)))[0]
    Xc = rng.standard_normal((300, 4)) @ Bs[:, :4].T
    Xt_same = rng.standard_normal((200, 4)) @ Bs[:, :4].T
    Xt_orth = rng.standard_normal((200, 4)) @ Bs[:, 4:8].T
    ms = lowrank_metrics(Xc.T @ Xc, Xc.sum(0), 300, Xt_same.T @ Xt_same, Xt_same.sum(0), 200, [4])
    mo = lowrank_metrics(Xc.T @ Xc, Xc.sum(0), 300, Xt_orth.T @ Xt_orth, Xt_orth.sum(0), 200, [4])
    check("校準→測試：同子空間 xfer≈0；正交子空間 xfer＝1、oracle≈0",
          ms["ranks"][4]["xfer_err"] < 1e-6 and abs(mo["ranks"][4]["xfer_err"] - 1) < 1e-6 and mo["ranks"][4]["xfer_oracle_err"] < 1e-6)
    Xc2, Xt2 = rng.standard_normal((300, Dm)), rng.standard_normal((200, Dm)) * np.linspace(2, 0.2, Dm)
    m2 = lowrank_metrics(Xc2.T @ Xc2, Xc2.sum(0), 300, Xt2.T @ Xt2, Xt2.sum(0), 200, [6])
    _, _, Vc2 = np.linalg.svd(Xc2, full_matrices=False)
    P = Vc2[:6].T @ Vc2[:6]
    d_unc = np.linalg.norm(Xt2 - Xt2 @ P) / np.linalg.norm(Xt2)
    muc = Xc2.mean(0)
    _, _, Vcc = np.linalg.svd(Xc2 - muc, full_matrices=False)
    Pc = Vcc[:6].T @ Vcc[:6]
    d_c = np.linalg.norm((Xt2 - muc) - (Xt2 - muc) @ Pc) / np.linalg.norm(Xt2)
    check("校準→測試（uncentered／centered）＝直接投影的誤差", abs(m2["ranks"][6]["xfer_err"] - d_unc) < 1e-9
          and abs(m2["ranks"][6]["xfer_err_centered"] - d_c) < 1e-9, f"{d_unc:.6f}／{d_c:.6f}")
    mu = rng.standard_normal(Dm) * 5
    Xm = mu + rng.standard_normal((T, 2)) @ rng.standard_normal((2, Dm))
    m3 = lowrank_metrics(np.zeros((Dm, Dm)), np.zeros(Dm), 0, Xm.T @ Xm, Xm.sum(0), T, [2, 3])
    check("平均＋秩 2：centered r=2 誤差≈0、uncentered r=2 > 0、uncentered r=3≈0",
          m3["ranks"][2]["rel_err_centered"] < 1e-6 and m3["ranks"][2]["rel_err"] > 1e-3 and m3["ranks"][3]["rel_err"] < 1e-6
          and m3["mean_energy_frac"] > 0.5)
    # ---- cosine 分布
    alpha = 0.3
    T2, D2 = 2000, 16
    seq = np.zeros((T2, D2))
    seq[0] = rng.standard_normal(D2)
    seq[0] /= np.linalg.norm(seq[0])
    for t in range(1, T2):
        u = rng.standard_normal(D2)
        u -= (u @ seq[t - 1]) * seq[t - 1]
        u /= np.linalg.norm(u)
        seq[t] = math.cos(alpha) * seq[t - 1] + math.sin(alpha) * u
    seq *= rng.uniform(0.5, 3.0, size=(T2, 1))                                  # 長度不同 → 方向正規化後不影響
    acc = SimAcc(1, [0.9, 0.95, 0.99])
    for b0 in range(0, T2, 128):                                                # 分 block（含邊界）＝整段
        lo, end = max(0, b0 - 1), min(b0 + 128, T2)
        cc_, vv_ = cos_rows(seq[None, lo:end - 1], seq[None, lo + 1:end])
        acc.add(cc_, vv_)
    st = acc.stats(0)
    check("相鄰 token：已知 cos α 的序列 → 平均＝cos α、分位數誤差 ≤ 0.001、n＝T−1",
          abs(st["mean"] - math.cos(alpha)) < 1e-9 and abs(st["p50"] - math.cos(alpha)) <= 0.001 and st["n"] == T2 - 1,
          f"mean {st['mean']:.6f}、p50 {st['p50']:.4f}、cos α {math.cos(alpha):.6f}、n {st['n']}")
    check("≥ 門檻比例逐值精確（cos 0.3＝0.955 → ≥0.95 全部、≥0.99 沒有）", st["frac_ge_0.95"] == 1.0 and st["frac_ge_0.99"] == 0.0)
    check("角距離（MiniCache 的 d＝arccos/π）＝α/π", abs(st["mean_angdist"] - alpha / math.pi) < 1e-9,
          f"{st['mean_angdist']:.6f} vs {alpha / math.pi:.6f}")
    mix = np.concatenate([np.full(300, 0.97), np.full(700, 0.5)])
    acc2 = SimAcc(1, [0.95])
    acc2.add(mix[None], np.ones((1, 1000), bool))
    st2 = acc2.stats(0)
    check("已知混合：≥0.95 比例＝0.3、p25≈0.5、p95≈0.97", abs(st2["frac_ge_0.95"] - 0.3) < 1e-12 and abs(st2["p25"] - 0.5) <= 0.001
          and abs(st2["p95"] - 0.97) <= 0.001, f"{st2['p25']:.4f}、{st2['p95']:.4f}")
    Zs = rng.standard_normal((5, D2))
    Zs[2] = 0
    cz, vz = cos_rows(Zs[None, :-1], Zs[None, 1:])
    acc3 = SimAcc(1, [0.9])
    acc3.add(cz, vz)
    check("長度 0 的向量：不算 cosine、記在 n_zero", acc3.stats(0)["n"] == 2 and acc3.stats(0)["n_zero"] == 2)
    A_, B_ = rng.standard_normal((4, 50, D)), rng.standard_normal((4, 50, D))
    cpre, _ = cos_rows(A_, B_)
    pp_ = rng.integers(0, 100000, 50)
    cq, sq = rope_cos_sin(f, pp_)
    cpost, _ = cos_rows(rope_apply64(A_, cq, sq), rope_apply64(B_, cq, sq))
    check("跨層 cosine 在 RoPE 前後相同（同位置同旋轉）", float(np.max(np.abs(cpre - cpost))) < 1e-2,
          f"max 差 {np.max(np.abs(cpre - cpost)):.2e}（BF16 的 c²+s²≠1 造成）")
    c64q, s64q = np.cos(pp_[:, None] * f[None].astype(np.float64)), np.sin(pp_[:, None] * f[None].astype(np.float64))
    cpost64, _ = cos_rows(rope_apply64(A_, c64q, s64q), rope_apply64(B_, c64q, s64q))
    check("（精確 cos/sin）跨層 cosine 不變 ≤ 1e-12", float(np.max(np.abs(cpre - cpost64))) < 1e-12)
    # ---- 假 dump：探針、資料核對、門檻、端到端
    with tempfile.TemporaryDirectory() as td:
        d_ok = os.path.join(td, "ok")
        truth = make_fake_dump(d_ok)
        dp = Dump(d_ok)
        rc, rope = rope_check(dp)
        p = rc["probe"]
        check("探針 (a) dump＝RoPE 後探針（逐位元）", p["a_dump_match"]["ok"] and p["a_dump_match"]["max_ulp"] == 0)
        check("探針 (b) cos/sin 模擬＝探針", p["b_cos_sin"]["ok"] and p["b_cos_sin"]["max_ulp"] == 0)
        check("探針 (c) 前向模擬＝探針", p["c_forward_emul"]["ok"] and p["c_forward_emul"]["max_ulp"] == 0)
        check("探針 (d) 逆旋轉≈RoPE 前（≤0.02）", p["d_inverse"]["ok"], f"max {p['d_inverse']['max_rel_err']:.2e}")
        check("資料自身核對 PASS（期望的 M 最大）", rc["data"]["status"] == "PASS",
              "M=" + json.dumps({k: round(v, 4) for k, v in rc["data"]["M"].items()}))
        check("RoPE 核對總狀態 PASS_PROBE", rc["status"] == "PASS_PROBE", str(rc["warnings"]))
        d_np = os.path.join(td, "noprobe")
        make_fake_dump(d_np, probe=False, meta_rope=False)
        dpn = Dump(d_np)
        rcn, _ = rope_check(dpn, model_config=_write_cfg(td, cfg_llama))
        check("沒有探針、沒有 meta.rope（舊 dump）：用 config 公式＋資料核對 → PASS_DATA", rcn["status"] == "PASS_DATA",
              rcn["rope"].get("inv_freq_source", ""))
        d_bad = os.path.join(td, "bad")
        make_fake_dump(d_bad, rotate=False)                                 # dump 其實是 RoPE 前，但 meta 說是 RoPE 後
        rcb, _ = rope_check(Dump(d_bad))
        check("門檻：dump 其實沒轉 → 探針 (a) 抓到 → FAIL", rcb["status"] == "FAIL" and rcb["probe"]["a_dump_match"]["ok"] is False)
        d_bad2 = os.path.join(td, "bad2")
        make_fake_dump(d_bad2, rotate=False, probe=False)
        rcb2, _ = rope_check(Dump(d_bad2))
        check("門檻：沒探針、dump 其實沒轉 → 資料核對抓到 → FAIL", rcb2["status"] == "FAIL" and rcb2["data"]["status"] == "FAIL",
              "M=" + json.dumps({k: round(v, 4) for k, v in rcb2["data"]["M"].items()}))
        d_wt = os.path.join(td, "wrongtheta")
        make_fake_dump(d_wt, probe=False, params=dict(LLAMA3, rope_theta=1000000.0), meta_rope=False)  # 真的 θ＝1e6，config 說 5e5
        rcw, _ = rope_check(Dump(d_wt), model_config=_write_cfg(td, cfg_llama))
        check("門檻：沒探針、θ 錯（dump 用 1e6、config 說 5e5）→ FAIL", rcw["status"] == "FAIL",
              "M=" + json.dumps({k: round(v, 4) for k, v in rcw["data"]["M"].items()}))
        # 端到端 analyze
        out = os.path.join(td, "out")
        ns = argparse.Namespace(dump=[d_ok], out=out, layers=None, heads=None, max_blocks=0, calib_tokens=128,
                                ranks=[2, 4, 8], thresholds=[0.75, 0.8, 0.9, 0.95, 0.99], joint=True, no_pre_rope=False,
                                centered=True,
                                spectra=os.path.join(td, "sp.npz"), max_mem_gb=16.0, model_config=None, data_blocks=16,
                                skip_first_tokens=0)
        os.environ["RUN_ID"] = "SELFTEST"
        rcode = cmd_analyze(ns)
        check("analyze 跑完（exit 0）", rcode == 0)
        rows = list(csv.DictReader(open(os.path.join(out, "lowrank.csv"))))
        eff = list(csv.DictReader(open(os.path.join(out, "lowrank_eff.csv"))))
        adjr = list(csv.DictReader(open(os.path.join(out, "adjtok.csv"))))
        xlr = list(csv.DictReader(open(os.path.join(out, "xlayer.csv"))))
        allrows = rows + eff + adjr + xlr
        check("每個 CSV 都有 run_id、ts，且 run_id＝SELFTEST", all(r.get("run_id") == "SELFTEST" and r.get("ts") for r in allrows),
              f"{len(rows)}／{len(eff)}／{len(adjr)}／{len(xlr)} 列")

        def g(kind, layer, head, rank, f="rel_err"):
            return [float(r[f]) for r in rows if r["kind"] == kind and r["layer"] == str(layer) and r["head"] == str(head)
                    and r["rank"] == str(rank)][0]
        e_pre4 = max(g("K_pre", L, h, 4) for L in (0, 1, 2, 4) for h in (0, 1))
        e_post4 = min(g("K_post", L, h, 4) for L in (0, 1, 2, 4) for h in (0, 1))
        check("合成 K（RoPE 前＝平均＋秩 3）：K_pre rank 4 誤差 ≤ 0.01（BF16 雜訊）", e_pre4 <= 0.01, f"{e_pre4:.4f}")
        check("RoPE 後秩變高：K_post rank 4 誤差 > K_pre", e_post4 > e_pre4 + 0.05, f"K_post 最小 {e_post4:.4f}")
        e_ctr3 = max(g("K_pre", L, h, 4, "rel_err_centered") for L in (0, 1, 2, 4) for h in (0, 1))
        check("centered：K_pre rank 4 的誤差也 ≤ 0.01", e_ctr3 <= 0.01, f"{e_ctr3:.4f}")
        kx = [float(r["xfer_err"]) for r in rows if r["kind"] == "K_pre" and r["rank"] == "4" and r["head"] != "all"]
        check("K_pre 校準→測試 rank 4 誤差 ≤ 0.01（子空間全段相同）", max(kx) <= 0.01, f"{max(kx):.4f}")
        va = [float(r["rel_err"]) for r in rows if r["kind"] == "V" and r["rank"] == "4" and r["head"] != "all" and r["layer"] != "4"]
        check("V（秩 4）rank 4 誤差 ≤ 0.01", max(va) <= 0.01, f"{max(va):.4f}")
        ja = [r for r in rows if r["head"] == "all"]
        check("整層串接（head=all）有列、rank＝r×H", len(ja) > 0 and sorted({int(r["rank"]) for r in ja}) == [4, 8, 16])

        def xm(kind, L, head):
            return [float(r["mean"]) for r in xlr if r["kind"] == kind and r["layer"] == str(L) and r["head"] == str(head)][0]
        check("跨層：1→2（擾動 0.01）cosine 平均 > 0.98；0→1（獨立）|cos| < 0.5",
              min(xm("K", 1, h) for h in (0, 1)) > 0.98 and min(xm("V", 1, h) for h in (0, 1)) > 0.98
              and max(abs(xm("K", 0, h)) for h in (0, 1)) < 0.5, f"K1→2 {xm('K', 1, 0):.4f}、K0→1 {xm('K', 0, 0):.4f}")
        s1 = json.load(open(os.path.join(out, "summary.json")))["dumps"][0]
        check("旁證：跨層 K cosine 的 RoPE 前後最大差 < 0.01", s1["xlayer_K_pre_vs_post_max_abs_diff"] < 0.01,
              f"{s1['xlayer_K_pre_vs_post_max_abs_diff']:.2e}")
        check("跨層：缺 l+1 的配對沒有列（2→3、4→5 → NOT_MEASURED）",
              not any(r["layer"] in ("2", "4") for r in xlr)
              and "2→3" in json.load(open(os.path.join(out, "summary.json")))["dumps"][0]["xlayer_not_measured"])
        check("中心化跨層：1→2（共用起伏）K_centered、V_centered > 0.98；0→1 |cos| < 0.5",
              min(xm("K_centered", 1, h) for h in (0, 1)) > 0.98 and min(xm("V_centered", 1, h) for h in (0, 1)) > 0.98
              and max(abs(xm("K_centered", 0, h)) for h in (0, 1)) < 0.5,
              f"K_c1→2 {xm('K_centered', 1, 0):.4f}、K_c0→1 {xm('K_centered', 0, 0):.4f}")
        kc = [r for r in adjr if r["kind"] == "K_pre_centered" and r["layer"] == "0" and r["head"] == "pooled"][0]
        kr = [r for r in adjr if r["kind"] == "K_pre" and r["layer"] == "0" and r["head"] == "pooled"][0]
        check("中心化相鄰 token：K_pre 的相似只來自平均方向 → 中心化後 |平均| < 0.1（未中心化 > 0.3）",
              abs(float(kc["mean"])) < 0.1 and float(kr["mean"]) > 0.3, f"{float(kc['mean']):.4f} vs {float(kr['mean']):.4f}")
        vadj = [r for r in adjr if r["kind"] == "V" and r["layer"] == "4" and r["head"] == "pooled"][0]
        kadj = [r for r in adjr if r["kind"] == "K_pre" and r["layer"] == "0" and r["head"] == "pooled"][0]
        check("相鄰 token：漂移序列的 V ≥0.9 比例 > 0.9；n＝(T−1)×H（block 邊界有算）",
              float(vadj["frac_ge_0.9"]) > 0.9 and int(vadj["n"]) == (6 * 64 - 1) * 2, f"{vadj['frac_ge_0.9']}、n {vadj['n']}")
        check("相鄰 token：K_pre（平均方向大）比 K_post 像", float(kadj["mean"]) > float(
            [r for r in adjr if r["kind"] == "K_post" and r["layer"] == "0" and r["head"] == "pooled"][0]["mean"]))
        md = open(os.path.join(out, "summary.md")).read()
        check("summary.md 有 RoPE 核對、三種量測的表", all(s_ in md for s_ in ("RoPE 核對", "K_pre", "跨層相似", "相鄰 token")))
        sp = np.load(os.path.join(td, "sp.npz"))
        check("--spectra 存了每個（層, 種類, head）的特徵值", len(sp.files) == 4 * 3 * 3 and sp[sp.files[0]].shape[0] in (32, 64))
        # 門檻端到端：RoPE 核對不過 → exit 2、不寫 lowrank.csv
        out2 = os.path.join(td, "out2")
        rc2 = cmd_analyze(argparse.Namespace(**dict(vars(ns), dump=[d_bad], out=out2, spectra=None)))
        check("analyze：RoPE 核對不過 → exit 2、沒有 lowrank.csv", rc2 == 2 and not os.path.exists(os.path.join(out2, "lowrank.csv")))
        out3 = os.path.join(td, "out3")
        rc3 = cmd_analyze(argparse.Namespace(**dict(vars(ns), dump=[d_bad], out=out3, spectra=None, no_pre_rope=True)))
        r3 = list(csv.DictReader(open(os.path.join(out3, "lowrank.csv"))))
        check("--no-pre-rope：其餘照跑、沒有 K_pre 列", rc3 == 0 and r3 and not any(r["kind"] == "K_pre" for r in r3))
        # 分層／head 抽樣、block 抽樣
        out4 = os.path.join(td, "out4")
        rc4 = cmd_analyze(argparse.Namespace(**dict(vars(ns), out=out4, spectra=None, layers=[1, 2], heads=[1], max_blocks=2)))
        s4 = json.load(open(os.path.join(out4, "summary.json")))["dumps"][0]
        check("抽樣：--layers 1 2 --heads 1 --max-blocks 2 → 只用那些", rc4 == 0 and s4["layers"] == [1, 2] and s4["heads"] == [1]
              and s4["n_blocks_used"] == 2 + 2, f"blocks {s4['n_blocks_used']}")
        out5 = os.path.join(td, "out5")
        rc5 = cmd_analyze(argparse.Namespace(**dict(vars(ns), out=out5, spectra=None, skip_first_tokens=3)))
        r5 = list(csv.DictReader(open(os.path.join(out5, "lowrank.csv"))))
        a5 = list(csv.DictReader(open(os.path.join(out5, "adjtok.csv"))))
        check("--skip-first-tokens 3：token 數＝T−3、相鄰配對＝(T−4)×H、校準少 3 個",
              rc5 == 0 and {r["n_tokens"] for r in r5} == {str(6 * 64 - 3)} and {r["n_calib"] for r in r5} == {str(128 - 3)}
              and [r["n"] for r in a5 if r["head"] == "pooled"][0] == str((6 * 64 - 4) * 2))
        check("記憶體估計（Llama 32 層、joint）< 16 GiB", mem_estimate_gb(32, 3, 8, 128, True, 31, 32, 512) < 16,
              f"{mem_estimate_gb(32, 3, 8, 128, True, 31, 32, 512):.2f} GiB")
    os.environ.pop("RUN_ID", None)
    assert_no_gpu_libs()
    check("全程沒有 import torch／vllm／transformers", not any(m_ in sys.modules for m_ in FORBIDDEN))
    print(f"{n_items} 項；" + ("ALL PASS" if ok else "SOME FAILED"))
    return ok


def _write_cfg(td: str, cfg: dict) -> str:
    p = os.path.join(td, f"config_{abs(hash(json.dumps(cfg, sort_keys=True)))}.json")
    json.dump(cfg, open(p, "w"))
    return p


def main():
    assert_no_gpu_libs()
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("ropecheck")
    p.add_argument("--dump", nargs="+", required=True)
    p.add_argument("--model-config", help="config.json（舊 dump 的 meta 沒有 rope、且 meta.model_path 找不到時）")
    p.add_argument("--data-blocks", type=int, default=16)
    p = sub.add_parser("analyze")
    p.add_argument("--dump", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--layers", nargs="*", type=int, help="只看這些層（模型的層號）；預設 dump 裡全部")
    p.add_argument("--heads", nargs="*", type=int, help="只看這些 KV head；預設全部")
    p.add_argument("--max-blocks", type=int, default=0, help="校準之後的 block 平均取幾個；0＝全部")
    p.add_argument("--calib-tokens", type=int, default=4096, help="前幾個 token 當 PCA 校準集（chunk 的倍數）")
    p.add_argument("--ranks", nargs="+", type=int, default=[8, 16, 32, 64, 96])
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.75, 0.8, 0.9, 0.95, 0.99],
                   help="≥ 門檻的比例（0.75＝KVMerger 的合併門檻；0.9、0.95＝計畫）")
    p.add_argument("--no-centered", dest="centered", action="store_false",
                   help="不做第二輪（中心化的相鄰 token／跨層 cosine）")
    p.add_argument("--no-joint", dest="joint", action="store_false", help="不算整層 head 串接的低秩（省記憶體與時間）")
    p.add_argument("--no-pre-rope", action="store_true", help="不做 RoPE 核對、不算 RoPE 前的 K")
    p.add_argument("--spectra", help="完整特徵值存成 npz（大；放 runs 目錄，不進 git）")
    p.add_argument("--max-mem-gb", type=float, default=16.0)
    p.add_argument("--skip-first-tokens", type=int, default=0,
                   help="不算位置 < k 的 token（看 attention sink 有沒有主導低秩能量；預設 0＝全部都算）")
    p.add_argument("--model-config")
    p.add_argument("--data-blocks", type=int, default=16, help="RoPE 資料核對用幾個 block（平均取樣）")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if a.cmd == "ropecheck":
        sys.exit(cmd_ropecheck(a))
    elif a.cmd == "analyze":
        sys.exit(cmd_analyze(a))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
