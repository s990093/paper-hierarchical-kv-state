#!/usr/bin/env python3
"""數值上分離「格式」與「縮放」對 KV 量化誤差的貢獻。

## 為什麼需要這支程式

大海撈針實測（ctx=32,768、20 樣本／設定）：

    BF16 100%　FP8（靜態未校正）5%　INT8（per-token-head 動態）95%　INT4 0%

我一開始把它解讀成「差別在縮放係數，不在位元寬」，並據此寫進 RUNLOG。
**那個解讀是錯的**——它建立在一個我做不到的比較上：
能分離兩個變數的 `fp8_per_token_head` 在 3090 上跑不起來
（ValueError: FP8 KV cache is not supported by the Triton attention backend
on compute capability 8.6）。

vLLM 跑不了，但量化誤差本身用 torch 算得出來。

## 結果（見 __main__ 的輸出）

    FP8  靜態 scale=1.0        2.64%
    FP8  per-token-head 動態   2.56%   <- 動態縮放幾乎沒幫上忙
    INT8 per-token-head 動態   0.65%   <- 好 4 倍

**格式才是主因。** FP8 e4m3 只有 3 個尾數位元，每個數量級 8 個刻度，
相對精度固定在 ~12.5%，縮放改變不了這件事。
INT8 在指定範圍內有 255 個刻度，範圍抓得準精度就是 1/255。

FP8 拿精度換動態範圍（0.0156–448，跨 4 個數量級），
但 KV 在同一個 token、同一個 head 內值域本來就窄——這是虧本的交換。

理論驗算（兩者都吻合）：
    FP8  3 尾數位元 -> 刻度間距 2^-3，RMS 相對誤差 ≈ 12.5%/√12 = 3.6%（實測 2.64%）
    INT8 255 刻度、max 縮放 -> 對高斯資料 ≈ 3/440 = 0.68%（實測 0.65%）

靜態縮放是**額外**的一層傷害，只打到值域小的 head：
值域 ±0.015 的 head 掉進 FP8 的次正規區（< 0.0156），誤差 11.29%（其餘 ~2.65%）。

## 保留條件

用高斯合成資料。真實 KV 有已知的離群值現象，實際數字可能不同。
本程式證明的是**機制**（格式的尾數位元數主導），不是精確的誤差值。

用法：python code/quant_error.py
"""
from __future__ import annotations
import argparse

import torch


def rel_err(y: torch.Tensor, x: torch.Tensor) -> float:
    return ((y - x).norm() / x.norm() * 100).item()


def fp8_static(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """vLLM 的 `--kv-cache-dtype fp8` 在 checkpoint 沒有校正過的 k/v scale 時。"""
    return (x / scale).to(torch.float8_e4m3fn).float() * scale


def fp8_dynamic(x: torch.Tensor) -> torch.Tensor:
    """`fp8_per_token_head`：每 token 每 head 各自算 scale。sm_86 跑不了。"""
    s = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / torch.finfo(
        torch.float8_e4m3fn).max
    return (x / s).to(torch.float8_e4m3fn).float() * s


def int8_dynamic(x: torch.Tensor) -> torch.Tensor:
    """`int8_per_token_head`：每 token 每 head 各自算 scale。"""
    s = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 127.0
    return torch.round(x / s).clamp(-127, 127) * s


def int4_dynamic(x: torch.Tensor) -> torch.Tensor:
    """`int4_per_token_head`。"""
    s = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 7.0
    return torch.round(x / s).clamp(-7, 7) * s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--head-scales", type=float, nargs="*",
                    default=[0.005, 0.05, 0.5, 5.0],
                    help="每個 head 的值域尺度。真實模型的 head 之間差很多，"
                         "這正是靜態縮放失效的地方")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    sc = torch.tensor(a.head_scales).view(1, -1, 1)
    x = torch.randn(a.tokens, len(a.head_scales), a.dim) * sc

    methods = [
        ("FP8  靜態 scale=1.0（vLLM 未校正時）", fp8_static),
        ("FP8  per-token-head 動態（sm_86 跑不了）", fp8_dynamic),
        ("INT8 per-token-head 動態", int8_dynamic),
        ("INT4 per-token-head 動態", int4_dynamic),
    ]
    print(f"資料：{a.tokens} tokens × {len(a.head_scales)} heads × {a.dim} dim，"
          f"各 head 的尺度 {a.head_scales}\n")
    print(f"{'量化方式':42s}{'相對誤差':>10s}")
    for name, fn in methods:
        print(f"{name:42s}{rel_err(fn(x), x):>9.2f}%")

    print(f"\n靜態縮放的誤差按 head 分開看（次正規區的傷害）：")
    y = fp8_static(x)
    print(f"{'head':>6s}{'值域':>12s}{'誤差':>10s}")
    for i, s in enumerate(a.head_scales):
        e = ((y[:, i] - x[:, i]).norm() / x[:, i].norm() * 100).item()
        print(f"{i:>6d}{'±' + f'{s * 3:.3f}':>12s}{e:>9.2f}%")
    fi = torch.finfo(torch.float8_e4m3fn)
    print(f"\nFP8 e4m3：最小正規數 {fi.smallest_normal:.5f}、最大 {fi.max}")
    print("→ 值域小於最小正規數的 head 掉進次正規區，可表示的值極少")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
