#!/usr/bin/env python3
"""
gpu_smoke.py — 證明 GPU 是「真的能算」而不是「偵測得到」

偵測得到 ≠ 能用。這支做四件事：
  1. 真的做 matmul（算力）
  2. 檢查數值正確性（不是只看有沒有 crash）
  3. 量 HBM device-to-device 頻寬
  4. 量 H2D / D2H pinned 頻寬 ← 論文 GPU↔CPU 階的成本來源

⚠️ 這裡量到的頻寬是 smoke test 等級，**不是論文數字**。
   論文的 M2 成本模型要走 runsh 並重複多次取分佈。
"""
import time
import torch

print("=" * 66)
print(f"torch {torch.__version__} | hip {torch.version.hip} | cuda {torch.version.cuda}")
assert torch.version.hip is not None, "不是 ROCm build"
assert torch.cuda.is_available(), "看不到 GPU"
d = torch.device("cuda:0")
p = torch.cuda.get_device_properties(0)
print(f"device: {p.name}  {p.total_memory / 2**30:.1f} GiB  arch={p.gcnArchName}")
print("=" * 66)

# 1) 算力
N = 8192
a = torch.randn(N, N, device=d, dtype=torch.bfloat16)
b = torch.randn(N, N, device=d, dtype=torch.bfloat16)
for _ in range(3):
    c = a @ b                      # warmup
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    c = a @ b
torch.cuda.synchronize()
dt = (time.time() - t0) / 20
print(f"[matmul  ] bf16 {N}^3   {dt*1000:8.2f} ms/iter   {2*N**3/dt/1e12:7.1f} TFLOPS"
      f"   (原廠標稱 BF16 1307.4)")

# 2) 數值正確性
x = torch.eye(1024, device=d, dtype=torch.float32)
y = torch.randn(1024, 1024, device=d, dtype=torch.float32)
err = (x @ y - y).abs().max().item()
print(f"[correct ] max|I@Y - Y| = {err:.3e}   -> {'OK' if err < 1e-4 else 'FAIL'}")
assert err < 1e-4, "數值不對，不要用這台機器量任何東西"

# 3) HBM d2d
g  = torch.empty(256 * 1024 * 1024, device=d, dtype=torch.float32)   # 1 GiB
g2 = torch.empty_like(g)
torch.cuda.synchronize(); t0 = time.time()
for _ in range(10):
    g2.copy_(g)
torch.cuda.synchronize()
print(f"[HBM d2d ] {10*2*g.numel()*4/(time.time()-t0)/1e9:8.1f} GB/s"
      f"   (原廠標稱 5300)")

# 4) H2D / D2H pinned —— 論文 CPU offload 階的成本來源
h = torch.empty(256 * 1024 * 1024, dtype=torch.float32, pin_memory=True)
torch.cuda.synchronize(); t0 = time.time()
for _ in range(10):
    g.copy_(h)
torch.cuda.synchronize()
print(f"[H2D pin ] {10*h.numel()*4/(time.time()-t0)/1e9:8.1f} GB/s   (PCIe Gen5 x16 理論 ~63)")
torch.cuda.synchronize(); t0 = time.time()
for _ in range(10):
    h.copy_(g)
torch.cuda.synchronize()
print(f"[D2H pin ] {10*h.numel()*4/(time.time()-t0)/1e9:8.1f} GB/s")

print(f"[mem     ] allocated={torch.cuda.memory_allocated()/2**30:.1f} GiB  "
      f"reserved={torch.cuda.memory_reserved()/2**30:.1f} GiB")
print("=" * 66)
print("GPU 可用：算得出來、數值正確、搬得動資料。")
