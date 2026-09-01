#!/usr/bin/env python3
"""headroom 的峰值隨硬體移動——論文 κ 主張的量化版本。

## 問題

3090 上的實測顯示 headroom 在請求長度 128–256K 見頂，512K 反而下降。
那是不是代表「512K 沒有價值」？不是——是**這張卡上**沒有。

## 機制

三個動作的成本結構決定一切：

    放 CPU    固定
    放 SSD    固定
    丟掉重算  base + slope × 位置        <- 只有這個隨位置成長

    交叉點 = (SSD 成本 − 重算 base) / 重算 slope

交叉點以前該重算，以後該放磁碟。**headroom 的峰值落在交叉點的 2–3.5 倍**
——那是「同一個請求裡，前半該重算、後半該放磁碟」的長度，決策才有價值。
更短則全部該重算、更長則全部該放磁碟，兩種情況都沒得選。

GPU 越快 -> 重算越便宜 -> slope 越小 -> **交叉點越往後** -> 峰值跟著移。

## 結果（壓力固定 5×、重用 81%）

                    交叉點      64K    128K    256K    512K    768K
    3090（實測）     37,717    0.0%   17.5%   18.7%   15.5%    8.7%
    快 2 倍          75,435    2.7%   23.8%   28.3%   25.2%   16.4%
    快 4 倍(A100)   150,870    5.1%   20.6%   37.5%   34.7%   25.0%
    快 6 倍(H100)   226,305    5.8%   18.9%   34.7%   39.5%   29.7%

**交叉點隨成本模型變動，此表非常數。**上表為 2026-09-01 重跑，
用的是含 `20260901-131533-m2-recompute`（位置掃到 258,048）的擬合。
先前的 37,615 來自只含 `20260831-223228` 的擬合。峰值百分比對這次
重擬合不敏感（三格差 0.1pp），但引用交叉點的絕對值時必須標明版本。

**峰值從 3090 的 256K 移到 H100 的 512K，數值從 18.7% 升到 39.5%。**

## 這是可證偽的預測

「快 N 倍」是把實測的 slope 除以 N，不是量測。借到 A100 跑一次就能驗證。
論文引用時必須標明這一點。

用法：python code/m4_hw_sweep.py
"""
import dataclasses, sys
sys.path.insert(0,'code')
from m4_oracle import BLOCK, Sim, load_cost_model, longctx_trace, profile
prof = profile("qwen-awq"); cm0 = load_cost_model("nvme", require_model_key="qwen-awq")
bpb = prof["kv_bytes_per_token"]*BLOCK
P = {"full_gpu":("lru",0,0),"cpu_lru":("lru",1,0),"cpu_arc":("arc",1,0),"tier_fs":("lru",1,1)}
SEM = dict(prefix_semantics=True, prefetch=True)
def run(L, mul, pressure=5.0, nreq=40):
    cm = dataclasses.replace(cm0, recompute_slope_per_token=cm0.recompute_slope_per_token*mul)
    doc_b=L//BLOCK; tail_b=max(1,int(doc_b*0.02))
    tr = longctx_trace(8, doc_b, tail_b, nreq, 0.9, 1234)
    uniq=len({b for r in tr for b in r}); gb=max(1,int(uniq/pressure))
    sim = Sim(cm, gb, int(24*1024**3)//bpb, int(512*1024**3)//bpb)
    res={k:sim.run_online(tr,*v,**SEM) for k,v in P.items()}
    res["oracle"]=sim.run_oracle(tr,True,True,**SEM)
    b=min((k for k in res if k!="oracle"), key=lambda k:res[k]["total_ms"])
    return 100*(res[b]["total_ms"]-res["oracle"]["total_ms"])/res[b]["total_ms"]
Ls=[65536,131072,262144,524288,786432]
print("壓力固定 5×、重用 81%（40 請求 × 8 文件），只變「prefill 有多快」\n")
print(f"{'':30s}" + "".join(f"{L//1024:>8d}K" for L in Ls))
for lab,mul in (("3090（實測）",1.0),("快 2 倍（5090 級）",0.5),
                ("快 4 倍（A100 級）",0.25),("快 6 倍（H100 級）",1/6)):
    xo=(cm0.ssd-cm0.recompute_base)/(cm0.recompute_slope_per_token*mul)
    row=f"{lab:16s}交叉 {xo:>7,.0f}  "
    for L in Ls:
        row += f"{run(L,mul):>8.1f}%"
    print(row, flush=True)
