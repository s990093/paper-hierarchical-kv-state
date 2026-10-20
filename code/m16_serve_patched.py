#!/usr/bin/env python3
"""m16_serve_patched.py — `vllm serve` 外面包一層：在 server 的每個行程裝上 F1 的 ROCm attention 修法，再執行 vLLM CLI。

只給 `m16_vllm_bench.py serve --rocm-attn-patch` 組出來的指令用：
    <venv>/bin/python code/m16_serve_patched.py serve <model> ... --attention-backend ROCM_ATTN --kv-transfer-config ...
⚠️ 這支會 import torch／vllm（會打開 /dev/kfd）。**client、測試、dry-run 都不准 import 它**；CPU 測試只用 compile() 檢查語法。

修法（照 code/m9_f1_run.py:72-107 的 cpu50_rocmfix，原理與驗證見 docs/research_20261010_followup/F1_offload_slowdown.md §3–4）：
  1. vLLM 0.28 在 ROCm 上只要有 KV connector 就把 ROCM_ATTN 排除（v1/attention/backends/rocm_attn.py:211-215
     supports_kv_connector() 回 False；v1/attention/backend.py:405-406），退回慢約 2.4 倍的 TRITON_ATTN
     （33K prefill 的 attention 1.367→3.277 s，RUNLOG_MI300X:176）。這裡讓它回 True，並由 serve 指定 --attention-backend ROCM_ATTN
     （platforms/rocm.py:628-642 走「指定 backend」那條路，會印 "Using ROCM_ATTN backend (selected via --attention-backend)."）。
  2. ROCM_ATTN 的 KV 每層是 (2, num_blocks, …)：不拆的話 connector 會把「第 2i、2i+1 個 block 的 K」當成第 i 個 block 搬
     （F1 負對照 0／512 block 正確、輸出被弄壞）。所以把每層拆成 K、V 兩個 blocks-first 的 view 註冊（F1：511／512 block
     位元組相同，和上游 TRITON 路徑一樣）。代價：每個 block 每層的 copy 次數變 2 倍（F1 §4.4：33K 從 CPU 載入 1.092→2.049 s）。

為什麼在模組最上層就裝：vLLM 的 EngineCore 是用 multiprocessing 開的子行程（v1/engine/utils.py:139-165）。
  fork：子行程直接繼承已經改好的類別；spawn：multiprocessing 會把這支主程式以 __mp_main__ 的名字重新 import，
  最上層的 _install() 會再跑一次。兩種都會在 EngineCore 裡生效。每個行程裝好會在 log 印一行 M16_ROCMFIX。
"""
import os
import sys


def _install() -> None:
    import torch
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading import worker as owk
    from vllm.v1.attention.backends import rocm_attn
    from vllm.v1.kv_offload.base import CanonicalKVCacheRef, CanonicalKVCaches, CanonicalKVCacheTensor

    rocm_attn.RocmAttentionBackend.supports_kv_connector = classmethod(lambda cls: True)
    _reg = owk.OffloadingConnectorWorker.register_kv_caches
    if getattr(_reg, "_m16_patched", False):
        return

    def reg(self, kv_caches):
        nb = self.kv_cache_config.num_blocks
        first = next(iter(kv_caches.values()))
        if not (isinstance(first, torch.Tensor) and first.dim() == 5 and first.shape[0] == 2 and first.shape[1] == nb):
            print(f"M16_ROCMFIX: layout not KV-outer {tuple(first.shape)} → 用上游的註冊", file=sys.stderr, flush=True)
            return _reg(self, kv_caches)
        groups = self.kv_cache_config.kv_cache_groups
        assert len(groups) == 1, "M16_ROCMFIX 只處理單一 KV group（Llama／Qwen2.5 都是）"
        tensors, refs, seen = [], [], {}
        for ln in groups[0].layer_names:
            kv = kv_caches[ln]
            key = kv.data_ptr()
            if key in seen:   # 共用 KV 的層：指到同一組 tensor
                refs += [CanonicalKVCacheRef(tensor_idx=i, page_size_bytes=tensors[i].page_size_bytes) for i in seen[key]]
                continue
            idx = []
            for half in (kv[0], kv[1]):   # K、V 各自是 (num_blocks, block, heads, dim)，連續
                assert half.is_contiguous()
                t = half.view(-1).view(torch.int8).view(nb, -1)
                tensors.append(CanonicalKVCacheTensor(tensor=t, page_size_bytes=t.shape[1]))
                idx.append(len(tensors) - 1)
                refs.append(CanonicalKVCacheRef(tensor_idx=len(tensors) - 1, page_size_bytes=t.shape[1]))
            seen[key] = idx
        print(f"M16_ROCMFIX: registered {len(tensors)} K/V tensors, page {tensors[0].page_size_bytes} B, "
              f"pid={os.getpid()}", file=sys.stderr, flush=True)
        self._init_worker(CanonicalKVCaches(tensors=tensors, group_data_refs=[refs]))
    reg._m16_patched = True
    owk.OffloadingConnectorWorker.register_kv_caches = reg
    print(f"M16_ROCMFIX: patch installed pid={os.getpid()} name={__name__}", file=sys.stderr, flush=True)


_install()

if __name__ == "__main__":
    from vllm.entrypoints.cli.main import main
    sys.argv = ["vllm"] + sys.argv[1:]
    sys.exit(main())
