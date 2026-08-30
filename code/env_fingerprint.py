#!/usr/bin/env python3
"""環境指紋 — EXPERIMENT_PLAN.md 驗收 A1。

產出 results/env.json。

⚠️ 這個腳本刻意**跑一個真的 CUDA kernel**，不只查 `torch.cuda.is_available()`。
   2026-08-30 的實測顯示：vLLM 的 CUDA-13 wheel 在 driver 550（CUDA 12.4）上
   `is_available()` 會回傳 True、`device_count()` 回傳 7，但任何 kernel 都會爆
   `RuntimeError: The NVIDIA driver on your system is too old`。
   **「CUDA 可用」的唯一合格判準是跑成一個 kernel。** 見 results/RUNLOG.md §0.5。

用法:
    python code/env_fingerprint.py [-o results/env.json]
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sh(cmd: list[str]) -> str | None:
    """跑一條指令，回傳 stdout（strip 過）。失敗回 None，不 raise。"""
    if not shutil.which(cmd[0]):
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (subprocess.TimeoutExpired, OSError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def gpus() -> list[dict]:
    q = "index,name,memory.total,compute_cap,driver_version,pci.bus_id"
    raw = sh(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"])
    if not raw:
        return []
    out = []
    for line in raw.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 6:
            continue
        out.append(
            {
                "index": int(f[0]),
                "name": f[1],
                "vram_total_mb": int(f[2].removesuffix(" MiB")),
                "compute_capability": f[3],
                "driver": f[4],
                "pci_bus_id": f[5],
            }
        )
    return out


def torch_info() -> dict:
    """torch/CUDA 資訊 + 真實 kernel 驗證。"""
    info: dict = {
        "torch_version": None,
        "torch_built_for_cuda": None,
        "cuda_is_available": None,
        "device_count": None,
        "real_kernel_ok": False,          # ← 唯一有意義的那一欄
        "real_kernel_error": None,
    }
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        info["real_kernel_error"] = f"import torch failed: {type(e).__name__}: {e}"
        return info

    info["torch_version"] = torch.__version__
    info["torch_built_for_cuda"] = torch.version.cuda
    try:
        info["cuda_is_available"] = bool(torch.cuda.is_available())
        info["device_count"] = int(torch.cuda.device_count())
    except Exception as e:  # noqa: BLE001
        info["real_kernel_error"] = f"probe failed: {type(e).__name__}: {e}"
        return info

    # 這才是驗收：實際配置記憶體並跑一個 matmul。
    try:
        x = torch.randn(1024, 1024, device="cuda")
        checksum = float((x @ x).sum())
        info["real_kernel_ok"] = True
        info["real_kernel_checksum"] = checksum
        info["device0_name"] = torch.cuda.get_device_name(0)
        info["device0_capability"] = list(torch.cuda.get_device_capability(0))
        info["device0_total_mem_gib"] = round(
            torch.cuda.get_device_properties(0).total_memory / 2**30, 3
        )
    except Exception as e:  # noqa: BLE001
        info["real_kernel_error"] = f"{type(e).__name__}: {e}"
    return info


def vllm_info() -> dict:
    info: dict = {"vllm_version": None, "offloading_specs": None,
                  "cache_policies": None, "secondary_tiers": None, "error": None}
    try:
        import vllm

        info["vllm_version"] = vllm.__version__
    except Exception as e:  # noqa: BLE001
        info["error"] = f"import vllm failed: {type(e).__name__}: {e}"
        return info

    # 記錄「計畫假設的東西實際存不存在」——A3 之前先靜態確認。
    try:
        from vllm.v1.kv_offload.factory import OffloadingSpecFactory

        info["offloading_specs"] = sorted(OffloadingSpecFactory._registry)
    except Exception as e:  # noqa: BLE001
        info["offloading_specs"] = f"ERR {type(e).__name__}: {e}"
    try:
        from vllm.v1.kv_offload.cpu.policies.factory import CachePolicyFactory

        info["cache_policies"] = sorted(CachePolicyFactory._registry)
    except Exception as e:  # noqa: BLE001
        info["cache_policies"] = f"ERR {type(e).__name__}: {e}"
    try:
        from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory

        reg = getattr(SecondaryTierFactory, "_registry", None)
        info["secondary_tiers"] = sorted(reg) if reg is not None else "no _registry"
    except Exception as e:  # noqa: BLE001
        info["secondary_tiers"] = f"ERR {type(e).__name__}: {e}"
    return info


def pkg_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for p in ("flashinfer-python", "xformers", "transformers", "triton",
              "numpy", "huggingface-hub", "flash-attn"):
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="results/env.json")
    args = ap.parse_args()

    g = gpus()
    t = torch_info()
    fp = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "platform": "A",  # A = RTX 3090 (本機); B = MI300X
        "host": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "gpus": g,
        "gpu_count": len(g),
        "driver": g[0]["driver"] if g else None,
        "nvidia_smi_cuda_version": (sh(["nvidia-smi"]) or "").split("CUDA Version:")[-1]
        .split("|")[0]
        .strip()
        or None,
        "torch": t,
        "vllm": vllm_info(),
        "packages": pkg_versions(),
        "paths": {
            "big_files_root": "/ssd7/hungwei/paper-hkv",
            "hf_home": "/ssd7/hungwei/paper-hkv/hf-cache/huggingface",
            "venv": "/ssd7/hungwei/paper-hkv/venv/vllm",
        },
        # 平台 A 量不到的東西，先寫死，避免日後有人拿估值來填。
        "not_supported_on_this_platform": {
            "native_fp8_kv": "sm_86 (Ampere) has no native FP8 — 論文動作空間的 GPU-FP8 階在此不可量測",
            "hbm_energy_counters": "GeForce 無記憶體/計算分軌能耗計數器 — 論文 §6.7 不適用於平台 A",
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fp, indent=2, ensure_ascii=False) + "\n")

    ok = bool(t["real_kernel_ok"])
    print(f"wrote {out}")
    print(f"  gpus              : {len(g)} × {g[0]['name'] if g else 'NONE'}")
    print(f"  driver            : {fp['driver']}")
    print(f"  torch             : {t['torch_version']} (built for cuda {t['torch_built_for_cuda']})")
    print(f"  vllm              : {fp['vllm']['vllm_version']}")
    print(f"  REAL KERNEL       : {'OK' if ok else 'FAIL — ' + str(t['real_kernel_error'])}")
    print(f"  offloading specs  : {fp['vllm']['offloading_specs']}")
    print(f"  cache policies    : {fp['vllm']['cache_policies']}")
    print(f"  secondary tiers   : {fp['vllm']['secondary_tiers']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
