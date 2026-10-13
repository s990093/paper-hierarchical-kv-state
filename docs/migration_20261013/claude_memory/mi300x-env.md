---
name: mi300x-env
description: Where things live on the MI300X box (env script, runs dir, models, vLLM sources) and gotchas
metadata:
  type: reference
---
Env: `source code/m7_env.sh` (venv /mlsteam/workspace/venv/tiara, vLLM 0.19.1 src /mlsteam/workspace/src/vllm; vLLM 0.28 in venv/tiara-v028, used by C6). Wrap runs with `m7run <short> cmd` → /mlsteam/data/tiara/runs/<run_id>/. Models: /mlsteam/data/tiara/hf-cache/hub (Llama-3.1-8B, LongAlpaca-7B MHA vocab 32001 & 32K max, Qwen3-30B-A3B, Qwen2.5-1M, etc.). Local PDFs: /mlsteam/data/tiara/papers.
Gotchas: random token ids must be < vocab_size; HF MoE (Qwen3) is ~7x slower than vLLM so its f(i) is unusable; crashed ROCm processes drop 20 GB `gpucore.*` files in cwd; amd-smi reports pid -1 right after own process exits (false contamination). Repo is public; push milestone results to main (user approved merging into main 2026-10-09).
runsh fixed 2026-10-10: args now written with %q (spaces and quotes survive) and m7run's RUN_ID is reused. Backup is at /mlsteam/workspace/bin/runsh.bak-20261010. Multiple GPU agents serialize with `flock /mlsteam/data/tiara/gpu.lock`.
