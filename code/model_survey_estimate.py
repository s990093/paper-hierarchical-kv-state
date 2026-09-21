#!/usr/bin/env python3
"""平台 B（單張 MI300X）選模估算：由 HF config.json 算記憶體、可達上下文與 κ 下界。

⚠️ 本檔輸出全部是「算術估計」，不是量測。CSV 的每一列都帶 `estimate_kind=ARITHMETIC`，
   引用時不得寫成實測值（CLAUDE.md §1 規則 1）。真值由 M1（vLLM 啟動時的
   `GPU KV cache size`）與 M2（成本常數）量出來。

輸入：model_survey_fetch.py 抓下的 <raw>/<org>__<name>/{config.json, api.json}
輸出：<out>.csv（每模型一列）

估算採用的常數與假設（全部寫進 CSV 欄位，不藏在程式裡）：
  VRAM          206,141,652,992 B   results/hw_mi300x.json（實測）
  gpu_util      0.90                vLLM 預設 gpu_memory_utilization
  runtime 餘裕  8 GiB               假設值（activation / workspace / graph），M1 量真值
  1M            1,048,576 token     = vLLM --max-model-len 1048576
  算力          1307.4 TFLOPS       論文 tab:ratio 的 MI300X BF16 峰值（規格值）
  主機鏈路      63.0 GB/s           PCIe Gen5 x16 單向理論值（論文同表）

逐層記憶體公式（與 vLLM 0.19.1 的 kv_cache_interface / mamba_utils 一致）：
  GQA 全注意力    2·H_kv·d_h·2 B/token
  滑動視窗 W      同上，但每序列上限 W token（HMA 開啟時）；
                  **OffloadingConnector 在 0.19.1 會關掉 HMA，滑動視窗層改配全長 KV**
  MLA             (kv_lora_rank + qk_rope_head_dim)·2 B/token
  GDN（Qwen3-Next/3.5）  conv (k_heads·k_dim·2 + v_heads·v_dim)·(kernel−1)·2 B
                         + temporal v_heads·v_dim·k_dim·(ssm dtype) B        每序列常數
  Mamba2          conv (d_inner + 2·groups·d_state)·(kernel−1)·2 + n_heads·head_dim·d_state·(ssm dtype)
  Mamba1          conv d_inner·(kernel−1)·2 + d_inner·d_state·2
  KDA             conv 3·(heads·head_dim)·(kernel−1)·2 + heads·head_dim²·4
  Lightning       heads·head_dim²·2
  short conv      hidden·(L_cache−1)·2
"""
import argparse, csv, datetime, json, os, re, sys

GIB = 2 ** 30
VRAM_BYTES = 206_141_652_992
GPU_UTIL = 0.90
RUNTIME_MARGIN_GIB = 8.0
L1M = 1_048_576
TFLOPS = 1307.4e12
LINK_BPS = 63.0e9

# 模型卡宣稱的上下文（grep README 所得；行號可在 raw/<repo>/README.md 反查）
# (上限 token, 達成方式)
CLAIMED_CTX = {
    "Qwen/Qwen2.5-7B-Instruct-1M": (1_010_000, "DCA+稀疏注意力（需 Qwen 客製 vLLM；>262,144 無 DCA 會退化）"),
    "Qwen/Qwen2.5-14B-Instruct-1M": (1_010_000, "DCA+稀疏注意力（需 Qwen 客製 vLLM；>262,144 無 DCA 會退化）"),
    "unsloth/Llama-3.1-8B-Instruct": (131_072, "原生（llama3 RoPE）"),
    "unsloth/Llama-3.3-70B-Instruct": (131_072, "原生（llama3 RoPE）"),
    "nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct": (1_073_152, "原生（1M 序列持續預訓練）"),
    "nvidia/Llama-3.1-Nemotron-8B-UltraLong-4M-Instruct": (4_292_608, "原生（4M 序列持續預訓練）"),
    "gradientai/Llama-3-70B-Instruct-Gradient-1048k": (1_048_576, "RoPE θ 放大 + 長序列微調"),
    "zai-org/glm-4-9b-chat-1m": (1_048_576, "原生（模型卡：1M）"),
    "ByteDance-Seed/Seed-OSS-36B-Instruct": (524_288, "原生（config）"),
    "Qwen/Qwen3-32B": (131_072, "32,768 原生；131,072 需 YaRN"),
    "google/gemma-4-31B-it": (262_144, "原生 256K"),
    "swiss-ai/Apertus-70B-Instruct-2509": (65_536, "原生"),
    "LGAI-EXAONE/EXAONE-4.5-33B": (262_144, "原生"),
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": (131_072, "原生"),
    "internlm/Intern-S2-Mobius": (262_144, "config（模型卡未宣稱更長）"),
    "CohereLabs/North-Mini-Code-1.0": (256_000, "原生 256K"),
    "Qwen/Qwen3-30B-A3B-Instruct-2507": (1_010_000, "262,144 原生；1M 需 DCA+MInference（卡：約需 240 GB GPU 記憶體）"),
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": (1_000_000, "262,144 原生；1M 需 YaRN"),
    "unsloth/Llama-4-Scout-17B-16E-Instruct": (10_485_760, "config 10M（iRoPE）"),
    "tencent/Hunyuan-A13B-Instruct": (262_144, "原生 256K（config 預設 32K，需改 max_position_embeddings）"),
    "zai-org/GLM-4.7-Flash": (202_752, "config"),
    "zai-org/GLM-4.5-Air": (131_072, "config"),
    "google/gemma-4-26B-A4B-it": (262_144, "原生 256K"),
    "openai/gpt-oss-120b": (131_072, "YaRN（config）"),
    "openai/gpt-oss-20b": (131_072, "YaRN（config）"),
    "meituan-longcat/LongCat-Flash-Lite": (262_144, "YaRN 256K"),
    "baidu/ERNIE-4.5-21B-A3B-PT": (131_072, "原生"),
    "arcee-ai/Trinity-Mini": (131_072, "原生 128K"),
    "mistralai/Mistral-Small-4-119B-2603": (262_144, "模型卡 256K（config 寫 1,048,576）"),
    "microsoft/Phi-3.5-MoE-instruct": (131_072, "LongRoPE 128K"),
    "mistralai/Mixtral-8x22B-Instruct-v0.1": (65_536, "原生"),
    "upstage/Solar-Open-100B": (131_072, "YaRN 128K"),
    "Qwen/Qwen-AgentWorld-35B-A3B": (262_144, "原生"),
    "nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16": (1_048_576, "模型卡 1M（config 預設 256K；RULER@1M 已報）"),
    "Qwen/Qwen3-Next-80B-A3B-Instruct": (1_010_000, "262,144 原生；1,010,000 需 YaRN"),
    "Qwen/Qwen3-Coder-Next": (262_144, "原生 256K（卡未宣稱 1M）"),
    "Qwen/Qwen3.5-27B": (1_010_000, "262,144 原生；1,010,000 需 YaRN"),
    "Qwen/Qwen3.5-35B-A3B": (1_010_000, "262,144 原生；1,010,000 需 YaRN"),
    "Qwen/Qwen3.5-122B-A10B": (1_010_000, "262,144 原生；1,010,000 需 YaRN"),
    "Qwen/Qwen3.6-27B": (1_010_000, "262,144 原生；1,010,000 需 YaRN"),
    "Qwen/Qwen3.6-35B-A3B": (1_010_000, "262,144 原生；1,010,000 需 YaRN"),
    "Qwen/Qwen3.8-27B": (1_000_000, "262,144 原生；1,000,000 需延伸"),
    "Qwen/Qwen3.8-Flash-Next": (1_000_000, "262,144 原生；1,000,000 需延伸"),
    "moonshotai/Kimi-Linear-48B-A3B-Instruct": (1_048_576, "原生 1M"),
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": (1_048_576, "模型卡 1M（config 預設 256K；RULER@1M 已報）"),
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": (1_048_576, "模型卡 1M（RULER@1M 已報）"),
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16": (1_048_576, "模型卡 1M"),
    "ai21labs/AI21-Jamba2-Mini": (262_144, "原生 256K"),
    "tiiuae/Falcon-H1-34B-Instruct": (262_144, "config"),
    "ibm-granite/granite-4.0-h-small": (131_072, "模型卡 128K"),
    "ibm-granite/granite-4.1-30b": (131_072, "config"),
    "ibm-granite/granite-4.2-30b": (131_072, "128K 原生（卡：可延伸至 512K）"),
    "openbmb/MiniCPM-SALA": (1_048_576, "模型卡 1M+（config 524,288）"),
    "inclusionAI/Ling-2.6-flash": (262_144, "131,072 原生；262,144 需 YaRN"),
    "LiquidAI/LFM2-24B-A2B": (128_000, "config"),
    "meituan-longcat/LongCat-Flash-Lite-Sparse": (1_048_576, "模型卡：原生 1M"),
    "MiniMaxAI/MiniMax-M1-80k-hf": (1_000_000, "原生 1M（Lightning 注意力）"),
}

# 模型卡宣稱的啟用參數（B）；沒有的由 config 計算，來源欄會標明
DECLARED_ACTIVE_B = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507": 3.3, "Qwen/Qwen3-Coder-30B-A3B-Instruct": 3.3,
    "Qwen/Qwen3-Next-80B-A3B-Instruct": 3.0, "Qwen/Qwen3-Coder-Next": 3.0,
    "Qwen/Qwen3.5-35B-A3B": 3.0, "Qwen/Qwen3.6-35B-A3B": 3.0, "Qwen/Qwen3.5-122B-A10B": 10.0,
    "Qwen/Qwen-AgentWorld-35B-A3B": 3.0, "Qwen/Qwen3.8-Flash-Next": 6.0,
    "unsloth/Llama-4-Scout-17B-16E-Instruct": 17.0, "tencent/Hunyuan-A13B-Instruct": 13.0,
    "google/gemma-4-26B-A4B-it": 3.8, "openai/gpt-oss-120b": 5.1, "openai/gpt-oss-20b": 3.6,
    "meituan-longcat/LongCat-Flash-Lite": 3.0, "meituan-longcat/LongCat-Flash-Lite-Sparse": 3.0,
    "baidu/ERNIE-4.5-21B-A3B-PT": 3.0, "arcee-ai/Trinity-Mini": 3.0,
    "mistralai/Mistral-Small-4-119B-2603": 6.5, "microsoft/Phi-3.5-MoE-instruct": 6.6,
    "upstage/Solar-Open-100B": 12.0, "CohereLabs/North-Mini-Code-1.0": 3.0,
    "moonshotai/Kimi-Linear-48B-A3B-Instruct": 3.0, "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": 12.0,
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16": 3.0,
    "nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16": 9.0,
    "ai21labs/AI21-Jamba2-Mini": 12.0, "inclusionAI/Ling-2.6-flash": 7.4, "LiquidAI/LFM2-24B-A2B": 2.3,
    "MiniMaxAI/MiniMax-M1-80k-hf": 45.9,
    "zai-org/GLM-4.5-Air": 12.0, "zai-org/GLM-4.7-Flash": 3.0,
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": 3.5,
}
# 名稱帶 A<n>B 但卡上沒寫數字者，以名稱為準（標明）
NAME_ACTIVE_B = {}

STATE_KINDS = {"gdn", "mamba2", "mamba1", "kda", "lightning", "shortconv"}

# repo → model_survey_papers.MODELS 的家族鍵（論文使用頻率用）
PAPER_FAMILY = {
    "Qwen/Qwen2.5-7B-Instruct-1M": "Qwen2.5-7B-1M", "Qwen/Qwen2.5-14B-Instruct-1M": "Qwen2.5-14B",
    "unsloth/Llama-3.1-8B-Instruct": "Llama-3.1-8B", "unsloth/Llama-3.3-70B-Instruct": "Llama-3.3-70B",
    "nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct": "UltraLong-8B-1M/4M",
    "nvidia/Llama-3.1-Nemotron-8B-UltraLong-4M-Instruct": "UltraLong-8B-1M/4M",
    "zai-org/glm-4-9b-chat-1m": "GLM-4-9B-1M", "ByteDance-Seed/Seed-OSS-36B-Instruct": "Seed-OSS-36B",
    "Qwen/Qwen3-32B": "Qwen3-32B", "google/gemma-4-31B-it": "Gemma-4", "google/gemma-4-26B-A4B-it": "Gemma-4",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": "Mistral-Small/Nemo", "mistralai/Mistral-Small-4-119B-2603": "Mistral-Small/Nemo",
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "Qwen3-30B-A3B", "Qwen/Qwen3-Coder-30B-A3B-Instruct": "Qwen3-30B-A3B",
    "unsloth/Llama-4-Scout-17B-16E-Instruct": "Llama-4-Scout", "tencent/Hunyuan-A13B-Instruct": "Hunyuan-A13B",
    "zai-org/GLM-4.7-Flash": "GLM-4.5/4.6/4.7", "zai-org/GLM-4.5-Air": "GLM-4.5/4.6/4.7",
    "openai/gpt-oss-120b": "gpt-oss-120b", "openai/gpt-oss-20b": "gpt-oss-20b",
    "microsoft/Phi-3.5-MoE-instruct": "Phi-3.5-MoE", "mistralai/Mixtral-8x22B-Instruct-v0.1": "Mixtral-8x22B",
    "nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16": "Nemotron-3",
    "Qwen/Qwen3-Next-80B-A3B-Instruct": "Qwen3-Next-80B-A3B", "Qwen/Qwen3-Coder-Next": "Qwen3-Next-80B-A3B",
    "Qwen/Qwen3.5-27B": "Qwen3.5/3.6", "Qwen/Qwen3.5-35B-A3B": "Qwen3.5/3.6", "Qwen/Qwen3.5-122B-A10B": "Qwen3.5/3.6",
    "Qwen/Qwen3.6-27B": "Qwen3.5/3.6", "Qwen/Qwen3.6-35B-A3B": "Qwen3.5/3.6", "Qwen/Qwen3.8-27B": "Qwen3.5/3.6",
    "Qwen/Qwen3.8-Flash-Next": "Qwen3.5/3.6",
    "moonshotai/Kimi-Linear-48B-A3B-Instruct": "Kimi-Linear",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "Nemotron-3", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": "Nemotron-3",
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16": "Nemotron-3",
    "ai21labs/AI21-Jamba2-Mini": "Jamba", "tiiuae/Falcon-H1-34B-Instruct": "Falcon-H1",
    "ibm-granite/granite-4.0-h-small": "Granite-4.0-H", "MiniMaxAI/MiniMax-M1-80k-hf": "MiniMax-M1/Text-01",
}


def dtype_bytes(name):
    return 4 if str(name).lower() in ("float32", "fp32") else 2


class Inv:
    def __init__(self):
        self.layers = []  # dict(kind, kv_bpt, window, qdim, state_b)

    def attn(self, kind, kv_bpt, qdim, window=None):
        self.layers.append(dict(kind=kind, kv_bpt=kv_bpt, window=window, qdim=qdim, state_b=0))

    def state(self, kind, state_b):
        self.layers.append(dict(kind=kind, kv_bpt=0, window=None, qdim=0, state_b=state_b))


def build_inventory(cfg):
    tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}

    def g(k, d=None):
        v = tc.get(k)
        return v if v is not None else cfg.get(k, d)

    arch = (cfg.get("architectures") or [""])[0]
    mt = g("model_type") or ""
    inv = Inv()
    hidden = g("hidden_size")
    heads = g("num_attention_heads")

    def hd_default():
        return g("head_dim") or g("attention_head_dim") or hidden // heads

    def gqa(kv, hd):
        return 2 * kv * hd * 2

    ssm_b = dtype_bytes(g("mamba_ssm_cache_dtype") or g("mamba_ssm_dtype") or "bfloat16")

    def gdn_state():
        kh, vh = g("linear_num_key_heads"), g("linear_num_value_heads")
        kd, vd, ker = g("linear_key_head_dim"), g("linear_value_head_dim"), g("linear_conv_kernel_dim")
        conv = (kd * kh * 2 + vd * vh) * (ker - 1) * 2
        temporal = vh * vd * kd * ssm_b
        return conv + temporal

    if arch in ("LongcatFlashNgramForCausalLM", "LongcatCausalLM"):
        mla = (g("kv_lora_rank") + g("qk_rope_head_dim")) * 2
        qd = heads * (g("qk_nope_head_dim") + g("qk_rope_head_dim"))
        for _ in range(2 * g("num_layers")):   # vLLM longcat_flash: 每個 decoder layer 兩個 MLA
            inv.attn("mla", mla, qd)
        note = "每個 decoder layer 含 2 個 MLA（vLLM longcat_flash.py）"
        if arch == "LongcatCausalLM":
            note += "；LSA 索引器的 KV 未計入（格式未查證）"
        return inv, note

    if mt in ("llama", "qwen2", "qwen3", "qwen3_moe", "seed_oss", "mistral", "granite", "apertus",
              "hunyuan_v1_moe", "phimoe", "mixtral", "glm4_moe", "ernie4_5_moe", "solar_open"):
        nl, kv, hd = g("num_hidden_layers"), g("num_key_value_heads") or heads, hd_default()
        for _ in range(nl):
            inv.attn("full", gqa(kv, hd), heads * hd)
        return inv, ""

    if mt == "chatglm":
        nl, kv, hd = g("num_layers"), g("multi_query_group_num"), g("hidden_size") // heads
        for _ in range(nl):
            inv.attn("full", gqa(kv, hd), heads * hd)
        return inv, ""

    if mt == "gemma4_text":
        for t in g("layer_types"):
            if t == "sliding_attention":
                kv, hd = g("num_key_value_heads"), g("head_dim")
                inv.attn("swa", gqa(kv, hd), heads * hd, window=g("sliding_window"))
            else:
                kv = g("num_global_key_value_heads") if g("attention_k_eq_v") else g("num_key_value_heads")
                hd = g("global_head_dim") or g("head_dim")
                inv.attn("full", gqa(kv, hd), heads * hd)   # k_eq_v 仍存 K 與 V 兩份（vLLM gemma4.py）
        return inv, "全域層 K=V 權重共用，但 KV cache 仍存兩份"

    if mt in ("exaone4_5_text", "cohere2_moe", "afmoe", "gpt_oss"):
        kv, hd = g("num_key_value_heads"), hd_default()
        for t in g("layer_types"):
            if t == "sliding_attention":
                inv.attn("swa", gqa(kv, hd), heads * hd, window=g("sliding_window"))
            else:
                inv.attn("full", gqa(kv, hd), heads * hd)
        return inv, ""

    if mt == "llama4_text":
        kv, hd = g("num_key_value_heads"), hd_default()
        for flag in g("no_rope_layers"):
            if flag:   # 用 RoPE 的層 = chunked local attention
                inv.attn("chunked", gqa(kv, hd), heads * hd, window=g("attention_chunk_size"))
            else:      # NoPE 全域層
                inv.attn("full", gqa(kv, hd), heads * hd)
        return inv, "3/4 層為 8,192 chunked local attention"

    if mt in ("glm4_moe_lite", "mistral4"):
        mla = (g("kv_lora_rank") + g("qk_rope_head_dim")) * 2
        qd = heads * (g("qk_nope_head_dim") + g("qk_rope_head_dim"))
        for _ in range(g("num_hidden_layers")):
            inv.attn("mla", mla, qd)
        return inv, ""

    if mt == "qwen3_next":
        kv, hd, interval = g("num_key_value_heads"), g("head_dim"), g("full_attention_interval")
        for i in range(g("num_hidden_layers")):
            if (i + 1) % interval == 0:
                inv.attn("full", gqa(kv, hd), heads * hd)
            else:
                inv.state("gdn", gdn_state())
        return inv, ""

    if mt in ("qwen3_5_text", "qwen3_5_moe_text", "interns2_mobius_text", "qwen4_exp_text"):
        kv, hd = g("num_key_value_heads"), g("head_dim")
        for t in g("layer_types"):
            if t == "linear_attention":
                inv.state("gdn", gdn_state())
            else:
                inv.attn("full", gqa(kv, hd), heads * hd)
        note = "QSA 稀疏注意力的索引器 KV 未計入" if mt == "qwen4_exp_text" else ""
        return inv, note

    if mt == "kimi_linear":
        lac = g("linear_attn_config")
        mla = (g("kv_lora_rank") + g("qk_rope_head_dim")) * 2
        qd = heads * (g("qk_nope_head_dim") + g("qk_rope_head_dim"))
        nh, hdim, ker = lac["num_heads"], lac["head_dim"], lac["short_conv_kernel_size"]
        kda = 3 * (nh * hdim) * (ker - 1) * 2 + nh * hdim * hdim * 4
        for _ in lac["full_attn_layers"]:
            inv.attn("mla", mla, qd)
        for _ in lac["kda_layers"]:
            inv.state("kda", kda)
        return inv, ""

    if mt in ("nemotron_h", "nemotron_h_puzzle"):
        kinds = g("layers_block_type")
        if not kinds:
            pat = g("hybrid_override_pattern")
            kinds = [{"M": "mamba", "*": "attention", "E": "moe", "-": "mlp"}[c] for c in pat]
        kv, hd = g("num_key_value_heads"), g("head_dim")
        nh, mhd, ds, ng, ker = g("mamba_num_heads"), g("mamba_head_dim"), g("ssm_state_size"), g("n_groups"), g("conv_kernel")
        d_inner = nh * mhd
        m2 = (d_inner + 2 * ng * ds) * (ker - 1) * 2 + nh * mhd * ds * ssm_b
        for k in kinds:
            if k == "attention":
                inv.attn("full", gqa(kv, hd), heads * hd)
            elif k == "mamba":
                inv.state("mamba2", m2)
        return inv, ""

    if mt == "jamba":
        kv, hd = g("num_key_value_heads"), hd_default()
        d_inner = g("mamba_expand") * hidden
        m1 = d_inner * (g("mamba_d_conv") - 1) * 2 + d_inner * g("mamba_d_state") * 2
        for i in range(g("num_hidden_layers")):
            if i % g("attn_layer_period") == g("attn_layer_offset"):
                inv.attn("full", gqa(kv, hd), heads * hd)
            else:
                inv.state("mamba1", m1)
        return inv, ""

    if mt == "falcon_h1":
        kv, hd = g("num_key_value_heads"), g("head_dim")
        nh, mhd, ds, ng, ker = g("mamba_n_heads"), g("mamba_d_head"), g("mamba_d_state"), g("mamba_n_groups"), g("mamba_d_conv")
        d_inner = g("mamba_d_ssm") or nh * mhd
        m2 = (d_inner + 2 * ng * ds) * (ker - 1) * 2 + nh * mhd * ds * ssm_b
        for _ in range(g("num_hidden_layers")):   # 平行混合：每層同時有 attention 與 Mamba2
            inv.attn("full", gqa(kv, hd), heads * hd)
            inv.state("mamba2", m2)
        return inv, "平行混合：每層同時有 attention 與 Mamba2"

    if mt == "granitemoehybrid":
        kv, hd = g("num_key_value_heads"), hd_default()
        nh, mhd, ds, ng, ker = g("mamba_n_heads"), g("mamba_d_head"), g("mamba_d_state"), g("mamba_n_groups"), g("mamba_d_conv")
        d_inner = g("mamba_expand") * hidden
        m2 = (d_inner + 2 * ng * ds) * (ker - 1) * 2 + nh * mhd * ds * ssm_b
        for t in g("layer_types"):
            if t == "attention":
                inv.attn("full", gqa(kv, hd), heads * hd)
            else:
                inv.state("mamba2", m2)
        return inv, ""

    if mt == "minicpm_sala":
        kv, hd = g("num_key_value_heads"), g("head_dim")
        ln = g("lightning_nh") * g("lightning_head_dim") ** 2 * 2
        for t in g("mixer_types"):
            if t == "lightning-attn":
                inv.state("lightning", ln)
            else:
                inv.attn("full", gqa(kv, hd), heads * hd)   # InfLLM-v2 稀疏注意力，KV 全存
        return inv, "稀疏注意力層的壓縮 KV（block 摘要）未計入"

    if mt == "bailing_hybrid":
        mla = (g("kv_lora_rank") + g("qk_rope_head_dim")) * 2
        qd = heads * (g("qk_nope_head_dim") + g("qk_rope_head_dim"))
        gs = g("layer_group_size")
        ln = heads * g("head_dim") ** 2 * 2
        for i in range(g("num_hidden_layers")):
            if (i + 1) % gs != 0:   # vLLM bailing_moe_linear.is_linear_layer
                inv.state("lightning", ln)
            else:
                inv.attn("mla", mla, qd)
        return inv, ""

    if mt == "lfm2_moe":
        kv, hd = g("num_key_value_heads"), hidden // heads
        sc = hidden * (g("conv_L_cache") - 1) * 2
        for t in g("layer_types"):
            if t == "full_attention":
                inv.attn("full", gqa(kv, hd), heads * hd)
            else:
                inv.state("shortconv", sc)
        return inv, ""

    if mt == "minimax":
        kv, hd = g("num_key_value_heads"), g("head_dim")
        ln = heads * hd * hd * 2
        for t in g("layer_types"):
            if t == "linear_attention":
                inv.state("lightning", ln)
            else:
                inv.attn("full", gqa(kv, hd), heads * hd)
        return inv, ""

    raise ValueError(f"未支援的架構：arch={arch} model_type={mt}")


def computed_active_b(cfg, total_params):
    """標準 MoE 的啟用參數（B）＝總量 − 未啟用專家。只在卡上沒寫時用。"""
    tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
    g = lambda k, d=None: tc.get(k) if tc.get(k) is not None else cfg.get(k, d)
    E = g("num_experts") or g("num_local_experts") or g("n_routed_experts") or g("moe_num_experts")
    k = g("num_experts_per_tok") or g("moe_k") or g("top_k_experts") or g("num_experts_per_token")
    if not E or not k:
        return None
    if isinstance(k, list):
        k = k[0]
    hidden = g("hidden_size")
    inter = g("moe_intermediate_size") or g("intermediate_size")
    if isinstance(inter, list):
        inter = inter[0]
    nl = g("num_hidden_layers") or g("num_layers")
    mt = g("model_type")
    if mt == "granitemoehybrid":
        n_moe = nl
    elif mt == "mixtral":
        n_moe = nl
    else:
        n_moe = nl - (g("first_k_dense_replace") or 0)
    per_expert = 3 * hidden * inter
    return (total_params - n_moe * (E - k) * per_expert) / 1e9


def category(inv, cfg):
    kinds = {l["kind"] for l in inv.layers}
    tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
    g = lambda kk: tc.get(kk) if tc.get(kk) is not None else cfg.get(kk)
    is_moe = bool(g("num_experts") or g("num_local_experts") or g("n_routed_experts") or g("moe_num_experts"))
    if kinds & STATE_KINDS:
        return "hybrid", ("MoE" if is_moe else "dense")
    return ("moe" if is_moe else "dense"), ("MoE" if is_moe else "dense")


def attn_type(inv):
    kinds = [l["kind"] for l in inv.layers if l["kv_bpt"]]
    s = set(kinds)
    parts = []
    if "full" in s:
        parts.append("GQA")
    if "mla" in s:
        parts.append("MLA")
    if "swa" in s:
        w = {l["window"] for l in inv.layers if l["kind"] == "swa"}
        parts.append(f"SWA{'/'.join(str(x) for x in sorted(w))}")
    if "chunked" in s:
        parts.append("chunked8192")
    st = sorted({l["kind"] for l in inv.layers if l["kind"] in STATE_KINDS})
    return "+".join(parts) + (" | " + "+".join(st) if st else "")


def kv_bytes(inv, L, kv_scale=1.0, hma=True):
    tot = 0.0
    for l in inv.layers:
        if l["kv_bpt"]:
            cap = l["window"] if (hma and l["window"]) else None
            tot += l["kv_bpt"] * kv_scale * (min(L, cap) if cap else L)
        tot += l["state_b"]
    return tot


def max_ctx(inv, budget_b, kv_scale=1.0, hma=True, limit=64 * L1M):
    if kv_bytes(inv, 1, kv_scale, hma) > budget_b:
        return 0
    lo, hi = 1, limit
    if kv_bytes(inv, hi, kv_scale, hma) <= budget_b:
        return hi
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if kv_bytes(inv, mid, kv_scale, hma) <= budget_b:
            lo = mid
        else:
            hi = mid - 1
    return lo


def prefill_flops(inv, L, n_active):
    f = 2 * n_active * L
    for l in inv.layers:
        if l["kind"] in ("full", "mla"):
            f += l["qdim"] * L * L
        elif l["kind"] in ("swa", "chunked"):
            w = l["window"]
            f += l["qdim"] * 2 * min(w, L) * L
    return f


def registry_has(arch, vllm_src):
    reg = open(os.path.join(vllm_src, "vllm/model_executor/models/registry.py")).read()
    return f'"{arch}"' in reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--vllm-src", default="/mlsteam/workspace/src/vllm")
    args = ap.parse_args()

    index = json.load(open(os.path.join(args.raw, "index.json")))
    ts = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    usable_b = GPU_UTIL * VRAM_BYTES
    rows = []
    for rec in index:
        repo = rec["repo"]
        d = os.path.join(args.raw, repo.replace("/", "__"))
        cfg = json.load(open(os.path.join(d, "config.json")))
        api = json.load(open(os.path.join(d, "api.json")))
        params = (api.get("safetensors") or {}).get("parameters") or {}
        total = sum(params.values())
        release = ",".join(f"{k}:{v/1e9:.1f}B" for k, v in sorted(params.items(), key=lambda x: -x[1]) if v > 5e7)
        inv, note = build_inventory(cfg)
        cat, ffn = category(inv, cfg)
        arch = (cfg.get("architectures") or [""])[0]

        if repo in DECLARED_ACTIVE_B:
            n_act, act_src = DECLARED_ACTIVE_B[repo] * 1e9, "model card"
        elif repo in NAME_ACTIVE_B:
            n_act, act_src = NAME_ACTIVE_B[repo] * 1e9, "model name (A<n>B)"
        elif ffn == "MoE":
            v = computed_active_b(cfg, total)
            if v is not None and 0 < v < total / 1e9:
                n_act, act_src = v * 1e9, "computed from config"
            else:
                n_act, act_src = float("nan"), "UNKNOWN（config 公式不適用：非標準專家結構）"
        else:
            n_act, act_src = total, "dense = total"

        w_b = total * 2
        budget_b = usable_b - w_b - RUNTIME_MARGIN_GIB * GIB
        attn_layers = sum(1 for l in inv.layers if l["kv_bpt"])
        full_like = sum(1 for l in inv.layers if l["kind"] in ("full", "mla"))
        bpt_full = sum(l["kv_bpt"] for l in inv.layers if l["kind"] in ("full", "mla"))
        bpt_all = sum(l["kv_bpt"] for l in inv.layers if l["kv_bpt"])
        state_b = sum(l["state_b"] for l in inv.layers)
        state_layers = sum(1 for l in inv.layers if l["state_b"])
        kv1m = kv_bytes(inv, L1M)
        kv1m_nohma = kv_bytes(inv, L1M, hma=False)
        claimed, method = CLAIMED_CTX.get(repo, (None, ""))
        has_state = state_layers > 0
        has_swa = any(l["kind"] in ("swa", "chunked") for l in inv.layers)
        in_reg = registry_has(arch, args.vllm_src)
        if not in_reg:
            offload_019 = "N/A（0.19.1 無此架構）"
        elif has_state:
            offload_019 = "BLOCKED：OffloadingConnector 非 SupportsHMA → HMA 關閉 → Mamba/線性層規格無法統一（靜態讀碼，未實跑）"
        elif has_swa:
            offload_019 = "可跑，但 HMA 關閉 → 滑動視窗層改配全長 KV"
        else:
            offload_019 = "可跑（KV 規格一致）"
        # κ 下界：與論文 tab:kappa 同法（線性層 2N_active、100% MFU；傳輸 = 每 token KV 位元組 / 鏈路）
        t_rec = 2 * n_act / TFLOPS
        # 傳輸只算「遠端位置仍需保留」的 KV：全注意力 / MLA 層（滑動視窗與 chunked 層的舊位置本來就不存）
        t_xfer = bpt_full / LINK_BPS if bpt_full else float("nan")
        kappa0 = t_rec / t_xfer if bpt_full else float("nan")
        # 混合模型開 prefix caching 時，每個 block 邊界要存一份全部線性層的 state（vLLM mamba_cache_mode="all"）
        # block 大小依 vLLM models/config.py：attention 單層頁 ≥ 單層 state 頁，並對齊 16（MLA 為 64）
        ckpt_bpt, blk = 0.0, None
        if state_layers:
            per_layer_state = max(l["state_b"] for l in inv.layers if l["state_b"])
            attn_layer_1tok = max(l["kv_bpt"] for l in inv.layers if l["kv_bpt"])
            align = 64 if any(l["kind"] == "mla" for l in inv.layers) else 16
            blk = align * -(-per_layer_state // (align * attn_layer_1tok))
            ckpt_bpt = state_b / blk
        kappa0_ckpt = (t_rec / ((bpt_full + ckpt_bpt) / LINK_BPS)) if (bpt_full + ckpt_bpt) else float("nan")
        pf1m = prefill_flops(inv, L1M, n_act)
        pf128 = prefill_flops(inv, 131_072, n_act)
        row = dict(
            run_id=args.run_id, ts=ts, estimate_kind="ARITHMETIC_NOT_MEASURED",
            category=cat, ffn=ffn, repo=repo, note_fetch=rec.get("note", ""), arch=arch,
            model_type=(cfg.get("text_config") or {}).get("model_type") or cfg.get("model_type") or "",
            hf_sha=rec.get("sha", ""), created=(api.get("createdAt") or "")[:10], downloads=api.get("downloads"),
            release_dtypes=release, total_params_b=round(total / 1e9, 2),
            active_params_b=round(n_act / 1e9, 2) if n_act == n_act else "", active_src=act_src,
            weights_bf16_gib=round(w_b / GIB, 1),
            layers_total=len({id(l) for l in inv.layers}) if False else None,
            attn_layers=attn_layers, full_or_mla_layers=full_like, state_layers=state_layers,
            attn_type=attn_type(inv),
            kv_kib_per_token_bf16_fullpart=round(bpt_full / 1024, 2),
            kv_kib_per_token_bf16_alllayers=round(bpt_all / 1024, 2),
            state_mib_per_seq=round(state_b / 2**20, 1),
            kv_gib_at_1m_bf16=round(kv1m / GIB, 1),
            kv_gib_at_1m_bf16_no_hma=round(kv1m_nohma / GIB, 1),
            total_gib_at_1m_bf16=round((w_b + kv1m) / GIB, 1),
            usable_gib=round(usable_b / GIB, 1), runtime_margin_gib=RUNTIME_MARGIN_GIB,
            kv_budget_gib=round(budget_b / GIB, 1),
            weights_fit=budget_b > 0,
            fit_1m_bf16kv=(budget_b >= kv1m),
            fit_1m_fp8kv=(budget_b >= kv_bytes(inv, L1M, kv_scale=0.5)),
            max_ctx_bf16kv=max_ctx(inv, budget_b) if budget_b > 0 else 0,
            max_ctx_fp8kv=max_ctx(inv, budget_b, kv_scale=0.5) if budget_b > 0 else 0,
            max_ctx_bf16kv_offload019=(max_ctx(inv, budget_b, hma=False) if budget_b > 0 and not has_state else ""),
            claimed_ctx=claimed, ctx_method=method,
            vllm_0191_registered=in_reg, offloading_connector_0191=offload_019,
            kappa0_mi300x=round(kappa0, 1) if bpt_full and kappa0 == kappa0 else "",
            t_recompute_us_per_tok=round(t_rec * 1e6, 2) if t_rec == t_rec else "",
            t_transfer_us_per_tok=round(t_xfer * 1e6, 3) if bpt_full else "",
            hybrid_block_size_tokens=blk or "",
            state_ckpt_kib_per_token=round(ckpt_bpt / 1024, 1) if state_layers else "",
            cached_prefix_kib_per_token=round((bpt_full + ckpt_bpt) / 1024, 1),
            kappa0_with_state_ckpt=round(kappa0_ckpt, 1) if state_layers and kappa0_ckpt == kappa0_ckpt else "",
            paper_family=PAPER_FAMILY.get(repo, ""),
            prefill_s_at_1m_mfu100=round(pf1m / TFLOPS, 0) if pf1m == pf1m else "",
            prefill_s_at_128k_mfu100=round(pf128 / TFLOPS, 1) if pf128 == pf128 else "",
            nfs_load_min_est=round(w_b / 1e6 / 643.9 / 60, 1),
            note=note,
        )
        row.pop("layers_total")
        rows.append(row)
        print(f"{cat:6} {repo:<52} W={row['weights_bf16_gib']:6.1f}GiB KV/tok={row['kv_kib_per_token_bf16_alllayers']:7.2f}KiB "
              f"state={row['state_mib_per_seq']:7.1f}MiB KV@1M={row['kv_gib_at_1m_bf16']:7.1f}GiB fit1M={row['fit_1m_bf16kv']!s:5} "
              f"maxctx={row['max_ctx_bf16kv']:>9} κ0={row['kappa0_mi300x']} ckpt={row['state_ckpt_kib_per_token']} κ0ckpt={row['kappa0_with_state_ckpt']}")
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out_csv, len(rows))


if __name__ == "__main__":
    main()
