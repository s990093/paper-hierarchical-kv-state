#!/usr/bin/env python3
"""平台 B 選模調查：抓 HuggingFace 上候選模型的 config.json / README / safetensors 參數量。

只抓、不算。算在 model_survey_estimate.py。
每個 repo 存成 <out>/<org>__<name>/{config.json, README.md, api.json}，
讓估算表的每一格都能追溯到一個原始檔（CLAUDE.md §1 規則 3）。

gated 的官方 repo（meta-llama 等）沒有 token 抓不到，改用 config 逐欄相同的公開鏡像，
鏡像名稱記在 CANDIDATES 的 note 欄，估算表會照抄。
"""
import argparse, json, os, sys, time, urllib.request, urllib.error

# (類別, repo, 備註)。類別是「預設歸類」，最後以 config 實際內容為準（估算腳本會覆核）。
CANDIDATES = [
    # ---------------- dense transformer ----------------
    ("dense", "Qwen/Qwen2.5-7B-Instruct-1M", "平台 A 主力（對照）"),
    ("dense", "Qwen/Qwen2.5-14B-Instruct-1M", ""),
    ("dense", "unsloth/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct 的公開鏡像（官方 gated）"),
    ("dense", "unsloth/Llama-3.3-70B-Instruct", "meta-llama/Llama-3.3-70B-Instruct 的公開鏡像（官方 gated）"),
    ("dense", "nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct", ""),
    ("dense", "nvidia/Llama-3.1-Nemotron-8B-UltraLong-4M-Instruct", ""),
    ("dense", "gradientai/Llama-3-70B-Instruct-Gradient-1048k", ""),
    ("dense", "zai-org/glm-4-9b-chat-1m", ""),
    ("dense", "ByteDance-Seed/Seed-OSS-36B-Instruct", ""),
    ("dense", "Qwen/Qwen3-32B", ""),
    ("dense", "google/gemma-4-31B-it", ""),
    ("dense", "swiss-ai/Apertus-70B-Instruct-2509", ""),
    ("dense", "LGAI-EXAONE/EXAONE-4.5-33B", ""),
    ("dense", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", ""),
    ("dense", "internlm/Intern-S2-Mobius", ""),
    ("dense", "CohereLabs/North-Mini-Code-1.0", ""),
    # ---------------- MoE ----------------
    ("moe", "Qwen/Qwen3-30B-A3B-Instruct-2507", ""),
    ("moe", "Qwen/Qwen3-Coder-30B-A3B-Instruct", ""),
    ("moe", "unsloth/Llama-4-Scout-17B-16E-Instruct", "meta-llama/Llama-4-Scout-17B-16E-Instruct 的公開鏡像（官方 gated）"),
    ("moe", "tencent/Hunyuan-A13B-Instruct", ""),
    ("moe", "zai-org/GLM-4.7-Flash", ""),
    ("moe", "zai-org/GLM-4.5-Air", ""),
    ("moe", "google/gemma-4-26B-A4B-it", ""),
    ("moe", "openai/gpt-oss-120b", "官方發布為 MXFP4"),
    ("moe", "openai/gpt-oss-20b", "官方發布為 MXFP4"),
    ("moe", "meituan-longcat/LongCat-Flash-Lite", ""),
    ("moe", "baidu/ERNIE-4.5-21B-A3B-PT", ""),
    ("moe", "arcee-ai/Trinity-Mini", ""),
    ("moe", "mistralai/Mistral-Small-4-119B-2603", "官方發布為 FP8"),
    ("moe", "microsoft/Phi-3.5-MoE-instruct", ""),
    ("moe", "mistralai/Mixtral-8x22B-Instruct-v0.1", ""),
    ("moe", "upstage/Solar-Open-100B", ""),
    ("moe", "Qwen/Qwen-AgentWorld-35B-A3B", ""),
    ("moe", "nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16", ""),
    # ---------------- hybrid ----------------
    ("hybrid", "Qwen/Qwen3-Next-80B-A3B-Instruct", ""),
    ("hybrid", "Qwen/Qwen3-Coder-Next", ""),
    ("hybrid", "Qwen/Qwen3.5-27B", ""),
    ("hybrid", "Qwen/Qwen3.5-35B-A3B", ""),
    ("hybrid", "Qwen/Qwen3.5-122B-A10B", ""),
    ("hybrid", "Qwen/Qwen3.6-27B", ""),
    ("hybrid", "Qwen/Qwen3.6-35B-A3B", ""),
    ("hybrid", "Qwen/Qwen3.8-27B", ""),
    ("hybrid", "Qwen/Qwen3.8-Flash-Next", ""),
    ("hybrid", "moonshotai/Kimi-Linear-48B-A3B-Instruct", ""),
    ("hybrid", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", ""),
    ("hybrid", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", ""),
    ("hybrid", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", ""),
    ("hybrid", "ai21labs/AI21-Jamba2-Mini", ""),
    ("hybrid", "tiiuae/Falcon-H1-34B-Instruct", ""),
    ("hybrid", "ibm-granite/granite-4.0-h-small", ""),
    ("hybrid", "ibm-granite/granite-4.1-30b", ""),
    ("hybrid", "ibm-granite/granite-4.2-30b", ""),
    ("hybrid", "openbmb/MiniCPM-SALA", ""),
    ("hybrid", "inclusionAI/Ling-2.6-flash", ""),
    ("hybrid", "LiquidAI/LFM2-24B-A2B", ""),
    ("hybrid", "meituan-longcat/LongCat-Flash-Lite-Sparse", ""),
    ("hybrid", "MiniMaxAI/MiniMax-M1-80k-hf", "456B，單卡放不下，列為上界參考"),
]

def get(url, binary=False):
    req = urllib.request.Request(url, headers={"User-Agent": "tiara-model-survey"})
    with urllib.request.urlopen(req, timeout=90) as r:
        data = r.read()
    return data if binary else data.decode("utf-8", "replace")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    index = []
    for cat, repo, note in CANDIDATES:
        d = os.path.join(args.out, repo.replace("/", "__"))
        os.makedirs(d, exist_ok=True)
        rec = {"category_hint": cat, "repo": repo, "note": note, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        try:
            api = json.loads(get(f"https://huggingface.co/api/models/{repo}?expand[]=safetensors&expand[]=config&expand[]=createdAt&expand[]=downloads&expand[]=gated&expand[]=cardData&expand[]=sha"))
            json.dump(api, open(os.path.join(d, "api.json"), "w"), indent=1)
            rec["sha"] = api.get("sha")
        except Exception as e:
            rec["api_error"] = repr(e)
        for fn in ("config.json", "README.md", "generation_config.json"):
            try:
                txt = get(f"https://huggingface.co/{repo}/resolve/main/{fn}")
                open(os.path.join(d, fn), "w").write(txt)
                rec[fn] = "ok"
            except urllib.error.HTTPError as e:
                rec[fn] = f"HTTP {e.code}"
            except Exception as e:
                rec[fn] = repr(e)
        print(f"{cat:6} {repo:<55} config={rec.get('config.json')} readme={rec.get('README.md')} {rec.get('api_error','')}")
        index.append(rec)
    json.dump(index, open(os.path.join(args.out, "index.json"), "w"), indent=1, ensure_ascii=False)

if __name__ == "__main__":
    main()
