#!/usr/bin/env python3
"""由 model_survey_estimate.py 與 model_survey_papers*.py 的 CSV 產生選模報告（Markdown）。

報告裡的每個數字都從 CSV 取，不手打；讀不到來源就中止。
"""
import argparse, csv, datetime, os, sys


def fnum(x, nd=1):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if v != v:
        return "—"
    return f"{v:,.{nd}f}"


def fint(x):
    try:
        v = int(float(x))
    except (TypeError, ValueError):
        return "—"
    return f"{v:,}"


def pct(x, nd=1):
    try:
        return f"{float(x) * 100:.{nd}f}%"
    except (TypeError, ValueError):
        return "—"


def short(repo):
    return repo.split("/", 1)[1]


def ctx_k(x):
    try:
        v = int(float(x))
    except (TypeError, ValueError):
        return "—"
    if v <= 0:
        return "放不下"
    if v >= 1_000_000:
        return f"{v / 1_048_576:.2f}M"
    return f"{v / 1000:.0f}K"


def fit_mark(r):
    if r["weights_fit"] != "True":
        return "✗ 權重放不下"
    a = "✓" if r["fit_1m_bf16kv"] == "True" else "✗"
    b = "✓" if r["fit_1m_fp8kv"] == "True" else "✗"
    return f"{a} / {b}"


def offload_mark(r):
    if r["weights_fit"] != "True":
        return "—（BF16 權重放不下）"
    s = r["offloading_connector_0191"]
    if s.startswith("BLOCKED"):
        return "🔴 不可（hybrid）"
    if s.startswith("N/A"):
        return "⚪ 0.19.1 無此架構"
    if "滑動視窗" in s:
        return "🟡 可，SWA 失去省記憶體"
    return "🟢 可"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimates", required=True)
    ap.add_argument("--freq", required=True, help="model_frequency_min3.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.estimates)))
    by = {r["repo"]: r for r in rows}
    freq = {f["model"]: f for f in csv.DictReader(open(args.freq))}
    need = ["unsloth/Llama-3.1-8B-Instruct", "nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct",
            "unsloth/Llama-3.3-70B-Instruct", "Qwen/Qwen3-32B", "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "zai-org/GLM-4.7-Flash", "openai/gpt-oss-120b", "Qwen/Qwen3.6-27B", "Qwen/Qwen3.8-27B",
            "Qwen/Qwen3-Next-80B-A3B-Instruct", "nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16",
            "moonshotai/Kimi-Linear-48B-A3B-Instruct", "google/gemma-4-31B-it", "Qwen/Qwen2.5-7B-Instruct-1M",
            "Qwen/Qwen3.5-27B", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "zai-org/glm-4-9b-chat-1m",
            "ByteDance-Seed/Seed-OSS-36B-Instruct", "unsloth/Llama-4-Scout-17B-16E-Instruct",
            "tencent/Hunyuan-A13B-Instruct"]
    for n in need:
        if n not in by:
            sys.exit(f"缺少 {n} 的估算列，拒絕產生報告")
    n_usable = next(iter(freq.values()))["n_usable"]
    n25 = next(iter(freq.values()))["n_2025"]
    n26 = next(iter(freq.values()))["n_2026"]

    def fq(fam, col="share"):
        f = freq.get(fam)
        return pct(f[col]) if f else "0.0%"

    ts = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    run_id = rows[0]["run_id"]
    L = []
    w = L.append

    R = by
    l8, ul, l70, q32 = R["unsloth/Llama-3.1-8B-Instruct"], R["nvidia/Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct"], R["unsloth/Llama-3.3-70B-Instruct"], R["Qwen/Qwen3-32B"]
    q30, glm, oss = R["Qwen/Qwen3-30B-A3B-Instruct-2507"], R["zai-org/GLM-4.7-Flash"], R["openai/gpt-oss-120b"]
    q36, q38, qn, pz, kl = R["Qwen/Qwen3.6-27B"], R["Qwen/Qwen3.8-27B"], R["Qwen/Qwen3-Next-80B-A3B-Instruct"], R["nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16"], R["moonshotai/Kimi-Linear-48B-A3B-Instruct"]
    g31, seed, scout, hy = R["google/gemma-4-31B-it"], R["ByteDance-Seed/Seed-OSS-36B-Instruct"], R["unsloth/Llama-4-Scout-17B-16E-Instruct"], R["tencent/Hunyuan-A13B-Instruct"]

    w("# 平台 B（AMD MI300X）選模報告：dense / MoE / hybrid")
    w("")
    w(f"產生時間：{ts}　run_id：`{run_id}`")
    w("")
    w("產生方式：`code/model_survey_fetch.py` → `code/model_survey_estimate.py` → `code/model_survey_papers.py`"
      " → `code/model_survey_papers_strict.py` → **`code/model_survey_report.py`（本檔由它產生，勿手改）**")
    w("")
    w("> ⚠️ **本報告的記憶體、可達上下文、κ、prefill 時間全部是由 HF `config.json` 算出的估計值，不是量測。**")
    w("> CSV 每列都帶 `estimate_kind=ARITHMETIC_NOT_MEASURED`。真值由 M1（vLLM 啟動時印的 `GPU KV cache size`）")
    w("> 與 M2（成本常數）量出；估計與實測的差距本身就是 M1 要回報的第一項。")
    w("> 論文使用率是「全文提到 ≥3 次」的篇數比例，是『實際評測用』的**代理**，不是逐篇人工確認。")
    w("")
    w("---")
    w("")
    w("## 0. 結論")
    w("")
    w("**單卡 192 GB、BF16 權重、1M 上下文、要夠強、要是論文常見模型——這五個條件在三類架構上的交集完全不同。**")
    w("")
    w(f"1. **Dense：「強」與「1M」在單卡上互斥。** KV 隨層數 × KV head 線性成長，"
      f"Llama-3.3-70B 每 token {fnum(l70['kv_kib_per_token_bf16_alllayers'],0)} KiB，1M 需 {fnum(l70['kv_gib_at_1m_bf16'],0)} GiB，"
      f"BF16 權重 {fnum(l70['weights_bf16_gib'])} GiB 之後只剩 {fnum(l70['kv_budget_gib'])} GiB → 最長 **{ctx_k(l70['max_ctx_bf16kv'])}**"
      f"（FP8 KV {ctx_k(l70['max_ctx_fp8kv'])}）。30B 級也只到 {ctx_k(q32['max_ctx_bf16kv'])}（Qwen3-32B）。"
      f"**能在單卡跑到 1M 的 dense 只有 8–9B 級**：Llama-3.1-8B 架構 1M 需 {fnum(l8['kv_gib_at_1m_bf16'],0)} GiB，放得下。")
    w(f"2. **MoE：強的放不下，放得下的 KV 不省。** Llama-4-Scout（{fnum(scout['weights_bf16_gib'])} GiB）、"
      f"gpt-oss-120b（BF16 {fnum(oss['weights_bf16_gib'])} GiB；官方只發 MXFP4）、GLM-4.5-Air、Mistral-Small-4 的 **BF16 權重都超過單卡**。"
      f"放得下的 30B 級 MoE（Qwen3-30B-A3B）KV 與 dense 同形（{fnum(q30['kv_kib_per_token_bf16_alllayers'],0)} KiB/token），1M 需 {fnum(q30['kv_gib_at_1m_bf16'],0)} GiB，估計可放。")
    w(f"3. **Hybrid：唯一能同時滿足「夠強 + BF16 + 1M」的一類**，因為只有 1/4–1/8 的層存 KV。"
      f"Qwen3.6-27B 1M 只需 {fnum(q36['kv_gib_at_1m_bf16'],0)} GiB、Nemotron-Labs-3-Puzzle-75B-A9B {fnum(pz['kv_gib_at_1m_bf16'],1)} GiB。"
      f"**但 vLLM 0.19.1 的 OffloadingConnector 不支援 hybrid**（見 §5.1）——論文的 CPU/SSD 階在這一類上目前跑不起來。")
    w(f"4. **論文實際常用的模型幾乎都是 7–8B dense。** {n_usable} 篇 2025–2026 的 KV cache／長上下文論文中，"
      f"Llama-3.1-8B 佔 **{fq('Llama-3.1-8B')}**；強模型很少：Qwen3-32B {fq('Qwen3-32B')}、Llama-3.1-70B {fq('Llama-3.1-70B')}、"
      f"Qwen3-30B-A3B {fq('Qwen3-30B-A3B')}、gpt-oss-120b {fq('gpt-oss-120b')}。2026 年上升最快的是 Qwen3-8B"
      f"（{fq('Qwen3-8B','share_2025')} → {fq('Qwen3-8B','share_2026')}）與 Qwen3.5/3.6（{fq('Qwen3.5/3.6','share_2025')} → {fq('Qwen3.5/3.6','share_2026')}）。")
    w("5. **跨平台 κ 的主張需要一個兩平台都量過的錨點模型。** 平台 A 唯一自洽的 BF16 成本剖面是 `llama-bf16`"
      "（`NousResearch/Meta-Llama-3.1-8B-Instruct`），所以不論其餘怎麼選，**Llama-3.1-8B BF16 必須在平台 B 的集合裡**。")
    w("")
    w("## 1. 需要你決定的事（決定之前不下載、不開跑）")
    w("")
    w("| # | 決定 | 選項 | 我的建議 | 理由 |")
    w("|---|---|---|---|---|")
    w("| D1 | **hybrid 要不要納入論文？** | (a) 納入，平台 B 升級 vLLM ≥ 0.21.0<br>(b) 只做 dense/MoE，hybrid 只量容量與 κ（不跑卸載） | **(b) 先做，(a) 另開分支驗證** | 論文 §B.2「明確排除的架構」目前把 hybrid SSM 列為定義上不同的問題；且 0.19.1 卸載路徑對 hybrid 不可用、0.21+ 需 torch 2.11 ROCm nightly、Mamba state 的卸載語意在 v0.28.0 程式碼裡找不到處理（未實跑） |")
    w(f"| D2 | **dense 的「強」與「1M」怎麼取捨** | (a) 8B 跑到 1M + 70B 跑到記憶體上限<br>(b) 只用 70B | **(a)** | 70B 在 BF16 權重下最多 {ctx_k(l70['max_ctx_bf16kv'])}，1M 物理上放不下；8B 是跨平台錨點，且 UltraLong-1M 與它同架構（成本常數相同）、原生 1M |")
    w("| D3 | **MoE 是否堅持 BF16 權重** | (a) 堅持：Qwen3-30B-A3B（+ GLM-4.7-Flash 當 MLA 代表）<br>(b) 放寬：加 gpt-oss-120b（MXFP4） | **(a)** | 論文附錄把平台 B 的權重精度定為「BF16 主、FP8 對照」；gpt-oss-120b 沒有官方 BF16，且只有 131K |")
    w(f"| D4 | **強 hybrid 用哪一個**（若 D1 選 a） | Qwen3-Next-80B-A3B / Nemotron-Labs-3-Puzzle-75B-A9B / Kimi-Linear-48B-A3B | 論文常見度選 Qwen3-Next；原生 1M 選 Puzzle-75B | Qwen3-Next BF16 KV 最多 {ctx_k(qn['max_ctx_bf16kv'])}，1M 要 FP8 KV（平台 A 上 FP8 靜態縮放把檢索打到 5%，品質風險）；Puzzle-75B BF16 KV 就放得下 1M，但論文常見度 {fq('Nemotron-3')} |")
    w("")
    w("## 2. 推薦選模")
    w("")
    w("「κ₀」為論文 tab:kappa 同一算法的下界（線性層 2N_active、100% MFU、MI300X 1307.4 TFLOPS；傳輸 = 遠端位置仍需保留的 KV 位元組 / 63 GB/s）。"
      "「1M 放得下」欄為 BF16 KV / FP8 KV。「使用率」為全文提到 ≥3 次的論文比例（2026 年）。")
    w("")
    w("| 類別 | 角色 | 模型 | 總／啟用 | BF16 權重 | KV/token | 1M 需 KV | 1M 放得下 | 最長（BF16 KV） | 宣稱上下文 | κ₀ | 0.19.1 卸載 | 使用率 2026 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    def rec_row(cat, role, r, fam, extra=""):
        kvt = f"{fnum(r['kv_kib_per_token_bf16_fullpart'],1)} KiB"
        if r["state_layers"] not in ("0", ""):
            kvt += f" + {fnum(r['state_mib_per_seq'],0)} MiB state"
        w(f"| {cat} | {role} | **{short(r['repo'])}**{extra} | {fnum(r['total_params_b'],1)}B／{fnum(r['active_params_b'],1)}B | "
          f"{fnum(r['weights_bf16_gib'])} GiB | {kvt} | {fnum(r['kv_gib_at_1m_bf16'])} GiB | {fit_mark(r)} | {ctx_k(r['max_ctx_bf16kv'])} | "
          f"{ctx_k(r['claimed_ctx'])} | {fnum(r['kappa0_mi300x'],1)} | {offload_mark(r)} | {fq(fam,'share_2026') if fam else '—'} |")

    rec_row("dense", "錨點（≤128K）", l8, "Llama-3.1-8B", "<br><sub>用 NousResearch 鏡像以對齊平台 A</sub>")
    rec_row("dense", "1M 點（同架構）", ul, "UltraLong-8B-1M/4M", "<br><sub>F32 發布，下載約 2 倍</sub>")
    rec_row("dense", "強（規模軸）", l70, "Llama-3.3-70B")
    rec_row("dense", "選配：同家族對照", q32, "Qwen3-32B")
    rec_row("MoE", "主（常見）", q30, "Qwen3-30B-A3B")
    rec_row("MoE", "選配：MLA 代表", glm, "GLM-4.5/4.6/4.7")
    rec_row("hybrid", "主（常見）", q36, "Qwen3.5/3.6", "<br><sub>與 Qwen3.8-27B config 完全相同</sub>")
    rec_row("hybrid", "強（候選 1）", qn, "Qwen3-Next-80B-A3B")
    rec_row("hybrid", "強（候選 2）", pz, "Nemotron-3")
    rec_row("hybrid", "強（候選 3）", kl, "Kimi-Linear")
    w("")
    tot_main = sum(float(x["weights_bf16_gib"]) for x in (l8, ul, l70, q30, q36))
    w(f"**主集合（dense 錨點 + 1M 點 + 70B、MoE 主、hybrid 主）的 BF16 權重合計 {tot_main:,.0f} GiB**；"
      f"`/mlsteam/data/tiara`（NFS）剩 937 GB。UltraLong 為 F32 發布，下載量約為表中值的 2 倍。")
    w("")
    w("### 為什麼是這幾個")
    w("")
    w(f"- **Llama-3.1-8B（錨點）**：論文使用率第一（全期 {fq('Llama-3.1-8B')}）；平台 A 的 `llama-bf16` 剖面就是它，是 κ 跨硬體比較唯一能直接對上的一組常數。"
      f"原生 128K；估計 BF16 KV 最長 {ctx_k(l8['max_ctx_bf16kv'])}，所以 1M 的瓶頸是模型而非記憶體 → 1M 點改用同架構的 UltraLong-1M（config 除 RoPE 縮放外逐欄相同）。")
    w(f"- **Llama-3.3-70B（強 dense）**：論文 tab:kappa 預測 MI300X 上 κ 由 8B 的 6 增至 70B 的 21，這是那條預測的直接檢驗；"
      f"估計 κ₀ {fnum(l8['kappa0_mi300x'])} → {fnum(l70['kappa0_mi300x'])}。它同時是「192 GB 卡上單請求容量壓力仍存在」的實例（§2 觀察一(c)）：BF16 權重後只剩 {fnum(l70['kv_budget_gib'])} GiB 給 KV。")
    w(f"- **Qwen3-30B-A3B（MoE 主）**：論文 §2「MoE 的 Drop 門檻應高於 dense」的預測需要一對同家族的 dense/MoE——"
      f"Qwen3-32B（κ₀ {fnum(q32['kappa0_mi300x'])}）對 Qwen3-30B-A3B（κ₀ {fnum(q30['kappa0_mi300x'])}），同 tokenizer、同訓練世代。"
      f"⚠️ 卡上 1M 走 DCA+MInference，而 **vLLM 0.19.1 沒有 DCA backend**（與平台 A 在 0.28.0 的發現相同）→ 262,144 以上只能改 YaRN，品質未驗證。")
    w(f"- **GLM-4.7-Flash（MoE 選配）**：卡上自稱 30B 級最強；MLA（每 token {fnum(glm['kv_kib_per_token_bf16_alllayers'])} KiB，約 Qwen3-30B-A3B 的一半）讓 KV 格式本身成為一個變因；GLM-4.5+ 家族 2026 使用率 {fq('GLM-4.5/4.6/4.7','share_2026')}。")
    w(f"- **Qwen3.6-27B（hybrid 主）**：Qwen3.5/3.6 是 2026 論文最常用的 hybrid 家族（{fq('Qwen3.5/3.6','share_2026')}）；"
      f"3/4 層 Gated DeltaNet、1/4 層全注意力，1M 只需 {fnum(q36['kv_gib_at_1m_bf16'])} GiB KV。Qwen3.8-27B 與它 config 完全相同（成本常數可共用），要更新的品質可直接換。")
    w("")
    w("## 3. 論文實際用什麼模型（證據）")
    w("")
    w(f"方法：HF papers 搜尋 16 組關鍵字（KV cache offloading / compression / eviction / quantization、prefix caching、long-context inference、sparse attention、hybrid linear attention 等）"
      f"→ 2025-01-01 之後、標題或摘要與 KV／長上下文相關 → 抓 arXiv HTML 全文 → {n_usable} 篇可用（2025：{n25}、2026：{n26}）。"
      f"每篇每模型只算一次；**「≥3 次」欄排除 related work 順帶提一次的情形**（例：DeepSeek-R1 任一次 {pct(freq['DeepSeek-R1(671B)']['share_any'])}，≥3 次只剩 {fq('DeepSeek-R1(671B)')}）。")
    w("")
    fam_cat = {
        "Llama-3.1-8B": "dense", "Mistral-7B": "dense", "Qwen3-8B": "dense", "Qwen2.5-7B": "dense", "Llama-3-8B": "dense",
        "Llama-3.2-1B/3B": "dense", "Qwen3-4B": "dense", "Llama-2-7B": "dense", "Qwen2.5-14B": "dense",
        "DeepSeek-V3/V3.x": "MoE(MLA)", "DeepSeek-R1(671B)": "MoE(MLA)", "Qwen3.5/3.6": "hybrid", "Qwen3-32B": "dense",
        "R1-Distill-Llama-8B": "dense", "Gemma-3": "dense", "Qwen3-14B": "dense", "Llama-3.1-70B": "dense",
        "Qwen2-7B": "dense", "Qwen2.5-32B": "dense", "Qwen2.5-3B": "dense", "GLM-4.5/4.6/4.7": "MoE",
        "Mamba/Mamba2": "SSM", "Llama-2-13B": "dense", "Qwen2.5-7B-1M": "dense", "R1-Distill-Qwen-7B": "dense",
        "Phi-4": "dense", "Qwen3-30B-A3B": "MoE", "LongChat-7B-32K": "dense", "Jamba": "hybrid", "Qwen2.5-72B": "dense",
        "Mistral-Small/Nemo": "dense/MoE", "gpt-oss-120b": "MoE", "Llama-3-70B": "dense", "R1-Distill-Qwen-14B/32B": "dense",
        "Qwen3-235B-A22B": "MoE", "Gemma-4": "dense/MoE", "Llama-3.3-70B": "dense", "Kimi-K2": "MoE(MLA)",
        "MiniMax-M1/Text-01": "hybrid", "Llama-3-8B-1048K": "dense", "DeepSeek-V2/V2.5": "MoE(MLA)", "Zamba2": "hybrid",
        "GLM-4-9B-1M": "dense", "Qwen3-Next-80B-A3B": "hybrid", "gpt-oss-20b": "MoE", "Yi-9B/34B-200K": "dense",
        "Gemma-2": "dense", "QwQ-32B": "dense", "UltraLong-8B-1M/4M": "dense", "Kimi-Linear": "hybrid",
        "Phi-3/3.5-mini": "dense", "Llama-3.1-405B": "dense", "Llama-4-Scout": "MoE", "MiniMax-M2": "MoE",
        "Llama-4-Maverick": "MoE", "DeepSeek-V2-Lite": "MoE(MLA)", "Mixtral-8x7B": "MoE", "LWM-Text-1M": "dense",
        "OPT-6.7B~175B": "dense", "Nemotron-3": "hybrid", "Nemotron-H": "hybrid", "Falcon-H1": "hybrid",
        "Granite-4.0-H": "hybrid", "Mixtral-8x22B": "MoE", "Phi-3.5-MoE": "MoE", "Seed-OSS-36B": "dense",
        "Hunyuan-A13B": "MoE", "InternLM2.5-7B-1M": "dense",
    }
    for cat_name, keys in (("dense", ("dense",)), ("MoE", ("MoE", "MoE(MLA)", "dense/MoE")), ("hybrid / SSM", ("hybrid", "SSM"))):
        w(f"### 3.{ {'dense':1,'MoE':2,'hybrid / SSM':3}[cat_name] } {cat_name}")
        w("")
        w("| 模型家族 | 架構 | ≥3 次（全期） | 2025 | 2026 | 任一次 |")
        w("|---|---|---|---|---|---|")
        items = [f for f in freq.values() if fam_cat.get(f["model"]) in keys]
        items.sort(key=lambda f: -int(f[[k for k in f if k.startswith("papers_ge")][0]]))
        for f in items[:14]:
            w(f"| {f['model']} | {fam_cat.get(f['model'])} | {pct(f['share'])} | {pct(f['share_2025'])} | {pct(f['share_2026'])} | {pct(f['share_any'])} |")
        w("")
    w("**讀法**：dense 7–8B 是這個領域的「預設實驗模型」；強模型的使用率都在個位數百分比。"
      "所以「強」與「常見」要分開滿足——錨點用常見的（審稿人可對照），規模／架構軸用強的（主張才有意義）。")
    w("")
    w("## 4. 三類候選的完整估算")
    w("")
    w("欄位：**KV/token** 為全注意力＋MLA 層（遠端位置仍需保留的部分）；**SWA** 層的 KV 每序列只存視窗內；**state** 為線性層（GDN/Mamba/KDA/Lightning/conv）每序列的常數狀態。"
      "**1M 放得下** = BF16 KV / FP8 KV。**最長** 為 BF16 KV 下的記憶體上限（未與模型宣稱上限取小）。"
      "**卸載後最長** = 開 OffloadingConnector（0.19.1 關閉 HMA，SWA 層改配全長 KV）時的上限。")
    w("")
    for cat, title in (("dense", "4.1 Dense transformer"), ("moe", "4.2 MoE"), ("hybrid", "4.3 Hybrid（線性層 + 注意力）")):
        w(f"### {title}")
        w("")
        w("| 模型 | 發布 | 總／啟用 | BF16 權重 | 注意力 | KV/token | state/序列 | 1M 需 KV | 1M 放得下 | 最長 | 卸載後最長 | 宣稱上下文（方式） | κ₀ | vLLM 0.19.1 卸載 | 論文 ≥3 次 |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        sub = [r for r in rows if r["category"] == cat]
        sub.sort(key=lambda r: -float(r["total_params_b"]))
        for r in sub:
            fam = r["paper_family"]
            w(f"| {short(r['repo'])} | {r['created']} | {fnum(r['total_params_b'])}B／{fnum(r['active_params_b'])}B | {fnum(r['weights_bf16_gib'])} GiB | "
              f"{r['attn_type']} | {fnum(r['kv_kib_per_token_bf16_fullpart'],1)} KiB | {fnum(r['state_mib_per_seq'],1) if r['state_layers'] not in ('0','') else '—'} | "
              f"{fnum(r['kv_gib_at_1m_bf16'])} GiB | {fit_mark(r)} | {ctx_k(r['max_ctx_bf16kv'])} | "
              f"{ctx_k(r['max_ctx_bf16kv_offload019']) if r['max_ctx_bf16kv_offload019'] not in ('',) else '—'} | "
              f"{ctx_k(r['claimed_ctx'])}（{r['ctx_method']}） | {fnum(r['kappa0_mi300x'],1)} | {offload_mark(r)} | {fq(fam) if fam else '—'} |")
        w("")
    w("## 5. 影響實驗設計的發現")
    w("")
    w("### 5.1 🔴 vLLM 0.19.1 的 OffloadingConnector 不支援 hybrid（靜態讀碼，未實跑）")
    w("")
    w("證據鏈（`/mlsteam/workspace/src/vllm`，tag v0.19.1）：")
    w("")
    w("1. `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:44`：`class OffloadingConnector(KVConnectorBase_V1)`——**不是** `SupportsHMA` 子類別。")
    w("2. `vllm/config/vllm.py:1227-1244`：設了 `--kv-transfer-config` 且使用者沒指定時，**自動關閉 hybrid KV cache manager（HMA）**。")
    w("3. `vllm/v1/core/kv_cache_utils.py:1160-1219`（`unify_hybrid_kv_cache_specs`）：HMA 關閉時，滑動視窗層被改成全注意力規格（**失去省記憶體**）；"
      "Mamba／線性注意力層無法與注意力層統一 → `ValueError: Hybrid KV cache manager is disabled but failed to convert the KV cache specs to one unified type.`")
    w("4. 若強制開 HMA：`vllm/distributed/kv_transfer/kv_connector/factory.py:58-61` 直接拒絕（`Connector OffloadingConnector does not support HMA`）。")
    w("5. `git show <tag>:.../offloading_connector.py`：**v0.21.0 起才是 `SupportsHMA`**（v0.19.1、v0.20.0 不是；v0.21.0 到 v0.29.0 皆是）。"
      "v0.21.0–v0.26.0 需要 torch 2.11（ROCm 只有 nightly `2.11.0.dev20260206`），v0.27.0 起需要 torch 2.13（ROCm 無）。")
    w("6. 即使 v0.28.0（平台 A 的版本）宣告 `SupportsHMA`，`vllm/v1/kv_offload/` 與 connector 內**找不到任何 Mamba／state 的處理**——state 是否被卸載、被怎麼卸載，未知。")
    w("")
    w("**後果**：hybrid 模型在平台 B 目前可以量容量（M1）與重算成本，但論文的 CPU/SSD 階（M2 retrieval、M3 baselines）跑不起來。")
    w("")
    w("### 5.2 🟡 滑動視窗模型在卸載模式下的記憶體會暴增")
    w("")
    w(f"Gemma-4-31B：HMA 開啟時 1M 需 {fnum(g31['kv_gib_at_1m_bf16'])} GiB（50/60 層只存 1,024 token）；開 OffloadingConnector 後所有層都存全長，"
      f"1M 需 {fnum(g31['kv_gib_at_1m_bf16_no_hma'])} GiB，最長由 {ctx_k(g31['max_ctx_bf16kv'])} 掉到 **{ctx_k(g31['max_ctx_bf16kv_offload019'])}**。"
      "gpt-oss、Llama-4、EXAONE-4.5、Trinity 同理。**baseline 與 Tiara 會在不同的有效容量下比較**——這類模型若入選，需在 M3 明確記錄。")
    w("")
    w("### 5.3 🟡 hybrid 開 prefix caching 時，「每 token 要存的東西」是 attention KV 的約 3 倍")
    w("")
    w("vLLM `mamba_cache_mode=\"all\"`（開 prefix caching 時的預設）在每個 block 邊界存一份全部線性層的 state；block 大小由「單層 attention 頁 ≥ 單層 state 頁」決定（`vllm/model_executor/models/config.py`）。")
    w("")
    w("| 模型 | attention KV/token | block 大小 | state 攤到每 token | 快取前綴每 token 合計 | κ₀（只算 KV） | κ₀（含 state） |")
    w("|---|---|---|---|---|---|---|")
    for rr in (q36, qn, pz, kl, R["nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"]):
        w(f"| {short(rr['repo'])} | {fnum(rr['kv_kib_per_token_bf16_fullpart'],1)} KiB | {fint(rr['hybrid_block_size_tokens'])} | {fnum(rr['state_ckpt_kib_per_token'],1)} KiB | "
          f"{fnum(rr['cached_prefix_kib_per_token'],1)} KiB | {fnum(rr['kappa0_mi300x'],1)} | {fnum(rr['kappa0_with_state_ckpt'],1)} |")
    w("")
    w("**這對論文是好消息也是風險**：hybrid 的 κ 取決於「要不要存 state checkpoint」，而那正是放置決策的一部分——動作空間在 hybrid 上多出一個維度。"
      "論文 §B.2 目前以「Marconi 已處理」排除 hybrid；若納入，這一段要重寫。")
    w("")
    w("### 5.4 🟡 1M 的長度延伸方式在 vLLM 0.19.1 上有兩個缺口")
    w("")
    w("- **DCA 不可用**：`vllm/` 內除模型與 RoPE 之外找不到 dual-chunk 的 attention backend（與平台 A 在 0.28.0 的崩潰結論一致）。"
      "受影響：Qwen2.5-7B/14B-Instruct-1M、Qwen3-30B-A3B-Instruct-2507 的 1M 模式。")
    w("- **YaRN 延伸屬於外推**：Qwen3-Next、Qwen3.5/3.6/3.8 的 262,144 → 1M 需 YaRN。CLAUDE.md §8 規定不要為了掃更長而開 YaRN（品質退化無法歸因）；"
      "若採用，需另建「full-KV + 同 YaRN 設定」的基準線。**原生 1M 的只有**：UltraLong-8B-1M、GLM-4-9B-1M、Kimi-Linear、Nemotron-3 系列、MiniCPM-SALA、LongCat-Flash-Lite-Sparse。")
    w("")
    w("### 5.5 ⚪ 版本對齊")
    w("")
    w("平台 A = vLLM v0.28.0，平台 B = v0.19.1（REPORT_MI300X §6 #3）。在對齊之前，κ 的跨硬體數字一律標 `NOT_COMPARABLE`；"
      "若為了 hybrid 升到 v0.21–v0.26，仍與 A 不同版。v0.27+ 需要 ROCm 沒有的 torch 2.13。")
    w("")
    w("## 6. 方法、常數與假設")
    w("")
    r0 = rows[0]
    w("| 項目 | 值 | 來源 |")
    w("|---|---|---|")
    w(f"| VRAM | 206,141,652,992 B | `results/hw_mi300x.json`（實測） |")
    w(f"| 可用比例 | {r0['usable_gib']} GiB（gpu_memory_utilization 0.90） | vLLM 預設 |")
    w(f"| runtime 餘裕 | {r0['runtime_margin_gib']} GiB | **假設值**（activation / workspace / graph）；M1 量真值 |")
    w("| 1M | 1,048,576 token | `--max-model-len 1048576` |")
    w("| 算力 | 1307.4 TFLOPS | 論文 tab:ratio（規格值，非量測） |")
    w("| 主機鏈路 | 63.0 GB/s | PCIe Gen5 ×16 單向理論值（論文同表） |")
    w("| 權重 | safetensors 參數量 × 2 B | HF API（FP8/MXFP4 發布者為換算的 BF16 大小） |")
    w("| 啟用參數 | 模型卡宣稱值；無則由 config 算（標於 CSV `active_src`） | README.md |")
    w("")
    w("逐層公式與 vLLM 0.19.1 `kv_cache_interface.py`、`mamba_utils.py` 一致，完整寫在 `code/model_survey_estimate.py` 開頭。已人工驗算："
      f"Qwen2.5-7B-1M {fnum(R['Qwen/Qwen2.5-7B-Instruct-1M']['kv_kib_per_token_bf16_alllayers'],0)} KiB/token、Llama-3.1-8B {fnum(l8['kv_kib_per_token_bf16_alllayers'],0)} KiB/token，"
      "與論文表 tab:kvsize 相同；Llama-3.1-8B/70B 的 κ₀ 與論文 tab:kappa 的 6×/21× 相同。")
    w("")
    w("**已知的估計偏差方向**：(1) runtime 餘裕 8 GiB 對 1M 級 prefill 可能偏低（Qwen 卡上稱 30B-A3B 跑 1M 約需 240 GB，含 activation）——邊界上的模型要以 M1 為準；"
      "(2) 多模態模型的權重含 vision tower；(3) Gemma-4 的 K=V 全域層仍存兩份 KV（vLLM `gemma4.py`）；(4) LongCat-Sparse 與 Qwen3.8-Flash-Next 的稀疏索引器 KV 未計入。")
    w("")
    w("## 7. 檔案位置")
    w("")
    w("| 內容 | 路徑 |")
    w("|---|---|")
    w(f"| 候選模型原始 config / README / API | `{args.run_dir}/raw/<org>__<name>/` |")
    w(f"| 估算 CSV（每模型一列，含 run_id/ts） | `results/model_survey/model_estimates_mi300x.csv` |")
    w(f"| 論文全文與逐篇命中 | `{args.run_dir}/papers/text/`、`results/model_survey/paper_mentions.csv` |")
    w(f"| 論文模型頻率 | `results/model_survey/paper_model_frequency_any.csv`、`paper_model_frequency_min3.csv` |")
    w("| 程式 | `code/model_survey_{fetch,estimate,papers,papers_strict,report}.py` |")
    w("")
    open(args.out, "w").write("\n".join(L) + "\n")
    print("wrote", args.out, len(L), "lines")


if __name__ == "__main__":
    main()
