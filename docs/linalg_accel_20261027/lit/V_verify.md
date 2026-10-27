# V　LA–LF 的獨立複核（2026-10-27）

- **執行者**：Claude，zero-context verifier subagent（沒看過 LA–LF 是怎麼寫的；只讀了委託者列的主張與這六個檔裡對應的句子）。**這是同一個模型家族的另一個 instance，不是 cross-model review。**
- **方法**：每個主張都回到原始來源：arXiv PDF（`curl` → `pdftotext -layout`，頁碼＝PDF 頁）、arXiv abs 頁的 `citation_*` meta、HF raw `config.json`、GitHub raw 檔（指定 commit）、`gh issue/pr/release view`、官方文件 HTML。下載檔放 `<SCRATCH>/verify/`，只用 `python3 -I` 讀，沒有執行任何下載內容。沒有在這台 Mac 跑任何實驗；只有算術。
- **證據等級**沿用 RULES.md：〔全文〕＝在原文該位置讀到原句／原表；〔程式碼〕〔文件〕〔算術〕〔判讀〕〔二手〕。

**一句話結論**：58 條主張中 **53 OK、2 WRONG（已就地修正）、3 PARTIAL、0 CANNOT_VERIFY**。scoop 風險論文（LD）的描述**全部與原文一致**，沒有任何一篇的實質內容需要改；特別是 **BellowsKV 確實沒有 CPU／SSD 卸載或還原、評測只到 32K**。

---

## 1. 逐條結果

| # | 檔 | 主張（摘要） | 判定 | 原文位置 | 深度 |
|--:|:--|:--|:--|:--|:--|
| 1 | LA | SSD 不涵蓋 softmax attention，也不涵蓋任何沒有有限特徵映射 ψ 的 attention 核轉換 | OK | 2405.21060 §10.3，PDF p.38（原句「does not generalize standard softmax attention, or any other transformation…finite feature map」） | 〔全文〕 |
| 2 | LA | Thm 5.2：有界階數自迴歸的結構化遮罩注意力，遮罩必為半可分 | OK | 同上 PDF p.17，Theorem 5.2（對象是 SMA，Def. 4.2；LA 摘要句未寫「SMA」但卡片 L84 有寫） | 〔全文〕 |
| 3 | LA | S4：HiPPO 直接對角化元素隨 N 指數變大，數值不可行 | OK | 2111.00396 §3.1，PDF p.5 | 〔全文〕 |
| 4 | LA | HyperAttention Table 1 數字：28 層全換 synthetic 84.00→8.33、single-qa 80.63→27.07；其餘列 | OK | 2310.05869 Table 1，PDF p.10（六欄五列逐字相符） | 〔全文〕 |
| 5 | LA | Table 1 的設定是「LongBench 長度 >32,768 的子集截到 32,768」；欄名「single-doc QA」 | PARTIAL | PDF p.9：「選 >32,768 的子集並截斷」這句是寫給 **perplexity／speedup 實驗（Fig. 3）** 的；Table 1 的資料設定原文沒有另外寫明。表中的值是各任務類別的分數（few-shot 186.58、code 102.06 超過 100，**不是百分比**，類別內如何加總原文沒說）。數字本身沒錯 | 〔全文〕 |
| 6 | LA | Kimi Linear RULER-128K 84.3、3.98× | OK | 2510.26692 Fig. 1a 圖說，PDF p.1 | 〔全文〕 |
| 7 | LA | 1M 時 TPOT 1.84 vs 11.48 ms（6.3×） | OK | Fig. 1b 圖說，PDF p.1（原文說這是「enables larger batches」的結果） | 〔全文〕 |
| 8 | LA | 公平比較 1.4T token | OK | PDF p.1 Fig. 1 圖說、p.2 貢獻、p.11 表 | 〔全文〕 |
| 9 | LA | KDA:全注意力＝3:1 | OK | PDF p.2（「uniform 3:1 ratio」） | 〔全文〕 |
| 10 | LB | Kimi Linear prefill 比全 MLA 只快 2.3×（512K）、2.9×（1M），batch 1 | OK | PDF p.13 §5.6 內文＋Fig. 7a | 〔全文〕 |
| 11 | LB | decode：1M 時 TPOT 1.84 vs 11.48 ms，6.3×（Fig. 1b、p.16） | PARTIAL | 1.84／11.48 只出現在 p.1 圖說；p.16 稱 6.3× 為「theoretical decoding speedup」、來自更大 batch；同頁又寫 **batch 1 的 decode 在 1M 只快 2.3×（Fig. 7b）**，p.13 則寫「6× faster」。LB 沒寫 batch 前提，容易被讀成同條件下 6.3× | 〔全文〕 |
| 12 | LA | Alman–Song：B＝o(√log n) 有 n^{1+o(1)} 演算法（Thm 1.4）；B＝C_b√log n 時 SETH 下無 O(n^{2−q})（Thm 1.3）；A＝exp(QKᵀ/d) | OK | 2302.13214 Def. 1.1（PDF p.2）、Thm 1.3＝下界、Thm 1.4＝上界（PDF p.3） | 〔全文〕 |
| 13 | LA | 總表列：「logit 達 Θ(√log n) 量級時 SETH 下不可能」 | **WRONG（已改）** | Thm 1.3 的參數是 **Q、K 元素** B＝C_b√log n；logit＝QKᵀ/d ≤ B²，所以 logit 是 Θ(log n)。LA 自己 L149 的算術也是這樣寫，只有 L290 表格寫錯 | 〔全文〕 |
| 14 | LA | needle 拿 ≥1/2 權重需 Δ ≥ ln(n−1)：11.78（128K）、12.48（256K） | OK | 重算：ln 131,071＝11.783；ln 262,143＝12.477 | 〔算術〕 |
| 15 | LE | Fast RoPE（Alman–Song 2025）說舊技巧不能處理 RoPE | OK | 2505.11892 p.1（「previous techniques…no longer seem to work」）、p.3（[AS23] cannot be applied）、p.5（polynomial method「fundamentally cannot apply」，因為 d＝1 時 M 可為任意 Toeplitz） | 〔全文〕 |
| 16 | LE | Fast RoPE Thm 1.3（上界）、Thm 1.4（下界，B＝Ω(√log n)），p.4；原文無實驗（p.16） | OK | PDF p.4；附錄 D Limitations，PDF p.16 | 〔全文〕 |
| 17 | LE | 〔判讀〕標準 RoPE 的 R_{n−m}＝R_mᵀR_n → 等於先旋轉 Q、K 再做標準 attention；每個 2×2 旋轉讓 ℓ∞ 最多放大 √2 → 舊演算法直接適用（B→√2B）；原文的「不能用」針對它的一般化 W_{i−j} | OK〔判讀〕 | 我的獨立推導：(1) 旋轉群同態 R(aθ)ᵀR(bθ)＝R((b−a)θ)，所以 qᵀR_{n−m}k＝(R_m q)ᵀ(R_n k)，exp 前的矩陣＝Q′K′ᵀ，秩 ≤ d；(2) |x cosθ − y sinθ| ≤ √(x²+y²) ≤ √2·max(|x|,|y|)（Cauchy–Schwarz），所以 ‖Q′‖∞ ≤ √2B，o(√log n) 不變；(3) 原文 Def. 1.1 允許任意 supp⊂S、‖W‖∞≤1 的 W_{i−j}（d＝1 時可造出任意 Toeplitz，p.5），那個一般化不一定能分解，才是「舊技巧不能用」的情況。推導成立。補充：實作上 RoPE 本來就是先旋轉 q、k 再做標準 attention，所以這個結論在工程上也自然 | 〔全文〕＋〔判讀〕 |
| 18 | LE | Def. 1.1 寫 /√d、Remark 1.2 又說 exp 裡是 1/d 正規化，原文前後不一致 | OK | PDF p.4 兩處原文確實如此 | 〔全文〕 |
| 19 | LC | HCache＝arXiv 2410.05004、EuroSys 2025 | OK | arXiv abs：標題「Fast State Restoration in LLM Serving with HCache」，Comments「EuroSys 2025」，2024-10-07 | 〔文件〕 |
| 20 | LC | HCache 只支援 MHA；MQA／GQA 要先把 hidden 投影到低秩、需改模型結構、超出範圍（p.13） | OK | PDF p.13 右欄（§7 段落「MQA and GQA」），列 Llama2、Gemma、Phi2、Qwen1.5 | 〔全文〕 |
| 21 | LC | hidden／KV 比：Llama-3.1-8B 2.0、Qwen2.5-7B 3.5、Qwen2.5-14B 2.5 | OK | HF raw config：Llama（unsloth 鏡像；官方 repo gated）4096／(2·8·128)＝2.0；Qwen2.5-7B(-Instruct, -1M) 3584／(2·4·128)＝3.5；Qwen2.5-14B(-Instruct, -1M) 5120／(2·8·128)＝2.5；層數 32／28／48 | 〔程式碼〕＋〔算術〕 |
| 22 | LC | KV：Llama 128K 16.0 GiB、Qwen-7B 256K 14.0 GiB、Qwen-14B 256K 48.0 GiB | OK | 重算：2048·2B·32·131072＝16.0 GiB；1024·2·28·262144＝14.0；2048·2·48·262144＝48.0 | 〔算術〕 |
| 23 | LC | HGCA 圖 6（p.5）：query 1 時 CPU 贏、query 32 時打平；PCIe 主導 | OK | 2507.03153 PDF p.5 內文（「query size of 1…slower」「query size of 32…matches CPU」「In all configurations, KV cache transfers…」） | 〔全文〕（圖本身未轉 PNG 看） |
| 24 | LC | SparseX p.14 明寫不是多層快取、全在 GPU 記憶體 | OK | 2606.01751v2 PDF p.14（「not a multi-level cache system that moves data across disk, CPU, and GPU」） | 〔全文〕 |
| 25 | LC | KVLinC Table 1：QuaRot 式在 Qwen2.5-7B 困惑度 3380 | OK | 2510.05373 Table 1，PDF p.7：Quarot（2.46-bit 平均）Wikitext 3380.0、GSM8K 0.1 | 〔全文〕 |
| 26 | LC | KVLinC Table 3：Llama-3.1-8B 只換「K 逐 channel＋V Hadamard 逐 token」GSM8K 34.1→36.9 | OK | Table 3，PDF p.8（KIVI 34.1；QC(K),QT(V H) 36.9） | 〔全文〕 |
| 27 | LD | **BellowsKV**（2610.35023）：存在；標題；方法＝n 層隱藏狀態疊成一個 latent、每層固定上投影讀回 K／V、降秩迴歸封閉解、R＝p 固定改量化、加 whisper；只在 GPU、只到 32K、無卸載與還原；Table 1 13.84× 98.0% vs xKV 16× 83.5%；H100 32K decode 0.995× | OK | abs 頁：「BellowsKV: One Latent for Many Layers, Expanded on Demand」，Jiamu Zhang、Hanjie Chen（Rice），2026-10-26 v1；Eq. 1–3 p.5；§3.3 p.6（R＝p、INT8／INT4、whisper）；Protocol B「4k–32k-token inputs」p.6；Table 1 p.7；§4.4 p.9。全文 37 頁 grep「offload／CPU／SSD／PCIe／restore／swap／disk」**0 筆**；「128K」只在引言出現。LD 沒提的：也測了 Qwen3-32B、Qwen3-30B-A3B（p.6），另有 BellowsKV-Evict 變體——不影響 scoop 判斷 | 〔全文〕 |
| 28 | LD | **RISK-KV**（2610.29544）：相鄰層 KV 錨＋跨層嶺迴歸預測注意力輸入、摺入 W_K／W_V；用 NLL 風險選層；≤1.5B（Qwen2.5-0.5B／1.5B、TinyLlama-1.1B）；512 token 歷史 1.51–3.31×；省 18.2–33.3% | OK | abs：單一作者 Wang Yuekai（哈工大），2026-10-21，10 頁；PDF p.1 摘要、p.2 模型清單；全文無 7B／8B、無 Cake | 〔全文〕 |
| 29 | LD | **OrbitWeaver**（2610.29183）：衛星 VLM 斷電恢復、隱藏狀態＋KV 區塊式檢查點；Jetson Orin；對 SONIC-layer／SONIC-chunk／HCache 最多 4.07／2.89／2.67×；HCache 自行重做 | OK | arXiv 標題「Resuming Vision-Language Inference across Onboard Interruptions」（OrbitWeaver 是摘要裡的系統名），Huang 等（北大／北郵），2026-10-21；PDF p.10（4.07／2.89／2.67×、「HCache does not open-source…we reimplement」）；全文無 Cake、無 128K | 〔全文〕 |
| 30 | LD | **Attention Matching**（2602.16284）：OMP 選 key＋NNLS 偏差＋最小平方 V；最多 50×；LongHealth 約 60K；Table 3 Gemma-3-12B 60K 單張 H200 | OK | abs：Zweiger、Fu、Guo、Kim，v1 2026-02-18、v2 2026-05-26；PDF p.3–4（NNLS、least squares、OMP）、p.8 Table 3 圖說 | 〔全文〕 |
| 31 | LD | **Qwen3.8-Next**（2608.30320）：每 4 層 1 層 full、其餘 GDN；保留 RoPE（NoPE 後訓練後更常停不下來）；Table 1 49.87／51.15／53.81（25B-A3B、32K）；Table 3 RULER 512K–1M 90.08→93.00、MRCR 1M 20.71→26.44；QSA 1M prefill 7.6×、decode 4.9×；36 位作者 | OK | abs：全名「On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability」（LD 標題截短）、2026-08-31、36 位作者；PDF p.1、p.2、p.4、p.5 Table 1、p.8 Table 3。註：論文主模型名是 Qwen3.8-Flash-Next（125B-A6B） | 〔全文〕 |
| 32 | LD | **Nemotron 3 Super**（2604.12374）：120B-A12B、1M；2.2×／7.5× 吞吐；RULER 1M 91.64／91.33／22.30、AA-LCR 58.31／66.90／51.00（p.31）；base RULER 64K 92.26→1M 71.00（p.17） | OK | abs 2026-04-14；PDF p.1 摘要；p.31 表頭依序 N-3-Super、Qwen3.5-122B-A10B、GPT-OSS-120B；p.17 表頭第一欄是 N-3-Super Base | 〔全文〕 |
| 33 | LD | **MSA**（2606.13392）：選稀疏 softmax 是為沿用既有軟硬體；28.4×、H800 14.2×／7.6×；RULER-128K 72.00 vs 72.12、HELMET 46.53 vs 45.93 | OK | abs：「MiniMax Sparse Attention」，Lai 等 17 人，v1 2026-06-11、v2 2026-06-12；PDF p.1、p.2、p.12 | 〔全文〕 |
| 34 | LD | MiniMax M2 部落格（2025-10-30）理由：評測抓不到多跳推理缺陷、基礎設施不成熟（線性架構訓練也 memory-bound）、數值精度敏感、低精度狀態儲存／prefix caching／speculative decoding；理論交叉點幾千 token；附錄 SWA 混合越長越差 | OK | HF 社群文章「Why Did MiniMax M2 End Up as a Full Attention Model?」，Published October 30, 2025（curl 原頁） | 〔全文〕 |
| 35 | LD | vLLM v0.32.0 已出貨 Hadamard 旋轉低位元 KV（TurboQuant `_build_hadamard`、`int4_per_token_head` RHT、UltraQuant K 寫入時 Hadamard、V 不轉） | OK | `gh release view v0.32.0`：2026-10-18 發布，tag→`5df63a7203cd`；release notes 列 `int4_per_token_head` KV（#56198）與 `ultraquant_4bit`（#57057）。該 commit 原始碼：`turboquant_attn.py` L105–116 `_build_hadamard`、L409 `_ensure_on_device`；`int4_per_token_head.py` L5「pre-rotates with a single RHT」、L866–868 寫入前 K、V 都做 RHT；`ultraquant/format.py` L17「No V rotation. K is Hadamard-rotated at store.」 | 〔程式碼〕＋〔文件〕 |
| 36 | LB | vLLM issue #63565（2026-10-25 開）：稀疏保留只在 prompt 最後一個 block 留狀態；在那之前分叉時 CPU 層有注意力 block 但沒有狀態 → offload 命中 0；Qwen3.6-35B-A3B、576-token block、vLLM 0.30.0；修法 PR #63566 未合併 | PARTIAL | `gh issue view`：2026-10-25T09:37Z、OPEN、內容相符（`scheduler.py:1696`、`:950`、六份約 16K token 文件、0/3 命中）。差異：環境欄寫「vLLM 0.30.0, and main as of 2026-10-23」，**表中數字是在 main 上量的**；硬體是 4× Intel Arc Pro B70、TP 4（LB 沒寫）。PR #63566 確為 OPEN | 〔文件〕 |
| 37 | LB | vLLM：NemotronH 的 ssm 狀態預設 FP32（「Only float32 is known to have no accuracy issues」） | OK | `vllm/model_executor/models/config.py` @`115b59a2`：`class NemotronHForCausalLMConfig`，`DEFAULT_MAMBA_SSM_CACHE_DTYPE = "float32"`（約 L812–815） | 〔程式碼〕 |
| 38 | LB | vLLM：KDA 在 `auto` 時 recurrent state 為 FP32 | OK | `mamba_utils.py` @`115b59a2` L133–149 `kda_state_dtype`；GDN 的 `auto` 跟 conv dtype（L98–110） | 〔程式碼〕 |
| 39 | LB／LD | SGLang 有 host 層 `MambaPoolHost` | OK | `python/sglang/srt/mem_cache/pool_host/mamba.py` L75 `class MambaPoolHost(HostKVCache)`（main，`gh api`） | 〔程式碼〕 |
| 40 | LB | Marconi（MLSys 2025）：逐出＝新近度＋α·FLOP efficiency（省下 FLOPs ÷ bytes）；單一容量層，無頻寬項、無 CPU／SSD | OK | 2411.19379 PDF p.6–7（FLOP efficiency、α）；全文 grep「bandwidth／SSD／offload／tier」0 筆；「CPU」只出現在背景與機器規格 | 〔全文〕 |
| 41 | LE | PyTorch `torch.backends.cuda.matmul.allow_tf32` 自 1.12 起預設 False | OK | docs.pytorch.org/docs/2.14/notes/cuda.html：「defaults to True in PyTorch 1.7 to 1.11, and False in PyTorch 1.12 and later」；A100 例子 0.016 s／0.11 s、0.0022／0.000039 亦相符 | 〔文件〕 |
| 42 | LE | repo `code/m16_xstack.py` 用 `torch.svd_lowrank`、niter＝4、多取 16 | OK | L104 `XL_NITER, XL_OVERSAMPLE, XL_SEED = 4, 16, 20261020`；L1539、L1621 `svd_lowrank(Z, q=min(r + XL_OVERSAMPLE, …), niter=XL_NITER)` | 〔程式碼〕 |
| 43 | LE | repo `code/` 沒有任何地方開 TF32 | OK | `grep -rn -i "tf32\|set_float32_matmul_precision\|allow_tf32" code/` → 0 筆（含新 API `fp32_precision="tf32"` 的寫法也會被 `tf32` 命中） | 〔程式碼〕 |
| 44 | LE | Halko–Martinsson–Tropp：q＝1 或 2 通常夠（p.9） | OK | 0909.4061 PDF p.9（「q = 1 or q = 2 usually suffices in practice」） | 〔全文〕 |
| 45 | LE | Multipole Attention（NeurIPS 2025）：A6000 batch 1／4／16 為 2.5／4.1／4.5×；A100 1.3／2.1／2.8× | OK | 2506.13059 Fig. 5（PDF p.9）、Fig. 6＋App. E（p.16）；這組數字是 **95% 稀疏**那條（90% 為 2.4／3.3／3.7 與 1.2／1.9／2.4）。venue：neurips.cc 2025 poster 頁（WebSearch 結果）〔二手〕 | 〔全文〕＋〔二手〕 |
| 46 | LE | RetrievalAttention：Llama-3-8B、100K token，top-1000（1%）平均回收 89%；固定首步則 71% | OK | 2409.10516 PDF p.3–4、Fig. 2 圖說 | 〔全文〕 |
| 47 | LF | uops.info Zen 4 `VDPBF16PS zmm` 吞吐 1／週期 → 64 BF16 FLOP／週期／核 | OK | uops.info 頁：Zen 4 Measured 1.00、延遲 6、Documentation 吞吐 1.00；Zen 5 0.50；Emerald Rapids 2.00。算術：16 個 FP32 通道 × 2 乘 2 加＝64 | 〔全文〕＋〔算術〕 |
| 48 | LF | Intel Opt. Manual 248966-051 Table 20-2：`TDP/*` 吞吐 16、延遲 52；→ 1,024 BF16 FLOP／週期／核 | OK | cdrdv2 getContent/671488（文件號 248966-051）PDF p.673（文件頁 20-3）：TDP/* 16／52、TILELOADD 8、LDTILECFG 204。算術 16·16·32·2／16＝1,024 | 〔文件〕＋〔算術〕 |
| 49 | LF | KTransformers（SOSP'25）：自寫 AMX kernel 單 socket 21.3 TFLOPS（Xeon 8452Y）；PyTorch AMX 5.4（約理論 73.7 的 7%） | OK | SOSP25-chen.pdf p.6 §3.2（21.3、3.98×）、p.4 §2（5.4／1.8、73.7、8452Y 36 核） | 〔全文〕 |
| 50 | LF | Fujitsu RX1440 M2 報告：1×EPYC 9554P STREAM Triad 390 GB/s（12×32 GB 2Rx8 DDR5-4800、NPS4、SMT off、RHEL 9.0） | OK | 報告 v1.9 PDF p.20 表（9554P 390；9124 266）、p.19 環境 | 〔全文〕 |
| 51 | LF | FA3（H100）：FP16 740 TFLOPS（75%）、FP8 約 1.2 PFLOPS、FP8 誤差低 2.6×（L44、L140） | OK | 2407.08608 摘要 PDF p.1 | 〔全文〕 |
| 52 | LF | L20：「FA3…FP8 1.2 PFLOPS 對 **BF16** 740 TFLOPS」 | **WRONG（已改）** | 原文是 **FP16** 740 TFLOPs/s；全文 pdftotext grep「BF16／bfloat16」0 筆。同檔 L44、L140 本來就寫 FP16 | 〔全文〕 |
| 53 | LF | HieraSparse（2604.16864）：prefill 只壓 K LongBench 掉 11.1、只壓 V 掉 1.8；A100 1.45／1.74× | OK | PDF p.8（11.1、1.8）、p.10–11（1.45、1.74）；abs v2 2026-08-17 | 〔全文〕 |
| 54 | LE | 〔算術〕Llama 32K k29v44：隨機 SVD q＝4 FP32 下界 K＋V 0.89 s；Gram FP32 0.26、TF32 0.036、BF16 0.018 s；少 3.4 倍；實測÷下界 5.9／6.0 | OK | 重算：T＝32768、n＝4096、8 窗口、ℓ＝944／1424；rSVD＝(2q+2)·2Tnℓ＋(q+1)·2Tℓ²＋q·2nℓ²＋4nℓ² → K 0.3427 s＋V 0.5477 s＝0.890 s（÷67e12）；Gram 2·8·2Tn²＝0.2626（FP32）／0.0356（TF32 494.5）／0.0178（BF16 989.5）；比值 3.39；2.02／0.3427＝5.89、3.29／0.5477＝6.01 | 〔算術〕 |
| 55 | LB | 〔算術〕Nemotron-H-8B 每序列狀態 97.4 MiB（ssm FP32） | OK | HF config：24 個 M 層、mamba_num_heads 128、head_dim 64、ssm_state 128、n_groups 8、conv 4 → 24·(128·64·128·4 B＋(8192＋2·8·128)·3·2 B)＝97.41 MiB | 〔程式碼〕＋〔算術〕 |
| 56 | LF | 〔算術〕16×64×3.1 GHz＝3.17；64 核 12.70；3.75 GHz 3.84；AMX 實測換算 21.3／36×16＝9.47；36×1024×2.0 GHz＝73.7 TFLOPS | OK | 重算：3.174／12.698／3.840／9.467／73.73 | 〔算術〕 |
| 57 | LE | 〔算術〕√ln n：131,072 時 ≈3.43、262,144 時 ≈3.53 | OK | 重算 3.4327／3.5322 | 〔算術〕 |
| 58 | LE | Qwen-14B 48 層（LE L119 標〔未查證：config〕） | OK | HF `Qwen/Qwen2.5-14B-Instruct(-1M)` config `num_hidden_layers` 48；原標籤可升為〔程式碼〕 | 〔程式碼〕 |

---

## 2. 統計

| 判定 | 條數 |
|:--|--:|
| OK | 53 |
| WRONG（已就地修正，並附〔複核 2026-10-27〕註記） | 2（#13 LA L290；#52 LF L20） |
| PARTIAL（沒改檔，記在這裡） | 3（#5 HyperAttention 設定歸屬；#11 Kimi decode 6.3× 的 batch 前提；#36 vLLM #63565 的版本與硬體） |
| CANNOT_VERIFY | 0 |

---

## 3. scoop 風險（LD）的複核結論

- 七篇 arXiv 全部存在，標題、作者、日期、核心方法和 LD 卡片一致。沒有任何一篇的描述需要實質修改。
- **BellowsKV 沒有做 CPU／SSD 的卸載或還原**：全文 37 頁搜尋 offload、CPU、SSD、PCIe、restore、swap、disk 都是 0 筆；評測 Protocol B 是 4K–32K（p.6），速度只量了 H100 上 32K 的 decode（p.9）。它是 GPU 常駐的 KV 壓縮格式，不是還原系統。
- RISK-KV 確實只做到 ≤1.5B、512 token；OrbitWeaver 確實是 Jetson 加上 ≤8B 的 VLM，而且沒有和 Cake 比。LD 的「沒有一篇同時具備暫停還原、線性代數、Cake 式雙向、128K 以上 GQA」這個判讀，**在這七篇的範圍內**站得住。這裡只核了這七篇，沒有另外做新的文獻搜尋。

---

## 4. 沒有核到的（範圍外，或委託清單沒列）

- LE 引用的 RoFormer 式 (16)（p.5）、§3.4.3；ALiBi／T5 bias 的 Toeplitz 判讀（LE 自己也標了未讀）。
- LF：`VFMADD231PS zmm` 的 uops.info 數字、LIA（SPR 20／GNR 40 TFLOPS）、AMD 新聞稿、Chips and Cheese 的 62.5 GB/s、CDNA3 白皮書、H200 規格表的數字。
- LD：Kimi K3 技術報告、RACE（2610.28385）、EncBank，以及只有摘要的那些表列論文（KVStream、Range、SARIA 等），還有 vLLM 的 OffloadingConnector `MambaSpec` 行號、LMCache `HiddenStateStore`、SGLang PR #40551。
- LB：SGLang int8 checkpoint pool 的行號、hicache_storage 的 `TRAILING_PAGES`、Qwen3-Next 等其他模型的狀態大小（只核了 Nemotron-H-8B 一列）。
- LC：XQuant、CacheBlend、KVLink、RotateKV、TransMLA、MHA2MLA、HGCA 圖 6 本身的 PNG（只核了內文）。
- 委託者另外要的「Multipole NeurIPS 2025」只核到 neurips.cc poster 頁的搜尋結果〔二手〕；arXiv Comments 欄只寫「15 pages」。

## 5. 來源（查證 2026-10-27）

arXiv：2405.21060、2111.00396、2310.05869、2510.26692、2302.13214、2505.11892、2410.05004、2507.03153、2606.01751、2510.05373、2610.35023、2610.29544、2610.29183、2602.16284、2608.30320、2604.12374、2606.13392、2411.19379、0909.4061、2506.13059、2409.10516、2407.08608、2604.16864（皆 https://arxiv.org/pdf/<id>，abs 頁取 meta）。
其他：https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model ；https://github.com/vllm-project/vllm/releases/tag/v0.32.0 ；https://github.com/vllm-project/vllm/issues/63565 ；vLLM raw 檔 @`5df63a7203cd`、@`115b59a2da86`；SGLang `pool_host/mamba.py`；https://docs.pytorch.org/docs/2.14/notes/cuda.html ；https://uops.info/html-instr/VDPBF16PS_ZMM_ZMM_ZMM.html ；https://cdrdv2.intel.com/v1/dl/getContent/671488 ；KTransformers SOSP25-chen.pdf（madsys.cs.tsinghua.edu.cn）；Fujitsu wp-performance-report-primergy-rx1440-m2-ww-en.pdf；HF config.json（Qwen2.5-7B／14B-Instruct(-1M)、unsloth/Meta-Llama-3.1-8B-Instruct、nvidia/Nemotron-H-8B-Base-8K）；https://neurips.cc/virtual/2025/poster/119872（搜尋結果）。

---

## 第二輪複核（LG、LH）

> **日期**：2026-10-27。**性質**：zero-context verifier（新開的 Claude subagent），**self-family review，不是 cross-model**。這台 Mac 只做下載、`pdftotext -layout`、`python3 -I` 算術；沒有跑任何實驗或模擬。下載與腳本在 `<SCRATCH>/verify2/`（`dl_*` 是下載、`*.py` 是我寫的算術）。G-M11 盲測：沒有打開 `results/s3_h200/b0/gm11/`，也沒有算 VQ 對 INT4 的比。

**一句話**：LG 的推導與所有算術都重算得出來；arith §9 確實沒算縮圖重建。LH 的 repo 數字幾乎都對得上原始 CSV／JSON，但「背景 decode 吞吐掉到 0.26–0.59」其實是**變慢的比例**，吞吐比應是 0.41–0.74；CommVQ 的成本是 (R＋1)/2 倍（約 6 倍），不是 R＋1 倍。

### 1. 表

| # | 檔 | 主張 | 判定 | 出處與核對方式 | 深度 |
|:--|:--|:--|:--|:--|:--|
| 1 | LG §2.1 | V 吸收：latent 比較便宜 ⇔ n_q·g·(r−d) < r·d，n_q\*＝r·d／(g(r−d)) | OK | 自己重推：每層重建 2·n·r·H·d（A[n×r] 乘 B_ℓ[r×H·d]，A 跨 W 層共用但 B_ℓ 每層不同）；PV 每 q head 2·n_q·n·d；latent 每 q head 2·n_q·n·r（＋2·n_q·r·d 可略）。H·g 個 q head：H·g·n_q·n·r < n·r·H·d＋H·g·n_q·n·d ⇔ 上式 | 〔推導〕 |
| 2 | LG §3.1 | K 用 RoPE 可交換基底時同一個交叉點 | OK | 複數係數 r/2 個、Re(w·c) 每個複數 2 個實數乘 → 每 (q, key) r 個實數乘加＝2r FLOP，對照 2d；相位乘在係數上每窗每 key 一次，O(n·r) 可略 → 同式 | 〔推導〕 |
| 3 | LG §1、§9 | B24 聯合秩 K 544、V 832；r＝每 head 每層等效秩 × W × H | OK | `A0_progressive_restore.md:44–53`（B24＝`xl:k17:v26`，倍數 25.13，f＝0.0398；L44「每 head 每層等效秩」）；`code/m16_xstack.py:18`「秩＝r×W×H」、L507、L870、L905 `r = rhe * W * H` → Llama 17·4·8＝544、26·4·8＝832；Qwen-7B（H＝4）272／416 | 〔程式碼〕 |
| 4 | LG §1 | g＝4／7／5；H＝8／4／8 | OK | HF config：unsloth/Meta-Llama-3.1-8B-Instruct 32/8；Qwen/Qwen2.5-7B-Instruct-1M 28/4；Qwen/Qwen2.5-14B-Instruct-1M 40/8（另見 `code/s3_mech_arith.py:37–43`） | 〔文件〕 |
| 5 | LG §9.2 | n_q\*：Llama 38(V)/42(K)、Qwen-7B 26/35、Qwen-14B 30/34，及 B16／B32 列 | OK | 重算 9 列全部一致（例 Llama B24：K 41.8、V 37.8；Qwen-7B B24 34.5／26.4；Qwen-14B B24 33.5／30.3） | 〔算術〕 |
| 6 | LG §9.3 | Llama 128K B24：重建 11.82 TFLOP；n_q＝128 重建＋注意力 20.62 vs 全 latent 47.28 vs 只 V latent 37.66；decode 11.89 vs 0.37（32×）；其他 5 列 | OK | 重算 18 個格子全部一致到小數第二位（公式照 §9.1，全層、全 context） | 〔算術〕 |
| 7 | LG §0、§9.3 | 重建 15.0 ms（788.1 TFLOPS）／30.0 ms（50%） | OK | 11.82e12÷788.1e12＝15.0 ms；÷394.05e12＝30.0 ms；788.1 出自 `results/RUNLOG_H200.md:63` | 〔算術〕 |
| 8 | LG §0、§10 | arith §9 的 TTFT 沒算縮圖重建（「不是錯，是漏項」） | OK | `arith.md` §9 標頭文字：「prefill 時間用第 6 節的公式；**反量化不計**」。§9 實際包含：(1) 問題 prefill＝注意力 4·n_q·n·(q heads·128)·L＋線性 2·參數·n_q，以 50%×788.1 TFLOPS 計（`code/s3_mech_arith.py:222–224` `prefill_s`）；(2) 傳輸：全部載入＝BF16 KV÷層頻寬；兩趟／逐層的精修＝KV÷3.2÷4（ρ＝1/4 INT4）÷層頻寬；逐層流水＝max(prefill, 精修)＋精修÷L（L227–245）。**沒有任何低秩重建（係數×基底）的項**，也沒有 RoPE 重套。Llama CPU n_q＝128 逐層 0.028 s、Qwen-14B 256K 0.095 s 都和 LG 引的一致 | 〔程式碼〕〔文件〕 |
| 9 | LG §3.2 | 和所有 RoPE 旋轉可交換 ⇔ 基底按頻率分塊、塊內複線性 | OK（補一個條件） | 〔推導〕R_n＝Σ_j (e^{inθ_j}Π_j⁺＋e^{−inθ_j}Π_j⁻)。S 對所有 R_n 不變 ⇔ S_ℂ 對 span_ℂ{R_n} 不變；若 128 個值 {e^{±iθ_j}} 兩兩不同，Vandermonde 讓 span 含每個 Π_j^±，於是含 I_j＝Π⁺＋Π⁻ 與 J_j＝i(Π⁺−Π⁻) → S＝⊕(S∩V_j) 且各塊 J_j 不變（複線性）；反向顯然。**精確條件比 LG 寫的「θ_j 互不相同」多一點**：還要 θ_j≢−θ_k、θ_j≢0、π（mod 2π）；θ_j≡0 的塊可以是任意實子空間 | 〔推導〕 |
| 10 | LG §3.2、§10 | Llama-3.1 llama3 scaling 後 θ_j 兩兩相異（LG 原標〔未查證〕） | OK（已查證，就地加註） | transformers `src/transformers/modeling_rope_utils.py` `_compute_llama3_parameters`（main 9ed172cd91，L655–660：低頻 ÷factor、中頻 (1−s)·f/factor＋s·f 平滑、高頻不動）＋config（500000、8、1、4、8192）代入：64 個 θ_j 嚴格遞減，相鄰比 ≥1.2276，範圍 (3.07e-7, 1] rad，fp32 下也 64 個相異 → 第 9 項的條件全部成立 | 〔程式碼〕〔算術〕 |
| 11 | LG §3.3 | TransMLA RoRoPE：同頻率跨 head、實虛部同一個 U_l（§4.2、式 (19) p7）；圖 3b 90% 移除時 log-PPL 約 2 vs MHA2MLA 約 6（p9） | OK | arXiv 2502.07864 PDF：§4.2 標題與式 (19) 在 p7；「same orthogonal matrix U_l must be applied to both the real … and imaginary」p7；p9 原句 log-perplexity about 2、MHA2MLA nearly 6 | 〔部分〕 |
| 12 | LG §0-6、§3.3、§6.1、§10 | CommVQ 成本降到「約 (R＋1) 倍」普通注意力，R＝11 → 約 12 倍 | **WRONG（已修正 5 處）** | arXiv 2506.18879 p6：原句 now approximately … times higher；pdftotext 印成「R+12」，arXiv HTML 的 TeX 是 `\frac{R+1}{2}` → **(R＋1)/2 倍，R＝11 時約 6 倍**。R＝11 是 1-bit 設定（p6、p15；2-bit 是 R＝21）。LG 的結論「不比普通注意力便宜」不變 | 〔部分〕 |
| 13 | LG §2.3、§3.4 | JoLT p9：把旋轉摺進因子打分是 kernel 端的主要未解問題 | OK | arXiv 2607.12550 v4 PDF p9：「…is the primary open problem on the kernel side」 | 〔部分〕 |
| 14 | LG §3.3 | PseudoHunt/rope-kv-compression：建立 2026-09-23、Llama-2-7B-chat（MHA）、25% 預算 88.9% vs 97.7%、輸出誤差 2.69×、停在 Gate 3 | PARTIAL（數字全對；一個欄名不精確） | `gh api`：created_at 2026-09-23T14:58:21Z，HEAD 9d7200c（15:41:37Z）；README L1、L7；`docs/FINDINGS.md` L3–4、L13、L20–21、L28–31、L79。研究 2 的 1.10×、+0.77%／+2.49% 也和 README L8 一致。不精確處：LG 表把 87.4 那欄標成「逐頻率實數（RoRoPE 基底）」，原表是「(b) real, global」；照 RoRoPE 原本分配的是 (b′) 85.6（FINDINGS L28–31）。沒改檔 | 〔文件〕 |
| 15 | LG §0-4、§4.3 | 「純量反量化 0.031 ms／64 MiB」「精修塊 INT4-KIVI，F3 量到 0.027–0.031 ms」 | PARTIAL（就地加註） | `F3_dequant_fp8.md:56`、`:69` 數字對；但 F3 是 **MI300X**（§4.1 標題 L125），INT4 是**對稱 per-(token, head)**、不是 KIVI（L113）。H200 上 int4_kivi 的融合 kernel 是 0.049–0.073 ms（k1、k3:8:2，`kernels_llama.csv`）→ LG「31 塊約 1 ms」在 H200 約 1.5–2.3 ms，量級不變 | 〔文件〕〔算術〕 |
| 16 | LH §0、§1、§4.2 | F3：INT8 0.031、FP8 0.026、INT4 0.027 ms／64 MiB（cold），3.3 TB/s，峰值 62%，**MI300X** | OK | `F3_dequant_fp8.md` §4.1（L125–143）：torch.compile 0.0309、FP8（整個 tensor 一個 scale）torch.compile 0.0261、INT4 Triton 0.0274；run `20261010-074630-f3-dequant`；L46、L115 Infinity Cache 註明 MI300X。注意 FP8 0.026 是 per-tensor scale 的格式 | 〔文件〕 |
| 17 | LH §1 表 | H200 融合反量化：k2 閒 0.060–0.066、忙 0.028–0.030；k1 閒 0.073–0.087、忙 0.049–0.078 | **WRONG（小，已修正）** | `results/m10_sweep_h200/dqsweep/A/kernels_llama.csv`（run `20261018-145622-m10h-dqs-A-llama`，CSV 有 run_id／ts，contaminated_gpu 全 False）：k2 的 int4_kivi 是閒 0.114、忙 0.085，原範圍沒有註明排除；k1 閒的下界是 0.0705（fp8_head）。LH 其他處的「H200 每塊 0.028–0.087 ms」以「每個格式最快的 kernel」來讀是對的（int4_kivi 的最佳 k3:8:2 閒 0.067、忙 0.049） | 〔程式碼〕 |
| 18 | LH §0、§3.2 | INT4 一塊過 PCIe 0.365 ms（BF16 1.167 ms） | OK | 64 MiB÷3.2＝20 MiB；20÷(53.55×1024) s＝0.3647 ms；53.55 GiB/s 出自 `results/RUNLOG_H200.md:63`；3.2 出自 `code/s3_mech_arith.py:50` | 〔算術〕 |
| 19 | LH §0、§3.3、§3.4 | MI300X X3（duty 0.9）：lr_r16 5.30×、lr_r64 4.82×、xKV f4 3.10×、xKV f8 1.79×、INT4 一次拆 1.44×；xKV 的 H2D 3.2× vs INT4 堆疊 copy 1.09×；Qwen lr_r16 5.2×、xKV f8 2.4×、INT4 2.6× | OK | `results/m16_stage3_mi300x/x3_time/micro_llama.csv`（run `20261020-212244-m16-x3-micro-llama`）、`micro_qwen.csv`；op＝dq_oneshot：int4_kivi 1.69→2.43、PCA r16 2.34→12.41、PCA r64 2.48→11.96、xKV r512 3.96→7.08、r1024 5.91→18.33；h2d_region xKV r512 7.30→23.71（3.25×）、h2d_stack int4_kivi 13.68→14.92（1.09×）；Qwen PCA r16 1.02→5.26、xKV r256 1.38→3.35、int4 0.59→1.55。全部 contaminated＝False。背景是 `m7_busy` matmul（`x3_time_howto.md:74`） | 〔程式碼〕 |
| 20 | LH §0、§1、§4.4 | H200 B0：cprefill 背景下 mps87 的 INT4 全載入比 stream 快 4.28×，背景吞吐 0.98 | OK | `results/s3_h200/b0/b0_report.json` `share`：llama cprefill:2048:65536 × e2e_load_int4_b32 × mps87，ratio_ci [4.26, 4.29]、bg_ratio 0.980（JSON 的 `speedup` 3.28＝ratio−1）；`B0_interference.md:140` 同 | 〔程式碼〕 |
| 21 | LH §1、§3.3 | stream priority 沒效：lr_recon stream 0.628 vs stream_hi 0.637 | OK | `b0_report.json` `m08`：llama decode:4:129024 lr_recon speed_vs_idle 0.6281（stream）／0.6367（stream_hi） | 〔程式碼〕 |
| 22 | LH §3.3 | H200 長 context decode 背景下 lr_recon 只剩閒時的 0.62–0.73（Llama）、0.39–0.53（Qwen） | OK | `m08` lr_recon：llama 0.621–0.731（5 格）；qwen 0.394–0.531（10 格，decode:4／6:258048）；speed＝t_閒(stream)÷t（`code/s3/b0.py:804`） | 〔程式碼〕 |
| 23 | LH §3.3 | 背景 decode 吞吐同時掉到 0.26–0.59 | **WRONG（已修正）** | `b0_report.json` 的 `bg_slowdown`＝1−吞吐比（`code/s3/b0.py:807` `slow = 1 - v["bg_ratio"]`）。lr_recon 格的 bg_slowdown 是 0.259–0.587 → **吞吐比 0.41–0.74**（慢 26–59%） | 〔程式碼〕 |
| 24 | LH §3.3 | 〔判讀〕MI300X「頻寬型背景幾乎不受影響」和 H200 長 context decode 的結果不一致 → 「只有算力型背景才翻轉」要限定在「copy 迴圈當頻寬型背景」 | PARTIAL | 數字面成立：H200 真的 decode 背景（4×129K、4–6×258K，fa3＋CUDA graph，`B0_interference.md:18–26`）把 lr_recon 拖到 0.39–0.73，和同背景下的 qprefill（0.65–0.76）、recompute1（0.54–0.68）同級；MI300X copy 迴圈背景（約 3.9 TB/s，duty 0.9）只讓端到端慢 5–30%（`results_analysis.md:280`）。**但不是乾淨的對照**：不同 GPU；一邊是基本動作速度、一邊是端到端翻轉判定；m08 只量計算型動作（`COMPUTE_PRIMS`），**沒有量 INT4 路徑**，所以 H200 decode 背景下有沒有翻轉是 NOT_MEASURED；B0 的 GB0-FLIP 不成立（`B0_interference.md:139`）也不涵蓋 lr_recon（它不在生產格式子集，L43）。另外 LH 引的是 L14，「幾乎不影響（像小批次 decode）」那句其實在 `results_analysis.md:282`，而且「小批次 decode」是推論、沒量過 | 〔程式碼〕〔判讀〕 |
| 25 | LH §1、H6 | PTX ISA 9.4：`wgmma` 沒有 s4／u4 | OK（小遺漏） | docs.nvidia.com PTX ISA 頁面標題「PTX ISA 9.4」；`wgmma.mma_async` 語法：f16、bf16、tf32、FP8 `.atype={.e4m3,.e5m2}`、整數 `.atype={.s8,.u8}`，另有 **單位元 `.b1`（`.op.popc`，`.and`）**；沒有 `.s4/.u4`（`.s4/.u4` 只出現在 `mma`／`wmma`）。LH 的型別清單漏了 `.b1`，不影響結論 | 〔文件〕 |
| 26 | LH §1、H6、H7 | SageAttention2 說 Hopper 沒有原生 INT4 tensor core | OK | arXiv 2411.10958 PDF p2、p8、p23 三處（LH 寫 p.2、p.23） | 〔部分〕 |
| 27 | LH §1、H7 | 對 FA2：H100 FA3-fp8 2.63、SageAttention2 2.61；H20 3.06、3.12 | OK | 同 PDF p15 Table 9（另 FA3 1.37／1.57）；p23 說 Hopper 上用 SageAttention2-8b | 〔部分〕 |
| 28 | LH §1、H8 | vLLM：128K 撈針 BF16 91% → FP8 13%，兩層累加回 89%；單張 H100 | OK | https://vllm.ai/blog/2026-04-22-fp8-kvcache 原始 HTML 文字：dropped from 91% … to just 13%；brought … back to 89%；All Hopper experiments run on a single H100；7,010 token／54%（v0.19.1）也在表中 | 〔文件〕 |
| 29 | LH §4.5 | CPU 端先解：DRAM 148–212 MiB，在 98–125 GB/s 下 ≥1.24 ms；GPU 路徑約 0.39–0.44 ms | OK | 20＋64＋64（＋64 RFO）MiB；148 MiB÷125 GB/s＝1.242 ms、212 MiB÷98 GB/s＝2.268 ms；PCIe 64 MiB＝1.167 ms < 1.24 → DRAM 是下界；GPU：0.365＋0.028–0.073。98–125 GB/s 出自 `LF_hw_isa.md:94`（算術＋判讀，非實測） | 〔算術〕 |
| 30 | LH §4.5 | 損益點：GPU 反量化 ≥ (1−1/3.2)×1.167＝0.80 ms／塊；重疊時 ≥1.167 ms | OK（偏向 CPU 的界） | 重算 0.8024 ms。這個界只比 PCIe、沒算 CPU 端的 DRAM 成本；若用第 29 項的 1.24 ms，損益點是 1.24−0.365≈0.88 ms，CPU 更難贏，結論不變 | 〔算術〕 |
| 31 | LH §0-5 | 「≥0.80 ms 是 H200 實測值的 12–16 倍」 | **WRONG（小，已修正）** | 和 LH §4.5 自己的 0.80÷(0.028–0.087)＝9.2–28.6 倍不一致，改為 9–29 倍 | 〔算術〕 |

### 2. 統計

| 判定 | 條數 |
|:--|--:|
| OK | 24（#1–11、13、16、18–22、25–30；#9、#25、#30 附小註） |
| WRONG（已就地修正，附〔複核 2026-10-27〕） | 4（#12 LG CommVQ (R＋1)→(R＋1)/2，5 處；#17 LH k2／k1 範圍；#23 LH 背景吞吐比；#31 LH 12–16 倍） |
| PARTIAL | 3（#14 PseudoHunt 欄名，沒改檔；#15 LG 的 F3 平台與格式，已加註；#24 LH 的判讀，沒改檔） |
| CANNOT_VERIFY | 0 |

另外就地加註（不是 WRONG）：LG §3.2 與 §10 的「θ_j 是否兩兩相異」由〔未查證〕改成已查證（#10）。

### 3. 對第三階段結論的影響

- **主張 2（arith §9 漏了縮圖重建）成立。** §9 只算問題 prefill 的 FLOPs 和精修的傳輸，明寫「反量化不計」，全檔沒有低秩重建的項。用同一套口徑（50% 的 788.1 TFLOPS），Llama 128K B24 的重建 30 ms，和逐層流水的 TTFT 28 ms 同量級；Qwen-14B 256K 是 90 ms 對 95 ms。所以 arith §9／§10「粗底＋逐層比 Cake 快約 10 倍」的 CPU 層數字，**在縮圖要先重建成 BF16 的做法下被高估**（粗估約 2 倍，〔判讀〕：假設重建和 prefill 搶同一批 SM、不能重疊）。慢層（NVMe、NFS）由傳輸主導，影響小。
- **主張 5（LH 的 repo 數字）大致成立，一處要改讀**：
  - X3、B0、G-M08 的速度數字都對得上原始 CSV／JSON（#16–22）。
  - 「背景吞吐 0.26–0.59」是**變慢的比例**；吞吐比是 0.41–0.74（#23）。G-M08 的 STOP 結論不受影響（門檻 0.95）。
  - LH 要把「只有算力型背景才翻轉」縮小範圍，這個方向**有根據但證據不完整**（#24）：H200 真的 decode 背景確實讓 lr_recon 慢 1.4–2.5 倍，不是「幾乎不影響」；但 H200 上沒有在同樣背景下量 INT4 路徑，所以**不能說翻轉會發生，只能說「頻寬型背景下不翻轉」目前只在 MI300X 的 copy 迴圈上量過**。`results_analysis.md:282` 的「（像小批次 decode）幾乎不影響」是外推，應該加限定。
- **順帶發現（範圍外，沒改檔）**：`docs/stage3_design_20261024/runs/B0_interference.md:142` 寫 G-M08「背景吞吐比 0.33–0.65」，但 `b0_report.json` 的 45 格算出來的吞吐比（1−bg_slowdown）是 0.354–0.741，變慢比例是 0.259–0.646；兩組都對不上 0.33–0.65。STOP 結論不變，但那句的數字要由 H200 session 重核。

### 4. 沒有核到的

- LG：Palu、xKV、ShadowKV、SALS、Loki、QServe、HACK、SageAttention1／3、KIVI、PQCache、MILLION、VQ-LLM、DeepSeek-V2、EliteKV、RAP、LoRC、Eigen Attention 的頁碼與數字；PseudoHunt 的 `PRIOR_ART.md`、`FINDINGS_KERNEL.md`；§7.1 的 HBM 8.2 ms 與融合 tile 倍數；§9.4 只核了 f＝0.0398 與 171 FLOP/B，其他兩列沒核。
- LH：Kim 2022、Marlin、Machete、BitDecoding、KIVI、FLUTE、LUT-GEMM、Ladder、CDNA3 白皮書、Green Context、HIP CU mask、uops.info、TRT-LLM `interleaved_numeric_conversion.h` 行號、§3.3 由 X3 換算的 MI300X GEMM 效率（278／372／174 TFLOPS）、H200「閒時比忙時慢」的原因、`kernels_qwen.csv`。
- B0 的 raw CSV（`results/s3_h200/b0/raw/`）不在這個 worktree，只核了 `b0_report.json` 與 `code/s3/b0.py`。

### 5. 來源（查證 2026-10-27）

arXiv PDF：2502.07864、2506.18879（另 HTML v1）、2607.12550、2411.10958。其他：https://docs.nvidia.com/cuda/parallel-thread-execution/index.html （PTX ISA 9.4）；https://vllm.ai/blog/2026-04-22-fp8-kvcache ；https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/modeling_rope_utils.py （main 9ed172cd91）；HF config.json（unsloth/Meta-Llama-3.1-8B-Instruct、Qwen/Qwen2.5-7B-Instruct-1M、Qwen/Qwen2.5-14B-Instruct-1M）；GitHub PseudoHunt/rope-kv-compression（`gh api`，README、docs/FINDINGS.md，commit 9d7200c）。repo：`code/m16_xstack.py`、`code/s3_mech_arith.py`、`code/s3/b0.py`、`docs/stage3_design_20261024/mechanisms/arith.md`、`runs/A0_progressive_restore.md`、`runs/B0_interference.md`、`docs/research_20261010_followup/F3_dequant_fp8.md`、`results/m10_sweep_h200/dqsweep/A/kernels_llama.csv`、`results/m16_stage3_mi300x/x3_time/micro_{llama,qwen}.csv`、`results/s3_h200/b0/b0_report.json`、`results/RUNLOG_H200.md:63`、`docs/stage3_20261019/results_analysis.md`、`x3_time_howto.md`。
