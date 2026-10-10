# D6_KVPR KVPR: Efficient LLM Inference with I/O-Aware KV Cache Partial Recomputation

- 出處：Chaoyi Jiang, Lei Gao, Hossein Entezari Zarch, Murali Annavaram（USC），arXiv 2411.17089v2（2025-06-04）；OpenAlex 顯示發表在 Findings of ACL 2025（DOI 10.18653/v1/2025.findings-acl.997，我沒有開 ACL 版本）。https://arxiv.org/abs/2411.17089 。讀了 p.1、p.4–5、p.8–9、p.14–15（grep 定位）。

- 精度決定：
  - 主體不做精度決定，是 FP16〔原文 p.2〕。
  - §4.4 加了一個相容性實驗：KV 用 group-wise 4-bit 量化，減少傳到 GPU 的資料量，decode throughput 再提升（Fig. 9，OPT-13B，只有圖）〔原文 p.8〕。均勻量化，沒有說只量化傳輸那段。

- 有沒有和重算／載入並行結合？精度跟位置有關嗎？
  - **有並行，而且是「前段重算、後段載入」**：CPU 先傳前 l 個 token 的激活 X[0:l]，GPU 用它重算 K、V[0:l]；同時 CPU 非同步傳剩下的 K、V[l:s′]〔原文 p.4 Fig. 3、p.5 式 6–9〕。
  - 跟 Cake 的差異：這裡的重算只是從**存著的激活**做 K、V 投影（4·b·l·h² FLOPs），不是從 token 重跑 prefill，所以成本跟位置線性、和 attention 無關〔原文 p.5〕。發生在 **decode** 階段的 offload 推論，不是 prefix 還原。
  - split point l 由排程器用 profiler 的硬體資訊算出〔原文 p.5〕。
  - 精度跟位置無關（4-bit 是全部均勻）。
  - 原文提到 ALISA（Zhao et al., 2024）：依稀疏性壓縮 KV，載入時「先重算一部分、再傳剩下的」；KVPR 的差別是讓兩者重疊並自適應選 split point〔原文 p.9〕。

- 反量化／解碼成本：沒有量。

- 品質與位置的關係：沒有量。

- 有沒有和延後版比？沒有。

- 和 D6 的關係〔判讀〕：
  - 又一篇「前段重算、後段載入、加上均勻量化」的工作（跟 Cake 的 Table 6 同一個格局），所以「量化 → split 往前移」不新。
  - 它重算的原料是激活，不是 token；如果 D6 要比「前段不存」，KVPR／HCache 那種「前段存激活（一半大小）」是另一個對照組。

- 證據等級：B。
