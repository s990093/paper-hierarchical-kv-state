# A_MORI Idleness is Relative: Exploiting Tool-Call Idle Windows for Offloading in Agentic Systems with MORI

- **出處**：Tian Xia, Hanchen Li, Zhifei Li, Xiaokun Chen, Hao Kang, Yifan Qiao, Yi Xu, Ion Stoica（UC Berkeley、Renmin、Stanford、Georgia Tech）。arXiv 2606.00866v1（2026-05-30），cs.OS。venue：未查證。<https://arxiv.org/abs/2606.00866>。讀了 p.1–2、p.6、p.8–10（PDF 實體頁）。
- **寫入時做了什麼決定**：**不是寫入時**。MORI 以「程式」（agent program）為單位，依「閒置程度」排名，最忙的放 GPU HBM、最閒的放 CPU DRAM，並動態移動分界以符合兩層的容量比；兩層都做准入控制〔原文 p.1 摘要〕。決定發生在**工具呼叫期間**（程式在等工具時才把 KV 卸到 CPU）〔原文 p.2〕。實作時給引擎的 block 打上程式 id，引擎插入 KV block 時依程式的層別決定淘汰順序〔原文 p.8〕。
- **用什麼資訊做決定？（N1）**：程式的閒置程度（工具呼叫的歷史型態）。觀察：agent 程式長時間停在兩種階段之一——忙碌階段是一串幾百毫秒的短工具呼叫，閒置階段是等幾秒以上的長呼叫（等人、等 sub-agent）〔原文 p.2〕。單次呼叫長度不可預測〔原文 p.2〕。資訊事後仍在，不是 N1。
- **有沒有和延後版、寫穿版、背景版比較？**
  - 對照組是 LRU、依靜態特性（例如 context 長度）選淘汰對象的程式感知排程器、以及 SGLang Model Gateway 等〔原文 p.2、p.9〕；比最好的「有卸載」基線，吞吐高 20–71%、TTFT 低 18–43%〔原文 p.1〕。
  - 這篇本身就是「**利用空閒時間做搬移**」的代表——工具呼叫的空檔就是背景版需要的空閒〔判讀〕。
- **硬體**：H200（把 HBM 限到 80 GB，模擬 H100 級容量）、H200、B200；分別配 7B、30B MoE、70B（TP=2）模型〔原文 p.9–10〕。
- **模型架構、模態**：文字 agent；**用 Claude Code 收集的真實 coding agent trace**〔原文 p.1〕。引擎 SGLang v0.5.10，卸載後端是 **SGLang HiCache**〔原文 p.8–9〕。
- **和本研究的關係**
  - **威脅 N3／H3（在 agent 負載上）**：agent 負載天生有工具呼叫的空檔，背景版（空閒時搬）有時間可用；MORI 就是靠這個贏〔原文 p.1–2；推論〔判讀〕〕。所以 H3 的「沒有空閒時間」不能用 agent 負載來論證——除非是忙碌階段（幾百毫秒的短呼叫）〔原文 p.2〕。
  - **支持 H8 的資訊來源**：「這個程式會不會很快回來」可以用閒置程度的歷史估計，不需要精準預測單次呼叫〔原文 p.2〕。這是 H8（預測 session 會不會回來）在 agent 上的現成做法，而且是延後版（等到工具呼叫時才決定）〔判讀〕。
  - 和 [A_EfficientAgent](A_EfficientAgent.md)、[A_Lachesis](A_Lachesis.md) 一起看：2026 年的 agent KV 論文分別在「什麼時候卸載」（MORI）、「要不要寫」（EfficientAgent）、「寫到哪」（Lachesis）三個點上做決定〔判讀〕。
- **證據等級**：〔原文 p.X〕；對 H3／H8 的意義〔判讀〕。
