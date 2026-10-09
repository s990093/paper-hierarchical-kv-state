# A_Krul Krul: Efficient State Restoration for Multi-turn Conversations with Dynamic Cross-layer KV Sharing

- **出處**：Junyi Wen, Junyuan Liang, Zicong Hong, Wuhui Chen, Ting Cai, Zibin Zheng（SYSU、HKUST、Peng Cheng Lab、湖北工大）。arXiv 2507.08045v2（v1 2025-07-10）。PDF 用 PVLDB 樣板，但 reference format 是佔位字（「PVLDB 14(1): XXX-XXX, 2020」）〔原文 p.1〕，所以 **venue 未查證**。<https://arxiv.org/abs/2507.08045>。**引用了 Cake**（Semantic Scholar 的 citations 清單；原文 p.12 也提到 Cake）。讀了 p.1–2、p.8–10、p.12（PDF 實體頁）。
- **寫入時做了什麼決定**：兩件事都在「壓縮時」（對話一輪結束、KV 存到 CPU 之前）決定：
  1. **每層存多少、哪一段不存**：先依硬體算出最佳重算比例 r_c（讓重算時間 T_C 和載入時間 T_L 最接近），再配置儲存空間 C＝{c×(1−r_c)}×N；被分配到「重算」的那部分**不存**〔原文 p.8 §6.1〕。重算與載入的量沿層呈金字塔：載入量隨層數線性增加，重算量線性減少，「這個預先規劃的分布在壓縮時就定好」〔原文 p.9 §6.2〕。還原時每層的重算用 `tensor[: target_length]` 切前段 hidden state，其餘從 CPU 載入，最後 `torch.cat()` 接起來〔原文 p.9 §7〕。
  2. **存什麼格式（有損）**：依每段對話的注意力相似度，選擇哪些相鄰層共享 KV（跨層壓縮，方法沿用 MiniCache）〔原文 p.2、p.8 註腳〕。
- **用什麼資訊做決定？寫完之後還在不在？（N1）**
  - 第 1 件：r_c 用校準資料離線算〔原文 p.8〕，事後仍在，不是 N1。
  - 第 2 件：用的是**生成過程中的注意力權重**——prefill 的注意力權重在計算時用另一個 CUDA stream 卸到 CPU 算相似度，decode 的在每一步算完就刪掉〔原文 p.2、p.9 §7〕。原文說直接把所有注意力權重存在 GPU 會 OOM（7B、4,000 token 要 30 GB）〔原文 p.2〕。**這是 N1**：注意力權重寫完就沒了，延後版無法取得。
- **有沒有和延後版、寫穿版、背景版比較？**
  - 對照組：全部重算、全部載入（AttentionStore 式，但不用 job queue 預取）、**partial KV recompute（引用 Cake、Pensieve 等：每層固定比例重算＋載入）**、H2O（淘汰）、MiniCache（固定跨層壓縮）〔原文 p.9–10〕。
  - 結果：準確度平均損失 <1%，TTFT 比 SOTA 快 1.28–2.68 倍〔原文 p.10〕（摘要寫 1.5–2.68 倍、儲存省 1.33–2.35 倍〔原文 p.1〕；兩處數字不一致，原文如此）。
  - **沒有拆開「寫入時決定不存前段」本身的貢獻**——和 partial recompute 的差距混了壓縮的效果〔判讀〕。沒有延後版或多層版。
- **硬體**：4×A100 80GB、128 GB DRAM、10 TB SSD、PCIe Gen4〔原文 p.9〕。KV 存在 CPU 記憶體〔原文 p.8〕。
- **模型架構、模態**：LLaMA-7B、LLaMA-30B、Qwen1.5-7B、Qwen2-72B（8-bit）〔原文 p.9〕；文字，LongBench（品質）、ShareGPT（吞吐）〔原文 p.9〕。
- **和本研究的關係**
  - **對 S5 新穎性的直接威脅（目前看到最接近的）**：Krul 在**寫入時**依「重算 vs 載入的平衡」決定**每層前段不存、讀取時重算**，而且明確把自己放在 Cake／Pensieve 這條「前段重算、後段載入」的路線上〔原文 p.8–9、p.12〕。這就是 S5 的「b 以前不值得放」在單層（CPU）、依層變化的版本。差別：Krul 是單層、有損（跨層共享）、r_c 由校準定死；S5 是 CPU／SSD 兩層、無損、依位置〔判讀〕。**相關研究必須引用並區分。**
  - **支持 N4**：原文說壓縮過的 KV 會打亂固定比例的重算／載入管線，所以要重新排程〔原文 p.2〕——寫入形式決定讀取時的最佳會合點〔判讀〕。
  - **支持 N1 的實例**：寫入時的注意力權重〔原文 p.2、p.9〕。
- **證據等級**：〔原文 p.X〕；威脅評估〔判讀〕。
