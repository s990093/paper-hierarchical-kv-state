---
type: report
created: 2026-08-29
tags: [paper, open-issues, todo, kv-cache, mvp-scope]
related_ideas: ["[[idea-20260828-hierarchical-kv-state]]"]
---

# 已知但暫緩處理的問題

（對應單一 idea：[[idea-20260828-hierarchical-kv-state]]，論文 `main.tex`）

這份清單記錄**已經知道、刻意先不處理**的事項。目的是讓 MVP 的範圍是「明確劃定的」而非「沒想到的」。
每一項都註明：現在論文裡怎麼寫的、什麼時候該處理、不處理的風險是什麼。

---

## 🔴 A 級：投稿前必須解決

### A1. 場所選擇（截止日已查證，見 `VENUES.md`）

| 場所 | 截止 | 剩餘天數 |
|---|---|---|
| EuroSys 2027 秋輪 | **9/24/2026** ✅ | 25 天 — 不可行 |
| **MLSys 2027** | **10/30/2026** ✅ | **61 天** — 拚一把 |
| ICPE 2027（瑞典） | ~11 月 🔶 | ~80 天 |
| CCGrid 2027 | 12/8/2026 ✅ | 100 天 |
| **EuroMLSys 2027**（6 頁） | ~2 月 2027 🔶 | ~180 天 — **推薦** |
| Euro-Par 2027（荷蘭） | ~2–3 月 🔶 | ~200 天 |

**判斷點：9 月底若還沒有 Oracle 結果，放棄 MLSys 這一輪。**
完整分析見 [[VENUES]]。

---

## 🟡 B 級：審稿人會問，但不擋實驗

### B1. K 與 V 未分離處理 ← 本次新增，已寫入論文 §7

| | |
|---|---|
| **現況** | 式(1) 前導的 2 對應 K 與 V；本文**把兩者綁在同一個動作**上 |
| **論文怎麼寫** | §7 新增段落〈K 與 V 未分離處理〉，明確承認並說明為何不影響核心主張 |
| **文獻依據** | KIVI（ICML'24）：K 宜 per-channel、V 宜 per-token<br>Hariri et al. 2025（arXiv:2502.15075）：固定預算下優先給 K 位元**嚴格更優**（4-bit K + 2-bit V）<br>KVTuner（**ICML'25**, arXiv:2502.04420）：逐層混合精度，Llama-3.1-8B-Instruct 達 3.25 bit 近無損 |
| **為何暫緩** | 動作空間 $\lvert\mathcal{A}\rvert$ 會由 **6 → 36**，標籤生成與 oracle 求解成本同步上升 |
| **不處理的風險** | **低**。論文的立論是「動作成本異質 → 損失函數須編碼成本」，粒度變細只會讓成本更分散，**強化而非削弱**該論點 |
| **何時做** | MVP 之後。若要做，最小版本是只在量化階採 KIVI 的非對稱軸，不動作空間 |

> ⚠️ **KVTuner 用的正是 Llama-3.1-8B-Instruct 與 Qwen2.5-7B-Instruct**，跟本文主模型完全重疊。審稿人很可能知道這篇。

### B2. 外部學習式 baseline 的重訓排程

三者（KVP / ForesightKV / LookaheadKV）**已釋出程式碼、未釋出權重**。

- KVP：112 個 layer-head agent，README 載明 8-GPU DDP
- ForesightKV：至少 2 張 GPU（訓練模型 + 參考模型）
- LookaheadKV：僅 minimal training example

**論文定位已修正**（§6.2 + §7）：這一列是**佐證整體競爭力**，不是核心主張的檢驗。
核心主張由**消融表 (B) 區塊**檢驗——同一系統只換損失函數。

**排程建議**：Oracle 顯示有 headroom 之後，挑訓練成本最低的 LookaheadKV 先跑。

### B3. 預測器容量與結構消融 (D) 的結果可能推翻設計

論文 §5.2 論證 GBDT 優於神經預測器，理由是 (i) 預測器延遲直接自其收益中扣除、(ii) GPU 上的預測器會與主模型爭用。
**但這是論證，不是證據。** 消融表 (D) 區塊（GBDT / MLP 2×64 / MLP 4×256，同特徵同損失，另報熱路徑延遲）會給出答案。

(D) 區塊有四列，檢驗兩件不同的事：

| 列 | 輸入 | 檢驗什麼 |
|---|---|---|
| GBDT（本文） | 攤平特徵 | 基準 |
| MLP 2×64 / 4×256 | 攤平特徵 | **容量**——更大的模型有用嗎 |
| **GRU** | **`deltas` 序列** | **結構**——時序該手工編碼還是讓模型學 |

GRU 那列不可省略：**LRB 比的 2 層 NN 同樣吃攤平輸入，所以「GBM 勝出」的結論不涵蓋序列模型**，本文若只援引該結論即屬過度推論。

⚠️ **若任一列在計入自身延遲後仍佔優，§5.2「選 GBDT 而非神經網路」一段必須改寫。GRU 若勝出，整條特徵工程路線（deltas + EDC）都要重新考慮。** 論文已明文承諾，不能事後迴避。

### B4. Elkan 基線可能吃掉整個增益 ← 2026-08-30 新增

**這是本文最大的單點風險。**

| 基線 | 做法 | 誰的貢獻 |
|---|---|---|
| 對稱 L2 @ 0.5 | 既有方法 | — |
| **對稱 L2 @ p\*** | **只移門檻，不重訓** | **Elkan 2001** |
| (L1) 加權 + 校準 | 本文 | 本文 |
| (L2) NDCG | 本文 | 本文 |

⚠️ **若 (L1)/(L2) 相對「對稱 L2 @ p\*」沒有額外增益，則本文的損失函數貢獻歸零**，只剩 Elkan 的門檻校準——而論文已自承那不是本文貢獻。

**論文的預期與可證偽條件**（§5.3 已明文寫入）：加權訓練的作用是把有限的擬合能力配置到決策邊界附近；κ 越大，邊界越深入尾端（MI300X/8B 為 0.14，3090/70B 為 **0.005**），對稱損失越沒有誘因學準該處。
**故預測：(B) 區塊的差距應隨 κ 增大而擴大。若兩平台差距相同，主張須縮減至門檻校準。**

### B5. 跨工作負載泛化與週期性重訓

§5.2「飄移由 token 類型特徵與週期性重訓吸收」一段提出三層因應（`token_type` 特徵／300 ms 週期重訓／跨 workload 交叉測試）。
**若跨 workload 退化可忽略，重訓機制即為不必要的複雜度，應移除。** 論文已明文承諾依結果決定。

### B6. ✅ 品質分數的附錄已補（2026-09-01）

§6.4 的式 \eqref{eq:quality} 說「逐任務的 $r_t$ 完整列於附錄」。附錄 A.5
〈長上下文品質評測的逐任務分數〉已寫，表 13 給出 LongBench 7 個任務 + RULER 7 個任務
× 4 個 KV 精度的逐任務分數、配對 bootstrap CI、與公開榜單不可並列的三項理由、
以及計分函式對上游 `metrics.py` 的逐筆驗證。

**仍缺的是另一半**：那張表的列是 KV **精度**，不是**放置策略** $\pi$。
式 \eqref{eq:quality} 的 $r_t(\pi)$ 要等 Tiara 實作出來才填得滿。
目前可宣稱的是位置分層三階逐字元無損（60/60），故 $r_t$ 在那三階上恆為 1。

---

### B7. `CachePolicy` 介面表達不了完整的動作空間 ← 2026-08-30 實測新增

| | |
|---|---|
| **現況** | Fig. 2 把 Tiara 畫成「插在 vLLM 的 `CachePolicy` 介面」，藉此劃清貢獻邊界（機制屬 vLLM、策略屬本文） |
| **實測到的好消息** | 這個掛載點**真的存在且是官方支援的**。`CachePolicyFactory` 的 docstring 明寫 out-of-tree policy「no vLLM fork/patch required」，`CPUOffloadingSpec` 讀 `cache_policy_module_path`。Tiara 可以是純 plugin |
| **實測到的問題** | `CachePolicy` ABC 的方法只有 `get / insert / remove / touch / evict / clear`——它決定的是**「CPU 階裡該淘汰哪個 block」**。論文的六元動作空間裡，**`DROP → 重算` 與 GPU 內的精度降級（BF16→FP8→INT4）表達不了** |
| **影響** | §5.1 與 Fig. 2 若讓讀者以為「整個 Tiara 都能靠這個介面實現」，那是**過度宣稱**。實情是：位置分層（GPU↔CPU↔SSD）落在這個介面內，精度分層與 DROP 落在介面外，需要改動 `OffloadingSpec` 甚至 attention 路徑 |
| **處置** | 寫 §5 時把邊界講清楚：哪一部分是 plugin、哪一部分需要更深的改動。**這反而是有利的**——它說明本文不只是一個 policy plugin，而是需要擴充機制，正好回應「這是不是只是個 plugin」的質疑 |
| **何時做** | 實作 §5 之前。介面的實際形狀會決定實作切分 |

證據：`results/RUNLOG.md` A3 附帶發現；`vllm/v1/kv_offload/cpu/policies/base.py`

### B8. 頁數會隨字型而變 ← 2026-08-30 新增

| | |
|---|---|
| **現況** | 同一份 `main.tex`，macOS（Songti TC + Times New Roman）建置為 **15 頁**，Linux（Noto Serif CJK TC + TeX Gyre Termes）為 **16 頁** |
| **原因** | TeX Gyre Termes 與 Times New Roman 度量相容但非位元相同；Noto Serif CJK 的字身高與 Songti 不同。兩者累積出一頁 |
| **風險** | EuroMLSys 有 **6 頁**上限，MLSys 有頁數限制。**在 A 機器上壓到剛好上限，投稿時在 B 機器上就會超頁** |
| **處置** | 投稿前必須在**實際產生投稿 PDF 的那台機器**上重新確認頁數。壓縮階段要留至少半頁餘裕 |

---

## ⚪ C 級：文字與一致性

| # | 問題 | 位置 |
|---|---|---|
| C1 | 表 5 缺成本欄位 | §5 |
| C2 | MORI-UMBP 只在 §7 出現，未進 §3 特徵矩陣 | §3 / §7 |
| C3 | GB 與 GiB 混用 | 全文 |
| C5 | CacheCast 未查證（僅 ResearchGate pub. 406373495，無 arXiv） | refs |
| C6 | OpenReview `Vj48eXaQDM`（Learned Prefix Caching）被 CAPTCHA 擋住 | refs |

C5／C6 若查出來是真的且高度重疊，會升為 🔴。

---

## 已確認、不必再查的事

| 事項 | 結論 | 查證日 |
|---|---|---|
| vLLM 內建置換策略有哪些 | **只有 `lru` 與 `arc`**。其餘須經 `CachePolicy` 介面註冊（`CachePolicyFactory.register_cache_policy`）。故 lru + arc 即 production 預設的**完整集合**，不必再找第三個 | 2026-08-29 |
| K 是否比 V 對量化更敏感 | **是**，且有 2025 年多篇獨立驗證 | 2026-08-29 |
| 三個學習式 baseline 有無權重 | **均無**，只有程式碼 | 2026-08-28 |
| **論文標題** | 已改為平台中性：**Cost-Asymmetric Hierarchical KV State Management for Long-Context LLM Inference**。摘要、貢獻列表、結論同步以成本不對稱為主軸，容量觀察降為輔證。「光譜兩端」措辭（原 C4）一併修正為「相距 2.8 倍、位於光譜中段」 | 2026-08-30 |
| **Ampere 能否用 FP8 KV cache** | **能。** `--kv-cache-dtype fp8` 在 sm_86 實測可用，兩個模型都給出恰好 2 倍的 KV token 數而位元組佔用不變（llama 41,648→83,312；qwen 106,512→213,040）。計畫書原本寫「平台 A 量不到 GPU-FP8」是錯的——混淆了 FP8 **運算**（Ampere 沒有）與 FP8 **儲存**（可以）。論文的動作空間是儲存狀態 | 2026-08-30 |
| **vLLM 0.28.0 能否跑 Qwen2.5-7B-1M 的 DCA** | **不能。** V1 engine 沒有可用的 dual-chunk attention 路徑（`FlashAttentionImpl` 不吃 `layer_idx`，`v1/attention/` 下無對應 backend）。`--hf-overrides` 設 null 或 `{}` 都繞不過。已建 no-DCA 變體，評測上限因此是 **262,144**（該模型真正訓練的長度）而非宣稱的 1M | 2026-08-30 |
| **M1 的容量量測能否預測 M3 的行為** | **能，而且很準。** llama 的 GPU KV 容量實測 41,648 token；M3 的 `full_gpu` 在工作集 32,768（ctx 8K×4）時 warm 仍快 95%，在 65,536（ctx 16K×4）時掉到 −1.4%。轉折點正好落在量到的容量上 | 2026-08-30 |
| **RTX 3090 dense BF16 峰值算力** | **71.2 TFLOPS，論文原值正確**。$328\times1{,}695\,\text{MHz}\times128\,\text{FLOP/clk}$，兩個獨立來源 + 第一原理算術三方一致。<br>⚠️ **35.6 TFLOPS 是 TF32（64 FLOP/clk）或非 tensor FP32，資料路徑不同，不可誤用**。κ(3090)=54/190 與 FLOP/byte=2,254 全部維持不變 | 2026-08-30 |

---

## 相關筆記
- [[idea-20260828-hierarchical-kv-state]]
- [[report-hierarchical-kv-state-20260828]]
- [[report-hierarchical-kv-state-amd-feasibility-20260828]]
