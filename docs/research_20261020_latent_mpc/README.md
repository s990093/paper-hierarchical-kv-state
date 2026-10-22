# Latent MPC 推理：查新、方向、模型與 PoC 計劃（2026-10-20）

> 這份是視覺化報告 `plan.html` 的文字底稿與可追溯紀錄。
> 對應使用者貼上的研究計劃「Latent MPC 推理——把 V-JEPA 2-AC 的世界模型＋規劃搬到 LLM 推理」。
> **本檔沒有任何新的 GPU／CPU 量測**（使用者 10/20 指示：先不用驗證，先看計劃）。

**標記**（沿用 `docs/research_20261010_directions/` 的慣例）：
〔HTML〕用抓取工具讀了論文 HTML 全文的前段（工具會先摘要，不是逐字閱讀）；
〔摘要〕只讀了 arXiv 摘要頁；〔搜尋摘要〕只看到搜尋結果的摘要片段；〔README〕讀了 GitHub README；
〔算術〕我自己用公式算的；〔判讀〕我的推論或主觀評分；〔未查證〕沒有在本次查證；
🔶 推估，投稿或引用前必須自己確認；`NOT_MEASURED` 沒有量。

**查詢範圍**：2024–2026 的 arXiv，加上 NeurIPS 2025、ICML 2025/2026、ICLR 2026、ACL/EMNLP 2025 的公開頁面。
用了 20 組查詢，關鍵字有 latent reasoning × MPC、JEPA × reasoning × planning、CEM × LLM test-time、
energy-based latent reasoning、latent world model × CoT、hidden-state correctness probe、KV × latent reasoning。
**「查不到」不等於「沒有」**（CLAUDE.md 規則 7 的精神）：結論是「本次搜尋層級沒找到」。

---

## 0. 一頁結論

1. **查新**：沒找到有人把 V-JEPA 2-AC 的完整流程用在 LLM 文字推理上。完整流程指的是「凍結 encoder → 小型動作條件 predictor（teacher forcing＋rollout）→ 能量 → CEM → MPC」。
   但**每個零件都有人做過**，其中四篇重疊度高，投入前必須讀全文：
   Thoughts-as-Planning（2605.28842）、CoLA（ICML 2025）、Soft Reasoning（ICML 2025 spotlight）、LTO（ICLR 2026）。
2. **原計劃有三個結構性問題**〔判讀〕：
   - **P1 高維 CEM**：動作 aₖ＝步驟向量（896–4096 維）。CEM 在這種維度上抽出來的點會落在流形外，value head 會被鑽漏洞。V-JEPA 2-AC 的動作只有 **7 維**〔HTML〕。
   - **P2 類比在「執行」這一步斷掉**：機器人執行動作又貴又不可逆，所以需要想像。LLM「執行」一個候選步驟只要一次 prefill，而且可以從 KV 分叉。
     **H＝1 時，想像出來的 ẑₖ₊₁ 比不上你免費拿到的真實 zₖ₊₁**，世界模型等於白做。
   - **P3 EMA 不需要**：V-JEPA 2-AC 第二階段本來就凍結 encoder〔HTML〕。凍結 LLM 的 hidden state 當目標時，目標固定，不會崩塌。EMA 開／關這個消融可以刪掉。
   - **P4 計算優勢要算過才成立**〔算術，§5〕：照搬 V-JEPA 2-AC 的 CEM 預算（800×10）時，7.6B base、H＝4 的 FLOPs 是 39.6，比文字前瞻的 32 還高。0.5B base 即使把 CEM 降到 64×5，也只省 1.16 倍。
3. **建議主線（方向 A）**：**凍結 LLM、第一步用真文字、後續步用潛動作**（text-first, latent-future MPC）。
   第 1 步是 LLM 真實產生的 N 個候選。第 2…H 步在學出來的 **8–32 維潛動作空間**裡用 CEM 搜尋，由小 predictor 想像，再用 value 評分。
   要訓練的只有 <100M 參數的小頭，base 不動，所以**任何開源模型都能外掛**。
4. **零訓練版（方向 B）**當 A 的地板 baseline。**KV 感知（方向 C）**當量測層：每個比較都報 KV bytes 與 wall-clock，用 Tiara 的工具量。這是別人沒有的資產。
5. **PoC**：三道停損關 K1（訊號在不在）→ K2（想像有沒有比直接 value 好）→ K3（同計算量贏不贏）。
   估計 1–2 GPU 天〔估算，`NOT_MEASURED`〕。**等 MI300X 租約 10/23 09:00Z 結束後在 3090 跑**，不搶 Tiara 的 stage-2 佇列。

---

## 1. V-JEPA 2-AC → LLM：類比對照

V-JEPA 2-AC 的事實〔HTML，arXiv 2506.09985〕：
- 後訓練時**凍結 video encoder**，另訓練約 300M 參數的動作條件 predictor（24 層、16 頭、1024 維）
- 動作是 **7 維**末端執行器狀態差（3 位置＋3 歐拉角＋1 夾爪）
- 資料是約 62 小時、未標註的 DROID 影片（約 23k 條軌跡，含失敗）
- 損失＝teacher forcing（T＝15 的平均 L1）＋2 步 rollout 的 L1
- 能量＝想像的表徵與目標影像表徵的 L1
- CEM：800 樣本、10 輪、規劃長度 1，動作限制在半徑 0.075 的 L1 球內（約 13 cm）
- 每個動作 16 秒 vs Cosmos 4 分鐘（單張 RTX 4090）
- 成功率：Pick-&-Place Cup 80%、Box 65%；Grasp Cup 65%、Box 25%

| 元素 | V-JEPA 2-AC | LLM 版（建議） | 類比成立？〔判讀〕 |
|---|---|---|---|
| 狀態 | 凍結 ViT 的影格表徵 | 凍結 LLM 第 ℓ 層、步驟邊界的 hidden state | 成立 |
| 動作 | 7 維連續 | 步驟文字（第 1 步）＋8–32 維潛動作（第 2…H 步） | **原計劃不成立**（896–4096 維）；改了才成立 |
| 世界模型 | 300M block-causal predictor | 20–60M block-causal predictor | 成立 |
| 目標 | 目標影像 | **沒有**，改用 value（預測答對機率） | 部分成立 |
| 能量 | ‖ẑ−z_goal‖₁ | −V(ẑₖ₊H)＋不確定度懲罰 | 部分成立 |
| 執行動作 | 機器人移動：貴、不可逆 | 一次 prefill：便宜、可從 KV 分叉 | **不成立**：H＝1 的想像沒有價值 |
| 對照的生成式基線 | Cosmos（生成像素再規劃） | RAP 式文字 rollout／MCTS（生成文字再規劃） | 成立：這正是要比的 |

**最重要的推論**〔判讀〕：value head 是用 base policy 的蒙地卡羅結果訓練的，所以 V(zₖ₊₁) 本身已經在估「照 base policy 走下去會答對的機率」。
世界模型如果只是照同一個 policy 往下想像，在期望值上**跟 value head 重複**。
它要有用，只能靠兩種機制：
- **(a) 搜尋未來動作**：取 max over a₂…a_H，而不是照 base policy 的期望。這是真正的 H 步 MPC，需要便宜的潛動作。
- **(b) 表徵學習**：把下一狀態預測當成輔助目標，讓 value 更準、更省資料。

PoC 的 K2 就是把這兩個機制分開測。

---

## 2. 文獻表（依主題分組）

重疊度是對本計劃而言〔判讀〕。

### 2.1 JEPA × 語言
| 論文 | 一句話 | 重疊 | 程式碼 | 證據 |
|---|---|---|---|---|
| LLM-JEPA（Huang, LeCun, Balestriero；2509.14252） | 在 LLM 訓練中加 JEPA 損失：從一個 view 預測另一個 view 的 embedding（例：自然語言→程式碼）。需要成對多 view 資料 | 低 | ✅ github.com/rbalestr-lab/llm-jepa | 〔搜尋摘要〕 |
| JEPA-Reasoner（Liu, Chen, Woodruff；2512.19171 v3 2026-01） | JEPA 式潛空間推理器＋Talker 還原文字。target 端是 EMA 的 embedding（動量 0.98），損失用 scaled cosine。0.9B（694M＋198M）。GSM8K 8-shot：51.9 vs 同資料 Transformer 20.8。**推理時沒有搜尋** | 中 | ❌ 頁面沒有連結 | 〔HTML〕 |
| Semantic Step Prediction（Yuan；2604.18464） | 在推理步驟邊界做 STP 幾何正則，多步潛預測比凍結 baseline 準 168 倍（ProcessBench）；3 層 MLP 比線性外插再好 3–12 倍。會微調模型、沒有規劃 | 中（支持 RQ1：步驟潛狀態可預測） | ❌ | 〔摘要〕 |
| Latent JEPA（化學；2610.01947） | 連續潛思考＋JEPA 未來 view 預測，用在 ChemCoTBench | 低 | ❌ | 〔摘要〕 |
| Pearl（2604.08065） | JEPA 式多模態潛推理，從工具使用軌跡學習 | 低 | 〔未查證〕 | 〔搜尋摘要〕 |
| NextLat（Teoh 等；2511.05963） | 預訓練時預測自己的下一個 hidden state，潛狀態可證收斂到 belief state | 中（給「hidden state＝世界模型狀態」理論支撐） | 〔未查證〕 | 〔搜尋摘要〕 |
| JELAR | **查無**。計劃書列的這篇我在本次搜尋找不到 | — | — | — |

### 2.2 推理的世界模型＋規劃
| 論文 | 一句話 | 重疊 | 程式碼 | 證據 |
|---|---|---|---|---|
| **Thoughts-as-Planning**（Liu, Yu, Wu；2605.28842） | 把 CoT 優化視為潛語意空間上的序列決策：學一個潛世界模型預測「改推理鏈」對輸出的影響，用梯度或 RL 規劃 | **高**（框架最接近）。是否凍結、有無 CEM 摘要沒寫 | ✅ github.com/FastLM/Thoughts-as-Planning | 〔摘要〕 |
| **CoLA**（Jia 等；ICML 2025；2503.21383） | 用逆動力學學出潛動作空間，把 LLM **微調**成以潛動作為條件的語言世界模型，再用 BC／RL／MCTS 訓 policy。Llama-3.1-8B，MATH500：RL 42.4 vs 38.2；MCTS 變體 68.2 | **高**（動作條件世界模型）。但 base 有微調 | 〔未查證〕 | 〔搜尋摘要〕 |
| RAP（Hao 等；EMNLP 2023） | 用 prompt 讓 LLM 當世界模型，在**文字空間**跑 MCTS | 中：它在這個類比裡相當於 Cosmos，是要打的生成式基線 | 〔未查證〕 | 〔搜尋摘要〕 |

### 2.3 推理時的潛空間最佳化（多半凍結 base）
| 論文 | 一句話 | 重疊 | 程式碼 | 證據 |
|---|---|---|---|---|
| **Soft Reasoning**（Zhu 等；ICML 2025 spotlight；2505.24688） | 用**貝氏最佳化**搜尋第一個 token 的 embedding，目標由 verifier 引導 | **高**：無導數的嵌入空間搜尋，最接近 CEM | ✅ github.com/alickzhu/Soft-Reasoning | 〔摘要〕 |
| **LTO**（Du, Dong, Ning；ICLR 2026；2509.26314） | 在 Huginn-3.5B 的潛思考上訓練潛獎勵模型（判對錯），再用機率式演算法最佳化潛思考 | **高**：E1 value＋潛空間 | ❌ | 〔摘要〕；ICLR 2026 來自 PDF 頁首〔搜尋摘要〕 |
| LatentSeek（Li 等；2505.13308） | 測試時對潛表徵做 policy gradient，用自產獎勵。GSM8K／MATH-500／AIME2024 | 中–高（凍結、潛空間搜尋，但每步都跑完整 LLM） | ✅ github.com/bigai-nlco/LatentSeek | 〔搜尋摘要〕 |
| LTPO（Ye, Liang, Shan；ICLR 2026；2510.04182） | 把潛思考向量當逐題參數，用信心型內在獎勵做線上 policy gradient，不更新權重 | 中–高 | ❌ | 〔摘要〕 |
| ∇-Reasoner（Wang 等；ICLR 2026；2603.04948） | 測試時對 token logits 做梯度下降（LLM 似然＋獎勵模型），並證明與 KL 正則 RL 對偶。模型呼叫數少 10–40% | 中 | ❌ | 〔摘要〕 |
| LED（Tan 等；2602.01698） | 免訓練：用中間層 posterior 的熵選探索候選。pass@1 ＋0.61 | 低 | ✅ github.com/AlbertTan404/LED | 〔摘要〕 |
| EBT（Gladstone 等；ICLR 2026 oral；2507.02092） | Energy-Based Transformer：把預測當成能量最小化（System 2）。從頭訓練 | 中（能量框架） | 〔未查證〕 | 〔搜尋摘要〕 |

### 2.4 連續／潛 CoT
| 論文 | 一句話 | 重疊 | 程式碼 | 證據 |
|---|---|---|---|---|
| Coconut（Hao 等，Meta；2412.06769） | 把 hidden state 接回輸入做連續推理，用課程學習。連續思考可同時編碼多條路徑，呈現類 BFS 的行為 | 中 | ✅ github.com/facebookresearch/coconut（官方，MIT） | 〔README〕 |
| CODI（Shen 等；EMNLP 2025；2502.21074） | 自蒸餾到連續 CoT，GPT-2 規模在 GSM8K 首度追平顯式 CoT | 低–中 | ✅ github.com/zhenyi4/codi | 〔搜尋摘要〕 |
| Soft Thinking（Zhang 等；NeurIPS 2025；2505.15778） | **免訓練**：用機率加權的 embedding 混合當「概念 token」。pass@1 最多＋2.48，token 最多少 22.4% | 低–中 | 〔未查證〕 | 〔搜尋摘要〕 |
| SoftCoT（ACL 2025）／SoftCoT++（2505.11484） | 凍結 backbone，由小助手產生軟思考。++ 版用多個初始 token＋對比學習讓軟思考多樣化，以支援 test-time scaling | 中（凍結＋潛） | ✅ github.com/xuyige/SoftCoT | 〔搜尋摘要〕 |

### 2.5 hidden state 當 value（凍結模型）
| 論文 | 一句話 | 重疊 | 證據 |
|---|---|---|---|
| Reasoning Models Know When They're Right（2504.05419） | hidden state 探針能驗證中間答案，且校準良好；拿來提早停止，token 少 24%、表現不降 | **高**（E1 的前作） | 〔搜尋摘要〕 |
| Mid-CoT 正確性探針（2511.14773） | Qwen3-8B／Llama3.1-8B 在 MATH 上，推理 4 個 token 後線性探針 ROC-AUC 約 0.8 | **高**（K1 門檻的先驗） | 〔搜尋摘要〕 |
| LYNX（2512.05325）、BLADE（2607.28966） | 探針加保形預測決定在哪裡提早停 | 低 | 〔搜尋摘要〕 |

### 2.6 潛狀態 × KV（多半免訓練）
| 論文 | 一句話 | 重疊 | 程式碼 | 證據 |
|---|---|---|---|---|
| KV Cache Steering（Belitsky 等；2507.08799） | 對 KV cache 做一次性介入，引導凍結 LLM 的推理風格 | 中 | 〔未查證〕 | 〔搜尋摘要〕 |
| LatentMAS（2511.20639） | 免訓練的多代理潛空間協作，以 KV 當工作記憶。output token 少 70.8–83.7%、端到端快 4–4.3 倍 | 中（KV 是無損的潛狀態載體） | ✅ github.com/Gen-Verse/LatentMAS | 〔搜尋摘要〕；ICML 2026 來自 hyper.ai 🔶 |
| Cache-to-Cache（Fu 等；ICLR 2026；2510.03215） | 投影並融合兩個模型的 KV 來溝通 | 低–中 | ✅ github.com/thu-nics/C2C | 〔搜尋摘要〕 |
| Hogwild! Inference（2504.06261） | 多個 worker 共用同一份 KV、靠 RoPE 重排，不需微調 | 中（方向 C：搜尋分支的 KV） | 〔未查證〕 | 〔搜尋摘要〕 |

### 2.7 潛世界模型規劃理論（控制領域）
| 論文 | 對本計劃的用處 | 證據 |
|---|---|---|
| A Control Theory of Predictability in Latent World Models（2607.10362） | CEM receding horizon；**規劃的 regret 取決於「預測的計劃成本能不能追蹤真實成本」，而不是平均預測誤差**。所以 K2 用排名相關係數，不只看 L1 | 〔搜尋摘要〕 |

### 2.8 計劃書引用的查核
| 計劃書的說法 | 查核 |
|---|---|
| JEPA-Reasoner「指出 V-JEPA 2-AC 需要預先給定目標」 | **沒找到**。v3 HTML 的工具摘要說全文沒提 V-JEPA 2-AC。可能是 v1／v2 才有，或是記錯。引用前要自己翻原文 |
| JELAR（音訊） | **查無** |
| LLM-JEPA、JEPA-Reasoner、Coconut 的開源程式碼 | LLM-JEPA ✅、Coconut ✅（官方，MIT）、JEPA-Reasoner ❌（沒有連結） |
| V-JEPA 2-AC 的數字 | ✅ 與原文一致（見 §1）。「65–80%」對應 Pick-&-Place；Grasp Box 只有 25% |

---

## 3. 新穎性：站得住的點與要調整的點

**站得住的點**〔判讀〕：
1. **凍結 base＋小 predictor 想像，取代昂貴的文字 rollout**。對照：V-JEPA 2-AC 對 Cosmos，相當於本計劃對 RAP／文字 MCTS。
   要成立，必須搭配**低維潛動作上的 CEM**，而且 **H≥2**。
2. **用跟決策有關的指標評估推理世界模型**（排名相關 vs 蒙地卡羅真值），並做「跟直接 value 探針重不重複」的檢定。**就算結果是負面的，這個分析本身也是新的**。
3. **失敗軌跡當作「動力學」的訓練資料**（RQ3）。注意：LTO 和探針文獻已經用失敗軌跡訓練 value，所以只剩 dynamics 這一半是新的。
4. **把 KV 算進計算量的比較**（方向 C）：test-time search 的 KV 足跡與放置，是系統面的貢獻。

**要調整的點**：
- 刪掉 EMA 消融（目標凍結，不會崩塌）。改成監控 predictor 輸出的方差，並一律跟 copy／線性 baseline 比。
- E2（goal predictor）降為選配：「從題目預測答案狀態」有循環論證的問題。RQ4 改比 E0（零訓練）／E1（value on 真實狀態）／E1′（value on 想像狀態）／E1′＋不確定度。
- 不在 896–4096 維上做 CEM，改在學出來的 8–32 維潛動作上做。
- 執行的動作一律是 LLM 產生的真文字步驟。soft token 寫回凍結 LLM 會分佈外（Coconut 需要訓練才行），降為選配。
- 用 0.5B 當 base 時，計算優勢只剩 1.16 倍（§5 算術）。主實驗用 1.5B／7B；CEM 預算從 V-JEPA 2-AC 的 800×10 降到約 64×5。

---

## 4. 方向比較

評分 1–5〔判讀〕。可行性以「單張 24 GB 卡、4 週內」為準；「大家可用」指凍結 base、可外掛。

| 方向 | 一句話 | 新穎 | 可行 | 大家可用 | 契合現有資產 | 建議 |
|---|---|---|---|---|---|---|
| **A 文字先、潛未來 MPC** | 第 1 步用 LLM 的真候選，第 2…H 步在 8–32 維潛動作上做 CEM，由小 predictor 想像、value 評分 | 4 | 3 | 4 | 3 | **主線** |
| **B 零訓練潛能量** | 能量全部取自凍結模型本身（強制作答的信心、候選 hidden state 的群聚一致性），不訓練任何東西 | 2 | 5 | 5 | 2 | 當 A 的地板 baseline |
| **C KV 感知搜尋** | 每個比較都報 KV 峰值 bytes 與 wall-clock；搜尋分支的 KV 依 value 做 Tiara 式放置（留 GPU／降精度／丟掉重算） | 3 | 4 | 4 | 5 | 量測層；之後可獨立成系統論文 |
| 原計劃照寫 | 潛空間提案、高維 CEM、soft token 寫回 | 4 | 2 | 3 | 2 | 不建議照做 |
| D 從頭訓練 JEPA reasoner | 照 JEPA-Reasoner／EBT 的路線預訓練 | 3 | 1 | 1 | 1 | 不建議 |

---

## 5. 計算量與 KV 的算術〔算術〕

**每次決策的成本**，單位是「解碼一個候選步驟」（≈ 2·N_base·L FLOPs）。設 N＝8 個候選、每步 L＝40 token、predictor 50M：
- 文字前瞻（每個候選真的往下寫 H 步）：N·H
- 方向 A：第 1 步真寫，成本 N。之後每個候選各跑一次 CEM，評估 S 個樣本 × (H−1) 個想像步，每個想像步的成本是 u ＝ N_pred／(N_base·L)
  - 合計：N ＋ N·S·(H−1)·u
  - u 在 0.5B／1.5B／7.6B／8B 上分別是 2.55×10⁻³／8.12×10⁻⁴／1.64×10⁻⁴／1.56×10⁻⁴

base 7.6B（例：Qwen2.5-7B）：

| H | 文字前瞻 | A，CEM 64 樣本×5 輪（S＝320） | A，CEM 800×10（S＝8,000，V-JEPA 2-AC 的預算） |
|---|---|---|---|
| 1 | 8 | 8.00 | 8.00 |
| 2 | 16 | 8.42 | 18.53 |
| 4 | 32 | 9.26 | **39.58**（比文字還貴） |
| 8 | 64 | 10.95 | 81.68 |

H＝4、CEM 64×5 時，文字前瞻／A 的倍數依 base 大小：0.5B **1.16×**、1.5B 2.25×、7.6B 3.45×、8B 3.48×（上限是 H＝4×）。

**結論**：
- H＝1 時 A 沒有成本優勢，也沒有品質優勢（見 §1）。
- **V-JEPA 2-AC 的 CEM 預算不能照搬**：800×10 在 7.6B base、H≥4 時 FLOPs 比文字前瞻還高。在 LLM 上要降到約 64×5。
- base 越大越划算；0.5B 幾乎沒有優勢，所以主實驗用 1.5B／7B 以上。
- 註：這是 FLOPs 近似。decode 受記憶體頻寬限制、CEM 樣本可以大批次跑，所以 wall-clock 很可能比 FLOPs 更偏向 A，但這要量（`NOT_MEASURED`）。

**KV 足跡**：每 token、BF16，由本機 config 算 2 × 層數 × KV 頭數 × head_dim × 2 bytes。
- Llama-3.1-8B（32 層、8 KV 頭、128）：131,072 B ＝ **128 KiB/token**
- Qwen2.5-7B（28 層、4 KV 頭、128，head_dim 由 3584/28 推得）：57,344 B ＝ **56 KiB/token**

| 情境 | 額外 KV token | Llama-8B | Qwen-7B |
|---|---|---|---|
| 文字前瞻 N＝8、H＝4、L＝40 | 1,280 | 160 MiB | 70 MiB |
| 方向 A，同設定（只有第 1 步） | 320 | 40 MiB | 17.5 MiB |
| 長 CoT 搜尋：16 分支 × 4,096 token | 65,536 | 8 GiB | 3.5 GiB |

這就是方向 C 的切入點：長 CoT 的搜尋分支在 24 GB 卡上就是 Tiara 的容量問題。

---

## 6. 對應模型

| 角色 | 模型 | 理由 | 本機狀態 | 3090 24 GB |
|---|---|---|---|---|
| 凍結 base：煙霧測試 | Qwen2.5-0.5B-Instruct | 跟計劃書一致，分鐘級就能跑完 | 需下載 | ✅ |
| 凍結 base：**PoC 主力** | Qwen2.5-1.5B-Instruct（或 Qwen2.5-Math-1.5B-Instruct） | GSM8K 對錯比例適中，有足夠失敗軌跡；計算比 0.5B 好看 | 需下載 | ✅ |
| 凍結 base：主實驗／方向 C | **Llama-3.1-8B-Instruct**、**Qwen2.5-7B-Instruct** | Tiara 已在這兩個模型上量過 κ（CPU／SSD／重算成本），方向 C 可以直接接 | Llama ✅ 已在 hf-cache（unsloth 鏡像）；Qwen 本機只有 **7B-Instruct-1M**（長上下文變體），跟文獻比較時建議另下標準版 | ✅（BF16 權重約 15–16 GB，GSM8K 短上下文夠用） |
| 長 CoT 推理模型（方向 C） | DeepSeek-R1-Distill-Qwen-1.5B／7B 或 Qwen3-1.7B／4B／8B（thinking） | 搜尋分支長，KV 成本最明顯 | 需下載 | ✅（8B 以下） |
| 潛推理對照 | Coconut（GPT-2 規模，官方碼）、CODI、Huginn-3.5B（LTO 用的） | 對照「訓練式潛推理」 | 需下載〔Huginn 的 HF id 未查證〕 | ✅ |
| PRM／verifier baseline | Qwen2.5-Math-PRM-7B | best-of-N 與步驟 beam 的標準強 baseline | 需下載〔模型 id 未查證〕 | ✅ |
| 新訓練：predictor | 4–6 層 block-causal Transformer，d＝512–768，約 20–60M | 規模對應 V-JEPA 2-AC 的 300M，按 base 縮小 | — | ✅ |
| 新訓練：潛動作編碼器 | 逆動力學 aₖ＝g(zₖ, zₖ₊₁)，瓶頸 8–32 維，約 1–5M | 讓 CEM 在低維上可行（對應 7 維動作） | — | ✅ |
| 新訓練：value head | 線性或 2 層 MLP 探針，約 1–5M | E1；文獻 AUC 約 0.8 | — | ✅ |
| 解碼（可解釋性） | 免訓練：logit lens／Patchscopes 用凍結 base 本身；訓練式 decoder 選配 | 不一定要訓練 decoder | — | ✅ |

**hidden 維度**（本機 config）：Llama-3.1-8B 是 4096，Qwen2.5-7B 是 3584。

---

## 7. PoC 計劃（停損關先寫死）

判準在跑之前寫下（預先登記）。門檻是我選的〔判讀〕，要改請在開跑前改。

| 階段 | 做什麼 | 通過 | 不通過就 |
|---|---|---|---|
| P0 資料 | Qwen2.5-1.5B-Instruct；GSM8K train 前 2,000 題 × 8 取樣（T＝0.7），test 全部 1,319 題。以換行切步，數值比對 `####` 標對錯。抽層 ℓ∈{L/4, L/2, 3L/4, L} 的步驟邊界 hidden（最後 token 與平均池化各一份）。**載入斷言：每條軌跡的步驟 token 數總和＝整條 token 數**（規則 6：用 A 驗 B） | 斷言全過 | 停下回報 |
| **K1 訊號** | 依題目切分 train／val（避免同題洩漏），線性探針 zₖ → 最終對錯 | 軌跡中段 AUC ≥ 0.70 | AUC < 0.60：E1 能量死，轉方向 B／C |
| K2a 健全性 | predictor（L_tf＋L_ro）vs copy baseline（ẑₖ₊₁＝zₖ）、線性 baseline | L1 ≤ copy 的 0.8 倍 | 檢查切步與層選擇 |
| **K2b 想像有沒有用** | 300 個前綴 × 8 個候選步，各續寫 8 次估真實 value。比 Spearman ρ：(i) V(真實 zₖ₊₁) 排名（H＝1，不用世界模型）；(ii) H＝2／4 的 CEM 想像分數排名；(iii) value head 改接 predictor 特徵（機制 b） | (ii) 或 (iii) 比 (i) 高 ≥ ＋0.05 | **世界模型多餘**：寫成負面結果，主線轉 C |
| **K3 同計算量** | GSM8K test：greedy、SC@{4,8,16}、BoN＋探針、步驟 beam＋V(真實 z)、PRM 步驟 beam、A（H＝2／4）。報 accuracy vs {生成 token、FLOPs〔算術〕、KV 峰值、wall-clock} | 在至少一個預算點勝過「步驟 beam＋V(真實 z)」≥ 1.0 pt，且 bootstrap 95% CI 不跨 0 | NO-GO：A 不成立，結果寫成分析 |
| RQ3 | predictor／value 只用正確 vs 正確＋錯誤軌跡訓練，用 K2b 指標比 | 報告差異 | — |

**資源估計**〔估算，`NOT_MEASURED`〕：取樣約 16k 條 × 約 250 token ≈ 4M token，K2b 續寫約 19k 條 × 約 150 token ≈ 2.9M token，hidden 抽取與小模型訓練加起來是小時級。全程約 1–2 GPU 天。
**時程**：10/21–10/23 MI300X 留給 Tiara stage-2（記憶：租約 10/23 09:00Z 結束）。第 1 週 P0＋K1，第 2 週 K2，第 3 週 K3，第 4 週 RQ3 與判定，全部在 3090 上做。

**照 CLAUDE.md**：每個 run 走 §4.1 的殼（`run_id`、`cmd.sh`、`context.txt`），用 `gpu_guard` 包住，CSV 帶 `run_id`／`ts`。
品質分數在污染時保留，但時間欄作廢（同 §3 的品質量測規則）。

---

## 8. 跟 Tiara 的連結

- **可直接重用**：`gpu_guard.py`（共用機器防插隊）、§4 的記錄協定、`datasets/gsm8k/`（本機 `/mlsteam/data/tiara/datasets/gsm8k/` 已有）、vLLM harness、m10 品質 harness，以及 Llama-8B／Qwen-7B 上已量的成本常數。
- **方向 C 的具體版本**：test-time search 同時養著 N 個分支的 KV。value head 預測「這個分支之後還會不會被展開」，可以直接當 Tiara 動作空間 {GPU-BF16, FP8, INT4, CPU, SSD, DROP＋重算} 的放置訊號：低 value 的分支降精度或丟掉。
  這是「品質約束下、動作成本異質」的同一個框架，換到搜尋情境〔判讀〕。

---

## 9. 投稿場所（頂會）🔶 全部要自己確認

依照典型年度週期推估，**沒有查官方頁面**：
- **K3 GO（正面結果）**：NeurIPS 2027（通常 5 月截稿）、ICML 2027（通常 1 月底，偏趕）、ICLR 2028（通常 9–10 月）、ACL／EMNLP 2027（走 ARR）、COLM 2027（通常 3 月）
- **K2b／K3 是負面結果**：NeurIPS／ICLR 的 workshop，或 TMLR（隨時可投）
- **方向 C 獨立成系統論文**：MLSys、EuroMLSys。注意要跟 Tiara 現有的投稿路線分開（見 `VENUES.md`）

---

## 10. 風險

| 風險 | 機率〔判讀〕 | 對策 |
|---|---|---|
| 世界模型跟 value 探針重複（K2b 不過） | 高 | K2b 就是為了提早抓到這件事；結果寫成負面分析 |
| CEM 鑽 predictor 漏洞（樂觀偏差） | 中–高 | ensemble 不一致度懲罰、潛動作限制在經驗分佈的 L1 球內（照 V-JEPA 2-AC 的 0.075 半徑做法） |
| 換行切步不穩 | 中 | 固定 token 長度切分當對照；載入時寫斷言 |
| 跟 Thoughts-as-Planning／CoLA 撞題 | 中 | 第 1 週先讀完四篇高重疊論文的全文，再決定是否開跑 K2 |
| 跟 Tiara 時程衝突 | 中 | PoC 上限 4 週、約 2 GPU 天；Tiara 的截稿優先 |

---

## 11. 投入前要讀的全文（第 1 週）

1. Thoughts-as-Planning（2605.28842）：是否凍結？規劃器是什麼？有沒有 CEM／MPC？
2. CoLA（2503.21383）：潛動作的維度與離散化方式；MCTS 的成本
3. Soft Reasoning（2505.24688）：貝氏最佳化的維度與預算；跟 SC 的同計算量比較
4. LTO（2509.26314）：潛獎勵模型的訓練資料、AUC、能不能用在一般 LLM

---

## 來源

- V-JEPA 2：https://arxiv.org/abs/2506.09985
- LLM-JEPA：https://arxiv.org/abs/2509.14252 ／ https://github.com/rbalestr-lab/llm-jepa
- JEPA-Reasoner：https://arxiv.org/abs/2512.19171
- Semantic Step Prediction：https://arxiv.org/abs/2604.18464
- Thoughts-as-Planning：https://arxiv.org/abs/2605.28842 ／ https://github.com/FastLM/Thoughts-as-Planning
- Latent JEPA（化學）：https://arxiv.org/abs/2610.01947
- Pearl：https://arxiv.org/abs/2604.08065
- NextLat：https://arxiv.org/abs/2511.05963
- CoLA：https://proceedings.mlr.press/v267/jia25e.html ／ https://arxiv.org/abs/2503.21383
- RAP：https://aclanthology.org/2023.emnlp-main.507
- Soft Reasoning：https://arxiv.org/abs/2505.24688 ／ https://github.com/alickzhu/Soft-Reasoning
- LTO：https://arxiv.org/abs/2509.26314
- LatentSeek：https://arxiv.org/abs/2505.13308 ／ https://github.com/bigai-nlco/LatentSeek
- LTPO：https://arxiv.org/abs/2510.04182
- ∇-Reasoner：https://arxiv.org/abs/2603.04948
- LED：https://arxiv.org/abs/2602.01698
- EBT：https://arxiv.org/abs/2507.02092
- Coconut：https://arxiv.org/abs/2412.06769 ／ https://github.com/facebookresearch/coconut
- CODI：https://aclanthology.org/2025.emnlp-main.36/
- Soft Thinking：https://arxiv.org/abs/2505.15778
- SoftCoT++：https://arxiv.org/abs/2505.11484 ／ https://github.com/xuyige/SoftCoT
- Probing self-verification：https://arxiv.org/abs/2504.05419
- Mid-CoT probe：https://arxiv.org/abs/2511.14773
- LYNX：https://arxiv.org/abs/2512.05325 ；BLADE：https://arxiv.org/abs/2607.28966
- KV Cache Steering：https://arxiv.org/abs/2507.08799
- LatentMAS：https://arxiv.org/abs/2511.20639 ／ https://github.com/Gen-Verse/LatentMAS
- Cache-to-Cache：https://arxiv.org/abs/2510.03215 ／ https://github.com/thu-nics/C2C
- Hogwild! Inference：https://arxiv.org/abs/2504.06261
- Control Theory of Predictability：https://arxiv.org/abs/2607.10362

---

## 12. 2026-10-20 讀全文後的更新、入門導讀 PDF

使用者要求「重新介紹＋故事＋論文導讀＋數據說明＋架構圖截圖，整包做成 PDF」。這次下載並閱讀 17 篇論文 PDF（閱讀深度見 PDF 附錄 C），結論有以下修正：

| 項目 | 原本（只讀摘要） | 讀全文後 |
|---|---|---|
| Thoughts-as-Planning | 是否凍結不明，可能落在 ★ 格 | 目標 LLM 凍結、另訓 4 層 512 維世界模型，但**規劃的是 prompt／推理鏈文字的編輯**，不是逐步推理〔原文 p.3–5〕。主表數字有疑點：StrategyQA 是是非題，所有方法卻都只有 29–37%〔原文 p.7〕 |
| LatentMAS 場所 | ICML 2026 🔶 | 確認：PMLR 306〔原文 p.1〕 |
| JEPA-Reasoner 引用 | v3 HTML 沒找到 V-JEPA | PDF 全文確認**沒有任何 V-JEPA 字樣** |
| 新增 P5 | — | CoLA 刻意用離散 codebook（N＝64），因為連續潛動作有「捷徑」問題〔原文 p.16、p.21〕→ PoC 加「連續 vs 離散」消融 |
| 新增 P6 | — | LTO：β→0 等於潛空間 best-of-N，獎勵模型不完美時會被鑽〔原文 p.7〕；探針論文：Llama-3.1-8B-Instruct 的 AUC 0.82 vs 推理模型 0.86，但 ECE 0.23 vs 0.03〔原文 p.6〕→ 能量只用排序、K1 加測 R1-Distill-Qwen-1.5B |
| Soft Reasoning | BO 搜第一個 token | 也明講 BO 超過 20–30 維沒效率，先隨機投影到 d＝50〔原文 p.5、p.8〕，直接支持 P1 |

**產出**（PDF 含原論文截圖，不進 git、不放公開 repo）：
- PDF：`/mlsteam/data/tiara/papers/latent_mpc_20261020/out/LatentMPC_guide_20261020.pdf`（55 頁）
- 建置：`guide/make_diagrams.py`（13 張示意／算術圖）→ `guide/build_guide.py`（PyMuPDF Story 排版）；內容在 `guide/content.py`
- 原始資料：`/mlsteam/data/tiara/papers/latent_mpc_20261020/{pdf,txt,fig,diagrams}`

**建置踩到的坑**（給之後要用 PyMuPDF Story 的人）：
1. `place()` 之後一定要 `draw()`（只量頁碼時畫到 `None`），不然同一段內容會一直重排、無限換頁。
2. CSS `page-break-before: always` 放在每頁第一個元素上也會無限迴圈 → 每章用獨立 Story。
3. 放不下的圖會被**縮小**塞進頁底，不會移到下一頁 → 建置後偵測被縮的圖，在它前面拆新頁再建一次。
4. `Document.subset_fonts()` 對 Noto CJK 的 CFF 字型報錯；改用 fontTools 先裁子集，而且要 `retain_gids=True`，否則中文全變空白。
5. 表格跨頁時表頭底色會在頁底多畫一條 → 表頭不用底色。
