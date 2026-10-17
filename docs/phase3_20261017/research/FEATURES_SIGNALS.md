# 特徵與訊號的文獻查核（補 02_predictor_inventory §4 的 G1–G5）

- **寫於**：2026-10-17，文獻查核子 agent。沒有跑任何第三階段的模擬或量測。**本文沒有任何實驗結果。**
- **狀態**：完成（中途因 API 用量上限中斷一次，先存檔後續寫；中斷造成的查不到項目列在 §7）。
- **不重做**：`LIT_AND_TRACES.md` 已查的 trace、loss、Copilot 的 AUC 0.73；E05／E06 卡已有的欄位。本文只補缺的數字。
- **標記**：〔原文 p.X〕自己讀到的原文（**PDF 頁序**）；〔讀圖〕從原文圖上目測，誤差約 ±0.02（比例）或 ±1（分數），只當量級；〔程式碼〕讀了程式碼；〔文件〕官方文件；〔摘要〕只讀摘要；〔未查證〕查不到；〔判讀〕我的推論。
- **下載**：PDF 存在 `/ssd8/hungwei/paper-hkv/tmp/p3feat/<名稱>/`，每份一個子目錄，單檔都 <50 MB，沒有進 repo。

---

## 0. 一頁摘要

1. **有沒有人在同一任務上正面比過「中繼資料 vs 文字 vs 中間狀態」？** 三者同場的**沒有**。兩兩同場的有：
   - **文字 vs 中間狀態（目標：輸出長度）**：有，而且結論一致——服務模型自己的 hidden state 比外部文字模型（BERT／DistilBERT／OPT）準。OUTLETS（EMNLP'26）在同一資料、同一目標（prompt 時的靜態預測）上，hidden-state MLP 比微調過的 BigBird-RoBERTa 的 MAE 低 **5.5–55%**（Llama-3.1-8B：27–55%；Llama-3.2-1B：12–19%；Qwen3-30B-A3B：5.5–20%）〔原文 p.7 Table 1〕。EGTP（ICLR'26）在 ForeLen 上，最後一層 hidden state＋熵加權 pooling 比每個模型上最好的文字代理（LTR-C 或 S3）低 **28–48%**；同一篇裡的 TRAIL（hidden state、普通 pooling）只比最好的文字代理低 **5–19%**〔原文 p.8 Table 1；百分比為〔計算〕〕。也就是「讀 hidden state」和「怎麼 pooling」各占一部分。
   - **中繼資料 vs 文字（目標：會不會續聊／會不會回到對話）**：有，**結論依資料而變**。LPC：只用輪數（LPC_turns）在 LMSys 上 MCC 0.351，完整版（文字嵌入＋輪數）0.386；在 ShareGPT 上 −0.001 vs 0.282；Arena 0.078 vs 0.369〔原文 p.21 Table 3〕。社群網路的 re-entry：Backstrom（WSDM'13）的文字基準 AUC 只有 0.49–0.53，結構＋時間特徵 0.64–0.86〔原文 p.8 Table 5〕；Zeng（ACL'19）反過來，「使用者過去回來的比例」AUC 50.1／50.7（接近隨機），內容模型 60.4／64.1，再加使用者歷史文字 62.7／67.1〔原文 p.7 Table 2〕。
2. **中間狀態的哪一層最好？** 三篇有層數掃描，**中段（約 1/3–2/3 深）最好；最後一層略差（EGTP 差 5%）或明顯差（TRAIL 約差 44%）；第 0 層（嵌入）最差**：TRAIL（Llama3-8B，32 層）第 10–15 層最好、第 11 層選用，最後幾層的 MAE 約 79 vs 最好約 55〔讀圖，原文 p.4 Fig.2〕；EGTP（Qwen2.5-0.5B，24 層）第 12 層 MAE 70.26、第 16 層 70.79、最後一層 73.93、第 1 層 85.91〔原文 p.25 Table 7〕；How-Much-Left（一個 7–8B 模型）早期層接近常數基準，中段大幅下降、上 1/3 持平，**所有層串接比任何單層都好**〔原文 p.12 Fig.4〕。**比文字嵌入好多少**見第 1 點。
3. **預測的目標都是「輸出長度」**，不是「會不會回來、多久回來」。**查不到**用服務模型 hidden state 預測續聊、使用者滿意度、或回來時間的論文；唯一聲稱這樣做的 SAECache，arXiv 仍只有 v1、程式碼沒釋出，p.7–8 與 p.19–20 的矛盾無法釐清（§1.4）。
4. **LPC_turns**（只用輪數、按輪數分箱取多數標籤）：見第 1 點；作者自己說 LMSys 上「輕量的 LPC_turns 可能就夠了」〔原文 p.21〕。另外 LPC 結論說，要再逼近 oracle「可能需要新的資料維度，例如使用者的存取型態」〔原文 p.23〕。
5. **Copilot 回來時間預測器的特徵重要度**（讀圖）：已閒置的平均時間 ≈28.7、輪次 ≈25.6、上一次閒置 ≈11.6、LLM 成功率 ≈10.8、該輪時長 ≈8.5、LLM 呼叫數 ≈7.6、prompt token ≈2.0、使用者取消 ≈1.9、completion token ≈1.5〔讀圖，原文 p.18 Fig.33a〕；原文：前兩項「合計超過一半」，「逐次呼叫的特徵（token 數）貢獻很小」〔原文 p.17〕。**沒有特徵消融**，只有「只看上一次閒置」的規則 AUC 0.58 vs 完整 0.73〔原文 p.17〕。
6. **頻率**：LRB 在 CDN 上，只用靜態特徵的 good decision ratio 0.30–0.54，加 deltas／EDC 到 0.75–0.87〔讀圖，原文 p.7 Fig.5〕；但在 LLM 前綴快取上，**頻率當排序完全失敗**（LFU、W-TinyLFU「completely collapse」）、**當准入有用**（拿到第 2 次命中後再被重用的機率 69.4%→88.7%；S3-FIFO 在 Qwen To-B 24 GiB 0.491 vs LRU 0.403）〔Fancy-eviction 原文 p.4、p.8、p.10〕。SGLang 與 Dynamo 的 2-hit 規則**都沒有公開評估**〔程式碼〕〔文件〕。
7. **別的領域「會不會回來」**：時間型態最強。Grob（電商）光用「已缺席多久」不回來 AUC 0.743，Cox 彙總特徵 0.793，RNN 從原始序列學 0.796〔原文 p.12 Table 1〕；Backstrom 的「最近兩則的間隔」是前兩名特徵〔原文 p.8〕。內容 vs 使用者歷史沒有一致答案（§4）。
8. **G5：query 能不能用在讀取時？** 能，而且已有前作做了一半：IMPRESS（FAST'25）在 prefix 被重用時依新 query 只從慢層載入重要 token〔原文 p.5、p.7〕；EpiCache（ICML'26）每個新 query 選一份 episode 的 KV 從 CPU 取回〔原文 p.4–5〕；Cocktail（DATE'25）依 query 與 chunk 相似度選 FP16／INT4／INT2〔原文 p.3〕。「回來時同時決定還原哪些、什麼精度、哪些重算」的組合**查不到**。SCBench 顯示只依一個 query 決定丟棄的方法下一輪就壞（Llama-3.1-8B 多輪平均：SnapKV（τ＝1/32）20.9 vs 全量 48.7）〔原文 p.4、p.9 Table 4〕，所以慢層必須留完整副本。

---

## 1. G1：用服務模型的中間狀態預測系統層目標

### 1.1 總表（每篇用哪個模型、哪一層、probe 形式、和誰比）

| 論文 | 目標 | 服務模型 | 層、pooling | probe／loss | 對照（同資料同目標） | 結果 | 推論開銷 | 出處 |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| **TRAIL**（Shahout et al.，ICLR'25；arXiv 2410.01035v1） | 剩餘輸出長度（10 個等寬桶，0–512 token） | Llama3-8B-Instruct，Alpaca | 32 層全掃；prefill 用所有輸入 token 的平均，decode 用當下 token；選第 11 層 | 2 層 MLP（512 隱藏），CE；逐步 Bayesian 平滑 | BERT（S3 式，只看 prompt、預測一次，之後每生成一個 token 減 1） | refined 第 11 層 MAE 約 32 vs BERT 約 86.5（原文「2.66× lower」）；**未平滑**的各層約 39–42〔讀圖 p.6 Fig.3〕 | MLP 約 2.1M 參數（≈ 模型 FLOPs 的 0.03%）；每筆 CPU 5.9–9.4 µs、GPU 0.43–0.62 µs〔原文 p.6 Table 1〕 | 〔原文 p.4–7〕 |
| **EGTP**（Xie et al.，ICLR'26；arXiv 2602.11812v2） | 總輸出長度（靜態，prompt 時）；PLP 另做逐步剩餘長度 | Qwen2.5 3B／7B、Llama3.2 1B／3B（ForeLen：LongSeq、Reasoning、RL）；LMSYS 列（GPT-4、Claude-2 的回應）用哪個模型的 hidden state，原文沒寫清楚〔未查證〕 | **最後一層**，用 token 熵加權 pooling | 20 桶 soft label，λ·CE＋(1−λ)·MSE，λ＝0.95 | SSJF-Reg／MC（BERT）、S3（DistilBERT）、LTR-C（OPT）、PiA（LLM 自己報長度）、TPV、TRAIL（hidden state） | 四模型平均 MAE：EGTP 101.5–110.8；TRAIL 138.0–157.9；LTR-C 150.4–207.9；S3 168.0–216.5；SSJF-Reg 178.8–265.9〔原文 p.8 Table 1〕 | 0.65–0.67 ms、5.5–7.2 MB；TRAIL 2.0–2.6 ms、268 MB；BERT 類 2.3–4.2 ms、264–270 MB；PiA ≈60 ms（RTX 4090）〔原文 p.27 Table 9–10〕 | 〔原文 p.6–9、p.25、p.27〕 |
| **OUTLETS**（Wen et al.，EMNLP'26；arXiv 2609.01068v2） | 總輸出長度（靜態）＋逐步剩餘長度 | Llama-3.2-1B、Llama-3.1-8B、Qwen3-30B-A3B；ShareGPT、Alpaca、LMSYS | 本法：低／中／高三層 hidden state 融合後過 EAGLE-3 的 draft decoder；**MLP 基準**：「目標模型的中間層 hidden state」（哪一層原文沒寫〔未查證〕） | 2 層 FC 回歸頭 | BERT（bigbird-roberta-base **微調**，[CLS]＋同樣的 2 層頭，MSE）、OPT-125m（50 桶分類）、PIA 式提示 | 見 §1.2 | 已有 SD backbone 時只多回歸頭 0.7 ms／步、2.6M 參數；SD backbone 自己 1.7 ms／步、72.3M（d＝2048）〔原文 p.7〕 | 〔原文 p.6–7、p.14〕 |
| **How Much is Left?**（Merzouk et al.，arXiv 2607.05316v1） | 總長度（prompt 最後一個位置）與剩餘長度 | Llama-3.1-8B、Olmo-3-7B、Mistral-7B；7 個資料集 | 主結果用**所有層串接**；附錄單層掃描 | 單一線性層（無非線性） | 只有常數中位數基準（**沒有文字嵌入對照**） | 自然語言資料集：prompt 結尾 probe 的 MAE 是常數基準的 0.5–0.9 倍〔原文 p.6 Table 1、p.12〕 | 一次前向抽 hidden state，probe 成本可忽略〔原文 p.6〕 | 〔原文 p.6、p.12〕 |
| **LTR**（Fu et al.，NeurIPS'24；arXiv 2408.15792v1）〔對照用，不是中間狀態〕 | 輸出長度的**排序** | Llama-3-8B／70B | 不用服務模型；OPT-125m／350m 讀 prompt | ListMLE 排序 loss vs 分類 | 分類（不同桶寬）、PO（LLM 自己先生成長度） | Kendall τ：排序 0.54／0.62；PO 0.51／0.67；「另一個 seed 的實際長度」當 oracle 也只有 0.74／0.84〔原文 p.8 Table 3〕 | — | 〔原文 p.8〕 |

- **只用文字的輸出長度預測器**（當 F3 類對照的前例）：S3 用微調 DistilBERT 分 10 桶；SSJF（μ-Serve 同一組作者）用微調 BERT 做回歸或分類；PiA 讓 Vicuna 自己報長度〔二手：EGTP 原文 p.26 附錄 E.1〕。ELIS（arXiv 2505.09142）用 BGE＋8 層 FC，並在生成過程中反覆重估〔摘要〕。μ-Serve 原文沒讀〔未查證〕。
- **EGTP 的同場對照，換算成「hidden state 比最好的文字代理低多少」**〔計算，原文 p.8 Table 1〕：Qwen2.5-3B 28.2%（TRAIL 5.0%）；Qwen2.5-7B 31.2%（8.2%）；Llama3.2-1B 40.3%（13.8%）；Llama3.2-3B 48.2%（19.4%）。注意：EGTP 的端到端（Table 2）裡，文字代理 LTR-C 的 padding ratio（0.21／0.14）比 TRAIL（0.51／0.34）好〔原文 p.9 Table 2〕——**MAE 較好不等於決策較好**。
- 〔判讀〕LTR 的「oracle＝另一個 seed」很重要：同一個 prompt 重抽一次，長度排序的 τ 也只有 0.74–0.84。也就是**目標本身有隨機性造成的天花板**；回來時間（人的行為）的天花板可能更低。

### 1.2 OUTLETS Table 1：同資料、同目標的「中間狀態 vs 文字」（靜態預測，MAE，越低越好）〔原文 p.7 Table 1〕

| 模型 | 資料 | 平均長度 | hidden-state MLP | BERT（微調） | OPT | PIA | MLP 比 BERT 低 |
|:--|:--|--:|--:|--:|--:|--:|--:|
| Llama-3.2-1B | ShareGPT | 324 | 109.5 | 128.6 | 130.6 | 204.3 | 15%〔計算〕 |
| | Alpaca | 238 | 56.9 | 70.0 | 70.4 | 132.7 | 19%〔計算〕 |
| | LMSYS | 247 | 73.6 | 83.7 | 84.5 | 171.8 | 12%〔計算〕 |
| Llama-3.1-8B | ShareGPT | 356 | 84.9 | 132.8 | 135.2 | 283.3 | 36%〔計算〕 |
| | Alpaca | 250 | 46.8 | 102.9 | 112.8 | 285.1 | 55%〔計算〕 |
| | LMSYS | 258 | 92.4 | 126.7 | 144.3 | 178.6 | 27%〔計算〕 |
| Qwen3-30B-A3B | ShareGPT | 1,054 | 210.8 | 262.2 | 255.2 | 936.2 | 20%〔計算〕 |
| | Alpaca | 726 | 189.0 | 200.0 | 201.8 | 656.2 | 5.5%〔計算〕 |
| | LMSYS | 792 | 189.1 | 224.4 | 229.4 | 800.0 | 16%〔計算〕 |

- 「比 BERT 低」＝(BERT−MLP)/BERT，我自己算的。OUTLETS 本法（多層融合＋draft decoder）再比 MLP 低 1.7–24%〔計算，同表〕。
- 〔判讀〕優勢隨模型而變：8B 最大、1B 與 30B-A3B（MoE、平均輸出更長）較小。BERT 是**微調過的**，所以這不是「凍結 vs 微調」的不公平比較。

### 1.3 層數掃描

| 論文 | 模型（層數） | 掃描結果 | 最好的層 | 最後一層 vs 最好 | 出處 |
|:--|:--|:--|:--|:--|:--|
| TRAIL | Llama3-8B（32） | 第 0 層 MAE 約 92 → 第 10 層約 55（最低）→ 第 23–26 層約 90 → 第 31 層約 79；誤差棒很大 | 10–15，選 11 | 約 79 vs 55（差約 44%）〔讀圖〕 | 〔原文 p.4 Fig.2、p.5〕 |
| EGTP | Qwen2.5-0.5B（24），GSM8K | MAE：第 1 層 85.91、第 8 層 81.29、第 12 層 70.26、第 16 層 70.79、第 24 層 73.93 | 12 | 73.93 vs 70.26（差 5%）；作者因此直接用最後一層 | 〔原文 p.25 Table 7〕 |
| How Much is Left? | 一個模型（原文說「只在一個模型上」） | 早期層接近常數基準，中段大幅下降，上 1/3 穩定在約 80；第 0 層最差 | 上 1/3 一帶；**全部層串接最好** | 〔原文沒給數字〕 | 〔原文 p.12 Fig.4、p.15〕 |
| OUTLETS | — | 沒有掃描；引用前作的理由是「多深度融合」，自己用低／中／高三層 | — | — | 〔原文 p.5〕 |

- 〔判讀〕三篇一致：**不要只用最後一層、也不要用第 0 層**；中間偏前（1/3–1/2 深）是好的起點，多層串接更好。但三篇的目標都是「模型自己還要講多少」，這是模型內部「計畫」的一部分（How-Much-Left 的論點）。**「使用者會不會回來」不是模型自己的行為**，沒有理由預期它同樣被編碼在 hidden state 裡。

### 1.4 SAECache 的 hidden-state 分類器（p.7–8 vs p.19–20 的矛盾）

- arXiv API 查到的最新版仍是 **v1（2026-05-12）**，沒有新版〔文件：arXiv API，2026-10-17〕。
- 原文說「接受後釋出」程式碼〔原文 p.26 Q5〕；本文沒有找到 repo〔未查證：搜尋次數受限，見未查證清單〕。
- 所以矛盾**無法釐清**：p.7–8 說讀「服務中 LLM 最後一層、最後一個輸入 token 的 hidden state」；p.19–20 的特徵重要度列 `open_ended_score`、`response_length` 等手工特徵（E06 卡已記）。〔判讀〕不能把它當「hidden state 預測續聊有效」的證據。

### 1.5 預測續聊、滿意度、回來的「中間狀態」前作

- **查不到**。範圍：WebSearch 2 次（hidden state／probe × user satisfaction／conversation continuation；next-turn continuation × features ablation）；結果只有用文字分類器的工作（例如 Chai 的 GPT-2 reward model 預測續聊，§4）與預測「模型自己行為」的 probe。
- 〔判讀〕F4（中間狀態）在回來時間上**沒有前例**；它是新問題，同時也代表沒有任何證據說它會比 F3（文字嵌入）好。

---

## 2. G2：LPC 的消融與 Copilot 的特徵

### 2.1 LPC（NeurIPS'25 proceedings）p.20–23

**LPC_turns vs 完整版**（Table 3；LPC_turns＝依輪數分箱、取訓練資料的多數標籤）〔原文 p.21〕

| 資料 | LPC_turns MCC | LPC MCC | LPC_turns F1-macro | LPC F1-macro |
|:--|--:|--:|--:|--:|
| LMSys | 0.3510 | 0.3863 | 0.6755 | 0.6923 |
| ShareGPT | −0.0012 | 0.2815 | 0.4165 | 0.6407 |
| Chatbot-Arena | 0.0775 | 0.3689 | 0.4741 | 0.6837 |

**加入模型回應**（Table 4）：MCC Arena 0.3370（加回應）vs 0.3485（只有 prompt）；LMSys 0.3843 vs 0.3797；ShareGPT 0.2779 vs 0.2777。原文：「沒有顯著、一致的改善，甚至略差」〔原文 p.21–22〕。

**prompt 視窗 5 vs 10**（Table 5）：MCC Arena 0.3485／0.3416；LMSys 0.3797／0.3727；ShareGPT 0.2777／0.2496〔原文 p.22〕。

**unified vs specialized**（Table 6）：MCC Arena 0.3357／0.3485；LMSys 0.3802／0.3797；ShareGPT **0.1271**／0.2777〔原文 p.22〕。

**和 oracle 比**（Fig.9；oracle＝完美知道會不會續聊）：LRU→oracle 的命中率差距被補回 LMSys 22–30%、ShareGPT 28–64%、Arena 25–28%；圖說寫 22–66%〔原文 p.23〕。原文結論：要再逼近 oracle「可能需要新的資料維度，例如使用者的存取型態」〔原文 p.23〕。

**開銷**（Table 2，LMSys、40 GB）：把預測器掛在 LRU 上，SM Active 41.74→42.04%（+0.72%）；hit ratio LPC 46.50% vs LRU 40.90%〔原文 p.20〕。

- **一個不一致**：Table 3 的 LPC MCC（0.3863／0.2815／0.3689）和 Table 4–6「Prompts Only／Specialized」的數字（0.3797／0.2777／0.3485）不同；原文沒有解釋〔原文 p.21–22〕。〔判讀〕可能是不同次訓練，差距 0.004–0.02，比 LMSys 上「文字 vs 輪數」的差距（0.035）小不了多少。
- 〔判讀〕「文字比輪數多出多少」**依資料而定**：LMSys 只多 0.035 MCC（輪數本身就有 0.351）；ShareGPT、Arena 上輪數完全沒用、文字是唯一訊號。LPC 的資料平均輸入只有 36–113 token、94–99% 的對話 ≤5 輪（E06 卡），和我們的長 context agent trace 差很遠。

### 2.2 Copilot 2026（arXiv 2608.00101v1）§9

- **特徵**（原文逐字整理）：每個結束的回合摘成三組特徵——
  - 回合層級：時長、LLM 呼叫數、token 數、工具時間、失敗、成功率；
  - session 層級：輪次（turn index）、已經過時間、上一次閒置、到目前為止的平均閒置；
  - 時間：例如 `is_weekday`〔原文 p.17〕。
  - 原文說是「O(10) 個表格特徵」〔原文 p.17〕。**沒有任何內容特徵**（trace 本身也沒有內容，LIT_AND_TRACES A.2）。
- **特徵重要度**（Fig.33a，重要度種類——gain 或 split——原文沒寫〔未查證〕）〔讀圖，原文 p.18〕：

| 特徵 | 重要度（讀圖） | 類別〔判讀〕 |
|:--|--:|:--|
| Avg Idle Time So Far | ≈28.7 | 使用者／session 歷史 |
| Turn Index | ≈25.6 | 中繼資料 |
| Prev. Idle Time | ≈11.6 | 使用者／session 歷史 |
| LLM Success Rate | ≈10.8 | 回合內容的代理（失敗） |
| Turn Duration | ≈8.5 | 中繼資料 |
| LLM Calls | ≈7.6 | 中繼資料 |
| Prompt Tokens | ≈2.0 | 中繼資料 |
| User Cancelled Turn | ≈1.9 | 回合結果 |
| Completion Tokens | ≈1.5 | 中繼資料 |
| Others | ≈1.0 | （含時間特徵〔判讀〕） |
| Tool Failure Rate | ≈1.0 | 回合結果 |

- **消融**：沒有逐特徵或逐家族的消融。只有：(a)「只看上一次閒置」的規則 AUC 0.58 vs 完整 0.73 vs 永遠猜正 0.5；(b)「掃過其他模型（例如決策樹），AUC 只有些微變化，表示沒有更豐富的輸入訊號，加模型複雜度不太會有幫助」〔原文 p.17〕。
- **隨時間的表現**（Fig.33b）：門檻從 30 s 到 30 min，accuracy 與 F1 從約 80–88% 降到 25–43%；「正確抓到的閒置時間比例」穩定在 86–90%〔原文 p.17、讀圖 p.18〕。
- 〔判讀〕Copilot 的資料裡，**session 歷史（已閒置平均、上一次閒置）約占 40%、輪次約 26%，token 數不到 4%**。這和 LPC 在 LMSys 上「輪數就很強」一致，也和我們 9 月 B12「static 特徵最好」的方向一致。原文自己說瓶頸在輸入訊號、不在模型。

---

## 3. G3：頻率特徵與頻率准入

### 3.1 LRB（NSDI'20）特徵累加消融（Fig.5，good decision ratio，越高越好）〔讀圖，原文 PDF p.7（proc. p.534）〕

s＝靜態（物件大小、類型）；d＝deltas；e＝EDC；full＝靜態＋32 deltas＋10 EDC。

| trace／容量（GB） | s | s+1d | s+1d+1e | s+2d+2e | s+8d+8e | full |
|:--|--:|--:|--:|--:|--:|--:|
| Wiki 64 | 0.36 | 0.37 | 0.43 | 0.67 | 0.79 | 0.80 |
| Wiki 256 | 0.30 | 0.43 | 0.43 | 0.60 | 0.76 | 0.75 |
| CDN-A1 64 | 0.49 | 0.63 | 0.81 | 0.81 | 0.83 | 0.84 |
| CDN-A1 128 | 0.50 | 0.70 | 0.74 | 0.78 | 0.79 | 0.80 |
| CDN-B1 128 | 0.51 | 0.52 | 0.59 | 0.74 | 0.83 | 0.87 |
| CDN-B1 512 | 0.54 | 0.76 | 0.78 | 0.85 | 0.86 | 0.86 |

- 原文：「加更多特徵會改善，但報酬遞減」；其他三條 trace 與其他容量「相似、沒畫」〔原文 p.7〕。32 deltas、10 EDC 是因為「都已過報酬遞減點」〔原文 p.7〕。
- 其他消融（只有文字結論，沒有數字）：GBM 在所有 trace／容量都最好（vs LogReg、LinReg、SVM、2 層 16 節點 NN，Fig.6）；log(time-to-next) 比「上次到下次」、二元分類、未取 log 都好；L2 是 LightGBM 8 種 loss 裡最好的；訓練集 128K 之後報酬遞減（試到 512K）；64 個候選之後報酬遞減〔原文 p.7–8〕。
- 〔判讀〕**CDN 上頻率特徵貢獻很大**（+0.26～+0.45），和我們 9 月在 Mooncake block 級量到的相反（B12：只用 static 最好）。這正是 Fancy-eviction 說的「前綴快取不同」（§3.2）。

### 3.2 頻率在 LLM KV trace 上的比較

| 來源 | 比較了什麼 | 結果 | 出處 |
|:--|:--|:--|:--|
| **Fancy-eviction**（arXiv 2609.28870v2；FreeInference、Chutes，block 級模擬） | 14 種線上演算法，含 LFU、W-TinyLFU（頻率排序）、S3-FIFO 等快速降級（等於「第二次才進主佇列」的頻率准入） | 頻率派「completely collapse」；ARC、LHD、LeCaR、3LCache、LRB 等≈LRU〔原文 p.4〕。原因：重用間隔和頻率幾乎無關（Fig.5a），頻率只追蹤 session 進度〔原文 p.8〕 | 〔原文 p.4、p.8〕 |
| 同上 | 早期頻率當「會不會再被用」的訊號 | 拿到第 2 次命中後，再被重用的機率 69.4%→88.7%（Fig.5c）；「頻率適合當准入、不適合排序」〔原文 p.8〕 | 〔原文 p.8〕 |
| 同上 | 快速降級（頻率准入）何時有用 | Qwen To-B（Bailian）24 GiB：S3-FIFO 0.491 vs LRU 0.403；但只在「單輪 prompt 量極大、足以洗掉快取」時才該開〔原文 p.10〕；在 Qwen To-C 上 LIRS 依容量從 +4.2 變 −10.1 點、W-TinyLFU 隨容量非單調（E05 卡，原文 p.17） | 〔原文 p.10、p.17〕 |
| **WhereKVLive**（arXiv 2609.16215v1；**合成負載**、模擬、batch 1） | recency vs reuse-frequency vs EWMA 預測（以 PCIe 搬移量計） | chat：recency 27.38 GB vs reuse-frequency 62.99 GB（2.30×）；agent：reuse-frequency 49.36 GB 最好，EWMA 53.16 GB 第二；doc QA：reuse-frequency 48.23 vs EWMA 48.45〔原文 p.5–6〕。作者自承合成負載的重用結構「依構造有利於對應的策略」〔原文 p.8〕 | 〔原文 p.1、p.5–6、p.8〕 |
| **SGLang HiCache**（`write_through_selective`） | 寫入 host 的門檻 | `write_through_threshold = 1 if write_policy == "write_through" else 2`〔程式碼：`python/sglang/srt/mem_cache/unified_radix_cache.py` L539–541，commit `a126e8d6`〕；官方文件與 LMSYS 部落格**沒有**依寫入策略分開的評估〔文件〕 | 〔程式碼〕〔文件〕 |
| **Dynamo KVBM**（磁碟下放過濾） | CPU→磁碟的准入 | 「只在 frequency ≥ 2 時才從 CPU 下放到磁碟」；frequency 初值 1、每次命中加倍、每個衰減步減 1；開磁碟下放時預設開啟，理由是延長 SSD 壽命；文件**沒有**開／關過濾的評估數字〔文件：docs.nvidia.com/dynamo/v1.0.0/components/kvbm/kvbm-guide〕 | 〔文件〕 |
| 2-hit／3-hit、LRU-K 准入在**長 context LLM KV trace** 上的端到端比較 | — | **查不到**。LRU-K 准入只看到 HBF 推薦系統的分析模型（A_HBF_LRUK 卡，不是 LLM 對話） | 〔未查證〕 |

- 〔判讀〕頻率有兩個角色要分開：**排序**（誰先降級）在 LLM KV 上沒用、甚至有害；**准入**（要不要寫到下一層）在單輪流量大時有用，而且是 SGLang、Dynamo 的實際預設。我們的 F2「過去回來幾次」是 session 級的頻率，比較接近「准入」的角色（有沒有回來過＝是不是多輪 session），不是排序。

---

## 4. G4：別的領域的「使用者會不會回到對話／多久回來」

| 論文 | 任務、資料 | 特徵家族 | 各家族的貢獻 | 出處 |
|:--|:--|:--|:--|:--|
| **Backstrom, Kleinberg, Lee, Danescu-Niculescu-Mizil**（WSDM'13；arXiv 1304.4602） | thread re-entry：給定前 5（或 9）則留言與第一位留言者，他之後還會不會再留言？Facebook（百萬級）、Wikipedia talk page | 連結（社交網路）、到達型態（誰在第幾則留言，`id_code`）、時間（前 i 則花多久）、原文詞（text regression 選的 30–50 個詞）、雜項（長度、問號、讚） | AUC（交叉驗證）：**文字基準 0.520／0.525／0.494**；全部特徵 **0.808**（FB 前 5）／**0.855**（FB 前 9）／**0.644**（Wiki）；正例比例基準 0.500。最重要的是**最近幾位留言者的身分**（到達型態），其次是**最近兩則留言的間隔**（越久越可能結束）；細部特徵分析「因篇幅省略」 | 〔原文 p.6 Table 2、p.8 §6、Table 5〕 |
| 同上（長度預測，參考） | thread 會不會到 8 則 | 同上 | 逐步特徵選擇：時間特徵 `time[5]`（前 5 則花多久）最關鍵；拿掉它 AUC 0.729→0.588 | 〔原文 p.7〕 |
| **Zeng, Li, Wang, Wong**（ACL'19；P19-1270） | first re-entry：Twitter（7,500 段對話、10,122 位使用者）、Reddit politics 2008（29,477 段、13,134 位）；隨機 80/10/10 依對話切 | 對話內容（BiLSTM 編碼每輪）、使用者歷史（使用者過去發的**文字**）、兩者的互動（concat／attention／memory／bi-attention）、輪次中繼資料 | AUC Twitter／Reddit：**只看使用者過去回來的比例（HISTORY）50.1／50.7**（≈隨機）；Backstrom 式手工特徵（CCCT）57.7／59.9；**只看內容** BiLSTM 60.4／64.1；**內容＋使用者歷史文字**（bi-attention）62.7／67.1；但簡單串接反而掉到 51.0／50.1。拿掉輪次中繼資料：F1 61.1→60.4（Twitter）、45.4→44.3（Reddit） | 〔原文 p.5 Table 1、p.7 Table 2、p.8 Table 3〕 |
| **Copilot 2026**（見 §2.2） | 回合邊界後的使用者閒置時間（>60 s） | 回合、session、時間 | session 歷史約 40%、輪次約 26%、token 數 <4%（重要度，讀圖）；「只看上一次閒置」AUC 0.58 vs 全部 0.73 | 〔原文 p.17–18〕 |
| **Grob et al.**（arXiv 1807.04098v1，ASOS 電商網站） | 使用者多久回來（下一個 session），預測視窗內不回來＝右設限；38,716／9,680 位使用者，63.6% 在視窗內回來 | 時間（session 時間、時長）＋行為（看了幾張圖、幾段影片）；**沒有內容文字** | 「只用距上次 session 多久」的基準：不回來 AUC 0.743、concordance 0.500；Cox（彙總特徵＋缺席時間）0.793／0.816；只用回來者訓練的 RNN 0.763／0.706；RNN 存活模型（原始 session 序列，處理設限）0.796／0.739。以已缺席時間為條件（CPHA、RNNSMA）主要提高「不回來」的 recall（0.246→0.461、0.538→0.605） | 〔原文 p.11–12 Table 1〕 |
| **Chai Research**（Irvine et al.，arXiv 2303.06135v2） | 聊天機器人：每則回應之後，對話會不會繼續、使用者會不會重抽 | 只用文字（GPT-2 small–XL、RoBERTa 當 reward model 讀對話） | 只報下游效果（用 reward model 選回應後的平均對話長度、留存率）：第 30 天，預測「會繼續」讓留存 +12.1±4.4%，預測「不重抽」+24.7±4.5%，兩者都要 +30.3±4.5%〔原文 p.8〕；全文檢索沒有預測準確度（accuracy／AUC），也沒有特徵消融〔原文，檢索〕 | 〔原文 p.8〕 |

- 〔判讀〕**內容 vs 使用者歷史誰重要，沒有一致答案，取決於「歷史」怎麼定義、平台怎麼運作**：
  - Backstrom 的資料：**結構與時間**（誰剛留言、間隔多久）壓倒文字（文字≈隨機）。
  - Zeng 的資料：**內容**最重要；使用者過去的**回來比例**≈隨機，但使用者過去的**文字**和當前內容對齊後有小幅幫助（+2.3／+3.0 AUC）。
  - Copilot（沒有內容）：**session 歷史＋輪次**占大部分。
  - Grob（電商，沒有內容）：**光是「已經缺席多久」就有 0.743 的 AUC**，彙總的歷史特徵只再加 0.05，讓 RNN 自己從原始序列學特徵幾乎沒有再加（0.796 vs 0.793）。
  - 共同點：**時間型態**（最近間隔、已閒置多久）在四篇有時間特徵的工作裡都是最強或前兩名；**單純的歷史頻率／比例**在 Zeng 裡沒用。這和我們 F0（閒置時間）、F2（過去間隔）的排序假設一致，也提醒 F2 不要只放「回來次數」。
  - 數字不能跨篇比（任務、資料、指標都不同）；只能比「同一篇內」各家族的相對大小。

---

## 5. G5：多輪情境下 query-aware 與 query-agnostic 的 KV 選取

### 5.1 SCBench（ICLR'25；arXiv 2412.10319v2）的發現

| 發現 | 數字 | 出處 |
|:--|:--|:--|
| sub-O(n) 記憶體的方法（丟 KV：SnapKV、PyramidKV、StreamingLLM）第一個 query 表現好，**後續輪次掉很多**；保留 O(n) 記憶體、只在解碼時稀疏讀（RetrievalAttention、KIVI）或只在 prefill 稀疏（A-shape、Tri-shape、MInference）的較穩 | — | 〔原文 p.4〕 |
| 原因：同一輪內重要的 KV 很穩定，**不同 query 之間差很多**（Retr.KV 任務的注意力圖） | Fig.6 | 〔原文 p.9〕 |
| 多輪模式平均分（Llama-3.1-8B，壓縮率 τ） | 全量 48.7；CacheBlend（τ＝1）49.3；MInference（1/32）42.5；KIVI（1/8）32.0；RetrievalAttention（1/32）29.4；Quest（1/32）22.1；SnapKV（1/32）20.9；PyramidKV（1/32）21.0；StreamingLLM（1/32）15.5 | 〔原文 p.9 Table 4〕 |
| query-aware 方法「有 query／沒有 query」（Llama-3.1-8B；Retr.String、Retr.Semantic、Global、Multi-task） | SnapKV 0.0／0.0、19.0／9.7、17.9／14.6、5.1／0.0；Tri-shape 12.1／7.8、31.4／25.7、31.1／45.6、28.0／24.6；MInference 28.1／28.9、40.4／35.6、35.4／50.1、28.3／30.9。原文：SnapKV 與 Tri-shape 沒有 query 時「難以維持準確度」，MInference（動態稀疏）泛化較好 | 〔原文 p.10 Table 5〕 |
| 建議的方向 | 「CPU-GPU 協作：完整的 O(n) 記憶體放在 CPU RAM，動態把相關的 KV 載到 GPU 做 sub-O(n) 解碼」 | 〔原文 p.9〕 |

- Table 5 裡 Tri-shape、MInference 在 Global 任務上「沒有 query」反而比較高（45.6 vs 31.1、50.1 vs 35.4），原文沒有解釋〔原文 p.10〕。

### 5.2 各方法：query 在什麼時候用、在多輪或 offload 情境的結果

| 方法 | query 怎麼用 | KV 放哪 | 多輪／多 query 的結果 | 出處 |
|:--|:--|:--|:--|:--|
| **Quest**（ICML'24；2406.10774v2） | 每個解碼步用當下的 query 和每頁 key 的逐通道最小／最大值算注意力上界，只讀 top-k 頁；原文觀察「關鍵 token 隨 query 變很多」 | 全部在 GPU（不省記憶體） | 自己沒做多輪；SCBench 多輪 22.1（1/32）vs 全量 48.7 | 〔原文 p.1–2、p.4〕；SCBench〔原文 p.9〕 |
| **InfiniGen**（OSDI'24；2406.19707v1） | 在第 i−1 層用近似的 query 預演第 i 層的注意力，挑出關鍵 KV 預取 | 卸載到 CPU | 全文檢索沒有多輪評估 | 〔原文 p.1〕 |
| **ShadowKV**（2410.21465v3） | 低秩的 pre-RoPE key 留 GPU，value 卸載到 CPU，每步依 query 稀疏讀回 | GPU＋CPU（O(n)） | 多輪 NIAH：SnapKV「從第二輪起顯著下降」，ShadowKV「維持準確度」（只有圖，Fig.7） | 〔原文 p.2、p.8〕 |
| **RetroInfer**（VLDB'26；2505.02922v3） | 向量索引（wave index）依當下生成步找重要 token | KV 在 CPU，GPU 有 wave buffer | 全文檢索只在引言提到多輪，沒有多輪評估 | 〔原文 p.1〕 |
| **ArkVale**（NeurIPS'24） | 頁面寫滿就非同步備份到 CPU、用 key 的包圍體當摘要；每個解碼步用 query 對摘要估分，**把之前逐出、現在變重要的頁召回** | GPU＋CPU 備份 | 動機是單一請求內的重要度漂移（Fig.4）；多輪只出現在動機 | 〔原文 p.1–2、p.5〕 |
| **KVzip**（NeurIPS'25 oral；2505.23416v2） | **query-agnostic**：用 LLM 重建原文的能力給 KV 打分，只壓一次、給之後所有 query 用 | GPU | SQuAD：SnapKV 用第一個問題壓縮後重用，準確度明顯低於每個問題各自 prefill（Fig.2，只有圖）；摘要：query-aware 方法在多 query 下「即使 90% 預算也掉分」；KVzip 在 30% 比例下近乎無損 | 〔原文 p.1、p.3、p.6–7〕 |
| **EpiCache**（ICML'26；2509.17396v4） | 對話依主題分群成 episode，每個 episode 各自壓一份 KV；**每個新 query 先做嵌入、找最近的 episode 中心、取回那一份 KV 來答**；之後的新 KV 寫回該 episode | 各 episode 的 KV 放 CPU（Fig.3） | 「預先知道未來的問題」當 patched prompt 是上界，但不可行（壓縮時不知道問題、每題都要重 prefill）；整體準確度最多 +30%、4–6 倍壓縮下接近全量 | 〔原文 p.1、p.3–5〕 |
| **Cocktail**（DATE'25；2503.23294v1） | 用 Contriever 算 query 與每個 context chunk 的相似度：高→FP16、中→INT4、低→INT2 | GPU；單一請求（query 在 prefill 時已知） | 沒有多輪 | 〔原文 p.3〕 |
| **IMPRESS**（FAST'25） | 重複的 prefix 來了，先從磁碟只載入少數 probe head 的 key（原文的例子用 3 個），和**新請求的 query** 算注意力，只載入重要 token 的 KV；原文觀察「同一個 prefix 的重要 token 依 query 而變，預先記下的集合會漏」 | GPU／CPU／SSD 三層 | few-shot 共用 prefix，不是多輪閒置；有損（保留 50%→5%，A_IMPRESS 卡） | 〔原文 p.5、p.7〕 |

### 5.3 回答：query 能不能用在讀取時？前作有沒有做？

- **能，而且已經有人做**：
  - **「還原哪些」**：IMPRESS（慢層→GPU，token 級，prefix 被重用時依新 query 選）、EpiCache（CPU→GPU，episode 級，每個新 query 選一份）；ArkVale、ShadowKV、InfiniGen、RetroInfer 是**每個解碼步**依 query 從 CPU 召回〔原文，見 §5.2〕。
  - **「用什麼精度」**：Cocktail 依 query 與 chunk 的相似度選位寬，但在 GPU 上、單一請求、不是從慢層還原〔原文 p.3〕。
  - **兩者合起來**（session 閒置後回來、依新 query 同時決定「從哪一層還原哪些 block、用什麼精度、哪些重算」）：**查不到**。範圍：本節 10 篇＋SCBench 的方法分類（Table 1）＋ E05、E06、A_IMPRESS 卡；網路搜尋因用量上限只做了少數幾次（§7）。
- **反面證據也很清楚**：只依某一個 query 決定「丟掉」的方法，下一輪就壞（SCBench、KVzip、ShadowKV 的 SnapKV 對照、EpiCache 的動機）。能撐多輪的都**保留完整的一份**（O(n)，放 CPU），只在讀取時選擇性載入。
- 〔判讀〕對我們：query 是**讀取時才有**的訊號，用途是「回來時先載哪些、哪些用低精度、哪些重算」，不能拿來決定降級誰。前作已經有「依 query 選擇性還原」，新穎性只可能在「和層級成本（CPU／SSD／重算）、精度一起決定，並用真實的回來間隔評估」。而且它是**有損**的，和 CLAUDE.md 定義的「CPU／SSD 是無損搬移」不同軌；如果要做，屬於第二階段的精度線（要量 ε），而且慢層必須保留完整副本。

---

## 6. 對 02_predictor_inventory §3 初步決定的含意〔判讀〕

以下全部是判讀；依據的事實在 §1–§5。

1. **「哪樣最高」仍然要自己量，但前作給了合理的先驗排序**：
   - 時間型態（已閒置多久、過去間隔）在四篇有時間特徵的工作裡都是最強或前兩名（Backstrom、Copilot、Grob；Zeng 的 CCCT 也含時間）→ **F0 與 F2 的時間部分很可能拿走大部分可預測性**。
   - 文字的增益**依資料而變**：LPC 在 LMSys 上只比輪數多 0.035 MCC，在 ShareGPT／Arena 上是唯一訊號；Backstrom 的文字≈隨機。→ F3 在 WildChat 上可能有用，在 agent trace 上無從量起（沒有文字）。
   - 中間狀態**只在「模型自己的行為」（輸出長度）上有證據**，比文字好 5–55%；在「使用者會不會回來」上**零前例**，而 LPC 發現連模型回應的文字都沒幫助。→ F4 的先驗應該是「≤ F3，除非意外」。

2. **建議調整 F0–F4 的消融設計**：
   - **(a) 把 F2 拆成兩塊**：F2a 時間型態（過去間隔、到目前為止的平均閒置、上一次閒置）、F2b 次數／頻率（回來過幾次、是不是多輪）。理由：Zeng 的「回來比例」≈隨機，Copilot 的「平均閒置」第一名；混在一起看不出是哪一塊。
   - **(b) 除了累加，也做「單獨加在 F0 上」**：F0＋F1、F0＋F2、F0＋F3、F0＋F4 各一次，再做全部與「全部減一家族」。累加的順序會影響歸因（LRB 的 Fig.5 就是固定順序）；而且「中繼 vs 文字 vs 中間狀態」**在同一任務上正面比**是前作沒有的，這正是可以補的空白。
   - **(c) F4 先做小規模探針再決定要不要大量抽 hidden state**：一個模型、在 WildChat 子集上，掃 {約 1/4、1/3、1/2、2/3、最後一層}＋多層串接（TRAIL、EGTP、How-Much-Left 都顯示中間層較好、串接更好）；pooling 至少比 mean 與 last-token（EGTP 顯示同一組 hidden state，pooling 不同 MAE 差 1.5 倍）。只有在 F0＋F3 明顯優於 F0＋F2 時才擴大，否則 F4 的結論就是「在這個任務上 F3 已經夠」，也是一個答案。
   - **(d) F3 與 F4 要公平**：同一個 probe（線性＋2 層 MLP）、同一組 pooling 選項；文字端除了凍結 e5-small，至少加一個微調過的小編碼器（OUTLETS 的 BERT 基準是微調的）。
   - **(e) 加一個 LPC_turns 式的超便宜基準**：依輪數分箱取多數標籤，當 F1 的下限。
   - **(f) 每格至少 3 個 seed 並報信賴區間**：LPC 自己的 Table 3 與 Table 4–6 同一設定差 0.004–0.02 MCC，和 LMSys 上「文字 vs 輪數」的 0.035 同量級；LTR 顯示連「另一個 seed 的實際長度」排序 τ 也只有 0.74–0.84，目標本身有雜訊天花板。
   - **(g) 主指標維持決策 regret**：EGTP 的 TRAIL MAE 比 LTR-C 好，但端到端 padding 比 LTR-C 差（§1.1），「預測較準」不保證「決策較好」。

3. **頻率（2-hit／3-hit）**：照原決定，當**准入規則**放在對手裡，不當排序特徵（Fancy-eviction：LFU、W-TinyLFU 當排序完全失敗；第 2 次命中當准入有用）。具體建議把「LRU＋慢層 2-hit 准入」（Dynamo KVBM 的預設）與「LRU＋host 2-hit 寫入」（SGLang `write_through_selective`）都列成對手；兩者都查不到公開評估（§3.2），所以這組對照本身就有資訊量。

4. **query**：照原決定不放進 F0–F4。若要做「回來時依 query 選擇性還原」，那是另一條有損的研究線（§5.3），前作（IMPRESS、EpiCache、Cocktail）已經各做了一半，要寫清楚差別。

5. **資料範圍**：F3／F4 只能在 WildChat 上做（唯一有文字＋逐輪時間），結論只對聊天成立；FreeInference、Copilot 只能做 F0–F2。Copilot 原文自己說瓶頸在「沒有更豐富的輸入訊號」，所以 agent trace 上 F0–F2 的天花板可能就在 AUC 0.7 左右（Copilot 0.73），這也是 P3-3 oracle 判定時要考慮的量級。

---

## 7. 未查證清單

- **SAECache** 的程式碼與新版本：arXiv 仍只有 v1；repo 沒找到（網路搜尋受用量上限，只查了 arXiv API）。hidden-state 分類器的實際輸入仍不明。
- **OUTLETS** 的 MLP 基準用哪一層；**EGTP** 的 LMSYS 列（GPT-4、Claude-2 的回應）用哪個模型的 hidden state、EGTP 裡的 TRAIL 基準在 Qwen 列讀哪個模型（附錄只寫「Llama 的內部嵌入」）。
- **How Much is Left?** 的單層掃描是三個模型中的哪一個（原文只說「只在一個模型上」）。
- **Copilot** Fig.33a 的重要度種類（gain 或 split）；表中數值是〔讀圖〕。**LRB** Fig.5 數值是〔讀圖〕，其餘三條 trace 原文沒畫。**TRAIL** Fig.2–3 數值是〔讀圖〕。
- **ShadowKV** Fig.7、**KVzip** Fig.2 只有圖，沒有讀數值。
- **μ-Serve** 原文沒讀；**ELIS** 只讀到搜尋摘要；**S3**、**SSJF** 的描述是 EGTP 附錄的二手轉述。
- **用服務模型 hidden state 預測續聊、滿意度、回來時間**：查不到（WebSearch 2 次，見 §1.5）。
- **2-hit／3-hit／LRU-K 准入在長 context LLM KV trace 上的端到端比較**、**SGLang／Dynamo 准入規則的評估**：查不到（文件與程式碼都沒有）。
- **Kapoor et al.（KDD'14，hazard 模型預測使用者回來時間）** 的特徵清單與消融：PDF 沒取得，只看到搜尋摘要說用了「行為與時間特徵」〔摘要〕。
- **聊天機器人使用者回來／流失、且有特徵消融的工作**：除了 Chai（只有下游留存、沒有消融）之外查不到；搜尋次數受限，不代表不存在。
- **SCBench** Table 5 的 Global 欄「沒有 query 反而較高」原文沒有解釋。
- **InfiniGen、RetroInfer** 只讀了 p.1 摘要並全文檢索「multi-turn」；沒有讀方法細節。
- **「session 回來時，依新 query 同時決定還原哪些 block、用什麼精度、哪些重算」的前作**：查不到（範圍見 §5.3）。

---

## 附錄：讀了什麼

| 代號 | 版本 | URL | 讀的範圍 |
|:--|:--|:--|:--|
| LPC | NeurIPS'25 proceedings | https://proceedings.neurips.cc/paper_files/paper/2025/file/414f642a1ea9350006669774cba9bcd4-Paper-Conference.pdf | p.20–23 |
| Copilot 2026 | arXiv 2608.00101v1 | https://arxiv.org/abs/2608.00101 | p.16–18（含 Fig.33 讀圖） |
| LRB | NSDI'20 | https://www.usenix.org/system/files/nsdi20-paper-song.pdf | PDF p.6–8（含 Fig.5 讀圖） |
| Fancy-eviction | arXiv 2609.28870v2 | https://arxiv.org/abs/2609.28870 | p.4、p.8、p.10（檢索） |
| WhereKVLive | arXiv 2609.16215v1 | https://arxiv.org/abs/2609.16215 | p.1、p.4–6、p.8 |
| SGLang | commit `a126e8d6` | https://github.com/sgl-project/sglang/blob/a126e8d640c707f855e8da31f906035883f4ff70/python/sglang/srt/mem_cache/unified_radix_cache.py | L530–548 |
| Dynamo KVBM | 文件 v1.0.0 | https://docs.nvidia.com/dynamo/v1.0.0/components/kvbm/kvbm-guide | 磁碟下放過濾一節 |
| TRAIL | arXiv 2410.01035v1（ICLR'25） | https://arxiv.org/abs/2410.01035 | p.1–7（含 Fig.2–3 讀圖） |
| EGTP | arXiv 2602.11812v2（ICLR'26） | https://arxiv.org/abs/2602.11812 | p.1、p.6–9、p.25–27 |
| OUTLETS | arXiv 2609.01068v2（EMNLP'26） | https://arxiv.org/abs/2609.01068 | p.1、p.5–7、附錄 B |
| How Much is Left? | arXiv 2607.05316v1 | https://arxiv.org/abs/2607.05316 | p.1、p.6、p.12、p.15 |
| LTR | arXiv 2408.15792v1（NeurIPS'24） | https://arxiv.org/abs/2408.15792 | p.8 |
| Zeng et al. | ACL'19（P19-1270） | https://aclanthology.org/P19-1270.pdf | p.5–8 |
| Backstrom et al. | WSDM'13；arXiv 1304.4602 | https://arxiv.org/abs/1304.4602 | p.6–8 |
| Grob et al. | arXiv 1807.04098v1 | https://arxiv.org/abs/1807.04098 | p.11–12 |
| Chai Research | arXiv 2303.06135v2 | https://arxiv.org/abs/2303.06135 | p.8（其餘只檢索） |
| SCBench | arXiv 2412.10319v2（ICLR'25） | https://arxiv.org/abs/2412.10319 | p.1、p.4、p.9–10 |
| Quest | arXiv 2406.10774v2（ICML'24） | https://arxiv.org/abs/2406.10774 | p.1–2、p.4 |
| InfiniGen | arXiv 2406.19707v1（OSDI'24） | https://arxiv.org/abs/2406.19707 | p.1（其餘只檢索） |
| ShadowKV | arXiv 2410.21465v3 | https://arxiv.org/abs/2410.21465 | p.2、p.8 |
| RetroInfer | arXiv 2505.02922v3（VLDB'26） | https://arxiv.org/abs/2505.02922 | p.1（其餘只檢索） |
| ArkVale | NeurIPS'24 proceedings | https://proceedings.neurips.cc/paper_files/paper/2024/file/cd4b49379efac6e84186a3ffce108c37-Paper-Conference.pdf | p.1–5（檢索為主） |
| KVzip | arXiv 2505.23416v2（NeurIPS'25） | https://arxiv.org/abs/2505.23416 | p.1、p.3、p.6–7 |
| EpiCache | arXiv 2509.17396v4（ICML'26） | https://arxiv.org/abs/2509.17396 | p.1、p.3–5 |
| Cocktail | arXiv 2503.23294v1（DATE'25） | https://arxiv.org/abs/2503.23294 | p.1、p.3 |
| IMPRESS | FAST'25 | https://www.usenix.org/system/files/fast25-chen-weijian-impress.pdf | p.2、p.5、p.7（另引 A_IMPRESS 卡） |
| SAECache | arXiv 2605.18825v1 | https://arxiv.org/abs/2605.18825 | 只用 arXiv API 確認版本；內容引 E06 卡 |
