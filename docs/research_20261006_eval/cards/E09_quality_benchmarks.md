# E09　長 context 品質 benchmark 評測卡

> 抽取者：子 agent E09（2026-10-06 查證，2026-10-07 寫成）。狀態：**抽取完成；V09 已獨立複核（2026-10-07），見檔尾〈複核紀錄〉**。
> 規則見 `../README.md`。每格附出處與證據等級；頁碼一律是 PDF 頁（`pdftotext -layout` 的分頁）。

## 範圍

長 context（16K–512K）LLM 推論的 KV 分層研究要量品質 ε（有損精度層 FP8/INT4、丟棄後重算）。本檔整理品質 benchmark：每個是什麼、怎麼被用、參數（長度、截斷、模板、生成長度、解碼、評分）、原文的設計理由、陷阱。

* 完整卡（讀論文全文＋官方程式碼）：LongBench v1／LongBench-E、LongBench v2、RULER、Needle-in-a-Haystack、InfiniteBench、SCBench、HELMET。
* 短卡：L-Eval、LooGLE、BABILong、LongProc、LongMemEval、NoLiMa、PG-19／WikiText 困惑度、長輸出推理題（GSM8K／AIME／MATH-500）。
* 另有：選擇準則表、「只量時間可用合成 token」原則的出處查證、實作差異表、使用者文件核對、對 PoC 的建議〔判讀〕、未查證清單。

## 來源清單（查證日 2026-10-06）

| 名稱 | 讀的版本 | 程式碼（commit，日期） | 資料 |
|:--|:--|:--|:--|
| LongBench v1／-E | arXiv 2308.14508v2（2024-06-19）全文；ACL 2024 Long（`2024.acl-long.172`，README 的 bibtex） | `THUDM/LongBench@2e00731f`（2025-01-15）：`LongBench/pred.py`、`eval.py`、`metrics.py`、`config/*.json`、`task.md`、`README.md` | HF `THUDM/LongBench`（現 `zai-org/LongBench`） |
| LongBench v2 | arXiv 2412.15204v2（2025-01-03）全文；ACL 2025 Long（https://aclanthology.org/2025.acl-long.183/） | 同 repo 根目錄：`pred.py`、`result.py`、`config/model2maxlen.json`、`prompts/` | HF `zai-org/LongBench-v2` |
| RULER | arXiv 2404.06654v3（2024-08-06）全文；COLM 2024（PDF 頁首） | `NVIDIA/RULER@c3f5e3b4`（2026-07-22）：`scripts/*`；另看 `rulerv1-ns` 分支 README（`e8bbff67`） | 腳本生成 |
| NIAH | 無論文；讀 repo | v1：`gkamradt/LLMTest_NeedleInAHaystack@7b90d285`（2024-04-12）；v2：`@021385d6`（2026-06-08） | repo 內 Paul Graham essays |
| InfiniteBench（∞Bench） | arXiv 2402.13718v3（2024-02-24）全文；ACL 2024 Long（https://aclanthology.org/2024.acl-long.814/） | `OpenBMB/InfiniteBench@51d9b37b`（2024-09-26）：`src/eval_utils.py`、`eval_*.py`、`compute_scores.py` | HF `xinrongzhang2022/InfiniteBench` |
| SCBench | arXiv 2412.10319v2（2025-03-11）全文；ICLR 2025（PDF 頁首） | `microsoft/MInference@29ef1974`（2026-09-10）`scbench/` | HF `microsoft/SCBench`（sha `283310bb`，2024-12-24） |
| HELMET | arXiv 2410.02694v3（2025-03-06）正文＋相關附錄；ICLR 2025（PDF 頁首） | `princeton-nlp/HELMET@aeadacc6`（2026-09-19）：`README.md`、`CHANGELOG.md`、`arguments.py`、`data.py`、`configs/*.yaml` | HF `princeton-nlp/HELMET` |
| L-Eval | arXiv 2307.11088v3（2023-10-04）摘要＋§1、§3 統計、§4 指標；ACL 2024 Long（https://aclanthology.org/2024.acl-long.776/） | `OpenLMLab/LEval@cd34b050`（2024-07-09）README、LICENSE | HF `L4NLP/LEval` |
| LooGLE | arXiv 2311.04939v2（2024-09-06）摘要＋§1、§3 統計、§4.1–4.2；ACL 2024 Long（https://aclanthology.org/2024.acl-long.859/） | `bigai-nlco/LooGLE@fb9aee3b`（2026-08-06）README、LICENSE | HF `bigai-nlco/LooGLE` |
| BABILong | arXiv 2406.10149v2（2024-11-06）摘要＋§2–3；NeurIPS 2024 D&B（https://neurips.cc/virtual/2024/poster/97462） | `booydar/babilong@7a6efee2`（2026-06-01）README | HF `RMT-team/babilong`、`babilong-1k-samples` |
| LongProc | arXiv 2501.05414v3（2025-09-28）摘要＋§2–3；COLM 2025（PDF 頁首） | `princeton-pli/LongProc@673ec4c2`（2026-02-26）LICENSE | — |
| LongMemEval | arXiv 2410.10813v2（2025-03-04）摘要＋§1–3、附錄評審；ICLR 2025（PDF 頁首） | `xiaowu0162/LongMemEval@9e0b455f`（2026-05-11）README、LICENSE | HF `xiaowu0162/longmemeval-cleaned` |
| NoLiMa | arXiv 2502.05167v3（2025-07-09）摘要＋§3–4；ICML 2025（PMLR 267:44554–44570，https://proceedings.mlr.press/v267/modarressi25a.html） | `adobe-research/NoLiMa@cb14780b`（2025-07-17）LICENSE | HF `amodaresi/NoLiMa` |
| PPL 批評 | Hu et al., arXiv 2405.06105v1（ICLR 2024 Tiny Paper）；Fang et al., arXiv 2410.23771v5（ICLR 2025） | — | — |
| 原則出處 | Cake arXiv 2410.03065v2 p5；Rethinking KV compression（MLSys 2025）arXiv 2503.24000v1 p5；Agrawal & Mayer arXiv 2607.05399v1 p6 | — | — |
| 長輸出推理 | ForesightKV arXiv 2602.03203v2（2026-06-01）§4.1；TRIM-KV arXiv 2512.03324v2（ICLR 2026）§5、附錄 | — | — |
| GitHub issue | LongBench #26、#61、#111、#144；RULER #70、#99、#107、#108、#112；MInference #154、#200、#202；HELMET #35、#43 | 以 `gh api` 讀取，2026-10-06 | — |

原文 PDF 與文字檔、clone 下來的 repo 存在 scratchpad `E09/`（不進 git）。

## 重點摘要

1. **長度單位不統一，同名「128K」意思不同。** LongBench v1／-E 用英文詞數、中文字數（Table 1 p4）；LongBench v2 用詞數，分組 Short <32K、Medium 32K–128K、Long >128K **詞**（p6）；HELMET 用 Llama-2 token，且 K=1024（p2 腳註、p7）；RULER 用受測模型的 tokenizer，且長度**包含模板與生成 token**（`niah.py` `generate_samples`）。〔原文〕〔程式碼〕
2. **截斷策略互相衝突。** LongBench／v2、InfiniteBench、SCBench 從中間截（保留頭尾）；HELMET 從尾端截（保留開頭）；LooGLE 頭尾拼接（〔複核補充〕效果等同中間截，只是截文件而不是整段 prompt）；RULER 不截，直接生成剛好的長度。截哪裡就決定哪些 token 進 KV，所以「同 benchmark、不同截斷」的分數不可比。〔原文〕〔程式碼〕
3. **原生超過 128K 的不多。** RULER（任意長度）、BABILong（HF 有 0–10M 的 split）、InfiniteBench（平均約 200K token；En.QA 192.6K）、LongBench v2 的 Long 組、LongMemEval_M（約 1.5M）、SCBench 的原始 context（但官方腳本預設截到 131,072）。LongBench v1／-E（平均 1,235–22,337 詞或字）、LooGLE（平均 21K–36K token）、NoLiMa（論文只測到 32K）、LongProc（輸入最多約 38K）都在 64K 以下。HELMET README 寫「>128k 尚未支援」。〔原文〕〔文件〕
4. **SCBench 是唯一為「KV 跨請求重用」設計的品質 benchmark，但重用是構造的。** 多輪模式把前幾輪的**標準答案**放進歷史，不是模型輸出（p8；維護者在 MInference issue #154 確認論文主表都這樣）；沒有到達時間；重用比例由輪數決定（約 1−1/T，T≈5 時約 80%〔計算〕）；不量時間。論文寫 931 sessions／4,853 輪，HF 實際 922 列〔文件〕。〔原文〕〔程式碼〕
5. **對有損 KV 最有鑑別力的是「不可壓縮的精確檢索」。** SCBench 指出 NIAH 的 haystack 是可壓縮的重複雜訊，會高估 sub-O(n) 方法（p10）；Llama-3.1-8B 多輪模式下，KIVI 2-bit 讓字串檢索從 57.1 掉到 12.0，全域類只從 35.1 到 31.0（Table 4 p9）。HELMET 量到原始 NIAH 與下游任務的 Spearman ρ 全部 ≤0.8，RAG 最能代表下游（p7）。〔原文〕
6. **「只量時間可用合成 token」的出處是 Cake p5（only token length matters）。**〔複核修正：原寫「但只對無損、輸出長度固定的情況成立」。就 TTFT 而言，真正的條件是 prefill 計算與 KV 搬移量與內容無關；輸出長度不影響 TTFT，固定位元寬的有損格式也不讓 TTFT 隨內容變。「無損、固定輸出長度」是量端到端延遲或吞吐時才要加的限定〕MLSys'25 的 Rethinking 一文指出：有損壓縮會改變回應長度，用固定回應長度量端到端延遲並不恰當（p5）；資料相依的稀疏 prefill（MInference，SCBench D.1 p22）連 TTFT 都與內容有關。PPL 不能代替任務：Hu et al. 發現 PPL 與長文理解沒有相關（p1）；Fang et al. 指出 PPL 把關鍵 token 平均掉了，在 GovReport 上 PPL 與 LongBench 分數無相關、LongPPL 則達 r=−0.96（p1–2）。〔原文〕
7. **版本漂移讓同名分數不可比。** RULER 獨立腳本在 2025-01-22 到 2026-07-22 之間遺失 answer prefix（issue #107，作者確認；#108 修正）；RULER main 已把舊流程標成 deprecated，新流程改走 NeMo-Skills，範例 `tokens_to_generate=4096`。NIAH repo 在 2026-05 改寫成 v2，評分從 GPT-3.5 打 1–10 分改成子字串比對。LongBench v2 的失敗請求被丟掉，分母只算寫出的列（`pred.py` L98–99、`result.py` L37；issue #144）。LongBench 的 HF 載入腳本已不被新版 `datasets` 支援。〔程式碼〕〔文件〕
8. **使用者文件的待改處。** intro 表 17 把系統論文的負載合成（到達、快取比例）和品質 benchmark（SCBench、RULER、LongBench v2，只提供內容）放在同一欄沒有區分，建議註明後者只提供內容、沒有到達過程〔複核修正：原寫「類別錯置」，說得太重；intro 的做法本來就是「真實 trace 的時間＋長 context 的內容」〕。「SCBench 重用率是構造出來的」正確，但要補上：歷史用標準答案、預設截到 128K、不量時間。「只量時間（TTFT）時可以用合成的 token」〔複核修正：原寫「要加限定：僅限無損路徑且固定輸出長度」，就 TTFT 不成立〕對 TTFT 成立，只需排除資料相依的方法；延伸到端到端延遲時才要加「有損方法要用真實內容並記錄回應長度」。〔判讀〕（詳見〈使用者文件核對〉）

---

## 完整卡

### LongBench v1／LongBench-E：LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding（ACL 2024；arXiv 2308.14508）

- **讀了什麼**：〔全文〕arXiv v2，https://arxiv.org/abs/2308.14508 ；〔程式碼〕`THUDM/LongBench@2e00731f` 的 `LongBench/` 子目錄（v1 檔案在 v2 發布後移到這裡，根目錄 README L10）；查證 2026-10-06。
- **一句話**：中英雙語、21 個任務的長 context 理解 benchmark，全自動評分。
- **評測要證明的主張**：開源小模型與商用模型在長 context 仍有差距；擴展位置編碼並在更長資料上續訓有幫助；截短輸入會降分，所以題目確實需要長 context（p6–7）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 任務與分類 | 6 類 21 任務：單文件 QA（NarrativeQA、Qasper、MultiFieldQA-en／zh）、多文件 QA（HotpotQA、2WikiMQA、MuSiQue、DuReader）、摘要（GovReport、QMSum、MultiNews、VCSUM）、few-shot（TREC、TriviaQA、SAMSum、LSHT）、合成（PassageCount、PassageRetrieval-en／zh）、程式碼（LCC、RepoBench-P）。14 英文、5 中文、2 程式碼 | 〔原文〕Table 1 p4；〔文件〕`LongBench/README.md` L14–25 |
| 規模 | 4,750 筆；多數任務 200 筆，MultiFieldQA-en 150，LCC 與 RepoBench-P 各 500 | 〔原文〕p1、Table 1 p4；〔計算〕18×200＋150＋2×500＝4,750 |
| 長度與單位 | 全體平均 6,711 詞（英）／13,386 字（中）；各任務平均從 1,235（LCC）到 22,337（LSHT）。英文與程式碼用 Python `split()` 的詞數，中文用字數 | 〔原文〕p1、Table 1 p4；〔文件〕`task.md` 註 |
| 指標 | QA：F1（中文先 jieba 分詞）；摘要、DuReader、SAMSum：ROUGE-L；TREC、LSHT：分類準確率；合成：論文寫 EM，程式實作是「預測中出現的所有數字裡，等於答案者的比例」；程式碼：Edit Sim（`fuzz.ratio`） | 〔原文〕Table 1 p4；〔程式碼〕`eval.py` L18–40、`metrics.py` `count_score`／`retrieval_score`／`code_sim_score` |
| 輸出後處理 | TREC、TriviaQA、SAMSum、LSHT 只取第一行；程式碼取第一個不含 `` ` ``、`#`、`//` 的行 | 〔原文〕p7；〔程式碼〕`eval.py` L70–71、`metrics.py` `code_sim_score` |
| 截斷 | 輸入長度 L 超過模型上限 M 時從中間截，保留前 ⌊M/2⌋ 與後 ⌊M/2⌋；理由是開頭與結尾含指令或問題。程式以 token 計，前後各 `int(max_length/2)` 再 decode。`model2maxlen` 設 3,500／7,500／15,500／31,500（比模型名稱上的 4k／8k／16k／32k 略小，留空間給輸出〔判讀〕）。README 說依 Lost in the Middle 的觀察，實驗顯示這種截法影響最小 | 〔原文〕p6；〔程式碼〕`pred.py` L56–62、`config/model2maxlen.json`；〔文件〕`README.md` L49 |
| Chat template | trec、triviaqa、samsum、lsht、lcc、repobench-p **不套** chat 模板（答案應是補全式），其餘套 `build_chat`。`build_chat` 只認得 chatglm、longchat／vicuna、llama2、xgen、internlm；其他模型名稱原樣返回，等於不套模板 | 〔原文〕p6；〔程式碼〕`pred.py` L21–42、L63–64；〔判讀〕新模型（例如 Llama-3.1）若沒改 `build_chat`，就完全沒有 chat 模板 |
| Prompt | 每個任務一個固定模板（`dataset2prompt.json`）；問題 `{input}` 放在 `{context}` 之後 | 〔程式碼〕`config/dataset2prompt.json` |
| 生成長度 | QA 32–128；摘要 512；few-shot 32–128；合成 32；程式碼 64 | 〔程式碼〕`config/dataset2maxlen.json` |
| 解碼 | greedy（`do_sample=False`、`num_beams=1`）；samsum 另以 "\n" 為 EOS，避免無限重複；seed 42 | 〔原文〕p6「greedy decoding for reproducibility」；〔程式碼〕`pred.py` L73–90、L133 |
| LongBench-E | 從 13 個英文（含程式碼）資料集，依詞數在 0–4K、4–8K、8K+ 三組各抽相近數量；共 3,668 筆。eval 以資料的 `length` 欄（詞數）分組 | 〔原文〕p6、Table 8 p14；〔文件〕`task.md`；〔程式碼〕`eval.py` `scorer_e` L48–64；〔計算〕task.md 各列相加 |
| 參考模型與設定 | 8 個模型（GPT-3.5-Turbo-16k、Llama2-7B-chat-4k、ChatGLM2-6B-32k 等）；zero-shot，few-shot 例子當作 context 的一部分 | 〔原文〕p6 |
| 主要結果 | 截到 4K／8K 會降分（Fig. 2 p7）；GPT-3.5-Turbo-16k 從 0–4K 到 8K+ 相對下降 17%，ChatGLM2-6B-32k 只降 4%（p7） | 〔原文〕p7 |
| 消融 | 檢索式與摘要式 context 壓縮（§4.2）；去掉 context 測記憶（§4.3） | 〔原文〕p8–9 |
| 重複與統計 | 未說明（單次 greedy） | 〔原文〕 |
| 程式碼／資料／license | GitHub MIT（GitHub API）。HF 卡未標 license。HF 用載入腳本 `LongBench.py`；datasets-server 回報「Dataset scripts are no longer supported, but found LongBench.py」，所以新版 `datasets` 不能用 `load_dataset` 直接載，要改用 README 提供的 `data.zip`。〔複核補充〕HF repo 檔案只有 `LongBench.py`、`README.md`、`data.zip`（HF API tree，2026-10-07）；`huggingface/datasets` 4.0.0（2025-07-09）release notes 列出 "Remove scripts altogether"（PR #7592），所以 ≥4.0 的 `datasets` 確實不能跑這支腳本 | 〔文件〕GitHub API；HF API；datasets-server `/size` 回應（2026-10-06）；`README.md` L109；〔文件〕datasets 4.0.0 release（`gh api repos/huggingface/datasets/releases/tags/4.0.0`） |
| 設計理由（原文） | 多任務、雙語、全自動、低成本（README L12）；中段截斷保留指令與問題（p6）；補全式任務不套 chat 模板（p6）；用詞數算長度，避免不同 tokenizer 造成差異（task.md 註）；LongBench-E 看同一任務內長度變化的影響（p7） | 〔原文〕〔文件〕 |
| 設計理由〔判讀〕 | 用詞數跨 tokenizer 公平，代價是無法直接換算 KV 大小。問題放在 context 之後，代表 prefill 時就看得到問題，有利於依 query 壓縮的方法 | 〔判讀〕 |
| 陷阱 | (a) 單位是詞，8K+ 組不等於 8K token；(b) 平均長度多數 <16K 詞，對 16K–512K 太短；(c) F1／ROUGE 對細微退化不敏感、對輸出格式敏感（LongBench v2 p2、HELMET p2 與 p5 都這樣批評）；(d) Agrawal & Mayer（arXiv 2607.05399 p6）所有任務都套 chat 模板、TriviaQA 報 EM，官方是 few-shot 不套模板、TriviaQA 用 F1（Table 1 p4、`eval.py` L32），分數不可直接比；(e) 合成任務的分數是「數字命中比例」，輸出多個數字會被稀釋；(f) issue #61：照官方腳本跑 Llama2-7B-chat 的 PassageRetrieval-zh 得 10.12，README 是 0.5，未解決 | 〔原文〕〔程式碼〕〔文件〕 |
| 原文沒講清楚 | 新模型的 maxlen 與 chat 模板要自己決定；合成任務「EM」的精確定義（看程式才知道） | 〔判讀〕 |
| 與既有整理不一致 | `workloads_eval.md` §3.3「多數任務平均 5K–15K 字」：README 原文是 "5k to 15k"，單位英文是**詞**、中文是字，不是 token。「4,750 筆」「0–4K／4–8K／8K+ 均勻抽樣」一致，但沒寫出單位是詞、LongBench-E 只有 13 個英文資料集 | 〔原文〕〔文件〕 |
| KV 文獻誰用過 | 〔二手：workloads_eval §3.3〕ArkVale、EvicPress、AdaptCache、KIVI、KVDrive、KVP、KVTuner、LeoAM、LMCache、LookaheadKV、QEvict、Quest、ShadowKV、TRIM-KV、YAKV。〔原文〕Agrawal & Mayer 2026 用其中 6 個資料集（p6） | |
| 對本研究的意義〔判讀〕 | 只適合當 16K 以下的回歸測試。中段截斷剛好把中段丟掉，若要研究「哪個位置的 KV 放哪一層」，截斷會混淆位置效應 | 〔判讀〕 |

### LongBench v2：Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks（ACL 2025；arXiv 2412.15204）

- **讀了什麼**：〔全文〕arXiv v2，https://arxiv.org/abs/2412.15204 ；〔程式碼〕同 repo 根目錄 `pred.py`、`result.py`、`config/model2maxlen.json`；〔文件〕HF datasets-server 統計；查證 2026-10-06。
- **一句話**：503 題人工出、需要深度推理的真實長 context 四選一題。
- **評測要證明的主張**：最好的模型直接作答只有 50.1%，o1-preview 加長推理達 57.7%，超過 15 分鐘限時的人類專家（53.7%）；推理與 inference-time compute 是長 context 的關鍵（abstract p1）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 任務 | 6 大類、20 子任務：單文件 QA 175、多文件 QA 125、長 ICL 81、長對話歷史 39、程式碼庫 50、長結構化資料 33 | 〔原文〕Table 1 p4；〔文件〕datasets-server `statistics` 的 domain 頻次相同 |
| 規模與分組 | 503 題；Easy 192／Hard 311；Short 180／Medium 215／Long 108 | 〔原文〕p6；〔文件〕datasets-server 相同 |
| 長度與單位 | 8K–2M **詞**，多數 <128K 詞；中位 54K、平均 104K 詞。分組依詞數：Short <32K、Medium 32K–128K、Long >128K。HF 的 context 字元數：最小 48,765、中位 416,747、最大 16,182,936 | 〔原文〕p1–2、p6；〔文件〕datasets-server `statistics` |
| 難度定義 | Hard：自動審查的三個 LLM（GPT-4o-mini、GLM-4-Air、GLM-4-Flash）至多一個答對，**且**人類審查者超過 10 分鐘；其餘為 Easy | 〔原文〕Fig. 2 p5、p6 |
| CoT | 兩種設定：zero-shot；zero-shot＋CoT（先生成推理鏈，再根據推理鏈作答），仿 Rein et al. 2023 | 〔原文〕p6、Appendix D.2 p24 |
| 解碼與生成長度 | temperature 0.1；zero-shot `max_new_tokens`=128；CoT 第一次 1,024、第二次 128 | 〔原文〕Appendix D.2 p24；〔程式碼〕`pred.py` L95–104 |
| 截斷 | 超過視窗時從中間截（沿用 LongBench）；程式 `max_len` 多數模型 120,000 token、Claude-3.5-Sonnet 200,000，前後各一半 | 〔原文〕p6；〔程式碼〕`pred.py` L24–35、`config/model2maxlen.json` |
| Chat template | 透過 vLLM 的 OpenAI 相容 API，以單一 user 訊息送出，模板由伺服器套 | 〔程式碼〕`pred.py` L42–47；〔文件〕README Step 1–2 |
| 評分 | 用正則抽 "The correct answer is (X)"；Accuracy；隨機猜為 25%。另有 compensated 版：沒給出有效選項的題算 0.25 | 〔程式碼〕`pred.py` L58–68、`result.py` L5 與 L17；〔原文〕Table 2 註 p7 |
| 參考 | 10 個開源（皆 128K 視窗）＋7 個閉源模型；Qwen2.5 為了公平**不開 YaRN** | 〔原文〕p6、Appendix D.1 p24 |
| 主要結果 | GPT-4o 50.1%、o1-preview 57.7%、人類 53.7%；作者明說**長度組之間分數不遞減，是因為各長度組的任務組成差很多**（Table 2 註 ⋄ p7） | 〔原文〕p1、p7 |
| 消融 | RAG（以 512 token 切塊，取 top 4–256）；去掉 context 測記憶 | 〔原文〕p8 |
| 重複與統計 | 未報誤差棒 | 〔原文〕 |
| 程式碼／資料／license | repo MIT；HF `zai-org/LongBench-v2` 標 apache-2.0；資料蒐集約花 10 萬人民幣 | 〔文件〕HF API；〔原文〕Appendix C.4 p24 |
| 設計理由（原文） | 既有 benchmark 多為抽取式題、現代模型 NIAH 已完美、F1／ROUGE 不可靠、LLM 評審昂貴有偏，所以統一用選擇題並人工審查難度（p1–3）〔複核修正：原寫 p2–3；「抽取式、NIAH 完美、F1／ROUGE 不可靠」在 p1 右欄，LLM 評審的批評在 p2–3〕；用長度與難度獎金鼓勵更長更難的題（p6） | 〔原文〕 |
| 設計理由〔判讀〕 | 選擇題評分可靠，但分數離散，對 KV 精度造成的小退化不敏感 | 〔判讀〕 |
| 陷阱 | (a) 樣本少：p=0.4 的二項近似下，全體 95% CI 約 ±4.3 pp，Long 組（108 題）約 ±9.2 pp〔計算：1.96×√(0.24/n)〕；(b) temperature 0.1＋多程序並行 → 重跑不一致；issue #111 回報與論文的差距最大 4.6 pp（c4ai-command-r-plus Long 組 +4.6；GPT-4o-2024-08-06 Medium 組 −3.3）〔複核修正：原寫「差到 3.3 pp」，那只是 GPT-4o 的最大差；出處 issue #111 本文表格，`gh api` 2026-10-07〕，社群回覆（Wangmerlyn）指出多執行緒呼叫順序與快取讓同 seed 也不一致；(c) 呼叫失敗 5 次的題直接 `continue`，不寫入結果，`result.py` 分母只算寫出的列，沒有完整性檢查（我讀程式碼確認；issue #144 也回報）。〔複核補充〕`pred.py` L130–139 的快取會在重跑時補跑缺的 `_id`，所以隨機失敗可靠重跑補齊；但每次都失敗的題（例如被伺服器以長度拒絕）會一直缺；另外模型若真的回傳空字串也會被當成失敗丟掉（L98 只比對 `''`）；(d) 長度組與任務組成混淆，不能用 Short／Medium／Long 推論長度效應；(e) Long 組 >128K 詞，對 128K-token 模型一定會被中段截斷 | 〔原文〕〔程式碼〕〔文件〕〔計算〕 |
| 原文沒講清楚 | 詞數與 token 的換算；CoT 第二次呼叫是否重送完整 context（程式碼顯示會，`pred.py` L103） | 〔程式碼〕 |
| 與既有整理不一致 | `workloads_eval.md`：503 題、apache-2.0（HF）一致，但沒註明單位是詞、有 CoT 與 temperature 0.1。使用者 intro 表 17 把它列為「負載合成方式」：它只提供內容，不提供到達或 session 結構，表中應註明〔複核修正：原寫「類別錯置」，說得太重〕（見〈使用者文件核對〉） | |
| KV 文獻誰用過 | TRIM-KV（abstract p1 列出 LongBenchV2）〔原文〕；`workloads_eval` 也列 TRIM-KV〔二手〕 | |
| 對本研究的意義〔判讀〕 | 適合在 128K 附近做「真實、需要推理」的品質檢查；不適合量 1–2 pp 級的 ε。Medium＋Long（323 題）可測 32K 詞以上，但要固定截斷與解碼 | 〔判讀〕 |

### RULER：What's the Real Context Size of Your Long-Context Language Models?（COLM 2024；arXiv 2404.06654）

- **讀了什麼**：〔全文〕arXiv v3，https://arxiv.org/abs/2404.06654 ；〔程式碼〕`NVIDIA/RULER@c3f5e3b4`：`scripts/config_tasks.sh`、`synthetic.yaml`、`config_models.sh`、`run.sh`、`data/synthetic/constants.py`、`data/synthetic/niah.py`、`data/template.py`、`eval/synthetic/constants.py`；`rulerv1-ns` 分支 README；查證 2026-10-06。
- **一句話**：長度與難度可調的合成 benchmark，用來找「有效長度」。
- **評測要證明的主張**：17 個宣稱 ≥32K 的模型，NIAH 幾乎滿分，但在 RULER 上隨長度大幅退化；只有一半在 32K 達門檻，幾乎都在宣稱長度之前就跌破（abstract p1、p2）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 任務 | 4 類 13 任務：檢索（NIAH 8 個：single×3、multikey×3、multivalue、multiquery）、多跳追蹤（variable tracking，VT）、聚合（common words extraction CWE、frequent words extraction FWE）、QA（SQuAD、HotpotQA） | 〔原文〕§3 p3–6、Table 5 p18；〔程式碼〕`config_tasks.sh`、`synthetic.yaml` |
| 任務設定 | S-NIAH：haystack 為重複雜訊句或 Paul Graham essays，值為 7 位數字或 UUID；MK-NIAH-1 加 3 個干擾 needle；MK-2／MK-3 的 haystack 全部是 needle；MV 4 個值；MQ 4 個查詢；VT 1 條鏈 4 跳；CWE 10 個常見詞各出現 30 次、其他詞 3 次；FWE Zeta α=2.0；QA 把含答案的段落插進同資料集隨機段落 | 〔原文〕Appendix B p18、§3.4 p6；〔程式碼〕`synthetic.yaml` |
| 長度 | 4K、8K、16K、32K、64K、128K（程式為 4096…131072），可任意設定。長度用**受測模型 tokenizer** 計，**包含模板 token 與生成 token**：以二分搜尋找 haystack 大小使 input＋`tokens_to_generate` ≤ `max_seq_length`。〔複核修正：原寫「先扣模板 token」，那只是 NeMo-Skills 模式（`--prepare_for_ns`：`prepare.py` L90–94 改用 base 模板，並把模型模板 token 數傳給 `niah.py` L212 從上限扣掉）。獨立腳本是把模型 chat 模板與 answer prefix 直接併進 template（`prepare.py` L101），在 `input_text` 裡一起計數（`niah.py` L245、L268）。兩條路徑都含模板與生成 token〕 | 〔原文〕p6；〔程式碼〕`config_models.sh` L18–25、`niah.py` L58、L209–257、`prepare.py` L86–101、L139–140（`NVIDIA/RULER@c3f5e3b4`） |
| 樣本數 | 每任務每長度 500 | 〔原文〕p6；〔程式碼〕`config_tasks.sh` `NUM_SAMPLES=500` |
| needle 位置 | 從 0–100% 等距 40 個深度中隨機抽，seed 42。〔複核補充〕這只適用 essay haystack（`niah.py` L147–157，以句子為單位插入）；noise 與 needle haystack（S-NIAH-1 用 noise；MK-2、MK-3 用 needle，見 `synthetic.yaml`）是在句子索引上均勻隨機抽位置（`niah.py` L168–182 `random.sample(range(num_haystack), …)`），沒有 40 個深度的格點 | 〔程式碼〕`niah.py` L61、L105、L157、L168–182 |
| 模板 | 模型 chat 模板（Table 6 p20；`template.py` 手寫，不用 `apply_chat_template`）＋任務模板（Table 7–9 p21–23〔複核修正：原寫 Table 7–8 p21–22；p20 寫任務模板在 "Table 7 8 9"，QA 模板是 Table 9 p23〕）。VT、CWE 附一個 in-context 範例。回應開頭加 answer prefix，避免拒答或解釋 | 〔原文〕p6、Appendix D p20–22；〔程式碼〕`data/template.py`、`prepare.py` L99–101 |
| 生成長度 | NIAH 128、VT 30、CWE 120、FWE 50、QA 32 | 〔程式碼〕`data/synthetic/constants.py` |
| 解碼與平台 | greedy、BF16、vLLM、8×A100；設定檔 temperature 0.0、top_p 1.0、top_k 32 | 〔原文〕p6；〔程式碼〕`config_models.sh` L15–17 |
| 評分 | recall-based：`string_match_all`（NIAH／VT／CWE／FWE：參考答案作為子字串出現的比例）、`string_match_part`（QA：任一參考出現即得分）；不分大小寫 | 〔原文〕p6；〔程式碼〕`eval/synthetic/constants.py` L24–49 |
| Effective length | 門檻是 Llama2-7B 在 4K 的 13 任務平均 85.6%；超過門檻的最大長度＝effective length。另用線性遞增（inc）與遞減（dec）權重的加權平均排名 | 〔原文〕p6、Table 3 p7；〔文件〕README L54 |
| 主要結果 | Gemini-1.5-Pro >128K；GPT-4 有效 64K；Llama3.1-70B 64K；多數 32K 以下 | 〔原文〕Table 3 p7 |
| 消融 | Yi-34B 到 256K：needle 類型、干擾數、值數、查詢數、鏈與跳數；長 context 下模型傾向從 context 複製 | 〔原文〕§5 p7–8 |
| 重複與統計 | 未報誤差棒 | 〔原文〕 |
| 程式碼／資料／license | Apache-2.0；資料由腳本生成（Paul Graham essays 從 NIAH repo 下載，另下載 SQuAD、HotpotQA） | 〔文件〕README |
| 設計理由（原文） | 合成任務可控長度與複雜度，減少參數知識的干擾（p2）；選擇 4K 時多數模型表現不錯的任務，專看長度增加能否維持（p6）；13 任務是從 18 個設定的相關性分群中去除冗餘得到（Appendix C p19）；用 "special magic number" 當 needle 是因為好擴充（p4 腳註 2）；answer prefix 是為了避免拒答（p6） | 〔原文〕 |
| 設計理由〔判讀〕 | 答案是精確字串（7 位數字、UUID），量化誤差讓一個字元錯就扣分；MK-2／MK-3 的 haystack 全是 needle，不可壓縮。這兩點讓它對有損 KV 很敏感 | 〔判讀〕 |
| 陷阱 | (a) 85.6 是絕對門檻，強弱模型意義不同（NoLiMa 改用 base score 的 85%，見短卡）；(b) main 分支 README 已把舊流程標 deprecated，新流程走 NeMo-Skills（`rulerv1-ns`，依賴分支名 `chsieh/ruler-remove-prefix`），範例 `tokens_to_generate=4096`；(c) **answer prefix 遺失**：2025-01-22 的重構把 prefix 拆成另一個欄位，但 `call_api.py` 沒接回，直到 2026-07-22 的 PR #108；issue #107 回報 base 模型 4K 平均 6.48→37.51（gemma-3-270m-pt），作者確認。2025 年間用官方獨立腳本跑的 RULER 分數要查 commit；(d) 推理模型要移除 prefix、加大輸出，維護者在 #99 建議 128K 模型用 112K 輸入＋16K 輸出；(e) `string_match_part` 沒有字邊界，參考答案 "no" 會命中 "not"（#112，對 qa_2 影響至多 3.8 分）；(f) README 排行榜混了作者自報的結果（Jamba-large、Qwen2.5-1M、Qwen3、EXAONE，README 註）；(g) issue #70：沒把 prefix 放在 assistant 標頭之後，Llama-3.1-8B 的分數重現不出來 | 〔程式碼〕〔文件〕 |
| 原文沒講清楚 | 門檻選 Llama2-7B@4K 的理由只說是「品質門檻」；模板 token 數因模型而異，各模型的「128K」實際文字量不同 | 〔判讀〕 |
| 與既有整理不一致 | `workloads_eval.md`「合成、長度可調、Apache-2.0」一致。使用者 intro「長度可控的合成任務」一致 | |
| KV 文獻誰用過 | 〔二手：workloads_eval〕KVDrive、KVP、LookaheadKV、QEvict、ShadowKV、YAKV。〔原文〕HELMET 選 MK-2、MK-3、MV 當 synthetic recall（Table 3 p4）；SCBench 的 Retr.MultiHop 來自 RULER VT（p6） | |
| 對本研究的意義〔判讀〕 | PoC 主力：唯一能在 16K–256K 用同一組任務掃長度的。但只給「檢索、追蹤、聚合」類的 ε，不能代表摘要或推理 | 〔判讀〕 |

### Needle-in-a-Haystack（NIAH）：Kamradt 的壓力測試（GitHub repo，2023）

- **讀了什麼**：〔程式碼〕v1 `gkamradt/LLMTest_NeedleInAHaystack@7b90d285`（2024-04-12）的 `needlehaystack/run.py`、`llm_needle_haystack_tester.py`、`evaluators/openai.py`、`original_results/`；v2 `@021385d6`（2026-06-08）的 `README.md`、`scorers/exact_match.py`、`tasks/single_needle.py`。〔文件〕README。原始推文（GPT-4 2023-11-08、Claude 2.1 2023-11-21）未讀〔未查證〕。
- **一句話**：把一句事實插在長文不同深度，看模型能否取回。
- **評測要證明的主張**：模型在不同長度與深度的取回能力（README v1）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| haystack | Paul Graham essays，重複串接直到超過最大長度，再截到目標長度 | 〔程式碼〕v1 tester `read_context_files`、`encode_and_trim` |
| needle 與問題 | "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."；問題 "What is the best thing to do in San Francisco?"。多 needle 版用三種披薩配料 | 〔程式碼〕v1 `run.py` `CommandArgs` |
| 網格預設 | context 1,000–16,000 等距 35 點 × 深度 0–100% 35 點（linear 或 sigmoid）；留 200 token buffer 給系統訊息、問題、回答；needle 往前對齊到最近的句點 | 〔程式碼〕v1 `run.py`；tester `insert_needle` |
| 原始兩次測試 | GPT-4-128K：15 個長度（1K–128K）×15 個深度＝225 格；Claude 2.1：35 個長度（1K–200K）×35 個深度，深度為 sigmoid 分布（兩端密） | 〔計算〕v1 `original_results/` 檔名與 JSON；〔文件〕README v1 |
| 評分 | v1：GPT-3.5-turbo-0125 當評審（LangChain `labeled_score_string`），1／3／5／7／10 分。v2：大小寫不敏感的子字串比對（1.0／0.0），多 needle 取命中比例 | 〔程式碼〕v1 `evaluators/openai.py`；v2 `scorers/exact_match.py` |
| v2 改寫 | 2026-05 重寫：任務 single／multi／uuid／uuid_chain；sweep 可設多個 seed；每列存「重建 context 的配方」 | 〔文件〕v2 README；git log |
| 常見變體 | RULER 改用 "special magic number" needle 與雜訊 haystack，並加 MK／MV／MQ（p4）；SCBench 的 Mix.Sum+NIAH（p7–8）；NoLiMa 去除字面重疊；BABILong 把 bAbI 事實藏進 PG19 | 〔原文〕各篇 |
| 批評 | RULER：只反映表層檢索（abstract p1）。HELMET：128K 下幾乎全模型飽和（Fig. 1 p2）；與下游任務的 Spearman ρ 全部 ≤0.8，多數模型不是滿分就是接近 0，難以區分（p7）。NoLiMa：needle 與問題的字面重疊可被利用（abstract p1）。SCBench：haystack 是可壓縮的重複雜訊，sub-O(n) 方法看起來還行，會高估能力（p10）。LongBench v2：現代模型在 NIAH 已完美（p1〔複核修正：原寫 p2，該句在 p1 右欄〕） | 〔原文〕 |
| 陷阱 | v1 與 v2 評分方式不同，分數不可比；LLM 評審有成本與偏誤；長度以受測模型 tokenizer 計；對齊句點會讓實際深度偏離設定（v2 另存 `actual_depth_percent`） | 〔程式碼〕〔判讀〕 |
| license | MIT | 〔文件〕LICENSE.txt |
| KV 文獻誰用過 | SCBench 把 NIAH 放進多任務（p7–8）〔原文〕；其他 KV 論文〔未查證〕 | |
| 對本研究的意義〔判讀〕 | 只當 sanity check；對量化 ε 太容易飽和，而且 haystack 可壓縮，會高估有損方法 | 〔判讀〕 |

### InfiniteBench（∞Bench）：Extending Long Context Evaluation Beyond 100K Tokens（ACL 2024；arXiv 2402.13718）

- **讀了什麼**：〔全文〕arXiv v3，https://arxiv.org/abs/2402.13718 ；〔程式碼〕`OpenBMB/InfiniteBench@51d9b37b` 的 `src/eval_utils.py`、`eval_yarn_mistral.py`、`eval_chatglm.py`、`eval_yi_200k.py`、`compute_scores.py`；查證 2026-10-06。
- **一句話**：第一個平均超過 100K token 的中英多領域 benchmark。
- **評測要證明的主張**：現有長 context 模型在 100K+ 仍需大幅進步（abstract p1）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 任務 | 12 任務、5 領域：Retrieve.PassKey、Retrieve.Number、Retrieve.KV、En.Sum、En.QA、En.MC、Zh.QA、En.Dia、Code.Debug、Code.Run、Math.Calc、Math.Find；真實 context（小說、劇本、程式碼）＋合成 context | 〔原文〕§3 p4–6、Table 2 p3 |
| 規模 | 3,946 例；平均約 200K token | 〔原文〕p4 |
| 長度（token，平均輸入） | PassKey、Number 122.4K；KV 121.1K；En.Sum 103.5K；En.QA 192.6K；En.MC 184.4K；Zh.QA 2,068.6K；En.Dia 103.6K；Code.Debug 114.7K；Code.Run 75.2K；Math.Calc 43.9K；Math.Find 87.9K | 〔原文〕Table 2 p3 |
| 防記憶 | 小說做關鍵實體替換，變成「假小說」 | 〔原文〕§3.1.1 p4 |
| 截斷 | 中間截斷到 128K（YaRN-Mistral 只宣稱到 128K；API 會拒絕過長輸入）。程式：YaRN-Mistral 與 ChatGLM 為 128×1024；GPT-4 路徑 128,000−1,000；Yi-200K 為 200,000 | 〔原文〕§4.2 p7；〔程式碼〕`eval_yarn_mistral.py` L23、`eval_utils.py` L413、`eval_yi_200k.py` L23 |
| 模板 | 每個模型×任務各自設計 prompt（在短的假例子上調過） | 〔原文〕p7、Appendix B |
| 生成長度 | PassKey 6、Number 12、KV 50、En.Sum 1,200、QA／MC／Dia 40、Code 5、Math.Calc 30,000、Math.Find 3 | 〔原文〕Table 5 p16；〔程式碼〕`eval_utils.py` L41–54 |
| 解碼 | API 用預設超參數；YaRN-Mistral 用其原作者預設，只改輸出上限 | 〔原文〕Appendix D p15（輸出上限表 Table 5 在 p16）〔複核修正：原寫 Appendix D p16〕。〔複核補充〕〔程式碼〕官方 repo 另有 `eval_yi_200k.py`、`eval_chatglm.py`（不在論文主表的 4 個模型內），兩者 L25 用 vLLM `SamplingParams(temperature=0.8, top_p=0.95)` 取樣、不是 greedy，且 `get_pred` 沒把 `DATA_NAME_TO_MAX_NEW_TOKENS` 傳進 SamplingParams（L52–67），所以輸出上限落到 vLLM 預設 `max_tokens=16`（vLLM v0.4.0 `sampling_params.py` L117）。用這兩支腳本跑 En.Sum 等長輸出任務會被截在 16 token（`OpenBMB/InfiniteBench@51d9b37b`） |
| 評分 | En.Sum ROUGE-L-Sum；En.QA F1；Zh.QA 中文 F1；其他多為精確比對；選擇題沒輸出選項就算 0 | 〔原文〕p5、Table 3 註 p4；〔程式碼〕`compute_scores.py` |
| 參考模型 | GPT-4、Claude 2、Kimi-Chat（後兩者手動在網頁輸入）、YaRN-Mistral-7B-128K（A100 80GB，每例約 10 分鐘） | 〔原文〕§4.1 p7、Appendix D p15〔複核修正：原寫 p16〕 |
| 主要結果 | GPT-4 平均 45.63；檢索類最好；作者**沒有**觀察到一致的 lost-in-the-middle（Fig. 5 p8） | 〔原文〕Table 3 p4、§5.2 p7–8 |
| 限制（原文） | 依賴精確比對，受 prompt 模板與答案解析影響，新模型可能要重新設計 | 〔原文〕Limitations p9 |
| 程式碼／資料／license | repo MIT；HF 卡未標 license | 〔文件〕GitHub API、HF API |
| 設計理由（原文） | 既有 benchmark 平均約 10K，跟不上 100K+ 模型（p2）；合成任務可自動擴到更長（p2）；中段截斷是假設指令與書名在頭尾（p7） | 〔原文〕 |
| 設計理由〔判讀〕 | 為每個模型各自調 prompt，換模型後分數受 prompt 影響很大 | 〔判讀〕 |
| 外部批評 | HELMET：∞Bench 上 Llama-3.1-70B 輸給 8B，趨勢異常（Fig. 1 p2）；只含簡單 QA（p2）；用 ROUGE（p5）；HELMET 加 2-shot 並改 prompt 重做 ∞Bench QA 後更能反映能力（p6） | 〔原文〕 |
| 陷阱 | Zh.QA 平均 2M token，截到 128K 等於只看頭尾；Code.Debug 給了選項，作者自己提醒容易被外部檢索解（p5）；Math.Calc 要輸出 30K token，所有模型近 0 分（Table 3 p4）；Kimi-Chat 在 Retrieval.KV 中段的掉分是答案被截斷造成的（Fig. 5 註 p8） | 〔原文〕 |
| 與既有整理不一致 | `workloads_eval.md` 未收錄 | |
| KV 文獻誰用過 | SCBench 的 En.QA、Zh.QA、En.MultiChoice、Math.Find 由 ∞Bench 延伸（p6–7）；HELMET 用 ∞Bench QA／MC／Sum（Table 3 p4）〔原文〕。KV 論文直接使用〔未查證〕 | |
| 對本研究的意義〔判讀〕 | En.QA、En.MC 是少數「真實、平均約 190K token」的任務，可做 128K–256K 的真實任務 ε；必須固定截斷策略與長度上限 | 〔判讀〕 |

### SCBench：A KV Cache-Centric Analysis of Long-Context Methods（ICLR 2025；arXiv 2412.10319）

- **讀了什麼**：〔全文〕arXiv v2，https://arxiv.org/abs/2412.10319 ；〔程式碼〕`microsoft/MInference@29ef1974` 的 `scbench/readme.md`、`run_scbench.py`、`eval_utils.py`、`args.py`、`scripts/run_all_tasks.sh`；〔文件〕HF `microsoft/SCBench` 與 datasets-server 統計；issue #154、#200、#202；查證 2026-10-06。
- **一句話**：在共享 context、多輪與多請求下評估 KV 相關的長 context 方法。
- **評測要證明的主張**：單請求 benchmark 看不到 KV 重用後的失敗；sub-O(n) 記憶的方法在多輪解碼幾乎不可行，O(n) 記憶＋稀疏 prefill 則穩健（abstract p1、p4）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| KV 生命週期四階段 | ① 生成（prefill：稀疏注意力 A-shape、Tri-shape、MInference；SSM／hybrid；prompt 壓縮）② 壓縮（丟棄：StreamingLLM、SnapKV、PyramidKV；量化：KIVI）③ 檢索（依前綴從 KV 池取回重用以降 TTFT；語意檢索如 CacheBlend）④ 載入（從 VRAM、DRAM、SSD 或 RDMA 動態載入部分 KV：Quest、RetrievalAttention） | 〔原文〕Fig. 1 p1、§2 p5、Table 1 p4 |
| 任務 | 12 任務、4 種能力：字串檢索（Retr.KV、Retr.Prefix-Suffix、Retr.MultiHop）、語意檢索（Code.RepoQA、En.QA、Zh.QA、En.MultiChoice）、全域資訊（Math.Find、ICL.ManyShot、En.Sum）、多任務（Mix.Sum+NIAH、Mix.RepoQA+KV） | 〔原文〕Table 2 p6、Table 3 p7 |
| 兩種共享模式 | **多輪**：同一 session 內，後續輪次的歷史用**標準答案**取代模型輸出（依 Zheng et al. 2023a）。**多請求**：跨 session 共享 context，用 `--same_context_different_query` 啟用。HF 實作先只 prefill context（不含 query）並保存 KV，每題 decode 時不更新全域 KV；vLLM 實作則是把 context＋query 重新送 `generate` | 〔原文〕§3.2 p8；〔程式碼〕`eval_utils.py` `GreedySearch.test_scdq` L1204–1236、`GreedySearch_vLLM.test_scdq` L1076–1105；〔文件〕issue #154 維護者回覆 |
| 規模 | 論文：931 sessions、4,853 輪，平均 5 輪。HF：922 列。差異：En.Sum 論文寫 79 sessions，HF 70 列（350 輪÷5＝70，論文可能是筆誤〔判讀〕）；另外 En.MultiChoice 論文 299 輪、HF 229 輪；Math.Find 論文 240 輪、HF 600 輪 | 〔原文〕p5、Table 2 p6；〔文件〕datasets-server `/size`；〔計算〕datasets-server `statistics` 的 `multi_turns` 長度平均×列數 |
| 長度 | 平均輸入 227K token；各任務從 22K（ICL.ManyShot）到 1.5M（Zh.QA）。HF context 字元數全體 28,207–6,556,639（皆在 Zh.QA） | 〔原文〕Table 2 p6；〔文件〕datasets-server `statistics`。〔複核補充〕Table 2 欄名只寫 "Avg. Input Length"，沒有標單位；正文 p6 說 RepoQA 輸入到 64K tokens（表中 65K）、p7 說 En.Sum 文件 8K–20K tokens，所以推定為 token〔判讀〕，tokenizer 未說明。字元數範圍只涵蓋 datasets-server 有字串統計的 config（`scbench_many_shot` 的 context 是 label 型別、`scbench_mf` 是 list、`scbench_vt` 無 context 統計） |
| 截斷 | 官方腳本 `--max_seq_length 131_072`；context 從中間截到 max_seq_length − max_new_tokens×輪數（多請求再減 1,000） | 〔程式碼〕`scripts/run_all_tasks.sh`；`run_scbench.py` L95–118、L385–394；`eval_utils.py` `truncate_input` L1008–1014 |
| Chat template | 官方腳本加 `--use_chat_template` | 〔程式碼〕`run_all_tasks.sh` |
| 生成長度 | 每任務 5（Math.Find）到 1,024（RepoQA）；混合任務依子任務 | 〔程式碼〕`eval_utils.py` L41–58 |
| 解碼與平台 | greedy、BF16；正文寫 4×A100，附錄寫 8×A100 40GB 或 4×H100、>7B 用 TP；vLLM 0.5.2，FlashAttention kernel 換成自寫的 | 〔原文〕p8、D.2 p24 |
| 指標 | Accuracy；RepoQA 用 Pass@1；En.Sum 用 ROUGE；混合任務兩者並報 | 〔原文〕Table 3 p7 |
| 被評方法與設定 | 13 個方法、8 類；壓縮率多為 1/32（約 4K／128K）；KIVI 2-bit、group 32、residual 32；Quest token budget 4,096；附錄掃 1/32–1/2 | 〔原文〕Table 1 p4、Table 9 p23、Appendix C p21 |
| 主要發現 | (1) sub-O(n) 記憶在多輪解碼幾乎不可行：不同查詢需要的關鍵 KV 不同（Fig. 6 p9）；(2) 稀疏 prefill＋dense decode（O(n) 記憶）跨請求穩健；(3) KV 壓縮在共享情境表現差，只在第一輪有小幅好處；(4) 動態稀疏優於靜態；(5) 長生成時注意力分布漂移；(6) 不給 query 時 SnapKV 與 Tri-shape 掉分，MInference 較穩（Table 5 p10） | 〔原文〕p4、p8–10 |
| 與量化有關的數字 | Llama-3.1-8B 多輪：全 KV 平均 48.7、字串檢索 57.1、全域 35.1；KIVI（2-bit，τ=1/8）平均 32.0、字串檢索 12.0、全域 31.0。多請求：字串檢索 29.5→7.6 | 〔原文〕Table 4 p9 |
| 可壓縮 vs 不可壓縮 | NIAH 的 haystack 是重複雜訊，摘要的 context 也可壓縮，sub-O(n) 方法可以過得去；Retr.KV 與 Prefix-Suffix 是隨機、不可壓縮的，需要完整視窗。可壓縮任務會高估能力 | 〔原文〕p10 |
| 重複與統計 | 未報誤差棒 | 〔原文〕 |
| 程式碼／資料／license | MIT（HF 與 repo） | 〔文件〕 |
| 設計理由（原文） | 實際應用會重用 KV（vLLM／SGLang 的 prefix caching、各家 prompt caching）；許多方法依 query 壓縮，單請求測不出後續查詢失敗（p2）；多輪用標準答案是為了避免錯誤累積（#154 維護者 iofu728 回覆）〔複核修正：原寫「p8；#154」。p8 只說依循 Zheng et al. 2023a 與 Wang et al. 2024 用標準答案，沒有寫理由；「避免錯誤累積」只出現在 #154〕 | 〔原文〕〔文件〕 |
| 設計理由〔判讀〕 | 它把「同一份 KV 被讀很多次」做成品質測試，這正是分層 KV 的使用情境；但它量品質，不量時間 | 〔判讀〕 |
| 陷阱 | (a) 標準答案歷史不是真的多輪生成（有 `--disable_golden_context`，但論文主表都用預設）；(b) 預設截到 131,072，Zh.QA 的 1.5M context 只剩頭尾；(c) README 的預設超參數與論文不同：A-shape／StreamingLLM `n_local` 3,968（論文 4,096）、Quest `token_budget` 1,024（論文 4,096）、Tri-shape `n_last` 100（論文 Table 9 寫 128 dense rows，附錄文字又寫 64）；(d) 論文與 HF 筆數、輪數不一致；(e) vLLM 路徑的多請求只是重送 context＋query，有沒有真的重用 KV 取決於 vLLM 的 prefix caching 設定〔判讀〕；(f) issue #202 質疑多輪實作只快取第一輪 prompt，未獲回覆；issue #200 問能否只用第一輪當單請求評測，未獲回覆 | 〔原文〕〔程式碼〕〔文件〕 |
| 原文沒講清楚 | 多請求模式下各方法如何在「沒有 query」時壓縮（只有 Table 5 的比較）；長生成漂移的量化方法 | 〔判讀〕 |
| 與既有整理不一致 | `workloads_eval.md` §3.3：「922 列」與 HF 一致，但論文寫 931；「context 299K–3.17M 字元」其實是 En.QA（`scbench_qa_eng`）的範圍（298,903–3,171,853；〔複核補充〕En.MultiChoice `scbench_choice_eng` 的字元數範圍完全相同；兩者都由 ∞Bench 延伸，而 ∞Bench p4 說所有英文任務共用同一批改寫小說），全體是 28,207–6,556,639 字元；「每列 2 個以上 turn」一致（最少 2）；「`main_remote.tex` L1616 記為重用 80%」不是原文數字，是 1−1/T 在 T≈5 的結果〔計算：4,853／931＝5.21 輪，1−1/5.21＝80.8%〕 | 〔文件〕〔計算〕 |
| KV 文獻誰用過 | TRIM-KV（Table 2 p9；並沿用 SCBench 的多輪多 session 協定測 LongMemEval，p18）〔原文〕 | |
| 對本研究的意義〔判讀〕 | 唯一能直接量「量化後的 KV 被多輪重用後 ε 會不會累積」的現成 benchmark。建議子集：Retr.KV、Retr.Prefix-Suffix、Retr.MultiHop、RepoQA、En.QA、En.MultiChoice；逐輪報分（第 1 輪 vs 第 2–5 輪） | 〔判讀〕 |

### HELMET：How to Evaluate Long-Context Language Models Effectively and Thoroughly（ICLR 2025；arXiv 2410.02694）

- **讀了什麼**：〔全文：正文 p1–10、附錄 E.4–E.6〕arXiv v3，https://arxiv.org/abs/2410.02694 ；〔程式碼〕`princeton-nlp/HELMET@aeadacc6` 的 `README.md`、`CHANGELOG.md`、`arguments.py`、`data.py`、`configs/*.yaml`；issue #35、#43；查證 2026-10-06。
- **一句話**：七類應用導向任務、可控長度到 128K、以模型評分取代 ROUGE。
- **評測要證明的主張**：合成任務（NIAH）不能可靠預測下游表現；各類別趨勢不同、相關性低；開源模型在需要整個 context 推理或複雜指令時明顯落後，且長度越長差越多（abstract p1）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 對既有 benchmark 的批評 | 四個缺陷：下游任務覆蓋不足；長度不足（多數自然語料資料集 <128K）；指標不可靠（ROUGE 等）；不相容 base model，開發者只好用困惑度或合成任務 | 〔原文〕p2 |
| 對 NIAH 的批評 | 128K 下 NIAH 幾乎所有模型都飽和；RULER 與 ∞Bench 出現反直覺排序（RULER 上 Gemini Flash 勝 Pro；∞Bench 上 Llama-3.1-70B 輸 8B） | 〔原文〕Fig. 1 p2 |
| 對困惑度的批評 | 只指出開發者依賴困惑度與合成任務（p1–2、p6、E.5 p39）；HELMET **本身沒有**做困惑度與下游的相關性實驗 | 〔原文〕 |
| 類別與資料 | RAG（NQ、TriviaQA、PopQA、HotpotQA；SubEM）；引用生成（ALCE ASQA／QAMPARI）；重排（MS MARCO，NDCG@10）；多樣本 ICL（TREC coarse／fine、NLU、BANKING77、CLINC150）；長文件 QA（NarrativeQA 模型評分、∞Bench QA ROUGE F1、∞Bench MC）；摘要（∞Bench Sum、Multi-LexSum，模型評分）；合成回憶（JSON KV、RULER MK needle、MK UUID、MV；SubEM） | 〔原文〕Table 3 p4 |
| 長度 | L ∈ {8K, 16K, 32K, 64K, 128K}，L 以 Llama-2 token 計，K＝1024；以 passage 數、示範數或截斷控制長度 | 〔原文〕p2 腳註 2、p6–7 |
| 截斷 | 文件**從尾端截**（保留開頭）；用 Llama-2 tokenizer 截 context，並接上 "... [the rest of the text is omitted]"，讓每個模型看到相同資訊；若超過受測模型上限再用其 tokenizer 截 | 〔原文〕p5；〔程式碼〕`data.py` `truncate_llama2` L162–175、`model_utils.py` L198–199、`arguments.py` L41 |
| 評分 | 參考答案導向的模型評分（GPT-4o-2024-05-13）：QA＝流暢度（0／1）×正確度（0–3）；摘要＝把參考摘要拆成原子主張，算 recall／precision 的 F1，再乘流暢度。人類一致性 Cohen κ：precision 0.91、recall 0.76 | 〔原文〕§2.2 p5–6 |
| Prompt | 除 ICL 與 RULER 外都加 2-shot 示範；長生成用 L-Eval 的長度指令 | 〔原文〕§2.3 p6 |
| 解碼 | greedy；`do_sample=False` 時強制 temperature 0 | 〔原文〕p7；〔程式碼〕`arguments.py` L44–47、L76–78 |
| 規模 | 每資料集抽 100–600 筆；評 59 個模型 | 〔原文〕p7 |
| 主要發現 | 合成任務與下游的平均相關都 <0.8；RULER MK 等較難的回憶任務相關較高；HotpotQA 與 ∞Bench QA 的 ρ＝0.88，RAG 是最好的代理；類別間相關低；開源模型在引用與重排落後 30–40 分 | 〔原文〕§3.1–3.3 p7–9、Fig. 3–5 |
| 建議 | 快速開發用 RAG 類；最終要跨類別整體評估 | 〔原文〕abstract p1、p7 |
| 程式碼／資料／license | MIT；HF 資料約 34GB；README：「Support >128k input length」尚未勾選；vLLM 結果與 HF 略有差異，最終評估建議用 HF | 〔文件〕README L44、L79、L224。〔複核補充〕MIT 是 GitHub repo 的 license；HF `princeton-nlp/HELMET` 資料卡沒有 license 標籤（HF API，2026-10-07），各子資料集沿用原始來源的授權〔未查證〕 |
| 版本差異 | CHANGELOG 2025-02-25：ICL 改評 500 筆、每題不同示範、標籤平衡（論文尚未更新）；2024-10-04 修 seed；issue #43（2026-08 開、2026-09 關）：NarrativeQA 與 Multi-LexSum 的示範抽樣原本不受 `--seed` 控制 | 〔文件〕 |
| config 不一致 | 同一 RAG 任務：`rag.yaml` 的 `use_chat_template: false`、`rag_vllm.yaml` 為 true；`recall.yaml` false、`recall_vllm.yaml` true。22 個 config 中 11 個 false、11 個 true | 〔程式碼〕`configs/*.yaml` |
| 設計理由（原文） | 涵蓋多種應用、可控長度到 128K、對 base 與 instruct 模型都可靠（p3）；RAG 用檢索得到的干擾 passage 比隨機抽更真實、更難（p3–4）；ICL 標籤換成數字，測學新任務而非先驗（p5） | 〔原文〕 |
| 設計理由〔判讀〕 | 尾端截斷讓所有模型看到同一段開頭，公平但會丟掉文件後半的證據 | 〔判讀〕 |
| 陷阱 | 依賴 GPT-4o 評審（成本、模型版本漂移）；長度單位是 Llama-2 token，不是受測模型 token；尾端截斷與 LongBench 中段截斷對位置效應的影響方向相反；HF 與 vLLM 的 config 預設不同 | 〔原文〕〔程式碼〕〔判讀〕 |
| 與既有整理不一致 | `workloads_eval.md` 未收錄 | |
| KV 文獻誰用過 | 〔未查證〕 | |
| 對本研究的意義〔判讀〕 | RAG（HotpotQA、NQ）＋Recall 子集可作為 PoC 便宜且與下游相關的代理；上限 128K | 〔判讀〕 |

---

## 短卡

### L-Eval：Instituting Standardized Evaluation for Long Context Language Models（ACL 2024；arXiv 2307.11088）

- **讀了什麼**：〔部分〕摘要、§1、§3 統計（Table 1 p4）、§4 指標；README、LICENSE；ACL 頁面。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 目的 | 從資料與指標兩方面標準化長 context 評測 | 〔原文〕abstract p1 |
| 長度範圍 | 3K–200K token；各任務平均 4K–60K，最大樣本近 200K（Llama-2 tokenizer） | 〔原文〕abstract p1、p4；Llama-2 tokenizer 見 Table 1 註 p5〔複核修正：Table 1 在 p5，不是 p4〕 |
| 任務型態 | 20 子任務、508 篇長文件、>2,000 筆人工標註；分 closed-ended（選擇題、精確比對）與 open-ended（摘要、抽象 QA） | 〔原文〕p1 |
| 指標 | closed：精確比對；open：n-gram 與人類判斷相關差，主張用 LLM 評審並加長度指令（LIE） | 〔原文〕abstract p1、§4 |
| 規模 | 同上 | |
| license | GPL-3.0（repo LICENSE；HF `L4NLP/LEval` 標 gpl-3.0） | 〔文件〕 |
| KV 文獻誰用過 | 〔二手：workloads_eval §3.3〕HCache、Mooncake、Tutti | |
| 陷阱 | GPL-3.0 對整合有傳染性；open-ended 需 GPT-4 評審；多數任務平均遠短於 200K〔判讀〕 | |
| 與既有整理不一致 | `workloads_eval` 寫「3K–200K（Tutti p9）」「GPL-3.0」，與原文一致 | |

### LooGLE：Can Long-Context Language Models Understand Long Contexts?（ACL 2024；arXiv 2311.04939）

- **讀了什麼**：〔部分〕摘要、§1、§3（Table 2 p4）、§4.1–4.2；README、LICENSE。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 目的 | 用 2022 年後的新文件，避免訓練資料洩漏；強調長依賴 | 〔原文〕abstract p1 |
| 長度範圍 | 776 篇文件，平均 19.3K 詞。平均 token：arXiv 20,887、Wikipedia 21,017、劇本 36,412；最大詞數 197,977（arXiv） | 〔原文〕p2、Table 2 p4 |
| 任務型態 | 短依賴 QA（GPT-3.5 生成）、cloze；長依賴：摘要、1,101 題人工長依賴 QA | 〔原文〕p2、Table 2 p4；短依賴 QA 由 GPT3.5-turbo-16k 生成，cloze 所用的事實摘要也由它生成（§3.3 p6）〔複核補充：原未給出處頁〕 |
| 指標 | BLEU、ROUGE、METEOR、BERTScore、GPT-4 評分；cloze 用精確／部分比對 | 〔原文〕§4.2 p7–8 |
| 規模 | 6,448 題 | 〔原文〕p2 |
| 截斷 | 頭尾拼接（例如 16K＝前 8K＋後 8K）。〔複核補充〕效果上就是丟掉中段，與 LongBench 的中段截斷同類；差別在 LooGLE 截的是「輸入文件」，LongBench 截的是套好模板的整段 prompt | 〔原文〕p7；LongBench `pred.py` L55–62 |
| license | repo MIT；HF CC-BY-SA-4.0 | 〔文件〕 |
| KV 文獻誰用過 | 〔二手：workloads_eval〕Strata、Tutti | |
| 陷阱 | 平均只有 21K–36K token；短依賴 QA 是 GPT-3.5 生成 | 〔原文〕 |
| 與既有整理不一致 | `workloads_eval` §3.3 寫「多數 >100K（Tutti p9）」：**與原文 Table 2 不符**，平均 token 是 20,887–36,412。〔複核修正：原寫「README 只說 many of which exceed 100k words」。實際上論文 p2 本身就有同一句（"over 6,448 test instances …, many of which exceed 100k words"），README L21 照抄；而 Tutti p9 寫的是 "many test samples exceeding 100k tokens"。所以 workloads_eval 的錯在把 "many" 寫成「多數」，並沿用 Tutti 把詞換成 token。此外論文自己這句也和 Table 2 不太合：Wikipedia、劇本的最大詞數只有 46,250、62,752，只有 arXiv 最大 197,977 詞超過 100K〕 | 〔原文〕LooGLE p2、Table 2 p4；Tutti（arXiv 2605.03375）p9；〔文件〕LooGLE README L21 |

### BABILong：Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack（NeurIPS 2024 D&B；arXiv 2406.10149）

- **讀了什麼**：〔部分〕摘要、§2–3 主要結果；README。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 目的 | 把 bAbI 推理事實散在超長自然文本（PG19 書籍）中，測跨事實推理 | 〔原文〕abstract p1、p3 |
| 長度範圍 | HF split：0、1、2、4、8、16、32、64、128、256、512K、1M、10M；論文用 ARMT 評到 50M | 〔文件〕README L5；〔原文〕p2 |
| 任務型態 | 20 個 bAbI 任務（QA1–QA20）；主表用 QA1–QA5 | 〔原文〕p4 |
| 指標 | 準確率；>85% 算滿意、<30% 算完全失敗 | 〔原文〕p5 |
| 規模 | 每任務每長度 100 筆（`RMT-team/babilong`）或 1,000 筆（`babilong-1k-samples`） | 〔文件〕README L5 |
| license | code Apache-2.0；PG19 Apache-2.0；bAbI BSD | 〔文件〕README L198–200 |
| 主要結果 | 主流 LLM 只有效利用 10–20% 的 context；RAG 在單事實 QA 約 60% | 〔原文〕abstract p1 |
| KV 文獻誰用過 | LASER-KV（arXiv 2602.02199）〔摘要〕 | |
| 陷阱 | haystack 與事實語域差很多，仍屬 NIAH 型〔判讀〕；100 筆版本 CI 寬 | |

### LongProc：Benchmarking Long-Context Language Models on Long Procedural Generation（COLM 2025；arXiv 2501.05414）

- **讀了什麼**：〔部分〕摘要、§2–3（Table 2 p4）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 目的 | 同時要求整合分散資訊與長輸出 | 〔原文〕abstract p1 |
| 長度範圍 | 輸出三級 0.5K／2K／8K token（Llama-3 tokenizer）；輸入最大約 38K（HTML→TSV 的 8K 級） | 〔原文〕Table 2 p4、p6 |
| 任務型態 | 6 個程序生成任務：HTML→TSV、Pseudocode→Code、Path Traversal、ToM Tracking、Countdown、Travel Planning | 〔原文〕p3–4 |
| 指標 | 規則式（列級 F1、單元測試、精確比對、驗證器） | 〔原文〕p4 |
| 規模 | 多數每級 100 筆 | 〔原文〕Table 2 p4 |
| license | Apache-2.0 | 〔文件〕LICENSE |
| KV 文獻誰用過 | TRIM-KV（greedy，p7、附錄 p17） | 〔原文〕 |
| 陷阱 | 輸入不長，測的是解碼期累積的 KV，不是長 prefill | 〔判讀〕 |

### LongMemEval：Benchmarking Chat Assistants on Long-Term Interactive Memory（ICLR 2025；arXiv 2410.10813）

- **讀了什麼**：〔部分〕摘要、§1–3、附錄評審說明；README。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 目的 | 聊天助理的長期記憶：資訊抽取、多 session 推理、時間推理、知識更新、拒答 | 〔原文〕abstract p1 |
| 長度範圍 | LongMemEval_S 約 115K token／題；LongMemEval_M 500 個 session，約 1.5M token；可擴充 | 〔原文〕p2、p5 |
| 任務型態 | 500 題人工設計，證據 session 插在不相關的歷史 session 中，附時間戳 | 〔原文〕p1、p5 |
| 指標 | GPT-4o 評審，與人類專家一致性 >97% | 〔原文〕p6、附錄 |
| license | MIT（repo、HF） | 〔文件〕 |
| 版本 | 2025-09 發布 cleaned 版（`longmemeval-cleaned`），與原版不可比 | 〔文件〕README L15（README 只說清理了歷史 session、附 change log；「與原版不可比」是〔判讀〕〔複核修正：原未標判讀〕） |
| KV 文獻誰用過 | TRIM-KV 用 LongMemEval_S（以 Qwen3 tokenizer 最長約 123K），沿用 SCBench 的多輪多 session 協定（p18） | 〔原文〕 |
| 陷阱 | 依賴 LLM 評審；資料有清理前後兩版 | |

### NoLiMa：Long-Context Evaluation Beyond Literal Matching（ICML 2025；arXiv 2502.05167）

- **讀了什麼**：〔部分〕摘要、§3–4；LICENSE。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 目的 | NIAH 的問題與 needle 有字面重疊，模型可走捷徑；改成需要潛在聯想才能找到 needle | 〔原文〕abstract p1 |
| 長度範圍 | 250、500、1K、2K、4K、8K、16K、32K；haystack 本身 >60K token | 〔原文〕p5 |
| 任務型態 | 58 組問題-needle（一跳、二跳）× 5 個 haystack × 26 個等距位置＝每長度 7,540 次測試 | 〔原文〕p4–5 |
| 指標 | 回答含正確角色名即算對。base score＝250／500／1K 中最高者的平均；effective length＝分數仍高於 base score 85% 的最大長度（RULER 用絕對 85.6%，NoLiMa 用相對門檻） | 〔原文〕p5 |
| 主要結果 | 32K 時 13 個模型中 11 個跌到 base score 一半以下；GPT-4o 從 99.3% 到 69.7% | 〔原文〕abstract p1 |
| license | Adobe Research License，僅限非商業研究；HF 標 other | 〔文件〕LICENSE、HF API |
| KV 文獻誰用過 | 〔未查證〕 | |
| 陷阱 | 論文只到 32K，更長要自己生成；授權限制 | |

### PG-19／WikiText 困惑度：用法與限制

- **讀了什麼**：〔原文〕Hu et al. 2024（arXiv 2405.06105，ICLR 2024 Tiny Paper）全文；Fang et al. 2025（arXiv 2410.23771，ICLR 2025）摘要與 §1；〔摘要〕KVQuant（arXiv 2401.18079）、StreamingLLM（arXiv 2309.17453）；〔文件〕HF 資料卡。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 目的 | 語言建模困惑度；長 context 研究常用來「證明模型能處理長文」 | 〔原文〕Hu et al. p1 |
| 資料 | PG-19（書籍；HF `deepmind/pg19` 標 apache-2.0）；WikiText（HF `Salesforce/wikitext` 標 cc-by-sa-3.0、GFDL） | 〔文件〕HF API |
| KV 文獻怎麼用 | KVQuant 報 3-bit 量化在 Wikitext-2 與 C4 的困惑度退化 <0.1〔摘要〕；StreamingLLM 宣稱穩定的串流語言建模到 400 萬 token〔摘要；用哪個語料未查證〕；BABILong 拿 PG19 當 haystack（README） | |
| 限制（原文） | Hu et al.：困惑度與長文理解**沒有相關**，困惑度主要反映局部資訊的建模，只有 4K 視窗的 LLaMA2 也能有可比的困惑度（p1）。Fang et al.：困惑度把對長 context 關鍵的 token 平均掉了；在 GovReport 上困惑度與 LongBench 分數無相關，只算關鍵 token 的 LongPPL 達 Pearson −0.96（p1–2）。HELMET：開發者因 benchmark 不相容 base model 而依賴困惑度（p2） | 〔原文〕 |
| 判讀 | StreamingLLM 的困惑度穩定到 4M token，但在 SCBench 多輪字串檢索只剩 0.4（Llama-3.1-8B，Table 4 p9）：困惑度穩定不代表檢索能力保留 | 〔判讀〕 |
| 陷阱 | teacher forcing 下量困惑度，看不到生成誤差累積；不同 tokenizer 的困惑度不可比 | 〔判讀〕 |

### 長輸出推理題（GSM8K、AIME、MATH-500）在 KV 逐出論文中的用法

- **讀了什麼**：〔原文〕ForesightKV arXiv 2602.03203v2 §4.1（p5–6）、Table 3；TRIM-KV arXiv 2512.03324v2 §5.1（p7）、附錄（p17–18）。

| 欄位 | ForesightKV | TRIM-KV | 出處 |
|:--|:--|:--|:--|
| 題目 | AIME2024、AIME2025 | AIME24、GSM8K、MATH-500；另有 LongProc、LongMemEval_S、SCBench、LongBench v2 | 〔原文〕FKV p6；TRIM p1、p7 |
| 模型 | DeepSeek-R1-Distill-Qwen-7B、Qwen3-4B、Qwen3-1.7B | Qwen3 1.7B–14B、R1-Distill 系列 | 〔原文〕FKV p6；TRIM p7 |
| 解碼 | temperature 0.6、top-k 20、top-p 0.95（依 DeepSeek-R1 與 Qwen3 的設定） | 依 SeerAttention-R；sampling 參數〔未查證〕；LongProc 用 greedy | 〔原文〕FKV p6；TRIM p7 |
| 統計 | 每個 benchmark 獨立評 32 次，報 pass@1 平均 | AIME24 64 次、GSM8K 與 MATH-500 8 次，報 pass@1 平均 | 〔原文〕FKV p6；TRIM p7 |
| KV 預算 | 1,024、2,048、4,096；每 256 步逐出一次 | 例如 AIME24 用 512、GSM8K／MATH-500 用 128 做同預算比較 | 〔原文〕FKV p6；TRIM p8〔複核修正：原寫 TRIM p7，該句在 p8〕 |
| 生成長度 | 吞吐量表在 8K、16K、32K 生成長度下量；準確率實驗的最大生成長度〔未查證〕 | 〔未查證〕 | 〔原文〕FKV Table 3 |
| 判讀 | 輸入很短、輸出很長，測的是**解碼期 KV** 的管理，與長 prefill 的分層放置是不同問題；必須多次取樣取平均，因為 temperature >0 | | 〔判讀〕 |
| 題數 | AIME、GSM8K 題數〔未查證〕；MATH-500 依名稱為 500 題〔判讀〕 | | |

---

## 選擇準則表

| 研究問題 | 適合的 benchmark | 理由 | 出處 |
|:--|:--|:--|:--|
| 量系統時間（無損：載入 vs 重算） | 目標長度的合成 token；固定輸出長度 | 無損方法的計算量與 I/O 只由長度決定 | Cake p5：只有 token 長度有影響，用合成 prompt；Rethinking p5 的限制見下一節 |
| 量有損壓縮或量化後的品質 ε | RULER（尤其 MK-2／MK-3、MV、VT）、SCBench 字串檢索、HELMET RAG＋Recall、真實任務（∞Bench En.QA／En.MC、LongBench v2） | 不可壓縮的精確檢索最敏感；真實任務補足摘要與推理 | SCBench p10、Table 4 p9；HELMET p7；RULER p6 |
| 量跨輪重用後的品質 | SCBench 多輪與多請求；LongMemEval（依 SCBench 協定） | 唯一把同一 context 多次查詢設計進題目的 benchmark | SCBench §3.2 p8；TRIM-KV p18 |
| 量「壓縮時不知道 query」 | SCBench 多請求模式 | context 先 prefill、query 後到 | SCBench Table 5 p10 |
| 量長度外推、有效長度 | RULER（絕對門檻 85.6）、BABILong（85%／30%）、NoLiMa（base score 的 85%）、HELMET 8K–128K | 同一組任務掃長度 | RULER p6；BABILong p5；NoLiMa p5；HELMET p7 |
| 量長輸出、解碼期 KV | LongProc 2K／8K；AIME、MATH-500、GSM8K（多次取樣） | 輸出長、輸入短 | LongProc p1；ForesightKV p6；TRIM-KV p7 |
| 量位置效應 | NIAH 深度網格、RULER 的深度抽樣、HELMET RAG 的 6 個位置 | 可控 needle 位置 | NIAH v1 code；RULER `niah.py`；HELMET p4 |
| 快速開發用的代理 | HELMET RAG（HotpotQA、NQ） | 與下游相關最高，不易飽和 | HELMET p7–8 |
| 不建議單獨使用 | 原始 NIAH；困惑度 | 飽和、可壓縮、與下游相關低 | HELMET p2、p7；SCBench p10；Hu et al. p1；Fang et al. p1–2 |

### 「只量時間可用合成 token、量品質必須用真任務」：出處查證

| 主張 | 支持的原文 | 限制或反例 | 判讀 |
|:--|:--|:--|:--|
| 只量時間可用合成 token | Cake（arXiv 2410.03065v2）p5 §5.1：token 的實際值不影響 Cake 的效能評估，只有長度有影響，所以依 CacheGen 的統計在 4K–16K 每 2K 取一個長度合成 prompt，並事先算好所有 KV 存起來 | Rethinking KV compression（MLSys 2025，arXiv 2503.24000）p5：一般用固定回應長度量吞吐量並不恰當；壓縮演算法會改變回應長度，而這個影響多被忽略（Missing Piece 2）。資料相依的稀疏 prefill（MInference 依輸入動態決定 pattern，SCBench D.1 p22）其計算量也可能隨內容變 | 只對「無損、輸出長度固定、計算量與內容無關」成立。Tiara 的 CPU／SSD 與重算是無損的，TTFT 可用合成 token；FP8／INT4 層的端到端時間則要用真實內容，並記錄回應長度。〔複核補充，判讀〕三個條件要分開看：只量 TTFT 時，輸出長度根本不進入 TTFT，「無損」也不是必要條件（固定位元寬的 FP8／INT4 格式，搬移與反量化的位元組數與內容無關）；TTFT 真正需要的只有「prefill 計算與 KV 搬移量與內容無關」。「無損」與「輸出長度固定」是量端到端延遲或吞吐時才需要的條件（Rethinking p5） |
| 量品質必須用真任務 | HELMET：合成任務與下游的相關都 <0.8（p7）；SCBench：可壓縮的合成 haystack 會高估（p10） | RULER 主張合成任務**有利於**品質評估：可控長度、減少參數知識干擾（p2）；HELMET 也承認較難的合成回憶（RULER MK）與下游相關較高，可作 sanity check（p7） | 原句太強。較準確的說法：品質要包含真實任務，合成任務只用「不可壓縮、較難」的（RULER MK／MV、SCBench Retr.KV），不能只用 NIAH 或困惑度 |
| 有無論文同時明說兩者 | 未找到同一篇明說「時間用合成、品質用真任務」；Agrawal & Mayer（arXiv 2607.05399）同時量品質與系統效能，但品質用 LongBench（p6） | — | 〔未查證〕是否有其他論文明說 |

---

## 常見的實作差異會讓分數不可比

| 差異 | 具體例子 | 出處 |
|:--|:--|:--|
| 截斷位置 | LongBench、LongBench v2、InfiniteBench、SCBench：中間截；HELMET：尾端截（保留開頭，用 Llama-2 tokenizer）；LooGLE：頭尾拼接（〔複核補充〕效果等同中間截，只是截的是文件而不是整段 prompt）；RULER：不截，直接生成剛好長度；NIAH v1：截 haystack 並留 200 token buffer | LongBench p6、`pred.py` L60–62；v2 `pred.py` L28–29；∞Bench p7、`eval_utils.py` L413；SCBench `run_scbench.py` L116–118；HELMET p5、`data.py` L162–175；LooGLE p7；RULER `niah.py` L209–257；NIAH v1 `insert_needle` |
| 截斷上限 | LongBench：比名義視窗略小（3,500／7,500／15,500／31,500）；v2：120,000；∞Bench：128×1024 或 128,000−1,000 或 200,000（依模型）；SCBench：131,072 減輸出×輪數 | 各 config 與程式 |
| 長度單位 | 詞（LongBench、v2）、Llama-2 token 且 K=1024（HELMET）、受測模型 token 含生成（RULER）、tiktoken／模型 token（∞Bench、NIAH） | LongBench Table 1 p4；v2 p6；HELMET p2、p7；RULER `niah.py` |
| Chat template | LongBench 6 個補全式任務不套、`build_chat` 不認得新模型；Agrawal & Mayer 全部套（p6）；RULER 手寫模板＋answer prefix（Table 6 p20）；HELMET 同任務 HF 與 vLLM config 預設不同；SCBench 腳本預設套 | `pred.py` L21–42、L63；2607.05399 p6；RULER `template.py`；HELMET `configs/`；SCBench `run_all_tasks.sh` |
| Answer prefix | RULER 2025-01-22 到 2026-07-22 的獨立腳本遺失 prefix；推理模型應移除 prefix | RULER #107、#108、#99 |
| max_new_tokens | LongBench 32–512；v2 128（CoT 1,024）；RULER 30–128，新流程範例 4,096；∞Bench 3–30,000；SCBench 5–1,024；HELMET 10–1,200〔複核修正：原寫 20–1,200；`configs/longqa.yaml` 的 ∞Bench QA／MC 是 10（`generation_max_length: 100,10,10`），`HELMET@aeadacc6`〕 | 各 config |
| 取樣 | LongBench、RULER、SCBench、HELMET：greedy；LongBench v2：temperature 0.1（重跑不一致，#111）；∞Bench：API 預設（〔複核補充〕repo 的 Yi-200K／ChatGLM 腳本是 temperature 0.8、top_p 0.95 取樣，見 ∞Bench 卡）；ForesightKV：0.6／top-p 0.95，32 次平均；TRIM-KV：64／8 次平均 | LongBench p6；v2 p24；RULER `config_models.sh`；∞Bench p16；FKV p6；TRIM p7 |
| 評分腳本 | LongBench 只取第一行（4 個任務）、程式碼取第一個非註解行；Agrawal & Mayer 的 TriviaQA 用 EM（官方是 F1）；RULER 子字串比對無字邊界（#112）；NIAH v1 用 GPT-3.5 打分、v2 用子字串；HELMET、LongMemEval 用 GPT-4o 評審；LongBench v2 的 compensated 模式與「失敗題丟掉、分母變小」（#144） | `eval.py` L70–71；2607.05399 p6；RULER `constants.py` L24–30；NIAH 兩版程式；HELMET p5；LongMemEval p6；v2 `result.py` |
| 資料版本 | SCBench 論文 931／HF 922 列，輪數也不同；LongMemEval 2025-09 cleaned 版；HELMET 2025-02 改 ICL 為 500 筆；NIAH 2026-05 v2；RULER 舊流程 deprecated、改 NeMo-Skills；LongBench HF 載入腳本不被新版 `datasets` 支援 | 見各卡 |
| 推論引擎 | HELMET：vLLM 與 HF 結果略有差異，最終建議 HF；SCBench：vLLM 0.5.2＋自寫 kernel | HELMET README L224；SCBench p24 |
| 多輪歷史 | SCBench 預設用標準答案，不是模型輸出 | SCBench p8；MInference #154 |
| 隨機種子 | HELMET 2024-10 修 seed、2026-09 修示範抽樣；LongBench v2 多程序造成順序不同 | HELMET CHANGELOG、#43；LongBench #111 |
| 是否開 YaRN | LongBench v2 為公平，Qwen2.5 不開 YaRN，另表報開 YaRN 的結果 | LongBench v2 p6 |

---

## 使用者文件核對

| 文件位置 | 原句（改寫） | 核對結果 | 出處 |
|:--|:--|:--|:--|
| `intro.txt` L857（表 17） | SCBench：多輪、共用 context 的品質評測 | 正確。可補：它也有多請求模式，且不量時間 | SCBench §3.2 p8 |
| `intro.txt` L859（表 17） | RULER、LongBench v2：長度可控的合成任務；真實的長 context 任務 | 兩個描述本身對（RULER 合成、可控長度；LongBench v2 真實任務）。~~但表 17 的標題是「現有工作的負載合成方式」，把它們放進來是類別錯置~~〔複核修正：「類別錯置」說得太重〕。事實部分成立：表 17 把兩種東西放在同一欄而沒有區分，一種是系統論文的負載合成（Cake 只取長度、Mooncake 模擬快取比例、HCache 合成到達），一種是內容或品質 benchmark（SCBench、RULER、LongBench v2）；後者沒有到達時間或 session 結構（SCBench 也一樣，只有構造出來的重用）。但 intro 下一段說的做法是「真實 trace 的時間與 session 結構＋長 context 的內容」，所以把 RULER、LongBench v2 當成「內容從哪裡來」列進表裡是說得通的，原句也沒有說它們提供到達過程。建議只加一欄或註明「只提供內容」。另外補充：LongBench v2 長度**不可控**，只按詞數分 Short／Medium／Long，且各組任務組成不同（p7 註 ⋄）；原句「真實的長 context 任務」本身沒有說它可控，這一點不算錯 | RULER p2；LongBench v2 p6–7；intro.txt L849–865 |
| `intro.txt` L864–865 | 只量時間（TTFT）時可以用合成的 token；量品質時要用真實任務 | 〔複核修正：原判「只對無損且輸出長度固定的情況成立」不成立〕原句明寫「只量時間（TTFT）時」。TTFT 不含解碼，輸出長度不影響它；固定位元寬的 FP8／INT4 格式，搬移量與反量化成本也與內容無關。所以就 TTFT 而言，前半有 Cake p5 支持，不需要加「無損」「固定輸出長度」兩個限定。真正要補的限定是：(a) 若要把同一做法延伸到端到端延遲或吞吐，有損方法會改變回應長度，必須用真實內容並記錄回應長度（Rethinking p5 Missing Piece 2）；(b) 資料相依的方法（例如 MInference 依輸入動態決定稀疏 pattern，SCBench D.1 p22）連 prefill 計算量都隨內容變，這類方法的 TTFT 也不能用合成 token。後半「量品質時要用真實任務」：若讀成「品質評估要包含真實任務」，有 HELMET p7 支持（合成任務與下游相關都 <0.8），不算錯；若讀成「只能用真實任務」則太強，RULER p2 與 HELMET p7 都認為較難的合成檢索任務是有用的檢查。建議寫成「品質要含真實任務；合成任務只用不可壓縮的檢索類，不能只用 NIAH 或困惑度」 | Cake p5；Rethinking p5；SCBench p22；RULER p2；HELMET p7；SCBench p10 |
| `sota.txt` L396（§4.6） | SCBench 量共享 context 下的品質，但重用率是構造出來的 | 正確。建議補：(a) 多輪歷史用標準答案，不是模型輸出；(b) 官方腳本預設中段截斷到 131,072 token；(c) 沒有到達時間、不量延遲；(d) 「80% 重用」是 1−1/T 在 T≈5 的結果，不是可調參數 | SCBench p8；`run_all_tasks.sh`；MInference #154；〔計算〕 |
| 參考文獻 [58][59][60]（intro）、[32]（sota） | SCBench ICLR 2025；RULER COLM 2024；LongBench v2 ACL 2025（第 63 屆年會） | 皆正確 | 各 PDF 頁首；ACL Anthology `2025.acl-long.183` |

---

## 對 PoC 設計的建議〔判讀〕

整節為判讀。前提：Tiara 的 CPU、SSD 層是無損位元組搬移，DROP＋重算若重算出完全相同的 KV 也是無損；所以 ε 主要來自 GPU-FP8／GPU-INT4 兩個精度層，以及它們與位置、重用次數的組合。

1. **分四段，從便宜到貴。**
   * **A（長度掃描，主力）**：RULER 13 任務，在 16K、32K、64K、128K，以及模型支援時的 256K。每格 n≥100（正式版用 500），與 BF16 基準用**同一批樣本**配對比較。逐任務報差值，不要只報平均；另外報「有效長度的位移」。記錄 RULER commit，必須 ≥ `c3f5e3b4`（含 answer prefix 修正）。
   * **B（不可壓縮＋重用）**：SCBench 多輪與多請求，子集 Retr.KV、Retr.Prefix-Suffix、Retr.MultiHop、RepoQA、En.QA、En.MultiChoice。`--max_seq_length` 設成模型上限而不是預設的 131,072。逐輪報分，看量化 KV 被重用多輪後 ε 是否累積。若要更接近真實，另跑一組 `--disable_golden_context`。
   * **C（真實任務，≥128K）**：∞Bench En.QA、En.MC（平均約 190K token，可測 128K 以上）；LongBench v2 Medium＋Long（323 題）；HELMET RAG＋Recall 在 128K 當代理。
   * **D（只有在解碼期 KV 也分層時才需要）**：LongProc 2K／8K。
   * **回歸**：LongBench-E 確認短 context 沒退化。**不要**只用原始 NIAH 或 PG-19 困惑度當證據。
2. **長度上限要對齊。** 16K–64K：大多數 benchmark 都夠。128K：RULER、SCBench、∞Bench、HELMET、LongBench v2。256K–512K：只有 RULER（任意長度）、BABILong（256K、512K split）、LongBench v2 Long 組、∞Bench 的部分任務不截斷時、LongMemEval_M。模型視窗若是 262,144，512K 只能用 RULER 或 BABILong 測「超過視窗」的行為，而那已不是 KV 精度的問題。
3. **統一協定（寫進論文）。**
   * 長度一律用受測模型的 tokenizer 計，並註明 benchmark 原本的單位。
   * 截斷：優先只選長度 ≤ 視窗的樣本；必須截時用中間截並寫明。不同精度層在同一份截斷後的輸入上比較。
   * chat template 用 `tokenizer.apply_chat_template`；照 benchmark 原設計，補全式任務不套（LongBench 的 6 個任務）。
   * greedy、固定 seed、`max_new_tokens` 照官方 config。偏離官方之處（例如 LongBench v2 的 temperature 0.1 改成 0）要列出。
   * 記錄每題的回應長度：有損 KV 可能改變輸出長度，進而改變延遲（Rethinking p5）。
4. **樣本數要夠看出小的 ε。** 配對比較時，若有 5% 的題目答案翻轉，n=500 的差值標準誤約 1 pp（95% CI ±2 pp）；n=2,000 約 ±1 pp〔計算：√(0.05/n)〕。RULER 每長度 13×500＝6,500 題足夠；LongBench v2（503）與 SCBench（922 個 session）只夠看幾個 pp 的差。用 McNemar 檢定或配對 bootstrap，不要用兩組獨立的信賴區間。
5. **可以考慮的較靈敏指標。** 對 BF16 參考輸出的 token 一致率、第一個不同 token 的位置、在答案 token 上的 log-prob 差。LongPPL 的「只看關鍵 token」思路（Fang et al.）可以借用；但這些只能當輔助，主結論仍要用任務分數。
6. **ε 的定義可以借 NoLiMa 的相對門檻**：以 BF16 在同長度的分數為 base，ε＝相對下降；這比 RULER 的絕對門檻更適合跨模型比較。

---

## 未查證清單

1. NIAH 原始兩則推文（GPT-4-128K 2023-11-08、Claude 2.1 2023-11-21）的分析內容與結論，只讀了 repo。
2. Lost in the Middle（Liu et al. 2023）原文未讀；本檔中相關說法都是轉述 LongBench README、InfiniteBench、HELMET。
3. AIME 與 GSM8K 的題數；ForesightKV 準確率實驗的最大生成長度；TRIM-KV 的取樣溫度。
4. StreamingLLM 的困惑度用哪個語料（只讀到摘要）。
5. KV 文獻中使用 RULER、LongBench、L-Eval、LooGLE 的名單：直接引自 `workloads_eval.md`，本次未逐篇重查。HELMET、NoLiMa、∞Bench 是否被 KV 論文直接使用：未查到。BABILong 的 LASER-KV 只看了摘要。
6. L-Eval 各子任務的長度分布；LooGLE 長依賴 QA 的評分細節。
7. InfiniteBench HF 資料的 license（卡上沒標）。
8. SCBench Appendix F（關閉 golden context 的結果）未讀；datasets-server 的 `multi_turns` 統計是否就是每列輪數（依欄位型別推定為 list 長度）。
9. RULER `rulerv1-ns`（NeMo-Skills）流程除了移除 prefix 之外，還改了什麼（只讀 README）。
10. LongBench v2 線上 leaderboard 的設定。
11. 詞數與 token 的換算比例（各 benchmark 都沒給）。
12. 「時間用合成、品質用真任務」是否有論文同時明說。

---

## 複核紀錄

- **複核者**：V09（獨立複核子 agent，未參與抽取，未讀抽取者的推理或筆記）
- **日期**：2026-10-07
- **做法**
  * 論文：用抽取者下載的 PDF（`scratchpad/E09/<id>.pdf`），先核對 arXiv 編號、版本與日期都和〈來源清單〉一致（19 份皆一致），再自己用 `pdftotext -layout` 重轉成文字（`scratchpad/V09/`），頁碼以 PDF 頁為準。
  * 程式碼：自己重新 clone 到 `scratchpad/V09/repos/`。HEAD 與卡上一致：`THUDM/LongBench@2e00731f`、`NVIDIA/RULER@c3f5e3b4`（另讀 `origin/rulerv1-ns@e8bbff67` 與 commit `48cbc8b`、`defffc8`、`8ef9e90` 的歷史）、`gkamradt/LLMTest_NeedleInAHaystack@021385d6`（v1 讀 `7b90d28`）、`OpenBMB/InfiniteBench@51d9b37b`、`microsoft/MInference@29ef1974`、`princeton-nlp/HELMET@aeadacc6`。其餘 repo 以 GitHub API 確認 HEAD：LEval `cd34b050`、LooGLE `fb9aee3b`、babilong `7a6efee2`、LongProc `673ec4c2`、LongMemEval `9e0b455f`、NoLiMa `cb14780b`。
  * 資料與 license：HF API、datasets-server `/size`、`/statistics`（SCBench 12 個 config、LongBench-v2）、GitHub API license 欄位；issue 以 `gh api` 讀本文與留言（LongBench #26、#61、#111、#144；RULER #70、#99、#107、#108、#112；MInference #154、#200、#202；HELMET #35、#43）。
  * 使用者文件：`scratchpad/intro.txt` L849–865、L1120–1125；`scratchpad/sota.txt` L396、L522；`docs/research_20260924/workloads_eval.md` §3.3 L255–265。另讀 Tutti（arXiv 2605.03375）p9，追 LooGLE「>100K」的來源。

### 統計

| 區塊 | 檢查格數 | ✅ | ❌ | ⚠️ | 複核補充 |
|:--|--:|--:|--:|--:|--:|
| 來源清單 | 20 | 20 | 0 | 0 | 0 |
| 重點摘要 | 8 | 6 | 1 | 1 | 1 |
| LongBench v1／-E | 26 | 26 | 0 | 0 | 1 |
| LongBench v2 | 24 | 22 | 1 | 1 | 1 |
| RULER | 24 | 21 | 1 | 2 | 0 |
| NIAH | 15 | 14 | 1 | 0 | 0 |
| InfiniteBench | 23 | 21 | 2 | 0 | 1 |
| SCBench | 26 | 25 | 0 | 1 | 2 |
| HELMET | 24 | 24 | 0 | 0 | 1 |
| L-Eval | 10 | 9 | 1 | 0 | 0 |
| LooGLE | 11 | 10 | 1 | 0 | 2 |
| BABILong | 10 | 10 | 0 | 0 | 0 |
| LongProc | 9 | 9 | 0 | 0 | 0 |
| LongMemEval | 9 | 8 | 0 | 1 | 0 |
| NoLiMa | 9 | 9 | 0 | 0 | 0 |
| PG-19／WikiText 困惑度 | 7 | 7 | 0 | 0 | 0 |
| 長輸出推理題 | 9 | 8 | 1 | 0 | 0 |
| 選擇準則表 | 9 | 9 | 0 | 0 | 0 |
| 「只量時間可用合成 token」出處查證 | 3 | 3 | 0 | 0 | 1 |
| 實作差異表 | 15 | 14 | 1 | 0 | 2 |
| 使用者文件核對 | 5 | 3 | 1 | 1 | 0 |
| **合計** | **296** | **278** | **11** | **7** | **12** |

「複核補充」是抽取者漏掉、但對評測設定重要的事實，不算錯誤。❌ 中有 6 格只是頁碼或表號錯（RULER 表號、NIAH、∞Bench 兩格、L-Eval、TRIM-KV）。

### 逐條修改（原內容 → 新內容，出處）

**❌ 錯誤**

1. 〈使用者文件核對〉intro L864–865：「前半只對無損且輸出長度固定的情況成立」→ 指控不成立。原句明寫「只量時間（TTFT）時」；TTFT 不含解碼，輸出長度不影響它，固定位元寬的 FP8／INT4 格式也不讓 TTFT 隨內容變。改成：對 TTFT 成立，只需排除資料相依的方法（MInference，SCBench D.1 p22）；延伸到端到端延遲時才要加 Rethinking p5 的限制。
2. 〈重點摘要〉#6：同上，「但只對無損、輸出長度固定的情況成立」→ 拆成 TTFT 與端到端兩種條件。
3. LongBench v2〈陷阱〉(b)：「issue #111 回報與論文差到 3.3 pp」→ 最大差 4.6 pp（c4ai-command-r-plus Long 組；GPT-4o-2024-08-06 Medium 組 −3.3）。出處：issue #111 本文的表格。
4. 〈實作差異表〉max_new_tokens：「HELMET 20–1,200」→「10–1,200」。出處：`HELMET@aeadacc6` `configs/longqa.yaml`（`generation_max_length: 100,10,10`，∞Bench QA／MC）。
5. LooGLE〈與既有整理不一致〉：「README 只說 many of which exceed 100k words」→ 論文 p2 本身就有這句，README L21 照抄；workloads_eval 標的 Tutti p9 寫的是 "many test samples exceeding 100k tokens"。所以錯在把 many 寫成「多數」，並沿用 Tutti 把詞換成 token；Table 2 p4 的平均 token 是 20,887–36,412，Wikipedia、劇本的最大詞數也只有 46,250、62,752。
6. RULER〈模板〉：「任務模板 Table 7–8 p21–22」→「Table 7–9 p21–23」（p20 原文寫 "Table 7 8 9"；QA 模板是 Table 9 p23）。
7. NIAH〈批評〉：LongBench v2「NIAH 已完美」的頁碼 p2 → p1。
8. InfiniteBench〈解碼〉：Appendix D p16 → p15（Table 5 才在 p16）。
9. InfiniteBench〈參考模型〉：Appendix D p16 → p15（A100 80GB、每例約 10 分鐘在 p15）。
10. L-Eval〈長度範圍〉：Llama-2 tokenizer 的註在 Table 1 p5，不是 p4。
11. 長輸出推理題〈KV 預算〉：TRIM-KV 同預算比較（AIME24 512、GSM8K／MATH-500 128）在 p8，不是 p7。

**⚠️ 不精確或說得太重**

1. 〈使用者文件核對〉intro L859（表 17）：「把它們放進來是類別錯置」→ 說得太重。事實部分成立：表 17 把系統論文的負載合成和品質 benchmark 放在同一欄，沒有區分。但 intro 下一段的做法本來就是「真實 trace 的時間與 session 結構＋長 context 的內容」，所以把 RULER、LongBench v2 當成內容來源列進表裡說得通。改成建議註明「只提供內容」。〈重點摘要〉#8 與 LongBench v2〈與既有整理不一致〉同步修改。
2. 〈重點摘要〉#8：同上，並改正 TTFT 的限定。
3. 〈使用者文件核對〉intro L864–865 後半「量品質時要用真實任務」被判「太強」：要看怎麼讀。讀成「品質要包含真實任務」有 HELMET p7 支持；讀成「只能用真實任務」才太強。已改寫成兩種讀法並列（這一格的主要錯誤已計入 ❌1）。
4. RULER〈長度〉：「先扣模板 token，再二分搜尋」→ 扣模板 token 只發生在 NeMo-Skills 模式（`prepare.py` L90–94、`niah.py` L212）；獨立腳本是把 chat 模板與 answer prefix 併進 template 一起計數（`prepare.py` L101）。兩條路徑都含模板與生成 token。
5. RULER〈needle 位置〉：「從 0–100% 等距 40 個深度中隨機抽」只適用 essay haystack（`niah.py` L147–157）；noise 與 needle haystack 是在句子索引上均勻隨機插入（L168–182）。
6. LongBench v2〈設計理由（原文）〉：頁碼 p2–3 → p1–3。
7. SCBench〈設計理由（原文）〉：「多輪用標準答案是為了避免錯誤累積（p8；#154）」→ 理由只出現在 #154 的維護者回覆；p8 只說依循 Zheng et al. 2023a、Wang et al. 2024。
8. LongMemEval〈版本〉：「與原版不可比」補標〔判讀〕（README 只說清理了歷史 session）。

（⚠️ 共 7 格計入統計；第 3 條與 ❌1 同一格，不重複計。）

**複核補充（新增事實）**

1. LongBench v1：HF repo 只有 `LongBench.py`、`README.md`、`data.zip`；`datasets` 4.0.0（2025-07-09）移除了載入腳本支援（PR #7592）。
2. LongBench v2：`pred.py` L130–139 的快取在重跑時會補跑缺的題，但每次都失敗的題會一直缺；模型回傳空字串也會被當成失敗丟掉。
3. InfiniteBench：repo 的 `eval_yi_200k.py`、`eval_chatglm.py` 用 vLLM `SamplingParams(temperature=0.8, top_p=0.95)` 取樣，而且沒有把任務的輸出上限傳進去，結果落到 vLLM 預設 `max_tokens=16`（vLLM v0.4.0 `sampling_params.py` L117）。論文主表的 4 個模型不走這兩支腳本。
4. SCBench〈長度〉：Table 2 沒有標單位，推定為 token〔判讀〕；字元數範圍只涵蓋有字串統計的 config。
5. SCBench〈與既有整理不一致〉：299K–3.17M 字元也正好是 En.MultiChoice 的範圍。
6. HELMET：MIT 是 GitHub repo 的 license；HF 資料卡沒有 license 標籤。
7. LooGLE〈任務型態〉：GPT3.5-turbo-16k 生成短依賴 QA 與 cloze 的摘要（§3.3 p6）。
8. LooGLE〈截斷〉、〈實作差異表〉、〈重點摘要〉#2：頭尾拼接在效果上就是中間截斷。
9. 「只量時間可用合成 token」出處查證：判讀欄補上 TTFT 與端到端兩種條件的區分。

### 指控核對結果（使用者文件與 workloads_eval）

| 指控 | 判定 | 依據 |
|:--|:--|:--|
| intro 表 17 把 RULER、LongBench v2 列成「負載合成方式」是類別錯置 | ⚠️ 部分成立，措辭降級為「建議註明只提供內容」 | 兩者確實沒有到達時間或 session（RULER p2；LongBench v2 p6–7）；但 intro L864 的做法本來就要「長 context 的內容」 |
| 「只量時間（TTFT）時可用合成 token」只在無損且固定輸出長度時成立 | ❌ 指控不成立（就 TTFT 而言） | Cake p5；輸出長度不進入 TTFT。限定只在端到端延遲時需要（Rethinking p5），另外要排除資料相依的稀疏 prefill（SCBench D.1 p22） |
| 「量品質時要用真實任務」太強 | ⚠️ 看讀法 | HELMET p7 支持「要包含真實任務」；RULER p2、HELMET p7 反對「只能用真實任務」 |
| sota L396「SCBench 重用率是構造出來的」 | ✅ 原句正確（卡上沒有指控，只建議補充） | SCBench p5、p8；#154 |
| workloads_eval：LooGLE「多數 >100K」 | ✅ 指控成立 | LooGLE Table 2 p4 的平均 token 是 20,887–36,412；源頭 Tutti p9 寫 "many"，不是「多數」 |
| workloads_eval：SCBench「context 299K–3.17M 字元」只是英文 QA 的範圍 | ✅ 指控成立 | datasets-server：`scbench_qa_eng` 298,903–3,171,853（`scbench_choice_eng` 相同）；全體 28,207–6,556,639 |
| workloads_eval：SCBench 922 列 vs 論文 931 sessions | ✅ 成立 | 論文 p5、Table 2 p6；HF `/size` 922（`scbench_summary` 70 vs 論文 79） |
| workloads_eval：「80% 重用」不是 SCBench 原文數字 | ✅ 成立 | 全文找不到 80% 或 reuse rate；4,853／931＝5.21 輪，1−1/5.21≈80.8%〔計算〕 |
| workloads_eval：LongBench「5K–15K 字」沒註明單位 | ✅ 成立（精確度問題，不是錯誤） | `LongBench/task.md` 註：英文與程式碼算詞，中文算字 |
| 參考文獻 [58][59][60]（intro）、[32]（sota） | ✅ 正確 | PDF 頁首；ACL Anthology `2025.acl-long.183` 為第 63 屆 ACL |

### 抽取者的重點發現（協調者指定逐條確認）

- 長度單位：LongBench／v2 用詞（中文用字）、HELMET 用 Llama-2 token 且 K＝1024、RULER 用受測模型 token 並含模板與輸出：✅（LongBench Table 1 p4、`task.md`；v2 p6；HELMET p2 腳註 2、p7；RULER `niah.py` L245、L268）。
- 截斷方式不同：✅（LooGLE 那一項補充為等同中間截斷）。
- SCBench：多輪歷史用 gold answer ✅（p8、#154）；官方腳本截到 131,072 ✅（`run_all_tasks.sh`、`args.py` L55）；不量時間 ✅（全文沒有任何延遲量測）；論文 931 vs HF 922 ✅。
- RULER 獨立腳本在 2025-01-22（`48cbc8b`）到 2026-07-22（PR #108 合併為 `c3f5e3b`）之間少了 answer prefix：✅（git 歷史；#107 作者 hsiehjackson 確認）。另外 2025-01-16 的 `defffc8` 在非 NeMo-Skills 模式下會在 `split("<answer_prefix>")[1]` 出錯，那 6 天的獨立腳本可能根本跑不起來〔判讀，未實跑〕。
- LongBench v2 會默默丟掉失敗的題：✅（`pred.py` L98–99、L105–106；`result.py` L37；#144）。

### 因時間沒有檢查的部分（如實列出）

1. 各卡「KV 文獻誰用過」中標〔二手：workloads_eval〕的名單（LongBench 15 篇、RULER 6 篇、L-Eval、LooGLE 的使用者）沒有逐篇回 KV 論文原文查；只確認了 TRIM-KV、HELMET、SCBench、LASER-KV（摘要）四筆。
2. 〈對 PoC 設計的建議〉整節是判讀，只核對了其中的數字（323 題、信賴區間公式、13×500、`c3f5e3b4`），沒有評論建議本身。
3. 〈未查證清單〉12 條沒有嘗試補查。
4. HELMET 附錄 E.4–E.6、L-Eval §4、LongProc §3、BABILong §2–3 只抽查卡上引用的句子，沒有通讀。
5. NIAH v2 的 `tasks/single_needle.py` 沒有逐行讀，只讀了 `scorers/exact_match.py` 與 README。
6. SCBench Appendix F（關閉 golden context 的結果）沒有讀，與抽取者相同。
7. Agrawal & Mayer「所有任務都套 chat 模板」：p6 只寫 prompt 用各模型的 chat 模板，沒有列例外。判 ✅，但這是讀法。
8. `docs/PAPERS_BY_LEVEL.md` 沒有比對（卡上也沒有引用）。
9. 沒有實際執行任何 benchmark 腳本；程式行為都是讀原始碼推得。
