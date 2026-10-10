# D5 新架構與新模態：VLM／影片、混合 SSM、滑動視窗、P/D 分離、MLA

**日期**：2026-10-10（判準寫於 05:35:55Z，本文完成約 06:10Z）
**作者**：D5 agent（**沒有用 GPU**；全部是算術、讀程式碼、讀論文）
**證據標記**：〔算術 run_id〕自己用公式算、可重跑；〔原文 p.X〕PDF 實體頁；〔程式碼 file:line〕；〔判讀〕推論；〔未查證〕。沒量到的寫 NOT_MEASURED。
**主要產出**：`code/m8_arch_arith.py`；`results/m8_directions/d5_{calib_check,vlm,ssm,swa,pd,mla}.csv`（全部 run_id `20261010-055012-d5-arith`）；卡片 `cards/D5_*.md`（22 張，見 §5.6）。
**上游（不重做）**：[Lit-C](../research_20261009_explore/lit_C_architectures.md)、[FULL_REPORT](../phase1_20261008/FULL_REPORT.md)、[10 §2 延後測試](../phase1_20261008/10_breakthrough_plan.md)。

---

## 0. 一句話結論＋判定

**總結**：五個子方向，沒有一個「有看頭」。只有 (b) 混合 SSM 留下「可能」，而且很窄。其餘四個判死路，理由各不相同：
- 延後版或寫穿版很便宜：(a)
- 已經有人做了：(c)
- 管線傳輸早就把傳輸時間藏起來了：(d)
- 載入太便宜，沒東西可以決定：(e)

| 子方向 | 判定 | 最重要的一個理由 |
|:--|:--|:--|
| **(a) VLM／影片**：寫入時選格式（存 KV、存 vision embedding，還是只留像素） | **死路** | 「embedding 也一起存」只多 **22%** 位元組，這就是寫穿版，而且 VLCache、LMCache、Dynamo 已經這樣做〔算術、原文〕。Qwen3-VL-8B 的 encoder 只占重算成本的 14–23%，存 embedding 只讓 Cake 還原快 **6.5–8.5%**（3.69 GiB/s、32K、一般圖或影格）〔算術 20261010-055012-d5-arith〕 |
| **(b) 混合 SSM／線性注意力**：檢查點的分層放置 | **可能（窄）** | 機制已經有了：SGLang main 會把 Mamba 狀態放在 GPU／host／storage 三層，vLLM main 預設只保留 Marconi 式的語意檢查點〔程式碼〕。還沒人做的只剩「每一層用不同的密度」，而且只有分叉型負載才用得到。數字上它站得住：每 512 token 存一份時，檢查點占快取 **51–90%** 的位元組（7 個模型有 6 個 ≥75%）；全部寫到 SSD 要 **0.17–1.09 倍**本地 SSD 的寫入頻寬（7 個有 5 個 ≥0.9 倍）〔算術〕。但相對 Sparse Prefix Caching＋引擎現況，只是增量 |
| **(c) 滑動視窗混合**：只存全域層 | **死路** | 浪費確實很大：Gemma-3-12B 在 32K 時 **81%** 的位元組用不到〔算術〕。但「分層感知卸載」SGLang 已經做了（#23391），vLLM main 也做了（#51886）。延後版（搬移時才丟）也拿得到同樣的節省。剩下 vLLM 對 Gemma 退回全存，這是工程修補，不是研究 |
| **(d) P/D 分離**：Cake 跨網路 | **死路** | 判準寫在 ≥25 Gbps。在那裡，逐 chunk 邊算邊傳只會露出 **0.6%** 的傳輸時間；Cake 分割省下 0%（decode GPU 半速）〔算術〕。Cake 分割要等每條 stream 分到的頻寬 ≤8 Gbps 才有用（**15–67%**），那是判準以外的區域 |
| **(e) MLA／MQA** | **死路** | 3.69 GiB/s 時 b/n：DeepSeek-V3 是 **0.047**，Kimi-K2 是 0.062，Falcon-7B 是 0.031〔算術，和 Lit-C 一致〕。例外是小模型 DeepSeek-V2-Lite（0.23）。MLA 只是把 Cake 的好用區間往慢的方向移到約 0.3 GiB/s，沒有新機制 |

**三個關鍵數字**
1. Qwen3-VL-8B：encoder 成本 ÷ LLM 重算成本 r_v＝0.14（448² 影格）、0.23（1024² 圖），32K 平均〔算術 20261010-055012-d5-arith〕。VLCache 在 H20 上實測同模型是 0.16（20K）到 0.37（1K）〔原文 VLCache p.7 表 1，D5 親自核對〕，同一個量級。
2. Nemotron-H-8B：一份檢查點 49.4 MiB，是一個 512-token chunk 注意力 KV 的 **6.2 倍**。用 vLLM `all` 模式時，檢查點占快取 83% 的位元組。以 prefill 速度全部寫出要 1.42 GiB/s，等於本地 SSD 寫入頻寬（1.91 GiB/s）的 0.75 倍〔算術〕。
3. P/D、Llama-3.1-8B、32K、每條 stream 25 Gbps：不重疊時傳輸露出 **27.5%**，逐 chunk 管線只露出 **0.59%**〔算術，用實測 f(i)〕。vLLM 0.28 的 NixlConnector **不做逐層傳輸**〔程式碼 `nixl/connector.py` L281–292〕，但修法是管線化，不是 Cake。

**跨方向的一個觀察**〔判讀〕：(b) 的 SSM 中間狀態和 (c) 的局部層視窗 KV 是同一種東西：「要在位置 L 接著算，一定要有的那一點狀態」。我叫它「命中點狀態」。
- 這種狀態在 GPU 上一被覆蓋或釋放就沒了（N1）。
- 引擎在 2026 年已經收斂到同一個做法：只留語意命中點，也就是 replay 邊界與分歧點（vLLM `prefix_cache_retention_interval`，預設 0）。
- 唯一還沒人做的是「**每一層用不同密度**」（GPU 疏、SSD 密，或反過來），並在寫入頻寬有限時做取捨。
- 這是 D5 唯一建議保留的線，條件是 D1（沒有空檔）與 D4（分叉負載有多常見）的結果支持它。

---

## 1. 問題是什麼、要證明什麼

**問題**：第一階段只測了 Llama-3.1-8B（GQA、純文字）。在它身上，「寫入時依位置放」被延後版追平了。換成別的模型或模態，會不會出現**延後版做不到**的東西？延後版做不到，只有四種原因：
- N1：資訊寫完就消失；
- N2：延後要多付稀缺資源；
- N3：沒有空檔；
- N4：寫成什麼格式，決定了讀取時能做什麼。

**要證明（或殺掉）的**，每個子方向一件事：
- (a) VLM／影片：重算視覺 token 要多付 vision encoder。存 embedding 這個「格式選擇」，能不能讓還原快到值得在寫入時決定？延後版或寫穿版會不會一樣好？
- (b) 混合 SSM：中間位置的遞迴狀態寫完就沒了（N1）。這些狀態**放哪一層**有沒有決定可做？有沒有人做過？
- (c) 滑動視窗：局部層的舊 KV 用不到。浪費有多大？vLLM 真的會全存嗎？有沒有人做過分層感知的卸載？
- (d) P/D 分離：prefill 節點傳 KV 給 decode 節點，能不能用 Cake 的方式「一邊傳、一邊在 decode 端重算」來省時間？
- (e) MLA／MQA：確認 b≈0（載入遠比重算便宜），看還有沒有剩下的東西。

**名詞白話**：
- **b/n**：Cake 還原時，前段由 GPU 重算的 chunk 比例。越大，「位置」越重要。
- **κ**：載入一個 chunk 的時間 ÷ 重算它的時間。
- **檢查點**：SSM 層在某個位置的遞迴狀態快照。
- **HMA**：vLLM 的混合 KV 管理器，會依層型分開管理 KV，例如丟掉視窗外的 KV。

---

## 2. 事先寫好的判準（PRE-REGISTERED）

**寫入時間**：2026-10-10T05:35:55Z（`date -u`；在任何算術、下載、文獻查詢之前）
**作者**：D5 agent（不碰 GPU）
**三級判定**：有看頭／可能／死路。
**共同規則**：寫入時（或提早）決定的想法，一律要先過延後測試（10 §2）：延後版、寫穿版、背景版、讀取時版。只有「延後版做不到」的部分才算寫入時的價值。不是寫入時的想法（例如讀取時的排程），照實寫成「某某規則」，不硬說成寫入時。

所有預測都是〔算術〕。所用的 f(i) 一律來自 `results/m7_write_policy_mi300x/calib_c1.csv`（run_id `20261008-130316-m7-c1`）＋Lit-C 的兩參數校準形式（α·線性 FLOPs＋β·注意力 FLOPs）；本 agent 會自己重做這個校準並核對 Lit-C 的 Llama b/n（0.797／0.578／0.312／0.141／0.047），核對不過就停下來回報。

### (a) VLM／影片
量：Qwen3-VL-8B（從 HF 下載 config 自己算），r_v＝每個視覺 token 的 vision encoder 成本 ÷ 同 token 的 LLM prefill 成本（32K 平均位置）；e＝embedding bytes ÷ KV bytes；ΔT＝「重算線要不要付 encoder」對 Cake 還原時間的影響（32K、CPU 3.69 GiB/s 與 SSD 類頻寬 0.3–4 GiB/s）。
- **死路**：以下任一成立
  - ΔT（存 embedding vs 只留像素、讀取時重跑 encoder）在所有頻寬都 <5%；或
  - 延後版（「KV＋embedding 都存，之後才丟 KV」或「保留原始影片／圖片檔，之後重算」）在成本上追平寫入時選格式（判斷：多付的寫入量 ≤25% 且沒有稀缺資源）；或
  - 找到前作已經做了「VLM 的 KV／embedding／像素多格式分層＋讀取時並行還原」並和延後版比過。
- **有看頭**：ΔT ≥15% 且延後版需要多付稀缺資源（N2）或結構上做不到（N1／N4），且前作沒做。
- **可能**：介於兩者之間（例如 ΔT 5–15%，或只有讀取時排程有價值、不是寫入時）。

### (b) 混合 SSM／線性注意力
量：每份檢查點的狀態大小 S、每個 512-token chunk 的注意力 KV 大小 K；在常見間隔（vLLM `all` 模式的 block、SGLang 預設 8192）下檢查點占快取總位元組的比例；prefill 時「全部檢查點都寫出去」需要的寫入頻寬 vs 第一階段量到的 SSD／NFS 寫入頻寬；讀一份檢查點 vs 重算兩個檢查點之間那段的時間比 κ_ckpt。
- **死路**：以下任一成立
  - 找到 2025–2026 的論文或系統已經做了「SSM 檢查點的多層（GPU／CPU／SSD）放置＋寫入時選擇」；或
  - 在 SGLang 預設間隔下檢查點 <20% 的快取位元組，**而且**全部寫出去需要的頻寬 <50% 的 SSD 寫入頻寬（沒有稀缺資源，延後版「全寫、之後再丟」免費）。
- **有看頭**：檢查點占快取位元組 ≥50%（在至少一種實際的間隔設定下），全部寫出去的頻寬 ≥ SSD 寫入頻寬（N2 真的稀缺），有一個真實負載（分叉／部分重疊）需要中間狀態，且前作沒做多層放置。
- **可能**：其餘。

### (c) 滑動視窗混合（Gemma-3、gpt-oss）
量：32K、128K 時「全部層都存」vs「只存全域層＋局部層最後 W 個 token」的位元組差（浪費比例）；這個差對 Cake 還原時間的影響；vLLM 本機 build 與 vLLM 0.28 的程式碼行號確認。
- **死路（寫入時的說法）**：延後版（搬到慢層時才丟局部層的舊 KV）能拿到同樣的節省——依 Lit-C 的結構判讀預期會成立；成立就把「寫入時」判死路，另外評估「分層感知卸載」本身（不是寫入時）。
- **分層感知卸載本身**：浪費 ≥50% 位元組、而且 LMCache／SGLang／vLLM 0.28 都還沒做 → 有看頭（但標明是工程修正，不是寫入時決定）；已經有人做 → 死路；浪費 20–50% 或只有部分系統做了 → 可能。

### (d) P/D 分離（H11）
量：Llama-3.1-8B、32K，用實測 f(i)；網路 25／50／100／200／400 Gbps（每 GPU）；比較三種：(1) prefill 全部算完才傳（不重疊）；(2) prefill 邊算邊傳（逐 chunk 管線，Mooncake／DistServe 式）；(3) Cake 式：decode 端重算前段、同時收後段。看 (3) 比 (2) 省多少 decode 端 TTFT。
- **死路**：以下任一成立
  - 在 ≥25 Gbps 時，(2) 的「沒被蓋住的傳輸時間」≤ 端到端時間的 5%（管線已經把傳輸藏起來，沒東西可以省）；或
  - 找到前作已經做了「P/D 之間部分傳、部分在 decode 端重算」。
- **有看頭**：至少一個 25–400 Gbps 的頻寬下 (3) 比 (2) 快 ≥10%，且前作沒做。
- **可能**：其餘（例如只在 <25 Gbps 才有好處，或只在 prefix 快取命中、KV 從遠端 store 拉的情境才有好處）。

### (e) MLA／MQA
- **死路**：DeepSeek-V3、Kimi-K2、Falcon-7B 用校準後的 f 算，3.69 GiB/s 時 b/n ≤0.10。
- 若 DeepSeek-V2-Lite（小模型）b/n >0.10，照實寫出「小的 MLA 模型例外」，不改判定，只記錄。

---

## 3. 做了什麼

| 步驟 | 內容 | run_id／出處 |
|:--|:--|:--|
| 1 | 只下載 Qwen3-VL-8B-Instruct 的 config（`config.json`、`preprocessor_config.json`、`video_preprocessor_config.json`、`generation_config.json`）與 HF API 的檔案清單。**沒下權重** | `20261010-053802-d5-qwen3vl-cfg` |
| 2 | 只下載 12 個模型的 `config.json`（Gemma-3-12B/27B、gpt-oss-20b/120b、Nemotron-H-8B、Qwen3-Next、Falcon-H1-7B、Granite-4.0-H-Small、DeepSeek-V2-Lite／V3、Kimi-K2、Falcon-7B） | `20261010-054103-d5-cfgs` |
| 3 | 寫 `code/m8_arch_arith.py`（不 import torch）：重做 Lit-C 的 α、β 校準，接著算 (a)–(e) | 最終版 `20261010-055012-d5-arith`；之前 5 次見 §7 |
| 4 | 讀本機 vLLM 0.19.1（`/mlsteam/workspace/src/vllm`，commit `b1388b1`）與 0.28.0（`/mlsteam/workspace/src/vllm-v0.28.0`，tag v0.28.0，commit `2cf0a6915c`）的原始碼 | 〔程式碼〕 |
| 5 | 開 3 個文獻小幫手並行（VLM、SSM＋SWA、P/D），每篇都打開原文；我再親自抽查關鍵的一篇（VLCache 表 1）與 vLLM main 的關鍵程式碼 | 卡片 §5.6 |

**校準核對（通過）**〔算術 20261010-055012-d5-arith，`d5_calib_check.csv`〕：
- α＝3.956e-12 ms/FLOP，β＝6.459e-12 ms/FLOP。Lit-C 是 3.96e-12 與 6.46e-12。
- Llama b/n 在 0.331／1.0／3.69／11.64／35.4 GiB/s 是 0.797／0.578／0.312／0.141／0.047。和 Lit-C 的五個值逐一相同；實測 f 和預測 f 也相同。
- 用 Lit-C 表的 GFLOP 欄重建 6 個模型的 f，b/n 和 Lit-C 表全部一致到小數第二位（`d5_mla.csv`）。
- Qwen3-VL-8B 參數交叉驗算（規則 6 的精神）：用公式算出 vision 0.576B＋文字 8.191B，合計 8,767,123,696。HF API 的 safetensors 總數也是 8,767,123,696，**完全相同**〔算術；HF API〕。

---

## 4. 結果

### 4(a) VLM／影片（Qwen3-VL-8B）

**大小與成本**〔算術 20261010-055012-d5-arith；config 來自 run 20261010-053802-d5-qwen3vl-cfg；encoder 結構依 transformers 5.17 `models/qwen3_vl/modeling_qwen3_vl.py` L75–97、L174–186、L642–745〕：

| 項目 | 每個 LLM 視覺 token |
|:--|--:|
| KV（36 層 × 8 KV head × 128 × 2 × BF16） | 144 KiB |
| vision embedding（主 embedding＋3 份 DeepStack，各 4096 維 BF16） | 32 KiB（KV 的 **22.2%**） |
| 原始像素（2 格 × 3 色 × 32×32，uint8，未壓縮） | 6 KiB（KV 的 4.2%） |
| LLM 線性 FLOPs | 13.89 GFLOP |
| encoder 線性 FLOPs（4 個 patch × 27 層＋4 個 merger） | 3.62 GFLOP |
| encoder 注意力 FLOPs（每格自己做全注意力，`vision_utils.py` L42–65） | 0.39（448² 影格）／2.04（1024²）／8.15（2048²）／32.6（4096²，上限）GFLOP |

**r_v 與 Cake 還原時間**（32K＝64 chunk，全部是視覺 token，是 encoder 影響的上限；「分開」＝embedding 放在另一條更快的通道；「共用」＝embedding 和 KV 走同一條 I/O）：

| 場景 | r_v | 3.69 GiB/s：存 embedding 省多少（分開／共用） | NFS 0.33：分開／共用 | 本地 SSD 6.98：分開 | CPU 35.4：分開 |
|:--|--:|:--|:--|--:|--:|
| 448² 影格 | 0.14 | **6.5%**／−1.9% | 10.6%／−15.8% | 3.8% | 0.5% |
| 1024² 圖 | 0.23 | **8.5%**／0.2% | 15.5%／−9.4% | 5.7% | 2.1% |
| 2048² 圖 | 0.57 | 15.7%／8.1% | 29.7%／9.0% | 10.7% | 3.6% |
| 4096² 圖（上限） | 1.92 | 24.6%／17.7% | 53.6%／39.9% | 16.7% | 5.1% |

（12K＝24 chunk 時，3.69 GiB/s 的「分開」依序是 9.3%／9.3%／18.9%／29.9%。）

讀表〔判讀〕：
- 一般的圖與影格（≤1024²），存 embedding 只省 4–9%（CPU 到 SSD 的頻寬）。
- **embedding 和 KV 走同一條 I/O 時，可能更慢**。讀 embedding 要佔頻寬（KV 的 22%），比重跑 encoder 還貴。
- 只有超大圖（≥2048²）或很慢的層（NFS）才到 15%。
- 外部實測對得上：VLCache 在 H20 上量 Qwen3-VL-8B，ViT ÷ LLM prefill＝0.106 s ÷ 0.286 s＝0.37（1K token）、1.219 ÷ 7.565＝0.16（20K）〔原文 VLCache p.7 表 1，D5 親自核對〕。我的 r_v 在 12K 是 0.22–0.35，在 32K 是 0.14–0.23。
- **模型差很多**：同一篇量到 Qwen2.5-VL-7B 在 20K 是 56.5 s ÷ 6.25 s＝9.0〔原文 VLCache p.13 表 6，小幫手讀，計算＝Origin − w/o ViT〕。Qwen2.5-VL 上存 embedding 的價值大得多。但它照樣是「多存 12.5%、寫穿就好」（Lit-C：embedding 是 KV 的 12.5%），不需要寫入時做決定。

**影片的特別之處**〔config＋程式碼〕：
- Qwen3-VL 的預設影片處理最多 768 格（`video_processing_qwen3_vl.py` L123–125）。總像素上限 25,165,824（`video_preprocessor_config.json`）。所以**一支影片最多 12,288 個視覺 token，KV 約 1.69 GiB**〔算術〕，不是「一次寫幾十 GiB」。
- 要長影片，就要換成 ReKV 這類串流系統。那裡的讀取是「依問題挑散落的格」，不是連續前綴，Cake 的前後切分不成立（V01 綜合判讀 3）。

**N1 有沒有**〔判讀〕：
- encoder 輸出只在寫入時存在，但隨時可以從像素重算。
- OpenAI 式無狀態 API 每輪都會重送圖片，所以讀取時像素在手上。
- 即使是串流攝影機，存壓縮後的影格也遠小於 KV（未壓縮都只有 4.2%）。
- 所以沒有真正的 N1。只剩 N4 的格式選擇，但它的最佳解是「embedding 也一起存」，這是寫穿版。

**判定：死路**。對應判準的死路條件 2：延後版或寫穿版「KV＋embedding 都存」只多付 22.2% 寫入量（≤25%），而且沒有稀缺資源。條件 1 不成立（ΔT 在 NFS 有 10.6%）。條件 3 也不完全成立：前作有拼圖，但沒有完整做過（§5.1）。依判準，任一條成立就判死路。

### 4(b) 混合 SSM／線性注意力

**狀態 vs KV**〔算術 20261010-055012-d5-arith，`d5_ssm.csv`；狀態大小取 Lit-C 表，其中 Nemotron-H-8B（49.4 MiB）與 Qwen3-Next（37.7 MiB）由 D5 從 config 加 vLLM 公式重算，程式裡有 assert 確認一致，公式見 `vllm/model_executor/layers/mamba/mamba_utils.py` L127–151、L177–200〕

| 模型 | 一份檢查點 | ÷ 一個 chunk 的 KV | vLLM `all` 模式 block | 檢查點占位元組（block／每 512／SGLang 8192） | 全部寫出要的頻寬（block／每 512） | ÷ 本地 SSD 寫入 1.91 GiB/s（每 512） |
|:--|--:|--:|--:|:--|:--|--:|
| Jamba-1.5-Mini | 8.3 MiB | 1.0× | 1024 | 34%／51%／6% | 0.23／0.31 GiB/s | 0.17 |
| Falcon-H1-7B | 66.9 MiB | 3.0× | 1792 | 47%／75%／16% | 0.94／2.03 | 1.06 |
| Nemotron-H-8B | 49.4 MiB | 6.2× | 640 | **83%**／86%／28% | **1.42**／1.72 | 0.90 |
| Nemotron-Nano-9B-v2 | 69.4 MiB | 8.7× | 768 | 85%／90%／35% | 1.46／2.08 | 1.09 |
| Granite-4.0-H-Small | 73.7 MiB | 9.2× | 768 | 86%／90%／37% | 1.45／2.07 | 1.09 |
| Qwen3-Next-80B-A3B | 37.7 MiB | 3.1× | 576 | 74%／76%／16% | 1.86／2.03 | 1.06 |
| Kimi-Linear-48B-A3B | 21.4 MiB | 5.4× | 1024 | 73%／84%／25% | 0.75／1.30 | 0.68 |

- vLLM `all` 模式的 block 大小依 `vllm/model_executor/models/config.py` L244–263 算：讓一頁注意力 KV ≥ 一層的 mamba 狀態，再向上取到 mamba chunk 的倍數。GPU 上 mamba 狀態會補齊到頁大小，補齊後占比是 50–90%（`ckpt_frac_bytes_vllm_padded` 欄）。
- 寫出頻寬＝(S/間隔＋KV/token)×單一 stream 的 prefill 速度（10–27K token/s，用 Lit-C 的 FLOPs 和 Llama 的 MFU 算，**偏樂觀**）。多個請求同時 prefill 時要乘上並行數〔判讀〕。
- 讀一份檢查點：本地 SSD 1–10 ms、vLLM 量到的 CPU 3.69 GiB/s 2–20 ms、NFS 24–217 ms。重算一個 chunk 要 19–51 ms〔算術〕。所以從本地 SSD 或 CPU 讀檢查點，比重算一個 chunk 還便宜；從 NFS 讀，等於重算 0.5–5.6 個 chunk（Jamba 最便宜，Granite 最貴）。
- 分叉點平均分布時，期望重算量：SGLang 每 8192 存一份是 153–405 ms，每 512 存一份是 10–25 ms〔算術〕。**密度是分叉負載的主要槓桿**。

**什麼時候需要中間狀態**〔判讀，延續 Lit-C §3.5〕：
1. **分叉、部分重疊**：同一份前綴、不同後續（agent 樹搜尋、best-of-n、改寫前面的訊息、system prompt 的不同版本）。
2. **尾段被丟掉**：要從中間某點重算。
3. **反向 Cake**：先載前段 KV＋檢查點，再重算後段。

另有兩個結構上的事實：
- 在混合模型上，分叉點 k 若沒有 ≤k 的檢查點，[c,k) 這段的注意力 KV **載了也沒用**。要得到位置 k 的狀態，必須把 [c,k) 整段前向重跑一次，KV 會順帶算出來。所以檢查點密度直接決定「存著的 KV 有多少能用」。
- 只接著問（append-only）時只需要最後一份狀態。SGLang PR #39853 的作者自報（128 個 deep-research agent，不分叉）也顯示：把內部狀態全丟掉反而更好，命中率 0.255→0.552〔PR 描述，未重現；見 cards/D5_SGLangHiCacheMamba.md〕。

**前作與系統（全部讀過原始碼或原文）**：
- **SGLang main**（@3831e7e，2026-10-10）：Mamba／GDN／KDA 狀態放在 GPU→host→storage 三層。寫入規則有：固定網格（`lcm(chunk, page)`）、分歧點補存、命中次數 ≥1 或 2 才備份；GPU 上可以事後疏化，但 host 全留〔程式碼，cards/D5_SGLangHiCacheMamba.md〕。程式碼註解自己承認：write-back 模式可能把只在 device 上的分歧狀態丟掉（`components/mamba.py:172–174`）。這是**延後版失敗的真實例子**（N1）。PR #37613 回報生產環境 L3 上的 KV 只有約 1.8% 有配對的狀態（作者自報）。
- **vLLM main**（@46fb84c，2026-10-10）：`prefix_cache_retention_interval` 預設 0，只保留語意檢查點（replay 邊界＋共享前綴分歧點，註解寫「Marconi-style」）。同一個遮罩**同時用在 GPU 快取和 OffloadingConnector 的卸載**（#51886，2026-09-04）〔程式碼 `config/cache.py:159–165`，D5 親自核對；cards/D5_vLLMRetentionInterval.md〕。vLLM 0.28.0 已經有 Mamba 狀態 CPU 卸載與分層（#44599、#44287），分層管理器「Always offload to all tiers」（寫穿）〔程式碼 0.28 `vllm/v1/kv_offload/tiering/manager.py` L11–13，D5 親自讀〕。
- **LMCache**（@7d7ca47）：每個 block 都存狀態，沒有選擇；store 遮罩列為「延後」〔cards/D5_LMCacheHybrid.md〕。
- **論文**：Sparse Prefix Caching（位置 DP，單層 CPU）、Marconi（准入）、DASC（壓縮每份檢查點，有損，只在 HBM）、HyPIC、Bole、SuffixReplay、Tail-Replay。**沒有一篇做「不同層用不同的位置集合」**〔小幫手搜尋 15 組 arXiv 查詢＋4 組 WebSearch；cards/D5_DASC.md〕。

**判定：可能（窄）**。照判準逐條對：
- 「有看頭」的四個條件：
  1. 占比 ≥50%：成立（每 512 一份時 51–90%；vLLM `all` 模式 block 時 34–86%）。
  2. 寫出頻寬 ≥ SSD 寫入頻寬：單一 stream 時只有部分成立（每 512 一份時 0.17–1.09 倍，Falcon-H1、Nano-9B、Granite、Qwen3-Next ≥1）；對 NFS（寫入 0.60 GiB/s）多數成立（0.5–3.5 倍）。
  3. 有真實分叉負載：成立。
  4. 前作沒做多層放置：**不成立**。SGLang 已經有三層放置，vLLM 有寫穿分層。
- 「死路」條件 1（多層放置＋寫入時選擇已經有人做）：**差一點成立**。兩個引擎都有，只是規則固定、各層共用同一套位置。
- 結論：剩下的空白只有「各層不同密度＋依重疊分布最佳化＋寫入頻寬預算」，是 Sparse Prefix Caching 加上分層成本的增量。

### 4(c) 滑動視窗混合

**浪費**〔算術 20261010-055012-d5-arith，`d5_swa.csv`；KV 大小與 Lit-C 一致（程式裡 assert）〕

| 模型（局部：全域，W） | context | 全部層存 | 只存需要的 | 浪費 | Cake 還原 @3.69 GiB/s（全存 → 只存需要的） |
|:--|--:|--:|--:|--:|:--|
| Gemma-3-12B（40:8，1024） | 32K | 12.0 GiB | 2.31 GiB | **80.7%** | 1626 → 534 ms（−67%） |
| Gemma-3-12B | 128K | 48.0 GiB | 8.31 GiB | 82.7% | 7012 → 1931 ms（−72%） |
| Gemma-3-27B（52:10，1024） | 32K | 15.5 GiB | 2.91 GiB | 81.2% | 2635 → 724 ms（−73%） |
| Gemma-3-27B | 128K | 62.0 GiB | 10.4 GiB | 83.2% | 10845 → 2577 ms（−76%） |
| gpt-oss-20b（12:12，128） | 32K | 1.50 GiB | 0.75 GiB | 49.8% | 292 → 171 ms（−41%） |
| gpt-oss-120b（18:18，128） | 128K | 9.0 GiB | 4.50 GiB | 50.0% | 1896 → 1048 ms（−45%） |

（「只存需要的」＝全域層全部＋局部層最後 W 個 token，適用於「只接著問」的回來請求。）

**程式碼確認（D5 親自讀）**：

| 版本 | 接上 OffloadingConnector 時 | 出處 |
|:--|:--|:--|
| 本機 0.19.1（2026-04-17） | HMA 預設關閉；滑動視窗層轉成全注意力格式、不丟視窗外的 KV；注意力＋Mamba 無法統一，丟 `ValueError`。OffloadingConnector 沒有 `SupportsHMA`。**所以存與載都是全部層**，上表左欄的還原時間就是它的情形 | 〔程式碼 0.19.1 `vllm/config/vllm.py` L1227–1244；`vllm/v1/core/kv_cache_utils.py` L1160–1219；`offloading_connector.py` L44〕 |
| 0.28.0（2026-08-24） | 只在 connector 不支援 HMA 時才關；`OffloadingConnector(..., SupportsHMA)`。滑動視窗組的 lookup 只要求命中點往回 W 個 token 在（**只載視窗內**）。但 store 時每個還有 GPU block 的 chunk 都存；Gemma 這類兩組 block 一樣大的模型，**局部層全存**〔判讀〕 | 〔程式碼 0.28 `vllm/config/vllm.py` L1750–1772；`offloading_connector.py` L49；`offloading/scheduler.py` L109–146、L207–218、L633–663、L1275–1312〕 |
| main（@46fb84c，2026-10-10） | store 也套 `reachable_block_mask`（retention interval），視窗外的不存。**但對齊不整除時退回全存，註解直接舉 Gemma 當例子** | 〔程式碼 main `vllm/v1/core/single_type_kv_cache_manager.py` L1079–1103，D5 親自核對〕 |

**前作**：
- SGLang HiCache 的 SWA 元件，備份到 host 時只備份視窗內，讀回也只讀視窗內（#23391，2026-05-06 合併）〔程式碼 `components/swa.py:101–137、1129–1164`，cards/D5_SGLangHiCacheSWA.md〕。
- SGLang 未合併的 PR #27557 回報：L3 每個 page 都存一份 SWA，DeepSeek-V4-Flash 1M token 佔 18.25 GB；改成每 2K 存一次，降到 5.74 GB（−68%），命中時 TTFT +16%（作者自報）。這就是 (b) 那種「命中點密度」的取捨。
- LMCache 只在視窗小於 chunk 時才裁切寫入，Gemma-3／gpt-oss 照全存〔判讀，cards/D5_LMCacheHybrid.md〕。

**判定：死路**。
- 「寫入時」的說法是死路：延後版（搬到慢層、或備份時才丟視窗外）一樣拿得到。SGLang 正是在備份時才裁。
- 「分層感知卸載本身」也是死路：SGLang 與 vLLM main 已經做了。
- 剩下的「vLLM 對 Gemma 退回全存」是一個工程 bug 級的修補（80% 的寫入和容量），值得回報上游，但不是研究題目。
- 其餘的「命中點密度」問題併入 (b)。

### 4(d) P/D 分離（Llama-3.1-8B、32K、實測 f(i)）

**模型**〔算術 20261010-055012-d5-arith，`d5_pd.csv`〕：
- prefill 節點算 64 個 chunk，共 3.63 s（calib_c1 的中位數總和）。
- KV 每個 chunk 64 MiB，鏈路一次送一個 chunk。
- 「Cake 分割」：decode 端從 t=0 重算 [0,b)，prefill 端只送 [b,n)。decode GPU 不是閒著的，因為它還要做 decode，所以用 φ 表示它分得到的算力比例。φ=1 時等於「不分離」，是退化情形。

| 每條 stream 的頻寬 | 不重疊（算完才傳）露出 | 逐 chunk 管線露出 | Cake 分割比管線快（φ=0.5／0.25） |
|--:|--:|--:|:--|
| 2 Gbps | 82.6% | 78.9% | 67.2%／53.8% |
| 4 Gbps | 70.3% | 57.9% | 45.9%／34.2% |
| 6 Gbps | 61.2% | 37.0% | 30.4%／22.6% |
| 8 Gbps | 54.2% | 16.1% | 14.6%／14.6% |
| 10 Gbps | 48.7% | 1.5% | 0%／0% |
| **25 Gbps** | **27.5%** | **0.59%** | **0%／0%** |
| 100 Gbps | 8.7% | 0.15% | 0%／0% |
| 400 Gbps | 2.3% | 0.04% | 0%／0% |

- 門檻：大約當每個 chunk 的傳輸時間 ℓ 小於 f 的平均（3.63 s ÷ 64＝57 ms）時，管線就追得上，傳輸被藏起來。換算約 9–10 Gbps 每條 stream〔算術〕。
- vLLM 0.28 的 NixlConnector「does not do layerwise saving」，push 模式在 `request_finished`（prefill 結束）之後才推〔程式碼 0.28 `nixl/connector.py` L281–292、`nixl/push_scheduler.py` L3–23〕。所以 vLLM 預設是「不重疊」那一欄（25 Gbps 時露出 27.5%）。但修法是**逐層或逐 chunk 傳**，Splitwise、DéjàVu、Mooncake 都做了〔原文 Splitwise p.7–9、DéjàVu p.5、Mooncake p.5–6，小幫手讀〕，不是 Cake。

**前作**（小幫手讀原文，cards/D5_*.md）：
- CacheGen（SIGCOMM'24）：每個 chunk 選一種壓縮等級，或送文字讓對方重算。是依序決定，不是並行。
- Cake 原文模擬過 7–100 Gbps 的網路。
- **CacheFlow**（arXiv 2604.25080，預印本）：把 Cake 推廣成 token×layer 的階梯形切分，用 10／40／80 Gbps、Llama-3.1-8B 與 Qwen3.5 混合模型，在 40／80 Gbps 下 TTFT 比 Cake 快 2.63／2.40 倍〔原文 p.7、p.9〕。「從遠端 store 拉 prefix KV、一邊算一邊載」這條路已經很擠。
- DroidSpeak：跨網路「部分層重算、部分層拉取」，是為了跨模型重用。它的 Limitations 把「依頻寬調整重算比例」列為未來工作〔原文 p.12〕。
- KVPR：重算的是 K/V 投影，成本和位置無關，不是 Cake 型；在 GQA 上傳 X 比傳 KV 大〔判讀〕。
- SmartGen、KAIROS：有損，或整個請求全有全無。
- **「同一模型、P/D 之間依頻寬切分、decode 端同時重算」沒找到。**

**判定：死路**（判準死路條件 1：25 Gbps 時管線只露出 0.59% ≤5%）。
- 判準以外的觀察：真實部署中，每條 stream 的有效頻寬可能遠低於網卡標稱。SmartGen 報告多個 prefill 節點擠一張 decode 網卡，25 Gbps 下傳輸是 prefill 的 4.7–6.5 倍〔原文 SmartGen p.1、p.3，小幫手讀，前提是 75% prefix 命中，未查證〕。
- 在 ≤8 Gbps／stream 時，Cake 分割可省 15–67%。這是一個「擠的網路＋P/D」的子題，但對手很多（壓縮、全有全無、CacheFlow），而且 decode 端重算會吃掉 decode 的算力（φ 的代價沒算進 TPOT）。**不建議**列為主線。

### 4(e) MLA／MQA

〔算術 20261010-055012-d5-arith，`d5_mla.csv`；FLOPs 欄取自 Lit-C 表，b/n 全部和 Lit-C 一致〕

| 模型 | KV/token | b/n @0.33（NFS） | @3.69 | @35.4 | Cake 比「只算或只載」較快者再快 @3.69 |
|:--|--:|--:|--:|--:|--:|
| Llama-3.1-8B（基準） | 128 KiB | 0.797 | 0.312 | 0.047 | 31.3% |
| DeepSeek-V3 | 68.6 KiB | 0.297 | **0.047** | 0 | 4.7% |
| Kimi-K2 | 68.6 KiB | 0.359 | **0.062** | 0 | 6.2% |
| Falcon-7B（MQA） | 8 KiB | 0.250 | **0.031** | 0 | 3.1% |
| DeepSeek-V2-Lite（小的例外） | 30.4 KiB | 0.703 | 0.234 | 0.031 | 23.4% |

**判定：死路**（V3、K2、Falcon 在 3.69 GiB/s 時 b/n ≤0.10）。
- 一段話：MLA 的 KV 是低秩 latent（每層 576 維），比 hidden state 小 3.6–12.4 倍（Lit-C §4）。所以「改存 hidden」只會更大，沒有格式選擇。
- 3.69 GiB/s 時，只載比只算快 29–55 倍（V3 46 倍、K2 29 倍、Falcon-7B 55 倍）〔算術〕。Cake 前段幾乎是 0，策略退化成「全部載入」，放置問題退化成一般的容量快取（LRU 之類）。
- 唯一會出現前段重算的是 NFS 級頻寬（約 0.3 GiB/s，b/n 0.25–0.36）。那只是同一個 Cake 換到更慢的頻寬，κ 地圖（D3）已經涵蓋，沒有 MLA 特有的機制。
- DeepSeek-V2-Lite 是例外：active 只有 2.24B，重算便宜，b/n 0.23。照判準只記錄、不改判定。

---

## 5. 延後版／更簡單的做法＋前作

### 5.1 (a) VLM
| 對照 | 能不能追平寫入時選格式 | 證據 |
|:--|:--|:--|
| 寫穿：KV＋embedding 都存 | **能**，只多 22.2%（Qwen3-VL）／12.5%（Qwen2.5-VL、LLaVA-OV、InternVL3，Lit-C） | 〔算術〕 |
| 保留像素、讀取時重跑 encoder | 對 Qwen3-VL 差 4–9%；對 Qwen2.5-VL 差很多 | 〔算術〕；VLCache 表 1、6 |
| 系統已有 | VLCache（encoder 輸出＋pre-RoPE KV 兩種都存，量了三條還原路徑）；LMCache encoder cache（CPU／disk／remote，和 KV 分開管理）；Dynamo embedding cache（CPU LRU）；vLLM EC connector（存磁碟範例，Lit-C） | cards/D5_VLCache.md；小幫手 WebFetch LMCache／Dynamo 文件 |
| 相關但不同 | ShallowStream：串流時只算淺層，存 embedding＋淺層 KV，問答時從 embedding 重跑深層，主張「先算深層再丟救不回算力」，是 N2（串流時的算力）〔原文 p.2、p.4〕；MPIC：缺的圖算、命中的圖載，按「有沒有」分，不是 Cake 式；RServe：encoder 和 prefill 重疊，會再縮小 ΔT | cards/D5_ShallowStream.md、D5_RServe.md；A_MPIC |
| 沒找到 | 依頻寬或 κ 決定「這一層存 embedding 還是 KV」；多模態的 Cake 式切分 | 小幫手 23 組查詢 |

### 5.2 (b) 混合 SSM
| 對照 | 狀況 |
|:--|:--|
| 寫穿：全部檢查點寫到所有層 | vLLM 0.28 分層管理器就是這樣〔程式碼〕；代價是寫入頻寬（上表 0.7–1.1 倍本地 SSD）與 SSD 容量 |
| 延後版：先全寫到 CPU（35 GiB/s 很夠），搬到 SSD 時再挑 | **結構上做得到**，因為狀態已經在 CPU。代價是 CPU 容量：Nemotron-H 每 512 存一份時，一個 32K session 的檢查點 3.1 GiB，是 KV（0.5 GiB）的 6 倍〔算術〕 |
| 延後版：等觀察到分叉再補存 | **做不到**（N1）。SGLang 程式碼註解與 PR 都記錄了 write-back 丟狀態、L3 配對失敗〔程式碼、PR 描述〕 |
| 系統已有 | SGLang 三層＋固定規則；vLLM main 的 Marconi 式語意檢查點、GPU 與卸載共用同一個遮罩；LMCache 全存 |
| 論文已有 | Sparse Prefix Caching（位置 DP）、Marconi（准入）、DASC（壓縮狀態，有損） |

### 5.3 (c) 滑動視窗
- 延後版（備份或搬移時才丟視窗外）就是 SGLang 的做法；vLLM main 用 store 遮罩。
- 「寫入時就不寫」只省 GPU→CPU 的寫入量，而這條路有 35 GiB/s，不稀缺〔判讀〕。

### 5.4 (d) P/D
- 最簡單的對照是**逐層或逐 chunk 傳**（Splitwise、DéjàVu、Mooncake），它在 ≥10 Gbps／stream 時把傳輸藏到 <1.5%。
- 遠端 store 的情境已有 Cake、CacheGen、CacheFlow。

### 5.5 (e) MLA
- 載入永遠贏，所以「全部載入＋一般快取」就是最佳。

### 5.6 卡片（本方向新寫 22 張；我親自核對的標 ★）
- **VLM**：D5_VLCache ★（表 1）、D5_ShallowStream ★（層數註記）、D5_Vista、D5_Kamera、D5_RServe、D5_PACE、D5_VRex、D5_EPD
- **SSM／SWA**：D5_SGLangHiCacheMamba、D5_SGLangHiCacheSWA、D5_vLLMRetentionInterval ★（`config/cache.py:159–165`、SWA 遮罩 L1079–1103）、D5_LMCacheHybrid、D5_DASC、D5_vLLM028_HybridOffload ★（D5 自己寫）
- **P/D**：D5_DistServe、D5_Splitwise、D5_DejaVu、D5_CacheGen、D5_KVPR、D5_SmartGen、D5_CacheFlow、D5_DroidSpeak
- 小幫手只看了摘要、沒做卡的：StreamingVLM（ICLR'26，依 arXiv comment）、StreamMem、LiveVLM（DAC'26）、StreamingTOM（CVPR'26）、InfiniPot-V（NeurIPS'25）、CONDUIT（EMNLP'26 Findings）、MiMo-V2.5、HyPIC、Bole、KAIROS、2608.14967。這些內容**只讀到摘要層級**。

---

## 6. 如果要繼續

只建議一條，而且要先看 D1、D4 的結果：

**「命中點狀態」的分層密度**（合併 (b) 與 (c) 剩下的部分）：
- **問題**：混合模型裡，要能在位置 L 接著算，需要 L 點的 SSM 狀態或局部層的視窗 KV。這些狀態很大（一份＝1–9 個 chunk 的 KV），而且在 GPU 上一被覆蓋就沒了（N1）。每一層（GPU／CPU／SSD）各該留多密？
- **為什麼可能有東西**：
  - 現行引擎各層共用一個遮罩（vLLM），或 GPU 疏、host 全留（SGLang）。
  - 全部寫穿要用掉約一整顆本地 SSD 的寫入頻寬（N2）。
  - 延後版要先佔 CPU 容量（N2），沒空檔時（N3，D1 的題目）會被逼著丟。
- **要先回答的**：
  1. D4：真實負載裡，分叉點落在「非語意邊界」的比例有多少？如果幾乎都在回合或步驟邊界，vLLM 的語意檢查點就夠了，這條線直接死。
  2. D1：沒有空檔時，CPU 暫存「全部檢查點」會不會被擠爆？
- **最便宜的第一步**：擴充 D1 的併發模擬器，加一種「命中點狀態」物件（大小用本文件的表），對照 vLLM 的 retention=0、SGLang 的寫穿＋疏化、Sparse Prefix Caching 的 DP。**不用 GPU**。
- **停損**：模擬裡「各層不同密度」比「各層共用最佳單一密度」快 <5%，就停。

其他子方向不建議繼續。如果硬要做 (a)，唯一值得量的是 Qwen2.5-VL 這種 encoder 很貴的模型。即使如此，它的答案也是「embedding 寫穿」，不是寫入時決定。下載指令留給之後（Qwen3-VL-8B，Apache-2.0，不需要 token，權重 4 個 safetensors 共 17,534,339,512 bytes＝16.3 GiB）：
```bash
source code/m7_env.sh   # HF_HOME=/mlsteam/data/tiara/hf-cache
hf download Qwen/Qwen3-VL-8B-Instruct    # 本 agent 沒有執行
```

---

## 7. 失敗與異常

1. **腳本錯誤（已修，重跑）**：run `20261010-054621-d5-arith`，exit 1。完整錯誤：
   ```
   File "/mlsteam/workspace/paper-hierarchical-kv-state/code/m8_arch_arith.py", line 287, in part_swa
       Lh, h, inter = c["num_hidden_layers"], c["hidden_size"], c["intermediate_size"]
   KeyError: 'num_hidden_layers'
   ```
   原因：Gemma-3 的 config 把文字部分放在 `text_config` 底下。修法：`c = c.get("text_config", c)`。
2. **校準核對抓到的錯（同一個 run 的 stdout）**：第一版用了 calib_c1 全部 80 個 chunk 當 n，b/n 出現 1/80 的倍數（0.775、0.6125…），和 Lit-C 對不上。改成 α、β 用 80 個點擬合、b/n 只看前 64 個（32K），同 Lit-C；之後五個核對值全部相同。另外補了 1.0 GiB/s 這一格，因為 Lit-C 的 0.578 對應的是它。
3. **中間的 run**：`20261010-054636`、`-054755`、`-054853`、`-054941-d5-arith` 是逐步加欄位的版本，CSV 已被最終版 `20261010-055012-d5-arith` 覆蓋。stdout 留在 runs 目錄。
4. **P/D 模型修正**：第一版讓 decode GPU 完全閒置（φ=1），最佳解永遠是「decode 自己全算」（b=64），這是退化結果。改成加入 φ=0.5／0.25，並把頻寬掃到 2–10 Gbps。φ=1 的列仍保留在 CSV，只當上限參考。
5. **文獻小幫手的異常**（他們的回報，原文照錄重點）：
   - arXiv API 回 HTTP 429（P/D、SSM 小幫手）；一開始用 `http://` 回 `301 Moved Permanently`（VLM 小幫手）。
   - `gh: command not found`，改用未認證的 GitHub API 與 sparse clone。
   - **規則違反（已修）**：SSM 小幫手的 clone 指令在 repo 根目錄執行，vllm、lmcache、dynamo 三個目錄在 repo 根目錄留了約 15 秒，之後才移到 scratchpad。**D5 已確認**：repo 根目錄乾淨，`git status` 只有本輪新增的未追蹤檔。
   - 一開始用 Semantic Scholar API 也回 429（D5 自己測），所以全程沒用 S2。
6. **未查證或低信心**：
   - SGLang／vLLM 的 PR 數字都是作者自報。
   - SmartGen 的「75% prefix 命中」前提未查證。
   - ShallowStream 原文說 Qwen3-VL-8B 有 28 層，config 是 36 層（原文如此，已在卡片註記）。
   - Splitwise 的 IEEE 版沒打開。
   - 「vLLM 0.28.0 對 Gemma 局部層全存」是讀程式碼的推論，**NOT_MEASURED**。
7. **限制**：
   - 所有 b/n、還原時間都是〔算術〕。它們假設各模型的 MFU 等於 Llama 在 m7 harness 上的值。
   - 小 GEMM 的 ViT 實際 MFU 可能更低，那樣 encoder 更貴，(a) 的 ΔT 會被低估。VLCache 的實測量級和本文件一致。
   - SSM 的 prefill 速度偏樂觀，所以寫出頻寬是單一 stream 的值。
   - 沒有任何新的 GPU 量測。
