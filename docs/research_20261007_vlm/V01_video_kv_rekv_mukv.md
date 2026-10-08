# V01 影片 KV 的儲存與檢索：ReKV、MuKV（評測卡）

> **一句話**：ReKV 把串流影片每一格的 KV 全部存到 CPU，問題來時每層各挑 64 格搬回 GPU；MuKV 在 ReKV 的框架上改存段、格、區塊三種粒度（不壓縮時是 ReKV 的 3 倍），寫入時剪掉其中 2/3，淨存量與 ReKV 相同，準確率反而更高。兩篇都沒有拆解 I/O 時間（ReKV 報端到端延遲、GPU 峰值記憶體、每小時 KV 大小；MuKV 主要報 token 數）。ReKV 論文說會存到 RAM 或磁碟（MuKV 論文沒說存在哪），但兩份釋出的程式碼都只有 CPU RAM，而且 MuKV 的主流程沒有用 ReKV 的 KV 管理器。〔複核修正：原句「剪掉 2/3」沒說是相對於 3 倍的多粒度快取；「兩篇都只用準確率和 token 數」不符 ReKV 表 5；「論文說會存到磁碟」只適用 ReKV〕

**日期**：2026-10-07
**緣由**：老師 10/7 提到可以用「長影片持續累積 KV」當視覺情境，並點名 MuKV。
**格式**：沿用 `docs/research_20261006_eval/README.md` 的評測卡。頁碼以 PDF 頁為準。
**證據等級**：〔原文〕〔程式碼〕〔計算〕〔判讀〕〔未查證〕。

---

### ReKV：Streaming Video Question-Answering with In-context Video KV-Cache Retrieval（ICLR 2025；arXiv 2503.00540）

- **讀了什麼**：〔全文〕ICLR 2025 proceedings 版 PDF（13 頁，無附錄）＋ arXiv v1 PDF（16 頁，正文與 proceedings 版相同，多附錄 A–C），https://arxiv.org/pdf/2503.00540v1 。程式碼 https://github.com/Becomebright/ReKV commit `1fd9a3d`（2025-11-04）。查證 2026-10-07。
- **一句話**：讓現成的 Video-LLM 不用訓練就能做串流影片問答：邊看邊存 KV，問題來時只取回相關的幾格。
- **評測要證明的主張**：(1) 檢索相關的 KV 比均勻取樣準；(2) 延遲與 GPU 記憶體不隨影片變長而增加，不會 OOM；(3) 不用訓練就能接上多種 Video-LLM（p.2 §1、p.2 圖 1、p.10 §6）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | 主要用 LLaVA-OV-0.5B、7B；附錄另測 Video-LLaVA-7B、LongVA-7B、LLaVA-OV-72B（72B 用多卡分片，降到 0.1 FPS、取 32 格）。FP16 | 〔原文〕p.6 §4.2；p.15 §B.1（原文只寫「model sharding」）；〔程式碼〕〔複核補充〕`video_qa/run_eval.py` 註解「llava_ov_72b needs 4x 80GB GPUs」 |
| 硬體 | 主要實驗：A100 80GB。圖 1 的效率測試：H800 80GB。README：8× H800 | 〔原文〕p.6 §4.2；p.2 圖 1；〔文件〕README |
| 軟體 | 自己的 PyTorch／HF 實作，不是 vLLM。KV 管理沿用 InfLLM 的寫法 | 〔原文〕p.4（「as in Xiao et al., 2024a」）；〔程式碼〕`model/attention/kv_cache_manager.py` |
| 資料 | MLVU-dev-mc、QAEgo4D-test-mc、EgoSchema、ActivityNet-QA（離線）；RVS-Ego、RVS-Movie（串流）；CGBench-mc。表 1：RVS-Ego 60 分鐘、10 部、1,465 題；RVS-Movie 30 分鐘、22 部、1,905 題。〔複核補充〕用兩個 repo 的 README 都指向的 HF 標註檔 `Becomebright/RVS`（`ego/ego4d_oe.json`、`movie/movienet_oe.json`）驗算：ego 10 部、`duration` 平均 60.1 分鐘、1,465 題；movie 22 部、`duration` 平均 27.9 分鐘、1,905 題（每題 `end_time` 都 ≤ `duration`，確認單位是秒）。與本表一致 | 〔原文〕p.6 表 1；〔資料〕HF 標註檔 |
| 長度 | 0.5 FPS；LLaVA-OV 每格 196 token；滑動視窗 15K token。1 小時＝1,800 格＝352,800 token〔計算：1800×196〕 | 〔原文〕p.6 §4.2；p.14 §A.3 |
| 到達與併發 | 串流：問題在標註的結束時間點後立刻送入。效率測試：一部 1 小時 1080P 的 RVS-Ego 影片、100 個分散的問題，問題補到 64 token，答案固定 128 token。〔複核補充〕釋出的評測程式：每個 process 綁一張卡、依序處理分到的影片；同一部影片的問題逐題處理，每題前先把影片編碼到該題的 `end_time`（每批 64 格）；RVS 開放題用 `max_new_tokens=256`（128 是效率測試的設定）。附錄 A.1 的「一個編碼 process＋問答 process pool」架構不在釋出的程式碼裡 | 〔原文〕p.8 §4.5；p.14 §A.1；〔程式碼〕`video_qa/rekv_stream_vqa.py`、`video_qa/run_eval.py`、`model/abstract_rekv.py` |
| 重用結構 | 同一部影片的 KV 被多個問題重用；附錄 B.3 顯示每題的 FLOPs 隨問題數增加而下降（100→360 題）。〔複核補充〕B.3 是在 B.2 自己訓練的 Base 骨幹（CLIP-ViT-L/14＋Vicuna-7B）上量的，不是 LLaVA-OV | 〔原文〕p.16 表 8–9；p.15 §B.2、p.16 §B.3 首句 |
| 掃描的自變數 | 取回的格數 r ∈ {8,16,32,48,64,80}；區塊大小 b ∈ {1,2,4,8,16}；內部／外部檢索 | 〔原文〕p.7 §4.3、圖 3 |
| 對手 | 均勻取樣（同一模型）、Flash-VStream（另在附錄用同一骨幹重訓做公平比較）、各種離線 Video-LLM 的公開數字 | 〔原文〕p.8 表 4–5；p.15 §B.2 |
| 系統指標 | 表 5：Video Enc.（每秒編碼幾格）、Latency（從問題輸入到回答完成）、GPU（峰值記憶體）、KV-Cache（每小時卸載的大小）。7B 內部檢索：11 FPS、3.3 s、38 GB、18.8 GB/h；0.5B：17 FPS、1.6 s、19 GB、4.0 GB/h | 〔原文〕p.8 表 5 |
| 品質指標 | 選擇題準確率；開放題用 GPT-3.5-turbo-0613 判分（Acc.＋1–5 分）。〔複核補充〕判分腳本寫死 `gpt-3.5-turbo-0613`、`temperature=0` | 〔原文〕p.6 §4.1；p.8 表 4 標題；〔程式碼〕`video_qa/eval/eval_open_ended.py` |
| 主要結果 | QAEgo4D：7B 均勻取樣 53.0 → 內部檢索 56.0（Oracle 64.4）。RVS-Ego／Movie（7B）：均勻 56.2／43.0 → 內部檢索 63.7／54.4 | 〔原文〕p.6 表 2；p.8 表 5 |
| 消融／敏感度／開銷 | 檢索方式、r、b；附錄有 FLOPs／MACs（calflops）。**沒有**拆解延遲（檢索、搬運、解碼各多少），**沒有**磁碟層的量測 | 〔原文〕p.6–7 §4.3；p.16 §B.3 |
| 重複與統計 | 未說明。〔複核補充〕程式碼固定 `random.seed(2024)`，解碼取 argmax（貪婪） | 〔原文〕；〔程式碼〕`video_qa/base.py`、`model/llava_onevision_rekv.py` |
| 程式碼／資料 | 公開；repo 沒有 LICENSE 檔（GitHub license 欄位為 null） | 〔程式碼〕 |
| 設計理由（原文） | 因果遮罩讓影片 token 的編碼不受問題影響，所以編碼和問答可以拆開；KV 全部保留以免資訊流失，靠檢索控制問答成本 | 〔原文〕p.3 Discussion-II；p.10 §5 |
| 設計理由〔判讀〕 | 用「每格一個區塊」當檢索單位，讓檢索、搬運、快取都以 196 token 為單位，實作最簡單 | 〔判讀〕 |
| 原文沒講清楚的地方 | (1) 3.3 s 裡搬運占多少；(2) 「RAM 或 disk」的磁碟路徑怎麼做；(3) 檢索出的 token 位置重排的細節（原文只說當成連續 token） | 〔原文〕p.5 Positional Encoding |
| 與既有整理不一致 | 無（既有整理沒有收這篇） | |
| 對本研究的意義〔判讀〕 | 見檔末「綜合判讀」 | |

**程式碼查到的 I/O 路徑**〔程式碼〕（commit `1fd9a3d`）：

- **寫入**：每格（196 token）每層的 K/V 在編碼時立刻 `.to("cpu", non_blocking=True)` 搬到 CPU，再 `.pin_memory()`。`load_model` 預設 `pin_memory=True`。等於「全存」，CPU 端沒有逐出（`kv_cache_manager.py` 的 `MemoryUnit.__init__`、`_append_global`；`llava_onevision_rekv.py` 的 `load_model`）。〔複核補充〕「立刻」要加條件：累積超過 `n_local`（15K）token、init 區填滿（`init_exc=True`）之前，KV 還留在 GPU 的 `global_remainder`；之後每批新 KV 在同一次 forward 裡就切成 196 token 的格卸載。GPU 上另外保留最近 15K token 的 local window（等於一份重複）。存到 CPU 的是**還沒套 RoPE** 的 K，問答時才把取回的格當連續位置套 RoPE（`rekv_attention.py`）。
- **讀取**：每層各自挑 top-k 格（預設 64），逐格從 pinned CPU 複製到 GPU；GPU 上每層有一個 LRU 區塊快取，上限 `max_cached_block=128`（`get_retrieved_kv`、`_remove_lru_blocks`）。〔複核補充〕每題只在問題 token 上檢索一次（問題 query 的平均對每格 key 的平均；程式碼是未正規化的內積，不是論文寫的 cosine），之後 prompt 和解碼都沿用這批 KV；問答時的上下文是 init＋取回的格，**不含** local window。命中 LRU 的格只做 GPU→GPU 複製；未命中的格 K、V 各做一次非同步 H2D（在獨立的 `GLOBAL_STREAM` 上）。
- **磁碟**：論文（p.1 摘要、p.4、p.14 §A.1）和 README 都說可以卸載到磁碟。但用 GitHub code search 查「disk」「torch.save」「memmap」，只在 README 和不相關的 `trl` 套件裡出現，**釋出的程式碼沒有磁碟層**。README 要求調高 `vm.max_map_count`（「needed for offloading KV-Caches」）。
- **單位核對**〔計算〕：每 token 的 KV＝2×28 層×4 頭×128×2 bytes＝57,344 bytes（7B）；1 小時 352,800 token → 20.2 GB＝**18.8 GiB**。所以原文的「18.8 GB/h」其實是 GiB；0.5B 同理是 4.04 GiB。每格每層 392 KiB；一題取 64 格 × 28 層 ≈ 0.67 GiB，〔複核修正〕拆成 1,792 次逐格載入；每次 K、V 分開複製，所以最多是 3,584 次 196 KiB 的 H2D。命中 GPU LRU 快取的格不經 PCIe，所以 0.67 GiB 是每題的上限（原文：「拆成 1,792 次小搬運」）。〔複核補充〕ReKV 程式碼自己的 log 也是先 `/(1024**3)` 再標「GB」（`model/abstract_rekv.py` 的 `encode_video`），和「GB 其實是 GiB」一致。

---

### MuKV：Multi-Grained KV Cache Compression for Long Streaming Video Question-Answering（CVPR 2026；arXiv 2605.22269）

- **讀了什麼**：〔全文〕CVF Open Access 版 PDF（正文 8 頁＋參考文獻），https://openaccess.thecvf.com/content/CVPR2026/papers/Xiao_MuKV_Multi-Grained_KV_Cache_Compression_for_Long_Streaming_Video_Question-Answering_CVPR_2026_paper.pdf ；補充材料 PDF（3 頁）。程式碼 https://github.com/IMBALDY/MuKV commit `2127b87`。arXiv 版未讀。查證 2026-10-07。
- **一句話**：在 ReKV 的框架上，把每段影片存成段、格、區塊三種粒度的 KV，存之前依注意力和頻率剪掉大部分，問答時兩階段檢索。
- **評測要證明的主張**：在不增加記憶體與問答成本的前提下提高準確率；壓縮模組單獨加在 ReKV 上也有用；影片越長優勢越大（p.1 摘要、p.2 貢獻列表）。

| 欄位 | 內容 | 出處 |
|:--|:--|:--|
| 模型 | LLaVA-OV 0.5B、7B；補充材料另測 Qwen2.5-VL-3B、Qwen3-VL-2B／4B／8B。程式碼預設 FP16 | 〔原文〕p.5 §4.1；補充 p.2 表 1；〔程式碼〕〔複核修正〕主流程是 `model/mukv_rerank.py` 的 `load_model`（`torch_dtype=torch.float16`，單卡 `cuda:0`）；原本引的 `llava_onevision_mukv.py` 不在執行腳本的路徑上 |
| 硬體 | 主要實驗在 **A5000 24GB** | 〔原文〕p.5 §4.1 |
| 軟體 | 〔複核修正〕repo 是從 ReKV 的程式碼改出來的；`model/attention/kv_cache_manager.py` 與 ReKV 的版本**逐位元組相同**（SHA-1 相同），但**主流程沒有用到它**：`scripts/run_mukv_rvs_*.py` → `mukv_rerank.load_model` → `MuKVDualSignalCompressionModel`，KV 存在自己的 `MuKVMultiGrainKVCache`，不經過 `patch_hf`／`ContextManager`。只有執行腳本沒呼叫的 `llava_onevision_mukv.py` 會走 ReKV 的管理器。I/O 路徑和 ReKV 不同，見表下。transformers 釘在 commit `66bc4de`（4.45.0.dev0） | 〔程式碼〕`scripts/run_mukv_rvs_ego.py`、`model/mukv_rerank.py`、`model/mukv_dualsignal_compression.py`、`prepare.sh`（原文：「I/O 路徑同上」） |
| 資料 | 串流：RVS-Ego、RVS-Movie、StreamingBench（只取 Real-Time 子集：2.5K 題、500 部、平均 10 分鐘）。離線（補充）：Video-MME、MLVU、EgoSchema。正文寫 RVS-Ego 11 部、平均 30 分鐘、1.4K 題；RVS-Movie 20 部、平均 1 小時、1.9K 題（〔複核補充〕時長和實際標註檔對不上，見下方不一致第 1 點） | 〔原文〕p.5 §4.1 |
| 長度 | 0.5 FPS；每段 4 格（8 秒）；每格 P=196 token，分成 S=4 個 super-patch。記憶體以「每 300 格（10 分鐘）」報告，59K≈196×300 | 〔原文〕p.5 §4.1；p.6 表 1 標題 |
| 到達與併發 | 沿用 RVS 的問題時間點；離線資料集假設問題都在影片結尾 | 〔原文〕補充 p.1 §2.1 |
| 重用結構 | 同 ReKV（同一部影片多題） | 〔原文〕p.5 §3.5 |
| 掃描的自變數 | 粒度組合、壓縮比（25%–90%）、FPS（0.5／2／3）、各粒度保留比例、λ、取用的層 | 〔原文〕p.6–8 表 2–8；補充表 2–4 |
| 對手 | ReKV（主要）、Flash-VStream、LongVA；InfiniPot-V（受控比較）；補充表 1 的 LiveVLM、StreamMem 數字**直接抄自 StreamMem** | 〔原文〕p.6 表 1；p.7 表 5；補充 p.2 表 1 標題 |
| 系統指標 | 主要用 **token 數**當「與裝置無關」的效率指標（#Inf. Tok＝問答時送入的視覺 token；#Mem. Tok＝存起來的 token）。只有表 4 報真實部署：每題秒數（s/Q）、每小時快取大小（G/h） | 〔原文〕p.5 §4.1；p.7 表 4 |
| 品質指標 | RVS：LLM 判分。原本的 GPT-3.5-turbo-0613 已停用，改用 GPT-3.5-turbo，並在表 1 把舊分數標灰；StreamingBench：選擇題準確率。〔複核補充〕判分腳本用的是別名 `gpt-3.5-turbo`（`temperature=0`），別名背後的版本會隨時間換，日後無法重現同一個判分模型。執行腳本對每部影片包了 `try/except … continue`，某部影片出錯（例如 OOM）時，它的題目會直接從結果檔消失，不會報錯 | 〔原文〕p.5 §4.1；p.6 表 1；〔程式碼〕`video_qa/eval/eval_open_ended.py`、`scripts/run_mukv_rvs_ego.py` |
| 主要結果 | 7B：RVS-Ego 59.5（ReKV 重新判分 56.2）、StreamingBench 64.4（ReKV 62.3），#Inf. Tok 8.3K vs 12.5K、#Mem. Tok 同為 59K。在 ReKV 上只加壓縮（DCP 50%）：RVS-Ego 51.5→56.1，記憶體減半。〔複核修正〕表 3 沒寫模型大小；ReKV 那列的 51.5／42.3 和表 1 的 ReKV-0.5B 相同，所以推定是 0.5B〔判讀〕（原文：「0.5B 在 ReKV 上只加壓縮」，未標判讀） | 〔原文〕p.6 表 1；p.7 表 3 |
| 消融／敏感度／開銷 | 粒度（表 2）、壓縮訊號與比例（表 3）、部署指標（表 4）、對 InfiniPot-V（表 5）、高／低頻（表 6）、檢索方式（表 7）、取用層（表 8）。**沒有**報寫入端（編碼）的成本；原文說三次 prefill 平行執行、不增加延遲，但沒有量測。〔複核補充〕程式碼裡三種粒度是在 Python 迴圈裡**依序**跑的，而且每個區塊各跑一次 forward（每 64 格：256＋64＋16＝336 次，並開 `output_attentions=True`）。執行腳本有記每部影片的 `encode_time`（`memory_stats.csv`），但論文沒報 | 〔原文〕p.4 §3.2；p.6–8；〔程式碼〕`model/mukv_dualsignal_compression.py` 的 `encode_video` |
| 重複與統計 | 未說明；超參數是在 RVS-Ego 上 greedy search 後套用到其他資料集。〔複核補充〕程式碼固定 seed 0，解碼取 argmax | 〔原文〕p.5 §4.1；〔程式碼〕`scripts/run_mukv_rvs_ego.py` |
| 程式碼／資料 | 公開（只有 RVS 的執行腳本，預設 0.5B）。README 掛 Apache-2.0 徽章，但 repo 沒有 LICENSE 檔 | 〔程式碼〕〔文件〕 |
| 設計理由（原文） | 逐格快取缺少區域細節與跨格的時間脈絡，而且冗餘太多，反而干擾檢索；頻率可以校正注意力的位置偏差（越早的 token 注意力越高） | 〔原文〕p.2 §1；〔複核修正〕圖 5 在 p.7，討論文字在 p.8「Effect of Frequency」段（原文：「p.7 圖 5 討論」） |
| 設計理由〔判讀〕 | 剪掉的 KV 讓檢索的候選變少、雜訊變少，所以「存得少」能換來「答得準」。這在文字的 KV 管理裡比較少見 | 〔判讀〕 |
| 原文沒講清楚的地方 | (1) 壓縮後的 KV 存在哪一層（〔複核修正〕程式碼是 CPU RAM，但不是 ReKV 的 pinned 管理器，而是用 `.cpu()` 存成一般 pageable tensor 的 Python list；原文：「程式碼同 ReKV」）；(2) 表 4 沒寫模型大小〔判讀：4.00 G/h 與 Acc 51.5 都對得上 ReKV-0.5B，應是 0.5B〕；(3) 記憶體數字對不上，見下；(4)〔複核補充〕§3.2（p.4）說每個粒度的表示要「連同同粒度所有過去的 KV」一起送進 LLM，但程式碼每個區塊只接 system prompt 的 KV，不接過去的區塊 | 〔原文〕p.3–4 §3.2；〔程式碼〕`model/mukv_multigrain_kvcache.py`、`model/mukv_dualsignal_compression.py` |
| 與既有整理不一致 | 見下方「原文之間的不一致」 | |
| 對本研究的意義〔判讀〕 | 見檔末「綜合判讀」 | |

**程式碼查到的 I/O 路徑**〔程式碼〕〔複核補充〕（commit `2127b87`；主流程 `scripts/run_mukv_rvs_ego.py` → `model/mukv_rerank.py` → `model/mukv_dualsignal_compression.py`，儲存在 `model/mukv_multigrain_kvcache.py` 的 `MuKVMultiGrainKVCache`）：

- **編碼**：每批 64 格，影片 token（64×196＝12,544）分別切成 49、196、784 token 的區塊，**三種粒度都各自鋪滿全部 token**（每一格都存成格區塊，每格切成 4 個區塊，不是只取段落中間那一格）。每個區塊單獨跑一次 LLM forward，前綴只有 system prompt 的 KV（`past_key_values=self.init_kv_cache`；在釘住的 transformers 4.45 下傳的是 tuple，不會被就地修改）。沒有滑動視窗，也不接同粒度過去的 KV。
- **寫入**：剪枝後每層的 K、V 用 `.detach().cpu()` 存成一般（非 pinned）CPU tensor，放在 Python list；代表向量也放 CPU。沒有逐出，沒有磁碟。
- **讀取**：每題先把該粒度**全部**代表向量 stack 起來搬上 GPU 算相似度。開 rerank（主實驗預設開）時，每個粒度先取 2×k（40／64／24，共 128 塊），**這 128 塊的全部層 KV 在 rerank 前就逐層 `.to(device)` 搬上 GPU**（從 pageable 記憶體同步複製）；rerank 後只留 20／32／12＝64 塊。跨題沒有 GPU 快取。
- **位置**：存的是 HF 快取裡已經套過 RoPE 的 K。每個區塊都從 system prompt 之後的同一個位置開始編碼，問答時直接串接，不重排位置。
- **與論文的其他差異**：問題向量取最後一層、**最後一個**問題 token 的 key（論文式 (6) 是問題 query token 的平均）。
- **用量核對**〔計算〕（主實驗 ρ＝{0.1, 0.1, 0.8}，p.5；腳本 `scripts/sh/run_mukv_rvs_ego.sh` 相同）：每塊保留 ⌊4.9⌋＝4、⌊19.6⌋＝19、⌊627.2⌋＝627 token。問答時 20×4＋32×19＋12×627＝8,212 token，對得上表 1 的 8.3K。每題實際搬上 GPU 的是 rerank 前的 128 塊：40×4＋64×19＋24×627＝16,424 token。0.5B 每 token 12,288 bytes → 0.19 GiB（6,144 次 H2D）；7B 每 token 57,344 bytes → 0.88 GiB（7,168 次 H2D），比 ReKV-7B 每題上限 0.67 GiB 還多。

**原文之間的不一致**（重現前要自己用資料確認，CLAUDE.md 規則 6）：

1. **RVS 的時長與部數**：
   - ReKV（p.6 表 1）：RVS-Ego **60 分鐘**、10 部；RVS-Movie **30 分鐘**、22 部。ReKV 的效率測試也用「1 小時的 RVS-Ego 影片」（p.8）。
   - MuKV 正文（p.5）：RVS-Ego **11 部、平均 30 分鐘**；RVS-Movie **20 部、平均 1 小時**。
   - MuKV 補充（p.1）：RVS-Ego **10 部**、30 分鐘；RVS-Movie **22 部**、1 小時。
   - 兩篇的時長剛好相反，MuKV 正文和補充的部數也不同。原始出處 Flash-VStream〔未查證〕。
   - 〔複核補充〕用兩個 repo 都指向的 HF 標註檔 `Becomebright/RVS` 驗算：ego 10 部、`duration` 平均 60.1 分鐘、1,465 題；movie 22 部、`duration` 平均 27.9 分鐘、1,905 題。**ReKV 的描述對，MuKV 正文和補充的時長都寫反了**，正文的部數（11、20）也不對。MuKV p.7 用「RVSMovie 影片長得多，例如 1 小時」解釋 Movie 需要較大的壓縮比，這個理由建立在寫反的時長上。
2. **換判分模型就差 7.5 分**：ReKV 7B 在 RVS-Ego 原本報 63.7（GPT-3.5-turbo-0613）；〔複核修正〕MuKV 表 1 用 GPT-3.5-turbo 判分的 ReKV-7B 是 56.2（p.6 表 1）。RVS-Movie 從 54.4 掉到 48.2。MuKV 把差異歸給判分模型，還說只是「slight accuracy differences」（p.5），但沒說明 ReKV 的預測是沿用原本的，還是在 A5000 上重跑，所以 7.5 分不能全部算在判分模型頭上（原文：「MuKV 改用 GPT-3.5-turbo 重新判分後是 56.2」）。〔複核補充〕同一篇裡也不一致：MuKV 表 5（p.7）的 ReKV（LLaVA-OV-7B）Acc@Ego 是 55.8，表 1 是 56.2。表 5 的 #Mem. Tok 5.9K 沒寫基準，數值等於補充表 1 的每 30 格基準。MuKV 表 1 灰字的 FVStream 是 59.0／56.1，ReKV 表 5 的 Flash-VStream-7B 是 57.3／53.1，兩者也不同。
3. **MuKV 的記憶體數字**〔計算〕：
   - 依 §3.2（p.3–4），每段存 784（段）＋196（格）＋49（區塊）＝1,029 token；ReKV 同樣 8 秒存 4×196＝784 token，所以不壓縮時約是 ReKV 的 1.31 倍。
   - 但表 3（p.7）寫不壓縮的 MuKV 是 177K（ReKV 59K 的 3 倍）；表 4（p.7）寫不壓縮的 MuKV 是 3.72 G/h，反而**少於** ReKV 的 4.00 G/h。三者對不上，可能是我對 §3.2 的讀法有誤，要看程式碼 `mukv_multigrain_kvcache.py` 才能確定。
   - 〔複核補充：讀程式碼後的結論〕
     - 卡片對 §3.2 的讀法忠於論文文字（中間那一格＋那一格的一個 super-patch），但**程式碼不是這樣存的**。見上方 I/O 路徑：三種粒度各自鋪滿全部影片 token，每段存 784（段）＋4×196（格）＋16×49（區塊）＝2,352 token，正好是 ReKV 的 3 倍。
     - 每 300 格不壓縮是 3×58,800＝176,400 ≈ **177K**，**對得上表 3**；照 §3.2 文字只有 75×1,029＝77K。
     - 用主實驗 ρ 剪完，每段 64＋76＋627＝767 token（ReKV 的 0.98 倍），每 300 格 57.5K，接近表 1 的「59K、與 ReKV 相同」；也和 §4.3 的「剪 2/3（67%）時與 ReKV 同大小」、p.7–8 的「因為有三種粒度」一致（1−784/2,352＝2/3）。照 §3.2 文字只有 650×75＝48.8K。
     - **表 4 對不上程式碼，也對不上表 3**。依程式碼，0.5B 不壓縮應約 3×4.04＝12.1 GiB/h，剪 2/3 應約 3.95 GiB/h。表 4 卻是 3.72、1.23；1.85、0.91 也剛好約等於 3.72×(1−壓縮比)，所以表 4 是從一個 3.72 的基準縮放出來的，這個基準怎麼來〔未查證〕。
     - 表 3 不壓縮那列的 #Inf. Tok 12.5K 也和主實驗的 k＝{20,32,12} 對不上（不壓縮應是 16,660），那列用的 k 沒寫。
     - 結論：「§3.2 文字 vs 表 3／表 1／程式碼」的矛盾，是 §3.2 的文字描述和實作不一致；「表 4 vs 其他」的矛盾仍然存在。

---

### 綜合判讀（對本研究的意義）

1. **這就是老師說的「長影片持續累積 KV」情境**。MuKV 是 ReKV 的直接後續。〔複核修正〕MuKV 的 repo 是從 ReKV 的程式碼改出來的（KV 管理器檔案逐位元組相同），但主流程換成自己的多粒度儲存，沒有用 ReKV 的 KV 管理器（原文：「兩份程式碼共用同一個 KV 管理器」）。
2. **寫入時少存，在視覺這邊已經有人做了，但是有損的**。〔複核修正〕MuKV 在快取當下、還不知道問題時，就把 3 倍於 ReKV 的多粒度快取剪掉 2/3，淨存量回到和 ReKV 一樣，準確率比不剪更高（表 3，0.5B 推定：53.7→57.3）。把同一套壓縮直接加在 ReKV 上剪掉一半，準確率也上升（51.5→56.1）（原文把「剪掉 2/3」和「ReKV＋DCP 50%」寫在同一句，容易誤讀成同一個實驗）。所以在 CVPR 的語境裡，「少存」的價值是用準確率證明的。無損的分層放置如果準確率不變，很難和這條線比〔判讀〕。
3. **Cake 的位置性質在這裡不成立**〔判讀，待驗證〕：
   - ReKV 用 15K 的滑動視窗編碼，每格的計算量不會隨位置變大；
   - 讀取是依問題挑散落的格，每層挑的還不一樣，不是連續前綴；
   - 所以「前段重算便宜 → 不存」的邏輯不能直接搬過來。另外，在滑動視窗下要精確重算某一格的 KV，需要前面視窗的狀態（多層疊起來，感受野會往前延伸），重算本身就不便宜。
   - 〔複核補充〕以上三點的事實對 ReKV 成立（p.5、p.6 §4.2；`kv_cache_manager.py`）。但**MuKV 的釋出程式碼不同**：每個區塊只接 system prompt 單獨編碼，重算某一塊只需要那一塊的視覺特徵，和位置、前文都無關。所以「重算需要前面視窗的狀態」只適用 ReKV。另外，ReKV 存的是未套 RoPE 的 K，問答時才重新指定連續位置；MuKV 存的是已套 RoPE 的 K，而且每塊的位置都相同。兩者存下來的 K 上的 RoPE 位置都不對應影片時間軸上的位置（`rekv_attention.py`；`mukv_dualsignal_compression.py`）。
4. **系統面是空白**：
   - 兩篇的效率都用 token 數，或端到端的秒數；
   - 沒有拆解搬運時間，沒有磁碟層，沒有多路串流。〔複核修正〕記憶體種類：ReKV 是 pinned CPU RAM 加獨立 CUDA stream 的非同步複製；MuKV 主流程是一般（pageable）CPU RAM 的同步 `.to(device)`（原文：「釋出的程式碼只有 pinned CPU RAM」）。〔複核補充〕ReKV 附錄 A.1 描述了「編碼 process＋問答 process pool」的架構，但沒有實驗，釋出的程式碼也沒有。
   - 7B 每小時 18.8 GiB：10 路串流各跑幾小時就超過單機 RAM，這時磁碟和放置策略才會變成問題〔推論〕。在 ReKV 自己的設定下（單一影片、資料在 RAM、答案 128 token），每題最多搬約 0.67 GiB（命中 LRU 時更少），3.3 s 的延遲很可能主要是解碼，搬運不是瓶頸〔推論，要量〕。
5. **24GB 的卡在 CVPR 是被接受的**：MuKV 的主要實驗用 A5000 24GB。
6. **baseline 走的是 HF 自寫的程式碼，不是 vLLM**。如果做影片情境，第一個要跑通的是 ReKV／MuKV 的 repo，而不是確認 vLLM 的多模態卸載。
7. **評測很脆弱**：〔複核修正〕ReKV-7B 在 RVS-Ego 從 63.7（舊判分模型、ReKV 自己跑）變成 56.2（新判分模型、MuKV 報的），差 7.5 分；MuKV 把差異歸給判分模型，但沒說 ReKV 的預測是不是重跑的（原文：「光換判分模型，ReKV 就差 7.5 分」）。資料集描述在兩篇之間也對不上；用標註檔驗算，是 MuKV 把時長寫反。新的判分模型 `gpt-3.5-turbo` 是會換版本的別名。這和家銘在 HTCS 遇到的 chat template 問題是同一類。
8. **下一步要查的**（MuKV 的參考文獻，皆〔未查證〕）：InfiniPot-V（NeurIPS 2025）、StreamMem（arXiv 2508.15717）、LiveVLM（arXiv 2505.15269）、StreamingVLM（arXiv 2510.09608）、StreamingTOM（arXiv 2510.18269）、Flash-VStream（ICCV 2025）。〔複核補充〕以上書目資訊已對過 MuKV 的參考文獻列表（p.9–11，[19][48][29][46][6][54]），一致；論文本身沒讀，內容仍〔未查證〕。**多頁文件反覆詢問**的情境，這兩篇都沒有涵蓋，要另外查。

---

## 複核紀錄

- **複核者**：zero-context subagent（沒看過本卡的撰寫過程；逐格回到原文與程式碼檢查）
- **日期**：2026-10-07
- **依據**：
  - 論文：ReKV arXiv v1（16 頁）、ICLR proceedings（13 頁，用 difflib 比過，正文只有符號編碼差異）；MuKV CVF 版（11 頁）、補充（3 頁）。全部用 `pdftotext -layout` 依換頁符號標 PDF 頁碼；MuKV 表 4 另外把 p.7 轉成圖片人工核對。
  - 程式碼：兩個 repo 在指定 commit 的完整 tarball（`codeload.github.com`），另外用 `gh api search/code` 交叉查。MuKV 釘住的 transformers commit `66bc4de` 的 `modeling_qwen2.py` 也抓下來確認了 cache 行為。
  - 資料：HF `Becomebright/RVS` 標註檔。
  - arXiv 2605.22269 的標題、作者已在 arxiv.org 確認。
- **檢查格數**：共 62 格。
  - 檔首一句話 1 格：❌ 1
  - ReKV 卡 27 格（3 個標頭項、20 列、4 條 I/O 路徑）：✅ 26、❌ 1、⚠️ 0
  - MuKV 卡 23 格（3 個標頭項、20 列）：✅ 18、❌ 4、⚠️ 1
  - 原文之間的不一致 3 格：✅ 2、❌ 0、⚠️ 1
  - 綜合判讀 8 格：✅ 4、❌ 3、⚠️ 1
  - **合計 ✅ 50、❌ 9、⚠️ 3**
- **特別交辦事項的結論**：
  - a. **兩個 repo 都沒有磁碟層**（成立）。完整原始碼 grep 和 GitHub code search 的 `disk`、`torch.save`、`memmap`、`np.save` 只命中 ReKV 的 README 和 vendored 的 `model/longva/trl/`（存模型權重用）。兩個 repo 都只有 `main` 分支。「只有 pinned CPU RAM」只對 ReKV 成立，MuKV 主流程是 pageable。
  - b. **逐位元組相同**（成立，但結論要改）。從 GitHub 重新抓的兩份 `model/attention/kv_cache_manager.py` 用 `cmp` 比對相同，SHA-1 都是 `26630747…`。但 MuKV 的執行腳本根本沒用到這個檔案。
  - c. **18.8 GB/h 是 GiB**（成立）。附錄 A.3（p.14）的公式 2×L×T×M×H×D×2 bytes，代 L=28、T=1,800、M=196、H=4、D=128，得 20,230,963,200 bytes＝18.84 GiB；0.5B（L=24、H=2、D=64）得 4,335,206,400 bytes＝4.04 GiB。ReKV 自己的 log 也是先除以 1024³ 再標 GB。
  - d. **MuKV 記憶體**：程式碼三種粒度各自鋪滿全部 token，不壓縮是 ReKV 的 3 倍，所以表 3 的 177K 對得上程式碼；§3.2 的文字（只取中間那一格與一個 super-patch）和實作不一致；表 4 的 3.72／1.23 G/h 對不上程式碼，也對不上表 3，原因〔未查證〕。詳見不一致第 3 點。
  - e. **RVS 描述**：卡片照抄兩篇的描述正確；用標註檔驗算，是 MuKV 把時長寫反。
  - f. **63.7→56.2**：數字和位置正確（ReKV p.8 表 5；MuKV p.6 表 1，灰字 63.7 與新值 56.2 都在 MuKV 表 1）。但「只換判分模型」原文沒有支持。
  - g. **綜合判讀第 3、4 點**：第 3 點引用的 ReKV 事實正確，補上 MuKV 程式碼的反例；第 4 點的「pinned」改正。
- **逐條修改**（改前 → 改後；出處）：
  1. ❌ 檔首一句話：「剪掉 2/3 的 KV」「兩篇都只用準確率和 token 數」「論文說會存到磁碟」 → 註明 2/3 是相對於 3 倍的多粒度快取、淨存量等於 ReKV；ReKV 報延遲、GPU 記憶體、KV 大小；磁碟只有 ReKV 論文提到。（ReKV p.8 表 5；MuKV p.6 表 1、p.7 表 3；MuKV 全文無 disk、RAM、CPU 字樣）
  2. ❌ ReKV 單位核對：「拆成 1,792 次小搬運」 → 1,792 次逐格載入，K、V 分開，最多 3,584 次 196 KiB 的 H2D；命中 LRU 時更少，0.67 GiB 是上限。（`kv_cache_manager.py` 的 `MemoryUnit.load`、`get_retrieved_kv`）
  3. ❌ MuKV 模型列的程式碼出處：`llava_onevision_mukv.py` → 主流程 `mukv_rerank.py` 的 `load_model`。（`scripts/run_mukv_rvs_ego.py` 第 19、292 行）
  4. ❌ MuKV 軟體列：「I/O 路徑同上」 → 主流程不經 `ContextManager`，用自己的 `MuKVMultiGrainKVCache`，並新增 MuKV 的 I/O 路徑一節。（`mukv_rerank.py`、`mukv_dualsignal_compression.py`、`mukv_multigrain_kvcache.py`；整個 repo 只有 `llava_onevision_mukv.py` 呼叫 `patch_hf`）
  5. ⚠️ MuKV 主要結果：「0.5B 在 ReKV 上只加壓縮」 → 表 3 沒寫模型大小，依 51.5／42.3 推定為 0.5B〔判讀〕。（MuKV p.7 表 3 標題；p.6 表 1）
  6. ❌ MuKV 設計理由（原文）頁碼：「p.7 圖 5 討論」 → 圖在 p.7，討論在 p.8。（MuKV p.8「Effect of Frequency」段）
  7. ❌ MuKV 原文沒講清楚 (1)：「程式碼同 ReKV」 → 程式碼用 `.cpu()` 存成 pageable tensor 的 Python list。（`mukv_multigrain_kvcache.py` 第 55 行）
  8. ⚠️ 不一致第 2 點：「MuKV 改用 GPT-3.5-turbo 重新判分後是 56.2」 → MuKV 用新判分模型報的 ReKV-7B 是 56.2，論文沒說預測是沿用還是重跑。（MuKV p.5 §4.1「This results in slight accuracy differences」）
  9. ❌ 綜合判讀 1：「兩份程式碼共用同一個 KV 管理器」 → 檔案相同，但 MuKV 主流程不用它。（同第 4 條）
  10. ❌ 綜合判讀 2：把「剪掉 2/3」和「ReKV＋DCP 50%：51.5→56.1」寫成同一件事 → 拆成 MuKV 剪 2/3（53.7→57.3）與 ReKV 剪 1/2（51.5→56.1）兩個實驗。（MuKV p.7 表 3）
  11. ❌ 綜合判讀 4：「釋出的程式碼只有 pinned CPU RAM」 → ReKV 是 pinned 加非同步 stream，MuKV 是 pageable 同步複製；0.67 GiB 改成「最多」。（`kv_cache_manager.py` 第 44–49 行；`rekv_attention.py` 的 `async_global_stream=True`；`mukv_multigrain_kvcache.py` 第 55、100 行；`mukv_rerank.py` 第 242 行）
  12. ⚠️ 綜合判讀 7：「光換判分模型，ReKV 就差 7.5 分」 → 只能說新舊數字差 7.5 分，歸因沒有證據；補上 MuKV 時長寫反、判分模型是別名。（同第 8 條；HF 標註檔）
- **〔複核補充〕清單**（原卡沒錯，只補上與評測設定或 I/O 路徑有關的事實）：
  - ReKV：72B 需 4×80GB；RVS 標註檔驗算；評測程式的處理順序與 `max_new_tokens=256`；A.1 的 process pool 不在程式碼裡；B.3 用的是 Vicuna 骨幹；判分 `temperature=0`；seed 與貪婪解碼；寫入前 15K token 留在 GPU、存的是未套 RoPE 的 K；檢索是未正規化內積、問答上下文不含 local window；程式碼的 GB 就是 GiB。
  - MuKV：完整 I/O 路徑與用量核對（8,212 token 對上表 1 的 8.3K；每題實際搬 16,424 token）；三種粒度依序編碼、每塊單獨 forward；§3.2「接過去的 KV」與程式碼不符；問題向量的取法和式 (6) 不同；判分模型是別名；執行腳本會靜默跳過出錯的影片；表 5 與表 1 的 ReKV 分數不一致；書目已對過參考文獻列表。
