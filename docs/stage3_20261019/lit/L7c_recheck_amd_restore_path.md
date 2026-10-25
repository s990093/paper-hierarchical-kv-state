# L7c：候選 3「AMD 上的快速 KV 還原路徑」與 vLLM 實測（V0／F1／F2）查新複查

> 寫於 2026-10-22（UTC）。執行者：文獻 agent。只讀檔、只上網；**沒有碰 GPU、沒有 import torch／vllm、沒有裝任何套件**。本檔是 **self-review**，不是 cross-model review（CLAUDE.md §5）。
>
> **查證方式**
> - 本機沒有 `gh`。GitHub issue／PR 改用 `curl` 打 GitHub REST API（未登入），讀 `title`、`state`、`state_reason`、`created_at`、`closed_at`、`merged_at` 與 body。狀態一律是 **2026-10-22 查到的**，之後可能會變。
> - 上游程式碼用 `raw.githubusercontent.com` 抓 `main`（2026-10-22）。本機 vLLM 0.28 原始碼在 `/mlsteam/workspace/src/vllm-v0.28.0`（commit `2cf0a6915c`），只用 grep／sed 讀。
> - 論文用 WebFetch 開 arXiv abs 頁（DirectKV 開 USENIX 頁）。WebFetch 會先由小模型轉述頁面，所以經它讀的東西深度最多標〔部分〕。
>
> **標記**
> - 閱讀深度：〔全文〕整份讀過（PR body 很短、整段讀到）；〔部分〕讀了相關段落，或 PR body／diff 只讀到前段；〔摘要〕只讀摘要頁；〔標題〕只看到標題與狀態。
> - 句子層級：〔程式碼〕附檔名與行號；〔算術〕由已知數字算出；〔推論〕來源沒寫、我推出來的；〔判讀〕我對來源的解讀。
> - 「沒查到」只表示這次搜尋沒找到，不等於不存在（CLAUDE.md 規則 7）。NOT_VERIFIED＝有線索，但沒核到原始來源。
> - 不同硬體、不同 ROCm 版本的數字**只並列，不排名**。
>
> **不重抄的部分**：下面這些已經整理過，本檔只補新的、漏掉的、或狀態有變的。
> - L0 §3.2：SDMA／blit、`HSA_ENABLE_SDMA`、hipCOMP、ROCm #11614 初查。
> - L2：C1（vLLM 卸載部落格，DMA 與 kernel copy 的干擾）、C3（AIC）、C6（DirectKV）、D9（ConCCL／DMA-Latte）。
> - L3：F4／缺口 2（#44294、py-kvcache 的單請求前沿）。
> - L6：§2.1、§2.6。
> - **`docs/research_20261010_followup/F1_offload_slowdown.md` §5**：#43660、#45234、#60316、#53695、#43018、#39965，以及 vLLM ROCm 後端部落格，都已在那裡核過。
> - Strata 已有卡：`docs/SOTA_ANALYSIS_20261006.md` §2.3、`docs/research_20260924/kappa_map.md`。
> - AVSG 收在 L1，LMCache SSD 特性量測收在 L5c。

---

## 0. 結論

1. **技術已被做掉**：「GPU kernel 直接讀 pinned host 收集零碎 KV」有 Strata／SGLang HiCache（NVIDIA）、vLLM #42212（已合併，在我們 0.28 樹裡但 ROCm 被 #49843 關掉）、SGLang HiCache 的 ROCm 路徑（AMD 的 #30024、#35233 已合併）。
2. **最大威脅是 vLLM PR #52700**（open）：用 `hipHostRegister(Mapped)` 在 ROCm 重開 OffloadingConnector 的 Triton zero-copy 快路徑，等於候選 3 的實作；其次是 ROCm ≥7.14 的批次 memcpy 已修好（#43018：MI350X 上比逐塊快 6–19 倍），可能不寫 kernel 就消掉大部分差距。
3. **還剩的新意（沒查到有人量過）**：MI300X 上逐塊 SDMA／批次 SDMA／zero-copy kernel 隨片段大小與 GPU 忙碌的交叉點與干擾，以及搬得快之後 F1 的 8K 邊界會不會翻正；但 8K 的損失可能主要來自寫入與排程而不是載入，要先拆解。
4. **F2 上游已知**：「開 connector 退回較慢後端」是 #60316（F1 文件已收）；我們退到 TRITON_ATTN 部分是因為 venv 沒裝 AITER，只能當 MI300X＋OffloadingConnector＋prefill burst 的補充資料點。
5. **建議**：不當方法論文；降成候選 2 的「搬運路徑」維度，加上給上游的 benchmark（在 MI300X 量 #52700）；第一步做不用 GPU 的 F1 寫入／載入拆解。

---

## 1. 對照表

重疊程度的定義：
- **完全重疊**：同一個機制、同一個接點，而且同平台或可以直接套用。
- **部分重疊**：機制相同但平台或引擎不同；或者問題相同但機制不同。
- **相關但不同**：問題相鄰。

### 1.1 搬運技術：zero-copy／GPU 輔助的 gather

| 作品 | 年份／出處 | 做了什麼 | 和我們重疊 | 理由 | 深度 |
|:--|:--|:--|:--|:--|:--|
| **vLLM PR #42212** | 2026-05-10 開，2026-06-03 合併 | OffloadingConnector 的 CPU→GPU 方向：block 數 ≥16、每個描述子 <28 KiB 時，改用 Triton kernel 直接讀 pinned host；用 12 個 SM。H100 端到端 req/s +105–154%，載入頻寬 4.8 → 37.4 GB/s | 部分重疊 | 機制和接點跟候選 3 **完全相同**，差在平台是 NVIDIA；ROCm 上被 #49843 關掉 | 〔部分〕PR body 前段；本機 0.28 程式碼確認存在 |
| **vLLM PR #49843** | 2026-07-25 開，2026-07-27 合併 | ROCm 改回 batch DMA。原因：Triton 直接讀 raw shared-mmap 的 host 指標會 fault | 相關 | 這解釋了為什麼我們的 0.28 在 ROCm 上走逐塊 copy | 〔全文〕 |
| **vLLM PR #52700** | 2026-08-18 開，**open**（標籤 needs-rebase；沒有說明、沒有數字） | ROCm 上對 mmap 區做 `hipHostRegister(…, Mapped)`，取得 device pointer 的位移，重新打開 Triton 快路徑 | **完全重疊** | 這就是候選 3 的實作；缺的只有 MI300X 上的量測與門檻調校 | 〔部分〕三個檔的 diff，`gpu_worker.py` 後段被截 |
| **SGLang HiCache 的 ROCm kernel 路徑** | `transfer.cu`（main，2026-10-22）；PR #30024（2026-08-15 合併）、#35233（2026-09-14 合併） | ROCm 專用的 128-bit nontemporal gather，用 `hipHostGetDevicePointer` 取得 device 位址；ROCm 平台 `hicache_device_kernels=True`；AMD 作者在 MI355X 上調過 | 部分重疊 | 同樣的技術已經在 AMD 上線；差在引擎是 SGLang、硬體是 MI355X，也沒做「短 prompt 卸載邊界」 | 〔部分〕程式碼＋兩個 PR body |
| **Strata**（arXiv 2508.18572） | 2025-08-26 v1，under review（repo 卡片記為 OSDI'26） | 用 GPU-assisted I/O 解決分頁造成的零碎傳輸；H200 上兩個 1024 執行緒的 block 約 50 GB/s，prefill 慢 <5%、decode 慢約 10% | 部分重疊 | 技術的學術前作；平台 NVIDIA | 〔部分〕abs＋HTML §3.1、§4.2、§5 由 WebFetch 轉述 |
| DirectKV（OSDI'26） | USENIX OSDI'26 pp.39–54 | GH200 NVLink-C2C 上的 zero-copy KV 卸載，KV 生成與 attention 融合 | 相關但不同 | 依賴 NVLink-C2C，不是 PCIe 的 MI300X | 〔摘要〕（L2 C6 已收，本次重核 USENIX 頁） |
| **LMCache PR #5308** | 2026-09-22 開，2026-09-24 合併 | **反方向**：把 SM kernel 的 scatter 改成 `cudaMemcpyBatchAsync`。原因：GPU 忙著 prefill 時，每個 scatter kernel 要等約 1.6 ms 才排到，128K 的一次取回約有 50 ms 的 stream 空轉。HIP 上這段沒有編譯 | 相關但不同 | **反證**：kernel 路徑在忙碌的 GPU 上會排隊 | 〔部分〕 |

### 1.2 ROCm 執行期的批次 copy

| 作品 | 年份／出處 | 做了什麼 | 和我們重疊 | 理由 | 深度 |
|:--|:--|:--|:--|:--|:--|
| **vLLM PR #43018** | 2026-05-18 開，2026-08-21 合併 | ROCm 7.2.x 的 `hipMemcpyBatchAsync` 等於沒有批次；7.14／7.15 修好了。MI350X（ROCm 7.15 nightly）上比逐塊 `hipMemcpyAsync` 快 6–19 倍；描述子超過 8192 個會 segfault，所以加了環境變數分段 | 部分重疊 | 用不同的機制解同一個「每次 copy 的固定成本」問題 | 〔全文〕body |
| vLLM issue #44341 | 2026-06-02 開，2026-10-01 以 not_planned 關閉 | 「AMD 卡上沒有 batch 對應，simple CPU offload 很慢」 | 部分重疊 | 和我們 I4 是同一個觀察 | 〔全文〕 |
| vLLM PR #41737 | 2026-05-05 開，2026-05-06 關閉、未合併 | 在 `swap_blocks_batch` 用 `hipMemcpyBatchAsync` | 相關 | 已被 #43018 取代〔判讀〕 | 〔部分〕 |
| vLLM PR #56110 | 2026-09-09 開，open | ROCm 上把相鄰 block 的 DMA 描述子合併；例子：30,208 個描述子只搬約 0.99 GB | 部分重疊 | 是「合併」這條路，F1（10/10）的簡單合併也走過 | 〔部分〕 |
| ROCm/rocm-systems PR #11614 | 2026-09-15 開，**仍 open** | 把 `hipMemcpy3DBatchAsync` 合成一次 SDMA 送出；文中寫「1D 的 `hipMemcpyBatchAsync` 已經透過 `hsa_amd_memory_async_batch_copy` 合併」；MI355X 每次送出 9–13 µs | 相關 | L0 已收；這次更新狀態，並確認 1D 已修 | 〔部分〕 |
| rocm-systems PR #9098、#12904、#11117、#13615 | 2026-07 到 10 月開，全部 open | `hipExtMemcpyBatchAsync` 加 `PreferComputeEngine` 旗標（把線性 copy 改走 shader）；小或沒對齊的 swap 用 blit kernel；host blit 快路徑；分段 copy | 部分重疊 | AMD 正把「kernel copy 還是 SDMA」做成執行期的選項 | #9098、#12904、#11117〔部分〕；#13615〔標題〕 |

### 1.3 ROCm 上 zero-copy／pinned 記憶體的坑

| 作品 | 年份／出處 | 做了什麼 | 和我們重疊 | 理由 | 深度 |
|:--|:--|:--|:--|:--|:--|
| vLLM PR #57160 | 2026-09-16 開，2026-09-18 合併 | 8×MI308X、500 GiB 的卸載池上，`cudaHostRegister` 失敗、引擎起不來。修法：ROCm 改用每個 rank 自己的 pinned tensor，不再用共用的 mmap | 相關 | #52700 依賴的 `pin_mmap_region` 在 ROCm 上可能已經不走了 → #52700 要重寫〔推論〕 | 〔全文〕 |
| SGLang PR #35233 | 2026-09-14 合併 | MI355X 上 `hipDeviceAttributeCanUseHostPointerForRegisteredMem` 是 false：把 CPU 位址直接給 kernel 會 GPU fault，必須換成 device 別名 | 相關 | 和 #49843 是同一類根因 | 〔全文〕body |
| LMCache issue #5361 | 2026-09-26 開，open | ROCm 上 `hipHostRegister` 註冊的頁仍可被搬移 → KFD 把佇列趕出 → MP server 凍結 30–79 s（MI350X、ROCm 7.2） | 相關 | zero-copy 依賴大塊 registered host 記憶體的風險 | 〔部分〕 |
| LMCache issue #4987＋ROCm issue #7520 | 2026-09-06 開／2026-06-19 開，都 open | ROCm 上銷毀已 record 的 IPC event 會等全部 GPU 工作，每次 43–47 ms；CUDA 約 5 µs | 相關但不同 | ROCm 的連接器固定成本另有來源 | 〔部分〕 |
| vLLM issue #52838 | open | AMD 上 GPU↔CPU KV 搬運 fault 會讓整個引擎掛掉 | 相關 | 穩健性 | 〔標題〕 |

### 1.4 注意力後端退回（F2）

| 作品 | 年份／出處 | 做了什麼 | 和我們重疊 | 理由 | 深度 |
|:--|:--|:--|:--|:--|:--|
| vLLM issue #60316 | 2026-10-06 開，**仍 open** | MI355X 上開 connector 後用不了 ROCM_ATTN，decode 每 token 慢 1.9–3.2 倍；開 `VLLM_ROCM_USE_AITER=1` 時縮到 1.1–1.4 倍 | 部分重疊（對 F2） | 同一個機制；差在它量的是 P/D 與 decode，我們量的是 OffloadingConnector、prefill burst、MI300X | F1 文件已核；本次補讀 AITER 那段〔部分〕 |
| vLLM `main` 的 `rocm_attn.py`／`platforms/rocm.py` | 2026-10-22 | `supports_kv_connector()` 仍回 False；開 connector 時，優先序依序是 ROCM_AITER_FA（要開 AITER MHA）→ ROCM_AITER_UNIFIED_ATTN（**只要裝了 aiter**）→ TRITON_ATTN | 狀態 | 我們 venv 沒裝 aiter，所以落到 TRITON_ATTN | 〔部分〕程式碼 |
| vLLM PR #62092 | open | 「在 CDNA3 以上預設開 AITER」 | 相關 | 如果合併，F2 的退回目標會換掉 | 〔標題〕 |
| vLLM 部落格〈Beyond Porting〉 | 2026-02-27，AMD＋Embedded LLM | Qwen3-235B-FP8 在 MI300X 上：TRITON_ATTN 比 AITER_FA 慢 1.30–1.36 倍；ROCM_ATTN 因 head size 退回，慢 2.65–3.82 倍；全文沒提 connector | 相關 | 「退回」要付多少，取決於模型與當成基準的後端 | 〔部分〕 |

### 1.5 「卸載不划算」的邊界（F1）

| 作品 | 年份／出處 | 做了什麼 | 和我們重疊 | 理由 | 深度 |
|:--|:--|:--|:--|:--|:--|
| py-kvcache（arXiv 2609.11744） | 2026-09-10 v1 | 「短前綴或快 GPU 時，重算可能比載入快」；把外部快取當成依環境決定的准入；H100 上平均請求落在打平點以下 | 部分重疊 | 單請求、NVIDIA、NVMe；沒有容量 × 到達率 × 卸載的地圖，也沒有 AMD | 〔摘要〕（L3 已讀正文 p.11、p.20） |
| vLLM PR #35259＋#35342 | #35259：2026-02-25 開，2026-05-02 關閉、未合併。#35342：2026-03-10 合併 | 只在 block 被重用夠多次時才存到 CPU（store gating）；#35259 另有「TTFT 退化 >5% 就暫停卸載」與「載入 vs 重算」成本模型 | 部分重疊 | 上游知道「沒有重用時卸載是白付」，但沒有量出邊界 | #35259〔部分〕；#35342〔標題〕 |
| vLLM RFC #61396 | 2026-10-13 開，open | lazy 卸載：GPU 要覆寫時才存到 CPU。目前是 eager，「the step waits for the store before the forward pass」 | 部分重疊 | 直接對應 F1 8K 變慢的可能原因（寫入成本），不是載入 | 〔部分〕 |
| vLLM #44294／#44295 | issue 2026-10-07 以 not_planned 關閉；PR #44295 **仍 open**（needs-rebase、stale） | — | — | **更正 L3 F4**：L3 寫 #44295「同樣關閉」，實際上還開著 | 〔標題＋狀態〕 |
| vLLM 部落格〈Tiered KV Cache Offloading in vLLM〉 | 2026-09-10 | CPU 主層＋fs／S3／P2P 二級層；只有 2×H100 的收益，沒有「何時變慢」 | 相關但不同 | 沒有短 prompt 或 AMD 的資料 | 〔部分〕 |

### 1.6 AMD 平台的 KV 分層（問題 3）

| 作品 | 年份／出處 | 做了什麼 | 和我們重疊 | 理由 | 深度 |
|:--|:--|:--|:--|:--|:--|
| LMCache 部落格〈AMD × LMCache〉 | 2026-01-09；AMD AI Group＋Tensormesh | MI300X 上開 LMCache 對不開 3–10 倍（長文件 QA、多輪 QA） | 相關但不同 | 是廠商 benchmark，沒有搬運路徑分析 | 〔部分〕 |
| VAST Data＋AMD 部落格 | 2026-07-23 | MI355X、vLLM 0.22.1＋LMCache、128K、NFSv3 over RDMA；宣稱 TTFT 快 9 倍（對重算）；頁面寫 AMD 沒有獨立驗證 | 相關但不同 | 儲存層、長上下文 | 〔部分〕 |
| SoK〈The KV Cache Is the New Memory Wall〉（arXiv 2609.30854） | 2026-09-25 v1 | H100、B200、MI300X 的解析模型與文獻分類；結論是分層把瓶頸推到 PCIe／NVLink | 相關但不同 | 解析模型，不是 MI300X 實測 | 〔摘要〕 |
| Mooncake Transfer Engine 的 ROCm 支援 | #4657、#4340、#3108、#2752 都 open | HIP IPC、dma-buf、ROCm wheel | 相關但不同 | P/D 與 store 的傳輸，不是本機 CPU 還原路徑 | 〔標題〕 |
| NVIDIA Dynamo KVBM 在 AMD 上 | — | 只看到 container、import 相容的 PR（#12537 open、#9929 合併） | **沒查到** KVBM 的 AMD 支援 | — | 〔標題〕 |
| TensorRT-LLM KV 卸載在 AMD 上 | — | 硬體矩陣只列 NVIDIA 架構（L6 §2.3） | **沒查到** AMD 支援 | 〔判讀〕產品本身只支援 NVIDIA | 引 L6 |
| AITER 的 KV 卸載運算子 | — | 在 ROCm/aiter 搜「offload」0 筆、「swap_blocks」2 筆無關 | **沒查到** | — | 搜尋結果 |
| ROCm 文件〈vLLM V1 performance optimization〉 | latest，2026-10-22 讀 | 全頁沒提 KV 卸載或 connector | 沒查到 | AMD 官方的 vLLM 最佳化指南不談卸載 | 〔部分〕 |
| AMD 自己的 MI300X 系統論文 | — | ConCCL、DMA-Latte 是集合通訊（L2 D9）；**KV 分層的 MI300X／MI325X／MI355X 系統論文沒查到** | — | — | 搜尋結果 |

### 1.7 其他 2025–2026 新論文

| 作品 | 年份／出處 | 做了什麼 | 和我們重疊 | 理由 | 深度 |
|:--|:--|:--|:--|:--|:--|
| MMA（arXiv 2512.16056） | 2025-12-18 v1、2026-05-13 v2 | 把一次 host→GPU copy 拆到直接與中繼多條路徑；8×H20 上 245 GB/s，比原生 CUDA copy 快 4.62 倍；KV 取回的 TTFT 少 1.14–2.38 倍 | 相關但不同 | 解的是頻寬聚合，不是小塊固定成本；平台 NVIDIA | 〔摘要〕 |
| AVSG（arXiv 2609.37538） | 2026-09-29 v1 | 稀疏注意力下的 host→NPU 向量化 gather＋HBM 緩衝復用 | 相關但不同 | 平台是 NPU，問題是稀疏取回 | 〔摘要〕（L1 已收） |
| SpecStream（arXiv 2609.33184） | 2026-09-27 v1 | 利用 KV H2D 期間閒著的算力跑 draft；對卸載基準吞吐 1.41／1.32 倍 | 相關但不同 | 處理的是搬運期間的計算重疊 | 〔摘要〕 |
| LMCache SSD 特性量測（arXiv 2610.21249） | 2026-10-15 v1 | SSD 層的可預測性與延遲取捨 | 相關但不同 | 是 SSD 層 | 〔摘要〕（L5c 已收） |

---

## 2. 卡片

### 2.1 vLLM 的 Triton zero-copy 快路徑：CUDA 上線、ROCm 被關、有人要重開（#42212 → #49843 → #52700）

**#42212**〔部分〕（作者 Etelis，2026-06-03 合併）
- **動機（原文）**：`cuMemcpyBatchAsync` 在 CPU→GPU 方向遇到小描述子會崩。H100 上，單次呼叫 N 個描述子時：
  - 4–24 KiB 都卡在約 5–7 GB/s；
  - 28 KiB 突然跳到 39.9–44.1 GB/s（N＝50,000／100,000）。
- **機制**：
  - `_swap_blocks_kernel` 是 grid-stride 的 Triton kernel，每次搬 8 位元組字，`tl.load(src)` 直接讀 pinned host。
  - 只在 CPU→GPU、`n ≥ 16`、每個描述子 `< 28 KiB` 而且 8 位元組對齊時才用。
  - `NUM_SMS = 12`：12、16、20 個 SM 的端到端 req/s 一樣，所以取最小，少搶 decode。
- **數字（原文，H100）**：
  - gpt-oss-120b、TP=2、24 GB 卸載：req/s 在 12K／24K／48K 對 vanilla DMA 高 105–154%；
  - 實得 C→G 頻寬 4.8 → 37.4 GB/s。
  - 交叉點：28 KiB 以上 DMA 贏（DMA 52.3 GB/s 對 Triton 43.2–46.7 GB/s）。
- **在我們的樹裡**〔程式碼〕：0.28 已有 `vllm/v1/kv_offload/cpu/swap_blocks_triton.py`。但 `gpu_worker.py:50` 寫了 `... or current_platform.is_rocm(): return ops.swap_blocks_batch`，所以在 MI300X 上永遠不走 Triton。

**#49843**〔全文〕（2026-07-27 合併）
- 原文：「The ROCm Triton path directly loaded a raw shared-mmap host pointer. The reproduced fault address matched the mmap base plus the selected block offset…」→ ROCm 改走 batch DMA。

**#52700**〔部分〕（nitsingh2，2026-08-18 開；2026-09-13 被 bot 標 merge conflict；沒有任何說明或 benchmark）
- **改法**：
  - ROCm 上對 mmap 區呼叫 `hipHostRegister(base, size, 2 /*Mapped*/)` 與 `hipHostGetDevicePointer`，記下 `device_delta`；
  - Triton kernel 讀之前，把 src 位址加上 delta；
  - 拿不到 delta 就退回 DMA。
- **對候選 3**〔判讀〕：這就是「把 zero-copy gather 接進 OffloadingConnector」的 ROCm 版。差別只有三點：
  1. 它沒有 MI300X 的量測；
  2. 它沿用 H100 調出來的門檻（28 KiB、12 個 SM、`MIN_N=16`）；
  3. 它依賴 shared mmap，但 #57160 之後 ROCm 已經不用 shared mmap（要 rebase、可能要重寫）〔推論〕。

**門檻的細節**（〔算術〕＋〔推論〕，NOT_VERIFIED）
- I4 的 scatter 片段是 32 KiB：`results/m16_stage3_mi300x/i4_copy_batching/copybench_sdma1.csv` 的 4 GiB ÷ 131,072 片。
- Llama-3.1-8B 每層 16-token 的 K 或 V 也是 16×8×128×2 B＝32 KiB；如果 K、V 打包在一起就是 64 KiB。兩種都 ≥28 KiB。
- 所以就算 #52700 照原樣合併，Llama 的描述子在 H100 門檻下**不會走 Triton**。0.28 在這個後端實際送出的描述子大小 NOT_VERIFIED。
- 但在 ROCm 7.2.2 上，I4 量到 32 KiB 時 zero-copy 比逐塊快 18–45 倍。也就是說，**AMD 上的交叉點和 H100 不同，而且取決於 ROCm 版本**——這是少數還沒人量的東西（§3 G2）。

### 2.2 SGLang HiCache 的 ROCm kernel 路徑（AMD 工程師已經做了）〔部分〕

- **程式碼**（main，2026-10-22）〔程式碼〕：
  - `python/sglang/srt/platforms/rocm.py` 宣告 `hicache_device_kernels=True`。
  - `python/sglang/kernels/aot/csrc/kvcacheio/transfer.cu` 有 `#elif defined(USE_ROCM)` 分支，註解：「ROCm: use 128-bit streaming load/store when 16B-aligned, so fewer CUs are needed to saturate the host fabric」。
  - CPU tensor 用 `hipHostGetDevicePointer` 換成 device 位址（`resolve_device_accessible_ptr`）。
- **#30024**（TianDi101 等 AMD 作者，2026-08-15 合併），MI355X 上：
  - MLA gather 改 128-bit：block_quota＝8 時 +32%，D2D +36%；
  - 修好 `SGLANG_HICACHE_BLOCK_QUOTA` 在執行期沒生效的問題：128K 前綴每層 gather 17.9 → 2.5 ms；
  - HIP 預設 block_quota 2 → 16：Kimi-K2.6 1P1D 的 prefill_forward 少約 27.6%，端到端 TTFT 少約 21%。
- **#35233**（2026-09-14 合併）：MI355X 上 registered host 記憶體的 CPU 位址與 GPU 位址不同，直接把 CPU 位址給 kernel 會 GPU fault；改成在 kernel 啟動邊界換成 device 別名。沒有跑速度測試。
- **對候選 3**〔判讀〕：
  - 「AMD 上用 GPU kernel 直接讀 host 來還原 KV」不但做過，而且由 AMD 在調。
  - 我們沒找到它在 MI300X 上的數字，也沒找到它和逐塊 SDMA 在不同 prompt 長度下的端到端比較。
  - SGLang 的 AMD 文件（`docs/docs/hardware-platforms/amd_gpu.mdx`）沒提 HiCache：沒查到。

### 2.3 ROCm 的批次 memcpy 已經修了（#43018 的數字＋執行期 PR）

- **#43018**〔全文〕（hongxiayang，AMD；2026-08-21 合併）：
  - 原文：「In ROCm 7.2.3, the performance of hipMemcpyBatchAsync is almost the same (i.e., 1x) as hipMemcpyAsync, as there is no batching implemented」。
  - ROCm 7.15 nightly、8×MI350X（gfx950）上，batch 對逐塊：

    | 方向 | 片段 | batch | 逐塊 | 倍數 |
    |:--|--:|--:|--:|--:|
    | H2D | 4 KiB | 31.5 GB/s | 1.7 GB/s | ×19.1 |
    | H2D | 64 KiB | 49.0 GB/s | 5.8 GB/s | ×8.4 |
    | D2H | 4 KiB | 25.4 GB/s | 1.6 GB/s | ×15.4 |
    | D2H | 64 KiB | 48.9 GB/s | 6.3 GB/s | ×7.7 |

  - 描述子超過 8192 個會 segfault，所以加了 `VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS` 分段；修正不在 7.14x。
- **並列**（不同硬體、不同 ROCm 版本，不排名）：
  - 我們的 I4：MI300X、ROCm 7.2.2、32 KiB scatter。zero-copy 53.1 GiB/s〔≈57.0 GB/s，算術〕；逐塊 1.1–3.0 GiB/s。
  - #43018：MI350X、ROCm 7.15、64 KiB batch H2D 49.0 GB/s。
  - 〔判讀〕升級 ROCm 之後，「zero-copy 對 batch DMA」的差距可能從 18–45 倍縮到個位數甚至接近 1。我們沒有在 ROCm ≥7.14 上量過：NOT_MEASURED。
- **執行期還在動**（rocm-systems，全部 open）：
  - #11614：3D batch 合併成一次 SDMA；文中說 1D 已經合併。
  - #9098：`hipExtMemcpyBatchAsync`，含 `hipMemcpyFlagExtPreferComputeEngine`（把線性 copy 改走 shader）。
  - #12904：小或沒對齊的 swap 用 blit kernel，門檻可用 `GPU_FORCE_BLIT_SWAP_SIZE` 調。
  - 〔判讀〕AMD 打算讓執行期自己在 SDMA 與 blit kernel 之間選，連「自己寫 gather kernel」的必要性都在變小。

### 2.4 Strata 的 GPU-assisted I/O（學術前作）〔部分〕

- **書目**：Zhiqiang Xie、Ziyi Xu、Mark Zhao、Yuwei An、Vikram Sharma Mailthody、Scott Mahlke、Michael Garland、Christos Kozyrakis。arXiv 2508.18572 v1（2025-08-26），頁面寫 under peer review。
- **轉述的重點**（WebFetch 讀 HTML）：
  - page＝32、8192 token 時只用到 PCIe 5.0 理論頻寬的約 22%；Grace-Hopper 上低到 5%。
  - GPU kernel 從「CPU registered pinned memory」逐小塊讀；用少量大 block，最少 1 個 SM。
  - H200 上兩個 1024 執行緒的 block 約 50 GB/s，prefill 慢 <5%、decode 慢約 10%。
- **沒做的**：短上下文時「載入不如重算」的交叉分析；AMD。
- **對我們**：這是「zero-copy gather 對零碎 KV」最直接的學術前作，候選 3 的技術新意因此歸零〔判讀〕。

### 2.5 反證：kernel 路徑在忙碌的 GPU 上會排隊（LMCache #5308）〔部分〕

- Kimi-Linear-48B、TP 2、128K 前綴：GPU 忙著做 vLLM 的 prefill 時，每個 scatter kernel 要等約 1.6 ms 才排到，一次 128K 取回約 50 ms 的 stream 空轉，而 copy 本身已經跑到線速。
- 改法：能用 `cudaMemcpyBatchAsync` 就用 copy engine，把 SM 留給 vLLM。這個路徑「compiled out on HIP」。
- **和其他證據並看**〔判讀〕：
  - vLLM 部落格（L2 C1）：H100 上自訂 copy kernel 在 0% 命中時比不卸載差 6%。
  - #42212：選最少的 SM 以免搶 decode。
  - Strata：decode 慢約 10%。
  - 結論：zero-copy kernel 的頻寬優勢，在 GPU 忙時要用干擾或排隊延遲來換。**MI300X 上這個取捨沒人量過**。我們 I4 的 busy2 只看了搬運頻寬，沒看對共跑計算的影響。這一點正好接上候選 2。

### 2.6 ROCm 上做 zero-copy 的坑（工程風險清單）

| 坑 | 平台 | 出處 |
|:--|:--|:--|
| 直接讀 raw shared-mmap 的 host 指標會 fault | ROCm（vLLM CI） | #49843〔全文〕 |
| registered 記憶體的 CPU 與 GPU 位址不同（`CanUseHostPointerForRegisteredMem`＝false） | MI355X | SGLang #35233〔全文〕 |
| 大池 `hipHostRegister` 失敗、啟動卡住 | 8×MI308X、500 GiB | vLLM #57160〔全文〕 |
| `hipHostRegister` 的頁仍可被搬移 → KFD 把佇列趕出、凍結 30–79 s | MI350X、ROCm 7.2 | LMCache #5361〔部分〕；原文的修法是先 `mlock` 再 register，並設 `vm.compact_unevictable_allowed=0` |
| IPC event 銷毀 43–47 ms | ROCm 7.2.x | ROCm #7520、LMCache #4987〔部分〕 |
| 批次描述子 >8192 個會 segfault | ROCm 7.14／7.15 | vLLM #43018〔全文〕 |

〔判讀〕「在 ROCm 上可靠地做 zero-copy」的坑已經被各專案零星踩過、各自修掉；**沒查到系統性的整理或論文**。這只值一篇工程經驗文，撐不起一篇研究論文。

### 2.7 F2：後端退回已知；我們的「慢一倍」部分來自沒裝 AITER

- F1 文件 §5 已核：#43660（加入排除）、#45234（作者自己關掉，「deprecating ROCM_ATTN in favor of AITER_FA」）、#60316（open）、#53695（2026-09-10 合併，讓 ROCM_AITER_UNIFIED_ATTN 支援 connector）。
- **本次新增**：
  - 2026-10-22 的 `main` 仍是 `supports_kv_connector() -> False`。原因寫在註解：「(2, num_blocks, ...) KV cache layout which is incompatible with KV connectors that require blocks-first layout」。最後一次改這個檔是 2026-10-22 的 `9d61ade7f`，跟這件事無關。
  - #60316 原文：不開 AITER 時，Qwen3 decode 慢 1.9–3.2 倍；`VLLM_ROCM_USE_AITER=1` 時開 connector 會改用 ROCM_AITER_FA，差距縮到 1.1–1.4 倍，「but does not close」。
  - 〈Beyond Porting〉部落格：Qwen3-235B-FP8 在 MI300X 上，TRITON_ATTN 比 AITER_FA 慢 1.30–1.36 倍。
  - #62092（open）要在 CDNA3 以上預設開 AITER。
- **〔判讀〕對 F2 的影響**：
  - F2 的「B 對 A 慢約 2 倍」是「ROCM_ATTN 對 TRITON_ATTN」，前提是 venv 沒裝 aiter（F1 文件 §5.1 已記）。
  - 照 vLLM 的優先序〔程式碼 `platforms/rocm.py`〕，用官方的 ROCm 映像（有 aiter）時，開 connector 應該落到 ROCM_AITER_UNIFIED_ATTN，開 AITER 時落到 ROCM_AITER_FA。代價可能小很多：MI300X 上 NOT_MEASURED。
  - 所以 F2 的結論要寫成「在沒有 AITER 的 vLLM 0.28 上」，不能泛化成「ROCm 上開卸載就慢一倍」。

### 2.8 F1：「8K 開卸載變慢」的邊界

- **前作**：
  - py-kvcache：單請求的打平前沿、准入閘門；H100 上平均請求落在打平點以下（L3 已讀正文）。
  - vLLM 上游：#35342（store gating，2026-03-10 合併）；#35259（成本模型＋自動暫停，未合併）；RFC #61396（lazy 卸載，open）。
  - 原文提到 eager 寫入時「the step waits for the store before the forward pass」。
- **沒查到**：在真實引擎上，同時掃長度 × 容量 × 到達率 × 卸載、畫出「卸載比只用 APC 差」的地圖，特別是 AMD——這和 L3 缺口 2 的結論一致。
- **〔判讀〕對候選 3 最要緊的一點**：F1 的 8K 損失可能主要來自**寫入**（eager store 卡住 step）或排程，不是**載入**。F1 在 c040、重疊 0% 時最差（−112%），而重疊 0% 時沒有東西可以從 CPU 載回。如果是這樣，把載入換成 zero-copy **翻不正 8K 的邊界**。這要先用 F1 已經收的 `vllm:kv_offload_store_time`／`load_time`、`store_bytes`／`load_bytes` 拆解（summary JSON 裡有這些欄位；是不是每一格都有樣本：NOT_VERIFIED）。

---

## 3. 缺口清單與「被做掉的部分」

### 3.1 被做掉的部分（不要再當新意寫）

| 主張 | 誰做掉的 |
|:--|:--|
| 「零碎 KV 用 GPU kernel 直接讀 pinned host，比逐塊 copy 快很多」 | Strata（2025，學術）、SGLang HiCache（含 ROCm 路徑，#30024、#35233）、vLLM #42212（CUDA，已合併） |
| 「把 zero-copy gather 接進 vLLM OffloadingConnector 的 ROCm 路徑」 | vLLM #52700（open，實作已寫好，沒有數字） |
| 「vLLM 在 AMD 上 CPU 卸載慢，是因為每次 copy 的固定成本／`hipMemcpyBatchAsync` 沒批次」 | vLLM #44341、#43018（AMD 工程師，有 MI350X 數字）；ROCm 7.14 以上的執行期已修 |
| 「開 connector 會讓 ROCm 退到較慢的注意力後端」 | vLLM #60316（＋#43660、#45234、#53695；F1 文件已收） |
| 「沒有重用時開卸載是白付，短 prompt 重算比載入快」 | py-kvcache（准入閘門）；vLLM #35342、#35259、#61396 |
| 「kernel copy 會和計算互搶」 | vLLM 部落格（L2 C1）、Strata、LMCache #5308（CUDA） |

### 3.2 還剩的缺口（這次都沒查到有人做；要做之前得先過 G0）

| # | 缺口 | 為什麼還算缺口 | 和我們手上資料的距離 |
|:--|:--|:--|:--|
| **G0（先決）** | F1 的 8K 損失裡，寫入、載入、排程各佔多少 | 如果載入不是主因，候選 3 的端到端目標根本不成立 | **不用 GPU**：拆 F1 已有的 offload metrics |
| G1 | MI300X 上，換掉載入路徑之後，8K 的卸載邊界會不會翻正（端到端 TTFT） | #42212 只在 H100 上量吞吐；SGLang ROCm 只在 MI355X 量 1P1D；沒人量「短 prompt 的邊界」 | 要跑 #52700（或自己的 kernel）＋F1 的 8K 格；租約 10/23 09:00Z 到期，這次做不到 |
| G2 | AMD 上「逐塊 SDMA／批次 SDMA（ROCm ≥7.14）／zero-copy kernel」三種搬法，隨片段大小（4–64 KiB）與 GPU 忙碌的交叉點 | H100 的交叉點是 28 KiB（#42212）；AMD 上沒人量；而且隨 ROCm 版本會變 | I4 只有 32 KiB、ROCm 7.2.2，沒有批次 SDMA 這一組 |
| G3 | zero-copy kernel 對**共跑計算**的拖慢、在忙碌 GPU 上的排隊延遲（MI300X） | CUDA 上有零星證據（L2 C1、#5308、Strata）；MI300X 的 KV 還原沒人量 | I4 的 busy2 只量了搬運頻寬；X3 的翻轉實驗可以加一個「路徑」軸 → 併進候選 2 |
| G4 | 有 AITER 時，F2 的退回代價（MI300X、OffloadingConnector、prefill burst） | #60316 是 MI355X decode／P/D | 要裝 aiter（venv 沒有）；這次做不到 |
| G5 | ROCm 上 zero-copy 的穩健性（可搬移的頁、別名位址、大池註冊）系統性整理 | 只有各專案零星的 issue／PR | 工程價值，不是研究 |

---

## 4. 建議的定位

1. **候選 3 不要當獨立的方法論文**。技術前作（Strata、SGLang HiCache 的 ROCm 路徑、vLLM #42212）和直接的實作（#52700）都在了；ROCm ≥7.14 的批次 memcpy 還可能不用 kernel 就把差距吃掉。寫成「我們提出 zero-copy gather」會被審稿人一句話打掉〔判讀〕。
2. **降成候選 2 的一個維度**：還原時要選的不只是「格式」（INT4／低秩組合／BF16），還有「路徑」（SDMA 逐塊、SDMA 批次、kernel zero-copy）。兩者都隨 GPU 忙碌翻轉：kernel 路徑搶 CU，SDMA 不搶但固定成本高。候選 2 的主張「前人假設搬運和計算互不干擾」可以直接延伸到路徑選擇。這裡要引用 L2 C1、Strata、LMCache #5308 當動機，並說明 MI300X 上沒人量過（G2、G3）。
3. **工程面當成上游貢獻**：
   - 在 MI300X 上量 #52700（照實 credit），回報 AMD 的門檻應該和 H100 不同（I4：32 KiB 時 zero-copy 遠勝 ROCm 7.2.2 的 batch）；
   - 回報 #57160 之後 #52700 的 mmap 前提可能失效；
   - 把 F2 的 MI300X 數字補到 #60316（註明沒有 AITER）。
   
   這是 issue／PR 等級的價值，不是論文貢獻。這和 L3 缺口 2 的「只重現 #44294 只值一個工程 issue」是同一種判斷。
4. **順序**：
   - (a) 不用 GPU，拆 F1 的寫入、載入、排程（G0）。載入不是主因 → 候選 3 的端到端目標收掉。
   - (b) 有新的 GPU 時間時，先把 ROCm 升到 ≥7.14、量批次 SDMA，再決定 kernel 路徑有沒有剩下的空間（G2）。
   - (c) G3 併進候選 2 的真實併發實驗。
   - MI300X 租約 10/23 09:00Z（台灣 17:00）到期，以上都不在這次租約內。
5. **寫 F1／F2 時的措辭**：
   - F2 要寫「vLLM 0.28、沒有 AITER、ROCm 7.2.2」；
   - F1 的 8K 邊界要寫「eager 寫入（0.28 預設）」，並引用 #61396 說明上游正在改；
   - 「#44295 已關閉」要更正成 open（L3 F4）。

---

## 5. 參考清單

核對欄說明：
- 「✓ API」：本次用 GitHub REST API 讀到 title、state、日期（2026-10-22）。
- 「✓ abs」：本次 WebFetch 開過 arXiv abs 頁。
- 「✓ 頁」：本次開過官方頁。
- 「✓ 碼」：本次讀過原始碼。
- 「既有」：沿用本 repo 其他檔已核的結果，本次沒有重開。

### 5.1 vLLM（GitHub）

| # | 標題（原文） | 狀態（2026-10-22） | URL | 核對 |
|:--|:--|:--|:--|:--|
| V1 | [Perf] Triton fast path for small CPU→GPU `swap_blocks_batch` in the offloading connector | 合併 2026-06-03 | https://github.com/vllm-project/vllm/pull/42212 | ✓ API |
| V2 | [Bugfix][ROCm] Use batch DMA for CPU KV cache loads | 合併 2026-07-27 | https://github.com/vllm-project/vllm/pull/49843 | ✓ API |
| V3 | [Bugfix][ROCm] Restore KV-offload Triton fast path via mapped device … | **open**（2026-08-18 開；needs-rebase） | https://github.com/vllm-project/vllm/pull/52700 | ✓ API（含 diff） |
| V4 | [ROCm] Cpu offload for ROCm 7.13+ to align the hipMemcpyBatchAsync params and perf in 7.14x | 合併 2026-08-21 | https://github.com/vllm-project/vllm/pull/43018 | ✓ API |
| V5 | [ROCm] Simple CPU offload is slow without hipMemcpyBatchAsync mapping | 關閉（not_planned）2026-10-01 | https://github.com/vllm-project/vllm/issues/44341 | ✓ API |
| V6 | [ROCm] Enable SimpleCPUOffloadConnector on ROCm | 合併 2026-05-07 | https://github.com/vllm-project/vllm/pull/40549 | ✓ API |
| V7 | [Perf][ROCm] Use hipMemcpyBatchAsync in swap_blocks_batch | 關閉、未合併 2026-05-06 | https://github.com/vllm-project/vllm/pull/41737 | ✓ API |
| V8 | [Perf] Batch KV cache swap copies via cuMemcpyBatchAsync | 合併 2026-04-03 | https://github.com/vllm-project/vllm/pull/38460 | ✓ API |
| V9 | [ROCm][Performance] Coalesce contiguous SimpleCPUOffload DMA copies | open（2026-09-09 開） | https://github.com/vllm-project/vllm/pull/56110 | ✓ API |
| V10 | [Bugfix][ROCm][KV Offload] Use private pinned tensors for CPU KV offload | 合併 2026-09-18 | https://github.com/vllm-project/vllm/pull/57160 | ✓ API |
| V11 | [Bugfix][AMD] A GPU-CPU KV transfer fault in OffloadingConnector takes the engine down instead of degrading the cache. | open | https://github.com/vllm-project/vllm/pull/52838 | 標題（搜尋結果） |
| V12 | [Performance][ROCm] KV connectors rule out ROCM_ATTN, so PD workers decode on ROCM_AITER_UNIFIED_ATTN … | open（2026-10-06 開） | https://github.com/vllm-project/vllm/issues/60316 | ✓ API（F1 文件也已核） |
| V13 | [ROCm][Feature] Support KV connectors with ROCM_AITER_UNIFIED_ATTN | 合併 2026-09-10 | https://github.com/vllm-project/vllm/pull/53695 | ✓ API |
| V14 | [ROCm] Enable ROCm Attention Sinks and Connector-Friendly KV Layouts | 關閉、未合併 2026-07-28 | https://github.com/vllm-project/vllm/pull/45234 | ✓ API |
| V15 | [ROCm][AITER] Enable AITER by default on CDNA3+, keep AMD CI on the non-AITER path | open | https://github.com/vllm-project/vllm/pull/62092 | 標題（搜尋結果） |
| V16 | [Bug][OffloadingConnector] _blocks_being_loaded serialises concurrent requests … | 關閉（not_planned）2026-10-07 | https://github.com/vllm-project/vllm/issues/44294 | ✓ API |
| V17 | [Bugfix][OffloadingConnector] avoid request convoy when concurrent requests share blocks being loaded | **open**（needs-rebase、stale） | https://github.com/vllm-project/vllm/pull/44295 | ✓ API（更正 L3） |
| V18 | Feature/zero cost kv offload | 關閉、未合併 2026-05-02 | https://github.com/vllm-project/vllm/pull/35259 | ✓ API |
| V19 | feat(kv-offload): Strategy A — StoreReusedOffloadingManager gates CPU stores on reuse frequency | 合併 2026-03-10 | https://github.com/vllm-project/vllm/pull/35342 | 標題＋合併日（搜尋結果） |
| V20 | [RFC] [KV Offload]: Store triggers and store modes for the offloading connector (lazy offloading) | open（2026-10-13 開） | https://github.com/vllm-project/vllm/issues/61396 | ✓ API |
| V21 | vLLM `main`：`vllm/v1/attention/backends/rocm_attn.py`、`vllm/platforms/rocm.py`、`vllm/v1/kv_offload/cpu/{gpu_worker,swap_blocks_triton}.py`、`vllm/_aiter_ops.py` | 2026-10-22 | https://github.com/vllm-project/vllm/tree/main | ✓ 碼 |
| V22 | 本機 vLLM 0.28：`vllm/v1/kv_offload/cpu/gpu_worker.py:50`、`swap_blocks_triton.py`、`vllm/platforms/rocm.py:481-495`、`csrc/libtorch_stable/cache_kernels.cu:155-173` | commit `2cf0a6915c` | `/mlsteam/workspace/src/vllm-v0.28.0` | ✓ 碼 |

### 5.2 SGLang、LMCache、Mooncake、Dynamo（GitHub）

| # | 標題（原文） | 狀態 | URL | 核對 |
|:--|:--|:--|:--|:--|
| S1 | [AMD] perf(sgl-kernel): default block_quota=16 for MLA page_first KV gather… | 合併 2026-08-15 | https://github.com/sgl-project/sglang/pull/30024 | ✓ API |
| S2 | [AMD] Fix registered HiCache host pointer aliases | 合併 2026-09-14 | https://github.com/sgl-project/sglang/pull/35233 | ✓ API |
| S3 | SGLang `main`：`python/sglang/kernels/aot/csrc/kvcacheio/transfer.cu`、`python/sglang/srt/platforms/rocm.py`、`python/sglang/srt/mem_cache/memory_pool_host.py` | 2026-10-22 | https://github.com/sgl-project/sglang/tree/main | ✓ 碼 |
| S4 | [Refactor] [AMD] hicache: Isolate HIP host allocation | open | https://github.com/sgl-project/sglang/pull/42991 | 標題 |
| L1 | [Perf][MP] Move LMCache-driven transfer to cudaMemcpyBatchAsync | 合併 2026-09-24 | https://github.com/LMCache/LMCache/pull/5308 | ✓ API |
| L2 | [MP][ROCm][Bug] L1 pinned via hipHostRegister stays movable: compaction triggers KFD queue eviction, MP server freezes 30–79 s | open | https://github.com/LMCache/LMCache/issues/5361 | ✓ API |
| L3 | [MP][ROCm] `LMCacheMPConnector` recreates an IPC event per transfer — on ROCm each destructor drains the whole device | open | https://github.com/LMCache/LMCache/issues/4987 | ✓ API |
| L4 | [ROCm] stable pypi and nightly wheels of ROCm LMCache | open | https://github.com/LMCache/LMCache/issues/4029 | 標題 |
| M1 | Mooncake：[TransferEngine][ROCm] Keep KFD free VRAM exact when registering HIP memory through dma-buf（#4657）；[TE] Fix HIP and NVLink IPC relocation for suballocated buffers（#4340）；[Installation] Publish prebuilt ROCm/HIP wheels to PyPI（#3108） | 都 open | https://github.com/kvcache-ai/Mooncake/pulls | 標題 |
| D1 | Dynamo：feat(container): add rocm device for vllm and sglang（#12537，open）；fix: import-time compat for AMD ROCm / Python 3.10 hosts（#9929，合併 2026-05-31） | — | https://github.com/ai-dynamo/dynamo | 標題；KVBM 的 AMD 支援**沒查到** |

### 5.3 ROCm（GitHub）

| # | 標題（原文） | 狀態 | URL | 核對 |
|:--|:--|:--|:--|:--|
| R1 | feat: fuse hipMemcpy3DBatchAsync into a single SDMA submission | open（2026-09-15 開） | https://github.com/ROCm/rocm-systems/pull/11614 | ✓ API（L0 已收；狀態更新） |
| R2 | feat (clr) : Add hipExtMemcpyBatchAsync with asymmetric swap | open（2026-07-23 開） | https://github.com/ROCm/rocm-systems/pull/9098 | ✓ API |
| R3 | [CLR] Add Swap blit kernel | open（2026-10-06 開） | https://github.com/ROCm/rocm-systems/pull/12904 | ✓ API |
| R4 | clr: add HostBlitManager fast path to WriteBufferBatch/ReadBufferBatch | open（2026-09-02 開） | https://github.com/ROCm/rocm-systems/pull/11117 | ✓ API |
| R5 | feat(clr): enabled chunked copies in hipMemcpyBatchAsync | open（2026-10-16 開） | https://github.com/ROCm/rocm-systems/pull/13615 | 標題 |
| R6 | [CLR/HIP] hipEventDestroy() on a recorded IPC event blocks on SyncAllStreams() … | open（2026-06-19 開） | https://github.com/ROCm/rocm-systems/issues/7520 | ✓ API |
| R7 | ROCm/aiter 搜尋「offload」「swap_blocks」 | 0 筆／2 筆無關 | https://github.com/ROCm/aiter | 搜尋；**沒查到** |

### 5.4 論文

| # | 書目 | URL | 核對 |
|:--|:--|:--|:--|
| P1 | Z. Xie, Z. Xu, M. Zhao, Y. An, V. S. Mailthody, S. Mahlke, M. Garland, C. Kozyrakis, "Strata: Hierarchical Context Caching for Long Context Language Model Serving," arXiv 2508.18572 v1（2025-08-26） | https://arxiv.org/abs/2508.18572 | ✓ abs＋HTML |
| P2 | S. Luo, H. Shen, "No Buffer, No Bottleneck: Efficient Zero-Copy KV Cache Offloading for Long-Context LLMs"（DirectKV），OSDI '26, pp.39–54 | https://www.usenix.org/conference/osdi26/presentation/luo | ✓ 頁 |
| P3 | L. Tang *et al.*, "MultiPath Memory Access: Breaking Host-GPU Bandwidth Bottlenecks in LLM Services," arXiv 2512.16056 v2（2026-05-13） | https://arxiv.org/abs/2512.16056 | ✓ abs |
| P4 | W. Kuang *et al.*, "AVSG: Accelerated Vectorized Sparse Gather for Efficient KV Cache Offload in Sparse-Attention LLM Serving," arXiv 2609.37538 v1（2026-09-29） | https://arxiv.org/abs/2609.37538 | ✓ abs |
| P5 | F. Li, S. Liu, S. Nie, J. Wang, W. Wu, "Resource-Efficient Speculative Decoding for Long-Context LLM Serving"（SpecStream），arXiv 2609.33184 v1（2026-09-27） | https://arxiv.org/abs/2609.33184 | ✓ abs |
| P6 | T. Singh, "The KV Cache Is the New Memory Wall," arXiv 2609.30854 v1（2026-09-25） | https://arxiv.org/abs/2609.30854 | ✓ abs |
| P7 | J. Kanichai, T. De Matteis, A. Trivedi, "Building py-kvcache: A Performance Characterization of External KV Caching for vLLM with NVMe SSDs," arXiv 2609.11744 v1（2026-09-10） | https://arxiv.org/abs/2609.11744 | ✓ abs（正文見 L3） |
| P8 | Y. He *et al.*, "Characterizing Predictability-Latency Trade-offs of KV-Cache SSD Offloading in LMCache for LLM Serving Systems," arXiv 2610.21249 v1（2026-10-15） | https://arxiv.org/abs/2610.21249 | ✓ abs |

### 5.5 部落格與官方文件

| # | 標題 | 日期 | URL | 核對 |
|:--|:--|:--|:--|:--|
| B1 | vLLM〈Tiered KV Cache Offloading in vLLM〉（Ozeri、Harnik、Schaffer、Etelis、Sundar Rabindranath） | 2026-09-10 | https://vllm.ai/blog/2026-09-10-tiered-kv-offloading | ✓ 頁 |
| B2 | vLLM〈Beyond Porting: How vLLM Orchestrates High-Performance Inference on AMD ROCm〉（AMD＋Embedded LLM） | 2026-02-27 | https://vllm.ai/blog/rocm-attention-backend | ✓ 頁 |
| B3 | LMCache〈AMD × LMcache: AMD GPU Acceleration with LMcache〉 | 2026-01-09 | https://blog.lmcache.ai/en/2026/01/09/amd-x-lmcache-amd-gpu-acceleration-with-lmcache | ✓ 頁（只讀前 100K 字元；圖表沒讀） |
| B4 | VAST Data〈Beyond HBM Limits: Accelerating Inference with VAST + AMD〉 | 2026-07-23 | https://www.vastdata.com/blog/beyond-hbm-limits-accelerating-inference-with-vast-amd | ✓ 頁 |
| B5 | ROCm 文件〈vLLM V1 performance optimization〉 | latest（2026-10-22 讀） | https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html | ✓ 頁（沒提卸載） |
| B6 | vLLM〈Inside vLLM's New KV Offloading Connector〉 | 2026-01-08 | https://vllm.ai/blog/2026-01-08-kv-offloading-connector | 既有（L2 C1〔全文〕） |
| B7 | ROCm〈Introducing ROCm AMD Infinity Context〉 | 2026-07-22 | https://rocm.blogs.amd.com/software-tools-optimization/amd-infinity-context/README.html | 既有（L2 C3）；這次搜尋沒找到之後公開的效能數字 |

### 5.6 本 repo 的出處

- `docs/research_20261010_followup/F1_offload_slowdown.md` §5（上游狀態：#43660、#45234、#60316、#53695、#43018、#39965）
- `docs/stage3_20261019/experiment_log.md`（I4：`20261020-142741-m16-i4-sdma0`、`20261020-143218-m16-i4-sdma1`；F2、F1 的 run 與判讀）
- `results/m16_stage3_mi300x/i4_copy_batching/copybench_sdma1.csv`（4 GiB／131,072 片＝32 KiB）
- `results/m16_stage3_mi300x/f1_prefix_reuse/summary_*.json`（有 `kv_offload_store_time_total`、`kv_offload_load_time_total` 等欄位；本檔沒有計算任何數字）

### 5.7 查了沒找到

- MI300X、MI325X、MI355X 上做 KV 分層或卸載的**系統論文**（只有廠商部落格、SoK 解析模型，以及 AMD 的集合通訊 DMA 論文）。
- 在真實引擎上同時掃長度 × 容量 × 到達率 × 卸載的「卸載比只用 APC 差」地圖（任何平台）。
- AMD 上 KV 還原路徑「SDMA 對 kernel copy」在有背景負載時的量測。
- #52700 或 SGLang ROCm HiCache 在 MI300X 上的數字。
- NVIDIA Dynamo KVBM、TensorRT-LLM 的 AMD 支援；AITER 的 KV 卸載運算子。
- AMD 官方（ROCm 文件或部落格）針對 vLLM CPU 卸載路徑的最佳化文章：只在上游 PR（#43018、#40549）看到 AMD 工程師的工作。
