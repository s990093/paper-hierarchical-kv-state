# R0：復現 Strata 與 KVFlow（計畫，**尚未登記**）

> 回到：[README.md](README.md)
> **性質**：計畫。先做可行性檢查（程式能不能在 H200 上跑、模型支不支援），再登記「怎樣算復現成功」的判準。**這一頁沒有判準，也還沒有任何數字。**
> **為什麼要復現**：候選 B 要主張「Strata＋改一處更好」，審稿人會要求同一台機器、同一組資料上的 Strata 與 KVFlow 數字（L2 §3.4）。

---

## 1. 程式在哪裡（2026-10-26 查）

| 系統 | 程式 | 狀態 |
|:--|:--|:--|
| **KVFlow**（NeurIPS'25） | `github.com/PanZaifeng/KVFlow`（Apache-2.0；SGLang 的 fork；README 寫是 KVFlow 與 ScaleSim 的原始碼；最後推送 2026-03-12） | **有官方程式**。repo 先前的 L3 §H4 寫「官方程式 NOT_FOUND」，要更正 |
| **Strata**（OSDI'26） | GitHub 搜尋沒有找到排程器的程式。I/O 部分已進上游 SGLang（`--hicache-io-backend kernel`、`--hicache-mem-layout page_first`；L1 §3.1） | 排程器（delay hit、balanced batch、bubble filling）**要自己照論文重寫** |

## 2. 他們用的資料與設定（要照著用）

| | Strata〔L1 §1.7，OSDI'26 全文〕 | KVFlow〔repo L3 §H4，部分〕 |
|:--|:--|:--|
| 資料 | LooGLE（平均輸入 21,613 token）、NarrativeQA（54,797；濾掉 >128K）、ReviewMT（17,708）、ShareGPT（680.9，插 60 s thinking time）；delay hit 另用 Mooncake tool-agent trace 模擬 | 合成的 10-agent 循序 workflow：固定前綴 4,096／8,192 token＋動態後綴 |
| 到達 | Poisson（資料集沒有時間戳） | batch 1；另有多 workflow 併發 |
| 模型 | Llama-3.1-8B-Instruct、Qwen2.5-14B-Instruct-1M、Llama-3.1-70B（4 卡 TP） | Llama-3.1-8B（A10G）、Qwen2.5-32B（H100） |
| 硬體 | 8×H200（PCIe 5.0）、H20＋NVMe、GH200 | A10G 24 GB、H100 80 GB |
| 對照 | vLLM 0.8.5＋LMCache 0.2.1、TRT-LLM 0.17.0、SGLang 0.4.5、SGLang-HiCache | SGLang（含 HiCache） |
| 指標 | 同 TTFT 下的吞吐、平均 TTFT | 端到端延遲（speedup） |
| 原文的主要數字 | LooGLE、Llama-8B，對 SGLang-HiCache 最多 3.2× 吞吐（p.10） | 對 SGLang 單 workflow 最多 1.83×、併發最多 2.19× |

上表的數字是文獻 agent 讀原文抄的；零前文 verifier **沒有**核對這幾個數字。登記前要再對一次原文。

## 3. 可行性檢查（下一步）

1. **KVFlow**：在 H200 的使用者空間建一個獨立的 venv（不碰 ICPE 與 stage-3 的環境；注意 /work 配額），裝 KVFlow 的 SGLang fork；看 `benchmark/` 有沒有論文的 workflow 腳本。能在 Llama-3.1-8B 上起 server、跑通一個 workflow 就算可行。
2. **Strata**：
   - I/O 部分：用上游 SGLang 的 HiCache（`kernel` vs `direct` I/O backend）在 LooGLE 上量；這部分能直接復現。
   - 排程部分：先在 F0 的模擬器裡用 P4 代表；要上真引擎時，在 SGLang 的排程器裡照論文 §4.3 實作三段規則（門檻預設 100），並在 RUNLOG 註明是我們的重寫，不是官方程式。
3. **資料**：LooGLE、NarrativeQA 下載前先確認授權與大小；Mooncake trace 已在 H200。
4. 可行性都過了，再寫 `R0` 的判準登記（例如：同設定下方向一致、量級在原文的 0.5–2 倍內），**登記之後才跑**。

## 4. 和 F0／F1 的關係

- F0（模擬）裡已經有 Strata 式（P4）與「vLLM 一律等」（P1）的規則；KVFlow 的核心（依流程預取、載入中先跳過）在單一共享前綴的情境下等於 P1＋預取，所以 F0 不另外列。
- F1（真引擎）要和真的 KVFlow、真的（重寫的）Strata 比。R0 的可行性結果決定 F1 用哪個引擎（vLLM 或 SGLang）。
