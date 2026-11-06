# 第 2–5 項的語意決定（icpe-rux，2026-11-07）

> **授權**：使用者 2026-11-07 夜裡（原話）：「第 2 到 5 細節 都可以你決定 符合實驗命題且有規範 其他 paper 合理這樣 我要睡了 都交給妳」；同一晚另一句：「也就是 python 不寫完整移植 rust」。
> 所以這 4 項**只有 Rust 實作、沒有 Python 參考**，語意由 icpe-rux 決定，依據寫在每一條後面。
> **還要做的程序**：PR:57 規定「排隊模型、W3 的線索 (a)、W5…要跑之前另補登記」，PR:55／95 規定 APC 的定義要登記。PREREG 由 h200（經 icpe-data）寫，不是 icpe-rux。這份文件是**要登記的內容草稿**；正式跑之前要登記，並列出已經看過的版本 A、B 結果（PR §6）。
> 依據的版本：PR＝`origin/icpe-next-main:…/next/PREREG.md` @ df8b74c（和 icpe-next-a @ 6b84a5c 相同）；NP＝`NEXT_PLAN_20261104.md`（icpe-next-rux HEAD）；SS＝`origin/icpe-next-sim:…/STATUS_SIM.md` @ 90dae78；py＝`code/p3x/x1_sim.py`。外部論文與系統的出處見 §5。

## 0. 判斷原則

- **論文的命題**（NP:9；PR:21）：預測比最強的不預測做法沒有多大優勢；推翻條件＝真實工具類紀錄上上限 ≥15%，**而且**沒有簡單線索能拿走大部分。
- **方向**：② 比真實系統弱 → 上限偏高（對命題不利、保守）；④ 少了 ② 有的機制 → 上限偏低（對命題有利、**不可以**）。所以：
  1. 照已登記的文字（PR、NP、SS）；
  2. 對照最廣用的真實系統（vLLM、SGLang）；
  3. 不讓 ④ 吃虧：② 有的機制，④ 也要有；
  4. 夠簡單、能驗證。
- **驗證**（沒有 Python 參考時的做法，使用者同意）：
  - **退化檢查**：新功能設成「等於沒開」的參數，必須和 v7 逐位元相同。v7 已經和 Python 逐位元相同。
  - **不變量稽核**：驗證模式下，每 k 步斷言一次。
  - **手算小例子**：先算好答案，再用程式跑。
  - **兩份實作互比**：有加速時，慢的照字面寫、快的另寫，兩份逐位元比。
  - **抓不到的錯**：語意理解錯。所以本檔先寫，程式照本檔；建議另一個 session 只審本檔與程式的對應。

## 1. 項目 2：block 粒度 APC（加進 ② 的 GPU 規則）

| # | 問題 | 決定 | 依據 |
|:--|:--|:--|:--|
| 2-1 | 逐出的單位與順序 | **以 session 的最近使用（回合結束時刻）排序，從最久沒用的 session 的尾端一塊一塊逐出**，直到 GPU 用量 ≤ 預算。共用塊只有在所有 GPU 上的持有者都放掉時才真的空出來 | SS:32 字面；vLLM：釋放的塊「in the reverse order」進 free queue、從頭逐出 LRU、「the last block of a request…evicted first」。本模擬器一次只服務一個請求，所以「全域 block LRU＋同一請求尾端先出」和「session LRU＋尾端先出」選到同樣的塊〔推論：共用塊的最後使用時刻＝最近持有它的 session 的回合結束〕 |
| 2-2 | wb 下被逐出的資料 | **寫進 CPU（BF16）**（PR:54 的資料流）。做法：session 回合結束後第一次被逐出塊時，把**整個 session** 寫進 CPU（session 進 CPU 層，CPU 規則照舊、以 session 為單位）；GPU 留下的前段變成「乾淨的副本」，之後從尾端逐出、不用再寫 | 保留「session 不在 GPU 的那份只在一層」的不變量，CPU／慢層規則一行不改。代價：寫入與 CPU 占用比逐塊寫回（SGLang HiCache write_back）早一點 → ② 略弱 → 上限略偏高（保守）。逐塊寫回要讓 CPU 層存半個 session，**不做**，論文寫明 |
| 2-3 | wt | **CPU 照舊（回合結束就寫）；GPU 副本照 2-1 的順序從尾端逐出，不寫** | 和 vLLM＋LMCache、SGLang HiCache 的預設（write_through）相同；P3 的敏感度 |
| 2-4 | 塊大小 | **512 token** | NP:269（常數都是 512 量的）；PR:28 的 GPU 預算 1,777 塊 |
| 2-5 | ④ 要不要也有 | **要**。④ 的 GPU 不預測規則集合和 ② 相同（含 APC）。另加 **BELP**：同樣「從尾端一塊一塊逐出」，但選 session 照 Belady（下次回來最晚的先逐出），讓 ④ 的 GPU oracle 也有 block 粒度 | PR:177–182「兩邊規則集合相同」；不加 BELP 時 ④ 少了 ② 有的機制（上限偏低，不可以） |
| 2-6 | 要重算哪些格子 | 建議登記：主結果（APC 不依時間，P2 的時間設定一起出）＋P3 的 wt；P4、慢層上限另外登記 | PR:95「受影響的格子全部重算」、PR:126「格子另外定」。**跑什麼不是 icpe-rux 決定**，這裡只是建議 |
| 2-7 | TTL 要不要也改 block 粒度 | **不改**，維持 session 粒度 | PR:30 的 TTL 以「這個對話的 KV」為單位 |
| 2-8 | 和其他功能 | 預取進 GPU（pf-target gpu）× APC：**不支援**（程式斷言），P3 的預取進 GPU 格子不放 APC；慢層上限、W4 雜訊、W5：照常 | 預取進 GPU 的路徑假設 GPU 上是整個 session；BELP 有雜訊時用有雜訊的「下次回來」（同 GPU BEL） |

## 2. 項目 3：排隊

| # | 問題 | 決定 | 依據 |
|:--|:--|:--|:--|
| 3-1 | 模型 | **單一 GPU、先到先服務（FCFS）、服務時間＝還原＋prefill（＝現在的 TTFT）**。請求 r 的開始時刻＝max(到達, 前一個請求完成)；回報的 TTFT＝等待＋服務 | vLLM／SGLang 預設 `fcfs`；Mooncake 的 prefill 佇列；repo 的 B6 GPU 重播（`code/icpe/x1_gpu.py`）同一個模型 |
| 3-2 | 解碼 | **不算**（＝prefill／decode 分離的 prefill 機） | NP:276「完成時間要模擬解碼，不做」 |
| 3-3 | 狀態什麼時候改 | **還原在開始時刻讀快取狀態；回合結束（寫入、TTL 起算）在完成時刻**。FCFS 下服務順序＝到達順序，所以 oracle 的「下次回來」照請求序號，不用改 | 和 B6 相同；oracle 預取用到達時刻（④ 本來就知道） |
| 3-4 | 範圍 | **只跑固定並行的紀錄**（這台：SemiAnalysis；acvlab：CacheWise），另外標「排隊（FCFS-1）」的敏感度；同時報 ρ（服務時間總和 ÷ 紀錄時長） | NP:266；原始到達的紀錄來自多 GPU 叢集，單一 GPU 會不穩定（Mooncake toolagent ρ≈2〔推算〕） |
| 3-5 | 排隊設定數 q | **q＝1**（FCFS-1） | 並行數 N 已經在掃負載 |
| 3-6 | 一個 job 幾個時間設定 | **1 個**（每個時間設定的服務時間不同 → 開始時刻不同） | ANALYSIS §5 |
| — | 排隊時的預取 | **不加新機制**：等待中的請求不會因為在佇列裡就被預取 | 加了會讓 ② 更強；不加 → ② 略弱 → 上限偏高（保守）。論文寫明 |

## 3. 項目 4：W3 的線索 (a)（③ 的簡單線索）

| # | 問題 | 決定 | 依據 |
|:--|:--|:--|:--|
| 4-1 | 線索做什麼 | **兩個 GPU 規則都做**：**CUEP**＝這一輪的輸出有工具呼叫（下一個請求由工具觸發）→ 釘在 GPU 直到這個 session 的下一個請求；其他輪照 GPU LRU。**CUEO**＝有工具呼叫 → 回合結束立刻逐出 GPU（wb：寫進 CPU；wt：放掉 GPU 副本）；其他輪照 GPU LRU。全部釘住而 GPU 滿時，放掉「對話開始時間最晚」的那個（PR:30 的規則） | NP:42「留在／搬回」、NP:320「留在 GPU／搬回 CPU」；InferCept 的 Preserve／Swap；Continuum 對工具呼叫的請求釘 GPU、TTL 的極端版（NP:42） |
| 4-2 | 線索怎麼來 | 每個請求一個旗標 cue[r]＝**同一個 session 的下一個請求是工具觸發的**（真實系統在回合結束時看得到輸出裡的工具呼叫）。session 的最後一輪＝0。只用有欄位的紀錄：TraceLab、CacheWise（`trigger`）、FreeInference（`step_trigger`）。**Mooncake、SemiAnalysis、Bailian 沒有欄位 → NOT_MEASURED** | 現在的 nb 檔只存 `user＝(trigger=="user")`，「none」也會變 False；要 icpe-data 在 nb 檔多存一欄「工具觸發」（`next_b_convert.py`），引擎只吃 0／1 陣列 |
| 4-3 | 指標 | 主要：f＝(T② − T③) ÷ (T② − T④c)，每個上限 ≥15% 的元件 c 各算；次要（和 C2 一樣）：NP:43 的 (T① − T③) ÷ (T① − T④O-PF)。「大部分」的門檻：**建議登記 f > 0.5** | PR:21 的字面是「拿走大部分」 |
| 4-4 | 配哪些 CPU 規則 | **NON_ORACLE（10 個）**；③＝每格在 {CUEP, CUEO} × CPU 規則裡取最快 | 只配 LRU 可能比 ② 還慢（f < 0）；和 ② 的「事後挑最快」同一種 |
| 4-5 | 範圍 | 有欄位的工具類變體全部（TraceLab c1／c2／c4、CacheWise、FreeInference），f 只在上限 ≥15% 的格子報 | TraceLab c1 是唯一「原始到達、上限 ≥15%、有欄位」的變體 |
| 4-6 | 列進 ICPE？ | 建議列（便宜，而且是推翻條件的另一半） | PR:245 |

## 4. 項目 5：W5（CPU 層 INT4＋一次拆完）

| # | 問題 | 決定 | 依據 |
|:--|:--|:--|:--|
| 5-1 | 常數怎麼放進模型 | **CPU 層的每個塊都是 INT4**：占 i4 位元組、讀 i4 ÷ r_cpu，再加每塊的「一次拆完殘差」d1＝(T_meas − 64·i4 ÷ r_cpu) ÷ 64。T_meas＝M3-g（P0-1b S4，fa3）32K、CPU、INT4 全部載入的還原中位數：閒 0.02914 s、忙 0.03327 s（`results/icpe_h200/p01/grid/llama_s4_fa3/summary.csv`，run `20261022-232344-icpe-p01b-llama-s4-fa3`） | 用現有的「每塊讀＋殘差」路徑，32K 的總時間和實測**完全相同**（照構造）。長度依賴 NOT_MEASURED（只量了 32K） |
| 5-2 | 格式 | **所有規則、② 和 ④ 都一樣**：CPU 層只存 INT4；CPU → 慢層＝直接複製（不再量化）；預取進 CPU 的塊也用 d1 | NP:52「CPU 層允許 INT4」；兩邊同樣條件 |
| 5-3 | 量化時間 | **不計**，和現有的慢層 INT4 寫入一樣（模型一致） | 逐塊編碼實測 2.60 ms／塊（run `20261017-094511-m10h-grid-int4-llama` 的 t_encode ÷ fp8_read 中位數），發生在逐出時、不在還原的關鍵路徑；計入寫入道只影響讀寫爭用（CPU k_read＝1.002）〔推論：影響可忽略〕 |
| 5-4 | 還原方式 | **照舊 Cake**（PR:48）；只載入維持 P2 敏感度 | Cake 已經幾乎全部載入 CPU 的塊 |
| 5-5 | 容量與範圍 | 建議登記：{111.0625, 128, 256} GiB × PR:118 的 7 個代表變體；**敏感度**，不是主結果 | NP:50–53；PR:118 |
| — | 預期 | NP:53「上限下降或不變」 | — |

## 5. 外部出處（研究由 subagent 整理，icpe-rux 抽查了 PR、NP、SS 的引文行號）

- vLLM automatic prefix caching 設計文件（docs.vllm.ai design/prefix_caching，11/7 讀）、`v1/core/kv_cache_manager.py`。
- SGLang RadixAttention（arXiv 2312.07104v2：「evicts the least recently used leaf first」）；HiCache 設計文件（write_through 預設、write_back＝逐出時才寫）。
- InferCept（arXiv 2402.01869v2：Discard／Preserve／Swap）；Continuum（arXiv 2511.02230v7：工具呼叫的請求釘 GPU、帶 TTL）。
- Mooncake（arXiv 2407.00079v5：prefill 機的排隊時間＝佇列裡請求 prefill 時間的總和）；CachedAttention（arXiv 2403.19708v3）。
- NOT_CHECKED：vLLM OffloadingConnector 的寫入語意、DistServe。

## 6. 實作狀態

見 `code/icpe/rux_x1fast/v8_features/README.md`（v8＝v7＋本檔的功能）與 STATUS_RUX。
