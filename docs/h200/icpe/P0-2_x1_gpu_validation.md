# P0-2　3b「猜誰會回來」的 GPU 驗證：X1 的模擬在 GPU 上站得住嗎？

> ICPE 補量 P0-2（[README](README.md)）。平台 H200（hgx-cn01）。**§1–§7 是事先登記**，寫在任何 P0-2 的資料（包括縮小比例的模擬）之前；結果寫在 §8 以後，登記那幾節不改。
> **登記寫於 2026-10-22T03:01:41Z（`date -u`），repo HEAD `cf31400`**；寫的時候沒有任何這個實驗的資料，程式也還沒寫。
> 標記：〔實測〕〔模擬〕〔算術〕〔判讀〕同 repo 慣例。

---

## 1. 問題

3b 的 X1（`docs/phase3_20261017/explore/X1_oracle_decomposition.md`，H200 常數重跑＝`results/p3_explore_h200/x1/`）全是模擬。論文要寫「預知回來時間只在窄區域有用、預取用打字提示就夠」，那個窄區域就是 X1 headroom 最大的設定：

> Mooncake conversation、慢層＝本地 SSD 限速、CPU 128／256 GiB、GPU 閒、Cake 還原（H200 模擬：回來請求平均 TTFT，LRU 0.466／0.425 s、O-PF 0.222／0.224 s、O-BEL 0.297／0.268 s；`x1_sweep.csv.gz`，run `20261021-010140-p3h-x1-sweep`，規則 R1）。

**要證明的**：在這個設定挑 2 格（CPU 128、256 GiB，等比例縮小）真的在 GPU 上還原，(1) 回來請求的 TTFT 實測÷模擬在 0.9–1.1，(2) 各做法的排序、(3) headroom 的正負號和模擬相同。

---

## 2. 要比的做法（5 個，事先列死）

| 代號 | 是什麼 | 來源 |
|:--|:--|:--|
| LRU | 降級最久沒來的 session | X1 原有 |
| FREQ2Q | 「2Q 式」：先比到目前的請求數、再比閒置時間（請求數優先） | X1 事後加的對手（原有程式） |
| O-BEL | oracle 降級順序（Belady：下次被讀最晚的先降） | X1 的 O-ORD 族；事先固定用 O-BEL（不在兩個 oracle 裡挑） |
| **HINT-PF** | LRU 順序＋**打字提示預取**（新寫，定義在 §3） | 新 |
| O-PF | LRU 順序＋oracle 預取（Aggressive：讀取線空著就預取下次被讀最早的慢層 session） | X1 原有 |

## 3. HINT-PF 的定義（寫在任何 HINT-PF 的模擬之前）

- **提示**：回來請求 r（X1 R1 重建規則的接續請求）在到達前 L_a(r) 秒，系統知道「session s(r) 會回來」。
  L_a 照 X3（`code/p3x/x3_hide.py::leads` 的 (a) 主設定）：
  - 打字時間＝max(新 token 數 − 10, 0) × 4.2693 字元/token ÷（51.56 WPM × 5 字元 ÷ 60 s）；新 token 數＝input(r) − input(p) − output(p)，p＝r 在 R1 重建裡的父請求；4.2693＝H200 的 `x3_typing_consts.json` 的 `cpt_llama_English`；
  - 思考時間＝max(到達間隔 − p 的 prefill 估計 − output(p) × 0.05 s, 0)；p 的 prefill 估計＝p 自己要新算的 token 數 ÷ 512 × f(p 的舊 context block 數)（X3 的寫法；f 用 X1 的 f）；
  - L_a＝min(打字時間, 思考時間)。這是不用預測的訊號（使用者開始打字），不看任何未來。
- **預取機制**＝和 O-PF 完全相同（讀取線空著時，從最後一個 block 往前讀到「全部在 CPU 時的會合點」為止，INT4 放 CPU，和需求讀取共用讀取線，讀完前不能用，預取進來的 block 釘到該 session 回來）。不同的只有：
  - **候選**：只有「已經收到提示、還沒回來、目前在慢層」的 session；先收到提示的先預取。收到提示時已經在 CPU 的，不做事；之後被降級了，降級當下重新排進候選。
  - **騰空間**：只准降級「沒有在等的提示」的 CPU session，最久沒來的先（LRU）。
  - **最後手段**（CPU 只剩預取的 block）：丟掉提示時間最晚的那個 session 的預取 block。
- 不變量：所有 L_a 設成 0 時，HINT-PF 和 LRU 的每個請求 TTFT 逐位元相同（提示在到達當下才來，來不及預取）。

## 4. 等比例縮小（為了在單卡上跑得完）

- **怎麼縮**：依 R1 重建的 session 抽樣，只留 `sha1(str(session_id)) mod k == 0` 的 session 的所有請求；CPU 容量同時除以 k（128／k、256／k GiB，取 64 MiB 的整數倍往下取）。時間戳不動，所以「CPU 裝得下幾分鐘的歷史」與慢層寫入負載 ρ_w 大致不變〔判讀〕；跨 session 的前綴共用會變少（只留同一批 session 之間的）。
- **k 怎麼選**（只用模擬、在任何 GPU 重播之前；候選 k ∈ {4, 8, 16}）：取最小的 k 同時滿足
  1. 估計的 GPU 使用率 ρ̂＝縮小後 LRU（128／k）所有請求的模擬 TTFT 總和 ÷ trace 長度 ≤ 0.30；
  2. 縮小後的模擬裡，O-PF 對 LRU 的 headroom 在 128／k 或 256／k 至少一格 ≥ 15%（還在「X1 headroom 最大」的區域）。
  都不滿足 → 用 k＝16，照實寫。選定的 k、ρ̂、縮小後的請求數／回來請求數，寫進結果第一節。

## 5. GPU 重播的做法（`code/icpe/x1_gpu.py`，新寫）

- **決策**：直接用 X1 的 `Engine`（同一份程式；HINT-PF 用子類別），和模擬一模一樣的狀態機、refcount、降級順序、預取規則。
- **時間**：真的跑的只有 GPU 上的東西——Cake 的重算線（harness `forward_span`，SDPA，和 X1 的 f(i) 同一個實作）、真實的 pinned H2D、INT4 的反量化（`m10_codec`，載入線上一塊一塊拆＝X1 的模型）、新一輪的 prefill、回合結束新 block 的 D2H。
  I/O 線（CPU 層 35.4 GiB/s、本地 SSD 讀 6.98／寫 1.91 GiB/s、寫入中讀取 ×3.306）用和模擬同一套保留公式（X1 的 `Chan`＝`m7_restore_harness.Tier` 的 share 模式；真實時鐘的 0.5 ms 背靠背規則照 Tier），由真實的還原程式（`Restorer`）去等它。降級寫入、預取讀取只佔 I/O 線的時間（和模擬相同：資料本來就在主機記憶體，量化不算時間）。
- **虛擬時鐘、跳過閒置**：GPU 閒著、下一個請求還沒到時，時鐘直接跳到下一個到達時間（這段時間裡模擬與實際都只有 I/O 線的保留在走，跳過不改變任何事）。
- **一次一個請求**：請求在 max(到達時間, GPU 空出來) 開始；TTFT＝從開始到第一個 token（和模擬的「GPU 不排隊」同義），排隊時間另記、不算進 TTFT。
- **KV 內容**：只量時間。主機上放 64 塊真的 KV（BF16）與它們的 INT4 編碼，第 b 個 block 用第 b mod 64 份；重算用固定的隨機 token。內容對不上不影響時間（attention 的成本和數值無關）。不做 KV 逐位元檢查。
- 模型 Llama-3.1-8B，GPU 閒，`gpu_guard` 全程監看。

## 6. 跑哪些、怎麼比

- **格子**：2 個容量（128／k、256／k GiB）× 5 個做法＝10 個 run；另外 LRU 在兩個容量各重跑一次（看 run 之間的雜訊）＝共 12 個 run。順序用固定 seed 打亂（seed 20261023）。
- **模擬參考**：
  - **主參考 S_orig**＝X1 模擬器在同一份縮小 trace、同一容量、原始時間戳上的結果（就是「模擬會怎麼說」）。
  - **診斷 S_real**＝同一個模擬器，但到達時間換成該 GPU run 實際的開始時間（把排隊造成的時間位移和每個請求的模型誤差分開）。
- **判準**（都對 S_orig；三條都成立＝「模擬可信」）：
  1. **比值**：10 格（容量 × 做法）每格的「回來請求 TTFT 平均，實測 ÷ S_orig」都在 [0.9, 1.1]。
  2. **排序**：每個容量內，S_orig 平均差 ≥5% 的每一對做法，GPU 上的先後相同（<5% 的對列出、不判）。
  3. **headroom 正負號**：每個容量，下面四個比較在 GPU 和 S_orig 上同號（S_orig 的 |headroom| < 2% 的列出、不判）：O-BEL 對 min(LRU, FREQ2Q)；O-PF 對 LRU；HINT-PF 對 LRU；O-PF 對 HINT-PF。
- **只報告**：每個請求的實測÷模擬分布（中位數、p10、p90）；對 S_real 的同一組比值；排隊次數與時間；D2H 時間；兩次 LRU 的差；縮小 trace 與全尺寸模擬的 headroom 對照；HINT-PF 在全尺寸 X1 格子（conversation、R1、Cake、三個慢層 × GPU 閒／忙 × 4 個容量）的模擬：拿到 O-PF 對 LRU 改善的多少（新數字，只報告）。

## 7. 停損、失敗處理、時間

- 先過 CPU 自測才上 GPU：(i) 重播程式用「模型時間」代替 GPU 執行時，5 個做法每個請求的 TTFT 和 X1 模擬器逐位元相同；(ii) §3 的 L_a＝0 不變量；(iii) X1 原本的 selftest 仍全過（HINT-PF 只在新檔，不改 `x1_sim.py`）。任一條不過就停。
- GPU 冒煙：LRU、縮小 trace 的前 100 個請求，跑完、記錄都對才跑正式的 12 個 run。
- 任何指令失敗：完整錯誤記下、停下來回報。gpu_guard 污染：該 run 重跑一次，第二次還污染就停。
- 量時間：不和我們自己的任何 job 同時在 hgx-cn01 跑。
- 時間〔估計〕：每個 run 約等於縮小 trace 的 GPU 總工作量（k＝8 時約 16 分）；12 個 run 約 3–4 小時。

---

## 8. 結果

GPU 重播（job 21655）還沒跑（hold 中，排在 p01brr 之後）。下面是已經有的 CPU 部分。

### 8.1 只報告：HINT-PF 在全尺寸 X1 格子〔模擬，H200 常數〕
run `20261025-202314-icpe-p02-fullsim`（job 21656）→ `results/icpe_h200/p02/fullsim_hint.csv`；conversation、R1、Cake，三個慢層 × GPU 閒／忙 × CPU 128／256／512／1,024 GiB＝24 格。
- HINT-PF 對 LRU 的回來請求平均 TTFT 改善 5.6–47.8%；O-PF 對 LRU 5.5–52.4%。
- HINT-PF 拿到 O-PF 改善的比例（hint_vs_lru ÷ opf_vs_lru）：本地 SSD 0.912–0.982（中位 0.939）、b32 0.916–1.009（0.949）、NFS 級 0.816–1.454（1.117）。
- **6 格 HINT-PF 比 O-PF 還好**（比例 > 1）：NFS 級 128／256 GiB 的閒、忙四格（1.25–1.45），b32 512／1,024 GiB 忙兩格（1.003、1.009）。〔推論，沒有另外驗證〕O-PF 不是所有預取做法的上界：兩者預取機制相同，但 O-PF 騰空間仍照 LRU、可能把快回來的 session 降級；HINT-PF 不降級有提示在等的 session（§3）。慢層越慢，這個差越大。論文若寫「HINT-PF 拿到 O-PF 的大部分」，要附這 6 格。

## 9. 偏離登記

（還沒有。）
