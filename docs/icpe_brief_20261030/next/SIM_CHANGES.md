# SIM_CHANGES：模擬器的改動（icpe-sim；`code/p3x/x1_sim.py`）

每一項：改了什麼、為什麼、對預測有利還是不利。「有利／不利」都是〔推論〕，沒有量過；量過的數字只在 `results/` 與 run 目錄。
**所有新功能預設關閉**；關閉時走原本的程式路徑，輸出欄位不變（G1 驗證逐值相同，run_id 記在 STATUS_SIM）。

---

## 第一批（ICPE；2026-11-04）

### 1. GPU 快取層（`--gpu-gib`、`--gpu-write`、`--gpu-policies`）

- **改了什麼**：在 CPU 層上面加一層 GPU（GPU→CPU→慢層）。預算以 GiB 給（PREREG：111.0625＝1,777 個 512-token BF16 塊）；0＝關閉。
  - 粒度＝session（和 CPU 層一樣）；共用前綴照 refcount 只存一份（`ref_gpu`）。GPU 上的 block 一定是請求的前綴（鏈式前綴＋session 粒度）。
  - **還原**：`View.restore(..., g)`：前 g 塊已在 GPU，不用還原；Cake 從第 g 塊開始重算、從最後一塊往前載入；只載入從第 g 塊開始。g＝0 時算式和原本逐位元相同。
  - **`wb` 回寫、互斥（主結果；PREREG §4 的資料流）**：回合結束整個 session 留在 GPU、不寫 CPU；它舊的那份釋放——CPU 上的 BF16 沒有別的 CPU 持有者就丟（不寫慢層，因為資料在 GPU 上；`_fix_g`），自己預取進 CPU 的 INT4 也丟，慢層那份照原本留著。GPU 超量 → 照 GPU 規則選 session 以 BF16 寫進 CPU（佔 CPU 寫入線，時間＝造成超量的那個請求的回合結束，同原本的 CPU 寫入時刻）→ CPU 超量 → 照 CPU 規則降級／丟。**CPU 層的順序只看「進入 CPU 層」的事件**（GPU 逐出＝一次存取；同 L2 只看得到 L1 逐出的階層快取）；LRU2、PENS、GDSF 用的真實請求序號不變。
  - **`wt` 直寫、包含（敏感度；使用者 11/4 決定兩種都做；PREREG 沒登記）**：CPU 層和原本完全一樣（回合結束全部 BF16 進 CPU），GPU 另外留一份；GPU 逐出不用寫。CPU 層的 hook（O-DROP、LRU+2HA）把 session 丟掉時 GPU 那份也丟；CPU 層之後因為容量降級／丟掉時 GPU 那份不動。
  - **GPU 規則**：`LRU`；`BEL`＝完美預測（GPU 超量時先擠出下次最晚回來的 session，和 O-BEL 同一個 `next_req`，用 `ref_gpu` 算獨有的第一塊）；`TTLA:<τ>`＝回合結束（＝這個時間設定的第一個 token）起釘住 τ 秒，到期移到 CPU（PREREG §2 第 3 項的語意；到期時刻寫入）；`TTLB:<τ>`＝釘住 τ 秒、到期後照 LRU、GPU 滿才逐出（使用者 11/4 要的第二種語意；PREREG 沒登記）。兩種 TTL：GPU 滿時先逐出沒在釘住期的（LRU 順序）；全部都在釘住期 → 放掉「對話開始時間最晚」的（Continuum §5.2；PREREG）。
  - **組合**：GPU 規則是新的一軸，和每一個 CPU 規則配對（使用者 11/4：每個 CPU 對手都配）。
  - **前綴快取式（vLLM APC）**：session 粒度下和 GPU-LRU 完全相同，第一批不另外跑；block 粒度（逐出時留前綴、丟尾端）在第二批（使用者 11/4）。
- **為什麼**：NEXT_PLAN §5.9：對手論文全部都有「KV 先留在 GPU」這層，原本的模擬器沒有，等於沒在它們的主戰場比。
- **有利還是不利**：`wb` 等於多一層容量 → 記憶體壓力降低 → 上限傾向變小（對預測不利）；但 GPU 多了一個「留誰」的決定，給完美預測多一個元件（O-GPU，對預測有利）。`wt` 不增加容量，GPU 逐出免費，GPU 層選錯的代價小。
- **已知的近似**：
  - 一次一個請求：還原時暫時要的 GPU 空間不算進預算，回合結束才檢查容量（同原本的 CPU 層）。
  - TTL 從第一個 token 起算（模擬器沒有解碼時間）→ 實際釘住的時間比真的系統短 → 對 TTL 對手略不利 → 上限可能略偏高（對預測有利）；掃 0.3–300 s 部分抵消。
  - TTL（及預取）規則和時間設定有關，一個 job 只跑一種時間設定。
  - TTLA 的到期逐出在下一個請求之前依時間順序處理；寫入排在到期時刻（I/O 線照 FIFO，可能排在已經送出的較晚寫入後面）。
  - 預取目的地仍是 CPU（主結果）；預取時已在 GPU 上的塊不預取。CPU 預取的會合點 `m_cpu` 沒有把 GPU 前綴算進去（近似）。
  - **Cake 的會合點是貪婪的雙指標**：前綴換成 GPU 後，剩下部分的 Cake 還原有時比「沒有 GPU 前綴」慢一點（自檢 T4：400 個隨機 layout 裡有些慢了，但都不到一塊的成本；只載入不會）。這是 Cake 原本的貪婪規則，不是 GPU 層的錯；照實保留。

### 2. 可選的慢層等級 `nvme807`（`--slows ssd b32 nfs nvme807`）

- **改了什麼**：新增慢層等級，預設不跑（`SLOWS` 不變）。讀＝`results/s3_h200/gfio/gfio_20261026-104426-s3-gfio.csv` 的 /tmp 64 MiB 循序讀 8 執行緒（8.068 GiB/s；載入時驗算它是 `gibps_reps` 的中位數）；寫＝讀 × SSD 級寫讀比（1.91065 ÷ 6.98115）＝2.2081 GiB/s；`k_read`＝SSD 級 3.306。常數從檔案讀，不寫死。PREREG §2 第 4 項（寫的是 8.07／2.209，四捨五入）。
- **為什麼**：H200 本機 NVMe 的實測點，當 SSD 級的敏感度（NEXT_PLAN §5.9）。
- **有利還是不利**：比 SSD 級快 15.6%〔算術〕→ 放錯的代價變小 → 對預測略不利。

### 3. 容量參數接受非整數（`--cpu`、`--gpu-gib`）

- **改了什麼**：整數照舊用 int（輸出逐位元不變），非整數用 float（PREREG 的參考點 111.0625 GiB＝1,777 塊）。
- **有利還是不利**：無（只是參數格式）。

### 4. 自檢 `code/icpe/next_sim_test.py`

T1 直寫＋GPU 1 byte ≡ 關閉（逐位元）；T2 每 25 步的 refcount／已用位元組／持有者稽核（`Engine.audit_gpu`）；T3 GPU 放得下全部時 wb ≡ wt、回來的請求不用載入、wb 不寫 CPU／慢層；T4 GPU 前綴的還原；T5 nvme807 常數。

---

### 5. 預取直接搬回 GPU（`--pf-target gpu`；第一批 1b）

- **改了什麼**（PREREG §2 第 2 項的敏感度；偏離 1：舊格子與 W2 全部格子）：預取規則（O-PF、LRU+RPF、O-ALL 的預取）與觸發時機不變（請求之間、I/O 線空著時），目的地改成 GPU。
  - 候選＝不在 GPU 上、資料在 CPU 或慢層的 session（`wb`：被 GPU 逐出時、被 CPU 降級時成為候選；`wt`：GPU 那份被丟掉時、被 CPU 降級時）。
  - 搬哪些：這個 session 存著、不在 GPU 上的全部塊，從最後一塊往前。CPU 上的 BF16 走 CPU 線（43.59 GiB/s）；慢層的 INT4 走慢層線＋反量化 d；放進 GPU 是 BF16（64 MiB／塊，佔 GPU 預算）。
  - 騰空間：只准逐出預取規則認為「比候選者冷」的 GPU session（O-PF：下次回來比候選者晚；RPF：上次存取比候選者早）；`wb` 逐出照常寫 CPU。
  - 回來時：預取進 GPU 的塊（layout 代碼 4）不用載入，只等預取完成；回合結束後變成常駐（同一份資料，GPU 位元組不重算）。GPU 超量時先照 GPU 規則逐出 session；GPU 上沒有 session（或 RPF）時才取消預取（同 CPU 預取的順序）。session 被丟掉時，它預取進 GPU 的塊一起丟。
  - 沒有預取的 CPU 規則不產生 `pf_target＝gpu` 的 job（結果相同）；分析時用 `pf_target＝cpu` 那份。
  - 預取迴圈先 pop 再處理（提早結束就推回）：`wb` 騰空間時 CPU 層可能降級別的 session、把新候選推進 heap。原本的 CPU 預取推進去的鍵一定比頂端大（O-PF：最後一塊的下次存取 ≥ 第一個獨有塊的下次存取 ＞ 候選者的；RPF：被降級的比候選者舊），所以沒有這個問題、沒有改。
- **為什麼**：PREREG：只預取到 CPU，oracle 回來時還要付 CPU→GPU 載入，上限偏低，和「去對手最有利的地方測」相反。
- **有利還是不利**：對預測有利（完美預取可以完全不付載入）；RPF 也一起受惠。

### 6. 有 GPU 軸的分析 `code/icpe/next_sim_analyze.py`；`x1_sim analyze` 加防呆

- `x1_sim analyze` 的樞紐表只看 `policy`（`aggfunc="first"`）：sweep 有 GPU 軸時會靜默蓋掉 → 現在直接停下、叫人用 `next_sim_analyze.py`。
- `next_sim_analyze.py`：每格 ④＝元件的 oracle 規則 × GPU 不預測規則裡最快、②＝對手 CPU 規則 × GPU 不預測規則裡最快（PREREG 偏離 2 第 5 項）；O-GPU＝GPU-BEL × CPU 不預測規則 對 GPU 不預測規則 × CPU 不預測規則；O-ALL(report)＝(O-ALL, BEL)；① 基準 (LRU, LRU) 另列。只算數字、不下判定字。
- 驗證：拿 H200 的 T1 V1 sweep（沒有 GPU 軸）跑，和 H200 的 `x1_headroom_components.csv` 1,152 列逐列相同（T_oracle、T_rival、headroom 差 0；write_saturated、oracle_policy、best_rival 全同；run `20261104-233301-icpe-sim-analyze-check`）。

---

## 下一步（未實作）

- 第二批：雜訊（PREREG 模型 P／M、B1／B2、10 seed）、慢層容量 ⌊0.5 × n_blocks_full⌋＋慢層淘汰（② O-ORD 那組、④ Belady）、排隊、W5、block 粒度的 APC（偏離 2 第 3 項）。
