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

### 7. 新紀錄的載入點 `nb:<name>`（`code/p3x/x1_nb.py`）

- **改了什麼**：`x1_sim.prep("nb:<name>")` 讀 `$X1_NB_DIR/<name>.npz`（預設 `/ssd6/hungwei/icpe-next/datasets/x1_nb`）；格式與檢查寫在 `x1_nb.py` 開頭（數值陣列＋JSON 字串，`allow_pickle=False`）。`icpe-data` 用 `x1_nb.write_nb(...)` 寫，寫之前和讀之後跑同一套檢查（鏈式前綴、parent 和 session 一致、t 非遞減、nall ≥ 完整 chunk 數、「session 的第一個請求 ⇔ kind≠cont」）。Mooncake 與 `fi:` 的路徑沒動（只多一個 `elif`）。
- **為什麼**：§3b：`icpe-data` 不能改 `code/p3x/*`，但新紀錄要進模擬器。
- **驗證**：Mooncake conversation／toolagent 經 `x1_trace.prepare` → `write_nb` → `prep("nb:…")`，LRU、O-BEL、O-PF、GPU 層 111 GiB＋BEL：逐請求 TTFT、還原時間、工作集大小都相同（run `20261104-233613-icpe-sim-nb-roundtrip`）。
- **有利還是不利**：無（只是載入方式）。

---

## 第二批

### 8. W4 雜訊（`--noise-models P M --noise-lambdas ... --noise-draws B1 B2 --noise-seeds 0..9`；PREREG §2 第 6 項、§3 B）

- **改了什麼**：完美預知的每一種用途都經過 `Engine._nz(s, idx)`：O-BEL／O-BELS 的順序（`select`）、O-DROP／O-NOW 的生死（`dead`）、O-PF 的預取順序與騰空間（`demote`／`make_room`）、GPU 層的 BEL（`_g_next`）、GPU 預取（`_pf_push`）。（慢層的 Belady 在第 9 項做好後也走這裡。）用 instance 覆蓋 `next_req`（同 `_fix` 的做法），沒有雜訊時原本的程式路徑不變；`demote` 只多一行有守衛的。
  - 單位：模擬器的「下次回來」是請求 index（`next_access`）。加雜訊時先換成時間、加雜訊、再換回 index（第一個到達時間 ≥ 它的請求、至少是下一個請求）。
  - **模型 P（照 PBKV 的混合）**：以機率 λ 把這個 session 的下次回來換成「上次回合結束時間＋經驗分佈裡抽的間隔」；經驗分佈＝這份紀錄每個請求到同一個 session 下一次請求的時間，之後不再回來＝INF（照它的經驗比例）。會猜錯生死。λ＝0、0.1、0.2、0.3、0.4。
  - **模型 M（乘法誤差）**：剩餘時間（從詢問的那一刻算）×(1＋ε)、ε～U[−λ, λ]；不再回來維持不再回來。λ＝0、0.1、0.2、0.3。
  - **B1**：每個回合結束抽一次（u、間隔、ε），到這個 session 下一次請求前不變；**B2**：每次問都重抽。`random.Random(seed)`，同一個 seed 逐位元可重現。
  - **λ＝0** 直接回傳真值 → 必須和沒有雜訊逐值相同（自檢 T8；分析時再驗一次）。λ＝0 只跑一次（每個模型）。
  - 只對「用到完美預知」的組合產生 job（`uses_oracle`：CPU 規則是 O-BEL／O-BELS／O-NOW／O-DROP／O-PF／O-ALL（含 -H 版），或 GPU 規則是 BEL）。
  - 開了雜訊的 sweep 每列多 `noise_model`、`noise_lambda`、`noise_draw`、`noise_seed`（沒有雜訊的列：none、0、none、−1）。
- **分析**：`next_sim_analyze.py` 每個雜訊設定各算一次（④ 用那個設定的列、② 用沒有雜訊的列；④ 那一邊沒有全部加雜訊的元件不算），輸出逐 seed 與 seed 間的中位數、最小–最大、平均±標準差。
- **有利還是不利**：雜訊只會讓 ④ 變慢或不變〔推論；不保證，因為完美預知的貪婪規則本身不是最優〕→ 上限變小。

### 9. 慢層容量上限＋慢層淘汰規則（`--slow-cap-frac 0.5 --slow-policies LRU LRU2 ARC S3FIFO GDSF PENS BEL`；PREREG §2 第 5 項）

- **改了什麼**：上限＝⌊frac × n_blocks_full⌋ 個 INT4 塊；慢層上**所有** INT4 塊都算（使用者 11/4）。超過時：① 先丟**乾淨的副本**（持有者都在 CPU／GPU、沒有慢層狀態的持有者；照變乾淨的先後；之後那個 session 再降級要重寫）；② 不夠再照慢層規則選一個**慢層狀態的 session** 整個丟掉（回來全部重算）。用 instance 覆蓋 `_fix`（計數、記下變乾淨的塊）、`demote`（進入慢層）、`enforce`（之後檢查上限），再加兩行有守衛的（回來時、被丟時離開慢層）；沒有上限時原本的程式路徑不變。
  - 慢層規則（`SlowOrder`）：「進入慢層」＝一次存取、session 回來＝一次命中（ARC、S3FIFO 的頻率訊號）；LRU＝進入慢層的先後；LRU2、PENS、GDSF 用真實的請求紀錄與成本（GDSF 的成本＝每塊「留在慢層 vs 丟掉重算」省下的時間，和 CPU 的 GDSF 一樣一個 job 一組 (慢層, GPU 狀態)）；BEL＝下次最晚回來的先丟（④；走同一個 `next_req`，所以雜訊也套得上）。
  - 組合：慢層規則和 GPU、CPU 規則全部配對（使用者 11/4：×7）。
  - 開了上限的 sweep 每列多 `slow_cap_frac`、`slow_policy`、`slow_cap_blocks`、`n_slow_clean_evict`、`n_slow_sess_drop`。
- **為什麼**：NEXT_PLAN §5.9「慢層容量」：容量有限時多了「慢層也要挑誰丟」的決定，對預測比較有利。
- **有利還是不利**：對預測有利（多一個可以預知的決定）；同時整體變慢（丟掉的要重算）。

### 10. `x1_nb` 向量化

- 檢查、重新編號、每塊的存取清單全部改用 numpy（TraceLab 有約 2.2 億個 chunk 存取，純 Python 要好幾分鐘、好幾 GB）。和原本的純 Python 版、以及 `x1_trace.prepare` 對拍：Mooncake 兩份的 full／acc 只差一對一的重新編號、n_blocks_full 與 nall 相同；故意弄壞鏈（在和 parent 共用的位置交換兩個鍵）與 parent 都會被擋下（run `20261104-235644-icpe-sim-nb-roundtrip3`）。

### 11. 加速（結果逐位元相同；改到原本的程式路徑，所以 G1 重做）

- **為什麼**：PREREG 網格很大（每份紀錄 48,258 個 job；icpe-data 量到 CacheWise N16 每個 job 7.6–13.7 s）。profile（run `20261104-235828-icpe-sim-profile-cw`）：12 種時間設定的 job，約 83% 的時間在 `View.restore` 的逐塊迴圈與 1 億次 `Chan.read` 呼叫。
- **改了什麼**：
  1. `View.restore`：`Chan.read` 內聯成區域變數（同樣的浮點運算、同樣的順序；還原期間沒有寫入，所以 `t_free_w` 固定；每塊的 BF／r、(BF／r)×k 先算好＝同一個除法的同一個值），最後把 I/O 線狀態寫回去。舊的保留為 `restore_ref`（只給對拍用）。
  2. 回合結束釋放舊 block 的迴圈（原本路徑、wb、wt）：`_fix` 只有在「沒人持有」或「CPU 沒有持有者卻還有 BF16」時才做事，其他情況直接跳過呼叫（沒有慢層容量上限時；慢層上限的 `_fix` 包了有副作用的一層，就照舊每塊都叫）。
  3. wb 的回合結束：GPU 位元組用區域整數累加（整數，順序不影響）；舊 block 的遞減、GPU 釋放、`_fix` 合成一個迴圈（`_fix(b)` 只讀寫 b 自己的狀態與整數計數，清單裡每塊只出現一次，所以和「先全部遞減再逐一 `_fix`」相同）；「釋放自己預取進 CPU 的 INT4」那個逐塊迴圈只在有預取的規則才跑（沒有預取就不會有 fmt＝2）。
- **驗證**：
  - `restore` 對 `restore_ref`：2,000 個隨機 layout（代碼 1–4、隨機的寫入干擾狀態、cake／load）回傳值與還原後的 I/O 線狀態逐位元相同；整個引擎 12 組（Mooncake 2 份 × LRU、O-BEL、O-PF、GDSF、GPU 層、預取到 GPU）逐請求逐位元相同（run `20261105-000344-icpe-sim-speedup-check`）。
  - 跨版本：舊 checkout（ac0c56e）對新程式，Mooncake 2 份 × 24 個 CPU 規則（含 PRED）× 5 種 GPU 設定（關閉；wb 16 GiB 的 LRU、BEL、TTLA:10；wt LRU）、CPU 128 GiB，逐請求陣列與 I/O、引擎計數的 sha256：第 1、2 項 240／240 相同（run `20261105-000853-icpe-sim-fastpath-xcheck`）；加上第 3 項後新的一邊重跑：240／240 相同（run `20261105-001801-icpe-sim-fastpath-xcheck2`）。
  - 最後由 G1（Mooncake X1 sweep V1 對 H200 的輸出）再驗一次。
- **快多少**（CacheWise N16 試跑檔、LRU、CPU 128 GiB、同一台、load 約 65–80；只有第 1 項時）：12 種時間設定 32.2 → 18.9 s；1 種 5.6 → 4.8 s。第 2 項的效果還沒量（NOT_MEASURED）。
- **有利還是不利**：無（結果相同）。

### 12. 加速：回合結束跳過「互相抵銷的前綴」（逐位元相同）

- **改了什麼**：session 回來時如果它還在 CPU（原本路徑、wt）或還在 GPU（wb），而且舊清單是新清單的前綴，前綴那段的 refcount +1／−1 互相抵銷（CPU 狀態的持有者的塊一定是 BF16，fmt、cpu_used、寫入量都不變；GPU 的 ref ≥ 1，GPU 位元組不變；`_fix` 不會做事）→ 只處理新加的尾巴、不跑釋放舊清單的迴圈。只在沒有慢層容量上限、沒有預取進 GPU 時用（那兩個有副作用）。
- **驗證**：跨版本對 ac0c56e，Mooncake 2 份 × 24 個 CPU 規則 × 5 種 GPU 設定的 job 指紋 240／240 相同（run `20261105-002409-icpe-sim-prefixskip-xcheck`）。
- **快多少**：NOT_MEASURED。

### 13. `x1_nb` 的記憶體（TraceLab）

- **問題**（icpe-data 量到）：09fd521 的向量化版本讀 TraceLab 試跑檔時最大 RSS 29.3 GB（6b17191 的純 Python 版 6.8 GB）。原因：`.tolist()` 每一格都造一個新的 Python int（約 36 bytes × 2.2 億 × 2 個清單）；純 Python 版用 `setdefault` 讓同一個 id 共用一個 int 物件（每格只有 8 bytes 指標）。
- **改了什麼**：用 numpy 的 object 陣列（每個 id、每個請求 index 一個共用的 int 物件）取值再 `.tolist()`；中間的大陣列用完就 `del`。值完全相同。
- **驗證**（run `20261105-003415-icpe-sim-nb-mem`）：Mooncake 來回轉換與負面測試照舊通過；TraceLab 試跑檔 `prepare_nb` 的結果（full、acc、nall、B、n_blocks_full）和 6b17191 的純 Python 版**完全相同**；最大 RSS 10.8 GB、119 s（純 Python 版 151 s）。

### 14. 分析程式：`--rival-only-gpu`（`code/icpe/next_sim_analyze.py`；PREREG 偏離 4 第 2 項的補跑）

- **為什麼**（icpe-data 在 STATUS_DATA 2026-11-05T09:00 提出）：偏離 4 的 ≥10% 補跑只把其餘 TTL（1／3／30／100 s 的 TTLA、TTLB）× 10 個不預測的 CPU 規則加進 ②。原本的分析程式讓 4 個 CPU 元件（O-ORD／O-DROP／O-NOW／O-PF）的 ④ 與 ② 用**同一組**「格子裡出現過的 GPU 不預測規則」，所以補跑過的格子 ④ 缺列 → 這 4 個元件整列不算（CacheWise：3,888 → 2,896 列）。
- **改了什麼**：新選項 `--rival-only-gpu <GPU 規則…>`：這些 GPU 規則只放進 ②，④ 的 GPU 規則集合拿掉它們（＝補跑前的那一組）。影響 O-ORD／O-DROP／O-NOW／O-PF 與 O-SLOW 的 ④；O-GPU（④ 只用 BEL）、O-ALL(report)（④ 是固定組合）本來就不受影響。violations 多一欄 `rival_gpu_rival_only`（那個 ② 組合的 GPU 規則是不是只在 ② 的）；輸出的 meta 多一欄 `rival_only_gpu`。**不給選項時和之前逐格相同**（這兩欄都不出現）。
- **防呆**：規則名必須在 sweep 裡出現過（打錯字就停）；不能含 BEL／none；開選項時自動再算一次「拿掉這些規則的列」，(格, 元件) 集合必須相同、每一列的上限都不能變大，不過就停。
- **有利還是不利**：對預測**不利**（② 只多不少、④ 不變 → 上限只會變小或不變）。這正是 PREREG 偏離 4「② 只會隨 TTL 值變多而變快」的設定；icpe-data 已請 h200 在 PREREG 補一句「補跑 ② 時，④ 的 GPU 規則集合維持原本 3 個 TTL」。

### 15. x1 sweep 的 worker 定期換新：`--maxtasks`（預設 4；`code/p3x/x1_sim.py`，commit 54fa2d8）

- **為什麼**：icpe-trx（STATUS_TRX 10:05，TRX40 SemiAnalysis 25–44 MiB／job）與 icpe-data（STATUS_DATA 10:10，acvlab TraceLab 約 0.1 GiB／job，c1 停在 827／6,174）都量到：worker 的記憶體每跑一個 job 就多一些、不會降。使用者選「請 icpe-sim 加 maxtasksperchild」（icpe-trx 轉達）。
- **原因**（診斷 run `20261105-100705-icpe-sim-memdiag-sa`，SA N16 s1、CPU 128、ssd、cake、GPU 111.0625 GiB wb LRU／BEL × 6 個 CPU 規則＝20 個 job；只量記憶體）：
  1. **開 GPU 層時 Engine 有循環參照**：`Engine.__init__` 裡 `self._fix = self._fix_g`（綁定方法存回自己身上；第一批 GPU 層是我加的）。`cycle_check.py`（Mooncake conversation R1）：GPU 關閉時，丟掉名字後 Engine 立刻被釋放；GPU 開（LRU、BEL、TTLA:10）時 Engine 還活著，要 `gc.collect()`（收 44 個物件）才釋放。雜訊（`self.next_req = self._next_req_nz`）、慢層上限的包裝層也是同一種寫法。
  2. **fork 出來的 worker 很少做完整 gc**：父行程先載入的紀錄有大量會被 gc 追蹤的物件，算進「長期存活」的總數，所以 gen2 的觸發條件很難達到。20 個 job 裡三種跑法的 gen2 次數都是 0。
  3. 釋放了的記憶體也不一定還給系統（碎片）。pymalloc 區塊數不是單調上升（plain：有些 job 降了最多約 60 萬塊；有預取的 O-PF／O-ALL／LRU+RPF 那幾個 job 每個多約 28–53 萬塊），但 Private_Dirty 從不下降。
  4. 結果（`results/icpe_next/sim/memdiag_sa.csv`；`verify_csv.py` 對過 jsonl，逐值相同）：
     - plain（原本、1 worker）：Private_Dirty 94 → 492 MiB；
     - freeze（fork 前 `gc.freeze()`）：94 → 478 MiB，20 個 job 內沒有用；
     - mt4（`maxtasksperchild=4`）：最大 226 MiB，每個新 worker 從 94–121 MiB 重新開始。
  5. GPU 關閉的 Mooncake（G0／G1）幾乎不漲，和第 1 點一致。
- **改了什麼**：`x1_sim.py sweep` 加 `--maxtasks k`（預設 4；環境變數 `$X1_MAXTASKS` 可改預設；`0`＝不換，就是原本的呼叫 `Pool(maxtasksperchild=None)`）。`mp.get_context("fork").Pool(workers, maxtasksperchild=k)`：一個 worker 跑完 k 個 job 就結束，Pool 再從父行程 fork 一個新的。`imap_unordered` 的 chunksize＝1，所以一個 task 就是一個 job。stdout 第二行印 `maxtasksperchild=<k>`。
- **為什麼結果不會變**：
  - job 之間本來就不共用狀態：父行程先 `prep` 好紀錄，worker 只讀。
  - 原本 `imap_unordered` 把哪個 job 分給哪個 worker 就不固定，H200 和 acvlab 照樣逐值相同。所以每個 job 的結果本來就和「同一個 worker 之前跑過什麼」無關。
  - 換 worker 只改變「哪個行程跑哪個 job」。
- **驗證**：G1 run `20261105-100635-icpe-sim-g1-54fa2d8`。這次用的是**預設 k＝4**（stdout 印 `maxtasksperchild=4`），所以檢查的是新預設本身。結果 4,416 列 × 29 欄、不同的格子 0；自檢 997／997。
  - icpe-trx 自己在 54fa2d8 上也做了：
    - G0：128,064 格 IDENTICAL；
    - xc_sa：1,280 格 IDENTICAL；
    - 40 worker 試跑 SA：PSS 持平 7.3–9.4 GiB（1,329 個 job）。
    - 出處：STATUS_TRX 9410199、d318434，是他們的數字。
- **代價**：每 k 個 job 多一次 fork（加上新 worker 重新寫到父行程頁面時的 copy-on-write）。這個時間 NOT_MEASURED。
- **沒做的**：直接打斷循環參照，例如 `_job` 最後清掉 Engine 的屬性，或父行程 `gc.freeze()` 加上每個 job 後 `gc.collect()`。這要另一次 G1，而且 k＝4 已經把累積限制住了，所以先不做。之後若要做，會另寫一項。
- **有利還是不利**：無（結果逐位元相同）。

---

## 第三批：Rust 引擎（2026-11-06）

### 16. Rust 引擎：`x1 sweep --engine rust`（預設 py＝原本；`code/p3x/x1fast/`、`code/p3x/x1_rust.py`，commit 0b0ef1e）

- **為什麼**：
  - 使用者 11/5 的決定「並行驗證、接手未開始的」，由 icpe-ana 轉達。
  - icpe-ana 把 `x1_sim.Engine` 逐行移植成 Rust：origin/icpe-next-ana @ aee1527 的 `code/icpe/ana_x1fast/v3_prefix_fork`。
  - 請 icpe-sim 做三件事：
    1. 審查；
    2. 包裝成和 x1 sweep 欄位相同的 `x1_sweep.csv`；
    3. 跑 G1、xcheck，並量 TraceLab 一個 block 的時間與記憶體。
- **審查**（self-review，不是另一個模型審的）：
  - 引擎部分（原檔第 1–1218 行）**一行沒改**，用 diff 確認過。我的版本只在第 9 行後多 5 行說明。
  - **同分時取哪一個**：Rust 的 `max_by_key` 取最後一個最大值，Python 的 `max` 取第一個。不會造成差異，因為所有比較鍵都含有不會重複的值（`-last[x]` 或 `x` 本身），不可能同分。
  - **nx_cache**（每個 session「下次回來」的快取）：只在兩個條件都成立時沿用。
    1. 群組的 epoch 沒變。群組＝共用 block 的 session，用 union-find 合併；群組內 ref 有變就 `bump`。
    2. `cur` 還沒走到快取的值（或快取值是 INF）。
    - `next_access(b, cur)`＝b 在 cur 之後第一次被讀。在 cur 走到它之前，這個值不會變，所以這樣沿用和每次重算相同。
  - **Rust 版不支援的功能**：W4 雜訊、慢層容量上限、STAIR。這些 job 由 `unsupported()` 交給 Python Engine，在 Rust 的 job 之後跑。
- **改了什麼**：
  - `code/p3x/x1fast/src/main.rs`：只改 main() 的輸入輸出。
    - 原本 main() 裡的 parse／build／finish 三個閉包搬成函式，內容照抄。
    - 加**常駐模式**（`X1FAST_SERVE=1`）：
      - 紀錄只讀一次；
      - stdin 每行一個 job 目錄，`X1FAST_THREADS` 個執行緒一起跑；
      - 每做完一個印一行 `done <目錄> <run_s> <wall_s>`。
    - jobs.txt 一行不是 13 欄就停，不默默忽略多出來的欄位。
  - `code/p3x/x1_rust.py`（新）：
    - `prepare_trace`：紀錄 → CSR 二進位檔，用內容 sha256 定址。
      - 用 A 驗 B：acc 總長＝full 總長；每個 block 的請求序號嚴格遞增；block id < 2^32。
    - `write_job`：f／Fc 直接取自 Python 的 View。
    - `read_outputs`：逐請求陣列、I/O 線狀態、預取位元組寫回 View。
    - `Server`：每份紀錄一個常駐的 x1fast。
    - 編譯：`cargo build --profile fast --offline --locked`，不加 target-cpu（不用 FMA，結果和 CPU 型號無關）。
      - 執行檔放在 `$X1_RUST_WORK/build-<原始碼 sha>/`。
  - `code/p3x/x1_sim.py`：
    - 原本 `_job` 的「摘要、存 npz、組一列」搬成 `_job_rows`，內容照抄；Python 與 Rust 共用這一段。所以 CSV 欄位、格式、npz 都相同。
    - `--engine {py,rust}`，預設 `$X1_ENGINE` 或 py。
    - rust 時：
      - code_sha 後面加 `+rs<原始碼 sha 前 8 碼>`；
      - stdout 印執行檔 sha256 與退回 Python 的 job 數；
      - 可選 `$X1_RUST_TIMING_LOG`（每個 job 一行 JSON）。
  - `code/icpe/next_sim_cmp.py`（新）：通用的逐格比對。
    - 比對時不看 run_id、ts、elapsed_s、code_sha、host 這幾欄。
    - 鍵自動含 GPU、雜訊、慢層上限的欄位。
- **CPU 用量**：
  - `--workers w` 時：每份紀錄一個 Rust 行程，裡面有 w 個執行緒；另有 `max(1, w//4)` 個 Python 摘要行程，加上父行程。
  - 合計約 1.25 w 個核心〔算術〕。
- **驗證**：run `20261106-002959-icpe-sim-g1-0b0ef1e`，detached checkout 0b0ef1e、共用 venv、nice 10。
  - **G1**：H200 V1 的 sweep 跑兩次，都是 4,416 列 × 29 欄、不同的格子 0。
    - Python 預設引擎（4 workers）：6,453 s。
    - `--engine rust`（3 執行緒）：283.5 s。
    - 兩條同時段跑，load 約 70，時間只供參考。
  - **自檢**：997／997。
  - **xcheck**：icpe-data 的 Python 輸出（origin/icpe-next-data @ 1546086 的 next_b_queue.py 參數）用 `--engine rust`（3 執行緒）重跑，逐格比。14 個 block 全部「不同的格子 0」，退回 Python 的 job 0：
    - xc_sa 32 列、xc_bl 32 列；
    - bl／mc／fi × nvme807、load、wt、pfgpu：544／2,176／1,632／288、1,088／4,352／3,264／576、544／2,176／1,632／288 列。
  - **TraceLab c1 一整個 block**：見 STATUS_SIM 的「Rust 引擎」那一節。
    - run `20261106-003131-icpe-sim-rust-tl_c1`，7 執行緒，對 icpe-data 的 tl_main/tl_c1 Python 輸出（7,344 列）。寫這項時還在跑，結果補在 STATUS_SIM。
- **有利還是不利**：無（逐格相同；Python 引擎只把摘要那一段搬成函式，計算沒有改，G1 用預設 py 驗過）。

### 17. Rust 引擎加 W4 雜訊的 B1 抽法（icpe-ana v4；`code/p3x/x1fast/src/main.rs`、`code/p3x/x1_rust.py`，commit 1304345）

- **為什麼**：
  - 使用者 11/6 的決定（STATUS_ANA 1ea44c9 轉達）：v4 的移植由 icpe-ana 做，icpe-sim 負責審查、包裝、G1。
  - v4 放在 origin/icpe-next-ana @ 279eb7e 的 `code/icpe/ana_x1fast/v4_noise_on_0b0ef1e/src/main.rs`。它以 0b0ef1e 的 main.rs 為底，只加了雜訊。
- **改了什麼**：
  - `main.rs` 換成 v4，**原樣複製**（sha256 1b0e7e08…）。Cargo.toml 不換：套件名維持 x1fast3，v4 也沒有新的相依。
  - `x1_rust.py`：
    - `unsupported()`：慢層上限 → `slow_cap`；雜訊抽法不是 B1 → `noise_B2`，這兩種退回 Python。其餘雜訊（B1）交給 Rust。
    - `write_job(…, nz=)`：jobs.txt 那行多一欄 `nz=<P|M>,<repr(λ)>,B1,<seed>`。
      - λ＝0 也照送，對應 Python 的 `Engine(noise=dict(…))`：λ＝0 仍會抽亂數，但直接回傳真值。
  - `x1_sim.py`：`_rust_sweep` 把 job 的雜訊設定傳給 `write_job`（只改這一行；Python 引擎的路徑沒動）。
- **審查**（self-review）：
  - **亂數**：對照 CPython 的 `_randommodule.c` 與 `random.py`，逐項相同：
    - `init_genrand`、`init_by_array`（abs(seed) 切成 32 位元的字、0 → [0]）；
    - `genrand_uint32`（產生 624 個值的那一段用 `%624` 的索引寫法，和 C 的三段迴圈等價）、tempering；
    - `random()`（53 位元）；
    - `choice`＝`_randbelow_with_getrandbits`（k＝bit_length；venv 是 Python 3.12.15）；
    - `uniform`＝a＋(b−a)·random()。
  - **雜訊的呼叫點**和 Python 一一對應：
    | Python | Rust |
    |:--|:--|
    | `_next_req_nz`（BEL／BELS 選擇、dead、make_room） | `next_req` |
    | `_g_next`（GPU 的 BEL、GPU 預取） | `g_next` |
    | demote 與 `_pf_push` 的兩個 `T = self._nz(s, T)` | 同樣兩處 |
  - **nx_cache**：存的是加雜訊**前**的值，每次查詢才算 `nz`；Python 也是每次查詢才算。所以模型 M 用到 `cur` 的部分不會因為快取而換了時間點。
  - **B1 的抽樣**：在 `step` 裡 `cur＝r` 之後、GPU 分流之前，和 Python 相同。
  - **其他多出來的欄位**（慢層上限等）→ Rust 直接停下來。
  - 有雜訊的 job 不參與批次模式的 `X1FAST_FORK` 分組；常駐模式本來就不用。
- **icpe-ana 的對拍**（逐請求陣列＋summarize 每欄，和 Python Engine 比）：
  - Bailian 雜訊 250／250；
  - Mooncake 雜訊 250／250；
  - Bailian 寬矩陣回歸 278／278。
  - 出處：STATUS_ANA 03:05、`results/icpe_next/ana/rust_difftest_jobs.csv`；是他們的數字。
- **驗證**（icpe-sim）：run `20261106-043047-icpe-sim-g1-1304345`（detached checkout 1304345、共用 venv、nice 10；Rust 6 執行緒，Python 8 workers；04:31–06:11）。
  - **G1**（`--engine rust`，新功能全關）：4,416 列 × 29 欄，不同的格子 0，退回 Python 0 個 → **PASS**。
  - **xcheck**：0b0ef1e 驗過的那 14 個 block 再用 Rust 跑一次，和 icpe-data 的 Python 輸出逐格比，全部「不同的格子 0」，退回 Python 0 個。
  - **雜訊對拍**：同一組參數先用 `--engine py` 跑、再用 `--engine rust` 跑，逐格比；比對的鍵含 4 個 noise 欄。
    | block | 參數 | job（Rust／退回 Python） | 列 × 欄 | 不同的格子 |
    |:--|:--|:--|:--|--:|
    | nz_bl | Bailian nb；P、M × λ 0、0.1、0.3 × B1、B2 × seed 0、7；pf-target cpu、gpu | 1,764（980／784，都是 B2） | 2,232 × 44 | 0 |
    | nz_mc | Mooncake conversation R1；同上但只有 B1、pf-target cpu | 740（740／0） | 1,000 × 41 | 0 |
  - **對拍不是空的**：λ>0、B1、seed 0 的列，和同一格 λ＝0 的列比，nz_bl 496／496、nz_mc 400／400 至少有一個數值欄不同（`nz_effect.py`、`nz_effect.txt`，在 run 目錄）。
  - 摘要：`results/icpe_next/sim/g1_compare.csv`（+1 列）、`rust_cmp.csv`（+16 列）、`rust_timing.csv`（+17 列）。
- **有利還是不利**：無（逐格相同）。


### 18. `--engine rust` 的父行程準備加快取（icpe-data 11/6 的請求；`code/p3x/x1_rust.py`、`code/p3x/x1_sim.py`，commit c7ef740）

- **為什麼**：
  - icpe-data 量到：Rust 佇列跑 tl_c4 時，父行程準備了約 16 分鐘，只用 1 個 CPU，另外 35 個執行緒閒著。準備的內容是讀 npz、檢查、建 full／acc、寫 Rust 紀錄檔。
  - P1–P3 還有 18 個 TraceLab block，同一個紀錄變體要重複準備 6–7 次。
  - 數字出處：STATUS_DATA 08:25 與 icpe-data 的訊息，是他們的數字。
- **改了什麼**：
  - `x1_rust.prepare_cached`：`cmd_sweep` 的 rust 分支改用它，取代原本的 `prep`＋`prepare_trace`。
    - **快取**：只對 nb: 紀錄。
      - 鍵＝npz 內容 sha256＋`x1_sim.py`／`x1_nb.py`／`x1_rust.py` 的 sha256＋numpy、Python 版本＋`BF`。
      - 放在 `$X1_RUST_WORK/prep-<鍵前 24 碼>/`：`lite.pkl`（P 去掉 full、acc）＋`KEY.json`（鍵的組成、lite.pkl 與紀錄檔的 sha256、請求數）。
      - 寫入方式：先寫暫存目錄再原子搬移。同時有別的 sweep 寫好了，就丟掉自己的。
    - **命中時的驗證**：
      - KEY.json 的鍵與組成要相同；
      - lite.pkl 的 sha256 要相同；
      - Rust 紀錄檔**重算** sha256 要相同；
      - 紀錄檔的 n／B／S 要和 lite P 對得上。
      - 任何一項不過就停下來，不會默默重建。只有「沒有這個目錄」才重建。
    - **lite P**：一份紀錄的 job 全部走 Rust 時，父行程的 P 不留 full、acc。
      - Rust 讀紀錄檔；父行程與摘要只用 `len(P["full"])`（`NoFull`：只有長度，其他用法丟錯誤）、`nmax`、`tl`、`kindc`、`ws_GiB`、`user_trigger`。
      - 有任何 job 退回 Python Engine（B2、慢層上限、STAIR）的紀錄，照舊建完整的 P。
    - `X1_PREP_CACHE=0` 關掉快取，lite P 不受影響。
  - `x1_sim.cmd_sweep`：
    - 一開始的 `prep` 迴圈只在非 rust 時跑，`--engine py` 的路徑不變；
    - rust 分支改呼叫 `prepare_cached`，stdout 那行多印 `prep=<built|built_saved|cache_hit> <秒> lite=<True|False>`。
- **驗證**：run `20261106-082036-icpe-sim-g1-c7ef740`（detached checkout c7ef740、nice 10、`X1_RUST_WORK`＝run 自己的 work/，所以第一次一定沒有快取；08:20–09:28，load 約 65–70）。
  - **G1（Python）**：`--engine py`、6 workers、新功能全關，對 H200 V1：4,416 列 × 29 欄，不同的格子 0 → **PASS**。
  - **G1（Rust）**：`--engine rust`、2 執行緒：4,416 列 × 29 欄，不同的格子 0 → **PASS**。4 個非 nb 紀錄都是 `prep=built … lite=True`（不快取，但走 lite P）。
  - **TraceLab c1 子集**（cpu 64，其餘參數照 icpe-data 的 tl_c1；`--engine rust --workers 2`）連跑兩次：
    - 第一次 `prep=built_saved 672.2s lite=True`，第二次 `prep=cache_hit 9.3s lite=True`（9.3 s 含重算 2.6 GB 紀錄檔的 sha256）。
    - 兩次都和 0b0ef1e 的 TL c1 整個 block（run `20261106-003131-icpe-sim-rust-tl_c1`，那份已和 icpe-data 的 Python 輸出逐格相同）比：816 列 × 40 欄，不同的格子 0；逐請求 npz 686 個檔、1,632 個陣列，不同 0 → 兩次都 **PASS**。
    - 單一行程最大 RSS（`time -v`）：14,896,312 KiB → 4,461,960 KiB。牆鐘 1,672.6 s → 1,108.3 s；sweep 本身 999.1 s → 1,098.2 s（共用機器，時段不同）。
  - **有 job 退回 Python 的紀錄**（Bailian coder、cpu 64、O-BEL、B1＋B2 雜訊）：`prep=built_saved 0.6s lite=False`；和 v4 驗證的 nz_bl Python 輸出比，144 列 × 44 欄，不同的格子 0 → **PASS**。
  - 摘要：`results/icpe_next/sim/g1_compare.csv`（+2）、`rust_cmp.csv`（+3）、`npz_cmp.csv`（+2）、`rust_prep_cache.csv`（新，+7）。
- **有利還是不利**：無（逐格相同；只改準備的方式，計算沒有改）。
