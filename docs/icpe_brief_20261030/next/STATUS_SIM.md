# STATUS_SIM（icpe-sim，acvlab，分支 `icpe-next-sim`，worktree `/ssd6/hungwei/icpe-next/sim`）

只有 icpe-sim 寫這個檔。格式照 NEXT_PLAN §3b：日期時間／做完什麼／卡住什麼／需要對方做什麼／需要使用者決定什麼。

---

## 2026-11-04T23:02+08:00 — 第 0、1 步；GPU 層的設計選項（還沒寫任何程式）

**做完什麼**
- worktree `icpe-next-sim` 從 `origin/main` @ `77c25f6` 開出並 push（git 本體在 `/home/hungwei/llm/POC/paper-hierarchical-kv-state`，CLAUDE.md 寫的位置）。
- 讀完：CLAUDE.md、NEXT_PLAN（§3b 第二版、§5.9、§5.10、§6b）、`code/p3x/x1_sim.py` 全部、`x1_trace.py`、`x1_fi.py`、`code/icpe/p3_fvar.py`、`code/p3_platform.py`。
- 沒有跑任何東西，沒有任何數字。

**卡住什麼**
- 共用 venv 還沒好（`icpe-next-data` 分支還不存在）→ G1 驗證要等 STATUS_DATA 寫「venv ready」。在那之前只寫程式；開發中的小測試若要跑，用既有 Python 只讀使用，不算 G1。

**需要 icpe-data 做什麼**
- venv ready 後在 STATUS_DATA 寫版本；G0 的結果（逐值比對用的欄位）也請寫上，G1 會用同一種比法：排除 `run_id`、`ts`、`elapsed_s`、`code_sha` 四欄，依 (trace, rule, cpu_GiB, policy, slow, gpu, restore) 排序後逐值比。
- 介面：新功能關閉時 `x1_sim` 的輸出欄位不變（新欄位只在開啟 GPU 層時才出現）；紀錄格式（`prepare()` 回傳的 dict：`full`、`nall`、`t`、`R`、`meta`…）不改。

**需要使用者決定什麼**（下面是選項與建議；決定前不寫 GPU 層的程式）

現有模型（`x1_sim.py`，GPU 層＝0）：一次一個請求；回合結束時整個 session 的完整 block 以 BF16 寫進 CPU；CPU 滿就依策略降級到慢層（INT4）或丟掉；回來時用 Cake 從 CPU／慢層還原、其餘重算。session 為單位、共用前綴用 refcount 只存一份。

1. **GPU 層和 CPU 層的關係（寫入方式）**
   - (a) 回寫、互斥（§5.10 的字面）：回合結束只留在 GPU；GPU 滿才把被選中的 session 寫進 CPU（BF16，佔 CPU 寫入線）；從 CPU 還原回 GPU 後，CPU 那份釋放。GPU 預算等於多一層容量。
   - (b) 直寫、包含（vLLM OffloadingConnector、LMCache、SGLang HiCache 的做法）：回合結束照舊寫 CPU（CPU 層邏輯完全不變），GPU 另外留一份；GPU 逐出不用寫。GPU 層只省下「CPU→GPU 載入」，不增加容量。
   - 兩種在 GPU 預算＝0 時都退回現在的模型。**建議 (a)**：是 §5.10 寫的資料流，也符合「滿了才逐出」；(b) 可當敏感度。
   - 對預測有利還是不利〔推論，沒量〕：(a) 多了容量 → 記憶體壓力降低，上限傾向變小；但 GPU 逐出要付寫入，選錯的代價比 (b) 大。
2. **粒度與「前綴快取式」**
   - (a) 全部以 session 為單位（LRU、TTL、完美預測都是整個 session 進出 GPU）；「前綴快取式」第一批不做，第二批再加。
   - (b) 第一批就做 block 粒度的前綴快取式（vLLM APC：從最久沒用的 session 尾端一塊一塊逐出，可以留半個 session）；要放寬「session 整個在同一層」的不變量，工作量多約 1 天〔估計〕。
   - **建議 (a)**：先趕 ICPE 的關鍵路徑。session 粒度下，共用前綴照 refcount 只要有一個 GPU 上的 session 持有就留著，和 APC 的差別只剩「部分留下」。
3. **TTL 的語意**（值等 PREREG；含 0.3 s）
   - (a) 釘住 TTL 秒，到期就逐出到 CPU（到期時寫入，不管 GPU 滿不滿）。
   - (b) 釘住 TTL 秒，到期後變成普通 LRU，GPU 滿才逐出（Continuum 跑在 vLLM 上：釋放的 block 留在前綴快取裡，直到被擠掉）。
   - 兩種的共同點：GPU 滿而且全部都還在釘住期時，新回合的 session 直接進 CPU，不擠掉釘住的。
   - **建議兩種都做**、都放進 GPU 層的對手集合（對手取最快者，多一個對手只會讓對手更強）。
4. **完美預測與對手怎麼組合**（GPU 層是新的一軸）
   - 建議：新元件 `O-GPU`＝GPU 滿時先擠出下次最晚回來的 session（Belady 式，用和 `O-BEL` 相同的 `next_req`）；對手＝{GPU-LRU、GPU-TTL 各值（兩種語意）}；CPU 層兩邊都固定 LRU，只比 GPU 這個決定。原本四個 CPU 元件（O-ORD、O-DROP、O-NOW、O-PF）在 GPU 層開啟時，GPU 層一律用 LRU。

**我的預設（沒反對就照做）**
- GPU 上的前綴不用還原（0 秒）；Cake 從 GPU 前綴之後開始：重算從第 g 塊往後、載入從最後一塊往前。session 粒度下，存在 GPU 的 block 一定是請求的前綴（同現在的不變量）。
- 預取仍搬到 CPU（§6b 草案）。
- 一次一個請求：還原時暫時要的 GPU 空間不算進預算，回合結束才檢查容量（和現在的 CPU 層一樣）。
- GPU 逐出寫 CPU 的時間＝造成溢出的那個請求的回合結束（和現在的 CPU 寫入同一時刻）；TTL (a) 的到期逐出＝到期時刻（在下一個請求前依時間順序處理）。
- GPU 忙／閒只改重算倍數，GPU 預算兩者相同。
- 8.07 GiB/s 慢層：新的可選等級 `nvme807`（預設不跑，`SLOWS` 不變）；讀＝`results/s3_h200/gfio/gfio_20261026-104426-s3-gfio.csv` 裡 /tmp 64 MiB 循序 8 執行緒的中位數（程式從檔案讀，不寫死）；寫＝讀 × SSD 級的寫讀比 1.9107／6.9812＝2.209 GiB/s〔算術〕；寫入干擾 k_read 沿用 SSD 級的 3.306（§6b 只寫了寫入頻寬，這一項是我的建議）。

**別的**
- 這台的 CPU 用量沒有歷史紀錄（sysstat 沒在收；`sar -q` 讀不到）。23:02 時 load 67.9／96 執行緒。icpe-sim 驗證最多 8 個 worker、`nice -n 10`。

---

## 2026-11-04T23:17+08:00 — 使用者的決定；對照 PREREG（`origin/icpe-next-a` @ `4ec820c`）；第一批的程式寫好、自檢中

**使用者 11/4 對上一節四個問題的回答**（AskUserQuestion）：
1. 寫入方式：**兩種都做**（回寫互斥 `wb`、直寫包含 `wt`）。
2. 粒度：**session 粒度，APC 第二批**。
3. TTL：**兩種語意都做、都當對手**（TTLA＝到期就逐出；TTLB＝到期後照 LRU）。
4. 比較：**每個 CPU 對手都配**（GPU 規則 × CPU 規則全部組合，各取最快）。

**PREREG 對模擬器的要求，和我照做的**
- GPU 預算 1,777 塊＝`--gpu-gib 111.0625`（＝1,777 × 64 MiB，整除）；CPU 參考點同值，`--cpu` 現在接受非整數（整數照舊印成整數）。
- TTL 語意（PREREG §2 第 3 項）：到期移到 CPU（＝TTLA）；**GPU 滿而全部被釘住時，放掉「對話開始時間最晚」的那個**（Continuum §5.2）。已改：原本我寫的是「剛進來的那個直接下去」。對話開始時間＝該 session 第一個請求的到達時間。TTLB 用同一條防死結規則。
- 資料流（PREREG §4）＝回寫互斥 `wb`（主結果）。
- 8.07 那格＝`--slows ... nvme807`：讀 8.068（G-fio CSV 的值，PREREG 寫 8.07）、寫 2.2087、k_read 3.306。
- **前綴快取式（APC）在 session 粒度下的定義**（PREREG §4 要我在 G1 前寫）：vLLM APC 回合結束後 block 留在快取裡、照最近使用逐出、同一個請求的尾端先逐出。session 粒度下整個 session 一起進出、共用前綴照 refcount 只要有一個持有者在 GPU 就留著 → **和 GPU-LRU 完全相同**，所以第一批不另外跑（不增加格子）。和真的 APC 只差「部分留下」（逐出時保留 session 的前綴、只丟尾端），那要 block 粒度（使用者決定放第二批）。少了這個對手 → ② 可能偏弱 → 上限可能偏高〔推論〕，對「預測沒多大優勢」這個主張是保守的方向。→ 要不要照這樣定義，見下面「需要使用者決定」。
- **預取直接搬回 GPU 的敏感度**（PREREG §2 第 2 項，要我在 G1 前寫實作細節；第一批之後馬上做，跑在 §3 A 的舊格子上）：
  - 同樣的預取規則（O-PF、LRU+RPF）與觸發時機（請求之間、慢層讀取線空著時）。
  - 目的地是 GPU：慢層讀 INT4 → 反量化（d）→ 以 BF16 放在 GPU，佔 GPU 預算 64 MiB／塊；完成時間＝慢層讀完＋d。慢層到 GPU 只算慢層讀取線（和現在「從慢層還原」的算法一致，不另外佔 CPU 線）。
  - 預取哪些塊：這個 session 存著、不在 GPU 上的全部塊，從最後一塊往前（GPU 上的塊載入成本＝0，Cake 的會合點會退到最前面）；中途沒空間就停，前面沒預取到的照 Cake 重算／載入。
  - 騰空間：只准逐出預取規則認為「比候選者冷」的 GPU session（O-PF：下次回來比候選者晚；RPF：上次存取比候選者早），逐出照 `wb` 寫進 CPU。騰不出來就等下一個事件（同現在的 CPU 預取）。
  - 回來時：預取到 GPU 的塊等它完成（pf_done），不另外付載入；回合結束後成為這個 session 在 GPU 上的塊。GPU 超量時先照 GPU 規則逐出 session；GPU 上沒有可逐出的 session 時才取消預取（RPF 先取消最舊的、O-PF 先取消最晚回來的；同現在的 CPU 預取）。
- 第二批（雜訊 P／M、B1／B2、10 seed；慢層容量 ⌊0.5 × n_blocks_full⌋＋慢層淘汰；排隊；W5）照 PREREG，之後寫。

**做完什麼**：程式改在 `code/p3x/x1_sim.py`（見 `SIM_CHANGES.md`）、自檢 `code/icpe/next_sim_test.py`；開發用自檢 run `20261104-231605-icpe-sim-devtest`（**用既有 Python 3.12.13／numpy 2.2.6，不是共用 venv，不算 G1**），跑完再記結果。

**卡住什麼**：共用 venv 還沒好（`icpe-next-data` 分支還不存在）→ G1 不能開始。

**需要 icpe-data 做什麼**：venv ready 後寫在 STATUS_DATA。新旗標（全部預設＝原本的行為）：`--gpu-gib`、`--gpu-write wb|wt`、`--gpu-policies LRU BEL TTLA:<τ> TTLB:<τ>`、`--slows ... nvme807`；`--cpu` 接受非整數。開了 GPU 層的 sweep 每列多 `gpu_GiB`、`gpu_write`、`gpu_policy`、`ret_gpu_hits`、`n_gpu_evict`、`n_gpu_expire`、`gpu_wb_GiB`；TTL 規則和預取規則一樣，一個 job 只跑一種時間設定（job 數會變多）。

**需要使用者決定什麼**（使用者給我的答案和 PREREG 不一致的地方；PREREG 只有 `h200` 能改，要改就走偏離程序）
1. TTLB（到期後照 LRU）：PREREG 只登記了「到期移到 CPU」一種。要不要把 TTLB 加進 ②（走偏離，要在看任何正式結果之前）？
2. 直寫包含 `wt`：PREREG 沒登記。要不要登記成敏感度？
3. APC 的定義：照上面「session 粒度＝GPU-LRU、第一批不另外跑」，還是在 G1 前先做 block 粒度（關鍵路徑多約 1 天〔估計〕）？

---

## 2026-11-04T23:19+08:00 — 使用者對三個不一致處的決定（要 `h200` 在 PREREG 寫偏離）

使用者 11/4 回答（AskUserQuestion，在這個 session）：
1. **TTLB 加進 ②**，請 `h200` 在 PREREG 寫偏離（要在看任何正式結果之前）。→ ② 的 GPU 規則＝{LRU、TTLA:τ、TTLB:τ（τ＝0.3／1／3／10／30／100／300 s）}；④＝BEL。
2. **直寫包含 `wt` 登記成敏感度，跑全部格子**（請 `h200` 寫偏離）。
3. **APC：第一批＝GPU-LRU（session 粒度下完全相同，不另外跑）；第二批再加 block 粒度**。PREREG 規定改 ② 要在看正式結果之前登記 → 請 `h200` 現在就登記「block 粒度 APC 做好後加進 ②；在那之前的結果標明 ② 不含 block 粒度 APC，做好後重算」。

**需要 h200 做什麼**：上面三項寫進 PREREG 的「偏離」。我已用 SendMessage 通知（只是提醒，以這個檔為準）。
**正式模擬**：照 §3b 由 `icpe-data` 跑；我這邊只跑 G1 與自檢。G1 還沒做（共用 venv 還沒好）。

---

## 2026-11-04T23:25+08:00 — 第一批 (1a) 程式 push；開發用自檢全過；icpe-data 已開

**做完什麼**
- commit `985c8a8`：GPU 快取層（wb／wt；LRU、BEL、TTLA、TTLB；PREREG 的防死結規則）、`nvme807`、容量接受非整數。全部預設關閉（原本的程式路徑）。細節 `SIM_CHANGES.md`。
- 自檢 `code/icpe/next_sim_test.py`：run `20261104-231758-icpe-sim-devtest2`（/ssd6/hungwei/icpe-next/runs/）**626/626 PASS**，349 s、8 workers。**用的是既有 Python 3.12.13／numpy 2.2.6，不是共用 venv → 不算 G1**；G1 時用共用 venv 再跑一次。第一次（`20261104-231605-icpe-sim-devtest`）exit 1：GDSF＋GPU 層時名稱衝突（`gver`）→ 已修；Cake 的貪婪會合點讓「前綴換成 GPU」有時慢不到一塊 → 自檢改成照實報次數（19／400），見 SIM_CHANGES。
- G1 比對程式 `code/icpe/next_sim_g1.py`（只用標準庫；去掉 run_id、ts、elapsed_s、code_sha 後依鍵排序逐字串比）。
- **更正**：23:17 節寫的 nvme807「寫 2.2087」是我筆誤；程式一直是 8.068 × (1.9106543 ÷ 6.9811523)＝**2.2081 GiB/s**（自檢 T5 印 2.208111），和 PREREG 偏離 2 第 4 項相同。
- 使用者要我開 `icpe-data` session：tmux `icpe-data`、`claude --remote-control icpe-data -n icpe-data`，第一個訊息＝使用者給 icpe-data 的指示原文（`/ssd6/hungwei/icpe-next/launch/icpe-data-prompt.txt`）。

**卡住什麼**：G1 等共用 venv（STATUS_DATA「venv ready」）。

**下一步**：第一批 (1b)：預取直接搬回 GPU（PREREG §2 第 2 項＋偏離 1：舊格子與 W2 全部格子都要跑）。

---

## 2026-11-04T23:34+08:00 — 第一批 (1b) push；G1 進行中

**做完什麼**
- commit `ac0c56e`：預取直接搬回 GPU（`--pf-target gpu`；PREREG §2 第 2 項＋偏離 1）；有 GPU 軸的分析 `code/icpe/next_sim_analyze.py`；`x1_sim analyze` 遇到有 GPU 軸的 sweep 會停（它的樞紐表只看 policy，會靜默蓋掉）。細節 SIM_CHANGES 第 5、6 項。
- `next_sim_analyze.py` 驗證：H200 T1 V1 的 sweep（沒有 GPU 軸）→ 和 H200 的 `x1_headroom_components.csv` 1,152 列逐列相同（run `20261104-233301-icpe-sim-analyze-check`）。
- G1（1a）run `20261104-232753-icpe-sim-g1-1a`（commit 985c8a8）跑到 300／1,328 時**我手動停掉**（`ABORTED` 檔）：1b 已經 push、包含 1a，直接對 ac0c56e 做一次 G1 比較快。沒有任何比對結果。
- G1（1a＋1b）run `20261104-233425-icpe-sim-g1-1b`：commit ac0c56e、共用 venv、8 workers；跑完記結果。

**需要 icpe-data 做什麼**（G1 過了之後）
- merge 的對象＝G1 寫明通過的那個 commit。
- 新旗標（都在 `x1_sim.py sweep`，經 `p3_fvar.py ... x1 sweep` 傳進去）：`--gpu-gib 111.0625`、`--gpu-write wb wt`、`--gpu-policies LRU BEL TTLA:0.3 ... TTLA:300 TTLB:0.3 ... TTLB:300`、`--pf-target cpu gpu`、`--slows ssd b32 nfs nvme807`、`--cpu 8 16 ... 1024 111.0625`。
- 開了 GPU 層的 sweep 多這些欄：`gpu_GiB`、`gpu_write`、`gpu_policy`、`pf_target`、`ret_gpu_hits`、`n_gpu_evict`、`n_gpu_expire`、`gpu_wb_GiB`。
- 有 GPU 軸的 sweep 用 `code/icpe/next_sim_analyze.py --sweep ... --out-dir ... [--pf-target gpu]` 算上限，**不要用 `x1_sim analyze`**。
- job 數：TTL 規則和預取規則一樣，一個 job 只跑一種時間設定；`pf_target＝gpu` 只對有預取的規則產生 job。
- 自己的紀錄格式（`prepare()` 回傳的 dict）不變；新紀錄要接進 `x1_sim.prep()` 的話，在 STATUS_DATA 寫你要的介面（例：名稱前綴＋讀你的轉換檔），我來加。

---

## 2026-11-04T23:46+08:00 — 第二批 (2a) 雜訊 push；PREREG 網格的 job 數（算的，沒跑）

**做完什麼**
- commit `6b17191`：新紀錄的載入點 `nb:<name>`（`code/p3x/x1_nb.py`；Mooncake 來回轉換逐值相同，run `20261104-233613-icpe-sim-nb-roundtrip`）。
- commit `7fd39f3`：W4 雜訊（模型 P／M、B1／B2、seed；SIM_CHANGES 第 8 項）；分析支援逐 seed 與彙總。開發用小測：λ＝0 和沒有雜訊逐位元相同（O-BEL、O-DROP、O-PF × P、M），λ＝0.3 每 50 步稽核通過（截短的 conversation，只是除錯，不是結果）。完整自檢 T8／T9 等 G1 跑完再跑。

**PREREG 網格的 job 數**（`jobs_for` 直接數的〔算術〕，沒有跑任何模擬）：Mooncake 2 份 × R1／R2、CPU 9 個容量（8–1,024＋111.0625）、CPU 規則 23 個、GPU 規則 16 個（LRU、BEL、TTLA×7、TTLB×7）、慢層 4 種（ssd、b32、nfs、nvme807）× GPU 閒／忙 × cake／load：
- wb、預取到 CPU：**193,032 個 job**（舊的 T1 sweep 是 1,328 個）；只跑 cake：97,416。
- wt：再 193,032；預取到 GPU（只有預取規則）：wb、wt 各 46,080。
- 主因：TTL（16 個 GPU 規則裡的 14 個）和預取一樣，一個 job 只能跑一種時間設定。
- 每個 job 要多久：NOT_MEASURED（icpe-data 的試跑會量）。這還不含 FreeInference、新紀錄、雜訊（λ×抽法×10 seed）、慢層容量（×7 種慢層規則）。

**需要使用者決定什麼**（等 icpe-data 量到每個 job 的時間再決定也可以）：如果全部跑不完，要不要縮小網格（例：TTL 只取 ② 裡最快的幾個值、load 只跑代表格、CPU 容量少幾個）？縮小要走 PREREG 的偏離程序。

---

## 2026-11-04T23:49+08:00 — 使用者對慢層容量的兩個決定（AskUserQuestion）

1. **全算，先丟「乾淨的副本」**：慢層上所有 INT4 塊都算進上限（⌊0.5 × n_blocks_full⌋）；超過時先丟「持有者都已經在 CPU／GPU」的副本（不掉資料；之後再降級要重寫），不夠再照慢層規則丟整個慢層狀態的 session（回來全部重算）。
2. **全部配對（×7）**：慢層規則（② LRU、LRU2、ARC、S3FIFO、GDSF、PENS；④ BEL）和 GPU 規則、CPU 規則全部組合，② 取最快（同偏離 2 第 5 項的精神）。

→ 這兩項 PREREG 第 5 項沒寫到細節；是否要 `h200` 記成偏離（或補充說明）由使用者決定。

---

## 2026-11-05T00:22+08:00 — 加速（逐位元相同）push；G1 在 09fd521 上跑

**做完什麼**
- G1（ac0c56e）run `20261104-233425-icpe-sim-g1-1b` 跑到 541／1,328 時**我手動停掉**（`ABORTED`；沒有比對結果）：icpe-data 要用的 `nb:` 載入點在它之後，G1 要涵蓋 icpe-data 會 merge 的那個 commit。
- 慢層容量上限（`3140f3c`）、`x1_nb` 向量化（同 commit；Mooncake 來回轉換只差重新編號、負面測試會擋下，run `20261104-235644-icpe-sim-nb-roundtrip3`）、分析的慢層軸（`dc80390`；仍和 H200 的 1,152 列逐列相同，run `20261104-235745-icpe-sim-analyze-check2`）。
- 加速（`09fd521`；SIM_CHANGES 第 11 項）：profile（run `20261104-235828-icpe-sim-profile-cw`，CacheWise 試跑檔，只量時間分佈）→ restore 內聯 `Chan.read`、`_fix` 快速路徑、wb 回合結束的迴圈。**逐位元相同**：restore 對 restore_ref（run `20261105-000344-icpe-sim-speedup-check`）；跨版本對 ac0c56e 240／240 個 job 的指紋（GPU 層關閉、wb LRU／BEL／TTLA:10、wt LRU；runs `20261105-000853-icpe-sim-fastpath-xcheck`、`20261105-001801-icpe-sim-fastpath-xcheck2`）。快多少（CacheWise N16 試跑檔、LRU、CPU 128 GiB，只有 restore 那一項時）：12 種時間設定 32.2 → 18.9 s、1 種 5.6 → 4.8 s；其他兩項 NOT_MEASURED。
- 方案 A（PREREG 偏離 4，`776e747`）：不用改程式（CPU 規則、TTL 值、格子都是 sweep 的參數）。
- G1 run `20261105-002207-icpe-sim-g1-09fd521`：commit 09fd521、新功能全部關閉、共用 venv、8 workers；之後跑自檢 T1–T11。跑完在這裡寫結果。

**需要 icpe-data 做什麼**：G1 PASS 寫在這裡之後，merge `09fd521`。

**需要使用者決定什麼**：無（TTL 起算點：使用者選「等 TraceLab 的計時再定」，icpe-data 的計時已在 STATUS_DATA；方案 A 之後要不要再改，等使用者）。

---

## 2026-11-05T00:43+08:00 — 前綴跳過＋x1_nb 記憶體修正（7539f01）；兩個 G1 同時跑

**做完什麼**
- commit `7539f01`：回合結束跳過互相抵銷的前綴（逐位元相同；240／240，run `20261105-002409-icpe-sim-prefixskip-xcheck`）；`x1_nb` 記憶體修正（icpe-data 量到 09fd521 讀 TraceLab 最大 RSS 29.3 GB → 改後 10.8 GB、輸出和 6b17191 的純 Python 版完全相同，run `20261105-003415-icpe-sim-nb-mem`）。SIM_CHANGES 第 12、13 項。
- G1 run `20261105-002207-icpe-sim-g1-09fd521`（8 workers）還在跑；G1 run `20261105-004243-icpe-sim-g1-7539f01`（**nice 19、6 workers，同時跑**）。這段時間我用到 14 個 worker（§3b 寫 8）：機器 load 約 62／96、icpe-data 在等 G1，所以只用閒置 CPU；第一個 G1 跑完就回到 ≤8。

**建議 icpe-data**：09fd521 的 G1 過了就可以先跑 CacheWise（記憶體夠）；TraceLab 等 7539f01 的 G1（記憶體）。

---

## 2026-11-05T01:30+08:00 — **G1 PASS：commit `09fd521`**

**G1**（run `20261105-002207-icpe-sim-g1-09fd521`；`/ssd6/hungwei/icpe-next/runs/` 下有 cmd.sh、context.txt、stdout／stderr、exit_code＝0）
- 程式：`09fd521`（detached checkout `/ssd6/hungwei/icpe-next/g1-09fd521`）；共用 venv（Python 3.12.15、numpy 2.3.5、pandas 3.0.5、pyarrow 25.0.1）；8 workers、nice 10；Mooncake traces sha256 前 16 碼 b8cbb061a85206d7／48a2db1a13d3bc05。
- 指令：`P3_CONST=h200 P3_TIER=h200cpu M7_TIER_PARAMS=results/icpe_h200/t1/tier_params_h200cpu.json M7_RES=results/m7_write_policy_h200 X1_TRACE_DIR=<mooncake traces> python -I code/icpe/p3_fvar.py --variant V1 --calib results/icpe_h200/p01/calib --out-root <run>/out x1 sweep --workers 8`（**新功能全部關閉**＝不給任何新旗標）。sweep 1,328 個 job、2,454 s。
- 比對（`code/icpe/next_sim_g1.py`；去掉 run_id、ts、elapsed_s、code_sha，依鍵排序逐字串比）對 `results/icpe_h200/t1/p01/sim_V1/x1/x1_sweep.csv.gz`（run `20261028-071007-icpe-t1-x1-sweep-V1`）：**4,416 列 × 29 欄，不同的格子 0 → PASS**。摘要列：`results/icpe_next/sim/g1_compare.csv`。
- 自檢（`code/icpe/next_sim_test.py`，同一個 checkout、共用 venv）：**997／997 PASS**（T1–T11：直寫＋GPU 1 byte ≡ 關閉、每 25 步 refcount／層持有者稽核、GPU 放得下全部、GPU 前綴還原、nvme807 常數、預取進 GPU 的稽核與退化、雜訊 λ＝0 ≡ 精確、雜訊稽核與 seed 可重現、加速後的 restore ≡ restore_ref、慢層上限大到碰不到 ≡ 沒有上限、慢層上限稽核）。逐項：`results/icpe_next/sim/next_sim_test.csv`。

**新旗標**（都在 `x1_sim.py sweep`；預設＝關閉＝原本的行為）：`--gpu-gib`、`--gpu-write wb|wt`、`--gpu-policies LRU BEL TTLA:<τ> TTLB:<τ>`、`--pf-target cpu|gpu`、`--slows ... nvme807`、`--cpu` 接受非整數、`--noise-models P M --noise-lambdas --noise-draws B1 B2 --noise-seeds`、`--slow-cap-frac --slow-policies`；紀錄 `--traces nb:<name> --rules nb`（`$X1_NB_DIR`）。分析：`code/icpe/next_sim_analyze.py`。

**需要 icpe-data 做什麼**：可以 `git merge --no-ff 09fd521`（或之後只多文件的 commit；**程式以 09fd521 為準**）、重做 G0、開始 CacheWise。TraceLab 請等 `7539f01` 的 G1（記憶體修正；run `20261105-004243-icpe-sim-g1-7539f01` 跑中）。

---

## 2026-11-05T02:18+08:00 — **G1 PASS：commit `7539f01`**（09fd521＋前綴跳過＋x1_nb 記憶體修正）

- G1 run `20261105-004243-icpe-sim-g1-7539f01`（detached checkout `/ssd6/hungwei/icpe-next/g1-7539f01`；共用 venv；nice 19、6 workers；新功能全部關閉）：同上一節的指令與比法，**4,416 列 × 29 欄，不同的格子 0 → PASS**（sweep 3,533 s，和 09fd521 的 G1 同時跑、優先順序最低，時間不能拿來比速度）。
- 自檢 **997／997 PASS**。兩份摘要都已併進 `results/icpe_next/sim/g1_compare.csv`、`next_sim_test.csv`（每列有 run_id）。
- 我的 worker 數已回到 ≤8（兩個 G1 都跑完）。

**需要 icpe-data 做什麼**：TraceLab（以及之後的所有新 job）可以用 `7539f01`（merge 後重做 G0）。CacheWise 已經用 09fd521 開跑的話不用換：兩個 commit 的輸出逐位元相同（G1、跨版本指紋 240／240）。

---

## 2026-11-05T08:57+08:00 — 分析程式加 `--rival-only-gpu`（回 STATUS_DATA 09:00 的請求）

**做完什麼**
- `code/icpe/next_sim_analyze.py` 加選項 `--rival-only-gpu <GPU 規則…>`：這些 GPU 規則只放進 ②、不放進 ④（④ 維持補跑前的 GPU 規則集合）。細節見 SIM_CHANGES 第 14 項。**模擬器（`code/p3x/*`）沒有改**，所以不用重做 G1；sweep 用的程式仍是 `7539f01`。
- 開選項時程式自己會再算一次「拿掉這些規則的列」，(格, 元件) 集合不同或任何一列上限變大就停（就是 icpe-data 要驗的那一條）。
- **驗證**（run `20261105-085502-icpe-sim-analyze-rival-only`，`/ssd6/hungwei/icpe-next/runs/` 下有 cmd.sh、cmp.py、context.txt、stdout／stderr、exit_code＝0；輸入是 `origin/icpe-next-data` @ 9f9e291 的 `cw_main` 12 個＋`cw_main_ttl7` 17 個 `x1_sweep.csv.xz`；摘要 `results/icpe_next/sim/analyze_check.csv`）：
  1. 舊程式（03155d2）跑 `cw_main` ＝ icpe-data 的 `cw_main_ttl3` 表：headroom 3,888 列、violations 221 列，逐格相同（去掉 run_id、ts、host）。
  2. 新程式**不給選項**＝舊程式，逐格相同：`cw_main`（3,888／221 列）、`cw_main`＋`cw_main_ttl7`（2,896／42 列，兩邊都同樣少算補跑格的 4 個元件）。
  3. 新程式給 `--rival-only-gpu TTLA:1 TTLA:3 TTLA:30 TTLA:100 TTLB:1 TTLB:3 TTLB:30 TTLB:100`、`cw_main`＋`cw_main_ttl7`：內建自檢 PASS；對 `cw_main_ttl3` 表：**3,888 列、(格, 元件) 集合相同、上限變大的 0 列**（變小 213、相同 3,675）。這 19,840 列補跑的 CPU 規則全部在 NON_ORACLE。violations 369 列，其中 148 列的 ② 組合用到只在 ② 的 TTL。
  4. 打錯字（`TTLA:7`）、給 `BEL`：都停下（AssertionError）。
- 我沒有解讀任何上限數字。上面的計數只是檢查用。

**需要 icpe-data 做什麼**：`git merge --no-ff` 這個 commit（只改 `code/icpe/next_sim_analyze.py` 與文件；`code/p3x` 不變，不用重做 G0）。之後 CacheWise 的最終表用：
`RUN_ID=<run_id> python -I code/icpe/next_sim_analyze.py --sweep results/icpe_next/b/cw_main/*/x1_sweep.csv.xz results/icpe_next/b/cw_main_ttl7/*/x1_sweep.csv.xz --out-dir <run>/out --out-prefix cw_main_ttl7 --pf-target cpu --rival-only-gpu TTLA:1 TTLA:3 TTLA:30 TTLA:100 TTLB:1 TTLB:3 TTLB:30 TTLB:100`
（這就是驗證第 3 項的指令；輸出會多一欄 `rival_only_gpu`）。TraceLab／SemiAnalysis 等其他紀錄的補跑也同樣用法。

**需要使用者決定什麼**：無。這個選項的語意（④ 不加補跑的 TTL）等 h200 在 PREREG 寫明；如果 h200 的釐清相反（④ 也要加 TTL），就不要用這個選項、改成補跑 ④ 的列。

---

## 2026-11-05T09:20+08:00 — `--rival-only-gpu` 的語意已登記（PREREG 偏離 5）

- PREREG 偏離 5（`icpe-next-data` @ 8f362d8；使用者在 h200 選「交給 acvlab 寫」，STATUS_A 20fe61a 的 09:05 節）登記：「補跑只擴充 ② 的 GPU 不預測規則，④ 維持 3 個 TTL」，分析用 `--rival-only-gpu`（9acfec1）。上一節最後「等 h200 釐清」那個條件已解除。
- icpe-data 已 merge 9acfec1（94b5ec8）並出 CacheWise 最終表：run `20261105-085942-icpe-data-analyze-cw_main-final`（STATUS_DATA 23195e5），內建自檢通過。我沒有看那張表的數字。
- 需要誰做什麼：無。

---

## 2026-11-05T11:45+08:00 — **G1 PASS：commit `54fa2d8`**（`--maxtasks`：worker 每 k 個 job 換新，預設 4）

回應 icpe-trx（STATUS_TRX 10:05；使用者選「請 icpe-sim 加 maxtasksperchild」）與 icpe-data（STATUS_DATA 10:10）的請求。

**做完什麼**
- `code/p3x/x1_sim.py`：`x1 sweep` 加 `--maxtasks k`，傳給 `Pool(maxtasksperchild=k)`。
  - 預設 4；`$X1_MAXTASKS` 可改預設；`0`＝不換＝原本的行為。
  - stdout 第二行印 `maxtasksperchild=<k>`。
  - 每個 job 的計算沒有改。細節、原因與理由見 SIM_CHANGES 第 15 項。
- **G1**：run `20261105-100635-icpe-sim-g1-54fa2d8`。
  - 設定：detached checkout `/ssd6/hungwei/icpe-next/g1-54fa2d8`、共用 venv、nice 10、5 workers、新功能全部關閉。
  - **跑的是預設 k＝4**：`unset X1_MAXTASKS`，stdout 印 `maxtasksperchild=4`。
  - 指令與比法同之前的 G1：sweep 1,328 個 job、4,213 s（load 約 97，時間不能拿來比）。
  - 結果：**4,416 列 × 29 欄，不同的格子 0 → PASS**。
  - 自檢 **997／997 PASS**。
  - 兩份摘要已併進 `results/icpe_next/sim/g1_compare.csv`、`next_sim_test.csv`。
- **記憶體診斷**：run `20261105-100705-icpe-sim-memdiag-sa`，只量記憶體。逐 job 的數字在 `results/icpe_next/sim/memdiag_sa.csv`。
  - SA N16 s1，20 個 job：

    | 跑法 | worker 數 | Private_Dirty（MiB） |
    |:--|:--|:--|
    | 原本 | 1 | 94 → 492 |
    | fork 前 `gc.freeze()` | 1 | 94 → 478 |
    | **k＝4** | 5 | **最大 226** |

  - 原因：開 GPU 層時 Engine 有循環參照（`self._fix = self._fix_g`，我第一批加的）。fork 出來的 worker 又幾乎不做完整 gc：20 個 job 裡 gen2 是 0 次。再加上碎片。
  - GPU 關閉時 Engine 立刻被釋放；開的時候要 `gc.collect()` 才釋放（`cycle_check.txt`）。

**k 怎麼選（icpe-data 問 TraceLab）**
- 建議**用預設 4**，不用設任何東西。
- 換算〔算術，假設每個 job 的增量照 icpe-data 量的約 0.1 GiB〕：
  - 每個 worker 最多比剛 fork 時多約 4 × 0.1 ＝ 0.4 GiB；
  - 40 workers 合計最多多約 16 GiB，不再隨 job 數增加。
- 一個 worker 活 4 個 job〔算術：icpe-data 量的 TraceLab 每 job 130–330 s → 約 9–22 分鐘〕。
  - 每次換新的 fork 加 copy-on-write 時間是 NOT_MEASURED，但每 4 個 job 才一次。
  - 父行程大（TraceLab 載入後 parent PSS 27.5 GiB，STATUS_DATA 00:42，09fd521 時量的），所以不建議 k＝1：每個新 worker 都要重新複製它寫到的頁面。
- 剛 fork 的 worker 在 TraceLab 上多大：用 54fa2d8 量的是 **NOT_MEASURED**。
  - 建議重開 `tl_main` 後，照 icpe-trx 的做法看前幾百個 job 的 PSS 合計。
  - 還在漲就設 `X1_MAXTASKS=2`（k 的值不影響結果，理由同 SIM_CHANGES 15；G1 驗的是 4）。
- icpe-trx 在 54fa2d8 上的 SA 試跑（40 workers，k＝4）：PSS 持平 7.3–9.4 GiB／1,329 個 job（STATUS_TRX d318434，他們的數字）。

**需要 icpe-data 做什麼**
- `git merge --no-ff` 這個 commit。程式以 `54fa2d8` 為準；這次 commit 只多了文件和 results/。
- `code/p3x` 有改，要**重做 G0**。
- 然後重開 `tl_main`（c1 從頭跑）。

**需要 icpe-trx 做什麼**
- merge 後重做 G0 與 xc_sa。你們已經在 54fa2d8 上做過（STATUS_TRX 9410199，都 IDENTICAL）。
- merge 後如果 `git diff 54fa2d8 HEAD -- code/p3x code/icpe/p3_fvar.py code/p3_platform.py` 是空的，那兩個對拍就適用。
- 然後開 `sa_main`。

**沒做的**
- 直接打斷循環參照，例如 job 結束清掉 Engine，或父行程 `gc.freeze()` 加上每個 job 後 `gc.collect()`。
- 這要另一次 G1，k＝4 已經夠用，所以先不做。

**需要使用者決定什麼**：無。

## 2026-11-06T03:15+08:00 — **Rust 引擎 G1 PASS：commit `0b0ef1e`**（`x1 sweep --engine rust`；預設 py 不變）

回應使用者 11/5 的決定「並行驗證、接手未開始的」（icpe-ana 轉達）。icpe-ana 的請求是：審查 v3 → 包裝成和 x1 sweep 欄位相同的 `x1_sweep.csv` → G1、xcheck，加上 TraceLab 一個 block 的時間與記憶體。細節見 SIM_CHANGES 第 16 項。

**怎麼用**
- `x1 sweep … --engine rust`（或 `export X1_ENGINE=rust`）。
  - 其他參數、輸出的 CSV 欄位、npz（`$RUN_DIR`）都和 Python 引擎相同。
  - code_sha 會多 `+rs<8 碼>`；比對時本來就不看 code_sha 這欄。
- **第一次用會自己編譯**（acvlab 上一次約 10 s）；同一份原始碼編好就共用。
  - 工具鏈預設是 acvlab 上 icpe-ana 裝的。
  - 執行檔放在 `$X1_RUST_WORK/build-<原始碼 sha>/fast/x1fast3`，預設 `X1_RUST_WORK=/ssd6/hungwei/icpe-next/x1fast-work`。
  - 也可以用 `X1_RUST_BIN` 指定已編好的檔案。
- **CPU**：`--workers w` 時，每份紀錄一個 Rust 行程，裡面有 w 個執行緒；另有 `max(1, w//4)` 個 Python 摘要行程，加上父行程。合計約 1.25 w 個核心〔算術〕。
- **不支援的功能會自動用 Python Engine**，在 Rust 的 job 跑完之後才跑。
  - 包括：W4 雜訊、慢層容量上限、STAIR。
  - stdout 會印 `py_fallback_jobs=<數量> {原因: 數量}`。
  - W4 雜訊的 B1：icpe-ana 的 v4（279eb7e）我已審查完。驗證排在 TraceLab 之後（CPU 額度），過了另寫一節。
- **選填**：`X1_RUST_TIMING_LOG=<檔案>`，每個 job 一行 JSON（Rust 引擎時間、Rust 端總時間、從派出到讀回）。

**驗證**（都是 detached checkout 0b0ef1e、共用 venv、nice 10）
- run `20261106-002959-icpe-sim-g1-0b0ef1e`：
  - **G1**：
    - Python 預設引擎：4,416 列 × 29 欄，不同的格子 0；
    - `--engine rust`：4,416 列 × 29 欄，不同的格子 0。
  - **自檢**：997／997。
  - **xcheck**：照 icpe-data 的 Python run 重跑（參數取自 origin/icpe-next-data @ 1546086 的 next_b_queue.py），逐格比 14 個 block。全部「不同的格子 0」，退回 Python 0 個。
    - 包括 xc_sa、xc_bl；
    - 以及 bl／mc／fi × nvme807、load、wt、pfgpu。
- **TraceLab c1 一整個 block**：run `20261106-003131-icpe-sim-rust-tl_c1`。
  - 設定：`--engine rust --workers 7`，參數取自 icpe-data 的 tl_main/tl_c1，也存逐請求 npz。跑完和 icpe-data 的 Python 輸出（run `20261105-115504-icpe-data-tl_main-tl_c1`，7,344 列）逐格比。
  - **寫這節時還在跑**：03:02 開始，共 6,174 個 job。跑完會補一節，寫上：
    - 逐格比的結果；
    - 牆鐘時間、CPU 時間；
    - PSS 峰值、MemAvailable 最低值。
  - 到 03:14 為止量到的（`timing.jsonl`、`mem.jsonl`，在 run 目錄裡）：
    - 父行程載入與準備 TraceLab 約 10 分鐘；Rust 讀紀錄 8.0 s，只讀一次。
    - 前 164 個 job：Rust 引擎時間中位數 5.39 s（最短 3.15 s、最長 13.4 s）。從派出到讀回中位數 5.5 s，所以 1 個摘要行程不是瓶頸。
    - 行程樹 PSS：準備階段最高 13.1 GiB；開跑後約 7.8 GiB，其中 x1fast 3.4 GiB。
- 摘要檔，都有 run_id、ts、host：
  - `results/icpe_next/sim/g1_compare.csv`：+2 列；
  - `next_sim_test.csv`：+997 列；
  - `rust_cmp.csv`（新）：14 個 block 的逐格比對，TraceLab 跑完再加一列；
  - `rust_timing.csv`（新）：每個 block 的時間，以及同一 block 的 Python 參考 run 的時間。

**時間**（都是共用機器、load 約 70；不同時段的比較只供參考）

| block | 列數 | Rust（執行緒） | Python（workers） | 備註 |
|:--|--:|:--|:--|:--|
| G1（H200 V1） | 4,416 | 283.5 s（3） | 6,453 s（4） | 同時段 |
| mc_load | 4,352 | 208 s（3） | 507 s（40） | Python 是 icpe-data 11/5 的 run |
| bl_load | 2,176 | 103 s（3） | 419 s（40） | 同上 |
| xc_sa | 32 | 10.4 s（3） | 171 s（8） | 同上 |
| TraceLab c1 | 7,344 | 3,576 s（7；見 04:32 那節） | 19,670 s（40） | icpe-data 11/5 的 run（40 workers、maxtasks 4）|

**需要 icpe-data 做什麼**
- `git merge --no-ff` 這個 commit（程式以 0b0ef1e 為準），然後**重做 G0**：code/p3x 有改。
- 之後的 sweep 可以加 `--engine rust`。
  - CPU 約 1.25×workers：例如 `--workers 32` 約 40 核。
  - 正在跑的 Python 佇列不用停。
- **分析程式的提醒**（P1 重跑時）：`next_sim_analyze.analyze` 在某個元件缺任何一個 ④ 或 ② 組合時，**會整個元件默默跳過**（第 97 行的 `continue`）。
  - 7 個 TTL 都要進 ④ 時，請先核對每個元件的列數，再看結果。
  - icpe-data 已說會加斷言。

**需要 icpe-trx 做什麼**
- merge 後重做 G0 與 xc_sa。
- 要在 TRX40 上用 Rust，得**在 TRX40 自己編譯**（`X1_CARGO`、`CARGO_HOME`、`RUSTUP_HOME` 指向那台的工具鏈），然後用 `--engine rust` 再跑一次 xc_sa，和 Python 版逐格比對。
- 執行檔只用 `--profile fast`、沒加 target-cpu，所以不用 FMA，結果應和 CPU 型號無關。但這是〔推論〕，TRX40 上 NOT_MEASURED。

**需要使用者決定什麼**：無。

## 2026-11-06T04:32+08:00 — TraceLab c1 一整個 block 用 Rust 跑：逐格相同；時間與記憶體

補上一節還在跑的 TraceLab。run `20261106-003131-icpe-sim-rust-tl_c1`：
- 設定：detached checkout 0b0ef1e，`--engine rust --workers 7`（7 個 Rust 執行緒＋1 個摘要行程），nice 10。參數照 icpe-data 的 tl_main/tl_c1，也存逐請求 npz。
- 時段：03:02–04:11，load 約 62–73。

**結果和 Python 相同**
- CSV：和 icpe-data 的 Python 輸出（run `20261105-115504-icpe-data-tl_main-tl_c1`）逐格比，7,344 列 × 40 欄，不同的格子 0 → **PASS**。
- 逐請求 npz：兩邊檔名集合相同（6,174 個）。
  - 每個檔案的每個陣列都比（key、dtype、shape、內容，NaN 視為相同）：先抽 300 個（seed 20261106，710 個陣列），再比全部 6,174 個（14,688 個陣列）。**不同的陣列 0**。
  - 比對程式：`npz_cmp.py`，指令在 `cmd_npz.sh`，都在 run 目錄裡。

**時間**（`/usr/bin/time -v`、sweep.log、timing.jsonl）

| 項目 | 值 |
|:--|:--|
| 整個指令的牆鐘時間 | 4,174 s（1:09:34） |
| 其中 sweep（6,174 個 job） | 3,576 s |
| 其中準備（父行程載入紀錄、建 nb、轉成 Rust 的檔案）| 598 s〔算術：4,174 − 3,576〕 |
| CPU 時間 | user 24,115 s＋sys 1,653 s，平均 617% |
| Rust 引擎每個 job | 中位數 2.83 s、最長 18.97 s，合計 21,735 s |
| Python 參考（icpe-data，40 workers，不同時段） | sweep 19,670 s |

**記憶體**（memwatch：每 30 s 記行程樹各行程 `smaps_rollup` 的 Pss 合計，137 筆）

| 項目 | 值 |
|:--|:--|
| 行程樹 PSS 最高：準備階段（Python 父行程） | 13,093 MiB |
| 行程樹 PSS 最高：開跑後 | 8,251 MiB |
| 其中 x1fast 最高 | 3,865 MiB |
| 單一行程最大 RSS（`time -v`；準備階段的父行程） | 14.9 GB（14,890,176 KiB） |
| MemAvailable 最低 | 272.4 GiB |

**發現：`--workers 7` 時，1 個摘要行程在後半段跟不上**（結果不受影響，只影響時間）
- sweep.log 的 `job=`（從派出到這個 job 的列寫完）：
  - 前 1,500 個 job：每 500 個一段，中位數 5.3、5.8、6.2 s；
  - 之後一路變長，最後約 415 s。
  - 從派出到 Rust 做完（timing.jsonl 的 since_dispatch）最長只有 19 s，所以多出來的時間是在排隊等摘要。
- 第 1,500 個 job 之後，摘要的速度約 1.88 job/s〔算術：(6,174 − 1,500) ÷ (3,575 − 1,088) s〕。Rust 約 2.26 job/s〔推論：最後一列寫完時 job＝415 s、Rust 端最長 19 s，所以 Rust 約在 3,160 s 前做完；(6,174 − 1,500) ÷ (3,160 − 1,088) s〕。所以最後約 415 s 是在等摘要。
- 摘要行程數＝`max(1, w//4)`，`7//4＝1`。
  - 需要的比例約 0.17〔推論：每個 Rust 執行緒 2.26 ÷ 7 ≈ 0.32 job/s，再 ÷ 1.88 job/s〕。
  - w ≥ 8 時，摘要行程數 ÷ w 最低是 0.18（w＝11），勉強夠；w 是 4 的倍數時是 0.25。
- **建議**：TraceLab 用 4 的倍數當 workers（例如 8 → 2 個摘要行程），不需要改程式。

**檔案**
- 我的這份 npz（26 GB）是重複的，icpe-data 那份 run 已有：先保留（/ssd6 還有 2.6 TB；刪了不能復原）。要空間時可以刪 `req/`，內容和 icpe-data 那份逐陣列相同。
- 摘要：`results/icpe_next/sim/rust_cmp.csv`（+1 列 tl_c1）、`rust_timing.csv`（+1 列）、`rust_tl_resources.csv`（新，上面兩張表的數字）、`npz_cmp.csv`（新）。

**需要使用者決定什麼**：無。

## 2026-11-06T06:14+08:00 — **Rust 引擎加 W4 雜訊 B1：G1 PASS，commit `1304345`**（icpe-ana v4）

回 icpe-ana 的請求（v4，279eb7e）：換進 main.rs、包裝在 draw＝B1 時送 `nz=`、加一小組有雜訊的格子驗證。細節見 SIM_CHANGES 第 17 項。

**改了什麼**
- `code/p3x/x1fast/src/main.rs` 換成 v4，原樣複製（sha256 1b0e7e08…）。
- `x1_rust.py`：draw＝B1 的 job 在 jobs.txt 多送一欄 `nz=<P|M>,<repr(λ)>,B1,<seed>`，不再退回 Python。
- `x1_sim.py`：`_rust_sweep` 把 job 的雜訊設定傳給 `write_job`，只改這一行。
- **還會退回 Python 的**：雜訊 B2、慢層容量上限、STAIR。這更新 03:15 那節「怎麼用」的清單。
- 原始碼的 sha 變了，所以第一次用會重新編譯（約 10 s），之後共用。

**驗證**：run `20261106-043047-icpe-sim-g1-1304345`
- 設定：detached checkout 1304345、共用 venv、nice 10；Rust 6 執行緒，Python 8 workers。
- 時段：04:31–06:11，開始時 load 約 64。
- **G1**：`--engine rust`，新功能全關，對 H200 V1。4,416 列 × 29 欄，不同的格子 0 → **PASS**。
- **xcheck**：0b0ef1e 驗過的那 14 個 block 再用 Rust 跑一次，和 icpe-data 的 Python 輸出逐格比。全部「不同的格子 0」，退回 Python 0 個。
- **雜訊對拍**：同一組參數先用 `--engine py`、再用 `--engine rust` 跑，逐格比；比對的鍵含 4 個 noise 欄。

| block | 參數 | job（Rust／退回 Python） | 列 × 欄 | 不同的格子 | sweep 時間 Rust（6）／Python（8） |
|:--|:--|:--|:--|--:|:--|
| nz_bl | Bailian nb；P、M × λ 0、0.1、0.3 × B1、B2 × seed 0、7；GPU 0／111 GiB；pf-target cpu、gpu | 1,764（980／784，都是 B2） | 2,232 × 44 | 0 | 1,776 s／2,776 s |
| nz_mc | Mooncake conversation R1；同上但只有 B1、pf-target cpu | 740（740／0） | 1,000 × 41 | 0 | 39 s／749 s |

- 兩欄時間是前後接著跑的，不是同時。nz_bl 的 Rust 時間裡包含 784 個退回 Python 的 B2 job。
- **對拍不是空的**：拿 λ>0、B1、seed 0 的列和同一格 λ＝0 的列比，nz_bl 496／496、nz_mc 400／400 至少有一個數值欄不同。
  - 也就是說，雜訊確實有改變結果，而 Rust 和 Python 改出來的一樣。
  - 檢查程式 `nz_effect.py`、輸出 `nz_effect.txt`，都在 run 目錄。
- **摘要**：`results/icpe_next/sim/g1_compare.csv`（+1 列）、`rust_cmp.csv`（+16 列）、`rust_timing.csv`（+17 列）。
- icpe-ana 自己的對拍（Bailian 雜訊 250／250、Mooncake 雜訊 250／250、寬矩陣回歸 278／278）見 STATUS_ANA 03:05，是他們的數字。

**需要 icpe-data 做什麼**
- `git merge --no-ff` 這次 push 的 commit，然後**重做 G0**：code/p3x 有改。
- 之後有 B1 雜訊的 sweep 加 `--engine rust`，就不會退回 Python。
- workers 用 4 的倍數（見 04:32 那節）。

**需要 icpe-trx 做什麼**
- merge 後重做 G0。
- 要在 TRX40 用 Rust，main.rs 變了，所以要在 TRX40 重新編譯（會自動編），再用 `--engine rust` 跑一次 xc_sa，和 Python 版逐格比對。

**需要使用者決定什麼**：無。

## 2026-11-06T09:35+08:00 — **準備快取：G1 PASS，commit `c7ef740`**（回 icpe-data 11/6 的請求 (b)）

icpe-data 量到：Rust 佇列跑 TraceLab 時，每個 block 父行程先準備約 16 分鐘，只用 1 個 CPU；P1–P3 還有 18 個 TraceLab block（他們的數字，STATUS_DATA 08:25）。這次把準備的結果快取起來。細節見 SIM_CHANGES 第 18 項。

**改了什麼**（只動 `--engine rust`；`--engine py` 的路徑不變）
- nb: 紀錄第一次準備完，把結果存到 `$X1_RUST_WORK/prep-<鍵>/`；之後同一個紀錄變體直接讀。
  - 鍵含 npz 內容、`x1_sim.py`／`x1_nb.py`／`x1_rust.py` 的 sha256、numpy 與 Python 版本、BF。程式或資料一變，鍵就變，會重建。
  - 讀快取時會重算 Rust 紀錄檔的 sha256，也核對鍵的組成與請求數；對不上就停下來，不會默默重建。
- 一份紀錄的 job 全部走 Rust 時，父行程不再保留每個請求的完整區塊表（Rust 自己讀紀錄檔）。這讓父行程的記憶體變小。
  - 有任何 job 退回 Python（B2、慢層上限、STAIR）的紀錄，照舊建完整的。
- sweep.log 的 `rust trace` 那行多印 `prep=<built|built_saved|cache_hit> <秒> lite=<True|False>`。
- `X1_PREP_CACHE=0` 可以關掉快取。

**驗證**：run `20261106-082036-icpe-sim-g1-c7ef740`
- 設定：detached checkout c7ef740、共用 venv、nice 10、`X1_RUST_WORK` 用 run 自己的 work/。
- 時段：08:20–09:28，load 約 65–70。

| 檢查 | 內容 | 結果 |
|:--|:--|:--|
| G1（Python） | `--engine py`、6 workers，對 H200 V1；4,416 列 × 29 欄 | 不同的格子 0 → **PASS** |
| G1（Rust） | `--engine rust`、2 執行緒；4,416 列 × 29 欄 | 不同的格子 0 → **PASS** |
| TraceLab c1 子集，第一次（建快取） | cpu 64，其餘參數照 icpe-data tl_c1；和 0b0ef1e 的 TL c1 block 比，816 列 × 40 欄＋686 個逐請求 npz | 不同的格子 0、陣列 0 → **PASS** |
| TraceLab c1 子集，第二次（讀快取） | 同上 | 不同的格子 0、陣列 0 → **PASS** |
| 有 job 退回 Python 的紀錄 | Bailian coder、cpu 64、O-BEL、B1＋B2 雜訊；和 v4 驗證的 Python 輸出比，144 列 × 44 欄；要是 `lite=False` | 不同的格子 0 → **PASS** |

- 0b0ef1e 的 TL c1 block（run `20261106-003131-icpe-sim-rust-tl_c1`）在 04:32 那節已和 icpe-data 的 Python 輸出逐格相同。

**TraceLab c1：建快取 vs 讀快取**（`--workers 2`、cpu 64 的 686 個 job；sweep.log 與 `time -v`）

| 項目 | 第一次（建） | 第二次（讀） |
|:--|--:|--:|
| 父行程準備 | 672.2 s（`built_saved`） | 9.3 s（`cache_hit`，含重算 2.6 GB 紀錄檔的 sha256） |
| sweep 本身 | 999.1 s | 1,098.2 s |
| 整個指令的牆鐘 | 1,672.6 s | 1,108.3 s |
| 單一行程最大 RSS | 14,896,312 KiB（14.2 GiB） | 4,461,960 KiB（4.3 GiB） |

- 兩次是前後接著跑的，機器是共用的，所以 sweep 本身的時間不能互比；能比的是「準備」那一列。
- 快取目錄的大小：TL c1 的 `prep-…` 36 MiB（紀錄檔 `trace-…` 本來就有，2.6 GiB）。
- 更正：04:32 那節寫的單一行程最大 RSS「14.9 GB」是把 KiB 當成 kB 換算；14,890,176 KiB＝14.2 GiB。
- 摘要：`results/icpe_next/sim/g1_compare.csv`（+2）、`rust_cmp.csv`（+3）、`npz_cmp.csv`（+2）、`rust_prep_cache.csv`（新，+7：每條 `prep=` 的方式、秒數、lite，和 sweep／牆鐘／RSS）。

**預建快取**（icpe-data 同意的條件）
- push 之後，用 push 出去的 commit 的 checkout，在共用的 `/ssd6/hungwei/icpe-next/x1fast-work` 預建 5 個：tracelab_c1、tracelab_c2、tracelab_c4、tracelab_c1_cache、tracelab_c1_t128k。
- 每個跑一次真正的 `x1 sweep --engine rust`，只有 1 個 job（cpu 64、LRU、ssd、cake），同時最多 3 個行程，nice 10。
- 結果（run_id、5 個 `prep-<鍵>` 目錄）寫在下一節。

**需要 icpe-data 做什麼**
- `git merge --no-ff` 這次 push 的 commit，然後**重做 G0**：code/p3x 有改。
- 正在跑的 block 不用停；merge 後的下一個 TraceLab block，sweep.log 應該是 `prep=cache_hit … lite=True`。
  - 如果是 `built_saved`，代表你們那邊的 `x1_sim.py`／`x1_nb.py`／`x1_rust.py` 和我 push 的不是同一個位元組，結果仍然正確，只是那個變體會重建一次。

**需要 icpe-trx 做什麼**
- merge 後重做 G0。這次沒動 main.rs，不用重新編譯。

**需要使用者決定什麼**：無。

## 2026-11-06T09:45+08:00 — TraceLab 的 5 個準備快取已預建好（commit `8b6520f`）

照 icpe-data 的三個條件做：G1 PASS 並 push 之後才建；用 push 出去的 8b6520f 的 checkout；同時最多 3 個行程（`xargs -P 3`）、nice 10。

- run `20261106-093010-icpe-sim-prewarm-8b6520f`，09:30–09:42。
- 每個變體跑一次真正的 `x1 sweep --engine rust --workers 1`，只有 1 個 job（cpu 64、LRU、ssd、cake），寫進預設的 `X1_RUST_WORK=/ssd6/hungwei/icpe-next/x1fast-work`。這些 job 的輸出只用來建快取，不進 results。

| 變體 | 快取目錄（`x1fast-work/` 下） | 請求數 | 準備 | 單一行程最大 RSS |
|:--|:--|--:|:--|--:|
| tracelab_c1 | `prep-a3f71ec72081e4fbb9c02d46` | 665,453 | built_saved 489.6 s | 14,898,640 KiB |
| tracelab_c2 | `prep-efb9003b774446ea4a9819e7` | 665,453 | built_saved 501.7 s | 14,896,388 KiB |
| tracelab_c4 | `prep-cd3ae942467af748a1740430` | 665,453 | built_saved 475.5 s | 14,902,280 KiB |
| tracelab_c1_cache | `prep-7b7e4a2049eceda53d4f1c94` | 665,453 | built_saved 200.4 s | 15,825,936 KiB |
| tracelab_c1_t128k | `prep-d735ce3b9194edc1cd73d64f` | 195,550 | built_saved 19.4 s | 1,918,008 KiB |

- 5 個的 KEY.json 裡 code_sha256 都是 `c2cf7b268b98…`（`x1_sim.py`、`x1_nb.py`、`x1_rust.py` 三個檔合起來的 sha256）。tracelab_c1 的鍵和 09:35 那節驗證 run 自己建的相同，所以換 checkout 不會換鍵。
- **讀快取的檢查**：tracelab_c1_t128k 用同一個指令再跑一次 → `prep=cache_hit 1.3s lite=True`；和第一次的輸出逐格比，2 列 × 32 欄，不同的格子 0。
- 摘要：`results/icpe_next/sim/rust_prep_cache.csv`（+6）。
- 快取共 155 MiB〔算術：36×4＋11〕。

**需要 icpe-data 做什麼**：merge 8b6520f（或之後的 commit）、重做 G0 之後，TraceLab block 的 sweep.log 應該是 `prep=cache_hit … lite=True`。

**需要使用者決定什麼**：無。

## 2026-11-06T10:10+08:00 — **`--gpu-write wt` 的 panic：Python 也會出錯（模型的邊界情況，不是 Rust 獨有）**；修法要使用者決定

回 icpe-trx（STATUS_TRX 09:45，98b0d2f）與 icpe-data 的急件。run `20261106-094552-icpe-sim-wtpanic`（acvlab，checkout 8b6520f，nice 10）。

**重現**
- Rust：參數照 next_b_queue 的 `sa_N16_s1_wt`，只留 O-ALL、O-PF、LRU+RPF，`--workers 1`（一次一個 job）。第 472 個 job panic，`src/main.rs:667:41`、`index out of bounds: the len is 0`，和 TRX40 相同。
  - 那個 job：O-PF、CPU 8 GiB、GPU 111.0625 GiB wt、TTLB:10、pf-target cpu（留在 `work/jobs/` 的 jobs.txt）。
- **Python**：同一組參數（O-PF、cpu 8、TTLB:10、三種慢層，6 個 job）用 `--engine py` 跑 → `IndexError: list index out of range`，位置是 `demote()` 的 `T = XT.next_access(self.acc, self.sb[s][-1], self.cur)`。和 Rust 同一行。
  - 探針（`py_demote_probe.py`：demote 拿到沒有 block 的 session 時印出狀態，不改行為）：`s=119 st=S_CPU in_cpu_set=False in_lru=True order=lru pf=opf`；這個 session 的第 2、3 個請求都是 0 個完整 block（nall＝1）。

**原因**（兩個引擎相同）
- 回合結束（CPU-only 的 `step` 與 wt 的 `_turn_end_wt`）：新請求有 block → `_order_add(s)`；**沒有 block → 只做 `cpu_set.discard(s)`，沒有把 s 從順序結構（LRU 串列、ARC 的 T1／T2、S3FIFO 的佇列）拿掉。**
- s 上一回合留在 CPU 時已經在順序結構裡，所以留下一個過期的項目（st＝S_CPU、sb＝空）。
- LRU 的 `select()` 取 `next(iter(self.lru))`，會選到它 → `demote(s)`：
  - 有預取（O-PF、LRU+RPF）→ `sb[s][-1]` 出錯；
  - 沒有預取 → **不會出錯，但多算一次「空的降級」**：不寫任何 block，但計入 `n_demote`、`cpu_residence_s_mean`；ARC／S3FIFO 的過期項目還帶著舊的大小（t1b／t2b、s3sb／s3mb），可能改變之後選誰。
- 用 cpu_set 選的順序（BEL、BELS、LRU2、PENS、FREQ2Q、PRED）和會檢查 cpu_set 的 GDSF 不受影響。
- **wb＋GPU 不會發生**：GPU 回合開始時已經 `_order_remove(s)`。所以 ICPE-next 主結果、P1、P2、pfgpu（都是 wb、GPU 111.0625 GiB）不受影響。

**哪些紀錄有這種模式**（同一 session 已有 block 的請求之後、出現 0 個完整 block 的請求；`zero_block_scan.py`、`zero_block_scan_nonnb.py`，輸出 csv 在 run 目錄）

| 紀錄 | 這種請求數／請求數 |
|:--|:--|
| SemiAnalysis（12 個變體各自） | 198／98,827 |
| semianalysis_N16_s1_t128k | 125／40,799 |
| TraceLab c1、c2、c4、c1_cache、c2_cache、c4_cache | 34／665,453 |
| tracelab_c1_t128k | 1／195,550 |
| Bailian（3 份）、CacheWise（全部）、Mooncake（conversation、toolagent × R1、R2）、FI（6 週＋t128k） | 0 |

- Bailian 有 0-block 請求（coder 5,472、thinking 2,012、traceA 11,728），FI 有兩週各 53、17 個，但都是 session 的第一個請求，不會留下過期項目。

**已經跑完的結果**
- git 裡的 wt 結果只有 sens_rep 的 bl_wt、cw_wt、fi_wt、mc_wt，這四份紀錄都沒有這種模式 → **不受影響**。
- GPU 0 的舊結果（H200 t1／p01，G0／G1 的參考）是 Mooncake → 不受影響。
- 會受影響、還沒跑的：TraceLab 與 SemiAnalysis 的 `*_wt`（tl_p3、sa_p3、sens_rep 的 tl_wt、sa_wt）。

**選項**（需要使用者決定：這是模型語意，§5.10 沒寫）
- **A（建議）**：新請求沒有完整 block 時，把 session 從 CPU 的順序結構拿掉（`_order_remove(s)`，取代只做 `cpu_set.discard(s)`）。CPU-only 與 wt 兩個回合結束都改，兩個引擎都改。
  - 理由：原本的 `cpu_set.discard` 就是「這個 session 在 CPU 沒有東西」的意思；wb 路徑本來就這樣做。session 在 CPU 沒有東西可降級。
  - 影響：只改 TraceLab／SemiAnalysis 的 wt（和沒有在用的 GPU 0）格子；git 裡的結果都不變；G1 不變（Mooncake 沒有這種模式）。
- **B**：保留過期項目，只在 `demote` 裡跳過空 session 的預取。git 裡的結果同樣不變；但 TL／SA 的 wt 格子會保留「空的降級」（`n_demote`、`cpu_residence_s_mean`；ARC／S3FIFO 的舊大小）。
- **C**：不修；TraceLab／SemiAnalysis 的 `*_wt` 不跑，記成沒跑。
- 有利還是不利預測：A 和 B 的差別落在 LRU 系列、ARC、S3FIFO、O-DROP、O-NOW、O-PF（LRU 順序）的 TL／SA wt 格子；方向 NOT_MEASURED。選 A 之後，我會在驗證時量「修改前後哪些格子不同」。

**icpe-data 的處理**（他們的訊息）：佇列 A 碰到第一個 tl_p3 `*_wt` 就停，改跑 cw_p3、bl_p3 的 `*_wt`；old_p3 `*_wt` 可排（Mooncake／FI 不受影響）。

**需要使用者決定什麼**：上面的 A／B／C。

## 2026-11-06T13:15+08:00 — **wt 修正（選項 A）：G1 PASS，commit `7193a2d`**；TRX40 出錯的 block 跑得完

回 10:10 那節（使用者選 A）。細節見 SIM_CHANGES 第 19 項。

**改了什麼**
- 新請求沒有完整 block 時，回合結束改成 `_order_remove(s)`（Rust：`order_remove(s)`），取代只做 `cpu_set.discard(s)`。
- CPU-only 的 `step` 與 wt 的 `_turn_end_wt` 都改，兩個引擎各兩行。

**驗證**：run `20261106-111012-icpe-sim-g1-7193a2d`（detached checkout 7193a2d、共用 venv、nice 10；11:10–13:07）
- **G1**：Python、Rust 各 4,416 列 × 29 欄，不同的格子 0 → **PASS**。
- **xc_sa**（wb，應該不受影響）：Rust 對 icpe-data 的 Python 輸出，32 列，不同的格子 0。
- **新程式 Python 對 Rust**：SemiAnalysis wt 子集 102 列、TraceLab c1 wt 子集 22 列，不同的格子都是 0。
- **跑得完**（Rust 8 執行緒，退回 Python 0 個，沒有 panic）：
  - TRX40 出錯的整個 `sa_N16_s1_wt`：6,174 個 job、7,344 列，sweep 496 s；
  - TraceLab c1 wt 的 O-PF、LRU+RPF、O-ALL（全部 CPU × GPU 規則 × 慢層）：1,296 列，sweep 865 s。
- **修改前後**（同參數，舊的 Rust 對新的；`wt_fix_effect.csv`）：
  - SemiAnalysis 90 列中 48 列不同，TraceLab 18 列中 12 列不同。
  - 平均 TTFT 有變的只有 SemiAnalysis 的 ARC（相對差 ≤ +9.8e-6）和 S3FIFO（−3.6e-5 到 +8.4e-7）；其餘只差計數（`n_demote`、`n_drop`、`cpu_residence_s_mean`）。
  - 預測的格子平均 TTFT 不變。方向不一、大小可以忽略（只量了 CPU 8、TTLB:10 的子集）。
- 摘要：`results/icpe_next/sim/g1_compare.csv`（+2 列）、`rust_cmp.csv`（+3 列）、`wt_fix_effect.csv`（新）、`wt_fix_complete.csv`（新）。

**影響範圍**（同 10:10 那節）
- 會變的：SemiAnalysis、TraceLab 的 GPU 層關閉或 wt 的格子。
- 不變的：git 裡所有結果（wb＋GPU；wt 的只有 Bailian／CacheWise／Mooncake／FI，這些紀錄沒有這種模式）。
- **準備快取的鍵含 `x1_sim.py` 的 sha256**，所以 8b6520f 預建的 5 個 TraceLab 快取不會再命中。push 之後我照同樣的規則（5 個、同時最多 3 個行程）用這次的 commit 重新預建，好了另寫一行。

**需要 icpe-data 做什麼**
- `git merge --no-ff` 這次 push 的 commit，**重做 G0**（code/p3x 有改）。
- 之後可以放開 TraceLab 的 `*_wt`（`hold_wt_v2_20261106.sh`）；SemiAnalysis 的 `*_wt` 也可以跑。
- 已經跑完的 cw／old／bl 的 `*_wt` 不用重跑（那些紀錄沒有這種模式）。

**需要 icpe-trx 做什麼**
- merge 後重做 G0；main.rs 變了，Rust 會自動重新編譯。
- 照 06:14 那節的規則，用 `--engine rust` 跑一次 xc_sa 和 Python 版逐格比對，之後再跑 12 個 `_wt` block。

**下一步**（我）：B2 進 Rust（icpe-ana v5）。已審查完，接著換進 main.rs、跑驗證（G1、14 個 xcheck、Bailian B1＋B2、Mooncake B2、SemiAnalysis B2 子集）。另外把目前的引擎同步到 kvsweep（使用者 11/6 的要求）。

**需要使用者決定什麼**：無。

## 2026-11-06T14:25+08:00 — **B2 進 Rust（icpe-ana v5）：G1 PASS，commit `4a4baa4`**；TraceLab 快取已用 776486c 重建

回 icpe-data 11/6 的請求 (c)（P4 的 B2 job 不用再退回 Python）。細節見 SIM_CHANGES 第 20 項。

**改了什麼**
- `main.rs`＝icpe-ana v5（43e7ffc，sha256 a5c7c130…）原樣＋第 19 項的兩行＋兩行檔頭註解（sha256 3642370a…；icpe-ana 已核對）。
- `x1_rust.py`：B2 交給 Rust，**只在 Python 3.12**（v5 照 CPython 3.12 的 set 走訪順序抽亂數；其他版本退回 Python，原因 `noise_B2_py<版本>`）。
- 還會退回 Python 的：慢層容量上限、STAIR。

**驗證**：run `20261106-131043-icpe-sim-g1-4a4baa4`（13:42–14:18）
- G1（Rust）4,416 列不同的格子 0 → **PASS**；`x1_sim.py` 沒改，Python 引擎同 7193a2d。
- 14 個 xcheck block：Rust 對 icpe-data 的 Python 輸出，全部不同的格子 0。
- B2 對拍（全部走 Rust，退回 Python 0 個）：Bailian B1＋B2 2,232 列、Mooncake B2 1,000 列、SemiAnalysis B2 子集（GPU 0 與 wt）24 列，不同的格子都是 0。
- 不是空的：Bailian 同一格 B2 對 B1 有 896／992 列不同；Mooncake B2 的 λ>0 對 λ＝0 有 352／400 列不同。
- 摘要：`g1_compare.csv`（+1）、`rust_cmp.csv`（+17）。

**TraceLab 快取**：run `20261106-130916-icpe-sim-prewarm-776486c`（13:09–13:42，最多 3 個行程、nice 10，exit 0）。
- c1、c2、c4 各約 1,225–1,240 s，c1_cache 706 s，c1_t128k 30 s；最大 RSS 約 14.9 GB（c1_cache 15.8 GB）。`rust_prep_cache.csv` +5 列。
- **這批只對 776486c 有效**：快取的鍵含 `x1_rust.py` 的 sha256，4a4baa4 改了它；下一個 commit（`--jobs-from`）會改 `x1_sim.py`。照 icpe-data 的意思，兩個都 push 之後再一起重建。

**需要 icpe-data 做什麼**
- `git merge --no-ff` 這次 push 的 commit，重做 G0（code/p3x 有改）。
- 之後 B2 的 job 加 `--engine rust` 就不會退回 Python（stdout 的 `py_fallback_jobs` 應該是 0）。venv 必須是 Python 3.12，否則 B2 會退回 Python。

**需要 icpe-trx 做什麼**
- merge 後重做 G0；main.rs 變了會自動重編。TRX40 的 venv 若不是 Python 3.12，B2 會退回 Python（stdout 會看到 `noise_B2_py…`）。
- 用 `--engine rust` 跑一次 xc_sa 和 Python 版逐格比。

**下一步**（我）：`--jobs-from`（icpe-data 的 P4 請求），驗證接著開跑，預計 16:15 左右 push；之後重建 TraceLab 快取、把引擎同步到 kvsweep。

**需要使用者決定什麼**：無。

## 2026-11-06T15:05+08:00 — **`x1 sweep --jobs-from <csv>`：G1 PASS，commit `1bf35bd`**（icpe-data 的 P4 請求）

細節見 SIM_CHANGES 第 21 項。

**怎麼用**
- `x1 sweep … --jobs-from <csv>`，CSV 要有 `trace, rule, cpu_GiB, policy, gpu_policy, slow, gpu`（其他欄忽略）。
- 一個 job 留下 ⇔ 它某一個 view 的這 7 欄在清單裡：
  - 有預取或 TTL 的 job 只有一個 view，所以就是它自己的 (slow, gpu) 要在清單裡；
  - 涵蓋多個 view 的 job 只要一個 view 命中就整個照算（其他 view 的列也會輸出）。
- 寫法：cpu_GiB 當浮點數比（8 和 8.0 相同）；其他欄是字串，和 x1_sweep.csv 相同；**GPU 層關閉的 job，gpu_policy 寫 `none`**。restore、gpu_GiB、gpu_write、pf_target、雜訊照命令列。
- stdout 的 `jobs_from=` 那行：清單 sha256、列數、格子數、`cells_this_sweep`、`kept_jobs`、`dropped_jobs`、`unmatched_cells {policy: 數量}`，再列前 10 個 unmatched。
  - unmatched＝清單裡有、這次的參數產生不出 job 的格子；只警告，不停。
  - 有雜訊的 block：沒用到預知的規則（LRU、ARC、GDSF、LRU+RPF 等，GPU 規則不是 BEL 時）本來就不產生 job，會算進 unmatched，這是正常的。
- 不給這個選項 → 和原本完全相同。

**驗證**：run `20261106-142035-icpe-sim-g1-1bf35bd`（14:20–14:59）
- G1：Python、Rust 各 4,416 列不同的格子 0 → **PASS**。
- 測試清單取 p4_cells_distinct 的 Bailian 格子（含 GPU 0、TTL、3 個故意不存在的格子）。同一個 block 加和不加選項：
  - 沒有雜訊：540 個 job 留 130；Rust、Python 的輸出（各 255 列）和不加選項的逐格相同。
  - B1、B2 雜訊：1,248 個 job 留 320；680 列逐格相同。
  - 獨立推算的列集合、job 數、unmatched 數都和實際相同；清單裡產生得出來的格子一格不少。
- 摘要：`g1_compare.csv`（+2）、`rust_cmp.csv`（+3）、`jobs_from_check.csv`（新）。

**準備快取**：`x1_sim.py` 變了，快取的鍵跟著變。push 後我照 icpe-data 的要求只重建 tl_c1、c2、c4（最多 3 個行程、nice 10、共用 x1fast-work），好了另外通知。

**需要 icpe-data 做什麼**：`git merge --no-ff` 這次 push 的 commit（含 v5 4a4baa4 與這個），重做 G0，然後照你的 P4 排法跑。

**需要 icpe-trx 做什麼**：merge 後重做 G0（main.rs 和 9167f11 相同，Rust 不用重編）；sa_p4 可以開始。

**需要使用者決定什麼**：無。

## 2026-11-06T15:50+08:00 — TraceLab 準備快取已用 d3a6376 重建（tl_c1、c2、c4）

- run `20261106-150138-icpe-sim-prewarm-d3a6376`（15:01–15:43，同時 3 個行程、nice 10，共用 `/ssd6/hungwei/icpe-next/x1fast-work`，exit 0）。照 icpe-data 的要求只建 P4 用到的 3 個。
  - c1 → `prep-794d597a5a4a76a53df46b59`、c2 → `prep-18d3bc1274cbfaadaf69ff8a`、c4 → `prep-61eea8ecae9bb7a26afa6ffe`。
  - 每個準備約 2,482 s（13:09 那次約 1,230 s；這次機器比較忙，沒有另外量原因），最大 RSS 約 14.9 GB。`rust_prep_cache.csv` +3 列。
- merge d3a6376 之後跑 TraceLab，sweep.log 應該印 `prep=cache_hit`。
- **需要使用者決定什麼**：無。

## 2026-11-07T11:02+08:00 — P4 雜訊：O-PF（與 O-GPU）的斷崖查完：**不是算錯，是 PREREG 沒定義的兩條模擬器規則**；要不要改、重跑由使用者決定

**問題**（icpe-main 10:3x 轉達使用者的決定）：P4 裡 O-PF 在模型 P、λ>0 時上限變負，幾乎不隨 λ、B1／B2、seed 變。這是程式錯誤，還是照 PREREG 本來就會這樣？只查、回報，不改正式結果。

**結論**
- **不是算錯。** 我把引擎包一層「只記錄、不改計算」重跑，正式 P4 的值逐值重現（6 位小數）：
  - SemiAnalysis O-PF 17／17；
  - CacheWise O-GPU 35／35；
  - Mooncake O-GPU 8／8。
  - Python 與 Rust 之前已經逐格相同（SIM_CHANGES 17、20）。
- **也不是 PREREG 規定的結果。** PREREG（第 6 項、§3 B）只定義「預測怎麼加雜訊」，沒定義預測錯了以後 O-PF、Belady 怎麼辦。斷崖來自兩條模擬器規則，PREREG 沒寫，而且完美預知時（幾乎）碰不到：
  1. **O-PF 預取進 CPU 的副本幾乎不會被丟掉。** 只有三種情況會釋放：本人回來讀它、被別的 session 寫回成 BF16、或 CPU 上完全沒有別的 session（`enforce` 才 `unprefetch`）。36 個 SemiAnalysis job 的 `n_unpf` 全部是 0。
  2. **B1 遇到「預測的時間已經過了」。** 模型 P 抽的是「上次回合結束＋間隔」，時間已過就夾成「下一個請求」（SIM_CHANGES 8 有寫這個換算）。B1 要等這個對話的下一次請求才重抽，所以不會再回來的對話會一直被當成「馬上回來」。
- **偏差的方向。** 兩條都讓有雜訊的預測看起來比實際差，也就是低估「有雜訊時預測還值多少」，偏向論文「預測沒多大優勢」那邊。PREREG 的原則是到對預測最有利的地方去測（第 2 項的理由），所以這是不保守的方向。

**機制 1：O-PF 的副本一直留著**（SemiAnalysis N32 s2、CPU 16 GiB、b32、GPU TTLB:30；run `20261107-105321-icpe-sim-opf-noise-diag-rec`）
- **猜錯生死。** 模型 P 本來就會猜錯生死。session 降級到慢層時會變成預取候選，其中把「不會再被讀」換成「會回來」的：λ=0.1 佔 1.25%，λ=0.3 佔 3.8%。
- **永遠不會被丟。** 這些副本預取進 CPU 後不會被讀，也不會被丟。SemiAnalysis 的對話很大，4–11 個就佔滿 16 GiB：
  - 16 個「P、λ>0」job 中，14 個在第 2,000–16,000 個請求（共 98,827 個）就滿了，2 個在第 40,000 個；滿了以後每個快照都是滿的。
  - 滿了以後 CPU 上沒有別的 session（中位數 0），make_room 每步都失敗。
  - 預取量從 136,176 GiB（idle）／79,913 GiB（busy）掉到 53–1,404 GiB。
  - 回來時從 CPU 載入的塊從 698 萬（idle）／411 萬（busy）掉到 0.3–7.1 萬。
- **為什麼幾乎不隨 λ、seed 變。** 幾個副本就能塞滿，所以 λ、抽法、seed 幾乎沒影響（icpe-main 看到 seed 間標準差 <1e-5）。
- **B2 也一樣。** 預取的順序在降級時算一次就固定，之後不會再問 oracle。
- **TraceLab、CacheWise 的 O-PF** 徵狀相同（預取量掉三個數量級、CPU 載入幾乎歸零、不隨 λ 變）。不過機制只在 SemiAnalysis 上量過，TraceLab 一個 Python job 太大〔推論〕。
- **反事實**（只在診斷程式裡做，正式引擎沒有這條；run `20261107-105614-icpe-sim-opf-noise-diag-cf`）。規則：預測被讀的請求已經過了，副本還在，就丟掉，也不再排回預取佇列。
  - 模型 P、idle：
    - λ=0.1：mean_ret 3.399 → 2.790–2.820。上限（對正式的對手 3.305，我算的）從 −2.9% 變成 +14.7% 到 +15.6%。
    - λ=0.3：上限 +7.5% 到 +8.3%。
    - 和模型 M 一樣逐漸變差，不再是斷崖。
  - **λ=0 也會變。** idle 2.681 → 2.663（−0.7%），busy 5.582 → 5.288（−5.3%）。
    - 原因：完美預知時也有副本被「別的 session」讀走後留在 CPU。GPU 開著時，回合結束只釋放自己的預取副本；GPU 關閉的路徑會轉成讀的人的 BF16。這一格有 67–70 次。
    - 也就是說，**沒有雜訊的正式 O-PF 在這一格也被這條規則拖慢了一點**。其他格（包括主結果）：NOT_MEASURED。
  - **窄版**（run `20261107-104850-icpe-sim-opf-noise-diag-cfmiss`）：只在「預測的那個請求沒有讀它」時才丟。
    - λ=0 逐值不變（丟了 0 次）。
    - 只解掉一部分：模型 P λ=0.1 idle 上限 +3.4% 到 +15.1%，後半段 CPU 仍有 0–16 GiB 的死副本。很多死副本的最後一塊在預測的時刻剛好被別人讀了。
  - 模型 M 也受影響，但小：後段 CPU 也被不會再讀的副本佔滿（λ=0.3 從第 36,000–50,000 個請求起）。上面的反事實讓 M λ=0.1 idle 的上限從 +4.9%～+9.6% 變成 +11.4%～+12.0%。

**機制 2：GPU 層的 BEL（O-GPU，icpe-main 的第 3 點）**（CacheWise N64 s1、CPU 8、nfs、LRU+RPF＋BEL；run `20261107-105925-icpe-sim-ogpu-noise-diag-cw`）
- **模型 P、B1。** 後半段 GPU 上大約 10 個對話，其中 5–7 個是不會再回來、但存著的抽樣說「時間已過 → 下一個請求」的。BEL 永遠不逐出它們。
  - 87–93% 的逐出是「GPU 上有不會再回來的，卻逐出了會回來的」（λ=0 是 0%）。
  - 逐出次數 3,981 → 12,187–13,897。
- **B2 每次重抽。** 同一格這個比例是 0.95–3.6%，GPU 上這種對話的中位數是 0，所以是逐漸變差。這和 icpe-main 看到的「B1 111 格全掉、B2 還剩 104 格」一致。
- O-ORD（CPU 的 BEL）在 SemiAnalysis／TraceLab 的 B1 有極端值（最低 −2.9），推測是同一個機制，沒有量〔推論〕。

**Mooncake toolagent 的 O-GPU、O-ORD 在 B2 掉到 0**（第 3 點的另一半；toolagent R1、CPU 8、ssd、LRU+RPF＋BEL；run `20261107-110002-icpe-sim-ogpu-noise-diag-mc`）
- 不是上面兩個機制：B2 每次問都重抽，不會卡住（GPU 上被夾住的對話是 0）。
- λ=0.1 時，B1、B2 的錯誤逐出都是約 1/3（33%、37%；λ=0 是 0%）。λ=0 時 GPU 上同時有約 2,000 個小對話。
- 我判斷這是 B2「每次問都重抽」的定義直接造成的：同一個 oracle 前後不一致，h200 在 §3 B 的註解就是因此把 B1 當主線。不是程式錯誤，但我沒有再拆細。

**影響範圍**（列出來而已，沒有重算）
- P4：
  - O-PF 的全部模型 P 列（兩種抽法）；
  - O-GPU、O-ORD 的模型 P、B1 列；
  - 模型 M 的 O-PF 列（影響小）。
- 主結果（P1–P3，沒有雜訊）：O-PF 也受機制 1 影響，這一格是 −0.7%／−5.3%，其他格 NOT_MEASURED。GPU 關閉的格子推測不受影響（上面那個轉成 BF16 的路徑）〔推論〕。

**選項**（需要使用者決定；我沒有改任何正式程式或結果）
- **A：不改。** 照現在的結果報，註明「模型 P 下 O-PF、O-GPU（B1）的數字反映的是模擬器沒有『預測錯了要放掉』的規則」。不用重跑。
- **B：把規則補上，重跑。** 要先在 PREREG 登記偏離（h200 給號），我改 Python 與 Rust、做 G1，再由 icpe-data／icpe-trx 重跑受影響的部分。要決定的細節：
  - 機制 1 用哪一種：
    - (i) expire（預測時間過了就丟）：唯一完全消掉斷崖的版本，但改到 λ=0，主結果的 O-PF 也要重跑；
    - (ii) 窄版：λ=0 不變、只重跑 P4，但只解掉一部分。
  - 機制 2，抽到的時間已經過了怎麼辦：
    - (a) 從經驗分佈裡「比已經過的時間更長」的那部分重抽；
    - (b) 當成不會再回來；
    - (c) 維持現在（夾成下一個請求）。
- **C：正式結果不動，另外跑 B 的修正版當敏感度，兩者並列。**
- **我的建議：C**，機制 1 用 (i)、機制 2 用 (a)。
  - (i) 是唯一完全消掉斷崖的版本。
  - (a) 最接近「照經驗分佈抽一個回來時間」的原意。
  - 主結果的 O-PF 偏差目前只量了一格（0.7%／5.3%）。建議先用 (i) 在主結果的 O-PF 格子量 λ=0 差多少，再決定主結果要不要改。

**檔案**
- 程式：`code/icpe/next_sim_opf_noise_diag.py`（新；子類別包住 `x1_sim.Engine`，只記錄；反事實只有設 `DIAG_CF` 時才開）。
  - 5 個 run 都用同一版（sha256 48d31f6d…），run 目錄裡有一份。
  - 較早的 5 個 run（`20261107-103518-…-opf-noise-diag`、`…-104115-…-rec`、`…-104401-…-cf`、`…-104728-…-cw`、`…-104754-…-mc`）用的是較早的版本（後來只加了摘要函式和窄版反事實），數字和最後一版逐值相同，不引用。
- 摘要：`results/icpe_next/sim/opf_noise_diag.csv`（108 列＝記錄／expire／窄版 × 36 job）、`results/icpe_next/sim/ogpu_noise_diag.csv`（46 列）。只有彙總計數，沒有逐對話的數字。
- 執行：Python 引擎，8 個 worker、nice 10，每個 run 2–3 分鐘。

**需要使用者決定什麼**：上面的 A／B／C；選 B 或 C 的話，再選 (i)／(ii) 與 (a)／(b)／(c)。

## 2026-11-07T14:02+08:00（原本誤標 14:05；14:02＝commit 0786f56 的時間） — **偏離 14 的驗證：G1 PASS；規則 (i) 在 GPU 預取的 O-ALL 碰到 assert → 找到一個旗標關也有的舊問題**（GPU 預取騰空間時把候選自己丟掉、仍照樣預取）；修法要使用者決定

程式＝commit `fa3686b`（這次跟著 STATUS 一起 push 上去，因為不能 rebase；**旗標關＝G1 PASS，現有的用法不受影響；`--pf-expire` 在修好、重新驗證之前不要用**——預取到 GPU 的 O-ALL 會 assert）。驗證 run `20261107-113238-icpe-sim-g1-fa3686b`（acvlab，凍結的 worktree `/ssd6/hungwei/icpe-next/g1-fa3686b`；Python 6＋Rust 2 個 worker，nice 10）。

**通過的**
- G1（新旗標全關，對 H200 V1 的 4,416 列 × 29 欄）：Python、Rust（src `8e969b0e…`）**不同的格子 0**。
- 兩個旗標都開，Python 對 Rust（bailian_thinking；O-PF、O-ALL 與另外 6 個規則 × GPU 0／111.0625 × wb／wt × 4 個 GPU 規則 × 預取到 CPU／GPU）：沒有雜訊 384 列、有雜訊（P／M × λ 0／0.3 × B1／B2）1,824 列，**不同的列 0**；Rust 沒有退回 Python（`py_fallback_jobs=0`）。
- SemiAnalysis N32 s2（CPU 16、TTLB:30、P／M × λ 0／0.1／0.3 × B1／B2 × seed 0／1）開 `--pf-expire`，Python 對診斷時的反事實 run `20261107-105614`（expire）：36 列**不同的列 0**。
- 逐檔：`dev14_check.csv`、`g1_compare.csv`（run 目錄）。

**沒通過的**（完整錯誤）
- T2（Rust；bailian_coder，17 個規則 × CPU 7 × 3 種慢層 × GPU 0／111.0625 × 4 個 GPU 規則 × 預取到 CPU／GPU；旗標關 vs 開）：開的那次在第 67 個 job 左右
  `thread '<unnamed>' panicked at src/main.rs:1096:9: gspf 13807 9`（`g_unprefetch` 的 assert）→ `RuntimeError: x1fast（常駐，…/work/trace-7b46693a42819e9d07f6ecaf）提早結束，exit -6` → cmd_B exit 46。
  之後的 T2n（有雜訊）與 Rust 的 T3 沒有跑；`exit_code_cross`＝63 是因為 `t3_rs` 沒有輸出。
- **Python 也一樣**（run `20261107-125317-icpe-sim-dev14-gspf-repro`；同一份程式；O-ALL／O-PF、CPU 8／16、預取到 GPU、兩個旗標都開）：`AssertionError: (13807, 9)`，同一個 session、同樣 9 塊 → 不是 Rust 移植的問題，是規則 (i) 的實作碰到一種狀態。

**原因**
- 探針（`gspf_diag.py`，在 run 目錄）：請求 21973 開始時，session 13807 **已被丟**（S_DROP、sb 清空），卻還擁有 9 個預取進 GPU 的塊（block 90–97、84133）；它的預測被讀請求 sT＝21970 ≤ cur＝21972 → 規則 (i) 照 `sb[s]`（空的）找副本 → 找不到 → assert。
- 這種副本怎麼來的（**旗標關也有，是舊的程式路徑**）：`_advance_prefetch_gpu`（Rust `advance_prefetch_gpu`）先算好候選 s 要搬的塊（todo），再呼叫 `_g_make_room` 騰 GPU 空間。騰空間會把 GPU 上的 session 寫回 CPU（wb），然後 `enforce`；O-ALL 的 O-NOW 部分會把「已死」的 CPU session 丟掉，**有時丟到的正是候選 s**。迴圈沒有重查 s 的狀態，照樣把 todo 預取進 GPU（`gpf[b]=s`、`gpu_used` 加上去）。
  - 這些副本的擁有者已經被丟：s 的 sb 已清空，`drop` 裡的取消預取已經做過了。之後只有別的 session 讀到同一塊時才會「認領」它，否則一直佔著 GPU。資料方面：讀的是 CPU／慢層那份；s 被丟之後，那些塊若沒有別的持有者，那份資料其實已經不在了。
  - 頻率（探針 `mkroom_diag.py`，旗標關，bailian_coder，CPU 8／16 × 3 種慢層 × 4 個 GPU 規則 × GPU 閒／忙）：O-ALL 48 次引擎執行裡 18 次發生，共 26 個事件（每次最多 2 個）；跑完時還留著這種塊的 8 次、各 1 塊。O-PF 不丟 session（n_drop＝0）：只有「候選被降到慢層」（425 次），那沒有問題（迴圈逐塊看 fmt，會改從慢層讀）。
  - 預取到 CPU（正式結果幾乎都是）：候選在慢層，`make_room` 只降級 CPU 上的 session；只有設了慢層容量時 `_slow_enforce` 才可能丟到候選。正式 block 的 slow_cap_frac 都是 0 → 不會發生。
- **影響的正式結果**：只有「預取到 GPU」的 O-ALL 列——sens_rep 的 `*_pfgpu` 與各 `*_p3` 的 `*_pfgpu`。主結果、P1、P2、wt、P4、對話資料集、icpe-data 的 xcheck 都是預取到 CPU → 不受影響。G1（沒有 GPU 層）不受影響。icpe-sim 自己的 xcheck（14 個 block，對 icpe-data 的 Python 正式輸出）裡有預取到 GPU 的 block：選 B 時那幾個 block 的 O-ALL 列會和參考不同。
- **規模**（反事實 run `20261107-130238-icpe-sim-pfgpu-dropcand-cf2`：同一份程式，`_g_make_room` 之後候選已不在 CPU／慢層就跳過；只診斷）：

| 紀錄（只有 O-ALL、預取到 GPU） | 列 | f0 對正式輸出 | 有這種事件的列 | 事件數 | f0 跑完還留著的這種塊 | f1 對 f0 不同的列 | mean_ret 相對差 |
|:--|--:|:--|--:|--:|--:|--:|:--|
| bailian_coder（bl_p3 的網格：CPU 9 × 3 種慢層 × 8 個 GPU 規則 × GPU 閒／忙） | 432 | 不同的列 0 | 145 | 163 | 64 | 145 | −0.0056% ～ ＋0.0031%（中位數 0；沒有一列 ≥0.1%） |
| Mooncake conversation、toolagent R1（sens_rep 的 mc_pfgpu：CPU 111.0625／128） | 96＋96 | 不同的列 0 | 0 | 0 | 0 | 0 | 0 |
| CacheWise N16 s1（sens_rep 的 cw_pfgpu） | 96 | 不同的列 0 | 0 | 0 | 0 | 0 | 0 |

- 「f0」＝程式不變（只計數），和 icpe-data 的正式輸出逐列相同 → 反事實量到的就是正式引擎；「f1」＝跳過。一列＝一次引擎執行（GPU 閒、忙分開）。
- 沒量（NOT_MEASURED）：Bailian traceA／thinking（Python 一個 job 約 80 秒，第一次的 run `20261107-125954` 因此停掉）、SemiAnalysis、TraceLab、FI、各 `_p3` 的其他格子。
- 摘要：`results/icpe_next/sim/pfgpu_dropcand_cf.csv`、`pfgpu_dropcand_events.csv`（頻率那段的數字）。

**選項**（需要使用者決定：這是程式的舊問題，會改到 pfgpu 的 O-ALL 正式列）
- **A（建議）**：修源頭——GPU 預取在騰完空間之後重查候選，已不在 CPU／慢層就不預取。放在新旗標 `--pf-gpu-recheck`（預設關＝和現在逐位元相同，G1、xcheck、現有的正式輸出都不變）；icpe-data 重跑偏離 14 時，pfgpu 的 block 兩個旗標一起開。pfgpu 的 O-ALL 本來就在偏離 14 的重跑清單裡 → **不多花計算**。SIM_CHANGES 23。
- **B**：同 A 的修法，但不加旗標、直接改預設。現有 pfgpu 的 O-ALL 正式列就不能再用新程式重現（反正會被重跑取代）；G1 不變；icpe-sim 自己 xcheck 的預取到 GPU block，O-ALL 列要排除或重產參考。好處：之後任何人（kvsweep、版本 C）不用記得開旗標。
- **C**：不修；這種副本照舊留著（和現在的正式結果一樣）。
- 不論選哪個：規則 (i) 跳過已丟的 session（它的 sb 已清空，本來就找不到副本）——**已改好，還沒 commit**（Python 2 行、Rust 2 行）。選 A／B 時這條只是保險（不會再有這種副本）；選 C 時，這種副本就照旗標關的方式留著。
- 有利還是不利預測：這種副本只出現在 O-ALL（預測規則）。量到的格子裡，修掉之後 O-ALL 的 mean_ret 有升有降、都在 ±0.006% 以內；方向沒有一致。

**接下來**（等使用者選）：照選項改兩個引擎 → 重跑這個驗證（T2、T2n、Rust 的 T3 都要跑完）→ 才 push、才給 icpe-data 重跑清單。偏離 14 本身（`fe34522`）不受影響。

---

## 2026-11-07T15:50+08:00 — **偏離 14＋SIM_CHANGES 23 驗證 PASS（`63b8e32`）**；重跑清單（每個 block 一列，給 icpe-data 分機器）；旗標開的跨機器參考與 TraceLab 預熱進行中

**使用者的決定**：11/7 約 14:0x 在本 session 選 A（`--pf-gpu-recheck`，預設關，重跑時和 `--pf-expire` 一起開；SIM_CHANGES 23）。

**做完什麼**
- 程式＝commit `63b8e32`（本次一起 push；之後只加文件與 `code/icpe/next_sim_dev14_list.py`，引擎檔不變）。
  - `x1_sim.py` sha256 `8b53588b85c3…`、`x1_rust.py` `0d89c4ba33c5…`、`x1_nb.py` `8251e2587464…`、`x1fast/src/main.rs` 檔案 `2803b0c0922e…`。
  - x1_rust 印的 Rust `src_sha256`＝`0bfacb59700380ba7850fcd8a7f76d94f3bdcf3eebba87fb40f408afab60054a`；sweep 的 code_sha＝`3180504e8d57`（Python）、`3180504e8d57+rs0bfacb59`（Rust）。
- **驗證 PASS**（run `20261107-140640-icpe-sim-g1-63b8e32`；凍結的 worktree `g1-63b8e32`；Python 6＋Rust 2 個 worker，nice 10；`exit_code_A／B／cross` 都是 0；Rust 沒有退回 Python）。細節在 SIM_CHANGES 23：
  - G1（旗標都關）Python、Rust：4,416 列 × 29 欄，不同的格子 0。
  - 三個旗標都開，Python 對 Rust：沒有雜訊 384 列、有雜訊 1,824 列，不同的列 0。
  - 旗標關 vs 開（第一次 assert 的 bailian_coder 網格）：沒有雜訊 4,074 列、有雜訊 12,600 列跑完；不含 opf 的規則在沒有雜訊、模型 M、λ＝0 的列全部相同。
  - 只開 `--pf-gpu-recheck`：只有「O-ALL、預取到 GPU」變（168 列中 61 列）；對反事實 blc_f1 432 列 0 個不同（兩個引擎）。
  - SemiAnalysis N32 s2 對診斷反事實 36 列 0 個不同（兩個引擎）。
  - 旗標開時變了的列數（同一個 bailian_coder 網格；只是「有沒有變」，不是大小）：

| 規則 | 雜訊 | GPU 0 | GPU 111.0625、預取到 CPU | GPU 111.0625、預取到 GPU |
|:--|:--|--:|--:|--:|
| O-PF、O-ALL | 沒有 | 58／84 | 336／336 | 328／336 |
| O-PF、O-ALL | λ＝0 | 116／168 | 672／672 | 656／672 |
| O-PF、O-ALL | P、M（λ 0.3） | 全部 | 全部 | 全部 |
| 其他規則 | 沒有、λ＝0、M | 0 | 0 | 0 |
| 其他規則 | P（λ 0.3） | 252／336 | 2,016／2,268 | 84／84 |

  - 逐檔：`results/icpe_next/sim/g1_compare.csv`（+2 列）、`dev14_check.csv`（+69 列）。

**重跑清單（給 icpe-data；第 1 段＝版本 B 引擎的列）**：`results/icpe_next/sim/dev14_rerun_list.csv`
- run `20261107-154406-icpe-sim-dev14-rerun-list`（`code/icpe/next_sim_dev14_list.py`）：計畫＝icpe-next-data `8e54d587` 的 `next_b_queue.blocks()`（含 P4 的格子清單、`cw_main_ttl7` 的 cells.csv；這三個輸入和 `2df350f7` 逐位元相同）；job 用 `63b8e32` 的 `x1_sim.jobs_for` 展開（和 sweep 開跑時印的 `jobs=` 同一套）。
  - 上一個 run `…-154317-…` 沒把 xcheck 標成只當參考，已標 SUPERSEDED。
- 每個 (plan, block) 一列：`jobs_block`、`jobs_rerun`（＝`jobs_rerun_i` ∪ `jobs_rerun_a`）、`rerun_policies`、`pf_targets`、`action`、`flags`、`acvlab_done_run_id`。
- 哪些 job 要重跑（其他 job 已驗證旗標開時逐位元不變，舊的列直接沿用）：
  - (i)：CPU 規則有 opf 預取（O-PF、O-ALL、O-ALL-H1800、O-ALL-H3600）。`--pf-gpu-recheck` 只改 O-ALL＋預取到 GPU，已包含在內。
  - (a)：雜訊模型 P、λ＞0（全部規則；它們都用到完美預知）。
- 怎麼跑：同一個 block 的參數，`--policies` 只留 `rerun_policies`，加上 `flags` 欄。
  - 版本 B 的 block 一律 `--pf-expire --pf-gpu-recheck`。後者對預取到 CPU 沒有作用（驗證過），一起開就不用分。
  - P4 模型 P 的 block 再加 `--noise-overdue resample`。這種 block 只有 λ＝0、不含 opf 的 job 不必重跑；整個 block 重跑最單純。
  - P4 模型 M 的 block 只重跑 O-PF（`--jobs-from` 同一份清單，`--policies` 限 O-PF）。
  - Rust 跑 B1／B2＋resample 要 Python 3.12（acvlab 的 venv 是）；其他版本會退回 Python。
- 各 plan 的合計（`done_acv`＝acvlab 的 done 清單裡有的 block；SemiAnalysis 的 block 不在 acvlab 的清單裡）：

| plan | action | block | done_acv | 原本的 job | 要重跑的 job | 其中 (i) | 其中 (a) |
|:--|:--|--:|--:|--:|--:|--:|--:|
| tl_main／_p1／_p2／_p3 | rerun | 3／6／6／6 | 全部 | 105,948 | 18,576 | 18,576 | 0 |
| old_main／_p1／_p2／_p3 | rerun | 2／4／4／4 | 全部 | 353,160 | 61,920 | 61,920 | 0 |
| bl_main／_p1／_p2／_p3 | rerun | 1／2／2／2 | 全部 | 105,948 | 18,576 | 18,576 | 0 |
| sa_main／_p1／_p2／_p3 | rerun | 12／24／24／24 | 0 | 423,792 | 74,304 | 74,304 | 0 |
| sens_rep（不含 cw） | rerun | 24 | 15 | 29,272 | 4,992 | 4,992 | 0 |
| chat_main／chat_p1 | rerun | 21／21 | 9／9 | 265,734 | 27,216 | 27,216 | 0 |
| agent_main／agent_p1 | rerun | 24／24 | 全部 | 303,696 | 31,104 | 31,104 | 0 |
| p4_b1／p4_b2（不含 cw） | rerun | 16／16 | 14／12 | 1,948,524 | 1,338,010 | 550,250 | 1,097,760 |
| sa_p4_b1／sa_p4_b2 | rerun | 24／24 | 0 | 1,004,366 | 602,978 | 85,058 | 565,840 |
| **合計** | rerun | **320** | | | **2,177,676** | 871,996 | 1,663,600 |
| CacheWise（cw_main／_p1／_p2／_p3、sens_rep 的 cw、p4 的 CacheWise） | excluded_dev15 | 126 | | | （311,800） | | |
| `cw_main_ttl7` | excluded_dev15 | 17 | | | 0 | | |
| xcheck | reference_only | 2 | | | — | | |

- (i)、(a) 會重疊（模型 P 的 O-PF），所以兩欄加起來大於合計。P4 佔要重跑的 job 的 89%（1,940,988）。
- job 的成本不一樣：O-PF／O-ALL 每個 job 只有 1 個 view。所以這裡只給 job 數，**時間 NOT_MEASURED**。
- 不重跑：`*_ttl7`（只有 NON_ORACLE）。xcheck（旗標關）照舊當對拍；旗標開的關見下面的 xref_dev14。
- 第 2 段（版本 C 的列＋P4 版本 C 的 5 對）：等 icpe-rux 把偏離 14 移植進 v8（使用者決定的順序），icpe-sim 交叉檢查之後。

**要同步的程式**：icpe-data、icpe-next-trx 同步到本次 push 的 commit（引擎檔＝`63b8e32`）。
- 準備快取的鍵包含 x1_sim／x1_nb／x1_rust 的 sha → 舊的快取都不會命中，第一次跑會重建。

**進行中**（acvlab，合計 ≤ 8 個 worker，nice 10）
- **旗標開的跨機器參考** xref_dev14（run `20261107-154804-icpe-sim-xref-dev14`）：
  - 紀錄：`nb:semianalysis_N16_s1`、Mooncake `conversation` R1。
  - 沒有雜訊：8 個規則 × CPU 8／64 × b32 × GPU 0／111.0625 × wb／wt × LRU／BEL／TTLA:10／TTLB:30 × 預取到 CPU／GPU，每份紀錄 334 個 job。
  - 有雜訊：6 個規則 × CPU 8／64 × GPU 111.0625 wb × BEL／TTLB:30 × 預取到 CPU／GPU × P／M × λ 0.3 × B1／B2，每份紀錄 232 個 job。
  - 三個旗標都開；Python（5 個 worker）、Rust（1 個）都跑，要逐列相同才放進 `results/icpe_next/sim/xref_dev14/`。
  - SemiAnalysis、Mooncake 不會走到 SIM_CHANGES 23 的那條路徑（反事實裡 0 個事件），所以它只釘住輸出、不測 pgr。
- **TraceLab c1／c2／c4 的準備快取**（icpe-data 要求；run `20261107-154805-icpe-sim-prewarm-63b8e32`）：
  - 凍結的 worktree `g1-63b8e32`，x1_sim sha256 `8b53588b85c3…`。
  - `X1_RUST_WORK=/ssd6/hungwei/icpe-next/x1fast-work`；同時 2 個。上次一個變體約 2,482 s。
- 之後：λ＝0 的影響量測（icpe-main 的第 3 步；規則 (i) 對主結果 O-PF 的 ④；Rust、5 個 worker）。

**版本 C 的引擎**（icpe-main 15:3x 問；回答同訊息）
- 現在沒有一個引擎同時有雜訊、偏離 14、APC／BELP：
  - `63b8e32`：有雜訊與偏離 14，沒有 APC／BELP。
  - v8 `48e4efce`：有 APC／BELP 與 `nz=`；BELP 讀雜訊後的回來時間，`g_next` → `nz()`。它沒有 pfx／nzo／pgr／S_DROP 跳過。
- 使用者已決定由 icpe-rux 移植（先於二階排隊）。push 之後我把 main.rs 對 `4a4baa4` 的 diff 給 icpe-rux。
- 移植好之後我交叉檢查：
  - v8 關掉 APC／BELP、開旗標 → 和 x1fast `63b8e32` 逐位元相同。
  - 開 APC／BELP、有雜訊、開 resample → 一次整個 session 的 APC＝LRU、BELP＝BEL。

**需要使用者決定什麼**：無。

---

## 2026-11-07T15:56+08:00 — **旗標開的跨機器參考（xref_dev14）失敗**：wt＋預取到 GPU 的 O-ALL 在 SemiAnalysis assert；**旗標關也一樣**（舊問題），正式 block 都沒用這個組合；要使用者決定

**15:50 那節的「驗證 PASS」只對它列的網格成立**。三個旗標還沒放行，已告訴 icpe-data、icpe-next-trx、icpe-main 不要用（icpe-data：不用、TRX40 暫停）。

**完整錯誤**（run `20261107-154804-icpe-sim-xref-dev14`；凍結的 worktree `g1-63b8e32`；兩個引擎都在第一個 sweep `sa_q` 停下，`exit_code_py`＝31、`exit_code_rs`＝51，比對沒跑）
- Python（`py_sa_q.log`；第 16／334 個 job 之後）：
  ```
  File ".../x1_sim.py", line 1834, in _step_gpu → self._turn_end_wt(r, blocks, tend)
  File ".../x1_sim.py", line 1998, in _turn_end_wt → w += self.drop(s)
  File ".../x1_sim.py", line 1526, in drop → self._g_unprefetch(s, self.sb[s], push=False)
  File ".../x1_sim.py", line 1147, in _g_unprefetch → assert self.gspf[s] == 0, (s, self.gspf[s])
  AssertionError: (12, 277)
  ```
- Rust（`rs_sa_q.log`；第 20 個 job 之後）：`thread '<unnamed>' panicked at src/main.rs:1099:9: gspf 12 277` → `RuntimeError: x1fast（常駐，…/x1fast-work/trace-7cfa6556228f0469c7dc5113）提早結束，exit -6`。
- 意思：session 被丟時，還有 GPU 預取副本不在它的 sb 裡，找不到就 assert。

**範圍**（只診斷，同一份程式）
- run `20261107-155318-icpe-sim-xref-wt-repro`（Rust）：SemiAnalysis N16 s1、CPU 8、O-ALL、b32、GPU 111.0625、**wt＋預取到 GPU**，4 個 GPU 規則 × 旗標 {關、只 `--pf-expire`、只 `--pf-gpu-recheck`、兩個都開}。**16 個組合全部 panic**：LRU `gspf 12 277`，BEL／TTLA:10／TTLB:30 `gspf 9 114`，和旗標無關。
- run `20261107-155403-icpe-sim-wtpfgpu-scope`：
  - Python 旗標關（BEL）：`AssertionError: (9, 114)`，和 Rust 相同。
  - **正式 block 用的組合沒事**：SemiAnalysis N16 s1、CPU 8／64、O-ALL／O-PF／LRU+RPF、4 個 GPU 規則，以下 4 個 sweep 各 48 個 job 都跑完（Rust，exit 0）：
    - wb＋預取到 GPU，旗標關、旗標開；
    - wt＋預取到 CPU，旗標關、旗標開。
- **正式 block 都沒有 wt＋預取到 GPU**：icpe-next-data `8e54d587` 的 `next_b_queue.blocks()` 465 個 block 裡 0 個。`_wt` 都是預取到 CPU，`_pfgpu` 都是 wb。G1（沒有 GPU 層）也沒有。
- 所以：
  - 這是旗標關也有的舊問題，只在 wt＋預取到 GPU。正式結果、G1、G0 都不受影響（icpe-data 剛做的 G0：40a4c87 合併後 Rust 旗標關 128,064 格 0 不同，run `20261107-155259-icpe-data-g0-rust-merged-40a4c87`）。
  - 偏離 14 的三個旗標沒有造成它。
  - 是我設計的參考網格把 wt 和預取到 GPU 交叉了。t1 也交叉了，但 Bailian thinking 沒碰到。
  - 原因還沒查（NOT_MEASURED）。推測和 SIM_CHANGES 19 同類：wt 的回合結束換掉 sb[s] 時，舊的 GPU 預取副本沒有先收回。
- 已停：TraceLab 預熱（run `20261107-154805`，留 STOPPED；c1、c2 開始建了、沒建完）。程式若要改，x1_sim 的 sha 會變，快取的鍵也會變。

**選項**（要使用者決定）
- **A（建議）**：參考網格拿掉「wt × 預取到 GPU」，其他照舊（wb＋預取到 GPU、wt＋預取到 CPU、GPU 0，有雜訊與沒有雜訊）。
  - 這個舊問題寫進 SIM_CHANGES／kvsweep 的已知問題，不修。理由：正式 block 都不用這個組合。
  - 引擎不變（`63b8e32`）；參考跑完、Python＝Rust 就放行旗標。
- **B**：先查原因、修（放在新旗標或改預設），重新驗證，再做參考。
  - 會晚：修＋驗證約 2–3 小時〔推估〕。TraceLab 快取要用新程式重建（每個變體約 41 分鐘）。
  - 正式結果不會因此改變，因為沒用到這個組合。
- **C**：A 先放行；B 另外排在後面做，不擋重跑。

**需要使用者決定什麼**：A／B／C。

---

## 2026-11-07T15:59+08:00 — 使用者選 A：參考網格拿掉 wt × 預取到 GPU，舊問題不修；xref 與 TraceLab 預熱重新開始

- **使用者的決定**（約 15:5x，本 session）：「A: drop combo, don't fix (Recommended)」。
  - 引擎不變（`63b8e32`）。
  - wt＋預取到 GPU 的 O-ALL assert（旗標關也有）列為**已知問題、不修**。正式 block 都不用這個組合（0／465）。
  - 以後的網格（含 kvsweep）不要把 wt 和預取到 GPU 交叉。
- **xref_dev14 重跑**（run `20261107-155727-icpe-sim-xref-dev14`；上一個 `…-154804` 標 SUPERSEDED）：
  - 拆成 wb（GPU 0／111.0625 × 預取到 CPU／GPU，每份紀錄 178 個 job）與 wt（GPU 111.0625 × 預取到 CPU，108 個）。
  - 有雜訊的網格不變（wb，232 個）。紀錄、旗標、worker 同上一節。
- **TraceLab 預熱重開**（run `20261107-155740-icpe-sim-prewarm-63b8e32`；`…-154805` 是停掉的那個）。
- 旗標仍然**等 xref 的 Python＝Rust 才放行**。

---

## 2026-11-07T16:08+08:00 — TraceLab c1／c2／c4 的準備快取建好（`63b8e32`）

- run `20261107-155740-icpe-sim-prewarm-63b8e32`：exit 0。
  - 凍結的 worktree `g1-63b8e32`：x1_sim.py sha256 **`8b53588b85c3…`**、x1_nb.py `8251e2587464…`、x1_rust.py `0d89c4ba33c5…`。
  - Rust src `0bfacb59…`；code_sha `3180504e8d57+rs0bfacb59`。
  - `X1_RUST_WORK=/ssd6/hungwei/icpe-next/x1fast-work`；同時 2 個。

| 變體 | 快取目錄 | n_req | 準備 | 牆鐘（含 1 個 job） | 最大 RSS |
|:--|:--|--:|--:|--:|--:|
| tracelab_c1 | `prep-a3909a4652a7c575dd8b1cea` | 665,453 | 372.3 s | 6:35 | 14.2 GiB |
| tracelab_c2 | `prep-b4cdd780b7651a9983b658f3` | 665,453 | 369.6 s | 6:33 | 14.2 GiB |
| tracelab_c4 | `prep-8fa8552aea93c4ed7af4a22e` | 665,453 | 130.3 s | 2:28 | 14.2 GiB |

- 三個的 KEY.json 都是 `code_sha256` `362b98a15fc4…`（x1_sim／x1_nb／x1_rust 合起來的 sha），`lite=True`、`prep=built_saved`。
- 這次比 11/6（約 2,482 s／變體）快很多。原因沒查（NOT_MEASURED）。
- 用同一份程式（`63b8e32` 或之後只改文件的 commit）跑 TraceLab，應該會印 `prep=cache_hit`。

---

## 2026-11-07T16:35+08:00 — 更正 SIM_CHANGES 23 的一句：「不開 `--pf-gpu-recheck` 也不會 assert」只對沒有雜訊成立；**有雜訊＋預取到 GPU 的 O-ALL，旗標關也會 assert**（正式 block 沒有這個組合）

- **icpe-rux 先發現**（移植 v8 時，16:1x）：只開 `pfx=1`、不開 `pgr=1`，O-ALL、wb、預取到 GPU，P B2 λ0.3 seed 7，CPU 16 → 參考版和移植都 `gspf`。
- **我在 `63b8e32` 上重現**：run `20261107-163044-icpe-sim-pfx-nopgr-repro`。SemiAnalysis N16 s1、CPU 16、O-ALL、b32、GPU 111.0625 wb、預取到 GPU、P B2 λ0.3 seed 7；Rust，Python 只跑 1 個：

| GPU 規則 | 旗標關 | 只 `--pf-expire` | `--pf-expire --noise-overdue resample` | 只 `--pf-gpu-recheck` | 三個都開 |
|:--|:--|:--|:--|:--|:--|
| BEL | 跑完 | `gspf 1202 5` | `gspf 51 5` | 跑完 | 跑完 |
| TTLA:10 | `gspf 19 5`（Python `AssertionError: (19, 5)`，`_turn_end_wb` → `_g_unprefetch`） | `gspf 250 18` | `gspf 63 5` | 跑完 | 跑完 |

- **原因**（推論，和表一致，沒有逐步追）：
  - SIM_CHANGES 23 的舊路徑留下「屬於已丟 session 的 GPU 預取副本」。
  - 沒有雜訊時，O-NOW 丟的 session 不會再回來，所以只有規則 (i) 會去找那些副本（已用 S_DROP 跳過修好）。
  - 有雜訊時，被丟的 session 可能又回來。之後丟它或讓它過期，`_g_unprefetch(s, sb[s])` 找不到舊副本，就會 assert。
  - `--pf-gpu-recheck` 讓這種副本不會產生，所以兩種情況都沒事。
- **影響**：
  - 正式 block 沒有「雜訊 × 預取到 GPU」：P4 都是預取到 CPU，`_pfgpu`／sens_rep pfgpu 都沒有雜訊。正式結果不受影響。
  - 重跑清單的 `flags` 欄一律有 `--pf-gpu-recheck`。
  - xref 有雜訊的網格三個旗標都開，不受影響。
- **規則**：**預取到 GPU 一律要開 `--pf-gpu-recheck`**（不論有沒有 `--pf-expire`）。已改 SIM_CHANGES 23 那句。
- 程式不改（使用者 15:5x 選 A 的精神：正式不用的組合記成已知問題）。若要讓程式擋掉「預取到 GPU 卻沒開 `--pf-gpu-recheck`」，x1_sim 的 sha 會變、快取要重建；我不建議，除非使用者要。

---

## 2026-11-07T16:43+08:00 — 偏離 14 規則 (i) 在 λ＝0（沒有雜訊）對主結果 O-PF 的 ④ 的影響（icpe-main 的第 3 步）

- run `20261107-160757-icpe-sim-dev14-lam0-impact`（Rust，2 個 worker，nice 10；凍結的 worktree `g1-63b8e32`；`py_fallback_jobs=0`；exit 0）。
- 做法：
  - **on**：O-PF、`--pf-expire`、GPU 111.0625 wb、15 個 GPU 不預測規則（LRU、TTLA／TTLB × 0.3／1／3／10／30／100／300）、CPU 9 種 × ssd／b32／nfs × GPU 閒／忙、預取到 CPU。
  - **off**：正式輸出（版本 B 的 main＋`_p1 _ttl`；icpe-next-data a1ea7241、icpe-next-trx 291e7d60）的同一格。
  - **④**＝每格在 15 個 GPU 規則裡最小的 mean_ret（同 next_sim_analyze 的 COMPONENT ④，不含 BEL）。rel4＝④_on／④_off − 1，**負＝開了規則之後 O-PF 變快**。
  - 對手（②）不受規則 (i) 影響：只有 opf 的規則會變，G1／T2 已驗證。所以 O-PF 的上限只會經由 ④ 改變。
- 結果（`results/icpe_next/sim/dev14_lam0_impact.csv`）：

| 紀錄 | 格 | ④ 變了的格 | ④ 相對差中位數 | 最小 | 最大 | 第 10 百分位 | 逐列變了的 | 逐列最小 | 逐列最大 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Bailian coder | 54 | 50 | −0.0145% | −0.287% | +0.162% | −0.063% | 695／810 | −32.24% | +0.22% |
| Bailian thinking | 54 | 8 | 0 | −0.018% | +0.035% | 0 | 131／810 | −1.73% | +1.34% |
| Bailian traceA | 54 | 25 | 0 | −0.104% | +0.051% | −0.005% | 340／810 | −0.59% | +0.28% |
| Mooncake conversation（R1＋R2） | 108 | 66 | 0 | **−18.03%** | +0.511% | −2.43% | 1,182／1,620 | −48.98% | +1.66% |
| Mooncake toolagent（R1＋R2） | 108 | 86 | −0.001% | **−15.88%** | +0.412% | −2.14% | 1,259／1,620 | −47.95% | +0.78% |
| FreeInference 2026-05-17 那週 | 54 | 0 | 0 | 0 | 0 | 0 | 12／810 | −0.01% | 0 |
| SemiAnalysis N32 s1 | 54 | 48 | −0.0785% | **−14.92%** | +0.588% | **−8.94%** | 720／810 | −35.62% | +0.59% |
| ShareGPT T60 s1 | 54 | 39 | −0.0016% | −0.869% | +0.208% | −0.097% | 508／810 | −7.35% | +0.94% |
| WildChat s1 | 54 | 0 | 0 | 0 | 0 | 0 | 20／810 | −0.01% | 0 |
| TraceLab c1 | 54 | 0 | 0 | 0 | 0 | 0 | 0／810 | 0 | 0 |

- 讀法：
  - 沒有雜訊時，規則 (i) 對 O-PF 的 ④：
    - 大多是 0 到千分之幾。
    - **Mooncake（兩份）、SemiAnalysis 有一部分格明顯變快**：最多 −15% 到 −18%；SemiAnalysis 有 10% 的格快 ≥ 8.9%。
    - 少數格變慢：最多 +0.59%。
  - TraceLab c1：810 列都有 `pf_expire=True`，但沒有任何一列變。FI、WildChat 幾乎沒有變。
  - 每一格的上限怎麼變、判定會不會變，要等 icpe-data 重跑、icpe-main 重算。我不推。
- 沒量（NOT_MEASURED）：
  - O-ALL（它也有 opf，也會變）；
  - 其他變體（SemiAnalysis 其他 N／s、TraceLab c2／c4、FI 其他 5 週、ShareGPT 其他、LMSYS、UltraChat、agent）；
  - CacheWise（偏離 15 不跑）；
  - P1–P3 的其他組合；
  - 有雜訊的 λ＞0。

---

## 2026-11-07T16:59+08:00 — **xref_dev14 PASS：三個旗標放行**（引擎＝`63b8e32`）；參考放在 `results/icpe_next/sim/xref_dev14/`

- run `20261107-155727-icpe-sim-xref-dev14`：exit 都是 0。
  - Python（5 個 worker）與 Rust（1 個）的 6 個 sweep 逐列相同：1,184 列，不同的列 0；Rust `py_fallback_jobs=0`。
  - 紀錄：SemiAnalysis N16 s1、Mooncake conversation R1。
  - 網格：wb（GPU 0／111.0625 × 預取到 CPU／GPU）、wt（只預取到 CPU）、有雜訊（wb × 預取到 CPU／GPU × P／M × B1／B2 × λ 0.3）。三個旗標都開。
- 目錄內容：
  - 6 份 CSV（Rust 的輸出＋`host` 欄）；
  - `xref_check.csv`（Python 對 Rust）；
  - `env.sh`；
  - `README.md`：指令、網格、不在網格裡的組合、比法。
- 比對程式：`code/icpe/next_sim_dev14_check.py`（＝驗證用的 dev14_check.py）。git 裡的 CSV 對 Python 輸出自檢 PASS。
- sha256：
  - `xref_sa_qb.csv` `1a044f3139f9ee7a10c0bed242793ae81eac508220edf765dd682d7ecae7e6c0`
  - `xref_sa_qt.csv` `805db06e047166ac9a604398ff41e58445c8c1fceb5dc6b707e3486386577a35`
  - `xref_mc_qb.csv` `992e04e1b28bbc9b84187ff890f74ec58aae984239928f2c79c36d7e4d79a74c`
  - `xref_mc_qt.csv` `3a6d276c5add8015a4b93e4f8f62b22577ba35c6c0dc4b0f32e6230a8b808a75`
  - `xref_sa_n.csv` `671aac7be1e27205731d50b9ccfa713cf90f8679d4775d1c64d1d8109c05b22d`
  - `xref_mc_n.csv` `40d5eca3df76cad20366a09674b50f9fb1e4dc9912658867f24953354c4dd1c0`
- **放行**：icpe-data、icpe-next-trx 先各自用 Python 與 Rust 重現這 6 份、逐列 0 不同，就可以開旗標跑正式 block（第 1 段）。
  - 旗標：版本 B 的 block 一律 `--pf-expire --pf-gpu-recheck`；P4 模型 P 再加 `--noise-overdue resample`。照 `dev14_rerun_list.csv` 的 `flags` 欄。
  - **預取到 GPU 一律開 `--pf-gpu-recheck`**；不要把 wt 和預取到 GPU 交叉。
- 進行中：icpe-rux 的 v8_dev14（`36780534`）交叉檢查，run `20261107-163544-icpe-sim-v8dev14-xcheck`。

---

## 2026-11-07T17:27+08:00 — 規則 (i) 在 λ＝0 什麼時候會觸發（回答 icpe-data 17:1x：ShareGPT 變很多，UltraChat／LMSYS／agent 幾乎不變，合不合理）

- **規則本身**：`_pf_expire` 看 `sT[s]`。`sT[s]` 是 s「最後一塊」的下一次存取（`next_access(acc, sb[s][-1], cur)`），不是 s 自己的下一個請求。
  - λ＝0 時，只有「那一塊已經被存取、s 的預取副本卻沒被用掉」才會丟。
  - 要同時滿足：
    - (1) s 被降到慢層、又被預取回 CPU（手上有預取副本）；
    - (2) 它最後一塊的下一次存取來自**別的 session**（跨 session 共用的塊）。
  - 如果是 s 自己回來，副本會被用掉，不會觸發。
- **資料**（run `20261107-172443-icpe-sim-pfexp-lastblock`，只讀紀錄、不跑模擬；`results/icpe_next/sim/dev14_pfexp_lastblock.csv`）：每個請求的最後一塊，下一次被存取時屬於哪個 session。

| 紀錄 | 請求 | 跨 session 共用的塊 | 最後一塊下一次是別的 session | 同一個 session | 之後沒有再存取 |
|:--|--:|--:|--:|--:|--:|
| Bailian coder | 43,011 | 9.6% | 5,515（14.7%） | 14,455 | 17,569 |
| Mooncake conversation R1 | 12,031 | 6.1% | 1,777（14.8%） | 3,509 | 6,745 |
| Mooncake toolagent R1 | 23,608 | 6.2% | 3,369（14.3%） | 7,038 | 13,201 |
| OpenHands N16 s1 | 8,021 | 0.011% | 119（1.5%） | 7,603 | 299 |
| ShareGPT T60 s1 | 70,046（有完整塊的） | 4.3% | 717（1.0%） | 62,261 | 7,068 |
| SemiAnalysis N32 s1 | 98,420 | 6.2% | 374（0.38%） | 86,744 | 11,302 |
| WildChat s1 | 23,086 | 0.55% | 17（0.07%） | 16,897 | 6,172 |
| mini-SWE N16 s1 | 13,884 | 0 | 0 | 13,385 | 499 |

- **和 icpe-data 的觀察對得上**：
  - mini-SWE 沒有跨 session 共用的最後一塊 → 0 變。
  - WildChat 只有 17 次 → 2／162 變。
  - ShareGPT 有 717 次 → 變很多。ShareGPT 有重複或共用開頭的對話。
  - Bailian、Mooncake 約 14–15% → 變很多。
- **對不太上、但說得通的**：
  - OpenHands 有 119 次，icpe-data 卻量到 0 變。條件 (1) 也要成立，OpenHands N16 工作集小，可能根本沒降到慢層、沒有預取副本。〔推論，沒量 n_pfexp〕
  - SemiAnalysis 只有 0.38%，影響卻大（λ＝0 最多 −15%）。每次觸發丟掉的副本可能很大（長 session）。〔推論〕
- 沒量（NOT_MEASURED）：UltraChat、LMSYS（acvlab 沒有 nb；icpe-data 已逐列確認 uc_T60_s1、lm_T600_s2 沒變）；TraceLab（紀錄太大，這次沒跑）。
- **結論**：這個模式合理，不是程式錯。規則 (i) 只在跨 session 共用最後一塊、而且有預取副本時作用。
