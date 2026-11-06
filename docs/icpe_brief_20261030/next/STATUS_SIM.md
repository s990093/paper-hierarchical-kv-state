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
