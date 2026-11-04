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
