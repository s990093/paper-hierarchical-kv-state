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
