# B_Ziggurat Ziggurat: A Tiered File System for Non-Volatile Main Memories and Disks

- **出處**：Shengan Zheng（上海交大）、Morteza Hoseinzadeh、Steven Swanson（UCSD），FAST '19（17th USENIX Conference on File and Storage Technologies），印刷 pp.207–219。
  連結：https://www.usenix.org/system/files/fast19-zheng.pdf
  頁碼規則：p.X＝PDF 第 X 頁，印刷頁＝p.X＋205（PDF p.1 是封面）。讀了 p.2–3、p.5–7、p.12。
- **寫入時做了什麼決定**：**每一筆寫入都在寫入當下決定送到哪一層**：
  - 同步的寫入（app 會等 fsync）或小的寫入 → NVMM；
  - 非同步的大寫入 → 直接走 DRAM page cache 寫到 disk，由背景寫回；
  - 理由：「writing to the DRAM page cache is faster than writing to NVMM, and Ziggurat can write to disk in the background」〔原文 p.3、p.5〕。
  - 另外，背景把冷資料從 NVMM 搬到 disk，並合併成大塊的循序寫入〔原文 p.2〕。
  - 搬移門檻是動態的：依整個檔案系統的讀寫比，在 50% 到 90% 之間變動〔原文 p.6〕。
- **用什麼資訊做決定？寫完之後還在不在？（N1）**：
  - synchronicity predictor：兩次 fsync 之間寫了多少塊，少於門檻（實驗用 1024）就算同步；用 O_SYNC 開的檔案也算〔原文 p.5〕；
  - write-size predictor：每個 write entry 有一個「又大又穩定」的計數，大於 4 才算大〔原文 p.5〕。
  - 都是從同一個檔案過去的寫入歷史推出來的，寫入當下就有。預測準確率 99%，最低 97%〔原文 p.12〕。
- **有沒有和延後版、寫穿版、背景版比較？**
  - 只和其他檔案系統比：對 SSD 上的 EXT4、XFS 分別最多快 38.9 倍、46.5 倍〔原文 p.2〕；也和 NOVA、Strata（只用 NVMM）比〔原文 p.7〕。
  - **我讀到的範圍內，沒有「關掉 predictor，全部先寫 NVMM 再搬」的消融**（原文未提）。
  - 定性說明：固定的高門檻不適合寫入為主的負載，因為 NVMM 的空間會被吃光，「the file writes have to either **stall** before the migration threads clean up enough space in NVMM, or **write to disk**」〔原文 p.6〕。
- **硬體**：用 NUMA 遠端節點的 DRAM 模擬 NVMM；NVMe SSD 的最大 I/O 是 128 KB〔原文 p.7、p.12〕。
- **模型架構、模態**：不適用。
- **和本研究的關係**：
  - **這是儲存系統裡最接近 S5 的設計**：依「讀取者（或寫入者）會不會等這筆資料」在寫入時選層，加上背景搬移。對應到 KV〔判讀〕：Cake 會從 SSD 或 CPU 載入的後段＝「會被等的」；會被重算的前段＝「不會被等的」，可以 write-around 到慢的層。
  - **支持 N3 的條件**：寫入為主時，快的層被吃光，只剩兩條路：等（hold），或寫入時直接繞過快的層。這就是 S5 想贏的情境。
  - **但它沒有證明「寫入時」比「延後套用同一規則」好**：沒有那組消融。所以只能當作「這種設計是合理的」，不能當作「延後做不到」的證據。
- **證據等級**：〔原文 p.X〕；KV 對應〔判讀〕。
