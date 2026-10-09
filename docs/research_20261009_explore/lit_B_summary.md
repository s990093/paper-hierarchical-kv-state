# Lit-B 摘要：OS 與儲存文獻裡，「寫入時決定」什麼時候勝過「之後再搬」

**日期**：2026-10-09
**執行者**：文獻 agent Lit-B（OS 與儲存）
**上游**：[10_breakthrough_plan.md](../phase1_20261008/10_breakthrough_plan.md) §2、§3E、§4、§5；[11_round1_plan.md](../phase1_20261008/11_round1_plan.md)
**卡片**：`cards/B_*.md`，共 15 張

**標記**：
- 〔X 原文 p.N〕：論文 X 的 PDF 第 N 頁（從 1 起算）。印刷頁碼的換算寫在各卡片的「出處」。
- 〔判讀〕：推論。
- 「未讀原文」：只查了書目，沒讀內文。

**工具與來源**：
- 有 WebSearch／WebFetch，也能用 curl 下載。
- Semantic Scholar API 回 429（rate limit）；DBLP API 回 429 或斷線。所以書目的查證改用三種來源：論文 PDF 首頁、其他論文的參考文獻、網路搜尋。

**讀了原文的（17 篇＋文件）**：

| 主題 | 論文 |
|:--|:--|
| 多層快取 | Wong & Wilkes ATC'02、Chen et al. ATC'03、Gill FAST'08、Karma FAST'07 |
| Flash 快取 | Flashield NSDI'19、Kangaroo SOSP'21、CacheLib OSDI'20、FairyWREN OSDI'24 |
| Tiered memory | TPP（arXiv v2）、Nomad（arXiv v2）、Colloid SOSP'24、Nimble ASPLOS'19 |
| 寫入策略與放置 | Ziggurat FAST'19、Sibyl ISCA'22、Multi-stream SSD HotStorage'14、Jouppi WRL 91/12 |
| 以重算代替儲存 | Tachyon SoCC'14、Checkmate MLSys'20、Nectar OSDI'10 |
| Linux 與 ROCm 文件 | `vm.rst`、open(2)、mlock(2)、nfs(5)、ROCm hipFile 與 HIP host memory 文件 |

FairyWREN、Nimble、Jouppi、Nectar 只寫進本摘要，沒有卡片（卡片上限 15 張）。

**未讀原文**：HeMem、Memtis、ULC、TMO，以及資料庫的 materialized view 選擇（Harinarayan et al. 等）。

**滾雪球**：
- 第 1 輪：Gill → Chen ATC'03（讀了）、ULC（只有書目）；FairyWREN → multi-stream；Colloid、Nomad 的參考文獻 → HeMem、Memtis 的書目。
- 第 2 輪：新的相關論文不到 2 篇，照 10 §5.3 停止。

---

## 0. 一段話結論〔判讀〕

OS 與儲存文獻裡，「放置時就決定」（PROMOTE、Ziggurat、TPP 的類型感知配置、Sibyl）勝過「之後再搬」（DEMOTE、eviction-based placement、滿了才回收），只在三類條件下才明顯：

1. **層間傳輸是瓶頸**：延後版要多付一次傳輸。
2. **搬移卡在請求路徑上，而且長期寫入速率超過背景排空的頻寬**。
3. **寫入者擁有之後的層拿不到的資訊**。

傳輸免費時，經典結果是兩者差 ≤5%：Gill 的 PROMOTE 對 DEMOTE 最多好 4–5%，其他 trace 平均只好 0.3–1.5%〔Gill 原文 p.13〕。這和 08 的 S5L 對 S4B 只快 0.7% 是同一個現象。

KV 有兩個特性，都讓延後版更強：
- **不會被修改（永遠 clean）**：寫穿之後，逐出是免費的（Nomad、CacheLib）。
- **丟了可以重算**：持久化可以變成可延後、可放棄的背景工作（Tachyon、Chen ATC'03）。

所以 OS 文獻**不支持**「依位置在寫入時放」本身能贏。有希望的只剩兩種：
- **N1（資訊不對稱）**：只有寫入當下才有的資訊。
- **一種 KV 特有的 N2**：寫入時決定可以走一條延後版走不了的資料路徑，例如 GPU 直接寫 SSD、不經過 CPU（見 §3 E9）。

---

## 1. 「寫入時決定 > 延後搬」成立的條件清單

「在 KV 上成立嗎」這一欄全部是〔判讀〕。

| # | 條件 | 證據（原文頁碼） | N | 對應的 H | 在 KV 上成立嗎 |
|:--|:--|:--|:--|:--|:--|
| C1 | **層間傳輸頻寬是瓶頸**，延後版要多傳一次 | 見表後說明 1 | N2 | H12、H3、H1；E4、E5 | **大多不成立**。見表後說明 2 |
| C2 | **搬移同步執行、卡在請求路徑上，而且寫入是突發的**（hold） | 見表後說明 3 | N3 | H0、H3 | **前提部分成立**。見表後說明 4 |
| C3 | **長期寫入速率 > 背景排空頻寬**（沒有空閒） | 見表後說明 5 | N3 | **H3** | 要算 doc 型突發寫入的速率對 SSD 寫入頻寬的比值。NOT_MEASURED，要靠模擬加併發 |
| C4 | **快的層遠小於工作集，而且搬移有成本**：初始放置等於最終放置 | 見表後說明 6 | N3／N2 | 條件 2；H3、H8 | 和本研究的條件 2 相同，但 08 已經測過：條件 2 必要但不夠 |
| C5 | **寫入者擁有、之後的層拿不到的資訊**（資訊不對稱） | 見表後說明 7 | **N1** | **H4、H7、H11、H6、H8** | **位置資訊不是這種**（S4B 也看得到）。真正的 N1 是：SSM 狀態被覆蓋（H7）、prefill 節點傳完就丟（H11）、應用層知道「會被很多人讀」（H4）或「會不會回來」（H8） |
| C6 | **放錯之後修正要多寫，而且寫入額度稀缺** | 見表後說明 8 | N2（裝置層） | **H1**、E6 | 只在 SSD 寫入額度有約束時成立。文獻的主要手段是 admission（少寫、不寫），不是放哪一層 |
| C7 | **寫下去的資料不會從這一層被讀**（write-around 的條件） | 見表後說明 9 | — | S5 本身的理由 | 成立（前段會被重算，不從 CPU 讀），但 S4B 延後套用同一條規則也拿得到 |
| C8 | **決定點被資源峰值逼在寫入當下** | Checkmate：activation 在 forward 時就要決定留不留〔Checkmate 原文 p.1〕；Tachyon：最大的 dataset 寫入時就同步寫 disk〔Tachyon 原文 p.6〕 | N3 | H6、H8 | prefill 時 GPU 記憶體是瓶頸才成立。現在的 S5 決定的是 CPU 或 SSD，不受 GPU 峰值限制 |
| C9 | **寫入時順便放置，不必另外付成本**；延後版要另外搬一次 | 見表後說明 10 | N2 | **H13、E9** | **可能成立**：GPU→SSD 直寫（hipFile）時，前段不經過也不占 CPU DRAM。延後版的資料已經在 host，第二段只能走 host→NVMe。只有 CPU DRAM 容量或頻寬稀缺時才有價值 |

**表後說明**（對應表中的「見表後說明 N」）：

1. **C1 的證據**：
   - Gill：頻寬限制在 300 blocks/s 時，PROMOTE 3.42 ms、DEMOTE 5.05 ms；DEMOTE 比單純 LRU 還差，cache 小時比完全不快取還差〔Gill 原文 p.13〕；圖 12〔p.14〕。
   - Wong：SAN 頻寬 <20–30 Mbit/s 時，NONE-LRU 贏 DEMOTE〔Wong 原文 p.7–8〕。
   - Chen：DEMOTE 讓 client→storage 流量幾乎加倍〔Chen ATC'03 原文 p.8–9〕。
   - Karma：沒配到空間的 range 用 READ-SAVE 讀，不做 demote〔Karma 原文 p.5〕。
2. **C1 在 KV 上為什麼大多不成立**：S4 和 S5 的前段都要寫一次 SSD，延後版多的只是一次 CPU DRAM 讀，以及時間點（B_PROMOTE 卡）。只有以下情況會成立：
   - 三層：延後版要搬兩次（H12）；
   - 多張 GPU 爭用 host DRAM 或 PCIe（E4）。
3. **C2 的證據**：
   - Gill：「reads stall until demotions … can create space」〔Gill 原文 p.3〕。
   - Chen：client 的 miss 太突發時，要等 DEMOTE 做完才拿得到 buffer〔Chen 原文 p.9〕。
   - TPP：新配置停下來，等回收補到 high watermark；突發的配置落到 CXL〔TPP 原文 p.7、p.10–11〕。
   - Ziggurat：「either stall … or write to disk」〔Ziggurat 原文 p.6〕。
   - Sibyl：eviction penalty〔Sibyl 原文 p.5–6、p.9〕。
4. **C2 在 KV 上的狀況**：H0 報告（`H0_offload_blocking.md`）顯示，真實系統預設是「放不下就丟」，只有幾個非預設組態會等。而且文獻對 C2 的標準解法是**背景水位線**（TPP）或**寫穿**（Nomad），都不看位置。
5. **C3 的證據**：
   - Tachyon：「if data is written faster than the available disk bandwidth」，固定間隔的 checkpoint 會讓復原時間無界〔Tachyon 原文 p.2〕；背景 checkpoint 靠的是突發之間的空閒〔Tachyon 原文 p.6〕。
   - TPP：回收比配置慢〔TPP 原文 p.7〕。
   - Gill：「idle time should never be considered free」〔Gill 原文 p.3〕。
6. **C4 的證據**：
   - Nomad：工作集大於快的層時，「access pages directly from their initial placement, completely disabling page migration」〔Nomad 原文 p.14〕；搬移進行中比不搬還差〔Nomad 原文 p.3〕。
   - Sibyl：快的層＝10% 工作集〔Sibyl 原文 p.3〕；兩裝置延遲差距越大，避開逐出的好處越大〔Sibyl 原文 p.9〕。
   - Flashield：DRAM 太小，延後觀察就失效〔Flashield 原文 p.12〕。
   - CacheLib：只有 14% 的物件在 DRAM 期間被讀或寫過〔CacheLib 原文 p.15〕。
7. **C5 的證據**：
   - Karma：下層缺少 block 的屬性〔Karma 原文 p.2〕。
   - Multi-stream：裝置很難可靠地推出 hotness〔Multi-stream 原文 p.2〕。
   - TPP：配置時就知道頁的類型〔TPP 原文 p.8〕。
8. **C6 的證據**：
   - Multi-stream：GC 讓吞吐掉約 56%〔Multi-stream 原文 p.1–2〕。
   - FairyWREN：2 TB QLC 要撐 6 年，只能寫 14 MB/s，等於可用寫入頻寬的 0.09%〔FairyWREN 原文 p.2–3〕。
   - CacheLib：全部收會比壽命允許的寫入率高 50%〔CacheLib 原文 p.15〕。
   - Flashield：60.6% 的寫入之後從沒被讀〔Flashield 原文 p.5〕。
9. **C7 的證據**：
   - Jouppi：「write-around performs well when the data being written by the processor is not read by it soon or ever」〔Jouppi WRL 91/12 原文 p.29（印刷 p.19）〕。
   - Ziggurat：大的非同步寫入直接走 disk〔Ziggurat 原文 p.5〕。
   - Tachyon：最大的 dataset 直接寫 disk〔Tachyon 原文 p.6〕。
10. **C9 的證據**：
    - Nectar：最終結果「we get it for free」，因為它本來就要寫出去；中間結果要多付 I/O，所以有門檻〔Nectar 原文 p.8〕。
    - FairyWREN：GC 本來就要寫，順便 admit 新物件「for free」〔FairyWREN 原文 p.3〕。

**沒有找到的東西**：在讀過的 Lit-B 文獻裡，**沒有一篇**在「其他都相同、只把同一條規則延後到逐出時才套用」的條件下，證明寫入時決定比延後版好 ≥5%，除非是傳輸受限（C1）。最接近 10 §2 延後測試的是 Gill 的 PROMOTE 對 DEMOTE：頻寬不限時差距 ≤5%。

### 1.1 依主題的回答

| 主題 | 文獻說寫入時決定在什麼條件下較好 | 對應 H | 延後版或背景版追得上的反例 |
|:--|:--|:--|:--|
| **多層快取**（exclusive、DEMOTE、PROMOTE、Karma、ULC） | 層間頻寬緊時，在填入時決定擁有者省一半頻寬〔Gill p.3、p.13〕；應用的 hint 只有寫入者有〔Karma p.2、p.5〕 | H12、H3、H4 | 頻寬不限時 PROMOTE 只好 ≤5%〔Gill p.13〕；eviction-based（延後）placement 比 access-based 命中率最多高 500%〔Chen p.2〕；DEMOTE 在頻寬夠時有 1.7–7.5× 的好處，而且對頻寬不敏感〔Wong p.6、p.8〕。ULC 未讀原文 |
| **Flash 快取 admission**（Flashield、Kangaroo、CacheLib、FairyWREN） | 快的層停留時間短、觀察不到時，要用 admission 當下就有的歷史特徵〔CacheLib p.15〕；寫入額度緊時，admission 是首要手段〔CacheLib p.15、FairyWREN p.2–3〕 | H1、H8、E6 | **主流是延後**：先放 DRAM 觀察再決定〔Flashield p.3〕；先放 log 再攢批次〔Kangaroo p.7〕；新物件一律先進 hot subset，每做 n 次 nest packing（把 GC 和 admission 合在一起的操作）才依熱度重新分 hot／cold〔FairyWREN p.9〕。在 flash 上、又沒改過的就不重寫〔CacheLib p.8〕 |
| **Tiered memory**（TPP、HeMem、Memtis、Colloid、Nimble、Nomad） | 依類型在配置時就放 CXL〔TPP p.8〕；工作集大於快的層時，初始放置最重要〔Nomad p.3、p.14〕 | H3、條件 2 | **背景水位線**解決突發配置〔TPP p.7–8、p.10–11〕；最佳分配取決於執行時的帶負載延遲〔Colloid p.1–2〕；搬移機制可以快 15 倍以上〔Nimble p.1〕；非 exclusive 讓 clean 頁免費 demote〔Nomad p.2〕。HeMem、Memtis 未讀原文 |
| **寫入策略**（write-allocate、write-through、write-back） | write-around 適合「寫了不會很快被讀」的資料〔Jouppi WRL p.29〕；同步寫的小資料放快的層、非同步的大寫入繞過快的層〔Ziggurat p.5〕 | S5 的理由；H10 | 多數程式較常讀剛寫的資料，write-validate（配置）普遍比 write-around 好〔Jouppi WRL p.29〕；寫穿＋immutable＝逐出免費〔Nomad p.2、CacheLib p.8〕 |
| **以重算代替儲存**（Tachyon、Checkmate、Nectar；materialized view 未讀） | 決定點被峰值逼在寫入當下〔Checkmate p.1〕；寫入時就知道的大小，讓最大的 dataset 直接寫 disk〔Tachyon p.6〕；本來就要寫的輸出，順便快取不必另外付成本〔Nectar p.8〕 | H6、H8、H2 | 能重算，就可以在背景、低優先做持久化〔Tachyon p.5〕；Nectar 空間沒壓力時積極快取，之後用 cost/benefit（大小 × 閒置時間 ÷ 使用次數 × 計算時間）在背景回收〔Nectar p.8–9〕 |
| **資料壽命或溫度在寫入時決定**（multi-stream、Sibyl、TPP 類型、Ziggurat） | 寫入者知道壽命，裝置只能猜〔Multi-stream p.2〕；把之後的逐出成本放進寫入時的目標函數〔Sibyl p.5–6、p.9〕 | H1、E6、H8 | Sibyl 贏背景版 HPS 23–46%，但兩者用的規則不同，不能歸因於「寫入時」〔判讀〕；stream 不能控制 GC，裝置層寫入放大仍在〔FairyWREN p.6〕 |
| **Linux 機制**（kswapd、dirty writeback、O_DIRECT、mlock、NUMA） | 見 §3 | H13、E1–E8 | kswapd 的水位線就是 OS 內建的背景版；`watermark_scale_factor` 可以為配置突發調高〔`vm.rst`〕 |

---

## 2. 延後搬或背景搬追得上的證據（對本研究的威脅）

| # | 證據 | 威脅哪個 H 或主張 |
|:--|:--|:--|
| T1 | **傳輸免費時，放置時決定只好 ≤5%**：PROMOTE 對 DEMOTE 在 P1 上最多好 4%（LRU）／5%（ARC），其他 trace 平均好 0.3%／1.5%〔Gill 原文 p.13〕 | 整體：「寫入時依位置放」。這是 08 結果的經典先例 |
| T2 | **延後放置比存取時放置好**：命中率最多高 500%，OLTP 交易率 1.2 倍。原因是存取時放會和上層重複〔Chen 原文 p.2〕 | 整體 |
| T3 | **背景、低優先、可放棄的搬移取代同步搬移**，解決「miss 等 demote」的問題〔Chen 原文 p.9〕；Tachyon 背景 checkpoint，不擋寫入〔Tachyon 原文 p.5〕 | H3（背景版）、H0 |
| T4 | **背景水位線**：配置與回收的水位線分開，local 配置速率提高 1.6 倍；不分開時，困在 CXL 的頁占 55% 流量，吞吐掉 12%〔TPP 原文 p.10–11〕 | H3（10 §1.3 的 S4W 對照組，文獻上是有效的） |
| T5 | **寫穿（non-exclusive）讓 clean 資料免費逐出**〔Nomad 原文 p.2〕；在 flash 上又沒改過就不重寫〔CacheLib 原文 p.8〕。KV 永遠 clean | 08 的「hold 時寫穿最快」有 OS 的對應 |
| T6 | **延後才看得到資訊**：Flashield 延後觀察，flash 寫入放大的中位數從 2.85（RIPQ）降到 0.54〔Flashield 原文 p.11〕；Kangaroo 延後攢批次，n＝2 時寫入率只剩 22.8%〔Kangaroo 原文 p.7〕 | H8（「會不會回來」適合延後觀察，不適合寫入時猜） |
| T7 | **最佳分配取決於執行時的負載**：default tier 在負載下可以比 alternate tier 慢 2.5 倍；靠持續調整才能接近最佳〔Colloid 原文 p.1–2〕 | S5 的 b 是寫入時算的，讀取時的頻寬不同就過時；Cake 在讀取時調整，S4B 用的也是當下的值 |
| T8 | **搬移可以變便宜**：Nimble 讓原始搬移吞吐提高 15 倍以上，原本可用記憶體頻寬最多有 95% 沒用到〔Nimble 原文 p.1〕 | N2：搬移越便宜，延後版越強 |
| T9 | **OS 已經內建背景版**：kswapd 依 `watermark_scale_factor` 決定多早醒來回收；allocstall 多時建議調高〔`vm.rst`〕 | H3：背景版不需要新機制 |
| T10 | **能重算，就能延後**：持久化不必同步〔Tachyon 原文 p.5〕；reload「沒做也完全沒關係」〔Chen 原文 p.9〕 | KV 的「丟了可以重算」讓背景版、甚至直接丟都合理（和 H0 報告「真實系統預設是丟」一致） |

**反方向（延後版的弱點）**：
- 收斂需要時間：TPP 在熱集合改變後要數百秒〔Colloid 原文 p.12〕。
- 背景版要有空閒：Tachyon 靠突發之間的空檔〔Tachyon 原文 p.6〕；Gill 說 idle time 不是免費的〔Gill 原文 p.3〕。
- 搬移進行中比不搬還差〔Nomad 原文 p.3〕。

這些都指向 **H3（沒有空閒時間）** 是 N3 唯一還站得住的方向。

---

## 3. 10 §3E 的 E1–E10：文獻怎麼說、值不值得在 MI300X 上量

「意義」和「值不值得量」兩欄都是〔判讀〕。

| # | 機制 | 文獻或文件怎麼說 | 對「寫入時 vs 延後」的意義 | 值不值得在 MI300X 量 |
|:--|:--|:--|:--|:--|
| E1 | page cache 與 O_DIRECT | 見表後說明 E1 | buffered 寫入時，「寫完」不等於離開 DRAM，雙重快取會吃掉 CPU 層的空間（H13）。第一階段已經用 O_DIRECT 或 `posix_fadvise`（`01_os_mapping.md` §5） | **值得，但只查真實系統的路徑**（vLLM 0.28 fs 層、LMCache disk 有沒有用 O_DIRECT）。半天，和 E2 合成一個 run |
| E2 | dirty writeback 的節奏 | 見表後說明 E2 | 在 OS 層，dirty 超過 `dirty_ratio` 時寫入者自己被節流。如果寫入在推論執行緒上，這就是 OS 版的 hold | **值得，便宜**。用 buffered 和 O_DIRECT 各寫 64 MiB chunk，同時記錄 `/proc/meminfo` 的 Dirty、Writeback |
| E3 | pinned memory | 見表後說明 E3 | 延後搬需要暫存區時，pinned 空間會和 CPU 層搶 | **低到中**。量 `hipHostMalloc` 的時間對大小、`ulimit -l` 的上限，半小時。不會直接決定 S5 的成敗 |
| E4 | NUMA、多 GPU 爭用 | Colloid：多個 in-flight 請求下，延遲比空載高很多；default tier 可以比 alternate tier 慢 2.5 倍〔Colloid 原文 p.1–2〕。Gill、Wong：頻寬決定延後搬划不划算〔Gill p.13–14、Wong p.7–8〕 | 條件 1（快的層慢）在多 GPU 下可能自然成立；C1、C9 的 N2 也只在這裡成立。**但**負載會變時，讀取時調整（Cake）比寫入時固定的分配更能跟上（T7） | **高**。這是 N2 在 KV 上唯一可能成立的地方。先量「N 個 GPU 同時 D2H／H2D 時，每張卡分到的 host 頻寬」，再決定要不要模擬 |
| E5 | PCIe 讀寫同時 | Gill：DEMOTE 讓層間流量加倍〔Gill 原文 p.3、p.13〕；Chen：同步 demote 擋住 miss〔Chen 原文 p.9〕 | 寫穿和延後搬的寫入（D2H 或 host→NVMe）會和回來時的載入（H2D）搶頻寬 | **先看已有的 `calib_c0_duplex.csv`，不必新量**。有明顯干擾才放進模擬 |
| E6 | SSD 內部 | 見表後說明 E6 | 只有 SSD 寫入額度有約束時，寫入量（H1）才是一級成本 | **低，先看 H1**。H1 活下來才做。跑 fio 到掉崖要數小時，而且會磨損共用機器的 SSD |
| E7 | NFS 的 close-to-open | 見表後說明 E7 | 08 的 G1 用 NFS。hold 的長度可能取決於 close 或 fsync，不是 write | **中**。fio 加 strace 半天。如果之後還用 NFS 當慢的層就要做 |
| E8 | 記憶體壓力 | TPP 的水位線〔TPP p.7–8〕；Nomad 的 thrashing〔Nomad p.3〕；Colloid：負載變得比系統收斂快時，tiering 會失效〔Colloid p.11–12〕；`vm.rst`：allocstall 和 `kswapd_low_wmark_hit_quickly` 表示空閒頁不夠應付配置突發 | 預期對延後版有利（看得到最新狀況），當作反證 | **值得，被動記錄就好**：每次 run 順便記 `/proc/meminfo` 和 `/proc/vmstat` 的 allocstall，幾乎零成本 |
| E9 | GPU 直接寫 SSD | 見表後說明 E9 | **這是 Lit-B 找到最像「延後做不到」的 N2**（C9）：寫入時就決定前段不進 CPU，才能走 GPU→SSD 直寫，不占也不經過 CPU DRAM。S4B 的資料已經在 host，第二段只能走 host→NVMe | **中到高（新穎性高，工程風險高）**。先花半天確認：MI300X 機器上有沒有 hipFile、本地 NVMe 能不能走 fastpath、vLLM 或 harness 能不能接。可行才排實驗 |
| E10 | 經典 OS 理論 | 見 §1、§2 | 結論：延後版在傳輸免費時追得上（T1）；寫入時決定只在 C1、C3、C5、C9 成立時有機會 | 文獻，已完成 |

**表後說明**（對應表中的「見表後說明 EN」）：

- **E1**：
  - open(2)：O_DIRECT「Try to minimize cache effects」；本身不保證 O_SYNC 的持久性；避免對同一個檔案混用 O_DIRECT 和 buffered I/O。
  - NFS 上，O_DIRECT 只繞過 client 端的 cache，server 仍可能 cache；client 會要求 server 同步寫，小 I/O 時表現差。
  - Ziggurat 刻意讓大的非同步寫入走 DRAM page cache 再寫回 disk〔Ziggurat 原文 p.5〕。
- **E2**：`vm.rst`：
  - `dirty_background_ratio`：達到時，背景 flusher 開始寫回；
  - `dirty_ratio`：達到時，「a process which is generating disk writes will itself start writing out dirty data」；
  - `dirty_expire_centisecs`、`dirty_writeback_centisecs`：決定多舊的 dirty 資料要寫、多久醒來一次。
- **E3**：
  - mlock(2)：沒有特權的行程，鎖定量受 RLIMIT_MEMLOCK 限制。
  - ROCm HIP 文件：pinned 記憶體的頻寬最多是 pageable 的 3 倍；缺點是「reduced availability of RAM for other processes」。
  - 官方文件沒寫配置要花多久；只有 NVIDIA 論壇的二手說法，未查證。
  - Chen：DEMOTE 要先騰出 buffer，讀取才能送出〔Chen 原文 p.9〕。
- **E6**：
  - Multi-stream：SSD 老化時，GC 讓 Cassandra 的最差吞吐掉約 56%〔Multi-stream 原文 p.1–2〕。
  - FairyWREN：QLC 的寫入額度只有 14 MB/s〔FairyWREN 原文 p.2–3〕。
  - Kangaroo：每天 3 次 device-writes〔Kangaroo 原文 p.1〕。
  - CacheLib：flash 超額配置 50%〔CacheLib 原文 p.12〕。
- **E7**：
  - nfs(5)：關檔時，client 把還沒送出的修改寫回 server（close-to-open）；開檔時一定送 GETATTR 或 ACCESS 檢查。
  - open(2)：NFS 上的 O_DIRECT 是同步的，server 仍可能 cache。
- **E9**：
  - ROCm 文件：hipFile 是「AMD's Infinity Storage library that provides direct-to-GPU I/O without requiring a host-side buffer」；無法走直通路徑時，自動退回 POSIX I/O。文件版本 0.5.0。
  - 有沒有 GA、支援哪些 GPU 與檔案系統，**未查證**。二手來源（套件 metadata）說是 early access。

---

## 4. 10 §5.6 中 Lit-B 候選的查證結果

| 候選（10 寫的） | 查證結果 | 依據 | 讀了原文？ |
|:--|:--|:--|:--|
| Wong & Wilkes "My cache or yours?"（DEMOTE，ATC'02） | **正確**。Theodore M. Wong、John Wilkes，USENIX ATC 2002，pp.161–175 | PDF 首頁；Gill 的參考文獻 [33] | 是 |
| Gill "On multi-level exclusive caching"（FAST'08） | **正確**。全名：*On Multi-level Exclusive Caching: Offline Optimality and Why promotions are better than demotions*；Binny S. Gill；FAST '08 pp.49–64 | PDF 頁尾 | 是 |
| Karma（FAST'07） | **正確**。Gala Yadgar、Michael Factor、Assaf Schuster；*Karma: Know-it-All Replacement for a Multilevel cAche*；FAST '07 pp.169–184 | PDF 頁尾 | 是 |
| ULC | Song Jiang、Xiaodong Zhang；*ULC: A File Block Placement and Replacement Protocol to Effectively Exploit Hierarchical Locality in Multi-Level Buffer Caches*；**ICDCS 2004**，pp.168–177 | Gill 的參考文獻 [19]；網路搜尋 | **未讀原文** |
| Flashield（NSDI'19） | **正確**。Eisenman、Cidon、Pergament、Haimovich、Stutsman、Alizadeh、Katti；NSDI '19 pp.65–78 | PDF 首頁與頁尾 | 是 |
| Kangaroo（SOSP'21） | **正確**。McAllister、Berg、Tutuncu-Macias、Yang、Gunasekar、Lu、Berger、Beckmann、Ganger；SOSP '21 pp.243–262；DOI 10.1145/3477132.3483568 | PDF | 是 |
| CacheLib（OSDI'20） | **正確**。Berg、Berger、McAllister、Grosof、Gunasekar、Lu、Uhlar、Carrig、Beckmann、Harchol-Balter、Ganger；*The CacheLib Caching Engine: Design and Experiences at Scale*；OSDI '20 pp.769–786 | PDF | 是 |
| FairyWREN（OSDI'24） | **正確**。McAllister、Wang、Berg、Berger、Amvrosiadis、Beckmann、Ganger；*FairyWREN: A Sustainable Cache for Emerging Write-Read-Erase Flash Interfaces*；OSDI '24 pp.745–764 | PDF | 部分（p.2–3、p.6–7、p.9） |
| TPP（ASPLOS'23） | **正確**。Al Maruf 等 10 人；ASPLOS '23 Vol. 3，pp.742–755，DOI 10.1145/3582016.3582063 | venue 與頁碼來自 NSF PAR 和 arXiv DOI（網路搜尋），Colloid 的參考文獻 [35] 也這樣寫 | 是（arXiv v2） |
| HeMem（SOSP'21） | **正確**。Amanda Raybuck、Tim Stamler、Wei Zhang、Mattan Erez、Simon Peter；*HeMem: Scalable Tiered Memory Management for Big Data Applications and Real NVM*；SOSP 2021。pp.392–407 只來自網路搜尋，未查證 | Colloid 的參考文獻 [48] | **未讀原文**（ACM 擋下載，NSF PAR 連線被拒） |
| Memtis（SOSP'23） | **正確**。Taehyung Lee、Sumit Kumar Monga、Changwoo Min、Young Ik Eom；*MEMTIS: Efficient Memory Tiering with Dynamic Page Classification and Page Size Determination*；SOSP '23。pp.17–34 只來自網路搜尋，未查證 | Colloid 的參考文獻 [29]、Nomad 的參考文獻 [37] | **未讀原文** |
| Colloid（SOSP'24） | **正確**，但論文標題不是 Colloid：*Tiered Memory Management: Access Latency is the Key!*；Midhul Vuppalapati、Rachit Agarwal；SOSP '24；DOI 10.1145/3694715.3695968 | PDF | 是 |
| Tachyon（SoCC'14） | **正確**。Haoyuan Li、Ali Ghodsi、Matei Zaharia、Scott Shenker、Ion Stoica；SoCC '14；DOI 10.1145/2670979.2670985 | PDF | 是 |
| Checkmate（MLSys'20） | **正確**。Paras Jain、Ajay Jain、Aniruddha Nrusimha、Amir Gholami、Pieter Abbeel、Kurt Keutzer、Ion Stoica、Joseph E. Gonzalez；MLSys 2020 | PDF（arXiv v3 首頁有 MLSys 字樣） | 是 |
| （任務另列）Nimble | **正確**。Zi Yan、Daniel Lustig、David Nellans、Abhishek Bhattacharjee；ASPLOS '19 | PDF | 只讀摘要（p.1） |
| （任務另列）write-allocate | Norman P. Jouppi，*Cache Write Policies and Performance*：ISCA 1993 pp.191–201（ISCA'93 目錄，網路搜尋）。**我讀的是 1991 年 12 月的 WRL Research Report 91/12**，內容可能和 ISCA 版不同 | 網路搜尋；WRL PDF | 是（WRL 版） |
| （任務另列）multi-stream SSD | Jeong-Uk Kang、Jeeseok Hyun、Hyunjoo Maeng、Sangyeun Cho；HotStorage '14。venue 是從 USENIX 的 URL 路徑判斷的 | PDF | 是 |
| （新增）Nomad | OSDI '24（USENIX 頁面，網路搜尋）；Xiang、Lin、Deng、Lu、Rao、Yuan、Wang | 網路搜尋；arXiv v2 | 是 |
| （新增）Chen et al. eviction-based placement | USENIX ATC 2003，pp.269–282 | PDF 首頁；Gill 的參考文獻 [6] | 是 |
| （新增）Sibyl | ISCA '22 | PDF 首頁 | 是 |
| （新增）Nectar | Gunda、Ravindranath、Thekkath、Yu、Zhuang；OSDI 2010。venue 是從 URL 路徑與致謝中的「OSDI review committee」判斷的 | PDF | 部分（p.1、p.8–9） |

**10 §5.6 的寫法和查證結果沒有出入**：所有 venue 和年份都對。只有兩點要補：
- Colloid 的論文標題不含「Colloid」；
- HeMem、Memtis、ULC 沒有讀原文，引用時只能用書目，不能引用內容。

---

## 5. 建議（三點）〔判讀〕

1. **模擬器先補兩個文獻上已證明有效的對照組，判準改成量「排空餘裕」**：
   - 兩個對照組：
     - **背景水位線**（TPP 式，配置與回收的水位線分開；11 的 S4W 已經排了）；
     - **「放不下就丟」**（H0 報告說這是真實系統的預設；Kangaroo、Chen 也說 cache 本來就可以丟）。
   - 文獻的共同結論是：寫入時決定只在**長期寫入速率超過背景排空頻寬**（Tachyon p.2、TPP p.7）或**層間傳輸是瓶頸**（Gill p.13）時才會贏。所以 H3 的模擬要明確報告每個設定的「寫入速率 ÷ SSD 排空頻寬」，只在比值 ≥1 的區域找 S5 的勝場；比值 <1 時，預期背景版追平。
2. **E9 和 E4 合成一條「KV 特有的 N2」路線，先做可行性檢查**：寫入時決定前段不進 CPU，才能用 hipFile 把前段從 GPU 直接寫到 SSD，不占也不經過 CPU DRAM；延後版的資料已經在 host，做不到。這是 Lit-B 找到唯一「延後版結構上用不了」的資源優勢（C9，對應 Nectar 的「順便寫不必另付成本」）。但它只在 host DRAM 容量或頻寬稀缺時才有價值，所以要和 E4（多 GPU 同時卸載時，每張卡分到的 host 頻寬）一起量。先花半天確認 MI300X 機器上 hipFile 和 NVMe fastpath 能不能用，不能用就停。
3. **「依位置在寫入時放」不要再加新條件掃描；把資源轉到真正的 N1，並在論文裡用 Gill FAST'08 當負面結果的先例**：
   - OS 文獻裡，寫入時決定的結構性優勢來自**資訊不對稱**（Karma p.2、Multi-stream p.2）。KV 的位置資訊不是這種，S4B 一樣看得到。
   - 有希望的方向是 H7（SSM 狀態被覆蓋）、H11（prefill 節點傳完就丟）、H4（應用層知道共享）。
   - 若最後寫成負面結果或測量型論文，Gill（頻寬不限時 PROMOTE 對 DEMOTE ≤5%，頻寬受限時才大贏）和 Chen（延後放置反而更好）可以當作「延後版在傳輸便宜時追得上」的經典支撐。
