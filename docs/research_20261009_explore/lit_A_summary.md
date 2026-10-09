# Lit-A 摘要：LLM KV 系統裡「寫入時決定」的文獻（2023–2026）

> **一句話**：「寫入時決定」在 2025–2026 年已經不是空白——Lachesis（寫入時依壽命放 HBM／HBF）、Krul（寫入時決定每層前段不存、讀取時重算）、TierKV（prefill 後依位置分層）、Marconi＋Sparse Prefix Caching（寫入時決定存哪些 SSM 狀態）都做了。但**幾乎沒有一篇和「延後版」比過**；唯一量過「同一條准入規則在不同容量下」的 EfficientAgent 顯示它會反轉（容量不夠時 −36% 重算，夠時 4.3 倍重算）。最站得住的新方向是 **N1＋N3 的交集：壓力下「寫不進去的 KV 就沒了」，該丟哪一段由位置決定**。

**日期**：2026-10-09
**執行者**：Lit-A（文獻 agent，不碰 GPU）
**上游**：[10_breakthrough_plan.md](../phase1_20261008/10_breakthrough_plan.md) §2、§4、§5；[11_round1_plan.md](../phase1_20261008/11_round1_plan.md)
**卡片**：`cards/A_*.md`，共 20 張（15 張新論文、5 張是 `03_paper_map.md` 已收論文的「補充」）

**怎麼讀**
- 〔原文 p.X〕＝打開原文讀到的，頁碼是 **PDF 實體頁**（不是期刊印刷頁）。
- 〔判讀〕＝我的推論，原文沒有直接寫。
- 「未讀原文，只看摘要」＝只讀了 arXiv 摘要或 web 搜尋結果。
- 本檔**沒有任何實測數字**。所有數字都是原論文自己報的。

---

## 0. 結論（先講）

1. **最被支持的 H：H1（寫入量／壽命）**，但要改成「寫入額度是稀缺資源」的版本。Lachesis、HBF LRU-K、Dynamo KVBM 的磁碟過濾都把「少寫」當主目標，而且做得到數倍壽命〔A_Lachesis p.9；A_HBF_LRUK p.1；§4 KVBM〕。但它們的對手都是「不看位置的頻率／壽命准入」，S5 要贏過這些，不能只跟全部寫比。
2. **已被別人做掉（或延後版已存在）的 H：H6、H7、H9、H10、H14**。
   - H6：HCache（依層）、HybridServe（依比例）已做「寫入時選 KV／activation」；HCache 量過「依 token 位置切」比「依層切」慢 7–12%〔A_HCache_補充 p.12〕。
   - H7：Marconi（存哪些條目）＋Sparse Prefix Caching（條目內存哪些位置，DP 最佳）〔A_Marconi_補充、A_SparsePrefixCaching〕。另外〔判讀〕Cake 還原整段 session 時**不需要**中間位置的 SSM 狀態，H7 只在分叉型負載成立。
   - H9：Jenga 在 GPU 上已用「視窗外 token 優先淘汰」〔A_Jenga p.4〕，延後版存在。
   - H10：Strata 在寫入路徑上免費轉成 page-first（TTFT 2.1 倍）〔A_Strata_補充 p.13〕；IMPRESS 在背景重排〔A_IMPRESS p.9〕，延後版存在。
   - H14：EvicPress「層滿時再加重壓縮」就是延後版（03 已收）；CacheGen 寫入時存多個等級、讀取時選。
3. **對 S5 新穎性最直接的威脅：Krul**（arXiv 2507.08045，引用 Cake）。它在寫入時依「重算 vs 載入平衡」決定每層前段**不存**、讀取時重算〔A_Krul p.8–9〕。差別只剩：單層 vs 兩層、有損（跨層共享）vs 無損、依層校準 vs 依位置。**TierKV**（EuroSys'27）也在 prefill 後依位置分層（但尾段放 flash、單請求、有損）〔A_TierKV p.4〕。
4. **最重要的新方向**（§3 第 1 點）：真實系統在 CPU 滿時**預設是丟掉或截斷**（Screen 的 H0 文件），HyMCache 在讀取壓力下跳過 >15% 的寫入〔A_HyMCache p.10〕，EfficientAgent 刻意拒絕寫入〔A_EfficientAgent p.5〕。沒寫進去的 KV 在 GPU 釋放後就**沒了**（N1＋N3，延後版補不回來）。而 LMCache 預設截斷時「只存前面幾個 chunk」——在 Cake 讀取下，應該反過來留尾段〔判讀〕。
5. **背景版與讀取時版都有現成出處**：CachedAttention 有水位線式的背景淘汰（可用記憶體低於門檻就往磁碟搬）和 job-queue 預取〔A_CachedAttention_補充 p.7〕；MORI 利用 agent 工具呼叫的空檔卸載〔A_MORI p.1–2〕。所以 N3 不能用 agent 負載論證，除非是忙碌階段。

---

## 1. 對 H0–H14：支持、威脅、已有人做過

| H | 支持 | 威脅（更簡單的做法／延後版已存在） | 已有人做過 | 小結〔判讀〕 |
|:--|:--|:--|:--|:--|
| **H0** hold 是真的 | Strata：「只在淘汰時寫……GPU 有壓力時可能把阻塞加到關鍵路徑上」〔[A_Strata_補充](cards/A_Strata_補充.md) p.9〕 | 系統都設計來**避免**擋住：CachedAttention 留 HBM 寫入緩衝、水位線保持 host 緩衝可用〔[A_CachedAttention_補充](cards/A_CachedAttention_補充.md) p.6–7〕；HyMCache 緩衝滿就跳過寫入〔[A_HyMCache](cards/A_HyMCache.md) p.10〕 | — | 文獻和 Screen 的原始碼結論一致：真實系統「丟」而不是「等」（`H0_offload_blocking.md` §0）。hold 模型應改成「寫入中占空間＋放不下就丟」 |
| **H1** 寫入量 Pareto | Lachesis：寫入時依壽命放置，HBF 壽命 1.19–3.13 倍〔[A_Lachesis](cards/A_Lachesis.md) p.1、p.9〕；HBF LRU-K：壽命約 1 年→>6 年〔[A_HBF_LRUK](cards/A_HBF_LRUK.md) p.1〕；Dynamo KVBM 磁碟過濾「為了延長 SSD 壽命」預設開（§4）；EfficientAgent：寫進 host 的 chunk 79.2% 在重用前就被淘汰〔[A_EfficientAgent](cards/A_EfficientAgent.md) p.2〕 | 頻率准入（KVBM ≥2、LRU-K、Strata selective）和容量條件准入（EfficientAgent：10 GiB 時 host 寫入少 73%、重算少 14%〔p.8〕）都不看位置就能少寫 | **Lachesis**（寫入時、依壽命、目標是 flash 壽命） | H1 活著，但對手要加「頻率准入」與「容量條件准入」；指標要用 SSD 寫入量／壽命，而不只是 TTFT |
| **H2** 重算便宜的模型 | EfficientAgent：offload 值不值得取決於「每 host 頻寬位元組的算力」，H800 比 H20、3090 高 6.7–7.0 倍；同一負載在 3090 變快、H800 變慢〔[A_EfficientAgent](cards/A_EfficientAgent.md) p.1、p.4〕（MoE 30B-A3B） | — | Krul 用校準求最佳重算比例 r_c〔[A_Krul](cards/A_Krul.md) p.8〕 | Lit-A 範圍內只有間接證據；κ 表要靠 Screen |
| **H3** 沒有空閒時間 | HyMCache：讀取優先讓寫入緩衝排不空，>15% 插入被跳過〔[A_HyMCache](cards/A_HyMCache.md) p.10〕；EfficientAgent：16 個 agent 併發時重用間隔中位 9.0 s〔p.2〕 | **背景版已存在**：CachedAttention 水位線〔p.7〕、Pensieve GPU 空位 <25% 提前換出（`related_papers_update.md` a2）；MORI 利用工具呼叫空檔〔[A_MORI](cards/A_MORI.md) p.1–2〕 | — | S4W（背景版）必須有；agent 負載有天然空檔，H3 要找「忙碌階段」或高讀取壓力 |
| **H4** 一寫多讀 | Marconi：「純輸入」prefix 第 3 次出現才受益，因為第 2 次才被認出〔[A_Marconi_補充](cards/A_Marconi_補充.md) p.6〕——這就是頻率感知要付的代價 | 同一段原文說：這些 prefix 被很多請求共用，「錯過一次的影響可忽略」〔p.6〕；IMPRESS 用「頻率×重要度」准入〔[A_IMPRESS](cards/A_IMPRESS.md) p.10〕 | MPIC 在**上傳時**就算好圖片 KV 寫磁碟〔[A_MPIC](cards/A_MPIC.md) p.6〕 | H4 的上限大約是「每個 prefix 少一次 miss」，讀取次數越多越不值錢；只有讀 2–3 次的情境有看頭 |
| **H5** VLM／影片 | Jenga 把 vision embedding cache 當獨立層型管理〔[A_Jenga](cards/A_Jenga.md) p.3〕 | MPIC：上傳時全寫、讀取時算與載並行〔p.6〕 | **MPIC**（寫入時預算＋位置無關格式，有損） | 對照組至少要有 MPIC 式「上傳時全寫」；格式選擇（embedding vs KV）跨層尚無人做（未查到） |
| **H6** 寫入時選格式 | HybridServe：寫入時依比例存 KV 或 activation，讀取時混合重算與載入〔[A_HybridServe](cards/A_HybridServe.md) p.9〕 | HCache：token-wise（依位置）切法比 layer-wise 慢 12%（對齊後仍慢 7%）〔[A_HCache_補充](cards/A_HCache_補充.md) p.12〕；GQA 在範圍外〔p.13〕 | **HCache、HybridServe**（都 MHA）；Apt-Serve（未讀原文，只看摘要）；SuffixReplay（混合模型，有損）〔[A_SuffixReplay](cards/A_SuffixReplay.md)〕 | 新穎性很薄；依位置那一維有負面證據 |
| **H7** SSM 檢查點 | Marconi：中間狀態只能在 prefill 時拿到，否則要兩段 prefill〔p.5–6〕；Sparse Prefix Caching：FLA kernel 會在 block-64 位置回傳狀態〔[A_SparsePrefixCaching](cards/A_SparsePrefixCaching.md) p.8〕 | SuffixReplay：不存檢查點，改存稀疏 hidden state 讀取時重播（有損，品質 91.4–100%）〔p.1〕；Sparse Prefix Caching 自己說 append-only chat 只需最後一個狀態〔p.2、p.11〕 | **Marconi**（MLSys'25）＋**Sparse Prefix Caching**（arXiv 2605.05219）；SGLang 的規則：chunk 邊界、請求結束、第一次分叉；狀態留在 HBM slot pool〔A_SuffixReplay p.8–9〕 | 「寫入時決定存哪些位置」已做掉；10 §3A 的動機（Cake 需要會合點的 SSM 狀態）〔判讀〕不成立；剩下「多層放檢查點」 |
| **H8** 預測會不會回來 | Lachesis：harness 在寫入前就知道 segment 壽命，reasoning 占一輪寫入 66–83% 但只活一輪〔p.2〕；MORI：閒置程度可用歷史估〔p.2〕 | EfficientAgent：**同一條准入規則**容量不夠時 −36% 重算，容量夠時 4.3 倍重算〔p.7〕；要用容量條件准入 | EfficientAgent（容量條件准入）、HBF LRU-K（第 K 次才寫）、Lachesis（壽命）、LPC（03） | oracle 上限的對照組要用「容量條件准入」，不能只跟全部寫比 |
| **H9** 滑動視窗層 | — | Jenga：視窗外 token 優先淘汰（延後版）〔p.4〕 | Jenga（GPU 內） | 寫入時版本只能省寫入量，省不了容量；整段還原時視窗外 KV 本來就用不到 |
| **H10** 寫入布局 | Strata：page-first 布局 TTFT 2.1 倍、吞吐 1.3 倍〔[A_Strata_補充](cards/A_Strata_補充.md) p.13〕 | Strata 在寫入路徑上免費轉換〔p.7〕；IMPRESS 背景重排、不在關鍵路徑〔p.9、p.13〕 | Strata、IMPRESS、KVDrive（03） | Strata 的問題是每片幾 KB；我們每層每 chunk ≥2 MiB，可能不存在。照 10 的 fio 停損測法就能確認 |
| **H11** P/D 分離 | Mooncake 的 prefill 節點把新 KV 全部存回、逐層串流給 decode〔[A_Mooncake_補充](cards/A_Mooncake_補充.md) p.5〕——傳輸就是寫入 | Mooncake 在**排程時**決定算或抓（門檻手調）〔p.11〕；P/D 傳輸壓縮（KVServe、SCD，有損） | **沒找到**（有限搜尋） | 可能仍是空白，但只能模擬 |
| **H12** CXL 三層 | — | HyMCache：CXL-HM 裝置內 DRAM 緩衝延後刷到 SSD，「搬兩次」被裝置吸收〔[A_HyMCache](cards/A_HyMCache.md) p.6〕 | HyMCache（CXL 機櫃）；ITME、Photonic-CXL、Composable CXL（只看到標題） | 低優先 |
| **H13** page cache 偷 DRAM | DUAL-BLADE：host 記憶體 <約 6 GB 時，page cache 在背景回寫前就被迫同步淘汰，寫入卡住；decode 命中率從 42% 崩到 <1%〔[A_DualBlade](cards/A_DualBlade.md) p.3〕 | 解法是 O_DIRECT／NVMe-direct，和寫入時放置無關〔p.5〕；Screen 查到 LMCache 磁碟後端 O_DIRECT 預設關、vLLM 0.28 能用就用（`H0_offload_blocking.md`） | DUAL-BLADE（ICDCS'26，邊緣） | 前提成立（至少對 LMCache 預設）；但「更簡單的做法」就是 O_DIRECT |
| **H14** 寫入時選精度 | Krul：寫入時依注意力相似度選壓縮策略〔[A_Krul](cards/A_Krul.md) p.2〕；HA-RAG：依熱度選精度與放置（未讀原文，只看摘要） | EvicPress「層滿時加重壓縮」＝延後版（03 A2）；CacheGen 寫入時存多個等級、讀取時選（03 B） | AdaptCache、EvicPress、CacheGen（03）、TierKV（位置＋SVD）〔[A_TierKV](cards/A_TierKV.md) p.4〕 | 延後版已是發表過的系統；要老師同意（有損）之外，新穎性也薄 |

**引用 Cake 的論文**：Semantic Scholar 列 39 篇（2026-10-09 查）。和 Lit-A 相關、本次寫了卡的：Krul、TierKV。已在 03 的：LMCache、Strata、CacheFlow、KVCache in the wild、AsymCache（MSA，2606.02964）。其餘多半不相關（見附錄 A）。

---

## 2. 「寫入時才有、之後就沒有」的實例

### N1 資訊消失
| 實例 | 原文 | 延後版能不能補 |
|:--|:--|:--|
| SSM／線性注意力的中間狀態 | Marconi：狀態原地更新、無法回推；要嘛 prefill 時存，要嘛兩段 prefill〔A_Marconi_補充 p.1、p.5–6〕；FLA kernel 在 block-64 位置回傳狀態，但 API 沒開放〔A_SparsePrefixCaching p.8〕 | 不能（只能重跑 prefill） |
| 生成時的注意力權重 | Krul 在 prefill 時把注意力權重卸到 CPU 算相似度，decode 每步算完就刪；全存在 GPU 會 OOM（7B、4K token 要 30 GB）〔A_Krul p.2、p.9〕 | 不能 |
| 每層的 hidden state／activation | HCache、HybridServe、SuffixReplay 的 anchor（每次 prefill 與 decode 都要寫）〔A_SuffixReplay p.6〕 | 不能（KV 存了就回不去 hidden state） |
| prefill 的 hidden state 當預測訊號 | TierKV：「prefill 在行動端是免費的預測器；伺服器上要另外跑預測器」〔A_TierKV p.3〕 | 能，但要多付一次計算（弱 N1） |
| **CPU 滿時沒存進去的 KV** | LMCache 預設截斷 store、只存前面幾個 chunk（`H0_offload_blocking.md` LMCache 段）；HyMCache 跳過 >15% 插入〔A_HyMCache p.10〕；EfficientAgent 拒絕整個請求的寫入〔A_EfficientAgent p.5〕 | 不能（GPU 釋放後就沒了，只能重算）〔判讀〕 |

### N2 延後要多付稀缺資源
| 實例 | 原文 |
|:--|:--|
| flash／HBF 的寫入額度 | Lachesis〔p.1、p.9〕；HBF LRU-K〔p.1〕；Characterizing HBF（arXiv 2609.39131，未讀原文，只看摘要：buffered cache-aware scheduling 讓壽命 1.21→14.82 年） |
| SSD 壽命 | Dynamo KVBM 文件：「啟用磁碟卸載時，磁碟卸載過濾預設開啟，以延長 SSD 壽命」（§4） |
| host 寫入頻寬與 host 容量被一次性請求浪費 | Strata〔p.9〕；EfficientAgent 79.2% 的寫入在重用前被淘汰〔p.2〕 |
| DRAM 被 page cache 占走 | DUAL-BLADE〔p.2–3〕 |
| 「延後版要 migration」的論證 | Lachesis 說 LRU 要事後 migration 才能修正放置〔p.5〕，**但沒量** |

### N3 沒有空閒時間
| 實例 | 原文 |
|:--|:--|
| 讀取壓力下，延後的寫入被跳過 | HyMCache〔p.10〕 |
| 延後寫（write-back）在 GPU 壓力下把阻塞放上關鍵路徑 | Strata〔p.9〕 |
| **反例**：agent 有工具呼叫的空檔 | MORI〔p.1–2〕；忙碌階段是幾百毫秒的短呼叫〔p.2〕 |
| **反例**：背景水位線 | CachedAttention〔p.7〕、Pensieve（`related_papers_update.md` a2） |

### N4 寫入形式決定讀取時能做什麼
| 實例 | 原文 |
|:--|:--|
| 格式：hidden state vs KV | HCache、HybridServe〔A_HybridServe p.9〕 |
| 格式：稀疏 hidden anchor（讀取時重播） | SuffixReplay〔p.2〕 |
| 格式：pre-RoPE KV（之後可截斷） | CachedAttention〔p.8〕（反旋轉可逆，弱 N4） |
| 格式：位置無關 KV | MPIC〔p.1、p.5〕、EPIC（03 未收，ICML'25） |
| 格式：多存一份 probe head 的 key（1.2%），讀取時才能便宜地選重要 KV | IMPRESS〔p.13〕 |
| 格式：跨層共享改變重算／載入的最佳比例 | Krul〔p.2〕 |
| 布局：page-first vs layer-first | Strata〔p.7、p.13〕（寫入路徑上免費轉換） |
| 布局：依重要度重排 | IMPRESS〔p.9〕（**背景做**，延後版存在） |
| 布局：連續 LBA、NVMe-direct vs page cache | DUAL-BLADE〔p.5〕 |
| SSM 檢查點的位置 | Sparse Prefix Caching、Marconi |

---

## 3. 新發現、但 §4 沒列的方向

1. **壓力下的「位置感知寫入截斷」**〔判讀〕。
   - 現況：真實系統 CPU 滿時預設是丟或截斷（Screen 的 H0）。LMCache 預設「只存前面幾個 chunk，後面的丟掉」——因為它的重用要求 prefix 從頭連續（EfficientAgent 也是這個假設〔p.4〕）。
   - 在 Cake 讀取下，前段重算便宜、後段載入划算，應該**反過來留尾段、丟前段**。
   - 為什麼是「寫入時才有」：沒存的 KV 在 GPU 釋放後就沒了（N1），而且壓力本身就是沒空閒（N3）。延後版是「先淘汰別的 session 的前段騰出空間」（S4B 式），只有在有可淘汰的 chunk 時才做得到；LMCache 的情況正是「寫入中的 chunk 不能淘汰」（H0 文件）。
   - 對照組：LMCache 預設（留頭）、EfficientAgent 容量條件准入（整個請求不寫）、S4B（淘汰別人的前段）、全部寫＋等（`force_store_wait`）。
   - 這和 H3、H8 都有關，但 §4 沒有這一條。
2. **Agent harness 在寫入前就知道的壽命**〔判讀〕。Lachesis 指出 reasoning segment 占一輪寫入的 66–83% 卻只活一輪〔A_Lachesis p.2〕。對 CPU＋SSD 兩層：「下一輪就會被剝掉的 segment 不寫 SSD」是確定的（不是預測），是 H8 的強化版，也直接對應 N2（SSD 寫入量）。要找有 reasoning trace 的 agent 資料。
3. **Cake 打破「前綴連續」假設後，准入規則要重寫**〔判讀〕。EfficientAgent 的 Proposition 1 與規則都建立在「從第一個 chunk 開始的連續覆蓋」上〔p.4–5〕。在 Cake 下，沒有前段的後段仍然可用，所以「拒絕寫入」應該以 chunk 位置為單位，而不是整個請求。這是 §3 第 1 點在非壓力情境的版本。
4. **混合模型的 SSM 檢查點不跟著 host 層走**。SuffixReplay 指出 SGLang 把線性狀態留在 HBM 的固定 slot pool，不會隨 KV page 搬到 host〔A_SuffixReplay p.9、p.12〕。多層放檢查點（H7 的剩餘空間）有一個具體的現況缺口，但只對分叉型負載有用。
5. **把 Krul 當成 S5 的最近前作**。Krul 已經是「寫入時決定不存前段＋讀取時 Cake 式重算」〔A_Krul p.8–9〕。S5 的貢獻要寫成「兩層、無損、依位置，而且和延後版比過」，並把 Krul 的「依層金字塔、單層」當基線或至少當相關研究〔判讀〕。

---

## 4. §5.6 Lit-A 清單的查證結果

| 名稱 | 查到的作者（前幾位） | 年份、venue | 和 10 §5.6／03 的記憶是否一致 | 怎麼查的 |
|:--|:--|:--|:--|:--|
| Cake | Shuowei Jin, Xueshen Liu, Qingzhao Zhang, Z. Morley Mao | ICML 2025（PMLR 267）；arXiv 2410.03065 | ✅ | 本機 PMLR PDF p.1；arXiv API |
| HCache | Shiwei Gao, Youmin Chen, Jiwu Shu | EuroSys 2025 | ✅ | 本機 PDF p.1 頁首；arXiv comment |
| CachedAttention／AttentionStore | Bin Gao, Zhuomin He, Puru Sharma, …, Pengfei Zuo | USENIX ATC 2024 | ✅；**同一篇**：arXiv v1 標題是「AttentionStore: …」，v3 改成 CachedAttention | arXiv comment；arXiv abs v1 頁 |
| Pensieve | Lingfan Yu, Jinkun Lin, Jinyang Li | EuroSys 2025 | ✅ | PDF 頁首（arXiv 2312.05516 v3） |
| Mooncake | Ruoyu Qin, Zheming Li, Weiran He, Mingxing Zhang, Yongwei Wu, Weimin Zheng, Xinran Xu | FAST 2025，pp.155–170 | ✅ venue；⚠️ **會議版標題不同**：「Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot」；本機是 arXiv 版（2407.00079） | Bidaw FAST'26、Strata OSDI'26 的參考文獻；arXiv API |
| LMCache | Yuhan Liu, Jiayi Yao, Yihua Cheng, …, Junchen Jiang | arXiv 2510.09665（2025）；**venue 未查證**（MLSys 2026 網站有同名 oral／invited talk 頁，無法確定是論文接受） | 03 只寫 arXiv，一致 | arXiv API；WebSearch |
| CacheGen | Yuhan Liu, Hanchen Li, Yihua Cheng, Siddhant Ray, Yuyang Huang, Qizheng Zhang, … | SIGCOMM 2024 | ✅ | arXiv comment（未開 PDF） |
| CacheBlend | Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang, Kuntai Du, Shan Lu, Junchen Jiang | EuroSys 2025 | ✅ | 本機 PDF p.1 頁首 |
| IMPRESS | Weijian Chen, Shuibing He, Haoyang Qu, Ruidong Zhang, Siling Yang, Ping Chen, Yi Zheng, Baoxing Huai, Gang Chen | FAST 2025，pp.187–201 | ✅ | USENIX PDF 封面；LMCache 參考文獻 |
| Marconi | Rui Pan, Zhuang Wang, Zhen Jia, Can Karakus, Luca Zancato, Tri Dao, … | MLSys 2025 | ✅ | arXiv comment「MLSys 2025 camera-ready」 |
| Jenga | Chen Zhang, Kuntai Du, Shu Liu, Woosuk Kwon, …, Ion Stoica | SOSP 2025（web 搜尋：清華翟季冬論文頁列 SOSP'25）；arXiv 2503.18292 | ✅（但沒打開 ACM DL） | WebSearch；arXiv PDF 無會議頁首 |
| EPIC | Junhao Hu, Wenrui Huang, Weidong Wang, Haoyi Wang, Tiancheng Hu, Qin Zhang, … | ICML 2025（PMLR 267:24391–24402）；arXiv 2410.15332 | ✅（10 沒寫 venue） | WebSearch（PMLR、icml.cc 頁）；**沒讀原文** |
| SGLang HiCache | — | 系統（docs.sglang.io） | 三種策略屬實：write_through「每次存取立刻寫回下一層」、write_through_selective「存取頻率超過門檻才寫回」、write_back「上層淘汰時才寫回」；文件頁**沒寫預設**；Screen 查到程式碼預設 `write_through`（`H0_offload_blocking.md`，`memory.py:118-124`） | WebFetch 文件頁 |
| Dynamo KVBM | — | 系統（docs.nvidia.com/dynamo） | v0.9.1 guide：「啟用磁碟卸載時，磁碟卸載過濾預設開啟，以延長 SSD 壽命」「只有頻率 ≥2 的 block 才從 CPU 卸到磁碟」「頻率初始 1、命中加倍、每次衰減減 1」。新版設定文件有 per-transition 的 policy 清單（pass_all／presence／presence_lfu），預設值**未查證**（只看到 web 搜尋摘要）。03 寫「v1.5.0 起棄用」：**本次未查證** | WebFetch v0.9.1 guide；WebSearch |
| 引用 Cake 的論文 | 見附錄 A | — | — | Semantic Scholar `/paper/arXiv:2410.03065/citations` |

---

## 5. 對研究方向的建議（三點）

1. 〔判讀〕**把 hold 換成「放不下就丟」，在這個模型下測「位置感知截斷」**（§3 第 1 點）。這是目前唯一同時具備 N1（沒存就沒了）和 N3（壓力即沒空閒）、而且有真實系統預設行為（LMCache 留頭丟尾）當對手的設定。對照組：LMCache 預設、EfficientAgent 容量條件准入、S4B、`force_store_wait`。先在模擬器裡做（Sim agent），判準照 10 §2。
2. 〔判讀〕**H1 改成「SSD 寫入額度」的版本，但對手要升級**：加入頻率准入（KVBM ≥2、Strata selective 門檻 2）和容量條件准入，指標用「同 TTFT 下的 SSD 寫入量」。Lachesis 和 HBF LRU-K 說明這個指標被審稿人接受；它們都沒和延後版比，這正好是我們可以補的。
3. 〔判讀〕**H6、H7、H9、H10、H12、H14 降到最低優先**（已有人做或延後版已存在，見 §1），**相關研究必須引用 Krul、TierKV、Lachesis** 並寫清楚差別（兩層、無損、依位置、和延後版比過）。H7 只有在換成分叉型負載時才值得做，而且要先確認 Cake 在混合模型上真的不需要中間狀態（Lit-C）。

---

## 附錄 A：引用 Cake 的 39 篇（Semantic Scholar，2026-10-09）分類

| 類別 | 論文 |
|:--|:--|
| **寫了卡** | Krul（2507.08045）、TierKV（2609.21172，EuroSys'27） |
| **已在 03** | LMCache（2510.09665）、Strata（2508.18572）、CacheFlow（2604.25080；S2 有兩筆）、KVCache Cache in the Wild（ATC'25，2506.02634）、Multi-Segment Attention（2606.02964，即 03 的 AsymCache） |
| **相關，但只看摘要（未讀原文）** | SparKV（2604.21231，IoTJ：行動端逐 chunk 決定串流或本地計算，讀取時）；HA-RAG（2510.20878：依熱度選精度與放置）；ShadowServe（2509.16857：SmartNIC 抓取，讀取時）；HotPrefix（Proc. ACM Manag. Data 2025，熱度感知的 prefix 排程；未找到全文）；Characterizing Predictability–Latency Trade-offs of KV-Cache SSD Offloading in LMCache（IMNS 2026；沒有 arXiv，未讀）；EC-RAG（ICDE 2026，未讀）；Leyline（2606.01065，讀了 p.1–2：讓 agent 下指令編輯 KV，與寫入放置無關） |
| **不相關**（看標題與 S2 資訊判斷） | LLMVisor、Elastic Memory Remapping／Oneiros／MIRAGE（參數重映射）、MomentKV（有損淘汰）、CoMem、多 agent 通訊媒介選擇、CachePrune（隱私）、Path-Compressed Trie、Don't Break the Cache（agent prompt caching 評估）、SAE 稀疏結構、PIM 加速、Fast LLM Post-training、KVComm、SubGCache、CE-LSLM、兩篇 survey、壓縮理論兩篇、Plato、Context-Aware Autoscaling、EuroMLSys 模擬平台、Learn from the Past（稀疏索引） |

## 附錄 B：搜尋紀錄與範圍

- **本機 PDF**：`/mlsteam/data/tiara/papers/`。用 PyMuPDF 抽字（本機沒有 `pdftotext`），頁碼是 PDF 實體頁。發現本機的 `09_WhereShouldKVLive` 沒有在 03 收，寫了卡 [A_WhereKVLive](cards/A_WhereKVLive.md)。
- **arXiv**：API 關鍵字搜尋約 20 次（10 §5.2 的 Lit-A 關鍵字：admission、write-through、hidden state restoration、prefix caching hybrid／Mamba、state checkpoint、sliding window、multimodal reuse、disaggregated recompute、endurance、page cache、CXL、layout），下載 17 篇 PDF。
- **Semantic Scholar**：引用 Cake 的清單成功；關鍵字搜尋大多被限流（只回了 IMPRESS 一筆）。
- **DBLP**：被限流（HTTP 429），沒用到。
- **Web**：WebSearch 5 次（P/D 選擇性傳輸、Dynamo KVBM、Jenga venue、EPIC venue、LMCache venue）；WebFetch 2 個文件頁（SGLang HiCache design、Dynamo KVBM v0.9.1 guide）。
- **滾雪球**：第 1 輪＝Cake 的 citations＋Lachesis、EfficientAgent、SuffixReplay 的相關研究段落；第 2 輪＝從這些找到的 HBF 系列、MORI、KVFlow、Continuum、Sparse Prefix Caching。第 2 輪新找到、讀了全文的相關論文 ≥2 篇（Sparse Prefix Caching、MORI），依 10 §5.3 到此為止。
- **看過但沒寫卡**（只看摘要或標題）：KV Admission／WG-KV（2512.17452，EMNLP 2026，單請求有損的「學什麼該寫」）；KVFlow（2507.07400，agent step graph 引導淘汰與背景預取）；Apt-Serve（2504.07494，hidden cache）；Semantic Cache Distillation（2606.07684，ICML 2026，P/D 有損傳輸）；Characterizing HBF（2609.39131）；LLM Inference in a Flash!（2609.16161，flash 內計算＋字典壓縮以減少 KV 寫入）；DASC（2608.30386，混合模型狀態壓縮，下載了沒讀）；OasisKV（2608.08097，只看 web 搜尋摘要）；Continuum（只在 EfficientAgent 的相關研究看到：依重載成本與排隊延遲設 TTL 釘住 KV）。
- **查不到不等於沒有**（CLAUDE.md §1-7）：H11（P/D 選擇性傳輸＋重算）只做了有限搜尋。

**Web 來源**
- [SGLang HiCache design](https://docs.sglang.io/advanced_features/hicache_design.html)
- [Dynamo KVBM guide v0.9.1](https://docs.nvidia.com/dynamo/v-0-9-1/components/kvbm/kvbm-guide.md)
- [Jenga 在清華翟季冬論文頁（SOSP 2025）](https://pacman.cs.tsinghua.edu.cn/~zjd/publication/dblp-confsosp-zhang-dlkmwlyllz-25)
- [EPIC, PMLR v267](https://proceedings.mlr.press/v267/hu25j.html)
- [LMCache MLSys 2026 頁](https://mlsys.org/virtual/2026/oral/3646)
- [OasisKV（arXiv 2608.08097）](https://arxiv.org/pdf/2608.08097)
- [Semantic Scholar API：Cake citations](https://api.semanticscholar.org/graph/v1/paper/arXiv:2410.03065/citations)
