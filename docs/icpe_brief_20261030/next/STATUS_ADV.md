# STATUS_ADV（icpe-adv，acvlab，分支 `icpe-next-adv`，worktree `/ssd6/hungwei/icpe-next/adv`）

> 只由 `icpe-adv` 寫。角色：回應老師 11/8 的意見（量測這篇；量化／反量化階梯不在範圍）。最新的一節在最上面。每節先講故事，再講細節。

## 2026-11-08T00:38+08:00 — 摘要第 3 版：照同類論文的寫法改短（266 → 221 字）

**故事**
- 使用者要「看其他 paper 怎麼寫，確保寫法類似」。
- 最像我們的是兩篇量測論文（When Fancy Eviction Fails、KVCache Cache in the Wild）。它們的摘要短（155、164 字）、幾乎沒有數字，走「缺口 → 量了什麼 → 發現 → 原因 → 給別人的建議」。
- 第 3 版照這條線，收在一句建議：「怎麼還原、寫到哪不用預測；預測留給記憶體吃緊時的淘汰」。

**做了什麼**
- `adv_20261108/ABSTRACT_v3.md`：英文 221 字＋中文。內容：逐句對照「學誰」（引原句）、題目格式的比較與建議、要確認的三件事。
- `ABSTRACT_v2.md` 標為已取代。
- 讀了 13 篇相關論文和 2 篇 ICPE 論文的摘要（arXiv API；原檔只在 scratchpad）。

**要使用者決定**
1. 「Return times matter only when memory is tight.」這句是判讀，要不要留；不留就用 v3 第 4 節的保守寫法。
2. 題目：建議 *When the Session Returns: Where Prediction Still Pays for KV-Cache Restoration*；另有兩個備選。
3. 上一節的 A／B（oracle 慢於對手的格子怎麼報）與 push，仍待決定。

**寫本節時已經看過的正式結果**：同上一節，沒有新增。

## 2026-11-08T00:31+08:00 — 回覆老師的草稿、摘要第 2 版；−4.4% 查清楚了

**故事**
- 老師 11/8 的意見針對的是 11/3–11/4 的 Discord 訊息與圖 2，也就是 `STORY.md` 版；使用者 11/8 轉貼了原訊息。
- 老師要的「哪裡不用預測、哪裡仍然值得、為什麼」，新結果已經是這個結構，所以不需要新實驗，要的是**改寫法**。
- **使用者 11/8 決定**：要投 ICPE；不再花時間在模擬；專心想摘要。

**做了什麼**
- `adv_20261108/REPLY_TEACHER.md`：給老師的回覆草稿。
  - 先講故事，再逐點一行回答。
  - 內容：−4.4% 的診斷、前作原文對照、「哪個決策的價值變了」的表、論文範圍。
- `adv_20261108/ABSTRACT_v2.md`：照老師意見改的摘要（英文 266 字＋中文），附每處改動對應哪一點。
  - 4 個數字標【早上填】，等 icpe-main 的版本 C 修正版總判定。
- **−4.4%**：run `20261108-002650-icpe-adv-oracle-neg`（`code/icpe/adv_oracle_neg.py`；輸出 `adv_20261108/oracle_neg/`）。
  - 出處：FreeInference、CPU 16 GiB、SSD 級、Cake、GPU 閒。完美預取 0.910 s，對手 LRU+RPF 0.872 s。和原表那一列逐值相同。
  - oracle 是「每條規則的預知版」（`x1_sim.py:136-142`，④ 不含 ②），所以可以比對手慢。
  - 負值只來自 6 週裡的 6/21–27 那一週：那一週降級別人 51,642 次，LRU 是 21,214 次。
  - 新模擬器（偏離 14 修正版）：FreeInference 沒有 ≤ −5% 的格子；完美預取、回合結束就丟、輪到淘汰才丟三個元件有。
    - 例：SemiAnalysis 的完美預取 139 格 ≤ −5%（版本 C）。
    - 都不是寫入飽和。
    - ≥15% 的判定不受影響。
  - 先前 run `20261108-002621-…` 把 Mooncake R1／R2 併在一起算，已被取代。
- 前作原文：PBKV、KVFlow、CacheWise、Ask the Tool（arXiv HTML）、SYMPHONY（NSDI'26 PDF）。
  - 只在 session scratchpad，不進 git；sha256 寫在 REPLY_TEACHER 第 5 節。

**要使用者決定**
1. 回覆草稿、摘要第 2 版要改哪裡；什麼時候送老師。
2. oracle 比對手慢的格子，論文怎麼報（PREREG 沒寫到）：
   - **A**：照原定義報、寫明「預知版」，另外只報告一欄「④ 和 ② 取快」（重新分析，不跑模擬）。
   - **B**：只改措辭。
   - C（新模擬）已因使用者「不再花時間在模擬」排除。
3. push 前會先問。

**需要別的 session**
- icpe-main：版本 C 修正版的總判定與淘汰類比例。出來後，填摘要的【早上填】，並重看「mostly by deciding what stays」。

**寫本節時已經看過的正式結果**
- `ADVISOR.md`、`REPORT.md`、`STATUS_MAIN.md`（到 11/7 22:33 節）。
- `results/icpe_next/main/verB_d14/*_p1_headroom_components.csv.xz` 全部、`results/icpe_next/apc_d14/*/verC_headroom_components.csv` 全部（TraceLab 還沒進 apc_d14）。只看負值的統計與最慢的格，沒有算任何判定。
- 舊的 `results/icpe_h200/t5/p12/x1/`（FreeInference T5）。
