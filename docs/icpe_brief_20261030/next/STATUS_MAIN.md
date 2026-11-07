# STATUS_MAIN（icpe-main，acvlab，分支 `icpe-next-main`，worktree `/ssd6/hungwei/icpe-next/main-ana`）

> 只由 `icpe-main` 寫。格式照 NEXT_PLAN §3b：日期時間／做完什麼（commit、run_id）／卡住什麼（完整錯誤）／需要別的 session 做什麼／需要使用者決定什麼。最新的一節在最上面。
> icpe-main＝**正式的主結果分析 session**：算正式的上限表、照 PREREG 下判定。不跑模擬、不動任何佇列、不碰 `/ssd6/hungwei/icpe-next/data`、不改別人的檔（`next_sim_analyze.py` 原樣使用）。自己的輔助腳本＝`code/icpe/main_*.py`。

## 2026-11-07T14:07+08:00 — agent 軌跡（偏離 17＋更正）的分析選擇：使用者在看任何 agent 數字之前決定；現在算、工具類、下判定

**現況**（icpe-data 訊息，`origin/icpe-next-data` a1ea724，STATUS_DATA 14:05）：mini-SWE-agent（ms）、OpenHands（oh）各 12 個變體（固定並行 N16／32／64／72 × seed 1–3），版本 B（`b/agent_main`、`b/agent_p1`、`b/analysis/{ms,oh}_p1`）、版本 C（`apc/{miniswe,openhands}_N*_s*`，24 個都 REPRODUCED cells＝324）都已進 git。icpe-main 只讀了 PREREG 偏離 17 和更正，**沒看任何 agent 的模擬數字**（也沒讀 STATUS_DATA 14:05 的摘要表）。

**使用者決定**（本 session，AskUserQuestion，11/7 14:0x）
1. **現在算，不等偏離 14**：做法同版本 C 主表——判定照算，另給「不含 O-PF」的判定；O-PF、O-ALL 的列標「待偏離 14 修正」，修正版出來後取代。
2. **兩份都算工具類、下判定**：歸「合成到達・工具類」；單獨成一節，不併進論文 28 個變體的計數；W3 沒跑，推翻條件後半寫「未定」。
   - OpenHands 的規則 6 (b) 沒過（增量比中位數 0.795），報告時附註「輸入可能少算」。
- 照使用者 12:4x「照對話資料集的做法」沿用（不是新選擇）：每個 seed 當一個變體各自判，另報每格 3 個 seed 的中位數和最小–最大；版本 C 正式、版本 B 並列、版本 C 另並列「④ 不含 BELP」；B → C 用 `main_ab.py --rel eqsel` 核對。

## 2026-11-07T14:03+08:00 — icpe-sim：偏離 14 的重跑清單延後（模擬器舊錯誤，修法待使用者選）；對我已算的結果的影響

- **icpe-sim 的訊息**（STATUS_SIM 2026-11-07T14:05、`origin/icpe-next-sim` 0786f56，我沒有去讀，只轉述）：
  - 通過（fa3686b）：G1 Python／Rust 4,416／4,416 格相同；兩個旗標都開時 Python 對 Rust 2,208 列相同；SemiAnalysis 對診斷用反事實 36／36。
  - 沒過：Bailian coder、預取進 GPU、O-ALL，旗標關對開；兩個引擎都 assert（`gspf 13807 9`）。
  - 原因（舊路徑，旗標關也有）：GPU 預取器替某個 session 騰空間時，可能把同一個 session 丟掉（O-ALL 裡的 O-NOW），之後仍把它的 block 預取進 GPU，留下屬於已丟 session 的副本。
  - 範圍：只有 `pfgpu` block（sens_rep、各 `_p3`）的 O-ALL 列。Bailian coder 432 列裡 145 列會變，mean_ret −0.0056%～+0.0031%；Mooncake、CacheWise 代表格 0；其他紀錄 NOT_MEASURED。
  - 選項：A（icpe-sim 建議）新旗標 `--pf-gpu-recheck`、預設關，在 `pfgpu` 重跑時和 `--pf-expire` 一起開；B 改成預設；C 不修。修好並重新驗證前不能用 `--pf-expire`。
- **對我已算的結果**（本機核對 `results/icpe_next/main/p23/*_pfgpu_headroom_components.csv.xz` 的元件欄）：
  - 受影響的只可能是第 4 步 P3「預取進 GPU」的 O-ALL(report) 列：bl 162、cw 648、fi 324、mc 216、sa 648、tl 162，共 2,160 列。
  - 這些列不進任何判定（偏離 7 只判 5 個元件），P3 那節的表、配對、p90 也都只用判定元件；給老師的文件沒有引用 O-ALL。〔判讀〕判定字不會變。
  - 主表（版本 B／C）是預取到 CPU，不受影響。
- **時程**：P4 修正版（偏離 14）要等修法選定、重新驗證、出清單、icpe-data 排程後才會有；11/10 摘要前更不可能。摘要和給老師的文件裡 P4 的數字維持「待修正」。
- **寫本節時已經看過的正式結果**：沒有新的。
- **需要使用者決定什麼**：icpe-sim 的 A／B／C（已轉告使用者）。

## 2026-11-07T12:56+08:00 — 給老師的文件改成版本 C、不含 CacheWise；使用者決定做 TraceLab c1 的 GPU 層實機重播、W6b 往後移；agent 軌跡已登記

- **給老師的文件**：`docs/icpe_brief_20261030/next/advisor_20261107/ADVISOR.{md,pdf}` 改成版本 C、論文版（圖：run `20261107-124716-icpe-main-advisor-figs-verC`；PDF：run `20261107-125008-icpe-main-advisor-pdf`，9 頁）；摘要換成版本 C 版（28 個變體、3／9、14／14、62%、±30% 誤差）。commit 100e970，**還沒 push**（使用者這次沒說 push）。
- **h200 的估計**（STATUS_A 75a100b，〔推算〕）：TraceLab c1 的 GPU 層重播不能照 W6b 縮小（GPU 也 ÷k，k＝8 只剩 13.9 GiB）；可行的是「時間窗＋模擬暖機、全尺寸」，LRU／TTL／BEL 約 20–50 GPU 小時；最快 11/10 晚上到 11/11 晚上跑完。
- **使用者決定**（本 session，對兩題回「好」）：做 TraceLab c1 的 GPU 層重播（可行版，只跑 LRU／最好的 TTL／BEL）；W6b 往後移、排在重播之後。已傳給 h200：先登記才跑（由 h200 寫），照實列出版本 C 的 TraceLab c1 已看過的數字。
- **agent 軌跡**（icpe-data，`origin/icpe-next-data` 1a92a02，STATUS_DATA 12:55）：登記在任何模擬之前；mini-SWE-agent 用 MIT 第三方軌跡（tarsur385，Qwen3-30B，SWE-bench Verified 499 條），OpenHands 用官方評測輸出（MIT，claude-3.5-sonnet，SWE-bench Lite 299 個）、系統提示從當時的原始碼還原。〔判讀〕論文要寫明這兩點限制。

## 2026-11-07T12:42+08:00 — 缺口怎麼補：使用者決定補 mini-SWE-agent、OpenHands 公開軌跡（合成到達、固定並行）；H200 照現在順序

- 使用者問「缺口怎麼做、H200 要不要跑多一點」。icpe-main 的回答（依 STATUS_A 與 PREREG §5、偏離 16）：
  - H200 的 W6c（重算校準到 128K）、W6a（打字線索，Mooncake 對話）、W6b（O-ORD 的 GPU 重播，Mooncake 對話）都還在排在 DL_Lab5 的 24507 後面；三項都沒有驗到新結論的核心（工具類紀錄的 O-GPU）。W6c 最相關：TraceLab、SemiAnalysis 的上下文中位數 118K–142K，重算成本 40K 以後是外插。
  - 真正對準新結論的是「TraceLab c1 的 GPU 層縮小重播」，要改重播程式、登記、排隊；〔判讀〕ICPE 全文前大概來不及。
  - 更正 icpe-main 前一則：mini-SWE-agent、OpenHands 的公開軌跡只有 session 內的真實時間，跨 session 的到達要合成，補了只算「合成到達」，不增加原始到達的證據。
- **使用者決定**（AskUserQuestion，本節之前）：「補，照對話資料集的做法」；到達方式「固定並行 16／32／64／72」。
  - 已傳給 icpe-data：先登記（STATUS_DATA → h200 收進 PREREG，號碼 17）；來源要有授權、要有真實時間，沒有就停；只跑主網格＋版本 C；單獨成一節；分析由 icpe-main 做；偏離 14 的 O-PF／O-ALL 重跑一起排。
- 還在等使用者：push（本機領先 61 個 commit）；給老師的文件改成版本 C、不含 CacheWise；要不要問 h200「TraceLab c1 的 GPU 層重播」可不可行。

## 2026-11-07T12:27+08:00 — **版本 C（正式主表，論文版不含 CacheWise）：判定字和版本 B 相同；推翻條件（原始、合成）都成立；不靠 BELP、也不靠 O-PF**

**使用者決定**（本 session）
- 11:5x（AskUserQuestion）：偏離 15「push」；版本 C「現在算，O-PF 標待修正」。
- 12:1x（AskUserQuestion）：run `20261107-121543` 的逐格核對停在 2 格（上限差 3.4×10⁻²¹，選到的組合和兩個 T 都相同）→「改成更完整的核對再跑」（`main_ab.py --rel eqsel`：組合與 T 逐位元相同、上限差 ≤1e-12）。

**做了什麼**（run `20261107-122216-icpe-main-verC`，exit 0；結果 `results/icpe_next/main/verC/`）
- 合併：`origin/icpe-next-trx` 291e7d6、`origin/icpe-next-rux` b3c351c（PREREG 衝突取本分支，是對方的超集）；PREREG 從 `origin/icpe-next-a` bb04c54 同步（偏離 16，0 行刪除）。
- 輸入：版本 C＝`results/icpe_next/apc/<變體>/verC_headroom_components.csv`（analyze_v8），28 個變體（SemiAnalysis 12、Mooncake 4、Bailian 3、TraceLab 3、FreeInference 6），每個都先逐值重現版本 B（icpe-rux／icpe-trx）。
- 新工具：`main_rowfilter.py`（拿掉 BELP 的列）、`main_summary.py --verdict-exclude`（迴歸：不給時版本 B 的輸出逐欄相同）、`main_w3c.py`（W3 換表重算 f）、`main_ab.py --rel eqsel`。

**核對（全部 PASS）**
- rowcheck 5／5：每個 (trace, rule) 54 格、CPU 17 × GPU 18、cake／wb／預取到 CPU。
- analyze_v8 對 next_sim_analyze（拿掉 BELP）：O-ORD／O-DROP／O-NOW／O-PF／O-ALL 的組合與 T 逐位元相同，上限 26 格在 1e-12 內、其餘完全相同；O-GPU 不含 BELP ≤ 含 BELP，每格都成立。
- 格數自檢 30 個 (trace, rule)；並列表三版格數相同（178 列）。
- `main_w3c.py` 用版本 B 的表重算 9 個變體，n_ge15、f ≤ 0.5 的格數、f 中位數都和 TRX40 的 `w3_verdict.csv` 相同。

**判定**（照偏離 7；`verC_verdict.csv`；論文版不含 CacheWise）

| 類別 | 版本 B（論文版） | **版本 C（正式）** | C、判定不用 O-PF | C 不含 BELP（只報告） |
|:--|:--|:--|:--|:--|
| 原始・工具類 | 3/9，最大 62.4% | **3/9，62.4%** | 3/9 | 3/9 |
| 合成・工具類 | 14/14，52.9% | **14/14，56.8%** | 14/14 | 14/14，52.7% |
| 原始・非工具類（只報告） | 5/5，61.6% | **5/5，61.6%** | 5/5 | 5/5 |

- ≥15% 的（格, 元件），判定元件、R1／R2：原始工具類 352 → 351（C）→ 336（不含 BELP）；合成工具類 632 → 586 → 542；原始非工具類 520 不變。
- SemiAnalysis（B → C）：O-GPU 333 → 321、O-ORD 183 → 161、O-PF 19 → 10；O-GPU 最大 52.9% → 56.8%。和 icpe-rux 回報的相同（icpe-main 自己算的）。Mooncake、Bailian 幾乎不變。
- **TraceLab c1（推翻條件靠的那個）**：O-GPU 38.9%（B）→ 39.2%（C）→ **17.0%（不含 BELP，2 格 ≥15%）**；O-ORD 23.3% → 21.9% → 21.9%（4 格）。所以不含 BELP 時前半仍成立（偏離 10 擔心的「39% → 5%」是探索只看 3 個 CPU 容量）。

**推翻條件（版本 C，偏離 9＋偏離 10 第 5 點；`w3_C_verdict.csv`）**
- TraceLab c1：≥15% 的 20 格全部 f ≤ 0.5 → 後半成立 → **原始到達・工具類：推翻條件成立**。
- TraceLab c2：35／35、c4：59／59 → **合成到達・工具類：推翻條件成立**。
- FreeInference 6 週：前半不成立。
- 不含 BELP（只報告）：c1 6／6、c2 21／21、c4 58／58，同樣成立。
- 判定的 270 列裡，TraceLab 三個變體的線索做法都沒有一格贏過最好的固定規則（`n_cue_beats_rival` 0）。

**待修正**：O-PF、O-ALL 兩欄等偏離 14 的修正版取代；判定不用 O-PF 時判定字相同，所以判定不受影響〔判讀〕。

**已看過哪些正式結果**：版本 C 的全部輸出（本節）；其他同前。對話資料集：仍然沒看過。

## 2026-11-07T11:47+08:00 — **CacheWise 從論文拿掉（授權），內部保留**；PREREG 偏離 15 草稿（號碼待 h200 確認，還沒 push）

- 來源：使用者在 icpe-data 說「CacheWise 下先停 放不了paper」（STATUS_DATA 11:45，ba6ae12；icpe-data 轉告原話）。
- 使用者在本 session 決定（AskUserQuestion，11:4x）：「論文拿掉，內部保留」；「登記，由 icpe-main 寫」。
- 拿掉前後（版本 B，`verB_verdict.csv`；icpe-main 用已進 git 的彙總算）：合成工具類 21／26 → 14／14（最大都 52.9%）；原始工具類 3／9 不變；推翻條件不變；P4 合成工具類的（格, 元件）822 → 632，模型 P、B1 仍 ≥15% 的格數不變；H2 合成工具類淘汰占比 93.1% → 95.9%。
- 做了什麼：PREREG 從 `origin/icpe-next-a` e93074f 同步（d0e6516；偏離 9–14，0 行刪除），在檔尾加偏離 15 草稿（寫明理由是授權、看過的結果、前後對照、方向）。號碼已請 h200 確認；push 前先給使用者看。
- 之後要改：給老師的文件與報告的論文用部分（摘要的「40 variants of seven traces」→ 28／six；「21 of 26 synthetic」→「14 of 14」；主結果表、圖 1–4）出「不含 CacheWise」的版本，內部版並列含 CacheWise 的數字。

## 2026-11-07T11:33+08:00 — 對話資料集（偏離 13）的分析選擇：使用者在看任何對話資料集的數字之前決定；等 4 份到齊、等偏離 14 的修正版再算

**現況**（icpe-data 訊息、`origin/icpe-next-data` 59be8e8）：ShareGPT（T60／T600 × s1–3）、WildChat（s1–3）、UltraChat（T60／T600 × s1–3）的版本 B（`chat_main`、`chat_p1`、`analysis/{sg,wc,uc}_p1`）和版本 C（`results/icpe_next/apc/<變體>/`）都已進 git；LMSYS 等使用者登入 HF。icpe-main 只列了檔名，**沒看任何對話資料集的模擬數字**（也沒讀 STATUS_DATA 11:35 的摘要表）。

**偏離 13 已寫明的**：版本 C 是正式表、版本 B 並列；四份都歸「合成到達」；單獨成一節，不和主結果的 40 個變體合併計數；每個資料集報 3 個 seed 的中位數和範圍；標籤照登記；判定與彙總由 icpe-main 照 `main_summary.py` 的流程做。
- 照偏離 7 和 Bailian／Mooncake 對話的先例，這四份是聊天資料，歸「非工具類」：只報告有沒有過 15%，推翻條件寫「不適用」。（照既有規則，不是新選擇。）
- 版本 C 也並列「④ 不含 BELP」（使用者 11/7 對版本 C 的決定）。

**使用者決定**（本 session，AskUserQuestion，11/7 11:2x–11:33，commit 6de573b 之前）
1. **等 4 份到齊再算**（LMSYS 進 git 之後）。
2. **判定單位**：每個 seed 當一個變體各自判（同主結果 SemiAnalysis s1–s3 的做法），另報每格 3 個 seed 的中位數和最小–最大。
3. **T600 只報告**：判定用 T60 和 WildChat；T600 並列數字、不下判定。
4. **等偏離 14 的修正版再算**：O-PF、O-ALL 的列要重跑，整份對話資料集的判定等重跑完成後才算。

**需要別的 session 做什麼**：icpe-data 把對話資料集的 O-PF／O-ALL 列放進偏離 14 的重跑清單（它 11:2x 已說會做）；LMSYS 和修正版都進 git 後通知 icpe-main。

## 2026-11-07T11:08+08:00 — P4 模型 P 的斷崖：icpe-sim 查明是模擬器兩條沒登記的規則；**使用者選 B（補規則、重跑、修正版取代正式結果、排進佇列）**；文件加註

**icpe-sim 的結論**（STATUS_SIM 11:02，origin/icpe-next-sim 631d17c；icpe-main 看過該節）
- 不是算錯：用「只記錄、不改計算」的包裝重跑，正式值逐值相同（SemiAnalysis O-PF 17／17、CacheWise O-GPU 35／35、Mooncake O-GPU 8／8）。
- 機制 1（O-PF）：預取進 CPU 的副本，預測錯了也幾乎不會被丟；4–11 個就把 SemiAnalysis 一格的 16 GiB 塞滿，之後一直滿著，所以幾乎和 λ、seed、B1／B2 無關。
- 機制 2（O-GPU，B1）：抽到的回來時間已經過了，就被夾成「下一個請求」；B1 不重抽，不會再回來的對話一直被當成「馬上回來」，GPU 的 BEL 不逐出它們。
- 兩者都讓有雜訊的預測看起來比實際差。機制 1 在 λ＝0 時也有影響：量到的一格，O-PF 慢 0.7%（idle）／5.3%（busy），其他格 NOT_MEASURED。

**使用者決定**（本 session）
- 10:52 之後、11:07 之前（commit 390870d 之後、本節的 PDF run 20261107-110709 之前），AskUserQuestion：先選「C：並列修正版」，以及「先加註雜訊的問題再 push」。icpe-main 已把 C 傳給 icpe-sim。
- 同一段時間裡、C 之後，使用者直接改成 **B**：「B：補規則、全部重跑…也加入排班」。icpe-main 已更正給 icpe-sim，並通知 icpe-data 排程。
- 流程：icpe-sim 登記偏離（號碼問 h200）→ 改 Python 與 Rust、做 G1 → 列出要重跑的範圍（P4 受影響的 block；main／_p1／P2／P3 的 O-PF 列；版本 C；icpe-data 補充：chat_main、chat_p1 與對話資料集的版本 C）→ icpe-data 排程並先回報 → 修正版取代正式結果。
- icpe-data 回覆：G1 通過、merge 並做完 G0 之前，不開跑任何修正版。

**文件加註**（不改任何數字）
- 給老師的文件：結論段、圖 4 圖說、第 6 節的表、一句話，都把模型 P 的下降標成「待修正版取代」；摘要（英文、中文）改用模型 M 的結果：「the gap survives ±30% error in predicted return times」。
- 11/7 的報告（`main_report_20261107/REPORT.md`）開頭加更正：修正版出來前，不要引用「10% 亂猜剩不到一半」那幾句。
- `results/icpe_next/main/p4/noise/` 不動，是正式結果；之後由修正版取代時，舊版保留並標明。

## 2026-11-07T10:51+08:00 — 推翻條件（以版本 B 為準）：**合成到達・工具類也成立**（TraceLab c2、c4 的 W3）；給老師的文件；O-PF 異常交給 icpe-sim

**W3 合成到達**（TRX40，`origin/icpe-next-trx` ab78226，STATUS_TRX 10:55；`w3_verdict.csv` 照抄）
- 先決條件（偏離 9＋補記）：9 個變體都重算版本 B 的表，324／324 個（格, 元件）、7 欄 0 差；c2 是 run `20261107-090206-icpe-trx-rux-tl_c2-reproB`，c4 與 FreeInference 是各自 `run_apc_formal` 的第 2 步。
- tracelab_c2（run `20261107-092256-icpe-trx-rux-tl_c2-w3`）：≥15% 的（格, 元件）36 個，③ 都齊全，36 個都 f ≤ 0.5（中位數 −5.06）；寫入飽和 0。
- tracelab_c4（run `20261107-095115-icpe-trx-rux-tl_c4-w3`）：61 個，都 f ≤ 0.5（中位數 −2.60）；寫入飽和 0。
- FreeInference 6 週：沒有 ≥15% 的格 → 前半不成立。
- → **合成到達・工具類：推翻條件成立**（TraceLab c2、c4；以版本 B 為準）。SemiAnalysis、CacheWise 沒有線索欄位或沒跑，未定，但不影響「只要一份」的判定。
- icpe-main 的核對（讀 `w3_rows.csv`）：c2、c4 判定的各 270 列裡，線索做法沒有一格贏過最好的固定規則（T③ ÷ T② 最小 1.40、1.29），所以判定和 ③ 的定義、彙總方式無關。〔判讀〕

**其他**
- 使用者 11/7 約 10:30：P4 的 O-PF 異常「好」→ 已傳訊息請 icpe-sim 查（只查、回報，不改正式結果）。附了 TraceLab c2 一格的證據：λ＝0.1 時 O-PF 的 `prefetch_GiB` 從 728,431 降到 86.7、`ret_cpu_loads` 從 38,046,877 降到 2,835（`p4_b1/tracelab_c2_P_B1`，10 個 seed 平均）。
- 給老師的文件：`docs/icpe_brief_20261030/next/advisor_20261107/ADVISOR.{md,pdf}`（新結論、資料集、主結果表、圖、8 篇同類論文的圖表與摘要寫法、改寫的摘要）。圖：run `20261107-104634-icpe-main-advisor-figs`。
- 版本 C：TRX40 已完成 TraceLab 3 個、FreeInference 6 週（ab78226）；RUX3DPRO 進行中。icpe-main 還沒算版本 C 的表（等全部到齊）。

**已看過哪些正式結果**：加上 W3 c2、c4 的判定與 `w3_rows.csv` 的比值；TRX40 訊息裡的版本 C O-ALL(report) 最大值（c1 15.3%、c2 32.4%、c4 51.4%；FreeInference ≤1.1%）。

## 2026-11-07T09:55+08:00 — **P4（W4 雜訊）結果**：核對全過；主表（模型 P、B1）上限 ≥15% 的格隨 λ 明顯減少，但原始工具類 3 個變體到 λ＝0.4 都還有格 ≥15%；模型 M 幾乎不變

**問題**：PREREG 第 6 項／§3 B：上限 ≥15% 的格（偏離 8 的 2,582 個（格, 元件），以版本 B 為準、② 不含 APC），加上 oracle 雜訊之後還剩多少。照使用者 11/6 21:13 的決定只報格數，不下新的判定字。

**做法與核對**
- 第 1 段 run `20261107-082202-icpe-main-p4-noise`（08:22→09:45）：`next_sim_analyze.py` 原樣；main／`_p1` 先用 `main_addnoisecols.py` 補欄（使用者 11/7），Mooncake 用 `main_rulesplit.py` 拆 R1、R2（使用者 11/7）。
  - 沒有雜訊的上限表和版本 B 逐格相同：12／12 PASS（5 類 × mean／p90，Mooncake 分 R1、R2）。
  - λ＝0 自檢 24／24 通過；涵蓋的格數 272＋201＋457＋434＋369＋849＝2,582。
- 第 2 段 run `20261107-094535-icpe-main-p4-summary`（`main_p4.py`）：2,582 × 16 個雜訊設定都有、seed 數 1／10 對、`headroom_exact`＝版本 B；Mooncake min（只報告）322 個，λ＝0 和版本 B 的 min 逐值相同；不在清單的列 0。
- 前兩次失敗（06:40、08:20 節）沒有收進結果。
- 檔案：`results/icpe_next/main/p4/noise/`（`manifest.json` 有 sha256；逐 seed 的檔留在 run 目錄）。

**主表：模型 P、B1**（`p4_by_class.csv`；只算 5 個判定元件、R1／R2；數字＝10 個 seed 的中位數仍 ≥15% 的（格, 元件）數；括號＝還有 ≥1 格的變體數）

| 類別 | 格數 | λ 0.1 | 0.2 | 0.3 | 0.4 |
|:--|--:|--:|--:|--:|--:|
| 原始・工具類 | 352（3） | 168（3） | 93（3） | 58（3） | 30（3） |
| 合成・工具類 | 822（21） | 181（14） | 139（14） | 117（14） | 78（13） |
| 原始・非工具類（只報告） | 520（5） | 201（5） | 97（4） | 52（4） | 24（4） |

- B2 並列：原始工具類 126／78／59／41（3 個變體都有）；合成工具類 708／650／560／426（20 個）。
- 模型 M（乘法誤差，λ 0.1／0.2／0.3）：原始工具類 350／341／334；合成工具類 797／770／763；B2 幾乎相同。
- 原始工具類逐變體（P、B1，λ 0.1／0.2／0.3／0.4）：Mooncake toolagent R1 60／23／12／2（共 167）；R2 100／64／42／27（共 163）；TraceLab c1 8／6／4／1（共 22，全是 O-GPU）。
- 寫入飽和（版本 B 的標記）：原始工具類仍 ≥15% 的格裡 45／29／19／10 格飽和；合成 0。
- 3 × 10 合起來（只用 3 個佇列 seed 都有選到的，P、B1）：CacheWise 53 組 → 0／0／0／0；SemiAnalysis 160 組 → 51／44／36／25。B2：CacheWise 45／45／41／35；SemiAnalysis 156／144／121／88。湊不滿的：CacheWise 19 組、SemiAnalysis 39 組（另列）。

**觀察**〔判讀，沒有另外驗證〕
- 合成到達上，B1 比 B2 傷得重很多（例：CacheWise O-GPU B1 λ＝0.1 111 → 0，B2 104）。可能的機制：B1 抽到太短的回來時間後，到下一個請求前都不再重抽，這個對話一直被當成「快回來了」。
- O-PF 在合成到達上只要 λ＞0 就掉到負的、而且幾乎和 λ、抽法無關（例：TraceLab c2 一格 λ＝0.1–0.4、B1／B2 都是 −0.239，seed 間標準差 <1e-5）。可能是「挑最快回來的那個」這種選法，只要有一部分預測是亂抽的，最小值就幾乎總是被亂抽的值佔走。要不要請 icpe-sim 確認，由使用者決定。
- Mooncake toolagent 的 O-GPU、O-ORD 在 B2 下 λ＝0.1 就幾乎歸零，和合成到達相反。

**限制**：P4 以版本 B 為準（② 不含 APC，偏離 8）。偏離 10 把正式主表換成版本 C、P4 改用版本 C 選格；P4 的 ②／④ 要不要也加 APC／BELP（要補跑雜訊）還沒決定。

**已看過哪些正式結果**：P4 的全部彙總（本節的表、`p4_by_trace`、`p4_pooled`、O-PF 的逐格值）。其他同 09:15 節。

## 2026-11-07T09:15+08:00 — **推翻條件（以版本 B 為準）：原始到達・工具類成立**（TraceLab c1，W3 照偏離 9）；版本 C 的判定等表；版本 C 另並列「④ 不含 BELP」（只報告）

**使用者決定**（本 session，11/7 約 09:12，回覆 icpe-main 的兩個問題「好」）
1. 現在就把「以版本 B 為準：原始工具類推翻條件成立（TraceLab c1）」寫進本檔的正式判定；版本 C 的判定等它的表出來再補。
2. 版本 C（偏離 10）的上限表與判定，另外並列一份「④ 不含 BELP」，標成只報告。
   - 理由：偏離 10 寫明，使用者在看過探索（TraceLab c1 的 O-GPU 上限：④ 只有 BEL 時 39.0%→5.1%，加 BELP 39.2%）之後才選「加 BELP」；而原始工具類的推翻條件只能靠 TraceLab c1（Mooncake toolagent 沒有線索欄位）。並列讓讀者看到結論靠哪個選擇。
   - 做法（之後做）：版本 C 的 sweep 拿掉 BELP 的列，用 `next_sim_analyze.py` 原樣算（APC 在那裡是不預測的 GPU 規則；O-GPU 的 ④＝NON_ORACLE × BEL）。

**依據**
- PREREG 偏離 9（W3 規則；origin/icpe-next-rux 8f04a91，commit 08:37:10+08:00）、偏離 9 更正（e7e0c77）、偏離 9 補記（aec99d1：重現檢查要涵蓋全部 9 個容量）。
- TRX40 的 W3：run `20261107-084104-icpe-trx-rux-tl_c1-w3`（worktree 8f04a91，開跑在登記之後）；結果 `origin/icpe-next-trx` 9e1064f／16ba40f／4ca0d65 的 `results/icpe_next/v8_explore/w3/tracelab_c1/`。
- 先決條件（偏離 9 第 3 點＋補記）：TraceLab c1 重算版本 B 的表，324／324 個（格, 元件）、7 欄全部 0 差（run `20261107-083210-icpe-trx-rux-tl_c1-apc` 108 個＋`20261107-084950-icpe-trx-rux-tl_c1-reproB` 216 個；STATUS_TRX 09:10）。

**判定**（照偏離 9 第 4 點；`w3_verdict.csv` 照抄）
- TraceLab c1（原始到達、工具類）：上限 ≥15% 的（格, 元件）22 個（O-GPU 18、O-ORD 4），③ 都齊全，**22 個都 f ≤ 0.5**；寫入飽和 0；f 中位數 −7.409、最小 −8.909 → 後半「沒有簡單線索能拿走大部分」**成立**。前半（版本 B 38.9%）成立。
- → **原始到達・工具類：推翻條件成立**，結論照 §1 改成「在這類工作負載上預測仍有空間」（以版本 B 為準）。
- 其他變體：Mooncake toolagent R1／R2 沒有線索欄位 → 後半未定（偏離 9 第 1 點）；FreeInference 6 週前半不成立（版本 B 最大 0.9%）；合成到達（TraceLab c2／c4）TRX40 在跑，SemiAnalysis 沒有欄位、CacheWise 沒跑 → 合成到達的推翻條件目前未定。

**icpe-main 的核對**（讀 `w3_rows.csv`；〔判讀〕標在句尾）
- 判定的 270 列（54 格 × 5 元件）：線索做法（T③）沒有一格比最好的固定規則（T②）快（`cue_beats_rival` 全 False），T③ ÷ T② 最小 1.47；T③ ÷ T①（LRU, LRU）中位數 1.678，54 格沒有一格 <1。
- 所以就算 ③ 改成「線索規則或 ② 的做法取最快」，f＝0，仍 ≤ 0.5；22 格全部 f ≤ 0.5，判定也和「逐格任一」的彙總方式無關。〔判讀〕
- 版本 C 的 ② 規則只多不少（多 APC），T② 只會更小，所以上限為正的格 f 仍為負，後半在版本 C 也成立；版本 C 會變的只有前半（取決於 ④ 的 BELP）。〔判讀〕
- 限制：W3 與版本 C 用的 v8 引擎、`analyze_w3.py`、`analyze_v8.py` 沒有經過 icpe-sim 審查（使用者 11/7「不用」轉給 icpe-sim）。

**時間標記更正（icpe-main 的錯）**：我給 icpe-rux、icpe-trx 的訊息把使用者決定「W3 先登記」的時間寫成「約 08:45」，實際約 08:28–08:30（訊息在 08:31:28 之前送出）；兩邊已照實更正（偏離 9 更正 e7e0c77、STATUS_TRX 08:45 105d0a0）。

**已看過哪些正式結果**：上面 W3 c1 的判定與 `w3_rows.csv` 的比值；偏離 9／10／11 的內容（含其中列出的探索摘要）。P4 加雜訊後的上限：還沒看過。

## 2026-11-07T08:20+08:00 — P4 第二次計算：Bailian、TraceLab 核對全過；**Mooncake 的 λ＝0 自檢失敗**（只在 rule＝min，原因已查明），停下來問使用者

**使用者決定**（本 session，AskUserQuestion，11/7 約 07:5x，看任何 P4 上限數字之前）
- 06:40 節的錯誤：「我先補欄位再餵進去」。`code/icpe/main_addnoisecols.py`（9ec8d94）把 main／`_p1` 的 sweep 逐字複製（只用 csv 模組讀字串，不解析數值），列尾補 `noise_*`＝`load()` 的預設，寫在 run 目錄的 `in/`、不進 git；`next_sim_analyze.py` 原樣使用。
- 3 × 10 合起來的表：「只合湊滿 30 的格」，湊不滿的另列格數。

**run `20261107-080102-icpe-main-p4-noise`**（exit 1；留了 `FAILED_NOT_USED`）
- Bailian、TraceLab：補欄檢查通過；沒有雜訊的上限表和版本 B 逐格相同（mean 與 p90 各 972／972，`out/cellcheck.csv`）；λ＝0 自檢通過（P、M 各 272 格、201 格，mean 與 p90）。
- Mooncake：補欄檢查通過，接著 `next_sim_analyze.py` 停在：
  ```
  File "code/icpe/next_sim_analyze.py", line 188, in main
    assert dmax == 0.0, f"λ＝0 的上限和沒有雜訊的不同（最大差 {dmax}）：{spec}"
  AssertionError: λ＝0 的上限和沒有雜訊的不同（最大差 0.13384690486638032）：('M', 0.0, 'B1', 0)
  ```
- CacheWise、SemiAnalysis 沒有跑到。

**原因**（診斷 run `20261107-081616-icpe-main-p4-diag-l0`；照 `main()` 的雜訊流程重做 λ＝0 的兩個設定，只數不同的格數）
- R1 457、R2 434 個（格, 元件）：**0 個不同**。P、M 都一樣。
- rule＝min 353 個：**31 個不同**，全部是「這格這個元件只有 R1 或只有 R2 有跑雜訊」的；兩個規則都有跑的 322 個：0 個不同。
  - 偏離 8 第 3 點：Mooncake 只列 R1／R2，所以只有一個規則 ≥15% 的格，另一個規則沒有跑雜訊。
  - `analyze()` 的 rule＝min 是對「有的規則」取小，只有一個規則時就等於那個規則，不是真的 min；沒有雜訊時是兩個規則取小。自檢比到這 31 個就不同。
  - 這 31 個的版本 B min 都 <15%（另一個規則 <15%），本來就不在選格清單裡。
- λ＝0 的原始列：P、M 各 11,827 列，和 main／`_p1` 同一個（格, CPU 規則, GPU 規則）的列在 mean_ret、median_ret、p90_ret、n_ret、n_req、sum_ret、rho_w、rho_r、total_all 都逐值相同。
- 〔判讀〕模擬本身在 λ＝0 和沒有雜訊完全相同；失敗來自 `next_sim_analyze.py` 的 rule＝min 遇到「只跑一個規則」的格。

**已看過哪些正式結果**：上面的格數與最大差；Bailian／TraceLab 的核對結果（PASS 與格數）。P4 加雜訊後的上限、seed 間統計，一個都沒看過。

**需要使用者決定什麼**：Mooncake 怎麼處理（下一節）。

## 2026-11-07T06:40+08:00 — P4 資料到齊；第一次計算失敗（`next_sim_analyze.py` 合併沒有雜訊欄的 sweep 時，沒有雜訊的列全部不見）；停下來問使用者

**做完什麼**
- icpe-data 06:40 通知 P4 全部進 git。合併進本分支（只在本機）：
  - f88ca7e＝`origin/icpe-next-data` a25dcba（`p4_b1`／`p4_b2` 68 個 block）。
  - 0812989＝`origin/icpe-next-trx` 3be77ac（`sa_p4_b1`／`sa_p4_b2` 48 個 block）。
  - `PREREG.md` 沒有差異；`next_sim_analyze.py` 三邊都是 blob aa8e3a9。

**卡住什麼**（run `20261107-063333-icpe-main-p4-noise`，exit 1，留了 `FAILED_NOT_USED`）
- 指令：`next_sim_analyze.py --sweep <bl_main> <bl_p1 _ttl> <12 個 bailian P4 檔> --out-prefix bl_p4 --pf-target cpu --metric mean_ret`。第一個紀錄（Bailian）就停了，`out/` 是空的。
- 完整錯誤：
  ```
  File "code/icpe/next_sim_analyze.py", line 177, in main
    hx = H.set_index(ck).headroom
  KeyError: "None of ['trace', 'rule', 'cpu_GiB', 'slow', 'gpu', 'restore', 'gpu_GiB', 'gpu_write', 'slow_cap_frac', 'component'] are in the columns"
  ```
- 原因（診斷 run `20261107-063414-icpe-main-p4-diag`，只數列數、不算上限）：
  - main 和 `_p1` 的 sweep 沒有 `noise_*` 欄，P4 的 sweep 有。
  - `load()` 只在「合併後整欄都沒有」時才補 `noise_model="none"`。合併後這欄存在，所以 main／`_p1` 的 41,472 列變成 NaN，等於 `"none"` 的列是 0。
  - 所以沒有雜訊的上限表 H 是空的，在 λ＝0 自檢之前就停了。
- **已看過哪些正式結果**：和 21:14 節相同。P4 一個上限數字都沒看過（失敗時沒有輸出任何上限）。

**需要使用者決定什麼**：怎麼處理這個錯誤（見下一節）。

## 2026-11-06T21:14+08:00 — P4（W4 雜訊）的分析選擇：使用者在看任何 P4 數字之前決定；全部到齊再算

**使用者決定**（本 session，AskUserQuestion，11/6 約 21:13）
- 當時 P4 只有 SemiAnalysis 進 git：`origin/icpe-next-trx` b6eb4cd，48 個 block、96 個檔。icpe-main 只看了檔名和 CSV 的欄位名稱，**沒有讀任何 P4 的數字**。
- 決定照 PREREG 第 6 項與 §3 B、C；PREREG 沒寫到的部分如下：
  1. **只報告：格數。** 對每種雜訊設定（模型, λ, 抽法），報 10 個 seed 的中位數仍 ≥15%／≥5% 的格數，加上 PREREG 要的 seed 間中位數、最小–最大、平均±標準差。**不下新的判定字**，所以不改 PREREG。
  2. **B1 主、B2 並列。** 主表用 B1，B2 放在旁邊當對照，和 `h200` 在 §3 B 的解讀一致。主模型照 PREREG 是 P。
  3. **佇列 seed：兩種都報，分開為主。** 主表每個佇列 seed（s1／s2／s3）各自 10 個雜訊 seed，和版本 A／B 判定的粒度相同。另報 3 × 10＝30 個合起來的中位數與範圍（§3 C 的「交叉 3 × 10」）。
  4. **全部到齊再算。** 要等 acvlab 的其餘 P4 和 TRX40 的 6 個 Mooncake block 都進 git，SemiAnalysis 不先算。
- 照之前的做法、不另外問的：
  - 上限用 `next_sim_analyze.py`（原樣使用），λ＝0 自檢不過就停。
  - p90 只報告；寫入飽和的格照列，另外標註。
  - 雜訊的 ② 用沒有雜訊的列，也就是 P1 正式的 sweep。

**已看過哪些正式結果**：和 14:23 那節相同。P4 一個數字都沒看過。

**時間**〔推算，別的 session 寫的〕
- acvlab 的其餘 P4 約 11/7 04:40–05:35（STATUS_DATA 19:50，8d70cbc）。
- TRX40 的 6 個 Mooncake block 21:03:52 開始，約 3 h（icpe-data 的訊息、STATUS_TRX 21:15）。
- 全部進 git 後 icpe-data 會通知 icpe-main。

**還沒 push**：照使用者的話，等資料出來再說。本節只在本機。

## 2026-11-06T14:23+08:00 — 第 4 步：P2／P3 敏感度（3 個 TTL）完成；工具類的判定字在每個敏感度都和版本 A 相同

**使用者決定**（本 session；都在 run 開跑 14:16:10 之前、沒有看過任何 P2／P3 結果時）
- 方案二：等 TL 的 `_wt` 進 git，再一次算完。
- 報告形式（AskUserQuestion）：每個敏感度出上限表，並照偏離 7 下判定字。只報告，主判定（版本 A／B）不變。
- 逐格配對（AskUserQuestion）：每組只換一個設定。
  - wt 對 wb；預取進 GPU 對預取到 CPU。
  - 只載入對 Cake，比同一個慢層；nvme807 的只載入對 P2 的 nvme807 Cake。
  - nvme807 Cake 對 ssd Cake（PREREG §2 第 4 項：其他設定和 SSD 級相同）。
- sens_rep 重疊（AskUserQuestion）：先逐格核對；數字只用 P2／P3，不重複算。

**輸入與做法**
- 合併：`origin/icpe-next-data` bb68038 → 0931518；`origin/icpe-next-trx` c8e554f → 2852684。
  - 兩次都沒有衝突；PREREG.md 沒變（對方的版本是我們的位元組前段）。
- 規則集合：17 CPU × 8 GPU（3 個 TTL），和 sens_rep 相同（icpe-data 對偏離 6 的解讀）；對照＝版本 A（`results/icpe_next/main/verA/*_ttl3_*`）。
- 預取進 GPU＝main＋`_pfgpu`，`--pf-target gpu`。`_pfgpu` 的 block 只有 O-PF、LRU+RPF、O-ALL 的列（54 格 × 3 × 8）。
- 工具（b54d3cf）：`main_pair.py`（逐格配對），`main_rowcheck.py` 加 `--expect-value`（每列的設定欄必須是該變體的值）。
- run `20261106-141610-icpe-main-p23-sens`（14:16:10 → 14:21:10，exit 0，stderr 空）；結果 a9b9aa6，`results/icpe_next/main/p23/`（129 個檔＋manifest，12 MB，單檔最大 0.75 MB）。

**核對（全部 PASS）**
- rowcheck 30／30：6 類 × {main、nvme807（18 格）、load（72 格）、wt（54）、pfgpu（54 格 × 3 CPU 規則）}，含每個變體的 slow／restore／gpu_write／pf_target。
- sens_rep 重疊 16／16 IDENTICAL：cw／bl／mc／fi × {nvme807、load、wt、pfgpu}，共 23,200 列。sens_rep 是 Python（f7038ddfaf9c），P2／P3 是 Rust（ed2f4c78c88a+rs43dbe828）。
- 預取進 GPU 時 O-ORD、O-DROP、O-NOW 和版本 A 逐格相同 6／6（這三個元件的 ④、② 都沒有預取規則，必然成立）。
- 格數自檢 8／8；配對 4／4（x 每列都配到 y 的唯一一列）。

**判定（照偏離 7；只報告）**：變體數 ≥15%／全部（判定元件的最大值）

| 類 | 版本 A | nvme807 | 只載入 | wt | 預取進 GPU |
|:--|:--|:--|:--|:--|:--|
| 原始到達、工具類 | 3/9（62.4%） | 2/9（34.2%） | 3/9（99.9%） | 3/9（60.1%） | 3/9（73.7%） |
| 合成到達、工具類 | 21/26（55.0%） | 16/26（52.8%） | 23/26（98.1%） | 22/26（55.1%） | 22/26（54.6%） |
| 原始到達、非工具類（只報告） | 5/5（61.6%） | 4/5（41.2%） | 5/5（99.9%） | 5/5（64.4%） | 5/5（70.4%） |

- 推翻條件：每個敏感度都是「未定：≥15% 成立，但簡單線索（W3）NOT_MEASURED」。
- **nvme807 只有一種慢層（18 格），版本 A 有 ssd／b32／nfs（54 格），判定字不能直接比**。公平的比法是同一格的 ssd：原始工具類 2/9 對 2/9、合成工具類 16 對 17（CacheWise N64 s1：ssd 16.5% → nvme807 14.9%）、原始非工具類 4 對 4。
- 和版本 A 判定字不同的變體（`verAS_verdict.csv`）：
  - 只載入：CacheWise N32 s1、s3 NO → YES（63.2%、57.8%）。
  - wt：CacheWise N32 s1 NO → YES（16.9%）。
  - 預取進 GPU：CacheWise N32 s1 NO → YES（15.6%）。
  - nvme807 的翻轉見上一點（要和 ssd 比）。
- 判定元件 ≥15% 的（格, 元件），不含 rule＝min：版本 A 1,705／10,800（寫入飽和 141）、nvme807 372／3,600（0）、只載入 2,520／14,400（88）、wt 1,689／10,800（130）、預取進 GPU 1,856／10,800（149）。
  - 預取進 GPU 只有 O-PF、O-GPU 會變：O-PF 281 → 467、O-GPU 814 → 779。

**逐格配對**（判定元件、不含 rule＝min；`pair_*_pairsum.csv`）

| 配對 | 對數 | 敏感度較高／較低／相同 | 跨過 15%：往上／往下 |
|:--|--:|:--|:--|
| wt 對 wb | 10,800 | 5,829／4,020／951 | 157／173 |
| 預取進 GPU 對預取到 CPU | 10,800 | 1,303／2,307／7,190 | 233／82 |
| nvme807 對 ssd（Cake） | 3,600 | 389／2,879／332 | 2／61 |
| 只載入對 Cake（同慢層） | 14,400 | 8,823／4,618／959 | 762／319 |

**只載入的極端格**：上限 >0.9 的有 84 個（格, 元件），不含 rule＝min：nfs 的 O-ORD 73、nfs 的 O-PF 6、b32 的 O-ORD 5。
- 這些格 ② 的平均 TTFT 中位數 218 秒、最長 1,300 秒（例：Mooncake toolagent R2、CPU 256 GiB、nfs、GPU 閒：② 684 秒、④ 0.49 秒）；其中 39 格標了寫入飽和。
- 同樣的格在 Cake 下是 10.9%–62.4%。
- 〔判讀〕這是只載入時慢層讀取把對手拖垮的情形，不是「預測能多給多少」的典型值；引用只載入的最大值要附 T 值。

**p90（只報告）** 判定元件最大值：nvme807 42.7%／40.8%／60.1%（原始非工具／原始工具／合成工具）、wt 72.9%／71.7%／87.0%、預取進 GPU 75.5%／83.3%／87.6%、只載入三類都是 1.000（四捨五入）。

**這一節沒做的**：sens_rep 的「快取命中前綴」（cw、tl）和「截斷 128K」（cw、fi、tl、sa）不在我的步驟清單裡；資料在 git，沒有算。

**寫本節時已經看過的正式結果**：前面各節；第 4 步的全部輸出（判定、by_trace、配對、只載入極端格的 T 值、p90 最大值）。W4 雜訊：沒有看過任何結果（還沒跑）。

**需要別的 session 做什麼**：無。

**需要使用者決定什麼**：W3 線索 (a) 要不要列進 ICPE（NEXT_PLAN §7 第 4 項）。沒有 W3，每一版的推翻條件都只能寫「未定」。
- 使用者 11/6（看過本節之後）：「push 跟第二點等數據出來再說」——W3 先不決定；第 4 步的 commit 也先不 push（只在本機）。

**接下來**：P4（W4 雜訊）的結果進 git 後再分析（開算前先問使用者）。

## 2026-11-06T12:42+08:00 — 版本 B（P1 正式值）完成：判定字和版本 A 相同；P4 選格清單；PREREG 偏離 8

**使用者決定**（AskUserQuestion，本 session）：
- P1 進 git 之後（icpe-data 10:46 通知；合併 c472cdb 11:03:54 之前）：版本 B「照計畫開始」；重疊列「要核對」。
- 看過版本 B 全部結果之後（run 11:15:21 跑完之後、baa508d 12:40:25 之前）：「全部 commit 並 push」；P4「加上 O-ALL(report)」；「登記，由 icpe-main 寫」。

**做完什麼**
- 合併：c472cdb＝`origin/icpe-next-data` @ d30ae7c。`PREREG.md` 有衝突，取我這邊：對方那份是我這邊的前段（有偏離 7、沒有偏離 7 更正），所以就是聯集，0 行刪除。df1ec60＝`origin/icpe-next-trx` @ 98b0d2f（sa_p1、sa_p2）。`next_sim_analyze.py` 沒變；`x1_sim.py` 只改了 Rust 的準備流程。
- 核對工具（fb61730）：`main_rowcheck.py`、`main_overlap.py`、`main_ab.py`。P4 選格（50b04e8）：`main_p4cells.py`。
- run `20261106-110735-icpe-main-verB-p1` **失敗**（exit 1），在算任何上限表之前：`FileNotFoundError: 'results/icpe_next/b/sa_main_ttl7/*/x1_sweep.csv.xz'`。原因是我的 cmd.sh 寫錯：`sa_main_ttl7` 只有 `cells.csv`，那次補跑照 STATUS_DATA 08:10 沒跑。run 目錄留了 `FAILED_SCRIPT_BUG`。
- run `20261106-110854-icpe-main-verB-p1`（PASS，6 分 24 秒）只拿掉 SemiAnalysis 那一行重疊核對，其他相同。結果在 `results/icpe_next/main/verB/`、`results/icpe_next/main/oldtests_verB/`（baa508d，已 push）。
  - 正式算法：每類紀錄 main＋`_p1` 的 `_ttl`；④ 和 ② 都有 7 個 TTL；沒有 `--rival-only-gpu`；不含 `cw_main_ttl7`。
  - 並列：偏離 5 的算法（`verB_dev5`）。
  - 只報告：`--dead-h 1800／3600`（加 `_trunc`）、p90。

**核對（全部 PASS）**
- **列數**（`rowcheck.csv`）：18 份 sweep 都是完整網格（main 17×8、`_ttl` 15×8、`_trunc` 6×16，每個（紀錄, 規則）54 格）。42 個（紀錄, 規則）× 每個元件都是 54 格，和版本 A 相同（`verAB_by_trace.csv`）。
- **重疊列**（`overlap.csv`）：`cw_main_ttl7` 對 `cw_p1` 的 `_ttl` 重疊 19,840 列，逐格相同。兩邊程式是 f7038ddfaf9c（Python）對 ed2f4c78c88a+rs43dbe828（Rust）。SemiAnalysis 沒有第二份結果，NOT_MEASURED。
- **逐格關係**（`cellcheck.csv`，30/30）：
  - B 的 O-GPU ≤ A。
  - 偏離 5 的算法 ≤ A。
  - B 和偏離 5 的 O-GPU、O-ALL 完全相同。
  - 截尾版的 O-ORD、O-PF、O-GPU 和不截尾的完全相同。
- **舊模擬器**：`--check-v0` 仍重現論文的 50.5%、95、36。

**判定**（`verB_verdict.csv`；和 A 並列在 `verAB_verdict.csv`）

| 類別 | 版本 A | 版本 B（正式） |
|:--|:--|:--|
| 原始到達・工具類 | 9 個變體中 3 個 ≥15%，最大 62.4% | 3/9，最大 62.4% |
| 合成到達・工具類 | 21/26，最大 55.0% | 21/26，最大 52.9% |
| 原始到達・非工具類（只報告） | 5/5，最大 61.6% | 5/5，最大 61.6% |

- 原始工具類成立的是 Mooncake toolagent R1 57.8%、R2 62.4%，和 TraceLab c1 38.9%。FreeInference 6 週都 ≤0.9%。
- 推翻條件：未定（W3 NOT_MEASURED）。偏離 5 的算法並列：判定字、各類的變體數都相同。

**A 到 B 變了什麼**
- SemiAnalysis 的 O-GPU 降 3–4 個百分點（例：N64 s1 從 54.7% 到 51.7%）；≥15% 的格從 355 降到 333。
- 有些元件因為 ④ 也多了 TTL 而小升：Mooncake toolagent R1 從 57.4% 到 57.8%；原始工具類的 O-PF 最大值從 52.0% 到 53.3%。
- CacheWise 剛好卡在門檻邊：N32 s1 14.8%（不成立），N32 s2 15.4%（成立，2 格）。
- O-NOW 在所有紀錄都沒有一格 ≥15%（最大 8.9%）。
- ≥15% 的 1,694 個（格, 判定元件）裡有 144 個寫入飽和，全部在 Mooncake。

**截尾（只報告，`verB_h1800`／`verB_h3600`）**
- 定義：`-H{H}`＝紀錄最後 H 秒內不准判「不回來」（`x1_sim.py:707`、`:1228`）。
- 只有 O-DROP、O-NOW 會變。O-DROP ≥15% 的格：
  - 原始非工具類 120 → 20（H1800）→ 0（H3600）。
  - 原始工具類 77 → 9 → 0。
  - Mooncake 的 O-DROP 最大值從 55.7% 降到 0.0%（H3600）。
- 〔判讀〕O-DROP 的上限幾乎全部來自紀錄尾端，和 EVIDENCE B2 的舊結論一致；撐住判定的是 O-GPU、O-ORD、O-PF。

**論文的兩個 Mooncake 舊測試**（`oldtests_verB_*`；rule=min、Cake）

| 測試 | old_V0（論文） | old_V1 | new 版本 A | new 版本 B |
|:--|--:|--:|--:|--:|
| T1：conversation O-PF、SSD、CPU 128 GiB | 50.5% | 53.2% | 39.1% | 37.8% |
| T2：192 格 ≥15%（不截尾） | 95 | 97 | 113 | 111 |
| T2：不截尾、只算不飽和 | 62 | 65 | 79 | 77 |
| T2：截尾 H3600＋只算不飽和 | 36 | 38 | NOT_MEASURED | **38** |

- 新 B 的 38 格依 CPU 分：128 GiB 16、256 GiB 15、512 GiB 7、1,024 GiB 0（V0 是 16／12／8／0）。
- 依慢層分：ssd 17、b32 21、nfs 0（V0 是 19／17／0）。依元件分：O-ORD 18、O-PF 20、O-DROP 0。
- 只報告：R1 40 格、R2 42 格；O-GPU 25 格。

**P4（W4 雜訊）選格清單：給 icpe-data 排模擬**（PREREG 偏離 8；run `20261106-124122-icpe-main-p4cells`；43134be）
- 選法：版本 B 正式上限 ≥0.15 的每一（格, 元件）。元件＝O-ORD、O-DROP、O-NOW、O-PF、O-GPU＋O-ALL(report)；O-ALL 加雜訊後仍只報告。Mooncake 只列 R1／R2（min 已涵蓋，程式自檢 322 筆）。
- 規模：2,582 個（格, 元件）、975 個不同的格。
  - 依元件：O-ALL(report) 888、O-GPU 781、O-ORD 434、O-PF 282、O-DROP 197、O-NOW 0。
  - 依紀錄（格×元件／格）：Mooncake 891／216、SemiAnalysis 849／389、CacheWise 369／179、Bailian 272／98、TraceLab 201／93、FreeInference 0。
- 檔案（`results/icpe_next/main/p4/`）：
  - `p4_cells_distinct.csv`：每格一列，`components`＝要加雜訊的元件；排模擬用這份。
  - `p4_cells.csv.xz`：每個（格, 元件）一列，含 ④／② 的做法、headroom、`write_saturated`。
  - `p4_summary.csv`：格數。
  - 格的座標＝(trace, rule, cpu_GiB, slow, gpu, restore, gpu_GiB, gpu_write, slow_cap_frac)，全部是主時間設定（wb、111.0625 GiB、cake）。
- 雜訊的模型、抽法、λ、seed、λ＝0 自檢，照 §3 B／偏離 6，偏離 8 沒有改。

**PREREG 偏離 8**（df8b74c，append-only，15 行新增、0 行刪除）：P4 選格的解讀。號碼由 h200 確認可用（h200 之後從 9 開始編）。

**寫本節時已經看過的正式結果**：版本 A 全部（前面各節）；版本 B 全部輸出（判定、偏離 5 的算法、截尾、p90、舊測試、各元件格數）；P4 清單。W4 雜訊：沒有看過任何結果（還沒跑）。

**需要別的 session 做什麼**
- icpe-data：依 `results/icpe_next/main/p4/p4_cells_distinct.csv` 排 P4。
- h200：同步 PREREG 偏離 8（df8b74c）。

**接下來**：第 4 步 P2／P3，等 `_p2`／`_p3` 進 git，在 3 個 TTL 的基礎上算，不重複算和 `sens_rep` 重疊的代表格。目前 git 上：sa_p2 已完成；sa_p3 的第一個 block 失敗（STATUS_TRX：Rust panic），等 icpe-trx 處理。

## 2026-11-06T09:05+08:00 — 論文的兩個 Mooncake 舊測試：照原定義在新、舊模擬器上重算並列（使用者決定）

**使用者決定**（AskUserQuestion，11/6 09:0x，本 session）：論文的 50.5%（conversation 預取、CPU 128 GiB）與「192 格裡截尾 H3600＋排除寫入飽和後 36 格 ≥15%」（EVIDENCE B9）——**新模擬器照原定義重算，和舊值並列**；**完全照舊定義**（rule=min、4 個元件、CPU 128–1,024 GiB、192 格），另外並列 R1／R2 與 O-GPU（只報告，不改偏離 7 的判定）。

**做完什麼**：`code/icpe/main_oldtests.py`；run `20261106-090440-icpe-main-oldtests-verA`（第一次 `…-090416-…` 因我的腳本查表錯誤 exit 1，留了 `FAILED_SCRIPT_BUG`，修正後重跑）→ `results/icpe_next/main/oldtests_verA/`。
- **定義核對 PASS**：在舊 V0 表（論文數字的來源 `results/icpe_h200/t1/p01/sim_V0/x1/`）上重現 50.5%、95／192（依 CPU 28／31／24／12、依慢層 nfs 42／ssd 29／b32 24）、H3600＋不飽和 36（16／12／8／0；ssd 19／b32 17／nfs 0）。
- **發現**：論文的數字來自 **V0** 常數（舊模擬器 sim_V0），新模擬器用 PREREG §4 的 **V1**。所以並列三欄：old_V0（論文）、old_V1（同一舊模擬器、V1 常數）、new（GPU 層，版本 A、3 個 TTL，不是正式值）。

| 測試（rule=min、Cake） | old_V0（論文） | old_V1 | new 版本 A |
|:--|--:|--:|--:|
| T1：conversation O-PF、SSD、CPU 128 GiB | 50.5% | 53.2% | 39.1% |
| T2：192 格 ≥15%（不截尾） | 95 | 97 | 113 |
| T2：不截尾、只算不飽和 | 62 | 65 | 79 |
| T2：截尾 H3600＋只算不飽和 | 36 | 38 | NOT_MEASURED（要 old_p1 的截尾列） |

- 新模擬器不截尾的 113 格裡 O-DROP 41 格；舊模擬器截尾 H3600 後 O-DROP 是 0 格（B2：大半是紀錄太短）。所以新的截尾後格數要等版本 B，現在不能推。
- 只報告：新模擬器的 O-GPU 在同一組 48 格（2 紀錄 × 4 容量 × 3 慢層 × 2）裡 36 格 ≥15%（rule=min）。

## 2026-11-06T08:58+08:00 — 更正：本檔與 PREREG 偏離 7 的時間標記（icpe-data 指出）

- **錯在我**：下面各節原本的節名時間（09:00、09:05、09:15）和「使用者決定 09:0x」是我估的，沒有用 `date`。icpe-data 在 SendMessage 指出：我傳訊息之後 acvlab 的鐘才 08:56。我傳給 icpe-data 的訊息也寫了「09:05 節」，那就是本檔現在的 08:54 節。
- 用 git 與檔案時間核對後的實際時間：開跑版本 A 08:48:46；版本 A 跑完 08:50:41（`exit_code` 的 mtime）；使用者選「commit＋push」與「偏離 7 由 icpe-main 寫」在 08:51–08:53（回報版本 A 之後、偏離 7 的 `date -u` 00:53:20Z 之前）；commit 9cba644／bcd0bad 08:54:39；commit 91a4d9d 08:56:14。下面各節的節名已改成實際時間。
- **先後順序不變**：使用者是在看過版本 A 之後才決定登記；做決定 1–3（08:4x，08:46:20 merge 之前）時只看過 manifest 的摘要。
- 偏離 7 的同步（git 核對）：`origin/icpe-next-data` @ b46635e、`origin/icpe-next-a` @ 1c0973f 的 `PREREG.md` 都和 bcd0bad 逐字相同。
- PREREG 偏離 7 原文寫的「09:0x」也是錯的。PREREG 只往後加，所以在檔尾加了一節「偏離 7 更正」（`date -u` 2026-11-06T00:57:25Z），不改原文。要不要 push、其他分支要不要再同步：等使用者決定。

## 2026-11-06T08:56+08:00 — 偏離 7 的同步狀態

- h200 回覆（SendMessage）：已核過、已同步。我在 git 上確認：`origin/icpe-next-a` @ 1c0973f 的 `PREREG.md` 和 `icpe-next-main` @ bcd0bad **完全相同**（`git diff` 是空的）。h200 下一個偏離編號是 8。
- `origin/icpe-next-data` 的 `PREREG.md` 還**沒有**偏離 7。已用 SendMessage 提醒 icpe-data；要不要同步由 icpe-data 決定。之後合併分支時要注意兩邊的 PREREG 一致。
- P1：背景每 15 分鐘 fetch 一次，等全部 P1 block 進 git（cw 24、tl 6、old 4、bl 2、sa 24）。

## 2026-11-06T08:54+08:00 — PREREG 偏離 7（判定方式的釐清）已寫；版本 A 進 git

**使用者決定**（AskUserQuestion，11/6 08:51–08:53：版本 A 08:50:41 跑完、回報之後，偏離 7 的 `date -u` 00:53:20Z 之前）
- 版本 A：全部 commit，push `icpe-next-main`。
- 判定規則（原始／合成到達各下判定字、Mooncake R1／R2 各自判定、寫入飽和照算另標）：**登記成 PREREG 偏離 7，由 icpe-main 寫**。

**做完什麼**
- **PREREG 偏離 7**（acvlab `date -u` 2026-11-06T00:53:20Z）：只在檔尾加一節（25 行，0 行刪除）。內容＝上面三點、icpe-main 照既有文件的解讀（算判定的元件、工具類的分法、「≥15%」照字面、簡單線索 NOT_MEASURED → 推翻條件「未定」），以及寫的當下已經看過的正式結果。
- **版本 A 的表**：`results/icpe_next/main/verA/`（逐格上限表與 violations 用 xz，解壓後和 run 目錄的原檔 `cmp` 相同；`manifest.json` 有 cmd、context、未壓縮檔的 sha256）。

**需要 h200 做什麼**：偏離 7 在 `icpe-next-main` 的這個 commit。要不要同步進 `icpe-next-a` 由你決定（你同步偏離 5、6 的做法：只取 `PREREG.md` 一個檔）。**下一個偏離編號是 8**。

**需要 icpe-data／icpe-trx 做什麼**：P1 的 block 照常 collect、commit、push 就好；我 fetch 後照 PREREG 偏離 6 算版本 B（主結果＋`_p1`，兩邊同一組 7 個 TTL；偏離 5 的算法並列；`--dead-h 1800`／`3600` 只報告），先核列數。`cw_main_ttl7` 不放（和 P1 重疊）。

**接下來**：等 `{cw,tl,old,bl}_p1` 與 `sa_p1` 全部進 git。

## 2026-11-06T08:51+08:00 — 版本 A（3 個 TTL）完成：對拍 6／6 相同、格數自檢 PASS；判定（版本 A，不是正式值）

**run** `20261106-084846-icpe-main-verA-ttl3`（`/ssd6/hungwei/icpe-next/runs/` 下；cmd.sh、context.txt、stdout／stderr、exit_code＝0；git HEAD bf7915f；牆鐘 1:40、最大 RSS 338 MB；nice 19、單一行程）。
- `next_sim_analyze.py` 原樣，`--pf-target cpu`，6 類紀錄分開跑（避免 `rule=min` 複製單一規則的紀錄）；CacheWise 另跑偏離 5 版（`cw_main`＋`cw_main_ttl7`＋`--rival-only-gpu` 8 個 TTL，內建自檢 PASS）。p90 另跑，只報告。
- **對拍**（`out/cmp.csv`）：重算的 cw 3 TTL（headroom 3,888、violations 221）、cw 偏離 5（3,888、369）、sa 3 TTL（3,888、5,322）和 git 上的 `analysis/cw_main_ttl3`、`cw_main_final`、`sa_main_ttl3` **逐格相同**（去掉 run_id、ts、host）。
- **格數自檢**（`main_summary.py`）：42 個 (trace, rule) × 每個元件都是 54 格，PASS。
- **版本標記**：`A_report`＝偏離 6 的解讀（CacheWise 用偏離 5 版；TraceLab、SemiAnalysis、Mooncake、FreeInference、Bailian 是 3 個 TTL、**沒做過 ≥10% 補跑**——補跑被 P1 取代——所以 ② 偏弱、上限只可能偏高）；`A_ttl3`＝全部 3 個 TTL。② 不含 block 粒度 APC（偏離 2 第 3 點）。**正式值等 P1（版本 B）。**

**判定（版本 A_report；`out/verA_report_verdict.csv`）**〔模擬〕
| 到達 | 類別 | 變體數 | 有格子 ≥15% 的變體 | ≥15% 這一半 | 推翻條件 |
|:--|:--|--:|:--|:--|:--|
| 原始 | 工具類 | 9 | 3：Mooncake toolagent R1、R2；TraceLab c1（FreeInference 6 週最大 0.9%） | 成立 | 未定（簡單線索 W3 NOT_MEASURED） |
| 合成 | 工具類 | 26 | 21：SemiAnalysis 12／12、CacheWise 7／12（N32 s2、N64、N72）、TraceLab c2、c4 | 成立 | 未定（同上） |
| 原始 | 非工具類（只報告） | 5 | 5：Bailian 3、Mooncake conversation R1、R2 | — | 不適用 |

- 寫入飽和：≥15% 的格子裡只有 Mooncake 有（conversation 69、toolagent 72 格），其他紀錄 0。
- CacheWise 3 TTL 與偏離 5 版：算判定的 5 個元件 ≥15% 格數相同（O-GPU 111、O-ORD 48、O-PF 31）。

## 2026-11-06T08:49+08:00 — 開工；使用者對 4 個分析選擇的決定；版本 A（3 個 TTL）開算

**讀了什麼**：CLAUDE.md、NEXT_PLAN_20261104、PREREG（§1、§3、偏離 1–6、偏離 6 補記）、STATUS_DATA（到 08:25）、STATUS_SIM（到 06:14）、STATUS_A（到 07:55）、STATUS_ANA（`origin/icpe-next-ana` 279eb7e，到 03:05）、STATUS_TRX（`origin/icpe-next-trx` 87a5a86，到 08:12）、`code/icpe/next_sim_analyze.py`。

**merge**：`origin/icpe-next-trx` @ 87a5a86 → a586e83（只加 `sa_main` 12 個 block、`analysis/sa_main_ttl3`、`sa_main_ttl7/cells.csv`、STATUS_TRX；沒有程式改動）。

**使用者的決定**（AskUserQuestion，11/6 08:4x，本 session；都是 PREREG 沒寫到的分析選擇）
1. 版本 A 照提的方式開始算。
2. **判定範圍**：推翻條件的「真實的工具類紀錄」——**原始到達與合成到達各下一個判定字**，不合併。
   - 原始到達：TraceLab c1、Mooncake、FreeInference、Bailian。合成到達：CacheWise、SemiAnalysis（固定並行 N）、TraceLab c2／c4（開始時間壓縮）。
   - **提問時我已經看過 SemiAnalysis N16 s1（合成到達）有 ≥15% 的格子**（見下面「已看過」），問題裡有寫明。
3. **Mooncake**：**R1、R2 各自下判定**（不是以前慣例的 `rule=min`；`rule=min` 的列照程式輸出、只報告）。
4. **寫入頻寬飽和的格子**（`write_saturated`）：照算、另外標格數，不排除。
- 工具類的分法（不是新決定，照 NEXT_PLAN W2／§5.2）：TraceLab、CacheWise、SemiAnalysis、FreeInference、Mooncake toolagent＝工具類；Mooncake conversation、Bailian＝只報告。
- 要不要把 2、3 登記成 PREREG 的釐清：→ 已登記為偏離 7（見 08:54 那節）。

**已看過哪些正式結果**（照偏離 5 的慣例照實列出）
- 開工前讀 manifest 時，`stdout_tail` 欄裡的摘要表（我只印了前 600 字元）：
  - `results/icpe_next/b/analysis/sa_main_ttl3/manifest.json`（run `20261106-074822-icpe-trx-analyze-sa_main-ttl3`）：SemiAnalysis N16 s1 的 O-ALL(report)、O-DROP、O-GPU、O-NOW、O-ORD 五列（格數、最大、中位數、≥15%／≥5% 格數）。O-GPU、O-ORD、O-ALL(report) 有 ≥15% 的格子；O-DROP、O-NOW 最大 <3%。
  - `analysis/cw_main_ttl3`、`cw_main_final` 的 manifest：CacheWise N16 s1、s2 的部分元件列，最大都 <1.2%、0 格 ≥5%。
- 讀 sweep 時只看了欄名和設定欄（trace、rule、gpu、slow、restore、gpu_GiB、gpu_write、pf_target、policy／gpu_policy 的種類數），沒看任何時間欄。
- 佇列 log 只看了 start／end 列。
