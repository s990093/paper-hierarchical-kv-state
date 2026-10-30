# PAPER.md v3 稽核（2026-10-30）

> 對象：`docs/icpe_paper_20261030/PAPER.md` v3（commit 7769949，476 行）、`EVIDENCE.md`（219 行）、`figs/`（含新圖 `fig1_system.png`、`fig2_method.png` 與 `DIAGRAM_NOTES.md`）。
> 方法：照 coordinator 給的四個優先順序：(1) 一頁看懂＋摘要；(2) §3.3 公式對程式與登記；(3) v1 稽核修過的項目有沒有掉；(4) 圖說對 sidecar／圖示說明。我打開原始 CSV／JSON 重算，讀 `code/p3x/x1_sim.py`、`code/icpe/x1_gpu.py`、`code/p3x/x1_fi.py`、`code/m8_trace_oracle.py` 與登記檔。只寫這個檔；沒有改其他檔。
> 本稽核由一個 Claude subagent 執行——**同一模型家族，不是 cross-model review**。行號＝v3 PAPER.md 的行號。

## 摘要

| 嚴重度 | 數量 |
|:--|--:|
| 高 | 2 |
| 中 | 7 |
| 低 | 14 |

- **數字抄錄**：v3 引用的數字都能在 EVIDENCE.md 或 sidecar 找到。7 張資料圖的 `plotted_values` 與輸入檔 sha256 和 v1 逐值相同，只改了版面與標籤。新加的數字（表 5、B8、B9、B1-R、B1-ORD、§0 的 cpu_frac_ws 逐週值）我都從原始檔重算過，見附錄 A。
- **§3.3 公式**：(1)–(6) 與判定規則都和程式一致，沒有一條和程式衝突。只有兩處小的措辭不一致（#13、#14）。
- **v1 的修正掉了幾項**：v2 摘要裡的「截尾＋排除飽和後 36 格」「TTFT 不含排隊／GPU 重播檢查」「0.81／0.79 是 1/8 重播、SSD 級」「提前量是合成的」，在 v3 的摘要與一頁看懂都不見了（#1、#4、#5）。§11 掉了「測過的」（#6）。新的圖 2 把 v1 #8 修掉的「固定規則＝底」又畫回去，而且規則本身寫錯（#2）。
- **v1 稽核本身的更正**：v1 #10 寫「FreeInference 128 GiB 占工作集 0.78–1.11%」只看了第 1、6 週；原始檔逐週是 0.78／1.11／1.60／2.08／7.27／8.06%（EVIDENCE 差異 14 指出，我已重算確認）。結論方向不變，v3 L344 的「2 週比 Mooncake 還吃緊」正確。

## 問題表

| # | 嚴重度 | 位置（§／行號） | 問題 | 證據（檔案、數字） | 建議修法 |
|--:|:--|:--|:--|:--|:--|
| 1 | 高 | 一頁看懂 L13「Mooncake 上，192 個設定裡 95 個 ≥15%」；摘要 L37「192 個設定有 95 個快 15% 以上」；§5 L235、§5.2 標題 L267「將近一半的格」；§11 L426「將近一半的格達到 15%」 | (a) **單位錯**：192 不是 192 個設定，而是 48 個設定 × 4 種上界（名詞表 L46 自己把「格」定義成含上界種類）。(b) 95 裡有 **28 格是 O-DROP**，論文自己（L282）與 X1 登記都說 O-DROP 大半是 trace 太短造成的，截尾後是 0。另有 **33 格寫入飽和**，論文 L193 說這些格偏樂觀。(c) X1 登記的總結規則是「以較保守的版本為準」，也就是截尾版。保守版是 67/192（H＝3,600 s）；再排除飽和是 **36/192＝19%**。以設定計：至少一種上界 ≥15% 的設定，原始 36/48，保守 19/48。「將近一半」只在登記說不能拿來做總結的原始計數下成立。§5.2 本文 L269 有寫 36，但一頁看懂、摘要、§11 都沒有 | `results/icpe_h200/t1/p01/sim_V0/x1/x1_headroom_components.csv`、`x1h3600_headroom_components.csv`（run `20261028-070955-icpe-t1-x1-an-V0`、`20261028-071003-icpe-t1-x1-h3600-V0`；Cake、rule＝min）：≥15% 95（O-PF 34、O-ORD 33、O-DROP 28；飽和 33）；H3600 67；H3600＋不飽和 36；設定數 48。`docs/phase3_20261017/explore/X1_oracle_decomposition.md` L7「總結以較保守的版本為準：有看頭，只靠 O-PF 與 O-ORD」、L116（飽和格要寫明靠 free 語意） | 頭條改成「Mooncake 上，48 個設定 × 4 種上界的 192 格裡，截尾並排除寫入飽和後仍有 36 格 ≥15%（不截尾 95 格）」；刪掉「將近一半」，或明寫是不截尾的計數 |
| 2 | 高 | 圖 2（`fig1_system.png`）(b) 的方框「底（不預測）：CPU 層全部載入；慢層用 Cake」；圖說 L108 | 有兩個錯。(i) 規則本身和 §6 L325、A5 衝突：BF16 在 GPU 忙時是 **Cake** 勝，不是全部載入（「不能寫的」#9）。(ii) 模擬的底根本不是這條規則，而是 Cake 用在整個前綴（§9 L391、EVIDENCE §0「X1 模擬的還原底」、「不能寫的」#20）。DIAGRAM_NOTES.md L37 要求圖說寫明這一點，L108 的圖說沒寫。這是 v1 #8 修過之後又回來的問題 | `results/icpe_h200/p01/p01b_cells.csv`：Llama CPU 忙 E0 Cake 0.152 vs 全部載入 0.238 s；Qwen 0.068 vs 0.236 s。`code/p3x/x1_sim.py` `View.restore()` mode＝cake 對 lay∈{1,2,3} 一律用 Cake | 方框改成「CPU 層：INT4 全部載入、BF16 在 GPU 忙時用 Cake；慢層用 Cake（GPU 量到的最佳規則）」；圖說加「模擬器把 Cake 用在整個前綴」 |
| 3 | 中 | 一頁看懂 L10「Cake…在慢層上比全部載入快 16.5–87.5% [A5]」；表 7 L320 | A5 支持「測過的 16 個慢層格 Cake 都勝」，但這個範圍有四個限定沒寫。(i) 用的是 **fa3** 那一組；論文的主重算是 sdpa（表 2 L126），sdpa 組是 11.0–85.9%。(ii) 16 格含 8 格 **BF16 存在慢層**（E0），但本文的系統模型慢層只存 INT4（L94、L127）；只算 INT4 是 16.5–71.9%（sdpa 11.0–67.2%）。最大值 87.5%＝Llama、BF16、NFS 級、閒。(iii) 只量了 b32 與 NFS 級，沒有 SSD 級、b16、b48。(iv) 這是單次 32K（64 塊）還原的微基準，不是 TTFT；NFS 級的常數沿用 MI300X | `results/icpe_h200/p01/p01b_cells.csv`：1−new_cake/new_loadall＝0.165–0.875；1−sdpa_cake/sdpa_loadall＝0.110–0.859；E1 只算 0.165–0.719／0.110–0.672；`grid/llama_g5b_fa3/restores.csv`：length 32768、bw_name {cpu, b32, nfs} | 一頁看懂改成「在 32K 還原的微基準裡，b32 與 NFS 級上 Cake 比全部載入快（INT4：11–67%，sdpa）」；表 7 註明 fa3 組與格式 |
| 4 | 中 | 摘要 L37；一頁看懂 L11、L13 | 摘要沒有說回來時的結果是**模擬**，也沒說 TTFT 不含排隊、經過縮小 GPU 重播檢查；只寫「我們在一張 H200 上做極限研究」，讀起來像 H200 實測。違反「不能寫的」#13（模擬上界要和 B6 一起寫）；#7 的精神也受影響（含排隊時 GPU 上的預取上界只剩 8.1／4.7%）。v2 摘要原本有這些，v3 刪掉了 | EVIDENCE L173、L167；v2 摘要（commit 9d8dc49）有「模擬以 GPU 縮小重播檢查（TTFT 不含排隊時…0.911–0.987）」 | 摘要第二句後補一句：「回來時的上界是 trace 模擬（CPU 層與重算常數在 H200 上量），經 1/8 GPU 重播檢查；TTFT 不含排隊」 |
| 5 | 中 | 一頁看懂 L16「GPU 上拿到 oracle 預取改善的 0.81 [C2-GPU]」 | 只引兩格裡較高的 0.81，漏了 0.79，也沒寫範圍（1/8 縮小重播、SSD 級、Llama、CPU 16／32 GiB、不含排隊）以及提前量是合成的。待決 3 自己建議引 0.81／0.79。v2 摘要原本有範圍與「提前量是合成的」 | `results/icpe_h200/t1/gpu/p02/analysis/p02_signs.csv`：0.1452/0.1792＝0.810、0.1290/0.1624＝0.794；EVIDENCE §0「打字提示的提前量」 | 改「1/8 縮小重播、SSD 級上拿到 0.81／0.79（提前量用打字速度模型合成）」 |
| 6 | 中 | §11 L426「FreeInference 上每一種上界都低於 15%」 | 掉了「測過的」。X5（知道誰不回來）在 FreeInference CPU 16–64 GiB 沒量（NOT_MEASURED，「不能寫的」#19）；一頁看懂、§5、表 4 都有寫範圍，只有 §11 漏了。v1 #5 的修正在這裡掉了 | `results/icpe_h200/p12/x5/x5_verdict_fi.json`（主格 CPU 128–1,024）；表 4 L248「沒量」 | 加「在測過的 CPU 容量」 |
| 7 | 中 | §3.4 偏離表 L199–208；§3.4 L197「每個判準在看到數字之前寫進登記檔」 | 偏離表漏了一項會碰到頭條的偏離：**FreeInference 是 X1、X5 登記之後才加的**（加的時候還沒看過任何結果），登記寫明「判定列兩版，總結以不含 FreeInference 的版本為準」。v3 的題目與故事框架（「預知值多少要看 trace」）正是建立在 FreeInference 上。截尾 H 也是事後的穩健性檢查（X9 提醒）。H200 上的 P1-2、T5 有各自的事先登記，但原登記的「總結以不含版為準」沒有被撤回 | `docs/phase3_20261017/explore/X1_oracle_decomposition.md` L133；`X5_precision_capacity.md` L16–17；`docs/phase3_20261017/explore/README.md` §4 L127–130；`code/p3x/x1_sim.py` L88–89 註解 | 偏離表加兩列：「FreeInference 是登記後、看結果前加入的；原登記的總結只用 Mooncake」「截尾 H 是事後的穩健性檢查」；並說明 P1-2／T5 的重新登記 |
| 8 | 中 | §9 限制 L395（只提 tokenizer 近似）；名詞表 L58 f(i) | f(i) 在 H200 上**只量到第 0–79 塊**（40,960 token），之後是線性外插。FreeInference 第 1 週 prompt 中位 89K token；X5 文件記載 FreeInference 的回來請求 34.8% 超過 128K（Llama-3.1-8B 的 context 上限），「全部是外插」。所以 FreeInference 的大部分還原成本靠外插，而且有些長度模型根本跑不了。§9 沒有寫 | `results/m7_write_policy_h200/calib_c1.csv`：f_chunk chunk_idx 0–79；`code/m8_trace_oracle.py` `load_f()`「i ≥ 80 線性外插」；`code/p3x/x1_sim.py` `f_array()`；`docs/phase3_20261017/explore/X5_precision_capacity.md` L16（3090 分支的 X5；H200 X1 的比例 NOT_VERIFIED）；`docs/phase3_20261017/explore/README.md` L129（第 1 週中位 89K） | §9 加一條：「f(i) 只量到 40K token，之後外插；FreeInference 有相當比例超過 Llama 的 128K 上限」，並報 X1 FreeInference 超過 40K／128K 的請求比例 |
| 9 | 中 | 圖 2 圖說 L108、圖 3（`fig2_method.png`）圖說 L142 | DIAGRAM_NOTES.md L34–39 列了圖說必須寫的四點：長條是示意；圖 1(b) 的底≠模擬器；MPS 端到端是模擬；重播只驗 conversation／SSD／Llama、不含排隊。兩個圖說都沒寫。圖 3 把「INT4、MPS 開與關」放在「在 GPU 上實測」框裡，但 MPS 端到端是模擬（§8.2 L372） | `figs/DIAGRAM_NOTES.md` L34–39；`fig2_method.png` 右框 | 照 DIAGRAM_NOTES 補圖說；圖 3 的 MPS 標「微基準實測，端到端模擬」 |
| 10 | 低 | §11 L429「上界低於 15%，就不必蓋回來時的預測器」 | 和本文的判定字衝突：5–15% 是「可能」，不是「死路」；專案規則是 5–15% 停下來問人。FreeInference 的淘汰順序 11.81% 就是「可能」。這句是〔判讀〕，沒有標 | 表 4 L249；CLAUDE.md §7 | 改「低於 5% 就不必蓋；5–15% 要再量」，或標〔判讀〕 |
| 11 | 低 | §1 L70「在 FreeInference 上幾乎沒有空間」 | CPU 16–32 GiB、NFS 級最高 11.8%，判定是「可能」；X5 在小 CPU 沒量 | B3-T5、B3-X5 | 加「CPU ≥128 GiB 時」或「最高 11.8%（可能）」 |
| 12 | 低 | §3.1 L118「harness 的重算比 vLLM 慢（fa3 時 1.30／1.32 倍）」 | 主重算是 sdpa（表 2 L126），差距是 1.75／1.76 倍；正文只引比較小的 fa3 那組（§9 L396 兩個都有） | E1、E2 | 正文寫「sdpa 1.75／1.76 倍（主結果）、fa3 1.30／1.32 倍」 |
| 13 | 低 | §3.3 L172「四種回來時 oracle 都從 trace 讀 next(s)（看 session 獨有的第一塊）」 | 預取不是：O-PF 用的是最後一塊（L176 自己也這樣寫），和 L172 的「都」矛盾 | `code/p3x/x1_sim.py` `demote()`：`XT.next_access(self.acc, self.sb[s][-1], self.cur)`；`next_req()` 用獨有的第一塊 | 改「淘汰順序與丟棄看獨有的第一塊，預取看最後一塊」 |
| 14 | 低 | §3.3 L193「用 2σ 的實驗另要求差的 bootstrap 95% CI 下界 >0」 | D2（INT4 速度）不是 2σ 實驗，但它也要求配對差的 CI 不含 0；A1 則沒有 CI 條件。寫成「只有 2σ 的實驗才要 CI」不準確 | `m10_precision_h200/speed_*/k0_verdict_int4.json` 的 `rule`「中位數的差 > δ\*，且配對差 bootstrap 95% CI 不含 0」 | 改成依實驗列出（A1：只看中位數；D2 與 2σ 實驗：另要 CI） |
| 15 | 低 | 公式 (2) L164–166 | 還有一點沒寫：已預取到 CPU 的 INT4 塊，要等預取讀完才能載入（`max(lt, pf_done)`）。「含 I/O 線上的排隊」大致涵蓋，但預取等待是另一種延遲 | `View.restore()` k＝2 分支 | 括號補「預取中的塊要等預取完成」 |
| 16 | 低 | §3.4 L197、公式 (6) | FreeInference 的 $\rho_w$ 取各週的**最大值**（寫入飽和的判定依此），文中沒寫 | `cmd_analyze`：`rho_w=("rho_w", "max")` | 一句話說明 |
| 17 | 低 | 表 8 L337–340 | 同一欄的「值得做」用了兩種門檻：X1 要同一組 ≥2 種 CPU 容量都 ≥15%；X5 只要任一主格 ≥15%（L183）。表中沒標 | X5 文件 L75；`cmd_analyze` | 在 X5 那列註明「X5 的登記：任一主格 ≥15%」 |
| 18 | 低 | 圖 3（`fig2_method.png`）底部的判定尺 | 尺是逐格分區（<5%／5–15%／≥15%），但判定是分組的（值得做要 ≥2 種容量；死路要所有格、含全部載入）。圖裡用「有看頭」，正文用「值得做」 | §3.3 L182；`cmd_analyze` | 圖內字改「值得做」，圖說補「判定是分組的」 |
| 19 | 低 | 圖 1 圖說 L20「圖中代號見文末對照表」 | 對照表 L444 標「內部用，投稿版刪除」；刪掉後圖內的 O-PF、X5、T1、T3、STAIR 沒有解釋 | PAPER L444 | 圖內改用名詞表的名稱，或把代號放進圖說 |
| 20 | 低 | 圖 4（`fig7_validation.png`）圖說 L154 | 正文 L150 寫了 TTFT 不含排隊，圖說沒寫（sidecar notes 有寫） | `fig7_validation.sources.json` notes | 圖說加「TTFT 不含排隊」 |
| 21 | 低 | 待決清單 L22–33 | 待決 1（故事框架）與待決 3（打字提示的數字）在內文沒有〔待決〕標記；其他待決都有 | PAPER L239、L289、L325、L346、L359、L373、L381、L212 有標 | 補標記 |
| 22 | 低 | `figs/DIAGRAM_NOTES.md` L3 | 寫「PAPER.md still embeds the `.svg` files」，v3 已改嵌 PNG，說明過時 | PAPER L106、L140 | 更新說明 |
| 23 | 低 | §3.1 L138、§9 L395、§10 | 外部說法 NOT_VERIFIED：Nixon et al. 的資料卡引用、Mooncake「一小時抽樣」（PREFLIGHT [U10]／[U11]）、FAST'25、PBKV 等的底是「沒命中就重算」 | 不在 EVIDENCE.md | 交給 citation-audit |

## §3.3 公式逐條對程式

| 公式／敘述 | 程式 | 結果 |
|:--|:--|:--|
| (1) $\ell_{BF16}=b/B_{CPU}$；$\ell_{INT4}=b/B_{層}+d$；寫入中讀取乘 $k_{read}$ | `Chan.read()`（`t_free_w > st` 時 `dur *= k`）；`View.restore()` k＝1：`cpu.read(BF)`，k＝2,3：`read(i4)+d` | ✓ |
| (2) Cake：落後的一邊前進一塊，會合於 $m$，取 max | `View.restore()` mode＝cake：`if rt <= lt` 重算 `f[p]`，否則從 `q=L-1` 往前載入；`end=max(rt,lt)` | ✓（預取等待見 #15） |
| (3) TTFT＝還原＋$\sum_{i=L}^{n_r-1}f(i)$；$T(\pi)$＝回來請求平均 | `ttft = base + Fc[n_all] − Fc[L]`；`summarize()` 的 `ret = kind==cont`、`mean_ret` | ✓ |
| next(s) 看獨有的第一塊 | `next_req()`：`excl_start` 的第一塊，沒有就用最後一塊 | ✓（O-PF 例外，#13） |
| 淘汰順序：Belady／Belady-Size $(\text{next}-r_{now})\cdot\text{bytes}$，取較快 | `select()` bel：`max (nx, −last)`；bels：`dist × max(freed,1)`，`freed＝(len−e)×BF`；`COMPONENT_POLICIES O-ORD = [O-BEL, O-BELS]`，`o_best = min` | ✓ |
| 丟棄：回合結束 next＝∞ 就丟；輪到淘汰時 next＝∞ 就丟，否則降級；截尾 H | `step()` hook `odrop and dead(s)`；`victim_action` `onow`；`dead()`：`t[cur] > dead_cut` 時回 False | ✓ |
| 預取：降級時記最後一塊的 next；讀取線空時先預取最早的；只搬 $[m_{cpu},L)$；INT4 進 CPU；只准降級 next 更晚的 | `demote()` 推 `(T, s)` 進 `pfheap`；`advance_prefetch()` 從 `slow.t_free` 起、`bl[m:]`、`fmt=2`；`make_room()` 只選 `nx > Tc` | ✓ |
| (4) $h_c=(T_{rival}-T_{oracle})/T_{rival}$，兩者各取族內最小；Mooncake 取兩種規則的較小值 | `cmd_analyze`：`o_best`、`rv_best` 取 min；`idxmin` 的 headroom → rule＝min | ✓ |
| 判定：同組 ≥2 種容量 ≥15% → 值得做；所有格（含全部載入）<5% → 死路 | `cmd_analyze`：`ncpu >= 2`（只算 cake）；`all_lt5 = (HC.headroom < 0.05).all()`（含 load） | ✓；與 X1 登記 L112–115 一致 |
| (5) $L_r=\min(\max(\Delta_r-10,0)\cdot c_{tok}/(51.56\cdot5/60),\ \text{think}_r)$；TPOT 50 ms | `x1_gpu.leads()`：`TEMPLATE_OV=10`、`WPM_DESKTOP=51.56`、`CHARS_PER_WORD=5`、`TPOT=0.050`；`cpt_llama_English`＝4.2693（`x3_typing_consts.json`）；think＝間隔 − 上一輪 prefill 估計（f 用閒時）− 輸出 token×TPOT；提示時間＝$t_r-L_r$（`HintEngine`） | ✓ |
| (6) $\rho_w$＝慢層寫入忙碌時間÷trace 長度 | `summarize()`：`rho_w = v.slow.busy_w / span` | ✓（FreeInference 取週最大，#16） |
| L138 FreeInference 32×16 token 併成 512、以最後一塊 id 當鍵、用 token 數驗算 | `x1_fi.py`：`SUB=32`、`key=block_ids[32j+31]`、`assert rate >= 0.95` | ✓ |
| L197 模擬是確定性的 | `x1_sim.py` 的 `random` 只出現在 selftest | ✓ |

## v1 稽核的高／中項在 v3 的狀態

| v1 # | 內容 | v3 |
|:--|:--|:--|
| 1 | agent trace 只對 FreeInference 成立 | ✓ 保留（全文改 FreeInference；toolagent 有報告 L281） |
| 2 | §7 條件只對 X5 | ✓ 保留（L346）；但 Mooncake 的頭條計數有新問題，見 #1 |
| 3 | 「常數都在 H200 上量」 | ✓ 保留（§9 L388、圖 1／5 圖說、圖 3）；摘要不再這樣寫 |
| 4 | 打字提示只對預取 | ✓ 保留（L16、L287、L303） |
| 5 | X5 在 16–64 GiB 沒量 | 部分掉了：§11 L426（#6） |
| 6 | 提前量是合成的 | 正文 ✓；一頁看懂 L16 沒寫（#5） |
| 7 | 同動作的對手 | ✓ 保留（摘要「做同樣動作的最好不預知策略」、B1-R L280） |
| 8 | 模擬的底≠固定規則 | §9 ✓；**圖 2 又畫回去**（#2） |
| 9 | 精度是品質檢定 | ✓ 保留（L216） |
| 10 | 條件 2 | ✓ 保留（L344；並更正了我在 v1 給的範圍） |
| 11 | B2 混了兩條 trace 與全部載入 | ✓ 保留（L282 只算 Cake 47.4%） |
| 12 | 三條 trace | ✓ 保留 |
| 13 | 「本來就沒差距」要加範圍 | ✓ 保留（L253、L419） |
| 14 | B5 是純模擬 | 正文 ✓（L283、表 8、§9）；摘要與一頁看懂沒寫「模擬」（#4） |
| 15 | ≤0.30 vs ≤2.44 | ✓ 保留 |
| 16 | E2E 0/32 計數 | ✓ 保留（L372「32 格裡 24 格可判」） |
| v2 摘要新增的四個限定 | 36 格、TTFT 不含排隊／重播、0.81／0.79 的範圍、合成提前量 | **都從 v3 摘要與一頁看懂刪掉了**（#1、#4、#5） |

## 圖說對 sidecar

| 圖（v3 編號） | 檔案 | 結果 |
|:--|:--|:--|
| 圖 1 | fig3_scorecard | 數值同 v1 ✓；上圖 SSD／閒／128–256 GiB、X5 長條＝conversation 24 格最大（b48、閒、128）✓；MI300X 常數的說明 ✓；代號問題見 #19 |
| 圖 2 | fig1_system（沒有 sidecar） | (b) 的底錯（#2）；長條是示意，沒說（#9）；(a) 的寫入／降級／丟棄／預取／還原箭頭與 §2 L94–95 一致 ✓ |
| 圖 3 | fig2_method（沒有 sidecar） | 三條 trace ✓；「CPU 層＋GPU 重算：H200 實測」「SSD／NFS：沿用前一個平台」✓；headroom 公式＝(4) ✓；0.911–0.987 ✓（沒寫不含排隊）；MPS 放在 GPU 框（#9）；判定尺（#18） |
| 圖 4 | fig7_validation | 0.911–0.987、10 格在 [0.9, 1.1] ✓；「模擬比實測略慢」＝比值 <1 ✓；沒寫不含排隊（#20） |
| 圖 5 | fig9_fi_capacity | 「淘汰順序與預取」✓；「空心圈只在 16–64 GiB 高於實線」✓ |
| 圖 6 | fig8_bridge | NFS 換成最好的對手後 5.6–11.8% ✓ |
| 圖 7 | fig4_hint | 0.81／0.79 ✓；y＝x 以上全是 NFS 級（Cake 4、STAIR 2）✓ |
| 圖 8 | fig5_int4 | 32K、每格 30 對、CPU 層沒畫 ✓ |
| 圖 9 | fig6_mps | 「登記版常數，只畫 Llama、conversation」✓（改成 T1 版是待決 7） |

## 附錄 A：v3 新數字的重算

| 位置 | 數字 | 原始檔 | 重算 |
|:--|:--|:--|:--|
| 表 5、L269、L278 | 95/192；128:28、256:31、512:24、1,024:12；截尾＋不飽和 36（16／12／8／0）；每容量 48 格 | `x1_headroom_components.csv`、`x1h3600_headroom_components.csv` | ✓；另：O-DROP 28、飽和 33、H3600 67、設定 36/48 → 19/48 |
| L281 | toolagent 40/96；截尾＋不飽和 12 | 同上 | ✓（36＝conversation 24＋toolagent 12） |
| L280 | 對 LRU2 45.3% | `x1_sweep.csv.gz`（R2、128 GiB：LRU 0.5154、LRU2 0.4667、O-PF 0.2551） | ✓（256 GiB 40.3%） |
| L303 | 淘汰順序 27.6／31.0% | `x1_headroom_components.csv` | ✓ |
| L344 | FreeInference 有 2 週比 Mooncake 吃緊 | `t5/p12/x1/x1_sweep_fi.csv.gz` `cpu_frac_ws`（128 GiB 逐週 1.11、2.08、7.27、8.06、1.60、0.78%）；Mooncake 1.20／1.28% | ✓ |
| L10、表 7 | 16.5–87.5% | `p01b_cells.csv` | 數值 ✓，範圍見 #3 |
| 表 4 | 0.22%、0.17%、≤4.89%、≤11.81%、≤0.65%、2.44%、3.92% | `fig3_scorecard.sources.json`、`t5/p12/x1/x1fi{,user}_headroom_components.csv`、`x5_verdict_fi.json` | ✓ |
| L369 | MPS 100／87／75／50% | `fig6_mps.sources.json` | ✓ |
| (5) | 10、4.269、51.56、5、60、50 ms | `x1_gpu.py` L48、`x3_typing_consts.json` | ✓ |
| 其餘 | 與 v1 相同的數字 | 見 `AUDIT.md` 附錄 A | 數值沒變 |
