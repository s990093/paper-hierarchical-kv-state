# T3：還原規劃器換成更強的（照 CacheFlow 的想法自己寫的簡化版），ICPE 的判定字會不會變

> **性質**：判準登記（事先寫死）。登記時間見文末。登記之後這一節以上不改；和登記不同的做法寫成文末的「偏離 N」。
> **來源**：使用者 2026-10-29「時間夠的話做一個敏感度檢查：在模擬器裡加一個簡化的『token × 層階梯』規劃器……看結論會不會變。舊故事也要做……加入實驗」。
> **為什麼**：ICPE 的 B、C 節都是「在 Cake 之上，預知值多少」。CacheFlow（arXiv 2604.25080，v2 2026-09，沒有程式）的規劃器比 Cake 強。規劃器更強時，oracle 剩的空間可能更小（負面結論更穩），也可能不同。
> **範圍**：只有 H200 的常數、CPU 層用 T1 的（`P3_TIER=h200cpu`）；只在模擬器裡（CPU、head node），不上 GPU。**這不是 CacheFlow 的程式，也不是它的重現**；論文裡寫成「我們照它的想法寫的簡化規劃器」。

---

## 1. 規劃器 STAIR（`x1_sim.View`，`X1_PLANNER=stair`；沒設＝原本的 Cake，逐位元不變）

X1 原本的 Cake（`View.restore`，mode＝cake）：從前面重算、從後面載入，在中間會合；會合點以「塊」為單位；CPU 層和慢層的載入**排在同一條線上**（一塊接一塊）。

STAIR 和它只差三點：
1. **層的粒度**：CacheFlow 把（token, 層）切成階梯。X1 的模型裡每一層的成本一樣、重算成本 f(i) 隨位置遞增（凸），這時最好的階梯就是「每一層同一個邊界」（Jensen），所以階梯在 X1 裡等於「會合點可以落在一塊的中間」，粒度 1/32 塊（Llama-3.1-8B 32 層）。
2. **兩條載入線同時用**：CPU 層的塊走 CPU 那條線、慢層的塊走慢層那條線，同時開始；兩條加起來不能超過 GPU 的輸入頻寬（＝CPU 層的讀取頻寬，T1 的 43.59 GiB/s）。
3. **規劃**：用名目成本（各線目前的排隊、寫入中的讀取倍數 k、INT4 的反量化 d）找讓 max(重算, CPU 線, 慢層線, 輸入頻寬) 最小的會合點，再用 X1 原本的 I/O 線（`Chan.read`）真的執行。

- **刻意偏樂觀**（＝比真的 CacheFlow 更強或一樣）：不算層與層之間的相依等待（CacheFlow 用邊界 activation 減少這個等待）。
- **沒有做的**（X1 本來就沒有）：同一批請求一起規劃（X1 一次一個請求、不排隊）、多張卡。這兩個是 CacheFlow 主要的加速來源之一，照實列為限制。
- **自檢**（`code/icpe/t3.py selftest`）：沒設 `X1_PLANNER` 時 `x1_sim.py selftest` 的四條不變量照過；隨機 2,000 個還原（新的 I/O 線、隨機的 CPU／預取／慢層排列）STAIR ≤ Cake（容差 1e-9 s）。

## 2. 重跑什麼（和 T1 同一條指令，只多 `X1_PLANNER=stair`，輸出放 `results/icpe_h200/t3/`）

| ICPE 列 | 指令（同 `code/r_h200_t1.sh`） | 輸出 |
|:--|:--|:--|
| B1／B2／B7（V0、V1、V2） | `p3_fvar.py --variant V x1 sweep`＋`x1 analyze`（原登記、`--dead-h 1800`、`--dead-h 3600`） | `t3/p01/sim_V*/x1/` |
| B3（X1 FreeInference） | 同 T1 的 P12-X1FI sweep＋analyze（含使用者發起） | `t3/p12/x1/` |
| C2（全尺寸 HINT-PF） | `x1_gpu.py fullsim` | `t3/p02/` |

- **不重跑**（照實列為限制）：X3（C1）、X4（B4）、X5（B5）、E2E-C、GPU 實測（A 節、B6）。
- 另外只報告：conversation 一個重播上，同一份放置、同一份 I/O 線狀態下，每個回來請求的 STAIR ÷ Cake 還原時間（看「更強」強多少）。

## 3. 判準

- 對 T1 版（`results/icpe_h200/t1/`）逐項比判定字，用 T1 的同一套定義（`code/icpe/t1.py summary`、`cmp_csv`）：B1／B2／B7 的 X1 判定字（V0／V1／V2 × 原登記／H1800／H3600）、B3 的 X1 FreeInference 判定字、C2 的 HINT-PF 拿到 O-PF 的比例（SSD／b32／NFS 中位數；>1 的格數，只報告）。
- **全部相同** → 「ICPE 的判定字對更強的規劃器穩健」。數字（headroom 變大還是變小、跨門檻的格子）照實報告。
- **任何一個判定字不同** → 先停下來回報使用者，不改 ICPE_EVIDENCE。
- 某一項跑不出來 → 那一項 NOT_MEASURED，照實列出。

## 4. 執行

- `code/icpe/t3.py`：`selftest`（上面的 STAIR ≤ Cake）、`ratio`（§2 的只報告比值）、`report`（T1 對 T3）。
- 指令都在 head node 跑（`source code/env_h200.sh; export TIARA_ALLOW_NO_SLURM=1 P3_CONST=h200 P3_TIER=h200cpu X1_PLANNER=stair`，`m7run`）；workers 不超過 8（head node 共用、62 GB）。

## 5. 已經看過的數字（登記前）

- T1 的全部結果（`t1_compare.json`）、ICPE_EVIDENCE、E2E-C。
- CacheFlow 的摘要（2.24–3.00 倍還原延遲、1.64 倍端到端，對「既有的方法」）。
- **沒有看過任何 STAIR 的數字**（程式還沒寫）。

---

## 登記

- 登記時間（`date -u`）：2026-10-29T04:10:54Z
- 登記時 repo HEAD：`ddaccf2`（本檔隨下一個 commit 進 git）

## 偏離（登記之後）
