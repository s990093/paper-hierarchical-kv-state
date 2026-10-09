# 09 消融報告：「寫入時決定」在什麼條件下不會被 Cake 抵消

**日期**：2026-10-09
**設計**：[08_ablation_plan.md](08_ablation_plan.md)（判準開跑前寫死；§8 是開跑後加的對照組 S4B，只會讓結論更保守）
**資料**：`results/m7_write_policy_mi300x/`（`sim_*.csv`、`b2_*.csv`，每列都有 run_id）
**程式**：`code/m7_sim.py`（模擬）、`code/m7_ablate_analyze.py`（模擬掃描分析）、`code/m7_b2_analyze.py`（GPU 確認分析）、`code/m7_run_ablate.sh`（GPU 執行鏈）
**判讀**：self-review，非 cross-model（沒有開 zero-context reviewer）。

---

## 0. 結論

**照 08 §6 的判準：在這台機器（MI300X）與這個模型（Llama-3.1-8B，GQA）上，寫入時的位置決定被 Cake 的讀取時調整抵消。**

- GPU 實測 3 個設定 × 2 個容量，共 16 組「寫入時決定 vs 對手」的比較，**沒有一組同時贏過「最好的不看位置策略」與 S4B 各 ≥5%**。
- 模擬裡最有希望的那組（doc＋hold＋CPU 3.69＋NFS＋50% 容量，seed 0），GPU 上 S5L 比最好的不看位置策略快 **18.9%**，但只比 S4B 快 **0.7%**。換成 seed 1，S5L 反而比寫穿（S1）慢 **6.8%**。
- 有效的東西有兩個，但都**不是「寫入時」**：
  1. **早點寫到慢的層**：在「搬移卡在關鍵路徑上」（hold）時很重要，但寫穿（S1、S2b）就做得到，不用看位置。chat＋hold 時，S1 比 S5 快一倍以上。
  2. **「b 以前的前段不值得放 CPU」這條規則**：等 CPU 滿了才套用（S4B）效果一樣。而且這條規則本身也不穩：seed 0 比最好的不看位置策略快 18.3%，seed 1 慢 6.8%。
- **模擬器在新設定上也準**：GPU 44 格裡 43 格誤差 ≤10%（中位 1.0%，最大 13.7%）。

**還沒排除的例外**（只有模擬，NOT_MEASURED on GPU）：重算比現在便宜（f×0.25～0.5）時。模擬掃描裡 8 格通過，其中 7 格需要這個條件。它大約對應 MHA 模型（每 token 的 KV 是現在的 4 倍），也就是 Cake 原文用的 LongAlpaca-7B。

---

## 1. 做了什麼

| 步驟 | 內容 | run_id |
|:--|:--|:--|
| 1. 模擬器驗證 | 用第一階段 B 的 GPU 實測（chat、free）逐格比 | `20261009-064908-m7-sim-validate` |
| 2. 模擬掃描 | F1 CPU 頻寬、F2 hold／free、F4 chat／doc、F6 NFS／本地、F7 重算 ×0.25–2、seed；單一因子 → 兩兩 → 全部 | `20261009-065112-m7-sim-sweep2` |
| 3. 換 seed | 掃描裡有通過的設定，各換 5 個 workload seed | `20261009-0653xx-m7-sim-probe-*`（5 個） |
| 4. GPU 確認 | G1：doc＋hold＋CPU 3.69＋NFS，seed 0、1；G2：chat＋hold＋CPU 35.4＋NFS，seed 0。容量 25%、50%，各 3 次 | `20261009-070326-m7-b2-g1-doc-hold-cpu3.69-s0`、`20261009-090235-m7-b2-g1-doc-hold-cpu3.69-s1`、`20261009-102010-m7-b2-g2-chat-hold-s0` |

- 三個 GPU run 都是 exit 0；GpuWatcher 判定 CLEAN；`kv_bad`（還原後的 KV 逐位元比對）0 列不符。
- **為什麼 GPU 只挑這 3 組**：模擬掃描通過的 8 格裡，只有 1 格是重算成本 ×1（GPU 能直接做），其他 7 格都要 f×0.25 或 ×0.5，GPU 上無法直接做（§3）。所以 G1 seed 0 選這一格，seed 1 用來檢查它換 seed 還在不在。G2 用來驗證 hold 模型：模擬說 chat＋hold 時寫穿會大贏。

## 2. 模擬器準不準

| 範圍 | 格數 | 誤差 ≤10% | 誤差中位 | 誤差最大 |
|:--|:--|:--|:--|:--|
| 第一階段 B（chat、free） | 114 | 114 | 0.4% | 5.0% |
| 08 GPU 確認（doc／chat、hold） | 44 | 43 | 1.0% | 13.7% |

- 唯一超過 10% 的是 doc、seed 0、50%、S5L：GPU 1.127 s，模擬 0.973 s，低估 13.7%。同一格的 S4B 低估 8.8%。
- 44 格的誤差**全部是負的**，也就是模擬一律略為樂觀，GPU 實測略慢。
- hold 造成的等待（`t_gate`）模擬和 GPU 也對得上（`b2_summary.csv` 的 `gpu_t_gate_mean`、`sim_t_gate_mean`）。但這只說明 harness 的 hold **規則**有被模擬器正確重現，不代表真實 vLLM 會這樣擋住請求（§5）。

## 3. 模擬掃描〔模擬，只用來挑設定〕

| 項目 | 結果 |
|:--|:--|
| 單一因子 | 每一個單獨改，寫入時決定都贏不到 5% |
| 組合（390 格） | 同時贏「最好的不看位置策略」與 S4B 各 ≥5% 的有 **8 格**，**全部是 doc**；其中 7 格的重算成本是 ×0.25 或 ×0.5 |
| 換 5 個 seed | 30 個設定裡，**沒有一個 5 個 seed 都通過**。最接近的是 CPU 11.6＋hold＋doc＋NFS＋f×0.5＋25%：4／5 個 seed 通過，剩下那個 seed 是平手 |
| hold 的效果 | 寫入時決定比 S4B 快，中位 8.3%、最多 76%；但寫穿（S1、S2b）也做得到 |
| b 規則的效果 | 重算便宜時（f×0.25）最多快 66.6%；但 S4B（滿了才套用）一樣好 |

通過的 8 格（`sim_sweep_gain.csv`，gain 單位是 %）：

| CPU GiB/s | 搬移 | 負載 | SSD | 重算 × | 容量 | 最好的不看位置 | 最好的寫入時 | vs 不看位置 | vs S4B |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| 3.69 | hold | doc | NFS | 0.25 | 12.5% | S2b | S5c | +5.9 | +23.0 |
| **3.69** | **hold** | **doc** | **NFS** | **1** | **50%** | **S2b** | **S5L** | **+28.9** | **+6.0** |
| 11.6 | free | doc | 本地 | 0.25 | 12.5% | S4+ | S5L | +20.0 | +15.4 |
| 11.6 | free | doc | 本地 | 0.25 | 25% | S4+P | S5P | +8.0 | +8.0 |
| 11.6 | hold | doc | NFS | 0.5 | 25% | S4+ | S5c | +7.9 | +33.7 |
| 11.6 | hold | doc | NFS | 0.5 | 50% | S2b | S5L | +17.0 | +9.4 |
| 35.4 | free | doc | 本地 | 0.25 | 25% | S4+ | S5L | +15.0 | +19.3 |
| 35.4 | hold | doc | NFS | 0.25 | 25% | S4+ | S5c | +13.2 | +41.7 |

粗體那格是唯一能直接上 GPU 的。

## 4. GPU 確認

回來請求（第 2 輪起）的 TTFT 中位數，單位秒，3 次合併。gain＝(對手 − 自己)／對手，正的代表自己較快。

| 負載 | seed | 容量 | 最好的不看位置 | S4B | 最好的寫入時 | 寫入時 vs 不看位置 | 寫入時 vs S4B | S4B vs 不看位置 |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| chat | 0 | 25% | S1 1.555 | 3.151 | S5c 3.016 | −94.0% | +4.3% | −102.7% |
| chat | 0 | 50% | S1 1.332 | 2.733 | S5c 1.360 | −2.1% | +50.2% | −105.2% |
| doc | 0 | 25% | S2b 2.061 | 2.851 | S5c 2.063 | −0.1% | +27.6% | −38.3% |
| doc | 0 | 50% | S2b 1.389 | 1.134 | S5L 1.127 | **+18.9%** | **+0.7%** | +18.3% |
| doc | 1 | 25% | S2b 2.079 | 2.851 | S5c 2.070 | +0.4% | +27.4% | −37.2% |
| doc | 1 | 50% | S1 0.891 | 0.952 | S5L 0.952 | −6.8% | −0.0% | −6.8% |

逐策略的數字在 `b2_summary.csv`，逐比較的判定在 `b2_verdict.csv`。

怎麼讀：
- **寫入時決定那欄沒有一格 ≥5%，只有 doc／seed 0／50% 的 +18.9% 例外。但那格對 S4B 只快 0.7%**：贏的是「b 規則」，不是「寫入時」。
- **對 S4B 贏很多的格（+27～50%）**，對最好的不看位置策略都沒贏。S4B 在 hold 下吃虧，是因為它等滿了才搬，搬移卡在關鍵路徑上；寫入時決定和寫穿都提早寫了，所以不吃這個虧。
- **chat＋hold**：寫穿 S1 比 S5 快一倍以上（25%：1.555 vs 3.450 s），和模擬預測一致。多輪對話每次新寫的都是尾巴，S5 沒東西可決定，又要在滿了時搬。
- **換 seed**：seed 0 → 1，最好的不看位置策略從 S2b 換成 S1，「b 規則」從 +18.3% 變成 −6.8%。

## 5. 限制

1. **只有一個模型**（Llama-3.1-8B，GQA）。重算相對載入越便宜（MHA、較小的模型、較快的 GPU），模擬說例外越多。MHA 的 LongAlpaca-7B 已下載，**GPU 實測 NOT_MEASURED**。
2. **hold 是 harness 的模型，不是 vLLM 的實測行為**。真實的 vLLM OffloadingConnector 搬到 SSD 時，會不會讓下一個請求等，**NOT_MEASURED**。如果不會等，hold 那組結果就沒有現實意義，只剩 free（第一階段）的結論。
3. **GPU 只跑了 2 個 seed**。這組是從模擬裡挑出來最好的，有選擇偏誤；seed 1 正好說明了這一點。
4. **SSD 層仍是限速器模擬**（NFS 參數），沿用第一階段的 share 干擾模型。
5. **多重比較**：模擬 390 格＋30 個設定 × 5 seed；GPU 3 個設定 × 2 個容量，16 組比較。上表全部列出，沒有挑。

## 6. 要找老師決定的

1. 第一階段（free）和這次消融（hold、doc）都指向同一個結論：**在 GQA 模型上，寫入時決定沒有用**。第二階段要換問題，還是補 MHA 的實驗，看看例外是否真的存在？
2. 如果要往「何時寫到慢的層」走：hold 下真正有用的是「提早寫」（寫穿），這和位置無關。要先量真實 vLLM 會不會因為搬移擋住請求，這個方向才有意義。

## 附錄：指令

```bash
# 模擬
python code/m7_sim.py validate            # run 20261009-064908-m7-sim-validate
python code/m7_sim.py sweep               # run 20261009-065112-m7-sim-sweep2
python code/m7_ablate_analyze.py          # → sim_sweep_gain.csv
# GPU（nohup bash code/m7_run_ablate.sh；log：/mlsteam/data/tiara/runs/m7_ablate_chain_0703.log）
M7_IO_MODEL=share M7_CPU_GIBPS=3.69 python code/m7_write_policy.py b --b2 --workload doc --release hold --ssd-dev nfs \
  --wl-seed {0,1} --strategies S1 S2b S4+ S4L S4B S5 S5L S5c --cpu-fracs 0.25 0.5 --reps 3 --verify
M7_IO_MODEL=share python code/m7_write_policy.py b --b2 --workload chat --release hold --ssd-dev nfs \
  --wl-seed 0 --strategies S1 S2b S4+ S4B S5 S5c --cpu-fracs 0.25 0.5 --reps 3 --verify
python code/m7_b2_analyze.py              # → b2_summary.csv、b2_verdict.csv
```
