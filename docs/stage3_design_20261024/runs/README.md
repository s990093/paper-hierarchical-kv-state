# 第三階段的 run 文件（H200）

> 每個閘一個 md：**最上面是事先登記的判準與量法（commit＋`date -u` 之後不改）**，結果與偏離附在後面。
> 規則照 `docs/h200/icpe/README.md`：登記在看到任何正式數字之前 commit；冒煙（smoke）只驗程式、不進 `results/`、不看分數；
> 和登記不同的地方一律寫成「偏離 N」附在文件後面，說明為什麼、在看到哪些數字之前或之後。

| 閘 | 文件 | 程式 | 結果 | 狀態 |
|:--|:--|:--|:--|:--|
| A0（＝G-M01/02） | [A0_progressive_restore.md](A0_progressive_restore.md) | `code/s3/a0.py`、`code/s3/a0_gpu.py` | `results/s3_h200/a0/` | **GA0-WEAK**（10/27）：條件 2 全不過；最接近 GO 的 Q128 B24·1/8 差 0.05 pp |
| B0＋G-M08（量時間） | [B0_interference.md](B0_interference.md) | `code/s3/b0.py`、`code/s3/bgengine.py` | `results/s3_h200/b0/` | **FLIP 不成立、SHARE 成立、STATIC 不成立（只有 dq_vq）；G-M08 STOP**（10/26） |
| G-M04（品質，併在 A0 TF） | [G_M04_prefix_select.md](G_M04_prefix_select.md) | `code/s3/a0_gpu.py --gm04` | `results/s3_h200/a0/raw/gm04_*` | 已登記 |
| C0（CPU） | [C0_cost_model.md](C0_cost_model.md) | `code/s3/c0.py` | `results/s3_h200/c0/` | 已登記 |
| M05（品質，併在 A0 L128 生成） | [M05_duo_heads.md](M05_duo_heads.md) | `code/s3/a0.py`（`m05_configs`）、`a0_gpu.py`（`kind=duo`） | `results/s3_h200/a0/raw/a0_gen_L128_*` | 已登記 |
| G-M12（品質＋量時間） | [G_M12_sparse_recompute.md](G_M12_sparse_recompute.md) | `code/s3/gm12.py` | `results/s3_h200/gm12/` | **G-M12-STOP**（10/28）：L128 品質（F＝1.0）−14.6／−8.6 nats、qwen 速度 0.47×／0.93×（要 ≤0.3×）；Q128 品質沒跑（使用者收掉） |
| G-M11（量時間，併在 B0） | [G_M11_vq.md](G_M11_vq.md) | `code/s3/b0.py`（`dq_vq`） | `results/s3_h200/b0/` | **NOT_DETERMINED**（閒時 dq_int4 太晃；10/26 重量 12 次仍晃）；10/28 使用者接受、不再量 |
| G-M06＋M07（2 卡〔偏離 2〕、量時間） | [G_M06_multipath.md](G_M06_multipath.md) | `code/s3/gm06.py` | `results/s3_h200/gm06/` | **G-M06-PASS**（k＝2 1.986×，CI 下界 1.985；run 20261028-043420-s3-gm06） |
| W0（量時間） | [W0_write_cost.md](W0_write_cost.md) | `code/s3/w0.py` | `results/s3_h200/w0/` | **GW0-FAIL**（llama、qwen；10/26，job 22279） |
| A0-C（品質；A0 最接近 GO 的配置的確認性實驗） | [A0C_confirm.md](A0C_confirm.md) | `code/s3/a0.py`、`a0_gpu.py --a0c` | `results/s3_h200/a0/a0c_*` | **A0C-STOP**（10/28）：Q128 的 P 對 U* +1.28 [−0.27, +2.88]（< +2）、對 E0 −2.58；A0 的 WEAK 沒有被確認 |
| E2E-C（端到端組合：INT4＋Cake＋打字提示預取＋MPS；第一段模擬） | [E2E_combo.md](E2E_combo.md) | `code/s3/e2e.py` | `results/s3_h200/e2e/` | **E2E-NOGO**（10/28）：0/32 格過；m\* 全部選不出來——任何 MPS 等級、任何開發格的背景吞吐 B 都 < 0.95（最高 0.944）→ K3＝K2；NOT_MEASURED 8 格（Qwen × cprefill）；第二段不做。FreeInference 第 9–16 週有未驗證的異常（預取變慢）。**GPU 附錄**（使用者 10/29 要求，[E2E_gpu_mps.md](E2E_gpu_mps.md)）：**MPS-GPU-INVALID**（10/30）——正負號全對，但 GPU 的 gain 比模擬大 5–31 pp → MPS 曲線只能寫〔模擬〕＋GPU 反例 |
| G-fio（量時間） | [G_fio.md](G_fio.md) | `code/s3/gfio.py` | `results/s3_h200/gfio/` | **完成**：本地 NVMe 8.07 GiB/s；/work 在 compute node 是本地 ext4（NFS 列續用限速層）（10/26，job 22417） |
| D0（品質；MiniCPM4.1-8B 原生索引） | [D0_native_index.md](D0_native_index.md) | `code/s3/d0.py`、`code/s3/d0_shim/` | `results/s3_h200/d0/` | **不做**（10/28 使用者；上游 cutlass 編譯錯誤，A0 STOP 後不影響決策表）→ NOT_MEASURED |

runner：`code/r_h200_s3.sh`；RUNLOG：`results/RUNLOG_H200_S3.md`；GPU 排程規則見 memory `stage3-gpu-scheduling`。

---

## 共同規則：速度判準的「雜訊」（2026-10-25 補；prereg deviation）

操作定義寫在 [05 §2.1](../05_poc_plan_h200.md)。所有**量時間**的 run 的判準段都引用它，摘要如下：

- σ_noise＝baseline 配置在同一個 Slurm allocation、同一張卡、同一輪隨機順序裡重複 ≥6 次的相對標準差（σ÷中位數），每個量測點各自算、不跨點共用。
- δ\*＝max(5%, 2×σ_noise)。「有改進」：中位數差 > δ\* 且差的 95% CI（對 rep 重抽 10,000 次）下界 > 0。「沒有變慢」：變慢 ≤ δ\* 且變慢的 95% CI 上界 ≤ δ\*。點估計過、CI 沒過 → INCONCLUSIVE（要加重複次數）。P95 同規則，σ 用 P95 的 bootstrap 標準誤。
- CSV 每列有 `n_rep`、`sigma_rel`、`delta_star`（P95：`p95_sigma_rel`、`p95_delta_star`）；量不到寫 NOT_MEASURED。
- 2σ > 5% 的點＝量測太晃：先回報，不拿來判定。
- 品質判準（A0、G-M04）不變，不套用 σ 規則。

**為什麼是 deviation**：05 §2 原本只寫「δ\*＝max(5%, H200 上量到的雜訊)」，沒有定義雜訊怎麼量；B0 的登記（第 66 行）直接取 δ\*＝5%。

**受影響的 run（逐一）**

| run | 有沒有看過時間數字 | 處理 |
|:--|:--|:--|
| B0＋G-M08 | **沒有**（冒煙 22238 在量任何還原之前就失敗；冒煙 22242 進行中，輸出在 run 目錄、不進 results、不判定）。看過的只有 P0b 背景自檢的單獨速度（bgcheck：每步毫秒），判準不用它 | **改用新規則**（B0 檔的偏離 1） |
| W0 | 沒有（還沒跑） | **改用新規則**（W0 檔的偏離 1） |
| G-fio | 沒有（還沒跑） | 沒有判定，只換常數；加 6 次重複與三個欄位（G-fio 檔的偏離 1） |
| C0-v1 | **有**（10/25 已出 GC-FAIL） | **維持原判準**；另寫一行「事後以 2σ 重算的結果」只供參考（C0 檔） |
| C0-v2（之後同程式重跑） | 判準在 v1 已經看過 | 維持原判準（和 v1 相同）；同樣附 2σ 參考行 |
| G-M13 | **有**（10/25 已出 GM13-PASS；MI300X 的 I3g 重播） | **維持原判準**；另寫一行「事後以 2σ 重算的結果」只供參考（G-M13 檔） |
| A0、G-M04 | 品質 | 不變 |
| P0e | 沒有時間判定（能／不能） | 不適用 |
| 之後的量時間的閘（G-M06、G-M11、G-M12 速度、A2…） | 還沒登記 | 登記時直接照 05 §2.1 |

---

## 共同規則：NUMA 位置（2026-10-25 補；只記錄，不是 deviation）

H200 節點是 2× EPYC 9554。pinned 記憶體和卡不在同一個 NUMA 節點時，H2D 會變慢（`docs/H200_SLURM.md` 第 158 行；慢多少 NOT_MEASURED）。每個 job 拿到的卡可能不同，而原本的程式沒有記錄這件事。所以每個量時間的 run 現在都會記三件事（`code/s3/numa_probe.py`）：GPU 在哪個節點、job 的 cpuset 涵蓋哪些節點、主機緩衝的頁實際落在哪個節點（`move_pages(2)` 每組抽 64 頁，另記所在 VMA 在 `/proc/self/numa_maps` 的整段頁數）。另外也記每個節點的可用記憶體、`zone_reclaim_mode`、`numa_balancing`。

| run | 記在哪 |
|:--|:--|
| B0＋G-M08（`b0.py run`） | 每列多四欄 `gpu_numa`、`cpuset_numa`、`buf_numa`（「節點:抽樣頁數;…」）、`numa_local`（yes／no／NOT_AVAILABLE＋原因）。CSV 旁邊另存 `<名稱>_numa.json`，含 start、bench（e2e 的 CPU 層第一次建好時，在暖機那一次裡）、end 三個時點 |
| P0e（`b0.py env`） | `env.json` 的 `numa`（另配 64 MiB pinned 來查） |
| W0（`w0.py run`） | 每列同樣四欄（緩衝＝CPU 編碼讀的那份 KV，一般記憶體）；另存 `<名稱>_numa.json`（start、end） |
| G-fio（`gfio.py run`） | `_fs.json` 的 `numa`（job 的 cpuset 與看得到的卡；gfio 不用 pinned 記憶體，所以沒有緩衝欄） |
| G-M12 速度（`gm12.py speed`） | `gm12s_*_numa.json`（卡與 cpuset；計時區間只有 GPU 重算、不碰主機緩衝，所以沒有加欄） |
| G-M06＋M07（`gm06.py run`） | 每列 `src_numa`（那一列用的來源 pinned 實際在哪個節點，h2d_simul 逐卡列出）；`gm06_<run_id>_numa.json`（srcs、big、numa_local、numa_remote）。卡的節點本來就記在 `topo_<run_id>.json` |

- **判準一律不變**。`numa_local` 不進任何判定，B0、W0、G-fio 的登記也不改。
- `numa_local≠yes` 的 run 怎麼處理，**還沒有規則**：先照實回報，要不要重量由使用者決定。
- 查不到就照實寫 NOT_AVAILABLE，不當成「在同一邊」（CLAUDE.md 規則 7）。
- **cpuset 對了，不代表 pinned 就在同一邊**：本地節點沒有空間、而且 `zone_reclaim_mode=0` 時，kernel 會把頁配到另一個節點。lab-3090 上的實例見 `results/RUNLOG_H200_S3.md` 的 S3-NUMA 節。
