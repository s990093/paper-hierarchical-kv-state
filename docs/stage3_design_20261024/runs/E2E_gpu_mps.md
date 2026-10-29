# E2E-C 第二段：MPS 取捨在 GPU 上驗證（附錄登記）

> **性質**：判準登記（事先寫死）。登記時間見文末。登記之後這一節以上不改；和登記不同的做法寫成文末的「偏離 N」。
> **來源**：[E2E_combo.md](E2E_combo.md) §9 規定第二段要另寫附錄登記、在任何 GPU 數字之前。第一段判 E2E-NOGO，照 §9 原本不做；**使用者 2026-10-29 決定還是要做 GPU 驗證**（「MPS 在論文裡怎麼寫？……gpu」）。所以這一份是「NOGO 之後、使用者要求的驗證」，不改第一段的判定。
> **要回答的**：E2E-C 的模擬把 B0 量到的「背景 × 共用方式」的倍數乘到 Cake 的還原上（第一段 §3.3、偏離 1），並用 §3.4 算背景吞吐。這條「MPS 等級 → TTFT 改善 vs 背景吞吐」的取捨曲線，在真的 GPU 上重播時對不對得上？
> **範圍**：只有 H200、只有 Llama-3.1-8B、只有 conversation（縮小版，和 ICPE P0-2／B6 相同）；CPU 層用 T1 的常數（`P3_TIER=h200cpu`）。

---

## 1. 設定

- **工作負載**：`x1_gpu.load_P(8)`（Mooncake conversation、R1、依 session 抽 1/8；1,535 個請求、467 個回來的請求），CPU 128 GiB → `cap_bytes(128, 8)`；慢層＝SSD 級；還原＝Cake；慢層存 INT4；**HINT-PF**（＝第一段的 K2／K3）。和 T1G 的 P02 HINT-PF 128 run 同一份工作負載與程式（`x1_gpu.GpuExec`、`replay`）。
- **背景**：第一段的兩個 {cprefill:2048:65536, mixed:512:32:8192}（`bgengine.Background`）。
- **共用方式**：stream（背景在同一個 process 的另一個 CUDA stream，同 B0）；mps、mps87、mps75、mps50（背景在另一個 process，`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`，同 B0 的 `start_bgproc`；MPS daemon 只在 mps 那一段開）。
- **另加一個參考**：idle（沒有背景）。
- **重複**：每格 1 次；另外 stream 與 mps（兩個背景）各多 1 次（rep 1），只用來報告雜訊。共 2×5＋1＋4＝**15 個 GPU run**，每個跑完整的 1,535 個請求。
- **背景吞吐**：每個 run 開始前背景單獨跑 6×1 s（同 B0，得到 solo）；重播期間背景一直跑，量重播那段牆鐘時間裡的步數 → ρ_gpu＝共跑速度 ÷ solo。cap_gpu(g, m)＝solo(g, m)／solo(g, mps)（stream、mps＝1）。

## 2. 模擬（同一份工作負載）

- 用第一段的 `e2e.BgView`（path＝e2e，§3.3）、`e2e.factors`（B0 的資料、同一套 2σ 可用規則）、`XG.make_engine(P, "HINT-PF", cap, ...)`，跑同一個縮小版 P。
- **S_orig**：原始到達時間。**S_real**（只報告）：用 GPU run 的實際開始時間（同 P0-2 的做法）。
- 模擬的背景吞吐：整段重播一個時間窗，B_sim＝cap × (1 − φ·(1 − ρ̄))（§3.4 的公式，不切開發／測試）。GPU 的 B_gpu 用同一個公式，φ 用 GPU run 自己的 [t_start, t_start＋TTFT]（虛擬時間），ρ 用 ρ_gpu、cap 用 cap_gpu。
- 模擬那一格的倍數不可用（2σ 規則）→ 那一格 NOT_MEASURED（只報 GPU 的數字）。

## 3. 判準

- 指標：回來請求（kind＝1）的平均 TTFT（不含排隊，同 B6）。
- **(a) 比值**：每格（g, m）GPU ÷ S_orig 在 [0.9, 1.1]。
- **(b) 取捨的形狀**：每個 g、每個 MPS 等級 m：gain(m)＝1 − TTFT(m)／TTFT(stream)。GPU 與 S_orig 的 gain 同號，而且相差 ≤ 10 個百分點。
- **(c) 背景吞吐**：|ρ_gpu − ρ̄_sim| ≤ 0.05，而且 |B_gpu − B_sim| ≤ 0.03。
- **判定**（只算可量的格；NOT_MEASURED 照實列出）：
  - **MPS-GPU-VALID**：(b) 全部成立，而且 (a)、(c) 各在 ≥ 80% 的格成立。
  - **MPS-GPU-PARTIAL**：(b) 全部成立，但 (a) 或 (c) 不到 80%。→ 論文只能寫「曲線的形狀在 GPU 上成立，絕對值〔模擬〕」。
  - **MPS-GPU-INVALID**：(b) 有任何一格不成立。→ 論文的 MPS 曲線只能寫〔模擬〕，並附 GPU 的反例。
  - 可量的 (g, m) 少於 6 格 → **NOT_DETERMINED**。
- **只報告**：S_real 的比值；rep 0／rep 1 的差（雜訊）；idle 對 stream 的差；P95；`queue_s`。
- 判定不改第一段 E2E-NOGO（那是 B ≥ 0.95 的判準，這裡驗的是模擬本身）。

## 4. 程式與執行

- `code/s3/e2e_gpu.py`：`gpu`（一個 run）、`sim`、`report`、`--self-test`（模擬的 BgView 在 idle 時和 `x1_gpu.replay(model_view=True)` 的 HINT-PF 逐請求相同）。
- runner `code/r_h200_e2e_gpu.sh`：先冒煙（同一個 job、`--max-req 40`、cprefill × stream 與 × mps 各一次，輸出在 run 目錄、不進 results、不看數字），再正式跑；stream 那一段不開 MPS daemon，之後 `mps_on` 再跑 mps 那一段；順序：idle → stream 那一段（兩個背景 × rep 0、1，固定 seed 打亂）→ 開 MPS → mps 那一段（4 個等級 × 兩個背景的 rep 0，加 mps × 兩個背景的 rep 1，固定 seed 打亂）。
- 量時間：照 T2（`docs/h200/icpe/T2_parallel_calibration.md`）的結論決定能不能和我們的別的 job 同時跑；T2 判之前 → 獨佔（等節點上沒有我們自己的別的 job；每個 run 查重疊，同 T1G）。
- 輸出：`results/s3_h200/e2e_gpu/`（`gpu_requests.csv`、`gpu_runs.csv`、`sim_requests.csv`、`report.json`）。

## 5. 已經看過的數字（登記前）

- 第一段的全部結果（E2E-NOGO、32 格、MPS 曲線、FreeInference 的診斷）、T1 版。
- B0 的全部結果；T1G 的 P02（HINT-PF 128 的 GPU 平均 TTFT 0.237 s 等）。
- **沒有看過**：任何「縮小版 × 背景」的模擬或 GPU 數字。

---

## 登記

- 登記時間（`date -u`）：2026-10-29T03:03:43Z
- 登記時 repo HEAD：`15eef10`（本檔隨下一個 commit 進 git）

## 偏離（登記之後）
