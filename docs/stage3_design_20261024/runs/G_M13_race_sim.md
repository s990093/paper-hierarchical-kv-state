# G-M13：搬運和重算賽跑（慢層的尾巴保險）——只用 CPU 的重播

> **性質**：判準登記（事先寫死）。登記時間見文末。來源：[mechanisms/04 M13](../mechanisms/04_recompute_scheduling.md)、[mechanisms/06 §2](../mechanisms/06_eval_gates.md)（G-M13）。
> **只用 CPU、只用既有資料**：I3g（MI300X、vLLM 0.28，`results/m16_stage3_mi300x/i3g_agentic/`）。**平台是 MI300X**（I3g 只在 MI300X 跑過），不和 H200 的數字混。程式：`code/s3/gm13.py`。

## 1. 資料
- Llama：設定 d（檔案層＝NFS）run `20261022-192954-m16-i3g-d`、設定 c（檔案層＝本地 /var/tmp）run `20261022-191335-m16-i3g-c`。
- 「回來那一輪」＝`m16_i3g_report.py` 的同一個篩選：turn ≥ 1、不是暖機、成功、有 ttft。
- 重算時間 t_cold(L)：MI300X 的 vLLM 冷 prefill（`results/m7_write_policy_mi300x/c6_vllm.csv` 的 `cold_compute`，8K／16K／32K 各自的中位數）在 prompt 長度 L 上分段線性內插（8K 以下照 0→8K 的直線）。
- Qwen：MI300X 上沒有 Qwen 的 vLLM 冷 prefill → **NOT_MEASURED**（不判定）。

## 2. 賽跑規則（逐請求重播，請求之間不互相影響）
- H0（主）：讀慢層的同時就開始重算；TTFT＝min(TTFT_d 實測, t_cold(prompt_len))。
- H1（保守，另報）：等了 t_cold 才開始重算；TTFT＝min(TTFT_d, 2·t_cold)。
- 沒有模擬的兩件事（照實寫）：額外的重算會搶 GPU（讓結果偏樂觀）；取消掉的讀取會讓 NFS 排隊變短（讓結果偏悲觀）。

## 3. 判準（事先寫死）
- **GM13-PASS**：H0 下回來那一輪的 TTFT P95 ≤ 1.5 × 設定 c 的回來那一輪 TTFT P95（P95 用 `m16_i3g_report.pct` 同一個內插）。否則 **GM13-STOP**。
- 另報：P50、平均、重算贏的比例、白做的重算（讀先到時重算已經花掉的 GPU 秒數）。

## 登記
- 登記時間（`date -u`）：**2026-10-25T13:00:31Z**；commit 訊息開頭 `docs(s3-gm13): prereg`。登記時沒有算過任何賽跑的數字（I3g 的 b／c／d P95 是已發表的舊數字）。
