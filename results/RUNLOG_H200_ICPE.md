# RUNLOG — 平台 C（H200，hgx-cn01）ICPE 2027 補量

> 格式照 CLAUDE.md §4.2。只有 ICPE 補量（`docs/h200/icpe/`）寫這份；第二階段寫 `RUNLOG_H200.md`、第三階段 3b 的移植寫 `RUNLOG_H200_P3.md`。
> 原始 log：`$TIARA_RUNS/<run_id>/`（＝`/work/ChihChungHsu/hungweilai/tiara-runs/`）。沒量到就寫 `NOT_MEASURED`。

---

## ICPE-0 — 前置（head node，CPU）

**狀態**: PASS
**執行時間**: 2026-10-22T02:43Z → （登記 commit 的時間見下）
**做了什麼**:
- 合併 `origin/main`（`09950ab`：MI300X 第三階段 m16_* 的程式與結果）進 `h200`：在暫時 worktree（detached 在 `ea87db9`）`merge --no-ff`，沒有衝突 → `cf31400`，再 `--ff-only` 進共用 checkout。合併時節點上沒有我們這個 repo 的 GPU job。
- 新增 `code/h200_gitpush.sh`：使用者說「push 只用 gitpush.sh、不要 git stash」；原本的 `code/m10_queue_mi300x/gitpush.sh` 寫死 MI300X 的路徑（`/mlsteam/workspace/...`）而且推 `main`，在這台跑不了，也違反「AMD 結束前只推 `origin/h200`」。H200 版照同一個原理（fetch → merge → push，不 stash、不 rebase），推 `origin/h200`。
- 事先登記：`docs/h200/icpe/P0-1_recompute_speed.md`、`P0-2_x1_gpu_validation.md`（這一節的 commit；寫的時候沒有任何 P0 的資料）。
**同節點的狀況**（2026-10-22T02:4xZ，`squeue`）：我們帳號在 hgx-cn01 上有 3 個 GPU job 在跑（DL_Lab4 的 `H_hr480`、`G_long300`、`I_xl192`，各 16 CPU；後面還排了 `G_swa`、`H_swa`、`F3_mim`），lab 上限 4 張卡／64 CPU。P0 的量時間 job 要等它們（見「要問的」）。
**失敗與異常**: 無。

### ICPE-0b — 磁碟與 GPU 排程（使用者 2026-10-22 決定）
**磁碟**（使用者問「目前空間還夠嗎」）：lab 的 `/work` 群組配額 1,533G／1,536G（剩約 3 GB，`quota -s -g ChihChungHsu`，2026-10-22T03:3xZ）；我們帳號 148G（hf-cache 99G、DL_Lab4 34G、conda 8.8G、tiara-runs 1.4G）。
- 使用者選「刪兩個沒用到的模型（建議）」＋另外跟老師要配額當緩衝。刪除（`huggingface_hub.scan_cache_dir().delete_revisions(...).execute()`，只刪沒有被別的模型引用的 blob）：
  - `Qwen/Qwen3-30B-A3B-Instruct-2507` revision `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`（56.89 GiB；第一階段第 1 輪的 MoE，已結束）
  - `Yukang/LongAlpaca-7B` revision `14c462dc860ba1818194ef0b5a31387ba57f0e4d`（12.55 GiB；同上）
  - 之後：群組 1,464G／1,536G、我們帳號 79,451M（2026-10-22T03:52Z）。Llama-3.1-8B（14.97 GiB）、Qwen2.5-7B-1M（14.2 GiB）、noDCA 變體（連到 Qwen 的 blob）都在。要重下載就用上面兩個 revision。
**GPU**：使用者自己的 DL_Lab4 有 3 個 job 在 hgx-cn01 跑（21566 `G_long300` 到 10/23 16:24、21567 `H_hr480` 到 20:24、21568 `I_xl192` 到 22:22，+08:00 的時限），排隊的 3 個（21570 `G_swa`、21571 `H_swa`、21618 `F3_mim`）。
- 使用者選「把排隊的 DL_Lab4 先 hold（建議）」：`scontrol hold 21570 21571 21618`（2026-10-22T03:5xZ）。在跑的 3 個不動；P0 量時間的 job 等它們結束、節點上沒有我們別的 job 才開跑。**P0 量時間做完要 `scontrol release 21570 21571 21618`**。
