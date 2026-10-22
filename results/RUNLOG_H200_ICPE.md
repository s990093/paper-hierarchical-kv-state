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
