# STATUS_DATA（icpe-data，acvlab）

> 只由 `icpe-data` 寫。格式照 NEXT_PLAN_20261104 §3b：日期時間／做完什麼（commit、run_id）／卡住什麼（完整錯誤）／需要 icpe-sim 做什麼／需要使用者決定什麼。最新的一節在最上面。

## 2026-11-04 23:3x（Asia/Taipei）— venv ready

**做完什麼**
- worktree：`/ssd6/hungwei/icpe-next/data`，分支 `icpe-next-data`（從 `origin/main` 77c25f6 開），已 push。來源 clone＝`/home/hungwei/llm/POC/paper-hierarchical-kv-state`（和 icpe-sim 的 `sim/` 同一個 repo）。
- **venv ready**：`/ssd6/hungwei/icpe-next/venv`（run `20261104-232611-icpe-data-venv`，目錄 `/ssd6/hungwei/icpe-next/runs/20261104-232611-icpe-data-venv/`，exit 0）
  - Python **3.12.15**（`3.12.15 (main, Oct 21 2026, 17:47:06) [Clang 23.1.3 ]`）
  - numpy **2.3.5**、pandas **3.0.5**、pyarrow **25.0.1**；相依：python-dateutil 4.0.0、tzdata 2026.6
  - 用法：`/ssd6/hungwei/icpe-next/venv/bin/python -I ...`（或 `source /ssd6/hungwei/icpe-next/venv/bin/activate`）
- 建法（為什麼和平常不一樣）：
  - 這台的 uv 0.12.3 下載清單最高只到 3.12.13 → 用 `uvx --from uv==0.13.7 uv ...` 臨時跑較新的 uv；`~/.local/bin/uv` 沒動。
  - 受控 Python 裝在**獨立目錄** `/ssd6/hungwei/icpe-next/uv-python/`，沒有裝進共用的 `/ssd8/hungwei/paper-hkv/uv-python/`：那裡的 `cpython-3.12-linux-x86_64-gnu` 是 minor 版 symlink，`venv/tiara` 的 `home` 就指著它；裝 3.12.15 進去會讓既有 venv 被透明升級。`venv/tiara`、`venv/vllm028` 沒動。
  - `uv python install --no-bin`：沒有改 `~/.local/bin/python3.12`（它指向別的專案）。

**還沒驗證**
- H200 那次 run（`20261028-071007-icpe-t1-x1-sweep-V1`）的 context 不在 git 裡，我看不到它的 python-dateutil／tzdata 版本：NOT_VERIFIED。核心四項（Python、numpy、pandas、pyarrow）照 NEXT_PLAN §3b 的版本。若 G0 對不上，這是要先查的一項。

**卡住什麼**：無。
**需要 icpe-sim 做什麼**：可以開始用 venv 做 G1。
**需要使用者決定什麼**：無（目前）。
