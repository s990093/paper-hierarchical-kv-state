# CLAUDE.md — Tiara 專案工作規則

> 這個檔案是給 **Claude Code（以及任何接手的 agent）** 讀的操作手冊。
> 論文主張看 `main.tex`；實驗怎麼跑看 `EXPERIMENT_PLAN.md`；**怎麼在這台機器上動手看這裡**。

---

## 0. 這是什麼

**Tiara**（**Ti**ered, quality-**A**ware, **R**ecompute-**A**ware KV manager）——
長上下文 LLM 推論的 KV cache 階層式放置策略。

一句話主張：**重算與傳輸的成本比 κ 跨硬體變動達 32 倍**，
因此「單一 KV 放置策略適用所有硬體」的隱含前提不成立；
放置決策應形式化為**品質約束下、動作成本異質的線上決策**，
動作空間 = {GPU-BF16, GPU-FP8, GPU-INT4, CPU, SSD, DROP+重算}。
**CPU 與 SSD 兩階是無損的位元組搬移**（2026-09-01 改：原本寫 CPU-INT8，但實作走 vLLM `OffloadingConnector` 搬原始位元組，且 §6.5 已用 60/60 逐字元比對驗證無損，所有成本常數也是從那條無損路徑量的——定義與量測必須一致）。

**目前狀態：論文初稿完成，實驗尚未執行。** §6 是實驗計畫不是結果，表 8 全是灰色佔位符。
整個專案的下一個決定點是 **Milestone 4 的 Oracle go/no-go**。

---

## 1. 🔴 不可協商的規則

這五條直接來自 `EXPERIMENT_PLAN.md` §0，適用於**每一個** agent、**每一次** session：

1. **不准編造任何數字。** 沒跑出來就寫 `NOT_MEASURED`。不要估、不要推、不要「依經驗約為」、
   不要從論文的算術值反填成實測值。**論文裡的數字是假設，不是結果。**
2. **不准跳過失敗。** 指令失敗 → 記錄完整錯誤訊息 → 停下來回報。不要換個方式硬幹到有輸出為止。
3. **每一個數字都要能追溯到一條指令與一個輸出檔。** 見 §4 的記錄協定。
4. **不准跳過 Milestone 4 的 go/no-go 判定。** 那是整個研究的停損點，`< 5%` headroom = 停止。
   **NO-GO 不是失敗，是省下數個月的有價值負面結果。不要為了讓專案繼續而美化數字。**
5. **實驗 agent 不准改 `main.tex`。** 產出是 `results/` 與 `RUNLOG.md`。
   論文的修改是另一個工作階段（用 `paper-writing` skill），且必須先有 `results/` 的證據。

6. **外部資料集的單位，一律用資料自身交叉驗證，不准看欄位名推斷。**
   2026-08-31 踩到：Mooncake 的 `hash_ids` 是 **512-token** 的 block，
   我當成 16-token，於是工作集少算 32 倍、**每個 block 的絕對位置少算 32 倍**
   （重算成本是位置的線性函數，所以 DROP 被算得太便宜），
   所有 trace 驅動的結果全部作廢重跑。
   能抓到只是因為同一個量（請求長度中位數）碰巧被兩條路徑算過而對不上。
   **匯入任何 trace 時，載入函式裡就要寫一條「用 A 欄位驗算 B 欄位」的斷言。**

7. **「查不到」不等於「沒有」。** 偵測類函式（爭用、容量、佇列）在查詢失敗時
   必須讓上層知道，不可以回傳空值。`gpu_guard._smi()` 原本查詢失敗回傳 `[]`，
   會被 `host_contention()` 讀成「整機沒有外來負載」= QUIET，
   等於把污染的量測標成乾淨的。現已改成丟 `SmiUnavailable`。

---

## 2. 🗂 目錄配置（大檔案一律放 /ssd7）

**根分割區 `/` 只剩 446G 且是共用的。所有大檔案放 `/ssd7/hungwei/paper-hkv/`。**

```
/home/hungwei/llm/POC/paper-hierarchical-kv-state/   ← git repo（推 GitHub）
├── CLAUDE.md               ← 本檔
├── main.tex refs.bib Makefile main.pdf              論文
├── EXPERIMENT_PLAN.md      實驗計畫（給執行 agent 的唯一指令書）
├── OPEN_ISSUES.md          已知但刻意暫緩的問題（A/B/C 三級）
├── VENUES.md               投稿場所與截止日
├── docs/SKILLS.md          skill 選型紀錄（為什麼裝這些）
├── code/                   實驗 harness（小、進 git）
├── results/                ★ 只放「小的、可版控的」結果
│   ├── RUNLOG.md           逐步流水帳
│   ├── env.json            環境指紋
│   ├── m1_capacity/  m2_harness/  m3_baseline/  m4_oracle/
│   └── *.csv *.json        摘要級數據
├── .claude/skills/         22 個 vendored skills（見 §5）
└── _big -> /ssd7/hungwei/paper-hkv   ← symlink，已 gitignore

/ssd7/hungwei/paper-hkv/    ← 大檔案，不進 git
├── texlive/.TinyTeX/       TeX Live（無 sudo 安裝）
├── venv/vllm/              vLLM venv（Python 3.12）
├── uv-cache/  uv-python/   uv 的快取與 managed python
├── hf-cache -> /ssd7/hungwei/hf-cache    模型權重（已有 160G）
├── models/                 自行量化的 AWQ 權重
├── runs/                   ★ 原始 log、vLLM server stdout、每次 run 的完整輸出
├── profiles/               nsys / torch profiler 產物
├── datasets/               評測資料
│   ├── gsm8k/              GSM8K train/test jsonl
│   ├── traces/             Mooncake 等 trace
│   ├── longbench/          LongBench v1 的 data/ + config/ + 上游 metrics.py、eval.py、pred.py
│   └── ruler_ref/          RULER 上游合成腳本（**對照用**，實際產生器是 code/ruler_tasks.py）
├── pylibs/                 側裝的純 Python 套件（rouge / fuzzywuzzy / wonderwords）
│                           ↳ 用 `--target` 裝在這裡而不是 venv，避免污染 vLLM 的相依
├── logs/                   安裝與環境建置腳本 + log
└── vendor-skills/          上游 skill repo 的 clone（供更新比對）
```

**判準：** 進 git 的東西必須是「人看得懂、diff 有意義、< 1 MB」。
CSV 摘要、JSON 指紋、Markdown 記錄 → git。
模型權重、server log、profiler trace、venv → `/ssd7`。

**兩者要用「指標」串起來**：`results/` 裡的每一列都要有欄位指回 `/ssd7/.../runs/<run_id>/`。

---

## 3. 🖥 這台機器

| 項目 | 值 | 備註 |
|---|---|---|
| GPU | **7 × RTX 3090 24 GB**（sm_86, Ampere） | `nvidia-smi` 確認 index 0–6 |
| Driver / CUDA | 550.163.01 / 12.4 | |
| 系統 Python | 3.13.12 | **不要用**，vLLM 用 venv 的 3.12 |
| RAM | tmpfs 顯示 221G `/dev/shm` | CPU offload 預算來源 |
| sudo | **沒有** | 所有安裝都必須 user-level |
| TeX | `/ssd7/hungwei/paper-hkv/texlive/.TinyTeX`，已 symlink 進 `~/.local/bin` | |
| CJK 字型 | `Noto Serif CJK TC` / `Noto Sans CJK TC`（系統已有） | **不是** macOS 的 Songti/Heiti |

### ⚠️ Ampere 的限制（2026-08-30 實測修正）

1. ~~sm_86 不支援原生 FP8~~ → **這條原本是錯的，已由 M1 實測推翻。**
   `--kv-cache-dtype fp8` **在 sm_86 上可用**，且給出乾淨的 2 倍 KV 容量
   （llama 41,648 → 83,312；qwen 106,512 → 213,040，GiB 佔用不變）。
   原本的錯誤在於混淆了兩件事：
   * **FP8 運算**（tensor core 原生 FP8 matmul）—— Ampere 確實沒有
   * **FP8 儲存**（KV 以 fp8 存、讀取時反量化）—— **與 tensor core 無關，可以做**

   論文動作空間的 `GPU-FP8` 是**儲存**狀態，所以**平台 A 量得到**。
   反量化的一次性成本仍待 M2 量。見 `results/RUNLOG.md` M1 發現 1。
2. **沒有記憶體/計算分軌的能耗計數器。** 論文 §6.7 的能耗分析**不適用於平台 A**，不要嘗試。
   （這一條仍然成立。）
3. **vLLM 0.28.0 的 V1 engine 沒有可用的 DCA 路徑**，所以 Qwen2.5-7B-Instruct-1M
   要用 no-DCA 變體 `/ssd7/hungwei/paper-hkv/models/Qwen2.5-7B-Instruct-1M-noDCA`
   （`code/make_nodca_model.py`）。上限因此是 **262,144** 不是 1M。

### 🔀 平台 B（AMD MI300X）的移植面

**策略：在 3090 上打磨方法，AMD 上只做量測。** 所以 harness 必須先在這裡就
把廠商相依收斂到最小面積。目前的移植面只有三處：

| 位置 | 相依 | 移植做法 |
|---|---|---|
| `gpu_guard.py` 的 `vendor()` / `_amd_compute_apps()` / `gpu_util()` | `nvidia-smi` ↔ `amd-smi`/`rocm-smi` | **已寫好 AMD 後端**，但**尚未在真機驗證**。第一次在 MI300X 上跑之前，必須先對照 `amd-smi process` 的輸出確認 pid 與 gpu index 對得上 |
| `env_fingerprint.py` | `torch.cuda` + `nvidia-smi` | `torch.cuda` 在 ROCm 上就是 HIP，可直接用；用 `torch.version.hip` 區分平台 |
| `CUDA_VISIBLE_DEVICES` | 各 mN 腳本 | ROCm 亦接受此變數（HIP 別名），但保險起見在 AMD 上同時設 `HIP_VISIBLE_DEVICES` |

模擬器（`m4_*.py`）與品質量測（`m5_quality.py`）**完全不碰 GPU 廠商 API**，
只吃 `results/` 的 CSV 與 HTTP API，所以零移植成本。

平台 B 上真正要重量的只有 **M1 容量** 與 **M2 成本模型**——
κ 的跨硬體主張就是由這兩者構成。其餘（Oracle、預算掃描、語意消融、
長上下文外插）都是拿新的成本常數重跑同一批腳本。

### 多卡的正確用法

有 7 張卡，但**論文的壓力軸是「單請求 × context 遞增」，不是並行度**。
多卡的用途是：**平行掃不同的 (model, 量化, context) 設定**，每個 job 綁一張卡
（`CUDA_VISIBLE_DEVICES=<i>`），**不是** tensor parallel。
容量懸崖必須在**單卡 24 GB** 上量，TP 會讓 §2.5 的算術失去意義。

`code/m3_baseline.py --all` 會自動把每個 baseline 排到一張空閒的卡上。

### 🔴 這是共用機器：每次量測都要防插隊

`/ssd7` 底下有二十幾個使用者的目錄，**隨時可能有人佔用 GPU**。被污染的量測會：
* 時間數字被 SM 爭用拉高，而且**偏多少無法事後修正**
* `peak_vram` 讀成兩個 process 的總和 → 容量結論直接錯
* **靜默發生**——跑出來的數字看起來完全正常

所以 **`code/gpu_guard.py` 是必用的，不是選用的**：

```bash
python code/gpu_guard.py --idle-gpus     # 目前乾淨的卡
python code/gpu_guard.py --check 3       # 這張卡乾不乾淨（乾淨回 0）
```

程式內用 `GpuWatcher` 包住整段量測：

```python
from gpu_guard import GpuWatcher
with GpuWatcher(gpu=g, out_path=...) as w:
    ...量測...
if w.contaminated:      # 開跑前不乾淨，或中途出現外來 PID
    ...結果作廢，重量...
```

**規則**：`contaminated == True` 的 run，結果**不得寫進 `results/`**，必須重量。
`m3_baseline.py` 已內建這個行為（會在 run 目錄留下 `CONTAMINATED` 檔）。

**這條規則的範圍是「時間」欄位。** 品質量測（`m5_quality.py`、`m5_understanding.py`）
量的是**分數**：同一批 prompt、`temperature=0`、固定 seed，外來 process 不改變輸出。
所以那兩支腳本在污染時**保留分數、逐列記錄爭用**
（`own_gpu_intruders`、`level`、`foreign_gpu_count`、`foreign_max_util`），
同樣在 run 目錄留下 `CONTAMINATED` 檔，但**該 run 的 `latency_ms` 欄作廢**，
不得用來下任何時間結論。丟掉一整輪三小時的品質量測換取一個與結論無關的乾淨度，
是拿嚴謹當儀式。

### 環境啟動

```bash
export UV_CACHE_DIR=/ssd7/hungwei/paper-hkv/uv-cache
export UV_PYTHON_INSTALL_DIR=/ssd7/hungwei/paper-hkv/uv-python
export HF_HOME=/ssd7/hungwei/paper-hkv/hf-cache/huggingface
export PATH=/ssd7/hungwei/paper-hkv/texlive/.TinyTeX/bin/x86_64-linux:$PATH
source /ssd7/hungwei/paper-hkv/venv/vllm/bin/activate
```
（`.claude/settings.json` 已把這些設成 session env，新開的 Bash 應該已經有。）

---

## 4. 📝 記錄協定（「所有東西都需要紀錄」）

### 4.1 每一次 run 的最小單位

任何會產生數字的指令，都要走這個殼：

```bash
RUN_ID=$(date +%Y%m%d-%H%M%S)-<短名>          # 例：20260830-183012-m1-qwen-awq-131072
RUN=/ssd7/hungwei/paper-hkv/runs/$RUN_ID
mkdir -p $RUN
# 1. 先記指令本身
printf '%s\n' "$CMD" > $RUN/cmd.sh
# 2. 記環境
{ date -Is; nvidia-smi --query-gpu=index,name,memory.used --format=csv; \
  git -C <repo> rev-parse HEAD; } > $RUN/context.txt
# 3. 跑，全部導進去
bash $RUN/cmd.sh > $RUN/stdout.log 2> $RUN/stderr.log; echo $? > $RUN/exit_code
```

### 4.2 `results/RUNLOG.md` 的格式

**每一個 Milestone 結束都要補一段**，格式照 `EXPERIMENT_PLAN.md` §7：

```markdown
## Milestone N — <名稱>
**狀態**: PASS / FAIL / BLOCKED
**執行時間**: <起> → <迄>
**run_id**: <對應 /ssd7/.../runs/ 的目錄名>
**指令**: <實際跑的指令，可複製貼上>
**產出檔**: <路徑>
**關鍵數字**: <只寫實際量到的>
**失敗與異常**: <完整錯誤訊息，沒有就寫「無」>
**與論文假設的差異**: <實測 vs main.tex 的預期>
```

### 4.3 CSV 的必要欄位

每個 `results/**/*.csv` 都必須有 `run_id` 與 `ts` 欄，讓任何一列都能反查原始 log。
沒有 `run_id` 的數字視同**不存在**。

### 4.4 Git commit

- 實驗產出的 commit 訊息開頭用 `results(mN):`，例：`results(m1): capacity cliff for qwen2.5-7b-1m awq`
- 論文修改用 `paper:`；工具/環境用 `infra:`；skill 更新用 `skills:`
- **每個 Milestone 結束 commit 一次並 push**，不要累積。

---

## 5. 🧰 已安裝的 Skills（`.claude/skills/`，22 個）

選型理由與來源見 `docs/SKILLS.md`。**兩個上游都是 MIT，已 vendored 進 repo**（不是 symlink，
所以 GitHub 上的 repo 自帶完整工具鏈，換一台機器 clone 就能用）。

### 論文寫作
| Skill | 來源 | 什麼時候用 |
|---|---|---|
| `paper-writing` | SNL-UCSB | **主力寫作流程**。五階段：Brainstorm → Architecture → Draft → Integrate → Compress。含 M1–M18 機械檢查與 S1–S31 語意檢查兩道 gate |
| `writing-systems-papers` | ARIS | systems venue 的段落級藍圖與頁數配置（OSDI/SOSP/EuroSys/NSDI） |
| `paper-compile` | ARIS | 編譯 + 修錯 + 驗證 PDF |
| `paper-figure` / `figure-spec` | ARIS | 從 `results/` 生圖 |
| `rebuttal` | ARIS | 審稿意見回覆（之後才用） |

### 誠實性 gate（★ 本專案的核心需求）
| Skill | 什麼時候用 |
|---|---|
| `experiment-audit` | **實驗跑完、寫 claim 之前**。查 fake ground truth、phantom results、scope 不足 |
| `result-to-claim` | 判定結果**支持**什麼 claim、**不支持**什麼、還缺什麼證據 |
| `paper-claim-audit` | **投稿前**。逐一核對論文裡每個數字 vs `results/` 的原始檔 |
| `citation-audit` | 核對 `refs.bib` 45 筆——作者、年份、venue、以及**引用語境是否被原文支持** |
| `kill-argument` | 投稿前的對抗式審查：先寫最強的拒稿理由，再逐點反駁 |

### 實驗執行
| Skill | 什麼時候用 |
|---|---|
| `experiment-bridge` | **讀 `EXPERIMENT_PLAN.md` → 實作 → 部署 → 收初步結果**。本專案的主入口 |
| `experiment-plan` | 需要把新想法變成 claim-driven roadmap 時 |
| `run-experiment` | 單一 job 的部署與執行 |
| `experiment-queue` | **多設定 sweep + OOM-aware retry** ← M1 容量懸崖二分搜尋正是這個形狀 |
| `monitor-experiment` | 長跑 job 的進度輪詢 |
| `analyze-results` | 統計、比較表 |
| `ablation-planner` | Oracle = GO 之後，設計消融矩陣 |

### 文獻
`arxiv` / `semantic-scholar` / `openalex` / `novelty-check`

### ⚠️ 已知落差：沒有 Codex MCP

ARIS 的多個 skill（`experiment-audit`、`result-to-claim`、`paper-claim-audit`、
`citation-audit`、`kill-argument`、`ablation-planner`、`novelty-check`）
在 `allowed-tools` 裡宣告 `mcp__codex__codex`，用途是**找一個沒有前文脈絡的外部模型當審查者**，
以避免確認偏誤。**這台機器沒有裝 Codex MCP。**

替代方案（依偏好順序）：
1. **開一個全新的 Claude subagent 當 zero-context reviewer** — 效果最接近原設計。
   ⚠️ **但 spawn subagent 需要使用者明確要求**，不要自作主張開。
2. 裝 ARIS 附的 `mcp-servers/claude-review` 或 `gemini-review`（見 `vendor-skills/aris/mcp-servers/`）
3. 降級成 self-review，**並在 RUNLOG 裡明確標註「此次 audit 為 self-review，非 cross-model」**

**不要假裝跑了 cross-model review。** 這正是這批 skill 要防的失敗模式。

---

## 6. 📄 編譯論文

```bash
export PATH=/ssd7/hungwei/paper-hkv/texlive/.TinyTeX/bin/x86_64-linux:$PATH
make          # xelatex → bibtex → xelatex ×2，並印出警告統計
make clean
make check    # 列出所有 TODO 與佔位符
```

**必須用 XeLaTeX**（內文中文，`xeCJK`）。
字型已從 macOS 的 `Songti TC`/`Heiti TC` 改為 Linux 的 `Noto Serif CJK TC`/`Noto Sans CJK TC`。
**改字型前先確認 `fc-list :lang=zh-tw` 有那個 family**，不然會靜默 fallback 成豆腐字。

目標：**0 字型警告、0 overfull box、0 未定義引用**。

---

## 7. 🎯 現在該做什麼（依序）

1. **環境驗收 A1–A3**（`EXPERIMENT_PLAN.md` §1）：`env.json`、vLLM 能起來、
   **`OffloadingConnector` 可用** ← A3 若失敗，整個計畫要重新設計，立刻回報
2. **Milestone 1**：容量懸崖實測（`capacity.csv`），比對論文 §2.5 的算術值
3. **Milestone 2**：成本模型 **2×5 矩陣**（不是 1×5 向量）+ `recompute_chain.csv`
4. **Milestone 3**：Tier 0 baselines（Full GPU / lru / arc / fs / LMCache）
5. **🔴 Milestone 4：Oracle** ← **決定性**。`> 15%` GO、`5–15%` 停下來問人、`< 5%` NO-GO 停止
6. Milestone 5+ 只在 GO 之後做

**時程壓力**：MLSys 2027 截止 2026-10-30（剩 61 天）。
**9 月底若還沒有 Oracle 結果，放棄 MLSys 這一輪**，改投 EuroMLSys 2027（~2 月，6 頁）。
見 `VENUES.md`。

---

## 8. ❌ 明確不要做的事

- ❌ 不要在平台 A 做**能耗**結論（消費卡無分軌計數器）
- ❌ 不要在平台 A 做**多租戶／機會成本**結論（24 GB 放不下多個長 session）
- ❌ 不要在 Oracle 出來之前**訓練任何模型**或調 policy 超參數
- ❌ 不要嘗試訓練 KVP / ForesightKV / LookaheadKV（皆未釋出權重，單卡不可行）——
  核心主張改由**消融表 (B) 區塊**（同一系統只換損失函數）檢驗
- ❌ 不要為了掃更長 context 而**開 YaRN**（品質退化來源無法歸因，ε 就失去意義）
- ❌ 不要把大檔案寫進 repo 或 `/tmp`（用 `/ssd7/hungwei/paper-hkv/`）
- ❌ 不要用 tensor parallel 量容量懸崖

---

## 9. 🔗 GitHub

Repo：<https://github.com/s990093/paper-hierarchical-kv-state>（public）
`gh` 已用帳號 `s990093` 登入，scope 含 `repo`。

推送前檢查：`git status` 不應出現任何 `/ssd7` 的實體檔（只有 `_big` symlink，且已 gitignore）。
