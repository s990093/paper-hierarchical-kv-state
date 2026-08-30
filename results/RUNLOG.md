# RUNLOG — 平台 A（RTX 3090）

> 逐步流水帳。格式見 `EXPERIMENT_PLAN.md` §7。
> **規則**：每一個數字都要能追溯到一條指令與一個輸出檔。沒量到就寫 `NOT_MEASURED`。
> 原始 log 在 `/ssd7/hungwei/paper-hkv/logs/` 與 `/ssd7/hungwei/paper-hkv/runs/<run_id>/`。

---

## Milestone 0 — 工作環境建立

**狀態**: 🟡 IN PROGRESS
**執行時間**: 2026-08-30 17:25 → 進行中
**執行者**: Claude Code (Opus 5)

### 0.1 機器盤點

**指令**
```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv
df -h ; python3 -V ; sudo -n true
```

**關鍵數字（實測）**

| 項目 | 值 |
|---|---|
| GPU | **7 × RTX 3090 24576 MiB**（index 0–6，sm_86） |
| Driver | **550.163.01** |
| Driver 支援的 CUDA | **12.4** ← 這一項後來決定了整個 vLLM 安裝路線 |
| 系統 Python | 3.13.12 |
| `/` 可用 | 446 G（3.6 T，88% 已用，**共用分割區**） |
| `/ssd7` 可用 | 523 G（7.3 T，93% 已用） |
| `/dev/shm` | 221 G |
| sudo | **無**（`sudo: a password is required`） |

**與計畫的差異**：`EXPERIMENT_PLAN.md` 假設單張 3090；實際有 7 張。
→ 用途是**平行掃不同設定**（每 job 綁一張卡），**不是** tensor parallel。
容量懸崖必須在單卡 24 GB 上量，否則 §2.5 的算術失去意義。已寫入 `CLAUDE.md` §3。

### 0.2 目錄配置

大檔案全部落在 `/ssd7/hungwei/paper-hkv/`，repo 內以 `_big` symlink 指過去（已 gitignore）。
配置見 `CLAUDE.md` §2。

### 0.3 Skill 工具鏈

網路調查 6 個候選 → 採用 2 個 MIT 專案，vendored 22 個 skill 進 `.claude/skills/`（772 KB）。
選型理由、拒絕理由、已知落差（無 Codex MCP）全記在 **`docs/SKILLS.md`**。

### 0.4 LaTeX 工具鏈

**問題**：本機**沒有任何 TeX**，且**沒有 sudo**。

**做法**：TinyTeX 安裝到 `/ssd7/hungwei/paper-hkv/texlive/`（user-level，無需 sudo），
再 `tlmgr install` 本文需要的套件。腳本與 log：
`_big/logs/install_tex.sh`、`install_tex_pkgs.sh`、`install_tex_pkgs.log`。

**失敗與異常**：第一版腳本用 `find -type f -name tlmgr` 找 binary，
但 TinyTeX 的 `tlmgr` 是 symlink → 誤報 `FATAL: tlmgr not found`。
**TeX 本身已裝成功**，是腳本的探測邏輯錯。第二版直接用絕對路徑，通過。

**字型移植**（`main.tex` 原本寫死 macOS 字型）：

| 用途 | 原（macOS） | 改為（Linux） | 來源 |
|---|---|---|---|
| 西文襯線 | `Times New Roman` | `TeX Gyre Termes` | TeX Live（Times 的度量相容克隆） |
| 西文無襯線 | `Helvetica` | `TeX Gyre Heros` | TeX Live |
| 西文等寬 | `Menlo` | `TeX Gyre Cursor` | TeX Live |
| CJK 襯線 | `Songti TC` | `Noto Serif CJK TC` | 系統已有 |
| CJK 無襯線 | `Heiti TC` | `Noto Sans CJK TC` | 系統已有 |
| CJK 等寬 | `Heiti TC` | `Noto Sans Mono CJK TC` | 系統已有 |

TeX Gyre 字型不在 fontconfig 的搜尋路徑中（fontspec 找不到 family name），
故新增 `~/.config/fontconfig/conf.d/09-texlive-paperhkv.conf` 把 TinyTeX 的
`fonts/opentype` 與 `fonts/truetype` 加進去，再 `fc-cache -f`。

原始 `main.tex` 已備份至 `_big/runs/main.tex.bak-macfonts`。

**驗收結果**
```
$ make distclean && make
font warnings : 0
overfull boxes: 0
undefined     : 0
Output written on main.pdf (16 pages)
```
✅ 通過。**唯一差異：16 頁，README 記錄的 macOS 建置是 15 頁。**
原因是 TeX Gyre Termes 與 Times New Roman 的度量雖相容但非位元相同，
加上 Noto Serif CJK 的字身高與 Songti 不同，累積出一頁。
→ **投稿前若有頁數上限，必須以投稿機器的字型重新確認頁數。** 已記入 `OPEN_ISSUES.md`。

### 0.5 vLLM 安裝 —— 🔴 三次嘗試，前兩次失敗

這一段完整保留，因為它是 `EXPERIMENT_PLAN.md` §8「風險表」沒預料到的失敗模式，
且**第一次失敗會靜默通過天真的檢查**。

#### 嘗試 1：`uv pip install vllm`（照計畫書寫的）

安裝結果：`vllm 0.28.0` + `torch 2.13.0+cu130`。

天真的檢查**會通過**：
```
>>> torch.cuda.is_available()
True
>>> torch.cuda.device_count()
7
```

**但真正跑一個 kernel 就爆**：
```
$ python -c "import torch; x=torch.randn(1024,1024,device='cuda'); (x@x).sum()"
RuntimeError: The NVIDIA driver on your system is too old (found version 12040).
```

**根因**：vLLM 0.28.0 的 PyPI 預設 wheel 是對 **CUDA 13.0** 編譯的；CUDA 13 是主版本跳躍，
需要 driver ≥ 580。本機 driver 550.163.01 只到 CUDA 12.4。
GeForce 卡**不支援** `cuda-compat` forward compatibility（那只給 datacenter 卡），無法繞過。

> ⚠️ **`torch.cuda.is_available()` 回傳 `True` 不代表 CUDA 能用。**
> 它只做惰性初始化前的探測。**驗收條件必須是「跑成一個真的 kernel」**，
> 這條已補進 `EXPERIMENT_PLAN.md` 的 A1 驗收與 `CLAUDE.md`。

#### 嘗試 2：`uv pip install vllm --torch-backend=cu128`

torch 換成 cu128 了（`nvidia-*-cu12` 系列正確下載），但：
```
ImportError: libcudart.so.13: cannot open shared object file
  at vllm/platforms/cuda.py:22 -> import vllm._C_stable_libtorch
```
**根因**：`--torch-backend` 只換 PyTorch，換不掉 **vLLM 自己的編譯擴充**。
PyPI 上的 vLLM wheel 只有 CUDA 13 一種。

#### 嘗試 3：GitHub release 的 `+cu129` wheel ← 目前進行中

vLLM 每個 release 在 GitHub 附一個 CUDA 12 的替代 wheel。實測**不是文件寫的 `cu128`，
而是 `cu129`**（v0.20.0–v0.28.0 逐一以 `gh release view` 確認，全部只有 `cu129`）：

```bash
uv pip install \
  "https://github.com/vllm-project/vllm/releases/download/v0.28.0/vllm-0.28.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
  --extra-index-url https://download.pytorch.org/whl/cu129
```

CUDA 12.9 runtime 可在 driver 550 上執行（CUDA 12.x minor-version compatibility，
最低 driver 525.60.13）。

**狀態**：`NOT_YET_VERIFIED` — 待 `_big/logs/install_vllm_cu129.log` 出現
`MATMUL_OK` 與 `OFFLOADING_CONNECTOR_OK` 才算通過。

---

## 驗收 A1 — 環境指紋

**狀態**: ⏳ 待 vLLM 安裝完成後產生 `results/env.json`

## 驗收 A2 — vLLM 能跑

**狀態**: `NOT_MEASURED`

## 驗收 A3 — 卸載連接器可用

**狀態**: `NOT_MEASURED`
⚠️ 這一步若失敗，依 `EXPERIMENT_PLAN.md` §1「整個計畫要重新設計」，必須立刻回報。

---

## Milestone 1 — 容量懸崖

**狀態**: `NOT_STARTED`

## Milestone 2 — 量測工具鏈

**狀態**: `NOT_STARTED`

## Milestone 3 — Baseline

**狀態**: `NOT_STARTED`

## Milestone 4 — Oracle 上界（🔴 決定性）

**狀態**: `NOT_STARTED`
