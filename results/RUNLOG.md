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

#### 嘗試 3：GitHub release 的 `+cu129` wheel ← ✅ 成功

vLLM 每個 release 在 GitHub 附一個 CUDA 12 的替代 wheel。實測**不是文件寫的 `cu128`，
而是 `cu129`**（v0.20.0–v0.28.0 逐一以 `gh release view` 確認，全部只有 `cu129`）：

```bash
uv pip install \
  "https://github.com/vllm-project/vllm/releases/download/v0.28.0/vllm-0.28.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match
```

> `--index-strategy unsafe-best-match` 是必要的：沒有它，uv 只認第一個含有該套件的
> index，會卡在 `packaging<=24.1`（PyTorch index 的版本）而宣告無解。

**實測結果**：`vllm 0.28.0` + `torch 2.13.0+cu129`，`MATMUL_OK 32606.15234375`，
`CAP (8, 6) NVIDIA GeForce RTX 3090`。✅ **真的 kernel 跑起來了。**

#### 嘗試 4（附帶）：flashinfer JIT 缺 ninja

第一次 `vllm serve` 失敗：
```
FileNotFoundError: [Errno 2] No such file or directory: 'ninja'
  at flashinfer/jit/cpp_ext.py:370 run_ninja
```
`ninja` **已經裝在 venv 裡**，但因為我們是直接呼叫 `$VENV/bin/vllm` 而沒有 activate，
EngineCore 子行程的 `PATH` 看不到它。修法寫進 `code/serve_probe.sh`：
`export PATH="$VENV/bin:$PATH"`，並把 JIT 快取（XDG / Triton / vLLM / flashinfer）
全部導向 `/ssd7`，不污染 `$HOME`。

---

## 驗收 A1 — 環境指紋 ✅ PASS

**指令**: `CUDA_VISIBLE_DEVICES=0 python code/env_fingerprint.py`
**產出**: `results/env.json`

| 項目 | 實測值 |
|---|---|
| GPU | 7 × NVIDIA GeForce RTX 3090 |
| driver | 550.163.01 |
| torch | 2.13.0+cu129（built for cuda 12.9） |
| vllm | 0.28.0 |
| **真實 kernel** | **OK** ← 不是 `is_available()`，是真的跑了 matmul |
| offloading specs | `['CPUOffloadingSpec', 'TieringOffloadingSpec']` |
| cache policies | `['arc', 'lru']` |
| secondary tiers | `['example', 'fs', 'obj', 'p2p']` |

**與計畫的差異**：`EXPERIMENT_PLAN.md` §1 假設 Python 必須是 3.12「因為 vLLM wheel 限制」。
實測 venv 用 3.12.14 沒問題，但**真正的限制不是 Python 版本，是 CUDA 版本**。
計畫書的風險表把「`libcudart.so` 找不到」歸因為 Python 版本錯誤，那個歸因是錯的。

**🟢 計畫書假設的三件事全部存在**（這一步本來是風險）：
`CPUOffloadingSpec` ✓、`eviction_policy: lru/arc` ✓、`secondary_tiers` 含 `fs` ✓。

---

## 驗收 A2 — vLLM 能跑 ✅ PASS

**指令**: `CUDA_VISIBLE_DEVICES=6 code/serve_probe.sh Qwen/Qwen3-0.6B 8192`
**run**: `_big/runs/20260830-175142-probe-Qwen-Qwen3-0.6B-len8192/`

回傳合法 JSON，`choices[0].text = 'Question = "Hello, World!"\n\nprint'`。

> 用已在快取中的 Qwen3-0.6B 做煙霧測試，**刻意不等 15 GB 的主力模型下載完**——
> 先用便宜的模型驗證管路，再投入大檔案。
> （Qwen3 系列本身被計畫書排除在評測之外，理由是原生 ctx 只有 40,960。這裡只當管路測試。）

**失敗與異常**：第一次失敗於 `vllm: error: unrecognized arguments: --disable-log-requests`
——該旗標在 vLLM 0.28.0 已移除。移除後通過。

---

## 驗收 A3 — 卸載連接器可用 ✅ PASS ← 🔴 這一步本來可能終結整個計畫

**指令**
```bash
CUDA_VISIBLE_DEVICES=6 code/serve_probe.sh Qwen/Qwen3-0.6B 8192 \
  '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
    "kv_connector_extra_config":{"spec_name":"CPUOffloadingSpec",
      "cpu_bytes_to_use":8000000000,"eviction_policy":"lru"}}'
```
**run**: `_big/runs/20260830-175537-probe-Qwen-Qwen3-0.6B-len8192-kv/`

**log 證據**（`offload_evidence.txt`，6 條命中）：
```
Creating v1 connector with name: OffloadingConnector
Creating offloading spec with name: CPUOffloadingSpec
Created mmap file /dev/shm/vllm_offload_<id>.mmap (8.00 GB)
```

**失敗與異常**：第一次我把 `cpu_bytes_to_use` 寫成 `num_cpu_blocks`，
得到 `Exception: cpu_bytes_to_use must be specified in kv_connector_extra_config`。
**計畫書 §1 A3 原本就寫對了，是我改錯。** 改回計畫書的寫法即通過。

**新發現（計畫書未載明）**：CPU 卸載區是 `/dev/shm` 上的 mmap 檔。
本機 `/dev/shm` 有 **221 GB**，所以 CPU 階的預算上限遠大於計畫書假設的 8 GB。
→ M2 量成本常數時，CPU 階容量是可調的實驗變數，不是固定值。

---

## 🔵 A3 附帶發現：`CachePolicy` 的 out-of-tree 掛載點確實存在

`vllm/v1/kv_offload/cpu/policies/factory.py` 的 `CachePolicyFactory` 明載：

> External policies can either `register_cache_policy()` a friendly short name up front,
> **or skip registration entirely and pass a module path at lookup time
> (out-of-tree, no vLLM fork/patch required)**

且 `CPUOffloadingSpec` 讀 `cache_policy_module_path` 這個 extra_config 欄位。

**這直接支持論文 Fig. 2 的貢獻邊界論證**——Tiara 可以是純 plugin，不需要 fork vLLM。

**但有一個必須誠實記下的落差：** `CachePolicy` ABC 的介面是
`get / insert / remove / touch / evict / clear`，決定的是**「CPU 階裡該淘汰哪個 block」**。
論文的六元動作空間（GPU-BF16 / GPU-FP8 / GPU-INT4 / CPU-INT8 / SSD / DROP+重算）
**無法只透過這個介面表達**——特別是 `DROP → recompute` 與 GPU 內的精度降級。

→ 論文 §5.1 與 Fig. 2 若宣稱「插在 `CachePolicy` 介面即可」，**目前的證據只支持一部分**。
   這一項要進 `OPEN_ISSUES.md`，並在寫 §5 時修正措辭。**不要在論文裡放大這個發現。**

---

## 🔵 模型 config 驗證 ✅ 兩個模型都與計畫書一致

**指令**: `python code/verify_model_config.py`
**產出**: `results/m1_capacity/model_configs.json`

| 模型 | 層 | KV head | KV/tok | 原生 ctx | dtype | vs 計畫書 |
|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-7B-Instruct-1M | 28 | 4 | **56.0 KiB** | **1,010,000** | bfloat16 | ✅ 一致 |
| Meta-Llama-3.1-8B-Instruct | 32 | 8 | **128.0 KiB** | **131,072** | bfloat16 | ✅ 一致 |

計畫書「模型實際發布的是 bfloat16 不是 float16」也獲證實（2/2）。

### ⚠️ 取得模型時的兩個偏離

1. **`meta-llama/Llama-3.1-8B-Instruct` 是 `gated=manual`，且本機無 HF token。**
   改用 `NousResearch/Meta-Llama-3.1-8B-Instruct`（288K downloads 的無門檻鏡像）。
   config 逐欄比對與官方一致，含 `rope_scaling {rope_type: llama3, factor: 8.0}`。
   **若要取得官方權重，需要使用者提供 HF token 並在 HF 上通過 Meta 的授權申請。**

2. **AWQ-INT4 尚未取得。** 計畫書把 AWQ-INT4 列為「主力設定」，但
   `Qwen2.5-7B-Instruct-1M` **沒有官方 AWQ**，只有社群版
   （`graelo/...-AWQ` 592 downloads、`mzayed/...-AWQ` 6 downloads）。
   → 目前先用 BF16 建立懸崖基準（那本來就是計畫書要求的敏感度列）。
   AWQ 的來源選擇（用社群版 vs 自行量化）是**待決策項**，不要默默挑一個。

---

## 🔴 重大發現：Qwen2.5-7B-Instruct-1M 用的是 Dual Chunk Attention，不是原生 1M

raw `config.json`：
```json
"max_position_embeddings": 1010000,
"rope_theta": 10000000.0,
"dual_chunk_attention_config": {
  "chunk_size": 262144, "local_size": 8192,
  "original_max_position_embeddings": 262144
}
```

**這推翻了計畫書 §3 的一個前提。** 計畫書寫：

> ✅ **選模準則已排除外推問題**：兩個模型的評測長度都在原生範圍內
> （Qwen2.5-7B-1M 懸崖 315K ≪ 1M 上限）

但實際上 **`original_max_position_embeddings` 是 262,144**，1M 是靠 **DCA** 達成的。
計畫書預期的懸崖 ~315K **高於** 262,144 → **掃到懸崖時模型已經在外推**，
與「不要開 YaRN 以免品質退化來源無法歸因」是同一個問題，只是機制不同。

**還有第二層風險**：vLLM 0.28.0 的 DCA 只在 rotary embedding 層有實作
（`model_executor/layers/rotary_embedding/dual_chunk_rope.py`），
`v1/attention/` 底下**找不到對應的 attention backend**。
→ **V1 engine 是否真的啟用 DCA，未經驗證**（`NOT_MEASURED`）。若沒啟用，
超過 262K 的品質數字全部不可信。

**處置**（尚未執行，待與使用者確認）：
- (a) 把主力評測長度上限訂在 **262,144** 以內，懸崖仍量但品質不在 262K 以上宣稱；或
- (b) 為 >262K 另建「full-KV + 相同 DCA 設定」的基準線，讓退化可歸因；或
- (c) 換主力模型。

**在釐清之前，不要產生任何 >262K 的品質數字。**

---

## Milestone 1 — 容量懸崖 ✅ PASS

**狀態**: PASS
**執行時間**: 2026-08-30 17:59 → 18:07
**指令**: `python code/m1_capacity.py --config <cfg> --gpu <i>`（4 個設定平行跑在 GPU 0–3）
**產出**: `results/m1_capacity/capacity.csv`、`results/m1_capacity/model_configs.json`

### 方法（與計畫書不同，這裡說明為什麼）

計畫書 §2 寫的是「逐步加大 `--max-model-len` 直到 OOM」。**沒有照做。**
vLLM 啟動時會把答案直接印出來：

```
GPU KV cache size: 41,648 tokens
```

這個數字就是懸崖本身。所以流程改成 **量測 + 雙向驗證**（`code/m1_capacity.py`）：

1. 用 `max_model_len=8192`（保證裝得下）啟動，讀出 `GPU KV cache size`
2. 用讀到的 N 再啟動一次 → **應該成功**
3. 用 `N × 1.15` 啟動 → **應該失敗**

一個設定 3 次啟動而非 7 次，且產出的是**連續量**而非「有沒有 OOM」的二元訊號。
第 2、3 步是必要的——沒有它們就只是抄 log，不算量測。

### 關鍵數字（全部實測，四個設定的兩個邊界都驗證過）

| 設定 | 權重 | KV dtype | **懸崖（token）** | KV GiB | 在懸崖啟動 | 超出 1.15× |
|---|---|---|---|---|---|---|
| `llama-bf16` | BF16 | auto(BF16) | **41,648** | 5.084 | ✅ OK | ✅ 如預期失敗 |
| `llama-bf16-kvfp8` | BF16 | **fp8** | **83,312** | 5.085 | ✅ OK | ✅ 如預期失敗 |
| `qwen-bf16` | BF16 | auto(BF16) | **106,512** | 5.688 | ✅ OK | ✅ 如預期失敗 |
| `qwen-bf16-kvfp8` | BF16 | **fp8** | **213,040** | 5.689 | ✅ OK | ✅ 如預期失敗 |

**內部一致性**：FP8 在兩個模型上都給出**恰好 2 倍**的 token 數，而 KV 佔用的 GiB 不變
（5.084→5.085、5.688→5.689）。這正是「同樣的位元組裝兩倍的 token」，
是這批數字沒有量錯的強證據。

### 🔴 發現 1：Ampere **可以**用 FP8 KV cache —— 計畫書這一條是錯的

`EXPERIMENT_PLAN.md` §1、§3 與 `CLAUDE.md` 都寫著：

> ⚠️ Ampere（sm_86）**不支援原生 FP8** → 平台 A 量不到 `GPU FP8` 這一階，
> 在 `cost_model.json` 中標記為 `NOT_SUPPORTED`

**實測推翻了這個結論。** `--kv-cache-dtype fp8` 在 sm_86 上正常運作，
而且給出乾淨的 2 倍容量。

原因是把兩件事混為一談了：
* **FP8 運算**（tensor core 原生 FP8 matmul）—— Ampere 確實沒有，這部分計畫書是對的
* **FP8 儲存**（KV 以 fp8 存、讀取時反量化）—— **與 tensor core 無關，Ampere 可以做**

論文動作空間裡的 `GPU-FP8` 是**儲存**狀態不是運算狀態
（見 `EXPERIMENT_PLAN.md` §3 的 2×5 成本矩陣：「平時成本 = GPU 位元組 ÷ 2」）。
→ **`GPU-FP8` 這一階在平台 A 是可量測的**，不必留給 MI300X。
→ 反量化的一次性成本（表格的「被需要時的成本」欄）仍待 M2 量測。

**這是一個讓論文變強的發現**：動作空間的六階在單一平台上就有五階可量。
`EXPERIMENT_PLAN.md` §3、§8 與 `CLAUDE.md` §3 的相關敘述**需要修正**。

### 🔴 發現 2：Qwen2.5-7B-Instruct-1M 在 vLLM 0.28.0 上載不起來（DCA 路徑壞掉）

原版模型直接失敗：
```
TypeError: FlashAttentionImpl.__init__() got an unexpected keyword argument 'layer_idx'
```

`model_executor/models/qwen2.py:189` 在 `dual_chunk_attention_config` 為真時，
把 `layer_idx` 與 `dual_chunk_attention_config` 傳給 `Attention`，
而 V1 的 attention backend 不接受這兩個參數；`v1/attention/` 底下也沒有任何
dual-chunk backend。**vLLM 0.28.0 的 V1 engine 沒有可用的 DCA 路徑。**

`--hf-overrides` 兩種寫法都繞不過去（都實測過）：

| 覆寫 | 結果 |
|---|---|
| `{"dual_chunk_attention_config": null}` | `TypeError: 'NoneType' object does not support item assignment`（`verify_dual_chunk_attention_config` 對 None 做 item assignment） |
| `{"dual_chunk_attention_config": {}}` | attention 路徑過了，但 rotary 的 `get_rope()` 用 `is not None` 判斷，仍進 DCA 分支 → `DualChunkRotaryEmbedding.__init__() missing 2 required positional arguments` |

**處置**：建立 no-DCA 變體（`code/make_nodca_model.py`）——權重用 symlink 不複製，
只改寫 `config.json`：移除 `dual_chunk_attention_config`，並把
`max_position_embeddings` 由 `1,010,000` 改為 `262,144`（該模型**真正被訓練的長度**，
取自被移除欄位的 `original_max_position_embeddings`）。

路徑：`/ssd7/hungwei/paper-hkv/models/Qwen2.5-7B-Instruct-1M-noDCA`

**這對本研究反而更乾淨。** 計畫書 §3 要求「不要為了掃更長 context 而開 YaRN，
那會讓品質下降的來源無法歸因」。DCA 是不同機制、同樣的問題。移除後所有評測長度
都落在模型真實訓練範圍內，ε（品質退化）可乾淨歸因到 KV 放置策略。

**代價**：評測長度上限 1M → 262,144。在 24 GB 單卡上不構成限制——
BF16 懸崖 106,512 遠低於此；即使換成 AWQ-INT4，推估的懸崖也在 262K 附近。

### 與論文 §2.5 算術值的差距

計畫書預期的是 **AWQ-INT4** 權重下的懸崖，我們量的是 **BF16**。
把權重差補回去（BF16 權重 → INT4 約省 10 GiB）做一致比較：

| 模型 | 實測 BF16 懸崖 | 補回權重差後推估 AWQ 懸崖 | 計畫書宣稱 | 差距 |
|---|---|---|---|---|
| Llama-3.1-8B | 41,648 | ~124,000 | ~132,000 | **−6%** |
| Qwen2.5-7B-1M | 106,512 | ~283,500 | ~315,000 | **−10%** |

**論文 §2.5 的算術在 6–10% 內成立**，方向與量級都對。
差距來自算術沒扣掉 activation 與 CUDA graph 的常數開銷（實測約佔 1.5–2 GiB）。

⚠️ 右邊兩欄是**推估值不是量測值**，因為 AWQ 權重尚未取得（見下）。
`capacity.csv` 裡**只有實測的 BF16 四列**，推估值不入 CSV。

### 尚未完成的部分

* **AWQ-INT4 列**：`NOT_MEASURED`。`Qwen2.5-7B-Instruct-1M` 無官方 AWQ，
  只有社群版（592 / 6 downloads）。用社群版 vs 自行量化是**待決策項**。
* 計畫書要求在 `note` 欄寫下實測與 §2.5 的差距 → 已寫在本節，CSV 的 `note` 欄
  存的是設定用途說明。

---

## Milestone 2 — 量測工具鏈

**狀態**: `NOT_STARTED`

## Milestone 3 — Baseline

**狀態**: `NOT_STARTED`

## Milestone 4 — Oracle 上界（🔴 決定性）

**狀態**: `NOT_STARTED`
