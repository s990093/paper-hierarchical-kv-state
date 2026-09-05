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

## Milestone 2 — 成本常數 2×N 矩陣 🟡 部分完成

**狀態**: capacity ✅ / recompute ✅ / retrieval ⏳ 初步值今晚、權威值排凌晨（見下）
**指令**: `python code/m2_cost_model.py --gpu 0 --stage <capacity|recompute|retrieval>`
**產出**: `results/m2_harness/capacity_by_dtype.csv`、`idle_cost_normalized.json`

### 🟢 發現 7：論文動作空間的**四個精度階全部可在單卡量到**

`EXPERIMENT_PLAN.md` 原本把 `GPU-FP8` 判為平台 A 量不到（M1 已推翻）。
這裡更進一步：vLLM 的 `--kv-cache-dtype` 除了 `fp8` 還提供
`int8_per_token_head` 與 `int4_per_token_head`。

**論文六階動作空間裡「住在 GPU 上」的四階，全部是同一個旗標的不同取值**，
換一個字串就換一階，不需要任何額外實作。

### 平時成本（每個狀態佔多少位元組）

⚠️ **必須除以 GiB 正規化**。原始 token 數 run-to-run 會跳（全距 13.5%），
因為 KV pool 的大小本身在 5.09 / 5.88 GiB 兩個值之間擺動（見發現 6）。
**正規化之後全距降到 0.00–0.04%**，抖動完全來自 pool 大小而非 dtype。

| dtype | tok/GiB | 全距 | **KiB/token** | 理想值 | 額外 | **相對 BF16** |
|---|---|---|---|---|---|---|
| `auto`(BF16) | 8,185.0 | 0.00% | **128.11** | 128 | +0.11 | 1.00× |
| `fp8` | 16,372.8 | 0.00% | **64.04** | 64 | +0.04 | **2.00×** |
| `int8_per_token_head` | 15,877.6 | 0.04% | **66.04** | 64 | **+2.04** | 1.94× |
| `int4_per_token_head` | 30,819.0 | 0.02% | **34.02** | 32 | **+2.02** | **3.77×** |

### 🔵 那多出來的 2 KiB/token 算得出來是什麼

Llama-3.1-8B：32 層 × 8 個 KV head，K 與 V 各一份
→ 每個 token 有 `2 × 32 × 8 = 512` 個 per-token-head 的 scale。
512 × 4 bytes（fp32）= **2,048 bytes = 2 KiB**。

實測額外開銷：int8 = 2,089 bytes（4.1 B/scale）、int4 = 2,068 bytes（4.0 B/scale）。
**與算術完全吻合。這不是量測誤差，是量化中繼資料。**

**對論文的影響**：動作空間的精度階梯**不是 2×/4×**，而是 **2.00× / 1.94× / 3.77×**。
* `fp8` 是純格式轉換，沒有額外中繼資料 → 恰好 2×
* `int8` / `int4` 走 per-token-head 量化，固定多帶 2 KiB/token
* → **把 FP8 與 INT8 當成同一階（都是 1 byte）會失準 3%**；
  INT4 的實際收益是 3.77× 不是 4×
* 式(1) 的成本模型若只寫 `bytes/elem`，會系統性高估低精度階的收益。
  中繼資料開銷是**與序列長度成正比的固定成本**，必須進成本模型。

### 🟢 發現 8：重算成本確實隨位置成長，**但是線性不是二次**

**指令**: `python code/m2_cost_model.py --gpu 0 --stage recompute --chunk 2048 --positions 0 4096 8192 16384 24576`
**產出**: `results/m2_harness/recompute_position.csv`（每個位置 3 次重複取中位數）

做法：先送長度 P 的前綴讓它進 prefix cache，再送「同樣的 P + 新的 2048 個 token」。
第二次的 TTFT 就是「前 P 個 token 命中、後 2048 個現算」的成本。

| 前序已快取 P | 實測 ms | 線性擬合 | 誤差 | vs P=0 | µs/token |
|---|---|---|---|---|---|
| 0 | 513.0 | 513.0 | 0.0% | 1.00× | 250.5 |
| 4,096 | 616.7 | 623.1 | 1.0% | 1.20× | 301.1 |
| 8,192 | 726.2 | 733.2 | 1.0% | 1.42× | 354.6 |
| 16,384 | 938.7 | 953.3 | 1.6% | 1.83× | 458.4 |
| 24,576 | 1,173.5 | 1,173.5 | 0.0% | **2.29×** | 573.0 |

**擬合**：`C_recompute(P) = 513.0 ms + 26.9 ms × (P / 1000)`
＝ 固定成本 513 ms，加上每 1000 個前序 token 多付 26.9 ms。

#### ✅ 計畫書對的地方

`EXPERIMENT_PLAN.md` §3：

> ⚠️ `C_recompute` **不是常數**，隨 block 的絕對位置成長。
> 要量成 `C_recompute(position)`，不是單一純量。

**完全正確，而且幅度不小**——同樣重算 2048 個 token，在位置 24,576 要 2.29 倍的時間。
把重算當成單一常數會在長 context 系統性低估它的成本。

#### 🔴 計畫書要修的地方：「二次項」

同一段寫的是「隨 block 的絕對位置成長（**attention 二次項**）」。

**成長的來源**確實是 attention 的二次項（整段 prefill 是 O(L²)），
**但對固定大小的 chunk 而言，觀察量是 O(chunk × P)——對 P 線性。**

實測的線性擬合誤差 **< 1.6%**，五個點都在線上，找不到二次成分。

→ 論文若寫「重算成本隨位置**二次**成長」，審稿人只要做這個實驗就會抓到。
   正確的說法是：**「對固定大小的 block，重算成本隨其絕對位置線性成長」**，
   斜率來自 attention 對前序 KV 的讀取量。
   `EXPERIMENT_PLAN.md` §3 與論文 §2.6 的措辭都要改。

#### 對成本模型的意義

這一項讓 `DROP` 這一階的成本變成**兩個參數**而非一個：

```
C_drop(P) = 513.0 ms + 0.0269 ms × P        （本平台、Llama-3.1-8B、chunk=2048）
             ↑ 固定：kernel 啟動 + chunk 自身的計算
                          ↑ 位置相關：讀前序 KV
```

`DROP` 的「平時成本 = 0」是它唯一的優勢；這條曲線量的是它要付的代價，
而**代價隨位置線性上升**正是論文「重算的價值取決於未來效用預測品質」的量化基礎——
越晚才被需要的 block，重算越貴。

### 🔴 發現 9：SSD 階先前量錯了，而且錯得很隱蔽

**先前的數字**：`ssd` warm = 550.9 ms，與 `cpu` 的 546.0 ms 只差 **1.2%**。
當時的解讀是「page cache 讓它沒碰到磁碟」。**那個解讀也是錯的。**

vLLM 的 metrics 直接給出真因：

```
kv_offload_tiering_chunk_queries:('0:primary',) = 2048   ← CPU 階
kv_offload_tiering_chunk_queries:('1:fs',)      =    2   ← 磁碟階
```

CPU 階開 24 GiB，而工作集只有 8 GiB（4 × 16,384 tok × 128 KiB）——
**東西整個塞在 CPU 階，從來沒 cascade 到磁碟。** 磁碟階只被查 2 次。
所以那個「550.9 ms」量到的是 CPU 階，不是磁碟階。

**修法**：量 SSD 時把 CPU 階縮到 1 GiB（<< 工作集），強迫 cascade。
修正後 `chunk_queries:('1:fs',) = 2048`，實際從磁碟讀了 3 GiB。✅

### 修正後的「被需要時的成本」（ctx=16,384，整機 HEAVY）

| 階 | SATA QLC (/ssd7) | NVMe Crucial P3 (/) |
|---|---|---|
| GPU 命中 | 134.4 ms | 163.3 ms |
| **CPU** | **736.9 ms** | **719.4 ms** |
| **SSD** | **5,803.3 ms（9.41× CPU）** | **6,600.5 ms（11.58× CPU）** |
| **DROP（重算）** | **5,467.5 ms（8.85×）** | **5,465.6 ms（9.53×）** |

### 🔴 發現 10：在這台機器上，**SSD 階被 DROP 支配**

| 動作 | 平時成本 | 被需要時 |
|---|---|---|
| SSD | 佔磁碟位元組 | **5,803 ms** |
| DROP | **0** | **5,467 ms** |

**DROP 在兩個維度上都不劣於 SSD**——平時不佔任何空間，被需要時還更便宜。
在賽局意義上 **SSD 這一階是被支配的策略，理性的 policy 永遠不該選它**。

⚠️ 這是**本平台、本裝置、本 vLLM 版本**的結論，不是普遍結論。
但它恰恰是論文核心主張的一個實例：**最佳動作空間本身就與硬體有關。**
論文若把六階當成普遍適用，這個結果就是反例。

### 🔴 發現 11：SSD 階的成本**不是** I/O 主導，是軟體層主導

| | SATA | NVMe |
|---|---|---|
| fs 階實際讀取量 | 1 GiB | 3 GiB |
| fs 階實際讀取時間 | 2.18 s | 1.94 s |
| **換算頻寬** | **0.49 GB/s** | **1.65 GB/s（3.4×）** |
| 端到端 warm TTFT | 5,803 ms | **6,600 ms（更慢）** |

**硬碟頻寬快 3.4 倍，端到端反而更慢。** 真正的磁碟讀取只佔搬運成本
（~5.7 s）的約 **35%**，其餘 ~65% 是 vLLM tiering 層的開銷：
lookup、cascade 排程、promotion job。

→ **論文 §5 把 SSD 階的成本寫成「NVMe I/O + PCIe」是歸因錯誤。**
   主導項是軟體層，而軟體層的開銷**不會因為換更快的硬碟而下降**。
   成本模型要拆成 `C_ssd = C_lookup + C_sched + C_io(device)`，
   只有最後一項與裝置有關。

⚠️ NVMe 那次的 `lookup_async_delay_seconds_sum` 累計到 **733 秒**
（SATA 只有 0.026 秒）。`/` 是全機共用且已用 88% 的根分割區，
可能受其他使用者影響。**此項標為待查，不作為結論依據。**

### 裝置盤點（先前未記錄）

| 掛載點 | 裝置 | 介面 | 可用 |
|---|---|---|---|
| `/ssd1`–`/ssd8` | Samsung 870 QVO | **SATA QLC** | 4.3G–547G |
| `/` | Crucial P3 | **NVMe** | 433G |
| （未掛載） | **WD_BLACK AN1500 1.8T** | **NVMe** | 閒置，需 root 掛載 |

論文動作空間寫的是「SSD：**NVMe** I/O」，而先前是在 SATA QLC 上量的。
兩者現在都量了，結果見上。

### ⏳ retrieval（被需要時的成本）—— 刻意延後

這一項量的是「把 block 從 CPU/SSD 搬回來要多久」，**走的正是 PCIe**。
量測當下整機是 `HEAVY`（6 個外來 process 在 GPU 1–6，97% 使用率），
而本專案已實測整機忙碌會讓卸載的 warm TTFT **灌水 26–52%**（見發現 5）。

→ 排進凌晨批次 `_big/logs/overnight_quiet.sh`（cron `0 3 * * *`，一次性），
開跑前先等整機 `QUIET`。

`code/gpu_guard.py` 新增 `host_contention()`：`GpuWatcher` 只看本卡有沒有別人，
但 PCIe / host RAM / `/dev/shm` 是**全機共用**——別人在其他卡上跑不會出現在
本卡的 compute-apps 裡。M2/M3 的每一列現在都帶
`host_contention` / `foreign_gpu_count` / `foreign_max_util`。

⚠️ **既有 640 列的 `host_contention` 是 `UNKNOWN` 不是 `QUIET`**——
那些 run 量測時根本沒有記錄整機狀態，無法回溯。
把「沒量」寫成「乾淨」就是編造數字。凌晨批次會重跑 M3 serial 全套並如實標記。

---

## Milestone 3 — Tier 0 Baselines ✅ PASS（兩個模型 × 5 baselines，parallel mode）

**狀態**: PASS（llama + qwen 各五個 baseline 全部有數據，共 320 列）
**⏳ 進行中**: `--serial` 定稿 pass（`_big/logs/m3_serial.log`）
**執行時間**: 2026-08-30 18:07 → 18:31
**指令**: `python code/m3_baseline.py --all --model llama`（+ lmcache 單獨一張卡）
**產出**: `results/m3_baseline/baseline.csv`（160 列）
**分析**: `python code/analyze_m3.py --mode parallel`

### 工作負載（與計畫書 §4 的差異，這裡說明為什麼）

計畫書只寫「單請求 × context 遞增」。**照字面做量不到任何東西**——
KV 卸載只在**有重用**時才有價值；單發一個長請求，連接器把 block 存到 CPU 就沒有
下文，`lru` 與 `arc` 會給出完全相同的數字。而計畫書 §4 的 M3 驗收正好把
「`lru` 與 `arc` 數字完全相同」列為**「卸載沒真的發生」的警訊**。

所以工作負載定為**兩輪、共享前綴**：

```
cold: 依序送 N=4 個各自不同的長前綴 P_1..P_4
warm: 用同樣順序再送一次同樣的 P_1..P_4
```

`N × ctx` 刻意跨過 M1 量到的 GPU KV 容量（41,648 token），逼出逐出。
**cold 與 warm 的 TTFT 差，就是那一階卸載的價值。**
`full_gpu` 沒有第二階，warm 只能重算，是乾淨的對照組。

前綴用固定 seed 的隨機 token id 生成（不是重複同一段文字——重複文字會讓
prefix cache 在不同「前綴」之間意外命中，把 cold/warm 的對比洗掉），
並實際 encode 後多退少補，確保**恰好** ctx 個 token。

### 關鍵數字（llama，GPU KV cache = 41,648 tokens，n=4，取中位數）

| baseline | ctx | 工作集 | cold TTFT | warm TTFT | **Δ%** |
|---|---|---|---|---|---|
| `full_gpu` | 4,096 | 16,384 | 1003.4 ms | 81.4 ms | 91.9% |
| `full_gpu` | 8,192 | 32,768 | 2142.0 ms | 102.4 ms | 95.2% |
| **`full_gpu`** | **16,384** | **65,536** | 5045.0 ms | 5118.1 ms | **−1.4%** |
| **`full_gpu`** | **32,768** | **131,072** | 12893.4 ms | 12936.8 ms | **−0.3%** |
| `cpu_lru` | 16,384 | 65,536 | 4933.3 ms | 759.5 ms | **84.6%** |
| `cpu_lru` | 32,768 | 131,072 | 12224.1 ms | 1033.2 ms | **91.5%** |
| `cpu_arc` | 16,384 | 65,536 | 4946.5 ms | 758.5 ms | **84.7%** |
| `cpu_arc` | 32,768 | 131,072 | 12473.4 ms | 4425.5 ms | **64.5%** |
| `tier_fs` | 16,384 | 65,536 | 5543.2 ms | 596.9 ms | **89.2%** |
| `tier_fs` | 32,768 | 131,072 | 13925.5 ms | 1094.8 ms | **92.1%** |
| `lmcache` | 16,384 | 65,536 | 5794.8 ms | 2433.6 ms | 58.0% |
| `lmcache` | 32,768 | 131,072 | 14191.4 ms | 3662.9 ms | 74.2% |

（4,096 與 8,192 的工作集都塞得進 GPU，五個 baseline 都是 92–96%，無鑑別力，故上表略。）

### 🟢 發現 1：M1 的容量數字**準確預測**了 M3 的行為轉折點

`full_gpu` 的 Δ% 在工作集跨過 41,648 時直接崩塌：

| 工作集 | vs 實測容量 41,648 | `full_gpu` Δ% |
|---|---|---|
| 32,768 | 塞得下 | **+95.2%** |
| 65,536 | 塞不下 | **−1.4%** |

轉折發生在 32,768 與 65,536 之間，而 M1 獨立量到的容量是 41,648——**正好落在區間內**。
兩個用完全不同方法得到的量測互相印證：M1 讀的是 vLLM 啟動時的
`GPU KV cache size`，M3 量的是端到端 TTFT，中間沒有共用的假設。

**這讓論文 §2.5「24 GB 級加速器上單請求容量本身即構成壓力」從算術主張變成可觀測現象。**

### 🟢 發現 2：`lru` 與 `arc` 有明確差異，且**`lru` 贏**

計畫書 §4 的驗收條件：「`lru` 與 `arc` 有可辨別的差異；若完全相同，代表卸載沒真的發生」。

實測 ctx=32,768：**`lru` 91.5% vs `arc` 64.5%**，差 27 個百分點。✅ 驗收通過。

方向值得注意：**`arc` 輸給 `lru`**。這與本工作負載的性質一致——
每個前綴恰好被存取兩次（cold 一次、warm 一次），是純掃描式的循環存取。
ARC 的頻率分量（T2/B2）在「每個 block 都只被重訪一次」時得不到有用訊號，
而它為此付出的空間（ghost list）反而排擠了實際資料。

⚠️ **這是單一工作負載下的結果，不足以宣稱「ARC 比 LRU 差」。**
它只說明：在論文設定的長上下文重用型負載上，production 預設的兩個策略
**都不是強對手，而且彼此差異很大**——這反而讓「需要更好的策略」這個動機更站得住。

### 🟢 發現 3：加了磁碟階（`tier_fs`）在大 context 上最好

ctx=32,768 時 `tier_fs` 92.1% > `cpu_lru` 91.5% > `cpu_arc` 64.5%。
差距不大但方向穩定，且 `tier_fs` 的 cold TTFT 反而較高（13925 vs 12224 ms）
——多一階的寫入成本。**這正是論文動作空間裡 SSD 那一階的取捨形狀。**

### 🟢 發現 4：Qwen 完整複製了同樣的形狀——**跨模型的獨立驗證**

Qwen2.5-7B-1M（no-DCA）的 KV/token 是 Llama 的一半（56 vs 128 KiB），
容量因此是 2.56 倍（106,512 vs 41,648）。ctx 階梯依各自的實測容量另訂
（工作集 2 個在容量下、2 個在容量上），所以兩個模型比的是**同一個相對位置**。

| baseline | ctx 8K<br>(WS 32,768) | ctx 16K<br>(WS 65,536) | ctx 32K<br>(WS 131,072) | ctx 64K<br>(WS 262,144) |
|---|---|---|---|---|
| **`full_gpu`** | 94.6% | 95.6% | **15.6%** | **2.3%** |
| `cpu_lru` | 94.7% | 96.3% | **94.2%** | **95.2%** |
| `cpu_arc` | 94.6% | 96.4% | 94.4% | **82.3%** |
| `tier_fs` | 95.1% | 96.7% | **94.7%** | **95.7%** |
| `lmcache` ⚠️ | 94.8% | 96.6% | 86.5% | 87.6% |

**三件事在兩個模型上都成立：**

1. **`full_gpu` 的崩塌點都落在自己的實測容量上。**
   Qwen：工作集 65,536（≤106,512）時 95.6%，131,072（>106,512）時掉到 15.6%。
   Llama：工作集 32,768（≤41,648）時 95.2%，65,536（>41,648）時掉到 −1.4%。
   **兩個容量差 2.56 倍，兩個崩塌點也差約 2 倍**——這不是巧合，是同一個機制。

2. **`arc` 在最大 context 上輸給 `lru`。**
   Qwen 82.3% vs 95.2%；Llama 64.5% vs 91.5%。**方向一致，兩次獨立觀察。**
   與工作負載性質相符：每個 block 恰好被重訪一次，ARC 的頻率分量（T2/B2）
   拿不到有用訊號，而 ghost list 佔的空間反而排擠實際資料。

3. **`tier_fs`（CPU+磁碟）在最大 context 上最好**，且 cold TTFT 較高
   （多一階的寫入成本）。兩個模型都是。

⚠️ 仍是**單一工作負載**。這組數字支持的是「production 預設的兩個策略在長上下文
重用型負載上都不強、且彼此差異很大」，**不足以**宣稱「ARC 普遍比 LRU 差」。

### 🔵 附帶：Qwen 第一次跑錯了 ctx 階梯（已重跑）

第一次 qwen sweep 沿用 llama 的階梯 `[4K, 8K, 16K, 32K]`。
但 qwen 的容量是 106,512，那組階梯的工作集是 `[16K, 33K, 66K, 131K]`——
**只有最後一點超過容量**，轉折看不出來。

→ 改成 `MODELS[k]["ctx_ladder"]`，由各模型的 M1 實測容量決定，
判準是「2 個工作集在容量下、2 個在容量上」。`--list` 會把這件事印出來。
**這不是調參，是讓兩個模型比較同一個相對位置。**

### ⚠️ 必須跟數字一起出現的兩個限制

**1. `lmcache` 的比較不對等。**
lmcache 0.5.4 的 wheel **沒有編譯擴充**，啟動 log 明寫：

> `lmcache.cuda_ops compiled extension not found; CudaDeviceOps stays on the torch baseline for all ops.`

它的搬運路徑因此走 torch baseline。**58.0% / 74.2% 這兩個數字是被 handicap 的**，
不能拿來宣稱「我們比 LMCache 好」。已寫進 CSV 的 `caveat` 欄，
`analyze_m3.py` 會強制把它印在表格下方。

**2. 這一批是 `concurrency_mode = parallel`，時間數字不可直接進論文。**
四個 baseline 平行跑在四張卡上。GPU 之間獨立，但 **PCIe、host RAM 頻寬、
`/dev/shm`、CPU 都是共用的**——而卸載 baseline 量的正是 PCIe 路徑。
`gpu_guard` 擋得住**別人**插隊，擋不住**自己的其他 job**。

→ 已加 `--serial` 模式。**定稿數字必須用 serial 重跑。**
CSV 的 `concurrency_mode` 欄記錄每一列屬於哪一種，`analyze_m3.py` 拒絕混合比較。
上表的**相對關係**（誰贏誰、轉折點在哪）是可信的；**絕對毫秒數**要等 serial。

### ✅ `--serial` 定稿 pass 完成（2026-08-30 18:37 → 19:31）

**指令**: `bash _big/logs/m3_serial.sh`（5 baselines × 2 models，全部跑在 GPU 0，一次一個）
**分析**: `python code/analyze_m3.py --mode serial`

#### Llama（GPU KV cache = 48,128 tokens）

| baseline | ctx 4K | ctx 8K | **ctx 16K** | **ctx 32K** |
|---|---|---|---|---|
| 工作集 | 16,384 | 32,768 | **65,536** | **131,072** |
| `full_gpu` | 91.7% | 95.1% | **−2.2%** | **−0.6%** |
| `cpu_lru` | 92.3% | 95.4% | **89.6%** | **92.2%** |
| `cpu_arc` | 92.3% | 95.5% | 89.6% | **62.6%** |
| `tier_fs` | 92.6% | 95.5% | 89.4% | **92.3%** |
| `lmcache` ⚠️ | 92.7% | 95.6% | 70.6% | 76.7% |

#### Qwen（GPU KV cache = 121,072 tokens）

| baseline | ctx 8K | ctx 16K | **ctx 32K** | **ctx 64K** |
|---|---|---|---|---|
| 工作集 | 32,768 | 65,536 | **131,072** | **262,144** |
| `full_gpu` | 94.6% | 95.9% | **59.8%** | **−0.8%** |
| `cpu_lru` | 95.2% | 96.6% | **96.6%** | **95.2%** |
| `cpu_arc` | 95.0% | 96.5% | 96.6% | **82.7%** |
| `tier_fs` | 94.9% | 96.7% | 96.5% | **94.9%** |
| `lmcache` ⚠️ | 94.9% | 96.1% | 93.0% | 88.5% |

**Qwen 的 `full_gpu` 在 ctx 32K 給出 59.8%——這是最漂亮的一格。**
工作集 131,072 只比容量 121,072 大 8%，所以有一部分前綴還留在 GPU 裡：
**轉折不是階梯而是斜坡，而斜坡的位置正好落在容量上。**
Llama 沒看到這個中間態，因為它的階梯在容量附近跳得比較粗（65,536 已是容量的 1.36 倍）。

### 🔴 發現 5：平行跑的污染是**真的**，而且只打在 PCIe 路徑上

把 serial 與 parallel 逐格相比（80 個格子），結果是乾淨的因果證據：

| 量的是什麼 | serial vs parallel |
|---|---|
| `full_gpu` 的所有格子（**不走 PCIe 搬 KV**） | **±2% 以內** ← 對照組 |
| 卸載 baseline 的 **cold** TTFT（算力為主） | +3% ~ +10% |
| 卸載 baseline 的 **warm** TTFT（**PCIe 搬運為主**） | **−26% ~ −52%** |

也就是說：五個 server 同時搶 PCIe 時，**warm TTFT 被灌水了 26–52%**，
而完全不碰 PCIe 的 `full_gpu` 幾乎沒有變化。

**這正是我先前標記「平行跑的絕對毫秒數不可進論文」的理由，現在有數字支持。**
`gpu_guard` 擋得住別人插隊，擋不住自己的其他 job——而卸載量的正是最容易被自己人
污染的那條路。

|Δ|>12% 的 12 個格子裡，11 個是卸載 baseline 的 warm。
唯一的例外是 `qwen full_gpu ctx=32768 warm`（9307→4398，−52.7%），
那**不是**污染而是容量差異：該設定在 parallel 時 KV pool 讀到 106,512、
serial 時 121,072，而工作集 131,072 正好夾在中間（見發現 6）。

**結論**：`results/m3_baseline/baseline.csv` 裡 `concurrency_mode = serial` 的 320 列
才是可以進論文的數字。parallel 的 320 列保留作為「污染有多大」的證據，不要刪。

### 🔴 發現 6：KV pool 大小**隨 `max_model_len` 變動**——M1 的懸崖不是單一常數

同一個模型、同一張卡、同樣 `--gpu-memory-utilization 0.90`，
`GPU KV cache size` 卻不一樣：

| 來源 | `max_model_len` | llama KV pool | qwen KV pool |
|---|---|---|---|
| **M1**（`code/m1_capacity.py`，`PROBE_LEN=8192`） | 8,192 | **41,648** | **106,512** |
| **M3**（`max(ctx)+GEN_TOKENS+1024`） | 33,824 / 66,592 | **48,128** | **121,072** |
| 差 | | **+15.6%** | **+13.7%** |

**方向與直覺相反**：`max_model_len` 變大，KV pool 反而變大。
（推測是 CUDA graph 的捕捉組態隨之改變——`max_model_len` 小的時候能容納更多並行
序列，捕捉的 graph 較多／較大。啟動 log 的 `Graph capturing finished ... took X GiB`
可以驗證。已用 `_big/logs/kvpool_sweep.sh` 量這條曲線。）

**影響**：
1. `results/m1_capacity/capacity.csv` 的四個數字**只在 `max_model_len=8192` 下成立**，
   不是「該模型在該卡上的容量」這個更一般的量。CSV 已有 `max_model_len` 欄可追溯，
   但**論文引用時必須連同設定一起講**。
2. 先前寫的「M1 的容量準確預測 M3 的轉折」**仍然成立，但要用同設定的容量值**：
   * llama：容量 48,128，轉折在工作集 32,768（95.1%）→ 65,536（−2.2%）之間 ✅
   * qwen：容量 121,072，轉折在 65,536（95.9%）→ 131,072（59.8%，**部分命中**）
     → 262,144（−0.8%）✅ **而且 131,072 的部分命中正好夾在容量上**
   兩個模型的轉折點都夾住自己的容量值。**論證不變，數字要換成同設定的那一組。**
3. 論文 §2.5 的「容量懸崖」論述需要加一句：懸崖位置與部署設定
   （`max_model_len`、`gpu_memory_utilization`）耦合，不是純硬體常數。

⚠️ 這一項**還沒有改寫 `capacity.csv`**——那些數字是在 `max_model_len=8192` 下如實量到的，
沒有錯，只是適用範圍比原本以為的窄。要補的是**同設定下的第二組量測**，不是覆蓋第一組。

### 失敗與異常（三個，都會靜默給出錯的結果）

**1. `/dev/shm` 被自己的洩漏檔佔滿。**
vLLM 的 CPU 階是 `/dev/shm` 上的 mmap 檔，server 被 kill 時**不會回收**。
連跑幾輪後 221 GB 全滿（8 個孤兒檔共 231 GB，全是自己的），
接著三個帶卸載的 baseline 全部在啟動時死掉，而錯誤訊息完全不指向真因。
→ 寫了 `code/shm_gc.py`（只刪自己的、且沒有行程持有的，預設 dry-run），
並在 `m3_baseline.py` 開跑前加了餘量檢查。CPU 階從 32 GiB 降為 24 GiB
（最大工作集 4×32768×128 KiB = 16 GiB，留 1.5 倍餘裕）。

**2. `make_prefix` 的 token 數不準，整批以 HTTP 400 收場。**
第一版用 `decode(random_ids)` 造前綴，但 decode 之後再 encode **長度會變**
（BPE 會合併或拆開相鄰片段）。ctx=32768 的 prompt 實際超過 `max_model_len`。
→ 改成實際 encode 後多退少補，並把 `actual_prompt_tokens` 寫進 CSV。

**3. 例外路徑把已量到的資料丟掉。**
原本 `except` 直接 `return 1`，導致前三個 context 的有效數據全部消失。
→ 改成部分失敗也寫檔——那些一樣是實測值。

---

## 🔴🔴 Milestone 4 — 第五版（定稿）：修掉 Oracle 的作弊 bug 後，判定隨配置而變

### 又一個 bug：Oracle 一直用位置 0 的價格重算

用「命中次數」而非「時間」檢視，立刻看到矛盾：**壓力 0.5× 時五個策略的命中數
完全相同（196,352 / 8,448），時間卻差 9.7%**。命中相同時間不可能差。

```
run_online（baseline）: cost("drop", pos × BLOCK)   ← 付位置成本
run_oracle（Oracle）  : cost("drop", 0)             ← 永遠用位置 0 的價格
```

重算成本隨位置線性成長（式 eq:recompute-position），所以 Oracle 一直在用最便宜的
價格重算，而 baseline 付全價。**Oracle 在給自己作弊。**

修正後（`flat` 改帶 block 在請求中的序號）：

| 壓力 | 修正前 | 修正後 |
|---|---|---|
| 0.5× | 9.7% | **0.0%** ✅ |
| 1× | 9.7% | **0.0%** ✅ |
| 2× | 18.2% | 9.7% |
| 5× | 28.4% | 20.8% |

**先前所有 Oracle 數字都灌水 8–9 個百分點。**

> 這個 bug **用時間看永遠抓不到**，用命中次數一眼就看出來。
> 教訓：報告快取實驗時，命中次數與時間要並列——前者是機制且硬體無關，
> 後者是結果但依賴成本模型。

### 定稿的 go/no-go 表

| 工作負載 | 重用率 | 壓力 | headroom | 判定 |
|---|---|---|---|---|
| 合成 Zipf α=0.9 | 85.1% | 1.0× | 0.0% | 🔴 |
| 合成 Zipf α=0.9 | 85.1% | 1.8× | 9.7% | 🟡 |
| 合成 Zipf α=0.9 | 85.1% | 3.3× | **20.8%** | ✅ |
| 合成 Zipf α=0.9 | 85.1% | 6.5× | **16.2%** | ✅ |
| **conversation @ BF16 預算** | 36.6% | 60.8× | **16.4%** | ✅ |
| **toolagent @ BF16 預算** | 55.3% | 60.9× | 14.1% | 🟡 |
| **conversation @ AWQ 預算** | 36.6% | 10.7× | 7.3% | 🟡 |
| **toolagent @ AWQ 預算** | 55.3% | 10.7× | **4.6%** | 🔴 **NO-GO** |

### 🔴 headroom 有兩個自變數，不是一個

先前以為只有「壓力」。實際上還有「重用率」，而且我的合成工作負載在這一項上
**與真實流量差很遠**：

| | 重用率 | 必然重算（compulsory miss） |
|---|---|---|
| 我的合成 Zipf α=0.9 | **85.1%** | 14.9% |
| 真實 conversation | 36.6% | **63.4%** |
| 真實 toolagent | 55.3% | 44.7% |

**真實流量有 45–63% 的 block 是第一次看到——那部分誰都躲不掉，Oracle 也一樣。**
所以合成在壓力 3.3× 就給 20.8%，而真實 trace 在壓力 60.8× 才給 16.4%。

### 🔴 最關鍵的一列：配置越好，論文的空間越小

同一份真實流量，只換權重精度：

| 權重 | KV 容量 | 壓力 | toolagent 的 headroom |
|---|---|---|---|
| BF16 | 48,128 | 60.9× | 14.1% 🟡 |
| **AWQ-INT4** | **273,872** | **10.7×** | **4.6%** 🔴 |

**AWQ 是論文自己指定的「主力設定」。** 也就是說，把系統配置成論文建議的樣子之後，
論文貢獻的空間反而掉到 NO-GO。

### 這個結論的效力範圍（必須一併陳述）

**支持「不繼續」的**：在配置良好的 24 GB 卡上，用公開的真實生產流量，
headroom 只有 4.6–7.3%。

**支持「繼續」的**：
1. 那兩份 trace **不是長上下文流量**——中位數 6,906 token，**零筆達到 128K**。
   論文的目標情境（128K–512K）在公開資料中**沒有對應的 trace**。
2. 模擬器的前綴語意偏差仍在（miss 記成重算一個 block 而非其後全部），
   修掉後 `full_gpu` 變差、headroom 上升。方向明確。
3. 成本常數混用了 Llama-BF16 的量測與 Qwen-AWQ 的容量（見下）。

**因此正確的陳述不是「這個方向可行/不可行」，而是**：

> 在重用率 37–55%、壓力 10.7× 的真實短上下文流量下，最佳放置相對最好的線上策略
> 只有 4.6–7.3% 的空間；壓力升至 60× 時為 14–16%；在重用率 85% 的合成負載上
> 為 16–21%。**論文的價值取決於目標部署是否落在高壓力或高重用的區域，
> 而公開資料無法回答長上下文流量落在哪裡。**

### ⚠️ 待修：成本常數與容量預算混用了兩個模型

| | 來源 |
|---|---|
| 成本常數（CPU 0.588 / SSD 5.536 / DROP 4.008 ms/block） | **Llama-3.1-8B、BF16 權重、BF16 KV** |
| GPU 容量預算 273,872 | **Qwen2.5-7B-1M、AWQ 權重** |

兩者不是同一個設定。正確做法是成對：要嘛用 Llama-BF16 的常數配 48,128，
要嘛用 AWQ 設定重量一次成本常數再配 273,872。**上表的 AWQ 那兩列因此帶有此偏差。**

---

## ✅ Milestone 4 — 第四版：跑完整 trace 後判定回到 GO

**先前的 MARGINAL 判定是「只跑前 2,000 個請求」造成的假象。**

| 工作負載 | 前 2,000 請求 | **完整 trace** |
|---|---|---|
| conversation（12,031 請求） | 11.0% MARGINAL | **18.7% GO** |
| toolagent（23,608 請求） | 6.8% MARGINAL | **16.2% GO** |

原因是工作集與 GPU 預算的比例：截斷版的工作集是預算的 12.9×，完整版是 **60.8×**。
壓力越大，「知道未來」的價值越高——這與合成掃描裡 headroom 隨偏斜下降的方向一致
（偏斜小 = 分散 = 壓力大 = headroom 大）。

**教訓**：為了跑得快而截斷 trace，改變的不只是樣本數，**還改變了工作集與容量的比例**，
而那正是決定 headroom 的關鍵參數。截斷版的數字不是「不夠精確」，是**測了不同的問題**。

⚠️ 此判定仍受兩項限制：模擬器驗證只過方向不過量級（差 1.5×）、
CPU 成本在整機 HEAVY 下量測。且 Mooncake trace 的中位數只有 6,906 token，
**零筆達到 128K**，並非長 context 工作負載——見下節。

---

## 🔴 Milestone 3/4 的共同問題：先前全部在錯的 context 區間

`EXPERIMENT_PLAN.md` §2 的主力設定是 **AWQ-INT4 權重**，BF16 只是敏感度那一列。
先前一路跑 BF16 → 權重佔 24 GB 中的 15 GB → KV 預算只剩 5.9 GB → 容量 41,648
→ **M3 的 ctx 階梯最高只到 32K、Oracle 預算只有 48,128**。

改用 AWQ 之後（M1 實測）：

| 設定 | KV 容量 | 模型定址上限 | 有效上限 |
|---|---|---|---|
| llama-awq | 120,320 | 131,072 | 120,320 |
| llama-awq-kvfp8 | **240,656** | 131,072 | 131,072 ★ |
| qwen-awq | **273,872** | 262,144 | 262,144 ★ |
| **qwen-awq-kvfp8** | **547,744** | 262,144 | 262,144 ★ |
| mla-awq（MLA 架構） | **399,376** | 163,840 | 163,840 ★ |

★ = **記憶體容量超出模型可定址長度，瓶頸由硬體轉為架構**

六個設定中四個如此。**單卡 3090 的 KV 記憶體不是長 context 的瓶頸。**
`qwen-awq-kvfp8` 的 547,744 已超過 512K 的目標，是其可定址上限的 2.1 倍。

### 🔵 M3 長 context 的一個 harness bug（已修）

ctx 階梯頂端設成模型上限 131,072 時，`max_len = ctx + GEN_TOKENS + 1024 = 132,128`
超過上限，五個 baseline 全部在啟動時死掉（pydantic `ValidationError`）。
修法：新增 `model_max_len` 欄位，ctx 頂端留 4,096 餘裕並把 `max_len` 夾在上限內。
修正後的階梯：llama-awq 到 126,976（工作集 507,904）、
qwen-awq 到 258,048（**工作集 1,032,192**）。

## 🔴🔴 Milestone 4 — 第三版：改用**真實生產流量**，判定從 GO 掉到 MARGINAL

**這是本專案目前最重要的結果，而且它推翻了前兩版的樂觀判斷。**

### 先前的弱點：請求分布是「假設」不是「量測」

前兩版的 Oracle 用 Zipf 合成 trace。Zipf 是快取研究的標準模型，
但**它是我假設的，不是量到的**。α 掃 0.4–1.5 只能說明「結論對這個假設不敏感」，
不能說明假設本身對。

### 找到了公開的真實 trace

**Mooncake**（FAST'25, Moonshot AI）隨論文釋出生產環境 trace：
<https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release/traces>

格式每列一個請求：`{timestamp, input_length, output_length, hash_ids}`。
**`hash_ids` 就是 block 層級的前綴共用資訊**——與模擬器需要的
`list[list[block_id]]` 完全同構，可直接餵入，不需要任何轉換或假設。

解析結果與論文回報值相符（解析正確的佐證）：

| trace | 請求數 | 前綴重用率 | 論文回報 | **擬合 Zipf α** |
|---|---|---|---|---|
| conversation | 12,031 | **36.6%** | ~40% | **0.37** |
| toolagent | 23,608 | **55.3%** | 59% | **0.58** |

**真實 α = 0.37–0.58，落在我合成掃描（0.4–1.5）的最低端。**

### 🔴 結果：三個真實 trace 全部落在 MARGINAL

（前 2,000 個請求；完整 trace 執行中）

| 工作負載 | 最佳 baseline | Oracle 改善 | 判定 |
|---|---|---|---|
| 合成 Zipf α=0.4 | `cpu_arc` | 30.6% | GO |
| 合成 Zipf α=1.5 | `cpu_arc` | 15.9% | GO |
| **真實 conversation** | `cpu_arc` | **11.0%** | 🟡 **MARGINAL** |
| **真實 toolagent** | `cpu_arc` | **6.8%** | 🟡 **MARGINAL** |
| **真實 mooncake** | `cpu_arc` | **6.8%** | 🟡 **MARGINAL** |

`EXPERIMENT_PLAN.md` §5 事先訂好的規則：**5–15% = 停下來問人。**

### 為什麼真實流量的空間小 2–4 倍

| trace | 總存取 | 不重複 block | 可最佳化的部分 |
|---|---|---|---|
| conversation | 54,559 | 38,788 | 15,771（**28.9%**） |
| toolagent | 38,193 | 21,548 | 16,645（**43.6%**） |

**「不重複 block」= 每個至少要算一次，這是誰都躲不掉的下限（compulsory miss）。**
而 **Oracle 的 recompute 次數正好等於這個數**（38,788 / 21,548）——
它已經打到理論下限，一次多餘的重算都沒有。

**策略只能在「重複存取」那部分發揮，而那只佔 29–44%。**

另一個原因：工作集 / GPU 預算
* 我的合成測試：**5.4×**
* 真實 conversation：**12.9×**
* 真實 toolagent：**7.2×**

**我把問題設得比真實容易。**

### ⚠️ 三個必須一併考慮的但書

1. **只跑了前 2,000 個請求**（為了速度）。完整 trace 執行中，數字可能變動。
2. **配對是否公平待商榷。** Mooncake trace 來自多卡生產叢集，把它重放到單張
   24 GB 卡上，工作集自然是預算的 7–13 倍，compulsory miss 因此壓倒一切。
   但反過來說，論文的平台 A **就是** 24 GB 卡，主張也正是「這類卡有容量壓力」，
   所以這個配對可以說是切題的。**這是需要人判斷的地方。**
3. **模擬器的前綴語意落差仍在**（miss 記成重算一個 block，而非其後全部）。
   修掉之後 `full_gpu` 會變更差 → baseline 變差 → **headroom 可能回升**。

### 這個結果本身是有價值的

即使最終判定是不繼續，「**在真實服務流量下，單張 24 GB 卡的 KV 放置策略
只有 7–11% 的槓桿，因為 compulsory miss 佔 56–71%**」是一個乾淨、
可發表的負面結果，而且是用**真實生產 trace + 實測成本常數**得到的。

---

## Milestone 4 — Oracle 上界 🟡 第二版（成本常數已修正，仍待乾淨資料）

### 🔴 第一版作廢：成本模型的 SSD 階便宜了 13.7 倍

`load_cost_model()` 讀的 `retrieval_cost.csv` 是**量錯的那一版**（見發現 9：
CPU 階開太大，東西沒 cascade 到磁碟）。修正前後：

| 階 | 舊（錯） | 新（對） | 差 |
|---|---|---|---|
| CPU | 0.3996 | 0.5884 | 1.5× |
| **SSD** | **0.4044** | **5.5360** | **13.7×** |
| DROP（位置 0） | 4.0079 | 4.0079 | 不變 |

**修正後 SSD(5.54) > DROP(4.01)** —— Oracle 舊版以為「放硬碟便宜所以要多用」，
實際上放硬碟比丟掉重算還貴。

### 修正後的 headroom（**不降反升**）

| α | 舊（成本錯） | **新（成本對）** | 最佳 baseline 的變化 |
|---|---|---|---|
| 0.4 | 20.0% | **30.6%** | `tier_fs` → `cpu_arc` |
| 0.9 | 17.3% | **20.0%** | `cpu_arc`（不變） |
| 1.5 | 14.1% | **15.9%** | `cpu_arc`（不變） |

**為什麼會升**：`tier_fs` 先前因為 SSD 被低估而看起來最好（α=0.4 時 98,263 ms），
修正後變成 128,592 ms、輸給 `cpu_arc` 的 123,320 ms。最佳 baseline 變差 →
Oracle 的相對優勢變大。

**這反而是好消息**：結論對成本誤差有韌性——把一個常數改正 13.7 倍，
判定仍然落在 GO 區間，而且三個 α 全部通過 15% 門檻。

### Oracle 自己避開了被支配的動作

α=0.4 時 Oracle 用 SSD 只有 **64 次**（`cpu_lru` 用 CPU 63,232 次）。
它知道 SSD 比 DROP 貴，所以幾乎不碰——**與發現 10 一致，互相印證**。

優勢來源仍然是「該把誰留在 GPU」：GPU 命中 20,438 → **51,200**（2.5×），
而重算次數差不多（19,522 vs 16,128）。

---

## Milestone 4 — Oracle（第一版，已作廢，保留供對照）

**狀態**: 初步完成。**判定延後到凌晨批次的乾淨資料出來之後。**
**指令**: `python code/m4_oracle.py --validate` → `python code/m4_oracle.py --alpha 0.4 0.6 0.9 1.2 1.5`
**產出**: `results/m4_oracle/oracle.csv`、`cost_model.json`、`simulator_validation.json`
**方法**: trace-driven 模擬。單階用 Bélády/MIN（可證明最優），多階用成本感知貪婪。

### 成本模型（全部來自 M2 實測，每個 block）

| 階 | ms/block | 來源 |
|---|---|---|
| GPU | 0.0 | 定義上的基準 |
| CPU | 0.3996 | `(546.0 − 136.8) ms ÷ 1024 blocks` |
| SSD | 0.4044 | `(550.9 − 136.8) ms ÷ 1024 blocks` |
| DROP（位置 0） | 4.0079 | `513.0 ms ÷ 128 blocks` |
| DROP 位置斜率 | 0.00021 ms/block/token | 由 `C_recompute(P)` 的斜率換算 |

**重算比從 CPU 取回貴 10 倍**（4.01 vs 0.40 ms/block）——這是論文 κ 論述在本平台的直接量測。

### 初步 headroom（Zipf 偏斜參數 α）

工作負載：64 個文件 × 4,096 tokens，400 個請求；GPU 預算 48,128 tokens（M3 實測值）；
工作集 16,384 blocks = **5.4× GPU 預算**。

| α | 最佳 baseline | baseline ms | oracle ms | **改善** | 形式判定 |
|---|---|---|---|---|---|
| 0.4 | `tier_fs` | 98,263 | 78,654 | **20.0%** | GO |
| 0.6 | `tier_fs` | 95,479 | 75,889 | **20.5%** | GO |
| 0.9 | `cpu_arc` | 84,041 | 69,515 | **17.3%** | GO |
| 1.2 | `cpu_arc` | 67,067 | 55,828 | **16.8%** | GO |
| 1.5 | `cpu_arc` | 51,327 | 44,087 | **14.1%** | MARGINAL |

**趨勢很清楚且合理**：偏斜越大，headroom 越小。α 大的時候熱門集合小，
LRU/ARC 本來就抓得住，Oracle 沒什麼可贏的；α 小的時候存取分散，
「知道未來」的價值才顯現。

Oracle 的優勢來源也很一致：它把 GPU 命中從 ~19K 拉到 ~51K（α=0.4），
**重算次數完全相同**（16,128 vs 16,128）。也就是說
**Oracle 贏在「該把誰留在 GPU」，不是贏在「少重算」。**

### 🔴 為什麼這組數字**還不能**拿來做 go/no-go

`code/m4_oracle.py` 的 docstring 自己寫了兩個前提，缺一則數字不可用。
**第二個沒過。**

#### 前提 1（成本常數來自實測）✅ 通過

#### 前提 2（模擬器能複現已量到的行為）🟡 **只過了一半**

```
    ctx        實測        模擬      比值差  判定
   4096      0.98         —        —  ⚪ 工作集塞得下，無鑑別力
   8192      0.96         —        —  ⚪ 工作集塞得下，無鑑別力
  16384      9.00     14.33      59%  🟡 同向但量級偏離
  32768     12.27     18.64      52%  🟡 同向但量級偏離
```

方向 2/2 正確，**但量級高估約 1.5 倍**。已知的兩個原因：

1. **模擬把 miss 記成「重算一個 block」**，但 vLLM 的 prefix cache 是**前綴語意**——
   中間缺一塊，其後全部要重算。模擬因此低估 `full_gpu` 的成本…
   ——等等，方向相反：模擬給的比值**更大**，代表模擬**高估**了 full_gpu 相對 lru 的劣勢。
   實際原因應是模擬的 CPU 取回成本（0.3996 ms/block）偏低，
   或實測 TTFT 含有模擬沒有的固定開銷（排程、tokenize、取樣），
   把兩邊的比值都往 1 拉。
2. **CPU 取回成本是在整機 `HEAVY` 下量的**（6 個外來 process、97% 使用率）。
   本專案已實測整機忙碌會讓卸載的 warm TTFT **灌水 26–52%**。
   CPU 成本被高估 → Oracle「少去 CPU 拿東西」的優勢被放大
   → **headroom 很可能是高估的**。

#### 結論

**這組 14–20% 是初步值。可引用的是趨勢（headroom 隨偏斜下降），不是絕對數字。**

正式的 go/no-go 判定需要：
1. 凌晨批次在整機 `QUIET` 下重量 M2 retrieval（`overnight_quiet.sh` 步驟 1）
2. 用乾淨的成本常數重跑 Oracle
3. 若量級偏離仍在，要改模擬器讓 miss 走前綴語意（重算 miss 點之後全部），
   而不是只算一個 block

⚠️ **在上述三項完成之前，不得宣稱「Oracle 顯示有 headroom，可以繼續」。**
`EXPERIMENT_PLAN.md` §0 禁令 4 與 §5 的停損點都指向這裡。

---

## Milestone 4 補充 — 模擬器的三項系統語意修正與模型混用修復
**狀態**: PASS（模擬部分）／部分 BLOCKED（精度階需要新的 GPU 量測）
**執行時間**: 2026-08-31 15:37 → 16:4x
**run_id**:
* `20260831-153729-m4-semantics`（第一版語意消融，**已被下一項取代**）
* `20260831-15xxxx-m4-budget-sweep`（第一版預算掃描，**已被下一項取代**）
* `20260831-16xxxx-m4-rerun-llama-profile`（正式結果，llama-bf16 剖面）
**指令**:
```
python code/m4_budget_sweep.py
python code/m4_semantics_ablation.py --trace toolagent conversation --pressure 1 2 4 8
python code/m4_oracle.py --pressure 0.5 1 2 4 8 16 32 --lookup prefix --prefetch
```
**產出檔**: `results/m4_oracle/{budget_sweep,semantics_ablation,oracle}.csv`

---

### 🔴 發現 A — 模型混用（先前結果的錯誤來源）

先前的 Oracle trace 跑法用 `--gpu-tokens 273872`（**qwen-awq** 的實測容量）
去配 M2 的成本常數，而 **M2 整組都是在 `model_key=llama`、BF16 權重、
`gpu_kv_cache_tokens=48128`、ctx=16384 量的**。
等於「A 模型的記憶體預算 × B 模型的搬運/重算成本」。
`cpu_blocks` 也寫死 128 KiB/token（Llama GQA BF16 的值）。

**受影響的結論**：混用時預算大了 5.69 倍，Oracle 的逐出變成 **100% 免費**
（被逐出的 block 之後再也用不到，丟掉成本 0），CPU/SSD 階完全沒被用到。
我一度把這寫成「多階層對最佳解毫無貢獻」——**那是混用造成的假象，已作廢**。

**修法**：`m4_oracle.py` 新增 `MODEL_PROFILES`，把
「GPU 預算 + KV 每 token 位元組 + 成本模型 model_key」綁成一個剖面；
`load_cost_model(require_model_key=...)` 在 model_key 對不上時**直接拒跑**。
實測驗證：`--model qwen-awq` 現在會被擋下並印出該跑哪一條 M2 指令。

目前**只有 `llama-bf16` 有自洽的成本量測**。`llama-awq` / `qwen-awq`
的容量已由 M1 量到（120,320 / 273,872 token），但成本模型尚未量 → 不可用。

### 發現 B — 前綴語意的修正對結果幾乎沒有影響（與我的預測相反）

修正內容：vLLM 的 cache lookup 是**連續前綴**
（`OffloadingConnectorScheduler._lookup_complete_chunks` 的 docstring 明寫
"prefix lookup"，且回傳 token 數——token 數表達不了「0,1,4,5 命中」的形狀）。
模擬器原本把每個 block 當獨立事件，理論上會低估 baseline 的重算量。

**預測**：修正後 baseline 變慢、headroom 上升。
**實測**：headroom 變化 **≤ 0.22 個百分點**（多數情形為 0.00）。

原因量到了，不是 bug：

| trace / 策略 | 缺口後的 block | 其中仍在某一階 | 其中是第一次出現 |
|---|---|---|---|
| toolagent / full_gpu | 180,699 | **144** | 160,975 (89.1%) |
| toolagent / cpu_lru | 166,646 | **7** | 160,269 (96.2%) |
| toolagent / tier_fs | 159,950 | **0** | 159,950 (100%) |
| conversation / full_gpu | 198,587 | **0** | 172,070 (86.6%) |
| conversation / tier_fs | 170,877 | **0** | 170,877 (100%) |

**真實 LLM 流量的重用本身就是前綴結構的**（Mooncake 的 `hash_ids` 就是前綴雜湊），
所以第一個未命中剛好落在共用前綴的結尾，其後 87–100% 是這輩子第一次出現的
block——本來就要重算。**「缺口之後還有東西可以損失」這件事幾乎不發生。**

合成 Zipf trace 的中間缺口次數是 **0**：文件大小固定、請求整篇取用，
LRU 逐出剛好以整篇為單位，永遠不會出現半篇。

→ 這反過來是模擬器的一項**有效性證據**：per-block 這個簡化對前綴結構的工作負載無害。

### 發現 C — 預取的修正讓 headroom **下降**（也與預測相反）

修正內容：卸載連接器的取回是非同步的，可與前一個請求的計算重疊。
原模型在存取當下才計價 = 假設預取從不發生。
重疊上界取「前一個請求的計算時間」，**同時套用於 Oracle 與所有 baseline**。

| 工作負載 | 無預取 | 有預取 | 差 |
|---|---|---|---|
| trace:toolagent | 14.13% | **12.86%** | −1.27 |
| trace:conversation | 16.40% | **15.16%** | −1.24 |
| pressure 2.0× | 19.75% | **19.25%** | −0.50 |
| pressure 4.0× | 14.97% | **13.97%** | −1.00 |
| pressure 7.8× | 24.84% | **22.55%** | −2.29 |

方向可解釋：預取只能隱藏**傳輸**，而 Oracle 的傳輸量遠少於 baseline
（Bélády 把該留的都留在 GPU），所以 baseline 受益較多，差距縮小。

### 發現 D — 免費逐出的存量，以及 headroom 的飽和

Bélády 逐出時優先挑「之後再也用不到」的 block，成本 0。
在正確的 48,128 預算下，這種免費逐出佔 **89.7–92.5%**——
CPU 階確實會被用到，但用得不多。壓力升高，免費存量才被吃掉：

| GPU 預算 (token) | 壓力 | 免費逐出 | headroom (toolagent / conversation) |
|---|---|---|---|
| **48,128（實測值）** | 60.9× | 92.5% / 89.7% | **12.86% / 15.16%** |
| 24,064 | 121.9× | 84.2% / 81.9% | 13.34% / 15.94% |
| 12,032 | 243.8× | 77.9% / 76.0% | 13.62% / 16.24% |
| 6,016 | 487.5× | 73.5% / 71.9% | 13.58% / 16.29% |
| 3,008 | 975.0× | 70.9% / 69.5% | 13.51% / 16.31% |
| 188 | 16,663× | 60.8% / 66.4% | 14.29% / 16.27% |

**headroom 在 ~240× 之後就飽和了**（13.5% / 16.3%），
跨四個數量級的壓力幾乎不動。這是一個可引用的形狀：
「壓力越大 headroom 越大」**不成立**，它有上界。

上界的來源也量到了：Oracle 的 `recompute` 在每一個預算下都**恰好等於
不重複 block 數**（toolagent 183,300、conversation 182,790）——
即強制未命中下限。Oracle 從不重算同一個 block 兩次。
真實 trace 有 45–63% 的 block 一生只被存取一次，這部分誰都省不掉。
（這同時是模擬器正確性的獨立驗證。）

### 發現 E — 合成工作負載的壓力標籤先前名不副實

`pressure:8x` 只配了文件數，但請求數固定 400，Zipf 抽樣根本碰不到那麼多文件：

| 標籤 | 配了幾篇 | 名目工作集 | 400 請求實際碰到 | **實際壓力** |
|---|---|---|---|---|
| pressure:8x | 535 | 136,960 | 47,360 | **2.8×** |
| pressure:16x | 1,070 | 273,920 | 56,320 | **3.3×** |
| pressure:32x | 2,140 | 547,840 | 63,744 | **3.7×** |

**修法**：請求數自動取 10×文件數；CSV 新增 `realized_pressure_x` 與
`unique_blocks` 欄；標籤改印實際值。舊檔留在
`/ssd7/hungwei/paper-hkv/runs/superseded/oracle.csv.mislabeled-pressure`。

### 修正後的正式數字（llama-bf16 剖面，語意 prefix + prefetch）

| 工作負載 | 最佳 baseline | baseline ms | oracle ms | headroom | 判定 |
|---|---|---|---|---|---|
| trace:toolagent（真實） | cpu_arc | 869,109 | 757,344 | **12.86%** | MARGINAL |
| trace:conversation（真實） | cpu_arc | 892,932 | 757,581 | **15.16%** | GO |
| pressure 1.0× | cpu_arc | 14,677 | 13,703 | 6.63% | MARGINAL |
| pressure 2.0× | cpu_arc | 42,426 | 34,260 | 19.25% | GO |
| pressure 4.0× | cpu_arc | 77,364 | 66,556 | 13.97% | MARGINAL |
| pressure 7.8× | cpu_lru | 259,768 | 200,955 | 22.64% | GO |

### 失敗與異常
無指令失敗。三項需要記錄的**方法錯誤**（均已修復並重跑）：
模型混用（發現 A）、壓力標籤名不副實（發現 E）、
以及先前基於混用結果所寫的「多階層無用」結論已作廢。

### 尚未完成（需要 GPU，排凌晨 3 點）
1. **M2 的 qwen-awq / llama-awq 成本模型** —— 沒有它就不能在 AWQ 預算下跑 Oracle
2. **GPU-FP8 / GPU-INT4 的取回（反量化）成本** —— M2 目前只有 4 階
   （`gpu_resident` / `cpu` / `ssd` / `drop`），論文的六階動作空間缺兩階的成本。
   **在量到之前不做精度階的 Oracle**（禁令 1）
3. M2 retrieval 在整機 `QUIET` 下重量（成本常數目前量自 `HEAVY`）

---

## 🔴 更正 — Mooncake trace 的 block 粒度解碼錯誤（2026-08-31 16:30）

**影響範圍**：所有 trace 驅動的模擬結果（本檔上一節「Milestone 4 補充」裡
`trace:toolagent` / `trace:conversation` 的每一個數字）。**全部作廢，已重跑。**

### 錯在哪

Mooncake 的 `hash_ids` 是 **512-token 的 block**，不是本模擬器用的 16-token block。
由資料本身可證（`input_length / len(hash_ids)`）：

| trace | 中位 | 平均 | 5–95% |
|---|---|---|---|
| conversation | **496.3** | 483.3 | 416–511 |
| toolagent | **487.9** | 482.3 | 444–510 |

`mooncake_trace()` 先前直接 `out.append(rec["hash_ids"])`，
等於把一個 512-token 的 block 當成一個 16-token 的 block。三重低估：

1. **工作集少算 32 倍** — toolagent 實際不重複 block 5,457,182 個
   （87.3M token），先前記為 183,300 個
2. **每個 block 的絕對位置少算 32 倍** — 這一項最嚴重。
   重算成本是位置的線性函數（式 eq:recompute-position），
   位置少算 32 倍 → 重算被算得太便宜 → **DROP 這個動作看起來比實際划算太多**
3. **請求長度中位數 6,909 token 被當成 216 token**

### 怎麼發現的

做長上下文實驗時，`m4_longctx.py` 印出「trace 中位數長度 208 token」，
與先前記錄的「Mooncake 中位數 6,346 token」矛盾。
兩個數字都是我自己算的，差 30 倍 → 去查 `input_length` 與 `len(hash_ids)` 的比值。

（先前那個 6,346 是直接讀 `input_length` 欄位算的，**那個數字是對的**；
錯的是模擬器對 `hash_ids` 的解讀。兩者從未放在一起比對過。）

### 修法

每個 `hash_id` 展開成 `512 // 16 = 32` 個連續 block，
並依 `input_length` 裁掉最後一塊多出來的部分（最後一塊通常不滿 512 token）。
這是**資料的正確解碼，不是假設**。

### 修正後的 trace 特性（與 `input_length` 欄位交叉驗證通過）

| trace | 請求 | block 存取 | 不重複 block | 長度中位 | 最大 | ≥128K | 壓力 | 重用率 |
|---|---|---|---|---|---|---|---|---|
| toolagent | 23,608 | 12,694,731 | 5,457,182 | **6,352** | 126,208 | **0** | 1,814× | 57.0% |
| conversation | 12,031 | 9,055,233 | 5,674,025 | **6,912** | 126,208 | **0** | 1,886× | 37.3% |

長度中位數 6,352 / 6,912 與 `input_length` 的 6,345 / 6,909 相符；
重用率 57.0% / 37.3% 與先前由 `hash_ids` 算的 55.3% / 36.6% 相符
（差異來自最後一塊的裁切）。**解碼正確。**

### 這對結論的方向

位置修正會讓**重算變貴**（位置 ×32 → 每 block 的重算成本從
4.008 + 0.00021×208 ≈ 4.05 ms 變成 4.008 + 0.00021×6,656 ≈ 5.40 ms，
長請求的尾端更高）。重算變貴 → **記憶體階層變得更有價值**，
但同時 **baseline 也會改用階層**（`tier_fs` 取代 `cpu_arc` 成為最佳 baseline），
所以 headroom 的淨方向要看實測。初步跡象（長上下文實驗 S=39 那列，
粒度恰好接近正確值）顯示 headroom 會**大幅下降**至 2–3%。

### 教訓（寫進 CLAUDE.md）

**任何外部資料集的單位都要用資料自身交叉驗證，不能看欄位名推斷。**
這次能抓到是因為同一個量（請求長度）被兩條不同路徑算出來過並且對不上。
以後匯入 trace 時，`mooncake_trace()` 這類函式要在載入時就做一次
「用 A 欄位驗算 B 欄位」的斷言。

## M2 補充 — 磁碟頻寬實測（2026-08-31 16:35）

**動機**：模擬器的成本模型只有「把 block 讀回來」的價格，**寫下去是免費的**。
但 toolagent 若把每個新 block 都寫一份到磁碟，需要持續 **7,172 MiB/s**
（202.9M token × 128 KiB ÷ 3,537 s）。所以「寫得下去嗎」是一個獨立於延遲的
可行性判準——而它需要**實測的**裝置頻寬，不是規格書數字（禁令 1）。

**指令**：`python code/disk_bw.py --paths /ssd7/hungwei /home/hungwei`
**產出**：`results/m2_harness/disk_bw.csv`、`disk_bw_sustained.csv`

| 裝置 | 掛載點 | 測試大小 | 寫 (MiB/s) | 讀 (MiB/s) |
|---|---|---|---|---|
| Samsung 870 QVO（SATA QLC） | `/ssd7` | 1 GiB | **492** | 522 |
| Samsung 870 QVO（SATA QLC） | `/ssd7` | **16 GiB** | **181** | 515 |
| Crucial P3（NVMe） | `/` | 1 GiB | **2,512** | 2,085 |

三次重複，中位數；整機爭用為 `HEAVY`，所以這些是**下界**。

### 🔴 QLC 的寫入懸崖

1 GiB 測到 492 MiB/s、16 GiB 只剩 **181 MiB/s**——**2.7 倍落差**。
短測整個落在 QLC 的 SLC 寫入快取裡。**KV 階是持續寫入，所以 181 才是該用的數字。**
這也是為什麼「用 dd 隨手量一下」會給出樂觀 2.7 倍的答案。

### 交叉驗證：M2 推出的 SSD 讀取成本 vs 原始頻寬

M2 由端到端 TTFT 推得 SSD = 5.536 ms/block = 2 MiB / 5.536 ms = **344 MiB/s**。
原始循序讀實測 **515–522 MiB/s**。vLLM 的 fs 階達到裸讀的 **66%**——
以每 block 一次檔案操作而言合理。
**兩條完全獨立的路徑（端到端延遲 vs 原始 I/O）互相印證。**

### 這對論文 κ 主張的意義

同一個「把逐出的 block 寫到磁碟」策略：
* 在 `/ssd7`（181 MiB/s）上——Oracle 在 32 GiB SSD 設定下需要 **267 MB/s**，**寫不下去**
* 在 `/`（2,512 MiB/s）上——同一個策略綽綽有餘

**跨裝置差 13.9 倍，而兩顆碟在同一台機器上。**
這是 κ 隨硬體變動的直接實證，且不需要第二台機器就能量到。

### 技術註記（給之後接手的人）

`O_DIRECT` 需要**頁對齊**的緩衝區。`bytearray` 是 malloc 出來的，
對齊與否看運氣——同一支程式在 `/`（nvme）成功、在 `/ssd7`（sata）回 `EINVAL`。
用 `mmap.mmap(-1, n)` 取得匿名映射即保證頁對齊。
（我一開始誤判成「檔案系統不支援 O_DIRECT」，實際兩邊都是 ext4。）

## 🔴 更正 — Oracle 的「成本感知」其實不是成本感知（2026-08-31 16:30）

**影響範圍**：所有 Oracle 的 headroom 數字。已修，正在重跑。

### 錯在哪（第一層）

論文的機制是「成本感知的多階放置」，但 `run_oracle` 的目的地選擇實際上只是
**cascade**：CPU 有位子就放 CPU、否則放 SSD、兩邊都滿才交換。從來沒有比較過
「放這裡」與「丟掉重算」哪個便宜。

而這兩者的價格是會交叉的：

    放 SSD    固定 5.536 ms/block
    丟掉重算  4.008 + 0.000210 × 位置 ms/block
    → 位置 < 7,278 token 時**丟掉比放 SSD 便宜**

真實 trace 的中位請求只有 6,352 token——**整段都在交叉點以下**。
也就是說舊 Oracle 把大量 block 塞進 SSD，之後用比重算更貴的價格讀回來。

修正後（`--oracle-dest cost-aware`）Oracle 平均快 **16.7%**（最多 29.8%）。
因為 NO-GO 判定依賴「Oracle 已經夠強」，這個修正對 NO-GO 的可信度是必要的。

### 錯在哪（第二層，被自己的修正引出來的）

修完之後，SSD = 2 TiB 那一格的 **headroom 變成 −3.16%**——Oracle 輸給 `tier_fs`。
Oracle 知道全部的未來，這在定義上不可能。

原因：成本感知規則比較的是**單一 block** 的成本，但在前綴語意下，
丟掉第 k 個 block 會迫使該請求**第 k 之後全部重算**。
規則做出了局部便宜、全域昂貴的決策。

修法有兩層：
1. 新增 `tail_of[i]` =「若第 i 次存取未命中，這個請求要付的重算總量」，
   成本感知規則改用它當丟棄的邊際成本。這是上界，會讓 Oracle **偏向保留**
   ——保守方向，不會高估 headroom。
2. `dest="best"`（新預設）：cascade 與 cost-aware 兩種規則都跑，取較佳者。
   合法，因為 Oracle 的定義是「我們能構造出的最佳**離線**策略」，
   而兩種規則都是可實作的策略。

### 教訓

**我的「修正」引入了一個新錯誤，而且是被不變量抓到的，不是被我看出來的。**
這正是為什麼 `code/m4_invariants.py` 存在。

---

## 今天（2026-08-31）在同一支模擬器裡找到的六個錯誤

| # | 錯誤 | 方向 | 被什麼抓到 |
|---|---|---|---|
| 1 | 模型混用（qwen-awq 的預算 × llama 的成本） | 預算大 5.7 倍 → 假的「多階層無用」結論 | 人工比對兩支腳本的預設值 |
| 2 | Mooncake 的 `hash_ids` 是 512-token，被當成 16-token | 工作集與**位置**都少算 32 倍 → 重算太便宜 | **同一個量兩條路徑算出來對不上** |
| 3 | SSD 階設成無限大 | 白送 `tier_fs` 一個 10 TB 快取 → 低估 headroom | **物理上不可能**（碟只有 7.3 TB） |
| 4 | 寫入完全免費 | `tier_fs` 需要 4,666 MiB/s，碟只有 181 | **物理上不可能** |
| 5 | 「成本感知」其實是 cascade | 低估 Oracle 16.7% | 讀了論文對機制的描述，回頭比對程式 |
| 6 | 修 5 時引入：規則不是前綴感知的 | headroom 變負 | **邏輯上不可能** |
| 7 | `gpu_guard._smi()` 查詢失敗回傳 `[]` | 會被讀成「整機乾淨」→ 污染的量測被標成乾淨 | 規劃 AMD 移植時發現 |

**沒有一個是靠讀程式碼看出來的。** 全部靠「數字對不上」或「這不可能」。

→ 已把這些交叉檢查固化成 `code/m4_invariants.py`，
   每支掃描腳本開跑前 `preflight()`、跑完後 `check_results()`。
   三項事後檢查（命中守恆、強制未命中下限、Oracle 是上界）
   都用故意注入的錯誤驗證過會被抓到。

→ CLAUDE.md 新增禁令 6（外部資料集的單位要用資料自身交叉驗證）
   與禁令 7（「查不到」不等於「沒有」）。

## 凌晨批次 v2 的乾跑驗證（2026-08-31 16:55）

**動機**：今晚 03:00 的批次會跑今天才改的三條 M2 新路徑。
沒驗證就排上去，等於賭一整夜——失敗要到明天早上才會發現。

**做法**：把 `Server` 換成假的，只檢查「哪些階會被跑到、每一階用什麼
`kv_dtype`、CSV 欄位與檔名對不對」。不碰 GPU，因此可以在 GPU 0 忙碌時做。

**🔴 抓到一個會讓批次直接崩掉的 bug**：

    ValueError: too many values to unpack (expected 3)
    m2_cost_model.py:455  for name, _, _ in TIERS:

我把 `TIERS` 從 3 元組改成可選 4 元組（新增 `kv_dtype`）時，只改了主迴圈，
漏了摘要輸出那個迴圈。**步驟 1 會在印摘要時崩掉，且前面的量測結果不會寫檔。**
已改為 `for name, *_rest in TIERS`。

**修正後的乾跑結果**：

| 步驟 | 起幾個 server | kv_dtype |
|---|---|---|
| 1（精度階） | 3 | `auto`、`fp8`、`int4_per_token_head` |
| 2/3（四階） | 4 | 全部 `auto`，spec 為 None／CPUOffloadingSpec／TieringOffloadingSpec／None |
| `--stage all` | 6 | 上述聯集 |

檔名互不覆蓋：`retrieval_cost.csv`、`retrieval_cost_precision_tiers.csv`、
`retrieval_cost_sata_quiet.csv`、`retrieval_cost_llama-awq.csv`、
`retrieval_cost_qwen-awq.csv`。

**重算位置掃到 114,688 的可行性**（步驟 4b）：

    max_len = 114,688 + 2,048 + 1,024 = 117,760 ≤ llama-awq 實測容量 120,320  ✅
    117,760 token 的 KV = 14.4 GiB；AWQ 權重約 5.7 GB，
    24 GB 卡 @ gpu_memory_utilization 0.90 可用 21.6 GB → 剩 15.9 GiB 給 KV  ✅

**模型可用性**：`llama-awq` 已在 HF 快取（5.4 GB）；
`qwen-awq` 為本機路徑，`config.json` 確認 `quant_method=awq`、
`max_position_embeddings=262144`、`rope_scaling=None`（無 DCA）。

**教訓**：排進 cron 的東西要先用假的外部相依乾跑一次。
這次省下的是一整夜。

## Milestone 5 — 品質：任務決定 ε，不是精度（2026-08-31 18:43）

**狀態**: PASS（先導）
**run_id**: `20260831-181834-m5-needle-pilot`
**指令**:
```
python code/m5_quality.py --gpu 0 --mode needle --model qwen-awq \
  --needle-ctx 32768 --needle-depths 0.05 0.25 0.5 0.75 0.95 --needle-repeats 4
```
**產出**: `results/m5_quality/needle_pilot_32k.csv`（80 列）、
`results/m5_quality/gsm8k_precision_n1000.csv`（4,000 列）

### 🔴 發現：同一組精度設定，在兩個任務上的 ε 差了 30 倍

| KV 精度 | 容量 | 縮放方式 | GSM8K n=1000 | 大海撈針 32K n=20 |
|---|---|---|---|---|
| BF16 | 273,872 | — | 77.9% | **100%** |
| FP8 | 547,744 | 靜態、未校正 | 76.2%（−1.7pp） | **5%（−95pp）** |
| INT8 | 531,136 | per-token-head 動態 | 76.4%（−1.5pp） | **95%（−5pp）** |
| INT4 | 1,031,056 | per-token-head 動態 | 75.1%（−2.8pp） | **0%（−100pp）** |

GSM8K 上四者的差異**全部與 0 無法區分**（n=1000 時差值的 95% CI 為 ±3.7pp）。
大海撈針上同樣的四者從 100% 掉到 0%。

**ε 不是精度的性質，是（精度 × 任務）的性質。**
只看 GSM8K 會得出「KV 量化幾乎免費」的錯誤結論。

### 每個深度的細節（大海撈針，ctx=32,768）

| 精度 | 0.05 | 0.25 | 0.50 | 0.75 | 0.95 | 總計 |
|---|---|---|---|---|---|---|
| BF16 | 100% | 100% | 100% | 100% | 100% | 100% |
| FP8 | 25% | 0% | 0% | 0% | 0% | 5% |
| INT8 | 100% | 75% | 100% | 100% | 100% | 95% |
| INT4 | 0% | 0% | 0% | 0% | 0% | 0% |

### 🔴 發現：同樣 8 位元，差別在縮放係數而非位元寬

`fp8` 5% 對 `int8_per_token_head` 95%。兩者都是 8 位元。
差別在 `int8_per_token_head` 每個 token 每個 head 各自動態計算縮放係數，
而 `fp8` 用的是未校正的靜態係數——vLLM 啟動時就警告過
"it may cause accuracy drop without a proper scaling factor"。

**這是實作／設定的性質，不是 8 位元 KV 的本質限制。**
論文不得寫成「FP8 會破壞檢索」，要寫成
「未校正縮放的 FP8 會破壞檢索，而動態縮放的 INT8 不會」。

### 直接可操作的結論

3090 上做 512K，唯一同時「放得下」又「檢索得到」的精度是
**`int8_per_token_head`**（容量 531,136 > 524,288、檢索 95%）。
`fp8` 雖然容量更大（547,744）但檢索只剩 5%。
凌晨批次的 512K 設定已據此從 `fp8` 改為 `int8_per_token_head`。

### 為什麼換任務

先前用 GSM8K many-shot 掃 ε(f) 曲線。功效分析顯示那條路走不通：
要在 95% 信心下區分 2.8pp（f=1.0 的**上限**效果），每個設定需要 n≈1,760，
7 個 f 值就是 12,320 個請求 ≈ 8.6 小時，而中間的 f 效果更小。
已停掉正在跑的 n=400 版本（CI ±5.87pp，連上限效果都區分不了）。

### 尚未釐清：任務還是長度？

GSM8K 的前綴是 12,575 token 而 FP8 正常（76.2%）；
大海撈針是 32,243 token 而 FP8 只剩 5%。所以還分不清是任務不同還是長度不同。
已啟動 4K / 8K / 16K 的長度對照（`20260831-182840-m5-needle-ctxsweep`）：
* 若短 context 也是 5% → 問題在**任務**
* 若短 context 是 100% → 問題在**長度**，存在可量測的崩潰點

### 清理注意事項

停掉 `m5_quality` 的父行程會留下**孤兒 vllm server**佔著 22 GB——
m5 用 `start_new_session=True` 讓 server 自成 process group，
父行程的 SIGTERM 傳不到。要停 m5 必須連 `vllm serve` 一起殺。

### 長度對照：FP8 的失敗與 context 長度無關（2026-08-31 18:55）

**run_id**: `20260831-182840-m5-needle-ctxsweep`
**產出**: `results/m5_quality/needle_ctx_sweep.csv`

| ctx | BF16 | FP8 | INT8 | INT4 |
|---|---|---|---|---|
| 4,096 | 100% | **5%** | 95% | **0%** |
| 8,192 | 100% | **0%** | 90% | — |
| 32,768（先導） | 100% | **5%** | 95% | **0%** |

**三個長度給出一致的結果，連 4K 都一樣。** 所以先前「是任務還是長度」
這個問題有了答案：**是任務。**

配上 GSM8K 的對照（前綴 12,575 token、FP8 76.2%），完整的敘述是：

> 未校正縮放的 KV 量化，對「靠最近幾個範例推理」幾乎無害（−1.7pp），
> 對「從任意位置精確取回一個字串」是毀滅性的（−95 到 −100pp）。
> 同樣位元寬、改用每 token 每 head 的動態縮放就沒事（−5pp）。

這對論文的意義：**ε 必須按任務分別報告，不能給單一數字。**
而且動作空間裡的 GPU-FP8 與 GPU-INT4 兩階，在檢索型工作負載下
**實際上不可用**——不是成本高，是結果錯。

### 合併版重跑：SSD 容量把 headroom 翻了一倍

`m4_sweep.py --axis budget --ssd-gib-fixed 512` vs 舊版的無限 SSD：

| trace | 預算 | 舊（無限 SSD） | 新（512 GiB） |
|---|---|---|---|
| conversation | 48,128 | 6.05% | **12.58%** |
| conversation | 6,016 | 5.29% | **11.87%** |

新版的 12.58% 與 `ssd_sweep.csv` 在 512 GiB 那一格**完全相同**——
兩次獨立執行互相印證。

headroom 在 8 倍的預算範圍內只從 12.58% 動到 11.87%：
**磁碟階夠大時，GPU 預算幾乎不影響結果。**

## 工作負載的客觀限制：沒有資料集同時具備長 context 與真實重用率（2026-08-31 20:15）

**動機**：使用者的論文目標是 512K–768K，但今天所有的數字都建立在
Mooncake 上，而 Mooncake 的中位長度只有 6.3K。

**做法**：查文獻找長 context 的公開資料。

### 找到的：SCBench（Microsoft, ICLR 2025，arXiv 2412.10319）

`microsoft/SCBench`，12 個子集、922 筆、942 MB。
每筆的結構是 `{id, context, multi_turns[{input, answer}]}`——
**一份長 context 被多輪查詢共用**，正是 KV 重用的形狀。

實測各子集的 token 數（用 qwen-awq 的 tokenizer）：

| 子集 | 筆數 | 輪數 | context token |
|---|---|---|---|
| **scbench_qa_eng** | 69 | 5 | **745,586** |
| scbench_kv | 100 | 5 | 169,035 |
| scbench_prefix_suffix | 100 | 5 | 112,577 |
| scbench_summary | 70 | 5 | 104,545 |
| scbench_summary_with_needles | 70 | 8 | 104,645 |
| scbench_repoqa_and_kv | 88 | 8 | 68,395 |
| scbench_repoqa | 88 | 5 | 65,656 |
| scbench_many_shot | 54 | 5 | 26,474 |

`scbench_qa_eng` 的 **745,586 token 正好落在 512K–768K 的目標區間**。

### 🔴 但它的重用率偏樂觀

SCBench 的結構是「一份 context × T 輪」，所以重用率 = 1 − 1/T：

| 資料 | 長度 | 重用率 |
|---|---|---|
| SCBench qa_eng | **745,586** ✅ | 80.0% |
| SCBench 8 輪的子集 | 68K–105K | 87.5% |
| Mooncake toolagent | 6,346 ❌ | **57.0%** ✅ |
| Mooncake conversation | 6,909 ❌ | **37.3%** ✅ |
| 本專案的合成 Zipf | 可調 | 96.0%（不真實） |

**沒有任何公開資料同時具備「長 context」與「真實的低重用率」。**
這是這個題目的客觀現況，論文必須寫明。

### 對評估設計的結論

不能報單一數字，要報一個曲面 `headroom = f(長度, 重用率)`，
以兩個真實資料集為錨點、合成填中間，且合成的每一點都標明重用率。

### 🔴 一個必須寫清楚的矛盾

「重用率低」與「headroom 高」**在算術上相反**：

    重用率低 → 第一次出現的 block 多 → 強制未命中多 → 誰都省不掉 → headroom 低

實測佐證：重用率 57% → headroom 11.5%；重用率 96% → 33.75%。

### 但長度是另一條獨立的路

headroom 不只由重用率決定，也由「放錯地方的代價」決定，
而那個代價隨**絕對位置**線性成長（式 eq:recompute-position）：

| 請求長度 | 重算 ms/block | 比 CPU 貴 | 比 SSD 貴 |
|---|---|---|---|
| 6,346（Mooncake 中位） | 5.34 | 9.8× | 0.8× |
| 131,072 | 31.53 | 58× | 5.0× |
| **524,288** | **114.09** | **210×** | **18.1×** |
| 786,432 | 169.13 | 311× | 26.9× |

**在 6K 時放錯地方只損失 10 倍，在 512K 時損失 210 倍。**
所以正確的問法不是「找低重用率讓 headroom 變高」，而是
**「在真實的（偏低的）重用率下，把長度拉到 512K，看放置的價值是否隨成本
不對稱一起放大」**。這正是論文原本的 κ 主張。

今天量到的長度分箱在 ~16K 就飽和，但那條曲線只到 128K；
**512K 那一段沒有任何人量過。**

## 🔴 端到端（含 decode）的 headroom 落入 NO_GO（2026-08-31 20:35）

**run_id**: `20260831-200058-m4-decode`
**指令**: `python code/m4_sweep.py --axis ssd --ssd-gib 512 --decode`
**產出**: `results/m4_oracle/ssd_sweep.csv`（含 `decode_ms` / `prefill_ms` 欄）

| trace | 只算 prefill | **端到端（含 decode）** | 判定 |
|---|---|---|---|
| toolagent | 11.51% | **2.83%** | 🔴 NO_GO |
| conversation | 12.58% | **3.34%** | 🔴 NO_GO |

### 為什麼

放置決策只能優化 prefill。decode 期間該請求的 KV 必須整份在 GPU 裡，
每一步都要讀完，沒有搬到 CPU/SSD 或丟掉重算的自由度。

用 M3 實測擬合的 decode 成本（llama-bf16 剖面）：

    每步 = 36.768 ms + 0.005581 ms × block 數　（R² = 0.9994）

36.768 ms 的固定項是讀 BF16 權重（15.2 GB）。輸出長的請求
（p90 = 507 token、max = 2,000）因此讓 decode 主導總時間。

我先前用 qwen 的 decode 成本估出 5.9%，實際用自洽的 llama-bf16 剖面
量出 2.83%。**估計與量測差了一倍，理由是模型不同（BF16 權重 vs AWQ 權重）。**

### 🔴 這是 EXPERIMENT_PLAN §0 禁令 4 的停損點

「`< 5%` headroom = 停止」。端到端是 2.83% / 3.34%。

### 但有一個明確的保留條件，不可略過

這是 **llama-bf16 剖面**（BF16 權重）的結果。論文的主設定是 **AWQ-INT4 權重**，
其 decode 的固定項只有約一半（qwen-awq 實測 18.158 ms/步 vs llama-bf16 的
36.768）。decode 變便宜 -> prefill 佔比上升 -> 端到端 headroom 上升。

**在 AWQ 剖面的成本模型量到之前（凌晨批次步驟 4），不得宣稱這個 NO_GO
適用於論文的主設定。** 目前只能說：
「在 BF16 權重的剖面上，端到端 headroom 為 2.83–3.34%，落入 NO_GO。」

### 另一個尚未做的切分

Mooncake 的輸出長度分佈很偏（toolagent 中位 30、p90 507、max 2,000）。
按輸出長度分箱之後，**短輸出的那群請求 headroom 應該遠高於平均**，
而那群在真實流量裡佔多數。這個切分尚未做。

## 🔴 決定性結果：vLLM 0.28 的 V1 engine **無法**執行 DCA（2026-08-31 20:35）

**run_id**: `20260831-203348-dca-probe`
**產出**: `results/m1_capacity/dca_probe.json`（`verdict: SERVER_DIED`）
**指令**: `python code/dca_probe.py --gpu 0`
（模型 `graelo/Qwen2.5-7B-Instruct-1M-AWQ`，DCA config 完整，`max_model_len=524288`）

### 錯誤

```
File ".../vllm/model_executor/models/qwen2.py", line 189, in __init__
File ".../vllm/model_executor/layers/attention/attention.py", line 410, in __init__
TypeError: FlashInferImpl.__init__() got an unexpected keyword argument 'layer_idx'
```

### 為什麼這是 DCA 專屬的路徑（不是別的問題）

`qwen2.py` 只有在 `dual_chunk_attention_config` 存在時才會多傳兩個參數：

```python
**{
    "layer_idx": extract_layer_index(prefix),
    "dual_chunk_attention_config": dual_chunk_attention_config,
}
if dual_chunk_attention_config
else {},
```

而 `FlashInferImpl.__init__()` 不接受 `layer_idx`。
同一顆模型移除 DCA config 之後（`Qwen2.5-7B-Instruct-1M-AWQ-noDCA`）**整天都跑得好好的**，
所以差別就是這個設定。

這與先前讀原始碼得到的推論一致：`DualChunkRotaryEmbedding` 完整存在並產生
5 倍寬的 query，但 `v1/attention/backends/` 底下沒有任何一支消化它。
**這次是崩潰而不是靜默給錯結果，所以結論是硬的。**

### 影響

* 手上的 Qwen2.5-7B-Instruct-1M（AWQ 或 BF16）在 vLLM 0.28 上的**有效位置範圍是 262,144**
* 超過之後 RoPE 落在未訓練區間，vLLM 自己警告 "lead to nan"
* **512K 的品質評估在這個 vLLM 版本上做不到，而且換更大的卡也解決不了**
  （限制在模型／框架，不在記憶體）
* 512K 的**延遲與記憶體**仍可量（計算照樣發生），但必須標明品質無效

### 可能的出路（都未驗證）

1. 升級 vLLM 到有 DCA 的 V1 實作的版本（需查 upstream 是否已支援）
2. 換原生支援 ≥512K 且不靠 DCA 的模型
3. 只做 ≤262,144 的品質評估，512K 只報延遲

### 測試方法的一個修正

探針原本預設 `--kv-cache-dtype fp8`。但當天稍早量到**未校正的 fp8 在大海撈針
上只有 5% 正確率**（BF16 100%、int8 95%），拿它做 DCA 測試會讓對照組也失敗，
測不出 DCA 的效果。已改預設為 `int8_per_token_head`
（容量 531,312 > 524,288，檢索 95%）。這個 confound 是在啟動前發現並修掉的。

## 🔴 更正方法：這台機器 24 小時都是 HEAVY，「等安靜」不可能成功（2026-08-31 21:00）

### 資料

把所有帶 `host_contention` 的量測按小時彙總（涵蓋 00–04、12–19、22–23 時，
共 5,900+ 個樣本）：

| 時段 | QUIET | LIGHT | HEAVY |
|---|---|---|---|
| 00–04 時 | **0** | **0** | 194 |
| 12–19 時 | **0** | **0** | 3,251 |
| 22–23 時 | **0** | **0** | 742 |

每一個小時的中位數都是 **6 張外來 GPU、最高使用率 100%**，完全平坦。
昨晚的 v1 批次等滿 60 分鐘，03:59 量到的還是 HEAVY，然後照跑。

**「排凌晨等機器安靜」這個前提從一開始就是錯的**，而我一整天的排程規劃
都建立在它上面。等待只是白白浪費 60–90 分鐘。

### 但負載是**穩定的**，這改變了正確的做法

一整天 6 張 100%，不是忽高忽低。穩定的背景干擾下：

* 絕對值被灌水，**但灌水的程度一致**
* **相對比較仍然有效**——只要各階在相同條件下量
* 一階量完再量下一階，會讓「階別」與「時間」混淆

### 修法：交錯量測

`m2_cost_model.stage_retrieval` 新增 `--retrieval-repeats`（預設 3）：
每一輪把各階的順序**旋轉一格**，最後取各階的中位數。

    第 1 輪  gpu_resident → cpu → ssd → drop
    第 2 輪  cpu → ssd → drop → gpu_resident
    第 3 輪  ssd → drop → gpu_resident → cpu

這樣每一階都在不同的時間位置被量到，漂移對各階平均地作用。
乾跑驗證：3 輪 × 4 階 = 12 個 server，各階樣本數相等。

凌晨批次同步修正：
* 步驟 0 從「等 90 分鐘的 QUIET」改為「等 30 分鐘的 GPU 記憶體」
* 三個 retrieval 步驟都加上 `--retrieval-repeats 3`
* 檔名後綴 `_nvme_quiet` 改為 `_nvme_interleaved`（不再宣稱 QUIET）

### 論文要寫的

**本專案的所有計時量測都在 `host_contention=HEAVY` 下取得**
（6 張外來 GPU、100% 使用率，24 小時無例外，非本專案可控）。
量測協定以交錯重複因應：各階輪替順序重複三輪取中位數，
使穩定的背景干擾對各階平均作用。**因此可引用的是階與階之間的比值，
而非絕對延遲。**

## 🔴 更正：INT8 勝過 FP8 的主因是**格式**，不是縮放（2026-08-31 23:10）

**產出**: `code/quant_error.py`（可重跑）

### 我先前寫錯了

RUNLOG 上一節寫「同樣 8 位元，差別在縮放係數而非位元寬」。
那個結論建立在一個**我做不到的比較**上：能分離兩個變數的
`fp8_per_token_head` 在 3090 上跑不起來
（`ValueError: FP8 KV cache is not supported by the Triton attention backend
on compute capability 8.6`）。

vLLM 跑不了，但量化誤差本身用 torch 算得出來：

| 量化方式 | 相對誤差 |
|---|---|
| FP8 靜態 scale=1.0（vLLM 未校正時） | 2.64% |
| **FP8 per-token-head 動態** | **2.56%** |
| **INT8 per-token-head 動態** | **0.65%** |
| INT4 per-token-head 動態 | 11.76% |

**給 FP8 動態縮放也只從 2.64% 進步到 2.56%。INT8 是 0.65%，好 4 倍。**
所以格式才是主因。

### 機制

FP8 e4m3 = 1 符號 + 4 指數 + **3 尾數**。每個數量級只有 8 個刻度，
相對精度固定在 ~12.5%，縮放改變不了。
INT8 在指定範圍內有 255 個刻度，範圍抓得準精度就是 1/255。

FP8 拿精度換動態範圍（0.0156–448，跨 4 個數量級），
但 KV 在同一個 token、同一個 head 內值域本來就窄——**虧本的交換**。

理論驗算（兩者都吻合）：
* FP8 3 尾數位元 → RMS 相對誤差 ≈ 12.5%/√12 = 3.6%（實測 2.64%）
* INT8 255 刻度、max 縮放、高斯資料 → ≈ 3/440 = 0.68%（實測 0.65%）

### 靜態縮放是**額外**的一層傷害，只打到值域小的 head

| head 的值域 | FP8 靜態的誤差 |
|---|---|
| ±0.015 | **11.29%**（掉進次正規區，< 0.0156） |
| ±0.15 | 2.68% |
| ±1.5 | 2.65% |
| ±15 | 2.64% |

### 數值誤差可以預測檢索結果

| 量化 | 數值誤差 | 實測檢索（32K，n=20） |
|---|---|---|
| INT8 動態 | **0.65%** | **95%** |
| FP8 靜態 | 2.64% | 5% |
| INT4 動態 | **11.76%** | **0%** |

**單調對應，臨界點落在 0.65% 與 2.64% 之間。**
檢索是「找得到／找不到」的二元任務，注意力分數失真 2.6% 就足以指到錯的位置。

### 保留條件

`quant_error.py` 用高斯合成資料。真實 KV 有已知的離群值現象，
實際數字可能不同。本程式證明的是**機制**（尾數位元數主導），不是精確值。

### 論文的措辭

不可寫成「FP8 會破壞檢索」或「差別在縮放」。正確的敘述是：

> 在 sm_86 上，FP8 KV 只有未校正的靜態縮放形式可用（動態縮放需要 Triton
> backend，該平台不支援）。其檢索正確率為 5%，而 INT8 的動態縮放形式
> 保住 95%。數值分析顯示主因是格式而非縮放：FP8 e4m3 的 3 個尾數位元
> 將相對誤差鎖在 2.6%，即使改用動態縮放也只降到 2.56%；
> INT8 的 255 個刻度在同一組資料上是 0.65%。

## 🔴 GPU 0 也會被別人搶（2026-09-01 00:06）

先前根據「別人一整天只用 GPU 1–6」判斷 GPU 0 是安全的。**那是錯的。**

2026-09-01 00:06:23，`pid 2994255` 佔用 GPU 0，峰值 **23,034 MiB**，
持續到 00:16:46（10 分鐘、196 個受污染的取樣）。

守衛的處理是正確的：
* `llama-awq` 那組被標成 `CONTAMINATED_DURING_RUN`，整批作廢（rc=3）
* `qwen-awq` 開跑前檢查發現只剩 1,201 MiB，直接拒跑（rc=5）

vLLM 的錯誤訊息也很清楚：

```
ValueError: Free memory on device cuda:0 (17.7/23.68 GiB) on startup is less
            than desired GPU memory utilization (0.9, 21.32 GiB)
```

### 修法：自動重試，而不是一次失敗就整批放棄

量測腳本外面包一層重試（最多 5 次），每次先 `wait_until_free` 等這張卡
真的空出來（連續取樣、最多 40 分鐘），失敗後等 5 分鐘再試。

`rc=3`（被插隊污染）與 `rc=5`（記憶體不足）都是可重試的，
而不是應該中止的錯誤。

### 對排程的意涵

這台機器沒有任何時段是安全的：
* 整機 24 小時 HEAVY（QUIET 零次，5,900+ 樣本）
* GPU 0 也會被搶（雖然頻率低）

所以量測的設計必須假設**隨時會被打斷**，而不是「找一個安靜的時段」。

## 🔴 512K 在這顆模型上連延遲都量不到（2026-09-01 08:32）

**這比昨天的 DCA 結論更強。** 昨天寫「品質無效但延遲仍可量」，實測推翻了後半。

**run_id**: `20260901-074723-m3-qwen-awq-int8-512k-full_gpu`（五次嘗試皆同）

```
CUDA error: device-side assert triggered
```

vLLM 啟動時就警告過：
「If the model uses absolute position encoding, positions exceeding
derived_max_model_len will cause a CUDA array out-of-bounds error.」

位置超過 `max_position_embeddings = 262,144` 之後 RoPE 索引越界，
**kernel 直接掛掉**，不是記憶體不足（KV 容量 559,584 > 525,568，綽綽有餘）。

### 成功的部分

| ctx | cold TTFT | warm TTFT | 比值 |
|---|---|---|---|
| **258,048** | **1,286,524 ms（21.4 分鐘）** | **1,762 ms** | **730×** |
| 524,288 | 🔴 CUDA assert | — | — |

258,048 的 730 倍 cold/warm 落差是目前量到最極端的 prefix cache 效益。

### 兩個結論合起來

* DCA 在 vLLM 0.28 的 V1 無法執行（昨天，崩潰佐證）
* 位置 > 262,144 觸發 CUDA assert（今天，崩潰佐證）

→ **這顆模型在這個框架上的硬上限就是 262,144，且換更大的卡無效。**
要做 512K 必須換模型或換 vLLM 版本。

### 🔴 我的重試設計有缺陷，白跑了 4 小時

`run_queue.sh` 對**所有**非零 rc 都重試。但這個錯誤是**確定性**的——
五次嘗試每次都在第 3 列之後死在同一個地方，04:24 到 08:32 白花 4 小時。

**修法**：只有 `rc=3`（被插隊污染）與 `rc=5`（記憶體不足）算環境問題、值得重試；
其餘視為確定性錯誤，立刻標記失敗並往下一個工作走。

### ctx 階梯已收到模型的實際能力

`qwen-awq-int8-512k` 改名為 `qwen-awq-int8-256k`，
階梯從 `[258048, 524288]` 改為 `[131072, 258048]`，
`model_max_len` 從 528,384 改為 262,144。
失敗的資料移到 `results/superseded/baseline_512k_cuda_assert.csv`。

---

## 論文修訂 — 成本模型剖面混用（2026-09-01 14:00）

**狀態**: FIXED（程式）／PARTIAL（結果待重跑）
**觸發**: 以 `paper-writing` skill 檢視 `main.pdf` 時，發現同一份稿子裡有
兩個交叉點（7,277 與 37,615）與三個位置擬合上限（24,576、114,688、258,048），
全都沒標剖面。

### 根因

`m4_oracle.py` 把成本模型寫到 `OUT/cost_model.json`（**固定路徑**），
而 `m4_sweep.py` 的 CSV `cost_model` 欄也硬寫同一個路徑。於是：

* 每一次掃描都覆蓋上一次的 JSON
* 各剖面子目錄（`qwen-awq/`、`qwen-awq-surface-*/`）的 CSV 全部指向那個共用檔
* **`results/m4_oracle/cost_model.json` 現在裝的是 `llama-bf16`/SATA 的常數，
  不描述任何一份 CSV**

論文的 §A.2 就是從那個檔抄到 SATA 常數（CPU 0.588 / SSD 5.536 / Drop 4.008），
而同一份稿子的 §7.6 用的是 `qwen-awq`/NVMe 的交叉點。兩者相差 5.2 倍。

### 修法

`m4_sweep.py`：成本模型改寫進 `--out-dir`（每次掃描自己的目錄），
JSON 自帶 `model_profile` / `cost_model_key` / `device` / `crossover_tokens`，
CSV 的 `cost_model` 欄指向該檔。讀的人不必再從 `retrieval_csv` 的檔名反推剖面。

### 🔴 連帶發現：qwen-awq 的掃描結果早於其成本模型的最後一次更新

`recompute_position_qwen-awq.csv` 含兩個 run：

| run_id | 時間 | 位置擬合上限 | P\* |
|---|---|---|---|
| `20260831-223228-m2-recompute` | 08-31 22:32 | 較短 | 37,615 |
| `+ 20260901-131533-m2-recompute` | 09-01 13:15 | 258,048 | **37,717** |

而 `qwen-awq/ssd_sweep.csv`（12:25）與 `qwen-awq-surface-e2e/`（12:51）
**都在 13:15 之前跑完**，用的是舊擬合。

* `m4_hw_sweep.py` 已用新擬合重跑：峰值百分比只有三格差 0.1pp（穩健），
  交叉點 37,615 → 37,717，docstring 已更新
* `ssd_sweep`：**另一個並行的 session 已在 14:14--14:42 用新擬合重跑完**
  （58 列、6 個 SSD 容量，較原本的 10 列完整）。512 GiB 的端到端 headroom
  自 8.465% / 9.28% 變為 **8.415% / 9.226%**；`best_baseline` 仍為 `cpu_arc`，
  `tier_fs` 與 Oracle 的寫入頻寬（2,016 / 2,122 與 283 / 330 MiB/s）不變。
  論文已改用新值。我自己起的重跑因為會覆蓋該 session 的輸出而中止
  （`20260901-144213-m4-qwen-awq-ssd-refit`，未寫入 `results/`）。

### ⚠️ 這台機器上有並行的實驗 session

本次修訂期間，另一個 session 於 13:35--14:50 產出了
`results/m4_oracle/llama-awq/`、`llama-awq-surface/`、`qwen-awq-surface/`
與 `qwen-awq/by_length.csv`。**這些檔案未納入本次 commit**，因為不是本次工作的產出，
且當時可能仍在寫入。`qwen-awq-surface/headroom_surface.csv` 的 `cost_model` 欄
已指向該目錄自己的 JSON——**此即上述修法生效的證據**（該 session 用到了改過的
`m4_sweep.py`）。

**教訓**：成本常數的來源檔一旦追加量測，所有下游掃描結果即過期。
`sim_version` 只涵蓋模擬器程式碼，不涵蓋成本模型的資料版本。
應把成本模型的來源 CSV 摘要（run_id 集合）納入指紋。

### 論文端已改

* 新增 `tab:costmodels`（附錄 A.2）：三組常數並列，NVMe 為主、SATA 僅對照
* NVMe 持續寫入標為 `NOT_MEASURED`（`disk_bw_sustained.csv` 只有 `/ssd7` 兩列），
  並刪除靠它成立的「Oracle 於 NVMe 上可行」
* §7.5 原標題「寫入頻寬使勝出的 baseline 不可部署」在主設定不成立
  （`qwen-awq` 的最佳 baseline 是 `cpu_arc`，不用磁碟階），改為跨剖面都成立的
  「Oracle 寫入需求是 `tier_fs` 的 1/4--1/7」

---

## Milestone 5-(c) — 長上下文理解：LongBench 與 RULER（2026-09-01 20:20）

**狀態**: PASS
**執行時間**: 2026-09-01 19:33 → 20:20（8 個 job 分散在 7 張卡上平行跑）
**run_id**:
* LongBench：`20260901-193344/46/48/50-m5-longbench`（bf16 / fp8 / int8 / int4，GPU 0–3）
* RULER：`20260901-193408/10/13-m5-ruler`（int4 / bf16 / fp8，GPU 4–6）
  ＋ `20260901-195709-m5-ruler`（int8，GPU 0，等 LongBench bf16 讓出卡）
* 啟動殼：`20260901-193314-m5c-longbench-par`、`20260901-193338-m5c-ruler-par`（cmd_*.sh、context.txt、各設定的 stdout）

**指令**（每個設定一張卡，四個設定平行）:
```
python code/m5_understanding.py --suite longbench --n-per-task 50 --gpu <i> --configs <cfg>
python code/m5_understanding.py --suite ruler --n-per-task 30 --ctx 16384 --gpu <i> --configs <cfg>
```

**產出檔**:
* `results/m5_quality/longbench_precision.csv`（1,400 列 = 7 任務 × 50 題 × 4 精度）
* `results/m5_quality/ruler_precision.csv`（840 列 = 7 任務 × 30 題 × 4 精度）
* `results/m5_quality/gpu_guard_{longbench,ruler}_gpu{0..6}.json`
* 新程式：`code/ruler_tasks.py`、`code/m5_understanding.py`、`code/analyze_m5c.py`、
  `code/test_m5c_metrics.py`

**失敗與異常**:
* RULER `int4` 的 job 以 exit 1 結束，**CSV 已完整寫出（210 列）之後**才炸在
  `summarise()`：該設定 14 個任務全 0，計算「相對基準保留率」時 0/0。
  已修（`ref` 為 0 時印 `—`）。資料不受影響。
* 開跑前殺掉一輪已跑 12 分鐘的序列版本，因為原本的
  `if g.contaminated: return 3` 會在共用機器被插隊時丟掉整輪三小時的量測。
  改為「保留分數、逐列記 `own_gpu_intruders`」後重跑。理由見 CLAUDE.md §3。
* 八個 job 的 `own_gpu_intruders` 全為 0：本輪量測期間七張卡都乾淨。

---

### 🔴 發現 1：真實文件的長上下文理解，落差與**檢索**一致，不與推理一致

LongBench 7 個英文任務（`qwen-awq`、32K 視窗、n=50/任務、上游 `pred.py` 協定）：

| 任務 | 類別 | BF16 | FP8 | INT8 | INT4 |
|---|---|---|---|---|---|
| MultiFieldQA | 單文件 QA | 51.46 | 7.42 | 51.29 | 0.23 |
| Qasper | 單文件 QA | 41.37 | 2.80 | 40.79 | 0.54 |
| HotpotQA | 多跳 QA | 53.45 | 2.20 | 54.14 | 0.00 |
| 2WikiMQA | 多跳 QA | 52.23 | 7.49 | 50.87 | 0.00 |
| GovReport | 摘要 | 33.96 | 2.09 | 34.09 | 1.45 |
| TREC | few-shot 分類 | 76.00 | 22.00 | 74.00 | 0.00 |
| PassageRetrieval | 合成檢索 | 100.00 | 2.00 | 94.00 | 0.29 |
| **巨觀平均** | | **58.35** | **6.57** | **57.02** | **0.36** |
| 相對 BF16 保留 | | 100% | 11.3% | 97.7% | 0.6% |

配對差值（BF16 − X，逐題相減後 bootstrap，10,000 次）：
FP8 **51.8 pp [47.3, 56.3]**、INT8 **1.3 pp [0.0, 2.8]**、INT4 **58.0 pp [53.7, 62.3]**。

**GSM8K 說「量化幾乎免費」，LongBench 說「掉掉九成」。同一組設定、同一台機器。**

---

### 🔴 發現 2：INT8 不是無條件安全——RULER 的 UUID 多鍵檢索掉掉一半

RULER 7 個合成任務（16K 上下文、n=30/任務）：

| 任務 | 類別 | BF16 | FP8 | INT8 | INT4 |
|---|---|---|---|---|---|
| NIAH multikey-2 | 多鍵檢索（干擾針為海） | 100.00 | 0.00 | 90.00 | 0.00 |
| NIAH multikey-3 | 多鍵檢索（UUID） | 100.00 | 0.00 | **53.33** | 0.00 |
| NIAH multivalue | 一鍵四值 | 100.00 | 0.00 | 93.33 | 0.00 |
| NIAH multiquery | 四鍵各查一次 | 100.00 | 0.00 | 92.50 | 0.00 |
| VT | 多跳追蹤 | 83.33 | 1.33 | 76.67 | 0.00 |
| CWE | 詞頻聚合（前 10） | 72.33 | 0.67 | 60.33 | 0.00 |
| FWE | 詞頻聚合（Zipf 前 3） | 85.56 | 21.11 | 87.78 | 0.00 |
| **巨觀平均** | | **91.60** | **3.30** | **79.13** | **0.00** |
| 相對 BF16 保留 | | 100% | 3.6% | 86.4% | 0% |

INT8 的配對差值：合併 **12.5 pp [8.6, 16.7]**——**與 0 可區分**。
其中絕大部分來自 `niah_multikey_3`：**46.7 pp [30.0, 63.3]**。

**這改寫了 M5 先導的結論。** 先導說「INT8 動態縮放不破壞檢索（95%）」，
那是**單鍵、有語意的針**。換成 **UUID 的鍵與值、且整片海都是同構的干擾針**，
同一個 INT8 只剩 53.3%。可宣稱的是：

> INT8 per-token-head 在 GSM8K（−1.5 pp，n.s.）、LongBench（−1.3 pp，CI 觸 0）
> 上讀起來無損，卻在 RULER 最難的檢索變體上掉掉一半。
> **「這個精度安全」這句話，取決於你跑的是哪個 benchmark。**

---

### 發現 3：答案是「整份上下文的統計量」的任務，對 KV 雜訊有抵抗力

FWE（找 Zipf 分佈下最高頻的三個詞）是唯一 FP8 沒有全失的任務（21.11 vs 其餘 ≤ 7.49），
也是唯一 INT8 沒有掉的任務（87.78 vs BF16 85.56，差值 −2.2 pp [−6.7, 2.2]）。
CWE（前 10 高頻詞，要精確列出 10 個詞）則掉 12.0 pp [4.7, 20.3]。

**梯度是「答案有多依賴精確位置的精確 token」**：
全域統計量（FWE）> 聚合但要逐項列出（CWE）> 多跳追蹤（VT）> 精確檢索（NIAH）。

---

### 方法學紀錄

* **配對統計**。四個設定吃同一批 prompt（同題目、同順序、`temperature=0`、
  `seed=12345`），所以差值逐題相減再 bootstrap。GSM8K 那輪的 ±3.7 pp
  有一部分就是沒用上這一點。
* **計分函式已對上游驗證**。`code/test_m5c_metrics.py`：9,500 組隨機字串 +
  真實預測，自寫的 `qa_f1`/`rouge_l`/`classification`/`retrieval`
  與 LongBench 上游 `metrics.py` **零筆不一致**。
* **與公開榜單不可直接比**：AWQ-INT4 權重 + 移除 DCA 的 Qwen2.5-7B-1M、
  每任務只取前 50 筆、32K 視窗。**但四個設定吃的是同一批 prompt，
  所以跨精度的相對變化不受影響**——論文要宣稱的正是這個。
* **RULER 的兩處刻意差異**：haystack 一律用上游自己的 `noise` 選項
  （essay 版需要爬 paulgraham.com，HF 鏡像需授權），且不含需要 SQuAD/HotpotQA
  原始檔的 `qa_1/qa_2`（真實文件由 LongBench 那一半負責）。
  影響 `niah_multivalue` 與 `niah_multiquery`：任務語意不變，難度略降。

### 論文端已改

* 表 9（`tab:eps-task`）由「兩種任務」擴為「四種任務型態」：
  GSM8K 推理 / 大海撈針檢索 / LongBench 理解 / RULER 整合，並加一列**全距**
  （2.8、100.0、58.0、91.6 pp）——同一組設定換個任務型態，$\epsilon$ 差 21 至 36 倍
* §6.6 新增三段：〈真實文件上的長上下文理解與檢索同側〉、
  〈「哪個精度安全」的答案取決於評測選了哪個 benchmark〉、
  〈梯度由「答案是否為某個確切位置的確切 token」決定〉
* 附錄新增 A.5〈長上下文品質評測的逐任務分數〉+ 表 13：
  LongBench 7 列、大海撈針 4 個長度、RULER 7 列 × 4 個精度，
  另附配對 bootstrap 的說明、與公開榜單不可並列的三項理由、計分函式的驗證
* §8 結論段與附錄 B「尚未執行的評估指標」一段同步更新
* **原本畫的三面板圖 4 已刪**：它與表 9 是同一批數字，圖表並列屬重複
  （2026-09-01 使用者決定）。`notebooks/paper_figures.py` 的 `fig_quality()`
  一併移除，需要投影片版時自 git history 取回

### 數字的可追溯性

新增 `code/audit_m5c_claims.py`：把表 9、表 13 的每一格、表題的極值句、
以及正文引用的 9 組配對 bootstrap 區間，對 `results/m5_quality/*.csv` 重算比對。
判準是**「論文印的一位小數 == CSV 值的一位小數」**，不是容忍值——
第一次跑就抓到 HotpotQA 的 BF16 被寫成 53.5（實為 53.448 → 53.4），
以及表 9 表題把全距倍數寫成「20 至 33 倍」（實為 20.7--35.7 → 21 至 36 倍）。
兩處已修，現在 `python code/audit_m5c_claims.py` 回傳 0。

---

## Milestone 2 補充 — GPU 精度階的反量化成本（QUIET 重量）
**狀態**: PASS（int4）／PARTIAL（fp8 與零無法區分）
**執行時間**: 2026-09-01 23:25 → 23:5x
**run_id**: `20260901-232519-m2-prec-tiers-quiet`
（前一次 `20260901-225552-m2-prec-tiers-quiet` **FAIL，exit 1**，被工作集守門擋下，見下）
**指令**:
```
python code/m2_cost_model.py --gpu 0 --stage retrieval \
  --tiers gpu_resident gpu_fp8 gpu_int4 --ctx 16384 --n-prefixes 1 \
  --retrieval-repeats 3 --csv-suffix _precision_tiers_quiet
```
**產出檔**: `results/m2_harness/retrieval_cost_precision_tiers_quiet.csv`（18 列，全部 `host_contention=QUIET`）

### 關鍵數字（warm TTFT，ctx=16,384，n=3 取中位數）

| 階 | 中位數 (ms) | 全距 (ms) | 相對基準 | 每 block 反量化 |
|---|---|---|---|---|
| `gpu_resident`（BF16） | 143.26 | 140.8–147.1 | — | 0（基準） |
| `gpu_fp8` | 145.49 | 137.8–149.1 | +2.22 | **NOT_MEASURED** |
| `gpu_int4` | 155.97 | 149.9–176.7 | +12.71 | **0.0124 ms/block** |

`fp8` 判為 NOT_MEASURED 的理由：warm 全距 137.8–149.1 與基準 140.8–147.1 **完全重疊**，
n=3 下與零無法區分。中位數的 +2.22 ms 小於階內全距（6.3–11.3 ms），是雜訊不是訊號。

### 🔴 發現 1 — 污染版的排序是錯的，而且錯得看不出來

舊檔 `retrieval_cost_precision_tiers.csv`（run_id `20260831-210558`）18 列**全部 HEAVY**
（外來 5–6 張卡、util 96–100%），依 CLAUDE.md §3 時間欄本應作廢，但
`m4_oracle.load_precision_tiers` 照樣從它算出常數：

| 階 | 污染版中位數 | 污染版換算 | 乾淨版中位數 | 乾淨版換算 |
|---|---|---|---|---|
| BF16 | 138.48 | 0 | 143.26 | 0 |
| FP8 | 164.12 | 0.0250 ms/blk | 145.49 | NOT_MEASURED |
| INT4 | 155.34 | **0.0165 ms/blk** | 155.97 | 0.0124 ms/blk |

污染版讓 **INT4 的反量化比 FP8 便宜**——位元更少、縮放更複雜卻更快，物理上說不通。
乾淨版恢復單調：BF16 < FP8 < INT4。

另一個獨立證據：污染版同一階的 `gpu_kv_cache_tokens` 逐列不同
（fp8 有 83,312 與 96,272 兩個值、int4 有 156,832 與 181,216），
代表那幾列根本不是同一個設定；乾淨版每階只有一個值（fp8 96,272、int4 181,216）。

### 發現 2 — 精度階的取回成本相對其他動作可忽略

以 `llama-bf16` 剖面的其他常數為參照（`results/m4_oracle/cost_model.json`）：

| 動作 | ms/block |
|---|---|
| `Gpu4` 反量化 | **0.0124** |
| `Cpu` 取回 | 0.588（**47×**） |
| `Ssd` 取回 | 5.536（**446×**） |
| `Drop` 重算（位置 0） | 4.008（**323×**） |

**意涵**：GPU 精度階在成本模型裡近似一個**純容量乘數**，取回端的附加費可忽略。
因此「該不該用精度階」不是成本問題，是 $\epsilon$ 問題——
而 §6.5 的品質量測顯示三個低精度階在檢索任務上都不合格。

### 失敗與異常

**第一次 run 失敗（exit 1）**，完整錯誤：
```
🔴 工作集 16,384 token（1 × 16,384）≤ GPU KV 容量 48,128 token。
   東西整個塞得進去，不會逐出，量到的四階會一模一樣，而且看起來完全正常。
   請把 --ctx 調到大於 48,128（或增加 --n-prefixes）。
```
原因：`_check_workset_exceeds_capacity` 是為**搬運階**（cpu/ssd/drop）寫的——那三階
必須發生逐出才量得到。GPU 常駐的精度階量法剛好相反：工作集必須**塞得進** GPU，
warm 才是純 prefix-cache 命中。守門對每次 `--stage retrieval` 都跑，於是擋掉了
唯一合法的精度階設定。照它的提示調大 `--n-prefixes` 也不行：fp8 的 KV 容量是
bf16 的 2 倍，同一工作集會讓 `gpu_resident` 逐出而 `gpu_fp8` 不逐出，兩邊條件不同。

**修法**（`code/m2_cost_model.py`）：把守門限定在有非 GPU 階的 run，其餘行為不變。
```python
needs_evict = any(not e[0].startswith("gpu_") for e in order)
...
if name == "gpu_resident" and needs_evict:
    _check_workset_exceeds_capacity(ctx, n_prefixes, s.kv_tokens)
```

### 連帶修正

`m4_oracle.load_precision_tiers` 加了兩道守門，並新增
`code/test_precision_tiers.py`（四條假輸入測試，全過）：
1. **只吃 `host_contention == QUIET` 的列**；否則回 NOT_MEASURED 並附原因。
2. **階內全距與基準重疊時回 NOT_MEASURED**（非參數判準，n 小時不得把雜訊讀成訊號）。

檔案優先序改為 `_quiet` > `_{device}` > 無後綴，舊的污染檔保留供對照，不刪。

### 與論文假設的差異

論文 §6.2 寫「精度階能額外帶來多少空間，本文未量」——這句仍然成立，
但現在有了成本側的答案：**取回成本可忽略（發現 2）**，所以缺的只有 $\epsilon$ 側。
`int8` 一階的反量化成本仍未量（本次未納入 `--tiers`）。

---

## Milestone 5 第二階段 — 未來效用預測器（實作 + 訓練）
**狀態**: PASS（PoC 成立：線上策略贏過所有可部署的 baseline），
但**三個發現都指向「預測不是瓶頸」**，且其中一個推翻了 M4 的 headroom 是不是緊的
**執行時間**: 2026-09-05 12:49 → 13:45
**run_id**:
* 特徵　`20260905-131229-m5p-feat-{toolagent,conversation}-qwen-awq`（W=1×）、
  `20260905-131946-m5p-feat-toolagent-qwen-awq`（W=35×）
* 訓練　`20260905-1314xx/1315xx/1316xx-m5p-train-*`（W=1×，葉 63/7/2）、
  `20260905-132147-m5p-train-toolagent-qwen-awq`（W=35×）
* 策略　見 `results/m5_predictor/policy_sim.csv` 的 `run_id` 欄（六次掃描）

**指令**:
```bash
export PYTHONPATH=/ssd7/hungwei/paper-hkv/pylibs
python code/test_m5_predictor.py                   # 八組手算單元測試
python code/m5_policy_sim.py --check-shim          # 記帳等價性（必跑）
python code/m5_predictor.py all --trace toolagent --model qwen-awq --device nvme --window-mult 1.0
python code/m5_policy_sim.py --trace toolagent --model qwen-awq --device nvme --segment test \
    --dests cost-aware cascade --losses sym_l2 cost_l2 --oracle-signal ordering both \
    --train-run /ssd7/hungwei/paper-hkv/runs/20260905-131441-m5p-train-toolagent-qwen-awq
python code/m5_summary.py                          # 判定材料
```

**產出檔**: `results/m5_predictor/{samples,predictor_metrics,calibration_bins,cost_by_position,feature_importance,policy_sim}.csv` + `README.md`；
程式 `code/m5_{predictor,policy_sim,summary}.py`、`code/test_m5_predictor.py`

### 設定

`qwen-awq` 剖面 / NVMe 成本常數（CPU 0.298、SSD 10.245、重算 3.546 + 0.000178×位置 ms/block），
GPU 273,872 token（17,117 blocks）、CPU 24 GiB、SSD 512 GiB、prefix lookup、預取開、**含 decode**。
評估段是**時序切分的後 30%**（模型沒看過），baseline 與 oracle 跑同一段、同樣空的起始快取。
LightGBM 4.7.0 側裝於 `/ssd7/hungwei/paper-hkv/pylibs`（`--target`，不動 vLLM venv）。
**全程不碰 GPU**；唯一的時間量測是 CPU 推論延遲，量測時的 loadavg 已逐列記錄。

### 三道自我檢查（先看這個，再看數字）

1. **記帳等價**：`--check-shim` 把預測換成「上次存取時刻」（＝LRU）之後，
   本檔的迴圈**逐位元**重現 `m4_oracle.Sim.run_online("lru")`——三種階層設定的
   `total_ms`、`hits`、`writes` 全等。不然「線上策略 vs oracle」是拿兩把尺在比。
2. **正樣本率的交叉驗證**：「下次存取落在 W 內」的比例（本檔）必須等於 M4 實測的
   GPU 命中率（另一條路徑）：toolagent **34.8% vs 34.9%**、conversation **4.3% vs 4.3%**。
3. **決定性**：七組訓練重跑，`model_*.txt` 的 md5 與原本**逐位元相同**
   （`seed=1234`、`deterministic=True`），所以 `predictor_metrics.csv`（重跑的 run_id）
   與 `policy_sim.csv`（原本的 run_id）指的是同一個模型。

### 關鍵數字

**(1) 資料與預測器**

| trace | W | 樣本 | 正樣本率（全體） | AUC | ECE | Spearman（正樣本內） |
|---|---|---|---|---|---|---|
| toolagent | 17,117 | 3,169,646 | 34.8% | 0.9995 | 0.0003 | **0.233** |
| conversation | 17,117 | 2,260,727 | 4.3% | 0.9988 | 0.0006 | 0.999 |
| toolagent | 599,095 | 3,024,068 | 48.7% | 0.896 | 0.032 | 0.701 |

負樣本 100% 來自機制 (b)（滑動視窗到期）——機制 (a) 只能產生正樣本，
這與 §5.2 的論證一致，也是這條管線唯一能產生 `Drop` 目標群體的路徑。

**(2) 線上策略（toolagent，端到端，後 30%）**

| 策略 | 總時間 (ms) | vs 最佳 baseline | 拿到 oracle headroom |
|---|---|---|---|
| oracle（M4 貪婪構造） | 21,037,931 | +8.49% | — |
| **tiara_sym_l2**（預測排序 + 成本門檻） | **22,793,531** | **+0.85%** | **10.0%** |
| cpu_lru（最佳 baseline） | 22,989,554 | 0 | 0 |
| tiara_sym_l2_cascade（同一個預測器，無成本模型） | 24,782,081 | **−7.80%** | −91.8% |
| tier_fs | 24,884,583 | −8.24% | −97.1% |

**成本模型才是把預測變成勝利的那一半**：同一個預測器換成 cascade 目的地規則就掉到
比 baseline 差 7.8%。學到的策略對 SSD 階的寫入是 **0 MiB/s**（tier_fs 需要 2,011 MiB/s）——
因為 $p^{*}_{\act{Ssd}}(pos) = C_{ssd}/C_{drop}(pos) > 1$ 在 $P^{*}=37{,}635$ token 以下恆成立，
而幾乎所有重用都落在 8K 以下。**這是式 (9) 直接推出來的，不是調出來的。**

conversation 同樣方向但幅度小：+0.38%、拿到 4.2%。

**(3) 熱路徑成本**：對 64 個候選推論 **173 μs**（k=1 為 30 μs、k=256 為 584 μs，
即「≈30 μs 固定 + 2.3 μs/候選」，單執行緒、LightGBM 4.7 Python API）。
訓練 1.88M 樣本 **4 秒**。論文引 LRB 的「64 候選 30 μs / 訓練 300 ms」**不是本機的數字**。
論證仍成立：2.7 μs/候選 vs 單次重算 3,546 μs ＝ **0.08%**。

### 🔴 三個「預測不是瓶頸」的發現

**發現 1 — AUC 0.9995 而排序幾乎沒學到，現有的診斷都測不到。**
把**逐出順序**換成真值、其餘（目的地規則、成本模型、記帳）完全不動：
toolagent 22.79M → **20.60M ms**、conversation 24.33M → **23.03M ms**。
線上策略與 oracle 的差距**幾乎全部來自排序**。機制有兩層：

* AUC 問「會不會在 $W$ 內被用到」，Bélády 要「誰先被用到」；
* 更關鍵：標籤在 $W$ 處**設限**，於是所有「$W$ 之外」的 block 預測值全部擠在
  $\log(W{+}1)$ 附近成為**同分**——而真值能分辨「$2W$ 後會用到」與「再也不會用到」。
  這在 conversation 特別致命：95.7% 的樣本都在那一堆同分裡。

論文 §B.6 提的兩個診斷（校準、成本加權錯誤）與 AUC 全部漂亮，**卻測不到這件事**。

**發現 2 — 加權訓練輸給「對稱訓練 + 移門檻」，而且 κ 越大輸越多（方向與 §5.3 的預測相反）。**
在預測夠難的設定（W=35×、測試段正樣本率 50.4%、AUC 0.896）上：

| | 成本加權錯誤 |
|---|---|
| 對稱 L2 @ 0.5 | 667,207 ms |
| 對稱 L2 @ $p^{*}$（Elkan 基線） | **132,291 ms（−80%）** |
| 加權 L2 @ $p^{*}$ | 165,298 ms（比 Elkan 差 25%） |

逐位置分箱（κ 由 12 增到 60）：加權的增益 −2.1% → −14.0% → −84.3% → **−428.5%**。
§5.3 預測「差距隨 κ 增大而擴大（有利加權）」，實測是**隨 κ 增大而惡化**。
依論文自訂的判準——「若增益全部來自門檻移動，則加權訓練並無貢獻」——
**這一輪的結論是主張應縮減至門檻校準**。$n{=}1$ seed、單 trace、單平台，
但高 κ 區差 4 倍以上，不像雜訊。（這正是 `OPEN_ISSUES.md` B4 擔心的情況。）

**發現 3 — 🔴 M4 的 oracle 不是緊的：多給它一個 512 GiB 的 SSD 階，它會變慢 3.3%。**

```
oracle（SSD 512 GiB）  21,037,931 ms   headroom 8.49%
oracle（SSD   0 GiB）  20,353,337 ms   headroom 11.47%
```

**最佳策略不可能因為多了一個可用資源而變差**——這是結構性的矛盾，不是數值誤差，
證明 `run_oracle` 的目的地規則在這組常數下**用錯了 SSD 階**：
它拿「整條尾巴」當丟棄的邊際成本（`tail_of`），於是 $C_{ssd} < \text{tail}$ 幾乎恆成立，
把 253,269 個 block 寫上 SSD、再以 10.245 ms 讀回來，而重算只要 3.5–4 ms
（$P^{*}=37{,}635$ token 以下重算比較便宜，而幾乎所有重用都在 8K 以下）。

獨立佐證：一個**不同的離線構造**（完美排序 + 式 (9) 的門檻式目的地規則）
在同樣有 SSD 的條件下也贏過它 **2.09%**（20,597,468 vs 21,037,931 ms）。

`run_oracle` 的 docstring 本來就自陳是**貪婪**構造、「不保證全域最優」、「是下界」，
所以這不牴觸任何東西——但**論文引用 headroom 時要寫「至少」**，而且這一段的值
由 8.49% 變成 **≥ 11.47%**（差 35%）。細節見 `PAPER_DELTAS.md` B7。

### 兩個超參數會互相作用（不要只調一個）

| W | 丟棄計價 | 拿到 oracle headroom |
|---|---|---|
| 1× (17,117) | tail | 10.0% |
| 1× | block | 3.7% |
| 35× (599,095) | tail | **−28.5%**（比 baseline 還差） |
| 35× | block | **15.1%（最好）** |

W 拉長會讓 $\hat p$ 出現中間質量，用 tail 計價的門檻就把大量 block 推進 SSD 階
（827 MiB/s 的寫入、SSD 讀比重算貴）→ 崩掉；改成逐 block 計價才救得回來。
**「視窗長度」與「邊際成本怎麼算」必須一起選**。

### 失敗與異常（五個，全部有修）

**1. 「取樣以 block 為單位」被讀反，正樣本率差 100 倍。**
第一版把 §5.2 的那句話實作成「每個 block 最多收 C 個樣本」（第 j 次存取以
機率 min(1, C/j) 收）。結果 toolagent 的正樣本率只有 **0.3%**，而真值是 34.8%——
因為重用幾乎全部集中在少數極熱的 block 上，這個上限正好把它們削掉約 600 倍。
**抓到的方法不是讀程式**：把它跟 M4 實測的 `cpu_lru` GPU 命中率（34.9%）一比就對不上。
修法：改成逐次存取以固定機率取樣（`--sample-rate 0.25`），與策略實際面對的分布一致
（每次存取都會重新 admit，所以每次存取最終都對應一次驅逐決策）。

**2. trace 結尾的假負樣本讓 AUC 看起來是 0.9998。**
重放結束時把待標記佇列全部標成「> W」，等於把「視窗還沒觀察完」當成「沒有被重用」。
時序切分下測試集正好是尾巴那段，於是測試集的正樣本率被壓到 1.3%，指標全部漂亮。
修法：只保留 `t + W < 總存取數`（視窗完整可觀測）的樣本，並回報丟掉幾個。

**3. 切片的單位檢查誤報。**
`check_trace_units` 拿模擬器的長度中位數比對**整個檔案**的 `input_length` 中位數。
評估段是後 30%，conversation 那一段的中位數 6,328 vs 全檔 6,909（差 8% > 5% 容忍），
被判為「hash_id 粒度沒展開」而中止。修法：單位檢查對**整條** trace 做，切片只做其餘檢查。

**4. 不變量擋下了一個不是錯誤的結果。**
`diag_true_ordering`（完美資訊）贏過 M4 的 oracle，`check_oracle_dominates` 中止。
逐項檢查後認定**不是記帳錯**：命中守恆與強制未命中下限都通過，容量上限也都有守住；
而 `m4_oracle.run_oracle` 的 docstring 本來就寫著它是**貪婪**構造、
「不保證全域最優」、「是下界的 Oracle」。兩個離線構造比較，輸贏是量測不是錯誤。
處置：`diag_*` 不參與該檢查但改為**明確回報差距**；`tiara_*`（真正的線上策略）
仍然受檢查——放寬的只有「離線 vs 離線」那一格。

**5. `write_csv` 在 append 模式下不檢查檔頭。**
新增兩個欄位（`spearman_*`）之後 append，新列會照新順序寫進舊檔頭底下，
整排錯位而檔案看起來完全正常。已加守門：欄位與既有檔頭不同就中止，
並要求把舊檔移到 `results/superseded/`。
（`results/superseded/policy_sim_pre_dropcost_schema.csv` 就是這樣來的，
其內容與新檔的對應列完全相同，模擬是決定性的。）

### 與論文假設的差異（六項，前四項已寫進 `PAPER_DELTAS.md` B5–B7）

1. **§5.2 的預測目標與策略實際消費的量不同。** 論文迴歸 $\log\tau$、§B.6 用校準與
   成本加權錯誤診斷；但驅逐吃的是**排序**。toolagent 上 AUC 0.9995 而正樣本內的
   Spearman 只有 0.233；把排序換成真值，總時間掉 9.6%。**現有的兩個診斷測不到這件事。**
2. **§5.3 的可證偽預測方向相反。** 加權訓練的增益應隨 $\kappa$ 擴大——實測是隨 $\kappa$
   **惡化**（−2.1% @ κ=12 → −428% @ κ=60）。門檻移動（Elkan）本身值 80%，
   加權在其上是 −25%。依論文自訂的判準，主張應縮減至門檻校準。
3. **§5.2 引 LRB 的「64 個候選 30 μs」不能直接沿用。** 本機實測 **173 μs**
   （k=1 為 30 μs、k=256 為 584 μs，約「30 μs 固定 + 2.3 μs/候選」，單執行緒、
   LightGBM 4.7 的 Python API）。論證仍成立（2.7 μs/候選 vs 單次重算 3,546 μs
   ＝ 0.08%），但數字要用自己量的。訓練成本同理：1.88M 樣本 4 秒，不是 LRB 的 300 ms。
4. **M4 的 headroom 是下界。** 見 `PAPER_DELTAS.md` B7。
5. **特徵只有三族。** `pooled key/value` 與 `attn_mass` 要跑模型才有，
   表 15 的 (C) 區塊（特徵集消融）**做不到**，本檔的數字不得拿去填。
6. **評分時機與演算法 1 不同。** 論文在**驅逐時**對取樣的 64 個候選評分；
   本檔在**每次存取時**評分一次並沿用到下次存取（12.7M 次存取一次批次推論 5 秒；
   逐次驅逐呼叫模型在 Python 端要數小時）。代價是看不到「已經多久沒被用了」，
   預測略偏保留。要驗證這個近似的影響，得實作驅逐時評分的模式。

### 下一步（依價值排序）

1. **重跑 M4 的 SSD 掃描並修目的地規則**（發現 3）。現在的 headroom 全部是被
   自己的 SSD 規則壓低的下界；`--ssd-gib 0` 的 oracle 就比 512 GiB 的快 3.3%。
   這會動到論文的主結果，優先做。
2. **換掉設限的迴歸目標**（發現 1）。策略吃的是排序，而 $\log\tau$ 設限在 $W$
   會把 $W$ 之外的一切壓成同分。可行的方向：兩段式（先分類「還會不會再用到」，
   再對會用到的迴歸），或直接上 §5.3 的 (L2) 排序損失。
3. **`--score-at eviction`**：實作演算法 1 的原始評分時機（驅逐時對 64 個候選評分），
   量本檔那個近似的代價。
4. 需要一條**預測會失手**的工作負載才能檢驗 §5.3 的 (B) 區塊——Mooncake 在 W=1× 下
   太好預測（AUC 0.9995），在 8K 以上的位置更是一個正樣本都沒有。
