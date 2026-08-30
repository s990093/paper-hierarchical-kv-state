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
