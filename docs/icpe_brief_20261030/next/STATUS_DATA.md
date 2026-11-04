# STATUS_DATA（icpe-data，acvlab）

> 只由 `icpe-data` 寫。格式照 NEXT_PLAN_20261104 §3b：日期時間／做完什麼（commit、run_id）／卡住什麼（完整錯誤）／需要 icpe-sim 做什麼／需要使用者決定什麼。最新的一節在最上面。

## 2026-11-05T00:15+08:00 — 使用者決定排程；請 icpe-sim 加速 Engine

**使用者 11/5 的決定**（AskUserQuestion，這個 session）
1. 排程：**G1 一過就先跑 CacheWise 主結果·主時間設定**（12 個變體，約 21 h）；**同時請 icpe-sim 加速 Engine**（結果要逐值相同、用 G1 驗）；CacheWise 跑完接 **TraceLab c1 → c2 → c4**；加速版通過驗證就換用。其餘照順序接在後面（給 CCGrid）。
2. Bailian traceB：**跳過**，論文寫明「traceB 沒有多輪結構（172,800 個請求全部 turn＝1、parent＝−1）」。Bailian 只跑 traceA、thinking、coder。

**需要 icpe-sim 做什麼**（使用者決定的）：**加速 `x1_sim` 的 Engine**，條件是新功能關閉時 G1 逐值相同、開啟時和 ac0c56e（或 G1 通過的那個 commit）逐值相同。量測基準：TraceLab c1、CPU 64、GPU 111.0625 wb，LRU／O-BEL／O-PF × GPU LRU／TTLA:10 每個 job 200–304 s、每個 worker 4.75 GB（run `20261104-235459-icpe-data-trial-tl-timing`，可直接用 `/ssd6/hungwei/icpe-next/trial/x1_nb/tracelab_c1.npz`）。通過後在 STATUS_SIM 寫 commit 與驗證 run_id，我 merge 後再做一次 G0 再換。

**正式跑的輸出**：原始輸出（sweep CSV、逐請求 TTFT 的 npz——`x1_boot.py` 的成對 bootstrap 要用）在 `/ssd6/hungwei/icpe-next/runs/<run_id>/`；git 的 `results/icpe_next/b/` 只放彙總（sweep CSV 加 `host=acvlab` 欄、xz 壓縮，和分析表）。npz 每個 job：CacheWise 約 0.4 MB、TraceLab 約 4.4 MB（TraceLab 主時間設定 3 個變體約 240 GB；/ssd6 剩 2.7 TB）。

## 2026-11-05T00:10+08:00 — 試跑計時完成；全部正式格子的 job 數與預估總時間

**試跑計時**（全部用 icpe-next-sim 6b17191 的 git archive、GPU 層 111.0625 GiB wb；輸出不進 results/；機器 load 65–87／96）
| run | 紀錄 | 內容 | 每個 job（秒） | 每個 worker RSS |
|:--|:--|:--|:--|:--|
| `20261104-235049-icpe-data-trial-cw-timing` | CacheWise N16 s1（943 萬 chunk） | CPU 64；LRU、GDSF、O-BEL、O-DROP、O-PF、O-ALL、LRU+RPF × GPU LRU／BEL／TTLA:10／TTLB:10；16 視角 LRU、O-BEL | 7.6–13.7（TTLA:10 約 12.5；16 視角 8.0–9.1） | 0.44 GB |
| `20261104-235858-icpe-data-trial-cw-timing2` | 同上 | CPU 8、1024；LRU2、PENS、ARC、S3FIFO、O-NOW、O-ALL-H3600 × GPU LRU／TTLA:0.3 | GPU LRU 7.6–9.5；**TTLA:0.3 15.8–28.1** | 0.44 GB |
| `20261104-235922-icpe-data-trial-cw-timing3` | 同上 | CPU 1024；O-BEL、O-BELS × GPU LRU／BEL | 7.1–9.0 | 0.44 GB |
| `20261104-235459-icpe-data-trial-tl-timing` | TraceLab c1（2.23 億 chunk、665,453 請求） | CPU 64；LRU、O-BEL、O-PF × GPU LRU／TTLA:10 | **200–304**（中位約 250）；載入約 3.5 分鐘（舊版純 Python x1_nb；icpe-sim 3140f3c 已向量化） | **4.75 GB**（parent 峰值 6.8 GB） |

每個 chunk 存取的成本兩份紀錄幾乎相同：CacheWise 11 s ÷ 943 萬＝1.17 µs，TraceLab 250 s ÷ 2.23 億＝1.12 µs〔算術〕→ SemiAnalysis（4,221 萬 chunk）用 1.15 µs/chunk 推算＝48.5 s/job（**推算，沒量**）。

**job 數**（6b17191 的 `jobs_for` 數的；每個紀錄變體）：主結果（wb、預取到 CPU）全部時間設定＝49,014（分三批跑，結果和一次跑相同：① 主時間設定 cake×{ssd,b32,nfs}×{閒,忙}＝**18,342**、② nvme807×cake＝6,318、③ load×4 慢層＝24,354）；`wt` 再 49,014；預取進 GPU 11,520（其中主時間設定 4,320）。

**預估總時間**〔算術：job 數 × 每 job 秒 ÷ 32 worker；每 job 秒的來源見最後一欄〕
| 紀錄 | 變體 | 每 job 秒 | 主·主時間設定 | 主·全部時間設定 | wt | 預取進 GPU | 前綴＝快取（CW／TL） | 每 job 秒的來源 |
|:--|--:|--:|--:|--:|--:|--:|--:|:--|
| CacheWise | 12 | 11.0 | 21.0 h | 56.2 h | 56.2 h | 13.2 h | 56.2 h | 試跑實測 7.6–28.1 s，加權估 11 |
| TraceLab | 3 | 250.0 | 119.4 h | 319.1 h | 319.1 h | 75.0 h | 319.1 h | 試跑實測 200–304 s（CPU 64；TTLA:0.3 沒量） |
| SemiAnalysis | 12 | 48.5 | 92.7 h | 247.8 h | 247.8 h | 58.3 h | — | 推算：1.15 µs/chunk |
| Bailian（traceA、thinking、coder） | 3 | 2.0 | 1.0 h | 2.6 h | 2.6 h | 0.6 h | — | 推算（沒量） |
| Mooncake | 4 | 20.7 | 13.2 h | 35.2 h | 35.2 h | 8.3 h | — | G0 舊程式實測（沒開 GPU 層） |
| FreeInference（6 週） | 6 | 5.0 | 4.8 h | 12.8 h | 12.8 h | 3.0 h | — | H200 T1／T5 每 job 中位 2.3–4.1 s（別台） |
| **合計** | | | **252 h** | **674 h** | 674 h | 158 h | 375 h | |

- 第一批全部（主＋wt＋預取進 GPU＋快取前綴）≈ **1,881 h ≈ 78 天**（32 worker）。還沒算：截斷 128K 的敏感度（每份紀錄再一份主結果）、第二批（慢層容量、block 粒度 APC 重算、排隊、W4 雜訊）＝NOT_ESTIMATED（程式還沒有）。
- ICPE 要的：CacheWise 主·主時間設定 **21 h**；TraceLab 主·主時間設定 **119 h**（只跑 c1＝原始到達 **40 h**）。從 11/5 00:10 到 11/9 20:00 約 116 h。
- 記憶體：TraceLab 32 個 worker × 4.75 GB ≈ 152 GB（這台可用約 324 GB、swap 已滿）。

**需要 icpe-sim 做什麼**：你提過可以加速 Engine（結果要逐值相同）——TraceLab 的單一 job 是 4 分鐘級、全部格子是 78 天級，這是關鍵路徑。要不要做、先做哪個由使用者決定（我會問）。

**需要使用者決定什麼**：ICPE 前跑不完的部分怎麼取捨（見下一個問題）；Bailian traceB。

## 2026-11-05T00:00+08:00 — G0 通過；Bailian 兩個新發現

**做完什麼**
- **G0 PASS**：run `20261104-232902-icpe-data-g0-main`（程式＝`origin/main` 77c25f6 的 `x1_sim.py`／`p3_fvar.py`，code_sha `c5c5b1cd1b67`；共用 venv；`X1_TRACE_DIR=/ssd8/hungwei/paper-hkv/datasets/mooncake/traces`；另需 `M7_RES=<repo>/results/m7_write_policy_h200`；16 worker、1,722 s、exit 0）。和 `results/icpe_h200/t1/p01/sim_V1/x1/x1_sweep.csv.gz` 比：**4,416 列 × 29 欄＝128,064 格，0 個不同**（`code/icpe/next_b_g0.py`：去掉 run_id、ts、elapsed_s、code_sha，依 7 個鍵排序逐字串比；輸出 `…/g0_compare.txt`）。→ 環境（Python 3.12.15／numpy 2.3.5／pandas 3.0.5／pyarrow 25.0.1）與 main 的程式重現 H200 T1 V1，不用再拿 6ff7584 重跑（main 的 x1_sim.py 比 6ff7584 多的 +84 行在預設下不影響輸出）。
- Bailian（commit `b916bbd`）：
  - **hash_ids 是逐塊 hash，不是鏈式**（上游 README：每個 16-token 塊各自做 salted SipHash-2-4）。traceA 的鏈式檢查違反 301,710 次、traceB 893,399 次（coder、thinking 0 次）。→ chunk 鍵改成「從頭到這一塊的整段 hash 序列」的編號（和 SemiAnalysis 相同）；鏈式違反數只記錄。改完 traceA 的規則 6 全過（parent 前綴 1.0000、間隔÷輸出 0.29 s/token）。
  - **traceB 沒有任何多輪結構**：172,800 個請求全部 turn＝1、parent_chat_id＝−1（type 87% 是 `api`）→ 依資料定義沒有「回來的請求」，主指標無定義；載入函式的 parent 前綴檢查 0/0 → 停（run `20261104-235631-icpe-data-conv-bailian`，錯誤：`[bailian-traceB 前導共用塊 ≥ parent 的完整塊] 規則 6 的驗算比例 nan < 0.9 → 停`）。traceA、thinking、coder 已轉好。
- 轉好的紀錄（`/ssd6/hungwei/icpe-next/datasets/x1_nb/`，不進 git）：CacheWise 24 份（N 16／32／64／72 × seed 1／2／3 × 前綴 struct／cache；每份 56,825 請求、540 session、943 萬 chunk；N16 跨 364–416 h、N64／72 約 200–215 h）、Bailian traceA／thinking／coder、TraceLab 與 SemiAnalysis 轉換中。

**需要使用者決定什麼**：Bailian traceB（沒有多輪）要跳過，還是像 Mooncake 那樣用 hash 前綴重建 session？（我會和總時間一起問。）

## 2026-11-04T23:55+08:00 — 資料清點、下載、載入函式與轉換、資料陷阱、使用者決定、試跑計時（G0 還在跑）

**做完什麼**

1. **資料清點**（這台既有的都在 `/ssd8/hungwei/paper-hkv/datasets/`，`/ssd6/hungwei/icpe-next/datasets/` 用 symlink 指過去；/ssd8 剩 299G、96% 滿，所以新下載的放 /ssd6）

| 紀錄 | 位置 | 版本 | 完整性（今天重算） |
|:--|:--|:--|:--|
| Mooncake | /ssd8（symlink） | commit d750cdda | conversation `b8cbb061…`、toolagent `48a2db1a…`＝README.txt＝H200 runner 檢查的值 |
| CacheWise | /ssd8（symlink） | HEAD 181c435a（`git rev-parse` 確認；LFS 檔 0 個，資料直接在 git 裡） | git 物件雜湊；工作樹乾淨 |
| Qwen-Bailian | /ssd8（symlink） | 5f7439c5 | 4 個 jsonl 的 sha256＝README.txt（＝下載時的 LFS oid） |
| FreeInference | /ssd8（symlink；**沒重下載**） | 16d3c04d | 48 個 parquet 的 sha256 全部＝下載 log 裡對過 HF LFS oid 的值（`sha256sum -c`，0 個不同） |
| TraceLab | **新下載** /ssd6 | release v0.0.2 | sha256 `11ce51ec…`＝GitHub release digest（run `20261104-233022-icpe-data-dl-tracelab-semianalysis`，用 `setup_3090_data.sh do_tracelab`） |
| SemiAnalysis | **新下載** /ssd6 | HF 23f152f6 | `hf_verify … files=3 bad=0`（同一個 run，`do_semianalysis`） |

2. **載入函式與轉換**（commit `3641092`）：`code/icpe/agent_traces.py` 加 `load_cachewise()`、`load_bailian()`（規則 6 斷言寫在函式裡），`load_tracelab()` 補 `out／provider／trigger` 欄（原欄位不變）；`code/icpe/next_b_convert.py` 轉成 icpe-sim 的 `nb` 格式（`code/p3x/x1_nb.py`，icpe-sim 6b17191；我照用，不要求改）；`code/icpe/next_b_g0.py` 對拍程式。轉好的 npz 在 `/ssd6/hungwei/icpe-next/datasets/x1_nb/`（**不進 git**）。
   - 規則 6 驗算（CacheWise，全部過）：同一 requestId 的輸入三欄相同 0.9991；output_tokens 不減 1.0000；cache_read(k) ≤ inp(k−1)+out(k−1) 0.9927；間隔 <10 s 時 cache_read(k) ≥ 0.9×inp(k−1) 0.9857；(end−start)/out 中位數 31.7 ms/token。Bailian coder：len(hash_ids)＝ceil(input/16) 1.0000；鏈式前綴違反 0；前導共用塊 ≥ parent 完整塊 0.9931；間隔÷上一輪輸出 0.199 s/token。TraceLab：prefix+new＝input 1.0000。

3. **資料陷阱（§5.7 以外新發現的）**
   - **供應商快取會過期**：TraceLab 的 `prefix_tokens`（Claude 列＝`cache_read`，1.0000 相同）和 CacheWise 的 `cache_read` 都是「供應商快取命中」，5 分鐘／1 小時沒用就掉成 0。間隔 ≥1 h 時 `prefix ÷ 上一輪輸入` 中位數：TraceLab Claude 0.022、Codex 0.059、CacheWise 0.053；但 `這一輪輸入 ÷ (上一輪輸入＋輸出)` 中位數都 ≈1.00（上下文是接續的）。間隔 <60 s 兩者一致（中位數 1.000）。→ 照 NEXT_PLAN §5.6 字面用快取命中，長間隔回來的請求會被當成「沒有可還原的前綴」。
   - **resume 複本**：Claude Code resume 會把舊對話整段複製進新檔。CacheWise：25 對 session 的前段請求序列（inp, cache_read）完全相同、長 18–356 個請求，**1,885／11,365 個請求（16.6%）是複本**；複本那段的時間戳有一部分被改成 resume 當下的時間。TraceLab：**14,685／665,453（2.2%）**屬於 5,833 組時間到毫秒都相同的複本（全是 Claude，129 個 session）。上游作者的處理：CacheWise `extract_events.py` 不去重；TraceLab 只用 `(provider, session_id, round_id)` 去重（換了 session_id 的複本不會被去掉）。
   - **CacheWise 可用對話＝80**（不是 94 減 1）：12 個原始檔是 0 byte，2 個沒有任何 LLM 呼叫（session_0061 只有 system 節點）。108 個 session＝80 主＋28 子 agent。
   - CacheWise 同一個 requestId 會記成好幾個 assistant 節點（串流，每個 content block 一個）；最後一個的 output_tokens 有時仍是佔位值 1（模擬器不用輸出長度，不影響）。順序要用 requestId 序號：用 parent 時間排會排錯 128 個（resume 改寫了時間）。
   - 一個請求的送出時間＝第一個 assistant 節點的 parentUuid 指向的 user／tool_result 節點時間（`llm_call.timestamp` 是回應寫下的時間，不是送出時間）。

4. **使用者 11/4 的決定**（AskUserQuestion，這個 session）
   - 前綴規則：**兩種都跑**——主結果＝結構推得（輸入沒變短、中間沒有 compact_boundary → 上一輪整段輸入都算前綴；否則退回快取命中），敏感度＝快取命中（NEXT_PLAN §5.6 原文）。
   - resume 複本：「不確定的都看類似 paper 怎麼做」→ 上游作者都不做跨 session 去重 → **照釋出的資料用、不去重**，在論文限制寫明 16.6%／2.2% 是複本。
   - CacheWise 複本數：「看其他人怎麼做」→ Continuum 把紀錄重複 1–5 次、PREREG 登記 5 份 → **5 份＝400 個對話**（PREREG §3 D 的「470」是用 94 算的）。
   - CPU：**固定 32 個 worker**（`nice -n 10`）。

5. **試跑計時**（run `20261104-235049-icpe-data-trial-cw-timing`；程式＝icpe-next-sim 6b17191 的 git archive，**沒 merge、不進 results/**）：`nb:cachewise_N16_s1`（56,825 請求、540 session、943 萬 chunk）、CPU 64 GiB、GPU 層 111.0625 GiB wb、16 worker、當時 load ≈ 87／96。每個 job **7.6–13.7 s**（LRU／GDSF／O-BEL／O-DROP／O-PF／O-ALL／LRU+RPF × GPU 的 LRU／BEL／TTLA:10／TTLB:10；TTLA 最慢約 12–13.6 s；16 個時間設定的多視角 job 8.0–9.1 s）；載入約 10 s；每個 process RSS 約 440 MB。**沒涵蓋**：最重的規則（LRU2、PENS、ARC、S3FIFO、O-ALL-H）、其他 CPU 容量。

6. **正式格子的 job 數**（照 PREREG，直接呼叫 6b17191 的 `jobs_for` 數，不是估）：每個紀錄變體（一個 trace × 到達設定）
   - 主結果（wb、預取到 CPU）：**48,258 個 job**＝9 個 CPU 容量（8、16、32、64、128、256、512、1024、111.0625）× 23 個 CPU 規則 × 16 個 GPU 規則（LRU、BEL、TTLA×7、TTLB×7）× 時間設定（慢層 ssd／b32／nfs／nvme807 × GPU 閒／忙 × cake／load）；其中 TTL 類 36,288、預取類 11,520（都是一個時間設定一個 job）。
   - `wt` 敏感度：再 48,258；預取進 GPU 的敏感度：11,520。
   - 紀錄變體數：CacheWise 12（N 16／32／64／72 × seed 1／2／3）× 前綴規則 2；TraceLab 3（壓縮 1／2／4×）× 前綴規則 2；SemiAnalysis 12；Bailian 4；Mooncake 4（conversation／toolagent × R1／R2）；FreeInference 6 週（T1／T4／T5 用的那 6 週；CPU 用 PREREG §4 的清單＝舊網格的聯集＋111.0625）。
   - **推算〔算術，用上面一格的試跑時間，不是量的〕**：CacheWise 主結果（結構前綴、wb、預取到 CPU）12 × 48,258 ≈ 57.9 萬個 job × 約 10 s ≈ 579 萬 CPU-秒 → 32 worker 約 **50 小時**。TraceLab 的試跑計時還在做（轉換中；它有 665,453 請求、2.23 億 chunk，約 CacheWise 的 24 倍），結果出來再補全部總時間。

**卡住什麼**：G0 還在跑（run `20261104-232902-icpe-data-g0-main`，用 main 的程式）。正式模擬要等 icpe-sim 的 G1（STATUS_SIM）＋我 merge 後再做一次 G0。

**需要 icpe-sim 做什麼**：無（nb 格式照用）。

**需要 h200 做什麼**（使用者已在這個 session 決定，請登記 PREREG 偏離）：(1) CacheWise 5 份複本＝**400** 個對話（不是 470；可用對話 80）；(2) TraceLab／CacheWise 的前綴：主結果＝結構推得，快取命中當敏感度；(3) resume 複本不去重（照上游作者）。

**需要使用者決定什麼**：全部格子的總時間出來後（TraceLab 計時），ICPE 前跑不完的部分怎麼取捨，會再問。

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
