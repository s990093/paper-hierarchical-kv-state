# Rust 引擎：怎麼寫、為什麼這樣寫、踩過的坑

> 目標：把參考模擬器（Python，`x1_sim.py` 的 `Engine`）改寫成 Rust，**輸出逐位元相同**，越快越好。
> 三個版本都在 `engine/`：`v1_port`（直接移植）→ `v2_nextreq_cache`（「下次回來」快取）→ `v3_prefix_fork`（共用前段再分歧）。
> 數字都有 run_id（目錄在 acvlab 的 `/ssd6/hungwei/icpe-next/runs/<run_id>/`）。

## 1　為什麼是 Rust（先試過的都沒用）

剖析（Python cProfile）顯示時間幾乎全在「每個請求逐塊做整數加減與查表」的迴圈：`_step_gpu`、`_turn_end_wb`、`_fix_g` 一個 job 被呼叫上百萬次。這種程式只有真正的編譯碼才快得起來。

| 方法 | 倍數（同一組 job，CPU 時間） | 為什麼 |
|:--|--:|:--|
| PyPy 3.11 | 2.0× | JIT 對 Python 物件迴圈有幫助，但程式裡有 numpy 逐元素存取，PyPy 跑 numpy 慢 |
| CPython 3.14 | 0.87× | 直譯器本身沒有幫助（這次量到略慢；機器 load 約 90，單次量測有雜訊） |
| Cython（不改程式直接編） | 0.88× | 沒有型別，仍然是 Python 物件運算 |
| mypyc | 0.76× | 原樣編不過（執行中替換方法 `self._fix = self._fix_g`、事後加屬性 `v.nG`）；改成屬性後能跑但沒型別 → 更慢 |
| AVX-512／AMX／GPU | 不適用 | 每一步依賴上一步的分支，沒有可向量化的大塊運算；acvlab 的 Xeon Gold 6248R 有 AVX-512、沒有 AMX |
| **Rust 全移植** | **31–35×（單執行緒中位數）** | 連續陣列、沒有物件開銷 |

（直譯器那組：SemiAnalysis N16 s1、CPU 64 GiB 的 4 個 job，各跑一次；run `20261105-232417-icpe-ana-bench-sa-cpy312` 等 5 個；現行 CPython 4 個 job 合計 178.5 CPU 秒。全部和現行版逐位元相同。）

## 2　逐位元相同的規則

- **同樣的浮點運算、同樣的順序**。Python 的 `a + b - c` 是 `(a + b) - c`；`int / float` 先把 int 轉成 double（值 < 2⁵³ 時精確）。照抄順序。
- **不要讓編譯器融合 FMA**：Rust 預設**不會**把 `a*b+c` 合成 FMA（RFC 3514）；所以 `-C target-cpu=native` 不改結果（實測輸出相同）。不要用 `mul_add`、fast-math、平行 reduce。
- **平手時挑誰**：Python 的 `min(set, key=…)` 在平手時取 set 走訪順序的第一個。我們檢查過每一個挑選的鍵都含 `last[x]`（每個 session 不同）或 session id，所以沒有平手 → 走訪順序不影響結果，Rust 可以用任何集合結構。
- **heap 的順序**：Python `heapq` 比的是整個 tuple；Rust `BinaryHeap<Reverse<(F, i64, usize)>>` 用同樣的 tuple 字典序（f64 包一層全序 `F`，沒有 NaN）。
- **`t[-1]`**：Python 的 `self.t[self.cur]` 在 `cur = -1` 時取最後一個；Rust 寫成 `tt(i)` 處理負索引。
- **`int(cap)`、`C`**：CPU 容量 `C` 是 float（`64.0 × GiB`），比較 `cpu_used > C` 時 Python 是 int 對 float 的精確比較；Rust 轉成 f64 比（< 2⁵³ 精確）。

## 3　資料結構對照

| Python | Rust |
|:--|:--|
| 每個 block 的 list（refcount、格式、在不在慢層…） | `Vec<i64>`／`Vec<u8>`／`Vec<bool>`，用 block id 直接索引 |
| `full[r]`（每個請求的 block 清單） | CSR：`req_off` + `req_blk: Vec<u32>`；session 的 block 清單存「請求 index」，不複製 |
| `acc[b]`（block 被讀的請求序號）＋`bisect` | CSR：`acc_off` + `acc_req`，`partition_point` |
| `OrderedDict`（LRU、ARC 的 T1/T2/B1/B2、S3-FIFO） | 自己寫的侵入式雙向串列 `OList`（`prev`／`next`／`inn`／`val`／`len`） |
| `set`（`cpu_set`、`pf_sessions`） | `ISet`：陣列＋位置表，O(1) 加入／移除（swap-remove） |
| `heapq` | `BinaryHeap<Reverse<…>>` |
| 執行中替換方法（`self._fix = self._fix_g`） | 一個 `fix()`，用布林旗標選分支 |

## 4　加速技巧（每一項都重新做過逐位元對拍）

| 技巧 | 版本 | 效果 | 備註 |
|:--|:--|:--|:--|
| 直接移植 | v1 | 單執行緒中位數 31–35×（Bailian 35.0×、Mooncake 30.9×） | 最慢的是預取類 job（O-PF、O-ALL） |
| 編譯設定 `lto="fat"`、`codegen-units=1`、`panic="abort"` | — | +1% | 不改語意 |
| `-C target-cpu=native` | — | +8% | 每台機器各自編、各自對拍（AMD 要另外驗） |
| 多執行緒共用一份唯讀紀錄（`std::thread::scope`＋原子計數的 job 佇列） | v1 起 | 16 執行緒 11.5× | 每個 job 獨立，結果和單執行緒相同；執行緒多了每個 job 變慢約 35%（記憶體頻寬） |
| 只輸出摘要（`X1FAST_SUMMARY_ONLY=1`） | v1 起 | 免寫幾十 GB 的逐請求陣列 | 摘要的平均是依序相加，和 numpy 的成對相加可能差幾個 ulp；只用在篩選實驗 |
| **「下次回來」快取** | v2 | 預取類 job 最慢的從 4.7× 提到約 24×（見 results） | 精確：session 的「下次回來」只取決於它自己的 block 清單與這些 block 的 refcount，以及目前請求序號。共用 block 的 session 用 union-find 分成一組；組內任何 refcount 變動 → epoch +1 → 快取失效；同組沒變動而且目前請求序號 < 快取值時，答案不變 |
| 合併時間設定（照正式的 job 切法） | 規劃器 | 整個 block 的 CPU 只少 4% | 正式網格裡約 94% 的 job 是 TTL 或預取，本來就只能一個時間設定一個 job |
| **共用前段再分歧** | v3 | 見 results | 沒有回合結束 hook、沒有預取的 12 個 CPU 規則，在第一次要擠 CPU 前完全一樣。第 1 趟找出第一次要擠的請求 r*；第 2 趟共用跑到 r* 之前（同時維護 LRU／ARC／S3／GDSF 四種順序結構），再 `clone()` 給每個規則跑完 |

**沒做、但查過的**：PGO（cargo-pgo，官方說約 10%）、mimalloc／jemalloc、bitset、預先算好每次存取的「下次存取」陣列（libCacheSim 的 oracleGeneral 格式）、以下次存取排序的 heap＋延遲刪除（libCacheSim 的 Belady、cachekit 的 OPT 設計、gem5 的 opt_cpu）、NUMA 綁定（`numactl --cpunodebind --membind`）。

## 5　踩過的坑

1. **對拍腳本的解析錯誤**：v2 多輸出一行 `all_jobs_wall_s … threads …`，腳本用 `k, v = line.split()` 讀檔就掛了 → 對拍其實沒比到。修好後重跑。**教訓：對拍失敗要先確定是「程式錯」還是「比對的工具錯」。**
2. **mypyc 的兩個限制**：不能在執行中替換方法、不能事後加屬性。
3. **抽樣 session 反而更貴**：容量同比例縮小後擠出次數變多，抽 25% 的 session 花了完整版 59% 的時間。
4. **perf 不能用**：`perf_event_paranoid=4`、沒有 sudo → 改用 cProfile（Python 端）和自己計時。
5. **正式 CSV 只存到小數 6 位**：Rust 的平均和正式 sweep 比對時要在同一精度下比。

## 6　怎麼用

```bash
# 編譯（每台機器各自編）
cargo build --profile fast            # Cargo.toml 的 [profile.fast]
RUSTFLAGS="-C target-cpu=native" cargo build --profile fast
# 執行：x1fast <trace_dir> <run_dir>；run_dir 裡放 views.txt、jobs.txt、f*.f64、fc*.f64、fc_idle.f64、i4.txt
X1FAST_THREADS=16 X1FAST_SUMMARY_ONLY=1 X1FAST_FORK=1 ./x1fast3 <trace_dir> <run_dir>
```
`harness/` 的 `difftest.py`（逐位元對拍）、`grid.py`（照正式切法跑整個 block）、`lowfi.py`（低保真度版本）、`variants.py`（編譯版本／執行緒跑分）會產生這些輸入檔。
