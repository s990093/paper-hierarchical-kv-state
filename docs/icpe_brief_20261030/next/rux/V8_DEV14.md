# v8_dev14：v8＋PREREG 偏離 14 的規則（icpe-rux，2026-11-07）

## 問題／要證明什麼
- 偏離 14 的修正規則（O-PF 預取副本過期就丟、模型 P 已經過去的時間要重抽），再加上 SIM_CHANGES 23（預取到 GPU 時騰完空間再查候選），之前只在 icpe-sim 的 x1fast 裡。
- x1fast 沒有 APC／BELP，v8 沒有這些規則，所以沒有任何一個引擎能同時跑「雜訊＋偏離 14＋APC／BELP」。
- 卡住的工作：
  - 修正版版本 C 的 O-PF／O-ALL 列（主表、chat、agent）；
  - 版本 C 的 P4 加跑（5 組）。
- 使用者在 icpe-main 選「先移植 v8，再做二階排隊（建議）」（11/7 約 15:4x，AskUserQuestion；icpe-main 轉達）。

## 做了什麼（`code/icpe/rux_x1fast/v8_dev14/`）
- `make_v8_dev14.py` 從 v8 的 `main.rs`（48e4efce）產生 `src/main.rs`。
  - 每個替換都必須剛好出現一次。
  - v8 一行不改，因為偏離 10–12 登記的是 48e4efce。
- 來源：`origin/icpe-next-sim` 的 x1fast，63b8e328 對 4a4baa48 的差異（40a4c87f 驗證過；SIM_CHANGES 22、23）。v8 的 x1fast 基底就是 4a4baa48 那一支，所以各段幾乎原樣搬過來。

| jobs.txt 欄位 | 意義 | CSV 欄（開了才有） |
|:--|:--|:--|
| `pfx=1` | 偏離 14 (i)：每一步開頭（GPU TTL 到期、推進預取之前），預測被讀的請求已經過去（st_t ≤ cur）的 session，丟掉全部預取副本、不再排回、不算 n_unpf。跳過已丟（S_DROP）的 session。只在 pf＝opf 時開 | `pf_expire` |
| `nzo=resample` | 偏離 14 (a)：模型 P 抽到「上一輪結束＋g ≤ 現在」→ 從排序後的經驗間隔、bisect_right(已經過的時間) 之後均勻重抽（同一條亂數）；B1 存回。重抽用亂數，所以 B1 也照 CPython 3.12 的 set 順序（PySet） | `noise_overdue` |
| `pgr=1` | SIM_CHANGES 23：預取到 GPU 時，`g_make_room` 之後候選已經不在 CPU／慢層（O-ALL 的 O-NOW 騰空間時把它丟了）→ 不預取、不排回 | `pf_gpu_recheck` |

- sweep：`sweep_v8_dev14.py`（先 import `sweep_v8.py`）。
  - 本分支的 x1_sim 沒有 icpe-sim 的三個命令列參數，所以用環境變數開：`RUX_PFX=1`、`RUX_NZO=resample`、`RUX_PGR=1`。
  - 旗標寫進每一個 job，CSV 欄名和 icpe-sim 相同。這段放在 `dev14_flags.py`，v9 的 `sweep_v9.py` 也用它。
- **APC／BELP 和這些規則的交互作用**（沒有 Python 參考，照實寫）：
  - v8 的 `parse_spec` 不准 APC／BELP 和「預取進 GPU」同時開。所以 `pf_expire` 的 GPU 路徑（`g_unprefetch`）和 `pgr` 都碰不到 APC 的部分 GPU 副本、`gn`、`ref_gc`。
  - APC／BELP 加 `pfx=1` 只走 CPU 路徑，程式和 x1fast 相同。
  - BELP 經 `g_next` → `nz()` 讀雜訊，所以 `nzo=resample` 也作用在 BELP；有雜訊時 `belp_fast` 本來就關。
  - icpe-sim 問的「APC／BELP 從尾端逐出時，`g_make_room` 會不會像 O-ALL 那樣把候選丟掉」：只發生在預取進 GPU。APC／BELP 不能預取進 GPU，所以不會。

## 驗證（`smoke_v8_dev14.py`；SemiAnalysis N16 s1；CPU 16、111.0625 GiB × CPU 17 個規則）
| 測試 | 內容 | 結果 |
|:--|:--|:--|
| D1 | 旗標全關 → 和 v8（48e4efce）逐位元相同。GPU 規則 LRU／BEL／TTLA:10／APC／BELP；雜訊無、P B1、M B2（λ 0.3）；預取到 CPU，O-PF／O-ALL／LRU+RPF 另跑預取到 GPU | 見下方 |
| D2 | 不開 APC／BELP、開旗標 → 和 icpe-sim 的 x1fast 63b8e328（用 git 編）逐位元相同。旗標組合：三個全開、只開 pfx、只開 nzo（模型 P）、只開 pgr（預取到 GPU）。GPU 關／LRU／BEL／TTLA:10；④ 的雜訊 P／M × B1／B2 × λ∈{0, 0.3}，其他規則雜訊無或 P B1 | 見下方 |
| D3 | 開 APC／BELP、雜訊（無、P B1、P B2）、三個旗標全開；`X1FAST_APC_WHOLE=1`（一次逐出整個 session）→ APC＝GPU LRU、BELP＝GPU BEL。逐塊逐出時開 APC 引用數稽核（`X1FAST_V8_CHECK=200`），要能跑完 | 見下方 |
| D4 | （只報告）開旗標和不開的差：確認旗標真的有作用 | 見下方 |

**結果**（wb：run `20261107-155514-icpe-rux-d14-smoke-full`；O-ALL 重跑：run `20261107-161328-icpe-rux-d14-smoke-oall`；wt：run `20261107-161521-icpe-rux-d14-smoke-wt`）
- D1：564／564 組和 v8 48e4efce 逐位元相同。
- D2：1,908 組和 x1fast 63b8e328 逐位元相同。另外 2 組兩邊都斷言，panic 訊息相同（`gspf 363 56`、`gspf 911 4`），所以行為相同：
  - 條件：O-ALL、預取到 GPU、**只開 pfx、沒開 pgr**、雜訊 P B2 λ 0.3、CPU 16 GiB、GPU BEL／TTLA:10。
  - 這是參考版本身的行為；三個旗標全開時同一格可以跑完。
- D3：APC＝LRU、BELP＝BEL（一次逐出整個 session，含雜訊與三個旗標）204／204；逐塊逐出加稽核 204／204 跑完。
- D4（只報告）：
  - 開旗標後，O-PF、O-ALL 每一組都變。
  - 其他規則只有在 GPU 用 BEL 而且有雜訊 P 時會變（`nzo=resample` 經 GPU BEL 的 `nz()` 作用），符合 icpe-sim 說的「沒有雜訊、模型 M、λ 0 時 opf 以外不變」。
- wt：只跑預取到 CPU。wt 加預取到 GPU 在旗標全關時就會斷言，使用者決定不修（icpe-sim STATUS_SIM 15:56；正式區塊 0／465 用到）。結果：D1、D2、D3 共 2,414 項全部通過，沒有斷言。
- sweep 端到端（run `20261107-162547-icpe-rux-d14-sweep-e2e`）：
  - `RUX_PFX=1 RUX_PGR=1` 時每列多 `pf_expire`、`pf_gpu_recheck`，`py_fallback_jobs=0`。
  - 和沒開比：LRU、O-BEL 的 32 列逐列相同；O-PF、O-ALL 的 32 列都變。

**版本**
- `v8_dev14/src/main.rs` sha256 `d2048fa38a42c78b316a97734101cb094854cc584be92ef5ec9c8717c6ad5e88`。
- sweep 的 src sha（Cargo.toml＋main.rs，`x1_rust_v6.src_sha`）`4eca5bda266e007b68a0fbf60cdb5fbb4d0d0a693bfcca5b4566e3fb87eb6db9`。CSV 的 code_sha 是 `3280b1c8799b+rs4eca5bda`。
- 參考的 x1fast 63b8e328 `main.rs` sha256 `2803b0c0922ea280…`。
- 旗標名稱和 x1fast 相同：`pfx=1`、`nzo=resample`、`pgr=1`。sweep 用環境變數開：`RUX_PFX=1`、`RUX_NZO=resample`、`RUX_PGR=1`。

- 試跑時看到：fa3686b0 的參考版在 O-ALL、預取到 GPU、`pfx=1` 時會 `gspf` 斷言，和我的移植一樣（run `20261107-154707-icpe-rux-d14-smoke-quick`）。這就是 63b8e328 用 S_DROP 跳過和 `pgr` 修掉的問題。之後改用 63b8e328 當參考。
