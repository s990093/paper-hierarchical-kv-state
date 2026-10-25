"""m10_speed.py — 精度這條線的速度實驗（03_plan_h200.md §2.4 K0、§4、§5.3、§5.4；CSV 欄位照 §6）

子命令：
  calib-f   逐 chunk 重算曲線 f(i)（64 個 chunk、32K；第一階段 C1 的做法），依 §4.1 算三個限速頻寬
            B＝(64−b*)×chunk_bytes÷Σ_{i<b*} f(i)，b*∈{16,32,48}，另加「CPU 層不限速、真的 H2D」錨點 → bw.json。
            同時量真實 pinned H2D（一個 BF16 chunk 進 KV 緩衝的 strided 視圖，C0 的方法），B 不明顯低於它就標出來。
  noise     §5.3：只有 E0，每個（頻寬, GPU 狀態）格子 launches 個 process × reps 次；
            δ＝各格「各 process 中位數的 (最大−最小)÷整體中位數」的最大值，δ*＝max(5%, δ) → noise.json
  k0        §2.4 K0：E0 對 E1（--fmt 換成 int4_* 時＝第 2 階的 E1-INT4），3 個限速＋CPU，GPU 閒／忙，32K。
            判準（寫死）：E1 在每一格都沒有「贏」E0 → K0 成立。「贏」照 §5.3：中位數的差 > δ*，
            而且配對差（同一次重複裡相鄰量的 E0、E1）的 bootstrap 95% CI 不含 0。
            δ* 從 --delta-json（noise 的輸出）讀；沒給就用這次 k0 自己的 E0 列算（同樣的定義），並寫明來源。
  grid      §4.2：8 個配置（E0、E1、E1s、E1sr、F25、M25、B25、R25）× 頻寬 × GPU 狀態；每格另加 compute_only、
            load_only（E0、E1 的存法）兩條參考線；同一次重複裡配置順序隨機打散；
            --lengths 8192 16384：只有 E0、E1、B25、R25、GPU 閒（合成 token，時間和內容無關，Cake p.5）。
  k0-verdict  不重跑：從 restores.csv 裡 --from-run 那次（grid）的 E0、E1 配對列算 K0（10/15 晚改動）
  summary   從 restores.csv 算每格的中位數、會合點分布、對只算／只載的比值 → summary.csv（不用 GPU）

grid、k0 跑完另外寫（合併進）meets.json，給 m10_quality.py「讀法 B 重放」用（格式寫死）：
  {"model": "qwen", "source_run_ids": [...],
   "meets": {"<config>": {"<bw_label>": {"idle": <b 中位數 int>, "busy": <int>, "idle_all": [...], "busy_all": [...]}}}}
  b＝會合點＝n_recompute（block < b 是重算的）；只取主長度、cake、沒被標污染的列；中位數取整（.5 往下）。
  config：fmt＝--base-fmt（預設 fp8_head）時用原名（E0…R25）；其他格式加 "@fmt"（例 "E1@int4_kivi"）；E0 永遠不加。

每個 process 只建一個 Restorer（同一個 process 建兩個，後建的會變慢：RUNLOG_MI300X Milestone 10-pilot 異常 4）。
noise／k0／grid 由外層這個 process 依序開 launches 個子 process（`_worker`），每個子 process 是一次「重開」。
要用 GPU 的外層請自己包 `flock <鎖檔>`（MI300X：/mlsteam/data/tiara/gpu.lock）；這支不拿鎖。

配置（§2.1，n 個 block；n＝64 時就是計劃寫的 block 範圍，n 較小時照比例）：
  E0 全 BF16；E1 全低精度（含 block 0）；E1s block 0 BF16；E1sr block 0 與最後一個 BF16；
  F25 block 1..n/4；M25 block 3n/8..5n/8−1（n＝64：24–39）；B25 最後 n/4 個；R25 從 block 1..n−1 隨機抽 n/4 個
  （placement_seed＝launch×1000＋rep，每次重複換一組）。「低精度」＝--fmt（預設 fp8_head；D1 選定粒度後改）。

時間欄（§5.4）：
  t_encode、t_d2h：寫入時（capture）每個 block 量一次（CUDA event），每個配置把「它存的那種格式」的 block 加總；
                   同一個 launch 內各次重複共用（寫入只做一次）。
  t_write：限速層的寫入佔用（Tier.reserve_write 的時間，對讀寫對稱的限速層）；CPU 錨點＝真實 D2H，記 t_d2h，t_write 記 NOT_MEASURED。
  t_read：載入線每次 _load 的牆鐘時間總和（含等限速器、H2D、反量化）；t_h2d：還原器 time_h2d=True 時用 CUDA event 量每次 H2D 的總和（10/15 加；限速層下 H2D 和等待重疊，所以 t_h2d ≤ t_read）。
  t_dequant、t_recompute、t_wait、waiting_line、bytes_read、copy_bound：取自還原器的 rec。
  t_new_turn：新一輪 --new-tokens 個 token 的 prefill＋第一個 token；ttft＝還原開始到第一個 token。

GPU 忙（§1）：code/m7_busy.py＋duty，duty 從 --busy-levels（預設 results/m7_write_policy_mi300x/busy_levels.json）
  的 --busy-key（預設 busy2：MI300X 上讓重算慢 2.07 倍）讀。**H200 上要先用 m7_write_policy.py busy-calib 重新校準
  （§R2 C5），產生 H200 自己的 busy_levels.json 再傳進來。**
污染：每個子 process 開 gpu_guard.GpuWatcher（smi_index(0)），每列記 contaminated_gpu；開頭記 host_contention 的 level
  （Slurm 裡看不到整個節點時是 UNOBSERVABLE_HOST，不當成乾淨；CLAUDE.md 規則 7）。

模型：--model llama|qwen；路徑取 M7_LLAMA_GLOB／M7_QWEN_GLOB，沒設用 m10_tasks.MODELS（MI300X 的位置）。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

GiB = 1 << 30
CONFIGS = ["E0", "E1", "E1s", "E1sr", "F25", "M25", "B25", "R25"]
SHORT_CONFIGS = ["E0", "E1", "B25", "R25"]          # 8K／16K 只跑這四個（§4.2）
TARGETS = (16, 32, 48)                               # §4.1 的目標會合點 b*（以 64 個 block 計）
NM = "NOT_MEASURED"

RESTORE_FIELDS = [
    "run_id", "ts", "model", "length", "n_blocks", "fmt", "config", "mode", "placement_seed", "bw_name", "bw_GiBps",
    "gpu_state", "duty", "launch", "rep", "order_idx", "b", "n_recompute", "n_load_bf16", "n_load_fp8", "n_load_int8",
    "n_load_int4", "fp8_stored", "fp8_read", "bytes_stored", "bytes_read", "t_encode", "t_d2h", "t_write", "t_read",
    "t_h2d", "t_dequant", "t_recompute", "t_wait", "waiting_line", "t_restore", "t_new_turn", "ttft", "copy_bound",
    "kv_bad", "contaminated_gpu", "host_level"]
BLOCK_FIELDS = ["run_id", "launch", "rep", "length", "bw_name", "gpu_state", "config", "mode", "block",
                "stored_dtype", "source", "t_start", "t_end"]


def run_id():
    return os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc"


def ts():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def model_glob(model: str) -> str:
    env = os.environ.get(f"M7_{model.upper()}_GLOB")
    if env:
        return env
    import m10_tasks
    return m10_tasks.MODELS[model]


def drop_set(config: str, n: int, seed: int) -> frozenset:
    """14 號 D3（10/19）：'<底>-D<F|B|R><佔比>'（例 E1s-DB25）→ 被丟掉的 block（從 1..n−1 挑 round(佔比×(n−1)) 個；
    F＝最舊、B＝最新、R＝隨機，seed 和 R25 相同）。其他配置回傳空集合。"""
    if "-D" not in config:
        return frozenset()
    spec = config.split("-D", 1)[1]
    pol, k = spec[0], int(round(int(spec[1:]) / 100 * (n - 1)))
    cand = list(range(1, n))
    if pol == "F":
        out = cand[:k]
    elif pol == "B":
        out = cand[len(cand) - k:]
    elif pol == "R":
        out = random.Random(seed).sample(cand, k)
    else:
        raise ValueError(config)
    return frozenset(out)


def layout(config: str, n: int, seed: int) -> list[bool]:
    """每個 block 存低精度（True）或 BF16（False）。§2.1；n 不是 64 時照比例。"""
    if "-D" in config:                                  # 14 號 D3：丟掉的 block 另外由 drop_set 決定；存的精度照底
        return layout(config.split("-D", 1)[0], n, seed)
    q = n // 4
    lp = [False] * n
    if config == "E0":
        pass
    elif config == "E1":
        lp = [True] * n
    elif config == "E1s":
        lp = [i != 0 for i in range(n)]
    elif config == "E1sr":
        lp = [i not in (0, n - 1) for i in range(n)]
    elif config == "F25":
        for i in range(1, 1 + q):
            lp[i] = True
    elif config == "M25":
        s = (3 * n) // 8
        for i in range(s, s + q):
            lp[i] = True
    elif config == "B25":
        for i in range(n - q, n):
            lp[i] = True
    elif config == "R25":
        for i in random.Random(seed).sample(range(1, n), q):
            lp[i] = True
    else:
        raise ValueError(config)
    return lp


def restore_variant(default: bool = False) -> dict:
    """09 §6.4 還原方法的變體（環境變數；預設全部＝原本）。M10_DQ_KERNEL 由 m10_dequant_fast.install 讀。"""
    if default:
        return dict(kernel="k1", sched="serial", group=1, prio=False, unit=512, meet="claim", io_prio=False)
    return dict(kernel=os.environ.get("M10_DQ_KERNEL", "k1"), sched=os.environ.get("M10_DQ_SCHED", "serial"),
                group=int(os.environ.get("M10_DQ_GROUP", "1")), prio=os.environ.get("M10_DQ_PRIO") == "1",
                unit=int(os.environ.get("M10_UNIT", "512")), meet=os.environ.get("M10_MEET", "claim"),
                io_prio=os.environ.get("M10_RS_PRIO") == "1")      # 13 §3 P1


# ====================================================================== GPU 端（子 process）
class Busy:
    """背景 matmul（m7_busy.py）；duty＝0 時什麼都不做。
    13 §3：環境變數 M10_BUSY_CU_MASK 有設時，背景 process 帶 HSA_CU_MASK＝它（ROCm 的運算單元遮罩）；
    結束後 self.stats 記背景做了幾次 matmul、多久（背景吞吐）。"""

    def __init__(self, duty: float, watch=None):
        self.duty, self.p, self.watch, self.stats = duty, None, watch, None

    def __enter__(self):
        if self.duty > 0:
            env = dict(os.environ)
            mask = os.environ.get("M10_BUSY_CU_MASK")
            if mask:
                env["HSA_CU_MASK"] = mask
            mps = os.environ.get("M10_BUSY_MPS_PCT")                # H200（13_h200_repro §14）：CUDA MPS 下背景能用的 SM 比例；沒設時不變
            if mps:
                env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = mps
            # 10/22（m16 X3 翻轉格子）：M10_BUSY_SCRIPT 可換成別的背景（例：code/m16_busy_mem.py＝記憶體頻寬型）；沒設＝m7_busy.py（原行為）
            script = os.environ.get("M10_BUSY_SCRIPT") or os.path.join(HERE, "m7_busy.py")
            self.p = subprocess.Popen([sys.executable, script, "--duty", str(self.duty)],
                                      stdout=subprocess.PIPE, text=True, env=env)
            line = self.p.stdout.readline()
            if not line.startswith("READY"):
                raise RuntimeError(f"{os.path.basename(script)} 沒有啟動：{line!r}")
            self.stats = {"duty": self.duty, "cu_mask": mask or "", "ready": line.strip(), "script": os.path.basename(script),
                          **({"mps_pct": mps} if mps else {})}
            time.sleep(1.0)
        return self

    def _read_done(self):
        try:
            for ln in self.p.stdout.read().splitlines():
                if ln.startswith("DONE"):
                    kv = dict(x.split("=") for x in ln.split()[1:])
                    self.stats.update(n_matmul=int(kv["n"]), seconds=float(kv["s"]),
                                      matmul_per_s=round(int(kv["n"]) / float(kv["s"]), 3))
        except Exception as ex:                                    # 讀不到就照實記，不當成 0
            self.stats["done_error"] = repr(ex)

    def __exit__(self, *a):
        if not self.p:
            return False
        if self.watch is None or os.environ.get("M10_BUSY_TEARDOWN_PAUSE") != "1":
            self.p.terminate()
            self.p.wait(30)
            self._read_done()
            return False
        # 2026-10-15 Qwen FP8 grid L0：m7_busy 結束、容器內 kfd 已關之後，amd-smi 仍列著它的 VRAM（同 09-15 vLLM 關閉那次）
        # → 數量判斷誤報 1 筆「容器外行程」。收尾期間暫停取樣（區間寫進 gpu_guard JSON），等 amd-smi 連 2 次不再多出條目
        # （最多 30 s）才恢復；真的外來行程那時還在的話，恢復後下一筆照樣抓到。
        from gpu_guard import foreign_on
        with self.watch.pause("m7_busy teardown"):
            self.p.terminate()
            self.p.wait(30)
            self._read_done()
            t0, ok = time.time(), 0
            while time.time() - t0 < 30 and ok < 2:
                ok = 0 if any(x["pid"] == -1 for x in foreign_on(self.watch.gpu, self.watch.own_root)) else ok + 1
                time.sleep(0.5)
        return False


class Bench:
    """一個 process 一份：模型、合成 token、prefill 後的參考 KV、寫入時量好的 BF16／低精度主機副本。"""

    def __init__(self, model: str, n_max: int, fmt: str, new_tokens: int, seed: int = 0, unit: int = 512):
        import torch
        os.environ["M7_MODEL_GLOB"] = model_glob(model)
        from m7_model import CHUNK, KVModel
        from m7_restore_harness import Restorer
        self.torch, self.CHUNK = torch, CHUNK
        self.model, self.fmt, self.new_tokens = model, fmt, new_tokens
        self.km = KVModel(max_len=n_max * CHUNK + max(512, new_tokens))
        self.rs = Restorer(self.km)                    # 一個 process 只建這一個
        self.rs.time_h2d = True                        # 10/15：還原器加了 H2D 的 CUDA event 計時（預設關，這裡開）
        g = torch.Generator().manual_seed(seed)
        self.ids = torch.randint(1000, min(120000, self.km.cfg.vocab_size), (n_max * CHUNK + new_tokens,),
                                 generator=g).to(self.km.device)
        self.n_max = n_max
        self.captured_n = None
        self.unit = unit                               # 09 §6.4 G 組：還原單位（token）；512＝原本
        self.k = 512 // unit

    # ---- 寫入（capture）：每個 block 存 BF16 與低精度兩份，量編碼與 D2H ----
    def capture(self, n: int):
        torch, km = self.torch, self.km
        from m10_codec import encode
        if self.unit == self.CHUNK:
            km.prefill_chunked(self.ids[:n * self.CHUNK])
        else:                                          # G 組：寫入時的 prefill 也用同樣的單位，重算＝參考才會逐位元相同
            for s0 in range(0, n * self.CHUNK, self.unit):
                km.forward_span(self.ids[s0:s0 + self.unit], s0)
        torch.cuda.synchronize()
        self.bf, self.lp, self.t_enc, self.t_d2h_bf, self.t_d2h_lp = [], [], [], [], []
        for i in range(n):
            v = km.chunk_view(i)
            h = km.host_chunk()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            h.copy_(v, non_blocking=True)
            e1.record()
            e1.synchronize()
            self.bf.append(h)
            self.t_d2h_bf.append(e0.elapsed_time(e1) / 1e3)
            if self.fmt != "bf16":
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                eg = encode(v, self.fmt)
                e1.record()
                e1.synchronize()
                self.t_enc.append(e0.elapsed_time(e1) / 1e3)
                hp = eg.map(lambda t: torch.empty(t.shape, dtype=t.dtype, pin_memory=True))
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                hp.copy_from(eg, non_blocking=True)
                e1.record()
                e1.synchronize()
                self.lp.append(hp)
                self.t_d2h_lp.append(e0.elapsed_time(e1) / 1e3)
        self.ref = km.kv[:, :, :, :n * self.CHUNK].clone()       # 驗證用（GPU）
        self.captured_n = n
        if self.unit != self.CHUNK:                    # 每個 block 切成 k 段，各自連續、pinned（編碼仍是每 512 一個 block）
            from m10_dequant_fast import split_enc
            U = self.unit
            self.bf_u = [h[:, :, :, j * U:(j + 1) * U].contiguous().pin_memory() for h in self.bf for j in range(self.k)]
            self.lp_u = [x.pin() for e in self.lp for x in split_enc(e, U)] if self.fmt != "bf16" else []

    def expected_lp(self, i: int):
        from m10_codec import decode
        if self.unit != self.CHUNK:
            return decode(self.lp_u[i].to(self.km.device))
        return decode(self.lp[i].to(self.km.device))

    def host_for(self, lp_mask):
        if self.unit != self.CHUNK:                    # 還原單位的清單（lp_mask 仍是每個 block 一格）
            return [self.lp_u[u] if lp_mask[u // self.k] else self.bf_u[u] for u in range(len(lp_mask) * self.k)]
        return [self.lp[i] if m else self.bf[i] for i, m in enumerate(lp_mask)]

    def write_cost(self, lp_mask, tier_spec):
        """寫入成本（§5.4）：編碼、D2H 照實量的加總；限速層的寫入佔用照 Tier.reserve_write。"""
        from m7_restore_harness import Tier
        t_enc = sum(self.t_enc[i] for i, m in enumerate(lp_mask) if m) if self.fmt != "bf16" else 0.0
        t_d2h = sum(self.t_d2h_lp[i] if m else self.t_d2h_bf[i] for i, m in enumerate(lp_mask))
        stored = [self.lp[i].nbytes if m else self.km.chunk_bytes for i, m in enumerate(lp_mask)]
        if tier_spec["kind"] == "cpu":
            t_write = NM
        else:
            Bps = tier_spec["GiBps"] * GiB
            t = Tier("w", read_Bps=Bps, write_Bps=Bps)
            now, done = 0.0, 0.0
            for b in stored:                                   # 背靠背寫入：虛擬時鐘，只算裝置佔用
                done = t.reserve_write(b, now=now)
                now = done
            t_write = done
        return t_enc, t_d2h, sum(stored), t_write

    def restore(self, lp_mask, tier_spec, mode: str, n: int):
        torch, km, rs = self.torch, self.km, self.rs
        from m7_restore_harness import Tier
        if tier_spec["kind"] == "cpu":
            tier = Tier("cpu_real", read_Bps=0, write_Bps=0)
        else:
            Bps = tier_spec["GiBps"] * GiB
            tier = Tier(f"ssd_{tier_spec['name']}", read_Bps=Bps, write_Bps=Bps)
        km.kv[:, :, :, :n * self.CHUNK].zero_()
        torch.cuda.synchronize()
        meet0 = rs.meet_rule
        if mode in ("loadall", "claim"):                       # 13 §2：同一個 process 裡交錯跑「全部載入」「照 Cake」
            rs.meet_rule = "none" if mode == "loadall" else "claim"
        t0 = time.perf_counter()
        try:
            rec = rs.restore(self.ids, n * self.k, self.host_for(lp_mask), [tier] * (n * self.k),
                             "cake" if mode in ("loadall", "claim") else mode)
        finally:
            rs.meet_rule = meet0
        t_new, _ = rs.new_turn(self.ids[n * self.CHUNK:n * self.CHUNK + self.new_tokens], n * self.CHUNK)
        rec["ttft"] = time.perf_counter() - t0
        rec["t_new_turn"] = t_new
        return rec

    def kv_bad(self, rec, n) -> int:
        bad = 0
        U = self.unit
        for i, src, _, _ in rec["blocks"]:
            v = self.km.kv[:, :, :, i * U:(i + 1) * U]
            want = self.ref[:, :, :, i * U:(i + 1) * U] \
                if src in ("recompute", "load_bf16") else self.expected_lp(i)
            bad += int(not self.torch.equal(v, want))
        return bad


def _watcher(out_path):
    try:
        from gpu_guard import GpuWatcher, host_contention, smi_index
        w = GpuWatcher(gpu=smi_index(0), own_root=os.getpid(), out_path=out_path)
        w.__enter__()
    except Exception as ex:                    # 查不到不等於乾淨（規則 7）：照跑，但每列標 NOT_CHECKED
        print("GpuWatcher unavailable:", repr(ex), flush=True)
        w = None
    try:
        from gpu_guard import host_contention
        level = host_contention(exclude_gpu=None)["level"]
    except Exception as ex:
        level = f"UNKNOWN({type(ex).__name__})"
    return w, level


def worker(spec: dict):
    """一個子 process＝一次「重開」。spec 由外層 JSON 傳進來。"""
    import torch
    # 大掃描（09）才用的開關，預設都關（主線的行為不變）：新格式（m10_codec_x）、融合的反量化 kernel（m10_dequant_fast）
    if os.environ.get("M10_CODEC_X") == "1":
        import m10_codec_x
        m10_codec_x.install()
    if os.environ.get("M10_DEQUANT_FAST") == "1":
        import m10_dequant_fast
        m10_dequant_fast.install()
    rng = random.Random(f"m10-speed|{spec['launch']}|{spec.get('seed', 0)}")
    lengths = spec["lengths"]
    n_max = max(lengths) // 512
    rv = restore_variant()
    bench = Bench(spec["model"], n_max, spec["fmt"], spec["new_tokens"], unit=rv["unit"])
    if rv != restore_variant(default=True):            # 09 §6.4：還原方法的變體（預設全關＝原本）
        bench.rs.set_options(unit=rv["unit"], sched=rv["sched"], group=rv["group"], compute_prio=rv["prio"],
                             meet=rv["meet"], io_prio=rv["io_prio"])
        print("[restore_variant]", rv, flush=True)
    rid = run_id()
    os.makedirs(spec["out_dir"], exist_ok=True)
    extra = spec.get("extra_modes") or []
    fields = RESTORE_FIELDS + (["meet_rule_used"] if rv["meet"] in ("none", "g5", "g5b") or extra else [])
    rpath = os.path.join(spec["out_dir"], "restores.csv")
    if os.path.exists(rpath) and os.path.getsize(rpath) > 0:
        with open(rpath) as f0:
            if next(csv.reader(f0)) != fields:
                raise SystemExit(f"{rpath} 的欄位和這次不同（13 號的 meet_rule_used）；換一個 --out-dir")
    rf = open(rpath, "a", newline="")
    rw = csv.DictWriter(rf, fieldnames=fields, extrasaction="ignore")
    if rf.tell() == 0:
        rw.writeheader()
    bpath = spec.get("blocks_out") or os.path.join(spec["out_dir"], "blocks.csv")
    os.makedirs(os.path.dirname(bpath), exist_ok=True)
    bf = open(bpath, "a", newline="")
    bw_ = csv.DictWriter(bf, fieldnames=BLOCK_FIELDS)
    if bf.tell() == 0:
        bw_.writeheader()
    watch, level = _watcher(os.path.join(spec["out_dir"], f"gpu_guard_{rid}_L{spec['launch']}.json"))
    busy_pid = set()
    busy_stats = []
    tiers = spec["tiers"]
    states = spec["gpu_states"]
    xtiers = spec.get("extra_tiers") or [t["name"] for t in tiers]

    def contaminated():
        if watch is None:
            return "NOT_CHECKED"
        bad = {p: v for p, v in watch.intruders.items() if p not in busy_pid}
        return bool(bad) or watch.started_clean is False

    for length in lengths:
        n = length // 512
        bench.capture(n)
        cfgs = spec["configs"] if length == spec["main_length"] else [c for c in spec["configs"] if c in SHORT_CONFIGS]
        st_here = states if length == spec["main_length"] else [s for s in states if s["name"] == "idle"]
        # 暖機：每種模式各一次（不記）
        for mode in ("cake", "compute_only", "load_only"):
            bench.restore(layout("E1", n, 0), tiers[0], mode, n)
        verified = set()
        for st in st_here:
            with Busy(st["duty"], watch) as busy:
                if busy.p:
                    busy_pid.add(busy.p.pid)
                if rv["meet"] in ("g5", "g5b"):                         # 13 §4：G5 的 profiler（每個頻寬先照 Cake 跑 E0、E1 各一次，不記）
                    bench.rs.g5_force_claim = True
                    for tier in tiers:
                        bench.rs.g5_key = (tier["name"], st["name"])
                        for cfg in ("E0", "E1"):
                            bench.restore(layout(cfg, n, 0), tier, "cake", n)
                    bench.rs.g5_force_claim = False
                for rep in range(spec["reps"]):
                    for tier in tiers:
                        seed = spec["launch"] * 1000 + rep
                        jobs = [(c, "cake") for c in cfgs]
                        if extra and tier["name"] in xtiers:
                            jobs += [(c, m) for c in cfgs for m in extra]
                        if spec.get("refs") and length == spec["main_length"]:
                            jobs += [("-", "compute_only")] + [(c, "load_only") for c in spec["refs"]]
                        rng.shuffle(jobs)                    # 同一次重複裡隨機打散（§4.2）
                        for k, (cfg, mode) in enumerate(jobs):
                            mask = layout("E0" if cfg == "-" else cfg, n, seed)
                            bench.rs.g5_key = (tier["name"], st["name"])
                            dset = drop_set(cfg, n, seed) if cfg != "-" else frozenset()
                            bench.rs.drop = frozenset(u for i in dset for u in range(i * bench.k, (i + 1) * bench.k))
                            try:
                                rec = bench.restore(mask, tier, mode, n)
                            finally:
                                bench.rs.drop = frozenset()
                            key = (length, cfg, mode, tier["name"], st["name"])
                            kvb = ""
                            if spec.get("verify") and key not in verified:
                                # nodq（只搬不拆）不會填 KV 緩衝，照 13 §2 不檢查
                                kvb = "SKIP_nodq" if bench.rs.dq_sched == "nodq" else bench.kv_bad(rec, n)
                                verified.add(key)
                            t_enc, t_d2h, stored, t_write = bench.write_cost(mask, tier)
                            srcs = [s for _, s, _, _ in rec["blocks"]]
                            n_lp_read = sum(s not in ("recompute", "load_bf16") for s in srcs)
                            row = dict(
                                run_id=rid, ts=ts(), model=spec["model"], length=length, n_blocks=n, fmt=spec["fmt"],
                                config=cfg, mode=mode, placement_seed=seed if (cfg == "R25" or "-DR" in cfg) else "",
                                bw_name=tier["name"], bw_GiBps=tier.get("GiBps", "real"), gpu_state=st["name"],
                                duty=st["duty"], launch=spec["launch"], rep=rep, order_idx=k, b=rec["meet"],
                                n_recompute=rec["n_recompute"], n_load_bf16=rec["n_load_bf16"],
                                n_load_fp8=rec["n_load_fp8"], n_load_int8=rec["n_load_int8"],
                                n_load_int4=rec["n_load_int4"], fp8_stored=sum(mask), fp8_read=n_lp_read,
                                bytes_stored=stored, bytes_read=rec["bytes_read"],
                                t_encode=round(t_enc, 6), t_d2h=round(t_d2h, 6),
                                t_write=t_write if t_write == NM else round(t_write, 6),
                                t_read=round(rec["t_load_cpu"] + rec["t_load_ssd"], 6), t_h2d=round(rec.get("t_h2d", 0.0), 6) if "t_h2d" in rec else NM,
                                t_dequant=round(rec["t_dequant"], 6), t_recompute=round(rec["t_recompute"], 6),
                                t_wait=round(rec.get("t_wait", 0.0), 6) if mode == "cake" else "",
                                waiting_line=rec.get("waiting_line", ""), t_restore=round(rec["t_restore"], 6),
                                t_new_turn=round(rec["t_new_turn"], 6), ttft=round(rec["ttft"], 6),
                                copy_bound=rec["n_copy_bound"], kv_bad=kvb, contaminated_gpu=contaminated(),
                                host_level=level, meet_rule_used=rec.get("meet_rule_used", "claim" if mode == "cake" else mode))
                            rw.writerow(row)
                            for i, s, t_s, t_e in rec["blocks"]:
                                bw_.writerow(dict(run_id=rid, launch=spec["launch"], rep=rep, length=length,
                                                  bw_name=tier["name"], gpu_state=st["name"], config=cfg, mode=mode,
                                                  block=i, stored_dtype=spec["fmt"] if mask[i // bench.k] else "bf16", source=s,
                                                  t_start=round(t_s, 6), t_end=round(t_e, 6)))
                        rf.flush()
                        bf.flush()
                    print(f"[L{spec['launch']} {length} {st['name']} rep {rep}] ok", flush=True)
            if busy.p:
                busy_pid.discard(busy.p.pid)
                busy_stats.append(dict(state=st["name"], length=length, **(busy.stats or {})))
    if busy_stats:
        json.dump(busy_stats, open(os.path.join(spec["out_dir"], f"busy_{rid}_L{spec['launch']}.json"), "w"), indent=1)
    rf.close()
    bf.close()
    if watch is not None:
        watch.__exit__(None, None, None)


# ====================================================================== 外層
def load_tiers(a) -> list[dict]:
    bw = json.load(open(a.bw_json))
    tiers = []
    for name in a.bw_names:
        if name == "cpu":
            tiers.append({"name": "cpu", "kind": "cpu"})
        else:
            tiers.append({"name": name, "kind": "throttle", "GiBps": float(bw["targets"][name]["GiBps"])})
    return tiers


def load_states(a) -> list[dict]:
    lv = json.load(open(a.busy_levels))
    out = []
    for s in a.gpu_states:
        out.append({"name": s, "duty": 0.0 if s == "idle" else float(lv[a.busy_key if s == "busy" else s])})
    return out


def run_launches(a, configs, refs, lengths):
    spec = dict(model=a.model, fmt=a.fmt, new_tokens=a.new_tokens, lengths=lengths, main_length=a.main_length,
                configs=configs, refs=refs, tiers=load_tiers(a), gpu_states=load_states(a), reps=a.reps,
                out_dir=a.out_dir, blocks_out=a.blocks_out, verify=not a.no_verify,
                extra_modes=getattr(a, "extra_modes", []), extra_tiers=getattr(a, "extra_tiers", []))
    json.dump({**spec, "argv": sys.argv, "run_id": run_id(), "restore_variant": restore_variant()},
              open(os.path.join(a.out_dir, f"spec_{a.cmd}.json"), "w"), indent=1, ensure_ascii=False)
    for launch in range(a.launches):
        s = dict(spec, launch=launch + getattr(a, "launch0", 0))
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "_worker", json.dumps(s)])
        if r.returncode != 0:
            raise SystemExit(f"launch {launch} 失敗（exit {r.returncode}）；照規則 2 停下來")


def _rows(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in ("t_restore", "ttft", "b"):
            r[k] = float(r[k])
    return rows


def compute_delta(rows, metric="t_restore"):
    """§5.3：E0、cake、主長度的每格，各 launch 中位數的 (最大−最小)÷整體中位數；δ＝最大值。"""
    cells = {}
    for r in rows:
        if r["config"] == "E0" and r["mode"] == "cake":
            cells.setdefault((r["length"], r["bw_name"], r["gpu_state"]), []).append(r)
    out = {}
    for k, v in cells.items():
        per = {}
        for r in v:
            per.setdefault(r["launch"], []).append(r[metric])
        meds = [statistics.median(x) for x in per.values()]
        allm = statistics.median([r[metric] for r in v])
        out["|".join(k)] = {"n_launch": len(meds), "launch_medians": [round(m, 5) for m in meds],
                           "overall_median": round(allm, 5),
                           "delta": (max(meds) - min(meds)) / allm if len(meds) > 1 else None,
                           "b_values": sorted({int(r["b"]) for r in v})}
    ds = [c["delta"] for c in out.values() if c["delta"] is not None]
    d = max(ds) if ds else None
    return {"cells": out, "delta": d, "delta_star": max(0.05, d) if d is not None else None,
            "rule": "δ＝各格（E0、cake）各 process 中位數 (max−min)÷整體中位數 的最大值；δ*＝max(5%, δ)（03 §5.3）"}


def bootstrap_ci(diffs, n_boot=2000, seed=0):
    rng = random.Random(seed)
    meds = []
    for _ in range(n_boot):
        s = [rng.choice(diffs) for _ in diffs]
        meds.append(statistics.median(s))
    meds.sort()
    return meds[int(0.025 * n_boot)], meds[int(0.975 * n_boot) - 1]


def write_meets(a, rows):
    """合併寫入 <out_dir>/meets.json（見檔頭格式）。"""
    import math
    p = os.path.join(a.out_dir, "meets.json")
    d = json.load(open(p)) if os.path.exists(p) else {"model": a.model, "source_run_ids": [], "meets": {}}
    if d.get("model") != a.model:
        raise SystemExit(f"meets.json 的模型是 {d.get('model')}，這次是 {a.model}；換一個 --out-dir")
    acc = {}
    for r in rows:
        if r["mode"] != "cake" or int(r["length"]) != a.main_length or r["contaminated_gpu"] == "True":
            continue
        cfg = r["config"]
        key = cfg if (cfg == "E0" or r["fmt"] == a.base_fmt) else f"{cfg}@{r['fmt']}"
        acc.setdefault(key, {}).setdefault(r["bw_name"], {}).setdefault(r["gpu_state"], []).append(int(r["b"]))
    for key, bws in acc.items():
        for bw, sts in bws.items():
            ent = d["meets"].setdefault(key, {}).setdefault(bw, {})
            for st, bs in sts.items():
                ent[st] = int(math.floor(statistics.median(bs)))
                ent[st + "_all"] = sorted(bs)
    d["source_run_ids"] = sorted(set(d["source_run_ids"]) | {r["run_id"] for r in rows})
    json.dump(d, open(p, "w"), indent=1, ensure_ascii=False)
    print("meets.json:", json.dumps({k: {b: {s: v for s, v in e.items() if not s.endswith("_all")}
                                         for b, e in x.items()} for k, x in d["meets"].items()}), flush=True)


def cmd_k0(a):
    run_launches(a, ["E0", "E1"], [], [a.main_length])
    rows = [r for r in _rows(os.path.join(a.out_dir, "restores.csv")) if r["run_id"] == run_id()]
    write_meets(a, rows)
    k0_verdict(a, rows, run_id())


def cmd_k0_verdict(a):
    """不重跑：從已經寫好的 restores.csv（例如 grid 那次）取 --from-run 的 E0、E1 列，照同一個判準算 K0。
    grid 裡同一次重複（launch, rep, 頻寬, GPU 狀態）的 E0、E1 就是配對（順序隨機打散）。10/15 晚改動，03 開頭。"""
    rows = [r for r in _rows(os.path.join(a.out_dir, "restores.csv"))
            if r["run_id"] == a.from_run and r["config"] in ("E0", "E1") and int(r["length"]) == a.main_length]
    if not rows:
        raise SystemExit(f"restores.csv 裡沒有 run_id={a.from_run} 的 E0／E1 列")
    k0_verdict(a, rows, a.from_run)


def k0_verdict(a, rows, rid):
    if a.delta_json:
        dj = json.load(open(a.delta_json))
        dstar, dsrc = dj["delta_star"], a.delta_json
    else:
        dj = compute_delta(rows)
        dstar, dsrc = dj["delta_star"], "這次 k0 自己的 E0 列（--delta-json 沒給）"
    cells, verdict_rows = {}, []
    for r in rows:
        if r["mode"] == "cake":
            cells.setdefault((r["bw_name"], r["gpu_state"]), []).append(r)
    any_win = False
    for (bw, st), v in sorted(cells.items()):
        e0 = {(r["launch"], r["rep"]): r["t_restore"] for r in v if r["config"] == "E0"}
        e1 = {(r["launch"], r["rep"]): r["t_restore"] for r in v if r["config"] == "E1"}
        keys = sorted(set(e0) & set(e1))
        m0, m1 = statistics.median(e0.values()), statistics.median(e1.values())
        diffs = [e0[k] - e1[k] for k in keys]                  # 正＝E1 比較快
        lo, hi = bootstrap_ci(diffs) if len(diffs) >= 2 else (None, None)
        gain = (m0 - m1) / m0
        win = dstar is not None and gain > dstar and lo is not None and lo > 0
        any_win |= bool(win)
        verdict_rows.append({"bw_name": bw, "gpu_state": st, "n_pairs": len(keys), "median_E0_s": round(m0, 5),
                             "median_E1_s": round(m1, 5), "gain_E1_over_E0": round(gain, 4),
                             "paired_diff_ci95_s": [lo, hi], "E1_wins": bool(win),
                             "b_E0": sorted({int(r["b"]) for r in v if r["config"] == "E0"}),
                             "b_E1": sorted({int(r["b"]) for r in v if r["config"] == "E1"}),
                             "contaminated_rows": sum(r["contaminated_gpu"] not in ("False", False) for r in v)})
    if dstar is None:
        verdict = "UNDECIDED（δ 算不出來：launches < 2，又沒給 --delta-json）"
    else:
        verdict = "K0 成立：E1 在每一格都沒有贏 E0 → 精度對還原時間沒有賺頭" if not any_win else \
            "K0 不成立：至少一格 E1 贏 E0（有賺頭）"
    out = {"run_id": rid, "model": a.model, "fmt": sorted({r["fmt"] for r in rows}), "delta_star": dstar, "delta_source": dsrc,
           "delta_detail": dj, "rule": "E1 贏＝中位數的差 > δ*，且配對差 bootstrap 95% CI 不含 0（03 §2.4、§5.3）",
           "cells": verdict_rows, "verdict": verdict}
    json.dump(out, open(os.path.join(a.out_dir, "k0_verdict.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, ensure_ascii=False, indent=1))


def cmd_noise(a):
    run_launches(a, ["E0"], [], [a.main_length])
    rows = [r for r in _rows(os.path.join(a.out_dir, "restores.csv")) if r["run_id"] == run_id()]
    d = compute_delta(rows)
    d.update(run_id=run_id(), model=a.model)
    json.dump(d, open(os.path.join(a.out_dir, "noise.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(d, ensure_ascii=False, indent=1))


def cmd_grid(a):
    run_launches(a, a.configs, a.refs, [a.main_length] + [L for L in a.lengths if L != a.main_length])
    write_meets(a, [r for r in _rows(os.path.join(a.out_dir, "restores.csv")) if r["run_id"] == run_id()])
    cmd_summary(argparse.Namespace(out_dir=a.out_dir, only_run=run_id()))


def cmd_summary(a):
    rows = _rows(os.path.join(a.out_dir, "restores.csv"))
    if getattr(a, "only_run", None):
        rows = [r for r in rows if r["run_id"] == a.only_run]
    cells = {}
    for r in rows:
        cells.setdefault((r["length"], r["bw_name"], r["gpu_state"], r["config"], r["mode"]), []).append(r)
    ref_c = {(k[0], k[1], k[2]): statistics.median(x["t_restore"] for x in v)
             for k, v in cells.items() if k[4] == "compute_only"}
    ref_l = {(k[0], k[1], k[2]): statistics.median(x["t_restore"] for x in v)
             for k, v in cells.items() if k[4] == "load_only" and k[3] == "E0"}
    out = []
    for k, v in sorted(cells.items()):
        m = statistics.median(x["t_restore"] for x in v)
        c = ref_c.get(k[:3])
        l_ = ref_l.get(k[:3])
        out.append(dict(length=k[0], bw_name=k[1], gpu_state=k[2], config=k[3], mode=k[4], n=len(v),
                        median_t_restore=round(m, 5), median_ttft=round(statistics.median(x["ttft"] for x in v), 5),
                        b_values=" ".join(str(b) for b in sorted({int(x["b"]) for x in v})),
                        median_fp8_read=statistics.median(int(x["fp8_read"]) for x in v),
                        median_bytes_read=statistics.median(int(x["bytes_read"]) for x in v),
                        speedup_vs_compute_only=round(c / m, 4) if c else "",
                        speedup_vs_load_only_E0=round(l_ / m, 4) if l_ else "",
                        copy_bound_rows=sum(int(x["copy_bound"]) > 0 for x in v),
                        contaminated_rows=sum(x["contaminated_gpu"] not in ("False",) for x in v),
                        run_ids=" ".join(sorted({x["run_id"] for x in v}))))
    p = os.path.join(a.out_dir, "summary.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for d in out:
        print(d, flush=True)


def cmd_calib_f(a):
    """f(i) 與頻寬選擇（§4.1）。只有 GPU 閒（選頻寬用閒的 f）。"""
    import torch
    os.environ["M7_MODEL_GLOB"] = model_glob(a.model)
    from m7_model import CHUNK, KVModel
    n = a.main_length // CHUNK
    km = KVModel(max_len=n * CHUNK + 512)
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(1000, min(120000, km.cfg.vocab_size), (n * CHUNK + 512,), generator=g).to(km.device)
    for i in range(4):                                    # 暖機（同 C1）
        km.forward_span(ids[i * CHUNK:(i + 1) * CHUNK], i * CHUNK)
    torch.cuda.synchronize()
    os.makedirs(a.out_dir, exist_ok=True)
    rid = run_id()
    fp = os.path.join(a.out_dir, "calib_f.csv")
    new = not os.path.exists(fp)
    f = open(fp, "a", newline="")
    w = csv.DictWriter(f, fieldnames=["run_id", "ts", "model", "chunk_idx", "pos_end", "rep", "ms", "gpu_state"])
    if new:
        w.writeheader()
    per = {i: [] for i in range(n)}
    for rep in range(a.reps):
        for i in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            km.forward_span(ids[i * CHUNK:(i + 1) * CHUNK], i * CHUNK)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1e3
            per[i].append(ms)
            w.writerow(dict(run_id=rid, ts=ts(), model=a.model, chunk_idx=i, pos_end=(i + 1) * CHUNK, rep=rep,
                            ms=round(ms, 3), gpu_state="idle"))
    f.close()
    fmed = [statistics.median(per[i]) / 1e3 for i in range(n)]
    # 真實 pinned H2D：一個 BF16 chunk 進 KV 緩衝的 strided 視圖（C0 的 H2D_pinned_chunk_into_kv_view）
    h = km.host_chunk()
    h.copy_(km.chunk_view(0))
    rates = []
    for _ in range(a.h2d_reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        km.chunk_view(1).copy_(h, non_blocking=True)
        torch.cuda.synchronize()
        rates.append(km.chunk_bytes / (time.perf_counter() - t0) / GiB)
    real = statistics.median(rates)
    targets = {}
    for bstar in TARGETS:
        bs = round(bstar * n / 64)                       # 長度不是 32K 時照比例
        B = (n - bs) * km.chunk_bytes / sum(fmed[:bs])
        targets[f"b{bstar}"] = {"b_star": bs, "GiBps": round(B / GiB, 4), "ratio_to_real_h2d": round(B / GiB / real, 4),
                                "below_real_h2d_by_2x": B / GiB < 0.5 * real}
    out = {"run_id": rid, "model": a.model, "model_path": km.cfg._name_or_path, "n_blocks": n,
           "chunk_bytes": km.chunk_bytes, "f_ms_median_by_chunk": [round(x * 1e3, 3) for x in fmed],
           "f_sum_s": round(sum(fmed), 4), "real_pinned_h2d_GiBps": round(real, 3),
           "real_pinned_h2d_note": "median of %d copies, BF16 chunk into strided KV view" % a.h2d_reps,
           "targets": targets, "cpu_anchor": "CPU 層不限速、真的 H2D（tier kind=cpu）",
           "rule": "B＝(n−b*)×chunk_bytes÷Σ_{i<b*} f(i)；f＝各 chunk 的中位數（GPU 閒）（03 §4.1）",
           "warning": [k for k, v in targets.items() if not v["below_real_h2d_by_2x"]]}
    json.dump(out, open(os.path.join(a.out_dir, "bw.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "f_ms_median_by_chunk"}, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("_worker")
    w.add_argument("spec")
    s = sub.add_parser("summary")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--only-run")
    c = sub.add_parser("calib-f")
    c.add_argument("--model", choices=["llama", "qwen"], required=True)
    c.add_argument("--out-dir", required=True)
    c.add_argument("--main-length", type=int, default=32768)
    c.add_argument("--reps", type=int, default=3)
    c.add_argument("--h2d-reps", type=int, default=20)
    for name in ("noise", "k0", "grid"):
        p = sub.add_parser(name)
        p.add_argument("--model", choices=["llama", "qwen"], required=True)
        p.add_argument("--out-dir", required=True)
        p.add_argument("--bw-json", required=True, help="calib-f 的 bw.json")
        p.add_argument("--bw-names", nargs="+", default=["b16", "b32", "b48", "cpu"])
        p.add_argument("--gpu-states", nargs="+", default=["idle", "busy"])
        p.add_argument("--busy-levels", default=os.environ.get("M10_BUSY_LEVELS")      # H200：環境變數（10/19；沒設時不變）
                       or os.path.join(HERE, "..", "results", "m7_write_policy_mi300x", "busy_levels.json"))
        p.add_argument("--busy-key", default="busy2", help="gpu_state=busy 用 busy_levels 裡哪個 duty（約 2 倍慢）")
        p.add_argument("--fmt", default="fp8_head")
        p.add_argument("--base-fmt", default="fp8_head", help="meets.json 裡不加 @fmt 的格式（第 1 階選定的 FP8 粒度）")
        p.add_argument("--main-length", type=int, default=32768)
        p.add_argument("--launches", type=int, default=3)
        p.add_argument("--reps", type=int, default=10)
        p.add_argument("--new-tokens", type=int, default=256)
        p.add_argument("--blocks-out", help="逐 block 檔（大；建議放 $TIARA_RUNS/$RUN_ID/）")
        p.add_argument("--no-verify", action="store_true", help="不做每格第一次的 KV 逐位元檢查")
        if name == "k0":
            p.add_argument("--delta-json", help="noise 的 noise.json；沒給就用這次 E0 列算 δ")
        if name == "grid":
            p.add_argument("--configs", nargs="+", default=CONFIGS)
            p.add_argument("--refs", nargs="*", default=["E0", "E1"], help="load_only 參考線用哪些存法")
            p.add_argument("--lengths", nargs="*", type=int, default=[8192, 16384])
            p.add_argument("--launch0", type=int, default=0, help="launch 編號從幾開始（09 §6.4 複賽用 200，避免和海選同 seed）")
            p.add_argument("--extra-modes", nargs="*", default=[], choices=["loadall", "claim"],
                           help="13 §2：每個配置另外跑「全部載入」（loadall）／「照 Cake」（claim），和 cake 交錯")
            p.add_argument("--extra-tiers", nargs="*", default=[], help="extra-modes 只在這些頻寬跑（空＝全部）")
    v = sub.add_parser("k0-verdict", help="不重跑，從 restores.csv 裡某一次（例 grid）的 E0、E1 列算 K0")
    v.add_argument("--model", choices=["llama", "qwen"], required=True)
    v.add_argument("--out-dir", required=True)
    v.add_argument("--from-run", required=True, help="restores.csv 的 run_id")
    v.add_argument("--delta-json", help="noise 的 noise.json")
    v.add_argument("--main-length", type=int, default=32768)
    a = ap.parse_args()
    if a.cmd == "_worker":
        worker(json.loads(a.spec))
        return
    if a.cmd == "summary":
        cmd_summary(a)
        return
    os.makedirs(a.out_dir, exist_ok=True)
    {"calib-f": cmd_calib_f, "noise": cmd_noise, "k0": cmd_k0, "grid": cmd_grid, "k0-verdict": cmd_k0_verdict}[a.cmd](a)


if __name__ == "__main__":
    main()
