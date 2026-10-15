"""m7_restore_harness.py — 儲存層（限速器）＋三種還原方式（phase1/03 演算法 2、3、4）

儲存層 Tier（演算法 4）：資料都放 pinned CPU 記憶體；時間由參數決定。
  讀：start=max(now,t_free)；done=start+c+S/B_read；t_free=done；真的做 H2D；等到 max(done, 真實複製結束)
  寫：start=max(now,t_free)；done=start+c+S/B_write；t_free=done（寫入只佔用裝置時間，
      真實的 D2H 由呼叫端在寫入當下做一次，見 SessionStore.capture）
  讀寫共用 t_free → 寫入會拖慢之後的讀取。

還原（每次回傳一筆時間拆解）：
  cake        演算法 2：重算線（GPU，從前往後）與載入線（I/O，從後往前）同時跑
  load_only   演算法 3：從頭載入連續前綴，第一個缺的之後全部重算
  compute_only 全部重算

精度這條線（m10，2026-10-15 加；03_plan_h200.md §1、§6）：
  host[i] 可以是 BF16 的 pinned tensor（原本的路徑，行為不變），或 m10_codec.Enc（低精度，已 pin）。
  Enc 的載入：限速器照「實際位元組」計時（Enc.nbytes）；先 H2D 到暫存區，等限速器的 done（資料到齊）
  之後才在 I/O stream 上反量化進 KV 緩衝（和重算搶 GPU）。反量化時間記在 t_dequant。
  每次還原另記：逐 block 來源（recompute／load_bf16／load_fp8／load_int8／load_int4）與起訖時間（rec["blocks"]）、
  各來源 block 數、bytes_read、copy_bound（限速器的 done 到了，真實複製還沒完成的次數）。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import torch

from m10_codec import Enc, decode
from m7_model import CHUNK, KVModel

GiB = 1 << 30
READ_SLACK_S = 0.0005


@dataclass
class Tier:
    name: str
    read_Bps: float          # 模擬讀取頻寬（bytes/s）；None/0 → 不限速（只做真實複製）
    write_Bps: float
    c_s: float = 0.0         # 每次 I/O 固定開銷（秒）
    t_free: float = 0.0
    # 干擾模型：fifo＝讀寫共用一個 t_free（phase1/03 演算法 4 原設計，A2 實測顯示過度悲觀）；
    # share＝讀、寫各自排隊，寫入進行中時讀取變慢 k_read 倍（k 由 A2 真實裝置校準）
    mode: str = "fifo"
    k_read: float = 1.0
    t_free_w: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)
    # 統計
    bytes_read: int = 0
    bytes_written: int = 0
    busy_write_s: float = 0.0

    def reserve_read(self, nbytes: int, now: float | None = None) -> float:
        """now=None → 真實時鐘；給 now → 虛擬時鐘（m7_sim.py），沒有軟體開銷所以不需要 SLACK。"""
        with self.lock:
            if now is None:
                now = time.perf_counter()
                # 連續讀取時，上一筆的軟體開銷（sleep 喚醒、event 同步）不該算成裝置閒置：
                # 距上一筆完成 < SLACK 就視為背靠背（C7 自檢 2026-10-08：未修正時 40 GiB/s 慢 6.5%）
                start = self.t_free if 0.0 <= now - self.t_free < READ_SLACK_S else max(now, self.t_free)
            else:
                start = max(now, self.t_free)
            dur = self.c_s + (nbytes / self.read_Bps if self.read_Bps else 0.0)
            if self.mode == "share" and self.t_free_w > start:
                dur *= self.k_read
            done = start + dur
            self.t_free = done
            self.bytes_read += nbytes
            return done

    def reserve_write(self, nbytes: int, now: float | None = None) -> float:
        with self.lock:
            now = time.perf_counter() if now is None else now
            dur = self.c_s + (nbytes / self.write_Bps if self.write_Bps else 0.0)
            if self.mode == "share":
                start = max(now, self.t_free_w)
                done = start + dur
                self.t_free_w = done
            else:
                start = max(now, self.t_free)
                done = start + dur
                self.t_free = done
            self.bytes_written += nbytes
            self.busy_write_s += dur
            return done

    def reset_clock(self):
        with self.lock:
            self.t_free = 0.0
            self.t_free_w = 0.0

    def backlog(self, now):
        return max(0.0, (self.t_free_w if self.mode == "share" else self.t_free) - now)


def sleep_until(t: float):
    while True:
        d = t - time.perf_counter()
        if d <= 0:
            return
        time.sleep(d if d > 0.002 else 0.0002)


class Restorer:
    """對一個 KVModel 執行還原。chunk 的主機資料由呼叫端提供：host[i] = pinned tensor 或 None，
    loc[i] = 這個 chunk 最快的 Tier 或 None（沒存）。"""

    def __init__(self, km: KVModel):
        self.km = km
        self.cstream = torch.cuda.Stream(device=km.device)
        self.iostream = torch.cuda.Stream(device=km.device)
        self._staging: dict[str, Enc] = {}     # 每種低精度格式一塊 GPU 暫存區（載入線一次只載一個 block）
        self.time_h2d = False                  # True：用 CUDA event 量每次 H2D（m10_speed 用；False 時路徑和原本完全相同）
        self.last_h2d_s = 0.0
        self.last_copy_bound = False
        self.last_dequant_s = 0.0
        self.last_nbytes = 0

    # ---- 載入一個 chunk（限速＋真實 H2D）----
    def _load(self, i: int, host, tier: Tier) -> float:
        if isinstance(host, Enc) and host.fmt != "bf16":
            return self._load_enc(i, host, tier)
        if isinstance(host, Enc):
            host = host.parts["x"]
        done = tier.reserve_read(self.km.chunk_bytes)
        if self.time_h2d:
            h0 = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(self.iostream):
                h0.record(self.iostream)
                self.km.chunk_view(i).copy_(host, non_blocking=True)
                ev = torch.cuda.Event(enable_timing=True)
                ev.record(self.iostream)
        else:
            with torch.cuda.stream(self.iostream):
                self.km.chunk_view(i).copy_(host, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(self.iostream)
        sleep_until(done)
        self.last_copy_bound = not ev.query()
        ev.synchronize()
        self.last_h2d_s = h0.elapsed_time(ev) / 1e3 if self.time_h2d else 0.0
        self.last_dequant_s = 0.0
        self.last_nbytes = self.km.chunk_bytes
        return done

    def _load_enc(self, i: int, host: Enc, tier: Tier) -> float:
        done = tier.reserve_read(host.nbytes)
        st = self._staging.get(host.fmt)
        if st is None:
            st = self._staging[host.fmt] = host.empty_like(self.km.device)
        with torch.cuda.stream(self.iostream):
            if self.time_h2d:
                h0 = torch.cuda.Event(enable_timing=True)
                h0.record(self.iostream)
            st.copy_from(host, non_blocking=True)
            ev = torch.cuda.Event(enable_timing=self.time_h2d)
            ev.record(self.iostream)
        sleep_until(done)
        self.last_copy_bound = not ev.query()
        with torch.cuda.stream(self.iostream):      # 資料到齊之後才反量化（排在 H2D 之後，同一個 stream）
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record(self.iostream)
            decode(st, out=self.km.chunk_view(i))
            e1.record(self.iostream)
        e1.synchronize()
        self.last_dequant_s = e0.elapsed_time(e1) / 1e3
        self.last_h2d_s = h0.elapsed_time(ev) / 1e3 if self.time_h2d else 0.0
        self.last_nbytes = host.nbytes
        return done

    @staticmethod
    def _src(host) -> str:
        return "load_" + (host.kind if isinstance(host, Enc) else "bf16")

    def _note_load(self, rec, i, host, ts, te):
        src = self._src(host)
        rec[f"n_{src}"] = rec.get(f"n_{src}", 0) + 1
        rec["bytes_read"] += self.last_nbytes
        rec["t_dequant"] += self.last_dequant_s
        rec["t_h2d"] += self.last_h2d_s
        rec["n_copy_bound"] += int(self.last_copy_bound)
        rec["blocks"].append((i, src, ts, te))

    def _recompute(self, ids, i: int):
        with torch.cuda.stream(self.cstream):
            self.km.forward_span(ids[i * CHUNK:(i + 1) * CHUNK], i * CHUNK)
        self.cstream.synchronize()

    def restore(self, ids: torch.Tensor, n: int, host: list, loc: list, mode: str = "cake") -> dict:
        """還原前 n 個 chunk 的 KV 到 GPU。回傳時間拆解（秒）。"""
        t0 = time.perf_counter()
        rec = {"mode": mode, "n_chunks": n, "n_recompute": 0, "n_load_cpu": 0, "n_load_ssd": 0,
               "t_recompute": 0.0, "t_load": 0.0, "t_load_cpu": 0.0, "t_load_ssd": 0.0,
               "n_load_bf16": 0, "n_load_fp8": 0, "n_load_int8": 0, "n_load_int4": 0,
               "bytes_read": 0, "t_dequant": 0.0, "t_h2d": 0.0, "n_copy_bound": 0, "blocks": []}
        if mode == "compute_only":
            for i in range(n):
                ts = time.perf_counter() - t0
                self._recompute(ids, i)
                rec["blocks"].append((i, "recompute", ts, time.perf_counter() - t0))
            rec["n_recompute"] = n
            rec["t_recompute"] = time.perf_counter() - t0
            rec["meet"] = n
        elif mode == "load_only":
            i = 0
            while i < n and loc[i] is not None:
                ts = time.perf_counter()
                self._load(i, host[i], loc[i])
                dt = time.perf_counter() - ts
                key = "cpu" if loc[i].name.startswith("cpu") else "ssd"
                rec[f"n_load_{key}"] += 1
                rec[f"t_load_{key}"] += dt
                self._note_load(rec, i, host[i], ts - t0, ts + dt - t0)
                i += 1
            rec["t_load"] = time.perf_counter() - t0
            t1 = time.perf_counter()
            for j in range(i, n):
                ts = time.perf_counter() - t0
                self._recompute(ids, j)
                rec["blocks"].append((j, "recompute", ts, time.perf_counter() - t0))
            rec["n_recompute"] = n - i
            rec["t_recompute"] = time.perf_counter() - t1
            rec["meet"] = i
        elif mode == "cake":
            st = {"p": 0, "q": n - 1}
            lk = threading.Lock()
            times = {}

            def rline():
                ts = time.perf_counter()
                while True:
                    with lk:
                        if st["p"] > st["q"]:
                            break
                        i = st["p"]
                        st["p"] += 1
                    ts1 = time.perf_counter() - t0
                    self._recompute(ids, i)
                    rec["blocks"].append((i, "recompute", ts1, time.perf_counter() - t0))
                    rec["n_recompute"] += 1
                times["r_end"] = time.perf_counter()
                rec["t_recompute"] = times["r_end"] - ts

            def lline():
                ts = time.perf_counter()
                while True:
                    with lk:
                        if st["q"] < st["p"]:
                            break
                        q = st["q"]
                        if loc[q] is None:
                            break   # 沒存的只能留給重算線
                        st["q"] -= 1
                    ts1 = time.perf_counter()
                    self._load(q, host[q], loc[q])
                    te1 = time.perf_counter()
                    key = "cpu" if loc[q].name.startswith("cpu") else "ssd"
                    rec[f"n_load_{key}"] += 1
                    rec[f"t_load_{key}"] += te1 - ts1
                    self._note_load(rec, q, host[q], ts1 - t0, te1 - t0)
                times["l_end"] = time.perf_counter()
                rec["t_load"] = times["l_end"] - ts

            th = threading.Thread(target=lline)
            th.start()
            rline()
            th.join()
            rec["meet"] = rec["n_recompute"]
            rec["t_wait"] = abs(times["r_end"] - times["l_end"])
            rec["waiting_line"] = "recompute" if times["r_end"] < times["l_end"] else "load"
        else:
            raise ValueError(mode)
        torch.cuda.current_stream().wait_stream(self.iostream)
        torch.cuda.current_stream().wait_stream(self.cstream)
        torch.cuda.synchronize()
        rec["t_restore"] = time.perf_counter() - t0
        return rec

    def new_turn(self, new_ids: torch.Tensor, start: int) -> tuple[float, int]:
        """算新一輪的 token（≤512 一段，否則以 512 切），回傳（秒, 第一個輸出 token）。"""
        t0 = time.perf_counter()
        n = new_ids.shape[-1]
        s = 0
        logits = None
        while s < n:
            e = min(n, s + CHUNK)
            logits = self.km.forward_span(new_ids[s:e], start + s, want_logits=(e == n))
            s = e
        tok = int(logits.argmax())
        torch.cuda.synchronize()
        return time.perf_counter() - t0, tok


# ---------------------------------------------------------------------------
# 演算法 1：寫入分界 b
def write_boundary(n: int, f: list[float], ell: float) -> int:
    """回傳使 max(sum f[0:m], (n-m)*ell) 最小的 m（預期重算的前段 chunk 數）。"""
    best, best_m, R = None, 0, 0.0
    for m in range(0, n + 1):
        if m > 0:
            R += f[m - 1]
        T = max(R, (n - m) * ell)
        if best is None or T < best:
            best, best_m = T, m
    return best_m
