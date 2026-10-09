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
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import torch

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

    # ---- 載入一個 chunk（限速＋真實 H2D）----
    def _load(self, i: int, host: torch.Tensor, tier: Tier) -> float:
        done = tier.reserve_read(self.km.chunk_bytes)
        with torch.cuda.stream(self.iostream):
            self.km.chunk_view(i).copy_(host, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self.iostream)
        sleep_until(done)
        ev.synchronize()
        return done

    def _recompute(self, ids, i: int):
        with torch.cuda.stream(self.cstream):
            self.km.forward_span(ids[i * CHUNK:(i + 1) * CHUNK], i * CHUNK)
        self.cstream.synchronize()

    def restore(self, ids: torch.Tensor, n: int, host: list, loc: list, mode: str = "cake") -> dict:
        """還原前 n 個 chunk 的 KV 到 GPU。回傳時間拆解（秒）。"""
        t0 = time.perf_counter()
        rec = {"mode": mode, "n_chunks": n, "n_recompute": 0, "n_load_cpu": 0, "n_load_ssd": 0,
               "t_recompute": 0.0, "t_load": 0.0, "t_load_cpu": 0.0, "t_load_ssd": 0.0}
        if mode == "compute_only":
            for i in range(n):
                self._recompute(ids, i)
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
                i += 1
            rec["t_load"] = time.perf_counter() - t0
            t1 = time.perf_counter()
            for j in range(i, n):
                self._recompute(ids, j)
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
                    self._recompute(ids, i)
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
                    key = "cpu" if loc[q].name.startswith("cpu") else "ssd"
                    rec[f"n_load_{key}"] += 1
                    rec[f"t_load_{key}"] += time.perf_counter() - ts1
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
