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
DQ_KERNEL = "k1"          # 09 §6.4 K 組：m10_dequant_fast.install() 設定；只有 S2／S4／S5（一次拆多個 block）會讀


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
        # 09 §6.4 方法掃描的選項（預設＝原本的行為，逐位元、逐步驟相同）：
        #   unit：還原單位（token；G 組），sched：serial／pipe／group／defer／defer_meet（S 組），
        #   dq_group：group 的 k，meet_rule：claim（原本）／estimate（G4 預估會合）；重算優先（S3）用 set_options。
        # 13 號計畫（10/17）另加：meet_rule none（全部載入、不重算）／g5（開始前估計，載完比重算第 0 塊快就不重算）；
        #   sched nodq（只搬不拆：拆融合進 attention 的對照組，KV 緩衝不會被填）；io_prio（還原的 H2D、拆用最高優先權的 stream）。
        self.unit = CHUNK
        self.dq_sched = "serial"
        self.dq_group = 1
        self.meet_rule = "claim"
        self.compute_prio = False
        self.io_prio = False
        self.g5_key = None                     # G5 的紀錄依（頻寬, GPU 狀態）分開；呼叫端每次還原前設定
        self.g5_force_claim = False            # G5 的 profiler：照 Cake 跑、只記錄
        self._g5_hist: dict = {}
        self._h2d_s = None
        self._stages: dict = {}
        self._arange = None

    def set_options(self, unit: int = CHUNK, sched: str = "serial", group: int = 1, compute_prio: bool = False,
                    meet: str = "claim", io_prio: bool = False):
        assert CHUNK % unit == 0 and unit % 32 == 0, unit
        assert sched in ("serial", "pipe", "group", "defer", "defer_meet", "nodq"), sched
        assert meet in ("claim", "estimate", "none", "g5", "g5b"), meet
        assert group >= 1 and (group == 1 or sched == "group")
        assert not (meet in ("g5", "g5b") and sched in ("defer", "defer_meet", "nodq")), "G5 的載入時間要含拆（13 §4 只配 S0／S2）"
        self.unit, self.dq_sched, self.dq_group, self.meet_rule = unit, sched, group, meet
        if compute_prio and not self.compute_prio:
            lo, hi = torch.cuda.Stream.priority_range()          # 數字越小優先權越高
            self.cstream = torch.cuda.Stream(device=self.km.device, priority=hi)
        self.compute_prio = compute_prio
        if io_prio and not self.io_prio:                         # 13 §3 P1：還原的 H2D、拆都在這兩條 stream 上
            lo, hi = torch.cuda.Stream.priority_range()
            self.iostream = torch.cuda.Stream(device=self.km.device, priority=hi)
            self._h2d_s = torch.cuda.Stream(device=self.km.device, priority=hi)
        self.io_prio = io_prio

    def _view(self, i: int) -> torch.Tensor:
        if self.unit == CHUNK:
            return self.km.chunk_view(i)
        u = self.unit
        return self.km.kv[:, :, :, i * u:(i + 1) * u, :]

    def _ubytes(self) -> int:
        return self.km.chunk_bytes if self.unit == CHUNK else self.km.chunk_bytes * self.unit // CHUNK

    # ---- 載入一個 chunk（限速＋真實 H2D）----
    def _load(self, i: int, host, tier: Tier) -> float:
        if isinstance(host, Enc) and host.fmt != "bf16":
            return self._load_enc(i, host, tier)
        if isinstance(host, Enc):
            host = host.parts["x"]
        done = tier.reserve_read(self._ubytes())
        if self.time_h2d:
            h0 = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(self.iostream):
                h0.record(self.iostream)
                self._view(i).copy_(host, non_blocking=True)
                ev = torch.cuda.Event(enable_timing=True)
                ev.record(self.iostream)
        else:
            with torch.cuda.stream(self.iostream):
                self._view(i).copy_(host, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(self.iostream)
        sleep_until(done)
        self.last_copy_bound = not ev.query()
        ev.synchronize()
        self.last_h2d_s = h0.elapsed_time(ev) / 1e3 if self.time_h2d else 0.0
        self.last_dequant_s = 0.0
        self.last_nbytes = self._ubytes()
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
            decode(st, out=self._view(i))
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
        u = self.unit
        with torch.cuda.stream(self.cstream):
            self.km.forward_span(ids[i * u:(i + 1) * u], i * u)
        self.cstream.synchronize()

    # ---- 09 §6.4 方法掃描：S 組（何時反量化）與 G4（預估會合）。只在 set_options 改了預設時才走這裡 ----
    def _stage(self, key, make):
        st = self._stages.get(key)
        if st is None:
            st = self._stages[key] = make()
        return st

    def _h2dstream(self):
        if self._h2d_s is None:
            self._h2d_s = torch.cuda.Stream(device=self.km.device)
        return self._h2d_s

    def _ev(self):
        return torch.cuda.Event(enable_timing=True)

    def _load_pipe(self, q: int, e: Enc, tier: Tier, pend: list) -> bool:
        """S1 邊搬邊拆：H2D 在自己的 stream、反量化在 I/O stream；兩塊暫存區輪流用，不每塊等反量化做完。"""
        done = tier.reserve_read(e.nbytes)
        ring = self._stage(("pipe", e.fmt, e.shape), lambda: {"slots": [e.empty_like(self.km.device) for _ in range(2)],
                                                              "free": [None, None], "i": 0})
        k = ring["i"] % 2
        ring["i"] += 1
        slot, hs = ring["slots"][k], self._h2dstream()
        with torch.cuda.stream(hs):
            if ring["free"][k] is not None:
                hs.wait_event(ring["free"][k])             # 這塊暫存區上一次的反量化做完才能覆寫（GPU 端等，主機不等）
            h0, h1 = self._ev(), self._ev()
            h0.record(hs)
            slot.copy_from(e, non_blocking=True)
            h1.record(hs)
        sleep_until(done)
        cb = not h1.query()
        with torch.cuda.stream(self.iostream):
            self.iostream.wait_event(h1)
            e0, e1 = self._ev(), self._ev()
            e0.record(self.iostream)
            decode(slot, out=self._view(q))
            e1.record(self.iostream)
        ring["free"][k] = e1
        pend.append((h0, h1, e0, e1))
        return cb

    def _load_group(self, blks: list, host: list, tier: Tier, pend: list) -> bool:
        """S2 多個 block 一組：連續的 nb 個同格式 block 一起搬，搬完一次 kernel 拆完（寫到各自位置），等它做完再領下一組。"""
        import m10_dequant_fast as DF
        blks = sorted(blks)
        nb, e = len(blks), host[blks[0]]
        done = 0.0
        for j in blks:
            done = tier.reserve_read(host[j].nbytes)
        g = self._stage(("group", e.fmt, e.shape, self.dq_group), lambda: DF.group_parts(e, self.dq_group, self.km.device))
        if self._arange is None:
            self._arange = torch.arange(self.km.max_len // self.unit + 1, dtype=torch.int64, device=self.km.device)
        with torch.cuda.stream(self.iostream):
            h0, h1 = self._ev(), self._ev()
            h0.record(self.iostream)
            for s_, j in enumerate(blks):
                DF.sub_enc(host[j], g, s_).copy_from(host[j], non_blocking=True)
            h1.record(self.iostream)
        sleep_until(done)
        cb = not h1.query()
        with torch.cuda.stream(self.iostream):
            e0, e1 = self._ev(), self._ev()
            e0.record(self.iostream)
            DF.decode_multi(e.fmt, e.shape, {k: t[:nb] for k, t in g.items()}, nb, self.km.kv,
                            self._arange[blks[0]:blks[0] + nb], **DF.kernel_cfg(DQ_KERNEL))
            e1.record(self.iostream)
        e1.synchronize()
        pend.append((h0, h1, e0, e1))
        return cb

    def _load_defer(self, q: int, e: Enc, tier: Tier, defer: dict, n: int, pend: list) -> bool:
        """S4／S5：只搬、不拆；低精度 block 依搬的順序放進暫存區，最後一次 kernel 拆完（_flush_defer）。"""
        import m10_dequant_fast as DF
        done = tier.reserve_read(e.nbytes)
        g = self._stage(("defer", e.fmt, e.shape, n), lambda: DF.group_parts(e, n, self.km.device))
        lst = defer.setdefault((e.fmt, e.shape), [])
        with torch.cuda.stream(self.iostream):
            h0, h1 = self._ev(), self._ev()
            h0.record(self.iostream)
            DF.sub_enc(e, g, len(lst)).copy_from(e, non_blocking=True)
            h1.record(self.iostream)
        lst.append(q)
        sleep_until(done)
        pend.append((h0, h1, None, None))
        return not h1.query()

    def _flush_defer(self, defer: dict, n: int, pend: list, keep: list):
        import m10_dequant_fast as DF
        for (fmt, shape), lst in defer.items():
            if not lst:
                continue
            g = self._stages[("defer", fmt, shape, n)]
            bids = torch.tensor(lst, dtype=torch.int64).pin_memory()
            keep.append(bids)                                  # 非同步 H2D 完成前不能被回收
            with torch.cuda.stream(self.iostream):
                bd = bids.to(self.km.device, non_blocking=True)
                e0, e1 = self._ev(), self._ev()
                e0.record(self.iostream)
                DF.decode_multi(fmt, shape, {k: t[:len(lst)] for k, t in g.items()}, len(lst), self.km.kv, bd,
                                **DF.kernel_cfg(DQ_KERNEL))
                e1.record(self.iostream)
            keep.append(bd)
            pend.append((None, None, e0, e1))

    def _cake_v2(self, ids, n: int, host: list, loc: list, rec: dict, t0: float):
        st = {"p": 0, "q": n - 1}
        lk = threading.Lock()
        times = {}
        est = {"r": [], "l": []}                               # 這次還原裡每塊重算、每塊載入的牆鐘（G4 用）
        sched = self.dq_sched
        K = self.dq_group if sched == "group" else 1
        pend, defer, keep = [], {}, []
        skip_r = False                                         # 13 §2／§4：none、g5 決定不重算
        if self.meet_rule in ("none", "g5", "g5b"):
            stored = all(loc[j] is not None for j in range(n))
            if not stored:
                rec["meet_rule_used"] = "claim(unstored)"
            elif self.meet_rule == "none":
                skip_r, rec["meet_rule_used"] = True, "none"
            elif self.g5_force_claim:
                rec["meet_rule_used"] = "claim(profile)"
            else:
                d = self._g5_decide(n, host)
                skip_r = d is True
                rec["meet_rule_used"] = {True: "none", False: "claim", None: "claim(no_history)"}[d]
        lt = rec.setdefault("_lt", [])                         # (來源, 每塊載入時間)：G5 的紀錄用

        def is_lp(h):
            return isinstance(h, Enc) and h.fmt != "bf16"

        def rline():
            ts = time.perf_counter()
            while not skip_r:
                with lk:
                    if st["p"] > st["q"]:
                        break
                    if self.meet_rule == "estimate" and est["r"] and est["l"] \
                            and all(loc[j] is not None for j in range(st["p"], st["q"] + 1)):
                        rem = st["q"] - st["p"] + 1
                        if rem * (sum(est["l"]) / len(est["l"])) <= sum(est["r"]) / len(est["r"]):
                            break                              # 載入線把剩下的做完比較快 → 重算線不再開新的一塊
                    i = st["p"]
                    st["p"] += 1
                ts1 = time.perf_counter() - t0
                self._recompute(ids, i)
                te1 = time.perf_counter() - t0
                est["r"].append(te1 - ts1)
                rec["blocks"].append((i, "recompute", ts1, te1))
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
                        break
                    grp = [q]
                    if K > 1 and is_lp(host[q]):
                        while len(grp) < K:
                            j = grp[-1] - 1
                            if j < st["p"] or loc[j] is not loc[q] or not is_lp(host[j]) or host[j].fmt != host[q].fmt:
                                break
                            grp.append(j)
                    st["q"] -= len(grp)
                ts1 = time.perf_counter()
                if not is_lp(host[q]) or sched == "serial":
                    self._load(q, host[q], loc[q])
                    self._note_load(rec, q, host[q], ts1 - t0, time.perf_counter() - t0)
                    cb = None
                elif sched == "pipe":
                    cb = self._load_pipe(q, host[q], loc[q], pend)
                elif sched == "group":
                    cb = self._load_group(grp, host, loc[q], pend)
                else:
                    cb = self._load_defer(q, host[q], loc[q], defer, n, pend)
                te1 = time.perf_counter()
                if cb is not None:                             # 低精度 block 的記帳（時間由 event 事後算）
                    for j in grp:
                        src = self._src(host[j])
                        rec[f"n_{src}"] = rec.get(f"n_{src}", 0) + 1
                        rec["bytes_read"] += host[j].nbytes
                        rec["blocks"].append((j, src, ts1 - t0, te1 - t0))
                    rec["n_copy_bound"] += int(cb)
                key = "cpu" if loc[q].name.startswith("cpu") else "ssd"
                rec[f"n_load_{key}"] += len(grp)
                rec[f"t_load_{key}"] += te1 - ts1
                est["l"].extend([(te1 - ts1) / len(grp)] * len(grp))
                lt.extend([(self._src(host[q]), (te1 - ts1) / len(grp))] * len(grp))
            if sched == "defer":
                self._flush_defer(defer, n, pend, keep)
            self.iostream.synchronize()                        # nodq：只搬不拆，暫存區留著（13 §6 (a)）
            times["l_end"] = time.perf_counter()
            rec["t_load"] = times["l_end"] - ts

        th = threading.Thread(target=lline)
        th.start()
        rline()
        th.join()
        if sched == "defer_meet":                              # S5：等重算線結束才拆
            self._flush_defer(defer, n, pend, keep)
            self.iostream.synchronize()
            times["l_end"] = time.perf_counter()
        for h0, h1, e0, e1 in pend:
            if e0 is not None:
                rec["t_dequant"] += e0.elapsed_time(e1) / 1e3
            if h0 is not None and self.time_h2d:
                rec["t_h2d"] += h0.elapsed_time(h1) / 1e3
        rec["meet"] = rec["n_recompute"]
        rec["t_wait"] = abs(times["r_end"] - times["l_end"])
        rec["waiting_line"] = "recompute" if times["r_end"] < times["l_end"] else "load"
        if est["r"]:
            rec["_r0"] = est["r"][0]                           # 第 0 塊的重算時間（重算線一定先拿第 0 塊）
        if self.meet_rule in ("g5", "g5b"):
            self._g5_note(rec)

    # ---- 13 §4 G5：開始前估計「全部載完」和「重算第 0 塊」 ----
    G5_KEEP = 20

    def _g5_decide(self, n: int, host: list):
        h = self._g5_hist.get(self.g5_key)
        if not h or not h["r0"]:
            return None
        cnt = {}
        for j in range(n):
            s_ = self._src(host[j])
            cnt[s_] = cnt.get(s_, 0) + 1
        if any(not h["l"].get(s_) for s_ in cnt):
            return None
        import statistics
        lmed = {s_: statistics.median(h["l"][s_]) for s_ in cnt}
        load_all = sum(c * lmed[s_] for s_, c in cnt.items())
        r = statistics.median(h["r0"])
        if self.meet_rule == "g5":                             # 13 §4：和「重算第 0 塊」比
            return load_all <= r
        # 13 §8c G5b：用同一組估計模擬 Cake 的兩條線（重算線從前面一塊一塊拿、每塊 r；載入線從後面拿、每塊照它的來源），
        # 和「全部載入」比。Cake 會在載入線還沒做完時多開一塊重算（整數跳動），只比第 0 塊會低估 Cake 的時間。
        tr = tl = 0.0
        p_, q_ = 0, n - 1
        while p_ <= q_:
            if tr <= tl:
                tr += r
                p_ += 1
            else:
                tl += lmed[self._src(host[q_])]
                q_ -= 1
        return load_all <= max(tr, tl)

    def _g5_note(self, rec: dict):
        h = self._g5_hist.setdefault(self.g5_key, {"l": {}, "r0": []})
        for s_, t in rec.get("_lt", []):
            h["l"].setdefault(s_, []).append(t)
            del h["l"][s_][:-self.G5_KEEP * 64]
        if "_r0" in rec:
            h["r0"].append(rec["_r0"])
            del h["r0"][:-self.G5_KEEP]

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
        elif mode == "cake" and (self.dq_sched != "serial" or self.meet_rule != "claim"):
            self._cake_v2(ids, n, host, loc, rec, t0)
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
