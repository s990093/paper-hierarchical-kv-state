"""ledger.py — 記帳（04 §4）：每層每 1 秒窗口的讀寫位元組與尖峰 p99（預取也算）、總讀寫量、品質帳、回來請求的 TTFT。

定義（06_simulator.md §4）：
  * 窗口：[t0, t1) 切成 1 秒；每筆傳輸的位元組依它在 I/O 線上**實際佔用**的時間 [開始, 結束) 平均攤到各秒
    （排隊造成的延後算在實際傳輸的那一秒）。t0＝第一個請求、t1＝最後一筆傳輸結束（向上取整）。
    尖峰＝所有窗口（含 0）的 p99 與最大值，單位 GiB/s。讀、寫分開，另給讀＋寫。
  * 截尾版：只算 [t0, t_obs_end − H) 的窗口、與到達時刻 < t_obs_end − H 的請求。
  * 總量：I/O 線上的位元組（Chan.bytes_r／bytes_w）。引擎另有依原因分的整數計數（store／demote／write_through／selective／ttl_demote／prefetch）。
  * 品質帳：每個「有載入任何 block」的請求記一次讀取；這次讀取的格式＝載入的 block 裡最差的品質來歷；長度＝存的前綴 token 數 → 分箱；
    狀態照格式表（pass／fail／unverified）。fail 的次數＝「品質沒過關的讀取」。回來的請求（kind＝1）另外列。
  * TTFT：回來的請求（kind＝1）；平均、p50、p90、p99。沒有 f(i) 或解壓常數的 view 標 NOT_MEASURED。
"""
from __future__ import annotations

import math

import numpy as np

from formats import FMT_NAMES

GiB = float(1 << 30)


def spread_bytes(st, en, nb, t0, nwin):
    """每筆 [st, en) 的位元組平均攤到 1 秒窗口；回傳長度 nwin 的陣列。"""
    out = np.zeros(nwin)
    if len(nb) == 0:
        return out
    st = np.asarray(st, float) - t0
    en = np.asarray(en, float) - t0
    nb = np.asarray(nb, float)
    ws, we = np.floor(st).astype(np.int64), np.floor(en).astype(np.int64)
    same = (ws == we) | (en <= st)
    idx = np.clip(ws[same], 0, nwin - 1)
    np.add.at(out, idx, nb[same])
    for i in np.nonzero(~same)[0]:
        rate = nb[i] / (en[i] - st[i])
        a, b = st[i], en[i]
        k = int(math.floor(a))
        while a < b and k < nwin:
            nxt = min(b, k + 1.0)
            if k >= 0:
                out[k] += rate * (nxt - a)
            a, k = nxt, k + 1
    return out


def window_stats(ch, t0, t1, prefix):
    nwin = max(1, int(math.ceil(t1 - t0)))
    r = spread_bytes(ch.rs, ch.re, ch.rb, t0, nwin)
    w = spread_bytes(ch.ws, ch.we, ch.wb, t0, nwin)
    rk = np.frombuffer(ch.rk, dtype=np.int8) if len(ch.rk) else np.zeros(0, np.int8)
    pf = np.nonzero(rk == 1)[0]
    rpf = spread_bytes(np.asarray(ch.rs)[pf], np.asarray(ch.re)[pf], np.asarray(ch.rb)[pf], t0, nwin) if len(pf) else np.zeros(nwin)
    out = {}
    for nm, a in (("read", r), ("write", w), ("rw", r + w), ("read_prefetch", rpf)):
        out[f"{prefix}_{nm}_p99_GiBps"] = float(np.percentile(a, 99)) / GiB
        out[f"{prefix}_{nm}_max_GiBps"] = float(a.max()) / GiB
        out[f"{prefix}_{nm}_sum_GiB"] = float(a.sum()) / GiB
    out[f"{prefix}_n_windows"] = nwin
    return out, (r, w)


def _pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def summarize_view(P, eng, v, H=0.0):
    kd = np.asarray(P["kindc"])
    tl = np.asarray(P["tl"])
    t0 = float(tl[0]) if len(tl) else 0.0
    ends = [t0 + 1.0]
    for ch in (v.cpu, v.slow):
        if len(ch.re):
            ends.append(max(ch.re))
        if len(ch.we):
            ends.append(max(ch.we))
    t1 = max(ends)
    row = dict(view=v.label, platform=v.key[0], gpu=v.gpu, cpu_factor=v.cpu_factor, ssd_share=v.ssd_share, restore=v.mode,
               ttft_valid=bool(v.has_f and v.dq_ok))
    row.update(cpu_read_GiB=v.cpu.bytes_r / GiB, cpu_write_GiB=v.cpu.bytes_w / GiB,
               ssd_read_GiB=v.slow.bytes_r / GiB, ssd_write_GiB=v.slow.bytes_w / GiB,
               ssd_rw_GiB=(v.slow.bytes_r + v.slow.bytes_w) / GiB)
    for ch, nm in ((v.cpu, "cpu"), (v.slow, "ssd")):
        st, _ = window_stats(ch, t0, t1, nm)
        row.update(st)
    # 截尾版（H>0）：窗口與請求都只到 t_obs_end − H
    if H > 0:
        oe = float(max(P["obs_end"])) if P.get("obs_end") else float(tl[-1])
        tc = oe - H
        row["censor_cut_s"] = tc
        for ch, nm in ((v.cpu, "cpu"), (v.slow, "ssd")):
            if tc > t0:
                st, (r, w) = window_stats(ch, t0, max(t1, tc), nm + "_H")
                n = max(1, int(math.ceil(tc - t0)))
                rr, ww = r[:n], w[:n]
                row[f"{nm}_H_read_GiB"] = float(rr.sum()) / GiB
                row[f"{nm}_H_write_GiB"] = float(ww.sum()) / GiB
                row[f"{nm}_H_rw_p99_GiBps"] = float(np.percentile(rr + ww, 99)) / GiB
                row[f"{nm}_H_read_p99_GiBps"] = float(np.percentile(rr, 99)) / GiB
                row[f"{nm}_H_write_p99_GiBps"] = float(np.percentile(ww, 99)) / GiB
    # TTFT（回來的請求）
    ret = kd == 1
    if H > 0:
        retH = ret & (tl < row["censor_cut_s"])
    x = v.ttft[ret]
    if row["ttft_valid"]:
        row.update(n_ret=int(ret.sum()), ttft_mean_s=float(x.mean()) if len(x) else float("nan"), ttft_p50_s=_pct(x, 50),
                   ttft_p90_s=_pct(x, 90), ttft_p99_s=_pct(x, 99), restore_mean_s=float(v.rest[ret].mean()) if len(x) else float("nan"))
        if H > 0:
            xh = v.ttft[retH]
            row.update(n_ret_H=int(retH.sum()), ttft_H_mean_s=float(xh.mean()) if len(xh) else float("nan"), ttft_H_p99_s=_pct(xh, 99))
    else:
        row.update(n_ret=int(ret.sum()), ttft_mean_s="NOT_MEASURED", ttft_p50_s="NOT_MEASURED", ttft_p90_s="NOT_MEASURED",
                   ttft_p99_s="NOT_MEASURED", restore_mean_s="NOT_MEASURED")
    row.update(ret_cpu_loads=int(v.nC[ret].sum()), ret_ssd_loads=int(v.nS[ret].sum()), ret_recompute=int(v.nR[ret].sum()))
    # 品質帳
    q = quality_ledger(eng, v, kd)
    row.update(q["counts"])
    return row, q["table"]


def quality_ledger(eng, v, kd):
    mf = eng.mf
    loaded = (v.nC + v.nS) > 0
    tab = {}
    cnt = dict(reads=0, reads_fail=0, reads_unverified=0, reads_pass=0, ret_reads=0, ret_reads_fail=0, ret_reads_unverified=0)
    Lt = eng.Ltok
    for r in np.nonzero(loaded)[0]:
        f = int(v.wq[r])
        b = mf.bin_of(int(Lt[r]))
        stt = mf.status(f, int(Lt[r]))
        key = (FMT_NAMES[f], b, stt, int(kd[r] == 1))
        tab[key] = tab.get(key, 0) + 1
        cnt["reads"] += 1
        cnt[f"reads_{stt}"] += 1
        if kd[r] == 1:
            cnt["ret_reads"] += 1
            if stt in ("fail", "unverified"):
                cnt[f"ret_reads_{stt}"] += 1
    table = [dict(view=v.label, read_fmt=k[0], len_bin=k[1], len_lo=mf.len_bins[k[1]], len_hi=mf.len_bins[k[1] + 1],
                  status=k[2], returning=k[3], n=n) for k, n in sorted(tab.items())]
    return dict(counts=cnt, table=table)


def engine_row(eng):
    """放置軌跡的統計（和 view 無關）。"""
    if hasattr(eng, "S"):       # block 引擎
        row = {f"blk_{k}": v for k, v in eng.S.items()}
    else:
        row = dict(n_demote=eng.n_demote, n_drop=eng.n_drop, n_ttl_demote=eng.n_ttl, n_lost_blocks=eng.n_lost)
    for k, x in eng.wbytes.items():
        row[f"wbytes_{k}_GiB"] = x / GiB
    for (tier, fid), n in sorted(eng.wblocks_fmt.items()):
        row[f"wblocks_{tier}_{FMT_NAMES[fid]}"] = n
    return row


def accounting_identities(P, eng, v):
    """記帳要加得起來（冒煙與測試共用）：回傳 [(名稱, 左, 右, 是否相等)]。"""
    if hasattr(eng, "S"):
        return _identities_block(eng, v)
    ch = []
    # (1) SSD 寫入（I/O 線）＝各原因的寫入位元組加總（demote＋write_through＋selective＋ttl_demote）
    w = eng.wbytes
    ssd_w = w["demote"] + w["write_through"] + w["selective"] + w["ttl_demote"]
    ch.append(("ssd_write_bytes==sum(causes)", v.slow.bytes_w, ssd_w if "ssd" in eng.outer["tiers"] else 0))
    # (2) CPU 寫入＝store
    ch.append(("cpu_write_bytes==store", v.cpu.bytes_w, w["store"]))
    # (3) SSD 寫入＝每個格式寫入 block 數 × 該格式大小
    sb = sum(n * eng.fb[f] for (t, f), n in eng.wblocks_fmt.items() if t == "ssd")
    ch.append(("ssd_write_bytes==sum(blocks×fmt_bytes)", v.slow.bytes_w, sb if "ssd" in eng.outer["tiers"] else 0))
    cb = sum(n * eng.fb[f] for (t, f), n in eng.wblocks_fmt.items() if t == "cpu")
    ch.append(("cpu_write_bytes==sum(blocks×fmt_bytes)", v.cpu.bytes_w, cb))
    # (4) 逐筆紀錄的加總＝I/O 線的總量
    ch.append(("ssd_read_records==bytes_r", int(sum(v.slow.rb)), v.slow.bytes_r))
    ch.append(("ssd_write_records==bytes_w", int(sum(v.slow.wb)), v.slow.bytes_w))
    ch.append(("cpu_read_records==bytes_r", int(sum(v.cpu.rb)), v.cpu.bytes_r))
    # (5) 讀到的 block 數 × 大小 ≥ 0、且 SSD 讀取＝每個 SSD 載入的 block 的大小加總（逐請求重算）
    return [(n, a, b, a == b) for n, a, b in ch]


def _identities_block(eng, v):
    w = eng.wbytes
    ch = [("ssd_write_bytes==demote", v.slow.bytes_w, w["demote"]),
          ("cpu_write_bytes==store+promote", v.cpu.bytes_w, w["store"] + w["promote"])]
    sb = sum(n * eng.fb[f] for (t, f), n in eng.wblocks_fmt.items() if t == "ssd")
    cb = sum(n * eng.fb[f] for (t, f), n in eng.wblocks_fmt.items() if t == "cpu")
    ch.append(("ssd_write_bytes==sum(blocks×fmt_bytes)", v.slow.bytes_w, sb))
    ch.append(("cpu_write_bytes==sum(blocks×fmt_bytes)", v.cpu.bytes_w, cb))
    rk = np.frombuffer(v.slow.rk, dtype=np.int8) if len(v.slow.rk) else np.zeros(0, np.int8)
    rb = np.frombuffer(v.slow.rb, dtype=np.int64) if len(v.slow.rb) else np.zeros(0, np.int64)
    ch.append(("ssd_prefetch_read_records==prefetch_read", int(rb[rk == 1].sum()), w["prefetch_read"]))
    ch.append(("ssd_read_records==bytes_r", int(rb.sum()), v.slow.bytes_r))
    ch.append(("ssd_write_records==bytes_w", int(sum(v.slow.wb)), v.slow.bytes_w))
    ch.append(("cpu_read_records==bytes_r", int(sum(v.cpu.rb)), v.cpu.bytes_r))
    return [(n, a, b, a == b) for n, a, b in ch]
