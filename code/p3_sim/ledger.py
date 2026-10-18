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
2026-10-18 新加的欄（06 §12；舊欄一個都沒改）：
  * cw（common window）：同一格的每個策略用同一個窗口範圍＝[trace 切片的起點, 切片的終點)（P["win"]），不是最後一筆傳輸的結束。
    窗口外的位元組另記 spill。served＝實際傳輸（受 SSD 頻寬限制）；off（offered）＝請求當下要的位元組（不受頻寬限制，記在要的那一刻）。
  * 飽和時間：served 的讀（寫）≥ 0.95 × SSD 讀（寫）頻寬的 1 秒窗口比例。排隊延遲：每筆 SSD 傳輸的「實際開始 − 準備好的時刻」。
  * load-only SSD 位元組：每個請求的有存前綴裡在 SSD 的 block 的位元組（只看放置；session 引擎和時間設定無關）。
  * ttftq_*：逐請求的 NOT_MEASURED 規則（只有真的載入了解壓常數 NOT_MEASURED 的格式的請求才算不能用）。
  * 0/0：沒有窗口（切片長度 0）→ 比例寫 NaN、n_windows＝0；沒有傳輸 → 延遲統計寫 NaN、n＝0；p99 的窗口包含 0 位元組的窗口。
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


def spread_in_range(st, en, nb, lo, nwin):
    """同 spread_bytes，但窗口外（< lo 或 ≥ lo+nwin）的位元組不塞進邊上的窗口，另外回傳 spill。"""
    out = np.zeros(nwin)
    if len(nb) == 0:
        return out, 0.0
    st = np.asarray(st, float) - lo
    en = np.asarray(en, float) - lo
    nb = np.asarray(nb, float)
    tot = float(nb.sum())
    if nwin == 0:
        return out, tot
    ws, we = np.floor(st).astype(np.int64), np.floor(en).astype(np.int64)
    same = (ws == we) | (en <= st)
    inr = same & (ws >= 0) & (ws < nwin)
    np.add.at(out, ws[inr], nb[inr])
    for i in np.nonzero(~same)[0]:
        rate = nb[i] / (en[i] - st[i])
        a, b = st[i], en[i]
        k = int(math.floor(a))
        if k < 0:
            a, k = 0.0, 0
        while a < b and k < nwin:
            nxt = min(b, k + 1.0)
            out[k] += rate * (nxt - a)
            a, k = nxt, k + 1
    return out, tot - float(out.sum())


def point_in_range(t, nb, lo, nwin):
    out = np.zeros(nwin)
    if len(nb) == 0:
        return out, 0.0
    nb = np.asarray(nb, float)
    k = np.floor(np.asarray(t, float) - lo).astype(np.int64)
    inr = (k >= 0) & (k < nwin)
    if nwin:
        np.add.at(out, k[inr], nb[inr])
    return out, float(nb[~inr].sum())


def win_range(P):
    """這一格的共同窗口 [lo, hi)（秒，和 P["tl"] 同一個原點）：trace.prepare 的 P["win"]；沒有時＝[第一個到達, 觀察窗終點)。"""
    w = P.get("win")
    if w is not None:
        return float(w[0]), float(w[1])
    tl = P["tl"]
    lo = float(tl[0]) if len(tl) else 0.0
    hi = float(max(P["obs_end"])) if P.get("obs_end") else (float(tl[-1]) if len(tl) else lo)
    return lo, max(hi, lo)


def _dstats(d, prefix, out):
    d = np.asarray(d, float)
    out[f"{prefix}_n"] = int(len(d))
    if len(d):
        out.update({f"{prefix}_mean_s": float(d.mean()), f"{prefix}_p50_s": float(np.percentile(d, 50)),
                    f"{prefix}_p99_s": float(np.percentile(d, 99)), f"{prefix}_max_s": float(d.max())})
    else:
        out.update({f"{prefix}_mean_s": float("nan"), f"{prefix}_p50_s": float("nan"), f"{prefix}_p99_s": float("nan"),
                    f"{prefix}_max_s": float("nan")})


def ssd_v4_stats(P, eng, v, H=0.0):
    """06 §12.2：共同窗口的 served／offered、飽和時間、排隊延遲（SSD 這一層）。"""
    ch = v.slow
    lo, hi = win_range(P)
    nwin = int(math.ceil(hi - lo)) if hi > lo else 0
    out = dict(win_lo_s=lo, win_hi_s=hi, cw_n_windows=nwin)
    r, rsp = spread_in_range(ch.rs, ch.re, ch.rb, lo, nwin)
    w, wsp = spread_in_range(ch.ws, ch.we, ch.wb, lo, nwin)
    rk = np.frombuffer(ch.rk, dtype=np.int8) if len(ch.rk) else np.zeros(0, np.int8)
    rs = np.frombuffer(ch.rs, dtype=np.float64) if len(ch.rs) else np.zeros(0)
    rb = np.frombuffer(ch.rb, dtype=np.int64) if len(ch.rb) else np.zeros(0, np.int64)
    rr = np.frombuffer(ch.rr, dtype=np.int32) if len(ch.rr) else np.zeros(0, np.int32)
    rd = np.frombuffer(ch.rd, dtype=np.float32).astype(np.float64) if len(ch.rd) else np.zeros(0)
    ws = np.frombuffer(ch.ws, dtype=np.float64) if len(ch.ws) else np.zeros(0)
    wb = np.frombuffer(ch.wb, dtype=np.int64) if len(ch.wb) else np.zeros(0, np.int64)
    wd = np.frombuffer(ch.wd, dtype=np.float32).astype(np.float64) if len(ch.wd) else np.zeros(0)
    # offered：還原的讀＝請求的時刻（session 引擎＝到達；block 引擎＝開始服務 t_go）；預取的讀、所有寫＝準備好的時刻（＝開始 − 排隊延遲）
    treq = np.asarray(getattr(v, "tgo", None) if getattr(v, "tgo", None) is not None else P["tl"], float)
    rot = np.where((rk == 0) & (rr >= 0), treq[np.clip(rr, 0, max(len(treq) - 1, 0))] if len(treq) else rs, rs - rd)
    ro, rosp = point_in_range(rot, rb, lo, nwin)
    wo, wosp = point_in_range(ws - wd, wb, lo, nwin)
    for nm, a in (("read", r), ("write", w), ("rw", r + w)):
        out[f"ssd_cw_{nm}_p99_GiBps"] = float(np.percentile(a, 99)) / GiB if nwin else float("nan")
        out[f"ssd_cw_{nm}_max_GiBps"] = float(a.max()) / GiB if nwin else float("nan")
    for nm, a in (("read", ro), ("write", wo), ("rw", ro + wo)):
        out[f"ssd_off_{nm}_p99_GiBps"] = float(np.percentile(a, 99)) / GiB if nwin else float("nan")
        out[f"ssd_off_{nm}_max_GiBps"] = float(a.max()) / GiB if nwin else float("nan")
    out.update(ssd_cw_read_spill_GiB=rsp / GiB, ssd_cw_write_spill_GiB=wsp / GiB,
               ssd_off_read_spill_GiB=rosp / GiB, ssd_off_write_spill_GiB=wosp / GiB)
    if nwin:
        sr, sw = r >= 0.95 * ch.r, w >= 0.95 * ch.w
        out.update(ssd_sat_read_frac=float(sr.mean()), ssd_sat_write_frac=float(sw.mean()), ssd_sat_any_frac=float((sr | sw).mean()))
    else:
        out.update(ssd_sat_read_frac=float("nan"), ssd_sat_write_frac=float("nan"), ssd_sat_any_frac=float("nan"))
    _dstats(rd, "ssd_qdelay_read", out)
    _dstats(wd, "ssd_qdelay_write", out)
    if H > 0:          # 截尾版：共同窗口只到 hi − H
        n = int(math.ceil(hi - H - lo)) if hi - H > lo else 0
        for nm, a in (("rw", r + w), ("read", r), ("write", w)):
            out[f"ssd_cwH_{nm}_p99_GiBps"] = float(np.percentile(a[:n], 99)) / GiB if n else float("nan")
        out["ssd_offH_rw_p99_GiBps"] = float(np.percentile((ro + wo)[:n], 99)) / GiB if n else float("nan")
        out["cwH_n_windows"] = n
    return out


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
    # ---- 06 §12：逐請求的 NOT_MEASURED 規則的 TTFT（ttftq_*）
    nmr = np.asarray(v.nmreq, bool)
    row["ttft_nm_ret_requests"] = int(nmr[ret].sum())
    okq = bool(v.has_f) and row["ttft_nm_ret_requests"] == 0
    row["ttftq_valid"] = okq
    if okq:
        row.update(ttftq_mean_s=float(x.mean()) if len(x) else float("nan"), ttftq_p50_s=_pct(x, 50), ttftq_p90_s=_pct(x, 90),
                   ttftq_p99_s=_pct(x, 99), restoreq_mean_s=float(v.rest[ret].mean()) if len(x) else float("nan"))
    else:
        row.update(ttftq_mean_s="NOT_MEASURED", ttftq_p50_s="NOT_MEASURED", ttftq_p90_s="NOT_MEASURED", ttftq_p99_s="NOT_MEASURED",
                   restoreq_mean_s="NOT_MEASURED")
    row["ssd_k_read_nm"] = bool(getattr(v, "k_nm", False))
    if row["ssd_k_read_nm"]:
        # 這個視角的 SSD k_read 沒量（06 §24）：碰到它的請求 ttftq＝NOT_MEASURED（整列也因此是 NOT_MEASURED）；另報「沒碰到的請求」的數字（占位 k=1.0 不影響它們）。
        # 其他由時間推出的欄位（SSD 忙碌、飽和、排隊、Cake 的 SSD 位元組）在這個視角用占位 k=1.0 算＝下界，不可當結論；位元組（寫入、只載入的讀取）和 k 無關。
        free = ret & ~nmr
        xf = v.ttft[free]
        row.update(ttft_nm_k_ret_requests=int(np.asarray(v.nmk, bool)[ret].sum()), ttftq_kfree_n=int(free.sum()),
                   ttftq_kfree_mean_s=float(xf.mean()) if len(xf) else float("nan"), ttftq_kfree_p99_s=_pct(xf, 99))
    if H > 0:
        okh = bool(v.has_f) and int(nmr[retH].sum()) == 0
        xh = v.ttft[retH]
        row["ttftq_H_mean_s"] = (float(xh.mean()) if len(xh) else float("nan")) if okh else "NOT_MEASURED"
        row["ttftq_H_p99_s"] = _pct(xh, 99) if okh else "NOT_MEASURED"
    # ---- 06 §12：SSD 的共同窗口、offered、飽和、排隊延遲；load-only 位元組；重算（前綴裡以前算過、這次重算的）
    row.update(ssd_v4_stats(P, eng, v, H))
    lo_b = np.asarray(eng.lo_bytes)
    row["ssd_loadonly_read_GiB"] = float(lo_b.sum()) / GiB
    row["ret_ssd_loadonly_read_GiB"] = float(lo_b[ret].sum()) / GiB
    if H > 0:      # 04 v4 §4 的主指標的截尾版：寫入＝決定時刻 < 終點 − H；load-only＝到達 < 終點 − H
        swH = eng.s_ssd_w_H.values() if isinstance(eng.s_ssd_w_H, dict) else eng.s_ssd_w_H
        row["ssd_H_write_issued_GiB"] = float(sum(swH)) / GiB
        row["ssd_H_loadonly_read_GiB"] = float(lo_b[tl < row["censor_cut_s"]].sum()) / GiB
        row["ret_H_ssd_loadonly_read_GiB"] = float(lo_b[retH].sum()) / GiB
    if getattr(eng, "cutH2", None) is not None:     # 第二個截尾（2H；放置不受影響，只換記帳的截斷點）
        swH2 = eng.s_ssd_w_H2.values() if isinstance(eng.s_ssd_w_H2, dict) else eng.s_ssd_w_H2
        row["censor_cut2_s"] = float(eng.cutH2)
        row["ssd_H2_write_issued_GiB"] = float(sum(swH2)) / GiB
        row["ssd_H2_loadonly_read_GiB"] = float(lo_b[tl < eng.cutH2].sum()) / GiB
        row["ret_H2_ssd_loadonly_read_GiB"] = float(lo_b[ret & (tl < eng.cutH2)].sum()) / GiB
    rec = np.asarray(v.nR, np.int64) - np.asarray(eng.nend, np.int64) + np.asarray(eng.nprev, np.int64)
    row["recompute_prefix_blocks"] = int(rec.sum())
    row["ret_recompute_prefix_blocks"] = int(rec[ret].sum())
    # 品質帳
    q = quality_ledger(eng, v, kd)
    row.update(q["counts"])
    return row, q["table"]


def quality_ledger(eng, v, kd):
    mf = eng.mf
    loaded = (v.nC + v.nS) > 0
    tab = {}
    cnt = dict(reads=0, reads_fail=0, reads_unverified=0, reads_pass=0, ret_reads=0, ret_reads_fail=0, ret_reads_unverified=0)
    Lt = eng.Lqual
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
    sb = np.asarray(eng.surp_blk)
    row.update(int2_on_read=eng.int2_on_read, unverified_len=eng.unverified_len, surprise_events=int((sb > 0).sum()), surprise_blocks=int(sb.sum()),
               surprise_tokens=int(sb.sum()) * eng.btok)
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
    # (5) 每個 session 的帳加得起來（06 §12.1）：SSD 寫入、讀取；寫入紀錄的 session 拆分
    ch.append(("sum(session ssd_write)==bytes_w", int(sum(eng.s_ssd_w)), v.slow.bytes_w if "ssd" in eng.outer["tiers"] else 0))
    ch.append(("sum(request ssd_read)==bytes_r", int(np.asarray(v.bS).sum()), v.slow.bytes_r))
    if v.slow.rec:
        ch.append(("sum(write splits)==bytes_w", int(sum(eng.wsp_b)), v.slow.bytes_w))
        ch.append(("n write records==n split records", len(v.slow.wb), eng.n_ssd_wrec))
        if len(eng.wsp_b):
            per = np.bincount(np.frombuffer(eng.wsp_rec, dtype=np.int64), weights=np.frombuffer(eng.wsp_b, dtype=np.int64).astype(float),
                              minlength=len(v.slow.wb))
            ch.append(("per-record split sum==record bytes", int(np.sum(per != np.frombuffer(v.slow.wb, dtype=np.int64))), 0))
    # (6) load-only 的時間設定：SSD 讀取＝load-only 位元組（每個有存前綴裡在 SSD 的 block 都讀）
    if v.mode == "load":
        ch.append(("load mode: bytes_r==sum(loadonly)", v.slow.bytes_r, int(np.asarray(eng.lo_bytes).sum())))
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
    ch.append(("sum(session ssd_write)==bytes_w", int(sum(eng.s_ssd_w.values())), v.slow.bytes_w))
    ch.append(("sum(request restore ssd_read)+prefetch==bytes_r", int(np.asarray(v.bS).sum()) + w["prefetch_read"], v.slow.bytes_r))
    return [(n, a, b, a == b) for n, a, b in ch]


# ================================================================== 06 §12.1：每個 session 的帳（給以 session 為單位的 bootstrap）
def per_session(P, eng, views):
    """回傳 dict：和時間設定無關的 (S,) 陣列，與每個時間設定一列的 (V, S) 陣列。session 的 index＝P["sessl"]。
    歸屬規則（06 §12.1）：寫進 CPU／寫穿／第 2 次才寫＝寫它的請求的 session；降級＝讓 block 離開 CPU 的那個 session
    （session 引擎：最後一個還在 CPU 參照它的 session；block 引擎：最後一次把它寫進 CPU 的請求的 session）；讀＝讀的 session；
    預取的讀＝替它預取的那個排隊中請求的 session。"""
    sl = np.asarray(P["sessl"], np.int64)
    S = int(P["R"]["n_sess"])
    kd = np.asarray(P["kindc"])
    mf = eng.mf

    def by(x, w=None):
        return np.bincount(sl, weights=None if w is None else np.asarray(w, float), minlength=S)

    out = dict(n_req=by(sl).astype(np.int64), n_ret=by(sl, kd == 1).astype(np.int64))
    if isinstance(eng.s_ssd_w, dict):
        sw = np.zeros(S, np.int64)
        for s_, nb in eng.s_ssd_w.items():
            if 0 <= s_ < S:
                sw[s_] += nb
    else:
        sw = np.asarray(eng.s_ssd_w, np.int64)
    out["ssd_write_bytes"] = sw
    sb = np.asarray(eng.surp_blk, np.int64)
    out["surprise_events"] = by(sl, sb > 0).astype(np.int64)
    out["surprise_blocks"] = by(sl, sb).astype(np.int64)
    out["surprise_tokens"] = out["surprise_blocks"] * eng.btok
    out["loadonly_ssd_read_bytes"] = by(sl, np.asarray(eng.lo_bytes, float)).astype(np.int64)
    out["ret_loadonly_ssd_read_bytes"] = by(sl, np.asarray(eng.lo_bytes, float) * (kd == 1)).astype(np.int64)
    # 截尾版（H>0；04 v4 §4 的主指標用的）：寫入＝決定時刻 < 終點 − H；load-only＝到達 < 終點 − H
    if getattr(eng, "cutH", None) is not None:
        swH = np.zeros(S, np.int64)
        items = eng.s_ssd_w_H.items() if isinstance(eng.s_ssd_w_H, dict) else enumerate(eng.s_ssd_w_H)
        for s_, nb in items:
            if 0 <= s_ < S:
                swH[s_] += nb
        tl_ = np.asarray(P["tl"], float)
        out["ssd_write_bytes_H"] = swH
        out["loadonly_ssd_read_bytes_H"] = by(sl, np.asarray(eng.lo_bytes, float) * (tl_ < eng.cutH)).astype(np.int64)
        out["ret_loadonly_ssd_read_bytes_H"] = by(sl, np.asarray(eng.lo_bytes, float) * ((tl_ < eng.cutH) & (kd == 1))).astype(np.int64)
        out["cutH_s"] = np.asarray(float(eng.cutH))
    if getattr(eng, "cutH2", None) is not None:
        sw2 = np.zeros(S, np.int64)
        items = eng.s_ssd_w_H2.items() if isinstance(eng.s_ssd_w_H2, dict) else enumerate(eng.s_ssd_w_H2)
        for s_, nb in items:
            if 0 <= s_ < S:
                sw2[s_] += nb
        tl_ = np.asarray(P["tl"], float)
        out["ssd_write_bytes_H2"] = sw2
        out["loadonly_ssd_read_bytes_H2"] = by(sl, np.asarray(eng.lo_bytes, float) * (tl_ < eng.cutH2)).astype(np.int64)
        out["ret_loadonly_ssd_read_bytes_H2"] = by(sl, np.asarray(eng.lo_bytes, float) * ((tl_ < eng.cutH2) & (kd == 1))).astype(np.int64)
        out["cutH2_s"] = np.asarray(float(eng.cutH2))
    V = len(views)
    keys = ("ssd_read_bytes", "ssd_prefetch_read_bytes", "recompute_tokens", "reads", "reads_pass", "reads_fail", "reads_unverified",
            "ret_reads", "ret_reads_fail", "ret_reads_unverified")
    for k in keys:
        out[k] = np.zeros((V, S), np.int64)
    for i, v in enumerate(views):
        out["ssd_read_bytes"][i] = by(sl, np.asarray(v.bS, float))
        rk = np.frombuffer(v.slow.rk, dtype=np.int8) if len(v.slow.rk) else np.zeros(0, np.int8)
        if len(rk) and (rk == 1).any():
            rr = np.frombuffer(v.slow.rr, dtype=np.int32)[rk == 1]
            rb = np.frombuffer(v.slow.rb, dtype=np.int64)[rk == 1]
            pf = np.bincount(sl[rr], weights=rb.astype(float), minlength=S).astype(np.int64)
            out["ssd_prefetch_read_bytes"][i] = pf
            out["ssd_read_bytes"][i] += pf
        rec = np.asarray(v.nR, np.int64) - np.asarray(eng.nend, np.int64) + np.asarray(eng.nprev, np.int64)
        out["recompute_tokens"][i] = by(sl, rec * eng.btok)
        loaded = (np.asarray(v.nC) + np.asarray(v.nS)) > 0
        Lt = np.asarray(eng.Lqual)
        st = np.array([mf.status(int(f), int(t)) if ld else "" for f, t, ld in zip(v.wq, Lt, loaded)], dtype=object)
        ret = kd == 1
        out["reads"][i] = by(sl, loaded)
        for s_ in ("pass", "fail", "unverified"):
            out[f"reads_{s_}"][i] = by(sl, st == s_)
        out["ret_reads"][i] = by(sl, loaded & ret)
        out["ret_reads_fail"][i] = by(sl, (st == "fail") & ret)
        out["ret_reads_unverified"][i] = by(sl, (st == "unverified") & ret)
    return out


def write_per_session(path, P, eng, views, meta):
    """npz（壓縮）：每個陣列一個 key；views 的 label 與 meta（JSON 字串）一起存。"""
    import json
    arr = per_session(P, eng, views)
    sk = P.get("sess_key")
    if sk is not None:
        arr["sess_key"] = np.asarray(sk, dtype=str)
    arr["view_labels"] = np.asarray([v.label for v in views], dtype=str)
    arr["meta_json"] = np.asarray(json.dumps(meta, ensure_ascii=False, default=str))
    np.savez_compressed(path, **arr)
    return arr
