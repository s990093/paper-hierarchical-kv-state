#!/usr/bin/env python3
"""Milestone 4 的 go/no-go 判定材料。**全部從 results/ 的 CSV 讀出，不手打數字。**

## 為什麼是一支程式而不是一份手寫摘要

2026-08-31 我曾把數字手打進互動儀表板，打錯了兩個
（`full_gpu` 寫成 178,161，實測是 237,356）。那違反 EXPERIMENT_PLAN §0 禁令 1。
判定材料是整個專案最重要的輸出，更不能手打。

這支程式只做三件事：讀檔、依 §0 的門檻分類、印出來。
任何數字若在 `results/` 裡找不到，它就印 `NOT_MEASURED`，不會猜。

## 判準（EXPERIMENT_PLAN §0 禁令 4）

    headroom > 15%   GO
    5% – 15%         MARGINAL：停下來問人
    < 5%             NO_GO：停止

用法：python code/m4_verdict.py
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
M2 = REPO / "results/m2_harness"
M4 = REPO / "results/m4_oracle"
NM = "NOT_MEASURED"


def verdict(h: float) -> str:
    return "GO" if h > 15 else "MARGINAL" if h >= 5 else "NO_GO"


# 🔴 過期資料偵測。
#    2026-08-31 修正了 Mooncake 的 block 粒度（hash_id 是 512 token 不是 16），
#    工作集因此變成 32 倍。修正前產生的 CSV 看起來完全正常，
#    只是每個數字都錯——把它們混進判定材料，就是拿錯的證據做決定。
#    每個 trace 的正確不重複 block 數是可以現算的，拿來當版本戳記。
_EXPECTED_UNIQ: dict[str, int] = {}


def expected_uniq(tname: str) -> int | None:
    """現算這條 trace 的不重複 block 數，當作資料版本的指紋。"""
    if tname not in _EXPECTED_UNIQ:
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from m4_oracle import mooncake_trace
            tr = mooncake_trace(tname)
            _EXPECTED_UNIQ[tname] = len({b for r in tr for b in r})
        except Exception:  # noqa: BLE001
            _EXPECTED_UNIQ[tname] = None
    return _EXPECTED_UNIQ[tname]


def rows(p: Path, check_stale: bool = True) -> list[dict]:
    """讀 CSV，並剔除粒度修正前產生的過期列。"""
    if not p.exists():
        return []
    out = list(csv.DictReader(p.open()))
    if not check_stale or not out or "unique_blocks" not in out[0]:
        return out
    fresh, stale, neg, oldsim = [], 0, 0, 0
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from m4_oracle import SIM_VERSION as _cur
    except Exception:  # noqa: BLE001
        _cur = None
    for r in out:
        # 🔴 第二層防線：可證明錯誤的列一律剔除。
        #    負的 headroom 代表 Oracle 輸給 baseline，在定義上不可能。
        #    2026-08-31 的成本感知規則 bug 就會產生這種列，
        #    而它的 unique_blocks 是對的，所以粒度檢查抓不到。
        h = r.get("oracle_headroom_pct")
        if h not in (None, "") and float(h) < 0:
            neg += 1
            continue
        # 🔴 第三層：模擬器版本戳記。任何對 m4_oracle.py 的改動都會改變它。
        sv = r.get("sim_version")
        if _cur and sv and sv != _cur:
            oldsim += 1
            continue
        t = r.get("trace") or ""
        exp = expected_uniq(t) if t else None
        got = r.get("unique_blocks")
        if exp is None or not got:
            fresh.append(r)
        elif int(got) == exp:
            fresh.append(r)
        else:
            stale += 1
    msgs = []
    if stale:
        msgs.append(f"{stale} 列 trace 解碼過期"
                    f"（unique_blocks 應為 "
                    f"{expected_uniq(out[0].get('trace', '')):,}）")
    if neg:
        msgs.append(f"{neg} 列 headroom 為負（Oracle 輸給 baseline，不可能）")
    if oldsim:
        msgs.append(f"{oldsim} 列由舊版模擬器產生（sim_version != {_cur}）")
    if msgs:
        print(f"   ⚠️ {p.name}：剔除 " + "、".join(msgs))
    return fresh


def band(vals: list[float]) -> str:
    if not vals:
        return NM
    return f"{min(vals):.2f}% – {max(vals):.2f}%"


def main() -> int:
    print("=" * 78)
    print(" Milestone 4 — go/no-go 判定材料")
    print(" 全部數字自 results/ 讀出。找不到的印 NOT_MEASURED。")
    print("=" * 78)

    cm = M4 / "cost_model.json"
    if cm.exists():
        d = json.loads(cm.read_text())
        m, dv = d["measured"], d["derived_ms_per_block"]
        print(f"\n## 成本模型（來源：{Path(m['retrieval_csv']).name}）")
        print(f"   GPU {dv['gpu']:.3f}、CPU {dv['cpu']:.3f}、SSD {dv['ssd']:.3f} ms/block")
        print(f"   重算 {dv['recompute_base']:.3f} + "
              f"{dv['recompute_slope_per_token']:.6f} × 位置 ms/block")
        x = (dv["ssd"] - dv["recompute_base"]) / dv["recompute_slope_per_token"]
        print(f"   SSD 與重算的交叉點：位置 {x:,.0f} token")
        fit = m.get("position_fit_max_tokens", d.get("position_fit_max_tokens"))
        print(f"   位置擬合上限：{fit if fit else NM} token")
    else:
        print(f"\n## 成本模型：{NM}")

    all_h: list[float] = []

    print("\n## A. SSD 容量掃描（主結果）")
    r = [x for x in rows(M4 / "ssd_sweep.csv") if x["policy"] == "oracle"]
    if not r:
        print(f"   {NM}")
    else:
        print(f"   {'trace':13s}{'SSD':>10s}{'best baseline':>15s}"
              f"{'headroom':>10s}{'判定':>10s}{'baseline 寫入':>14s}{'可行?':>7s}")
        dev = float(r[0].get("device_write_mibps_sustained") or 0) or None
        for x in r:
            h = float(x["oracle_headroom_pct"])
            all_h.append(h)
            b = [y for y in rows(M4 / "ssd_sweep.csv")
                 if y["trace"] == x["trace"] and y["ssd_gib"] == x["ssd_gib"]
                 and y["policy"] == x["best_baseline"]]
            w = float(b[0]["ssd_write_mibps"]) if b and b[0].get("ssd_write_mibps") else 0.0
            feas = "✅" if (dev is None or w <= dev) else "🔴"
            print(f"   {x['trace']:13s}{x['ssd_gib']:>10s}{x['best_baseline']:>15s}"
                  f"{h:>9.2f}%{verdict(h):>10s}{w:>11,.0f}MiB/s{feas:>6s}")
        if dev:
            print(f"   裝置持續寫入能力（實測）：{dev:,.0f} MiB/s")

    print("\n## B. GPU 預算掃描")
    r = [x for x in rows(M4 / "budget_sweep.csv") if x["policy"] == "oracle"]
    if not r:
        print(f"   {NM}")
    else:
        for x in r:
            h = float(x["oracle_headroom_pct"])
            all_h.append(h)
            print(f"   {x['trace']:13s}預算 {int(x['gpu_budget_tokens']):>7,} token"
                  f"　壓力 {float(x['pressure_x']):>8,.0f}×"
                  f"　免費逐出 {float(x['evict_free_pct']):>5.1f}%"
                  f"　headroom {h:>6.2f}% {verdict(h)}")

    print("\n## C. 節省按請求長度分箱")
    r = rows(M4 / "by_length.csv")
    if not r:
        print(f"   {NM}")
    else:
        for t in sorted({x["trace"] for x in r}):
            rr = [x for x in r if x["trace"] == t]
            print(f"   {t}（最佳 baseline = {rr[0]['best_baseline']}）")
            for x in rr:
                print(f"      {x['bin']:>10s}  {int(x['requests']):>6,} 筆"
                      f"　佔時間 {float(x['share_of_total_time_pct']):>5.1f}%"
                      f"　節省 {float(x['saving_pct_within_bin']):>6.2f}%"
                      f"　佔總節省 {float(x['share_of_total_saving_pct']):>5.1f}%")

    print("\n## D. 模擬器的系統語意假設各自的影響")
    r = [x for x in rows(M4 / "semantics_ablation.csv") if x["policy"] == "oracle"]
    if not r:
        print(f"   {NM}")
    else:
        for w in sorted({x["workload"] for x in r}):
            rr = {x["semantics"]: float(x["oracle_headroom_pct"])
                  for x in r if x["workload"] == w}
            base = rr.get("per-block/no-prefetch")
            out = "　".join(f"{k}={v:.2f}%" for k, v in rr.items())
            print(f"   {w:32s}{out}")
            if base is not None and "prefix/prefetch" in rr:
                print(f"   {'':32s}→ 兩項修正合計 "
                      f"{rr['prefix/prefetch'] - base:+.2f} 個百分點")

    print("\n## E. 前綴語意為什麼幾乎沒有影響")
    r = rows(M4 / "prefix_gap_probe.csv")
    if not r:
        print(f"   {NM}")
    else:
        v = [float(x["post_gap_resident_pct"]) for x in r]
        f = [float(x["post_gap_first_ever_pct"]) for x in r]
        print(f"   缺口之後仍留在任一階的 block：{min(v):.3f}% – {max(v):.3f}%"
              f"（{len(r)} 組 trace × 策略）")
        print(f"   缺口之後是首次出現的 block：{min(f):.1f}% – {max(f):.1f}%")

    print("\n## F. 磁碟頻寬（可行性判定的依據）")
    r = []
    for n, tag in (("disk_bw.csv", "burst 1 GiB"),
                   ("disk_bw_sustained.csv", "sustained 16 GiB")):
        for x in rows(M2 / n):
            if "write_mibps" in x and x["write_mibps"]:
                r.append((tag, x["path"], float(x["write_mibps"]),
                          float(x["read_mibps"])))
    if not r:
        print(f"   {NM}")
    else:
        from statistics import median
        for tag in dict.fromkeys(t for t, *_ in r):
            for path in dict.fromkeys(p for t, p, *_ in r if t == tag):
                w = [x[2] for x in r if x[0] == tag and x[1] == path]
                rd = [x[3] for x in r if x[0] == tag and x[1] == path]
                print(f"   {tag:18s}{path:18s}寫 {median(w):>8,.0f}"
                      f"　讀 {median(rd):>8,.0f} MiB/s")

    print("\n## G. 只比「在真機上跑得起來」的策略")
    print("   模擬的成本模型只向讀取收費，寫入免費。加上實測的持續寫入頻寬之後，")
    print("   有些策略根本寫不下去。此節把不可行的策略排除後重新比較。")
    r = rows(M4 / "ssd_sweep.csv")
    if not r:
        print(f"   {NM}")
    else:
        # 🔴 可行性判定用的裝置，必須與成本常數的來源裝置一致。
        #    第一版這裡同時列了兩顆碟，但成本常數只有一份（SATA），
        #    等於「SATA 的成本 × NVMe 的頻寬」——裝置混用，結論不成立。
        #    現在只取 CSV 裡記載的那一顆。
        dev_used = sorted({x.get("device", "") for x in r}) or [""]
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from m4_oracle import DEVICE_WRITE_MIBPS
        except Exception:  # noqa: BLE001
            DEVICE_WRITE_MIBPS = {}
        DEV = {d: DEVICE_WRITE_MIBPS[d] for d in dev_used
               if d in DEVICE_WRITE_MIBPS}
        if not DEV:
            print(f"   {NM}（CSV 沒有記 device 欄，或該裝置沒有實測寫入頻寬）")
            DEV = {}
        else:
            print(f"   成本常數與寫入頻寬上限同時來自："
                  + "、".join(f"{d} ({v:,.0f} MiB/s)" for d, v in DEV.items()))
        by = {}
        for x in r:
            by.setdefault((x["trace"], x["ssd_gib"]), {})[x["policy"]] = x
        print(f"   {'trace':13s}{'SSD':>10s}{'裝置':>16s}"
              f"{'可行的 best':>13s}{'oracle 可行?':>13s}{'headroom':>10s}{'判定':>10s}")
        for (t, g), pol in sorted(by.items(),
                                  key=lambda kv: (kv[0][0],
                                                  float("inf") if kv[0][1] == "unlimited"
                                                  else float(kv[0][1]))):
            for dname, cap in DEV.items():
                def w(p_):
                    v = pol.get(p_, {}).get("ssd_write_mibps")
                    return float(v) if v not in (None, "") else 0.0
                feas_base = [k for k in pol
                             if k != "oracle" and w(k) <= cap]
                if not feas_base:
                    print(f"   {t:13s}{g:>10s}{dname:>16s}"
                          f"{'（無可行 baseline）':>15s}")
                    continue
                b = min(feas_base, key=lambda k: float(pol[k]["total_ms"]))
                o_ok = w("oracle") <= cap
                h = 100 * (float(pol[b]["total_ms"])
                           - float(pol["oracle"]["total_ms"])) \
                    / float(pol[b]["total_ms"])
                print(f"   {t:13s}{g:>10s}{dname:>16s}{b:>13s}"
                      f"{'✅' if o_ok else '🔴':>10s}"
                      f"{h:>9.2f}%{verdict(h) if o_ok else '（不可行）':>12s}")

    print("\n" + "=" * 78)
    if all_h:
        vs = {verdict(h) for h in all_h}
        print(f" headroom 全距：{band(all_h)}（{len(all_h)} 個設定）")
        print(f" 判定分佈：" + "、".join(
            f"{v} {sum(1 for h in all_h if verdict(h) == v)} 個"
            for v in ("GO", "MARGINAL", "NO_GO") if v in vs))
        if vs == {"MARGINAL"}:
            print("\n 🟡 全部落在 MARGINAL。EXPERIMENT_PLAN §0 禁令 4：")
            print("    「5–15% 停下來問人」。**不得自行決定繼續或停止。**")
    else:
        print(f" headroom：{NM}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
