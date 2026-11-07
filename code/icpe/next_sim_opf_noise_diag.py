"""icpe-sim 11/7（icpe-main 轉達使用者的請求）：P4（W4 雜訊）裡 O-PF 在模型 P 的斷崖——診斷用，不改引擎。

做法
- 把 x1_sim.Engine 換成子類別 DiagEngine：只在旁邊讀狀態、記數，**不改任何狀態、不多抽亂數**（不呼叫 _nz、next_req）。
- 照原本的入口跑：`python -I next_sim_opf_noise_diag.py --variant V1 --calib ... --out-root ... x1 sweep ... --engine py`
  （參數＝p3_fvar.py 的參數）。輸出的 x1_sweep.csv 必須和正式 run 的同一格逐值相同 → 證明記錄沒有改到計算。
- 每個 job 跑完，把記錄寫成一行 JSON 到 $DIAG_OUT（預設 ./opf_diag.jsonl）。只有彙總的計數，沒有逐對話的數字。

記錄什麼（只在 CPU 規則是 O-PF 時；pf_target＝cpu）
1. push：session 降級到慢層時（demote，含 make_room 裡的降級）變成預取候選，讀它的 sT（加雜訊後的「下次被讀」，請求 index）
   和真值 T＝next_access(最後一塊, cur)：
   exact（相同）、dead_to_next／dead_to_later（真值＝不再被讀，雜訊說下一個請求／之後某個請求會讀）、
   alive_to_dead、alive_to_next（真值有限、雜訊說就是下一個請求）、alive_other。
2. make_room（為堆頂的候選騰空間）：呼叫次數、失敗次數（失敗 → advance_prefetch 直接 return，這一輪不再預取）、
   失敗時候選者的類別、有失敗的請求數。
3. 快照（每 snap 個請求，在 step(r) 之前；cur＝r−1）：CPU 已用量拆成「預取進來的 INT4」與其他；預取的再依持有者分：
   dead（真值：之後不會再被讀）、overdue（真值有限、但 sT ≤ cur：預測的時間已過、還沒被讀）、pending（sT > cur）。
4. 結束時：每個時間設定的 pf_bytes、n_unpf、n_demote。
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CODE, "p3x"))
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)

import x1_sim as XS  # noqa: E402

GiB = float(1 << 30)
SNAP = int(os.environ.get("DIAG_SNAP", "2000"))
CF = os.environ.get("DIAG_CF", "")      # ""＝只記錄（計算和正式引擎相同）；"expire"／"miss"＝反事實，見 _cf_expire
OUT = os.environ.get("DIAG_OUT", "opf_diag.jsonl")
Base = XS.Engine


class DiagEngine(Base):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._dg = self.pf == "opf" and getattr(self, "gpf", None) is None
        self.dg_push = {}
        self.dg_mr_calls = self.dg_mr_fail = self.dg_mr_fail_steps = 0
        self.dg_mr_fail_cls = {}
        self.dg_first_fail = -1
        self._dg_last_fail_r = -2
        self.dg_snaps = []

    # ---- 只讀的輔助
    def _true_T(self, s):
        bl = self.sb[s]
        return XS.XT.next_access(self.acc, bl[-1], self.cur) if bl else XS.INF

    def _cls_pf(self, s):
        if self._true_T(s) >= XS.INF:
            return "dead"
        return "overdue" if self.sT[s] <= self.cur else "pending"

    # ---- 包裝（先呼叫原本的，再讀）
    def demote(self, s, now_by_view=None):
        w = Base.demote(self, s, now_by_view)
        if self._dg:
            T, Tn, nx = self._true_T(s), self.sT[s], self.cur + 1
            if Tn == T:
                k = "exact"
            elif T >= XS.INF:
                k = "dead_to_next" if Tn == nx else "dead_to_later"
            elif Tn >= XS.INF:
                k = "alive_to_dead"
            elif Tn == nx:
                k = "alive_to_next"
            else:
                k = "alive_other"
            self.dg_push[k] = self.dg_push.get(k, 0) + 1
        return w

    def make_room(self, cand, need, now, v):
        if not self._dg:
            return Base.make_room(self, cand, need, now, v)
        c = self._cls_pf(cand)
        ok = Base.make_room(self, cand, need, now, v)
        self.dg_mr_calls += 1
        if not ok:
            self.dg_mr_fail += 1
            self.dg_mr_fail_cls[c] = self.dg_mr_fail_cls.get(c, 0) + 1
            r = self.cur + 1      # advance_prefetch 在 step(r) 開頭、cur 更新之前
            if r != self._dg_last_fail_r:
                self.dg_mr_fail_steps += 1
                self._dg_last_fail_r = r
            if self.dg_first_fail < 0:
                self.dg_first_fail = r
        return ok

    def step(self, r):
        if self._dg and r % SNAP == 0:
            self._snap(r)
        if self._gdg and r % SNAP == 0:
            self._gsnap(r)
        if self._dg and CF in ("expire", "miss"):
            self._cf_expire()
        return Base.step(self, r)

    # ---- 反事實（只給診斷；DIAG_CF＝expire／miss 才開；正式引擎沒有這條規則）
    def _cf_expire(self):
        """預取進 CPU 的 session，預測被讀的請求（sT）已經過了（sT ≤ cur）卻還留著預取的塊 → 丟掉那份 INT4 副本（資料還在慢層），
        不再放回預取佇列（每個預測只預取一次）。在 step(r) 開頭、advance_prefetch 之前做。
        expire：只要 sT 過了就丟。miss：只在「預測的那個請求（index sT）沒有讀這個 session 的最後一塊」時才丟
        （完美預知時 sT＝最後一塊真的下次被讀的請求，所以永遠不會丟 → λ＝0 應該和正式引擎逐值相同）。"""
        cur = self.cur
        if not hasattr(self, "_cf_read"):
            self._cf_read = {}
        exp = []
        for s in self.pf_sessions:
            T = self.sT[s]
            if self.spf[s] <= 0 or T > cur:
                continue
            if CF == "miss":
                if self._cf_read.get(s) == T:
                    continue
                bl = self.sb[s]
                if bl and bl[-1] in self.full[T]:
                    self._cf_read[s] = T          # 預測的讀取真的發生了（被別的 session 讀走也算）→ 留著，同正式引擎
                    continue
            exp.append(s)
        for s in exp:
            for b in self.sb[s]:
                if self.pf_owner[b] == s:
                    self.cpu_used -= self.i4
                    self.fmt[b] = 0
                    self.pf_owner[b] = -1
            self.spf[s] = 0
            self.pf_sessions.discard(s)
            self.pfdone[s] = True
            self.dg_cf_expired = getattr(self, "dg_cf_expired", 0) + 1

    # ---- GPU 層的 BEL（O-GPU；icpe-main 的第 3 點）：逐出的選擇和 GPU 上「不會再回來」的 session
    @property
    def _gdg(self):
        return self.G > 0 and getattr(self, "gorder", None) == "bel"

    def _g_true(self, x):
        bl = self._gbl(x)
        return self._next_req_bl(bl, self.ref_gpu) if bl else XS.INF

    def _b1_overdue(self, x):
        """B1、模型 P：x 存著的抽樣（不抽新的）是否已經換成「上次回合結束＋間隔」且那個時間已過（→ _nz 夾到下一個請求）。
        None＝還沒抽過（不知道）。不呼叫 _nz、不動亂數。"""
        if self.noise is None or not self.nz_b1 or self.nzm != "P" or self.nzl == 0.0:
            return False
        d = self.nzd[x]
        if d is None:
            return None
        u, g, _ = d
        if u >= self.nzl or g >= XS.INF:
            return False
        return self.t[self.last[x]] + g <= self.t[self.cur]

    def _g_select(self, s_new, now):
        x = Base._g_select(self, s_new, now)
        if self._gdg:
            if not hasattr(self, "dg_gev"):
                self.dg_gev = dict(n=0, victim_dead=0, dead_existed=0, miss=0, victim_alive_ok=0)
            e = self.dg_gev
            e["n"] += 1
            tv = self._g_true(x)
            anydead = False
            for y in self.glru:
                if self._g_true(y) >= XS.INF:
                    anydead = True
                    break
            if tv >= XS.INF:
                e["victim_dead"] += 1
            elif anydead:
                e["miss"] += 1          # 有不會再回來的可以逐出，卻逐出了會回來的
            else:
                e["victim_alive_ok"] += 1
            if anydead:
                e["dead_existed"] += 1
        return x

    def _gsnap(self, r):
        n = nd = ndo = 0
        for x in self.glru:
            n += 1
            if self._g_true(x) >= XS.INF:
                nd += 1
                if self._b1_overdue(x):
                    ndo += 1
        if not hasattr(self, "dg_gsnaps"):
            self.dg_gsnaps = []
        self.dg_gsnaps.append(dict(r=r, gpu_used_GiB=self.gpu_used / GiB, G_GiB=self.G / GiB, n_gpu=n, n_gpu_dead=nd,
                                   n_gpu_dead_b1_overdue=ndo))

    def _snap(self, r):
        i4 = self.i4
        pf = {"dead": 0, "overdue": 0, "pending": 0}
        npf = {"dead": 0, "overdue": 0, "pending": 0}
        for s in self.pf_sessions:
            n = self.spf[s]
            if n <= 0:
                continue
            c = self._cls_pf(s)
            pf[c] += n * i4
            npf[c] += 1
        tot = sum(pf.values())
        self.dg_snaps.append(dict(r=r, cpu_used_GiB=self.cpu_used / GiB, C_GiB=self.C / GiB,
                                  pf_GiB=tot / GiB, pf_dead_GiB=pf["dead"] / GiB, pf_overdue_GiB=pf["overdue"] / GiB,
                                  pf_pending_GiB=pf["pending"] / GiB, n_pf_dead=npf["dead"], n_pf_overdue=npf["overdue"],
                                  n_pf_pending=npf["pending"], n_cpu_set=len(self.cpu_set), heap=len(self.pfheap)))

    def run(self):
        Base.run(self)
        if self._gdg:
            rec = dict(kind="gpu_bel", policy=self.policy, C_GiB=self.C / GiB, nreq=len(self.full), noise=self.noise,
                       G_GiB=self.G / GiB, views=[dict(pf_GiB=v.pf_bytes / GiB) for v in self.views],
                       gev=getattr(self, "dg_gev", {}), gsnaps=getattr(self, "dg_gsnaps", []), snap_every=SNAP)
            with open(OUT.replace(".jsonl", "_gpu.jsonl"), "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        if self._dg:
            rec = dict(policy=self.policy if hasattr(self, "policy") else None, C_GiB=self.C / GiB, nreq=len(self.full),
                       noise=self.noise, gorder=getattr(self, "gorder", None), gttl=getattr(self, "gttl", None),
                       G_GiB=self.G / GiB,
                       views=[dict(slow=v.slow_name if hasattr(v, "slow_name") else None, pf_GiB=v.pf_bytes / GiB,
                                   mean_ret=None) for v in self.views],
                       push=self.dg_push, mr_calls=self.dg_mr_calls, mr_fail=self.dg_mr_fail,
                       mr_fail_steps=self.dg_mr_fail_steps, mr_fail_cls=self.dg_mr_fail_cls, first_fail_r=self.dg_first_fail,
                       n_unpf=self.n_unpf, n_demote=self.n_demote, snaps=self.dg_snaps, snap_every=SNAP,
                       cf=CF, cf_expired=getattr(self, "dg_cf_expired", 0))
            with open(OUT, "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        return self


XS.Engine = DiagEngine


def summary(run_dir, official, out_csv, host="acvlab"):
    """run_dir 的 opf_diag.jsonl＋out/x1/x1_sweep.csv → 每個 job 一列（只有彙總）。official＝icpe-main 的 *_noise_headroom.csv
    （拿 T_oracle 對照：同一格、同一個雜訊設定的 mean_ret 必須相同）。"""
    import datetime
    import pandas as pd
    rid = os.path.basename(os.path.normpath(run_dir))
    sw = pd.read_csv(os.path.join(run_dir, "out", "x1", "x1_sweep.csv"))
    off = pd.read_csv(official) if official else None
    recs = [json.loads(x) for x in open(os.path.join(run_dir, "opf_diag.jsonl"))]
    recs = [d for d in recs if d.get("kind") != "gpu_bel"]
    rows = []
    for d in recs:
        nz = d["noise"]
        q = sw[(sw.cpu_GiB == d["C_GiB"]) & (sw.noise_model == nz["model"]) & (sw.noise_lambda == nz["lam"]) &
               (sw.noise_draw == nz["draw"]) & (sw.noise_seed == nz["seed"])]
        # 一個 job 一個時間設定（O-PF＋TTL）：用 prefetch_GiB 認出是哪一列（同一個雜訊設定有 idle、busy 兩個 job）
        pf = d["views"][0]["pf_GiB"]
        q = q[(q.prefetch_GiB - pf).abs() < 1e-5]
        assert len(q) == 1, (nz, pf, len(q))
        q = q.iloc[0]
        T_off = float("nan")
        if off is not None:
            o = off[(off.oracle_policy == q.policy) & (off.trace == q.trace) & (off.cpu_GiB == q.cpu_GiB) & (off.slow == q.slow) &
                    (off.gpu == q.gpu) & (off.oracle_gpu_policy == q.gpu_policy) & (off.noise_model == nz["model"]) &
                    (off.noise_lambda == nz["lam"]) & (off.noise_draw == nz["draw"]) & (off.noise_seed == nz["seed"])]
            if len(o):
                T_off = float(o.T_oracle.iloc[0])
        sn = pd.DataFrame(d["snaps"])
        late = sn[sn.r >= d["nreq"] // 2]       # 後半段的快照
        p = d["push"]
        npush = sum(p.values())
        rows.append(dict(
            run_id=rid, ts=datetime.datetime.now().astimezone().isoformat(timespec="seconds"), host=host,
            trace=q.trace, cpu_GiB=q.cpu_GiB, slow=q.slow, gpu=q.gpu, policy=q.policy, gpu_policy=q.gpu_policy,
            noise_model=nz["model"], noise_lambda=nz["lam"], noise_draw=nz["draw"], noise_seed=nz["seed"],
            mean_ret=q.mean_ret, T_oracle_p4=T_off,
            same_as_p4=(abs(q.mean_ret - T_off) <= 5e-7) if T_off == T_off else "NOT_IN_P4",
            prefetch_GiB=q.prefetch_GiB, ret_cpu_loads=q.ret_cpu_loads, ret_slow_loads=q.ret_slow_loads,
            n_push=npush, push_exact=p.get("exact", 0), push_dead_to_next=p.get("dead_to_next", 0),
            push_dead_to_later=p.get("dead_to_later", 0), push_alive_to_dead=p.get("alive_to_dead", 0),
            push_alive_to_next=p.get("alive_to_next", 0), push_alive_other=p.get("alive_other", 0),
            mr_calls=d["mr_calls"], mr_fail=d["mr_fail"], mr_fail_steps=d["mr_fail_steps"],
            mr_fail_cand_dead=d["mr_fail_cls"].get("dead", 0), mr_fail_cand_overdue=d["mr_fail_cls"].get("overdue", 0),
            mr_fail_cand_pending=d["mr_fail_cls"].get("pending", 0),
            nreq=d["nreq"], first_fail_r=d["first_fail_r"], n_unpf=d["n_unpf"], n_demote=d["n_demote"],
            cf=d.get("cf", "") or "none", cf_expired=d.get("cf_expired", 0),
            snap_every=d["snap_every"], n_snaps_2nd_half=len(late),
            C_GiB=d["C_GiB"],
            pf_GiB_2nd_half_median=float(late.pf_GiB.median()),
            pf_dead_GiB_2nd_half_median=float(late.pf_dead_GiB.median()),
            pf_overdue_GiB_2nd_half_median=float(late.pf_overdue_GiB.median()),
            pf_pending_GiB_2nd_half_median=float(late.pf_pending_GiB.median()),
            n_cpu_set_2nd_half_median=float(late.n_cpu_set.median()),
            pf_dead_GiB_end=float(sn.pf_dead_GiB.iloc[-1]), pf_GiB_end=float(sn.pf_GiB.iloc[-1]),
        ))
    out = pd.DataFrame(rows).sort_values(["gpu", "noise_model", "noise_lambda", "noise_draw", "noise_seed"])
    out.to_csv(out_csv, index=False)
    return out


def gsummary(run_dir, official, out_csv, host="acvlab"):
    """GPU 層 BEL（O-GPU）的記錄：opf_diag_gpu.jsonl＋x1_sweep.csv → 每個 job 一列（只有彙總）。"""
    import datetime
    import pandas as pd
    rid = os.path.basename(os.path.normpath(run_dir))
    sw = pd.read_csv(os.path.join(run_dir, "out", "x1", "x1_sweep.csv"))
    off = pd.read_csv(official) if official else None
    recs = [json.loads(x) for x in open(os.path.join(run_dir, "opf_diag_gpu.jsonl"))]
    rows = []
    for d in recs:
        nz = d["noise"]
        q = sw[(sw.cpu_GiB == d["C_GiB"]) & (sw.noise_model == nz["model"]) & (sw.noise_lambda == nz["lam"]) &
               (sw.noise_draw == nz["draw"]) & (sw.noise_seed == nz["seed"])]
        pf = d["views"][0]["pf_GiB"]
        q = q[(q.prefetch_GiB - pf).abs() < 1e-5]
        assert len(q) == 1, (nz, pf, len(q))
        q = q.iloc[0]
        T_off = float("nan")
        if off is not None:
            o = off[(off.oracle_policy == q.policy) & (off.trace == q.trace) & (off.rule == q.rule) & (off.cpu_GiB == q.cpu_GiB) &
                    (off.slow == q.slow) & (off.gpu == q.gpu) & (off.oracle_gpu_policy == q.gpu_policy) &
                    (off.noise_model == nz["model"]) & (off.noise_lambda == nz["lam"]) & (off.noise_draw == nz["draw"]) &
                    (off.noise_seed == nz["seed"])]
            if len(o):
                T_off = float(o.T_oracle.iloc[0])
        sn = pd.DataFrame(d["gsnaps"])
        late = sn[sn.r >= d["nreq"] // 2]
        e = d["gev"]
        rows.append(dict(
            run_id=rid, ts=datetime.datetime.now().astimezone().isoformat(timespec="seconds"), host=host,
            trace=q.trace, rule=q.rule, cpu_GiB=q.cpu_GiB, slow=q.slow, gpu=q.gpu, policy=q.policy, gpu_policy=q.gpu_policy,
            noise_model=nz["model"], noise_lambda=nz["lam"], noise_draw=nz["draw"], noise_seed=nz["seed"],
            mean_ret=q.mean_ret, T_oracle_p4=T_off,
            same_as_p4=(abs(q.mean_ret - T_off) <= 5e-7) if T_off == T_off else "NOT_IN_P4",
            g_evict=e.get("n", 0), g_victim_dead=e.get("victim_dead", 0), g_miss_alive_while_dead_existed=e.get("miss", 0),
            g_victim_alive_no_dead=e.get("victim_alive_ok", 0), g_dead_existed=e.get("dead_existed", 0),
            nreq=d["nreq"], snap_every=d["snap_every"], n_snaps_2nd_half=len(late), G_GiB=d["G_GiB"],
            n_gpu_2nd_half_median=float(late.n_gpu.median()), n_gpu_dead_2nd_half_median=float(late.n_gpu_dead.median()),
            n_gpu_dead_b1_overdue_2nd_half_median=float(late.n_gpu_dead_b1_overdue.median()),
            gpu_used_GiB_2nd_half_median=float(late.gpu_used_GiB.median()),
        ))
    out = pd.DataFrame(rows).sort_values(["trace", "gpu", "noise_model", "noise_lambda", "noise_draw", "noise_seed"])
    out.to_csv(out_csv, index=False)
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "gsummary":    # gsummary <run_dir> <official headroom csv|""> <out csv>
        o = gsummary(sys.argv[2], sys.argv[3] or None, sys.argv[4])
        print(o.to_string())
    elif len(sys.argv) > 1 and sys.argv[1] == "summary":     # summary <run_dir> <official headroom csv|""> <out csv>
        o = summary(sys.argv[2], sys.argv[3] or None, sys.argv[4])
        print(o.to_string())
    else:
        import p3_fvar
        sys.argv = ["p3_fvar.py"] + sys.argv[1:]
        p3_fvar.main()
