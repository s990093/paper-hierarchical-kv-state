"""把 v7（code/icpe/rux_x1fast/v7_fast/src/main.rs）改成 v8：加第 2–5 項的新功能（只有 Rust，語意見
docs/icpe_brief_20261030/next/rux/SEMANTICS_2to5.md）。只做精確的字串取代，每處都要剛好出現一次。
用法：python -I patch_v8.py <v7 main.rs> <輸出 main.rs>
  A. 排隊（項目 3）：jobs.txt 多一欄 `q=1` → 單一 GPU、FCFS；開始＝max(到達, 前一個完成)；回報的 TTFT＝等待＋服務；一個 job 1 個時間設定。
  B. W3 線索（項目 4）：GPU 規則 `cuep`（有工具呼叫就釘到下一個請求）、`cueo`（有工具呼叫就回合結束立刻逐出）；`cue=1` → 讀 <job 目錄>/cue.u8。
  C. W5（項目 5）：`w5=<CPU 每塊位元組>` → CPU 層每塊占這麼多位元組；讀 CPU 的塊另加每塊殘差（<job 目錄>/w5.f64：每個時間設定 [dc1, dc2]）。
  D. APC（項目 2）：GPU 規則 `apc`（session LRU、尾端一塊一塊逐出）、`belp`（同樣逐塊，選 session 照 Belady）。
沒有這些欄位時每條路徑和 v7 相同。"""
import hashlib, sys

src, dst = sys.argv[1:3]
s = open(src, encoding="utf-8").read()
V7 = hashlib.sha256(s.encode()).hexdigest()


def rep(old, new, n=1):
    global s
    c = s.count(old)
    assert c == n, (c, old[:200])
    s = s.replace(old, new)


# ================================================================== 檔頭
rep("// icpe-rux v7（2026-11-07，code/icpe/rux_x1fast/v7_fast）", """// icpe-rux v8（2026-11-07，code/icpe/rux_x1fast/v8_features）：v7＋第 2–5 項的新功能（只有 Rust；語意見
//   docs/icpe_brief_20261030/next/rux/SEMANTICS_2to5.md；patch_v8.py）：排隊 q=1、W3 線索 GPU 規則 cuep／cueo、W5 w5=、APC apc／belp。
// icpe-rux v7（2026-11-07，code/icpe/rux_x1fast/v7_fast）""")

# ================================================================== Spec：新欄位
rep("""                  gwb: bool, pf_gpu: bool, vids: Vec<usize>, nz: Option<(bool, f64, bool, i64)>, sc: Option<(SRule, i64)> }""",
    """                  gwb: bool, pf_gpu: bool, vids: Vec<usize>, nz: Option<(bool, f64, bool, i64)>, sc: Option<(SRule, i64)>,
                  q: bool, qc: bool, cue: bool, w5: Option<i64>, qpost: bool }""")
rep("""    let gorder: u8 = match x[8] { "-" => 0, "lru" => 0, "bel" => 1, "ttla" => 2, "ttlb" => 3, o => panic!("gorder {}", o) };""",
    """    let gorder: u8 = match x[8] { "-" => 0, "lru" => 0, "bel" => 1, "ttla" => 2, "ttlb" => 3,
                                  "cuep" => 4, "cueo" => 5, "apc" => 6, "belp" => 7, o => panic!("gorder {}", o) };   // v8：4–7""")
rep("""    let mut sc: Option<(SRule, i64)> = None;
    for extra in &x[13..] {""", """    let mut sc: Option<(SRule, i64)> = None;
    let (mut q, mut qc, mut cue, mut w5, mut qpost) = (false, false, false, None, false);
    for extra in &x[13..] {
        if *extra == "q=1" || *extra == "q=2" { assert!(!q); q = true; qc = *extra == "q=2"; continue; }   // v8 排隊：1＝open-loop、2＝closed-loop
        if *extra == "cue=1" { assert!(!cue); cue = true; continue; }                 // v8 W3 線索：<job 目錄>/cue.u8
        if *extra == "qpost=1" { assert!(!qpost); qpost = true; continue; }           // v8 排隊（一階近似，事後算；SEMANTICS §2b）
        if let Some(v) = extra.strip_prefix("w5=") {                                  // v8 W5：CPU 每塊位元組
            assert!(w5.is_none()); let b: i64 = v.parse().unwrap(); assert!(b > 0); w5 = Some(b); continue;
        }""")
rep("""                    vids: x[12].split(',').map(|v| v.parse().unwrap()).collect(), nz, sc };
    if sp.pf != Pf::No { assert!(sp.vids.len() == 1); }
    if sp.g > 0 && sp.gorder >= 2 { assert!(sp.vids.len() == 1); }""",
    """                    vids: x[12].split(',').map(|v| v.parse().unwrap()).collect(), nz, sc, q, qc, cue, w5, qpost };
    if sp.qpost { assert!(!sp.q, "qpost 和 q 不能同時開"); }
    if sp.pf != Pf::No { assert!(sp.vids.len() == 1); }
    if sp.g > 0 && (sp.gorder == 2 || sp.gorder == 3) { assert!(sp.vids.len() == 1); }     // v8：4–7 不依時間
    if sp.q { assert!(sp.vids.len() == 1, "排隊：一個 job 只能 1 個時間設定"); }
    if sp.gorder >= 4 { assert!(sp.g > 0, "GPU 規則 {} 要開 GPU 層", x[8]); assert!(!(sp.pf_gpu && sp.pf != Pf::No), "GPU 規則 {} 不支援預取進 GPU", x[8]); }
    if sp.gorder == 4 || sp.gorder == 5 { assert!(sp.cue, "cuep／cueo 要 cue=1"); }""")

# ================================================================== RunData：cue.u8、w5.f64（有才讀）
rep("""struct RunData { vspecs: Vec<ViewSpec>, fs_: Vec<Vec<f64>>, fcs: Vec<Vec<f64>>, fc_idle: Vec<f64>, i4: i64 }""",
    """struct RunData { vspecs: Vec<ViewSpec>, fs_: Vec<Vec<f64>>, fcs: Vec<Vec<f64>>, fc_idle: Vec<f64>, i4: i64,
                 cue: Option<Vec<u8>>, w5d: Option<Vec<f64>> }""")
rep("""    RunData { vspecs, fs_, fcs, fc_idle, i4 }""", """    let cue = fs::read(format!("{}/cue.u8", rd_)).ok();
    let w5d = if std::path::Path::new(&format!("{}/w5.f64", rd_)).exists() { Some(rd_f64(&format!("{}/w5.f64", rd_))) } else { None };
    RunData { vspecs, fs_, fcs, fc_idle, i4, cue, w5d }""")

# ================================================================== Eng：新欄位
rep("""    sc: Option<Box<SlowCap>>,          // v6：慢層容量上限（None＝沒有上限＝v5 的路徑）
}""", """    sc: Option<Box<SlowCap>>,          // v6：慢層容量上限（None＝沒有上限＝v5 的路徑）
    // v8 排隊：qon＝開；qfree＝GPU 空出來的時刻（前一個請求完成）；qprev＝前一個請求的開始時刻；qwait＝每個請求的等待
    qon: bool, qfree: f64, qprev: f64, qwait: Vec<f64>, qstart: Vec<f64>, qsvc: Vec<f64>,   // qstart／qsvc＝開始時刻／服務時間（稽核用）
    // closed-loop（q=2）：qd[s]＝session s 到目前為止累積的等待；請求的到達＝紀錄的時刻＋qd[s]（等待會延後這個 session 之後的請求）
    qc: bool, qd: Vec<f64>, qarr: Vec<f64>,
    // v8 W3 線索：cue[r]＝1 → 這一輪的輸出有工具呼叫（同一 session 的下一個請求是工具觸發）
    cue: &'a [u8],
    // v8 W5：cpu_bf＝CPU 層每塊位元組（沒開＝BF）
    cpu_bf: i64,
}""")
rep("""        grp, gep: vec![0; ns], multi, diverged: false, noise, sc,""",
    """        grp, gep: vec![0; ns], multi, diverged: false, noise, sc,
        qon: sp.q, qfree: f64::NEG_INFINITY, qprev: f64::NAN, qwait: if sp.q { vec![0.0; nreq] } else { Vec::new() },
        qstart: if sp.q { vec![0.0; nreq] } else { Vec::new() }, qsvc: if sp.q { vec![0.0; nreq] } else { Vec::new() },
        qc: sp.qc, qd: if sp.qc { vec![0.0; ns] } else { Vec::new() }, qarr: if sp.q { vec![0.0; nreq] } else { Vec::new() },
        cue: if sp.cue { let c = rdat.cue.as_deref().expect("cue=1 但沒有 cue.u8"); assert_eq!(c.len(), nreq, "cue.u8 的長度"); c } else { &[] },
        cpu_bf: sp.w5.unwrap_or(BF),""")
rep("""    let (g, pf) = (sp.g, sp.pf);
    let has_gpf = g > 0 && sp.pf_gpu && pf != Pf::No;""", """    let (g, pf) = (sp.g, sp.pf);
    let has_gpf = g > 0 && sp.pf_gpu && pf != Pf::No;
    if sp.q || sp.cue || sp.w5.is_some() || sp.gorder >= 4 || sp.qpost { assert!(!multi, "v8 的新功能不支援共用前段（multi）"); }""")

# ================================================================== A. 排隊
rep("""        let tr = self.tr;
        let t = tr.t[r];
        if self.g > 0 && self.gorder == 2 { self.g_expire(t); }
        if self.pf != Pf::No {
            let tf = if r > 0 { tr.t[r - 1] } else { t };""", """        let tr = self.tr;
        let arr = if self.qc { tr.t[r] + self.qd[tr.sess[r] as usize] } else { tr.t[r] };   // v8 排隊：closed-loop 的到達含這個 session 累積的等待
        let t = if self.qon && self.qfree > arr { self.qfree } else { arr };   // v8 排隊：開始＝max(到達, 前一個完成)
        if self.g > 0 && self.gorder == 2 { self.g_expire(t); }
        if self.pf != Pf::No {
            let tf = if r > 0 { if self.qon { self.qprev } else { tr.t[r - 1] } } else { t };""")
rep("""        self.cur = r as i64;
        if let Some(cell) = &self.noise {""", """        self.cur = r as i64;
        if self.qon { self.qprev = t; }
        if let Some(cell) = &self.noise {""")
rep("""                tend.push(t + tt);
            }
        }
        // ---- 回合結束：全部 BF16 進 CPU""", """                tend.push(t + tt);
            }
        }
        if self.qon { self.q_note(r, t, tend[0]); }
        // ---- 回合結束：全部 BF16 進 CPU""")
rep("""                tend.push(t + tt);
            }
        }
        if self.gwb { self.turn_end_wb(r, blocks, tend); } else { self.turn_end_wt(r, blocks, tend); }""",
    """                tend.push(t + tt);
            }
        }
        if self.qon { self.q_note(r, t, tend[0]); }
        if self.gwb { self.turn_end_wb(r, blocks, tend); } else { self.turn_end_wt(r, blocks, tend); }""")
rep("""    fn step_gpu(&mut self,""", """    // v8 排隊：記等待、TTFT 加上等待、GPU 忙到這個請求完成（＝開始＋服務時間）
    fn q_note(&mut self, r: usize, t: f64, done: f64) {
        let a = if self.qc { self.tr.t[r] + self.qd[self.tr.sess[r] as usize] } else { self.tr.t[r] };
        let w = t - a;
        assert!(w >= 0.0);
        self.qwait[r] = w;
        self.qarr[r] = a;
        if self.qc { self.qd[self.tr.sess[r] as usize] += w; }
        self.qstart[r] = t;
        self.qsvc[r] = self.views[0].ttft[r];
        self.views[0].ttft[r] += w;
        self.qfree = done;
    }

    fn step_gpu(&mut self,""")
rep("""    if let Some(sc) = &e.sc { wr_i64(&format!("{}/eng_sc.i64", od), &[sc.cap, sc.n_clean, sc.n_drop]); }""",
    """    if let Some(sc) = &e.sc { wr_i64(&format!("{}/eng_sc.i64", od), &[sc.cap, sc.n_clean, sc.n_drop]); }
    if e.qon {
        wr_f64(&format!("{}/qwait.f64", od), &e.qwait);
        wr_f64(&format!("{}/qstart.f64", od), &e.qstart);
        wr_f64(&format!("{}/qsvc.f64", od), &e.qsvc);
        wr_f64(&format!("{}/qarr.f64", od), &e.qarr);
    }""")

# ================================================================== B. W3 線索
rep("""    fn g_add(&mut self, s: usize, te: f64) {
        self.glru.move_to_end(s);
        if self.gorder >= 2 {""", """    fn g_add(&mut self, s: usize, te: f64) {
        self.glru.move_to_end(s);
        if self.gorder == 4 {          // v8 CUEP：這一輪有工具呼叫 → 釘到下一個請求（∞）；沒有 → 隨時可逐出（−∞）＝照 LRU
            self.gexp[s] = if self.cue[self.cur as usize] != 0 { f64::INFINITY } else { f64::NEG_INFINITY };
        } else if self.gorder == 2 || self.gorder == 3 {""")
rep("""        match self.gorder {
            0 => self.glru.head as usize,""", """        match self.gorder {
            0 | 5 | 6 => self.glru.head as usize,           // v8：CUEO 平常照 LRU；APC 的 session 順序＝LRU""")
rep("""        if !blocks.is_empty() {
            self.g_add(s, tend[0]);
            if (self.hook == Hook::H2a && self.nreq[s] < 2) || (self.hook == Hook::Odrop && self.dead(s)) { w += self.drop_(s); }
        } else {
            self.g_remove(s);
        }
        let cw = self.g_enforce(tend[0]);""", """        let mut cw0 = 0i64;
        if !blocks.is_empty() {
            self.g_add(s, tend[0]);
            if (self.hook == Hook::H2a && self.nreq[s] < 2) || (self.hook == Hook::Odrop && self.dead(s)) { w += self.drop_(s); }
            else if self.gorder == 5 && self.cue[r] != 0 { cw0 = self.g_evict_wb(s); }     // v8 CUEO：有工具呼叫 → 立刻寫進 CPU
        } else {
            self.g_remove(s);
        }
        let cw = cw0 + self.g_enforce(tend[0]);""")
rep("""        if keep {
            self.gsb[s] = r as i64;
            self.g_add(s, tend[0]);
        } else {""", """        if keep {
            self.gsb[s] = r as i64;
            self.g_add(s, tend[0]);
            if self.gorder == 5 && self.cue[r] != 0 { self.g_free_wt(s); }                  // v8 CUEO（wt）：放掉 GPU 副本
        } else {""")

# ================================================================== C. W5（CPU 層 INT4＋一次拆完的殘差）
rep("""    ng: Vec<i32>,
}

impl<'a> View<'a> {""", """    ng: Vec<i32>,
    // v8 W5：w5＝開；cpu_b＝CPU 層每塊位元組（沒開＝BF）；dc1／dc2＝讀 CPU 的 BF16 位置／INT4 預取位置的塊之後加的殘差（沒開不用）
    w5: bool, cpu_b: i64, dc1: f64, dc2: f64,
}

impl<'a> View<'a> {""")
rep("""        let l = self.i4 as f64 / self.cpu.r + self.d;
        let (mut p, mut q, mut rt, mut lt) = (0i64, n as i64 - 1, 0.0f64, 0.0f64);""",
    """        let l = self.i4 as f64 / self.cpu.r + if self.w5 { self.dc2 } else { self.d };     // v8 W5
        let (mut p, mut q, mut rt, mut lt) = (0i64, n as i64 - 1, 0.0f64, 0.0f64);""")
rep("""        let c_b = BF as f64 / self.cpu.r;""", """        let c_b = self.cpu_b as f64 / self.cpu.r;                 // v8 W5：CPU 層每塊位元組（沒開＝BF）
        let (cpu_b, w5, dc1, dc2) = (self.cpu_b, self.w5, self.dc1, self.dc2);""")
rep("""                    cf = st + dur; cb += dur; cn += BF; $lt = cf; nc += 1;""",
    """                    cf = st + dur; cb += dur; cn += cpu_b; $lt = if w5 { cf + dc1 } else { cf }; nc += 1;""")
rep("""                    cf = st + dur; cb += dur; cn += i4; $lt = cf + d; nc += 1;""",
    """                    cf = st + dur; cb += dur; cn += i4; $lt = cf + if w5 { dc2 } else { d }; nc += 1;""")
rep("""                self.cpu_used -= if fm == 1 { BF } else { self.i4 };""", """                self.cpu_used -= if fm == 1 { self.cpu_bf } else { self.i4 };""")
rep("""        if self.ref_cpu[b] == 0 && self.fmt[b] == 1 {
            self.cpu_used -= BF;""", """        if self.ref_cpu[b] == 0 && self.fmt[b] == 1 {
            self.cpu_used -= self.cpu_bf;""")
rep("""                let size = self.sbl(s).len() as i64 * BF;""", """                let size = self.sbl(s).len() as i64 * self.cpu_bf;""", n=2)
rep("""                let cps = if e < n { (self.g_cc[n] - self.g_cc[e]) / (((n - e) as i64 * BF) as f64) }
                          else { self.g_c[if n == 0 { self.g_c.len() - 1 } else { n - 1 }] / BF as f64 };
                let h = self.gl + self.nreq[s] as f64 * cps * BF as f64;""",
    """                let cps = if e < n { (self.g_cc[n] - self.g_cc[e]) / (((n - e) as i64 * self.cpu_bf) as f64) }
                          else { self.g_c[if n == 0 { self.g_c.len() - 1 } else { n - 1 }] / self.cpu_bf as f64 };
                let h = self.gl + self.nreq[s] as f64 * cps * self.cpu_bf as f64;""")
rep("""                    let freed = (self.sbl(x).len() - e) as i64 * BF;""", """                    let freed = (self.sbl(x).len() - e) as i64 * self.cpu_bf;""")
rep("""                self.fmt[b] = 1; self.cpu_used += BF; cw += 1;
            } else if fm == 2 {
                self.fmt[b] = 1; self.cpu_used += BF - self.i4; cw += 1;""",
    """                self.fmt[b] = 1; self.cpu_used += self.cpu_bf; cw += 1;
            } else if fm == 2 {
                self.fmt[b] = 1; self.cpu_used += self.cpu_bf - self.i4; if self.cpu_bf != self.i4 { cw += 1; }   // v8（審查 F2）：W5 下 INT4 預取塊轉成 CPU 塊不用寫""", n=3)
rep("""                self.views[0].cpu.write(cw * BF, exp.0);""", """                self.views[0].cpu.write(cw * self.cpu_bf, exp.0);""")
rep("""            self.views[0].cpu.write(cw * BF, now);""", """            self.views[0].cpu.write(cw * self.cpu_bf, now);""")
rep("""            v.cpu.write(cw * BF, te);""", """            v.cpu.write(cw * cpu_bf, te);""", n=3)
rep("""        self.w_slow_blocks += w;
        let i4 = self.i4;
        for (v, &te) in self.views.iter_mut().zip(tend.iter()) {""", """        self.w_slow_blocks += w;
        let (i4, cpu_bf) = (self.i4, self.cpu_bf);
        for (v, &te) in self.views.iter_mut().zip(tend.iter()) {""", n=3)
# 建構：View 的 W5 欄位、GDSF（CPU）成本表的 l_cpu
rep("""               nr: vec![0; nreq], ng: vec![0; nreq] }
    }).collect();""", """               nr: vec![0; nreq], ng: vec![0; nreq],
               w5: sp.w5.is_some(), cpu_b: sp.w5.unwrap_or(BF),
               dc1: w5dv.map(|w| w[2 * j]).unwrap_or(0.0), dc2: w5dv.map(|w| w[2 * j + 1]).unwrap_or(0.0) }
    }).collect();""")
rep("""    let views: Vec<View> = sp.vids.iter().map(|&k| {
        let v = &rdat.vspecs[k];""", """    let w5dv: Option<&Vec<f64>> = if sp.w5.is_some() {
        let w = rdat.w5d.as_ref().expect("w5= 但沒有 w5.f64");
        assert_eq!(w.len(), 2 * sp.vids.len(), "w5.f64：每個時間設定 2 個值");
        assert!(!(sp.pf_gpu && sp.pf != Pf::No), "W5 不支援預取進 GPU");
        Some(w)
    } else { None };
    let views: Vec<View> = sp.vids.iter().enumerate().map(|(j, &k)| {
        let v = &rdat.vspecs[k];""")
rep("""        let l_cpu = BF as f64 / v.cpu.r;""", """        let l_cpu = if v.w5 { v.cpu_b as f64 / v.cpu.r + v.dc1 } else { BF as f64 / v.cpu.r };   // v8 W5""")

# ================================================================== D. APC／BELP（GPU 層 block 粒度：從尾端一塊一塊逐出）
rep("""    cpu_bf: i64,
}""", """    cpu_bf: i64,
    // v8 APC：apc＝GPU 規則是 apc／belp；gn[s]＝s 的 GPU 副本還在 GPU 上的前段塊數；ref_gc[b]＝wb 下「已經寫進 CPU 的乾淨 GPU 副本」的引用數
    //（ref_gpu 裡面的一部分；活的 GPU 引用＝ref_gpu − ref_gc）；n_gblk＝逐出的塊數；apc_whole＝驗證用：一次逐出整個 session（＝GPU LRU／BEL）
    apc: bool, apc_whole: bool, gn: Vec<usize>, ref_gc: Vec<i64>, n_gblk: i64, v8chk: usize,
    // BELP 的捷徑（不改結果；X1FAST_BELP_SLOW=1 關掉＝每塊都全掃）：逐出的塊沒有別人持有時，其他 session 的鍵都不變；
    // 被選的 session 的鍵沒有變小 → 它仍是最大（max_by_key 取最後一個，它後面的都比它小）→ 不用重掃。只在沒有雜訊時用
    belp_fast: bool,
}""")
rep("""        cpu_bf: sp.w5.unwrap_or(BF),""", """        cpu_bf: sp.w5.unwrap_or(BF),
        apc: g > 0 && sp.gorder >= 6, apc_whole: env::var("X1FAST_APC_WHOLE").map(|s| s == "1").unwrap_or(false),
        gn: vec![0; ns], ref_gc: vec![0; if g > 0 { nb } else { 0 }], n_gblk: 0,
        v8chk: env::var("X1FAST_V8_CHECK").ok().and_then(|s| s.parse().ok()).unwrap_or(0),
        belp_fast: g > 0 && sp.gorder == 7 && sp.nz.is_none() && env::var("X1FAST_BELP_SLOW").map(|s| s != "1").unwrap_or(true),""")
# 活的 GPU 引用＝ref_gpu − ref_gc（沒有 APC 時 ref_gc 全 0，和 v7 相同）
rep("""        if self.fix_g { n - self.ref_gpu[b] } else { n }""", """        if self.fix_g { n - (self.ref_gpu[b] - self.ref_gc[b]) } else { n }""")
rep("""            let cond = if self.fix_g { !self.on_slow[b] && self.ref_live[b] - self.ref_gpu[b] > 0 } else { !self.on_slow[b] };""",
    """            let cond = if self.fix_g { !self.on_slow[b] && self.ref_live[b] - (self.ref_gpu[b] - self.ref_gc[b]) > 0 } else { !self.on_slow[b] };""")
# GPU 副本的塊清單：APC 一律用 gsb（wb 下回合結束時設成 sb）
rep("""    fn gbl(&self, s: usize) -> &'a [u32] { if self.gwb { self.sbl(s) } else { self.bl_of(self.gsb[s]) } }""",
    """    fn gbl(&self, s: usize) -> &'a [u32] { if self.gwb && !self.apc { self.sbl(s) } else { self.bl_of(self.gsb[s]) } }
    // v8 APC：GPU 上還在的前段（沒有 APC＝整個副本）
    fn gheld(&self, s: usize) -> &'a [u32] { let b = self.gbl(s); if self.apc { &b[..self.gn[s]] } else { b } }""")
rep("""    fn g_next_raw(&self, x: usize) -> i64 {
        let bl = self.gbl(x);""", """    fn g_next_raw(&self, x: usize) -> i64 {
        let bl = self.gheld(x);""")
rep("""            1 => self.glru.items().into_iter().max_by_key(|&x| (self.g_next(x), -self.last[x])).unwrap(),""",
    """            1 | 7 => self.glru.items().into_iter().max_by_key(|&x| (self.g_next(x), -self.last[x])).unwrap(),   // v8：BELP 選 session 同 BEL""")
rep("""    fn g_remove(&mut self, s: usize) { self.glru.remove(s); self.gpver[s] += 1; }""",
    """    fn g_remove(&mut self, s: usize) { self.glru.remove(s); self.gpver[s] += 1; if self.apc { self.gn[s] = 0; self.gsb[s] = -1; } }""")
rep("""    fn g_add(&mut self, s: usize, te: f64) {
        self.glru.move_to_end(s);""", """    fn g_add(&mut self, s: usize, te: f64) {
        self.glru.move_to_end(s);
        if self.apc { self.gn[s] = self.gbl(s).len(); }                     // v8 APC：回合結束＝整個 session 在 GPU 上""")
# 逐出
rep("""            let x = self.g_select(now);
            if self.gwb { cw += self.g_evict_wb(x); } else { self.g_free_wt(x); }""", """            let x = self.g_select(now);
            if self.apc {                                                      // v8 APC／BELP：一次一塊（驗證用 apc_whole：整個 session）
                if self.apc_whole { while self.glru.inn[x] { cw += self.g_evict_blk(x); } } else { cw += self.g_evict_blk(x); }
                continue;
            }
            if self.gwb { cw += self.g_evict_wb(x); } else { self.g_free_wt(x); }""")
rep("""    fn g_free_wt(&mut self, x: usize) {""", """    // v8 APC／BELP（SEMANTICS 2-1–2-3）：從 x 的 GPU 副本尾端逐出一塊。wb 下 x 還是「只在 GPU」（S_GPU）時，先把整個 session
    // 寫進 CPU（session 進 CPU 層；照 g_evict_wb 的寫法），GPU 上的變成乾淨副本（ref_gc）；之後逐出不用再寫。wt：只放掉 GPU 副本。
    fn g_evict_blk(&mut self, x: usize) -> i64 {
        self.bump(x);
        let mut cw = 0;
        if self.gwb && self.st[x] == S_GPU {
            self.st[x] = S_CPU;
            for &b in self.sbl(x) {
                let b = b as usize;
                self.ref_cpu[b] += 1;
                let fm = self.fmt[b];
                if fm == 0 {
                    self.fmt[b] = 1; self.cpu_used += self.cpu_bf; cw += 1;
                } else if fm == 2 {
                    self.fmt[b] = 1; self.cpu_used += self.cpu_bf - self.i4; if self.cpu_bf != self.i4 { cw += 1; }   // 審查 F2
                    let o = self.pf_owner[b];
                    if o >= 0 { self.spf[o as usize] -= 1; self.pf_owner[b] = -1; }
                }
                self.ref_gc[b] += 1;
            }
            self.order_add(x);
            self.n_gevict += 1;
            self.g_wb_blocks += cw;
        }
        let n = self.gn[x];
        assert!(n > 0 && self.glru.inn[x], "APC：session {} 不在 GPU 上", x);
        let b = self.gbl(x)[n - 1] as usize;
        self.ref_gpu[b] -= 1;
        if self.gwb { self.ref_gc[b] -= 1; }
        if self.ref_gpu[b] == 0 { self.gpu_used -= BF; }
        self.gn[x] = n - 1;
        self.n_gblk += 1;
        if n == 1 {
            self.g_remove(x);
            if !self.gwb { self.n_gevict += 1; }
        }
        cw
    }
    // v8 APC 的稽核（X1FAST_V8_CHECK=k，每 k 個請求）：ref_gpu、ref_gc、gpu_used 從頭重算；S_GPU 的 session 整個在 GPU 上
    fn apc_audit(&self, r: usize) {
        let mut rg = vec![0i64; self.ref_gpu.len()];
        let mut rc = vec![0i64; self.ref_gpu.len()];
        let mut x = self.glru.head;
        while x >= 0 {
            let u = x as usize;
            assert!(self.gn[u] > 0 && self.gn[u] <= self.gbl(u).len(), "r={} APC：session {} 的 gn {}", r, u, self.gn[u]);
            if self.gwb && self.st[u] == S_GPU {
                assert_eq!(self.gn[u], self.sbl(u).len(), "r={} APC：S_GPU 的 session {} 不完整", r, u);
                assert_eq!(self.gsb[u], self.sb[u], "r={} APC：S_GPU 的 session {} 的 gsb", r, u);
            }
            for &b in self.gheld(u) {
                rg[b as usize] += 1;
                if self.gwb && self.st[u] != S_GPU { rc[b as usize] += 1; }
            }
            x = self.glru.next[u];
        }
        assert!(rg == self.ref_gpu, "r={} APC：ref_gpu 和重算的不同", r);
        assert!(rc == self.ref_gc, "r={} APC：ref_gc 和重算的不同", r);
        let used = rg.iter().filter(|&&c| c > 0).count() as i64 * BF;
        assert_eq!(used, self.gpu_used, "r={} APC：gpu_used", r);
        // 審查 F7：ref_live＝S_GPU／S_CPU／S_SLOW 的 session 的塊、ref_cpu＝S_CPU 的、cpu_used＝每塊的格式大小（含 W5）
        let nbk = self.ref_live.len();
        let (mut rl, mut rcp) = (vec![0i64; nbk], vec![0i64; nbk]);
        for u in 0..self.st.len() {
            let su = self.st[u];
            if su == S_GPU || su == S_CPU || su == S_SLOW {
                for &b in self.sbl(u) { rl[b as usize] += 1; if su == S_CPU { rcp[b as usize] += 1; } }
            }
        }
        assert!(rl == self.ref_live, "r={} APC：ref_live 和重算的不同", r);
        assert!(rcp == self.ref_cpu, "r={} APC：ref_cpu 和重算的不同", r);
        let cu: i64 = self.fmt.iter().map(|&f| if f == 1 { self.cpu_bf } else if f == 2 { self.i4 } else { 0 }).sum();
        assert_eq!(cu, self.cpu_used, "r={} APC：cpu_used", r);
        assert!(self.gpu_used <= self.g, "r={} APC：GPU 超過預算", r);
    }
    #[inline]
    fn v8_check(&self, r: usize) {
        if self.v8chk > 0 && self.apc && (r % self.v8chk == 0 || r + 1 == self.tr.t.len()) { self.apc_audit(r); }
    }

    fn g_free_wt(&mut self, x: usize) {""")
# wb 回合結束：放掉乾淨副本；gsb＝這個請求
rep("""        let mut w = 0i64;
        if old_state == S_CPU { self.order_remove(s); }""", """        if self.apc && old_state != S_GPU && self.glru.inn[s] {           // v8 APC（wb）：放掉這個 session 的乾淨 GPU 副本
            let og: &'a [u32] = self.gheld(s);
            for &b in og {
                let b = b as usize;
                self.ref_gpu[b] -= 1; self.ref_gc[b] -= 1;
                if self.ref_gpu[b] == 0 { self.gpu_used -= BF; }
            }
        }
        let mut w = 0i64;
        if old_state == S_CPU { self.order_remove(s); }""")
rep("""        self.sb[s] = r as i64;
        self.st[s] = S_GPU;""", """        self.sb[s] = r as i64;
        self.st[s] = S_GPU;
        if self.apc { self.gsb[s] = r as i64; }""")
# wt 回合結束：舊副本只有前段還在
rep("""        let oldg: Option<&'a [u32]> = if gin { Some(self.bl_of(self.gsb[s])) } else { None };""",
    """        let oldg: Option<&'a [u32]> = if gin { Some(self.gheld(s)) } else { None };      // v8 APC：只有前段還在""")
# 稽核掛在每個請求結束
rep("""        if self.g > 0 { self.step_gpu(r, t, blocks, lay, tend); self.sc_check(r); return; }""",
    """        if self.g > 0 { self.step_gpu(r, t, blocks, lay, tend); self.sc_check(r); self.v8_check(r); return; }""")
rep("""    if e.qon {
        wr_f64(&format!("{}/qwait.f64", od), &e.qwait);""", """    if e.apc { wr_i64(&format!("{}/eng_v8.i64", od), &[e.n_gblk]); }
    if e.qon {
        wr_f64(&format!("{}/qwait.f64", od), &e.qwait);""")

# BELP 的捷徑（g_enforce 開頭）
rep("""    fn g_enforce(&mut self, now: f64) -> i64 {
        let mut cw = 0;
        while self.gpu_used > self.g {""", """    fn g_enforce(&mut self, now: f64) -> i64 {
        let mut cw = 0;
        let mut bc: Option<(usize, (i64, i64))> = None;                       // v8 BELP 捷徑：(上次全掃選到的 session, 那時的最大鍵)
        while self.gpu_used > self.g {
            if self.belp_fast && !self.apc_whole {
                let x = match bc {
                    Some((x0, k0)) if self.glru.inn[x0] && (self.g_next(x0), -self.last[x0]) >= k0 => x0,
                    _ => { let x = self.g_select(now); bc = Some((x, (self.g_next(x), -self.last[x]))); x }
                };
                if self.v8chk > 0 { assert_eq!(x, self.g_select(now), "BELP 捷徑和全掃選的不同"); }
                let b = self.gbl(x)[self.gn[x] - 1] as usize;
                cw += self.g_evict_blk(x);
                if self.ref_gpu[b] != 0 { bc = None; }                       // 逐出的塊還有別人持有 → 別人的鍵可能變了 → 下次全掃
                continue;
            }""")

# ================================================================== E. 排隊的一階近似（qpost=1；SEMANTICS §2b）
# 快取照原本（不排隊、照紀錄順序）跑完；之後每個時間設定各做一次工作守恆的 closed-loop FCFS：
#   每個 session 的第一個請求在紀錄的時刻到；之後的請求到達＝紀錄的時刻＋這個 session 上一個請求「開始服務比紀錄晚多少」；
#   GPU 一次服務一個、照到達先後（同時到照請求序號）、服務時間＝這個時間設定的 TTFT（還原＋prefill）；TTFT 加上等待。
rep("""    cpu_bf: i64,
    // v8 APC：""", """    cpu_bf: i64,
    // v8 排隊的一階近似：qpost＝開；qpw／qps＝每個時間設定的等待、開始時刻（稽核用）
    qpost: bool, qpw: Vec<Vec<f64>>, qps: Vec<Vec<f64>>,
    // v8 APC：""")
rep("""        cpu_bf: sp.w5.unwrap_or(BF),""", """        cpu_bf: sp.w5.unwrap_or(BF),
        qpost: sp.qpost, qpw: Vec::new(), qps: Vec::new(),""")
rep("""    // v8 排隊：記等待、TTFT 加上等待、GPU 忙到這個請求完成（＝開始＋服務時間）""", """    // v8 排隊的一階近似（qpost）：模擬跑完之後呼叫一次
    fn q_post(&mut self) {
        let tr = self.tr;
        let n = tr.t.len();
        let ns = tr.ns;
        let mut first = vec![usize::MAX; ns];
        let mut prev = vec![usize::MAX; ns];
        let mut nxt = vec![usize::MAX; n];
        for r in 0..n {
            let s = tr.sess[r] as usize;
            if first[s] == usize::MAX { first[s] = r; } else { nxt[prev[s]] = r; }
            prev[s] = r;
        }
        for v in self.views.iter_mut() {
            let mut wait = vec![0.0f64; n];
            let mut start = vec![0.0f64; n];
            let mut heap: BinaryHeap<Reverse<(F, usize)>> = BinaryHeap::new();
            for s in 0..ns { if first[s] != usize::MAX { heap.push(Reverse((F(tr.t[first[s]]), first[s]))); } }
            let mut free = f64::NEG_INFINITY;
            let mut done = 0usize;
            while let Some(Reverse((a, r))) = heap.pop() {
                let a = a.0;
                let st = if free > a { free } else { a };
                wait[r] = st - a;
                start[r] = st;
                free = st + v.ttft[r];
                let late = st - tr.t[r];
                let nr = nxt[r];
                if nr != usize::MAX { heap.push(Reverse((F(tr.t[nr] + late), nr))); }
                done += 1;
            }
            assert_eq!(done, n, "qpost：有請求沒排到");
            for r in 0..n { v.ttft[r] += wait[r]; }
            self.qpw.push(wait);
            self.qps.push(start);
        }
    }

    // v8 排隊：記等待、TTFT 加上等待、GPU 忙到這個請求完成（＝開始＋服務時間）""")
rep("""    for r in 0..tr.t.len() { e.step(r, &mut lay, &mut tend); }
    let run_s = tj.elapsed().as_secs_f64();""", """    for r in 0..tr.t.len() { e.step(r, &mut lay, &mut tend); }
    if e.qpost { e.q_post(); }
    let run_s = tj.elapsed().as_secs_f64();""")
rep("""            for r in 0..nreq { e.step(r, &mut lay, &mut tend); }
            return vec![(u[0], finish(&e, sp.jid, tj.elapsed().as_secs_f64(), ""))];""", """            for r in 0..nreq { e.step(r, &mut lay, &mut tend); }
            if e.qpost { e.q_post(); }
            return vec![(u[0], finish(&e, sp.jid, tj.elapsed().as_secs_f64(), ""))];""")
rep("""    if e.apc { wr_i64(&format!("{}/eng_v8.i64", od), &[e.n_gblk]); }""", """    if e.apc { wr_i64(&format!("{}/eng_v8.i64", od), &[e.n_gblk]); }
    for (k, (w, st)) in e.qpw.iter().zip(e.qps.iter()).enumerate() {
        wr_f64(&format!("{}/v{}_qpw.f64", od, k), w);
        wr_f64(&format!("{}/v{}_qps.f64", od, k), st);
    }""")

open(dst, "w", encoding="utf-8").write(s)
print("v7 sha256", V7[:16], "-> v8 sha256", hashlib.sha256(s.encode()).hexdigest())
