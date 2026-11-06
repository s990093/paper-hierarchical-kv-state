"""把 v5（code/p3x/x1fast/src/main.rs，sha256 3642370a…）改成 v6（加慢層容量上限）。只做精確的字串取代，每處都要剛好出現一次。
用法：python -I patch_v6.py <v5 main.rs> <輸出 main.rs>。改了什麼、為什麼，見 README.md。"""
import hashlib, sys

src, dst = sys.argv[1:3]
s = open(src, encoding="utf-8").read()
assert hashlib.sha256(s.encode()).hexdigest().startswith("3642370a"), "輸入不是 v5"


def rep(old, new, n=1):
    global s
    c = s.count(old)
    assert c == n, (c, old[:120])
    s = s.replace(old, new)


# ---- 檔頭
rep("// x1fast：x1_sim.py 的 Engine 全移植", """// icpe-rux v6（2026-11-07，code/icpe/rux_x1fast/v6_slowcap）：v5（code/p3x/x1fast，sha256 3642370a…）＋慢層容量上限
// （x1_sim.SlowOrder＋Engine._sc_*，PREREG §2 第 5 項）。jobs.txt 多一欄 `sc=<frac>,<規則>,<上限塊數>` 才開；沒有這欄時每條路徑和 v5 相同。
// 開上限時照 x1_sim（_fix_fast=False）關掉三個回合結束的快速路徑：每塊都呼叫 fix（fix 會把變乾淨的塊放進 clean_q，少叫一次順序就不同）。
// x1fast：x1_sim.py 的 Engine 全移植""")
rep("use std::collections::BinaryHeap;", "use std::collections::{BinaryHeap, VecDeque};")

# ---- SlowCap 結構（放在 Eng 之前）
rep("#[derive(Clone)]\nstruct Eng<'a> {", r'''// ------------------------------------------------------------------ 慢層容量上限（v6）＝x1_sim.SlowOrder＋Engine._sc_*
#[derive(Clone, Copy, PartialEq, Debug)]
enum SRule { Lru, Lru2, Arc, S3, Gdsf, Pens, Bel }

#[derive(Clone)]
struct SlowCap {
    rule: SRule, cap: i64, c: f64,                       // cap＝⌊frac × n_blocks_full⌋（Python 算好傳進來）；c＝float(cap × i4)
    n_slow: i64, clean_q: VecDeque<u32>, n_clean: i64, n_drop: i64,
    mem: OList, hits: Vec<i64>,                          // mem＝OrderedDict（進入慢層的先後；也是成員集合）
    t1: OList, t2: OList, b1: OList, b2: OList, t1b: i64, t2b: i64, b1b: i64, b2b: i64, p: f64,    // ARC
    s3s: OList, s3m: OList, s3g: OList, f: Vec<i64>, sz: Vec<i64>, sb: i64, mb: i64,              // S3FIFO（f＝−1：不在 dict 裡）
    gver: Vec<i64>, gheap: BinaryHeap<Reverse<(F, i64, i64, usize)>>, gl: f64, g_c: Vec<f64>, g_cc: Vec<f64>,   // GDSF
    check_every: usize, n_ties: i64,                     // 驗證模式（X1FAST_SC_CHECK=k：每 k 個請求稽核一次）
}
impl SlowCap {
    // = SlowOrder.remove：s 離開慢層（hit＝回來了；false＝被丟掉）。不在裡面就什麼都不做。
    fn remove(&mut self, s: usize, hit: bool) {
        if !self.mem.inn[s] { return; }
        self.mem.remove(s);
        if hit { self.hits[s] += 1; }
        match self.rule {
            SRule::Arc => {
                if self.t1.inn[s] { self.t1b -= self.t1.val[s]; self.t1.remove(s); }
                else if self.t2.inn[s] { self.t2b -= self.t2.val[s]; self.t2.remove(s); }
            }
            SRule::S3 => {
                if self.s3s.inn[s] { self.s3s.remove(s); self.sb -= self.sz[s]; }
                else if self.s3m.inn[s] { self.s3m.remove(s); self.mb -= self.sz[s]; }
                self.f[s] = -1;
            }
            SRule::Gdsf => { self.gver[s] += 1; }
            _ => {}
        }
    }
}

#[derive(Clone)]
struct Eng<'a> {''')
rep("    noise: Option<RefCell<Noise>>,\n}\n", "    noise: Option<RefCell<Noise>>,\n    sc: Option<Box<SlowCap>>,          // v6：慢層容量上限（None＝沒有上限＝v5 的路徑）\n}\n")

# ---- fix：v5 的本體改名 fix_inner，外面包 _fix_sc
rep("    fn fix(&mut self, b: usize) -> i64 {\n        if self.ref_live[b] == 0 {",
    """    // = Engine._fix_sc（沒有上限時＝原本的 fix）
    #[inline]
    fn fix(&mut self, b: usize) -> i64 {
        if self.sc.is_none() { return self.fix_inner(b); }
        let was = self.on_slow[b];
        let w = self.fix_inner(b);
        let now = self.on_slow[b];
        let clean = now && self.sc_holders(b) == 0;
        let sc = self.sc.as_mut().unwrap();
        if now != was { sc.n_slow += if now { 1 } else { -1 }; }
        if clean { sc.clean_q.push_back(b as u32); }            // 可能變乾淨了（丟的時候再驗一次）
        w
    }
    // = Engine._slow_holders
    #[inline]
    fn sc_holders(&self, b: usize) -> i64 {
        let n = self.ref_live[b] - self.ref_cpu[b];
        if self.fix_g { n - self.ref_gpu[b] } else { n }
    }
    fn fix_inner(&mut self, b: usize) -> i64 {
        if self.ref_live[b] == 0 {""")

# ---- 慢層的方法（放在 demote 之前）
rep("    // ---------------------------------------------------------- 動作\n    fn demote(&mut self, s: usize) -> i64 {",
    r'''    // ---------------------------------------------------------- 慢層容量上限（v6）
    // = SlowOrder.add（＝_demote_sc 在 demote 之後呼叫）
    fn sc_add(&mut self, s: usize) {
        let i4 = self.i4;
        let size = self.sbl(s).len() as i64 * i4;
        let rule = self.sc.as_ref().unwrap().rule;
        let (mut gh, mut glast) = (0.0f64, 0i64);
        if rule == SRule::Gdsf {
            let bl = self.sbl(s);
            let n = bl.len();
            let ex = Self::excl_start(bl, &self.ref_live);
            let sc = self.sc.as_ref().unwrap();
            let cps = if ex < n { (sc.g_cc[n] - sc.g_cc[ex]) / (((n - ex) as i64 * i4) as f64) }
                      else { sc.g_c[if n == 0 { sc.g_c.len() - 1 } else { n - 1 }] / i4 as f64 };
            gh = sc.gl + self.nreq[s] as f64 * cps * i4 as f64;
            glast = self.last[s];
        }
        let sc = self.sc.as_mut().unwrap();
        sc.mem.move_to_end(s);
        match rule {
            SRule::Arc => {
                if sc.b1.inn[s] {
                    let x = sc.b1.val[s]; sc.b1.remove(s); sc.b1b -= x;
                    sc.p = f64::min(sc.c, sc.p + f64::max(1.0, sc.b2b as f64 / (std::cmp::max(sc.b1b, 1) as f64)) * size as f64);
                    sc.t2.set(s, size); sc.t2b += size;
                } else if sc.b2.inn[s] {
                    let x = sc.b2.val[s]; sc.b2.remove(s); sc.b2b -= x;
                    sc.p = f64::max(0.0, sc.p - f64::max(1.0, sc.b1b as f64 / (std::cmp::max(sc.b2b, 1) as f64)) * size as f64);
                    sc.t2.set(s, size); sc.t2b += size;
                } else if sc.hits[s] > 0 {
                    sc.t2.set(s, size); sc.t2b += size;
                } else {
                    sc.t1.set(s, size); sc.t1b += size;
                }
                while ((sc.t1b + sc.b1b) as f64) > sc.c && sc.b1.len > 0 { let x = sc.b1.pop_first(); sc.b1b -= sc.b1.val[x]; }
                while ((sc.t1b + sc.t2b + sc.b1b + sc.b2b) as f64) > 2.0 * sc.c && sc.b2.len > 0 { let x = sc.b2.pop_first(); sc.b2b -= sc.b2.val[x]; }
            }
            SRule::S3 => {
                sc.sz[s] = size;
                sc.f[s] = std::cmp::min(sc.hits[s], 3);
                if sc.s3g.inn[s] { sc.s3g.remove(s); sc.s3m.move_to_end(s); sc.mb += size; }
                else { sc.s3s.move_to_end(s); sc.sb += size; }
            }
            SRule::Gdsf => {
                let v = sc.gver[s] + 1;
                sc.gver[s] = v;
                sc.gheap.push(Reverse((F(gh), glast, v, s)));
            }
            _ => {}
        }
    }
    // = SlowOrder.select：選一個慢層狀態的 session（呼叫端負責 drop）；沒有就 None。Python 的 min／max 取「第一個」最小／最大，
    // 這裡走訪 mem（插入順序）、只在嚴格更小／更大時才換，所以同分時也和 Python 相同（鍵含 last，本來就不會同分；驗證模式計數）。
    fn sc_select(&mut self) -> Option<usize> {
        let sc = self.sc.as_ref().unwrap();
        if sc.mem.len == 0 { return None; }
        let chk = sc.check_every > 0;
        let mut ties = 0i64;
        let picked: Option<usize> = match sc.rule {
            SRule::Lru => Some(sc.mem.head as usize),
            SRule::Lru2 => {
                let (mut best, mut bk) = (usize::MAX, (0i64, 0i64));
                let mut x = sc.mem.head;
                while x >= 0 {
                    let u = x as usize;
                    let k = (if self.nreq[u] >= 2 { self.prev[u] } else { -1 }, self.last[u]);
                    if best == usize::MAX || k < bk { best = u; bk = k; } else if chk && k == bk { ties += 1; }
                    x = sc.mem.next[u];
                }
                Some(best)
            }
            SRule::Pens => {
                let (cur, fc) = (self.cur, self.pens_fc);
                let (mut best, mut bk) = (usize::MAX, (F(0.0), 0i64));
                let mut x = sc.mem.head;
                while x >= 0 {
                    let u = x as usize;
                    let bl = self.sbl(u);
                    let n = bl.len();
                    let ex = Self::excl_start(bl, &self.ref_live);
                    let cost = if ex < n { fc[n] - fc[ex] } else { fc[n] - fc[if n == 0 { fc.len() - 1 } else { n - 1 }] };
                    let k = (F(cost / (std::cmp::max(1, cur - self.last[u] + 1) as f64)), self.last[u]);
                    if best == usize::MAX || k < bk { best = u; bk = k; } else if chk && k == bk { ties += 1; }
                    x = sc.mem.next[u];
                }
                Some(best)
            }
            SRule::Bel => {
                let (mut best, mut bk) = (usize::MAX, (0i64, 0i64));
                let mut x = sc.mem.head;
                while x >= 0 {
                    let u = x as usize;
                    let k = (self.next_req(u, 1).0, -self.last[u]);      // 雜訊 B2 時每個成員抽一次，順序＝mem 的順序（同 Python）
                    if best == usize::MAX || k > bk { best = u; bk = k; } else if chk && k == bk { ties += 1; }
                    x = sc.mem.next[u];
                }
                Some(best)
            }
            _ => None,
        };
        let sc = self.sc.as_mut().unwrap();
        sc.n_ties += ties;
        let s = match picked {
            Some(s) => s,
            None => match sc.rule {
                SRule::Gdsf => {
                    let mut got = None;
                    while let Some(Reverse((h, _, v, x))) = sc.gheap.pop() {
                        if sc.gver[x] == v && sc.mem.inn[x] { sc.gl = h.0; sc.gver[x] = v + 1; got = Some(x); break; }
                    }
                    match got { Some(x) => x, None => sc.mem.head as usize }
                }
                SRule::Arc => {
                    if sc.t1.len > 0 && ((sc.t1b as f64) > sc.p || sc.t2.len == 0) {
                        let x = sc.t1.pop_first(); let z = sc.t1.val[x];
                        assert!(!sc.b1.inn[x], "ARC：B1 已經有 {}", x);
                        sc.t1b -= z; sc.b1.set(x, z); sc.b1b += z; x
                    } else {
                        let x = sc.t2.pop_first(); let z = sc.t2.val[x];
                        assert!(!sc.b2.inn[x], "ARC：B2 已經有 {}", x);
                        sc.t2b -= z; sc.b2.set(x, z); sc.b2b += z; x
                    }
                }
                SRule::S3 => {
                    let thr = 0.1 * sc.c;
                    loop {
                        if sc.s3s.len > 0 && ((sc.sb as f64) > thr || sc.s3m.len == 0) {
                            let x = sc.s3s.pop_first();
                            sc.sb -= sc.sz[x];
                            if sc.f[x] >= 1 {
                                sc.f[x] = 0;
                                assert!(!sc.s3m.inn[x]);
                                sc.s3m.move_to_end(x); sc.mb += sc.sz[x];
                                continue;
                            }
                            assert!(!sc.s3g.inn[x], "S3FIFO：鬼表已經有 {}", x);
                            sc.s3g.move_to_end(x);
                            while sc.s3g.len > std::cmp::max(sc.s3m.len, 1) { sc.s3g.pop_first(); }
                            sc.f[x] = -1;
                            break x;
                        }
                        let x = sc.s3m.pop_first();
                        if sc.f[x] > 0 { sc.f[x] -= 1; sc.s3m.move_to_end(x); continue; }
                        sc.mb -= sc.sz[x];
                        sc.f[x] = -1;
                        break x;
                    }
                }
                _ => unreachable!(),
            },
        };
        sc.mem.remove(s);
        Some(s)
    }
    // = Engine._slow_enforce（_enforce_sc 在 CPU 的 enforce 之後呼叫）
    fn slow_enforce(&mut self) {
        loop {
            let sc = self.sc.as_mut().unwrap();
            if sc.n_slow <= sc.cap { break; }
            if let Some(b) = sc.clean_q.pop_front() {
                let b = b as usize;
                if self.on_slow[b] && self.sc_holders(b) == 0 && self.fmt[b] != 2 {
                    self.on_slow[b] = false;
                    let sc = self.sc.as_mut().unwrap();
                    sc.n_slow -= 1;
                    sc.n_clean += 1;
                }
                continue;
            }
            let Some(s) = self.sc_select() else { break };
            assert!(self.st[s] == S_SLOW, "慢層丟的 session {} 不是慢層狀態（{}）", s, self.st[s]);
            let _ = self.drop_(s);                   // 回傳的寫入量不算（同 Python：self.drop(s) 的回傳值丟掉）
            self.sc.as_mut().unwrap().n_drop += 1;
        }
    }
    // = Engine.audit 的慢層那段（py:1997–2000）＋驗證模式的同分計數
    fn sc_audit(&self, r: usize) {
        let sc = self.sc.as_ref().unwrap();
        let n = self.on_slow.iter().filter(|&&x| x).count() as i64;
        assert_eq!(sc.n_slow, n, "r={} n_slow 和 on_slow 的總數對不上", r);
        assert!(sc.n_slow <= sc.cap || sc.mem.len == 0, "r={} n_slow {} > cap {}，但慢層結構不是空的", r, sc.n_slow, sc.cap);
        let mut cnt = 0usize;
        for s in 0..self.st.len() {
            let is = self.st[s] == S_SLOW;
            assert_eq!(is, sc.mem.inn[s], "r={} session {} 的慢層狀態和結構對不上", r, s);
            if is { cnt += 1; }
        }
        assert_eq!(cnt, sc.mem.len, "r={} 結構的長度", r);
        assert_eq!(sc.n_ties, 0, "r={} 慢層規則選擇時有同分", r);
    }
    #[inline]
    fn sc_check(&self, r: usize) {
        if let Some(sc) = &self.sc {
            if sc.check_every > 0 && (r % sc.check_every == 0 || r + 1 == self.tr.t.len()) { self.sc_audit(r); }
        }
    }

    // ---------------------------------------------------------- 動作
    fn demote(&mut self, s: usize) -> i64 {''')

# ---- demote 尾端：sorder.add
rep("""            if self.pf == Pf::Opf && t < INF { self.pfheap.push(Reverse((t, s))); }
            else if self.pf == Pf::Rpf { self.pfheap.push(Reverse((-self.last[s], s))); }
        }
        w
    }""", """            if self.pf == Pf::Opf && t < INF { self.pfheap.push(Reverse((t, s))); }
            else if self.pf == Pf::Rpf { self.pfheap.push(Reverse((-self.last[s], s))); }
        }
        if self.sc.is_some() { self.sc_add(s); }            // = _demote_sc
        w
    }""")

# ---- make_room（CPU 預取騰空間）尾端：= py:1580–1581 的 `if self.sc is not None: self._slow_enforce()`
# （第一版漏了這處：Mooncake 的 LRU+RPF × GPU BEL／TTLA 有 22 組不同，run 20261107-004914-icpe-rux-v6-val-mc）
rep("""        if w != 0 {
            let i4 = self.i4;
            self.views[0].slow.write(w * i4, now);
            self.w_slow_blocks += w;
        }
        self.c - self.cpu_used as f64 >= self.i4 as f64
    }""", """        if w != 0 {
            let i4 = self.i4;
            self.views[0].slow.write(w * i4, now);
            self.w_slow_blocks += w;
        }
        if self.sc.is_some() { self.slow_enforce(); }        // = make_room 尾端的 _slow_enforce（py:1580–1581）
        self.c - self.cpu_used as f64 >= self.i4 as f64
    }""")

# ---- drop_ 開頭：慢層狀態的 session 被丟 → 離開結構
rep("""    fn drop_(&mut self, s: usize) -> i64 {
        self.bump(s);
        let st = self.st[s];
        let mut w = 0;
""", """    fn drop_(&mut self, s: usize) -> i64 {
        self.bump(s);
        let st = self.st[s];
        let mut w = 0;
        if st == S_SLOW { if let Some(sc) = self.sc.as_mut() { sc.remove(s, false); } }
""")

# ---- enforce 尾端：_enforce_sc
rep("""                    panic!("CPU 超量 {} > {}，但沒有可以降級的", self.cpu_used, self.c);
                }
            }
        }
        w
    }""", """                    panic!("CPU 超量 {} > {}，但沒有可以降級的", self.cpu_used, self.c);
                }
            }
        }
        if self.sc.is_some() { self.slow_enforce(); }        // = _enforce_sc
        w
    }""")

# ---- step：回來＝命中；GPU 路徑之後也稽核；快速路徑的閘門
rep("""        if let Some(cell) = &self.noise { let mut nz = cell.borrow_mut(); if nz.b1 { let d = Self::nz_sample(&mut nz); nz.nzd[tr.sess[r] as usize] = Some(d); } }
        let blocks = tr.blocks(r);
        if self.g > 0 { return self.step_gpu(r, t, blocks, lay, tend); }""",
    """        if let Some(cell) = &self.noise { let mut nz = cell.borrow_mut(); if nz.b1 { let d = Self::nz_sample(&mut nz); nz.nzd[tr.sess[r] as usize] = Some(d); } }
        if self.sc.is_some() {                                    // 回來了：離開慢層（= step 的 sorder.remove(hit=True)）
            let s0 = tr.sess[r] as usize;
            if self.st[s0] == S_SLOW { self.sc.as_mut().unwrap().remove(s0, true); }
        }
        let blocks = tr.blocks(r);
        if self.g > 0 { self.step_gpu(r, t, blocks, lay, tend); self.sc_check(r); return; }""")
rep("""        let mut cw = 0i64;
        let mut k0 = 0usize;
        if old_state == S_CPU && !old.is_empty() && old.len() <= blocks.len() && blocks[old.len() - 1] == old[old.len() - 1]
            && &blocks[..old.len()] == old { k0 = old.len(); }
        for &b in &blocks[k0..] {
            let b = b as usize;
            self.ref_live[b] += 1;
            self.ref_cpu[b] += 1;
            let fm = self.fmt[b];
            if fm == 0 {
                self.fmt[b] = 1; self.cpu_used += BF; cw += 1;
            } else if fm == 2 {
                self.fmt[b] = 1; self.cpu_used += BF - self.i4; cw += 1;
                let o = self.pf_owner[b];
                if o >= 0 { self.spf[o as usize] -= 1; self.pf_owner[b] = -1; }
            }
        }
        let mut w = 0i64;
        if (old_state == S_CPU || old_state == S_SLOW) && k0 == 0 {
            let isc = old_state == S_CPU;
            for &b in old {
                let b = b as usize;
                self.ref_live[b] -= 1;
                if isc { self.ref_cpu[b] -= 1; }
                if self.ref_live[b] == 0 || (self.ref_cpu[b] == 0 && self.fmt[b] == 1) { w += self.fix(b); }
            }
        }""", """        let mut cw = 0i64;
        let mut k0 = 0usize;
        let fast = self.sc.is_none();           // = _fix_fast（開上限時 False：不跳前綴、每塊都 fix）
        if fast && old_state == S_CPU && !old.is_empty() && old.len() <= blocks.len() && blocks[old.len() - 1] == old[old.len() - 1]
            && &blocks[..old.len()] == old { k0 = old.len(); }
        for &b in &blocks[k0..] {
            let b = b as usize;
            self.ref_live[b] += 1;
            self.ref_cpu[b] += 1;
            let fm = self.fmt[b];
            if fm == 0 {
                self.fmt[b] = 1; self.cpu_used += BF; cw += 1;
            } else if fm == 2 {
                self.fmt[b] = 1; self.cpu_used += BF - self.i4; cw += 1;
                let o = self.pf_owner[b];
                if o >= 0 { self.spf[o as usize] -= 1; self.pf_owner[b] = -1; }
            }
        }
        let mut w = 0i64;
        if (old_state == S_CPU || old_state == S_SLOW) && k0 == 0 {
            let isc = old_state == S_CPU;
            if fast {
                for &b in old {
                    let b = b as usize;
                    self.ref_live[b] -= 1;
                    if isc { self.ref_cpu[b] -= 1; }
                    if self.ref_live[b] == 0 || (self.ref_cpu[b] == 0 && self.fmt[b] == 1) { w += self.fix(b); }
                }
            } else {
                for &b in old {
                    let b = b as usize;
                    self.ref_live[b] -= 1;
                    if isc { self.ref_cpu[b] -= 1; }
                    w += self.fix(b);
                }
            }
        }""", n=2)   # step 與 turn_end_wt 的這段完全相同
rep("""        for (v, &te) in self.views.iter_mut().zip(tend.iter()) {
            v.cpu.write(cw * BF, te);
            v.slow.write(w * i4, te);
        }
    }

    fn step_gpu(""", """        for (v, &te) in self.views.iter_mut().zip(tend.iter()) {
            v.cpu.write(cw * BF, te);
            v.slow.write(w * i4, te);
        }
        self.sc_check(r);
    }

    fn step_gpu(""")

# ---- turn_end_wb：快速路徑的閘門；慢路徑＝全部遞減 → g_release(old) → 全部 fix（py:1844–1852）
rep("""        if old_state == S_GPU && !self.has_gpf && !old.is_empty() && old.len() <= blocks.len()
            && blocks[old.len() - 1] == old[old.len() - 1] && &blocks[..old.len()] == old { k0 = old.len(); }""",
    """        let fast = self.sc.is_none();           // = _fix_fast
        if fast && old_state == S_GPU && !self.has_gpf && !old.is_empty() && old.len() <= blocks.len()
            && blocks[old.len() - 1] == old[old.len() - 1] && &blocks[..old.len()] == old { k0 = old.len(); }""")
rep("""        if (old_state == S_CPU || old_state == S_SLOW || old_state == S_GPU) && k0 == 0 {
            let mut nf = 0i64;""", """        if (old_state == S_CPU || old_state == S_SLOW || old_state == S_GPU) && k0 == 0 && !fast {
            let (isc, isg) = (old_state == S_CPU, old_state == S_GPU);
            for &b in old {
                let b = b as usize;
                self.ref_live[b] -= 1;
                if isc { self.ref_cpu[b] -= 1; }
            }
            if isg { self.g_release(old); }
            for &b in old { w += self.fix(b as usize); }
        } else if (old_state == S_CPU || old_state == S_SLOW || old_state == S_GPU) && k0 == 0 {
            let mut nf = 0i64;""")

# ---- parse_spec：sc= 欄
rep("""// ---- 一個 job 的規格
struct Spec<'b> { jid: &'b str, order: Ord_, act: Act, hook: Hook, pf: Pf, dead_cut: f64, c: f64, g: i64, gorder: u8, gttl: f64,
                  gwb: bool, pf_gpu: bool, vids: Vec<usize>, nz: Option<(bool, f64, bool, i64)> }""",
    """// ---- 一個 job 的規格
struct Spec<'b> { jid: &'b str, order: Ord_, act: Act, hook: Hook, pf: Pf, dead_cut: f64, c: f64, g: i64, gorder: u8, gttl: f64,
                  gwb: bool, pf_gpu: bool, vids: Vec<usize>, nz: Option<(bool, f64, bool, i64)>, sc: Option<(SRule, i64)> }""")
rep("""    let mut nz = None;
    for extra in &x[13..] {""", """    let mut nz = None;
    let mut sc: Option<(SRule, i64)> = None;
    for extra in &x[13..] {
        if let Some(v) = extra.strip_prefix("sc=") {       // v6：慢層容量上限 sc=<frac>,<規則>,<上限塊數>
            let f: Vec<&str> = v.split(',').collect();
            assert!(f.len() == 3, "sc={}", v);
            assert!(parse_f(f[0]) > 0.0, "sc 的比例要 > 0：{}", v);
            let rule = match f[1] { "LRU" => SRule::Lru, "LRU2" => SRule::Lru2, "ARC" => SRule::Arc, "S3FIFO" => SRule::S3,
                                    "GDSF" => SRule::Gdsf, "PENS" => SRule::Pens, "BEL" => SRule::Bel, o => panic!("慢層規則 {}", o) };
            let cap: i64 = f[2].parse().unwrap();
            assert!(cap >= 0, "sc={}", v);
            assert!(sc.is_none(), "sc 欄位重複：{}", line);
            sc = Some((rule, cap));
            continue;
        }""")
rep("""        } else { panic!("jobs.txt 有不支援的欄位 {}（這一版不支援慢層上限）：{}", extra, line); }""",
    """        } else { panic!("jobs.txt 有不支援的欄位 {}：{}", extra, line); }""")
rep("""                    vids: x[12].split(',').map(|v| v.parse().unwrap()).collect(), nz };""",
    """                    vids: x[12].split(',').map(|v| v.parse().unwrap()).collect(), nz, sc };""")

# ---- new_eng：建 SlowCap
rep("""    let (g, pf) = (sp.g, sp.pf);
    let has_gpf = g > 0 && sp.pf_gpu && pf != Pf::No;""", """    let (g, pf) = (sp.g, sp.pf);
    let has_gpf = g > 0 && sp.pf_gpu && pf != Pf::No;
    // v6：慢層容量上限（= Engine._sc_init；上限塊數由 Python 算好）。慢層 GDSF 的成本表＝views[0]（同 x1_sim._job 的 gdsf_view）：
    // g_c[i]＝max(f_i − (i4 ÷ slow_r ＋ d), 0)（SlowOrder.__init__；max 照 Python：0 > x 才取 0）
    let sc = sp.sc.map(|(rule, cap)| {
        assert!(!multi, "慢層上限不支援共用前段（multi）");
        let (mut sg, mut sgc) = (Vec::new(), Vec::new());
        if rule == SRule::Gdsf {
            let v = &views[0];
            let ls = i4 as f64 / v.slow.r + v.d;
            sg = v.f.iter().map(|&fi| { let x = fi - ls; if 0.0 > x { 0.0 } else { x } }).collect();
            sgc.push(0.0);
            for &xx in &sg { let lst = *sgc.last().unwrap(); sgc.push(lst + xx); }
        }
        let check_every: usize = env::var("X1FAST_SC_CHECK").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
        Box::new(SlowCap {
            rule, cap, c: (cap * i4) as f64, n_slow: 0, clean_q: VecDeque::new(), n_clean: 0, n_drop: 0,
            mem: OList::new(ns), hits: vec![0; ns],
            t1: OList::new(ns), t2: OList::new(ns), b1: OList::new(ns), b2: OList::new(ns), t1b: 0, t2b: 0, b1b: 0, b2b: 0, p: 0.0,
            s3s: OList::new(ns), s3m: OList::new(ns), s3g: OList::new(ns), f: vec![-1; ns], sz: vec![0; ns], sb: 0, mb: 0,
            gver: vec![0; ns], gheap: BinaryHeap::new(), gl: 0.0, g_c: sg, g_cc: sgc, check_every, n_ties: 0,
        })
    });""")
rep("""        grp, gep: vec![0; ns], multi, diverged: false, noise,""", """        grp, gep: vec![0; ns], multi, diverged: false, noise, sc,""")

# ---- write_out：慢層的 3 個計數另寫一個檔（eng.i64 照舊 7 個值）
rep("""    wr_f64(&format!("{}/eng.f64", od), &[e.res_sum]);
}""", """    wr_f64(&format!("{}/eng.f64", od), &[e.res_sum]);
    if let Some(sc) = &e.sc { wr_i64(&format!("{}/eng_sc.i64", od), &[sc.cap, sc.n_clean, sc.n_drop]); }
}""")

open(dst, "w", encoding="utf-8").write(s)
print("patched ->", dst, hashlib.sha256(s.encode()).hexdigest())
