// x1fast：x1_sim.py 的 Engine 全移植（第一階段：17 個 CPU 規則、GPU 層關閉／wb／wt、GPU LRU／BEL／TTLA／TTLB、
// 預取到 CPU／GPU、Cake／只載入、多個時間設定）。還沒有：W4 雜訊、慢層容量上限、STAIR 規劃器。
// 每個函式照 x1_sim.py 同名函式的運算順序寫，目標是輸出和 Python 逐位元相同。只為跑分與對拍，不是正式模擬器。
//
// 用法：x1fast <trace_dir> <run_dir>
//   run_dir/views.txt：每行 `id d cpu_r cpu_w cpu_k slow_r slow_w slow_k mode fidx`（數字用 Python repr）；
//   run_dir/f<fidx>.f64、fc<fidx>.f64：重算時間與前綴和；fc_idle.f64：Pensieve 用（GPU 閒的前綴和）
//   run_dir/jobs.txt：每行 `jobid order act hook pf dead_cut cap G gorder gttl gwrite pft view_id,view_id,...`
//   輸出 run_dir/out/<jobid>/...
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::cell::Cell;
use std::env;
use std::fs;
use std::time::Instant;

const BF: i64 = 64 << 20;
const INF: i64 = 1 << 62;
const S_NONE: u8 = 0;
const S_CPU: u8 = 1;
const S_SLOW: u8 = 2;
const S_DROP: u8 = 3;
const S_GPU: u8 = 4;

fn rd(path: &str) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|e| panic!("read {}: {}", path, e))
}
fn rd_f64(p: &str) -> Vec<f64> {
    rd(p).chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
}
fn rd_i64(p: &str) -> Vec<i64> {
    rd(p).chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
}
fn rd_u32(p: &str) -> Vec<u32> {
    rd(p).chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}
fn wr_f64(p: &str, v: &[f64]) {
    let mut b = Vec::with_capacity(v.len() * 8);
    for x in v { b.extend_from_slice(&x.to_le_bytes()); }
    fs::write(p, b).unwrap();
}
fn wr_i64(p: &str, v: &[i64]) {
    let mut b = Vec::with_capacity(v.len() * 8);
    for x in v { b.extend_from_slice(&x.to_le_bytes()); }
    fs::write(p, b).unwrap();
}
fn wr_i32(p: &str, v: &[i32]) {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v { b.extend_from_slice(&x.to_le_bytes()); }
    fs::write(p, b).unwrap();
}

// f64 的全序包裝（沒有 NaN）
#[derive(Clone, Copy, PartialEq, PartialOrd)]
struct F(f64);
impl Eq for F {}
impl Ord for F {
    fn cmp(&self, o: &Self) -> Ordering { self.0.partial_cmp(&o.0).unwrap() }
}

// ------------------------------------------------------------------ I/O 線
#[derive(Clone)]
struct Chan { r: f64, w: f64, k: f64, t_free: f64, t_free_w: f64, busy_r: f64, busy_w: f64, bytes_r: i64, bytes_w: i64 }
impl Chan {
    fn new(r: f64, w: f64, k: f64) -> Self { Chan { r, w, k, t_free: 0.0, t_free_w: 0.0, busy_r: 0.0, busy_w: 0.0, bytes_r: 0, bytes_w: 0 } }
    fn read(&mut self, nbytes: i64, now: f64) -> f64 {
        let st = if now > self.t_free { now } else { self.t_free };
        let mut dur = nbytes as f64 / self.r;
        if self.t_free_w > st { dur *= self.k; }
        self.t_free = st + dur;
        self.busy_r += dur;
        self.bytes_r += nbytes;
        self.t_free
    }
    fn write(&mut self, nbytes: i64, now: f64) {
        if nbytes <= 0 { return; }
        let st = if now > self.t_free_w { now } else { self.t_free_w };
        let dur = nbytes as f64 / self.w;
        self.t_free_w = st + dur;
        self.busy_w += dur;
        self.bytes_w += nbytes;
    }
}

struct ViewSpec { d: f64, cpu: (f64, f64, f64), slow: (f64, f64, f64), cake: bool, fidx: usize }

#[derive(Clone)]
struct View<'a> {
    f: &'a [f64],
    fc: &'a [f64],
    i4: i64,
    d: f64,
    cake: bool,
    cpu: Chan,
    slow: Chan,
    pf_bytes: i64,
    mcpu: Vec<i64>,
    ttft: Vec<f64>,
    rest: Vec<f64>,
    nc: Vec<i32>,
    ns: Vec<i32>,
    nr: Vec<i32>,
    ng: Vec<i32>,
}

impl<'a> View<'a> {
    fn m_cpu(&mut self, n: usize) -> usize {
        if n < self.mcpu.len() && self.mcpu[n] >= 0 { return self.mcpu[n] as usize; }
        let l = self.i4 as f64 / self.cpu.r + self.d;
        let (mut p, mut q, mut rt, mut lt) = (0i64, n as i64 - 1, 0.0f64, 0.0f64);
        while p <= q {
            if rt <= lt { rt += self.f[p as usize]; p += 1; } else { lt += l; q -= 1; }
        }
        if n >= self.mcpu.len() { self.mcpu.resize(n + 1, -1); }
        self.mcpu[n] = p;
        p as usize
    }
    // = View.restore（PLANNER=cake；STAIR 不支援）
    #[allow(clippy::too_many_arguments)]
    fn restore(&mut self, lay: &[u8], blocks: &[u32], l: usize, n_all: usize, t0: f64, old_n: usize, pf_done: &[f64], g: usize,
               gpf_done: Option<&[f64]>) -> (f64, f64, i32, i32, i64) {
        let (mut nc, mut ns) = (0i32, 0i32);
        let (mut cf, cwf, mut cb, mut cn) = (self.cpu.t_free, self.cpu.t_free_w, self.cpu.busy_r, self.cpu.bytes_r);
        let (mut sf, swf, mut sb, mut sn) = (self.slow.t_free, self.slow.t_free_w, self.slow.busy_r, self.slow.bytes_r);
        let c_b = BF as f64 / self.cpu.r;
        let c_bk = c_b * self.cpu.k;
        let c4 = self.i4 as f64 / self.cpu.r;
        let c4k = c4 * self.cpu.k;
        let s4 = self.i4 as f64 / self.slow.r;
        let s4k = s4 * self.slow.k;
        let (d, i4, f) = (self.d, self.i4, self.f);
        let end;
        let meet;
        macro_rules! load_one {
            ($k:expr, $idx:expr, $lt:ident) => {{
                let k = $k;
                if k == 1 {
                    let st = if $lt > cf { $lt } else { cf };
                    let dur = if cwf > st { c_bk } else { c_b };
                    cf = st + dur; cb += dur; cn += BF; $lt = cf; nc += 1;
                } else if k == 3 {
                    let st = if $lt > sf { $lt } else { sf };
                    let dur = if swf > st { s4k } else { s4 };
                    sf = st + dur; sb += dur; sn += i4; $lt = sf + d; ns += 1;
                } else if k == 2 {
                    let pd = pf_done[blocks[$idx] as usize];
                    let nw = if $lt > pd { $lt } else { pd };
                    let st = if nw > cf { nw } else { cf };
                    let dur = if cwf > st { c4k } else { c4 };
                    cf = st + dur; cb += dur; cn += i4; $lt = cf + d; nc += 1;
                } else {
                    let pd = gpf_done.unwrap()[blocks[$idx] as usize];
                    $lt = if $lt > pd { $lt } else { pd };
                }
            }};
        }
        if self.cake {
            let (mut p, mut q, mut rt, mut lt) = (g as i64, l as i64 - 1, t0, t0);
            while p <= q {
                if rt <= lt {
                    rt += f[p as usize];
                    p += 1;
                } else {
                    load_one!(lay[q as usize], q as usize, lt);
                    q -= 1;
                }
            }
            end = if rt > lt { rt } else { lt };
            meet = p;
        } else {
            let mut lt = t0;
            for i in g..l {
                load_one!(lay[i], i, lt);
            }
            end = lt;
            meet = g as i64;
        }
        self.cpu.t_free = cf; self.cpu.busy_r = cb; self.cpu.bytes_r = cn;
        self.slow.t_free = sf; self.slow.busy_r = sb; self.slow.bytes_r = sn;
        let fc = self.fc;
        let base = end - t0;
        let ttft = base + fc[n_all] - fc[l];
        let rest = base + if old_n > l { fc[old_n] - fc[l] } else { 0.0 };
        (ttft, rest, nc, ns, meet - g as i64 + (n_all as i64 - l as i64))
    }
}

// ------------------------------------------------------------------ 有序串列（OrderedDict）與集合
#[derive(Clone)]
struct OList { prev: Vec<i64>, next: Vec<i64>, inn: Vec<bool>, val: Vec<i64>, head: i64, tail: i64, len: usize }
impl OList {
    fn new(n: usize) -> Self { OList { prev: vec![-1; n], next: vec![-1; n], inn: vec![false; n], val: vec![0; n], head: -1, tail: -1, len: 0 } }
    fn remove(&mut self, s: usize) -> bool {
        if !self.inn[s] { return false; }
        let (p, n) = (self.prev[s], self.next[s]);
        if p >= 0 { self.next[p as usize] = n; } else { self.head = n; }
        if n >= 0 { self.prev[n as usize] = p; } else { self.tail = p; }
        self.prev[s] = -1; self.next[s] = -1; self.inn[s] = false; self.len -= 1;
        true
    }
    fn move_to_end(&mut self, s: usize) {
        self.remove(s);
        self.prev[s] = self.tail; self.next[s] = -1;
        if self.tail >= 0 { self.next[self.tail as usize] = s as i64; } else { self.head = s as i64; }
        self.tail = s as i64; self.inn[s] = true; self.len += 1;
    }
    fn set(&mut self, s: usize, v: i64) { self.move_to_end(s); self.val[s] = v; }   // d[s] = v（s 不在裡面時）
    fn pop_first(&mut self) -> usize { let s = self.head as usize; self.remove(s); s }
    fn items(&self) -> Vec<usize> {
        let mut v = Vec::with_capacity(self.len);
        let mut x = self.head;
        while x >= 0 { v.push(x as usize); x = self.next[x as usize]; }
        v
    }
}

#[derive(Clone)]
struct ISet { items: Vec<usize>, pos: Vec<i64> }
impl ISet {
    fn new(n: usize) -> Self { ISet { items: Vec::new(), pos: vec![-1; n] } }
    fn add(&mut self, s: usize) { if self.pos[s] < 0 { self.pos[s] = self.items.len() as i64; self.items.push(s); } }
    fn discard(&mut self, s: usize) {
        let p = self.pos[s];
        if p < 0 { return; }
        let last = *self.items.last().unwrap();
        self.items[p as usize] = last; self.pos[last] = p; self.items.pop(); self.pos[s] = -1;
    }
    fn has(&self, s: usize) -> bool { self.pos[s] >= 0 }
    fn is_empty(&self) -> bool { self.items.is_empty() }
}

struct Trace {
    req_off: Vec<usize>, req_blk: Vec<u32>, nall: Vec<i64>, kind: Vec<i64>, sess: Vec<i64>, parent: Vec<i64>, t: Vec<f64>,
    acc_off: Vec<usize>, acc_req: Vec<i64>, nb: usize, ns: usize,
}
impl Trace {
    fn blocks(&self, r: usize) -> &[u32] { &self.req_blk[self.req_off[r]..self.req_off[r + 1]] }
    fn next_access(&self, b: u32, after: i64) -> i64 {
        let lst = &self.acc_req[self.acc_off[b as usize]..self.acc_off[b as usize + 1]];
        if lst.is_empty() { return INF; }
        let i = lst.partition_point(|&x| x <= after);
        if i < lst.len() { lst[i] } else { INF }
    }
    fn tt(&self, i: i64) -> f64 { if i < 0 { self.t[(self.t.len() as i64 + i) as usize] } else { self.t[i as usize] } }
}

#[derive(Clone, Copy, PartialEq)]
enum Ord_ { Lru, Pred, Lru2, Arc, S3, Gdsf, Pens, Bel, Bels, Freq }
#[derive(Clone, Copy, PartialEq)]
enum Act { Demote, None_, H2w, Onow }
#[derive(Clone, Copy, PartialEq)]
enum Hook { No, H2a, Odrop }
#[derive(Clone, Copy, PartialEq)]
enum Pf { No, Opf, Rpf }

#[derive(Clone)]
struct Eng<'a> {
    tr: &'a Trace,
    views: Vec<View<'a>>,
    order: Ord_, act: Act, hook: Hook, pf: Pf,
    dead_cut: f64,
    c: f64,
    i4: i64,
    ref_cpu: Vec<i64>, ref_live: Vec<i64>, on_slow: Vec<bool>, fmt: Vec<u8>, pf_done: Vec<f64>, pf_owner: Vec<i64>,
    sb: Vec<i64>, st: Vec<u8>, last: Vec<i64>, prev: Vec<i64>, nreq: Vec<i64>, spf: Vec<i64>, st_t: Vec<i64>, pfdone: Vec<bool>,
    cpu_used: i64, cur: i64,
    cpu_set: ISet, lru: OList,
    n_demote: i64, n_drop: i64, n_unpf: i64, res_sum: f64, w_slow_blocks: i64,
    // ARC
    t1: OList, t2: OList, b1: OList, b2: OList, t1b: i64, t2b: i64, b1b: i64, b2b: i64, arc_p: f64,
    // S3-FIFO
    s3s: OList, s3m: OList, s3g: OList, s3f: Vec<i64>, s3sb: i64, s3mb: i64, s3size: Vec<i64>,
    // GDSF
    gver: Vec<i64>, gheap: BinaryHeap<Reverse<(F, i64, i64, usize)>>, gl: f64, g_c: Vec<f64>, g_cc: Vec<f64>,
    pens_fc: &'a [f64],
    pfheap: BinaryHeap<Reverse<(i64, usize)>>, pf_sessions: ISet,
    // GPU
    g: i64, gorder: u8, gttl: f64, gwb: bool, fix_g: bool,
    ref_gpu: Vec<i64>, gpu_used: i64, gsb: Vec<i64>, glru: OList, gexp: Vec<f64>, gpver: Vec<i64>,
    gttlheap: BinaryHeap<Reverse<(F, i64, usize)>>, n_gevict: i64, n_gexpire: i64, g_wb_blocks: i64, sstart: Vec<f64>,
    has_gpf: bool, gpf: Vec<i64>, gpf_done: Vec<f64>, gspf: Vec<i64>, gpf_sessions: ISet,
    // 「下次回來」的快取（精確）：session 的值只取決於它自己的 block 清單與這些 block 的 refcount，以及目前的請求序號。
    // 共用 block 的 session 同一組（union-find）；任何一個 session 的 refcount 有變動 → 這組的 epoch +1 → 快取失效。
    // 同一組沒變動、而且目前請求序號 < 快取的「下次回來」時，答案不變（下次回來＝快取當下之後第一次存取）。
    grp: &'a [u32], gep: Vec<u64>, nx_cache: [Vec<Cell<(i64, usize, u64)>>; 3],
    // 共用前段：multi＝同時維護 LRU／ARC／S3／GDSF 的順序結構（GDSF 只在 g_c 有算時）；第一次要擠 CPU 時設 diverged、不做任何選擇
    multi: bool, diverged: bool,
}

impl<'a> Eng<'a> {
    fn bl_of(&self, r: i64) -> &'a [u32] { if r < 0 { &[] } else { self.tr.blocks(r as usize) } }
    fn sbl(&self, s: usize) -> &'a [u32] { self.bl_of(self.sb[s]) }

    #[inline]
    fn bump(&mut self, s: usize) { self.gep[self.grp[s] as usize] += 1; }

    // ---------------------------------------------------------- _fix／_fix_g
    fn fix(&mut self, b: usize) -> i64 {
        if self.ref_live[b] == 0 {
            let fm = self.fmt[b];
            if fm != 0 {
                self.cpu_used -= if fm == 1 { BF } else { self.i4 };
                self.fmt[b] = 0;
                let o = self.pf_owner[b];
                if o >= 0 { self.spf[o as usize] -= 1; self.pf_owner[b] = -1; }
            }
            self.on_slow[b] = false;
            return 0;
        }
        if self.ref_cpu[b] == 0 && self.fmt[b] == 1 {
            self.cpu_used -= BF;
            self.fmt[b] = 0;
            let cond = if self.fix_g { !self.on_slow[b] && self.ref_live[b] - self.ref_gpu[b] > 0 } else { !self.on_slow[b] };
            if cond { self.on_slow[b] = true; return 1; }
        }
        0
    }

    fn excl_start(bl: &[u32], rf: &[i64]) -> usize {
        let (mut lo, mut hi) = (0usize, bl.len());
        while lo < hi {
            let mid = (lo + hi) >> 1;
            if rf[bl[mid] as usize] <= 1 { hi = mid; } else { lo = mid + 1; }
        }
        lo
    }
    // next_req(s, ref)：ref 0＝ref_cpu、1＝ref_live
    fn next_req(&self, s: usize, which: u8) -> (i64, usize) {
        let ep = self.gep[self.grp[s] as usize] + 1;
        let c = &self.nx_cache[which as usize][s];
        let (cnx, ce, cep) = c.get();
        if cep == ep && (cnx >= INF || self.cur < cnx) { return (cnx, ce); }
        let r = self.next_req_raw(s, which);
        c.set((r.0, r.1, ep));
        r
    }
    fn next_req_raw(&self, s: usize, which: u8) -> (i64, usize) {
        let bl = self.sbl(s);
        let rf = if which == 0 { &self.ref_cpu } else { &self.ref_live };
        let e = Self::excl_start(bl, rf);
        let b = if e < bl.len() { bl[e] } else { bl[bl.len() - 1] };
        (self.tr.next_access(b, self.cur), e)
    }
    fn dead(&self, s: usize) -> bool {
        if self.tr.tt(if self.cur >= 0 { self.cur } else { 0 }) > self.dead_cut { return false; }
        if self.sbl(s).is_empty() { return true; }
        self.next_req(s, 1).0 >= INF
    }

    // ---------------------------------------------------------- 順序結構
    fn order_add(&mut self, s: usize) {
        self.cpu_set.add(s);
        if self.multi {
            for o in [Ord_::Lru, Ord_::Arc, Ord_::S3, Ord_::Gdsf] {
                if o == Ord_::Gdsf && self.g_c.is_empty() { continue; }
                self.order_add_one(o, s);
            }
        } else {
            let o = self.order;
            self.order_add_one(o, s);
        }
    }
    fn order_add_one(&mut self, o: Ord_, s: usize) {
        match o {
            Ord_::Lru => self.lru.move_to_end(s),
            Ord_::Arc => {
                let size = self.sbl(s).len() as i64 * BF;
                if self.t1.inn[s] {
                    self.t1b -= self.t1.val[s]; self.t1.remove(s);
                    self.t2.set(s, size); self.t2b += size;
                } else if self.t2.inn[s] {
                    self.t2b -= self.t2.val[s]; self.t2.remove(s);
                    self.t2.set(s, size); self.t2b += size;
                } else if self.b1.inn[s] {
                    let sz = self.b1.val[s]; self.b1.remove(s);
                    self.b1b -= sz;
                    let delta = f64::max(1.0, self.b2b as f64 / (std::cmp::max(self.b1b, 1) as f64)) * size as f64;
                    self.arc_p = f64::min(self.c, self.arc_p + delta);
                    self.t2.set(s, size); self.t2b += size;
                } else if self.b2.inn[s] {
                    let sz = self.b2.val[s]; self.b2.remove(s);
                    self.b2b -= sz;
                    let delta = f64::max(1.0, self.b1b as f64 / (std::cmp::max(self.b2b, 1) as f64)) * size as f64;
                    self.arc_p = f64::max(0.0, self.arc_p - delta);
                    self.t2.set(s, size); self.t2b += size;
                } else {
                    self.t1.set(s, size); self.t1b += size;
                }
                while ((self.t1b + self.b1b) as f64) > self.c && self.b1.len > 0 {
                    let x = self.b1.pop_first(); self.b1b -= self.b1.val[x];
                }
                while ((self.t1b + self.t2b + self.b1b + self.b2b) as f64) > 2.0 * self.c && self.b2.len > 0 {
                    let x = self.b2.pop_first(); self.b2b -= self.b2.val[x];
                }
            }
            Ord_::S3 => {
                let size = self.sbl(s).len() as i64 * BF;
                if self.s3s.inn[s] || self.s3m.inn[s] {
                    self.s3f[s] = std::cmp::min(self.s3f[s] + 1, 3);
                    let old = self.s3size[s];
                    if self.s3s.inn[s] { self.s3sb += size - old; } else { self.s3mb += size - old; }
                    self.s3size[s] = size;
                } else if self.s3g.inn[s] {
                    self.s3g.remove(s);
                    self.s3m.move_to_end(s);
                    self.s3mb += size; self.s3f[s] = 0; self.s3size[s] = size;
                } else {
                    self.s3s.move_to_end(s);
                    self.s3sb += size; self.s3f[s] = 0; self.s3size[s] = size;
                }
            }
            Ord_::Gdsf => {
                let bl = self.sbl(s);
                let n = bl.len();
                let e = Self::excl_start(bl, &self.ref_cpu);
                let cps = if e < n { (self.g_cc[n] - self.g_cc[e]) / (((n - e) as i64 * BF) as f64) }
                          else { self.g_c[if n == 0 { self.g_c.len() - 1 } else { n - 1 }] / BF as f64 };
                let h = self.gl + self.nreq[s] as f64 * cps * BF as f64;
                let v = self.gver[s] + 1;
                self.gver[s] = v;
                self.gheap.push(Reverse((F(h), self.last[s], v, s)));
            }
            _ => {}
        }
    }
    fn order_remove(&mut self, s: usize) {
        self.cpu_set.discard(s);
        if self.multi {
            for o in [Ord_::Lru, Ord_::Arc, Ord_::S3, Ord_::Gdsf] {
                if o == Ord_::Gdsf && self.g_c.is_empty() { continue; }
                self.order_remove_one(o, s);
            }
        } else {
            let o = self.order;
            self.order_remove_one(o, s);
        }
    }
    fn order_remove_one(&mut self, o: Ord_, s: usize) {
        match o {
            Ord_::Lru => { self.lru.remove(s); }
            Ord_::Arc => {
                if self.t1.inn[s] { self.t1b -= self.t1.val[s]; self.t1.remove(s); }
                else if self.t2.inn[s] { self.t2b -= self.t2.val[s]; self.t2.remove(s); }
            }
            Ord_::S3 => {
                if self.s3s.inn[s] { self.s3s.remove(s); self.s3sb -= self.s3size[s]; }
                else if self.s3m.inn[s] { self.s3m.remove(s); self.s3mb -= self.s3size[s]; }
                self.s3f[s] = -1;
            }
            Ord_::Gdsf => { self.gver[s] += 1; }
            _ => {}
        }
    }
    fn select(&mut self) -> Option<usize> {
        if self.cpu_set.is_empty() { return None; }
        let items: Vec<usize> = self.cpu_set.items.clone();
        match self.order {
            Ord_::Lru => Some(self.lru.head as usize),
            Ord_::Pred => items.iter().copied().max_by_key(|&x| -self.last[x]),
            Ord_::Lru2 => items.iter().copied().min_by_key(|&x| (if self.nreq[x] >= 2 { self.prev[x] } else { -1 }, self.last[x])),
            Ord_::Freq => items.iter().copied().min_by_key(|&x| (self.nreq[x], self.last[x])),
            Ord_::Pens => {
                let cur = self.cur;
                let fc = self.pens_fc;
                items.iter().copied().min_by_key(|&x| {
                    let bl = self.sbl(x);
                    let n = bl.len();
                    let e = Self::excl_start(bl, &self.ref_cpu);
                    let cost = if e < n { fc[n] - fc[e] } else { fc[n] - fc[n - 1] };
                    (F(cost / (std::cmp::max(1, cur - self.last[x] + 1) as f64)), self.last[x])
                })
            }
            Ord_::Bel => items.iter().copied().max_by_key(|&x| (self.next_req(x, 0).0, -self.last[x])),
            Ord_::Bels => {
                let cur = self.cur;
                items.iter().copied().max_by_key(|&x| {
                    let (nx, e) = self.next_req(x, 0);
                    let dist = if nx >= INF { 1e12 } else { (nx - cur) as f64 };
                    let freed = (self.sbl(x).len() - e) as i64 * BF;
                    (F(dist * std::cmp::max(freed, 1) as f64), -self.last[x])
                })
            }
            Ord_::Arc => {
                let s;
                if self.t1.len > 0 && ((self.t1b as f64) > self.arc_p || self.t2.len == 0) {
                    s = self.t1.pop_first(); let sz = self.t1.val[s];
                    self.t1b -= sz; self.b1.set(s, sz); self.b1b += sz;
                } else {
                    s = self.t2.pop_first(); let sz = self.t2.val[s];
                    self.t2b -= sz; self.b2.set(s, sz); self.b2b += sz;
                }
                self.cpu_set.discard(s);
                Some(s)
            }
            Ord_::S3 => {
                let sc = 0.1 * self.c;
                loop {
                    if self.s3s.len > 0 && ((self.s3sb as f64) > sc || self.s3m.len == 0) {
                        let s = self.s3s.pop_first();
                        self.s3sb -= self.s3size[s];
                        if self.s3f[s] >= 1 {
                            self.s3f[s] = 0;
                            self.s3m.move_to_end(s);
                            self.s3mb += self.s3size[s];
                            continue;
                        }
                        self.s3g.move_to_end(s);
                        while self.s3g.len > std::cmp::max(self.s3m.len, 1) { self.s3g.pop_first(); }
                        self.s3f[s] = -1;
                        self.cpu_set.discard(s);
                        return Some(s);
                    }
                    let s = self.s3m.pop_first();
                    if self.s3f[s] > 0 {
                        self.s3f[s] -= 1;
                        self.s3m.move_to_end(s);
                        continue;
                    }
                    self.s3mb -= self.s3size[s];
                    self.s3f[s] = -1;
                    self.cpu_set.discard(s);
                    return Some(s);
                }
            }
            Ord_::Gdsf => {
                while let Some(Reverse((h, _, v, s))) = self.gheap.pop() {
                    if self.gver[s] == v && self.cpu_set.has(s) {
                        self.gl = h.0;
                        self.gver[s] = v + 1;
                        self.cpu_set.discard(s);
                        return Some(s);
                    }
                }
                None
            }
        }
    }

    // ---------------------------------------------------------- 動作
    fn demote(&mut self, s: usize) -> i64 {
        self.bump(s);
        assert!(self.st[s] == S_CPU);
        self.st[s] = S_SLOW;
        self.cpu_set.discard(s);
        if self.order == Ord_::Lru { self.lru.remove(s); }
        let mut w = 0;
        for &b in self.sbl(s) {
            self.ref_cpu[b as usize] -= 1;
            w += self.fix(b as usize);
        }
        self.n_demote += 1;
        self.res_sum += self.tr.tt(self.cur) - self.tr.tt(self.last[s]);
        if self.pf != Pf::No {
            let bl = self.sbl(s);
            let t = self.tr.next_access(bl[bl.len() - 1], self.cur);
            self.st_t[s] = t;
            self.pfdone[s] = false;
            if self.pf == Pf::Opf && t < INF { self.pfheap.push(Reverse((t, s))); }
            else if self.pf == Pf::Rpf { self.pfheap.push(Reverse((-self.last[s], s))); }
        }
        w
    }
    fn drop_(&mut self, s: usize) -> i64 {
        self.bump(s);
        let st = self.st[s];
        let mut w = 0;
        if self.has_gpf && self.gspf[s] > 0 { let bl = self.sbl(s); self.g_unprefetch(s, bl, false); }
        if st == S_CPU || st == S_SLOW {
            if st == S_CPU {
                self.cpu_set.discard(s);
                if self.order == Ord_::Lru { self.lru.remove(s); }
            }
            for &b in self.sbl(s) {
                let b = b as usize;
                if self.pf_owner[b] == s as i64 {
                    self.cpu_used -= self.i4; self.fmt[b] = 0; self.pf_owner[b] = -1; self.spf[s] -= 1;
                }
                self.ref_live[b] -= 1;
                if st == S_CPU { self.ref_cpu[b] -= 1; }
                w += self.fix(b);
            }
            self.pf_sessions.discard(s);
        } else if st == S_GPU {
            self.g_remove(s);
            let bl = self.sbl(s);
            for &b in bl { self.ref_live[b as usize] -= 1; }
            self.g_release(bl);
            for &b in bl { w += self.fix(b as usize); }
        }
        self.st[s] = S_DROP;
        self.sb[s] = -1;
        self.n_drop += 1;
        w
    }
    fn unprefetch(&mut self, s: usize) {
        for &b in self.sbl(s) {
            let b = b as usize;
            if self.pf_owner[b] == s as i64 { self.cpu_used -= self.i4; self.fmt[b] = 0; self.pf_owner[b] = -1; }
        }
        self.spf[s] = 0;
        self.pf_sessions.discard(s);
        self.pfdone[s] = false;
        self.n_unpf += 1;
        if self.pf == Pf::Opf && self.st[s] == S_SLOW && self.st_t[s] < INF { self.pfheap.push(Reverse((self.st_t[s], s))); }
    }
    fn victim_action(&mut self, s: usize) -> i64 {
        match self.act {
            Act::Demote => self.demote(s),
            Act::None_ => self.drop_(s),
            Act::H2w => if self.nreq[s] < 2 { self.drop_(s) } else { self.demote(s) },
            Act::Onow => if self.dead(s) { self.drop_(s) } else { self.demote(s) },
        }
    }
    fn min_pf_by_last(&self, set: &ISet) -> usize { *set.items.iter().min_by_key(|&&x| self.last[x]).unwrap() }
    fn max_pf_by_st(&self, set: &ISet) -> usize { *set.items.iter().max_by_key(|&&x| (self.st_t[x], -self.last[x])).unwrap() }
    fn enforce(&mut self) -> i64 {
        if self.multi && (self.cpu_used as f64) > self.c { self.diverged = true; return 0; }
        let mut w = 0;
        while (self.cpu_used as f64) > self.c {
            if self.pf == Pf::Rpf && !self.pf_sessions.is_empty() {
                let s = self.min_pf_by_last(&self.pf_sessions);
                self.unprefetch(s);
                continue;
            }
            match self.select() {
                Some(s) => w += self.victim_action(s),
                None => {
                    if !self.pf_sessions.is_empty() {
                        let s = if self.pf == Pf::Opf { self.max_pf_by_st(&self.pf_sessions) } else { self.min_pf_by_last(&self.pf_sessions) };
                        self.unprefetch(s);
                        continue;
                    }
                    panic!("CPU 超量 {} > {}，但沒有可以降級的", self.cpu_used, self.c);
                }
            }
        }
        w
    }

    // ---------------------------------------------------------- 預取（CPU）
    fn make_room(&mut self, cand: usize, need: i64, now: f64) -> bool {
        if self.c - self.cpu_used as f64 >= need as f64 { return true; }
        let order: Vec<usize> = if self.pf == Pf::Opf {
            let tc = self.st_t[cand];
            let mut el: Vec<(i64, i64, usize)> = Vec::new();
            for &x in &self.cpu_set.items {
                let nx = self.next_req(x, 0).0;
                if nx > tc { el.push((nx, -self.last[x], x)); }
            }
            el.sort_by(|a, b| b.cmp(a));
            el.into_iter().map(|e| e.2).collect()
        } else {
            let lc = self.last[cand];
            let mut v: Vec<usize> = self.cpu_set.items.iter().copied().filter(|&x| self.last[x] < lc).collect();
            v.sort_by_key(|&x| self.last[x]);
            v
        };
        let mut w = 0;
        for x in order {
            if self.c - self.cpu_used as f64 >= need as f64 { break; }
            self.order_remove(x);
            self.cpu_set.add(x);
            if self.act == Act::Onow && self.dead(x) { w += self.drop_(x); } else { w += self.demote(x); }
        }
        if w != 0 {
            let i4 = self.i4;
            self.views[0].slow.write(w * i4, now);
            self.w_slow_blocks += w;
        }
        self.c - self.cpu_used as f64 >= self.i4 as f64
    }
    fn advance_prefetch(&mut self, t_from: f64, t_to: f64) {
        loop {
            let Some(&Reverse((key, s))) = self.pfheap.peek() else { return };
            let sf = self.views[0].slow.t_free;
            let mut st = if sf > t_from { sf } else { t_from };
            if st >= t_to { return; }
            if self.st[s] != S_SLOW || self.pfdone[s] || (self.pf == Pf::Opf && key != self.st_t[s]) || (self.pf == Pf::Rpf && key != -self.last[s]) {
                self.pfheap.pop();
                continue;
            }
            let bl = self.sbl(s);
            let m = self.views[0].m_cpu(bl.len());
            let todo: Vec<usize> = bl[m..].iter().rev().map(|&b| b as usize)
                .filter(|&b| self.fmt[b] == 0 && self.on_slow[b] && (self.g == 0 || self.ref_gpu[b] == 0)).collect();
            if todo.is_empty() { self.pfdone[s] = true; self.pfheap.pop(); continue; }
            if !self.make_room(s, todo.len() as i64 * self.i4, st) { return; }
            for b in todo {
                let sf = self.views[0].slow.t_free;
                st = if sf > t_from { sf } else { t_from };
                if st >= t_to { return; }
                if self.c - (self.cpu_used as f64) < self.i4 as f64 { return; }
                let i4 = self.i4;
                let done = self.views[0].slow.read(i4, st);
                self.fmt[b] = 2; self.pf_done[b] = done; self.pf_owner[b] = s as i64;
                self.cpu_used += i4; self.spf[s] += 1; self.views[0].pf_bytes += i4;
                self.pf_sessions.add(s);
            }
            self.pfdone[s] = true;
            self.pfheap.pop();
        }
    }

    // ---------------------------------------------------------- GPU 層
    fn gbl(&self, s: usize) -> &'a [u32] { if self.gwb { self.sbl(s) } else { self.bl_of(self.gsb[s]) } }
    fn g_next(&self, x: usize) -> i64 {
        let ep = self.gep[self.grp[x] as usize] + 1;
        let c = &self.nx_cache[2][x];
        let (cnx, ce, cep) = c.get();
        if cep == ep && (cnx >= INF || self.cur < cnx) { let _ = ce; return cnx; }
        let r = self.g_next_raw(x);
        c.set((r, 0, ep));
        r
    }
    fn g_next_raw(&self, x: usize) -> i64 {
        let bl = self.gbl(x);
        let (mut lo, mut hi) = (0usize, bl.len());
        while lo < hi {
            let mid = (lo + hi) >> 1;
            if self.ref_gpu[bl[mid] as usize] <= 1 { hi = mid; } else { lo = mid + 1; }
        }
        let b = if lo < bl.len() { bl[lo] } else { bl[bl.len() - 1] };
        self.tr.next_access(b, self.cur)
    }
    fn g_add(&mut self, s: usize, te: f64) {
        self.glru.move_to_end(s);
        if self.gorder >= 2 {
            self.gexp[s] = te + self.gttl;
            if self.gorder == 2 {
                self.gpver[s] += 1;
                self.gttlheap.push(Reverse((F(self.gexp[s]), self.gpver[s], s)));
            }
        }
    }
    fn g_remove(&mut self, s: usize) { self.glru.remove(s); self.gpver[s] += 1; }
    fn g_select(&self, now: f64) -> usize {
        match self.gorder {
            0 => self.glru.head as usize,
            1 => self.glru.items().into_iter().max_by_key(|&x| (self.g_next(x), -self.last[x])).unwrap(),
            _ => {
                let mut x = self.glru.head;
                while x >= 0 {
                    if self.gexp[x as usize] <= now { return x as usize; }
                    x = self.glru.next[x as usize];
                }
                self.glru.items().into_iter().max_by_key(|&x| (F(self.sstart[x]), x)).unwrap()
            }
        }
    }
    fn g_release(&mut self, bl: &[u32]) {
        for &b in bl {
            let b = b as usize;
            self.ref_gpu[b] -= 1;
            if self.ref_gpu[b] == 0 { self.gpu_used -= BF; }
        }
    }
    fn g_evict_wb(&mut self, x: usize) -> i64 {
        self.bump(x);
        self.g_remove(x);
        self.st[x] = S_CPU;
        let mut cw = 0;
        let bl = self.sbl(x);
        for &b in bl {
            let b = b as usize;
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
        self.g_release(bl);
        self.order_add(x);
        self.n_gevict += 1;
        self.g_wb_blocks += cw;
        if self.has_gpf { self.pf_push(x); }
        cw
    }
    fn g_free_wt(&mut self, x: usize) {
        self.bump(x);
        self.g_remove(x);
        let bl = self.bl_of(self.gsb[x]);
        self.g_release(bl);
        self.gsb[x] = -1;
        self.n_gevict += 1;
        if self.has_gpf && (self.st[x] == S_CPU || self.st[x] == S_SLOW) { self.pf_push(x); }
    }
    fn g_enforce(&mut self, now: f64) -> i64 {
        let mut cw = 0;
        while self.gpu_used > self.g {
            if self.has_gpf && !self.gpf_sessions.is_empty() && (self.pf == Pf::Rpf || self.glru.len == 0) {
                let u = if self.pf == Pf::Rpf { self.min_pf_by_last(&self.gpf_sessions) } else { self.max_pf_by_st(&self.gpf_sessions) };
                let bl = self.sbl(u);
                self.g_unprefetch(u, bl, true);
                continue;
            }
            let x = self.g_select(now);
            if self.gwb { cw += self.g_evict_wb(x); } else { self.g_free_wt(x); }
        }
        cw
    }
    fn g_expire(&mut self, t_now: f64) {
        loop {
            let Some(Reverse((e, _, _))) = self.gttlheap.peek() else { return };
            if !(e.0 <= t_now) { return; }
            let Reverse((exp, ver, x)) = self.gttlheap.pop().unwrap();
            if !self.glru.inn[x] || self.gpver[x] != ver { continue; }
            self.n_gexpire += 1;
            if self.gwb {
                let cw = self.g_evict_wb(x);
                let w = self.enforce();
                self.w_slow_blocks += w;
                let i4 = self.i4;
                self.views[0].cpu.write(cw * BF, exp.0);
                self.views[0].slow.write(w * i4, exp.0);
            } else {
                self.g_free_wt(x);
            }
        }
    }
    fn pf_push(&mut self, s: usize) {
        let bl = self.sbl(s);
        if bl.is_empty() { return; }
        let t = self.tr.next_access(bl[bl.len() - 1], self.cur);
        self.st_t[s] = t;
        self.pfdone[s] = false;
        if self.pf == Pf::Opf && t < INF { self.pfheap.push(Reverse((t, s))); }
        else if self.pf == Pf::Rpf { self.pfheap.push(Reverse((-self.last[s], s))); }
    }
    fn gpf_claim(&mut self, b: usize) {
        let o = self.gpf[b] as usize;
        self.gpf[b] = -1;
        self.gspf[o] -= 1;
        if self.gspf[o] <= 0 { self.gpf_sessions.discard(o); }
    }
    fn g_unprefetch(&mut self, s: usize, bl: &[u32], push: bool) {
        for &b in bl {
            let b = b as usize;
            if self.gpf[b] == s as i64 {
                self.gpf[b] = -1;
                if self.ref_gpu[b] == 0 { self.gpu_used -= BF; }
                self.gspf[s] -= 1;
            }
        }
        assert!(self.gspf[s] == 0, "gspf {} {}", s, self.gspf[s]);
        self.gpf_sessions.discard(s);
        self.pfdone[s] = false;
        if push {
            self.n_unpf += 1;
            if self.pf == Pf::Opf && (self.st[s] == S_CPU || self.st[s] == S_SLOW) && self.st_t[s] < INF {
                self.pfheap.push(Reverse((self.st_t[s], s)));
            }
        }
    }
    fn g_make_room(&mut self, cand: usize, need: i64, now: f64) -> bool {
        if self.g - self.gpu_used >= need { return true; }
        let order: Vec<usize> = if self.pf == Pf::Opf {
            let tc = self.st_t[cand];
            let mut el: Vec<(i64, i64, usize)> = Vec::new();
            for x in self.glru.items() {
                let nx = self.g_next(x);
                if nx > tc { el.push((nx, -self.last[x], x)); }
            }
            el.sort_by(|a, b| b.cmp(a));
            el.into_iter().map(|e| e.2).collect()
        } else {
            let lc = self.last[cand];
            let mut v: Vec<usize> = self.glru.items().into_iter().filter(|&x| self.last[x] < lc).collect();
            v.sort_by_key(|&x| self.last[x]);
            v
        };
        let mut cw = 0;
        for x in order {
            if self.g - self.gpu_used >= need { break; }
            if self.gwb { cw += self.g_evict_wb(x); } else { self.g_free_wt(x); }
        }
        if cw != 0 {
            self.views[0].cpu.write(cw * BF, now);
            let w = self.enforce();
            self.w_slow_blocks += w;
            let i4 = self.i4;
            self.views[0].slow.write(w * i4, now);
        }
        self.g - self.gpu_used >= BF
    }
    fn advance_prefetch_gpu(&mut self, t_from: f64, t_to: f64) {
        while let Some(Reverse((key, s))) = self.pfheap.pop() {
            if !(self.st[s] == S_CPU || self.st[s] == S_SLOW) || self.glru.inn[s] || self.pfdone[s]
                || (self.pf == Pf::Opf && key != self.st_t[s]) || (self.pf == Pf::Rpf && key != -self.last[s]) {
                continue;
            }
            let todo: Vec<usize> = self.sbl(s).iter().rev().map(|&b| b as usize).filter(|&b| self.ref_gpu[b] == 0 && self.gpf[b] < 0).collect();
            if todo.is_empty() { self.pfdone[s] = true; continue; }
            let (ctf, stf) = (self.views[0].cpu.t_free, self.views[0].slow.t_free);
            let sc = if ctf > t_from { ctf } else { t_from };
            let ss = if stf > t_from { stf } else { t_from };
            let st0 = if sc < ss { sc } else { ss };
            if st0 >= t_to || !self.g_make_room(s, todo.len() as i64 * BF, st0) {
                self.pfheap.push(Reverse((key, s)));
                return;
            }
            for b in todo {
                let fm = self.fmt[b];
                let tf = if fm != 0 { self.views[0].cpu.t_free } else { self.views[0].slow.t_free };
                let st = if tf > t_from { tf } else { t_from };
                if st >= t_to || self.g - self.gpu_used < BF {
                    self.pfheap.push(Reverse((key, s)));
                    return;
                }
                let i4 = self.i4;
                let v = &mut self.views[0];
                let done;
                if fm == 1 {
                    done = v.cpu.read(BF, st); v.pf_bytes += BF;
                } else if fm == 2 {
                    let pd = self.pf_done[b];
                    done = v.cpu.read(i4, if st > pd { st } else { pd }) + v.d; v.pf_bytes += i4;
                } else {
                    done = v.slow.read(i4, st) + v.d; v.pf_bytes += i4;
                }
                self.gpf[b] = s as i64;
                self.gpf_done[b] = done;
                self.gpu_used += BF;
                self.gspf[s] += 1;
                self.gpf_sessions.add(s);
            }
            self.pfdone[s] = true;
        }
    }

    // ---------------------------------------------------------- 一個請求
    fn step(&mut self, r: usize, lay: &mut Vec<u8>, tend: &mut Vec<f64>) {
        let tr = self.tr;
        let t = tr.t[r];
        if self.g > 0 && self.gorder == 2 { self.g_expire(t); }
        if self.pf != Pf::No {
            let tf = if r > 0 { tr.t[r - 1] } else { t };
            if self.has_gpf { self.advance_prefetch_gpu(tf, t); } else { self.advance_prefetch(tf, t); }
        }
        self.cur = r as i64;
        let blocks = tr.blocks(r);
        if self.g > 0 { return self.step_gpu(r, t, blocks, lay, tend); }
        lay.clear();
        for &b in blocks {
            let b = b as usize;
            let fm = self.fmt[b];
            if fm == 1 { lay.push(1); } else if fm == 2 { lay.push(2); } else if self.on_slow[b] { lay.push(3); } else { break; }
        }
        let l = lay.len();
        let old_n = if tr.kind[r] == 1 { tr.blocks(tr.parent[r] as usize).len() } else { 0 };
        let n_all = tr.nall[r] as usize;
        tend.clear();
        {
            let pf_done = &self.pf_done;
            for v in self.views.iter_mut() {
                let (tt, rs, nc, ns, nr) = v.restore(lay, blocks, l, n_all, t, old_n, pf_done, 0, None);
                v.ttft[r] = tt; v.rest[r] = rs; v.nc[r] = nc; v.ns[r] = ns; v.nr[r] = nr as i32;
                tend.push(t + tt);
            }
        }
        // ---- 回合結束：全部 BF16 進 CPU
        let s = tr.sess[r] as usize;
        self.bump(s);
        let old = self.sbl(s);
        let old_state = self.st[s];
        let mut cw = 0i64;
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
        }
        if self.spf[s] <= 0 { self.pf_sessions.discard(s); }
        self.sb[s] = r as i64;
        self.st[s] = S_CPU;
        self.prev[s] = self.last[s];
        self.last[s] = r as i64;
        self.nreq[s] += 1;
        if !blocks.is_empty() {
            self.order_add(s);
            if (self.hook == Hook::H2a && self.nreq[s] < 2) || (self.hook == Hook::Odrop && self.dead(s)) {
                self.order_remove(s);
                self.cpu_set.add(s);
                w += self.drop_(s);
            }
        } else {
            self.cpu_set.discard(s);
        }
        w += self.enforce();
        self.w_slow_blocks += w;
        let i4 = self.i4;
        for (v, &te) in self.views.iter_mut().zip(tend.iter()) {
            v.cpu.write(cw * BF, te);
            v.slow.write(w * i4, te);
        }
    }

    fn step_gpu(&mut self, r: usize, t: f64, blocks: &'a [u32], lay: &mut Vec<u8>, tend: &mut Vec<f64>) {
        let tr = self.tr;
        let n = blocks.len();
        let mut g = 0usize;
        while g < n && self.ref_gpu[blocks[g] as usize] > 0 { g += 1; }
        lay.clear();
        lay.resize(g, 0);
        let mut n4 = 0usize;
        for &b in &blocks[g..] {
            let b = b as usize;
            let fm = self.fmt[b];
            if self.has_gpf && self.gpf[b] >= 0 { lay.push(4); n4 += 1; }
            else if fm == 1 { lay.push(1); } else if fm == 2 { lay.push(2); } else if self.on_slow[b] { lay.push(3); } else { break; }
        }
        let l = lay.len();
        let old_n = if tr.kind[r] == 1 { tr.blocks(tr.parent[r] as usize).len() } else { 0 };
        let n_all = tr.nall[r] as usize;
        let g4 = g + n4;
        tend.clear();
        {
            let pf_done = &self.pf_done;
            let gdone = if self.has_gpf { Some(&self.gpf_done[..]) } else { None };
            for v in self.views.iter_mut() {
                let (tt, rs, nc, ns, nr) = v.restore(lay, blocks, l, n_all, t, old_n, pf_done, g, gdone);
                v.ttft[r] = tt; v.rest[r] = rs; v.nc[r] = nc; v.ns[r] = ns; v.nr[r] = nr as i32; v.ng[r] = g4 as i32;
                tend.push(t + tt);
            }
        }
        if self.gwb { self.turn_end_wb(r, blocks, tend); } else { self.turn_end_wt(r, blocks, tend); }
    }

    fn turn_end_wb(&mut self, r: usize, blocks: &'a [u32], tend: &[f64]) {
        let s = self.tr.sess[r] as usize;
        self.bump(s);
        let old = self.sbl(s);
        let old_state = self.st[s];
        let mut k0 = 0usize;
        if old_state == S_GPU && !self.has_gpf && !old.is_empty() && old.len() <= blocks.len()
            && blocks[old.len() - 1] == old[old.len() - 1] && &blocks[..old.len()] == old { k0 = old.len(); }
        if !self.has_gpf {
            let mut nn = 0i64;
            for &b in &blocks[k0..] {
                let b = b as usize;
                self.ref_live[b] += 1;
                let x = self.ref_gpu[b] + 1;
                self.ref_gpu[b] = x;
                if x == 1 { nn += 1; }
            }
            self.gpu_used += nn * BF;
        } else {
            for &b in blocks {
                let b = b as usize;
                self.ref_live[b] += 1;
                self.ref_gpu[b] += 1;
                if self.ref_gpu[b] == 1 {
                    if self.gpf[b] >= 0 { self.gpf_claim(b); } else { self.gpu_used += BF; }
                }
            }
            if self.gspf[s] > 0 { self.g_unprefetch(s, old, false); }
        }
        let mut w = 0i64;
        if old_state == S_CPU { self.order_remove(s); }
        if (old_state == S_CPU || old_state == S_SLOW || old_state == S_GPU) && k0 == 0 {
            let mut nf = 0i64;
            let (isc, isg) = (old_state == S_CPU, old_state == S_GPU);
            for &b in old {
                let b = b as usize;
                self.ref_live[b] -= 1;
                if isc { self.ref_cpu[b] -= 1; }
                else if isg {
                    let x = self.ref_gpu[b] - 1;
                    self.ref_gpu[b] = x;
                    if x == 0 { nf += 1; }
                }
                if self.ref_live[b] == 0 || (self.ref_cpu[b] == 0 && self.fmt[b] == 1) {
                    if nf != 0 { self.gpu_used -= nf * BF; nf = 0; }
                    w += self.fix(b);
                }
            }
            self.gpu_used -= nf * BF;
        }
        if self.pf != Pf::No {
            for &b in blocks {
                let b = b as usize;
                if self.fmt[b] == 2 && self.pf_owner[b] == s as i64 {
                    self.cpu_used -= self.i4; self.fmt[b] = 0; self.pf_owner[b] = -1; self.spf[s] -= 1;
                }
            }
        }
        if self.spf[s] <= 0 { self.pf_sessions.discard(s); }
        self.sb[s] = r as i64;
        self.st[s] = S_GPU;
        self.prev[s] = self.last[s];
        self.last[s] = r as i64;
        self.nreq[s] += 1;
        if !blocks.is_empty() {
            self.g_add(s, tend[0]);
            if (self.hook == Hook::H2a && self.nreq[s] < 2) || (self.hook == Hook::Odrop && self.dead(s)) { w += self.drop_(s); }
        } else {
            self.g_remove(s);
        }
        let cw = self.g_enforce(tend[0]);
        w += self.enforce();
        self.w_slow_blocks += w;
        let i4 = self.i4;
        for (v, &te) in self.views.iter_mut().zip(tend.iter()) {
            v.cpu.write(cw * BF, te);
            v.slow.write(w * i4, te);
        }
    }

    fn turn_end_wt(&mut self, r: usize, blocks: &'a [u32], tend: &[f64]) {
        let s = self.tr.sess[r] as usize;
        self.bump(s);
        let old = self.sbl(s);
        let old_state = self.st[s];
        let mut cw = 0i64;
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
        }
        if self.spf[s] <= 0 { self.pf_sessions.discard(s); }
        self.sb[s] = r as i64;
        self.st[s] = S_CPU;
        self.prev[s] = self.last[s];
        self.last[s] = r as i64;
        self.nreq[s] += 1;
        if !blocks.is_empty() {
            self.order_add(s);
            if (self.hook == Hook::H2a && self.nreq[s] < 2) || (self.hook == Hook::Odrop && self.dead(s)) {
                self.order_remove(s);
                self.cpu_set.add(s);
                w += self.drop_(s);
            }
        } else {
            self.cpu_set.discard(s);
        }
        // ---- GPU 那份
        let gin = self.glru.inn[s];
        let oldg: Option<&'a [u32]> = if gin { Some(self.bl_of(self.gsb[s])) } else { None };
        let keep = !blocks.is_empty() && self.st[s] != S_DROP;
        let mut g0 = 0usize;
        if keep && !self.has_gpf {
            if let Some(og) = oldg {
                if !og.is_empty() && og.len() <= blocks.len() && blocks[og.len() - 1] == og[og.len() - 1] && &blocks[..og.len()] == og { g0 = og.len(); }
            }
        }
        if keep {
            for &b in &blocks[g0..] {
                let b = b as usize;
                self.ref_gpu[b] += 1;
                if self.ref_gpu[b] == 1 {
                    if self.has_gpf && self.gpf[b] >= 0 { self.gpf_claim(b); } else { self.gpu_used += BF; }
                }
            }
        }
        if self.has_gpf && self.gspf[s] > 0 { self.g_unprefetch(s, old, false); }
        if let Some(og) = oldg { if g0 == 0 { self.g_release(og); } }
        if keep {
            self.gsb[s] = r as i64;
            self.g_add(s, tend[0]);
        } else {
            self.gsb[s] = -1;
            self.g_remove(s);
        }
        self.g_enforce(tend[0]);
        w += self.enforce();
        self.w_slow_blocks += w;
        let i4 = self.i4;
        for (v, &te) in self.views.iter_mut().zip(tend.iter()) {
            v.cpu.write(cw * BF, te);
            v.slow.write(w * i4, te);
        }
    }
}

fn parse_f(s: &str) -> f64 { s.parse::<f64>().unwrap_or_else(|_| panic!("float {}", s)) }

fn main() {
    let a: Vec<String> = env::args().collect();
    let (td, rd_) = (&a[1], &a[2]);
    let t0 = Instant::now();
    let meta = rd_i64(&format!("{}/meta.i64", td));
    let (nreq, nb, ns) = (meta[0] as usize, meta[1] as usize, meta[2] as usize);
    let tr = Trace {
        req_off: rd_i64(&format!("{}/req_off.i64", td)).iter().map(|&x| x as usize).collect(),
        req_blk: rd_u32(&format!("{}/req_blk.u32", td)),
        nall: rd_i64(&format!("{}/nall.i64", td)),
        kind: rd_i64(&format!("{}/kind.i64", td)),
        sess: rd_i64(&format!("{}/sess.i64", td)),
        parent: rd_i64(&format!("{}/parent.i64", td)),
        t: rd_f64(&format!("{}/t.f64", td)),
        acc_off: rd_i64(&format!("{}/acc_off.i64", td)).iter().map(|&x| x as usize).collect(),
        acc_req: rd_i64(&format!("{}/acc_req.i64", td)),
        nb, ns,
    };
    assert_eq!(tr.t.len(), nreq);
    assert_eq!(tr.acc_off.len(), nb + 1);
    // session 分組：共用任何 block 的 session 同一組（union-find）
    let mut uf: Vec<u32> = (0..ns as u32).collect();
    fn find(uf: &mut Vec<u32>, x: u32) -> u32 {
        let mut r = x;
        while uf[r as usize] != r { r = uf[r as usize]; }
        let mut y = x;
        while uf[y as usize] != r { let n = uf[y as usize]; uf[y as usize] = r; y = n; }
        r
    }
    {
        let mut owner: Vec<u32> = vec![u32::MAX; nb];
        for r in 0..nreq {
            let s = tr.sess[r] as u32;
            for &b in tr.blocks(r) {
                let o = owner[b as usize];
                if o == u32::MAX { owner[b as usize] = s; } else {
                    let (a, c) = (find(&mut uf, s), find(&mut uf, o));
                    if a != c { uf[a as usize] = c; }
                }
            }
        }
    }
    let grp: Vec<u32> = (0..ns as u32).map(|x| find(&mut uf, x)).collect();
    {
        let mut sz = vec![0usize; ns];
        for &g in &grp { sz[g as usize] += 1; }
        let ng = sz.iter().filter(|&&x| x > 0).count();
        eprintln!("session groups: {} groups for {} sessions, largest {}", ng, ns, sz.iter().max().unwrap());
    }
    let mut sstart = vec![f64::INFINITY; ns];
    for r in 0..nreq {
        let s = tr.sess[r] as usize;
        if sstart[s] == f64::INFINITY { sstart[s] = tr.t[r]; }
    }
    // 時間設定
    let mut vspecs: Vec<ViewSpec> = Vec::new();
    let mut nf = 0usize;
    for line in fs::read_to_string(format!("{}/views.txt", rd_)).unwrap().lines() {
        let x: Vec<&str> = line.split_whitespace().collect();
        if x.is_empty() { continue; }
        assert_eq!(x[0].parse::<usize>().unwrap(), vspecs.len());
        let fidx: usize = x[9].parse().unwrap();
        nf = nf.max(fidx + 1);
        vspecs.push(ViewSpec { d: parse_f(x[1]), cpu: (parse_f(x[2]), parse_f(x[3]), parse_f(x[4])),
                               slow: (parse_f(x[5]), parse_f(x[6]), parse_f(x[7])), cake: x[8] == "cake", fidx });
    }
    let fs_: Vec<Vec<f64>> = (0..nf).map(|i| rd_f64(&format!("{}/f{}.f64", rd_, i))).collect();
    let fcs: Vec<Vec<f64>> = (0..nf).map(|i| rd_f64(&format!("{}/fc{}.f64", rd_, i))).collect();
    let fc_idle = rd_f64(&format!("{}/fc_idle.f64", rd_));
    let i4: i64 = fs::read_to_string(format!("{}/i4.txt", rd_)).unwrap().trim().parse().unwrap();
    let t_load = t0.elapsed().as_secs_f64();
    let mut timing = format!("trace_load_s {}\n", t_load);
    // 多執行緒：X1FAST_THREADS 個執行緒共用同一份唯讀紀錄，從共用佇列各取一個 job；每個 job 的計算和單執行緒完全相同
    let nthreads: usize = env::var("X1FAST_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let summary_only = env::var("X1FAST_SUMMARY_ONLY").map(|s| s == "1").unwrap_or(false);
    let jobs_txt = fs::read_to_string(format!("{}/jobs.txt", rd_)).unwrap();
    let lines: Vec<&str> = jobs_txt.lines().filter(|l| !l.trim().is_empty()).collect();

    // ---- 一個 job 的規格
    struct Spec<'b> { jid: &'b str, order: Ord_, act: Act, hook: Hook, pf: Pf, dead_cut: f64, c: f64, g: i64, gorder: u8, gttl: f64,
                      gwb: bool, pf_gpu: bool, vids: Vec<usize> }
    let parse = |line: &'static str| -> Spec<'static> {
        let x: Vec<&str> = line.split_whitespace().collect();
        let order = match x[1] { "lru" => Ord_::Lru, "pred" => Ord_::Pred, "lru2" => Ord_::Lru2, "arc" => Ord_::Arc, "s3" => Ord_::S3,
                                 "gdsf" => Ord_::Gdsf, "pens" => Ord_::Pens, "bel" => Ord_::Bel, "bels" => Ord_::Bels, "freq" => Ord_::Freq,
                                 o => panic!("order {}", o) };
        let act = match x[2] { "demote" => Act::Demote, "none" => Act::None_, "2hw" => Act::H2w, "onow" => Act::Onow, o => panic!("act {}", o) };
        let hook = match x[3] { "-" => Hook::No, "2ha" => Hook::H2a, "odrop" => Hook::Odrop, o => panic!("hook {}", o) };
        let pf = match x[4] { "-" => Pf::No, "opf" => Pf::Opf, "rpf" => Pf::Rpf, o => panic!("pf {}", o) };
        let gorder: u8 = match x[8] { "-" => 0, "lru" => 0, "bel" => 1, "ttla" => 2, "ttlb" => 3, o => panic!("gorder {}", o) };
        let sp = Spec { jid: x[0], order, act, hook, pf, dead_cut: parse_f(x[5]), c: parse_f(x[6]), g: x[7].parse().unwrap(), gorder,
                        gttl: if x[9] == "-" { 0.0 } else { parse_f(x[9]) }, gwb: x[10] != "wt", pf_gpu: x[11] == "gpu",
                        vids: x[12].split(',').map(|v| v.parse().unwrap()).collect() };
        if sp.pf != Pf::No { assert!(sp.vids.len() == 1); }
        if sp.g > 0 && sp.gorder >= 2 { assert!(sp.vids.len() == 1); }
        sp
    };
    let build = |sp: &Spec, order: Ord_, act: Act, multi: bool, need_gdsf: bool| -> Eng {
        let views: Vec<View> = sp.vids.iter().map(|&k| {
            let v = &vspecs[k];
            View { f: &fs_[v.fidx], fc: &fcs[v.fidx], i4, d: v.d, cake: v.cake,
                   cpu: Chan::new(v.cpu.0, v.cpu.1, v.cpu.2), slow: Chan::new(v.slow.0, v.slow.1, v.slow.2), pf_bytes: 0,
                   mcpu: Vec::new(), ttft: vec![0.0; nreq], rest: vec![0.0; nreq], nc: vec![0; nreq], ns: vec![0; nreq],
                   nr: vec![0; nreq], ng: vec![0; nreq] }
        }).collect();
        let (mut g_c, mut g_cc) = (Vec::new(), Vec::new());
        if order == Ord_::Gdsf || need_gdsf {
            let v = &views[0];
            let l_cpu = BF as f64 / v.cpu.r;
            let l_sl = i4 as f64 / v.slow.r + v.d;
            g_c = v.f.iter().map(|&fi| f64::max(f64::min(fi, l_sl) - l_cpu, 0.0)).collect();
            g_cc.push(0.0);
            for &xx in &g_c { let lst = *g_cc.last().unwrap(); g_cc.push(lst + xx); }
        }
        let (g, pf) = (sp.g, sp.pf);
        let has_gpf = g > 0 && sp.pf_gpu && pf != Pf::No;
        Eng {
            tr: &tr, views, order, act, hook: sp.hook, pf, dead_cut: sp.dead_cut, c: sp.c, i4,
            ref_cpu: vec![0; nb], ref_live: vec![0; nb], on_slow: vec![false; nb], fmt: vec![0; nb], pf_done: vec![0.0; nb], pf_owner: vec![-1; nb],
            sb: vec![-1; ns], st: vec![S_NONE; ns], last: vec![-1; ns], prev: vec![-1; ns], nreq: vec![0; ns], spf: vec![0; ns],
            st_t: vec![INF; ns], pfdone: vec![false; ns],
            cpu_used: 0, cur: -1, cpu_set: ISet::new(ns), lru: OList::new(ns),
            n_demote: 0, n_drop: 0, n_unpf: 0, res_sum: 0.0, w_slow_blocks: 0,
            t1: OList::new(ns), t2: OList::new(ns), b1: OList::new(ns), b2: OList::new(ns), t1b: 0, t2b: 0, b1b: 0, b2b: 0, arc_p: 0.0,
            s3s: OList::new(ns), s3m: OList::new(ns), s3g: OList::new(ns), s3f: vec![-1; ns], s3sb: 0, s3mb: 0, s3size: vec![0; ns],
            gver: vec![0; ns], gheap: BinaryHeap::new(), gl: 0.0, g_c, g_cc, pens_fc: &fc_idle,
            pfheap: BinaryHeap::new(), pf_sessions: ISet::new(ns),
            g, gorder: sp.gorder, gttl: sp.gttl, gwb: sp.gwb, fix_g: g > 0 && sp.gwb,
            ref_gpu: vec![0; if g > 0 { nb } else { 0 }], gpu_used: 0, gsb: vec![-1; ns], glru: OList::new(ns), gexp: vec![0.0; ns],
            gpver: vec![0; ns], gttlheap: BinaryHeap::new(), n_gevict: 0, n_gexpire: 0, g_wb_blocks: 0, sstart: sstart.clone(),
            has_gpf, gpf: vec![-1; if has_gpf { nb } else { 0 }], gpf_done: vec![0.0; if has_gpf { nb } else { 0 }],
            gspf: vec![0; ns], gpf_sessions: ISet::new(ns),
            grp: &grp, gep: vec![0; ns], multi, diverged: false,
            nx_cache: [(0..ns).map(|_| Cell::new((0, 0, 0))).collect(), (0..ns).map(|_| Cell::new((0, 0, 0))).collect(),
                       (0..ns).map(|_| Cell::new((0, 0, 0))).collect()],
        }
    };
    let finish = |e: &Eng, jid: &str, run_s: f64, extra: &str| -> String {
        if summary_only {
            let mut sline = format!("{} {}", jid, run_s);
            for v in e.views.iter() {
                let (mut sum, mut n) = (0.0f64, 0usize);
                for r in 0..nreq { if tr.kind[r] == 1 { sum += v.ttft[r]; n += 1; } }
                sline.push_str(&format!(" {:?}", sum / n as f64));
            }
            sline.push_str(extra);
            sline.push('\n');
            return sline;
        }
        let od = format!("{}/out/{}", rd_, jid);
        fs::create_dir_all(&od).unwrap();
        for (k, v) in e.views.iter().enumerate() {
            wr_f64(&format!("{}/v{}_ttft.f64", od, k), &v.ttft);
            wr_f64(&format!("{}/v{}_rest.f64", od, k), &v.rest);
            wr_i32(&format!("{}/v{}_nc.i32", od, k), &v.nc);
            wr_i32(&format!("{}/v{}_ns.i32", od, k), &v.ns);
            wr_i32(&format!("{}/v{}_nr.i32", od, k), &v.nr);
            wr_i32(&format!("{}/v{}_ng.i32", od, k), &v.ng);
            wr_f64(&format!("{}/v{}_chan.f64", od, k), &[v.cpu.t_free, v.cpu.t_free_w, v.cpu.busy_r, v.cpu.busy_w,
                v.slow.t_free, v.slow.t_free_w, v.slow.busy_r, v.slow.busy_w]);
            wr_i64(&format!("{}/v{}_chanb.i64", od, k), &[v.cpu.bytes_r, v.cpu.bytes_w, v.slow.bytes_r, v.slow.bytes_w, v.pf_bytes]);
        }
        wr_i64(&format!("{}/eng.i64", od), &[e.n_demote, e.n_drop, e.n_unpf, e.n_gevict, e.n_gexpire, e.g_wb_blocks, e.w_slow_blocks]);
        wr_f64(&format!("{}/eng.f64", od), &[e.res_sum]);
        format!("{} {}{}\n", jid, run_s, extra)
    };
    // ---- 分組：X1FAST_FORK=1 時，同一組設定（除了 CPU 的順序與動作）、沒有 hook 也沒有預取的 job 共用前段
    let fork = env::var("X1FAST_FORK").map(|s| s == "1").unwrap_or(false);
    let jobs_static: &'static str = Box::leak(jobs_txt.clone().into_boxed_str());
    let all: Vec<&'static str> = jobs_static.lines().filter(|l| !l.trim().is_empty()).collect();
    let mut units: Vec<Vec<usize>> = Vec::new();
    {
        let mut key_of: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for (i, l) in all.iter().enumerate() {
            let x: Vec<&str> = l.split_whitespace().collect();
            if fork && x[3] == "-" && x[4] == "-" {
                let key = format!("{} {}", x[3..].join(" "), "");
                if let Some(&u) = key_of.get(&key) { units[u].push(i); continue; }
                key_of.insert(key, units.len());
            }
            units.push(vec![i]);
        }
    }
    let run_unit = |u: &Vec<usize>| -> Vec<(usize, String)> {
        let mut lay: Vec<u8> = Vec::new();
        let mut tend: Vec<f64> = Vec::new();
        if u.len() == 1 {
            let sp = parse(all[u[0]]);
            let tj = Instant::now();
            let mut e = build(&sp, sp.order, sp.act, false, false);
            for r in 0..nreq { e.step(r, &mut lay, &mut tend); }
            return vec![(u[0], finish(&e, sp.jid, tj.elapsed().as_secs_f64(), ""))];
        }
        let specs: Vec<Spec> = u.iter().map(|&i| parse(all[i])).collect();
        let need_gdsf = specs.iter().any(|p| p.order == Ord_::Gdsf);
        let tg = Instant::now();
        // 第 1 趟：找第一次要擠 CPU 的請求 r*
        let mut probe = build(&specs[0], Ord_::Lru, Act::Demote, true, need_gdsf);
        let mut rstar = nreq;
        for r in 0..nreq {
            probe.step(r, &mut lay, &mut tend);
            if probe.diverged { rstar = r; break; }
        }
        drop(probe);
        // 第 2 趟：共用前段跑到 r* 之前，複製給每個規則
        let mut shared = build(&specs[0], Ord_::Lru, Act::Demote, true, need_gdsf);
        for r in 0..rstar { shared.step(r, &mut lay, &mut tend); }
        assert!(!shared.diverged);
        let t_shared = tg.elapsed().as_secs_f64();
        let mut out = Vec::new();
        for (k, sp) in specs.iter().enumerate() {
            let tj = Instant::now();
            let mut e = shared.clone();
            e.order = sp.order; e.act = sp.act; e.multi = false; e.dead_cut = sp.dead_cut;
            if sp.order != Ord_::Gdsf && !need_gdsf { e.g_c.clear(); }
            for r in rstar..nreq { e.step(r, &mut lay, &mut tend); }
            let extra = format!(" fork_rstar {} fork_n {} fork_shared_s {}", rstar, specs.len(), t_shared);
            out.push((u[k], finish(&e, sp.jid, tj.elapsed().as_secs_f64(), &extra)));
        }
        out
    };
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results: std::sync::Mutex<Vec<(usize, String)>> = std::sync::Mutex::new(Vec::new());
    let tall = Instant::now();
    std::thread::scope(|sc| {
        for _ in 0..nthreads {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if i >= units.len() { break; }
                let rs = run_unit(&units[i]);
                results.lock().unwrap().extend(rs);
            });
        }
    });
    let mut res = results.into_inner().unwrap();
    res.sort();
    for (_, s) in res { timing.push_str(&s); }
    timing.push_str(&format!("all_jobs_wall_s {} threads {} units {} fork {}\n", tall.elapsed().as_secs_f64(), nthreads, units.len(), fork));
    fs::write(format!("{}/timing.txt", rd_), timing).unwrap();
}
