// icpe-rux v9（2026-11-07，code/icpe/rux_x1fast/v9_qev）：v8_dev14＋二階排隊 qev=1（事件驅動）＋一階 qpost=2（make_v9.py；PREREG 偏離 19）。
//   沒有 qev=1／qpost=2 時每條路徑和 v8_dev14 相同。
// icpe-rux v8_dev14（2026-11-07，code/icpe/rux_x1fast/v8_dev14）：v8＋PREREG 偏離 14 的規則（make_v8_dev14.py；來源 icpe-sim 的 x1fast
//   63b8e328，SIM_CHANGES 22、23）：jobs.txt 可多 `pfx=1`（O-PF 預取的副本過期就丟，Eng::pf_expire）、`nzo=resample`（模型 P 抽到已經
//   過去的時間就重抽，Eng::nz）、`pgr=1`（預取到 GPU 時騰完空間再查候選）。沒有這三欄時和 v8 逐位元相同。
// icpe-rux v8（2026-11-07，code/icpe/rux_x1fast/v8_features）：v7＋第 2–5 項的新功能（只有 Rust；語意見
//   docs/icpe_brief_20261030/next/rux/SEMANTICS_2to5.md；patch_v8.py）：排隊 q=1、W3 線索 GPU 規則 cuep／cueo、W5 w5=、APC apc／belp。
// icpe-rux v7（2026-11-07，code/icpe/rux_x1fast/v7_fast）：v6＋慢層上限的加速（不改語意；patch_v7.py）：
//   慢層 LRU2 用有序集合（候選 2）、慢層 PENS 的成本快取（候選 3）、開上限也走快速路徑＋補記 clean_q（候選 5）。
// icpe-rux v6（2026-11-07，code/icpe/rux_x1fast/v6_slowcap）：v5（code/p3x/x1fast，sha256 3642370a…）＋慢層容量上限
// （x1_sim.SlowOrder＋Engine._sc_*，PREREG §2 第 5 項）。jobs.txt 多一欄 `sc=<frac>,<規則>,<上限塊數>` 才開；沒有這欄時每條路徑和 v5 相同。
// 開上限時照 x1_sim（_fix_fast=False）關掉三個回合結束的快速路徑：每塊都呼叫 fix（fix 會把變乾淨的塊放進 clean_q，少叫一次順序就不同）。
// x1fast：x1_sim.py 的 Engine 全移植（第一階段：17 個 CPU 規則、GPU 層關閉／wb／wt、GPU LRU／BEL／TTLA／TTLB、
// 預取到 CPU／GPU、Cake／只載入、多個時間設定）。還沒有：W4 雜訊、慢層容量上限、STAIR 規劃器。
// 每個函式照 x1_sim.py 同名函式的運算順序寫，目標是輸出和 Python 逐位元相同。只為跑分與對拍，不是正式模擬器。
//
// 用法：x1fast <trace_dir> <run_dir>
//   run_dir/views.txt：每行 `id d cpu_r cpu_w cpu_k slow_r slow_w slow_k mode fidx`（數字用 Python repr）；
//   run_dir/f<fidx>.f64、fc<fidx>.f64：重算時間與前綴和；fc_idle.f64：Pensieve 用（GPU 閒的前綴和）
//   run_dir/jobs.txt：每行 `jobid order act hook pf dead_cut cap G gorder gttl gwrite pft view_id,view_id,...`
//   輸出 run_dir/out/<jobid>/...
//
// icpe-sim（2026-11-06，SIM_CHANGES 20）：本檔＝origin/icpe-next-ana @ 43e7ffc 的 code/icpe/ana_x1fast/v5_noise_b2/src/main.rs（sha256 a5c7c130…）
// 原樣複製，只加 SIM_CHANGES 19 的兩行（turn_end 新請求沒有完整 block → order_remove，取代只做 cpu_set.discard）。
// icpe-ana v5（2026-11-06）：v4 加 B2 抽法（每次問都重抽）。B2 的亂數順序取決於 Python 走訪 cpu_set（CPython 3.12 的 set）的順序，
// 所以 B2 時 ISet 另外照 CPython setobject.c 維護同樣的雜湊表，select（BEL／BELS）與 make_room（O-PF）照它的順序走訪。
// icpe-ana v4（2026-11-06）：在 icpe-sim 0b0ef1e 的 code/p3x/x1fast/src/main.rs 上加 W4 雜訊（B1；CPython random.Random 逐位元重現）。
// 沒有 nz= 欄位時和 0b0ef1e 逐位元相同。以下是 icpe-sim 原本的說明：
// icpe-sim（2026-11-06）：本檔＝origin/icpe-next-ana @ aee1527 的 code/icpe/ana_x1fast/v3_prefix_fork/src/main.rs
// （sha256 237ba2f8…）複製過來。引擎（Chan、View、OList、ISet、Trace、Eng 的全部方法）一行沒改；只改 main() 的輸入輸出：
// 把 parse／build／finish 搬成函式（內容照抄）、加常駐模式 X1FAST_SERVE=1（給 code/p3x/x1_rust.py）、jobs.txt 多欄位就停。
// 正式模擬經 `x1 sweep --engine rust`；驗證見 docs/icpe_brief_20261030/next/SIM_CHANGES.md 第 16 項。
use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeSet, BinaryHeap, VecDeque};
use std::cell::{Cell, RefCell};
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
    // v8 W5：w5＝開；cpu_b＝CPU 層每塊位元組（沒開＝BF）；dc1／dc2＝讀 CPU 的 BF16 位置／INT4 預取位置的塊之後加的殘差（沒開不用）
    w5: bool, cpu_b: i64, dc1: f64, dc2: f64,
}

impl<'a> View<'a> {
    fn m_cpu(&mut self, n: usize) -> usize {
        if n < self.mcpu.len() && self.mcpu[n] >= 0 { return self.mcpu[n] as usize; }
        let l = self.i4 as f64 / self.cpu.r + if self.w5 { self.dc2 } else { self.d };     // v8 W5
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
        let c_b = self.cpu_b as f64 / self.cpu.r;                 // v8 W5：CPU 層每塊位元組（沒開＝BF）
        let (cpu_b, w5, dc1, dc2) = (self.cpu_b, self.w5, self.dc1, self.dc2);
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
                    cf = st + dur; cb += dur; cn += cpu_b; $lt = if w5 { cf + dc1 } else { cf }; nc += 1;
                } else if k == 3 {
                    let st = if $lt > sf { $lt } else { sf };
                    let dur = if swf > st { s4k } else { s4 };
                    sf = st + dur; sb += dur; sn += i4; $lt = sf + d; ns += 1;
                } else if k == 2 {
                    let pd = pf_done[blocks[$idx] as usize];
                    let nw = if $lt > pd { $lt } else { pd };
                    let st = if nw > cf { nw } else { cf };
                    let dur = if cwf > st { c4k } else { c4 };
                    cf = st + dur; cb += dur; cn += i4; $lt = cf + if w5 { dc2 } else { d }; nc += 1;
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

// ------------------------------------------------------------------ CPython random.Random（MT19937）的逐位元重現
#[derive(Clone)]
struct Mt { mt: Vec<u32>, mti: usize }
impl Mt {
    fn init_genrand(s: u32) -> Self {
        let mut mt = vec![0u32; 624];
        mt[0] = s;
        for i in 1..624 { mt[i] = 1812433253u32.wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_add(i as u32); }
        Mt { mt, mti: 624 }
    }
    // random.Random(seed) 的 int 種子：key＝abs(seed) 的 32-bit 字（little-endian）；0 → [0]
    fn from_seed(seed: i64) -> Self {
        let mut n = seed.unsigned_abs();
        let mut key: Vec<u32> = Vec::new();
        while n > 0 { key.push((n & 0xffff_ffff) as u32); n >>= 32; }
        if key.is_empty() { key.push(0); }
        let mut m = Mt::init_genrand(19650218);
        let (mut i, mut j) = (1usize, 0usize);
        let kl = key.len();
        let mut k = if 624 > kl { 624 } else { kl };
        while k > 0 {
            m.mt[i] = (m.mt[i] ^ (m.mt[i - 1] ^ (m.mt[i - 1] >> 30)).wrapping_mul(1664525)).wrapping_add(key[j]).wrapping_add(j as u32);
            i += 1; j += 1;
            if i >= 624 { m.mt[0] = m.mt[623]; i = 1; }
            if j >= kl { j = 0; }
            k -= 1;
        }
        k = 623;
        while k > 0 {
            m.mt[i] = (m.mt[i] ^ (m.mt[i - 1] ^ (m.mt[i - 1] >> 30)).wrapping_mul(1566083941)).wrapping_sub(i as u32);
            i += 1;
            if i >= 624 { m.mt[0] = m.mt[623]; i = 1; }
            k -= 1;
        }
        m.mt[0] = 0x8000_0000;
        m
    }
    fn u32(&mut self) -> u32 {
        const MAG: [u32; 2] = [0, 0x9908_b0df];
        if self.mti >= 624 {
            let mt = &mut self.mt;
            for kk in 0..624 {
                let y = (mt[kk] & 0x8000_0000) | (mt[(kk + 1) % 624] & 0x7fff_ffff);
                mt[kk] = mt[(kk + 397) % 624] ^ (y >> 1) ^ MAG[(y & 1) as usize];
            }
            self.mti = 0;
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11; y ^= (y << 7) & 0x9d2c_5680; y ^= (y << 15) & 0xefc6_0000; y ^= y >> 18;
        y
    }
    fn random(&mut self) -> f64 {
        let a = (self.u32() >> 5) as f64;
        let b = (self.u32() >> 6) as f64;
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }
    fn randbelow(&mut self, n: usize) -> usize {      // _randbelow_with_getrandbits（n < 2^32）
        let k = (usize::BITS - n.leading_zeros()) as u32;
        loop {
            let r = (self.u32() >> (32 - k)) as usize;
            if r < n { return r; }
        }
    }
    fn uniform(&mut self, a: f64, b: f64) -> f64 { a + (b - a) * self.random() }
}

// W4 雜訊（照 x1_sim._noise_init／_nz_sample／_nz；只支援 B1 抽法）
#[derive(Clone)]
struct Noise { model_p: bool, b1: bool, lam: f64, rng: Mt, gaps: Vec<f64>, nzd: Vec<Option<(f64, f64, f64)>>,
               rs: bool, sorted: Vec<f64> }   // rs＝偏離 14 (a)；sorted＝gaps 照值排序（只有 rs 時才建）

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

// CPython 3.12 Objects/setobject.c 的雜湊表（鍵＝session id，hash(int)＝int）：set_add_entry（重用探測路上最後一個 dummy）、
// set_insert_clean、set_table_resize（fill*5 >= mask*3 時 → used>50000 ? used*2 : used*4）、set_discard_entry（留 dummy）、走訪＝表的索引順序。
#[derive(Clone)]
struct PySet { tab: Vec<i64>, mask: usize, fill: usize, used: usize, slot: Vec<i64> }
const PS_EMPTY: i64 = -1;
const PS_DUMMY: i64 = -2;
impl PySet {
    fn new(n: usize) -> Self { PySet { tab: vec![PS_EMPTY; 8], mask: 7, fill: 0, used: 0, slot: vec![-1; n] } }
    // 呼叫者保證 k 不在集合裡（found_active 的情況由 ISet 的 pos 先擋掉）
    fn add(&mut self, k: usize) {
        let mask = self.mask;
        let mut i = k & mask;
        let mut perturb = k;
        let mut free: Option<usize> = None;
        let e_unused;
        'outer: loop {
            let mut probes = if i + 9 <= mask { 9 } else { 0 };
            let mut e = i;
            loop {
                let v = self.tab[e];
                if v == PS_EMPTY { e_unused = e; break 'outer; }
                if v == PS_DUMMY { free = Some(e); }
                e += 1;
                if probes == 0 { break; }
                probes -= 1;
            }
            perturb >>= 5;
            i = (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb)) & mask;
        }
        if let Some(f) = free {
            self.used += 1; self.tab[f] = k as i64; self.slot[k] = f as i64;
            return;
        }
        self.fill += 1; self.used += 1; self.tab[e_unused] = k as i64; self.slot[k] = e_unused as i64;
        if self.fill * 5 < mask * 3 { return; }
        let minused = if self.used > 50000 { self.used * 2 } else { self.used * 4 };
        self.resize(minused);
    }
    fn resize(&mut self, minused: usize) {
        let mut newsize = 8usize;
        while newsize <= minused { newsize <<= 1; }
        if newsize == 8 && self.mask == 7 && self.fill == self.used { return; }
        let old = std::mem::replace(&mut self.tab, vec![PS_EMPTY; newsize]);
        self.mask = newsize - 1;
        self.fill = self.used;
        for &k in &old {
            if k >= 0 { self.insert_clean(k as usize); }
        }
    }
    fn insert_clean(&mut self, k: usize) {
        let mask = self.mask;
        let mut i = k & mask;
        let mut perturb = k;
        loop {
            if self.tab[i] == PS_EMPTY { self.tab[i] = k as i64; self.slot[k] = i as i64; return; }
            if i + 9 <= mask {
                for j in 1..=9 {
                    if self.tab[i + j] == PS_EMPTY { self.tab[i + j] = k as i64; self.slot[k] = (i + j) as i64; return; }
                }
            }
            perturb >>= 5;
            i = (i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb)) & mask;
        }
    }
    fn discard(&mut self, k: usize) {
        let p = self.slot[k];
        if p < 0 { return; }
        self.tab[p as usize] = PS_DUMMY; self.used -= 1; self.slot[k] = -1;
    }
    fn order(&self) -> Vec<usize> { self.tab.iter().filter(|&&v| v >= 0).map(|&v| v as usize).collect() }
}

#[derive(Clone)]
struct ISet { items: Vec<usize>, pos: Vec<i64>, py: Option<PySet> }
impl ISet {
    fn new(n: usize) -> Self { ISet { items: Vec::new(), pos: vec![-1; n], py: None } }
    fn add(&mut self, s: usize) {
        if self.pos[s] < 0 {
            self.pos[s] = self.items.len() as i64; self.items.push(s);
            if let Some(p) = &mut self.py { p.add(s); }
        }
    }
    fn discard(&mut self, s: usize) {
        let p = self.pos[s];
        if p < 0 { return; }
        let last = *self.items.last().unwrap();
        self.items[p as usize] = last; self.pos[last] = p; self.items.pop(); self.pos[s] = -1;
        if let Some(p) = &mut self.py { p.discard(s); }
    }
    // 走訪順序：B2 時＝CPython set 的順序；否則任意（鍵都含 last／x，不影響結果）
    fn order(&self) -> Vec<usize> { match &self.py { Some(p) => p.order(), None => self.items.clone() } }
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
    fn acc(&self, b: u32) -> &[i64] { &self.acc_req[self.acc_off[b as usize]..self.acc_off[b as usize + 1]] }
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

// ------------------------------------------------------------------ 慢層容量上限（v6）＝x1_sim.SlowOrder＋Engine._sc_*
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
    // v7 候選 2：LRU2 的有序集合。元素＝(prev 或 −1, last, 進入順序, s)；進入順序＝mem 的 move_to_end 先後 → 同分時取 mem 裡第一個（＝Python min）
    l2set: BTreeSet<(i64, i64, u64, usize)>, l2key: Vec<(i64, i64, u64)>, l2seq: u64,
    // v7 候選 3：PENS 的成本快取 (成本, 那時的 epoch＋1)；0＝沒有快取
    pc: Vec<Cell<(f64, u64)>>,
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
            SRule::Lru2 => { self.l2_del(s); }
            _ => {}
        }
    }
    // v7：s 離開 LRU2 的有序集合（不在就不做）
    fn l2_del(&mut self, s: usize) {
        let k = self.l2key[s];
        if k.2 != 0 {
            assert!(self.l2set.remove(&(k.0, k.1, k.2, s)), "LRU2 集合裡沒有 {}", s);
            self.l2key[s].2 = 0;
        }
    }
}

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
    noise: Option<RefCell<Noise>>,
    pfx: bool,   // 偏離 14 (i)：只在 pf＝Opf 時為真
    pgr: bool,   // SIM_CHANGES 23
    sc: Option<Box<SlowCap>>,          // v6：慢層容量上限（None＝沒有上限＝v5 的路徑）
    // v8 排隊：qon＝開；qfree＝GPU 空出來的時刻（前一個請求完成）；qprev＝前一個請求的開始時刻；qwait＝每個請求的等待
    qon: bool, qfree: f64, qprev: f64, qwait: Vec<f64>, qstart: Vec<f64>, qsvc: Vec<f64>,   // qstart／qsvc＝開始時刻／服務時間（稽核用）
    // closed-loop（q=2）：qd[s]＝session s 到目前為止累積的等待；請求的到達＝紀錄的時刻＋qd[s]（等待會延後這個 session 之後的請求）
    qc: bool, qd: Vec<f64>, qarr: Vec<f64>,
    // v8 W3 線索：cue[r]＝1 → 這一輪的輸出有工具呼叫（同一 session 的下一個請求是工具觸發）
    cue: &'a [u8],
    // v8 W5：cpu_bf＝CPU 層每塊位元組（沒開＝BF）
    cpu_bf: i64,
    // v8 排隊的一階近似：qpost＝開；qpw／qps＝每個時間設定的等待、開始時刻（稽核用）
    qpost: bool, qpw: Vec<Vec<f64>>, qps: Vec<Vec<f64>>,
    // v8 APC：apc＝GPU 規則是 apc／belp；gn[s]＝s 的 GPU 副本還在 GPU 上的前段塊數；ref_gc[b]＝wb 下「已經寫進 CPU 的乾淨 GPU 副本」的引用數
    //（ref_gpu 裡面的一部分；活的 GPU 引用＝ref_gpu − ref_gc）；n_gblk＝逐出的塊數；apc_whole＝驗證用：一次逐出整個 session（＝GPU LRU／BEL）
    apc: bool, apc_whole: bool, gn: Vec<usize>, ref_gc: Vec<i64>, n_gblk: i64, v8chk: usize,
    // BELP 的捷徑（不改結果；X1FAST_BELP_SLOW=1 關掉＝每塊都全掃）：逐出的塊沒有別人持有時，其他 session 的鍵都不變；
    // 被選的 session 的鍵沒有變小 → 它仍是最大（max_by_key 取最後一個，它後面的都比它小）→ 不用重掃。只在沒有雜訊時用
    belp_fast: bool,
    // v9 二階排隊（qev）：served[r]＝已經開始服務；acc_ptr[b]＝b 的存取清單裡第一個還沒服務的位置（只會往後）；front＝還沒服務的最小序號；
    // cur_r＝目前請求的序號（qev 時 cur＝服務順序的序號 seq）；qev_t＝這一步的開始時刻；tnow＝目前請求的開始時刻；tl[s]＝s 上一個請求的開始時刻
    qev: bool, served: Vec<bool>, acc_ptr: Vec<Cell<u32>>, front: usize, cur_r: usize, seq: i64, qev_t: f64, tnow: f64, tl: Vec<f64>,
    qe_arr: Vec<f64>, qe_start: Vec<f64>, qe_svc: Vec<f64>, qe_wait: Vec<f64>, qe_order: Vec<i64>, qe_audit: usize, qe_naudit: i64,
    // v9 qpost=2：一階的到達不早於上一個完成
    qpost2: bool,
}

impl<'a> Eng<'a> {
    // v9：「下次回來」。沒有 qev＝v8（cur 之後第一次存取）；qev＝還沒服務的存取裡序號最小的（④ 只知道紀錄的順序）
    fn nacc(&self, b: u32) -> i64 {
        if !self.qev { return self.tr.next_access(b, self.cur); }
        let lst = self.tr.acc(b);
        let c = &self.acc_ptr[b as usize];
        let mut i = c.get() as usize;
        while i < lst.len() && self.served[lst[i] as usize] { i += 1; }
        c.set(i as u32);
        if i < lst.len() { lst[i] } else { INF }
    }
    // v9：快取的「下次回來」還有效嗎（沒有 qev＝v8 的 cur < cnx；qev＝cnx 還沒服務：還沒服務的集合只會變小，所以它仍是第一個）
    fn nx_live(&self, cnx: i64) -> bool { if self.qev { !self.served[cnx as usize] } else { self.cur < cnx } }
    fn bl_of(&self, r: i64) -> &'a [u32] { if r < 0 { &[] } else { self.tr.blocks(r as usize) } }
    fn sbl(&self, s: usize) -> &'a [u32] { self.bl_of(self.sb[s]) }

    #[inline]
    fn bump(&mut self, s: usize) { self.gep[self.grp[s] as usize] += 1; }

    // ---------------------------------------------------------- _fix／_fix_g
    // = Engine._fix_sc（沒有上限時＝原本的 fix）
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
        if self.fix_g { n - (self.ref_gpu[b] - self.ref_gc[b]) } else { n }
    }
    // v7 候選 5：快速路徑跳過 fix(b) 時補做它唯一的效果（fix_inner 這時什麼都不改；驗證模式斷言這一點）
    #[inline]
    fn sc_note(&mut self, b: usize) {
        if self.sc.as_ref().unwrap().check_every > 0 {
            assert!(!(self.ref_live[b] == 0 || (self.ref_cpu[b] == 0 && self.fmt[b] == 1)), "sc_note：塊 {} 的 fix 不是空動作", b);
        }
        if self.on_slow[b] && self.sc_holders(b) == 0 { self.sc.as_mut().unwrap().clean_q.push_back(b as u32); }
    }
    fn fix_inner(&mut self, b: usize) -> i64 {
        if self.ref_live[b] == 0 {
            let fm = self.fmt[b];
            if fm != 0 {
                self.cpu_used -= if fm == 1 { self.cpu_bf } else { self.i4 };
                self.fmt[b] = 0;
                let o = self.pf_owner[b];
                if o >= 0 { self.spf[o as usize] -= 1; self.pf_owner[b] = -1; }
            }
            self.on_slow[b] = false;
            return 0;
        }
        if self.ref_cpu[b] == 0 && self.fmt[b] == 1 {
            self.cpu_used -= self.cpu_bf;
            self.fmt[b] = 0;
            let cond = if self.fix_g { !self.on_slow[b] && self.ref_live[b] - (self.ref_gpu[b] - self.ref_gc[b]) > 0 } else { !self.on_slow[b] };
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
    fn nz_sample(nz: &mut Noise) -> (f64, f64, f64) {
        let u = nz.rng.random();
        let g = nz.gaps[nz.rng.randbelow(nz.gaps.len())];
        let lam = nz.lam;
        let eps = nz.rng.uniform(-lam, lam);
        (u, g, eps)
    }
    // = Engine._nz：真的「下次回來」（請求 index）→ 有雜訊的 index（B1）
    fn nz(&self, s: usize, idx: i64) -> i64 {
        let Some(cell) = &self.noise else { return idx };
        let mut nz = cell.borrow_mut();
        if nz.lam == 0.0 { return idx; }
        let d = if nz.b1 {
            match nz.nzd[s] { Some(d) => d, None => { let d = Self::nz_sample(&mut nz); nz.nzd[s] = Some(d); d } }
        } else { Self::nz_sample(&mut nz) };   // B2：每次問都重抽
        let (u, g, eps) = d;
        let tl = &self.tr.t;
        let tq_target: f64;
        if nz.model_p {
            if u >= nz.lam { return idx; }
            if g >= INF as f64 { return INF; }
            let ls = tl[self.last[s] as usize];
            let mut tqt = ls + g;
            if nz.rs {
                // 偏離 14 (a)＝Engine._nz：已經過去 → 從「間隔 > 已經過的時間」那部分均勻重抽（bisect_right＋randrange）；B1 取代存著的
                let tq = if self.cur >= 0 { tl[self.cur as usize] } else { tl[0] };
                if tqt <= tq {
                    let e = tq - ls;
                    let k = nz.sorted.partition_point(|&x| x <= e);
                    let n = nz.sorted.len() - k;
                    let j = k + nz.rng.randbelow(n);
                    let g2 = nz.sorted[j];
                    if nz.b1 { nz.nzd[s] = Some((u, g2, eps)); }
                    if g2 >= INF as f64 { return INF; }
                    tqt = ls + g2;
                }
            }
            tq_target = tqt;
        } else {
            if idx >= tl.len() as i64 || eps == 0.0 { return idx; }
            let tq = if self.cur >= 0 { tl[self.cur as usize] } else { tl[0] };
            tq_target = tq + (tl[idx as usize] - tq) * (1.0 + eps);
        }
        let j = tl.partition_point(|&x| x < tq_target) as i64;     // bisect_left
        let c = self.cur + 1;
        if j > c { j } else { c }
    }
    // next_req(s, ref)：ref 0＝ref_cpu、1＝ref_live（有雜訊時＝_next_req_nz：(nz(真的), e)）
    fn next_req(&self, s: usize, which: u8) -> (i64, usize) {
        let (nx, e) = self.next_req_cached(s, which);
        if self.noise.is_some() { (self.nz(s, nx), e) } else { (nx, e) }
    }
    fn next_req_cached(&self, s: usize, which: u8) -> (i64, usize) {
        let ep = self.gep[self.grp[s] as usize] + 1;
        let c = &self.nx_cache[which as usize][s];
        let (cnx, ce, cep) = c.get();
        if cep == ep && (cnx >= INF || self.nx_live(cnx)) { return (cnx, ce); }
        let r = self.next_req_raw(s, which);
        c.set((r.0, r.1, ep));
        r
    }
    fn next_req_raw(&self, s: usize, which: u8) -> (i64, usize) {
        let bl = self.sbl(s);
        let rf = if which == 0 { &self.ref_cpu } else { &self.ref_live };
        let e = Self::excl_start(bl, rf);
        let b = if e < bl.len() { bl[e] } else { bl[bl.len() - 1] };
        (self.nacc(b), e)
    }
    fn dead(&self, s: usize) -> bool {
        if self.tr.tt(if self.qev { self.cur_r as i64 } else if self.cur >= 0 { self.cur } else { 0 }) > self.dead_cut { return false; }
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
                let size = self.sbl(s).len() as i64 * self.cpu_bf;
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
                let size = self.sbl(s).len() as i64 * self.cpu_bf;
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
                let cps = if e < n { (self.g_cc[n] - self.g_cc[e]) / (((n - e) as i64 * self.cpu_bf) as f64) }
                          else { self.g_c[if n == 0 { self.g_c.len() - 1 } else { n - 1 }] / self.cpu_bf as f64 };
                let h = self.gl + self.nreq[s] as f64 * cps * self.cpu_bf as f64;
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
        let items: Vec<usize> = self.cpu_set.order();
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
                let cur = if self.qev { self.front as i64 - 1 } else { self.cur };   // v9：④ 的「現在」＝還沒服務的最小序號之前
                items.iter().copied().max_by_key(|&x| {
                    let (nx, e) = self.next_req(x, 0);
                    let dist = if nx >= INF { 1e12 } else { (nx - cur) as f64 };
                    let freed = (self.sbl(x).len() - e) as i64 * self.cpu_bf;
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

    // ---------------------------------------------------------- 慢層容量上限（v6）
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
        let l2k = (if self.nreq[s] >= 2 { self.prev[s] } else { -1 }, self.last[s]);
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
            SRule::Lru2 => {
                sc.l2_del(s);
                sc.l2seq += 1;
                let q = sc.l2seq;
                sc.l2key[s] = (l2k.0, l2k.1, q);
                sc.l2set.insert((l2k.0, l2k.1, q, s));
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
                let fst = sc.l2set.iter().next().map(|e| e.3);
                if chk {
                    let (mut best, mut bk) = (usize::MAX, (0i64, 0i64));
                    let mut x = sc.mem.head;
                    while x >= 0 {
                        let u = x as usize;
                        let k = (if self.nreq[u] >= 2 { self.prev[u] } else { -1 }, self.last[u]);
                        if best == usize::MAX || k < bk { best = u; bk = k; } else if k == bk { ties += 1; }
                        x = sc.mem.next[u];
                    }
                    assert_eq!(fst, Some(best), "LRU2：有序集合和全掃選的不同");
                }
                fst
            }
            SRule::Pens => {
                let (cur, fc) = (self.cur, self.pens_fc);
                let (mut best, mut bk) = (usize::MAX, (F(0.0), 0i64));
                let mut x = sc.mem.head;
                while x >= 0 {
                    let u = x as usize;
                    let ep = self.gep[self.grp[u] as usize] + 1;
                    let (cc, cep) = sc.pc[u].get();
                    let cost = if cep == ep && !chk { cc } else {
                        let bl = self.sbl(u);
                        let n = bl.len();
                        let ex = Self::excl_start(bl, &self.ref_live);
                        let c = if ex < n { fc[n] - fc[ex] } else { fc[n] - fc[if n == 0 { fc.len() - 1 } else { n - 1 }] };
                        if cep == ep { assert!(c.to_bits() == cc.to_bits(), "PENS 成本快取過期：session {} 快取 {} 重算 {}", u, cc, c); }
                        sc.pc[u].set((c, ep));
                        c
                    };
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
        if sc.rule == SRule::Lru2 { sc.l2_del(s); }
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
        if sc.rule == SRule::Lru2 {
            assert_eq!(sc.l2set.len(), sc.mem.len, "r={} LRU2 集合的大小", r);
            let mut x = sc.mem.head;
            while x >= 0 {
                let u = x as usize;
                let k = sc.l2key[u];
                assert!(k.2 != 0 && (k.0, k.1) == (if self.nreq[u] >= 2 { self.prev[u] } else { -1 }, self.last[u]),
                        "r={} LRU2 的鍵在慢層期間變了：session {}", r, u);
                x = sc.mem.next[u];
            }
        }
        assert_eq!(sc.n_ties, 0, "r={} 慢層規則選擇時有同分", r);
    }
    #[inline]
    fn sc_check(&self, r: usize) {
        if let Some(sc) = &self.sc {
            if sc.check_every > 0 && (r % sc.check_every == 0 || r + 1 == self.tr.t.len()) { self.sc_audit(r); }
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
        self.res_sum += if self.qev { self.tnow - self.tl[s] } else { self.tr.tt(self.cur) - self.tr.tt(self.last[s]) };
        if self.pf != Pf::No {
            let bl = self.sbl(s);
            let t = self.nacc(bl[bl.len() - 1]);
            let t = if self.noise.is_some() { self.nz(s, t) } else { t };
            self.st_t[s] = t;
            self.pfdone[s] = false;
            if self.pf == Pf::Opf && t < INF { self.pfheap.push(Reverse((t, s))); }
            else if self.pf == Pf::Rpf { self.pfheap.push(Reverse((-self.last[s], s))); }
        }
        if self.sc.is_some() { self.sc_add(s); }            // = _demote_sc
        w
    }
    fn drop_(&mut self, s: usize) -> i64 {
        self.bump(s);
        let st = self.st[s];
        let mut w = 0;
        if st == S_SLOW { if let Some(sc) = self.sc.as_mut() { sc.remove(s, false); } }
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
        if self.sc.is_some() { self.slow_enforce(); }        // = _enforce_sc
        w
    }

    // ---------------------------------------------------------- 預取（CPU）
    fn make_room(&mut self, cand: usize, need: i64, now: f64) -> bool {
        if self.c - self.cpu_used as f64 >= need as f64 { return true; }
        let order: Vec<usize> = if self.pf == Pf::Opf {
            let tc = self.st_t[cand];
            let mut el: Vec<(i64, i64, usize)> = Vec::new();
            for x in self.cpu_set.order() {
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
        if self.sc.is_some() { self.slow_enforce(); }        // = make_room 尾端的 _slow_enforce（py:1580–1581）
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
    fn gbl(&self, s: usize) -> &'a [u32] { if self.gwb && !self.apc { self.sbl(s) } else { self.bl_of(self.gsb[s]) } }
    // v8 APC：GPU 上還在的前段（沒有 APC＝整個副本）
    fn gheld(&self, s: usize) -> &'a [u32] { let b = self.gbl(s); if self.apc { &b[..self.gn[s]] } else { b } }
    fn g_next(&self, x: usize) -> i64 {
        let nx = self.g_next_cached(x);
        if self.noise.is_some() { self.nz(x, nx) } else { nx }
    }
    fn g_next_cached(&self, x: usize) -> i64 {
        let ep = self.gep[self.grp[x] as usize] + 1;
        let c = &self.nx_cache[2][x];
        let (cnx, ce, cep) = c.get();
        if cep == ep && (cnx >= INF || self.nx_live(cnx)) { let _ = ce; return cnx; }
        let r = self.g_next_raw(x);
        c.set((r, 0, ep));
        r
    }
    fn g_next_raw(&self, x: usize) -> i64 {
        let bl = self.gheld(x);
        let (mut lo, mut hi) = (0usize, bl.len());
        while lo < hi {
            let mid = (lo + hi) >> 1;
            if self.ref_gpu[bl[mid] as usize] <= 1 { hi = mid; } else { lo = mid + 1; }
        }
        let b = if lo < bl.len() { bl[lo] } else { bl[bl.len() - 1] };
        self.nacc(b)
    }
    fn g_add(&mut self, s: usize, te: f64) {
        self.glru.move_to_end(s);
        if self.apc { self.gn[s] = self.gbl(s).len(); }                     // v8 APC：回合結束＝整個 session 在 GPU 上
        if self.gorder == 4 {          // v8 CUEP：這一輪有工具呼叫 → 釘到下一個請求（∞）；沒有 → 隨時可逐出（−∞）＝照 LRU
            self.gexp[s] = if self.cue[self.cur as usize] != 0 { f64::INFINITY } else { f64::NEG_INFINITY };
        } else if self.gorder == 2 || self.gorder == 3 {
            self.gexp[s] = te + self.gttl;
            if self.gorder == 2 {
                self.gpver[s] += 1;
                self.gttlheap.push(Reverse((F(self.gexp[s]), self.gpver[s], s)));
            }
        }
    }
    fn g_remove(&mut self, s: usize) { self.glru.remove(s); self.gpver[s] += 1; if self.apc { self.gn[s] = 0; self.gsb[s] = -1; } }
    fn g_select(&self, now: f64) -> usize {
        match self.gorder {
            0 | 5 | 6 => self.glru.head as usize,           // v8：CUEO 平常照 LRU；APC 的 session 順序＝LRU
            1 | 7 => self.glru.items().into_iter().max_by_key(|&x| (self.g_next(x), -self.last[x])).unwrap(),   // v8：BELP 選 session 同 BEL
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
                self.fmt[b] = 1; self.cpu_used += self.cpu_bf; cw += 1;
            } else if fm == 2 {
                self.fmt[b] = 1; self.cpu_used += self.cpu_bf - self.i4; if self.cpu_bf != self.i4 { cw += 1; }   // v8（審查 F2）：W5 下 INT4 預取塊轉成 CPU 塊不用寫
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
    // v8 APC／BELP（SEMANTICS 2-1–2-3）：從 x 的 GPU 副本尾端逐出一塊。wb 下 x 還是「只在 GPU」（S_GPU）時，先把整個 session
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
            }
            if self.has_gpf && !self.gpf_sessions.is_empty() && (self.pf == Pf::Rpf || self.glru.len == 0) {
                let u = if self.pf == Pf::Rpf { self.min_pf_by_last(&self.gpf_sessions) } else { self.max_pf_by_st(&self.gpf_sessions) };
                let bl = self.sbl(u);
                self.g_unprefetch(u, bl, true);
                continue;
            }
            let x = self.g_select(now);
            if self.apc {                                                      // v8 APC／BELP：一次一塊（驗證用 apc_whole：整個 session）
                if self.apc_whole { while self.glru.inn[x] { cw += self.g_evict_blk(x); } } else { cw += self.g_evict_blk(x); }
                continue;
            }
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
                self.views[0].cpu.write(cw * self.cpu_bf, exp.0);
                self.views[0].slow.write(w * i4, exp.0);
            } else {
                self.g_free_wt(x);
            }
        }
    }
    // 偏離 14 (i)＝Engine._pf_expire：預測被讀的請求 st_t ≤ cur、還留著預取副本 → 丟掉全部副本，不再排回（pfdone）
    // 已丟（S_DROP）的 session 跳過（sb 已清空；那種副本來自 GPU 預取騰空間時候選被丟，見 Python 的說明）
    fn pf_expire(&mut self) {
        if self.has_gpf {
            let ex: Vec<usize> = self.gpf_sessions.items.iter().copied().filter(|&x| self.gspf[x] > 0 && self.st_passed(self.st_t[x]) && self.st[x] != S_DROP).collect();
            for s in ex {
                let bl = self.sbl(s);
                self.g_unprefetch(s, bl, false);
                self.pfdone[s] = true;
            }
        } else {
            let ex: Vec<usize> = self.pf_sessions.items.iter().copied().filter(|&x| self.spf[x] > 0 && self.st_passed(self.st_t[x]) && self.st[x] != S_DROP).collect();
            for s in ex {
                for &b in self.sbl(s) {
                    let b = b as usize;
                    if self.pf_owner[b] == s as i64 { self.cpu_used -= self.i4; self.fmt[b] = 0; self.pf_owner[b] = -1; }
                }
                self.spf[s] = 0;
                self.pf_sessions.discard(s);
                self.pfdone[s] = true;
            }
        }
    }
    // v9：預測被讀的請求 t 已經過去？沒有 qev＝t ≤ cur（同 v8_dev14）；qev＝t 已經服務過（cur 是服務順序，不能和序號比）
    fn st_passed(&self, t: i64) -> bool { if self.qev { t < INF && self.served[t as usize] } else { t <= self.cur } }
    fn pf_push(&mut self, s: usize) {
        let bl = self.sbl(s);
        if bl.is_empty() { return; }
        let t = self.nacc(bl[bl.len() - 1]);
        let t = if self.noise.is_some() { self.nz(s, t) } else { t };
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
            self.views[0].cpu.write(cw * self.cpu_bf, now);
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
            // SIM_CHANGES 23（pgr=1）：騰空間時候選被丟（O-ALL 的 O-NOW）→ 不預取（Python 的說明）
            if self.pgr && !(self.st[s] == S_CPU || self.st[s] == S_SLOW) { continue; }
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
        let arr = if self.qc { tr.t[r] + self.qd[tr.sess[r] as usize] } else { tr.t[r] };   // v8 排隊：closed-loop 的到達含這個 session 累積的等待
        let t = if self.qon && self.qfree > arr { self.qfree } else { arr };   // v8 排隊：開始＝max(到達, 前一個完成)
        let t = if self.qev { self.qev_t } else { t };                         // v9 qev：開始時刻由 run_qev 給
        if self.pfx { self.pf_expire(); }     // 偏離 14 (i)；沒開＝原本的路徑
        if self.g > 0 && self.gorder == 2 { self.g_expire(t); }
        if self.pf != Pf::No {
            let tf = if self.qev { if self.seq > 0 { self.qprev } else { t } } else if r > 0 { if self.qon { self.qprev } else { tr.t[r - 1] } } else { t };
            if self.has_gpf { self.advance_prefetch_gpu(tf, t); } else { self.advance_prefetch(tf, t); }
        }
        if self.qev {                                   // v9：cur＝服務順序的序號；r 標成已服務（之後的「下次回來」不再算它）
            self.cur = self.seq; self.cur_r = r; self.tnow = t; self.qprev = t;
            self.served[r] = true;
            while self.front < self.served.len() && self.served[self.front] { self.front += 1; }
            self.seq += 1;
        } else { self.cur = r as i64; }
        if self.qon { self.qprev = t; }
        if let Some(cell) = &self.noise { let mut nz = cell.borrow_mut(); if nz.b1 { let d = Self::nz_sample(&mut nz); nz.nzd[tr.sess[r] as usize] = Some(d); } }
        if self.sc.is_some() {                                    // 回來了：離開慢層（= step 的 sorder.remove(hit=True)）
            let s0 = tr.sess[r] as usize;
            if self.st[s0] == S_SLOW { self.sc.as_mut().unwrap().remove(s0, true); }
        }
        let blocks = tr.blocks(r);
        if self.g > 0 { self.step_gpu(r, t, blocks, lay, tend); self.sc_check(r); self.v8_check(self.cur as usize); return; }
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
        if self.qon { self.q_note(r, t, tend[0]); }
        // ---- 回合結束：全部 BF16 進 CPU
        let s = tr.sess[r] as usize;
        self.bump(s);
        let old = self.sbl(s);
        let old_state = self.st[s];
        let mut cw = 0i64;
        let mut k0 = 0usize;
        let fast = true;                        // v7 候選 5：開上限也走快速路徑（v6＝self.sc.is_none()）
        if fast && old_state == S_CPU && !old.is_empty() && old.len() <= blocks.len() && blocks[old.len() - 1] == old[old.len() - 1]
            && &blocks[..old.len()] == old { k0 = old.len(); }
        for &b in &blocks[k0..] {
            let b = b as usize;
            self.ref_live[b] += 1;
            self.ref_cpu[b] += 1;
            let fm = self.fmt[b];
            if fm == 0 {
                self.fmt[b] = 1; self.cpu_used += self.cpu_bf; cw += 1;
            } else if fm == 2 {
                self.fmt[b] = 1; self.cpu_used += self.cpu_bf - self.i4; if self.cpu_bf != self.i4 { cw += 1; }   // v8（審查 F2）：W5 下 INT4 預取塊轉成 CPU 塊不用寫
                let o = self.pf_owner[b];
                if o >= 0 { self.spf[o as usize] -= 1; self.pf_owner[b] = -1; }
            }
        }
        let mut w = 0i64;
        if k0 > 0 && self.sc.is_some() { for &b in old { self.sc_note(b as usize); } }    // v7 候選 5：跳過的前綴
        if (old_state == S_CPU || old_state == S_SLOW) && k0 == 0 {
            let isc = old_state == S_CPU;
            if fast {
                for &b in old {
                    let b = b as usize;
                    self.ref_live[b] -= 1;
                    if isc { self.ref_cpu[b] -= 1; }
                    if self.ref_live[b] == 0 || (self.ref_cpu[b] == 0 && self.fmt[b] == 1) { w += self.fix(b); }
                    else if self.sc.is_some() { self.sc_note(b); }                          // v7 候選 5
                }
            } else {
                for &b in old {
                    let b = b as usize;
                    self.ref_live[b] -= 1;
                    if isc { self.ref_cpu[b] -= 1; }
                    w += self.fix(b);
                }
            }
        }
        if self.spf[s] <= 0 { self.pf_sessions.discard(s); }
        self.sb[s] = r as i64;
        self.st[s] = S_CPU;
        self.prev[s] = self.last[s];
        self.last[s] = self.cur;
        if self.qev { self.tl[s] = self.tnow; }
        self.nreq[s] += 1;
        if !blocks.is_empty() {
            self.order_add(s);
            if (self.hook == Hook::H2a && self.nreq[s] < 2) || (self.hook == Hook::Odrop && self.dead(s)) {
                self.order_remove(s);
                self.cpu_set.add(s);
                w += self.drop_(s);
            }
        } else {
            self.order_remove(s);          // 沒有完整 block：離開 CPU 的順序結構，不只 cpu_set（否則 LRU／ARC／S3FIFO 留下過期項目；SIM_CHANGES 19，使用者 11/6 選 A）
        }
        w += self.enforce();
        self.w_slow_blocks += w;
        let (i4, cpu_bf) = (self.i4, self.cpu_bf);
        for (v, &te) in self.views.iter_mut().zip(tend.iter()) {
            v.cpu.write(cw * cpu_bf, te);
            v.slow.write(w * i4, te);
        }
        self.sc_check(r);
    }

    // v9 二階排隊（qev）：closed-loop FCFS、單一 GPU。到達＝max(紀錄時刻＋上一個開始的延後, 上一個完成)（同 qpost=2）；
    // 差別是服務時間在開始時刻用當下的快取狀態算，快取狀態照服務順序演進（step 照服務順序呼叫）。
    fn run_qev(&mut self, lay: &mut Vec<u8>, tend: &mut Vec<f64>) {
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
        let mut heap: BinaryHeap<Reverse<(F, usize)>> = BinaryHeap::new();
        for s in 0..ns { if first[s] != usize::MAX { heap.push(Reverse((F(tr.t[first[s]]), first[s]))); } }
        let mut free = f64::NEG_INFINITY;
        let mut done = 0usize;
        while let Some(Reverse((a, r))) = heap.pop() {
            let a = a.0;
            let st = if free > a { free } else { a };
            self.qev_t = st;
            self.step(r, lay, tend);
            let svc = self.views[0].ttft[r];
            let w = st - a;
            assert!(w >= 0.0 && svc >= 0.0);
            self.qe_arr[r] = a; self.qe_start[r] = st; self.qe_svc[r] = svc; self.qe_wait[r] = w; self.qe_order[done] = r as i64;
            self.views[0].ttft[r] = svc + w;
            free = st + svc;
            let late = st - tr.t[r];
            let nr = nxt[r];
            if nr != usize::MAX {
                let an = tr.t[nr] + late;
                let an = if free > an { free } else { an };          // 不早於同 session 上一個請求完成（free＝這一個的完成）
                heap.push(Reverse((F(an), nr)));
            }
            done += 1;
            if self.qe_audit > 0 && done % self.qe_audit == 0 { self.qev_audit(); }
        }
        assert_eq!(done, n, "qev：有請求沒排到");
        assert!(self.front == n && self.served.iter().all(|&x| x));
        if self.qe_audit > 0 { self.qev_audit(); }
    }
    // v9 稽核（X1FAST_QEV_AUDIT=k：每 k 個請求一次；只給驗證用，慢）：
    //   front＝還沒服務的最小序號；每個塊的 nacc＝暴力找的第一個還沒服務的存取；CPU／GPU 上每個 session 的快取「下次回來」＝重算
    fn qev_audit(&mut self) {
        let n = self.served.len();
        let mut f = 0usize;
        while f < n && self.served[f] { f += 1; }
        assert_eq!(f, self.front, "qev 稽核：front");
        for b in 0..self.tr.nb {
            let brute = self.tr.acc(b as u32).iter().copied().find(|&x| !self.served[x as usize]).unwrap_or(INF);
            assert_eq!(self.nacc(b as u32), brute, "qev 稽核：塊 {} 的下次回來", b);
        }
        for s in self.cpu_set.items.clone() {
            if self.sbl(s).is_empty() { continue; }
            for w in 0..2u8 { assert_eq!(self.next_req_cached(s, w), self.next_req_raw(s, w), "qev 稽核：session {} 的下次回來（{}）", s, w); }
        }
        if self.g > 0 {
            for x in self.glru.items() {
                if self.gheld(x).is_empty() { continue; }
                assert_eq!(self.g_next_cached(x), self.g_next_raw(x), "qev 稽核：GPU session {} 的下次回來", x);
            }
        }
        self.qe_naudit += 1;
    }

    // v8 排隊的一階近似（qpost）：模擬跑完之後呼叫一次
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
                if nr != usize::MAX {
                    let an = tr.t[nr] + late;
                    let an = if self.qpost2 && free > an { free } else { an };      // v9 qpost=2：不早於上一個完成
                    heap.push(Reverse((F(an), nr)));
                }
                done += 1;
            }
            assert_eq!(done, n, "qpost：有請求沒排到");
            for r in 0..n { v.ttft[r] += wait[r]; }
            self.qpw.push(wait);
            self.qps.push(start);
        }
    }

    // v8 排隊：記等待、TTFT 加上等待、GPU 忙到這個請求完成（＝開始＋服務時間）
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
        if self.qon { self.q_note(r, t, tend[0]); }
        if self.gwb { self.turn_end_wb(r, blocks, tend); } else { self.turn_end_wt(r, blocks, tend); }
    }

    fn turn_end_wb(&mut self, r: usize, blocks: &'a [u32], tend: &[f64]) {
        let s = self.tr.sess[r] as usize;
        self.bump(s);
        let old = self.sbl(s);
        let old_state = self.st[s];
        let mut k0 = 0usize;
        let fast = true;                        // v7 候選 5：開上限也走快速路徑（v6＝self.sc.is_none()）
        if fast && old_state == S_GPU && !self.has_gpf && !old.is_empty() && old.len() <= blocks.len()
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
        if self.apc && old_state != S_GPU && self.glru.inn[s] {           // v8 APC（wb）：放掉這個 session 的乾淨 GPU 副本
            let og: &'a [u32] = self.gheld(s);
            for &b in og {
                let b = b as usize;
                self.ref_gpu[b] -= 1; self.ref_gc[b] -= 1;
                if self.ref_gpu[b] == 0 { self.gpu_used -= BF; }
            }
        }
        let mut w = 0i64;
        if old_state == S_CPU { self.order_remove(s); }
        if k0 > 0 && self.sc.is_some() { for &b in old { self.sc_note(b as usize); } }    // v7 候選 5：跳過的前綴
        if (old_state == S_CPU || old_state == S_SLOW || old_state == S_GPU) && k0 == 0 && !fast {
            let (isc, isg) = (old_state == S_CPU, old_state == S_GPU);
            for &b in old {
                let b = b as usize;
                self.ref_live[b] -= 1;
                if isc { self.ref_cpu[b] -= 1; }
            }
            if isg { self.g_release(old); }
            for &b in old { w += self.fix(b as usize); }
        } else if (old_state == S_CPU || old_state == S_SLOW || old_state == S_GPU) && k0 == 0 {
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
                } else if self.sc.is_some() { self.sc_note(b); }                              // v7 候選 5
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
        if self.apc { self.gsb[s] = r as i64; }
        self.prev[s] = self.last[s];
        self.last[s] = self.cur;
        if self.qev { self.tl[s] = self.tnow; }
        self.nreq[s] += 1;
        let mut cw0 = 0i64;
        if !blocks.is_empty() {
            self.g_add(s, tend[0]);
            if (self.hook == Hook::H2a && self.nreq[s] < 2) || (self.hook == Hook::Odrop && self.dead(s)) { w += self.drop_(s); }
            else if self.gorder == 5 && self.cue[r] != 0 { cw0 = self.g_evict_wb(s); }     // v8 CUEO：有工具呼叫 → 立刻寫進 CPU
        } else {
            self.g_remove(s);
        }
        let cw = cw0 + self.g_enforce(tend[0]);
        w += self.enforce();
        self.w_slow_blocks += w;
        let (i4, cpu_bf) = (self.i4, self.cpu_bf);
        for (v, &te) in self.views.iter_mut().zip(tend.iter()) {
            v.cpu.write(cw * cpu_bf, te);
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
        let fast = true;                        // v7 候選 5：開上限也走快速路徑（v6＝self.sc.is_none()）
        if fast && old_state == S_CPU && !old.is_empty() && old.len() <= blocks.len() && blocks[old.len() - 1] == old[old.len() - 1]
            && &blocks[..old.len()] == old { k0 = old.len(); }
        for &b in &blocks[k0..] {
            let b = b as usize;
            self.ref_live[b] += 1;
            self.ref_cpu[b] += 1;
            let fm = self.fmt[b];
            if fm == 0 {
                self.fmt[b] = 1; self.cpu_used += self.cpu_bf; cw += 1;
            } else if fm == 2 {
                self.fmt[b] = 1; self.cpu_used += self.cpu_bf - self.i4; if self.cpu_bf != self.i4 { cw += 1; }   // v8（審查 F2）：W5 下 INT4 預取塊轉成 CPU 塊不用寫
                let o = self.pf_owner[b];
                if o >= 0 { self.spf[o as usize] -= 1; self.pf_owner[b] = -1; }
            }
        }
        let mut w = 0i64;
        if k0 > 0 && self.sc.is_some() { for &b in old { self.sc_note(b as usize); } }    // v7 候選 5：跳過的前綴
        if (old_state == S_CPU || old_state == S_SLOW) && k0 == 0 {
            let isc = old_state == S_CPU;
            if fast {
                for &b in old {
                    let b = b as usize;
                    self.ref_live[b] -= 1;
                    if isc { self.ref_cpu[b] -= 1; }
                    if self.ref_live[b] == 0 || (self.ref_cpu[b] == 0 && self.fmt[b] == 1) { w += self.fix(b); }
                    else if self.sc.is_some() { self.sc_note(b); }                          // v7 候選 5
                }
            } else {
                for &b in old {
                    let b = b as usize;
                    self.ref_live[b] -= 1;
                    if isc { self.ref_cpu[b] -= 1; }
                    w += self.fix(b);
                }
            }
        }
        if self.spf[s] <= 0 { self.pf_sessions.discard(s); }
        self.sb[s] = r as i64;
        self.st[s] = S_CPU;
        self.prev[s] = self.last[s];
        self.last[s] = self.cur;
        if self.qev { self.tl[s] = self.tnow; }
        self.nreq[s] += 1;
        if !blocks.is_empty() {
            self.order_add(s);
            if (self.hook == Hook::H2a && self.nreq[s] < 2) || (self.hook == Hook::Odrop && self.dead(s)) {
                self.order_remove(s);
                self.cpu_set.add(s);
                w += self.drop_(s);
            }
        } else {
            self.order_remove(s);          // 沒有完整 block：離開 CPU 的順序結構，不只 cpu_set（否則 LRU／ARC／S3FIFO 留下過期項目；SIM_CHANGES 19，使用者 11/6 選 A）
        }
        // ---- GPU 那份
        let gin = self.glru.inn[s];
        let oldg: Option<&'a [u32]> = if gin { Some(self.gheld(s)) } else { None };      // v8 APC：只有前段還在
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
            if self.gorder == 5 && self.cue[r] != 0 { self.g_free_wt(s); }                  // v8 CUEO（wt）：放掉 GPU 副本
        } else {
            self.gsb[s] = -1;
            self.g_remove(s);
        }
        self.g_enforce(tend[0]);
        w += self.enforce();
        self.w_slow_blocks += w;
        let (i4, cpu_bf) = (self.i4, self.cpu_bf);
        for (v, &te) in self.views.iter_mut().zip(tend.iter()) {
            v.cpu.write(cw * cpu_bf, te);
            v.slow.write(w * i4, te);
        }
    }
}

fn parse_f(s: &str) -> f64 { s.parse::<f64>().unwrap_or_else(|_| panic!("float {}", s)) }

// ==================================================================== 以下是輸入輸出（icpe-sim 2026-11-06 重排；引擎在上面，一行沒動）
// 原本 main() 裡的 parse／build／finish 三個閉包搬成函式（內容照抄），讓「批次模式」（原本的用法）和「常駐模式」共用。
// 常駐模式（X1FAST_SERVE=1；icpe-sim 的 x1_rust.py 用）：紀錄只讀一次；stdin 每行一個 job 目錄（裡面有 views.txt、f*／fc*、
// fc_idle.f64、i4.txt、只有一行的 jobs.txt），X1FAST_THREADS 個執行緒各取一個跑，輸出寫進 <job 目錄>/out/<jobid>/ 與
// <job 目錄>/timing.txt，跑完在 stdout 印 `done <job 目錄> <秒>`。stdin 關掉 → 跑完手上的就結束。

// 一個時間設定群組（＝原本 main() 讀 run_dir 的那段）
struct RunData { vspecs: Vec<ViewSpec>, fs_: Vec<Vec<f64>>, fcs: Vec<Vec<f64>>, fc_idle: Vec<f64>, i4: i64,
                 cue: Option<Vec<u8>>, w5d: Option<Vec<f64>> }

fn load_run_data(rd_: &str) -> RunData {
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
    let cue = fs::read(format!("{}/cue.u8", rd_)).ok();
    let w5d = if std::path::Path::new(&format!("{}/w5.f64", rd_)).exists() { Some(rd_f64(&format!("{}/w5.f64", rd_))) } else { None };
    RunData { vspecs, fs_, fcs, fc_idle, i4, cue, w5d }
}

// ---- 一個 job 的規格
struct Spec<'b> { jid: &'b str, order: Ord_, act: Act, hook: Hook, pf: Pf, dead_cut: f64, c: f64, g: i64, gorder: u8, gttl: f64,
                  gwb: bool, pf_gpu: bool, vids: Vec<usize>, nz: Option<(bool, f64, bool, i64)>, sc: Option<(SRule, i64)>,
                  q: bool, qc: bool, cue: bool, w5: Option<i64>, qpost: bool, pfx: bool, nzo_rs: bool, pgr: bool, qev: bool, qpost2: bool }

fn parse_spec(line: &str) -> Spec<'_> {
    let x: Vec<&str> = line.split_whitespace().collect();
    let order = match x[1] { "lru" => Ord_::Lru, "pred" => Ord_::Pred, "lru2" => Ord_::Lru2, "arc" => Ord_::Arc, "s3" => Ord_::S3,
                             "gdsf" => Ord_::Gdsf, "pens" => Ord_::Pens, "bel" => Ord_::Bel, "bels" => Ord_::Bels, "freq" => Ord_::Freq,
                             o => panic!("order {}", o) };
    let act = match x[2] { "demote" => Act::Demote, "none" => Act::None_, "2hw" => Act::H2w, "onow" => Act::Onow, o => panic!("act {}", o) };
    let hook = match x[3] { "-" => Hook::No, "2ha" => Hook::H2a, "odrop" => Hook::Odrop, o => panic!("hook {}", o) };
    let pf = match x[4] { "-" => Pf::No, "opf" => Pf::Opf, "rpf" => Pf::Rpf, o => panic!("pf {}", o) };
    let gorder: u8 = match x[8] { "-" => 0, "lru" => 0, "bel" => 1, "ttla" => 2, "ttlb" => 3,
                                  "cuep" => 4, "cueo" => 5, "apc" => 6, "belp" => 7, o => panic!("gorder {}", o) };   // v8：4–7
    // icpe-ana v4：13 欄，或 13 欄＋`nz=P|M,<lam>,B1,<seed>`（W4 雜訊，只支援 B1）；其他多出來的欄位（慢層上限等）→ 停下來
    let mut nz = None;
    let mut sc: Option<(SRule, i64)> = None;
    let (mut q, mut qc, mut cue, mut w5, mut qpost, mut qev, mut qpost2) = (false, false, false, None, false, false, false);
    let (mut pfx, mut nzo_rs, mut seen_pfx, mut seen_nzo) = (false, false, false, false);   // 偏離 14
    let mut pgr = false;                                                                    // SIM_CHANGES 23
    for extra in &x[13..] {
        if *extra == "q=1" || *extra == "q=2" { assert!(!q); q = true; qc = *extra == "q=2"; continue; }   // v8 排隊：1＝open-loop、2＝closed-loop
        if *extra == "cue=1" { assert!(!cue); cue = true; continue; }                 // v8 W3 線索：<job 目錄>/cue.u8
        if *extra == "qpost=1" { assert!(!qpost); qpost = true; continue; }           // v8 排隊（一階近似，事後算；SEMANTICS §2b）
        if *extra == "qev=1" { assert!(!qev); qev = true; continue; }                 // v9 二階排隊（事件驅動；偏離 19）
        if *extra == "qpost=2" { assert!(!qpost); qpost = true; qpost2 = true; continue; }   // v9 一階、到達不早於上一個完成（偏離 19 的對照）
        if let Some(v) = extra.strip_prefix("w5=") {                                  // v8 W5：CPU 每塊位元組
            assert!(w5.is_none()); let b: i64 = v.parse().unwrap(); assert!(b > 0); w5 = Some(b); continue;
        }
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
        }
        if let Some(v) = extra.strip_prefix("pgr=") {          // SIM_CHANGES 23
            assert!(v == "1" && !pgr, "pgr={}：{}", v, line);
            pgr = true; continue;
        }
        if let Some(v) = extra.strip_prefix("pfx=") {          // 偏離 14 (i)
            assert!(v == "1" && !seen_pfx, "pfx={}：{}", v, line);
            pfx = true; seen_pfx = true; continue;
        }
        if let Some(v) = extra.strip_prefix("nzo=") {          // 偏離 14 (a)
            assert!((v == "resample" || v == "clamp") && !seen_nzo, "nzo={}：{}", v, line);
            nzo_rs = v == "resample"; seen_nzo = true; continue;
        }
        if let Some(v) = extra.strip_prefix("nz=") {
            let f: Vec<&str> = v.split(',').collect();
            assert!(f.len() == 4 && (f[0] == "P" || f[0] == "M"), "nz={}", v);
            assert!(f[2] == "B1" || f[2] == "B2", "抽法只有 B1／B2：nz={}", v);
            assert!(nz.is_none(), "nz 欄位重複：{}", line);
            nz = Some((f[0] == "P", parse_f(f[1]), f[2] == "B1", f[3].parse::<i64>().unwrap()));
        } else { panic!("jobs.txt 有不支援的欄位 {}：{}", extra, line); }
    }
    let sp = Spec { jid: x[0], order, act, hook, pf, dead_cut: parse_f(x[5]), c: parse_f(x[6]), g: x[7].parse().unwrap(), gorder,
                    gttl: if x[9] == "-" { 0.0 } else { parse_f(x[9]) }, gwb: x[10] != "wt", pf_gpu: x[11] == "gpu",
                    vids: x[12].split(',').map(|v| v.parse().unwrap()).collect(), nz, sc, q, qc, cue, w5, qpost, pfx, nzo_rs, pgr, qev, qpost2 };
    if sp.qpost { assert!(!sp.q, "qpost 和 q 不能同時開"); }
    if sp.qev {                                                         // v9：二階排隊只接偏離 19 用到的路徑
        assert!(!sp.q && !sp.qpost, "qev 和 q／qpost 不能同時開");
        assert!(sp.vids.len() == 1, "qev：一個 job 只能 1 個時間設定（服務順序取決於時間設定）");
        // 雜訊（nz 用 tl[cur]、cur＋1 當請求序號）和線索（cue[cur]）把 cur 當請求序號；qev 時 cur＝服務順序 → 一定要關（審查第 3 點）
        assert!(sp.nz.is_none() && sp.sc.is_none() && !sp.cue && sp.w5.is_none(), "qev 不支援雜訊／慢層上限／線索／W5");
    }
    if sp.pf != Pf::No { assert!(sp.vids.len() == 1); }
    if sp.g > 0 && (sp.gorder == 2 || sp.gorder == 3) { assert!(sp.vids.len() == 1); }     // v8：4–7 不依時間
    if sp.q { assert!(sp.vids.len() == 1, "排隊：一個 job 只能 1 個時間設定"); }
    if sp.gorder >= 4 { assert!(sp.g > 0, "GPU 規則 {} 要開 GPU 層", x[8]); assert!(!(sp.pf_gpu && sp.pf != Pf::No), "GPU 規則 {} 不支援預取進 GPU", x[8]); }
    if sp.gorder == 4 || sp.gorder == 5 { assert!(sp.cue, "cuep／cueo 要 cue=1"); }
    sp
}

#[allow(clippy::too_many_arguments)]
fn new_eng<'a>(tr: &'a Trace, grp: &'a [u32], sstart: &[f64], rdat: &'a RunData, sp: &Spec, order: Ord_, act: Act, multi: bool,
               need_gdsf: bool) -> Eng<'a> {
    let (nreq, nb, ns, i4) = (tr.t.len(), tr.nb, tr.ns, rdat.i4);
    let w5dv: Option<&Vec<f64>> = if sp.w5.is_some() {
        let w = rdat.w5d.as_ref().expect("w5= 但沒有 w5.f64");
        assert_eq!(w.len(), 2 * sp.vids.len(), "w5.f64：每個時間設定 2 個值");
        assert!(!(sp.pf_gpu && sp.pf != Pf::No), "W5 不支援預取進 GPU");
        Some(w)
    } else { None };
    let views: Vec<View> = sp.vids.iter().enumerate().map(|(j, &k)| {
        let v = &rdat.vspecs[k];
        View { f: &rdat.fs_[v.fidx], fc: &rdat.fcs[v.fidx], i4, d: v.d, cake: v.cake,
               cpu: Chan::new(v.cpu.0, v.cpu.1, v.cpu.2), slow: Chan::new(v.slow.0, v.slow.1, v.slow.2), pf_bytes: 0,
               mcpu: Vec::new(), ttft: vec![0.0; nreq], rest: vec![0.0; nreq], nc: vec![0; nreq], ns: vec![0; nreq],
               nr: vec![0; nreq], ng: vec![0; nreq],
               w5: sp.w5.is_some(), cpu_b: sp.w5.unwrap_or(BF),
               dc1: w5dv.map(|w| w[2 * j]).unwrap_or(0.0), dc2: w5dv.map(|w| w[2 * j + 1]).unwrap_or(0.0) }
    }).collect();
    let (mut g_c, mut g_cc) = (Vec::new(), Vec::new());
    if order == Ord_::Gdsf || need_gdsf {
        let v = &views[0];
        let l_cpu = if v.w5 { v.cpu_b as f64 / v.cpu.r + v.dc1 } else { BF as f64 / v.cpu.r };   // v8 W5
        let l_sl = i4 as f64 / v.slow.r + v.d;
        g_c = v.f.iter().map(|&fi| f64::max(f64::min(fi, l_sl) - l_cpu, 0.0)).collect();
        g_cc.push(0.0);
        for &xx in &g_c { let lst = *g_cc.last().unwrap(); g_cc.push(lst + xx); }
    }
    let (g, pf) = (sp.g, sp.pf);
    let has_gpf = g > 0 && sp.pf_gpu && pf != Pf::No;
    if sp.q || sp.cue || sp.w5.is_some() || sp.gorder >= 4 || sp.qpost || sp.qev { assert!(!multi, "v8 的新功能不支援共用前段（multi）"); }
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
            l2set: BTreeSet::new(), l2key: vec![(0, 0, 0); ns], l2seq: 0,
            pc: (0..ns).map(|_| Cell::new((0.0, 0))).collect(),
        })
    });
    // W4 雜訊（照 x1_sim._noise_init：gaps[r]＝同一個 session 下一次請求的時間差，沒有下一次＝INF）
    let noise = sp.nz.map(|(model_p, lam, b1, seed)| {
        assert!((0.0..1.0).contains(&lam));
        let mut gaps = vec![0.0f64; nreq];
        let mut nxt: Vec<i64> = vec![-1; ns];
        for r in (0..nreq).rev() {
            let s = tr.sess[r] as usize;
            gaps[r] = if nxt[s] >= 0 { tr.t[nxt[s] as usize] - tr.t[r] } else { INF as f64 };
            nxt[s] = r as i64;
        }
        let sorted = if sp.nzo_rs { let mut v = gaps.clone(); v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v } else { Vec::new() };
        RefCell::new(Noise { model_p, b1, lam, rng: Mt::from_seed(seed), gaps, nzd: vec![None; ns], rs: sp.nzo_rs, sorted })
    });
    Eng {
        tr, views, order, act, hook: sp.hook, pf, dead_cut: sp.dead_cut, c: sp.c, i4,
        ref_cpu: vec![0; nb], ref_live: vec![0; nb], on_slow: vec![false; nb], fmt: vec![0; nb], pf_done: vec![0.0; nb], pf_owner: vec![-1; nb],
        sb: vec![-1; ns], st: vec![S_NONE; ns], last: vec![-1; ns], prev: vec![-1; ns], nreq: vec![0; ns], spf: vec![0; ns],
        st_t: vec![INF; ns], pfdone: vec![false; ns],
        cpu_used: 0, cur: -1, cpu_set: { let mut c = ISet::new(ns); if matches!(sp.nz, Some((_, _, false, _))) || (sp.nz.is_some() && sp.nzo_rs) { c.py = Some(PySet::new(ns)); } c }, lru: OList::new(ns),
        n_demote: 0, n_drop: 0, n_unpf: 0, res_sum: 0.0, w_slow_blocks: 0,
        t1: OList::new(ns), t2: OList::new(ns), b1: OList::new(ns), b2: OList::new(ns), t1b: 0, t2b: 0, b1b: 0, b2b: 0, arc_p: 0.0,
        s3s: OList::new(ns), s3m: OList::new(ns), s3g: OList::new(ns), s3f: vec![-1; ns], s3sb: 0, s3mb: 0, s3size: vec![0; ns],
        gver: vec![0; ns], gheap: BinaryHeap::new(), gl: 0.0, g_c, g_cc, pens_fc: &rdat.fc_idle,
        pfheap: BinaryHeap::new(), pf_sessions: ISet::new(ns),
        g, gorder: sp.gorder, gttl: sp.gttl, gwb: sp.gwb, fix_g: g > 0 && sp.gwb,
        ref_gpu: vec![0; if g > 0 { nb } else { 0 }], gpu_used: 0, gsb: vec![-1; ns], glru: OList::new(ns), gexp: vec![0.0; ns],
        gpver: vec![0; ns], gttlheap: BinaryHeap::new(), n_gevict: 0, n_gexpire: 0, g_wb_blocks: 0, sstart: sstart.to_vec(),
        has_gpf, gpf: vec![-1; if has_gpf { nb } else { 0 }], gpf_done: vec![0.0; if has_gpf { nb } else { 0 }],
        gspf: vec![0; ns], gpf_sessions: ISet::new(ns),
        grp, gep: vec![0; ns], multi, diverged: false, noise, sc, pfx: sp.pfx && pf == Pf::Opf, pgr: sp.pgr,
        qon: sp.q, qfree: f64::NEG_INFINITY, qprev: f64::NAN, qwait: if sp.q { vec![0.0; nreq] } else { Vec::new() },
        qstart: if sp.q { vec![0.0; nreq] } else { Vec::new() }, qsvc: if sp.q { vec![0.0; nreq] } else { Vec::new() },
        qc: sp.qc, qd: if sp.qc { vec![0.0; ns] } else { Vec::new() }, qarr: if sp.q { vec![0.0; nreq] } else { Vec::new() },
        cue: if sp.cue { let c = rdat.cue.as_deref().expect("cue=1 但沒有 cue.u8"); assert_eq!(c.len(), nreq, "cue.u8 的長度"); c } else { &[] },
        cpu_bf: sp.w5.unwrap_or(BF),
        qpost: sp.qpost, qpw: Vec::new(), qps: Vec::new(),
        apc: g > 0 && sp.gorder >= 6, apc_whole: env::var("X1FAST_APC_WHOLE").map(|s| s == "1").unwrap_or(false),
        gn: vec![0; ns], ref_gc: vec![0; if g > 0 { nb } else { 0 }], n_gblk: 0,
        v8chk: env::var("X1FAST_V8_CHECK").ok().and_then(|s| s.parse().ok()).unwrap_or(0),
        belp_fast: g > 0 && sp.gorder == 7 && sp.nz.is_none() && env::var("X1FAST_BELP_SLOW").map(|s| s != "1").unwrap_or(true),
        nx_cache: [(0..ns).map(|_| Cell::new((0, 0, 0))).collect(), (0..ns).map(|_| Cell::new((0, 0, 0))).collect(),
                   (0..ns).map(|_| Cell::new((0, 0, 0))).collect()],
        qev: sp.qev, served: if sp.qev { vec![false; nreq] } else { Vec::new() },
        acc_ptr: if sp.qev { (0..nb).map(|_| Cell::new(0u32)).collect() } else { Vec::new() },
        front: 0, cur_r: 0, seq: 0, qev_t: f64::NAN, tnow: f64::NAN, tl: if sp.qev { vec![0.0; ns] } else { Vec::new() },
        qe_arr: if sp.qev { vec![0.0; nreq] } else { Vec::new() }, qe_start: if sp.qev { vec![0.0; nreq] } else { Vec::new() },
        qe_svc: if sp.qev { vec![0.0; nreq] } else { Vec::new() }, qe_wait: if sp.qev { vec![0.0; nreq] } else { Vec::new() },
        qe_order: if sp.qev { vec![-1; nreq] } else { Vec::new() },
        qe_audit: if sp.qev { env::var("X1FAST_QEV_AUDIT").ok().and_then(|s| s.parse().ok()).unwrap_or(0) } else { 0 }, qe_naudit: 0,
        qpost2: sp.qpost2,
    }
}

// 逐請求陣列、I/O 線狀態、引擎計數（＝原本 finish 的非摘要那段）
fn write_out(e: &Eng, od: &str) {
    fs::create_dir_all(od).unwrap();
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
    if let Some(sc) = &e.sc { wr_i64(&format!("{}/eng_sc.i64", od), &[sc.cap, sc.n_clean, sc.n_drop]); }
    if e.apc { wr_i64(&format!("{}/eng_v8.i64", od), &[e.n_gblk]); }
    for (k, (w, st)) in e.qpw.iter().zip(e.qps.iter()).enumerate() {
        wr_f64(&format!("{}/v{}_qpw.f64", od, k), w);
        wr_f64(&format!("{}/v{}_qps.f64", od, k), st);
    }
    if e.qev {                                    // v9：每個請求的到達、開始、服務、等待（序號順序）；服務順序；稽核次數
        wr_f64(&format!("{}/v0_qe_arr.f64", od), &e.qe_arr);
        wr_f64(&format!("{}/v0_qe_start.f64", od), &e.qe_start);
        wr_f64(&format!("{}/v0_qe_svc.f64", od), &e.qe_svc);
        wr_f64(&format!("{}/v0_qe_wait.f64", od), &e.qe_wait);
        wr_i64(&format!("{}/v0_qe_order.i64", od), &e.qe_order);
        wr_i64(&format!("{}/qe_audit.i64", od), &[e.qe_audit as i64, e.qe_naudit]);
    }
    if e.qon {
        wr_f64(&format!("{}/qwait.f64", od), &e.qwait);
        wr_f64(&format!("{}/qstart.f64", od), &e.qstart);
        wr_f64(&format!("{}/qsvc.f64", od), &e.qsvc);
        wr_f64(&format!("{}/qarr.f64", od), &e.qarr);
    }
}

// 常駐模式的一個 job（＝原本 run_unit 只有一個 job 的那條路）
fn run_job_dir(tr: &Trace, grp: &[u32], sstart: &[f64], jd: &str) -> f64 {
    let rdat = load_run_data(jd);
    let txt = fs::read_to_string(format!("{}/jobs.txt", jd)).unwrap();
    let lines: Vec<&str> = txt.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(lines.len(), 1, "常駐模式：jobs.txt 只能有一行（{}）", jd);
    let sp = parse_spec(lines[0]);
    let mut lay: Vec<u8> = Vec::new();
    let mut tend: Vec<f64> = Vec::new();
    let tj = Instant::now();
    let mut e = new_eng(tr, grp, sstart, &rdat, &sp, sp.order, sp.act, false, false);
    if e.qev { e.run_qev(&mut lay, &mut tend); } else { for r in 0..tr.t.len() { e.step(r, &mut lay, &mut tend); } }
    if e.qpost { e.q_post(); }
    let run_s = tj.elapsed().as_secs_f64();
    write_out(&e, &format!("{}/out/{}", jd, sp.jid));
    fs::write(format!("{}/timing.txt", jd), format!("{} {}\n", sp.jid, run_s)).unwrap();
    run_s
}

fn serve(tr: &Trace, grp: &[u32], sstart: &[f64], nthreads: usize) {
    use std::io::{BufRead, Write};
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    let rx = std::sync::Mutex::new(rx);
    let out = std::sync::Mutex::new(std::io::stdout());
    std::thread::scope(|sc| {
        for _ in 0..nthreads {
            sc.spawn(|| loop {
                let jd = { let r = rx.lock().unwrap(); match r.recv() { Ok(x) => x, Err(_) => break } };
                let tw = Instant::now();
                let run_s = run_job_dir(tr, grp, sstart, &jd);
                let mut o = out.lock().unwrap();
                writeln!(o, "done {} {} {}", jd, run_s, tw.elapsed().as_secs_f64()).unwrap();
                o.flush().unwrap();
            });
        }
        for line in std::io::stdin().lock().lines() {
            let l = line.unwrap();
            let l = l.trim();
            if l.is_empty() { continue; }
            tx.send(l.to_string()).unwrap();
        }
        drop(tx);
    });
}

fn main() {
    let a: Vec<String> = env::args().collect();
    let td = &a[1];
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
    let nthreads: usize = env::var("X1FAST_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    if env::var("X1FAST_SERVE").map(|s| s == "1").unwrap_or(false) {
        eprintln!("x1fast serve: trace_load_s {} threads {}", t0.elapsed().as_secs_f64(), nthreads);
        serve(&tr, &grp, &sstart, nthreads);
        return;
    }
    // ---- 批次模式（原本的用法）：x1fast <trace_dir> <run_dir>
    let rd_ = &a[2];
    let rdat = load_run_data(rd_);
    let t_load = t0.elapsed().as_secs_f64();
    let mut timing = format!("trace_load_s {}\n", t_load);
    // 多執行緒：X1FAST_THREADS 個執行緒共用同一份唯讀紀錄，從共用佇列各取一個 job；每個 job 的計算和單執行緒完全相同
    let summary_only = env::var("X1FAST_SUMMARY_ONLY").map(|s| s == "1").unwrap_or(false);
    let jobs_txt = fs::read_to_string(format!("{}/jobs.txt", rd_)).unwrap();
    let build = |sp: &Spec, order: Ord_, act: Act, multi: bool, need_gdsf: bool| -> Eng {
        new_eng(&tr, &grp, &sstart, &rdat, sp, order, act, multi, need_gdsf)
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
        write_out(e, &format!("{}/out/{}", rd_, jid));
        format!("{} {}{}\n", jid, run_s, extra)
    };
    // ---- 分組：X1FAST_FORK=1 時，同一組設定（除了 CPU 的順序與動作）、沒有 hook 也沒有預取的 job 共用前段
    let fork = env::var("X1FAST_FORK").map(|s| s == "1").unwrap_or(false);
    let all: Vec<&str> = jobs_txt.lines().filter(|l| !l.trim().is_empty()).collect();
    let mut units: Vec<Vec<usize>> = Vec::new();
    {
        let mut key_of: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for (i, l) in all.iter().enumerate() {
            let x: Vec<&str> = l.split_whitespace().collect();
            if fork && x[3] == "-" && x[4] == "-" && x.len() == 13 {
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
            let sp = parse_spec(all[u[0]]);
            let tj = Instant::now();
            let mut e = build(&sp, sp.order, sp.act, false, false);
            if e.qev { e.run_qev(&mut lay, &mut tend); } else { for r in 0..nreq { e.step(r, &mut lay, &mut tend); } }
            if e.qpost { e.q_post(); }
            return vec![(u[0], finish(&e, sp.jid, tj.elapsed().as_secs_f64(), ""))];
        }
        let specs: Vec<Spec> = u.iter().map(|&i| parse_spec(all[i])).collect();
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
