// w7gen — W7（合成參數地圖）的合成紀錄產生器（icpe-rux；PREREG 偏離 12）。不用外部 crate、不讀任何下載的資料。
//
// 指令：
//   w7gen design --n 90 --out design.csv            拉丁超立方（LHS）：n 個點 × 9 維，種子固定 LHS_SEED
//   w7gen gen --design design.csv --point P --seed S --out-dir DIR [--target-req 20000]
//        → DIR/{t.f64,sess.i64,parent.i64,kind.i8,nall.i64,full_off.i64,full_key.i64,user.u8,svc.f64,gap.f64,start.f64,end.f64,meta.json}
//        （小端原始陣列；w7_pack.py 用 x1_nb.write_nb 包成 nb 的 npz，並做獨立檢查）
//
// 模型（W7_DESIGN.md §2）：
//   - 固定並行 N：N 個槽；一個 session 結束（最後一個請求的時刻＋它的服務時間）就在那個時刻開下一個（同 SemiAnalysis 的到達合成）。
//     開到總請求數 ≥ target 就不再開新的；已開的 session 全部保留。
//   - 每個 session 的輪數 K ~ Geometric(1/turns_mean)（≥1，平均 turns_mean）。
//   - 開新 session 時，以機率 fork_frac 從「當下活著的 session」均勻挑一個 fork：第一個請求＝那個 session 當下最新請求的完整 chunk
//     （共用鍵）＋一次增量；否則是新對話：system prompt（sys_tokens，所有新對話共用同一段鍵）＋自己的起始內容
//     （log-normal，中位數 init_ctx_tokens、σ＝SIGMA_INIT）。
//   - 第 k≥1 輪：觸發者以機率 tool_share 是工具、否則是人；間隔＝分佈族（exp／lognormal／lomax，平均 1）× 該類的平均，
//     兩類平均的比固定為 FreeInference week01 的中位數比 R_HUMAN_TOOL，混合平均＝gap_mean_s；上下文＋增量（log-normal，
//     中位數 inc_tokens、σ＝SIGMA_INC，≥1）；超過 WIN（1,048,576）token 就結束這個 session（那一輪和之後都不產生）。
//   - 時刻：t_k＝t_{k−1}＋svc_{k−1}＋gap_k；svc＝(A＋B·out＋C·input)/1000 秒（FreeInference 擬合，只用來排時刻，引擎自己算 TTFT）。
//   - user_trigger：新對話的第一個請求、人觸發的請求＝true；fork 的第一個請求、工具觸發＝false。
//   - 鍵：系統 prompt 的完整塊＝0..sys_full−1（全部新對話共用）；其餘都是新鍵；同一個 session 下一輪＝上一輪的完整塊＋新鍵（鏈式前綴）。
//   - 每個 session 的亂數：固定順序（u_fork, u_parent, u_K, z_init），每輪固定 4 組（u_trig, g1, g2, z_inc(2), z_out(2)）。
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs;
use std::io::Write;

const CHUNK: i64 = 512;
const WIN: i64 = 1048576; // 上下文上限 2^20 token：涵蓋 SemiAnalysis 實測最大 990,208（N16 s1 的 nall 最大 1934）；主結果的真實紀錄沒有截在 128K
const LHS_SEED: u64 = 20261107;
const GEN_SALT: u64 = 0x5737_4745_4e00_0001; // "W7GEN" 的鹽，和 LHS 的亂數流分開
// FreeInference week01 的擬合值（icpe-next-sim：results/p3_data/synth_fit_fi_week01.json、service_constants.json）
const SIGMA_LN_GAP: f64 = 1.5684286763930078; // 工具間隔 log-normal 的 σ
const Q50_TOOL_MS: f64 = 3409.0;
const Q50_HUMAN_MS: f64 = 174983.5;
const SVC_A: f64 = 1509.1056832666832;
const SVC_B: f64 = 14.212994865389332;
const SVC_C: f64 = 0.014841663618334697;
const OUT_Q10: f64 = 43.0;
const OUT_Q50: f64 = 137.0;
const OUT_Q90: f64 = 813.0;
const Z90: f64 = 1.2815515655446004;
const SIGMA_INIT: f64 = 0.5;
const SIGMA_INC: f64 = 1.0;
const LOMAX_ALPHA: f64 = 1.5;
const FAMILIES: [&str; 3] = ["exp", "lognormal", "lomax"];

// ---------------------------------------------------------------- 亂數：xoshiro256**，用 splitmix64 播種
struct Rng {
    s: [u64; 4],
}
fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}
impl Rng {
    fn new(seed_words: &[u64]) -> Rng {
        let mut x = 0x243F6A8885A308D3u64;
        for &w in seed_words {
            x ^= w;
            splitmix(&mut x);
        }
        let mut s = [0u64; 4];
        for v in s.iter_mut() {
            *v = splitmix(&mut x);
        }
        Rng { s }
    }
    fn next(&mut self) -> u64 {
        let r = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }
    /// (0,1) 開區間：不會是 0 或 1
    fn u(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) * (1.0 / 9007199254740992.0)
    }
    /// 標準常態（Box–Muller，每次用 2 個均勻數、只取 cos 那一個）
    fn normal(&mut self) -> f64 {
        let u1 = self.u();
        let u2 = self.u();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    fn below(&mut self, n: usize) -> usize {
        // Fisher–Yates 用；n ≤ 數百，偏差可忽略但仍用乘法取整，固定可重現
        let v = (self.u() * n as f64) as usize;
        v.min(n - 1)
    }
}

// ---------------------------------------------------------------- 參數
#[derive(Clone, Debug)]
struct Params {
    point: usize,
    turns_mean: f64,
    gap_family: usize,
    gap_mean_s: f64,
    tool_share: f64,
    init_ctx_tokens: f64,
    inc_tokens: f64,
    sys_tokens: i64,
    fork_frac: f64,
    n_conc: usize,
}
const DIMS: [&str; 9] =
    ["turns_mean", "gap_family", "gap_mean_s", "tool_share", "init_ctx_tokens", "inc_tokens", "sys_tokens", "fork_frac", "N"];

fn logu(u: f64, lo: f64, hi: f64) -> f64 {
    (lo.ln() + u * (hi.ln() - lo.ln())).exp()
}
fn map_point(point: usize, u: &[f64; 9]) -> Params {
    Params {
        point,
        turns_mean: logu(u[0], 2.0, 200.0),
        gap_family: ((u[1] * 3.0) as usize).min(2),
        gap_mean_s: logu(u[2], 1.0, 600.0),
        tool_share: u[3],
        init_ctx_tokens: logu(u[4], 512.0, 65536.0),
        inc_tokens: logu(u[5], 16.0, 8192.0),
        sys_tokens: ((u[6] * 32768.0) as i64).min(32767),
        fork_frac: 0.5 * u[7],
        n_conc: logu(u[8], 8.0, 128.0).round() as usize,
    }
}

fn design(n: usize, out: &str) {
    let mut rng = Rng::new(&[LHS_SEED]);
    let d = 9;
    let mut u = vec![[0f64; 9]; n];
    for j in 0..d {
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let k = rng.below(i + 1);
            perm.swap(i, k);
        }
        for i in 0..n {
            u[i][j] = (perm[i] as f64 + rng.u()) / n as f64;
        }
    }
    let mut s = String::from("point");
    for k in 0..d {
        s += &format!(",u{}", k);
    }
    for nm in DIMS {
        s += &format!(",{}", nm);
    }
    s += "\n";
    for i in 0..n {
        let p = map_point(i, &u[i]);
        s += &format!("{}", i);
        for k in 0..d {
            s += &format!(",{:?}", u[i][k]);
        }
        s += &format!(
            ",{:?},{},{:?},{:?},{:?},{:?},{},{:?},{}\n",
            p.turns_mean, FAMILIES[p.gap_family], p.gap_mean_s, p.tool_share, p.init_ctx_tokens, p.inc_tokens, p.sys_tokens,
            p.fork_frac, p.n_conc
        );
    }
    fs::write(out, s).expect("寫 design.csv");
    eprintln!("design: {} 個點 → {}", n, out);
}

fn read_point(path: &str, point: usize) -> Params {
    let txt = fs::read_to_string(path).expect("讀 design.csv");
    let mut lines = txt.lines();
    let head: Vec<&str> = lines.next().unwrap().split(',').collect();
    let col = |nm: &str| head.iter().position(|h| *h == nm).unwrap_or_else(|| panic!("design.csv 沒有欄 {}", nm));
    for ln in lines {
        let f: Vec<&str> = ln.split(',').collect();
        if f[0].parse::<usize>().unwrap() != point {
            continue;
        }
        let mut u = [0f64; 9];
        for k in 0..9 {
            u[k] = f[col(&format!("u{}", k))].parse().unwrap();
        }
        let p = map_point(point, &u);
        // design.csv 的映射欄必須和由 u 重算的一致（防止手改）
        assert_eq!(f[col("gap_family")], FAMILIES[p.gap_family]);
        assert_eq!(f[col("N")].parse::<usize>().unwrap(), p.n_conc);
        assert_eq!(f[col("sys_tokens")].parse::<i64>().unwrap(), p.sys_tokens);
        assert_eq!(f[col("turns_mean")].parse::<f64>().unwrap().to_bits(), p.turns_mean.to_bits());
        return p;
    }
    panic!("design.csv 沒有點 {}", point);
}

// ---------------------------------------------------------------- 產生
struct Req {
    t: f64,
    c: i64,
    sess: usize,
    keys: Vec<i64>,
    user: bool,
    svc: f64,
    gap: f64, // 前一個請求結束到這個請求的間隔；session 的第一個請求＝NaN
}
struct Sess {
    start: f64,
    end: f64,
    fork: bool,
    reqs: Vec<usize>, // 依時間
}

fn family_unit(fam: usize, g1: f64, g2: f64) -> f64 {
    match fam {
        0 => -g1.ln(),
        1 => {
            let z = (-2.0 * g1.ln()).sqrt() * (2.0 * std::f64::consts::PI * g2).cos();
            (SIGMA_LN_GAP * z - 0.5 * SIGMA_LN_GAP * SIGMA_LN_GAP).exp()
        }
        _ => (g1.powf(-1.0 / LOMAX_ALPHA) - 1.0) * (LOMAX_ALPHA - 1.0),
    }
}

struct Gen {
    reqs: Vec<Req>,
    sess: Vec<Sess>,
    stats: HashMap<&'static str, f64>,
}

fn generate(p: &Params, seed: u64, target: usize) -> Gen {
    let mut rng = Rng::new(&[GEN_SALT, LHS_SEED, p.point as u64, seed]);
    let p_end = 1.0 / p.turns_mean;
    let ln1p = (1.0 - p_end).ln();
    let r = Q50_HUMAN_MS / Q50_TOOL_MS;
    let a_tool = p.gap_mean_s / (p.tool_share + r * (1.0 - p.tool_share));
    let a_human = r * a_tool;
    let out_sigma = (OUT_Q90.ln() - OUT_Q10.ln()) / (2.0 * Z90);
    let sys_full = p.sys_tokens / CHUNK;
    let mut next_key: i64 = sys_full;
    let mut reqs: Vec<Req> = Vec::new();
    let mut sess: Vec<Sess> = Vec::new();
    let mut active: Vec<usize> = Vec::new();
    let mut heap: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
    for slot in 0..p.n_conc {
        heap.push(Reverse((0f64.to_bits(), slot)));
    }
    let (mut n_fork, mut n_fork_fallback, mut n_cap, mut n_first_clip) = (0usize, 0usize, 0usize, 0usize);
    let mut total = 0usize;
    while total < target {
        let Reverse((tb, slot)) = heap.pop().unwrap();
        let t0 = f64::from_bits(tb);
        active.retain(|&j| sess[j].end > t0);
        let u_fork = rng.u();
        let u_par = rng.u();
        let u_k = rng.u();
        let z_init = rng.normal();
        let k_total: usize = if p_end >= 1.0 { 1 } else { 1 + (u_k.ln() / ln1p).floor() as usize };
        let want_fork = u_fork < p.fork_frac;
        let fork = want_fork && !active.is_empty();
        if want_fork && !fork {
            n_fork_fallback += 1;
        }
        let sid = sess.len();
        sess.push(Sess { start: t0, end: f64::NAN, fork, reqs: Vec::new() });
        let (mut c, mut keys, mut t, mut svc_prev) = (0i64, Vec::<i64>::new(), t0, 0f64);
        for k in 0..k_total {
            let u_trig = rng.u();
            let g1 = rng.u();
            let g2 = rng.u();
            let z_inc = rng.normal();
            let z_out = rng.normal();
            let inc = ((p.inc_tokens * (SIGMA_INC * z_inc).exp()).round() as i64).max(1);
            let (user, gap);
            if k == 0 {
                if fork {
                    let j = active[((u_par * active.len() as f64) as usize).min(active.len() - 1)];
                    let rs = &sess[j].reqs;
                    // 那個 session 在 t0（含）之前最新的請求
                    let mut li = rs[0];
                    for &ri in rs.iter() {
                        if reqs[ri].t <= t0 {
                            li = ri;
                        } else {
                            break;
                        }
                    }
                    let pc = reqs[li].c;
                    keys = reqs[li].keys.clone();
                    c = (pc + inc).min(WIN);
                    n_fork += 1;
                } else {
                    let own = ((p.init_ctx_tokens * (SIGMA_INIT * z_init).exp()).round() as i64).max(1);
                    c = p.sys_tokens + own;
                    if c > WIN {
                        c = WIN;
                        n_first_clip += 1;
                    }
                    keys = (0..sys_full.min(c / CHUNK)).collect();
                }
                user = !fork;
                gap = f64::NAN;
            } else {
                let tool = u_trig < p.tool_share;
                let g = family_unit(p.gap_family, g1, g2) * if tool { a_tool } else { a_human };
                let cn = c + inc;
                if cn > WIN {
                    n_cap += 1;
                    break;
                }
                c = cn;
                t = t + svc_prev + g;
                user = !tool;
                gap = g;
            }
            let nf = c / CHUNK;
            while (keys.len() as i64) < nf {
                keys.push(next_key);
                next_key += 1;
            }
            let out = ((OUT_Q50 * (out_sigma * z_out).exp()).round()).max(1.0);
            let svc = (SVC_A + SVC_B * out + SVC_C * c as f64).max(0.0) / 1000.0;
            let ri = reqs.len();
            reqs.push(Req { t, c, sess: sid, keys: keys.clone(), user, svc, gap });
            sess[sid].reqs.push(ri);
            svc_prev = svc;
        }
        let last = *sess[sid].reqs.last().unwrap();
        let end = reqs[last].t + reqs[last].svc;
        sess[sid].end = end;
        total += sess[sid].reqs.len();
        active.push(sid);
        heap.push(Reverse((end.to_bits(), slot)));
    }
    let mut stats = HashMap::new();
    stats.insert("n_fork", n_fork as f64);
    stats.insert("n_fork_fallback", n_fork_fallback as f64);
    stats.insert("n_cap_end", n_cap as f64);
    stats.insert("n_first_clip", n_first_clip as f64);
    stats.insert("a_tool_s", a_tool);
    stats.insert("a_human_s", a_human);
    stats.insert("out_sigma", out_sigma);
    stats.insert("sys_full", sys_full as f64);
    stats.insert("n_keys", next_key as f64);
    Gen { reqs, sess, stats }
}

// ---------------------------------------------------------------- 活工作集（和 w7_ws.py 同一個定義、同一個累加順序）
fn ws_profile(t: &[f64], sess: &[i64], off: &[i64], key: &[i64]) -> (f64, i64, f64) {
    let n = t.len();
    let mut last: HashMap<i64, usize> = HashMap::new();
    for i in 0..n {
        last.insert(sess[i], i);
    }
    let mut is_last = vec![false; n];
    for &i in last.values() {
        is_last[i] = true;
    }
    let mut refc: HashMap<i64, u32> = HashMap::new();
    let mut cur: HashMap<i64, Vec<i64>> = HashMap::new();
    let (mut ws, mut wmax, mut acc) = (0i64, 0i64, 0f64);
    for i in 0..n {
        if i > 0 {
            let dt = t[i] - t[i - 1];
            if dt > 0.0 {
                acc += dt * ws as f64;
            }
        }
        let s = sess[i];
        let newk: Vec<i64> = if is_last[i] { Vec::new() } else { key[off[i] as usize..off[i + 1] as usize].to_vec() };
        let old = cur.remove(&s).unwrap_or_default();
        let so: HashSet<i64> = old.iter().copied().collect();
        let sn: HashSet<i64> = newk.iter().copied().collect();
        for k in sn.iter() {
            if !so.contains(k) {
                let c = refc.entry(*k).or_insert(0);
                if *c == 0 {
                    ws += 1;
                }
                *c += 1;
            }
        }
        for k in so.iter() {
            if !sn.contains(k) {
                let c = refc.get_mut(k).unwrap();
                *c -= 1;
                if *c == 0 {
                    ws -= 1;
                    refc.remove(k);
                }
            }
        }
        if !newk.is_empty() {
            cur.insert(s, newk);
        }
        wmax = wmax.max(ws);
    }
    assert!(ws == 0 && refc.is_empty());
    let span = if n > 0 { t[n - 1] - t[0] } else { 0.0 };
    let mean = if span > 0.0 { acc / span } else { ws as f64 };
    (mean, wmax, span)
}

fn wr<T: Copy>(dir: &str, nm: &str, v: &[T], f: impl Fn(T) -> Vec<u8>) {
    let mut b = Vec::with_capacity(v.len() * 8);
    for &x in v {
        b.extend_from_slice(&f(x));
    }
    fs::write(format!("{}/{}", dir, nm), b).expect("寫陣列");
}

fn gen_cmd(design_path: &str, point: usize, seed: u64, dir: &str, target: usize) {
    let p = read_point(design_path, point);
    let g = generate(&p, seed, target);
    // 依 (t, 產生順序) 排序；session id 依排序後第一次出現的順序編號
    let mut order: Vec<usize> = (0..g.reqs.len()).collect();
    order.sort_by(|&a, &b| g.reqs[a].t.partial_cmp(&g.reqs[b].t).unwrap().then(a.cmp(&b)));
    let n = order.len();
    let mut sid_map: HashMap<usize, i64> = HashMap::new();
    let mut lastreq: HashMap<i64, i64> = HashMap::new();
    let (mut t, mut sessv, mut parent, mut kind, mut nall, mut user, mut svc, mut gap) =
        (vec![], vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    let mut off: Vec<i64> = vec![0];
    let mut key: Vec<i64> = vec![];
    for &ri in order.iter() {
        let r = &g.reqs[ri];
        let nid = sid_map.len() as i64;
        let s = *sid_map.entry(r.sess).or_insert(nid);
        let i = t.len() as i64;
        match lastreq.get(&s) {
            Some(&pi) => {
                parent.push(pi);
                kind.push(1i8);
            }
            None => {
                parent.push(-1);
                kind.push(if g.sess[r.sess].fork { 2i8 } else { 0i8 });
            }
        }
        lastreq.insert(s, i);
        t.push(r.t);
        sessv.push(s);
        nall.push((r.c + CHUNK - 1) / CHUNK);
        user.push(r.user as u8);
        svc.push(r.svc);
        gap.push(r.gap);
        key.extend_from_slice(&r.keys);
        off.push(key.len() as i64);
    }
    assert!(t[0] == 0.0, "第一個請求的時刻應該是 0");
    // 每個 session 的開始／結束（槽的占用區間）
    let mut start = vec![0f64; g.sess.len()];
    let mut end = vec![0f64; g.sess.len()];
    for (old, &new) in sid_map.iter() {
        start[new as usize] = g.sess[*old].start;
        end[new as usize] = g.sess[*old].end;
    }
    let (ws_ref, ws_max, span) = ws_profile(&t, &sessv, &off, &key);
    fs::create_dir_all(dir).expect("建目錄");
    wr(dir, "t.f64", &t, |x: f64| x.to_le_bytes().to_vec());
    wr(dir, "sess.i64", &sessv, |x: i64| x.to_le_bytes().to_vec());
    wr(dir, "parent.i64", &parent, |x: i64| x.to_le_bytes().to_vec());
    wr(dir, "kind.i8", &kind, |x: i8| x.to_le_bytes().to_vec());
    wr(dir, "nall.i64", &nall, |x: i64| x.to_le_bytes().to_vec());
    wr(dir, "full_off.i64", &off, |x: i64| x.to_le_bytes().to_vec());
    wr(dir, "full_key.i64", &key, |x: i64| x.to_le_bytes().to_vec());
    wr(dir, "user.u8", &user, |x: u8| vec![x]);
    wr(dir, "svc.f64", &svc, |x: f64| x.to_le_bytes().to_vec());
    wr(dir, "gap.f64", &gap, |x: f64| x.to_le_bytes().to_vec());
    wr(dir, "start.f64", &start, |x: f64| x.to_le_bytes().to_vec());
    wr(dir, "end.f64", &end, |x: f64| x.to_le_bytes().to_vec());
    let src = include_str!("main.rs");
    let mut m = String::from("{\n");
    m += &format!(" \"generator\": \"w7gen\", \"point\": {}, \"seed\": {}, \"target_req\": {},\n", p.point, seed, target);
    m += &format!(
        " \"params\": {{\"turns_mean\": {:?}, \"gap_family\": \"{}\", \"gap_mean_s\": {:?}, \"tool_share\": {:?}, \"init_ctx_tokens\": {:?}, \"inc_tokens\": {:?}, \"sys_tokens\": {}, \"fork_frac\": {:?}, \"N\": {}}},\n",
        p.turns_mean, FAMILIES[p.gap_family], p.gap_mean_s, p.tool_share, p.init_ctx_tokens, p.inc_tokens, p.sys_tokens, p.fork_frac, p.n_conc
    );
    let mut ks: Vec<_> = g.stats.iter().collect();
    ks.sort_by(|a, b| a.0.cmp(b.0));
    m += " \"gen_stats\": {";
    m += &ks.iter().map(|(k, v)| format!("\"{}\": {:?}", k, v)).collect::<Vec<_>>().join(", ");
    m += "},\n";
    m += &format!(
        " \"n_req\": {}, \"n_sess\": {}, \"n_cont\": {}, \"n_fork_req\": {}, \"full_chunks_total\": {}, \"span_s\": {:?},\n",
        n,
        sid_map.len(),
        kind.iter().filter(|&&k| k == 1).count(),
        kind.iter().filter(|&&k| k == 2).count(),
        key.len(),
        span
    );
    m += &format!(" \"WS_ref_GiB\": {:?}, \"WS_max_GiB\": {:?},\n", ws_ref * 64.0 / 1024.0, ws_max as f64 * 64.0 / 1024.0);
    m += &format!(" \"w7gen_src_bytes\": {}\n}}\n", src.len());
    fs::write(format!("{}/meta.json", dir), m).expect("寫 meta");
    eprintln!("gen: point {} seed {} → {}（{} 個請求、{} 個 session、WS_ref {:.2} GiB）", point, seed, dir, n, sid_map.len(),
        ws_ref * 64.0 / 1024.0);
}

fn arg<'a>(a: &'a [String], k: &str) -> Option<&'a str> {
    a.iter().position(|x| x == k).map(|i| a[i + 1].as_str())
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(|s| s.as_str()) {
        Some("design") => design(arg(&a, "--n").unwrap_or("90").parse().unwrap(), arg(&a, "--out").expect("--out")),
        Some("gen") => gen_cmd(
            arg(&a, "--design").expect("--design"),
            arg(&a, "--point").expect("--point").parse().unwrap(),
            arg(&a, "--seed").expect("--seed").parse().unwrap(),
            arg(&a, "--out-dir").expect("--out-dir"),
            arg(&a, "--target-req").unwrap_or("20000").parse().unwrap(),
        ),
        _ => {
            eprintln!("用法：w7gen design --n 90 --out design.csv | w7gen gen --design d.csv --point P --seed S --out-dir DIR [--target-req 20000]");
            std::process::exit(2);
        }
    }
    let _ = std::io::stdout().flush();
}
