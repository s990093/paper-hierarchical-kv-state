"""make_v9.py — 從 v8_dev14 的 main.rs 產生 v9 的 main.rs：v8_dev14（v8＋偏離 14 的規則）＋二階排隊 `qev=1`（事件驅動、closed-loop FCFS、
單一 GPU；PREREG 偏離 19）。
用法：python -I code/icpe/rux_x1fast/v9_qev/make_v9.py   （讀 ../v8_dev14/src/main.rs，寫 src/main.rs；每個替換都必須剛好出現一次）
v8、v8_dev14 一行不改（v8 main.rs 48e4efce 是偏離 10–12 登記的引擎）。沒有 `qev=1`／`qpost=2` 時，v9 每條路徑的運算和 v8_dev14 相同（smoke_v9.py T2）。
使用者 11/7 約 15:4x 選「先移植 v8，再做二階排隊」→ 二階也帶偏離 14 的規則：
  - pfx 的「預測被讀的請求已經過去」：沒有 qev＝st_t ≤ cur（v8_dev14）；qev＝那個請求已經服務過（served[st_t]）。沒有重排時兩者相同。
  - nzo（雜訊）在 qev 不能開（雜訊本來就關）；pgr 只在預取進 GPU 時有作用。

語意（docs/icpe_brief_20261030/next/rux/QUEUE2_DESIGN.md）：
  - 服務順序：GPU 一次一個；到達＝紀錄時刻＋這個 session 上一個請求「開始服務比紀錄晚多少」（同一階 qpost）；照到達先後服務（同時到照序號）。
  - 快取狀態照服務順序演進：請求在開始時刻讀快取（還原），回合結束的狀態變動也在這一步做（下一個請求一定在它完成之後才開始）。
  - ④ 只知道紀錄的順序（使用者 11/7 15:12 選「紀錄順序」）：塊 b 的「下次回來」＝還沒服務的請求裡，序號最小、用到 b 的那個。
  - 新近度（LRU2、FREQ、PENS、GDSF、Pred、預取 RPF、make_room 的 last 排序）＝服務順序的序號 seq。
  - BELS 的距離：nx −（還沒服務的最小序號 − 1）；沒有重排時＝nx − cur（和 v8 相同）。
  - 回報的 TTFT＝等待＋服務；服務＝在開始時刻用當下快取狀態算的還原＋prefill。
  - 到達不早於同一個 session 上一個請求完成（使用者 11/7 約 15:40 選「二階改，一階也重跑」）：到達＝max(紀錄時刻＋上一個開始的延後, 上一個完成)。
另外加 `qpost=2`：一階近似（qpost）換成同樣「不早於上一個完成」的到達（上一個完成＝開始＋沒排隊的 TTFT），當二階的對照。`qpost=1` 不變。
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "v8_dev14", "src", "main.rs")
DST = os.path.join(HERE, "src", "main.rs")
s = open(SRC, encoding="utf-8").read()
R = []


def rep(old, new):
    R.append((old, new))


# ---- 0. 檔頭
rep("// icpe-rux v8_dev14（2026-11-07，code/icpe/rux_x1fast/v8_dev14）",
    "// icpe-rux v9（2026-11-07，code/icpe/rux_x1fast/v9_qev）：v8_dev14＋二階排隊 qev=1（事件驅動）＋一階 qpost=2（make_v9.py；PREREG 偏離 19）。\n"
    "//   沒有 qev=1／qpost=2 時每條路徑和 v8_dev14 相同。\n"
    "// icpe-rux v8_dev14（2026-11-07，code/icpe/rux_x1fast/v8_dev14）")
# ---- pf_expire（偏離 14 (i)）的「已經過去」：qev 時＝已經服務過
rep("    fn pf_expire(&mut self) {\n        let cur = self.cur;\n",
    "    fn pf_expire(&mut self) {\n")
rep("    fn pf_push(&mut self, s: usize) {",
    "    // v9：預測被讀的請求 t 已經過去？沒有 qev＝t ≤ cur（同 v8_dev14）；qev＝t 已經服務過（cur 是服務順序，不能和序號比）\n"
    "    fn st_passed(&self, t: i64) -> bool { if self.qev { t < INF && self.served[t as usize] } else { t <= self.cur } }\n"
    "    fn pf_push(&mut self, s: usize) {")
# ---- 1. Eng 的新欄位
rep("    belp_fast: bool,\n}",
    "    belp_fast: bool,\n"
    "    // v9 二階排隊（qev）：served[r]＝已經開始服務；acc_ptr[b]＝b 的存取清單裡第一個還沒服務的位置（只會往後）；front＝還沒服務的最小序號；\n"
    "    // cur_r＝目前請求的序號（qev 時 cur＝服務順序的序號 seq）；qev_t＝這一步的開始時刻；tnow＝目前請求的開始時刻；tl[s]＝s 上一個請求的開始時刻\n"
    "    qev: bool, served: Vec<bool>, acc_ptr: Vec<Cell<u32>>, front: usize, cur_r: usize, seq: i64, qev_t: f64, tnow: f64, tl: Vec<f64>,\n"
    "    qe_arr: Vec<f64>, qe_start: Vec<f64>, qe_svc: Vec<f64>, qe_wait: Vec<f64>, qe_order: Vec<i64>, qe_audit: usize, qe_naudit: i64,\n"
    "    // v9 qpost=2：一階的到達不早於上一個完成\n"
    "    qpost2: bool,\n"
    "}")
# ---- 2. 「下次回來」：qev 時＝第一個還沒服務的存取
rep("    fn tt(&self, i: i64) -> f64 {",
    "    fn acc(&self, b: u32) -> &[i64] { &self.acc_req[self.acc_off[b as usize]..self.acc_off[b as usize + 1]] }\n"
    "    fn tt(&self, i: i64) -> f64 {")
rep("    fn bl_of(&self, r: i64) -> &'a [u32] {",
    "    // v9：「下次回來」。沒有 qev＝v8（cur 之後第一次存取）；qev＝還沒服務的存取裡序號最小的（④ 只知道紀錄的順序）\n"
    "    fn nacc(&self, b: u32) -> i64 {\n"
    "        if !self.qev { return self.tr.next_access(b, self.cur); }\n"
    "        let lst = self.tr.acc(b);\n"
    "        let c = &self.acc_ptr[b as usize];\n"
    "        let mut i = c.get() as usize;\n"
    "        while i < lst.len() && self.served[lst[i] as usize] { i += 1; }\n"
    "        c.set(i as u32);\n"
    "        if i < lst.len() { lst[i] } else { INF }\n"
    "    }\n"
    "    // v9：快取的「下次回來」還有效嗎（沒有 qev＝v8 的 cur < cnx；qev＝cnx 還沒服務：還沒服務的集合只會變小，所以它仍是第一個）\n"
    "    fn nx_live(&self, cnx: i64) -> bool { if self.qev { !self.served[cnx as usize] } else { self.cur < cnx } }\n"
    "    fn bl_of(&self, r: i64) -> &'a [u32] {")
rep("        if cep == ep && (cnx >= INF || self.cur < cnx) { return (cnx, ce); }",
    "        if cep == ep && (cnx >= INF || self.nx_live(cnx)) { return (cnx, ce); }")
rep("        if cep == ep && (cnx >= INF || self.cur < cnx) { let _ = ce; return cnx; }",
    "        if cep == ep && (cnx >= INF || self.nx_live(cnx)) { let _ = ce; return cnx; }")
rep("        (self.tr.next_access(b, self.cur), e)", "        (self.nacc(b), e)")
rep("        self.tr.next_access(b, self.cur)\n    }\n    fn g_add", "        self.nacc(b)\n    }\n    fn g_add")
rep("            let t = self.tr.next_access(bl[bl.len() - 1], self.cur);\n            let t = if self.noise.is_some() { self.nz(s, t) } else { t };\n            self.st_t[s] = t;",
    "            let t = self.nacc(bl[bl.len() - 1]);\n            let t = if self.noise.is_some() { self.nz(s, t) } else { t };\n            self.st_t[s] = t;")
rep("        let t = self.tr.next_access(bl[bl.len() - 1], self.cur);\n        let t = if self.noise.is_some() { self.nz(s, t) } else { t };",
    "        let t = self.nacc(bl[bl.len() - 1]);\n        let t = if self.noise.is_some() { self.nz(s, t) } else { t };")
# ---- 3. dead／BELS／駐留時間
rep("        if self.tr.tt(if self.cur >= 0 { self.cur } else { 0 }) > self.dead_cut { return false; }",
    "        if self.tr.tt(if self.qev { self.cur_r as i64 } else if self.cur >= 0 { self.cur } else { 0 }) > self.dead_cut { return false; }")
rep("            Ord_::Bels => {\n                let cur = self.cur;",
    "            Ord_::Bels => {\n                let cur = if self.qev { self.front as i64 - 1 } else { self.cur };   // v9：④ 的「現在」＝還沒服務的最小序號之前")
rep("        self.res_sum += self.tr.tt(self.cur) - self.tr.tt(self.last[s]);",
    "        self.res_sum += if self.qev { self.tnow - self.tl[s] } else { self.tr.tt(self.cur) - self.tr.tt(self.last[s]) };")
# ---- 4. 新近度＝cur（沒有 qev 時 cur＝r，同 v8；qev 時＝seq）
PFX_OLD = "self.st_t[x] <= cur && self.st[x] != S_DROP"
PFX_NEW = "self.st_passed(self.st_t[x]) && self.st[x] != S_DROP"
assert s.count(PFX_OLD) == 2, s.count(PFX_OLD)
LAST_OLD = "        self.prev[s] = self.last[s];\n        self.last[s] = r as i64;\n"
LAST_NEW = "        self.prev[s] = self.last[s];\n        self.last[s] = self.cur;\n        if self.qev { self.tl[s] = self.tnow; }\n"
assert s.count(LAST_OLD) == 3, s.count(LAST_OLD)
# ---- 5. step：開始時刻、預取的時間窗、cur
rep("        let t = if self.qon && self.qfree > arr { self.qfree } else { arr };   // v8 排隊：開始＝max(到達, 前一個完成)\n",
    "        let t = if self.qon && self.qfree > arr { self.qfree } else { arr };   // v8 排隊：開始＝max(到達, 前一個完成)\n"
    "        let t = if self.qev { self.qev_t } else { t };                         // v9 qev：開始時刻由 run_qev 給\n")
rep("            let tf = if r > 0 { if self.qon { self.qprev } else { tr.t[r - 1] } } else { t };",
    "            let tf = if self.qev { if self.seq > 0 { self.qprev } else { t } } else if r > 0 { if self.qon { self.qprev } else { tr.t[r - 1] } } else { t };")
rep("        self.cur = r as i64;\n        if self.qon { self.qprev = t; }\n",
    "        if self.qev {                                   // v9：cur＝服務順序的序號；r 標成已服務（之後的「下次回來」不再算它）\n"
    "            self.cur = self.seq; self.cur_r = r; self.tnow = t; self.qprev = t;\n"
    "            self.served[r] = true;\n"
    "            while self.front < self.served.len() && self.served[self.front] { self.front += 1; }\n"
    "            self.seq += 1;\n"
    "        } else { self.cur = r as i64; }\n"
    "        if self.qon { self.qprev = t; }\n")
# ---- 5b. 一階 qpost=2：到達不早於上一個完成（free＝這一個的開始＋沒排隊的 TTFT，已經先更新）
rep("                let late = st - tr.t[r];\n                let nr = nxt[r];\n                if nr != usize::MAX { heap.push(Reverse((F(tr.t[nr] + late), nr))); }\n",
    "                let late = st - tr.t[r];\n                let nr = nxt[r];\n"
    "                if nr != usize::MAX {\n"
    "                    let an = tr.t[nr] + late;\n"
    "                    let an = if self.qpost2 && free > an { free } else { an };      // v9 qpost=2：不早於上一個完成\n"
    "                    heap.push(Reverse((F(an), nr)));\n"
    "                }\n")
# APC 稽核的「最後一個」：qev 時照服務順序（cur＝seq）；沒有 qev 時 cur＝r，同 v8
rep("        if self.g > 0 { self.step_gpu(r, t, blocks, lay, tend); self.sc_check(r); self.v8_check(r); return; }",
    "        if self.g > 0 { self.step_gpu(r, t, blocks, lay, tend); self.sc_check(r); self.v8_check(self.cur as usize); return; }")
# ---- 6. 事件迴圈與稽核（放在 q_post 前面）
rep("    // v8 排隊的一階近似（qpost）：模擬跑完之後呼叫一次\n",
    r'''    // v9 二階排隊（qev）：closed-loop FCFS、單一 GPU。到達＝max(紀錄時刻＋上一個開始的延後, 上一個完成)（同 qpost=2）；
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
''')
# ---- 7. Spec／parse／new_eng
rep("                  q: bool, qc: bool, cue: bool, w5: Option<i64>, qpost: bool, pfx: bool, nzo_rs: bool, pgr: bool }",
    "                  q: bool, qc: bool, cue: bool, w5: Option<i64>, qpost: bool, pfx: bool, nzo_rs: bool, pgr: bool, qev: bool, qpost2: bool }")
rep("    let (mut q, mut qc, mut cue, mut w5, mut qpost) = (false, false, false, None, false);",
    "    let (mut q, mut qc, mut cue, mut w5, mut qpost, mut qev, mut qpost2) = (false, false, false, None, false, false, false);")
rep("        if *extra == \"qpost=1\" { assert!(!qpost); qpost = true; continue; }           // v8 排隊（一階近似，事後算；SEMANTICS §2b）\n",
    "        if *extra == \"qpost=1\" { assert!(!qpost); qpost = true; continue; }           // v8 排隊（一階近似，事後算；SEMANTICS §2b）\n"
    "        if *extra == \"qev=1\" { assert!(!qev); qev = true; continue; }                 // v9 二階排隊（事件驅動；偏離 19）\n"
    "        if *extra == \"qpost=2\" { assert!(!qpost); qpost = true; qpost2 = true; continue; }   // v9 一階、到達不早於上一個完成（偏離 19 的對照）\n")
rep("nz, sc, q, qc, cue, w5, qpost, pfx, nzo_rs, pgr };", "nz, sc, q, qc, cue, w5, qpost, pfx, nzo_rs, pgr, qev, qpost2 };")
rep("    if sp.qpost { assert!(!sp.q, \"qpost 和 q 不能同時開\"); }\n",
    "    if sp.qpost { assert!(!sp.q, \"qpost 和 q 不能同時開\"); }\n"
    "    if sp.qev {                                                         // v9：二階排隊只接偏離 19 用到的路徑\n"
    "        assert!(!sp.q && !sp.qpost, \"qev 和 q／qpost 不能同時開\");\n"
    "        assert!(sp.vids.len() == 1, \"qev：一個 job 只能 1 個時間設定（服務順序取決於時間設定）\");\n"
    "        // 雜訊（nz 用 tl[cur]、cur＋1 當請求序號）和線索（cue[cur]）把 cur 當請求序號；qev 時 cur＝服務順序 → 一定要關（審查第 3 點）\n"
    "        assert!(sp.nz.is_none() && sp.sc.is_none() && !sp.cue && sp.w5.is_none(), \"qev 不支援雜訊／慢層上限／線索／W5\");\n"
    "    }\n")
rep("    if sp.q || sp.cue || sp.w5.is_some() || sp.gorder >= 4 || sp.qpost { assert!(!multi,",
    "    if sp.q || sp.cue || sp.w5.is_some() || sp.gorder >= 4 || sp.qpost || sp.qev { assert!(!multi,")
rep("        nx_cache: [(0..ns).map(|_| Cell::new((0, 0, 0))).collect(), (0..ns).map(|_| Cell::new((0, 0, 0))).collect(),\n"
    "                   (0..ns).map(|_| Cell::new((0, 0, 0))).collect()],\n    }\n}",
    "        nx_cache: [(0..ns).map(|_| Cell::new((0, 0, 0))).collect(), (0..ns).map(|_| Cell::new((0, 0, 0))).collect(),\n"
    "                   (0..ns).map(|_| Cell::new((0, 0, 0))).collect()],\n"
    "        qev: sp.qev, served: if sp.qev { vec![false; nreq] } else { Vec::new() },\n"
    "        acc_ptr: if sp.qev { (0..nb).map(|_| Cell::new(0u32)).collect() } else { Vec::new() },\n"
    "        front: 0, cur_r: 0, seq: 0, qev_t: f64::NAN, tnow: f64::NAN, tl: if sp.qev { vec![0.0; ns] } else { Vec::new() },\n"
    "        qe_arr: if sp.qev { vec![0.0; nreq] } else { Vec::new() }, qe_start: if sp.qev { vec![0.0; nreq] } else { Vec::new() },\n"
    "        qe_svc: if sp.qev { vec![0.0; nreq] } else { Vec::new() }, qe_wait: if sp.qev { vec![0.0; nreq] } else { Vec::new() },\n"
    "        qe_order: if sp.qev { vec![-1; nreq] } else { Vec::new() },\n"
    "        qe_audit: if sp.qev { env::var(\"X1FAST_QEV_AUDIT\").ok().and_then(|s| s.parse().ok()).unwrap_or(0) } else { 0 }, qe_naudit: 0,\n"
    "        qpost2: sp.qpost2,\n"
    "    }\n}")
# ---- 8. 輸出與兩個執行路徑
rep("    if e.qon {\n        wr_f64(&format!(\"{}/qwait.f64\", od), &e.qwait);",
    "    if e.qev {                                    // v9：每個請求的到達、開始、服務、等待（序號順序）；服務順序；稽核次數\n"
    "        wr_f64(&format!(\"{}/v0_qe_arr.f64\", od), &e.qe_arr);\n"
    "        wr_f64(&format!(\"{}/v0_qe_start.f64\", od), &e.qe_start);\n"
    "        wr_f64(&format!(\"{}/v0_qe_svc.f64\", od), &e.qe_svc);\n"
    "        wr_f64(&format!(\"{}/v0_qe_wait.f64\", od), &e.qe_wait);\n"
    "        wr_i64(&format!(\"{}/v0_qe_order.i64\", od), &e.qe_order);\n"
    "        wr_i64(&format!(\"{}/qe_audit.i64\", od), &[e.qe_audit as i64, e.qe_naudit]);\n"
    "    }\n"
    "    if e.qon {\n        wr_f64(&format!(\"{}/qwait.f64\", od), &e.qwait);")
rep("    let mut e = new_eng(tr, grp, sstart, &rdat, &sp, sp.order, sp.act, false, false);\n    for r in 0..tr.t.len() { e.step(r, &mut lay, &mut tend); }\n",
    "    let mut e = new_eng(tr, grp, sstart, &rdat, &sp, sp.order, sp.act, false, false);\n"
    "    if e.qev { e.run_qev(&mut lay, &mut tend); } else { for r in 0..tr.t.len() { e.step(r, &mut lay, &mut tend); } }\n")
rep("            let mut e = build(&sp, sp.order, sp.act, false, false);\n            for r in 0..nreq { e.step(r, &mut lay, &mut tend); }\n",
    "            let mut e = build(&sp, sp.order, sp.act, false, false);\n"
    "            if e.qev { e.run_qev(&mut lay, &mut tend); } else { for r in 0..nreq { e.step(r, &mut lay, &mut tend); } }\n")

for old, new in R:
    c = s.count(old)
    assert c == 1, (c, old[:120])
    s = s.replace(old, new)
s = s.replace(LAST_OLD, LAST_NEW)
s = s.replace(PFX_OLD, PFX_NEW)
os.makedirs(os.path.dirname(DST), exist_ok=True)
open(DST, "w", encoding="utf-8").write(s)
print("make_v9: wrote", DST, "replacements", len(R) + 3)
