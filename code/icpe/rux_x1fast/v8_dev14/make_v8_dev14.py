"""make_v8_dev14.py — 從 v8 的 main.rs（48e4efce，偏離 10–12 的引擎）產生 v8_dev14 的 main.rs：v8＋PREREG 偏離 14 的規則（icpe-sim 的 x1fast）。
用法：python -I code/icpe/rux_x1fast/v8_dev14/make_v8_dev14.py   （讀 ../v8_features/src/main.rs，寫 src/main.rs；每個替換都必須剛好出現一次）
來源：origin/icpe-next-sim 的 code/p3x/x1fast/src/main.rs，63b8e328（40a4c87f 驗證過）對 4a4baa48 的差異（SIM_CHANGES 22、23）。
  - jobs.txt `pfx=1`：(i) O-PF 預取的副本，預測被讀的請求已經過去（st_t ≤ cur）就丟掉、不再排回（Eng::pf_expire；只在 pf＝Opf 時開）。
  - jobs.txt `nzo=resample`：(a) 模型 P 抽到已經過去的時間 → 從「間隔 > 已經過的時間」那部分均勻重抽（Eng::nz）；B1 存回；
    重抽用亂數，所以 B1 也照 Python 走訪 cpu_set 的順序（PySet，同 B2）。
  - pfx 的過濾跳過已丟（S_DROP）的 session（SIM_CHANGES 23）。
  - jobs.txt `pgr=1`（SIM_CHANGES 23）：預取到 GPU 時，騰完空間再查候選；候選已經不在 CPU／慢層（O-ALL 的 O-NOW 騰空間時丟了它）→ 不預取、不排回。
  - 沒有這三欄時和 v8 逐位元相同（smoke_v8_dev14.py D1）。
  - APC／BELP：v8 不准它們和「預取進 GPU」同時開（parse_spec 的 assert），所以 pf_expire 的 GPU 路徑和 pgr 碰不到 APC／BELP 的部分 GPU 副本；
    APC／BELP 加 pfx 只走 CPU 路徑（和 x1fast 相同的程式）。BELP 經 g_next → nz() 讀雜訊，所以 nzo=resample 也作用在 BELP。
使用者 11/7 約 15:4x 在 icpe-main 選「先移植 v8，再做二階排隊」。v8 一行不改。"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "v8_features", "src", "main.rs")
DST = os.path.join(HERE, "src", "main.rs")
s = open(SRC, encoding="utf-8").read()
R = []


def rep(old, new):
    R.append((old, new))


rep("// icpe-rux v8（2026-11-07，code/icpe/rux_x1fast/v8_features）",
    "// icpe-rux v8_dev14（2026-11-07，code/icpe/rux_x1fast/v8_dev14）：v8＋PREREG 偏離 14 的規則（make_v8_dev14.py；來源 icpe-sim 的 x1fast\n"
    "//   63b8e328，SIM_CHANGES 22、23）：jobs.txt 可多 `pfx=1`（O-PF 預取的副本過期就丟，Eng::pf_expire）、`nzo=resample`（模型 P 抽到已經\n"
    "//   過去的時間就重抽，Eng::nz）、`pgr=1`（預取到 GPU 時騰完空間再查候選）。沒有這三欄時和 v8 逐位元相同。\n"
    "// icpe-rux v8（2026-11-07，code/icpe/rux_x1fast/v8_features）")
# ---- Noise
rep("struct Noise { model_p: bool, b1: bool, lam: f64, rng: Mt, gaps: Vec<f64>, nzd: Vec<Option<(f64, f64, f64)>> }",
    "struct Noise { model_p: bool, b1: bool, lam: f64, rng: Mt, gaps: Vec<f64>, nzd: Vec<Option<(f64, f64, f64)>>,\n"
    "               rs: bool, sorted: Vec<f64> }   // rs＝偏離 14 (a)；sorted＝gaps 照值排序（只有 rs 時才建）")
# ---- Eng 欄位
rep("    noise: Option<RefCell<Noise>>,\n",
    "    noise: Option<RefCell<Noise>>,\n"
    "    pfx: bool,   // 偏離 14 (i)：只在 pf＝Opf 時為真\n"
    "    pgr: bool,   // SIM_CHANGES 23\n")
# ---- nz：模型 P 的重抽
rep("            tq_target = tl[self.last[s] as usize] + g;\n",
    "            let ls = tl[self.last[s] as usize];\n"
    "            let mut tqt = ls + g;\n"
    "            if nz.rs {\n"
    "                // 偏離 14 (a)＝Engine._nz：已經過去 → 從「間隔 > 已經過的時間」那部分均勻重抽（bisect_right＋randrange）；B1 取代存著的\n"
    "                let tq = if self.cur >= 0 { tl[self.cur as usize] } else { tl[0] };\n"
    "                if tqt <= tq {\n"
    "                    let e = tq - ls;\n"
    "                    let k = nz.sorted.partition_point(|&x| x <= e);\n"
    "                    let n = nz.sorted.len() - k;\n"
    "                    let j = k + nz.rng.randbelow(n);\n"
    "                    let g2 = nz.sorted[j];\n"
    "                    if nz.b1 { nz.nzd[s] = Some((u, g2, eps)); }\n"
    "                    if g2 >= INF as f64 { return INF; }\n"
    "                    tqt = ls + g2;\n"
    "                }\n"
    "            }\n"
    "            tq_target = tqt;\n")
# ---- pf_expire（放在 pf_push 前面）
rep("    fn pf_push(&mut self, s: usize) {",
    "    // 偏離 14 (i)＝Engine._pf_expire：預測被讀的請求 st_t ≤ cur、還留著預取副本 → 丟掉全部副本，不再排回（pfdone）\n"
    "    // 已丟（S_DROP）的 session 跳過（sb 已清空；那種副本來自 GPU 預取騰空間時候選被丟，見 Python 的說明）\n"
    "    fn pf_expire(&mut self) {\n"
    "        let cur = self.cur;\n"
    "        if self.has_gpf {\n"
    "            let ex: Vec<usize> = self.gpf_sessions.items.iter().copied().filter(|&x| self.gspf[x] > 0 && self.st_t[x] <= cur && self.st[x] != S_DROP).collect();\n"
    "            for s in ex {\n"
    "                let bl = self.sbl(s);\n"
    "                self.g_unprefetch(s, bl, false);\n"
    "                self.pfdone[s] = true;\n"
    "            }\n"
    "        } else {\n"
    "            let ex: Vec<usize> = self.pf_sessions.items.iter().copied().filter(|&x| self.spf[x] > 0 && self.st_t[x] <= cur && self.st[x] != S_DROP).collect();\n"
    "            for s in ex {\n"
    "                for &b in self.sbl(s) {\n"
    "                    let b = b as usize;\n"
    "                    if self.pf_owner[b] == s as i64 { self.cpu_used -= self.i4; self.fmt[b] = 0; self.pf_owner[b] = -1; }\n"
    "                }\n"
    "                self.spf[s] = 0;\n"
    "                self.pf_sessions.discard(s);\n"
    "                self.pfdone[s] = true;\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    fn pf_push(&mut self, s: usize) {")
# ---- step：g_expire 之前（x1fast：t 算好之後、g_expire 之前；cur 仍是上一個請求）
rep("        if self.g > 0 && self.gorder == 2 { self.g_expire(t); }\n        if self.pf != Pf::No {\n",
    "        if self.pfx { self.pf_expire(); }     // 偏離 14 (i)；沒開＝原本的路徑\n"
    "        if self.g > 0 && self.gorder == 2 { self.g_expire(t); }\n        if self.pf != Pf::No {\n")
# ---- Spec／parse
rep("                  q: bool, qc: bool, cue: bool, w5: Option<i64>, qpost: bool }",
    "                  q: bool, qc: bool, cue: bool, w5: Option<i64>, qpost: bool, pfx: bool, nzo_rs: bool, pgr: bool }")
rep("    let (mut q, mut qc, mut cue, mut w5, mut qpost) = (false, false, false, None, false);\n",
    "    let (mut q, mut qc, mut cue, mut w5, mut qpost) = (false, false, false, None, false);\n"
    "    let (mut pfx, mut nzo_rs, mut seen_pfx, mut seen_nzo) = (false, false, false, false);   // 偏離 14\n"
    "    let mut pgr = false;                                                                    // SIM_CHANGES 23\n")
rep("        if let Some(v) = extra.strip_prefix(\"nz=\") {",
    "        if let Some(v) = extra.strip_prefix(\"pgr=\") {          // SIM_CHANGES 23\n"
    "            assert!(v == \"1\" && !pgr, \"pgr={}：{}\", v, line);\n"
    "            pgr = true; continue;\n"
    "        }\n"
    "        if let Some(v) = extra.strip_prefix(\"pfx=\") {          // 偏離 14 (i)\n"
    "            assert!(v == \"1\" && !seen_pfx, \"pfx={}：{}\", v, line);\n"
    "            pfx = true; seen_pfx = true; continue;\n"
    "        }\n"
    "        if let Some(v) = extra.strip_prefix(\"nzo=\") {          // 偏離 14 (a)\n"
    "            assert!((v == \"resample\" || v == \"clamp\") && !seen_nzo, \"nzo={}：{}\", v, line);\n"
    "            nzo_rs = v == \"resample\"; seen_nzo = true; continue;\n"
    "        }\n"
    "        if let Some(v) = extra.strip_prefix(\"nz=\") {")
rep("nz, sc, q, qc, cue, w5, qpost };", "nz, sc, q, qc, cue, w5, qpost, pfx, nzo_rs, pgr };")
# ---- new_eng
rep("        RefCell::new(Noise { model_p, b1, lam, rng: Mt::from_seed(seed), gaps, nzd: vec![None; ns] })",
    "        let sorted = if sp.nzo_rs { let mut v = gaps.clone(); v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v } else { Vec::new() };\n"
    "        RefCell::new(Noise { model_p, b1, lam, rng: Mt::from_seed(seed), gaps, nzd: vec![None; ns], rs: sp.nzo_rs, sorted })")
rep("cpu_set: { let mut c = ISet::new(ns); if matches!(sp.nz, Some((_, _, false, _))) { c.py = Some(PySet::new(ns)); } c },",
    "cpu_set: { let mut c = ISet::new(ns); if matches!(sp.nz, Some((_, _, false, _))) || (sp.nz.is_some() && sp.nzo_rs) { c.py = Some(PySet::new(ns)); } c },")
rep("        grp, gep: vec![0; ns], multi, diverged: false, noise, sc,",
    "        grp, gep: vec![0; ns], multi, diverged: false, noise, sc, pfx: sp.pfx && pf == Pf::Opf, pgr: sp.pgr,")
# ---- advance_prefetch_gpu：騰完空間再查候選（SIM_CHANGES 23）
rep("            if st0 >= t_to || !self.g_make_room(s, todo.len() as i64 * BF, st0) {\n                self.pfheap.push(Reverse((key, s)));\n                return;\n            }\n",
    "            if st0 >= t_to || !self.g_make_room(s, todo.len() as i64 * BF, st0) {\n                self.pfheap.push(Reverse((key, s)));\n                return;\n            }\n"
    "            // SIM_CHANGES 23（pgr=1）：騰空間時候選被丟（O-ALL 的 O-NOW）→ 不預取（Python 的說明）\n"
    "            if self.pgr && !(self.st[s] == S_CPU || self.st[s] == S_SLOW) { continue; }\n")

for old, new in R:
    c = s.count(old)
    assert c == 1, (c, old[:120])
    s = s.replace(old, new)
os.makedirs(os.path.dirname(DST), exist_ok=True)
open(DST, "w", encoding="utf-8").write(s)
print("make_v8_dev14: wrote", DST, "replacements", len(R))
