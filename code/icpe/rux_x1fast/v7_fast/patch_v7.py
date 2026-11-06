"""把 v6（code/icpe/rux_x1fast/v6_slowcap/src/main.rs，sha256 2eb5a040…）改成 v7（慢層上限的加速，不改語意）。
只做精確的字串取代，每處都要剛好出現一次。用法：python -I patch_v7.py <v6 main.rs> <輸出 main.rs>。
  候選 2：慢層 LRU2 用有序集合（鍵＝(prev 或 −1, last, 進入順序)，session 待在慢層時鍵不變）→ 每次丟 O(log n)，不再全掃。
  候選 3：慢層 PENS 的「成本」快取（只在同一組 session 的 refcount 變動時失效，同 v5 的「下次回來」快取）；掃描還在，每個成員只剩除法與比較。
  候選 5：開上限時也走三個回合結束的快速路徑；被跳過的 fix(b)（那時 fix_inner 什麼都不改）只剩「在慢層、沒有慢層持有者 → 放進 clean_q」
          這個效果，由 sc_note 照原本的呼叫順序補做（前綴：在尾段遞增之後、照 old 的順序；條件式 fix：在同一個位置的 else）。
  驗證模式（X1FAST_SC_CHECK=k）：每次選擇都另外全掃一次比對；PENS 每次都重算成本、和快取比位元；稽核時檢查 LRU2 的鍵沒有變。"""
import hashlib, sys

src, dst = sys.argv[1:3]
s = open(src, encoding="utf-8").read()
assert hashlib.sha256(s.encode()).hexdigest().startswith("2eb5a040"), "輸入不是 v6"


def rep(old, new, n=1):
    global s
    c = s.count(old)
    assert c == n, (c, old[:160])
    s = s.replace(old, new)


# ---- 檔頭
rep("// icpe-rux v6（2026-11-07，code/icpe/rux_x1fast/v6_slowcap）", """// icpe-rux v7（2026-11-07，code/icpe/rux_x1fast/v7_fast）：v6＋慢層上限的加速（不改語意；patch_v7.py）：
//   慢層 LRU2 用有序集合（候選 2）、慢層 PENS 的成本快取（候選 3）、開上限也走快速路徑＋補記 clean_q（候選 5）。
// icpe-rux v6（2026-11-07，code/icpe/rux_x1fast/v6_slowcap）""")
rep("use std::collections::{BinaryHeap, VecDeque};", "use std::collections::{BTreeSet, BinaryHeap, VecDeque};")

# ---- SlowCap 多的欄位
rep("""    check_every: usize, n_ties: i64,                     // 驗證模式（X1FAST_SC_CHECK=k：每 k 個請求稽核一次）
}""", """    check_every: usize, n_ties: i64,                     // 驗證模式（X1FAST_SC_CHECK=k：每 k 個請求稽核一次）
    // v7 候選 2：LRU2 的有序集合。元素＝(prev 或 −1, last, 進入順序, s)；進入順序＝mem 的 move_to_end 先後 → 同分時取 mem 裡第一個（＝Python min）
    l2set: BTreeSet<(i64, i64, u64, usize)>, l2key: Vec<(i64, i64, u64)>, l2seq: u64,
    // v7 候選 3：PENS 的成本快取 (成本, 那時的 epoch＋1)；0＝沒有快取
    pc: Vec<Cell<(f64, u64)>>,
}""")
rep("""            SRule::Gdsf => { self.gver[s] += 1; }
            _ => {}
        }
    }
}""", """            SRule::Gdsf => { self.gver[s] += 1; }
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
}""")

# ---- sc_add：LRU2 進集合
rep("""            glast = self.last[s];
        }
        let sc = self.sc.as_mut().unwrap();
        sc.mem.move_to_end(s);""", """            glast = self.last[s];
        }
        let l2k = (if self.nreq[s] >= 2 { self.prev[s] } else { -1 }, self.last[s]);
        let sc = self.sc.as_mut().unwrap();
        sc.mem.move_to_end(s);""")
rep("""                sc.gheap.push(Reverse((F(gh), glast, v, s)));
            }
            _ => {}""", """                sc.gheap.push(Reverse((F(gh), glast, v, s)));
            }
            SRule::Lru2 => {
                sc.l2_del(s);
                sc.l2seq += 1;
                let q = sc.l2seq;
                sc.l2key[s] = (l2k.0, l2k.1, q);
                sc.l2set.insert((l2k.0, l2k.1, q, s));
            }
            _ => {}""")

# ---- sc_select：LRU2 取集合的第一個（驗證模式另外全掃比對）
rep("""            SRule::Lru2 => {
                let (mut best, mut bk) = (usize::MAX, (0i64, 0i64));
                let mut x = sc.mem.head;
                while x >= 0 {
                    let u = x as usize;
                    let k = (if self.nreq[u] >= 2 { self.prev[u] } else { -1 }, self.last[u]);
                    if best == usize::MAX || k < bk { best = u; bk = k; } else if chk && k == bk { ties += 1; }
                    x = sc.mem.next[u];
                }
                Some(best)
            }""", """            SRule::Lru2 => {
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
            }""")
# ---- sc_select：PENS 成本快取
rep("""                while x >= 0 {
                    let u = x as usize;
                    let bl = self.sbl(u);
                    let n = bl.len();
                    let ex = Self::excl_start(bl, &self.ref_live);
                    let cost = if ex < n { fc[n] - fc[ex] } else { fc[n] - fc[if n == 0 { fc.len() - 1 } else { n - 1 }] };
                    let k = (F(cost / (std::cmp::max(1, cur - self.last[u] + 1) as f64)), self.last[u]);""", """                while x >= 0 {
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
                    let k = (F(cost / (std::cmp::max(1, cur - self.last[u] + 1) as f64)), self.last[u]);""")
# ---- 選到的 session 離開集合
rep("""        sc.mem.remove(s);
        Some(s)""", """        if sc.rule == SRule::Lru2 { sc.l2_del(s); }
        sc.mem.remove(s);
        Some(s)""")

# ---- 稽核：LRU2 的鍵在慢層期間沒有變、集合大小＝成員數
rep("""        assert_eq!(cnt, sc.mem.len, "r={} 結構的長度", r);""", """        assert_eq!(cnt, sc.mem.len, "r={} 結構的長度", r);
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
        }""")

# ---- 建構
rep("""            gver: vec![0; ns], gheap: BinaryHeap::new(), gl: 0.0, g_c: sg, g_cc: sgc, check_every, n_ties: 0,
        })""", """            gver: vec![0; ns], gheap: BinaryHeap::new(), gl: 0.0, g_c: sg, g_cc: sgc, check_every, n_ties: 0,
            l2set: BTreeSet::new(), l2key: vec![(0, 0, 0); ns], l2seq: 0,
            pc: (0..ns).map(|_| Cell::new((0.0, 0))).collect(),
        })""")

# ---- 候選 5：sc_note（快速路徑跳過的 fix 的唯一效果）
rep("""    fn sc_holders(&self, b: usize) -> i64 {
        let n = self.ref_live[b] - self.ref_cpu[b];
        if self.fix_g { n - self.ref_gpu[b] } else { n }
    }""", """    fn sc_holders(&self, b: usize) -> i64 {
        let n = self.ref_live[b] - self.ref_cpu[b];
        if self.fix_g { n - self.ref_gpu[b] } else { n }
    }
    // v7 候選 5：快速路徑跳過 fix(b) 時補做它唯一的效果（fix_inner 這時什麼都不改；驗證模式斷言這一點）
    #[inline]
    fn sc_note(&mut self, b: usize) {
        if self.sc.as_ref().unwrap().check_every > 0 {
            assert!(!(self.ref_live[b] == 0 || (self.ref_cpu[b] == 0 && self.fmt[b] == 1)), "sc_note：塊 {} 的 fix 不是空動作", b);
        }
        if self.on_slow[b] && self.sc_holders(b) == 0 { self.sc.as_mut().unwrap().clean_q.push_back(b as u32); }
    }""")
# step、turn_end_wt：開上限也走快速路徑
rep("""        let fast = self.sc.is_none();           // = _fix_fast（開上限時 False：不跳前綴、每塊都 fix）""",
    """        let fast = true;                        // v7 候選 5：開上限也走快速路徑（v6＝self.sc.is_none()）""", n=2)
rep("""        let mut w = 0i64;
        if (old_state == S_CPU || old_state == S_SLOW) && k0 == 0 {""", """        let mut w = 0i64;
        if k0 > 0 && self.sc.is_some() { for &b in old { self.sc_note(b as usize); } }    // v7 候選 5：跳過的前綴
        if (old_state == S_CPU || old_state == S_SLOW) && k0 == 0 {""", n=2)
rep("""                    if self.ref_live[b] == 0 || (self.ref_cpu[b] == 0 && self.fmt[b] == 1) { w += self.fix(b); }
                }
            } else {""", """                    if self.ref_live[b] == 0 || (self.ref_cpu[b] == 0 && self.fmt[b] == 1) { w += self.fix(b); }
                    else if self.sc.is_some() { self.sc_note(b); }                          // v7 候選 5
                }
            } else {""", n=2)
# turn_end_wb
rep("""        let fast = self.sc.is_none();           // = _fix_fast
""", """        let fast = true;                        // v7 候選 5：開上限也走快速路徑（v6＝self.sc.is_none()）
""")
rep("""        if old_state == S_CPU { self.order_remove(s); }
        if (old_state == S_CPU || old_state == S_SLOW || old_state == S_GPU) && k0 == 0 && !fast {""", """        if old_state == S_CPU { self.order_remove(s); }
        if k0 > 0 && self.sc.is_some() { for &b in old { self.sc_note(b as usize); } }    // v7 候選 5：跳過的前綴
        if (old_state == S_CPU || old_state == S_SLOW || old_state == S_GPU) && k0 == 0 && !fast {""")
rep("""                    if nf != 0 { self.gpu_used -= nf * BF; nf = 0; }
                    w += self.fix(b);
                }""", """                    if nf != 0 { self.gpu_used -= nf * BF; nf = 0; }
                    w += self.fix(b);
                } else if self.sc.is_some() { self.sc_note(b); }                              // v7 候選 5""")

open(dst, "w", encoding="utf-8").write(s)
print("v7 sha256", hashlib.sha256(s.encode()).hexdigest())
