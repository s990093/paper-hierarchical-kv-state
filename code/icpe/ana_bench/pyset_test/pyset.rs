// CPython 3.12 Objects/setobject.c 的雜湊表（鍵＝session id，hash(int)＝int）：set_add_entry（重用探測路上最後一個 dummy）、
// set_insert_clean、set_table_resize（fill*5 >= mask*3 時 → used>50000 ? used*2 : used*4）、set_discard_entry（留 dummy）、走訪＝表的索引順序。
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

