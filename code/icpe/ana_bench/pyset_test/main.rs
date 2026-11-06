include!("pyset.rs");
use std::io::{self, BufRead, Write};
fn main() {
    // 輸入：每行 "a k"／"d k"／"p"（印出目前順序）
    let st = io::stdin(); let mut out = io::BufWriter::new(io::stdout());
    let mut s = PySet::new(1 << 20); let mut inn = vec![false; 1 << 20];
    for l in st.lock().lines() {
        let l = l.unwrap(); let x: Vec<&str> = l.split_whitespace().collect();
        match x[0] {
            "a" => { let k: usize = x[1].parse().unwrap(); if !inn[k] { s.add(k); inn[k] = true; } }
            "d" => { let k: usize = x[1].parse().unwrap(); if inn[k] { s.discard(k); inn[k] = false; } }
            _ => { let o: Vec<String> = s.order().iter().map(|v| v.to_string()).collect(); writeln!(out, "{}", o.join(" ")).unwrap(); }
        }
    }
}
