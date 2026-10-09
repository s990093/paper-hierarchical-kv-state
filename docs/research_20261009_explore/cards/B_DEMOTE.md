# B_DEMOTE My cache or yours? Making storage more exclusive

- **出處**：Theodore M. Wong（CMU）、John Wilkes（HP Labs），USENIX ATC 2002（Monterey，2002-06-10～15），pp.161–175（頁碼取自 Gill FAST'08 的參考文獻 [33]）。
  連結：https://www.usenix.org/legacy/publications/library/proceedings/usenix02/full_papers/wong/wong.pdf
  頁碼規則：p.X＝PDF 第 X 頁（PDF p.1 是 USENIX 封面）。讀了 p.2–8、p.10、p.13–15。
- **寫入時做了什麼決定**：**沒有**。DEMOTE 正好是「延後版」：client 要逐出一個 clean block 時，才把它送回 array cache；array 已經有這塊、或當下騰不出空間時，就不傳資料〔原文 p.3〕。array 端只決定插入點：送給 client 的塊放在 LRU 最快被丟的那一端，demote 回來的塊放在最晚被丟的那一端〔原文 p.3–4〕。多個 client 共用資料時，改用 ghost cache 自適應決定插入點（DEMOTE-ADAPT）〔原文 p.11〕。
- **用什麼資訊做決定？寫完之後還在不在？（N1）**：用的是「這塊要離開上層了」這個事件，只有逐出時才有。讀入當下 array 也知道「這塊剛送上去」，所以 DEMOTE 把它放在最先被丟的位置。沒有「寫完就消失」的資訊。
- **有沒有和延後版、寫穿版、背景版比較？**（論文本身就是延後版，對照組是 inclusive 的 NONE-LRU）
  - 成本模型：一次 demote 約等於一次 array hit，所以近似成「array hit 的延遲加倍」〔原文 p.4〕。
  - 單一 client 的合成負載：DEMOTE 對 NONE-LRU 快 1.7–7.5 倍〔原文 p.6 表 3、p.8〕。
  - **SAN 頻寬低於約 20–30 Mbit/s 時，NONE-LRU 反而贏 DEMOTE**〔原文 p.7–8〕。高於這個門檻，好處「大致不受 SAN 頻寬影響」〔原文 p.8〕。
  - TPC-H、2 GB array cache：DEMOTE **慢 3%**（0.97×），因為付了 demote 成本，命中率卻沒增加〔原文 p.10〕。
  - 多個 client 大量共用資料（HTTPD）：DEMOTE 只有 **0.55×**（變慢）；DEMOTE-ADAPT-EXP 是 1.18×〔原文 p.13〕。
- **硬體**：模擬（Pantheon、fscachesim）。RAID5 array，1 Gbit/s FibreChannel SAN，Ta＝0.2 ms，Td≈10 ms；合成實驗的 cache 是 64 MB〔原文 p.5〕。
- **模型架構、模態**：不適用（儲存區塊）。
- **和本研究的關係**：
  - DEMOTE 對應本研究的 **S4（滿了才搬，被擠出時才寫下一層）**。它和 KV 一樣只 demote **clean** block。
  - **支持 N2 的形式**：延後搬多付的那一次傳輸，只有在傳輸資源稀缺時才致命（20–30 Mbit/s 門檻）；另外，demote 之後沒人讀，就是純成本（TPC-H 0.97×）。對應 H1（寫入量）、H12（多層要搬兩次）。
  - **威脅**：頻寬夠時，延後搬本身就有大好處，而且對頻寬不敏感。這和 08 的「S4B 追平 S5L」一致。
- **證據等級**：上述數字都是〔原文 p.X〕。「對應 S4」「對應 H1／H12」是〔判讀〕。
