# B_Flashield Flashield: a Hybrid Key-value Cache that Controls Flash Write Amplification

- **出處**：Assaf Eisenman、Asaf Cidon、Evgenya Pergament、Or Haimovich、Ryan Stutsman、Mohammad Alizadeh、Sachin Katti，NSDI '19（Boston，2019-02-26～28），印刷 pp.65–78。
  連結：https://www.usenix.org/system/files/nsdi19-eisenman.pdf
  頁碼規則：p.X＝PDF 第 X 頁，印刷頁＝p.X＋63（PDF p.1 是封面）。讀了 p.2–3、p.5、p.11–12。
- **寫入時做了什麼決定**：**刻意不在寫入時決定**：
  - 所有物件先進 DRAM。在 DRAM 停留的期間觀察它，證明自己 flash-worthy 才搬上 flash；不夠格的就永遠不寫 flash〔原文 p.3〕。
  - 用 SVM 預測「未來會被讀 ≥n 次、而且不會被更新」（flashiness）。寫上 flash 時，用大塊循序寫入〔原文 p.3〕。
- **用什麼資訊做決定？寫完之後還在不在？（N1）**：用物件在 DRAM 期間累積的讀取次數與更新次數。原文：「when objects first enter the cache, it does not know which objects are good candidates for SSD」〔原文 p.3〕。**寫入當下資訊不足，延後才有**，和 N1 剛好相反。
- **有沒有和延後版、寫穿版、背景版比較？**
  - 對手：RIPQ（全部寫 flash，依讀取次數選插入點），以及 victim cache（DRAM 逐出的全部寫 SSD，也就是「延後全寫」）。
  - Memcachier 全 trace：victim cache 命中率 69.72%、CLWA 4.00；RIPQ 70.59%、2.59〔原文 p.5 表 3〕。60.6% 的寫入是「寫了之後從來沒被讀」〔原文 p.5〕。
  - CLWA 中位數：Flashield 0.54、RIPQ 2.85、victim cache 3.67。Flashield 和 RIPQ 的命中率幾乎一樣，兩者都比 victim cache 低〔原文 p.11〕。
  - **DRAM 占比太低時，物件沒有足夠時間證明自己，命中率下降**〔原文 p.12〕。
- **硬體**：DRAM:SSD 預設 1:7〔原文 p.11〕。
- **模型架構、模態**：不適用（KV store 的小物件，平均 257 B〔原文 p.5〕）。
- **和本研究的關係**：
  - **威脅（最強的一種）**：寫入時資訊差、寫錯的代價又高（flash 壽命）時，「延後觀察再決定」會贏。
  - **KV 的不同**〔判讀〕：位置規則需要的資訊（回來時這段會被重算或被載入），寫入時就確定，之後也不變；KV 也不會被更新。所以 Flashield 的「延後才看得到」不適用於位置規則，S4B 也看得到。但它**適用於「會不會回來」**（H8），那是延後才觀察得到的。
  - **支持 H1**：admission 可以用很小的命中率代價，大幅減少 flash 寫入。這就是「TTFT 差不多、寫得少很多」的 Pareto 型態。
  - **延後觀察的前提**是快的層夠大、停得夠久（p.12）。KV 的條件 2（快的層空間緊）正好削弱這個前提。這是「需要寫入時就預測」的一個理由（CacheLib 在 Facebook 也遇到同樣的事，見 B_CacheLib）。
- **證據等級**：〔原文 p.X〕；和 H1／H8 的對應〔判讀〕。
