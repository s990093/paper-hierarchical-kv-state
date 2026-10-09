# B_Kangaroo Kangaroo: Caching Billions of Tiny Objects on Flash

- **出處**：Sara McAllister、Benjamin Berg、Julian Tutuncu-Macias、Juncheng Yang、Sathya Gunasekar、Jimmy Lu、Daniel S. Berger、Nathan Beckmann、Gregory R. Ganger，SOSP '21（Virtual，2021-10-26～29），印刷 pp.243–262，DOI 10.1145/3477132.3483568。
  連結：https://www.pdl.cmu.edu/PDL-FTP/NVM/McAllister-SOSP21.pdf
  頁碼規則：p.X＝PDF 第 X 頁，印刷頁＝p.X＋242。讀了 p.1–2、p.4、p.6–7、p.12。
- **寫入時做了什麼決定**：分成兩段，決定點故意往後放：
  1. **pre-flash admission**：物件從 DRAM 被逐出時，以機率 p 決定要不要放進 KLog〔原文 p.6〕；
  2. 先寫進**小的 log（KLog，約 flash 的 5%）**，之後才搬到 set-associative 的 KSet。**只有同一個 set 在 KLog 裡湊到 ≥n 個物件時才搬**（threshold admission）；在 KLog 期間被命中的物件，重新放回 log 的開頭〔原文 p.2、p.7〕。
  - 「since Kangaroo is a cache, not a key-value store, it is free to drop objects instead of admitting them」〔原文 p.2〕。
- **用什麼資訊做決定？寫完之後還在不在？（N1）**：用的是 KLog 裡的 hash 碰撞數，只有暫存一段時間之後才知道。這也和 N1 相反：延後讓寫入可以攤提。
- **有沒有和延後版、寫穿版、背景版比較？**
  - 門檻 n＝2 時只收 44.4% 的物件，寫入率卻只有 n＝1 的 22.8%。省下的寫入比例大於被拒絕的物件比例，純機率式 admission 做不到這點〔原文 p.7〕。
  - 拆解（對照「全部收的 set-associative cache」）：寫入率 −67%，miss −2%。各項的寫入／miss 變化：pre-flash admission −8.2%／+1.9%；KLog −42.6%／<0.05%；KSet threshold −32.0%／+6.9%；RRIParoo −8.3%／−8.4%〔原文 p.12〕。
  - Facebook 生產 trace，1.9 TB drive、16 GB DRAM、每天 3 次 device-writes：miss ratio 從 0.29 降到 0.20〔原文 p.1–2〕。
  - CacheLib 的 SOC 為了壓低 DLWA，生產環境超過一半的 flash 空著，並用 pre-flash admission 拒絕一部分物件〔原文 p.4〕。
- **硬體**：見上，1.9 TB flash、16 GB DRAM、3 DWPD。
- **模型架構、模態**：不適用（約 100 B 的小物件）。
- **和本研究的關係**：
  - **威脅**：「延後才能批次寫」是延後版的一個優勢。
  - **但 KV 不適用**〔判讀〕：KV chunk 本來就是 MiB 級的大塊，沒有小物件寫入放大的問題，延後攢批次沒有好處。
  - 「cache 可以直接丟」和 H0 的發現一致：五個真實系統在 CPU 滿時預設也是丟掉。
  - **支持 H1、H8**：admission 能大幅減少寫入，miss 只增加一點。
- **證據等級**：〔原文 p.X〕；KV 不適用的判斷〔判讀〕。
