# 回覆老師：短版（Discord 一則可貼；icpe-adv，2026-11-08）

> 詳細版是 `REPLY_TEACHER.md`。數字標「明早確認」的，等 icpe-main 的版本 C 修正版總判定。

## 先講：和 Cake、CacheWise §6.4 的關係怎麼說

**不要說「比 Cake 強」**：
- Cake 是我們對手的一部分，我們沒有改進 Cake。
- 能說的是：我們的對手＝Cake 加 INT4；H200 實測在慢層上，比次佳的還原方法快 11.0–40.7%〔實測〕。
- 它的用途是讓「還原」變便宜，不是我們的貢獻。

**不要說「前作的對手不夠強」**：
- 老師第 4 點已經點出：PBKV、KVFlow 都有 CPU 備份。

**建議的說法：接著 CacheWise §6.4 往下做**
- CacheWise §6.4 已觀察到：KV 從 CPU 搬回比重算便宜時，預測型淘汰的改善變小（N＝30、對 vLLM、對話完成時間約 1.19×）。
- 我們把這個觀察系統化，做了三件事：
  1. 把還原再推得更便宜：多層＋INT4＋Cake。
  2. 在 6 份紀錄上量完美預知還剩多少。
  3. 拆出剩下的落在哪一個決策上。
- 我們手上有同方向的數字（舊模擬器、FreeInference、CPU 32 GiB、對 LRU）：只重算時，完美預知快 15.9%；加慢層 3.4%；再加 Cake 2.6%〔模擬〕。這是 CacheWise §6.4 的同一個現象，多看了幾階。
- **一句話（中文）**：CacheWise §6.4 看到「搬回便宜時，預測的改善變小」；我們把還原一階一階變便宜，量出剩下的空間還有多少、剩在哪個決策。
- **一句話（論文英文）**：*CacheWise observes that its gains shrink when evicted KV is reloaded from CPU memory instead of recomputed (§6.4). We extend this observation: we make restoration progressively cheaper (multi-tier storage, INT4, Cake) and measure, across six traces, how much a clairvoyant policy still saves and in which decision.*

## Discord 短版（照貼）

```
老師好，謝謝老師的意見。我先整理重點（詳細版另附）：

1. 結論改了：之前「預測空間有限」是只看兩份紀錄、舊模擬器（沒有 GPU 層）的結果。之後擴大到 6 份紀錄、加了 GPU 層和更強的對手（TTL、前綴快取），結果變成「看負載」：一般 agent 服務（FreeInference）就算預測完全準確，首字延遲也少不到 1%；工具類 agent 在記憶體吃緊時最多少約 62%（數字明早確認），主要在決定「誰留在 GPU／CPU」。所以論文照老師說的，寫成「哪裡不用預測、哪裡仍然值得、為什麼」。

2. −4.4%：我的 oracle 是「知道未來，但仍照 LRU 騰空間」的預知版，不是在所有做法裡找最快。負值來自 6 週裡的一週：它為了預取把別的對話擠到慢層，降級次數是 LRU 的 2.4 倍。論文會寫成「每條規則的預知版」，並列出比對手慢的格子。

3. 前作：PBKV、KVFlow 都已經有 CPU 備份和預取，我之前說「對手沒命中就重算」不對，收回。CacheWise §6.4 已經看到「搬回比重算便宜時，改善空間變小」；我們是接著往下做：把還原一階一階變便宜（多層＋INT4＋Cake），在 6 份紀錄上量剩下多少、剩在哪個決策。沒有做同設定的比較，所以不寫贏過它們。

4. 打字提示是 SYMPHONY 已有的做法，我改成量「它能拿走多少」，並寫明提前時間是用字數和打字速度合成的。「八成」是 1/8 縮小重播裡，完美預取改善量的 0.81，不是首字延遲少八成，也不能套到 50.5%。51.56 WPM 在程式裡是 1 word＝5 字元，和原文一致。工具線索：只有 TraceLab 記了工具呼叫，那個線索拿不到；其他紀錄沒測。所有紀錄都沒存工具「執行中」的進度（Ask the Tool 是重新執行才讀到），所以只寫「這些紀錄沒提供」，不說未來猜不到。

5. 1/8 重播只縮了對話數和 CPU 容量，頻寬沒縮、排隊也不同（縮小後上限本來就掉一半），論文會照實寫。

摘要會照這個方向改，11/10 登記。
```

## 出處

- CacheWise §6.4 原句與條件：`W8_source_check.md` 第 2 節；2026-11-08 重讀原文（`REPLY_TEACHER.md` 第 5 節）。
- 15.9／3.4／2.6%：`STORY.md` 第 3 章（舊 T4，登記檔 `docs/h200/icpe/T4_hit_vs_ttft_bridge.md`）。
- 11.0–40.7%：`ADVISOR.md` 第 5、8 節（PAPER.md v4.2 的實測值）。
- 其他數字：`REPLY_TEACHER.md` 第 8 節。
