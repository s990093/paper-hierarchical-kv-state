#!/usr/bin/env python3
"""m16_narr2md.py — 把報告的敘述（docs/stage3_20261019/report/narrative/*.html）轉成任務要求的兩份 md：
results_analysis.md（各實驗解讀）與 stage3_candidates.md（候選排名）。只用標準函式庫。

  python3 code/m16_narr2md.py
"""
import html
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(REPO, "docs/stage3_20261019")
N = os.path.join(D, "report/narrative")
HDR = ("> 這份 md 是報告 PDF（`/mlsteam/data/tiara/stage3/report/stage3_report.pdf`，由 `code/m16_report_pdf.py build` 產生）的文字版，"
       "由 `code/m16_narr2md.py` 從 `report/narrative/*.html` 轉出；圖表、事先寫死的判定、run_id 出處以 PDF 與 `experiment_log.md`、`results.csv` 為準。\n\n")
SECS = [("4.1 K1　KV 結構（含 K1b：跨層共用與 Tucker）", "k1"), ("4.2 I4　搬法", "i4"), ("4.3 I1／I2　多 session 還原", "i1"),
        ("4.4 C1a　能不能猜中第 2 輪的 block", "c1a"), ("4.5 S1　無損壓縮", "s1"), ("4.6 X1　疊加篩選（TF）", "x1"),
        ("4.7 X2＋X2b　生成答對率", "x2"), ("4.8 xKV 事先校準的基底", "xlcal"), ("4.9 X3　還原時間與決策圖", "x3"),
        ("4.10 C1b　選擇性還原", "c1b"), ("4.11 X4　推到極限：哪樣配、配比、極限", "x4"), ("4.12 I3　agent 暫停（模擬；I3g 實測見 4.14）", "i3"),
        ("4.13 vLLM：V0／F2／F1", "vllm"), ("4.14 10/22 補跑", "supp")]


def h2md(s: str) -> str:
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S)

    def table(m):
        rows = re.findall(r"<tr>(.*?)</tr>", m.group(0), flags=re.S)
        out = []
        for i, r in enumerate(rows):
            cells = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", c)).strip().replace("|", "／")
                     for c in re.findall(r"<t[hd]>(.*?)</t[hd]>", r, flags=re.S)]
            out.append("| " + " | ".join(cells) + " |")
            if i == 0:
                out.append("|" + "|".join([":--"] * len(cells)) + "|")
        return "\n" + "\n".join(out) + "\n"
    s = re.sub(r"<table>.*?</table>", table, s, flags=re.S)
    s = re.sub(r"<h4>(.*?)</h4>", r"\n#### \1\n", s)
    s = re.sub(r"<b>(.*?)</b>", r"**\1**", s, flags=re.S)
    s = re.sub(r"<code>(.*?)</code>", r"`\1`", s)
    s = re.sub(r"<sub>(.*?)</sub>", r"_\1", s)
    s = re.sub(r"<sup>(.*?)</sup>", r"^\1", s)
    s = re.sub(r"<ol>(.*?)</ol>", lambda m: "\n" + "\n".join(f"{i + 1}. " + re.sub(r"\s+", " ", x).strip()
               for i, x in enumerate(re.findall(r"<li>(.*?)</li>", m.group(1), flags=re.S))) + "\n", s, flags=re.S)
    s = re.sub(r"<li>(.*?)</li>", lambda m: "- " + re.sub(r"\s+", " ", m.group(1)).strip() + "\n", s, flags=re.S)
    s = re.sub(r"</?ul>", "\n", s)
    s = re.sub(r"<p>(.*?)</p>", lambda m: "\n" + re.sub(r"\s+", " ", m.group(1)).strip() + "\n", s, flags=re.S)
    s = html.unescape(re.sub(r"<[^>]+>", "", s))
    return re.sub(r"\n{3,}", "\n\n", s).strip() + "\n"


def rd(name):
    p = os.path.join(N, name + ".html")
    return h2md(open(p, encoding="utf-8").read()) if os.path.exists(p) else "（NOT_WRITTEN：這一節的敘述還沒寫）\n"


def main():
    a = ["# 第三階段前置探索：結果分析\n", HDR, "## 摘要\n", rd("summary"), "\n## 背景與問題\n", rd("background"),
         "\n## 文獻與查新\n", rd("literature"), "\n## 各實驗的解讀\n"]
    for t, f in SECS:
        a += [f"\n### {t}\n", rd(f)]
    a += ["\n## 綜合\n", rd("synthesis"), "\n## 限制與誠實聲明\n", rd("limitations")]
    open(os.path.join(D, "results_analysis.md"), "w").write("\n".join(a))
    c = ["# 第三階段候選題目（排名）\n", HDR, rd("candidates"), "\n## 收掉的方向（負面結果）\n",
         "依位置決定（第一、二階段）、多 session 聯合決定（I1）、agent 的 κ 策略（I3，模擬）、逐 token 跨層共用（K1）、token 合併（X1／X2）、"
         "依重要性丟 block（X1／X2／X4d）、用 Quest 摘要做選擇性還原（C1b）、只在單一格式之間依連線選（X3 STOP-0）、F1 的「非當然不划算區域」（沒找到）。"
         "詳見 `results_analysis.md` 與 PDF 第 5 章；各方向的前人對照見 `lit/L7a–L7d`。\n"]
    open(os.path.join(D, "stage3_candidates.md"), "w").write("\n".join(c))
    print(os.path.getsize(os.path.join(D, "results_analysis.md")), os.path.getsize(os.path.join(D, "stage3_candidates.md")))


if __name__ == "__main__":
    main()
