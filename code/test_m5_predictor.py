#!/usr/bin/env python3
"""`m5_predictor.py` 的單元測試。**用手算的例子，不是快照。**

跑：
    PYTHONPATH=/ssd7/hungwei/paper-hkv/pylibs python code/test_m5_predictor.py

這裡測的六件事，每一件都對應一個「錯了但看起來完全正常」的失敗模式：

| 測什麼 | 錯了會怎樣 |
|---|---|
| 特徵取自更新前的狀態 | `delta_1` 變成 0 = 把答案洩漏進特徵，準確率漂亮但無效 |
| EDC 的遞迴式 | 半衰期算錯只會讓特徵變弱，沒有任何訊號 |
| 兩個標記機制 | 少了機制 (b) 就沒有負樣本，模型永遠說「留著」 |
| 負樣本的唯一來源 | 同上，且從指標上看不出來 |
| embargo | 訓練集含測試期才確定的標籤，分數系統性偏高 |
| p* 的公式 | 門檻錯置一個數量級，正是本文要指出的那個錯誤 |
"""
from __future__ import annotations
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, os.environ.get("PAPER_HKV_PYLIBS",
                                  "/ssd7/hungwei/paper-hkv/pylibs"))

from m4_oracle import BLOCK, CostModel                        # noqa: E402
from m5_predictor import (EDC_HALFLIVES, Labeler, Replay,     # noqa: E402
                          build_samples, drop_cost, ece, feature_names,
                          isotonic_fit, isotonic_predict, p_star, time_split)

FAILED = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'✅' if ok else '🔴'} {name}" + (f"　{detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def collect(trace, k=4):
    rep = Replay(trace, k_deltas=k)
    rows, metas = [], []
    rep.run(lambda X, M, t0: (rows.append(X.copy()), metas.append(M.copy())))
    return np.vstack(rows), np.vstack(metas), feature_names(k)


def test_features_are_pre_update():
    """特徵必須是「決策當下」看得到的：第一次存取沒有任何歷史。"""
    X, M, names = collect([[1, 2], [1]])
    i_nacc = names.index("n_acc")
    i_d1 = names.index("log_delta_1")
    i_edc0 = names.index("edc_0")
    check("第一次存取 n_acc == 0", X[0, i_nacc] == 0)
    check("第一次存取 delta_1 是 NaN（不是 0）", math.isnan(X[0, i_d1]),
          "0 代表『剛剛才被用過』，是完全相反的訊號")
    check("第一次存取 EDC == 0", X[0, i_edc0] == 0.0)
    # b=1 在 t=0 與 t=2 被存取，間隔 2
    check("第二次存取的 delta_1 == log1p(2)",
          abs(X[2, i_d1] - math.log1p(2)) < 1e-6,
          f"{X[2, i_d1]:.6f} vs {math.log1p(2):.6f}")
    check("第二次存取 n_acc == 1", X[2, i_nacc] == 1)


def test_edc_recurrence():
    """EDC_i <- 1 + EDC_i * 2^(-Δ1/2^(9+i))（main.tex 演算法 1 第 2 行）。"""
    X, M, names = collect([[7], [7], [7]])
    i0 = names.index("edc_0")
    # t=0 第一次：0；更新後 = 1
    # t=1 第二次：看到 1；更新後 = 1 + 1*2^(-1/512)
    # t=2 第三次：看到 1 + 2^(-1/512)
    want = 1.0 + 2.0 ** (-1.0 / 512.0)
    check("第三次存取看到的 edc_0", abs(X[2, i0] - want) < 1e-5,
          f"{X[2, i0]:.6f} vs {want:.6f}")
    i9 = names.index("edc_9")
    want9 = 1.0 + 2.0 ** (-1.0 / EDC_HALFLIVES[9])
    check("最長半衰期那一格", abs(X[2, i9] - want9) < 1e-5)


def test_two_labeling_mechanisms():
    """機制 (a) 下次存取、機制 (b) 視窗到期。手算的例子。"""
    # t:      0   1   2   3   4   5   6   7   8   9  10
    # block:  1   2   3   1   4   5   6   7   8   9  10
    trace = [[1, 2], [3], [1], [4], [5], [6], [7], [8], [9], [10]]
    W = 4
    with tempfile.TemporaryDirectory() as d:
        rep = Replay(trace, k_deltas=4)
        st = build_samples(rep, window=W, sample_rate=1.0, max_per_block=0,
                           out_dir=Path(d), seed=0)
        lab = np.load(Path(d) / "labels.npz")
        tau, cen, sids = lab["tau"], lab["censored"], lab["sample_ids"]
    check("11 次存取全部收成樣本（尾巴另計）",
          st["n_labeled_by_next_access"] + st["n_labeled_by_window"] == 11)
    check("機制 (a) 標了 1 個（block 1 在 t=3 被重用）",
          st["n_labeled_by_next_access"] == 1,
          f"{st['n_labeled_by_next_access']}")
    check("機制 (b) 標了 10 個", st["n_labeled_by_window"] == 10,
          f"{st['n_labeled_by_window']}")
    check("尾巴 W 個視窗不完整的樣本被丟掉", st["n_dropped_tail"] == 4,
          f"{st['n_dropped_tail']}（t + W < 11 的才留，即 t ≤ 6）")
    check("留下 7 個樣本", st["n_samples"] == 7, f"{st['n_samples']}")
    check("t=0 的樣本 tau == 3（下次存取在 t=3）", tau[0] == 3, f"{tau[0]}")
    check("未被重用者 tau == W+1", set(tau[1:]) == {float(W + 1)},
          f"{set(tau[1:])}")
    check("censored 恰好是機制 (b) 標的那些",
          int(cen.sum()) == 6 and cen[0] == 0, f"{cen}")


def test_negatives_only_from_window():
    """把視窗關掉（W 大於整條 trace）就一個負樣本都沒有——這正是 §5.2 的論證。"""
    trace = [[1, 2], [3], [1], [4], [5], [6], [7], [8], [9], [10]]
    with tempfile.TemporaryDirectory() as d:
        rep = Replay(trace, k_deltas=4)
        st = build_samples(rep, window=10**9, sample_rate=1.0, max_per_block=0,
                           out_dir=Path(d), seed=0)
    # W 無限大時，結尾的強制清算仍會標記；但「下次存取」機制只標到 1 個。
    check("只有 1 個樣本能靠『下次被存取』拿到標籤",
          st["n_labeled_by_next_access"] == 1,
          "其餘 10 個若沒有滑動視窗就永遠等不到標籤，"
          "訓練集裡就一個負樣本都沒有")


def test_isotonic():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    xk, yk = isotonic_fit(x, y)
    check("PAVA 輸出單調不減", bool(np.all(np.diff(yk) >= -1e-12)), str(yk))
    # 手算：0 ≤ 1 不違反；加入第三點的 0 才違反，與前一格合併成 0.5。
    # 結果 [0, 0.5, 0.5, 1, 1]（**不是** 把前三點一起平均成 1/3）
    check("PAVA 只合併真正違反的相鄰格",
          np.allclose(yk, [0, 0.5, 0.5, 1, 1]), str(yk))
    p = isotonic_predict(xk, yk, np.array([0.0, 3.5, 9.0]))
    check("節點之間線性內插、兩端取端點值",
          np.allclose(p, [0.0, 0.75, 1.0]), str(p))


def test_p_star():
    """p* 的兩個可驗證後果（見 m5_predictor 檔頭）。"""
    cm = CostModel(gpu=0.0, cpu=0.588, ssd=5.536, recompute_base=4.008,
                   recompute_slope_per_token=0.000210)
    p0 = float(p_star(cm, np.array([0.0]), "cpu")[0])
    check("p*_cpu(0) == C_cpu / C_drop(0)", abs(p0 - 0.588 / 4.008) < 1e-9,
          f"{p0:.4f}（≪ 0.5，正是 §5.3 說的錯置一個數量級）")
    # p*_ssd == 1 的位置就是成本模型的交叉點 P*
    cross = (cm.ssd - cm.recompute_base) / cm.recompute_slope_per_token
    pc = float(p_star(cm, np.array([cross]), "ssd")[0])
    check("p*_ssd 在交叉點 P* 恰為 1", abs(pc - 1.0) < 1e-6,
          f"P* = {cross:,.0f} token、p*_ssd = {pc:.6f}")
    check("P* 以下 SSD 階不可選（p*_ssd > 1）",
          float(p_star(cm, np.array([cross - 1000]), "ssd")[0]) > 1.0)


def test_embargo():
    """embargo：訓練集不得含有「標籤在測試期才確定」的樣本。"""
    n = 1000
    sids = np.arange(n) * 10
    rng = np.random.default_rng(0)
    tau = rng.integers(1, 400, size=n).astype(float)
    cen = (rng.random(n) < 0.5).astype(np.int8)
    W = 300
    tr, cal, te, info = time_split(sids, tau, cen, W, 0.7, 0.15, True)
    t_split = info["t_split"]
    settle = sids + np.where(cen == 1, W, tau)
    check("embargo 後沒有任何訓練樣本的標籤晚於切分點",
          bool(np.all(settle[np.concatenate([tr, cal])] <= t_split)),
          f"丟掉 {info['embargo_dropped']} 個")
    tr2, cal2, te2, info2 = time_split(sids, tau, cen, W, 0.7, 0.15, False)
    check("關掉 embargo 就會有洩漏的樣本",
          bool(np.any(settle[np.concatenate([tr2, cal2])] > t_split)),
          "這證明這道檢查確實在擋東西")


def test_ece():
    p = np.array([0.1, 0.1, 0.9, 0.9])
    y = np.array([0, 0, 1, 1])
    e, rows = ece(p, y, bins=10)
    check("完美校準的 ECE ≈ 0.1", abs(e - 0.1) < 1e-9, f"{e:.4f}")


if __name__ == "__main__":
    for fn in (test_features_are_pre_update, test_edc_recurrence,
               test_two_labeling_mechanisms, test_negatives_only_from_window,
               test_isotonic, test_p_star, test_embargo, test_ece):
        print(f"\n{fn.__name__}　{fn.__doc__.splitlines()[0] if fn.__doc__ else ''}")
        fn()
    print()
    if FAILED:
        print(f"🔴 {len(FAILED)} 項失敗：{FAILED}")
        raise SystemExit(1)
    print("✅ 全部通過")
