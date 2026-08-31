"""m4_oracle 的回歸測試：改模擬器之後跑這支，確認
(1) 線上策略的行為完全不變、(2) Oracle 只會變好、不會變差。

用法：python code/test_m4_regression.py <放 m4_old.py 的目錄>
m4_old.py 是修改前的快照，見 /ssd7/hungwei/paper-hkv/runs/*.bak-*
"""
sys.path.insert(0, "code")
sys.path.insert(0, sys.argv[1])
import m4_oracle as new
spec = importlib.util.spec_from_file_location("m4_old", sys.argv[1] + "/m4_old.py")
old = importlib.util.module_from_spec(spec); sys.modules["m4_old"] = old
spec.loader.exec_module(old)

cm = new.load_cost_model("sata")
bad = 0
gains = []
for seed in (1, 2, 3):
    for ndocs, nreq, gpu_b in ((40, 300, 200), (12, 200, 60), (80, 500, 300)):
        tr = new.zipf_trace(ndocs, 32, nreq, 0.9, seed)
        sn = new.Sim(cm, gpu_b, 400, 10**9)
        so = old.Sim(cm, gpu_b, 400, 10**9)
        for name, args in (("full_gpu", ("lru", False, False)),
                           ("cpu_lru", ("lru", True, False)),
                           ("cpu_arc", ("arc", True, False)),
                           ("tier_fs", ("lru", True, True))):
            a = sn.run_online(tr, *args, prefix_semantics=False, prefetch=False)
            b = so.run_online(tr, *args)
            if abs(a["total_ms"] - b["total_ms"]) > 1e-6 or a["hits"] != b["hits"]:
                bad += 1
                print(f"✗ {name} seed={seed} docs={ndocs} gpu={gpu_b}")
                print(f"   new {a['total_ms']:.2f} {a['hits']}")
                print(f"   old {b['total_ms']:.2f} {b['hits']}")
        # Oracle 是**刻意**改強的（成本感知的目的地選擇 + CPU 滿載時交換）。
        # 正確的斷言不是「一模一樣」，而是：
        #   1. 重算次數不變（強制未命中下限是策略無關的常數）
        #   2. 新版絕不比舊版差（Oracle 是上界，只能往上）
        for tag, kw in (("cascade", {"dest": "cascade"}),
                        ("cost-aware", {"dest": "cost-aware"})):
            a = sn.run_oracle(tr, True, True, prefix_semantics=False,
                              prefetch=False, **kw)
            b = so.run_oracle(tr, True, True)
            # 重算次數的正確斷言：
            #   cascade   —— 從不主動丟棄，應等於強制未命中下限
            #   cost-aware —— 會在「重算比存放便宜」時主動丟，所以 ≥ 下限。
            #                 這不是 bug，是論文 DROP 動作被啟用的徵兆。
            floor = len({x for r in tr for x in r})
            if tag == "cascade" and a["hits"]["drop"] != floor:
                bad += 1
                print(f"✗ oracle/cascade 重算 {a['hits']['drop']} != 下限 {floor}")
            if a["hits"]["drop"] < floor:
                bad += 1
                print(f"✗ oracle/{tag} 重算 {a['hits']['drop']} < 下限 {floor}"
                      f"（不可能：每個 block 至少要算一次）")
            if a["total_ms"] > b["total_ms"] + 1e-6:
                bad += 1
                print(f"✗ oracle/{tag} 變差了 seed={seed} docs={ndocs} gpu={gpu_b}:"
                      f" {a['total_ms']:.2f} > {b['total_ms']:.2f}")
            gains.append((tag, 100 * (b["total_ms"] - a["total_ms"]) / b["total_ms"]))
print("回歸：線上策略完全相同、Oracle 只變好 ✓" if bad == 0
      else f"回歸：{bad} 個問題 ✗")
for tag in ("cascade", "cost-aware"):
    g = [v for t, v in gains if t == tag]
    if g:
        print(f"  Oracle/{tag:11s} 比舊版快 {min(g):.1f}–{max(g):.1f}%"
              f"（平均 {sum(g)/len(g):.1f}%）")
