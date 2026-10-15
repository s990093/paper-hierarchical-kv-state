"""m10_verify_summary.py — m10_verify.sh 的小幫手：環境資訊、各項判定、總表

子命令（都只讀檔、印結果；判定不過就以 1 結束）：
  env <out.json>          印 torch／CUDA／HIP 版本、GPU 名稱、記憶體；最後一行印 MEM_GIB=<數字> 給 bash 用
  pick-n <llama|qwen> <mem_gib>   mixed 測試在這張卡上能用的最大 block 數（64／32／16／8）
  corr <correctness.json> 正確性：逐位元的三項（HF 跑兩次、重算兩次、主機往返）都要相同；對 HF 的 logit 差照實印
  mixed <restore.json>    低精度載入：all_pass，並印每個配置 × 模式的來源
  regress <dir>           每個 process 一個 Restorer 的新舊比較（dir 裡 new*.json、old*.json），
                          判準＝results/m10_pilot_mi300x/regress_xproc/CRITERION.md：KV 全部逐位元相同；
                          每格 t_restore 中位數 新/舊 ∈ [0.95, 1.05]；會合點集合相同
  report <out_dir>        讀 steps.tsv，印總表、寫 verify_summary.json；有 FAIL 就以 1 結束

這是「能不能跑、算得對不對」的驗證，不是實驗：輸出不寫進 results/。
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st
import sys

# 估記憶體用（GiB）：權重、每個 block（512 token）的 KV。mixed 會同時放 KV 緩衝、參考 KV、預期的解碼結果三份
_W = {"llama": 15.0, "qwen": 14.2}
_BLK = {"llama": 0.0625, "qwen": 28 / 1024}
_HEADROOM = 3.0


def env(out):
    import torch
    info = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
            "hip": torch.version.hip, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(device=p.name, mem_gib=round(p.total_memory / 2 ** 30, 2), n_visible=torch.cuda.device_count(),
                    capability=".".join(map(str, torch.cuda.get_device_capability(0))))
    info["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    info["HIP_VISIBLE_DEVICES"] = os.environ.get("HIP_VISIBLE_DEVICES")
    json.dump(info, open(out, "w"), indent=1)
    print(json.dumps(info, indent=1))
    print(f"MEM_GIB={info.get('mem_gib', 0)}")
    return 0 if info["cuda_available"] else 1


def pick_n(model, mem):
    mem = float(mem)
    for n in (64, 32, 16, 8):
        if _W[model] + 3 * n * _BLK[model] + _HEADROOM <= 0.9 * mem:
            print(n)
            return 0
    print(8)
    return 0


def corr(path):
    d = json.load(open(path))
    need = ["hf_twice_bitwise_equal", "recompute_same_chunking_bitwise_equal", "host_roundtrip_bitwise_equal"]
    ok = all(d.get(k) is True for k in need)
    for k in need:
        print(f"  {k}: {d.get(k)}")
    print(f"  對 HF：logit 最大差 {d.get('own_vs_hf_maxabs_logit')}、argmax 相同 {d.get('own_vs_hf_argmax_equal')}、"
          f"top-5 重疊 {d.get('own_vs_hf_top5_overlap')}（照實報告，不判定）")
    print(f"  L={d.get('L')} chunk_bytes={d.get('chunk_bytes')} torch={d.get('torch')}")
    print("PASS" if ok else "FAIL：逐位元的項目沒有全部相同（03 §R1：不過 → 全部停）")
    return 0 if ok else 1


def mixed(path):
    d = json.load(open(path))
    print(f"  model={os.path.basename(str(d.get('model')))} fmt={d.get('fmt')} n={d.get('n')} "
          f"lp_block_bytes={d.get('lp_block_bytes')}/{d.get('chunk_bytes')}")
    for r in d.get("runs", []):
        flags = [k for k in ("kv_bitwise_ok", "every_block_once", "bytes_read_ok", "dequant_timed", "counts_ok")
                 if not r.get(k)]
        print(f"  {r['config']:4s} {r['mode']:12s} meet={r['meet']:3d} {r['sources']} "
              + ("OK" if not flags else "壞：" + ",".join(flags) + f" bad={r.get('bad_blocks')}"))
    ok = bool(d.get("all_pass"))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def regress(dirpath):
    rows, kv, runs = [], True, []
    for f in sorted(glob.glob(os.path.join(dirpath, "*.json"))):
        name = os.path.basename(f)[:-5]
        if not (name.startswith("new") or name.startswith("old")):
            continue
        d = json.load(open(f))
        kv &= bool(d.get("kv_bitwise_all"))
        runs.append(d.get("run_id", name))
        for r in d["rows"]:
            r["proc"] = name
            rows.append(r)
    impls = {r["impl"] for r in rows}
    if not {"new", "old"} <= impls:
        print(f"FAIL：缺 new 或 old 的結果（找到 {sorted(impls)}）")
        return 1
    ok = kv
    out = {"kv_bitwise_all": kv, "cells": [], "processes": runs,
           "criterion": "KV 全部逐位元相同；每格 t_restore 中位數 新/舊 ∈ [0.95, 1.05]；會合點集合相同"}
    print(f"  KV 全部逐位元相同：{kv}")
    for t, m in sorted({(r["tier"], r["mode"]) for r in rows}):
        sel = {i: [r for r in rows if r["tier"] == t and r["mode"] == m and r["impl"] == i] for i in ("new", "old")}
        med = {i: st.median(r["t_restore"] for r in v) for i, v in sel.items()}
        meets = {i: sorted({r["meet"] for r in v}) for i, v in sel.items()}
        ratio = med["new"] / med["old"]
        good = 0.95 <= ratio <= 1.05 and meets["new"] == meets["old"]
        ok &= good
        out["cells"].append(dict(tier=t, mode=m, median_new=med["new"], median_old=med["old"], ratio=ratio,
                                 meets_new=meets["new"], meets_old=meets["old"], pass_=good))
        print(f"  {t:5s} {m:12s} 新 {med['new']:.4f} 舊 {med['old']:.4f} 比 {ratio:.4f} "
              f"會合 新{meets['new']} 舊{meets['old']} {'PASS' if good else 'FAIL'}")
    out["all_pass"] = ok
    json.dump(out, open(os.path.join(dirpath, "regress_summary.json"), "w"), indent=1, ensure_ascii=False)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def report(outdir):
    p = os.path.join(outdir, "steps.tsv")
    steps = []
    for line in open(p, encoding="utf-8"):
        sid, name, status, rc, secs, log, note = (line.rstrip("\n").split("\t") + [""] * 7)[:7]
        steps.append(dict(id=sid, name=name, status=status, rc=rc, secs=secs, log=log, note=note))
    envj = os.path.join(outdir, "env.json")
    info = json.load(open(envj)) if os.path.exists(envj) else {}
    w = max([len(s["name"]) for s in steps] + [10])
    print("\n══════════ m10_verify 總表 ══════════")
    print(f"機器：{info.get('device', '?')}（{info.get('mem_gib', '?')} GiB）torch {info.get('torch', '?')} "
          f"CUDA {info.get('cuda')} HIP {info.get('hip')}")
    for s in steps:
        mark = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭ "}.get(s["status"], "?")
        print(f"{mark} {s['id']:3s} {s['name']:<{w}s} {s['status']:4s} {s['secs']:>5s}s  {s['note']}")
    n = {k: sum(1 for s in steps if s["status"] == k) for k in ("PASS", "FAIL", "SKIP")}
    print(f"通過 {n['PASS']}、失敗 {n['FAIL']}、略過 {n['SKIP']}　（每項的 log：{outdir}/<編號>.log）")
    json.dump({"env": info, "steps": steps, "counts": n, "all_pass": n["FAIL"] == 0},
              open(os.path.join(outdir, "verify_summary.json"), "w"), indent=1, ensure_ascii=False)
    return 0 if n["FAIL"] == 0 else 1


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    fn = {"env": env, "pick-n": pick_n, "corr": corr, "mixed": mixed, "regress": regress, "report": report}[cmd]
    sys.exit(fn(*args))
