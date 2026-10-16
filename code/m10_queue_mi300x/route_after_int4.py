"""第 2 階（INT4）端點判完後，只調「後面 GPU 工作的順序」（不動判準、不動任何數字）。stage 33bz 呼叫。

判定一律由凍結的 m10_analyze.py 算（analyze_int4.sh）；K0 由 33a 的 k0-verdict 算。這支只讀它們的 JSON：
  - 33a 速度格子的 GPU 監控有任何一份不是 CLEAN → NEEDS_HUMAN（CLAUDE.md §3：時間要重量），不動佇列。
  - K0（INT4）成立 → 03 §2.4：精度對還原時間沒有賺頭 → 位置策略停；INT4 混合配置變成只報告 → 33c 搬到 46。
  - 檢定力擋判定 → NEEDS_HUMAN（07 §9 的加題只寫了 Step 1 的版本），不動佇列（33c 照跑，GPU 不空著）。
  - 兩個模型都過 → 03 §2.5 第 3 步：整個停；INT4 混合配置變成只報告 → 33c 搬到 46（大掃描 40–42 之後）。
  - 有模型沒過 → Step 2 在 decide 指定的模型上做（03 §2.5 第 4 步、§1.1）：33c 只跑那個模型；
    另一個模型的混合配置（只報告）拆成 46。
  - 其他（資料不全、程式出錯）→ NEEDS_HUMAN，不動佇列。
用法：route_after_int4.py
"""
import glob
import json
import os
import sys

J = os.path.dirname(os.path.abspath(__file__))
Q = os.path.join(J, "queue")
REPO = "/mlsteam/workspace/paper-hierarchical-kv-state"
OUT = os.environ.get("M10_OUT", f"{REPO}/results/m10_precision_mi300x")
C33 = os.path.join(Q, "33c_quality_int4_mixed.sh")


def log(x):
    print(f"[route-int4] {x}", flush=True)


def human(why):
    open(os.path.join(J, "NEEDS_HUMAN"), "a").write(why + "\n")
    log("NEEDS_HUMAN: " + why)
    sys.exit(0)


def guard_clean(m):
    """33a 那次 grid（最新的 m10-grid-int4-<m>）的每一份 gpu_guard JSON 都要 CLEAN。"""
    fs = sorted(glob.glob(f"{OUT}/speed_{m}/gpu_guard_*-m10-grid-int4-{m}_L*.json"))
    if not fs:
        return None, "找不到 gpu_guard JSON"
    rid = max(os.path.basename(f)[len("gpu_guard_"):].rsplit("_L", 1)[0] for f in fs)
    bad = []
    for f in fs:
        if rid not in f:
            continue
        d = json.load(open(f))
        if d.get("verdict") != "CLEAN" or d.get("contaminated"):
            bad.append(f"{os.path.basename(f)}={d.get('verdict')}")
    return (not bad), (rid if not bad else "；".join(bad))


def split_33c(main_model):
    """33c 只跑 main_model；另一個模型（只報告）寫成 46。"""
    other = "qwen" if main_model == "llama" else "llama"
    src = open(C33).read()
    key = 'MODELS=${MODELS_51:-"llama qwen"}'
    if key not in src:
        human(f"33c 的內容和預期不同（找不到 {key}），沒有拆")
    open(C33, "w").write(src.replace(key, f'MODELS="{main_model}"   # stage 33bz：Step 2 在 {main_model} 上做'))
    open(os.path.join(Q, f"46_quality_int4_mixed_{other}.sh"), "w").write(
        src.replace(key, f'MODELS="{other}"   # stage 33bz：{other} 不進 Step 2，只報告'))
    log(f"33c 只跑 {main_model}；{other} 的 INT4 混合配置 → 46（只報告，排在大掃描後面）")


def to_report_only(why):
    if os.path.exists(C33):
        os.rename(C33, os.path.join(Q, "46_quality_int4_mixed.sh"))
        log(f"queue 33c → 46（{why}；INT4 混合配置變成只報告，排在大掃描 40–42 後面）")
    else:
        log("33c 不在（已跑過或已搬）")


def main():
    for m in ("llama", "qwen"):
        ok, info = guard_clean(m)
        if ok is None or not ok:
            human(f"INT4 速度格子 {m} 的 GPU 監控不是 CLEAN 或找不到（{info}）→ 時間要重量（CLAUDE.md §3）")
        log(f"{m} 速度格子監控 CLEAN（{info}）")
    k0 = {}
    for m in ("llama", "qwen"):
        v = json.load(open(f"{OUT}/speed_{m}/k0_verdict_int4.json"))["verdict"]
        k0[m] = v.startswith("K0 成立")
        log(f"K0(INT4, {m}) = {k0[m]}（{v}）")
    d = json.load(open(f"{OUT}/analysis/int4/decide_int4.json"))
    log("next = " + d["next"])
    log("next_frozen_code = " + d["next_frozen_code"])
    if all(k0.values()):
        to_report_only("K0 在 INT4 兩個模型都成立")
        return
    blk = [m for m in ("llama", "qwen") if d[m].get("power_blocks")]
    if blk:
        human(f"INT4 端點檢定力擋判定：{blk}（07 §9 的加題只寫了 Step 1 的版本）→ 要人判斷；33c 照跑")
    L, Qp = d["llama"]["model_pass"], d["qwen"]["model_pass"]
    if L is None or Qp is None:
        human("decide_int4 說資料不全：" + d["next"])
    if L and Qp:
        to_report_only("INT4 兩個模型都過 → 03 §2.5 第 3 步：整個停")
        return
    m = "qwen" if (L and not Qp) else "llama"      # 照 m10_analyze.cmd_decide 的分支（兩個都沒過 → Llama）
    if k0[m]:
        to_report_only(f"Step 2 會在 {m} 上做，但 {m} 的 K0（INT4）成立 → 沒有賺頭，位置策略停")
        return
    log(f"INT4 階 {m} 均勻不夠 → Step 2 在 {m} 上做")
    split_33c(m)


if __name__ == "__main__":
    main()
