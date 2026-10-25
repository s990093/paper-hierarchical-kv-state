"""guard.py — 第三階段量時間的 run 的 GPU 監看（CLAUDE.md §3：gpu_guard 必用；contaminated 的 run 不得寫進 results/）。

10/25 晚上補的（RUNLOG S3-GUARD-2）：W0、G-fio、G-M12 速度、G-M06 量測期間原本沒有監看，B0 有監看但被污染的 CSV 照樣留在
results/ 裡、report 也照讀。現在五個量時間的 part 都走這裡：

  gw = G.Timing(out, gpus=[0])     量測開始前。監看開不起來（nvidia-smi 查不到）→ 丟錯：量時間的 run 不能在看不到的情況下跑（規則 7）
  ...量測...
  dest = gw.finish()               乾淨 → out（原路徑）；污染（含監看查不到的時段）→ run 目錄裡的 <檔名>.CONTAMINATED，並留 CONTAMINATED
  gw.status()                      每列的 own_gpu_intruders：外來者數，或 UNOBSERVED:<n>（監看失敗過）
  G.evict(paths)                   B0 這種邊跑邊寫進 results 的：污染 → 把那幾個檔搬到 run 目錄
  G.drop_contaminated(rows)        report 用：run 目錄有 CONTAMINATED 的 run_id 整個排除（第二道防線）
"""
from __future__ import annotations

import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def runs_root() -> str:
    return os.environ.get("TIARA_RUNS", "")


def run_dir(out: str) -> str:
    """這個 run 的原始輸出目錄（tiara.sbatch 設的 RUN_ID）；沒有就用輸出檔旁邊（adhoc）。"""
    rid = os.environ.get("RUN_ID")
    if rid and runs_root():
        return os.path.join(runs_root(), rid)
    return os.path.dirname(os.path.abspath(out))


class Timing:
    def __init__(self, out: str, gpus=(0,)):
        from gpu_guard import GpuWatcher, smi_index
        self.out, self.gpus, self.watches, self.contaminated = out, list(gpus), [], None
        for i in self.gpus:
            w = GpuWatcher(gpu=smi_index(i), own_root=os.getpid())
            w.__enter__()                                       # 開頭就查不到 → SmiUnavailable 往上丟
            self.watches.append(w)
        print(f"[guard] 監看 GPU {self.gpus}（nvidia-smi 編號 {[w.gpu for w in self.watches]}）", flush=True)

    def status(self):
        n = sum(len(w.intruders) for w in self.watches)
        if any(getattr(w, "smi_failures", None) for w in self.watches):
            return f"UNOBSERVED:{n}"
        return n

    def finish(self) -> str:
        """停止監看；回傳結果檔該寫的位置。"""
        for w in self.watches:
            w.__exit__(None, None, None)
        self.contaminated = any(w.contaminated for w in self.watches)
        rep = dict(out=self.out, contaminated=self.contaminated, gpus=self.gpus,
                   watches=[dict(w.report(), logical_gpu=i) for i, w in zip(self.gpus, self.watches)])
        dest = self.out
        if self.contaminated:
            d = run_dir(self.out)
            os.makedirs(d, exist_ok=True)
            dest = os.path.join(d, os.path.basename(self.out) + ".CONTAMINATED")
            with open(os.path.join(d, "CONTAMINATED"), "w") as f:
                json.dump(dict(rep, verdicts=[w.verdict() for w in self.watches], moved_to=dest,
                               note="量時間的 run：不得寫進 results/，要重量（CLAUDE.md §3）"), f, indent=1, default=str, ensure_ascii=False)
            print(f"[guard] CONTAMINATED（{[w.verdict() for w in self.watches]}）→ 結果改寫到 {dest}，不進 results", flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        with open(os.path.splitext(dest)[0] + ".gpu_guard.json", "w") as f:
            json.dump(rep, f, indent=1, default=str, ensure_ascii=False)
        return dest


def evict(gw: Timing, paths: list) -> list:
    """已經寫好的檔（B0 邊跑邊寫）：gw 污染 → 搬到 run 目錄（<檔名>.CONTAMINATED）。回傳每個檔最後的位置。"""
    if not gw.contaminated:
        return list(paths)
    d = run_dir(gw.out)
    out = []
    for p in paths:
        if os.path.exists(p):
            q = os.path.join(d, os.path.basename(p) + ".CONTAMINATED")
            shutil.move(p, q)
            print(f"[guard] 搬出 results：{p} → {q}", flush=True)
            out.append(q)
    return out


def drop_contaminated(rows: list) -> tuple:
    """run 目錄有 CONTAMINATED 的 run_id → 那個 run 的列全部排除。回傳（留下的列, 排除的 run_id）。"""
    root = runs_root()
    bad = sorted(rid for rid in {r.get("run_id") for r in rows}
                 if rid and root and os.path.exists(os.path.join(root, rid, "CONTAMINATED")))
    if bad:
        print(f"[guard] 排除被污染的 run：{bad}", flush=True)
    return [r for r in rows if r.get("run_id") not in bad], bad


def self_test() -> bool:
    import tempfile
    from unittest import mock
    import gpu_guard as gg
    ok = True

    def chk(name, c):
        nonlocal ok
        print(("PASS " if c else "FAIL ") + name)
        ok &= bool(c)
    tmp = tempfile.mkdtemp()
    res = os.path.join(tmp, "results")
    os.makedirs(res)
    apps = {"rows": []}

    def _smi(query, extra=None):
        return {"gpu=index": [["0"], ["1"]], "gpu=index,uuid": [["0", "GPU-0"], ["1", "GPU-1"]],
                "compute-apps=gpu_uuid,pid,used_memory": apps["rows"]}[query]
    env = {"RUN_ID": "r1", "TIARA_RUNS": os.path.join(tmp, "runs")}
    with mock.patch.object(gg, "vendor", lambda: "nvidia"), mock.patch.object(gg, "_smi", _smi), \
         mock.patch.dict(os.environ, env), mock.patch.object(gg.GpuWatcher, "_loop", lambda self: None):
        out = os.path.join(res, "x.csv")
        gw = Timing(out, gpus=[0, 1])
        chk("乾淨：status 0", gw.status() == 0)
        dest = gw.finish()
        chk("乾淨：寫回原路徑、旁邊有 gpu_guard.json", dest == out and os.path.exists(os.path.join(res, "x.gpu_guard.json")))
        # 監看失敗過 → 污染、搬出 results
        gw = Timing(out, gpus=[0])
        gw.watches[0].smi_failures.append({"ts": "t", "error": "timeout"})
        chk("監看失敗：status UNOBSERVED:0", gw.status() == "UNOBSERVED:0")
        dest = gw.finish()
        chk("監看失敗：結果改寫到 run 目錄、留 CONTAMINATED", dest.startswith(env["TIARA_RUNS"]) and dest.endswith(".CONTAMINATED")
            and os.path.exists(os.path.join(env["TIARA_RUNS"], "r1", "CONTAMINATED")))
        open(out, "w").write("a\n")
        moved = evict(gw, [out, out + ".missing"])
        chk("evict：已寫的檔搬走、不在 results", not os.path.exists(out) and len(moved) == 1 and os.path.exists(moved[0]))
        rows, bad = drop_contaminated([{"run_id": "r1"}, {"run_id": "r2"}])
        chk("drop_contaminated：r1 排除、r2 留下", bad == ["r1"] and rows == [{"run_id": "r2"}])
        # 外來 process（不是自己的子孫）在卡上 → 污染
        apps["rows"] = [["GPU-0", "999999", "100"]]
        with mock.patch.object(gg, "_holders", lambda gpu: {}, create=True):
            try:
                gw = Timing(out, gpus=[0])
                chk("開跑前就有外來者 → contaminated", gw.finish().endswith(".CONTAMINATED"))
            except gg.SmiUnavailable as ex:          # 這台機器上 foreign_on 需要的其他查詢不在 mock 裡
                print("SKIP 外來者（mock 不完整）：", ex)
    # 開頭就查不到 → 丟錯
    def bad_smi(query, extra=None):
        raise gg.SmiUnavailable("nvidia-smi 執行失敗：timeout")
    with mock.patch.object(gg, "vendor", lambda: "nvidia"), mock.patch.object(gg, "_smi", bad_smi):
        try:
            Timing(os.path.join(res, "y.csv"))
            chk("開頭查不到 → 丟錯", False)
        except gg.SmiUnavailable:
            chk("開頭查不到 → 丟錯", True)
    shutil.rmtree(tmp)
    print("self-test", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(0 if self_test() else 1)
    print(__doc__)
