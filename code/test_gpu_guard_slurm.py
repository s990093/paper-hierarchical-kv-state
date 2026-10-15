"""test_gpu_guard_slurm.py — gpu_guard 在 Slurm（平台 C，H200）下的判斷，用 mock 測（這台沒有 Slurm、沒有 nvidia-smi）。

要確認的事（03_plan_h200 §6、H200_SLURM §5.3、CLAUDE.md 規則 7）：
  1. Slurm job 看得到的卡比節點上少 → host_contention 回 UNOBSERVABLE_HOST，不回 QUIET。
  2. 節點卡數查不到（scontrol 失敗、沒有 Gres）→ 也是 UNOBSERVABLE_HOST。
  3. 看得到整個節點、沒有外來 process → QUIET（原本的判斷照舊）。
  4. 不在 Slurm、AMD：行為和改之前一樣。
  5. smi_index：CUDA_VISIBLE_DEVICES 對到 nvidia-smi 的哪張卡；對不上就丟錯，不猜。

  python code/test_gpu_guard_slurm.py      # 全部通過印 ALL OK，失敗丟 AssertionError
"""
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_guard as gg  # noqa: E402

SCONTROL_8 = ("NodeName=hgx-cn01 Arch=x86_64 CoresPerSocket=56\n"
              "   Gres=gpu:h200:8(S:0-1)\n   OS=Linux\n")


def fake_smi(rows_by_query):
    def _smi(query, extra=None):
        return rows_by_query[query]
    return _smi


def fake_run(scontrol_out=None, fail=False):
    def _run(cmd):
        if cmd[0] == "scontrol":
            if fail:
                raise gg.SmiUnavailable("scontrol: command not found")
            return scontrol_out
        raise AssertionError(f"unexpected command {cmd}")
    return _run


def nvidia_env(n_visible, apps=(), slurm=True, scontrol_out=SCONTROL_8, scontrol_fail=False, cvd=None):
    rows = {
        "gpu=index": [[str(i)] for i in range(n_visible)],
        "gpu=index,uuid": [[str(i), f"GPU-{i:04d}"] for i in range(n_visible)],
        "gpu=index,utilization.gpu": [[str(i), "0"] for i in range(n_visible)],
        "compute-apps=gpu_uuid,pid,used_memory": [list(a) for a in apps],
    }
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM") and k != "CUDA_VISIBLE_DEVICES"}
    if slurm:
        env.update(SLURM_JOB_ID="123", SLURMD_NODENAME="hgx-cn01")
    if cvd is not None:
        env["CUDA_VISIBLE_DEVICES"] = cvd
    return [mock.patch.object(gg, "vendor", lambda: "nvidia"),
            mock.patch.object(gg, "_smi", fake_smi(rows)),
            mock.patch.object(gg, "_run", fake_run(scontrol_out, scontrol_fail)),
            mock.patch.dict(os.environ, env, clear=True)]


def run_with(patches, fn):
    for p in patches:
        p.start()
    try:
        return fn()
    finally:
        for p in reversed(patches):
            p.stop()


def test_slurm_constrained_is_unobservable():
    r = run_with(nvidia_env(1), gg.host_contention)
    assert r["level"] == "UNOBSERVABLE_HOST", r
    assert r["visible_gpus"] == 1 and r["node_gpus"] == 8, r


def test_slurm_scontrol_fails_is_unobservable():
    r = run_with(nvidia_env(8, scontrol_fail=True), gg.host_contention)
    assert r["level"] == "UNOBSERVABLE_HOST" and r["node_gpus"] is None, r


def test_slurm_no_gres_is_unobservable():
    r = run_with(nvidia_env(8, scontrol_out="NodeName=x Gres=(null)\n"), gg.host_contention)
    assert r["level"] == "UNOBSERVABLE_HOST" and r["node_gpus"] is None, r


def test_slurm_full_view_quiet():
    r = run_with(nvidia_env(8), gg.host_contention)
    assert r["level"] == "QUIET", r


def test_slurm_full_view_sees_neighbour():
    # 別人的 process 在 3 號卡上（pid 不屬於本行程樹）
    r = run_with(nvidia_env(8, apps=[("GPU-0003", "999999", "40000")]), gg.host_contention)
    assert r["level"] in ("LIGHT", "HEAVY") and r["foreign_gpus"] == [3], r


def test_not_slurm_unchanged():
    r = run_with(nvidia_env(1, slurm=False), gg.host_contention)
    assert r["level"] == "QUIET", r          # 和改之前一樣：沒有 Slurm 就不做這個判斷
    assert "node_gpus" not in r


def test_amd_unchanged():
    with mock.patch.object(gg, "vendor", lambda: "amd"), \
         mock.patch.object(gg, "gpu_util", lambda: {0: 0}), \
         mock.patch.dict(os.environ, {"SLURM_JOB_ID": "1"}):
        r = gg.host_contention()
    assert r["level"] == "UNOBSERVABLE_HOST" and "node_gpus" not in r, r
    with mock.patch.object(gg, "vendor", lambda: "amd"), \
         mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "5"}):
        assert gg.smi_index(0) == 0


def test_gres_parse():
    for out, want in [(SCONTROL_8, 8), ("Gres=gpu:4 ", 4), ("Gres=gpu:a100:2,gpu:h200:2(S:0)", 4),
                      ("Gres=(null)", None), ("NodeName=x", None)]:
        with mock.patch.object(gg, "_run", fake_run(out)):
            got = gg._slurm_node_gpu_count()
        assert got == want, (out, got)


def test_smi_index():
    cases = [
        (dict(n_visible=8, cvd=None), 0, 0),          # 沒設 CUDA_VISIBLE_DEVICES：照舊
        (dict(n_visible=8, cvd="3"), 0, 3),           # 沒有 cgroup 限制：job 的 cuda:0＝節點的 3 號
        (dict(n_visible=8, cvd="1,0"), 1, 0),
        (dict(n_visible=1, cvd="0"), 0, 0),           # cgroup 只露出自己的卡、編號一致
        (dict(n_visible=8, cvd="GPU-0005"), 0, 5),    # UUID
    ]
    for kw, logical, want in cases:
        got = run_with(nvidia_env(**kw), lambda: gg.smi_index(logical))
        assert got == want, (kw, logical, got)
    # cgroup 只露出一張、nvidia-smi 顯示節點編號 5、Slurm 給 job 內編號 0
    rows = {"gpu=index,uuid": [["5", "GPU-0005"]]}
    with mock.patch.object(gg, "vendor", lambda: "nvidia"), mock.patch.object(gg, "_smi", fake_smi(rows)), \
         mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
        assert gg.smi_index(0) == 5
    # 對不上：丟錯，不猜
    for kw, logical in [(dict(n_visible=8, cvd="9"), 0), (dict(n_visible=8, cvd="0"), 1),
                        (dict(n_visible=8, cvd="GPU-zzzz"), 0)]:
        try:
            run_with(nvidia_env(**kw), lambda: gg.smi_index(logical))
        except gg.SmiUnavailable:
            continue
        raise AssertionError(f"smi_index should raise for {kw} logical={logical}")


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
            n += 1
    print(f"ALL OK ({n} tests)")
