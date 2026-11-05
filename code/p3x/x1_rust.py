"""x1_rust.py — ICPE next（icpe-sim）：x1 sweep 的 Rust 引擎（`x1 sweep --engine rust`；預設 py＝原本的 Python Engine）。

引擎＝icpe-ana 的移植（origin/icpe-next-ana @ aee1527 的 code/icpe/ana_x1fast/v3_prefix_fork），複製到 code/p3x/x1fast/：
引擎部分一行沒改，只改 main() 的輸入輸出（加常駐模式）。審查與驗證見 SIM_CHANGES 第 16 項。
W4 雜訊的 B1 抽法＝icpe-ana v4（origin/icpe-next-ana @ 279eb7e 的 code/icpe/ana_x1fast/v4_noise_on_0b0ef1e/src/main.rs，原樣複製）：
jobs.txt 那一行多一欄 `nz=P|M,<λ>,B1,<seed>`。審查與驗證見 SIM_CHANGES 第 17 項。
這裡只做「Python 準備輸入 → Rust 跑 → 讀回逐請求陣列填進原本的 View」：之後的 summarize、存 npz、寫 CSV 全部是
x1_sim 原本那一段（_job_rows），所以輸出欄位與格式和 Python 引擎相同。
  prepare_trace：紀錄 → CSR 二進位檔（父行程做一次，內容 sha256 定址、可共用）
  Server：每份紀錄一個常駐的 x1fast（紀錄只讀一次、多執行緒）；write_job／read_outputs：一個 job 的輸入目錄與讀回
Rust 版沒有的功能（W4 雜訊的 B2 抽法、慢層容量上限、STAIR）→ unsupported() 回傳原因，那些 job 用 Python Engine（x1_sim 先跑 Rust 的、再跑這些）。

環境變數：
  X1_RUST_BIN    已編好的執行檔（不給就用 X1_CARGO 編 code/p3x/x1fast，放 $X1_RUST_WORK/build-<原始碼 sha>/）
  X1_RUST_WORK   工作目錄（紀錄的二進位檔、每個 job 的暫存；預設 /ssd6/hungwei/icpe-next/x1fast-work）
  X1_CARGO、CARGO_HOME、RUSTUP_HOME   編譯用（預設 acvlab 上 icpe-ana 裝的工具鏈）
  X1_RUST_TIMING_LOG   選填：每個 job 一行 JSON（Rust 引擎時間、Rust 端總時間、從派出到讀回）
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "x1fast")
SRC_FILES = ("Cargo.toml", os.path.join("src", "main.rs"))
WORK = os.environ.get("X1_RUST_WORK", "/ssd6/hungwei/icpe-next/x1fast-work")
TOOLS = "/ssd6/hungwei/icpe-next/ana-bench/tools"
GiB = 1 << 30

_TDIR = {}          # (trace, rule) → 紀錄的二進位檔目錄（父行程準備；fork 出來的 worker 繼承）
_BIN = None


def _sha_files(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(os.path.basename(p).encode() + b"\0")
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
    return h.hexdigest()


def src_sha():
    return _sha_files([os.path.join(SRC, f) for f in SRC_FILES])


def file_sha(p):
    return _sha_files([p])


def binary():
    """執行檔路徑：$X1_RUST_BIN，或編 code/p3x/x1fast（profile fast、不加 target-cpu：不用 FMA，結果和 CPU 型號無關）。"""
    global _BIN
    if _BIN:
        return _BIN
    b = os.environ.get("X1_RUST_BIN", "")
    if not b:
        tgt = os.path.join(WORK, f"build-{src_sha()[:16]}")
        b = os.path.join(tgt, "fast", "x1fast3")
        if not os.path.exists(b):
            env = dict(os.environ, CARGO_TARGET_DIR=tgt,
                       CARGO_HOME=os.environ.get("CARGO_HOME", f"{TOOLS}/cargo"),
                       RUSTUP_HOME=os.environ.get("RUSTUP_HOME", f"{TOOLS}/rustup"))
            env.pop("RUSTFLAGS", None)
            cargo = os.environ.get("X1_CARGO", f"{TOOLS}/cargo/bin/cargo")
            cp = subprocess.run([cargo, "build", "--profile", "fast", "--offline", "--locked"], cwd=SRC, env=env,
                                capture_output=True, text=True)
            if cp.returncode != 0:
                raise RuntimeError(f"cargo build 失敗（exit {cp.returncode}）：\n{cp.stderr[-4000:]}")
    if not os.path.isfile(b):
        raise FileNotFoundError(b)
    _BIN = b
    return b


def unsupported(planner, nz, sc):
    """Rust 版不支援的功能 → 原因字串；支援 → None。W4 雜訊只支援 B1 抽法（v4）。"""
    if sc is not None:
        return "slow_cap"
    if nz is not None and nz["draw"] != "B1":
        return "noise_" + str(nz["draw"])
    if planner != "cake":
        return "stair"
    return None


def nz_field(nz):
    """W4 雜訊（B1）→ jobs.txt 多出來的那一欄 `nz=P|M,<λ>,B1,<seed>`；λ 用 repr（Rust 的 parse 讀回同一個 double）。"""
    m, lam, dr, seed = nz["model"], float(nz["lam"]), nz["draw"], int(nz["seed"])
    assert m in ("P", "M") and dr == "B1" and 0.0 <= lam < 1.0, nz
    return f"nz={m},{lam!r},{dr},{seed}"


def prepare_trace(P, trace, rule):
    """紀錄 → Rust 讀的 CSR 二進位檔。寫進暫存目錄、算內容 sha256、搬到 $X1_RUST_WORK/trace-<sha>/（同內容就共用）。
    用 A 驗 B：acc（block → 讀它的請求）的總長＝full（請求 → block）的總長；每個 block 的請求序號嚴格遞增；
    block id < B < 2^32。"""
    os.makedirs(WORK, exist_ok=True)
    full, acc = P["full"], P["acc"]
    n = len(full)
    B, S = int(P["B"]), int(P["R"]["n_sess"])
    lens = np.fromiter(map(len, full), np.int64, n)
    off = np.zeros(n + 1, np.int64)
    np.cumsum(lens, out=off[1:])
    m = int(off[-1])
    blk = np.fromiter(itertools.chain.from_iterable(full), np.int64, m)
    assert m == 0 or (0 <= blk.min() and blk.max() < B < 2 ** 32), (trace, B)
    keys = np.fromiter(acc.keys(), np.int64, len(acc))
    alen = np.fromiter(map(len, acc.values()), np.int64, len(acc))
    assert int(alen.sum()) == m, f"{trace}: acc 總長 {int(alen.sum())} ≠ full 總長 {m}"
    assert len(keys) == 0 or (0 <= keys.min() and keys.max() < B)
    flat = np.fromiter(itertools.chain.from_iterable(acc.values()), np.int64, m)
    kk = np.repeat(keys, alen)
    o = np.argsort(kk, kind="stable")            # 依 block 排（同一個 block 內保持原本的順序）
    areq = flat[o]
    kk = kk[o]
    del flat, o
    if m > 1:
        same = kk[1:] == kk[:-1]
        assert bool(np.all(areq[1:][same] > areq[:-1][same])), f"{trace}: acc 的請求序號不是嚴格遞增"
    al = np.zeros(B, np.int64)
    al[keys] = alen
    aoff = np.zeros(B + 1, np.int64)
    np.cumsum(al, out=aoff[1:])
    del kk, al
    tmp = tempfile.mkdtemp(prefix="trace-tmp-", dir=WORK)
    files = []

    def put(name, arr, dt):
        p = os.path.join(tmp, name)
        np.ascontiguousarray(np.asarray(arr, dt)).tofile(p)
        files.append(p)
    put("meta.i64", [n, B, S], np.int64)
    put("req_off.i64", off, np.int64)
    put("req_blk.u32", blk, np.uint32)
    put("nall.i64", P["nall"], np.int64)
    put("kind.i64", P["kindc"], np.int64)
    put("sess.i64", P["sessl"], np.int64)
    put("parent.i64", P["parentl"], np.int64)
    put("t.f64", P["tl"], np.float64)
    put("acc_off.i64", aoff, np.int64)
    put("acc_req.i64", areq, np.int64)
    sha = _sha_files(files)
    with open(os.path.join(tmp, "SOURCE.txt"), "w") as fh:
        fh.write(f"trace={trace} rule={rule} n_req={n} B={B} S={S} chunks={m} sha256={sha}\n")
    dst = os.path.join(WORK, f"trace-{sha[:24]}")
    if os.path.isdir(dst):
        shutil.rmtree(tmp)
    else:
        try:
            os.rename(tmp, dst)
        except OSError:                          # 別的 sweep 同時搬好了同一份
            shutil.rmtree(tmp)
            assert os.path.isdir(dst), dst
    _TDIR[(trace, rule)] = dst
    return dst, sha


class RustEng:
    """summarize／_gpu_summary 讀的引擎計數（和 x1_sim.Engine 同名的屬性）。"""

    def __init__(self, ei, res_sum, G, policy):
        self.n_demote, self.n_drop, self.n_unpf, self.n_gevict, self.n_gexpire, self.g_wb_blocks, self.w_slow_blocks = ei
        self.res_sum = res_sum
        self.G = G
        self.policy = policy


def _write_f(p, x):
    np.asarray(x, np.float64).tofile(p)


def check_job(XS, policy, views, K, gkw):
    """同 Engine.__init__ 的檢查＋Rust 版的前提（每個時間設定的 i4 相同、cake 規劃器；GDSF 的 gdsf_view＝views[0]）。"""
    o, a, h, pf = XS.POLICIES[policy]
    G = int(gkw.get("gpu_cap", 0))
    if pf:
        assert len(views) == 1, "預取策略的狀態和時間有關，一次只能跑一種時間設定"
    if G > 0:
        assert gkw.get("gpu_write", "wb") in XS.GPU_WRITES, gkw
        if XS.parse_gpol(gkw.get("gpu_policy", "LRU"))[1] is not None:
            assert len(views) == 1, "TTL：一次只能跑一種時間設定"
        assert gkw.get("pf_target", "cpu") in ("cpu", "gpu"), gkw
    else:
        assert gkw.get("pf_target", "cpu") == "cpu", "預取搬回 GPU 要開 GPU 層"
    for v in views:
        assert v.i4 == K["int4_bytes"] and v.planner == "cake", (v.key, v.i4, v.planner)
    if o == "gdsf":
        assert all(v.slow_name == views[0].slow_name and v.gpu == views[0].gpu for v in views)


def write_job(XS, P, policy, cap_bytes, views, K, gkw, nz=None):
    """一個 job 的輸入目錄（views.txt、f*／fc*、fc_idle.f64、i4.txt、一行的 jobs.txt）。f／Fc 直接取自 Python 的 View。
    nz＝W4 雜訊（x1_sim 的 noise spec；只能是 B1，見 unsupported）→ jobs.txt 多一欄 nz=…（同 Engine(noise=nz)；λ＝0 也照送）。"""
    check_job(XS, policy, views, K, gkw)
    o, a, h, pf = XS.POLICIES[policy]
    G = int(gkw.get("gpu_cap", 0))
    os.makedirs(os.path.join(WORK, "jobs"), exist_ok=True)
    jd = tempfile.mkdtemp(prefix=f"job-{os.getpid()}-", dir=os.path.join(WORK, "jobs"))
    fid = {}
    with open(os.path.join(jd, "views.txt"), "w") as fh:
        for i, v in enumerate(views):
            if v.gpu not in fid:
                fid[v.gpu] = len(fid)
                _write_f(os.path.join(jd, f"f{fid[v.gpu]}.f64"), v.f)
                _write_f(os.path.join(jd, f"fc{fid[v.gpu]}.f64"), v.Fc)
            fh.write(" ".join([str(i)] + [repr(float(x)) for x in (v.d, v.cpu.r, v.cpu.w, v.cpu.k, v.slow.r, v.slow.w, v.slow.k)]
                              + [v.mode, str(fid[v.gpu])]) + "\n")
    fi = XS.f_array(K, P["nmax"], "idle")        # 同 Engine.__init__ 的 pens_fc（Pensieve）
    pfc = [0.0]
    for x in fi:
        pfc.append(pfc[-1] + x)
    _write_f(os.path.join(jd, "fc_idle.f64"), pfc)
    with open(os.path.join(jd, "i4.txt"), "w") as fh:
        fh.write(str(K["int4_bytes"]))
    dead = float(P["tl"][-1] - float(policy.split("-H")[1])) if "-H" in policy else float("inf")
    if G > 0:
        go, gt = XS.parse_gpol(gkw.get("gpu_policy", "LRU"))
        gf = [str(G), go, "-" if gt is None else repr(gt), gkw.get("gpu_write", "wb"), gkw.get("pf_target", "cpu")]
    else:
        gf = ["0", "-", "-", "wb", "cpu"]
    with open(os.path.join(jd, "jobs.txt"), "w") as fh:
        fh.write(" ".join(["j0", o, a, h or "-", pf or "-", repr(dead), repr(cap_bytes)] + gf
                          + [",".join(str(i) for i in range(len(views)))] + ([] if nz is None else [nz_field(nz)])) + "\n")
    return jd


def read_outputs(jd, P, views, gkw, policy):
    """Rust 的輸出 → 寫回 views（逐請求陣列、I/O 線狀態、預取位元組），回傳 RustEng。"""
    G = int(gkw.get("gpu_cap", 0))
    od = os.path.join(jd, "out", "j0")
    nreq = len(P["full"])
    for k, v in enumerate(views):
        def rd(nm, dt):
            x = np.fromfile(os.path.join(od, f"v{k}_{nm}"), dt)
            assert len(x) == nreq, (nm, len(x), nreq)
            return x
        v.ttft = rd("ttft.f64", np.float64)
        v.rest = rd("rest.f64", np.float64)
        v.nC = rd("nc.i32", np.int32)
        v.nS = rd("ns.i32", np.int32)
        v.nR = rd("nr.i32", np.int32)
        if G > 0:
            v.nG = rd("ng.i32", np.int32)
        ch = np.fromfile(os.path.join(od, f"v{k}_chan.f64"), np.float64).tolist()
        cb = [int(x) for x in np.fromfile(os.path.join(od, f"v{k}_chanb.i64"), np.int64).tolist()]
        assert len(ch) == 8 and len(cb) == 5
        v.cpu.t_free, v.cpu.t_free_w, v.cpu.busy_r, v.cpu.busy_w, v.slow.t_free, v.slow.t_free_w, v.slow.busy_r, v.slow.busy_w = ch
        v.cpu.bytes_r, v.cpu.bytes_w, v.slow.bytes_r, v.slow.bytes_w, v.pf_bytes = cb
    assert not os.path.exists(os.path.join(od, f"v{len(views)}_ttft.f64")), "Rust 輸出的時間設定比 views 多"
    ei = [int(x) for x in np.fromfile(os.path.join(od, "eng.i64"), np.int64).tolist()]
    ef = np.fromfile(os.path.join(od, "eng.f64"), np.float64).tolist()
    assert len(ei) == 7 and len(ef) == 1
    return RustEng(ei, float(ef[0]), G, policy)


def run(XS, P, trace, rule, policy, cap_bytes, views, K, gkw):
    """（單一 job、批次模式、一個子行程；自檢與對拍用。sweep 用 Server。）結果同 Engine(...).run()。"""
    jd = write_job(XS, P, policy, cap_bytes, views, K, gkw)
    try:
        env = dict(os.environ, X1FAST_THREADS="1", X1FAST_FORK="0", X1FAST_SUMMARY_ONLY="0")
        env.pop("X1FAST_SERVE", None)
        cp = subprocess.run([binary(), _TDIR[(trace, rule)], jd], env=env, capture_output=True, text=True)
        if cp.returncode != 0:
            raise RuntimeError(f"x1fast 失敗（exit {cp.returncode}）：{trace} {rule} {policy} cap={cap_bytes} {gkw}\n{cp.stderr[-4000:]}")
        return read_outputs(jd, P, views, gkw, policy)
    finally:
        shutil.rmtree(jd, ignore_errors=True)


class Server:
    """常駐的 x1fast（X1FAST_SERVE=1）：紀錄只讀一次、threads 個執行緒。submit(jd) 送一個 job 目錄；
    完成時讀取執行緒把 ("done", jd, run_s, wall_s) 放進 events；行程結束（stdout 關掉）放 ("eof", self)。stderr 沿用父行程的。"""

    def __init__(self, tdir, threads, events):
        env = dict(os.environ, X1FAST_SERVE="1", X1FAST_THREADS=str(threads), X1FAST_FORK="0", X1FAST_SUMMARY_ONLY="0")
        self.tdir = tdir
        self.closing = False
        self.p = subprocess.Popen([binary(), tdir], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env, bufsize=1)
        self.th = threading.Thread(target=self._read, args=(events,), daemon=True)
        self.th.start()

    def _read(self, events):
        for ln in self.p.stdout:
            x = ln.split()
            if len(x) == 4 and x[0] == "done":
                events.put(("done", x[1], float(x[2]), float(x[3])))
        events.put(("eof", self))

    def submit(self, jd):
        self.p.stdin.write(jd + "\n")
        self.p.stdin.flush()

    def close(self):
        self.closing = True
        try:
            self.p.stdin.close()
        except OSError:
            pass
        rc = self.p.wait()
        self.th.join()
        return rc

    def kill(self):
        self.closing = True
        self.p.kill()
        self.p.wait()
