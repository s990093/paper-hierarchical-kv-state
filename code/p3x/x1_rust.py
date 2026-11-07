"""x1_rust.py — ICPE next（icpe-sim）：x1 sweep 的 Rust 引擎（`x1 sweep --engine rust`；預設 py＝原本的 Python Engine）。

引擎＝icpe-ana 的移植（origin/icpe-next-ana @ aee1527 的 code/icpe/ana_x1fast/v3_prefix_fork），複製到 code/p3x/x1fast/：
引擎部分一行沒改，只改 main() 的輸入輸出（加常駐模式）。審查與驗證見 SIM_CHANGES 第 16 項。
W4 雜訊的 B1 抽法＝icpe-ana v4（origin/icpe-next-ana @ 279eb7e 的 code/icpe/ana_x1fast/v4_noise_on_0b0ef1e/src/main.rs，原樣複製）：
jobs.txt 那一行多一欄 `nz=P|M,<λ>,B1,<seed>`。審查與驗證見 SIM_CHANGES 第 17 項。
W4 雜訊的 B2 抽法＝icpe-ana v5（origin/icpe-next-ana @ 43e7ffc 的 code/icpe/ana_x1fast/v5_noise_b2/src/main.rs，原樣複製＋SIM_CHANGES 19 的兩行）：
`nz=P|M,<λ>,B2,<seed>`。B2 每次問都重抽，亂數的順序跟著 Python 走訪 cpu_set（CPython 3.12 的 set）的順序，v5 照 setobject.c 重現；
所以只有 Python 3.12 時才交給 Rust，其他版本退回 Python Engine。審查與驗證見 SIM_CHANGES 第 20 項。
這裡只做「Python 準備輸入 → Rust 跑 → 讀回逐請求陣列填進原本的 View」：之後的 summarize、存 npz、寫 CSV 全部是
x1_sim 原本那一段（_job_rows），所以輸出欄位與格式和 Python 引擎相同。
  prepare_trace：紀錄 → CSR 二進位檔（父行程做一次，內容 sha256 定址、可共用）
  prepare_cached：父行程的準備（x1_sim.prep＋prepare_trace）。nb: 紀錄存一份快取（$X1_RUST_WORK/prep-<鍵>/），
    鍵＝紀錄檔內容 sha256＋x1_sim／x1_nb／x1_rust 的 sha256＋numpy／Python 版本；命中就不用重建（SIM_CHANGES 第 18 項）。
    這份紀錄的 job 全部走 Rust 時，父行程的 P 不留 full／acc（lite P；Rust 讀紀錄檔，父行程與摘要只用到其他欄位）。
  Server：每份紀錄一個常駐的 x1fast（紀錄只讀一次、多執行緒）；write_job／read_outputs：一個 job 的輸入目錄與讀回
Rust 版沒有的功能（慢層容量上限、STAIR；Python 不是 3.12 時的 B2）→ unsupported() 回傳原因，那些 job 用 Python Engine（x1_sim 先跑 Rust 的、再跑這些）。

環境變數：
  X1_RUST_BIN    已編好的執行檔（不給就用 X1_CARGO 編 code/p3x/x1fast，放 $X1_RUST_WORK/build-<原始碼 sha>/）
  X1_RUST_WORK   工作目錄（紀錄的二進位檔、每個 job 的暫存；預設 /ssd6/hungwei/icpe-next/x1fast-work）
  X1_CARGO、CARGO_HOME、RUSTUP_HOME   編譯用（預設 acvlab 上 icpe-ana 裝的工具鏈）
  X1_RUST_TIMING_LOG   選填：每個 job 一行 JSON（Rust 引擎時間、Rust 端總時間、從派出到讀回）
  X1_PREP_CACHE  0＝不用也不寫準備的快取（預設 1）；lite P 不受影響
"""
from __future__ import annotations

import gc
import hashlib
import itertools
import json
import os
import pickle
import shutil
import subprocess
import socket
import sys
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
TRACE_FILES = ("meta.i64", "req_off.i64", "req_blk.u32", "nall.i64", "kind.i64", "sess.i64", "parent.i64", "t.f64",
               "acc_off.i64", "acc_req.i64")     # prepare_trace 寫的順序（sha256 照這個順序算）
PREP_CACHE = os.environ.get("X1_PREP_CACHE", "1") != "0"

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


def unsupported(planner, nz, sc, nzo="clamp"):
    """Rust 版不支援的功能 → 原因字串；支援 → None。W4 雜訊：B1（v4）、B2（v5；只在 Python 3.12，見檔頭）。
    PREREG 偏離 14 (a)（nzo＝resample）：重抽會用亂數，B1 也要照 CPython 3.12 走訪 set 的順序 → 同 B2，只在 Python 3.12 交給 Rust。"""
    if sc is not None:
        return "slow_cap"
    if nz is not None and nz["draw"] not in ("B1", "B2"):
        return "noise_" + str(nz["draw"])
    if nz is not None and nz["draw"] == "B2" and sys.version_info[:2] != (3, 12):
        return "noise_B2_py" + "%d%d" % sys.version_info[:2]
    if nz is not None and nzo != "clamp" and sys.version_info[:2] != (3, 12):
        return "noise_overdue_py" + "%d%d" % sys.version_info[:2]
    if planner != "cake":
        return "stair"
    return None


def nz_field(nz):
    """W4 雜訊（B1／B2）→ jobs.txt 多出來的那一欄 `nz=P|M,<λ>,B1|B2,<seed>`；λ 用 repr（Rust 的 parse 讀回同一個 double）。"""
    m, lam, dr, seed = nz["model"], float(nz["lam"]), nz["draw"], int(nz["seed"])
    assert m in ("P", "M") and dr in ("B1", "B2") and 0.0 <= lam < 1.0, nz
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
    assert tuple(os.path.basename(p) for p in files) == TRACE_FILES
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


class NoFull:
    """lite P 的 full：只記請求數（make_views、read_outputs 只用 len(P["full"])）。其他用法 → 錯誤，不會默默用到。"""
    __slots__ = ("n",)

    def __init__(self, n):
        self.n = int(n)

    def __len__(self):
        return self.n

    def _no(self, *a):
        raise RuntimeError("lite P 沒有 full（逐請求的 block 清單）：這份紀錄應該只有 Rust 的 job（x1_rust.prepare_cached）")
    __getitem__ = __iter__ = _no


def _lite(P):
    """P 去掉 full、acc（父行程最大的兩個結構）；其他欄位是同一個物件。"""
    return {k: v for k, v in P.items() if k not in ("full", "acc")}


def prep_key(XS, trace, rule):
    """準備的快取鍵。只有 nb: 紀錄（$X1_NB_DIR/<名稱>.npz，一個檔案就是全部輸入）；其他紀錄 → None（不快取）。
    P 只由紀錄檔、x1_nb.prepare_nb、x1_sim.prep（用到模組常數 BF）決定；p3_fvar 只換 constants() 與 OUTDIR，不影響 P。"""
    if not trace.startswith("nb:"):
        return None
    import x1_nb
    path = os.path.join(x1_nb.NB_DIR, f"{trace[3:]}.npz")
    parts = dict(trace=trace, rule=rule, nb_path=path, nb_sha256=file_sha(path),
                 code_sha256=_sha_files([XS.__file__, x1_nb.__file__, os.path.abspath(__file__)]),
                 numpy=np.__version__, python=sys.version.split()[0], BF=XS.BF)    # prep 的 ws_GiB 用到 BF
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest(), parts


def _prep_dir(key):
    return os.path.join(WORK, f"prep-{key[:24]}")


def _save_prep(key, parts, P, tsha):
    """lite P（pickle）＋KEY.json，寫進暫存目錄再搬到 prep-<鍵>/（同時有別的 sweep 寫好了就丟掉自己的）。"""
    tmp = tempfile.mkdtemp(prefix="prep-tmp-", dir=WORK)
    pk = os.path.join(tmp, "lite.pkl")
    with open(pk, "wb") as fh:
        pickle.dump(_lite(P), fh, protocol=5)
    info = dict(key=key, parts=parts, n_req=len(P["full"]), trace_sha256=tsha, lite_sha256=file_sha(pk),
                created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), host=socket.gethostname())
    with open(os.path.join(tmp, "KEY.json"), "w") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=1)
    dst = _prep_dir(key)
    if os.path.isdir(dst):
        shutil.rmtree(tmp)
    else:
        try:
            os.rename(tmp, dst)
        except OSError:
            shutil.rmtree(tmp)
            assert os.path.isdir(dst), dst


def _load_prep(XS, key, parts, trace, rule):
    """命中 → 驗證後放進 XS._PCACHE（lite P）與 _TDIR，回傳 (紀錄檔目錄, sha)；沒有這個目錄 → None。
    驗證：KEY.json 的鍵與組成相同、lite.pkl 的 sha256、紀錄檔重算 sha256、紀錄檔的 n／B／S 和 lite P 對得上。不過 → 停下來。"""
    d = _prep_dir(key)
    if not os.path.isdir(d):
        return None
    with open(os.path.join(d, "KEY.json")) as fh:
        info = json.load(fh)
    assert info["key"] == key and info["parts"] == parts, f"{d}: KEY.json 和這次的鍵對不上"
    pk = os.path.join(d, "lite.pkl")
    assert file_sha(pk) == info["lite_sha256"], f"{d}: lite.pkl 的 sha256 不對（刪掉這個目錄會重建）"
    tdir = os.path.join(WORK, f"trace-{info['trace_sha256'][:24]}")
    assert os.path.isdir(tdir), f"{d}: 紀錄檔目錄 {tdir} 不見了（刪掉 {d} 會重建）"
    tsha = _sha_files([os.path.join(tdir, f) for f in TRACE_FILES])
    assert tsha == info["trace_sha256"], f"{tdir}: 紀錄檔的 sha256 不對"
    with open(pk, "rb") as fh:          # 自己寫的檔（上面已驗 sha256）；內容只有 dict／list／numpy 陣列
        L = pickle.load(fh)
    n, B, S = np.fromfile(os.path.join(tdir, "meta.i64"), np.int64).tolist()
    assert n == info["n_req"] == len(L["tl"]) == len(L["nall"]) and B == int(L["B"]) and S == int(L["R"]["n_sess"]), \
        (d, n, B, S, info["n_req"])
    L["full"] = NoFull(n)
    XS._PCACHE[(trace, rule)] = L
    _TDIR[(trace, rule)] = tdir
    return tdir, tsha


def prepare_cached(XS, trace, rule, lite):
    """sweep --engine rust 的父行程準備：P（XS._PCACHE）＋Rust 的紀錄檔（_TDIR）。回傳 (紀錄檔目錄, sha, 怎麼來的, 秒)。
    lite＝這份紀錄的 job 全部走 Rust（沒有退回 Python Engine 的）：P 不留 full、acc。
      - nb: 紀錄、快取有 → 讀快取（cache_hit）；沒有 → 照舊建（XS.prep＋prepare_trace）並存快取（built_saved）。
    lite＝False（有 job 要退回 Python Engine，要完整的 P）→ 照舊建；nb: 紀錄的快取沒有就順便存。"""
    t0 = time.time()
    k = prep_key(XS, trace, rule) if PREP_CACHE else None
    if k is not None and lite:
        r = _load_prep(XS, k[0], k[1], trace, rule)
        if r is not None:
            return r + ("cache_hit", time.time() - t0)
    P = XS.prep(trace, rule)
    tdir, tsha = prepare_trace(P, trace, rule)
    how = "built"
    if k is not None:
        assert file_sha(k[1]["nb_path"]) == k[1]["nb_sha256"], f"{k[1]['nb_path']}：準備期間檔案被改了"
        if not os.path.isdir(_prep_dir(k[0])):
            _save_prep(k[0], k[1], P, tsha)
            how = "built_saved"
    if lite:
        L = _lite(P)
        L["full"] = NoFull(len(P["full"]))
        XS._PCACHE[(trace, rule)] = L
        del P
        gc.collect()
    return tdir, tsha, how, time.time() - t0


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


def write_job(XS, P, policy, cap_bytes, views, K, gkw, nz=None, pfx=False, nzo="clamp"):
    """一個 job 的輸入目錄（views.txt、f*／fc*、fc_idle.f64、i4.txt、一行的 jobs.txt）。f／Fc 直接取自 Python 的 View。
    nz＝W4 雜訊（x1_sim 的 noise spec；B1 或 B2，見 unsupported）→ jobs.txt 多一欄 nz=…（同 Engine(noise=nz)；λ＝0 也照送）。
    PREREG 偏離 14（SIM_CHANGES 22）：pfx＝True → 多一欄 `pfx=1`（Engine(pf_expire=True)）；nzo＝"resample" → 多一欄
    `nzo=resample`（Engine(noise_overdue="resample")）。沒開＝和原本相同的一行。"""
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
                          + [",".join(str(i) for i in range(len(views)))] + ([] if nz is None else [nz_field(nz)])
                          + (["pfx=1"] if pfx else []) + ([] if nzo == "clamp" else [f"nzo={nzo}"])) + "\n")
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
