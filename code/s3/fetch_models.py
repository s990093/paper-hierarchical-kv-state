#!/usr/bin/env python3
"""fetch_models.py — 第三階段（docs/stage3_design_20261024/）的模型權重：下載到 hgx-cn01 本地 /tmp、逐檔驗證、每個 job 開跑前檢查。

為什麼放 /tmp（使用者 2026-10-25）：/work 的 hf-cache 從 10/23 起是空的（Llama、Qwen-7B 都不在），群組配額只剩 49G；
權重是固定 revision 的公開檔，丟了可以重抓，所以全部放 compute node 本地的 /tmp（md0，不算配額）。
/tmp 的清理（head node 的設定，compute node 由 `fetch` 記錄下來核對）：systemd-tmpfiles 每天刪 30 天沒碰過的檔案、開機時整個清空。
所以每個 GPU job 開跑前跑 `check`：檔案大小、inode、mtime 和下載時的驗證戳記相同才算在；不在就 exit 3（runner 再呼叫 fetch 重抓）。

  fetch  [--models a b ...]   下載（huggingface_hub.snapshot_download，固定 revision）→ 每個檔案和 HF 的雜湊比對
                              （LFS 檔：sha256；一般檔：git blob sha1）→ 寫驗證戳記與 manifest（進 git 的小 JSON）
  check  [--models ...]       快速檢查（大小＋戳記）；--full 再重算 sha256
  env    [--models ...]       印 export 指令（HF_HOME、M7_*_GLOB）給 runner eval
  --self-test                 CPU、不連網：git blob 雜湊、戳記比對、manifest 欄位

規則：雜湊不對就停（規則 2），不重抓第二次；manifest 每列有 run_id、ts（CLAUDE.md §4.3）。
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ROOT = os.environ.get("S3_MODEL_ROOT", "/tmp/hungweilai-models/hf-cache")   # HF_HOME 的版面：<ROOT>/hub/models--org--name/snapshots/<rev>
MANIFEST_DIR = os.path.join(REPO, "results", "s3_h200", "env")
STAMP = ".s3_verified.json"

# 固定 revision（2026-10-25 從 HF API 讀；Llama、Qwen-7B 和第一、二階段同一版：env_h200.sh、migration_20261013 README §1.1）
MODELS = {
    "llama": dict(repo="unsloth/Llama-3.1-8B-Instruct", rev="4699cc75b550f9c6f3173fb80f4703b62d946aa5", glob_env="M7_LLAMA_GLOB"),
    "qwen": dict(repo="Qwen/Qwen2.5-7B-Instruct-1M", rev="e28526f7bb80e2a9c8af03b831a9af3812f18fba", glob_env="M7_QWEN_GLOB"),
    "qwen14": dict(repo="Qwen/Qwen2.5-14B-Instruct-1M", rev="620fad32de7bdd2293b3d99b39eba2fe63e97438", glob_env="M7_QWEN14_GLOB"),
    # D0（runs/D0_native_index.md；2026-10-25 從 HF API 讀）：原生稀疏注意力 InfLLM-v2
    "minicpm": dict(repo="openbmb/MiniCPM4.1-8B", rev="3a8dfed9c79a45e07dbff95bcd49d792343fa1a3", glob_env="M7_MINICPM_GLOB"),
}


def run_id() -> str:
    return os.environ.get("RUN_ID") or datetime.now().strftime("%Y%m%d-%H%M%S") + "-adhoc"


def ts() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def snap_dir(key: str, root: str = ROOT) -> str:
    m = MODELS[key]
    return os.path.join(root, "hub", "models--" + m["repo"].replace("/", "--"), "snapshots", m["rev"])


def git_blob_sha1(path: str) -> str:
    """git 的 blob id：sha1(b"blob <size>\\0" + 內容)。HF 對非 LFS 檔給的是這個。"""
    h = hashlib.sha1()
    h.update(b"blob %d\0" % os.path.getsize(path))
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def expected_files(key: str) -> list:
    """HF API 的檔案清單（名稱、大小、雜湊種類與值）。要連網。"""
    from huggingface_hub import HfApi
    m = MODELS[key]
    info = HfApi().model_info(m["repo"], revision=m["rev"], files_metadata=True)
    if info.sha != m["rev"]:
        raise SystemExit(f"{m['repo']}：API 回的 revision {info.sha} ≠ 固定的 {m['rev']}（規則 2：停）")
    out = []
    for s in info.siblings:
        if s.lfs:
            out.append(dict(name=s.rfilename, size=s.size, kind="sha256", hash=s.lfs.sha256))
        else:
            out.append(dict(name=s.rfilename, size=s.size, kind="git_sha1", hash=s.blob_id))
    return sorted(out, key=lambda r: r["name"])


def verify_one(snap: str, f: dict) -> dict:
    p = os.path.join(snap, f["name"])
    if not os.path.exists(p):
        return dict(f, ok=False, why="missing")
    size = os.path.getsize(p)
    if size != f["size"]:
        return dict(f, ok=False, why=f"size {size} ≠ {f['size']}")
    got = sha256_file(p) if f["kind"] == "sha256" else git_blob_sha1(p)
    return dict(f, ok=(got == f["hash"]), got=got, why="" if got == f["hash"] else "hash")


def file_ids(snap: str, names: list) -> dict:
    """戳記：每個檔案（跟著 symlink 到 blob）的大小、inode、mtime。check 只比大小與 inode（被清掉重抓、被截斷都會變）；
    不比 mtime：compute node 的 /tmp 是 noatime 掛載，tmpfiles 30 天的清理只看 mtime／ctime，所以每個 job 會 refresh（touch）一次。"""
    out = {}
    for n in names:
        st = os.stat(os.path.join(snap, n))
        out[n] = [st.st_size, st.st_ino, int(st.st_mtime)]
    return out


def same_ids(a: dict, b: dict) -> list:
    """不同的檔名（只比 [大小, inode]）。"""
    return [n for n in b if list(a.get(n, [None, None]))[:2] != list(b[n])[:2]]


def node_tmp_policy() -> dict:
    """compute node 上 /tmp 的清理設定（head node 是 D /tmp 1777 root root 30d、每天的 timer）：照實記下來。"""
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as ex:   # noqa: BLE001 — 記錄失敗本身（規則 7：查不到要讓人知道）
            return f"QUERY_FAILED: {ex!r}"
    return dict(host=platform.node(),
                tmpfiles_tmp_conf=sh("grep -h -v '^#' /usr/lib/tmpfiles.d/tmp.conf /etc/tmpfiles.d/tmp.conf 2>&1 | grep -v '^$'"),
                etc_tmpfiles_d=sh("ls /etc/tmpfiles.d/ 2>&1"),
                clean_timer=sh("systemctl list-timers systemd-tmpfiles-clean.timer --no-pager 2>&1 | head -2"),
                uptime=sh("uptime -s 2>&1"), df_tmp=sh("df -h /tmp 2>&1 | tail -1"), mount_tmp=sh("findmnt -no SOURCE,FSTYPE,OPTIONS -T /tmp 2>&1"))


def cmd_fetch(a):
    from huggingface_hub import snapshot_download
    rid, rows = run_id(), []
    os.makedirs(os.path.join(a.root, "hub"), exist_ok=True)
    lock = open(os.path.join(a.root, ".fetch.lock"), "w")          # 同一台機器上兩個 job 不同時下載（等前一個做完）
    t_lock = time.time()
    fcntl.flock(lock, fcntl.LOCK_EX)
    print(f"[fetch] 拿到下載鎖（等了 {time.time() - t_lock:.0f}s）", flush=True)
    free = shutil.disk_usage(a.root).free
    print(f"[fetch] root={a.root} free={free / 2**30:.0f} GiB", flush=True)
    for key in a.models:
        m = MODELS[key]
        exp = expected_files(key)
        need = sum(f["size"] for f in exp)
        if free < need * 1.1:
            raise SystemExit(f"{a.root} 只剩 {free / 2**30:.1f} GiB，{key} 要 {need / 2**30:.1f} GiB（規則 2：停）")
        t0 = time.time()
        p = snapshot_download(m["repo"], revision=m["rev"], cache_dir=os.path.join(a.root, "hub"), max_workers=8)
        t_dl = time.time() - t0
        snap = snap_dir(key, a.root)
        if os.path.realpath(p) != os.path.realpath(snap):
            raise SystemExit(f"snapshot_download 回 {p}，預期 {snap}（規則 2：停）")
        t1 = time.time()
        with ThreadPoolExecutor(max_workers=8) as ex:
            res = list(ex.map(lambda f: verify_one(snap, f), exp))
        bad = [r for r in res if not r["ok"]]
        if bad:
            raise SystemExit(f"{key}：{len(bad)} 個檔案驗證沒過：{[(r['name'], r['why']) for r in bad]}（規則 2：停，不重抓）")
        stamp = dict(model=key, repo=m["repo"], rev=m["rev"], run_id=rid, ts=ts(), files=exp,
                     ids=file_ids(snap, [f["name"] for f in exp]))
        json.dump(stamp, open(os.path.join(snap, STAMP), "w"), indent=1)
        row = dict(run_id=rid, ts=ts(), model=key, repo=m["repo"], rev=m["rev"], path=snap, n_files=len(exp),
                   bytes=need, gib=round(need / 2**30, 3), t_download_s=round(t_dl, 1), t_verify_s=round(time.time() - t1, 1),
                   files=[dict(name=r["name"], size=r["size"], kind=r["kind"], hash=r["hash"]) for r in res])
        rows.append(row)
        print(f"[fetch] {key} {need / 2**30:.2f} GiB 下載 {t_dl:.0f}s、驗證 {row['t_verify_s']}s、{len(exp)} 個檔案全對 → {snap}", flush=True)
    doc = dict(run_id=rid, ts=ts(), root=a.root, node=node_tmp_policy(), models=rows,
               note="權重在 hgx-cn01 本地 /tmp（不算配額、不保證保留）；每個 job 開跑前用 `fetch_models.py check` 核對戳記。")
    os.makedirs(a.manifest_dir, exist_ok=True)
    out = os.path.join(a.manifest_dir, f"models_manifest_{rid}.json")
    json.dump(doc, open(out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(doc["node"], ensure_ascii=False, indent=1))
    print(f"[fetch] manifest → {out}")


def check_one(key: str, root: str, full: bool) -> tuple:
    snap = snap_dir(key, root)
    sp = os.path.join(snap, STAMP)
    if not os.path.exists(sp):
        return False, f"{key}：沒有 {sp}（沒下載過或被清掉）"
    stamp = json.load(open(sp))
    if stamp.get("rev") != MODELS[key]["rev"] or stamp.get("repo") != MODELS[key]["repo"]:
        return False, f"{key}：戳記的 repo／rev 對不上"
    names = [f["name"] for f in stamp["files"]]
    try:
        ids = file_ids(snap, names)
    except FileNotFoundError as ex:
        return False, f"{key}：檔案不見了（{ex.filename}）"
    diff = same_ids(ids, stamp["ids"])
    if diff:
        return False, f"{key}：{len(diff)} 個檔案的大小／inode 和戳記不同：{diff[:4]}"
    if full:
        bad = [r for r in (verify_one(snap, f) for f in stamp["files"]) if not r["ok"]]
        if bad:
            return False, f"{key}：重算雜湊不對：{[(r['name'], r['why']) for r in bad]}"
    return True, f"{key}：OK（{snap}，下載 run {stamp['run_id']}）"


def cmd_check(a):
    ok_all = True
    for key in a.models:
        ok, msg = check_one(key, a.root, a.full)
        print(("[check] " if ok else "[check] ✗ ") + msg, flush=True)
        ok_all &= ok
    sys.exit(0 if ok_all else 3)


def refresh_one(key: str, root: str) -> int:
    """touch 這個模型的每個檔（blob 與 snapshot 裡的 symlink）、戳記、目錄：/tmp 是 noatime，tmpfiles 的 30 天看 mtime。"""
    snap = snap_dir(key, root)
    stamp = json.load(open(os.path.join(snap, STAMP)))
    n = 0
    for f in stamp["files"]:
        p = os.path.join(snap, f["name"])
        os.utime(p)                                            # 跟著 symlink：blob
        os.utime(p, follow_symlinks=False)                     # symlink 自己
        n += 1
    os.utime(os.path.join(snap, STAMP))
    d = snap
    while os.path.realpath(d) != os.path.realpath(root) and d != "/":
        os.utime(d)
        d = os.path.dirname(d)
    return n


def cmd_refresh(a):
    for key in a.models:
        ok, msg = check_one(key, a.root, False)
        if not ok:
            print("[refresh] ✗ " + msg, flush=True)
            sys.exit(3)
        print(f"[refresh] {key}：touch {refresh_one(key, a.root)} 個檔", flush=True)


def cmd_env(a):
    print(f"export HF_HOME={a.root}")
    for key in a.models:
        print(f"export {MODELS[key]['glob_env']}={snap_dir(key, a.root)}")


def self_test() -> bool:
    ok = True
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.txt")
        open(p, "wb").write(b"hello\n")
        ok &= git_blob_sha1(p) == "ce013625030ba8dba906f756967f9e9ca394464a"   # `git hash-object` 的已知值
        ok &= sha256_file(p) == hashlib.sha256(b"hello\n").hexdigest()
        # 戳記：假的一個 snapshot
        global MODELS
        saved = MODELS
        MODELS = dict(t=dict(repo="org/name", rev="r1", glob_env="M7_T_GLOB"))
        try:
            snap = snap_dir("t", d)
            os.makedirs(snap)
            f1 = os.path.join(snap, "a.json")
            open(f1, "w").write("{}")
            exp = [dict(name="a.json", size=2, kind="git_sha1", hash=git_blob_sha1(f1))]
            ok &= verify_one(snap, exp[0])["ok"]
            json.dump(dict(repo="org/name", rev="r1", run_id="x", files=exp, ids=file_ids(snap, ["a.json"])),
                      open(os.path.join(snap, STAMP), "w"))
            ok &= check_one("t", d, True)[0]
            os.utime(f1, (0, 0))                                   # mtime 變了（refresh）→ 還是算在
            ok &= check_one("t", d, False)[0]
            ok &= refresh_one("t", d) == 1 and check_one("t", d, True)[0]
            open(f1, "w").write("{}")                               # 同一個 inode 重寫（大小一樣）→ check 抓不到，--full 抓得到雜湊（這裡內容相同所以仍過）
            ok &= check_one("t", d, True)[0]
            os.remove(f1)
            ok &= not check_one("t", d, False)[0]
            ok &= "missing" in verify_one(snap, exp[0])["why"]
        finally:
            MODELS = saved
    ok &= all(set(m) == {"repo", "rev", "glob_env"} and len(m["rev"]) == 40 for m in MODELS.values())
    print("self-test", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    for name in ("fetch", "check", "env", "refresh"):
        s = sub.add_parser(name)
        s.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
        s.add_argument("--root", default=ROOT)
        if name == "fetch":
            s.add_argument("--manifest-dir", default=MANIFEST_DIR)
        if name == "check":
            s.add_argument("--full", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    {"fetch": cmd_fetch, "check": cmd_check, "env": cmd_env, "refresh": cmd_refresh}[a.cmd](a)


if __name__ == "__main__":
    main()
