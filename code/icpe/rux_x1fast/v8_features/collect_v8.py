"""把 v8 探索的結果收進 repo：results/icpe_next/v8_explore/<實驗>/<變體>/（只放彙總，>1 MB 用 xz）。
用法：python -I collect_v8.py <實驗> <變體> --sweep <x1_sweep.csv>[=<存成的名字>] ... --files <分析輸出> ... --runs <run 目錄> ...
  實驗＝apc｜w3｜w5｜queue｜slowcap；變體＝例如 semianalysis_N16_s1、tracelab_c1、fi_2026-05-17_2026-05-23、mooncake_toolagent
  --sweep：x1_sweep.csv（存成 <名字>.csv.xz；預設名字 x1_sweep）；--files：分析的 CSV／JSON（>1 MB 才壓）；
  --runs：run 目錄（讀 context.txt 的 ts、host）。寫 MANIFEST.json：run_id、ts、host、repo commit、v8 main.rs 的 sha256、
  sweep 裡的 code_sha、每個檔的 sha256 與列數。不放：原始資料、nb 檔、cue.u8、逐請求 npz、job 目錄。"""
import argparse, hashlib, json, lzma, os, shutil, socket, subprocess, sys
from datetime import datetime

ap = argparse.ArgumentParser()
ap.add_argument("exp", choices=["apc", "w3", "w5", "queue", "slowcap"])
ap.add_argument("variant")
ap.add_argument("--sweep", nargs="*", default=[])
ap.add_argument("--files", nargs="*", default=[])
ap.add_argument("--runs", nargs="*", default=[])
ap.add_argument("--note", default="")
a = ap.parse_args()
REPO = os.environ["REPO"]
out = os.path.join(REPO, "results", "icpe_next", "v8_explore", a.exp, a.variant)
os.makedirs(out, exist_ok=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def put(src, name, force_xz=False):
    size = os.path.getsize(src)
    if force_xz or size > 1 << 20:
        dst = os.path.join(out, name + ".xz")
        with open(src, "rb") as fi, lzma.open(dst, "wb", preset=9) as fo:
            shutil.copyfileobj(fi, fo)
    else:
        dst = os.path.join(out, name)
        shutil.copyfile(src, dst)
    nrows = sum(1 for _ in open(src, encoding="utf-8", errors="replace")) - 1 if src.endswith(".csv") else None
    return dict(file=os.path.basename(dst), source=os.path.abspath(src), sha256_source=sha(src), bytes_source=size, rows=nrows)


files = []
code_sha = set()
for s in a.sweep:
    src, _, nm = s.partition("=")
    nm = nm or "x1_sweep"
    files.append(put(src, nm + ".csv", force_xz=True))
    import csv
    for r in csv.DictReader(open(src)):
        code_sha.add(r.get("code_sha", ""))
for f in a.files:
    files.append(put(f, os.path.basename(f)))
runs = []
for d in a.runs:
    ctx = open(os.path.join(d, "context.txt")).read().splitlines()
    runs.append(dict(run_id=os.path.basename(d.rstrip("/")), ts=ctx[0].strip(), host=ctx[1].strip()))
v8 = os.path.join(REPO, "code/icpe/rux_x1fast/v8_features/src/main.rs")
m = dict(exp=a.exp, variant=a.variant, collected_ts=datetime.now().astimezone().isoformat(timespec="seconds"), collected_host=socket.gethostname(),
         repo_commit=subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
         v8_main_rs_sha256=sha(v8), sweep_code_sha=sorted(code_sha), runs=runs, files=files, note=a.note,
         not_in_git="原始資料、nb 檔、cue.u8、逐請求 npz、job 目錄")
json.dump(m, open(os.path.join(out, "MANIFEST.json"), "w"), ensure_ascii=False, indent=1)
print(f"→ {out}：{len(files)} 個檔", flush=True)
