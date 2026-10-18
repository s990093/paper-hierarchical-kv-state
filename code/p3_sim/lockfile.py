"""lockfile.py — 04 §9 的「凍結鎖」：解析與驗證（04 v4 第二輪審查 #5；06 §25）。

04_prereg_map.md §9 在凍結的 commit 裡要有下面三樣東西（格式固定，程式用正規表示式解析；沒有就拒絕跑）：

  <!-- LOCK:BEGIN -->
  | 路徑 | sha256 |
  |:--|:--|
  | `results/p3_formats/format_table.json` | `<64 位十六進位>` |
  ...（REQUIRED_INPUTS 列的每一個檔都要有；多列也可以）
  <!-- LOCK:END -->
  程式 commit（凍結）：`<40 位十六進位>`
  護欄的參考：C0            （或：護欄的參考：min(C0, O)）

驗證（verify_freeze；run_map.py 對真實資料的工作一律先過這一關）：
  1. 凍結的 commit 裡的 04 不含「未凍結」。
  2. 鎖定表含 REQUIRED_INPUTS 的每一個檔；工作目錄裡每個列出的檔的 sha256 都等於表上的值。
  3. 程式 commit：HEAD＝它，或 HEAD 是它的後代而且 `code/p3_sim/` 底下從它到 HEAD 沒有任何檔案不同（只有文件、results 等的 commit 不影響）。
     凍結的 commit 本身必須是 HEAD 的祖先（或＝HEAD）。
  4. 工作目錄乾淨：沒有任何已追蹤檔案的修改（`git status --porcelain --untracked-files=no` 為空），而且 `code/p3_sim/` 底下沒有未追蹤的檔案。
任何一項不成立 → SystemExit（訊息列出哪一項）。
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess

PREREG = "docs/phase3_20261017/04_prereg_map.md"
CODE_DIR = "code/p3_sim"
REQUIRED_INPUTS = [
    "results/p3_formats/format_table.json",
    "code/p3_sim/config/costs_mi300x.json",
    "code/p3_sim/config/costs_3090.json",
    "results/p3_sim/kgpu_kmin_all.csv",
    "results/p3_sim/return_gaps.csv",
    "results/p3_sim/session_counts.csv",
    "results/p3_calib_3090/cpubusy_slowdown.csv",
    "results/p3_calib_3090/constants_3090_v2.json",
]
GUARD_REFS = {"C0": "C0", "min(C0, O)": "min_C0_O"}


def _git(repo, *args, check=True):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} 失敗（exit {r.returncode}）：{r.stderr.strip()}")
    return r


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_lock(text):
    """回傳 dict(files={路徑: sha256}, code_commit=str|None)。沒有 LOCK 區塊 → files＝{}。"""
    m = re.search(r"<!--\s*LOCK:BEGIN\s*-->(.*?)<!--\s*LOCK:END\s*-->", text, re.S)
    files = {}
    if m:
        for line in m.group(1).split("\n"):
            r = re.match(r"^\|\s*`?([^|`\s]+)`?\s*\|\s*`?([0-9a-f]{64})`?\s*\|\s*$", line)
            if r:
                if r.group(1) in files and files[r.group(1)] != r.group(2):
                    raise SystemExit(f"鎖定表裡 {r.group(1)} 出現兩個不同的 sha256")
                files[r.group(1)] = r.group(2)
    c = re.search(r"^程式 commit（凍結）：\s*`?([0-9a-f]{40})`?\s*$", text, re.M)
    return dict(files=files, code_commit=c.group(1) if c else None)


def parse_guard_ref(text):
    """04 §9 的 `護欄的參考：C0`｜`護欄的參考：min(C0, O)` → 'C0'｜'min_C0_O'；沒有這一行 → None。"""
    m = re.search(r"^護欄的參考：\s*(C0|min\(C0,\s*O\))\s*$", text, re.M)
    if not m:
        return None
    return GUARD_REFS["C0" if m.group(1) == "C0" else "min(C0, O)"]


def verify_freeze(frozen, repo, prereg=PREREG, required=None, text=None):
    """見模組說明。frozen＝凍結的 commit（任何 git 認得的名字）。回傳 dict(frozen, prereg_sha256, files, code_commit, head, guard_ref)。"""
    required = REQUIRED_INPUTS if required is None else required
    full = _git(repo, "rev-parse", frozen).stdout.strip()
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    if text is None:
        text = _git(repo, "show", f"{full}:{prereg}").stdout
    errs = []
    if "未凍結" in text:
        errs.append(f"{prereg} 在 {full[:8]} 還含「未凍結」")
    lk = parse_lock(text)
    if not lk["files"]:
        errs.append("04 §9 沒有 `<!-- LOCK:BEGIN -->…<!-- LOCK:END -->` 鎖定表（或表裡沒有任何 路徑｜sha256 列）")
    miss = [p for p in required if p not in lk["files"]]
    if lk["files"] and miss:
        errs.append(f"鎖定表缺這些必要的輸入檔：{miss}")
    for p, want in lk["files"].items():
        fp = os.path.join(repo, p)
        if not os.path.exists(fp):
            errs.append(f"鎖定表列的檔不存在：{p}")
        elif sha256_file(fp) != want:
            errs.append(f"sha256 不符：{p}（表 {want[:12]}…，實際 {sha256_file(fp)[:12]}…）")
    cc = lk["code_commit"]
    if cc is None:
        errs.append("04 §9 沒有「程式 commit（凍結）：<40 位十六進位>」那一行")
    else:
        if _git(repo, "cat-file", "-e", f"{cc}^{{commit}}", check=False).returncode != 0:
            errs.append(f"程式 commit {cc[:8]} 不在這個 repo")
        elif _git(repo, "merge-base", "--is-ancestor", cc, "HEAD", check=False).returncode != 0:
            errs.append(f"程式 commit {cc[:8]} 不是 HEAD {head[:8]} 的祖先")
        else:
            d = _git(repo, "diff", "--name-only", cc, "HEAD", "--", CODE_DIR).stdout.split()
            if d:
                errs.append(f"{CODE_DIR}/ 從凍結的程式 commit {cc[:8]} 到 HEAD {head[:8]} 有修改：{d[:6]}")
    if _git(repo, "merge-base", "--is-ancestor", full, "HEAD", check=False).returncode != 0:
        errs.append(f"凍結的 commit {full[:8]} 不是 HEAD {head[:8]} 的祖先")
    dirty = _git(repo, "status", "--porcelain", "--untracked-files=no").stdout.strip()
    if dirty:
        errs.append("工作目錄不乾淨（已追蹤的檔案有未提交的修改）：" + "；".join(dirty.split("\n")[:6]))
    untr = _git(repo, "ls-files", "--others", "--exclude-standard", CODE_DIR).stdout.split()
    if untr:
        errs.append(f"{CODE_DIR}/ 底下有未追蹤的檔案：{untr[:6]}")
    if errs:
        raise SystemExit("凍結檢查失敗，拒絕跑真實資料：\n  - " + "\n  - ".join(errs))
    return dict(frozen=full, prereg_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(), files=lk["files"], code_commit=cc, head=head,
                guard_ref=parse_guard_ref(text))
