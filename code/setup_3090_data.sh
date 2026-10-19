#!/usr/bin/env bash
# =============================================================================
# setup_3090_data.sh — 平台 A（3090）第一次：下載不在 git 裡的資料，逐一檢查雜湊、寫 README.txt（2026-10-17）
#
#   用法：  bash code/setup_3090_data.sh                 # 全部
#           bash code/setup_3090_data.sh mooncake scbench  # 只做這幾項
#   項目：hotpotqa longbench mooncake cachewise bailian burstgpt azure2023 azure2024 scbench wildchat lmsys
#
# 規則：
#   - 每份資料放在 $TIARA_DATASETS 底下自己的新目錄；目錄已存在且非空 → 跳過（不覆寫、不刪）。
#   - 版本全部釘死（commit／revision／release tag），雜湊對不上 → 該項 FAILED、停下來回報；不准改 checksum（CLAUDE.md 規則 2）。
#   - 下載的東西是不受信任的資料：不執行裡面的程式；讀它的 Python 一律 python -I，從 stdin 讀程式。
#   - 匯入 trace 時用資料自身交叉驗證單位（CLAUDE.md 規則 6）：len(hash_ids)==ceil(input_length/區塊大小)。
#   - 某一項失敗不影響其他項；最後印每項的 OK／FAILED／NEEDS_USER，有任何 FAILED 就 exit 1。
# =============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env_3090.sh"
D="$TIARA_DATASETS"
mkdir -p "$D"
PY="$TIARA_VENV/bin/python"
step() { echo "=== $(date -Is) $*"; }
fresh() { if [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; then echo "[skip] $1 已存在且非空"; return 1; fi; mkdir -p "$1"; }
get() { curl -fL --retry 3 --retry-delay 5 -sS -o "$2" "$1"; }                       # $1=url $2=out
sha_ok() { echo "$2  $1" | sha256sum -c - ; }                                         # $1=file $2=sha256
blob_ok() { local g; g=$(git hash-object "$1"); [ "$g" = "$2" ] && echo "$1: git-blob OK ($g)" || { echo "$1: git-blob MISMATCH have=$g want=$2"; return 1; }; }

# README.txt：$1=目錄、stdin=表頭（來源、版本、license…）；後面附上每個檔的 sha256 與大小（略過 .git、.cache）
readme() {
  local d=$1
  { cat; echo "download_date: $(date -Is)"; echo "downloaded_by: code/setup_3090_data.sh @ $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)"
    "$PY" -I - "$d" <<'PY'
import hashlib, os, sys
root = sys.argv[1]
rows, tot = [], 0
for dp, dns, fns in os.walk(root):
    dns[:] = sorted(x for x in dns if x not in (".git", ".cache"))
    for fn in sorted(fns):
        p = os.path.join(dp, fn)
        rel = os.path.relpath(p, root)
        if rel == "README.txt" or os.path.islink(p):
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        n = os.path.getsize(p)
        tot += n
        rows.append(f"  {h.hexdigest()}  {n:>13}  {rel}")
print(f"total_bytes: {tot}  ({tot / 2**30:.3f} GiB, {len(rows)} files; .git/.cache 不計)")
print("files (sha256  bytes  path):")
print("\n".join(rows))
PY
  } > "$d/README.txt"
}

# 單位驗算（規則 6）：$1=jsonl $2=每個 hash_id 的 token 數。印吻合率與幾個描述量；吻合率 < 0.95 → 失敗
unit_check() {
  "$PY" -I - "$1" "$2" <<'PY'
import json, math, sys
path, blk = sys.argv[1], int(sys.argv[2])
n = match = 0
keys = None
tss, inl = [], []
for line in open(path):
    r = json.loads(line)
    if keys is None:
        keys = sorted(r)
    n += 1
    match += len(r["hash_ids"]) == math.ceil(r["input_length"] / blk)
    tss.append(r["timestamp"]); inl.append(r["input_length"])
rate = match / n
inl.sort()
print(f"unit_check {path.rsplit('/', 1)[-1]}: block={blk} n_req={n} match={match} rate={rate:.6f} "
      f"ts_sorted={tss == sorted(tss)} ts_min={min(tss)} ts_max={max(tss)} "
      f"input_len_median={inl[n // 2]} input_len_max={inl[-1]} fields={keys}")
sys.exit(0 if rate >= 0.95 else 1)
PY
}

# ---------------------------------------------------------------------------------------------------
do_hotpotqa() {   # 同 h200_setup_data.sh
  local d="$D/hotpotqa" f="$D/hotpotqa/hotpot_qa_distractor_validation.parquet"
  local url=https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/1908d6afbbead072334abe2965f91bd2709910ab/distractor/validation-00000-of-00001.parquet
  local sha=c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6
  fresh "$d" || return 0
  get "$url" "$f" && sha_ok "$f" "$sha" || return 1
  readme "$d" <<EOF
dataset: HotpotQA dev distractor（m10_tasks T4；M10_HOTPOT）
source: $url
revision: 1908d6afbbead072334abe2965f91bd2709910ab（HF hotpotqa/hotpot_qa）
expected_sha256: $sha（code/h200_setup_data.sh，MI300X datasets/hotpotqa/README.txt 同一份）
license: CC BY-SA 4.0（HF dataset card）
EOF
}

do_longbench() {  # 同 h200_setup_data.sh
  local d="$D/longbench" url=https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip
  local sha=cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64
  fresh "$d" || return 0
  local commit; commit=$(curl -sIL "$url" | tr -d '\r' | awk -F': ' 'tolower($1)=="x-repo-commit"{print $2}' | tail -1)
  get "$url" "$d/data.zip" && sha_ok "$d/data.zip" "$sha" || return 1
  "$PY" -I -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$d/data.zip" "$d" || return 1
  mkdir -p "$d/config" && cp "$HERE"/longbench_config/*.json "$d/config/" && cp "$HERE"/longbench_config/metrics.py "$HERE"/longbench_config/pred.py "$d/" || return 1
  echo "data/ 有 $(ls "$d/data" | wc -l) 個檔：$(ls "$d/data" | head -3 | tr '\n' ' ')…"
  readme "$d" <<EOF
dataset: LongBench v1 data.zip（m10_longbench；M10_LONGBENCH）
source: $url（THUDM/LongBench 已轉址到 zai-org/LongBench）
revision: main @ 下載當下 x-repo-commit=${commit:-NOT_RECORDED}
expected_sha256(data.zip): $sha（MI300X 2026-09-15 下載的那份；code/h200_setup_data.sh）
config/、metrics.py、pred.py：從 repo 的 code/longbench_config/ 複製（上游 LongBench 的檔，自測對拍用）
license: MIT（LongBench GitHub repo）；各子資料集沿用原始來源的授權
EOF
}

do_mooncake() {
  local d="$D/mooncake" c=d750cddafa4b62387ca135949595008c51e05815
  local B="https://raw.githubusercontent.com/kvcache-ai/Mooncake/$c"
  fresh "$d" || return 0
  mkdir -p "$d/traces" "$d/arxiv-trace" || return 1
  # git blob sha1：GitHub tree API（recursive）於 commit $c 的值
  local -a F=(
    "FAST25-release/traces/conversation_trace.jsonl traces/conversation_trace.jsonl"
    "FAST25-release/traces/synthetic_trace.jsonl traces/synthetic_trace.jsonl"
    "FAST25-release/traces/toolagent_trace.jsonl traces/toolagent_trace.jsonl"
    "FAST25-release/arxiv-trace/mooncake_trace.jsonl arxiv-trace/mooncake_trace.jsonl"
    "FAST25-release/README.md FAST25-README.md"
    "LICENSE-APACHE LICENSE-APACHE"
  )
  local tree; tree=$(curl -fsS "https://api.github.com/repos/kvcache-ai/Mooncake/git/trees/$c?recursive=1") || return 1
  local e src dst want
  for e in "${F[@]}"; do
    src=${e%% *}; dst=${e##* }
    want=$(printf '%s' "$tree" | "$PY" -I -c "import json,sys; t=json.load(sys.stdin)['tree']; print([x['sha'] for x in t if x['path']==sys.argv[1]][0])" "$src") || return 1
    get "$B/$src" "$d/$dst" && blob_ok "$d/$dst" "$want" || return 1
  done
  local uc=""
  for f in traces/conversation_trace.jsonl traces/synthetic_trace.jsonl traces/toolagent_trace.jsonl arxiv-trace/mooncake_trace.jsonl; do
    uc+="$(unit_check "$d/$f" 512)"$'\n' || { echo "$uc"; echo "[mooncake] $f 單位驗算失敗"; return 1; }
  done
  echo "$uc"
  # repo 原本的載入器（m8_trace_oracle.load_trace：單位、timestamp 排序、前綴鏈）照樣跑一次
  local m8; m8=$("$PY" -I - "$REPO/code" "$d/traces" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import m8_trace_oracle as m
m.TRACE_DIR = sys.argv[2]
for name in ("conversation", "synthetic", "toolagent"):
    _, meta = m.load_trace(name)
    print("m8_trace_oracle.load_trace", meta)
PY
) || { echo "$m8"; echo "[mooncake] m8_trace_oracle.load_trace 的斷言失敗"; return 1; }
  echo "$m8"
  readme "$d" <<EOF
dataset: Mooncake traces（FAST'25 release 三份＋arxiv 原始 trace）
source: https://github.com/kvcache-ai/Mooncake/tree/$c/FAST25-release
commit: $c（main，2026-10-17 查）
integrity: 每個檔的 git blob sha1 和 GitHub tree API 在該 commit 的值相同（下載時逐檔比對）
license: Apache-2.0（repo 根目錄 LICENSE-APACHE，已一併下載）
layout: traces/{conversation,synthetic,toolagent}_trace.jsonl 對應 m8_trace_oracle.py 的 TRACE_DIR/{name}_trace.jsonl
unit (CLAUDE.md 規則 6)：hash_ids 一個＝512 token；len(hash_ids)==ceil(input_length/512) 的吻合率：
$uc
$m8
EOF
}

do_cachewise() {
  local d="$D/cachewise-coding-traces" c=181c435a090d328d00bbbee4c8eeb27d32f3abd2
  fresh "$d" || return 0
  GIT_LFS_SKIP_SMUDGE=1 git clone -q https://github.com/cachewise-project/cachewise-coding-traces.git "$d/repo" || return 1
  git -C "$d/repo" -c advice.detachedHead=false checkout -q "$c" || return 1
  [ "$(git -C "$d/repo" rev-parse HEAD)" = "$c" ] || return 1
  local lfs; lfs=$(git -C "$d/repo" lfs ls-files 2>/dev/null | wc -l)
  readme "$d" <<EOF
dataset: CacheWise coding traces（真實 Claude Code 使用者的 trace，已去識別化；94 sessions）
source: https://github.com/cachewise-project/cachewise-coding-traces
commit: $c（main，2026-10-17 查）；git clone 在 repo/（git 本身驗證物件雜湊）
git-lfs files: $lfs
license: repo 裡沒有 LICENSE 檔，GitHub API 的 license 欄是 null → 預設著作權保留；
         用在論文前要向作者確認授權（NEEDS_USER）。repo 內附的 .py 是上游程式，未執行、不要執行。
layout: repo/raw_traces/project_*/session_*.jsonl（原始）、repo/parsed_traces/…/events.json、repo/tool_duration_prediction/datasets/
EOF
}

do_bailian() {
  local d="$D/qwen-bailian-usagetraces-anon" c=5f7439c51ec248a0c585f7d90a41a6f57773b912
  local R0="https://raw.githubusercontent.com/alibaba-edu/qwen-bailian-usagetraces-anon/$c"
  local M0="https://media.githubusercontent.com/media/alibaba-edu/qwen-bailian-usagetraces-anon/$c"
  fresh "$d" || return 0
  local f ptr oid uc=""
  for f in README.md LICENSE docs/qa-context-growth-pattern.md; do mkdir -p "$d/$(dirname "$f")"; get "$R0/$f" "$d/$f" || return 1; done
  for f in qwen_traceA_blksz_16.jsonl qwen_traceB_blksz_16.jsonl qwen_thinking_blksz_16.jsonl qwen_coder_blksz_16.jsonl; do
    ptr=$(curl -fsS "$R0/$f") || return 1                                  # git-lfs pointer（oid sha256 + size）
    oid=$(printf '%s\n' "$ptr" | awk '$1=="oid"{sub("sha256:","",$2); print $2}')
    [ -n "$oid" ] || { echo "[bailian] $f 不是 LFS pointer：$ptr"; return 1; }
    get "$M0/$f" "$d/$f" && sha_ok "$d/$f" "$oid" || return 1
    uc+="$(unit_check "$d/$f" 16)"$'\n' || { echo "$uc"; echo "[bailian] $f 單位驗算失敗"; return 1; }
  done
  echo "$uc"
  readme "$d" <<EOF
dataset: Qwen-Bailian anonymized usage traces（To-C traceA、To-B traceB、thinking、coder）
source: https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon
commit: $c（main，2026-10-17 查）
integrity: jsonl 是 git-lfs 物件，從 media.githubusercontent.com 下載，sha256 和該 commit 的 LFS pointer oid 相同
license: Apache-2.0（repo 的 LICENSE，已一併下載）
fields: chat_id, parent_chat_id, timestamp（秒，trace 相對）, input_length, output_length, type（text/search/image/file）, turn, hash_ids
unit (CLAUDE.md 規則 6)：README 說 hash_ids 一個＝16 token（blksz_16）；用 input_length 驗算：
$uc
EOF
}

do_burstgpt() {
  local d="$D/burstgpt" c=d895a53bb7b8ec137d0d2fe203b335835a78c10a tag=v2.0
  local base="https://github.com/HPMLL/BurstGPT/releases/download/$tag"
  fresh "$d" || return 0
  # sha256：GitHub release API 的 asset digest（2026-10-17 查）
  local -a A=(
    "BurstGPT_1.csv 4bb3783693d0a435686fbfc885615d2349bd067239079fa4b749f2e679e12122"
    "BurstGPT_2.csv 44bf5942b03fca42c01a226a545ad3a750ec62688faecd67b6831c56a8928bf7"
    "BurstGPT_3.csv 2299986a07388aa303ec2c41d1131e756db650a39ed6ef9dfe7cc3d7f9a43b8f"
    "BurstGPT_without_fails_1.csv a4d068a7113ec0290e74063a1b3447dc6001a30e4298eb313581b71006dda1f4"
    "BurstGPT_without_fails_2.csv 56193aa9b2bb26128ded43d2d29a960df6bf5af062bcfc9b005f3fcaa4e6e501"
    "BurstGPT_without_fails_3.csv 3326259f9efb11845bc5ef85fa97e6f691050b0974621c91ef22acd566c43a40"
  )
  local e
  for e in "${A[@]}"; do get "$base/${e%% *}" "$d/${e%% *}" && sha_ok "$d/${e%% *}" "${e##* }" || return 1; done
  get "https://raw.githubusercontent.com/HPMLL/BurstGPT/$c/LICENSE" "$d/LICENSE" || return 1
  get "https://raw.githubusercontent.com/HPMLL/BurstGPT/$c/README.md" "$d/README.md" || return 1
  local hdr; hdr=$(head -1 "$d/BurstGPT_1.csv" | tr -d '\r')
  readme "$d" <<EOF
dataset: BurstGPT（Azure OpenAI GPT 服務的請求 trace）
source: $base/（GitHub release $tag，2026-01-15 發布）；LICENSE、README.md 取自 commit $c
integrity: 每個 CSV 的 sha256 和 GitHub release asset digest 相同
license: CC-BY-4.0（GitHub API；LICENSE 已一併下載）
columns: $hdr
EOF
}

do_azure2023() {
  local d="$D/azure-llm-2023" c=215becdacba1ce682c7368642ce97d5a332de7a6
  local B="https://raw.githubusercontent.com/Azure/AzurePublicDataset/$c"
  fresh "$d" || return 0
  local tree; tree=$(curl -fsS "https://api.github.com/repos/Azure/AzurePublicDataset/git/trees/$c?recursive=1") || return 1
  local src want
  for src in data/AzureLLMInferenceTrace_code.csv data/AzureLLMInferenceTrace_conv.csv AzureLLMInferenceDataset2023.md LICENSE; do
    want=$(printf '%s' "$tree" | "$PY" -I -c "import json,sys; t=json.load(sys.stdin)['tree']; print([x['sha'] for x in t if x['path']==sys.argv[1]][0])" "$src") || return 1
    get "$B/$src" "$d/$(basename "$src")" && blob_ok "$d/$(basename "$src")" "$want" || return 1
  done
  readme "$d" <<EOF
dataset: Azure LLM inference trace 2023（Splitwise, ISCA'24；2023-11-11 收集）
source: https://github.com/Azure/AzurePublicDataset/blob/$c/AzureLLMInferenceDataset2023.md（data/ 底下兩個 CSV）
commit: $c（master，2026-10-17 查）
integrity: 每個檔的 git blob sha1 和 GitHub tree API 在該 commit 的值相同
license: CC-BY-4.0（repo LICENSE，已一併下載）
columns: $(head -1 "$d/AzureLLMInferenceTrace_code.csv" | tr -d '\r')
EOF
}

do_azure2024() {
  local d="$D/azure-llm-2024" c=215becdacba1ce682c7368642ce97d5a332de7a6
  local base=https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024
  fresh "$d" || return 0
  get "$base/AzureLLMInferenceTrace_code_1week.csv" "$d/AzureLLMInferenceTrace_code_1week.csv" \
    && sha_ok "$d/AzureLLMInferenceTrace_code_1week.csv" 71de5c55cbc35f8f1ed0b6b7806b4cd1e9764b0058469725a6aac98023a1448f || return 1
  get "$base/AzureLLMInferenceTrace_conv_1week.csv" "$d/AzureLLMInferenceTrace_conv_1week.csv" \
    && sha_ok "$d/AzureLLMInferenceTrace_conv_1week.csv" a0cc9b969a9bbf0fd811802cbf4323edd3a209ace791e3799ad4f9207f213941 || return 1
  get "https://raw.githubusercontent.com/Azure/AzurePublicDataset/$c/AzureLLMInferenceDataset2024.md" "$d/AzureLLMInferenceDataset2024.md" || return 1
  get "https://raw.githubusercontent.com/Azure/AzurePublicDataset/$c/LICENSE" "$d/LICENSE" || return 1
  readme "$d" <<EOF
dataset: Azure LLM inference trace 2024（DynamoLLM, HPCA'25；2024-05-10～19 收集，一週）
source: $base/（連結見 AzureLLMInferenceDataset2024Links.txt @ commit $c）
integrity: 兩個 CSV 的 sha256 和 GitHub release asset digest 相同
license: CC-BY-4.0（repo LICENSE @ $c，已一併下載）
columns: $(head -1 "$d/AzureLLMInferenceTrace_code_1week.csv" | tr -d '\r')
EOF
}

_hf_dataset() {   # $1=repo $2=revision $3=目錄：hf download 到 local-dir，再逐檔比對 HF API 的 lfs sha256／git blob sha1
  local repo=$1 rev=$2 d=$3
  "$TIARA_VENV/bin/hf" download "$repo" --repo-type dataset --revision "$rev" --local-dir "$d" 2>&1 | grep -vE 'FutureWarning|warnings.warn|Hint:' | tail -3
  [ "${PIPESTATUS[0]}" = 0 ] || return 1
  curl -fsS "https://huggingface.co/api/datasets/$repo/revision/$rev?blobs=true" > "$TIARA_HOME/logs/hfds_$(echo "$repo" | tr / _).json" || return 1
  "$PY" -I - "$TIARA_HOME/logs/hfds_$(echo "$repo" | tr / _).json" "$d" <<'PY'
import hashlib, json, os, sys
d = json.load(open(sys.argv[1])); root = sys.argv[2]; bad = 0
for s in d["siblings"]:
    p = os.path.join(root, s["rfilename"])
    if not os.path.exists(p):
        print("MISSING", s["rfilename"]); bad += 1; continue
    lfs = s.get("lfs")
    if lfs:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
        ok = h.hexdigest() == lfs["sha256"]
    else:
        n = os.path.getsize(p); h = hashlib.sha1(b"blob %d\0" % n); h.update(open(p, "rb").read())
        ok = h.hexdigest() == s.get("blobId")
    if not ok:
        print("HASH_MISMATCH", s["rfilename"]); bad += 1
print(f"hf_verify {d.get('id')} sha={d.get('sha')} files={len(d['siblings'])} bad={bad}")
sys.exit(1 if bad else 0)
PY
}

do_scbench() {
  local d="$D/scbench" rev=283310bb8c5ba6909dd9a6b1be087d2937f76f6d
  fresh "$d" || return 0
  local v; v=$(_hf_dataset microsoft/SCBench "$rev" "$d") || { echo "$v"; return 1; }
  echo "$v"
  readme "$d" <<EOF
dataset: SCBench（shared-context long-context benchmark）
source: https://huggingface.co/datasets/microsoft/SCBench
revision: $rev（main，2026-10-17 查；gated=False）
integrity: $(echo "$v" | tail -1)（每檔 sha256 和 HF API 的 lfs.sha256 相同）
license: MIT（HF dataset card）
layout: data/scbench_*.jsonl（m8_trace_oracle.py 的 SCB_DIR＝<這裡>/data）、scbench_*/test-*.parquet
EOF
}

do_wildchat() {
  local d="$D/wildchat-1m" rev=7d6490e462285cf85d91eabea0f9a954fbddcd1f
  local g; g=$(curl -fsS "https://huggingface.co/api/datasets/allenai/WildChat-1M" | "$PY" -I -c "import json,sys; print(json.load(sys.stdin).get('gated'))") || return 1
  if [ "$g" != "False" ]; then echo "[wildchat] gated=$g → NEEDS_USER（不繞過）"; return 3; fi
  fresh "$d" || return 0
  local v; v=$(_hf_dataset allenai/WildChat-1M "$rev" "$d") || { echo "$v"; return 1; }
  echo "$v"
  readme "$d" <<EOF
dataset: WildChat-1M
source: https://huggingface.co/datasets/allenai/WildChat-1M
revision: $rev（main，2026-10-17 查；gated=$g）
integrity: $(echo "$v" | tail -1)（每檔 sha256 和 HF API 的 lfs.sha256 相同）
license: ODC-BY（HF dataset card；LICENSE.md 已一併下載。內容含真實使用者對話，使用時照 LICENSE.md 的條款）
EOF
}

do_lmsys() {   # 只查狀態，不下載
  local g; g=$(curl -fsS "https://huggingface.co/api/datasets/lmsys/lmsys-chat-1m" | "$PY" -I -c "import json,sys; print(json.load(sys.stdin).get('gated'))") || return 1
  echo "[lmsys] lmsys/lmsys-chat-1m gated=$g"
  [ "$g" = "False" ] && { echo "[lmsys] 沒有 gate，但這支腳本只記狀態不下載"; return 0; }
  echo "[lmsys] 需要使用者登入 HF、同意 LMSYS-Chat-1M Dataset License Agreement 後才能下載 → NEEDS_USER"; return 3
}

ALL=(hotpotqa longbench mooncake cachewise bailian burstgpt azure2023 azure2024 scbench wildchat lmsys)
[ $# -gt 0 ] && ALL=("$@")
declare -A ST
for it in "${ALL[@]}"; do
  step "$it"
  if ! declare -F "do_$it" >/dev/null; then echo "未知項目 $it"; ST[$it]=UNKNOWN; continue; fi
  "do_$it"; rc=$?
  case $rc in 0) ST[$it]=OK ;; 3) ST[$it]=NEEDS_USER ;; *) ST[$it]="FAILED(rc=$rc)" ;; esac
  echo "[$it] ${ST[$it]}"
done
step "SUMMARY"
fail=0
for it in "${ALL[@]}"; do printf '  %-12s %s\n' "$it" "${ST[$it]}"; [[ "${ST[$it]}" == FAILED* || "${ST[$it]}" == UNKNOWN ]] && fail=1; done
du -sh "$D"/* 2>/dev/null
exit $fail
