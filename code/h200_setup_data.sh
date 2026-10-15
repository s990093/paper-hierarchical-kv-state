#!/usr/bin/env bash
# =============================================================================
# h200_setup_data.sh — H200 第一次：下載不在 git 裡的模型與資料，逐一檢查 sha256（2026-10-15）
#
#   用法（head node，conda 環境 tiara 已照 docs/H200_SLURM.md §4.1 建好）：
#     bash code/h200_setup_data.sh              # 全部（模型約 31 GB）
#     bash code/h200_setup_data.sh --no-models  # 只下資料（先試網路、checksum）
#
#   head node 沒有對外網路時：在 Mac 上照下面的 URL 下載，rsync 到同樣的路徑，再用 --no-download 只做檢查。
#   sha256 對不上 → 停下來回報（HF 上的檔案可能換過），不准改 checksum 硬過（CLAUDE.md 規則 2）。
# =============================================================================
set -Eeuo pipefail
MODELS=1; DOWNLOAD=1
for a in "$@"; do
  case "$a" in
    --no-models) MODELS=0 ;;
    --no-download) DOWNLOAD=0 ;;
    *) echo "未知參數 $a" >&2; exit 2 ;;
  esac
done
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env_h200.sh"
export M10_LONGBENCH="${M10_LONGBENCH:-$TIARA_DATASETS/longbench}"
mkdir -p "$HF_HOME" "$(dirname "$M10_HOTPOT")" "$M10_LONGBENCH" "$PAPER_HKV_PYLIBS"

HOTPOT_URL=https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/1908d6afbbead072334abe2965f91bd2709910ab/distractor/validation-00000-of-00001.parquet
HOTPOT_SHA=c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6
LB_URL=https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip
LB_SHA=cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64     # MI300X 上 2026-09-15 下載的那份

step() { echo "=== $(date -Is) $*"; }

if [ "$MODELS" = 1 ]; then
  step "模型（固定 revision，和 MI300X 相同；03 §1、migration_20261013 §1.1）"
  python - <<'PY'
from huggingface_hub import snapshot_download
for repo, rev in [("unsloth/Llama-3.1-8B-Instruct", "4699cc75b550f9c6f3173fb80f4703b62d946aa5"),
                  ("Qwen/Qwen2.5-7B-Instruct-1M", "e28526f7bb80e2a9c8af03b831a9af3812f18fba")]:
    print(repo, "->", snapshot_download(repo, revision=rev), flush=True)
PY
fi

step "HotpotQA dev distractor（m10_tasks T4）→ $M10_HOTPOT"
[ "$DOWNLOAD" = 1 ] && curl -fL --retry 3 -o "$M10_HOTPOT" "$HOTPOT_URL"
echo "$HOTPOT_SHA  $M10_HOTPOT" | sha256sum -c -

step "LongBench v1 data.zip（m10_longbench）→ $M10_LONGBENCH"
[ "$DOWNLOAD" = 1 ] && curl -fL --retry 3 -o "$M10_LONGBENCH/data.zip" "$LB_URL"
echo "$LB_SHA  $M10_LONGBENCH/data.zip" | sha256sum -c -
python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$M10_LONGBENCH/data.zip" "$M10_LONGBENCH"
mkdir -p "$M10_LONGBENCH/config" && cp "$HERE"/longbench_config/*.json "$M10_LONGBENCH/config/"
cp "$HERE"/longbench_config/metrics.py "$HERE"/longbench_config/pred.py "$M10_LONGBENCH/"   # 自測對拍用
ls "$M10_LONGBENCH/data" | head -3

step "wonderwords 3.0.1 的詞表（m10_tasks 的 T1、T2 鍵名、T5 詞表）→ $PAPER_HKV_PYLIBS"
[ "$DOWNLOAD" = 1 ] && pip install --no-deps --target "$PAPER_HKV_PYLIBS" wonderwords==3.0.1 -q
ls "$PAPER_HKV_PYLIBS/wonderwords/assets/"

if [ "$MODELS" = 1 ]; then
  step "自測（CPU）：出題、LongBench"
  python -I "$HERE/m10_tasks.py" --self-test --seeds 1 | tail -1
  python -I "$HERE/m10_longbench.py" --self-test | tail -1
fi
step "DONE"
