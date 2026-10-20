#!/bin/bash
# =============================================================================
# h200_setup_p3_env.sh — 平台 C（H200）第三階段的 Python 環境（2026-10-19）
#
#   bash code/h200_setup_p3_env.sh            # 建 venv、裝套件、寫 pip freeze（head node；要網路）
#   bash code/h200_setup_p3_env.sh --check    # 只檢查 import 與版本
#
# 做法：在 conda 環境 tiara（第一、二階段用的那個，vLLM 0.28.0、torch 2.13.0+cu130）之上開一個
#   python -m venv --system-site-packages $TIARA_P3_VENV
#   → torch／numpy／pandas／transformers 直接用 tiara 的（同一份檔），只把第三階段多出來的套件裝進 venv。
#   tiara 本身一個檔都不動（第二階段 session 的 job 還在用它）。
# 版本：照 3090 的 venv/tiara（docs/ENV_3090.md §2.1）釘死；tiara 已有的套件用 tiara 的版本當 constraint，不准升降。
#   和 3090 不同的地方（照實記，不硬湊）：numpy 2.3.5（3090 2.2.6）、torch 2.13.0+cu130（3090 2.10.0+cu126）、
#   transformers 5.19.0（3090 5.17.0）——都是 tiara 原本的版本。xgboost 用 PyPI 的 3.4.1（CUDA 13 build；H200 的 driver 支援 13，
#   3090 的 driver 550 只到 12.4 才要換 xgboost-cu12）。
# 輸出：$TIARA_P3_VENV、$TIARA_HOME/logs/p3_env/{constraints_tiara.txt, pip_install.log, pip_freeze_p3.txt, check.txt}
# 失敗就停（CLAUDE.md 規則 2）。
# =============================================================================
set -uo pipefail
CHECK=0; [ "${1:-}" = "--check" ] && CHECK=1
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO
# shellcheck disable=SC1091
source "$REPO/code/env_h200.sh"
export TIARA_P3_VENV="${TIARA_P3_VENV:-$TIARA_HOME/venvs/tiara-p3}"
LOG="$TIARA_HOME/logs/p3_env"; mkdir -p "$LOG"
# head node 的 /tmp（20G，五十幾人共用）常常是滿的（memory：h200-headnode-tmp-full）→ pip 的暫存放 /work
export TMPDIR="${P3_PIP_TMPDIR:-$TIARA_HOME/tmp/pip-p3}"; mkdir -p "$TMPDIR"
BASE_PY="$(command -v python)"
[ -n "${CONDA_PREFIX:-}" ] && [ "$BASE_PY" = "$CONDA_PREFIX/bin/python" ] || { echo "[p3env] 不是 conda tiara 的 python（$BASE_PY，CONDA_PREFIX=${CONDA_PREFIX:-}）→ 停" >&2; exit 2; }

# 3090 venv/tiara 的版本（docs/ENV_3090.md §2.1）
PKGS=(scikit-learn==1.9.1 lightgbm==4.7.0 xgboost==3.4.1 statsmodels==0.15.0 lifelines==0.30.0
      scikit-survival==0.28.0 pycox==0.3.0 torchtuples==0.2.2 pytest==9.1.1)
# 不裝 sentence-transformers：3090 的 6.1.0 要 huggingface-hub<2.0，tiara 是 2.2.0（10/19 pip ResolutionImpossible）。
# 只有 P3-7（內容型預測器，可選、不在這次範圍）與 smoke_3090 libs 用到；要做 P3-7 時另外決定版本。
MODS=(sklearn lightgbm xgboost statsmodels lifelines sksurv pycox torchtuples pytest
      numpy pandas scipy pyarrow torch transformers numba)

check() {
  "$TIARA_P3_VENV/bin/python" - "${MODS[@]}" <<'PY'
import importlib, sys
bad = 0
print("python", sys.version.split()[0], "prefix", sys.prefix, "base", sys.base_prefix)
for m in sys.argv[1:]:
    try:
        mod = importlib.import_module(m)
        print(f"{m:24s} {getattr(mod, '__version__', '?'):16s} {mod.__file__}")
    except Exception as e:
        bad += 1
        print(f"{m:24s} FAILED {type(e).__name__}: {e}")
sys.exit(1 if bad else 0)
PY
}

if [ "$CHECK" = 0 ]; then
  if [ ! -x "$TIARA_P3_VENV/bin/python" ]; then
    echo "[p3env] $(date -Is) 建 venv：$BASE_PY -m venv --system-site-packages $TIARA_P3_VENV"
    "$BASE_PY" -m venv --system-site-packages "$TIARA_P3_VENV" || { echo "[p3env] venv 建不起來 → 停" >&2; exit 1; }
  fi
  # tiara 已有的套件：版本當 constraint（--system-site-packages 看得到，pip 不會重裝；constraint 防止升降）
  "$BASE_PY" -m pip list --format=freeze 2>/dev/null | grep -E '^[A-Za-z0-9_.-]+==' > "$LOG/constraints_tiara.txt"
  echo "[p3env] constraints: $(wc -l < "$LOG/constraints_tiara.txt") 個 tiara 套件"
  "$TIARA_P3_VENV/bin/python" -m pip install --no-cache-dir -c "$LOG/constraints_tiara.txt" "${PKGS[@]}" > "$LOG/pip_install.log" 2>&1
  rc=$?
  tail -5 "$LOG/pip_install.log"
  [ "$rc" -eq 0 ] || { echo "[p3env] pip install 失敗（rc=$rc，完整 log：$LOG/pip_install.log）→ 停" >&2; exit 1; }
  "$TIARA_P3_VENV/bin/python" -m pip freeze --local > "$LOG/pip_freeze_p3.txt" 2>&1
  echo "[p3env] venv 裡新裝的：$(wc -l < "$LOG/pip_freeze_p3.txt") 個（$LOG/pip_freeze_p3.txt）"
fi
check | tee "$LOG/check.txt"
rc=${PIPESTATUS[0]}
du -sh "$TIARA_P3_VENV" 2>/dev/null
exit "$rc"
