#!/usr/bin/env python3
"""PreToolUse hook（Bash）：指令指定某張 GPU（CUDA_VISIBLE_DEVICES=<數字>）時，先用 code/gpu_guard.py --check
確認那張卡沒有別人在用；不乾淨、或查不到，就擋下這個指令。

- 為什麼：CLAUDE.md §3「這是共用機器：每次量測都要防插隊」；規則 7「查不到不等於沒有」。
  使用者 2026-10-18：「別人在跑就等，寫腳本看合適沒有，跟寫 hook」。
- 不檢查：沒有 CUDA_VISIBLE_DEVICES=<數字>（例如設成空字串）、或指令裡有 TIARA_NO_GPU=1（只用 CPU）。
- TIARA_ALLOW_DIRTY_START=1：不擋，只提醒（品質量測可以在不乾淨時跑，但時間欄作廢，CLAUDE.md §3）。
- 被擋時請改用 `bash code/gpu_wait_run.sh <短名> <指令>`：它會等到有連續乾淨的卡再跑。
只用標準函式庫（gpu_guard.py 也是），所以用系統 python3 就能跑。
"""
import json
import os
import re
import subprocess
import sys

REPO = os.environ.get("CLAUDE_PROJECT_DIR") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GUARD = os.path.join(REPO, "code", "gpu_guard.py")
_CVD = re.compile(r"CUDA_VISIBLE_DEVICES=([\"']?)([0-9][0-9,]*)\1")


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0                                   # 讀不到輸入就不管（不擋與 GPU 無關的指令）
    cmd = (data.get("tool_input") or {}).get("command") or ""
    if "CUDA_VISIBLE_DEVICES" not in cmd or re.search(r"\bTIARA_NO_GPU=1\b", cmd):
        return 0
    gpus = sorted({int(x) for m in _CVD.finditer(cmd) for x in m.group(2).split(",") if x})
    if not gpus:
        return 0
    bad = []
    for g in gpus:
        try:
            r = subprocess.run([sys.executable, GUARD, "--check", str(g)], capture_output=True, text=True, timeout=45)
            out = " ".join((r.stdout + r.stderr).split())
            if r.returncode != 0:
                bad.append(f"GPU {g}：{out[:300] or f'gpu_guard rc={r.returncode}'}")
        except Exception as e:                    # 查不到＝不能當成乾淨（規則 7）
            bad.append(f"GPU {g}：查不到（{type(e).__name__}: {e}）")
    if not bad:
        return 0
    detail = "；".join(bad)
    if re.search(r"\bTIARA_ALLOW_DIRTY_START=1\b", cmd):
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": f"[gpu_guard hook] 卡不乾淨但指令設了 TIARA_ALLOW_DIRTY_START=1，照跑：{detail}。"
                                 "這次 run 的時間欄作廢，只能用分數（CLAUDE.md §3）。"}}, ensure_ascii=False))
        return 0
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": (
            f"[gpu_guard hook] 擋下：要用的 GPU 現在不乾淨或查不到（{detail}）。共用機上別人在用就等（CLAUDE.md §3）。"
            "請改用 `bash code/gpu_wait_run.sh [--need 1|2] [--gpus \"…\"] <短名> <指令…>`，它會等到有連續乾淨的卡再交給 run3090.sh；"
            "或先 `python3 code/gpu_guard.py --idle-gpus` 挑一張乾淨的卡。")}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
