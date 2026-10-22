# GPU-NEED: both
# GPU: 1
# NEED-GB: 180
# MAX-H: 1.5
# V0：vLLM 0.28（ROCm）的 APC、CPU 卸載、tiering 能不能跑、命中與不命中輸出是否相同（docs/stage3_20261019/v0_f1_i3_howto.md §2）。
# a、b、c 不過 → F1／F2／I3g 不能用 vLLM，停（NEEDS_HUMAN）；d（ROCM_ATTN patch）、e（fs 層）不過只記下，不擋後面。
source /mlsteam/workspace/paper-hierarchical-kv-state/code/m16_queue/jobs/lib.sh
JOBNAME=40_v0
source $REPO/code/m16_queue/jobs/vllm_lib.sh
OUT=$RES/v0_vllm_smoke; mkdir -p $OUT
TXT=/mlsteam/data/tiara/datasets/longbench/data/narrativeqa.jsonl
tie_ok() {   # $1＝label。v0check FAIL（rc=3）只因為 *_match 有分歧、而且每個分歧位置 miss 的 top-1／top-2 差 ≤ 1e-3（數值平手），
             # 其他檢查（miss_cold、*_cached、determinism、loaded bytes…）全部 PASS → 記 NEEDS_HUMAN（資訊）、當成通過。
             # 依據：v0_f1_i3_howto.md §2「不一致時怎麼讀」；10/22 04:09Z V0-b 文字題兩題在 token 7、15 分歧、margin 都是 0.0。
  local f; f=$(ls -t $OUT/v0check_*-m16-$1.json 2>/dev/null | head -1)
  [ -n "$f" ] || { nh "$JOBNAME：$1 找不到 v0check json"; return 1; }
  python3 -I - "$f" <<'PY' || { nh "$JOBNAME：$1 v0check FAIL 不只是數值平手（$f）"; return 1; }
import json, sys
d = json.load(open(sys.argv[1]))
bad = [k for k, v in d["verdict"].items() if v != "PASS" and not k.endswith("_match")]
rows = [ph for p in d["prompts"] for name, ph in p["phases"].items() if str(ph.get("match_vs_miss")) == "0"]
margins = [float(ph["miss_margin_at_diverge"]) if ph.get("miss_margin_at_diverge") not in (None, "") else float("nan") for ph in rows]
ok = not bad and rows and all(m == m and m <= 1e-3 for m in margins)
print(f"tie_ok：非 match 項不過 {bad}；分歧 {len(rows)} 列、margin {margins} → {'只是數值平手' if ok else '不是'}")
sys.exit(0 if ok else 1)
PY
  nh "$JOBNAME：$1 的 v0check 分歧只在數值平手（margin ≤ 1e-3），照 howto §2 當成通過（資訊，不擋）"
}
v0() {
  local tag=$1; shift
  srv_start v0$tag --max-model-len 34816 "$@" || return 1
  bench_run v0$tag-rnd v0check --out-dir $OUT --lengths 8192 16384 32768 --n-prompts 3 --max-tokens 32 --settle-s 3 \
    --label v0$tag-rnd || tie_ok v0$tag-rnd || return 1
  bench_run v0$tag-txt v0check --out-dir $OUT --n-prompts 3 --max-tokens 32 --settle-s 3 --text-file $TXT --text-chars 30000 \
    --label v0$tag-txt || tie_ok v0$tag-txt || return 1
}
v0 a --apc off || exit 1
v0 b --apc on || exit 1
v0 c --apc on --offload cpu:64 || exit 1
v0 d --apc on --offload cpu:64 --rocm-attn-patch || nh "40_v0：V0-d（ROCM_ATTN patch）沒過 → F1／F2 不用 patch（不擋）"
rm -rf /var/tmp/m16_fs_v0e
v0 e --apc on --offload tier:16:/var/tmp/m16_fs_v0e:64 || nh "40_v0：V0-e（fs 層）沒過（不擋）"
srv_stop; rm -rf /var/tmp/m16_fs_v0e
echo "40_v0 結束 $(date -u +%FT%TZ)"
