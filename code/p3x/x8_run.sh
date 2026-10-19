#!/bin/bash
# x8_run.sh — X8 的記錄殼（CLAUDE.md §4.1）：x8_run.sh <短名> '<指令字串>'
# 產生 /ssd8/hungwei/paper-hkv/runs/<RUN_ID>/{cmd.sh,context.txt,stdout.log,stderr.log,exit_code}；只用 CPU。
set -u
SHORT=$1
CMD=$2
REPO=/home/hungwei/paper-hierarchical-kv-state
RUN_ID=$(date +%Y%m%d-%H%M%S)-p3x8-$SHORT
RUN=/ssd8/hungwei/paper-hkv/runs/$RUN_ID
mkdir -p "$RUN"
{
  echo "export RUN_ID=$RUN_ID RUN=$RUN CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES="
  echo "cd $REPO"
  printf '%s\n' "$CMD"
} > "$RUN/cmd.sh"
{
  date -Is
  nvidia-smi --query-gpu=index,name,memory.used --format=csv 2>&1
  git -C "$REPO" rev-parse HEAD
  git -C "$REPO" status --short
  sha256sum "$REPO"/code/p3x/x8_*.py
} > "$RUN/context.txt"
bash "$RUN/cmd.sh" > "$RUN/stdout.log" 2> "$RUN/stderr.log"
echo $? > "$RUN/exit_code"
echo "$RUN_ID exit=$(cat "$RUN/exit_code")"
