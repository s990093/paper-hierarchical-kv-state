# Adaptive Hierarchical KV State Management — build
# 必須用 XeLaTeX（內文含中文，使用 xeCJK）

MAIN = main

.PHONY: all clean distclean watch check

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex refs.bib
	xelatex -interaction=nonstopmode $(MAIN).tex
	bibtex  $(MAIN)
	xelatex -interaction=nonstopmode $(MAIN).tex
	xelatex -interaction=nonstopmode $(MAIN).tex
	@echo "--- build summary ---"
	@echo "font warnings : $$(grep -c 'Font Warning' $(MAIN).log || true)"
	@echo "overfull boxes: $$(grep -c 'Overfull'     $(MAIN).log || true)"
	@echo "undefined     : $$(grep -cE 'Citation .* undefined|Reference .* undefined' $(MAIN).log || true)"
	@grep -oE 'Output written on $(MAIN).pdf \(.*\)' $(MAIN).log || true

# 只檢查不重建
check:
	@grep -nE 'TODO|placeholder' $(MAIN).tex | head -30

clean:
	# *.out 不能全刪：H200 上 Slurm 的輸出 slurm-*.out 也在根目錄
	rm -f *.aux *.bbl *.blg *.log $(filter-out slurm-%.out,$(wildcard *.out)) *.toc *.fls *.fdb_latexmk

distclean: clean
	rm -f $(MAIN).pdf

# =============================================================================
# 平台 C（H200，學校 Slurm）——`make help` 看全部。docs/H200_SLURM.md
# =============================================================================
SLURM_BIN := /cm/local/apps/slurm/current/bin
ifneq ($(wildcard $(SLURM_BIN)/squeue),)
export PATH := $(SLURM_BIN):$(PATH)
export SLURM_CONF ?= /cm/shared/apps/slurm/etc/slurm/slurm.conf
endif
RUNS ?= $(HOME)/tiara-runs
JOBNAME = r-h200-all

.PHONY: help queue queue-watch status log runs submit test-submit cancel probe dry-run env-check setup setup-check usage

help:
	@echo "H200（在 repo 根目錄、head node 上用）"
	@echo "  make queue         排第幾、前面是誰、什麼時候開始（code/h200_queue.py）"
	@echo "  make queue-watch   同上，每 5 分鐘更新（Ctrl-C 離開）"
	@echo "  make status        自己的 job（squeue）＋今天跑過的（sacct）"
	@echo "  make log           最新的 slurm-*.out 與最新 run 目錄的 stdout／stderr 末段"
	@echo "  make runs          最近 15 個 run 目錄與 exit code（\$$(RUNS)=$(RUNS)）"
	@echo "  make test-submit   只問 Slurm 預估什麼時候開始，不送"
	@echo "  make submit        送 R0＋R1–R3（code/r_h200_all.sh）；已有同名 job 或有未 commit 的改動就不送"
	@echo "  make cancel JOB=<id>  撤回"
	@echo "  make probe         只送 R0 硬體報告（1 小時）"
	@echo "  make dry-run       印出 R1–R3 每一步的指令（不用 GPU）"
	@echo "  make env-check     印出 env_h200.sh 的所有路徑、標出 MISSING"
	@echo "  make setup / setup-check   下載模型與資料並檢查 sha256／只檢查"
	@echo "  make usage         lab 內每個人的用量與 fairshare（協調用）"
	@echo "論文：make（編譯 main.pdf）、make check、make clean"

queue:
	@python3 code/h200_queue.py

queue-watch:
	watch -n 300 python3 code/h200_queue.py --no-history

status:
	@squeue -u $(USER) -o '%.10i %.14j %.9T %.5Q %.20r %.12M %.12l %.20S'
	@echo "--- 今天的 job（sacct）---"
	@sacct -u $(USER) -X -S today -o jobid%10,jobname%14,state%12,start%20,end%20,elapsed%12,exitcode

log:
	@f=$$(ls -t slurm-*.out 2>/dev/null | head -1); \
	if [ -n "$$f" ]; then echo "== $$f（末 30 行）"; tail -n 30 "$$f"; else echo "還沒有 slurm-*.out（job 還沒開始）"; fi
	@d=$$(ls -td $(RUNS)/*/ 2>/dev/null | head -1); \
	if [ -n "$$d" ]; then echo; echo "== 最新 run：$$d（exit=$$(cat $$d/exit_code 2>/dev/null || echo 還在跑)）"; \
	  echo "-- stdout 末 15 行"; tail -n 15 "$$d/stdout.log" 2>/dev/null; echo "-- stderr 末 10 行"; tail -n 10 "$$d/stderr.log" 2>/dev/null; fi

runs:
	@for d in $$(ls -td $(RUNS)/*/ 2>/dev/null | head -15); do \
	  printf '%-58s exit=%s\n' "$$(basename $$d)" "$$(cat $$d/exit_code 2>/dev/null || echo 還在跑或沒寫)"; done

test-submit:
	@sbatch --test-only code/r_h200_all.sh 2>&1 | tail -1

submit:
	@j=$$(squeue -h -u $(USER) -n $(JOBNAME) -o '%i %T'); \
	if [ -n "$$j" ]; then echo "已經有 $(JOBNAME)：$$j。要重送先 make cancel JOB=<id>"; exit 1; fi
	@git diff --quiet && git diff --cached --quiet || \
	  { echo "有未 commit 的改動：job 開跑時才讀 repo，會跑到沒 commit 的版本。先 commit 再送"; git status --short; exit 1; }
	@sbatch --test-only code/r_h200_all.sh 2>&1 | tail -1
	sbatch code/r_h200_all.sh

cancel:
	@[ -n "$(JOB)" ] || { echo "用法：make cancel JOB=<job id>（make status 看 id）"; exit 1; }
	scancel $(JOB)
	@squeue -u $(USER)

probe:
	sbatch --job-name=r0-h200-probe --time=01:00:00 code/tiara.sbatch bash code/h200_probe.sh

dry-run:
	@bash code/r_h200.sh --dry-run

env-check:
	@bash -c 'source code/env_h200.sh --print'

setup:
	bash code/h200_setup_data.sh

setup-check:
	bash code/h200_setup_data.sh --no-download

usage:
	@sshare -A chihchunghsu -a -o account%14,user%12,rawshares,normshares,rawusage,effectvusage,fairshare
