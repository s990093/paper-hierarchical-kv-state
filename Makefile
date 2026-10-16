# Adaptive Hierarchical KV State Management — build
# 必須用 XeLaTeX（內文含中文，使用 xeCJK）

MAIN = main
# 排版過的輸出都經過 `| python3 code/h200_show.py ...`（rich）：pipefail 讓前面的指令失敗時 make 也失敗
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c
SHOW = python3 code/h200_show.py

.PHONY: all clean distclean watch check

all: $(MAIN).pdf

# 預設安靜：每一趟只印一行，完整輸出在 build.log；V=1 看原始輸出。
# 失敗時印出 .log 裡的錯誤行（-file-line-error 的 `檔名:行號:` 與 `!` 開頭）
V ?= 0
LATEX = xelatex -interaction=nonstopmode -file-line-error -halt-on-error $(MAIN).tex
ifeq ($(V),1)
Q =
pass = $(2)
else
Q = @
pass = printf '  ▸ %s\n' '$(1)'; $(2) >>build.log 2>&1 || { \
	  printf '\033[31m  ✗ %s 失敗\033[0m（完整輸出：build.log）\n' '$(1)'; \
	  grep -A3 -E '^(\./)?[^ :]+\.(tex|sty|cls|bbl):[0-9]+:|^!' $(MAIN).log 2>/dev/null | head -20 | grep . \
	    || tail -n 15 build.log; exit 1; }
endif

$(MAIN).pdf: $(MAIN).tex refs.bib
	$(Q)rm -f build.log
	$(Q)$(call pass,xelatex,$(LATEX))
	$(Q)$(call pass,bibtex,bibtex $(MAIN))
	$(Q)$(call pass,xelatex,$(LATEX))
	$(Q)$(call pass,xelatex,$(LATEX))
	@# rich 版摘要；沒有 rich 的機器（例如 3090 的系統 python）退回 bash 版
	@if python3 -c 'import rich' 2>/dev/null; then $(SHOW) latex $(MAIN); \
	else bash code/latex_summary.sh $(MAIN); fi

# 只檢查不重建
check:
	@grep -nE 'TODO|placeholder' $(MAIN).tex | head -30

clean:
	# *.out 不能全刪：H200 上 Slurm 的輸出 slurm-*.out 也在根目錄
	rm -f *.aux *.bbl *.blg *.log build.log $(filter-out slurm-%.out,$(wildcard *.out)) *.toc *.fls *.fdb_latexmk

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
RESTNAME = r-h200-rest
LAF2NAME = r-h200-la-f2
QWNAME = r-h200-qwen3
F2NAME = r-h200-f2

.PHONY: help queue queue-watch status log runs gpu nvtop submit resume submit-rest submit-r1x submit-f2 test-submit cancel probe dry-run dry-run-rest dry-run-r1x env-check setup setup-check usage

help:
	@JOBNAME=$(JOBNAME) RESTNAME=$(RESTNAME) LAF2NAME=$(LAF2NAME) QWNAME=$(QWNAME) F2NAME=$(F2NAME) \
	  RUNS=$(RUNS) python3 code/h200_show.py help

queue:
	@python3 code/h200_queue.py

queue-watch:
	@python3 code/h200_queue.py --no-history --watch 300

status:
	@$(SHOW) status

# 整理過的版本：步驟表（耗時、exit）＋失敗步驟的錯誤＋最新 run 末段。
# make log JOB=<id> 看指定 job；make log RAW=1 看 slurm-*.out 原始末 40 行
log:
	@RUNS=$(RUNS) $(SHOW) log $(if $(JOB),--job $(JOB)) $(if $(RAW),--raw)

runs:
	@RUNS=$(RUNS) $(SHOW) runs -n $(or $(N),15)

# 進 job 的配置裡看 GPU（srun --overlap：和 job 共用它的卡與 16 核；nvidia-smi／nvtop 只讀 NVML，不在 GPU 上開 process）
# JOB 沒給時用自己第一個 RUNNING 的 job；同時有兩個在跑（例如 r-h200-rest 和 r-h200-la-f2）時用 JOB=<id> 指定
RUNJOB = $(or $(JOB),$(shell $(SLURM_BIN)/squeue -h -u $(USER) -t R -o '%i' 2>/dev/null | head -1))
NVTOP ?= $(HOME)/.conda/envs/tools/bin/nvtop
gpu:
	@[ -n "$(RUNJOB)" ] || { echo "沒有正在跑的 job（make status 看）"; exit 1; }
	@srun --jobid=$(RUNJOB) --overlap -n1 bash -c 'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm --format=csv; echo ---; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv' \
	  | $(SHOW) gpu --job $(RUNJOB)

nvtop:
	@[ -n "$(RUNJOB)" ] || { echo "沒有正在跑的 job（make status 看）"; exit 1; }
	srun --jobid=$(RUNJOB) --overlap -n1 --pty $(NVTOP)

test-submit:
	@sbatch --test-only code/r_h200_all.sh 2>&1 | tail -1 | $(SHOW) testsubmit --script code/r_h200_all.sh

submit:
	@j=$$(squeue -h -u $(USER) -n $(JOBNAME) -o '%i %T'); \
	if [ -n "$$j" ]; then echo "已經有 $(JOBNAME)：$$j。要重送先 make cancel JOB=<id>"; exit 1; fi
	@git diff --quiet && git diff --cached --quiet || \
	  { echo "有未 commit 的改動：job 開跑時才讀 repo，會跑到沒 commit 的版本。先 commit 再送"; git status --short; exit 1; }
	@sbatch --test-only code/r_h200_all.sh 2>&1 | tail -1
	sbatch code/r_h200_all.sh

# R1–R3 中途停下時（例如 10/16 job 20417 停在 R2-C6）：從那一步接著跑。R0 已過，M7_DISK 讀 env_h200.sh。
# job 名沿用 $(JOBNAME)：submit-rest／submit-r1x 用這個名字找要接在誰後面。TIME 預設 1 小時（MI300X 上 C6 約 4 分）
RESUME_TIME ?= 01:00:00
resume:
	@[ -n "$(FROM)" ] || { echo "用法：make resume FROM=<步驟名> [RESUME_TIME=01:00:00]（步驟名：bash code/r_h200.sh --list）"; exit 1; }
	@bash code/r_h200.sh --list | grep -qx -- '$(FROM)' || { echo "沒有叫做 $(FROM) 的步驟（bash code/r_h200.sh --list）"; exit 1; }
	@j=$$(squeue -h -u $(USER) -n $(JOBNAME) -o '%i %T'); \
	if [ -n "$$j" ]; then echo "已經有 $(JOBNAME)：$$j。要重送先 make cancel JOB=<id>"; exit 1; fi
	@git diff --quiet && git diff --cached --quiet || \
	  { echo "有未 commit 的改動：job 開跑時才讀 repo，會跑到沒 commit 的版本。先 commit 再送"; git status --short; exit 1; }
	@sbatch --test-only --job-name=$(JOBNAME) --time=$(RESUME_TIME) code/r_h200.sh --from $(FROM) 2>&1 | tail -1
	sbatch --job-name=$(JOBNAME) --time=$(RESUME_TIME) code/r_h200.sh --from $(FROM)

submit-rest:
	@j=$$(squeue -h -u $(USER) -n $(RESTNAME) -o '%i %T'); \
	if [ -n "$$j" ]; then echo "已經有 $(RESTNAME)：$$j。要重送先 make cancel JOB=<id>"; exit 1; fi
	@git diff --quiet && git diff --cached --quiet || \
	  { echo "有未 commit 的改動：job 開跑時才讀 repo，會跑到沒 commit 的版本。先 commit 再送"; git status --short; exit 1; }
	@dep=$$(squeue -h -u $(USER) -n $(JOBNAME) -o '%i' | head -1); \
	if [ -n "$$dep" ]; then echo "接在 $(JOBNAME) $$dep 後面（afterok；它沒過就自動撤回）"; \
	  sbatch --dependency=afterok:$$dep --kill-on-invalid-dep=yes code/r_h200_rest.sh; \
	else echo "沒有排隊中的 $(JOBNAME)：直接送（r_h200_rest.sh 開跑時會檢查 R3 的 repro_verdict.json）"; \
	  sbatch code/r_h200_rest.sh; fi

submit-r1x:
	@for n in $(LAF2NAME) $(QWNAME); do j=$$(squeue -h -u $(USER) -n $$n -o '%i %T'); \
	  if [ -n "$$j" ]; then echo "已經有 $$n：$$j。要重送先 make cancel JOB=<id>"; exit 1; fi; done
	@git diff --quiet && git diff --cached --quiet || \
	  { echo "有未 commit 的改動：job 開跑時才讀 repo，會跑到沒 commit 的版本。先 commit 再送"; git status --short; exit 1; }
	@all=$$(squeue -h -u $(USER) -n $(JOBNAME) -o '%i' | head -1); rest=$$(squeue -h -u $(USER) -n $(RESTNAME) -o '%i' | head -1); \
	d1=""; [ -n "$$all" ] && d1="--dependency=afterok:$$all --kill-on-invalid-dep=yes"; \
	d2=""; [ -n "$$all" ] && d2="afterok:$$all"; [ -n "$$rest" ] && d2="$${d2:+$$d2,}afterany:$$rest"; \
	[ -n "$$d2" ] && d2="--dependency=$$d2 --kill-on-invalid-dep=yes"; \
	echo "$(LAF2NAME)：$${d1:-（沒有排隊中的 $(JOBNAME)：直接送，開跑時檢查 repro_verdict.json）}"; \
	sbatch --job-name=$(LAF2NAME) --time=04:00:00 $$d1 code/r_h200_r1x.sh --part la; \
	echo "$(QWNAME)：$${d2:-（沒有排隊中的 $(JOBNAME)／$(RESTNAME)：直接送，下載前檢查配額）}"; \
	sbatch --job-name=$(QWNAME) --time=05:00:00 $$d2 code/r_h200_r1x.sh --part qwen3

# 10/16：F2-gate 沒過（NVML 計數器約 100 ms 才更新）→ 使用者選「K=21，只做 Llama」、「重跑 probe，k_rec=119」。
# LongAlpaca 的 ok／C1／vllmf2 在 job 20555 已經跑完，這裡從 F2-probe 跑到 F2-final 就停（--until）
F2_TIME ?= 10:00:00
submit-f2:
	@j=$$(squeue -h -u $(USER) -n $(F2NAME) -o '%i %T'); \
	if [ -n "$$j" ]; then echo "已經有 $(F2NAME)：$$j。要重送先 make cancel JOB=<id>"; exit 1; fi
	@git diff --quiet && git diff --cached --quiet || \
	  { echo "有未 commit 的改動：job 開跑時才讀 repo，會跑到沒 commit 的版本。先 commit 再送"; git status --short; exit 1; }
	@sbatch --test-only --job-name=$(F2NAME) --time=$(F2_TIME) code/r_h200_r1x.sh --part la --from F2-probe --until F2-final 2>&1 | tail -1
	sbatch --job-name=$(F2NAME) --time=$(F2_TIME) code/r_h200_r1x.sh --part la --from F2-probe --until F2-final

cancel:
	@[ -n "$(JOB)" ] || { echo "用法：make cancel JOB=<job id>（make status 看 id）"; exit 1; }
	scancel $(JOB)
	@squeue -u $(USER)

probe:
	sbatch --job-name=r0-h200-probe --time=01:00:00 code/tiara.sbatch bash code/h200_probe.sh

dry-run:
	@bash code/r_h200.sh --dry-run 2>&1 | $(SHOW) dryrun --title 'R1–R3（code/r_h200.sh）'

dry-run-rest:
	@bash code/r_h200_rest.sh --dry-run 2>&1 | $(SHOW) dryrun --title 'code/r_h200_rest.sh'

dry-run-r1x:
	@bash code/r_h200_r1x.sh --part la --dry-run 2>&1 | $(SHOW) dryrun --title 'r_h200_r1x.sh --part la'
	@bash code/r_h200_r1x.sh --part qwen3 --dry-run 2>&1 | $(SHOW) dryrun --title 'r_h200_r1x.sh --part qwen3'

env-check:
	@bash -c 'source code/env_h200.sh --print' 2>&1 | $(SHOW) env

setup:
	bash code/h200_setup_data.sh

setup-check:
	bash code/h200_setup_data.sh --no-download

usage:
	@$(SHOW) usage --account chihchunghsu
