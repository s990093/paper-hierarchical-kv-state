# Paper build. English since the 2026-10-16 rewrite, so pdflatex (IEEEtran) is enough.
# figures/*.pdf come from `python3 code/paper_figs.py` (reads results/ only).

MAIN = main
# 排版過的輸出都經過 `| python3 code/h200_show.py ...`（rich）：pipefail 讓前面的指令失敗時 make 也失敗
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c
SHOW = python3 code/h200_show.py

.PHONY: all clean distclean figs check

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex refs.bib $(wildcard figures/*.pdf)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	bibtex  $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex
	@echo "--- build summary ---"
	@echo "overfull boxes: $$(grep -c 'Overfull'     $(MAIN).log || true)"
	@echo "undefined     : $$(grep -cE 'Citation .* undefined|Reference .* undefined' $(MAIN).log || true)"
	@echo "pending marks : $$(grep -c 'pending{' $(MAIN).tex || true)"
	@grep -oE 'Output written on $(MAIN).pdf \(.*\)' $(MAIN).log || true

figs:
	python3 code/paper_figs.py

# list pending markers and TODOs without rebuilding
check:
	@grep -nE '\\pending\{|TODO' $(MAIN).tex | head -30

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

.PHONY: help q queue queue-watch status log runs gpu nvtop submit resume submit-rest submit-r1x submit-f2 submit-m10 resume-m10 dry-run-m10 analyze-m10 submit-lc test-submit cancel probe dry-run dry-run-rest dry-run-r1x env-check setup setup-check usage

help:
	@JOBNAME=$(JOBNAME) RESTNAME=$(RESTNAME) LAF2NAME=$(LAF2NAME) QWNAME=$(QWNAME) F2NAME=$(F2NAME) \
	  RUNS=$(RUNS) python3 code/h200_show.py help

# 精簡版：lab 要等幾個人、每個 job（含等前置的）預測開始／結束、全部跑完要多久
q:
	@python3 code/h200_queue.py --brief

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

# 第二階段主線在 H200 上復現（docs/precision_20261013/13_h200_repro.md）：
#   速度 job（1 張卡，量時間）→ 兩個品質 job（Llama、Qwen 各 1 張卡，afterok 接在速度後面；速度 job 失敗就自動撤回）
# 速度 job 不能和我們自己的其他 GPU job 同時跑（第一階段 F2 疊到 B 的教訓）：有自己的 job 在排隊或在跑就不送。
# TMPDIR：Claude Code 的 shell 設在 /work，sbatch 會帶進 job；第一階段的 job 沒有這個變數 → 送出時拿掉。
M10S = m10h-speed
M10_SPEED_TIME ?= 24:00:00
M10_Q_TIME ?= 40:00:00
SBATCH_CLEAN = env -u TMPDIR -u CLAUDE_CODE_TMPDIR sbatch
submit-m10:
	@o=$$(squeue -h -u $(USER) -o '%i %j %T'); \
	if [ -n "$$o" ]; then echo "我們自己還有 job（速度 job 不能和它們同時跑）："; echo "$$o"; echo "等它們結束，或 make cancel JOB=<id>"; exit 1; fi
	@git diff --quiet && git diff --cached --quiet || \
	  { echo "有未 commit 的改動：job 開跑時才讀 repo，會跑到沒 commit 的版本。先 commit 再送"; git status --short; exit 1; }
	@python3 code/h200_m10_gates.py code && python3 code/h200_m10_gates.py params
	@$(SBATCH_CLEAN) --test-only --job-name=$(M10S) --time=$(M10_SPEED_TIME) code/r_h200_m10.sh --part speed 2>&1 | tail -1
	@s=$$($(SBATCH_CLEAN) --parsable --job-name=$(M10S) --time=$(M10_SPEED_TIME) code/r_h200_m10.sh --part speed) || exit 1; \
	echo "$(M10S) = $$s"; \
	for m in llama qwen; do \
	  q=$$($(SBATCH_CLEAN) --parsable --job-name=m10h-q-$$m --time=$(M10_Q_TIME) --dependency=afterok:$$s --kill-on-invalid-dep=yes \
	       code/r_h200_m10.sh --part quality --model $$m) || exit 1; \
	  echo "m10h-q-$$m = $$q（afterok:$$s）"; done

# 中途停下時從某一步接著跑：make resume-m10 PART=speed FROM=GRID-FP8-qwen ／ PART=quality MODEL=llama FROM=I4E-llama
resume-m10:
	@[ -n "$(PART)" ] && [ -n "$(FROM)" ] || { echo "用法：make resume-m10 PART=speed|quality [MODEL=llama|qwen] FROM=<步驟>（步驟名：bash code/r_h200_m10.sh --part <p> [--model <m>] --list）"; exit 1; }
	@bash code/r_h200_m10.sh --part $(PART) $(if $(MODEL),--model $(MODEL)) --list 2>/dev/null | grep -qx -- '$(FROM)' || { echo "沒有叫做 $(FROM) 的步驟"; exit 1; }
	@git diff --quiet && git diff --cached --quiet || \
	  { echo "有未 commit 的改動：先 commit 再送"; git status --short; exit 1; }
	$(SBATCH_CLEAN) --job-name=m10h-$(if $(MODEL),q-$(MODEL),speed) --time=$(if $(MODEL),$(M10_Q_TIME),$(M10_SPEED_TIME)) \
	  code/r_h200_m10.sh --part $(PART) $(if $(MODEL),--model $(MODEL)) --from $(FROM)

dry-run-m10:
	@bash code/r_h200_m10.sh --part speed --dry-run 2>&1 | $(SHOW) dryrun --title 'r_h200_m10.sh --part speed'
	@bash code/r_h200_m10.sh --part quality --model llama --dry-run 2>&1 | $(SHOW) dryrun --title 'r_h200_m10.sh --part quality --model llama'

# 長 context（10_longctx_plan；13 §8「先跑 128」）：lc-speed（量時間）接在 m10h-speed 後面，lc-screen（量分數）接在 lc-speed 後面。
# 主線品質（m10h-q-*）若還在排隊：改成接在 lc-speed 後面（不和量時間的 job 同時跑），並等 lc-screen 開始後才開始。
submit-lc:
	@for n in m10h-lc-speed m10h-lc-screen; do j=$$(squeue -h -u $(USER) -n $$n -o '%i %T'); \
	  if [ -n "$$j" ]; then echo "已經有 $$n：$$j"; exit 1; fi; done
	@git diff --quiet -- code results docs && git diff --cached --quiet || { echo "有未 commit 的改動：先 commit 再送"; git status --short; exit 1; }
	@s=$$(squeue -h -u $(USER) -n $(M10S) -o '%i' | head -1); d=""; [ -n "$$s" ] && d="--dependency=afterok:$$s --kill-on-invalid-dep=yes"; \
	a=$$($(SBATCH_CLEAN) --parsable --job-name=m10h-lc-speed --time=16:00:00 $$d code/r_h200_lc.sh --part speed) || exit 1; echo "m10h-lc-speed = $$a $$d"; \
	b=$$($(SBATCH_CLEAN) --parsable --job-name=m10h-lc-screen --time=48:00:00 --dependency=afterok:$$a --kill-on-invalid-dep=yes code/r_h200_lc.sh --part screen) || exit 1; \
	echo "m10h-lc-screen = $$b（afterok:$$a）"; \
	for q in $$(squeue -h -u $(USER) -t PD -o '%i %j' | awk '$$2 ~ /^m10h-q-/ {print $$1}'); do \
	  scontrol update JobId=$$q Dependency=afterok:$$a,after:$$b && echo "$$q → afterok:$$a,after:$$b"; done

# 判定（CPU，head node）：照凍結的 m10_analyze.py 走判定流程
analyze-m10:
	PYTHON=$(HOME)/.conda/envs/tiara/bin/python bash code/h200_m10_analyze.sh all

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
