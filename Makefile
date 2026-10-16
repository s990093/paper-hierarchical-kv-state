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
RESTNAME = r-h200-rest
LAF2NAME = r-h200-la-f2
QWNAME = r-h200-qwen3
F2NAME = r-h200-f2

.PHONY: help queue queue-watch status log runs gpu nvtop submit resume submit-rest submit-r1x submit-f2 test-submit cancel probe dry-run dry-run-rest dry-run-r1x env-check setup setup-check usage

help:
	@echo "H200（在 repo 根目錄、head node 上用）"
	@echo "  make queue         排第幾、前面是誰、什麼時候開始（code/h200_queue.py）"
	@echo "  make queue-watch   同上，每 5 分鐘更新（Ctrl-C 離開）"
	@echo "  make status        自己的 job（squeue）＋今天跑過的（sacct）"
	@echo "  make log           最新的 slurm-*.out 與最新 run 目錄的 stdout／stderr 末段"
	@echo "  make runs          最近 15 個 run 目錄與 exit code（\$$(RUNS)=$(RUNS)）"
	@echo "  make gpu [JOB=<id>]    進正在跑的 job 看一次 GPU（nvidia-smi：使用率、記憶體、功率、process）"
	@echo "  make nvtop [JOB=<id>]  進正在跑的 job 開 nvtop（q 離開；nvtop 裝在 ~/.conda/envs/tools，不在 tiara 環境）"
	@echo "  make test-submit   只問 Slurm 預估什麼時候開始，不送"
	@echo "  make submit        送 R0＋R1–R3（code/r_h200_all.sh）；已有同名 job 或有未 commit 的改動就不送"
	@echo "  make resume FROM=<步驟>  R1–R3 中途停了（R0 已過）：從那一步接著跑（code/r_h200.sh --from；步驟名 bash code/r_h200.sh --list）"
	@echo "  make submit-rest   送第一階段其餘的實驗（code/r_h200_rest.sh）：接在 $(JOBNAME) 後面（afterok），它失敗就自動撤回"
	@echo "  make submit-r1x    送第 1 輪的 LongAlpaca＋F2 能耗（$(LAF2NAME)，接在 $(JOBNAME) 後面、和 $(RESTNAME) 平行）"
	@echo "                     與 Qwen3-30B-A3B（$(QWNAME)，等 $(RESTNAME) 跑完：D2 tier50 會暫寫約 35 GB 到 /work）"
	@echo "  make submit-f2     F2 能耗（K=21 次還原一起量，只做 Llama）：probe（k_rec 119）→ gate → cakee → sweep → final（$(F2NAME)，10 小時）"
	@echo "  make cancel JOB=<id>  撤回"
	@echo "  make probe         只送 R0 硬體報告（1 小時）"
	@echo "  make dry-run       印出 R1–R3 每一步的指令（不用 GPU）"
	@echo "  make dry-run-rest  印出 r_h200_rest.sh 每一步的指令"
	@echo "  make dry-run-r1x   印出 r_h200_r1x.sh（--part la、--part qwen3）每一步的指令"
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

# 進 job 的配置裡看 GPU（srun --overlap：和 job 共用它的卡與 16 核；nvidia-smi／nvtop 只讀 NVML，不在 GPU 上開 process）
# JOB 沒給時用自己第一個 RUNNING 的 job；同時有兩個在跑（例如 r-h200-rest 和 r-h200-la-f2）時用 JOB=<id> 指定
RUNJOB = $(or $(JOB),$(shell $(SLURM_BIN)/squeue -h -u $(USER) -t R -o '%i' 2>/dev/null | head -1))
NVTOP ?= $(HOME)/.conda/envs/tools/bin/nvtop
gpu:
	@[ -n "$(RUNJOB)" ] || { echo "沒有正在跑的 job（make status 看）"; exit 1; }
	@echo "== job $(RUNJOB)"
	@srun --jobid=$(RUNJOB) --overlap -n1 bash -c 'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm --format=csv; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv'

nvtop:
	@[ -n "$(RUNJOB)" ] || { echo "沒有正在跑的 job（make status 看）"; exit 1; }
	srun --jobid=$(RUNJOB) --overlap -n1 --pty $(NVTOP)

test-submit:
	@sbatch --test-only code/r_h200_all.sh 2>&1 | tail -1

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
	@bash code/r_h200.sh --dry-run

dry-run-rest:
	@bash code/r_h200_rest.sh --dry-run

dry-run-r1x:
	@bash code/r_h200_r1x.sh --part la --dry-run; bash code/r_h200_r1x.sh --part qwen3 --dry-run

env-check:
	@bash -c 'source code/env_h200.sh --print'

setup:
	bash code/h200_setup_data.sh

setup-check:
	bash code/h200_setup_data.sh --no-download

usage:
	@sshare -A chihchunghsu -a -o account%14,user%12,rawshares,normshares,rawusage,effectvusage,fairshare
