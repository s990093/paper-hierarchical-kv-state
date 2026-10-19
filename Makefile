# Paper build. English since the 2026-10-16 rewrite, so pdflatex (IEEEtran) is enough.
# figures/*.pdf come from `python3 code/paper_figs.py` (reads results/ only).

MAIN = main

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
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk

distclean: clean
	rm -f $(MAIN).pdf

# ── MI300X GPU 佇列（10/19；像 H200 那邊一樣用 make 看 queue）──────────────────
#   make status  全部（runner、卡、顯存帳、跑中的 job、排隊、最近完成、NEEDS_HUMAN）
#   make q／gpu／jobs／nh／done／tail（N=50 看更多行）／watch（每 30 秒刷新）
QSTAT := code/m10_queue_mi300x/qstat.sh
QRICH := /mlsteam/workspace/venv/tiara/bin/python code/m10_queue_mi300x/qstat_rich.py
.PHONY: status status-plain q gpu jobs nh done tail watch
status:                      # rich 狀態板；rich 不能用時退回純文字版
	@$(QRICH) || bash $(QSTAT) status
status-plain:
	@bash $(QSTAT) status
q gpu jobs nh done tail:
	@bash $(QSTAT) $@
watch:
	@watch -n 30 -t -c "FORCE_COLOR=1 COLUMNS=$$(tput cols) $(QRICH)"
