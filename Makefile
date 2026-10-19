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
.PHONY: status status-plain q queue gpu jobs nh done tail watch help-q
# 全部用 rich（qstat_rich.py）；rich 跑不起來時退回純文字的 qstat.sh
status:
	@$(QRICH) status || bash $(QSTAT) status
q queue:
	@$(QRICH) q || bash $(QSTAT) q
gpu jobs nh done:
	@$(QRICH) $@ || bash $(QSTAT) $@
tail:                        # make tail N=40 ／ make tail JOB=ds1-llama-s0（看某個 job 的 pool log、stdout、stderr）
	@$(QRICH) tail $(or $(N),30) $(JOB)
watch:                       # 全螢幕、每 SEC 秒刷新（預設 30），Ctrl-C 離開
	@$(QRICH) watch $(or $(SEC),30)
status-plain:
	@bash $(QSTAT) status
help-q:                      # 注意：distclean 是論文的清理，會刪 main.pdf
	@echo "make status｜watch [SEC=10]｜q（queue）｜gpu｜jobs｜nh｜done｜tail [N=40] [JOB=…]｜status-plain"
