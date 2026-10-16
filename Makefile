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
