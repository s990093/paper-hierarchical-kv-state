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
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk

distclean: clean
	rm -f $(MAIN).pdf
