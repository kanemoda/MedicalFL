TEX = pdflatex
MAIN = dp-fedbn

.PHONY: all clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex figures/fig_eval_regime.pdf figures/fig_privacy_utility.pdf figures/fig_cross_task.pdf
	$(TEX) -interaction=nonstopmode -halt-on-error $(MAIN).tex
	$(TEX) -interaction=nonstopmode -halt-on-error $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).pdf
