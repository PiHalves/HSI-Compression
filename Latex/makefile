:
f=main
minted=
# minted=-shell-escape 
minteddir=_minted-$(f)

.PHONY : short
.PHONY : clean


xelatex : $(f).tex
	xelatex $(minted) -synctex=1  -shell-escape main.tex $(f)
	bibtex $(f)
	xelatex $(minted) -synctex=1  -shell-escape main.tex $(f)
	xelatex $(minted) -synctex=1  -shell-escape main.tex $(f)

pdflatex : $(f).tex
	pdflatex $(minted) -synctex=1  -shell-escape main.tex $(f)
	bibtex $(f)
	pdflatex $(minted) -synctex=1  -shell-escape main.tex $(f)
	pdflatex $(minted) -synctex=1  -shell-escape main.tex $(f)

$(f).pdf : $(f).tex
	pdflatex $(minted) -synctex=1  -shell-escape main.tex $(f)
	bibtex $(f)
	pdflatex $(minted) -synctex=1  -shell-escape main.tex $(f)
	pdflatex $(minted) -synctex=1  -shell-escape main.tex $(f)

short : $(f).tex
	pdflatex $(minted) -synctex=1  -shell-escape main.tex $(f)

clean : 
	for p in *.aux *.bbl *.bcf *.blg *.lof *.log *.lot *.out *.toc *.xml *.gz *-blx.bib ; \
	do \
		if [ -f $$p ] ; then rm $$p; fi; \
	done;  \
	if [ -d $(minteddir) ] ; then rm -r $(minteddir) ; fi; 
