TEX=paper
.PHONY: all,clean


all: $(TEX).pdf


$(TEX).pdf: $(TEX).tex vampy.bib
	# for bibtex to work correctly we need to compile twice
#	cp ~/literature/bibtex/vampy_paper.bib ./vampy.bib
	pdflatex $(TEX)
	biber $(TEX)
	pdflatex $(TEX)
	pdflatex $(TEX)


clean:
	# remove all files except for .tex, .bib and the Makefile
	find -type f | grep -v 'tex\|bib\|Makefile\|figures\|cls' | xargs rm
