rm *.bbl *.blg *.aux *.log *.out
xelatex milestone.tex
bibtex milestone.aux
xelatex milestone.tex
xelatex milestone.tex
rm *.bbl *.blg *.aux *.log *.out
xdg-open ./milestone.pdf
clear
