rm *.bbl *.blg *.aux *.log *.out
xelatex report.tex
bibtex report.aux
xelatex report.tex
xelatex report.tex
rm *.bbl *.blg *.aux *.log *.out
xdg-open ./report.pdf
clear
