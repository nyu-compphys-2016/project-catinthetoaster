#!/bin/sh

pdflatex ms
bibtex ms
pdflatex ms
pdflatex ms
rm -f ms.aux
rm -f ms.blg
rm -f ms.log
