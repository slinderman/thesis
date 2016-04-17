#!/bin/bash
for i in `seq 1 10`;
do
    echo "Compiling " $i
    j=chapters/chapter$i;
    echo $j
    xelatex -jobname=chapter$i "\includeonly{$j}\input{dissertation}";
    bibtex chapter$i
    xelatex -jobname=chapter$i "\includeonly{$j}\input{dissertation}";
    xelatex -jobname=chapter$i "\includeonly{$j}\input{dissertation}";

    rm chapter$i.aux
    rm chapter$i.bbl
    rm chapter$i.blg
    rm chapter$i.log
    rm chapter$i.out
done

echo "Compiling appendixA"
j=chapters/appendixA;
xelatex -jobname=appendixA "\includeonly{$j}\input{dissertation}";
bibtex appendixA
xelatex -jobname=appendixA "\includeonly{$j}\input{dissertation}";
xelatex -jobname=appendixA "\includeonly{$j}\input{dissertation}";

rm appendixA.aux
rm appendixA.bbl
rm appendixA.blg
rm appendixA.log
rm appendixA.out
