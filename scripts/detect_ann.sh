#!/bin/bash


types=("static" "clean" "dynamic") 

algos=("abs" "nc") 

start=0
end=9

for a in "${algos[@]}"
do
  for t in "${types[@]}"
  do
    for ((l=$start; l<=$end; l++))
    do
      python defence_ann.py -d gesture --algorithm "$a" -t "$t" -l "$l"
    done
  done
done

for a in "${algos[@]}"
do
  for t in "${types[@]}"
  do
    for ((l=$start; l<=$end; l++))
    do
      python defence_ann.py -d cifar10 --algorithm "$a" -t "$t" -l "$l"
    done
  done
done


numbers=(81 14 3 94 35 31 28 17 13 86)

for a in "${algos[@]}"
do
  for t in "${types[@]}"
  do
    for num in "${numbers[@]}"
    do
      python defence_ann.py -d caltech --algorithm "$a" -t "$t" -l "$num"
    done
  done
done

