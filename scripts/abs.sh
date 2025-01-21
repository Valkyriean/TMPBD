#!/bin/bash


types=("clean" "static" "dynamic") 

start=0
end=9


for t in "${types[@]}"
do

  for ((l=$start; l<=$end; l++))
  do
    # python detect.py -d mnist -t "$t" -l "$l" --nstep 10 --npara 5
    python defence_abs.py -d gesture -t "$t" -l "$l" --algorithm abs
  done
done


for t in "${types[@]}"
do
  for ((l=$start; l<=$end; l++))
  do
    python defence_abs.py -d cifar10 -t "$t" -l "$l" --algorithm abs
  done
done


numbers=(81 14 3 94 35 31 28 17 13 86)

for t in "${types[@]}"
do
  for num in "${numbers[@]}"
  do
    python defence_abs.py -d caltech -t "$t" -l "$num" --algorithm abs
  done
done
