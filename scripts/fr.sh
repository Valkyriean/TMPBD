#!/bin/bash




types=("clean" "static" "dynamic") 

start=0
end=9

for t in "${types[@]}"
do
  for ((l=$start; l<=$end; l++))
  do
    python detect.py -d cifar10 -t "$t" -l "$l" --target fr
  done
done



numbers=(81 14 3 94 35 31 28 17 13 86)
types=("dynamic") 

for t in "${types[@]}"
do
  for num in "${numbers[@]}"
  do
    python detect.py -d caltech -t "$t" -l "$num" --target fr
  done
done
