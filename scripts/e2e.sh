#!/bin/bash


types=("static" "moving" "clean" "dynamic") 


start=0
end=9


for t in "${types[@]}"
do
  for ((l=$start; l<=$end; l++))
  do
    python mitigate.py --algorithm e2e --nsample 20 -t "$t" -l "$l"
  done
done

