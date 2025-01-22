#!/bin/bash


types=("static" "moving" "clean" "dynamic") 

algos=("original" "self_tuning" "mmbm" "finetuning" "e2e") 

start=0
end=9

for a in "${algos[@]}"
do
  for t in "${types[@]}"
  do
    for ((l=$start; l<=$end; l++))
    do
      python mitigate.py --algorithm "$a" --nsample 20 -t "$t" -l "$l"
    done
  done
done

clamps=("max" "abs" "clamp")

for c in "${clamps[@]}"
do
  for t in "${types[@]}"
  do
    for ((l=$start; l<=$end; l++))
    do
      python mitigate.py --algorithm clamping --clamp_method "$c" --nsample 20 -t "$t" -l "$l"
    done
  done
done

