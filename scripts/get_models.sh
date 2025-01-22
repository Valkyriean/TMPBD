#!/bin/bash

types=("clean" "static" "dynamic") 

start=0
end=9


for ((l=$start; l<=$end; l++))
do
  # clean
  python get_models.py --dataset gesture --polarity 1 --pos top-left --trigger_size 0.1 --epsilon 0 --type static --cupy --epochs 64 --trigger_label "$l"
  # static
  python get_models.py --dataset gesture --polarity 1 --pos top-left --trigger_size 0.1 --epsilon 0.1 --type static --cupy --epochs 64 --trigger_label "$l"
  # dynamic 
  python dynamic.py --dataset gesture --batch_size 4 --cupy --epochs 64 --train_epochs 1 --alpha 0.5 --beta 0.01 --trigger_label "$num"
  # moving
  python get_models.py --dataset gesture --polarity 1 --pos top-left --trigger_size 0.1 --epsilon 0.1 --type moving --cupy --epochs 64 --trigger_label "$l"
done


for ((l=$start; l<=$end; l++))
do
  # clean
  python get_models.py --dataset cifar10 --polarity 1 --pos top-left --trigger_size 0.1 --epsilon 0 --type static --cupy --epochs 28 --trigger_label "$l"
  # static
  python get_models.py --dataset cifar10 --polarity 1 --pos top-left --trigger_size 0.1 --epsilon 0.1 --type static --cupy --epochs 28 --trigger_label "$num"
  # dynamic 
  python dynamic.py --dataset cifar10 --batch_size 4 --cupy --epochs 28 --train_epochs 1 --alpha 0.5 --beta 0.01 --trigger_label "$num"
done


numbers=(81 14 3 94 35 31 28 17 13 86)

for num in "${numbers[@]}"
do
  # clean
  python get_models.py --dataset caltech --polarity 1 --pos top-left --trigger_size 0.1 --epsilon 0 --type static --cupy --epochs 30 --trigger_label "$num"
  # static
  python get_models.py --dataset caltech --polarity 1 --pos top-left --trigger_size 0.1 --epsilon 0.1 --type static --cupy --epochs 30 --trigger_label "$num"
  # dynamic 
  python dynamic.py --dataset caltech --batch_size 4 --cupy --epochs 30 --train_epochs 1 --alpha 0.5 --beta 0.01 --trigger_label "$num"
done
