#!/bin/bash

numbers=(90 100 110 120 130 140 150 160 170 180 190)


for num in "${numbers[@]}"
do
  python get_imb_model.py -s "$num"
done
