#!/bin/bash

numbers=(90 100 110 120 130 140 150 160 170 180 190)


for num in "${numbers[@]}"
do
  python detect.py -d gesture -t clean -l "$num"
done
