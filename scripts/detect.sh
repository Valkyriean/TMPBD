#!/bin/bash

$types = @("clean", "static", "dynamic")

$start = 0
$end = 9

foreach ($t in $types) {
    for ($l = $start; $l -le $end; $l++) {
        python detect.py -d gesture -t $t -l $l
    }
}

foreach ($t in $types) {
    for ($l = $start; $l -le $end; $l++) {
        python detect.py -d cifar10 -t $t -l $l
    }
}

$numbers = @(81, 14, 3, 94, 35, 31, 28, 17, 13, 86)

foreach ($t in $types) {
    foreach ($num in $numbers) {
        python detect.py -d caltech -t $t -l $num
    }
}


