#!/bin/bash
python mitigate.py --algorithm e2e --nsample 20 -t clean -l 0
python mitigate.py --algorithm e2e --nsample 20 -t clean -l 1
python mitigate.py --algorithm e2e --nsample 20 -t clean -l 2
python mitigate.py --algorithm e2e --nsample 20 -t clean -l 3
python mitigate.py --algorithm e2e --nsample 20 -t clean -l 4
python mitigate.py --algorithm e2e --nsample 20 -t clean -l 5
python mitigate.py --algorithm e2e --nsample 20 -t clean -l 6
python mitigate.py --algorithm e2e --nsample 20 -t clean -l 7
python mitigate.py --algorithm e2e --nsample 20 -t clean -l 9
python mitigate.py --algorithm e2e --nsample 20 -t moving -l 7
python mitigate.py --algorithm e2e --nsample 20 -t moving -l 9
python mitigate.py --algorithm e2e --nsample 20 -t static -l 9