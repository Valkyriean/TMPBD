# python adaptive.py --weight 0
# python adaptive.py --weight 1e-7
# python adaptive.py --weight 1e-6
# python adaptive.py --weight 1e-5
# python adaptive.py --weight 1e-4
# python adaptive.py --weight 1e-3

python detect.py --model_name gesture-static-adp-0-1e-05
python detect.py --model_name gesture-static-adp-0-1e-06
python detect.py --model_name gesture-static-adp-0-1e-07
python detect.py --model_name gesture-static-adp-0-0.0
python detect.py --model_name gesture-static-adp-0-0.1