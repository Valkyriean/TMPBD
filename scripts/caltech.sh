declare -a arr=("81" "14" "3" "94" "35" "31" "28" "17" "13" "86")

## now loop through the above array
for i in "${arr[@]}"
do
   python detect.py -d caltech -t dynamic -l "$i"

done