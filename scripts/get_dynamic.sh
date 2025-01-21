
# Gesture
for i in {0..9}; do
    python dynamic.py --dataset gesture --batch_size 4 --cupy --epochs 64 --train_epochs 1 --alpha 0.5 --beta 0.01 --trigger_label $i
done

# Caltech-101
# for i in {0..9}; do
#     python dynamic.py --dataset caltech --batch_size 4 --cupy --epochs 30 --train_epochs 1 --alpha 0.5 --beta 0.01 --trigger_label $i > cal_dyn$i.txt
# done

# CIFAR-10
# for i in {1..9}; do
#     python dynamic.py --dataset cifar10 --batch_size 4 --cupy --epochs 28 --train_epochs 1 --alpha 0.5 --beta 0.01 --trigger_label $i > co$i.txt
# done


# python detect.py -d gesture -t dynamic -l 0