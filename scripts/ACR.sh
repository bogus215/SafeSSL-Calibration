CIFAR10_SEEDS=(2 6 7 8 9)
CIFAR100_SEEDS=(1 2 5 6 10)
TINY_SEEDS=(1 2 5 6 10)
SVHN_SEEDS=(1 2 5 6 10)

for seed in 0
do
    for ratio in 0.3
    do
        python ./main/run_ACR.py --gpus 0 --seed ${TINY_SEEDS[$seed]} \
                                    --data tiny --server main \
                                    --n-label-per-class 100 \
                                    --n-valid-per-class 50 \
                                    --mismatch-ratio $ratio --mixed-precision \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --weight-decay 0 --warm-up 1
    done
done