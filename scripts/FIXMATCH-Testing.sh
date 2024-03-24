CIFAR10_SEEDS=(2 6 7 8 9)
HASH=(2023-09-06_19-04-09 2023-09-07_09-25-24 2023-09-08_03-11-00 2023-09-08_17-46-20 2023-09-09_10-26-00)

for seed in 0 1 2 3 4
do
    python ./main/run_testing.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
                                --data cifar10 --server main \
                                --n-label-per-class 400 \
                                --n-valid-per-class 500 \
                                --mismatch-ratio 0.6 \
                                --backbone-type wide28_2 \
                                --checkpoint-hash ${HASH[$seed]} \
                                --for-what FIXMATCH
done