CIFAR10_SEEDS=(2 6 7 8 9)
HASH=(2024-03-21_20-58-29 2024-04-02_00-46-56 2024-04-02_02-18-43 2024-04-02_03-50-53 2024-04-02_05-23-24)

for seed in 0 1 2 3 4
do
    python ./main/run_testing.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
                                --data cifar10 --server main \
                                --n-label-per-class 400 \
                                --n-valid-per-class 500 \
                                --mismatch-ratio 0.6 \
                                --backbone-type wide28_2 \
                                --checkpoint-hash ${HASH[$seed]} \
                                --for-what SafeStudent
done

SVHN_SEEDS=(1)
HASH=(2024-03-22_02-06-26)

for seed in 0
do
    python ./main/run_testing.py --gpus 0 --seed ${SVHN_SEEDS[$seed]} \
                                --data svhn --server main \
                                --n-label-per-class 50 \
                                --mismatch-ratio 0.6 \
                                --backbone-type wide28_2 \
                                --checkpoint-hash ${HASH[$seed]} \
                                --for-what SafeStudent
done