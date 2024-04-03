CIFAR10_SEEDS=(9 7 8 6 2)
HASH=(2024-03-21_23-13-39 2024-03-21_23-12-00 2024-03-21_17-45-19 2024-03-21_17-45-11 2024-03-11_18-48-44)

for seed in 0 1 2 3 4
do
    python ./main/run_testing.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
                                --data cifar10 --server main \
                                --n-label-per-class 400 \
                                --n-valid-per-class 500 \
                                --mismatch-ratio 0.6 \
                                --backbone-type wide28_2 \
                                --checkpoint-hash ${HASH[$seed]} \
                                --for-what Proposed --normalize
done


SVHN_SEEDS=(1 2 5 6 10)
HASH=(2024-03-11_08-24-24 2024-03-18_20-44-53 2024-03-19_01-11-52 2024-03-19_05-52-59 2024-03-19_10-48-34)

for seed in 0 1 2 3 4
do
    python ./main/run_testing.py --gpus 0 --seed ${SVHN_SEEDS[$seed]} \
                                --data svhn --server main \
                                --n-label-per-class 50 \
                                --mismatch-ratio 0.6 \
                                --backbone-type wide28_2 \
                                --checkpoint-hash ${HASH[$seed]} \
                                --for-what Proposed
done

CIFAR100_SEEDS=(1 1)
HASH=(2024-03-11_13-50-23 2024-04-03_14-20-48)

for seed in 0 1
do
    python ./main/run_testing.py --gpus 0 --seed ${CIFAR100_SEEDS[$seed]} \
                                --data cifar100 --server main \
                                --n-label-per-class 100 \
                                --n-valid-per-class 50 \
                                --mismatch-ratio 0.6 \
                                --backbone-type wide28_2 \
                                --checkpoint-hash ${HASH[$seed]} \
                                --for-what Proposed --normalize
done