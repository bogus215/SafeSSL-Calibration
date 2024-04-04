CIFAR10_SEEDS=(2 6 7 8 9)
HASH=(2024-02-14_22-35-30 2024-03-19_00-32-01 2024-03-19_07-41-57 2024-03-19_14-48-34 2024-03-19_21-56-54)

for seed in 0 1 2 3 4
do
    python ./main/run_testing.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
                                --data cifar10 --server main \
                                --n-label-per-class 400 \
                                --n-valid-per-class 500 \
                                --mismatch-ratio 0.6 \
                                --backbone-type wide28_2 \
                                --checkpoint-hash ${HASH[$seed]} \
                                --for-what OPENMATCH
done


SVHN_SEEDS=(1 2 5 6 10)
HASH=(2024-03-07_18-50-57 2024-03-19_00-32-53 2024-03-19_07-39-05 2024-03-19_14-41-11 2024-03-19_21-48-14)

for seed in 0 1 2 3 4
do
    python ./main/run_testing.py --gpus 0 --seed ${SVHN_SEEDS[$seed]} \
                                --data svhn --server main \
                                --n-label-per-class 50 \
                                --mismatch-ratio 0.6 \
                                --backbone-type wide28_2 \
                                --checkpoint-hash ${HASH[$seed]} \
                                --for-what OPENMATCH
done

CIFAR100_SEEDS=(1 2 5 6 10)
HASH=(2023-12-07_15-22-45 2023-12-28_03-30-36 2024-01-01_03-55-31 2024-01-06_13-42-41 2024-01-10_02-56-55)

for seed in 0 1 2 3 4
do
    python ./main/run_testing.py --gpus 0 --seed ${CIFAR100_SEEDS[$seed]} \
                                --data cifar100 --server main \
                                --n-label-per-class 100 \
                                --n-valid-per-class 50 \
                                --mismatch-ratio 0.6 \
                                --backbone-type wide28_2 \
                                --checkpoint-hash ${HASH[$seed]} \
                                --for-what OPENMATCH
done