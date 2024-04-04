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

SVHN_SEEDS=(1 2 5 6 10)
HASH=(2024-03-08_00-26-05 2024-03-19_00-25-33 2024-03-19_05-01-27 2024-03-19_09-37-53 2024-03-19_14-18-15)

for seed in 0 1 2 3 4
do
    python ./main/run_testing.py --gpus 0 --seed ${SVHN_SEEDS[$seed]} \
                                --data svhn --server main \
                                --n-label-per-class 50 \
                                --mismatch-ratio 0.6 \
                                --backbone-type wide28_2 \
                                --checkpoint-hash ${HASH[$seed]} \
                                --for-what FIXMATCH
done

CIFAR100_SEEDS=(1 2 5 6 10)
HASH=(2023-09-06_23-39-19 2023-09-07_12-25-10 2023-09-08_08-18-48 2023-09-08_22-01-15 2023-09-09_14-36-12)

for seed in 0 1 2 3 4
do
    python ./main/run_testing.py --gpus 0 --seed ${CIFAR100_SEEDS[$seed]} \
                                --data cifar100 --server main \
                                --n-label-per-class 100 \
                                --n-valid-per-class 50 \
                                --mismatch-ratio 0.6 \
                                --backbone-type wide28_2 \
                                --checkpoint-hash ${HASH[$seed]} \
                                --for-what FIXMATCH
done

TINY_SEEDS=(1 2 5 6 10)
HASH=(2023-11-29_07-31-10 2023-11-29_14-13-54 2023-11-29_21-09-01 2023-11-30_04-12-26 2023-11-30_11-14-25)

for seed in 0 1 2 3 4
do
    python ./main/run_testing.py --gpus 0 --seed ${TINY_SEEDS[$seed]} \
                                --data tiny --server main \
                                --n-label-per-class 100 \
                                --n-valid-per-class 50 \
                                --mismatch-ratio 0.6 \
                                --backbone-type wide28_2 \
                                --checkpoint-hash ${HASH[$seed]} \
                                --for-what FIXMATCH
done