CIFAR10_SEEDS=(2 6 7 8 9)
HASH=(2024-03-14_17-14-22 2024-03-19_18-12-27 2024-03-20_07-56-49 2024-03-19_18-21-35 2024-03-19_18-18-01)

for seed in 0 1 2 3 4
do
    python ./main/run_testing.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
                                --data cifar10 --server main \
                                --n-label-per-class 400 \
                                --n-valid-per-class 500 \
                                --mismatch-ratio 0.6 \
                                --backbone-type wide28_2 \
                                --checkpoint-hash ${HASH[$seed]} \
                                --for-what MTC
done

SVHN_SEEDS=(1 2 5 6 10)
HASH=(2024-03-14_21-05-16 2024-03-19_23-57-57 2024-03-20_13-09-08 2024-03-19_21-42-40 2024-03-20_00-38-31)

for seed in 0 1 2 3 4
do
    python ./main/run_testing.py --gpus 0 --seed ${SVHN_SEEDS[$seed]} \
                                --data svhn --server main \
                                --n-label-per-class 50 \
                                --mismatch-ratio 0.6 \
                                --backbone-type wide28_2 \
                                --checkpoint-hash ${HASH[$seed]} \
                                --for-what MTC
done

CIFAR100_SEEDS=(1 2 5 6 10)
HASH=(Cifar100-Seed1 Cifar100-Seed2 Cifar100-Seed3 Cifar100-Seed4 Cifar100-Seed5)

for seed in 0 1 2 3 4
do
    python ./main/run_testing.py --gpus 0 --seed ${CIFAR100_SEEDS[$seed]} \
                                --data cifar100 --server main \
                                --n-label-per-class 100 \
                                --n-valid-per-class 50 \
                                --mismatch-ratio 0.6 \
                                --backbone-type wide28_2 \
                                --checkpoint-hash ${HASH[$seed]} \
                                --for-what MTC
done

TINY_SEEDS=(1 2 5 6 10)
HASH=(Tiny-Seed1 Tiny-Seed2 Tiny-Seed3 Tiny-Seed4 Tiny-Seed5)

for seed in 0 1 2 3 4
do
    python ./main/run_testing.py --gpus 0 --seed ${TINY_SEEDS[$seed]} \
                                --data tiny --server main \
                                --n-label-per-class 100 \
                                --n-valid-per-class 50 \
                                --mismatch-ratio 0.6 \
                                --backbone-type wide28_2 \
                                --checkpoint-hash ${HASH[$seed]} \
                                --for-what MTC
done