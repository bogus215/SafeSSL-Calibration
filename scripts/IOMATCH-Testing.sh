CIFAR10_SEEDS=(2 6 7 8 9)
HASH=(2023-12-01_08-56-11 2023-12-02_03-37-55 2023-12-02_18-31-37 2023-12-03_09-40-57 2023-12-04_00-31-11)

for seed in 0 1 2 3 4
do
    python ./main/run_testing.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
                                --data cifar10 --server main \
                                --n-label-per-class 400 \
                                --n-valid-per-class 500 \
                                --mismatch-ratio 0.6 \
                                --backbone-type wide28_2 \
                                --checkpoint-hash ${HASH[$seed]} \
                                --for-what IOMATCH
done

SVHN_SEEDS=(1 2 5 6 10)
HASH=(2023-12-01_12-41-54 2023-12-02_03-36-16 2023-12-02_18-40-01 2023-12-03_09-51-48 2023-12-04_01-01-18)

for seed in 0 1 2 3 4
do
    python ./main/run_testing.py --gpus 0 --seed ${SVHN_SEEDS[$seed]} \
                                --data svhn --server main \
                                --n-label-per-class 50 \
                                --mismatch-ratio 0.6 \
                                --backbone-type wide28_2 \
                                --checkpoint-hash ${HASH[$seed]} \
                                --for-what IOMATCH
done


CIFAR100_SEEDS=(1 2 5 6 10)
HASH=(2023-12-01_12-30-34 2023-12-02_03-16-55 2023-12-02_18-27-51 2023-12-03_09-39-52 2023-12-04_00-21-34)

for seed in 0 1 2 3 4
do
    python ./main/run_testing.py --gpus 0 --seed ${CIFAR100_SEEDS[$seed]} \
                                --data cifar100 --server main \
                                --n-label-per-class 100 \
                                --n-valid-per-class 50 \
                                --mismatch-ratio 0.6 \
                                --backbone-type wide28_2 \
                                --checkpoint-hash ${HASH[$seed]} \
                                --for-what IOMATCH
done

TINY_SEEDS=(1 2 5 6 10)
HASH=(2023-12-01_12-33-36 2023-12-02_03-34-54 2023-12-02_18-11-39 2023-12-03_09-22-29 2023-12-04_00-35-57)

for seed in 0 1 2 3 4
do
    python ./main/run_testing.py --gpus 0 --seed ${TINY_SEEDS[$seed]} \
                                --data tiny --server main \
                                --n-label-per-class 100 \
                                --n-valid-per-class 50 \
                                --mismatch-ratio 0.6 \
                                --backbone-type wide28_2 \
                                --checkpoint-hash ${HASH[$seed]} \
                                --for-what IOMATCH
done