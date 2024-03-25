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