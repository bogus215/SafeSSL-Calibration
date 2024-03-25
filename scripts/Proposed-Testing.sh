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
                                --for-what Proposed
done