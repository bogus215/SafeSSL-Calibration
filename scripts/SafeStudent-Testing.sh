CIFAR10_SEEDS=(2)
HASH=(2024-03-21_20-58-29)

for seed in 0
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