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