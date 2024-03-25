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