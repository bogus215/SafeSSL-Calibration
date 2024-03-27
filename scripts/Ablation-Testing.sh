CIFAR10_SEEDS=(2 6 7 8 9)
HASH=(2024-03-21_17-54-25 2024-03-22_12-25-14 2024-03-23_02-45-43 2024-03-23_17-07-49 2024-03-24_07-29-45)

for seed in 0 1 2 3 4
do
    python ./main/run_testing.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
                                --data cifar10 --server main \
                                --n-label-per-class 400 \
                                --n-valid-per-class 500 \
                                --mismatch-ratio 0.6 \
                                --backbone-type wide28_2 \
                                --checkpoint-hash ${HASH[$seed]} \
                                --for-what Ablation1 --normalize
done

CIFAR10_SEEDS=(2 6 7 8 9)
HASH=(2024-03-22_01-36-31 2024-03-22_17-18-14 2024-03-23_07-38-08 2024-03-23_22-00-29 2024-03-24_12-22-02)

for seed in 0 1 2 3 4
do
    python ./main/run_testing.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
                                --data cifar10 --server main \
                                --n-label-per-class 400 \
                                --n-valid-per-class 500 \
                                --mismatch-ratio 0.6 \
                                --backbone-type wide28_2 \
                                --checkpoint-hash ${HASH[$seed]} \
                                --for-what Ablation2 --normalize
done