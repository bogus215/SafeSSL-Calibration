# CIFAR10_SEEDS=(9 7 8 6 2)
# HASH=(2024-03-21_23-13-39 2024-03-21_23-12-00 2024-03-21_17-45-19 2024-03-21_17-45-11 2024-03-11_18-48-44)

# for seed in 0 1 2 3 4
# do
#     python ./main/run_testing.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
#                                 --data cifar10 --server main \
#                                 --n-label-per-class 400 \
#                                 --n-valid-per-class 500 \
#                                 --mismatch-ratio 0.6 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${HASH[$seed]} \
#                                 --for-what Proposed --normalize
# done


# SVHN_SEEDS=(1 2 5 6 10)
# HASH=(2024-03-11_08-24-24 2024-03-18_20-44-53 2024-03-19_01-11-52 2024-03-19_05-52-59 2024-03-19_10-48-34)

# for seed in 0 1 2 3 4
# do
#     python ./main/run_testing.py --gpus 0 --seed ${SVHN_SEEDS[$seed]} \
#                                 --data svhn --server main \
#                                 --n-label-per-class 50 \
#                                 --mismatch-ratio 0.6 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${HASH[$seed]} \
#                                 --for-what Proposed
# done

# CIFAR100_SEEDS=(1 1 1)
# HASH=(2024-03-11_13-50-23 2024-04-03_14-20-48 2024-04-03_14-07-36)

# for seed in 0 1 2
# do
#     python ./main/run_testing.py --gpus 0 --seed ${CIFAR100_SEEDS[$seed]} \
#                                 --data cifar100 --server main \
#                                 --n-label-per-class 100 \
#                                 --n-valid-per-class 50 \
#                                 --mismatch-ratio 0.6 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${HASH[$seed]} \
#                                 --for-what Proposed --normalize
# done

# TINY_SEEDS=(1 2 5 6 10)
# HASH=(2024-04-04_11-45-38 2024-04-04_14-26-17 2024-04-04_17-06-33 2024-04-04_19-47-26)

# for seed in 0 1 2 3
# do
#     python ./main/run_testing.py --gpus 0 --seed ${TINY_SEEDS[$seed]} \
#                                 --data tiny --server main \
#                                 --n-label-per-class 100 \
#                                 --n-valid-per-class 50 \
#                                 --mismatch-ratio 0.6 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${HASH[$seed]} \
#                                 --for-what Proposed --normalize
# done


####################################################################################################################################################################################
CIFAR10_SEEDS=(2 6 7 8 9)
HASH=(2024-04-13_13-53-56 2024-04-20_16-38-39 2024-04-20_19-15-18 2024-04-20_21-51-57 2024-04-21_00-28-36)

for seed in 0 1 2 3 4
do
    python ./main/run_testing.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
                                --data cifar10 --server main \
                                --n-label-per-class 400 \
                                --n-valid-per-class 500 \
                                --mismatch-ratio 0.6 \
                                --backbone-type wide28_2 \
                                --checkpoint-hash ${HASH[$seed]} \
                                --for-what Proposed --normalize
done

TINY_SEEDS=(1 2 5 6 10)
HASH=(2024-04-10_14-11-36 2024-04-11_15-24-32 2024-04-11_20-50-34 2024-04-12_12-01-24 2024-04-12_18-31-18)

for seed in 0 1 2 3 4
do
    python ./main/run_testing.py --gpus 0 --seed ${TINY_SEEDS[$seed]} \
                                --data tiny --server main \
                                --n-label-per-class 100 \
                                --n-valid-per-class 50 \
                                --mismatch-ratio 0.6 \
                                --backbone-type wide28_2 \
                                --checkpoint-hash ${HASH[$seed]} \
                                --for-what Proposed --normalize
done

CIFAR100_SEEDS=(1 2 5 6 10)
HASH=(2024-04-11_13-09-49 2024-04-11_10-32-36 2024-04-11_15-47-45 2024-04-11_21-03-31 2024-04-11_18-25-40)

for seed in 0 1 2 3 4
do
    python ./main/run_testing.py --gpus 0 --seed ${CIFAR100_SEEDS[$seed]} \
                                --data cifar100 --server main \
                                --n-label-per-class 100 \
                                --n-valid-per-class 50 \
                                --mismatch-ratio 0.6 \
                                --backbone-type wide28_2 \
                                --checkpoint-hash ${HASH[$seed]} \
                                --for-what Proposed --normalize
done

# TINY_SEEDS=(1)
# HASH=(2024-04-10_14-11-36)

# for seed in 0
# do
#     python ./main/run_testing.py --gpus 0 --seed ${TINY_SEEDS[$seed]} \
#                                 --data tiny --server main \
#                                 --n-label-per-class 100 \
#                                 --n-valid-per-class 50 \
#                                 --mismatch-ratio 0.6 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${HASH[$seed]} \
#                                 --for-what Proposed --normalize
# done

# CIFAR100_SEEDS=(1)
# HASH=(2024-04-10_06-46-05)

# for seed in 0
# do
#     python ./main/run_testing.py --gpus 0 --seed ${CIFAR100_SEEDS[$seed]} \
#                                 --data cifar100 --server main \
#                                 --n-label-per-class 100 \
#                                 --n-valid-per-class 50 \
#                                 --mismatch-ratio 0.6 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${HASH[$seed]} \
#                                 --for-what Proposed --normalize
# done


CIFAR10_SEEDS=(2 6 7 8 9)
PI_1=(2024-04-13_13-53-56 2024-04-20_16-38-39 2024-04-20_19-15-18 2024-04-20_21-51-57 2024-04-21_00-28-36)
PI_2=(2024-04-17_10-25-16 2024-04-18_02-04-11 2024-04-18_17-43-29 2024-04-19_09-23-26 2024-04-20_01-01-03)
PI_3=(2024-04-17_15-38-37 2024-04-18_07-18-26 2024-04-18_22-58-02 2024-04-19_14-37-06 2024-04-20_06-14-41)
PI_4=(2024-04-17_20-51-37 2024-04-18_12-32-13 2024-04-19_04-12-29 2024-04-19_19-50-26 2024-04-20_11-28-05)

for seed in 0 1 2 3 4
do
    python ./main/run_testing.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
                                --data cifar10 --server main \
                                --n-label-per-class 400 \
                                --n-valid-per-class 500 \
                                --mismatch-ratio 0.6 \
                                --backbone-type wide28_2 \
                                --checkpoint-hash ${PI_1[$seed]} \
                                --for-what Proposed --normalize --ova-pi 0.5
done

for seed in 0 1 2 3 4
do
    python ./main/run_testing.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
                                --data cifar10 --server main \
                                --n-label-per-class 400 \
                                --n-valid-per-class 500 \
                                --mismatch-ratio 0.6 \
                                --backbone-type wide28_2 \
                                --checkpoint-hash ${PI_2[$seed]} \
                                --for-what Proposed --normalize --ova-pi 0.6
done

for seed in 0 1 2 3 4
do
    python ./main/run_testing.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
                                --data cifar10 --server main \
                                --n-label-per-class 400 \
                                --n-valid-per-class 500 \
                                --mismatch-ratio 0.6 \
                                --backbone-type wide28_2 \
                                --checkpoint-hash ${PI_3[$seed]} \
                                --for-what Proposed --normalize --ova-pi 0.7
done

for seed in 0 1 2 3 4
do
    python ./main/run_testing.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
                                --data cifar10 --server main \
                                --n-label-per-class 400 \
                                --n-valid-per-class 500 \
                                --mismatch-ratio 0.6 \
                                --backbone-type wide28_2 \
                                --checkpoint-hash ${PI_4[$seed]} \
                                --for-what Proposed --normalize --ova-pi 0.8
done