# CIFAR10_SEEDS=(2 6 7 8 9)
# HASH=(2024-02-14_22-35-30 2024-03-19_00-32-01 2024-03-19_07-41-57 2024-03-19_14-48-34 2024-03-19_21-56-54)

# for seed in 0 1 2 3 4
# do
#     python ./main/run_testing.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
#                                 --data cifar10 --server main \
#                                 --n-label-per-class 400 \
#                                 --n-valid-per-class 500 \
#                                 --mismatch-ratio 0.6 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${HASH[$seed]} \
#                                 --for-what OPENMATCH
# done


# SVHN_SEEDS=(1 2 5 6 10)
# HASH=(2024-03-07_18-50-57 2024-03-19_00-32-53 2024-03-19_07-39-05 2024-03-19_14-41-11 2024-03-19_21-48-14)

# for seed in 0 1 2 3 4
# do
#     python ./main/run_testing.py --gpus 0 --seed ${SVHN_SEEDS[$seed]} \
#                                 --data svhn --server main \
#                                 --n-label-per-class 50 \
#                                 --mismatch-ratio 0.6 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${HASH[$seed]} \
#                                 --for-what OPENMATCH
# done

# CIFAR100_SEEDS=(1 2 5 6 10)
# HASH=(2023-12-07_15-22-45 2023-12-28_03-30-36 2024-01-01_03-55-31 2024-01-06_13-42-41 2024-01-10_02-56-55)

# for seed in 0 1 2 3 4
# do
#     python ./main/run_testing.py --gpus 0 --seed ${CIFAR100_SEEDS[$seed]} \
#                                 --data cifar100 --server main \
#                                 --n-label-per-class 100 \
#                                 --n-valid-per-class 50 \
#                                 --mismatch-ratio 0.6 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${HASH[$seed]} \
#                                 --for-what OPENMATCH
# done

# TINY_SEEDS=(1 2 5 6 10)
# HASH=(2023-12-07_16-37-39 2023-12-28_13-17-33 2024-01-01_17-59-53 2024-01-07_03-38-00 2024-01-10_12-57-37)

# for seed in 0 1 2 3 4
# do
#     python ./main/run_testing.py --gpus 0 --seed ${TINY_SEEDS[$seed]} \
#                                 --data tiny --server main \
#                                 --n-label-per-class 100 \
#                                 --n-valid-per-class 50 \
#                                 --mismatch-ratio 0.6 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${HASH[$seed]} \
#                                 --for-what OPENMATCH
# done


# CIFAR10_SEEDS=(2 6 7 8 9)

# PI_2=(2024-04-29_10-42-11 2024-04-29_20-40-53 2024-04-30_06-41-46 2024-04-30_16-45-57 2024-05-01_02-48-37)
# PI_3=(2024-04-29_14-01-32 2024-04-30_00-00-51 2024-04-30_10-03-32 2024-04-30_20-07-00 2024-05-01_06-09-14)
# PI_4=(2024-04-29_17-21-14 2024-04-30_03-20-51 2024-04-30_13-24-41 2024-04-30_23-27-57 2024-05-01_09-29-38)

# for seed in 0 1 2 3 4
# do
#     python ./main/run_testing.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
#                                 --data cifar10 --server main \
#                                 --n-label-per-class 400 \
#                                 --n-valid-per-class 500 \
#                                 --mismatch-ratio 0.6 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${PI_2[$seed]} \
#                                 --for-what OPENMATCH --ova-pi 0.6
# done

# for seed in 0 1 2 3 4
# do
#     python ./main/run_testing.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
#                                 --data cifar10 --server main \
#                                 --n-label-per-class 400 \
#                                 --n-valid-per-class 500 \
#                                 --mismatch-ratio 0.6 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${PI_3[$seed]} \
#                                 --for-what OPENMATCH --ova-pi 0.7
# done

# for seed in 0 1 2 3 4
# do
#     python ./main/run_testing.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
#                                 --data cifar10 --server main \
#                                 --n-label-per-class 400 \
#                                 --n-valid-per-class 500 \
#                                 --mismatch-ratio 0.6 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${PI_4[$seed]} \
#                                 --for-what OPENMATCH --ova-pi 0.8
# done


# python ./main/run_testing.py --gpus 0 --seed 2 \
#                             --data cifar10 --server main \
#                             --n-label-per-class 400 \
#                             --n-valid-per-class 500 \
#                             --mismatch-ratio 0.6 \
#                             --backbone-type wide28_2 \
#                             --checkpoint-hash 2024-07-31_04-35-13 \
#                             --for-what OPENMATCH+Smoothing

# python ./main/run_testing.py --gpus 0 --seed 2 \
#                             --data cifar10 --server main \
#                             --n-label-per-class 400 \
#                             --n-valid-per-class 500 \
#                             --mismatch-ratio 0.6 \
#                             --backbone-type wide28_2 \
#                             --checkpoint-hash 2024-07-31_17-52-19 \
#                             --for-what OPENMATCH+Mixup


# python ./main/run_testing.py --gpus 0 --seed 6 \
#                             --data cifar10 --server main \
#                             --n-label-per-class 400 \
#                             --n-valid-per-class 500 \
#                             --mismatch-ratio 0.6 \
#                             --backbone-type wide28_2 \
#                             --checkpoint-hash 2024-08-02_08-29-38 \
#                             --for-what OPENMATCH+Mixup

# python ./main/run_testing.py --gpus 0 --seed 7 \
#                             --data cifar10 --server main \
#                             --n-label-per-class 400 \
#                             --n-valid-per-class 500 \
#                             --mismatch-ratio 0.6 \
#                             --backbone-type wide28_2 \
#                             --checkpoint-hash 2024-08-02_09-50-02 \
#                             --for-what OPENMATCH+Mixup


# python ./main/run_testing.py --gpus 0 --seed 8 \
#                             --data cifar10 --server main \
#                             --n-label-per-class 400 \
#                             --n-valid-per-class 500 \
#                             --mismatch-ratio 0.6 \
#                             --backbone-type wide28_2 \
#                             --checkpoint-hash 2024-08-02_10-48-07 \
#                             --for-what OPENMATCH+Mixup

# python ./main/run_testing.py --gpus 0 --seed 9 \
#                             --data cifar10 --server main \
#                             --n-label-per-class 400 \
#                             --n-valid-per-class 500 \
#                             --mismatch-ratio 0.6 \
#                             --backbone-type wide28_2 \
#                             --checkpoint-hash 2024-08-02_11-48-06 \
#                             --for-what OPENMATCH+Mixup

# python ./main/run_testing.py --gpus 0 --seed 7 \
#                             --data cifar10 --server main \
#                             --n-label-per-class 400 \
#                             --n-valid-per-class 500 \
#                             --mismatch-ratio 0.6 \
#                             --backbone-type wide28_2 \
#                             --checkpoint-hash 2024-08-02_02-02-47 \
#                             --for-what OPENMATCH+Smoothing

# python ./main/run_testing.py --gpus 0 --seed 8 \
#                             --data cifar10 --server main \
#                             --n-label-per-class 400 \
#                             --n-valid-per-class 500 \
#                             --mismatch-ratio 0.6 \
#                             --backbone-type wide28_2 \
#                             --checkpoint-hash 2024-08-02_03-03-20 \
#                             --for-what OPENMATCH+Smoothing

# python ./main/run_testing.py --gpus 0 --seed 9 \
#                             --data cifar10 --server main \
#                             --n-label-per-class 400 \
#                             --n-valid-per-class 500 \
#                             --mismatch-ratio 0.6 \
#                             --backbone-type wide28_2 \
#                             --checkpoint-hash 2024-08-02_03-53-47 \
#                             --for-what OPENMATCH+Smoothing

# python ./main/run_testing.py --gpus 0 --seed 2 \
#                             --data cifar10 --server main \
#                             --n-label-per-class 400 \
#                             --n-valid-per-class 500 \
#                             --mismatch-ratio 0.6 \
#                             --backbone-type wide28_2 \
#                             --checkpoint-hash 2024-07-31_08-51-14 \
#                             --for-what OPENMATCH+MMCE

# python ./main/run_testing.py --gpus 0 --seed 6 \
#                             --data cifar10 --server main \
#                             --n-label-per-class 400 \
#                             --n-valid-per-class 500 \
#                             --mismatch-ratio 0.6 \
#                             --backbone-type wide28_2 \
#                             --checkpoint-hash 2024-08-01_15-50-26 \
#                             --for-what OPENMATCH+MMCE

# python ./main/run_testing.py --gpus 0 --seed 7 \
#                             --data cifar10 --server main \
#                             --n-label-per-class 400 \
#                             --n-valid-per-class 500 \
#                             --mismatch-ratio 0.6 \
#                             --backbone-type wide28_2 \
#                             --checkpoint-hash 2024-08-01_17-07-43 \
#                             --for-what OPENMATCH+MMCE

# python ./main/run_testing.py --gpus 0 --seed 8 \
#                             --data cifar10 --server main \
#                             --n-label-per-class 400 \
#                             --n-valid-per-class 500 \
#                             --mismatch-ratio 0.6 \
#                             --backbone-type wide28_2 \
#                             --checkpoint-hash 2024-08-01_19-21-03 \
#                             --for-what OPENMATCH+MMCE

# python ./main/run_testing.py --gpus 0 --seed 9 \
#                             --data cifar10 --server main \
#                             --n-label-per-class 400 \
#                             --n-valid-per-class 500 \
#                             --mismatch-ratio 0.6 \
#                             --backbone-type wide28_2 \
#                             --checkpoint-hash 2024-08-01_20-54-28 \
#                             --for-what OPENMATCH+MMCE

#####
# 0.86700 0.08200 0.85500 0.12000
# 0.86800 0.08105 0.84830 0.12628
# 0.86200 0.08722 0.85265 0.12144
# 0.87233 0.07869 0.86824 0.10637
# 0.87350 0.07592 0.84505 0.13230

# 0.87183 0.10005 0.83276 0.14265
# 0.84433 0.10455 0.83515 0.12562
# 0.86000 0.09000 0.81095 0.15858
# 0.85717 0.09870 0.84531 0.12073
# 0.84717 0.10236 0.83693 0.12597
 
# 0.83050 0.06636 0.87302 0.08696
# 0.82167 0.07816 0.86953 0.08796
# 0.83683 0.09086 0.89789 0.06035
# 0.84783 0.09269 0.91025 0.05178
# 0.83783 0.04510 0.88675 0.06837

# 0.86867 0.06031 0.90193 0.04498
# 0.86133 0.06250 0.83446 0.10908
# 0.86850 0.05873 0.89382 0.04458
# 0.86150 0.05690 0.89235 0.04757
# 0.86883 0.06428 0.87053 0.07904

python ./main/run_testing.py --gpus 0 --seed 2 \
                            --data cifar10 --server main \
                            --n-label-per-class 400 \
                            --n-valid-per-class 500 \
                            --mismatch-ratio 0.6 \
                            --backbone-type wide28_2 \
                            --checkpoint-hash 2025-03-01_01-47-53 \
                            --for-what OPENMATCH+MbLS

python ./main/run_testing.py --gpus 0 --seed 2 \
                            --data cifar10 --server main \
                            --n-label-per-class 400 \
                            --n-valid-per-class 500 \
                            --mismatch-ratio 0.6 \
                            --backbone-type wide28_2 \
                            --checkpoint-hash 2025-03-01_06-30-42 \
                            --for-what OPENMATCH+RankMixup