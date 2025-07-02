# CIFAR10_SEEDS=(2 6 7 8 9)
# HASH=(2024-03-21_17-54-25 2024-03-22_12-25-14 2024-03-23_02-45-43 2024-03-23_17-07-49 2024-03-24_07-29-45)

# for seed in 0 1 2 3 4
# do
#     python ./main/run_testing.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
#                                 --data cifar10 --server main \
#                                 --n-label-per-class 400 \
#                                 --n-valid-per-class 500 \
#                                 --mismatch-ratio 0.6 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${HASH[$seed]} \
#                                 --for-what Ablation1 --normalize
# done

# CIFAR10_SEEDS=(2 6 7 8 9)
# HASH=(2024-03-22_01-36-31 2024-03-22_17-18-14 2024-03-23_07-38-08 2024-03-23_22-00-29 2024-03-24_12-22-02)

# for seed in 0 1 2 3 4
# do
#     python ./main/run_testing.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
#                                 --data cifar10 --server main \
#                                 --n-label-per-class 400 \
#                                 --n-valid-per-class 500 \
#                                 --mismatch-ratio 0.6 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${HASH[$seed]} \
#                                 --for-what Ablation2 --normalize
# done

# CIFAR10_SEEDS=(2 6)
# HASH=(2024-03-27_18-13-42 2024-03-28_04-19-58)

# for seed in 0 1
# do
#     python ./main/run_testing.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
#                                 --data cifar10 --server main \
#                                 --n-label-per-class 400 \
#                                 --n-valid-per-class 500 \
#                                 --mismatch-ratio 0.6 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${HASH[$seed]} \
#                                 --for-what Ablation4 --normalize
# done

# CIFAR10_SEEDS=(2 6)
# HASH=(2024-03-27_18-17-56 2024-03-28_04-26-51)

# for seed in 0 1
# do
#     python ./main/run_testing.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
#                                 --data cifar10 --server main \
#                                 --n-label-per-class 400 \
#                                 --n-valid-per-class 500 \
#                                 --mismatch-ratio 0.9 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${HASH[$seed]} \
#                                 --for-what Ablation4 --normalize
# done

# CIFAR10_SEEDS=(2 6)
# HASH=(2024-03-27_23-19-48 2024-03-28_09-19-33)

# for seed in 0 1
# do
#     python ./main/run_testing.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
#                                 --data cifar10 --server main \
#                                 --n-label-per-class 400 \
#                                 --n-valid-per-class 500 \
#                                 --mismatch-ratio 0.6 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${HASH[$seed]} \
#                                 --for-what Ablation5 --normalize
# done

# CIFAR10_SEEDS=(2 6)
# HASH=(2024-03-27_23-26-12 2024-03-28_09-34-33)

# for seed in 0 1
# do
#     python ./main/run_testing.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
#                                 --data cifar10 --server main \
#                                 --n-label-per-class 400 \
#                                 --n-valid-per-class 500 \
#                                 --mismatch-ratio 0.9 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${HASH[$seed]} \
#                                 --for-what Ablation5 --normalize
# done


# SVHN_SEEDS=(1 2 5 6 10)
# HASH=(2024-03-28_20-30-37 2024-03-29_01-42-04 2024-03-29_06-41-02 2024-03-28_20-30-30 2024-03-29_01-40-21)

# for seed in 0 1 2 3 4
# do
#     python ./main/run_testing.py --gpus 0 --seed ${SVHN_SEEDS[$seed]} \
#                                 --data svhn --server workstation3 \
#                                 --n-label-per-class 50 \
#                                 --mismatch-ratio 0.6 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${HASH[$seed]} \
#                                 --for-what Ablation1
# done

# SVHN_SEEDS=(1 2 5 6 10)
# HASH=(2024-03-28_20-30-44 2024-03-29_01-29-52 2024-03-29_06-27-30 2024-03-29_06-45-08)

# for seed in 0 1 2 3
# do
#     python ./main/run_testing.py --gpus 0 --seed ${SVHN_SEEDS[$seed]} \
#                                 --data svhn --server workstation3 \
#                                 --n-label-per-class 50 \
#                                 --mismatch-ratio 0.6 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${HASH[$seed]} \
#                                 --for-what Ablation2
# done

# CIFAR100_SEEDS=(1 5 10 1)
# HASH=(2024-03-29_13-36-05 2024-03-30_00-05-08 2024-03-30_10-09-20 2024-04-03_00-28-17)

# for seed in 0 1 2 3
# do
#     python ./main/run_testing.py --gpus 0 --seed ${CIFAR100_SEEDS[$seed]} \
#                                 --data cifar100 --server main \
#                                 --n-label-per-class 100 \
#                                 --n-valid-per-class 50 \
#                                 --mismatch-ratio 0.6 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${HASH[$seed]} \
#                                 --for-what Ablation1
# done

# CIFAR100_SEEDS=(1 5 10)
# HASH=(2024-03-29_16-05-35 2024-03-30_02-34-42 2024-03-30_12-39-24)

# for seed in 0 1 2
# do
#     python ./main/run_testing.py --gpus 0 --seed ${CIFAR100_SEEDS[$seed]} \
#                                 --data cifar100 --server main \
#                                 --n-label-per-class 100 \
#                                 --n-valid-per-class 50 \
#                                 --mismatch-ratio 0.6 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${HASH[$seed]} \
#                                 --for-what Ablation2
# done

# TINY_SEEDS=(1 5 10 1)
# HASH=(2024-03-29_18-33-05 2024-03-30_05-02-48 2024-03-30_15-07-26 2024-04-02_08-29-55)

# for seed in 0 1 2 3
# do
#     python ./main/run_testing.py --gpus 0 --seed ${TINY_SEEDS[$seed]} \
#                                 --data tiny --server main \
#                                 --n-label-per-class 100 \
#                                 --n-valid-per-class 50 \
#                                 --mismatch-ratio 0.6 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${HASH[$seed]} \
#                                 --for-what Ablation1
# done

# TINY_SEEDS=(1 5 10)
# HASH=(2024-03-29_21-32-50 2024-03-30_07-36-40 2024-03-30_17-41-53)

# for seed in 0 1 2
# do
#     python ./main/run_testing.py --gpus 0 --seed ${TINY_SEEDS[$seed]} \
#                                 --data tiny --server main \
#                                 --n-label-per-class 100 \
#                                 --n-valid-per-class 50 \
#                                 --mismatch-ratio 0.6 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${HASH[$seed]} \
#                                 --for-what Ablation2
# done


#####################################################################################################################################
# CIFAR10_SEEDS=(2 6 7 8 9)
# SVHN_SEEDS=(1 2 5 6 10)
# CIFAR100_SEEDS=(1 2 5 6 10)
# TINY_SEEDS=(1 2 5 6 10)

# CIFAR_HASH_1=(2024-04-06_20-24-35 2024-04-06_20-25-10 2024-04-06_20-25-34 2024-04-06_20-25-44 2024-04-06_20-25-53)
# CIFAR_HASH_2=(2024-04-07_01-34-14 2024-04-07_01-27-20 2024-04-07_01-33-02 2024-04-07_01-33-33 2024-04-07_01-34-51)

# SVHN_HASH_1=(2024-04-08_03-23-11 2024-04-08_02-50-49 2024-04-08_03-06-35 2024-04-08_03-08-32 2024-04-08_03-08-21)
# SVHN_HASH_2=(2024-04-08_08-24-14 2024-04-08_07-50-14 2024-04-08_08-16-51 2024-04-08_08-17-05 2024-04-08_08-17-37)

# CIFAR100_HASH_1=(2024-04-07_06-32-19 2024-04-07_06-20-13 2024-04-07_06-27-18 2024-04-07_06-39-42 2024-04-07_06-32-30)
# CIFAR100_HASH_2=(2024-04-07_11-38-33 2024-04-07_11-24-09 2024-04-07_11-35-02 2024-04-07_11-45-05 2024-04-07_11-36-23)

# TINY_HASH_1=(2024-04-07_16-36-44 2024-04-07_16-21-44 2024-04-07_16-34-16 2024-04-07_16-42-38 2024-04-07_16-39-29)
# TINY_HASH_2=(2024-04-07_22-04-50 2024-04-07_21-41-06 2024-04-07_21-54-51 2024-04-07_22-03-43 2024-04-07_21-58-08)

# for seed in 0 1 2 3 4
# do
#     python ./main/run_testing.py --gpus 5 --seed ${CIFAR10_SEEDS[$seed]} \
#                                 --data cifar10 --server workstation3 \
#                                 --n-label-per-class 400 \
#                                 --n-valid-per-class 500 \
#                                 --mismatch-ratio 0.6 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${CIFAR_HASH_1[$seed]} \
#                                 --for-what Ablation1 --normalize

#     python ./main/run_testing.py --gpus 5 --seed ${CIFAR10_SEEDS[$seed]} \
#                                 --data cifar10 --server workstation3 \
#                                 --n-label-per-class 400 \
#                                 --n-valid-per-class 500 \
#                                 --mismatch-ratio 0.6 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${CIFAR_HASH_2[$seed]} \
#                                 --for-what Ablation2 --normalize

#     python ./main/run_testing.py --gpus 5 --seed ${SVHN_SEEDS[$seed]} \
#                                 --data svhn --server workstation3 \
#                                 --n-label-per-class 50 \
#                                 --mismatch-ratio 0.6 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${SVHN_HASH_1[$seed]} \
#                                 --for-what Ablation1

#     python ./main/run_testing.py --gpus 5 --seed ${SVHN_SEEDS[$seed]} \
#                                 --data svhn --server workstation3 \
#                                 --n-label-per-class 50 \
#                                 --mismatch-ratio 0.6 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${SVHN_HASH_2[$seed]} \
#                                 --for-what Ablation2

#     python ./main/run_testing.py --gpus 5 --seed ${CIFAR100_SEEDS[$seed]} \
#                                 --data cifar100 --server workstation3 \
#                                 --n-label-per-class 100 \
#                                 --n-valid-per-class 50 \
#                                 --mismatch-ratio 0.6 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${CIFAR100_HASH_1[$seed]} \
#                                 --for-what Ablation1

#     python ./main/run_testing.py --gpus 5 --seed ${CIFAR100_SEEDS[$seed]} \
#                                 --data cifar100 --server workstation3 \
#                                 --n-label-per-class 100 \
#                                 --n-valid-per-class 50 \
#                                 --mismatch-ratio 0.6 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${CIFAR100_HASH_2[$seed]} \
#                                 --for-what Ablation2

#     python ./main/run_testing.py --gpus 5 --seed ${TINY_SEEDS[$seed]} \
#                                 --data tiny --server workstation3 \
#                                 --n-label-per-class 100 \
#                                 --n-valid-per-class 50 \
#                                 --mismatch-ratio 0.6 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${TINY_HASH_1[$seed]} \
#                                 --for-what Ablation1 --normalize

#     python ./main/run_testing.py --gpus 5 --seed ${TINY_SEEDS[$seed]} \
#                                 --data tiny --server workstation3 \
#                                 --n-label-per-class 100 \
#                                 --n-valid-per-class 50 \
#                                 --mismatch-ratio 0.6 \
#                                 --backbone-type wide28_2 \
#                                 --checkpoint-hash ${TINY_HASH_2[$seed]} \
#                                 --for-what Ablation2 --normalize

# done

CIFAR10_SEEDS=(2 6 7 8 9)
CIFAR_HASH_1=(2024-04-12_06-38-15 2024-04-23_09-09-40 2024-04-23_11-40-01 2024-04-23_14-10-34 2024-04-23_16-41-13)

for seed in 0 1 2 3 4
do
    python ./main/run_testing.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
                                --data cifar10 --server main \
                                --n-label-per-class 400 \
                                --n-valid-per-class 500 \
                                --mismatch-ratio 0.6 \
                                --backbone-type wide28_2 \
                                --checkpoint-hash ${CIFAR_HASH_1[$seed]} \
                                --for-what Ablation1 --normalize
done


# python ./main/run_testing.py --gpus 0 --seed 1 \
#                             --data tiny --server main \
#                             --n-label-per-class 100 \
#                             --n-valid-per-class 50 \
#                             --mismatch-ratio 0.6 \
#                             --backbone-type wide28_2 \
#                             --checkpoint-hash 2024-04-11_01-20-56 \
#                             --for-what Ablation1 --normalize