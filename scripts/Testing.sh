# python ./main/run_testing.py --gpus 0 --seed 2 \
#                             --data cifar10 --server main \
#                             --n-label-per-class 400 \
#                             --n-valid-per-class 500 \
#                             --mismatch-ratio 0.6 \
#                             --backbone-type densenet121 \
#                             --checkpoint-hash 2025-07-10_03-57-40 \
#                             --for-what SL

# python ./main/run_testing.py --gpus 0 --seed 2 \
#                             --data cifar10 --server main \
#                             --n-label-per-class 400 \
#                             --n-valid-per-class 500 \
#                             --mismatch-ratio 0.6 \
#                             --backbone-type densenet121 \
#                             --checkpoint-hash 2025-07-10_11-03-09 \
#                             --for-what Proposed

# python ./main/run_testing.py --gpus 0 --seed 2 \
#                             --data cifar10 --server main \
#                             --n-label-per-class 400 \
#                             --n-valid-per-class 500 \
#                             --mismatch-ratio 0.6 \
#                             --backbone-type densenet121 \
#                             --checkpoint-hash 2025-07-10_21-06-27 \
#                             --for-what OPENMATCH

python ./main/run_testing.py --gpus 0 --seed 2 \
                            --data cifar10 --server main \
                            --n-label-per-class 400 \
                            --n-valid-per-class 500 \
                            --mismatch-ratio 0.6 \
                            --backbone-type wide28_2 \
                            --checkpoint-hash 2025-05-09_13-33-33 \
                            --for-what SCOMATCH

python ./main/run_testing.py --gpus 0 --seed 1 \
                            --data cifar100 --server main \
                            --n-label-per-class 100 \
                            --n-valid-per-class 50 \
                            --mismatch-ratio 0.6 \
                            --backbone-type wide28_2 \
                            --checkpoint-hash 2025-05-09_19-54-30 \
                            --for-what SCOMATCH

python ./main/run_testing.py --gpus 0 --seed 1 \
                            --data svhn --server main \
                            --n-label-per-class 50 \
                            --mismatch-ratio 0.6 \
                            --backbone-type wide28_2 \
                            --checkpoint-hash 2025-05-10_06-51-14 \
                            --for-what SCOMATCH

python ./main/run_testing.py --gpus 0 --seed 1 \
                            --data tiny --server main \
                            --n-label-per-class 100 \
                            --n-valid-per-class 50 \
                            --mismatch-ratio 0.6 \
                            --backbone-type wide28_2 \
                            --checkpoint-hash 2025-05-10_02-33-37 \
                            --for-what SCOMATCH