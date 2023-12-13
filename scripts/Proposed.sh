CIFAR10_SEEDS=(2 6 7 8 9)
CIFAR100_SEEDS=(1 2 5 6 10)
TINY_SEEDS=(1 2 5 6 10)
SVHN_SEEDS=(1 2 5 6 10)

for seed in 0
do
    for ratio in 0.3 0.6
    do
        echo PROPOSED+SVHN, ${SVHN_SEEDS[$seed]}, $ratio
        python ./main/run_PROPOSED.py --gpus 0 --seed ${SVHN_SEEDS[$seed]} \
                                    --data svhn --server main --enable-wandb \
                                    --n-label-per-class 50 \
                                    --mismatch-ratio $ratio --mixed-precision \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --weight-decay 0 --normalize --pi 1e-4

        echo PROPOSED+CIFAR10, ${CIFAR10_SEEDS[$seed]}, $ratio
        python ./main/run_PROPOSED.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
                                    --data cifar10 --server main --enable-wandb \
                                    --n-label-per-class 400 \
                                    --n-valid-per-class 500 \
                                    --mismatch-ratio $ratio --mixed-precision \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --weight-decay 0 --normalize --pi 1e-2

        echo PROPOSED+CIFAR100, ${CIFAR100_SEEDS[$seed]}, $ratio
        python ./main/run_PROPOSED.py --gpus 0 --seed ${CIFAR100_SEEDS[$seed]} \
                                    --data cifar100 --server main --enable-wandb \
                                    --n-label-per-class 100 \
                                    --n-valid-per-class 50 \
                                    --mismatch-ratio $ratio --mixed-precision \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --weight-decay 0 --normalize --pi 5e-2

        echo PROPOSED+TINY, ${TINY_SEEDS[$seed]}, $ratio
        python ./main/run_PROPOSED.py --gpus 0 --seed ${TINY_SEEDS[$seed]} \
                                    --data tiny --server main --enable-wandb \
                                    --n-label-per-class 100 \
                                    --n-valid-per-class 50 \
                                    --mismatch-ratio $ratio --mixed-precision \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --weight-decay 0 --normalize --pi 1e-1
    done
done