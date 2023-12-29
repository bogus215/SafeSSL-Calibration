CIFAR10_SEEDS=(2 6 7 8 9)
CIFAR100_SEEDS=(1 2 5 6 10)
TINY_SEEDS=(1 2 5 6 10)
SVHN_SEEDS=(1 2 5 6 10)

python ./main/run_ablation1.py --gpus 0 --seed 1 \
                            --data svhn --server main --enable-wandb \
                            --n-label-per-class 50 \
                            --mismatch-ratio 0.6 --mixed-precision \
                            --save-every 5000 --learning-rate 3e-3 \
                            --backbone-type wide28_2 --optimizer adam \
                            --weight-decay 0

python ./main/run_ablation1.py --gpus 0 --seed 2 \
                            --data cifar10 --server main --enable-wandb \
                            --n-label-per-class 400 \
                            --n-valid-per-class 500 \
                            --mismatch-ratio 0.6 --mixed-precision \
                            --save-every 5000 --learning-rate 3e-3 \
                            --backbone-type wide28_2 --optimizer adam \
                            --weight-decay 0 --normalize

python ./main/run_ablation1.py --gpus 0 --seed 1 \
                            --data cifar100 --server main --enable-wandb \
                            --n-label-per-class 100 \
                            --n-valid-per-class 50 \
                            --mismatch-ratio 0.6 --mixed-precision \
                            --save-every 5000 --learning-rate 3e-3 \
                            --backbone-type wide28_2 --optimizer adam \
                            --weight-decay 0 --normalize

for ratio in 0.3 0.6
do
    python ./main/run_ablation1.py --gpus 0 --seed 1 \
                                    --data tiny --server main --enable-wandb \
                                    --n-label-per-class 100 \
                                    --n-valid-per-class 50 \
                                    --mismatch-ratio $ratio --mixed-precision \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --weight-decay 0 --normalize
done

for seed in 1 2 3 4
do
    for ratio in 0.3 0.6
    do
        echo Ablation1+SVHN, ${SVHN_SEEDS[$seed]}, $ratio
        python ./main/run_ablation1.py --gpus 0 --seed ${SVHN_SEEDS[$seed]} \
                                    --data svhn --server main --enable-wandb \
                                    --n-label-per-class 50 \
                                    --mismatch-ratio $ratio --mixed-precision \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --weight-decay 0

        echo Ablation1+CIFAR10, ${CIFAR10_SEEDS[$seed]}, $ratio
        python ./main/run_ablation1.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
                                    --data cifar10 --server main --enable-wandb \
                                    --n-label-per-class 400 \
                                    --n-valid-per-class 500 \
                                    --mismatch-ratio $ratio --mixed-precision \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --weight-decay 0 --normalize

        echo Ablation1+CIFAR100, ${CIFAR100_SEEDS[$seed]}, $ratio
        python ./main/run_ablation1.py --gpus 0 --seed ${CIFAR100_SEEDS[$seed]} \
                                    --data cifar100 --server main --enable-wandb \
                                    --n-label-per-class 100 \
                                    --n-valid-per-class 50 \
                                    --mismatch-ratio $ratio --mixed-precision \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --weight-decay 0 --normalize

        echo Ablation1+tiny, ${TINY_SEEDS[$seed]}, $ratio
        python ./main/run_ablation1.py --gpus 0 --seed ${TINY_SEEDS[$seed]} \
                                        --data tiny --server main --enable-wandb \
                                        --n-label-per-class 100 \
                                        --n-valid-per-class 50 \
                                        --mismatch-ratio $ratio --mixed-precision \
                                        --save-every 5000 --learning-rate 3e-3 \
                                        --backbone-type wide28_2 --optimizer adam \
                                        --weight-decay 0 --normalize
    done
done