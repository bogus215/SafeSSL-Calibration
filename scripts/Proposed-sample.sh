CIFAR10_SEEDS=(2 6 7 8 9)
CIFAR100_SEEDS=(1 2 5 6 10)
TINY_SEEDS=(1 2 5 6 10)
SVHN_SEEDS=(1 2 5 6 10)

for seed in 0 1 2 3 4
do
    for ratio in 0.6
    do
        python ./main/run_PROPOSED.py --gpus 0 --seed ${CIFAR100_SEEDS[$seed]} \
                                    --data cifar100 --server main --enable-wandb \
                                    --n-label-per-class 100 \
                                    --n-valid-per-class 50 \
                                    --mismatch-ratio $ratio \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --lambda-ova-cali 0.05 --lambda-ova 0.1 \
                                    --weight-decay 0 --normalize --wandb-proj-v=-v23

        python ./main/run_PROPOSED.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
                                    --data cifar10 --server main --enable-wandb \
                                    --n-label-per-class 400 \
                                    --n-valid-per-class 500 \
                                    --mismatch-ratio $ratio \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --lambda-ova-cali 0.05 --lambda-ova 0.1 \
                                    --weight-decay 0 --normalize --wandb-proj-v=-v23

        python ./main/run_PROPOSED.py --gpus 0 --seed ${TINY_SEEDS[$seed]} \
                                    --data tiny --server main --enable-wandb \
                                    --n-label-per-class 100 \
                                    --n-valid-per-class 50 \
                                    --mismatch-ratio $ratio \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --lambda-ova-cali 0.05 --lambda-ova 0.1 \
                                    --weight-decay 0 --normalize --wandb-proj-v=-v23

        python ./main/run_PROPOSED.py --gpus 0 --seed ${SVHN_SEEDS[$seed]} \
                                    --data svhn --server main --enable-wandb \
                                    --n-label-per-class 50 \
                                    --mismatch-ratio $ratio \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --lambda-ova-cali 0.05 --lambda-ova 0.5 \
                                    --weight-decay 0 --wandb-proj-v=-v23

        python ./main/run_ablation1.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
                                    --data cifar10 --server main --enable-wandb \
                                    --n-label-per-class 400 \
                                    --n-valid-per-class 500 \
                                    --mismatch-ratio $ratio \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --lambda-ova-cali 0.05 --lambda-ova 0.1 \
                                    --weight-decay 0 --normalize --wandb-proj-v=-v23

        python ./main/run_ablation2.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
                                    --data cifar10 --server main --enable-wandb \
                                    --n-label-per-class 400 \
                                    --n-valid-per-class 500 \
                                    --mismatch-ratio $ratio \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --lambda-ova 0.1 --weight-decay 0 --normalize --wandb-proj-v=-v23

        python ./main/run_ablation1.py --gpus 0 --seed ${CIFAR100_SEEDS[$seed]} \
                                    --data cifar100 --server main --enable-wandb \
                                    --n-label-per-class 100 \
                                    --n-valid-per-class 50 \
                                    --mismatch-ratio $ratio \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --lambda-ova-cali 0.05 --lambda-ova 0.1 \
                                    --weight-decay 0 --wandb-proj-v=-v23

        python ./main/run_ablation2.py --gpus 0 --seed ${CIFAR100_SEEDS[$seed]} \
                                    --data cifar100 --server main --enable-wandb \
                                    --n-label-per-class 100 \
                                    --n-valid-per-class 50 \
                                    --mismatch-ratio $ratio \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --lambda-ova 0.1 \
                                    --weight-decay 0 --wandb-proj-v=-v23

        python ./main/run_ablation1.py --gpus 0 --seed ${TINY_SEEDS[$seed]} \
                                    --data tiny --server main --enable-wandb \
                                    --n-label-per-class 100 \
                                    --n-valid-per-class 50 \
                                    --mismatch-ratio $ratio \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --lambda-ova-cali 0.05 --lambda-ova 0.1 \
                                    --weight-decay 0 --normalize --wandb-proj-v=-v23

        python ./main/run_ablation2.py --gpus 0 --seed ${TINY_SEEDS[$seed]} \
                                    --data tiny --server main --enable-wandb \
                                    --n-label-per-class 100 \
                                    --n-valid-per-class 50 \
                                    --mismatch-ratio $ratio \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --lambda-ova 0.1 \
                                    --weight-decay 0 --normalize --wandb-proj-v=-v23

        python ./main/run_ablation1.py --gpus 0 --seed ${SVHN_SEEDS[$seed]} \
                                    --data svhn --server main --enable-wandb \
                                    --n-label-per-class 50 \
                                    --mismatch-ratio $ratio \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --lambda-ova-cali 0.05 --lambda-ova 0.5 \
                                    --weight-decay 0 --wandb-proj-v=-v23

        python ./main/run_ablation2.py --gpus 0 --seed ${SVHN_SEEDS[$seed]} \
                                    --data svhn --server main --enable-wandb \
                                    --n-label-per-class 50 \
                                    --mismatch-ratio $ratio \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --lambda-ova 0.5 \
                                    --weight-decay 0 --wandb-proj-v=-v23

    done
done