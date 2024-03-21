CIFAR10_SEEDS=(2 6 7 8 9)

for seed in 0 1 2 3 4
do
    for ratio in 0.6
    do
        python ./main/run_ablation3.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
                                    --data cifar10 --server main --enable-wandb \
                                    --n-label-per-class 400 \
                                    --n-valid-per-class 500 \
                                    --mismatch-ratio $ratio \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --lambda-ova 0.1 --weight-decay 0 --normalize --wandb-proj-v=-v20

        python ./main/run_ablation1.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
                                    --data cifar10 --server main --enable-wandb \
                                    --n-label-per-class 400 \
                                    --n-valid-per-class 500 \
                                    --mismatch-ratio $ratio \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --lambda-ova-cali 0.1 --lambda-ova 0.1 \
                                    --weight-decay 0 --normalize --wandb-proj-v=-v20

        python ./main/run_ablation2.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
                                    --data cifar10 --server main --enable-wandb \
                                    --n-label-per-class 400 \
                                    --n-valid-per-class 500 \
                                    --mismatch-ratio $ratio \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --lambda-ova 0.1 --weight-decay 0 --normalize --wandb-proj-v=-v20
    done
done