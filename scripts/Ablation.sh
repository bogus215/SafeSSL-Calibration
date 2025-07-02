CIFAR100_SEEDS=(1 2 5 6 10)

for seed in 1 2 3 4
do
    for ratio in 0.6
    do
        python ./main/run_ablation2.py --gpus 0 --seed ${CIFAR100_SEEDS[$seed]} \
                                    --data cifar100 --server main --enable-wandb \
                                    --n-label-per-class 100 \
                                    --n-valid-per-class 50 \
                                    --mismatch-ratio $ratio \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --lambda-ova 0.1 --weight-decay 0 --normalize --wandb-proj-v=-final
    done
done