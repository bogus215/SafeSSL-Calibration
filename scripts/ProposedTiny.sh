CIFAR10_SEEDS=(2 6 7 8 9)
CIFAR100_SEEDS=(1 2 5 6 10)
TINY_SEEDS=(1 2 5 6 10)
SVHN_SEEDS=(1 2 5 6 10)

for seed in 0
do
    for ratio in 0.3 0.6
    do
        echo PROPOSED+TINY, ${TINY_SEEDS[$seed]}, $ratio
        python ./main/run_PROPOSED.py --gpus 0 --seed ${TINY_SEEDS[$seed]} \
                                    --data tiny --server main --enable-wandb \
                                    --n-label-per-class 100 \
                                    --n-valid-per-class 50 \
                                    --mismatch-ratio $ratio \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer sgd \
                                    --lambda-ova-cali 0.001 --lambda-ova 1 \
                                    --weight-decay 0 --batch-size 256 --wandb-proj-v=-tiny
    done
done