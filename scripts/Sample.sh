CIFAR10_SEEDS=(2 6 7 8 9)
SVHN_SEEDS=(1 2 5 6 10)

for seed in 1 2 3 4
do
    for ratio in 0.6
    do
        python ./main/run_SAFE_STUDENT.py --gpus 0 --seed ${SVHN_SEEDS[$seed]} \
                                          --data svhn --server main --enable-wandb \
                                          --n-label-per-class 50 \
                                          --mismatch-ratio $ratio --mixed-precision \
                                          --save-every 5000 --learning-rate 3e-3 \
                                          --backbone-type wide28_2 --optimizer adam 
    done
done

for seed in 1 2 3 4
do
    for ratio in 0.6
    do
        python ./main/run_IOMATCH.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
                                    --data cifar10 --server main --enable-wandb \
                                    --n-label-per-class 400 \
                                    --n-valid-per-class 500 \
                                    --mismatch-ratio $ratio --mixed-precision \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --weight-decay 0 --warm-up 1

        python ./main/run_IOMATCH.py --gpus 0 --seed ${SVHN_SEEDS[$seed]} \
                                    --data svhn --server main --enable-wandb \
                                    --n-label-per-class 50 \
                                    --mismatch-ratio $ratio --mixed-precision \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --weight-decay 0 --warm-up 1
    done
done