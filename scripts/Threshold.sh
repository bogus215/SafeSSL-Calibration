CIFAR10_SEEDS=(2 6 7 8 9)

for seed in 0 1 2 3 4
do
    for ratio in 0.6
    do
        for tau in 0.93 0.97
        do
            for pi in 0.5 0.6 0.7 0.8
            do
                python ./main/run_PROPOSED.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
                                            --data cifar10 --server main --enable-wandb \
                                            --n-label-per-class 400 \
                                            --n-valid-per-class 500 \
                                            --mismatch-ratio $ratio \
                                            --save-every 5000 --learning-rate 3e-3 \
                                            --backbone-type wide28_2 --optimizer adam \
                                            --lambda-ova-cali 0.1 --lambda-ova 0.1 \
                                            --weight-decay 0 --tau $tau --pi $pi \
                                            --normalize --wandb-proj-v=-final

                python ./main/run_OPENMATCH.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
                                            --data cifar10 --server main --enable-wandb \
                                            --n-label-per-class 400 \
                                            --n-valid-per-class 500 \
                                            --mismatch-ratio $ratio --mixed-precision \
                                            --save-every 5000 --learning-rate 3e-3 \
                                            --backbone-type wide28_2 --optimizer adam \
                                            --weight-decay 0 --warm-up 1 --p-cutoff $tau --pi $pi

            done
        done
    done
done