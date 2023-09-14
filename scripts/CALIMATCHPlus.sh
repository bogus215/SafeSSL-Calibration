for coef in 0.05 0.01
do
    for ratio in 0.3 0.6
    do
        python ./main/run_CaliMATCHPlus.py --gpus 0 --seed 2 \
                                    --data cifar10 --server main --enable-wandb \
                                    --n-label-per-class 400 \
                                    --n-valid-per-class 500 \
                                    --mismatch-ratio $ratio --mixed-precision \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --weight-decay 0 --consis-coef2 $coef

        python ./main/run_CaliMATCHPlus.py --gpus 0 --seed 2 \
                                    --data cifar100 --server main --enable-wandb \
                                    --n-label-per-class 100 \
                                    --n-valid-per-class 50 \
                                    --mismatch-ratio $ratio --mixed-precision \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --weight-decay 0 --consis-coef2 $coef
    done
done