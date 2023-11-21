for ratio in 0.3 0.6
do
    for pi in 0.005 0.01 0.05
    do
        python ./main/run_PROPOSED.py --gpus 0 --seed 2 --data cifar10 --server main --enable-wandb \
        --n-label-per-class 400 --n-valid-per-class 500 \
        --mismatch-ratio $ratio \
        --save-every 5000 --learning-rate 3e-3 \
        --backbone-type wide28_2 --optimizer adam \
        --weight-decay 0 --tau 0.95 --pi $pi
    done
done