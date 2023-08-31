N_LABELS=(400 100)
N_VALID=(500 50)

for seed in 1;
do
    A=0
    for data in cifar10 cifar100;
    do
        echo $seed, $data, ${N_LABELS[$A]}, ${N_VALID[$A]}
        python ./main/run_SL.py \
        --gpus 0 \
        --seed $seed \
        --data $data \
        --server main \
        --enable-wandb \
        --n-label-per-class ${N_LABELS[$A]} \
        --n-valid-per-class ${N_VALID[$A]} \
        --mismatch-ratio 0.60 \
        --mixed-precision \
        --save-every 5000 \
        --learning-rate 0.1 \
        --backbone-type wide28_2 \
        --optimizer sgd \
        --weight-decay 5e-4 \
        --milestones 100000 200000 400000
        let "A+=1"
    done
done