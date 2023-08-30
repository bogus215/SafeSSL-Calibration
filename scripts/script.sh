# python ./main/run_SL.py --gpus 0 --data cifar10 --server main --n-label-per-class 400 --n-valid-per-class 500 --mismatch-ratio 0.60 \
# --mixed-precision --save-every 5000 --learning-rate 0.03 \
# --backbone-type wide28_2 --optimizer sgd

# python ./main/run_SL.py --gpus 0 --data cifar100 --server main --n-label-per-class 100 --n-valid-per-class 50 --mismatch-ratio 0.60 \
# --mixed-precision --save-every 5000 --learning-rate 0.03 --backbone-type wide28_2 --optimizer sgd

# for ratio in 0.00 0.30 0.60;
# do
#     python ./main/run_FIXMATCH.py --gpus 0 --enable-wandb --server main --n-label-per-class 400 --n-valid-per-class 500 --mismatch-ratio $ratio --mixed-precision --save-every 5000 --learning-rate 0.03 --backbone-type wide28_2 --optimizer sgd
#     python ./main/run_FIXMATCH.py --gpus 0 --data cifar100 --enable-wandb --server main --n-label-per-class 100 --n-valid-per-class 50 --mismatch-ratio $ratio --mixed-precision --save-every 5000 --learning-rate 0.03 --backbone-type wide28_2 --optimizer sgd
# done

for ratio in 0.00 0.30 0.60;
do
    python ./main/run_CaliMATCH.py --gpus 0 --enable-wandb --server main --n-label-per-class 400 --n-valid-per-class 500 --mismatch-ratio $ratio --mixed-precision --save-every 5000 --learning-rate 0.03 --backbone-type wide28_2 --optimizer sgd --n-bins 15 --train-n-bins 30
    python ./main/run_CaliMATCH.py --gpus 0 --data cifar100 --enable-wandb --server main --n-label-per-class 100 --n-valid-per-class 50 --mismatch-ratio $ratio --mixed-precision --save-every 5000 --learning-rate 0.03 --backbone-type wide28_2 --optimizer sgd --n-bins 15 --train-n-bins 30
done