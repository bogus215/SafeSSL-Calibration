python ./main/run_SL.py --gpus 0 --data cifar10 --server main --n-label-per-class 400 --n-valid-per-class 500 --mismatch-ratio 0.60 --mixed-precision --save-every 5000 --learning-rate 0.001 --backbone-type wide28_2

# python ./main/run_FIXMATCH.py --gpus 0 --data cifar100 --enable-wandb --server main --n-label-per-class 100 --n-valid-per-class 50 --mismatch-ratio 0.50 --mixed-precision --save-every 25000

# python ./main/run_CaliMATCH.py --gpus 0 --data cifar100 --enable-wandb --server main --n-label-per-class 100 --n-valid-per-class 50 --mismatch-ratio 0.50 --mixed-precision --save-every 25000 --n-bins 30

# for ratio in 0.00 0.25 0.50;
# do
#     python ./main/run_CaliMATCH.py --gpus 0 --enable-wandb --server main --n-label-per-class 400 --n-valid-per-class 500 --mismatch-ratio $ratio --mixed-precision --save-every 25000 --n-bins 30
# done