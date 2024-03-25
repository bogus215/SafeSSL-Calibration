CIFAR10_SEEDS=(2 6 7 8 9)
SVHN_SEEDS=(1 2 5 6 10)

for seed in 0
do
    for ratio in 0.6
    do
        ############################## PROPOSED ##############################
        python ./main/run_PROPOSED.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
                                    --data cifar10 --server main --enable-wandb \
                                    --n-label-per-class 40 \
                                    --n-valid-per-class 500 \
                                    --mismatch-ratio $ratio --mixed-precision \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --lambda-ova-cali 0.1 --lambda-ova 0.1 \
                                    --weight-decay 0 --normalize --wandb-proj-v=-v20

        python ./main/run_PROPOSED.py --gpus 0 --seed ${SVHN_SEEDS[$seed]} \
                                    --data svhn --server main --enable-wandb \
                                    --n-label-per-class 10 \
                                    --mismatch-ratio $ratio --mixed-precision \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --lambda-ova-cali 0.5 --lambda-ova 0.5 \
                                    --weight-decay 0 --wandb-proj-v=-v20

        ############################## IOMATCH ##############################
        python ./main/run_IOMATCH.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
                                    --data cifar10 --server main --enable-wandb \
                                    --n-label-per-class 40 \
                                    --n-valid-per-class 500 \
                                    --mismatch-ratio $ratio --mixed-precision \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --weight-decay 0 --warm-up 1

        python ./main/run_IOMATCH.py --gpus 0 --seed ${SVHN_SEEDS[$seed]} \
                                    --data svhn --server main --enable-wandb \
                                    --n-label-per-class 10 \
                                    --mismatch-ratio $ratio --mixed-precision \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --weight-decay 0 --warm-up 1

        ############################## OPENMATCH ##############################
        python ./main/run_OPENMATCH.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
                                    --data cifar10 --server workstation1 --enable-wandb \
                                    --n-label-per-class 40 \
                                    --n-valid-per-class 500 \
                                    --mismatch-ratio $ratio --mixed-precision \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --weight-decay 0 --warm-up 1

        python ./main/run_OPENMATCH.py --gpus 0 --seed ${SVHN_SEEDS[$seed]} \
                                    --data svhn --server workstation1 --enable-wandb \
                                    --n-label-per-class 10 \
                                    --mismatch-ratio $ratio --mixed-precision \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --weight-decay 0 --warm-up 1

        ############################## MTC ##############################
        python ./main/run_MTC.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
                                    --data cifar10 --server main --enable-wandb \
                                    --n-label-per-class 40 \
                                    --n-valid-per-class 500 \
                                    --mismatch-ratio $ratio --mixed-precision \
                                    --save-every 500 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam 

        python ./main/run_MTC.py --gpus 0 --seed ${SVHN_SEEDS[$seed]} \
                                    --data svhn --server main --enable-wandb \
                                    --n-label-per-class 10 \
                                    --mismatch-ratio $ratio --mixed-precision \
                                    --save-every 500 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam 

        ############################## FIXMATCH ##############################
        python ./main/run_FIXMATCH.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
                                    --data cifar10 --server main --enable-wandb \
                                    --n-label-per-class 40 \
                                    --n-valid-per-class 500 \
                                    --mismatch-ratio $ratio --mixed-precision \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --weight-decay 0

        python ./main/run_FIXMATCH.py --gpus 0 --seed ${SVHN_SEEDS[$seed]} \
                                    --data svhn --server main --enable-wandb \
                                    --n-label-per-class 10 \
                                    --mismatch-ratio $ratio --mixed-precision \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --weight-decay 0

        ############################## SAFESTUDENT ##############################
        python ./main/run_SAFE_STUDENT.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
                                          --data cifar10 --server main --enable-wandb \
                                          --n-label-per-class 40 \
                                          --n-valid-per-class 500 \
                                          --mismatch-ratio $ratio --mixed-precision \
                                          --save-every 5000 --learning-rate 3e-3 \
                                          --backbone-type wide28_2 --optimizer adam 

        python ./main/run_SAFE_STUDENT.py --gpus 0 --seed ${SVHN_SEEDS[$seed]} \
                                          --data svhn --server main --enable-wandb \
                                          --n-label-per-class 10 \
                                          --mismatch-ratio $ratio --mixed-precision \
                                          --save-every 5000 --learning-rate 3e-3 \
                                          --backbone-type wide28_2 --optimizer adam 
    done
done