CIFAR10_SEEDS=(2 6 7 8 9)
for seed in 0
do
    for ratio in 0.6
    do
        # python ./main/run_FIXMATCH.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
        #                             --data cifar10 --server main \
        #                             --n-label-per-class 400 \
        #                             --n-valid-per-class 500 \
        #                             --mismatch-ratio $ratio \
        #                             --save-every 5000 --learning-rate 3e-3 \
        #                             --backbone-type wide28_2 --optimizer adam \
        #                             --weight-decay 0

        python ./main/run_OPENMATCH.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
                                    --data cifar10 --server main \
                                    --n-label-per-class 400 \
                                    --n-valid-per-class 500 \
                                    --mismatch-ratio $ratio \
                                    --save-every 5000 --learning-rate 3e-3 \
                                    --backbone-type wide28_2 --optimizer adam \
                                    --weight-decay 0 --warm-up 1

        # python ./main/run_IOMATCH.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
        #                             --data cifar10 --server main \
        #                             --n-label-per-class 400 \
        #                             --n-valid-per-class 500 \
        #                             --mismatch-ratio $ratio \
        #                             --save-every 5000 --learning-rate 3e-3 \
        #                             --backbone-type wide28_2 --optimizer adam \
        #                             --weight-decay 0 --warm-up 1

        # python ./main/run_PROPOSED.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
        #                             --data cifar10 --server main \
        #                             --n-label-per-class 400 \
        #                             --n-valid-per-class 500 \
        #                             --mismatch-ratio $ratio \
        #                             --save-every 5000 --learning-rate 3e-3 \
        #                             --backbone-type wide28_2 --optimizer adam \
        #                             --lambda-ova-cali 0.1 --lambda-ova 0.1 \
        #                             --weight-decay 0 --normalize --wandb-proj-v=-final

        # python ./main/run_SAFE_STUDENT.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
        #                                   --data cifar10 --server main \
        #                                   --n-label-per-class 400 \
        #                                   --n-valid-per-class 500 \
        #                                   --mismatch-ratio $ratio \
        #                                   --save-every 5000 --learning-rate 3e-3 \
        #                                   --backbone-type wide28_2 --optimizer adam 

        # python ./main/run_MTC.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
        #                             --data cifar10 --server main \
        #                             --n-label-per-class 400 \
        #                             --n-valid-per-class 500 \
        #                             --mismatch-ratio $ratio \
        #                             --save-every 500 --learning-rate 3e-3 \
        #                             --backbone-type wide28_2 --optimizer adam 

        # python ./main/run_SL.py --gpus 0 --seed ${CIFAR10_SEEDS[$seed]} \
        #                         --data cifar10 --server main \
        #                         --n-label-per-class 400 \
        #                         --n-valid-per-class 500 \
        #                         --mismatch-ratio 0.60 \
        #                         --save-every 5000 --learning-rate 3e-3 \
        #                         --backbone-type wide28_2 --optimizer adam \
        #                         --weight-decay 0
    done
done