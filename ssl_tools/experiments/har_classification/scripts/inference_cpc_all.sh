#/bin/bash

cd ..

for dset in  "KuHar" "MotionSense" "RealWorld_thigh" "RealWorld_waist" "UCI"; 
do 
    ./cpc.py test \
        --data /workspaces/hiaac-m4/ssl_tools/data/standartized_balanced/${dset} \
        --load logs/finetune/CPC/2023-12-06_19-32-19/checkpoints/last.ckpt \
        --batch_size 128 \
        --accelerator gpu \
        --devices 1 \
        --window_size 60 \
        --num_classes 6 \
        --encoding_size 150 
done