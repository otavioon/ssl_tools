#/bin/bash

cd ..

for dset in  "KuHar" "MotionSense" "RealWorld_thigh" "RealWorld_waist" "UCI"; 
do 
    ./tfc.py test \
        --data /workspaces/hiaac-m4/ssl_tools/data/standartized_balanced/${dset} \
        --load logs/finetune/TFC/2023-12-06_19-35-44/checkpoints/last.ckpt \
        --batch_size 128 \
        --accelerator gpu \
        --devices 1 \
        --encoding_size 128 \
        --features_as_channels True \
        --length_alignment 60 \
        --in_channels 6
done