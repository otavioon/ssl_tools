#/bin/bash

cd ..

for dset in  "KuHar" "MotionSense" "RealWorld_thigh" "RealWorld_waist" "UCI"; 
do 
    ./tnc.py test \
        --data /workspaces/hiaac-m4/ssl_tools/data/standartized_balanced/${dset} \
        --load logs/finetune/TNC/2023-12-06_19-34-49/checkpoints/last.ckpt \
        --batch_size 128 \
        --accelerator gpu \
        --devices 1 \
        --mc_sample_size 20 \
        --window_size 60 \
        --encoding_size 150 \
        --w 0.05
done