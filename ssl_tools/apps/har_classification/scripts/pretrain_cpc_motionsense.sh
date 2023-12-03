#/bin/bash

cd ..

./train.py \
    --data /workspaces/hiaac-m4/ssl_tools/data/view_concatenated/MotionSense_cpc \
    --epochs 10 \
    --batch_size 1 \
    --accelerator gpu \
    --devices 1 \
    --training_mode pretrain \
    cpc \
    --window_size 60 \
    --encoding_size 150