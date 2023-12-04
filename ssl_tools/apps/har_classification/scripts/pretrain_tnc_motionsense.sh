#/bin/bash

cd ..

./train.py \
    --data /workspaces/hiaac-m4/ssl_tools/data/view_concatenated/MotionSense_cpc \
    --epochs 100 \
    --batch_size 16 \
    --accelerator gpu \
    --devices 1 \
    --training_mode pretrain \
    --checkpoint_metric train_loss \
    tnc \
    --repeat 5 \
    --mc_sample_size 20 \
    --window_size 60 \
    --encoding_size 150 \
    --w 0.05
