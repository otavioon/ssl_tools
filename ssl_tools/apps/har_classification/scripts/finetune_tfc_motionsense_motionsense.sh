#/bin/bash

cd ..

./train.py \
    --data /workspaces/hiaac-m4/ssl_tools/data/standartized_balanced/MotionSense \
    --epochs 15 \
    --batch_size 100 \
    --accelerator gpu \
    --devices 1 \
    --load_backbone /workspaces/hiaac-m4/ssl_tools/ssl_tools/apps/har_classification/logs/TFC_Pretrain/20231203.234054/checkpoints/last.ckpt \
    --training_mode finetune \
    tfc \
    --features_as_channels False \
    --length_alignment 360
