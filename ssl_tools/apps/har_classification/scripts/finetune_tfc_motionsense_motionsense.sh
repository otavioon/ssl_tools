#/bin/bash

cd ..

./tfc.py \
    /workspaces/hiaac-m4/ssl_tools/data/standartized_balanced/MotionSense \
    --epochs 100 \
    --batch_size 128 \
    --accelerator gpu \
    --devices 1 \
    --load_backbone /workspaces/hiaac-m4/ssl_tools/ssl_tools/apps/har_classification/logs/TFC_pretrain/20231204.002134/checkpoints/last.ckpt \
    --training_mode finetune \
    --encoding_size 128 \
    --features_as_channels False \
    --length_alignment 360