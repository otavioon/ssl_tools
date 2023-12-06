#/bin/bash

cd ..

./tfc.py \
    /workspaces/hiaac-m4/ssl_tools/data/standartized_balanced/MotionSense \
    --epochs 100 \
    --batch_size 128 \
    --accelerator gpu \
    --devices 1 \
    --load_backbone /workspaces/hiaac-m4/ssl_tools/ssl_tools/apps/har_classification/logs/pretrain/TFC/2023-12-06_12-52-18/checkpoints/last.ckpt \
    --training_mode finetune \
    --encoding_size 128 \
    --features_as_channels True \
    --length_alignment 60 \
    --in_channels 6 \
    --update_backbone False