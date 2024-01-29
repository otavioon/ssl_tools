#/bin/bash

cd ..

./cpc.py fit \
    --data /workspaces/hiaac-m4/ssl_tools/data/standartized_balanced/MotionSense \
    --epochs 100 \
    --batch_size 128 \
    --accelerator gpu \
    --devices 1 \
    --load_backbone logs/pretrain/CPC/2024-01-29_19-54-51/checkpoints/last.ckpt \
    --training_mode finetune \
    --window_size 60 \
    --num_classes 6 \
    --encoding_size 150 