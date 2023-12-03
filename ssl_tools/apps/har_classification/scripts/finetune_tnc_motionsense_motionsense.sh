#/bin/bash

cd ..

./train.py \
    --data /workspaces/hiaac-m4/ssl_tools/data/standartized_balanced/MotionSense \
    --epochs 15 \
    --batch_size 100 \
    --accelerator gpu \
    --devices 1 \
    --load_backbone /workspaces/hiaac-m4/ssl_tools/ssl_tools/apps/har_classification/logs/TNC_pretrain/20231203.233329/checkpoints/last.ckpt \
    --training_mode finetune \
    tnc \
    --repeat 5 \
    --mc_sample_size 20 \
    --window_size 60 \
    --encoding_size 150 \
    --w 0.05
