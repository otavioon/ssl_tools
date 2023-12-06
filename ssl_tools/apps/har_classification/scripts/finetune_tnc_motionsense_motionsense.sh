#/bin/bash

cd ..

./tnc.py fit \
    --data /workspaces/hiaac-m4/ssl_tools/data/standartized_balanced/MotionSense \
    --epochs 100 \
    --batch_size 128 \
    --accelerator gpu \
    --devices 1 \
    --load_backbone logs/pretrain/TNC/2023-12-06_19-26-23/checkpoints/last.ckpt \
    --training_mode finetune \
    --repeat 5 \
    --mc_sample_size 20 \
    --window_size 60 \
    --encoding_size 150 \
    --w 0.05