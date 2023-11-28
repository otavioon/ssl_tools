#/bin/bash

cd ..

./train.py \
    --data /workspaces/hiaac-m4/ssl_tools/data/view_concatenated/MotionSense_cpc \
    --epochs 15 \
    --batch_size 1 \
    --accelerator gpu \
    --devices 1 \
    --load /workspaces/hiaac-m4/ssl_tools/ssl_tools/apps/har_classification/logs/CPC_Pretrain/20231128.122354/checkpoints/last.ckpt \
    --resume  /workspaces/hiaac-m4/ssl_tools/ssl_tools/apps/har_classification/logs/CPC_Pretrain/20231128.122354/checkpoints/last.ckpt \
    cpc \
    --window_size 60
