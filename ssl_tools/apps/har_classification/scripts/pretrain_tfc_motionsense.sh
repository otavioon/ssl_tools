#/bin/bash

cd ..

./tfc.py fit \
    --data /workspaces/hiaac-m4/ssl_tools/data/standartized_balanced/MotionSense \
    --epochs 100 \
    --batch_size 128 \
    --accelerator gpu \
    --devices 1  \
    --checkpoint_metric train_loss \
    --encoding_size 128 \
    --features_as_channels True \
    --length_alignment 60 \
    --in_channels 6 