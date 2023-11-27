#/bin/bash

cd ..

./pretrain.py --data /workspaces/hiaac-m4/ssl_tools/data/standartized_balanced/MotionSense --epochs 10 --batch_size 128 --accelerator gpu --devices 1  --checkpoint_metric train_loss tfc --features_as_channels False
