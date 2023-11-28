#/bin/bash

cd ..

./train.py --training_mode pretrain --data /workspaces/hiaac-m4/ssl_tools/data/view_concatenated/MotionSense_cpc --epochs 10 --batch_size 10 --accelerator gpu --devices 1  tnc --repeat 5 --mc_sample_size 20 --window_size 60
