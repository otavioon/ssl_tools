#/bin/bash

cd ..

./pretrain.py --data /workspaces/hiaac-m4/ssl_tools/data/view_concatenated/MotionSense_cpc --epochs 100 --batch_size 10 tnc_light --repeat 5 --mc_sample_size 20 --window_size 60
