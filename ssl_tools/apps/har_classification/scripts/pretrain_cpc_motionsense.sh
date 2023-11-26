#/bin/bash

cd ..

./pretrain.py --data /workspaces/hiaac-m4/ssl_tools/data/view_concatenated/MotionSense_cpc --epochs 1 --batch_size 1 cpc
