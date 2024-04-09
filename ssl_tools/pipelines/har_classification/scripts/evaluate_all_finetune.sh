#!/bin/bash

cd ..
python evaluator.py evaluate-all --experiment_names "['har_results', 'har_results_train', 'har_results_finetune']" --experiment_id 405350719407705640  --root_dataset_dir /workspaces/hiaac-m4/data/standartized_balanced/ --batch_size 256 --accelerator gpu --devices 1 --config_dir configs/base/test --log_dir mlruns/ --skip_existing true --use_ray true --ray_address 172.17.0.3:6379