#!/bin/bash
WINDOW_VIEW_DIR="/workspaces/hiaac-m4/data/standartized_balanced"
CONCATENATED_VIEW_DIR="/workspaces/hiaac-m4/data/view_concatenated"
FINETUNE_DIR="/workspaces/hiaac-m4/data/standartized_balanced"

echo "-----------------------------------------------------------------------"
echo "Running window view"
./train_all_grid.sh --mode window --root_dataset_dir $WINDOW_VIEW_DIR

echo "-----------------------------------------------------------------------"
echo "Running concatenated view"
./train_all_grid.sh --mode concatenated --root_dataset_dir $CONCATENATED_VIEW_DIR

# echo "-----------------------------------------------------------------------"
# echo "Running finetune"
# ./finetune_all_grid.sh --root_dataset_dir $FINETUNE_DIR
